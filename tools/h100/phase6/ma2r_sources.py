"""MA2 execution revision: native instance binding and independently owned output.

V is the exact MA2 source transform. R preserves its decoder, single packet
workspace and both barriers. Configuration/output ownership follows the prior
op4 engineering idea, without its double buffering or reuse-barrier removal.
"""
from pathlib import Path

import ma2_sources
from common import replace_once, require

RUNTIME = ma2_sources.RUNTIME
DECODER = ma2_sources.DECODER
HEADER = 'src/gemm_rs/taco_runtime.h'
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
FRONTEND = 'python/flux/gemm_rs_taco.py'


def header(source):
    # Keep FluxTacoConfig itself identical: decoder arguments and packet layout
    # do not acquire any lifecycle fields.
    return source + '''
// MA2-R host-only instance binding; FluxTacoConfig's wire/kernel ABI is unchanged.
extern "C" int taco_ma2r_protocol_version();
extern "C" int taco_ma2r_submission_enabled();
extern "C" int taco_ma2r_bind(int k_local);
extern "C" int taco_ma2r_bound_k();
extern "C" void taco_ma2r_restore(FluxTacoConfig config, int k_local);
'''


def runtime(source):
    source = replace_once(source, 'static thread_local FluxTacoConfig active;',
                          'static thread_local FluxTacoConfig active;\n'
                          'static thread_local int ma2r_bound_k = 0;')
    source = replace_once(source, 'extern "C" void taco_reset() { active = FluxTacoConfig{}; }',
                          'extern "C" void taco_reset() { active = FluxTacoConfig{}; ma2r_bound_k = 0; }')
    return source + '''
extern "C" int taco_ma2r_protocol_version() { return 1; }
extern "C" int taco_ma2r_submission_enabled() { return 1; }
extern "C" int taco_ma2r_bind(int k_local) {
  if (!active.enabled || active.world != 8 || active.m != 2048 || active.n != 4096 ||
      (k_local != 512 && k_local != 2048) || taco_placement() != 1)
    return int(cudaErrorInvalidValue);
  ma2r_bound_k = k_local;
  return int(cudaSuccess);
}
extern "C" int taco_ma2r_bound_k() { return ma2r_bound_k; }
extern "C" void taco_ma2r_restore(FluxTacoConfig config, int k_local) {
  active = config;
  ma2r_bound_k = k_local;
}
'''


def wrapper(source):
    source = replace_once(source, '#include <ATen/core/jit_type.h>',
                          '#include <ATen/core/jit_type.h>\n#include <ATen/core/grad_mode.h>')
    source = replace_once(source, '  const bool h100_taco = taco_config().enabled;', '''  const FluxTacoConfig ma2r_config = taco_config();
  const int ma2r_k_local = taco_ma2r_bound_k();
  const c10::cuda::CUDAStream ma2r_stream = c10::cuda::getCurrentCUDAStream();
  const bool h100_taco = ma2r_config.enabled;''')
    anchor = '''      const ReduceScatterOptionWithOptional &opt) {
    return forward_impl('''
    return replace_once(source, anchor, '''      const ReduceScatterOptionWithOptional &opt) {
#ifdef FLUX_TACO_BASELINE
    // Every exit, including validation/launch exceptions, restores the caller's
    // TLS state. Kernel arguments already contain the selected config by value.
    struct RestoreMA2R {
      FluxTacoConfig previous;
      int previous_k;
      ~RestoreMA2R() { taco_ma2r_restore(previous, previous_k); }
    } restore{taco_config(), taco_ma2r_bound_k()};
    if (ma2r_config.enabled) {
      const auto current = c10::cuda::getCurrentCUDAStream();
      cudaStreamCaptureStatus status;
      CUDA_CHECK(cudaStreamIsCapturing(current, &status));
      TORCH_CHECK(current == ma2r_stream && status == cudaStreamCaptureStatusNone,
                  "MA2-R requires the creation device/stream and rejects graph capture");
      TORCH_CHECK(ma2r_config.world == 8 && ma2r_config.m == 2048 && ma2r_config.n == 4096 &&
                      (ma2r_k_local == 512 || ma2r_k_local == 2048),
                  "MA2-R instance is not bound to its exact attention/MLP shape");
      TORCH_CHECK(input.is_cuda() && weight.is_cuda() && input.device() == weight.device() &&
                      input.device().index() == ma2r_stream.device_index() &&
                      input.scalar_type() == at::ScalarType::BFloat16 &&
                      weight.scalar_type() == at::ScalarType::BFloat16 &&
                      input.is_contiguous() && weight.is_contiguous() &&
                      input.dim() == 2 && weight.dim() == 2 &&
                      input.size(0) == ma2r_config.m && weight.size(0) == ma2r_config.n &&
                      input.size(1) == ma2r_k_local && weight.size(1) == ma2r_k_local,
                  "MA2-R tensor device, BF16 dtype, layout or exact shape mismatch");
      TORCH_CHECK(!at::GradMode::is_enabled() ||
                      (!input.requires_grad() && !weight.requires_grad()),
                  "Forward-only MA2-R requires no_grad or an explicit autograd adapter");
      taco_ma2r_restore(ma2r_config, ma2r_k_local);
    }
#endif
    return forward_impl(''')


def frontend(source):
    source = replace_once(source, 'class GemmRSTaco:', '''class GemmRSTaco:
    # local_reduction allocates a fresh torch::empty output on every call.
    # No reusable output view escapes, so the adapter need not clone it again.
    owns_output = True
    owns_configuration = True
''')
    source = replace_once(source, '        self._lock = threading.Lock()', '''        self._lock = threading.Lock()
        self._failed = False
        if (self.world, m, n) != (8, 2048, 4096) or k_local not in (512, 2048):
            raise ValueError('MA2-R supports exact TP8 attention/MLP shapes only')''')
    source = replace_once(source, '            self.lib.taco_reset.argtypes = []', '''            self.lib.taco_ma2r_protocol_version.argtypes = []
            self.lib.taco_ma2r_protocol_version.restype = ctypes.c_int
            self.lib.taco_ma2r_submission_enabled.argtypes = []
            self.lib.taco_ma2r_submission_enabled.restype = ctypes.c_int
            self.lib.taco_ma2r_bind.argtypes = [ctypes.c_int]
            self.lib.taco_ma2r_bind.restype = ctypes.c_int
            if (self.lib.taco_ma2r_protocol_version() != 1 or
                    self.lib.taco_ma2r_submission_enabled() != 1):
                raise RuntimeError('Wrong MA2-R runtime protocol or submission mode')
            self.lib.taco_reset.argtypes = []''')
    source = replace_once(source, '''        try:
            self.op = cpp_mod.GemmRS''', '''        try:
            code = self.lib.taco_ma2r_bind(k_local)
            if code:
                raise RuntimeError(f'MA2-R instance binding rejected: CUDA error {code}')
            self.op = cpp_mod.GemmRS''')
    start = source.index('    def forward(self, input, weight):')
    end = source.index('    @property\n    def wire_bytes_per_rank', start)
    return source[:start] + '''    def forward(self, input, weight):
        with self._lock:
            if self._failed:
                raise RuntimeError('MA2-R cannot reuse an instance after a failed forward')
            try:
                # Native code validates exact tensors, device, stream and capture,
                # then scopes this instance's immutable configuration using RAII.
                # The original two barriers and fresh decoder allocation remain.
                return self.op.forward(input, weight, reduce_scatter_option=self.option)
            except BaseException:
                # A failure can follow an enqueued producer: the group must stop,
                # not retry a packet workspace whose rendezvous may be incomplete.
                self._failed = True
                raise

''' + source[end:]


def sources(reference, arm):
    require(arm in ('V', 'R'), 'MA2-R transforms only V/R')
    reference = Path(reference)
    result = ma2_sources.sources(reference, 'V')
    if arm == 'R':
        result[RUNTIME] = runtime(result[RUNTIME])
        result.update({HEADER: header((reference/HEADER).read_text()),
                       WRAPPER: wrapper((reference/WRAPPER).read_text()),
                       FRONTEND: frontend((reference/FRONTEND).read_text())})
    return result
