"""A-derived fixed-cost experiment: decoder D, lifetime L, their combination F.

No arrival scheduling, selective mask, epilogue change, or codec substitution.
L/F use two complete instances and retain the post-GEMM publication barrier.
"""
from pathlib import Path

import arrival_op3_v2_sources as decoder
from common import replace_once, require

HEADER, RUNTIME, EPILOGUE = decoder.HEADER, decoder.RUNTIME, decoder.EPILOGUE
WRAPPER, FRONTEND, DECODER = decoder.WRAPPER, decoder.FRONTEND, decoder.DECODER


def header(source):
    text = replace_once(source, '  int n = 0;\n};', '''  int n = 0;
  // Exact K is captured with this instance; it never changes across forwards.
  int op4_k_local = 0;
};''')
    return text + '''
extern "C" int taco_op4_protocol_version();
extern "C" int taco_op4_variant();
extern "C" int taco_op4_bind(int k_local);
extern "C" void taco_op4_set_config(FluxTacoConfig config);
'''


def runtime(source, variant):
    require(variant in ('L', 'F'), 'Lifecycle runtime requires L or F')
    return source + f'''
// This ABI is deliberately separate from old instance-owned/selective builds.
extern "C" int taco_op4_protocol_version() {{ return 4; }}
extern "C" int taco_op4_variant() {{ return {2 if variant == 'L' else 3}; }}
extern "C" int taco_op4_bind(int k_local) {{
  if (!active.enabled || active.world != 8 || active.m != 2048 || active.n != 4096 ||
      (k_local != 512 && k_local != 2048) || taco_placement() != 1)
    return int(cudaErrorInvalidValue);
  active.op4_k_local = k_local;
  return int(cudaSuccess);
}}
extern "C" void taco_op4_set_config(FluxTacoConfig config) {{ active = config; }}
'''


def wrapper(source):
    # This is the H100 A wrapper, whose construction-time algorithm identity is
    # already preserved separately from the temporary thread-local config.
    text = replace_once(source, '  const bool h100_taco = taco_config().enabled;', '''  const FluxTacoConfig owned_taco = taco_config();
  const c10::cuda::CUDAStream taco_stream = c10::cuda::getCurrentCUDAStream();
  const bool h100_taco = owned_taco.enabled;''')
    text = replace_once(text, '''      // All receivers must finish decoding before any next invocation overwrites
      // peer packets. This synchronization is part of complete-operator timing.
      group_barrier.barrier_all(stream);''', '''      // The public op4 wrapper alternates TWO complete packet/GemmRS slots.
      // Before k+2 can overwrite slot k, this stream passed k+1's all-rank
      // post-GEMM barrier. Each rank's decode(k) precedes that barrier. Thus
      // it already certifies that every receiver stopped reading slot k.
      // Non-owned legacy callers retain the explicit reuse rendezvous.
      if (!owned_taco.enabled) group_barrier.barrier_all(stream);''')
    anchor = '''      const ReduceScatterOptionWithOptional &opt) {
    return forward_impl('''
    return replace_once(text, anchor, '''      const ReduceScatterOptionWithOptional &opt) {
#ifdef FLUX_TACO_BASELINE
    // Scope the immutable instance config in C++; restore even on exceptions.
    struct RestoreOp4 {
      FluxTacoConfig previous;
      ~RestoreOp4() { taco_op4_set_config(previous); }
    } restore{taco_config()};
    if (owned_taco.enabled) {
      auto current = c10::cuda::getCurrentCUDAStream();
      cudaStreamCaptureStatus status;
      CUDA_CHECK(cudaStreamIsCapturing(current, &status));
      TORCH_CHECK(current == taco_stream && status == cudaStreamCaptureStatusNone,
                  "Op4 requires the creation device/stream and rejects graph capture");
      TORCH_CHECK(owned_taco.world == 8 && owned_taco.m == 2048 && owned_taco.n == 4096 &&
                      (owned_taco.op4_k_local == 512 || owned_taco.op4_k_local == 2048),
                  "Op4 instance configuration is not bound to its exact shape");
      TORCH_CHECK(input.is_cuda() && weight.is_cuda() &&
                      input.device() == weight.device() &&
                      input.device().index() == taco_stream.device_index() &&
                      input.scalar_type() == at::ScalarType::BFloat16 &&
                      weight.scalar_type() == at::ScalarType::BFloat16 &&
                      input.is_contiguous() && weight.is_contiguous() &&
                      input.dim() == 2 && weight.dim() == 2 &&
                      input.size(0) == owned_taco.m && weight.size(0) == owned_taco.n &&
                      input.size(1) == owned_taco.op4_k_local &&
                      weight.size(1) == owned_taco.op4_k_local,
                  "Op4 tensor device, BF16 dtype, layout or exact shape mismatch");
      taco_op4_set_config(owned_taco);
    }
#endif
    return forward_impl(''')


def frontend(source, variant):
    require(variant in ('L', 'F'), 'Lifecycle frontend requires L or F')
    text = replace_once(source, 'class GemmRSTaco:', 'class _GemmRSTacoSlot:')
    text = replace_once(text, '        self._lock = threading.Lock()', '''        self._lock = threading.Lock()
        if (self.world, m, n) != (8, 2048, 4096) or k_local not in (512, 2048):
            raise ValueError('Op4 supports exact TP8 attention/MLP shapes only')
        if os.environ.get('H100_OP4_PROTOCOL') != 'h100-arrival-op4':
            raise RuntimeError('Op4 requires its independent experiment protocol')''')
    text = replace_once(text, '            self.lib.taco_reset.argtypes = []', f'''            self.lib.taco_op4_protocol_version.argtypes = []
            self.lib.taco_op4_protocol_version.restype = ctypes.c_int
            self.lib.taco_op4_variant.argtypes = []
            self.lib.taco_op4_variant.restype = ctypes.c_int
            self.lib.taco_op4_bind.argtypes = [ctypes.c_int]
            self.lib.taco_op4_bind.restype = ctypes.c_int
            if (self.lib.taco_op4_protocol_version() != 4 or
                    self.lib.taco_op4_variant() != {2 if variant == 'L' else 3}):
                raise RuntimeError('Wrong op4 library protocol or variant')
            self.lib.taco_reset.argtypes = []''')
    text = replace_once(text, '''        try:
            self.op = cpp_mod.GemmRS''', '''        try:
            code = self.lib.taco_op4_bind(k_local)
            if code:
                raise RuntimeError(f'Op4 instance binding rejected: CUDA error {code}')
            self.op = cpp_mod.GemmRS''')
    start = text.index('    def forward(self, input, weight):')
    end = text.index('    @property\n    def wire_bytes_per_rank', start)
    text = text[:start] + '''    def forward(self, input, weight):
        if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
            raise ValueError('Forward-only codec; use no_grad or an explicit autograd adapter')
        # C++ validates exact tensors/stream and restores the captured config.
        # A already allocates a fresh decoder output; no extra clone is needed.
        return self.op.forward(input, weight, reduce_scatter_option=self.option)

''' + text[end:]
    return text + '''

class GemmRSTaco:
    """Alternate two complete IPC workspaces on one fixed CUDA stream.

    Call k+2 reuses slot k only after k+1's publication barrier. Every rank
    completed decode(k) before entering that barrier, so no receiver still
    reads the reused slot. The two barriers have independent storage; their
    stream ordering, not shared storage, supplies the cross-slot guarantee.
    All ranks must construct/call the same instances in the same order.
    """
    owns_output = True
    owns_configuration = True

    def __init__(self, group, m, n, k_local, placement=None):
        self.slots = [_GemmRSTacoSlot(group, m, n, k_local, placement) for _ in range(2)]
        self.next_slot = 0
        self._failed = False
        self._lock = threading.Lock()
        for name in ('group', 'rank', 'world', 'm', 'n', 'k_local', 'device', 'stream',
                     'placement', 'wire_bytes_per_rank', 'bf16_wire_bytes_per_rank'):
            setattr(self, name, getattr(self.slots[0], name))
        if self.slots[0].peers[self.rank].data_ptr() == self.slots[1].peers[self.rank].data_ptr():
            raise RuntimeError('Op4 needs two independent packet workspaces')

    def forward(self, input, weight):
        with self._lock:
            if self._failed:
                raise RuntimeError('Op4 cannot reuse an instance after a failed forward')
            try:
                output = self.slots[self.next_slot].forward(input, weight)
            except BaseException:
                # A C++ failure may follow an enqueued producer. Do not retry
                # that slot without the missing intervening collective.
                self._failed = True
                raise
            self.next_slot ^= 1
            return output
'''


def sources(root, variant):
    root = Path(root)
    require(variant in ('D', 'L', 'F'), 'Expected transformed variant D, L or F')
    result = decoder.sources(root, decoder_only=True) if variant in ('D', 'F') else {}
    if variant in ('L', 'F'):
        result.update({HEADER: header((root/HEADER).read_text()),
                       WRAPPER: wrapper((root/WRAPPER).read_text()),
                       FRONTEND: frontend((root/FRONTEND).read_text(), variant)})
        result[RUNTIME] = runtime(result.get(RUNTIME, (root/RUNTIME).read_text()), variant)
    return result
