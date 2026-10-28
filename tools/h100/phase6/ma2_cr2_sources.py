"""CR2: cache fixed-shape host submission while preserving CR's CUDA work.

Input is an exact CR source tree from ma2_ablation_build. No frozen comparison
source is edited. CUTLASS is still initialized on every call with fresh tensor
pointers; its workspace initialization and the publication barrier are retained.
"""
from pathlib import Path

from common import replace_once, require

RUNTIME = 'src/gemm_rs/taco_runtime.cu'
HEADER = 'src/gemm_rs/taco_runtime.h'
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
WRAPPER_HEADER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.h'
PYBIND = 'src/pybind/gemm_rs.cc'
FRONTEND = 'python/flux/gemm_rs_taco.py'


def runtime(source):
    require('taco_ma2c_protocol_version() { return 1; }' in source,
            'CR2 requires the original mixed-source CR decoder')
    source = replace_once(source, 'static thread_local FluxTacoConfig active;',
                          'static thread_local FluxTacoConfig active;\n'
                          'static thread_local int cr2_bound_k = 0;')
    source = replace_once(source, 'extern "C" void taco_reset() { active = FluxTacoConfig{}; }',
                          'extern "C" void taco_reset() { active = FluxTacoConfig{}; cr2_bound_k = 0; }')
    return source + '''
// Host-only binding; the CUDA/wire config struct and decoder are unchanged.
extern "C" int taco_cr2_protocol_version() { return 1; }
extern "C" int taco_cr2_bound_k() { return cr2_bound_k; }
extern "C" int taco_cr2_bind(int k) {
  if (!active.enabled || !active.selected || active.world != 8 ||
      active.m != 2048 || active.n != 4096 || !active.deferred_reuse_barrier ||
      active.allow_capture || (k != 512 && k != 2048) || taco_placement() != 1)
    return int(cudaErrorInvalidValue);
  cr2_bound_k = k;
  return int(cudaSuccess);
}
'''


def header(source):
    return source + '''
// CR2 exact-shape native submission, independent of the CUDA packet ABI.
extern "C" int taco_cr2_protocol_version();
extern "C" int taco_cr2_bind(int k);
extern "C" int taco_cr2_bound_k();
'''


CR2_METHOD = '''
  torch::Tensor
  cr2_forward(torch::Tensor input, torch::Tensor weight,
              const ReduceScatterOptionWithOptional &opt) {
    TORCH_CHECK(!cr2_failed, "CR2 cannot retry a failed submission");
    try {
      TORCH_CHECK(owned_taco.world == 8 && owned_taco.rank == rank &&
                      owned_taco.world == world_size && owned_taco.m == 2048 && owned_taco.n == 4096 &&
                      owned_taco.selected && owned_taco.deferred_reuse_barrier &&
                      !owned_taco.allow_capture && (cr2_k == 512 || cr2_k == 2048) &&
                      nnodes == 1 && !no_nvlink && !fuse_reduction && ring_reduction &&
                      !transpose_weight && phase6_arch() == _Sm80{} &&
                      input_dtype == at::ScalarType::BFloat16 &&
                      output_dtype == at::ScalarType::BFloat16,
                  "CR2 requires its frozen TP8 selective double-workspace configuration");
      TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16 &&
                      weight.scalar_type() == at::ScalarType::BFloat16 &&
                      input.size(1) == cr2_k && weight.size(1) == cr2_k,
                  "CR2 immutable BF16/K cache key changed");
      TORCH_CHECK(!at::GradMode::is_enabled() ||
                      (!input.requires_grad() && !weight.requires_grad()),
                  "Forward-only CR2 requires no_grad or an explicit autograd adapter");
      TORCH_CHECK(opt.use_1d_ring == true && opt.use_p2p_read == true &&
                      opt.use_cudaMemcpyAsync == false && opt.use_gemmk == false &&
                      opt.per_tile_flags == false && opt.use_barrier_queue == false &&
                      opt.num_blocks == 6 && opt.n_split == 1 && opt.ring_mode == RingMode::Ring1D,
                  "CR2 requires the exact immutable ReduceScatterOption");
      auto stream = c10::cuda::getCurrentCUDAStream();
      if (!cr2_op) {
        auto meta = get_gemm_meta(false, false);
        auto rt = get_rt_conf(input, weight, c10::nullopt, c10::nullopt, c10::nullopt);
        auto hparams = OpRegistry::instance().get_hparams(meta, rt);
        auto op = OpRegistry::instance().get_op(meta, hparams);
        const auto materialized = materialize(opt);
        ReduceScatterArguments rs{
            .reduce_scatter_num_blocks = materialized.num_blocks,
            .rs_stream = rs_stream_, .event = event_,
            .use_barrier_queue = materialized.use_barrier_queue,
            .use_gemmk = materialized.use_gemmk,
            .per_tile_flags = materialized.per_tile_flags,
            .use_cudaMemcpyAsync = materialized.use_cudaMemcpyAsync,
            .n_split = materialized.n_split, .sub_world_size = sub_world_size,
            .opaque = nullptr, .use_1d_ring = materialized.use_1d_ring,
            .use_p2p_read = materialized.use_p2p_read};
        GemmReduceScatterArguments args{
            .m = rt.m(), .n = rt.n(), .k = rt.k(), .rank = rank,
            .world_size = world_size, .nnodes = nnodes, .alpha = 1.0f, .beta = 0.0f,
            .input = input.data_ptr(), .weight = weight.data_ptr(), .bias = nullptr,
            .output_scatter_ptrs = output_scatter_ptrs.data(),
            .reduce_buffer_ptrs = reduce_buffer_ptrs.data(),
            .barrier_ptrs = barrier_ptrs.data(), .avail_sms = -1,
            .Aux = nullptr, .Vector = nullptr, .abs_max_Aux = nullptr, .abs_max_D = nullptr,
            .scaleA = nullptr, .scaleB = nullptr, .scaleC = nullptr, .scaleD = nullptr,
            .scaleAux = nullptr, .reduce_scatter_args = rs};
        lazy_init_gemm_buffer(input, op->get_workspace_size(args));
        void *workspace = gemm_buffer.defined() ? gemm_buffer.data_ptr() : nullptr;
        initialize_args_workspace(hparams, args, workspace, stream);
        lazy_init_barrier_buffer(op->get_barrier_workspace_size(args) *
                                (sizeof(PerTileFlags) / sizeof(int)) * 8);
        // lazy_init_barrier_buffer fills the existing pointer vector in place.
        cr2_args = args;
        cr2_op = std::move(op);
        ++cr2_initializations;
      }
      // Input and optimizer-updated weight STORAGE may change between calls.
      // Never reuse a stale device pointer or an initialized CUTLASS launch.
      auto &args = cr2_args.value();
      args.input = input.data_ptr();
      args.weight = weight.data_ptr();
      void *workspace = gemm_buffer.defined() ? gemm_buffer.data_ptr() : nullptr;
      cr2_op->run(args, workspace, stream);
      // Identical C publication rendezvous. Double buffering continues to guard
      // previous consumers before a later invocation reuses this source slot.
      group_barrier.barrier_all(stream);
      auto output = torch::empty({owned_taco.m / owned_taco.world, owned_taco.n},
                                 output_buffer.options());
      CUDA_CHECK(static_cast<cudaError_t>(taco_decode_reduce(
          owned_taco, output_buffer.data_ptr(), output.data_ptr(), stream)));
      ++cr2_calls;
      return output;
    } catch (...) {
      cr2_failed = true;
      throw;
    }
  }

 public:
  std::vector<int64_t>
  cr2_submission_state() const {
    return {1, cr2_initializations, cr2_calls, cr2_k,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(owned_taco.peers[owned_taco.rank])),
            cr2_failed ? 1 : 0};
  }

'''


def wrapper(source):
    source = replace_once(source, '#include <ATen/core/jit_type.h>',
                          '#include <ATen/core/jit_type.h>\n#include <ATen/core/grad_mode.h>')
    source = replace_once(source, '  const FluxTacoConfig owned_taco = taco_config();', '''  const FluxTacoConfig owned_taco = taco_config();
  const int cr2_k = taco_cr2_bound_k();
  OpRegistry::OpPtr cr2_op;
  c10::optional<GemmReduceScatterArguments> cr2_args;
  int64_t cr2_initializations = 0;
  int64_t cr2_calls = 0;
  bool cr2_failed = false;''')
    source = replace_once(source, '''  void
  forward_gemm_impl(''', CR2_METHOD + '''  void
  forward_gemm_impl(''')
    source = replace_once(source, '''      taco_set_config(owned_taco);
    }
#endif
    return forward_impl(''', '''      taco_set_config(owned_taco);
      TORCH_CHECK(cr2_k != 0 && !bias.has_value() && !input_scale.has_value() &&
                      !weight_scale.has_value() && !output_scale.has_value() && !fast_accum,
                  "CR2 owned instance requires the fixed no-bias/no-scale submission");
      return cr2_forward(std::move(input), std::move(weight), opt);
    }
#endif
    return forward_impl(''')
    source = replace_once(source, 'GemmRS::~GemmRS() { delete impl_; }', '''GemmRS::~GemmRS() { delete impl_; }
std::vector<int64_t>
GemmRS::cr2_submission_state() const {
  FLUX_CHECK(impl_ != nullptr) << "GemmRS is not initialized";
  return impl_->cr2_submission_state();
}''')
    return source


def wrapper_header(source):
    return replace_once(source, '  void zero_buffers();',
                         '  void zero_buffers();\n  std::vector<int64_t> cr2_submission_state() const;')


def pybind(source):
    return replace_once(source, '        .def("zero_buffers", &GemmRSOpCls::zero_buffers)',
                         '        .def("zero_buffers", &GemmRSOpCls::zero_buffers)\n'
                         '        .def("cr2_submission_state", &GemmRSOpCls::cr2_submission_state)')


def frontend(source):
    source = replace_once(source, '        self.graph_enabled = graph', '''        self._failed = False
        if graph or not _defer_reuse_barrier or (self.world, m, n) != (8, 2048, 4096) or k_local not in (512, 2048):
            raise ValueError('CR2 requires exact TP8 double-buffered eager attention/MLP')
        self.graph_enabled = graph''')
    source = replace_once(source, '            self.lib.taco_reset.argtypes = []', '''            self.lib.taco_cr2_protocol_version.argtypes = []
            self.lib.taco_cr2_protocol_version.restype = ctypes.c_int
            self.lib.taco_cr2_bind.argtypes = [ctypes.c_int]
            self.lib.taco_cr2_bind.restype = ctypes.c_int
            if self.lib.taco_cr2_protocol_version() != 1:
                raise RuntimeError('Wrong CR2 native submission protocol')
            self.lib.taco_reset.argtypes = []''')
    source = replace_once(source, '''        try:
            self.op = cpp_mod.GemmRS''', '''        try:
            code = self.lib.taco_cr2_bind(k_local)
            if code:
                raise RuntimeError(f'CR2 exact-shape binding rejected: CUDA error {code}')
            self.op = cpp_mod.GemmRS''')
    start = source.index('    def forward(self, input, weight):')
    end = source.index('    @property\n    def wire_bytes_per_rank', start)
    source = source[:start] + '''    def forward(self, input, weight):
        with self._lock:
            if self._failed:
                raise RuntimeError('CR2 cannot reuse an instance after a failed forward')
            try:
                return self.op.forward(input, weight, reduce_scatter_option=self.option)
            except BaseException:
                self._failed = True
                raise

    def cr2_submission_state(self):
        keys = ('protocol_version', 'initializations', 'successful_calls', 'bound_k', 'local_packet_ptr', 'failed')
        state = dict(zip(keys, self.op.cr2_submission_state()))
        state['failed'] = bool(state['failed'] or self._failed)
        return state

''' + source[end:]
    source = replace_once(source, '        self.next_slot = 0',
                          '        self.next_slot = 0\n        self._failed = False')
    source = replace_once(source, '''        with self._lock:
            output = self.slots[self.next_slot].forward(input, weight)
            self.next_slot ^= 1
            return output''', '''        with self._lock:
            if self._failed:
                raise RuntimeError('CR2 cannot reuse double workspaces after a failed forward')
            try:
                output = self.slots[self.next_slot].forward(input, weight)
            except BaseException:
                self._failed = True
                raise
            self.next_slot ^= 1
            return output

    def cr2_submission_state(self):
        return [dict(slot=index, **slot.cr2_submission_state())
                for index, slot in enumerate(self.slots)]''')
    return source


def sources(reference):
    reference = Path(reference)
    transforms = {RUNTIME: runtime, HEADER: header, WRAPPER: wrapper,
                  WRAPPER_HEADER: wrapper_header, PYBIND: pybind, FRONTEND: frontend}
    return {name: transform((reference/name).read_text())
            for name, transform in transforms.items()}
