"""MA4-G: typed, dynamic-address CUDA Graph replay over an exact CR2 tree.

Only the isolated G build receives these transforms. Recording uses a private
nonblocking stream and never advances forward counters or the double-buffer slot.
The graph retains CR2's StreamK reset, GEMM, IPC publication and exact decoder.
"""
from pathlib import Path

from common import replace_once, require
import ma2_cr2_sources as cr2

OPERATOR = 'include/flux/gemm_operator_base.h'
MIXIN = 'include/flux/cuda/gemm_impls/gemm_operator_base_default_impl.hpp'
GEMM = 'src/gemm_rs/gemm_v2_reduce_scatter.hpp'
BARRIER = 'src/cuda/cudaipc_barrier_all.cu'
RUNTIME, HEADER, WRAPPER = cr2.RUNTIME, cr2.HEADER, cr2.WRAPPER
WRAPPER_HEADER, PYBIND, FRONTEND = cr2.WRAPPER_HEADER, cr2.PYBIND, cr2.FRONTEND


def operator(source):
    return replace_once(source, '  virtual ~GemmOperatorBase() = default;', '''  virtual ~GemmOperatorBase() = default;
  // MA4-G is deliberately opt-in. Unsupported kernels fail closed.
  virtual void *ma4_graph_kernel_function() const {
    throw std::logic_error("MA4-G requires its typed SM80 BF16 NVLink kernel");
  }
  virtual void ma4_graph_bind_kernel(void *) {
    throw std::logic_error("MA4-G graph binding is unsupported by this operator");
  }
  virtual void ma4_graph_update_kernel(std::any const &, void *, void *, void *) {
    throw std::logic_error("MA4-G typed graph updates are unsupported by this operator");
  }''')


def mixin(source):
    return replace_once(source, ' private:\n  std::any gemm_op_;',
                         ' protected:\n  std::any gemm_op_;')


TYPED_GEMM = '''
// The real CUTLASS Params object supplies every kernel argument. No byte
// scanning, guessed offsets, device parameter tables or staging copies.
template <typename Kernel>
class Ma4GraphGemm : public cutlass::gemm::device::GemmUniversalBase<Kernel> {
 public:
  using Base = cutlass::gemm::device::GemmUniversalBase<Kernel>;
  cudaKernelNodeParams ma4_parameters{};
  bool ma4_bound = false;
  void bind_graph(cudaGraphNode_t node) {
    FLUX_CHECK(!ma4_bound);
    CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &ma4_parameters));
    const auto grid = this->params_.get_grid_dims();
    FLUX_CHECK(ma4_parameters.func == reinterpret_cast<void *>(cutlass::Kernel2<Kernel>));
    FLUX_CHECK(ma4_parameters.gridDim.x == grid.x && ma4_parameters.gridDim.y == grid.y &&
               ma4_parameters.gridDim.z == grid.z);
    FLUX_CHECK(ma4_parameters.blockDim.x == Kernel::kThreadCount &&
               ma4_parameters.blockDim.y == 1 && ma4_parameters.blockDim.z == 1);
    FLUX_CHECK(ma4_parameters.sharedMemBytes == sizeof(typename Kernel::SharedStorage));
    ma4_parameters.kernelParams = nullptr;
    ma4_parameters.extra = nullptr;
    ma4_bound = true;
  }
  void update_graph(cudaGraphExec_t exec, cudaGraphNode_t node,
                    typename Base::Arguments const &args) {
    FLUX_CHECK(ma4_bound);
    CUTLASS_CHECK(Base::update(args));
    void *arguments[] = {&this->params_};
    // Reuse the captured launch geometry, changing only a typed argument value.
    cudaKernelNodeParams parameters = ma4_parameters;
    parameters.kernelParams = arguments;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec, node, &parameters));
  }
};

'''

GEMM_METHODS = '''
  static constexpr bool ma4_supported =
      meta.arch() == _Sm80{} && rs_meta.comm_kind() == _IntraNode{} &&
      dt_conf.a() == _BF16{} && dt_conf.b() == _BF16{} &&
      dt_conf.d() == _BF16{} && !Base::has_bias;

  void *ma4_graph_kernel_function() const override {
    if constexpr (ma4_supported) {
      return reinterpret_cast<void *>(cutlass::Kernel2<GemmKernelT>);
    } else {
      throw std::logic_error("MA4-G only supports the frozen BF16 NVLink kernel");
    }
  }

  void ma4_graph_bind_kernel(void *node) override {
    if constexpr (ma4_supported) {
      using Gemm = identity_t<decltype(this->gemm_device())>;
      std::any_cast<Gemm &>(this->gemm_op_).bind_graph(static_cast<cudaGraphNode_t>(node));
    } else {
      throw std::logic_error("MA4-G graph binding unavailable");
    }
  }

  void ma4_graph_update_kernel(std::any const &args, void *workspace,
                               void *exec, void *node) override {
    if constexpr (ma4_supported) {
      FLUX_CHECK(this->get_args_workspace_size(args) == 0);
      using Gemm = identity_t<decltype(this->gemm_device())>;
      auto gemm_args = this->to_gemm_args(args, workspace);
      auto &gemm = std::any_cast<Gemm &>(this->gemm_op_);
      gemm.update_graph(static_cast<cudaGraphExec_t>(exec),
                        static_cast<cudaGraphNode_t>(node), gemm_args);
    } else {
      throw std::logic_error("MA4-G typed graph update unavailable");
    }
  }

'''


def gemm(source):
    source = replace_once(source, 'namespace bytedance::flux {\n',
                          'namespace bytedance::flux {\n' + TYPED_GEMM)
    marker = '  using Base::has_bias;\n  using Base::is_fp8_gemm;\n  using Base::is_sm89;'
    source = replace_once(source, marker, marker + '\n' + GEMM_METHODS)
    return replace_once(source, '''    } else {
      return make_declval<cutlass::gemm::device::GemmUniversalBase<GemmKernelT>>();
    }
  }

  auto
  to_gemm_args_impl''', '''    } else if constexpr (ma4_supported) {
      return make_declval<Ma4GraphGemm<GemmKernelT>>();
    } else {
      return make_declval<cutlass::gemm::device::GemmUniversalBase<GemmKernelT>>();
    }
  }

  auto
  to_gemm_args_impl''')


def runtime(source):
    require('taco_cr2_protocol_version() { return 1; }' in source,
            'MA4-G requires an exact CR2 runtime')
    return source + '''
extern "C" int taco_ma4g_protocol_version() { return 1; }
extern "C" void *taco_ma4g_decoder_function() {
  return reinterpret_cast<void *>(taco_ma2c_decode_kernel);
}
extern "C" int taco_ma4g_update_decoder(cudaGraphExec_t exec, cudaGraphNode_t node,
    FluxTacoConfig c, const void *local, void *output) {
  if (!exec || !node || !c.enabled || !c.selected || !taco_ma2_exact_shape(c) ||
      !local || !output || !c.deferred_reuse_barrier || c.allow_capture)
    return int(cudaErrorInvalidValue);
  auto local_bf16 = static_cast<const __nv_bfloat16 *>(local);
  auto output_bf16 = static_cast<__nv_bfloat16 *>(output);
  void *arguments[] = {&c, &local_bf16, &output_bf16};
  cudaKernelNodeParams parameters{};
  parameters.func = reinterpret_cast<void *>(taco_ma2c_decode_kernel);
  parameters.gridDim = dim3(2048, 1, 1);
  parameters.blockDim = dim3(128, 1, 1);
  parameters.kernelParams = arguments;
  return int(cudaGraphExecKernelNodeSetParams(exec, node, &parameters));
}
'''


def header(source):
    return source + '''
extern "C" int taco_ma4g_protocol_version();
extern "C" void *taco_ma4g_decoder_function();
extern "C" int taco_ma4g_update_decoder(cudaGraphExec_t, cudaGraphNode_t,
    FluxTacoConfig, const void *, void *);
'''


GRAPH_FIELDS = '''
  // One graph per complete CR2 workspace; the Python wrapper owns two instances.
  cudaGraph_t ma4_graph = nullptr;
  cudaGraphExec_t ma4_exec = nullptr;
  cudaGraphNode_t ma4_gemm = nullptr, ma4_barrier = nullptr;
  cudaGraphNode_t ma4_decoder = nullptr, ma4_reset = nullptr;
  void *ma4_workspace = nullptr;
  int64_t ma4_builds = 0, ma4_launches = 0;
  int64_t ma4_gemm_updates = 0, ma4_decoder_updates = 0;
  int64_t ma4_build_host_ns = 0;
  bool ma4_prepare_only = false;
  std::vector<torch::Tensor> ma4_resources;
  mutable std::recursive_mutex ma4_mutex;
'''

GRAPH_METHODS = '''
  void ma4_build_graph(torch::Tensor const &input, torch::Tensor const &weight,
                       torch::Tensor const &output, void *workspace,
                       cudaStream_t stream) {
    TORCH_CHECK(!ma4_graph && !ma4_exec, "MA4-G graph was already constructed");
    const auto started = std::chrono::steady_clock::now();
    // Prime CUTLASS's device properties and allocate all resources before capture.
    // This initial reset is outside the replay graph. No IPC handshake is reset.
    cr2_op->initialize(cr2_args.value(), workspace, stream);
    cudaStream_t recording = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&recording, cudaStreamNonBlocking));
    bool capturing = false;
    try {
      CUDA_CHECK(cudaStreamBeginCapture(recording, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      // A recording-only path: it never calls forward or advances any counter.
      // initialize() records the SAME StreamK memset required by eager CR2.
      cr2_op->run(cr2_args.value(), workspace, recording);
      group_barrier.barrier_all(recording);
      CUDA_CHECK(static_cast<cudaError_t>(taco_decode_reduce(
          owned_taco, output_buffer.data_ptr(), output.data_ptr(), recording)));
      auto ended = cudaStreamEndCapture(recording, &ma4_graph);
      capturing = false;
      CUDA_CHECK(ended);
      CUDA_CHECK(cudaStreamDestroy(recording));
      recording = nullptr;
      size_t count = 0;
      CUDA_CHECK(cudaGraphGetNodes(ma4_graph, nullptr, &count));
      TORCH_CHECK(count == 4, "MA4-G requires exactly reset + GEMM + barrier + decoder");
      std::vector<cudaGraphNode_t> nodes(count);
      CUDA_CHECK(cudaGraphGetNodes(ma4_graph, nodes.data(), &count));
      for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeMemset) {
          TORCH_CHECK(!ma4_reset, "MA4-G has multiple workspace resets");
          ma4_reset = node;
          cudaMemsetParams reset{};
          CUDA_CHECK(cudaGraphMemsetNodeGetParams(node, &reset));
          const auto begin = reinterpret_cast<uintptr_t>(workspace);
          const auto end = begin + gemm_buffer.nbytes();
          const auto address = reinterpret_cast<uintptr_t>(reset.dst);
          TORCH_CHECK(reset.value == 0 && reset.height == 1 &&
                          address >= begin && address < end &&
                          reset.width * reset.elementSize <= end - address,
                      "MA4-G may only reset its private GEMM workspace");
        } else {
          TORCH_CHECK(type == cudaGraphNodeTypeKernel,
                      "MA4-G forbids copies, events and hidden work in its graph");
          cudaKernelNodeParams parameters{};
          CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &parameters));
          if (parameters.func == cr2_op->ma4_graph_kernel_function()) {
            TORCH_CHECK(!ma4_gemm, "MA4-G duplicate GEMM node");
            ma4_gemm = node;
          } else if (parameters.func == taco_ma4g_decoder_function()) {
            TORCH_CHECK(!ma4_decoder, "MA4-G duplicate decoder node");
            ma4_decoder = node;
          } else if (parameters.func == ::bytedance::flux::ma4_barrier_kernel_function()) {
            TORCH_CHECK(!ma4_barrier, "MA4-G duplicate publication barrier");
            ma4_barrier = node;
          } else {
            TORCH_CHECK(false, "MA4-G unexpected kernel function");
          }
        }
      }
      TORCH_CHECK(ma4_reset && ma4_gemm && ma4_barrier && ma4_decoder,
                  "MA4-G required node identity missing");
      // Capture must reproduce the original single-stream dependency chain.
      cudaGraphNode_t ordered[] = {ma4_reset, ma4_gemm, ma4_barrier, ma4_decoder};
      for (int index = 0; index < 4; ++index) {
        size_t dependencies = 0;
        CUDA_CHECK(cudaGraphNodeGetDependencies(ordered[index], nullptr, &dependencies));
        TORCH_CHECK(dependencies == (index == 0 ? 0 : 1), "MA4-G graph dependency changed");
        if (index) {
          cudaGraphNode_t previous;
          CUDA_CHECK(cudaGraphNodeGetDependencies(ordered[index], &previous, &dependencies));
          TORCH_CHECK(previous == ordered[index - 1], "MA4-G graph order changed");
        }
      }
      cr2_op->ma4_graph_bind_kernel(ma4_gemm);
      CUDA_CHECK(cudaGraphInstantiate(&ma4_exec, ma4_graph, nullptr, nullptr, 0));
      ma4_workspace = workspace;
      ++ma4_builds;
      ma4_build_host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started).count();
    } catch (...) {
      if (capturing) {
        cudaGraph_t invalid = nullptr;
        cudaStreamEndCapture(recording, &invalid);
        if (invalid) cudaGraphDestroy(invalid);
      }
      if (recording) cudaStreamDestroy(recording);
      throw;
    }
  }

 public:
  void ma4_hold_resources(std::vector<torch::Tensor> resources) {
    std::lock_guard<std::recursive_mutex> lock(ma4_mutex);
    TORCH_CHECK(!ma4_graph && ma4_resources.empty(), "MA4-G resources are immutable");
    TORCH_CHECK(resources.size() == (owned_taco.quant_tile_count >= 0 ? 10 : 9),
                "MA4-G requires its exact IPC, mask and optional tile-index resources");
    for (int index = 0; index < owned_taco.world; ++index)
      TORCH_CHECK(resources[index].defined() &&
                      resources[index].data_ptr() == owned_taco.peers[index],
                  "MA4-G peer resource ownership mismatch");
    TORCH_CHECK(resources[8].defined() && resources[8].data_ptr() == owned_taco.selected,
                "MA4-G mask resource ownership mismatch");
    if (owned_taco.quant_tile_count >= 0)
      TORCH_CHECK(resources[9].defined() && resources[9].data_ptr() == owned_taco.quant_tiles,
                  "MA4-G tile-index resource ownership mismatch");
    ma4_resources = std::move(resources);
  }

  void ma4_prepare_graph(torch::Tensor input, torch::Tensor weight,
                         const ReduceScatterOptionWithOptional &opt) {
    std::lock_guard<std::recursive_mutex> lock(ma4_mutex);
    TORCH_CHECK(!cr2_failed, "MA4-G cannot prepare a failed instance");
    struct Reset { bool &flag; ~Reset() { flag = false; } } reset{ma4_prepare_only};
    ma4_prepare_only = true;
    (void)forward(std::move(input), std::move(weight), c10::nullopt,
                  c10::nullopt, c10::nullopt, c10::nullopt, false, opt);
  }

  std::vector<int64_t> ma4_graph_state() const {
    std::lock_guard<std::recursive_mutex> lock(ma4_mutex);
    auto id = [](auto pointer) { return static_cast<int64_t>(reinterpret_cast<uintptr_t>(pointer)); };
    return {1, ma4_builds, ma4_launches, ma4_gemm_updates, ma4_decoder_updates,
            ma4_exec ? 4 : 0, ma4_exec ? 3 : 0, ma4_exec ? 1 : 0,
            cr2_k, id(ma4_graph), id(ma4_exec), id(ma4_gemm), id(ma4_barrier),
            id(ma4_decoder), id(ma4_reset), id(owned_taco.peers[owned_taco.rank]),
            cr2_failed ? 1 : 0, ma4_build_host_ns,
            static_cast<int64_t>(taco_stream.device_index()), id(taco_stream.stream()),
            id(ma4_workspace)};
  }

'''

GRAPH_FORWARD = '''      // Every output owns its allocation; no graph-stable output or clone.
      auto output = torch::empty({owned_taco.m / owned_taco.world, owned_taco.n},
                                 output_buffer.options());
      if (!ma4_exec) ma4_build_graph(input, weight, output, workspace, stream);
      TORCH_CHECK(workspace == ma4_workspace, "MA4-G workspace identity changed");
      if (ma4_prepare_only) return output;
      // PyTorch's allocator must know every storage used on this bound stream.
      c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
      c10::cuda::CUDACachingAllocator::recordStream(weight.storage().data_ptr(), stream);
      c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
      // CUTLASS update reconstructs epilogue/TACO as well as A/B pointers.
      cr2_op->ma4_graph_update_kernel(args, workspace, ma4_exec, ma4_gemm);
      ++ma4_gemm_updates;
      CUDA_CHECK(static_cast<cudaError_t>(taco_ma4g_update_decoder(
          ma4_exec, ma4_decoder, owned_taco, output_buffer.data_ptr(), output.data_ptr())));
      ++ma4_decoder_updates;
      CUDA_CHECK(cudaGraphLaunch(ma4_exec, stream));
      ++ma4_launches;
      ++cr2_calls;
      return output;'''


def wrapper(source):
    require('cr2_forward(' in source, 'MA4-G requires CR2 native wrapper')
    source = replace_once(source, '#include <ATen/core/grad_mode.h>',
                          '#include <ATen/core/grad_mode.h>\n#include <chrono>\n#include <mutex>\n'
                          '#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDACachingAllocator.h>')
    source = replace_once(source, 'namespace bytedance::flux::ths_op {',
                          'namespace bytedance::flux { void *ma4_barrier_kernel_function(); }\n'
                          'namespace bytedance::flux::ths_op {')
    source = replace_once(source, '  bool cr2_failed = false;',
                          '  bool cr2_failed = false;\n' + GRAPH_FIELDS)
    source = replace_once(source, '    TORCH_CHECK(!cr2_failed, "CR2 cannot retry a failed submission");',
                          '    std::lock_guard<std::recursive_mutex> lock(ma4_mutex);\n'
                          '    TORCH_CHECK(!cr2_failed, "MA4-G cannot retry a failed submission");')
    start = source.index('      cr2_op->run(args, workspace, stream);')
    end = source.index('      return output;', start) + len('      return output;')
    source = source[:start] + GRAPH_FORWARD + source[end:]
    source = replace_once(source, '      auto stream = c10::cuda::getCurrentCUDAStream();\n      if (!cr2_op) {',
                          '      TORCH_CHECK(ma4_resources.size() >= 9, "MA4-G IPC resources must be owned");\n'
                          '      TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&\n'
                          '                      reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,\n'
                          '                  "MA4-G requires 16-byte aligned input and weight");\n'
                          '      auto stream = c10::cuda::getCurrentCUDAStream();\n      if (!cr2_op) {')
    source = replace_once(source, '  torch::Tensor\n  cr2_forward(',
                          GRAPH_METHODS + '  torch::Tensor\n  cr2_forward(')
    source = replace_once(source, '  ~GemmRSImpl() {', '''  ~GemmRSImpl() {
#ifdef FLUX_TACO_BASELINE
    // Quiesce outstanding graphs before destroying graph handles or IPC tensors.
    if (ma4_graph || ma4_exec) {
      c10::cuda::CUDAGuard guard(taco_stream.device_index());
      CUDA_CHECK(cudaStreamSynchronize(taco_stream));
      if (ma4_exec) CUDA_CHECK(cudaGraphExecDestroy(ma4_exec));
      if (ma4_graph) CUDA_CHECK(cudaGraphDestroy(ma4_graph));
    }
#endif''')
    source = replace_once(source, 'GemmRS::~GemmRS() { delete impl_; }', '''GemmRS::~GemmRS() { delete impl_; }
void GemmRS::ma4_hold_resources(std::vector<torch::Tensor> resources) {
  FLUX_CHECK(impl_ != nullptr);
  impl_->ma4_hold_resources(std::move(resources));
}
void GemmRS::ma4_prepare_graph(torch::Tensor input, torch::Tensor weight,
                             const ReduceScatterOptionWithOptional &opt) {
  FLUX_CHECK(impl_ != nullptr);
  impl_->ma4_prepare_graph(std::move(input), std::move(weight), opt);
}
std::vector<int64_t> GemmRS::ma4_graph_state() const {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->ma4_graph_state();
}''')
    return source


def wrapper_header(source):
    return replace_once(source, '  std::vector<int64_t> cr2_submission_state() const;',
                         '''  std::vector<int64_t> cr2_submission_state() const;
  std::vector<int64_t> ma4_graph_state() const;
  void ma4_hold_resources(std::vector<torch::Tensor>);
  void ma4_prepare_graph(torch::Tensor, torch::Tensor,
                         const ReduceScatterOptionWithOptional &);''')


def pybind(source):
    return replace_once(source, '        .def("cr2_submission_state", &GemmRSOpCls::cr2_submission_state)',
                         '''        .def("cr2_submission_state", &GemmRSOpCls::cr2_submission_state)
        .def("ma4_graph_state", &GemmRSOpCls::ma4_graph_state)
        .def("ma4_hold_resources", &GemmRSOpCls::ma4_hold_resources)
        .def("ma4_prepare_graph", &GemmRSOpCls::ma4_prepare_graph)''')


STATE_KEYS = ('protocol_version', 'graph_build_count', 'graph_launch_count',
              'gemm_param_update_count', 'decoder_param_update_count', 'node_count',
              'kernel_node_count', 'memset_node_count', 'bound_k', 'graph_id',
              'graph_exec_id', 'gemm_node_id', 'barrier_node_id', 'decoder_node_id',
              'workspace_reset_node_id', 'local_packet_ptr', 'failed',
              'graph_build_host_ns', 'device_index', 'stream_id', 'workspace_ptr')


def frontend(source):
    source = replace_once(source, '            self.lib.taco_cr2_protocol_version.argtypes = []',
                         '''            self.lib.taco_ma4g_protocol_version.argtypes = []
            self.lib.taco_ma4g_protocol_version.restype = ctypes.c_int
            if self.lib.taco_ma4g_protocol_version() != 1:
                raise RuntimeError('Wrong MA4-G native graph protocol')
            self.lib.taco_cr2_protocol_version.argtypes = []''')
    source = replace_once(source, '                                    transpose_weight=False, fuse_reduction=False, ring_reduction=True)\n        finally:',
                          '                                    transpose_weight=False, fuse_reduction=False, ring_reduction=True)\n'
                          '            self.op.ma4_hold_resources(self.peers + [self.selected] +\n'
                          '                ([] if self.quant_tiles is None else [self.quant_tiles]))\n        finally:')
    marker = '    def cr2_submission_state(self):'
    position = source.index(marker)
    methods = '''    def prepare_graph(self, input, weight):
        with self._lock:
            if self._failed:
                raise RuntimeError('MA4-G cannot prepare a failed instance')
            try:
                self.op.ma4_prepare_graph(input, weight, self.option)
            except BaseException:
                self._failed = True
                raise

    def ma4_graph_state(self):
        keys = STATE_KEYS_LITERAL
        values = self.op.ma4_graph_state()
        if len(values) != len(keys):
            raise RuntimeError('MA4-G native state ABI mismatch')
        state = dict(zip(keys, values))
        state['failed'] = bool(state['failed'] or self._failed)
        return state

'''.replace('STATE_KEYS_LITERAL', repr(STATE_KEYS))
    source = source[:position] + methods + source[position:]
    position = source.index(marker, source.index('class GemmRSTacoDoubleBuffered:'))
    double_methods = '''    def prepare_graph(self, input, weight):
        with self._lock:
            if self._failed:
                raise RuntimeError('MA4-G cannot prepare failed double workspaces')
            try:
                for slot in self.slots:
                    slot.prepare_graph(input, weight)
            except BaseException:
                self._failed = True
                raise

    def ma4_graph_state(self):
        return [dict(slot=index, **slot.ma4_graph_state())
                for index, slot in enumerate(self.slots)]

'''
    return source[:position] + double_methods + source[position:]


def sources(reference):
    reference = Path(reference)
    transforms = {OPERATOR: operator, MIXIN: mixin, GEMM: gemm, BARRIER: barrier,
                  RUNTIME: runtime, HEADER: header, WRAPPER: wrapper,
                  WRAPPER_HEADER: wrapper_header, PYBIND: pybind, FRONTEND: frontend}
    return {name: transform((reference / name).read_text())
            for name, transform in transforms.items()}


def barrier(source):
    return source + '\nnamespace bytedance::flux {\nvoid *ma4_barrier_kernel_function() {\n  return reinterpret_cast<void *>(CudaIpcBarrierAllKernel);\n}\n}\n'


def compile_probe_source():
    """Instantiate the actual frozen V2 kernel and its virtual graph updater."""
    return r'''#include "gemm_rs/gemm_v2_reduce_scatter.hpp"
namespace bytedance::flux {
using namespace cute;
using Meta = decltype(make_gemm_meta(
    make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{}),
    _Sm80{}, _A100{}, _ReduceScatter{}, _RCR{}, _GemmV2{}, None{},
    make_reduce_scatter_meta(_False{}, _IntraNode{})));
using HP = decltype(materialize_hparams(Meta{}, make_gemm_hparams(
    Auto{}, Auto{}, Shape<_128, _128, _32>{}, Auto{}, _3{})));
using Kernel = decltype(GemmV2ReduceScatter_Kernel<Meta, HP>().gemm_kernel());
using Device = GemmV2ReduceScatter_Device<Meta, HP, Kernel>;
static_assert(Device::ma4_supported);
extern "C" void ma4_typed_compile_probe(std::any const &args, void *workspace,
                                       void *exec, void *node) {
  Device op;
  op.ma4_graph_kernel_function();
  op.ma4_graph_update_kernel(args, workspace, exec, node);
}
}
'''
