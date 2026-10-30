"""MA4-F: fuse the two ordered BF16 projection-consumer adds into CR2's decoder.

Only F4 receives this transform. Ordinary forward remains an explicit projection
reference; forward_consumer is the measured path and has separate native counters.
No graph, wire protocol, source reduction order, or publication barrier changes.
"""
from pathlib import Path

from common import replace_once, require
import ma2_cr2_sources as cr2

RUNTIME, HEADER, WRAPPER = cr2.RUNTIME, cr2.HEADER, cr2.WRAPPER
WRAPPER_HEADER, PYBIND, FRONTEND = cr2.WRAPPER_HEADER, cr2.PYBIND, cr2.FRONTEND
DECODER = 'src/gemm_rs/ma4_f_decode.cuh'
CR2_DECODER = 'src/gemm_rs/ma2_c_decode.cuh'


CONSUMER_STORE = r'''
// Match aten BF16 x+bias, then residual+out: two distinct rounded additions.
// The original ordered reduction is already complete before this function.
__device__ __forceinline__ void taco_ma4f_store(
    __nv_bfloat16* output, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, int64_t offset, uint2 packed) {
  const uint2 b = *reinterpret_cast<const uint2*>(bias + offset % 4096);
  const uint2 r = *reinterpret_cast<const uint2*>(residual + offset);
  auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
  const auto* add_bias = reinterpret_cast<const __nv_bfloat162*>(&b);
  const auto* add_residual = reinterpret_cast<const __nv_bfloat162*>(&r);
  sum[0] = __hadd2(sum[0], add_bias[0]);
  sum[1] = __hadd2(sum[1], add_bias[1]);
  sum[0] = __hadd2(add_residual[0], sum[0]);
  sum[1] = __hadd2(add_residual[1], sum[1]);
  *reinterpret_cast<uint2*>(output + offset) = packed;
}
'''


def decoder(source):
    start = source.index('__global__ void taco_ma2c_decode_kernel(')
    kernel = source[start:]
    kernel = replace_once(kernel, 'taco_ma2c_decode_kernel', 'taco_ma4f_decode_consumer_kernel')
    kernel = replace_once(kernel, 'const __nv_bfloat16* local, __nv_bfloat16* output)',
                          'const __nv_bfloat16* local, __nv_bfloat16* output,\n'
                          '    const __nv_bfloat16* bias, const __nv_bfloat16* residual)')
    kernel = replace_once(kernel, '*reinterpret_cast<uint2*>(output + offset) = packed;',
                          'taco_ma4f_store(output, bias, residual, offset, packed);')
    kernel = replace_once(kernel, '  taco_ma2::store(output, group, accum);', '''  uint2 packed;
  auto* pair = reinterpret_cast<__nv_bfloat162*>(&packed);
  pair[0] = __halves2bfloat162(accum[0], accum[1]);
  pair[1] = __halves2bfloat162(accum[2], accum[3]);
  taco_ma4f_store(output, bias, residual, int64_t(group) * 128 + lane * 4, packed);''')
    return ('// CR2 decoder plus post-reduction bias/residual; preserve two BF16 roundings.\n'
            '// TACO attribution: tools/a800/phase4/TACO_LICENSE.txt.\n#pragma once\n'
            '#include "gemm_rs/ma2_c_decode.cuh"\n' + CONSUMER_STORE + '\n' + kernel)


def runtime(source):
    require('taco_cr2_protocol_version() { return 1; }' in source, 'MA4-F requires the exact CR2 runtime')
    source = replace_once(source, '#include "gemm_rs/ma2_c_decode.cuh"',
                          '#include "gemm_rs/ma2_c_decode.cuh"\n#include "gemm_rs/ma4_f_decode.cuh"')
    return source + r'''
extern "C" int taco_ma4f_protocol_version() { return 1; }
extern "C" int taco_ma4f_decode_reduce(FluxTacoConfig c, const void* local,
    void* output, const void* bias, const void* residual, cudaStream_t stream) {
  if (!c.enabled || !local || !output || !bias || !residual || !c.selected ||
      !taco_ma2_exact_shape(c) || !c.deferred_reuse_barrier || c.allow_capture)
    return int(cudaErrorInvalidValue);
  taco_ma4f_decode_consumer_kernel<<<2048, 128, 0, stream>>>(c,
      static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output),
      static_cast<const __nv_bfloat16*>(bias), static_cast<const __nv_bfloat16*>(residual));
  return int(cudaGetLastError());
}
'''


def header(source):
    return source + r'''
extern "C" int taco_ma4f_protocol_version();
extern "C" int taco_ma4f_decode_reduce(FluxTacoConfig, const void*, void*,
    const void*, const void*, cudaStream_t);
'''


CONSUMER_METHODS = r'''
 public:
  torch::Tensor ma4_consumer_forward(torch::Tensor input, torch::Tensor weight,
      torch::Tensor bias, torch::Tensor residual,
      const ReduceScatterOptionWithOptional &opt) {
    std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
    TORCH_CHECK(!cr2_failed && !ma4_consumer_active, "MA4-F cannot retry or reenter a failed consumer");
    try {
      TORCH_CHECK(owned_taco.enabled && owned_taco.world == 8 && owned_taco.m == 2048 &&
                      owned_taco.n == 4096 && cr2_k != 0,
                  "MA4-F requires the frozen CR2 instance");
      TORCH_CHECK(bias.is_cuda() && residual.is_cuda() && bias.device() == input.device() &&
                      residual.device() == input.device() &&
                      bias.scalar_type() == at::ScalarType::BFloat16 &&
                      residual.scalar_type() == at::ScalarType::BFloat16 &&
                      bias.is_contiguous() && residual.is_contiguous() &&
                      bias.dim() == 1 && bias.size(0) == 4096 &&
                      residual.dim() == 2 && residual.size(0) == 256 && residual.size(1) == 4096,
                  "MA4-F requires BF16 contiguous bias[4096] and residual[256,4096]");
      TORCH_CHECK(reinterpret_cast<uintptr_t>(bias.data_ptr()) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(residual.data_ptr()) % 8 == 0,
                  "MA4-F consumer tensors must have 8-byte aligned addresses");
      TORCH_CHECK(!at::GradMode::is_enabled() ||
                      (!input.requires_grad() && !weight.requires_grad() &&
                       !bias.requires_grad() && !residual.requires_grad()),
                  "MA4-F requires no_grad or its explicit autograd adapter");
      struct ClearConsumer {
        GemmRSImpl* self;
        ~ClearConsumer() {
          self->ma4_bias = torch::Tensor();
          self->ma4_residual = torch::Tensor();
          self->ma4_consumer_active = false;
        }
      } clear{this};
      ma4_bias = std::move(bias);
      ma4_residual = std::move(residual);
      ma4_consumer_active = true;
      // Reuse CR2's layout, device, stream and TLS checks plus its exact producer.
      return forward(std::move(input), std::move(weight), c10::nullopt,
                     c10::nullopt, c10::nullopt, c10::nullopt, false, opt);
    } catch (...) {
      cr2_failed = true;
      throw;
    }
  }

  std::vector<int64_t> ma4_consumer_state() const {
    std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
    return {1, cr2_initializations, cr2_calls, ma4_consumer_calls, cr2_k,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(owned_taco.peers[owned_taco.rank])),
            cr2_failed ? 1 : 0};
  }

'''


def wrapper(source):
    require('cr2_forward(' in source, 'MA4-F requires CR2 native submission')
    source = replace_once(source, '#include <ATen/core/grad_mode.h>',
                          '#include <ATen/core/grad_mode.h>\n#include <mutex>\n'
                          '#include <c10/cuda/CUDACachingAllocator.h>')
    source = replace_once(source, '  bool cr2_failed = false;', '''  bool cr2_failed = false;
  mutable std::recursive_mutex ma4_consumer_mutex;
  bool ma4_consumer_active = false;
  int64_t ma4_consumer_calls = 0;
  torch::Tensor ma4_bias, ma4_residual;''')
    source = replace_once(source, '  torch::Tensor\n  cr2_forward(', CONSUMER_METHODS + '  torch::Tensor\n  cr2_forward(')
    source = replace_once(source, '    TORCH_CHECK(!cr2_failed, "CR2 cannot retry a failed submission");',
                          '    std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);\n'
                          '    TORCH_CHECK(!cr2_failed, "CR2 cannot retry a failed submission");')
    source = replace_once(source, '''      CUDA_CHECK(static_cast<cudaError_t>(taco_decode_reduce(
          owned_taco, output_buffer.data_ptr(), output.data_ptr(), stream)));''', '''      if (ma4_consumer_active) {
        c10::cuda::CUDACachingAllocator::recordStream(ma4_bias.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(ma4_residual.storage().data_ptr(), stream);
        CUDA_CHECK(static_cast<cudaError_t>(taco_ma4f_decode_reduce(owned_taco,
            output_buffer.data_ptr(), output.data_ptr(), ma4_bias.data_ptr(), ma4_residual.data_ptr(), stream)));
        ++ma4_consumer_calls;
      } else {
        // Explicit projection-only reference; measured F4 calls must use forward_consumer.
        CUDA_CHECK(static_cast<cudaError_t>(taco_decode_reduce(
            owned_taco, output_buffer.data_ptr(), output.data_ptr(), stream)));
      }''')
    source = replace_once(source, 'GemmRS::~GemmRS() { delete impl_; }', '''GemmRS::~GemmRS() { delete impl_; }
torch::Tensor GemmRS::ma4_consumer_forward(torch::Tensor input, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor residual, const ReduceScatterOptionWithOptional &opt) {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->ma4_consumer_forward(std::move(input), std::move(weight),
      std::move(bias), std::move(residual), opt);
}
std::vector<int64_t> GemmRS::ma4_consumer_state() const {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->ma4_consumer_state();
}''')
    return source


def wrapper_header(source):
    return replace_once(source, '  std::vector<int64_t> cr2_submission_state() const;', '''  std::vector<int64_t> cr2_submission_state() const;
  std::vector<int64_t> ma4_consumer_state() const;
  torch::Tensor ma4_consumer_forward(torch::Tensor, torch::Tensor, torch::Tensor,
      torch::Tensor, const ReduceScatterOptionWithOptional &);''')


def pybind(source):
    return replace_once(source, '        .def("cr2_submission_state", &GemmRSOpCls::cr2_submission_state)', '''        .def("cr2_submission_state", &GemmRSOpCls::cr2_submission_state)
        .def("ma4_consumer_state", &GemmRSOpCls::ma4_consumer_state)
        .def("ma4_consumer_forward", &GemmRSOpCls::ma4_consumer_forward)''')


STATE_KEYS = ('protocol_version', 'initializations', 'successful_calls', 'bias_residual_calls',
              'bound_k', 'local_packet_ptr', 'failed')


def frontend(source):
    source = replace_once(source, '            self.lib.taco_cr2_protocol_version.argtypes = []', '''            self.lib.taco_ma4f_protocol_version.argtypes = []
            self.lib.taco_ma4f_protocol_version.restype = ctypes.c_int
            if self.lib.taco_ma4f_protocol_version() != 1:
                raise RuntimeError('Wrong MA4-F consumer protocol')
            self.lib.taco_cr2_protocol_version.argtypes = []''')
    marker = '    def cr2_submission_state(self):'
    index = source.index(marker)
    methods = '''    def forward_consumer(self, input, weight, bias, residual):
        with self._lock:
            if self._failed:
                raise RuntimeError('MA4-F cannot reuse a failed consumer')
            try:
                return self.op.ma4_consumer_forward(input, weight, bias, residual, self.option)
            except BaseException:
                self._failed = True
                raise

    def ma4_consumer_state(self):
        keys = STATE_KEYS_LITERAL
        values = self.op.ma4_consumer_state()
        if len(values) != len(keys):
            raise RuntimeError('MA4-F native consumer state ABI mismatch')
        state = dict(zip(keys, values))
        state['failed'] = bool(state['failed'] or self._failed)
        return state

'''.replace('STATE_KEYS_LITERAL', repr(STATE_KEYS))
    source = source[:index] + methods + source[index:]
    index = source.index(marker, source.index('class GemmRSTacoDoubleBuffered:'))
    methods = '''    def forward_consumer(self, input, weight, bias, residual):
        with self._lock:
            if self._failed:
                raise RuntimeError('MA4-F cannot reuse failed double workspaces')
            try:
                output = self.slots[self.next_slot].forward_consumer(input, weight, bias, residual)
            except BaseException:
                self._failed = True
                raise
            self.next_slot ^= 1
            return output

    def ma4_consumer_state(self):
        return [dict(slot=index, **slot.ma4_consumer_state())
                for index, slot in enumerate(self.slots)]

'''
    return source[:index] + methods + source[index:]


def sources(reference):
    reference = Path(reference)
    transforms = {RUNTIME: runtime, HEADER: header, WRAPPER: wrapper,
                  WRAPPER_HEADER: wrapper_header, PYBIND: pybind, FRONTEND: frontend}
    result = {name: transform((reference/name).read_text()) for name, transform in transforms.items()}
    result[DECODER] = decoder((reference/CR2_DECODER).read_text())
    return result
