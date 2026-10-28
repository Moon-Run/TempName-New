"""MA6-v3: independent packed-mask and homogeneous-CTA receiver experiments.

Both retain F4's 2048 CTAs, 128 threads, one H128 group per warp, producer,
packet ABI, source fold order, and two BF16 consumer additions.
"""
from pathlib import Path

from common import replace_once, require
import ma4_f_sources as f4

RUNTIME, HEADER, WRAPPER, DECODER = f4.RUNTIME, f4.HEADER, f4.WRAPPER, f4.DECODER
WRAPPER_HEADER, PYBIND, FRONTEND = f4.WRAPPER_HEADER, f4.PYBIND, f4.FRONTEND
MASK_HEADER = 'src/gemm_rs/ma6_mask.h'
BASELINE_TUNING = dict(decoder_threads=128, decoder_blocks=2048, decoder_rows=1)
ROLES = ('attn', 'mlp')
ROLE_K = dict(attn=512, mlp=2048)
DEFAULT_TUNING = {role:dict(BASELINE_TUNING) for role in ROLES}
VARIANTS = ('F4', 'F4-mask', 'F4-layout')
RECEIVER_VARIANTS = {'F4':0, 'F4-mask':1, 'F4-layout':2}
GROUP_MAPPING = '(blockIdx.x / 32) * 128 + blockIdx.x % 32 + (threadIdx.x / 32) * 32'


def tuning_config(tuning=None):
    tuning = DEFAULT_TUNING if tuning is None else tuning
    require(isinstance(tuning, dict) and set(tuning) == set(ROLES),
            'MA6 tuning requires attn and mlp')
    for role in ROLES:
        item = tuning[role]
        require(isinstance(item, dict) and set(item) == set(BASELINE_TUNING)
                and all(type(v) is int for v in item.values()) and item == BASELINE_TUNING,
                'MA6-v3 fixes decoder geometry to 128 threads, 2048 blocks, one row: ' + role)
    return {role:dict(BASELINE_TUNING) for role in ROLES}


def kernel_name(role, tuning=None, arm='F4-mask'):
    tuning_config(tuning)
    require(role in ROLES and arm in VARIANTS, 'Unknown MA6 decoder role or arm')
    return ('taco_ma4f_decode_consumer_kernel' if arm == 'F4' else
            'taco_ma6_' + arm.removeprefix('F4-') + '_decode_consumer_kernel')


def decode_entry(arm):
    require(arm in VARIANTS[1:], 'MA6 receiver transform needs a candidate arm')
    return 'taco_ma6_decode_' + arm.removeprefix('F4-')


# Host-only helper is compiled and executed against frozen A/M masks in compile-check.
# Keeping it separate also makes the exact production packer independently testable.
PACK_MASK = '''#pragma once
#include <cstdint>
inline bool taco_ma6_pack_mask(const uint8_t* selected, uint8_t* packed) {
  for (int tile = 0; tile < 512; ++tile) {
    uint8_t bits = 0;
    for (int src = 0; src < 8; ++src) {
      const uint8_t value = selected[src * 512 + tile];
      if (value > 1 || (src == tile / 64 && value != 0)) return false;
      bits |= static_cast<uint8_t>(value << src);
    }
    packed[tile] = bits;
  }
  return true;
}
'''


def decoder(source, arm, tuning=None):
    tuning_config(tuning)
    signature = '''__global__ void taco_ma4f_decode_consumer_kernel(const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* residual) {'''
    require(source.count(signature) == 1, 'MA6 requires the exact fused F4 kernel')
    original = source[source.index(signature):]
    changed = replace_once(original, 'taco_ma4f_decode_consumer_kernel', kernel_name('attn', tuning, arm))
    if arm == 'F4-layout':
        changed = replace_once(changed, 'blockIdx.x * 4 + threadIdx.x / 32', GROUP_MAPPING)
    elif arm == 'F4-mask':
        changed = replace_once(changed, 'const __nv_bfloat16* residual) {',
                               'const __nv_bfloat16* residual, const uint8_t* packed_mask) {')
        begin = changed.index('  bool selected[8];')
        end = changed.index('  if (!any_quant) {', begin)
        changed = changed[:begin] + '  const uint8_t source_mask = packed_mask[physical_tile];\n' + changed[end:]
        changed = replace_once(changed, '  if (!any_quant) {', '  if (source_mask == 0) {')
        require(changed.count('selected[src]') == 2, 'Unexpected F4 mixed source loads')
        changed = changed.replace('selected[src]', '((source_mask >> src) & 1) != 0')
    else:
        raise ValueError('Only MA6 candidate kernels are transformed')
    # Keep the original F4 entry verbatim for projection/reference provenance.
    return source + '\n' + changed


def runtime(source, arm, tuning=None):
    tuning_config(tuning)
    require('taco_ma4f_protocol_version() { return 1; }' in source,
            'MA6 requires frozen F4 native consumption')
    mask = arm == 'F4-mask'
    result = source + f'''
extern "C" int {decode_entry(arm)}(FluxTacoConfig c, const void* local,
    void* output, const void* bias, const void* residual,{" const void* packed_mask," if mask else ""} cudaStream_t stream) {{
  if (!c.enabled || !local || !output || !bias || !residual || !c.selected ||
      {"!packed_mask || " if mask else ""}!taco_ma2_exact_shape(c) || !c.deferred_reuse_barrier || c.allow_capture)
    return int(cudaErrorInvalidValue);
  {kernel_name('attn', tuning, arm)}<<<2048, 128, 0, stream>>>(c,
      static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output),
      static_cast<const __nv_bfloat16*>(bias), static_cast<const __nv_bfloat16*>(residual){", static_cast<const uint8_t*>(packed_mask)" if mask else ""});
  return int(cudaGetLastError());
}}
extern "C" int taco_ma6_protocol_version() {{ return 3; }}
extern "C" int taco_ma6_receiver_variant() {{ return {RECEIVER_VARIANTS[arm]}; }}
'''
    for field, value in BASELINE_TUNING.items():
        result += f'extern "C" int taco_ma6_{field}(int k) {{ return k == 512 || k == 2048 ? {value} : -2; }}\n'
    return result


def header(source, arm, tuning=None):
    tuning_config(tuning)
    return source + f'''\nextern "C" int {decode_entry(arm)}(FluxTacoConfig,
    const void*, void*, const void*, const void*,{" const void*," if arm == 'F4-mask' else ""} cudaStream_t);
'''


MASK_METHODS = '''  void ma6_bind_packed_mask(torch::Tensor selection_cpu, torch::Tensor selected) {
    std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
    TORCH_CHECK(!cr2_failed && !ma4_consumer_active && cr2_calls == 0 &&
                    cr2_initializations == 0 && ma6_mask_bindings == 0,
                "MA6 packed mask can only bind once before first submission");
    try {
      auto stream = c10::cuda::getCurrentCUDAStream();
      cudaStreamCaptureStatus status;
      CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
      TORCH_CHECK(stream == taco_stream && status == cudaStreamCaptureStatusNone &&
                      owned_taco.enabled && owned_taco.world == 8 && owned_taco.m == 2048 &&
                      owned_taco.n == 4096 && (cr2_k == 512 || cr2_k == 2048) &&
                      owned_taco.deferred_reuse_barrier && !owned_taco.allow_capture,
                  "MA6 mask initialization requires frozen eager instance creation stream");
      TORCH_CHECK(selection_cpu.device().is_cpu() && selection_cpu.scalar_type() == at::kByte &&
                      selection_cpu.is_contiguous() && selection_cpu.dim() == 2 &&
                      selection_cpu.size(0) == 8 && selection_cpu.size(1) == 512 &&
                      selected.is_cuda() && selected.device().index() == taco_stream.device_index() &&
                      selected.scalar_type() == at::kByte && selected.is_contiguous() &&
                      selected.dim() == 2 && selected.size(0) == 8 && selected.size(1) == 512 &&
                      selected.data_ptr() == owned_taco.selected,
                  "MA6 mask binding must use the immutable instance selection");
      // Validate against the actual producer selection once, outside all timed forwards.
      TORCH_CHECK(selection_cpu.equal(selected.cpu()), "MA6 CPU and producer selection differ");
      auto packed_cpu = torch::empty({512}, selection_cpu.options());
      TORCH_CHECK(taco_ma6_pack_mask(selection_cpu.data_ptr<uint8_t>(), packed_cpu.data_ptr<uint8_t>()),
                  "MA6 selection must be binary with local contributions in BF16");
      // Blocking setup copy, private allocation: neither CPU metadata nor Python
      // can alias the decoder's immutable CUDA mask. No per-forward repacking.
      ma6_packed_mask = packed_cpu.to(selected.device(), at::kByte, false, true);
      ma6_selected_owner = std::move(selected);
      ++ma6_mask_bindings;
    } catch (...) {
      cr2_failed = true;
      throw;
    }
  }

  std::vector<int64_t> ma6_mask_state() const {
    std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
    return {3, ma6_mask_bindings,
            ma6_packed_mask.defined() ? static_cast<int64_t>(reinterpret_cast<uintptr_t>(ma6_packed_mask.data_ptr())) : 0,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(owned_taco.selected))};
  }

'''


def wrapper(source, arm, tuning=None):
    tuning_config(tuning)
    if arm == 'F4-layout':
        return replace_once(source, 'taco_ma4f_decode_reduce(owned_taco,',
                            decode_entry(arm) + '(owned_taco,')
    require(arm == 'F4-mask', 'Unknown MA6 receiver wrapper')
    source = replace_once(source, '#include <mutex>', '#include <mutex>\n#include "gemm_rs/ma6_mask.h"')
    source = replace_once(source, '  torch::Tensor ma4_bias, ma4_residual;', '''  torch::Tensor ma4_bias, ma4_residual;
  torch::Tensor ma6_packed_mask, ma6_selected_owner;
  int64_t ma6_mask_bindings = 0;''')
    source = replace_once(source, '  torch::Tensor ma4_consumer_forward(', MASK_METHODS + '  torch::Tensor ma4_consumer_forward(')
    source = replace_once(source, '    TORCH_CHECK(!cr2_failed, "CR2 cannot retry a failed submission");',
                          '    TORCH_CHECK(!cr2_failed, "CR2 cannot retry a failed submission");\n'
                          '    TORCH_CHECK(ma6_mask_bindings == 1 && ma6_packed_mask.defined(), "MA6 packed mask was not bound");')
    source = replace_once(source, '      if (ma4_consumer_active) {', '''      if (ma4_consumer_active) {
        c10::cuda::CUDACachingAllocator::recordStream(ma6_packed_mask.storage().data_ptr(), stream);''')
    source = replace_once(source, '''taco_ma4f_decode_reduce(owned_taco,
            output_buffer.data_ptr(), output.data_ptr(), ma4_bias.data_ptr(), ma4_residual.data_ptr(), stream)''',
        '''taco_ma6_decode_mask(owned_taco,
            output_buffer.data_ptr(), output.data_ptr(), ma4_bias.data_ptr(), ma4_residual.data_ptr(),
            ma6_packed_mask.data_ptr(), stream)''')
    return replace_once(source, 'GemmRS::~GemmRS() { delete impl_; }', '''GemmRS::~GemmRS() { delete impl_; }
void GemmRS::ma6_bind_packed_mask(torch::Tensor selection_cpu, torch::Tensor selected) {
  FLUX_CHECK(impl_ != nullptr);
  impl_->ma6_bind_packed_mask(std::move(selection_cpu), std::move(selected));
}
std::vector<int64_t> GemmRS::ma6_mask_state() const {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->ma6_mask_state();
}''')


def wrapper_header(source):
    return replace_once(source, '  std::vector<int64_t> ma4_consumer_state() const;', '''  std::vector<int64_t> ma4_consumer_state() const;
  void ma6_bind_packed_mask(torch::Tensor, torch::Tensor);
  std::vector<int64_t> ma6_mask_state() const;''')


def pybind(source):
    return replace_once(source, '        .def("ma4_consumer_state", &GemmRSOpCls::ma4_consumer_state)', '''        .def("ma4_consumer_state", &GemmRSOpCls::ma4_consumer_state)
        .def("ma6_bind_packed_mask", &GemmRSOpCls::ma6_bind_packed_mask)
        .def("ma6_mask_state", &GemmRSOpCls::ma6_mask_state)''')


def frontend(source):
    return replace_once(source, '        self.option = cpp_mod.ReduceScatterOption()', '''        # Native packs and privately owns one immutable 512-byte mask per slot.
        self.op.ma6_bind_packed_mask(self.selection_cpu, self.selected)
        self.option = cpp_mod.ReduceScatterOption()''')


def sources(reference, arm='F4-mask', tuning=None):
    tuning = tuning_config(tuning)
    reference = Path(reference)
    require(arm in VARIANTS[1:], 'Only MA6 candidate sources are transformed')
    result = {name: transform((reference / name).read_text(), arm, tuning)
              for name, transform in ((RUNTIME, runtime), (HEADER, header),
                                      (WRAPPER, wrapper), (DECODER, decoder))}
    if arm == 'F4-mask':
        result.update({name:transform((reference / name).read_text()) for name,transform in
                       ((WRAPPER_HEADER, wrapper_header), (PYBIND, pybind), (FRONTEND, frontend))})
        result[MASK_HEADER] = PACK_MASK
    return result
