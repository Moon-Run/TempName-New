"""Direct H128 wire addressing; retain pinned upstream quantization arithmetic."""
import hashlib
from pathlib import Path
from rtx5090_loader import HERE, H100, load

base = load('sources')
upstream = base.basic_quant_sources.codec_build


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError('OP1 ABC upstream source anchor changed: ' + before)
    return text.replace(before, after, 1)


def encoder_source():
    text = upstream.kernel_source()
    text = text[text.index('template<typename T, int GROUP_SIZE>'):text.index(
        'template<typename T, int GROUP_SIZE>\n__global__ void __launch_bounds__(GROUP_SIZE) fused_ash_decompress_kernel')]
    text = replace(text, 'fused_ash_compress_kernel(', 'op1_abc_encode_kernel(')
    text = replace(text, '__nv_fp8_interpretation_t fp8_type)',
                   '__nv_fp8_interpretation_t fp8_type, int rank, int begin, int raw_groups)')
    text = replace(text, 'int chunk_id = blockIdx.y;',
                   'int chunk_id = blockIdx.y;\n    int destination = chunk_id + int(chunk_id >= rank);')
    text = replace(text, 'chunk_id * in_chunk_stride_bytes', 'destination * in_chunk_stride_bytes')
    text = replace(text, 'char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes;', '''char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes
        + (group_id < raw_groups ? size_t(group_id) * 256
                                 : size_t(raw_groups) * 256 + size_t(group_id - raw_groups) * 136);
    if (group_id < raw_groups) {
        reinterpret_cast<__nv_bfloat16*>(out_base)[tid] =
            reinterpret_cast<const __nv_bfloat16*>(in_base)[(begin + group_id) * GROUP_SIZE + tid];
        return;
    }''')
    text = replace(text, '(float*)(out_base + orgChunkCount)', '(float*)(out_base + 128)')
    text = replace(text, '(float*)(out_base + orgChunkCount + scale_chunk_stride_bytes)',
                   '(float*)(out_base + 132)')
    text = replace(text, 'int start_idx = group_id * GROUP_SIZE;',
                   'int start_idx = (begin + group_id) * GROUP_SIZE;')
    text = replace(text, 'chunk_ada_scales[group_id] = ada_scale;', 'chunk_ada_scales[0] = ada_scale;')
    text = replace(text, 'chunk_q_scales[group_id] = quant_scale;', 'chunk_q_scales[0] = quant_scale;')
    text = replace(text, 'reinterpret_cast<unsigned int*>(out_base)[idx / 4]',
                   'reinterpret_cast<unsigned int*>(out_base)[tid / 4]')
    text = replace(text, '((__nv_fp8_storage_t*)out_base)[idx]',
                   '((__nv_fp8_storage_t*)out_base)[tid]')
    return text


def source_files():
    return {
        'baseline_codec.cu': (H100/'baseline_codec.cu').read_text(),
        'basic_quant_receiver.cuh': (H100/'basic_quant_receiver.cuh').read_text(),
        'upstream_kernels.cuh': upstream.kernel_source(),
        'op1_abc_encoder.cuh': encoder_source(),
        'op1_abc_codec.cu': (HERE/'op1_abc_codec.cu').read_text(),
        'op1_abc.cu': '#include "baseline_codec.cu"\n#include "basic_quant_receiver.cuh"\n'
                       '#include "op1_abc_encoder.cuh"\n#include "op1_abc_codec.cu"\n',
    }


def identity():
    return dict(protocol='rtx5090-op1-abc-direct-h128-v1', upstream=upstream.source_identity(),
                source_sha256={name: hashlib.sha256(text.encode()).hexdigest()
                               for name, text in source_files().items()},
                recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
