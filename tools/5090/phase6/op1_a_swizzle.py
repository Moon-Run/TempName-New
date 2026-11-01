"""Isolated GemmOnly CTA permutation; no tensor or packet permutation."""
import hashlib
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
HEADER = 'src/comm_none/op1_a_swizzle.hpp'
PROBE = 'src/comm_none/op1_a_swizzle_probe.cuh'
KERNEL = 'src/comm_none/gemm_v2_comm_none.hpp'
EXPORTS = ('op1_a_sorted_version', 'op1_a_sorted_mapping')
TYPE_NAME = 'Op1AInterleavedSwizzle'
REGISTRATION_INCLUDE = '\n#include "comm_none/op1_a_swizzle_probe.cuh"\n'


def mapping(rank):
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('OP1 swizzle requires rank 0..7')
    return [((index % 8 + rank) % 8 * 2 + index // 8 // 32,
             index // 8 % 32) for index in range(512)]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('OP1 swizzle source anchor changed: ' + old)
    return text.replace(old, new, 1)


def overrides(source_root):
    source_root = Path(source_root)
    source = replace_once((source_root/KERNEL).read_text(), '#pragma once',
                          '#pragma once\n#include "comm_none/op1_a_swizzle.hpp"')
    source = replace_once(source,
        '    auto params = this->default_kernel_params();\n'
        '    return this->default_gemm_kernel(params);',
        '    auto params = this->default_kernel_params();\n'
        '    using AlignmentC = cute::Int<128 / cute::sizeof_bits_v<typename Base::ElementCNonVoid>>;\n'
        '    using SortedParams = gemm_v2_impl::KernelParams<\n'
        '        cutlass::gemm::threadblock::Op1AInterleavedSwizzle, AlignmentC, decltype(params.evt())>;\n'
        '    return this->default_gemm_kernel(SortedParams{});')
    version = replace_once((source_root/'src/cuda/version.ld').read_text(), 'global:',
                           'global:\n    op1_a_sorted_*;')
    return {KERNEL: source, HEADER: (HERE/'op1_a_swizzle.hpp').read_text(),
            PROBE: (HERE/'op1_a_swizzle_probe.cuh').read_text(), 'src/cuda/version.ld': version}


def registration_source(original):
    """Expose the actual scheduler in the GPU kernel name used by the audit.

    Flux's generated wrapper otherwise hides all inner CUTLASS template
    arguments. This renames only that wrapper and its single device reference.
    """
    names = re.findall(r'struct (flux_\w+): public GemmKernel_0 \{\};', original)
    if len(names) != 1 or original.count(names[0]) != 2 or TYPE_NAME in original:
        raise ValueError('Expected one untouched Flux kernel registration')
    return original.replace(names[0], names[0] + '_' + TYPE_NAME) + REGISTRATION_INCLUDE


def original_registration(transformed):
    if not transformed.endswith(REGISTRATION_INCLUDE):
        raise ValueError('Missing OP1 actual-kernel mapping export')
    original = transformed[:-len(REGISTRATION_INCLUDE)].replace('_' + TYPE_NAME, '')
    if registration_source(original) != transformed:
        raise ValueError('OP1 generated kernel registration changed')
    return original


def identity():
    names = ('op1_a_swizzle.py', 'op1_a_swizzle.hpp', 'op1_a_swizzle_probe.cuh')
    return dict(protocol='rtx5090-op1-a-gemmonly-cta-interleaved-v1',
                kernel_type=TYPE_NAME, gemm_schedule='StreamkSK',
                may_change_streamk_reduction_assignment=True,
                floating_point_reduction_assignment_preserved=False,
                arithmetic_reference='original_Flux_GemmOnly_with_changed_StreamK_tile_assignment',
                historical_Flux_StreamkSK_reused=True,
                packet_order_unchanged=True, rank_scope='host_kernel_parameters',
                shape=[2048, 4096], k_local=[512, 2048], tile=[128, 128, 32],
                source_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names})
