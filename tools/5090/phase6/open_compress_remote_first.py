"""Actual GEMM CTA mapping: seven contiguous remote partitions, then local."""
import hashlib
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
HEADER = 'src/comm_none/open_compress_remote_first.hpp'
PROBE = 'src/comm_none/open_compress_remote_first_probe.cuh'
KERNEL = 'src/comm_none/gemm_v2_comm_none.hpp'
TYPE_NAME = 'OpenCompressRemoteFirstSwizzle'
EXPORTS = ('oc_remote_first_version', 'oc_remote_first_mapping')
REGISTRATION_INCLUDE = '\n#include "comm_none/open_compress_remote_first_probe.cuh"\n'


def mapping(rank):
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('Remote-first requires rank 0..7')
    return [(((rank + 1 + slot // 64) % 8) * 2 + (slot % 64) // 32,
             slot % 32) for slot in range(512)]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Remote-first source anchor changed: ' + old)
    return text.replace(old, new, 1)


def overrides(source_root):
    source_root = Path(source_root)
    source = replace_once((source_root/KERNEL).read_text(), '#pragma once',
                          '#pragma once\n#include "comm_none/open_compress_remote_first.hpp"')
    source = replace_once(source,
        '    auto params = this->default_kernel_params();\n'
        '    return this->default_gemm_kernel(params);',
        '    auto params = this->default_kernel_params();\n'
        '    using AlignmentC = cute::Int<128 / cute::sizeof_bits_v<typename Base::ElementCNonVoid>>;\n'
        '    using SortedParams = gemm_v2_impl::KernelParams<\n'
        '        cutlass::gemm::threadblock::OpenCompressRemoteFirstSwizzle, AlignmentC, decltype(params.evt())>;\n'
        '    return this->default_gemm_kernel(SortedParams{});')
    version = replace_once((source_root/'src/cuda/version.ld').read_text(), 'global:',
                           'global:\n    oc_remote_first_*;')
    return {KERNEL: source, HEADER: (HERE/'open_compress_remote_first.hpp').read_text(),
            PROBE: (HERE/'open_compress_remote_first_probe.cuh').read_text(), 'src/cuda/version.ld': version}


def registration_source(original):
    names = re.findall(r'struct (flux_\w+): public GemmKernel_0 \{\};', original)
    if len(names) != 1 or original.count(names[0]) != 2 or TYPE_NAME in original:
        raise ValueError('Expected one untouched Flux kernel registration')
    return original.replace(names[0], names[0] + '_' + TYPE_NAME) + REGISTRATION_INCLUDE


def original_registration(transformed):
    if not transformed.endswith(REGISTRATION_INCLUDE):
        raise ValueError('Missing remote-first actual-kernel mapping export')
    original = transformed[:-len(REGISTRATION_INCLUDE)].replace('_' + TYPE_NAME, '')
    if registration_source(original) != transformed:
        raise ValueError('Remote-first generated kernel registration changed')
    return original


def identity():
    names = ('open_compress_remote_first.py', 'open_compress_remote_first.hpp',
             'open_compress_remote_first_probe.cuh')
    return dict(protocol='rtx5090-open-compress-remote-first-v1', kernel_type=TYPE_NAME,
        source_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names},
        mapping='destination=(rank+1+slot//64)%8; within=slot%64',
        remote_slots=448, local_slots=64, packet_order_unchanged=True,
        may_change_streamk_reduction_assignment=True, actual_completion_order_guaranteed=False)
