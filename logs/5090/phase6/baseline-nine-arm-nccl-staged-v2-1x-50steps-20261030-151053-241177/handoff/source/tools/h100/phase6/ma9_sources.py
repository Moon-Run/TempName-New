"""CR3/F4 control plus a candidate-only optimizer zero-count helper.

All existing projection code remains byte identical. The candidate adds one
runtime include, its helper header and explicit dynamic exports only.
"""
from pathlib import Path
from common import HERE, replace_once, require
import ma4_f_sources as f4

RUNTIME, HEADER, WRAPPER, DECODER = f4.RUNTIME, f4.HEADER, f4.WRAPPER, f4.DECODER
VARIANTS = ('F4', 'F4-zcount')
BASELINE_TUNING = dict(decoder_threads=128, decoder_blocks=2048, decoder_rows=1)
ROLES = ('attn', 'mlp')
ROLE_K = dict(attn=512, mlp=2048)
DEFAULT_TUNING = {role: dict(BASELINE_TUNING) for role in ROLES}
ZERO_COUNT_HEADER = 'src/gemm_rs/ma9_zero_count.cuh'
ZERO_COUNT_INCLUDE = '\n#include "gemm_rs/ma9_zero_count.cuh"\n'
VERSION_SCRIPT = 'src/cuda/version.ld'
NATIVE_EXPORTS = ('ma9_zero_count_protocol', 'ma9_zero_count_chunk_size', 'ma9_zero_count_cuda')


def sources(reference):
    reference = Path(reference)
    source = (reference / RUNTIME).read_text()
    require('ma9_zero_count' not in source, 'MA9 helper is already installed')
    exports = (reference / VERSION_SCRIPT).read_text()
    require('ma9_zero_count' not in exports, 'MA9 exports are already installed')
    exports = replace_once(exports, 'global:\n',
                           'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return {RUNTIME: source + ZERO_COUNT_INCLUDE,
            ZERO_COUNT_HEADER: (HERE / 'ma9_zero_count.cuh').read_text(),
            VERSION_SCRIPT: exports}


def tuning_config(tuning=None):
    tuning = DEFAULT_TUNING if tuning is None else tuning
    require(isinstance(tuning, dict) and set(tuning) == set(ROLES),
            'MA9 tuning requires attn and mlp')
    for role in ROLES:
        item = tuning[role]
        require(isinstance(item, dict) and set(item) == set(BASELINE_TUNING)
                and all(type(value) is int for value in item.values()) and item == BASELINE_TUNING,
                'MA9 fixes the original F4 decoder geometry: ' + role)
    return {role: dict(BASELINE_TUNING) for role in ROLES}


def kernel_name(role, tuning=None, arm='F4'):
    tuning_config(tuning)
    require(role in ROLES and arm in VARIANTS, 'Unknown MA9 role or arm')
    return 'taco_ma4f_decode_consumer_kernel'
