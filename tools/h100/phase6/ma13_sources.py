"""MA11 in both arms; candidate adds a resident-exponential forward only."""
from pathlib import Path

from common import HERE, replace_once, require
import ma11_sources as softmax

RUNTIME, HEADER = softmax.RUNTIME, softmax.HEADER
WRAPPER, DECODER = softmax.WRAPPER, softmax.DECODER
ZERO_COUNT_HEADER = softmax.ZERO_COUNT_HEADER
ZERO_COUNT_INCLUDE = softmax.ZERO_COUNT_INCLUDE
ZERO_COUNT_EXPORTS = softmax.ZERO_COUNT_EXPORTS
SOFTMAX_EXPORTS = softmax.NATIVE_EXPORTS
VERSION_SCRIPT, CUDA_CMAKE = softmax.VERSION_SCRIPT, softmax.CUDA_CMAKE
SOFTMAX_HEADER = 'src/gemm_rs/ma13_softmax.cuh'
SOFTMAX_TU = 'src/cuda/ma13_softmax.cu'
SOFTMAX_FLAGS = softmax.SOFTMAX_FLAGS
NATIVE_EXPORTS = ('ma13_softmax_protocol', 'ma13_softmax_forward_cuda')
VARIANTS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-resident')
BASELINE_TUNING = softmax.BASELINE_TUNING
ROLES, ROLE_K = softmax.ROLES, softmax.ROLE_K
DEFAULT_TUNING = softmax.DEFAULT_TUNING
tuning_config = softmax.tuning_config


def sources(reference, arm=VARIANTS[1]):
    require(arm in VARIANTS, 'Unknown MA13 source arm')
    # This dictionary is byte-for-byte the original MA11 candidate transform.
    # In particular, its backward kernel, flags and original native ABI remain.
    result = softmax.sources(Path(reference), softmax.VARIANTS[1])
    if arm == VARIANTS[1]:
        require('ma13_softmax' not in result[VERSION_SCRIPT], 'MA13 already installed')
        result[SOFTMAX_HEADER] = (HERE / 'ma13_softmax.cuh').read_text()
        result[SOFTMAX_TU] = '#include "gemm_rs/ma13_softmax.cuh"\n'
        cmake = replace_once(result[CUDA_CMAKE], 'set(LIB_FILES\n',
                             'set(LIB_FILES\n  ma13_softmax.cu\n')
        result[CUDA_CMAKE] = replace_once(cmake,
            'set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)',
            'set_source_files_properties(ma13_softmax.cu PROPERTIES\n'
            '  COMPILE_OPTIONS "' + ';'.join(SOFTMAX_FLAGS) + '")\n\n'
            'set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)')
        result[VERSION_SCRIPT] = replace_once(result[VERSION_SCRIPT], 'global:\n',
            'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return result


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MA13 arm')
    return softmax.kernel_name(role, tuning, softmax.VARIANTS[1])
