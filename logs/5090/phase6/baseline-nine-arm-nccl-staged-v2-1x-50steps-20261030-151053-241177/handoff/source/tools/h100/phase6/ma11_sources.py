"""Register MA89 in both arms and causal softmax only in the candidate.

The control is the exact MA9 native transform of F4, accompanied at runtime by
the unchanged MA8-v2 activation adapter. Row projections remain untouched.
"""
from pathlib import Path
from common import HERE, replace_once, require
import ma9_sources as zero_count

RUNTIME, HEADER = zero_count.RUNTIME, zero_count.HEADER
WRAPPER, DECODER = zero_count.WRAPPER, zero_count.DECODER
ZERO_COUNT_HEADER = zero_count.ZERO_COUNT_HEADER
ZERO_COUNT_INCLUDE = zero_count.ZERO_COUNT_INCLUDE
ZERO_COUNT_EXPORTS = zero_count.NATIVE_EXPORTS
VERSION_SCRIPT = zero_count.VERSION_SCRIPT
SOFTMAX_HEADER = 'src/gemm_rs/ma11_softmax.cuh'
SOFTMAX_TU = 'src/cuda/ma11_softmax.cu'
SOFTMAX_LICENSE = 'src/gemm_rs/ma11_pytorch_LICENSE.txt'
CUDA_CMAKE = 'src/cuda/CMakeLists.txt'
SOFTMAX_INCLUDE = '#include "gemm_rs/ma11_softmax.cuh"\n'
SOFTMAX_FLAGS = ('--ftz=false', '--prec-div=true', '--prec-sqrt=true')
NATIVE_EXPORTS = ('ma11_softmax_protocol', 'ma11_softmax_forward_cuda',
                  'ma11_softmax_backward_cuda')
VARIANTS = ('F4-cache-zcount', 'F4-cache-zcount-csoftmax')
BASELINE_TUNING = zero_count.BASELINE_TUNING
ROLES, ROLE_K = zero_count.ROLES, zero_count.ROLE_K
DEFAULT_TUNING = zero_count.DEFAULT_TUNING
tuning_config = zero_count.tuning_config


def sources(reference, arm=VARIANTS[1]):
    require(arm in VARIANTS, 'Unknown MA11 source arm')
    reference = Path(reference)
    result = zero_count.sources(reference)
    if arm == VARIANTS[1]:
        require('ma11_softmax' not in result[RUNTIME] and
                'ma11_softmax' not in result[VERSION_SCRIPT],
                'MA11 causal softmax helper is already installed')
        result[SOFTMAX_HEADER] = (HERE / 'ma11_softmax.cuh').read_text()
        result[SOFTMAX_TU] = SOFTMAX_INCLUDE
        result[SOFTMAX_LICENSE] = (HERE / 'ma11_pytorch_LICENSE.txt').read_text()
        cmake = (reference / CUDA_CMAKE).read_text()
        cmake = replace_once(cmake, 'set(LIB_FILES\n', 'set(LIB_FILES\n  ma11_softmax.cu\n')
        # Source COMPILE_OPTIONS follow target/global options. Only this new TU
        # disables FTZ and approximate division/sqrt; original Flux kernels keep
        # their original compilation settings and source bytes.
        cmake = replace_once(cmake, 'set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)',
            'set_source_files_properties(ma11_softmax.cu PROPERTIES\n'
            '  COMPILE_OPTIONS "' + ';'.join(SOFTMAX_FLAGS) + '")\n\n'
            'set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)')
        result[CUDA_CMAKE] = cmake
        result[VERSION_SCRIPT] = replace_once(result[VERSION_SCRIPT], 'global:\n',
            'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return result


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MA11 arm')
    return zero_count.kernel_name(role, tuning, 'F4-zcount')
