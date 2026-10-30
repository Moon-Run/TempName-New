"""Exact MA11 control; candidate adds only master-to-model publication APIs."""
from pathlib import Path
from common import HERE, replace_once, require
import ma11_sources as baseline

RUNTIME, HEADER = baseline.RUNTIME, baseline.HEADER
WRAPPER, DECODER = baseline.WRAPPER, baseline.DECODER
ZERO_COUNT_HEADER = baseline.ZERO_COUNT_HEADER
ZERO_COUNT_INCLUDE = baseline.ZERO_COUNT_INCLUDE
ZERO_COUNT_EXPORTS = baseline.ZERO_COUNT_EXPORTS
SOFTMAX_EXPORTS = baseline.NATIVE_EXPORTS
VERSION_SCRIPT, CUDA_CMAKE = baseline.VERSION_SCRIPT, baseline.CUDA_CMAKE
COPY_TU = 'src/cuda/ma19_param_copy.cu'
COPY_FLAGS = ('--ftz=false', '--prec-div=true', '--prec-sqrt=true')
NATIVE_EXPORTS = ('ma19_copy_protocol', 'ma19_copy_chunk_size', 'ma19_copy_cuda')
VARIANTS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-pcopy')
BASELINE_TUNING = baseline.BASELINE_TUNING
ROLES, ROLE_K = baseline.ROLES, baseline.ROLE_K
DEFAULT_TUNING = baseline.DEFAULT_TUNING
tuning_config = baseline.tuning_config


def sources(reference, arm=VARIANTS[1]):
    require(arm in VARIANTS, 'Unknown MA19 source arm')
    result = baseline.sources(Path(reference), baseline.VARIANTS[1])
    if arm == VARIANTS[1]:
        result[COPY_TU] = (HERE / 'ma19_param_copy.cu').read_text()
        result[CUDA_CMAKE] = replace_once(result[CUDA_CMAKE], 'set(LIB_FILES\n',
                                        'set(LIB_FILES\n  ma19_param_copy.cu\n')
        result[CUDA_CMAKE] = replace_once(result[CUDA_CMAKE],
            'set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)',
            'set_source_files_properties(ma19_param_copy.cu PROPERTIES\n'
            '  COMPILE_OPTIONS "' + ';'.join(COPY_FLAGS) + '")\n\n'
            'set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)')
        result[VERSION_SCRIPT] = replace_once(result[VERSION_SCRIPT], 'global:\n',
            'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return result


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MA19 arm')
    return baseline.kernel_name(role, tuning, baseline.VARIANTS[1])
