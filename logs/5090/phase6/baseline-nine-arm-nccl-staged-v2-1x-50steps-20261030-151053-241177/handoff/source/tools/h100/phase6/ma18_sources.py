"""Exact MA11 control; candidate adds only fixed-size IPC copy submission APIs."""
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
IPC_TU = 'src/cuda/ma18_ipc_allgather.cu'
NATIVE_EXPORTS = ('ma18_ipc_protocol', 'ma18_ipc_stage_cuda', 'ma18_ipc_pull_cuda')
VARIANTS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-ipcag')
BASELINE_TUNING = baseline.BASELINE_TUNING
ROLES, ROLE_K = baseline.ROLES, baseline.ROLE_K
DEFAULT_TUNING = baseline.DEFAULT_TUNING
tuning_config = baseline.tuning_config


def sources(reference, arm=VARIANTS[1]):
    require(arm in VARIANTS, 'Unknown MA18 source arm')
    result = baseline.sources(Path(reference), baseline.VARIANTS[1])
    if arm == VARIANTS[1]:
        result[IPC_TU] = (HERE / 'ma18_ipc_allgather.cu').read_text()
        result[CUDA_CMAKE] = replace_once(result[CUDA_CMAKE], 'set(LIB_FILES\n',
                                         'set(LIB_FILES\n  ma18_ipc_allgather.cu\n')
        result[VERSION_SCRIPT] = replace_once(result[VERSION_SCRIPT], 'global:\n',
            'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return result


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MA18 arm')
    return baseline.kernel_name(role, tuning, baseline.VARIANTS[1])
