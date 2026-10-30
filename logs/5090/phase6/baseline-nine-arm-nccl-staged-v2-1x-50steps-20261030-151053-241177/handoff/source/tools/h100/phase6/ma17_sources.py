"""Identical original MA11 native sources in both direct-allgather arms.

MA17 submits the column forward AllGather directly through the already loaded
NCCL library on the current compute stream. Full GEMMs, the retained MA8 input,
row projections, softmax and zero-count source trees remain identical.
"""
from pathlib import Path
from common import require
import ma11_sources as baseline

RUNTIME, HEADER = baseline.RUNTIME, baseline.HEADER
WRAPPER, DECODER = baseline.WRAPPER, baseline.DECODER
ZERO_COUNT_HEADER = baseline.ZERO_COUNT_HEADER
ZERO_COUNT_INCLUDE = baseline.ZERO_COUNT_INCLUDE
ZERO_COUNT_EXPORTS = baseline.ZERO_COUNT_EXPORTS
SOFTMAX_EXPORTS = baseline.NATIVE_EXPORTS
VERSION_SCRIPT, CUDA_CMAKE = baseline.VERSION_SCRIPT, baseline.CUDA_CMAKE
NATIVE_EXPORTS = ()
VARIANTS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-directag')
BASELINE_TUNING = baseline.BASELINE_TUNING
ROLES, ROLE_K = baseline.ROLES, baseline.ROLE_K
DEFAULT_TUNING = baseline.DEFAULT_TUNING
tuning_config = baseline.tuning_config


def sources(reference, arm=VARIANTS[1]):
    require(arm in VARIANTS, 'Unknown MA17 source arm')
    return baseline.sources(Path(reference), baseline.VARIANTS[1])


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MA17 arm')
    return baseline.kernel_name(role, tuning, baseline.VARIANTS[1])
