"""MA8-v2 cache in both arms; the combined arm adds exact MA9 native sources.

Cache retention is Python autograd only. The control therefore uses registered
F4 native bytes, and the combined native transform is delegated unchanged to MA9.
"""
from common import require
import ma9_sources as zero_count

RUNTIME, HEADER = zero_count.RUNTIME, zero_count.HEADER
WRAPPER, DECODER = zero_count.WRAPPER, zero_count.DECODER
ZERO_COUNT_HEADER = zero_count.ZERO_COUNT_HEADER
ZERO_COUNT_INCLUDE = zero_count.ZERO_COUNT_INCLUDE
VERSION_SCRIPT = zero_count.VERSION_SCRIPT
NATIVE_EXPORTS = zero_count.NATIVE_EXPORTS
VARIANTS = ('F4-cache', 'F4-cache-zcount')
BASELINE_TUNING = zero_count.BASELINE_TUNING
ROLES, ROLE_K = zero_count.ROLES, zero_count.ROLE_K
DEFAULT_TUNING = zero_count.DEFAULT_TUNING
tuning_config = zero_count.tuning_config
sources = zero_count.sources


def kernel_name(role, tuning=None, arm='F4-cache'):
    require(arm in VARIANTS, 'Unknown MA8+9 arm')
    return zero_count.kernel_name(role, tuning, 'F4' if arm == 'F4-cache' else 'F4-zcount')
