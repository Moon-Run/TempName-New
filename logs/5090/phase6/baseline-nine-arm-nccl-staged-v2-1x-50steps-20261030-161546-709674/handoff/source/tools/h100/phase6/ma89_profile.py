"""Compose the unchanged MA8-v2 cache and MA9 zero-count trace gates.

Both arms must execute the original raw cache implementation. Only the
optimizer statistic differs; MA9 kernel names and collective checks remain
those of the existing MA9 parser.
"""
import ma8_profile
import ma9_profile

ARMS = ('F4-cache', 'F4-cache-zcount')
ZERO_COUNT_ARMS = {'F4-cache': 'F4', 'F4-cache-zcount': 'F4-zcount'}


def _arm(arm):
    if arm not in ARMS:
        raise ValueError('MA89 accepts only F4-cache and F4-cache-zcount')


def trace_activation_cache_records(events, arm):
    _arm(arm)
    return ma8_profile.trace_activation_cache_records(events, 'F4-cache')


def trace_row_backward_records(events, arm):
    _arm(arm)
    return ma8_profile.trace_row_backward_records(events, 'F4-cache')


def trace_zero_count_records(events, arm, rank=None):
    _arm(arm)
    source_arm = ZERO_COUNT_ARMS[arm]
    result = ma9_profile.trace_zero_count_records(events, source_arm, rank)
    return dict(result, arm=arm, optimizer_source_arm=source_arm)


def trace_composed_records(events, arm, rank=None):
    _arm(arm)
    return dict(activation_cache=trace_activation_cache_records(events, arm),
                zero_count=trace_zero_count_records(events, arm, rank),
                row_backward=trace_row_backward_records(events, arm))
