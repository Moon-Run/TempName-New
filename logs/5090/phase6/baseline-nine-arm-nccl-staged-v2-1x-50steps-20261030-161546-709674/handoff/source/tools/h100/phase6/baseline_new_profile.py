"""Reuse original single-node, launch-correlated path checks for all five arms."""
import ma4_f_fullstep
import ma4_f_bench
import ma8_profile
import ma9_profile
import ma11_profile
from tp8_dp2_profile import projection_records
from baseline_new_protocol import ARMS, SELECTIVE, NATIVE_ARM, require


def analyze(events, arm, rank):
    require(arm in ARMS and 0 <= rank < 8,'Unknown profile identity')
    if arm in SELECTIVE:
        result=dict(row_forward=ma4_f_fullstep.trace_projection_records(events,'F4',ma4_f_bench.MECHANISM))
        if arm=='cr3':
            result.update(activation_cache=ma8_profile.trace_activation_cache_records(events,'F4'),
                          row_backward=ma8_profile.trace_row_backward_records(events,'F4'),
                          zero_count=ma9_profile.trace_zero_count_records(events,'F4',rank),
                          attention_softmax=ma11_profile.trace_attention_softmax_records(events,'F4-cache-zcount'))
        else:result.update(ma11_profile.trace_composed_records(events,NATIVE_ARM[arm],rank))
    else:result=dict(row_forward=projection_records(events,arm,expected_calls=128))
    return dict(arm=arm,rank=rank,execution_validated=True,**result)
