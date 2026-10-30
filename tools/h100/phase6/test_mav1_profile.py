"""Native context tokens cannot hide missing work, fallback or wrong dependencies."""
import copy
import unittest

import mav1_profile as profile
from test_ma11_profile import fixture as attention_fixture
from test_ma89_profile import fixture as base_fixture, _shift


def fixture(native=True):
    events = base_fixture(True) + _shift(attention_fixture(True), 1000000, 1000000)
    if not native:
        return events
    original = {v: k for k, v in profile.ALIASES.items()}
    for event in events:
        if event.get('cat') != 'cpu_op' or event['name'] not in original:
            continue
        kind, phase = original[event['name']]
        seq = event['args']['Sequence number']
        if kind == 'column':
            plan, call = seq % 65, seq // 65 + 41
        else:
            # _shift offsets sequence ids as well; reduce to original 256 rows.
            seq %= 10000
            plan, call = 64 + seq % 64, seq // 64 + 41
        event.update(ph='X', cat='user_annotation',
                     name=f'mav1.{kind}.{phase}.{plan}.{call}')
        event['args']['External id'] = -100000 - plan * 10 - call
    return events


class ProfileTests(unittest.TestCase):
    def test_whole_step_extra_or_missing_collectives_rejected(self):
        events = [dict(cat='kernel', name='nccl_AllGather') for _ in range(524)]
        events += [dict(cat='kernel', name='nccl_ReduceScatter') for _ in range(260)]
        events += [dict(cat='kernel', name='CudaIpcBarrierAllKernel') for _ in range(256)]
        self.assertEqual(profile.fullstep_collective_inventory(events)['AllGather'], 524)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            profile.fullstep_collective_inventory(events + [dict(cat='kernel', name='nccl_AllGather')])

    def test_both_arms_retain_all_original_ma11_math_and_communication(self):
        for native in (False, True):
            events = fixture(native)
            before = copy.deepcopy(events)
            result = profile.trace_composed_records(events, profile.ARMS[int(native)], 0)
            self.assertEqual(events, before)
            self.assertEqual(result['activation_cache']['backward_allgathers'], 0)
            self.assertEqual(result['activation_cache']['retained_forward_allgathers'], 256)
            self.assertEqual(result['activation_cache']['retained_reduce_scatters'], 256)
            self.assertEqual(result['zero_count']['compute_kernel_calls'], 2)
            self.assertEqual(result['attention_softmax']['softmax_compute_kernel_calls'], 256)
            self.assertEqual(result['native_linear']['installed'], native)

    def test_missing_or_duplicate_native_phase_rejected(self):
        for duplicate in (False, True):
            events = fixture()
            marker = next(e for e in events if e['name'].startswith('mav1.'))
            if duplicate:
                events.append(copy.deepcopy(marker))
            else:
                events.remove(marker)
            with self.assertRaisesRegex(ValueError, '1024'):
                profile.validated_event_view(events, profile.ARMS[1])

    def test_saved_context_pair_cannot_be_swapped_or_reused(self):
        for suffix in ('999', '40'):
            events = fixture()
            marker = next(e for e in events if e['name'].startswith('mav1.column.backward.'))
            marker['name'] = marker['name'].rsplit('.', 1)[0] + '.' + suffix
            with self.assertRaisesRegex(ValueError, 'coverage|Unpaired'):
                profile.validated_event_view(events, profile.ARMS[1])

    def test_original_python_fallback_rejected(self):
        events = fixture()
        events.append(dict(cat='cpu_op', name='_MA8CachedRawLinearBackward'))
        with self.assertRaisesRegex(ValueError, 'Python'):
            profile.validated_event_view(events, profile.ARMS[1])

    def test_native_marker_cannot_contaminate_control(self):
        with self.assertRaisesRegex(ValueError, 'control'):
            profile.validated_event_view(fixture(), profile.ARMS[0])

    def test_missing_launch_wrong_stream_and_removed_collective_rejected(self):
        for mutation in ('launch', 'stream', 'collective'):
            events = fixture()
            kernel = next(e for e in events if e.get('cat') == 'kernel' and 'ReduceScatter' in e['name'])
            if mutation == 'stream':
                kernel['args']['stream'] = 7
            elif mutation == 'collective':
                events.remove(kernel)
            else:
                correlation = kernel['args']['correlation']
                events[:] = [e for e in events if not (e.get('cat') == 'cuda_runtime'
                              and e.get('args', {}).get('correlation') == correlation)]
            with self.assertRaises(ValueError):
                profile.trace_composed_records(events, profile.ARMS[1], 0)


if __name__ == '__main__':
    unittest.main()
