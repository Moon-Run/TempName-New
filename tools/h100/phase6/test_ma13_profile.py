"""MA13 rejects stale forward paths, wrong geometry and changed dependencies."""
import copy
import unittest

import ma13_profile as profile
from test_ma11_profile import fixture as ma11_fixture, find


def fixture(candidate=False):
    events = ma11_fixture(True)
    for event in events:
        if event['cat'] == 'kernel':
            if profile.CONTROL_FORWARD_KERNEL in event['name']:
                if candidate:
                    event['name'] = profile.FORWARD_KERNEL
                event['args'].update(grid=[8192, 1, 1], block=[256 if candidate else 1024, 1, 1])
            elif profile.BACKWARD_KERNEL in event['name']:
                event['args'].update(grid=[8192, 1, 1], block=[128, 1, 1])
        if candidate and event['cat'] == 'user_annotation':
            event['name'] = event['name'].replace('ma11.attention.softmax.', 'ma13.attention.softmax.')
    return events


class ProfileTests(unittest.TestCase):
    def test_both_arms_keep_full_ma11_graph_and_backward(self):
        for candidate in (False, True):
            result = profile.trace_attention_softmax_records(fixture(candidate), profile.ARMS[int(candidate)])
            self.assertEqual(result['softmax_compute_kernel_calls'], 256)
            self.assertEqual(result['attention_view_replays'], 0)
            self.assertEqual(result['copy_slices_calls'], 4)
            self.assertTrue(result['original_ma11_backward'])
            self.assertEqual(result['forward_block'], 256 if candidate else 1024)

    def test_control_cannot_contain_candidate_markers(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            profile.trace_attention_softmax_records(fixture(True), profile.ARMS[0])

    def test_candidate_cannot_retain_old_forward_kernel(self):
        events = fixture(True)
        find(events, profile.FORWARD_KERNEL)['name'] = profile.CONTROL_FORWARD_KERNEL
        with self.assertRaisesRegex(ValueError, 'wrong softmax kernel'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_original_geometry_is_required_for_each_arm(self):
        for candidate in (False, True):
            events = fixture(candidate)
            token = profile.FORWARD_KERNEL if candidate else profile.CONTROL_FORWARD_KERNEL
            find(events, token)['args']['block'] = [512, 1, 1]
            with self.assertRaisesRegex(ValueError, 'geometry'):
                profile.trace_attention_softmax_records(events, profile.ARMS[int(candidate)])

    def test_wrong_owner_and_wrong_stream_are_rejected(self):
        for mutation, error in [('owner', 'exactly one'), ('stream', 'stream changed')]:
            events = fixture(True)
            value = find(events, profile.FORWARD_KERNEL)
            if mutation == 'owner':
                runtime = next(e for e in events if e['cat'] == 'cuda_runtime' and
                               e['args']['correlation'] == value['args']['correlation'])
                runtime['ts'] -= 10
            else:
                value['args']['stream'] = 8
            with self.assertRaisesRegex(ValueError, error):
                profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_extra_marker_work_rejected(self):
        events = fixture(True)
        value = copy.deepcopy(find(events, profile.FORWARD_KERNEL))
        value.update(cat='gpu_memset', name='Memset (Device)')
        events.append(value)
        with self.assertRaisesRegex(ValueError, 'extra copies'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_qk_dependency_and_backward_coverage_are_required(self):
        events = fixture(True)
        find(events, profile.FORWARD_KERNEL)['ts'] -= 30
        with self.assertRaisesRegex(ValueError, 'dependency order'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])
        events = fixture(True)
        events.remove(find(events, profile.BACKWARD_KERNEL))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_atten_fallback_rejected_in_both_arms(self):
        for candidate in (False, True):
            events = fixture(candidate)
            events.append(dict(cat='cpu_op', name='MaskedFillBackward0', ts=0, dur=1, tid=1))
            with self.assertRaisesRegex(ValueError, 'fell back'):
                profile.trace_attention_softmax_records(events, profile.ARMS[int(candidate)])

    def test_both_arms_retain_ma8_and_ma9(self):
        from test_ma89_profile import fixture as original, _shift
        for candidate in (False, True):
            events = original(True) + _shift(fixture(candidate), 1000000, 1000000)
            result = profile.trace_composed_records(events, profile.ARMS[int(candidate)], 0)
            self.assertEqual(result['activation_cache']['backward_allgathers'], 0)
            self.assertEqual(result['zero_count']['compute_kernel_calls'], 2)
            self.assertEqual(result['attention_softmax']['softmax_compute_kernel_calls'], 256)


if __name__ == '__main__':
    unittest.main()
