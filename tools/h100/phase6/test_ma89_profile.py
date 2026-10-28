"""Combined traces must contain both real mechanisms and the correct control."""
import copy
import unittest

import ma89_profile as profile
from test_ma8_profile import fixture as cache_fixture
from test_ma9_profile import fixture as zero_fixture


def _shift(events, time, correlation):
    events = copy.deepcopy(events)
    for event in events:
        if 'ts' in event:
            event['ts'] += time
        args = event.get('args', {})
        if 'correlation' in args:
            args['correlation'] += correlation
    return events


def fixture(candidate=False, rank=0, cache=True):
    return (cache_fixture(cache) + _shift(cache_fixture(row=True), 300000, 100000) +
            _shift(zero_fixture(candidate, rank), 600000, 200000))


class ProfileTests(unittest.TestCase):
    def test_control_retains_cache_and_original_zero_statistic(self):
        value = profile.trace_composed_records(fixture(), 'F4-cache', 0)
        self.assertEqual(value['activation_cache']['backward_allgathers'], 0)
        self.assertEqual(value['activation_cache']['column_calls'], 256)
        self.assertEqual(value['activation_cache']['original_vocabulary_heads']['calls'], 4)
        self.assertEqual(value['zero_count']['original_count_nonzero_calls'], 389)
        self.assertEqual(value['zero_count']['compute_kernel_calls'], 1945)
        self.assertEqual(value['zero_count']['optimizer_source_arm'], 'F4')
        self.assertEqual(value['row_backward']['attn']['backward_calls'], 128)

    def test_candidate_retains_cache_and_exact_ma9_kernels(self):
        value = profile.trace_composed_records(fixture(True), 'F4-cache-zcount', 0)
        self.assertEqual(value['activation_cache']['backward_allgathers'], 0)
        self.assertEqual(value['zero_count']['compute_kernel_calls'], 2)
        self.assertEqual(value['zero_count']['original_count_nonzero_calls'], 0)
        self.assertEqual(value['zero_count']['original_allreduce_calls'], 1)
        self.assertEqual(value['zero_count']['optimizer_source_arm'], 'F4-zcount')

    def test_remote_rank_filter_remains_original(self):
        value = profile.trace_composed_records(fixture(rank=7), 'F4-cache', 7)
        self.assertEqual(value['zero_count']['eligible_parameter_count'], 194)
        self.assertEqual(value['zero_count']['compute_kernel_calls'], 970)

    def test_original_f4_cannot_masquerade_as_common_cache(self):
        for arm in profile.ARMS:
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, 'Wrong raw column arm'):
                profile.trace_composed_records(fixture(arm.endswith('zcount'), cache=False), arm, 0)

    def test_candidate_with_original_zero_statistic_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fell back'):
            profile.trace_composed_records(fixture(), 'F4-cache-zcount', 0)

    def test_control_with_candidate_zero_statistic_rejected(self):
        with self.assertRaisesRegex(ValueError, 'wrappers or kernels'):
            profile.trace_composed_records(fixture(True), 'F4-cache', 0)

    def test_unrequested_arms_rejected(self):
        for arm in ('F4', 'F4-zcount', 'F4-cache-extra'):
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, 'only F4-cache'):
                profile.trace_composed_records([], arm)

    def test_old_cache_wrapper_rejected(self):
        events = fixture(True)
        event = next(e for e in events if e['name'] == '_MA8CachedRawLinear')
        event['name'] = 'MA8RetainedActivationLinear'
        with self.assertRaisesRegex(ValueError, 'old per-call wrappers'):
            profile.trace_composed_records(events, 'F4-cache-zcount', 0)

    def test_wrong_zero_count_kernel_namespace_rejected(self):
        events = fixture(True)
        event = next(e for e in events if e['cat'] == 'kernel' and 'ma9_zero_count_chunks_kernel' in e['name'])
        event['name'] = event['name'].replace('ma9_zero_count', 'ma89_zero_count')
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            profile.trace_composed_records(events, 'F4-cache-zcount', 0)

    def test_original_row_projection_still_required(self):
        events = fixture(True)
        events.remove(next(e for e in events if e['name'] == 'ConsumerGemmRSFunctionBackward'))
        with self.assertRaisesRegex(ValueError, 'row projection autograd coverage'):
            profile.trace_composed_records(events, 'F4-cache-zcount', 0)


if __name__ == '__main__':
    unittest.main()
