"""Execution evidence must distinguish binary and epoch barriers in both scopes."""
import unittest

import ma14_bench as bench
import ma14_fullstep as full
import ma14_profile as profile
from test_ma12_profile import fixture as binary_fixture, find
from test_ma11_profile import fixture as softmax_fixture
from test_ma89_profile import fixture as compute_fixture, _shift


def fixture(candidate=False, full_model=False):
    events = binary_fixture(False, full_model)
    if candidate:
        for event in events:
            if event['cat'] == 'kernel' and 'CudaIpcBarrierAllKernel' in event['name']:
                event['name'] = 'taco_ma14_epoch_barrier_kernel'
    return events


class ProfileTests(unittest.TestCase):
    def test_operator_and_fullstep_paths(self):
        for candidate in (False, True):
            arm = bench.ARMS[int(candidate)]
            result = bench.trace_path(fixture(candidate), arm, 'attn')
            self.assertTrue(result['execution_path_validated'])
            self.assertEqual(result['median_us']['barrier_us'], 1)
            self.assertAlmostEqual(result['median_us']['cpu_marker_to_gemm_launch_us'], 10)
            self.assertAlmostEqual(result['median_us']['gemm_to_barrier_api_gap_us'], 9.9)
            self.assertAlmostEqual(result['sum_us']['cpu_marker_to_gemm_launch_us'], 40)
            result = full._trace_forward(fixture(candidate, True), arm)
            self.assertEqual(result['attn']['publication_barriers'], 128)
            self.assertEqual(result['mlp']['barrier_us'], 1)
            self.assertAlmostEqual(result['attn']['cpu_marker_to_gemm_launch_us'], 9)
            self.assertAlmostEqual(result['mlp']['sum_us']['gemm_to_barrier_api_gap_us'], 128*9.9)

    def test_submission_intervals_require_unique_owned_launches(self):
        import copy
        events = fixture(True)
        marker = next(e for e in events if e.get('cat') == 'user_annotation')
        path = [e for e in events if e.get('cat') == 'kernel'][:3]
        launch = next(e for e in events if e.get('cat') == 'cuda_runtime' and
                      e['args']['correlation'] == path[0]['args']['correlation'])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            profile.submission_intervals(marker, path, events + [copy.deepcopy(launch)])
        launch['tid'] = 999
        with self.assertRaisesRegex(ValueError, 'correlated launch'):
            profile.submission_intervals(marker, path, events)

    def test_submission_gap_uses_cpu_api_end_not_gpu_kernel_end(self):
        events = fixture(True)
        marker = next(e for e in events if e.get('cat') == 'user_annotation')
        path = [e for e in events if e.get('cat') == 'kernel'][:3]
        # Synthetic GPU timestamps are a million us after their host launches.
        self.assertGreater(path[0]['ts'], 1000000)
        value = profile.submission_intervals(marker, path, events)
        self.assertAlmostEqual(value['gemm_to_barrier_api_gap_us'], 9.9)
        launch = next(e for e in events if e.get('cat') == 'cuda_runtime' and
                      e['args']['correlation'] == path[0]['args']['correlation'])
        launch['dur'] = 30
        with self.assertRaisesRegex(ValueError, 'precedes'):
            profile.submission_intervals(marker, path, events)

    def test_wrong_barrier_or_extra_clear_rejected(self):
        for arm in bench.ARMS:
            with self.assertRaises(ValueError):
                bench.trace_path(fixture(arm == bench.ARMS[0]), arm, 'attn')
        events = fixture(True)
        find(events, 'Memset', 'gpu_memset')['args']['bytes'] = 64
        with self.assertRaisesRegex(ValueError, 'epoch counters'):
            bench.trace_path(events, bench.ARMS[1], 'attn')

    def test_epoch_geometry_and_original_consumer_successor(self):
        events = fixture(True)
        find(events, 'epoch_barrier')['args']['block'] = [128, 1, 1]
        with self.assertRaisesRegex(ValueError, 'Epoch barrier launch'):
            bench.trace_path(events, bench.ARMS[1], 'attn')
        events = fixture(True, True)
        find(events, 'layernorm_forward')['name'] = 'extra_add'
        with self.assertRaisesRegex(ValueError, 'LayerNorm'):
            full._trace_forward(events, bench.ARMS[1])

    def test_both_arms_keep_all_ma11_compute(self):
        events = compute_fixture(True) + _shift(softmax_fixture(True), 1000000, 1000000)
        for arm in bench.ARMS:
            result = profile.trace_composed_records(events, arm, 0)
            self.assertEqual(result['attention_softmax']['softmax_compute_kernel_calls'], 256)
            self.assertEqual(result['zero_count']['compute_kernel_calls'], 2)
            self.assertEqual(result['activation_cache']['backward_allgathers'], 0)


if __name__ == '__main__':
    unittest.main()
