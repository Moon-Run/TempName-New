"""Target-version stream rules preserve synchronous and asynchronous dependencies."""
import importlib.util
import unittest

from rtx5090_loader import H100, load
from baseline_profile_nccl import pytorch28_profiles
import ma8_profile
import ma9_profile


def _fixture_module(name):
    spec = importlib.util.spec_from_file_location('rtx5090_fixture_' + name, H100/(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


column_fixture = _fixture_module('test_ma8_profile').fixture
zero_fixture = _fixture_module('test_ma9_profile').fixture
column, zero = pytorch28_profiles(ma8_profile, ma9_profile)


def current_column(candidate=True, row=False):
    events = column_fixture(candidate=candidate, row=row)
    # H100 fixture forward AllGathers have GPU timestamps below 100000;
    # row AllGathers are all synchronous. Backward column/head gathers stay async.
    for event in events:
        if event['cat'] == 'kernel' and 'AllGather' in event['name'] and (row or event['ts'] < 100000):
            event['args']['stream'] = event['tid'] = 7
    return events


def current_zero(candidate=True, rank=0):
    events = zero_fixture(candidate=candidate, rank=rank)
    for event in events:
        if event['cat'] == 'kernel' and 'AllReduce' in event['name']:
            event['args']['stream'] = event['tid'] = 7
    return events


def kernel(events, token):
    return next(event for event in events if event['cat'] == 'kernel' and token in event['name'])


def splitk_row():
    events = current_column(row=True)
    gemm = kernel(events, 'gemm')
    correlation = max(event.get('args', {}).get('correlation', 0) for event in events) + 1
    launch = dict(cat='cuda_driver', name='cuLaunchKernel', ts=100032., dur=.1, tid=1,
                  args=dict(correlation=correlation))
    reduction = dict(cat='kernel', name='void cublasLt::splitKreduce_kernel<32, 16, float, __nv_bfloat16>(...)',
                     ts=gemm['ts']+gemm['dur']+.5, dur=1., tid=7,
                     args=dict(correlation=correlation, stream=7))
    events.extend([launch, reduction])
    return events, launch, reduction


class NCCLProfileTests(unittest.TestCase):
    def test_target_column_control_and_candidate(self):
        for candidate in (False, True):
            with self.subTest(candidate=candidate):
                result = column.trace_activation_cache_records(current_column(candidate),
                                                               'F4-cache' if candidate else 'F4')
                self.assertEqual(result['backward_allgathers'], 0 if candidate else 256)
                self.assertEqual(result['original_vocabulary_heads']['backward_allgather_kernels'], 4)
                self.assertEqual(result['collective_stream_policy'], 'pytorch28-sync-current-async-nccl')

    def test_same_stream_cannot_hide_premature_forward_consumer(self):
        events = current_column()
        kernel(events, 'gemm')['ts'] = 1005
        with self.assertRaisesRegex(ValueError, 'unfinished gather'):
            column.trace_activation_cache_records(events, 'F4-cache')

    def test_sync_gather_must_use_current_stream(self):
        events = current_column()
        kernel(events, 'AllGather')['args']['stream'] = 26
        with self.assertRaisesRegex(ValueError, 'Forward communication stream'):
            column.trace_activation_cache_records(events, 'F4-cache')

    def test_async_reduce_scatter_must_stay_on_separate_stream(self):
        events = current_column()
        kernel(events, 'ReduceScatter')['args']['stream'] = 7
        with self.assertRaisesRegex(ValueError, 'Column streams'):
            column.trace_activation_cache_records(events, 'F4-cache')

    def test_async_head_gather_must_stay_on_nccl_stream(self):
        events = current_column()
        head_gather = next(event for event in events if event['cat'] == 'kernel'
                           and 'AllGather' in event['name'] and event['ts'] > 100000)
        head_gather['args']['stream'] = 7
        with self.assertRaisesRegex(ValueError, 'vocabulary head streams'):
            column.trace_activation_cache_records(events, 'F4-cache')

    def test_async_control_gather_must_share_reduce_scatter_stream(self):
        for stream in (7, 99):
            with self.subTest(stream=stream):
                events = current_column(False)
                gather = next(event for event in events if event['cat'] == 'kernel'
                              and 'AllGather' in event['name'] and event['ts'] > 100000)
                gather['args']['stream'] = stream
                with self.assertRaisesRegex(ValueError, 'Asynchronous backward gather stream'):
                    column.trace_activation_cache_records(events, 'F4')

    def test_target_row_backward(self):
        result = column.trace_row_backward_records(current_column(row=True), 'F4-cache')
        self.assertEqual(result['attn']['backward_calls'], 128)
        self.assertEqual(result['mlp']['backward_calls'], 128)

    def test_row_gather_cannot_overtake_bias(self):
        events = current_column(row=True)
        kernel(events, 'AllGather')['ts'] -= 5
        with self.assertRaisesRegex(ValueError, 'order changed'):
            column.trace_row_backward_records(events, 'F4-cache')

    def test_row_dgrad_splitk_has_separate_count_and_duration(self):
        events, _, _ = splitk_row()
        result = column.trace_row_backward_records(events, 'F4-cache')
        self.assertEqual(result['attn']['splitk_reduction_kernels'], 1)
        self.assertEqual(result['mlp']['splitk_reduction_kernels'], 0)
        index = column.TraceIndex(events)
        call = next(event for event in events if event['name'] == 'ConsumerGemmRSFunctionBackward')
        work, launches = index.work(call)
        gemms = sorted([event for event in work if 'gemm' in event['name']], key=lambda event: event['ts'])
        from baseline_profile_nccl import _row_splitk, _is_splitk
        self.assertEqual(_row_splitk(index, call, launches, gemms,
                                    [event for event in work if _is_splitk(event)])['dgrad_us'], 21.)

    def test_row_splitk_must_share_original_mm_owner(self):
        events, launch, _ = splitk_row()
        launch['ts'] = 100034  # Outside dgrad aten::mm, still inside row backward.
        with self.assertRaisesRegex(ValueError, 'Unowned row split-K'):
            column.trace_row_backward_records(events, 'F4-cache')

    def test_row_splitk_must_share_stream_and_device(self):
        for field, value in (('stream', 26), ('device', 1)):
            with self.subTest(field=field):
                events, _, reduction = splitk_row()
                reduction['args'][field] = value
                with self.assertRaisesRegex(ValueError, 'split-K reduction stream or dependency'):
                    column.trace_row_backward_records(events, 'F4-cache')

    def test_row_splitk_must_finish_before_wgrad(self):
        events, _, reduction = splitk_row()
        reduction['dur'] = 10
        with self.assertRaisesRegex(ValueError, 'Row bias, gather, dgrad, wgrad order'):
            column.trace_row_backward_records(events, 'F4-cache')

    def test_unknown_splitk_like_kernel_rejected(self):
        events, _, reduction = splitk_row()
        reduction['name'] = reduction['name'].replace('cublasLt::', 'unknown::')
        with self.assertRaisesRegex(ValueError, 'kernel coverage'):
            column.trace_row_backward_records(events, 'F4-cache')

    def test_target_zero_control_and_candidate_all_ranks(self):
        for candidate in (False, True):
            for rank in range(8):
                with self.subTest(candidate=candidate, rank=rank):
                    result = zero.trace_zero_count_records(current_zero(candidate, rank),
                                                          'F4-zcount' if candidate else 'F4', rank)
                    self.assertEqual(result['original_allreduce_calls'], 1)
                    self.assertTrue(result['optimizer_after_statistic'])

    def test_same_stream_allreduce_cannot_overtake_total(self):
        events = current_zero()
        kernel(events, 'AllReduce')['ts'] = 1005
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            zero.trace_zero_count_records(events, 'F4-zcount', 0)

    def test_zero_collective_still_requires_owning_launch(self):
        events = current_zero()
        correlation = kernel(events, 'AllReduce')['args']['correlation']
        launch = next(event for event in events if event['cat'] == 'cuda_runtime'
                      and event['args']['correlation'] == correlation)
        launch['ts'] = 129  # Before c10d::allreduce_, still inside the zero-count marker.
        with self.assertRaisesRegex(ValueError, 'collective owner'):
            zero.trace_zero_count_records(events, 'F4-zcount', 0)

    def test_scalar_readback_cannot_overtake_allreduce(self):
        events = current_zero()
        next(event for event in events if event['cat'] == 'gpu_memcpy')['ts'] = 1021
        with self.assertRaisesRegex(ValueError, 'premature'):
            zero.trace_zero_count_records(events, 'F4-zcount', 0)

    def test_h100_modules_keep_original_stream_contract(self):
        self.assertEqual(ma8_profile.trace_activation_cache_records(column_fixture(True), 'F4-cache')
                         ['backward_allgathers'], 0)
        self.assertTrue(ma9_profile.trace_zero_count_records(zero_fixture(True), 'F4-zcount')
                        ['optimizer_after_statistic'])
        with self.assertRaisesRegex(ValueError, 'Forward communication stream'):
            ma8_profile.trace_activation_cache_records(current_column(), 'F4-cache')

    def test_composed_analyzer_uses_private_adapters(self):
        profile = load('profile')
        self.assertIs(profile.ma11_profile.ma8_profile, profile.ma8_profile)
        self.assertIs(profile.ma11_profile.ma9_profile, profile.ma9_profile)
        self.assertIsNot(profile.ma8_profile, ma8_profile)
        self.assertIsNot(profile.ma9_profile, ma9_profile)


if __name__ == '__main__':
    unittest.main()
