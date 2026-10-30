"""Dependency/ownership tests, including an async trace with no real overlap."""
import copy
import unittest

import ma8_profile as cache
import ma16_profile as profile
from test_ma8_profile import fixture as original_fixture


def fixture(rank=3):
    events = original_fixture(True)
    index = cache.TraceIndex(events)
    pairs, _ = cache._raw_column_pairs(index, 'F4-cache', 'ma8-v2')
    remove, added = set(), []
    corr = max(e.get('args', {}).get('correlation', 0) for e in events)

    def event(cat, name, ts, dur, tid=1, **args):
        value = dict(cat=cat, name=name, ts=ts, dur=dur, tid=tid, args=args)
        added.append(value)
        return value

    def kernel(cpu, gpu, name, dur, stream=7):
        nonlocal corr
        corr += 1
        event('cuda_runtime', 'cudaLaunchKernel', cpu, .1, correlation=corr)
        return event('kernel', name, gpu, dur, stream, correlation=corr, stream=stream)

    for role, forward, backward in pairs:
        work, _ = index.work(forward)
        remove.update(map(id, work))
        remove.update(id(e) for e in events if e is not forward and
                      e.get('cat') in ('cpu_op', 'cuda_runtime') and cache.contains(forward, e))
        forward['name'] = profile.COLUMN
        backward['name'] = profile.COLUMN + 'Backward'
        t = forward['ts']
        event('user_annotation', f'ma16.column.forward.{role}.rank{rank}', t + 1, 65)
        event('user_annotation', 'ma16.column.gather', t + 5, 10)
        kernel(t + 10, t + 1000, 'nccl_AllGather', 50, 26)
        event('user_annotation', 'ma16.column.local', t + 20, 10)
        event('cpu_op', 'aten::mm', t + 21, 4)
        kernel(t + 22, t + 1005, 'sm90_cublas_gemm', 20)
        event('user_annotation', 'ma16.column.wait', t + 32, 3)
        event('cuda_runtime', 'cudaStreamWaitEvent', t + 33, .1)
        next_gpu = t + 1052
        if rank:
            event('user_annotation', 'ma16.column.prefix', t + 37, 10)
            event('cpu_op', 'aten::mm', t + 39, 4)
            kernel(t + 40, next_gpu, 'sm90_cublas_gemm', 20)
            next_gpu += 22
        if rank < 7:
            event('user_annotation', 'ma16.column.suffix', t + 50, 10)
            event('cpu_op', 'aten::mm', t + 52, 4)
            kernel(t + 53, next_gpu, 'sm90_cublas_gemm', 20)
            next_gpu += 22
        if role == 'qkv':
            kernel(t + 70, next_gpu, 'elementwise_add', 1)
    return [e for e in events if id(e) not in remove] + added


def stage_kernel(events, stage):
    scope = next(e for e in events if e['name'] == 'ma16.column.' + stage)
    work, _ = cache.TraceIndex(events).work(scope)
    return next(e for e in work if e['cat'] == 'kernel')


class ProfileTests(unittest.TestCase):
    def test_original_ma11_keeps_serial_forward(self):
        activation, result = profile.trace_column_overlap_records(original_fixture(True), profile.ARMS[0])
        self.assertFalse(result['overlap_observed'])
        self.assertEqual(activation['backward_allgathers'], 0)

    def test_all_rank_placements_have_real_overlap_and_original_backward(self):
        for rank in range(8):
            with self.subTest(rank=rank):
                activation, result = profile.trace_column_overlap_records(fixture(rank), profile.ARMS[1], rank)
                self.assertEqual(result['gemms_per_call'], 2 if rank in (0, 7) else 3)
                self.assertEqual(result['roles']['qkv']['overlapping_calls'], 128)
                self.assertEqual(result['roles']['fc1']['sum_ms']['local_allgather_overlap_us'], 2.56)
                self.assertTrue(activation['no_gpu_copies'])
                self.assertEqual(activation['original_vocabulary_heads']['calls'], 4)

    def test_async_without_device_overlap_is_not_success(self):
        events = fixture()
        index = cache.TraceIndex(events)
        for scope in [e for e in events if e['name'] == 'ma16.column.local']:
            work, _ = index.work(scope)
            work[0]['ts'] += 45
            # Move remotes too, retaining valid dependency order.
        for stage in ('prefix', 'suffix'):
            for scope in [e for e in events if e['name'] == 'ma16.column.' + stage]:
                work, _ = index.work(scope)
                work[0]['ts'] += 25
        for e in events:
            if e['cat'] == 'kernel' and e['name'] == 'elementwise_add':
                e['ts'] += 25
        with self.assertRaisesRegex(ValueError, 'no observed GPU overlap'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_remote_read_before_completion_rejected(self):
        events = fixture()
        stage_kernel(events, 'prefix')['ts'] -= 10
        with self.assertRaisesRegex(ValueError, 'unfinished gather'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_missing_wait_rejected(self):
        events = fixture()
        events.remove(next(e for e in events if e['name'] == 'ma16.column.wait'))
        with self.assertRaisesRegex(ValueError, 'stage coverage'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_empty_wait_marker_rejected_even_with_safe_timestamps(self):
        events = fixture()
        events.remove(next(e for e in events if e['name'] == 'cudaStreamWaitEvent'))
        with self.assertRaisesRegex(ValueError, 'actual CUDA stream dependency'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_wait_before_local_rejected(self):
        events = fixture()
        next(e for e in events if e['name'] == 'ma16.column.wait')['ts'] = 16
        with self.assertRaisesRegex(ValueError, 'must submit'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_same_stream_collective_rejected(self):
        events = fixture()
        stage_kernel(events, 'gather')['args']['stream'] = 7
        with self.assertRaisesRegex(ValueError, 'not independent'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_extra_output_copy_rejected(self):
        events = fixture()
        value = copy.deepcopy(stage_kernel(events, 'local'))
        value.update(cat='gpu_memcpy', name='Memcpy DtoD')
        events.append(value)
        with self.assertRaisesRegex(ValueError, 'Unexpected GPU copy'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_missing_gemm_launch_rejected(self):
        events = fixture()
        correlation = stage_kernel(events, 'local')['args']['correlation']
        events.remove(next(e for e in events if e['cat'] == 'cuda_runtime' and
                           e['args']['correlation'] == correlation))
        with self.assertRaisesRegex(ValueError, 'exactly one kernel'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_changed_backward_rejected(self):
        events = fixture()
        backward = next(e for e in events if e['name'] == profile.COLUMN + 'Backward')
        work, _ = cache.TraceIndex(events).work(backward)
        next(e for e in work if e['cat'] == 'kernel' and 'ReduceScatter' in e['name'])['name'] = 'nccl_AllGather'
        with self.assertRaisesRegex(ValueError, 'changed cached column backward'):
            profile.trace_column_overlap_records(events, profile.ARMS[1])

    def test_control_cannot_contain_candidate(self):
        with self.assertRaisesRegex(ValueError, 'control contains'):
            profile.trace_column_overlap_records(fixture(), profile.ARMS[0])

    def test_candidate_cannot_be_a_relabelled_control(self):
        with self.assertRaisesRegex(ValueError, 'Wrong raw column|Unexpected MA8 wrapper'):
            profile.trace_column_overlap_records(original_fixture(True), profile.ARMS[1])

    def test_rank_crosscheck(self):
        with self.assertRaisesRegex(ValueError, 'rank or compute stream'):
            profile.trace_column_overlap_records(fixture(3), profile.ARMS[1], 4)


if __name__ == '__main__':
    unittest.main()
