"""Trace gates reject missing work, wrong ownership and premature consumers."""
import copy
import unittest

import ma9_profile as profile


def fixture(candidate=False, rank=0, clear=False):
    events = []
    correlation = 0

    def event(cat, name, ts, dur, tid=1, **args):
        value = dict(cat=cat, name=name, ts=ts, dur=dur, tid=tid, args=args)
        events.append(value)
        return value

    def device(cpu, gpu, name, stream=7, cat='kernel', dur=1, **extra):
        nonlocal correlation
        correlation += 1
        event('cuda_runtime', 'cudaLaunchKernel', cpu, .1, correlation=correlation)
        return event(cat, name, gpu, dur, tid=stream, correlation=correlation,
                     stream=stream, device=rank, **extra)

    if candidate:
        event('user_annotation', profile.MARKER, 100, 60)
        device(110, 1000, 'void ma9_zero_count::' + profile.CHUNKS + '(...)', dur=2)
        device(120, 1010, 'void ma9_zero_count::' + profile.TOTAL + '(...)', dur=2)
        comm_cpu, comm_gpu = 130, 1020
    else:
        n = 389 if rank == 0 else 194
        for i in range(n):
            cpu, gpu = 100 + i * 40, 100000 + i * 100
            event('cpu_op', 'aten::count_nonzero', cpu, 20)
            event('cpu_op', 'aten::count_nonzero', cpu + 1, 18)
            event('cpu_op', 'aten::sum', cpu + 6, 12)
            device(cpu + 2, gpu, 'kernel CompareEqFunctor<float>')
            device(cpu + 7, gpu + 2, 'kernel direct_copy_kernel_cuda')
            if clear and i == 0:
                device(cpu + 11, gpu + 3, 'Memset (Device)', cat='gpu_memset', dur=.5, bytes=4)
            device(cpu + 13, gpu + 4, 'kernel ReduceOp<long>')
            device(cpu + 23, gpu + 6, 'kernel CUDAFunctorOnOther_add<long>')
            device(cpu + 28, gpu + 8, 'kernel CUDAFunctor_add<float>')
        comm_cpu, comm_gpu = 100 + n * 40, 100000 + n * 100
    event('cpu_op', 'c10d::allreduce_', comm_cpu, 5)
    event('cpu_op', 'record_param_comms', comm_cpu + .5, 4,
          **{'Collective name': 'allreduce', 'In msg nelems': 1, 'Out msg nelems': 1,
             'dtype': 'Float', 'Group size': 8, 'Process Group Description': 'MODEL_PARALLEL_GROUP'})
    device(comm_cpu + 1, comm_gpu, 'ncclDevKernel_AllReduce_Sum_f32_RING_LL', stream=26, dur=5)
    event('cpu_op', 'aten::item', comm_cpu + 10, 8)
    device(comm_cpu + 11, comm_gpu + 10, 'Memcpy DtoH (Device -> Pinned)',
           cat='gpu_memcpy', dur=2, bytes=4)
    device(comm_cpu + 30, comm_gpu + 20, 'multi_tensor_apply_kernel AdamFunctor<float>', dur=5)
    return events


def find(events, token, cat='kernel'):
    return next(e for e in events if e['cat'] == cat and token in e['name'])


class ProfileTests(unittest.TestCase):
    def test_untouched_baseline_all_ranks(self):
        for rank in range(8):
            result = profile.trace_zero_count_records(fixture(rank=rank), 'F4', rank)
            self.assertEqual(result['original_count_nonzero_calls'], 389 if rank == 0 else 194)
            self.assertEqual(result['compute_kernel_calls'], 1945 if rank == 0 else 970)

    def test_candidate_exact_two_kernels_and_original_collective(self):
        result = profile.trace_zero_count_records(fixture(True), 'F4-zcount', 0)
        self.assertEqual(result['compute_kernel_calls'], 2)
        self.assertEqual(result['original_count_nonzero_calls'], 0)
        self.assertTrue(result['optimizer_after_statistic'])

    def test_missing_fused_finish_rejected(self):
        events = fixture(True)
        events.remove(find(events, profile.TOTAL))
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_extra_fused_kernel_rejected(self):
        events = fixture(True)
        events.append(copy.deepcopy(find(events, profile.TOTAL)))
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_candidate_cannot_fall_back(self):
        with self.assertRaisesRegex(ValueError, 'fell back'):
            profile.trace_zero_count_records(fixture(), 'F4-zcount')

    def test_reference_cannot_be_wrapped(self):
        with self.assertRaisesRegex(ValueError, 'wrappers'):
            profile.trace_zero_count_records(fixture(True), 'F4')

    def test_candidate_marker_required(self):
        events = fixture(True)
        events.remove(find(events, profile.MARKER, 'user_annotation'))
        with self.assertRaisesRegex(ValueError, 'marker'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_wrong_compute_stream_rejected(self):
        events = fixture(True)
        find(events, profile.TOTAL)['args']['stream'] = 8
        with self.assertRaisesRegex(ValueError, 'compute stream'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_reversed_compute_order_rejected(self):
        events = fixture(True)
        find(events, profile.TOTAL)['ts'] = 990
        with self.assertRaisesRegex(ValueError, 'order changed'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_collective_cannot_read_unfinished_total(self):
        events = fixture(True)
        find(events, 'AllReduce')['ts'] = 1005
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_wrong_tp_group_rejected(self):
        events = fixture(True)
        find(events, 'record_param_comms', 'cpu_op')['args']['Group size'] = 1
        with self.assertRaisesRegex(ValueError, 'TP group'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_unowned_fused_launch_rejected(self):
        events = fixture(True)
        kernel = find(events, profile.CHUNKS)
        launch = next(e for e in events if e['cat'] == 'cuda_runtime' and
                      e['args']['correlation'] == kernel['args']['correlation'])
        launch['ts'] = 90
        with self.assertRaisesRegex(ValueError, 'coverage'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_unknown_copy_rejected(self):
        events = fixture(True)
        find(events, 'Memcpy', 'gpu_memcpy')['args']['bytes'] = 1024
        with self.assertRaisesRegex(ValueError, 'Unknown copy'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_premature_scalar_readback_rejected(self):
        events = fixture(True)
        find(events, 'Memcpy', 'gpu_memcpy')['ts'] = 1021
        with self.assertRaisesRegex(ValueError, 'premature'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_premature_optimizer_rejected(self):
        events = fixture(True)
        find(events, 'AdamFunctor')['ts'] = 1015
        with self.assertRaisesRegex(ValueError, 'Optimizer update'):
            profile.trace_zero_count_records(events, 'F4-zcount')

    def test_original_sum_semaphore_clear_allowed(self):
        result = profile.trace_zero_count_records(fixture(clear=True), 'F4')
        self.assertEqual(result['allowed_original_reduction_memsets'], 1)

    def test_unknown_original_clear_rejected(self):
        events = fixture(clear=True)
        find(events, 'Memset', 'gpu_memset')['args']['bytes'] = 8
        with self.assertRaisesRegex(ValueError, 'Unknown or duplicate'):
            profile.trace_zero_count_records(events, 'F4')

    def test_rank_filtering_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'TP rank filtering'):
            profile.trace_zero_count_records(fixture(rank=0), 'F4', 1)

    def test_ma8_reference_wrappers_rejected(self):
        events = fixture()
        events.append(dict(cat='cpu_op', name='MA8ReferenceLinear', ts=1, dur=2, tid=1, args={}))
        with self.assertRaisesRegex(ValueError, 'MA8 column'):
            profile.trace_zero_count_records(events, 'F4')


if __name__ == '__main__':
    unittest.main()
