"""Gradient batching trace gates must prove coverage and dependency ownership."""
import copy
import unittest

import ma10_profile as profile


def fixture(candidate=False, rank=0):
    events = []
    correlation = 0

    def event(cat, name, ts, dur, tid=2, **args):
        value = dict(cat=cat, name=name, ts=ts, dur=dur, tid=tid, args=args)
        events.append(value)
        return value

    def device(cpu, gpu, name, tid=2, stream=7, cat='kernel', **args):
        nonlocal correlation
        correlation += 1
        event('cuda_runtime', 'cudaLaunchKernel', cpu, .1, tid,
              correlation=correlation)
        return event(cat, name, gpu, 1, stream, correlation=correlation,
                     stream=stream, device=rank, **args)

    for microbatch in range(4):
        base = microbatch * 10000
        if candidate:
            event('user_annotation', profile.BACKWARD, base, 8000, tid=1)
        queued = 0
        for param in range(389):
            start = base + 100 + param * 10
            event('cpu_op', profile.ACCUMULATE, start, 5)
            event('cpu_op', 'torch::autograd::AccumulateGrad', start + .1, .5)
            if not candidate:
                event('cpu_op', 'aten::add_', start + 1, 2)
                device(start + 1.2, 1000000 + start, 'original CUDAFunctor_add<float>')
                continue
            queued += 1
            if queued == 32 or param == 388:
                event('user_annotation', profile.FLUSH + str(queued), start + 6, 2)
                device(start + 6.2, 1000000 + start + 5, 'void ' + profile.KERNEL + '(...)')
                queued = 0
    # This is the original optimizer, with both ATen count_nonzero overloads.
    for param in range(389 if rank == 0 else 194):
        start = 60000 + param * 5
        event('cpu_op', 'aten::count_nonzero', start, 4, tid=1)
        event('cpu_op', 'aten::count_nonzero', start + 1, 2, tid=1)
        device(start + 1.5, 1000000 + start, 'original CompareEqFunctor<float>', tid=1)
    device(65000, 1065000, 'multi_tensor_apply_kernel AdamFunctor<float>', tid=1)
    return events


def find(events, token, cat='kernel'):
    return next(e for e in events if e['cat'] == cat and token in e['name'])


def launch_for(events, kernel):
    return next(e for e in events if e['cat'] == 'cuda_runtime' and
                e['args']['correlation'] == kernel['args']['correlation'])


class ProfileTests(unittest.TestCase):
    def test_original_unwrapped_control_retains_all_gradient_adds(self):
        for rank in (0, 1):
            result = profile.trace_gradient_batching_records(fixture(rank=rank), 'F4')
            self.assertEqual(result['gradient_kernel_calls'], 1556)
            self.assertEqual(result['original_add_calls'], 1556)
            self.assertEqual(result['flush_calls'], 0)

    def test_candidate_flushes_all_four_microbatches(self):
        result = profile.trace_gradient_batching_records(fixture(True), 'F4-gacc')
        self.assertEqual(result['gradient_kernel_calls'], 52)
        self.assertEqual(result['tensor_commits'], 1556)
        self.assertEqual(result['original_add_calls'], 0)
        self.assertTrue(result['optimizer_after_commits'])

    def test_missing_parameter_hook_rejected(self):
        events = fixture(True)
        events.remove(find(events, profile.ACCUMULATE, 'cpu_op'))
        with self.assertRaisesRegex(ValueError, '389 parameter hooks'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_original_control_cannot_have_candidate_wrapper(self):
        events = fixture()
        events.append(dict(cat='user_annotation', name=profile.BACKWARD,
                           ts=0, dur=1, tid=1, args={}))
        with self.assertRaisesRegex(ValueError, 'Original CR3'):
            profile.trace_gradient_batching_records(events, 'F4')

    def test_previous_mechanisms_rejected(self):
        for name in ('ma7.drain', 'MA8ReferenceLinear', 'ma9.optimizer.zero_count'):
            events = fixture()
            events.append(dict(cat='user_annotation', name=name, ts=0, dur=1, tid=1, args={}))
            with self.assertRaisesRegex(ValueError, 'earlier mechanism'):
                profile.trace_gradient_batching_records(events, 'F4')

    def test_candidate_per_parameter_add_fallback_rejected(self):
        events = fixture(True)
        leaf = find(events, profile.ACCUMULATE, 'cpu_op')
        events.append(dict(cat='cpu_op', name='aten::add_', ts=leaf['ts'] + 1,
                           dur=1, tid=leaf['tid'], args={}))
        with self.assertRaisesRegex(ValueError, 'fell back'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_missing_microbatch_boundary_rejected(self):
        events = fixture(True)
        events.remove(find(events, profile.BACKWARD, 'user_annotation'))
        with self.assertRaisesRegex(ValueError, 'microbatch boundaries'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_overlapping_microbatch_boundaries_rejected(self):
        events = fixture(True)
        find(events, profile.BACKWARD, 'user_annotation')['dur'] = 11000
        with self.assertRaisesRegex(ValueError, 'microbatch boundaries'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_gradient_commit_count_cannot_cross_microbatch(self):
        events = fixture(True)
        rows = [e for e in events if e['cat'] == 'user_annotation' and e['name'].startswith(profile.FLUSH)]
        # Preserve the global 1556 total while moving one claimed tensor from
        # microbatch 1 to microbatch 2. Per-microbatch ownership must reject it.
        rows[0]['name'] = profile.FLUSH + '31'
        rows[25]['name'] = profile.FLUSH + '6'
        with self.assertRaisesRegex(ValueError, 'crossed a microbatch'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_missing_native_kernel_rejected(self):
        events = fixture(True)
        events.remove(find(events, profile.KERNEL))
        with self.assertRaisesRegex(ValueError, 'exactly one native'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_missing_flush_marker_rejected(self):
        events = fixture(True)
        events.remove(find(events, profile.FLUSH, 'user_annotation'))
        with self.assertRaisesRegex(ValueError, 'unique owning flush'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_unowned_native_launch_rejected(self):
        events = fixture(True)
        launch_for(events, find(events, profile.KERNEL))['ts'] = 0
        with self.assertRaisesRegex(ValueError, 'exactly one native'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_extra_descriptor_copy_rejected(self):
        events = fixture(True)
        kernel = find(events, profile.KERNEL)
        copied = copy.deepcopy(kernel)
        copied.update(cat='gpu_memcpy', name='Memcpy HtoD (Pageable -> Device)')
        copied['args']['bytes'] = 1024
        events.append(copied)
        with self.assertRaisesRegex(ValueError, 'without descriptor copies'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_one_commit_on_auxiliary_stream_rejected(self):
        events = fixture(True)
        find(events, profile.KERNEL)['args']['stream'] = 8
        with self.assertRaisesRegex(ValueError, 'stream'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_all_commits_on_auxiliary_stream_rejected(self):
        events = fixture(True)
        for event in events:
            if event['cat'] == 'kernel' and profile.KERNEL in event['name']:
                event['args']['stream'] = 8
        with self.assertRaisesRegex(ValueError, 'stream'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_overlapping_gradient_writes_rejected(self):
        events = fixture(True)
        kernels = [e for e in events if e['cat'] == 'kernel' and profile.KERNEL in e['name']]
        kernels[1]['ts'] = kernels[0]['ts'] + .5
        with self.assertRaisesRegex(ValueError, 'overlap|order'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_gpu_commit_order_must_match_owning_flush_order(self):
        events = fixture(True)
        kernels = [e for e in events if e['cat'] == 'kernel' and profile.KERNEL in e['name']]
        kernels[0]['ts'], kernels[13]['ts'] = kernels[13]['ts'], kernels[0]['ts']
        with self.assertRaisesRegex(ValueError, 'order|microbatch'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_optimizer_cannot_read_unfinished_commits(self):
        for token in ('CompareEqFunctor', 'AdamFunctor'):
            events = fixture(True)
            find(events, token)['ts'] = 1000000
            with self.assertRaisesRegex(ValueError, 'unfinished gradient'):
                profile.trace_gradient_batching_records(events, 'F4-gacc')

    def test_zero_statistics_remain_original(self):
        events = fixture(True)
        events.remove(find(events, 'aten::count_nonzero', 'cpu_op'))
        with self.assertRaisesRegex(ValueError, 'zero statistics'):
            profile.trace_gradient_batching_records(events, 'F4-gacc')


if __name__ == '__main__':
    unittest.main()
