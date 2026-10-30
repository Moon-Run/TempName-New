"""Mechanism evidence must prove publication ownership and ordering."""
import copy
import unittest

import ma19_profile as profile


def fixture(candidate):
    events = []
    correlation = 0

    def operation(name, cpu_ts, gpu_ts, duration=1, owner=None, **args):
        nonlocal correlation
        correlation += 1
        events.append(dict(cat='cuda_runtime', name='cudaLaunchKernel', tid=1,
                           ts=cpu_ts, dur=.1, args=dict(correlation=correlation)))
        event = dict(cat='kernel', name=name, tid=7, ts=gpu_ts, dur=duration,
                     args=dict(correlation=correlation, stream=7, **args))
        events.append(event)
        if owner:
            events.append(dict(cat='cpu_op', name=owner, tid=1, ts=cpu_ts-.1, dur=.3, args={}))
        return event

    for i in range(4):
        operation('bfloat16_copy_kernel_cuda', i, i*2)
    for i in range(49):
        operation('AdamFunctor<float, float, int>', 10+i, 100+i*2)
    if candidate:
        events.append(dict(cat='user_annotation', name=profile.MARKER, tid=1,
                           ts=60, dur=2, args={}))
        operation(profile.KERNEL, 61, 200, duration=300,
                  grid=[profile.CHUNKS, 1, 1], block=[256, 1, 1])
    else:
        for i in range(389):
            operation('bfloat16_copy_kernel_cuda', 60+i, 200+i*2, owner='aten::copy_')
    for i in range(2):
        operation('FillFunctor<float>', 500+i, 1000+i*2)
    return events


class PublicationProfileTests(unittest.TestCase):
    def reject(self, events, message):
        with self.assertRaisesRegex(ValueError, message):
            profile.trace_parameter_publish_records(events, profile.ARMS[1])

    def native(self, events):
        return next(e for e in events if e['name'] == profile.KERNEL)

    def test_both_paths_have_unchanged_adam_and_clear_dependencies(self):
        for candidate in (False, True):
            result = profile.trace_parameter_publish_records(fixture(candidate), profile.ARMS[candidate])
            self.assertEqual(result['original_copy_calls'], 0 if candidate else 389)
            self.assertEqual(result['publication_kernels'], 1 if candidate else 389)
            self.assertTrue(result['publication_before_original_clears'])

    def test_marker_without_launch_rejected(self):
        events = fixture(True)
        key = self.native(events)['args']['correlation']
        events[:] = [e for e in events if not (e['cat'] == 'cuda_runtime' and
                                             e['args']['correlation'] == key)]
        self.reject(events, 'uniquely own')

    def test_wrong_stream_and_launch_geometry(self):
        for key, value in [('stream', 26), ('grid', [1, 1, 1]), ('block', [128, 1, 1])]:
            events = fixture(True)
            self.native(events)['args'][key] = value
            self.reject(events, 'stream|geometry')

    def test_premature_publication_rejected(self):
        events = fixture(True)
        self.native(events)['ts'] = 195
        self.reject(events, 'precedes completed Adam')

    def test_cpu_submission_before_adam_rejected(self):
        events = fixture(True)
        native = self.native(events)
        api = next(e for e in events if e['cat'] == 'cuda_runtime' and
                   e['args']['correlation'] == native['args']['correlation'])
        api['ts'] = 57.5
        marker = next(e for e in events if e['name'] == profile.MARKER)
        marker.update(ts=57.4, dur=.3)
        self.reject(events, 'submitted before original Adam')

    def test_original_copy_fallback_rejected(self):
        events = fixture(True)
        extra = copy.deepcopy(self.native(events))
        extra.update(name='bfloat16_copy_kernel_cuda', ts=700)
        events.append(extra)
        self.reject(events, 'retained per-parameter copies')

    def test_extra_host_sync_or_h2d_rejected(self):
        for name in ('cudaStreamSynchronize', 'cudaMemcpyAsync', 'cudaStreamWaitEvent'):
            events = fixture(True)
            events.append(dict(cat='cuda_runtime', name=name, tid=1, ts=60.2, dur=.1, args={}))
            self.reject(events, 'synchronization or stream handoff')

    def test_missing_adam_clear_or_head_copy_rejected(self):
        for token, message in [('AdamFunctor', '49 FP32 Adam'), ('FillFunctor', 'two original'),
                               ('bfloat16_copy', 'vocabulary-head')]:
            events = fixture(True)
            events.remove(next(e for e in events if e['cat'] == 'kernel' and token in e['name']))
            self.reject(events, message)

    def test_native_work_cannot_contaminate_control(self):
        with self.assertRaisesRegex(ValueError, 'control contains MA19'):
            profile.trace_parameter_publish_records(fixture(True), profile.ARMS[0])


if __name__ == '__main__':
    unittest.main()
