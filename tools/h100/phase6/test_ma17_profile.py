"""Reject missing launches, silent PG fallback and unsafe stream placement."""
import copy
import unittest

import ma17_profile as profile
from ma8_profile import TraceIndex, contains
from test_ma8_profile import fixture as original_fixture


def fixture():
    events = original_fixture(True)
    index = TraceIndex(events)
    for call in [e for e in events if e['name'] == '_MA8CachedRawLinear']:
        seq = call['args']['Sequence number']
        role = 'qkv' if seq % 65 % 2 == 0 else 'fc1'
        call['name'] = profile.COLUMN
        work, launches = index.work(call)
        gather = next(e for e in work if e['cat'] == 'kernel' and 'AllGather' in e['name'])
        gather['args']['stream'] = gather['tid'] = 7
        launch = launches[gather['args']['correlation']]
        events.append(dict(cat='user_annotation', name=profile.MARKERS[role],
                           ts=launch['ts'] - 1, dur=3, tid=call['tid'], args={}))
    for event in events:
        if event['name'] == '_MA8CachedRawLinearBackward':
            event['name'] = profile.COLUMN + 'Backward'
    return events


def first_gather(events):
    marker = next(e for e in events if e['name'] in profile.MARKERS.values())
    work, _ = TraceIndex(events).work(marker)
    return work[0]


class ProfileTests(unittest.TestCase):
    def test_control_and_candidate_preserve_compute_and_backward(self):
        for arm, events in [(profile.ARMS[0], original_fixture(True)), (profile.ARMS[1], fixture())]:
            activation, record = profile.trace_column_direct_records(events, arm, 3)
            self.assertEqual(record['calls'], 256)
            self.assertEqual(record['gemms_per_call'], 1)
            self.assertEqual(record['direct_compute_stream'], arm == profile.ARMS[1])
            self.assertEqual(activation['backward_allgathers'], 0)
            self.assertEqual(activation['original_vocabulary_heads']['calls'], 4)

    def test_missing_marker(self):
        events = fixture()
        events.remove(next(e for e in events if e['name'] in profile.MARKERS.values()))
        with self.assertRaisesRegex(ValueError, '256 native gather markers'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_wrong_stream(self):
        events = fixture()
        first_gather(events)['args']['stream'] = 26
        with self.assertRaisesRegex(ValueError, 'not on the compute stream'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_unfinished_gather(self):
        events = fixture()
        first_gather(events)['dur'] = 100
        with self.assertRaisesRegex(ValueError, 'unfinished gather'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_native_call_requires_cuda_launch(self):
        events = fixture()
        corr = first_gather(events)['args']['correlation']
        events.remove(next(e for e in events if e['cat'] == 'cuda_runtime' and
                           e['args'].get('correlation') == corr))
        with self.assertRaisesRegex(ValueError, 'exactly one column AllGather'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_marker_must_own_actual_launch(self):
        events = fixture()
        next(e for e in events if e['name'] in profile.MARKERS.values())['ts'] += 4
        with self.assertRaisesRegex(ValueError, 'unique AllGather launch ownership'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_pg_fallback(self):
        events = fixture()
        events.append(dict(cat='cpu_op', name='c10d::_allgather_base_', ts=5, dur=10, tid=1, args={}))
        with self.assertRaisesRegex(ValueError, 'fell back to ProcessGroup'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_explicit_work_wait_rejected(self):
        events = fixture()
        events.append(dict(cat='cpu_op', name='record_param_comms', ts=15, dur=1, tid=1,
                           args={'Collective name': 'wait'}))
        with self.assertRaisesRegex(ValueError, 'explicit ProcessGroup Work.wait'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_external_handoff_rejected_but_nccl_internal_events_reported(self):
        events = fixture()
        internal = dict(cat='cuda_runtime', name='cudaStreamWaitEvent', ts=10.5, dur=.1, tid=1, args={})
        events.append(internal)
        _, record = profile.trace_column_direct_records(events, profile.ARMS[1])
        self.assertEqual(record['roles']['qkv']['event_api_counts']['cudaStreamWaitEvent'], 1)
        internal['ts'] = 15
        with self.assertRaisesRegex(ValueError, 'external stream handoff'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_host_sync_rejected(self):
        events = fixture()
        events.append(dict(cat='cuda_runtime', name='cudaStreamSynchronize', ts=15, dur=.1, tid=1, args={}))
        with self.assertRaisesRegex(ValueError, 'host/device synchronization'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_added_copy_rejected(self):
        events = fixture()
        value = copy.deepcopy(first_gather(events))
        value.update(cat='gpu_memcpy', name='Memcpy DtoD')
        events.append(value)
        with self.assertRaisesRegex(ValueError, 'GPU copy'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_changed_backward_rejected(self):
        events = fixture()
        backward = next(e for e in events if e['name'] == profile.COLUMN + 'Backward')
        work, _ = TraceIndex(events).work(backward)
        next(e for e in work if 'ReduceScatter' in e['name'])['name'] = 'nccl_AllGather'
        with self.assertRaisesRegex(ValueError, 'changed cached column backward'):
            profile.trace_column_direct_records(events, profile.ARMS[1])

    def test_candidate_cannot_contaminate_control(self):
        with self.assertRaisesRegex(ValueError, 'control contains MA17'):
            profile.trace_column_direct_records(fixture(), profile.ARMS[0])

    def test_control_is_not_candidate(self):
        with self.assertRaisesRegex(ValueError, '256 native gather markers'):
            profile.trace_column_direct_records(original_fixture(True), profile.ARMS[1])


if __name__ == '__main__':
    unittest.main()
