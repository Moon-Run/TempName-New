"""Reject fake DMA evidence, missing publication and unsafe gather consumers."""
import copy
import unittest

import ma18_profile as profile
from ma8_profile import TraceIndex
from test_ma8_profile import fixture as original_fixture


def fixture():
    events = original_fixture(True)
    index = TraceIndex(events)
    correlation = 100000
    for call in [e for e in events if e['name'] == '_MA8CachedRawLinear']:
        role = 'qkv' if call['args']['Sequence number'] % 65 % 2 == 0 else 'fc1'
        call['name'] = profile.COLUMN
        work, _ = index.work(call)
        gather = next(e for e in work if e['cat'] == 'kernel' and 'AllGather' in e['name'])
        gather.update(name='bytedance::flux::CudaIpcBarrierAllKernel(Args)',
                      ts=call['ts'] + 1001, dur=.5, tid=7)
        gather['args']['stream'] = 7
        for i in range(9):
            correlation += 1
            api_ts = call['ts'] + (8 if i == 0 else 11 + (i - 1) * .7)
            gpu_ts = call['ts'] + (1000 if i == 0 else 1002 + (i - 1) * .9)
            events.append(dict(cat='cuda_runtime', name='cudaMemcpyAsync', ts=api_ts,
                               dur=.1, tid=call['tid'], args={'correlation': correlation}))
            events.append(dict(cat='gpu_memcpy', name='Memcpy DtoD (Device -> Device)',
                               ts=gpu_ts, dur=.5, tid=7,
                               args=dict(correlation=correlation, stream=7, bytes=2097152)))
        events.append(dict(cat='user_annotation', name=profile.MARKERS[role],
                           ts=call['ts'] + 7, dur=12, tid=call['tid'], args={}))
    for event in events:
        if event['name'] == '_MA8CachedRawLinearBackward':
            event['name'] = profile.COLUMN + 'Backward'
    return events


def first_work(events):
    marker = next(e for e in events if e['name'] in profile.MARKERS.values())
    return sorted(TraceIndex(events).work(marker)[0], key=lambda e: e['ts'])


class ProfileTests(unittest.TestCase):
    def test_control_and_candidate_keep_full_gemm_backward_heads(self):
        for arm, events in [(profile.ARMS[0], original_fixture(True)), (profile.ARMS[1], fixture())]:
            activation, result = profile.trace_column_ipc_records(events, arm, 3)
            self.assertEqual(result['calls'], 256)
            self.assertEqual(result['gemms_per_call'], 1)
            self.assertEqual(result['native_copy_count'], 2304 if arm == profile.ARMS[1] else 0)
            self.assertEqual(activation['backward_allgathers'], 0)
            self.assertEqual(activation['original_vocabulary_heads']['calls'], 4)

    def reject(self, events, message):
        with self.assertRaisesRegex(ValueError, message):
            profile.trace_column_ipc_records(events, profile.ARMS[1])

    def test_missing_copy(self):
        events = fixture()
        events.remove(first_work(events)[0])
        self.reject(events, 'nine copies')

    def test_marker_without_cuda_api_is_rejected(self):
        events = fixture()
        corr = first_work(events)[0]['args']['correlation']
        events.remove(next(e for e in events if e['cat'] == 'cuda_runtime' and
                           e['args'].get('correlation') == corr))
        self.reject(events, 'nine copies')

    def test_wrong_bytes_and_host_copy(self):
        for field, value in [('bytes', 1048576), ('name', 'Memcpy HtoD')]:
            events = fixture()
            event = first_work(events)[0]
            (event if field == 'name' else event['args'])[field] = value
            self.reject(events, 'nine 2MiB device-to-device')

    def test_wrong_stream(self):
        events = fixture()
        first_work(events)[2]['args']['stream'] = 26
        self.reject(events, 'original compute stream')

    def test_epoch_or_missing_barrier_rejected(self):
        events = fixture()
        first_work(events)[1]['name'] = 'ma14_epoch_barrier'
        self.reject(events, 'original binary IPC barrier')

    def test_pull_before_publication_rejected(self):
        events = fixture()
        first_work(events)[2]['ts'] -= 2
        self.reject(events, 'device dependencies')

    def test_gemm_before_last_pull_rejected(self):
        events = fixture()
        first_work(events)[-1]['dur'] = 10
        self.reject(events, 'unfinished gather')

    def test_extra_payload_kernel_rejected(self):
        events = fixture()
        extra = copy.deepcopy(first_work(events)[0])
        extra.update(cat='kernel', name='hidden_payload_copy_kernel')
        events.append(extra)
        self.reject(events, 'nine copies')

    def test_runtime_fallback_and_host_sync_rejected(self):
        for category, name, message in [
                ('cpu_op', 'c10d::_allgather_base_', 'fell back'),
                ('cuda_runtime', 'cudaStreamSynchronize', 'host/device synchronization'),
                ('cuda_runtime', 'cudaStreamWaitEvent', 'stream handoff')]:
            events = fixture()
            events.append(dict(cat=category, name=name, ts=15, dur=.1, tid=1, args={}))
            self.reject(events, message)

    def test_missing_marker(self):
        events = fixture()
        events.remove(next(e for e in events if e['name'] in profile.MARKERS.values()))
        self.reject(events, '256 native gather markers')

    def test_changed_backward_rejected(self):
        events = fixture()
        backward = next(e for e in events if e['name'] == profile.COLUMN + 'Backward')
        work, _ = TraceIndex(events).work(backward)
        next(e for e in work if 'ReduceScatter' in e['name'])['name'] = 'nccl_AllGather'
        self.reject(events, 'changed cached column backward')

    def test_candidate_cannot_contaminate_control(self):
        with self.assertRaisesRegex(ValueError, 'control contains MA18'):
            profile.trace_column_ipc_records(fixture(), profile.ARMS[0])

    def test_control_cannot_be_reported_as_candidate(self):
        self.reject(original_fixture(True), '256 native gather markers')


if __name__ == '__main__':
    unittest.main()
