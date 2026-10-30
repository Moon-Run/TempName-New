"""Profile evidence regressions: retain dependencies, reject silent fallback/copies."""
import unittest

import ma8_profile as profile


def fixture(candidate=False, row=False):
    events = []
    correlation = 0

    def event(cat, name, ts, dur, tid=1, **args):
        value = dict(cat=cat, name=name, ts=ts, dur=dur, tid=tid, pid=123, args=args)
        events.append(value)
        return value

    def device(call_ts, cpu_offset, gpu_ts, name, duration, stream, category='kernel', size=None):
        nonlocal correlation
        correlation += 1
        event('cuda_runtime', 'cudaMemsetAsync' if category == 'gpu_memset' else 'cudaLaunchKernel',
              call_ts + cpu_offset, .1, correlation=correlation)
        args = dict(correlation=correlation, stream=stream)
        if size is not None:
            args['bytes'] = size
        return event(category, name, gpu_ts, duration, tid=stream, **args)

    def gemm(ts, offset, gpu, clear=True):
        event('cpu_op', 'aten::mm', ts + offset - 1, 4)
        if clear:
            device(ts, offset, gpu - 1, 'Memset (Device)', .5, 7, 'gpu_memset', 4)
        return device(ts, offset + 1, gpu, 'sm90_cublas_gemm', 20, 7)

    for i in range(256 if row else 260):
        head = not row and i % 65 == 64
        role = ('head' if head else 'qkv' if (i % 2 if row else i % 65 % 2) == 0 else 'fc1')
        classname = ('ConsumerGemmRSFunction' if row else
                     profile.CACHED_COLUMN if candidate and not head else profile.ORIGINAL_COLUMN)
        fts, bts = i * 200., 100000. + i * 200.
        event('cpu_op', classname, fts, 90, **{'Sequence number': i})
        event('cpu_op', classname + 'Backward', bts, 90, **{'Sequence number': i})
        if row:
            event('user_annotation', 'scope.' + ('attn' if role == 'qkv' else 'mlp') + '.candidate', fts - 1, 92)
            device(bts, 10, bts + 1000, 'reduce_kernel', 8, 7)
            device(bts, 20, bts + 1010, 'nccl_AllGather', 20, 26)
            gemm(bts, 30, bts + 1032, clear=False)
            gemm(bts, 40, bts + 1054, clear=role == 'fc1')
            continue
        device(fts, 10, fts + 1000, 'nccl_AllGather', 10, 26)
        gemm(fts, 20, fts + 1012, clear=False)
        if role == 'qkv':
            device(fts, 30, fts + 1034, 'elementwise_add', 1, 7)
        if not candidate or head:
            device(bts, 10, bts + 1000, 'nccl_AllGather', 80, 26)
        gemm(bts, 20, bts + 1005)
        rs_gpu = bts + (1030 if candidate and not head else 1082)
        w_gpu = rs_gpu + 2
        device(bts, 40, rs_gpu, 'nccl_ReduceScatter', 20, 26)
        gemm(bts, 50, w_gpu)
        if role == 'qkv':
            event('cpu_op', 'aten::sum', bts + 59, 5)
            device(bts, 60, w_gpu + 21, 'Memset (Device)', .5, 7, 'gpu_memset', 48)
            device(bts, 62, w_gpu + 22, 'reduce_kernel', 1, 7)
    return events


class ProfileTests(unittest.TestCase):
    def test_reference_keeps_both_gathers(self):
        result = profile.trace_activation_cache_records(fixture(), 'F4')
        self.assertEqual(result['backward_allgathers'], 256)
        self.assertEqual(result['roles']['qkv']['allowed_workspace_memsets']['qkv_bias_48byte'], 128)
        self.assertEqual(result['original_vocabulary_heads']['backward_allgather_kernels'], 4)
        self.assertFalse(result['per_call_wrappers_and_annotations'])

    def test_candidate_removes_only_backward_gather(self):
        result = profile.trace_activation_cache_records(fixture(True), 'F4-cache')
        self.assertEqual(result['backward_allgathers'], 0)
        self.assertEqual(result['retained_forward_allgathers'], 256)
        self.assertEqual(result['retained_reduce_scatters'], 256)
        self.assertEqual(result['original_vocabulary_heads']['calls'], 4)
        self.assertEqual(result['original_vocabulary_heads']['backward_allgather_kernels'], 4)

    def test_reference_fallback_cannot_pass_candidate(self):
        with self.assertRaisesRegex(ValueError, 'Wrong raw column arm'):
            profile.trace_activation_cache_records(fixture(), 'F4-cache')

    def test_candidate_cannot_pass_raw_control(self):
        with self.assertRaisesRegex(ValueError, 'Wrong raw column arm'):
            profile.trace_activation_cache_records(fixture(True), 'F4')

    def test_old_per_call_annotation_is_rejected(self):
        events = fixture(True)
        events.append(dict(cat='user_annotation', name='ma8.column.backward.decoder.layers.0.self_attention.linear_qkv'))
        with self.assertRaisesRegex(ValueError, 'old per-call wrappers or annotations'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_old_function_wrapper_is_rejected(self):
        events = fixture(True)
        next(e for e in events if e['name'] == profile.CACHED_COLUMN)['name'] = 'MA8RetainedActivationLinear'
        with self.assertRaisesRegex(ValueError, 'old per-call wrappers or annotations'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_unknown_extra_function_wrapper_is_rejected(self):
        events = fixture(True)
        events.append(dict(cat='cpu_op', name='_MA8ExtraWrapper', tid=1))
        with self.assertRaisesRegex(ValueError, 'Unexpected MA8 wrapper'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_missing_original_head_is_rejected(self):
        events = fixture(True)
        events.remove(next(e for e in events if e['name'] == profile.ORIGINAL_COLUMN + 'Backward'))
        with self.assertRaisesRegex(ValueError, 'four original heads'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_cached_head_is_rejected(self):
        events = fixture(True)
        for e in events:
            if e['name'] == profile.ORIGINAL_COLUMN:
                e['name'] = profile.CACHED_COLUMN
            elif e['name'] == profile.ORIGINAL_COLUMN + 'Backward':
                e['name'] = profile.CACHED_COLUMN + 'Backward'
        with self.assertRaisesRegex(ValueError, 'Wrong raw column arm'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_original_head_gather_must_remain(self):
        events = fixture(True)
        head = next(e for e in events if e['name'] == profile.ORIGINAL_COLUMN + 'Backward')
        launches = {e['args']['correlation'] for e in events
                    if e['cat'] == 'cuda_runtime' and profile.contains(head, e)}
        gather = next(e for e in events if e['cat'] == 'kernel' and 'AllGather' in e['name']
                      and e['args']['correlation'] in launches)
        events.remove(gather)
        with self.assertRaisesRegex(ValueError, 'vocabulary head path changed'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_duplicate_backward_sequence_is_rejected(self):
        events = fixture(True)
        backwards = [e for e in events if e['name'] == profile.CACHED_COLUMN + 'Backward']
        backwards[1]['args']['Sequence number'] = backwards[0]['args']['Sequence number']
        with self.assertRaisesRegex(ValueError, 'duplicate raw backward sequence'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_wrong_raw_backward_class_is_rejected(self):
        events = fixture(True)
        next(e for e in events if e['name'] == profile.CACHED_COLUMN + 'Backward')['name'] = profile.ORIGINAL_COLUMN + 'Backward'
        with self.assertRaisesRegex(ValueError, 'forward/backward class differs'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_relabelled_baseline_with_gathers_is_rejected(self):
        events = fixture()
        for e in events:
            if e['cat'] == 'cpu_op' and e['name'].startswith(profile.ORIGINAL_COLUMN):
                if e['args']['Sequence number'] % 65 != 64:
                    e['name'] = e['name'].replace(profile.ORIGINAL_COLUMN, profile.CACHED_COLUMN)
        with self.assertRaisesRegex(ValueError, 'backward AllGather'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_raw_qkv_fc1_topology_checked_by_bias(self):
        events = fixture(True)
        forwards = [e for e in events if e['name'] == profile.CACHED_COLUMN]
        forwards[0]['ts'], forwards[1]['ts'] = forwards[1]['ts'], forwards[0]['ts']
        with self.assertRaises(ValueError):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_dropped_forward_gather_is_rejected(self):
        events = fixture(True)
        events.remove(next(e for e in events if e['cat'] == 'kernel' and 'AllGather' in e['name']))
        with self.assertRaisesRegex(ValueError, 'forward path'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_moved_weight_stream_is_rejected(self):
        events = fixture(True)
        target = next(e for e in events if e['cat'] == 'kernel' and 'gemm' in e['name'] and e['ts'] == 101032.)
        target['args']['stream'] = 9
        with self.assertRaises(ValueError):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_unowned_four_byte_clear_is_rejected(self):
        events = fixture(True)
        owner = next(e for e in events if e['name'] == 'aten::mm' and e['ts'] == 100019.)
        owner['name'] = 'aten::empty'
        with self.assertRaisesRegex(ValueError, 'outside the original library'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_large_clear_is_rejected(self):
        events = fixture(True)
        next(e for e in events if e['cat'] == 'gpu_memset')['args']['bytes'] = 4096
        with self.assertRaisesRegex(ValueError, 'memset size'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_copy_is_rejected(self):
        events = fixture(True)
        clear = next(e for e in events if e['cat'] == 'gpu_memset')
        clear['cat'] = 'gpu_memcpy'
        with self.assertRaisesRegex(ValueError, 'GPU copy'):
            profile.trace_activation_cache_records(events, 'F4-cache')

    def test_unchanged_rows_allow_original_cublas_clears(self):
        result = profile.trace_row_backward_records(fixture(row=True), 'F4-cache')
        self.assertEqual(result['mlp']['allowed_cublas_4byte_memsets'], 128)
        self.assertEqual(result['attn']['allowed_cublas_4byte_memsets'], 0)

    def test_rows_reject_unknown_clear(self):
        events = fixture(row=True)
        next(e for e in events if e['cat'] == 'gpu_memset')['args']['bytes'] = 48
        with self.assertRaisesRegex(ValueError, 'memset size'):
            profile.trace_row_backward_records(events, 'F4')

    def test_rows_reject_gather_before_bias(self):
        events = fixture(row=True)
        gather = next(e for e in events if e['cat'] == 'kernel' and 'AllGather' in e['name'])
        gather['ts'] -= 5
        with self.assertRaisesRegex(ValueError, 'order changed'):
            profile.trace_row_backward_records(events, 'F4')


if __name__ == '__main__':
    unittest.main()
