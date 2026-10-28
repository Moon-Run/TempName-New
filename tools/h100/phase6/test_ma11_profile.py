"""Mechanism checks reject absent kernels, fallback and changed dependencies."""
import copy
import unittest

import ma11_profile as profile


def fixture(candidate=False):
    events = []
    correlation = 0

    def event(cat, name, ts, dur, tid=1, **args):
        value = dict(cat=cat, name=name, ts=ts, dur=dur, tid=tid, args=args)
        events.append(value)
        return value

    def kernel(ts, name, stream=7):
        nonlocal correlation
        correlation += 1
        event('cuda_runtime', 'cudaLaunchKernel', ts, .1, correlation=correlation)
        return event('kernel', name, ts + 1000000, 1, stream,
                     correlation=correlation, stream=stream, device=0)

    def bmm(ts):
        event('cpu_op', 'aten::bmm', ts, 5)
        kernel(ts + 1, 'sm90_xmma_gemm_bf16')

    for micro in range(4):
        for layer in range(32):
            t = micro * 64000 + layer * 1000
            event('cpu_op', 'aten::baddbmm', t + 10, 5)
            kernel(t + 11, 'sm90_xmma_gemm_bf16')
            if candidate:
                event('user_annotation', profile.FORWARD_MARKER, t + 30, 5)
                kernel(t + 31, profile.FORWARD_KERNEL)
            else:
                event('cpu_op', 'aten::masked_fill_', t + 20, 5)
                kernel(t + 21, 'masked_fill_kernel')
                event('cpu_op', 'aten::_softmax', t + 30, 5)
                kernel(t + 31, 'cunn_SoftMaxForwardSmem<8, c10::BFloat16, float>')
            bmm(t + 40)
        for layer in range(32):
            t = micro * 64000 + 32000 + layer * 1000
            bmm(t + 1)
            bmm(t + 10)
            if candidate:
                event('user_annotation', profile.BACKWARD_MARKER, t + 20, 10)
                kernel(t + 22, profile.BACKWARD_KERNEL)
            else:
                event('cpu_op', 'aten::_softmax_backward_data', t + 20, 10)
                event('cpu_op', 'aten::mul', t + 21, 3)
                kernel(t + 22, 'MulFunctor<c10::BFloat16>')
                kernel(t + 25, 'cunn_SoftMaxBackward')
                event('cpu_op', 'AsStridedBackward0', t + 40, 10)
                kernel(t + 42, 'FillFunctor<c10::BFloat16>')
                event('cpu_op', 'torch::autograd::CopySlices', t + 60, 30)
                event('cpu_op', 'MaskedFillBackward0', t + 70, 10)
                event('cpu_op', 'aten::masked_fill_', t + 71, 3)
                kernel(t + 72, 'masked_fill_kernel')
            event('cpu_op', 'BaddbmmBackward0', t + 100, 20)
            bmm(t + 101)
            bmm(t + 111)
    for i in range(4):
        event('cpu_op', 'torch::autograd::CopySlices', 300000 + i * 10, 2)
    for i in range(20):
        event('cpu_op', 'aten::masked_fill_', 310000 + i * 10, 2)
    for i in range(532):
        event('cpu_op', 'aten::mul', 320000 + i * 10, 2)
    return events


def find(events, token, cat='kernel'):
    return next(e for e in events if e['cat'] == cat and token in e['name'])


class ProfileTests(unittest.TestCase):
    def test_original_ma89_without_added_wrappers(self):
        result = profile.trace_attention_softmax_records(fixture(), profile.ARMS[0])
        self.assertEqual(result['softmax_compute_kernel_calls'], 640)
        self.assertEqual(result['attention_view_replays'], 128)
        self.assertEqual(result['microbatches'], 4)

    def test_candidate_eliminates_dense_mask_and_view_replay(self):
        result = profile.trace_attention_softmax_records(fixture(True), profile.ARMS[1])
        self.assertEqual(result['softmax_compute_kernel_calls'], 256)
        self.assertEqual(result['attention_view_replays'], 0)
        self.assertEqual(result['copy_slices_calls'], 4)

    def test_unknown_arm(self):
        with self.assertRaisesRegex(ValueError, 'only MA89'):
            profile.trace_attention_softmax_records([], 'F4')

    def test_control_must_remain_original(self):
        with self.assertRaisesRegex(ValueError, 'wrappers or kernels'):
            profile.trace_attention_softmax_records(fixture(True), profile.ARMS[0])

    def test_candidate_requires_all_native_kernels(self):
        events = fixture(True)
        events.remove(find(events, profile.BACKWARD_KERNEL))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_original_mask_view_fallback_rejected(self):
        events = fixture(True)
        events.append(dict(cat='cpu_op', name='MaskedFillBackward0', ts=0, dur=1, tid=1))
        with self.assertRaisesRegex(ValueError, 'fell back'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_wrong_owner(self):
        events = fixture(True)
        value = find(events, profile.FORWARD_KERNEL)
        runtime = next(e for e in events if e['cat'] == 'cuda_runtime' and
                       e['args']['correlation'] == value['args']['correlation'])
        runtime['ts'] -= 10
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_wrong_stream(self):
        events = fixture(True)
        find(events, profile.FORWARD_KERNEL)['args']['stream'] = 8
        with self.assertRaisesRegex(ValueError, 'stream changed'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_softmax_cannot_precede_qk(self):
        events = fixture(True)
        find(events, profile.FORWARD_KERNEL)['ts'] -= 30
        with self.assertRaisesRegex(ValueError, 'dependency order'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_missing_original_qk(self):
        events = fixture(True)
        events.remove(find(events, 'aten::baddbmm', 'cpu_op'))
        with self.assertRaisesRegex(ValueError, 'GEMM coverage'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_extra_marker_work_rejected(self):
        events = fixture(True)
        value = copy.deepcopy(find(events, profile.FORWARD_KERNEL))
        value.update(cat='gpu_memset', name='Memset (Device)')
        events.append(value)
        with self.assertRaisesRegex(ValueError, 'extra copies'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_microbatch_order_checked(self):
        events = fixture(True)
        value = find(events, profile.BACKWARD_MARKER, 'user_annotation')
        value['ts'] = 500
        with self.assertRaisesRegex(ValueError, 'microbatch'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_old_bf16_product_is_required_in_reference(self):
        events = fixture()
        find(events, 'MulFunctor')['name'] = 'MulFunctor<float>'
        with self.assertRaisesRegex(ValueError, 'BF16 backward'):
            profile.trace_attention_softmax_records(events, profile.ARMS[0])

    def test_attention_copy_replay_cannot_survive_candidate(self):
        events = fixture(True)
        events.append(dict(cat='cpu_op', name='torch::autograd::CopySlices', ts=400000, dur=1, tid=1))
        with self.assertRaisesRegex(ValueError, 'CopySlices'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_unowned_dense_product_rejected(self):
        events = fixture(True)
        events.append(dict(cat='cpu_op', name='aten::mul', ts=400000, dur=1, tid=1))
        with self.assertRaisesRegex(ValueError, 'dense attention mask/product'):
            profile.trace_attention_softmax_records(events, profile.ARMS[1])

    def test_both_arms_retain_ma8_and_ma9(self):
        from test_ma89_profile import fixture as original, _shift
        for candidate in (False, True):
            events = original(True) + _shift(fixture(candidate), 1000000, 1000000)
            result = profile.trace_composed_records(events, profile.ARMS[int(candidate)], 0)
            self.assertEqual(result['activation_cache']['backward_allgathers'], 0)
            self.assertEqual(result['zero_count']['compute_kernel_calls'], 2)
            self.assertEqual(result['attention_softmax']['softmax_compute_kernel_calls'],
                             256 if candidate else 640)

    def test_view_replay_counts_actual_memcpy_bytes(self):
        events = fixture()
        scope = find(events, 'torch::autograd::CopySlices', 'cpu_op')
        correlation = 900000
        ts = scope['ts'] + 5
        events.extend([
            dict(cat='cuda_runtime', name='cudaMemcpyAsync', ts=ts, dur=.1, tid=1,
                 args=dict(correlation=correlation)),
            dict(cat='gpu_memcpy', name='Memcpy DtoD (Device -> Device)',
                 ts=ts + 1000000, dur=2, tid=7,
                 args=dict(correlation=correlation, stream=7, bytes=33554432))])
        result = profile.trace_attention_softmax_records(events, profile.ARMS[0])
        copies = result['view_replay_device']['copy_slices']
        self.assertEqual(copies['gpu_memcpy_calls'], 1)
        self.assertEqual(copies['gpu_memcpy_bytes'], 33554432)
        self.assertEqual(result['gpu_work_ms']['copy_slices_device'], .002)


if __name__ == '__main__':
    unittest.main()
