"""Reject incomplete graph traces, eager fallback, and missing ownership copies."""
import copy
import unittest
from unittest.mock import patch

import ma15_profile as profile


GEMM = 'sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_cublas'
COPY = 'void direct_copy_kernel_cuda<c10::BFloat16>'
SCALE = 'AUnaryFunctor<c10::BFloat16, MulFunctor<float>>'
RNG_FILL = 'void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>>'


def fixture():
    events = []
    serial = 0

    def event(cat, name, ts, dur=1, **args):
        value = dict(cat=cat, name=name, ts=ts, dur=dur, tid=1, args=args)
        events.append(value)
        return value

    def correlation():
        nonlocal serial
        serial += 1
        return serial

    def copy_op(ts, clone=False):
        if clone:
            event('cpu_op', 'aten::clone', ts, 5)
        event('cpu_op', 'aten::copy_', ts + 1, 3)
        c = correlation()
        event('cuda_runtime', 'cudaLaunchKernel', ts + 2, .1, correlation=c)
        event('kernel', COPY, ts + 1000002, .5, correlation=c, stream=7)

    for micro in range(4):
        for direction in ('forward', 'backward'):
            for layer in range(32):
                t = micro * 12800 + (6400 if direction == 'backward' else 0) + layer * 200
                marker = profile.FORWARD_MARKER if direction == 'forward' else profile.BACKWARD_MARKER
                event('user_annotation', marker, t, 100)
                for i in range(4 if direction == 'forward' else 1):
                    copy_op(t + 1 + i * 7)
                for i in range(2):
                    ts = t + 30 + i * 4
                    event('cpu_op', 'aten::fill_', ts, 3)
                    fill_correlation = correlation()
                    event('cuda_runtime', 'cudaLaunchKernel', ts + 1, .1,
                          correlation=fill_correlation)
                    event('kernel', RNG_FILL, ts + 1000001, .5,
                          correlation=fill_correlation, stream=7, grid=[1, 1, 1], block=[128, 1, 1])
                c = correlation()
                event('cuda_runtime', 'cudaGraphLaunch', t + 40, 2, correlation=c)
                if direction == 'forward':
                    nodes = [('gpu_memset', 'Memset (Device)'), ('kernel', GEMM),
                             ('kernel', profile.FORWARD_KERNEL), ('kernel', GEMM), ('kernel', COPY)]
                else:
                    nodes = [('kernel', GEMM), ('gpu_memset', 'Memset (Device)'), ('kernel', GEMM),
                             ('kernel', profile.BACKWARD_KERNEL), ('kernel', GEMM), ('kernel', SCALE),
                             ('gpu_memset', 'Memset (Device)'), ('kernel', GEMM), ('kernel', SCALE)]
                for node, (cat, name) in enumerate(nodes):
                    args = dict(correlation=c, stream=7)
                    if cat == 'gpu_memset':
                        args['bytes'] = 4
                    elif name in (profile.FORWARD_KERNEL, profile.BACKWARD_KERNEL):
                        args.update(grid=[8192, 1, 1], block=[1024 if direction == 'forward' else 128, 1, 1])
                    event(cat, name, t + 1000040 + node, .5, **args)
                for i in range(1 if direction == 'forward' else 3):
                    copy_op(t + 70 + i * 7, clone=True)
    for i in range(4):
        event('cpu_op', 'torch::autograd::CopySlices', 60000 + i * 10)
    return events


def find(events, name, cat=None):
    return next(e for e in events if name in e['name'] and (cat is None or e['cat'] == cat))


class GraphProfileTests(unittest.TestCase):
    def check(self, events):
        return profile.trace_attention_graph_records(events, profile.ARMS[1])

    def test_complete_actual_launch_correlations_and_staging(self):
        record = self.check(fixture())
        self.assertEqual(record['forward_graph_launches'], 128)
        self.assertEqual(record['backward_graph_launches'], 128)
        self.assertEqual(record['graph_nodes_per_replay']['forward']['gemm'], 2)
        self.assertEqual(record['graph_nodes_per_replay']['backward']['gemm'], 4)
        self.assertGreater(record['staging_gpu_work_ms'], 0)
        prologue = record['rng_replay_prologue']
        self.assertEqual(prologue['fill_kernel_calls'], 512)
        self.assertEqual(prologue['logical_write_bytes_from_pinned_source'], 4096)
        self.assertGreater(prologue['gpu_work_ms'], 0)
        self.assertFalse(prologue['tensor_shape_observed'])
        self.assertAlmostEqual(record['total_attention_device_work_ms'],
                               record['graph_gpu_work_ms'] + record['staging_gpu_work_ms'] +
                               prologue['gpu_work_ms'])

    def test_control_delegates_to_original_ma11_without_instrumentation(self):
        with patch('ma15_profile.ma11_profile.trace_composed_records', return_value={'original': True}) as raw:
            record = profile.trace_composed_records([], profile.ARMS[0], 3)
        raw.assert_called_once_with([], profile.ARMS[0], 3)
        self.assertFalse(record['attention_graph']['installed'])

    def test_synthetic_original_ma11_control(self):
        from test_ma11_profile import fixture as original
        from test_ma89_profile import fixture as common, _shift
        events = common(True) + _shift(original(True), 1000000, 1000000)
        record = profile.trace_composed_records(events, profile.ARMS[0], 0)
        self.assertEqual(record['attention_softmax']['softmax_compute_kernel_calls'], 256)

    def test_no_graph_kernel_correlations_fails_closed(self):
        events = fixture()
        c = find(events, 'cudaGraphLaunch')['args']['correlation']
        for e in events:
            if e['cat'] in profile.DEVICE_CATEGORIES and e['args'].get('correlation') == c:
                e['args']['correlation'] = 999999
        with self.assertRaisesRegex(ValueError, 'CUPTI association'):
            self.check(events)

    def test_missing_native_node(self):
        events = fixture()
        events.remove(find(events, profile.BACKWARD_KERNEL))
        with self.assertRaisesRegex(ValueError, 'node coverage'):
            self.check(events)

    def test_communication_inside_graph(self):
        events = fixture()
        find(events, GEMM)['name'] = 'ncclDevKernel_AllGather_RING_LL'
        with self.assertRaisesRegex(ValueError, 'Communication or Flux'):
            self.check(events)

    def test_unknown_graph_kernel(self):
        events = fixture()
        find(events, GEMM)['name'] = 'unexpected_custom_gemm'
        with self.assertRaisesRegex(ValueError, 'node coverage'):
            self.check(events)

    def test_softmax_must_follow_qk(self):
        events = fixture()
        find(events, profile.FORWARD_KERNEL)['ts'] -= 1.5
        with self.assertRaisesRegex(ValueError, 'dependency order'):
            self.check(events)

    def test_changed_graph_node_geometry(self):
        events = fixture()
        find(events, profile.FORWARD_KERNEL)['args']['block'] = [256, 1, 1]
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.check(events)

    def test_wrong_stream(self):
        events = fixture()
        find(events, profile.FORWARD_KERNEL)['args']['stream'] = 8
        with self.assertRaisesRegex(ValueError, 'stream changed'):
            self.check(events)

    def test_bounded_cupti_graph_timestamp_overlap_is_reported(self):
        events = fixture()
        first = find(events, GEMM)
        second = find(events, profile.FORWARD_KERNEL)
        first['dur'] = 1.1
        second['dur'] = 2.5
        # The fixture's following nodes must still follow softmax completion.
        correlation = first['args']['correlation']
        for event in events:
            if event['cat'] in profile.DEVICE_CATEGORIES and event['args'].get('correlation') == correlation:
                event['args'].update(device=2, context=1)
                if event['ts'] > second['ts']:
                    event['ts'] += 3
        # 40 ns is below both the 500 ns absolute and 5% relative bounds.
        first['dur'] = second['ts'] - first['ts'] + .04
        result = self.check(events)['timestamp_order_check']
        self.assertEqual(result['tolerated_overlap_count'], 1)
        self.assertAlmostEqual(result['max_observed_overlap_us'], .04)

    def test_observed_graph_pair_accepts_95ns_but_rejects_real_overlap(self):
        # Geometry/time scale from the failing rank2 graph replay; no GPU run.
        def pair(overlap):
            args = dict(stream=7, device=2, context=1, correlation=2098006)
            return [dict(cat='kernel', name=GEMM, ts=1000, dur=19.04, args=args.copy()),
                    dict(cat='kernel', name=SCALE, ts=1019.04-overlap, dur=2.592, args=args.copy())]
        work = pair(.095)
        stream, recorded = profile._same_stream_order(work, work)
        self.assertEqual(stream, 7)
        self.assertAlmostEqual(recorded[0]['observed_overlap_us'], .095)
        for overlap in (.14, 1.0):
            work = pair(overlap)
            with self.assertRaisesRegex(ValueError, 'dependency order'):
                profile._same_stream_order(work, work)

    def test_timestamp_exception_cannot_cross_launch_or_device_or_staging(self):
        args = dict(stream=7, device=2, context=1, correlation=10)
        original = [dict(cat='kernel', name=GEMM, ts=0, dur=20, args=args.copy()),
                    dict(cat='kernel', name=SCALE, ts=19.9, dur=3, args=args.copy())]
        for key in ('correlation', 'device', 'context', 'stream'):
            events = copy.deepcopy(original)
            events[1]['args'][key] += 1
            with self.assertRaises(ValueError):
                profile._same_stream_order(events, events)
        events = copy.deepcopy(original)
        events[1]['cat'] = 'gpu_memcpy'
        with self.assertRaises(ValueError):
            profile._same_stream_order(events, events)
        with self.assertRaises(ValueError):
            profile._same_stream_order(original, [original[0]])

    def test_missing_clone_is_not_free_performance(self):
        events = fixture()
        events.remove(find(events, 'aten::clone'))
        with self.assertRaisesRegex(ValueError, 'clone coverage'):
            self.check(events)

    def test_input_copy_cannot_execute_after_graph(self):
        events = fixture()
        find(events, COPY)['ts'] += 60
        with self.assertRaisesRegex(ValueError, 'overtook its input copies'):
            self.check(events)

    def test_missing_rng_fill_rejected(self):
        events = fixture()
        events.remove(find(events, 'aten::fill_'))
        with self.assertRaisesRegex(ValueError, 'exactly two pinned PyTorch RNG'):
            self.check(events)

    def test_extra_rng_fill_rejected(self):
        events = fixture()
        events.append(copy.deepcopy(find(events, 'aten::fill_')))
        with self.assertRaisesRegex(ValueError, 'exactly two pinned PyTorch RNG'):
            self.check(events)

    def test_non_long_fill_is_not_rng_prologue(self):
        events = fixture()
        find(events, RNG_FILL)['name'] = RNG_FILL.replace('long', 'float')
        with self.assertRaisesRegex(ValueError, 'scalar int64 fill kernel'):
            self.check(events)

    def test_wrong_rng_fill_geometry_rejected(self):
        events = fixture()
        find(events, RNG_FILL)['args']['grid'] = [2, 1, 1]
        with self.assertRaisesRegex(ValueError, 'scalar int64 fill kernel'):
            self.check(events)

    def test_contrary_rng_shape_evidence_rejected(self):
        events = fixture()
        find(events, 'aten::fill_')['args']['Input Dims'] = [[2048], []]
        with self.assertRaisesRegex(ValueError, 'tensor shape differs'):
            self.check(events)

    def test_unowned_extra_fill_is_not_allowed(self):
        events = fixture()
        events.append(copy.deepcopy(find(events, RNG_FILL)))
        with self.assertRaisesRegex(ValueError, 'scalar int64 fill kernel'):
            self.check(events)

    def test_rng_fill_cannot_execute_after_graph(self):
        events = fixture()
        find(events, RNG_FILL)['ts'] += 30
        with self.assertRaisesRegex(ValueError, 'precede graph execution'):
            self.check(events)

    def test_eager_fallback(self):
        events = fixture()
        events.append(dict(cat='cpu_op', name='aten::bmm', ts=70000, dur=1, tid=1))
        with self.assertRaisesRegex(ValueError, 'eager attention'):
            self.check(events)

    def test_capture_in_profile_is_rejected(self):
        events = fixture()
        events.append(dict(cat='cuda_runtime', name='cudaStreamBeginCapture', ts=70000, dur=1, tid=1))
        with self.assertRaisesRegex(ValueError, 'capture or instantiation'):
            self.check(events)

    def test_unowned_softmax(self):
        events = fixture()
        extra = copy.deepcopy(find(events, profile.FORWARD_KERNEL))
        extra['args']['correlation'] = 999999
        events.append(extra)
        with self.assertRaisesRegex(ValueError, 'missing or lack graph ownership'):
            self.check(events)

    def test_unowned_tagged_graph_node(self):
        events = fixture()
        extra = copy.deepcopy(find(events, GEMM))
        extra['args'].update(correlation=999999, **{'graph node id': 17})
        events.append(extra)
        with self.assertRaisesRegex(ValueError, 'Unowned CUDA graph node'):
            self.check(events)

    def test_reused_launch_correlations(self):
        events = fixture()
        launches = [e for e in events if e['name'] == 'cudaGraphLaunch']
        launches[1]['args']['correlation'] = launches[0]['args']['correlation']
        with self.assertRaisesRegex(ValueError, 'missing or reused'):
            self.check(events)

    def test_control_cannot_run_graphs(self):
        with self.assertRaisesRegex(ValueError, 'control acquired'):
            profile.trace_attention_graph_records(fixture(), profile.ARMS[0])

    def test_union_does_not_double_count_nested_cpu_intervals(self):
        self.assertEqual(profile.interval_union_ms([
            dict(ts=0, dur=10), dict(ts=2, dur=3), dict(ts=8, dur=5), dict(ts=20, dur=2)]), .015)


if __name__ == '__main__':
    unittest.main()
