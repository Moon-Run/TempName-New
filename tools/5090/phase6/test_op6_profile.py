"""CPU corruption tests for the 5090 graph trace acceptance boundary."""
import json
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import op6_profile as profile
from test_ma15_profile import fixture as h100_fixture, find


def fixture():
    events = h100_fixture()
    # The 2.8 build has neither the 2.6 RNG prologue fills nor the H100
    # attention GEMM scratch clears. Replace dispatch with observed OP4 names.
    fill_correlations = {e['args']['correlation'] for e in events
                         if e['cat'] == 'kernel' and 'FillFunctor<long>' in e['name']}
    events = [e for e in events if e['cat'] != 'gpu_memset' and e['name'] != 'aten::fill_'
              and e.get('args', {}).get('correlation') not in fill_correlations]
    for event in events:
        if event['cat'] in profile.graph_auditor.DEVICE_CATEGORIES:
            event['args'].update(device=0, context=1)
    for launch in [e for e in events if profile.graph_auditor._is_graph_launch(e)]:
        nodes = sorted([e for e in events if e['cat'] == 'kernel' and
                        e['args'].get('correlation') == launch['args']['correlation']],
                       key=lambda e: e['ts'])
        forward = len(nodes) == 4
        gemms = [e for e in nodes if 'sm90_xmma_gemm' in e['name']]
        specs = ([(0, [512, 8, 4]), (1, [16, 1, 4])] if forward else
                 [(2, [16, 1, 4]), (0, [512, 8, 4]), (1, [16, 1, 4]), (2, [8, 4, 4])])
        for gemm, (kind, grid) in zip(gemms, specs):
            gemm['name'] = 'void cutlass::Kernel2<' + profile.GEMM_PREFIXES[kind] + '>'
            gemm['args'].update(grid=grid, block=[128, 1, 1])
        native = next(e for e in nodes if 'ma11_softmax' in e['name'])
        native['name'] = 'rtx5090_ma11_softmax::' + native['name']
        if forward:
            native['args'].update(grid=[2048, 1, 1], block=[32, 4, 1])
    return events


class GraphProfileTests(unittest.TestCase):
    def test_valid_5090_graph_with_no_rng_fills(self):
        result = profile.attention_graph_records(fixture(), 'op6', 0)
        self.assertEqual(result['forward_graph_launches'], 128)
        self.assertEqual(result['backward_graph_launches'], 128)
        self.assertEqual(result['rng_replay_prologue']['fill_kernel_calls'], 0)
        self.assertEqual(result['graph_nodes_per_replay']['forward'],
                         dict(gemm=2, softmax_forward=1, copy=1))
        self.assertAlmostEqual(result['total_attention_device_work_ms'],
                               result['graph_gpu_work_ms'] + result['staging_gpu_work_ms'])
        self.assertEqual(result, json.loads(json.dumps(result)))

    def test_real_sm120_graph_timestamp_edges_are_reported_and_unmodified(self):
        saved = json.loads((Path(__file__).parent/'fixtures/op6_graph_timestamp_edges.json').read_text())
        for example in saved['examples']:
            events = example['events']
            original = copy.deepcopy(events)
            with self.subTest(rank=example['rank']):
                stream, overlaps = profile.graph_auditor._same_stream_order(events, events)
                self.assertEqual(stream, events[0]['args']['stream'])
                self.assertEqual(len(overlaps), 1)
                self.assertGreater(overlaps[0]['observed_overlap_us'], example['old_limit_us'])
                self.assertEqual(events, original)

    def test_both_nov4_failures_are_diagnostics_not_dependency_failures(self):
        saved = json.loads((Path(__file__).parent/'fixtures/op6_graph_timestamp_nov4.json').read_text())
        for example in saved['examples']:
            events = example['events']
            original = copy.deepcopy(events)
            with self.subTest(batch=example['batch'], rank=example['rank']):
                _, overlaps = profile.graph_auditor._same_stream_order(events, events)
                self.assertEqual(len(overlaps), 1)
                self.assertTrue(overlaps[0]['exceeds_v1_reference'])
                self.assertEqual(overlaps[0]['observed_overlap_us'], example['observed_overlap_us'])
                self.assertEqual(events, original)

    def test_timestamp_policy_reports_reference_exceedance_without_claiming_serialization(self):
        events = fixture()
        launch = next(e for e in events if profile.graph_auditor._is_graph_launch(e))
        nodes = sorted([e for e in events if e['cat'] == 'kernel' and
                        e['args']['correlation'] == launch['args']['correlation']], key=lambda e: e['ts'])
        nodes[0]['dur'] = nodes[1]['ts'] - nodes[0]['ts'] + .2
        order = profile.attention_graph_records(events, 'op6', 0)['timestamp_order_check']
        self.assertEqual(order['exceeds_v1_reference_count'], 1)
        self.assertIsNone(order['graph_kernel_tolerance_us'])
        self.assertIsNone(order['max_fraction_of_shorter_kernel'])
        self.assertTrue(order['kernel_endpoint_overlap_is_diagnostic'])
        self.assertFalse(order['physical_serialization_proven_by_timestamps'])

    def test_timestamp_check_rejects_nested_intervals_and_wrong_ownership(self):
        def pair(overlap=.1, duration=10):
            return [dict(cat='kernel', name='gemm', ts=10., dur=10.,
                         args=dict(stream=7, device=0, context=1, correlation=31)),
                    dict(cat='kernel', name='scale', ts=20.-overlap, dur=duration,
                         args=dict(stream=7, device=0, context=1, correlation=31))]
        for events in (pair(1.01, 1.0), pair(1.0, 1.0), pair(10.)):
            with self.assertRaisesRegex(ValueError, 'dependency order'):
                profile.graph_auditor._same_stream_order(events, events)
        for key in ('stream', 'device', 'context', 'correlation'):
            events = pair()
            events[1]['args'][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile.graph_auditor._same_stream_order(events, events)
        for kind in ('staging', 'memcpy', 'same_start'):
            events = pair()
            graph = events
            if kind == 'staging':
                graph = [events[0]]
            elif kind == 'memcpy':
                events[1]['cat'] = 'gpu_memcpy'
            else:
                events[1]['ts'] = events[0]['ts']
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'dependency order'):
                profile.graph_auditor._same_stream_order(events, graph)

    def test_invalid_timestamps_and_unmatched_nonoverlapping_context_fail(self):
        for key, value in (('ts', float('nan')), ('ts', float('inf')),
                           ('dur', -1), ('dur', 0), ('dur', float('nan'))):
            events = [dict(cat='kernel', name='gemm', ts=10., dur=1.,
                           args=dict(stream=7, device=0, context=1, correlation=31))]
            events[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'timestamp'):
                profile.graph_auditor._same_stream_order(events, events)
        for key in ('stream', 'device', 'context'):
            events = [dict(cat='kernel', name='gemm', ts=10.+10*i, dur=1.,
                           args=dict(stream=7, device=0, context=1, correlation=31)) for i in range(2)]
            events[1]['args'][key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'device/context/stream'):
                profile.graph_auditor._same_stream_order(events, events)

    def test_error_identifies_rank_launch_and_failing_edge(self):
        events = fixture()
        launch = next(e for e in events if profile.graph_auditor._is_graph_launch(e))
        nodes = sorted([e for e in events if e['cat'] == 'kernel' and
                        e['args']['correlation'] == launch['args']['correlation']], key=lambda e: e['ts'])
        nodes[0]['dur'] = nodes[1]['ts'] - nodes[0]['ts'] + nodes[1]['dur'] + .1
        with self.assertRaisesRegex(ValueError, 'OP6 op6 profile rank0:.*nested intervals') as error:
            profile.analyze(events, 'op3-combined', 0)
        for detail in ('correlations=', 'predecessor=', 'successor=', 'overlap_us='):
            self.assertIn(detail, str(error.exception))

    def test_h100_timestamp_policy_is_unchanged(self):
        import ma15_profile
        self.assertIsNot(ma15_profile._same_stream_order, profile._same_stream_order)
        self.assertEqual(ma15_profile.GRAPH_KERNEL_TIMESTAMP_TOLERANCE_US, .5)
        self.assertEqual(ma15_profile.GRAPH_KERNEL_TIMESTAMP_MAX_FRACTION, .05)

    def test_missing_graph_association_fails(self):
        events = fixture()
        find(events, 'rtx5090_ma11_softmax')['args']['correlation'] = -1
        with self.assertRaisesRegex(ValueError, 'coverage'):
            profile.attention_graph_records(events, 'op6', 0)

    def test_h100_geometry_fails(self):
        events = fixture()
        find(events, 'rtx5090_ma11_softmax')['args']['grid'] = [8192, 1, 1]
        with self.assertRaisesRegex(ValueError, 'geometry'):
            profile.attention_graph_records(events, 'op6', 0)

    def test_missing_output_clone_fails(self):
        events = fixture()
        events.remove(find(events, 'aten::clone'))
        with self.assertRaisesRegex(ValueError, 'clone coverage'):
            profile.attention_graph_records(events, 'op6', 0)

    def test_communication_inside_graph_fails(self):
        events = fixture()
        find(events, 'cutlass::Kernel2')['name'] = 'ncclDevKernel_AllGather'
        with self.assertRaisesRegex(ValueError, 'Communication or Flux'):
            profile.attention_graph_records(events, 'op6', 0)

    def test_wrong_device_fails(self):
        events = fixture()
        find(events, 'rtx5090_ma11_softmax')['args']['device'] = 1
        with self.assertRaisesRegex(ValueError, 'device/context/stream|wrong CUDA device'):
            profile.attention_graph_records(events, 'op6', 0)

    def test_changed_gemm_geometry_fails_even_if_all_replays_match(self):
        events = fixture()
        for e in events:
            if profile.GEMM_PREFIXES[0] in e['name']:
                e['args']['grid'] = [256, 16, 4]
        with self.assertRaisesRegex(ValueError, 'GEMM dispatch'):
            profile.attention_graph_records(events, 'op6', 0)

    def test_control_cannot_acquire_graph(self):
        with self.assertRaisesRegex(ValueError, 'control acquired'):
            profile.attention_graph_records(fixture(), 'op4', 0)

    def test_rng_fill_is_not_allowed_in_pinned_nonrandom_graph(self):
        events = fixture()
        scope = find(events, profile.FORWARD_MARKER)
        events.append(dict(cat='cpu_op', name='aten::fill_', ts=scope['ts']+30,
                           dur=1, tid=scope['tid'], args={}))
        with self.assertRaisesRegex(ValueError, 'RNG fills'):
            profile.attention_graph_records(events, 'op6', 0)

    def test_scoped_install_restores_original_analysis(self):
        original = profile.op3_profile.analyze
        with patch.object(profile, 'analyze', return_value={'tested': True}) as checker:
            restore = profile.install('op6')
            self.assertEqual(profile.op3_profile.analyze([], 'op3-combined', 0), {'tested': True})
            checker.assert_called_once_with([], 'op3-combined', 0, 'op6')
            restore()
            restore()
        self.assertIs(profile.op3_profile.analyze, original)

    def test_invalid_variant_rejected_without_installing(self):
        original = profile.op3_profile.analyze
        with self.assertRaises(ValueError):
            profile.install('op5')
        self.assertIs(profile.op3_profile.analyze, original)


if __name__ == '__main__':
    unittest.main()
