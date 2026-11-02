"""CPU corruption tests for the 5090 graph trace acceptance boundary."""
import json
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
        with self.assertRaisesRegex(ValueError, 'wrong CUDA device'):
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
