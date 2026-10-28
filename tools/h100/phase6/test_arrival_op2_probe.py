"""CPU protocol tests; synthetic timings only exercise selection and corruption gates."""
import ast
import copy
import math
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import arrival_op2_probe as probe
from common import read, write, sha


def timings():
    return {role: {label: [100.+i for i in range(12)] for label, _ in probe.CATALOG}
            for role in probe.ROLES}


class ProbeTests(unittest.TestCase):
    def test_schedule_balances_all_candidates_and_reverses_second_round(self):
        jobs = probe.schedule()
        self.assertEqual(len(jobs), 72)
        self.assertEqual([x['label'] for x in jobs[36:]], [x['label'] for x in jobs[:36]][::-1])
        labels = {label for label, _ in probe.CATALOG}
        for round_number in (1, 2):
            rows = [x for x in jobs if x['round'] == round_number]
            for block in range(6):
                self.assertEqual({x['label'] for x in rows if x['block'] == block}, labels)
            for position in range(6):
                self.assertEqual({x['label'] for x in rows if x['position'] == position}, labels)

    def test_no_gain_retains_A_and_identity_control_cannot_win(self):
        values = timings()
        for role in values:
            values[role]['identity32'] = [1.]*12
        result = probe.select(values)
        self.assertEqual(result['selected_spans'], dict(attn=32, mlp=32))
        self.assertTrue(all(d['label'] == 'A' for d in result['decisions'].values()))

    def test_selection_is_per_role_and_keeps_catalog_order_on_tie(self):
        values = timings()
        for label in ('g1', 'g4'):
            values['attn'][label] = [v*.95 for v in values['attn']['A']]
        values['mlp']['g8'] = [v*.97 for v in values['mlp']['A']]
        result = probe.select(values)
        self.assertEqual(result['selected_spans'], dict(attn=1, mlp=8))
        self.assertEqual(result['decisions']['attn']['admitted'], ['g1', 'g4'])
        self.assertEqual(result['decisions']['attn']['comparisons']['g1']['confidence'], .99375)

    def test_incomplete_nonfinite_or_nonpositive_measurements_fail(self):
        for mutation in ('missing', 'length', 'nan', 'zero'):
            values = timings()
            if mutation == 'missing':
                del values['attn']['g1']
            elif mutation == 'length':
                values['attn']['g1'].pop()
            else:
                values['attn']['g1'][0] = float('nan') if mutation == 'nan' else 0.
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.select(values)

    def test_family_interval_requires_lower_bound_above_zero(self):
        values = timings()
        values['attn']['g1'] = [v*ratio for v, ratio in zip(values['attn']['A'], [.8, 1.19]*6)]
        result = probe.select(values)
        comparison = result['decisions']['attn']['comparisons']['g1']
        self.assertGreater(comparison['percent'], 0)
        self.assertLess(comparison['ci_family'][0], 0)
        self.assertEqual(result['selected_spans']['attn'], 32)

    def test_worker_uses_A_api_clone_and_no_profiler(self):
        source = probe.worker_source()
        ast.parse(source)
        ast.parse(probe.reference_source())
        self.assertIn("GemmRSTaco(group, m, n, k_local, placement='fused')", source)
        self.assertIn('op.forward(inputs, weights).clone()', source)
        self.assertIn('mask=None', source)
        self.assertNotIn('torch.profiler', source)
        self.assertNotIn('selected=', source)
        self.assertIn("H100_OP2_ATTN_SPAN", source)
        self.assertTrue(probe.worker_source(True).startswith(
            "raise RuntimeError('Source-only probe worker is permanently non-executable')"))

    def test_probe_interval_is_not_a_full_step_target(self):
        result = probe.paired_statistics([math.log(.98)]*12)
        self.assertAlmostEqual(result['percent'], 2)
        self.assertEqual(result['blocks'], 12)
        self.assertEqual(probe.BOOTSTRAP['comparisons'], 8)
        self.assertNotIn('accepted', result)

    def test_source_only_selection_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = dict(protocol=probe.PROTOCOL, source_only=True, config=dict(tp=8))
            write(root/'build.json', build)
            builder = mock.Mock(MANIFEST='build.json')
            builder.verify.return_value = build
            with mock.patch.object(probe, '_builder', return_value=builder), \
                    mock.patch.object(probe, '_runtime_identities', return_value={'frozen': 'sources'}):
                path = root/'selection.json'
                actual = probe.make_source_only_selection(path, root)
                self.assertEqual(probe.selection_verify(path, root), actual)
                actual['gpu_measured'] = True
                actual['selection_sha256'] = probe.digest({
                    k: v for k, v in actual.items() if k != 'selection_sha256'})
                write(path, actual)
                with self.assertRaises(ValueError):
                    probe.selection_verify(path, root)

    def test_selection_does_not_accept_another_build(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = dict(protocol=probe.PROTOCOL, source_only=True, config=dict(tp=8))
            write(root/'build.json', build)
            builder = mock.Mock(MANIFEST='build.json')
            builder.verify.return_value = build
            with mock.patch.object(probe, '_builder', return_value=builder), \
                    mock.patch.object(probe, '_runtime_identities', return_value={}):
                path = root/'selection.json'
                probe.make_source_only_selection(path, root)
                write(root/'build.json', dict(build, changed=True))
                with self.assertRaises(ValueError):
                    probe.selection_verify(path, root)


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root/'scripts').mkdir()
        write(cls.root/'scripts/config.json', {'frozen': 'config'})
        cls.plan = dict(run_id='synthetic-protocol-test', schedule=probe.schedule(),
                        communication={}, runtime_identities={p: dict(libraries={})
                                                              for p in (probe.A, probe.S)})
        devices = [dict(name='NVIDIA H100', uuid=f'GPU-test{i}', sm_count=132, capability=[9, 0])
                   for i in range(8)]
        write(cls.root/'start-gpu-check.json',
              dict(passed=True, run_id=cls.plan['run_id'], host='test-host',
                   uuids=[f'test{i}' for i in range(8)]))
        for job in cls.plan['schedule']:
            folder = cls.root/'results'/job['directory']
            folder.mkdir(parents=True)
            for rank in range(8):
                sample = dict(inputs_sha256='a'*64, weights_sha256='b'*64, output_sha256='c'*64,
                              final_output_sha256='c'*64, repeated_bitwise_equal=True,
                              correctness=dict(passed=True, max_abs=0., relative_l2=0., protocol_relative_l2=0.),
                              wall_us=100+rank+job['block'], gpu_us=50.)
                row = dict(completed=True, protocol=probe.PROTOCOL, probe_protocol=probe.PROBE_PROTOCOL,
                           run_id=cls.plan['run_id'], plan_sha256=sha(cls.root/'scripts/config.json'),
                           rank=rank, world_size=8, job=job,
                           policy=probe.A if job['label'] == 'A' else probe.S,
                           span=dict(probe.CATALOG)[job['label']], timing=dict(probe.TIMING),
                           profiling=False, sampling=False, communication={}, library_hashes={},
                           hardware=dict(host='test-host', devices=devices, local_rank=rank),
                           software={'test': 'synthetic_only'},
                           results={role: dict(sample, shape=shape) for role, shape in probe.ROLES.items()})
                write(folder/f'rank{rank}.json', row)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rank_max_is_taken_once_per_complete_process_block(self):
        measured = probe._records(self.root, self.plan)
        self.assertEqual(len(measured['files']), 576)
        for role in probe.ROLES:
            for label, _ in probe.CATALOG:
                self.assertEqual(measured['values'][role][label], list(range(107, 113))*2)

    def test_changed_input_output_or_rank_is_rejected(self):
        path = self.root/'results'/self.plan['schedule'][1]['directory']/'rank3.json'
        original = read(path)
        for kind in ('input', 'output', 'rank', 'codec', 'timing'):
            bad = copy.deepcopy(original)
            if kind == 'input':
                bad['results']['attn']['inputs_sha256'] = 'd'*64
            elif kind == 'output':
                bad['results']['attn']['output_sha256'] = bad['results']['attn']['final_output_sha256'] = 'd'*64
            elif kind == 'rank':
                bad['rank'] = 2
            elif kind == 'codec':
                bad['results']['attn']['correctness']['protocol_relative_l2'] = .02
            else:
                bad['results']['attn']['wall_us'] = float('inf')
            try:
                write(path, bad)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    probe._records(self.root, self.plan)
            finally:
                write(path, original)

    def test_extra_rank_record_fails_instead_of_discarding_it(self):
        path = self.root/'results'/'rank8.json'
        try:
            write(path, {})
            with self.assertRaises(ValueError):
                probe._records(self.root, self.plan)
        finally:
            path.unlink()


if __name__ == '__main__':
    unittest.main()
