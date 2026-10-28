"""CPU admission tests; synthetic inputs never certify a real GPU run."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import arrival_probe as probe


def write(path, value):
    path.write_text(json.dumps(value)+'\n')


def contract_fixture(identity=False):
    base = [[((i % 8+r) % 8)*64+i//8 for i in range(512)] for r in range(8)]
    entry = dict(shape=[2048, 4096, 4096], maps=[list(range(512)) for _ in range(8)],
                 coords=copy.deepcopy(base), base_coords=base,
                 mask=[[int(t//64 != r) for t in range(512)] for r in range(8)])
    candidate = copy.deepcopy(entry)
    if not identity:
        candidate['maps'][0][0], candidate['maps'][0][8] = 8, 0
        candidate['coords'][0][0], candidate['coords'][0][8] = base[0][8], base[0][0]
    plans = {p: {role: dict(entry=copy.deepcopy(e), entry_sha256=probe.digest(e), mask_sha256='mask')
                 for role in ('attn', 'mlp')} for p, e in (('A', entry), ('B', candidate))}
    routing = SimpleNamespace(A='A', B='B', definition=lambda p: dict(policy=p, family=p, implementation='same',
                               order='base' if p == 'A' else 'arrival'), role_plans=lambda plan, p: plan[p])
    sources = [dict(codec='same', **{'phase6_host.hpp': p}) for p in ('A', 'B')]
    return routing, plans, sources


def report_fixture(low=0.1, rounds=2):
    summaries = {str(i): dict(blocks=6, comparisons={'B/A': dict(percent=1., ci95=[low, 2.])})
                 for i in range(1, rounds+1)}
    summaries['combined'] = dict(blocks=6*rounds, comparisons={'B/A': dict(percent=1., ci95=[low, 2.])})
    return dict(completed=True, correctness_passed=True, round_count=rounds, rounds=summaries,
                execution_timing=dict(warmup_steps=10, timed_steps=60),
                measurement_identity=dict(fingerprints=['fixed'], libraries='fixed'), offline=False)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='arrival-gate-test-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def records(self, count=2, identity=False):
        pair = probe.pair_contract(*contract_fixture(identity))
        records = []
        for i in range(count):
            root = self.folder/str(i); (root/'scripts').mkdir(parents=True)
            for name in ('submission.json', 'scripts/selection-plan.json', 'report.json', 'preflight.json'):
                write(root/name, {})
            records.append(dict(root=root, info=dict(run_id=str(i)), pair=pair,
                                signature={'fixed': True, 'timing': dict(warmup_steps=10, timed_steps=60)},
                                signature_sha256='fixed'))
        return records

    def invoke(self, records, reports=None, evaluate=True):
        with mock.patch.object(probe, 'inspect', side_effect=records), \
                mock.patch.object(probe, 'replay_report', side_effect=reports or [report_fixture() for _ in records]) as replay:
            result = probe.probe([r['root'] for r in records], evaluate)
            return result, replay.call_count

    def test_identity_skips_all_timing_even_with_evaluate(self):
        result, calls = self.invoke(self.records(identity=True))
        self.assertEqual(result['status'], 'not_tested_identity')
        self.assertFalse(result['evaluated'])
        self.assertFalse(result['validated_for_same_configuration'])
        self.assertEqual(calls, 0)

    def test_default_is_inspection_only(self):
        result, calls = self.invoke(self.records(), evaluate=False)
        self.assertEqual(result['status'], 'dry_run_needs_evaluation')
        self.assertEqual(calls, 0)

    def test_single_positive_run_requires_independent_confirmation(self):
        result, _ = self.invoke(self.records(count=1))
        self.assertEqual(result['status'], 'inconclusive_need_confirmation')
        self.assertFalse(result['validated_for_same_configuration'])

    def test_two_positive_runs_admit_only_same_joint_configuration(self):
        result, calls = self.invoke(self.records())
        self.assertEqual(result['status'], 'validated_for_same_configuration')
        self.assertTrue(result['validated_for_same_configuration'])
        self.assertFalse(result['gpu_launched'])
        self.assertEqual(result['scope'], 'joint_attn_mlp_full_optimizer_step')
        self.assertEqual(calls, 2)

    def test_crossing_zero_in_any_run_or_round_rejects(self):
        reports = [report_fixture(), report_fixture()]
        reports[1]['rounds']['2']['comparisons']['B/A']['ci95'][0] = 0.
        result, _ = self.invoke(self.records(), reports)
        self.assertEqual(result['status'], 'not_validated_no_repeatable_gain')
        self.assertFalse(result['validated_for_same_configuration'])

    def test_single_round_cannot_replace_reversed_round_confirmation(self):
        result, _ = self.invoke(self.records(), [report_fixture(rounds=1)]*2)
        self.assertFalse(result['validated_for_same_configuration'])

    def test_duplicate_run_id_rejected(self):
        records = self.records(); records[1]['info']['run_id'] = records[0]['info']['run_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate run_id'):
            self.invoke(records)

    def test_cross_run_configuration_and_measurement_drift_rejected(self):
        records = self.records(); records[1]['signature'] = {'fixed': False}
        with self.assertRaisesRegex(ValueError, 'identical plans'):
            self.invoke(records)
        records[1]['signature'] = records[0]['signature']
        reports = [report_fixture(), report_fixture()]
        reports[1]['measurement_identity']['libraries'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'worker/software'):
            self.invoke(records, reports)

    def test_same_codec_and_mask_are_mandatory(self):
        routing, plans, sources = contract_fixture()
        sources[1]['codec'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source hashes'):
            probe.pair_contract(routing, plans, sources)
        sources[1]['codec'] = sources[0]['codec']
        plans['B']['attn']['entry']['mask'][0][64] = 0
        with self.assertRaisesRegex(ValueError, 'shape, mask'):
            probe.pair_contract(routing, plans, sources)

    def test_unknown_protocol_and_cpu_fixture_cannot_enter_gate(self):
        root = self.folder/'inspect'; root.mkdir()
        write(root/'submission.json', dict(protocol='unsupported'))
        with self.assertRaisesRegex(ValueError, 'Only strict ablation4/7'):
            probe.inspect(root)
        write(root/'submission.json', dict(protocol='h100-gpt67-tp8-ablation7-v1'))
        reporter = SimpleNamespace(routing=None, verify_inputs=mock.Mock(
            return_value=dict(executable=False, source_only=True)))
        with mock.patch.object(probe.importlib, 'import_module', return_value=reporter):
            with self.assertRaisesRegex(ValueError, 'CPU fixtures'):
                probe.inspect(root)
        reporter.verify_inputs.assert_called_once_with(root.resolve(), offline=True)

    def test_invalid_ci_and_missing_round_rejected(self):
        result = report_fixture(); result['rounds']['1']['comparisons']['B/A']['ci95'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'confidence interval'):
            probe.paired_gate(result, 'A', 'B')
        result = report_fixture(); del result['rounds']['1']
        with self.assertRaisesRegex(ValueError, 'timing round'):
            probe.paired_gate(result, 'A', 'B')

    def test_missing_archived_timing_metadata_is_not_inferred_from_old_config(self):
        fresh = report_fixture(); old = copy.deepcopy(fresh)
        del old['execution_timing']
        with self.assertRaisesRegex(ValueError, 'changed fields: execution_timing'):
            probe.require_same_json(fresh, old, 'Archived schema changed')

    def test_replay_isolated_outputs_and_detects_archived_report_tampering(self):
        root = self.folder/'evidence'; root.mkdir()
        gate, report = {'passed': True, 'fingerprints': [('rank0', 'hash')]}, report_fixture()
        (root/'preflight.json').write_text(json.dumps(gate, sort_keys=True, indent=2)+'\n')
        report['evidence'] = dict(preflight_sha256=probe.sha(root/'preflight.json'))
        write(root/'report.json', report)
        (root/'report.md').write_text('original report\n')
        (root/'input.txt').write_text('immutable input\n')
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        def preflight(audit, offline):
            self.assertTrue(offline)
            self.assertTrue((audit/'input.txt').is_symlink())
            self.assertFalse((audit/'preflight.json').exists())
            write(audit/'preflight.json', gate)
            return gate
        def replay(audit, offline):
            value = dict(report, offline=offline)
            value['evidence'] = dict(preflight_sha256=probe.sha(audit/'preflight.json'))
            write(audit/'report.json', value); (audit/'report.md').write_text('recomputed\n')
            return value
        reporter = SimpleNamespace(preflight=preflight, report=replay)
        record = dict(root=root, reporter=reporter)
        probe.replay_report(record)
        self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
        tampered = dict(report, correctness_passed=False); write(root/'report.json', tampered)
        with self.assertRaisesRegex(ValueError, 'differs from archived'):
            probe.replay_report(record)


if __name__ == '__main__':
    unittest.main()
