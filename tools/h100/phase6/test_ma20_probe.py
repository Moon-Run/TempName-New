"""CPU protocol checks; synthetic records are never GPU/performance evidence."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import ma20_probe as probe


def metrics():
    return {b: {arm: 100.0 + b for arm in probe.ARMS} for b in range(14)}


def mask_evidence():
    record = dict(mask_sha256='a' * 64, selected_per_source=[7] * 8)
    evidence = [dict(slot=i, device_readback=True, dtype='torch.uint8', shape=[8, 512],
                     mask_sha256=record['mask_sha256'], selected_per_source=record['selected_per_source'])
                for i in range(2)]
    return record, evidence


def correctness():
    return dict(passed=True, budget_vs_bf16=.05, budget_vs_codec=.01,
                relative_l2=.001, protocol_relative_l2=.0001, max_abs=.01)


def smoke_role():
    record, masks = mask_evidence()
    value = dict(shape=probe.ROLES['attn'], gpu_mask_before=copy.deepcopy(masks),
                 gpu_mask_after=copy.deepcopy(masks), wall_us=None,
                 checks=[dict(seed=seed, case=case, correctness=correctness())
                         for seed in probe.SEEDS for case in probe.CASES],
                 retained=[dict(generation=i, passed=True, output=str(i), reference=str(i), pointer=i + 1)
                           for i in range(6)],
                 consumer=[dict(slot=i, protocol_version=1, failed=False, initializations=1,
                                successful_calls=7, bias_residual_calls=7, bound_k=512,
                                local_packet_ptr=100 + i) for i in range(2)])
    return record, value


class CalibrationTests(unittest.TestCase):
    def test_every_split_balances_arm_positions(self):
        jobs = probe.schedule()
        self.assertEqual(len(jobs), 98)
        for split in ('train', 'heldout'):
            selected = [job for job in jobs if job['split'] == split]
            self.assertEqual(len(selected), 49)
            for arm in probe.ARMS:
                self.assertEqual(sorted(j['index'] % 7 for j in selected if j['arm'] == arm), list(range(7)))
        self.assertEqual(len(probe.smoke_jobs()), 7)

    def test_training_selects_once_and_heldout_confirms(self):
        values = metrics()
        for b in values:
            values[b]['Q28'] *= .9
            values[b]['Q7'] *= .95
        selected = probe.choose(values)
        self.assertEqual(selected['candidate'], 'Q28')
        self.assertEqual(selected['proposed_candidate'], 'Q28')
        self.assertGreater(selected['heldout']['ci95'][0], 0)

    def test_heldout_cannot_select_the_second_candidate(self):
        values = metrics()
        for b in range(7):
            values[b]['Q7'] *= .9
            values[b]['Q14'] *= .95
        for b in range(7, 14):
            values[b]['Q7'] *= 1.1
            values[b]['Q14'] *= .5
        selected = probe.choose(values)
        self.assertIsNone(selected['candidate'])
        self.assertEqual(selected['proposed_candidate'], 'Q7')
        self.assertEqual(selected['reason'], 'heldout_rejected_no_reselection')

    def test_zero_quantization_is_eligible_but_no_gain_stays_null(self):
        self.assertIsNone(probe.choose(metrics())['candidate'])
        values = metrics()
        for b in values:
            values[b]['Q0'] *= .9
        self.assertEqual(probe.choose(values)['candidate'], 'Q0')

    def test_incomplete_blocks_and_nonfinite_timing_rejected(self):
        values = metrics(); del values[13]['Q112']
        with self.assertRaisesRegex(ValueError, 'fourteen complete'):
            probe.choose(values)
        values = metrics(); values[3]['Q7'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid selection latency'):
            probe.choose(values)

    def test_missing_rank_record_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ma20-probe-missing-') as tmp:
            with self.assertRaisesRegex(ValueError, 'Missing MA20 rank'):
                probe.collect(Path(tmp), {}, probe.smoke_jobs())

    def test_smoke_and_timing_must_use_the_same_ordered_gpu_identity(self):
        hardware = dict(host='test-host', devices=[dict(uuid='GPU-' + str(i), name='NVIDIA H100', sm_count=132)
                                                 for i in range(8)])
        with tempfile.TemporaryDirectory(prefix='ma20-probe-machine-') as tmp:
            root = Path(tmp)
            (root / 'plan.json').write_text('{}\n')
            record, _ = mask_evidence()
            plan = dict(run_id='cpu-validation-fixture', libraries={'MA11': {'files': {}}},
                        bank={'configurations': {'MA11': {'roles': {'attn': record, 'mlp': record}}}})
            jobs = [probe.smoke_jobs()[0], probe.schedule()[0]]
            for job in jobs:
                dest = root / 'results' / job['directory']; dest.mkdir(parents=True)
                for rank in range(8):
                    roles = {}
                    for role, shape in probe.ROLES.items():
                        _, value = smoke_role(); value['shape'] = shape
                        for state in value['consumer']:
                            state['bound_k'] = shape[2]
                        if job['split'] != 'smoke':
                            value.update(checks=[], retained=[], wall_us=100.0,
                                         final_correctness=correctness(), inputs=['fixture-input'])
                            for state in value['consumer']:
                                state.update(successful_calls=60, bias_residual_calls=60)
                        roles[role] = value
                    row = dict(protocol=probe.PROTOCOL, run_id=plan['run_id'],
                               plan_sha256=probe.sha(root / 'plan.json'), job=job, rank=rank,
                               completed=True, loaded_libraries={}, environment=probe.ENVIRONMENT,
                               hardware=hardware, software={'torch': 'fixture'}, roles=roles)
                    (dest / f'rank{rank}.json').write_text(json.dumps(row))
            values, _ = probe.collect(root, plan, jobs)
            self.assertEqual(values, {0: {'MA11': 200.0}})
            path = root / 'results' / jobs[1]['directory'] / 'rank0.json'
            row = probe.read(path); row['hardware']['devices'][0]['uuid'] = 'different-GPU'
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'hardware/software changed'):
                probe.collect(root, plan, jobs)

    def test_hardware_normalization_retains_gpu_order_and_ignores_capability_extra(self):
        hardware = dict(host='test-host', devices=[dict(uuid='GPU-' + str(i), name='NVIDIA H100', sm_count=132)
                                                 for i in range(8)])
        extra = copy.deepcopy(hardware)
        for device in extra['devices']:
            device['capability'] = [9, 0]
        self.assertEqual(probe.hardware_identity(extra), hardware)
        extra['devices'].reverse()
        self.assertNotEqual(probe.hardware_identity(extra), hardware)

    def test_frozen_probe_loads_builder_common_without_importing_torch(self):
        with tempfile.TemporaryDirectory(prefix='ma20-frozen-import-') as tmp:
            root = Path(tmp); scripts = root / 'scripts'; scripts.mkdir()
            builder = root / 'builder'; builder.mkdir(); bank = root / 'bank'; bank.mkdir()
            shutil.copy2(probe.__file__, scripts / 'ma20_probe.py')
            shutil.copy2(Path(probe.helpers.__file__), scripts / 'ma5_bench.py')
            (builder / 'common.py').write_text('TOKEN = "correct-builder-common"\n')
            (builder / 'ma20_build.py').write_text(
                'import sys\nimport common\nMANIFEST="ma20-manifest.json"\nPLAN="ma20-plan.json"\n'
                'def verify(root,built=False):\n'
                '    assert built and common.TOKEN=="correct-builder-common"\n'
                '    assert "torch" not in sys.modules\n'
                '    return {"inputs":{}}\n')
            for name in ('manifest.json', 'ma20-manifest.json', 'ma20-plan.json'):
                (bank / name).write_text('{}\n')
            code = ('import sys\nfrom pathlib import Path\nimport ma20_probe\n'
                    'value=ma20_probe.identity(Path(sys.argv[1]),Path(sys.argv[2]))\n'
                    'assert value["builder_root"]==sys.argv[2]\nassert "torch" not in sys.modules\n')
            subprocess.run([sys.executable, '-c', code, str(bank), str(builder)], cwd=scripts,
                           env=dict(os.environ, PYTHONPATH=str(scripts)), check=True, capture_output=True)

    def test_gpu_readback_is_required_for_both_slots_and_exact_counts(self):
        record, evidence = mask_evidence()
        probe.validate_gpu_mask(evidence, record)
        for altered in (None, evidence[:1], [dict(evidence[0], device_readback=False), evidence[1]],
                        [dict(evidence[0], selected_per_source=[6] * 8), evidence[1]]):
            with self.assertRaisesRegex(ValueError, 'GPU mask'):
                probe.validate_gpu_mask(altered, record)

    def test_smoke_requires_real_codec_metrics_reuse_and_consumer_calls(self):
        record, value = smoke_role()
        probe.validate_role(value, record, True)
        bad = copy.deepcopy(value); bad['checks'][0]['correctness']['protocol_relative_l2'] = .02
        with self.assertRaisesRegex(ValueError, 'codec correctness'):
            probe.validate_role(bad, record, True)
        bad = copy.deepcopy(value); bad['retained'][2]['output'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'retained output'):
            probe.validate_role(bad, record, True)
        bad = copy.deepcopy(value); bad['consumer'][0]['bias_residual_calls'] = 0
        with self.assertRaisesRegex(ValueError, 'fallback'):
            probe.validate_role(bad, record, True)

    def test_actual_independent_consumer_oracle_is_frozen_and_compiles(self):
        with tempfile.TemporaryDirectory(prefix='ma20-probe-reference-') as tmp:
            root = Path(tmp)
            probe.freeze_reference(root)
            source = (root / 'codec_reference.py').read_text()
            compile(source, str(root / 'codec_reference.py'), 'exec')
            functions = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
            self.assertEqual(functions, {'masked_roundtrip', 'check_consumer_output'})
            self.assertIn('bf16 = residual + (bf16 + bias)', source)
            self.assertIn('coded = residual + (coded + bias)', source)
            self.assertNotIn('from routing import', source)
            self.assertTrue((root / 'reference.py').is_file())


if __name__ == '__main__':
    unittest.main()
