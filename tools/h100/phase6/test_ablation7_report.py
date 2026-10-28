"""CPU-only report admission and synthetic full 10+50-step evidence checks."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ablation4_build
import ablation7_build as builder
import ablation7_prepare as preparation
import ablation7_protocol as routing
import ablation7_report as report
from common import read, sha, write
import scope_plan
import test_scope as fixtures


def jobs(info):
    for policy in routing.POLICIES:
        for mode in ('smoke', 'profile'):
            yield dict(policy=policy, mode=mode, block=-1, index=-1, directory=f'{mode}-{policy}')
    for job in info['schedule']:
        yield dict(job, mode='timing')


def synthetic_evidence(root):
    """Never enable generated source-only workers; mock only GPU admission."""
    info = read(root/'submission.json')
    info.update(executable=True, source_only=False, calibration_hardware=scope_plan.frozen_mlp()['hardware'])
    for policy, variant in info['variants'].items():
        variant['libraries'] = {name: dict(path='/CPU-FIXTURE/'+policy+'/'+name, sha256='f'*64)
                                for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
    write(root/'submission.json', info)
    write(root/'start-gpu-check.json', dict(passed=True, run_id=info['run_id'], synthetic=True))
    for job in jobs(info):
        policy = job['policy']; original_read = fixtures.read
        def per_policy(path):
            data = original_read(path)
            return dict(roles=routing.role_plans(data, policy)) if Path(path).name == 'selection-plan.json' else data
        with mock.patch.object(fixtures, 'sp', routing), mock.patch.object(fixtures, 'read', side_effect=per_policy):
            fixtures.make_window(root, info, job)
        folder = root/'results'/job['directory']
        write(folder/'command.json', dict(job, scope=routing.definition(policy), communication=info['communication']))
        for rank in range(8):
            path = folder/f'rank{rank}.json'; row = read(path)
            row.update(communication=info['communication'], execution_timing=dict(routing.TIMING))
            if routing.is_fused_v2(policy):
                for check in row['selection_checks'].values():
                    check.update(mask_sha256=None, plan_sha256=None, wire_bytes=7*256*32*136)
                row['workspace_checks'] = {name: [dict(shape=slots[0]['shape'], packet_ptr=slots[0]['packet_ptr'])]
                                           for name, slots in row['workspace_checks'].items()}
            write(path, row)
    plans = read(root/'scripts/selection-plan.json')
    for policy in routing.POLICIES:
        for role, record in routing.role_plans(plans, policy).items():
            with (root/'results'/f'mapping-{policy}-{role}.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['rank', 'index', 'm', 'n', 'destination', 'sm_occupancy'])
                writer.writeheader()
                for rank, coords in enumerate(record['entry']['coords']):
                    for index, tile in enumerate(coords):
                        writer.writerow(dict(rank=rank, index=index, m=tile//32, n=tile%32,
                                             destination=tile//64, sm_occupancy=2))
    return info


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='ablation7-REPORT-CPU-FIXTURE-')
        cls.root = Path(cls.tmp.name)
        parent = ablation4_build.prepare(cls.root/'parent', source_only=True)
        cls.build = builder.prepare(cls.root/'build', parent, source_only=True)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, '/absent', source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_source_only_is_verifiable_but_cannot_certify_gpu(self):
        info = report.verify_inputs(self.compare)
        self.assertTrue(info['source_only'])
        self.assertEqual(info['execution_timing'], dict(warmup_steps=10, timed_steps=50))
        self.assertEqual(info['timing_windows'], 36)
        self.assertEqual(info['budget_fractions'], dict(zip(routing.POLICIES, (1/16, 1, 1))))
        report.verify_inputs(self.compare, offline=True)
        for action in (report.preflight, report.report):
            with self.assertRaisesRegex(ValueError, 'CPU rehearsal'):
                action(self.compare, offline=True)
        path = self.compare/'submission.json'; saved = read(path)
        try:
            write(path, dict(saved, executable=True, source_only=False))
            with self.assertRaises(ValueError): report.verify_inputs(self.compare, offline=True)
        finally:
            write(path, saved)

    def test_offline_uses_snapshots_when_original_builds_are_unavailable(self):
        parent = self.root/'parent'; parent_hidden = self.root/'parent-unavailable'
        build_hidden = self.root/'build-unavailable'
        parent.rename(parent_hidden); self.build.rename(build_hidden)
        try:
            info = report.verify_inputs(self.compare, offline=True)
            self.assertEqual(info['protocol'], routing.PROTOCOL)
            with self.assertRaises((FileNotFoundError, ValueError)):
                report.verify_inputs(self.compare, offline=False)
        finally:
            build_hidden.rename(self.build); parent_hidden.rename(parent)

    def test_duration_budget_schedule_and_source_hash_drift_rejected(self):
        path = self.compare/'submission.json'; saved = read(path)
        changed_schedule = copy.deepcopy(saved['schedule'])
        changed_schedule[0]['policy'] = changed_schedule[1]['policy']
        for field, value in (('execution_timing', dict(warmup_steps=10, timed_steps=60)),
                             ('budget_fractions', dict(zip(routing.POLICIES, (1/64, 1, 1)))),
                             ('schedule', changed_schedule)):
            try:
                write(path, dict(saved, **{field: value}))
                with self.assertRaises(ValueError): report.verify_inputs(self.compare, offline=True)
            finally:
                write(path, saved)
        worker = self.compare/'scripts/worker.py'; content = worker.read_bytes()
        try:
            worker.write_bytes(content+b'\n')
            with self.assertRaises(ValueError): report.verify_inputs(self.compare, offline=True)
        finally:
            worker.write_bytes(content)
        snapshot = self.compare/'provenance/parent-ablation4-manifest.json'; content = snapshot.read_bytes()
        try:
            snapshot.write_bytes(content+b'\n')
            changed = copy.deepcopy(saved)
            changed['files']['provenance/parent-ablation4-manifest.json'] = sha(snapshot)
            write(path, changed)
            with self.assertRaisesRegex(ValueError, 'Frozen build snapshot'):
                report.verify_inputs(self.compare, offline=True)
        finally:
            snapshot.write_bytes(content); write(path, saved)
        config_path = self.compare/'scripts/config.json'; config = read(config_path)
        try:
            changed = dict(config, warmup_steps=10, timed_steps=60,
                           execution_timing=dict(warmup_steps=10, timed_steps=60))
            write(config_path, changed)
            with self.assertRaisesRegex(ValueError, '10 warmup / 50'):
                report.validate_timing(dict(saved, execution_timing=changed['execution_timing']), changed)
        finally:
            write(config_path, config)

    def test_physical_mask_budget_local_bf16_and_all_remote(self):
        plans = read(self.compare/'scripts/selection-plan.json')
        metadata = report.budget_metadata(plans)
        self.assertEqual(metadata[routing.POLICIES[0]]['max_remote_tiles_per_source'], 28)
        for role in routing.SHAPES:
            self.assertEqual(metadata[routing.POLICIES[1]]['roles'][role]['selected_per_source'], [448]*8)
        for policy, count in ((routing.POLICIES[0], 29), (routing.POLICIES[1], 447)):
            changed = copy.deepcopy(plans)
            record = routing.role_plans(changed, policy)['attn']
            record['entry']['mask'][0] = [int(64 <= tile < 64+count) for tile in range(512)]
            with self.assertRaisesRegex(ValueError, '1/16 budget'):
                report.budget_metadata(changed)
        changed = copy.deepcopy(plans)
        routing.role_plans(changed, routing.POLICIES[0])['mlp']['entry']['mask'][0][0] = 1
        with self.assertRaisesRegex(ValueError, 'Local contributions'):
            report.budget_metadata(changed)

    def test_profile_launches_and_exact_gpu_order_are_required(self):
        plans = read(self.compare/'scripts/selection-plan.json')
        for policy in routing.POLICIES:
            with mock.patch.object(fixtures, 'sp', routing): events = fixtures.profile_fixture(policy)
            self.assertEqual(set(report.profile_roles(events, policy)), set(routing.SHAPES))
            for bad in ([e for e in events if e.get('name') != 'taco_decode_ring_kernel'],
                        [e for e in events if e.get('name') != 'scope.attn.candidate']):
                with self.assertRaises(ValueError): report.profile_roles(bad, policy)
            changed = copy.deepcopy(events)
            for event in changed:
                if event.get('name') == 'flux_bf16_sm80': event['name'] = 'flux_bf16_sm90'
            with self.assertRaises(ValueError): report.profile_roles(changed, policy)
            for role, record in routing.role_plans(plans, policy).items():
                rows = [dict(rank=rank, index=index, m=tile//32, n=tile%32, destination=tile//64, sm_occupancy=2)
                        for rank, coords in enumerate(record['entry']['coords']) for index, tile in enumerate(coords)]
                report.verify_mapping(rows, policy, role, plans)
                changed = copy.deepcopy(rows); changed[0]['m'], changed[1]['m'] = changed[1]['m'], changed[0]['m']
                changed[0]['n'], changed[1]['n'] = changed[1]['n'], changed[0]['n']
                with self.assertRaisesRegex(ValueError, 'GPU mapping'): report.verify_mapping(changed, policy, role, plans)
                with self.assertRaisesRegex(ValueError, 'Incomplete GPU'): report.verify_mapping(rows[:-1], policy, role, plans)

    def test_full_three_policy_report_and_rank_evidence_failures(self):
        root = preparation.prepare(self.root/'SYNTHETIC-NO-GPU', self.build, '/absent', source_only=True)
        info = synthetic_evidence(root)
        # Only GPU build admission is mocked. All evidence is temporary and the
        # generated source-only workers retain their permanent execution stop.
        with mock.patch.object(report, 'verify_inputs', return_value=info):
            gate = report.preflight(root, offline=True)
            result = report.report(root, offline=True)
            self.assertTrue(result['completed']); self.assertEqual(result['timing_windows'], 36)
            self.assertEqual(result['formal_optimizer_steps'], 1800)
            self.assertEqual(result['timing_rank_records'], 288)
            self.assertEqual(result['preflight_rank_records'], 48)
            self.assertEqual(result['profile_traces'], 24); self.assertEqual(result['candidate_shape_mappings'], 6)
            self.assertEqual(result['rounds']['combined']['blocks'], 12)
            self.assertEqual(len(result['rounds']['combined']['comparisons']), 3)
            self.assertEqual(result['rounds']['1']['mean_ms'][routing.POLICIES[1]], 1000.)
            self.assertEqual(result['rounds']['combined']['mean_loss'][routing.POLICIES[0]], 1.)
            self.assertEqual(gate['execution_timing'], routing.TIMING)
            for policy in routing.POLICIES:
                job = next(j for j in info['schedule'] if j['policy'] == policy)
                path = root/'results'/job['directory']/'rank7.json'; saved = read(path)
                self.assertEqual(set(saved['timed_calls'].values()), {200})
                self.assertEqual(set(saved['warmup_calls'].values()), {40})
                changes = [('losses', saved['losses'][:20]), ('timed_steps', 60), ('warmup_steps', 9),
                           ('timed_calls', {}), ('fallback_calls', 1), ('skips', 1), ('sampling', True),
                           ('execution_timing', dict(warmup_steps=10, timed_steps=60)),
                           ('communication', {}), ('post_audit_routes', {})]
                for field, value in changes:
                    write(path, dict(saved, **{field: value}))
                    with self.subTest(policy=policy, field=field), self.assertRaises(ValueError):
                        report.window(root, path.parent, policy, 'timing', job['block'], job['index'])
                changed = copy.deepcopy(saved)
                next(iter(changed['selection_checks'].values()))['wire_bytes'] += 1
                write(path, changed)
                with self.assertRaisesRegex(ValueError, 'physical mask'):
                    report.window(root, path.parent, policy, 'timing', job['block'], job['index'])
                write(path, saved)
            job = info['schedule'][0]; path = root/'results'/job['directory']/'rank0.json'; saved = read(path)
            write(path, dict(saved, tokens_sha256='different'))
            with self.assertRaisesRegex(ValueError, 'Timing initialization'): report.report(root, offline=True)
            path.unlink()
            with self.assertRaises(FileNotFoundError): report.report(root, offline=True)
            write(path, saved)
            extra = root/'results/round3/window-extra'; extra.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Missing or extra'): report.report(root, offline=True)
            extra.rmdir(); extra.parent.rmdir()
            # Hash-registered preflight evidence must remain frozen for reports.
            smoke = root/'results'/f'smoke-{routing.POLICIES[0]}'/'rank0.json'; saved = read(smoke)
            changed = copy.deepcopy(saved); next(iter(changed['forward_checks'].values()))['passed'] = False
            write(smoke, changed)
            with self.assertRaises(ValueError): report.preflight(root, offline=True)
            with self.assertRaises(ValueError): report.report(root, offline=True)
            write(smoke, saved)


if __name__ == '__main__':
    unittest.main()
