"""CPU coverage for the one-trial G/CR2 experiment; never runs a GPU worker."""
import argparse
from collections import Counter
import contextlib
import copy
import gzip
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import detach_ma4_g as launcher
import ma4_g_bench as bench
import ma4_g_build as builder
import ma4_g_fullstep as full
from test_ma4_g import fixture_plan, fixture_rows


def pair_plan():
    plan = fixture_plan()
    plan.update(full.experiment(bench.PAIR_ARMS), trial_ids={'1': 'trial1'},
                arms=list(bench.PAIR_ARMS), schedule=full.expected_schedule(1, 8, bench.PAIR_ARMS))
    plan['libraries'] = {arm: plan['libraries'][arm] for arm in bench.PAIR_ARMS}
    return plan


def pair_rows(plan, job, digest):
    rows = fixture_rows(plan, job, digest)
    for row in rows:
        row['protocol'] = plan['protocol']
    return rows


class PairProtocolTests(unittest.TestCase):
    def test_default_and_pair_launch_workloads_and_read_only_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'untouched'
            args = launcher.arguments(['--pair', '--out', str(root)])
            tasks = dict(launcher.commands(args))
            for label in ('build-prepare', 'operator-prepare', 'fullstep-prepare'):
                self.assertIn('--pair', tasks[label])
            command = tasks['fullstep-prepare']
            self.assertEqual(command[command.index('--trials') + 1], '1')
            self.assertEqual(launcher.settings(True)['formal_steps'], 800)
            self.assertEqual(launcher.settings(True)['windows'], 16)
            self.assertEqual(launcher.settings()['formal_steps'], 3200)
            with contextlib.redirect_stdout(io.StringIO()) as preview:
                launcher.main(['--pair', '--out', str(root)])
            self.assertIn('800 formal optimizer steps', preview.getvalue())
            self.assertFalse(root.exists())
            reused = launcher.arguments(['--pair', '--out', str(root), '--ma4-g-build', str(root/'existing')])
            self.assertEqual(len(launcher.commands(reused)), 6)
        self.assertIn('ma4-g-pair-1x-50steps-', launcher.arguments(['--pair']).out.name)

    def test_schedule_and_every_worker_stage_exclude_c_r(self):
        plan = pair_plan()
        self.assertEqual(len(plan['schedule']), 16)
        for half in (plan['schedule'][:8], plan['schedule'][8:]):
            for arm in bench.PAIR_ARMS:
                self.assertEqual(Counter(j['index'] % 2 for j in half if j['arm'] == arm), {0: 2, 1: 2})
        self.assertEqual(Counter(j['trial'] for j in plan['schedule']), {1: 16})
        for stage, count in (('preflight', 2), ('profile', 2), ('timing', 16), ('all', 18)):
            jobs = list(full.jobs(plan, stage))
            self.assertEqual(len(jobs), count)
            self.assertEqual({j['arm'] for j in jobs}, set(bench.PAIR_ARMS))
        for stage in ('preflight', 'profile'):
            self.assertEqual([j['arm'] for j in bench.stage_jobs(stage, bench.PAIR_ARMS)], ['G', 'CR2'])
        with self.assertRaisesRegex(ValueError, 'not registered'):
            bench.execution_environment(plan, 'C')

    def test_pair_acceptance_does_not_claim_the_c_r_target(self):
        good = {'G/CR2': dict(percent=1.0, ci95=[.2, 1.5])}
        result = full.comparison_acceptance(good, bench.PAIR_ARMS)
        self.assertEqual(result, dict(target_validated=False, increment_over_CR2_validated=True))
        for estimate in (dict(percent=0., ci95=[-.1, .1]), dict(percent=1., ci95=[0., 2.]),
                         dict(percent=-1., ci95=[-2., -.1])):
            value = full.comparison_acceptance({'G/CR2': estimate}, bench.PAIR_ARMS)
            self.assertFalse(value['increment_over_CR2_validated'])
        self.assertEqual(full.aggregate_acceptance({'compare1': result}, bench.PAIR_ARMS), result)
        with self.assertRaises(ValueError):
            full.aggregate_acceptance({'compare1': result, 'compare2': result}, bench.PAIR_ARMS)
        with self.assertRaises(ValueError):
            full.comparison_acceptance(dict(good, **{'G/C': good['G/CR2']}), bench.PAIR_ARMS)

    def test_operator_report_and_profile_gate_need_only_pair_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(protocol=bench.PAIR_PROTOCOL, run_id='pair-operator-fixture',
                        arms=list(bench.PAIR_ARMS), schedule=bench.schedule(8, bench.PAIR_ARMS))
            bench.write(root/'plan.json', plan)
            samples = [dict(job, rankmax_wall_us=dict.fromkeys(bench.ROLES, 110. if job['arm']=='G' else 100.))
                       for job in plan['schedule']]
            seen = []

            def validate(root, plan, jobs, stage, reference=None):
                self.assertEqual({j['arm'] for j in jobs}, {'G', 'CR2'})
                seen.append(stage)
                return {'host': 'CPU-only'}, {}, {}, samples if stage=='timing' else []

            def trace(events, arm, role):
                self.assertIn(arm, bench.PAIR_ARMS)
                return dict(execution_path_validated=True)

            for arm in bench.PAIR_ARMS:
                dest = root/'results'/f'profile-{arm}'
                dest.mkdir(parents=True)
                for rank in range(8):
                    for role in bench.ROLES:
                        with gzip.open(dest/f'rank{rank}-{role}-profile.json.gz', 'wt') as handle:
                            json.dump({'traceEvents': []}, handle)
            with patch.object(bench, 'verify', return_value=plan), \
                 patch.object(bench, 'mapping_evidence', return_value={}), \
                 patch.object(bench, 'validate_rows', side_effect=validate), \
                 patch.object(bench, 'trace_path', side_effect=trace), \
                 contextlib.redirect_stdout(io.StringIO()):
                bench.preflight(root)
                report = bench.report(root)
                profile = bench.profile_summary(root)
                bench.write(root/'profile-summary.json', profile)
                result = launcher.check_operator(root, pair=True)
            self.assertTrue(result['operator_execution_path_validated'])
            self.assertEqual(report['equivalence_pairs'], [['G', 'CR2']])
            self.assertEqual(len(profile['profiles']), 32)
            for role in bench.ROLES:
                self.assertEqual(set(report['roles'][role]['comparisons']), {'G/CR2'})
                self.assertLess(report['roles'][role]['comparisons']['G/CR2']['percent'], 0.)
            self.assertIn('profile', seen)


class PairPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='ma4-pair-')
        cls.root = Path(cls.temporary.name)
        cls.build = builder.prepare(cls.root / 'build', pair=True)
        cls.identities = {arm: dict(root='/synthetic/'+arm, files={}) for arm in bench.PAIR_ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_build_has_only_g_cr2_and_preserves_native_sources(self):
        record = builder.verify(self.build)
        self.assertEqual(tuple(record['variants']), bench.PAIR_ARMS)
        self.assertFalse((self.build/'C').exists())
        self.assertFalse((self.build/'R').exists())
        reference = builder.read(self.build/'reference/manifest.json')
        self.assertEqual(record['variants']['CR2'], reference['variants']['CR2']['source_sha256'])
        with self.assertRaisesRegex(ValueError, 'Missing MA4-G/control runtime'):
            builder.verify(self.build, built=True)

    def test_operator_prepare_freezes_pair_workers_and_rejects_mixed_plan(self):
        root = self.root/'operator'
        args = argparse.Namespace(root=root, build=self.build, pair=True, blocks=8,
                                  warmup=100, iterations=200, seeds=[817, 919], reuse=2)
        with patch.object(bench, 'build_identity', return_value={}), \
             patch.object(bench, 'libraries', return_value=self.identities) as libraries, \
             contextlib.redirect_stdout(io.StringIO()):
            bench.prepare(args)
        libraries.assert_called_once_with(self.build, bench.PAIR_ARMS)
        plan = bench.verify(root, offline=True)
        self.assertEqual(plan['protocol'], bench.PAIR_PROTOCOL)
        self.assertEqual(plan['arms'], ['G', 'CR2'])
        self.assertEqual(len(plan['schedule']), 16)
        self.assertEqual((root/'scripts/bench.py').read_bytes(), Path(bench.__file__).read_bytes())
        bad = copy.deepcopy(plan)
        bad['libraries']['C'] = {}
        with patch.object(bench, 'read', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'libraries'):
                bench.verify(root, offline=True)
        bad = copy.deepcopy(plan)
        bad['protocol'] = bench.PROTOCOL
        with patch.object(bench, 'read', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'plan'):
                bench.verify(root, offline=True)

    def test_fullstep_prepare_freezes_one_trial_and_no_c_r_build_links(self):
        root = self.root/'training'
        args = argparse.Namespace(root=root, build=self.build, megatron=self.root/'megatron',
                                  pair=True, trials=None, blocks=8)
        with patch.object(bench, 'build_identity', return_value={}), \
             patch.object(bench, 'libraries', return_value=self.identities) as libraries, \
             patch.object(full, 'snapshot_megatron', return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            full.prepare(args)
        libraries.assert_called_once_with(self.build, bench.PAIR_ARMS)
        plan = full.verify(root, offline=True)
        self.assertEqual(plan['protocol'], full.PAIR_PROTOCOL)
        self.assertEqual(plan['trials'], 1)
        self.assertIsNone(plan['target_percent'])
        self.assertEqual({p.name for p in (root/'build').iterdir()}, {'G', 'CR2'})
        self.assertEqual(bench.read(root/'scripts/config.json')['policies'], [full.C_POLICY])
        for key, value in (('trials', 2), ('protocol', full.PROTOCOL), ('target_percent', 2.)):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with patch.object(full, 'read', return_value=bad):
                with self.assertRaises(ValueError):
                    full.verify(root, offline=True)

    def test_wrong_trial_count_fails_before_build_checks(self):
        for pair, trials in ((True, 2), (False, 1)):
            args = argparse.Namespace(root=self.root/'invalid', build=self.build, megatron=self.root,
                                      pair=pair, trials=trials, blocks=8)
            with patch.object(bench, 'build_identity') as build_check:
                with self.assertRaisesRegex(ValueError, 'one trial'):
                    full.prepare(args)
                build_check.assert_not_called()


class PairReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = pair_plan()
        bench.write(self.root/'plan.json', self.plan)
        self.digest = bench.sha(self.root/'plan.json')

    def write_rows(self, slow=False):
        for job in full.jobs(self.plan, 'all'):
            directory = self.root/'results'/job['directory']
            directory.mkdir(parents=True)
            for row in pair_rows(self.plan, job, self.digest):
                if slow and job['arm'] == 'G':
                    row['elapsed_seconds'] = .6 * row['timed_steps']
                bench.write(directory/f"rank{row['rank']}.json", row)

    def test_raw_rank_records_report_one_pair_and_pass_batch_gate(self):
        self.write_rows()
        with patch.object(full, 'verify', return_value=self.plan), contextlib.redirect_stdout(io.StringIO()):
            report = full.report(self.root)
            self.assertEqual(launcher.fullstep_gate(self.root, pair=True), report)
        self.assertEqual(report['formal_optimizer_steps'], 800)
        self.assertEqual(report['timing_rank_records'], 128)
        self.assertEqual(set(report['comparisons']), {'compare1'})
        combined = report['comparisons']['compare1']['rounds']['combined']
        self.assertEqual(set(combined['comparisons']), {'G/CR2'})
        self.assertEqual(combined['comparisons']['G/CR2']['paired_blocks'], 8)
        self.assertEqual(report['equivalence_pairs'], [['G', 'CR2']])
        self.assertTrue(report['increment_over_CR2_validated'])
        self.assertFalse(report['target_evaluated'])
        self.assertFalse(report['target_validated'])
        self.assertIn('未测 C/R', (self.root/'report.md').read_text())
        with patch.object(full, 'verify', return_value=self.plan):
            with self.assertRaises(ValueError):
                launcher.fullstep_gate(self.root)

    def test_negative_increment_preserves_all_windows_and_finishes_report(self):
        self.write_rows(slow=True)
        with patch.object(full, 'verify', return_value=self.plan):
            report = full.report(self.root, save=False)
        self.assertEqual(len(report['all_samples']), 16)
        self.assertFalse(report['increment_over_CR2_validated'])
        self.assertEqual(report['acceptance_status'], 'CR2_increment_not_validated')

    def test_missing_rank_and_changed_model_parameters_fail(self):
        self.write_rows()
        path = self.root/'results'/self.plan['schedule'][0]['directory']/'rank0.json'
        original = bench.read(path)
        bad = copy.deepcopy(original)
        bad['ma4_g_final_parameters_sha256'] = 'd'*64
        path.write_text(json.dumps(bad))
        with patch.object(full, 'verify', return_value=self.plan):
            with self.assertRaisesRegex(ValueError, 'trajectories'):
                full.report(self.root, save=False)
        path.unlink()
        with patch.object(full, 'verify', return_value=self.plan):
            with self.assertRaises(FileNotFoundError):
                full.report(self.root, save=False)


if __name__ == '__main__':
    unittest.main()
