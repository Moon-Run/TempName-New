"""CPU-only supervisor tests: serial gates, independent repeats and build failures."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import ablation6_protocol as a6
import ablation7_protocol as a7
import detach_ablation6_then7 as batch
from common import read, sha, write


class BatchTests(unittest.TestCase):
    def fake_invoke(self, calls, mutate=None):
        """Successful child processes with small synthetic report metadata, no workers."""
        def invoke(cmd, log, env):
            action, root = cmd[2], Path(cmd[3])
            family = Path(cmd[1]).stem
            calls.append((family, log.stem))
            protocol = a6 if family == 'ablation6' else a7
            windows = 50 if family == 'ablation6' else 36
            blocks = 5 if family == 'ablation6' else 6
            if action == 'prepare':
                root.mkdir()
                info = dict(protocol=protocol.PROTOCOL, run_id=family+'/'+root.name, executable=True, source_only=False,
                    execution_timing=dict(batch.TIMING), rounds=2, blocks_per_round=blocks, timing_windows=windows,
                    schedule=protocol.schedule(blocks, rounds=2), communication=dict(environment={'NCCL_NVLS_ENABLE': '0'}),
                    config={}, variants={}, calibration_hardware={}, megatron='same', megatron_commit='same',
                    candidate_source_commit='same', build_manifest={}, build_files={}, controllers={}, files={})
                if family == 'ablation6':
                    info.update(naive_codec_identity={'implementation': a6.NAIVE_CODEC}, flux_build_manifest={})
                else:
                    info['budget_fractions'] = {p: a7.definition(p)['budget_fraction'] for p in a7.POLICIES}
                write(root/'submission.json', info)
            elif action == 'run' and '--execute' in cmd:
                info = read(root/'submission.json')
                write(root/'all-state.json', dict(status='completed'))
                gate = dict(completed=True, protocol=protocol.PROTOCOL, execution_timing=batch.TIMING,
                            submission_sha256=sha(root/'submission.json'), communication=info['communication'])
                result = dict(protocol=protocol.PROTOCOL, run_id=info['run_id'], completed=True, correctness_passed=True,
                    execution_timing=dict(batch.TIMING), formal_optimizer_steps=windows*50, timing_windows=windows,
                    all_samples=[{}]*windows, round_count=2, blocks_per_round=blocks, timing_rank_records=windows*8,
                    preflight_rank_records=80 if family == 'ablation6' else 48,
                    profile_traces=40 if family == 'ablation6' else 24, candidate_shape_mappings=6,
                    communication=info['communication'], measurement_identity={'same_runtime': family},
                    role_plan_sha256={'frozen_plans': family})
                if family == 'ablation6':
                    gate['naive_codec_identity'] = result['naive_codec_identity'] = info['naive_codec_identity']
                write(root/'preflight.json', gate)
                result['evidence'] = dict(preflight_sha256=sha(root/'preflight.json'))
                write(root/'report.json', result)
            if mutate:
                mutate(family, log.stem, root, cmd)
        return invoke

    def args(self, root, build=False):
        flags = ['--out', str(root)]
        if build:
            nccl = root/'nccl'
            for name in ('include/nccl.h', 'lib/libnccl_static.a'):
                path = nccl/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic')
            flags += ['--nccl-root', str(nccl)]
        else:
            flags += ['--flux-build', '/already-built/flux', '--ablation7-build', '/already-built/a7']
        return batch.arguments(flags)

    def run_fake(self, args, calls, mutate=None):
        with mock.patch.object(batch.launcher, 'invoke', side_effect=self.fake_invoke(calls, mutate)), \
                contextlib.redirect_stdout(io.StringIO()):
            return batch.supervise(args)

    def test_preview_is_read_only_and_has_six_separate_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(os, 'fork', side_effect=AssertionError('fork')), \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('process')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(tmp)/'new'; batch.main(['--out', str(root)])
            self.assertFalse(root.exists())
            self.assertIn('12900 formal steps', output.getvalue())
            steps = batch.commands(batch.arguments(['--out', str(root)]))
            controllers = [(f, s) for f, s, _ in steps if s.endswith('-controller')]
            self.assertEqual(controllers, [(f, f'compare{i}-controller') for f in ('ablation6', 'ablation7') for i in (1, 2, 3)])
            self.assertEqual([f for f, s, _ in steps if s == 'compile'], ['ablation6', 'ablation7'])
            for family, label, cmd in steps:
                if label.endswith('-prepare') and label != 'build-prepare':
                    self.assertEqual(cmd[cmd.index('--blocks')+1], '5' if family == 'ablation6' else '6')
                    self.assertEqual(cmd[cmd.index('--rounds')+1], '2')
                    self.assertEqual(cmd[cmd.index('--nvls-enable')+1], '0')

    def test_all_six_reports_in_order_and_builds_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.args(root, build=True); calls = []
            result = self.run_fake(args, calls)
            expected = []
            for family in ('ablation6', 'ablation7'):
                expected += [(family, 'build-prepare'), (family, 'compile')]
                for i in (1, 2, 3):
                    expected += [(family, f'compare{i}-{stage}') for stage in ('prepare', 'dry-run', 'controller', 'report')]
            self.assertEqual(calls, expected)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual((result['timing_windows'], result['timing_rank_records'], result['formal_optimizer_steps']),
                             (258, 2064, 12900))
            self.assertEqual(result['experiments']['ablation6']['formal_optimizer_steps'], 7500)
            self.assertEqual(result['experiments']['ablation7']['formal_optimizer_steps'], 5400)
            for family in ('ablation6', 'ablation7'):
                for trial in ('compare1', 'compare2', 'compare3'):
                    record = result['experiments'][family]['comparisons'][trial]
                    self.assertEqual(record['report_sha256'], sha(root/family/trial/'report.json'))

    def test_failed_controller_stops_before_next_prepare(self):
        for family in ('ablation6', 'ablation7'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); calls = []
                def fail(f, stage, folder, cmd):
                    if f == family and stage == 'compare2-controller':
                        raise subprocess.CalledProcessError(1, cmd)
                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_fake(self.args(root), calls, fail)
                self.assertEqual(calls[-1], (family, 'compare2-controller'))
                self.assertFalse((root/family/'compare3').exists())
                self.assertEqual(read(root/'batch-state.json')['status'], 'failed')
                if family == 'ablation6':
                    self.assertFalse((root/'ablation7').exists())

    def test_invalid_third_report_never_advances_or_marks_batch_complete(self):
        for family, change in (('ablation6', {'execution_timing': {'warmup_steps': 10, 'timed_steps': 60}}),
                               ('ablation6', {'role_plan_sha256': {'changed': True}}),
                               ('ablation7', {'preflight_rank_records': 40}),
                               ('ablation7', {'measurement_identity': {'changed': True}}),
                               ('ablation7', {'evidence': {'preflight_sha256': 'stale'}})):
            with self.subTest(family=family, change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); calls = []
                def corrupt(f, stage, folder, cmd):
                    if f == family and stage == 'compare3-report':
                        write(folder/'report.json', dict(read(folder/'report.json'), **change))
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(root), calls, corrupt)
                self.assertEqual(calls[-1], (family, 'compare3-report'))
                self.assertEqual(read(root/'batch-state.json')['status'], 'failed')
                if family == 'ablation6':
                    self.assertFalse((root/'ablation7').exists())

    def test_third_run_is_distinct_and_matches_first_inputs(self):
        for family, key, value in (('ablation6', 'run_id', 'ablation6/compare2'),
                                   ('ablation6', 'naive_codec_identity', {}),
                                   ('ablation7', 'budget_fractions', {}),
                                   ('ablation7', 'build_files', {'drift': 'x'})):
            with self.subTest(family=family, key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); calls = []
                def corrupt(f, stage, folder, cmd):
                    if f == family and stage == 'compare3-prepare':
                        write(folder/'submission.json', dict(read(folder/'submission.json'), **{key: value}))
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(root), calls, corrupt)
                self.assertNotIn((family, 'compare3-controller'), calls)

    def test_build_failure_stops_at_correct_family(self):
        for family in ('ablation6', 'ablation7'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); calls = []
                def fail(f, stage, folder, cmd):
                    if f == family and stage == 'compile':
                        raise subprocess.CalledProcessError(1, cmd)
                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_fake(self.args(root, build=True), calls, fail)
                self.assertEqual(calls[-1], (family, 'compile'))
                self.assertFalse((root/family/'compare1').exists())
                if family == 'ablation7':
                    self.assertIn(('ablation6', 'compare3-report'), calls)

    def test_reuse_builds_omits_compilation_but_keeps_all_preflights(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []; self.run_fake(self.args(Path(tmp)), calls)
            self.assertNotIn('compile', [s for _, s in calls])
            self.assertEqual(sum(s.endswith('-controller') for _, s in calls), 6)

    def test_nccl_recorded_files_are_verified_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.args(root, build=True)
            nccl = args.nccl_root
            args.build = root/'parent'; (args.build/'fused').mkdir(parents=True)
            write(args.build/'fused/manifest.json', dict(nccl_files={
                str(nccl/name): sha(nccl/name) for name in ('include/nccl.h', 'lib/libnccl_static.a')}))
            args.nccl_root = None; batch.resolve_nccl(args)
            self.assertEqual(args.nccl_root, nccl)
            (nccl/'lib/libnccl_static.a').write_text('changed')
            args.nccl_root = None; calls = []
            with self.assertRaises(ValueError):
                self.run_fake(args, calls)
            self.assertEqual(calls, [])
            self.assertEqual(read(root/'batch-state.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
