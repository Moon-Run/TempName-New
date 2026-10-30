"""CPU-only checks for read-only preview and one/two-comparison completion gates."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import mav2_protocol as routing
import detach_mav2 as launch
from common import read, sha, write


def submission(trial):
    info = dict(protocol=routing.PROTOCOL, run_id=trial, execution_timing=routing.TIMING,
                allow_quality_failure=False,
                executable=True, source_only=False, rounds=2, blocks_per_round=6,
                timing_windows=36, schedule=routing.schedule(),
                variants={p: routing.definition(p) for p in routing.POLICIES},
                communication={'environment': {'NCCL_NVLS_ENABLE': '0'}},
                files={'scripts/worker.py': 'CPU-FIXTURE'})
    for key in ('config', 'calibration_hardware', 'megatron', 'megatron_commit',
                'build_manifest', 'flux_build_manifest', 'codec_build_manifest',
                'codec_identity', 'codec_library', 'fused_identity',
                'shared_environment', 'hardware', 'controllers'):
        info[key] = {'fixture': key}
    return info


def report_fixture(root, info):
    # These files stay in a test TemporaryDirectory and never describe GPU work.
    gate = dict(protocol=routing.PROTOCOL, completed=True,
                allow_quality_failure=info['allow_quality_failure'],
                submission_sha256=sha(root/'submission.json'))
    gate.update({k: info[k] for k in ('communication', 'codec_identity', 'codec_library')})
    write(root/'preflight.json', gate)
    write(root/'all-state.json', {'status': 'completed'})
    result = dict(protocol=routing.PROTOCOL, completed=True, correctness_passed=True,
                  allow_quality_failure=info['allow_quality_failure'],
                  quality_by_candidate={'mav2': True, 'd8': True},
                  execution_path_validated=True, quality_within_pairs=True, equivalence_pairs=routing.EQUIVALENCE_PAIRS,
                  run_id=info['run_id'], execution_timing=routing.TIMING,
                  round_count=2, blocks_per_round=6, timing_windows=36,
                  all_samples=[{'synthetic': True} for _ in range(36)],
                  formal_optimizer_steps=1800, timing_rank_records=288,
                  preflight_rank_records=48, profile_traces=24, candidate_shape_mappings=0,
                  evidence={'preflight_sha256': sha(root/'preflight.json')},
                  measurement_identity={'synthetic': True}, expected_order_validated=False)
    result.update({k: info[k] for k in ('communication', 'codec_identity', 'codec_library')})
    write(root/'report.json', result)
    return result


class LauncherTests(unittest.TestCase):
    def test_shell_wrapper_loads_recovery_environment_and_forwards_arguments(self):
        with tempfile.TemporaryDirectory(prefix='basic quant wrapper ') as tmp:
            root = Path(tmp)
            phase = root/'tools/h100/phase6'
            phase.mkdir(parents=True)
            (phase/'run_mav2.sh').write_text((launch.HERE/'run_mav2.sh').read_text())
            recovery = root/'outputs/h100/container-recovery'
            recovery.mkdir(parents=True)
            import sys
            import shlex
            (recovery/'env.sh').write_text('export H100_RECOVERY_PYTHON='+shlex.quote(sys.executable)+'\n'
                'export H100_RECOVERY_BUILD_CUDA=/recovered/build-cuda\n'
                'export CUDA_HOME=/recovered/runtime-cuda\nexport CC=/recovered/gcc\n')
            (phase/'detach_mav2.py').write_text('import os,sys,json\n'
                'print(json.dumps(dict(args=sys.argv[1:],cwd=os.getcwd(),cc=os.environ["CC"])))\n')
            result = subprocess.run(['bash', str(phase/'run_mav2.sh'), '--trials', '1', '--execute',
                                     '--out', 'path with spaces'], capture_output=True, text=True, check=True)
            record = json.loads(result.stdout)
            self.assertEqual(record['args'], ['--cuda-home', '/recovered/runtime-cuda', '--build-cuda-home',
                '/recovered/build-cuda', '--trials', '1', '--execute', '--out', 'path with spaces'])
            self.assertEqual(record['cc'], '/recovered/gcc')
            self.assertEqual(record['cwd'], str(root))

    def test_preview_never_creates_batch_or_starts_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'must-not-exist'
            with mock.patch.object(launch.launcher, 'invoke', side_effect=AssertionError('subprocess')), \
                 mock.patch.object(launch.launcher, 'detach', side_effect=AssertionError('fork')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                launch.main(['--out', str(out)])
            self.assertFalse(out.exists())
            self.assertIn('36 windows / 1800 formal steps', output.getvalue())

    def test_negative_ordering_still_finishes(self):
        self.exercise(False, trials=2)

    def test_single_comparison_finishes_without_compare2(self):
        self.exercise(False, trials=1)

    def test_opt_in_finishes_both_trials_with_failed_quality_status(self):
        self.exercise(False, trials=2, quality_failed=True)

    def test_quality_opt_in_is_explicit_and_forwarded_only_to_preparation(self):
        for enabled in (False, True):
            args = launch.arguments(['--out', '/CPU/batch'] + (['--allow-quality-failure'] if enabled else []))
            commands = dict(launch.commands(args))
            self.assertEqual('--allow-quality-failure' in commands['compare1-prepare'], enabled)
            self.assertNotIn('--allow-quality-failure', commands['compare1-controller'])

    def test_single_preview_and_directory_describe_one_comparison(self):
        args = launch.arguments(['--trials', '1'])
        self.assertIn('mav2-three-arm-1x-50steps-', args.out.name)
        commands = dict(launch.commands(args))
        self.assertFalse(any(key.startswith('compare2-') for key in commands))
        self.assertEqual(sum(key.endswith('-controller') for key in commands), 1)
        prepare = commands['compare1-prepare']
        self.assertEqual(prepare[prepare.index('--rounds')+1], '2')
        self.assertEqual(prepare[prepare.index('--blocks')+1], '6')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'single-preview'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                launch.main(['--out', str(out), '--trials', '1'])
            self.assertIn('36 windows / 1800 formal steps', output.getvalue())
            self.assertNotIn('compare2', output.getvalue())
            self.assertFalse(out.exists())

    def test_incomplete_first_report_prevents_second_training(self):
        self.exercise(True, trials=2)

    def exercise(self, broken, trials=1, quality_failed=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'handoff').mkdir()
            write(root/'handoff/controllers.json', {})
            args = launch.arguments(['--out', str(root), '--trials', str(trials), '--flux-build', '/CPU/flux', '--codec-build', '/CPU/codec']
                                    + (['--allow-quality-failure'] if quality_failed else []))
            seen = []

            def invoke(command, log, env):
                label = log.stem
                seen.append(label)
                self.assertEqual(env['CUDA_DEVICE_MAX_CONNECTIONS'], '1')
                self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
                trial = label.split('-')[0]
                folder = root/trial
                if label.endswith('-prepare'):
                    folder.mkdir()
                    write(folder/'submission.json', dict(submission(trial), allow_quality_failure=quality_failed))
                if label.endswith('-report'):
                    result = report_fixture(folder, read(folder/'submission.json'))
                    if quality_failed:
                        result.update(quality_within_pairs=False, correctness_passed=False,
                                      quality_by_candidate={'mav2': False, 'd8': False})
                        write(folder/'report.json', result)
                        write(folder/'all-state.json', {'status': 'completed_with_quality_failure'})
                    if broken and trial == 'compare1':
                        result['all_samples'].pop()
                        write(folder/'report.json', result)

            with mock.patch.object(launch.launcher, 'invoke', side_effect=invoke), \
                 mock.patch('mav2_report.report', side_effect=lambda root, save=False: read(root/'report.json')), \
                 contextlib.redirect_stdout(io.StringIO()):
                if broken:
                    with self.assertRaisesRegex(ValueError, 'Incomplete'):
                        launch.supervise(args)
                    self.assertFalse(any(s.startswith('compare2-') for s in seen))
                    self.assertEqual(read(root/'batch-state.json')['status'], 'failed')
                else:
                    result = launch.supervise(args)
                    status = 'completed_with_quality_failure' if quality_failed else 'completed'
                    self.assertEqual(result['status'], status)
                    self.assertEqual(result['quality_within_pairs'], not quality_failed)
                    self.assertTrue(all(c['status'] == status for c in result['comparisons'].values()))
                    self.assertEqual(result['trials'], trials)
                    self.assertEqual(result['formal_optimizer_steps'], 1800 * trials)
                    self.assertEqual(result['timing_windows'], 36 * trials)
                    self.assertEqual(result['timing_rank_records'], 288 * trials)
                    self.assertEqual(seen[-1], f'compare{trials}-report')
                    self.assertEqual(set(result['comparisons']), set(launch.TRIALS[:trials]))
                    if trials == 1:
                        self.assertFalse((root/'compare2').exists())
                        self.assertFalse(any(s.startswith('compare2-') for s in seen))

    def test_repeated_inputs_reject_changed_codec_or_shared_environment(self):
        first, second = submission('first'), submission('second')
        launch.same_inputs(first, second)
        for key in ('codec_library', 'shared_environment', 'hardware', 'allow_quality_failure'):
            changed = copy.deepcopy(second)
            changed[key] = {'changed': True}
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.same_inputs(first, changed)
        with self.assertRaises(ValueError):
            launch.same_inputs(first, first)

    def test_missing_forced_d8_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = submission('trial')
            del info['variants']['d8']
            write(root/'submission.json', info)
            with self.assertRaisesRegex(ValueError, 'three-arm'):
                launch.check_inputs(root)

    def test_old_two_arm_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = submission('trial')
            write(root/'submission.json', info)
            result = report_fixture(root, info)
            result.update(timing_windows=24, formal_optimizer_steps=1200,
                          timing_rank_records=192, preflight_rank_records=32,
                          profile_traces=16)
            result['all_samples'] = result['all_samples'][:24]
            write(root/'report.json', result)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                launch.check_report(root, info)

    def test_stale_success_report_fails_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = submission('trial')
            write(root/'submission.json', info)
            report_fixture(root, info)
            with mock.patch('mav2_report.report', return_value={'changed': True}):
                with self.assertRaisesRegex(ValueError, 'raw evidence'):
                    launch.check_report(root, info)

    def test_failed_quality_cannot_finish_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = submission('trial')
            write(root/'submission.json', info)
            result = report_fixture(root, info)
            result['quality_within_pairs'] = False
            write(root/'report.json', result)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                launch.check_report(root, info)

    def test_fresh_build_has_no_improved_arm_or_second_comparison(self):
        args = launch.arguments(['--out', '/CPU/batch'])
        commands = dict(launch.commands(args))
        self.assertEqual(args.trials, 1)
        self.assertFalse(any(k.startswith(('fused-', 'compare2-')) for k in commands))
        self.assertEqual(sum(k.endswith('-controller') for k in commands), 1)
        self.assertEqual(set(routing.POLICIES), {'basic', 'mav2', 'd8'})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.arguments(['--trials', '3'])


if __name__ == '__main__':
    unittest.main()
