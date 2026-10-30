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

import basic_quant_protocol as routing
import detach_basic_quant as launch
from common import read, sha, write


def submission(trial):
    info = dict(protocol=routing.PROTOCOL, run_id=trial, execution_timing=routing.TIMING,
                executable=True, source_only=False, rounds=2, blocks_per_round=6,
                timing_windows=24, schedule=routing.schedule(),
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
                submission_sha256=sha(root/'submission.json'))
    gate.update({k: info[k] for k in ('communication', 'codec_identity', 'codec_library')})
    write(root/'preflight.json', gate)
    write(root/'all-state.json', {'status': 'completed'})
    result = dict(protocol=routing.PROTOCOL, completed=True, correctness_passed=True,
                  execution_path_validated=True, bitwise_equal_within_pairs=True,
                  identical_trajectories_within_pairs=True, equivalence_pairs=routing.EQUIVALENCE_PAIRS,
                  run_id=info['run_id'], execution_timing=routing.TIMING,
                  round_count=2, blocks_per_round=6, timing_windows=24,
                  all_samples=[{'synthetic': True} for _ in range(24)],
                  formal_optimizer_steps=1200, timing_rank_records=192,
                  preflight_rank_records=32, profile_traces=16, candidate_shape_mappings=0,
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
            (phase/'run_basic_quant.sh').write_text((launch.HERE/'run_basic_quant.sh').read_text())
            recovery = root/'outputs/h100/container-recovery'
            recovery.mkdir(parents=True)
            import sys
            import shlex
            (recovery/'env.sh').write_text('export H100_RECOVERY_PYTHON='+shlex.quote(sys.executable)+'\n'
                'export H100_RECOVERY_BUILD_CUDA=/recovered/build-cuda\n'
                'export CUDA_HOME=/recovered/runtime-cuda\nexport CC=/recovered/gcc\n')
            (phase/'detach_basic_quant.py').write_text('import os,sys,json\n'
                'print(json.dumps(dict(args=sys.argv[1:],cwd=os.getcwd(),cc=os.environ["CC"])))\n')
            result = subprocess.run(['bash', str(phase/'run_basic_quant.sh'), '--trials', '1', '--execute',
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
            self.assertIn('24 windows / 1200 formal steps', output.getvalue())

    def test_negative_ordering_still_finishes(self):
        self.exercise(False)

    def test_single_comparison_finishes_without_compare2(self):
        self.exercise(False, trials=1)

    def test_single_preview_and_directory_describe_one_comparison(self):
        args = launch.arguments(['--trials', '1'])
        self.assertIn('basic-quant-receiver-1x-50steps-', args.out.name)
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
            self.assertIn('24 windows / 1200 formal steps', output.getvalue())
            self.assertNotIn('compare2', output.getvalue())
            self.assertFalse(out.exists())

    def test_incomplete_first_report_prevents_second_training(self):
        self.exercise(True)

    def exercise(self, broken, trials=1):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launch.arguments(['--out', str(root), '--trials', str(trials), '--flux-build', '/CPU/flux', '--codec-build', '/CPU/codec'])
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
                    write(folder/'submission.json', submission(trial))
                if label.endswith('-report'):
                    result = report_fixture(folder, read(folder/'submission.json'))
                    if broken and trial == 'compare1':
                        result['all_samples'].pop()
                        write(folder/'report.json', result)

            with mock.patch.object(launch.launcher, 'invoke', side_effect=invoke), \
                 contextlib.redirect_stdout(io.StringIO()):
                if broken:
                    with self.assertRaisesRegex(ValueError, 'Incomplete'):
                        launch.supervise(args)
                    self.assertFalse(any(s.startswith('compare2-') for s in seen))
                    self.assertEqual(read(root/'batch-state.json')['status'], 'failed')
                else:
                    result = launch.supervise(args)
                    self.assertEqual(result['status'], 'completed')
                    self.assertEqual(result['trials'], trials)
                    self.assertEqual(result['formal_optimizer_steps'], 1200 * trials)
                    self.assertEqual(result['timing_windows'], 24 * trials)
                    self.assertEqual(result['timing_rank_records'], 192 * trials)
                    self.assertEqual(seen[-1], f'compare{trials}-report')
                    self.assertEqual(set(result['comparisons']), set(launch.TRIALS[:trials]))
                    if trials == 1:
                        self.assertFalse((root/'compare2').exists())
                        self.assertFalse(any(s.startswith('compare2-') for s in seen))

    def test_repeated_inputs_reject_changed_codec_or_shared_environment(self):
        first, second = submission('first'), submission('second')
        launch.same_inputs(first, second)
        for key in ('codec_library', 'shared_environment', 'hardware'):
            changed = copy.deepcopy(second)
            changed[key] = {'changed': True}
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.same_inputs(first, changed)
        with self.assertRaises(ValueError):
            launch.same_inputs(first, first)

    def test_fresh_build_has_no_improved_arm_or_second_comparison(self):
        args = launch.arguments(['--out', '/CPU/batch'])
        commands = dict(launch.commands(args))
        self.assertEqual(args.trials, 1)
        self.assertFalse(any(k.startswith(('fused-', 'compare2-')) for k in commands))
        self.assertEqual(sum(k.endswith('-controller') for k in commands), 1)
        self.assertEqual(set(routing.POLICIES), {'basic', 'cuda'})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.arguments(['--trials', '2'])


if __name__ == '__main__':
    unittest.main()
