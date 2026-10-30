"""CPU-only checks for read-only preview and one/two-comparison completion gates."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import a_baseline_protocol as routing
import detach_a_baseline as launch
from common import read, sha, write


def submission(trial):
    info = dict(protocol=routing.PROTOCOL, run_id=trial, execution_timing=routing.TIMING,
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
                submission_sha256=sha(root/'submission.json'))
    gate.update({k: info[k] for k in ('communication', 'codec_identity', 'codec_library')})
    write(root/'preflight.json', gate)
    write(root/'all-state.json', {'status': 'completed'})
    result = dict(protocol=routing.PROTOCOL, completed=True, correctness_passed=True,
                  execution_path_validated=True, bitwise_equal_within_pairs=True,
                  identical_trajectories_within_pairs=True, equivalence_pairs=routing.EQUIVALENCE_PAIRS,
                  run_id=info['run_id'], execution_timing=routing.TIMING,
                  round_count=2, blocks_per_round=6, timing_windows=36,
                  all_samples=[{'synthetic': True} for _ in range(36)],
                  formal_optimizer_steps=1800, timing_rank_records=288,
                  preflight_rank_records=48, profile_traces=24, candidate_shape_mappings=4,
                  evidence={'preflight_sha256': sha(root/'preflight.json')},
                  measurement_identity={'synthetic': True}, expected_order_validated=False)
    result.update({k: info[k] for k in ('communication', 'codec_identity', 'codec_library')})
    write(root/'report.json', result)
    return result


class LauncherTests(unittest.TestCase):
    def test_preview_never_creates_batch_or_starts_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'must-not-exist'
            with mock.patch.object(launch.launcher, 'invoke', side_effect=AssertionError('subprocess')), \
                 mock.patch.object(launch.launcher, 'detach', side_effect=AssertionError('fork')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                launch.main(['--out', str(out)])
            self.assertFalse(out.exists())
            self.assertIn('72 windows / 3600 formal steps', output.getvalue())

    def test_negative_ordering_finishes_both_comparisons(self):
        self.exercise(False)

    def test_single_comparison_finishes_without_compare2(self):
        self.exercise(False, trials=1)

    def test_single_preview_and_directory_describe_one_comparison(self):
        args = launch.arguments(['--trials', '1'])
        self.assertIn('a-baseline-fusion-1x-50steps-', args.out.name)
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
        self.exercise(True)

    def exercise(self, broken, trials=2):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launch.arguments(['--out', str(root), '--trials', str(trials), '--fused-build', '/CPU/fused',
                                     '--flux-build', '/CPU/flux', '--codec-build', '/CPU/codec'])
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
        for key in ('codec_library', 'shared_environment', 'fused_identity', 'hardware'):
            changed = copy.deepcopy(second)
            changed[key] = {'changed': True}
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.same_inputs(first, changed)
        with self.assertRaises(ValueError):
            launch.same_inputs(first, first)

    def test_fresh_build_shares_nccl_and_uses_exact_three_arms(self):
        args = launch.arguments(['--out', '/CPU/batch'])
        commands = dict(launch.commands(args))
        self.assertIn('/CPU/batch/fused-build/nccl-source/build', commands['flux-compile'])
        self.assertEqual(sum(k.endswith('-controller') for k in commands), 2)
        self.assertEqual(set(routing.POLICIES), {'basic', 'improved', 'cuda'})


if __name__ == '__main__':
    unittest.main()
