"""CPU-only sequence, strict completion and detach dispatch checks."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation_timing
import detach_ablation5_then4 as batch
from common import read, sha, write


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ablation5-then4-CPU-'); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = batch.arguments(['--out', str(self.root/'batch'), '--build', str(self.root/'build')])
        self.args.out.mkdir(); self.order = []

    def fake_invoke(self, cmd, log, env):
        label = log.stem; protocol = (batch.a5 if label == 'ablation5' else batch.a4).protocol
        self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
        root = self.args.out/label; root.mkdir()
        for trial in ('compare1', 'compare2'):
            self.order.append(label+'/'+trial)
            folder = root/trial; timing = ablation_timing.execution_timing()
            folder.mkdir(); (folder/'scripts').mkdir(); (folder/'provenance').mkdir()
            info = dict(protocol=protocol.PROTOCOL, executable=True, source_only=False, run_id=label+'-'+trial,
                        rounds=2, blocks_per_round=5, timing_windows=50, schedule=protocol.schedule(5, rounds=2),
                        execution_timing=timing, config={}, variants={}, calibration_hardware={},
                        megatron='fixed', megatron_commit='fixed', candidate_source_commit='fixed',
                        build_manifest={}, build_files={}, controllers={}, files={'scripts/worker.py': 'same'},
                        communication=dict(environment={'NCCL_NVLS_ENABLE': '0'}, nvls_request='0'))
            if label == 'ablation5': info['budget_denominators'] = list(protocol.DENOMINATORS)
            write(folder/'submission.json', info)
            write(folder/'scripts/config.json', timing)
            write(folder/'provenance/preparation.json', dict(execution_timing=timing))
            write(folder/'preflight.json', dict(completed=True, submission_sha256=sha(folder/'submission.json'),
                                               communication=info['communication']))
            write(folder/'all-state.json', dict(status='completed'))
            result = dict(protocol=protocol.PROTOCOL, completed=True, correctness_passed=True, run_id=info['run_id'],
                round_count=2, blocks_per_round=5, timing_windows=50, timing_rank_records=400,
                preflight_rank_records=80, profile_traces=40, candidate_shape_mappings=10 if label == 'ablation5' else 8,
                execution_timing=timing, formal_optimizer_steps=3000, all_samples=[{} for _ in range(50)],
                communication=info['communication'], measurement_identity={'same': True}, role_plan_sha256={},
                evidence=dict(preflight_sha256=sha(folder/'preflight.json')))
            if label == 'ablation5': result['budget_denominators'] = list(protocol.DENOMINATORS)
            write(folder/'report.json', result)
        write(root/'batch-state.json', dict(status='completed', execution_timing=ablation_timing.execution_timing(),
            timing_windows=100, timing_rank_records=800, formal_optimizer_steps=6000))

    def test_four_runs_in_order_and_all_strict_reports_required(self):
        with mock.patch.object(batch.a5, 'invoke', side_effect=self.fake_invoke), contextlib.redirect_stdout(io.StringIO()):
            result = batch.supervise(self.args)
        self.assertEqual(self.order, result['order'])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['timing_windows'], 200)
        self.assertEqual(result['formal_optimizer_steps'], 12000)
        commands = batch.commands(self.args)
        self.assertEqual([label for label, _ in commands], ['ablation5', 'ablation4'])
        self.assertIn('--foreground', commands[0][1])
        self.assertEqual(Path(commands[0][1][commands[0][1].index('--candidate-build')+1]), self.args.build/'candidate')

    def test_failure_or_old_or_missing_report_prevents_ablation4(self):
        for failure in ('process', 'missing', 'old_steps'):
            with self.subTest(failure=failure):
                self.args.out = self.root/failure; self.args.out.mkdir(); self.order = []
                def invoke(cmd, log, env):
                    self.fake_invoke(cmd, log, env)
                    if failure == 'process': raise subprocess.CalledProcessError(1, cmd)
                    path = self.args.out/'ablation5/compare2/report.json'
                    if failure == 'missing': path.unlink()
                    else:
                        result = read(path); result['execution_timing']['timed_steps'] = 20; write(path, result)
                with mock.patch.object(batch.a5, 'invoke', side_effect=invoke), self.assertRaises((subprocess.CalledProcessError, FileNotFoundError, ValueError)):
                    batch.supervise(self.args)
                self.assertEqual(self.order, ['ablation5/compare1', 'ablation5/compare2'])
                self.assertFalse((self.args.out/'ablation4').exists())
                self.assertEqual(read(self.args.out/'batch-state.json')['status'], 'failed')

    def test_plan_only_has_no_side_effects(self):
        root = self.root/'preview'
        with mock.patch.object(os, 'fork', side_effect=AssertionError('fork')), \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('process')), contextlib.redirect_stdout(io.StringIO()) as output:
            batch.main(['--out', str(root)])
        self.assertFalse(root.exists())
        self.assertIn('60 timed steps', output.getvalue())
        self.assertIn('12000 formal steps', output.getvalue())

    def test_execute_detaches_combined_supervisor_and_rejects_existing_output(self):
        root = self.root/'launch'; build = self.args.build
        (build/'candidate').mkdir(parents=True)
        write(build/'ablation4-manifest.json', {}); write(build/'candidate/manifest.json', {})
        argv = ['--out', str(root), '--build', str(build), '--python', sys.executable, '--execute']
        with mock.patch.object(batch.a5, 'detach', return_value=dict(ready=True)) as detach, contextlib.redirect_stdout(io.StringIO()):
            batch.main(argv)
            self.assertIs(detach.call_args.kwargs['supervisor'], batch.supervise)
            self.assertEqual(read(root/'batch-state.json')['status'], 'launching')
            with self.assertRaises(FileExistsError): batch.main(argv)
            self.assertEqual(detach.call_count, 1)


if __name__ == '__main__': unittest.main()
