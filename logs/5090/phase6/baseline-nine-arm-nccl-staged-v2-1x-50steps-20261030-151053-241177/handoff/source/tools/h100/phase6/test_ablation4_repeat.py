"""Exercise the repeat supervisor without launching training or querying GPUs."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation4_repeat as repeat
from ablation_timing import execution_timing
from common import read, sha, write


class RepeatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ablation4-repeat-CPU-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        build = self.root/'build'; build.mkdir(); write(build/'ablation4-manifest.json', {})
        self.args = repeat.normalize(repeat.parser().parse_args([
            '--build', str(build), '--out', str(self.root/'batch'), '--python', sys.executable,
            '--megatron', str(self.root/'Megatron-LM'), '--execute']))
        self.calls = []

    def fake_command(self, cmd, log, env):
        self.calls.append(cmd)
        self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
        action, root = cmd[2], Path(cmd[3])
        log.write_text('SYNTHETIC CPU test; no GPU/process execution\n')
        if action == 'prepare':
            root.mkdir(); (root/'scripts').mkdir(); (root/'provenance').mkdir()
            info = dict(protocol=repeat.protocol.PROTOCOL, executable=True, source_only=False,
                run_id=root.name, blocks_per_round=5, rounds=2, timing_windows=50,
                schedule=repeat.protocol.schedule(5, rounds=2),
                communication=dict(environment={'NCCL_NVLS_ENABLE': '0'}, nvls_request='0'),
                config={}, execution_timing=execution_timing(), variants={}, calibration_hardware={}, megatron='fixed', megatron_commit='fixed',
                candidate_source_commit='fixed', build_manifest={}, build_files={}, controllers={},
                files={'scripts/worker.py': 'same-worker'})
            write(root/'submission.json', info)
            write(root/'scripts/config.json', execution_timing())
            write(root/'provenance/preparation.json', dict(execution_timing=execution_timing()))
        elif action == 'run' and '--execute' in cmd:
            info = read(root/'submission.json')
            write(root/'preflight.json', dict(completed=True, submission_sha256=sha(root/'submission.json'),
                                             communication=info['communication']))
            write(root/'all-state.json', {'status': 'completed'})
            write(root/'report.json', dict(completed=True, correctness_passed=True,
                protocol=repeat.protocol.PROTOCOL, run_id=info['run_id'], round_count=2, blocks_per_round=5,
                timing_windows=50, timing_rank_records=400, preflight_rank_records=80,
                execution_timing=execution_timing(), formal_optimizer_steps=3000,
                profile_traces=40, candidate_shape_mappings=8, all_samples=[{} for _ in range(50)],
                evidence={'preflight_sha256': sha(root/'preflight.json')},
                communication=info['communication'], measurement_identity={'fixture': 'same'}, role_plan_sha256={}))

    def run_batch(self, fake=None):
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(repeat, 'run_logged', side_effect=fake or self.fake_command):
            return repeat.execute(self.args)

    def test_two_fresh_suites_exact_flags_and_strict_sequential_reports(self):
        result = self.run_batch()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['timing_windows'], 100)
        self.assertEqual(result['timing_rank_records'], 800)
        self.assertEqual(result['formal_optimizer_steps'], 6000)
        self.assertEqual([(c[2], Path(c[3]).name, '--execute' in c) for c in self.calls], [
            ('prepare','compare1',False), ('run','compare1',False),
            ('prepare','compare2',False), ('run','compare2',False),
            ('run','compare1',True), ('report','compare1',False),
            ('run','compare2',True), ('report','compare2',False)])
        for cmd in self.calls:
            if cmd[2] == 'prepare':
                for flag,value in (('--blocks','5'),('--rounds','2'),('--nvls-enable','0')):
                    self.assertEqual(cmd[cmd.index(flag)+1],value)
            if '--execute' in cmd:
                self.assertEqual(cmd[cmd.index('--stage')+1],'all')
        self.assertNotEqual(result['comparisons']['compare1']['run_id'],result['comparisons']['compare2']['run_id'])

    def test_first_failure_or_incomplete_report_never_starts_second_suite(self):
        for failure in ('process', 'incomplete_report', 'old_20_steps'):
            with self.subTest(failure=failure):
                self.args.out = self.root/failure
                self.calls = []
                def fake(cmd, log, env):
                    self.fake_command(cmd, log, env)
                    if cmd[2] == 'report' and Path(cmd[3]).name == 'compare1':
                        if failure == 'process': raise subprocess.CalledProcessError(1,cmd)
                        path = Path(cmd[3])/'report.json'; result = read(path)
                        if failure == 'old_20_steps':
                            result['execution_timing']['timed_steps'] = 20
                            result['formal_optimizer_steps'] = 1000
                        else: result['timing_windows'] = 49
                        write(path,result)
                with self.assertRaises((subprocess.CalledProcessError, ValueError)):
                    self.run_batch(fake)
                self.assertFalse(any(Path(c[3]).name=='compare2' and '--execute' in c for c in self.calls))
                self.assertEqual(read(self.args.out/'batch-state.json')['status'],'failed')
                self.assertTrue((self.args.out/'compare1/report.json').exists())

    def test_input_drift_stops_before_any_gpu_command(self):
        def fake(cmd, log, env):
            self.fake_command(cmd, log, env)
            if cmd[2] == 'prepare' and Path(cmd[3]).name == 'compare2':
                path = Path(cmd[3])/'submission.json'; info = read(path)
                info['variants'] = {'changed': True}; write(path,info)
        with self.assertRaisesRegex(ValueError,'inputs differ'):
            self.run_batch(fake)
        self.assertFalse(any('--execute' in c for c in self.calls))

    def test_existing_batch_is_never_reused(self):
        self.args.out.mkdir(); marker = self.args.out/'preserve.txt'; marker.write_text('old evidence')
        with self.assertRaises(FileExistsError): self.run_batch()
        self.assertEqual(marker.read_text(),'old evidence')
        self.assertEqual(self.calls,[])

    def test_default_plan_is_read_only(self):
        dest = self.root/'preview'
        with mock.patch.object(subprocess,'run',side_effect=AssertionError('Process started')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            repeat.main(['--out',str(dest)])
        self.assertFalse(dest.exists())
        self.assertIn('--blocks 5 --rounds 2 --nvls-enable 0',output.getvalue())
        self.assertIn('PLAN ONLY',output.getvalue())


if __name__ == '__main__':
    unittest.main()
