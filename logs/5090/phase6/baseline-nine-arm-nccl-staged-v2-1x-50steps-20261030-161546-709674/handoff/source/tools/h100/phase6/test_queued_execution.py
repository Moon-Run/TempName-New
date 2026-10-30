"""CPU-only checks for the queue's no-resource-polling execution mode."""
import contextlib
import io
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import execute_queued as runner
from common import read, sha, write


class QueuedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.build = self.repo / 'build-source'
        self.log = self.repo / 'logs'
        (self.build / 'base').mkdir(parents=True)
        (self.log / 'handoff').mkdir(parents=True)
        self.plan = self.repo / 'plan.json'
        write(self.plan, {})
        write(self.build / 'base/manifest.json', dict(status='built', variants={
            'taco_fused': dict(implementation=runner.IMPLEMENTATION)}))
        write(self.log / 'handoff/reuse-verification.json', dict(plan_sha256=sha(self.plan)))
        self.calls = []
        self.states = []
        self.fail_stage = None
        self.valid_decoder = True

    def fake_popen(self, command, **kwargs):
        self.calls.append(command)
        script = Path(command[1]).name
        stage = command[command.index('--stage') + 1] if '--stage' in command else script
        compare = self.log / 'compare'
        if script == 'prepare.py':
            compare.mkdir()
            write(compare / 'submission.json', dict(variants={
                'taco_fused': dict(implementation=runner.IMPLEMENTATION)}))
        elif stage == 'preflight':
            write(compare / 'preflight.json', dict(completed=True, kernel_names={
                f'taco_fused/rank{rank}': ['taco_decode_ring_kernel(FluxTacoConfig)']
                if self.valid_decoder else ['wrong_decoder()'] for rank in range(8)}))
        elif script == 'report.py':
            write(compare / 'report.json', dict(completed=True,
                fused_baseline_implementation=runner.IMPLEMENTATION, accepted={}))
        test = self

        class Child:
            pid = 1234
            returncode = None
            waits = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.waits += 1
                test.states.append(read(test.log / 'execution.json'))
                if self.waits == 1:
                    raise subprocess.TimeoutExpired(command, timeout)
                self.returncode = 7 if stage == test.fail_stage else 0
                return self.returncode

        return Child()

    def execute(self):
        with patch.object(runner.subprocess, 'Popen', side_effect=self.fake_popen), \
             patch.object(runner.subprocess, 'check_output',
                          side_effect=AssertionError('Unexpected GPU/resource query')), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.repo, self.build, self.log, self.repo / 'candidates',
                           self.plan, self.repo / 'megatron', self.repo / 'cuda',
                           Path(sys.executable))

    def test_success_keeps_gates_and_progress_without_resource_queries(self):
        self.execute()
        self.assertEqual([Path(c[1]).name for c in self.calls],
                         ['prepare.py', 'run.py', 'run.py', 'report.py'])
        self.assertEqual([c[c.index('--stage') + 1] for c in self.calls if '--stage' in c],
                         ['preflight', 'timing'])
        self.assertTrue(all('--execute' in c for c in self.calls if '--stage' in c))
        self.assertTrue(all(s['gpu_monitoring'] == 'before_each_run_only' for s in self.states))
        self.assertEqual(read(self.log / 'execution.json')['status'], 'completed')
        self.assertTrue(read(self.log / 'handoff/cta-trace-verification.json')['passed'])
        self.assertFalse((self.log / 'handoff/gpu-occupancy.jsonl').exists())

    def test_failed_timing_records_failure_without_resource_queries(self):
        self.fail_stage = 'timing'
        with self.assertRaisesRegex(RuntimeError, 'timing.log exited with 7'):
            self.execute()
        state = read(self.log / 'execution.json')
        self.assertEqual((state['status'], state['stage']), ('failed', 'timing'))
        self.assertFalse(any(Path(c[1]).name == 'report.py' for c in self.calls))

    def test_wrong_baseline_stops_before_launch(self):
        path = self.build / 'base/manifest.json'
        data = read(path)
        data['variants']['taco_fused']['implementation'] = 'phase6-warp-v1'
        write(path, data)
        with self.assertRaisesRegex(ValueError, 'aligned fused-v2'):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_missing_cta_evidence_prevents_timing(self):
        self.valid_decoder = False
        with self.assertRaisesRegex(ValueError, 'CTA decoder evidence'):
            self.execute()
        self.assertFalse(any('timing' in c for c in self.calls))

    def test_changed_plan_stops_before_launch(self):
        write(self.plan, dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'selection plan changed'):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_cli_without_execute_never_launches(self):
        with patch.object(sys, 'argv', ['execute_queued.py']), \
             patch.object(runner, 'execute') as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
