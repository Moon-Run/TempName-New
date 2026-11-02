"""Candidate-only orchestration, historical evidence and fail-closed CPU checks."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import op4

HISTORICAL_MS = 850.1950773783028


def samples(blocks=1, trials=1, ms=780.0):
    return [dict(job, ms_per_step=ms) for job in op4.schedule(trials, blocks)]


def summarize(rows=None, trials=1, blocks=1, operators=None):
    return op4.report(samples(blocks, trials) if rows is None else rows, trials, blocks,
                      operators, historical_reference_ms=HISTORICAL_MS)


def reference(root):
    return dict(root=str(root/'reference'), report_sha256='a'*64,
                operator_report_sha256='b'*64, evidence_sha256='c'*64,
                historical_mean_ms=HISTORICAL_MS, read_only=True, reference_gpu_jobs=0)


class Protocol(unittest.TestCase):
    def test_only_candidate_runs_and_never_claims_paired_gain(self):
        self.assertEqual(op4.ARMS, ('candidate',))
        self.assertEqual(len(op4.schedule(2, 3)), 6)
        self.assertEqual({job['arm'] for job in op4.schedule(2, 3)}, {'candidate'})
        for blocks in (1, 4):
            result = summarize(trials=2, blocks=blocks)
            self.assertEqual(result['timing_windows'], 2*blocks)
            self.assertEqual(result['formal_optimizer_steps'], 100*blocks)
            self.assertFalse(result['concurrent_control'])
            self.assertFalse(result['pairwise_ci_available'])
            self.assertFalse(result['target_validated'])
            self.assertFalse(result['historical_comparison_is_causal'])
            self.assertEqual(set(result['comparisons']['compare1']['arms']), {'candidate'})
            self.assertAlmostEqual(result['comparisons']['compare1']['historical_delta_ms'], HISTORICAL_MS-780)
            self.assertNotIn('pairs', result['comparisons']['compare1'])
        self.assertIn('不是同期净收益', op4.markdown(result))
        for blocks in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                op4.schedule(blocks=blocks)

    def test_slow_windows_are_retained_and_invalid_samples_fail(self):
        rows = samples(4)
        rows[-1]['ms_per_step'] = 1200
        result = summarize(rows, blocks=4)
        self.assertEqual(result['all_samples'], rows)
        self.assertEqual(result['comparisons']['compare1']['arms']['candidate']['max_ms'], 1200)
        with self.assertRaisesRegex(ValueError, 'Missing/reordered'):
            summarize(rows[:-1], blocks=4)
        rows[0]['ms_per_step'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid OP4'):
            summarize(rows, blocks=4)

    def test_operator_only_never_claims_training_or_speedup(self):
        operators = [dict(op4.schedule()[0], roles={'attn': {'gpu_us': {'mean': 800}, 'wall_us': {'mean': 850}}})]
        result = summarize([], operators=operators)
        self.assertEqual(result['formal_optimizer_steps'], 0)
        self.assertFalse(result['full_step_completed'])
        self.assertFalse(result['target_validated'])
        self.assertEqual(result['comparisons'], {})
        self.assertIn('未执行完整模型训练', op4.markdown(result))


class Launch(unittest.TestCase):
    def test_candidate_environment_and_external_override_guards(self):
        env = {'PATH': '/CPU/bin', 'NCCL_P2P_DISABLE': '1'}
        with patch.object(op4, 'check_overrides'):
            candidate = op4.child_environment('candidate', env)
        self.assertEqual(candidate['NCCL_MIN_NCHANNELS'], '16')
        self.assertEqual(candidate['NCCL_MAX_NCHANNELS'], '16')
        self.assertNotIn('NCCL_PROTO', candidate)
        with self.assertRaisesRegex(ValueError, 'Unknown OP4'):
            op4.child_environment('control', env)
        for key in op4.EXTERNAL_OVERRIDES:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                op4.check_overrides({key: ''})
        with tempfile.TemporaryDirectory() as tmp:
            conf = Path(tmp)/'nccl.conf'
            conf.write_text('# NCCL_PROTO=LL\nNCCL_MIN_CTAS = 4\n')
            with self.assertRaisesRegex(ValueError, 'NCCL_MIN_CTAS'):
                op4.check_overrides({'NCCL_CONF_FILE': str(conf)})

    def test_preview_only_candidate_creates_no_files_in_either_mode(self):
        for operator_only in (False, True):
            with self.subTest(operator_only=operator_only), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'absent'
                command = ['bash', str(op4.HERE/'run_op4.sh'), '--out', str(out)]
                if operator_only:
                    command += ['--operator-only']
                process = subprocess.run(command, capture_output=True, text=True,
                    env=dict(os.environ, RTX5090_PYTHON=sys.executable))
                self.assertEqual(process.returncode, 0, process.stderr)
                plan = json.loads(process.stdout[process.stdout.index('{'):])
                self.assertEqual(plan['arms'], ['candidate'])
                self.assertEqual(plan['timing_windows'], 0 if operator_only else 1)
                self.assertEqual(plan['formal_optimizer_steps'], 0 if operator_only else 50)
                self.assertEqual(plan['reference_gpu_jobs'], 0)
                self.assertFalse(out.exists())
        with self.assertRaisesRegex(ValueError, '--op3-build or --abc-build'):
            op4.arguments(['--execute'])

    def test_child_and_supervisor_forward_operator_flag_without_reference_gpu_job(self):
        for operator_only in (False, True):
            args = op4.arguments(['--op3-build', '/CPU/build', '--blocks', '7',
                                  *(['--operator-only'] if operator_only else [])])
            command = op4.child_command(args, Path('/CPU/new child'), args.op3_build)
            self.assertIn('--foreground', command)
            self.assertEqual(command[command.index('--blocks')+1], '1')
            self.assertEqual('--operator-only' in command, operator_only)
            self.assertNotIn('--reference-batch', command)
            supervisor = op4.supervisor_command(args)
            self.assertEqual('--operator-only' in supervisor, operator_only)
            self.assertIn('--reference-batch', supervisor)


class Supervisor(unittest.TestCase):
    def fixture(self, root, *, fail=None, fresh=False, operator_only=False, blocks=1):
        args = op4.arguments(['--out', str(root), '--execute', '--blocks', str(blocks),
            '--abc-build' if fresh else '--op3-build', str(root/'build'),
            *(['--operator-only'] if operator_only else [])])
        op4.write(root/'batch-state.json', {})
        evidence = ModuleType('op4_evidence')
        calls, audits = [], []
        def run(command, env, log):
            calls.append((command, env))
            if fail == 'child':
                raise ValueError('child failed')
            child = Path(command[command.index('--out')+1])
            child.mkdir(parents=True)
            op4.write(child/'plan.json', dict(build={'root': str(root/'build')}))
        def audit(child, ref):
            audits.append((child, ref))
            if fail == 'audit':
                raise ValueError('candidate gate failed')
            return {'passed': True}
        evidence.audit_candidate = audit
        operators = [dict(job, roles={}) for job in op4.schedule(blocks=blocks)] if operator_only else None
        result = summarize([] if operator_only else samples(blocks), blocks=blocks, operators=operators)
        error = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'op4_evidence': evidence}))
            stack.enter_context(patch.object(op4, 'REPO', root))
            stack.enter_context(patch.object(op4, 'check_overrides'))
            stack.enter_context(patch.object(op4.op3.base, 'verify_controllers'))
            stack.enter_context(patch.object(op4, 'reference_record', return_value=reference(root)))
            stack.enter_context(patch.object(op4, 'child_result', return_value={}))
            stack.enter_context(patch.object(op4, 'run_child', side_effect=run))
            stack.enter_context(patch.object(op4, 'replay', return_value=result))
            try:
                with redirect_stdout(io.StringIO()):
                    op4.supervise(args)
            except ValueError as exc:
                error = exc
        return calls, audits, error

    def test_one_candidate_child_and_no_control_in_both_modes(self):
        for operator_only in (False, True):
            with self.subTest(operator_only=operator_only), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, audits, error = self.fixture(root, operator_only=operator_only)
                self.assertIsNone(error)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(audits), 1)
                self.assertEqual(calls[0][1]['NCCL_MIN_NCHANNELS'], '16')
                self.assertEqual(calls[0][1]['NCCL_MAX_NCHANNELS'], '16')
                self.assertEqual('--operator-only' in calls[0][0], operator_only)
                state = op4.read(root/'batch-state.json')
                self.assertEqual(state['status'], 'operator_completed' if operator_only else 'completed')
                self.assertEqual(state['formal_optimizer_steps'], 0 if operator_only else 50)
                self.assertFalse(state['target_validated'])
                self.assertTrue((root/'candidate-evidence.json').is_file())
                self.assertFalse((root/'paired-evidence.json').exists())

    def test_new_codec_is_reused_after_first_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, audits, error = self.fixture(Path(tmp), fresh=True, blocks=2)
            self.assertIsNone(error)
            self.assertEqual(len(calls), 2)
            self.assertIn('--abc-build', calls[0][0])
            self.assertIn('--op3-build', calls[1][0])

    def test_child_or_candidate_failure_never_writes_success(self):
        for failure in ('child', 'audit'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, audits, error = self.fixture(root, fail=failure)
                self.assertIsNotNone(error)
                self.assertEqual(len(calls), 1)
                self.assertEqual(op4.read(root/'batch-state.json')['status'], 'failed')
                self.assertFalse((root/'report.json').exists())


class Replay(unittest.TestCase):
    def fixture(self, root, operator_only=False):
        plan = dict(protocol=op4.PROTOCOL, run_id='cpu-fixture', trials=1, blocks=1,
            model=op4.p.MODEL, target_ms=op4.TARGET_MS, nccl_overrides=op4.OVERRIDES,
            operator_only=operator_only, reference=reference(root), schedule=op4.schedule())
        op4.write(root/'plan.json', plan)
        op4.write(root/'candidate-evidence.json', {'compare1/block00-candidate': {'passed': True}})
        job = plan['schedule'][0]
        child = root/job['directory']
        op4.write(child/'plan.json', dict(operator_only=operator_only, model=op4.p.MODEL,
            environment={}, communication={}, cuda_visible_devices='0,1,2,3,4,5,6,7',
            build={'root': '/CPU/build', 'manifest_sha256': 'a'*64}))
        op4.write(child/('operator/report.json' if operator_only else 'report.json'), {'fixture': True})
        op4.write(child.with_name(child.name+'-launch.json'),
                  dict(job=job, nccl_overrides=op4.OVERRIDES['candidate'], command=['fixture']))
        def child_result(child, recompute_profiles):
            return {'roles': {}} if operator_only else {'all_samples': [{'ms_per_step': 780}]}
        evidence = ModuleType('op4_evidence')
        evidence.audit_candidate = lambda *args: {'passed': True}
        return child_result, evidence

    def test_readonly_replay_both_modes_and_tamper_rejection(self):
        for operator_only in (False, True):
            with self.subTest(operator_only=operator_only), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                child, evidence = self.fixture(root, operator_only)
                stack.enter_context(patch.dict(sys.modules, {'op4_evidence': evidence}))
                stack.enter_context(patch.object(op4.op3.base, 'verify_controllers'))
                children = stack.enter_context(patch.object(op4, 'child_result', side_effect=child))
                stack.enter_context(patch.object(op4, 'training_identity', return_value=['fixed-input-identity']))
                stack.enter_context(patch.object(op4, 'reference_record', return_value=reference(root)))
                before = {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}
                result = op4.replay(root)
                self.assertEqual(children.call_count, 1)
                self.assertTrue(children.call_args.args[1])
                self.assertFalse(result['target_validated'])
                self.assertEqual(result['formal_optimizer_steps'], 0 if operator_only else 50)
                after = {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}
                self.assertEqual(before, after)
                op4.write(root/'candidate-evidence.json', {'compare1/block00-candidate': {'passed': False}})
                with self.assertRaisesRegex(ValueError, 'differs from raw child evidence'):
                    op4.replay(root)

    def test_incomplete_child_is_rejected_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op4.write(root/'plan.json', {'operator_only': False})
            op4.write(root/'batch-state.json', {'status': 'failed'})
            with patch.object(op4.op3, 'replay') as replay:
                with self.assertRaisesRegex(ValueError, 'Incomplete OP3 child'):
                    op4.child_result(root)
            replay.assert_not_called()

    def test_real_historical_reference_is_read_without_gpu_calls(self):
        if not op4.DEFAULT_REFERENCE.is_dir():
            self.skipTest('Historical reference fixture is not available')
        with patch.object(op4.op3, 'replay') as replay, patch.object(op4, 'run_child') as run:
            record = op4.reference_record(op4.DEFAULT_REFERENCE)
        self.assertAlmostEqual(record['historical_mean_ms'], HISTORICAL_MS)
        self.assertEqual(record['reference_gpu_jobs'], 0)
        replay.assert_not_called()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
