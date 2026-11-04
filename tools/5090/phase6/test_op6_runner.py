"""CPU checks for paired admission, reusable builds and scoped child overlays."""
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

import op6
import op6_child


def samples(trials=1, blocks=2):
    return [dict(job, ms_per_step=832.0 if job['arm'] == 'op4' else 798.0,
                 peak_allocated_gib=18.0 if job['arm'] == 'op4' else 21.0,
                 peak_reserved_gib=24.0, native_workspace_gib=0.0,
                 peak_allocated_plus_taco_workspace_gib=18.0 if job['arm'] == 'op4' else 21.0)
            for job in op6.schedule(trials, blocks)]


class Protocol(unittest.TestCase):
    def test_paired_order_reverses_by_block_and_trial(self):
        jobs = op6.schedule(2, 2)
        self.assertEqual([job['arm'] for job in jobs],
                         ['op4', 'op6', 'op6', 'op4', 'op6', 'op4', 'op4', 'op6'])
        self.assertEqual(len({job['directory'] for job in jobs}), 8)
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                op6.schedule(blocks=bad)
            with self.assertRaises(ValueError):
                op6.schedule(trials=bad)
        self.assertEqual(len(op6.schedule(3, 1)), 6)

    def test_complete_pairs_and_800ms_observation(self):
        result = op6.report(samples(), 1, 2)
        self.assertEqual(result['formal_optimizer_steps'], 200)
        self.assertEqual(result['mean_paired_saved_ms'], 34.0)
        self.assertTrue(result['target_800ms_observed'])
        self.assertTrue(result['counterbalanced_order'])
        self.assertFalse(result['pairwise_ci_available'])
        self.assertFalse(result['control_and_candidate_run_simultaneously'])
        self.assertEqual(result['aggregate']['op6']['max_peak_allocated_gib'], 21.0)
        self.assertEqual(result['comparisons']['compare1']['faster_blocks'], 2)
        rows = samples()
        rows[2]['ms_per_step'] = 1200
        result = op6.report(rows, 1, 2)
        self.assertFalse(result['target_800ms_observed'])
        self.assertEqual(result['all_samples'], rows)
        self.assertIn('completed 表示验证完成', op6.markdown(result))
        self.assertFalse(op6.report(samples(blocks=1), 1, 1)['counterbalanced_order'])

    def test_missing_reordered_foreign_and_nonfinite_windows_fail(self):
        rows = samples()
        for invalid in (rows[:-1], rows[::-1], [dict(rows[0], arm='foreign')]+rows[1:]):
            with self.assertRaisesRegex(ValueError, 'Missing, reordered'):
                op6.report(invalid, 1, 2)
        for bad in (float('nan'), float('inf'), 0, -1):
            with self.assertRaisesRegex(ValueError, 'Invalid OP6'):
                op6.report([dict(rows[0], ms_per_step=bad)]+rows[1:], 1, 2)


class Launch(unittest.TestCase):
    def test_build_resolution_rejects_open_compress_and_ambiguous_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = op6.arguments(['--build', str(root)])
            (root/'open-compress-build.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'open-compress'):
                op6.resolve_reused_build(args)
            (root/'op3-build.json').write_text('{}')
            op6.resolve_reused_build(args)
            self.assertEqual(args.op3_build, root)
            (root/'op1-abc-build.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                op6.resolve_reused_build(args)
            (root/'op3-build.json').unlink()
            args = op6.arguments(['--build', str(root)])
            op6.resolve_reused_build(args)
            self.assertEqual(args.abc_build, root)

    def test_place_defaults_jobs_and_supervisor_arguments(self):
        for place in (60, 61):
            with patch.object(op6, 'default_build_jobs', return_value=37):
                args = op6.arguments(['--place', str(place), '--trials', '3', '--blocks', '2',
                                      '--build', '/CPU/reusable build'])
            self.assertEqual(args.out.parent.name, f'SL30{place}')
            self.assertEqual(args.jobs, 37)
            self.assertEqual(args.reserve_mib, 30000)
            rebuilt = op6.arguments(op6.supervisor_command(args)[2:])
            self.assertEqual(rebuilt.build, args.build)
            self.assertEqual(rebuilt.trials, 3)
            self.assertEqual(rebuilt.blocks, 2)
            self.assertEqual(rebuilt.reserve_mib, 30000)
        args = op6.arguments(['--place', '60', '--out', '/CPU/explicit'])
        self.assertEqual(args.out, Path('/CPU/explicit'))

    def test_child_commands_never_run_both_arms_at_once(self):
        args = op6.arguments(['--out', '/CPU/batch'])
        for job in op6.schedule():
            command = op6.child_command(args, job, Path('/CPU/build with spaces'))
            self.assertEqual(command[command.index('--variant')+1], job['arm'])
            self.assertEqual(command[command.index('--blocks')+1], '1')
            self.assertEqual(command[command.index('--trials')+1], '1')
            self.assertIn('--foreground', command)
            self.assertEqual(command[command.index('--op3-build')+1], '/CPU/build with spaces')
            self.assertEqual(command[command.index('--reserve-mib')+1], '30000')

    def test_channel_and_op5_override_guards(self):
        with patch.object(op6, 'check_overrides'):
            env = op6.child_environment({'PATH': '/CPU/bin'})
        self.assertEqual({key: env[key] for key in op6.CHANNEL_ENV}, op6.CHANNEL_ENV)
        self.assertNotIn('NCCL_SHM_USE_CUDA_MEMCPY', env)
        for key in op6.EXTERNAL_OVERRIDES:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                op6.check_overrides({key: ''})
        with tempfile.TemporaryDirectory() as tmp:
            conf = Path(tmp)/'nccl.conf'
            conf.write_text('NCCL_SHM_USE_CUDA_MEMCPY = 1\n')
            with self.assertRaisesRegex(ValueError, 'NCCL_SHM_USE_CUDA_MEMCPY'):
                op6.check_overrides({'NCCL_CONF_FILE': str(conf)})

    def test_preview_is_side_effect_free_even_with_unavailable_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'not-created'
            command = ['bash', str(op6.HERE/'run_op6.sh'), '--out', str(out),
                       '--trials', '1', '--blocks', '2', '--build', '/CPU/unavailable']
            process = subprocess.run(command, text=True, capture_output=True,
                env=dict(os.environ, RTX5090_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(process.returncode, 0, process.stderr)
            plan = json.loads(process.stdout[process.stdout.index('{'):])
            self.assertEqual(plan['arms'], ['op4', 'op6'])
            self.assertEqual(plan['formal_optimizer_steps'], 200)
            self.assertEqual(plan['memory_reservation']['mib_per_gpu'], 30000)
            self.assertFalse(plan['memory_reservation']['held_during_gpu_work'])
            self.assertFalse(out.exists())


class ChildScope(unittest.TestCase):
    def test_variant_is_retained_in_memory_and_all_hooks_restore_on_failure(self):
        events, modules = [], {}
        for name in ('op6_sources', 'op6_validation', 'op6_profile'):
            module = ModuleType(name)
            def install(variant, name=name):
                events.append(('install', name, variant))
                return lambda: events.append(('restore', name))
            module.install = install
            modules[name] = module
        plan = {}
        with patch.dict(sys.modules, modules), patch.object(op6_child.op3, 'write') as writer:
            with self.assertRaisesRegex(ValueError, 'intentional'):
                with op6_child.configured('op6'):
                    op6_child.op3.write(Path('/CPU/plan.json'), plan)
                    self.assertEqual(plan['op6_variant'], 'op6')
                    self.assertEqual(plan['op6_protocol'], op6.PROTOCOL)
                    raise ValueError('intentional')
            self.assertIs(op6_child.op3.write, writer)
            writer.assert_called_once_with(Path('/CPU/plan.json'), plan)
        self.assertEqual(events[-3:], [('restore', name) for name in reversed(tuple(modules))])

    def test_report_rejects_wrong_variant_before_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op6.write(root/'plan.json', dict(op6_variant='op4', op6_protocol=op6.PROTOCOL))
            with self.assertRaisesRegex(ValueError, 'differs from saved child'):
                op6_child.main(['--variant', 'op6', '--report', '--out', str(root)])


class Supervision(unittest.TestCase):
    def run_fixture(self, root, failure=None, reserve_mib=0, blocks=2, inspect_child=None):
        args = op6.arguments(['--out', str(root), '--execute', '--op3-build', str(root/'reused'),
                             '--blocks', str(blocks), '--reserve-mib', str(reserve_mib)])
        (root/'reused').mkdir()
        op6.write(root/'batch-state.json', {})
        calls = []
        evidence = ModuleType('op6_evidence')
        def audit(left, right, **kwargs):
            if failure == 'audit':
                raise ValueError('pair audit failure')
            return {'passed': True}
        evidence.audit_pair = audit
        def run(command, env, log):
            calls.append(command)
            if inspect_child is not None:
                inspect_child(command, env, log)
            if failure == 'child':
                raise ValueError('child failure')
            child = Path(command[command.index('--out')+1])
            op6.write(child/'plan.json', {'build': {'root': str(root/'reused')}})
        error = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'op6_evidence': evidence}))
            stack.enter_context(patch.object(op6, 'REPO', root))
            stack.enter_context(patch.object(op6, 'check_overrides'))
            stack.enter_context(patch.object(op6.op3.base, 'verify_controllers'))
            stack.enter_context(patch.object(op6, 'run_child', side_effect=run))
            stack.enter_context(patch.object(op6, 'child_result', return_value={}))
            stack.enter_context(patch.object(op6, 'replay', return_value=op6.report(samples(blocks=blocks), 1, blocks)))
            try:
                with redirect_stdout(io.StringIO()):
                    op6.supervise(args)
            except ValueError as exc:
                error = exc
        return calls, error

    def test_same_build_is_reused_and_both_pairs_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, error = self.run_fixture(root)
            self.assertIsNone(error)
            self.assertEqual([cmd[cmd.index('--variant')+1] for cmd in calls], ['op4', 'op6', 'op6', 'op4'])
            self.assertEqual((root/'build').resolve(), root/'reused')
            self.assertEqual(op6.read(root/'batch-state.json')['status'], 'completed')
            self.assertEqual(set(op6.read(root/'pair-evidence.json')), {'compare1/block00', 'compare1/block01'})

    def test_failed_child_or_pair_stops_without_success_report(self):
        for failure, expected_calls in (('child', 1), ('audit', 2)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, error = self.run_fixture(root, failure)
                self.assertIsNotNone(error)
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(op6.read(root/'batch-state.json')['status'], 'failed')
                self.assertFalse((root/'report.json').exists())


if __name__ == '__main__':
    unittest.main()
