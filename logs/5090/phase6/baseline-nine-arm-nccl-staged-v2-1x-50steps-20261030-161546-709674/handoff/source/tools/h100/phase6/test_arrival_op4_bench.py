"""CPU-only protocol tests; synthetic records never become experiment evidence."""
import copy
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import arrival_op4_bench as bench


def events(arm):
    output = []
    for invocation in range(4):
        names = ['flux_bf16_reducescatter', 'CudaIpcBarrierAllKernel',
                 'taco_op3_v2_decode(' if arm in ('D', 'F') else 'taco_decode_ring_kernel(']
        if arm in ('A', 'D'):
            names.append('CudaIpcBarrierAllKernel')
            output.append(dict(cat='cpu_op', name='aten::clone'))
        for i, name in enumerate(names):
            output.append(dict(cat='kernel', name=name, ts=invocation * 100 + i * 10,
                               dur=5, args={'stream': 7}))
    return output


class PathTests(unittest.TestCase):
    def test_four_paths(self):
        for arm in bench.ARMS:
            result = bench.trace_path(events(arm), arm)
            self.assertTrue(result['execution_path_validated'])
            self.assertEqual(result['clone_count'], 4 if arm in ('A', 'D') else 0)

    def test_extra_barrier_rejected(self):
        with self.assertRaisesRegex(ValueError, 'counts'):
            bench.trace_path(events('A'), 'L')

    def test_unremoved_clone_rejected(self):
        trace = events('F') + [dict(cat='cpu_op', name='aten::clone')]
        with self.assertRaisesRegex(ValueError, 'clone'):
            bench.trace_path(trace, 'F')

    def test_wrong_decoder_rejected(self):
        with self.assertRaisesRegex(ValueError, 'counts'):
            bench.trace_path(events('L'), 'F')

    def test_barrier_after_decode_rejected(self):
        trace = events('F')
        trace[1]['ts'] = 25
        with self.assertRaisesRegex(ValueError, 'order'):
            bench.trace_path(trace, 'F')

    def test_different_stream_rejected(self):
        trace = events('F')
        trace[1]['args']['stream'] = 8
        with self.assertRaisesRegex(ValueError, 'order'):
            bench.trace_path(trace, 'F')

    def test_old_arrival_kernel_rejected(self):
        trace = events('F') + [dict(cat='kernel', name='taco_op3_v2_stop', ts=500, dur=1)]
        with self.assertRaisesRegex(ValueError, 'Arrival'):
            bench.trace_path(trace, 'F')

    def test_schedules_balance_positions_and_reverse(self):
        for blocks in (4, 8, 12, 24):
            jobs = bench.schedule(blocks)
            rows = [[j['arm'] for j in jobs if j['block'] == b] for b in range(blocks)]
            self.assertEqual(rows[blocks//2:], [r[::-1] for r in rows[:blocks//2][::-1]])
            for pos in range(4):
                self.assertEqual({a: sum(r[pos] == a for r in rows) for a in bench.ARMS},
                                 {a: blocks//4 for a in bench.ARMS})
        for blocks in (0, 2, 6, 12.0):
            with self.assertRaises(ValueError):
                bench.schedule(blocks)

    def test_output_clone_only_for_nonowners(self):
        class Tensor:
            def clone(self): return 'cloned'
        class Op:
            def forward(self, *_): return Tensor()
        op = Op()
        self.assertEqual(bench.owned_forward(op, None, None), 'cloned')
        op.owns_output = True
        self.assertIsInstance(bench.owned_forward(op, None, None), Tensor)

    def test_dry_run_keeps_virtualenv_python_symlink(self):
        with tempfile.TemporaryDirectory(prefix='op4-python-') as folder:
            root = Path(folder)
            interpreter = root/'venv/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to('/usr/bin/python3')
            args = argparse.Namespace(root=root, python=interpreter, profile=False,
                                      execute=False, cuda_home=None)
            stdout = io.StringIO()
            with patch.object(bench, 'verify', return_value={'schedule':[]}), contextlib.redirect_stdout(stdout):
                bench.run(args)
            self.assertEqual(json.loads(stdout.getvalue())['command'][0], str(interpreter))

    def test_running_controller_must_match_frozen_worker(self):
        with tempfile.TemporaryDirectory(prefix='op4-frozen-') as folder:
            root = Path(folder)
            worker = root/'scripts/bench.py'
            worker.parent.mkdir()
            worker.write_text('# A different frozen controller\n')
            plan = dict(protocol=bench.PROTOCOL, arms=list(bench.ARMS), roles=bench.ROLES,
                        source_files={'scripts/bench.py':bench.sha(worker)},
                        schedule=bench.schedule(4), retained_output_calls=6)
            bench.write(root/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'controller'):
                bench.verify(root)
            self.assertEqual(bench.verify(root, offline=True), plan)


class ReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='op4-synthetic-cpu-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.plan = dict(protocol=bench.PROTOCOL, run_id='CPU_ONLY', schedule=bench.schedule(4),
            shared_environment={}, communication={}, roles=bench.ROLES, seeds=[17, 19], reuse=2,
            retained_output_calls=6, slow_rank_stress={'ranks':[0,7], 'cycles':2000000},
            warmup=1, iterations=2)
        bench.write(self.root/'plan.json', self.plan)
        for job in self.plan['schedule']:
            dest = self.root/'results'/job['directory']
            dest.mkdir(parents=True)
            for rank in range(8):
                owned = job['arm'] in ('L','F')
                checks = []
                for epoch in range(2):
                    for seed in self.plan['seeds']:
                        r, cycles = bench.correctness_delay(self.plan, epoch, seed)
                        checks.append(dict(epoch=epoch, seed=seed, delayed_rank=r, delay_cycles=cycles,
                            inputs_sha256=f'i{rank}{epoch}{seed}', weights_sha256=f'w{rank}{epoch}{seed}',
                            output_sha256=f'o{rank}{epoch}{seed}', correctness={'passed':True}))
                burst = [dict(generation=g, inputs_sha256=f'bi{g}', weights_sha256=f'bw{g}',
                              output_sha256=f'bo{g}', correctness={'passed':True}) for g in range(6)]
                row = dict(completed=True, protocol=bench.PROTOCOL, run_id='CPU_ONLY',
                    plan_sha256=bench.sha(self.root/'plan.json'), rank=rank, job=job, profiling=False,
                    execution_environment={'H100_OP4_ARM':job['arm'], 'H100_OP4_PROTOCOL':bench.SOURCE_PROTOCOL},
                    hardware={'synthetic':True}, software={'synthetic':True}, roles={})
                for role, shape in bench.ROLES.items():
                    row['roles'][role] = dict(shape=shape, checks=checks, retained_output_checks=burst,
                        owns_output=owned, workspace_count=2 if owned else 1,
                        packet_pointers=[11,22] if owned else [11],
                        timing=dict(iterations=2, warmup=1, wall_us=80 if job['arm']=='F' else 100))
                bench.write(dest/f'rank{rank}.json', row)
        patcher = patch.object(bench, 'verify', return_value=self.plan)
        patcher.start()
        self.addCleanup(patcher.stop)

    def change(self, fn):
        job = next(j for j in self.plan['schedule'] if j['arm']=='F')
        p = self.root/'results'/job['directory']/'rank7.json'
        row = bench.read(p)
        fn(row['roles']['attn'])
        p.write_text(json.dumps(row))

    def test_complete_report_is_diagnostic_only(self):
        result = bench.report(self.root, save=False)
        self.assertFalse(result['target_validated'])
        self.assertAlmostEqual(result['roles']['attn']['comparisons']['F/A']['percent'], 20)
        self.assertEqual(len(result['all_samples']),16)

    def test_retained_output_changed_rejected(self):
        self.change(lambda r: r['retained_output_checks'][3].update(output_sha256='stale'))
        with self.assertRaisesRegex(ValueError, 'retained output'):
            bench.report(self.root, save=False)

    def test_aliasing_workspace_rejected(self):
        self.change(lambda r: r.update(packet_pointers=[11,11]))
        with self.assertRaisesRegex(ValueError, 'workspace'):
            bench.report(self.root, save=False)

    def test_bad_codec_rejected(self):
        self.change(lambda r: r['checks'][0]['correctness'].update(passed=False))
        with self.assertRaisesRegex(ValueError, 'correctness'):
            bench.report(self.root, save=False)

    def test_missing_rank_rejected(self):
        (self.root/'results'/self.plan['schedule'][0]['directory']/'rank6.json').unlink()
        with self.assertRaises(FileNotFoundError):
            bench.report(self.root, save=False)

    def test_negative_savings_does_not_become_acceptance(self):
        samples = [dict(block=b, arm=a, rankmax_wall_us={role:200 if a=='F' else 100 for role in bench.ROLES})
                   for b in range(4) for a in bench.ARMS]
        self.assertLess(bench.savings_estimate(samples)['mean_ms'],0)

    def test_saved_report_recomputed_without_overwrite(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = bench.report(self.root)
            self.assertEqual(bench.report(self.root, offline=True), first)
        path = self.root/'report.json'
        changed = dict(first, target_validated=True)
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'Saved operator report'):
            bench.report(self.root, offline=True)
        self.assertEqual(bench.read(path), changed)


if __name__ == '__main__':
    unittest.main()
