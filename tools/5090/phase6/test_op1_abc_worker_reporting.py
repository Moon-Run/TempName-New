"""Execute generated worker reporting in lifecycle order with CPU-only stubs.

These tests run the real warmup/timing/profile branches and final JSON write.
Operators start absent and are created by the first fake model step. Their
state is sampled from live counters, never inserted into a fabricated report.
"""
import ast
from contextlib import AbstractContextManager
import gzip
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import op1_abc_protocol as p
import op1_abc_sources as sources


def transport_assignment(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "report['op1_abc_transport']")


def reporting_program(source, move_snapshot=None):
    tree = ast.parse(source)
    if move_snapshot:
        capture = next(node for node in tree.body if transport_assignment(node))
        tree.body.remove(capture)
        anchor = 'clear_gradients()' if move_snapshot == 'startup' else 'adapter.stop_audit()'
        position = next(i for i, node in enumerate(tree.body) if ast.unparse(node) == anchor)
        tree.body.insert(position if move_snapshot == 'startup' else position+1, capture)
    start = next(i for i, node in enumerate(tree.body) if ast.unparse(node) == 'clear_gradients()')
    stop = next(i for i, node in enumerate(tree.body)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'write_text'
                and 'json.dumps(report' in ast.unparse(node))
    # Keep an accidentally early capture at its original lifecycle position.
    # Dropping it would conceal the exact failure this regression addresses.
    prefix = [node for node in tree.body[:start] if transport_assignment(node)]
    return compile(ast.Module(body=prefix+tree.body[start:stop+1], type_ignores=[]),
                   '<generated-worker-reporting>', 'exec')


class WorkerLifecycle:
    def __init__(self, arm, mode, output):
        self.arm, self.mode, self.output = arm, mode, output
        self.steps, self.last_sync = 0, -1
        self.snapshots = []
        self.native = arm == 'flux-improved'
        self.names = [f'decoder.layers.{layer}.{suffix}' for layer in range(32)
                      for suffix in ('self_attention.linear_proj', 'mlp.linear_fc2')]
        self.route = 'candidate' if self.native else 'flux_tensor'
        self.routes = dict.fromkeys(self.names, self.route)
        self.adapter = SimpleNamespace(
            modules=dict.fromkeys(self.names), scope_ops={}, observed={}, audit=True,
            calls=dict.fromkeys(self.names, 0), timed_calls={}, forward_checks={},
            placements={}, selection_checks={}, actual_routes=self.routes,
            post_audit_routes={}, forward_preserved={}, workspace_checks={},
            smoke_outputs={}, retention_checks={})
        self.adapter.stop_audit = self.stop_audit
        self.adapter.codec_calls = lambda: {name: op.codec.tensor_codec.calls
                                           for name, op in self.adapter.scope_ops.items()}

    def new_operator(self, name):
        owner = self

        class Codec:
            calls = 0

            def state(self):
                owner.snapshots.append((name, owner.steps, owner.last_sync, self.calls))
                return dict(calls=self.calls, sampled_after_step=owner.steps)

        codec = Codec()
        return SimpleNamespace(
            codec=SimpleNamespace(tensor_codec=codec,
                transport_state=lambda: {'slots': [{'native_workspace_bytes': 128}]*2}),
            owns_output=True, owns_configuration=True, workspace_info=[{}, {}])

    def step(self):
        self.steps += 1
        for name in self.names:
            if name not in self.adapter.scope_ops:
                self.adapter.scope_ops[name] = self.new_operator(name)
                self.adapter.observed[name] = {'M': 2048, 'N': 4096}
            self.adapter.scope_ops[name].codec.tensor_codec.calls += 4
            counters = self.adapter.calls if self.adapter.audit else self.adapter.timed_calls
            counters[name] += 4
        if self.mode == 'smoke':
            self.adapter.smoke_outputs = dict.fromkeys(self.names, 'a'*64)
            self.adapter.retention_checks = {'attn': {'passed': True}, 'mlp': {'passed': True}}
        return float(self.steps), True, 1.0

    def stop_audit(self):
        self.adapter.audit = False
        self.adapter.post_audit_routes = self.routes
        self.adapter.forward_preserved = dict.fromkeys(self.names, False)
        self.adapter.timed_calls = dict.fromkeys(self.names, 0)

    def synchronize(self):
        self.last_sync = self.steps

    def namespace(self):
        class Profiler(AbstractContextManager):
            def __exit__(self, *args):
                return False

            def export_chrome_trace(self, path):
                Path(path).write_text('{"traceEvents": []}')

        ticks = iter((10.0, 11.0))
        return dict(
            adapter=self.adapter, MODE=self.mode, POLICY=self.arm, PLAN={'arm': self.arm},
            SPEC=dict(p.TIMING, model=p.MODEL), microsteps=4,
            report={'tokens_per_optimizer_step': p.TOKENS_PER_STEP},
            clear_gradients=lambda: None, step=self.step, loss_number=float,
            module_routes=lambda *args: self.routes, is_native=lambda policy: False,
            standalone_codec=SimpleNamespace(loaded_library=lambda: None), codec_validation=None,
            dist=SimpleNamespace(barrier=lambda: None),
            torch=SimpleNamespace(cuda=SimpleNamespace(
                synchronize=self.synchronize, reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048),
                profiler=SimpleNamespace(profile=lambda **kwargs: Profiler(),
                    ProfilerActivity=SimpleNamespace(CPU='cpu', CUDA='cuda'))),
            os=SimpleNamespace(environ={'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'FLUX_GEMM_RS_GEMM_ONLY': '1'}),
            consumer_parameter_trajectory=[], consumer_parameter_universe={},
            fingerprint=lambda values: 'a'*64, raw=SimpleNamespace(named_parameters=lambda: []),
            time=SimpleNamespace(perf_counter=lambda: next(ticks)), math=math,
            tempfile=tempfile, Path=Path, gzip=gzip, json=json, OUT=self.output, rank=0)

    def execute(self, program):
        namespace = self.namespace()
        exec(program, namespace)
        return json.loads((self.output/'rank0.json').read_text())


class WorkerReporting(unittest.TestCase):
    def test_all_abc_modes_serialize_live_state_after_last_step_and_sync(self):
        for arm in p.BASIC_ARMS:
            for mode in ('smoke', 'profile', 'timing'):
                with self.subTest(arm=arm, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    worker = WorkerLifecycle(arm, mode, Path(tmp))
                    self.assertFalse(worker.adapter.scope_ops)
                    report = worker.execute(reporting_program(sources.worker_source(arm)))
                    warmup = p.TIMING['smoke_warmup_steps'] if mode == 'smoke' else p.TIMING['warmup_steps']
                    measured = 1 if mode == 'profile' else p.TIMING['smoke_steps' if mode == 'smoke' else 'timed_steps']
                    expected_steps = warmup+measured
                    self.assertEqual(worker.steps, expected_steps)
                    self.assertTrue(report['completed'])
                    self.assertEqual(set(report['op1_abc_transport']), set(worker.names))
                    self.assertEqual(report['codec_calls'], dict.fromkeys(worker.names, expected_steps*4))
                    self.assertEqual(len(worker.snapshots), 64)
                    for _, step, synced, calls in worker.snapshots:
                        self.assertEqual((step, synced, calls), (expected_steps, expected_steps, expected_steps*4))
                    self.assertTrue(all(value['calls'] == expected_steps*4
                                        for value in report['op1_abc_transport'].values()))

    def test_comparator_retains_native_state_without_abc_snapshots(self):
        for mode in ('smoke', 'profile', 'timing'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                worker = WorkerLifecycle('flux-improved', mode, Path(tmp))
                report = worker.execute(reporting_program(sources.worker_source('flux-improved')))
                self.assertEqual(report['op1_abc_transport'], {})
                self.assertEqual(set(report['rtx5090_transport']), set(worker.names))
                self.assertEqual(worker.snapshots, [])

    def test_startup_snapshot_regression_fails_before_rank_json_is_written(self):
        for arm in p.BASIC_ARMS:
            for mode in ('smoke', 'profile', 'timing'):
                with self.subTest(arm=arm, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    worker = WorkerLifecycle(arm, mode, Path(tmp))
                    with self.assertRaisesRegex(AssertionError, 'Incomplete final ABC transport coverage'):
                        worker.execute(reporting_program(sources.worker_source(arm), 'startup'))
                    self.assertFalse((Path(tmp)/'rank0.json').exists())

    def test_warmup_only_snapshot_regression_rejects_stale_counters(self):
        for arm in p.BASIC_ARMS:
            for mode in ('smoke', 'profile', 'timing'):
                with self.subTest(arm=arm, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    worker = WorkerLifecycle(arm, mode, Path(tmp))
                    with self.assertRaisesRegex(AssertionError, 'Stale final ABC transport counters'):
                        worker.execute(reporting_program(sources.worker_source(arm), 'after_warmup'))
                    self.assertFalse((Path(tmp)/'rank0.json').exists())


if __name__ == '__main__':
    unittest.main()
