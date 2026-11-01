"""Execute generated OP3 setup and final reporting with lazy CPU operators."""
import ast
from contextlib import ExitStack
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rtx5090_loader import H100, HERE
import op3_protocol as p
import op3_sources as sources
import op2_abc_build as optimizers
from test_op1_abc_worker_reporting import WorkerLifecycle


def library_plan(root):
    result = {}
    for name in ('ma9', 'ma11'):
        path = root/(name+'.so')
        path.write_bytes(b'CPU-FIXTURE-'+name.encode())
        result[name+'_compat'] = dict(library=str(path),
            library_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def transport_assignment(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "report['op3_transport']")


def reporting_program(source, snapshot=None):
    tree = ast.parse(source)
    if snapshot:
        capture = next(node for node in tree.body if transport_assignment(node))
        tree.body.remove(capture)
        anchor = 'clear_gradients()' if snapshot == 'startup' else 'adapter.stop_audit()'
        position = next(i for i, node in enumerate(tree.body) if ast.unparse(node) == anchor)
        tree.body.insert(position if snapshot == 'startup' else position+1, capture)
    start = next(i for i, node in enumerate(tree.body) if ast.unparse(node) == 'clear_gradients()')
    stop = next(i for i, node in enumerate(tree.body)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'write_text'
                and 'json.dumps(report' in ast.unparse(node))
    prefix = [node for node in tree.body[:start] if transport_assignment(node)]
    return compile(ast.Module(body=prefix+tree.body[start:stop+1], type_ignores=[]),
                   '<op3-generated-worker-reporting>', 'exec')


class GeneratedWorkers(unittest.TestCase):
    def execute_worker(self, root, arm, mode, snapshot=None):
        worker = WorkerLifecycle(arm, mode, root)
        namespace = worker.namespace()
        libraries = optimizers.libraries(library_plan(root))
        namespace['PLAN'].update(op3_optimization_libraries=libraries,
                                op3_optimization_stack=sources.STACK)
        namespace.update(hashlib=hashlib, optimizer=object())
        calls, tracker_snapshots = [], []
        modules = {name: ModuleType(name) for name in ('ma8_adapter', 'ma9_adapter', 'ma11_adapter', 'ctypes')}
        modules['ctypes'].CDLL = lambda path: path
        def install_cache(raw, arm):
            calls.append(('cache', worker.steps))
            return {'installed': True}
        def state(name):
            tracker_snapshots.append((name, worker.steps, worker.last_sync))
            return dict(calls=worker.steps)
        def install_counter(optimizer, library, mode):
            calls.append(('zero', worker.steps))
            return SimpleNamespace(state=lambda: state('zero'))
        def install_softmax(raw, library, mode):
            calls.append(('softmax', worker.steps))
            return SimpleNamespace(state=lambda: state('softmax'))
        modules['ma8_adapter'].install_activation_cache = install_cache
        modules['ma9_adapter'].install_zero_count = install_counter
        modules['ma11_adapter'].install_softmax = install_softmax
        worker.adapter.consumer_state = lambda: state('consumer')
        source = sources.worker_source(arm)
        tree = ast.parse(source)
        begin = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Import)
                     and any(a.name == 'ctypes' for a in n.names))
        end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'ma11_softmax_tracker' for t in n.targets))
        with patch.dict(sys.modules, modules):
            exec(compile(ast.Module(body=tree.body[begin:end+1], type_ignores=[]),
                         '<op3-real-optimization-setup>', 'exec'), namespace)
            self.assertFalse(worker.adapter.scope_ops)
            self.assertEqual(calls, [('cache', 0), ('zero', 0), ('softmax', 0)])
            exec(reporting_program(source, snapshot), namespace)
        return worker, json.loads((root/'rank0.json').read_text()), tracker_snapshots, libraries

    def test_both_arms_all_modes_capture_actual_counters_after_final_step_and_sync(self):
        for arm, mode in itertools.product(p.ARMS, ('smoke', 'profile', 'timing')):
            with self.subTest(arm=arm, mode=mode), tempfile.TemporaryDirectory() as tmp:
                worker, report, tracker_snapshots, libraries = self.execute_worker(Path(tmp), arm, mode)
                warmup = p.TIMING['smoke_warmup_steps'] if mode == 'smoke' else p.TIMING['warmup_steps']
                measured = 1 if mode == 'profile' else p.TIMING['smoke_steps' if mode == 'smoke' else 'timed_steps']
                steps = warmup+measured
                self.assertEqual(worker.steps, steps)
                self.assertTrue(report['completed'])
                self.assertEqual(set(report['op3_transport']), set(worker.names))
                self.assertEqual(report['codec_calls'], dict.fromkeys(worker.names, steps*4))
                self.assertEqual(len(worker.snapshots), 64)
                self.assertTrue(all((step, synced, calls) == (steps, steps, steps*4)
                                    for _, step, synced, calls in worker.snapshots))
                self.assertEqual(report['ma11_zero_count_state']['calls'], steps)
                self.assertEqual(report['ma11_softmax_state']['calls'], steps)
                self.assertEqual(report['op3_optimization_libraries'], libraries)
                self.assertEqual(report['op3_optimization_stack'], sources.STACK)
                expected_trackers = ['zero', 'softmax']+(['consumer'] if arm == 'op3-r' else [])
                self.assertEqual(tracker_snapshots, [(name, steps, steps) for name in expected_trackers])
                self.assertEqual(report['op3_consumer_state'], {'calls': steps} if arm == 'op3-r' else {})
                sources.base.validate_worker(sources.worker_source(arm))

    def test_startup_or_warmup_snapshot_fails_before_rank_record(self):
        for arm, mode, snapshot in itertools.product(p.ARMS, ('smoke', 'profile', 'timing'),
                                                     ('startup', 'after_warmup')):
            with self.subTest(arm=arm, mode=mode, snapshot=snapshot), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                message = 'Incomplete final OP3 transport coverage' if snapshot == 'startup' else 'Stale final OP3 transport counters'
                with self.assertRaisesRegex(AssertionError, message):
                    self.execute_worker(root, arm, mode, snapshot)
                self.assertFalse((root/'rank0.json').exists())

    def test_generated_packages_register_exact_sources_helpers_and_distinct_mechanisms(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            native = root/'parent'
            native.mkdir()
            plan = dict(environment={'torch': optimizers.compat.TORCH}, communication={}, **library_plan(root))
            codec = {'path': '/CPU/libop3_codec.so', 'sha256': 'c'*64}
            stack.enter_context(patch.object(sources.builder, 'codec_library', return_value=codec))
            stack.enter_context(patch.object(sources.builder, 'native_root', return_value=native))
            stack.enter_context(patch.object(sources.builder, 'libraries', return_value={}))
            workers = sources.prepare_workers(root, root, plan)
            self.assertEqual(set(workers), set(p.ARMS))
            for arm, worker in workers.items():
                scripts = Path(worker['root'])
                for name in ('ma8_adapter.py', 'ma9_adapter.py'):
                    self.assertEqual((scripts/name).read_bytes(), (H100/name).read_bytes())
                self.assertEqual((scripts/'op3_reference_codec.py').read_bytes(), (HERE/'op1_abc_codec.py').read_bytes())
                self.assertEqual((scripts/'megatron_taco.py').read_bytes(), (HERE/'op3_codec.py').read_bytes())
                namespace = {}
                exec((scripts/'routing.py').read_text(), namespace)
                self.assertEqual(namespace['POLICIES'], p.ARMS)
                self.assertEqual(namespace['PROTOCOL'], p.PROTOCOL)
                self.assertEqual(namespace['definition'](arm), sources.definition(arm))
                with self.assertRaises(ValueError):
                    namespace['definition']('flux-basic-interleaved')
                cfg = json.loads((scripts/'config.json').read_text())
                self.assertEqual(cfg['op3_optimization_libraries'], optimizers.libraries(plan))
                self.assertEqual(cfg['op3_optimization_stack'], sources.STACK)
                self.assertEqual(cfg['op3_implementation']['r_decoder'], arm == 'op3-r')
                self.assertEqual(cfg['op3_implementation']['cr3_consumer_fusion'], arm == 'op3-r')
                self.assertEqual(cfg['codec_identity']['receiver_implementation'], sources.implementation(arm)['decoder'])
                self.assertEqual((scripts/'build'/arm).resolve(), native)
                for name, digest in worker['files'].items():
                    self.assertEqual(hashlib.sha256((scripts/name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
