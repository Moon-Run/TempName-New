"""Real CPU autograd / original DDP hook tests; native CUDA arithmetic is mocked.

These tests exercise the actual accumulation hook ordering, lifetime and install
boundary. They are not GPU performance or native floating-point evidence.
"""
import ast
import ctypes
import gc
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import ma10_adapter as ma10

REPO = Path(__file__).resolve().parents[3]
MEGATRON = REPO.parent / 'Megatron-LM/megatron/core'


class CallableABI:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class CPUReferenceABI:
    def __init__(self):
        self.status = 0
        self.calls = []
        self.ma10_gradient_batching_protocol = CallableABI(lambda: 10)
        self.ma10_gradient_batching_chunk_size = CallableABI(lambda: ma10.CHUNK_SIZE)
        self.ma10_gradient_batching_max_tensors = CallableABI(lambda: ma10.TENSOR_LIMIT)
        self.ma10_gradient_batching_cuda = CallableABI(self.launch)

    def launch(self, targets, sources, sizes, count, stream):
        if self.status:
            return self.status
        self.calls.append((count, list(sizes), stream))
        for i in range(count):
            target = torch.frombuffer((ctypes.c_float * sizes[i]).from_address(targets[i]),
                                      dtype=torch.float32)
            source = torch.frombuffer((ctypes.c_int16 * sizes[i]).from_address(sources[i]),
                                      dtype=torch.bfloat16)
            target.add_(source)
        return 0


class CPUTorch:
    def __init__(self):
        self.stream = types.SimpleNamespace(cuda_stream=17)
        self.capturing = False
        self.cuda = types.SimpleNamespace(current_device=lambda: 0,
            current_stream=lambda device=None: self.stream,
            is_current_stream_capturing=lambda: self.capturing)

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, *args):
        return torch.device('cpu')

    def tensor(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return torch.tensor(*args, **kwargs)

    def arange(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return torch.arange(*args, **kwargs)


def load_fixture(case, backend):
    """Use exact whole-line source, so inspect.getsource sees original indentation."""
    text = (MEGATRON / 'distributed/distributed_data_parallel.py').read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
               and n.name == 'DistributedDataParallel')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == '_make_backward_post_hook')
    method = ''.join(text.splitlines(True)[node.lineno-1:node.end_lineno])
    text = (MEGATRON / 'pipeline_parallel/schedules.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                and n.name == 'backward_step')
    backward = ''.join(text.splitlines(True)[node.lineno-1:node.end_lineno])
    tmp = tempfile.TemporaryDirectory(prefix='ma10-source-')
    case.addCleanup(tmp.cleanup)
    path = Path(tmp.name) / 'fixture.py'
    source = ('class DistributedDataParallel:\n' + method + '\n\n' + backward)
    path.write_text(source)
    module = types.ModuleType('ma10_fixture')
    module.__dict__.update(torch=backend, is_graph_capturing=lambda: False,
        parallel_state=types.SimpleNamespace(get_pipeline_model_parallel_world_size=lambda: 1),
        ModelType=types.SimpleNamespace(encoder_and_decoder='encoder_and_decoder'))
    module.__file__ = str(path)
    reg = patch.dict(sys.modules, {module.__name__: module})
    reg.start()
    case.addCleanup(reg.stop)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def make_owner(cls, shapes):
    owner = cls()
    owner.config = types.SimpleNamespace(gradient_accumulation_fusion=False,
        sequence_parallel=True, tensor_model_parallel_size=8, pipeline_model_parallel_size=1,
        context_parallel_size=1, recompute_granularity=None, timers=None,
        grad_scale_func=None, deallocate_pipeline_outputs=False)
    owner.ddp_config = types.SimpleNamespace(overlap_grad_reduce=False,
        use_distributed_optimizer=False, overlap_param_gather=False)
    owner.module = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
                                           for shape in shapes])
    owner.param_to_bucket_group = {}
    owner.grad_accs = []
    for p in owner.module.parameters():
        p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        p.grad_added_to_main_grad = False
        owner.param_to_bucket_group[p] = object()
        node = p.expand_as(p).grad_fn.next_functions[0][0]
        node.register_hook(owner._make_backward_post_hook(p))
        owner.grad_accs.append(node)
    return owner


def loss_for(owner, microbatch):
    # Every tensor has one AccumulateGrad, even when the same parameter is used
    # twice in the graph. Its BF16 leaf gradient is retained for the native ABI.
    parts = []
    for i, p in enumerate(owner.module.parameters()):
        scale = float(((microbatch + 1) * (i + 3)) % 41 - 20) / 16
        parts.append((p.float() * scale).sum())
    return sum(parts)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.backend = CPUTorch()
        self.fixture = load_fixture(self, self.backend)
        self.library = CPUReferenceABI()

    def tracker(self, shapes, mode='timing'):
        cls = self.fixture.DistributedDataParallel
        factory = cls._make_backward_post_hook
        original_backward = self.fixture.backward_step
        tracker = ma10.GradientBatcher(self.backend, mode, cls, self.fixture, factory, original_backward)
        tracker.factory = lambda owner, param: tracker.make_hook(owner, param)
        cls._make_backward_post_hook = tracker.factory
        self.addCleanup(setattr, cls, '_make_backward_post_hook', factory)
        self.addCleanup(setattr, self.fixture, 'backward_step', original_backward)
        owner = make_owner(cls, shapes)
        tracker.bind([owner], self.library, expected_parameters=len(shapes))
        return tracker, owner

    def run_backward(self, owner, loss):
        return self.fixture.backward_step(None, loss, None, 'encoder', owner.config)

    def test_original_source_fingerprints_use_runtime_inspect(self):
        ma10.require_source(self.fixture.DistributedDataParallel._make_backward_post_hook,
                            '_make_backward_post_hook')
        ma10.require_source(self.fixture.backward_step, 'backward_step')

    def test_four_microbatches_three_updates_match_original_ddp_hooks(self):
        cls = self.fixture.DistributedDataParallel
        original = self.fixture.backward_step
        shapes = [(i % 7 + 1,) for i in range(389)]
        reference = make_owner(cls, shapes)
        tracker, candidate = self.tracker(shapes)
        for step in range(3):
            for micro in range(4):
                original(None, loss_for(reference, micro), None, 'encoder', reference.config)
                self.run_backward(candidate, loss_for(candidate, micro))
                for a, b in zip(reference.module.parameters(), candidate.module.parameters()):
                    self.assertTrue(torch.equal(a.main_grad.view(torch.int32), b.main_grad.view(torch.int32)))
                    self.assertIsNone(b.grad)
            with torch.no_grad():
                for a, b in zip(reference.module.parameters(), candidate.module.parameters()):
                    a.add_(a.main_grad, alpha=-0.03125)
                    b.add_(b.main_grad, alpha=-0.03125)
                    self.assertTrue(torch.equal(a, b))
                    a.main_grad.zero_()
                    b.main_grad.zero_()
        state = tracker.state()
        self.assertEqual(state['microbatches'], 12)
        self.assertEqual(state['tensor_commits'], 389 * 12)
        self.assertEqual(state['flushes'], 13 * 12)
        self.assertTrue(state['original_factory_restored'])

    def test_byte_bound_and_large_single_tensor(self):
        tracker, owner = self.tracker([(3,), (7,), (20,), (2,)])
        with patch.object(ma10, 'BYTE_LIMIT', 16):
            self.run_backward(owner, loss_for(owner, 0))
        self.assertLessEqual(tracker.max_pending_bytes, 40)
        self.assertTrue(any(size == [20] for _, size, _ in self.library.calls))
        self.assertEqual(tracker.tensor_commits, 4)

    def test_smoke_audits_native_batches_and_does_not_change_rng(self):
        before = torch.random.get_rng_state().clone()
        tracker, owner = self.tracker([(7,), (ma10.CHUNK_SIZE + 1,)], mode='smoke')
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.run_backward(owner, loss_for(owner, 0))
        state = tracker.state()
        self.assertEqual(state['audits'], state['flushes'])
        self.assertEqual(state['self_test']['cases'], 12)

    def test_shared_parameter_alias_has_one_commit_after_all_contributions(self):
        tracker, owner = self.tracker([(5,)])
        p = owner.module[0]
        loss = (p * 2).sum() + (p * 3).sum()
        self.run_backward(owner, loss)
        self.assertTrue(torch.equal(p.main_grad, torch.full((5,), 5.)))
        self.assertEqual(tracker.hooks, 1)

    def test_reference_factory_restored_before_first_forward(self):
        original = self.fixture.DistributedDataParallel._make_backward_post_hook
        tracker, owner = self.tracker([(3,)])
        self.assertIs(self.fixture.DistributedDataParallel._make_backward_post_hook, original)
        # A later separately constructed model receives ORIGINAL hooks, so it
        # cannot silently become part of the candidate's parameter universe.
        other = make_owner(self.fixture.DistributedDataParallel, [(3,)])
        self.fixture.backward_step = tracker.original_backward
        tracker.original_backward(None, loss_for(other, 0), None, 'encoder', other.config)
        self.assertEqual(tracker.hooks, 0)
        self.assertTrue(torch.count_nonzero(other.module[0].main_grad).item() > 0)

    def test_missing_parameter_fails_before_gradient_consumer(self):
        tracker, owner = self.tracker([(2,), (2,)])
        with self.assertRaisesRegex(RuntimeError, 'every trainable gradient'):
            self.run_backward(owner, owner.module[0].sum())
        self.assertTrue(tracker.failed)
        with self.assertRaises(RuntimeError):
            tracker.state()

    def test_duplicate_hook_missing_grad_flags_and_stream_fail_closed(self):
        for kind in ('duplicate', 'none', 'flag', 'stream'):
            with self.subTest(kind=kind):
                self.fixture = load_fixture(self, self.backend)
                tracker, owner = self.tracker([(2,)])
                p = owner.module[0]
                tracker.active = True
                p.grad = torch.ones_like(p)
                if kind == 'duplicate':
                    tracker.seen.add(id(p))
                elif kind == 'none':
                    p.grad = None
                elif kind == 'flag':
                    p.grad_added_to_main_grad = True
                elif kind == 'stream':
                    self.backend.stream = types.SimpleNamespace(cuda_stream=18)
                with self.assertRaises(RuntimeError):
                    tracker.enqueue(owner, p)
                self.backend.stream = types.SimpleNamespace(cuda_stream=17)
                self.backend.capturing = False

    def test_capture_is_rejected_at_microbatch_entry(self):
        tracker, owner = self.tracker([(2,)])
        self.backend.capturing = True
        with self.assertRaisesRegex(RuntimeError, 'CUDA Graph capture'):
            self.run_backward(owner, loss_for(owner, 0))
        self.assertEqual(tracker.hooks, 0)

    def test_native_failure_stops_state_and_next_backward(self):
        tracker, owner = self.tracker([(2,)])
        self.library.status = 11
        with self.assertRaisesRegex(RuntimeError, 'CUDA launch failed'):
            self.run_backward(owner, loss_for(owner, 0))
        self.assertTrue(tracker.failed)
        with self.assertRaises(RuntimeError):
            tracker.state()
        with self.assertRaises(RuntimeError):
            self.run_backward(owner, loss_for(owner, 0))

    def test_gradient_source_alias_with_any_target_is_rejected(self):
        tracker, owner = self.tracker([(4,), (4,)])
        p, other = owner.module
        p.grad = other.main_grad.view(torch.bfloat16)[:4]
        tracker.active = True
        with self.assertRaisesRegex(RuntimeError, 'source overlaps'):
            tracker.enqueue(owner, p)

    def test_bind_rejects_overlapping_destinations_and_restores_factory(self):
        cls = self.fixture.DistributedDataParallel
        original = cls._make_backward_post_hook
        tracker = ma10.GradientBatcher(self.backend, 'timing', cls, self.fixture,
                                      original, self.fixture.backward_step)
        tracker.factory = lambda owner, p: tracker.make_hook(owner, p)
        cls._make_backward_post_hook = tracker.factory
        self.addCleanup(setattr, cls, '_make_backward_post_hook', original)
        owner = make_owner(cls, [(4,), (4,)])
        owner.module[1].main_grad = owner.module[0].main_grad
        with self.assertRaisesRegex(RuntimeError, 'destinations overlap'):
            tracker.bind([owner], self.library, expected_parameters=2)
        self.assertIs(cls._make_backward_post_hook, original)
        self.assertTrue(tracker.failed)

    def test_source_ownership_survives_param_grad_none_until_commit(self):
        tracker, owner = self.tracker([(13,)])
        p = owner.module[0]
        tracker.active = True
        p.grad = torch.arange(13, dtype=torch.bfloat16)
        tracker.enqueue(owner, p)
        self.assertIsNone(p.grad)
        gc.collect()
        tracker.flush()
        self.assertTrue(torch.equal(p.main_grad, torch.arange(13, dtype=torch.float32)))

    def test_noncontiguous_and_replaced_main_grad_are_rejected(self):
        for replace in (False, True):
            self.fixture = load_fixture(self, self.backend)
            tracker, owner = self.tracker([(2, 2)])
            p = owner.module[0]
            tracker.active = True
            p.grad = torch.ones((2, 2), dtype=torch.bfloat16)
            if replace:
                p.main_grad = p.main_grad.clone()
            else:
                p.grad = p.grad.t()
            with self.assertRaises(RuntimeError):
                tracker.enqueue(owner, p)

    def test_public_prepare_and_install_only_patch_candidate_construction(self):
        cls = self.fixture.DistributedDataParallel
        factory = cls._make_backward_post_hook
        packages = {name: types.ModuleType(name) for name in (
            'megatron', 'megatron.core', 'megatron.core.distributed',
            'megatron.core.distributed.distributed_data_parallel',
            'megatron.core.pipeline_parallel')}
        packages['megatron.core.distributed.distributed_data_parallel'].DistributedDataParallel = cls
        packages['megatron.core.pipeline_parallel'].schedules = self.fixture
        packages['megatron.core'].parallel_state = types.SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 8,
            get_pipeline_model_parallel_world_size=lambda: 1,
            get_data_parallel_world_size=lambda: 1)
        with patch.dict(sys.modules, dict(packages, torch=self.backend)):
            tracker = ma10.prepare_gradient_batching('timing')
            try:
                owner = make_owner(cls, [(1,)] * 389)
                ma10.install_gradient_batching(tracker, [owner], self.library, 'timing')
                self.assertIs(cls._make_backward_post_hook, factory)
                self.run_backward(owner, loss_for(owner, 0))
                self.assertEqual(tracker.state()['tensor_commits'], 389)
            finally:
                cls._make_backward_post_hook = factory


if __name__ == '__main__':
    unittest.main()
