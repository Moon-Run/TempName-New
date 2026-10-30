"""CPU execution tests for MA16 dependencies, retained inputs and original backward."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

import ma16_adapter as ma16
import test_ma8_adapter as ma8_tests

REPO = Path(__file__).resolve().parents[3]
LAYERS = REPO.parent / 'Megatron-LM/megatron/core/tensor_parallel/layers.py'


def original_source():
    text = LAYERS.read_text()
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef)
                and node.name == 'LinearWithGradAccumulationAndAsyncCommunication')
    return ma16.cache.normalized(ast.get_source_segment(text, node))


class DeferredGather:
    """Remote rows contain NaNs until wait; local rows carry actual rank data."""
    def __init__(self, rank):
        self.rank, self.events, self.buffers = rank, [], []
        self.local = None

    def __call__(self, output, input, group=None, async_op=False):
        self.events.append(('gather', async_op))
        self.local = input
        output.fill_(float('nan'))
        self.buffers.append(output)

        def finish():
            self.events.append(('wait',))
            output.copy_(torch.cat([input + peer - self.rank for peer in range(8)]))
        if not async_op:
            finish()
        return types.SimpleNamespace(wait=finish)

    def scatter(self, output, input, group=None, async_op=False):
        self.events.append(('scatter', async_op))
        rows = output.shape[0]
        output.copy_(input[self.rank * rows:(self.rank + 1) * rows] * 8)
        return types.SimpleNamespace(wait=lambda: self.events.append(('scatter_wait',)))


class TorchProxy:
    cuda = types.SimpleNamespace(current_device=lambda: 'cpu')

    def __init__(self, gather=None):
        self.gather = gather

    def __getattr__(self, name):
        return getattr(torch, name)

    def matmul(self, left, right, out=None):
        if self.gather is not None:
            self.gather.events.append(('matmul', left.shape[0], left is self.gather.local))
            if not torch.isfinite(left).all():
                raise AssertionError('GEMM read incomplete gathered data')
            if ('wait',) not in self.gather.events and left is not self.gather.local:
                raise AssertionError('Pre-wait GEMM did not read the original local input')
        return torch.matmul(left, right, out=out)


def make_function(rank, candidate):
    io = DeferredGather(rank)
    proxy = TorchProxy(io)
    namespace = dict(torch=proxy, custom_fwd=lambda f: f, custom_bwd=lambda f: f,
        get_tensor_model_parallel_world_size=lambda: 8,
        get_tensor_model_parallel_group=lambda: None,
        get_global_memory_buffer=lambda: None,
        dist_all_gather_func=io, dist_reduce_scatter_func=io.scatter,
        prepare_input_tensors_for_wgrad_compute=lambda grad, inp:
            (grad.contiguous().view(-1, grad.shape[-1]), inp.view(-1, inp.shape[-1])))
    namespace['_ma16_projection'] = lambda value, weight: ma16.local_first_matmul(
        proxy, value, weight, io, None, rank, 8)
    source = (ma16.local_first_class_source(original_source()) if candidate
              else ma16.cache.cached_class_source(original_source()))
    exec(compile(source, '<ma16-cpu-autograd-test>', 'exec'), namespace)
    name = '_MA16LocalFirstLinear' if candidate else '_MA8CachedRawLinear'
    return namespace[name], io


def run_forward_backward(rank, candidate, bias=True):
    function, io = make_function(rank, candidate)
    value = (torch.arange(16).reshape(2, 1, 8).float() / 8 + rank).requires_grad_()
    weight = (torch.arange(24).reshape(3, 8).float() / 8 - 2).requires_grad_()
    offset = torch.tensor([.5, -1., 2.], requires_grad=True) if bias else None
    output = function.apply(value, weight, offset, False, False, True, None, None)
    grad = (torch.arange(output.numel()).reshape_as(output).float() / 8 - 1)
    output.backward(grad)
    return [output.detach(), value.grad, weight.grad] + ([offset.grad] if bias else []), io


def valid_state(steps=3, mode='smoke', rank=4):
    calls = 4 * steps
    return dict(protocol=ma16.PROTOCOL, installed=True,
        local_shape=list(ma16.LOCAL_SHAPE), world_size=8,
        source_sha256=ma16.CLASS_SHA256, backward_sha256=ma16.BACKWARD_SHA256,
        modules={f'decoder.layers.{index}{suffix}':
            dict(role=role, rank=rank, calls=calls,
                 forward_audits=calls if mode == 'smoke' else 0,
                 local_input_audits=calls if mode == 'smoke' else 0,
                 gemms_per_call=2 if rank in (0, 7) else 3,
                 device=rank, stream=17, failed=False, active=False)
            for index in range(32) for suffix, role in ma16.cache.SUFFIX_ROLES.items()})


class SourceTests(unittest.TestCase):
    def test_registered_forward_and_unchanged_original_backward(self):
        source = original_source()
        cached = ma16.cache.cached_class_source(source)
        candidate = ma16.local_first_class_source(source)
        self.assertEqual(hashlib.sha256(candidate.encode()).hexdigest(), ma16.CLASS_SHA256)
        self.assertEqual(ma16.backward_sha256(candidate), ma16.BACKWARD_SHA256)
        self.assertEqual(ma16.backward_sha256(candidate), ma16.backward_sha256(cached))
        self.assertIn('ctx.save_for_backward(input, weight, total_input)', candidate)
        self.assertIn('if bias is not None:\n            output = output + bias', candidate)
        self.assertNotIn('dist_all_gather_func(', candidate)
        with self.assertRaisesRegex(RuntimeError, 'pinned'):
            ma16.local_first_class_source(source.replace('grad_input =', 'grad_input  ='))


class DependencyTests(unittest.TestCase):
    def test_all_ranks_use_local_gemm_before_wait_and_remote_rows_after_wait(self):
        for rank in range(8):
            with self.subTest(rank=rank):
                io = DeferredGather(rank)
                proxy = TorchProxy(io)
                value = torch.arange(16).reshape(2, 1, 8).float() + rank
                weight = torch.arange(24).reshape(3, 8).float() / 8
                total, output = ma16.local_first_matmul(proxy, value, weight, io, None, rank, 8)
                expected = torch.matmul(total, weight.t())
                self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))
                self.assertEqual(io.events[:3], [('gather', True), ('matmul', 2, True), ('wait',)])
                self.assertEqual(sum(event[0] == 'gather' for event in io.events), 1)
                self.assertEqual(sum(event[0] == 'matmul' for event in io.events),
                                 2 if rank in (0, 7) else 3)
                self.assertTrue(output.is_contiguous())
                self.assertIsNone(output._base)
                self.assertEqual(output.stride(), expected.stride())

    def test_original_backward_and_bias_match_on_every_rank(self):
        for rank in range(8):
            for bias in (False, True):
                with self.subTest(rank=rank, bias=bias):
                    reference, ref_io = run_forward_backward(rank, False, bias)
                    actual, io = run_forward_backward(rank, True, bias)
                    for left, right in zip(reference, actual):
                        self.assertTrue(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))
                    self.assertEqual(sum(event[0] == 'gather' for event in io.events), 1)
                    self.assertIn(('scatter', True), io.events)
                    self.assertEqual(io.events[-1], ('scatter_wait',))

    def test_each_forward_owns_distinct_complete_retained_input_and_output(self):
        io = DeferredGather(3)
        value = torch.arange(16).reshape(2, 1, 8).float()
        weight = torch.arange(24).reshape(3, 8).float()
        first, first_out = ma16.local_first_matmul(torch, value, weight, io, None, 3, 8)
        snapshot, output_snapshot = first.clone(), first_out.clone()
        second, second_out = ma16.local_first_matmul(torch, value + 8, weight, io, None, 3, 8)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertNotEqual(first_out.data_ptr(), second_out.data_ptr())
        self.assertTrue(torch.equal(first, snapshot))
        self.assertTrue(torch.equal(first_out, output_snapshot))

    def test_full_gather_is_used_for_wgrad_not_only_local_slice(self):
        _, io = run_forward_backward(5, True)
        self.assertEqual(tuple(io.buffers[0].shape), (16, 1, 8))
        self.assertFalse(torch.equal(io.buffers[0][:2], io.buffers[0][-2:]))

    def test_invalid_rank_is_rejected_before_collective(self):
        io = DeferredGather(0)
        with self.assertRaisesRegex(RuntimeError, 'rank'):
            ma16.local_first_matmul(torch, torch.zeros(2, 1, 8), torch.zeros(3, 8), io, None, 8, 8)
        self.assertEqual(io.events, [])


class EvidenceTests(unittest.TestCase):
    def test_valid_all_ranks_and_modes(self):
        for rank in range(8):
            for mode in ('smoke', 'timing', 'profile'):
                self.assertTrue(ma16.validate_state(valid_state(mode=mode, rank=rank), 3, mode, rank))

    def test_wrong_rank_source_backward_module_and_counts_are_rejected(self):
        for mutate in (
            lambda s: s.update(source_sha256='0' * 64),
            lambda s: s.update(backward_sha256='0' * 64),
            lambda s: s['modules'].pop(next(iter(s['modules']))),
            lambda s: next(iter(s['modules'].values())).update(calls=11),
            lambda s: next(iter(s['modules'].values())).update(rank=3),
            lambda s: next(iter(s['modules'].values())).update(stream=18),
            lambda s: next(iter(s['modules'].values())).update(active=True),
            lambda s: next(iter(s['modules'].values())).update(failed=True),
        ):
            state = valid_state()
            mutate(state)
            with self.assertRaises(RuntimeError):
                ma16.validate_state(state, 3, 'smoke', 4)

    def test_audit_work_cannot_enter_timing_or_profile_evidence(self):
        for mode in ('timing', 'profile'):
            state = valid_state(mode=mode)
            next(iter(state['modules'].values()))['forward_audits'] = 1
            with self.assertRaises(RuntimeError):
                ma16.validate_state(state, 3, mode)


class InstallationTests(unittest.TestCase):
    def fixture(self):
        model, modules, packages, original, layers = ma8_tests.InstallationTests.fixture(self)
        model.config.context_parallel_size = 1
        layers.get_tensor_model_parallel_rank = lambda: 4
        with patch.dict(sys.modules, packages):
            ma16.cache.install_activation_cache(model)
        return model, modules, packages, original, layers

    def test_all_columns_installed_atomically_with_same_backwards_and_global_head(self):
        model, modules, packages, original, layers = self.fixture()
        cached = [module.forward for _, module in modules]
        with patch.dict(sys.modules, packages), \
                patch.object(torch.cuda, 'current_device', return_value=4), \
                patch.object(torch.cuda, 'current_stream', return_value=types.SimpleNamespace(cuda_stream=19)):
            tracker = ma16.install_local_first(model, 'timing')
            with self.assertRaisesRegex(RuntimeError, 'duplicate'):
                ma16.install_local_first(model, 'timing')
        self.assertEqual(len(tracker.projections), 64)
        for _, module in modules[:-1]:
            wrapper = module.forward.__func__.__globals__['linear_with_grad_accumulation_and_async_allreduce']
            function = wrapper.__globals__['LinearWithGradAccumulationAndAsyncCommunication']
            self.assertEqual(function.__name__, '_MA16LocalFirstLinear')
            self.assertIs(module.forward.__func__.__code__, original.__code__)
        self.assertIs(modules[-1][1].forward.__func__, original)
        self.assertIs(layers.ColumnParallelLinear.forward, original)
        self.assertEqual(layers.LinearWithGradAccumulationAndAsyncCommunication.__name__,
                         'LinearWithGradAccumulationAndAsyncCommunication')
        self.assertEqual(tracker.state()['source_sha256'], ma16.CLASS_SHA256)
        self.assertEqual(tracker.state()['backward_sha256'], ma16.BACKWARD_SHA256)
        json.dumps(tracker.state())

    def test_late_bad_module_does_not_partially_replace_cached_columns(self):
        model, modules, packages, _, _ = self.fixture()
        before = [module.forward for _, module in modules]
        modules[-2][1].weight.shape = (3, 4)
        with patch.dict(sys.modules, packages), \
                patch.object(torch.cuda, 'current_device', return_value=4), \
                patch.object(torch.cuda, 'current_stream', return_value=types.SimpleNamespace(cuda_stream=19)):
            with self.assertRaisesRegex(RuntimeError, 'incompatible cached'):
                ma16.install_local_first(model, 'smoke')
        self.assertEqual(before, [module.forward for _, module in modules])
        self.assertFalse(hasattr(model, '_ma16_local_first'))

    def test_failure_poisons_projection_and_reporting(self):
        with patch.object(torch.cuda, 'current_device', return_value=0), \
                patch.object(torch.cuda, 'current_stream', return_value=types.SimpleNamespace(cuda_stream=19)):
            projection = ma16.LocalFirstProjection(torch, 'test', 'qkv', 0, None, None, 'timing')
            with self.assertRaisesRegex(RuntimeError, 'device'):
                projection(torch.zeros(1), torch.zeros(1))
            with self.assertRaisesRegex(RuntimeError, 'failed or reentrant'):
                projection(torch.zeros(1), torch.zeros(1))
            with self.assertRaisesRegex(RuntimeError, 'failed/active'):
                projection.state()


if __name__ == '__main__':
    unittest.main()
