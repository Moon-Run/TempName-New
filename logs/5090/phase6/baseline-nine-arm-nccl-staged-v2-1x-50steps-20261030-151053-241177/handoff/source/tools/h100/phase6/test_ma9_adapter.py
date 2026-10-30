"""CPU adapter regressions using real Torch arithmetic and inspectable source.

The CUDA ABI is simulated here; these tests are not native-kernel/GPU evidence.
The worker's native_self_test separately exercises the actual CUDA kernels.
"""
import ast
import inspect
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import ma9_adapter as ma9


REPO = Path(__file__).resolve().parents[3]
MEGATRON = REPO.parent / 'Megatron-LM/megatron/core/optimizer'


class CallableABI:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class CPUTorch:
    """Only CUDA allocation/device resolution and distributed I/O are replaced."""
    def __init__(self):
        self.storage = {}
        self.stream = types.SimpleNamespace(cuda_stream=7)
        self.reductions = []
        self.cuda = types.SimpleNamespace(current_device=lambda: 0,
                                         current_stream=lambda device=None: self.stream)
        self.distributed = types.SimpleNamespace(
            ReduceOp=types.SimpleNamespace(SUM='sum'), all_reduce=self.all_reduce)

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, *args):
        return torch.device('cpu')

    def remember(self, tensor):
        self.storage[tensor.data_ptr()] = tensor
        return tensor

    def tensor(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return self.remember(torch.tensor(*args, **kwargs))

    def empty(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return self.remember(torch.empty(*args, **kwargs))

    def all_reduce(self, tensor, op=None, group=None):
        self.reductions.append((op, group))
        tensor.mul_(8)

    def count_nonzero(self, tensor):
        return torch.count_nonzero(tensor.tensor if isinstance(tensor, Gradient) else tensor)


class CPUReferenceABI:
    """Compute the reference statistic, testing ABI wiring but not CUDA math."""
    def __init__(self, backend, protocol=9, chunk_size=ma9.CHUNK_SIZE):
        self.backend = backend
        self.status = 0
        self.launches = 0
        self.ma9_zero_count_protocol = CallableABI(lambda: protocol)
        self.ma9_zero_count_chunk_size = CallableABI(lambda: chunk_size)
        self.ma9_zero_count_cuda = CallableABI(self.launch)

    def launch(self, descriptors_ptr, n, chunks, partials_ptr, counts_ptr, total_ptr, stream):
        self.launches += 1
        if self.status:
            return self.status
        tensors = self.backend.storage
        descriptors = tensors[descriptors_ptr].reshape(-1, 3)
        counts = tensors[counts_ptr] if n else None
        total = tensors[total_ptr]
        assert tuple(descriptors.shape) == (n, 3)
        assert stream == self.backend.stream.cuda_stream
        total.zero_()
        for i, (ptr, size, first_chunk) in enumerate(descriptors.tolist()):
            grad = tensors[ptr]
            assert size == grad.numel()
            count = size - torch.count_nonzero(grad)
            counts[i] = count
            total.copy_(count + total)
        return 0


class Gradient:
    is_cuda = True

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def detach(self):
        return self


def helpers():
    return types.SimpleNamespace(
        param_is_not_shared=lambda p: not getattr(p, 'shared', False),
        param_is_not_tensor_parallel_duplicate=lambda p: not getattr(p, 'duplicate', False),
        get_data_parallel_group_if_dtensor=lambda grad, group: group,
        to_local_if_dtensor=lambda grad: grad)


def parameter(backend, values, **attrs):
    tensor = backend.remember(torch.tensor(values, dtype=torch.float32))
    return types.SimpleNamespace(grad=Gradient(tensor), **attrs)


def inspectable_modules(case, backend, change_step=False):
    """Load whole original decorated methods from actual source lines."""
    clip_text = (MEGATRON / 'clip_grads.py').read_text()
    clip_node = next(n for n in ast.parse(clip_text).body if isinstance(n, ast.FunctionDef)
                     and n.name == 'count_zeros_fp32')
    clip_source = ''.join(clip_text.splitlines(True)[clip_node.lineno - 1:clip_node.end_lineno])
    optimizer_text = (MEGATRON / 'optimizer.py').read_text()
    cls = next(n for n in ast.parse(optimizer_text).body if isinstance(n, ast.ClassDef)
               and n.name == 'ChainedOptimizer')
    methods = []
    for name in ('count_zeros', 'step'):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        begin = min([method.lineno] + [d.lineno for d in method.decorator_list]) - 1
        methods.append(''.join(optimizer_text.splitlines(True)[begin:method.end_lineno]))
    if change_step:
        methods[-1] = methods[-1].replace('found_inf_flag =', 'found_inf_flag  =', 1)
    temp = tempfile.TemporaryDirectory(prefix='ma9-source-')
    case.addCleanup(temp.cleanup)
    root = Path(temp.name)
    clip = types.ModuleType('ma9_fixture_clip')
    clip.__dict__.update(vars(helpers()), torch=backend)
    optimizer_module = types.ModuleType('ma9_fixture_optimizer')
    for module, filename, source in (
        (clip, 'clip.py', 'from __future__ import annotations\n' + clip_source),
        (optimizer_module, 'optimizer.py', 'class ChainedOptimizer:\n' + '\n'.join(methods))):
        path = root / filename
        path.write_text(source)
        module.__file__ = str(path)
        if module is optimizer_module:
            module.__dict__.update(torch=backend, count_zeros_fp32=clip.count_zeros_fp32)
        registered = patch.dict(sys.modules, {module.__name__: module})
        registered.start()
        case.addCleanup(registered.stop)
        exec(compile(source, str(path), 'exec'), module.__dict__)
    package = types.ModuleType('megatron.core.optimizer')
    package.clip_grads, package.optimizer = clip, optimizer_module
    registration = patch.dict(sys.modules, {'torch': backend, 'megatron': types.ModuleType('megatron'),
        'megatron.core': types.ModuleType('megatron.core'), 'megatron.core.optimizer': package})
    registration.start()
    case.addCleanup(registration.stop)
    return clip, optimizer_module


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.backend = CPUTorch()
        self.library = CPUReferenceABI(self.backend)

    def test_ordered_float_rounding_above_two_pow24(self):
        # Three sequential float additions differ from one final integer sum.
        counts = [2**24, 1, 1]
        reference = torch.tensor([0.], dtype=torch.float32)
        for n in counts:
            reference = torch.tensor(n, dtype=torch.int64) + reference
        self.assertEqual(reference.item(), float(2**24))
        self.assertEqual(torch.tensor(sum(counts), dtype=torch.float32).item(), float(2**24 + 2))
        # Exercise actual NativeCounter descriptor/buffer wiring with real data.
        gradients = [self.backend.remember(torch.zeros(n)) for n in counts]
        native = ma9.NativeCounter(self.backend, self.library)
        self.assertTrue(torch.equal(native.local(gradients), reference))
        self.assertEqual(native.counts.tolist(), counts)

    def test_chunk_boundaries_empty_values_and_scratch_reuse(self):
        gradients = [self.backend.remember(torch.empty(0)),
                     self.backend.remember(torch.zeros(ma9.CHUNK_SIZE + 1)),
                     self.backend.remember(torch.tensor([0., -0., 1., float('nan'), float('inf')])),
                     self.backend.remember(torch.tensor([1, -2147483647], dtype=torch.int32).view(torch.float32)),
                     self.backend.remember(torch.empty(0))]
        native = ma9.NativeCounter(self.backend, self.library)
        self.assertEqual(native.local(gradients).item(), ma9.CHUNK_SIZE + 3)
        self.assertEqual(native.descriptors[:, 2].tolist(), [0, 0, 2, 3, 4])
        self.assertEqual(native.counts.tolist(), [0, ma9.CHUNK_SIZE + 1, 2, 0, 0])
        gradients[1].fill_(1)
        self.assertEqual(native.local(gradients).item(), 2)
        self.assertEqual(native.initializations, 1)
        self.assertEqual(native.calls, 2)

    def test_storage_order_and_stream_changes_fail_before_launch(self):
        gradients = [self.backend.remember(torch.zeros(2)), self.backend.remember(torch.ones(3))]
        native = ma9.NativeCounter(self.backend, self.library)
        native.local(gradients)
        for changed in (gradients[::-1], [self.backend.remember(gradients[0].clone()), gradients[1]]):
            with self.assertRaisesRegex(RuntimeError, 'stale pointers'):
                native.local(changed)
        self.backend.stream.cuda_stream = 8
        with self.assertRaisesRegex(RuntimeError, 'stream changed'):
            native.local(gradients)
        self.assertEqual(self.library.launches, 1)

    def test_wrong_native_identity_and_cuda_error_are_not_fallbacks(self):
        for library in (CPUReferenceABI(self.backend, protocol=8),
                        CPUReferenceABI(self.backend, chunk_size=2048)):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                ma9.NativeCounter(self.backend, library)
        native = ma9.NativeCounter(self.backend, self.library)
        self.library.status = 700
        with self.assertRaisesRegex(RuntimeError, 'failed: 700'):
            native.local([self.backend.remember(torch.ones(1))])
        self.assertEqual(native.calls, 0)


class FilterTests(unittest.TestCase):
    def test_original_filters_preserve_order_and_replicas(self):
        backend = CPUTorch()
        params = [parameter(backend, [0., 1.]), parameter(backend, [0.], shared=True),
                  parameter(backend, [0.], duplicate=True), types.SimpleNamespace(grad=None),
                  parameter(backend, [1., 0., 0.])]
        selected = ma9.select_gradients(params, helpers(), backend)
        self.assertEqual([g.data_ptr() for g in selected], [params[0].grad.data_ptr(), params[4].grad.data_ptr()])

    def test_wrong_precision_layout_and_dtensor_are_rejected(self):
        backend = CPUTorch()
        for grad in (torch.zeros(3, dtype=torch.bfloat16), torch.zeros(3, 4).t()):
            with self.assertRaisesRegex(RuntimeError, 'contiguous local CUDA FP32'):
                ma9.select_gradients([types.SimpleNamespace(grad=Gradient(grad))], helpers(), backend)
        with self.assertRaisesRegex(RuntimeError, 'DTensor/FSDP'):
            bad = helpers()
            bad.get_data_parallel_group_if_dtensor = lambda grad, group: object()
            ma9.select_gradients([parameter(backend, [1.])], bad, backend)


class InstallTests(unittest.TestCase):
    def fixture(self, change_step=False):
        backend = CPUTorch()
        clip, module = inspectable_modules(self, backend, change_step)
        optimizer = module.ChainedOptimizer()
        params = [parameter(backend, [0., 1., 0.]), parameter(backend, [0.], duplicate=True)]
        optimizer.config = types.SimpleNamespace(use_precision_aware_optimizer=False)
        optimizer.chained_optimizers = [types.SimpleNamespace(get_parameters=lambda: params,
            config=types.SimpleNamespace(clip_grad=0, log_num_zeros_in_grad=False))]
        optimizer.grads_states_parallel_group_is_shared = lambda: True
        optimizer.get_grad_stats_parallel_group = lambda: 'tp8'
        optimizer.prepare_grads = lambda: False
        optimizer.get_grad_norm = lambda: 2.5
        optimizer.step_with_ready_grads = lambda: True
        return backend, clip, module, optimizer, params

    def test_real_decorated_source_identity_and_instance_only_install(self):
        backend, clip, module, optimizer, params = self.fixture()
        for function, name in ((clip.count_zeros_fp32, 'count_zeros_fp32'),
                               (module.ChainedOptimizer.count_zeros, 'count_zeros'),
                               (module.ChainedOptimizer.step, 'step')):
            ma9.require_source(function, name)
        original_count, original_step = module.ChainedOptimizer.count_zeros, module.ChainedOptimizer.step
        global_reference = inspect.unwrap(original_count).__globals__['count_zeros_fp32']
        tracker = ma9.install_zero_count(optimizer, CPUReferenceABI(backend), 'timing')
        self.assertIs(module.ChainedOptimizer.count_zeros, original_count)
        self.assertIs(module.ChainedOptimizer.step, original_step)
        self.assertIs(inspect.unwrap(original_count).__globals__['count_zeros_fp32'], global_reference)
        self.assertEqual(optimizer.step(), (True, 2.5, 16.0))
        self.assertEqual(backend.reductions, [('sum', 'tp8')])
        self.assertEqual(tracker.state()['selected_tensors'], 1)
        self.assertEqual(tracker.state()['calls'], 1)
        with self.assertRaisesRegex(RuntimeError, 'untouched'):
            ma9.install_zero_count(optimizer, CPUReferenceABI(backend), 'timing')

    def test_source_drift_rejected_without_partial_install(self):
        backend, clip, module, optimizer, params = self.fixture(change_step=True)
        with self.assertRaisesRegex(RuntimeError, 'original step'):
            ma9.install_zero_count(optimizer, CPUReferenceABI(backend), 'timing')
        self.assertNotIn('count_zeros', optimizer.__dict__)

    def test_smoke_compares_original_local_counts_and_reduced_statistic(self):
        backend, clip, module, optimizer, params = self.fixture()
        # Actual CUDA self-test remains in production smoke; only it is replaced
        # for this CPU test of the subsequent original-reference comparison.
        with patch.object(ma9, 'native_self_test', return_value={'cpu_test_only': True}) as probe:
            tracker = ma9.install_zero_count(optimizer, CPUReferenceABI(backend), 'smoke')
        probe.assert_called_once()
        self.assertEqual(optimizer.step(), (True, 2.5, 16.0))
        self.assertEqual(backend.reductions, [('sum', 'tp8'), ('sum', 'tp8')])
        self.assertEqual(tracker.state()['audits'], 1)
        # Deliberate reference mismatch must latch failure, with no timing result.
        clip.count_zeros_fp32 = lambda *args, **kwargs: 99.0
        with self.assertRaisesRegex(RuntimeError, 'reduced statistic differs'):
            optimizer.step()
        self.assertTrue(tracker.failed)

    def test_failure_latches_and_cannot_report_success(self):
        backend, clip, module, optimizer, params = self.fixture()
        library = CPUReferenceABI(backend)
        tracker = ma9.install_zero_count(optimizer, library, 'timing')
        optimizer.step()
        params[0].grad = Gradient(backend.remember(torch.zeros(3)))
        with self.assertRaisesRegex(RuntimeError, 'stale pointers'):
            optimizer.step()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            optimizer.step()
        with self.assertRaisesRegex(RuntimeError, 'after failure'):
            tracker.state()
        self.assertEqual(library.launches, 1)


if __name__ == '__main__':
    unittest.main()
