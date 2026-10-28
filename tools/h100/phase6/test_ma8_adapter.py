"""MA8-v2 CPU checks: raw pinned math, zero instrumentation, read-only F4."""
import ast
from functools import partial
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import List, Optional
import unittest
from unittest.mock import patch
import warnings

import torch
import ma8_adapter as ma8

REPO = Path(__file__).resolve().parents[3]
LAYERS = REPO.parent / 'Megatron-LM/megatron/core/tensor_parallel/layers.py'
OLD_COLUMN_AST_SHA256 = 'fd242d309b08e9711486439da4b3881513056deb16643f2d98d241da3191a8b9'


def source_parts(path=LAYERS):
    text = path.read_text()
    tree = ast.parse(text)
    raw = next(n for n in tree.body if isinstance(n, ast.ClassDef) and
               n.name == 'LinearWithGradAccumulationAndAsyncCommunication')
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
                   n.name == 'linear_with_grad_accumulation_and_async_allreduce')
    column = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ColumnParallelLinear')
    forward = next(n for n in column.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    lines = text.splitlines(keepends=True)
    return (ma8.normalized(ast.get_source_segment(text, raw)),
            ''.join(lines[wrapper.lineno - 1:wrapper.end_lineno]),
            ''.join(lines[forward.lineno - 1:forward.end_lineno]))


def pinned_source(path=LAYERS):
    return source_parts(path)[0]


class Collectives:
    def __init__(self):
        self.buffer = None
        self.calls = []
        self.waits = []

    def get_tensor(self, shape, dtype, name):
        if self.buffer is None or list(self.buffer.shape) != list(shape) or self.buffer.dtype != dtype:
            self.buffer = torch.empty(shape, dtype=dtype)
        return self.buffer

    def gather(self, output, input, group=None, async_op=False):
        self.calls.append(('gather', async_op, output.data_ptr()))
        # Real collectives only move values. Distinct simulated rank slices
        # catch accidentally using only the local activation in wgrad.
        output.copy_(torch.cat([input + rank for rank in range(8)], dim=0))
        return types.SimpleNamespace(wait=lambda: self.waits.append('gather'))

    def scatter(self, output, input, group=None, async_op=False):
        self.calls.append(('scatter', async_op, output.data_ptr()))
        output.copy_(input[:output.shape[0]] * 8)
        return types.SimpleNamespace(wait=lambda: self.waits.append('scatter'))


class TorchProxy:
    # The raw pinned CPU test must preserve its CUDA destination expression;
    # only device resolution and TP collectives are substituted.
    cuda = types.SimpleNamespace(current_device=lambda: 'cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


def fixture(include_scope=False, real_amp=False):
    io = Collectives()
    scope = dict(torch=TorchProxy(), custom_fwd=lambda f: f, custom_bwd=lambda f: f,
        get_tensor_model_parallel_world_size=lambda: 8,
        get_tensor_model_parallel_group=lambda: None,
        get_global_memory_buffer=lambda: io,
        dist_all_gather_func=io.gather, dist_reduce_scatter_func=io.scatter,
        prepare_input_tensors_for_wgrad_compute=lambda grad, inp:
            (grad.contiguous().view(-1, grad.shape[-1]), inp.view(-1, inp.shape[-1])),
        os=os, warnings=warnings, Optional=Optional, List=List)
    if real_amp:
        scope.update(custom_fwd=partial(torch.amp.custom_fwd, device_type='cuda'),
                     custom_bwd=partial(torch.amp.custom_bwd, device_type='cuda'))
    source, wrapper, _ = source_parts()
    exec(compile(source + '\n' + wrapper + '\nlinear_with_grad_accumulation_and_async_allreduce.warned = False\n',
                 '<ma8-reference-test>', 'exec'), scope)
    exec(compile(ma8.cached_class_source(source), '<ma8-candidate-test>', 'exec'), scope)
    functions = {'F4': scope['LinearWithGradAccumulationAndAsyncCommunication'],
                 'F4-cache': scope['_MA8CachedRawLinear']}
    return (io, functions, scope) if include_scope else (io, functions)


def inspectable_layers(testcase, changed_column=False, changed_wrapper=False):
    """Use real files and real inspect.getsource, preserving method indentation."""
    raw, wrapper, column = source_parts()
    if changed_column:
        column = ma8.replace_once(column, 'return output, output_bias', 'return output, None')
    if changed_wrapper:
        wrapper = ma8.replace_once(wrapper, 'return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)',
                                  'return LinearWithGradAccumulationAndAsyncCommunication.apply(*args[:])')
    source = (raw + '\n' + wrapper + '\nlinear_with_grad_accumulation_and_async_allreduce.warned = False\n'
              + '\nclass ColumnParallelLinear:\n' + column)
    temp = tempfile.TemporaryDirectory(prefix='ma8-inspect-source-')
    testcase.addCleanup(temp.cleanup)
    file = Path(temp.name) / 'layers.py'
    file.write_text(source)
    _, _, namespace = fixture(include_scope=True)
    module = types.ModuleType('ma8_source_fixture_' + file.parent.name.replace('-', '_'))
    module.__file__ = str(file)
    module.__dict__.update(namespace)
    registered = patch.dict(sys.modules, {module.__name__: module})
    registered.start()
    testcase.addCleanup(registered.stop)
    exec(compile(source, str(file), 'exec'), module.__dict__)
    return module


def apply(function, values):
    return function.apply(*values, False, False, True, None, None)


class SourceTests(unittest.TestCase):
    def test_source_identity_bounded_change_and_no_added_instrumentation(self):
        source = pinned_source()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), ma8.CLASS_SHA256)
        candidate = ma8.cached_class_source(source)
        ast.parse(candidate)
        self.assertEqual(candidate.count('dist_all_gather_func('), 1)
        self.assertEqual(candidate.count('dist_reduce_scatter_func('), 1)
        self.assertNotIn('get_global_memory_buffer()', candidate)
        for math in ('grad_input = grad_output.matmul(weight)',
                     'grad_weight = grad_output.t().matmul(total_input)',
                     'grad_bias = grad_output.sum(dim=0) if use_bias else None',
                     'prepare_input_tensors_for_wgrad_compute(\n                grad_output, total_input'):
            self.assertIn(math, candidate)
        for forbidden in ('record_function', 'CacheState', 'ActivationCacheTracker', 'MA8ReferenceLinear',
                          'MA8RetainedActivationLinear', 'data_ptr', 'ma8_state', 'forward_calls', 'backward_calls'):
            self.assertNotIn(forbidden, Path(ma8.__file__).read_text())
        with self.assertRaises(RuntimeError):
            ma8.cached_class_source(source.replace('grad_input =', 'grad_input  ='))

    def test_real_runtime_source_hashes(self):
        layers = inspectable_layers(self)
        source = inspect.getsource(layers.ColumnParallelLinear.forward)
        self.assertTrue(source.startswith('    def forward('))
        ma8.require_source(source, ma8.COLUMN_FORWARD_SHA256)
        ma8.require_source(inspect.getsource(layers.linear_with_grad_accumulation_and_async_allreduce),
                           ma8.LINEAR_WRAPPER_SHA256)

    def test_historical_hash_failure_reproduced_without_missing_log_dependency(self):
        layers = inspectable_layers(self)
        runtime_source = inspect.getsource(layers.ColumnParallelLinear.forward)
        with self.assertRaisesRegex(RuntimeError, 'pinned, unmodified Megatron source'):
            ma8.require_source(runtime_source, OLD_COLUMN_AST_SHA256)
        ma8.require_source(runtime_source, ma8.COLUMN_FORWARD_SHA256)
        text = LAYERS.read_text()
        cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'ColumnParallelLinear')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        ma8.require_source(ast.get_source_segment(text, method), OLD_COLUMN_AST_SHA256)
        self.assertNotEqual(OLD_COLUMN_AST_SHA256, ma8.COLUMN_FORWARD_SHA256)

    def test_original_wrapper_code_and_warning_state_are_preserved_and_isolated(self):
        _, functions, namespace = fixture(include_scope=True)
        original = namespace['linear_with_grad_accumulation_and_async_allreduce']
        before = dict(original.__dict__)
        clone = ma8._clone_function(original,
            {'LinearWithGradAccumulationAndAsyncCommunication': functions['F4-cache']})
        self.assertIs(clone.__code__, original.__code__)
        self.assertEqual(clone.__defaults__, original.__defaults__)
        self.assertIs(clone.__globals__[original.__name__], clone)
        values = [torch.randn(2, 1, 8, requires_grad=True), torch.randn(6, 8, requires_grad=True), None]
        with patch.dict(os.environ, {'CUDA_DEVICE_MAX_CONNECTIONS': '2'}), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            output = clone(*values, False, False, True, None, None)
        self.assertEqual(len(caught), 1)
        self.assertTrue(clone.warned)
        self.assertEqual(original.__dict__, before)
        output.sum().backward()


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8123)
        torch.set_num_threads(1)

    def test_bf16_fp32_gradients_and_three_updates(self):
        for dtype in (torch.bfloat16, torch.float32):
            for use_bias in (False, True):
                with self.subTest(dtype=dtype, bias=use_bias):
                    fixtures = [fixture(), fixture()]
                    w0, b0 = torch.randn(6, 8, dtype=dtype), torch.randn(6, dtype=dtype)
                    weights = [w0.clone().requires_grad_() for _ in ma8.ARMS]
                    biases = [b0.clone().requires_grad_() if use_bias else None for _ in ma8.ARMS]
                    for step in range(3):
                        for microbatch in range(4):
                            x0 = torch.randn(2, 1, 8, dtype=dtype)
                            inputs = [x0.clone().requires_grad_() for _ in ma8.ARMS]
                            outputs = [apply(functions[arm], [x, w, b])
                                for (io, functions), arm, x, w, b in
                                zip(fixtures, ma8.ARMS, inputs, weights, biases)]
                            self.assertTrue(torch.equal(*outputs))
                            upstream = torch.randn(16, 1, 12, dtype=dtype)[..., ::2]
                            self.assertFalse(upstream.is_contiguous())
                            for output in outputs: output.backward(upstream)
                            self.assertTrue(torch.equal(inputs[0].grad, inputs[1].grad))
                            self.assertTrue(torch.equal(weights[0].grad, weights[1].grad))
                            if use_bias: self.assertTrue(torch.equal(biases[0].grad, biases[1].grad))
                        with torch.no_grad():
                            for w, b in zip(weights, biases):
                                w.add_(w.grad, alpha=-0.01)
                                w.grad = None
                                if b is not None:
                                    b.add_(b.grad, alpha=-0.01)
                                    b.grad = None
                        self.assertTrue(torch.equal(*weights))
                    for arm, (io, _) in zip(ma8.ARMS, fixtures):
                        self.assertEqual(sum(name == 'gather' for name, _, _ in io.calls), 24 if arm == 'F4' else 12)
                        self.assertEqual(sum(name == 'scatter' for name, _, _ in io.calls), 12)
                        self.assertEqual(io.waits.count('gather'), 12 if arm == 'F4' else 0)
                        self.assertEqual(io.waits.count('scatter'), 12)

    def test_multiple_live_owned_inputs_survive_global_buffer_overwrite(self):
        io, functions = fixture()
        inputs = [torch.randn(2, 1, 8, dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
        weights = [torch.randn(6, 8, dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
        outputs = [apply(functions['F4-cache'], [inp, weight, None]) for inp, weight in zip(inputs, weights)]
        saved = [output.grad_fn.saved_tensors[2] for output in outputs]
        global_buffer = io.get_tensor(saved[0].shape, saved[0].dtype, 'mpu')
        global_buffer.fill_(float('nan'))
        self.assertEqual(len({x.data_ptr() for x in saved} | {global_buffer.data_ptr()}), 3)
        expected = [x.clone() for x in saved]
        for index in (1, 0):
            upstream = torch.randn_like(outputs[index])
            expected_wgrad = upstream.view(-1, 6).t().matmul(expected[index].view(-1, 8))
            outputs[index].backward(upstream)
            self.assertTrue(torch.equal(weights[index].grad, expected_wgrad))
            self.assertTrue(torch.equal(saved[index], expected[index]))
        self.assertEqual(len([c for c in io.calls if c[0] == 'gather']), 2)

    def test_native_saved_tensor_version_checks_remain(self):
        for changed in ('input', 'weight', 'cache'):
            _, functions = fixture()
            inp, weight = torch.randn(2, 1, 8, requires_grad=True), torch.randn(6, 8, requires_grad=True)
            output = apply(functions['F4-cache'], [inp, weight, None])
            with torch.no_grad():
                {'input': inp, 'weight': weight, 'cache': output.grad_fn.saved_tensors[2]}[changed].add_(1)
            with self.assertRaisesRegex(RuntimeError, 'modified by an inplace'):
                output.sum().backward()

    def test_native_retain_graph_semantics_have_no_added_guard(self):
        io, functions = fixture()
        values = [torch.randn(2, 1, 8, requires_grad=True), torch.randn(6, 8, requires_grad=True), None]
        output = apply(functions['F4-cache'], values)
        output.sum().backward(retain_graph=True)
        output.sum().backward()
        self.assertEqual(len([c for c in io.calls if c[0] == 'gather']), 1)
        self.assertEqual(len([c for c in io.calls if c[0] == 'scatter']), 2)

    def test_automatic_profiler_names_and_sequence_without_custom_markers(self):
        for arm, class_name in zip(ma8.ARMS,
                ('LinearWithGradAccumulationAndAsyncCommunication', '_MA8CachedRawLinear')):
            _, functions = fixture(real_amp=True)
            values = [torch.randn(2, 1, 8, requires_grad=True), torch.randn(6, 8, requires_grad=True), None]
            with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                output = apply(functions[arm], values)
                output.sum().backward()
            with tempfile.TemporaryDirectory(prefix='ma8-v2-cpu-profile-') as tmp:
                path = str(Path(tmp) / 'trace.json')
                prof.export_chrome_trace(path)
                events = json.loads(Path(path).read_text())['traceEvents']
            forward = [e for e in events if e.get('name') == class_name]
            backward = [e for e in events if e.get('name') == class_name + 'Backward']
            self.assertEqual(len(forward), 1)
            self.assertEqual(len(backward), 1)
            self.assertEqual(forward[0]['args']['Sequence number'], backward[0]['args']['Sequence number'])
            self.assertFalse(any(e.get('name', '').startswith('ma8.column.') for e in events))
            self.assertFalse(any(e.get('name') in ('MA8ReferenceLinear', 'MA8RetainedActivationLinear') for e in events))

    def test_real_amp_decorators_preserve_context_and_gradients(self):
        _, functions = fixture(real_amp=True)
        originals = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((2, 1, 8), (6, 8), (6,))]
        arms = [[value.clone().requires_grad_() for value in originals] for _ in ma8.ARMS]
        outputs = [apply(functions[arm], values) for arm, values in zip(ma8.ARMS, arms)]
        self.assertTrue(all(hasattr(output.grad_fn, '_fwd_used_autocast') and
                            hasattr(output.grad_fn, '_dtype') for output in outputs))
        self.assertTrue(torch.equal(*outputs))
        upstream = torch.randn_like(outputs[0])
        for output in outputs: output.backward(upstream)
        for baseline, candidate in zip(*arms): self.assertTrue(torch.equal(baseline.grad, candidate.grad))


class InstallationTests(unittest.TestCase):
    def fixture(self, changed_column=False, changed_wrapper=False):
        layers = inspectable_layers(self, changed_column, changed_wrapper)
        Column = layers.ColumnParallelLinear
        original_forward = Column.forward

        class FakeWeight:
            requires_grad, dtype, is_cuda = True, torch.bfloat16, True
            def __init__(self, role): self.shape = (1536 if role == 'qkv' else 2048, 4096)
            def is_contiguous(self): return True

        Column.training = Column.sequence_parallel = True
        Column.allreduce_dgrad = Column.gradient_accumulation_fusion = Column.explicit_expert_comm = False
        Column.disable_grad_reduce = Column.gather_output = False
        config = types.SimpleNamespace(num_layers=32, hidden_size=4096,
            tensor_model_parallel_size=8, pipeline_model_parallel_size=1,
            sequence_parallel=True, gradient_accumulation_fusion=False,
            recompute_granularity=None, enable_cuda_graph=False, external_cuda_graph=False,
            cpu_offloading=False, defer_embedding_wgrad_compute=False, _cpu_offloading_context=None)
        modules = [(f'decoder.layers.{i}{suffix}', Column())
            for i in range(32) for suffix, role in ma8.SUFFIX_ROLES.items()]
        modules.append(('output_layer', Column()))
        for name, module in modules:
            module.weight = FakeWeight('qkv' if name.endswith('linear_qkv') else 'fc1')
            module.config, module.bias, module.skip_bias_add = config, None, False
            module._forward_impl = layers.linear_with_grad_accumulation_and_async_allreduce
        model = types.SimpleNamespace(training=True, config=config, named_modules=lambda: iter(modules))
        packages = {name: types.ModuleType(name) for name in
            ('megatron', 'megatron.core', 'megatron.core.tensor_parallel')}
        packages['megatron.core.tensor_parallel'].layers = layers
        return model, modules, packages, original_forward, layers

    def test_f4_readonly_setup_changes_no_method_or_instance_and_cannot_be_installed(self):
        model, modules, packages, original, layers = self.fixture()
        before_model = dict(vars(model))
        before_modules = [dict(vars(module)) for _, module in modules]
        before_wrapper = dict(vars(layers.linear_with_grad_accumulation_and_async_allreduce))
        with patch.dict(sys.modules, packages):
            metadata = ma8.inspect_activation_setup(model, 'F4')
            with self.assertRaisesRegex(RuntimeError, 'candidate-only'):
                ma8.install_activation_cache(model, 'F4')
        self.assertEqual(metadata['installed'], False)
        self.assertEqual(metadata['function'], 'LinearWithGradAccumulationAndAsyncCommunication')
        self.assertEqual(len(metadata['module_roles']), 64)
        self.assertEqual(vars(model), before_model)
        self.assertEqual([vars(module) for _, module in modules], before_modules)
        self.assertEqual(vars(layers.linear_with_grad_accumulation_and_async_allreduce), before_wrapper)
        self.assertTrue(all(module.forward.__func__ is original for _, module in modules))
        json.dumps(metadata)

    def test_candidate_original_code_objects_and_head_unchanged(self):
        model, modules, packages, original, layers = self.fixture()
        wrapper = layers.linear_with_grad_accumulation_and_async_allreduce
        with patch.dict(sys.modules, packages):
            metadata = ma8.install_activation_cache(model, 'F4-cache')
            with self.assertRaisesRegex(RuntimeError, 'installed twice'):
                ma8.install_activation_cache(model, 'F4-cache')
        self.assertEqual(len(metadata['module_roles']), 64)
        self.assertTrue(metadata['installed'])
        self.assertEqual(metadata['function'], '_MA8CachedRawLinear')
        for _, module in modules[:-1]:
            forward = module.forward.__func__
            cloned_wrapper = forward.__globals__['linear_with_grad_accumulation_and_async_allreduce']
            self.assertIs(forward.__code__, original.__code__)
            self.assertIs(cloned_wrapper.__code__, wrapper.__code__)
            self.assertEqual(cloned_wrapper.__globals__['LinearWithGradAccumulationAndAsyncCommunication'].__name__,
                             '_MA8CachedRawLinear')
        self.assertIs(modules[-1][1].forward.__func__, original)
        self.assertIs(Column_forward := layers.ColumnParallelLinear.forward, original)
        self.assertIs(layers.linear_with_grad_accumulation_and_async_allreduce, wrapper)
        self.assertEqual(layers.LinearWithGradAccumulationAndAsyncCommunication.__name__,
                         'LinearWithGradAccumulationAndAsyncCommunication')
        json.dumps(metadata)

    def test_original_column_forward_runs_through_raw_candidate_without_dispatch_checks(self):
        model, modules, packages, original, layers = self.fixture()
        with patch.dict(sys.modules, packages): ma8.install_activation_cache(model, 'F4-cache')
        target = modules[0][1]
        # CPU/small-shape replacement happens after setup to exercise the actual
        # original method code; an added per-call CUDA/shape checker would fail.
        target.weight = torch.randn(6, 8, requires_grad=True)
        inp = torch.randn(2, 1, 8, requires_grad=True)
        with patch.dict(os.environ, {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}):
            first, _ = target.forward(inp)
            second, _ = target.forward(inp)
        self.assertEqual(type(first.grad_fn).__name__, '_MA8CachedRawLinearBackward')
        self.assertEqual(type(second.grad_fn).__name__, '_MA8CachedRawLinearBackward')
        self.assertIsNot(first.grad_fn.saved_tensors[2], second.grad_fn.saved_tensors[2])
        (first.sum() + second.sum()).backward()
        self.assertIs(modules[-1][1].forward.__func__, original)

    def test_invalid_setup_never_half_patches(self):
        for defect in ('last_shape', 'missing_module', 'column_source', 'wrapper_source', 'recompute'):
            with self.subTest(defect=defect):
                model, modules, packages, original, _ = self.fixture(
                    changed_column=defect == 'column_source', changed_wrapper=defect == 'wrapper_source')
                if defect == 'last_shape': modules[-2][1].weight.shape = (999, 4096)
                if defect == 'missing_module': modules.pop(-2)
                if defect == 'recompute': model.config.recompute_granularity = 'full'
                with patch.dict(sys.modules, packages):
                    with self.assertRaises(RuntimeError): ma8.install_activation_cache(model, 'F4-cache')
                self.assertTrue(all(module.forward.__func__ is original for _, module in modules))
                self.assertFalse(hasattr(model, '_ma8_activation_setup'))


if __name__ == '__main__':
    unittest.main()
