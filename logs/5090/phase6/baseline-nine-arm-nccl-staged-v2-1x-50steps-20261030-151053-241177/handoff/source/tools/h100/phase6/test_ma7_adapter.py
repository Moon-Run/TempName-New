"""Exact arithmetic and bounded lifetime checks; CPU is explicitly test-only."""
import ast
import contextlib
import json
from types import SimpleNamespace
import unittest

import ma4_f_adapter
import ma7_adapter

try:
    import torch
except ImportError:
    torch = None


class SourceTests(unittest.TestCase):
    def test_original_forward_backward_and_generated_source(self):
        source = ma7_adapter.consumer_autograd_source()
        original = ma4_f_adapter.consumer_autograd_source().replace(
            'class ConsumerGemmRSFunction(', 'class _MA7ReferenceConsumerGemmRSFunction(', 1)
        self.assertTrue(source.startswith(original))
        ast.parse(source)
        import ma5_fullstep
        original_adapter = ma4_f_adapter.adapter_source(ma5_fullstep.adapter_source())
        transformed = ma7_adapter.adapter_source(original_adapter)
        ast.parse(transformed)
        self.assertIn('def install_wgrad_schedule(', transformed)
        self.assertIn("consumer['backward'])", transformed)
        with self.assertRaises(ValueError):
            ma7_adapter.adapter_source(transformed)

    def test_counter_serialization_and_identity(self):
        state = ma7_adapter.new_backward_state('F4-wgrad')
        state._weight = object()
        self.assertEqual(json.loads(json.dumps(state)), dict(state))
        self.assertFalse(state._allow_cpu)
        self.assertEqual(ma7_adapter.BACKWARD_SCHEDULE, 'independent_weight_stream')
        with self.assertRaises(ValueError):
            ma7_adapter.new_backward_state('F4-mask')

    def test_cuda_stream_edges_and_lifetime_without_a_gpu(self):
        log = []
        class Stream:
            def __init__(self, name): self.name = name
            def wait_stream(self, other): log.append('wait:' + self.name + '<-' + other.name)
        main, aux = Stream('main'), Stream('aux')
        class Cuda:
            current = main
            def current_stream(self, device): return self.current
            def Stream(self, device):
                log.append('create_aux')
                return aux
            @contextlib.contextmanager
            def stream(self, stream):
                before, self.current = self.current, stream
                try: yield
                finally: self.current = before
        cuda = Cuda()
        class Tensor:
            is_cuda = True
            is_leaf = True
            device = 'cuda:0'
            grad = None
            def __init__(self, name, shape, dtype='bf16', requires_grad=False):
                self.name, self.shape, self.dtype, self.requires_grad = name, shape, dtype, requires_grad
            def is_contiguous(self): return True
            def data_ptr(self): return id(self)
            def record_stream(self, stream): log.append('record:' + self.name + ':' + stream.name)
            def reshape(self, *shape): return self
            def t(self): return self
            def matmul(self, other):
                log.append('wgrad:' + cuda.current.name)
                return Tensor('grad_weight', (12, 8))
            def add_(self, other):
                log.append('main_grad_add:' + cuda.current.name)
                return self
        fake_torch = SimpleNamespace(cuda=cuda, bfloat16='bf16', float32='fp32',
            autograd=SimpleNamespace(Function=object,
                profiler=SimpleNamespace(record_function=lambda name: contextlib.nullcontext())))
        scope = dict(torch=fake_torch)
        exec(ma7_adapter.consumer_autograd_source(), scope)
        weight = Tensor('weight', (12, 8), requires_grad=True)
        weight.main_grad = Tensor('main_grad', (12, 8), dtype='fp32')
        baseline = scope['DeferredWgradQueue']()
        baseline.register(ma7_adapter.new_backward_state('F4'), weight)
        self.assertIsNone(baseline.stream)
        self.assertEqual(log, [])
        state = ma7_adapter.new_backward_state('F4-wgrad')
        queue = scope['DeferredWgradQueue']()
        queue.register(state, weight)
        queue.submit(state, Tensor('input', (16, 1, 8)), Tensor('full_grad', (16, 1, 12)), weight)
        self.assertEqual([x for x in log if x.startswith('wait:')], ['wait:aux<-main'])
        self.assertLess(log.index('wait:aux<-main'), log.index('wgrad:aux'))
        self.assertLess(log.index('wgrad:aux'), log.index('main_grad_add:aux'))
        self.assertEqual(len(queue.pending), 1)
        for name in ('input', 'full_grad', 'weight', 'main_grad', 'grad_weight'):
            self.assertIn('record:' + name + ':aux', log)
        queue.drain()
        self.assertEqual([x for x in log if x.startswith('wait:')], ['wait:aux<-main', 'wait:main<-aux'])
        self.assertEqual(queue.pending, [])
        self.assertEqual(state['cuda_wgrad_calls'], 1)
        self.assertEqual(state['pending_wgrad'], 0)


@unittest.skipIf(torch is None, 'Run in the PyTorch environment')
class NumericalAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.scope = dict(torch=torch,
            parallel_state=SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8,
                get_data_parallel_world_size=lambda: 1),
            _gather_along_first_dim=lambda x: torch.cat([x] * 8, dim=0))
        exec(ma7_adapter.consumer_autograd_source(), self.scope)
        self.apply = self.scope['consumer_apply']
        self.reference = self.scope['_MA7ReferenceConsumerGemmRSFunction']
        self.drain = self.scope['drain_backward_state']
        torch.manual_seed(1307)
        self.op = SimpleNamespace(forward_consumer=lambda x, w, b, r, **kwargs:
            r + (x.matmul(w.t()).reshape(8, 2, -1).sum(0) + b))

    def values(self, dtype=None):
        if dtype is None: dtype = torch.bfloat16
        return [torch.randn(shape, dtype=dtype).requires_grad_()
                for shape in ((16, 1, 8), (12, 8), (12,), (2, 1, 12))]

    def state(self, arm='F4-wgrad'):
        return ma7_adapter.new_backward_state(arm, allow_cpu=True)

    def test_four_gradients_and_three_exact_updates(self):
        for dtype in (torch.bfloat16, torch.float32):
            for arm in ma7_adapter.ARMS:
                with self.subTest(dtype=dtype, arm=arm):
                    values = self.values(dtype)
                    refs = [x.detach().clone().requires_grad_() for x in values]
                    state = self.state(arm)
                    retained = []
                    for step in range(3):
                        actual = self.apply(*values, self.op, None, state)
                        expected = self.reference.apply(*refs, self.op, None)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertIsNone(actual._base)
                        self.assertTrue(all(actual.data_ptr() != old.data_ptr() for old, _ in retained))
                        upstream = torch.randn((2, 1, 24), dtype=dtype)[..., ::2]
                        actual.backward(upstream)
                        if arm == 'F4-wgrad':
                            self.assertIsNone(values[1].grad)
                            self.assertEqual(state['pending_wgrad'], 1)
                        self.drain(state)
                        expected.backward(upstream)
                        for index, (left, right) in enumerate(zip(values, refs)):
                            self.assertTrue(torch.equal(left.grad, right.grad), (dtype, arm, step, index))
                        retained.append((actual, actual.detach().clone()))
                        with torch.no_grad():
                            for left, right in zip(values, refs):
                                left.add_(left.grad, alpha=-0.03125)
                                right.add_(right.grad, alpha=-0.03125)
                                self.assertTrue(torch.equal(left, right))
                                left.grad = right.grad = None
                    self.assertTrue(all(torch.equal(x, saved) for x, saved in retained))
                    self.assertEqual(state['drain_calls'], 3)
                    self.assertEqual(state['pending_wgrad'], 0)
                    self.assertEqual(state['baseline_backward_calls'], 3 if arm == 'F4' else 0)
                    self.assertEqual(state['standalone_grad_assignments'], 3 if arm != 'F4' else 0)
                    self.assertEqual(state['cpu_wgrad_calls'], 3 if arm != 'F4' else 0)

    def test_four_microbatches_match_native_bf16_to_fp32_main_grad(self):
        values = self.values()
        refs = [x.detach().clone().requires_grad_() for x in values]
        values[1].main_grad = torch.zeros_like(values[1], dtype=torch.float32)
        refs[1].main_grad = torch.zeros_like(refs[1], dtype=torch.float32)
        def native_ddp_hook(weight):
            weight.main_grad.add_(weight.grad)
            weight.grad = None
        handle = refs[1].register_post_accumulate_grad_hook(native_ddp_hook)
        state = self.state()
        for microbatch in range(4):
            output = self.apply(*values, self.op, None, state)
            expected = self.reference.apply(*refs, self.op, None)
            grad = torch.randn_like(output)
            output.backward(grad)
            self.drain(state)
            expected.backward(grad)
            self.assertTrue(torch.equal(values[1].main_grad, refs[1].main_grad), microbatch)
            self.assertIsNone(values[1].grad)
            for index in (0, 2, 3):
                self.assertTrue(torch.equal(values[index].grad, refs[index].grad))
        handle.remove()
        self.assertEqual(state['main_grad_accumulations'], 4)
        self.assertEqual(state['standalone_grad_assignments'], 0)
        self.assertEqual(state['drained_wgrad_calls'], 4)

    def test_standalone_gradient_accumulation_is_not_overwritten(self):
        values = self.values()
        refs = [x.detach().clone().requires_grad_() for x in values]
        state = self.state()
        for _ in range(4):
            actual = self.apply(*values, self.op, None, state)
            expected = self.reference.apply(*refs, self.op, None)
            grad = torch.randn_like(actual)
            actual.backward(grad)
            self.drain(state)
            expected.backward(grad)
        self.assertTrue(torch.equal(values[1].grad, refs[1].grad))

    def test_missing_drain_repeated_backward_and_double_drain_fail(self):
        for mode in ('missing', 'repeated', 'double'):
            values, state = self.values(), self.state()
            actual = self.apply(*values, self.op, None, state)
            actual.backward(torch.ones_like(actual), retain_graph=True)
            if mode != 'missing':
                self.drain(state)
            with self.assertRaises(RuntimeError):
                if mode == 'missing':
                    self.apply(*values, self.op, None, state)
                elif mode == 'repeated':
                    actual.backward(torch.ones_like(actual), retain_graph=True)
                else:
                    self.drain(state)
            self.assertTrue(state['failed'])
            self.assertEqual(state['backward_calls'], 1)

    def test_main_grad_and_parameter_storage_replacement_poison(self):
        for changed in ('main_grad_object', 'main_grad_storage', 'weight_storage', 'weight_object'):
            values, state = self.values(), self.state()
            values[1].main_grad = torch.zeros_like(values[1], dtype=torch.float32)
            self.apply(*values, self.op, None, state)
            with torch.no_grad():
                if changed == 'main_grad_object':
                    values[1].main_grad = values[1].main_grad.clone()
                elif changed == 'main_grad_storage':
                    values[1].main_grad.data = values[1].main_grad.clone()
                elif changed == 'weight_storage':
                    values[1].data = values[1].clone()
                else:
                    values[1] = values[1].detach().clone().requires_grad_()
            with self.assertRaisesRegex(RuntimeError, 'ownership/storage changed'):
                self.apply(*values, self.op, None, state)
            self.assertTrue(state['failed'])

    def test_queue_capacity_poison_is_shared(self):
        Queue = self.scope['DeferredWgradQueue']
        queue = Queue(max_pending=1)
        left, right = self.state(), self.state()
        first, second = self.values(), self.values()
        queue.register(left, first[1])
        queue.register(right, second[1])
        with torch.no_grad():
            queue.submit(left, first[0], torch.randn((16, 1, 12), dtype=torch.bfloat16), first[1])
            with self.assertRaisesRegex(RuntimeError, 'bounded capacity'):
                queue.submit(right, second[0], torch.randn((16, 1, 12), dtype=torch.bfloat16), second[1])
        self.assertTrue(left['failed'])
        self.assertTrue(right['failed'])
        with self.assertRaisesRegex(RuntimeError, 'poisoned'):
            queue.drain()

    def test_cpu_requires_explicit_test_opt_in(self):
        state = ma7_adapter.new_backward_state('F4-wgrad')
        with self.assertRaisesRegex(RuntimeError, 'CPU is test-only'):
            self.apply(*self.values(), self.op, None, state)

    def test_failed_collective_never_retries(self):
        calls = []
        def failed_gather(x):
            calls.append(1)
            raise RuntimeError('injected collective failure')
        self.scope['_gather_along_first_dim'] = failed_gather
        for arm in ma7_adapter.ARMS:
            values, state = self.values(), self.state(arm)
            output = self.apply(*values, self.op, None, state)
            with self.assertRaisesRegex(RuntimeError, 'injected collective'):
                output.backward(torch.ones_like(output), retain_graph=True)
            count = len(calls)
            with self.assertRaisesRegex(RuntimeError, 'failed backward'):
                output.backward(torch.ones_like(output), retain_graph=True)
            self.assertEqual(len(calls), count)
            self.assertFalse(state['active'])

    def test_model_schedule_drains_once_and_preserves_finalization(self):
        for arm in ma7_adapter.ARMS:
            exec('class TestAdapter:\n' + ma7_adapter.ADAPTER_METHODS, self.scope)
            adapter = self.scope['TestAdapter']()
            adapter._consumer_failed = False
            adapter._consumer_contexts = {}
            adapter.modules = {}
            values_by_name = {}
            for index in range(64):
                name = str(index)
                values = self.values()
                values[1].main_grad = torch.zeros_like(values[1], dtype=torch.float32)
                adapter.modules[name] = SimpleNamespace(weight=values[1])
                adapter._consumer_contexts[name] = dict(backward=self.state(arm))
                values_by_name[name] = values
            finalizations = []
            config = SimpleNamespace(gradient_accumulation_fusion=False, sequence_parallel=True,
                pipeline_model_parallel_size=1, context_parallel_size=1,
                finalize_model_grads_func=lambda *args: finalizations.append(args))
            model = [SimpleNamespace(ddp_config=SimpleNamespace(
                overlap_grad_reduce=False, overlap_param_gather=False))]
            outputs = []
            def original_backward():
                for output in outputs:
                    output.backward(torch.ones_like(output))
                return 'original-result'
            schedules = SimpleNamespace(backward_step=original_backward)
            adapter.install_wgrad_schedule(schedules, config, model)
            for name, values in values_by_name.items():
                outputs.append(self.apply(*values, self.op, None,
                    adapter._consumer_contexts[name]['backward']))
            self.assertEqual(schedules.backward_step(), 'original-result')
            self.assertEqual(adapter._ma7_queue.pending, [])
            evidence = adapter.ma7_backward_state()
            self.assertTrue(all(s['drain_calls'] == s['backward_calls'] == 1 for s in evidence.values()))
            config.finalize_model_grads_func('model')
            self.assertEqual(finalizations, [('model',)])
            with self.assertRaisesRegex(RuntimeError, 'twice'):
                adapter.install_wgrad_schedule(schedules, config, model)
            next(iter(adapter._consumer_contexts.values()))['backward']['pending_wgrad'] = 1
            with self.assertRaisesRegex(RuntimeError, 'unfinished backward'):
                config.finalize_model_grads_func('model')
            self.assertEqual(len(finalizations), 1)


if __name__ == '__main__':
    unittest.main()
