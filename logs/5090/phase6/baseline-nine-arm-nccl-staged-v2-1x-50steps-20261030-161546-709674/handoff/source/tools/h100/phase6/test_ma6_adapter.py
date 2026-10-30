"""CPU numerical and scheduling contracts for the three MA6 consumer arms.

The CPU collective below models TP8 concatenation. It does not claim CUDA
native CUDA behavior; that remains an eight-GPU correctness/profile gate.
"""
import ast
import json
from types import SimpleNamespace
import unittest

import ma4_f_adapter
import ma5_fullstep
import ma6_adapter

try:
    import torch
except ImportError:
    torch = None


class SourceContractTests(unittest.TestCase):
    def test_reference_math_is_preserved_and_frozen_source_compiles(self):
        source = ma6_adapter.consumer_autograd_source()
        reference = ma4_f_adapter.consumer_autograd_source().replace(
            'class ConsumerGemmRSFunction(', 'class _MA6ReferenceConsumerGemmRSFunction(', 1)
        self.assertTrue(source.startswith(reference))
        ast.parse(source)
        original = ma4_f_adapter.adapter_source(ma5_fullstep.adapter_source())
        transformed = ma6_adapter.adapter_source(original)
        ast.parse(transformed)
        self.assertNotIn('dist_all_gather_func', transformed)
        self.assertNotIn("os.environ.get('H100_MA4_F_ARM') == 'F4'", transformed)
        self.assertIn("backward=new_backward_state(self._ma6_arm)", transformed)
        self.assertIn("consumer['backward'])", transformed)
        with self.assertRaisesRegex(ValueError, 'source contract changed'):
            ma6_adapter.adapter_source(transformed)

    def test_backward_state_is_independent_and_serializes_only_counters(self):
        left = ma6_adapter.new_backward_state('F4')
        right = ma6_adapter.new_backward_state('F4')
        left['forward_calls'] += 1
        left._ma6_op_id = 123
        self.assertEqual(right['forward_calls'], 0)
        self.assertIsNone(right._ma6_op_id)
        self.assertEqual(json.loads(json.dumps(left)), dict(left))
        self.assertEqual(set(left), {'protocol', 'arm', 'gather_schedule', 'forward_calls', 'backward_calls',
            'baseline_backward_calls', 'async_gather_calls', 'wait_calls', 'failed', 'active'})
        self.assertEqual(left['gather_schedule'], 'baseline')
        with self.assertRaises(ValueError):
            ma6_adapter.new_backward_state('CR2')

    def test_v3_arms_and_original_backward_are_baked_into_generated_source(self):
        self.assertEqual(ma6_adapter.ARMS, ('F4', 'F4-mask', 'F4-layout'))
        source = ma6_adapter.consumer_autograd_source()
        self.assertIn("BACKWARD_SCHEDULE = 'baseline'", source)
        self.assertNotIn('async_op', source)
        for arm in ('F4-tile', 'F4-gather', 'unknown'):
            with self.assertRaisesRegex(ValueError, 'Unknown MA6 arm'):
                ma6_adapter.new_backward_state(arm)


@unittest.skipIf(torch is None, 'Run with the actual PyTorch CPU environment')
class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.scope = dict(torch=torch,
            parallel_state=SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8),
            _gather_along_first_dim=lambda input_: torch.cat([input_] * 8, dim=0))
        exec(ma6_adapter.consumer_autograd_source(), self.scope)
        self.Function = self.scope['ConsumerGemmRSFunction']

    @staticmethod
    def projection(x, weight):
        # All eight simulated sequence shards contribute to the local output.
        # Its exact adjoint is the TP8 gathered gradient used by the adapter.
        full = x.reshape(-1, x.shape[-1]).matmul(weight.t())
        return full.reshape(8, full.shape[0] // 8, full.shape[1]).sum(dim=0)

    def test_all_four_gradients_and_three_parameter_updates_match_reference(self):
        for dtype in (torch.bfloat16, torch.float32):
            for arm in ma6_adapter.ARMS:
                with self.subTest(dtype=dtype, arm=arm):
                    torch.manual_seed(487)
                    initial = [torch.randn(shape, dtype=dtype) for shape in
                               ((16, 1, 8), (12, 8), (12,), (2, 1, 12))]
                    candidate = [value.clone().requires_grad_() for value in initial]
                    reference = [value.clone().requires_grad_() for value in initial]
                    state = ma6_adapter.new_backward_state(arm)
                    projection = self.projection

                    class Op:
                        def forward_consumer(self, x, weight, bias, residual, reduce_scatter_option=None):
                            return residual + (projection(x, weight) + bias)

                    op = Op()
                    kept = []
                    for step in range(3):
                        output = self.Function.apply(*candidate, op, None, state)
                        expected = reference[3] + (projection(reference[0], reference[1]).reshape_as(reference[3]) + reference[2])
                        self.assertTrue(torch.equal(output, expected))
                        self.assertIsNone(output._base)
                        self.assertTrue(all(output.data_ptr() != previous.data_ptr() for previous, _ in kept))
                        grad = torch.randn_like(output)
                        output.backward(grad)
                        expected.backward(grad)
                        for index, (left, right) in enumerate(zip(candidate, reference)):
                            self.assertTrue(torch.equal(left.grad, right.grad), (dtype, arm, step, index))
                        kept.append((output, output.detach().clone()))
                        with torch.no_grad():
                            for left, right in zip(candidate, reference):
                                left.add_(left.grad, alpha=-0.03125)
                                right.add_(right.grad, alpha=-0.03125)
                                self.assertTrue(torch.equal(left, right))
                                left.grad = right.grad = None
                    self.assertTrue(all(torch.equal(value, snapshot) for value, snapshot in kept))
                    self.assertEqual(state['forward_calls'], 3)
                    self.assertEqual(state['backward_calls'], 3)
                    self.assertEqual(state['async_gather_calls'], 0)
                    self.assertEqual(state['wait_calls'], state['async_gather_calls'])
                    self.assertEqual(state['baseline_backward_calls'], 3)
                    self.assertFalse(state['failed'])
                    self.assertFalse(state['active'])

    def test_noncontiguous_gradient_keeps_original_bias_semantics(self):
        torch.manual_seed(16)
        values = [torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
                  for shape in ((16, 1, 8), (12, 8), (12,), (2, 1, 12))]
        projection = self.projection
        op = SimpleNamespace(forward_consumer=lambda x, w, b, r, **kwargs: r + (projection(x, w) + b))
        state = ma6_adapter.new_backward_state('F4-mask')
        output = self.Function.apply(*values, op, None, state)
        grad = torch.randn((2, 1, 24), dtype=torch.bfloat16)[..., ::2]
        self.assertFalse(grad.is_contiguous())
        output.backward(grad)
        self.assertTrue(torch.equal(values[2].grad, grad.sum_to_size(values[2].shape)))
        self.assertTrue(torch.equal(values[3].grad, grad))
        full = torch.cat([grad.contiguous()] * 8, dim=0)
        self.assertTrue(torch.equal(values[0].grad, full.matmul(values[1])))
        expected_weight = full.reshape(-1, 12).t().matmul(values[0].reshape(-1, 8))
        self.assertTrue(torch.equal(values[1].grad, expected_weight))

    def test_same_state_cannot_cross_native_instances_or_change_arm(self):
        validate = self.scope['_ma6_validate_state']
        state = ma6_adapter.new_backward_state('F4-mask')
        first, second = object(), object()
        validate(state, first)
        with self.assertRaisesRegex(RuntimeError, 'another projection'):
            validate(state, second)
        self.assertTrue(state['failed'])
        with self.assertRaisesRegex(RuntimeError, 'failed backward'):
            validate(state, first)
        changed = ma6_adapter.new_backward_state('F4')
        changed['arm'] = 'F4-layout'
        with self.assertRaisesRegex(RuntimeError, 'Invalid MA6'):
            validate(changed, first)


class SchedulingTests(unittest.TestCase):
    def make(self, arm, fail=None):
        log = []

        class Tensor:
            is_cuda = False
            device = 'cpu'
            dtype = 'BF16'
            shape = (2, 1, 12)

            def __init__(self, name):
                self.name = name

            def size(self):
                return self.shape

            def stride(self):
                return (12, 12, 1)

            def contiguous(self):
                log.append('contiguous')
                return self

            def sum_to_size(self, shape):
                log.append('bias')
                if fail == 'bias':
                    raise RuntimeError('synthetic bias failure')
                return Tensor('bias')

            def reshape(self, *shape):
                return self

            def t(self):
                return self

            def matmul(self, other):
                stage = 'dgrad' if other.name == 'weight' else 'wgrad'
                log.append(stage)
                if fail == stage:
                    raise RuntimeError('synthetic ' + stage + ' failure')
                return Tensor('result')

        def synchronous(input_):
            log.append('sync_gather')
            if fail == 'sync_gather':
                raise RuntimeError('synthetic gather failure')
            return Tensor('full')

        fake_torch = SimpleNamespace(autograd=SimpleNamespace(Function=object),
            distributed=SimpleNamespace(get_world_size=lambda group: 8),
            empty=lambda *args, **kwargs: Tensor('full'))
        scope = dict(torch=fake_torch,
            parallel_state=SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8,
                                           get_tensor_model_parallel_group=lambda: object()),
            _gather_along_first_dim=synchronous)
        exec(ma6_adapter.consumer_autograd_source(), scope)
        state = scope['new_backward_state'](arm)
        ctx = SimpleNamespace(ma6_state=state, ma6_backward_done=False,
                              saved_tensors=(Tensor('input'), Tensor('weight')), bias_shape=(12,))
        return scope['ConsumerGemmRSFunction'], ctx, Tensor('g'), state, log

    def test_all_arms_keep_exact_original_backward_order(self):
        for arm in ma6_adapter.ARMS:
            function, ctx, grad, state, log = self.make(arm)
            result = function.backward(ctx, grad)
            self.assertIs(result[3], grad)
            self.assertEqual(len(result), 7)
            self.assertEqual(log, ['bias', 'contiguous', 'sync_gather', 'dgrad', 'wgrad'])
            self.assertEqual(state['backward_calls'], 1)
            self.assertEqual(state['baseline_backward_calls'], 1)
            self.assertEqual(state['async_gather_calls'], 0)
            self.assertEqual(state['wait_calls'], 0)

    def test_schedule_or_async_counter_mutation_is_rejected_before_execution(self):
        for key, value in (('gather_schedule', 'overlap'), ('async_gather_calls', 1), ('wait_calls', 1)):
            function, ctx, grad, state, log = self.make('F4-mask')
            state[key] = value
            with self.assertRaisesRegex(RuntimeError, 'Invalid MA6'):
                function.backward(ctx, grad)
            self.assertEqual(log, [])

    def test_failures_never_retry_a_collective(self):
        for arm in ma6_adapter.ARMS:
            for stage in ('bias', 'sync_gather', 'dgrad', 'wgrad'):
                with self.subTest(arm=arm, stage=stage):
                    function, ctx, grad, state, log = self.make(arm, fail=stage)
                    with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                        function.backward(ctx, grad)
                    self.assertTrue(state['failed'])
                    self.assertFalse(state['active'])
                    self.assertEqual(state['backward_calls'], 0)
                    previous = list(log)
                    with self.assertRaisesRegex(RuntimeError, 'failed backward'):
                        function.backward(ctx, grad)
                    self.assertEqual(log, previous)

    def test_repeated_backward_is_rejected_before_relaunch(self):
        function, ctx, grad, state, log = self.make('F4-layout')
        function.backward(ctx, grad)
        with self.assertRaisesRegex(RuntimeError, 'repeated backward'):
            function.backward(ctx, grad)
        self.assertEqual(log.count('sync_gather'), 1)
        self.assertTrue(state['failed'])

    def test_reentrant_backward_is_rejected(self):
        function, ctx, grad, state, log = self.make('F4-mask')
        state['active'] = True
        with self.assertRaisesRegex(RuntimeError, 'reenter'):
            function.backward(ctx, grad)
        self.assertEqual(log, [])
        self.assertTrue(state['failed'])


if __name__ == '__main__':
    unittest.main()
