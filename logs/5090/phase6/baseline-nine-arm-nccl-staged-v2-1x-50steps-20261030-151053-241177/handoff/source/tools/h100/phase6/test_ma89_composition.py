"""CPU composition of unchanged MA8 caching and MA9 zero-count installation.

Tensor-parallel collectives and the CUDA counter ABI are simulated by the
existing fixtures. This checks cross-mechanism data flow and exact arithmetic;
it is not a GPU/NCCL test or an Adam performance/convergence experiment.
"""
import types
import unittest
from unittest.mock import patch

import torch

import ma9_adapter
from test_ma8_adapter import apply as cached_apply, fixture as cache_fixture
from test_ma9_adapter import CPUTorch, CPUReferenceABI, Gradient, inspectable_modules


class CompositionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bf16_cache_and_zero_count_across_three_updates_and_four_microbatches(self):
        self._exercise(torch.bfloat16)

    def test_fp32_cache_and_zero_count_across_three_updates_and_four_microbatches(self):
        self._exercise(torch.float32)

    def _exercise(self, dtype):
        torch.manual_seed(8901)
        backend = CPUTorch()
        clip, optimizer_module = inspectable_modules(self, backend)
        # Actual pinned source checks remain active in the candidate installer.
        original_count = optimizer_module.ChainedOptimizer.count_zeros
        original_step = optimizer_module.ChainedOptimizer.step
        initial = [torch.randn((6, 8), dtype=dtype), torch.randn((6,), dtype=dtype)]
        arms = []
        for _ in range(2):
            io, functions = cache_fixture()
            parameters = [value.clone().requires_grad_() for value in initial]
            master_weights = [value.float().clone() for value in initial]
            master_grads = [backend.remember(torch.zeros_like(value, dtype=torch.float32))
                            for value in parameters]
            # These are the stable FP32 optimizer gradients that the real
            # zero-statistic helper reads, rather than the BF16 model .grad.
            optimizer_parameters = [types.SimpleNamespace(grad=Gradient(value)) for value in master_grads]
            optimizer = optimizer_module.ChainedOptimizer()
            optimizer.config = types.SimpleNamespace(use_precision_aware_optimizer=False)
            optimizer.chained_optimizers = [types.SimpleNamespace(
                get_parameters=lambda values=optimizer_parameters: values,
                config=types.SimpleNamespace(clip_grad=0, log_num_zeros_in_grad=False))]
            optimizer.grads_states_parallel_group_is_shared = lambda: True
            optimizer.get_grad_stats_parallel_group = lambda: 'tp8'
            optimizer.prepare_grads = lambda: False
            optimizer.get_grad_norm = lambda values=master_grads: torch.cat(
                [value.reshape(-1) for value in values]).norm().item()

            def update(parameters=parameters, weights=master_weights, gradients=master_grads):
                # A shared deterministic update isolates the composition. It
                # deliberately does not stand in for a native Adam GPU test.
                with torch.no_grad():
                    for parameter, weight, grad in zip(parameters, weights, gradients):
                        weight.add_(grad, alpha=-0.001)
                        parameter.copy_(weight.to(parameter.dtype))
                return True

            optimizer.step_with_ready_grads = update
            arms.append(types.SimpleNamespace(io=io, function=functions['F4-cache'],
                parameters=parameters, master_weights=master_weights, master_grads=master_grads,
                optimizer_parameters=optimizer_parameters, optimizer=optimizer,
                gradient_pointers=[value.data_ptr() for value in master_grads]))

        control, candidate = arms
        library = CPUReferenceABI(backend)
        tracker = ma9_adapter.install_zero_count(candidate.optimizer, library, 'timing')
        self.assertNotIn('count_zeros', control.optimizer.__dict__)
        self.assertIs(control.optimizer.count_zeros.__func__, original_count)
        self.assertIs(optimizer_module.ChainedOptimizer.step, original_step)
        self.assertIs(optimizer_module.ChainedOptimizer.count_zeros, original_count)
        self.assertIs(tracker.helpers.count_zeros_fp32, clip.count_zeros_fp32)

        observations = []
        # No new column instrumentation or timing-mode statistics markers may
        # enter this composed path. Automatic ATen/autograd tracing is unaffected.
        with patch.object(torch.autograd.profiler, 'record_function',
                          side_effect=AssertionError('Unexpected custom per-call profiler marker')) as marker:
            for step, active_channels in enumerate((2, 4, 6)):
                for arm in arms:
                    for grad in arm.master_grads:
                        grad.zero_()
                for microbatch in range(4):
                    value = (torch.arange(1, 17, dtype=torch.float32).reshape(2, 1, 8)
                             + step + microbatch * 0.25).to(dtype)
                    inputs = [value.clone().requires_grad_() for _ in arms]
                    outputs = [cached_apply(arm.function, [input_, *arm.parameters])
                               for arm, input_ in zip(arms, inputs)]
                    self.assertTrue(torch.equal(*outputs))
                    self.assertTrue(all(type(output.grad_fn).__name__ == '_MA8CachedRawLinearBackward'
                                        for output in outputs))
                    upstream = torch.zeros_like(outputs[0])
                    upstream[..., :active_channels] = (microbatch + 1) * 0.125
                    for output in outputs:
                        output.backward(upstream)
                    self.assertTrue(torch.equal(inputs[0].grad, inputs[1].grad))
                    for left, right in zip(control.parameters, candidate.parameters):
                        self.assertEqual(left.grad.dtype, dtype)
                        self.assertTrue(torch.equal(left.grad, right.grad))
                    with torch.no_grad():
                        for arm in arms:
                            for parameter, master_grad in zip(arm.parameters, arm.master_grads):
                                master_grad.add_(parameter.grad.float())
                                parameter.grad = None

                for left, right in zip(control.master_grads, candidate.master_grads):
                    self.assertTrue(torch.equal(left, right))
                before = [[grad.view(torch.uint8).clone() for grad in arm.master_grads] for arm in arms]
                # Calls the unchanged original ChainedOptimizer.step in both
                # arms; only candidate.count_zeros has the existing MA9 binding.
                outcomes = [arm.optimizer.step() for arm in arms]
                self.assertEqual(outcomes[0], outcomes[1])
                expected_per_tensor = [(6 - active_channels) * 8, 6 - active_channels]
                expected_reduced = float(8 * sum(expected_per_tensor))
                self.assertEqual(outcomes[0][2], expected_reduced)
                self.assertEqual(tracker.native.counts.tolist(), expected_per_tensor)
                observations.append(outcomes[0][2])
                for arm, saved in zip(arms, before):
                    self.assertEqual([grad.data_ptr() for grad in arm.master_grads], arm.gradient_pointers)
                    for grad, old_bytes in zip(arm.master_grads, saved):
                        self.assertTrue(torch.equal(grad.view(torch.uint8), old_bytes))
                for left, right in zip(control.parameters, candidate.parameters):
                    self.assertTrue(torch.equal(left, right))
                for left, right in zip(control.master_weights, candidate.master_weights):
                    self.assertTrue(torch.equal(left, right))
                self.assertEqual(tracker.native.initializations, 1)
                self.assertEqual(tracker.native.calls, step + 1)
            marker.assert_not_called()

        # Each count differs, so a stale cached scalar cannot pass this test.
        self.assertEqual(observations, [288.0, 144.0, 0.0])
        self.assertEqual(len(set(observations)), 3)
        self.assertTrue(any(not torch.equal(value, original)
                            for value, original in zip(candidate.parameters, initial)))
        self.assertEqual(library.launches, 3)
        state = tracker.state()
        self.assertEqual((state['calls'], state['initializations'], state['audits']), (3, 1, 0))
        self.assertEqual((state['kernel_calls'], state['selected_tensors']), (6, 2))
        self.assertEqual(tracker.native.descriptors[:, 0].tolist(), candidate.gradient_pointers)
        self.assertEqual(backend.reductions, [('sum', 'tp8')] * 6)
        for arm in arms:
            self.assertEqual(sum(name == 'gather' for name, _, _ in arm.io.calls), 12)
            self.assertEqual(sum(name == 'scatter' for name, _, _ in arm.io.calls), 12)
            self.assertEqual(arm.io.waits.count('gather'), 0)
            self.assertEqual(arm.io.waits.count('scatter'), 12)


if __name__ == '__main__':
    unittest.main()
