"""CPU data-flow checks; the native GPU oracle remains a required launch gate."""
import unittest
from unittest.mock import patch

import torch
import ma11_adapter as adapter


class CPUOracle:
    def forward(self, value, mask):
        return value.masked_fill(mask, -10000).softmax(-1)

    def backward(self, grad, result, mask):
        return torch.ops.aten._softmax_backward_data(
            grad.contiguous(), result, -1, result.dtype).masked_fill(mask, 0)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def exercise(self, mode):
        evidence = dict(forward_audits=0, backward_audits=0)
        function = adapter.make_function(torch, CPUOracle(), mode, evidence)
        mask = torch.triu(torch.ones(1, 1, 16, 16, dtype=torch.bool), diagonal=1)
        initial = (torch.arange(1024).reshape(1, 4, 16, 16).remainder(37) * .125).bfloat16()
        left, right = [initial.clone().requires_grad_() for _ in range(2)]
        # Match the original in-place masked view, including CopySlices backward.
        raw = left * 1
        view = raw.view_as(raw)
        view.masked_fill_(mask, -10000)
        reference = view.softmax(-1)
        candidate = function.apply(right, mask)
        self.assertTrue(adapter.exact_equal(torch, candidate, reference))
        grad = initial.remainder(3)
        reference.backward(grad)
        candidate.backward(grad)
        self.assertTrue(adapter.exact_equal(torch, left.grad, right.grad))
        self.assertEqual(evidence, dict(forward_audits=int(mode == 'smoke'),
                                       backward_audits=int(mode == 'smoke')))

    def test_saved_output_and_view_gradient_equivalence(self):
        self.exercise('smoke')

    def test_timing_has_no_custom_markers_or_audits(self):
        with patch.object(torch.autograd.profiler, 'record_function',
                          side_effect=AssertionError('marker leaked into timing')):
            self.exercise('timing')

    def test_smoke_rejects_corrupt_forward(self):
        class Broken(CPUOracle):
            def forward(self, value, mask):
                return super().forward(value, mask) + .125
        function = adapter.make_function(torch, Broken(), 'smoke', dict(forward_audits=0, backward_audits=0))
        with self.assertRaisesRegex(RuntimeError, 'forward differs'):
            function.apply(torch.zeros(1, 1, 4, 4).bfloat16(), torch.zeros(1, 1, 4, 4, dtype=torch.bool))

    def test_byte_oracle_does_not_hide_signed_zero(self):
        self.assertFalse(adapter.exact_equal(torch, torch.tensor([0.]).bfloat16(),
                                            torch.tensor([-0.]).bfloat16()))


if __name__ == '__main__':
    unittest.main()
