"""Host contract tests; actual CUDA/collective byte tests run in GPU smoke."""
import ast
import hashlib
from pathlib import Path
import types
import unittest

import torch

import mav1_adapter as mav1


def valid_state(steps=3, mode='smoke', rank=2):
    ids = mav1.plan_ids()
    modules = {}
    for name, role in mav1.module_roles().items():
        width = mav1.OUTPUT_WIDTHS.get(role, mav1.ROW_WIDTHS.get(role))
        modules[name] = dict(role=role, plan_id=ids[name], forward_calls=steps * 4,
            backward_calls=steps * 4,
            native=[1, steps * 4, steps * 4, 0, -1 if role in mav1.OUTPUT_WIDTHS else 0, width])
    return dict(protocol=mav1.PROTOCOL, installed=True, failed=False, mode=mode,
        rank=rank, world_size=8, plan_count=128, per_call_audits=0,
        retains_original_process_group=True, native_backward=True,
        source_sha256=mav1.source_identities(), modules=modules,
        totals={key: steps * 256 for key in
                ('column_forward', 'column_backward', 'row_forward', 'row_backward')},
        self_test=dict(passed=True, cases=8, roles=['qkv', 'fc1', 'attn', 'fc2'],
                       strided_gradients=True, private_leaves=True, rng_unchanged=True)
                  if mode == 'smoke' else None)


class StateTests(unittest.TestCase):
    def test_extension_requires_exact_native_abi(self):
        ext = types.SimpleNamespace(Mav1ColumnPlan=object, Mav1RowPlan=object,
                                    mav1_native_protocol=lambda: 1)
        mav1.validate_extension(ext)
        ext.mav1_native_protocol = lambda: 2
        with self.assertRaisesRegex(RuntimeError, 'ABI'):
            mav1.validate_extension(ext)
        del ext.Mav1RowPlan
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            mav1.validate_extension(ext)

    def test_helper_source_pins_match_frozen_megatron_arithmetic(self):
        base = Path(__file__).resolve().parents[4] / 'Megatron-LM/megatron/core'
        for filename, name, expected in (
                ('utils.py', 'prepare_input_tensors_for_wgrad_compute', mav1.WGRAD_PREPARE_SHA256),
                ('tensor_parallel/mappings.py', '_gather_along_first_dim', mav1.ROW_GATHER_SHA256)):
            source = (base / filename).read_text()
            node = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
            original = mav1.cache.normalized(ast.get_source_segment(source, node))
            self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), expected)
            with self.assertRaisesRegex(RuntimeError, 'pinned'):
                mav1.cache.require_source(original + '\n# changed helper\n', expected)

    def test_complete_cover_and_unique_profile_identifiers(self):
        roles, ids = mav1.module_roles(), mav1.plan_ids()
        self.assertEqual(len(roles), 128)
        self.assertEqual(set(ids.values()), set(range(128)))
        for role in ('qkv', 'fc1', 'attn', 'fc2'):
            self.assertEqual(list(roles.values()).count(role), 32)
        for name, role in roles.items():
            self.assertEqual(ids[name] < 64, role in mav1.OUTPUT_WIDTHS)
        for mode in ('smoke', 'timing', 'profile'):
            self.assertTrue(mav1.validate_state(valid_state(mode=mode), 3, mode, 2))

    def test_missing_native_backward_or_stale_counters_rejected(self):
        for mutate in (
            lambda s: s.update(native_backward=False),
            lambda s: s.update(retains_original_process_group=False),
            lambda s: s.update(rank=3),
            lambda s: s['totals'].update(column_backward=0),
            lambda s: s['source_sha256'].update(column_class='0' * 64),
            lambda s: next(iter(s['modules'].values())).update(backward_calls=0),
            lambda s: next(iter(s['modules'].values()))['native'].__setitem__(3, 1),
            lambda s: s['modules'].pop(next(iter(s['modules']))),
        ):
            value = valid_state()
            mutate(value)
            with self.assertRaises(RuntimeError):
                mav1.validate_state(value, 3, 'smoke', 2)

    def test_audit_leaks_and_missing_numeric_tests_rejected(self):
        value = valid_state(mode='timing')
        value['self_test'] = valid_state()['self_test']
        with self.assertRaisesRegex(RuntimeError, 'self-test'):
            mav1.validate_state(value, 3, 'timing', 2)
        value = valid_state()
        value['self_test']['strided_gradients'] = False
        with self.assertRaisesRegex(RuntimeError, 'self-test'):
            mav1.validate_state(value, 3, 'smoke', 2)
        value = valid_state()
        value['per_call_audits'] = 1
        with self.assertRaises(RuntimeError):
            mav1.validate_state(value, 3, 'smoke', 2)

    def test_tracker_reports_actual_native_counters(self):
        tracker = mav1.Tracker(2, 'timing', mav1.source_identities())
        expected = valid_state(mode='timing')
        for name, row in expected['modules'].items():
            values = row['native']
            tracker.plans[name] = dict(role=row['role'], plan_id=row['plan_id'],
                plan=types.SimpleNamespace(state=lambda values=values: values))
        actual = tracker.state()
        self.assertEqual(actual, expected)
        tracker.failed = True
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            tracker.state()


class DispatchTests(unittest.TestCase):
    def test_native_plan_owns_both_slots_packet_mask_and_optional_indices(self):
        retained = []
        slots = [types.SimpleNamespace(peers=[object() for _ in range(8)],
                                      selected=object(), quant_tiles=object()),
                 types.SimpleNamespace(peers=[object() for _ in range(8)],
                                      selected=object(), quant_tiles=None)]
        plan = types.SimpleNamespace(retain_storage=lambda values: retained.extend(values))
        mav1.retain_row_storage(plan, slots)
        self.assertEqual(retained, slots[0].peers + [slots[0].selected, slots[0].quant_tiles]
                         + slots[1].peers + [slots[1].selected])
        slots[1].selected = None
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            mav1.retain_row_storage(plan, slots)

    def test_column_preserves_explicit_parameter_edges_and_bias_ownership(self):
        for skip in (False, True):
            calls = []
            def apply(value, weight, bias):
                calls.append((value, weight, bias))
                result = value.matmul(weight.t())
                return result if bias is None else result + bias
            module = types.SimpleNamespace(training=True,
                weight=torch.ones(3, 4, requires_grad=True),
                bias=torch.ones(3, requires_grad=True))
            tracker = types.SimpleNamespace(failed=False)
            value = torch.ones(2, 4, requires_grad=True)
            fn = mav1.column_forward(tracker, types.SimpleNamespace(apply=apply), skip)
            output, bias = fn(module, value)
            loss = (output + bias if bias is not None else output).sum()
            loss.backward()
            self.assertIs(calls[0][0], value)
            self.assertIs(calls[0][1], module.weight)
            self.assertIs(calls[0][2], None if skip else module.bias)
            self.assertIs(bias, module.bias if skip else None)
            self.assertTrue(torch.equal(module.weight.grad, torch.full((3, 4), 2.)))
            self.assertTrue(torch.equal(module.bias.grad, torch.full((3,), 2.)))
            self.assertFalse(tracker.failed)

    def test_column_override_rejected_without_entering_native(self):
        tracker = types.SimpleNamespace(failed=False)
        plan = types.SimpleNamespace(apply=lambda *args: self.fail('called native'))
        fn = mav1.column_forward(tracker, plan, False)
        with self.assertRaisesRegex(RuntimeError, 'override'):
            fn(types.SimpleNamespace(training=True), None, weight=object())
        self.assertTrue(tracker.failed)

    def test_row_dispatch_never_hides_differentiable_inputs(self):
        calls = []
        tracker = types.SimpleNamespace(failed=False)
        plan = types.SimpleNamespace(apply=lambda *args: calls.append(args) or 'output')
        op, option = object(), object()
        dispatch = mav1.RowDispatch(tracker, plan, op, option)
        inputs = tuple(object() for _ in range(4))
        self.assertEqual(dispatch.apply(*inputs, op, option), 'output')
        self.assertEqual(calls, [inputs])
        with self.assertRaisesRegex(RuntimeError, 'operation/option'):
            dispatch.apply(*inputs, object(), option)
        self.assertTrue(tracker.failed)

    def test_row_closure_clone_preserves_original_and_audit_body(self):
        def factory(adapter, name):
            op = expected_shape = None
            def forward(module, input_):
                nonlocal op, expected_shape
                adapter.append(name)
                if op is None:
                    op, expected_shape = 'legacy', (1,)
                return ConsumerGemmRSFunction.apply(module, input_, op, expected_shape)
            return forward
        original_calls, native_calls = [], []
        original = factory(original_calls, 'layer')
        legacy = types.SimpleNamespace(apply=lambda *args: ('legacy', args))
        original.__globals__['ConsumerGemmRSFunction'] = legacy
        dispatch = types.SimpleNamespace(apply=lambda *args: native_calls.append(args) or 'native')
        op = object()
        clone = mav1.clone_row_forward(original, dispatch, op, (2048, 1, 512))
        self.assertIs(clone.__code__, original.__code__)
        self.assertEqual(clone('module', 'input'), 'native')
        self.assertEqual(native_calls, [('module', 'input', op, (2048, 1, 512))])
        self.assertEqual(original('module', 'input')[0], 'legacy')
        self.assertIs(original.__globals__['ConsumerGemmRSFunction'], legacy)
        self.assertEqual(original_calls, ['layer', 'layer'])
        with self.assertRaisesRegex(RuntimeError, 'before the first'):
            mav1.clone_row_forward(original, dispatch, op, (1,))


class OracleTests(unittest.TestCase):
    def test_private_leaf_oracle_compares_forward_and_all_gradients(self):
        inputs = (torch.arange(8.).reshape(2, 4).requires_grad_(),
                  torch.arange(12.).reshape(3, 4).requires_grad_(),
                  torch.ones(3, requires_grad=True))
        reference = lambda value, weight, bias: value.matmul(weight.t()) + bias
        gradient = torch.arange(12.).reshape(2, 6)[:, ::2]
        self.assertFalse(gradient.is_contiguous())
        mav1.compare_autograd(torch, reference, reference, inputs, gradient)
        self.assertTrue(all(value.grad is None for value in inputs))
        # Same forward value with an intentionally wrong derivative must fail.
        def wrong(value, weight, bias):
            output = reference(value, weight, bias)
            return 2 * output - output.detach()
        with self.assertRaisesRegex(RuntimeError, 'gradient differs'):
            mav1.compare_autograd(torch, reference, wrong, inputs, gradient)

    def test_optional_bias_and_signed_zero_bytes(self):
        inputs = (torch.ones(2, 4), torch.ones(3, 4), None)
        fn = lambda value, weight, bias: value.matmul(weight.t())
        mav1.compare_autograd(torch, fn, fn, inputs, torch.ones(2, 3))
        self.assertFalse(mav1.exact_equal(torch, torch.tensor([0.]), torch.tensor([-0.])))

    def test_deterministic_tensor_does_not_advance_rng(self):
        state = torch.random.get_rng_state().clone()
        first = mav1.deterministic_tensor(torch, (2, 3), 'cpu', 2)
        second = mav1.deterministic_tensor(torch, (2, 3), 'cpu', 2)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
