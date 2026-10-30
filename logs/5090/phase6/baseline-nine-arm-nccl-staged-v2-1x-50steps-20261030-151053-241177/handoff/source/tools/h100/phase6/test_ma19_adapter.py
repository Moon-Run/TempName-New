"""CPU checks of real pinned optimizer-copy semantics and MA19 lifecycle."""
import ast
import copy
import inspect
from pathlib import Path
import sys
import textwrap
import types
import unittest
from unittest.mock import patch

import torch

import ma19_adapter as ma19

OPTIMIZER = Path(__file__).resolve().parents[4] / 'Megatron-LM/megatron/core/optimizer/optimizer.py'


def valid_state(steps=3, mode='smoke'):
    audits = steps if mode == 'smoke' else 0
    child = dict(tensors=389, elements=849207296, blocks=26146, calls=steps,
                 audits=audits, initializations=1, failed=False, storage_verified=True,
                 device=0, stream=19, descriptor_bytes=26146 * 16 + 389 * 24)
    return dict(protocol=ma19.PROTOCOL, installed=True, failed=False, storage_verified=True,
                chunk_size=32768, threads=256, source_sha256=ma19.SOURCE_HASHES,
                tensors=389, elements=849207296, child_count=1, children=[child],
                calls=steps, audits=audits, self_test=(dict(passed=True, patterns=262144,
                repetitions=2, layouts=['aligned_full', 'unaligned_split'],
                bitwise=True, signed_zero_subnormal_nan_inf=True)
                if mode == 'smoke' else None))


def pinned_functions():
    source = OPTIMIZER.read_text()
    nodes = {n.name: n for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.FunctionDef) and n.name in ma19.SOURCE_HASHES}
    namespace = dict(torch=torch, List=list, Optional=__import__('typing').Optional)
    for name, node in nodes.items():
        lines = source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
        code = textwrap.dedent(''.join(lines))
        # inspect sees the actual repository method through linecache.
        import linecache
        filename = '<ma19-test-original-' + name + '>'
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        exec(compile(code, filename, 'exec'), namespace)
    return namespace


class AdapterTests(unittest.TestCase):
    def test_source_pins_match_real_repository_methods_and_reject_drift(self):
        ns = pinned_functions()
        for name in ma19.SOURCE_HASHES:
            ma19.require_source(ns[name], name)
        with self.assertRaisesRegex(RuntimeError, 'pinned'):
            ma19.require_source(self.test_source_pins_match_real_repository_methods_and_reject_drift,
                                 '_copy_main_params_to_model_params')

    def test_real_original_bf16_copy_and_parameter_versions(self):
        ns = pinned_functions()
        sources = [torch.nn.Parameter(torch.randn(9)), torch.nn.Parameter(torch.randn(3, 7))]
        targets = [torch.nn.Parameter(torch.zeros_like(x, dtype=torch.bfloat16)) for x in sources]
        owner = types.SimpleNamespace(float16_groups=[targets], fp32_from_float16_groups=[sources],
                                      _dummy_overflow_buf=None)
        owner._get_model_and_main_params_data_float16 = types.MethodType(
            ns['_get_model_and_main_params_data_float16'], owner)
        before = [target._version for target in targets]
        ns['_copy_main_params_to_model_params'](owner)
        self.assertEqual(before, [target._version for target in targets])
        for source, target in zip(sources, targets):
            self.assertTrue(torch.equal(source.to(torch.bfloat16), target))

    def test_chunk_coverage_has_no_holes_overlap_or_padding_writes(self):
        sizes = [1, 32767, 32768, 32769, 65539]
        pairs = [(torch.empty(n), torch.empty(n, dtype=torch.bfloat16)) for n in sizes]
        rows, chunks = ma19.descriptor_rows(pairs)
        for index, size in enumerate(sizes):
            owned = [offset for tensor, offset in chunks if tensor == index]
            self.assertEqual(owned, list(range(0, size, ma19.CHUNK_SIZE)))
            self.assertEqual(sum(min(ma19.CHUNK_SIZE, size - start) for start in owned), size)
            self.assertEqual(rows[index], [pairs[index][0].data_ptr(), pairs[index][1].data_ptr(), size])
        for pairs in ([], [(torch.empty(0), torch.empty(0))], [(torch.empty(2), torch.empty(3))]):
            with self.assertRaises(RuntimeError): ma19.descriptor_rows(pairs)

    def test_storage_alias_survives_parameter_data_rebind_and_verifier_rejects(self):
        source = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        target = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        plan = ma19.NativePlan.__new__(ma19.NativePlan)
        plan.failed = False
        plan.pairs = ((source, target),)
        plan.identities = ((ma19.tensor_identity(source), ma19.tensor_identity(target)),)
        plan.storage_aliases = ((source.detach(), target.detach()),)
        plan.verify_storage()
        old_pointer = target.data_ptr()
        target.data = torch.ones(8, dtype=torch.bfloat16)
        self.assertEqual(plan.storage_aliases[0][1].data_ptr(), old_pointer)
        self.assertTrue(torch.equal(plan.storage_aliases[0][1], torch.zeros(8, dtype=torch.bfloat16)))
        with self.assertRaisesRegex(RuntimeError, 'storage/layout'):
            plan.verify_storage()
        self.assertTrue(plan.failed)

    def test_native_submission_stream_device_capture_and_failed_retry(self):
        for failure in ('device', 'stream', 'capture', 'launch'):
            plan = ma19.NativePlan.__new__(ma19.NativePlan)
            plan.failed, plan.calls, plan.device, plan.stream, plan.blocks = False, 0, 0, 19, 7
            plan.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
                current_device=lambda: 1 if failure == 'device' else 0,
                current_stream=lambda: types.SimpleNamespace(cuda_stream=20 if failure == 'stream' else 19),
                is_current_stream_capturing=lambda: failure == 'capture'))
            plan.descriptors = plan.chunks = types.SimpleNamespace(data_ptr=lambda: 1234)
            calls = []
            plan.launch = lambda *args: calls.append(args) or (1 if failure == 'launch' else 0)
            with self.assertRaises(RuntimeError): plan.publish()
            self.assertTrue(plan.failed)
            self.assertEqual(len(calls), int(failure == 'launch'))
            with self.assertRaisesRegex(RuntimeError, 'refusing retry'): plan.publish()
            self.assertEqual(plan.calls, 0)

    def test_smoke_calls_actual_original_boundary_and_detects_corruption(self):
        ns = pinned_functions()
        source = torch.nn.Parameter(torch.arange(17, dtype=torch.float32) / 11)
        target = torch.nn.Parameter(torch.zeros(17, dtype=torch.bfloat16))
        owner = types.SimpleNamespace(float16_groups=[[target]], fp32_from_float16_groups=[[source]],
                                      _dummy_overflow_buf=None)
        owner._get_model_and_main_params_data_float16 = types.MethodType(
            ns['_get_model_and_main_params_data_float16'], owner)
        original = types.MethodType(ns['_copy_main_params_to_model_params'], owner)
        native = types.SimpleNamespace(pairs=((source, target),))
        native.publish = lambda: target.data.copy_(source.data)
        publisher = ma19.ParameterPublisher(torch, owner, original, native, 'smoke')
        publisher.publish()
        self.assertEqual(publisher.audits, 1)
        native.publish = lambda: target.data.zero_()
        with self.assertRaisesRegex(RuntimeError, 'differ'): publisher.publish()
        self.assertTrue(publisher.failed)
        with self.assertRaisesRegex(RuntimeError, 'refusing retry'): publisher.publish()

    def test_installation_is_instance_local_transactional_and_checks_parameter_order(self):
        ns = pinned_functions()
        cls = type('Float16OptimizerWithFloat16Params', (), {
            name: ns[name] for name in ('_get_model_and_main_params_data_float16',
                                       '_copy_main_params_to_model_params')})
        chain = type('ChainedOptimizer', (), {})
        module = types.SimpleNamespace(Float16OptimizerWithFloat16Params=cls, ChainedOptimizer=chain,
            _multi_tensor_copy_this_to_that=ns['_multi_tensor_copy_this_to_that'])
        package = types.ModuleType('megatron.core.optimizer'); package.optimizer = module
        source = torch.nn.Parameter(torch.arange(7, dtype=torch.float32))
        target = torch.nn.Parameter(torch.empty(7, dtype=torch.bfloat16))
        owner = cls()
        owner.float16_groups, owner.fp32_from_float16_groups = [[target]], [[source]]
        owner.fp32_from_fp32_groups = [[]]
        owner.is_stub_optimizer = False
        owner.config = types.SimpleNamespace(bf16=True)
        owner._dummy_overflow_buf = owner.grad_scaler = None
        class CPUPlan:
            def __init__(self, torch_arg, library, pairs):
                self.pairs, self.elements, self.blocks, self.calls = pairs, 7, 1, 0
            def publish(self):
                self.calls += 1
                for s, t in self.pairs: t.data.copy_(s.data)
            def state(self):
                return dict(tensors=1, elements=7, blocks=1, calls=self.calls, initializations=1,
                            failed=False, storage_verified=True, descriptor_bytes=40)
        with patch.dict(sys.modules, {'megatron.core.optimizer': package}), \
                patch.object(ma19, 'NativePlan', CPUPlan), \
                patch.object(ma19, 'PARAMETER_COUNT', 1), \
                patch.object(ma19, 'ELEMENT_COUNT', 7), \
                patch.object(ma19, 'BLOCK_COUNT', 1):
            with patch.object(ma19, 'native_self_test', side_effect=RuntimeError('oracle failed')):
                with self.assertRaisesRegex(RuntimeError, 'oracle failed'):
                    ma19.install_parameter_publish(owner, object(), 'smoke')
            self.assertNotIn('_copy_main_params_to_model_params', owner.__dict__)
            tracker = ma19.install_parameter_publish(owner, object(), 'timing')
            self.assertIs(cls._copy_main_params_to_model_params, ns['_copy_main_params_to_model_params'])
            owner._copy_main_params_to_model_params()
            self.assertTrue(torch.equal(source.to(torch.bfloat16), target))
            self.assertEqual(tracker.state()['calls'], 1)
            with self.assertRaisesRegex(RuntimeError, 'duplicate'):
                ma19.install_parameter_publish(owner, object(), 'timing')
            owner.float16_groups[0][0] = torch.nn.Parameter(target.detach().clone())
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                tracker.state()

    def test_native_submission_does_not_construct_or_walk_tensor_metadata(self):
        source = inspect.getsource(ma19.NativePlan.publish)
        tree = ast.parse(textwrap.dedent(source))
        self.assertFalse(any(isinstance(n, (ast.For, ast.ListComp, ast.GeneratorExp)) for n in ast.walk(tree)))
        self.assertNotIn('tensor_identity', source)
        self.assertNotIn('self.pairs', source)

    def test_evidence_rejects_missing_or_inconsistent_performance_guards(self):
        for mode in ('smoke', 'timing', 'profile'):
            ma19.validate_state(valid_state(3, mode), 3, mode)
        changes = [('calls', 2), ('audits', 0), ('elements', 849207295), ('child_count', 2),
                   ('tensors', 388), ('storage_verified', False), ('chunk_size', 16384),
                   ('failed', True), ('source_sha256', {})]
        for key, value in changes:
            state = valid_state(); state[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                ma19.validate_state(state, 3, 'smoke')
        state = valid_state(); state['children'][0]['initializations'] = 2
        with self.assertRaises(RuntimeError): ma19.validate_state(state, 3, 'smoke')
        state = valid_state(); state['self_test']['bitwise'] = False
        with self.assertRaises(RuntimeError): ma19.validate_state(state, 3, 'smoke')


if __name__ == '__main__': unittest.main()
