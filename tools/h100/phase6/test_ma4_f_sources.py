"""MA4-F lifecycle and CPU autograd contracts; GPU numerical gates run on TP8."""
import ast
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import ma2_cr2_sources
import ma2_ablation_sources
import ma4_f_sources as native
import ma4_f_adapter as adapter
import ma5_fullstep
from common import REPO, HERE

try:
    import torch
except ImportError:
    torch = None

class FakeNative:
    def __init__(self, pointer):
        self.pointer = pointer
        self.calls = 0
        self.consumer_calls = 0
        self.seen = []
        self.fail = False

    def forward(self, x, w, reduce_scatter_option=None):
        self.calls += 1
        return object()

    def ma4_consumer_forward(self, x, w, b, r, option):
        self.seen.append((x, w, b, r, option))
        if self.fail:
            raise RuntimeError('synthetic native submission failure')
        self.calls += 1
        self.consumer_calls += 1
        return object()

    def ma4_consumer_state(self):
        return [1, int(self.calls>0), self.calls, self.consumer_calls, 512, self.pointer, 0]


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reconstruct the transform inputs from tracked maintained source. No
        # ignored developer build or pre-existing generated tree is required.
        cls.temporary = tempfile.TemporaryDirectory(prefix='ma4-f-source-fixture-')
        cls.reference = Path(cls.temporary.name)
        transforms = {native.FRONTEND: ma2_cr2_sources.frontend, native.WRAPPER: ma2_cr2_sources.wrapper,
                      native.WRAPPER_HEADER: ma2_cr2_sources.wrapper_header, native.PYBIND: ma2_cr2_sources.pybind,
                      native.HEADER: ma2_cr2_sources.header}
        base = {name: fn((REPO/name).read_text()) for name, fn in transforms.items()}
        base[native.RUNTIME] = ma2_cr2_sources.runtime(ma2_ablation_sources.sources(REPO)[native.RUNTIME])
        base[native.CR2_DECODER] = (HERE/'ma2_c_decode.cuh').read_text()
        for name, source in base.items():
            path = cls.reference/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        cls.changed = native.sources(cls.reference)
        tree = ast.parse(cls.changed[native.FRONTEND])
        classes = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[])
        scope = {}
        exec(compile(classes, '<MA4-F frontend>', 'exec'), scope)
        cls.Single, cls.Double = scope['GemmRSTaco'], scope['GemmRSTacoDoubleBuffered']

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def make(self):
        double = self.Double.__new__(self.Double)
        double._lock, double._failed, double.next_slot = threading.Lock(), False, 0
        double.slots = []
        for ptr in (1000, 2000):
            single = self.Single.__new__(self.Single)
            single._lock, single._failed = threading.Lock(), False
            single.op, single.option = FakeNative(ptr), object()
            double.slots.append(single)
        return double

    def test_dynamic_inputs_and_retained_outputs_alternate_full_slots(self):
        op = self.make()
        inputs = [tuple(object() for _ in range(4)) for _ in range(8)]
        outputs = [op.forward_consumer(*values) for values in inputs]
        self.assertEqual(len({id(x) for x in outputs}), 8)
        self.assertEqual(op.next_slot, 0)
        for index, slot in enumerate(op.slots):
            self.assertEqual(slot.op.seen, [(*values, slot.option) for values in inputs[index::2]])
        states = op.ma4_consumer_state()
        self.assertEqual([s['bias_residual_calls'] for s in states], [4, 4])
        self.assertEqual([s['local_packet_ptr'] for s in states], [1000, 2000])

    def test_plain_reference_and_consumer_calls_cannot_be_confused(self):
        op = self.make()
        op.forward(object(), object())
        op.forward_consumer(*(object() for _ in range(4)))
        states = op.ma4_consumer_state()
        self.assertEqual(sum(s['successful_calls'] for s in states), 2)
        self.assertEqual(sum(s['bias_residual_calls'] for s in states), 1)

    def test_native_failure_poison_preserves_slot_and_rejects_retry(self):
        op = self.make()
        op.forward_consumer(*(object() for _ in range(4)))
        op.slots[1].op.fail = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic native'):
            op.forward_consumer(*(object() for _ in range(4)))
        self.assertEqual(op.next_slot, 1)
        op.slots[1].op.fail = False
        with self.assertRaisesRegex(RuntimeError, 'failed double'):
            op.forward_consumer(*(object() for _ in range(4)))
        self.assertTrue(op.ma4_consumer_state()[1]['failed'])

    def test_state_abi_length_is_checked(self):
        op = self.make()
        op.slots[0].op.ma4_consumer_state = lambda: [1]
        with self.assertRaisesRegex(RuntimeError, 'ABI mismatch'):
            op.ma4_consumer_state()

    def test_only_consumer_store_changes_gpu_arithmetic(self):
        original = (self.reference/native.CR2_DECODER).read_text()
        changed = self.changed[native.DECODER]
        core = original[original.index('  const int lane'):original.index('  taco_ma2::store')]
        core = core.replace('*reinterpret_cast<uint2*>(output + offset) = packed;',
                            'taco_ma4f_store(output, bias, residual, offset, packed);')
        self.assertIn(core, changed)
        self.assertIn('sum[0] = __hadd2(sum[0], add_bias[0]);', changed)
        self.assertIn('sum[0] = __hadd2(add_residual[0], sum[0]);', changed)
        self.assertNotIn(native.CR2_DECODER, self.changed)
        self.assertNotIn('src/gemm_rs/taco_codec.cuh', self.changed)
        self.assertNotIn('cudaGraphLaunch', self.changed[native.WRAPPER])


class PhaseBindingTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(adapter.adapter_source(ma5_fullstep.adapter_source()))
        self.grad_enabled, self.inference_mode = True, False
        fake_torch = SimpleNamespace(is_grad_enabled=lambda: self.grad_enabled,
                                     is_inference_mode_enabled=lambda: self.inference_mode)
        scope = {'torch': fake_torch}
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='Adapter')
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<MA4-F adapter>', 'exec'), scope)
        self.op = scope['Adapter'].__new__(scope['Adapter'])
        self.op._consumer_failed = False
        self.state = dict(residual=None, residual_version=None, output=None, projection_calls=0, bda_calls=0)
        self.residual = SimpleNamespace(_version=0)
        self.layer = SimpleNamespace(training=True)
        self.output = object()

    def phase(self, alter=None):
        def original(residual):
            self.state['projection_calls'] += 1
            self.state['output'] = self.output
            if alter:
                alter()
            return self.op._consumer_bda(self.state)(True, False)((self.output, None), residual, 0.)
        return self.op._consumer_phase(original, self.state, 0, 'hidden_states')

    def test_valid_consumption_clears_references_without_changing_tensor(self):
        call = self.phase()
        for _ in range(3):
            self.assertIs(call(self.layer, self.residual), self.output)
            self.assertIsNone(self.state['residual'])
            self.assertIsNone(self.state['output'])
        self.assertEqual((self.state['projection_calls'], self.state['bda_calls']), (3, 3))
        self.assertFalse(self.op._consumer_failed)

    def test_wrong_identity_or_mutated_residual_poison_and_clear(self):
        for mutate in (lambda: self.state.update(residual=SimpleNamespace(_version=0)),
                       lambda: setattr(self.residual, '_version', 1)):
            self.op._consumer_failed = False
            self.residual._version = 0
            with self.assertRaisesRegex(RuntimeError, 'BDA ownership'):
                self.phase(mutate)(self.layer, self.residual)
            self.assertTrue(self.op._consumer_failed)
            self.assertIsNone(self.state['residual'])
            self.assertIsNone(self.state['output'])

    def test_no_grad_inference_and_eval_are_rejected_before_consuming(self):
        for mode in ('grad', 'inference', 'eval'):
            self.op._consumer_failed = False
            self.grad_enabled = mode != 'grad'
            self.inference_mode = mode == 'inference'
            self.layer.training = mode != 'eval'
            with self.assertRaisesRegex(RuntimeError, 'grad-enabled training'):
                self.phase()(self.layer, self.residual)
            self.assertEqual(self.state['projection_calls'], 0)
            self.assertIsNone(self.state['residual'])

    def test_version_error_and_missing_bda_cleanup(self):
        class InferenceTensor:
            @property
            def _version(self):
                raise RuntimeError('version unavailable')
        with self.assertRaisesRegex(RuntimeError, 'version unavailable'):
            self.phase()(self.layer, InferenceTensor())
        self.assertIsNone(self.state['residual'])
        self.op._consumer_failed = False
        with self.assertRaisesRegex(RuntimeError, 'exactly one fused'):
            self.op._consumer_phase(lambda x: self.output, self.state, 0, 'hidden_states')(self.layer, self.residual)
        self.assertTrue(self.op._consumer_failed)
        self.assertIsNone(self.state['output'])


@unittest.skipIf(torch is None, 'Run with the PyTorch environment for CPU autograd checks')
class AutogradTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = {'torch': torch, '_gather_along_first_dim': lambda x: x,
                 'parallel_state': SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 1)}
        exec(adapter.consumer_autograd_source(), scope)
        cls.Consumer = scope['ConsumerGemmRSFunction']
        tree = ast.parse((REPO/'tools/a800/phase5/e2e/adapter.py').read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='GemmRSFunction')
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<original CR2 autograd>', 'exec'), scope)
        cls.Control = scope['GemmRSFunction']

    def test_complete_input_parameter_residual_gradients_and_three_updates(self):
        class Operator:
            owns_output = True
            def forward(self, x, w, reduce_scatter_option=None):
                return x @ w.t()
            def forward_consumer(self, x, w, b, r, reduce_scatter_option=None):
                return r + ((x @ w.t()) + b)
        torch.manual_seed(871)
        tensors = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((4,2,3),(5,3),(5,),(4,2,5))]
        left = [t.clone().requires_grad_() for t in tensors]
        right = [t.clone().requires_grad_() for t in tensors]
        for step in range(3):
            actual = self.Consumer.apply(*left, Operator(), None)
            self.assertIsNone(actual._base)
            projection = self.Control.apply(right[0], right[1], Operator(), None)
            expected = right[3] + (projection + right[2])
            self.assertTrue(torch.equal(actual, expected))
            gradient = torch.randn(actual.shape, dtype=torch.bfloat16)
            a = torch.autograd.grad(actual, left, gradient)
            b = torch.autograd.grad(expected, right, gradient)
            for index, (ga, gb) in enumerate(zip(a,b)):
                self.assertTrue(torch.equal(ga, gb), (step,index))
            with torch.no_grad():
                for pa, pb, ga, gb in zip(left,right,a,b):
                    pa.add_(ga, alpha=-.125); pb.add_(gb, alpha=-.125)
                    self.assertTrue(torch.equal(pa,pb))

    def test_half_ulp_rounding_witness_and_subnormal_adds(self):
        x = torch.tensor(1., dtype=torch.bfloat16)
        bias = torch.tensor(1/256, dtype=torch.bfloat16)
        residual = torch.tensor(-1., dtype=torch.bfloat16)
        rounded_twice = residual + (x+bias)
        rounded_once = (residual.float()+x.float()+bias.float()).bfloat16()
        self.assertEqual(rounded_twice.item(), 0.)
        self.assertNotEqual(rounded_twice.item(), rounded_once.item())
        tiny = torch.nextafter(torch.tensor(0.,dtype=torch.bfloat16),torch.tensor(1.,dtype=torch.bfloat16))
        self.assertNotEqual((tiny+tiny).item(), 0.)


if __name__ == '__main__':
    unittest.main()
