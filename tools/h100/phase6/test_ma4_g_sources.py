"""MA4-G generated Python boundary contracts; no CUDA execution is simulated.

The fake native operator only records arguments and owns ordinary Python output
snapshots. These tests establish wrapper forwarding/lifecycle behavior, while
real native compilation and TP8 gates validate graph APIs and GPU semantics.
"""
import ast
from dataclasses import dataclass
import threading
import unittest
from types import SimpleNamespace

from common import REPO
import ma2_cr2_sources as cr2
import ma4_g_sources as source


@dataclass
class Storage:
    values: list


@dataclass
class OwnedOutput:
    inputs: tuple
    weights: tuple
    generation: int


class FakeNative:
    def __init__(self, packet=1000, k=512):
        self.packet, self.k = packet, k
        self.calls = self.builds = 0
        self.prepared = []
        self.seen = []
        self.resources = None
        self.fail_prepare = self.fail_forward = False
        self.state_delta = 0

    def ma4_hold_resources(self, resources):
        self.resources = list(resources)

    def ma4_prepare_graph(self, input, weight, option):
        self.prepared.append((input, weight, option))
        if self.fail_prepare:
            raise RuntimeError('synthetic graph construction failure')
        self.builds = 1

    def forward(self, input, weight, reduce_scatter_option):
        self.seen.append((input, weight, reduce_scatter_option))
        if self.fail_forward:
            raise RuntimeError('synthetic graph update failure')
        self.builds = 1
        self.calls += 1
        return OwnedOutput(tuple(input.values), tuple(weight.values), self.calls)

    def cr2_submission_state(self):
        return [1, self.builds, self.calls, self.k, self.packet, 0]

    def ma4_graph_state(self):
        ready = self.builds != 0
        identities = [self.packet * 10 + i if ready else 0 for i in range(6)]
        # Keep this fixture independent of the Python key list: a schema drift
        # must not silently relabel native counters or resource identities.
        values = [1, self.builds, self.calls, self.calls, self.calls,
                  4 if ready else 0, 3 if ready else 0, 1 if ready else 0,
                  self.k, *identities, self.packet, 0,
                  100 if ready else 0, 0, 77, self.packet * 100 if ready else 0]
        if self.state_delta < 0:
            return values[:self.state_delta]
        return values + [999] * self.state_delta


class GraphFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Apply both production transforms to maintained source, without
        # depending on an ignored build artifact or creating a full build tree.
        cls.frontend = source.frontend(cr2.frontend((REPO / source.FRONTEND).read_text()))
        tree = ast.parse(cls.frontend)
        classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)],
                             type_ignores=[])
        namespace = {}
        exec(compile(classes, '<generated-ma4-g-frontend>', 'exec'), namespace)
        cls.Single = namespace['GemmRSTaco']
        cls.Double = namespace['GemmRSTacoDoubleBuffered']
        cls.frontend_tree = tree

    def single(self, packet=1000, k=512):
        obj = self.Single.__new__(self.Single)
        obj._lock = threading.Lock()
        obj._failed = False
        obj.op = FakeNative(packet, k)
        obj.option = object()
        # The real constructor establishes this native ownership before use.
        obj.op.ma4_hold_resources([object() for _ in range(10)])
        return obj

    def double(self, k=512):
        obj = self.Double.__new__(self.Double)
        obj._lock = threading.Lock()
        obj._failed = False
        obj.slots = [self.single(1000, k), self.single(2000, k)]
        obj.next_slot = 0
        return obj

    def test_prepare_builds_both_slots_without_consuming_forward(self):
        for k in (512, 2048):
            with self.subTest(k=k):
                op = self.double(k)
                x, w = Storage([1]), Storage([2])
                op.prepare_graph(x, w)
                self.assertEqual(op.next_slot, 0)
                self.assertEqual([s['graph_build_count'] for s in op.ma4_graph_state()], [1, 1])
                for slot in op.slots:
                    self.assertEqual(slot.op.prepared, [(x, w, slot.option)])
                    self.assertEqual(slot.op.seen, [])
                self.assertEqual([s['successful_calls'] for s in op.cr2_submission_state()], [0, 0])
                self.assertTrue(all(s['graph_launch_count'] == s['gemm_param_update_count'] ==
                                    s['decoder_param_update_count'] == 0 for s in op.ma4_graph_state()))
                op.forward(x, w)
                # Re-preparation is idempotent and preserves a nonzero next slot.
                op.prepare_graph(Storage([3]), Storage([4]))
                self.assertEqual(op.next_slot, 1)
                self.assertEqual([s['graph_launch_count'] for s in op.ma4_graph_state()], [1, 0])
                self.assertEqual([s['graph_build_count'] for s in op.ma4_graph_state()], [1, 1])

    def test_fresh_addresses_and_live_outputs_cross_multiple_slot_reuses(self):
        op = self.double()
        inputs = [(Storage([i, i + 1]), Storage([i + 2])) for i in range(8)]
        op.prepare_graph(*inputs[0])
        outputs = [op.forward(*pair) for pair in inputs]
        self.assertEqual(len({id(output) for output in outputs}), 8)
        snapshots = [(out.inputs, out.weights, out.generation) for out in outputs]
        for slot_id, slot in enumerate(op.slots):
            self.assertEqual(slot.op.seen, [(x, w, slot.option) for x, w in inputs[slot_id::2]])
            self.assertTrue(all(record[0] is pair[0] and record[1] is pair[1]
                                for record, pair in zip(slot.op.seen, inputs[slot_id::2])))
        for x, w in inputs:
            x.values[:] = [99]
            w.values[:] = [98]
        for _ in range(6):
            op.forward(Storage([10]), Storage([11]))
        self.assertEqual([(out.inputs, out.weights, out.generation) for out in outputs], snapshots)
        self.assertEqual(op.next_slot, 0)

    def test_optimizer_style_inplace_updates_reach_both_native_slots(self):
        op = self.double()
        x, w = Storage([1]), Storage([2])
        outputs = []
        for step in range(6):
            w.values[:] = [step]
            outputs.append(op.forward(x, w))
        self.assertEqual([out.weights for out in outputs], [(i,) for i in range(6)])
        self.assertTrue(all(a is x and b is w for slot in op.slots for a, b, _ in slot.op.seen))
        self.assertEqual([s['graph_launch_count'] for s in op.ma4_graph_state()], [3, 3])

    def test_prepare_failure_poisons_double_wrapper_without_advancing(self):
        op = self.double()
        op.slots[1].op.fail_prepare = True
        with self.assertRaisesRegex(RuntimeError, 'construction failure'):
            op.prepare_graph(Storage([1]), Storage([2]))
        self.assertEqual(op.next_slot, 0)
        self.assertEqual([slot.op.calls for slot in op.slots], [0, 0])
        self.assertTrue(op.ma4_graph_state()[1]['failed'])
        op.slots[1].op.fail_prepare = False
        with self.assertRaisesRegex(RuntimeError, 'failed double workspaces'):
            op.prepare_graph(Storage([3]), Storage([4]))
        with self.assertRaisesRegex(RuntimeError, 'failed forward'):
            op.forward(Storage([3]), Storage([4]))
        self.assertEqual([len(slot.op.prepared) for slot in op.slots], [1, 1])

    def test_update_failure_poisons_both_slots_and_preserves_next_slot(self):
        op = self.double()
        op.forward(Storage([1]), Storage([2]))
        op.slots[1].op.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, 'update failure'):
            op.forward(Storage([3]), Storage([4]))
        self.assertEqual(op.next_slot, 1)
        self.assertTrue(op.ma4_graph_state()[1]['failed'])
        op.slots[1].op.fail_forward = False
        with self.assertRaisesRegex(RuntimeError, 'failed forward'):
            op.forward(Storage([5]), Storage([6]))
        with self.assertRaisesRegex(RuntimeError, 'failed double workspaces'):
            op.prepare_graph(Storage([5]), Storage([6]))
        self.assertEqual([len(slot.op.seen) for slot in op.slots], [1, 1])

    def test_state_abi_rejects_truncated_or_extended_native_vectors(self):
        op = self.single(k=2048)
        op.prepare_graph(Storage([1]), Storage([2]))
        state = op.ma4_graph_state()
        self.assertEqual((state['protocol_version'], state['bound_k'], state['local_packet_ptr']),
                         (1, 2048, 1000))
        self.assertEqual((state['node_count'], state['kernel_node_count'], state['memset_node_count']),
                         (4, 3, 1))
        self.assertEqual((state['graph_build_host_ns'], state['stream_id'], state['workspace_ptr']),
                         (100, 77, 100000))
        self.assertFalse(state['failed'])
        for delta in (-1, 1):
            op.op.state_delta = delta
            with self.subTest(delta=delta), self.assertRaisesRegex(RuntimeError, 'state ABI mismatch'):
                op.ma4_graph_state()

    def test_native_constructor_holds_exact_peer_mask_and_optional_index_objects(self):
        single = next(n for n in self.frontend_tree.body
                      if isinstance(n, ast.ClassDef) and n.name == 'GemmRSTaco')
        init = next(n for n in single.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        construction = next(n for n in init.body if isinstance(n, ast.Try) and any(
            isinstance(child, ast.Attribute) and child.attr == 'ma4_hold_resources'
            for child in ast.walk(n)))
        code = compile(ast.Module(body=[construction], type_ignores=[]), '<native-resource-init>', 'exec')
        for indices in (None, Storage([]), Storage([3])):
            with self.subTest(indices=indices):
                resets = []
                obj = SimpleNamespace(peers=[object() for _ in range(8)], selected=object(),
                    quant_tiles=indices, owns_configuration=True,
                    lib=SimpleNamespace(taco_cr2_bind=lambda k: 0, taco_reset=lambda: resets.append(True)))
                native = FakeNative()
                namespace = dict(self=obj, group=object(), m=2048, n=4096, k_local=512,
                    torch=SimpleNamespace(bfloat16=object()),
                    cpp_mod=SimpleNamespace(GemmRS=lambda *args, **kwargs: native))
                exec(code, namespace)
                expected = obj.peers + [obj.selected] + ([] if indices is None else [indices])
                self.assertEqual(len(native.resources), len(expected))
                self.assertTrue(all(a is b for a, b in zip(native.resources, expected)))
                self.assertEqual(resets, [True])

    def test_unmodified_decoder_and_type_safe_update_contract(self):
        original = (REPO / source.GEMM).read_text()
        generated = source.gemm(original)
        self.assertIn('CUTLASS_CHECK(Base::update(args));', generated)
        self.assertIn('void *arguments[] = {&this->params_};', generated)
        self.assertIn('cudaKernelNodeParams parameters = ma4_parameters;', generated)
        self.assertNotIn('memcpy(', source.TYPED_GEMM)
        self.assertNotIn('reinterpret_cast<char', source.TYPED_GEMM)
        # G appends host-only helpers; the CR2 CUDA source is an exact prefix.
        fake_runtime = 'extern "C" int taco_cr2_protocol_version() { return 1; }\n'
        self.assertTrue(source.runtime(fake_runtime).startswith(fake_runtime))
        with self.assertRaisesRegex(ValueError, 'exact CR2 runtime'):
            source.runtime('extern "C" int taco_ma2c_protocol_version() { return 1; }')


if __name__ == '__main__':
    unittest.main()
