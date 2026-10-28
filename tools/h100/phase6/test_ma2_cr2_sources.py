"""CR2 host/lifecycle regression checks; CUDA timing is performed on TP8."""
import ast
from pathlib import Path
import threading
import unittest

from common import REPO
import ma2_cr2_sources as source


REFERENCE = REPO / 'outputs/h100/phase6/ma2-c-ablation-validation-20261023/build-final/CR'


class FakeNative:
    def __init__(self, packet=1000, k=512):
        self.packet, self.k = packet, k
        self.calls = 0
        self.seen = []
        self.fail = False

    def forward(self, input, weight, reduce_scatter_option):
        self.seen.append((input, weight, reduce_scatter_option))
        if self.fail:
            raise RuntimeError('synthetic failure after producer enqueue')
        self.calls += 1
        return object()

    def cr2_submission_state(self):
        return [1, int(self.calls > 0), self.calls, self.k, self.packet, 0]


class CR2SourcesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = {name: (REFERENCE / name).read_text() for name in
                        (source.RUNTIME, source.HEADER, source.FRONTEND, source.WRAPPER)}
        cls.changed = source.sources(REFERENCE)
        tree = ast.parse(cls.changed[source.FRONTEND])
        classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)],
                             type_ignores=[])
        namespace = {}
        exec(compile(classes, '<generated-cr2-frontend>', 'exec'), namespace)
        cls.Single = namespace['GemmRSTaco']
        cls.Double = namespace['GemmRSTacoDoubleBuffered']

    def single(self, packet=1000, k=512):
        obj = self.Single.__new__(self.Single)
        obj._lock = threading.Lock()
        obj._failed = False
        obj.op = FakeNative(packet, k)
        obj.option = object()
        return obj

    def double(self):
        obj = self.Double.__new__(self.Double)
        obj._lock = threading.Lock()
        obj._failed = False
        obj.slots = [self.single(1000), self.single(2000)]
        obj.next_slot = 0
        return obj

    def test_frozen_decoder_and_packet_abi_are_unchanged(self):
        before = self.original[source.RUNTIME]
        after = self.changed[source.RUNTIME]
        # Only TLS binding/reset and append-only host ABI change in the CUDA TU.
        restored = after[:after.index('// Host-only binding;')].rstrip()
        restored = restored.replace('static thread_local int cr2_bound_k = 0;\n', '')
        restored = restored.replace('active = FluxTacoConfig{}; cr2_bound_k = 0;',
                                    'active = FluxTacoConfig{};')
        self.assertEqual(restored, before.rstrip())
        self.assertTrue(self.changed[source.HEADER].startswith(self.original[source.HEADER]))
        self.assertNotIn('src/gemm_rs/ma2_c_decode.cuh', self.changed)
        self.assertNotIn('src/gemm_rs/taco_codec.cuh', self.changed)

    def test_non_cr_input_is_rejected(self):
        c_runtime = REFERENCE.parent / 'C' / source.RUNTIME
        with self.assertRaisesRegex(ValueError, 'original mixed-source CR'):
            source.runtime(c_runtime.read_text())

    def test_changed_source_contract_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'Source contract changed'):
            source.wrapper(self.original[source.WRAPPER].replace(
                '  const FluxTacoConfig owned_taco = taco_config();', '/* contract drift */'))

    def test_double_buffer_reuse_passes_fresh_inputs_and_owns_outputs(self):
        op = self.double()
        inputs = [(object(), object()) for _ in range(8)]
        outputs = [op.forward(*pair) for pair in inputs]
        self.assertEqual(len({id(output) for output in outputs}), 8)
        for slot_id, slot in enumerate(op.slots):
            self.assertEqual(slot.op.seen,
                             [(a, b, slot.option) for a, b in inputs[slot_id::2]])
        self.assertEqual(op.next_slot, 0)

    def test_failure_poisons_both_slots_and_preserves_slot_index(self):
        op = self.double()
        op.forward(object(), object())
        op.slots[1].op.fail = True
        with self.assertRaisesRegex(RuntimeError, 'after producer enqueue'):
            op.forward(object(), object())
        self.assertEqual(op.next_slot, 1)
        op.slots[1].op.fail = False
        with self.assertRaisesRegex(RuntimeError, 'after a failed forward'):
            op.forward(object(), object())
        self.assertEqual(len(op.slots[0].op.seen), 1)
        self.assertEqual(len(op.slots[1].op.seen), 1)
        self.assertTrue(op.cr2_submission_state()[1]['failed'])

    def test_runtime_evidence_uses_per_slot_native_counters(self):
        op = self.double()
        initial = op.cr2_submission_state()
        self.assertEqual([x['local_packet_ptr'] for x in initial], [1000, 2000])
        self.assertEqual([x['initializations'] for x in initial], [0, 0])
        for _ in range(7):
            op.forward(object(), object())
        final = op.cr2_submission_state()
        self.assertEqual([x['slot'] for x in final], [0, 1])
        self.assertEqual([x['successful_calls'] for x in final], [4, 3])
        self.assertTrue(all(x['initializations'] == 1 and x['bound_k'] == 512 and
                            x['protocol_version'] == 1 and not x['failed'] for x in final))


if __name__ == '__main__':
    unittest.main()
