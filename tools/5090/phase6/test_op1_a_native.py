"""Direct packet interoperability, isolated transforms and stream-lifetime gates."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import op1_a_native as op1
from test_nccl_transport import Cuda, Dist, Native, Tensor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def frontend():
    result = ModuleType('op1_native_test_runtime')
    result.__file__ = str(HERE / 'op1_a_native.py')
    exec(compile(op1.frontend_source(), result.__file__, 'exec'), result.__dict__)
    return result


def exchange(stage=None):
    module = frontend()
    log, cuda = [], Cuda()
    native = Native(log, stage)

    @contextmanager
    def scope(name):
        log.append(('scope', name))
        yield

    torch = SimpleNamespace(cuda=cuda, profiler=SimpleNamespace(record_function=scope))
    tensors = {key: Tensor(key, log) for key in ('input', 'weight', 'output', 'tx_wire', 'rx_wire')}
    instance = module._Exchange(native, torch, Dist(native, log, stage), tensors,
                               module.wire_layout(None, 3), 'TP8', 3, 1729,
                               profile=True)
    return instance, tensors, log


class DirectLayout(unittest.TestCase):
    def test_all_56_packets_match_exact_destination_and_source_slots(self):
        # This models all_to_all_single's rank-ordered nonuniform splits. The
        # actual CUDA smoke separately proves encoder/decoder numeric identity.
        sends = [[None] * 7 for _ in range(8)]
        receives = [[None] * 7 for _ in range(8)]
        for source in range(8):
            for destination in range(8):
                if source == destination:
                    continue
                sends[source][op1.peer_slot(source, destination)] = (source, destination)
        for destination in range(8):
            for source in range(8):
                if source != destination:
                    receives[destination][op1.peer_slot(destination, source)] = sends[source][op1.peer_slot(source, destination)]
        for rank in range(8):
            expected = [(source, rank) for source in range(8) if source != rank]
            self.assertEqual(receives[rank], expected)
            layout = op1.wire_layout(rank)
            self.assertEqual(layout['tx_splits'][rank], 0)
            self.assertEqual(sum(layout['tx_splits']), 7 * 256 * 32 * 136)
            self.assertEqual(layout, frontend().wire_layout(None, rank))
            with self.assertRaises(ValueError):
                op1.peer_slot(rank, rank)

    def test_selective_masks_cannot_enter_direct_packet_protocol(self):
        with self.assertRaises(RuntimeError):
            frontend().wire_layout([[0] * 512] * 8, 0)
        for rank in (-1, 8, True, 1.5):
            with self.assertRaises(ValueError):
                op1.wire_layout(rank)


class SourceIsolation(unittest.TestCase):
    def fixture(self, root):
        frozen = REPO / 'tools/h100/phase6/fused_v2/src/gemm_rs'
        files = {
            'src/gemm_rs/taco_runtime.cu': (frozen / 'taco_runtime.cu').read_text(),
            'src/gemm_rs/epilogue_evt.hpp': (frozen / 'epilogue_evt.hpp').read_text().replace(
                'unsigned char* slot = c.peers[dst_rank] + c.rank * taco_source_bytes(c);',
                'unsigned char* slot = c.peers[dst_rank];'),
            'src/gemm_rs/local_transport.cu': '#define RTX5090_LOCAL_CONSUMER 0\n#define RTX5090_LOCAL_SELECTIVE 0\n'
                                           + (HERE / 'local_transport.cu').read_text()}
        for name, source in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        return files

    def test_only_receiver_addresses_change_in_frozen_decoder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = self.fixture(root)
            changes = op1.overrides(root)
            for name, source in original.items():
                self.assertEqual((root / name).read_text(), source)
            runtime = changes['src/gemm_rs/taco_runtime.cu']
            self.assertEqual(runtime.replace(
                'const unsigned char* slot = c.peers[c.rank] + (src - int(src > c.rank)) * source_bytes;',
                'const unsigned char* slot = c.peers[c.rank] + src * source_bytes;').replace(
                'unsigned char* slot = c.peers[dst];  // Local destination packet, already compacted.',
                'unsigned char* slot = c.peers[dst] + c.rank*taco_source_bytes(c);'),
                original['src/gemm_rs/taco_runtime.cu'])
            native = changes['src/gemm_rs/local_transport.cu']
            self.assertNotIn('rtx5090_pack_tiles_kernel', native)
            self.assertNotIn('rtx5090_unpack_tiles_kernel', native)
            self.assertIn('OP1 A configuration is immutable', native)
            self.assertIn('rtx5090_local_protocol_version() { return 2; }', native)
            ast.parse(changes['python/flux/gemm_rs_taco.py'])
            self.assertNotIn('cudaIpc', native)

    def test_reject_old_peer_or_selective_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / 'src/gemm_rs/epilogue_evt.hpp'
            path.write_text('legacy peer producer')
            with self.assertRaisesRegex(ValueError, 'local producer'):
                op1.overrides(root)
        with self.assertRaisesRegex(ValueError, 'nonselective'):
            op1.native_source('#define RTX5090_LOCAL_SELECTIVE 1\n')

    def test_native_abi_rejects_old_library_and_uses_null_offset_arguments(self):
        module = frontend()
        for version in (1, 2):
            library = Mock()
            library.rtx5090_local_protocol_version.return_value = version
            library.rtx5090_local_create.return_value = 0
            torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 3))
            buffers = {key: SimpleNamespace(data_ptr=lambda: 1234) for key in
                       ('tx_bf16', 'tx_packets', 'rx_bf16', 'rx_packets', 'tx_wire', 'rx_wire')}
            with patch.object(module.ctypes, 'CDLL', return_value=library):
                if version == 1:
                    with self.assertRaisesRegex(RuntimeError, 'protocol'):
                        module._NativeRuntime('lib', torch, 3, 512, False, None, buffers,
                                              module.wire_layout(None, 3), 1729)
                else:
                    module._NativeRuntime('lib', torch, 3, 512, False, None, buffers,
                                          module.wire_layout(None, 3), 1729)
                    arguments = library.rtx5090_local_create.call_args.args
                    self.assertEqual(arguments[3], 0)
                    self.assertEqual(arguments[9:11], (None, None))


class Synchronization(unittest.TestCase):
    def test_64_module_instances_share_scratch_and_keep_receive_storage_private(self):
        module = frontend()
        allocations = []

        class Storage:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
                allocations.append(self)
                self.pointer = len(allocations) * 4096

            def data_ptr(self):
                return self.pointer

            def numel(self):
                result = 1
                for dimension in self.shape:
                    result *= dimension
                return result

            def element_size(self):
                return 2 if self.dtype == 'bf16' else 1

        cuda = Cuda()
        cuda.device_count = lambda: 8
        cuda.get_device_properties = lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170)
        dist = SimpleNamespace(get_rank=lambda _: 3, get_world_size=lambda _: 8,
                               get_backend=lambda _: 'nccl',
                               all_gather_object=lambda target, value, **_: target.__setitem__(slice(None), [value]*8))
        torch = SimpleNamespace(cuda=cuda, distributed=dist, bfloat16='bf16', uint8='u8',
                                empty=lambda shape, dtype, **_: Storage(shape, dtype))
        library = (HERE / 'lib/libflux_cuda.so').resolve()
        group = object()
        with patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist}), \
             patch.dict(module.os.environ, NCCL_P2P_DISABLE='1', LOCAL_WORLD_SIZE='8', LOCAL_RANK='3'), \
             patch.object(module, '_NativeRuntime', side_effect=lambda *args: Mock()), \
             patch.object(module.Path, 'read_text', return_value=str(library)):
            ops = [module.GemmRSTaco(group, 2048, 4096, k) for _ in range(32) for k in (512, 2048)]
            self.assertEqual(len(module._SCRATCH_POOL.entries), 2)
            self.assertEqual(len({op.buffers['rx_wire'].data_ptr() for op in ops}), 64)
            for key in module.SHARED_BUFFERS:
                self.assertEqual(len({op.buffers[key].data_ptr() for op in ops}), 2)
            for op in ops:
                self.assertIs(op.buffers['tx_packets'], op.buffers['tx_wire'])
                self.assertIs(op.buffers['rx_packets'], op.buffers['rx_wire'])
                self.assertIs(op.buffers['tx_bf16'], op.buffers['rx_bf16'])
                self.assertTrue(op.owns_output and op.owns_configuration)
                self.assertNotIn('tx_offsets', op.buffers)
            self.assertEqual(sum(value.numel()*value.element_size() for value in ops[0]._scratch.buffers.values()),
                             7 * op1.PACKET_BYTES + 8 * 256 * 4096 * 2)
            cuda.stream += 1
            other = module.GemmRSTaco(group, 2048, 4096, 512)
            self.assertEqual(len(module._SCRATCH_POOL.entries), 3)
            self.assertNotEqual(other.buffers['tx_wire'].data_ptr(), ops[0].buffers['tx_wire'].data_ptr())
            for op in [*ops, other]:
                op.close()

    def test_direct_stages_order_and_no_false_repacking_counters(self):
        instance, tensors, log = exchange()
        for _ in range(7):
            instance.run(tensors['input'], tensors['weight'], tensors['output'])
        self.assertEqual([entry[1] for entry in log if entry[0] == 'scope'],
                         ['rtx5090.transport.produce_direct', 'rtx5090.transport.exchange',
                          'rtx5090.transport.decode_direct'] * 7)
        state = instance.state()
        for key in ('pack_calls', 'unpack_calls', 'reconfigure_calls', 'consumer_calls'):
            self.assertEqual(state[key], 0)
        for key in ('calls', 'producer_calls', 'exchange_calls', 'wait_calls', 'decode_calls'):
            self.assertEqual(state[key], 7)
        self.assertEqual(state['native_api'], op1.NATIVE_API)
        self.assertEqual(state['packet_layout'], op1.PACKET_LAYOUT)
        self.assertFalse(state['gemm_submission_cache'])

    def test_failures_cannot_reuse_or_decode_unreceived_packet(self):
        for stage in ('produce', 'exchange', 'wait', 'false_wait', 'decode'):
            instance, tensors, log = exchange(stage)
            with self.subTest(stage=stage):
                with self.assertRaises(RuntimeError):
                    instance.run(tensors['input'], tensors['weight'], tensors['output'])
                before = list(log)
                with self.assertRaisesRegex(RuntimeError, 'failed/closed'):
                    instance.run(tensors['input'], tensors['weight'], tensors['output'])
                self.assertEqual(log, before)
                self.assertEqual(instance.calls, 0)


if __name__ == '__main__':
    unittest.main()
