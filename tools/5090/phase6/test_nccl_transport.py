"""CPU-only packet schema and stream-order tests; no fake GPU measurements."""
from contextlib import contextmanager
import gc
import importlib.util
from pathlib import Path
import random
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rtx5090_nccl_transport_tested', HERE/'nccl_transport.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class Tensor:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record_stream(self, stream):
        self.log.append(('retain', self.name, stream.cuda_stream))


class Cuda:
    device = 3
    stream = 1729
    capture = False

    def current_device(self):
        return self.device

    def current_stream(self):
        return SimpleNamespace(cuda_stream=self.stream)

    def is_current_stream_capturing(self):
        return self.capture


class Native:
    def __init__(self, log, stage=None):
        self.log, self.stage = log, stage
        self.phase = self.producers = self.decoders = self.consumers = self.configures = 0
        self.received = False
        self.closed = False

    def reconfigure(self, stream):
        self.log.append(('configure', stream))
        self.configures += 1

    def produce(self, x, w, stream):
        if self.stage == 'produce':
            raise RuntimeError('injected producer failure')
        if self.phase:
            raise RuntimeError('overwrote prior native transaction')
        self.log.append(('produce_pack', stream))
        self.received = False
        self.phase = 1
        self.producers += 1

    def decode(self, output, bias, residual, stream):
        if not self.received:
            raise RuntimeError('decode raced unfinished NCCL work')
        if self.stage == 'decode':
            raise RuntimeError('injected decoder failure')
        self.log.append(('unpack_decode', stream))
        self.decoders += 1
        self.consumers += int(bias is not None)
        self.phase = 0

    def state(self):
        return dict(protocol_version=1, initializations=int(self.producers > 0),
                    producer_calls=self.producers, decode_calls=self.decoders,
                    bias_residual_calls=self.consumers, phase=self.phase, failed=False,
                    bound_k=512, reconfigure_calls=self.configures, workspace_bytes=4096)

    def close(self):
        self.log.append(('close',))
        self.closed = True


class Dist:
    def __init__(self, native, log, stage=None):
        self.native, self.log, self.stage = native, log, stage

    def all_to_all_single(self, received, sent, **kwargs):
        if self.stage == 'exchange':
            raise RuntimeError('injected exchange failure')
        if self.native.phase != 1:
            raise RuntimeError('exchange raced producer')
        self.log.append(('exchange', received.name, sent.name, kwargs))

        def wait():
            self.log.append(('wait',))
            if self.stage == 'wait':
                raise RuntimeError('injected NCCL wait failure')
            if self.stage == 'false_wait':
                return False
            self.native.received = True
            return True
        return SimpleNamespace(wait=wait)


def exchange(stage=None, reconfigure=False, profile=False):
    log, cuda = [], Cuda()
    native = Native(log, stage)
    @contextmanager
    def scope(name):
        log.append(('scope', name))
        yield
    torch = SimpleNamespace(cuda=cuda, profiler=SimpleNamespace(record_function=scope))
    tensors = {key: Tensor(key, log) for key in ('input', 'weight', 'output', 'bias', 'residual', 'tx_wire', 'rx_wire')}
    instance = t._Exchange(native, torch, Dist(native, log, stage), tensors,
                           t.wire_layout(None, 3), 'TP8', 3, 1729,
                           reconfigure=reconfigure, profile=profile)
    return instance, tensors, log


class NativeInitialization(unittest.TestCase):
    def test_all_eight_ranks_reach_native_create_with_matching_device(self):
        for rank in range(8):
            for k_local in (512, 2048):
                for cache in (False, True):
                    with self.subTest(rank=rank, k_local=k_local, cache=cache):
                        library = Mock()
                        library.rtx5090_local_protocol_version.return_value = 1
                        library.rtx5090_local_create.return_value = 0
                        torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: rank))
                        buffers = {name: SimpleNamespace(data_ptr=lambda: 1234) for name in
                                   ('tx_bf16', 'tx_packets', 'rx_bf16', 'rx_packets',
                                    'tx_offsets', 'rx_offsets', 'tx_wire', 'rx_wire')}
                        layout = t.wire_layout(None, rank)
                        with patch.object(t.ctypes, 'CDLL', return_value=library):
                            runtime = t._NativeRuntime('CPU-FIXTURE.so', torch, rank, k_local,
                                                       cache, None, buffers, layout, 1729)
                        self.assertEqual((runtime.rank, runtime.device), (rank, rank))
                        arguments = library.rtx5090_local_create.call_args.args
                        self.assertEqual(arguments[1:4], (rank, k_local, int(cache)))
                        self.assertEqual(arguments[4:13], (None, *([1234]*8)))
                        self.assertEqual(arguments[13:], (layout['tx_bytes'], layout['rx_bytes'], 1729))
                        library.rtx5090_local_create.assert_called_once()

    def test_wrong_device_is_rejected_before_loading_native_library(self):
        for rank in range(8):
            torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: (rank+1) % 8))
            with patch.object(t.ctypes, 'CDLL') as library:
                with self.subTest(rank=rank), self.assertRaisesRegex(RuntimeError, 'matching CUDA device'):
                    t._NativeRuntime('unused.so', torch, rank, 512, True, None, {}, {}, 1729)
                library.assert_not_called()


class ScratchReuse(unittest.TestCase):
    def test_pool_releases_unused_storage_and_rejects_poisoned_storage(self):
        pool = t._ScratchPool()
        factory = Mock(return_value={'CPU': object()})
        first = pool.acquire(('stream', 0), factory, 'group')
        self.assertIs(pool.acquire(('stream', 0), factory, 'group'), first)
        factory.assert_called_once()
        second = pool.acquire(('stream', 1), factory, 'group')
        self.assertIsNot(first, second)
        first.failed = True
        with self.assertRaisesRegex(RuntimeError, 'failed shared'):
            pool.acquire(('stream', 0), factory, 'group')
        del first
        gc.collect()
        self.assertNotIn(('stream', 0), pool.entries)
        self.assertFalse(pool.acquire(('stream', 0), factory, 'group').failed)

    def test_siblings_serialize_and_share_failure_but_close_independently(self):
        first, x, _ = exchange()
        second, y, log = exchange()
        scratch = t._Scratch({}, 'TP8')
        for instance in (first, second):
            instance.scratch, instance.lock = scratch, scratch.lock
        first.run(x['input'], x['weight'], x['output'])
        second.run(y['input'], y['weight'], y['output'])
        first.close()
        second.run(y['input'], y['weight'], y['output'])
        self.assertEqual(second.calls, 2)
        scratch.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'reentrant'):
                first.run(x['input'], x['weight'], x['output'])
        finally:
            scratch.lock.release()
        before = list(log)
        with self.assertRaisesRegex(RuntimeError, 'failed shared'):
            second.run(y['input'], y['weight'], y['output'])
        self.assertEqual(before, log)
        self.assertTrue(second.state()['failed'])

    def test_64_projections_keep_private_packets_and_only_four_scratch_sets(self):
        # Model the actual constructors and allocation sizes without allocating
        # GPU memory. Different role/slot/stream/group/mask must not alias.
        allocations = []
        class Storage:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
                allocations.append(self)
                self.pointer = len(allocations)*4096
            def numel(self):
                result = 1
                for dimension in self.shape:
                    result *= dimension
                return result
            def element_size(self):
                return {'bf16': 2, 'u8': 1, 'i64': 8}[self.dtype]
            def data_ptr(self):
                return self.pointer
            def to(self, **_):
                return Storage(self.shape, self.dtype)
        def tensor(value, dtype, **_):
            shape = (len(value), len(value[0])) if isinstance(value[0], list) else (len(value),)
            return Storage(shape, dtype)
        cuda = Cuda()
        cuda.device_count = lambda: 8
        cuda.get_device_properties = lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170)
        dist = SimpleNamespace(get_rank=lambda _: 3, get_world_size=lambda _: 8,
                               get_backend=lambda _: 'nccl',
                               all_gather_object=lambda target, value, **_: target.__setitem__(slice(None), [value]*8))
        torch = SimpleNamespace(cuda=cuda, distributed=dist, bfloat16='bf16', uint8='u8', int64='i64',
                                tensor=tensor, empty=lambda shape, dtype, **_: Storage(shape, dtype))
        library = (HERE/'lib/libflux_cuda.so').resolve()
        pool = t._ScratchPool()
        mask = [[0]*512 for _ in range(8)]
        group = object()
        with patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist}), \
             patch.dict(t.os.environ, NCCL_P2P_DISABLE='1', LOCAL_WORLD_SIZE='8', LOCAL_RANK='3'), \
             patch.object(t, 'NATIVE_ARM', 'ma11'), patch.object(t, '_SCRATCH_POOL', pool), \
             patch.object(t, '_NativeRuntime', side_effect=lambda *args: Mock()), \
             patch.object(t.Path, 'read_text', return_value=str(library)):
            ops = [t.GemmRSTacoDoubleBuffered(group, 2048, 4096, k, selected=mask, quantization_mode='targeted')
                   for _ in range(32) for k in (512, 2048)]
            slots = [slot for op in ops for slot in op.slots]
            self.assertEqual(len(pool.entries), 4)
            for key in t.SHARED_BUFFERS:
                self.assertEqual(len({slot.buffers[key].data_ptr() for slot in slots}), 4)
            self.assertEqual(len({slot.buffers['rx_packets'].data_ptr() for slot in slots}), 128)
            self.assertEqual(len({slot.selected.data_ptr() for slot in slots}), 128)
            self.assertEqual(len({id(slot._exchange.native) for slot in slots}), 128)
            scratch_bytes = sum(value.numel()*value.element_size() for value in ops[0].slots[0]._scratch.buffers.values())
            # The all-BF16 wire is the largest possible selective allocation.
            self.assertEqual(scratch_bytes, 68.5*1024**2)
            self.assertGreater((128-4)*scratch_bytes, 8*1024**3)
            cuda.stream += 1
            other_stream = t.GemmRSTacoDoubleBuffered(group, 2048, 4096, 512, selected=mask, quantization_mode='targeted')
            self.assertEqual(len(pool.entries), 6)
            cuda.stream -= 1
            other_group = t.GemmRSTacoDoubleBuffered(object(), 2048, 4096, 512, selected=mask, quantization_mode='targeted')
            mask[3][0] = 1
            other_mask = t.GemmRSTacoDoubleBuffered(group, 2048, 4096, 512, selected=mask, quantization_mode='targeted')
            self.assertEqual(len(pool.entries), 10)
            for op in [*ops, other_stream, other_group, other_mask]:
                op.close()


class Layout(unittest.TestCase):
    def test_all_remote_and_all_bf16_byte_counts(self):
        for rank in range(8):
            compressed = t.wire_layout(None, rank)
            plain = t.wire_layout([[0]*512 for _ in range(8)], rank)
            self.assertEqual(compressed['tx_bytes'], 7*64*t.TILE_PACKET_BYTES)
            self.assertEqual(plain['tx_bytes'], 7*64*t.TILE_BF16_BYTES)
            self.assertEqual(compressed['tx_splits'][rank], 0)
            self.assertEqual(plain['rx_splits'][rank], 0)
            self.assertEqual(compressed['tx_offsets'][rank*64:(rank+1)*64], [-1]*64)

    def test_mixed_layout_matches_all_56_directed_edges_and_tile_boundaries(self):
        rng = random.Random(503)
        mask = [[0 if tile//64 == src else int(rng.random() < .2) for tile in range(512)] for src in range(8)]
        layouts = [t.wire_layout(mask, rank) for rank in range(8)]
        for src in range(8):
            tx = layouts[src]
            self.assertEqual(tx['tx_bytes'], sum(tx['tx_splits']))
            for dst in range(8):
                rx = layouts[dst]
                self.assertEqual(tx['tx_splits'][dst], rx['rx_splits'][src])
                if src == dst:
                    continue
                tx_base = sum(tx['tx_splits'][:dst])
                rx_base = sum(rx['rx_splits'][:src])
                cursor = 0
                for tile in range(64):
                    self.assertEqual(tx['tx_offsets'][dst*64+tile]-tx_base, cursor)
                    self.assertEqual(rx['rx_offsets'][src*64+tile]-rx_base, cursor)
                    cursor += t.TILE_PACKET_BYTES if mask[src][dst*64+tile] else t.TILE_BF16_BYTES
                self.assertEqual(cursor, tx['tx_splits'][dst])

    def test_invalid_or_local_quantization_masks_rejected(self):
        for mask in ([], [[0]*512]*7, [[2]*512]*8, [[1]*512]*8):
            with self.assertRaises(RuntimeError):
                t.wire_layout(mask, 0)
        with self.assertRaises(RuntimeError):
            t.wire_layout(None, 8)

    def test_generated_frontends_are_standalone_and_do_not_reference_ipc(self):
        for arm in t.NATIVE_ARMS:
            source = t.frontend_source(arm)
            compile(source, arm, 'exec')
            self.assertIn('NATIVE_ARM = '+repr(arm), source)
            for token in ('init_flux_shm(', 'create_tensor_list(', 'can_device_access_peer(', 'cudaIpc', '.peers'):
                self.assertNotIn(token, source)


class Synchronization(unittest.TestCase):
    def test_every_reuse_is_produce_then_exchange_wait_then_decode(self):
        instance, tensors, log = exchange(profile=True)
        for _ in range(3):
            self.assertIs(instance.run(tensors['input'], tensors['weight'], tensors['output']), tensors['output'])
        stages = [event[0] for event in log if event[0] in ('produce_pack', 'exchange', 'wait', 'unpack_decode')]
        self.assertEqual(stages, ['produce_pack', 'exchange', 'wait', 'unpack_decode']*3)
        calls = [event for event in log if event[0] == 'exchange']
        self.assertTrue(all(event[3]['async_op'] is True for event in calls))
        self.assertTrue(all(event[3]['input_split_sizes'][3] == event[3]['output_split_sizes'][3] == 0 for event in calls))
        state = instance.state()
        for key in ('calls', 'producer_calls', 'pack_calls', 'exchange_calls', 'wait_calls', 'unpack_calls', 'decode_calls'):
            self.assertEqual(state[key], 3)
        self.assertEqual(state['native_phase'], 0)
        self.assertFalse(state['peer_memory'])
        self.assertFalse(state['pending_exchange'])
        self.assertEqual(state['native_workspace_bytes'], 4096)
        self.assertEqual([event[1] for event in log if event[0] == 'scope'],
                         ['rtx5090.transport.produce_pack', 'rtx5090.transport.exchange', 'rtx5090.transport.unpack_decode']*3)

    def test_failure_poisoning_at_every_stage(self):
        for stage in ('produce', 'exchange', 'wait', 'false_wait', 'decode'):
            instance, tensors, log = exchange(stage)
            with self.subTest(stage=stage), self.assertRaises(RuntimeError):
                instance.run(tensors['input'], tensors['weight'], tensors['output'])
            before = list(log)
            with self.assertRaisesRegex(RuntimeError, 'failed/closed'):
                instance.run(tensors['input'], tensors['weight'], tensors['output'])
            self.assertEqual(before, log)
            self.assertTrue(instance.failed)
            self.assertEqual(instance.calls, 0)

    def test_other_stream_device_capture_and_reentry_are_rejected(self):
        for field, value in (('stream', 7), ('device', 4), ('capture', True)):
            instance, tensors, log = exchange()
            setattr(instance.torch.cuda, field, value)
            with self.assertRaises(RuntimeError):
                instance.run(tensors['input'], tensors['weight'], tensors['output'])
            self.assertTrue(instance.failed)
            self.assertFalse(log)
        instance, tensors, _ = exchange()
        instance.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'reentrant'):
                instance.run(tensors['input'], tensors['weight'], tensors['output'])
        finally:
            instance.lock.release()
        self.assertTrue(instance.failed)

    def test_a_reconfigures_each_call_and_consumer_tensors_stay_on_stream(self):
        instance, tensors, log = exchange(reconfigure=True, profile=True)
        for _ in range(2):
            instance.run(*(tensors[name] for name in ('input', 'weight', 'output', 'bias', 'residual')))
        state = instance.state()
        self.assertEqual(state['configuration_scope'], 'per_call')
        self.assertEqual(state['reconfigure_calls'], 2)
        self.assertEqual(state['consumer_calls'], 2)
        self.assertEqual(sum(event[0] == 'retain' and event[1] == 'residual' for event in log), 2)
        self.assertIn(('scope', 'rtx5090.transport.unpack_decode_consumer'), log)

    def test_double_buffer_rotation_and_failure(self):
        double = object.__new__(t.GemmRSTacoDoubleBuffered)
        indices = []
        def slot(index):
            def forward(*_):
                indices.append(index)
                return index
            return SimpleNamespace(forward=forward)
        double.slots = [slot(0), slot(1)]
        double.next_slot, double._failed, double._lock = 0, False, threading.Lock()
        for expected in (0, 1, 0, 1, 0):
            self.assertEqual(double.forward(None, None), expected)
        self.assertEqual(indices, [0, 1, 0, 1, 0])
        def fail(*_):
            raise RuntimeError('slot failure')
        double.slots[1].forward = fail
        with self.assertRaisesRegex(RuntimeError, 'slot failure'):
            double.forward(None, None)
        self.assertEqual(double.next_slot, 1)
        with self.assertRaisesRegex(RuntimeError, 'failed double'):
            double.forward(None, None)

    def test_close_shares_the_submission_lock_and_is_idempotent(self):
        instance, tensors, log = exchange()
        instance.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'active NCCL'):
                instance.close()
        finally:
            instance.lock.release()
        self.assertFalse(instance.closed)
        instance.close()
        instance.close()
        self.assertEqual(sum(event == ('close',) for event in log), 1)
        with self.assertRaisesRegex(RuntimeError, 'failed/closed'):
            instance.run(tensors['input'], tensors['weight'], tensors['output'])


if __name__ == '__main__':
    unittest.main()
