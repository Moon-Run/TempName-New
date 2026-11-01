"""CPU arithmetic-layout and submission tests; target GPU probes remain required."""
import ast
from contextlib import nullcontext
from pathlib import Path
import random
import struct
import threading
from types import SimpleNamespace
import unittest

import op3_native as native


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bits(values):
    return b''.join(struct.pack('<f', x) for x in values)


def scalar_hadamard(values):
    result = list(values)
    for distance in (1, 2, 4, 8, 16, 32, 64):
        before = list(result)
        result = [f32(before[i ^ distance] - before[i]) if i & distance
                  else f32(before[i] + before[i ^ distance]) for i in range(128)]
    return result


def register_hadamard(values, adjacent):
    # Simulate independently executing lanes and registers, not a flattened XOR
    # implementation. This detects reordered butterflies and wrong lane signs.
    physical = (lambda lane, j: 4*lane+j) if adjacent else (lambda lane, j: lane+32*j)
    x = [[values[physical(lane, j)] for j in range(4)] for lane in range(32)]

    def registers(distance):
        before = [list(row) for row in x]
        for lane in range(32):
            for j in range(4):
                x[lane][j] = f32(before[lane][j ^ distance] - before[lane][j]) if j & distance \
                    else f32(before[lane][j] + before[lane][j ^ distance])

    def lanes(distance):
        before = [list(row) for row in x]
        for lane in range(32):
            for j in range(4):
                x[lane][j] = f32(before[lane ^ distance][j] - before[lane][j]) if lane & distance \
                    else f32(before[lane][j] + before[lane ^ distance][j])

    if adjacent:
        registers(1)
        registers(2)
    for distance in (1, 2, 4, 8, 16):
        lanes(distance)
    if not adjacent:
        registers(1)
        registers(2)
    result = [None]*128
    for lane in range(32):
        for j in range(4):
            result[physical(lane, j)] = x[lane][j]
    return result


def warp_sum(values):
    x = list(values)
    for distance in (16, 8, 4, 2, 1):
        before = list(x)
        x = [f32(before[i] + before[i+distance if i+distance < 32 else i]) for i in range(32)]
    return x[0]


def runtime_namespace():
    tree = ast.parse((native.HERE/'op3_codec.py').read_text())
    keep = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ('chunk_plan', '_consumer', 'TensorTacoRS')]
    ns = dict(ARMS=('op3-r', 'op3-improved'), PROTOCOL='rtx5090-op3-direct-h128-v1',
              _LOCKS={}, _POISONED=set(), _RETAINED=[], threading=threading, nullcontext=nullcontext)
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<op3-cpu-runtime>', 'exec'), ns)
    return ns


class Tensor:
    serial = 0

    def __init__(self, shape, dtype='bf16', device=None):
        self.shape = tuple(shape) if isinstance(shape, tuple) else (shape,)
        self.dtype, self.device = dtype, device
        Tensor.serial += 1
        self.pointer = Tensor.serial*8

    def is_contiguous(self):
        return True

    def reshape(self, *shape):
        return Tensor(shape, self.dtype, self.device)

    def __getitem__(self, index):
        return Tensor(self.shape[1:], self.dtype, self.device)

    def data_ptr(self):
        return self.pointer


def fake_runtime(arm):
    ns, log = runtime_namespace(), []
    device, current = SimpleNamespace(index=0), [SimpleNamespace(cuda_stream=73)]

    def exchange(rx, tx, **kwargs):
        log.append(('exchange', kwargs))
        return SimpleNamespace(wait=lambda: log.append(('wait',)))

    ns.update(torch=SimpleNamespace(bfloat16='bf16', uint8='uint8',
        device=lambda *args: device,
        empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
        empty_like=lambda x: Tensor(x.shape, x.dtype, x.device),
        cuda=SimpleNamespace(current_device=lambda: 0, current_stream=lambda _: current[0])),
        dist=SimpleNamespace(get_rank=lambda _: 0, get_world_size=lambda _: 8,
            all_to_all_single=exchange), os=SimpleNamespace(environ={'E2E_POLICY': arm}),
        load_library=lambda: None,
        encode_chunk=lambda *args, **kwargs: log.append(('encode', kwargs)),
        decode_chunk=lambda *args, **kwargs: log.append(('decode', kwargs)))
    codec = ns['TensorTacoRS'](object(), 2048, 4096)
    return ns, codec, Tensor((2048, 4096), device=device), log, current


class ArithmeticLayout(unittest.TestCase):
    def test_register_mappings_keep_every_original_float32_butterfly(self):
        rng = random.Random(509003)
        samples = [[0.0, -0.0]*64, [1.0]*128, [float(i-64) for i in range(128)]]
        samples += [[f32(rng.uniform(-1, 1)*2**rng.randint(-20, 20)) for _ in range(128)]
                    for _ in range(40)]
        for values in samples:
            expected = bits(scalar_hadamard(values))
            self.assertEqual(bits(register_hadamard(values, True)), expected)
            self.assertEqual(bits(register_hadamard(values, False)), expected)

    def test_encoder_quarter_reductions_preserve_upstream_float32_tree(self):
        rng = random.Random(509089)
        detects_wrong_tree = False
        for _ in range(80):
            values = [f32(rng.uniform(-1, 1)*2**rng.randint(-10, 10)) for _ in range(128)]
            # Original 128-thread CTA: one reduction in each physical warp.
            square = [f32(x*x) for x in values]
            original = warp_sum([warp_sum(square[w*32:(w+1)*32]) for w in range(4)] + [0.0]*28)
            # New one-warp CTA: four independent registers follow identical lanes.
            columns = [[f32(values[lane+32*j]**2) for lane in range(32)] for j in range(4)]
            proposed = warp_sum([warp_sum(column) for column in columns] + [0.0]*28)
            self.assertEqual(bits([original]), bits([proposed]))
            wrong = f32(f32(f32(warp_sum(columns[0])+warp_sum(columns[1]))+
                            warp_sum(columns[2]))+warp_sum(columns[3]))
            detects_wrong_tree |= bits([wrong]) != bits([original])
        self.assertTrue(detects_wrong_tree, 'Fixture must detect a reordered reduction')

    def test_all_rank_source_slots_and_group_addresses_are_aligned(self):
        for rank in range(8):
            sources = [(rank+offset) % 8 for offset in range(1, 8)]
            slots = [source-int(source > rank) for source in sources]
            self.assertEqual(sorted(slots), list(range(7)))
            for slot in slots:
                for group in (0, 1, 8191):
                    packet = (slot*8192+group)*136
                    self.assertEqual(packet % 8, 0)
                    self.assertEqual((packet+128) % 4, 0)
                    self.assertEqual((packet+132) % 4, 0)

    def test_original_upstream_oracle_is_built_without_modification(self):
        for name, source in native.original.source_files().items():
            self.assertEqual(native.source_files()[name], source)
        self.assertIn('op3_r_decode', native.EXPORTS)
        self.assertIn('baseline_taco_encode', native.EXPORTS)
        self.assertEqual(native.source_files()['op3.cu'],
                         '#include "op1_abc.cu"\n#include "op3_codec.cu"\n')


class Submission(unittest.TestCase):
    def test_decode_follows_nccl_completion_and_outputs_are_independent(self):
        for arm in ('op3-r', 'op3-improved'):
            ns, codec, partial, log, current = fake_runtime(arm)
            first, second = codec.reduce(partial), codec.reduce(partial)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            self.assertEqual([event[0] for event in log], ['encode', 'exchange', 'wait', 'decode']*2)
            state = codec.state()
            for key in ('calls', 'encode_calls', 'exchange_calls', 'wait_calls', 'decode_calls', 'projection_calls'):
                self.assertEqual(state[key], 2)
            self.assertEqual(state['fused_consumer_calls'], 0)
            self.assertEqual(state['caller_stream'], state['decode_stream'])
            self.assertFalse(state['pending_exchange'])

    def test_fused_consumer_passes_dynamic_tensors_and_counts_only_r(self):
        ns, codec, partial, log, current = fake_runtime('op3-r')
        for _ in range(2):
            bias = Tensor((4096,), device=codec.device)
            residual = Tensor((256, 4096), device=codec.device)
            codec.reduce(partial, bias, residual)
            self.assertIs(log[-1][1]['bias'], bias)
            self.assertIs(log[-1][1]['residual'], residual)
        self.assertEqual(codec.state()['fused_consumer_calls'], 2)
        ns, codec, partial, log, current = fake_runtime('op3-improved')
        with self.assertRaisesRegex(ValueError, 'Only the R'):
            codec.reduce(partial, Tensor((4096,), device=codec.device),
                         Tensor((256, 4096), device=codec.device))
        self.assertEqual(log, [])

    def test_incomplete_consumer_and_changed_stream_reject_before_enqueue(self):
        ns, codec, partial, log, current = fake_runtime('op3-r')
        with self.assertRaisesRegex(ValueError, 'supplied together'):
            codec.reduce(partial, Tensor((4096,), device=codec.device))
        self.assertEqual(log, [])
        codec.reduce(partial)
        log.clear()
        current[0] = SimpleNamespace(cuda_stream=88)
        with self.assertRaisesRegex(ValueError, 'original caller stream'):
            codec.reduce(partial)
        self.assertEqual(log, [])

    def test_failed_exchange_cannot_decode_or_retry_and_keeps_references(self):
        ns, codec, partial, log, current = fake_runtime('op3-r')
        def failed_wait(*args, **kwargs):
            raise RuntimeError('synthetic NCCL failure')
        ns['dist'].all_to_all_single = lambda *args, **kwargs: SimpleNamespace(wait=failed_wait)
        bias, residual = Tensor((4096,), device=codec.device), Tensor((256, 4096), device=codec.device)
        with self.assertRaisesRegex(RuntimeError, 'synthetic NCCL'):
            codec.reduce(partial, bias, residual)
        self.assertEqual([event[0] for event in log], ['encode'])
        self.assertIs(ns['_RETAINED'][0][1], partial)
        self.assertIs(ns['_RETAINED'][0][3], bias)
        self.assertIs(ns['_RETAINED'][0][4], residual)
        with self.assertRaisesRegex(ValueError, 'poisoned'):
            codec.reduce(partial)

    def test_concurrent_calls_do_not_touch_persistent_packets(self):
        ns, codec, partial, log, current = fake_runtime('op3-r')
        codec._submit_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'Concurrent OP3'):
                codec.reduce(partial)
            self.assertEqual(log, [])
        finally:
            codec._submit_lock.release()


if __name__ == '__main__':
    unittest.main()
