"""CPU model of source-parallel staging and the independent scalar ring oracle.

These tests cover routing, buffer coverage, stage rounding and source ordering.
They do not substitute for executing the actual S/D/A kernels on eight H100s.
"""
import random
import struct
import unittest

from test_ma2_decode import (
    adjacent_stages, bf16, bits, decode, f32, pack_payload, packed_decode,
    packed_local, scalar_ring, scalar_stages, unpack_local,
)


def from_bf16(word):
    return struct.unpack('<f', struct.pack('<I', word << 16))[0]


def serial_ring(packets, local, rank):
    accumulator = [bf16(0.0)] * 128
    events = []
    for offset in range(1, 8):
        source = (rank + offset) & 7
        events.append(('load', source))
        values = packed_decode(packets[source])
        events.append(('fold', source))
        accumulator = [bf16(f32(a + bf16(b))) for a, b in zip(accumulator, values)]
    events.append(('local', rank))
    return [bf16(f32(a + b)) for a, b in zip(accumulator, local)], events


def parallel_ring(packets, local, rank, source_work):
    shared = [[[None] * 32 for _ in range(4)] for _ in range(8)]
    local = unpack_local(packed_local(local))
    for warp, round_index in source_work:
        offset = warp + 1 + 4 * round_index
        if offset == 8:
            values = local
        else:
            source = (rank + offset) & 7
            if source == rank:
                raise AssertionError('Reserved local FP8 packet was read')
            values = packed_decode(packets[source])
        for lane in range(32):
            for j in range(4):
                if shared[offset - 1][j][lane] is not None:
                    raise AssertionError('Duplicate source/coefficient write')
                shared[offset - 1][j][lane] = bits(bf16(values[4 * lane + j])) >> 16
    if any(value is None for slot in shared for row in slot for value in row):
        raise AssertionError('A source/coefficient was not published before the barrier')
    result = [None] * 128
    for warp in range(4):
        for lane in range(32):
            accumulator = bf16(0.0)
            for offset in range(8):
                value = from_bf16(shared[offset][warp][lane])
                accumulator = bf16(f32(accumulator + value))
            result[4 * lane + warp] = accumulator
    return result


class DecodeTests(unittest.TestCase):
    def assertBitsEqual(self, actual, expected):
        self.assertEqual([bits(v) for v in actual], [bits(v) for v in expected])

    def test_all_ranks_generations_and_decode_orders_preserve_scalar_ring(self):
        rng = random.Random(20261022)
        for generation in range(6):
            for rank in range(8):
                local = [bf16(rng.uniform(-20, 20)) for _ in range(128)]
                packets, values = [None] * 8, [None] * 8
                for source in range(8):
                    if source == rank:
                        values[source] = local
                        continue
                    payload = [rng.randrange(127) | (rng.randrange(2) << 7) for _ in range(128)]
                    quant, adaptive = f32(.001 * (generation + source + 1)), f32(1.0 + source)
                    packets[source] = pack_payload(payload), quant, adaptive
                    values[source] = decode(payload, quant, adaptive, scalar_stages)
                expected = scalar_ring(values, rank)
                actual, events = serial_ring(packets, local, rank)
                self.assertBitsEqual(actual, expected)
                self.assertEqual(events, [(event, (rank + offset) & 7)
                    for offset in range(1, 8) for event in ('load', 'fold')] + [('local', rank)])
                work = [(warp, round_index) for warp in range(4) for round_index in range(2)]
                self.assertBitsEqual(parallel_ring(packets, local, rank, work), expected)
                rng.shuffle(work)
                self.assertBitsEqual(parallel_ring(packets, local, rank, work), expected)

    def test_each_source_and_coefficient_has_unique_shared_producer_and_consumer(self):
        produced = set()
        for warp in range(4):
            for round_index in range(2):
                offset = warp + 1 + 4 * round_index
                for j in range(4):
                    addresses = [(offset - 1) * 128 + j * 32 + lane for lane in range(32)]
                    self.assertEqual({address % 32 for address in addresses}, set(range(32)))
                    produced.update(addresses)
        self.assertEqual(produced, set(range(8 * 128)))
        consumed = {(offset * 128 + warp * 32 + lane)
                    for offset in range(8) for warp in range(4) for lane in range(32)}
        self.assertEqual(consumed, produced)
        self.assertEqual({4 * lane + warp for warp in range(4) for lane in range(32)}, set(range(128)))
        self.assertEqual({block * 4 + warp for block in range(2048) for warp in range(4)},
                         set(range(8192)))

    def test_missing_or_duplicate_source_work_is_detected(self):
        packets = [(pack_payload([0] * 128), 1.0, 1.0)] * 8
        work = [(warp, round_index) for warp in range(4) for round_index in range(2)]
        with self.assertRaisesRegex(AssertionError, 'not published'):
            parallel_ring(packets, [0.0] * 128, 0, work[:-1])
        with self.assertRaisesRegex(AssertionError, 'Duplicate'):
            parallel_ring(packets, [0.0] * 128, 0, work + work[:1])

    def test_original_hadamard_stage_order_is_required_for_each_coefficient(self):
        rng = random.Random(2233)
        for _ in range(12):
            values = [f32(rng.uniform(-1e6, 1e6)) for _ in range(128)]
            for packed, scalar in zip(adjacent_stages(values), scalar_stages(values)):
                self.assertBitsEqual(packed, scalar)

    def test_reduction_order_and_each_BF16_rounding_remain_observable(self):
        values = [[value] * 128 for value in (256.0, 1.0, -256.0, 0, 0, 0, 0, 0)]
        expected = scalar_ring(values, 7)[0]
        self.assertEqual(expected, 0.0)
        self.assertNotEqual(expected, scalar_ring(values, 0)[0])
        self.assertNotEqual(expected, bf16(sum(source[0] for source in values)))


if __name__ == '__main__':
    unittest.main()
