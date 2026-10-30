"""Independent CPU arithmetic, layout and source-pipeline model for MA2.

Every physical coefficient is compared after each rounded H128 stage against
the original scalar model. These checks do not execute a GPU or replace the
required eight-rank bitwise-output and training-trajectory preflight.
"""
import math
import random
import struct
import unittest

try:
    from tools.h100.phase6.test_arrival_op3_v2_decode import (
        bf16, bits, decode, e4m3, f32, scalar_ring, scalar_stages,
    )
except ModuleNotFoundError:
    from test_arrival_op3_v2_decode import (
        bf16, bits, decode, e4m3, f32, scalar_ring, scalar_stages,
    )


def adjacent_stages(values):
    """Lane l owns scalar indices 4*l+j; retain physical d=1,2,...,64."""
    lanes = [list(values[4 * lane:4 * lane + 4]) for lane in range(32)]
    trace = []

    def flatten():
        return [x for lane in lanes for x in lane]

    for lane, (a, b, c, d) in enumerate(lanes):
        lanes[lane] = [f32(a + b), f32(a - b), f32(c + d), f32(c - d)]
    trace.append(flatten())
    for lane, (a, b, c, d) in enumerate(lanes):
        lanes[lane] = [f32(a + c), f32(b + d), f32(a - c), f32(b - d)]
    trace.append(flatten())
    for lane_distance in (1, 2, 4, 8, 16):
        previous = [row[:] for row in lanes]
        for lane in range(32):
            for j in range(4):
                x, y = previous[lane][j], previous[lane ^ lane_distance][j]
                lanes[lane][j] = f32(y - x if lane & lane_distance else x + y)
        trace.append(flatten())
    return trace


def pack_payload(payload):
    return tuple(sum(payload[lane * 4 + j] << (8 * j) for j in range(4))
                 for lane in range(32))


def packed_decode(packet):
    raw, quant, adaptive = packet
    payload = [(word >> (8 * j)) & 255 for word in raw for j in range(4)]
    return decode(payload, quant, adaptive, adjacent_stages)


def packed_local(values):
    return tuple(sum((bits(values[4 * lane + j]) >> 16) << (16 * j) for j in range(4))
                 for lane in range(32))


def unpack_local(words):
    return [struct.unpack('<f', struct.pack('<I', ((word >> (16 * j)) & 65535) << 16))[0]
            for word in words for j in range(4)]


def prefetched_ring(packets, local, rank):
    events, accumulator = [], [bf16(0.0)] * 128

    def load(source):
        if source == rank:
            raise AssertionError('The local FP8 slot must never be read')
        events.append(('load', source))
        return packets[source]

    def fold(packet, source):
        events.append(('fold', source))
        values = packed_decode(packet)
        for i in range(128):
            accumulator[i] = bf16(f32(accumulator[i] + bf16(values[i])))

    order = [(rank + i) % 8 for i in range(1, 8)]
    current = load(order[0])
    for i, source in enumerate(order[:-1]):
        following = load(order[i + 1])
        fold(current, source)
        current = following
    events.append(('load_local', rank))
    raw_local = packed_local(local)
    fold(current, order[-1])
    events.append(('fold_local', rank))
    for i, value in enumerate(unpack_local(raw_local)):
        accumulator[i] = bf16(f32(accumulator[i] + value))
    return unpack_local(packed_local(accumulator)), events


class MA2DecodeTests(unittest.TestCase):
    def assertBitsEqual(self, left, right):
        self.assertEqual([bits(v) for v in left], [bits(v) for v in right])

    def test_every_basis_vector_and_signed_zero_preserve_each_physical_stage(self):
        values = [[-0.0 if i & 1 else 0.0 for i in range(128)]]
        for i in range(128):
            sample = [0.0] * 128
            sample[i] = -1.0 if i & 1 else 1.0
            values.append(sample)
        for sample in values:
            for actual, expected in zip(adjacent_stages(sample), scalar_stages(sample)):
                self.assertBitsEqual(actual, expected)

    def test_float32_cancellation_subnormals_and_wide_ranges_keep_stage_rounding(self):
        rng = random.Random(22026)
        cases = [[f32((-1 if i & 1 else 1) * (2**24 + i)) for i in range(128)],
                 [f32(math.ldexp((-1 if i & 1 else 1) * (i + 1), -149)) for i in range(128)]]
        cases += [[f32(math.ldexp(rng.uniform(-1, 1), rng.randrange(-120, 121)))
                   for _ in range(128)] for _ in range(48)]
        for sample in cases:
            for actual, expected in zip(adjacent_stages(sample), scalar_stages(sample)):
                self.assertBitsEqual(actual, expected)

    def test_every_finite_fp8_pair_is_exactly_representable_as_two_fp16_values(self):
        finite = [value for value in range(256) if value & 127 != 127]
        for low in finite:
            for high in finite:
                word = low | (high << 8)
                halves = struct.pack('<ee', e4m3(word & 255), e4m3(word >> 8))
                actual = struct.unpack('<ee', halves)
                self.assertBitsEqual(actual, [e4m3(low), e4m3(high)])

    def test_byte_order_all_codes_scales_and_nonfinite_decode_filter(self):
        finite = [value for value in range(256) if value & 127 != 127]
        for offset in (0, 63, 126, 189):
            payload = [finite[(offset + i) % len(finite)] for i in range(128)]
            words = pack_payload(payload)
            for quant, adaptive in ((1.0, 1.0), (f32(1e-12), 0.0),
                                    (f32(1e6), f32(1e-3)), (2.0, math.nan)):
                self.assertBitsEqual(packed_decode((words, quant, adaptive)),
                                     decode(payload, quant, adaptive, scalar_stages))
        for invalid in (127, 255):
            payload = [1] * 128
            payload[65] = invalid
            self.assertBitsEqual(packed_decode((pack_payload(payload), 1.0, 1.0)), [0.0] * 128)

    def test_aligned_loads_and_store_cover_every_group_without_overlap(self):
        groups, source_bytes, chunk = 8192, 8192 * 136, 256 * 4096
        for group in range(groups):
            payload = [group * 128 + lane * 4 for lane in range(32)]
            output_bytes = [group * 256 + lane * 8 for lane in range(32)]
            self.assertTrue(all(address % 4 == 0 for address in payload))
            self.assertTrue(all(address % 8 == 0 for address in output_bytes))
            self.assertEqual([address + j for address in payload for j in range(4)],
                             list(range(group * 128, (group + 1) * 128)))
            self.assertEqual(output_bytes[-1] + 8, (group + 1) * 256)
            self.assertLess(groups * 128 + 4 * (group + groups) + 3, source_bytes)
        self.assertEqual(groups * 128, chunk)
        self.assertEqual(source_bytes % 8, 0)

    def test_six_calls_all_ranks_pipeline_matches_original_source_fold(self):
        rng = random.Random(8822)
        for generation in range(6):
            for rank in range(8):
                local = [bf16(rng.uniform(-20, 20)) for _ in range(128)]
                packets, decoded = [None] * 8, []
                for source in range(8):
                    if source == rank:
                        decoded.append(local)
                        continue
                    payload = [rng.randrange(127) | (rng.randrange(2) << 7) for _ in range(128)]
                    quant, adaptive = f32(.001 * (generation + source + 1)), f32(1.0 + source)
                    packets[source] = (pack_payload(payload), quant, adaptive)
                    decoded.append(decode(payload, quant, adaptive, scalar_stages))
                actual, events = prefetched_ring(packets, local, rank)
                self.assertBitsEqual(actual, scalar_ring(decoded, rank))
                order = [(rank + i) % 8 for i in range(1, 8)]
                self.assertEqual([source for event, source in events if event.startswith('fold')], order + [rank])
                for current, following in zip(order, order[1:]):
                    self.assertLess(events.index(('load', following)), events.index(('fold', current)))
                self.assertLess(events.index(('load_local', rank)), events.index(('fold', order[-1])))

    def test_packed_bf16_transport_retains_all_bits_and_signed_zero(self):
        for start in range(0, 65536, 128):
            values = [struct.unpack('<f', struct.pack('<I', word << 16))[0]
                      for word in range(start, start + 128)]
            # Signalling NaNs can be quieted by Python unpack/repack. Compare
            # against the same scalar representation, not the pre-unpack bits.
            self.assertBitsEqual(unpack_local(packed_local(values)), values)

    def test_oracle_detects_reordered_sources_and_removed_bf16_rounding(self):
        values = [[value] * 128 for value in (256.0, 1.0, -256.0, 0, 0, 0, 0, 0)]
        self.assertEqual(scalar_ring(values, 7)[0], 0.0)
        self.assertNotEqual(scalar_ring(values, 7)[0], bf16(sum(source[0] for source in values)))
        self.assertNotEqual(scalar_ring(values, 7)[0], scalar_ring(values, 0)[0])

    def test_quant_product_must_round_before_first_local_butterfly(self):
        # A real nvcc contraction hazard introduced by adjacent lane ownership:
        # scalar q*x/q*y rounded separately, versus FFMA(x,q,rounded(q*y)).
        # The float64 operations below are exact for these FP8/FP32 operands.
        quant = struct.unpack('<f', struct.pack('<I', 0x3e902d75))[0]
        x, y = e4m3(246), e4m3(233)
        separate = f32(f32(x * quant) + f32(y * quant))
        contracted = f32(x * quant + f32(y * quant))
        self.assertEqual(bits(separate), 0xc2a6b490)
        self.assertEqual(bits(contracted), 0xc2a6b48f)
        self.assertNotEqual(bits(separate), bits(contracted))


if __name__ == '__main__':
    unittest.main()
