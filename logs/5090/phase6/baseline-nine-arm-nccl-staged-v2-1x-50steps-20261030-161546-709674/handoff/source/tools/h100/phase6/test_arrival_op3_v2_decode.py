"""CPU arithmetic model for the scalar-CTA -> four-register warp mapping.

Checks every rounded H128 stage, source order and BF16 rounding, independently
of CUDA scheduling. These are NOT GPU execution, race or performance results.
The common rsqrt/division operations are modeled identically in both paths;
only GPU A/V bitwise checks establish compiler/device numerical equivalence.
"""
import math
import random
import struct
import unittest


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bf16(value):
    raw = bits(f32(value))
    if raw & 0x7f800000 == 0x7f800000:
        rounded = raw & 0xffff0000
        if raw & 0x007fffff:
            rounded |= 0x00400000
    else:
        rounded = (raw + 0x7fff + ((raw >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def e4m3(byte):
    sign = -1.0 if byte & 128 else 1.0
    exponent, fraction = (byte >> 3) & 15, byte & 7
    if exponent == 0:
        return sign * math.ldexp(float(fraction), -9)
    if exponent == 15 and fraction == 7:
        return math.nan
    return sign * math.ldexp(1.0 + fraction / 8.0, exponent - 7)


def scalar_stages(values):
    """Independent 128-element mathematical butterfly, one scalar per index."""
    current = list(values)
    trace = []
    for distance in (1, 2, 4, 8, 16, 32, 64):
        following = []
        for index, value in enumerate(current):
            partner = current[index ^ distance]
            following.append(f32(partner - value if index & distance else value + partner))
        current = following
        trace.append(current)
    return trace


def warp_stages(values):
    """32 lanes, four registers per lane, with only d<32 crossing lanes."""
    registers = [[values[lane + 32 * quarter] for quarter in range(4)]
                 for lane in range(32)]

    def flatten():
        return [registers[lane][quarter] for quarter in range(4)
                for lane in range(32)]

    trace = []
    for distance in (1, 2, 4, 8, 16):
        following = [[0.0] * 4 for _ in range(32)]
        for lane in range(32):
            for quarter in range(4):
                value = registers[lane][quarter]
                partner = registers[lane ^ distance][quarter]
                following[lane][quarter] = f32(
                    partner - value if lane & distance else value + partner)
        registers = following
        trace.append(flatten())
    for lane, (a, b, c, d) in enumerate(registers):
        registers[lane] = [f32(a + b), f32(a - b), f32(c + d), f32(c - d)]
    trace.append(flatten())
    for lane, (a, b, c, d) in enumerate(registers):
        registers[lane] = [f32(a + c), f32(b + d), f32(a - c), f32(b - d)]
    trace.append(flatten())
    return trace


def decode(payload, quant, adaptive, stages):
    values = [f32(e4m3(byte) * quant) for byte in payload]
    transformed = stages(values)[-1]
    normalization = f32(1.0 / math.sqrt(128))
    floor = f32(1e-12)
    divisor = floor if math.isnan(adaptive) else max(adaptive, floor)
    normalized = [f32(value * normalization) for value in transformed]
    return [f32((value if math.isfinite(value) else 0.0) / divisor)
            for value in normalized]


def scalar_ring(decoded, rank):
    result = []
    world = len(decoded)
    for coefficient in range(128):
        accum = bf16(0.0)
        for offset in range(1, world + 1):
            value = decoded[(rank + offset) % world][coefficient]
            accum = bf16(f32(accum + bf16(value)))
        result.append(accum)
    return result


def warp_ring(decoded, rank):
    accum = [[bf16(0.0)] * 4 for _ in range(32)]
    world = len(decoded)
    for offset in range(1, world + 1):
        source = (rank + offset) % world
        for lane in range(32):
            for quarter in range(4):
                value = decoded[source][lane + 32 * quarter]
                accum[lane][quarter] = bf16(f32(accum[lane][quarter] + bf16(value)))
    return [accum[lane][quarter] for quarter in range(4) for lane in range(32)]


class ArrivalOp3V2DecodeTests(unittest.TestCase):
    def assertFloatBitsEqual(self, left, right):
        self.assertEqual([bits(value) for value in left], [bits(value) for value in right])

    def test_every_basis_vector_has_same_index_sign_and_intermediate_stage(self):
        for index in range(128):
            values = [0.0] * 128
            values[index] = -1.0 if index & 1 else 1.0
            for reference, candidate in zip(scalar_stages(values), warp_stages(values)):
                self.assertFloatBitsEqual(reference, candidate)

    def test_each_stage_preserves_float32_rounding_for_cancellation_and_large_ranges(self):
        rng = random.Random(704)
        cases = [[-0.0 if i & 1 else 0.0 for i in range(128)],
                 [f32((-1 if i & 1 else 1) * (2**24 + i)) for i in range(128)]]
        for _ in range(80):
            cases.append([f32(math.ldexp(rng.uniform(-1, 1), rng.randrange(-100, 101)))
                          for _ in range(128)])
        for values in cases:
            for reference, candidate in zip(scalar_stages(values), warp_stages(values)):
                self.assertFloatBitsEqual(reference, candidate)

    def test_decode_all_finite_e4m3_codes_and_adaptive_floor(self):
        finite = [code for code in range(256) if code not in (127, 255)]
        for offset in (0, 63, 126, 189):
            payload = [finite[(offset + i) % len(finite)] for i in range(128)]
            for quant, adaptive in ((1.0, 1.0), (f32(1e-8), f32(1e3)),
                                    (f32(1e6), f32(1e-3)),
                                    (f32(1e-12), 0.0), (2.0, -1.0)):
                self.assertFloatBitsEqual(
                    decode(payload, quant, adaptive, scalar_stages),
                    decode(payload, quant, adaptive, warp_stages))

    def test_nonfinite_transform_is_zeroed_before_adaptive_division(self):
        for special in (127, 255):
            payload = [1] * 128
            payload[65] = special
            for adaptive in (1.0, 0.0, math.nan):
                for stages in (scalar_stages, warp_stages):
                    self.assertFloatBitsEqual(
                        decode(payload, 1.0, adaptive, stages), [0.0] * 128)
        for stages in (scalar_stages, warp_stages):
            self.assertFloatBitsEqual(
                decode([126] * 128, f32(3e38), 1.0, stages), [0.0] * 128)

    def test_all_rank_ring_outputs_keep_decode_and_each_add_bf16_rounding(self):
        rng = random.Random(151)
        payloads = [[rng.randrange(127) | (rng.randrange(2) << 7) for _ in range(128)]
                    for _ in range(8)]
        for rank in range(8):
            reference = []
            candidate = []
            for source in range(8):
                if source == rank:
                    local = [bf16(rng.uniform(-20, 20)) for _ in range(128)]
                    reference.append(local)
                    candidate.append(local)
                else:
                    quant = f32(0.001 * (source + 1))
                    adaptive = f32(1.0 + source)
                    reference.append(decode(payloads[source], quant, adaptive, scalar_stages))
                    candidate.append(decode(payloads[source], quant, adaptive, warp_stages))
            self.assertFloatBitsEqual(scalar_ring(reference, rank), warp_ring(candidate, rank))

    def test_delayed_bf16_rounding_and_reordered_sources_are_detectably_different(self):
        # At rank 7 the source order is 0,...,7. BF16 cancellation here loses
        # the +1 before -256; a single final rounding or reordered ring keeps it.
        decoded = [[value] * 128 for value in (256.0, 1.0, -256.0, 0, 0, 0, 0, 0)]
        reference = scalar_ring(decoded, 7)
        self.assertEqual(reference[0], 0.0)
        self.assertNotEqual(reference[0], bf16(sum(source[0] for source in decoded)))
        self.assertNotEqual(reference[0], scalar_ring(decoded, 0)[0])
        # Decoded sources must also round to BF16 before the add, even when
        # accumulating in BF16. 1.00390625 is a tie, which first rounds to 1.
        decoded = [[value] * 128 for value in (1.00390625, -1.0, 0, 0, 0, 0, 0, 0)]
        reference = scalar_ring(decoded, 0)
        without_decode_round = bf16(0.0)
        for offset in range(1, 9):
            without_decode_round = bf16(f32(
                without_decode_round + decoded[offset % 8][0]))
        self.assertNotEqual(reference[0], without_decode_round)


if __name__ == '__main__':
    unittest.main()
