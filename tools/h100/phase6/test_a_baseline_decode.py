"""CPU roundoff/cross-source layout cases for staged BF16 materialization.

These checks do not execute CUDA or replace the required eight-rank output test.
"""
import random
import struct
import unittest


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16_bits(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def value(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def rounded(x):
    return value(bf16_bits(x))


class StagedDecodeTests(unittest.TestCase):
    def test_materialization_preserves_each_source_rounding_and_ring_order(self):
        rng = random.Random(20261024)
        for rank in range(8):
            sources = [[f32(rng.uniform(-128, 128)) for _ in range(128)] for _ in range(8)]
            sources[rank] = list(map(rounded, sources[rank]))
            original = [0.0] * 128
            for offset in range(1, 9):
                src = (rank+offset) % 8
                original = [rounded(f32(a+rounded(b))) for a, b in zip(original, sources[src])]
            # Actual scratch is a BF16 byte representation, in cyclic source order.
            scratch = b''.join(struct.pack('<H', bf16_bits(x))
                for offset in range(1, 8) for x in sources[(rank+offset) % 8])
            staged = [0.0]*128
            for offset in range(7):
                row = [value(struct.unpack_from('<H', scratch, 2*(offset*128+j))[0]) for j in range(128)]
                staged = [rounded(f32(a+b)) for a, b in zip(staged, row)]
            staged = [rounded(f32(a+b)) for a, b in zip(staged, sources[rank])]
            self.assertEqual(list(map(bf16_bits, original)), list(map(bf16_bits, staged)))

    def test_source_reordering_or_final_only_rounding_is_not_equivalent(self):
        # Individually rounded additions at 1.0 tie to even; adding the two
        # increments first crosses one BF16 ULP. This guards against a parallel sum.
        increments = [1.0, 1/256, 1/256]
        accum = 0.0
        for x in increments:
            accum = rounded(f32(accum+rounded(x)))
        final_only = rounded(f32(sum(increments)))
        reordered = rounded(f32(rounded(1/256+1/256)+1.0))
        self.assertEqual(accum, 1.0)
        self.assertNotEqual(accum, final_only)
        self.assertNotEqual(accum, reordered)

    def test_remote_scratch_slots_exclude_local_and_cover_all_sources(self):
        for rank in range(8):
            sources = [(rank+slot+1) % 8 for slot in range(7)]
            self.assertEqual(set(sources), set(range(8))-{rank})
            self.assertEqual(len(set(sources)), 7)
            self.assertEqual((sources+[rank]), [(rank+i)%8 for i in range(1, 9)])


if __name__ == '__main__':
    unittest.main()
