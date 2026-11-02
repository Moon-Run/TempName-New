"""CPU protocol/oracle regression tests; GPU gates live in the distributed probe."""
import math
import random
import struct
import unittest

import open_compress_codec as codec


class Int8ReferenceTests(unittest.TestCase):
    def test_zero_scale_and_payload(self):
        values, scale = codec.reference_block([0.0] * 128)
        self.assertEqual(scale, 1.0)
        self.assertEqual(values, [0] * 128)
        self.assertEqual(codec.reference_block_roundtrip([0.0] * 128), [0.0] * 128)

    def test_nearest_even_and_symmetric_extrema(self):
        source = [127., -127., 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5] + [0.] * 118
        values, scale = codec.reference_block(source)
        self.assertEqual(scale, 1.0)
        self.assertEqual(values[:10], [127, -127, 0, 2, 2, 4, 0, -2, -2, -4])
        self.assertNotIn(-128, values)

    def test_quantization_bound_for_normal_and_outlier_blocks(self):
        rng = random.Random(7331)
        for magnitude in (0.001, 1, 1024):
            for outlier in (False, True):
                values = [codec.bf16_round(rng.uniform(-1, 1) * magnitude) for _ in range(128)]
                if outlier:
                    values[17] = codec.bf16_round(16 * magnitude)
                quantized, scale = codec.reference_block(values)
                for value, q in zip(values, quantized):
                    self.assertLessEqual(abs(value - q * scale), scale * 0.50001)
                    self.assertGreaterEqual(q, -127)
                    self.assertLessEqual(q, 127)

    def test_bf16_reference_ties_and_source_order(self):
        self.assertEqual(codec.bf16_round(1 + 1 / 256), 1.)
        self.assertEqual(codec.bf16_round(1 + 3 / 256), 1 + 2 / 128)
        # A BF16 sum must round after each source, not only after the FP32 sum.
        values = [256., 1., -256., 0., 0., 0., 0., 0.]
        accum = 0.
        for value in values:
            accum = codec.bf16_round(accum + value)
        self.assertEqual(accum, 0.)
        self.assertEqual(codec.bf16_round(sum(values)), 1.)

    def test_dequantization_rounding(self):
        values = [codec.bf16_round(i / 11) for i in range(-64, 64)]
        q, scale = codec.reference_block(values)
        actual = codec.reference_block_roundtrip(values)
        self.assertTrue(all(math.isfinite(x) for x in actual))
        for quantized, value in zip(q, actual):
            expected = codec.bf16_round(struct.unpack('<f', struct.pack('<f', quantized * scale))[0])
            self.assertEqual(value, expected)

    def test_nonfinite_and_partial_groups_rejected_by_oracle(self):
        for invalid in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                codec.reference_block([invalid] + [0.] * 127)
        for length in (0, 127, 129):
            with self.assertRaises(ValueError):
                codec.reference_block([0.] * length)

    def test_self_zero_and_all_directed_links(self):
        links = set()
        for rank in range(8):
            splits = codec.splits_for_rank(rank)
            self.assertEqual(splits[rank], 0)
            self.assertEqual(sum(splits), codec.WIRE_BYTES)
            for destination, size in enumerate(splits):
                if size:
                    self.assertEqual(size, codec.splits_for_rank(destination)[rank])
                    links.add((rank, destination))
        self.assertEqual(len(links), 56)

    def test_wire_budget_includes_scale_bytes(self):
        self.assertEqual(codec.PEER_BYTES, 1081344)
        self.assertEqual(codec.WIRE_BYTES, 7569408)
        self.assertEqual(codec.BF16_REMOTE_BYTES, 14680064)
        self.assertEqual(codec.WIRE_BYTES / codec.BF16_REMOTE_BYTES, 0.515625)
        self.assertEqual(codec.PEER_BYTES % 4, 0)

    def test_rank_domain(self):
        for rank in (-1, 8, True, 1.0, None):
            with self.assertRaises(ValueError):
                codec.splits_for_rank(rank)


if __name__ == '__main__':
    unittest.main()
