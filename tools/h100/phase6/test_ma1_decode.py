"""CPU arithmetic and address/ordering checks for MA1's exact TP8 decoder.

This tests the independent scalar model against the interleaved four-register
and source-prefetch schedules. It does not execute CUDA or establish bitwise
equivalence of compiled GPU instructions; the eight-rank preflight must do so.
"""
import math
import random
import unittest

try:
    from tools.h100.phase6.test_arrival_op3_v2_decode import (
        bf16, bits, decode, e4m3, f32, scalar_ring, scalar_stages,
    )
except ModuleNotFoundError:
    from test_arrival_op3_v2_decode import (
        bf16, bits, decode, e4m3, f32, scalar_ring, scalar_stages,
    )


def interleaved_stages(values):
    """Simulate lanes/registers, with a fresh snapshot at every shuffle level."""
    registers = [[values[lane + 32 * quarter] for quarter in range(4)]
                 for lane in range(32)]
    result = []

    def snapshot():
        return [registers[lane][quarter] for quarter in range(4) for lane in range(32)]

    for distance in (1, 2, 4, 8, 16):
        previous = [row[:] for row in registers]
        for quarter in range(4):
            for lane in range(32):
                x = previous[lane][quarter]
                y = previous[lane ^ distance][quarter]
                registers[lane][quarter] = f32(y - x if lane & distance else x + y)
        result.append(snapshot())
    for lane, (a, b, c, d) in enumerate(registers):
        registers[lane] = [f32(a + b), f32(a - b), f32(c + d), f32(c - d)]
    result.append(snapshot())
    for lane, (a, b, c, d) in enumerate(registers):
        registers[lane] = [f32(a + c), f32(b + d), f32(a - c), f32(b - d)]
    result.append(snapshot())
    return result


def scheduled_ring(packets, local, rank, prefetch):
    """The independent source schedule; raw packets are immutable tuples.

    Record LOAD and FOLD separately to show that future data can be read early
    without changing source arithmetic. A stale or misrotated packet changes
    the full result, not merely the trace asserted by the ordering test.
    """
    events = []
    accumulator = [bf16(0.0)] * 128

    def load(source):
        if source == rank:
            raise AssertionError("The local packet is intentionally poison/unreadable")
        events.append(("load_remote", source))
        return packets[source]

    def fold(packet, source):
        events.append(("fold_remote", source))
        payload, quant, adaptive = packet
        values = decode(payload, quant, adaptive, interleaved_stages)
        for index in range(128):
            accumulator[index] = bf16(f32(accumulator[index] + bf16(values[index])))

    order = [(rank + offset) % 8 for offset in range(1, 8)]
    if prefetch:
        current = load(order[0])
        for index, source in enumerate(order[:-1]):
            following = load(order[index + 1])
            fold(current, source)
            current = following
        events.append(("load_local", rank))
        local_registers = local[:]
        fold(current, order[-1])
    else:
        for source in order:
            fold(load(source), source)
        events.append(("load_local", rank))
        local_registers = local[:]
    events.append(("fold_local", rank))
    for index in range(128):
        accumulator[index] = bf16(f32(accumulator[index] + local_registers[index]))
    return accumulator, events


class MA1DecodeTests(unittest.TestCase):
    def assertBitsEqual(self, left, right):
        self.assertEqual([bits(x) for x in left], [bits(x) for x in right])

    def test_every_basis_and_signed_zero_preserve_each_hadamard_stage(self):
        cases = [[-0.0 if index & 1 else 0.0 for index in range(128)]]
        for index in range(128):
            values = [0.0] * 128
            values[index] = -1.0 if index & 1 else 1.0
            cases.append(values)
        for values in cases:
            for expected, actual in zip(scalar_stages(values), interleaved_stages(values)):
                self.assertBitsEqual(expected, actual)

    def test_cancellation_and_float32_extremes_preserve_stage_rounding(self):
        rng = random.Random(20261021)
        cases = [[f32((-1.0 if index & 1 else 1.0) * (2**24 + index))
                  for index in range(128)]]
        for _ in range(40):
            cases.append([f32(math.ldexp(rng.uniform(-1, 1), rng.randrange(-100, 101)))
                          for _ in range(128)])
        for values in cases:
            for expected, actual in zip(scalar_stages(values), interleaved_stages(values)):
                self.assertBitsEqual(expected, actual)

    def test_all_finite_fp8_codes_scales_and_nonfinite_filter(self):
        finite = [code for code in range(256) if math.isfinite(e4m3(code))]
        for start in (0, 63, 126, 189):
            payload = tuple(finite[(start + index) % len(finite)] for index in range(128))
            for quant, adaptive in ((1.0, 1.0), (f32(1e-12), 0.0),
                                    (f32(1e6), f32(1e-3)), (2.0, math.nan)):
                self.assertBitsEqual(decode(payload, quant, adaptive, scalar_stages),
                                     decode(payload, quant, adaptive, interleaved_stages))
        for code in (127, 255):
            payload = [1] * 128
            payload[65] = code
            self.assertBitsEqual(decode(payload, 1.0, 1.0, interleaved_stages), [0.0] * 128)

    def test_pipeline_all_ranks_matches_scalar_with_new_packets_each_call(self):
        rng = random.Random(81021)
        for generation in range(6):
            for rank in range(8):
                packets = [None] * 8
                decoded = []
                local = [bf16(rng.uniform(-20, 20)) for _ in range(128)]
                for source in range(8):
                    if source == rank:
                        decoded.append(local)
                        continue
                    payload = tuple(rng.randrange(127) | (rng.randrange(2) << 7)
                                    for _ in range(128))
                    quant = f32(0.001 * (source + generation + 1))
                    adaptive = f32(1.0 + source)
                    packets[source] = (payload, quant, adaptive)
                    decoded.append(decode(payload, quant, adaptive, scalar_stages))
                expected = scalar_ring(decoded, rank)
                for prefetch in (False, True):
                    actual, _ = scheduled_ring(packets, local, rank, prefetch)
                    self.assertBitsEqual(expected, actual)

    def test_future_source_load_precedes_current_fold_without_reordering_folds(self):
        packet = (tuple([0] * 128), 1.0, 1.0)
        for rank in range(8):
            packets = [packet] * 8
            packets[rank] = None
            for prefetch in (False, True):
                _, events = scheduled_ring(packets, [0.0] * 128, rank, prefetch)
                order = [(rank + offset) % 8 for offset in range(1, 8)]
                self.assertEqual([source for event, source in events if event.startswith("fold")],
                                 order + [rank])
                self.assertEqual([source for event, source in events if event == "load_remote"],
                                 order)
                for current, following in zip(order, order[1:]):
                    relation = events.index(("load_remote", following)) < events.index(("fold_remote", current))
                    self.assertEqual(relation, prefetch)
                self.assertEqual(events.index(("load_local", rank)) <
                                 events.index(("fold_remote", order[-1])), prefetch)

    def test_all_groups_have_unique_contiguous_output_and_packet_addresses(self):
        groups, source_bytes, chunk = 8192, 8192 * 136, 256 * 4096
        self.assertEqual(groups * 128, chunk)
        for group in range(groups):
            output = sorted(group * 128 + lane + 32 * quarter
                            for lane in range(32) for quarter in range(4))
            self.assertEqual(output, list(range(group * 128, (group + 1) * 128)))
            quant = groups * 128 + 4 * group
            adaptive = quant + 4 * groups
            self.assertGreaterEqual(quant, groups * 128)
            self.assertLess(adaptive + 3, source_bytes)
        for rank in range(8):
            order = [(rank + offset) % 8 for offset in range(1, 8)]
            self.assertEqual(sorted(order + [rank]), list(range(8)))
            self.assertNotIn(rank, order)
            self.assertLess((rank + 1) * chunk - 1, 8 * chunk)

    def test_regression_oracle_rejects_changed_bf16_rounding_and_source_order(self):
        decoded = [[value] * 128 for value in (256.0, 1.0, -256.0, 0, 0, 0, 0, 0)]
        result = scalar_ring(decoded, 7)
        self.assertEqual(result[0], 0.0)
        self.assertNotEqual(result[0], bf16(sum(source[0] for source in decoded)))
        self.assertNotEqual(result[0], scalar_ring(decoded, 0)[0])


if __name__ == "__main__":
    unittest.main()
