"""Independent closure-oracle checks. Optional Torch tests run on CPU only.

Packet tests exercise the full fixed packet layout with fake distributed input
gathering; they neither launch CUDA nor claim device/compiler validation.
"""
import hashlib
import math
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import arrival_op4_v2_plan as plan
import arrival_op4_v2_reference as reference

try:
    import torch
except ImportError:
    torch = None


class PairedSampleTests(unittest.TestCase):
    def test_same_sample_is_zero_and_gate_detects_changed_output(self):
        sample = [0.0, 1.0, -2.0, 3.0]
        self.assertEqual(reference.paired_sample_error(sample, sample), 0.0)
        self.assertEqual(reference.paired_sample_error([0.0] * 4, [0.0] * 4), 0.0)
        self.assertLess(reference.paired_sample_error(sample, [v * 1.001 for v in sample]), .005)
        self.assertGreater(reference.paired_sample_error(sample, [v * 1.02 for v in sample]), .005)
        self.assertAlmostEqual(reference.paired_sample_error(sample, [-v for v in sample]), 2.0)

    def test_rejects_missing_nonfinite_or_unmatched_samples(self):
        for first, second in (([], []), ([1.], [1., 2.]), ([math.nan], [1.]),
                              ([1.], [math.inf]), ([math.inf], [math.inf])):
            with self.subTest(first=first, second=second):
                with self.assertRaises(ValueError):
                    reference.paired_sample_error(first, second)


@unittest.skipIf(torch is None, 'Torch is optional in system Python; run separately in the Torch environment')
class TorchReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.h = reference.hadamard_matrix(torch, 'cpu')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_parity_matrix_is_orthogonal_symmetric_and_involutory(self):
        self.assertEqual(self.h.dtype, torch.float64)
        self.assertEqual(self.h.device.type, 'cpu')
        self.assertTrue(torch.equal(self.h, self.h.t()))
        torch.testing.assert_close(self.h @ self.h,
                                   torch.eye(128, dtype=torch.float64), rtol=0, atol=3e-15)
        vector = torch.arange(128, dtype=torch.float64) - 63.5
        torch.testing.assert_close((vector @ self.h) @ self.h, vector, rtol=2e-14, atol=2e-13)
        self.assertEqual(float(self.h[5, 3]), -1 / math.sqrt(128))
        self.assertEqual(float(self.h[5, 5]), 1 / math.sqrt(128))

    def test_distinct_source_scales_move_inside_transform_in_fp64(self):
        generator = torch.Generator().manual_seed(451)
        payload = torch.randint(-96, 97, (7, 5, 128), generator=generator).to(torch.float8_e4m3fn)
        quant = torch.logspace(-8, 5, 35).reshape(7, 5)
        adaptive = torch.logspace(-3, 3, 35).reshape(7, 5).flip(0)
        separate = reference.dense_decode(torch, payload, quant, adaptive, self.h).sum(0)
        coefficients = payload.double() * (quant.double() / adaptive.double()).unsqueeze(-1)
        merged = coefficients.sum(0) @ self.h
        torch.testing.assert_close(merged, separate, rtol=2e-12, atol=2e-8)
        wrong_common_scale = (payload.double().sum(0) *
                              (quant[0].double() / adaptive[0].double()).unsqueeze(-1)) @ self.h
        self.assertGreater(float((wrong_common_scale - separate).norm()), 1)

    def test_per_source_and_merged_fp32_arithmetic_need_not_be_bitwise_equal(self):
        generator = torch.Generator().manual_seed(47)
        payload = torch.randint(-96, 97, (7, 128), generator=generator).to(torch.float8_e4m3fn)
        quant = torch.tensor([.001, .7, 10., .05, 2., .01, 3.])
        adaptive = torch.tensor([2., .1, .03, 5., 30., .7, .01])
        scaled = payload.float() * (quant / adaptive).unsqueeze(-1)
        physical = torch.zeros(128)
        coefficients = torch.zeros(128)
        for source in range(7):
            physical += scaled[source] @ self.h.float()
            coefficients += scaled[source]
        merged = coefficients @ self.h.float()
        self.assertFalse(torch.equal(physical, merged))
        oracle = reference.dense_decode(torch, payload, quant, adaptive, self.h).sum(0)
        for actual in (physical, merged):
            self.assertLess(float((actual.double() - oracle).norm() / oracle.norm()), 3e-6)

    def test_old_bf16_rounding_and_final_bf16_rounding_have_counterexamples(self):
        values = torch.tensor([256., 1., -256.])
        old = torch.zeros((), dtype=torch.bfloat16)
        for value in values:
            old = (old + value.to(torch.bfloat16)).to(torch.bfloat16)
        final_only = values.sum().to(torch.bfloat16)
        self.assertEqual(float(old), 0.)
        self.assertEqual(float(final_only), 1.)
        decoded = torch.tensor([1.00390625, -1.0])
        self.assertEqual(float(decoded.to(torch.bfloat16).float().sum()), 0.)
        self.assertEqual(float(decoded.sum().to(torch.bfloat16)), .00390625)

    def test_extreme_legal_scales_and_exact_cancellation_remain_finite(self):
        values = torch.tensor([448., -448., 0., 1.]).repeat(32)
        payload = torch.stack([values, -values]).to(torch.float8_e4m3fn)
        for quant_value, adaptive_value in ((1e-12, 1e3), (1e6, 1e-3)):
            quant = torch.full((2,), quant_value)
            adaptive = torch.full((2,), adaptive_value)
            decoded = reference.dense_decode(torch, payload, quant, adaptive, self.h)
            self.assertTrue(bool(torch.isfinite(decoded).all()))
            torch.testing.assert_close(decoded.sum(0), torch.zeros(128, dtype=torch.float64), rtol=0, atol=0)
        tiny = torch.zeros((1, 128)).to(torch.float8_e4m3fn)
        decoded = reference.dense_decode(torch, tiny, torch.tensor([1e-12]), torch.tensor([1e3]), self.h)
        self.assertEqual(int(torch.count_nonzero(decoded)), 0)

    def test_rejects_fp8_nan_and_nonfinite_or_out_of_domain_scales(self):
        payload = torch.ones((1, 128)).to(torch.float8_e4m3fn)
        quant, adaptive = torch.ones(1), torch.ones(1)
        raw = payload.view(torch.uint8).clone()
        for code in (127, 255):
            raw[0, 17] = code
            with self.assertRaisesRegex(ValueError, 'Nonfinite FP8'):
                reference.dense_decode(torch, raw.view(torch.float8_e4m3fn), quant, adaptive, self.h)
        for invalid in (float('nan'), float('inf'), -1., 0., 1e-13, 1e7):
            with self.subTest(quant=invalid), self.assertRaises(ValueError):
                reference.dense_decode(torch, payload, torch.tensor([invalid]), adaptive, self.h)
        for invalid in (float('nan'), float('inf'), -1., 0., 1e-4, 1e4):
            with self.subTest(adaptive=invalid), self.assertRaises(ValueError):
                reference.dense_decode(torch, payload, quant, torch.tensor([invalid]), self.h)

    def test_group_selectors_expand_the_physical_tile_mask_without_local_sources(self):
        for arm in plan.ARMS:
            for phase in (0, 3, 7):
                for source in range(8):
                    for destination in range(8):
                        flags = reference.group_selector(torch, arm, phase, source, destination, 'cpu')
                        self.assertEqual(tuple(flags.shape), (256, 32))
                        expected = torch.tensor(plan.mask(arm, phase)[source]
                                                [destination * 64:(destination + 1) * 64],
                                                dtype=torch.bool).reshape(2, 32)
                        self.assertTrue(torch.equal(flags[:128], expected[0].expand(128, 32)))
                        self.assertTrue(torch.equal(flags[128:], expected[1].expand(128, 32)))
                        self.assertEqual(int(flags.sum()), 0 if source == destination else 1024)

    def test_check_taco_output_accepts_tensor_or_list_mask_and_rejects_wrong_mask(self):
        class ReachedGather(Exception):
            pass
        # The sentinel stops before distributed work; it verifies the actual
        # public mask-validation branch accepts CPU Tensor input unambiguously.
        for arm in ('SP', 'CM'):
            with mock.patch.dict(os.environ, H100_CLOSURE_ARM=arm, H100_CLOSURE_PHASE='3'), \
                 mock.patch.dict('sys.modules', {'reference': SimpleNamespace(encode=None)}), \
                 mock.patch.object(reference, '_parts', side_effect=ReachedGather) as gather:
                values = plan.mask(arm, 3)
                for supplied in (values, torch.tensor(values, dtype=torch.uint8), None):
                    with self.assertRaises(ReachedGather):
                        reference.check_taco_output(None, None, None, mask=supplied)
                bad = torch.tensor(values, dtype=torch.uint8)
                bad[0, 0] ^= 1
                count = gather.call_count
                with self.assertRaisesRegex(ValueError, 'External mask differs'):
                    reference.check_taco_output(None, None, None, mask=bad)
                self.assertEqual(gather.call_count, count)

    def test_metrics_reject_nonfinite_and_use_independent_reference_errors(self):
        bf16 = torch.tensor([1., 2., 3.], dtype=torch.bfloat16)
        result = reference.normalized_metrics(torch, bf16, bf16.double(), bf16)
        self.assertEqual(result['relative_l2'], 0.)
        self.assertEqual(result['protocol_relative_l2'], 0.)
        result = reference.normalized_metrics(torch, bf16, bf16.double() + 1, bf16)
        self.assertEqual(result['relative_l2'], 0.)
        self.assertGreater(result['protocol_relative_l2'], 0.)
        for location in range(3):
            args = [bf16.clone(), bf16.clone(), bf16.clone()]
            args[location][0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                reference.normalized_metrics(torch, *args)

    def packet_case(self, arm, phase=3, rank=2):
        groups, source_bytes = 8192, 8192 * 136
        # Deliberately poison ALL unused payload and metadata bytes. A checker
        # that decodes the unselected area will encounter FP8 or FP32 NaNs.
        packet = torch.full((8 * source_bytes,), 255, dtype=torch.uint8)
        pieces = [torch.full((256, 4096), (source + 1) / 16,
                             dtype=torch.bfloat16) for source in range(8)]
        expected = torch.zeros((256, 32, 128), dtype=torch.float64)
        encoded = {}
        scale_grid = torch.linspace(.25, 2., groups).reshape(256, 32)
        for source in range(8):
            flags = reference.group_selector(torch, arm, phase, source, rank, 'cpu')
            contribution = pieces[source].double().reshape(256, 32, 128)
            if bool(flags.any()):
                payload = torch.zeros((int(flags.sum()), 128))
                basis = source * 17
                payload[:, basis] = 2.
                payload = payload.to(torch.float8_e4m3fn)
                quant = scale_grid[flags] * (source + 1)
                adaptive = torch.full_like(quant, source + 2)
                slot = packet[source * source_bytes:(source + 1) * source_bytes]
                slot[:groups * 128].reshape(256, 32, 128)[flags] = payload.view(torch.uint8)
                slot[groups * 128:groups * 132].view(torch.float32).reshape(256, 32)[flags] = quant
                slot[groups * 132:groups * 136].view(torch.float32).reshape(256, 32)[flags] = adaptive
                signs = torch.tensor([(-1. if (basis & col).bit_count() % 2 else 1.)
                                      for col in range(128)], dtype=torch.float64)
                contribution[flags] = ((2. * quant.double() / adaptive.double()).unsqueeze(-1)
                                       * signs / math.sqrt(128))
                encoded[source] = (payload, quant, adaptive)
            expected += contribution
        digest = hashlib.sha256()
        for offset in range(1, 9):
            for tensor in encoded.get((rank + offset) % 8, ()):
                digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())

        class Part:
            def __init__(self, piece):
                self.piece = piece
            def __getitem__(self, item):
                if item != slice(rank * 256, (rank + 1) * 256):
                    raise AssertionError('Wrong destination slice')
                return self.piece

        op = SimpleNamespace(peers=[packet] * 8)
        parts = [Part(piece) for piece in pieces]
        return op, parts, expected.reshape(256, 4096).to(torch.bfloat16), digest.hexdigest()

    def test_real_packet_offsets_fp8_views_and_only_selected_bytes_are_consumed(self):
        for arm in ('SP', 'CM'):
            op, parts, actual, digest = self.packet_case(arm)
            with mock.patch.dict(os.environ, H100_CLOSURE_ARM=arm, H100_CLOSURE_PHASE='3'), \
                 mock.patch.object(reference, '_parts', return_value=(2, parts)):
                result = reference.check_actual_packets(op, None, actual, None)
            self.assertTrue(result['passed'])
            self.assertLess(result['relative_l2'], .005)
            self.assertEqual(result['actual_packet_sha256'], digest)
            self.assertEqual(len(result['sample']), 512)
            self.assertEqual(result['sample_pattern'], 'multiplicative-512-v1')

    def test_actual_packet_oracle_rejects_selected_nan_and_wrong_output(self):
        op, parts, actual, _ = self.packet_case('CM')
        with mock.patch.dict(os.environ, H100_CLOSURE_ARM='CM', H100_CLOSURE_PHASE='3'), \
             mock.patch.object(reference, '_parts', return_value=(2, parts)):
            with self.assertRaisesRegex(ValueError, 'Actual packet FP64 oracle failed'):
                reference.check_actual_packets(op, None, actual + 1., None)
            flags = reference.group_selector(torch, 'CM', 3, 0, 2, 'cpu')
            row, col = torch.nonzero(flags, as_tuple=False)[0].tolist()
            op.peers[2][(row * 32 + col) * 128 + 17] = 127
            with self.assertRaisesRegex(ValueError, 'Nonfinite FP8'):
                reference.check_actual_packets(op, None, actual, None)


if __name__ == '__main__':
    unittest.main()
