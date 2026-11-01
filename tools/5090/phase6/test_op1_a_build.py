"""OP1 source provenance and genuine CTA mapping contracts (CPU-only)."""
from pathlib import Path
import tempfile
import unittest

import op1_a_build as build
import op1_a_swizzle as swizzle


class SwizzleTest(unittest.TestCase):
    def test_rank_interleave_is_complete_tile_bijection(self):
        expected = {(m, n) for m in range(16) for n in range(32)}
        for rank in range(8):
            mapping = swizzle.mapping(rank)
            self.assertEqual(len(mapping), 512)
            self.assertEqual(set(mapping), expected)
            # Each successive group of eight visits every peer once; every
            # peer's own 64 tiles keep their original row/column order.
            for start in range(0, 512, 8):
                self.assertEqual([m // 2 for m, _ in mapping[start:start+8]],
                                 [(rank + peer) % 8 for peer in range(8)])
            for destination in range(8):
                self.assertEqual([(m % 2, n) for m, n in mapping if m // 2 == destination],
                                 [(m, n) for m in range(2) for n in range(32)])

    def test_bad_ranks_fail(self):
        for rank in (-1, 8, True, 1.0, '0'):
            with self.assertRaises(ValueError):
                swizzle.mapping(rank)

    def test_sorted_transform_is_confined_to_mapping(self):
        updates = swizzle.overrides(build.REPO)
        self.assertEqual(set(updates), {swizzle.HEADER, swizzle.PROBE, swizzle.KERNEL,
                                       'src/cuda/version.ld'})
        self.assertTrue(swizzle.identity()['may_change_streamk_reduction_assignment'])
        header = updates[swizzle.HEADER]
        self.assertIn('LOCAL_RANK', header)
        self.assertIn('problem.m() == 2048', header)
        self.assertIn('Base::get_tile_offset(tile_index)', header)
        probe = updates[swizzle.PROBE]
        self.assertIn('GemmDevice_0', probe)
        self.assertIn('Op1MappingKernel::Params', probe)
        self.assertIn('op.to_gemm_args', probe)
        self.assertIn('std::is_same_v', probe)
        kernel = updates[swizzle.KERNEL]
        # A source-level round-trip checks the full body outside the type
        # substitution, including the original GemmOnly arguments and EVT.
        kernel = kernel.replace('#include "comm_none/op1_a_swizzle.hpp"\n', '')
        start = kernel.index('    using AlignmentC = ')
        stop = kernel.index('    return this->default_gemm_kernel(SortedParams{});', start)
        kernel = kernel[:start] + '    return this->default_gemm_kernel(params);' + kernel[
            stop+len('    return this->default_gemm_kernel(SortedParams{});'):]
        self.assertEqual(kernel, (build.REPO/swizzle.KERNEL).read_text())

    def test_generated_kernel_name_proves_active_swizzle(self):
        original = ('struct flux_bf16_sm89: public GemmKernel_0 {};\n'
                    'using GemmDevice_0 = Device<GemmMetaT, HParams, flux_bf16_sm89>;\n')
        transformed = swizzle.registration_source(original)
        self.assertIn('struct flux_bf16_sm89_Op1AInterleavedSwizzle:', transformed)
        self.assertEqual(transformed.count('flux_bf16_sm89_Op1AInterleavedSwizzle'), 2)
        self.assertEqual(swizzle.original_registration(transformed), original)
        with self.assertRaisesRegex(ValueError, 'untouched'):
            swizzle.registration_source(transformed)
        with self.assertRaisesRegex(ValueError, 'Missing OP1'):
            swizzle.original_registration(transformed + '\n')


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='rtx5090-op1-source-test-')
        cls.base = Path(cls.temporary.name)/'baseline'
        cls.root = Path(cls.temporary.name)/'op1'
        build.baseline.prepare(cls.base)
        cls.parent_digest = build.sha(cls.base/'baseline-build.json')
        build.prepare(cls.root, cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_original_baseline_remains_read_only(self):
        record = build.verify(self.root)
        self.assertEqual(build.sha(self.base/'baseline-build.json'), self.parent_digest)
        self.assertEqual(build.base_root(self.root), self.base)
        self.assertEqual(build.native_root(self.root, 'flux-basic'),
                         build.baseline.native_root(self.base, 'flux-basic'))
        self.assertEqual(set(record['variants']), set(build.NEW_ARMS))
        self.assertFalse(record['gpu_validated'])
        self.assertEqual(record['actual_arch'], 'sm120')

    def test_changed_source_rejected(self):
        path = build.native_root(self.root, 'flux-basic-interleaved')/swizzle.HEADER
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n// drift\n')
            with self.assertRaisesRegex(ValueError, 'Missing/changed OP1 build input'):
                build.verify(self.root)
        finally:
            path.write_bytes(original)

    def test_changed_parent_manifest_rejected(self):
        path = self.base/'baseline-build.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Parent nine-arm build identity changed'):
                build.verify(self.root)
        finally:
            path.write_bytes(original)

    def test_prepared_build_cannot_execute(self):
        with self.assertRaisesRegex(ValueError, 'Build incomplete'):
            build.verify(self.root, built=True)

    def test_invalid_arm_and_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown OP1 arm'):
            build.native_root(self.root, 'ma11')
        with self.assertRaisesRegex(ValueError, 'fresh OP1 build'):
            build.prepare(self.root, self.base)


if __name__ == '__main__':
    unittest.main()
