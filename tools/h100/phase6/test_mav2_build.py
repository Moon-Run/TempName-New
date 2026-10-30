"""Frozen-source and provenance checks for the shared SM90 codec."""
from pathlib import Path
import tempfile
import unittest

import basic_quant_build as baseline
import mav2_build as build


class BuildTests(unittest.TestCase):
    def test_prepare_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=build.prepare(Path(tmp)/'build',source_only=True)
            m=build.verify(root)
            self.assertFalse(m['gpu_validated'])
            self.assertEqual(m['source_identity'],build.source_identity())
            self.assertEqual(m['policies'],build.POLICIES)
            self.assertEqual((root/'baseline_codec.cu').read_bytes(),(build.HERE/'baseline_codec.cu').read_bytes())
            self.assertEqual((root/'basic_quant_receiver.cuh').read_bytes(),(build.HERE/'basic_quant_receiver.cuh').read_bytes())

    def test_h8_device_math_is_not_rewritten(self):
        original=baseline.kernel_source()
        start=original.index('    float val_sq = val * val;')
        end=original.index('template<typename T, int GROUP_SIZE>',start)
        generated=build.encoder_source()
        self.assertEqual(generated[generated.index('    float val_sq = val * val;'):],original[start:end])

    def test_modified_kernel_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=build.prepare(Path(tmp)/'build',source_only=True)
            (root/'mav2_encoder.cuh').write_text('// substitute')
            with self.assertRaises(ValueError): build.verify(root)

    def test_direct_branch_returns_before_quality_or_transform(self):
        source=(build.HERE/'mav2_codec.cuh').read_text()
        start=source.index('  if (mode == 1) {')
        end=source.index('  const __half_raw hr',start)
        branch=source[start:end]
        self.assertIn('return true;',branch)
        for operation in ('block_sum(', 'block_max(', '__nv_cvt_fp8_to_halfraw(', 'kRelativeL2', 'mse'):
            self.assertNotIn(operation,branch)

    def test_d8_identity_cannot_enable_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=build.prepare(Path(tmp)/'build',source_only=True)
            m=build.read(root/'manifest.json'); m['policies']['d8']['fallback']='H8'
            build.write(root/'manifest.json',m)
            with self.assertRaises(ValueError): build.verify(root)

    def test_prepared_build_is_not_gpu_or_binary_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=build.prepare(Path(tmp)/'build',source_only=True)
            with self.assertRaises(ValueError): build.verify(root,built=True)
            with self.assertRaises(ValueError): build.compile_build(root,'/missing/cuda')


if __name__ == '__main__': unittest.main()
