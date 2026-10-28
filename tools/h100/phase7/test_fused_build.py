"""CPU source derivation checks; temporary artifacts are not GPU evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import fused_build as builder
import fused_plan
from fused_support import REPO
from common import read, sha


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='phase7-fused-CPU-SOURCE-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build', source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_compute_codec_ipc_and_mask_are_identical(self):
        record = builder.verify(self.root)
        plans = read(self.root / 'plans.json')
        common = None
        for order in builder.ORDERS:
            child = read(self.root / order / 'manifest.json')
            hashes = child['variants'][builder.NAME]['source_sha256']
            compute = {k: v for k, v in hashes.items() if k not in builder.ORDER_FILES}
            if common is None:
                common = compute
            self.assertEqual(common, compute)
            self.assertIn('src/gemm_rs/epilogue_evt.hpp', compute)
            self.assertIn('src/gemm_rs/ths_op/gemm_reduce_scatter.cc', compute)
            self.assertIn('python/flux/gemm_rs_taco.py', compute)
            for role, entry in fused_plan.order_entries(plans, order).items():
                self.assertEqual(entry['mask'], plans['parent']['roles'][role]['entry']['mask'])
        self.assertFalse(record['gpu_validated'])

    def test_preparation_refuses_overwrite_and_diagnostic_cannot_compile_for_execution(self):
        with self.assertRaises(FileExistsError):
            builder.prepare(self.root, source_only=True)
        with mock.patch.object(builder.phase6_build, 'compile_build', side_effect=AssertionError('Unexpected compiler launch')):
            with self.assertRaisesRegex(ValueError, 'real-plan'):
                builder.compile_build(self.root, '/absent')

    def test_semantically_changed_compute_rejected_even_if_registry_hash_is_updated(self):
        record = read(self.root / 'fused-manifest.json')
        plans = read(self.root / 'plans.json')
        children = {o: read(self.root / o / 'manifest.json') for o in builder.ORDERS}
        name = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
        r, c = copy.deepcopy(record), copy.deepcopy(children)
        c['improved']['variants'][builder.NAME]['source_sha256'][name] = '0' * 64
        r['builds']['improved']['source_sha256'][name] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Non-order computation'):
            builder.validate_snapshots(r, plans, c, verify_evidence=False)

    def test_mutated_cuda_or_host_table_is_rejected(self):
        for name in builder.ORDER_FILES:
            path = self.root / 'improved' / builder.NAME / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n// bad lookup\n')
                with self.assertRaises(ValueError):
                    builder.verify(self.root)
            finally:
                path.write_bytes(original)
        builder.verify(self.root)

    def test_gpu_artifacts_are_not_inherited_from_phase6(self):
        for order in builder.ORDERS:
            src = self.root / order / builder.NAME
            self.assertFalse((src / 'libflux_cuda.so').exists())
            self.assertFalse((src / 'libflux_cuda_ths_op.so').exists())
            self.assertTrue((src / '3rdparty/cutlass').is_symlink())
            self.assertEqual((src / '3rdparty/cutlass').resolve(), REPO / '3rdparty/cutlass')
        with self.assertRaises(ValueError):
            builder.verify(self.root, built=True)


if __name__ == '__main__':
    unittest.main()
