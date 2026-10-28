"""CPU source-contract checks for the optional MA5 producer K tile."""
from pathlib import Path
import tempfile
import unittest

from common import HERE, REPO
import ma5_gemm as gemm


class KTileTests(unittest.TestCase):
    def reference(self, root):
        root = Path(root)
        (root / gemm.GENERATOR).parent.mkdir(parents=True)
        (root / gemm.GENERATOR).write_text((HERE / 'generator.cc').read_text())
        (root / gemm.EPILOGUE).parent.mkdir(parents=True)
        (root / gemm.EPILOGUE).write_text((REPO / gemm.EPILOGUE).read_text())
        return root

    def test_default_has_no_producer_changes(self):
        self.assertEqual(gemm.sources('/nonexistent', 32), {})

    def test_k64_keeps_four_warps_and_exact_encoder_layout_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.reference(folder)
            before = {path: (root / path).read_bytes() for path in (gemm.GENERATOR, gemm.EPILOGUE)}
            changes = gemm.sources(root, 64)
            self.assertEqual(set(changes), set(before))
            generator = changes[gemm.GENERATOR]
            self.assertIn('Shape<_64,_64,_64>{}, Shape<_16,_8,_16>{}', generator)
            self.assertIn('Shape<_128,_128,_64>{}', generator)
            epilogue = changes[gemm.EPILOGUE]
            self.assertIn('std::is_same_v<ThreadMap, TacoThreadMap>', epilogue)
            self.assertIn('typename TacoOriginalThreadMap::Base', epilogue)
            self.assertIn('typename TacoOriginalThreadMap::ThreadMapShape', epilogue)
            # No encoding math, physical tile indexing or ownership code is edited.
            old = (root / gemm.EPILOGUE).read_text()
            start = old.index('  struct Arguments {')
            old_rest = old[start:].replace('TACO requires BF16 SM80 NVLink 128x128x32/128 threads and exact shape/rank',
                'MA5 K64 requires BF16 SM80 NVLink 128x128x64/128 threads and exact shape/rank')
            self.assertEqual(epilogue[epilogue.index('  struct Arguments {'):], old_rest)
            self.assertEqual(before, {path: (root / path).read_bytes() for path in before})

    def test_rejects_invalid_k_and_stale_contract(self):
        for value in (True, 16, 128, '64', None):
            with self.assertRaises(ValueError):
                gemm.sources('/nonexistent', value)
        with tempfile.TemporaryDirectory() as folder:
            root = self.reference(folder)
            (root / gemm.GENERATOR).write_text('changed generator')
            with self.assertRaises(ValueError):
                gemm.sources(root, 64)

    def test_real_kernel_probe_covers_both_tiles(self):
        for tile in (32, 64):
            source = gemm.compile_probe_source(tile)
            self.assertIn(f'Shape<_128, _128, _{tile}>', source)
            self.assertIn('Kernel::kThreadCount == 128', source)
            self.assertIn('StoreD::kTacoSupported', source)
            self.assertIn('op.run(args, workspace, stream)', source)
            self.assertEqual('make_gemm_v2_hparams' in source, tile == 64)


if __name__ == '__main__':
    unittest.main()
