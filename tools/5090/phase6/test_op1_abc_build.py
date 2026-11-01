"""CPU-only source provenance of the composite ABC/Flux build."""
from pathlib import Path
import tempfile
import unittest

import op1_abc_build as builder
import op1_abc_native as native


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='rtx5090-abc-source-test-')
        cls.parent = Path(cls.temporary.name)/'baseline'
        cls.root = Path(cls.temporary.name)/'abc'
        builder.base.prepare(cls.parent)
        cls.parent_manifest = builder.sha(cls.parent/'baseline-build.json')
        builder.prepare(cls.root, cls.parent)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_source_composition_is_registered_and_parent_readonly(self):
        record = builder.verify(self.root)
        self.assertEqual(builder.sha(self.parent/'baseline-build.json'), self.parent_manifest)
        self.assertEqual(builder.base_root(self.root), self.parent)
        self.assertFalse(record['gpu_validated'])
        self.assertEqual(record['actual_arch'], 'sm120')
        self.assertEqual(record['status'], 'prepared')
        self.assertEqual(record['inputs']['native'], native.identity())
        self.assertEqual(set(record['files']), {'codec/'+name for name in native.source_files()})
        self.assertEqual({builder.native_root(self.root, arm) for arm in builder.ARMS[:3]},
                         {builder.sorted_build.native_root(self.root/'interleaved', 'flux-basic-interleaved')})
        self.assertEqual(builder.native_root(self.root, 'flux-improved'),
                         builder.improved_build.native_root(builder.flux_root(self.root), 'flux-improved'))
        for name, content in native.source_files().items():
            self.assertEqual((self.root/'codec'/name).read_text(), content)

    def test_tampered_codec_source_rejected(self):
        path = self.root/'codec/op1_abc_encoder.cuh'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'\n// unregistered source drift\n')
            with self.assertRaisesRegex(ValueError, 'ABC codec source/binary changed'):
                builder.verify(self.root)
        finally:
            path.write_bytes(original)

    def test_dependency_manifest_changes_are_rejected(self):
        for path in (self.root/'interleaved'/builder.sorted_build.MANIFEST,
                     builder.flux_root(self.root)/builder.improved_build.MANIFEST):
            original = path.read_bytes()
            try:
                path.write_bytes(original+b'\n')
                with self.assertRaisesRegex(ValueError, 'ABC dependency manifest changed'):
                    builder.verify(self.root)
            finally:
                path.write_bytes(original)

    def test_parent_manifest_changes_are_rejected(self):
        path = self.parent/'baseline-build.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError, 'ABC parent baseline manifest changed'):
                builder.verify(self.root)
        finally:
            path.write_bytes(original)

    def test_prepared_sources_cannot_execute_or_reuse_as_compiled_flux(self):
        with self.assertRaisesRegex(ValueError, 'Build incomplete'):
            builder.verify(self.root, True)
        with self.assertRaisesRegex(ValueError, 'Build incomplete'):
            builder.prepare(self.root.parent/'invalid-reuse', self.parent,
                            flux_root=builder.flux_root(self.root))
        self.assertFalse((self.root.parent/'invalid-reuse').exists())

    def test_unknown_arm_and_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown ABC arm'):
            builder.native_root(self.root, 'flux-basic')
        with self.assertRaisesRegex(ValueError, 'fresh ABC build directory'):
            builder.prepare(self.root, self.parent)


if __name__ == '__main__':
    unittest.main()
