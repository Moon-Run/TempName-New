"""OP3 codec composition and immutable parent integrity, without GPU execution."""
from pathlib import Path
import tempfile
import unittest

import op3_build as builder
import op3_native as native


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='op3-build-test-')
        cls.base = Path(cls.temporary.name)/'baseline'
        cls.abc = Path(cls.temporary.name)/'abc'
        cls.root = Path(cls.temporary.name)/'op3'
        builder.parent.base.prepare(cls.base)
        builder.parent.prepare(cls.abc, cls.base)
        cls.before = builder.sha(cls.abc/builder.parent.MANIFEST)
        builder.prepare(cls.root, cls.abc)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_composition_retains_one_verified_gemm_for_both_paths(self):
        record = builder.verify(self.root)
        self.assertEqual(record['status'], 'prepared')
        self.assertFalse(record['gpu_validated'])
        self.assertEqual(builder.base_root(self.root), self.base)
        self.assertEqual(builder.sha(self.abc/builder.parent.MANIFEST), self.before)
        self.assertEqual(record['inputs']['native'], native.identity())
        self.assertEqual({builder.native_root(self.root, arm) for arm in builder.ARMS},
                         {builder.parent.native_root(self.abc, 'flux-basic-interleaved')})
        self.assertEqual(set(record['files']), {'codec/'+name for name in native.source_files()})

    def test_missing_or_tampered_inputs_are_rejected(self):
        for path in (self.root/'codec/op3_codec.cu', self.abc/builder.parent.MANIFEST):
            old = path.read_bytes()
            try:
                path.write_bytes(old+b'\n')
                with self.assertRaises(ValueError):
                    builder.verify(self.root)
            finally:
                path.write_bytes(old)
        with self.assertRaises(ValueError):
            builder.verify(self.root, True)
        with self.assertRaises(ValueError):
            builder.prepare(self.root, self.abc)
        with self.assertRaises(ValueError):
            builder.native_root(self.root, 'flux-basic')


if __name__ == '__main__':
    unittest.main()
