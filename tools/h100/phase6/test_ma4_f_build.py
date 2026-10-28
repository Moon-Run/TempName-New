"""Source provenance and fail-closed build tests for the MA4 consumer fusion candidate."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import ma4_f_build as builder
import ma4_f_sources as source
from common import read, sha, write


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma4-f-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_control_is_exact_cr2_and_fusion_is_registered(self):
        record = builder.verify(self.root)
        reference = read(self.root / 'reference/manifest.json')
        self.assertEqual(set(record['variants']), {'F4', 'CR2'})
        for arm in ('CR2',):
            self.assertEqual(record['variants'][arm],
                             reference['variants'][arm]['source_sha256'])
        expected = dict(record['variants']['CR2'])
        expected.update({name: hashlib.sha256(text.encode()).hexdigest()
                         for name, text in source.sources(self.root / 'CR2').items()})
        self.assertEqual(record['variants']['F4'], expected)
        self.assertNotEqual(record['variants']['F4'], record['variants']['CR2'])
        for name in ('src/gemm_rs/ma2_c_decode.cuh', 'src/gemm_rs/taco_codec.cuh',
                     'src/gemm_rs/gemm_v2_reduce_scatter.hpp'):
            self.assertEqual(record['variants']['F4'][name], record['variants']['CR2'][name])

    def test_prepared_tree_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA4-F/control runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_modified_control_is_rejected(self):
        path = self.root / 'CR2' / source.RUNTIME
        saved = path.read_bytes()
        try:
            path.write_bytes(saved + b'\n// unregistered change\n')
            with self.assertRaises(ValueError):
                builder.verify(self.root)
        finally:
            path.write_bytes(saved)

    def test_rehashed_candidate_tampering_does_not_bypass_transform(self):
        path = self.root / 'F4' / source.RUNTIME
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {p: p.read_bytes() for p in (path, manifest_path, record_path)}
        try:
            path.write_bytes(originals[path] + b'\n// adversarial extra edit\n')
            manifest, record = read(manifest_path), read(record_path)
            manifest['variants']['F4']['source_sha256'][source.RUNTIME] = sha(path)
            record['variants']['F4'][source.RUNTIME] = sha(path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered transform'):
                builder.verify(self.root)
        finally:
            for item, data in originals.items():
                item.write_bytes(data)


if __name__ == '__main__':
    unittest.main()
