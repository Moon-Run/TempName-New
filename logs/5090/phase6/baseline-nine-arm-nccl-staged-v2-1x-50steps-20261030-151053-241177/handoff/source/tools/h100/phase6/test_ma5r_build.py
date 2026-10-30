"""MA5-R source provenance, configuration identity, and fail-closed checks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import ma5r_build as builder
import ma5r_sources as source
from common import read, sha, write


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma5r-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_control_and_candidate_source_identity(self):
        record = builder.verify(self.root)
        reference = read(self.root / 'reference/manifest.json')
        self.assertEqual(tuple(record['variants']), ('T', 'CR2'))
        self.assertEqual(record['variants']['CR2'], reference['variants']['CR2']['source_sha256'])
        expected = dict(record['variants']['CR2'])
        expected.update({name: hashlib.sha256(text.encode()).hexdigest()
                         for name, text in source.sources(self.root / 'CR2', record['tuning']).items()})
        self.assertEqual(record['variants']['T'], expected)
        self.assertEqual(read(self.root / 'tuning.json'), source.DEFAULT_TUNING)

    def test_prepared_sources_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA5-R/control runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_unregistered_control_change_is_rejected(self):
        path = self.root / 'CR2' / source.RUNTIME
        saved = path.read_bytes()
        try:
            path.write_bytes(saved + b'\n// unregistered change\n')
            with self.assertRaises(ValueError):
                builder.verify(self.root)
        finally:
            path.write_bytes(saved)

    def test_rehashing_candidate_does_not_bypass_registered_transform(self):
        path = self.root / 'T' / source.RUNTIME
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {p: p.read_bytes() for p in (path, manifest_path, record_path)}
        try:
            path.write_bytes(originals[path] + b'\n// unregistered change\n')
            manifest, record = read(manifest_path), read(record_path)
            manifest['variants']['T']['source_sha256'][source.RUNTIME] = sha(path)
            record['variants']['T'][source.RUNTIME] = sha(path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered transform'):
                builder.verify(self.root)
        finally:
            for item, data in originals.items():
                item.write_bytes(data)

    def test_tuning_record_must_match_frozen_file(self):
        path = self.root / builder.MANIFEST
        saved = path.read_bytes()
        try:
            record = read(path)
            record['tuning']['attn']['decoder_blocks'] = 264
            write(path, record)
            with self.assertRaisesRegex(ValueError, 'tuning identity'):
                builder.verify(self.root)
        finally:
            path.write_bytes(saved)


if __name__ == '__main__':
    unittest.main()
