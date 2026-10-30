"""The retained-activation experiment must not change any native CR3/F4 source."""
from pathlib import Path
import tempfile
import unittest

import ma8_build as builder
import ma8_sources as source
from common import read, sha, write


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma8-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_both_arms_are_byte_identical_to_registered_f4(self):
        record = builder.verify(self.root)
        frozen = read(self.root / 'reference/manifest.json')['variants']['F4']['source_sha256']
        self.assertEqual(tuple(record['variants']), ('F4', 'F4-cache'))
        for arm in builder.VARIANTS:
            self.assertEqual(record['variants'][arm], frozen)
        self.assertEqual(read(self.root / 'tuning.json'), source.DEFAULT_TUNING)
        self.assertIn(str(builder.HERE / 'ma8_adapter.py'), record['inputs'])
        self.assertTrue(record['native_sources_unchanged'])

    def test_prepared_sources_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA8 runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_v1_build_cannot_be_reused_for_uninstrumented_v2(self):
        path = self.root / builder.MANIFEST
        saved = path.read_bytes()
        try:
            record = read(path)
            self.assertEqual(record['protocol'], 'h100-ma8-f4-cache-v2')
            record['protocol'] = 'h100-ma8-f4-cache-v1'
            write(path, record)
            with self.assertRaisesRegex(ValueError, 'Wrong MA8 source protocol'):
                builder.verify(self.root)
        finally:
            path.write_bytes(saved)

    def test_rehashing_native_candidate_changes_is_rejected(self):
        path = self.root / 'F4-cache' / source.RUNTIME
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {item: item.read_bytes() for item in (path, manifest_path, record_path)}
        try:
            path.write_bytes(originals[path] + b'\n// unregistered native change\n')
            manifest, record = read(manifest_path), read(record_path)
            manifest['variants']['F4-cache']['source_sha256'][source.RUNTIME] = sha(path)
            record['variants']['F4-cache'][source.RUNTIME] = sha(path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered unchanged F4'):
                builder.verify(self.root)
        finally:
            for item, data in originals.items():
                item.write_bytes(data)

    def test_recorded_library_mutation_is_rejected(self):
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {item: item.read_bytes() for item in (manifest_path, record_path)}
        artifact = self.root / 'synthetic-runtime.so'
        try:
            artifact.write_bytes(b'recorded test library')
            manifest, record = read(manifest_path), read(record_path)
            manifest['files'][artifact.name] = sha(artifact)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            builder.verify(self.root)
            artifact.write_bytes(b'changed test library')
            with self.assertRaisesRegex(ValueError, 'changed|mismatch'):
                builder.verify(self.root)
        finally:
            artifact.unlink(missing_ok=True)
            for item, data in originals.items():
                item.write_bytes(data)

    def test_geometry_cannot_become_a_second_experiment(self):
        changed = source.tuning_config()
        changed['mlp']['decoder_blocks'] = 1024
        with self.assertRaisesRegex(ValueError, 'fixes the original F4'):
            source.tuning_config(changed)
        self.assertEqual(source.kernel_name('attn', arm='F4-cache'),
                         source.kernel_name('mlp', arm='F4'))


if __name__ == '__main__':
    unittest.main()
