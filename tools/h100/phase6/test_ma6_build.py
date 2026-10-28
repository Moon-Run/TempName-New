"""Three-arm source provenance and rejection of rehashed unregistered changes."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import ma6_build as builder
import ma6_sources as source
from common import read, sha, write


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma6-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_control_is_exact_f4_and_candidates_are_independent_registered_transforms(self):
        record = builder.verify(self.root)
        frozen = read(self.root / 'reference/manifest.json')['variants']['F4']['source_sha256']
        self.assertEqual(tuple(record['variants']), ('F4', 'F4-mask', 'F4-layout'))
        self.assertEqual(record['variants']['F4'], frozen)
        for arm in builder.VARIANTS[1:]:
            expected = dict(frozen)
            updates = source.sources(self.root / 'F4', arm, record['tuning'])
            expected.update({name:hashlib.sha256(text.encode()).hexdigest() for name,text in updates.items()})
            self.assertEqual(record['variants'][arm], expected)
            changed = {name for name in expected if expected[name] != frozen.get(name)}
            fields = {source.RUNTIME, source.HEADER, source.WRAPPER, source.DECODER}
            if arm == 'F4-mask':
                fields |= {source.FRONTEND,source.PYBIND,source.WRAPPER_HEADER,source.MASK_HEADER}
            self.assertEqual(changed, fields)
        self.assertEqual(read(self.root / 'tuning.json'), source.DEFAULT_TUNING)

    def test_prepared_sources_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA6 runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_native_changes_are_rejected(self):
        for arm in builder.VARIANTS:
            path = self.root / arm / source.RUNTIME
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n// unregistered change\n')
                with self.assertRaises(ValueError):
                    builder.verify(self.root)
            finally:
                path.write_bytes(original)

    def test_rehashing_candidate_changes_does_not_bypass_registered_transform(self):
        path = self.root / 'F4-mask' / source.RUNTIME
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {p:p.read_bytes() for p in (path, manifest_path, record_path)}
        try:
            path.write_bytes(originals[path] + b'\n// unregistered extra edit\n')
            manifest, record = read(manifest_path), read(record_path)
            manifest['variants']['F4-mask']['source_sha256'][source.RUNTIME] = sha(path)
            record['variants']['F4-mask'][source.RUNTIME] = sha(path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered transform'):
                builder.verify(self.root)
        finally:
            for item, data in originals.items():
                item.write_bytes(data)

    def test_frozen_tuning_and_record_cannot_diverge(self):
        path = self.root / builder.MANIFEST
        original = path.read_bytes()
        try:
            record = read(path)
            record['tuning']['mlp'] = dict(decoder_threads=256, decoder_blocks=528, decoder_rows=1)
            write(path, record)
            with self.assertRaisesRegex(ValueError, 'tuning identity|fixes decoder geometry'):
                builder.verify(self.root)
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
