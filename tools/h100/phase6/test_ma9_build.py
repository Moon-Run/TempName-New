"""MA9 preserves exact F4 control and registers only the optimizer helper."""
import ctypes
from pathlib import Path
import tempfile
import unittest

import ma9_build as builder
import ma9_sources as source
from common import read, sha, write
from native_exports import verify_native_exports
from test_native_exports import compile_cpu_runtime


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma9-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_control_exact_and_candidate_only_adds_registered_helper(self):
        record = builder.verify(self.root)
        frozen = read(self.root / 'reference/manifest.json')['variants']['F4']['source_sha256']
        self.assertEqual(tuple(record['variants']), ('F4', 'F4-zcount'))
        self.assertEqual(record['variants']['F4'], frozen)
        candidate = record['variants']['F4-zcount']
        self.assertEqual(set(candidate) - set(frozen), {source.ZERO_COUNT_HEADER})
        self.assertEqual({name for name in frozen if candidate[name] != frozen[name]},
                         {source.RUNTIME, source.VERSION_SCRIPT})
        self.assertEqual((self.root / 'F4-zcount' / source.RUNTIME).read_bytes(),
                         (self.root / 'F4' / source.RUNTIME).read_bytes() + source.ZERO_COUNT_INCLUDE.encode())
        self.assertEqual(read(self.root / 'tuning.json'), source.DEFAULT_TUNING)
        self.assertIn(str(builder.HERE / 'ma9_adapter.py'), record['inputs'])
        self.assertTrue(record['control_native_sources_unchanged'])
        self.assertTrue(record['projection_sources_unchanged'])
        self.assertEqual(record['candidate_native_addition'], 'optimizer_zero_count')
        self.assertIn(str(builder.HERE / 'ma9_zero_count.cuh'), record['inputs'])

    def test_actual_linker_map_exports_helpers_only_for_candidate(self):
        symbols = dict(taco_ma4f_protocol_version=1, taco_ma4f_decode_reduce=0)
        symbols.update({name: 0 for name in source.NATIVE_EXPORTS})
        with tempfile.TemporaryDirectory(prefix='ma-export-map-test-') as tmp:
            for arm in builder.VARIANTS:
                with self.subTest(arm=arm):
                    library = Path(tmp) / (arm + '.so')
                    script = (self.root / arm / source.VERSION_SCRIPT).read_text()
                    compile_cpu_runtime(library, script, symbols)
                    loaded = ctypes.CDLL(str(library))
                    self.assertEqual(loaded.taco_ma4f_protocol_version(), 1)
                    for name in source.NATIVE_EXPORTS:
                        self.assertEqual(hasattr(loaded, name), arm != 'F4', name)
                    verify_native_exports(library,
                        required=source.NATIVE_EXPORTS if arm != 'F4' else (),
                        forbidden=source.NATIVE_EXPORTS if arm == 'F4' else ())

    def test_rehashing_export_map_cannot_broaden_candidate_api(self):
        path = self.root / 'F4-zcount' / source.VERSION_SCRIPT
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {item: item.read_bytes() for item in (path, manifest_path, record_path)}
        try:
            path.write_text(path.read_text().replace('global:', 'global: unregistered_helper;'))
            manifest, record = read(manifest_path), read(record_path)
            manifest['variants']['F4-zcount']['source_sha256'][source.VERSION_SCRIPT] = sha(path)
            record['variants']['F4-zcount'][source.VERSION_SCRIPT] = sha(path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered MA9 transform'):
                builder.verify(self.root)
        finally:
            for item, data in originals.items():
                item.write_bytes(data)

    def test_prepared_sources_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA9 runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_rehashing_native_candidate_changes_is_rejected(self):
        path = self.root / 'F4-zcount' / source.RUNTIME
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {item: item.read_bytes() for item in (path, manifest_path, record_path)}
        try:
            path.write_bytes(originals[path] + b'\n// unregistered native change\n')
            manifest, record = read(manifest_path), read(record_path)
            manifest['variants']['F4-zcount']['source_sha256'][source.RUNTIME] = sha(path)
            record['variants']['F4-zcount'][source.RUNTIME] = sha(path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered MA9 transform'):
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

    def test_rehashing_helper_changes_cannot_register_a_new_algorithm(self):
        path = self.root / 'F4-zcount' / source.ZERO_COUNT_HEADER
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {item: item.read_bytes() for item in (path, manifest_path, record_path)}
        try:
            path.write_bytes(originals[path] + b'\n// unregistered zero-count change\n')
            manifest, record = read(manifest_path), read(record_path)
            manifest['variants']['F4-zcount']['source_sha256'][source.ZERO_COUNT_HEADER] = sha(path)
            record['variants']['F4-zcount'][source.ZERO_COUNT_HEADER] = sha(path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered MA9 transform'):
                builder.verify(self.root)
        finally:
            for item, data in originals.items():
                item.write_bytes(data)

    def test_geometry_cannot_become_a_second_experiment(self):
        changed = source.tuning_config()
        changed['mlp']['decoder_blocks'] = 1024
        with self.assertRaisesRegex(ValueError, 'fixes the original F4'):
            source.tuning_config(changed)
        self.assertEqual(source.kernel_name('attn', arm='F4-zcount'),
                         source.kernel_name('mlp', arm='F4'))


if __name__ == '__main__':
    unittest.main()
