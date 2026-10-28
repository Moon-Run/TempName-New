"""MA11 provenance: exact MA89 control and explicit candidate-only native ABI."""
import ctypes
import hashlib
from pathlib import Path
import tempfile
import unittest

import ma11_build as builder
import ma11_sources as source
import ma9_sources
from common import read, sha, write
from native_exports import verify_native_exports
from test_native_exports import compile_cpu_runtime


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma11-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_control_is_exact_ma89_and_candidate_adds_only_softmax(self):
        record = builder.verify(self.root)
        frozen = read(self.root / 'reference/manifest.json')['variants']['F4']['source_sha256']
        updates = ma9_sources.sources(self.root / 'reference/F4')
        expected = dict(frozen, **{name: hashlib.sha256(value.encode()).hexdigest()
                                  for name, value in updates.items()})
        control, candidate = (record['variants'][arm] for arm in source.VARIANTS)
        self.assertEqual(control, expected)
        self.assertEqual(set(candidate) - set(control),
                         {source.SOFTMAX_HEADER, source.SOFTMAX_TU, source.SOFTMAX_LICENSE})
        self.assertEqual({name for name in control if control[name] != candidate[name]},
                         {source.CUDA_CMAKE, source.VERSION_SCRIPT})
        self.assertEqual(control[source.ZERO_COUNT_HEADER], candidate[source.ZERO_COUNT_HEADER])
        self.assertEqual(control[source.RUNTIME], candidate[source.RUNTIME])
        cmake = (self.root / source.VARIANTS[1] / source.CUDA_CMAKE).read_text()
        self.assertIn('set_source_files_properties(ma11_softmax.cu PROPERTIES', cmake)
        self.assertIn(';'.join(source.SOFTMAX_FLAGS), cmake)
        for name in ('ma8_adapter.py', 'ma9_adapter.py', 'ma11_adapter.py', 'ma11_softmax.cuh'):
            self.assertIn(str(builder.HERE / name), record['inputs'])

    def test_actual_linker_map_preserves_ma9_in_both_arms(self):
        symbols = dict(taco_ma4f_protocol_version=1, taco_ma4f_decode_reduce=0)
        symbols.update({name: 0 for name in (*source.ZERO_COUNT_EXPORTS, *source.NATIVE_EXPORTS)})
        with tempfile.TemporaryDirectory(prefix='ma11-exports-') as tmp:
            for arm in source.VARIANTS:
                with self.subTest(arm=arm):
                    library = Path(tmp) / (arm + '.so')
                    compile_cpu_runtime(library,
                        (self.root / arm / source.VERSION_SCRIPT).read_text(), symbols)
                    loaded = ctypes.CDLL(str(library))
                    self.assertTrue(all(hasattr(loaded, name) for name in source.ZERO_COUNT_EXPORTS))
                    for name in source.NATIVE_EXPORTS:
                        self.assertEqual(hasattr(loaded, name), arm == source.VARIANTS[1])
                    verify_native_exports(library,
                        required=source.ZERO_COUNT_EXPORTS + (source.NATIVE_EXPORTS if arm == source.VARIANTS[1] else ()),
                        forbidden=source.NATIVE_EXPORTS if arm == source.VARIANTS[0] else ())

    def test_rehashed_unregistered_change_in_either_arm_is_rejected(self):
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        for arm, name in ((source.VARIANTS[0], source.ZERO_COUNT_HEADER),
                          (source.VARIANTS[1], source.SOFTMAX_HEADER),
                          (source.VARIANTS[1], source.VERSION_SCRIPT)):
            path = self.root / arm / name
            originals = {item: item.read_bytes() for item in (path, manifest_path, record_path)}
            try:
                path.write_bytes(originals[path] + b'\n// changed\n')
                manifest, record = read(manifest_path), read(record_path)
                manifest['variants'][arm]['source_sha256'][name] = sha(path)
                record['variants'][arm][name] = sha(path)
                write(manifest_path, manifest)
                record['build_manifest_sha256'] = sha(manifest_path)
                write(record_path, record)
                with self.assertRaisesRegex(ValueError, 'registered MA11 transform'):
                    builder.verify(self.root)
            finally:
                for item, data in originals.items():
                    item.write_bytes(data)

    def test_unbuilt_sources_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA11 runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_fixed_geometry_and_source_arm_validation(self):
        invalid = source.tuning_config()
        invalid['attn']['decoder_blocks'] = 1024
        with self.assertRaisesRegex(ValueError, 'fixes the original F4'):
            source.tuning_config(invalid)
        with self.assertRaisesRegex(ValueError, 'source arm'):
            source.sources(self.root / 'reference/F4', 'F4-cache')


if __name__ == '__main__':
    unittest.main()
