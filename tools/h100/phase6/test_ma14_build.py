"""MA14 provenance: exact MA11 control and explicit candidate-only native ABI."""
import ctypes
import hashlib
from pathlib import Path
import tempfile
import unittest

import ma14_build as builder
import ma14_sources as source
import ma11_sources
from common import read, sha, write
from native_exports import verify_native_exports
from test_native_exports import compile_cpu_runtime


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma14-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_control_is_exact_ma11_and_candidate_changes_only_publication(self):
        record = builder.verify(self.root)
        frozen = read(self.root / 'reference/manifest.json')['variants']['F4']['source_sha256']
        updates = ma11_sources.sources(self.root / 'reference/F4', ma11_sources.VARIANTS[1])
        expected = dict(frozen, **{name: hashlib.sha256(value.encode()).hexdigest()
                                  for name, value in updates.items()})
        control, candidate = (record['variants'][arm] for arm in source.VARIANTS)
        self.assertEqual(control, expected)
        self.assertEqual(set(candidate) - set(control), {source.EPOCH_HEADER, source.PAIR_HEADER})
        self.assertEqual({name for name in control if control[name] != candidate[name]},
                         {source.RUNTIME, source.HEADER, source.WRAPPER, source.WRAPPER_HEADER,
                          source.PYBIND, source.FRONTEND, source.VERSION_SCRIPT})
        for name in (source.ZERO_COUNT_HEADER, source.DECODER, ma11_sources.SOFTMAX_HEADER,
                     ma11_sources.SOFTMAX_TU, source.CUDA_CMAKE):
            self.assertEqual(control[name], candidate[name])
        for name in ('ma8_adapter.py', 'ma9_adapter.py', 'ma14_epoch.cuh', 'ma14_pair.h', 'ma11_adapter.py', 'ma11_softmax.cuh'):
            self.assertIn(str(builder.HERE / name), record['inputs'])

    def test_actual_linker_map_preserves_ma9_in_both_arms(self):
        symbols = dict(taco_ma4f_protocol_version=1, taco_ma4f_decode_reduce=0)
        symbols.update({name: 0 for name in (*source.ZERO_COUNT_EXPORTS, *source.SOFTMAX_EXPORTS, *source.NATIVE_EXPORTS)})
        with tempfile.TemporaryDirectory(prefix='ma14-exports-') as tmp:
            for arm in source.VARIANTS:
                with self.subTest(arm=arm):
                    library = Path(tmp) / (arm + '.so')
                    compile_cpu_runtime(library,
                        (self.root / arm / source.VERSION_SCRIPT).read_text(),
                        {name: value for name, value in symbols.items()
                         if arm == source.VARIANTS[1] or name not in source.NATIVE_EXPORTS})
                    loaded = ctypes.CDLL(str(library))
                    self.assertTrue(all(hasattr(loaded, name) for name in source.ZERO_COUNT_EXPORTS + source.SOFTMAX_EXPORTS))
                    for name in source.NATIVE_EXPORTS:
                        self.assertEqual(hasattr(loaded, name), arm == source.VARIANTS[1])
                    verify_native_exports(library,
                        required=source.ZERO_COUNT_EXPORTS + source.SOFTMAX_EXPORTS + (source.NATIVE_EXPORTS if arm == source.VARIANTS[1] else ()),
                        forbidden=source.NATIVE_EXPORTS if arm == source.VARIANTS[0] else ())

    def test_rehashed_unregistered_change_in_either_arm_is_rejected(self):
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        for arm, name in ((source.VARIANTS[0], source.ZERO_COUNT_HEADER),
                          (source.VARIANTS[1], source.EPOCH_HEADER),
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
                with self.assertRaisesRegex(ValueError, 'registered MA14 transform'):
                    builder.verify(self.root)
            finally:
                for item, data in originals.items():
                    item.write_bytes(data)

    def test_unbuilt_sources_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA14 runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_version_one_manifest_is_rejected(self):
        path = self.root / builder.MANIFEST
        original = path.read_bytes()
        try:
            record = read(path)
            record['protocol'] = 'h100-ma14-epoch-reuse-v1'
            write(path, record)
            with self.assertRaisesRegex(ValueError, 'Wrong MA14 source protocol'):
                builder.verify(self.root)
        finally:
            path.write_bytes(original)

    def test_fixed_geometry_and_source_arm_validation(self):
        invalid = source.tuning_config()
        invalid['attn']['decoder_blocks'] = 1024
        with self.assertRaisesRegex(ValueError, 'fixes the original F4'):
            source.tuning_config(invalid)
        with self.assertRaisesRegex(ValueError, 'source arm'):
            source.sources(self.root / 'reference/F4', 'F4-cache')


if __name__ == '__main__':
    unittest.main()
