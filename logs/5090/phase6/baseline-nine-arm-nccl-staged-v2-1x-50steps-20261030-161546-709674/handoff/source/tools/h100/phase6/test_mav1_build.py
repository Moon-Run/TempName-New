"""Control identity, registered host-only changes, and rehashed drift rejection."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import ma11_sources
import mav1_build as builder
import mav1_sources as source
from common import read, sha, write


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='mav1-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_control_is_exact_registered_ma11_and_cuda_sources_identical(self):
        record = builder.verify(self.root)
        frozen = read(self.root / 'reference/manifest.json')['variants']['F4']['source_sha256']
        updates = ma11_sources.sources(self.root / 'reference/F4', ma11_sources.VARIANTS[1])
        expected = dict(frozen, **{name: hashlib.sha256(text.encode()).hexdigest()
                                  for name, text in updates.items()})
        control, candidate = (record['variants'][arm] for arm in source.VARIANTS)
        self.assertEqual(control, expected)
        changed = {name for name in set(control) | set(candidate)
                   if control.get(name) != candidate.get(name)}
        self.assertEqual(changed, {source.NATIVE_HEADER, source.NATIVE_SOURCE,
                                  source.PYBIND_SOURCE, source.GEMM_PYBIND})
        self.assertFalse(any(name.endswith(('.cu', '.cuh')) for name in changed))
        for name in ('mav1_adapter.py', 'mav1_native.h', 'mav1_native.cc', 'mav1_pybind.cc'):
            self.assertIn(str(builder.HERE / name), record['inputs'])

    def test_rehashed_source_drift_is_rejected_in_both_arms(self):
        manifest_path = self.root / 'manifest.json'
        record_path = self.root / builder.MANIFEST
        for arm, name in ((source.VARIANTS[0], ma11_sources.SOFTMAX_HEADER),
                          (source.VARIANTS[1], source.NATIVE_SOURCE),
                          (source.VARIANTS[1], source.GEMM_PYBIND)):
            path = self.root / arm / name
            originals = {p: p.read_bytes() for p in (path, manifest_path, record_path)}
            try:
                path.write_bytes(originals[path] + b'\n// unregistered mutation\n')
                manifest, record = read(manifest_path), read(record_path)
                manifest['variants'][arm]['source_sha256'][name] = sha(path)
                record['variants'][arm][name] = sha(path)
                write(manifest_path, manifest)
                record['build_manifest_sha256'] = sha(manifest_path)
                write(record_path, record)
                with self.assertRaisesRegex(ValueError, 'registered MAV1 transform'):
                    builder.verify(self.root)
            finally:
                for p, content in originals.items():
                    p.write_bytes(content)

    def test_unbuilt_source_tree_is_not_runtime_evidence(self):
        with self.assertRaisesRegex(ValueError, 'Missing MAV1 runtime|Unbuilt'):
            builder.verify(self.root, built=True)

    def test_unknown_arm_and_geometry_are_rejected(self):
        with self.assertRaises(ValueError):
            source.sources(self.root / 'reference/F4', 'F4')
        tuning = source.tuning_config()
        tuning['attn']['decoder_blocks'] = 1
        with self.assertRaises(ValueError):
            source.tuning_config(tuning)


if __name__ == '__main__':
    unittest.main()
