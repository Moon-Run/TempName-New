"""Four-arm closure source integrity and real host compilation; no device use."""
import shutil
import unittest

import arrival_op4_v2_build as builder
import arrival_op4_v2_sources as source
import test_arrival_op3_build as fixture
from common import HERE, read, sha, write


class ClosureBuildTests(unittest.TestCase):
    setUp = fixture.SourceIntegrityTests.setUp
    tearDown = fixture.SourceIntegrityTests.tearDown

    def test_four_arms_retain_sealed_reference_and_source_only_provenance(self):
        root = builder.prepare(self.root / 'closure', self.baseline)
        record = builder.verify(root)
        self.assertEqual(record['protocol'], source.PROTOCOL)
        self.assertEqual(set(record['variants']), set(source.VARIANTS))
        self.assertTrue(record['baseline_source_only'])
        self.assertFalse(record['gpu_validated'])
        self.assertEqual(record['baseline_source_sha256'],
                         read(self.baseline.parent / 'manifest.json')['variants']['interleaved_all']['source_sha256'])
        info = read(root / 'manifest.json')
        self.assertTrue(info['source_only'])
        self.assertTrue(info['diagnostic_only'])
        for arm in builder.VARIANTS:
            self.assertEqual((root / arm / source.WRAPPER).read_text(),
                             (root / 'reference' / source.WRAPPER).read_text())
            self.assertEqual((root / arm / 'src/gemm_rs/taco_codec.cuh').read_text(),
                             (root / 'reference' / 'src/gemm_rs/taco_codec.cuh').read_text())
            self.assertEqual(record['transformed_files'][arm], sorted(source.sources(root / 'reference', arm)))
        self.assertEqual(set(record['inputs']), {str(HERE / name) for name in builder.INPUTS})

    def test_changed_baseline_is_rejected_before_output_creation(self):
        (self.baseline / builder.RUNTIME).write_text('// changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
            builder.prepare(self.root / 'closure', self.baseline)
        self.assertFalse((self.root / 'closure').exists())

    def test_detects_each_changed_arm_and_the_reference(self):
        for arm in (*builder.VARIANTS, 'reference'):
            with self.subTest(arm=arm):
                root = builder.prepare(self.root / arm, self.baseline)
                (root / arm / builder.RUNTIME).write_text('// changed')
                with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
                    builder.verify(root)

    def test_resealed_source_still_must_match_declared_transform(self):
        root = builder.prepare(self.root / 'closure', self.baseline)
        path = root / 'CM' / builder.RUNTIME
        path.write_text(path.read_text() + '\n// alternative\n')
        info, record = read(root / 'manifest.json'), read(root / builder.MANIFEST)
        info['variants']['CM']['source_sha256'][builder.RUNTIME] = sha(path)
        record['variants']['CM'][builder.RUNTIME] = sha(path)
        write(root / 'manifest.json', info)
        record['build_manifest_sha256'] = sha(root / 'manifest.json')
        write(root / builder.MANIFEST, record)
        with self.assertRaisesRegex(ValueError, 'CM differs from the registered transformation'):
            builder.verify(root)

    def test_unbuilt_and_legacy_labels_rejected(self):
        root = builder.prepare(self.root / 'closure', self.baseline)
        with self.assertRaisesRegex(ValueError, 'have not been built'):
            builder.verify(root, built=True)
        for old in ('A', 'D', 'L', 'F', 'O'):
            with self.assertRaisesRegex(ValueError, 'Expected SP, CP, SM or CM'):
                builder.runtime_root(root, old)

    @unittest.skipUnless(shutil.which('g++'), 'host compiler required')
    def test_host_mapping_executes_in_all_four_arms(self):
        root = builder.prepare(self.root / 'closure', self.baseline)
        result = builder.compile_check(root, self.root / 'compile', host_only=True)
        self.assertTrue(result['completed'])
        self.assertFalse(result['gpu_used'])
        self.assertFalse(result['gpu_validated'])
        self.assertFalse(result['linked_runtime'])
        self.assertEqual(result['checks'], {arm: {'host_mapping': True} for arm in builder.VARIANTS})
        self.assertEqual(result['objects'], {})


if __name__ == '__main__':
    unittest.main()
