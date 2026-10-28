"""Source identity, exact interleaved mapping, and sealed MA3 transformations."""
import unittest

import ma3_build as builder
import ma3_sources
import test_arrival_op3_build as fixture
from common import read, sha, write


class BuildTests(unittest.TestCase):
    setUp = fixture.SourceIntegrityTests.setUp
    tearDown = fixture.SourceIntegrityTests.tearDown

    def test_A_is_pinned_and_both_candidates_only_change_receiver(self):
        root = builder.prepare(self.root/'ma3', self.baseline)
        record = builder.verify(root)
        self.assertEqual(record['variants']['A'], record['baseline_source_sha256'])
        self.assertTrue(record['baseline_source_only'])
        self.assertFalse(record['gpu_validated'])
        for arm in ('D', 'S'):
            self.assertEqual(record['transformed_files'][arm],
                             sorted([ma3_sources.RUNTIME, ma3_sources.DECODER]))
            for name, digest in record['variants']['A'].items():
                if name != ma3_sources.RUNTIME:
                    self.assertEqual(record['variants'][arm][name], digest)
        self.assertEqual((root/'D'/ma3_sources.DECODER).read_bytes(),
                         (root/'S'/ma3_sources.DECODER).read_bytes())
        with self.assertRaisesRegex(ValueError, 'unbuilt or diagnostic'):
            builder.verify(root, built=True)

    def test_rejects_changed_input_before_creating_build(self):
        (self.baseline/builder.RUNTIME).write_text('// changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
            builder.prepare(self.root/'ma3', self.baseline)
        self.assertFalse((self.root/'ma3').exists())

    def test_resealed_receiver_cannot_change_registered_transform(self):
        for arm in ('D', 'S'):
            with self.subTest(arm=arm):
                root = builder.prepare(self.root/('ma3-'+arm), self.baseline)
                path = root/arm/builder.RUNTIME
                path.write_text(path.read_text()+'\n// unexpected receiver\n')
                record, info = read(root/builder.MANIFEST), read(root/'manifest.json')
                info['variants'][arm]['source_sha256'][builder.RUNTIME] = sha(path)
                record['variants'][arm][builder.RUNTIME] = sha(path)
                write(root/'manifest.json', info)
                record['build_manifest_sha256'] = sha(root/'manifest.json')
                write(root/builder.MANIFEST, record)
                with self.assertRaisesRegex(ValueError, 'differs from the registered MA3 transformation'):
                    builder.verify(root)

    def test_all_host_mappings_cover_both_shapes_without_arrival(self):
        root = builder.prepare(self.root/'ma3', self.baseline)
        result = builder.compile_check(root, self.root/'checks', host_only=True)
        self.assertTrue(result['completed'])
        self.assertFalse(result['gpu_used'])
        self.assertEqual(result['checks'], {a: dict(host_mapping=True) for a in builder.VARIANTS})


if __name__ == '__main__':
    unittest.main()
