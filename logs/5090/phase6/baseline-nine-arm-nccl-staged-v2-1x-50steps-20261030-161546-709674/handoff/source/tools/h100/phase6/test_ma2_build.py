"""MA2 isolation: same scalar baseline and unchanged MA1 prefetch control."""
import unittest

import ma1_sources
import ma2_build as builder
import ma2_sources
import test_arrival_op3_build as fixture
from common import read, sha, write


class BuildTests(unittest.TestCase):
    setUp = fixture.SourceIntegrityTests.setUp
    tearDown = fixture.SourceIntegrityTests.tearDown

    def test_old_prefetch_is_preserved_and_only_new_receiver_changes(self):
        root = builder.prepare(self.root/'ma2', self.baseline)
        record = builder.verify(root)
        self.assertEqual(record['variants']['A'], record['baseline_source_sha256'])
        self.assertTrue(record['baseline_source_only'])
        self.assertFalse(record['gpu_validated'])
        for name, text in ma1_sources.sources(root/'A', 'P').items():
            self.assertEqual((root/'P'/name).read_text(), text)
        self.assertEqual(record['transformed_files']['P'], sorted([ma1_sources.RUNTIME, ma1_sources.DECODER]))
        self.assertEqual(record['transformed_files']['V'], sorted([ma2_sources.RUNTIME, ma2_sources.DECODER]))
        for arm in ('P', 'V'):
            for name, digest in record['variants']['A'].items():
                if name != ma2_sources.RUNTIME:
                    self.assertEqual(record['variants'][arm][name], digest)
        self.assertIn('taco_ma1_decode_kernel<true>', (root/'P'/builder.RUNTIME).read_text())
        self.assertIn('taco_ma2_decode_kernel<<<', (root/'V'/builder.RUNTIME).read_text())
        self.assertNotIn('taco_ma1_decode_kernel', (root/'V'/builder.RUNTIME).read_text())
        with self.assertRaisesRegex(ValueError, 'unbuilt or diagnostic'):
            builder.verify(root, built=True)

    def test_rejects_changed_input_before_creating_build(self):
        (self.baseline/builder.RUNTIME).write_text('// changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
            builder.prepare(self.root/'ma2', self.baseline)
        self.assertFalse((self.root/'ma2').exists())

    def test_resealed_receiver_cannot_change_registered_transform(self):
        root = builder.prepare(self.root/'ma2', self.baseline)
        for arm in ('P', 'V'):
            with self.subTest(arm=arm):
                path = root/arm/builder.RUNTIME
                before = path.read_text()
                path.write_text(before+'\n// unexpected receiver\n')
                record, info = read(root/builder.MANIFEST), read(root/'manifest.json')
                info['variants'][arm]['source_sha256'][builder.RUNTIME] = sha(path)
                record['variants'][arm][builder.RUNTIME] = sha(path)
                write(root/'manifest.json', info)
                record['build_manifest_sha256'] = sha(root/'manifest.json')
                write(root/builder.MANIFEST, record)
                with self.assertRaisesRegex(ValueError, 'differs from the registered MA2 transformation'):
                    builder.verify(root)
                path.write_text(before)
                info['variants'][arm]['source_sha256'][builder.RUNTIME] = sha(path)
                record['variants'][arm][builder.RUNTIME] = sha(path)
                write(root/'manifest.json', info)
                record['build_manifest_sha256'] = sha(root/'manifest.json')
                write(root/builder.MANIFEST, record)

    def test_all_host_mappings_cover_both_shapes_without_arrival(self):
        root = builder.prepare(self.root/'ma2', self.baseline)
        result = builder.compile_check(root, self.root/'checks', host_only=True)
        self.assertTrue(result['completed'])
        self.assertFalse(result['gpu_used'])
        self.assertEqual(result['checks'], {a: dict(host_mapping=True) for a in builder.VARIANTS})


if __name__ == '__main__':
    unittest.main()
