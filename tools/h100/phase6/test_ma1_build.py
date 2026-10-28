"""MA1 source isolation and mapping checks; never substitutes for GPU tests."""
import unittest

import ma1_build as builder
import ma1_sources
import test_arrival_op3_build as fixture
from common import read, sha, write


class BuildTests(unittest.TestCase):
    setUp = fixture.SourceIntegrityTests.setUp
    tearDown = fixture.SourceIntegrityTests.tearDown

    def test_only_receiver_changes_and_wp_share_identical_math_header(self):
        root = builder.prepare(self.root/'ma1', self.baseline)
        record = builder.verify(root)
        self.assertEqual(record['variants']['A'], record['baseline_source_sha256'])
        self.assertTrue(record['baseline_source_only'])
        self.assertFalse(record['gpu_validated'])
        for arm in ('W', 'P'):
            self.assertEqual(record['transformed_files'][arm], sorted([ma1_sources.RUNTIME, ma1_sources.DECODER]))
            for name, digest in record['variants']['A'].items():
                if name != ma1_sources.RUNTIME:
                    self.assertEqual(record['variants'][arm][name], digest)
        self.assertEqual(record['variants']['W'][ma1_sources.DECODER], record['variants']['P'][ma1_sources.DECODER])
        w = (root/'W'/ma1_sources.RUNTIME).read_text()
        p = (root/'P'/ma1_sources.RUNTIME).read_text()
        self.assertEqual(w.replace('kernel<false>', 'kernel<true>').replace(
            'taco_ma1_prefetch_enabled() { return 0;', 'taco_ma1_prefetch_enabled() { return 1;'), p)
        with self.assertRaisesRegex(ValueError, 'unbuilt or diagnostic'):
            builder.verify(root, built=True)

    def test_rejects_changed_input_before_creating_build(self):
        (self.baseline/builder.RUNTIME).write_text('// changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
            builder.prepare(self.root/'ma1', self.baseline)
        self.assertFalse((self.root/'ma1').exists())

    def test_resealed_receiver_cannot_change_registered_transform(self):
        root = builder.prepare(self.root/'ma1', self.baseline)
        path = root/'P'/builder.RUNTIME
        path.write_text(path.read_text()+'\n// alternate path\n')
        record, info = read(root/builder.MANIFEST), read(root/'manifest.json')
        info['variants']['P']['source_sha256'][builder.RUNTIME] = sha(path)
        record['variants']['P'][builder.RUNTIME] = sha(path)
        write(root/'manifest.json', info)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/builder.MANIFEST, record)
        with self.assertRaisesRegex(ValueError, 'differs from the registered MA1 transformation'):
            builder.verify(root)

    def test_all_host_mappings_cover_both_shapes_without_arrival(self):
        root = builder.prepare(self.root/'ma1', self.baseline)
        result = builder.compile_check(root, self.root/'checks', host_only=True)
        self.assertTrue(result['completed'])
        self.assertFalse(result['gpu_used'])
        self.assertEqual(result['checks'], {a: dict(host_mapping=True) for a in builder.VARIANTS})


if __name__ == '__main__':
    unittest.main()
