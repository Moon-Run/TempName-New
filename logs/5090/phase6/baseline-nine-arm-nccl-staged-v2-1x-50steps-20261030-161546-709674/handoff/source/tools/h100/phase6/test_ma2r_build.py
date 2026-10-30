"""MA2-R source integrity: preserve A/V and reject undeclared R changes."""
import unittest

import ma2_sources
import ma2r_build as builder
import ma2r_sources
import test_arrival_op3_build as fixture
from common import read, sha, write


class BuildTests(unittest.TestCase):
    setUp = fixture.SourceIntegrityTests.setUp
    tearDown = fixture.SourceIntegrityTests.tearDown

    def test_old_ma2_is_preserved_and_receiver_arithmetic_is_shared(self):
        root = builder.prepare(self.root/'ma2r', self.baseline)
        record = builder.verify(root)
        self.assertEqual(record['variants']['A'], record['baseline_source_sha256'])
        self.assertTrue(record['baseline_source_only'])
        self.assertFalse(record['gpu_validated'])
        for name, text in ma2_sources.sources(root/'A', 'V').items():
            self.assertEqual((root/'V'/name).read_text(), text)
        self.assertEqual(record['transformed_files']['V'], sorted([ma2_sources.RUNTIME, ma2_sources.DECODER]))
        self.assertEqual((root/'V'/ma2_sources.DECODER).read_bytes(),
                         (root/'R'/ma2_sources.DECODER).read_bytes())
        for arm in ('V', 'R'):
            for name, digest in record['variants']['A'].items():
                if name not in record['transformed_files'][arm]:
                    self.assertEqual(record['variants'][arm][name], digest)
        with self.assertRaisesRegex(ValueError, 'unbuilt or diagnostic'):
            builder.verify(root, built=True)

    def test_rejects_changed_input_before_creating_build(self):
        (self.baseline/builder.RUNTIME).write_text('// changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
            builder.prepare(self.root/'ma2r', self.baseline)
        self.assertFalse((self.root/'ma2r').exists())

    def test_resealed_frontend_cannot_change_registered_transform(self):
        root = builder.prepare(self.root/'ma2r', self.baseline)
        name = ma2r_sources.FRONTEND
        path = root/'R'/name
        path.write_text(path.read_text()+'\n# undeclared submission change\n')
        record, info = read(root/builder.MANIFEST), read(root/'manifest.json')
        info['variants']['R']['source_sha256'][name] = sha(path)
        record['variants']['R'][name] = sha(path)
        write(root/'manifest.json', info)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/builder.MANIFEST, record)
        with self.assertRaisesRegex(ValueError, 'differs from the registered MA2-R transformation'):
            builder.verify(root)

    def test_all_host_mappings_cover_both_shapes_without_arrival(self):
        root = builder.prepare(self.root/'ma2r', self.baseline)
        result = builder.compile_check(root, self.root/'checks', host_only=True)
        self.assertTrue(result['completed'])
        self.assertFalse(result['gpu_used'])
        self.assertEqual(result['checks'], {a: dict(host_mapping=True) for a in builder.VARIANTS})


if __name__ == '__main__':
    unittest.main()
