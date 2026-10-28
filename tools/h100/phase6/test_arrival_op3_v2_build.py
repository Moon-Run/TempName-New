"""Integrity checks for A/D/O source builds; no GPU validation is implied."""
import shutil
import unittest

import arrival_op3_v2_build as builder
import test_arrival_op3_build as legacy_fixture
from common import HERE, read, sha, write


class ArrivalOp3V2BuildTests(unittest.TestCase):
    # The registered historical A fixture is shared; all actions under test use
    # the new builder and independently transformed D/O trees.
    setUp = legacy_fixture.SourceIntegrityTests.setUp
    tearDown = legacy_fixture.SourceIntegrityTests.tearDown

    def test_preserves_A_and_source_only_provenance_with_three_distinct_variants(self):
        root = builder.prepare(self.root/'v2', self.baseline)
        record = builder.verify(root)
        info = read(root/'manifest.json')
        self.assertEqual(record['protocol'], 'h100-arrival-op3-v2')
        self.assertEqual(set(record['variants']), {'A', 'D', 'O'})
        self.assertTrue(record['baseline_source_only'])
        self.assertTrue(info['source_only'])
        self.assertTrue(info['diagnostic_only'])
        self.assertFalse(record['gpu_validated'])
        self.assertEqual(record['variants']['A'], record['baseline_source_sha256'])
        self.assertEqual(info['status'], 'prepared')
        self.assertEqual(record['transformed_files']['D'],
                         sorted(['src/gemm_rs/taco_runtime.cu',
                                 'src/gemm_rs/arrival_op3_v2_decode.cuh']))
        self.assertFalse((root/'D/src/gemm_rs/arrival_op3_v2.cuh').exists())
        self.assertTrue((root/'O/src/gemm_rs/arrival_op3_v2.cuh').exists())
        for variant in ('D', 'O'):
            self.assertFalse((root/variant/'src/gemm_rs/arrival_op3.cuh').exists())
        self.assertEqual(set(record['inputs']), {str(HERE/name) for name in builder.INPUTS})
        self.assertIn(str(HERE/'arrival_op3_sources.py'), record['inputs'])

    def test_rejects_changed_registered_baseline_before_creating_output(self):
        (self.baseline/builder.RUNTIME).write_text('// changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
            builder.prepare(self.root/'v2', self.baseline)
        self.assertFalse((self.root/'v2').exists())

    def test_detects_mutation_of_each_variant(self):
        for variant in builder.VARIANTS:
            with self.subTest(variant=variant):
                root = builder.prepare(self.root/('v2-'+variant), self.baseline)
                (root/variant/builder.RUNTIME).write_text('// changed')
                with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
                    builder.verify(root)

    def test_rejects_resealed_D_source_that_differs_from_registered_transformation(self):
        root = builder.prepare(self.root/'v2', self.baseline)
        path = root/'D'/builder.RUNTIME
        path.write_text(path.read_text()+'\n// unexpected alternative\n')
        record = read(root/builder.MANIFEST)
        info = read(root/'manifest.json')
        info['variants']['D']['source_sha256'][builder.RUNTIME] = sha(path)
        record['variants']['D'][builder.RUNTIME] = sha(path)
        write(root/'manifest.json', info)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/builder.MANIFEST, record)
        with self.assertRaisesRegex(ValueError, 'D differs from the registered transformation'):
            builder.verify(root)

    def test_unbuilt_sources_are_rejected_and_old_variant_labels_are_not_accepted(self):
        root = builder.prepare(self.root/'v2', self.baseline)
        with self.assertRaisesRegex(ValueError, 'have not been built'):
            builder.verify(root, built=True)
        for variant in ('S', 'V', 'old-O'):
            with self.assertRaisesRegex(ValueError, 'Expected A, D or O'):
                builder.runtime_root(root, variant)
        self.assertEqual(builder.runtime_root(root, 'D'), root/'D')

    def test_nonidentity_schedule_is_rejected_even_with_updated_baseline_hash(self):
        path = self.baseline/'phase6_host.hpp'
        path.write_text(path.read_text()+'\n// unsupported schedule source\n')
        parent = read(self.baseline.parent/'manifest.json')
        parent['variants']['interleaved_all']['source_sha256']['phase6_host.hpp'] = sha(path)
        write(self.baseline.parent/'manifest.json', parent)
        with self.assertRaisesRegex(ValueError, 'identity schedule'):
            builder.prepare(self.root/'v2', self.baseline)

    @unittest.skipUnless(shutil.which('g++'), 'g++ needed for real host mapping check')
    def test_host_only_check_executes_all_three_mappings_without_GPU_claims(self):
        root = builder.prepare(self.root/'v2', self.baseline)
        output = self.root/'compile'
        result = builder.compile_check(root, output, host_only=True)
        self.assertTrue(result['completed'])
        self.assertEqual(result['mode'], 'host-only')
        self.assertFalse(result['gpu_used'])
        self.assertFalse(result['gpu_validated'])
        self.assertFalse(result['linked_runtime'])
        self.assertEqual(result['checks'], {
            variant: {'host_mapping': True} for variant in ('A', 'D', 'O')})
        self.assertEqual(result['objects'], {})
        self.assertEqual(read(output/'validation.json'), result)
        self.assertTrue(all(event['completed'] for event in result['commands']))


if __name__ == '__main__':
    unittest.main()
