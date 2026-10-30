"""Integrity checks for isolated A/O sources; these tests make no GPU claims."""
import tempfile
import unittest
from pathlib import Path

import arrival_op3_build as builder
import build
import scope_build
from common import HERE, REPO, read, sha, write
from scope_plan import model_config


class SourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='arrival-op3-build-test-')
        self.root = Path(self.temp.name)
        self.baseline = self.root/'parent/interleaved_all'
        pinned = HERE/'fused_v2'
        paths = read(pinned/'provenance.json')['files']
        for relative in paths:
            text = (pinned/relative).read_text()
            if relative == builder.WRAPPER:
                text = build.wrapper(text, legacy_fused=True)
            elif relative.endswith('.py'):
                text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)').replace(
                    'supports SM80 only', 'requires the H100 SM90 port').replace(
                    'tools/a800/phase4/build.py', 'tools/h100/phase6/build.py')
            target = self.baseline/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        entries = builder._identity_entries()
        scheduler = self.baseline/scope_build.SWIZZLE
        scheduler.parent.mkdir(parents=True, exist_ok=True)
        scheduler.write_text(scope_build.swizzle(
            (REPO/scope_build.SWIZZLE).read_text(), entries))
        (self.baseline/'phase6_host.hpp').write_text(
            '#pragma once\n'+scope_build.tables(entries))
        hashes = {str(p.relative_to(self.baseline)): sha(p)
                  for p in self.baseline.rglob('*') if p.is_file()}
        write(self.baseline.parent/'manifest.json', dict(
            protocol=build.PROTOCOL, arch='sm90', config=model_config(), submodules={},
            source_only=True, diagnostic_only=True, source_commit='unit-test-fixture',
            variants={'interleaved_all': dict(role='fused', base='interleaved',
                sampling=False, source_sha256=hashes)}))

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_baseline_and_source_only_provenance(self):
        root = builder.prepare(self.root/'op3', self.baseline)
        record = builder.verify(root)
        self.assertTrue(record['baseline_source_only'])
        self.assertFalse(record['gpu_validated'])
        self.assertEqual(record['variants']['A'], record['baseline_source_sha256'])
        self.assertEqual(read(root/'manifest.json')['status'], 'prepared')
        self.assertTrue((root/'O/src/gemm_rs/arrival_op3.cuh').is_file())
        self.assertFalse((root/'A/src/gemm_rs/arrival_op3.cuh').exists())

    def test_rejects_changed_registered_baseline_before_creating_output(self):
        (self.baseline/builder.RUNTIME).write_text('// changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
            builder.prepare(self.root/'op3', self.baseline)
        self.assertFalse((self.root/'op3').exists())

    def test_detects_modified_A_and_O(self):
        for variant in builder.VARIANTS:
            with self.subTest(variant=variant):
                root = builder.prepare(self.root/('op3-'+variant), self.baseline)
                (root/variant/builder.RUNTIME).write_text('// changed')
                with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
                    builder.verify(root)

    def test_rejects_nonidentity_A_schedule_even_with_updated_hash(self):
        path = self.baseline/'phase6_host.hpp'
        path.write_text(path.read_text()+'\n// changed schedule source\n')
        parent = read(self.baseline.parent/'manifest.json')
        parent['variants']['interleaved_all']['source_sha256']['phase6_host.hpp'] = sha(path)
        write(self.baseline.parent/'manifest.json', parent)
        with self.assertRaisesRegex(ValueError, 'identity schedule'):
            builder.prepare(self.root/'op3', self.baseline)

    def test_unbuilt_sources_cannot_pass_built_verification(self):
        root = builder.prepare(self.root/'op3', self.baseline)
        with self.assertRaisesRegex(ValueError, 'have not been built'):
            builder.verify(root, built=True)


if __name__ == '__main__':
    unittest.main()
