"""MA20 source bank, actual compiled host maps and unregistered-change rejection."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import ma20_build as builder
import ma11_sources
from common import read, sha, write


class BuildIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma20-build-test-')
        cls.root = builder.prepare(Path(cls.temp.name) / 'bank')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_ma11_control_and_only_registered_tables_change(self):
        record = builder.verify(self.root)
        reference = Path(record['reference'])
        expected = read(reference / 'manifest.json')['variants'][builder.REFERENCE_ARM]['source_sha256']
        self.assertEqual(record['variants']['MA11'], expected)
        self.assertEqual(record['variants']['Q0'], expected)
        for arm in builder.VARIANTS[2:]:
            changed = {name for name in expected if record['variants'][arm][name] != expected[name]}
            self.assertEqual(changed, set(builder.ALLOWED_SOURCE_CHANGES))
            for name in (ma11_sources.RUNTIME, ma11_sources.DECODER,
                         ma11_sources.WRAPPER, ma11_sources.SOFTMAX_HEADER,
                         ma11_sources.ZERO_COUNT_HEADER):
                self.assertEqual(record['variants'][arm][name], expected[name])

    def test_actual_host_mapping_compiles_and_matches_every_role_rank_slot(self):
        plans = read(self.root / builder.PLAN)
        for arm in builder.VARIANTS:
            body = '#include "phase6_host.hpp"\nint main() {\n'
            for role, k in (('attn', 512), ('mlp', 2048)):
                values = plans['configurations'][arm]['roles'][role]['entry']['coords']
                body += '{ const int expected[] = {' + ','.join(str(v) for row in values for v in row) + '};\n'
                body += ('for(int r=0;r<8;++r) for(int i=0;i<512;++i) '
                         'if(phase6_host_lookup(16,32,' + str(k) + ',r,i)!=expected[r*512+i]) return 1; }\n')
            body += 'return phase6_host_lookup(16,32,1024,0,0)==-1?0:2; }\n'
            source = Path(self.temp.name) / (arm + '-mapping.cc')
            executable = source.with_suffix('')
            source.write_text(body)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I' + str(self.root / arm),
                            str(source), '-o', str(executable)], check=True, capture_output=True)
            subprocess.run([str(executable)], check=True, capture_output=True)

    def test_rehashed_unregistered_native_changes_are_rejected(self):
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        for arm, name in (('MA11', ma11_sources.SOFTMAX_HEADER),
                          ('Q112', ma11_sources.RUNTIME), ('Q7', builder.SWIZZLE)):
            path = self.root / arm / name
            originals = {item: item.read_bytes() for item in (path, manifest_path, record_path)}
            try:
                path.write_bytes(originals[path] + b'\n// unregistered mutation\n')
                manifest, record = read(manifest_path), read(record_path)
                manifest['variants'][arm]['source_sha256'][name] = sha(path)
                record['variants'][arm][name] = sha(path)
                write(manifest_path, manifest)
                record['build_manifest_sha256'] = sha(manifest_path)
                write(record_path, record)
                with self.assertRaisesRegex(ValueError, 'registered MA20 transform'):
                    builder.verify(self.root)
            finally:
                for item, content in originals.items():
                    item.write_bytes(content)

    def test_rehashed_plan_mutation_cannot_replace_registered_mask(self):
        plan_path = self.root / builder.PLAN
        manifest_path, record_path = self.root / 'manifest.json', self.root / builder.MANIFEST
        originals = {item: item.read_bytes() for item in (plan_path, manifest_path, record_path)}
        try:
            plan = read(plan_path)
            plan['configurations']['Q7']['roles']['attn']['entry']['mask'][0][64] ^= 1
            write(plan_path, plan)
            manifest, record = read(manifest_path), read(record_path)
            manifest['files'][builder.PLAN] = record['files'][builder.PLAN] = sha(plan_path)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'registered joint-placement planner'):
                builder.verify(self.root)
        finally:
            for item, content in originals.items():
                item.write_bytes(content)

    def test_source_only_bank_cannot_execute_and_unknown_arm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing MA20 runtime|Unbuilt'):
            builder.verify(self.root, built=True)
        with self.assertRaisesRegex(ValueError, 'source arm'):
            builder.runtime_root(self.root, 'F4')
        self.assertEqual(builder.runtime_root(self.root, 'Q7'), self.root / 'Q7')

    def test_reuse_ma11_reference_creates_identical_fresh_bank(self):
        reference = Path(read(self.root / builder.MANIFEST)['reference'])
        other = builder.prepare(Path(self.temp.name) / 'reused', ma11_build=reference,
                                plan_path=self.root / builder.PLAN)
        self.assertEqual(read(other / builder.MANIFEST)['variants'],
                         read(self.root / builder.MANIFEST)['variants'])
        self.assertEqual(read(other / builder.PLAN), read(self.root / builder.PLAN))


if __name__ == '__main__':
    unittest.main()
