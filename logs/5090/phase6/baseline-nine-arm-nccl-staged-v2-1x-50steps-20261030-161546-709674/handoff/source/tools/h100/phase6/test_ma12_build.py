"""Native source identity, public exports, and immutable MA89 control."""
import ctypes
import hashlib
from pathlib import Path
import tempfile
import unittest

import ma12_build as builder
import ma12_sources as source
import ma9_sources
from common import read, sha, write
from test_native_exports import compile_cpu_runtime


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma12-build-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = builder.prepare(Path(cls.temp.name)/'build')

    def test_exact_ma89_control_and_registered_candidate(self):
        record = builder.verify(self.root)
        frozen = read(self.root/'reference/manifest.json')['variants']['F4']['source_sha256']
        zero = ma9_sources.sources(self.root/'reference/F4')
        control = dict(frozen, **{name:hashlib.sha256(value.encode()).hexdigest() for name,value in zero.items()})
        self.assertEqual(record['variants'][source.VARIANTS[0]], control)
        delta = source.sources(self.root/source.VARIANTS[0])
        expected = dict(control, **{name:hashlib.sha256(value.encode()).hexdigest() for name,value in delta.items()})
        self.assertEqual(record['variants'][source.VARIANTS[1]], expected)
        self.assertEqual(record['variants'][source.VARIANTS[1]][source.ZERO_COUNT_HEADER], control[source.ZERO_COUNT_HEADER])
        self.assertFalse(record['projection_sources_unchanged'])
        self.assertTrue(record['projection_arithmetic_unchanged'])
        self.assertIn(str(builder.HERE/'ma12_tile_ready.cuh'), record['inputs'])

    def test_actual_export_map_retains_ma9_and_isolates_ma12(self):
        symbols = dict(taco_ma4f_protocol_version=1,taco_ma4f_decode_reduce=0)
        symbols.update({name:0 for name in source.ZERO_COUNT_EXPORTS})
        with tempfile.TemporaryDirectory(prefix='ma12-exports-') as tmp:
            for arm in source.VARIANTS:
                path = Path(tmp)/(arm+'.so')
                definitions = dict(symbols)
                if arm == source.VARIANTS[1]:
                    definitions.update({name:0 for name in source.NATIVE_EXPORTS})
                compile_cpu_runtime(path,(self.root/arm/source.VERSION_SCRIPT).read_text(),definitions)
                library = ctypes.CDLL(str(path))
                self.assertTrue(all(hasattr(library,name) for name in source.ZERO_COUNT_EXPORTS))
                for name in source.NATIVE_EXPORTS:
                    self.assertEqual(hasattr(library,name),arm==source.VARIANTS[1])

    def test_rehashed_unknown_source_edit_is_rejected(self):
        manifest_path, record_path = self.root/'manifest.json',self.root/builder.MANIFEST
        for arm,name in ((source.VARIANTS[0],source.ZERO_COUNT_HEADER),(source.VARIANTS[1],source.READY_HEADER)):
            path = self.root/arm/name
            originals = {p:p.read_bytes() for p in (path,manifest_path,record_path)}
            try:
                path.write_bytes(originals[path]+b'\n// unexpected change\n')
                manifest,record = read(manifest_path),read(record_path)
                manifest['variants'][arm]['source_sha256'][name]=sha(path)
                record['variants'][arm][name]=sha(path)
                write(manifest_path,manifest)
                record['build_manifest_sha256']=sha(manifest_path)
                write(record_path,record)
                with self.assertRaisesRegex(ValueError,'registered MA12 transform'):
                    builder.verify(self.root)
            finally:
                for p,value in originals.items():p.write_bytes(value)

    def test_source_only_build_cannot_be_measured(self):
        with self.assertRaisesRegex(ValueError,'Missing MA12 runtime|Unbuilt'):
            builder.verify(self.root,built=True)


if __name__=='__main__':unittest.main()
