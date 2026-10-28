"""Source provenance, scratch ownership and original-order baseline checks."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import a_baseline_build as builder
import a_baseline_sources as transform
from common import HERE, REPO, read, sha, write


def frozen_text(name):
    text = (HERE/'fused_v2'/name).read_text()
    if name == transform.WRAPPER:
        return builder.build.wrapper(text, legacy_fused=True)
    if name.endswith('.py'):
        return builder.build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)').replace(
            'supports SM80 only', 'requires the H100 SM90 port').replace(
            'tools/a800/phase4/build.py', 'tools/h100/phase6/build.py')
    return text


def fake_prepare(out, config_path, only_variants):
    """A CPU fixture containing the actual frozen receiver/producer source."""
    assert only_variants == ['taco_fused']
    out = Path(out)
    src = out/'taco_fused'
    src.mkdir(parents=True)
    files = read(HERE/'fused_v2/provenance.json')['files']
    for name in files:
        dest = src/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(frozen_text(name))
    for name, text in {
        builder.SWIZZLE: (REPO/builder.SWIZZLE).read_text(),
        'check_mapping.cu': (HERE/'check_mapping.cu').read_text(),
        'phase6_host.hpp': '#pragma once\ninline int phase6_host_lookup(int,int,int,int,int){return -1;}\n',
    }.items():
        path = src/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    hashes = {str(p.relative_to(src)): sha(p) for p in src.rglob('*') if p.is_file()}
    write(out/'manifest.json', dict(protocol=builder.build.PROTOCOL, arch='sm90',
        config=read(config_path), source_commit='cpu-fixture', submodules={},
        variants={'taco_fused': dict(role='fused', base='original', sampling=False,
            implementation='a800-fused-v2-cta-v1', python='taco_fused/python', source_sha256=hashes)}))
    return out


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = mock.patch.object(builder.build, 'prepare', side_effect=fake_prepare)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def prepare(self):
        return builder.prepare(self.root/'build', source_only=True)

    def test_basic_is_only_receiver_materialization_and_retains_frozen_producer(self):
        root = self.prepare()
        record = builder.verify(root)
        expected = {transform.RUNTIME, transform.HEADER, transform.WRAPPER, transform.DECODER}
        self.assertEqual(set(record['transformed_files']['basic']), expected)
        self.assertEqual(record['transformed_files']['improved'], [])
        unchanged = set(record['variants']['improved']) - expected
        for name in unchanged:
            self.assertEqual(record['variants']['basic'][name], record['variants']['improved'][name])
        for arm in builder.VARIANTS:
            self.assertEqual((root/arm/builder.SWIZZLE).read_bytes(), (REPO/builder.SWIZZLE).read_bytes())
        self.assertEqual(builder.runtime_root(root, 'basic'), root/'basic')
        with self.assertRaisesRegex(ValueError, 'Expected basic or improved'):
            builder.runtime_root(root, 'cuda')

    def test_materialized_intermediate_is_per_call_and_preserves_barriers(self):
        root = self.prepare()
        old = (root/'improved'/transform.WRAPPER).read_text()
        new = (root/'basic'/transform.WRAPPER).read_text()
        self.assertEqual(old.count('group_barrier.barrier_all(stream);'), new.count('group_barrier.barrier_all(stream);'))
        self.assertIn('torch::empty({taco.world - 1, taco.m / taco.world, taco.n}', new)
        self.assertIn('decoded.data_ptr(), output.data_ptr(), stream)', new)
        self.assertEqual((root/'basic/python/flux/gemm_rs_taco.py').read_bytes(),
                         (root/'improved/python/flux/gemm_rs_taco.py').read_bytes())
        runtime = (root/'basic'/transform.RUNTIME).read_text()
        self.assertNotIn('taco_decode_ring_kernel', runtime)
        self.assertIn('if (status != cudaSuccess) return int(status);', runtime)
        self.assertNotIn('cudaMalloc', runtime)
        self.assertNotIn('cudaDeviceSynchronize', runtime)
        self.assertEqual(7*256*4096*2, 14680064)

    def test_source_fixture_cannot_be_admitted_as_runnable(self):
        root = self.prepare()
        with self.assertRaisesRegex(ValueError, 'unbuilt, source-only'):
            builder.verify(root, built=True)
        with self.assertRaisesRegex(ValueError, 'source-only fixture'):
            builder.compile_build(root, '/does/not/exist')

    def test_changed_source_is_rejected_even_when_manifest_digests_are_resealed(self):
        root = self.prepare()
        path = root/'basic'/transform.RUNTIME
        path.write_text(path.read_text()+'\n// unregistered change\n')
        record, info = read(root/builder.MANIFEST), read(root/'manifest.json')
        info['variants']['basic']['source_sha256'][transform.RUNTIME] = sha(path)
        record['variants']['basic'][transform.RUNTIME] = sha(path)
        write(root/'manifest.json', info)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/builder.MANIFEST, record)
        with self.assertRaisesRegex(ValueError, 'registered basic-fusion transformation'):
            builder.verify(root)

    def test_mapping_checker_uses_base_coordinates_instead_of_interleaved_formula(self):
        root = self.prepare()
        for arm in builder.VARIANTS:
            checker = (root/arm/'check_mapping.cu').read_text()
            self.assertIn('x != (bx + tm / tp * rank) % tm || y != by', checker)
            self.assertIn('k/tp', checker)
            self.assertEqual((root/arm/'phase6_host.hpp').read_text(),
                             '#pragma once\ninline int phase6_host_lookup(int,int,int,int,int){return -1;}\n')


if __name__ == '__main__':
    unittest.main()
