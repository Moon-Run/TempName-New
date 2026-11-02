"""CPU fixtures validate the read-only legacy overlay, never GPU compatibility."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import open_compress_build as b
import open_compress_remote_first as remote

DEPENDENCIES = {name: dict(commit=char*40, tracked_clean=True)
                for name, char in (('cutlass', 'a'), ('nccl', 'b'))}
ENVIRONMENT = dict(python='/CPU/python', torch='fixture', cuda='12.9', cxx11_abi=True, nvcc_sha256='c'*64)
ORIGINAL_REGISTRATION = 'struct flux_cpu_streamksk_: public GemmKernel_0 {};\nusing Device = flux_cpu_streamksk_;\n'


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def legacy_fixture(root):
    root.mkdir()
    record = dict(protocol=b.LEGACY_PROTOCOL, actual_arch='sm120', arms=list(b.LEGACY_ARMS),
        variants={}, inputs=dict(b.LEGACY_INPUTS), status='built', gpu_validated=False,
        submodules=copy.deepcopy(DEPENDENCIES), environment=dict(ENVIRONMENT), files={}, registrations={},
        taco_build_required=False, taco_runtime_required=False, peer_memory_required=False,
        codec_source_sha256=b.LEGACY_INPUTS['open_compress_codec.cu'],
        megatron=dict(commit=b.MEGATRON_COMMIT, archive_sha256='d'*64, paths=[]))
    for name in b.LEGACY_VARIANTS:
        source = root/name
        hashes = {}
        for relative in (remote.KERNEL, 'src/cuda/version.ld'):
            path = put(source/relative, (b.REPO/relative).read_text())
            hashes[relative] = b.sha(path)
        record['variants'][name] = dict(source_sha256=hashes, modified_files=[])
        (source/'3rdparty').mkdir()
        (source/'3rdparty/cutlass').symlink_to(b.REPO/'3rdparty/cutlass', target_is_directory=True)
        registration = put(source/'build/src/comm_none/registers/fixture_sm89.cu',
            b.op1_a_swizzle.registration_source(ORIGINAL_REGISTRATION) if name == 'flux-interleaved' else ORIGINAL_REGISTRATION)
        record['registrations'][name] = dict(path=str(registration.relative_to(root)), sha256=b.sha(registration),
            generated_sha256=hashlib.sha256(ORIGINAL_REGISTRATION.encode()).hexdigest())
        paths = [registration, put(source/'python/flux_ths_pybind.fixture.so', b'CPU fixture pybind')]
        for library in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            paths.append(put(source/'python/flux/lib'/library, b'CPU fixture native'))
            (source/library).symlink_to(Path('python/flux/lib')/library)
            paths.append(source/library)
        record['files'].update({str(path.relative_to(root)): b.sha(path) for path in paths})
    put(root/'codec/open_compress_codec.cu', (b.HERE/'open_compress_codec.cu').read_bytes())
    paths = [put(root/'codec/libopen_compress.so', b'CPU fixture codec'),
             put(root/'megatron/megatron/core/datasets/helpers.fixture.so', b'CPU fixture dataset')]
    record['files'].update({str(path.relative_to(root)): b.sha(path) for path in paths})
    nccl = root/'nccl-source/build'
    paths = [put(nccl/'include/nccl.h', 'fixture header'), put(nccl/'lib/libnccl_static.a', b'CPU archive')]
    record['nccl_files'] = {str(path): b.sha(path) for path in paths}
    b.write(root/b.MANIFEST, record)
    return record


def file_hashes(root):
    return {str(path.relative_to(root)): b.sha(path) for path in root.rglob('*') if path.is_file() and not path.is_symlink()}


class RemoteFirst(unittest.TestCase):
    def test_every_remote_tile_precedes_local_for_every_rank(self):
        for rank in range(8):
            values = remote.mapping(rank)
            self.assertEqual(set(values), {(m, n) for m in range(16) for n in range(32)})
            self.assertEqual([m//2 for m, n in values[:448]],
                             [((rank+offset)%8) for offset in range(1, 8) for _ in range(64)])
            self.assertEqual([m//2 for m, n in values[448:]], [rank]*64)
        for rank in (True, -1, 8, 1.5):
            with self.assertRaises(ValueError): remote.mapping(rank)

    def test_real_kernel_override_and_registration_are_reversible(self):
        updates = remote.overrides(b.REPO)
        self.assertIn('OpenCompressRemoteFirstSwizzle', updates[remote.KERNEL])
        self.assertIn('(op1_rank + 1 + tile_index / 64) % 8', updates[remote.HEADER])
        self.assertIn('tile_index % 64', updates[remote.HEADER])
        self.assertIn('oc_remote_first_mapping', updates[remote.PROBE])
        self.assertIn('oc_remote_first_*;', updates['src/cuda/version.ld'])
        transformed = remote.registration_source(ORIGINAL_REGISTRATION)
        self.assertEqual(remote.original_registration(transformed), ORIGINAL_REGISTRATION)
        with self.assertRaises(ValueError): remote.original_registration(transformed+'// changed')

    def test_megatron_never_exposes_a_flux_library_path(self):
        for arm in ('megatron-bf16', 'megatron-int8'):
            self.assertIsNone(b.native_root('/tmp/unused', arm))
            self.assertEqual(b.libraries('/tmp/unused', arm), {})
        self.assertEqual(b.native_root('/tmp/build', 'flux-int8-remote-first'), Path('/tmp/build/flux-remote-first'))


class LegacyOverlay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root/'old'
        self.manifest = legacy_fixture(self.old)
        self.dependencies = patch.object(b, 'dependency_identity', return_value=copy.deepcopy(DEPENDENCIES))
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)
        self.exports = patch('native_exports.verify_native_exports')
        self.exports.start()
        self.addCleanup(self.exports.stop)

    def test_only_the_exact_old_recipe_can_be_admitted(self):
        b.verify(self.old, True)
        self.assertFalse(b.has_remote_first(self.old))
        for key in ('open_compress_build.py', 'open_compress_codec.py', 'open_compress_codec.cu'):
            bad = copy.deepcopy(self.manifest)
            bad['inputs'][key] = '0'*64
            b.write(self.old/b.MANIFEST, bad)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Unknown legacy'):
                b.verify(self.old, True)
        b.write(self.old/b.MANIFEST, self.manifest)
        with patch.object(b, 'inputs', return_value=dict(b.inputs(), **{'gemm_only_generator.cc':'0'*64})):
            with self.assertRaisesRegex(ValueError, 'Legacy CUDA/generator/swizzle'):
                b.verify(self.old, True)

    def test_old_source_binary_and_registration_hashes_are_still_checked(self):
        targets = [self.old/'flux-original'/remote.KERNEL, self.old/'codec/libopen_compress.so',
                   self.old/self.manifest['registrations']['flux-interleaved']['path'],
                   self.old/'megatron/megatron/core/datasets/helpers.fixture.so']
        for path in targets:
            original = path.read_bytes()
            path.write_bytes(original+b'changed')
            with self.subTest(path=path), self.assertRaises(ValueError): b.verify(self.old, True)
            path.write_bytes(original)

    def test_prepare_does_not_mutate_parent_and_links_only_verified_resources(self):
        before = file_hashes(self.old)
        new = self.root/'overlay'
        b.prepare(new, reference=self.old)
        self.assertTrue(b.has_remote_first(new))
        self.assertEqual(file_hashes(self.old), before)
        for name in (*b.LEGACY_VARIANTS, 'codec', 'megatron'):
            self.assertTrue((new/name).is_symlink())
            self.assertEqual((new/name).resolve(), (self.old/name).resolve())
        self.assertFalse((new/'flux-remote-first').is_symlink())
        self.assertIn('OpenCompressRemoteFirstSwizzle', (new/'flux-remote-first'/remote.KERNEL).read_text())
        b.verify(new)
        link = new/'codec'
        link.unlink()
        link.symlink_to(self.old/'megatron', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'inherited build link changed'): b.verify(new)

    def test_parent_manifest_change_invalidates_overlay(self):
        new = self.root/'overlay'
        b.prepare(new, reference=self.old)
        changed = dict(self.manifest, unexpected='changed')
        b.write(self.old/b.MANIFEST, changed)
        with self.assertRaisesRegex(ValueError, 'Parent build identity changed'): b.verify(new)

    def test_overlay_compile_only_builds_remote_variant(self):
        new = self.root/'overlay'
        b.prepare(new, reference=self.old)
        before = file_hashes(self.old)
        commands = []
        def command(args, **kwargs):
            commands.append(args)
            source = new/'flux-remote-first'
            if args[0] == 'cmake' and args[1] == '-S':
                put(source/'build/src/comm_none/registers/fixture_sm89.cu', ORIGINAL_REGISTRATION)
            if args[0] == 'cmake' and args[1] == '--install':
                for library in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                    put(source/'python/flux/lib'/library, b'new CPU library')
            if len(args) > 1 and args[1] == 'setup.py':
                put(source/'python/flux_ths_pybind.fixture.so', b'new CPU pybind')
        with patch.object(b, 'environment_identity', return_value=dict(ENVIRONMENT)), \
             patch.object(b.subprocess, 'run', side_effect=command), \
             patch.object(b, 'megatron_check', side_effect=AssertionError('external Megatron consulted')), \
             patch.object(b, 'archive_megatron', side_effect=AssertionError('parent Megatron rebuilt')):
            record = b.compile_build(new, self.root/'missing-megatron', self.root/'cuda')
        self.assertEqual(record['status'], 'built')
        self.assertEqual(len(commands), 4)
        self.assertTrue(all('flux-remote-first' in ' '.join(command) or 'setup.py' in command for command in commands))
        self.assertFalse(any('nvcc' in command[0] or command[0] == 'make' for command in commands))
        self.assertEqual(file_hashes(self.old), before)
        b.verify(new, True)

    def test_overlay_cannot_cross_python_cuda_abi(self):
        new = self.root/'overlay'
        b.prepare(new, reference=self.old)
        with patch.object(b, 'environment_identity', return_value=dict(ENVIRONMENT, torch='different')):
            with self.assertRaisesRegex(ValueError, 'same Python/CUDA ABI'):
                b.compile_build(new, self.root/'missing', self.root/'cuda')
        self.assertEqual(b.read(new/b.MANIFEST)['status'], 'prepared')


if __name__ == '__main__':
    unittest.main()
