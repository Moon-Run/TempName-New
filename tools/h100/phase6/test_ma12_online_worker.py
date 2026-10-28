"""Exercise the real frozen worker's online checks before its first Torch import.

All fixtures live in /tmp. The runtime is a host-only ELF stub so real export
checks execute; other artifacts are labeled placeholders. No GPU code is loaded.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import ma12_bench as bench
import ma12_build as builder
from common import sha, read, write
from test_native_exports import compile_cpu_runtime


WORKER_ENTRY = r'''
import argparse, builtins, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'scripts'))
import bench
import common
frozen_common = common
identity = (common.__file__, common.HERE, common.REPO)
assert Path(common.__file__).resolve() == root/'scripts/common.py'
assert common.HERE == root/'scripts'
assert 'torch' not in sys.modules

class GPUBoundary(Exception):
    pass

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise GPUBoundary('Online checks passed; stopping before any Torch/GPU use')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
os.environ['H100_MA12_BENCH_EXECUTE'] = '1'
records = []
for arm in json.loads(sys.argv[2]):
    job = next(j for j in bench.stage_jobs('preflight') if j['arm'] == arm)
    args = argparse.Namespace(root=root, job=json.dumps(job), preflight=True, profile=False)
    try:
        bench.worker(args)
    except GPUBoundary:
        records.append(dict(arm=arm, status='gpu_boundary'))
    except Exception as error:
        records.append(dict(arm=arm, status='rejected', error=type(error).__name__+': '+str(error)))
    else:
        raise AssertionError('Worker unexpectedly returned before its Torch boundary')
    assert sys.modules['common'] is frozen_common
    assert identity == (common.__file__, common.HERE, common.REPO)
    assert 'torch' not in sys.modules
print(json.dumps(dict(records=records, common_unchanged=True, torch_imported=False)))
'''


class FrozenOnlineWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma12-online-cpu-', dir='/tmp')
        cls.base = Path(cls.temp.name)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.build = builder.prepare(cls.base/'build')
        placeholder = b'CPU TEST PLACEHOLDER: NOT A GPU LIBRARY OR EXECUTABLE\n'
        (cls.base/'CPU_FIXTURE_ONLY.txt').write_bytes(placeholder)
        for arm in bench.ARMS:
            root = cls.build/arm
            library = root/'python/flux/lib'
            library.mkdir(parents=True, exist_ok=True)
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                if name == 'libflux_cuda.so':
                    symbols = dict(taco_ma4f_protocol_version=1, taco_ma4f_decode_reduce=0)
                    symbols.update({name: 0 for name in builder.ma12_sources.ZERO_COUNT_EXPORTS})
                    if arm == 'F4-cache-zcount-tilepub':
                        symbols.update({name: 0 for name in builder.ma12_sources.NATIVE_EXPORTS})
                    compile_cpu_runtime(root/name,
                        (root/builder.ma12_sources.VERSION_SCRIPT).read_text(), symbols)
                    shutil.copy2(root/name, library/name)
                else:
                    (root/name).write_bytes(placeholder+arm.encode()+name.encode())
                    (library/name).write_bytes(placeholder+arm.encode()+name.encode())
            (root/'check_mapping').write_bytes(placeholder)
            (root/'python/flux_ths_pybind_cpu_fixture.so').write_bytes(placeholder)
        manifest = read(cls.build/'manifest.json')
        manifest['status'] = 'built'
        for arm in bench.ARMS:
            root = cls.build/arm
            for path in [root/'check_mapping', root/'python/flux/lib/libflux_cuda.so',
                         root/'python/flux/lib/libflux_cuda_ths_op.so',
                         root/'python/flux_ths_pybind_cpu_fixture.so']:
                manifest['files'][str(path.relative_to(cls.build))] = sha(path)
        write(cls.build/'manifest.json', manifest)
        record = read(cls.build/builder.MANIFEST)
        record['build_manifest_sha256'] = sha(cls.build/'manifest.json')
        write(cls.build/builder.MANIFEST, record)
        # Actual source validation is intentionally retained. Only executable
        # outputs are host stubs/placeholders; the worker stops before importing Torch.
        builder.verify(cls.build, built=True)
        cls.root = cls.base/'operator'
        args = argparse.Namespace(root=cls.root, build=cls.build, blocks=8,
                                  warmup=1, iterations=2)
        with contextlib.redirect_stdout(io.StringIO()):
            bench.prepare(args)

    def invoke(self, root=None, arms=None):
        root = root or self.root
        process = subprocess.run([sys.executable, '-I', '-c', WORKER_ENTRY,
                                  str(root), json.dumps(arms or list(bench.ARMS))],
                                 cwd=root/'scripts', capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertTrue(result['common_unchanged'])
        self.assertFalse(result['torch_imported'])
        return result['records']

    def reject_mutation(self, path, replacement, expected):
        saved = path.read_bytes()
        try:
            path.write_bytes(replacement)
            rows = self.invoke()
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row['status'], 'rejected', row)
                self.assertIn(expected, row['error'])
        finally:
            path.write_bytes(saved)

    def test_actual_frozen_worker_reaches_gpu_boundary_for_every_arm(self):
        self.assertEqual(self.invoke(), [dict(arm=arm, status='gpu_boundary') for arm in bench.ARMS])

    def test_frozen_dependencies_keep_original_implementations(self):
        plan = read(self.root/'plan.json')
        for original, frozen in ((bench.DRIVER, 'ma12_bench_driver.py'),
                                  (bench.HERE/'ma8_adapter.py', 'ma8_adapter.py'),
                                  (bench.HERE/'ma9_adapter.py', 'ma9_adapter.py'),
                                  (bench.HERE/'ma9_sources.py', 'ma9_sources.py'),
                                  (bench.HERE/'ma12_profile.py', 'ma12_profile.py')):
            self.assertEqual((self.root/'scripts'/frozen).read_bytes(), original.read_bytes())
            self.assertEqual(plan['source_files']['scripts/'+frozen], sha(original))
        self.assertEqual(plan['activation_policy'], 'retain_owned_column_input_uninstrumented')
        self.assertEqual(plan['zero_count_policy'], 'fused_optimizer_zero_count')

    def test_frozen_driver_mutation_is_rejected_during_import(self):
        path = self.root/'scripts/ma12_bench_driver.py'
        saved = path.read_bytes()
        try:
            path.write_bytes(saved+b'\n# invalid modified driver\n')
            process = subprocess.run([sys.executable, '-I', '-c', WORKER_ENTRY,
                                      str(self.root), json.dumps(list(bench.ARMS))],
                                     cwd=self.root/'scripts', capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('registered unchanged MA9 controller', process.stderr)
        finally:
            path.write_bytes(saved)

    def test_only_two_requested_arms_are_available(self):
        self.assertEqual(bench.ARMS, ('F4-cache-zcount', 'F4-cache-zcount-tilepub'))
        self.assertEqual({j['arm'] for j in bench.schedule()}, set(bench.ARMS))
        for arm in ('F4', 'F4-zcount', 'CR2'):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                bench.family(arm)

    def test_builder_input_digest_drift_fails_before_gpu(self):
        path = self.build/builder.MANIFEST
        record = read(path)
        key = str(builder.HERE/'ma12_build.py')
        self.assertIn(key, record['inputs'])
        record['inputs'][key] = '0'*64
        self.reject_mutation(path, json.dumps(record).encode(), 'ma12_build.py')

    def test_native_source_drift_fails_before_gpu(self):
        path = self.build/'F4-cache-zcount-tilepub'/builder.ma12_sources.READY_HEADER
        self.reject_mutation(path, path.read_bytes()+b'\n// unexpected native source edit\n',
                             'ma12_tile_ready.cuh')

    def test_rehashed_library_missing_exports_is_rejected_before_gpu(self):
        source = builder.ma12_sources
        library = self.build/'F4-cache-zcount-tilepub'/'python/flux/lib/libflux_cuda.so'
        manifest_path, record_path = self.build/'manifest.json', self.build/builder.MANIFEST
        saved = {path: path.read_bytes() for path in (library, manifest_path, record_path)}
        try:
            # Reproduce an old library lacking MA12 definitions, while preserving
            # all registered sources. Its generic taco* export pattern alone
            # cannot manufacture the absent ABI. Rehashing must not bypass it.
            symbols = dict(taco_ma4f_protocol_version=1, taco_ma4f_decode_reduce=0)
            symbols.update({name: 0 for name in source.ZERO_COUNT_EXPORTS})
            compile_cpu_runtime(library, (self.build/'F4-cache-zcount'/source.VERSION_SCRIPT).read_text(), symbols)
            manifest, record = read(manifest_path), read(record_path)
            manifest['files'][str(library.relative_to(self.build))] = sha(library)
            write(manifest_path, manifest)
            record['build_manifest_sha256'] = sha(manifest_path)
            write(record_path, record)
            with self.assertRaisesRegex(ValueError, 'Missing exported native functions'):
                builder.verify(self.build, built=True)
            for row in self.invoke():
                self.assertEqual(row['status'], 'rejected', row)
                self.assertIn('Missing exported native functions', row['error'])
        finally:
            for path, data in saved.items():
                path.write_bytes(data)

    def test_linked_library_drift_fails_before_gpu(self):
        path = self.build/'F4-cache-zcount-tilepub/python/flux/lib/libflux_cuda.so'
        self.reject_mutation(path, path.read_bytes()+b'\nchanged runtime\n', 'libflux_cuda.so')



if __name__ == '__main__':
    unittest.main()
