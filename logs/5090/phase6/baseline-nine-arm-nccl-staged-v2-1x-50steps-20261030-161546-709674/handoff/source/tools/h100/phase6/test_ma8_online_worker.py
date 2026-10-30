"""Exercise the real frozen worker's online checks before its first Torch import.

All fixtures live in /tmp. Library files are clearly marked CPU placeholders:
they are hashed, never loaded, and are not GPU build or performance evidence.
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

import ma8_bench as bench
import ma8_build as builder
from common import sha, read, write


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
os.environ['H100_MA8_BENCH_EXECUTE'] = '1'
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
        cls.temp = tempfile.TemporaryDirectory(prefix='ma8-online-cpu-', dir='/tmp')
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
        # outputs are placeholders, and the worker stops before importing Torch.
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

    def test_builder_input_digest_drift_fails_before_gpu(self):
        path = self.build/builder.MANIFEST
        record = read(path)
        key = str(builder.HERE/'ma8_build.py')
        self.assertIn(key, record['inputs'])
        record['inputs'][key] = '0'*64
        self.reject_mutation(path, json.dumps(record).encode(), 'ma8_build.py')

    def test_native_source_drift_fails_before_gpu(self):
        path = self.build/'F4-cache'/builder.RUNTIME
        self.reject_mutation(path, path.read_bytes()+b'\n// unexpected native source edit\n',
                             'taco_runtime.cu')

    def test_linked_library_drift_fails_before_gpu(self):
        path = self.build/'F4-cache/python/flux/lib/libflux_cuda.so'
        self.reject_mutation(path, path.read_bytes()+b'\nchanged runtime\n', 'libflux_cuda.so')



if __name__ == '__main__':
    unittest.main()
