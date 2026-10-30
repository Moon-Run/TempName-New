"""Compile and run the actual host generator; CUDA headers, but no GPU required.

CUDA_HOME=/path/to/cuda CXX=/usr/bin/g++-11 python3 -m unittest discover \
    -s tools/h100/phase6 -p test_ablation6_generator.py -v
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from common import HERE, REPO


class GeneratorCompileTests(unittest.TestCase):
    def test_host_generator_builds_and_emits_only_native_sm90_bf16_registry(self):
        nvcc = shutil.which('nvcc')
        cuda = Path(os.environ.get('CUDA_HOME') or (str(Path(nvcc).resolve().parents[1]) if nvcc else '/usr/local/cuda'))
        cxx = shutil.which(os.environ.get('CXX', 'g++'))
        if not cxx or not (cuda/'include/cuda_runtime.h').is_file():
            self.skipTest('Set CUDA_HOME and CXX to run the real host-generator compile regression')
        with tempfile.TemporaryDirectory(prefix='ablation6-generator-test-') as tmp:
            root = Path(tmp); registers = root/'registers'; registers.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            cmd = [cxx, '-std=c++17', '-O2', str(HERE/'ablation6_gemm_only_generator.cc'), '-o', str(root/'generator')]
            cmd += ['-I'+str(p) for p in (REPO/'include', REPO/'src/generator',
                    REPO/'3rdparty/cutlass/include', REPO/'3rdparty/cutlass/tools/util/include', cuda/'include')]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            result = subprocess.run([str(root/'generator'), '--archs=90', '--sm_cores=132',
                '--dir='+str(registers), '--output_file='+str(root/'generated.txt')],
                env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            files = list(registers.glob('*.cu'))
            self.assertEqual(len(files), 1)
            self.assertEqual((root/'generated.txt').read_text().splitlines(), [str(files[0])])
            source = files[0].read_text()
            for part in ('bf16', 'sm90', 'h800', 'commnone', 'rcr'):
                self.assertIn(part, files[0].name.lower())
            self.assertEqual(source.count('using GemmDevice_'), 6)
            self.assertEqual(source.count('GemmV3CommNone_Device<'), 6)
            self.assertNotIn('GemmReduceScatter', source)
            self.assertNotIn('taco', source.lower())


if __name__ == '__main__':
    unittest.main()
