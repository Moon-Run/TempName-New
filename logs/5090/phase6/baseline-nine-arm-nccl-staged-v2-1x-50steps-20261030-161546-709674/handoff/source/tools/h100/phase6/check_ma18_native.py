#!/usr/bin/env python3
"""Build the actual standalone CUDA TU and check its non-launching ABI guards."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-home', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('ma18_ipc_allgather.cu')
    library = args.out / 'libma18_native.so'
    command = [str(args.cuda_home / 'bin/nvcc'), '-shared', '-std=c++17', '-Xcompiler', '-fPIC',
               str(source), '-o', str(library), '-gencode', 'arch=compute_90,code=sm_90']
    result = subprocess.run(command, capture_output=True, text=True)
    (args.out / 'compile.log').write_text(result.stdout + result.stderr)
    result.check_returncode()
    loaded = ctypes.CDLL(str(library.resolve()))
    loaded.ma18_ipc_stage_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    loaded.ma18_ipc_pull_cuda.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p]
    assert loaded.ma18_ipc_protocol() == 18
    cases = [loaded.ma18_ipc_stage_cuda(None, None, 2097152, None),
             loaded.ma18_ipc_stage_cuda(1, 1, 2097152, None),
             loaded.ma18_ipc_stage_cuda(1, 2, 1, None),
             loaded.ma18_ipc_pull_cuda(None, None, 0, 2097152, None),
             loaded.ma18_ipc_pull_cuda(1, None, -1, 2097152, None),
             loaded.ma18_ipc_pull_cuda(1, (ctypes.c_void_p * 8)(), 0, 2097152, None)]
    assert all(status != 0 for status in cases)
    report = dict(passed=True, command=command, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  invalid_argument_cases=len(cases), gpu_calls_executed=False,
                  full_flux_build=False, performance_measured=False)
    (args.out / 'native-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
