#!/usr/bin/env python3
"""Compile real SM90 publication TU, test its ABI guards, optionally GPU oracle."""
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
    parser.add_argument('--gpu', action='store_true', help='Run exact CUDA copy_ oracle; requires CUDA PyTorch')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('ma19_param_copy.cu')
    library = args.out / 'libma19_native.so'
    command = [str(args.cuda_home / 'bin/nvcc'), '-shared', '-std=c++17', '-Xcompiler', '-fPIC',
               '--ftz=false', '--prec-div=true', '--prec-sqrt=true',
               '-Xptxas=-v', '-gencode', 'arch=compute_90,code=sm_90',
               str(source), '-o', str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    (args.out / 'compile.log').write_text(result.stdout + result.stderr)
    result.check_returncode()
    loaded = ctypes.CDLL(str(library.resolve()))
    loaded.ma19_copy_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    assert loaded.ma19_copy_protocol() == 19
    assert loaded.ma19_copy_chunk_size() == 32768
    cases = [loaded.ma19_copy_cuda(None, None, 1, None),
             loaded.ma19_copy_cuda(1, None, 1, None),
             loaded.ma19_copy_cuda(None, 1, 1, None),
             loaded.ma19_copy_cuda(1, 2, 0, None),
             loaded.ma19_copy_cuda(1, 2, -1, None)]
    assert all(code != 0 for code in cases)
    oracle = None
    if args.gpu:
        import torch
        from ma19_adapter import native_self_test
        if not torch.cuda.is_available():
            raise RuntimeError('--gpu requested but CUDA is unavailable')
        oracle = native_self_test(torch, loaded)
    report = dict(passed=True, command=command,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  invalid_argument_cases=len(cases), gpu_calls_executed=args.gpu,
                  gpu_oracle=oracle, full_flux_build=False, performance_measured=False)
    (args.out / 'native-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
