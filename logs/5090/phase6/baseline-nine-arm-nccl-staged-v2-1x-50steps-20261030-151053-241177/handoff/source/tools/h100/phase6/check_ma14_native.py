#!/usr/bin/env python3
"""Compile the actual MA14 CUDA runtime, Torch wrapper and binding without a GPU.

This is diagnostic evidence, never an executable experiment build. Use
run_ma14.sh to compile the full Flux extension and run the TP8 comparison.
"""
import argparse
import ctypes
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import ma4_f_build
import ma14_sources
from common import HERE, sha, write


class EpochConfig(ctypes.Structure):
    _fields_ = [('peers', ctypes.c_void_p * 8),
                ('epoch', ctypes.c_ulonglong), ('rank', ctypes.c_int)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda-home', type=Path, required=True)
    parser.add_argument('--torch-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    record = dict(completed=False, gpu_used=False, gpu_validated=False,
                  full_flux_linked=False, diagnostic_only=True, commands=commands,
                  inputs={name: sha(HERE / name) for name in
                          ('ma14_sources.py', 'ma14_epoch.cuh', 'ma14_pair.h',
                           'check_ma14_native.py')},
                  limitations=['Compilation and CPU ABI rejection checks only; no GPU execution.'])

    def run(name, command):
        command = list(map(str, command))
        commands.append(dict(name=name, argv=command))
        with (out / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        base = ma4_f_build.prepare(out / 'f4-reference') / 'F4'
        candidate = out / 'candidate'
        shutil.copytree(base, candidate, symlinks=True)
        changes = ma14_sources.sources(base, ma14_sources.VARIANTS[1])
        for name, text in changes.items():
            target = candidate / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        record['source_sha256'] = {name: sha(candidate / name) for name in changes}
        includes = ['-I' + str(candidate / p) for p in
                    ('src', 'include', '3rdparty/cutlass/include',
                     '3rdparty/cutlass/tools/util/include')]
        nvcc = args.cuda_home / 'bin/nvcc'
        flags = ['-std=c++17', '-O2', '--use_fast_math', '--expt-relaxed-constexpr',
                 '--expt-extended-lambda', '-arch=sm_90', '-Xptxas=-v,-warn-spills']
        library = out / 'runtime.so'
        run('runtime', [nvcc, *flags, *includes, '-Xcompiler=-fPIC', '--shared',
            candidate / ma14_sources.RUNTIME, '-Xlinker',
            '--version-script=' + str(candidate / ma14_sources.VERSION_SCRIPT), '-o', library])
        run('runtime-ptx', [nvcc, *flags, *includes, '--ptx',
            candidate / ma14_sources.RUNTIME, '-o', out / 'runtime.ptx'])
        info = json.loads(subprocess.check_output([str(args.torch_python), '-c',
            'import json,sysconfig,torch;from torch.utils.cpp_extension import include_paths;'
            'print(json.dumps(dict(includes=include_paths()+[sysconfig.get_path("include")],'
            'abi=int(torch._C._GLIBCXX_USE_CXX11_ABI),torch=torch.__version__,'
            'cuda=torch.version.cuda,device_count=torch.cuda.device_count())))'],
            text=True).splitlines()[-1])
        record['torch_environment'] = info
        host = [args.cxx, '-std=c++17', '-O0', '-fPIC', '-DFLUX_TACO_BASELINE',
                '-D_GLIBCXX_USE_CXX11_ABI=' + str(info['abi']), *includes,
                '-I' + str(args.cuda_home / 'include'),
                *['-I' + p for p in info['includes']]]
        for name, source in (('wrapper', ma14_sources.WRAPPER), ('pybind', ma14_sources.PYBIND)):
            run(name, [*host, '-c', candidate / source, '-o', out / (name + '.o')])
        exports = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
        (out / 'exports.txt').write_text(exports)
        names = {line.split()[-1] for line in exports.splitlines()}
        required = ma14_sources.NATIVE_EXPORTS + ma14_sources.ZERO_COUNT_EXPORTS
        if not set(required) <= names:
            raise RuntimeError('Missing diagnostic runtime exports: ' + str(set(required) - names))
        ctypes.CDLL(str(args.cuda_home / 'lib64/libcudart.so'), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(str(library))
        lib.taco_ma14_epoch_protocol.argtypes = []
        lib.taco_ma14_epoch_protocol.restype = ctypes.c_int
        if lib.taco_ma14_epoch_protocol() != ma14_sources.NATIVE_PROTOCOL:
            raise RuntimeError('Wrong MA14 native protocol')
        barrier = lib.taco_ma14_epoch_barrier
        barrier.argtypes, barrier.restype = [EpochConfig, ctypes.c_void_p], ctypes.c_int
        rejected = []
        # These synthetic pointers are never dereferenced: every call must be
        # rejected by the native argument checks before any CUDA API or launch.
        for label, epoch, rank, peer, pointer in (
                ('zero_epoch', 0, 0, None, None),
                ('overflow', 2**63, 0, None, None),
                ('negative_rank', 1, -1, None, None),
                ('rank_out_of_bounds', 1, 8, None, None),
                ('null_peer', 1, 0, 7, None),
                ('unaligned_peer', 1, 0, 3, 4097)):
            config = EpochConfig((ctypes.c_void_p * 8)(*[4096 + 64*i for i in range(8)]), epoch, rank)
            if peer is not None:
                config.peers[peer] = pointer
            if barrier(config, None) != 1:
                raise RuntimeError('Invalid native config accepted: ' + label)
            rejected.append(label)
        ptx = (out / 'runtime.ptx').read_text()
        match = re.search(r'\.visible \.entry [^\n]*taco_ma14_epoch_barrier_kernel', ptx)
        if match is None:
            raise RuntimeError('MA14 barrier kernel missing from PTX')
        body = ptx[match.start():]
        end = body.find('\n}', body.index('{'))
        body = body[:end+2]
        if ('st.release.sys.global.u64' not in body or 'ld.acquire.sys.global.u64' not in body
                or 'atom.' in body or 'red.' in body):
            raise RuntimeError('Compiled epoch protocol differs from store/acquire design')
        record.update(completed=True, native_protocol=ma14_sources.NATIVE_PROTOCOL, rejected_configs=rejected,
                      system_release_acquire_ptx=True, barrier_has_no_rmw=True,
                      checks=['actual_runtime_link', 'actual_torch_wrapper', 'actual_pybind_object',
                              'native_exports', 'invalid_abi_rejection', 'epoch_ptx'])
    except BaseException as error:
        record['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        record['log_sha256'] = {p.name: sha(p) for p in out.glob('*.log')}
        write(out / 'native-validation.json', record)
    print(out / 'native-validation.json')


if __name__ == '__main__':
    main()
