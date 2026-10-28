"""Clone immutable candidates and recompile their changed runtime/C++ objects.

Unlike rebuild_runtime.py this rebuilds both sides of the private TACO ABI.
The GEMM registration is rebuilt too when an epilogue or codec header changes.
Baseline libraries and the parent's frozen sources are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    repo = Path(__file__).resolve().parents[3]
    base, out = args.base.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for policy in args.policies:
        old, new = base / policy, out / policy
        manifest = json.loads((old / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            assert sha(old / name) == digest, name
        shutil.copytree(old, new, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        registration_changed = False
        for name in ('taco_runtime.cu', 'taco_runtime.h', 'taco_codec.cuh',
                     'epilogue_evt.hpp', 'ths_op/gemm_reduce_scatter.cc'):
            src = repo / 'src/gemm_rs' / name
            dest = new / 'taco/overlay/gemm_rs' / name
            if name in ('taco_runtime.h', 'taco_codec.cuh', 'epilogue_evt.hpp'):
                registration_changed |= sha(src) != sha(dest)
            shutil.copy2(src, dest)
            manifest['inputs'][str(src)] = sha(src)
        src = repo / 'python/flux/gemm_rs_taco.py'
        shutil.copy2(src, new / 'taco/python/flux/gemm_rs_taco.py')
        manifest['inputs'][str(src)] = sha(src)
        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            link = new / 'taco/python/flux/lib' / name
            link.unlink()
            link.symlink_to(new / 'taco' / name)
        commands = [dict(args=[v.replace(str(old), str(new)) for v in c['args']],
                         cwd=c['cwd'].replace(str(old), str(new)))
                    for c in manifest['commands']]
        targets = {'taco_runtime.o', 'device_link.o', 'gemm_reduce_scatter.cc.o',
                   'libflux_cuda.so', 'libflux_cuda_ths_op.so'}
        if registration_changed:
            targets.add('registration.o')
        for command in commands:
            argv = command['args']
            if '-o' in argv and Path(argv[argv.index('-o') + 1]).name in targets:
                print('BUILD', policy, Path(argv[argv.index('-o') + 1]).name, flush=True)
                subprocess.run(argv, cwd=command['cwd'], check=True)
        manifest['commands'] = commands
        manifest['candidate_parent_manifest'] = dict(path=str(old / 'manifest.json'),
                                                     sha256=sha(old / 'manifest.json'))
        manifest['inputs'][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
        manifest['files'] = {str(p.relative_to(new)): sha(p) for p in new.rglob('*')
                             if p.is_file() and not p.is_symlink() and
                             p.name not in ('manifest.json', 'build-progress.json')}
        (new / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('CANDIDATE REBUILD COMPLETE', policy, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--policies', nargs='+',
                        default=['remote_arrival', 'interleaved_arrival'],
                        choices=['remote_arrival', 'interleaved_arrival',
                                 'interleaved_remote_arrival', 'interleaved_remote_group_arrival'])
    main(parser.parse_args())
