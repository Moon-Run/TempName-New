"""Build isolated A/W/P libraries; only W/P receiver scheduling differs from A."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess

import build
import scope_build
import ma1_sources
from arrival_op3_v2_build import _baseline_metadata, _copy_variant, _digest, _identity_entries
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-ma1-source-prefetch-v1'
MANIFEST = 'ma1-manifest.json'
VARIANTS = ('A', 'W', 'P')
RUNTIME = ma1_sources.RUNTIME
INPUTS = ('ma1_build.py', 'ma1_sources.py', 'ma1_decode.cuh', 'arrival_op3_v2_build.py',
          'build.py', 'scope_build.py', 'scope_plan.py', 'scope_protocol.py', 'common.py')


def prepare(out, baseline=None):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh MA1 build directory')
    if baseline is not None:
        baseline = Path(baseline).resolve()
        _baseline_metadata(baseline)
    out.mkdir(parents=True)
    write(out/'config.json', model_config())
    if baseline is None:
        fresh = out/'baseline-source'
        build.prepare(fresh, out/'config.json', only_variants=['taco_fused'])
        parent = read(fresh/'manifest.json')
        baseline = fresh/'taco_fused'
        entries = _identity_entries()
        (baseline/scope_build.SWIZZLE).write_text(scope_build.swizzle(
            (REPO/scope_build.SWIZZLE).read_text(), entries))
        (baseline/'phase6_host.hpp').write_text('#pragma once\n'+scope_build.tables(entries))
        original = parent['variants']['taco_fused']
        original.update(base='interleaved', order='interleaved')
        for name in (scope_build.SWIZZLE, 'phase6_host.hpp'):
            original['source_sha256'][name] = sha(baseline/name)
        write(fresh/'manifest.json', parent)
    parent_path, parent, original = _baseline_metadata(baseline)
    hashes = original['source_sha256']
    variants, transformed = {}, {}
    for arm in VARIANTS:
        _copy_variant(baseline, out/arm, hashes)
        updates = {} if arm == 'A' else ma1_sources.sources(out/'A', arm)
        for name, text in updates.items():
            (out/arm/name).write_text(text)
        variant = copy.deepcopy(original)
        variant.update(python=f'{arm}/python',
                       source_sha256={p: sha(out/arm/p) for p in sorted(set(hashes) | set(updates))})
        if arm != 'A':
            variant['implementation'] = 'ma1-warp-'+('prefetch' if arm == 'P' else 'serial')
        variants[arm] = variant
        transformed[arm] = sorted(updates)
    shutil.copy2(parent_path, out/'baseline-manifest.json')
    shutil.copy2(REPO/'tools/a800/phase4/TACO_LICENSE.txt', out/'TACO_LICENSE.txt')
    inputs = {str(HERE/p): sha(HERE/p) for p in INPUTS}
    info = dict(protocol=build.PROTOCOL, arch='sm90', kind='baselines', config=model_config(),
                status='prepared', gpu_validated=False, source_only=bool(parent.get('source_only', False)),
                diagnostic_only=bool(parent.get('diagnostic_only', False)), scope_roles=['attn', 'mlp'],
                source_commit=parent.get('source_commit'), submodules=parent['submodules'],
                inputs=inputs, variants=variants,
                files={p: sha(out/p) for p in ('config.json', 'TACO_LICENSE.txt')},
                algorithm='A scalar; W/P same four-chain warp inverse and source-order BF16 fold; P alone preloads next source after publication')
    write(out/'manifest.json', info)
    record = dict(protocol=PROTOCOL, gpu_validated=False,
                  baseline_root=str(baseline), baseline_manifest_sha256=sha(parent_path),
                  baseline_source_only=info['source_only'], baseline_source_sha256=hashes,
                  inputs=inputs, transformed_files=transformed,
                  variants={a: v['source_sha256'] for a, v in variants.items()},
                  files={p: sha(out/p) for p in ('config.json', 'TACO_LICENSE.txt', 'baseline-manifest.json')},
                  build_manifest_sha256=sha(out/'manifest.json'))
    write(out/MANIFEST, record)
    verify(out)
    return out


def runtime_root(root, variant):
    require(variant in VARIANTS, 'Expected MA1 A, W or P')
    return Path(root).resolve()/variant


def verify(root, built=False):
    root = Path(root).resolve()
    record, info = read(root/MANIFEST), read(root/'manifest.json')
    require(record['protocol'] == PROTOCOL and record['gpu_validated'] is False,
            'Wrong MA1 protocol/validation state')
    verify_files(root, record['files'])
    verify_files('/', record['inputs'])
    require(record['inputs'] == {str(HERE/p): sha(HERE/p) for p in INPUTS}, 'MA1 builder changed')
    require(record['build_manifest_sha256'] == sha(root/'manifest.json'), 'MA1 build manifest changed')
    require(info['protocol'] == build.PROTOCOL and info['arch'] == 'sm90' and
            info['config'] == model_config() and info['gpu_validated'] is False and
            info['source_only'] == record['baseline_source_only'] and
            info['inputs'] == record['inputs'] and set(info['variants']) == set(VARIANTS),
            'Wrong A/W/P build configuration')
    verify_files(root, info['files'])
    require(record['variants']['A'] == record['baseline_source_sha256'], 'A changed from pinned baseline')
    for arm in VARIANTS:
        hashes = info['variants'][arm]['source_sha256']
        require(hashes == record['variants'][arm], 'MA1 source registration changed')
        verify_files(root/arm, hashes)
        updates = {} if arm == 'A' else ma1_sources.sources(root/'A', arm)
        require(sorted(updates) == record['transformed_files'][arm], 'MA1 transform file set changed')
        require(hashes == dict(record['baseline_source_sha256'], **{p: _digest(t) for p, t in updates.items()}),
                arm+' differs from the registered MA1 transformation')
    require(sha(root/'baseline-manifest.json') == record['baseline_manifest_sha256'], 'A snapshot changed')
    if built:
        require(info['status'] == 'built' and not info['source_only'] and not info['diagnostic_only'],
                'MA1 libraries are unbuilt or diagnostic-only')
        for arm in VARIANTS:
            require(all((root/arm/p).is_file() for p in
                        ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')), 'Missing MA1 binary')
    return record


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root = Path(root).resolve()
    verify(root)
    require(type(jobs) is int and jobs > 0, 'Positive jobs required')
    try:
        build.compile_build(root, Path(cuda_home), jobs, None if nccl_root is None else Path(nccl_root))
    finally:
        record = read(root/MANIFEST)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/MANIFEST, record)
    verify(root, built=True)


def compile_check(root, out, cuda_home=None, cxx='g++', host_only=False):
    """Verify every mapping and compile actual receiver objects; no GPU launch."""
    root, out = Path(root).resolve(), Path(out).resolve()
    verify(root)
    require(host_only or cuda_home is not None, 'CUDA home is required')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, gpu_used=False, gpu_validated=False, linked_runtime=False,
                  checks={}, commands=[], objects={})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')

    def run(command):
        command = list(map(str, command)); result['commands'].append(command)
        with (out/'compile.log').open('a') as log:
            log.write(repr(command)+'\n'); log.flush()
            subprocess.run(command, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        for arm in VARIANTS:
            src, dest = root/arm, out/arm
            dest.mkdir()
            host = dest/'mapping.cc'
            host.write_text('''#include <initializer_list>
#include "phase6_host.hpp"
int main() {
  for(int k: {512,2048}) for(int r=0;r<8;++r) for(int i=0;i<512;++i)
    if(phase6_host_lookup(16,32,k,r,i)!=((i%8+r)%8)*64+i/8)return 1;
  return phase6_host_lookup(16,32,1024,0,0)==-1 ? 0 : 2;
}
''')
            run([cxx, '-std=c++17', '-O2', '-I'+str(src), host, '-o', dest/'mapping'])
            run([dest/'mapping']); result['checks'][arm] = dict(host_mapping=True)
            if host_only:
                continue
            cuda_home = Path(cuda_home).resolve()
            require((cuda_home/'bin/nvcc').is_file(), 'CUDA Toolkit missing')
            includes = [src, src/'src', src/'include', src/'3rdparty/cutlass/include',
                        src/'3rdparty/cutlass/tools/util/include']
            flags = [cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                     '--expt-extended-lambda', *['-I'+str(p) for p in includes]]
            run([*flags, '-arch=sm_90', '-Xptxas=-v,-warn-spills', '-c', src/RUNTIME, '-o', dest/'runtime.o'])
            run([*flags, '-arch=compute_90', '-ptx', src/RUNTIME, '-o', dest/'runtime.ptx'])
            with (dest/'runtime.sass').open('w') as f:
                subprocess.run([str(cuda_home/'bin/cuobjdump'), '--dump-sass', str(dest/'runtime.o')],
                               env=env, stdout=f, check=True)
            result['checks'][arm]['runtime_object'] = True
        result['completed'] = True
    finally:
        result['objects'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                             if p.is_file() and p.suffix in ('.o', '.ptx', '.sass')}
        write(out/'validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', type=Path, required=True); p.add_argument('--baseline', type=Path)
    p = sub.add_parser('verify'); p.add_argument('--build', type=Path, required=True); p.add_argument('--built', action='store_true')
    p = sub.add_parser('build'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True); p.add_argument('--jobs', type=int, default=2); p.add_argument('--nccl-root', type=Path)
    p = sub.add_parser('compile-check'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--cuda-home', type=Path)
    p.add_argument('--host-only', action='store_true'); p.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    if args.action == 'prepare': print(prepare(args.out, args.baseline))
    elif args.action == 'verify': verify(args.build, args.built)
    elif args.action == 'build': compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)
    else: compile_check(args.build, args.out, args.cuda_home, args.cxx, args.host_only)


if __name__ == '__main__':
    main()
