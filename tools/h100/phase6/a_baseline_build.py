"""Isolated basic/improved producer-fused baselines, with the original tile order."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import a_baseline_sources as transform
import build
from arrival_op3_v2_build import _copy_variant
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-basic-fused-baseline-build-v1'
MANIFEST = 'a-baseline-manifest.json'
VARIANTS = ('basic', 'improved')
SWIZZLE = 'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
INPUTS = ('a_baseline_build.py', 'a_baseline_sources.py', 'a_baseline_decode.cuh',
          'build.py', 'common.py', 'scope_plan.py', 'scope_protocol.py',
          'arrival_op3_v2_build.py', 'generator.cc', 'check_mapping.cu')


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _inputs():
    paths = [HERE/name for name in INPUTS]
    frozen = HERE/'fused_v2'
    paths += [frozen/'provenance.json'] + [frozen/name for name in read(frozen/'provenance.json')['files']]
    paths.append(REPO/SWIZZLE)
    return {str(path): sha(path) for path in paths}


def _frozen_sources():
    frozen = HERE/'fused_v2'
    record = read(frozen/'provenance.json')
    verify_files(frozen, record['files'])
    result = {}
    for name in record['files']:
        text = (frozen/name).read_text()
        if name == transform.WRAPPER:
            text = build.wrapper(text, legacy_fused=True)
        elif name.endswith('.py'):
            text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)').replace(
                'supports SM80 only', 'requires the H100 SM90 port').replace(
                'tools/a800/phase4/build.py', 'tools/h100/phase6/build.py')
        result[name] = _digest(text)
    return result


def _mapping_checker(text):
    old = '      int expected=phase6_host_lookup(tm,tn,k/tp,rank,i);'
    new = '''      // Original Flux order is checked against this launch's base swizzle,
      // including its real occupancy/cohort behavior, for either local K.
      if (x != (bx + tm / tp * rank) % tm || y != by) return 73;
      int expected=phase6_host_lookup(tm,tn,k/tp,rank,i);'''
    from common import replace_once
    return replace_once(text, old, new)


def prepare(out, source_only=False):
    out = Path(out).resolve()
    require(type(source_only) is bool, 'source_only must be a bool')
    require(not out.exists(), 'Use a fresh a-baseline build directory')
    out.mkdir(parents=True)
    write(out/'config.json', model_config())
    parent_root = out/'baseline-source'
    build.prepare(parent_root, out/'config.json', only_variants=['taco_fused'])
    parent = read(parent_root/'manifest.json')
    baseline = parent_root/'taco_fused'
    original = parent['variants']['taco_fused']
    require(original['role'] == 'fused' and original['base'] == 'original' and
            original.get('implementation') == 'a800-fused-v2-cta-v1',
            'Expected the frozen original-order fused-v2 implementation')
    expected = _frozen_sources()
    require(all(original['source_sha256'].get(p) == digest for p, digest in expected.items()),
            'Frozen fused-v2 source changed')
    require(sha(baseline/SWIZZLE) == sha(REPO/SWIZZLE), 'Original tile order changed')
    shared_changes = {'check_mapping.cu': _mapping_checker((baseline/'check_mapping.cu').read_text())}
    source_hashes = dict(original['source_sha256'],
                         **{p: _digest(t) for p, t in shared_changes.items()})
    for arm in VARIANTS:
        _copy_variant(baseline, out/arm, original['source_sha256'])
        for name, text in shared_changes.items():
            (out/arm/name).write_text(text)
    updates = transform.sources(out/'improved')
    for name, text in updates.items():
        path = out/'basic'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    variants = {}
    for arm in VARIANTS:
        names = set(source_hashes) | (set(updates) if arm == 'basic' else set())
        v = copy.deepcopy(original)
        v.update(python=f'{arm}/python', order='original',
                 implementation=('a-baseline-producer-fused-staged-decode-v1' if arm == 'basic'
                                 else 'a800-fused-v2-cta-v1'),
                 source_sha256={p: sha(out/arm/p) for p in sorted(names)})
        variants[arm] = v
    shutil.copy2(parent_root/'manifest.json', out/'baseline-manifest.json')
    shutil.copy2(REPO/'tools/a800/phase4/TACO_LICENSE.txt', out/'TACO_LICENSE.txt')
    info = dict(protocol=build.PROTOCOL, arch='sm90', kind='baselines',
        config=model_config(), status='prepared', gpu_validated=False,
        source_only=source_only, diagnostic_only=False, scope_roles=['attn', 'mlp'],
        source_commit=parent['source_commit'], submodules=parent['submodules'],
        inputs=_inputs(), variants=variants,
        files={p: sha(out/p) for p in ('config.json', 'TACO_LICENSE.txt')},
        algorithm='Original-order producer fusion; basic materializes decoded BF16 sources; improved fuses decode and ordered BF16 reduction')
    write(out/'manifest.json', info)
    record = dict(protocol=PROTOCOL, gpu_validated=False, source_only=source_only,
        original_source_sha256=original['source_sha256'],
        baseline_source_sha256=source_hashes,
        shared_transformed_files=sorted(shared_changes),
        transformed_files={'basic': sorted(updates), 'improved': []},
        inputs=info['inputs'], variants={a: v['source_sha256'] for a, v in variants.items()},
        files={p: sha(out/p) for p in ('config.json', 'TACO_LICENSE.txt', 'baseline-manifest.json')},
        build_manifest_sha256=sha(out/'manifest.json'))
    write(out/MANIFEST, record)
    verify(out)
    return out


def runtime_root(root, arm):
    require(arm in VARIANTS, 'Expected basic or improved fused baseline')
    return Path(root).resolve()/arm


def verify(root, built=False):
    root = Path(root).resolve()
    record, info = read(root/MANIFEST), read(root/'manifest.json')
    require(record['protocol'] == PROTOCOL and record['gpu_validated'] is False,
            'Wrong basic-fused build protocol')
    verify_files(root, record['files'])
    require(record['inputs'] == _inputs(), 'Basic-fused builder inputs changed')
    verify_files('/', record['inputs'])
    require(record['build_manifest_sha256'] == sha(root/'manifest.json'), 'Build manifest changed')
    require(info['protocol'] == build.PROTOCOL and info['arch'] == 'sm90' and
            info['config'] == model_config() and info['gpu_validated'] is False and
            info['source_only'] == record['source_only'] and
            info['inputs'] == record['inputs'] and set(info['variants']) == set(VARIANTS),
            'Wrong basic/improved build configuration')
    verify_files(root, info['files'])
    parent = read(root/'baseline-manifest.json')
    original = parent['variants']['taco_fused']
    require(original['source_sha256'] == record['original_source_sha256'] and
            original['role'] == 'fused' and original['base'] == 'original' and
            original.get('implementation') == 'a800-fused-v2-cta-v1',
            'Parent is not the original-order frozen fused-v2 source')
    frozen = _frozen_sources()
    require(all(record['original_source_sha256'].get(p) == digest for p, digest in frozen.items()),
            'Pinned fused-v2 source differs')
    checker = _mapping_checker((HERE/'check_mapping.cu').read_text())
    shared = dict(record['original_source_sha256'], **{'check_mapping.cu': _digest(checker)})
    require(record['baseline_source_sha256'] == shared and
            record['shared_transformed_files'] == ['check_mapping.cu'] and
            shared[SWIZZLE] == sha(REPO/SWIZZLE), 'Common original-order build changed')
    updates = transform.sources(root/'improved')
    for arm in VARIANTS:
        variant = info['variants'][arm]
        require(variant['base'] == 'original' and variant['order'] == 'original' and
                variant['role'] == 'fused' and not variant.get('sampling') and
                variant['python'] == f'{arm}/python', 'Fused baseline routing changed')
        changes = updates if arm == 'basic' else {}
        expected = dict(shared, **{p: _digest(t) for p, t in changes.items()})
        require(record['transformed_files'][arm] == sorted(changes) and
                variant['source_sha256'] == record['variants'][arm] == expected,
                arm+' differs from the registered basic-fusion transformation')
        verify_files(root/arm, expected)
    if built:
        require(info['status'] == 'built' and not info['source_only'] and not info['diagnostic_only'],
                'Fused libraries are unbuilt, source-only or diagnostic-only')
        for arm in VARIANTS:
            require(all((root/arm/p).is_file() for p in
                        ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')),
                    'Missing fused baseline binary')
    return record


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root = Path(root).resolve()
    verify(root)
    require(not read(root/'manifest.json')['source_only'], 'Cannot compile a source-only fixture')
    require(type(jobs) is int and jobs > 0, 'Positive jobs required')
    try:
        build.compile_build(root, Path(cuda_home), jobs,
                            None if nccl_root is None else Path(nccl_root))
    finally:
        record = read(root/MANIFEST)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/MANIFEST, record)
    verify(root, built=True)


def compile_check(root, out, cuda_home):
    """Compile the actual SM90 receiver objects/PTX without using a GPU."""
    root, out, cuda_home = Path(root).resolve(), Path(out).resolve(), Path(cuda_home).resolve()
    verify(root)
    require((cuda_home/'bin/nvcc').is_file(), 'CUDA Toolkit missing')
    out.mkdir(parents=True, exist_ok=False)
    record = dict(completed=False, gpu_used=False, linked_runtime=False, commands=[], objects={})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        for arm in VARIANTS:
            src, dest = root/arm, out/arm
            dest.mkdir()
            includes = [src/'src', src/'include', src/'3rdparty/cutlass/include',
                        src/'3rdparty/cutlass/tools/util/include']
            flags = [str(cuda_home/'bin/nvcc'), '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                     '--expt-extended-lambda', *['-I'+str(p) for p in includes]]
            for extra, filename in [(['-arch=sm_90', '-Xptxas=-v,-warn-spills', '-c'], 'runtime.o'),
                                    (['-arch=compute_90', '-ptx'], 'runtime.ptx')]:
                command = [*flags, *extra, str(src/transform.RUNTIME), '-o', str(dest/filename)]
                record['commands'].append(command)
                with (out/'compile.log').open('a') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (dest/'runtime.sass').open('w') as log:
                subprocess.run([str(cuda_home/'bin/cuobjdump'), '--dump-sass', str(dest/'runtime.o')],
                               env=env, stdout=log, check=True)
        record['completed'] = True
    finally:
        record['objects'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                             if p.is_file() and p.suffix in ('.o', '.ptx', '.sass')}
        write(out/'validation.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-only', action='store_true')
    p = sub.add_parser('verify'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--built', action='store_true')
    p = sub.add_parser('compile'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True); p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--nccl-root', type=Path)
    p = sub.add_parser('compile-check'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--cuda-home', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': print(prepare(args.out, args.source_only))
    elif args.action == 'verify': verify(args.build, args.built)
    elif args.action == 'compile': compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)
    else: compile_check(args.build, args.out, args.cuda_home)


if __name__ == '__main__':
    main()
