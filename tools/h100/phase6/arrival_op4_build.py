"""Isolated A/D/L/F source preparation and CPU-only SM90 compilation for fixed-cost op4."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import build
import scope_build
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-arrival-op4'
MANIFEST = 'arrival-op4-manifest.json'
RUNTIME = 'src/gemm_rs/taco_runtime.cu'
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
VARIANTS = ('A', 'D', 'L', 'F')
INPUTS = ('arrival_op4_build.py', 'arrival_op4_sources.py', 'arrival_op3_v2_sources.py', 'build.py',
          'arrival_op3_sources.py', 'arrival_op3_v2_decode.cuh',
          'scope_build.py', 'scope_plan.py', 'scope_protocol.py', 'common.py')


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _safe_relative(relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts,
            'Source manifest contains an unsafe path: '+str(relative))
    return path


def _identity_entries():
    entries = {}
    for role, k in (('attn', 512), ('mlp', 2048)):
        entries[role] = dict(M=2048, N=4096, K_local=k, tiles=512,
            maps=[list(range(512)) for _ in range(8)],
            coords=[[((i % 8+r) % 8)*64+i//8 for i in range(512)] for r in range(8)])
    return entries


def _baseline_metadata(root):
    """Require a registered H100 A source tree, never silently hash arbitrary input."""
    path = root.parent/'manifest.json'
    require(path.is_file(), 'A source root must have a parent manifest.json')
    manifest = read(path)
    require(manifest.get('protocol') == build.PROTOCOL and manifest.get('arch') == 'sm90',
            'Baseline must be a prepared H100 Phase6 build')
    require(manifest.get('config') == model_config(), 'Baseline model configuration changed')
    variant = manifest.get('variants', {}).get(root.name)
    require(variant is not None and variant.get('role') == 'fused' and
            variant.get('base') == 'interleaved' and not variant.get('sampling'),
            'Expected the fused interleaved A variant')
    hashes = variant['source_sha256']
    for name in hashes:
        _safe_relative(name)
    verify_files(root, hashes)
    pinned = HERE/'fused_v2'
    provenance = read(pinned/'provenance.json')
    verify_files(pinned, provenance['files'])
    for relative in provenance['files']:
        text = (pinned/relative).read_text()
        if relative == WRAPPER:
            text = build.wrapper(text, legacy_fused=True)
        elif relative.endswith('.py'):
            text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)').replace(
                'supports SM80 only', 'requires the H100 SM90 port').replace(
                'tools/a800/phase4/build.py', 'tools/h100/phase6/build.py')
        require(hashes.get(relative) == _digest(text),
                'A must retain the pinned H100 fused-v2 source: '+relative)
    require((root/scope_build.SWIZZLE).read_text() == scope_build.swizzle(
                (REPO/scope_build.SWIZZLE).read_text(), _identity_entries()) and
            (root/'phase6_host.hpp').read_text() ==
                '#pragma once\n'+scope_build.tables(_identity_entries()),
            'A must retain the dual-shape interleaved identity schedule')
    return path, manifest, variant


def _copy_variant(source, dest, hashes):
    for relative in hashes:
        target = dest/_safe_relative(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target)
    cutlass = REPO/'3rdparty/cutlass'
    require((cutlass/'include/cutlass/cutlass.h').is_file(), 'CUTLASS submodule missing')
    (dest/'3rdparty').mkdir(exist_ok=True)
    (dest/'3rdparty/cutlass').symlink_to(cutlass, target_is_directory=True)


def prepare(out, baseline=None):
    """Copy a sealed A source, or reconstruct A from the pinned fused-v2 template."""
    import arrival_op4_sources

    out = Path(out).resolve()
    require(not out.exists(), 'Use a new op4 output directory')
    if baseline is None:
        out.mkdir(parents=True)
        write(out/'config.json', model_config())
        fresh = out/'baseline-source'
        build.prepare(fresh, out/'config.json', only_variants=['taco_fused'])
        parent = read(fresh/'manifest.json')
        src = fresh/'taco_fused'
        entries = _identity_entries()
        (src/scope_build.SWIZZLE).write_text(
            scope_build.swizzle((REPO/scope_build.SWIZZLE).read_text(), entries))
        (src/'phase6_host.hpp').write_text('#pragma once\n'+scope_build.tables(entries))
        original = parent['variants']['taco_fused']
        original.update(base='interleaved', order='interleaved')
        for relative in (scope_build.SWIZZLE, 'phase6_host.hpp'):
            original['source_sha256'][relative] = sha(src/relative)
        write(fresh/'manifest.json', parent)
        baseline = src
        origin = 'fresh-pinned-fused-v2-dual-shape-interleaved'
    else:
        baseline = Path(baseline).resolve()
        # Validate before creating an output when an existing source was supplied.
        _baseline_metadata(baseline)
        out.mkdir(parents=True)
        write(out/'config.json', model_config())
        origin = 'registered-A-source-copy'
    baseline = Path(baseline).resolve()
    parent_path, parent, original = _baseline_metadata(baseline)
    source_hashes = original['source_sha256']
    for name in VARIANTS:
        _copy_variant(baseline, out/name, source_hashes)
    replacements = {name: arrival_op4_sources.sources(out/'A', name)
                    for name in ('D', 'L', 'F')}
    for name, transformed in replacements.items():
        require(transformed and all(isinstance(text, str) for text in transformed.values()),
                'op4 source transform returned no source text')
        for relative, text in transformed.items():
            target = out/name/_safe_relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    inputs = {str(HERE/name): sha(HERE/name) for name in INPUTS}
    variants = {}
    for name in VARIANTS:
        variant = copy.deepcopy(original)
        names = set(source_hashes) | set(replacements.get(name, {}))
        variant.update(python=f'{name}/python',
                       source_sha256={p: sha(out/name/p) for p in sorted(names)})
        if name != 'A':
            variant.update(implementation={'D': 'op4-warp-decoder', 'L': 'op4-lifetime',
                                           'F': 'op4-lifetime-warp-decoder'}[name])
        variants[name] = variant
    shutil.copy2(parent_path, out/'baseline-manifest.json')
    shutil.copy2(REPO/'tools/a800/phase4/TACO_LICENSE.txt', out/'TACO_LICENSE.txt')
    info = dict(protocol=build.PROTOCOL, arch='sm90', kind='baselines',
                config=model_config(), status='prepared', gpu_validated=False,
                source_only=bool(parent.get('source_only', False)),
                diagnostic_only=bool(parent.get('diagnostic_only', False)),
                scope_roles=['attn', 'mlp'],
                source_commit=parent.get('source_commit'), submodules=parent['submodules'],
                inputs=inputs, variants=variants,
                files={p: sha(out/p) for p in ('config.json', 'TACO_LICENSE.txt')},
                algorithm='A: unchanged fused-v2; D: warp decoder; L: two complete workspaces and instance configuration; F: L plus D; all retain original publication barrier')
    write(out/'manifest.json', info)
    record = dict(protocol=PROTOCOL, gpu_validated=False, origin=origin,
                  baseline_root=str(baseline), baseline_manifest_sha256=sha(parent_path),
                  baseline_source_only=bool(parent.get('source_only', False)),
                  baseline_source_sha256=source_hashes, inputs=inputs,
                  transformed_files={name: sorted(files) for name, files in replacements.items()},
                  variants={name: v['source_sha256'] for name, v in variants.items()},
                  files={p: sha(out/p) for p in
                         ('config.json', 'TACO_LICENSE.txt', 'baseline-manifest.json')},
                  build_manifest_sha256=sha(out/'manifest.json'))
    write(out/MANIFEST, record)
    verify(out)
    return out


def runtime_root(root, variant):
    require(variant in VARIANTS, 'Expected A, D, L or F')
    return Path(root).resolve()/variant


def verify(root, built=False):
    import arrival_op4_sources

    root = Path(root).resolve()
    record, info = read(root/MANIFEST), read(root/'manifest.json')
    require(record['protocol'] == PROTOCOL and record['gpu_validated'] is False,
            'Wrong op4 protocol/validation state')
    verify_files(root, record['files'])
    verify_files('/', record['inputs'])
    require(record['build_manifest_sha256'] == sha(root/'manifest.json'),
            'Build manifest changed outside the op4 builder')
    require(info['protocol'] == build.PROTOCOL and info['arch'] == 'sm90' and
            info['config'] == model_config() and info['gpu_validated'] is False and
            info['source_only'] == record['baseline_source_only'] and
            info['inputs'] == record['inputs'] and set(info['variants']) == set(VARIANTS),
            'Wrong A/D/L/F build configuration')
    verify_files(root, info['files'])
    for name in VARIANTS:
        hashes = info['variants'][name]['source_sha256']
        require(hashes == record['variants'][name], 'Source manifest changed: '+name)
        verify_files(root/name, hashes)
    require(record['variants']['A'] == record['baseline_source_sha256'],
            'A differs from its source baseline')
    for name in ('D', 'L', 'F'):
        replacements = arrival_op4_sources.sources(root/'A', name)
        require(sorted(replacements) == record['transformed_files'][name], 'Transform file set changed')
        expected = dict(record['variants']['A'],
                        **{p: _digest(text) for p, text in replacements.items()})
        require(record['variants'][name] == expected, name+' differs from the registered transformation')
    require(sha(root/'baseline-manifest.json') == record['baseline_manifest_sha256'],
            'Baseline source snapshot changed')
    if built:
        require(info['status'] == 'built', 'A/D/L/F libraries have not been built')
        for name in VARIANTS:
            require(all((root/name/p).is_file() for p in
                        ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')),
                    'Missing compiled A/D/L/F artifacts')
    return record


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root = Path(root).resolve()
    verify(root)
    require(type(jobs) is int and jobs > 0, 'Positive jobs required')
    try:
        build.compile_build(root, Path(cuda_home), jobs,
                            None if nccl_root is None else Path(nccl_root))
    finally:
        record = read(root/MANIFEST)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/MANIFEST, record)
    verify(root, built=True)


def compile_check(root, out, cuda_home=None, cxx='g++', host_only=False):
    """Compile real runtime, instantiated GEMM and THS wrapper; never launch a GPU."""
    root, out = Path(root).resolve(), Path(out).resolve()
    verify(root)
    require(host_only or cuda_home is not None, 'CUDA home required unless --host-only')
    cuda_home = Path(cuda_home).resolve() if cuda_home is not None else None
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, mode='host-only' if host_only else 'sm90-objects',
                  gpu_used=False, gpu_validated=False, linked_runtime=False,
                  commands=[], checks={}, objects={})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')

    def run(command, capture=False):
        command = list(map(str, command))
        event = dict(args=command, completed=False)
        result['commands'].append(event)
        with (out/'compile.log').open('a') as log:
            log.write(repr(command)+'\n')
            log.flush()
            child = subprocess.run(command, env=env, cwd=out, text=True,
                                   stdout=subprocess.PIPE if capture else log,
                                   stderr=log, check=True)
            if capture:
                log.write(child.stdout)
        event['completed'] = True
        return child.stdout if capture else None

    try:
        for name in VARIANTS:
            src, dest = root/name, out/name
            dest.mkdir()
            result['checks'][name] = {}
            host = dest/'mapping.cc'
            host.write_text('''#include "phase6_host.hpp"
int main() {
  for(int k: {512,2048}) for(int rank=0;rank<8;++rank)
    for(int i=0;i<512;++i)
      if(phase6_host_lookup(16,32,k,rank,i)!=((i%8+rank)%8)*64+i/8)return 1;
  if(phase6_host_lookup(16,32,1024,0,0)!=-1)return 2;
}
'''.replace('#include "phase6_host.hpp"',
            '#include <initializer_list>\n#include "phase6_host.hpp"'))
            run([cxx, '-std=c++17', '-O2', '-I'+str(src), host, '-o', dest/'mapping'])
            run([dest/'mapping'])
            result['checks'][name]['host_mapping'] = True
            if host_only:
                continue
            require((cuda_home/'bin/nvcc').is_file(), 'CUDA Toolkit nvcc not found')
            includes = [src, src/'src', src/'include', cuda_home/'include',
                        src/'3rdparty/cutlass/include',
                        src/'3rdparty/cutlass/tools/util/include']
            inc = ['-I'+str(p) for p in includes]
            nvcc = [cuda_home/'bin/nvcc', '-std=c++17', '-O2',
                    '--expt-relaxed-constexpr', '--expt-extended-lambda',
                    '-DFLUX_TACO_BASELINE', '-gencode=arch=compute_90,code=sm_90', *inc]
            for label, relative in (('mapping', 'check_mapping.cu'), ('runtime', RUNTIME)):
                run([*nvcc, '-c', src/relative, '-o', dest/(label+'.o')])
                result['checks'][name][label+'_object'] = True
            # Generate and compile the entire concrete BF16 RCR V2 registration.
            # An include-only probe would miss errors in epilogue instantiation.
            generator = dest/'generator'
            run([cxx, '-std=c++17', '-O2', *inc, '-I'+str(src/'src/generator'),
                 src/'src/generator/gen_gemm_rs.cc', '-o', generator])
            registers = dest/'registers'
            registers.mkdir()
            run([generator, '--archs=80', '--sm_cores=108',
                 '--dir='+str(registers), '--output_file='+str(dest/'generated.txt')])
            generated = list(registers.glob('*sm80*.cu'))
            require(len(generated) == 1, 'Expected one concrete V2 GEMM registration')
            run([*nvcc, '-c', generated[0], '-o', dest/'gemm.o'])
            result['checks'][name]['instantiated_gemm_object'] = True
            metadata = json.loads(run([sys.executable, '-c',
                'import json, sysconfig, torch; from torch.utils.cpp_extension import include_paths; '
                'print(json.dumps(dict(includes=include_paths()+[sysconfig.get_path("include")], '
                'abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))))'], capture=True))
            run([cxx, '-std=c++17', '-O0', '-fPIC', '-DFLUX_TACO_BASELINE',
                 '-D_GLIBCXX_USE_CXX11_ABI='+str(metadata['abi']), *inc,
                 *['-I'+p for p in metadata['includes']],
                 '-c', src/WRAPPER, '-o', dest/'wrapper.o'])
            result['checks'][name]['ths_wrapper_object'] = True
        result['completed'] = True
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        result['objects'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*.o')}
        write(out/'validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p = actions.add_parser('verify')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--built', action='store_true')
    p = actions.add_parser('compile-check')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path)
    p.add_argument('--host-only', action='store_true')
    p.add_argument('--cxx', default='g++')
    p = actions.add_parser('build')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True)
    p.add_argument('--nccl-root', type=Path)
    p.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.out, args.baseline))
    elif args.action == 'verify':
        verify(args.build, args.built)
        print('A/D/L/F source manifests verified; GPU validation is not implied.')
    elif args.action == 'compile-check':
        compile_check(args.build, args.out, args.cuda_home, args.cxx, args.host_only)
    else:
        compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)


if __name__ == '__main__':
    main()
