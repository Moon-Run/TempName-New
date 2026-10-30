"""MA4-G: native dynamic-parameter CUDA graphs over frozen CR2/C/R controls."""
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
import ma2_cr2_build as baseline
from ma2_ablation_build import copy_registered
import ma4_g_sources
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-ma4-g-v1'
MANIFEST = 'ma4-g-manifest.json'
VARIANTS = ('G', 'CR2', 'C', 'R')
PAIR_VARIANTS = ('G', 'CR2')
RUNTIME = baseline.RUNTIME
C_POLICY = baseline.C_POLICY
INPUTS = tuple(dict.fromkeys(('ma4_g_build.py', 'ma4_g_sources.py', *baseline.INPUTS)))
canonical = baseline.canonical


def prepare(out, pair=False):
    arms = PAIR_VARIANTS if pair else VARIANTS
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh MA4-G build')
    out.mkdir(parents=True)
    reference = baseline.prepare(out/'reference')
    old = read(reference/'manifest.json')
    for arm in arms[1:]:
        copy_registered(reference/arm, out/arm, old['variants'][arm]['source_sha256'])
    copy_registered(reference/'CR2', out/'G', old['variants']['CR2']['source_sha256'])
    updates = ma4_g_sources.sources(reference/'CR2')
    for name, text in updates.items():
        dest = out/'G'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt'):
        shutil.copy2(reference/name, out/name)
    variants = {}
    for arm in arms:
        item = copy.deepcopy(old['variants']['CR2' if arm == 'G' else arm])
        names = set(item['source_sha256']) | (set(updates) if arm == 'G' else set())
        item.update(python=arm+'/python', source_sha256={name:sha(out/arm/name) for name in sorted(names)})
        variants[arm] = item
    inputs = {str(HERE/name):sha(HERE/name) for name in INPUTS}
    files = {name:sha(out/name) for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt')}
    info = dict(protocol=build.PROTOCOL, arch='sm90', kind='baselines', status='prepared',
        config=model_config(), gpu_validated=False, source_only=False, diagnostic_only=False,
        submodules=old['submodules'], source_commit=old.get('source_commit'), inputs=inputs,
        files=files, variants=variants)
    write(out/'manifest.json', info)
    record = dict(protocol=PROTOCOL, inputs=inputs, files=files, baseline_source_only=False,
        gpu_validated=False, reference_protocol=baseline.PROTOCOL,
        baseline_variants={a:old['variants'][a]['source_sha256'] for a in ('CR2','C','R')},
        variants={a:v['source_sha256'] for a,v in variants.items()},
        build_manifest_sha256=sha(out/'manifest.json'))
    write(out/MANIFEST, record)
    verify(out)
    return out


def verify(root, built=False):
    root = Path(root).resolve()
    record, info = read(root/MANIFEST), read(root/'manifest.json')
    arms = tuple(info['variants'])
    require(record['protocol']==PROTOCOL and record['reference_protocol']==baseline.PROTOCOL,
            'Wrong MA4-G source protocol')
    verify_files('/', record['inputs'])
    verify_files(root, record['files'])
    require(record['inputs']=={str(HERE/name):sha(HERE/name) for name in INPUTS}, 'MA4-G builder changed')
    require(sha(root/'manifest.json')==record['build_manifest_sha256'], 'MA4-G build manifest changed')
    require(info['protocol']==build.PROTOCOL and info['arch']=='sm90' and info['config']==model_config()
            and not info['source_only'] and not info['diagnostic_only'] and arms in (VARIANTS, PAIR_VARIANTS)
            and set(record['variants']) == set(arms),
            'Wrong MA4-G build configuration')
    baseline.verify(root/'reference')
    ref = read(root/'reference/manifest.json')
    expected = {arm:ref['variants'][arm]['source_sha256'] for arm in ('CR2','C','R')}
    require(record['baseline_variants']==expected, 'Frozen CR2/C/R changed')
    require(read(root/'selection-plan.json')==read(root/'reference/selection-plan.json'), 'C masks/order changed')
    updates = ma4_g_sources.sources(root/'CR2')
    expected['G'] = dict(expected['CR2'], **{name:hashlib.sha256(text.encode()).hexdigest() for name,text in updates.items()})
    for arm in arms:
        hashes = info['variants'][arm]['source_sha256']
        require(hashes==record['variants'][arm]==expected[arm], arm+' differs from registered transform')
        verify_files(root/arm, hashes)
        if built:
            require(all((root/arm/name).is_file() for name in ('libflux_cuda.so','libflux_cuda_ths_op.so','check_mapping')),
                    'Missing MA4-G/control runtime: '+arm)
    if built:
        require(info['status']=='built', 'Unbuilt MA4-G experiment')
    return record


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root = Path(root).resolve()
    verify(root)
    try:
        build.compile_build(root, Path(cuda_home), jobs, None if nccl_root is None else Path(nccl_root))
    finally:
        record = read(root/MANIFEST)
        record['build_manifest_sha256'] = sha(root/'manifest.json')
        write(root/MANIFEST, record)
    verify(root, built=True)


def compile_check(root, out, cuda_home=None, cxx='g++', host_only=False, torch_python=None):
    """Compile mappings, runtime, typed CUTLASS graph update and PyTorch bindings."""
    root, out = Path(root).resolve(), Path(out).resolve()
    verify(root)
    require(host_only or cuda_home is not None, 'CUDA home is required')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, gpu_used=False, gpu_validated=False, linked_runtime=False,
                  build_manifest_sha256=sha(root/'manifest.json'),
                  ma4_g_manifest_sha256=sha(root/MANIFEST),
                  builder_inputs=read(root/MANIFEST)['inputs'],
                  changed_sources={name:sha(root/'G'/name) for name in ma4_g_sources.sources(root/'CR2')},
                  checks={}, commands=[], objects={})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')

    def run(command, capture=False):
        command = list(map(str, command)); result['commands'].append(command)
        with (out/'compile.log').open('a') as log:
            log.write(repr(command)+'\n'); log.flush()
            child = subprocess.run(command, env=env, cwd=out,
                                   stdout=subprocess.PIPE if capture else log,
                                   stderr=log, text=True, check=True)
            if capture:
                log.write(child.stdout)
                return child.stdout

    try:
        for arm in read(root/'manifest.json')['variants']:
            src, dest = root/arm, out/arm
            dest.mkdir()
            host = dest/'mapping.cc'
            plans=read(root/'selection-plan.json')
            role_entries=plans['variants'][C_POLICY]['roles']
            body='#include <initializer_list>\n#include "phase6_host.hpp"\nint main(){\n'
            for role,k in [('attn',512),('mlp',2048)]:
                coords=role_entries[role]['entry']['coords'] if arm in ('G','CR2','C') else [[((i%8+r)%8)*64+i//8 for i in range(512)] for r in range(8)]
                flat=','.join(str(v) for row in coords for v in row)
                body+='{const int expected[]={'+flat+'}; for(int r=0;r<8;++r)for(int i=0;i<512;++i)if(phase6_host_lookup(16,32,'+str(k)+',r,i)!=expected[r*512+i])return 1;}\n'
            host.write_text(body+'return phase6_host_lookup(16,32,1024,0,0)==-1?0:2;}\n')
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
            if arm == 'G':
                probe = dest/'graph_gemm_probe.cu'
                probe.write_text(ma4_g_sources.compile_probe_source())
                run([*flags, '-arch=sm_90', '-DFLUX_TACO_BASELINE',
                     '-c', probe, '-o', dest/'graph_gemm.o'])
                result['checks'][arm]['typed_gemm_graph_object'] = True
                run([*flags, '-arch=sm_90', '-DFLUX_TACO_BASELINE',
                     '-c', src/ma4_g_sources.BARRIER, '-o', dest/'graph_barrier.o'])
                result['checks'][arm]['graph_barrier_object'] = True
        if not host_only:
            # The experiment changes the native submission wrapper, not the
            # decoder alone. Compile that real translation unit with PyTorch
            # headers; a runtime-only check would miss its C++ integration.
            metadata = json.loads(run([torch_python or sys.executable, '-c',
                'import json, sysconfig, torch; from torch.utils.cpp_extension import include_paths; '
                'print(json.dumps(dict(includes=include_paths()+[sysconfig.get_path("include")], '
                'abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))))'], capture=True))
            src, dest = root/'G', out/'G'
            includes = [src, src/'src', src/'include', Path(cuda_home)/'include',
                        src/'3rdparty/cutlass/include', src/'3rdparty/cutlass/tools/util/include']
            run([cxx, '-std=c++17', '-O0', '-fPIC', '-DFLUX_TACO_BASELINE',
                 '-D_GLIBCXX_USE_CXX11_ABI='+str(metadata['abi']),
                 *['-I'+str(p) for p in includes], *['-I'+p for p in metadata['includes']],
                 '-c', src/ma4_g_sources.WRAPPER, '-o', dest/'wrapper.o'])
            result['checks']['G']['ths_wrapper_object'] = True
            run([cxx, '-std=c++17', '-O0', '-fPIC', '-DFLUX_TACO_BASELINE',
                 '-D_GLIBCXX_USE_CXX11_ABI='+str(metadata['abi']),
                 *['-I'+str(p) for p in includes], *['-I'+p for p in metadata['includes']],
                 '-c', src/'src/pybind/gemm_rs.cc', '-o', dest/'pybind.o'])
            result['checks']['G']['pybind_object'] = True
        result['completed'] = True
    finally:
        result['objects'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                             if p.is_file() and p.suffix in ('.o', '.ptx', '.sass')}
        write(out/'validation.json', result)
    return result
def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pair', action='store_true', help='Prepare only G and CR2 for compilation')
    p = sub.add_parser('verify'); p.add_argument('--build', type=Path, required=True); p.add_argument('--built', action='store_true')
    p = sub.add_parser('build'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True); p.add_argument('--jobs', type=int, default=2); p.add_argument('--nccl-root', type=Path)
    p = sub.add_parser('compile-check'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--cuda-home', type=Path)
    p.add_argument('--host-only', action='store_true'); p.add_argument('--cxx', default='g++')
    p.add_argument('--torch-python', type=Path, help='Python with PyTorch headers for the native wrapper check')
    args = parser.parse_args()
    if args.action == 'prepare': print(prepare(args.out, args.pair))
    elif args.action == 'verify': verify(args.build, args.built)
    elif args.action == 'build': compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)
    else: compile_check(args.build, args.out, args.cuda_home, args.cxx, args.host_only, args.torch_python)


if __name__ == '__main__':
    main()
