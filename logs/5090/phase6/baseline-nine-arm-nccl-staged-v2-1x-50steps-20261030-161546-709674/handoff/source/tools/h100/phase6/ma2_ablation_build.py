"""Frozen baseline-C crossed with the MA2-R receive implementation."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import build
import candidate
import scope_build
import ma2r_build
import ma2r_sources
import ma2_ablation_sources
from ablation_plan import FREEZE
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-ma2r-c-ablation-v1'
MANIFEST = 'ma2-ablation-manifest.json'
VARIANTS = ('CR', 'C', 'R', 'A')
RUNTIME = ma2_ablation_sources.RUNTIME
C_POLICY = 'interleaved_arrival_selective_attn_mlp'
REFERENCE = REPO/'logs/h100/phase6/baseline-taco-cuda-3x-50steps-20261021-162929-989672/compare1'
REFERENCE_FILES = {
    'scripts/selection-plan.json': '2d8f0dc9a77a037282e8ff9355f0b40cc3354b973b1a8c677bb7d6872af5c57c',
    'provenance/candidate-manifest.json': 'e1b80ed428cd8b1c3baf187f10389d7684411d508983bd8887fc575f6489da12',
    'provenance/fused-manifest.json': '8888d26228f104abfa9a29a8283bf1a115ede2e3180fcf3c2036bf8e57443a6d'}
INPUTS = ('ma2_ablation_build.py', 'ma2_ablation_sources.py', 'ma2_c_decode.cuh',
          'ma2r_build.py', 'ma2r_sources.py', 'ma2_sources.py', 'ma2_decode.cuh',
          'ma1_sources.py', 'ma1_decode.cuh', 'candidate.py', 'scope_build.py',
          'build.py', 'scope_plan.py', 'scope_protocol.py', 'common.py', 'arrival_op3_v2_build.py')


def canonical():
    verify_files(REFERENCE, REFERENCE_FILES)
    plans = read(REFERENCE/'scripts/selection-plan.json')
    roles = plans['variants'][C_POLICY]['roles']
    require([sum(map(sum, roles[r]['entry']['mask'])) for r in ('attn','mlp')] == [53,50],
            'C must retain baseline 1/64 masks, not a new calibration')
    return plans, read(REFERENCE/'provenance/candidate-manifest.json'), read(REFERENCE/'provenance/fused-manifest.json')


def copy_registered(source, target, hashes):
    verify_files(source, hashes)
    for name in hashes:
        dest = target/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, dest)
    (target/'3rdparty').mkdir(exist_ok=True)
    (target/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)


def reconstruct_c(dest, plans, manifest):
    # Recover exactly the C used by base-baseline.md from registered sources.
    # No calibration, new mask, fallback C, or existing binary is required.
    frozen, old, measured = candidate.sources()
    source = old/'interleaved_arrival_selective'
    expected = manifest['variants']['scope-candidate']['source_sha256']
    roles = plans['variants'][C_POLICY]['roles']
    entries = {r: p['entry'] for r,p in roles.items()}
    overrides = {scope_build.SWIZZLE: scope_build.swizzle((FREEZE/'source'/scope_build.SWIZZLE).read_text(), entries),
                 'phase6_host.hpp': '#pragma once\n'+scope_build.tables(entries),
                 'python/flux/gemm_rs_taco.py': candidate.codec_source((source/'python/flux/gemm_rs_taco.py').read_text())}
    for name, digest in expected.items():
        import hashlib
        data = overrides[name].encode() if name in overrides else (source/name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, 'Cannot reconstruct exact baseline C: '+name)
        path=dest/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    (dest/'3rdparty').mkdir(exist_ok=True)
    (dest/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)


def prepare(out):
    out=Path(out).resolve(); require(not out.exists(), 'Use a fresh MA2 ablation build')
    plans, cm, fm=canonical()
    out.mkdir(parents=True)
    base=ma2r_build.prepare(out/'allremote-source')
    bm=read(base/'manifest.json')
    require(bm['variants']['A']['source_sha256'] == fm['variants']['interleaved_all']['source_sha256'],
            'A must equal the documented baseline A')
    for arm in ('A','R'):
        copy_registered(base/arm,out/arm,bm['variants'][arm]['source_sha256'])
    reconstruct_c(out/'C',plans,cm)
    ch=cm['variants']['scope-candidate']['source_sha256']
    copy_registered(out/'C',out/'CR',ch)
    updates=ma2_ablation_sources.sources(out/'C')
    for name,text in updates.items(): (out/'CR'/name).write_text(text)
    write(out/'selection-plan.json',plans); write(out/'config.json',model_config())
    shutil.copy2(REPO/'tools/a800/phase4/TACO_LICENSE.txt',out/'TACO_LICENSE.txt')
    variants={}
    for arm in VARIANTS:
        old=bm['variants'][arm] if arm in ('A','R') else cm['variants']['scope-candidate']
        v=copy.deepcopy(old)
        names=set(old['source_sha256']) | (set(updates) if arm=='CR' else set())
        v.update(python=arm+'/python',source_sha256={p:sha(out/arm/p) for p in sorted(names)})
        variants[arm]=v
    inputs={str(HERE/p):sha(HERE/p) for p in INPUTS}
    files={p:sha(out/p) for p in ('selection-plan.json','config.json','TACO_LICENSE.txt')}
    info=dict(protocol=build.PROTOCOL,arch='sm90',kind='baselines',status='prepared',
        config=model_config(),gpu_validated=False,source_only=False,diagnostic_only=False,
        submodules=bm['submodules'],source_commit=bm.get('source_commit'),inputs=inputs,files=files,variants=variants)
    write(out/'manifest.json',info)
    record=dict(protocol=PROTOCOL,inputs=inputs,files=files,baseline_source_only=False,gpu_validated=False,
        reference_files=REFERENCE_FILES,variants={a:v['source_sha256'] for a,v in variants.items()},
        build_manifest_sha256=sha(out/'manifest.json'))
    write(out/MANIFEST,record); verify(out); return out


def verify(root,built=False):
    root=Path(root).resolve(); record,info=read(root/MANIFEST),read(root/'manifest.json')
    plans,cm,fm=canonical()
    require(record['protocol']==PROTOCOL and record['reference_files']==REFERENCE_FILES,'Wrong ablation source protocol')
    verify_files('/',record['inputs']);verify_files(root,record['files'])
    require(record['inputs']=={str(HERE/p):sha(HERE/p) for p in INPUTS},'Builder changed')
    require(sha(root/'manifest.json')==record['build_manifest_sha256'],'Build manifest changed')
    require(info['protocol']==build.PROTOCOL and info['config']==model_config() and info['arch']=='sm90'
            and not info['source_only'] and not info['diagnostic_only'] and set(info['variants'])==set(VARIANTS),'Wrong build configuration')
    require(read(root/'selection-plan.json')==plans,'Frozen C masks/order changed')
    import hashlib
    expected={'A':fm['variants']['interleaved_all']['source_sha256'],
              'C':cm['variants']['scope-candidate']['source_sha256']}
    for arm in ('A','C'):
        require(record['variants'][arm]==expected[arm],arm+' changed from documented baseline')
    expected['R']=dict(expected['A'],**{p:hashlib.sha256(t.encode()).hexdigest() for p,t in ma2r_sources.sources(root/'A','R').items()})
    expected['CR']=dict(expected['C'],**{p:hashlib.sha256(t.encode()).hexdigest() for p,t in ma2_ablation_sources.sources(root/'C').items()})
    for arm in VARIANTS:
        hashes=info['variants'][arm]['source_sha256']
        require(hashes==record['variants'][arm]==expected[arm],arm+' differs from registered transform')
        verify_files(root/arm,hashes)
        if built: require(all((root/arm/p).is_file() for p in ('libflux_cuda.so','libflux_cuda_ths_op.so','check_mapping')),'Missing binary')
    if built: require(info['status']=='built','Unbuilt ablation')
    return record


def compile_build(root,cuda_home,jobs=2,nccl_root=None):
    root=Path(root).resolve();verify(root)
    try: build.compile_build(root,Path(cuda_home),jobs,None if nccl_root is None else Path(nccl_root))
    finally:
        record=read(root/MANIFEST);record['build_manifest_sha256']=sha(root/'manifest.json');write(root/MANIFEST,record)
    verify(root,built=True)

def compile_check(root, out, cuda_home=None, cxx='g++', host_only=False, torch_python=None):
    """Compile mapping, actual runtime and changed native wrapper without a GPU."""
    root, out = Path(root).resolve(), Path(out).resolve()
    verify(root)
    require(host_only or cuda_home is not None, 'CUDA home is required')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, gpu_used=False, gpu_validated=False, linked_runtime=False,
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
        for arm in VARIANTS:
            src, dest = root/arm, out/arm
            dest.mkdir()
            host = dest/'mapping.cc'
            plans=read(root/'selection-plan.json')
            role_entries=plans['variants'][C_POLICY]['roles']
            body='#include <initializer_list>\n#include "phase6_host.hpp"\nint main(){\n'
            for role,k in [('attn',512),('mlp',2048)]:
                coords=role_entries[role]['entry']['coords'] if arm in ('C','CR') else [[((i%8+r)%8)*64+i//8 for i in range(512)] for r in range(8)]
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
        if not host_only:
            # The experiment changes the native submission wrapper, not the
            # decoder alone. Compile that real translation unit with PyTorch
            # headers; a runtime-only check would miss its C++ integration.
            metadata = json.loads(run([torch_python or sys.executable, '-c',
                'import json, sysconfig, torch; from torch.utils.cpp_extension import include_paths; '
                'print(json.dumps(dict(includes=include_paths()+[sysconfig.get_path("include")], '
                'abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))))'], capture=True))
            src, dest = root/'R', out/'R'
            includes = [src, src/'src', src/'include', Path(cuda_home)/'include',
                        src/'3rdparty/cutlass/include', src/'3rdparty/cutlass/tools/util/include']
            run([cxx, '-std=c++17', '-O0', '-fPIC', '-DFLUX_TACO_BASELINE',
                 '-D_GLIBCXX_USE_CXX11_ABI='+str(metadata['abi']),
                 *['-I'+str(p) for p in includes], *['-I'+p for p in metadata['includes']],
                 '-c', src/ma2r_sources.WRAPPER, '-o', dest/'wrapper.o'])
            result['checks']['R']['ths_wrapper_object'] = True
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
    p = sub.add_parser('verify'); p.add_argument('--build', type=Path, required=True); p.add_argument('--built', action='store_true')
    p = sub.add_parser('build'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True); p.add_argument('--jobs', type=int, default=2); p.add_argument('--nccl-root', type=Path)
    p = sub.add_parser('compile-check'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--cuda-home', type=Path)
    p.add_argument('--host-only', action='store_true'); p.add_argument('--cxx', default='g++')
    p.add_argument('--torch-python', type=Path, help='Python with PyTorch headers for the native wrapper check')
    args = parser.parse_args()
    if args.action == 'prepare': print(prepare(args.out))
    elif args.action == 'verify': verify(args.build, args.built)
    elif args.action == 'build': compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)
    else: compile_check(args.build, args.out, args.cuda_home, args.cxx, args.host_only, args.torch_python)


if __name__ == '__main__':
    main()
