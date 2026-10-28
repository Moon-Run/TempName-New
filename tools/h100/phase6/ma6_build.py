"""MA6-v3: frozen F4, packed selection masks, and homogeneous CTA layout."""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import build
import ma4_f_build as baseline
from ma2_ablation_build import copy_registered
import ma6_sources
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-ma6-f4-v3'
MANIFEST = 'ma6-manifest.json'
VARIANTS = ma6_sources.VARIANTS
RUNTIME = baseline.RUNTIME
C_POLICY = baseline.C_POLICY
INPUTS = tuple(dict.fromkeys(('ma6_build.py', 'ma6_sources.py', *baseline.INPUTS)))
canonical = baseline.canonical


def prepare(out, tuning=None):
    tuning = ma6_sources.tuning_config(tuning)
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh MA6 build')
    out.mkdir(parents=True)
    reference = baseline.prepare(out/'reference')
    old = read(reference/'manifest.json')
    frozen = old['variants']['F4']['source_sha256']
    for arm in VARIANTS:
        copy_registered(reference/'F4', out/arm, frozen)
    updates = {arm:ma6_sources.sources(reference/'F4', arm, tuning) for arm in VARIANTS[1:]}
    for arm, changes in updates.items():
        for name, text in changes.items():
            destination = out/arm/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
    for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt'):
        shutil.copy2(reference/name, out/name)
    write(out/'tuning.json', tuning)
    variants = {}
    for arm in VARIANTS:
        item = copy.deepcopy(old['variants']['F4'])
        names = set(frozen) | set(updates.get(arm, {}))
        item.update(python=arm+'/python', source_sha256={name:sha(out/arm/name) for name in sorted(names)})
        variants[arm] = item
    inputs = {str(HERE/name):sha(HERE/name) for name in INPUTS}
    files = {name:sha(out/name) for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt', 'tuning.json')}
    info = dict(protocol=build.PROTOCOL, arch='sm90', kind='baselines', status='prepared',
        config=model_config(), gpu_validated=False, source_only=False, diagnostic_only=False,
        submodules=old['submodules'], source_commit=old.get('source_commit'), inputs=inputs,
        files=files, variants=variants)
    write(out/'manifest.json', info)
    record = dict(protocol=PROTOCOL, tuning=tuning, inputs=inputs, files=files,
        baseline_source_only=False, gpu_validated=False, reference_protocol=baseline.PROTOCOL,
        baseline_variants={'F4':frozen}, variants={a:v['source_sha256'] for a,v in variants.items()},
        build_manifest_sha256=sha(out/'manifest.json'))
    write(out/MANIFEST, record)
    verify(out)
    return out


def verify(root, built=False):
    root = Path(root).resolve()
    record, info = read(root/MANIFEST), read(root/'manifest.json')
    require(record['protocol']==PROTOCOL and record['reference_protocol']==baseline.PROTOCOL,
            'Wrong MA6 source protocol')
    verify_files('/', record['inputs'])
    verify_files(root, record['files'])
    require(record['inputs']=={str(HERE/name):sha(HERE/name) for name in INPUTS}, 'MA6 builder changed')
    require(sha(root/'manifest.json')==record['build_manifest_sha256'], 'MA6 build manifest changed')
    require(info['protocol']==build.PROTOCOL and info['arch']=='sm90' and info['config']==model_config()
            and not info['source_only'] and not info['diagnostic_only']
            and tuple(info['variants']) == VARIANTS and tuple(record['variants']) == VARIANTS,
            'Wrong MA6 build configuration')
    baseline.verify(root/'reference')
    ref = read(root/'reference/manifest.json')
    frozen = ref['variants']['F4']['source_sha256']
    require(record['baseline_variants']=={'F4':frozen}, 'Frozen F4 changed')
    require(read(root/'selection-plan.json')==read(root/'reference/selection-plan.json'), 'C masks/order changed')
    tuning = ma6_sources.tuning_config(record['tuning'])
    require(read(root/'tuning.json') == tuning, 'MA6 tuning identity changed')
    expected = {'F4':frozen}
    for arm in VARIANTS[1:]:
        updates = ma6_sources.sources(root/'F4', arm, tuning)
        expected[arm] = dict(frozen, **{name:hashlib.sha256(text.encode()).hexdigest()
                                      for name,text in updates.items()})
    for arm in VARIANTS:
        hashes = info['variants'][arm]['source_sha256']
        require(hashes==record['variants'][arm]==expected[arm], arm+' differs from registered transform')
        verify_files(root/arm, hashes)
        if built:
            require(all((root/arm/name).is_file() for name in ('libflux_cuda.so','libflux_cuda_ths_op.so','check_mapping')),
                    'Missing MA6 runtime: '+arm)
    if built:
        require(info['status']=='built', 'Unbuilt MA6 experiment')
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
    """Compile mappings, fused decoder runtime and actual PyTorch C++/pybind objects."""
    root, out = Path(root).resolve(), Path(out).resolve()
    verify(root)
    require(host_only or cuda_home is not None, 'CUDA home is required')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, gpu_used=False, gpu_validated=False, linked_runtime=False,
                  build_manifest_sha256=sha(root/'manifest.json'),
                  ma6_manifest_sha256=sha(root/MANIFEST),
                  builder_inputs=read(root/MANIFEST)['inputs'],
                  changed_sources={arm:{name:sha(root/arm/name) for name in ma6_sources.sources(
                      root/'F4', arm, read(root/MANIFEST)['tuning'])} for arm in VARIANTS[1:]},
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

    def decoder_ptx(path, kernel):
        text = path.read_text()
        entries = [line for line in text.splitlines()
                   if '.entry ' in line and kernel in line]
        require(len(entries) == 1, 'Missing original F4 PTX decoder')
        start = text.index(entries[0])
        opening = text.index('{', start)
        depth = 0
        for offset in range(opening, len(text)):
            depth += (text[offset] == '{') - (text[offset] == '}')
            if depth == 0:
                return text[start:offset + 1]
        raise ValueError('Unclosed F4 PTX decoder')

    try:
        for arm in read(root/'manifest.json')['variants']:
            src, dest = root/arm, out/arm
            dest.mkdir()
            host = dest/'mapping.cc'
            plans=read(root/'selection-plan.json')
            role_entries=plans['variants'][C_POLICY]['roles']
            body='#include <initializer_list>\n#include "phase6_host.hpp"\nint main(){\n'
            for role,k in [('attn',512),('mlp',2048)]:
                coords=role_entries[role]['entry']['coords']
                flat=','.join(str(v) for row in coords for v in row)
                body+='{const int expected[]={'+flat+'}; for(int r=0;r<8;++r)for(int i=0;i<512;++i)if(phase6_host_lookup(16,32,'+str(k)+',r,i)!=expected[r*512+i])return 1;}\n'
            host.write_text(body+'return phase6_host_lookup(16,32,1024,0,0)==-1?0:2;}\n')
            run([cxx, '-std=c++17', '-O2', '-I'+str(src), host, '-o', dest/'mapping'])
            run([dest/'mapping']); result['checks'][arm] = dict(host_mapping=True)
            if arm == 'F4-mask':
                host = dest/'mask.cc'
                body = '#include "gemm_rs/ma6_mask.h"\nint main(){\n'
                for role in ma6_sources.ROLES:
                    mask = role_entries[role]['entry']['mask']
                    flat = ','.join(str(v) for row in mask for v in row)
                    body += ('{const uint8_t selected[]={' + flat + '}; uint8_t packed[512];\n'
                             'if(!taco_ma6_pack_mask(selected,packed))return 1;\n'
                             'for(int rank=0;rank<8;++rank)for(int group=0;group<8192;++group){\n'
                             'int tile=(rank*2+(group/32)/128)*32+group%32;\n'
                             'for(int src=0;src<8;++src)if(((packed[tile]>>src)&1)!='
                             '(src!=rank && selected[src*512+tile]))return 2;}}\n')
                host.write_text(body+'return 0;}\n')
                run([cxx, '-std=c++17', '-O2', '-I'+str(src/'src'), host, '-o', dest/'mask'])
                run([dest/'mask'])
                result['checks'][arm]['frozen_role_masks_equivalent'] = True
            elif arm == 'F4-layout':
                host = dest/'layout.cc'
                mapping = ma6_sources.GROUP_MAPPING.replace('blockIdx.x','block').replace('threadIdx.x','thread')
                host.write_text('int main(){int seen[8192]={}; for(int rank=0;rank<8;++rank){'
                    'for(int block=0;block<2048;++block){int tile0=-1;'
                    'for(int thread=0;thread<128;thread+=32){int group='+mapping+';'
                    'if(group<0||group>=8192)return 1; if(rank==0)++seen[group];'
                    'int tile=(rank*2+(group/32)/128)*32+group%32;'
                    'if(tile0<0)tile0=tile;else if(tile!=tile0)return 2;}}}'
                    'for(int group=0;group<8192;++group)if(seen[group]!=1)return 3;return 0;}')
                run([cxx, '-std=c++17', '-O2', host, '-o', dest/'layout'])
                run([dest/'layout'])
                result['checks'][arm]['complete_homogeneous_cta_mapping'] = True
            if host_only:
                continue
            cuda_home = Path(cuda_home).resolve()
            require((cuda_home/'bin/nvcc').is_file(), 'CUDA Toolkit missing')
            includes = [src, src/'src', src/'include', src/'3rdparty/cutlass/include',
                        src/'3rdparty/cutlass/tools/util/include']
            flags = [cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                     '--expt-extended-lambda', *['-I'+str(p) for p in includes]]
            run([*flags, '-arch=sm_90', '-Xptxas=-v,-warn-spills', '-c', src/RUNTIME, '-o', dest/'runtime.o'])
            run([*flags, '-arch=sm_90', src/'check_mapping.cu', '-ldl', '-o', dest/'mapping_gpu'])
            result['checks'][arm]['gpu_mapping_compiled_not_run'] = True
            run([*flags, '-arch=compute_90', '-ptx', src/RUNTIME, '-o', dest/'runtime.ptx'])
            with (dest/'runtime.sass').open('w') as f:
                subprocess.run([str(cuda_home/'bin/cuobjdump'), '--dump-sass', str(dest/'runtime.o')],
                               env=env, stdout=f, check=True)
            result['checks'][arm]['runtime_object'] = True
        if not host_only:
            original = decoder_ptx(out/'F4/runtime.ptx', ma6_sources.kernel_name('attn', arm='F4'))
            for arm in VARIANTS[1:]:
                require(decoder_ptx(out/arm/'runtime.ptx', ma6_sources.kernel_name('attn', arm='F4')) == original,
                        'Original F4 decoder PTX changed: ' + arm)
                result['checks'][arm]['original_f4_decoder_ptx_identical'] = True
            mask_ptx = decoder_ptx(out/'F4-mask/runtime.ptx', ma6_sources.kernel_name('attn', arm='F4-mask'))
            require(not re.search(r'\b(?:ld|st)\.local', mask_ptx), 'P1 still uses PTX local memory')
            sass = (out/'F4-mask/runtime.sass').read_text()
            sections = [part for part in sass.split('Function : ') if
                        ma6_sources.kernel_name('attn', arm='F4-mask') in part.splitlines()[0]]
            require(len(sections)==1 and not re.search(r'\b(?:LDL|STL)(?:\.|\s)', sections[0]),
                    'P1 still uses SASS local memory')
            result['checks']['F4-mask']['decoder_no_ptx_or_sass_local_memory'] = True
            # The experiment changes the native submission wrapper, not the
            # decoder alone. Compile that real translation unit with PyTorch
            # headers; a runtime-only check would miss its C++ integration.
            metadata = json.loads(run([torch_python or sys.executable, '-c',
                'import json, sysconfig, torch; from torch.utils.cpp_extension import include_paths; '
                'print(json.dumps(dict(includes=include_paths()+[sysconfig.get_path("include")], '
                'abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))))'], capture=True))
            for arm in VARIANTS[1:]:
                src, dest = root/arm, out/arm
                includes = [src, src/'src', src/'include', Path(cuda_home)/'include',
                            src/'3rdparty/cutlass/include', src/'3rdparty/cutlass/tools/util/include']
                for unit, relative in [('wrapper', ma6_sources.WRAPPER), ('pybind', ma6_sources.PYBIND)]:
                    run([cxx, '-std=c++17', '-O0', '-fPIC', '-DFLUX_TACO_BASELINE',
                         '-D_GLIBCXX_USE_CXX11_ABI='+str(metadata['abi']),
                         *['-I'+str(p) for p in includes], *['-I'+p for p in metadata['includes']],
                         '-c', src/relative, '-o', dest/(unit+'.o')])
                    result['checks'][arm][('ths_' if unit == 'wrapper' else '') + unit + '_object'] = True
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
    for role in ma6_sources.ROLES:
        p.add_argument('--'+role+'-threads', type=int, default=ma6_sources.DEFAULT_TUNING[role]['decoder_threads'])
        p.add_argument('--'+role+'-blocks', type=int, default=ma6_sources.DEFAULT_TUNING[role]['decoder_blocks'])
        p.add_argument('--'+role+'-rows', type=int, default=ma6_sources.DEFAULT_TUNING[role]['decoder_rows'])
    p = sub.add_parser('verify'); p.add_argument('--build', type=Path, required=True); p.add_argument('--built', action='store_true')
    p = sub.add_parser('build'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True); p.add_argument('--jobs', type=int, default=2); p.add_argument('--nccl-root', type=Path)
    p = sub.add_parser('compile-check'); p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--cuda-home', type=Path)
    p.add_argument('--host-only', action='store_true'); p.add_argument('--cxx', default='g++')
    p.add_argument('--torch-python', type=Path, help='Python with PyTorch headers for the native wrapper check')
    args = parser.parse_args()
    if args.action == 'prepare':
        tuning = {role:dict(decoder_threads=getattr(args,role+'_threads'),
                            decoder_blocks=getattr(args,role+'_blocks'),
                            decoder_rows=getattr(args,role+'_rows')) for role in ma6_sources.ROLES}
        print(prepare(args.out, tuning))
    elif args.action == 'verify': verify(args.build, args.built)
    elif args.action == 'build': compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)
    else: compile_check(args.build, args.out, args.cuda_home, args.cxx, args.host_only, args.torch_python)


if __name__ == '__main__':
    main()
