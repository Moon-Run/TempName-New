"""Three orders on the SAME frozen fused V2 compute/codec/IPC implementation."""
import copy
import os
from pathlib import Path
import shutil
import subprocess

from fused_support import HERE7, PHASE6, REPO
import build as phase6_build
import scope_build
import fused_plan
from ablation_plan import FREEZE
from common import read, require, sha, verify_files, write
from fused_protocol import PROTOCOL, SHAPES
from scope_plan import model_config

ORDERS = ('base', 'legacy', 'improved')
NAME = 'scope-candidate'
ORDER_FILES = {scope_build.SWIZZLE, 'phase6_host.hpp'}


def child_plan(plans, order):
    return dict(protocol=PROTOCOL, order=order, source_only=plans['source_only'],
                roles=fused_plan.order_entries(plans, order),
                calibration_basis='original measured BF16; derived order is not recalibrated',
                gpu_validated=False)


def prepare(out, attn=None, mlp=None, source_only=False):
    plans = fused_plan.bundle(attn, mlp, source_only)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # This produces source files only, from the same measured candidate used by
    # Phase6. It never copies a compiled library or rewrites the measured source.
    scope_build.prepare(out / 'legacy', attn, mlp, source_only)
    shutil.copytree(out / 'legacy', out / 'base', symlinks=True)
    shutil.copytree(out / 'legacy', out / 'improved', symlinks=True)
    write(out / 'plans.json', plans)
    upstream = (FREEZE / 'source' / scope_build.SWIZZLE).read_text()
    children = {}
    for order in ORDERS:
        root = out / order
        m = read(root / 'manifest.json')
        variant = m['variants'][NAME]
        src = root / NAME
        original = dict(variant['source_sha256'])
        entries = fused_plan.order_entries(plans, order)
        (src / scope_build.SWIZZLE).write_text(scope_build.swizzle(upstream, entries))
        (src / 'phase6_host.hpp').write_text('#pragma once\n' + scope_build.tables(entries))
        after = {name: sha(src / name) for name in original}
        require({name for name in original if original[name] != after[name]} <= ORDER_FILES,
                'Fused Phase7 must not replace GEMM, codec, IPC, barriers or wrappers')
        variant.update(source_sha256=after, phase7_order=order,
                       phase6_source_sha256=original,
                       phase7_changes={name: dict(before=original[name], after=after[name])
                                       for name in original if original[name] != after[name]})
        write(root / 'plans.json', child_plan(plans, order))
        identity = dict(protocol=PROTOCOL, order=order, source_only=source_only, sampling=False,
                        status='prepared', derived_order_gpu_validated=False)
        write(root / 'identity.json', identity)
        m.update(protocol=PROTOCOL, kind='phase7_fused_candidate', gpu_validated=False)
        m['files'] = {name: sha(root / name) for name in ('plans.json', 'identity.json', 'config.json')}
        m['inputs'].update({str(HERE7 / name): sha(HERE7 / name) for name in
                           ('fused_support.py', 'fused_build.py', 'fused_plan.py', 'fused_protocol.py')})
        write(root / 'manifest.json', m)
        children[order] = m
    registry = dict(protocol=PROTOCOL, source_only=source_only, diagnostic_only=source_only,
                    gpu_validated=False, config=model_config(),
                    parent_source_commit=children['legacy']['parent_source_commit'],
                    builds={order: dict(path=order, source_sha256=children[order]['variants'][NAME]['source_sha256'])
                            for order in ORDERS},
                    files={'plans.json': sha(out / 'plans.json')},
                    compute_invariant='Only threadblock_swizzle.hpp and phase6_host.hpp differ between orders',
                    inputs={str(HERE7 / name): sha(HERE7 / name) for name in
                            ('fused_support.py', 'fused_build.py', 'fused_plan.py', 'fused_protocol.py')})
    write(out / 'fused-manifest.json', registry)
    verify(out)
    return out


def validate_snapshots(record, plans, children, built=False, verify_evidence=True):
    require(record['protocol'] == PROTOCOL and set(record['builds']) == set(ORDERS), 'Wrong fused registry')
    fused_plan.validate(plans, verify_evidence=verify_evidence)
    require(record['source_only'] == record['diagnostic_only'] == plans['source_only'], 'Rehearsal mode changed')
    require(record['config'] == model_config(), 'Wrong frozen model')
    common_compute = None
    for order in ORDERS:
        child = children[order]
        require(child['protocol'] == PROTOCOL and child['kind'] == 'phase7_fused_candidate' and
                child['config'] == record['config'] and child['source_only'] == record['source_only'] and
                child['diagnostic_only'] == record['source_only'] and set(child['variants']) == {NAME},
                'Wrong child build/configuration')
        require(child['parent_source_commit'] == record['parent_source_commit'], 'Compute ancestry changed')
        v = child['variants'][NAME]
        require(v['role'] == 'candidate' and v['sampling'] is False and v['phase7_order'] == order and
                v['source_sha256'] == record['builds'][order]['source_sha256'], 'Wrong candidate source identity')
        require(set(v['source_sha256']) == set(v['phase6_source_sha256']) and
                {name for name in v['source_sha256'] if v['source_sha256'][name] != v['phase6_source_sha256'][name]} <= ORDER_FILES,
                'Non-order computation differs from Phase6')
        compute = {name: value for name, value in v['source_sha256'].items() if name not in ORDER_FILES}
        if common_compute is None:
            common_compute = compute
        require(compute == common_compute, 'Three orders do not share the same compute/codec/IPC')
        if built:
            require(child['status'] == 'built' and not record['source_only'], 'Unbuilt/diagnostic library')


def verify(out, built=False):
    out = Path(out).resolve()
    m = read(out / 'fused-manifest.json')
    verify_files(out, m['files'])
    plans = read(out / 'plans.json')
    children = {order: read(out / order / 'manifest.json') for order in ORDERS}
    validate_snapshots(m, plans, children, built=built)
    upstream = (FREEZE / 'source' / scope_build.SWIZZLE).read_text()
    for order, child in children.items():
        root, src = out / order, out / order / NAME
        verify_files(root, child['files'])
        verify_files(src, child['variants'][NAME]['source_sha256'])
        require(read(root / 'plans.json') == child_plan(plans, order), 'Derived child plan changed')
        identity = read(root / 'identity.json')
        require(identity['protocol'] == PROTOCOL and identity['order'] == order and
                identity['sampling'] is False and identity['source_only'] == m['source_only'], 'Build identity changed')
        entries = fused_plan.order_entries(plans, order)
        require((src / 'phase6_host.hpp').read_text() == '#pragma once\n' + scope_build.tables(entries), 'Host mapping mismatch')
        require((src / scope_build.SWIZZLE).read_text() == scope_build.swizzle(upstream, entries), 'CUDA mapping mismatch')
        if built:
            require(all((src / name).is_file() for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')),
                    'Compiled libraries/mapping executable missing')
    return m


def compile_build(out, cuda_home, jobs=2, nccl_root=None):
    out = Path(out).resolve()
    m = verify(out)
    require(jobs > 0 and not m['source_only'], 'A real-plan build and positive jobs are required')
    for order in ORDERS:
        if read(out / order / 'manifest.json')['status'] != 'built':
            phase6_build.compile_build(out / order, Path(cuda_home), jobs, Path(nccl_root) if nccl_root else None)
    verify(out, built=True)


def compile_check(out, destination, cuda_home, cxx='g++'):
    """Host mapping execution plus SM90 object compilation, never a GPU launch."""
    out, destination, cuda_home = map(lambda p: Path(p).resolve(), (out, destination, cuda_home))
    verify(out)
    destination.mkdir(parents=True, exist_ok=False)
    plans = read(out / 'plans.json')
    result = dict(completed=False, diagnostic_only=True, gpu_used=False, gpu_validated=False, commands=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')

    def run(cmd):
        cmd = [str(x) for x in cmd]
        result['commands'].append(cmd)
        with (destination / 'compile.log').open('a') as log:
            log.write(repr(cmd) + '\n'); log.flush()
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        for order in ORDERS:
            src = out / order / NAME
            for role, entry in fused_plan.order_entries(plans, order).items():
                coords = ','.join(str(x) for row in entry['coords'] for x in row)
                code = '#include "phase6_host.hpp"\nint main(){const int a[]={' + coords + '};\n'
                code += f'for(int r=0;r<8;++r)for(int i=0;i<512;++i)if(phase6_host_lookup(16,32,{entry["K_local"]},r,i)!=a[r*512+i])return 1;\n'
                code += 'if(phase6_host_lookup(16,32,1024,0,0)!=-1)return 2;return 0;}\n'
                path = destination / f'{order}-{role}.cc'; path.write_text(code)
                exe = destination / f'{order}-{role}'
                run([cxx, '-std=c++17', '-O2', '-I' + str(src), path, '-o', exe]); run([exe])
            includes = [src, src / 'src', src / 'include', src / '3rdparty/cutlass/include', src / '3rdparty/cutlass/tools/util/include']
            run([cuda_home / 'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                 '-gencode=arch=compute_90,code=sm_90', *['-I' + str(p) for p in includes], '-c',
                 src / 'check_mapping.cu', '-o', destination / (order + '-mapping.o')])
        # Compute source is identical for all three orders; compile its shared
        # runtime once. This does not link a Flux library or validate occupancy.
        src = out / 'legacy' / NAME
        run([cuda_home / 'bin/nvcc', '-std=c++17', '-O2', '-gencode=arch=compute_90,code=sm_90',
             '-I' + str(src / 'src'), '-c', src / 'src/gemm_rs/taco_runtime.cu', '-o', destination / 'runtime.o'])
        result.update(completed=True, objects={p.name: sha(p) for p in destination.glob('*.o')})
    finally:
        write(destination / 'validation.json', result)
    return result
