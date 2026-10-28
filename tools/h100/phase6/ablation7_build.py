"""Reuse verified legacy C/A libraries; build only B's independent residual map."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import ablation3_plan
import ablation4_build
import ablation4_protocol
import ablation7_plan as planner
import ablation7_protocol as routing
import build
import scope_build
from ablation_plan import FREEZE
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

IMPROVED_NAME = 'improved_arrival_all'
PARENT_POLICIES = {routing.C: ablation4_protocol.POLICIES[4], routing.A: ablation4_protocol.POLICIES[2]}
SNAPSHOTS = {'parent-ablation4-manifest.json': 'ablation4-manifest.json',
             'parent-candidate-manifest.json': 'candidate/manifest.json',
             'parent-fused-manifest.json': 'fused/manifest.json', 'parent-plans.json': 'plans.json'}


def json_sha(value):
    return hashlib.sha256((json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode()).hexdigest()


def runtime_root(root, policy, record=None):
    root = Path(root).resolve(); routing.definition(policy)
    if policy == routing.B:
        return root/'improved'/IMPROVED_NAME
    if record is None:
        record = read(root/'ablation7-manifest.json')
    return ablation4_build.runtime_root(record['parent_build'], PARENT_POLICIES[policy])


def order_sources(plans):
    entries = {role: record['entry'] for role, record in routing.role_plans(plans, routing.B).items()}
    upstream = (FREEZE/'source'/scope_build.SWIZZLE).read_text()
    return {scope_build.SWIZZLE: scope_build.swizzle(upstream, entries),
            'phase6_host.hpp': '#pragma once\n'+scope_build.tables(entries)}


def parent_plan(plans):
    """Pure reconstruction also works when only returned snapshots are available."""
    parent = plans['parent']
    return dict(protocol=ablation3_plan.PROTOCOL, parent=copy.deepcopy(parent), source_only=parent['source_only'],
                gpu_calibrated=parent['gpu_calibrated'],
                variants={p: ablation3_plan.derive(parent, p) for p in ablation3_plan.POLICIES})


def parent_files(root, record, candidate, fused):
    files = {str(root/'ablation4-manifest.json'): sha(root/'ablation4-manifest.json')}
    files.update({str(root/p): h for p, h in record['files'].items()})
    for folder, manifest in (('candidate', candidate), ('fused', fused)):
        child = root/folder
        files[str(child/'manifest.json')] = sha(child/'manifest.json')
        files.update({str(child/p): h for p, h in manifest['files'].items()})
        for name, variant in manifest['variants'].items():
            files.update({str(child/name/p): h for p, h in variant['source_sha256'].items()})
    return files


def prepare(out, parent_build, source_only=False):
    parent_build = Path(parent_build).resolve(); out = Path(out).resolve()
    parent = ablation4_build.verify(parent_build, built=not source_only)
    require(parent['source_only'] == source_only, 'Cannot promote a source-only parent or mix build modes')
    legacy = read(parent_build/'plans.json')
    plans = planner.bundle(legacy['parent'], source_only)
    candidate = read(parent_build/'candidate/manifest.json'); fused = read(parent_build/'fused/manifest.json')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plans.json', plans)
    for name, relative in SNAPSHOTS.items():
        shutil.copy2(parent_build/relative, out/name)
    improved_root = out/'improved'; improved_root.mkdir()
    improved = copy.deepcopy(fused)
    old_a = ablation4_build.FUSED_NAMES[ablation4_protocol.POLICIES[2]]
    variant = copy.deepcopy(fused['variants'][old_a]); source = parent_build/'fused'/old_a
    dest = improved_root/IMPROVED_NAME
    # Copy registered sources only. Existing binaries/install trees must never
    # masquerade as a newly compiled B after its device table has changed.
    for relative in variant['source_sha256']:
        target = dest/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target)
    (dest/'3rdparty').mkdir(exist_ok=True)
    (dest/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)
    for relative, text in order_sources(plans).items():
        (dest/relative).write_text(text); variant['source_sha256'][relative] = sha(dest/relative)
    variant.update(order='arrival', arrival_method='tail_residual', python=f'{IMPROVED_NAME}/python')
    files = {}
    for relative, value in fused['files'].items():
        if Path(relative).parts[0] not in fused['variants']:
            target = improved_root/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent_build/'fused'/relative, target); files[relative] = value
    inputs = {str(HERE/name): sha(HERE/name) for name in (
        'ablation7_build.py', 'ablation7_plan.py', 'ablation7_protocol.py', 'ablation5_plan.py',
        'arrival_tail.py', 'plan.py', 'scope_build.py', 'build.py')}
    for key in ('commands', 'nccl_files', 'plan_sha256'):
        improved.pop(key, None)
    improved.update(status='prepared', gpu_validated=False, files=files, inputs=inputs,
                    variants={IMPROVED_NAME: variant},
                    algorithm='Ablation7 B: unchanged fixed fused-v2 compute/codec; independent tail_residual order only.')
    write(improved_root/'manifest.json', improved)
    record = dict(protocol=routing.PROTOCOL, config=model_config(), execution_timing=routing.execution_timing(),
        source_only=source_only, diagnostic_only=source_only, gpu_validated=False,
        parent_build=str(parent_build), parent_manifest=parent, parent_source_commit=parent['parent_source_commit'],
        parent_files=parent_files(parent_build, parent, candidate, fused), inputs=inputs,
        improved_source_sha256=variant['source_sha256'],
        files={p: sha(out/p) for p in (*SNAPSHOTS, 'plans.json', 'improved/manifest.json')})
    write(out/'ablation7-manifest.json', record)
    verify(out)
    return out


def validate_snapshots(record, plans, candidate, fused, improved, built=False, verify_evidence=True):
    require(record['protocol'] == routing.PROTOCOL and record['config'] == model_config() and
            record['execution_timing'] == routing.execution_timing(), 'Wrong ablation7 build/model/timing')
    planner.validate(plans, verify_evidence)
    legacy = parent_plan(plans); parent = record['parent_manifest']
    require(record['source_only'] == record['diagnostic_only'] == plans['source_only'] == parent['source_only'] ==
            improved['source_only'] == improved['diagnostic_only'], 'Ablation7 rehearsal flags changed')
    ablation4_build.validate_snapshots(parent, legacy, candidate, fused, built=built, verify_evidence=verify_evidence)
    snapshots = {'parent-ablation4-manifest.json': parent, 'parent-candidate-manifest.json': candidate,
                 'parent-fused-manifest.json': fused, 'parent-plans.json': legacy,
                 'plans.json': plans, 'improved/manifest.json': improved}
    require(set(record['files']) == set(snapshots), 'Build snapshot set changed')
    for name, snapshot in snapshots.items():
        require(record['files'][name] == json_sha(snapshot), 'Build snapshot hash changed: '+name)
    require(record['parent_source_commit'] == parent['parent_source_commit'], 'Original C source commit changed')
    require(improved['protocol'] == build.PROTOCOL and improved['config'] == model_config() and
            improved['kind'] == 'baselines' and improved['arch'] == 'sm90' and
            improved['scope_roles'] == ['attn', 'mlp'] and set(improved['variants']) == {IMPROVED_NAME},
            'Wrong improved B build')
    v = improved['variants'][IMPROVED_NAME]
    require(v['role'] == 'fused' and v['base'] == 'interleaved' and not v['sampling'] and
            v['implementation'] == 'a800-fused-v2-cta-v1' and v['order'] == 'arrival' and
            v['arrival_method'] == 'tail_residual' and v['python'] == f'{IMPROVED_NAME}/python' and
            v['source_sha256'] == record['improved_source_sha256'], 'Improved B source identity changed')
    a = fused['variants'][ablation4_build.FUSED_NAMES[ablation4_protocol.POLICIES[2]]]['source_sha256']
    expected = dict(a, **{name: hashlib.sha256(text.encode()).hexdigest() for name, text in order_sources(plans).items()})
    require(v['source_sha256'] == expected, 'A/B compute/codec/wrapper differ, or B mapping changed')
    if built:
        require(not record['source_only'] and improved['status'] == 'built', 'Unbuilt/diagnostic B cannot execute')
    return record


def verify(root, built=False):
    root = Path(root).resolve(); record = read(root/'ablation7-manifest.json')
    if built:
        require(not record['source_only'] and not record['diagnostic_only'], 'Unbuilt/diagnostic runtime cannot execute')
    verify_files(root, record['files']); verify_files('/', record['parent_files'])
    parent_root = Path(record['parent_build'])
    parent = ablation4_build.verify(parent_root, built=not record['source_only'])
    require(parent == record['parent_manifest'], 'Parent ablation4 manifest changed')
    for name, relative in SNAPSHOTS.items():
        require(sha(parent_root/relative) == record['files'][name], 'Parent snapshot differs: '+relative)
    candidate = read(root/'parent-candidate-manifest.json'); fused = read(root/'parent-fused-manifest.json')
    require(record['parent_files'] == parent_files(parent_root, parent, candidate, fused), 'Parent file seal changed')
    improved = read(root/'improved/manifest.json')
    verify_files(root/'improved', improved['files'])
    for name, variant in improved['variants'].items():
        verify_files(root/'improved'/name, variant['source_sha256'])
        if built:
            for relative in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping'):
                require((root/'improved'/name/relative).is_file(), 'Missing improved B artifact: '+relative)
    return validate_snapshots(record, read(root/'plans.json'), candidate, fused, improved, built=built)


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root = Path(root).resolve(); record = verify(root)
    require(type(jobs) is int and jobs > 0, 'Positive integer build jobs required')
    try:
        if read(root/'improved/manifest.json')['status'] != 'built':
            build.compile_build(root/'improved', Path(cuda_home), jobs, None if nccl_root is None else Path(nccl_root))
    finally:
        record['files']['improved/manifest.json'] = sha(root/'improved/manifest.json')
        write(root/'ablation7-manifest.json', record)
    verify(root, built=not record['source_only'])


def compile_check(root, out, cuda_home, cxx='g++'):
    root, out, cuda_home = map(lambda p: Path(p).resolve(), (root, out, cuda_home))
    verify(root); out.mkdir(parents=True, exist_ok=False)
    src = runtime_root(root, routing.B); plans = read(root/'plans.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = dict(completed=False, gpu_used=False, gpu_validated=False, diagnostic_only=True, commands=[])
    def run(command):
        command = list(map(str, command)); result['commands'].append(command)
        with (out/'compile.log').open('a') as log:
            log.write(repr(command)+'\n'); log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for role, record in routing.role_plans(plans, routing.B).items():
            values = ','.join(str(x) for row in record['entry']['coords'] for x in row)
            k = record['entry']['K_local']; host = out/(role+'.cc')
            host.write_text('#include "phase6_host.hpp"\nint main(){const int coords[]={'+values+'};\n'+
                'for(int r=0;r<8;++r)for(int i=0;i<512;++i)if(phase6_host_lookup(16,32,'+str(k)+',r,i)!=coords[r*512+i])return 1;\n'+
                'if(phase6_host_lookup(16,32,1024,0,0)!=-1)return 2;}\n')
            run([cxx, '-std=c++17', '-O2', '-I'+str(src), host, '-o', out/(role+'-host')])
            run([out/(role+'-host')])
        inc = [src, src/'src', src/'include', src/'3rdparty/cutlass/include', src/'3rdparty/cutlass/tools/util/include']
        for label, relative in (('mapping', 'check_mapping.cu'), ('runtime', 'src/gemm_rs/taco_runtime.cu')):
            run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                 '-gencode=arch=compute_90,code=sm_90', *['-I'+str(p) for p in inc], '-c', src/relative, '-o', out/(label+'.o')])
        result.update(completed=True, objects={p.name: sha(p) for p in out.glob('*.o')})
    finally:
        write(out/'validation.json', result)
    return result
