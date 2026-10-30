"""One parameterized order-only build; every non-scheduler source equals historical A."""
import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import ablation4_build
import ablation4_protocol
import ablation7_new_build as parent_helpers
import arrival_pair_order
import arrival_op2_plan as planner
import arrival_op2_protocol as routing
import build
import scope_build
from ablation_plan import FREEZE
from common import HERE, REPO, read, replace_once, require, sha, verify_files, write
from scope_plan import model_config

MANIFEST = 'arrival-op2-manifest.json'
NAMES = {routing.S: 'paired_order'}
PARENT_POLICY = ablation4_protocol.POLICIES[2]
SNAPSHOTS = parent_helpers.SNAPSHOTS
ORDER_FILES = {scope_build.SWIZZLE, 'phase6_host.hpp'}
RUNTIME = 'src/gemm_rs/taco_runtime.cu'
ALGORITHM_INPUTS = (
    'arrival_op2_build.py', 'arrival_op2_plan.py', 'arrival_op2_protocol.py',
    'arrival_pair_order.py', 'ablation7_new_build.py', 'scope_build.py', 'build.py',
)



def runtime_root(root, policy, record=None):
    root = Path(root).resolve()
    routing.definition(policy)
    if policy in NAMES:
        return root/'improved'/NAMES[policy]
    record = read(root/MANIFEST) if record is None else record
    return ablation4_build.runtime_root(record['parent_build'], PARENT_POLICY)


def scheduling_sources(base_root):
    base_root = Path(base_root)
    return arrival_pair_order.sources((base_root/scope_build.SWIZZLE).read_text(),
                                      (base_root/'phase6_host.hpp').read_text())


def expected_sources(plans):
    # Reconstruct A's exact identity-table sources from its sealed role entries.
    entries = {r: record['entry'] for r, record in routing.role_plans(plans, routing.A).items()}
    upstream = (FREEZE/'source'/scope_build.SWIZZLE).read_text()
    return arrival_pair_order.sources(scope_build.swizzle(upstream, entries),
                                      '#pragma once\n'+scope_build.tables(entries))


def prepare(out, parent_build, source_only=False):
    out, parent_build = Path(out).resolve(), Path(parent_build).resolve()
    parent = ablation4_build.verify(parent_build, built=not source_only)
    require(parent['source_only'] == source_only, 'Cannot promote a source-only parent')
    legacy = read(parent_build/'plans.json')
    plans = planner.bundle(legacy['parent'], source_only)
    candidate, fused = (read(parent_build/name/'manifest.json') for name in ('candidate', 'fused'))
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plans.json', plans)
    for name, relative in SNAPSHOTS.items():
        shutil.copy2(parent_build/relative, out/name)
    improved_root = out/'improved'
    improved_root.mkdir()
    improved = copy.deepcopy(fused)
    original_name = ablation4_build.FUSED_NAMES[PARENT_POLICY]
    original = fused['variants'][original_name]
    original_root = parent_build/'fused'/original_name
    variants = {}
    for policy, name in NAMES.items():
        dest = improved_root/name
        variant = copy.deepcopy(original)
        for relative in original['source_sha256']:
            target = dest/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_root/relative, target)
        (dest/'3rdparty').mkdir(exist_ok=True)
        (dest/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)
        for relative, text in scheduling_sources(original_root).items():
            (dest/relative).write_text(text)
            variant['source_sha256'][relative] = sha(dest/relative)
        variant.update(order=routing.definition(policy)['order'],
                       scheduling_method=routing.definition(policy)['scheduling_method'],
                       implementation=routing.definition(policy)['implementation'], python=f'{name}/python')
        variants[name] = variant
    files = {}
    for relative, value in fused['files'].items():
        if Path(relative).parts[0] not in fused['variants']:
            target = improved_root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent_build/'fused'/relative, target)
            files[relative] = value
    inputs = {str(HERE/name): sha(HERE/name) for name in ALGORITHM_INPUTS}
    for key in ('commands', 'nccl_files', 'plan_sha256'):
        improved.pop(key, None)
    improved.update(status='prepared', gpu_validated=False, files=files, inputs=inputs,
                    schedule_options=plans['schedule_options'], variants=variants,
                    algorithm='S: paired-row production order selected by same-codec calibration. '
                              'Only swizzle and host mapping differ from A; all compute/codec/barrier sources identical.')
    write(improved_root/'manifest.json', improved)
    record = dict(
        protocol=routing.PROTOCOL, config=model_config(), execution_timing=routing.execution_timing(),
        schedule_options=plans['schedule_options'], source_only=source_only,
        diagnostic_only=source_only, gpu_validated=False, parent_build=str(parent_build),
        parent_manifest=parent, parent_source_commit=parent['parent_source_commit'],
        parent_files=parent_helpers.parent_files(parent_build, parent, candidate, fused), inputs=inputs,
        improved_source_sha256={name: variant['source_sha256'] for name, variant in variants.items()},
        files={p: sha(out/p) for p in (*SNAPSHOTS, 'plans.json', 'improved/manifest.json')})
    write(out/MANIFEST, record)
    verify(out)
    return out


def validate_snapshots(record, plans, candidate, fused, improved, built=False, verify_evidence=True):
    require(record['protocol'] == routing.PROTOCOL and record['config'] == model_config() and
            record['execution_timing'] == routing.execution_timing(), 'Wrong op2 build/model/timing')
    planner.validate(plans, verify_evidence)
    require(record['schedule_options'] == improved['schedule_options'] == plans['schedule_options'] ==
            {'attn': 32, 'mlp': 32}, 'Build template must contain identity spans')
    parent = record['parent_manifest']
    legacy = parent_helpers.parent_plan(plans)
    ablation4_build.validate_snapshots(parent, legacy, candidate, fused, built, verify_evidence)
    require(record['source_only'] == record['diagnostic_only'] == plans['source_only'] ==
            parent['source_only'] == improved['source_only'] == improved['diagnostic_only'],
            'Op2 source-only flags changed')
    snapshots = {'parent-ablation4-manifest.json': parent, 'parent-candidate-manifest.json': candidate,
                 'parent-fused-manifest.json': fused, 'parent-plans.json': legacy,
                 'plans.json': plans, 'improved/manifest.json': improved}
    require(set(record['files']) == set(snapshots), 'Op2 snapshot set changed')
    for name, snapshot in snapshots.items():
        require(record['files'][name] == parent_helpers.json_sha(snapshot), 'Op2 snapshot changed: '+name)
    require(record['parent_source_commit'] == parent['parent_source_commit'], 'Parent source changed')
    require(improved['protocol'] == build.PROTOCOL and improved['config'] == model_config() and
            improved['kind'] == 'baselines' and improved['arch'] == 'sm90' and
            improved['scope_roles'] == ['attn', 'mlp'] and set(improved['variants']) == set(NAMES.values()),
            'Wrong op2 scheduler variants')
    require(improved['inputs'] == record['inputs'] and len(record['inputs']) == len(ALGORITHM_INPUTS) and
            all(sum(str(path).endswith('/'+name) for path in record['inputs']) == 1
                for name in ALGORITHM_INPUTS),
            'Op2 input seal changed')
    original = fused['variants'][ablation4_build.FUSED_NAMES[PARENT_POLICY]]['source_sha256']
    require(original[RUNTIME] == sha(HERE/'fused_v2'/RUNTIME),
            'Parent runtime is not the pinned A scalar decoder')
    for policy, name in NAMES.items():
        variant = improved['variants'][name]
        expected = dict(original, **{p: hashlib.sha256(text.encode()).hexdigest()
                                    for p, text in expected_sources(plans).items()})
        require(variant['source_sha256'] == expected == record['improved_source_sha256'][name],
                'Op2 source change exceeds the two scheduling files; codec changes are forbidden')
        require({p:h for p,h in variant['source_sha256'].items() if p not in ORDER_FILES} ==
                {p:h for p,h in original.items() if p not in ORDER_FILES},
                'Non-scheduling implementation changed')
        require(variant['role'] == 'fused' and variant['base'] == 'interleaved' and
                variant['sampling'] is False and variant['python'] == f'{name}/python' and
                all(variant[key] == routing.definition(policy)[key]
                    for key in ('order', 'scheduling_method', 'implementation')), 'Op2 variant identity changed')
    if built:
        require(not record['source_only'] and improved['status'] == 'built', 'Unbuilt/diagnostic op2 build')
    return record


def verify(root, built=False):
    root = Path(root).resolve()
    record = read(root/MANIFEST)
    require(not built or not record['source_only'], 'Source-only op2 build cannot execute')
    verify_files(root, record['files'])
    verify_files('/', record['parent_files'])
    verify_files('/', record['inputs'])
    parent_root = Path(record['parent_build'])
    parent = ablation4_build.verify(parent_root, built=not record['source_only'])
    require(parent == record['parent_manifest'], 'Parent build changed')
    for name, relative in SNAPSHOTS.items():
        require(sha(parent_root/relative) == record['files'][name], 'Parent snapshot differs: '+name)
    candidate, fused, improved = (read(root/name) for name in
        ('parent-candidate-manifest.json', 'parent-fused-manifest.json', 'improved/manifest.json'))
    require(record['parent_files'] == parent_helpers.parent_files(parent_root, parent, candidate, fused),
            'Parent file seal changed')
    verify_files(root/'improved', improved['files'])
    for name, variant in improved['variants'].items():
        verify_files(root/'improved'/name, variant['source_sha256'])
        if built:
            require(all((root/'improved'/name/p).is_file() for p in
                        ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')), 'Missing op2 binary')
    return validate_snapshots(record, read(root/'plans.json'), candidate, fused, improved,
                              built=built)


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root = Path(root).resolve()
    record = verify(root)
    require(type(jobs) is int and jobs > 0, 'Positive build jobs required')
    try:
        if read(root/'improved/manifest.json')['status'] != 'built':
            build.compile_build(root/'improved', Path(cuda_home), jobs,
                                None if nccl_root is None else Path(nccl_root))
    finally:
        record['files']['improved/manifest.json'] = sha(root/'improved/manifest.json')
        write(root/MANIFEST, record)
    verify(root, built=not record['source_only'])


def compile_check(root, out, cuda_home, cxx='g++'):
    """Host table and SM90 object checks only. Never initialize a CUDA context."""
    root, out, cuda_home = map(lambda p: Path(p).resolve(), (root, out, cuda_home))
    verify(root)
    out.mkdir(parents=True, exist_ok=False)
    plans = planner.bundle(read(root/'plans.json')['parent'], source_only=True, spans={'attn': 4, 'mlp': 1}) if read(root/MANIFEST)['source_only'] else planner.bundle(read(root/'plans.json')['parent'], spans={'attn': 4, 'mlp': 1})
    result = dict(completed=False, gpu_used=False, gpu_validated=False, commands=[])
    def run(command):
        command = list(map(str, command))
        result['commands'].append(command)
        with (out/'compile.log').open('a') as log:
            log.write(repr(command)+'\n')
            log.flush()
            subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', H100_OP2_ATTN_SPAN='4', H100_OP2_MLP_SPAN='1'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for policy, name in NAMES.items():
            src = runtime_root(root, policy)
            for role, record in routing.role_plans(plans, policy).items():
                values = ','.join(str(v) for row in record['entry']['coords'] for v in row)
                k = record['entry']['K_local']
                host = out/f'{name}-{role}.cc'
                host.write_text('#include "phase6_host.hpp"\nint main(){const int expected[]={'+values+'};\n'
                    'for(int r=0;r<8;++r)for(int i=0;i<512;++i)'
                    f'if(phase6_host_lookup(16,32,{k},r,i)!=expected[r*512+i])return 1;\n'
                    'if(phase6_host_lookup(16,32,1024,0,0)!=-1)return 2;}\n')
                run([cxx, '-std=c++17', '-O2', '-I'+str(src), host, '-o', host.with_suffix('')])
                run([host.with_suffix('')])
            includes = [src, src/'src', src/'include', src/'3rdparty/cutlass/include',
                        src/'3rdparty/cutlass/tools/util/include']
            for label, relative in (('mapping', 'check_mapping.cu'), ('runtime', RUNTIME)):
                run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                     '--expt-extended-lambda', '-gencode=arch=compute_90,code=sm_90',
                     *['-I'+str(p) for p in includes], '-c', src/relative,
                     '-o', out/f'{name}-{label}.o'])
        result.update(completed=True, objects={p.name: sha(p) for p in out.glob('*.o')})
    finally:
        write(out/'validation.json', result)
    return result
