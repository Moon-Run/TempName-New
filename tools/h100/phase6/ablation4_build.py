"""Original/A/B share the pinned fused-v2 codec; C retains frozen double buffers."""
import copy
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess

import ablation3_plan
import build
import scope_build
from ablation_plan import FREEZE
from ablation4_protocol import PROTOCOL, POLICIES, definition, is_native, is_fused_v2, role_plans
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_plan import model_config

FUSED_NAMES = dict(zip(POLICIES[1:4], ('taco_fused', 'interleaved_all', 'interleaved_arrival_all')))
ORDER_FILES = {scope_build.SWIZZLE, 'phase6_host.hpp'}


def runtime_root(root, policy):
    root = Path(root)
    require(not is_native(policy), 'Native Megatron has no Flux build')
    return root/'fused'/FUSED_NAMES[policy] if is_fused_v2(policy) else root/'candidate/scope-candidate'


def order_sources(plans, policy):
    entries = {role: p['entry'] for role, p in role_plans(plans, policy).items()}
    upstream = (FREEZE/'source'/scope_build.SWIZZLE).read_text()
    return {scope_build.SWIZZLE: scope_build.swizzle(upstream, entries),
            'phase6_host.hpp': '#pragma once\n'+scope_build.tables(entries)}


def prepare(out, attn=None, mlp=None, source_only=False):
    require(source_only or (attn is not None and mlp is not None),
            'Ablation4 requires both explicit measured role plans on this host')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', model_config())
    scope_build.prepare(out/'candidate', attn, mlp, source_only)
    plans = ablation3_plan.bundle(read(out/'candidate/plans.json')); write(out/'plans.json', plans)
    build.prepare(out/'fused', out/'config.json', only_variants=['taco_fused'])
    fused = read(out/'fused/manifest.json')
    fused.update(source_only=source_only, diagnostic_only=source_only, scope_roles=['attn', 'mlp'],
                 algorithm='Original/A/B: identical fixed fused-v2 compute/codec/frontend; only order tables differ.')
    original = fused['variants']['taco_fused']
    for policy in POLICIES[2:4]:
        name = FUSED_NAMES[policy]; src = out/'fused'/name
        shutil.copytree(out/'fused/taco_fused', src, symlinks=True)
        v = copy.deepcopy(original); v.update(base='interleaved', order=definition(policy)['order'], python=f'{name}/python')
        for path, text in order_sources(plans, policy).items():
            (src/path).write_text(text); v['source_sha256'][path] = sha(src/path)
        fused['variants'][name] = v
    fused['inputs'][str(HERE/'ablation4_build.py')] = sha(HERE/'ablation4_build.py')
    write(out/'fused/manifest.json', fused)
    candidate = read(out/'candidate/manifest.json')
    record = dict(protocol=PROTOCOL, config=model_config(), source_only=source_only,
                  diagnostic_only=source_only, gpu_validated=False,
                  parent_source_commit=candidate['parent_source_commit'],
                  candidate_manifest_sha256=sha(out/'candidate/manifest.json'),
                  fused_provenance=read(HERE/'fused_v2/provenance.json'),
                  fused_source_sha256={p: fused['variants'][n]['source_sha256'] for p, n in FUSED_NAMES.items()},
                  files={p: sha(out/p) for p in ('config.json', 'plans.json')},
                  inputs={str(HERE/p): sha(HERE/p) for p in ('ablation4_build.py', 'ablation4_protocol.py')})
    write(out/'ablation4-manifest.json', record)
    verify(out)
    return out


def validate_snapshots(record, plans, candidate, fused, built=False, verify_evidence=True):
    require(record['protocol'] == PROTOCOL and record['config'] == model_config(), 'Wrong ablation4 build/model')
    ablation3_plan.validate(plans, verify_evidence)
    require(record['source_only'] == record['diagnostic_only'] == plans['source_only'] == candidate['source_only'] ==
            candidate['diagnostic_only'] == fused['source_only'] == fused['diagnostic_only'], 'Ablation4 rehearsal flags changed')
    require(candidate['protocol'] == scope_build.PROTOCOL and candidate['kind'] == 'scope_candidate' and
            candidate['config'] == model_config() and candidate['parent_source_commit'] == record['parent_source_commit'] and
            set(candidate['variants']) == {'scope-candidate'}, 'Wrong frozen C build')
    cv = candidate['variants']['scope-candidate']
    require(cv['role'] == 'candidate' and cv['sampling'] is False and cv['base'] == 'interleaved', 'C must retain the frozen candidate')
    parent_digest = hashlib.sha256((json.dumps(plans['parent'], indent=2, ensure_ascii=False)+'\n').encode()).hexdigest()
    require(candidate['files']['plans.json'] == parent_digest, 'C parent plan snapshot changed')
    for relative, text in order_sources(plans, POLICIES[4]).items():
        require(cv['source_sha256'][relative] == hashlib.sha256(text.encode()).hexdigest(), 'C arrival order changed')
    require(fused['protocol'] == build.PROTOCOL and fused['arch'] == 'sm90' and
            fused['kind'] == 'baselines' and fused['config'] == model_config() and
            fused['scope_roles'] == ['attn', 'mlp'] and set(fused['variants']) == set(FUSED_NAMES.values()), 'Wrong fused-v2 build')
    pinned = HERE/'fused_v2'; provenance = read(pinned/'provenance.json')
    require(record['fused_provenance'] == provenance, 'Wrong pinned fused-v2 baseline')
    verify_files(pinned, provenance['files'])
    original = fused['variants']['taco_fused']['source_sha256']
    require(original[scope_build.SWIZZLE] == sha(REPO/scope_build.SWIZZLE), 'Fused-v2 must preserve original Flux swizzle')
    common = {k: h for k, h in original.items() if k not in ORDER_FILES}
    for policy, name in FUSED_NAMES.items():
        v = fused['variants'][name]
        require(v['role'] == 'fused' and v['base'] == definition(policy)['base'] and not v['sampling'] and
                v['implementation'] == 'a800-fused-v2-cta-v1' and
                v['source_sha256'] == record['fused_source_sha256'][policy], 'Fused-v2 source/order identity changed')
        require({k: h for k, h in v['source_sha256'].items() if k not in ORDER_FILES} == common,
                'Original/A/B must share all compute, codec and wrapper sources')
        if policy != POLICIES[1]:
            require(v['order'] == definition(policy)['order'], 'A/B order changed')
            for relative, text in order_sources(plans, policy).items():
                require(v['source_sha256'][relative] == hashlib.sha256(text.encode()).hexdigest(), 'Wrong A/B order table')
    for relative in provenance['files']:
        text = (pinned/relative).read_text()
        if relative.endswith('gemm_reduce_scatter.cc'):
            text = build.wrapper(text, legacy_fused=True)
        elif relative.endswith('.py'):
            text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)').replace(
                'supports SM80 only', 'requires the H100 SM90 port').replace(
                'tools/a800/phase4/build.py', 'tools/h100/phase6/build.py')
        require(original[relative] == hashlib.sha256(text.encode()).hexdigest(), 'Pinned fused-v2 kernel/codec changed: '+relative)
    if built:
        require(not record['source_only'] and fused['status'] == candidate['status'] == 'built', 'Unbuilt/diagnostic runtime')


def verify(root, built=False):
    root = Path(root).resolve(); record = read(root/'ablation4-manifest.json')
    verify_files(root, record['files'])
    candidate = scope_build.verify(root/'candidate', built, kind='scope_candidate')
    plans = read(root/'plans.json')
    require(sha(root/'candidate/manifest.json') == record['candidate_manifest_sha256'] and
            read(root/'candidate/plans.json') == plans['parent'], 'Shared C build or plans changed')
    fused = read(root/'fused/manifest.json'); verify_files(root/'fused', fused['files'])
    for name, variant in fused['variants'].items():
        verify_files(root/'fused'/name, variant['source_sha256'])
        if built:
            for path in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping'):
                require((root/'fused'/name/path).is_file(), 'Missing fused-v2 artifact: '+name+'/'+path)
    validate_snapshots(record, plans, candidate, fused, built)
    return record


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root = Path(root).resolve(); verify(root)
    require(jobs > 0, 'Positive build jobs required')
    try:
        for child in (root/'fused', root/'candidate'):
            if read(child/'manifest.json')['status'] != 'built':
                build.compile_build(child, Path(cuda_home), jobs, nccl_root)
    finally:
        # Preserve a verifiable snapshot after success or a failed compilation;
        # source/plan identity is checked again on every retry.
        record = read(root/'ablation4-manifest.json')
        record['candidate_manifest_sha256'] = sha(root/'candidate/manifest.json')
        write(root/'ablation4-manifest.json', record)
    verify(root, built=not record['source_only'])


def compile_check(root, out, cuda_home, cxx='g++'):
    root, out, cuda_home = map(lambda p: Path(p).resolve(), (root, out, cuda_home))
    verify(root); out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, gpu_used=False, gpu_validated=False, checks={})
    plans = read(root/'plans.json'); env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        result['checks']['C'] = scope_build.compile_check(root/'candidate', out/'C', cuda_home, cxx)
        for policy, name in FUSED_NAMES.items():
            src = root/'fused'/name; dest = out/name; dest.mkdir(); commands = []
            def run(cmd):
                cmd = list(map(str, cmd)); commands.append(cmd)
                with (dest/'compile.log').open('a') as log:
                    log.write(repr(cmd)+'\n'); log.flush()
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if policy != POLICIES[1]:
                for role, record in role_plans(plans, policy).items():
                    values = ','.join(str(x) for row in record['entry']['coords'] for x in row)
                    k = record['entry']['K_local']
                    host = dest/(role+'.cc')
                    host.write_text('#include "phase6_host.hpp"\nint main(){const int coords[]={'+values+'};\n'+
                        'for(int r=0;r<8;++r)for(int i=0;i<512;++i)if(phase6_host_lookup(16,32,'+str(k)+',r,i)!=coords[r*512+i])return 1;\n'+
                        'if(phase6_host_lookup(16,32,1024,0,0)!=-1)return 2;}\n')
                    run([cxx, '-std=c++17', '-O2', '-I'+str(src), host, '-o', dest/(role+'-host')]); run([dest/(role+'-host')])
            inc = [src, src/'src', src/'include', src/'3rdparty/cutlass/include', src/'3rdparty/cutlass/tools/util/include']
            for label, relative in (('mapping', 'check_mapping.cu'), ('runtime', 'src/gemm_rs/taco_runtime.cu')):
                run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                     '-gencode=arch=compute_90,code=sm_90', *['-I'+str(p) for p in inc], '-c', src/relative, '-o', dest/(label+'.o')])
            result['checks'][name] = dict(completed=True, commands=commands, objects={p.name: sha(p) for p in dest.glob('*.o')})
        result['completed'] = True
    finally:
        write(out/'validation.json', result)
    return result
