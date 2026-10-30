"""Freeze nine policies, immutable selection tables, binaries and model scripts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
from routing import POLICIES, ARRIVAL_POLICIES, base_policy, mapping_policy


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(repo,out,job,joint,plan_path,arrival_build=None,arrival_plan=None,native_quant=False,
         fused_build=None, only_policies=None, blocks=None, graph_forward=False, double_buffered=False,
         acceptance=None):
    repo,out,joint,plan_path=[p.resolve() for p in (repo,out,joint,plan_path)]
    here=Path(__file__).resolve().parent
    assert repo/'logs/a800' in out.parents
    plan=json.loads(plan_path.read_text())
    for p,h in plan['inputs'].items():assert sha(Path(p))==h,p
    assert bool(arrival_build)==bool(arrival_plan)
    policies=list(POLICIES)
    if arrival_build:
        arrival_build,arrival_plan=arrival_build.resolve(),arrival_plan.resolve()
        ordering=json.loads(arrival_plan.read_text())
        assert ordering['schema_version']==2 and ordering['score_mode']=='tail'
        for p,h in ordering['inputs'].items():assert sha(Path(p))==h,p
        policies += ARRIVAL_POLICIES
    if native_quant:policies.append('native_taco')
    if fused_build:
        fused_build=fused_build.resolve()
        policies.append('taco_fused')
    if only_policies:
        assert len(set(only_policies))==len(only_policies) and set(only_policies)<=set(policies)
        policies=list(only_policies)
    out.mkdir(parents=True,exist_ok=False)
    (out/'results').mkdir();(out/'scripts').mkdir()
    artifacts=repo/'outputs/a800/phase5'/out.name
    build=artifacts/'build';build.mkdir(parents=True,exist_ok=False)
    (build/'mapping').mkdir()
    stock=repo/'outputs/a800/phase3/three-way-build'
    separate=repo/'outputs/a800/phase4/taco-separate-20261003'
    manifest=dict(files={},policies={},source_manifests={},scope='phase5 MLP-only joint order and quantization')
    for policy in policies:
        if policy in ('native','native_taco'):continue
        source=(stock/'original' if policy=='original' else separate/'taco' if policy=='taco_separate'
                else fused_build/'taco' if policy=='taco_fused'
                else (arrival_build if policy in ARRIVAL_POLICIES else joint)/mapping_policy(policy)/'taco')
        parent=source.parent
        m=json.loads((parent/'manifest.json').read_text())
        if policy not in ('original','taco_separate'):
            if policy=='taco_fused':
                assert not m['reorder'] and m['placement']=='fused'
                assert sha(source/'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp')==m['original_swizzle_sha256']
                assert m['original_swizzle_sha256']==sha(stock/'original/overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp')
            else:
                assert m['mapping']==mapping_policy(policy) and m['placement']=='fused'
                assert sha(source/'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp')==m['swizzle_sha256']
            if policy in ARRIVAL_POLICIES:assert m['arrival_plan_sha256']==sha(arrival_plan)
        for name,h in m['files'].items():assert sha(parent/name)==h,(policy,name)
        (build/policy).symlink_to(source,target_is_directory=True)
        prefix=source.name+'/'
        manifest['files'].update({policy+'/'+name[len(prefix):]:h for name,h in m['files'].items() if name.startswith(prefix)})
        manifest['policies'][policy]=dict(library_sha256=sha(source/'libflux_cuda.so'),mapping=mapping_policy(policy))
        manifest['source_manifests'][policy]=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json'))
    mappings=sorted({mapping_policy(p) for p in policies if mapping_policy(p)!='original'})
    for policy in mappings:
        p=(arrival_build if policy.endswith('_arrival') else joint)/policy/'check_mapping'
        (build/'mapping'/policy).symlink_to(p)
        manifest['files']['mapping/'+policy]=sha(p)
    (build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('adapter.py','worker.py','launch.py','summarize.py','routing.py','taco_support.py','megatron_taco.py','config.json'):
        shutil.copy2(here/name,out/'scripts'/name)
    config_path=out/'scripts/config.json'
    config=json.loads(config_path.read_text())
    if acceptance is not None:
        assert isinstance(acceptance['version'],str) and 0 < float(acceptance['target_percent']) < 100
        config['acceptance']=dict(acceptance)
    config['policies']=policies
    config['graph_forward']=graph_forward
    assert not (graph_forward and double_buffered)
    config['double_buffered']=double_buffered
    if double_buffered:
        config['scope'] += ' Two complete IPC workspaces alternate; the next invocation publication barrier protects prior decode before reuse, removing the post-decode barrier.'
    if graph_forward:
        config['scope'] += ' Selective MLP forward uses fixed-shape CUDA graph replay; input copy, replay and output clone are timed; attention/backward/optimizer remain eager.'
    config['orders']=[policies[i:]+policies[:i] for i in range(len(policies))]
    if blocks is not None:
        assert 1<=blocks<=len(policies)
        config['orders']=config['orders'][:blocks]
        config['scope'] += ' EXPLORATORY pilot; not the final balanced acceptance campaign.'
    if arrival_build:
        config['scope'] += ' Arrival-priority variants reuse the same physical-tile selection masks; no refitting after reordering.'
        shutil.copy2(arrival_plan,out/'scripts/arrival-plan.json')
    if native_quant:
        config['scope'] += ' native_taco preserves native Megatron attention/linear GEMMs/backward and replaces MLP RS with tensor TACO plus NCCL all-to-all; no Flux kernels.'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    shutil.copy2(plan_path,out/'scripts/selection-plan.json')
    shutil.copy2(here.parent/'validate.py',out/'scripts/validate.py')
    shutil.copy2(here.parent/'validate_megatron.py',out/'scripts/validate_megatron.py')
    shutil.copy2(here.parents[1]/'phase4/validate.py',out/'scripts/validate_separate.py')
    shutil.copy2(here.parents[1]/'phase4/reference.py',out/'scripts/reference.py')
    for name in ('campaign.py','report.py','verify.py','check_mappings.py','profile_components.py'):
        shutil.copy2(here/name,out/name)
    for p in out.rglob('*.py'):ast.parse(p.read_text())
    config=json.loads((out/'scripts/config.json').read_text())
    info=dict(job_id=job,repo=str(repo),artifact_root=str(artifacts),files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
              build_manifest_sha256=sha(build/'manifest.json'),repetitions=2,blocks=len(config['orders']),policies=policies,
              mappings=mappings,codec_checks=mappings+(['separate'] if 'taco_separate' in policies else [])+(['fused'] if 'taco_fused' in policies else [])+(['native_taco'] if 'native_taco' in policies else []),
              scope='Two fresh-process repetitions within one allocation; calibrated masks frozen before timing',
              original_campaign=str(repo/'outputs/a800/arrival-v2-tp4-20261003'))
    (out/'submission.json').write_text(json.dumps(info,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','out','joint_build','plan'):p.add_argument(name,type=Path)
    p.add_argument('--job-id',type=int,required=True)
    p.add_argument('--arrival-build',type=Path)
    p.add_argument('--arrival-plan',type=Path)
    p.add_argument('--native-quant',action='store_true')
    a=p.parse_args();main(a.repo,a.out,a.job_id,a.joint_build,a.plan,a.arrival_build,a.arrival_plan,a.native_quant)
