"""Freeze a six/eight-policy H100 full optimizer-step comparison, CPU only."""
import argparse
from pathlib import Path
import shutil
import subprocess
import uuid

from common import (HERE, REPO, MEGATRON_COMMIT, PROTOCOL, manifest, plan, policies,
                    policy, read, replace_once, require, sha, write)


def model_worker():
    source=(REPO/'tools/a800/phase5/e2e/worker.py').read_text()
    source=replace_once(source,"assert os.environ.get('SLURM_JOB_ID') and POLICY in SPEC['policies']",
                        "assert os.environ.get('H100_RUN_ID') and POLICY in SPEC['policies']")
    source=replace_once(source,"assert all('A800' in torch.cuda.get_device_name(i) for i in range(WORLD))",
                        "from runtime import check_topology\nhardware = check_topology(WORLD)")
    source=replace_once(source,"job_id=os.environ['SLURM_JOB_ID']",
                        "run_id=os.environ['H100_RUN_ID'],hardware=hardware")
    # Native controls must not accidentally import a Flux extension either.
    source=replace_once(source,"elif POLICY == 'native_taco':", "elif POLICY in ('native', 'native_taco'):")
    anchor="report['initial_parameters_sha256'] = fingerprint(raw.named_parameters())"
    source=replace_once(source,anchor,"""report.update(local_rank=local_rank,
    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
    dp_rank=parallel_state.get_data_parallel_rank(),
    tp_group_ranks=dist.get_process_group_ranks(parallel_state.get_tensor_model_parallel_group()),
    dp_group_ranks=dist.get_process_group_ranks(parallel_state.get_data_parallel_group()))
"""+anchor)
    return source


def prepare(out,base_build,candidate_build,plan_path,megatron):
    base_build,candidate_build,plan_path,megatron=[p.resolve() for p in (base_build,candidate_build,plan_path,megatron)]
    baseline=manifest(base_build,'baselines');candidate=manifest(candidate_build,'candidates')
    c=baseline['config'];require(candidate['config']==c,'Candidate and baseline configurations differ')
    measured=plan(plan_path,c)
    require(candidate['plan_sha256']==sha(plan_path),'Candidate was compiled against a different arrival plan')
    require(subprocess.check_output(['git','-C',str(megatron),'rev-parse','HEAD'],text=True).strip()==MEGATRON_COMMIT,'Wrong Megatron commit')
    require(not subprocess.check_output(['git','-C',str(megatron),'status','--porcelain','--untracked-files=no'],text=True).strip(),'Megatron tracked files must match the pinned commit')
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    scripts=out/'scripts';scripts.mkdir();(out/'build').mkdir();(out/'results').mkdir()
    sources={}
    for name in ('adapter.py','taco_support.py','megatron_taco.py','routing.py','config.json','worker.py'):
        src=REPO/'tools/a800/phase5/e2e'/name
        shutil.copy2(src,scripts/name);sources[str(src)]=sha(src)
    (scripts/'worker.py').write_text(model_worker())
    path=scripts/'megatron_taco.py'
    path.write_text(replace_once(path.read_text(),'self.world==4','self.world in (4,8)'))
    for src in (REPO/'tools/a800/phase4/reference.py',HERE/'runtime.py'):
        shutil.copy2(src,scripts/src.name);sources[str(src)]=sha(src)
    cfg=read(scripts/'config.json')
    cfg.update(policies=policies(c),orders=[],repetitions=2,warmup_steps=c['warmup_steps'],timed_steps=c['timed_steps'],
        cases=[dict(name='h100-phase6',hidden=c['hidden'],ffn=c['ffn'],heads=c['heads'],sequence=c['sequence'],tp=c['tp'])],
        common={k:c[k] for k in ('layers','micro_batch','global_batch','seed','token_seed','vocab')},
        graph_forward=False,double_buffered=True,ring_reduction=True,
        scope='Single-node H100, full optimizer steps; native Hopper Flux reference, SM90-compiled V2 MLP fused/selective paths; BF16 STE; synthetic data, no convergence claim.')
    write(scripts/'config.json',cfg)
    shutil.copy2(plan_path,scripts/'selection-plan.json')
    variants={};build_files={}
    for name in policies(c):
        if name in ('native','native_taco'):continue
        parent=base_build if name in ('original','taco_fused') else candidate_build
        data=baseline if parent==base_build else candidate
        v=data['variants'][name]
        require(not v['sampling'],'A sampler cannot be timed as a baseline/candidate')
        expected='hopper' if name=='original' else 'fused' if name=='taco_fused' else 'candidate'
        require(v['role']==expected,'Incorrect baseline role')
        if name in ('original','taco_fused'):require(v['base']=='original','Required baseline must not be reordered')
        else:require(name==policy(v['base']),'Candidate base mismatch')
        (out/'build'/name).symlink_to(parent/name,target_is_directory=True)
        variants[name]=dict(root=str(parent/name),role=v['role'],base=v['base'])
        if name=='taco_fused':
            variants[name]['implementation']=v.get('implementation','phase6-warp-v1')
        for relative,digest in data['files'].items():
            if relative.startswith(name+'/'):build_files[str(parent/relative)]=digest
    info=dict(protocol=PROTOCOL,run_id=uuid.uuid4().hex,config=c,policies=policies(c),megatron=str(megatron),
        megatron_commit=MEGATRON_COMMIT,variants=variants,calibration_hardware=measured['hardware'],
        plan_sha256=sha(plan_path),source_inputs=sources,
        source_manifests={str(p/'manifest.json'):sha(p/'manifest.json') for p in (base_build,candidate_build)},
        build_files=build_files,files={str(p.relative_to(out)):sha(p) for p in scripts.iterdir()},
        controllers={str(p):sha(p) for p in (HERE/'common.py',HERE/'calibrate.py',HERE/'run.py',HERE/'report.py')},
        gpu_validated=False)
    write(out/'submission.json',info);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path)
    p.add_argument('--base-build',type=Path,required=True);p.add_argument('--candidate-build',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    a=p.parse_args();prepare(a.out,a.base_build,a.candidate_build,a.plan,a.megatron)
