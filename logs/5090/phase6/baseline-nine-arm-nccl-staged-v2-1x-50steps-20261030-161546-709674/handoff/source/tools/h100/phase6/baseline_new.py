#!/usr/bin/env python3
"""One contemporaneous five-arm H100 TP8/DP1 baseline. Preview by default."""
import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

import ablation4_runtime as communication
import baseline_new_build as builder
import baseline_new_profile as profiler
import baseline_new_sources as sources
import baseline_new_protocol as protocol
from baseline_new_protocol import (ARMS, SELECTIVE, POLICY, NATIVE_ARM, MODEL, PROTOCOL,
    read, write, require, sha, schedule, jobs, identity, equivalent, validate_rows, summarize)
from common import HERE, REPO, verify_files
from detach_ablation5_run_two import detach
from run import arguments as model_arguments
from tp8_dp2_train import run_process


def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    p.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--build-cuda-home',type=Path,default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--baseline-build',type=Path,help='Reuse only a verified build of this five-arm protocol')
    p.add_argument('--trials',type=int,choices=(1,2),default=1)
    p.add_argument('--blocks',type=int,default=10,help='Multiple of 10, per independent trial')
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--worker-timeout',type=int,default=1800)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--foreground',action='store_true')
    p.add_argument('--report',action='store_true',help='Offline timing replay; verify saved profile evidence hashes')
    a=p.parse_args(argv)
    schedule(a.trials,a.blocks)
    require(a.jobs>0 and a.worker_timeout>300,'Invalid build jobs or worker timeout')
    require(not (a.report and a.execute) and (not a.report or a.out),'--report requires --out and no --execute')
    if a.out is None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        a.out=REPO/'logs/h100/phase6'/f'baseline-new-{a.trials}x-50steps-{stamp}'
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.expanduser().absolute() if key=='python' else value.expanduser().resolve())
    return a


def controller_files():
    # Bounded import closure, plus templates read indirectly by existing overlays.
    import export_tp8_dp2
    import restore_container
    modules=[m for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and
             str(m.__file__).endswith('.py') and Path(m.__file__).resolve().is_relative_to(HERE)]
    paths={Path(m.__file__).resolve() for m in modules}
    # Registered overlays also read base controllers and templates as text.
    paths.update(value.resolve() for m in modules for value in vars(m).values()
                 if isinstance(value,Path) and value.is_file() and value.resolve().is_relative_to(REPO))
    paths.update(HERE/n for n in (*builder.ma11_build.INPUTS,'run_baseline_new.sh',
                                  'ablation6_gemm_only_generator.cc','baseline_codec.cu','basic_quant_receiver.cuh'))
    paths.update(REPO/'tools/a800/phase5/e2e'/n for n in ('config.json','adapter.py','megatron_taco.py','worker.py')
                 if (REPO/'tools/a800/phase5/e2e'/n).is_file())
    paths.add(REPO/'tools/a800/phase4/reference.py')
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(paths)}


def verify_controllers(root, live=True):
    files=read(root/'handoff/controllers.json')
    verify_files(root/'handoff/source',files)
    if live:verify_files(REPO,files)


def gpu_idle():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                  '--format=csv,noheader,nounits'],text=True)
    require(not value.strip(),'GPU compute processes already present; require eight idle GPUs:\n'+value)


def capture_communication():
    saved=dict(os.environ)
    try:
        for key in list(os.environ):
            if communication.relevant(key):del os.environ[key]
        return communication.capture('0')
    finally:os.environ.clear();os.environ.update(saved)


def worker_environment(a, build, scripts, job, dest, plan):
    arm=job['arm']
    env={k:v for k,v in os.environ.items() if not k.startswith(('NCCL_','TORCH_NCCL_','H100_','E2E_','FLUX_'))}
    for key in ('PYTHONHOME','PYTHONOPTIMIZE','MECH_SMOKE'):env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        CUDA_DEVICE_MAX_CONNECTIONS='1',FLUX_GEMM_RS_GEMM_ONLY='1',H100_BASELINE_NEW_EXECUTE='1',
        H100_TP='8',H100_SM_COUNT='132',H100_RUN_ID=plan['run_id'],
        H100_MA4_F_ARM='F4',H100_MA4_F_PROTOCOL='h100-ma4-f-v1',
        H100_MA11_ARM=NATIVE_ARM.get(arm,'F4-cache-zcount'),H100_MA11_PROTOCOL='h100-ma11-causal-softmax-v1',
        E2E_POLICY=POLICY[arm],E2E_MODE=job['mode'],E2E_CASE='h100-phase6',
        E2E_BUILD=str(scripts/'build'),E2E_PROCESS_OUT=str(dest),E2E_TRIAL=str(job['trial']),
        E2E_BLOCK=str(job['block']),E2E_WINDOW=str(job['index']),
        CUDA_HOME=str(a.cuda_home),PATH=f'{a.python.parent}:{a.cuda_home}/bin:'+env.get('PATH',''))
    py=[scripts,build/'megatron'];ld=[a.cuda_home/'lib64']
    if arm!='megatron-cuda':
        native=builder.native_root(build,arm)
        py.append(native/'python');ld=[native/'python/flux/lib',native,native/'build/lib']+ld
    env['PYTHONPATH']=':'.join(map(str,py))
    # Preserve the CUDA container's driver paths, excluding any other Flux
    # library directory from the caller's environment.
    for value in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        if value and not any((Path(value)/n).exists() for n in ('libflux_cuda.so','libflux_cuda_ths_op.so')):
            if Path(value) not in ld:ld.append(Path(value))
    env['LD_LIBRARY_PATH']=':'.join(map(str,ld))
    return communication.apply(env,plan['communication'])


def check_worker(root, arm, worker):
    scripts=root/'workers'/arm/'scripts'
    verify_files(scripts,worker['files'])
    require(sha(scripts.parent/'plan.json')==worker['plan_sha256'],'Worker plan changed')
    return scripts


def window(root, job, plan, workers):
    dest=root/'results'/job['directory']
    command=read(dest/'command.json')
    require(command['job']==job,'Worker command/job changed')
    check_worker(root,job['arm'],workers[job['arm']])
    rows=[read(dest/f'rank{i}.json') for i in range(8)]
    return validate_rows(rows,job,plan,workers[job['arm']])


def replay(root):
    """Read-only reconstruction, including smoke gates and selective equivalence."""
    root=Path(root);verify_controllers(root)
    plan,workers=read(root/'plan.json'),read(root/'workers.json')
    require(plan['protocol']==PROTOCOL and plan['arms']==list(ARMS) and plan['model']==MODEL and
            plan['schedule']==schedule(plan['trials'],plan['blocks']),'Wrong baseline plan')
    require(read(root/'preflight.json')['completed'] is True,'Preflight incomplete')
    profiles=read(root/'profile-summary.json')
    require(profiles['completed'] and profiles['execution_validated'] and profiles['run_id']==plan['run_id'] and
            set(profiles['profiles'])=={f'{a}/rank{r}' for a in ARMS for r in range(8)},'All five profiles required')
    require(all(x.get('execution_validated') is True and key==f"{x.get('arm')}/rank{x.get('rank')}"
                for key,x in profiles['profiles'].items()),'Incomplete rank profile validation')
    verify_files(root,profiles['evidence_files'])
    initial=None;references={};samples=[];records={}
    for job in jobs(plan['trials'],plan['blocks']):
        rows=window(root,job,plan,workers)
        current=identity(rows)
        if initial is None:initial=current
        require(current==initial,'Hardware/software/weights/RNG/input changed between windows')
        key=(job['trial'],job['block'],job['mode'])
        if job['arm'] in SELECTIVE:
            if key in references:equivalent(rows,references[key],job['mode']=='smoke')
            else:references[key]=rows
        if job['mode']=='timing':samples.append(summarize(rows,job))
        for r in range(8):
            name=f"results/{job['directory']}/rank{r}.json";records[name]=sha(root/name)
    result=protocol.report(samples,plan['trials'],plan['blocks'])
    result.update(run_id=plan['run_id'],correctness_passed=True,execution_path_validated=True,
                  selective_trajectories_identical=True,hardware=initial[0]['hardware'],
                  worker_records=records,profile_summary_sha256=sha(root/'profile-summary.json'),
                  plan_sha256=sha(root/'plan.json'),build=plan['build'])
    return result


class Peer:
    def check(self):pass


def supervise(a):
    root=a.out;state=read(root/'batch-state.json')
    def stage(name):
        state.update(status='running',stage=name);write(root/'batch-state.json',state)
        print('STAGE',name,flush=True)
    try:
        verify_controllers(root);gpu_idle()
        stage('environment-preflight')
        from restore_container import GPU_PROBE, IMPORT_PROBE, APEX_PROBE
        check=(IMPORT_PROBE+'\n'+APEX_PROBE+'\n'+
               'assert sys.version_info[:2] == (3,10)\nassert tuple(torch.cuda.nccl.version()) == (2,21,5)\n'+GPU_PROBE)
        env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
        run_process([a.python,'-c',check],REPO,env,root/'environment-preflight.log',180,Peer())
        build=a.baseline_build or root/'build'
        if a.baseline_build:
            stage('verify-build');built=builder.verify(build,True)
            require(built['environment']==builder.environment_identity(a.build_cuda_home),'Build environment changed')
        else:
            stage('prepare-build');builder.prepare(build)
            stage('compile');built=builder.compile_build(build,a.megatron,a.build_cuda_home,a.jobs)
        comm=capture_communication()
        plan=dict(protocol=PROTOCOL,run_id=uuid.uuid4().hex,arms=list(ARMS),model=MODEL,
                  trials=a.trials,blocks=a.blocks,schedule=schedule(a.trials,a.blocks),communication=comm,
                  build=dict(root=str(build),manifest_sha256=sha(build/'baseline-build.json')))
        write(root/'plan.json',plan)
        stage('prepare-workers');workers=sources.prepare_workers(build,root,plan);write(root/'workers.json',workers)
        smoke={};initial=None;references={};samples=[]
        profiles=dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=False,execution_validated=False,
                      profiles={},evidence_files={})
        for job in jobs(a.trials,a.blocks):
            if job['mode']!='smoke':require(set(smoke)==set(ARMS),'All five smokes must pass first')
            verify_controllers(root)
            require(sha(build/'baseline-build.json')==plan['build']['manifest_sha256'],'Build identity changed')
            verify_files(build,built['files'])
            scripts=check_worker(root,job['arm'],workers[job['arm']]);gpu_idle()
            dest=root/'results'/job['directory'];dest.mkdir(parents=True)
            command=[a.python,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','--max-restarts=0',
                     scripts/'worker.py',*model_arguments(MODEL),'--distributed-timeout-minutes','5']
            write(dest/'command.json',dict(job=job,command=list(map(str,command))))
            stage(job['directory'])
            run_process(command,build/'megatron',worker_environment(a,build,scripts,job,dest,plan),
                        dest/'process.log',a.worker_timeout,Peer())
            rows=window(root,job,plan,workers)
            current=identity(rows)
            if initial is None:initial=current
            require(current==initial,'Hardware/software/weights/RNG/input changed between windows')
            key=(job['trial'],job['block'],job['mode'])
            if job['arm'] in SELECTIVE:
                if key in references:equivalent(rows,references[key],job['mode']=='smoke')
                else:references[key]=rows
            if job['mode']=='smoke':
                smoke[job['arm']]=True
                write(root/'preflight.json',dict(completed=set(smoke)==set(ARMS),arms=smoke,
                      selective_equivalence=set(SELECTIVE)<=set(smoke)))
            elif job['mode']=='timing':
                samples.append(summarize(rows,job));write(root/'samples.json',samples)
            else:
                for rank in range(8):
                    trace=dest/f'rank{rank}-profile.json.gz'
                    with gzip.open(trace,'rt') as f:events=json.load(f)['traceEvents']
                    profiles['profiles'][f"{job['arm']}/rank{rank}"]=profiler.analyze(events,job['arm'],rank)
                    profiles['evidence_files'][str(trace.relative_to(root))]=sha(trace)
                write(root/'profile-summary.json',profiles)
        profiles.update(completed=True,execution_validated=True);write(root/'profile-summary.json',profiles)
        verify_controllers(root)
        result=replay(root)
        write(root/'report.json',result);(root/'report.md').write_text(protocol.markdown(result))
        state.update(status='completed',stage='done',timing_windows=result['timing_windows'],
                     formal_optimizer_steps=result['formal_optimizer_steps'],report=str(root/'report.md'))
        write(root/'batch-state.json',state);print('COMPLETED',root/'report.md',flush=True)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(root/'batch-state.json',state);raise


def main(argv=None):
    a=arguments(argv)
    if a.report:
        result=replay(a.out)
        require(result==read(a.out/'report.json'),'Saved report differs from raw records')
        print(protocol.markdown(result));return
    print('Batch:',a.out,flush=True)
    if not a.execute:
        print(json.dumps(dict(preview=True,arms=list(ARMS),model=MODEL,trials=a.trials,blocks=a.blocks,
            timing_windows=a.trials*a.blocks*5,formal_optimizer_steps=a.trials*a.blocks*250,
            warmup_steps=10,timed_steps=50,python=str(a.python),megatron=str(a.megatron),
            workflow='build -> five full-model smokes -> balanced five-arm timing -> five profiles -> report'),indent=2))
        return
    require(a.python.is_file() and a.megatron.is_dir() and (a.cuda_home/'lib64').is_dir() and
            (a.build_cuda_home/'bin/nvcc').is_file(),'Python/Megatron/CUDA missing')
    if a.python.absolute()!=Path(sys.executable).absolute():
        os.execv(str(a.python),[str(a.python),str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)])
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'handoff').mkdir()
    write(a.out/'batch-state.json',dict(protocol=PROTOCOL,status='launching',stage='freeze-controllers'))
    try:
        files=controller_files()
        for name in files:
            dst=a.out/'handoff/source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dst)
        write(a.out/'handoff/controllers.json',files)
        write(a.out/'handoff/launch.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
        if a.foreground:return supervise(a)
        print(json.dumps(detach(a,supervisor=supervise),ensure_ascii=False),flush=True)
        print('Started; inspect batch-state.json and handoff/container-supervisor.log.',flush=True)
    except BaseException as error:
        state=read(a.out/'batch-state.json')
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(a.out/'batch-state.json',state);raise


if __name__=='__main__':main()
