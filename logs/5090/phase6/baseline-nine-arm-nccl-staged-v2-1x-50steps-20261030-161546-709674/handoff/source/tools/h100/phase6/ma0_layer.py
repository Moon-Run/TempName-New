#!/usr/bin/env python3
"""Single-node H100 TP8: complete MA11 versus Flux basic Attention/MLP layers."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

from common import HERE, REPO
from ma0_layer_protocol import (ARMS, MODEL, PROTOCOL, identity, region_identity, markdown,
    read, write, sha, require, report, schedule, summarize_window, validate_rows)


def arguments(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    p.add_argument('--package',type=Path,help='Reuse a verified TP8/DP2 SOURCE package; workers still run TP8 DP1')
    p.add_argument('--build-root',type=Path,help='Reuse its exact native build with --package')
    p.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--blocks',type=int,default=8)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--worker-timeout',type=int,default=3600)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--foreground',action='store_true')
    a = p.parse_args(argv)
    schedule(a.blocks)
    require(a.jobs > 0 and a.warmup >= 2 and a.iterations >= 5 and a.worker_timeout > 0,'Invalid counts/timeouts')
    require(not a.build_root or a.package,'--build-root requires --package')
    if a.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        a.out = REPO/'logs/h100/phase6'/f'ma0-layer-{stamp}'
    for key,value in vars(a).items():
        if isinstance(value,Path):
            setattr(a,key,value.expanduser().absolute() if key == 'python' else value.expanduser().resolve())
    return a


def frozen_sources():
    import ma0_layer_sources
    import export_tp8_dp2
    # Only imported controller files and explicit templates; no repository walk.
    files = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
             if getattr(m,'__file__',None) and str(m.__file__).endswith('.py')
             and Path(m.__file__).resolve().is_relative_to(HERE)}
    files.update(HERE/name for name in ('ma0_layer_worker.py','run_ma0_layer.sh'))
    return {str(p):sha(p) for p in files}


def check_files(files,root=Path('/')):
    for name,value in files.items(): require(sha(root/name) == value,'Frozen file changed: '+name)


def gpu_idle():
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader,nounits'],text=True)
    require(not text.strip(),'GPU compute processes already present; stop this campaign before mixing workloads:\n'+text)


def environment(a, scripts, build, job, dest, plan, communication):
    import ablation4_runtime
    from tp8_dp2_protocol import POLICIES
    arm = job['arm']; native = build/arm
    env = {k:v for k,v in os.environ.items() if not k.startswith(('NCCL_','TORCH_NCCL_','H100_','E2E_','FLUX_'))}
    for key in ('PYTHONOPTIMIZE','MECH_SMOKE'): env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        CUDA_DEVICE_MAX_CONNECTIONS='1',FLUX_GEMM_RS_GEMM_ONLY='1',H100_MA0_LAYER_EXECUTE='1',
        H100_TP='8',H100_SM_COUNT='132',H100_RUN_ID=plan['run_id'],
        H100_MA4_F_ARM='F4',H100_MA11_ARM='F4-cache-zcount-csoftmax',
        H100_MA11_PROTOCOL='h100-ma11-causal-softmax-v1',E2E_POLICY=POLICIES[arm],
        E2E_MODE=job['mode'],E2E_CASE='h100-phase6',E2E_BUILD=str(scripts/'build'),
        E2E_PROCESS_OUT=str(dest),E2E_BLOCK=str(job['block']),E2E_WINDOW=str(job['index']),
        CUDA_HOME=str(a.cuda_home),PATH=f'{a.python.parent}:{a.cuda_home}/bin:'+env.get('PATH',''),
        PYTHONPATH=':'.join(map(str,(scripts,build/'megatron',native/'python'))))
    env['LD_LIBRARY_PATH']=':'.join(map(str,(native/'python/flux/lib',native,native/'build/lib',a.cuda_home/'lib64')))+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return ablation4_runtime.apply(env,communication)


class Peer:
    def check(self): pass


def execute(a):
    import ablation4_runtime
    import export_tp8_dp2
    import tp8_dp2_build
    from tp8_dp2_protocol import verify_package,build_package_id
    from tp8_dp2_train import run_process
    from ma0_layer_sources import prepare_workers
    from run import arguments as model_arguments
    state_path = a.out/'batch-state.json'
    state = read(state_path)
    try:
        sources = read(a.out/'handoff/controllers.json')
        check_files(sources)
        gpu_idle()
        package = a.package or a.out/'source-package'
        state.update(status='preparing',stage='source-package');write(state_path,state)
        if a.package is None: export_tp8_dp2.export(package,a.megatron)
        info = verify_package(package)
        native_id = build_package_id(package,info)
        build = a.build_root or a.out/'build'
        state.update(stage='build');write(state_path,state)
        # Unchanged portable recipe builds the exact MA11/basic libraries.
        # Its small Megatron-CUDA codec is built but never benchmarked here.
        tp8_dp2_build.build(package,build,a.cuda_home,a.jobs,native_id)
        plan = dict(protocol=PROTOCOL,run_id=uuid.uuid4().hex,arms=list(ARMS),model=MODEL,
                    blocks=a.blocks,warmup=a.warmup,iterations=a.iterations,
                    package_id=info['package_id'],native_build_id=native_id,
                    schedule=schedule(a.blocks),source_files=sources)
        # Single-node jobs must not inherit a previous two-node NCCL_NET/HCA
        # override; retain a frozen, verified local communication environment.
        saved = dict(os.environ)
        try:
            for k in list(os.environ):
                if ablation4_runtime.relevant(k): del os.environ[k]
            communication = ablation4_runtime.capture('0')
        finally:
            os.environ.clear();os.environ.update(saved)
        plan['communication'] = communication
        write(a.out/'plan.json',plan)
        workers = prepare_workers(package,build,a.out,plan,communication)
        write(a.out/'workers.json',workers)
        jobs = [dict(arm=arm,mode='smoke',block=-1,index=i,directory='smoke-'+arm)
                for i,arm in enumerate(ARMS)] + plan['schedule']
        samples=[];expected_identity=None;expected_regions=None;smoke=set()
        for job in jobs:
            if job['mode'] == 'timing': require(smoke == set(ARMS),'Both full-model smokes required before layer timing')
            check_files(sources)
            scripts = Path(workers[job['arm']]['root'])
            check_files(workers[job['arm']]['files'],scripts)
            tp8_dp2_build.verify(build,native_id)
            gpu_idle()
            dest = a.out/'results'/job['directory'];dest.mkdir(parents=True,exist_ok=False)
            command = [a.python,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                       '--max-restarts=0',scripts/'worker.py',*model_arguments(MODEL),
                       '--distributed-timeout-minutes','5']
            state.update(status='running',stage=job['directory']);write(state_path,state)
            print('START',job['directory'],flush=True)
            write(dest/'command.json',dict(command=list(map(str,command)),job=job))
            run_process(command,build/'megatron',environment(a,scripts,build,job,dest,plan,communication),
                        dest/'process.log',a.worker_timeout,Peer())
            rows = validate_rows([read(dest/f'rank{i}.json') for i in range(8)],job,plan)
            current = identity(rows)
            if expected_identity is None: expected_identity=current
            require(current == expected_identity,'Hardware, initial parameters/RNG or tokens differ across arms/windows')
            if job['mode'] == 'smoke':
                smoke.add(job['arm'])
                write(a.out/'preflight.json',dict(completed=smoke == set(ARMS),arms=sorted(smoke)))
            else:
                current_regions = region_identity(rows)
                if expected_regions is None: expected_regions=current_regions
                require(current_regions == expected_regions,'Layer input, upstream gradient or weights differ across arms/windows')
                samples.append(dict(job,regions=summarize_window(rows)))
                write(a.out/'samples.json',samples)
        check_files(sources)
        result = report(samples,a.blocks)
        result.update(run_id=plan['run_id'],preflight=read(a.out/'preflight.json'),
                      worker_records={str(p.relative_to(a.out)):sha(p) for j in jobs
                                      for p in (a.out/'results'/j['directory']).glob('rank*.json')})
        write(a.out/'report.json',result)
        (a.out/'report.md').write_text(markdown(result))
        state.update(status='completed',stage='done',report=str(a.out/'report.md'))
        write(state_path,state)
        print('COMPLETED',a.out/'report.md',flush=True)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(state_path,state)
        raise


def main(argv=None):
    a = arguments(argv)
    print('Batch:',a.out,flush=True)
    if not a.execute:
        print(json.dumps(dict(preview=True,arms=ARMS,model=MODEL,blocks=a.blocks,
            regions='all 32 Attention + all 32 MLP; fwd/bwd/total',warmup=a.warmup,iterations=a.iterations,
            source_package=str(a.package) if a.package else 'export frozen sources',
            build=str(a.build_root) if a.build_root else 'compile exact native libraries',
            workflow='build -> both original full-model smoke checks -> balanced isolated layer windows -> report'),indent=2))
        return
    require(a.python.is_file() and (a.cuda_home/'bin/nvcc').is_file(),'Python or CUDA compiler missing')
    # Preserve environment selection even when --python overrides the launcher.
    if a.python.absolute() != Path(sys.executable).absolute():
        os.execv(str(a.python),[str(a.python),str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)])
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'handoff').mkdir()
    write(a.out/'batch-state.json',dict(protocol=PROTOCOL,status='launching'))
    write(a.out/'handoff/controllers.json',frozen_sources())
    if a.foreground: return execute(a)
    from detach_ablation5_run_two import detach
    print(json.dumps(detach(a,supervisor=execute),ensure_ascii=False),flush=True)
    print('Started; inspect batch-state.json and handoff/container-supervisor.log.',flush=True)


if __name__ == '__main__': main()
