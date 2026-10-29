#!/usr/bin/env python3
"""Portable two-container TP8/DP2 campaign. Preview unless --execute is given."""
import argparse
from datetime import timedelta
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from tp8_dp2_protocol import (ARMS, MODEL, POLICIES, PROTOCOL, build_package_id, digest, jobs, read, report,
                             require, sha, validate_rows, verify_package, write)


def parser(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--node-rank',type=int,choices=(0,1),required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--master-addr',default='10.18.133.74')
    p.add_argument('--control-port',type=int,default=29580)
    p.add_argument('--master-port',type=int,default=29581)
    p.add_argument('--socket-ifname',default='bond0')
    p.add_argument('--hca',default='mlx5_1:1,mlx5_4:1,mlx5_6:1,mlx5_8:1')
    p.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--run-id',default='tp8-dp2-20261028')
    p.add_argument('--out',type=Path)
    p.add_argument('--build-root',type=Path)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--trials',type=int,choices=(1,2),default=1)
    p.add_argument('--blocks',type=int,default=12)
    p.add_argument('--coordination-timeout',type=int,default=7200)
    p.add_argument('--worker-timeout',type=int,default=1800)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    a.package=a.package.expanduser().resolve()
    a.python=a.python.expanduser().absolute()  # Preserve the selected environment's executable path.
    a.cuda_home=a.cuda_home.expanduser().resolve()
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,70}',a.run_id) and a.run_id not in ('.','..'), 'Invalid run ID')
    require(1<=a.control_port<=65535 and 1<=a.master_port<=65535 and a.control_port!=a.master_port,'Use two distinct valid ports')
    require(a.jobs>0 and a.coordination_timeout>0 and a.worker_timeout>300,'Invalid jobs/timeouts')
    jobs(a.trials,a.blocks)
    return a


class Coordinator:
    def __init__(self,args):
        import torch.distributed as dist
        self.rank=args.node_rank
        self.timeout=args.coordination_timeout
        # No GPUs or distributed training group in the controller.
        self.store=dist.TCPStore(args.master_addr,args.control_port,world_size=2,
            is_master=self.rank==0,timeout=timedelta(seconds=self.timeout),wait_for_workers=True)

    def check(self):
        if self.store.check(['abort']):
            raise RuntimeError('Peer/controller aborted: '+self.store.get('abort').decode())

    def fail(self,error):
        try: self.store.set('abort',f'node{self.rank}: {type(error).__name__}: {error}')
        except Exception: pass

    def exchange(self,key,value):
        self.store.set(f'{key}/{self.rank}',json.dumps(value,allow_nan=False))
        keys=[f'{key}/{r}' for r in range(2)]
        deadline=time.monotonic()+self.timeout
        while not self.store.check(keys):
            self.check()
            require(time.monotonic()<deadline,'Timed out waiting for both nodes: '+key)
            time.sleep(.5)
        self.check()
        return [json.loads(self.store.get(k)) for k in keys]

    def finish(self):
        # Rank 1 performs no store read after this acknowledgement. A symmetric
        # last barrier can race TCPStore destruction on rank 0.
        if self.rank == 1:
            self.store.set('node1-final-state-written','done')
            return
        deadline=time.monotonic()+self.timeout
        while not self.store.check(['node1-final-state-written']):
            self.check()
            require(time.monotonic()<deadline,'Peer did not write its final state')
            time.sleep(.5)


def run_process(command,cwd,env,log,timeout,coordinator):
    with Path(log).open('x') as handle:
        handle.write('COMMAND '+repr(list(map(str,command)))+'\n');handle.flush()
        process=subprocess.Popen(list(map(str,command)),cwd=cwd,env=env,stdout=handle,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+timeout
        try:
            while process.poll() is None:
                coordinator.check()
                require(time.monotonic()<deadline,'Child process timed out; see '+str(log))
                time.sleep(.5)
            require(process.returncode==0,f'Child exited {process.returncode}; see {log}')
        except BaseException:
            # Only the process group created by this launcher is terminated.
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
            raise


def worker_environment(args,scripts,build_root,job,dest,comm):
    import communication
    arm=job['arm']
    native=build_root/arm
    env={k:v for k,v in os.environ.items() if not k.startswith(('NCCL_','TORCH_NCCL_','H100_','E2E_','FLUX_'))}
    for key in ('PYTHONOPTIMIZE','MECH_SMOKE'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONUNBUFFERED='1',CUDA_DEVICE_MAX_CONNECTIONS='1',FLUX_GEMM_RS_GEMM_ONLY='1',
        H100_TP8_DP2_EXECUTE='1',H100_TP='8',H100_SM_COUNT='132',H100_RUN_ID=args.run_id,
        E2E_POLICY=POLICIES[arm],E2E_MODE=job['mode'],E2E_CASE='h100-phase6',
        E2E_BUILD=str(scripts/'build'),E2E_PROCESS_OUT=str(dest),E2E_BLOCK=str(job['block']),E2E_WINDOW=str(job['index']),
        H100_MA4_F_ARM='F4',H100_MA11_ARM='F4-cache-zcount-csoftmax',H100_MA11_PROTOCOL='h100-ma11-causal-softmax-v1',
        CUDA_HOME=str(args.cuda_home),PATH=f'{args.python.parent}:{args.cuda_home}/bin:'+env.get('PATH',''))
    py=[scripts,build_root/'megatron']
    ld=[args.cuda_home/'lib64']
    if arm!='megatron-cuda':
        py.append(native/'python')
        ld=[native/'python/flux/lib',native,native/'build/lib']+ld
    env['PYTHONPATH']=':'.join(map(str,py))
    env['LD_LIBRARY_PATH']=':'.join(map(str,ld))+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return communication.apply(env,comm)


def torchrun(args,worker):
    return [args.python,'-m','torch.distributed.run','--nnodes=2','--nproc-per-node=8',
            f'--node-rank={args.node_rank}',f'--master-addr={args.master_addr}',f'--master-port={args.master_port}',
            '--rdzv-backend=static','--rdzv-conf=timeout=300','--max-restarts=0',worker,
            *read(args.package/'app/model-arguments.json')]


def prepare_workers(args,out,build_root,package_id,comm):
    workers={}
    for arm in ARMS:
        scripts=out/'workers'/arm
        shutil.copytree(args.package/'workers'/arm,scripts)
        cfg=read(scripts/'config.json')
        cfg.update(package_id=package_id,communication=comm)
        if arm!='ma11':
            lib=build_root/('codec-basic/libbasic_quant.so' if arm=='flux-basic' else 'codec-cuda/libbaseline_taco.so')
            cfg['codec_library']=dict(path=str(lib),sha256=sha(lib))
        write(scripts/'config.json',cfg)
        (scripts/'build').mkdir()
        if arm!='megatron-cuda':
            (scripts/'build'/POLICIES[arm]).symlink_to(build_root/arm,target_is_directory=True)
        workers[arm]=dict(root=scripts,files={p.name:sha(p) for p in scripts.iterdir() if p.is_file()})
    return workers


def fingerprint_rows(rows):
    return [{key:r[key] for key in ('rank','gpu_uuid','torch','cuda','nccl','python','hardware','initial_parameters_sha256',
                                   'initial_rng_sha256','tokens_sha256','global_tokens_sha256','sample_ids',
                                   'parameter_count_local','padded_vocab_size','model','communication')} for r in rows]


def execute(args,info):
    import torch
    import communication
    import tp8_dp2_build
    import check_tp8_dp2_link as link
    import basic_quant_validation
    import baseline_codec_validation
    from tp8_dp2_profile import analyze
    require(args.python.is_file() and (args.cuda_home/'bin/nvcc').is_file(),'Python/CUDA compiler not found')
    require(torch.__version__=='2.6.0+cu124' and torch.version.cuda=='12.4' and
            tuple(torch.cuda.nccl.version())==(2,21,5),'Use the probed Torch 2.6.0+cu124 / NCCL 2.21.5 environment')
    out=(args.out or args.package/'runs'/args.run_id/f'node{args.node_rank}').resolve()
    args.out=out
    require(not out.exists(),'Use a new run ID/output; partial runs are never overwritten or mixed')
    out.mkdir(parents=True)
    build_id=build_package_id(args.package,info)
    build_root=(args.build_root or args.package/'build'/f'node{args.node_rank}-{build_id[:12]}').resolve()
    state=dict(status='starting',protocol=PROTOCOL,run_id=args.run_id,package_id=info['package_id'],node_rank=args.node_rank)
    write(out/'state.json',state)
    coordinator=None
    try:
        # This does not reserve GPUs; check again immediately before each job.
        write(out/'local-preflight.json',link.local_requirements(args))
        coordinator=Coordinator(args)
        comm_env=link.settings(args)
        for key in ('NCCL_DEBUG_SUBSYS','NCCL_DEBUG_FILE'):
            comm_env.pop(key,None)
        os.environ.update(comm_env)
        comm=communication.capture('0')
        agreement=dict(protocol=PROTOCOL,package_id=info['package_id'],build_package_id=build_id,model=MODEL,run_id=args.run_id,
                       trials=args.trials,blocks=args.blocks,worker_port=args.master_port,communication=comm['environment'],
                       nccl_config_hashes=sorted(v or 'absent' for v in comm['config_files'].values()))
        hello=coordinator.exchange('hello',agreement)
        require(hello[0]==hello[1],'The two nodes selected different packages, run IDs, workload, or NCCL settings')
        write(out/'agreement.json',agreement)
        print('LINK PREFLIGHT',flush=True)
        link_command=[args.python,args.package/'tools/h100/phase6/check_tp8_dp2_link.py','--execute',
            '--python',args.python,'--node-rank',args.node_rank,'--master-addr',args.master_addr,
            '--master-port',args.master_port,'--socket-ifname',args.socket_ifname,'--hca',args.hca,
            '--run-id',args.run_id+'-link','--out',out/'link']
        coordinator.exchange('link-ready',True)
        run_process(link_command,args.package,dict(os.environ),out/'link-launch.log',480,coordinator)
        linked=coordinator.exchange('link-finished',read(out/'link/state.json'))
        require(all(x['status']=='completed' for x in linked),'Both nodes must pass IB/TP8/DP2 communication tests')
        print('VERIFY cached native libraries' if build_root.exists() else 'BUILD native libraries (first run only)',
              build_root,flush=True)
        built=tp8_dp2_build.build(args.package,build_root,args.cuda_home,args.jobs,build_id)
        coordinator.exchange('build-finished',dict(files=built['files'],environment=built['environment']))
        workers=prepare_workers(args,out,build_root,info['package_id'],comm)
        write(out/'plan.json',dict(agreement=agreement,jobs=jobs(args.trials,args.blocks),build_root=str(build_root),
              build_sha256=sha(build_root/'build.json'),worker_files={a:v['files'] for a,v in workers.items()}))
        state['status']='running';write(out/'state.json',state)
        samples=[];reference=None;profile_summary={};smoke_summary={}
        for number,job in enumerate(jobs(args.trials,args.blocks)):
            coordinator.check()
            if job['mode']=='timing':
                require(set(smoke_summary)==set(profile_summary)==set(ARMS) and
                        all(set(profile_summary[arm])=={str(r) for r in range(16)} for arm in ARMS),
                        'All three smoke and 16-rank profile checks must pass before timing')
            # Exact package manifest scope, never a repository-wide scan.
            verify_package(args.package)
            tp8_dp2_build.verify(build_root,build_id)
            scripts=workers[job['arm']]['root']
            for name,value in workers[job['arm']]['files'].items():
                require(sha(scripts/name)==value,'Frozen worker changed: '+name)
            dest=out/'results'/job['directory'];dest.mkdir(parents=True)
            write(dest/'idle-check.json',link.local_requirements(args))
            coordinator.exchange(f'job{number}-ready',job)
            env=worker_environment(args,scripts,build_root,job,dest,comm)
            command=torchrun(args,scripts/'worker.py')
            write(dest/'command.json',dict(command=list(map(str,command)),job=job,communication=comm))
            print('START',job['directory'],flush=True)
            run_process(command,build_root/'megatron',env,dest/'process.log',args.worker_timeout,coordinator)
            local_rows=[read(dest/f'rank{rank}.json') for rank in range(args.node_rank*8,args.node_rank*8+8)]
            require(all(r['worker_config_sha256']==workers[job['arm']]['files']['config.json'] for r in local_rows),
                    'Worker loaded a different configuration')
            arm=job['arm']
            if arm!='megatron-cuda':
                for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                    path=build_root/arm/'python/flux/lib'/name
                    expected=dict(path=str(path.resolve()),sha256=built['files'][str(path.relative_to(build_root))])
                    require(all(row[name]==expected for row in local_rows),'Worker loaded a different native library: '+name)
            if arm!='ma11':
                expected=read(scripts/'config.json')['codec_library']
                require(all(row['codec_library']==expected for row in local_rows),'Worker codec library differs from its frozen build')
                if job['mode']=='smoke':
                    for row in local_rows:
                        if arm=='flux-basic':
                            basic_quant_validation.verify(row['codec_validation'],expected,row['tp_rank'])
                        else:
                            baseline_codec_validation.verify(row['codec_validation'],expected)
            node_rows=coordinator.exchange(f'job{number}-rows',local_rows)
            rows=validate_rows(node_rows[0]+node_rows[1],job,info['package_id'])
            require(all(r['run_id']==args.run_id for r in rows),'Worker run ID differs')
            current=fingerprint_rows(rows)
            if reference is None: reference=current
            require(current==reference,'Cross-arm/window initial model, input, software or hardware identity changed')
            write(dest/'all-ranks.json',rows)
            if job['mode']=='timing':
                sample=dict(job,ms_per_step=max(r['elapsed_seconds'] for r in rows)*1000/50,
                            max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows))
                samples.append(sample);write(out/'timing-progress.json',dict(completed=False,samples=samples))
            elif job['mode']=='smoke':
                smoke_summary[job['arm']]=dict(passed=True,dp_gradient_and_parameter_pairs=8,steps_checked=3)
            else:
                local_profiles={}
                for row in local_rows:
                    path=dest/f'rank{row["rank"]}-profile.json.gz'
                    with gzip.open(path,'rt') as handle: events=json.load(handle)['traceEvents']
                    local_profiles[str(row['rank'])]=dict(analyze(events,job['arm'],row['rank']),trace_sha256=sha(path))
                combined=coordinator.exchange(f'job{number}-profiles',local_profiles)
                profile_summary[job['arm']]=dict(combined[0],**combined[1])
                require(len(profile_summary[job['arm']])==16,'Incomplete per-rank profile coverage')
            coordinator.exchange(f'job{number}-validated',True)
            print('PASS',job['directory'],flush=True)
        require(set(smoke_summary)==set(profile_summary)==set(ARMS),'Incomplete correctness/profile gates')
        result=report(samples,args.trials,args.blocks,info['package_id'])
        result.update(run_id=args.run_id,smoke=smoke_summary,profiles=profile_summary,
                      initial_identity=reference,link_checks=linked)
        write(out/'report.json',result)
        coordinator.exchange('reports-written',sha(out/'report.json'))
        state['status']='completed';write(out/'state.json',state)
        # Keep node 0's TCPStore alive until node 1 has written its final state.
        coordinator.finish()
        print('COMPLETED',out/'report.json',flush=True)
        print(json.dumps(result['comparisons'],ensure_ascii=False,indent=2),flush=True)
    except BaseException as error:
        if coordinator is not None: coordinator.fail(error)
        state.update(status='failed',error=type(error).__name__+': '+str(error));write(out/'state.json',state)
        raise


def main(argv=None):
    args=parser(argv)
    info=verify_package(args.package)
    if not args.execute:
        print(json.dumps(dict(execute=False,package_id=info['package_id'],build_package_id=build_package_id(args.package,info),node_rank=args.node_rank,
            model=MODEL,arms=ARMS,control_port=args.control_port,worker_port=args.master_port,
            jobs=jobs(args.trials,args.blocks),communication_preflight=True,
            compile_on_first_run=True),ensure_ascii=False,indent=2))
        return
    if not args.inside:
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        env.pop('PYTHONOPTIMIZE',None)
        env.pop('PYTHONPATH',None)
        os.execve(str(args.python),[str(args.python),str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv),'--inside'],env)
    execute(args,info)


if __name__=='__main__':
    main()
