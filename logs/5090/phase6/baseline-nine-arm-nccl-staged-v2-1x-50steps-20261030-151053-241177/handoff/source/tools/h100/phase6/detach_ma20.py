#!/usr/bin/env python3
"""Build MA20 bank, calibrate/hold out, then compare a frozen candidate to MA11."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import detach_ablation5_run_two as launcher
from common import HERE, REPO, read, require, sha, write

PROTOCOL='h100-ma20-codec-first-detached-v1'


def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path);p.add_argument('--trials',type=int,choices=(1,2),default=1)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    p.add_argument('--build-cuda-home',type=Path,default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--nccl-root',type=Path);p.add_argument('--ma11-build',type=Path)
    p.add_argument('--ma20-build',type=Path,help='Reuse a verified complete seven-arm MA20 bank')
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--worker-timeout',type=int,default=1800)
    p.add_argument('--execute',action='store_true');p.add_argument('--foreground',action='store_true')
    args=p.parse_args(argv)
    require(args.jobs>0 and args.worker_timeout>0,'Positive jobs and worker timeout required')
    require(not (args.ma11_build and args.ma20_build),'Specify only one reused build')
    if args.out is None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out=REPO/'logs/h100/phase6'/f'ma20-codec-first-{args.trials}x-50steps-{stamp}'
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.expanduser().absolute() if name=='python' else value.expanduser().resolve())
    return args


def commands(args,candidate=None):
    cli=lambda n:[str(args.python),str(HERE/n)]
    bank=args.ma20_build or args.out/'build'
    tasks=[]
    if candidate is None:
        if args.ma20_build is None:
            options=[] if args.ma11_build is None else ['--ma11-build',str(args.ma11_build)]
            tasks.append(('build-prepare',cli('ma20_build.py')+['prepare','--out',str(bank),*options]))
            flags=['--cuda-home',str(args.build_cuda_home),'--jobs',str(args.jobs)]
            if args.nccl_root:flags+=['--nccl-root',str(args.nccl_root)]
            tasks.append(('compile',cli('ma20_build.py')+['build','--build',str(bank),*flags]))
        tasks.append(('operator-prepare',cli('ma20_probe.py')+['prepare',str(args.out/'operator'),'--build',str(bank)]))
        tasks.append(('operator-calibrate',cli('ma20_probe.py')+['run',str(args.out/'operator'),
            '--python',str(args.python),'--cuda-home',str(args.cuda_home),
            '--worker-timeout',str(args.worker_timeout),'--execute']))
    else:
        tasks.append(('fullstep-prepare',cli('ma20_experiment.py')+['prepare',str(args.out/'fullstep'),
            '--build',str(bank),'--selection',str(args.out/'operator/selection.json'),
            '--candidate',candidate,'--megatron',str(args.megatron),'--trials',str(args.trials)]))
        for stage in ('preflight','profile','timing'):
            tasks.append(('fullstep-'+stage,cli('ma20_experiment.py')+['run',str(args.out/'fullstep'),
                '--stage',stage,'--python',str(args.python),'--cuda-home',str(args.cuda_home),
                '--worker-timeout',str(args.worker_timeout),'--execute']))
    return tasks


def controller_hashes():
    import ma20_experiment
    import ma20_probe
    result=ma20_experiment.controller_inputs()
    for name in ('ma20_probe.py','detach_ma20.py','run_ma20.sh'):
        result[str(HERE/name)]=sha(HERE/name)
    return result


def supervise(args):
    hashes=read(args.out/'handoff/controllers.json')
    env=dict(os.environ,PYTHONNOUSERSITE='1',NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7')
    state=dict(protocol=PROTOCOL,status='preparing',stage='build',candidate=None,
               fullstep_executed=False,increment_over_MA11_validated=False)
    def invoke(label,command):
        for path,digest in hashes.items():require(sha(path)==digest,'Controller changed: '+path)
        state.update(status='running',stage=label);write(args.out/'batch-state.json',state)
        launcher.invoke(command,args.out/(label+'.log'),env)
        for path,digest in hashes.items():require(sha(path)==digest,'Controller changed: '+path)
    try:
        for label,command in commands(args):invoke(label,command)
        decision=read(args.out/'operator/selection.json')
        import ma20_probe
        require(ma20_probe.verify_selection(args.out/'operator') == decision,
                'Saved operator selection differs from raw calibration/heldout evidence')
        require(decision.get('completed') is True,'Operator calibration/heldout did not complete')
        candidate=decision.get('candidate')
        state.update(candidate=candidate,operator_selection=str(args.out/'operator/selection.json'))
        if candidate is None:
            state.update(status='completed',stage='no_candidate',outcome='no_heldout_positive_candidate')
            write(args.out/'batch-state.json',state)
            print(json.dumps(state,ensure_ascii=False),flush=True)
            return state
        import ma20_build
        require(candidate in ma20_build.VARIANTS and candidate!='MA11','Invalid calibrated candidate')
        for label,command in commands(args,candidate):invoke(label,command)
        import ma20_experiment
        result=ma20_experiment.report(args.out/'fullstep',save=False)
        require(result==read(args.out/'fullstep/report.json'),'Fullstep report differs from raw evidence')
        require(result['formal_optimizer_steps']==args.trials*800 and
                result['timing_rank_records']==args.trials*128,'Incomplete fullstep measurement')
        state.update(status='completed',stage='done',fullstep_executed=True,
                     increment_over_MA11_validated=result['increment_over_MA11_validated'],
                     performance_increment_over_MA11_validated=result['performance_increment_over_MA11_validated'],
                     accuracy_equivalence_validated=False,
                     report=str(args.out/'fullstep/report.json'),
                     formal_optimizer_steps=result['formal_optimizer_steps'])
        write(args.out/'batch-state.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
        return state
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json',state)
        raise


def main(argv=None):
    args=arguments(argv)
    print('Batch:',args.out,flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, GPU queries, builds or workers.')
        for _,command in commands(args):print(shlex.join(command))
        print('Only a candidate selected by training and independently confirmed by heldout proceeds:')
        for _,command in commands(args,'SELECTED_Q'):print(shlex.join(command))
        print(f'{args.trials} independent fullstep trial(s), 8 complete paired blocks/trial, '
              '10 warmup + 50 measured optimizer steps/window. No candidate means stop with no_candidate; '
              'all negative fullstep windows are retained. Changed masks require independent codec checks; '
              'cross-mask bitwise equivalence is not assumed.')
        return
    require(args.python.is_file() and args.megatron.is_dir(),'Python/Megatron missing')
    require((args.cuda_home/'lib64').is_dir(),'Runtime CUDA missing')
    if args.ma20_build:
        import ma20_build
        ma20_build.verify(args.ma20_build,built=True)
    else:require((args.build_cuda_home/'bin/nvcc').is_file(),'Build CUDA missing')
    hashes=controller_hashes()
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'handoff').mkdir()
    write(args.out/'batch-state.json',dict(protocol=PROTOCOL,status='launching'))
    write(args.out/'handoff/controllers.json',hashes)
    write(args.out/'handoff/launch.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if args.foreground:return supervise(args)
    print(json.dumps(launcher.detach(args,supervisor=supervise),ensure_ascii=False),flush=True)
    print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.')


if __name__=='__main__':main()
