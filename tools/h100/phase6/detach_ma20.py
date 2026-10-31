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
    p.add_argument('--reuse-batch',type=Path,
                   help='Reuse a completed operator calibration and its build; skip build and operator jobs')
    p.add_argument('--candidate',choices=('Q0','Q7','Q14','Q28','Q56','Q112'),
                   help='Manually choose the fullstep arm with --reuse-batch; preserve the original selection')
    p.add_argument('--reuse-ma11',action='store_true',
                   help='With --reuse-batch and --candidate, run only the candidate and compare to saved MA11 timing')
    p.add_argument('--candidate-only',action='store_true',
                   help='Keep operator calibration, but run only the selected fullstep candidate without an MA11 comparison')
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--worker-timeout',type=int,default=1800)
    p.add_argument('--execute',action='store_true');p.add_argument('--foreground',action='store_true')
    args=p.parse_args(argv)
    require(args.jobs>0 and args.worker_timeout>0,'Positive jobs and worker timeout required')
    require(not (args.ma11_build and args.ma20_build),'Specify only one reused build')
    require(not args.candidate or args.reuse_batch,'--candidate requires --reuse-batch')
    require(not args.reuse_ma11 or (args.reuse_batch and args.candidate),
            '--reuse-ma11 requires --reuse-batch and --candidate')
    require(not (args.reuse_ma11 and args.candidate_only),
            '--candidate-only does not borrow historical MA11 timing')
    require(not (args.reuse_batch and args.ma11_build),'--reuse-batch cannot use --ma11-build')
    if args.out is None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out=REPO/'logs/h100/phase6'/f'ma20-codec-first-{args.trials}x-50steps-{stamp}'
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.expanduser().absolute() if name=='python' else value.expanduser().resolve())
    if args.reuse_batch:
        require(not args.out.is_relative_to(args.reuse_batch),'Use a new output directory outside the reused batch')
        if args.ma20_build is None:
            args.ma20_build=args.reuse_batch/'build'
    return args


def operator_selection(args):
    return (args.reuse_batch or args.out)/'operator/selection.json'


def commands(args,candidate=None):
    cli=lambda n:[str(args.python),str(HERE/n)]
    bank=args.ma20_build or args.out/'build'
    tasks=[]
    if candidate is None:
        if args.reuse_batch:
            return tasks
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
        flags=['--manual-candidate'] if args.candidate else []
        if args.reuse_ma11:flags+=['--reuse-ma11-from',str(args.reuse_batch/'fullstep')]
        if getattr(args,'candidate_only',False):flags+=['--candidate-only']
        tasks.append(('fullstep-prepare',cli('ma20_experiment.py')+['prepare',str(args.out/'fullstep'),
            '--build',str(bank),'--selection',str(operator_selection(args)),
            '--candidate',candidate,'--megatron',str(args.megatron),'--trials',str(args.trials),*flags]))
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
    if getattr(args,'candidate_only',False):
        state.update(comparison_mode='candidate_only',ma11_fullstep_executed=False)
    if args.reuse_batch:
        state.update(stage='reuse-operator',reused_batch=str(args.reuse_batch),
                     build_reused=True,operator_jobs_executed=False)
    def invoke(label,command):
        for path,digest in hashes.items():require(sha(path)==digest,'Controller changed: '+path)
        state.update(status='running',stage=label);write(args.out/'batch-state.json',state)
        launcher.invoke(command,args.out/(label+'.log'),env)
        for path,digest in hashes.items():require(sha(path)==digest,'Controller changed: '+path)
    try:
        for label,command in commands(args):invoke(label,command)
        selection=operator_selection(args)
        decision=read(selection)
        import ma20_probe
        require(ma20_probe.verify_selection(selection.parent) == decision,
                'Saved operator selection differs from raw calibration/heldout evidence')
        require(decision.get('completed') is True,'Operator calibration/heldout did not complete')
        candidate=args.candidate or decision.get('candidate')
        state.update(candidate=candidate,operator_selection=str(selection))
        if candidate is None:
            state.update(status='completed',stage='no_candidate',outcome='no_heldout_positive_candidate')
            write(args.out/'batch-state.json',state)
            print(json.dumps(state,ensure_ascii=False),flush=True)
            return state
        import ma20_build
        require(candidate in ma20_build.VARIANTS and candidate!='MA11','Invalid calibrated candidate')
        import ma20_experiment
        state['candidate_selection']=ma20_experiment.candidate_selection(
            decision,candidate,manual=bool(args.candidate))
        for label,command in commands(args,candidate):invoke(label,command)
        result=ma20_experiment.report(args.out/'fullstep',save=False)
        require(result==read(args.out/'fullstep/report.json'),'Fullstep report differs from raw evidence')
        arms=1 if args.reuse_ma11 or getattr(args,'candidate_only',False) else 2
        require(result['formal_optimizer_steps']==args.trials*400*arms and
                result['timing_rank_records']==args.trials*64*arms,'Incomplete fullstep measurement')
        state.update(status='completed',stage='done',fullstep_executed=True,
                     increment_over_MA11_validated=result['increment_over_MA11_validated'],
                     performance_increment_over_MA11_validated=result['performance_increment_over_MA11_validated'],
                     accuracy_equivalence_validated=False,
                     comparison_mode=result.get('comparison_mode','paired'),
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
        if args.reuse_batch:
            print('Reuse verified build and operator evidence:',args.reuse_batch)
            print('No compilation or operator jobs.')
        if args.reuse_ma11:
            print('Run only',args.candidate,'; reuse saved MA11 timing for a descriptive cross-batch comparison.')
        if args.candidate_only:
            print('Keep MA20 operator calibration/heldout; run only the selected fullstep candidate. '
                  'No MA11 fullstep, historical timing, paired CI or speedup claim.')
        if args.candidate:
            print('Manual fullstep candidate:',args.candidate,'; original operator selection is retained.')
        else:
            print('Only a candidate selected by training and independently confirmed by heldout proceeds:')
        for _,command in commands(args,args.candidate or 'SELECTED_Q'):print(shlex.join(command))
        windows='8 candidate-only windows/trial' if args.reuse_ma11 or args.candidate_only else '8 complete paired blocks/trial'
        print(f'{args.trials} independent fullstep trial(s), {windows}, '
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
