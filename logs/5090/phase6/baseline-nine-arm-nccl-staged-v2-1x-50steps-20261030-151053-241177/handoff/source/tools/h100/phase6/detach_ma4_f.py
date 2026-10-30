#!/usr/bin/env python3
"""Build and test MA4-F's fused consumption chain versus CR2, without CUDA graphs.

Preview is read-only. --execute runs one complete 8-block paired experiment by
default; --trials 2 repeats the comparison independently.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import detach_ablation5_run_two as launcher
from common import HERE, REPO, read, write, require, sha, verify_files

PROTOCOL='h100-ma4-f-detached-v1'
BLOCKS=8
STEPS=50


def arguments(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--trials',type=int,choices=(1,2),default=1)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    parser.add_argument('--build-cuda-home',type=Path,default=Path('/usr/local/cuda-12.8'))
    parser.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.4'))
    parser.add_argument('--nccl-root',type=Path)
    parser.add_argument('--ma4-f-build',type=Path,help='Reuse a verified F4/CR2 build')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--worker-timeout',type=int,default=1800)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--foreground',action='store_true')
    args=parser.parse_args(argv)
    require(args.jobs>0 and args.worker_timeout>0,'Positive jobs and worker timeout required')
    if args.out is None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out=REPO/'logs/h100/phase6'/f'ma4-f-{args.trials}x-50steps-{stamp}'
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().absolute() if key=='python' else value.expanduser().resolve())
    return args


def controller_hashes():
    import ma4_f_build
    import ma4_f_bench
    import ma4_f_fullstep
    paths={Path(m.__file__).resolve() for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
           and str(m.__file__).endswith('.py') and Path(m.__file__).resolve().is_relative_to(HERE)}
    paths.update(HERE/name for name in ma4_f_build.INPUTS)
    paths.update((Path(__file__).resolve(),HERE/'ma5_fullstep.py',HERE/'ma2_cr2_bench.py',HERE/'run_ma4_f.sh',REPO/'tools/a800/phase4/reference.py'))
    paths.update(p for p in (REPO/'tools/a800/phase5/e2e').iterdir() if p.suffix in ('.py','.json'))
    return {str(p):sha(p) for p in sorted(paths)}


def commands(args):
    cli=lambda name:[str(args.python),str(HERE/name)]
    build=args.ma4_f_build or args.out/'build'
    tasks=[]
    if args.ma4_f_build is None:
        flags=['--cuda-home',str(args.build_cuda_home),'--jobs',str(args.jobs)]
        if args.nccl_root:flags+=['--nccl-root',str(args.nccl_root)]
        tasks+=[('build-prepare',cli('ma4_f_build.py')+['prepare','--out',str(build)]),
                ('compile',cli('ma4_f_build.py')+['build','--build',str(build),*flags])]
    runtime=['--python',str(args.python),'--cuda-home',str(args.cuda_home),'--worker-timeout',str(args.worker_timeout)]
    operator,train=args.out/'operator',args.out/'fullstep'
    tasks += [
        ('operator-prepare',cli('ma4_f_bench.py')+['prepare',str(operator),'--build',str(build),'--blocks',str(BLOCKS)]),
        ('operator-timing',cli('ma4_f_bench.py')+['run',str(operator),*runtime,'--execute']),
        ('operator-profile',cli('ma4_f_bench.py')+['run',str(operator),*runtime,'--profile','--execute']),
        ('fullstep-prepare',cli('ma4_f_fullstep.py')+['prepare',str(train),'--build',str(build),'--megatron',str(args.megatron),
                                                  '--trials',str(args.trials),'--blocks',str(BLOCKS)]),
        ('fullstep-run',cli('ma4_f_fullstep.py')+['run',str(train),'--stage','all',*runtime,'--execute']),
        ('fullstep-profile',cli('ma4_f_fullstep.py')+['run',str(train),'--stage','profile',*runtime,'--execute'])]
    return tasks


def check_operator(root):
    import ma4_f_bench as bench
    plan=bench.verify(root)
    result,profile=read(root/'report.json'),read(root/'profile-summary.json')
    require(result['protocol']==bench.PROTOCOL and result['run_id']==plan['run_id'] and result['completed'] and
            result['correctness_passed'] and result['bitwise_equal_within_pairs'] and result['consumer_gradients_validated'] and
            result['equivalence_pairs']==[['F4','CR2']], 'Complete consumer correctness and gradients required')
    require(result==bench.report(root,save=False),'Operator report differs from raw evidence')
    require(profile['protocol']==bench.PROTOCOL and profile['run_id']==plan['run_id'] and profile['completed'] and
            profile['execution_path_validated'] and profile==bench.profile_summary(root),'Consumer operator path not validated')
    return result


def check_fullstep(root,trials,profile=False):
    import ma4_f_fullstep as full
    plan=full.verify(root);result=read(root/'report.json')
    windows=trials*BLOCKS*2
    require(plan['trials']==trials and plan['blocks']==BLOCKS and result['protocol']==full.PROTOCOL and
            result['run_id']==plan['run_id'] and result['completed'] and result['correctness_passed'] and
            result['consumer_parameter_gradients_validated'] and result['identical_trajectories_within_pairs'] and
            result['formal_optimizer_steps']==windows*STEPS and result['timing_rank_records']==windows*8 and
            len(result['all_samples'])==windows and set(result['comparisons'])=={f'compare{i+1}' for i in range(trials)},
            'Incomplete requested F4/CR2 full-step experiment')
    require(result==full.report(root,save=False),'Training report differs from raw evidence')
    if profile:
        evidence=read(root/'profile-summary.json')
        require(evidence['protocol']==full.PROTOCOL and evidence['run_id']==plan['run_id'] and
                evidence['completed'] and evidence['execution_validated'] and evidence==full.profile_summary(root),
                'Full-model consumer chain evidence missing')
    return result


def supervise(args):
    hashes=read(args.out/'handoff/controllers.json')
    state=dict(protocol=PROTOCOL,status='preparing',trials=args.trials,arms=['F4','CR2'],main_contrasts=['F4/CR2'],
               target_evaluated=False,target_validated=False,increment_over_CR2_validated=False,
               expected_timing_windows=args.trials*BLOCKS*2,expected_formal_optimizer_steps=args.trials*BLOCKS*2*STEPS,
               reports={})
    state_path=args.out/'batch-state.json'
    env=dict(os.environ,PYTHONNOUSERSITE='1',NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7')
    try:
        for label,command in commands(args):
            verify_files('/',hashes);state.update(status='running',stage=label);write(state_path,state)
            launcher.invoke(command,args.out/(label+'.log'),env);verify_files('/',hashes)
            if label=='operator-profile':
                check_operator(args.out/'operator')
                state.update(operator_correctness_validated=True,operator_execution_path_validated=True)
                state['reports']['operator']=str(args.out/'operator/report.json')
            elif label=='fullstep-run':
                result=check_fullstep(args.out/'fullstep',args.trials)
                state.update(acceptance_status=result['acceptance_status'])
                state['reports']['fullstep']=str(args.out/'fullstep/report.json')
            elif label=='fullstep-profile':
                result=check_fullstep(args.out/'fullstep',args.trials,profile=True)
                state.update(fullstep_execution_path_validated=True,
                             increment_over_CR2_validated=result['increment_over_CR2_validated'],
                             formal_optimizer_steps=result['formal_optimizer_steps'],timing_windows=len(result['all_samples']))
                state['reports']['fullstep_profile']=str(args.out/'fullstep/profile-summary.json')
        state.update(status='completed',stage='done');write(state_path,state)
        print(json.dumps(state,ensure_ascii=False),flush=True);return state
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');write(state_path,state);raise


def main(argv=None):
    args=arguments(argv);print('Batch:',args.out,flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, GPU queries, builds or workers.')
        for _,command in commands(args):print(shlex.join(command))
        print(f'F4/CR2 only; {args.trials} trial(s), {args.trials*16} windows, {args.trials*800} formal optimizer steps. '
              'Each window: 10 warmup + 50 measured steps. Consumer correctness, every smoke parameter gradient/update '
              'and separate profiles are required. Negative timings are preserved. C/R 2% target is not evaluated.')
        return
    require(args.python.is_file() and args.megatron.is_dir(),'Python/Megatron missing')
    require((args.cuda_home/'lib64').is_dir(),'Runtime CUDA missing')
    if args.ma4_f_build is None:require((args.build_cuda_home/'bin/nvcc').is_file(),'Build CUDA missing')
    if args.nccl_root:require(all((args.nccl_root/name).is_file() for name in ('include/nccl.h','lib/libnccl_static.a')),'NCCL development files missing')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'handoff').mkdir()
    write(args.out/'batch-state.json',dict(protocol=PROTOCOL,status='launching',target_validated=False))
    write(args.out/'handoff/launch.json',{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()})
    write(args.out/'handoff/controllers.json',controller_hashes())
    if args.foreground:return supervise(args)
    print(json.dumps(launcher.detach(args,supervisor=supervise),ensure_ascii=False),flush=True)
    print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.')


if __name__=='__main__':main()
