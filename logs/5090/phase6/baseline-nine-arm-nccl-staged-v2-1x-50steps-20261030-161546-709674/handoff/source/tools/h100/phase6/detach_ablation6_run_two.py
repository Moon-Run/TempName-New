#!/usr/bin/env python3
"""Build the GemmOnly baseline once, then run two fresh 50-step comparisons.

Default: read-only preview. --execute detaches under container PID 1.
Existing A/B/C builds are reused; each comparison performs a fresh preflight.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import ablation4_repeat
import ablation6_protocol as routing
import detach_ablation5_run_two as launcher
from common import REPO, read, require, sha, write

ENTRY=Path(__file__).with_name('ablation6.py')
TRIALS=('compare1','compare2')


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ablation4_repeat.DEFAULT_BUILD,help='Existing ablation4 A/B/C build root')
    p.add_argument('--flux-build',type=Path,help='Reuse an already compiled ablation6 GemmOnly build; otherwise build inside this new batch')
    p.add_argument('--out',type=Path)
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    p.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--build-cuda-home',type=Path,default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--nccl-root',type=Path,help='NCCL include/nccl.h and lib/libnccl_static.a; default reuse recorded ablation4 development files')
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--execute',action='store_true');p.add_argument('--foreground',action='store_true')
    a=p.parse_args(argv)
    if a.out is None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        a.out=REPO/'logs/h100/phase6'/('ablation6-baselines-50steps-'+stamp)
    require(a.jobs>0,'jobs must be positive')
    for key in ('build','flux_build','out','python','megatron','cuda_home','build_cuda_home','nccl_root'):
        value=getattr(a,key)
        if value is not None:setattr(a,key,value.expanduser().resolve())
    return a


def commands(args):
    cli=[str(args.python),str(ENTRY)]; flux=args.flux_build or args.out/'flux-build'; steps=[]
    if args.flux_build is None:
        steps += [('build-prepare',cli+['build-prepare',str(flux)]),
                  ('compile',cli+['compile',str(flux),'--cuda-home',str(args.build_cuda_home),'--jobs',str(args.jobs)]+
                   (['--nccl-root',str(args.nccl_root)] if args.nccl_root else []))]
    for trial in TRIALS:
        root=args.out/trial
        steps += [(trial+'-prepare',cli+['prepare',str(root),'--build',str(args.build),'--flux-build',str(flux),
                    '--megatron',str(args.megatron),'--blocks','5','--rounds','2','--nvls-enable','0']),
                  (trial+'-dry-run',cli+['run',str(root),'--python',str(args.python),'--cuda-home',str(args.cuda_home)])]
    for trial in TRIALS:
        root=args.out/trial
        steps += [(trial+'-controller',cli+['run',str(root),'--stage','all','--python',str(args.python),
                    '--cuda-home',str(args.cuda_home),'--execute']), (trial+'-report',cli+['report',str(root)])]
    return steps


def check_inputs(root):
    info=read(root/'submission.json')
    require(info['protocol']==routing.PROTOCOL and info['execution_timing']==routing.TIMING and
            info['executable'] and not info['source_only'] and info['rounds']==2 and info['blocks_per_round']==5 and
            info['timing_windows']==50 and info['schedule']==routing.schedule(5,rounds=2) and
            info['communication']['environment'].get('NCCL_NVLS_ENABLE')=='0','Expected a fresh 50-step ablation6 comparison')
    return info


def check_report(root, info):
    result=read(root/'report.json');gate=read(root/'preflight.json')
    require(read(root/'all-state.json')['status']=='completed' and result['completed'] and result['correctness_passed'] and
        result['protocol']==routing.PROTOCOL and result['run_id']==info['run_id'] and result['execution_timing']==routing.TIMING and
        result['formal_optimizer_steps']==2500 and result['timing_windows']==50 and len(result['all_samples'])==50 and
        result['round_count']==2 and result['blocks_per_round']==5 and result['timing_rank_records']==400 and
        result['preflight_rank_records']==80 and result['profile_traces']==40 and result['candidate_shape_mappings']==6,
        'Incomplete 50-step report; next comparison will not start')
    require(gate['completed'] and gate['submission_sha256']==sha(root/'submission.json') and
        result['evidence']['preflight_sha256']==sha(root/'preflight.json') and
        result['communication']==gate['communication']==info['communication'] and
        result['naive_codec_identity']==gate['naive_codec_identity']==info['naive_codec_identity'],
        'Report/preflight/naive-codec identity mismatch')
    return result


def supervise(args):
    path=args.out/'batch-state.json'
    state=dict(status='preparing',execution_timing=routing.TIMING,expected_timing_windows=100,
               expected_formal_optimizer_steps=5000,blocks_per_round=5,rounds=2,nvls_enable='0',comparisons={})
    inputs={};reports=[]; env=dict(os.environ,PYTHONNOUSERSITE='1',NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7')
    try:
        if args.flux_build is None and args.nccl_root is None:
            recorded=read(args.build/'fused/manifest.json').get('nccl_files',{})
            headers=[Path(p) for p in recorded if p.endswith('/include/nccl.h')]
            require(len(headers)==1,'Pass --nccl-root: no unique NCCL development directory in original build')
            args.nccl_root=headers[0].parent.parent
            require(all(Path(p).is_file() and sha(p)==h for p,h in recorded.items()),'Recorded NCCL development files changed')
        for label,cmd in commands(args):
            state.update(stage=label,status='running' if 'controller' in label or 'report' in label else 'preparing')
            write(path,state);launcher.invoke(cmd,args.out/(label+'.log'),env)
            if label.endswith('-dry-run'):
                trial=label.removesuffix('-dry-run');info=check_inputs(args.out/trial);inputs[trial]=info
                if trial=='compare2':
                    ablation4_repeat.same_inputs(inputs['compare1'],info)
                    require(inputs['compare1']['flux_build_manifest']==info['flux_build_manifest'] and
                            inputs['compare1']['naive_codec_identity']==info['naive_codec_identity'],'Repeat baseline build/codec changed')
                state['comparisons'][trial]=dict(run_id=info['run_id'],status='prepared')
            if label.endswith('-report'):
                trial=label.removesuffix('-report');result=check_report(args.out/trial,inputs[trial])
                if reports:require(result['measurement_identity']==reports[0]['measurement_identity'],'Repeat runtime changed')
                reports.append(result)
                state['comparisons'][trial].update(status='completed',timing_windows=50,formal_optimizer_steps=2500,
                                                   report=str(args.out/trial/'report.json'))
            write(path,state)
        require(len(reports)==2,'Both strict reports required')
        state.update(status='completed',stage='done',timing_windows=100,timing_rank_records=800,formal_optimizer_steps=5000)
        write(path,state);print('Completed: two independent 50-step reports, 100 windows / 5000 formal steps.',flush=True)
        return state
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');write(path,state);raise


def main(argv=None):
    args=arguments(argv);print('Batch:',args.out,flush=True)
    print('A/B/C + Flux GemmOnly/Tensor TACO + Megatron/Tensor TACO; 10 warmup + 50 timed steps/window.',flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, fork, compilation, subprocesses or GPU queries.')
        for _,cmd in commands(args):print(shlex.join(cmd))
        print('Two independent comparisons: --blocks 5 --rounds 2 --nvls-enable 0; 100 formal windows / 5000 formal steps.')
        return
    require(args.python.is_file() and (args.build/'ablation4-manifest.json').is_file(),'Training Python / existing A/B/C build missing')
    if args.flux_build is None:require((args.build_cuda_home/'bin/nvcc').is_file(),'Build CUDA Toolkit missing')
    else:require((args.flux_build/'manifest.json').is_file(),'GemmOnly build manifest missing')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'handoff').mkdir()
    write(args.out/'batch-state.json',dict(status='launching',execution_timing=routing.TIMING,expected_timing_windows=100))
    write(args.out/'handoff/launch.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        if args.foreground:supervise(args)
        else:
            print(json.dumps(launcher.detach(args,supervisor=supervise),ensure_ascii=False),flush=True)
            print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.',flush=True)
    except Exception as error:
        state=read(args.out/'batch-state.json');state.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json',state);raise


if __name__=='__main__':main()
