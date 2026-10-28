#!/usr/bin/env python3
"""Run ablation5 twice, then ablation4 twice, with 60 timed steps per window.

Default: read-only preview. --execute double-forks inside the container.
Each comparison has its own preflight and strict report; any failure stops.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import ablation4_repeat as a4
import ablation_timing
import detach_ablation5_run_two as a5
from common import REPO, read, require, sha, write


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=a4.DEFAULT_BUILD,
                   help='Existing ablation4 build; ablation5 reuses its candidate/ subdirectory')
    p.add_argument('--out', type=Path, help='New parent batch directory; existing paths are rejected')
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true', help='Keep the supervisor attached for diagnostics')
    args = p.parse_args(argv)
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/('ablation5-then4-60steps-'+stamp)
    for key in ('build', 'out', 'python', 'megatron', 'cuda_home'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def commands(args):
    shared = ['--python', str(args.python), '--megatron', str(args.megatron),
              '--cuda-home', str(args.cuda_home), '--execute']
    return [
        ('ablation5', [str(args.python), str(Path(a5.__file__).resolve()),
                      '--candidate-build', str(args.build/'candidate'), '--out', str(args.out/'ablation5'),
                      '--foreground', *shared]),
        ('ablation4', [str(args.python), str(Path(a4.__file__).resolve()),
                      '--build', str(args.build), '--out', str(args.out/'ablation4'), *shared]),
    ]


def completed_batch(root, label):
    state = read(root/'batch-state.json')
    require(state['status'] == 'completed' and state['timing_windows'] == 100 and
            state['timing_rank_records'] == 800, 'Both comparisons must complete before the next ablation')
    ablation_timing.require_report(state, 100)
    inputs, reports = [], []
    for trial in ('compare1', 'compare2'):
        folder = root/trial
        info = (a5.check_inputs if label == 'ablation5' else a4.submission)(folder)
        result = (a5.check_report if label == 'ablation5' else a4.completed_report)(folder, info)
        inputs.append(info); reports.append(result)
    a4.same_inputs(*inputs)
    require(reports[0]['measurement_identity'] == reports[1]['measurement_identity'] and
            reports[0]['role_plan_sha256'] == reports[1]['role_plan_sha256'], 'Repeat runtime/masks changed')
    return {trial: dict(run_id=result['run_id'], report=str(root/trial/'report.json'),
                       report_sha256=sha(root/trial/'report.json'), timing_windows=50, formal_optimizer_steps=3000)
            for trial, result in zip(('compare1', 'compare2'), reports)}


def supervise(args):
    state_path = args.out/'batch-state.json'
    state = dict(status='preparing', execution_timing=ablation_timing.execution_timing(),
                 order=['ablation5/compare1', 'ablation5/compare2', 'ablation4/compare1', 'ablation4/compare2'],
                 blocks_per_round=5, rounds=2, nvls_enable='0', expected_timing_windows=200,
                 expected_formal_optimizer_steps=12000, experiments={})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    try:
        for label, cmd in commands(args):
            state.update(status='running', stage=label); write(state_path, state)
            a5.invoke(cmd, args.out/(label+'.log'), env)
            # Even a successful process exit is insufficient without both reports.
            reports = completed_batch(args.out/label, label)
            state['experiments'][label] = dict(status='completed', comparisons=reports)
            write(state_path, state)
        state.update(status='completed', stage='done', timing_windows=200,
                     timing_rank_records=1600, formal_optimizer_steps=12000)
        write(state_path, state)
        print('Completed: ablation5 x2, then ablation4 x2; 200 windows / 12000 formal steps.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); write(state_path, state)
        raise


def main(argv=None):
    args = arguments(argv)
    print('Batch:', args.out, flush=True)
    print('Order: ablation5/compare1 -> ablation5/compare2 -> ablation4/compare1 -> ablation4/compare2.', flush=True)
    print('Each: --blocks 5 --rounds 2 --nvls-enable 0; 10 warmup + 60 timed steps/window; 50 formal windows.', flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, fork, subprocesses or GPU queries. Add --execute to detach and run.')
        for _, cmd in commands(args): print(shlex.join(cmd))
        print('Total: 4 fresh preflights/reports, 200 formal windows / 1600 rank records / 12000 formal steps.')
        return
    require(args.python.is_file() and (args.build/'ablation4-manifest.json').is_file() and
            (args.build/'candidate/manifest.json').is_file(), 'Training Python / existing ablation4 and C builds missing')
    args.out.mkdir(parents=True, exist_ok=False); (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(status='launching', expected_timing_windows=200,
          expected_formal_optimizer_steps=12000, execution_timing=ablation_timing.execution_timing()))
    write(args.out/'handoff/launch.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        if args.foreground:
            supervise(args)
        else:
            identity = a5.detach(args, supervisor=supervise)
            print(json.dumps(identity, ensure_ascii=False), flush=True)
            print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.', flush=True)
    except Exception as error:
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); write(args.out/'batch-state.json', state)
        raise


if __name__ == '__main__': main()
