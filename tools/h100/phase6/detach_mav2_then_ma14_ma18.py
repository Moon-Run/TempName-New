#!/usr/bin/env python3
"""Run MAV2 with quality-failure continuation, then MA14 through MA20.

Preview unless --execute. One supervisor detaches under container PID 1 and
waits for each foreground child to finish, including its profiles and reports.
Closing the terminal or VS Code does not end the sequence; keep the container
and host running. Execution failures stop the sequence without retrying.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

import detach_ablation5_run_two as launcher
from common import HERE, REPO, read, require, write

DEFAULT_FLUX_BUILD = REPO/'logs/h100/phase6/mav2-three-arm-1x-50steps-20261030-215237-552847/flux-build'
MAV2_SCRIPT = HERE/'run_mav2.sh'
MA_SCRIPT = HERE/'run_ma14_ma15_ma16_ma17_ma18_ma19_ma20.sh'
ORDER = ['mav2', 'ma14', 'ma15', 'ma16', 'ma17', 'ma18', 'ma19', 'ma20']


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flux-build', type=Path, default=DEFAULT_FLUX_BUILD,
                   help='Existing GemmOnly build for MAV2; defaults to the 20261030-215237 batch')
    p.add_argument('--out', type=Path, help='Fresh sequence log directory')
    p.add_argument('--trials', type=int, choices=(1, 2), default=1,
                   help='Number of comparisons per experiment (default: 1)')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true', help='Keep the outer supervisor attached for diagnostics')
    args = p.parse_args(argv)
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/('mav2-then-ma14-ma20-'+stamp)
    args.out = args.out.expanduser().resolve()
    args.flux_build = args.flux_build.expanduser().resolve()
    return args


def commands(args):
    return [
        ('mav2', ['bash', str(MAV2_SCRIPT), '--flux-build', str(args.flux_build),
                  '--allow-quality-failure', '--foreground', '--trials', str(args.trials),
                  '--out', str(args.out/'mav2'), '--execute']),
        # This script already runs every MA child with --foreground and waits.
        ('ma14-ma20', ['bash', str(MA_SCRIPT), '--trials', str(args.trials), '--execute']),
    ]


def supervise(args):
    state_path = args.out/'batch-state.json'
    steps = commands(args)
    state = dict(status='running', order=ORDER, trials=args.trials,
                 stages={name: dict(status='pending', log=str(args.out/(name+'.log')))
                         for name, _ in steps})
    state['stages']['mav2']['batch'] = str(args.out/'mav2')
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    try:
        for name, command in steps:
            state['stage'] = name
            stage = state['stages'][name]
            stage['status'] = 'running'
            write(state_path, state)
            launcher.invoke(command, Path(stage['log']), env)
            stage.update(status='completed', exit_code=0)
            if name == 'mav2':
                child = read(args.out/'mav2/batch-state.json')
                require(child['status'] in ('completed', 'completed_with_quality_failure') and
                        child['stage'] == 'done' and child['allow_quality_failure'] is True and
                        child['trials'] == args.trials,
                        'MAV2 did not finish its full batch; MA14-MA20 will not start')
                stage.update(status=child['status'], quality_within_pairs=child['quality_within_pairs'])
            write(state_path, state)
        status = state['stages']['mav2']['status']
        state.update(status=status, stage='done')
        write(state_path, state)
        print('Sequence '+status+': '+' -> '.join(ORDER), flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        stage = state['stages'][state['stage']]
        stage['status'] = 'failed'
        if isinstance(error, subprocess.CalledProcessError):
            stage['exit_code'] = error.returncode
        write(state_path, state)
        raise


def main(argv=None):
    args = arguments(argv)
    print('Sequence batch:', args.out, flush=True)
    print('Order: '+' -> '.join(ORDER), flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, subprocesses, forks or GPU queries.')
        for _, command in commands(args):
            print(shlex.join(command))
        return
    require(MAV2_SCRIPT.is_file() and MA_SCRIPT.is_file(), 'Sequence launcher script missing')
    require((args.flux_build/'manifest.json').is_file(), 'MAV2 Flux build manifest missing; pass --flux-build')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(status='launching', order=ORDER, trials=args.trials))
    write(args.out/'handoff/launch.json', dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commands=commands(args)))
    try:
        if args.foreground:
            return supervise(args)
        print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
        print('Detached and started. You may close the terminal/VS Code; keep the container and host running.', flush=True)
        print('Status:', args.out/'batch-state.json', flush=True)
        print('MAV2 log:', args.out/'mav2.log', flush=True)
        print('MA14-MA20 log:', args.out/'ma14-ma20.log', flush=True)
    except Exception as error:
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json', state)
        raise


if __name__ == '__main__':
    main()
