#!/usr/bin/env python3
"""Two independent budget sweeps, detached under container PID 1 with --execute.

Default: print a read-only plan. --execute creates a fresh batch, double-forks,
redirects all standard descriptors and confirms PID/PPID before starting work.
--foreground is available for an intentionally attached diagnostic run.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import ablation5_protocol as protocol
import ablation_timing
from common import REPO, read, require, sha, write

ENTRY = Path(__file__).with_name('ablation5.py')
DEFAULT_BUILD = REPO/'outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build/candidate'
TRIALS = ('compare1', 'compare2')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-build', type=Path, default=DEFAULT_BUILD)
    p.add_argument('--out', type=Path, help='New batch directory; existing paths are rejected')
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true', help='Explicitly keep the supervisor attached to this terminal')
    args = p.parse_args(argv)
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/('ablation5-budget-'+stamp)
    for key in ('candidate_build', 'out', 'python', 'megatron', 'cuda_home'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def commands(args):
    cli = [str(args.python), str(ENTRY)]
    plan = args.out/'budget-plan.json'
    steps = [('plan', cli+['plan', str(plan), '--candidate-build', str(args.candidate_build)])]
    for trial in TRIALS:
        steps += [(trial+'-prepare', cli+['prepare', str(args.out/trial), '--candidate-build', str(args.candidate_build),
                   '--plan', str(plan), '--megatron', str(args.megatron), '--blocks', '5', '--rounds', '2', '--nvls-enable', '0']),
                  (trial+'-dry-run', cli+['run', str(args.out/trial), '--python', str(args.python), '--cuda-home', str(args.cuda_home)])]
    for trial in TRIALS:
        steps += [(trial+'-controller', cli+['run', str(args.out/trial), '--stage', 'all', '--python', str(args.python),
                                            '--cuda-home', str(args.cuda_home), '--execute']),
                  (trial+'-report', cli+['report', str(args.out/trial)])]
    return steps


def invoke(cmd, log, env):
    print(shlex.join(cmd), flush=True)
    print('Log:', log, flush=True)
    with log.open('x') as f:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def check_inputs(root):
    info = read(root/'submission.json')
    ablation_timing.require_new(info)
    ablation_timing.validate(root, info)
    require(info['protocol'] == protocol.PROTOCOL and info['executable'] and not info['source_only'] and
            info['rounds'] == 2 and info['blocks_per_round'] == 5 and info['timing_windows'] == 50 and
            info['schedule'] == protocol.schedule(5, rounds=2) and
            info['budget_denominators'] == list(protocol.DENOMINATORS) and
            info['communication']['environment'].get('NCCL_NVLS_ENABLE') == '0', 'Wrong budget comparison')
    return info


def check_report(root, info):
    result = read(root/'report.json'); gate = read(root/'preflight.json')
    ablation_timing.require_report(result, 50)
    require(read(root/'all-state.json')['status'] == 'completed' and result['completed'] and result['correctness_passed'] and
            result['protocol'] == protocol.PROTOCOL and result['run_id'] == info['run_id'] and
            result['round_count'] == 2 and result['blocks_per_round'] == 5 and result['timing_windows'] == 50 and
            len(result['all_samples']) == 50 and result['timing_rank_records'] == 400 and
            result['preflight_rank_records'] == 80 and result['profile_traces'] == 40 and result['candidate_shape_mappings'] == 10 and
            result['budget_denominators'] == list(protocol.DENOMINATORS), 'Incomplete budget report; stop before next comparison')
    require(gate['completed'] and gate['submission_sha256'] == sha(root/'submission.json') and
            result['evidence']['preflight_sha256'] == sha(root/'preflight.json') and
            result['communication'] == gate['communication'] == info['communication'], 'Report/preflight identity mismatch')
    return result


def supervise(args):
    state_path = args.out/'batch-state.json'
    state = dict(status='preparing', budget_denominators=list(protocol.DENOMINATORS), rounds=2,
                 blocks_per_round=5, nvls_enable='0', expected_timing_windows=100, comparisons={},
                 expected_formal_optimizer_steps=6000, execution_timing=ablation_timing.execution_timing())
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    inputs = {}; reports = []
    try:
        for label, cmd in commands(args):
            state.update(stage=label, status='running' if 'controller' in label or 'report' in label else 'preparing')
            write(state_path, state); invoke(cmd, args.out/(label+'.log'), env)
            if label.endswith('-dry-run'):
                trial = label.removesuffix('-dry-run'); info = check_inputs(args.out/trial); inputs[trial] = info
                if trial == 'compare2':
                    from ablation4_repeat import same_inputs
                    same_inputs(inputs['compare1'], info)
                state['comparisons'][trial] = dict(run_id=info['run_id'], status='prepared')
            if label.endswith('-report'):
                trial = label.removesuffix('-report'); report = check_report(args.out/trial, inputs[trial])
                if reports:
                    require(report['measurement_identity'] == reports[0]['measurement_identity'] and
                            report['role_plan_sha256'] == reports[0]['role_plan_sha256'], 'Repeat runtime/masks changed')
                reports.append(report)
                state['comparisons'][trial].update(status='completed', timing_windows=50, report=str(args.out/trial/'report.json'))
            write(state_path, state)
        require(len(reports) == 2, 'Both independent reports are required')
        state.update(status='completed', stage='done', timing_windows=100, timing_rank_records=800, formal_optimizer_steps=6000)
        write(state_path, state)
        print('Completed both strict reports: 100 formal windows / 800 rank records / 6000 formal steps.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); write(state_path, state)
        raise


def detach(args, supervisor=None):
    """Parent receives readiness only after reparenting and descriptor setup."""
    read_fd, write_fd = os.pipe()
    first = os.fork()
    if first:
        os.close(write_fd); os.waitpid(first, 0)
        ready, _, _ = select.select([read_fd], [], [], 15)
        if not ready:
            os.close(read_fd)
            raise RuntimeError('Detach readiness timed out; inspect handoff/container-supervisor.log')
        with os.fdopen(read_fd) as pipe: message = json.loads(pipe.readline())
        require(message.get('ready') is True, message.get('error', 'Detached supervisor failed'))
        return message
    os.close(read_fd)
    try:
        os.setsid()
        if os.fork(): os._exit(0)
        os.chdir(REPO); os.umask(0o027)
        fd = os.open('/dev/null', os.O_RDONLY)
        if fd != 0: os.dup2(fd, 0); os.close(fd)
        fd = os.open(args.out/'handoff/container-supervisor.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        os.dup2(fd, 1); os.dup2(fd, 2)
        if fd not in (1, 2): os.close(fd)
        deadline = time.monotonic()+5
        while os.getppid() != 1 and time.monotonic() < deadline: time.sleep(.05)
        require(os.getppid() == 1, 'Supervisor was not adopted by container PID 1; no experiment started')
        identity = dict(ready=True, pid=os.getpid(), ppid=os.getppid(), sid=os.getsid(0), pgid=os.getpgrp(),
                        cgroup=Path('/proc/self/cgroup').read_text(), repo=str(REPO), batch=str(args.out))
        write(args.out/'handoff/supervisor.json', identity)
        (args.out/'handoff/container-supervisor.pid').write_text(str(os.getpid())+'\n')
        os.write(write_fd, (json.dumps(identity)+'\n').encode()); os.close(write_fd); write_fd = -1
        (supervisor or supervise)(args)
        os._exit(0)
    except BaseException as error:
        traceback.print_exc()
        if write_fd >= 0:
            try:
                os.write(write_fd, (json.dumps(dict(ready=False, error=str(error)))+'\n').encode())
                os.close(write_fd)
            except OSError: pass
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json', state)
        os._exit(1)


def main(argv=None):
    args = arguments(argv)
    print('Batch:', args.out, flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, fork, subprocesses or GPU queries. Add --execute to detach and run.')
        for _, cmd in commands(args): print(shlex.join(cmd))
        print('Each run: 10 mappings, 10 preflight windows, 50 formal windows (10 warmup + 60 timed steps/window); two runs total 100 windows / 6000 formal steps.')
        return
    require(args.python.is_file() and (args.candidate_build/'manifest.json').is_file(), 'Training Python / existing C build missing')
    args.out.mkdir(parents=True, exist_ok=False); (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(status='launching', expected_timing_windows=100))
    write(args.out/'handoff/launch.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if args.foreground:
        supervise(args)
    else:
        try:
            identity = detach(args)
        except Exception as error:
            state = read(args.out/'batch-state.json')
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            write(args.out/'batch-state.json', state)
            raise
        print(json.dumps(identity, ensure_ascii=False), flush=True)
        print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.', flush=True)


if __name__ == '__main__': main()
