#!/usr/bin/env python3
"""Run two fresh ablation4 comparisons (5 blocks x 2 rounds, NVLS=0).

Run inside the experiment container with its training Python. The default only
prints commands; --execute prepares and runs compare1 then compare2 in the
foreground. Existing outputs are never reused, and any failure stops the batch.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

import ablation4_protocol as protocol
import ablation_timing
from common import REPO, read, require, sha, write

ENTRY = Path(__file__).with_name('ablation4.py')
DEFAULT_BUILD = REPO/'outputs/h100/phase6/ablation4-gpt67-tp8-both-20261018-hostservice/build'
TRIALS = ('compare1', 'compare2')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=DEFAULT_BUILD, help='Existing verified ablation4 build; no rebuilding')
    p.add_argument('--out', type=Path, help='New batch directory (default: timestamped logs/h100/phase6 directory)')
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--python', type=Path, default=Path(sys.executable), help='Training Python for controllers and workers')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'), help='Runtime CUDA, not build CUDA')
    p.add_argument('--execute', action='store_true', help='Create the new batch and actually run both experiments')
    return p


def normalize(args):
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/('ablation4-repeat-'+stamp)
    for key in ('build', 'out', 'megatron', 'python', 'cuda_home'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def commands(args, trial):
    root = args.out/trial
    cli = [str(args.python), str(ENTRY)]
    runtime = ['--python', str(args.python), '--cuda-home', str(args.cuda_home)]
    return [
        ('prepare', cli+['prepare', str(root), '--build', str(args.build), '--megatron', str(args.megatron),
                         '--blocks', '5', '--rounds', '2', '--nvls-enable', '0']),
        ('dry-run', cli+['run', str(root), *runtime]),
        ('controller', cli+['run', str(root), '--stage', 'all', *runtime, '--execute']),
        ('report', cli+['report', str(root)]),
    ]


def run_logged(command, path, env):
    print(shlex.join(command), flush=True)
    print('Log:', path, flush=True)
    with path.open('x') as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def submission(root):
    info = read(root/'submission.json')
    ablation_timing.require_new(info)
    ablation_timing.validate(root, info)
    require(info['protocol'] == protocol.PROTOCOL and info['executable'] is True and
            info['source_only'] is False and info['blocks_per_round'] == 5 and info['rounds'] == 2 and
            info['timing_windows'] == 50 and info['schedule'] == protocol.schedule(5, rounds=2),
            'Expected a real five-policy, two-round, 50-window comparison')
    require(info['communication']['environment'].get('NCCL_NVLS_ENABLE') == '0' and
            info['communication']['nvls_request'] == '0', 'NVLS=0 must be frozen for both comparisons')
    return info


def same_inputs(first, second):
    require(first['run_id'] != second['run_id'], 'Each comparison needs a new run_id')
    for key in ('config', 'execution_timing', 'schedule', 'variants', 'communication', 'calibration_hardware', 'megatron',
                'megatron_commit', 'candidate_source_commit', 'build_manifest', 'build_files', 'controllers'):
        require(first[key] == second[key], 'Comparison inputs differ: '+key)
    worker_hashes = lambda info: {k: v for k, v in info['files'].items() if k.startswith('scripts/')}
    require(worker_hashes(first) == worker_hashes(second), 'Frozen workers/plans differ')


def completed_report(root, info):
    result = read(root/'report.json'); gate = read(root/'preflight.json')
    ablation_timing.require_report(result, 50)
    require(read(root/'all-state.json')['status'] == 'completed' and result['completed'] is True and
            result['correctness_passed'] is True and result['protocol'] == protocol.PROTOCOL and
            result['run_id'] == info['run_id'] and result['round_count'] == 2 and result['blocks_per_round'] == 5 and
            result['timing_windows'] == 50 and len(result['all_samples']) == 50 and
            result['timing_rank_records'] == 400 and result['preflight_rank_records'] == 80 and
            result['profile_traces'] == 40 and result['candidate_shape_mappings'] == 8,
            'Strict report/state is incomplete; the next comparison will not start')
    require(gate['completed'] is True and gate['submission_sha256'] == sha(root/'submission.json') and
            result['evidence']['preflight_sha256'] == sha(root/'preflight.json') and
            result['communication'] == gate['communication'] == info['communication'],
            'Report must certify this comparison and its own preflight')
    return result


def execute(args):
    require((args.build/'ablation4-manifest.json').is_file(), 'Existing ablation4 build not found: '+str(args.build))
    require(args.python.is_file(), 'Training Python not found: '+str(args.python))
    # Reserve the whole batch before any subprocess: even an empty existing
    # directory is rejected, so a rerun cannot overwrite evidence.
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    state = dict(status='preparing', blocks_per_round=5, rounds=2, nvls_enable='0',
                 expected_timing_windows=100, expected_formal_optimizer_steps=6000,
                 execution_timing=ablation_timing.execution_timing(), build=str(args.build), comparisons={})
    state_path = args.out/'batch-state.json'
    write(state_path, state)
    try:
        inputs = {}
        for trial in TRIALS:
            state.update(current_trial=trial, stage='prepare'); write(state_path, state)
            for label, cmd in commands(args, trial)[:2]:
                state['stage'] = label; write(state_path, state)
                run_logged(cmd, args.out/f'{trial}-{label}.log', env)
            inputs[trial] = submission(args.out/trial)
            state['comparisons'][trial] = dict(run_id=inputs[trial]['run_id'], status='prepared')
        same_inputs(inputs['compare1'], inputs['compare2'])
        results = []
        for trial in TRIALS:
            state.update(status='running', current_trial=trial)
            for label, cmd in commands(args, trial)[2:]:
                state['stage'] = label; write(state_path, state)
                run_logged(cmd, args.out/f'{trial}-{label}.log', env)
            result = completed_report(args.out/trial, inputs[trial])
            if results:
                require(result['measurement_identity'] == results[0]['measurement_identity'],
                        'Runtime/worker/library identity changed between comparisons')
            results.append(result)
            state['comparisons'][trial].update(status='completed', timing_windows=50,
                report=str(args.out/trial/'report.json'))
            write(state_path, state)
        state.update(status='completed', stage='done', timing_windows=100, timing_rank_records=800, formal_optimizer_steps=6000)
        write(state_path, state)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(state_path, state)
        raise
    print('Completed: two independent reports, 100 formal windows / 800 rank records / 6000 formal steps.', flush=True)
    print('Batch:', args.out, flush=True)
    return state


def main(argv=None):
    args = normalize(parser().parse_args(argv))
    print('Batch:', args.out, flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, subprocesses, GPU queries or training. Add --execute to run.')
        for phase in (slice(0, 2), slice(2, None)):
            for trial in TRIALS:
                for _, cmd in commands(args, trial)[phase]: print(shlex.join(cmd))
        print('Each comparison: 8 mappings + 10 preflight windows + 50 formal windows (10 warmup + 60 timed steps/window); total 100 windows / 6000 formal steps.')
        return
    execute(args)


if __name__ == '__main__':
    main()
