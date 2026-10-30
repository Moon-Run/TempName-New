#!/usr/bin/env python3
"""Build A/D/L/F, verify operator evidence, and confirm full optimizer steps twice.

Without --execute, print a read-only plan. With --execute, use the double-fork
supervisor inside the intended eight-H100 server/container. op4 tests decoder
and workspace lifecycle changes; it makes no arrival-order benefit claim.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import arrival_op4_bench as bench
import arrival_op4_fullstep as fullstep
import detach_ablation5_run_two as launcher
from common import HERE, REPO, read, require, sha, verify_files, write

PROTOCOL = 'h100-arrival-op4-detached-v1'
ATTRIBUTIONS = {
    'lifecycle_benefit_validated': 'L/A',
    'lifecycle_increment_validated': 'F/D',
    'decoder_control_passed': 'D/A',
    'decoder_increment_validated': 'F/L',
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    parser.add_argument('--nccl-root', type=Path,
                        help='Optional NCCL headers/static library; otherwise build a private copy')
    parser.add_argument('--op4-build', type=Path, help='Reuse a complete executable op4 A/D/L/F build')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=1800,
                        help='Seconds per operator/training torchrun process group; stop on timeout')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args(argv)
    require(args.jobs > 0 and args.worker_timeout > 0, 'Positive jobs and worker timeout required')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/f'arrival-op4-2x-50steps-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # A venv's Python commonly symlinks to a system interpreter. Keep
            # its invocation path so Python still discovers that pyvenv.cfg.
            expanded = value.expanduser()
            setattr(args, key, expanded.absolute() if key == 'python' else expanded.resolve())
    return args


def commands(args):
    def cli(name):
        return [str(args.python), str(HERE/name)]
    build = args.op4_build or args.out/'build'
    tasks = []
    if args.op4_build is None:
        compile_flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
        if args.nccl_root:
            compile_flags += ['--nccl-root', str(args.nccl_root)]
        tasks += [
            ('build-prepare', cli('arrival_op4_build.py')+['prepare', '--out', str(build)]),
            ('compile', cli('arrival_op4_build.py')+['build', '--build', str(build), *compile_flags]),
        ]
    runtime_flags = ['--python', str(args.python), '--cuda-home', str(args.cuda_home),
                     '--worker-timeout', str(args.worker_timeout)]
    micro, train = args.out/'operator', args.out/'fullstep'
    tasks += [
        ('operator-prepare', cli('arrival_op4_bench.py')+['prepare', str(micro), '--build', str(build)]),
        ('operator-timing', cli('arrival_op4_bench.py')+['run', str(micro), *runtime_flags, '--execute']),
        ('operator-profile', cli('arrival_op4_bench.py')+[
            'run', str(micro), *runtime_flags, '--profile', '--execute']),
        ('fullstep-prepare', cli('arrival_op4_fullstep.py')+[
            'prepare', str(train), '--build', str(build), '--megatron', str(args.megatron),
            '--trials', '2', '--blocks', '12']),
        ('fullstep-run', cli('arrival_op4_fullstep.py')+[
            'run', str(train), '--stage', 'all', *runtime_flags, '--execute']),
        ('fullstep-profile', cli('arrival_op4_fullstep.py')+[
            'run', str(train), '--stage', 'profile', *runtime_flags, '--execute']),
    ]
    return tasks


def check_operator(root):
    plan = bench.verify(root)
    report, profile = read(root/'report.json'), read(root/'profile-summary.json')
    require(report['protocol'] == bench.PROTOCOL and report['run_id'] == plan['run_id'] and
            report['completed'] is True and report['correctness_passed'] is True and
            report['bitwise_equal_to_A'] is True and report['target_validated'] is False,
            'Complete same-build A/D/L/F operator correctness is required')
    verify_files(root, report['evidence_files'])
    require(report == bench.report(root, save=False),
            'Operator report differs from raw paired rank records')
    require(profile['protocol'] == bench.PROTOCOL and profile['run_id'] == plan['run_id'] and
            profile['completed'] is True and
            profile['execution_path_validated'] is True and profile['target_validated'] is False,
            'Operator execution-path evidence is required')
    verify_files(root, profile['evidence_files'])
    require(profile == bench.profile_summary(root), 'Profile summary differs from raw evidence')
    # All performance estimates are diagnostics; their sign never gates training.
    return dict(operator_execution_path_validated=True,
                micro_savings_estimate=report['micro_savings_estimate'],
                operator_comparisons={role: report['roles'][role]['comparisons'] for role in bench.ROLES})


def fullstep_gate(root):
    plan = fullstep.verify(root)
    report = read(root/'report.json')
    require(report['protocol'] == fullstep.PROTOCOL and report['run_id'] == plan['run_id'] and
            report['completed'] is True and report['correctness_passed'] is True and
            report['full_optimizer_steps_measured'] is True and
            report['formal_optimizer_steps'] == 4800 and report['timing_rank_records'] == 768 and
            len(report['all_samples']) == 96 and
            set(report['comparisons']) == {'compare1', 'compare2'},
            'Both complete 10+50-step confirmations are required')
    verify_files(root, report['evidence_files'])
    require(report == fullstep.report(root, save=False),
            'Full-step report differs from raw paired rank records')
    accepted, attributed = [], {field: [] for field in ATTRIBUTIONS}
    for name, comparison in report['comparisons'].items():
        number = name.removeprefix('compare')
        require(comparison['run_id'] == plan['trial_ids'][number], 'Comparison identity changed')
        estimates = comparison['rounds']['combined']['comparisons']
        total = estimates['F/A']
        passed = total['percent'] >= 1.0 and total['ci95'][0] >= 1.0
        require(comparison['target_validated'] is passed, 'Comparison acceptance changed')
        accepted.append(passed)
        for field, contrast in ATTRIBUTIONS.items():
            estimate = estimates[contrast]
            positive = estimate['percent'] > 0 and estimate['ci95'][0] > 0
            require(comparison[field] is positive, f'{contrast} attribution changed')
            attributed[field].append(positive)
    require(report['target_validated'] is all(accepted), 'Full-step acceptance changed')
    for field, outcomes in attributed.items():
        require(report[field] is all(outcomes), f'Full-step {ATTRIBUTIONS[field]} attribution changed')
    return report


def check_fullstep_profile(root):
    profile = read(root/'profile-summary.json')
    require(profile['protocol'] == fullstep.PROTOCOL and profile['completed'] is True and
            profile['execution_validated'] is True,
            'Complete full-step execution-path profile is required')
    verify_files(root, profile['evidence_files'])
    require(profile == fullstep.profile_summary(root), 'Full-step profile differs from raw evidence')
    return profile


def supervise(args):
    state_path = args.out/'batch-state.json'
    state = dict(protocol=PROTOCOL, status='preparing', target_percent=1.0,
                 target_validated=False, operator_execution_path_validated=False,
                 fullstep_execution_path_validated=False,
                 expected_formal_optimizer_steps=4800, expected_timing_windows=96,
                 worker_timeout=args.worker_timeout, micro_savings_estimate=None, reports={},
                 **{field: False for field in ATTRIBUTIONS})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    try:
        if args.op4_build:
            bench.build_identity(args.op4_build)
        for label, command in commands(args):
            state.update(status='running', stage=label)
            write(state_path, state)
            launcher.invoke(command, args.out/(label+'.log'), env)
            if label == 'operator-profile':
                state.update(check_operator(args.out/'operator'))
                state['reports'].update(operator=str(args.out/'operator/report.json'),
                                        operator_profile=str(args.out/'operator/profile-summary.json'))
            elif label == 'fullstep-run':
                result = fullstep_gate(args.out/'fullstep')
                state['reports']['fullstep'] = str(args.out/'fullstep/report.json')
                state.update(target_validated=result['target_validated'],
                             acceptance_status=result['acceptance_status'],
                             formal_optimizer_steps=4800, timing_windows=96,
                             **{field: result[field] for field in ATTRIBUTIONS})
                print('F/A total speedup; L/A and F/D lifecycle effects; D/A and F/L decoder effects. '
                      'These results do not measure arrival-order benefit.', flush=True)
            elif label == 'fullstep-profile':
                check_fullstep_profile(args.out/'fullstep')
                state['fullstep_execution_path_validated'] = True
                state['reports']['fullstep_profile'] = str(args.out/'fullstep/profile-summary.json')
        state.update(status='completed', stage='done')
        write(state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(state_path, state)
        raise


def main(argv=None):
    args = arguments(argv)
    print('Batch:', args.out, flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, subprocesses, forks or GPU queries.')
        for _, command in commands(args):
            print(shlex.join(command))
        print('After correctness and evidence checks, run 2 comparisons, 96 windows, 4800 formal steps. '
              'Operator speedups are diagnostics only; they never gate full training. '
              'Full-step profiles run separately after formal timing. '
              'Both F/A point estimates and CI95 lower bounds must reach 1%.')
        return
    require(args.python.is_file() and args.megatron.is_dir(), 'Python / Megatron source missing')
    require((args.cuda_home/'lib64').is_dir(), 'Runtime CUDA toolkit missing; pass --cuda-home')
    if args.op4_build is None:
        require((args.build_cuda_home/'bin/nvcc').is_file(),
                'Build CUDA toolkit missing; pass --build-cuda-home')
    if args.nccl_root:
        require(all((args.nccl_root/name).is_file()
                    for name in ('include/nccl.h', 'lib/libnccl_static.a')),
                'NCCL headers/static library missing')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(protocol=PROTOCOL, status='launching', target_validated=False))
    write(args.out/'handoff/launch.json',
          {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write(args.out/'handoff/controllers.json', {
        str(path): sha(path) for path in
        (Path(__file__), HERE/'arrival_op4_build.py', HERE/'arrival_op4_bench.py',
         HERE/'arrival_op4_fullstep.py', HERE/'detach_ablation5_run_two.py')})
    if args.foreground:
        supervise(args)
    else:
        try:
            print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
            print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.')
        except Exception as error:
            state = read(args.out/'batch-state.json')
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            write(args.out/'batch-state.json', state)
            raise


if __name__ == '__main__':
    main()
