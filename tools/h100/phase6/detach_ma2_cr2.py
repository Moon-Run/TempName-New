#!/usr/bin/env python3
"""Build CR2/CR/C/R and test native submission in two independent four-arm full-step comparisons.

Default: read-only command preview. --execute starts the detached supervisor
inside the intended eight-H100 server/container and preserves every result.
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
import ma2_cr2_bench as bench
import ma2_cr2_fullstep as fullstep
from common import HERE, REPO, read, require, sha, verify_files, write

PROTOCOL = 'h100-ma2-cr2-detached-v1'
ARMS = ('CR2', 'CR', 'C', 'R')
TRIALS, BLOCKS, STEPS = 2, 8, 50
WINDOWS = TRIALS * BLOCKS * len(ARMS)
FORMAL_STEPS = WINDOWS * STEPS
ATTRIBUTIONS = {'one_percent_over_C_validated':'CR2/C', 'one_percent_over_R_validated':'CR2/R',
                'strict_one_percent_over_C_validated':'CR2/C', 'strict_one_percent_over_R_validated':'CR2/R',
                'increment_over_CR_validated':'CR2/CR', 'beats_both_one_percent_validated':'CR2/C and CR2/R',
                'strict_one_percent_confidence_validated':'same control in both trials'}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    parser.add_argument('--nccl-root', type=Path, help='Optional NCCL development root; otherwise build a private copy')
    parser.add_argument('--cr2-build', type=Path, help='Reuse a verified executable CR2/CR/C/R build')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=1800)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args(argv)
    require(args.jobs > 0 and args.worker_timeout > 0, 'Positive jobs and worker timeout required')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/f'ma2-cr2-{TRIALS}x-50steps-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            expanded = value.expanduser()
            setattr(args, key, expanded.absolute() if key == 'python' else expanded.resolve())
    return args


def commands(args):
    def cli(name):
        return [str(args.python), str(HERE/name)]
    build = args.cr2_build or args.out/'build'
    tasks = []
    if args.cr2_build is None:
        flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
        if args.nccl_root:
            flags += ['--nccl-root', str(args.nccl_root)]
        tasks += [('build-prepare', cli('ma2_cr2_build.py')+['prepare', '--out', str(build)]),
                  ('compile', cli('ma2_cr2_build.py')+['build', '--build', str(build), *flags])]
    runtime = ['--python', str(args.python), '--cuda-home', str(args.cuda_home),
               '--worker-timeout', str(args.worker_timeout)]
    micro, train = args.out/'operator', args.out/'fullstep'
    tasks += [
        ('operator-prepare', cli('ma2_cr2_bench.py')+['prepare', str(micro), '--build', str(build), '--blocks', str(BLOCKS)]),
        ('operator-timing', cli('ma2_cr2_bench.py')+['run', str(micro), *runtime, '--execute']),
        ('operator-profile', cli('ma2_cr2_bench.py')+['run', str(micro), *runtime, '--profile', '--execute']),
        ('fullstep-prepare', cli('ma2_cr2_fullstep.py')+['prepare', str(train), '--build', str(build),
            '--megatron', str(args.megatron), '--trials', str(TRIALS), '--blocks', str(BLOCKS)]),
        ('fullstep-run', cli('ma2_cr2_fullstep.py')+['run', str(train), '--stage', 'all', *runtime, '--execute']),
        ('fullstep-profile', cli('ma2_cr2_fullstep.py')+['run', str(train), '--stage', 'profile', *runtime, '--execute']),
    ]
    return tasks


def check_operator(root):
    plan = bench.verify(root)
    report, profile = read(root/'report.json'), read(root/'profile-summary.json')
    require(report['protocol'] == bench.PROTOCOL and report['run_id'] == plan['run_id'] and
            report['completed'] is True and report['correctness_passed'] is True and
            report['bitwise_equal_within_pairs'] is True and report['target_validated'] is False,
            'Complete within-family bitwise correctness evidence required')
    require(report['equivalence_pairs']==[['CR2','CR'],['CR','C']], 'Wrong bitwise equivalence pairs')
    verify_files(root, report['evidence_files'])
    require(report == bench.report(root, save=False), 'Operator report differs from raw rank records')
    require(profile['protocol'] == bench.PROTOCOL and profile['run_id'] == plan['run_id'] and
            profile['completed'] is True and profile['execution_path_validated'] is True and
            profile['target_validated'] is False, 'Operator execution-path evidence is required')
    verify_files(root, profile['evidence_files'])
    require(profile == bench.profile_summary(root), 'Operator profile differs from raw evidence')
    return dict(operator_execution_path_validated=True, operator_bitwise_equal_within_pairs=True,
                operator_comparisons={role: report['roles'][role]['comparisons'] for role in bench.ROLES})


def fullstep_gate(root):
    plan = fullstep.verify(root)
    report = read(root/'report.json')
    require(plan['trials'] == TRIALS and plan['blocks'] == BLOCKS and
            report['protocol'] == fullstep.PROTOCOL and report['run_id'] == plan['run_id'] and
            report['completed'] is True and report['correctness_passed'] is True and
            report['identical_trajectories_within_pairs'] is True and report['full_optimizer_steps_measured'] is True and
            report['target_percent'] == fullstep.TARGET_PERCENT and
            report['formal_optimizer_steps'] == FORMAL_STEPS and report['timing_rank_records'] == WINDOWS * 8 and
            len(report['all_samples']) == WINDOWS and
            set(report['comparisons']) == {f'compare{i}' for i in range(1, TRIALS+1)},
            'Two complete independent 8-block, 10+50-step comparisons are required')
    verify_files(root, report['evidence_files'])
    require(report == fullstep.report(root, save=False), 'Full-step report differs from raw rank records')
    recomputed = {}
    for name, comparison in report['comparisons'].items():
        require(comparison['run_id'] == plan['trial_ids'][name.removeprefix('compare')], 'Comparison identity changed')
        estimates = comparison['rounds']['combined']['comparisons']
        acceptance = fullstep.comparison_acceptance(estimates)
        recomputed[name] = acceptance
        for field, contrast in ATTRIBUTIONS.items():
            require(comparison[field] is acceptance[field], contrast+' acceptance field changed: '+field)
        passed = acceptance['target_validated']
        require(comparison['target_validated'] is passed, 'Comparison acceptance changed')
    aggregate = fullstep.aggregate_acceptance(recomputed)
    for field, value in aggregate.items():
        require(report[field] is value, 'Full-step same-control two-trial acceptance changed: '+field)
    return report


def check_fullstep_profile(root):
    plan = fullstep.verify(root)
    profile = read(root/'profile-summary.json')
    require(profile['protocol'] == fullstep.PROTOCOL and profile['run_id'] == plan['run_id'] and
            profile['completed'] is True and profile['execution_validated'] is True,
            'Complete same-run full-step execution profile is required')
    verify_files(root, profile['evidence_files'])
    require(profile == fullstep.profile_summary(root), 'Full-step profile differs from raw evidence')
    return profile


def supervise(args):
    state_path = args.out/'batch-state.json'
    state = dict(protocol=PROTOCOL, status='preparing', main_contrasts=['CR2/C','CR2/R'], mechanism_contrast='CR2/CR',
                 target_percent=fullstep.TARGET_PERCENT, target_validated=False, operator_bitwise_equal_within_pairs=False,
                 operator_execution_path_validated=False, fullstep_execution_path_validated=False,
                 expected_formal_optimizer_steps=FORMAL_STEPS, expected_timing_windows=WINDOWS,
                 worker_timeout=args.worker_timeout, reports={}, **{field: False for field in ATTRIBUTIONS})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    try:
        if args.cr2_build:
            bench.build_identity(args.cr2_build)
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
                state.update(target_validated=result['target_validated'], acceptance_status=result['acceptance_status'],
                             formal_optimizer_steps=FORMAL_STEPS, timing_windows=WINDOWS,
                             **{field: result[field] for field in ATTRIBUTIONS})
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
        print(f'{TRIALS} independent comparisons, {WINDOWS} windows, {FORMAL_STEPS} formal steps; 10 warmup + 50 timed steps/window. '
              'CR2 adds cached native submission and direct output to CR; CR, C and R preserve their original implementations. '
              'Both trials must show >=1% CR2 speedup and CI95 lower >0 against the SAME C or R control. '
              'Passing both controls and CI95 lower >=1% are separate diagnostics. '
              'Operator speedups are diagnostics only; negative results still run full training. '
              'Separate full-step profiles follow timing.')
        return
    require(args.python.is_file() and args.megatron.is_dir(), 'Python / Megatron source missing')
    require((args.cuda_home/'lib64').is_dir(), 'Runtime CUDA toolkit missing; pass --cuda-home')
    if args.cr2_build is None:
        require((args.build_cuda_home/'bin/nvcc').is_file(), 'Build CUDA toolkit missing; pass --build-cuda-home')
    if args.nccl_root:
        require(all((args.nccl_root/name).is_file() for name in ('include/nccl.h', 'lib/libnccl_static.a')),
                'NCCL headers/static library missing')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(protocol=PROTOCOL, status='launching', target_validated=False))
    write(args.out/'handoff/launch.json',
          {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write(args.out/'handoff/controllers.json', {str(path): sha(path) for path in
          (Path(__file__), HERE/'ma2_cr2_build.py', HERE/'ma2_cr2_sources.py', HERE/'ma2_cr2_bench.py',
           HERE/'ma2_cr2_fullstep.py', HERE/'detach_ablation5_run_two.py')})
    if args.foreground:
        return supervise(args)
    try:
        print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
        print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.')
    except BaseException as error:
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json', state)
        raise


if __name__ == '__main__':
    main()
