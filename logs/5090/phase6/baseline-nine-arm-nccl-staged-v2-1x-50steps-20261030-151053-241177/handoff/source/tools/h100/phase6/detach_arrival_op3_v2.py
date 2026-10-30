#!/usr/bin/env python3
"""Build A/D/O, collect operator diagnostics, then confirm full steps twice.

Without --execute, only print the plan. With --execute, use the same detached
supervisor as op2. Run inside the intended eight-H100 server/container.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import arrival_op3_v2_bench as bench
import arrival_op3_v2_fullstep as fullstep
import detach_ablation5_run_two as launcher
from common import HERE, REPO, read, require, sha, verify_files, write

PROTOCOL = 'h100-arrival-op3-v2-detached-v1'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    parser.add_argument('--nccl-root', type=Path,
                        help='Optional NCCL headers/static library; otherwise build a private copy')
    parser.add_argument('--op3-build', type=Path, help='Reuse a complete executable op3 A/D/O build')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--rounds', type=int, choices=range(1, 5), default=2,
                        help='Finite early-scan rounds, identical in operator and full-step runs')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args(argv)
    require(args.jobs > 0, 'Positive build jobs required')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/f'arrival-op3-v2-2x-50steps-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def commands(args):
    def cli(name):
        return [str(args.python), str(HERE/name)]
    build = args.op3_build or args.out/'build'
    tasks = []
    if args.op3_build is None:
        compile_flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
        if args.nccl_root:
            compile_flags += ['--nccl-root', str(args.nccl_root)]
        tasks += [
            ('build-prepare', cli('arrival_op3_v2_build.py')+['prepare', '--out', str(build)]),
            ('compile', cli('arrival_op3_v2_build.py')+['build', '--build', str(build), *compile_flags]),
        ]
    runtime_flags = ['--python', str(args.python), '--cuda-home', str(args.cuda_home)]
    micro, train = args.out/'operator', args.out/'fullstep'
    tasks += [
        ('operator-prepare', cli('arrival_op3_v2_bench.py')+[
            'prepare', str(micro), '--build', str(build), '--rounds', str(args.rounds)]),
        ('operator-timing', cli('arrival_op3_v2_bench.py')+[
            'run', str(micro), *runtime_flags, '--execute']),
        ('operator-profile', cli('arrival_op3_v2_bench.py')+[
            'run', str(micro), *runtime_flags, '--profile', '--execute']),
        ('fullstep-prepare', cli('arrival_op3_v2_fullstep.py')+[
            'prepare', str(train), '--build', str(build), '--megatron', str(args.megatron),
            '--trials', '2', '--blocks', '12', '--rounds', str(args.rounds)]),
        ('fullstep-run', cli('arrival_op3_v2_fullstep.py')+[
            'run', str(train), *runtime_flags, '--execute']),
    ]
    return tasks


def check_operator(root):
    plan = bench.verify(root)
    report, profile = read(root/'report.json'), read(root/'profile-summary.json')
    require(report['protocol'] == bench.PROTOCOL and report['run_id'] == plan['run_id'] and
            report['completed'] is True and report['correctness_passed'] is True and
            report['bitwise_equal_to_A'] is True and report['target_validated'] is False,
            'Complete same-build A/D/S/O operator correctness is required')
    verify_files(root, report['evidence_files'])
    require(report == bench.report(root, save=False),
            'Operator report differs from raw paired rank records')
    # Recompute from diagnostic records/traces; a cached Boolean is not proof.
    require(profile == bench.profile_summary(root), 'Profile summary differs from raw evidence')
    # Performance and overlap are diagnostics, never prerequisites for training.
    return dict(useful_early_consumption_validated=profile['useful_early_consumption_validated'] is True,
                micro_savings_estimate=report['micro_savings_estimate'],
                decoder_control={role: report['roles'][role]['comparisons']['D/A'] for role in bench.ROLES})


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
    accepted, arrival_accepted, decoder_accepted = [], [], []
    for name, comparison in report['comparisons'].items():
        number = name.removeprefix('compare')
        require(comparison['run_id'] == plan['trial_ids'][number], 'Comparison identity changed')
        estimate = comparison['rounds']['combined']['comparisons']['O/A']
        passed = estimate['percent'] >= 1.0 and estimate['ci95'][0] >= 1.0
        require(comparison['target_validated'] is passed, 'Comparison acceptance changed')
        accepted.append(passed)
        arrival = comparison['rounds']['combined']['comparisons']['O/S']
        arrival_passed = arrival['percent'] > 0 and arrival['ci95'][0] > 0
        require(comparison['arrival_benefit_validated'] is arrival_passed,
                'Arrival benefit attribution changed')
        arrival_accepted.append(arrival_passed)
        decoder = comparison['rounds']['combined']['comparisons']['D/A']
        decoder_passed = decoder['percent'] > 0 and decoder['ci95'][0] > 0
        require(comparison['decoder_control_passed'] is decoder_passed,
                'Decoder control attribution changed')
        decoder_accepted.append(decoder_passed)
    require(report['target_validated'] is all(accepted), 'Full-step acceptance changed')
    require(report['arrival_benefit_validated'] is all(arrival_accepted) and
            report['decoder_control_passed'] is all(decoder_accepted), 'Full-step attribution changed')
    return report


def supervise(args):
    state_path = args.out/'batch-state.json'
    state = dict(protocol=PROTOCOL, status='preparing', target_percent=1.0,
                 target_validated=False, useful_early_consumption_validated=False,
                 arrival_benefit_validated=False, decoder_control_passed=False,
                 expected_formal_optimizer_steps=4800, expected_timing_windows=96,
                 rounds=args.rounds, micro_savings_estimate=None, reports={})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    try:
        if args.op3_build:
            bench.build_identity(args.op3_build)
        for label, command in commands(args):
            state.update(status='running', stage=label)
            write(state_path, state)
            launcher.invoke(command, args.out/(label+'.log'), env)
            if label == 'operator-profile':
                diagnostics = check_operator(args.out/'operator')
                state.update(diagnostics)
                state['reports'].update(operator=str(args.out/'operator/report.json'),
                                        profile=str(args.out/'operator/profile-summary.json'))
            elif label == 'fullstep-run':
                result = fullstep_gate(args.out/'fullstep')
                state['reports']['fullstep'] = str(args.out/'fullstep/report.json')
                state.update(target_validated=result['target_validated'],
                             arrival_benefit_validated=result['arrival_benefit_validated'],
                             decoder_control_passed=result['decoder_control_passed'],
                             attribution_status=result['attribution_status'],
                             acceptance_status=result['acceptance_status'],
                             formal_optimizer_steps=4800, timing_windows=96)
                if not result['arrival_benefit_validated']:
                    print('The incremental training benefit from early consumption (O/S) was not proven. '
                          'O/A total speedup and D/A decoder-control results are reported separately.', flush=True)
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
              'Operator savings and useful overlap are diagnostics only; neither gates full training.')
        return
    require(args.python.is_file() and args.megatron.is_dir(), 'Python / Megatron source missing')
    require((args.cuda_home/'lib64').is_dir(), 'Runtime CUDA toolkit missing; pass --cuda-home')
    if args.op3_build is None:
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
        (Path(__file__), HERE/'arrival_op3_v2_build.py', HERE/'arrival_op3_v2_bench.py',
         HERE/'arrival_op3_v2_fullstep.py', HERE/'detach_ablation5_run_two.py')})
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
