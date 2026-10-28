#!/usr/bin/env python3
"""Run two or four fresh C/A/joint-assignment-B comparisons, at 10+50 steps/window.

Default: read-only preview. --execute double-forks under container PID 1.
Build B once, then prepare/preflight/time/report each comparison in sequence.
Identity fallback is recorded and still measured in all requested runs.
Any execution or evidence failure stops the batch; negative speedups do not.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import ablation4_repeat as repeat
import ablation7_new_protocol as routing
import ablation7_new_report as reporter
import arrival_assignment
import detach_ablation5_run_two as launcher
from common import REPO, read, require, sha, write

ENTRY = Path(__file__).with_name('ablation7_new.py')
TRIALS = ('compare1', 'compare2', 'compare3', 'compare4')
BLOCKS = 6
ROUNDS = 2
WINDOWS = 36
TIMING = dict(warmup_steps=10, timed_steps=50)
TOTAL_WINDOWS = WINDOWS * len(TRIALS)
BUILD_MANIFEST = 'ablation7-new-manifest.json'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=repeat.DEFAULT_BUILD,
                        help='Verified original ablation4 parent build, including both role calibrations')
    parser.add_argument('--ablation7-build', type=Path,
                        help='Reuse a compiled ablation7-new joint-assignment build; old ablation7 is rejected')
    parser.add_argument('--out', type=Path, help='Fresh batch directory; existing paths are rejected')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    parser.add_argument('--nccl-root', type=Path, help='Default: verified NCCL development files in the parent manifest')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--trials', type=int, choices=(2, 4), default=4,
                        help='Independent complete comparisons (default: 4; use 2 for a shorter batch)')
    parser.add_argument('--arrival-margin-multiplier', type=float, default=1.0,
                        help='Frozen training margin in epsilon units (1..6; default 1 is exploratory)')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true', help='Keep the supervisor attached for diagnostics')
    args = parser.parse_args(argv)
    require(args.jobs > 0, 'jobs must be positive')
    requested_options(args)
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/(f'ablation7-new-{args.trials}x-50steps-'+stamp)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def trial_names(args):
    return TRIALS[:args.trials]


def requested_options(args):
    return arrival_assignment.normalize_options(
        {'uncertainty_multiplier': args.arrival_margin_multiplier})


def check_build_options(root, expected):
    """Reusing a v2/6-epsilon build must never bypass a new margin request."""
    expected = arrival_assignment.normalize_options(expected)
    record = read(Path(root)/BUILD_MANIFEST)
    plans = read(Path(root)/'plans.json')
    require(reporter.builder.planner.frozen_options(record) ==
            reporter.builder.planner.frozen_options(plans) == expected,
            'Reused build arrival margin differs from this request; prepare a fresh build')
    return record


def commands(args):
    cli = [str(args.python), str(ENTRY)]
    build = args.ablation7_build or args.out/'build'
    steps = []
    if args.ablation7_build is None:
        flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
        if args.nccl_root:
            flags += ['--nccl-root', str(args.nccl_root)]
        steps += [('build-prepare', cli+['build-prepare', str(build), '--parent-build', str(args.build),
                    '--arrival-margin-multiplier', str(requested_options(args)['uncertainty_multiplier'])]),
                  ('compile', cli+['compile', str(build), *flags])]
    for trial in trial_names(args):
        root = str(args.out/trial)
        run = cli+['run', root, '--python', str(args.python), '--cuda-home', str(args.cuda_home)]
        steps += [(trial+'-prepare', cli+['prepare', root, '--build', str(build),
                   '--megatron', str(args.megatron), '--blocks', str(BLOCKS),
                   '--rounds', str(ROUNDS), '--nvls-enable', '0']),
                  (trial+'-dry-run', run),
                  (trial+'-controller', run+['--stage', 'all', '--execute']),
                  (trial+'-report', cli+['report', root])]
    return steps


def order_summary(root):
    plans = read(root/'scripts/selection-plan.json')
    options = reporter.builder.planner.frozen_options(plans)
    method = arrival_assignment.METHOD if options is None else arrival_assignment.CONFIGURED_METHOD
    summary = reporter.arrival_metadata(plans)
    require(set(summary) == set(routing.SHAPES), 'Missing role arrival diagnostics')
    for role, record in summary.items():
        require(record['method'] == method and
                record['training_passes'] == [0, 1] and record['heldout_passes'] == [2] and
                record['identical_to_A'] == (not any(record['changed_per_rank'])),
                'Expected joint-assignment order with unchanged training/heldout split: '+role)
    return summary


def check_inputs(root, expected_options=None):
    info = read(root/'submission.json')
    require(info['protocol'] == routing.PROTOCOL and info['execution_timing'] == TIMING and
            info['executable'] and not info['source_only'] and info['rounds'] == ROUNDS and
            info['blocks_per_round'] == BLOCKS and info['timing_windows'] == WINDOWS and
            info['schedule'] == routing.schedule(BLOCKS, rounds=ROUNDS) and
            info['budget_fractions'] == {p: routing.definition(p)['budget_fraction'] for p in routing.POLICIES} and
            info['variants'][routing.B]['arrival_method'] == 'tail_assignment' and
            info['communication']['environment'].get('NCCL_NVLS_ENABLE') == '0',
            'Expected a fresh ablation7-new C(1/16)/A/joint-B comparison at 10+50 steps')
    if expected_options is not None:
        require(reporter.builder.planner.frozen_options(info) == arrival_assignment.normalize_options(expected_options),
                'Comparison arrival margin differs from this batch request')
    return info


def check_report(root, info):
    result, gate = read(root/'report.json'), read(root/'preflight.json')
    require(read(root/'all-state.json')['status'] == 'completed' and result['completed'] and
            result['correctness_passed'] and result['protocol'] == routing.PROTOCOL and
            result['run_id'] == info['run_id'] and result['execution_timing'] == TIMING and
            result['round_count'] == ROUNDS and result['blocks_per_round'] == BLOCKS and
            result['timing_windows'] == WINDOWS and len(result['all_samples']) == WINDOWS and
            result['formal_optimizer_steps'] == WINDOWS*TIMING['timed_steps'] and
            result['timing_rank_records'] == WINDOWS*8 and result['preflight_rank_records'] == 48 and
            result['profile_traces'] == 24 and result['candidate_shape_mappings'] == 6,
            'Incomplete ablation7-new strict report; next comparison will not start')
    require(gate['completed'] and gate['protocol'] == routing.PROTOCOL and gate['execution_timing'] == TIMING and
            gate['submission_sha256'] == sha(root/'submission.json') and
            result['evidence']['preflight_sha256'] == sha(root/'preflight.json') and
            result['communication'] == gate['communication'] == info['communication'],
            'Report/preflight identity mismatch')
    options = reporter.builder.planner.frozen_options(info)
    require(reporter.builder.planner.frozen_options(result) ==
            reporter.builder.planner.frozen_options(gate) == options,
            'Report/preflight arrival margin differs from the frozen submission')
    require(result['arrival_schedule'] == order_summary(root), 'Report arrival order differs from the frozen plan')
    return result


def resolve_nccl(args):
    if args.ablation7_build is not None:
        return
    if args.nccl_root is None:
        recorded = read(args.build/'fused/manifest.json').get('nccl_files', {})
        headers = [Path(path) for path in recorded if path.endswith('/include/nccl.h')]
        require(len(headers) == 1, 'Pass --nccl-root: no unique recorded NCCL development directory')
        require(all(Path(path).is_file() and sha(path) == value for path, value in recorded.items()),
                'Recorded NCCL files changed')
        args.nccl_root = headers[0].parent.parent
    require(all((args.nccl_root/path).is_file() for path in ('include/nccl.h', 'lib/libnccl_static.a')),
            'NCCL headers/static library missing')


def supervise(args):
    state_path = args.out/'batch-state.json'
    options = requested_options(args)
    trials = trial_names(args)
    total_windows = WINDOWS * len(trials)
    state = dict(status='preparing', protocol=routing.PROTOCOL, order=list(trials), trial_count=len(trials),
                 arrival_method=arrival_assignment.CONFIGURED_METHOD, arrival_options=options,
                 exploratory=options['uncertainty_multiplier'] < 4, execution_timing=TIMING, rounds=ROUNDS,
                 blocks_per_round=BLOCKS, nvls_enable='0', expected_timing_windows=total_windows,
                 expected_formal_optimizer_steps=total_windows*TIMING['timed_steps'], comparisons={})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    inputs, reports, run_ids = {}, [], set()
    try:
        resolve_nccl(args)
        if args.ablation7_build is not None:
            check_build_options(args.ablation7_build, options)
        state['nccl_root'] = str(args.nccl_root) if args.nccl_root else None
        for label, command in commands(args):
            state.update(status='running', stage=label)
            write(state_path, state)
            launcher.invoke(command, args.out/(label+'.log'), env)
            if label.endswith('-dry-run'):
                trial = label.removesuffix('-dry-run')
                info = check_inputs(args.out/trial, options)
                require(info['run_id'] not in run_ids, 'All requested comparisons require distinct run_ids')
                run_ids.add(info['run_id'])
                summary = order_summary(args.out/trial)
                if inputs:
                    repeat.same_inputs(inputs[trials[0]], info)
                    require(inputs[trials[0]]['budget_fractions'] == info['budget_fractions'] and
                            state['arrival_schedule'] == summary, 'Repeat budget/arrival identity changed')
                else:
                    state['arrival_schedule'] = summary
                    state['identity_fallback'] = all(record['identical_to_A'] for record in summary.values())
                    if state['identity_fallback']:
                        print('Joint B equals A in both roles. Continuing all requested comparisons; '
                              'these runs do not test a nonidentity schedule.', flush=True)
                inputs[trial] = info
                state['comparisons'][trial] = dict(run_id=info['run_id'], status='prepared')
            if label.endswith('-report'):
                trial = label.removesuffix('-report')
                result = check_report(args.out/trial, inputs[trial])
                if reports:
                    for key in ('measurement_identity', 'role_plan_sha256', 'arrival_schedule'):
                        require(reports[0][key] == result[key], 'Repeat runtime/plan changed: '+key)
                reports.append(result)
                state['comparisons'][trial].update(status='completed', timing_windows=WINDOWS,
                    formal_optimizer_steps=WINDOWS*TIMING['timed_steps'], report=str(args.out/trial/'report.json'),
                    report_sha256=sha(args.out/trial/'report.json'))
            write(state_path, state)
        require(len(reports) == len(trials), 'All requested strict reports are required')
        state.update(status='completed', stage='done', timing_windows=total_windows,
                     formal_optimizer_steps=total_windows*TIMING['timed_steps'],
                     timing_rank_records=total_windows*8)
        write(state_path, state)
        print(f'Completed: ablation7-new x{len(trials)}; {total_windows} formal windows / '
              f'{total_windows*8} rank records / {total_windows*TIMING["timed_steps"]} formal steps.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(state_path, state)
        raise


def main(argv=None):
    args = arguments(argv)
    trials = trial_names(args)
    total_windows = WINDOWS * len(trials)
    print('Batch:', args.out, flush=True)
    print('Order: '+' -> '.join(trials), flush=True)
    print('C: legacy arrival + 1/16 mask; A: interleaved/full remote; B: joint tail_assignment/full remote.', flush=True)
    options = requested_options(args)
    print(f'Arrival margin: {options["uncertainty_multiplier"]:g} epsilon in both training passes; '
          f'exploratory={options["uncertainty_multiplier"] < 4}; displacement=8.', flush=True)
    print('Each: 6 blocks x 2 rounds, 10 warmup + 50 timed steps/window, NVLS=0.', flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, fork, compilation, subprocesses or GPU queries.')
        for _, command in commands(args):
            print(shlex.join(command))
        print(f'Total: {len(trials)} fresh preflights/reports, {total_windows} formal windows / '
              f'{total_windows*8} rank records / {total_windows*TIMING["timed_steps"]} formal steps.')
        print('Identity fallback is recorded and still runs all requested comparisons.')
        return
    require(args.python.is_file() and (args.build/'ablation4-manifest.json').is_file(),
            'Training Python / original ablation4 parent build missing')
    if args.ablation7_build is None:
        require((args.build_cuda_home/'bin/nvcc').is_file(), 'Build CUDA Toolkit missing')
    else:
        record = check_build_options(args.ablation7_build, options)
        require(record['protocol'] == routing.PROTOCOL and not record['source_only'] and
                Path(record['parent_build']).resolve() == args.build,
                'Reused build must be executable ablation7-new from the same parent; old ablation7 is incompatible')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(status='launching', protocol=routing.PROTOCOL,
          order=list(trials), trial_count=len(trials),
          execution_timing=TIMING, arrival_options=options, exploratory=options['uncertainty_multiplier'] < 4,
          expected_timing_windows=total_windows,
          expected_formal_optimizer_steps=total_windows*TIMING['timed_steps']))
    write(args.out/'handoff/launch.json', {key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items()})
    try:
        if args.foreground:
            supervise(args)
        else:
            print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
            print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.', flush=True)
    except Exception as error:
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json', state)
        raise


if __name__ == '__main__':
    main()
