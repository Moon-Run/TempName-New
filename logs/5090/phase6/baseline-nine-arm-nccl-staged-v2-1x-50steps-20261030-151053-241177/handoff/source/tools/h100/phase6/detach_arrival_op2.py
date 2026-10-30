#!/usr/bin/env python3
"""Calibrate pure producer order with A's codec, then freeze and confirm twice."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import ablation4_repeat as repeat
import arrival_op2_build as builder
import arrival_op2_protocol as routing
import arrival_op2_report as reporter
import arrival_op2_probe as probe
import arrival_pair_order as order
import detach_ablation5_run_two as launcher
from common import REPO, read, require, sha, write

ENTRY = Path(__file__).with_name('arrival_op2.py')
BLOCKS, ROUNDS, WINDOWS = 6, 2, 24


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=repeat.DEFAULT_BUILD,
                   help='Complete original ablation4 parent used by ablation7 A')
    p.add_argument('--op2-build', type=Path, help='Reuse this exact compiled order-only v2 build')
    p.add_argument('--selection', type=Path, help='Reuse a completed same-codec probe selection for this build')
    p.add_argument('--out', type=Path)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--nccl-root', type=Path)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--trials', type=int, choices=(2, 4), default=2)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true')
    args = p.parse_args(argv)
    require(args.jobs > 0, 'Positive build jobs required')
    require(args.selection is None or args.op2_build is not None,
            'Reusing a selection requires its --op2-build, not a freshly compiled library')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/f'arrival-op2-order-{args.trials}x-50steps-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


def trial_names(args):
    return tuple(f'compare{i+1}' for i in range(args.trials))


def selection_path(args):
    return args.selection or args.out/'probe/selection.json'


def commands(args):
    cli = [str(args.python), str(ENTRY)]
    root = args.op2_build or args.out/'build'
    tasks = []
    if args.op2_build is None:
        flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
        if args.nccl_root:
            flags += ['--nccl-root', str(args.nccl_root)]
        tasks += [('build-prepare', cli+['build-prepare', str(root), '--parent-build', str(args.build)]),
                  ('compile', cli+['compile', str(root), *flags])]
    if args.selection is None:
        run = cli+['probe-run', str(args.out/'probe'), '--python', str(args.python),
                   '--cuda-home', str(args.cuda_home)]
        tasks += [('probe-prepare', cli+['probe-prepare', str(args.out/'probe'), '--build', str(root)]),
                  ('probe-dry-run', run), ('probe-controller', run+['--execute'])]
    for trial in trial_names(args):
        dest = str(args.out/trial)
        run = cli+['run', dest, '--python', str(args.python), '--cuda-home', str(args.cuda_home)]
        tasks += [(trial+'-prepare', cli+['prepare', dest, '--build', str(root),
                   '--selection', str(selection_path(args)), '--megatron', str(args.megatron),
                   '--blocks', str(BLOCKS), '--rounds', str(ROUNDS), '--nvls-enable', '0']),
                  (trial+'-dry-run', run),
                  (trial+'-controller', run+['--stage', 'all', '--execute']),
                  (trial+'-report', cli+['report', dest])]
    return tasks


def check_report(root, info):
    result, gate = read(root/'report.json'), read(root/'preflight.json')
    require(read(root/'all-state.json')['status'] == 'completed' and
            result['completed'] is True and result['correctness_passed'] is True and
            result['protocol'] == routing.PROTOCOL and result['run_id'] == info['run_id'] and
            result['execution_timing'] == routing.TIMING and result['round_count'] == ROUNDS and
            result['blocks_per_round'] == BLOCKS and result['timing_windows'] == WINDOWS and
            len(result['all_samples']) == WINDOWS and result['formal_optimizer_steps'] == WINDOWS*50 and
            result['timing_rank_records'] == WINDOWS*8 and result['preflight_rank_records'] == 32 and
            result['profile_traces'] == 16 and result['candidate_shape_mappings'] == 4,
            'Incomplete order-only comparison; stop before next comparison')
    require(gate['completed'] is True and gate['protocol'] == routing.PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and
            result['evidence']['preflight_sha256'] == sha(root/'preflight.json') and
            result['communication'] == gate['communication'] == info['communication'] and
            result['schedule_options'] == gate['schedule_options'] == info['schedule_options'] and
            result['selection_sha256'] == gate['selection_sha256'] == info['selection_sha256'] and
            gate['execution_timing'] == routing.TIMING and
            result['role_plan_sha256'] == read(root/'scripts/config.json')['role_plan_sha256'],
            'Order-only report is not tied to its frozen selection/preflight')
    nonidentity = order.validate_spans(result['schedule_options']) != routing.IDENTITY_SPANS
    require(result['primary_comparisons'] == [list(pair) for pair in routing.PRIMARY] and
            result['acceptance_threshold'] == routing.ACCEPTANCE_PERCENT and
            result['acceptance'] == reporter.acceptance(result['rounds']['combined'], nonidentity=nonidentity) and
            type(result['acceptance']['passed']) is bool,
            'Cached acceptance differs from predetermined nonidentity S/A statistics')
    return result


def resolve_nccl(args):
    if args.op2_build is not None:
        return
    if args.nccl_root is None:
        files = read(args.build/'fused/manifest.json').get('nccl_files', {})
        headers = [Path(p) for p in files if p.endswith('/include/nccl.h')]
        require(len(headers) == 1 and all(Path(p).is_file() and sha(p) == h for p, h in files.items()),
                'Pass --nccl-root; parent NCCL development files missing or changed')
        args.nccl_root = headers[0].parent.parent
    require(all((args.nccl_root/p).is_file() for p in ('include/nccl.h', 'lib/libnccl_static.a')),
            'NCCL headers/static library missing')


def supervise(args):
    state_path = args.out/'batch-state.json'
    trials = trial_names(args)
    state = dict(status='preparing', protocol=routing.PROTOCOL, order=list(trials), trial_count=len(trials),
                 primary_comparison=[routing.S, routing.A], threshold_percent=1.5,
                 execution_timing=routing.TIMING, expected_timing_windows=WINDOWS*len(trials),
                 expected_formal_optimizer_steps=50*WINDOWS*len(trials), comparisons={})
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    inputs, reports, selection = {}, [], None
    root = args.op2_build or args.out/'build'
    try:
        resolve_nccl(args)
        if args.op2_build is not None:
            record = builder.verify(args.op2_build, built=True)
            require(Path(record['parent_build']) == args.build, 'Reused order-only build has another parent')
        for label, command in commands(args):
            if label.startswith('compare') and selection is None:
                selection = probe.selection_verify(selection_path(args), root)
                require(selection['gpu_measured'] is True and not selection['source_only'],
                        'Real same-codec calibration must precede full-step confirmation')
                spans = order.validate_spans(selection['selected_spans'])
                state.update(selected_spans=spans, selection_sha256=sha(selection_path(args)),
                             probe_run_id=selection['run_id'])
                if spans == routing.IDENTITY_SPANS:
                    state.update(status='completed_no_candidate', stage='done', target_validated=False,
                                 acceptance_status='not_validated_no_nonidentity_candidate',
                                 timing_windows=0, formal_optimizer_steps=0, timing_rank_records=0)
                    write(state_path, state)
                    print('All candidates rejected by same-codec screening; retain A. No confirmation claim.', flush=True)
                    return state
            state.update(status='running', stage=label)
            write(state_path, state)
            launcher.invoke(command, args.out/(label+'.log'), env)
            if label.startswith('compare') and label.endswith('-dry-run'):
                trial = label.removesuffix('-dry-run')
                info = reporter.verify_inputs(args.out/trial)
                require(info['executable'] and not info['source_only'] and
                        info['schedule'] == routing.schedule(BLOCKS, rounds=ROUNDS) and
                        info['timing_windows'] == WINDOWS and info['execution_timing'] == routing.TIMING and
                        info['schedule_options'] == state['selected_spans'] and
                        info['selection_sha256'] == state['selection_sha256'],
                        'Confirmation must retain the same frozen selected order')
                require(info['run_id'] not in {old['run_id'] for old in inputs.values()} | {selection['run_id']},
                        'Calibration and all confirmations require distinct run_ids')
                if inputs:
                    repeat.same_inputs(next(iter(inputs.values())), info)
                inputs[trial] = info
                state['comparisons'][trial] = dict(status='prepared', run_id=info['run_id'])
            elif label.startswith('compare') and label.endswith('-report'):
                trial = label.removesuffix('-report')
                result = check_report(args.out/trial, inputs[trial])
                if reports:
                    require(result['measurement_identity'] == reports[0]['measurement_identity'] and
                            result['role_plan_sha256'] == reports[0]['role_plan_sha256'],
                            'Measurement identity changed across confirmations')
                reports.append(result)
                state['comparisons'][trial].update(status='completed', acceptance=result['acceptance'],
                                                   report=str(args.out/trial/'report.json'))
            write(state_path, state)
        passed = len(reports) == len(trials) and all(r['acceptance']['passed'] is True for r in reports)
        state.update(status='completed', stage='done', timing_windows=WINDOWS*len(trials),
                     timing_rank_records=8*WINDOWS*len(trials), formal_optimizer_steps=50*WINDOWS*len(trials),
                     target_validated=passed, acceptance_status='validated_at_least_1.5_percent'
                     if passed else 'not_validated_1.5_percent')
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
    print('A: original ablation7 A. S: identical compute/codec/lifecycle, changed producer order only.', flush=True)
    print('Calibrate spans with the same codec; freeze once; confirm S/A >=1.5% full-step time reduction.', flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, compilation, subprocesses or GPU queries.')
        for _, command in commands(args):
            print(shlex.join(command))
        print(f'If a nonidentity candidate qualifies: {args.trials} confirmations; '
              f'{24*args.trials} timing windows; {1200*args.trials} formal steps.')
        return
    require(args.python.is_file() and (args.build/'ablation4-manifest.json').is_file(),
            'Training Python / complete original ablation4 build missing')
    if args.op2_build is None:
        require((args.build_cuda_home/'bin/nvcc').is_file(), 'Build CUDA Toolkit required')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(status='launching', protocol=routing.PROTOCOL))
    write(args.out/'handoff/launch.json',
          {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    try:
        if args.foreground:
            supervise(args)
        else:
            print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
            print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.')
    except Exception as error:
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.out/'batch-state.json', state)
        raise


if __name__ == '__main__':
    main()
