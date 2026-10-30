#!/usr/bin/env python3
"""One basic receiver-fusion versus standalone CUDA comparison on H100 TP8.

Default is a read-only preview. --execute builds isolated implementations,
double-forks under container PID 1; each comparison measures 24 windows / 1200
formal steps. Only compare1 is run; no improved arm is built or measured.
Performance ordering is an experimental outcome, never a gate for continuing.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

import basic_quant_protocol as routing
import detach_ablation5_run_two as launcher
from common import HERE, REPO, read, require, sha, write

ENTRY = HERE / 'basic_quant.py'
TRIALS = ('compare1',)
BLOCKS = 6
ROUNDS = 2
WINDOWS_PER_COMPARE = 24
STEPS_PER_COMPARE = 1200


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flux-build', type=Path, help='Reuse a verified native GemmOnly build')
    p.add_argument('--codec-build', type=Path, help='Reuse a verified standalone CUDA codec build')
    p.add_argument('--out', type=Path)
    p.add_argument('--trials', type=int, choices=(1,), default=1,
                   help='Independent comparisons; 1 runs only compare1 (24 windows / 1200 steps)')
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--megatron', type=Path, default=REPO.parent / 'Megatron-LM')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--nccl-root', type=Path)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--worker-timeout', type=int, default=1800)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true')
    args = p.parse_args(argv)
    require(args.jobs > 0, 'jobs must be positive')
    require(args.worker_timeout > 0, 'worker-timeout must be positive')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO / 'logs/h100/phase6' / (f'basic-quant-receiver-{args.trials}x-50steps-' + stamp)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # Keep the selected environment's interpreter spelling intact.
            setattr(args, key, value.expanduser().absolute() if key == 'python'
                    else value.expanduser().resolve())
    return args


def commands(args):
    cli = [str(args.python), str(ENTRY)]
    flux = args.flux_build or args.out / 'flux-build'
    codec = args.codec_build or args.out / 'codec-build'
    compile_args = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
    if args.nccl_root:
        compile_args += ['--nccl-root', str(args.nccl_root)]
    steps = []
    if args.flux_build is None:
        flux_args = list(compile_args)
        steps += [('flux-prepare', cli + ['flux-prepare', str(flux)]),
                  ('flux-compile', cli + ['flux-compile', str(flux), *flux_args])]
    if args.codec_build is None:
        steps += [('codec-prepare', cli + ['codec-prepare', str(codec)]),
                  ('codec-compile', cli + ['codec-compile', str(codec),
                                         '--cuda-home', str(args.build_cuda_home)])]
    for trial in TRIALS[:args.trials]:
        root = str(args.out / trial)
        run = cli + ['run', root, '--python', str(args.python), '--cuda-home', str(args.cuda_home),
                     '--worker-timeout', str(args.worker_timeout)]
        steps += [(trial + '-prepare', cli + ['prepare', root, '--flux-build', str(flux), '--codec-build', str(codec),
                    '--megatron', str(args.megatron), '--blocks', str(BLOCKS),
                    '--rounds', str(ROUNDS), '--nvls-enable', '0']),
                  (trial + '-dry-run', run),
                  (trial + '-controller', run + ['--stage', 'all', '--execute']),
                  (trial + '-report', cli + ['report', root])]
    return steps


def check_inputs(root):
    info = read(root / 'submission.json')
    require(info['protocol'] == routing.PROTOCOL and info['execution_timing'] == routing.TIMING
            and info['executable'] is True and info['source_only'] is False
            and info['rounds'] == ROUNDS and info['blocks_per_round'] == BLOCKS
            and info['timing_windows'] == WINDOWS_PER_COMPARE
            and set(info['variants']) == set(routing.POLICIES)
            and info['schedule'] == routing.schedule(BLOCKS, rounds=ROUNDS)
            and info['communication']['environment'].get('NCCL_NVLS_ENABLE') == '0',
            'Expected a fresh two-arm, both-role, 24-window receiver fusion comparison')
    return info


def same_inputs(first, second):
    require(first['run_id'] != second['run_id'], 'Comparisons must have independent run IDs')
    for key in ('config', 'execution_timing', 'schedule', 'variants', 'communication',
                'calibration_hardware', 'megatron', 'megatron_commit',
                'flux_build_manifest', 'codec_build_manifest', 'codec_identity',
                'codec_library', 'shared_environment', 'hardware', 'controllers'):
        require(first[key] == second[key], 'Repeat input changed: ' + key)
    scripts = lambda x: {k: v for k, v in x['files'].items() if k.startswith('scripts/')}
    require(scripts(first) == scripts(second), 'Frozen training scripts changed between comparisons')


def check_report(root, info):
    result, gate = read(root / 'report.json'), read(root / 'preflight.json')
    require(read(root / 'all-state.json')['status'] == 'completed'
            and result['completed'] is True and result['correctness_passed'] is True
            and result['execution_path_validated'] is True
            and result['bitwise_equal_within_pairs'] is True
            and result['identical_trajectories_within_pairs'] is True
            and result['equivalence_pairs'] == routing.EQUIVALENCE_PAIRS
            and result['protocol'] == routing.PROTOCOL and result['run_id'] == info['run_id']
            and result['execution_timing'] == routing.TIMING
            and result['round_count'] == ROUNDS and result['blocks_per_round'] == BLOCKS
            and result['timing_windows'] == len(result['all_samples']) == WINDOWS_PER_COMPARE
            and result['formal_optimizer_steps'] == STEPS_PER_COMPARE
            and result['timing_rank_records'] == WINDOWS_PER_COMPARE * 8
            and result['preflight_rank_records'] == 32 and result['profile_traces'] == 16
            and result['candidate_shape_mappings'] == 0,
            'Incomplete two-arm fusion comparison; preserve the failed evidence')
    require(gate['completed'] is True and gate['protocol'] == routing.PROTOCOL
            and gate['submission_sha256'] == sha(root / 'submission.json')
            and result['evidence']['preflight_sha256'] == sha(root / 'preflight.json')
            and all(result[k] == gate[k] == info[k]
                    for k in ('communication', 'codec_identity', 'codec_library')),
            'Report/preflight identity mismatch')
    # No performance threshold here: an inverted ordering is a valid result.
    return result


def supervise(args):
    state_path = args.out / 'batch-state.json'
    windows = args.trials * WINDOWS_PER_COMPARE
    formal_steps = args.trials * STEPS_PER_COMPARE
    state = dict(status='preparing', protocol=routing.PROTOCOL,
                 trials=args.trials, expected_timing_windows=windows,
                 expected_formal_optimizer_steps=formal_steps,
                 execution_timing=routing.TIMING, comparisons={},
                 expected_order='basic < cuda (hypothesis, not execution gate)')
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0',
               CUDA_DEVICE_MAX_CONNECTIONS='1')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    inputs, reports = {}, []
    try:
        for label, cmd in commands(args):
            state.update(status='running', stage=label)
            write(state_path, state)
            launcher.invoke(cmd, args.out / (label + '.log'), env)
            if label.endswith('-dry-run'):
                trial = label.removesuffix('-dry-run')
                info = check_inputs(args.out / trial)
                if inputs:
                    same_inputs(next(iter(inputs.values())), info)
                inputs[trial] = info
                state['comparisons'][trial] = dict(status='prepared', run_id=info['run_id'])
            if label.endswith('-report'):
                trial = label.removesuffix('-report')
                result = check_report(args.out / trial, inputs[trial])
                if reports:
                    require(result['measurement_identity'] == reports[0]['measurement_identity'],
                            'Hardware/software identity changed between comparisons')
                reports.append(result)
                state['comparisons'][trial].update(status='completed', timing_windows=24,
                    formal_optimizer_steps=1200, report=str(args.out / trial / 'report.json'),
                    report_sha256=sha(args.out / trial / 'report.json'))
            write(state_path, state)
        require(len(reports) == args.trials, 'All requested complete comparison reports are required')
        state.update(status='completed', stage='done', timing_windows=windows,
                     timing_rank_records=windows * 8, formal_optimizer_steps=formal_steps)
        write(state_path, state)
        print(f'Completed: {args.trials} comparison(s), {windows} windows / '
              f'{formal_steps} formal optimizer steps.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(state_path, state)
        raise


def main(argv=None):
    args = arguments(argv)
    print('Batch:', args.out, flush=True)
    print('Basic receiver-only fusion / standalone CUDA codec; shared GemmOnly + encoder + NCCL; Attention + MLP.', flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, subprocesses, compilation, forks or GPU queries.')
        for _, command in commands(args):
            print(shlex.join(command))
        print(f'{args.trials} comparison(s), {args.trials * WINDOWS_PER_COMPARE} windows / '
              f'{args.trials * STEPS_PER_COMPARE} formal steps; each window 10 warmup + 50 timed steps.')
        print('All performance outcomes are retained; the proposed speed ordering is not guaranteed.')
        return
    require(args.python.is_file() and args.megatron.is_dir(), 'Training Python / Megatron source missing')
    require((args.cuda_home / 'lib64').is_dir(), 'Runtime CUDA Toolkit missing; pass --cuda-home')
    if any(root is None for root in (args.flux_build, args.codec_build)):
        require((args.build_cuda_home / 'bin/nvcc').is_file(), 'Build CUDA Toolkit missing')
    if args.nccl_root:
        require(all((args.nccl_root / n).is_file() for n in ('include/nccl.h', 'lib/libnccl_static.a')),
                'NCCL development files missing')
    for root in (args.flux_build, args.codec_build):
        if root:
            require((root / 'manifest.json').is_file(), 'Requested reused build manifest missing')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'handoff').mkdir()
    write(args.out / 'batch-state.json', dict(status='launching', protocol=routing.PROTOCOL,
          trials=args.trials, expected_timing_windows=args.trials * WINDOWS_PER_COMPARE,
          expected_formal_optimizer_steps=args.trials * STEPS_PER_COMPARE))
    write(args.out / 'handoff/launch.json',
          {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if args.foreground:
        return supervise(args)
    print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
    print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.', flush=True)


if __name__ == '__main__':
    main()
