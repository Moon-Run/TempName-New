#!/usr/bin/env python3
"""MA5 tile/block experiments against frozen CR2; preview unless --execute.

Default: one decoder candidate, one comparison, eight balanced T/CR2 blocks.
--sweep runs four decoder configurations sequentially, with a fresh CR2 control
in every pair. GEMM K-tile and scheduling changes are explicit opt-in inputs.
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
from common import HERE, REPO, read, require, sha, verify_files, write

PROTOCOL = 'h100-ma5-tile-block-detached-v1'
BLOCKS, STEPS = 8, 50
SWEEP = ((128, 528), (256, 1024), (256, 528), (256, 264))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--megatron', type=Path, default=REPO.parent / 'Megatron-LM')
    p.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    p.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    p.add_argument('--nccl-root', type=Path)
    p.add_argument('--ma5-build', type=Path, help='Reuse a verified pair build with matching tuning')
    p.add_argument('--trials', type=int, choices=(1, 2), default=1)
    p.add_argument('--decoder-threads', type=int, choices=(64, 128, 256, 512), default=256)
    p.add_argument('--decoder-blocks', type=int, default=528,
                   help='Positive decoder grid size, at most one block per group tile')
    p.add_argument('--gemm-sms', type=int, choices=(-1, 128, 132), default=-1,
                   help='CUTLASS available-SM scheduling budget; not a literal grid size')
    p.add_argument('--gemm-tile-k', type=int, choices=(32, 64), default=32)
    p.add_argument('--sweep', action='store_true', help='Four decoder pairs; no automatic winner selection')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--worker-timeout', type=int, default=1800)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--foreground', action='store_true')
    args = p.parse_args(argv)
    require(args.jobs > 0 and args.worker_timeout > 0, 'Positive jobs and worker timeout required')
    require(not (args.sweep and args.ma5_build), 'A sweep requires separate fresh builds')
    require(not args.sweep or (args.decoder_threads, args.decoder_blocks) == (256, 528),
            '--sweep owns decoder settings; omit individual decoder overrides')
    from ma5_sources import tuning_config
    for _, tuning in candidates(args):
        tuning_config(tuning)
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO / 'logs/h100/phase6' / f'ma5-tile-block-{args.trials}x-50steps-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().absolute() if key == 'python' else value.expanduser().resolve())
    return args


def candidates(args):
    pairs = SWEEP if args.sweep else ((args.decoder_threads, args.decoder_blocks),)
    return [(f't{threads}-b{blocks}-k{args.gemm_tile_k}-s{args.gemm_sms}',
             dict(decoder_threads=threads, decoder_blocks=blocks,
                  gemm_sms=args.gemm_sms, gemm_tile_k=args.gemm_tile_k))
            for threads, blocks in pairs]


def controller_hashes():
    # Importing the CPU controllers discovers their local Python dependencies;
    # no CUDA context, worker, subprocess or build is started here.
    import ma5_build
    import ma5_bench
    import ma5_fullstep
    paths = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module, '__file__', None) and str(module.__file__).endswith('.py')
             and Path(module.__file__).resolve().is_relative_to(HERE)}
    paths.update(HERE / name for name in ma5_build.INPUTS)
    paths.update((Path(__file__).resolve(), HERE / 'run_ma5_tile_block.sh',
                  REPO / 'tools/a800/phase4/reference.py'))
    paths.update(p for p in (REPO / 'tools/a800/phase5/e2e').iterdir()
                 if p.suffix in ('.py', '.json'))
    return {str(path): sha(path) for path in sorted(paths)}


def commands(args):
    def cli(name):
        return [str(args.python), str(HERE / name)]
    steps = []
    for name, tuning in candidates(args):
        root = args.out / name
        build = args.ma5_build or root / 'build'
        if args.ma5_build is None:
            params = [x for key, value in tuning.items() for x in ('--' + key.replace('_', '-'), str(value))]
            flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
            if args.nccl_root:
                flags += ['--nccl-root', str(args.nccl_root)]
            steps += [(name, 'build-prepare', cli('ma5_build.py') + ['prepare', '--out', str(build), *params]),
                      (name, 'compile', cli('ma5_build.py') + ['build', '--build', str(build), *flags])]
        runtime = ['--python', str(args.python), '--cuda-home', str(args.cuda_home),
                   '--worker-timeout', str(args.worker_timeout)]
        operator, train = root / 'operator', root / 'fullstep'
        steps += [
            (name, 'operator-prepare', cli('ma5_bench.py') + ['prepare', str(operator), '--build', str(build), '--blocks', str(BLOCKS)]),
            (name, 'operator-timing', cli('ma5_bench.py') + ['run', str(operator), *runtime, '--execute']),
            (name, 'operator-profile', cli('ma5_bench.py') + ['run', str(operator), *runtime, '--profile', '--execute']),
            (name, 'fullstep-prepare', cli('ma5_fullstep.py') + ['prepare', str(train), '--build', str(build),
                '--megatron', str(args.megatron), '--trials', str(args.trials), '--blocks', str(BLOCKS)]),
            (name, 'fullstep-run', cli('ma5_fullstep.py') + ['run', str(train), '--stage', 'all', *runtime, '--execute']),
            (name, 'fullstep-profile', cli('ma5_fullstep.py') + ['run', str(train), '--stage', 'profile', *runtime, '--execute']),
        ]
    return steps


def check_operator(root):
    import ma5_bench as bench
    plan = bench.verify(root)
    result, profile = read(root / 'report.json'), read(root / 'profile-summary.json')
    require(result['protocol'] == bench.PROTOCOL and result['run_id'] == plan['run_id']
            and result['completed'] and result['correctness_passed'] and result['bitwise_equal_within_pairs'],
            'Complete T/CR2 operator correctness required')
    require(result == bench.report(root, save=False), 'Operator report differs from raw evidence')
    require(profile['completed'] and profile['execution_path_validated']
            and profile == bench.profile_summary(root), 'Operator launch evidence required')
    return dict(operator_correctness_passed=True, operator_execution_path_validated=True)


def check_fullstep(root, trials, profile=False):
    import ma5_fullstep as fullstep
    plan = fullstep.verify(root)
    require(plan['trials'] == trials and plan['blocks'] == BLOCKS, 'Wrong requested workload')
    if profile:
        result = read(root / 'profile-summary.json')
        require(result['completed'] and result['execution_validated']
                and result == fullstep.profile_summary(root), 'Full-step execution profile required')
    else:
        result = read(root / 'report.json')
        windows = trials * BLOCKS * 2
        require(result['completed'] and result['correctness_passed']
                and result['identical_trajectories_within_pairs']
                and result['full_optimizer_steps_measured']
                and result['equivalence_pairs'] == [['T', 'CR2']]
                and len(result['all_samples']) == windows
                and result['formal_optimizer_steps'] == windows * STEPS
                and result['timing_rank_records'] == windows * 8,
                'Complete paired 10+50-step training evidence required')
        require(result == fullstep.report(root, save=False), 'Full-step report differs from raw evidence')
    verify_files(root, result['evidence_files'])
    return result


def supervise(args):
    state_path = args.out / 'batch-state.json'
    state = read(state_path)
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0', CUDA_DEVICE_MAX_CONNECTIONS='1')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    try:
        frozen_controllers = read(args.out / 'handoff/controllers.json')
        verify_files('/', frozen_controllers)
        if args.ma5_build:
            import ma5_build
            require(ma5_build.verify(args.ma5_build, built=True)['tuning'] == candidates(args)[0][1],
                    'Reused build tuning differs from requested settings')
        for name, label, command in commands(args):
            verify_files('/', frozen_controllers)
            root = args.out / name
            root.mkdir(exist_ok=True)
            state.update(status='running', candidate=name, stage=label)
            write(state_path, state)
            launcher.invoke(command, root / (label + '.log'), env)
            verify_files('/', frozen_controllers)
            record = state['candidates'][name]
            if label == 'operator-profile':
                record.update(check_operator(root / 'operator'))
            elif label == 'fullstep-run':
                result = check_fullstep(root / 'fullstep', args.trials)
                record.update(report=str(root / 'fullstep/report.json'), report_sha256=sha(root / 'fullstep/report.json'),
                    comparisons=result['comparisons'],
                    increment_over_CR2_validated=result['increment_over_CR2_validated'],
                    absolute_500ms_observed=result['absolute_500ms_observed'],
                    target_validated=result['target_validated'])
            elif label == 'fullstep-profile':
                check_fullstep(root / 'fullstep', args.trials, profile=True)
                record.update(status='completed', fullstep_execution_path_validated=True)
            write(state_path, state)
        require(all(x['status'] == 'completed' for x in state['candidates'].values()), 'Incomplete candidate set')
        state.update(status='completed', stage='done',
                     formal_optimizer_steps=state['expected_formal_optimizer_steps'])
        write(state_path, state)
        print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(state_path, state)
        raise


def main(argv=None):
    args = arguments(argv)
    configs = candidates(args)
    windows = len(configs) * args.trials * BLOCKS * 2
    print('Batch:', args.out, flush=True)
    print(f'{len(configs)} candidate(s), {args.trials} trial(s) each, {windows} training windows / {windows * STEPS} formal steps.', flush=True)
    if not args.execute:
        print('PLAN ONLY: no directories, subprocesses, compilation, forks or GPU queries.')
        for _, _, command in commands(args):
            print(shlex.join(command))
        print('Every window: 10 warmup + 50 timed full optimizer steps; all performance outcomes retained.')
        return
    require(args.python.is_file() and args.megatron.is_dir(), 'Training Python / Megatron source missing')
    require((args.cuda_home / 'lib64').is_dir(), 'Runtime CUDA Toolkit missing')
    if args.ma5_build is None:
        require((args.build_cuda_home / 'bin/nvcc').is_file(), 'Build CUDA Toolkit missing')
    if args.nccl_root:
        require(all((args.nccl_root / p).is_file() for p in ('include/nccl.h', 'lib/libnccl_static.a')), 'NCCL development files missing')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'handoff').mkdir()
    write(args.out / 'batch-state.json', dict(protocol=PROTOCOL, status='launching', trials=args.trials,
        expected_timing_windows=windows, expected_formal_optimizer_steps=windows * STEPS,
        sweep_is_exploratory=args.sweep,
        candidates={name: dict(status='pending', tuning=tuning) for name, tuning in configs}))
    write(args.out / 'handoff/launch.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write(args.out / 'handoff/controllers.json', controller_hashes())
    if args.foreground:
        return supervise(args)
    try:
        print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
        print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.', flush=True)
    except BaseException as error:
        state = read(args.out / 'batch-state.json')
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(args.out / 'batch-state.json', state)
        raise


if __name__ == '__main__':
    main()
