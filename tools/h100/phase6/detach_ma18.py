#!/usr/bin/env python3
"""Compare complete MA11 and MA11 plus IPC DMA column AllGather in complete, balanced training blocks.

Preview does not create directories or query GPUs. Each trial measures eight
two-arm blocks, with ten warmup and fifty measured optimizer steps per window.
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
from common import HERE, REPO, read, write, require, sha, verify_files

PROTOCOL = 'h100-ma18-ipc-allgather-detached-v1'
ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-ipcag')
BLOCKS = 8
STEPS = 50


def tuning(args):
    from ma18_sources import tuning_config
    return tuning_config()


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--trials', type=int, choices=(1, 2), default=1)
    parser.add_argument('--candidate-only', action='store_true',
                        help='Measure only this candidate; retain full-model smoke/profile, no MA11 comparison')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda-12.8'))
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.4'))
    parser.add_argument('--nccl-root', type=Path)
    parser.add_argument('--ma18-build', type=Path, help='Reuse a verified MA18-v1 two-arm build; older MA8/MA9/MA11 builds are not accepted')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=1800)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args(argv)
    require(args.jobs > 0 and args.worker_timeout > 0, 'Positive jobs and worker timeout required')
    tuning(args)
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO/'logs/h100/phase6'/f'ma18-v1-ipc-allgather-{args.trials}x-50steps-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().absolute() if key == 'python' else value.expanduser().resolve())
    return args


def controller_hashes():
    import candidate_fullstep
    import ma18_build
    import ma18_bench
    import ma18_fullstep
    paths = set()
    for module in tuple(sys.modules.values()):
        # torch.classes/torch.ops expose virtual modules and dynamic attributes.
        # Inspect the module dictionary without invoking __getattr__, and freeze
        # only real files; relative virtual names must not resolve into HERE.
        filename = getattr(module, '__dict__', {}).get('__file__')
        if isinstance(filename, str) and filename.endswith('.py'):
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(HERE):
                paths.add(path)
    paths.update(HERE/name for name in ma18_build.INPUTS)
    # Registered overlays read these templates without importing them as
    # modules. Freeze them before the first subprocess, not only at prepare.
    paths.update((ma18_bench.DRIVER, ma18_bench.TEMPLATE, ma18_bench.base_path,
                  ma18_fullstep._BASE_PATH, ma18_fullstep._F4_PATH))
    paths.update((Path(__file__).resolve(), HERE/'run_ma18.sh', REPO/'tools/a800/phase4/reference.py'))
    paths.update(p for p in (REPO/'tools/a800/phase5/e2e').iterdir() if p.suffix in ('.py', '.json'))
    return {str(path): sha(path) for path in sorted(paths)}


def commands(args):
    cli = lambda name: [str(args.python), str(HERE/name)]
    build = args.ma18_build or args.out/'build'
    tasks = []
    if args.ma18_build is None:
        flags = ['--cuda-home', str(args.build_cuda_home), '--jobs', str(args.jobs)]
        if args.nccl_root:
            flags += ['--nccl-root', str(args.nccl_root)]
        tasks += [('build-prepare', cli('ma18_build.py') + ['prepare', '--out', str(build)]),
                  ('compile', cli('ma18_build.py') + ['build', '--build', str(build), *flags])]
    runtime = ['--python', str(args.python), '--cuda-home', str(args.cuda_home),
               '--worker-timeout', str(args.worker_timeout)]
    operator, train = args.out/'operator', args.out/'fullstep'
    if getattr(args, 'candidate_only', False):
        tasks += [
            ('fullstep-prepare', cli('candidate_fullstep.py') + ['prepare', str(train),
                '--experiment', 'ma18', '--build', str(build), '--megatron', str(args.megatron),
                '--trials', str(args.trials)]),
            ('fullstep-run', cli('candidate_fullstep.py') + ['run', str(train),
                '--stage', 'all', *runtime, '--execute']),
            ('fullstep-profile', cli('candidate_fullstep.py') + ['run', str(train),
                '--stage', 'profile', *runtime, '--execute'])]
        return tasks
    tasks += [
        ('operator-prepare', cli('ma18_bench.py') + ['prepare', str(operator), '--build', str(build), '--blocks', str(BLOCKS)]),
        ('operator-timing', cli('ma18_bench.py') + ['run', str(operator), *runtime, '--execute']),
        ('operator-profile', cli('ma18_bench.py') + ['run', str(operator), *runtime, '--profile', '--execute']),
        ('fullstep-prepare', cli('ma18_fullstep.py') + ['prepare', str(train), '--build', str(build),
            '--megatron', str(args.megatron), '--trials', str(args.trials), '--blocks', str(BLOCKS)]),
        ('fullstep-run', cli('ma18_fullstep.py') + ['run', str(train), '--stage', 'all', *runtime, '--execute']),
        ('fullstep-profile', cli('ma18_fullstep.py') + ['run', str(train), '--stage', 'profile', *runtime, '--execute'])]
    return tasks


def check_operator(root):
    import ma18_bench as bench
    plan = bench.verify(root)
    result, profile = read(root/'report.json'), read(root/'profile-summary.json')
    require(tuple(plan['arms']) == ARMS and result['protocol'] == bench.PROTOCOL
            and result['run_id'] == plan['run_id'] and result['completed']
            and result['correctness_passed'] and result['bitwise_equal_within_pairs']
            and result['consumer_gradients_validated'], 'Incomplete two-arm operator correctness')
    require(result == bench.report(root, save=False), 'Operator report differs from raw evidence')
    require(profile['protocol'] == bench.PROTOCOL and profile['run_id'] == plan['run_id']
            and profile['completed'] and profile['execution_path_validated']
            and profile == bench.profile_summary(root), 'Two-arm operator execution path missing')
    return result


def check_fullstep(root, trials, profile=False):
    import ma18_fullstep as full
    plan, result = full.verify(root), read(root/'report.json')
    windows = trials * BLOCKS * len(ARMS)
    require(tuple(plan['arms']) == ARMS and plan['trials'] == trials and plan['blocks'] == BLOCKS
            and result['protocol'] == full.PROTOCOL and result['run_id'] == plan['run_id']
            and result['completed'] and result['correctness_passed']
            and result['consumer_parameter_gradients_validated'] and result['identical_trajectories_within_pairs']
            and result['activation_setup_validated'] and result['zero_statistics_exact_smoke_validated']
            and result['zero_count_counters_validated'] and result['softmax_setup_validated']
            and result['softmax_exact_smoke_validated'] and result['ipc_allgather_state_validated']
            and result['formal_optimizer_steps'] == windows * STEPS
            and result['timing_rank_records'] == windows * 8 and len(result['all_samples']) == windows
            and set(result['comparisons']) == {f'compare{i+1}' for i in range(trials)},
            'Incomplete requested cache/combined training')
    require(result == full.report(root, save=False), 'Training report differs from raw evidence')
    require(set(result['increment_over_MA11_validated']) == set(ARMS[1:])
            and all(type(v) is bool for v in result['increment_over_MA11_validated'].values()),
            'Missing IPC AllGather performance outcome')
    require(result['target_evaluated'] is True and result['target_percent']==2.0
            and type(result['target_validated']) is bool and type(result['target_point_observed']) is bool,
            'Missing separate 2% target outcome')
    if profile:
        evidence = read(root/'profile-summary.json')
        require(evidence['protocol'] == full.PROTOCOL and evidence['run_id'] == plan['run_id']
                and evidence['completed'] and evidence['execution_validated']
                and evidence == full.profile_summary(root), 'Full-model IPC AllGather execution evidence missing')
    return result


def supervise(args):
    hashes = read(args.out/'handoff/controllers.json')
    candidate_only = getattr(args, 'candidate_only', False)
    if candidate_only:
        from candidate_fullstep import check_complete as check
    else:
        check = check_fullstep
    windows = args.trials * BLOCKS * (1 if candidate_only else len(ARMS))
    state = dict(protocol=PROTOCOL, status='preparing', trials=args.trials, arms=list(ARMS),
        main_contrasts=['F4-cache-zcount-csoftmax-ipcag/F4-cache-zcount-csoftmax'], tuning=tuning(args),
        activation_policy='owned_forward_activation_cache', zero_count_policy='common_fused_optimizer_zero_count',
        softmax_policy='ma11_in_both_arms', column_policy='candidate_ipc_dma_column_allgather',
        fullstep_execution_path_validated=False,
        acceptance_status={'F4-cache-zcount-csoftmax-ipcag': 'pending_execution_evidence'},
        target_evaluated=False, target_validated=False, target_percent=2.0, target_point_observed=False,
        increment_over_MA11_validated={arm: False for arm in ARMS[1:]},
        expected_timing_windows=windows, expected_formal_optimizer_steps=windows * STEPS, reports={})
    state_path = args.out/'batch-state.json'
    state.update(comparison_mode='candidate_only' if candidate_only else 'paired',
                 standalone_operator_executed=not candidate_only)
    if candidate_only:
        state.update(arms=[ARMS[1]], main_contrasts=[], target_percent=None,
                     softmax_policy='original_ma11_in_candidate')
    env = dict(os.environ, PYTHONNOUSERSITE='1', NCCL_NVLS_ENABLE='0')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    try:
        for label, command in commands(args):
            verify_files('/', hashes)
            state.update(status='running', stage=label)
            write(state_path, state)
            launcher.invoke(command, args.out/(label + '.log'), env)
            verify_files('/', hashes)
            if label == 'operator-profile':
                check_operator(args.out/'operator')
                state.update(operator_correctness_validated=True, operator_execution_path_validated=True)
                state['reports']['operator'] = str(args.out/'operator/report.json')
            elif label == 'fullstep-run':
                result = check(args.out/'fullstep', args.trials)
                state.update(timing_report_verified=True, statistical_acceptance_status=result['acceptance_status'])
                state['reports']['fullstep'] = str(args.out/'fullstep/report.json')
            elif label == 'fullstep-profile':
                result = check(args.out/'fullstep', args.trials, profile=True)
                state.update(fullstep_execution_path_validated=True, acceptance_status=result['acceptance_status'],
                    target_evaluated=result['target_evaluated'], target_validated=result['target_validated'],
                    target_point_observed=result['target_point_observed'],
                    increment_over_MA11_validated=result['increment_over_MA11_validated'],
                    formal_optimizer_steps=result['formal_optimizer_steps'], timing_windows=len(result['all_samples']))
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
        print('PLAN ONLY: no directories, GPU queries, builds or workers.')
        for _, command in commands(args):
            print(shlex.join(command))
        if getattr(args, 'candidate_only', False):
            print(f'{ARMS[1]} only; {args.trials*8} windows, {args.trials*400} formal optimizer steps. '
                  '10 warmup + 50 measured steps/window; candidate full-model smoke and profile retained. '
                  'No MA11 control, paired-equivalence claim, speedup CI or 2% target.')
            return
        print(f'F4-cache-zcount-csoftmax / F4-cache-zcount-csoftmax-ipcag only; MA11 in both arms; '
              f'original MA11 softmax in both arms; IPC DMA column AllGather only in candidate; '
              f'{args.trials} trial(s), {args.trials*16} windows, '
              f'{args.trials*800} formal optimizer steps. Each window: 10 warmup + 50 measured steps. '
              'Negative and slow windows are retained. The 2% target is incremental over MA11 and requires '
              'CI95 lower >=2% in each trial plus successful profile checks.')
        return
    require(args.python.is_file() and args.megatron.is_dir(), 'Python/Megatron missing')
    require((args.cuda_home/'lib64').is_dir(), 'Runtime CUDA missing')
    if args.ma18_build is None:
        require((args.build_cuda_home/'bin/nvcc').is_file(), 'Build CUDA missing')
    else:
        import ma18_build
        require(ma18_build.verify(args.ma18_build, built=True)['tuning'] == tuning(args),
                'Reused build differs from fixed MA18-v1 MA11-control configuration')
    if args.nccl_root:
        require(all((args.nccl_root/name).is_file() for name in ('include/nccl.h', 'lib/libnccl_static.a')),
                'NCCL development files missing')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(protocol=PROTOCOL, status='launching', target_validated=False))
    write(args.out/'handoff/launch.json', {key: str(value) if isinstance(value, Path) else value
                                        for key, value in vars(args).items()})
    write(args.out/'handoff/controllers.json', controller_hashes())
    if args.foreground:
        return supervise(args)
    print(json.dumps(launcher.detach(args, supervisor=supervise), ensure_ascii=False), flush=True)
    print('Started, not completed. Follow batch-state.json and handoff/container-supervisor.log.')


if __name__ == '__main__':
    main()
