#!/usr/bin/env python3
"""OP4 candidate only: measure the 16-channel operator and complete training path."""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO
import op3
import op3_protocol as p

PROTOCOL = 'rtx5090-phase6-op4-nccl-channels-single-v1'
ARMS = ('candidate',)
OVERRIDES = {'candidate': {'NCCL_MIN_NCHANNELS': '16', 'NCCL_MAX_NCHANNELS': '16'}}
EXTERNAL_OVERRIDES = ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_TUNER_PLUGIN',
    'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS', 'NCCL_MIN_NRINGS', 'NCCL_MAX_NRINGS',
    'NCCL_MIN_CTAS', 'NCCL_MAX_CTAS')
TARGET_MS = 60.0
DEFAULT_REFERENCE = REPO.parent/'Load-5090/logs/5090/phase6/op3-combined-v2-1x-50steps-20261102-100813-883313'
read, write, require, sha = p.read, p.write, p.require, p.sha


def schedule(trials=1, blocks=1):
    require(type(trials) is int and trials in (1, 2)
            and type(blocks) is int and blocks > 0, 'Use 1/2 trials and positive integer blocks')
    return [dict(trial=trial, block=block, arm='candidate',
                 directory=f'compare{trial}/block{block:02d}-candidate')
            for trial in range(1, trials+1) for block in range(blocks)]


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--reference-batch', type=Path, default=DEFAULT_REFERENCE,
                        help='Read-only completed OP3 reference; no reference GPU jobs are launched')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda'))
    build = parser.add_mutually_exclusive_group()
    build.add_argument('--op3-build', type=Path)
    build.add_argument('--abc-build', type=Path)
    parser.add_argument('--trials', type=int, choices=(1, 2), default=1)
    parser.add_argument('--blocks', type=int, default=1, help='Candidate windows per trial; each has 50 formal training steps')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=3600)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--operator-only', action='store_true', help='Stop after candidate operator checks, timings and profiles')
    parser.add_argument('--report', action='store_true', help='Read-only recursive verification of a completed batch')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    schedule(args.trials, args.blocks)
    require(args.jobs > 0 and args.worker_timeout > 300, 'Positive jobs and worker timeout > 300 required')
    require(not args.report or (args.out and not args.execute and not args.operator_only), '--report requires --out without --execute/--operator-only')
    require(not args.inside or (args.execute and args.out and not args.report), 'Invalid supervisor invocation')
    require(not args.execute or args.op3_build or args.abc_build, '--execute requires --op3-build or --abc-build')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO.parent/'Load-5090/logs/5090/phase6'/f'op4-nccl-channels-single-v1-{args.trials}x-50steps-{stamp}'
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().absolute() if name == 'python' else value.expanduser().resolve())
    return args


def check_overrides(environment=None):
    """Reject hidden controls before creating a batch or initializing NCCL."""
    environment = os.environ if environment is None else environment
    for key in EXTERNAL_OVERRIDES:
        require(key not in environment, f'OP4 requires {key} unset in the launching environment')
    paths = [Path('/etc/nccl.conf'), Path.home()/'.nccl.conf']
    if environment.get('NCCL_CONF_FILE'):
        paths.append(Path(environment['NCCL_CONF_FILE']))
    for path in paths:
        if path.is_file():
            for line in path.read_text().splitlines():
                entry = line.split('#', 1)[0].strip()
                key = entry.split('=', 1)[0].strip()
                require(key not in EXTERNAL_OVERRIDES,
                        f'OP4 requires no {key} override in {path}')


def child_environment(arm, environment=None):
    require(arm in ARMS, 'Unknown OP4 arm')
    env = dict(os.environ if environment is None else environment)
    check_overrides(env)
    env.update(OVERRIDES[arm])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def child_command(args, destination, op3_build=None):
    command = [str(args.python), str(HERE/'op3.py'), '--execute', '--foreground',
               '--out', str(destination), '--trials', '1', '--blocks', '1']
    for name in ('python', 'megatron', 'cuda_home', 'build_cuda_home', 'jobs', 'worker_timeout'):
        command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    if op3_build is not None:
        command += ['--op3-build', str(op3_build)]
    else:
        require(args.abc_build is not None, 'Missing initial ABC build')
        command += ['--abc-build', str(args.abc_build)]
    if args.operator_only:
        command.append('--operator-only')
    return command


def run_child(command, env, log):
    with log.open('x') as handle:
        subprocess.run(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def child_result(root, recompute_profiles=True):
    operator_only = read(root/'plan.json')['operator_only']
    expected_state = 'operator_completed' if operator_only else 'completed'
    require(read(root/'batch-state.json')['status'] == expected_state, 'Incomplete OP3 child: '+str(root))
    result = op3.replay(root, recompute_profiles=recompute_profiles)
    report_path = root/('operator/report.json' if operator_only else 'report.json')
    require(result == read(report_path), 'OP4 child report differs from its original evidence')
    if operator_only:
        require(result['completed'] is True and result['arm'] == p.COMBINED
                and result['operator_gradients_validated'] is True and result['full_step_measured'] is False,
                'OP4 requires a complete candidate operator measurement')
    else:
        require(result['trials'] == result['blocks'] == 1 and result['arms'] == [p.COMBINED]
                and result['timing_windows'] == 1 and result['formal_optimizer_steps'] == 50,
                'OP4 requires one complete OP3-combined window')
    return result


def training_identity(root):
    from op3_validation import identity
    return identity([read(root/'results/smoke-op3-combined'/f'rank{rank}.json') for rank in range(8)])


def reference_record(root):
    root = Path(root)
    result = read(root/'report.json')
    require(result['protocol'] == p.PROTOCOL and result['completed'] is True
            and result['arms'] == [p.COMBINED] and result['timing_windows'] == 1
            and result['trials'] == result['blocks'] == 1,
            'Reference must be a completed single-window OP3-combined batch')
    profiles, operator = read(root/'profile-summary.json'), read(root/'operator/report.json')
    files = dict(result['worker_records'])
    files.update(profiles['evidence_files'])
    files.update({'operator/'+name: digest for name, digest in operator['evidence_files'].items()})
    files.update({'plan.json': result['plan_sha256'],
                  'profile-summary.json': result['profile_summary_sha256'],
                  'operator/report.json': result['operator_report_sha256']})
    require(all(sha(root/name) == digest for name, digest in files.items()),
            'Historical reference evidence changed')
    values = [sample['ms_per_step'] for sample in result['all_samples']]
    require(len(values) == 1 and all(math.isfinite(value) and value > 0 for value in values),
            'Invalid historical reference timing')
    observed = mean(values)
    require(result['comparisons']['compare1']['arms'][p.COMBINED]['mean_ms'] == observed,
            'Historical reference timing summary changed')
    return dict(root=str(root), report_sha256=sha(root/'report.json'),
        operator_report_sha256=sha(root/'operator/report.json'), evidence_sha256=p.digest(files),
        historical_mean_ms=observed, read_only=True, reference_gpu_jobs=0)


def report(samples, trials, blocks, operator_batches=None, historical_reference_ms=None):
    operator_only = operator_batches is not None
    expected = schedule(trials, blocks)
    observed = operator_batches if operator_only else samples
    require(len(observed) == len(expected) and all(all(row.get(k) == v for k, v in job.items())
            for row, job in zip(observed, expected)), 'Missing/reordered/foreign OP4 candidate windows')
    require(not operator_only or samples == [], 'Operator-only report cannot include training windows')
    require(historical_reference_ms is not None and math.isfinite(historical_reference_ms)
            and historical_reference_ms > 0, 'Historical reference must be supplied from recorded evidence')
    comparisons = {}
    if not operator_only:
        require(all(math.isfinite(row['ms_per_step']) and row['ms_per_step'] > 0 for row in samples),
                'Invalid OP4 timing sample')
        for trial in range(1, trials+1):
            own = [row for row in samples if row['trial'] == trial]
            values = [row['ms_per_step'] for row in own]
            stats = dict(mean_ms=mean(values), median_ms=median(values), min_ms=min(values),
                         max_ms=max(values), tokens_per_second=p.TOKENS_PER_STEP*1000/mean(values))
            comparisons[f'compare{trial}'] = dict(arms={'candidate': stats},
                historical_delta_ms=historical_reference_ms-stats['mean_ms'])
    return dict(protocol=PROTOCOL, completed=True, arms=list(ARMS), trials=trials, blocks=blocks,
        measured_configuration=p.COMBINED, nccl_overrides=OVERRIDES, target_ms=TARGET_MS,
        timing_windows=len(samples), formal_optimizer_steps=len(samples)*50,
        timing_rank_records=len(samples)*8, all_windows_retained=True,
        operator_only=operator_only, full_step_completed=not operator_only,
        concurrent_control=False, counterbalanced_order=False, pairwise_ci_available=False,
        comparisons=comparisons, all_samples=samples, operator_batches=operator_batches or [],
        historical_reference_ms=historical_reference_ms, historical_comparison_is_causal=False,
        target_validated=False, correctness_passed=True, execution_path_validated=True,
        limitation='Only the OP4 candidate is measured. No concurrent control, paired gain, '
            'bootstrap confidence interval or validated 60 ms improvement is available. '
            'The read-only reference is historical and any difference is observational, '
            'not a causal performance gain. Operator timing does not measure a complete training step.')


def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete OP4 report')
    lines = ['# RTX 5090 OP4：NCCL 16 通道候选组', '',
        '仅测试 OP4 候选：原 OP3 组合路径，设置 `NCCL_MIN_NCHANNELS=16` 和 `NCCL_MAX_NCHANNELS=16`。', '']
    if result['operator_only']:
        lines += ['仅算子检查和计时已完成，未执行完整模型训练。', '',
            '| trial / block | 角色 | GPU 平均 μs | 墙钟平均 μs |',
            '| --- | --- | ---: | ---: |']
        for batch in result['operator_batches']:
            for role, values in batch['roles'].items():
                lines.append(f"| {batch['trial']} / {batch['block']} | {role} | "
                    f"{values['gpu_us']['mean']:.3f} | {values['wall_us']['mean']:.3f} |")
    else:
        lines += [f"{result['trials']} trial × {result['blocks']} 个候选窗口，共 "
            f"{result['formal_optimizer_steps']} 个正式 optimizer step；每窗口 10 步预热＋50 步正式计时。", '',
            '| trial | 平均 ms/step | 中位数 | 最快 | 最慢 | tokens/s | 历史参考差值 ms |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for trial, record in result['comparisons'].items():
            v = record['arms']['candidate']
            lines.append(f"| {trial} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | "
                f"{v['min_ms']:.3f} | {v['max_ms']:.3f} | {v['tokens_per_second']:.2f} | "
                f"{record['historical_delta_ms']:.3f} |")
        lines += ['', f"历史参考为 {result['historical_reference_ms']:.9f} ms；表中差值仅供观察，不是同期净收益。"]
    lines += ['', '各子批保留完整算子报告、原始窗口、显存和 trace；执行路径审计见 `candidate-evidence.json`。',
              '没有同期对照，`target_validated=false`；completed 不代表验证了 60ms 提升。', '', result['limitation'], '']
    return '\n'.join(lines)


def replay(root, recompute_profiles=True):
    from op4_evidence import audit_candidate
    root = Path(root)
    op3.base.verify_controllers(root)
    plan = read(root/'plan.json')
    require(plan['protocol'] == PROTOCOL and plan['nccl_overrides'] == OVERRIDES
            and plan['target_ms'] == TARGET_MS and plan['model'] == p.MODEL
            and type(plan['operator_only']) is bool
            and plan['schedule'] == schedule(plan['trials'], plan['blocks']), 'Wrong OP4 plan')
    reference_root = Path(plan['reference']['root'])
    require(reference_record(reference_root) == plan['reference'], 'Historical reference identity changed')
    samples, operators, records, evidence = [], [], {}, {}
    initial_build = initial_identity = initial_environment = None
    for job in plan['schedule']:
        child = root/job['directory']
        launch_path = child.with_name(child.name+'-launch.json')
        launch = read(launch_path)
        require(launch['job'] == job and launch['nccl_overrides'] == OVERRIDES['candidate'], 'Changed OP4 child launch')
        child_plan = read(child/'plan.json')
        require(child_plan['operator_only'] is plan['operator_only'], 'OP4 child mode changed')
        result = child_result(child, recompute_profiles)
        build = child_plan['build']
        require(initial_build is None or build == initial_build, 'OP4 children used different codec builds')
        initial_build = build
        environment = {key: child_plan[key] for key in ('model', 'environment', 'communication', 'cuda_visible_devices')}
        require(initial_environment is None or environment == initial_environment,
                'OP4 candidate environment changed between windows')
        initial_environment = environment
        if plan['operator_only']:
            operators.append(dict(job, roles=result['roles']))
        else:
            current = training_identity(child)
            require(initial_identity is None or current == initial_identity,
                    'OP4 candidate hardware/software/weights/RNG/input changed between blocks')
            initial_identity = current
            samples.append(dict(result['all_samples'][0], **job))
        report_name = 'operator/report.json' if plan['operator_only'] else 'report.json'
        records[job['directory']+'/'+report_name] = sha(child/report_name)
        records[job['directory']+'-launch.json'] = sha(launch_path)
        evidence[job['directory']] = audit_candidate(child, reference_root)
        require(evidence[job['directory']].get('passed') is True, 'Candidate audit did not pass')
    require(evidence == read(root/'candidate-evidence.json'), 'OP4 candidate evidence differs from raw child evidence')
    result = report(samples, plan['trials'], plan['blocks'], operators if plan['operator_only'] else None,
                    historical_reference_ms=plan['reference']['historical_mean_ms'])
    result.update(run_id=plan['run_id'], plan_sha256=sha(root/'plan.json'), child_reports=records,
                  candidate_evidence_sha256=sha(root/'candidate-evidence.json'), build=initial_build, reference=plan['reference'])
    return result


def supervise(args):
    from op4_evidence import audit_candidate
    root, lock = args.out, None
    state = read(root/'batch-state.json')
    try:
        check_overrides()
        lock_path = REPO/'logs/5090/.op4.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        op3.base.verify_controllers(root)
        reference = reference_record(args.reference_batch)
        plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex, trials=args.trials, blocks=args.blocks,
            model=p.MODEL, target_ms=TARGET_MS, nccl_overrides=OVERRIDES,
            operator_only=args.operator_only, reference=reference, schedule=schedule(args.trials, args.blocks))
        write(root/'plan.json', plan)
        build, evidence = args.op3_build, {}
        for job in plan['schedule']:
            op3.base.verify_controllers(root)
            require(reference_record(args.reference_batch) == reference, 'Historical reference changed during execution')
            child = root/job['directory']
            child.parent.mkdir(parents=True, exist_ok=True)
            command = child_command(args, child, build)
            write(child.with_name(child.name+'-launch.json'),
                  dict(job=job, command=command, nccl_overrides=OVERRIDES['candidate']))
            state.update(status='running', stage=job['directory'])
            write(root/'batch-state.json', state)
            print('STAGE', job['directory'], flush=True)
            run_child(command, child_environment('candidate'), child.with_name(child.name+'-supervisor.log'))
            child_result(child, recompute_profiles=False)
            actual_build = Path(read(child/'plan.json')['build']['root'])
            require(build is None or actual_build == build, 'OP3 child changed its requested build')
            build = actual_build
            evidence[job['directory']] = audit_candidate(child, args.reference_batch)
            require(evidence[job['directory']].get('passed') is True, 'Candidate audit did not pass')
            write(root/'candidate-evidence.json', evidence)
        state.update(stage='replay-report')
        write(root/'batch-state.json', state)
        result = replay(root, recompute_profiles=False)
        write(root/'report.json', result)
        (root/'report.md').write_text(markdown(result))
        state.update(status='operator_completed' if args.operator_only else 'completed', stage='done',
                     timing_windows=result['timing_windows'], formal_optimizer_steps=result['formal_optimizer_steps'],
                     full_step_completed=result['full_step_completed'], target_validated=False,
                     report=str(root/'report.md'))
        write(root/'batch-state.json', state)
        print('COMPLETED', root/'report.md', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write(root/'batch-state.json', state)
        raise
    finally:
        if lock is not None:
            lock.close()


def supervisor_command(args):
    command = [str(args.python), str(Path(__file__).resolve()), '--execute', '--inside', '--out', str(args.out)]
    for name in ('python', 'megatron', 'cuda_home', 'build_cuda_home', 'trials', 'blocks', 'jobs', 'worker_timeout', 'reference_batch'):
        command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    for name in ('op3_build', 'abc_build'):
        if getattr(args, name) is not None:
            command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    if args.operator_only:
        command.append('--operator-only')
    return command


def main(argv=None):
    args = arguments(argv)
    if args.report:
        require(read(args.out/'batch-state.json')['status'] in ('completed', 'operator_completed'), 'OP4 batch is incomplete')
        result = replay(args.out)
        require(result == read(args.out/'report.json'), 'Saved OP4 report differs from original evidence')
        print(markdown(result))
        return
    if not args.execute:
        print('PLAN ONLY: no files, builds, GPU queries or training. Add --execute to run.')
        print(json.dumps(dict(protocol=PROTOCOL, out=str(args.out), arms=list(ARMS),
            measured_configuration=p.COMBINED, nccl_overrides=OVERRIDES, target_ms=TARGET_MS,
            reference_batch=str(args.reference_batch), reference_gpu_jobs=0,
            trials=args.trials, blocks=args.blocks, warmup_steps=10, timed_steps=50,
            operator_only=args.operator_only,
            timing_windows=0 if args.operator_only else args.trials*args.blocks,
            formal_optimizer_steps=0 if args.operator_only else 50*args.trials*args.blocks,
            schedule=schedule(args.trials, args.blocks),
            workflow=('Candidate hardware/native -> operator correctness/timing/profile'
                      + ('' if args.operator_only else ' -> model smoke/profile -> full-step timing')
                      + ' -> candidate evidence audit'),
            op3_build=str(args.op3_build) if args.op3_build else None,
            abc_build=str(args.abc_build) if args.abc_build else None), indent=2))
        return
    check_overrides()
    require(args.python.is_file(), 'Python missing: '+str(args.python))
    if args.python.absolute() != Path(sys.executable).absolute():
        os.execv(str(args.python), [str(args.python), str(Path(__file__).resolve()),
                                  *(sys.argv[1:] if argv is None else argv)])
    if args.inside:
        return supervise(args)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(protocol=PROTOCOL, status='launching', stage='freeze-controllers'))
    try:
        files = op3.base.controller_files()
        for name in files:
            dest = args.out/'handoff/source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, dest)
        write(args.out/'handoff/controllers.json', files)
        write(args.out/'handoff/launch.json', {key: str(value) if isinstance(value, Path) else value
                                             for key, value in vars(args).items()})
        print('Batch:', args.out, flush=True)
        if args.foreground:
            return supervise(args)
        command = supervisor_command(args)
        with (args.out/'handoff/supervisor.log').open('x') as log:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write(args.out/'handoff/supervisor.json', dict(pid=process.pid, command=command))
        print(f'Started supervisor PID {process.pid}; inspect batch-state.json and handoff/supervisor.log.', flush=True)
    except BaseException as exc:
        state = read(args.out/'batch-state.json')
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write(args.out/'batch-state.json', state)
        raise


if __name__ == '__main__':
    main()
