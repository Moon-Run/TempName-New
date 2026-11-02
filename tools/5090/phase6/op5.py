#!/usr/bin/env python3
"""OP5 v2: OP4 channels, SHM receive CE and P2P LL compatibility for packet exchanges."""
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
import op4
from op5_evidence import INTERVENTION_ENV, OVERRIDES as CANDIDATE_ENV, reference_record
import op5_probe
import op3_protocol as p

PROTOCOL = 'rtx5090-phase6-op5-shm-recv-ce-p2p-ll-single-v2'
ARMS = ('candidate',)
OVERRIDES = {'candidate': CANDIDATE_ENV}
EXTERNAL_OVERRIDES = op4.EXTERNAL_OVERRIDES + tuple(INTERVENTION_ENV)

TARGET_MS = 50.0
DEFAULT_REFERENCE = REPO.parent/'Load-5090/logs/5090/phase6/op4-nccl-channels-single-v1-1x-50steps-20261102-105103-438654'
read, write, require, sha = p.read, p.write, p.require, p.sha


schedule = op4.schedule
child_command = op4.child_command
run_child = op4.run_child
child_result = op4.child_result
training_identity = op4.training_identity


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--reference-batch', type=Path, default=DEFAULT_REFERENCE,
                        help='Read-only completed OP4 parent batch; all historical windows retained')
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
        args.out = REPO.parent/'Load-5090/logs/5090/phase6'/f'op5-shm-recv-ce-p2p-ll-single-v2-{args.trials}x-50steps-{stamp}'
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().absolute() if name == 'python' else value.expanduser().resolve())
    return args


def check_overrides(environment=None):
    """Reject hidden controls before creating a batch or initializing NCCL."""
    environment = os.environ if environment is None else environment
    for key in EXTERNAL_OVERRIDES:
        require(key not in environment, f'OP5 requires {key} unset in the launching environment')
    paths = [Path('/etc/nccl.conf'), Path.home()/'.nccl.conf']
    if environment.get('NCCL_CONF_FILE'):
        paths.append(Path(environment['NCCL_CONF_FILE']))
    for path in paths:
        if path.is_file():
            for line in path.read_text().splitlines():
                entry = line.split('#', 1)[0].strip()
                key = entry.split('=', 1)[0].strip()
                require(key not in EXTERNAL_OVERRIDES,
                        f'OP5 requires no {key} override in {path}')


def child_environment(arm, environment=None):
    require(arm in ARMS, 'Unknown OP5 arm')
    env = dict(os.environ if environment is None else environment)
    check_overrides(env)
    env.update(OVERRIDES[arm])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def report(samples, trials, blocks, operator_batches=None, historical_reference_ms=None):
    operator_only = operator_batches is not None
    expected = schedule(trials, blocks)
    observed = operator_batches if operator_only else samples
    require(len(observed) == len(expected) and all(all(row.get(k) == v for k, v in job.items())
            for row, job in zip(observed, expected)), 'Missing/reordered/foreign OP5 candidate windows')
    require(not operator_only or samples == [], 'Operator-only report cannot include training windows')
    require(historical_reference_ms is not None and math.isfinite(historical_reference_ms)
            and historical_reference_ms > 0, 'Historical reference must be supplied from recorded evidence')
    comparisons = {}
    if not operator_only:
        require(all(math.isfinite(row['ms_per_step']) and row['ms_per_step'] > 0 for row in samples),
                'Invalid OP5 timing sample')
        for trial in range(1, trials+1):
            own = [row for row in samples if row['trial'] == trial]
            values = [row['ms_per_step'] for row in own]
            stats = dict(mean_ms=mean(values), median_ms=median(values), min_ms=min(values),
                         max_ms=max(values), tokens_per_second=p.TOKENS_PER_STEP*1000/mean(values))
            comparisons[f'compare{trial}'] = dict(arms={'candidate': stats},
                historical_delta_ms=historical_reference_ms-stats['mean_ms'])
    return dict(historical_target_ms=historical_reference_ms-TARGET_MS,
        historical_target_observed=(not operator_only and mean(row['ms_per_step'] for row in samples) <= historical_reference_ms-TARGET_MS),
        protocol=PROTOCOL, completed=True, arms=list(ARMS), trials=trials, blocks=blocks,
        measured_configuration=p.COMBINED, nccl_overrides=OVERRIDES, target_ms=TARGET_MS,
        timing_windows=len(samples), formal_optimizer_steps=len(samples)*50,
        timing_rank_records=len(samples)*8, all_windows_retained=True,
        operator_only=operator_only, full_step_completed=not operator_only,
        concurrent_control=False, counterbalanced_order=False, pairwise_ci_available=False,
        comparisons=comparisons, all_samples=samples, operator_batches=operator_batches or [],
        historical_reference_ms=historical_reference_ms, historical_comparison_is_causal=False,
        target_validated=False, correctness_passed=True, execution_path_validated=True,
        historical_full_model_smoke_gradient_hash_equality_required=False,
        limitation='Only the OP5 receive-CE/P2P-LL candidate is measured against the historical OP4 windows. No concurrent control, paired gain, '
            'bootstrap confidence interval or validated 50 ms improvement is available. '
            'The read-only reference is historical and any difference is observational, '
            'not a causal performance gain. Full-model smoke gradient hashes may differ from '
            'the historical reference; hash differences do not measure their numerical magnitude. '
            'HtoD attribution uses temporal containment and bounded byte counts; it does not '
            'prove copy ownership or complete payload coverage. '
            'Operator timing does not measure a complete training step.')


def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete OP5 report')
    lines = ['# RTX 5090 OP5 v2：16 通道＋接收端 CE＋P2P LL 兼容路径', '',
        '仅测试 OP5 候选：保留 OP4 的 16 通道，设置 `NCCL_SHM_USE_CUDA_MEMCPY=1`、`NCCL_SHM_MEMCPY_MODE=2`、`NCCL_SHM_LOCALITY=2`、`NCCL_P2P_LL_THRESHOLD=2097152`。', '']
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
        lines += ['', f"历史参考为 {result['historical_reference_ms']:.9f} ms；表中差值仅供观察，不是同期净收益。",
                  f"相对 OP4 历史均值的 50ms 目标线为 {result['historical_target_ms']:.3f} ms；历史达线观察：{result['historical_target_observed']}。",
                  '全模型 smoke 梯度与历史参考的哈希允许不同，差异见 `candidate-evidence.json`；哈希无法表明数值差异大小。']
    lines += ['', '`transport-probe/report.json` 保留真实 packet 大小的 All-to-All 和同尺寸 AllGather 的传输日志、逐位检查、延迟与 profile；仅探针使用 INFO 日志，正式训练仍为 WARN。',
              '各子批保留完整算子报告、原始窗口、显存和 trace；执行路径审计见 `candidate-evidence.json`。',
              '没有同期对照，`target_validated=false`；completed 不代表验证了 50ms 提升。', '', result['limitation'], '']
    return '\n'.join(lines)


def replay(root, recompute_profiles=True):
    from op5_evidence import audit_candidate
    root = Path(root)
    op3.base.verify_controllers(root)
    plan = read(root/'plan.json')
    require(plan['protocol'] == PROTOCOL and plan['nccl_overrides'] == OVERRIDES
            and plan['target_ms'] == TARGET_MS and plan['model'] == p.MODEL
            and type(plan['operator_only']) is bool
            and plan['schedule'] == schedule(plan['trials'], plan['blocks']), 'Wrong OP5 plan')
    reference_root = Path(plan['reference']['root'])
    require(reference_record(reference_root) == plan['reference'], 'Historical reference identity changed')
    probe_root = root/'transport-probe'
    probe = op5_probe.evidence(probe_root)
    require(probe == read(probe_root/'report.json')
            and sha(probe_root/'report.json') == plan['transport_probe_sha256'], 'Transport probe changed')
    samples, operators, records, evidence = [], [], {}, {}
    initial_build = initial_identity = initial_environment = None
    for job in plan['schedule']:
        child = root/job['directory']
        launch_path = child.with_name(child.name+'-launch.json')
        launch = read(launch_path)
        require(launch['job'] == job and launch['nccl_overrides'] == OVERRIDES['candidate'], 'Changed OP5 child launch')
        child_plan = read(child/'plan.json')
        require(child_plan['operator_only'] is plan['operator_only'], 'OP5 child mode changed')
        result = child_result(child, recompute_profiles)
        op5_probe.check_training_identity(probe_root, read(child/'environment-preflight.json'), child_plan['communication'])
        build = child_plan['build']
        require(initial_build is None or build == initial_build, 'OP5 children used different codec builds')
        initial_build = build
        environment = {key: child_plan[key] for key in ('model', 'environment', 'communication', 'cuda_visible_devices')}
        require(initial_environment is None or environment == initial_environment,
                'OP5 candidate environment changed between windows')
        initial_environment = environment
        if plan['operator_only']:
            operators.append(dict(job, roles=result['roles']))
        else:
            current = training_identity(child)
            require(initial_identity is None or current == initial_identity,
                    'OP5 candidate hardware/software/weights/RNG/input changed between blocks')
            initial_identity = current
            samples.append(dict(result['all_samples'][0], **job))
        report_name = 'operator/report.json' if plan['operator_only'] else 'report.json'
        records[job['directory']+'/'+report_name] = sha(child/report_name)
        records[job['directory']+'-launch.json'] = sha(launch_path)
        evidence[job['directory']] = audit_candidate(child, reference_root)
        require(evidence[job['directory']].get('passed') is True, 'Candidate audit did not pass')
    require(evidence == read(root/'candidate-evidence.json'), 'OP5 candidate evidence differs from raw child evidence')
    result = report(samples, plan['trials'], plan['blocks'], operators if plan['operator_only'] else None,
                    historical_reference_ms=plan['reference']['historical_mean_ms'])
    result.update(run_id=plan['run_id'], plan_sha256=sha(root/'plan.json'), child_reports=records,
                  candidate_evidence_sha256=sha(root/'candidate-evidence.json'), build=initial_build, reference=plan['reference'],
                  transport_probe_sha256=plan['transport_probe_sha256'], receive_ce_probe_validated=True,
                  packet_alltoall_probe_validated=True)
    return result


def supervise(args):
    from op5_evidence import audit_candidate
    root, lock = args.out, None
    state = read(root/'batch-state.json')
    try:
        check_overrides()
        lock_path = REPO/'logs/5090/.op5.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        op3.base.verify_controllers(root)
        reference = reference_record(args.reference_batch)
        plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex, trials=args.trials, blocks=args.blocks,
            model=p.MODEL, target_ms=TARGET_MS, nccl_overrides=OVERRIDES,
            operator_only=args.operator_only, reference=reference, schedule=schedule(args.trials, args.blocks))
        write(root/'plan.json', plan)
        state.update(status='running', stage='transport-probe')
        write(root/'batch-state.json', state)
        print('STAGE transport-probe', flush=True)
        command = [str(args.python), str(HERE/'op5_probe.py'), '--out', str(root/'transport-probe')]
        for name in ('python', 'cuda_home', 'build_cuda_home'):
            command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
        env = dict(child_environment('candidate'), RTX5090_OP5_PROBE_EXECUTE='1')
        run_child(command, env, root/'handoff/transport-probe.log')
        require(op5_probe.evidence(root/'transport-probe') == read(root/'transport-probe/report.json'), 'Transport probe failed')
        plan['transport_probe_sha256'] = sha(root/'transport-probe/report.json')
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
            op5_probe.check_training_identity(root/'transport-probe', read(child/'environment-preflight.json'),
                                              read(child/'plan.json')['communication'])
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
        require(read(args.out/'batch-state.json')['status'] in ('completed', 'operator_completed'), 'OP5 batch is incomplete')
        result = replay(args.out)
        require(result == read(args.out/'report.json'), 'Saved OP5 report differs from original evidence')
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
            workflow=('Full-sized packet All-to-All + AllGather CE probe -> candidate hardware/native -> operator correctness/timing/profile'
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
