#!/usr/bin/env python3
"""Paired OP4 versus local-attention CUDA Graph replay on eight RTX 5090 GPUs."""
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
from rtx5090_loader import HERE, REPO, load
from rtx5090_defaults import default_build_jobs
import op3
import op4
import op3_protocol as p
import op6_controllers
from op6_child import PROTOCOL

ARMS = ('op4', 'op6')
CHANNEL_ENV = {'NCCL_MIN_NCHANNELS': '16', 'NCCL_MAX_NCHANNELS': '16'}
EXTERNAL_OVERRIDES = op4.EXTERNAL_OVERRIDES + (
    'NCCL_SHM_USE_CUDA_MEMCPY', 'NCCL_SHM_MEMCPY_MODE', 'NCCL_SHM_LOCALITY',
    'NCCL_P2P_LL_THRESHOLD')
TARGET_MS = 800.0
MEMORY_FIELDS = ('peak_allocated_gib', 'peak_reserved_gib', 'native_workspace_gib',
                 'peak_allocated_plus_taco_workspace_gib')
read, write, require, sha = p.read, p.write, p.require, p.sha


def schedule(trials=1, blocks=2):
    require(type(trials) is int and trials > 0 and type(blocks) is int and blocks > 0,
            'Use positive integer trials and blocks')
    result = []
    for trial in range(1, trials+1):
        for block in range(blocks):
            order = ARMS if (trial-1+block) % 2 == 0 else ARMS[::-1]
            for index, arm in enumerate(order):
                result.append(dict(trial=trial, block=block, index=index, arm=arm,
                    directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--place', type=int, choices=(60, 61),
                        help='Write under Load-5090/SL3060 or SL3061; --out takes precedence')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda'))
    build = parser.add_mutually_exclusive_group()
    build.add_argument('--build', type=Path, help='Reuse an OP3/ABC build; open-compress builds are incompatible')
    build.add_argument('--op3-build', type=Path)
    build.add_argument('--abc-build', type=Path)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--blocks', type=int, default=2,
                        help='Complete paired blocks per trial; each arm has 50 formal steps per block')
    parser.add_argument('--jobs', type=int, default=default_build_jobs(), help='Build workers; default nproc')
    parser.add_argument('--worker-timeout', type=int, default=3600)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--report', action='store_true', help='Read-only verification of a completed --out')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    schedule(args.trials, args.blocks)
    require(args.jobs > 0 and args.worker_timeout > 300, 'Positive jobs and worker timeout >300 required')
    require(not args.report or (args.out and not args.execute), '--report requires --out without --execute')
    require(not args.inside or (args.execute and args.out and not args.report), 'Invalid supervisor invocation')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        output = REPO.parent/'Load-5090'
        output /= f'SL30{args.place}' if args.place is not None else 'logs/5090/phase6'
        args.out = output/f'op6-local-attention-graph-paired-v1-{args.trials}x-{args.blocks}blocks-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().absolute() if key == 'python' else value.expanduser().resolve())
    return args


def resolve_reused_build(args):
    if args.build is None:
        return
    op3_manifest = (args.build/'op3-build.json').is_file()
    abc_manifest = (args.build/'op1-abc-build.json').is_file()
    require(op3_manifest != abc_manifest,
            '--build requires exactly one OP3 or OP1 ABC manifest. An open-compress build '
            'cannot supply the OP4 comparison stack; omit --build for the first OP6 run: '+str(args.build))
    if op3_manifest:
        args.op3_build = args.build
    else:
        args.abc_build = args.build


def check_overrides(environment=None):
    env = os.environ if environment is None else environment
    for key in EXTERNAL_OVERRIDES:
        require(key not in env, 'OP6 requires '+key+' unset in the launching environment')
    paths = [Path('/etc/nccl.conf'), Path.home()/'.nccl.conf']
    if env.get('NCCL_CONF_FILE'):
        paths.append(Path(env['NCCL_CONF_FILE']))
    for path in paths:
        if path.is_file():
            for line in path.read_text().splitlines():
                key = line.split('#', 1)[0].strip().split('=', 1)[0].strip()
                require(key not in EXTERNAL_OVERRIDES, f'OP6 requires no {key} override in {path}')


def child_environment(environment=None):
    env = dict(os.environ if environment is None else environment)
    check_overrides(env)
    env.update(CHANNEL_ENV)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def child_command(args, job, build=None):
    command = [str(args.python), str(HERE/'op6_child.py'), '--variant', job['arm'],
               '--execute', '--foreground', '--out', str(args.out/job['directory']),
               '--trials', '1', '--blocks', '1']
    for key in ('python', 'megatron', 'cuda_home', 'build_cuda_home', 'jobs', 'worker_timeout'):
        command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    if build is not None:
        command += ['--op3-build', str(build)]
    else:
        require(args.abc_build is not None, 'Missing ABC dependency for first OP6 child')
        command += ['--abc-build', str(args.abc_build)]
    return command


run_child = op4.run_child


def prepare_initial_build(args, root, state):
    if args.op3_build or args.abc_build:
        return
    import op1_abc_build as abc_builder
    baseline_builder = load('build')
    # Preserve the GPU/environment check before an expensive first build.
    command = [str(args.python), str(HERE/'op6_child.py'), '--variant', 'op4', '--execute', '--foreground',
               '--preflight-only', '--out', str(root/'initial-preflight')]
    for key in ('python', 'megatron', 'cuda_home', 'build_cuda_home', 'jobs', 'worker_timeout'):
        command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    state.update(status='running', stage='initial-hardware-preflight')
    write(root/'batch-state.json', state)
    run_child(command, child_environment(), root/'initial-preflight-supervisor.log')
    require(read(root/'initial-preflight/batch-state.json')['status'] == 'preflight_completed',
            'Initial hardware preflight did not complete')
    baseline, abc = root/'dependencies/baseline', root/'dependencies/abc'
    for stage, action in (
        ('prepare-baseline-build', lambda: baseline_builder.prepare(baseline)),
        ('compile-baseline-build', lambda: baseline_builder.compile_build(baseline, args.megatron, args.build_cuda_home, args.jobs)),
        ('prepare-abc-build', lambda: abc_builder.prepare(abc, baseline)),
        ('compile-abc-build', lambda: abc_builder.compile_build(abc, args.build_cuda_home, args.jobs)),
    ):
        state.update(status='running', stage=stage)
        write(root/'batch-state.json', state)
        print('STAGE', stage, flush=True)
        action()
    args.abc_build = abc


def child_result(root, variant, recompute_profiles=True):
    import op6_child
    root = Path(root)
    plan = read(root/'plan.json')
    require(plan.get('op6_variant') == variant and plan.get('op6_protocol') == PROTOCOL,
            'Changed OP6 child variant')
    require(read(root/'batch-state.json')['status'] == 'completed', 'Incomplete OP6 child: '+str(root))
    result = op6_child.replay(root, recompute_profiles=recompute_profiles)
    require(result == read(root/'report.json'), 'OP6 child report differs from raw evidence')
    require(result['trials'] == result['blocks'] == result['timing_windows'] == 1
            and result['arms'] == [p.COMBINED] and result['formal_optimizer_steps'] == 50,
            'OP6 requires one complete 50-step OP3-combined window in each child')
    return result


def report(samples, trials, blocks):
    expected = schedule(trials, blocks)
    require(len(samples) == len(expected) and all(all(row.get(k) == value for k, value in job.items())
            for row, job in zip(samples, expected)), 'Missing, reordered or foreign OP6 windows')
    require(all(math.isfinite(row['ms_per_step']) and row['ms_per_step'] > 0 for row in samples),
            'Invalid OP6 timing sample')
    require(all(math.isfinite(row[key]) and row[key] >= 0 for row in samples for key in MEMORY_FIELDS),
            'Invalid OP6 memory sample')
    def stats(rows):
        values = [row['ms_per_step'] for row in rows]
        return dict(mean_ms=mean(values), median_ms=median(values), min_ms=min(values), max_ms=max(values),
                    tokens_per_second=p.TOKENS_PER_STEP*1000/mean(values),
                    **{'max_'+key: max(row[key] for row in rows) for key in MEMORY_FIELDS})
    comparisons = {}
    for trial in range(1, trials+1):
        own = [row for row in samples if row['trial'] == trial]
        arms = {arm: stats([row for row in own if row['arm'] == arm]) for arm in ARMS}
        pairs = []
        for block in range(blocks):
            matched = {row['arm']: row['ms_per_step'] for row in own if row['block'] == block}
            pairs.append(dict(block=block, op4_ms=matched['op4'], op6_ms=matched['op6'],
                              saved_ms=matched['op4']-matched['op6'],
                              reduction_percent=100*(1-matched['op6']/matched['op4'])))
        comparisons[f'compare{trial}'] = dict(arms=arms, pairs=pairs,
            mean_paired_saved_ms=mean(pair['saved_ms'] for pair in pairs),
            faster_blocks=sum(pair['saved_ms'] > 0 for pair in pairs),
            target_800ms_observed=arms['op6']['mean_ms'] <= TARGET_MS)
    aggregate = {arm: stats([row for row in samples if row['arm'] == arm]) for arm in ARMS}
    first = [job['arm'] for job in expected if job['index'] == 0]
    return dict(protocol=PROTOCOL, completed=True, arms=list(ARMS), trials=trials, blocks=blocks,
        nccl_overrides=CHANNEL_ENV, target_ms=TARGET_MS, model=p.MODEL,
        timing_windows=len(samples), timing_rank_records=len(samples)*8,
        formal_optimizer_steps=len(samples)*50, concurrent_control=True,
        control_and_candidate_run_simultaneously=False,
        counterbalanced_order=first.count('op4') == first.count('op6'),
        pairwise_ci_available=False, all_windows_retained=True,
        correctness_passed=True, execution_path_validated=True,
        comparisons=comparisons, aggregate=aggregate, all_samples=samples,
        mean_paired_saved_ms=aggregate['op4']['mean_ms']-aggregate['op6']['mean_ms'],
        target_800ms_observed=aggregate['op6']['mean_ms'] <= TARGET_MS,
        limitation='Both arms are newly measured, sequentially paired by block. No historical timings are pooled. '
          'Small block counts are exploratory; no confidence interval or guaranteed future latency is claimed. '
          'Graph preparation is outside steady-state formal timing; its startup cost and private GPU memory '
          'remain relevant to short-lived jobs. Reusing the OP3 build still compiles per-child MA9/MA11 compatibility libraries.')


def markdown(result):
    lines = ['# RTX 5090 OP6：OP4 与局部 Attention CUDA Graph 同期对比', '',
        f"{result['trials']} trial × {result['blocks']} 完整配对块，"
        f"{result['timing_windows']} 窗口、{result['formal_optimizer_steps']} 正式 optimizer step。",
        '两组均使用 OP4 的 16 通道、OP3 量化与 MA8/MA9/MA11；仅 OP6 回放局部注意力计算图。', '',
        '| trial | OP4 ms/step | OP6 ms/step | 平均节省 ms | 更快块数 | OP6 均值≤800ms |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for trial, row in result['comparisons'].items():
        lines.append(f"| {trial} | {row['arms']['op4']['mean_ms']:.3f} | "
            f"{row['arms']['op6']['mean_ms']:.3f} | {row['mean_paired_saved_ms']:.3f} | "
            f"{row['faster_blocks']}/{result['blocks']} | {row['target_800ms_observed']} |")
    lines += ['', f"全部窗口平均：OP4 {result['aggregate']['op4']['mean_ms']:.3f} ms；"
        f"OP6 {result['aggregate']['op6']['mean_ms']:.3f} ms；"
        f"观察到的平均节省 {result['mean_paired_saved_ms']:.3f} ms。",
        '', '| 组别 | 最大 Torch allocated GiB | 最大 Torch reserved GiB | allocated＋native GiB |',
        '| --- | ---: | ---: | ---: |']
    for arm in ARMS:
        memory = result['aggregate'][arm]
        lines.append(f"| {arm.upper()} | {memory['max_peak_allocated_gib']:.3f} | "
            f"{memory['max_peak_reserved_gib']:.3f} | {memory['max_peak_allocated_plus_taco_workspace_gib']:.3f} |")
    lines += ['', 'Torch 显存统计保留图私有池及暂存张量的可见占用，但不覆盖 NCCL、CUDA context 等全部显存；reserved 不与 allocated 相加。',
        '原始窗口、每块差值、显存和数值证据保存在 report.json、pair-evidence.json 及各子批目录。',
        'completed 表示验证完成，不等于达到 800ms；达线结果是本批观察值。', '', result['limitation'], '']
    return '\n'.join(lines)


def replay(root, recompute_profiles=True):
    from op6_evidence import audit_pair
    root = Path(root)
    op3.base.verify_controllers(root)
    plan = read(root/'plan.json')
    require(plan['protocol'] == PROTOCOL and plan['model'] == p.MODEL
            and plan['nccl_overrides'] == CHANNEL_ENV and plan['target_ms'] == TARGET_MS
            and plan['schedule'] == schedule(plan['trials'], plan['blocks']), 'Wrong OP6 plan')
    samples, records, evidence = [], {}, {}
    initial_build = initial_identity = initial_environment = None
    for job in plan['schedule']:
        child = root/job['directory']
        launch = child.with_name(child.name+'-launch.json')
        saved = read(launch)
        require(saved['job'] == job and saved['nccl_overrides'] == CHANNEL_ENV, 'Changed OP6 child launch')
        result = child_result(child, job['arm'], recompute_profiles)
        child_plan = read(child/'plan.json')
        build = child_plan['build']
        require(initial_build is None or build == initial_build, 'OP6 children used different builds')
        initial_build = build
        environment = {key: child_plan[key] for key in ('model', 'environment', 'communication', 'cuda_visible_devices')}
        require(initial_environment is None or environment == initial_environment,
                'OP6 child environment changed between arms/windows')
        initial_environment = environment
        for key, value in CHANNEL_ENV.items():
            require(environment['communication']['environment'].get(key) == value, 'OP4 channel settings changed')
        current = op4.training_identity(child)
        require(initial_identity is None or current == initial_identity,
                'OP6 hardware/software/weights/RNG/input changed between windows')
        initial_identity = current
        samples.append(dict(result['all_samples'][0], **job))
        records[job['directory']+'/report.json'] = sha(child/'report.json')
        records[job['directory']+'-launch.json'] = sha(launch)
        if job['index'] == 1:
            key = f"compare{job['trial']}/block{job['block']:02d}"
            evidence[key] = audit_pair(root/(key+'-op4'), root/(key+'-op6'),
                                       recompute_profiles=recompute_profiles)
            require(evidence[key].get('passed') is True, 'OP6 pair audit failed')
    require(evidence == read(root/'pair-evidence.json'), 'OP6 pair evidence differs from raw files')
    require((root/'build').resolve() == Path(initial_build['root']).resolve(), 'Reusable build link changed')
    result = report(samples, plan['trials'], plan['blocks'])
    result.update(run_id=plan['run_id'], plan_sha256=sha(root/'plan.json'), child_reports=records,
                  pair_evidence_sha256=sha(root/'pair-evidence.json'), build=initial_build)
    return result


def supervise(args):
    from op6_evidence import audit_pair
    root, lock = args.out, None
    state = read(root/'batch-state.json')
    try:
        check_overrides()
        lock_path = REPO/'logs/5090/.op6.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        op3.base.verify_controllers(root)
        resolve_reused_build(args)
        plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex, trials=args.trials, blocks=args.blocks,
            model=p.MODEL, target_ms=TARGET_MS, nccl_overrides=CHANNEL_ENV,
            schedule=schedule(args.trials, args.blocks))
        write(root/'plan.json', plan)
        prepare_initial_build(args, root, state)
        build, evidence = args.op3_build, {}
        for job in plan['schedule']:
            op3.base.verify_controllers(root)
            child = root/job['directory']
            child.parent.mkdir(parents=True, exist_ok=True)
            command = child_command(args, job, build)
            write(child.with_name(child.name+'-launch.json'),
                  dict(job=job, command=command, nccl_overrides=CHANNEL_ENV))
            state.update(status='running', stage=job['directory'])
            write(root/'batch-state.json', state)
            print('STAGE', job['directory'], flush=True)
            run_child(command, child_environment(), child.with_name(child.name+'-supervisor.log'))
            child_result(child, job['arm'], recompute_profiles=False)
            actual = Path(read(child/'plan.json')['build']['root']).resolve()
            require(build is None or actual == build.resolve(), 'OP6 child changed its requested build')
            build = actual
            if not (root/'build').exists():
                (root/'build').symlink_to(build, target_is_directory=True)
            require((root/'build').resolve() == build, 'OP6 reusable build link changed')
            if job['index'] == 1:
                key = f"compare{job['trial']}/block{job['block']:02d}"
                evidence[key] = audit_pair(root/(key+'-op4'), root/(key+'-op6'), recompute_profiles=False)
                require(evidence[key].get('passed') is True, 'OP6 pair audit failed')
                write(root/'pair-evidence.json', evidence)
        state.update(stage='replay-report')
        write(root/'batch-state.json', state)
        result = replay(root, recompute_profiles=False)
        write(root/'report.json', result)
        (root/'report.md').write_text(markdown(result))
        state.update(status='completed', stage='done', timing_windows=result['timing_windows'],
            formal_optimizer_steps=result['formal_optimizer_steps'],
            target_800ms_observed=result['target_800ms_observed'], report=str(root/'report.md'))
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
    for key in ('python', 'megatron', 'cuda_home', 'build_cuda_home', 'trials', 'blocks', 'jobs', 'worker_timeout'):
        command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    for key in ('build', 'op3_build', 'abc_build'):
        if getattr(args, key) is not None:
            command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    return command


def main(argv=None):
    args = arguments(argv)
    if args.report:
        require(read(args.out/'batch-state.json')['status'] == 'completed', 'OP6 batch is incomplete')
        result = replay(args.out)
        require(result == read(args.out/'report.json'), 'Saved OP6 report differs from original evidence')
        print(markdown(result))
        return
    if not args.execute:
        print('PLAN ONLY: no files, builds, GPU queries or training. Add --execute to run.')
        print(json.dumps(dict(protocol=PROTOCOL, out=str(args.out), arms=list(ARMS),
            nccl_overrides=CHANNEL_ENV, target_ms=TARGET_MS, trials=args.trials, blocks=args.blocks,
            jobs=args.jobs, warmup_steps=10, timed_steps=50,
            timing_windows=2*args.trials*args.blocks, formal_optimizer_steps=100*args.trials*args.blocks,
            schedule=schedule(args.trials, args.blocks),
            build=str(args.build or args.op3_build or args.abc_build) if (args.build or args.op3_build or args.abc_build) else None,
            workflow='Paired OP4/OP6 hardware, native, operator, model smoke, profile, full-step timing and pair audit',
            build_reuse='OP3/ABC core libraries; MA9/MA11 small compatibility libraries remain per child'), indent=2))
        return
    check_overrides()
    require(args.python.is_file(), 'Python missing: '+str(args.python))
    if args.python.absolute() != Path(sys.executable).absolute():
        os.execv(str(args.python), [str(args.python), str(Path(__file__).resolve()),
                                  *(sys.argv[1:] if argv is None else argv)])
    # Reject an incompatible requested build before allocating a result directory.
    if args.build is not None:
        resolved = argparse.Namespace(**vars(args))
        resolve_reused_build(resolved)
    if args.inside:
        return supervise(args)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(protocol=PROTOCOL, status='launching', stage='freeze-controllers'))
    try:
        files = op6_controllers.collect(op3.base.controller_files)
        for name in files:
            destination = args.out/'handoff/source'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, destination)
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
