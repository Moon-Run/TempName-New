#!/usr/bin/env python3
"""RTX5090 OP1-A: direct-wire A / Flux basic / Flux basic with interleaved CTA order."""
import argparse
from datetime import datetime
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO, load
import op1_a_protocol as p

spec = importlib.util.spec_from_file_location('rtx5090_op1_base_controller', HERE/'baseline.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
read, write, require, sha = p.read, p.write, p.require, p.sha


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda'))
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument('--baseline-build', type=Path, help='Reuse a verified nine-arm build as the frozen source/reference')
    reuse.add_argument('--op1-build', type=Path, help='Reuse an already compiled OP1 build and its frozen baseline reference')
    parser.add_argument('--trials', type=int, choices=(1, 2), default=1)
    parser.add_argument('--blocks', type=int, default=6, help='Positive multiple of 6 complete blocks per trial')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=3600)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--preflight-only', action='store_true')
    stages.add_argument('--transport-smoke-only', action='store_true', help='Stop after native bitwise and actual CTA mapping probes')
    parser.add_argument('--report', action='store_true', help='Read-only verification of a completed --out batch')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    p.schedule(args.trials, args.blocks)
    require(args.jobs > 0 and args.worker_timeout > 300, 'Positive jobs and worker timeout > 300 required')
    require(not args.report or (args.out and not args.execute and not args.preflight_only
                               and not args.transport_smoke_only), '--report requires --out without execution flags')
    require(not args.inside or (args.execute and args.out and not args.report), 'Invalid supervisor invocation')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO.parent/'Load-5090/logs/5090/phase6'/f'op1-a-direct-wire-v1-{args.trials}x-50steps-{stamp}'
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().absolute() if name == 'python' else value.expanduser().resolve())
    return args


def worker_environment(args, build, scripts, job, dest, plan, variant=None):
    import op1_a_build as builder
    arm = job['arm']
    reference = builder.base_root(build)
    old_arm = 'a' if arm == 'a' else 'flux-basic'
    env = base.worker_environment(args, reference, scripts, dict(job, arm=old_arm), dest, plan)
    old_native = load('build').native_root(reference, old_arm)
    native = (load('build').native_root(reference, 'a') if variant == 'old-a'
              else builder.native_root(build, arm))
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH'):
        env[key] = env[key].replace(str(old_native), str(native))
    env['RTX5090_OP1_EXECUTE'] = '1'
    return env


def window(root, job, plan, workers):
    import op1_a_validation as validation
    directory = root/'results'/job['directory']
    require(read(directory/'command.json')['job'] == job, 'Worker command/job changed')
    base.check_worker(root, job['arm'], workers[job['arm']])
    rows = validation.validate_rows([read(directory/f'rank{rank}.json') for rank in range(8)],
                                    job, plan, workers[job['arm']])
    hardware = read(root/'environment-preflight.json')
    keys = ('name', 'uuid', 'capability', 'sm_count')
    expected = [{key: device[key] for key in keys} for device in hardware['devices']]
    require(all(row['hardware']['host'] == hardware['host']
                and [{key: device[key] for key in keys} for device in row['hardware']['devices']] == expected
                for row in rows), 'Worker hardware changed after preflight')
    return rows


def check_identity(rows, initial, references, job):
    import op1_a_validation as validation
    current = validation.identity(rows)
    require(initial is None or current == initial, 'Hardware/software/weights/RNG/input changed between windows')
    if job['arm'] in ('flux-basic', 'flux-basic-interleaved'):
        key = (job['trial'], job['block'], job['mode'])
        pair = references.setdefault(key, {})
        require(job['arm'] not in pair, 'Duplicate Flux numerical comparison window')
        pair[job['arm']] = rows
    return current


def numerical_comparison(references):
    """Describe the explicitly permitted Stream-K rounding change; do not certify equivalence."""
    windows = []
    for (trial, block, mode), pair in sorted(references.items()):
        require(set(pair) == {'flux-basic', 'flux-basic-interleaved'}, 'Incomplete paired Flux numerical evidence')
        original, sorted_rows = pair['flux-basic'], pair['flux-basic-interleaved']
        record = dict(trial=trial, block=block, mode=mode)
        for key, label in (('losses', 'loss'), ('grad_norms', 'grad_norm')):
            differences, relative = [], []
            for left, right in zip(original, sorted_rows):
                require(len(left[key]) == len(right[key]), 'Mismatched numerical trajectory lengths')
                for a, b in zip(left[key], right[key]):
                    differences.append(abs(b-a))
                    relative.append(abs(b-a)/max(abs(a), 1e-12))
            record[f'max_{label}_abs_difference'] = max(differences)
            record[f'max_{label}_relative_difference'] = max(relative)
        record['final_parameters_identical'] = all(
            left['final_parameters_sha256'] == right['final_parameters_sha256']
            for left, right in zip(original, sorted_rows))
        record['trajectory_identical'] = all(
            all(left[key] == right[key] for key in ('losses', 'grad_norms', 'final_parameters_sha256'))
            for left, right in zip(original, sorted_rows))
        if mode == 'smoke':
            record['smoke_projection_outputs_identical'] = all(
                left['smoke_output_sha256'] == right['smoke_output_sha256']
                for left, right in zip(original, sorted_rows))
            record['smoke_parameter_trajectory_identical'] = all(
                left['parameter_trajectory'] == right['parameter_trajectory']
                for left, right in zip(original, sorted_rows))
            record['trajectory_identical'] &= record['smoke_parameter_trajectory_identical']
        windows.append(record)
    require(bool(windows), 'Missing Flux numerical comparison')
    return dict(paired_windows=len(windows), all_trajectories_identical=all(r['trajectory_identical'] for r in windows),
        final_parameters_identical_windows=sum(r['final_parameters_identical'] for r in windows),
        **{key: max(r[key] for r in windows) for key in (
            'max_loss_abs_difference', 'max_loss_relative_difference',
            'max_grad_norm_abs_difference', 'max_grad_norm_relative_difference')},
        windows=windows, equivalence_required=False, reduction_changes_allowed=True)


def native_evidence(root, plan, hardware):
    import op1_a_probe as probe
    result = probe.check_records(root/'native-probe', hardware, plan['build']['manifest_sha256'],
                                 build=Path(plan['build']['root']))
    require(result == read(root/'native-probe/report.json'), 'Native probe evidence changed')
    record = read(root/'mapping-summary.json')
    require(record['completed'] is True and record['build_manifest_sha256'] == plan['build']['manifest_sha256']
            and set(record['evidence_files']) == {'mapping/a-attn.log', 'mapping/a-mlp.log'},
            'A actual CUDA mapping evidence missing')
    base.verify_files(root, record['evidence_files'])


def native_checks(args, build, plan):
    import op1_a_build as builder
    import op1_a_probe as probe
    from tp8_dp2_train import run_process
    mapping = {}
    for role, k in (('attn', 4096), ('mlp', 16384)):
        native = builder.native_root(build, 'a')
        job = dict(arm='a', mode='smoke', trial=0, block=-1, index=-1)
        log = args.out/'mapping'/f'a-{role}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        env = worker_environment(args, build, args.out/'workers/a/scripts', job, log.parent, plan)
        run_process([native/'check_mapping', '2048', '4096', str(k), '8', native/'libflux_cuda.so'],
                    REPO, env, log, 180, base.Peer())
        mapping[str(log.relative_to(args.out))] = sha(log)
    write(args.out/'mapping-summary.json', dict(completed=True, evidence_files=mapping,
                                               build_manifest_sha256=plan['build']['manifest_sha256']))
    for variant in ('old-a', 'a', 'flux-basic-interleaved'):
        base.verify_controllers(args.out)
        base.gpu_idle()
        arm = 'a' if variant == 'old-a' else variant
        job = dict(arm=arm, mode='smoke', trial=0, block=-1, index=-1)
        dest = args.out/'native-probe'/variant
        dest.mkdir(parents=True)
        env = worker_environment(args, build, args.out/'workers'/arm/'scripts', job, dest, plan, variant)
        command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '--max-restarts=0', HERE/'op1_a_probe.py', '--build', build,
                   '--variant', variant, '--out', dest, '--execute']
        write(dest/'command.json', dict(command=list(map(str, command))))
        print('NATIVE PROBE', variant, flush=True)
        try:
            run_process(command, REPO, env, dest/'process.log', max(900, args.worker_timeout), base.Peer())
        except Exception:
            base.print_smoke_failures(dest)
            raise
    hardware = read(args.out/'environment-preflight.json')
    write(args.out/'native-probe/report.json', probe.check_records(args.out/'native-probe', hardware,
                                                                 plan['build']['manifest_sha256'], build=build))
    native_evidence(args.out, plan, hardware)


def replay(root, recompute_profiles=True):
    import op1_a_build as builder
    import op1_a_profile as profiler
    root = Path(root)
    base.verify_controllers(root)
    plan, workers = read(root/'plan.json'), read(root/'workers.json')
    require(plan['protocol'] == p.PROTOCOL and plan['transport'] == p.TRANSPORT
            and plan['model'] == p.MODEL and plan['arms'] == list(p.ARMS)
            and plan['schedule'] == p.schedule(plan['trials'], plan['blocks'])
            and set(workers) == set(p.ARMS), 'Wrong OP1 plan')
    build = Path(plan['build']['root'])
    require(sha(build/'op1-build.json') == plan['build']['manifest_sha256'], 'OP1 build changed')
    builder.verify(build, True)
    hardware = read(root/'environment-preflight.json')
    require(sha(root/'environment-preflight.json') == plan['preflight_sha256'] and hardware['completed'] is True
            and hardware['environment'] == plan['environment'] and hardware['communication'] == plan['communication'],
            'Missing/changed hardware preflight')
    base.collective_evidence(root, hardware)
    native_evidence(root, plan, hardware)
    smoke, profile = read(root/'preflight.json'), read(root/'profile-summary.json')
    require(smoke == dict(completed=True, arms=dict.fromkeys(p.ARMS, True)), 'Three full-model smokes are required')
    require(profile['protocol'] == p.PROTOCOL and profile['run_id'] == plan['run_id']
            and profile['completed'] and profile['execution_validated']
            and set(profile['profiles']) == {f'{arm}/rank{rank}' for arm in p.ARMS for rank in range(8)},
            'Three complete profile sets are required')
    expected = {f'results/profile-{arm}/rank{rank}-profile.json.gz' for arm in p.ARMS for rank in range(8)}
    require(set(profile['evidence_files']) == expected, 'Missing raw profile trace')
    base.verify_files(root, profile['evidence_files'])
    for arm in p.ARMS:
        for rank in range(8):
            record = profile['profiles'][f'{arm}/rank{rank}']
            require(record['execution_validated'] is True and record['arm'] == arm and record['rank'] == rank,
                    'Incomplete profile mechanism evidence')
            if recompute_profiles:
                with gzip.open(root/'results'/f'profile-{arm}'/f'rank{rank}-profile.json.gz', 'rt') as handle:
                    events = json.load(handle)['traceEvents']
                require(profiler.analyze(events, arm, rank) == record, 'Profile summary differs from raw trace')
    initial, references, samples, records = None, {}, [], {}
    for job in p.jobs(plan['trials'], plan['blocks']):
        rows = window(root, job, plan, workers)
        initial = check_identity(rows, initial, references, job)
        if job['mode'] == 'timing':
            samples.append(p.summarize(rows, job))
        for rank in range(8):
            name = f"results/{job['directory']}/rank{rank}.json"
            records[name] = sha(root/name)
    result = p.report(samples, plan['trials'], plan['blocks'])
    numerical = numerical_comparison(references)
    result.update(run_id=plan['run_id'], correctness_passed=True, execution_path_validated=True,
        flux_ordering_trajectories_identical=numerical['all_trajectories_identical'],
        flux_numerical_comparison=numerical, a_native_old_new_bitwise=True,
        hardware=initial[0]['hardware'], environment=plan['environment'], worker_records=records,
        profile_summary_sha256=sha(root/'profile-summary.json'), plan_sha256=sha(root/'plan.json'),
        native_probe_sha256=sha(root/'native-probe/report.json'), build=plan['build'])
    return result


def supervise(args):
    import op1_a_build as builder
    import op1_a_sources as sources
    import op1_a_profile as profiler
    from run import arguments as model_arguments
    from tp8_dp2_train import run_process
    root, lock = args.out, None
    state = read(root/'batch-state.json')
    def stage(name):
        state.update(status='running', stage=name)
        write(root/'batch-state.json', state)
        print('STAGE', name, flush=True)
    try:
        lock_path = REPO/'logs/5090/.baseline.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.verify_controllers(root)
        stage('worker-source-preflight')
        for arm in p.ARMS:
            for generate in (sources.worker_source, sources.adapter_source, sources.taco_source):
                compile(generate(arm), f'{arm}/{generate.__name__}', 'exec')
        stage('hardware-preflight')
        hardware = base.hardware_preflight(args)
        if args.preflight_only:
            state.update(status='preflight_completed', stage='done', gpu_benchmark_completed=False)
            write(root/'batch-state.json', state)
            print('PREFLIGHT PASSED', root/'environment-preflight.json', flush=True)
            return
        if args.op1_build:
            stage('verify-op1-build')
            build = args.op1_build
            builder.verify(build, True)
            require(read(build/'op1-build.json')['environment'] == load('build').environment_identity(args.build_cuda_home),
                    'OP1 build ABI/Python/CUDA environment changed')
        else:
            reference = args.baseline_build or root/'baseline-reference'
            if args.baseline_build:
                base.reuse_build(args, reference, stage)
            else:
                stage('prepare-frozen-baseline')
                load('build').prepare(reference)
                stage('compile-frozen-baseline')
                load('build').compile_build(reference, args.megatron, args.build_cuda_home, args.jobs)
            build = root/'build'
            stage('prepare-op1-build')
            builder.prepare(build, reference)
            stage('compile-op1-build')
            builder.compile_build(build, args.build_cuda_home, args.jobs)
        plan = dict(protocol=p.PROTOCOL, transport=p.TRANSPORT, run_id=uuid.uuid4().hex,
            arms=list(p.ARMS), model=p.MODEL, trials=args.trials, blocks=args.blocks,
            schedule=p.schedule(args.trials, args.blocks), communication=hardware['communication'],
            environment=hardware['environment'], cuda_visible_devices=hardware['cuda_visible_devices'],
            preflight_sha256=sha(root/'environment-preflight.json'),
            build=dict(root=str(build), manifest_sha256=sha(build/'op1-build.json')))
        write(root/'plan.json', plan)
        stage('prepare-workers')
        workers = sources.prepare_workers(build, root, plan)
        write(root/'workers.json', workers)
        stage('native-bitwise-and-mapping-probes')
        native_checks(args, build, plan)
        if args.transport_smoke_only:
            state.update(status='transport_smoke_completed', stage='done', gpu_benchmark_completed=False,
                         report=str(root/'native-probe/report.json'))
            write(root/'batch-state.json', state)
            print('OP1 NATIVE PROBES PASSED', root/'native-probe/report.json', flush=True)
            return
        smoke, initial, references, samples = {}, None, {}, []
        profile = dict(protocol=p.PROTOCOL, run_id=plan['run_id'], completed=False,
                       execution_validated=False, profiles={}, evidence_files={})
        for job in p.jobs(args.trials, args.blocks):
            if job['mode'] != 'smoke':
                require(set(smoke) == set(p.ARMS), 'All three model smokes must pass')
            if job['mode'] == 'timing':
                require(set(profile['profiles']) == {f'{a}/rank{r}' for a in p.ARMS for r in range(8)}
                        and all(v['execution_validated'] is True for v in profile['profiles'].values()),
                        'All three actual execution paths must pass before timing')
            base.verify_controllers(root)
            require(sha(build/'op1-build.json') == plan['build']['manifest_sha256'], 'OP1 build identity changed')
            # Recheck registered binary/source hashes without preparing or recompiling anything.
            builder.verify(build, True)
            scripts = base.check_worker(root, job['arm'], workers[job['arm']])
            base.gpu_idle()
            dest = root/'results'/job['directory']
            dest.mkdir(parents=True)
            command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                       '--max-restarts=0', scripts/'worker.py', *model_arguments(p.MODEL),
                       '--distributed-timeout-minutes', '5']
            write(dest/'command.json', dict(job=job, command=list(map(str, command))))
            stage(job['directory'])
            run_process(command, builder.base_root(build)/'megatron',
                        worker_environment(args, build, scripts, job, dest, plan), dest/'process.log',
                        args.worker_timeout, base.Peer())
            rows = window(root, job, plan, workers)
            initial = check_identity(rows, initial, references, job)
            if job['mode'] == 'smoke':
                smoke[job['arm']] = True
                write(root/'preflight.json', dict(completed=set(smoke) == set(p.ARMS), arms=smoke))
            elif job['mode'] == 'timing':
                samples.append(p.summarize(rows, job))
                write(root/'samples.json', samples)
            else:
                for rank in range(8):
                    trace = dest/f'rank{rank}-profile.json.gz'
                    with gzip.open(trace, 'rt') as handle:
                        events = json.load(handle)['traceEvents']
                    profile['profiles'][f"{job['arm']}/rank{rank}"] = profiler.analyze(events, job['arm'], rank)
                    profile['evidence_files'][str(trace.relative_to(root))] = sha(trace)
                write(root/'profile-summary.json', profile)
        profile.update(completed=True, execution_validated=True)
        write(root/'profile-summary.json', profile)
        stage('replay-report')
        result = replay(root, recompute_profiles=False)
        write(root/'report.json', result)
        (root/'report.md').write_text(p.markdown(result))
        state.update(status='completed', stage='done', timing_windows=result['timing_windows'],
                     formal_optimizer_steps=result['formal_optimizer_steps'], report=str(root/'report.md'))
        write(root/'batch-state.json', state)
        print('COMPLETED', root/'report.md', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write(root/'batch-state.json', state)
        raise
    finally:
        if lock is not None:
            lock.close()


def main(argv=None):
    args = arguments(argv)
    if args.report:
        result = replay(args.out)
        require(result == read(args.out/'report.json'), 'Saved OP1 report differs from original evidence')
        print(p.markdown(result))
        return
    if not args.execute:
        early = args.preflight_only or args.transport_smoke_only
        print('PLAN ONLY: no files, builds, GPU queries or training workers. Add --execute to run.')
        print(json.dumps(dict(protocol=p.PROTOCOL, out=str(args.out), arms=list(p.ARMS), model=p.MODEL,
            transport=p.TRANSPORT, peer_access_required=False, trials=args.trials, blocks=args.blocks,
            warmup_steps=10, timed_steps=50, timing_windows=0 if early else args.trials*args.blocks*3,
            formal_optimizer_steps=0 if early else args.trials*args.blocks*3*50,
            baseline_build=str(args.baseline_build) if args.baseline_build else None,
            op1_build=str(args.op1_build) if args.op1_build else None,
            preflight_only=args.preflight_only, transport_smoke_only=args.transport_smoke_only,
            workflow='hardware/NCCL -> isolated build -> old/new A bitwise + actual CTA mapping '
                     '-> three model smokes -> three profiles -> balanced full-step timing -> report'),
            indent=2, ensure_ascii=False))
        return
    require(args.python.is_file(), 'Python missing: '+str(args.python))
    if args.python.absolute() != Path(sys.executable).absolute():
        os.execv(str(args.python), [str(args.python), str(Path(__file__).resolve()),
                                  *(sys.argv[1:] if argv is None else argv)])
    if args.inside:
        return supervise(args)
    if not args.preflight_only and not args.baseline_build and not args.op1_build:
        require(args.megatron.is_dir(), 'Megatron source missing: '+str(args.megatron))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(protocol=p.PROTOCOL, status='launching', stage='freeze-controllers'))
    try:
        files = base.controller_files()
        for name in files:
            dest = args.out/'handoff/source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, dest)
        write(args.out/'handoff/controllers.json', files)
        write(args.out/'handoff/launch.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        print('Batch:', args.out, flush=True)
        if args.foreground or args.preflight_only:
            return supervise(args)
        command = [str(args.python), str(Path(__file__).resolve()), '--execute', '--inside', '--out', str(args.out)]
        for name in ('python', 'megatron', 'cuda_home', 'build_cuda_home', 'trials', 'blocks', 'jobs', 'worker_timeout'):
            command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
        for name in ('baseline_build', 'op1_build'):
            if getattr(args, name):
                command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
        if args.transport_smoke_only:
            command.append('--transport-smoke-only')
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
