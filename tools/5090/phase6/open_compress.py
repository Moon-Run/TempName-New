#!/usr/bin/env python3
"""RTX5090: Flux BF16 / open block INT8 / INT8 with true GEMM tile interleaving."""
import argparse
from datetime import datetime
import fcntl
import gzip
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import sys
import uuid
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO
import open_compress_protocol as p

spec = importlib.util.spec_from_file_location('open_compress_base_controller', HERE/'baseline.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--build', type=Path, help='Reuse a completed open-compress build, never an old TACO build')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--blocks', type=int, default=6, help='Positive complete blocks; 1 is exploratory, multiples of 6 balanced')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=3600)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true', help='Compatibility option; this entry always runs in foreground')
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--preflight-only', action='store_true')
    stage.add_argument('--operator-only', action='store_true')
    parser.add_argument('--report', action='store_true', help='Read-only replay of a completed --out batch')
    args = parser.parse_args(argv)
    p.schedule(args.trials, args.blocks)
    p.require(args.jobs > 0 and args.worker_timeout >= 300, 'Positive jobs and timeout >=300 required')
    p.require(not args.report or (args.out and not args.execute and not args.preflight_only and not args.operator_only),
              '--report requires --out without execution/stage flags')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO.parent/'Load-5090/logs/5090/phase6'/f'open-compress-int8-v1-{args.trials}x-{args.blocks}blocks-{stamp}'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().absolute() if key == 'python' else value.expanduser().resolve())
    return args


def environment(args, build, scripts, job, dest, plan):
    import ablation4_runtime as comm
    import open_compress_build as builder
    env = {k:v for k,v in os.environ.items() if not k.startswith(
        ('NCCL_', 'TORCH_NCCL_', 'H100_', 'E2E_', 'FLUX_', 'RTX5090_', 'OPEN_COMPRESS_'))}
    for key in ('PYTHONHOME', 'PYTHONOPTIMIZE', 'MECH_SMOKE'):
        env.pop(key, None)
    native = builder.native_root(build, job['arm'])
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=plan['cuda_visible_devices'],
        CUDA_DEVICE_MAX_CONNECTIONS='1', FLUX_GEMM_RS_GEMM_ONLY='1', RTX5090_BASELINE_EXECUTE='1',
        RTX5090_TRANSPORT=p.TRANSPORT, H100_TP='8', H100_SM_COUNT='170', H100_RUN_ID=plan['run_id'],
        E2E_POLICY=job['arm'], E2E_MODE=job['mode'], E2E_CASE='rtx5090-open-compress',
        E2E_BUILD=str(scripts/'build'), E2E_PROCESS_OUT=str(dest), E2E_TRIAL=str(job['trial']),
        E2E_BLOCK=str(job['block']), E2E_WINDOW=str(job['index']), CUDA_HOME=str(args.cuda_home),
        OPEN_COMPRESS_LIBRARY=str(build/'codec/libopen_compress.so'),
        PATH=f'{args.python.parent}:{args.cuda_home}/bin:'+env.get('PATH', ''),
        PYTHONPATH=':'.join(map(str, (scripts, build/'megatron', native/'python'))))
    libraries = [native/'python/flux/lib', native, native/'build/lib', args.cuda_home/'lib64']
    libraries += [Path(value) for value in os.environ.get('LD_LIBRARY_PATH','').split(':') if value
                  and not any((Path(value)/name).exists() for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'))]
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries))
    return comm.apply(env, plan['communication'])


def operator_summary(root, plan, workers, recompute_profiles=True):
    import open_compress_probe as probe
    import open_compress_profile as profiler
    hardware = p.read(root/'environment-preflight.json')
    records, evidence, stats = {}, {}, {}
    for arm in p.ARMS:
        directory = root/'operator'/arm
        rows = probe.validate(directory, arm, plan, workers[arm])
        records[arm] = rows
        for rank, row in enumerate(rows):
            p.require(row['gpu_uuid'] == hardware['devices'][rank]['uuid'] and all(
                      row[k] == hardware['environment'][k] for k in ('torch','cuda','nccl')), 'Operator environment changed')
            trace = directory/f'rank{rank}-profile.json.gz'
            p.require(p.sha(trace) == row['profile_sha256'] and row['profile']['execution_validated'] is True,
                      'Missing/changed operator profile')
            if recompute_profiles:
                with gzip.open(trace, 'rt') as handle:
                    events = json.load(handle)['traceEvents']
                p.require(profiler.analyze(events, arm, rank, 1) == row['profile'], 'Operator profile summary differs')
            for path in (directory/f'rank{rank}.json', trace):
                evidence[str(path.relative_to(root))] = p.sha(path)
        stats[arm] = {}
        for role in probe.ROLES:
            stats[arm][role] = {}
            for component in rows[0]['roles'][role]['timings']:
                windows = [row['roles'][role]['timings'][component]['samples'] for row in rows]
                stats[arm][role][component] = dict(
                    mean_max_rank_cuda_ms=mean(max(w[i]['cuda_ms'] for w in windows) for i in range(5)),
                    mean_max_rank_wall_ms=mean(max(w[i]['wall_ms'] for w in windows) for i in range(5)),
                    max_peak_allocated_gib=max(s['peak_allocated_gib'] for w in windows for s in w),
                    max_peak_reserved_gib=max(s['peak_reserved_gib'] for w in windows for s in w),
                    max_incremental_allocated_bytes=max(s['incremental_allocated_bytes'] for w in windows for s in w))
    for rank in range(8):
        for role in probe.ROLES:
            for case in probe.CASES:
                for key in ('input_sha256','weight_sha256'):
                    p.require(len({records[a][rank]['roles'][role]['cases'][case][key] for a in p.ARMS}) == 1,
                              'Operator arms received different inputs')
    return dict(protocol=p.PROTOCOL, run_id=plan['run_id'], completed=True, evidence_files=evidence, measurements=stats,
        limitation='Components timed separately; their sums do not predict integrated latency. '
          'Operator process is independent of full-model process. Peaks include live operator buffers. '
          'Wall time includes Python and synchronization; CUDA events cover caller stream with NCCL wait dependencies.')


def window(root, job, plan, workers):
    import open_compress_validation as validation
    directory = root/'results'/job['directory']
    p.require(p.read(directory/'command.json')['job'] == job, 'Worker command/job changed')
    base.check_worker(root, job['arm'], workers[job['arm']])
    rows = validation.validate_rows([p.read(directory/f'rank{r}.json') for r in range(8)], job, plan, workers[job['arm']])
    hardware = p.read(root/'environment-preflight.json')
    keys = ('name','uuid','capability','sm_count')
    expected = [{k:d[k] for k in keys} for d in hardware['devices']]
    p.require(all(row['hardware']['host'] == hardware['host'] and
                  [{k:d[k] for k in keys} for d in row['hardware']['devices']] == expected for row in rows),
              'Worker hardware changed after preflight')
    return rows


def replay(root, recompute_profiles=True):
    import open_compress_build as builder
    import open_compress_profile as profiler
    import open_compress_validation as validation
    root = Path(root)
    base.verify_controllers(root)
    plan, workers = p.read(root/'plan.json'), p.read(root/'workers.json')
    p.require(plan['protocol'] == p.PROTOCOL and plan['model'] == p.MODEL and plan['arms'] == list(p.ARMS)
              and plan['schedule'] == p.schedule(plan['trials'],plan['blocks']) and set(workers) == set(p.ARMS),
              'Wrong experiment plan')
    build = Path(plan['build']['root'])
    p.require(p.sha(build/builder.MANIFEST) == plan['build']['manifest_sha256'], 'Build changed')
    builder.verify(build, True)
    hardware = p.read(root/'environment-preflight.json')
    p.require(hardware['completed'] is True and p.sha(root/'environment-preflight.json') == plan['preflight_sha256']
              and hardware['environment'] == plan['environment'] and hardware['communication'] == plan['communication'],
              'Hardware preflight missing/changed')
    base.collective_evidence(root, hardware)
    operators = operator_summary(root, plan, workers, recompute_profiles)
    p.require(operators == p.read(root/'operator-summary.json'), 'Operator summary differs from raw evidence')
    initial, samples, evidence, groups = None, [], {}, {}
    profiles = p.read(root/'profile-summary.json')
    p.require(set(profiles) == {f'{a}/rank{r}' for a in p.ARMS for r in range(8)}, 'Missing training profiles')
    for job in p.jobs(plan['trials'],plan['blocks']):
        rows = window(root, job, plan, workers)
        current = validation.identity(rows)
        p.require(initial is None or current == initial, 'Hardware/software/initial weights/RNG/input changed')
        initial = current
        groups.setdefault((job['trial'],job['block'],job['mode']), {})[job['arm']] = rows
        if job['mode'] == 'timing':
            samples.append(p.summarize(rows, job))
        for rank in range(8):
            path = root/'results'/job['directory']/f'rank{rank}.json'
            evidence[str(path.relative_to(root))] = p.sha(path)
            if job['mode'] == 'profile':
                trace = path.with_name(f'rank{rank}-profile.json.gz')
                saved = profiles[f"{job['arm']}/rank{rank}"]
                p.require(saved['trace_sha256'] == p.sha(trace) and saved['analysis']['execution_validated'], 'Training trace changed')
                if recompute_profiles:
                    with gzip.open(trace,'rt') as handle:
                        events = json.load(handle)['traceEvents']
                    p.require(profiler.analyze(events,job['arm'],rank) == saved['analysis'], 'Training profile summary differs')
                evidence[str(trace.relative_to(root))] = p.sha(trace)
    p.require(all(set(g) == set(p.ARMS) for g in groups.values()), 'Incomplete paired numerical groups')
    numerical = {}
    for a,b in itertools.combinations(p.ARMS,2):
        numerical[a+'/'+b] = dict(
            max_loss_abs_difference=max(abs(x-y) for g in groups.values() for x,y in zip(g[a][0]['losses'],g[b][0]['losses'])),
            max_grad_norm_abs_difference=max(abs(x-y) for g in groups.values() for rank in range(8)
                for x,y in zip(g[a][rank]['grad_norms'],g[b][rank]['grad_norms'])),
            final_parameter_fingerprints_equal=all(g[a][rank]['final_parameters_sha256'] ==
                g[b][rank]['final_parameters_sha256'] for g in groups.values() for rank in range(8)))
    return dict(p.report(samples,plan['trials'],plan['blocks']), run_id=plan['run_id'], correctness_passed=True,
        execution_path_validated=True, numerical_comparisons=numerical, operator_measurements=operators['measurements'],
        evidence_files=evidence, operator_summary_sha256=p.sha(root/'operator-summary.json'),
        plan_sha256=p.sha(root/'plan.json'), build=plan['build'], environment=plan['environment'])


def execute(args):
    import open_compress_build as builder
    import open_compress_sources as sources
    import open_compress_profile as profiler
    from run import arguments as model_arguments
    from tp8_dp2_train import run_process
    # Validate frozen training source before GPU work or a long build.
    for arm in p.ARMS:
        sources.worker_source(arm)
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    state = dict(protocol=p.PROTOCOL,status='running',stage='freeze')
    p.write(root/'batch-state.json',state)
    def stage(name):
        state.update(stage=name)
        p.write(root/'batch-state.json',state)
        print('STAGE',name,flush=True)
    lock = None
    try:
        lockpath = REPO/'logs/5090/.baseline.lock'
        lockpath.parent.mkdir(parents=True,exist_ok=True)
        lock = lockpath.open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files = base.controller_files()
        for name in files:
            destination = root/'handoff/source'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(REPO/name,destination)
        p.write(root/'handoff/controllers.json',files)
        p.write(root/'handoff/launch.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        print('Batch:',root,flush=True)
        stage('hardware-and-nccl-preflight')
        hardware = base.hardware_preflight(args)
        if args.preflight_only:
            state.update(status='preflight_completed',stage='done',gpu_benchmark_completed=False)
            p.write(root/'batch-state.json',state)
            return
        build = args.build or root/'build'
        if args.build:
            stage('verify-build')
            builder.verify(build,True)
            p.require(p.read(build/builder.MANIFEST)['environment'] == builder.environment_identity(args.build_cuda_home),
                      'Reused build Python/CUDA ABI differs')
        else:
            stage('prepare-build')
            builder.prepare(build)
            stage('compile-build')
            builder.compile_build(build,args.megatron,args.build_cuda_home,args.jobs)
        plan = dict(protocol=p.PROTOCOL,transport=p.TRANSPORT,run_id=uuid.uuid4().hex,arms=list(p.ARMS),model=p.MODEL,
            trials=args.trials,blocks=args.blocks,schedule=p.schedule(args.trials,args.blocks),
            communication=hardware['communication'],environment=hardware['environment'],
            cuda_visible_devices=hardware['cuda_visible_devices'],preflight_sha256=p.sha(root/'environment-preflight.json'),
            build=dict(root=str(build),manifest_sha256=p.sha(build/builder.MANIFEST)))
        p.write(root/'plan.json',plan)
        stage('prepare-workers')
        workers = sources.prepare_workers(build,root,plan)
        p.write(root/'workers.json',workers)
        for arm in p.ARMS:
            stage('operator-'+arm)
            scripts = base.check_worker(root,arm,workers[arm])
            dest = root/'operator'/arm
            dest.mkdir(parents=True)
            job = dict(arm=arm,mode='profile',trial=0,block=-1,index=-1)
            command = [args.python,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','--max-restarts=0',
                       HERE/'open_compress_probe.py','--build',build,'--out',dest,'--arm',arm]
            p.write(dest/'command.json',dict(command=list(map(str,command))))
            base.gpu_idle()
            try:
                run_process(command,REPO,environment(args,build,scripts,job,dest,plan),dest/'process.log',args.worker_timeout,base.Peer())
            except Exception:
                base.print_smoke_failures(dest)
                raise
        p.write(root/'operator-summary.json',operator_summary(root,plan,workers))
        if args.operator_only:
            state.update(status='operator_completed',stage='done',gpu_benchmark_completed=False)
            p.write(root/'batch-state.json',state)
            print('OPERATOR CHECKS COMPLETED',root/'operator-summary.json',flush=True)
            return
        import open_compress_validation as validation
        profiles, smokes, samples, initial = {}, set(), [], None
        for job in p.jobs(args.trials,args.blocks):
            if job['mode'] != 'smoke':
                p.require(smokes == set(p.ARMS),'All model smokes must pass first')
            if job['mode'] == 'timing':
                p.require(len(profiles) == 24,'All training profiles must pass first')
            base.verify_controllers(root)
            builder.verify(build,True)
            scripts = base.check_worker(root,job['arm'],workers[job['arm']])
            dest = root/'results'/job['directory']
            dest.mkdir(parents=True)
            command = [args.python,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','--max-restarts=0',
                       scripts/'worker.py',*model_arguments(p.MODEL),'--distributed-timeout-minutes','5']
            p.write(dest/'command.json',dict(job=job,command=list(map(str,command))))
            stage(job['directory'])
            base.gpu_idle()
            try:
                run_process(command,build/'megatron',environment(args,build,scripts,job,dest,plan),
                            dest/'process.log',args.worker_timeout,base.Peer())
            except Exception:
                base.print_smoke_failures(dest)
                raise
            rows = window(root,job,plan,workers)
            current = validation.identity(rows)
            p.require(initial is None or current == initial,
                      'Hardware/software/initial weights/RNG/input changed between windows')
            initial = current
            if job['mode'] == 'smoke':
                smokes.add(job['arm'])
            elif job['mode'] == 'timing':
                samples.append(p.summarize(rows,job))
                p.write(root/'samples.json',samples)
            else:
                for rank in range(8):
                    trace = dest/f'rank{rank}-profile.json.gz'
                    with gzip.open(trace,'rt') as handle:
                        events = json.load(handle)['traceEvents']
                    profiles[f"{job['arm']}/rank{rank}"] = dict(trace_sha256=p.sha(trace),
                        analysis=profiler.analyze(events,job['arm'],rank))
                p.write(root/'profile-summary.json',profiles)
        stage('replay-and-report')
        result = replay(root,recompute_profiles=False)
        p.write(root/'report.json',result)
        (root/'report.md').write_text(p.markdown(result))
        state.update(status='completed',stage='done',gpu_benchmark_completed=True,report=str(root/'report.md'))
        p.write(root/'batch-state.json',state)
        print('COMPLETED',root/'report.md',flush=True)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        p.write(root/'batch-state.json',state)
        raise
    finally:
        if lock is not None:
            lock.close()


def main(argv=None):
    args = arguments(argv)
    if args.report:
        result = replay(args.out)
        p.require(result == p.read(args.out/'report.json'),'Saved report differs from evidence')
        print(p.markdown(result))
    elif not args.execute:
        print('PLAN ONLY: no files, GPU queries, builds or training. Add --execute to run.')
        print(json.dumps(dict(protocol=p.PROTOCOL,out=str(args.out),arms=p.ARMS,trials=args.trials,blocks=args.blocks,
            balanced_order=args.blocks%6 == 0,exploratory=args.blocks<6,model=p.MODEL,
            warmup_steps=10,timed_steps=50,timing_windows=0 if args.preflight_only or args.operator_only else args.trials*args.blocks*3,
            build=str(args.build) if args.build else None,arrival=False,selective=False,
            stages='hardware/NCCL -> isolated Flux+codec build -> codec/operator/profile -> model smoke/profile -> full steps -> report'),
            indent=2,ensure_ascii=False))
    else:
        if args.python.absolute() != Path(sys.executable).absolute():
            os.execv(str(args.python),[str(args.python),str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)])
        execute(args)


if __name__ == '__main__':
    main()
