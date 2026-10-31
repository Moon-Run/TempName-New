#!/usr/bin/env python3
"""Experimental nine-arm RTX5090 TP8 NCCL staged baseline; preview unless --execute."""
import argparse
from datetime import datetime
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO, H100, load

p = load('protocol')
read, write, require, sha = p.read, p.write, p.require, p.sha


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path,
                        help='Batch directory; default: ../Load-5090/logs/5090/phase6/<batch>')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--baseline-build', type=Path, help='Reuse a verified build of this RTX5090 protocol')
    parser.add_argument('--trials', type=int, choices=(1, 2), default=1)
    parser.add_argument('--blocks', type=int, default=18, help='Positive multiple of 18, per trial')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker-timeout', type=int, default=3600)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='With --execute, check hardware/environment and exit')
    parser.add_argument('--transport-smoke-only', action='store_true',
                        help='With --execute, build/verify and validate local transport without full-model timing')
    parser.add_argument('--report', action='store_true', help='Read-only replay of a completed --out campaign')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    p.schedule(args.trials, args.blocks)
    require(args.jobs > 0 and args.worker_timeout > 300, 'Positive jobs and worker timeout > 300 required')
    require(not (args.preflight_only and args.transport_smoke_only), 'Choose one early-exit stage')
    require(not args.report or (args.out and not args.execute and not args.preflight_only and not args.transport_smoke_only),
            '--report requires --out and cannot be combined with --execute/--preflight-only')
    require(not args.inside or (args.out and args.execute and not args.preflight_only), 'Invalid supervisor invocation')
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = (REPO.parent/'Load-5090/logs/5090/phase6'/
                    f'baseline-nine-arm-nccl-staged-v2-{args.trials}x-50steps-{stamp}')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().absolute() if name == 'python' else value.expanduser().resolve())
    return args


def verify_files(root, hashes):
    for name, expected in hashes.items():
        path = Path(root)/name
        require(path.is_file() and sha(path) == expected, 'Missing/changed evidence: ' + str(path))


def controller_files():
    # Freeze all local overlays, including files opened as text rather than imported.
    paths = set()
    extensions = {'.py', '.sh', '.cu', '.cuh', '.cc', '.cpp', '.h', '.hpp', '.json', '.txt'}
    for directory in (HERE, H100, REPO/'tools/a800/phase5/e2e'):
        paths.update(x for x in directory.rglob('*') if x.is_file() and x.suffix in extensions
                     and '__pycache__' not in x.parts)
    paths.add(REPO/'tools/a800/phase4/reference.py')
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(paths)}


def verify_controllers(root):
    files = read(root/'handoff/controllers.json')
    verify_files(root/'handoff/source', files)
    verify_files(REPO, files)


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, timeout=30, check=True)
    require(not result.stdout.strip(), 'Require eight idle GPUs; active compute processes:\n' + result.stdout)


def capture_communication():
    import ablation4_runtime as comm
    # Preserve interface and other settings, but pin transport for every arm.
    # Allowing SHM does not assert that NCCL actually selected a specific host
    # transport. Device buffers never become remote pointers in this protocol.
    record = comm.capture('0')
    record['environment'].update(p.TRANSPORT_ENV)
    return comm.validate(record)


def worker_environment(args, build, scripts, job, dest, plan):
    import ablation4_runtime as comm
    builder = load('build')
    arm = job['arm']
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('NCCL_', 'TORCH_NCCL_', 'H100_', 'E2E_', 'FLUX_', 'RTX5090_'))}
    for key in ('PYTHONHOME', 'PYTHONOPTIMIZE', 'MECH_SMOKE'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES=plan['cuda_visible_devices'], CUDA_DEVICE_MAX_CONNECTIONS='1',
               FLUX_GEMM_RS_GEMM_ONLY='1', RTX5090_BASELINE_EXECUTE='1',
               RTX5090_TRANSPORT=p.TRANSPORT,
               H100_TP='8', H100_SM_COUNT='170', H100_RUN_ID=plan['run_id'],
               H100_MA4_F_ARM='F4', H100_MA4_F_PROTOCOL='h100-ma4-f-v1',
               H100_MA8_ARM='F4-cache' if arm == 'ma8' else 'F4', H100_MA8_PROTOCOL='h100-ma8-f4-cache-v2',
               H100_MA9_ARM='F4-zcount' if arm == 'ma9' else 'F4', H100_MA9_PROTOCOL='h100-ma9-f4-zcount-v1',
               H100_MA11_ARM='F4-cache-zcount-csoftmax' if arm == 'ma11' else 'F4-cache-zcount',
               H100_MA11_PROTOCOL='h100-ma11-causal-softmax-v1',
               H100_MA2C_ARM=p.NATIVE_ARM.get(arm, 'A'), H100_MA2C_PROTOCOL='h100-ma2r-c-ablation-v1',
               E2E_POLICY=p.POLICY[arm], E2E_MODE=job['mode'], E2E_CASE='rtx5090-phase6',
               E2E_BUILD=str(scripts/'build'), E2E_PROCESS_OUT=str(dest), E2E_TRIAL=str(job['trial']),
               E2E_BLOCK=str(job['block']), E2E_WINDOW=str(job['index']),
               CUDA_HOME=str(args.cuda_home), PATH=f'{args.python.parent}:{args.cuda_home}/bin:' + env.get('PATH', ''))
    native = builder.native_root(build, arm)
    if arm == 'ma11' and plan.get('ma11_compat'):
        env['RTX5090_MA11_SOFTMAX_LIBRARY'] = plan['ma11_compat']['library']
    env['PYTHONPATH'] = ':'.join(map(str, (scripts, build/'megatron', native/'python')))
    libraries = [native/'python/flux/lib', native, native/'build/lib', args.cuda_home/'lib64']
    for value in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        if value and not any((Path(value)/name).exists() for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')):
            if Path(value) not in libraries:
                libraries.append(Path(value))
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries))
    return comm.apply(env, plan['communication'])


def check_worker(root, arm, worker):
    scripts = root/'workers'/arm/'scripts'
    verify_files(scripts, worker['files'])
    require(sha(scripts.parent/'plan.json') == worker['plan_sha256'], 'Worker plan changed')
    return scripts


def window(root, job, plan, workers):
    directory = root/'results'/job['directory']
    require(read(directory/'command.json')['job'] == job, 'Worker command/job changed')
    check_worker(root, job['arm'], workers[job['arm']])
    rows = [read(directory/f'rank{rank}.json') for rank in range(8)]
    rows = load('validation').validate_rows(rows, job, plan, workers[job['arm']])
    hardware = read(root/'environment-preflight.json')
    keys = ('name', 'uuid', 'capability', 'sm_count')
    expected = [{key: device[key] for key in keys} for device in hardware['devices']]
    require(all(row['hardware']['host'] == hardware['host']
                and [{key: device[key] for key in keys} for device in row['hardware']['devices']] == expected
                for row in rows), 'Worker hardware differs from preflight')
    return rows


def check_identity(rows, initial, references, job):
    validation = load('validation')
    current = validation.identity(rows)
    if initial is not None:
        require(current == initial, 'Hardware/software/weights/RNG/input changed between windows')
    for number, family in enumerate(p.EQUIVALENCE_GROUPS):
        if job['arm'] in family:
            key = (number, job['trial'], job['block'], job['mode'])
            if key in references:
                validation.equivalent(rows, references[key], job['mode'] == 'smoke')
            else:
                references[key] = rows
    return current


def profile_evidence(root, plan, profile, recompute=True):
    expected = {f"results/{job['directory']}/rank{rank}-profile.json.gz"
                for job in p.jobs(plan['trials'], plan['blocks']) if job['mode'] == 'profile'
                for rank in range(8)}
    require(set(profile['evidence_files']) == expected, 'All 72 profile traces must be retained')
    verify_files(root, profile['evidence_files'])
    if recompute:
        profiler = load('profile')
        for arm in p.ARMS:
            for rank in range(8):
                with gzip.open(root/'results'/f'profile-{arm}'/f'rank{rank}-profile.json.gz', 'rt') as handle:
                    events = json.load(handle)['traceEvents']
                require(profiler.analyze(events, arm, rank) == profile['profiles'][f'{arm}/rank{rank}'],
                        'Profile summary differs from the original trace')


def mapping_jobs(build):
    builder = load('build')
    seen, result = set(), []
    for arm in p.ARMS[:-1]:
        native = builder.native_root(build, arm).resolve()
        if native not in seen:
            seen.add(native)
            result += [(arm, role, native, k) for role, k in (('attn', 4096), ('mlp', 16384))]
    return result


def mapping_evidence(root, plan):
    record = read(root/'mapping-summary.json')
    expected = {f'mapping/{arm}-{role}/mapping.log' for arm, role, _, _ in mapping_jobs(Path(plan['build']['root']))}
    require(record['completed'] is True and record['build_manifest_sha256'] == plan['build']['manifest_sha256']
            and set(record['evidence_files']) == expected, 'All native mapping checks must be retained')
    verify_files(root, record['evidence_files'])


def local_smoke_evidence(root, plan):
    import local_smoke
    summary = read(root/'local-smoke-summary.json')
    require(summary['completed'] is True and summary['transport'] == p.TRANSPORT
            and set(summary['arms']) == set(local_smoke.ARMS), 'All local TACO transport smokes are required')
    for arm in local_smoke.ARMS:
        dest = root/'local-smoke'/arm
        result = local_smoke.validate_reports(dest, arm, plan['build']['manifest_sha256'])
        require(result == read(dest/'report.json') == summary['arms'][arm], 'Local transport smoke report changed')
        rows = [read(dest/f'rank{rank}.json') for rank in range(8)]
        hardware = read(root/'environment-preflight.json')
        require(all(row['gpu_uuid'] == hardware['devices'][rank]['uuid'] and
                    all(row[key] == plan['environment'][key] for key in ('torch', 'cuda', 'nccl'))
                    and row['entrypoint_sha256'] == sha(HERE/'local_smoke.py')
                    and row['reference_sha256'] == sha(REPO/'tools/a800/phase4/reference.py')
                    for rank, row in enumerate(rows)), 'Local smoke hardware/software/reference identity changed')
    return summary


def collective_evidence(root, hardware):
    probe = load('collectives')
    record = hardware['collective_probe']
    expected = {f'nccl-preflight/rank{rank}.json' for rank in range(8)} | {
        'nccl-preflight/process.log', 'nccl-preflight/command.json', 'preflight-communication.json'}
    require(hardware['environment_probe_passed'] is True and hardware['collective_probe_passed'] is True
            and hardware['peer_access_required'] is False and hardware['transport'] == p.TRANSPORT
            and record['protocol'] == probe.PROTOCOL and record['completed'] is True
            and set(record['evidence_files']) == expected, 'Complete non-P2P NCCL preflight evidence required')
    verify_files(root, record['evidence_files'])
    communication = read(root/'preflight-communication.json')
    require(communication == hardware['communication'], 'Preflight communication configuration changed')
    rows = [read(root/'nccl-preflight'/f'rank{rank}.json') for rank in range(8)]
    probe.validate_records(rows, record['run_id'], hardware, communication)


def replay(root, recompute_profiles=True):
    root = Path(root)
    verify_controllers(root)
    plan, workers = read(root/'plan.json'), read(root/'workers.json')
    if plan.get('ma11_compat'):
        load('ma11_compat').verify(plan['ma11_compat'], root, plan['environment']['torch'])
    require(plan['protocol'] == p.PROTOCOL and plan['transport'] == p.TRANSPORT
            and plan['arms'] == list(p.ARMS) and plan['model'] == p.MODEL
            and plan['schedule'] == p.schedule(plan['trials'], plan['blocks']), 'Wrong nine-arm plan')
    hardware = read(root/'environment-preflight.json')
    if plan.get('ma11_compat'):
        verify_ma11_probe(root, plan['ma11_compat'], plan['ma11_probe'], hardware)
    require(sha(root/'environment-preflight.json') == plan['preflight_sha256']
            and hardware['completed'] is True and hardware['environment'] == plan['environment']
            and hardware['communication'] == plan['communication'],
            'Hardware preflight missing/changed')
    collective_evidence(root, hardware)
    smoke = read(root/'preflight.json')
    require(smoke['completed'] is True and set(smoke['arms']) == set(p.ARMS)
            and all(smoke['arms'].values()), 'All nine model smokes are required')
    profile = read(root/'profile-summary.json')
    require(profile['completed'] and profile['execution_validated'] and profile['protocol'] == p.PROTOCOL
            and profile['run_id'] == plan['run_id']
            and set(profile['profiles']) == {f'{arm}/rank{rank}' for arm in p.ARMS for rank in range(8)},
            'All nine mechanism profiles are required')
    require(all(item['execution_validated'] is True and key == f"{item['arm']}/rank{item['rank']}"
                for key, item in profile['profiles'].items()), 'Incomplete profile validation')
    profile_evidence(root, plan, profile, recompute_profiles)
    mapping_evidence(root, plan)
    local_smoke_evidence(root, plan)
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
    result.update(run_id=plan['run_id'], correctness_passed=True, execution_path_validated=True,
                  equivalent_policy_trajectories_identical=True, hardware=initial[0]['hardware'],
                  environment=plan['environment'], worker_records=records,
                  profile_summary_sha256=sha(root/'profile-summary.json'), plan_sha256=sha(root/'plan.json'),
                  build=plan['build'], calibration_origin='frozen H100 tables; not recalibrated on RTX5090')
    if plan.get('ma11_compat'):
        result['ma11_compat'] = plan['ma11_compat']
    return result


class Peer:
    def check(self):
        pass


def verify_ma11_probe(root, compat, evidence, hardware):
    path = Path(root)/'ma11-compat/self-test/probe-result.json'
    require(path.is_file() and evidence.get('result_sha256') == sha(path), 'Missing/changed MA11 GPU probe')
    result = read(path)
    require(result.get('protocol') == compat['protocol'] and result.get('completed') is True
            and result.get('performance_measured') is False and result.get('gpu') == 0
            and result.get('uuid') == hardware['devices'][0]['uuid']
            and result.get('torch') == compat['torch'] and result.get('cuda') == hardware['environment']['cuda']
            and result.get('library') == compat
            and result.get('self_test') == dict(passed=True, cases=6, forward_bitwise=True,
                                                backward_bitwise=True, rng_unchanged=True),
            'MA11 compatibility GPU probe did not pass on the preflight environment')


def ma11_gpu_preflight(args, compat, hardware):
    from tp8_dp2_train import run_process
    root = args.out
    directory = root/'ma11-compat/self-test'
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES=hardware['cuda_visible_devices'], CUDA_HOME=str(args.cuda_home))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE'):
        env.pop(key, None)
    env['PATH'] = f'{args.python.parent}:{args.cuda_home}/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(args.cuda_home/'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
    log = root/'ma11-compat/self-test-process.log'
    command = [args.python, HERE/'ma11_softmax_probe.py', '--execute', '--prepared-root', root,
               '--out', directory, '--cuda-home', args.cuda_home]
    try:
        run_process(command, REPO, env, log, 600, Peer())
    except Exception:
        path = directory/'probe-result.json'
        if path.is_file():
            print('MA11 SOFTMAX PROBE FAILED:', read(path).get('error'), file=sys.stderr, flush=True)
        raise
    evidence = dict(result_sha256=sha(directory/'probe-result.json'))
    verify_ma11_probe(root, compat, evidence, hardware)
    return evidence


def hardware_preflight(args):
    from tp8_dp2_train import run_process
    import ablation4_runtime as comm
    gpu_idle()
    communication = capture_communication()
    write(args.out/'preflight-communication.json', communication)
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'),
               CUDA_HOME=str(args.cuda_home))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE'):
        env.pop(key, None)
    env['PATH'] = f'{args.python.parent}:{args.cuda_home}/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(args.cuda_home/'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
    env = comm.apply(env, communication)
    output = args.out/'environment-preflight.json'
    try:
        run_process([args.python, HERE/'baseline_preflight.py', '--out', output,
                     '--cuda-home', args.cuda_home, '--build-cuda-home', args.build_cuda_home],
                    REPO, env, args.out/'environment-preflight.log', 300, Peer())
        hardware = read(output)
        require(hardware['environment_probe_passed'] is True and hardware['completed'] is False,
                'Environment-only check cannot complete the distributed preflight')
        probe = load('collectives')
        directory = args.out/'nccl-preflight'
        directory.mkdir()
        run_id = uuid.uuid4().hex
        command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '--max-restarts=0', HERE/'baseline_collectives.py', '--out', directory,
                   '--run-id', run_id, '--communication', args.out/'preflight-communication.json']
        write(directory/'command.json', dict(protocol=probe.PROTOCOL, run_id=run_id,
                                            command=list(map(str, command))))
        run_process(command, REPO, dict(env, RTX5090_TRANSPORT_PROBE_EXECUTE='1'),
                    directory/'process.log', 300, Peer())
        rows = [read(directory/f'rank{rank}.json') for rank in range(8)]
        probe.validate_records(rows, run_id, hardware, communication)
        names = [f'nccl-preflight/rank{rank}.json' for rank in range(8)] + [
            'nccl-preflight/process.log', 'nccl-preflight/command.json', 'preflight-communication.json']
        hardware.update(completed=True, collective_probe_passed=True, communication=communication,
                        collective_probe=dict(protocol=probe.PROTOCOL, run_id=run_id, completed=True,
                                              evidence_files={name: sha(args.out/name) for name in names}))
        write(output, hardware)
        collective_evidence(args.out, hardware)
        return hardware
    except BaseException as exc:
        hardware = read(output) if output.is_file() else {}
        hardware.update(completed=False, collective_probe_passed=False, error=f'{type(exc).__name__}: {exc}')
        write(output, hardware)
        raise


def mapping_checks(args, build, plan):
    from tp8_dp2_train import run_process
    evidence = {}
    for arm, role, native, k in mapping_jobs(build):
        dest = args.out/'mapping'/f'{arm}-{role}'
        dest.mkdir(parents=True)
        job = dict(arm=arm, mode='smoke', trial=0, block=-1, index=-1)
        env = worker_environment(args, build, args.out/'workers'/arm/'scripts', job, dest, plan)
        run_process([native/'check_mapping', '2048', '4096', str(k), '8', native/'libflux_cuda.so'],
                    REPO, env, dest/'mapping.log', 180, Peer())
        evidence[str((dest/'mapping.log').relative_to(args.out))] = sha(dest/'mapping.log')
    write(args.out/'mapping-summary.json', dict(completed=True, evidence_files=evidence,
                                               build_manifest_sha256=plan['build']['manifest_sha256']))


def print_smoke_failures(directory):
    errors = {}
    for path in sorted(directory.glob('rank*.json')):
        try:
            row = read(path)
        except (OSError, ValueError):
            continue
        if isinstance(row, dict) and isinstance(row.get('error'), str):
            errors.setdefault(row['error'], []).append(path.stem)
    for error, ranks in errors.items():
        print('LOCAL TRANSPORT FAILURE ' + ','.join(ranks) + ': ' + error, file=sys.stderr, flush=True)
    if not errors:
        path = directory/'process.log'
        if path.is_file():
            print('\n'.join(path.read_text(errors='replace').splitlines()[-80:]), file=sys.stderr, flush=True)


def worker_source_preflight():
    sources = load('sources')
    for arm in p.ARMS:
        for name, generate in (('worker', sources.worker_source), ('adapter', sources.adapter_source),
                               ('taco_support', sources.taco_source)):
            compile(generate(arm), f'{arm}/{name}.py', 'exec')


def reuse_build(args, build, stage):
    """Migrate only the registered Python defect under the campaign lock."""
    from repair_native_rank import LEGACY_SHA256, repair
    builder = load('build')
    record = read(build/'baseline-build.json')
    environment = builder.environment_identity(args.build_cuda_home)
    require(record.get('environment') == environment, 'Build Python/CUDA/ABI environment changed')
    if record.get('transport_inputs', {}).get('nccl_transport.py') == LEGACY_SHA256:
        stage('repair-native-rank')
        receipt = repair(build)
        write(args.out/'build-frontend-repair.json', receipt)
        print('FRONTEND REPAIR PASSED', receipt['backup'], flush=True)
    stage('verify-build')
    return builder.verify(build, True)


def local_smoke_checks(args, build, plan):
    import local_smoke
    from tp8_dp2_train import run_process
    summary = dict(completed=False, transport=p.TRANSPORT, arms={})
    for arm in local_smoke.ARMS:
        verify_controllers(args.out)
        gpu_idle()
        dest = args.out/'local-smoke'/arm
        dest.mkdir(parents=True)
        job = dict(arm=arm, mode='smoke', trial=0, block=-1, index=-1)
        env = worker_environment(args, build, args.out/'workers'/arm/'scripts', job, dest, plan)
        command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '--max-restarts=0', HERE/'local_smoke.py', '--build', build, '--arm', arm,
                   '--out', dest, '--execute']
        write(dest/'command.json', dict(command=list(map(str, command)), arm=arm))
        print('LOCAL TRANSPORT SMOKE', arm, flush=True)
        try:
            run_process(command, REPO, env, dest/'process.log', max(900, args.worker_timeout), Peer())
        except Exception:
            print_smoke_failures(dest)
            raise
        result = local_smoke.validate_reports(dest, arm, plan['build']['manifest_sha256'])
        require(result == read(dest/'report.json'), 'Local transport smoke disagrees with original records')
        summary['arms'][arm] = result
        write(args.out/'local-smoke-summary.json', summary)
    summary['completed'] = True
    write(args.out/'local-smoke-summary.json', summary)
    return local_smoke_evidence(args.out, plan)


def supervise(args):
    from run import arguments as model_arguments
    from tp8_dp2_train import run_process
    root = args.out
    state = read(root/'batch-state.json')

    def stage(name):
        state.update(status='running', stage=name)
        write(root/'batch-state.json', state)
        print('STAGE', name, flush=True)

    lock = None
    try:
        # One campaign per repository/host; GPU-idle checks additionally catch
        # unrelated jobs. Never kill another user's job or change GPU settings.
        lock_path = REPO/'logs/5090/.baseline.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_controllers(root)
        if not args.preflight_only:
            stage('worker-source-preflight')
            worker_source_preflight()
        stage('hardware-preflight')
        hardware = hardware_preflight(args)
        if args.preflight_only:
            state.update(status='preflight_completed', stage='done', gpu_benchmark_completed=False)
            write(root/'batch-state.json', state)
            print('PREFLIGHT PASSED', root/'environment-preflight.json', flush=True)
            return
        builder, sources, profiler = load('build'), load('sources'), load('profile')
        build = args.baseline_build or root/'build'
        if args.baseline_build:
            built = reuse_build(args, build, stage)
        else:
            stage('prepare-build')
            builder.prepare(build)
            stage('compile')
            built = builder.compile_build(build, args.megatron, args.build_cuda_home, args.jobs)
        compat = None
        compat_probe = None
        if not args.transport_smoke_only:
            stage('compile-ma11-compat')
            compat = load('ma11_compat').prepare(root, args.build_cuda_home, hardware['environment']['torch'])
            stage('ma11-compat-preflight')
            compat_probe = ma11_gpu_preflight(args, compat, hardware)
        plan = dict(protocol=p.PROTOCOL, transport=p.TRANSPORT, original_mechanisms_not_identical=True,
                    run_id=uuid.uuid4().hex, arms=list(p.ARMS), model=p.MODEL,
                    trials=args.trials, blocks=args.blocks, schedule=p.schedule(args.trials, args.blocks),
                    communication=hardware['communication'], environment=hardware['environment'],
                    cuda_visible_devices=hardware['cuda_visible_devices'],
                    preflight_sha256=sha(root/'environment-preflight.json'),
                    ma11_compat=compat,
                    ma11_probe=compat_probe,
                    build=dict(root=str(build), manifest_sha256=sha(build/'baseline-build.json')))
        write(root/'plan.json', plan)
        stage('prepare-workers')
        workers = sources.prepare_workers(build, root, plan)
        write(root/'workers.json', workers)
        stage('native-mapping-checks')
        mapping_checks(args, build, plan)
        stage('local-transport-smoke')
        local_smoke_checks(args, build, plan)
        if args.transport_smoke_only:
            state.update(status='transport_smoke_completed', stage='done', gpu_benchmark_completed=False,
                         report=str(root/'local-smoke-summary.json'))
            write(root/'batch-state.json', state)
            print('TRANSPORT SMOKE PASSED', root/'local-smoke-summary.json', flush=True)
            return
        smoke, initial, references, samples = {}, None, {}, []
        profile = dict(protocol=p.PROTOCOL, run_id=plan['run_id'], completed=False,
                       execution_validated=False, profiles={}, evidence_files={})
        for job in p.jobs(args.trials, args.blocks):
            if job['mode'] != 'smoke':
                require(set(smoke) == set(p.ARMS), 'All nine model smokes must pass before timing')
            if job['mode'] == 'timing':
                require(set(profile['profiles']) == {f'{arm}/rank{rank}' for arm in p.ARMS for rank in range(8)}
                        and all(value['execution_validated'] is True for value in profile['profiles'].values()),
                        'All nine GPU mechanism profiles must pass before formal timing')
            verify_controllers(root)
            load('ma11_compat').verify(plan['ma11_compat'], root, plan['environment']['torch'])
            verify_ma11_probe(root, plan['ma11_compat'], plan['ma11_probe'], hardware)
            require(sha(build/'baseline-build.json') == plan['build']['manifest_sha256'], 'Build identity changed')
            verify_files(build, built['files'])
            scripts = check_worker(root, job['arm'], workers[job['arm']])
            gpu_idle()
            dest = root/'results'/job['directory']
            dest.mkdir(parents=True)
            command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                       '--max-restarts=0', scripts/'worker.py', *model_arguments(p.MODEL),
                       '--distributed-timeout-minutes', '5']
            write(dest/'command.json', dict(job=job, command=list(map(str, command))))
            stage(job['directory'])
            run_process(command, build/'megatron', worker_environment(args, build, scripts, job, dest, plan),
                        dest/'process.log', args.worker_timeout, Peer())
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
        # Every trace was just analyzed above in this process; offline --report
        # also reruns the analyzers from the retained raw traces.
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
        require(result == read(args.out/'report.json'), 'Saved report differs from original rank records')
        print(p.markdown(result))
        return
    if not args.execute:
        print('PLAN ONLY: no files, builds, GPU queries or training workers. Add --execute to run.')
        print(json.dumps(dict(protocol=p.PROTOCOL, transport=p.TRANSPORT, peer_access_required=False,
                              transport_environment=p.TRANSPORT_ENV,
                              experimental=True, out=str(args.out), arms=list(p.ARMS),
                              model=p.MODEL, trials=args.trials, blocks=args.blocks,
                              timing_windows=0 if args.preflight_only or args.transport_smoke_only else args.trials*args.blocks*9,
                              formal_optimizer_steps=0 if args.preflight_only or args.transport_smoke_only else args.trials*args.blocks*9*50,
                              warmup_steps=10, timed_steps=50, python=str(args.python),
                              cuda_home=str(args.cuda_home), build_cuda_home=str(args.build_cuda_home),
                              megatron=str(args.megatron), preflight_only=args.preflight_only,
                              transport_smoke_only=args.transport_smoke_only,
                              workflow=('hardware/NCCL collectives only' if args.preflight_only else
                                        'hardware/NCCL -> build -> mappings -> local transport smoke (stop)'
                                        if args.transport_smoke_only else
                                        'hardware/NCCL -> build -> mappings -> local transport smoke -> nine model smokes '
                                        '-> nine profiles -> balanced timing -> report')), indent=2, ensure_ascii=False))
        return
    require(args.python.is_file(), 'Python missing: ' + str(args.python))
    if args.python.absolute() != Path(sys.executable).absolute():
        os.execv(str(args.python), [str(args.python), str(Path(__file__).resolve()),
                                  *(sys.argv[1:] if argv is None else argv)])
    if args.inside:
        return supervise(args)
    if not args.preflight_only and not args.baseline_build:
        require(args.megatron.is_dir(), 'Megatron source missing: ' + str(args.megatron))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'handoff').mkdir()
    write(args.out/'batch-state.json', dict(protocol=p.PROTOCOL, status='launching', stage='freeze-controllers'))
    try:
        files = controller_files()
        for name in files:
            target = args.out/'handoff/source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, target)
        write(args.out/'handoff/controllers.json', files)
        write(args.out/'handoff/launch.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        print('Batch:', args.out, flush=True)
        if args.foreground or args.preflight_only:
            return supervise(args)
        command = [str(args.python), str(Path(__file__).resolve()), '--execute', '--inside', '--out', str(args.out)]
        for name in ('python', 'megatron', 'cuda_home', 'build_cuda_home', 'trials', 'blocks', 'jobs', 'worker_timeout'):
            command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
        if args.baseline_build:
            command += ['--baseline-build', str(args.baseline_build)]
        if args.transport_smoke_only:
            command += ['--transport-smoke-only']
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
