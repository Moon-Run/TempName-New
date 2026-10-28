"""Scope experiments: default dry-run, explicit execution, no queue or retry."""
from pathlib import Path
import subprocess
import sys
import scope_protocol

from ablation_prepare import megatron_check
from ablation2_guard import check_start
from calibrate import environment
from common import read, require, sha, verify_files, write
from run import arguments


def jobs(info, stage, routing=scope_protocol):
    if stage in ('preflight', 'all'):
        for mode in ('smoke', 'profile'):
            for policy in routing.POLICIES:
                yield dict(policy=policy, mode=mode, block=-1, index=-1, directory=mode+'-'+policy)
    if stage in ('timing', 'all'):
        for job in info['schedule']:
            yield dict(job, mode='timing')


def command(root, info, python):
    return [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            str(root/'scripts/worker.py'), *arguments(info['config'])]


def mapping_jobs(root, info, routing=scope_protocol):
    for policy in routing.POLICIES:
        if not getattr(routing, 'needs_mapping', lambda p: not routing.is_native(p))(policy):
            continue
        lib = Path(info['variants'][policy]['root'])
        for role in routing.definition(policy)['roles']:
            yield policy, role, [str(lib/'check_mapping'), *map(str, routing.SHAPES[role]), '8', str(lib/'libflux_cuda.so')]


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None,
        routing=scope_protocol, reporter=None, execute_env='H100_SCOPE_EXECUTE', environment_fn=None):
    if reporter is None:
        import scope_report as reporter
    verify_inputs, preflight, window, report = (getattr(reporter, n) for n in ('verify_inputs', 'preflight', 'window', 'report'))
    definition, is_native = routing.definition, routing.is_native
    make_environment = environment if environment_fn is None else environment_fn
    require(stage in ('preflight', 'timing', 'all'), 'Unknown stage')
    root = Path(root).resolve(); info = verify_inputs(root)
    require(info['protocol'] == routing.PROTOCOL, 'Wrong runner protocol')
    tasks = list(jobs(info, stage, routing))
    if not execute:
        print('DRY RUN scope comparison: no Torch import, GPU query, mapping or worker')
        print(f'Formal windows: {info["timing_windows"]}; requested worker windows: {len(tasks)}')
        print('COMMAND', command(root, info, python))
        for item in mapping_jobs(root, info, routing): print('MAPPING', item)
        for job in tasks: print(job)
        return tasks
    require(info['executable'] and not info['source_only'], 'Source-only scope rehearsal is permanently non-executable')
    verify_files('/', info['controllers']); megatron_check(info['megatron'])
    import fcntl
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = root/f'{stage}-state.json'
        require(not state.exists(), 'Stage already started; preserve evidence and prepare a new session')
        write(state, dict(status='running'))
        try:
            if stage in ('preflight', 'all'):
                check_start(root, info)
                for policy, role, cmd in mapping_jobs(root, info, routing):
                    lib = Path(info['variants'][policy]['root'])
                    env = make_environment(python, lib, info['config'], info['run_id'], root/'scripts', cuda_home)
                    with (root/'results'/f'mapping-{policy}-{role}.csv').open('x') as f:
                        subprocess.run(cmd, env=env, stdout=f, check=True)
            gated = False
            for job in tasks:
                policy, mode = job['policy'], job['mode']
                if mode == 'timing' and not gated:
                    gate = preflight(root) if stage == 'all' else read(root/'preflight.json')
                    require(gate['completed'] and gate['submission_sha256'] == sha(root/'submission.json'), 'Full preflight required')
                    verify_files(root, gate['files']); gated = True
                verify_inputs(root)
                lib = root/'no-flux-library' if is_native(policy) else Path(info['variants'][policy]['root'])
                env = make_environment(python, lib, info['config'], info['run_id'], root/'scripts', cuda_home)
                folder = root/'results'/job['directory']; folder.mkdir(parents=True, exist_ok=False)
                env.update({execute_env: '1'})
                env.update(E2E_POLICY=policy, E2E_MODE=mode, E2E_CASE='h100-phase6',
                           E2E_BUILD=str(root/'build'), E2E_PROCESS_OUT=str(folder),
                           E2E_BLOCK=str(job['block']), E2E_WINDOW=str(job['index']),
                           PYTHONPATH=f'{root}/scripts:{info["megatron"]}'+('' if is_native(policy) else f':{lib}/python'))
                cmd = command(root, info, python)
                metadata = dict(job, command=cmd, scope=definition(policy))
                if 'communication' in info:
                    metadata['communication'] = info['communication']
                write(folder/'command.json', metadata)
                print('START', job['directory'], flush=True)
                with (folder/'process.log').open('x') as f:
                    subprocess.run(cmd, cwd=info['megatron'], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
                window(root, folder, policy, mode, job['block'], job['index'])
            (preflight if stage == 'preflight' else report)(root)
            write(state, dict(status='completed'))
        except BaseException:
            write(state, dict(status='failed')); raise
