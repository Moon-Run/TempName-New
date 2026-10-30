"""Ablation dry-run by default. Only --execute can start GPU mappings/workers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from ablation_plan import definitions, experiment, PROTOCOL
from ablation_prepare import megatron_check
from ablation_report import preflight, report, verify_inputs, window
from calibrate import environment
from common import read, require, sha, shape, verify_files, write
from run import arguments


def jobs(info, stage):
    definitions = experiment(info['protocol']).definitions
    if stage in ('preflight', 'all'):
        for base in info['bases']:
            for mode in ('smoke', 'profile'):
                for key in definitions(base, info['seeds']):
                    yield dict(base=base, policy=key, mode=mode, block=-1, index=-1,
                               directory=f'{base}/{mode}-{key}')
    if stage in ('timing', 'all'):
        for job in info['schedule']:
            yield dict(job, mode='timing')


def command(root, info, base, python):
    return [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            str(root/'scripts'/base/'worker.py'), *arguments(info['config'])]


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    root = Path(root).resolve()
    info = verify_inputs(root)
    tasks = list(jobs(info, stage))
    if not execute:
        print('DRY RUN: no CUDA import/initialization, no GPU mapping, no workers. Pass --execute on H100.')
        print(f'Formal windows: {info["timing_windows"]}; requested worker windows: {len(tasks)}')
        for base in info['bases']:
            print('COMMAND', command(root, info, base, python))
        for job in tasks:
            print(job)
        return tasks
    require(info['executable'] and not info['source_only'], 'Source-only rehearsal is permanently non-executable; prepare a new run after compilation')
    verify_files('/', info['controllers'])
    megatron_check(Path(info['megatron']))
    import fcntl
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = root/f'{stage}-state.json'
        require(not state.exists(), 'Stage already started; preserve it and prepare a new independent run')
        write(state, dict(status='running', stage=stage))
        try:
            if info['protocol'] != PROTOCOL and stage in ('preflight', 'all'):
                from ablation2_guard import check_start
                check_start(root, info)
            if stage in ('preflight', 'all'):
                for identity, v in info['variants'].items():
                    if v['group'] in ('native', 'native_taco'):
                        continue
                    library = Path(v['root']); base = v['base']
                    directory = root/'results'/base
                    directory.mkdir(exist_ok=True)
                    target = directory/f'mapping-{v["policy"]}.csv'
                    env = environment(python, library, info['config'], info['run_id'], root/'scripts'/base, cuda_home)
                    with target.open('x') as f:
                        subprocess.run([str(library/'check_mapping'), *map(str, shape(info['config'])), '8',
                                        str(library/'libflux_cuda.so')], env=env, stdout=f, check=True)
            gated = False
            for job in tasks:
                base, key, mode = job['base'], job['policy'], job['mode']
                if mode == 'timing' and not gated:
                    gate = preflight(root) if stage == 'all' else read(root/'preflight.json')
                    require(gate['completed'] and gate['submission_sha256'] == sha(root/'submission.json'), 'Full preflight required')
                    verify_files(root, gate['files']); gated = True
                verify_inputs(root)
                v = info['variants'][base+'-'+key]
                library = Path(v['root']) if key not in ('native', 'native_taco') else root/'no-flux-library'
                scripts = root/'scripts'/base
                env = environment(python, library, info['config'], info['run_id'], scripts, cuda_home)
                directory = root/'results'/job['directory']
                directory.mkdir(parents=True, exist_ok=False)
                env.update(H100_ABLATION_EXECUTE='1', E2E_POLICY=key, E2E_MODE=mode, E2E_CASE='h100-phase6',
                           E2E_BUILD=str(root/'build'/base), E2E_PROCESS_OUT=str(directory),
                           E2E_BLOCK=str(job['block']), E2E_WINDOW=str(job['index']),
                           PYTHONPATH=f'{scripts}:{info["megatron"]}'+(f':{library}/python' if key not in ('native', 'native_taco') else ''))
                cmd = command(root, info, base, python)
                write(directory/'command.json', dict(job, command=cmd,
                      definition=experiment(info['protocol']).definitions(base, info['seeds'])[key]))
                print('START', job['directory'], flush=True)
                with (directory/'process.log').open('x') as log:
                    subprocess.run(cmd, cwd=info['megatron'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                window(root, directory, base, key, mode, job['block'], job['index'])
            if stage == 'preflight':
                preflight(root)
            else:
                report(root)
            write(state, dict(status='completed', stage=stage))
        except BaseException:
            write(state, dict(status='failed', stage=stage))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path); p.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    p.add_argument('--execute', action='store_true'); p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--cuda-home', type=Path)
    a = p.parse_args()
    run(a.root, a.stage, a.execute, a.python, a.cuda_home)
