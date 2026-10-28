#!/usr/bin/env python3
"""Build and compare basic fusion, improved fusion and standalone CUDA TACO."""
import argparse
from pathlib import Path
import sys

import a_baseline_build as fused_builder
import a_baseline_prepare as preparation
import a_baseline_protocol as routing
import a_baseline_report as reporter
import ablation4_runtime as communication
import ablation6_build as flux_builder
import baseline_codec_build as codec_builder
import scope_run
from ablation2_guard import check_start
from calibrate import environment
from common import REPO, read, require, sha, verify_files, write
from ma1_bench import run_worker


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None, worker_timeout=1800):
    root, python = Path(root).resolve(), Path(python).expanduser().absolute()
    require(stage in ('preflight', 'timing', 'all') and worker_timeout > 0, 'Invalid stage/worker timeout')
    info = reporter.verify_inputs(root)
    work = list(scope_run.jobs(info, stage, routing))
    command = scope_run.command(root, info, python)
    if not execute:
        print('PLAN ONLY: no workers, mappings, CUDA imports or GPU queries')
        print('COMMAND', command)
        for item in scope_run.mapping_jobs(root, info, routing):
            print('MAPPING', item)
        for job in work:
            print(job)
        return work
    require(info['executable'] and not info['source_only'], 'Source-only comparison cannot run GPU work')
    import fcntl
    import subprocess
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root/f'{stage}-state.json'
        require(not state_path.exists(), 'Stage already started; create a fresh session instead of retrying')
        write(state_path, dict(status='running', worker_timeout=worker_timeout))
        try:
            def runtime_environment(lib):
                env = communication.apply(environment(python, lib, info['config'], info['run_id'], root/'scripts', cuda_home),
                                          info['communication'])
                env.update(preparation.SHARED_ENVIRONMENT)
                return env
            if stage in ('preflight', 'all'):
                check_start(root, info)
                for policy, role, cmd in scope_run.mapping_jobs(root, info, routing):
                    lib = Path(info['variants'][policy]['root'])
                    with (root/'results'/f'mapping-{policy}-{role}.csv').open('x') as output:
                        subprocess.run(cmd, env=runtime_environment(lib), stdout=output, check=True, timeout=worker_timeout)
            gated = False
            for job in work:
                policy, mode = job['policy'], job['mode']
                if mode == 'timing' and not gated:
                    gate = reporter.preflight(root) if stage == 'all' else read(root/'preflight.json')
                    require(gate['completed'] and gate['submission_sha256'] == sha(root/'submission.json'), 'Complete preflight required')
                    verify_files(root, gate['files'])
                    gated = True
                reporter.verify_inputs(root)
                lib = Path(info['variants'][policy]['root'])
                env = runtime_environment(lib)
                folder = root/'results'/job['directory']
                folder.mkdir(parents=True, exist_ok=False)
                env.update(H100_A_BASELINE_EXECUTE='1', E2E_POLICY=policy, E2E_MODE=mode,
                           E2E_CASE='h100-phase6', E2E_BUILD=str(root/'build'), E2E_PROCESS_OUT=str(folder),
                           E2E_BLOCK=str(job['block']), E2E_WINDOW=str(job['index']),
                           PYTHONPATH=f'{root}/scripts:{info["megatron"]}:{lib}/python')
                write(folder/'command.json', dict(job, command=command, scope=routing.definition(policy),
                                                  communication=info['communication'], shared_environment=preparation.SHARED_ENVIRONMENT))
                print('START', job['directory'], flush=True)
                with (folder/'process.log').open('x') as log:
                    run_worker(command, info['megatron'], env, log, worker_timeout)
                reporter.window(root, folder, policy, mode, job['block'], job['index'])
            result = reporter.preflight(root) if stage == 'preflight' else reporter.report(root)
            write(state_path, dict(status='completed', run_id=info['run_id']))
            return result
        except BaseException as error:
            write(state_path, dict(status='failed', run_id=info['run_id'], error=f'{type(error).__name__}: {error}'))
            raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('fused-prepare', 'flux-prepare', 'codec-prepare'):
        p = sub.add_parser(name)
        p.add_argument('out', type=Path)
        p.add_argument('--source-only', action='store_true')
    for name in ('fused-compile', 'flux-compile', 'codec-compile'):
        p = sub.add_parser(name)
        p.add_argument('root', type=Path)
        p.add_argument('--cuda-home', type=Path, required=True)
        if name != 'codec-compile':
            p.add_argument('--jobs', type=int, default=2)
            p.add_argument('--nccl-root', type=Path)
    p = sub.add_parser('prepare')
    p.add_argument('out', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--flux-build', type=Path, required=True)
    p.add_argument('--codec-build', type=Path, required=True)
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--blocks', type=int, default=6)
    p.add_argument('--rounds', type=int, choices=(2,), default=2)
    p.add_argument('--nvls-enable', choices=('0',), default='0')
    p.add_argument('--scope', choices=('both',), default='both')
    p.add_argument('--source-only', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('root', type=Path)
    p.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--cuda-home', type=Path)
    p.add_argument('--worker-timeout', type=int, default=1800)
    p.add_argument('--execute', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('root', type=Path)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--offline', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    builders = {'fused': fused_builder, 'flux': flux_builder, 'codec': codec_builder}
    if args.action.endswith('-prepare'):
        print(builders[args.action.split('-')[0]].prepare(args.out, args.source_only))
    elif args.action.endswith('-compile'):
        builder = builders[args.action.split('-')[0]]
        if args.action == 'codec-compile':
            builder.compile_build(args.root, args.cuda_home)
        else:
            builder.compile_build(args.root, args.cuda_home, args.jobs, args.nccl_root)
    elif args.action == 'prepare':
        print(preparation.prepare(args.out, args.build, args.flux_build, args.codec_build, args.megatron,
                                  args.blocks, args.rounds, args.nvls_enable, args.source_only, args.scope))
    elif args.action == 'run':
        return run(args.root, args.stage, args.execute, args.python, args.cuda_home, args.worker_timeout)
    else:
        return (reporter.preflight if args.preflight else reporter.report)(args.root, offline=args.offline)


if __name__ == '__main__':
    main()
