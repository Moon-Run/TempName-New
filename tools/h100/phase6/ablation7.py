"""Three-group C/A/improved-B ablation: explicit GPU execution, frozen 10+50 steps."""
import argparse
from pathlib import Path
import sys

import ablation7_build as builder
import ablation7_prepare as preparation
import ablation7_protocol as routing
import ablation7_report as reporter
import ablation4_runtime as communication
import scope_run
from calibrate import environment
from common import REPO, read


def jobs(info, stage):
    return scope_run.jobs(info, stage, routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root, info, routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    record = read(Path(root)/'submission.json')['communication']
    return scope_run.run(root, stage, execute, python, cuda_home, routing, reporter,
        'H100_ABLATION7_EXECUTE',
        environment_fn=lambda *args: communication.apply(environment(*args), record))


def parser():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('build-prepare')
    q.add_argument('out', type=Path)
    q.add_argument('--parent-build', type=Path, required=True, help='Verified original ablation4 build; never modified')
    q.add_argument('--source-only', action='store_true')
    q = sub.add_parser('compile')
    q.add_argument('build', type=Path)
    q.add_argument('--cuda-home', type=Path, required=True)
    q.add_argument('--jobs', type=int, default=2)
    q.add_argument('--nccl-root', type=Path)
    q = sub.add_parser('compile-check')
    q.add_argument('build', type=Path)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--cuda-home', type=Path, required=True)
    q.add_argument('--cxx', default='g++')
    q = sub.add_parser('prepare')
    q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True)
    q.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    q.add_argument('--blocks', type=int, default=6)
    q.add_argument('--rounds', type=int, choices=(1, 2), default=2)
    q.add_argument('--nvls-enable', choices=('0', 'auto'), default='0')
    q.add_argument('--source-only', action='store_true')
    q = sub.add_parser('run')
    q.add_argument('root', type=Path)
    q.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    q.add_argument('--python', type=Path, default=Path(sys.executable))
    q.add_argument('--cuda-home', type=Path)
    q.add_argument('--execute', action='store_true')
    q = sub.add_parser('report')
    q.add_argument('root', type=Path)
    q.add_argument('--preflight', action='store_true')
    q.add_argument('--offline', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.action == 'build-prepare':
        print(builder.prepare(args.out, args.parent_build, args.source_only))
    elif args.action == 'compile':
        builder.compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)
    elif args.action == 'compile-check':
        builder.compile_check(args.build, args.out, args.cuda_home, args.cxx)
    elif args.action == 'prepare':
        print(preparation.prepare(args.out, args.build, args.megatron,
                                  args.blocks, args.rounds, args.nvls_enable, args.source_only))
    elif args.action == 'run':
        run(args.root, args.stage, args.execute, args.python, args.cuda_home)
    else:
        (reporter.preflight if args.preflight else reporter.report)(args.root, args.offline)


if __name__ == '__main__':
    main()
