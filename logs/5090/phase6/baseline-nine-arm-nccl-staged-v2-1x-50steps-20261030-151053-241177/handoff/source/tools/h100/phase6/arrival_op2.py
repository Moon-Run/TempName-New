#!/usr/bin/env python3
"""Order-only op2: calibrate with A's exact codec, freeze, then confirm full steps."""
import argparse
import os
from pathlib import Path
import sys

import arrival_op2_build as builder
import arrival_op2_prepare as preparation
import arrival_op2_protocol as routing
import arrival_op2_report as reporter
import arrival_op2_probe as probe
import arrival_pair_order as order
import ablation4_runtime as communication
import scope_run
from calibrate import environment
from common import REPO, read, require


def jobs(info, stage):
    return scope_run.jobs(info, stage, routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root, info, routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    root = Path(root)
    info = read(root/'submission.json')
    spans = order.validate_spans(read(root/'scripts/selection-plan.json')['schedule_options'])
    def scoped_environment(*args):
        env = communication.apply(environment(*args), info['communication'])
        lib = Path(args[1]).resolve()
        policy = next((p for p,v in info['variants'].items() if Path(v['root']).resolve() == lib), None)
        require(policy is not None, 'Unknown order-only runtime')
        for role, key in order.ENV.items():
            env[key] = str(32 if policy == routing.A else spans[role])
        return env
    return scope_run.run(root, stage, execute, python, cuda_home, routing, reporter,
                        'H100_ARRIVAL_OP2_EXECUTE', environment_fn=scoped_environment)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('build-prepare')
    q.add_argument('out', type=Path)
    q.add_argument('--parent-build', type=Path, required=True)
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
    q = sub.add_parser('probe-prepare')
    q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True)
    q.add_argument('--source-only', action='store_true')
    q = sub.add_parser('probe-run')
    q.add_argument('root', type=Path)
    q.add_argument('--python', type=Path, default=Path(sys.executable))
    q.add_argument('--cuda-home', type=Path)
    q.add_argument('--execute', action='store_true')
    q = sub.add_parser('prepare')
    q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True)
    q.add_argument('--selection', type=Path, required=True)
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
    elif args.action == 'probe-prepare':
        print(probe.prepare(args.out, args.build, args.source_only))
    elif args.action == 'probe-run':
        probe.run(args.root, execute=args.execute, python=args.python, cuda_home=args.cuda_home)
    elif args.action == 'prepare':
        print(preparation.prepare(args.out, args.build, args.megatron, args.blocks, args.rounds,
                                  args.nvls_enable, args.source_only, selection=args.selection))
    elif args.action == 'run':
        run(args.root, args.stage, args.execute, args.python, args.cuda_home)
    else:
        (reporter.preflight if args.preflight else reporter.report)(args.root, args.offline)


if __name__ == '__main__':
    main()
