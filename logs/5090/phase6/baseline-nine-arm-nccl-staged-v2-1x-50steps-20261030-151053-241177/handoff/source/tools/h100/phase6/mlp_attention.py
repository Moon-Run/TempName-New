"""Independent Attention/MLP scope experiment. All run commands default to dry-run."""
import argparse
from pathlib import Path
import sys

import build
from common import REPO, require
import scope_build
import scope_calibrate
import scope_prepare
import scope_run
import scope_report


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('calibration-build'); q.add_argument('out', type=Path)
    q.add_argument('--roles', nargs='+', choices=('attn', 'mlp'), default=['attn'])
    q = sub.add_parser('build-prepare'); q.add_argument('out', type=Path)
    q.add_argument('--attn-plan', type=Path); q.add_argument('--mlp-plan', type=Path)
    q.add_argument('--source-only', action='store_true')
    q = sub.add_parser('compile'); q.add_argument('build', type=Path)
    q.add_argument('--cuda-home', type=Path, required=True); q.add_argument('--jobs', type=int, default=2)
    q.add_argument('--nccl-root', type=Path)
    q = sub.add_parser('compile-check'); q.add_argument('build', type=Path)
    q.add_argument('--out', type=Path, required=True); q.add_argument('--cuda-home', type=Path, required=True)
    q.add_argument('--cxx', default='g++')
    q = sub.add_parser('calibration-prepare'); q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True); q.add_argument('--source-only', action='store_true')
    q = sub.add_parser('fit'); q.add_argument('calibration', type=Path); q.add_argument('out', type=Path)
    q = sub.add_parser('prepare'); q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True); q.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    q.add_argument('--blocks', type=int, default=4); q.add_argument('--source-only', action='store_true')
    for action in ('calibration-run', 'run'):
        q = sub.add_parser(action); q.add_argument('root', type=Path)
        q.add_argument('--execute', action='store_true'); q.add_argument('--cuda-home', type=Path)
        q.add_argument('--python', type=Path, default=Path(sys.executable))
        if action == 'run': q.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    q = sub.add_parser('report'); q.add_argument('root', type=Path)
    q.add_argument('--preflight', action='store_true'); q.add_argument('--offline', action='store_true')
    a = p.parse_args()
    if a.action == 'calibration-build':
        print(scope_build.prepare(a.out, sampler=True, roles=a.roles))
    elif a.action == 'build-prepare':
        print(scope_build.prepare(a.out, a.attn_plan, a.mlp_plan, a.source_only))
    elif a.action == 'compile':
        m = scope_build.verify(a.build); require(a.jobs > 0, 'Positive build jobs required')
        if m['status'] != 'built': build.compile_build(a.build, a.cuda_home, a.jobs, a.nccl_root)
    elif a.action == 'compile-check':
        scope_build.compile_check(a.build, a.out, a.cuda_home, a.cxx)
    elif a.action == 'calibration-prepare':
        print(scope_calibrate.prepare(a.out, a.build, a.source_only))
    elif a.action == 'calibration-run':
        scope_calibrate.run(a.root, a.execute, a.python, a.cuda_home)
    elif a.action == 'fit':
        print(scope_calibrate.fit(a.calibration, a.out))
    elif a.action == 'prepare':
        print(scope_prepare.prepare(a.out, a.build, a.megatron, a.blocks, a.source_only))
    elif a.action == 'run':
        scope_run.run(a.root, a.stage, a.execute, a.python, a.cuda_home)
    elif a.action == 'report':
        (scope_report.preflight if a.preflight else scope_report.report)(a.root, a.offline)


if __name__ == '__main__':
    main()
