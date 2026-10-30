"""Nine-group scope ablation. Preparation is CPU only; GPU runs require --execute."""
import argparse
from pathlib import Path
import sys

import build
import ablation3_build
import ablation3_prepare
import ablation3_run
import ablation3_report
import scope_build
import scope_calibrate
from common import REPO, require, write


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('calibration-build'); q.add_argument('out', type=Path)
    q.add_argument('--roles', nargs='+', choices=('attn', 'mlp'), default=['attn', 'mlp'])
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
    q.add_argument('--build', type=Path, required=True)
    q.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    q.add_argument('--blocks', type=int, default=9); q.add_argument('--source-only', action='store_true')
    q.add_argument('--rounds', type=int, choices=(1, 2), default=2,
                   help='Number of timing rounds; one keeps the original first-round schedule (default: two)')
    for action in ('calibration-run', 'run'):
        q = sub.add_parser(action); q.add_argument('root', type=Path)
        q.add_argument('--execute', action='store_true'); q.add_argument('--cuda-home', type=Path)
        q.add_argument('--python', type=Path, default=Path(sys.executable))
        if action == 'run':
            q.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    q = sub.add_parser('report'); q.add_argument('root', type=Path)
    q.add_argument('--preflight', action='store_true'); q.add_argument('--offline', action='store_true')
    a = p.parse_args()
    if a.action == 'calibration-build':
        print(scope_build.prepare(a.out, sampler=True, roles=a.roles))
    elif a.action == 'build-prepare':
        print(ablation3_build.prepare(a.out, a.attn_plan, a.mlp_plan, a.source_only))
    elif a.action in ('compile', 'compile-check'):
        if (a.build/'ablation3-manifest.json').is_file():
            ablation3_build.verify(a.build)
            roots = {order: a.build/order for order in ablation3_build.ORDERS}
        else:
            scope_build.verify(a.build, kind='scope_sampler')
            roots = {'sampler': a.build}
        if a.action == 'compile':
            require(a.jobs > 0, 'Positive build jobs required')
            for root in roots.values():
                if scope_build.verify(root)['status'] != 'built':
                    build.compile_build(root, a.cuda_home, a.jobs, a.nccl_root)
        else:
            a.out.mkdir(parents=True, exist_ok=False)
            checks = {name: scope_build.compile_check(root, a.out/name, a.cuda_home, a.cxx) for name, root in roots.items()}
            write(a.out/'validation.json', dict(completed=True, gpu_used=False, gpu_validated=False, checks=checks))
    elif a.action == 'calibration-prepare':
        print(scope_calibrate.prepare(a.out, a.build, a.source_only))
    elif a.action == 'calibration-run':
        scope_calibrate.run(a.root, a.execute, a.python, a.cuda_home)
    elif a.action == 'fit':
        print(scope_calibrate.fit(a.calibration, a.out))
    elif a.action == 'prepare':
        print(ablation3_prepare.prepare(a.out, a.build, a.megatron, a.blocks, a.source_only, rounds=a.rounds))
    elif a.action == 'run':
        ablation3_run.run(a.root, a.stage, a.execute, a.python, a.cuda_home)
    elif a.action == 'report':
        (ablation3_report.preflight if a.preflight else ablation3_report.report)(a.root, a.offline)


if __name__ == '__main__':
    main()
