"""Improve arrival ordering on the original fused path; all GPU runs are explicit."""
import argparse
from pathlib import Path
import sys

from fused_support import REPO
import fused_build
import fused_prepare
import fused_run
import fused_report

DEFAULT = REPO / 'outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236/plans/measured'


def main():
    p = argparse.ArgumentParser(__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    q = commands.add_parser('build-prepare'); q.add_argument('out', type=Path)
    q.add_argument('--attn-plan', type=Path); q.add_argument('--mlp-plan', type=Path)
    q.add_argument('--source-only', action='store_true')
    q = commands.add_parser('verify'); q.add_argument('root', type=Path); q.add_argument('--built', action='store_true')
    q.add_argument('--refit', action='store_true', help='Recompute the fitted order from the original hashed receiver records')
    q = commands.add_parser('compile'); q.add_argument('root', type=Path)
    q.add_argument('--cuda-home', type=Path, required=True); q.add_argument('--jobs', type=int, default=2)
    q.add_argument('--nccl-root', type=Path)
    q = commands.add_parser('compile-check'); q.add_argument('root', type=Path)
    q.add_argument('--out', type=Path, required=True); q.add_argument('--cuda-home', type=Path, required=True)
    q.add_argument('--cxx', default='g++')
    q = commands.add_parser('prepare'); q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True); q.add_argument('--megatron', type=Path, default=REPO.parent / 'Megatron-LM')
    q.add_argument('--source-only', action='store_true'); q.add_argument('--blocks', type=int, default=9)
    q.add_argument('--rounds', type=int, choices=(1, 2), default=2,
                   help='Formal rounds: 1 keeps the balanced first round; default 2 includes its reversed repeat')
    q = commands.add_parser('run'); q.add_argument('root', type=Path)
    q.add_argument('--stage', choices=('all', 'preflight', 'timing'), default='all')
    q.add_argument('--execute', action='store_true'); q.add_argument('--python', type=Path, default=Path(sys.executable))
    q.add_argument('--cuda-home', type=Path)
    q = commands.add_parser('report'); q.add_argument('root', type=Path)
    q.add_argument('--preflight', action='store_true'); q.add_argument('--offline', action='store_true')
    a = p.parse_args()
    if a.action == 'build-prepare':
        attn = a.attn_plan or (None if a.source_only else DEFAULT / 'attn.json')
        mlp = a.mlp_plan or (None if a.source_only else DEFAULT / 'mlp.json')
        print(fused_build.prepare(a.out, attn, mlp, a.source_only))
    elif a.action == 'verify':
        fused_build.verify(a.root, a.built)
        if a.refit:
            import fused_plan
            from common import read
            fused_plan.verify_fit(read(a.root / 'plans.json'))
        print('Fused source, mappings, plans and masks verified')
    elif a.action == 'compile':
        fused_build.compile_build(a.root, a.cuda_home, a.jobs, a.nccl_root)
    elif a.action == 'compile-check':
        fused_build.compile_check(a.root, a.out, a.cuda_home, a.cxx)
    elif a.action == 'prepare':
        print(fused_prepare.prepare(a.out, a.build, a.megatron, a.blocks, a.source_only, rounds=a.rounds))
    elif a.action == 'run':
        fused_run.run(a.root, a.stage, a.execute, a.python, a.cuda_home)
    else:
        (fused_report.preflight if a.preflight else fused_report.report)(a.root, a.offline)


if __name__ == '__main__':
    main()
