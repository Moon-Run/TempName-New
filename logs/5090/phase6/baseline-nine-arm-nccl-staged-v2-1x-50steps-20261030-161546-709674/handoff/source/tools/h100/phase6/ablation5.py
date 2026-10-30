"""Budget-only ablation5: reuse C, prepare fresh sessions, explicitly execute."""
import argparse
from pathlib import Path
import sys

import ablation5_prepare as preparation
import ablation5_protocol as routing
import ablation5_report as reporter
import ablation4_runtime as communication
from calibrate import environment
from common import REPO, read
import scope_run


def jobs(info, stage):
    return scope_run.jobs(info, stage, routing)


def mapping_jobs(root, info):
    return scope_run.mapping_jobs(root, info, routing)


def run(root, stage='all', execute=False, python=Path(sys.executable), cuda_home=None):
    record = read(Path(root)/'submission.json')['communication']
    return scope_run.run(root, stage, execute, python, cuda_home, routing, reporter,
        'H100_ABLATION5_EXECUTE', environment_fn=lambda *args: communication.apply(environment(*args), record))


def main():
    parser = argparse.ArgumentParser(__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan'); p.add_argument('out', type=Path)
    p.add_argument('--candidate-build', type=Path, required=True); p.add_argument('--source-only', action='store_true')
    p = sub.add_parser('prepare'); p.add_argument('out', type=Path)
    p.add_argument('--candidate-build', type=Path, required=True); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--blocks', type=int, default=5); p.add_argument('--rounds', type=int, choices=(1, 2), default=2)
    p.add_argument('--nvls-enable', choices=('0', 'auto'), default='0'); p.add_argument('--source-only', action='store_true')
    p = sub.add_parser('run'); p.add_argument('root', type=Path)
    p.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    p.add_argument('--python', type=Path, default=Path(sys.executable)); p.add_argument('--cuda-home', type=Path)
    p.add_argument('--execute', action='store_true')
    p = sub.add_parser('report'); p.add_argument('root', type=Path)
    p.add_argument('--offline', action='store_true'); p.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.action == 'plan':
        print(preparation.prepare_plan(args.out, args.candidate_build, args.source_only))
    elif args.action == 'prepare':
        print(preparation.prepare(args.out, args.candidate_build, args.plan, args.megatron,
                                  args.blocks, args.rounds, args.nvls_enable, args.source_only))
    elif args.action == 'run':
        run(args.root, args.stage, args.execute, args.python, args.cuda_home)
    else:
        (reporter.preflight if args.preflight else reporter.report)(args.root, args.offline)


if __name__ == '__main__': main()
