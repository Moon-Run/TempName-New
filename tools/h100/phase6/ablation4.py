"""Five MLP+Attention groups; CPU preparation by default, GPU only with --execute."""
import argparse
import contextlib
import os
from pathlib import Path
import shutil
import sys

import ablation4_build as builder
import ablation4_prepare as preparation
import ablation4_report as reporter
import ablation4_run as runner
import ablation4_runtime as communication
import build
import scope_build
import scope_calibrate
from common import HERE, REPO, read, require, replace_once, sha, write


def calibration_prepare(out, build_root, source_only=False, nvls='0'):
    root = scope_calibrate.prepare(out, build_root, source_only)
    comm = communication.capture(nvls)
    cfg = read(root/'config.json'); cfg['communication'] = comm; write(root/'config.json', cfg)
    shutil.copy2(HERE/'ablation4_runtime.py', root/'communication.py')
    worker = (root/'worker.py').read_text()
    worker = replace_once(worker, 'import torch\n',
                          "from communication import verify as verify_communication\n"
                          "communication = verify_communication(json.loads(Path(__file__).with_name('config.json').read_text())['communication'])\n"
                          'import torch\n')
    worker = replace_once(worker, "report.update(protocol=config['protocol']", "report.update(communication=communication,protocol=config['protocol']")
    (root/'worker.py').write_text(worker)
    plan = read(root/'plan.json'); plan['communication'] = comm
    for name in ('config.json', 'worker.py', 'communication.py'):
        plan['files'][name] = sha(root/name)
    write(root/'plan.json', plan)
    return root


@contextlib.contextmanager
def communication_environment(record):
    prior = {k: v for k, v in os.environ.items() if communication.relevant(k)}
    try:
        for k in list(os.environ):
            if communication.relevant(k): del os.environ[k]
        os.environ.update(record['environment'])
        communication.verify(record)
        yield
    finally:
        for k in list(os.environ):
            if communication.relevant(k): del os.environ[k]
        os.environ.update(prior)


def main():
    parser = argparse.ArgumentParser(__doc__); sub = parser.add_subparsers(dest='action', required=True)
    q = sub.add_parser('calibration-build'); q.add_argument('out', type=Path)
    q = sub.add_parser('calibration-prepare'); q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True); q.add_argument('--source-only', action='store_true')
    q.add_argument('--nvls-enable', choices=('0', 'auto'), default='0')
    q = sub.add_parser('fit'); q.add_argument('calibration', type=Path); q.add_argument('out', type=Path)
    q.add_argument('--arrival-order', choices=('tail_residual', 'tail_assignment', 'legacy_tail'), default='legacy_tail',
                   help='Keep the historical default; explicitly opt into residual swaps or joint tail assignment')
    q.add_argument('--arrival-margin-multiplier', type=float,
                   help='Explicit tail_assignment v3 margin in [1,6]; values below 4 are exploratory')
    q = sub.add_parser('build-prepare'); q.add_argument('out', type=Path)
    q.add_argument('--attn-plan', type=Path); q.add_argument('--mlp-plan', type=Path)
    q.add_argument('--source-only', action='store_true')
    for action in ('compile', 'compile-check'):
        q = sub.add_parser(action); q.add_argument('build', type=Path)
        q.add_argument('--cuda-home', type=Path, required=True)
        if action == 'compile':
            q.add_argument('--jobs', type=int, default=2); q.add_argument('--nccl-root', type=Path)
        else:
            q.add_argument('--out', type=Path, required=True); q.add_argument('--cxx', default='g++')
    q = sub.add_parser('prepare'); q.add_argument('out', type=Path)
    q.add_argument('--build', type=Path, required=True)
    q.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    q.add_argument('--blocks', type=int, default=5); q.add_argument('--rounds', type=int, choices=(1, 2), default=1)
    q.add_argument('--source-only', action='store_true'); q.add_argument('--nvls-enable', choices=('0', 'auto'), default='0')
    for action in ('calibration-run', 'run'):
        q = sub.add_parser(action); q.add_argument('root', type=Path)
        q.add_argument('--execute', action='store_true'); q.add_argument('--cuda-home', type=Path)
        q.add_argument('--python', type=Path, default=Path(sys.executable))
        if action == 'run': q.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    q = sub.add_parser('report'); q.add_argument('root', type=Path)
    q.add_argument('--preflight', action='store_true'); q.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.action == 'calibration-build':
        print(scope_build.prepare(args.out, sampler=True, roles=('attn', 'mlp')))
    elif args.action == 'calibration-prepare':
        print(calibration_prepare(args.out, args.build, args.source_only, args.nvls_enable))
    elif args.action == 'calibration-run':
        if args.execute:
            record = read(args.root/'plan.json')['communication']
            with communication_environment(record):
                scope_calibrate.run(args.root, True, args.python, args.cuda_home)
        else:
            scope_calibrate.run(args.root, False, args.python, args.cuda_home)
    elif args.action == 'fit':
        options = {}
        if args.arrival_margin_multiplier is not None:
            from arrival_assignment import normalize_options
            require(args.arrival_order == 'tail_assignment', 'Margin options require tail_assignment')
            options['arrival_options'] = normalize_options(dict(uncertainty_multiplier=args.arrival_margin_multiplier))
        plan = scope_calibrate.verify(args.calibration)
        for row in scope_calibrate.calibration_records(args.calibration, plan):
            require(row.get('communication') == plan['communication'], 'Calibration communication changed')
        print(scope_calibrate.fit(args.calibration, args.out, arrival_order=args.arrival_order, **options))
    elif args.action == 'build-prepare':
        require(args.source_only or (args.attn_plan is not None and args.mlp_plan is not None),
                'Ablation4 requires both explicit measured role plans on this host')
        print(builder.prepare(args.out, args.attn_plan, args.mlp_plan, args.source_only))
    elif args.action in ('compile', 'compile-check'):
        suite = (args.build/'ablation4-manifest.json').is_file()
        if not suite: scope_build.verify(args.build, kind='scope_sampler')
        if args.action == 'compile':
            require(args.jobs > 0, 'Positive build jobs required')
            (builder.compile_build if suite else build.compile_build)(args.build, args.cuda_home, args.jobs, args.nccl_root)
        else:
            (builder.compile_check if suite else scope_build.compile_check)(args.build, args.out, args.cuda_home, args.cxx)
    elif args.action == 'prepare':
        print(preparation.prepare(args.out, args.build, args.megatron, args.blocks, args.source_only,
                                  args.rounds, args.nvls_enable))
    elif args.action == 'run':
        runner.run(args.root, args.stage, args.execute, args.python, args.cuda_home)
    elif args.action == 'report':
        (reporter.preflight if args.preflight else reporter.report)(args.root, args.offline)


if __name__ == '__main__':
    main()
