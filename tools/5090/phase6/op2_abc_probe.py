"""Run only the three unchanged ABC native probes; never add a fourth arm."""
from pathlib import Path

from rtx5090_loader import HERE, REPO
import op1_abc_probe as original
import op1_abc_build as builder
import op2_abc_protocol as p

read, write, require, sha = p.read, p.write, p.require, p.sha


def check_records(root, hardware, plan):
    root, build = Path(root), Path(plan['build']['root'])
    groups, summaries = {}, {}
    for arm in p.ARMS:
        groups[arm] = [read(root/arm/f'rank{rank}.json') for rank in range(8)]
        summaries[arm] = original.check_basic_rows(groups[arm], arm, plan['build']['manifest_sha256'],
            hardware, builder.libraries(build, arm), builder.codec_library(build))
    for rank in range(8):
        for role in original.ROLES:
            for case in original.CASES:
                a, b = [groups[arm][rank]['roles'][role]['cases'][case] for arm in p.ARMS[:2]]
                require(all(a[key] == b[key] for key in ('input_sha256', 'weight_sha256', 'output_sha256', 'wire')),
                        'OP2 native ABC parity failed')
    return dict(protocol=p.PROTOCOL, completed=True, arms=summaries, arrival_packet_output_bitwise=True,
        build_manifest_sha256=plan['build']['manifest_sha256'],
        evidence_files={str(path.relative_to(root)): sha(path) for arm in p.ARMS
                        for path in sorted((root/arm).glob('rank*.json'))})


def native_evidence(root, plan, hardware):
    root = Path(root)
    require(check_records(root/'native-probe', hardware, plan) == read(root/'native-probe/report.json'),
            'OP2 native evidence changed')


def native_checks(args, build, plan):
    import op2_abc as controller
    from tp8_dp2_train import run_process
    for arm in p.ARMS:
        controller.base.verify_controllers(args.out)
        controller.base.gpu_idle()
        dest = args.out/'native-probe'/arm
        dest.mkdir(parents=True)
        job = dict(arm=arm, mode='smoke', trial=0, block=-1, index=-1)
        env = controller.worker_environment(args, build, args.out/'workers'/arm/'scripts', job, dest, plan)
        command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '--max-restarts=0', HERE/'op1_abc_probe.py', '--build', build,
                   '--arm', arm, '--out', dest, '--execute']
        write(dest/'command.json', dict(command=list(map(str, command))))
        print('NATIVE PROBE', arm, flush=True)
        try:
            run_process(command, REPO, env, dest/'process.log', max(900, args.worker_timeout), controller.base.Peer())
        except Exception:
            controller.base.print_smoke_failures(dest)
            raise
    write(args.out/'native-probe/report.json', check_records(args.out/'native-probe',
          read(args.out/'environment-preflight.json'), plan))
    native_evidence(args.out, plan, read(args.out/'environment-preflight.json'))
