#!/usr/bin/env python3
"""Run the frozen OP3 workflow with an explicit OP6 comparison variant."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import op3
import op3_protocol as p
import op6_controllers
import op6_memory

PROTOCOL = 'rtx5090-phase6-op6-local-attention-graph-paired-v1'
VARIANTS = ('op4', 'op6')


@contextmanager
def configured(variant, reserve_mib=0):
    p.require(variant in VARIANTS, 'Unknown OP6 child variant')
    import op6_sources
    import op6_validation
    import op6_profile
    restore = []
    original_write = op3.write
    def write(path, value):
        if Path(path).name == 'plan.json':
            # OP3 retains this object for worker generation after writing it.
            value.update(op6_variant=variant, op6_protocol=PROTOCOL, op6_reserve_mib=reserve_mib)
        return original_write(path, value)
    try:
        restore.append(op6_controllers.install(op3.base))
        for module in (op6_sources, op6_validation, op6_profile):
            restore.append(module.install(variant))
        op3.write = write
        yield
    finally:
        op3.write = original_write
        for undo in reversed(restore):
            undo()


def replay(root, recompute_profiles=True):
    root = Path(root)
    plan = p.read(root/'plan.json')
    p.require(plan.get('op6_protocol') == PROTOCOL and plan.get('op6_variant') in VARIANTS,
              'Missing OP6 child protocol/variant')
    with configured(plan['op6_variant'], plan.get('op6_reserve_mib', 0)):
        return op3.replay(root, recompute_profiles=recompute_profiles)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, add_help=False)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--reserve-mib', type=int, default=0)
    args, remaining = parser.parse_known_args(argv)
    p.require(0 <= args.reserve_mib < 32768, 'Use --reserve-mib 0..32767')
    p.require('--execute' not in remaining or '--foreground' in remaining,
              'OP6 child execution requires --foreground; the parent owns supervision')
    if '--report' in remaining:
        # op3.main uses its own replay global. Validate the variant before it can
        # select hooks, including when this private CLI is invoked directly.
        read_args = op3.arguments(remaining)
        plan = p.read(read_args.out/'plan.json')
        p.require(plan.get('op6_protocol') == PROTOCOL and plan.get('op6_variant') == args.variant,
                  'Requested OP6 variant differs from saved child')
    original_supervise = op3.supervise

    def supervise(child_args):
        # OP3 main has already created/frozen the new output directory. Starting
        # the holder before that would create handoff logs and break exist_ok=False.
        with op6_memory.session(child_args.python, child_args.out, args.reserve_mib,
                                op3.base.gpu_idle) as reservation:
            with op6_memory.child_handoff(reservation, op3.base):
                return original_supervise(child_args)

    try:
        if '--execute' in remaining:
            op3.supervise = supervise
        with configured(args.variant, args.reserve_mib):
            op3.main(remaining)
    finally:
        op3.supervise = original_supervise


if __name__ == '__main__':
    main()
