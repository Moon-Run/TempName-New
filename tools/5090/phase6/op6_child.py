#!/usr/bin/env python3
"""Run the frozen OP3 workflow with an explicit OP6 comparison variant."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import op3
import op3_protocol as p

PROTOCOL = 'rtx5090-phase6-op6-local-attention-graph-paired-v1'
VARIANTS = ('op4', 'op6')


@contextmanager
def configured(variant):
    p.require(variant in VARIANTS, 'Unknown OP6 child variant')
    import op6_sources
    import op6_validation
    import op6_profile
    restore = []
    original_write = op3.write
    def write(path, value):
        if Path(path).name == 'plan.json':
            # OP3 retains this object for worker generation after writing it.
            value.update(op6_variant=variant, op6_protocol=PROTOCOL)
        return original_write(path, value)
    try:
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
    with configured(plan['op6_variant']):
        return op3.replay(root, recompute_profiles=recompute_profiles)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, add_help=False)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args, remaining = parser.parse_known_args(argv)
    p.require('--execute' not in remaining or '--foreground' in remaining,
              'OP6 child execution requires --foreground; the parent owns supervision')
    if '--report' in remaining:
        # op3.main uses its own replay global. Validate the variant before it can
        # select hooks, including when this private CLI is invoked directly.
        read_args = op3.arguments(remaining)
        plan = p.read(read_args.out/'plan.json')
        p.require(plan.get('op6_protocol') == PROTOCOL and plan.get('op6_variant') == args.variant,
                  'Requested OP6 variant differs from saved child')
    with configured(args.variant):
        op3.main(remaining)


if __name__ == '__main__':
    main()
