"""Execution timing is independent of the immutable, previously built model.

New ablation4/5 preparations use 10 warmup + 60 timed optimizer steps.
Historical submissions without execution_timing retain their sealed 20 steps.
"""
from pathlib import Path

from common import read, require

WARMUP_STEPS = 10
TIMED_STEPS = 60


def execution_timing():
    return dict(warmup_steps=WARMUP_STEPS, timed_steps=TIMED_STEPS)


def validate(root, info):
    root = Path(root)
    cfg = read(root/'scripts/config.json')
    seal = read(root/'provenance/preparation.json')
    if 'execution_timing' in info or 'execution_timing' in seal:
        expected = execution_timing()
        require(info.get('execution_timing') == seal.get('execution_timing') == expected,
                'Execution timing seal changed; new ablation4/5 runs require 60 timed steps')
    else:
        expected = dict(warmup_steps=10, timed_steps=20)
    require(all(type(cfg.get(k)) is int and cfg[k] == v for k, v in expected.items()),
            'Worker warmup/timed steps differ from sealed execution timing')
    return expected


def require_new(info):
    require(info.get('execution_timing') == execution_timing(),
            'Repeat batches require fresh 10-warmup / 60-timed-step submissions')


def require_report(result, windows):
    require(result.get('execution_timing') == execution_timing() and
            result.get('formal_optimizer_steps') == windows*TIMED_STEPS,
            'Repeat batches require complete 60-step reports')
