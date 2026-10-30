"""Fixed interleaved five-group protocol; all helpers are CPU only."""
import random

from ablation_plan import FREEZE_ID, GROUPS, derive
from common import require

PROTOCOL = 'h100-gpt67-tp8-ablation2-v1'
BASES = ('interleaved',)
POLICIES = ('native_taco', 'G1', 'G2', 'G3', 'G4')


def definitions(base='interleaved', seeds=()):
    require(base == 'interleaved' and not seeds, 'Ablation2 requires interleaved and no G5 seeds')
    result = {'native_taco': dict(group='native_taco', base=base, order='native', arrival=False,
              quantization_mode='native_tensor_taco', seed=None, implementation='native_megatron_tensor_taco',
              parent_freeze=FREEZE_ID)}
    for key in POLICIES[1:]:
        _, arrival, mode = GROUPS[key]
        result[key] = dict(group=key, base=base, order=base, arrival=arrival, quantization_mode=mode,
                           seed=None, implementation='frozen-candidate-v1-sm90-gemmv2', parent_freeze=FREEZE_ID)
    return result


def schedule(base, definitions_, blocks=5, seed=20261013):
    require(base == 'interleaved' and definitions_ == definitions(), 'Wrong five-group definitions')
    require(type(blocks) is int and blocks > 0 and blocks % 5 == 0, 'Blocks must be a positive multiple of 5')
    require(type(seed) is int and seed == 20261013, 'Ablation2 freezes schedule seed 20261013')
    row = list(POLICIES)
    random.Random(seed).shuffle(row)
    first = [row[i % 5:]+row[:i % 5] for i in range(blocks)]
    result = []
    for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]]), 1):
        for block, order in enumerate(rows):
            for position, key in enumerate(order):
                index = block*5+position
                result.append(dict(base=base, round=rep, block=block, index=index, policy=key,
                                   directory=f'{base}/round{rep}/window-{index:03d}-{key}'))
    return result
