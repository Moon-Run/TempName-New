"""Freeze a bounded critical-tail selection from training passes 0/1 only."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('arrival_plan', HERE.parent/'arrival/plan.py')
arrival = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arrival)


def select(scores, world, fraction):
    assert 0 <= fraction <= 1 and len(scores) == world
    tiles = len(scores[0])
    assert tiles % world == 0 and all(len(row) == tiles for row in scores)
    limit = math.floor(tiles*(world-1)/world*fraction)
    result = [[0]*tiles for _ in range(world)]
    for source in range(world):
        candidates = [t for t in range(tiles) if t//(tiles//world) != source and scores[source][t] > 0]
        for tile in sorted(candidates, key=lambda t: (-scores[source][t], t))[:limit]:
            result[source][tile] = 1
    return result


def fit(reports, fraction=.25):
    world, tiles = 4, 256
    scores = [[0.]*tiles for _ in range(world)]
    observed = {}
    valid = 0
    overhead = []
    for target, report in enumerate(reports):
        sh = next(s for s in report['shapes'] if (s['M'], s['N'], s['K_global']) == (2048, 2048, 8192))
        assert sh['partition_tiles'] == tiles//world
        obs = arrival.observations(sh, target, world, relative=True)
        observed.update(obs)
        for tile in range(target*64, (target+1)*64):
            if all((p,tile) in obs for p in (0,1)):
                valid += 1
                values = arrival.priority([obs[(p,tile)] for p in (0,1)], world, 'tail')
                for src in range(world): scores[src][tile] = values[src]
        off = {s['repeat']:s['forward_us'] for s in sh['samples'] if s['mode']=='off'}
        overhead += [100*(s['forward_us']/off[s['repeat']]-1) for s in sh['samples'] if s['mode']=='receiver_sparse']
    assert valid == tiles, ('Missing training observations', valid)
    mask = select(scores, world, fraction)
    selected = [(s,t) for s in range(world) for t in range(tiles) if mask[s][t]]
    hits = [max(observed[(2,t)][0])-observed[(2,t)][0][s] <= observed[(2,t)][1]
            for s,t in selected if (2,t) in observed]
    return dict(mask=mask, selected_per_source=[sum(row) for row in mask],
                selected_remote_fraction=len(selected)/(tiles*(world-1)),
                training_valid_tiles=valid, heldout_selected_observations=len(hits),
                heldout_near_critical_fraction=statistics.mean(hits) if hits else None,
                sampled_median_overhead_percent=statistics.median(overhead),
                identity_fallback=not selected)


def main(calibration, out, fraction):
    config = json.loads((calibration/'calibration-complete.json').read_text())
    assert config['completed']
    plan = dict(schema=1, shape=[2048,2048,8192], world=4, budget_fraction=fraction,
                training_passes=[0,1], heldout_passes=[2], policies={}, inputs={},
                rule='At most 25% remote tiles per source: highest receiver-local join scores among sources near-critical in BOTH training passes; ties by physical tile index; local BF16',
                limitations='Exploratory perturbed labels; not a positive net-benefit estimator; no online adaptation or arrival-priority permutation.')
    plan['rule'] = plan['rule'].replace('25%', f'{100*fraction:g}%')
    for policy in ('remote_first','interleaved'):
        paths = [calibration/f'b0-{policy}-instrumented-rank{r}.json' for r in range(4)]
        reports = [json.loads(p.read_text()) for p in paths]
        assert all(r['completed'] for r in reports)
        plan['policies'][policy] = fit(reports, fraction)
        for p in paths+[calibration/f'mapping-{policy}-2048-2048-8192.csv']:
            plan['inputs'][str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (calibration/'calibration-complete.json', Path(__file__), HERE.parent/'arrival/plan.py'):
        plan['inputs'][str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({p:{k:v for k,v in row.items() if k!='mask'} for p,row in plan['policies'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('calibration', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--fraction', type=float, default=.25)
    args = parser.parse_args()
    main(args.calibration, args.out, args.fraction)
