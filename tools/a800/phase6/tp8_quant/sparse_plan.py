"""Retain only the highest training-tail-priority remote tile per source."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tp8_plan', HERE/'plan.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def main(args):
    source, out = args.source.resolve(), args.out.resolve()
    plan = json.loads(source.read_text())
    assert plan['world'] == 8 and plan['budget_fraction'] == 1/64
    for name,digest in plan['inputs'].items():
        assert base.sha(name) == digest, name
    for policy,entries in plan['policies'].items():
        entry = entries[0]
        scores = [[0.]*256 for _ in range(8)]
        heldout = {}
        for receiver in range(8):
            path = next(Path(p) for p in plan['inputs'] if p.endswith(f'b0-{policy}-instrumented-rank{receiver}.json'))
            report = json.loads(path.read_text())
            shape = next(s for s in report['shapes'] if [s['M'],s['N'],s['K_global']] == entry['shape'])
            obs = base.arrival.observations(shape,receiver,8,relative=True)
            for tile in range(receiver*32,(receiver+1)*32):
                if all((p,tile) in obs for p in (0,1)):
                    values = base.arrival.priority([obs[(p,tile)] for p in (0,1)],8,'tail')
                    for src in range(8): scores[src][tile] = values[src]
                if (2,tile) in obs: heldout[tile] = obs[(2,tile)]
        mask = base.selection.select(scores,8,1/224)
        assert all(sum(row) == 1 for row in mask)
        assert all(not mask[s][t] or entry['mask'][s][t] for s in range(8) for t in range(256))
        chosen = [(s,t) for s in range(8) for t in range(256) if mask[s][t]]
        hits = [max(heldout[t][0])-heldout[t][0][s] <= heldout[t][1] for s,t in chosen if t in heldout]
        entry.update(mask=mask,selected_per_source=list(map(sum,mask)),
                     selected_remote_fraction=len(chosen)/(7*256),
                     heldout_selected_observations=len(hits),
                     heldout_near_critical_fraction=statistics.mean(hits) if hits else None)
    plan.update(selection_limit_tiles_per_source=1,
        limitation='Same BF16 training passes 0/1 and same arrival permutations; only the highest positive tail score per source is quantized. Existing 1/64 upper bound retained. Pass 2 is diagnostic only; no new or joint recalibration.',
        sparse_parent=dict(path=str(source),sha256=base.sha(source)))
    for p in (source,Path(__file__).resolve(),HERE/'plan.py'):
        plan['inputs'][str(p)] = base.sha(p)
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    out.write_text(json.dumps(plan,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('out',type=Path)
    main(p.parse_args())
