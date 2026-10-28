"""Evaluate frozen training labels on a separate base-order calibration run."""
import argparse
import json
import math
from pathlib import Path
import statistics
from plan import observations, priority


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        for i in order[begin:end]:
            result[i] = (begin + end - 1) / 2
        begin = end
    return result


def spearman(left, right):
    if len(left) < 2:
        return None
    a, b = ranks(left), ranks(right)
    x, y = statistics.mean(a), statistics.mean(b)
    denom = math.sqrt(sum((v-x)**2 for v in a) * sum((v-y)**2 for v in b))
    return sum((u-x)*(v-y) for u, v in zip(a, b)) / denom if denom else None


def main(training, validation, plan_path, out):
    plan = json.loads(plan_path.read_text())
    assert plan['world_size'] == 8 and plan['score_mode'] == 'tail'
    assert training.resolve() == Path(plan['calibration']).resolve()
    assert training.resolve() != validation.resolve()
    cfg = json.loads((validation/'calibration-complete.json').read_text())
    assert cfg['completed'] and cfg['config']['world_size'] == 8
    rows = []
    for policy in ('remote_first', 'interleaved'):
        for target in range(8):
            train = json.loads((training/f'b0-{policy}-instrumented-rank{target}.json').read_text())
            valid = json.loads((validation/f'b0-{policy}-instrumented-rank{target}.json').read_text())
            assert train['completed'] and valid['completed']
            for ts, vs in zip(train['shapes'], valid['shapes']):
                shape = tuple(ts[k] for k in ('M', 'N', 'K_global'))
                assert shape == tuple(vs[k] for k in ('M', 'N', 'K_global'))
                a, b = observations(ts, target, 8, relative=True), observations(vs, target, 8, relative=True)
                per = ts['partition_tiles']
                pairs, hits, sizes, uncertainties, repeat_ids = [], [], [], [], set()
                for tile in range(target*per, (target+1)*per):
                    if not all((p, tile) in a for p in (0, 1)) or not all((p, tile) in b for p in (0, 1, 2)):
                        continue
                    predicted = [a[(p, tile)] for p in (0, 1)]
                    actual = [b[(p, tile)] for p in (0, 1, 2)]
                    candidates = {r for r, value in enumerate(priority(predicted, 8, 'tail')) if value > 0}
                    pairs.append((tile, statistics.median(max(v) for v, e in predicted),
                                  statistics.median(max(v) for v, e in actual)))
                    sizes.append(len(candidates))
                    arrivals = [statistics.median(v[r] for v, e in actual) for r in range(8)]
                    winner = max(range(8), key=arrivals.__getitem__)
                    uncertainty = max(e for v, e in actual)
                    uncertainties.append(uncertainty)
                    if candidates and arrivals[winner] - max(arrivals[r] for r in range(8) if r != winner) > uncertainty:
                        hits.append(winner in candidates)
                    repeat_ids.update(p*per + tile-target*per for p in (0, 1, 2))
                n = max(1, math.ceil(len(pairs)*.1))
                predicted_tail = {v[0] for v in sorted(pairs, key=lambda v: (-v[1], v[0]))[:n]}
                actual_tail = {v[0] for v in sorted(pairs, key=lambda v: (-v[2], v[0]))[:n]}
                off = {s['repeat']: s['forward_us'] for s in vs['samples'] if s['mode'] == 'off'}
                overhead = [100*(s['forward_us']/off[s['repeat']]-1) for s in vs['samples']
                            if s['mode'] == 'receiver_sparse' and s['repeat'] in repeat_ids]
                rows.append(dict(policy=policy, target=target, M=shape[0], N=shape[1], K_global=shape[2],
                    paired_tiles=len(pairs), partition_tiles=per,
                    join_rank_spearman=spearman([v[1] for v in pairs], [v[2] for v in pairs]),
                    tail10_recall=len(predicted_tail & actual_tail)/len(actual_tail) if actual_tail else None,
                    admitted_set_size_mean=statistics.mean(sizes) if sizes else None,
                    resolved_source_tiles=len(hits), source_set_hit_rate=statistics.mean(hits) if hits else None,
                    poll_uncertainty_median_us=statistics.median(uncertainties) if uncertainties else None,
                    matched_observer_overhead_median_percent=statistics.median(overhead) if overhead else None))
    result = dict(completed=True, training=str(training), validation=str(validation), plan=str(plan_path), rows=rows,
                  scope='Separate fresh-process calibration in the same allocation; base-order prediction only. No post-reorder or low-perturbation claim. Tail ties are tile-ID stable.')
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('training', 'validation', 'plan', 'out'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    main(a.training, a.validation, a.plan, a.out)
