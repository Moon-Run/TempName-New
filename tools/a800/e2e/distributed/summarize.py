"""Reject incomplete distributed windows and aggregate whole-job performance."""
import csv
import gzip
import re
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import sys


def check_window(path, c, policy, mode):
    assert (path/'exit-code.txt').read_text().strip() == '0'
    for node in range(c['nodes']):
        assert (path/f'node{node}-exit-code.txt').read_text().strip() == '0'
    rows = [json.loads((path/f'rank{rank}.json').read_text()) for rank in range(c['world_size'])]
    expected_shapes = {(c['sequence']*c['micro_batch'], c['hidden'], k) for k in (c['hidden'], c['ffn'])}
    for rank, r in enumerate(rows):
        assert r['rank'] == rank and r['world_size'] == c['world_size'] and r['model'] == c
        assert r['policy'] == policy and r['mode'] == mode and r['completed'] and r['passed']
        assert r['parameters_updated'] and r['skips'] == 0 and r['fallback_calls'] == 0
        assert r['local_rank'] == rank % c['gpus_per_node']
        assert r['tp_rank'] == rank % c['tp'] and r['dp_rank'] == rank // c['tp']
        assert r['tokens_per_optimizer_step'] == c['tokens_per_step']
        assert len(r['target_modules']) == c['layers']*2
        assert {(s['M'],s['N'],s['K_global']) for s in r['observed_shapes'].values()} == expected_shapes
        warmup, count = (2, 3) if mode == 'smoke' else (c['warmup_steps'], 1 if mode == 'profile' else c['timed_steps'])
        assert r['warmup_steps'] == warmup
        if mode != 'profile':
            assert r['timed_steps'] == count
        assert len(r['losses']) == len(r['grad_norms']) == count
        assert all(v == warmup*c['microsteps'] for v in r['warmup_calls'].values())
        assert len(r['warmup_calls']) == c['layers']*2
        assert all(math.isfinite(x) for x in r['losses']+r['grad_norms']+[r['initial_loss']])
        if mode != 'profile':
            assert math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0
        else:
            check_profile(path, c, policy, rank)
        if mode == 'smoke' and policy != 'native':
            assert len(r['numerical_checks']) == c['layers']*2
            assert all(v['passed'] for v in r['numerical_checks'].values())
    assert len({r.get('timing_protocol','legacy') for r in rows}) == 1
    assert len({r['gpu_uuid'] for r in rows}) == c['world_size']
    assert len({r['host'] for r in rows}) == c['nodes']
    for node in range(c['nodes']):
        rr = rows[node*c['gpus_per_node']:(node+1)*c['gpus_per_node']]
        assert len({r['host'] for r in rr}) == 1
    assert len({r['global_tokens_sha256'] for r in rows}) == 1
    for dp in range(c['dp']):
        rr = rows[dp*c['tp']:(dp+1)*c['tp']]
        assert len({r['tokens_sha256'] for r in rr}) == 1
    # Every TP shard is replicated across DP; a missing gradient sync must not pass silently.
    for tp in range(c['tp']):
        rr = rows[tp::c['tp']]
        for key in ('initial_parameters_sha256', 'final_parameters_sha256'):
            assert len({r[key] for r in rr}) == 1, (tp, key)
    return rows


def check_profile(path, c, policy, rank):
    with gzip.open(path/f'rank{rank}-profile.json.gz', 'rt') as f:
        events = json.load(f)['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'flux_bf16' in e['name']]
    calls = c['layers']*2*c['microsteps']*c['tp_nodes']
    assert len(kernels) == (0 if policy == 'native' else calls), (policy,rank,len(kernels),calls)
    if not kernels:
        return
    # Mapping checkers describe this frozen 128x128 tile only; fail for a new dispatch.
    expected = Counter()
    for k in (c['hidden'], c['ffn']):
        log = (path.parent/f"mapping-{policy}-k{k}-node{rank//c['gpus_per_node']}.log").read_text()
        grid = int(re.search(rf"PASS rank={rank%c['gpus_per_node']} .*grid=(\d+)",log).group(1))
        expected[grid] += calls//2
    for e in kernels:
        assert '64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic' in e['name']
        assert e['args']['registers per thread'] == 254 and e['args']['block'] == [128,1,1]
    assert Counter(e['args']['grid'][0] for e in kernels) == expected


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(out):
    protocol = json.loads((out/'protocol.json').read_text())
    c = protocol['config']
    windows = []
    signatures = {}
    for block, order in enumerate(c['orders']):
        for policy in order:
            rows = check_window(out/f'block-{block:03d}-{policy}', c, policy, 'timing')
            for rank, r in enumerate(rows):
                assert r['block'] == block
                signature = tuple(r[k] for k in ('initial_parameters_sha256', 'initial_rng_sha256',
                                                 'tokens_sha256', 'gpu_uuid', 'host')) + (r.get('timing_protocol','legacy'),)
                if rank not in signatures:
                    signatures[rank] = signature
                assert signature == signatures[rank], (block, policy, rank)
            elapsed = max(r['elapsed_seconds'] for r in rows)
            windows.append(dict(block=block, policy=policy, nodes=c['nodes'], gpus_per_node=c['gpus_per_node'],
                TP=c['tp'], DP=c['dp'], ms_per_step=elapsed*1000/c['timed_steps'],
                tokens_per_second=c['tokens_per_step']*c['timed_steps']/elapsed,
                initial_loss=rows[0]['initial_loss'], last_loss=rows[0]['losses'][-1],
                grad_norm=rows[0]['grad_norms'][-1], peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows)))
    summary = []
    pairs = []
    for policy in c['policies']:
        points = [r for r in windows if r['policy'] == policy]
        times = [r['ms_per_step'] for r in points]
        result = dict(policy=policy, median_ms_per_step=statistics.median(times),
            median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in points),
            cv_percent=100*statistics.stdev(times)/statistics.mean(times) if len(times)>1 else 0,
            initial_loss=statistics.median(r['initial_loss'] for r in points),
            last_loss=statistics.median(r['last_loss'] for r in points),
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in points))
        for baseline in [p for p in ('native', 'original') if p in c['policies']]:
            references = [r for r in windows if r['policy'] == baseline]
            logs = []
            for candidate, reference in zip(points, references):
                assert candidate['block'] == reference['block']
                ratio = candidate['ms_per_step']/reference['ms_per_step']
                logs.append(math.log(ratio))
                pairs.append(dict(block=candidate['block'], policy=policy, baseline=baseline,
                                  latency_reduction_percent=100*(1-ratio)))
            rng = random.Random(1234)
            boot = sorted(100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(2000))
            result[f'reduction_vs_{baseline}_percent'] = 100*(1-math.exp(statistics.mean(logs)))
            result[f'ci95_low_vs_{baseline}'] = boot[49]
            result[f'ci95_high_vs_{baseline}'] = boot[1949]
        summary.append(result)
    write(out/'windows.csv', windows)
    write(out/'summary.csv', summary)
    if pairs:
        write(out/'paired_blocks.csv', pairs)
    result = dict(completed=True, config=c, summary=summary,
                  interval_unit='paired blocks within one job; independent jobs must be repeated separately')
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    analyze(Path(sys.argv[1]))
