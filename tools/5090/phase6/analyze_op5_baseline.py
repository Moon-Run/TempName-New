#!/usr/bin/env python3
"""Read-only attribution of every window in the OP4 reference used by OP5."""
import argparse
import gzip
import json
import os
from pathlib import Path
from statistics import median

from analyze_op4_baseline import analyze_trace, summarize, REPO
from op5_evidence import reference_record


def analyze(batch):
    batch = Path(batch).resolve()
    reference = reference_record(batch)
    windows = []
    owners = {'_MA8CachedRawLinear', 'ConsumerGemmRSFunctionBackward'}
    for child in reference['children']:
        root = batch/child['directory']
        ranks, htod = [], []
        for rank in range(8):
            path = root/f'results/profile-op3-combined/rank{rank}-profile.json.gz'
            record = analyze_trace(path, batch, rank)
            ranks.append(record)
            with gzip.open(path, 'rt') as stream:
                events = json.load(stream)['traceEvents']
            gathers = [e for e in events if e.get('cat') == 'kernel' and 'AllGather' in e['name']]
            copies = [e for e in events if e.get('cat') == 'gpu_memcpy' and 'HtoD' in e['name']
                      and e.get('args', {}).get('device') == rank]
            contained = [e for e in copies if any(g['ts'] <= e['ts'] and e['ts']+e['dur'] <= g['ts']+g['dur']+.05
                                                 for g in gathers)]
            htod.append(dict(rank=rank, htod_in_allgather=len(contained),
                             bytes_in_allgather=sum(e['args']['bytes'] for e in contained)))
        major = [sum(g['kernel_ms'] for g in r['groups'] if g['collective'] == 'AllGather' and g['owner'] in owners)
                 for r in ranks]
        windows.append(dict(directory=child['directory'], formal_ms=child['historical_mean_ms'],
            ranks=ranks, summary=summarize(ranks), copy_diagnostics=htod,
            major_512_gathers_median_rank_ms=median(major)))
    # Keep artifact paths portable; fingerprints bind original reports and traces.
    return dict(schema='op5-op4-reference-attribution-v1', source_batch=os.path.relpath(batch, REPO),
        source_report_sha256=reference['report_sha256'], historical_mean_ms=reference['historical_mean_ms'],
        target_50ms_line=reference['historical_mean_ms']-50,
        windows=windows, limitations=[
            'Profiler kernel durations include rank waits and perturbation; not formal optimizer-step latency.',
            'Compute overlap uses non-NCCL GPU kernels on the same rank; not a causal critical-path reconstruction.',
            'Sum per rank before the rank median. Do not sum category medians into a step latency.',
            'No historical transport INFO log: missing HtoD copies is not proof of a specific host transport.',
            'No OP5 GPU measurements or predicted 50ms gain are contained in this artifact.'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.out.resolve()
    if out == batch or batch in out.parents:
        parser.error('--out must be outside the read-only historical batch')
    result = analyze(batch)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('historical_mean_ms', 'target_50ms_line')}))
    for row in result['windows']:
        print(row['directory'], 'major 512 gathers ms:', row['major_512_gathers_median_rank_ms'],
              'HtoD counts:', [r['htod_in_allgather'] for r in row['copy_diagnostics']])


if __name__ == '__main__':
    main()
