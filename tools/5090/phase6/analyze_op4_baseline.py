#!/usr/bin/env python3
"""Read-only OP3 profile attribution for the OP4 communication-channel experiment.

Durations are profiler kernel time, not formal full-step latency or predicted
speedup. Exposed time here only means no temporal overlap with a non-NCCL GPU
kernel on the same rank; it is not a causal critical-path reconstruction.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import median


REPO = Path(__file__).resolve().parents[3]
ALLOWED_ARMS = (('op3-r', 'op3-improved'), ('op3-combined',))
OWNERS = frozenset((
    '_MA8CachedRawLinear', '_MA8CachedRawLinearBackward',
    'ConsumerGemmRSFunction', 'ConsumerGemmRSFunctionBackward',
    'GemmRSFunction', 'GemmRSFunctionBackward',
    'LinearWithGradAccumulationAndAsyncCommunication',
    'LinearWithGradAccumulationAndAsyncCommunicationBackward',
    '_ScatterToSequenceParallelRegionBackward', '_ReduceFromModelParallelRegion',
    '_VocabParallelCrossEntropy',
))


class Coverage:
    """Integral of the union of half-open intervals, for O(log n) queries."""
    def __init__(self, intervals):
        merged = []
        for start, stop in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(stop, merged[-1][1])
            else:
                merged.append([start, stop])
        self.starts = [a for a, _ in merged]
        self.stops = [b for _, b in merged]
        self.prefix = [0.]
        for a, b in merged:
            self.prefix.append(self.prefix[-1] + b - a)

    def integral(self, point):
        index = bisect_right(self.starts, point) - 1
        if index < 0:
            return 0.
        return self.prefix[index] + min(point, self.stops[index]) - self.starts[index]

    def overlap(self, start, stop):
        return self.integral(stop) - self.integral(start)


def communication_owners(events):
    """Resolve collective External id to its enclosing autograd CPU owner."""
    threads = defaultdict(list)
    for event in events:
        if event.get('cat') == 'cpu_op' and event.get('ph') == 'X':
            threads[(event.get('pid'), event.get('tid'))].append(event)
    owners = {}
    for rows in threads.values():
        stack = []
        for event in sorted(rows, key=lambda e: (e['ts'], -e['dur'])):
            stop = event['ts'] + event['dur']
            while stack and stop > stack[-1]['ts'] + stack[-1]['dur'] + .001:
                stack.pop()
            if event['name'] == 'record_param_comms':
                matching = [parent['name'] for parent in stack if parent['name'] in OWNERS]
                owners[event['args']['External id']] = matching[-1] if matching else 'outside-target-autograd'
            stack.append(event)
    return owners


def kernel_kind(name):
    for kind in ('AllGather', 'ReduceScatter', 'AllReduce', 'SendRecv'):
        if kind in name:
            return kind
    return 'OtherNCCL'


def analyze_trace(path, batch, rank):
    with gzip.open(path, 'rt') as source:
        events = json.load(source)['traceEvents']
    kernels = [event for event in events if event.get('cat') == 'kernel']
    compute = Coverage((e['ts'], e['ts'] + e['dur']) for e in kernels
                       if 'nccl' not in e['name'].lower())
    owners = communication_owners(events)
    groups = defaultdict(list)
    for event in kernels:
        if 'nccl' not in event['name'].lower():
            continue
        args = event['args']
        if args.get('device') != rank:
            raise ValueError(f'Unexpected kernel device in {path}')
        name = event['name'].split('(')[0]
        key = (name, owners.get(args.get('External id'), 'unresolved'),
               args.get('In msg nelems'), args.get('Out msg nelems'), args.get('dtype'))
        overlap = compute.overlap(event['ts'], event['ts'] + event['dur'])
        geometry = (tuple(args.get('grid', ())), tuple(args.get('block', ())))
        if (len(geometry[0]) != 3 or len(geometry[1]) != 3
                or any(type(n) is not int or n <= 0 for dims in geometry for n in dims)):
            raise ValueError(f'Missing or invalid NCCL launch geometry in {path}')
        groups[key].append((event['dur'], overlap, args.get('stream'), geometry))
    rows = []
    for (name, owner, input_count, output_count, dtype), samples in sorted(groups.items()):
        geometries = Counter(v[3] for v in samples)
        rows.append(dict(kernel=name, collective=kernel_kind(name), owner=owner,
                         input_elements=input_count, output_elements=output_count, dtype=dtype,
                         calls=len(samples), streams=sorted({v[2] for v in samples}),
                         launch_geometries=[dict(grid=list(grid), block=list(block), calls=calls)
                                            for (grid, block), calls in sorted(geometries.items())],
                         kernel_ms=sum(v[0] for v in samples) / 1000,
                         non_nccl_kernel_overlap_ms=sum(v[1] for v in samples) / 1000,
                         exposed_ms=sum(v[0] - v[1] for v in samples) / 1000,
                         median_call_us=median(v[0] for v in samples),
                         minimum_call_us=min(v[0] for v in samples),
                         maximum_call_us=max(v[0] for v in samples)))
    counts = {kind: sum(row['calls'] for row in rows if row['collective'] == kind)
              for kind in ('AllGather', 'ReduceScatter', 'SendRecv')}
    if counts != dict(AllGather=524, ReduceScatter=260, SendRecv=256):
        raise ValueError(f'{path} is not a complete single-step OP3 profile: {counts}')
    if any(row['owner'] == 'unresolved' for row in rows):
        raise ValueError(f'{path} contains a collective without a CPU launch owner')
    return dict(rank=rank, trace=str(path.relative_to(batch)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), groups=rows)


def summarize(ranks):
    keys = sorted({(row['kernel'], row['owner'], row['input_elements'],
                    row['output_elements'], row['dtype'])
                   for rank in ranks for row in rank['groups']})
    groups = []
    for kernel, owner, in_n, out_n, dtype in keys:
        by_rank = []
        for rank in ranks:
            matches = [row for row in rank['groups']
                       if (row['kernel'], row['owner'], row['input_elements'],
                           row['output_elements'], row['dtype']) ==
                       (kernel, owner, in_n, out_n, dtype)]
            if len(matches) != 1:
                raise ValueError('Collective attribution differs across the eight ranks')
            by_rank.append(matches[0])
        geometries = sorted({(tuple(item['grid']), tuple(item['block']))
                             for row in by_rank for item in row['launch_geometries']})
        geometry_counts = [dict(grid=list(grid), block=list(block),
                                calls_per_rank=[sum(item['calls'] for item in row['launch_geometries']
                                                    if (tuple(item['grid']), tuple(item['block'])) == (grid, block))
                                                for row in by_rank]) for grid, block in geometries]
        groups.append(dict(kernel=kernel, collective=kernel_kind(kernel), owner=owner,
                           input_elements=in_n, output_elements=out_n, dtype=dtype,
                           calls_per_rank=[row['calls'] for row in by_rank],
                           launch_geometry_counts=geometry_counts,
                           median_rank_kernel_ms=median(row['kernel_ms'] for row in by_rank),
                           median_rank_non_nccl_kernel_overlap_ms=median(
                               row['non_nccl_kernel_overlap_ms'] for row in by_rank),
                           median_rank_exposed_ms=median(row['exposed_ms'] for row in by_rank),
                           median_of_rank_call_medians_us=median(row['median_call_us'] for row in by_rank)))
    totals = {}
    for kind in ('AllGather', 'ReduceScatter', 'AllReduce', 'SendRecv'):
        sums = [{metric: sum(row[metric] for row in rank['groups'] if row['collective'] == kind)
                 for metric in ('calls', 'kernel_ms', 'non_nccl_kernel_overlap_ms', 'exposed_ms')}
                for rank in ranks]
        totals[kind] = {f'median_rank_{metric}': median(row[metric] for row in sums)
                        for metric in sums[0]}
    return dict(groups=groups, totals=totals)


def analyze(batch):
    batch = Path(batch).resolve()
    report = json.loads((batch/'report.json').read_text())
    source_arms = tuple(report['arms'])
    if report.get('completed') is not True or source_arms not in ALLOWED_ARMS:
        raise ValueError('Expected a completed OP3 v1 or combined batch')
    arms = {}
    for arm in source_arms:
        ranks = [analyze_trace(batch / 'results' / f'profile-{arm}' /
                               f'rank{rank}-profile.json.gz', batch, rank)
                 for rank in range(8)]
        arms[arm] = dict(ranks=ranks, summary=summarize(ranks))
    return dict(schema='op4-historical-profile-attribution-v3',
                source_batch_relative_to_repository=os.path.relpath(batch, REPO),
                source_report_sha256=hashlib.sha256((batch/'report.json').read_bytes()).hexdigest(),
                units=dict(kernel_ms='milliseconds per profiled optimizer step',
                           median_call_us='microseconds per collective kernel'),
                method='Group kernel events by collective, CPU launch owner, message size and dtype; '
                       'sum each rank before taking the eight-rank median. Temporal overlap is the '
                       'intersection with the union of all non-NCCL GPU kernel intervals on that rank. '
                       'Launch grid and block dimensions are copied from each kernel event and counted '
                       'within its attributed group.',
                limitations=[
                    'Source OP3 profiles, not an OP4 measurement; source arms are recorded explicitly.',
                    'Kernel durations may include waiting for other ranks and profiler overhead.',
                    'Exposed means no simultaneous non-NCCL GPU kernel, not a causal critical-path proof.',
                    'Do not sum group medians to reconstruct formal step time or a predicted speedup.',
                    'All ranks and all collectives are retained, including startup and outliers.',
                    'NCCL kernel entry-point names do not establish the actual algorithm or protocol: '
                    'NCCL 2.27 can dispatch different algorithms/protocols through the same specialized entry.',
                    'Grid dimensions establish launched CTAs, not communicator channel capacity or '
                    'per-message ring topology. No LL-to-Simple transition is inferred from these traces.',
                ], arms=arms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True, help='Completed OP3 v1 or combined batch')
    parser.add_argument('--out', type=Path, required=True, help='New analysis JSON; historical batch stays read-only')
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.out.resolve()
    if out == batch or batch in out.parents:
        parser.error('--out must be outside the read-only historical batch')
    result = analyze(batch)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(out)


if __name__ == '__main__':
    main()
