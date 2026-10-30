#!/usr/bin/env python3
"""Read historical MA11 traces and measure the column communication schedule.

This script writes only its requested analysis artifact; source traces are read
without modification. It does not execute MA16 or estimate an unprofiled gain.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

import ma8_profile
import ma11_profile
from ma9_profile import end, require


SOURCE_ARM = 'F4-cache-zcount-csoftmax'
METRICS = (
    'forward_allgather_us', 'forward_gemm_us', 'allgather_end_to_gemm_start_us',
    'forward_allgather_gemm_overlap_us', 'backward_reduce_scatter_us',
    'backward_wgrad_us', 'backward_reduce_scatter_wgrad_overlap_us',
)


def overlap_us(left, right):
    return max(0., min(end(left), end(right)) - max(left['ts'], right['ts']))


def summarize(calls):
    require(calls, 'Cannot summarize an empty column call set')
    return dict(
        calls=len(calls),
        forward_allgather_before_gemm_calls=sum(
            call['metrics']['allgather_end_to_gemm_start_us'] >= -.01 for call in calls),
        forward_overlap_calls=sum(
            call['metrics']['forward_allgather_gemm_overlap_us'] > .01 for call in calls),
        backward_reduce_scatter_wgrad_overlap_calls=sum(
            call['metrics']['backward_reduce_scatter_wgrad_overlap_us'] > .01 for call in calls),
        median_us={key: median(call['metrics'][key] for call in calls) for key in METRICS},
        sum_ms={key: sum(call['metrics'][key] for call in calls) / 1000 for key in METRICS},
        minimum_allgather_end_to_gemm_start_us=min(
            call['metrics']['allgather_end_to_gemm_start_us'] for call in calls),
    )


def analyze_trace(path, rank):
    trace_bytes = path.read_bytes()
    events = json.loads(gzip.decompress(trace_bytes))['traceEvents']
    validated = ma11_profile.trace_composed_records(events, SOURCE_ARM, rank)
    index = ma8_profile.TraceIndex(events)
    pairs, heads = ma8_profile._raw_column_pairs(index, 'F4-cache', 'ma8-v2')
    require(len(heads) == 4 and len(pairs) == 256,
            'Expected 256 target columns and four excluded vocabulary heads')
    calls = []
    for ordinal, (role, forward, backward) in enumerate(pairs):
        fw, _ = index.work(forward)
        bw, _ = index.work(backward)
        forward_kernels = [event for event in fw if event['cat'] == 'kernel']
        backward_kernels = [event for event in bw if event['cat'] == 'kernel']
        gathers = [event for event in forward_kernels
                   if 'nccl' in event['name'].lower() and 'AllGather' in event['name']]
        forward_gemms = [event for event in forward_kernels if 'gemm' in event['name'].lower()]
        scatters = [event for event in backward_kernels
                    if 'nccl' in event['name'].lower() and 'ReduceScatter' in event['name']]
        backward_gemms = sorted(
            [event for event in backward_kernels if 'gemm' in event['name'].lower()],
            key=lambda event: event['ts'])
        require(len(gathers) == len(forward_gemms) == len(scatters) == 1
                and len(backward_gemms) == 2, 'Original column kernel coverage changed')
        gather, gemm, scatter = gathers[0], forward_gemms[0], scatters[0]
        wgrad = backward_gemms[1]
        gap = gemm['ts'] - end(gather)
        require(gap >= -.01 and overlap_us(gather, gemm) <= .01,
                'Historical forward is no longer whole-gather-before-GEMM')
        require(gather['args']['stream'] != gemm['args']['stream']
                and scatter['args']['stream'] != wgrad['args']['stream'],
                'Original compute/communication stream distinction changed')
        calls.append(dict(
            ordinal=ordinal, role=role,
            autograd_sequence=forward['args']['Sequence number'],
            forward_allgather_kernel=gather['name'], forward_gemm_kernel=gemm['name'],
            streams=dict(forward_allgather=gather['args']['stream'],
                         forward_gemm=gemm['args']['stream'],
                         backward_reduce_scatter=scatter['args']['stream'],
                         backward_wgrad=wgrad['args']['stream']),
            metrics=dict(
                forward_allgather_us=gather['dur'], forward_gemm_us=gemm['dur'],
                allgather_end_to_gemm_start_us=gap,
                forward_allgather_gemm_overlap_us=overlap_us(gather, gemm),
                backward_reduce_scatter_us=scatter['dur'], backward_wgrad_us=wgrad['dur'],
                backward_reduce_scatter_wgrad_overlap_us=overlap_us(scatter, wgrad)),
        ))
    roles = {role: summarize([call for call in calls if call['role'] == role])
             for role in ('qkv', 'fc1')}
    require(all(value['calls'] == 128 for value in roles.values()),
            'Expected 128 QKV and 128 FC1 forwards per rank')
    return dict(
        rank=rank, trace=str(path.resolve()), sha256=hashlib.sha256(trace_bytes).hexdigest(),
        original_ma11_composed_profile_validated=True,
        original_ma8_backward_allgathers=validated['activation_cache']['backward_allgathers'],
        original_ma9_zero_count_kernels=validated['zero_count']['compute_kernel_calls'],
        original_ma11_softmax_kernels=validated['attention_softmax']['softmax_compute_kernel_calls'],
        excluded_original_vocabulary_heads=len(heads),
        measurements=dict(roles=roles, combined=summarize(calls)),
        per_call=calls,
    )


def aggregate(rows):
    def one(measurements):
        return dict(
            median_per_rank_calls=median(item['calls'] for item in measurements),
            total_calls=sum(item['calls'] for item in measurements),
            total_forward_allgather_before_gemm_calls=sum(
                item['forward_allgather_before_gemm_calls'] for item in measurements),
            total_forward_overlap_calls=sum(item['forward_overlap_calls'] for item in measurements),
            total_backward_reduce_scatter_wgrad_overlap_calls=sum(
                item['backward_reduce_scatter_wgrad_overlap_calls'] for item in measurements),
            median_of_rank_median_us={key: median(item['median_us'][key] for item in measurements)
                                      for key in METRICS},
            median_of_rank_sum_ms={key: median(item['sum_ms'][key] for item in measurements)
                                   for key in METRICS},
            minimum_allgather_end_to_gemm_start_us=min(
                item['minimum_allgather_end_to_gemm_start_us'] for item in measurements),
        )
    return dict(
        roles={role: one([row['measurements']['roles'][role] for row in rows])
               for role in ('qkv', 'fc1')},
        combined=one([row['measurements']['combined'] for row in rows]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.trace_dir / f'rank{rank}-profile.json.gz' for rank in range(8)]
    require(args.out.resolve() not in {path.resolve() for path in paths},
            'Analysis output must not overwrite an input trace')
    rows = [analyze_trace(path, rank) for rank, path in enumerate(paths)]
    result = dict(
        schema='ma16-historical-ma11-column-communication-evidence-v1',
        source_arm=SOURCE_ARM, candidate_gpu_measured=False,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        definition=dict(
            topology='32 layers x four microbatches; 128 QKV and 128 FC1 calls per rank; '
                     'four original vocabulary heads excluded.',
            attribution='Pinned MA11 composed profile validation, original raw column autograd '
                        'sequence pairing, and CUDA launch correlation inside each Function.',
            original_forward='Whole input AllGather finishes before full GEMM starts.',
            proposed_ma16='Submit the original whole AllGather asynchronously; compute this '
                          'rank\'s 256 owned rows; join AllGather; compute remote prefix/suffix rows.',
            original_backward='MA8 retains the forward input. Existing dgrad ReduceScatter '
                              'and wgrad may overlap; MA16 does not claim to introduce that overlap.',
            aggregation='Sum within each rank first, then median over eight ranks. Combined '
                        'statistics are calculated from both roles per rank before aggregation.',
            overlap='Same-rank GPU interval intersection; values above 0.01 microseconds count '
                    'as observed overlap. Never subtract absolute timestamps across ranks.',
            budget='Owned rows are 256 of 2048 (one eighth by row count). A 256-row GEMM '
                   'need not take one eighth of full-GEMM time. No candidate-time scaling or '
                   'performance prediction is inferred from these baseline durations.',
        ),
        caveat='Profiler changes host submission and rank arrival. NCCL durations include waits. '
               'GPU work, communication duration and their overlap are not additive unprofiled '
               'critical-path savings. The artifact contains no MA16 correctness or speed result.',
        per_rank=rows, aggregate=aggregate(rows),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(validated_ranks=len(rows), aggregate=result['aggregate']), indent=2))


if __name__ == '__main__':
    main()
