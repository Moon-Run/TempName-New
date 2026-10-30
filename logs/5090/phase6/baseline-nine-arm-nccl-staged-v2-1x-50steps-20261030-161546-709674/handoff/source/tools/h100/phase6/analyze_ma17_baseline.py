#!/usr/bin/env python3
"""Reproduce MA11 column-AllGather submission evidence for the MA17 design.

Reads the eight original traces without modifying them; it does not run MA17.
CPU/API time includes profiler overhead and is not a speedup prediction.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

import ma8_profile
import ma11_profile
from ma8_profile import contains, end, require


SOURCE_ARM = 'F4-cache-zcount-csoftmax'
METRICS = (
    'column_forward_cpu_us', 'allgather_cpu_us', 'work_wait_cpu_us',
    'allgather_to_wait_end_cpu_us', 'allgather_runtime_api_us',
    'wait_runtime_api_us', 'allgather_launch_api_us',
    'allgather_event_record_wait_api_us', 'work_wait_stream_wait_api_us',
    'allgather_gpu_us', 'full_gemm_gpu_us', 'allgather_end_to_gemm_start_us',
)
EVENT_NAMES = ('cudaEventRecord', 'cudaStreamWaitEvent')


def exactly_one(events, message):
    require(len(events) == 1, message)
    return events[0]


def summarize(calls):
    require(calls, 'Empty target call set')
    count = Counter()
    for call in calls:
        count.update(call['runtime_api_counts'])
    return dict(calls=len(calls),
                different_compute_communication_stream_calls=sum(
                    row['streams']['allgather'] != row['streams']['gemm'] for row in calls),
                runtime_api_counts=dict(sorted(count.items())),
                median_us={key: median(row['metrics'][key] for row in calls) for key in METRICS},
                sum_ms={key: sum(row['metrics'][key] for row in calls) / 1000 for key in METRICS})


def analyze_trace(path, rank):
    raw = path.read_bytes()
    events = json.loads(gzip.decompress(raw))['traceEvents']
    ma11_profile.trace_composed_records(events, SOURCE_ARM, rank)
    index = ma8_profile.TraceIndex(events)
    pairs, heads = ma8_profile._raw_column_pairs(index, 'F4-cache', 'ma8-v2')
    require(len(pairs) == 256 and len(heads) == 4, 'Target column coverage changed')
    cpu = [event for event in events if event.get('ph') == 'X' and event.get('cat') == 'cpu_op']
    calls = []
    for ordinal, (role, forward, _) in enumerate(pairs):
        nested = [event for event in cpu if event.get('tid') == forward['tid'] and contains(forward, event)]
        gather_scope = exactly_one([event for event in nested if event['name'] == 'c10d::_allgather_base_'],
                                   'Expected exactly one original ProcessGroup AllGather')
        wait_scope = exactly_one([event for event in nested if event['name'] == 'record_param_comms'
                                 and event.get('args', {}).get('Collective name') == 'wait'],
                                'Expected one original explicit Work.wait scope')
        comm = exactly_one([event for event in nested if event['name'] == 'record_param_comms'
                            and event.get('args', {}).get('Collective name') == '_allgather_base'],
                           'Expected one AllGather communication metadata scope')
        meta = comm['args']
        require(meta.get('In msg nelems') == 1048576 and meta.get('Out msg nelems') == 8388608
                and meta.get('Group size') == 8 and meta.get('dtype') == 'BFloat16',
                'Target 2 MiB -> 16 MiB TP8 BF16 AllGather payload changed')
        require(end(gather_scope) <= wait_scope['ts'], 'Original collective/wait host order changed')
        work, launches = index.work(forward)
        kernels = [event for event in work if event['cat'] == 'kernel']
        gather = exactly_one([event for event in kernels if 'nccl' in event['name'].lower()
                              and 'AllGather' in event['name']], 'Expected one NCCL AllGather kernel')
        gemm = exactly_one([event for event in kernels if 'gemm' in event['name'].lower()],
                          'Expected one full projection GEMM')
        require(gemm['ts'] >= end(gather) - .01, 'Original complete-gather-before-GEMM dependency changed')
        launch = launches[gather['args']['correlation']]
        require(contains(gather_scope, launch), 'AllGather kernel launch is outside collective scope')
        gather_runtime = [event for event in index.runtime[forward['tid']] if contains(gather_scope, event)]
        wait_runtime = [event for event in index.runtime[forward['tid']] if contains(wait_scope, event)]
        require(any(event['name'] == 'cudaStreamWaitEvent' for event in wait_runtime),
                'Original Work.wait lacks a CUDA stream dependency event')
        counts = Counter('collective.' + event['name'] for event in gather_runtime)
        counts.update('explicit_work_wait.' + event['name'] for event in wait_runtime)
        calls.append(dict(
            ordinal=ordinal, role=role, autograd_sequence=forward['args']['Sequence number'],
            process_group=meta['Process Group Description'],
            input_elements=meta['In msg nelems'], output_elements=meta['Out msg nelems'],
            dtype=meta['dtype'], group_size=meta['Group size'],
            allgather_kernel=gather['name'], allgather_launch_api=launch['name'],
            streams=dict(allgather=gather['args']['stream'], gemm=gemm['args']['stream']),
            runtime_api_counts=dict(sorted(counts.items())),
            metrics=dict(
                column_forward_cpu_us=forward['dur'], allgather_cpu_us=gather_scope['dur'],
                work_wait_cpu_us=wait_scope['dur'],
                allgather_to_wait_end_cpu_us=end(wait_scope) - gather_scope['ts'],
                allgather_runtime_api_us=sum(event['dur'] for event in gather_runtime),
                wait_runtime_api_us=sum(event['dur'] for event in wait_runtime),
                allgather_launch_api_us=launch['dur'],
                allgather_event_record_wait_api_us=sum(event['dur'] for event in gather_runtime
                                                      if event['name'] in EVENT_NAMES),
                work_wait_stream_wait_api_us=sum(event['dur'] for event in wait_runtime
                                                if event['name'] == 'cudaStreamWaitEvent'),
                allgather_gpu_us=gather['dur'], full_gemm_gpu_us=gemm['dur'],
                allgather_end_to_gemm_start_us=gemm['ts'] - end(gather))))
    require(all(sum(call['role'] == role for call in calls) == 128 for role in ('qkv', 'fc1')),
            'Expected 128 QKV and 128 FC1 forwards')
    return dict(rank=rank, trace=str(path.resolve()), trace_sha256=hashlib.sha256(raw).hexdigest(),
                original_ma11_composed_profile_validated=True,
                excluded_original_vocabulary_heads=len(heads),
                measurements=dict(combined=summarize(calls),
                    roles={role: summarize([call for call in calls if call['role'] == role])
                           for role in ('qkv', 'fc1')}), per_call=calls)


def aggregate(rows):
    def one(measurements):
        names = sorted({key for item in measurements for key in item['runtime_api_counts']})
        return dict(total_calls=sum(item['calls'] for item in measurements),
                    median_per_rank_calls=median(item['calls'] for item in measurements),
                    total_different_compute_communication_stream_calls=sum(
                        item['different_compute_communication_stream_calls'] for item in measurements),
                    median_per_rank_runtime_api_counts={key: median(item['runtime_api_counts'].get(key, 0)
                                                                   for item in measurements) for key in names},
                    median_of_rank_median_us={key: median(item['median_us'][key] for item in measurements)
                                              for key in METRICS},
                    median_of_rank_sum_ms={key: median(item['sum_ms'][key] for item in measurements)
                                           for key in METRICS})
    return dict(combined=one([row['measurements']['combined'] for row in rows]),
                roles={role: one([row['measurements']['roles'][role] for row in rows])
                       for role in ('qkv', 'fc1')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.trace_dir / f'rank{rank}-profile.json.gz' for rank in range(8)]
    require(args.out.resolve() not in {path.resolve() for path in paths}, 'Output would overwrite input trace')
    rows = [analyze_trace(path, rank) for rank, path in enumerate(paths)]
    result = dict(schema='ma17-historical-ma11-column-submission-evidence-v1', source_arm=SOURCE_ARM,
        candidate_gpu_measured=False,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        definition=dict(
            topology='128 QKV and 128 FC1 forwards per rank; four vocabulary heads excluded.',
            attribution='Original MA11 composed checker, raw autograd sequence pairing, same-thread '
                        'CPU scope containment and exact CUDA launch correlation.',
            payload='Per call: 1,048,576 BF16 input elements (2 MiB), 8,388,608 output elements '
                    '(16 MiB), TP group of eight ranks. AllGather is a byte-preserving operation.',
            original='AllGather uses a NCCL stream distinct from the full GEMM compute stream; '
                     'the explicit Work.wait records a cudaStreamWaitEvent before full GEMM.',
            proposed='Use a dedicated native NCCL communicator on the current compute stream '
                     'for these already serialized forward gathers; preserve payload and full GEMM.',
            events='The collective scope includes both ProcessGroup and NCCL internal API calls. '
                   'The trace cannot assign every event record/wait to a specific software owner. '
                   'Direct NCCL can retain internal event calls; do not claim all observed calls disappear.',
            aggregation='Sum or median inside each rank, then median across eight ranks. '
                        'CPU scope, its runtime children, GPU duration and inter-operation gap overlap '
                        'or nest and must not be added as a savings estimate.'),
        caveat='Profiler perturbs host submission and peer arrival. NCCL duration includes waiting. '
               'Removing ProcessGroup side-stream handoffs does not remove network transfer or '
               'rank synchronization. These traces establish an existing path, not its removable '
               'unprofiled critical-path cost or an MA17 speedup.',
        per_rank=rows, aggregate=aggregate(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(validated_ranks=len(rows), aggregate=result['aggregate']), indent=2))


if __name__ == '__main__':
    main()
