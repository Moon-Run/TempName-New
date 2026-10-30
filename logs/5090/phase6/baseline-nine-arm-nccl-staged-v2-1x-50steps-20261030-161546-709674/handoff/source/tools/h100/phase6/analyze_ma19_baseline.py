#!/usr/bin/env python3
"""Audit the original MA11 optimizer's FP32-master to BF16-model publication.

Read-only with respect to the historical batch. This does not measure MA19.
Each target GPU cast is tied to one CUDA launch and one aten::copy_ CPU scope.
The historical profiler lacks tensor addresses/shapes, so logical bytes are
derived separately from its saved parameter inventory, with launch-grid counts
checked against that inventory. No next-step forward is present in these traces.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCE_ARM = 'F4-cache-zcount-csoftmax'
COPY_KERNEL = 'bfloat16_copy_kernel_cuda'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def end(event):
    return event['ts'] + event['dur']


def analyze_trace(path, rank):
    path = Path(path)
    record_path = path.with_name(f'rank{rank}.json')
    record = json.loads(record_path.read_text())
    require(record['completed'] and record['passed'], 'Historical profile worker failed')
    require(record['rank'] == rank and record['mode'] == 'profile', 'Wrong rank or mode')
    require(record['ma11_arm'] == SOURCE_ARM, 'Expected the original MA11 arm')
    parameters = record['ma11_trainable_parameters']
    require(len(parameters) == 389, 'Unexpected original trainable parameter universe')
    require(all(p['dtype'] == 'torch.bfloat16' for p in parameters.values()),
            'Historical parameter inventory is not entirely BF16')
    numels = [math.prod(p['shape']) for p in parameters.values()]
    require(all(n > 0 and n % 512 == 0 for n in numels), 'Unexpected parameter geometry')

    with gzip.open(path, 'rt') as stream:
        events = json.load(stream)['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    cpu = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X']
    runtime = [e for e in events if e.get('cat') == 'cuda_runtime' and e.get('ph') == 'X']
    launches = defaultdict(list)
    externals = defaultdict(list)
    for event in runtime:
        if event['name'] in ('cudaLaunchKernel', 'cudaLaunchKernelExC'):
            launches[event.get('args', {}).get('correlation')].append(event)
    for event in cpu:
        externals[event.get('args', {}).get('External id')].append(event)

    def launch_of(kernel):
        candidates = launches[kernel['args']['correlation']]
        require(len(candidates) == 1, 'Missing or ambiguous CUDA launch correlation')
        launch = candidates[0]
        require(launch['ts'] <= kernel['ts'], 'GPU kernel precedes its CPU launch')
        return launch

    adam = sorted([e for e in kernels if 'AdamFunctor<' in e['name']], key=lambda e: e['ts'])
    require(len(adam) == 49, 'Historical Adam kernel count changed')
    adam_launches = sorted([launch_of(e) for e in adam], key=lambda e: e['ts'])
    casts = [e for e in kernels if COPY_KERNEL in e['name']]
    target = sorted([e for e in casts if launch_of(e)['ts'] > adam_launches[0]['ts']],
                    key=lambda e: launch_of(e)['ts'])
    require(len(target) == len(parameters), 'Expected one post-Adam cast per BF16 parameter')
    require(len(casts) - len(target) == 4, 'Unexpected non-optimizer BF16 casts')
    require(Counter(e['args']['grid'][0] for e in target) == Counter(n // 512 for n in numels),
            'Target launch-grid multiset differs from the saved parameter inventory')
    require(all(e['args']['grid'][1:] == [1, 1] and e['args']['block'] == [128, 1, 1]
                for e in target), 'Historical cast launch geometry changed')
    streams = {e['args']['stream'] for e in adam + target}
    require(len(streams) == 1, 'Adam and model publication are not on one CUDA stream')
    stream_id = next(iter(streams))

    per_copy, copy_scopes, copy_launches = [], [], []
    for index, kernel in enumerate(target):
        launch = launch_of(kernel)
        scope_candidates = [e for e in externals[kernel['args']['External id']]
                            if e['name'] == 'aten::copy_' and e['tid'] == launch['tid']
                            and e['ts'] <= launch['ts'] and end(launch) <= end(e) + 0.01]
        require(len(scope_candidates) == 1, 'Cast is not uniquely owned by aten::copy_')
        scope = scope_candidates[0]
        require(scope['ts'] > end(adam_launches[-1]), 'Copy precedes final Adam submission')
        require(kernel['ts'] >= end(adam[-1]) - 0.01, 'Copy precedes final Adam completion')
        copy_scopes.append(scope)
        copy_launches.append(launch)
        per_copy.append(dict(index=index, external_id=kernel['args']['External id'],
                             correlation=kernel['args']['correlation'], stream=stream_id,
                             grid=kernel['args']['grid'], block=kernel['args']['block'],
                             cpu_scope_ts_us=scope['ts'], cpu_scope_us=scope['dur'],
                             cuda_launch_ts_us=launch['ts'], cuda_launch_us=launch['dur'],
                             gpu_ts_us=kernel['ts'], gpu_us=kernel['dur']))
    require(len({x['external_id'] for x in per_copy}) == len(target),
            'A target copy scope was reused')

    copy_first = min(e['ts'] for e in copy_scopes)
    copy_last = max(end(e) for e in copy_scopes)
    # The common original clear_gradients follows optimizer.step. In this one-step
    # profile it is observable; next-step forward consumption is not.
    zeros = [e for e in cpu if e['name'] == 'aten::zero_' and e['ts'] >= copy_last]
    require(zeros, 'No post-publication gradient clear in historical trace')
    first_clear = min(zeros, key=lambda e: e['ts'])
    clear_correlations = {e['args']['correlation'] for e in runtime
                          if e['tid'] == first_clear['tid'] and first_clear['ts'] <= e['ts']
                          and end(e) <= end(first_clear) + 0.01
                          and e['name'] in ('cudaLaunchKernel', 'cudaLaunchKernelExC',
                                            'cudaMemsetAsync')}
    clear_work = [e for e in events if e.get('cat') in ('kernel', 'gpu_memset')
                  and e.get('args', {}).get('correlation') in clear_correlations]
    require(clear_work, 'Gradient clear has no correlated GPU work')
    require(all(e['args']['stream'] == stream_id for e in clear_work),
            'Gradient clear differs from the publication stream')
    require(min(e['ts'] for e in clear_work) >= end(target[-1]) - 0.01,
            'Gradient clear starts before model publication completes')
    synchronizes = [e for e in runtime if e['name'] == 'cudaDeviceSynchronize'
                    and e['ts'] >= end(first_clear)]
    require(synchronizes, 'Missing end-of-step profile synchronization')
    final_sync = min(synchronizes, key=lambda e: e['ts'])
    gpu_work = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    metrics = dict(
        target_copy_cpu_scope_sum_ms=sum(e['dur'] for e in copy_scopes) / 1000,
        target_copy_cpu_first_start_to_last_end_ms=(copy_last - copy_first) / 1000,
        target_copy_cuda_launch_sum_ms=sum(e['dur'] for e in copy_launches) / 1000,
        target_copy_gpu_work_sum_ms=sum(e['dur'] for e in target) / 1000,
        target_copy_gpu_first_start_to_last_end_ms=(end(target[-1]) - target[0]['ts']) / 1000,
        final_adam_to_first_copy_gpu_gap_ms=(target[0]['ts'] - end(adam[-1])) / 1000,
        final_adam_to_last_copy_gpu_span_ms=(end(target[-1]) - end(adam[-1])) / 1000,
        final_adam_gpu_end_minus_last_copy_cpu_end_ms=(end(adam[-1]) - copy_last) / 1000,
        final_adam_gpu_end_minus_last_copy_launch_end_ms=(end(adam[-1]) - end(copy_launches[-1])) / 1000,
        final_copy_to_first_clear_gpu_gap_ms=(min(e['ts'] for e in clear_work) - end(target[-1])) / 1000,
        copy_cpu_span_to_final_profile_sync_end_ms=(end(final_sync) - copy_first) / 1000,
        total_profile_gpu_span_ms=(max(end(e) for e in gpu_work) - min(e['ts'] for e in gpu_work)) / 1000)
    return dict(rank=rank, trace=str(path), trace_sha256=sha256(path),
                worker_record=str(record_path), worker_record_sha256=sha256(record_path),
                target_copies=len(target), other_bf16_casts=len(casts) - len(target),
                adam_kernels=len(adam), compute_stream=stream_id,
                saved_parameter_elements=sum(numels),
                fp32_source_logical_bytes=4 * sum(numels),
                bf16_destination_logical_bytes=2 * sum(numels),
                read_plus_write_logical_bytes=6 * sum(numels),
                parameter_numel_histogram=dict(sorted(Counter(numels).items())),
                copy_grid_histogram=dict(sorted(Counter(e['args']['grid'][0] for e in target).items())),
                all_copies_after_final_adam=True, same_stream_clear_after_copies=True,
                next_step_forward_observed=False, metrics=metrics, per_copy=per_copy)


def analyze(trace_dir):
    trace_dir = Path(trace_dir)
    rows = [analyze_trace(trace_dir / f'rank{rank}-profile.json.gz', rank) for rank in range(8)]
    require(len({row['saved_parameter_elements'] for row in rows}) == 1,
            'Ranks have different saved parameter element counts')
    fullstep = trace_dir.parent.parent
    source_paths = [fullstep / 'megatron/megatron/core/optimizer/optimizer.py',
                    fullstep / 'scripts/worker.py']
    source = source_paths[0].read_text()
    require('for this_, that_ in zip(this, that):\n            that_.copy_(this_)' in source,
            'Historical BF16 per-parameter copy source changed')
    require('self._copy_main_params_to_model_params()' in source,
            'Historical optimizer no longer publishes model parameters')
    return dict(
        schema='ma19-historical-ma11-parameter-publication-v1', source_arm=SOURCE_ARM,
        candidate_gpu_measured=False,
        analysis_script_sha256=sha256(Path(__file__)),
        frozen_source_sha256={str(path): sha256(path) for path in source_paths},
        attribution='GPU FP32-to-BF16 cast signature, unique CUDA launch correlation, unique '
                    'aten::copy_ external ID and enclosing CPU scope; all 389 casts follow '
                    'the final of 49 Adam kernels on the same CUDA stream. Four earlier '
                    'non-optimizer BF16 casts are excluded. Launch-grid multiplicities match '
                    'the saved 389-parameter inventory.',
        bytes_definition='Parameter shapes come from the historical worker record, not GPU '
                         'activity byte counters. The profile lacks tensor addresses/shapes; '
                         'grid agreement does not establish an individual parameter-name to '
                         'copy mapping. FP32 source reads plus BF16 destination writes are '
                         'logical bytes, not measured HBM traffic.',
        aggregation='Each metric is first summed or spanned within one rank, then the median '
                    'of eight ranks is taken. CPU scopes, nested CUDA API time, GPU work '
                    'and spans overlap and must not be added.',
        dependency_limit='These are one-step profiles: Adam -> parameter casts -> gradient '
                         'clear -> profile-end synchronization is observed. Original worker '
                         'source orders optimizer.step before the following step; no next-step '
                         'forward is recorded, so its consumption is a source-derived '
                         'same-stream dependency, not a measured cross-step latency.',
        performance_limit='The casts still require FP32 reads, BF16 conversion and BF16 writes. '
                          'MA19 seeks fewer host submissions at a stable optimizer boundary. '
                          'Positive final_adam_gpu_end_minus_last_copy_cpu_end_ms means the '
                          'entire CPU copy loop was submitted before Adam finished on the GPU; '
                          'that CPU time is already covered in this profile and is not an '
                          'additional exposed saving. '
                          'This analysis does not establish unprofiled critical-path savings, '
                          'a 2% budget or candidate speedup. Only paired complete optimizer-step '
                          'windows without a profiler can establish that result.',
        aggregate={key: statistics.median(row['metrics'][key] for row in rows)
                   for key in rows[0]['metrics']},
        per_rank=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.resolve().is_relative_to(args.trace_dir.resolve().parent.parent),
            'Output must not be written inside the historical fullstep directory')
    result = analyze(args.trace_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(schema=result['schema'], aggregate=result['aggregate'],
                          candidate_gpu_measured=False), indent=2))


if __name__ == '__main__':
    main()
