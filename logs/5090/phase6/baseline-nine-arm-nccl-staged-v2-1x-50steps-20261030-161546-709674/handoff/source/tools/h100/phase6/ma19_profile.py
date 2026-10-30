"""Verify original Adam -> parameter publication -> gradient clear dependencies.

The fixed BF16 model has 389 parameter tensors and one nonempty child optimizer.
Markers alone are insufficient: every publication needs a unique CUDA launch,
the original compute stream, and the unchanged Adam and clear operations.
"""
import ma11_profile as baseline
from ma9_profile import Index, contains, end, require

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-pcopy')
MARKER = 'ma19.optimizer.parameter_publish'
KERNEL = 'ma19_parameter_publish_kernel'
PARAMETERS = 389
ELEMENTS = 849207296
CHUNKS = 26146


def _one(rows, message):
    require(len(rows) == 1, message)
    return rows[0]


def trace_parameter_publish_records(events, arm):
    require(arm in ARMS, 'Unknown MA19 arm')
    candidate = arm == ARMS[1]
    index = Index(events)
    kernels = [e for e in events if e.get('cat') == 'kernel']
    adam = sorted([e for e in kernels if 'AdamFunctor' in e['name']], key=lambda e: e['ts'])
    require(len(adam) == 49 and all('AdamFunctor<float, float, int>' in e['name'] for e in adam),
            'MA19 requires the original 49 FP32 Adam kernels')
    stream = baseline._stream_path(adam, 'MA19 original Adam')
    markers = [e for e in events if e.get('cat') == 'user_annotation' and
               e['name'].startswith('ma19.')]
    native = [e for e in kernels if KERNEL in e['name']]
    copies = [e for e in kernels if 'bfloat16_copy_kernel_cuda' in e['name']]
    tail_copies = sorted([e for e in copies if e['ts'] >= adam[-1]['ts']], key=lambda e: e['ts'])
    # The four vocabulary-head conversions belong to forward, before Adam.
    require(len(copies) - len(tail_copies) == 4, 'Original vocabulary-head BF16 conversions changed')
    runtime = [e for rows in index.runtime.values() for e in rows]
    by_correlation = {}
    for event in runtime:
        key = event.get('args', {}).get('correlation')
        if key is not None:
            by_correlation.setdefault(key, []).append(event)

    def launch(event):
        api = _one(by_correlation.get(event.get('args', {}).get('correlation'), []),
                   'Publication/Adam lacks a unique CUDA launch')
        require('LaunchKernel' in api['name'], 'Publication/Adam changed CUDA submission API')
        return api

    adam_launches = [launch(e) for e in adam]
    if candidate:
        scope = _one(markers, 'MA19 needs one parameter publication marker')
        require(scope['name'] == MARKER and len(native) == 1,
                'MA19 needs one parameter publication kernel')
        require(not tail_copies, 'MA19 retained per-parameter copies after Adam')
        work, _ = index.work(scope)
        require(len(work) == 1 and work[0] is native[0],
                'Publication marker does not uniquely own its kernel or contains extra work')
        compute = native
        kernel = native[0]
        require(kernel.get('args', {}).get('grid') == [CHUNKS, 1, 1] and
                kernel['args'].get('block') == [256, 1, 1],
                'MA19 publication launch geometry changed')
        local_runtime = [e for e in runtime if contains(scope, e)]
        require(not any(any(token in e['name'] for token in
                            ('Synchronize', 'Memcpy', 'Memset', 'EventRecord', 'StreamWaitEvent'))
                        for e in local_runtime),
                'MA19 publication inserted copies, clears, synchronization or stream handoff')
        require(not any(contains(scope, e) and e['name'] == 'aten::copy_' for e in index.cpu),
                'MA19 publication fell back to ATen copies')
        cpu_span = scope['dur']
        cpu_work = scope['dur']
    else:
        require(not markers and not native, 'Original MA11 control contains MA19 publication')
        require(len(tail_copies) == PARAMETERS, 'MA11 needs all 389 original parameter copies')
        compute = tail_copies
        owners = []
        for kernel in compute:
            api = launch(kernel)
            owner = _one([e for e in index.cpu if e['name'] == 'aten::copy_' and contains(e, api)],
                         'Original parameter copy lacks its unique ATen owner')
            work, _ = index.work(owner)
            require(len(work) == 1 and work[0] is kernel,
                    'Original parameter copy contains unexpected device work')
            owners.append(owner)
        require(len({id(e) for e in owners}) == PARAMETERS,
                'Original parameter copy ownership is duplicated')
        cpu_span = max(map(end, owners)) - min(e['ts'] for e in owners)
        cpu_work = sum(e['dur'] for e in owners)

    require(baseline._stream_path(compute, 'MA19 parameter publication') == stream and
            end(adam[-1]) <= compute[0]['ts'] + .01,
            'Publication precedes completed Adam or changes compute stream')
    compute_launches = [launch(e) for e in compute]
    require(max(map(end, adam_launches)) <= min(e['ts'] for e in compute_launches) + .01,
            'Publication was submitted before original Adam')
    require(len({e['args']['correlation'] for e in compute}) == len(compute),
            'Publication reuses a CUDA launch correlation')
    after = sorted([e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
                    and e['ts'] > compute[-1]['ts']], key=lambda e: e['ts'])
    require(len(after) == 2 and all(e['cat'] == 'kernel' and 'FillFunctor<float>' in e['name']
                                  for e in after),
            'Expected only the two original gradient-buffer clears after publication')
    require(baseline._stream_path(compute + after, 'Publication before gradient clear') == stream,
            'Parameter publication changed gradient-clear stream')
    return dict(arm=arm, installed=candidate, parameters=PARAMETERS, elements=ELEMENTS,
                original_adam_kernels=49, original_head_conversions=4,
                publication_kernels=len(compute), original_copy_calls=len(tail_copies),
                descriptor_chunks=CHUNKS if candidate else None,
                logical_read_write_bytes=6 * ELEMENTS, stream=stream,
                cpu_submission_span_ms=cpu_span / 1000,
                cpu_scope_work_ms=cpu_work / 1000,
                gpu_publication_work_ms=sum(e['dur'] for e in compute) / 1000,
                gpu_publication_span_ms=(end(compute[-1]) - compute[0]['ts']) / 1000,
                adam_to_publication_gap_ms=(compute[0]['ts'] - end(adam[-1])) / 1000,
                publication_before_original_clears=True,
                limitation='Profile intervals include perturbation. Shared bytes remain; neither '
                           'CPU and GPU sums nor the tail span predict unprofiled step savings.')


def trace_composed_records(events, arm, rank=None):
    require(arm in ARMS, 'Unknown MA19 arm')
    result = baseline.trace_composed_records(events, baseline.ARMS[1], rank)
    result['parameter_publish'] = trace_parameter_publish_records(events, arm)
    return result
