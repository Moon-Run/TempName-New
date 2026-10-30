"""Launch-correlated evidence for MA12 fragment publication and slot reuse.

GPU work sums include waiting and profiler perturbation. They are never used
as estimates of removable critical-path time or substitutes for paired steps.
"""
from bisect import bisect_right
from collections import defaultdict
import statistics

import ma8_profile
import ma9_profile
import ma9_sources
from ma8_profile import TraceIndex, contains, end, require

ARMS = ('F4-cache-zcount', 'F4-cache-zcount-tilepub')
WAIT = 'taco_ma12_wait_reuse_kernel'
DONE = 'taco_ma12_publish_done_kernel'
DECODER = 'taco_ma12_decode_consumer_kernel'
BASE_DECODER = 'taco_ma4f_decode_consumer_kernel'


def _arm(arm):
    require(arm in ARMS, 'Unknown MA12 arm')
    return arm == ARMS[1]


def _chain(work, arm, role, tuning=None):
    candidate = _arm(arm)
    ma9_sources.tuning_config(tuning)
    require(role in ('attn', 'mlp'), 'Unknown projection role')
    require(not any(e['cat'] == 'gpu_memcpy' for e in work), 'Unexpected projection copy')
    clears = [e for e in work if e['cat'] == 'gpu_memset']
    # Original CUTLASS initialize() clears a 128-byte workspace on every run.
    # This is retained; it is not a new per-call flags initialization.
    require(len(clears) == 1 and clears[0]['args'].get('bytes') == 128,
            'Original CUTLASS workspace clear changed or flags were cleared per call')
    require(all(e['cat'] in ('kernel', 'gpu_memset') for e in work), 'Unexpected projection work')
    path = sorted((e for e in work if e['cat'] == 'kernel'), key=lambda e: e['ts'])
    require(len(path) == (4 if candidate else 3), 'Wrong MA12 projection kernel count')
    if candidate:
        wait, gemm, decoder, done = path
        require(WAIT in wait['name'] and DONE in done['name'], 'Missing slot ownership handoff')
        require(DECODER in decoder['name'], 'Missing fragment-acquiring decoder')
        for event in (wait, done):
            require(event['args'].get('grid') == [1, 1, 1] and
                    event['args'].get('block') == [32, 1, 1], 'Slot handoff launch changed')
    else:
        gemm, barrier, decoder = path
        require('CudaIpcBarrierAllKernel' in barrier['name'] and BASE_DECODER in decoder['name'],
                'Original MA89 publication path changed')
    require('flux_bf16' in gemm['name'] and 'reducescatter' in gemm['name'], 'Wrong producer')
    require(gemm['args'].get('grid') == [512, 1, 1] and
            gemm['args'].get('block') == [128, 1, 1], 'Producer launch changed')
    require(decoder['args'].get('grid') == [2048, 1, 1] and
            decoder['args'].get('block') == [128, 1, 1], 'Decoder layout changed')
    stream = path[0]['args'].get('stream')
    require(stream is not None and all(e['args'].get('stream') == stream for e in path),
            'Projection moved between CUDA streams')
    require(clears[0]['args'].get('stream') == stream and end(clears[0]) <= gemm['ts'] + .01 and
            (not candidate or end(path[0]) <= clears[0]['ts'] + .01),
            'CUTLASS workspace clear moved across producer/reuse dependency')
    require(all(end(a) <= b['ts'] + .01 for a, b in zip(path, path[1:])),
            'Projection producer/consumer/reuse order changed')
    launch = dict(gemm_grid=gemm['args']['grid'], gemm_block=gemm['args']['block'],
                  decoder_grid=decoder['args']['grid'], decoder_block=decoder['args']['block'],
                  decoder_rows=1, receiver_variant='tilepub' if candidate else 'original',
                  decoder_kernel=DECODER if candidate else BASE_DECODER)
    values = dict(producer_us=gemm['dur'], decoder_us=decoder['dur'],
                  publication_barrier_us=0 if candidate else path[1]['dur'],
                  reuse_wait_us=path[0]['dur'] if candidate else 0,
                  completion_publish_us=path[-1]['dur'] if candidate else 0,
                  chain_us=end(path[-1]) - path[0]['ts'])
    return path, gemm, decoder, launch, values


def trace_operator_path(events, arm, role, calls=4, tuning=None):
    candidate = _arm(arm)
    index = TraceIndex(events)
    used, values, launches = set(), [], []
    for i in range(calls):
        markers = [e for e in events if e.get('cat') == 'user_annotation' and
                   e.get('name') == f'ma4f.{role}.{i}']
        require(len(markers) == 1, 'Missing unique projection marker')
        marker = markers[0]
        work, _ = index.work(marker)
        path, _, _, launch, durations = _chain(work, arm, role, tuning)
        require(not used.intersection(map(id, path)), 'Repeated projection attribution')
        used.update(map(id, path))
        require(not any(e['name'] in ('aten::clone', 'aten::copy_', 'aten::_to_copy') and
                        contains(marker, e) for e in index.cpu[marker['tid']]),
                'Unexpected projection clone')
        values.append(dict(cpu_us=marker['dur'], **durations))
        launches.append(launch)
    require(used == {id(e) for e in events if e.get('cat') == 'kernel'},
            'Unattributed operator kernel')
    return dict(calls=calls, execution_path_validated=True, fused_consumer=True,
                consumer_add_kernels_per_call=0, fragment_publication=candidate,
                publication_barriers_per_call=0 if candidate else 1,
                launch=launches[0], median_us={k: statistics.median(v[k] for v in values)
                                             for k in values[0]})


def trace_forward_records(events, arm, tuning=None):
    candidate = _arm(arm)
    index = TraceIndex(events)
    scopes, streams, host_external = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in events:
        if event.get('cat') == 'user_annotation' and event.get('name') in (
                'scope.attn.candidate', 'scope.mlp.candidate'):
            scopes[event['tid']].append(event)
        if event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            streams[event.get('args', {}).get('stream')].append(event)
        if event.get('cat') == 'cpu_op':
            host_external[event.get('args', {}).get('External id')].append(event)
    for rows in scopes.values():
        rows.sort(key=lambda e: e['ts'])
    starts = {tid: [e['ts'] for e in rows] for tid, rows in scopes.items()}
    positions = {}
    for stream, rows in streams.items():
        rows.sort(key=lambda e: e['ts'])
        positions.update({id(e): (stream, i) for i, e in enumerate(rows)})
    calls = [e for rows in index.cpu.values() for e in rows
             if e['name'] in ('GemmRSFunction', 'ConsumerGemmRSFunction')]
    result, used_scopes, used_kernels = defaultdict(list), set(), set()
    for call in calls:
        tid = call['tid']
        pos = bisect_right(starts.get(tid, []), call['ts']) - 1
        require(pos >= 0 and contains(scopes[tid][pos], call) and
                (tid, pos) not in used_scopes, 'Projection scope ownership changed')
        scope = scopes[tid][pos]
        role = scope['name'].split('.')[1]
        used_scopes.add((tid, pos))
        require(not any(e['name'] in ('aten::clone', 'aten::copy_', 'aten::_to_copy') and
                        contains(call, e) for e in index.cpu[tid]), 'Projection acquired a copy')
        work, _ = index.work(call)
        path, gemm, decoder, launch, durations = _chain(work, arm, role, tuning)
        require(not used_kernels.intersection(map(id, path)), 'Repeated GPU projection attribution')
        used_kernels.update(map(id, path))
        # Candidate completion publication precedes the original LayerNorm;
        # it must not be mistaken for a standalone bias/residual add.
        stream, position = positions[id(path[-1])]
        require(position + 1 < len(streams[stream]), 'Missing following LayerNorm')
        norm = streams[stream][position + 1]
        name = norm.get('name', '').lower()
        require(norm['cat'] == 'kernel' and ('layernorm' in name or 'layer_norm' in name),
                'Projection no longer directly feeds original LayerNorm')
        require(any(e['name'] == 'FusedLayerNormAffineFunction' for e in
                    host_external[norm.get('args', {}).get('External id')]),
                'LayerNorm lacks its original CPU owner')
        require(end(path[-1]) <= norm['ts'] + .01, 'LayerNorm ran before projection completion')
        result[role].append(dict(cpu_us=call['dur'], launch=launch, **durations,
            decoder_start_to_layernorm_end_us=end(norm) - decoder['ts'],
            gemm_end_to_layernorm_end_us=end(norm) - end(gemm)))
    require(len(used_scopes) == sum(map(len, scopes.values())) == 256 and
            set(result) == {'attn', 'mlp'} and all(len(v) == 128 for v in result.values()),
            'Incomplete projection layer/microbatch coverage')
    named = {id(e) for e in events if e.get('cat') == 'kernel' and
             any(token in e['name'] for token in (WAIT, DONE, DECODER, BASE_DECODER))}
    require(named <= used_kernels, 'Unowned MA12/MA89 decoder or handoff kernel')
    return {role: dict(projection_calls=128, publication_barriers=0 if candidate else 128,
        reuse_waits=128 if candidate else 0, completion_publications=128 if candidate else 0,
        fragment_publication=candidate, consumer_add_kernels=0, layernorm_successors=128,
        fused_consumer=True, launch=rows[0]['launch'],
        **{key: statistics.median(row[key] for row in rows) for key in rows[0] if key != 'launch'})
        for role, rows in result.items()}


def trace_composed_records(events, arm, rank=None):
    _arm(arm)
    return dict(activation_cache=ma8_profile.trace_activation_cache_records(events, 'F4-cache'),
                zero_count=ma9_profile.trace_zero_count_records(events, 'F4-zcount', rank),
                row_backward=ma8_profile.trace_row_backward_records(events, 'F4-cache'))


def trace_projection_records(events, arm, tuning=None):
    forward = trace_forward_records(events, arm, tuning)
    evidence = trace_composed_records(events, arm)
    result = {role: dict(value, backward=evidence['row_backward'][role])
              for role, value in forward.items()}
    result.update(activation_cache=evidence['activation_cache'], zero_count=evidence['zero_count'])
    return result
