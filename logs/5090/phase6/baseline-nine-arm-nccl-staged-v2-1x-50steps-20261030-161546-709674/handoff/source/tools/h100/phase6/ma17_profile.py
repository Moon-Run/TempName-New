"""Verify direct column AllGather submission using real CUDA launch ownership.

NCCL may itself record/wait on events: those inside the native-call marker are
reported, not mistaken for ProcessGroup overhead or required to disappear.
"""
from collections import Counter, defaultdict
import statistics
import types

import ma8_profile as cache
import ma11_profile as baseline
from ma8_profile import TraceIndex, contains, end, require

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-directag')
COLUMN = '_MA17DirectAllGatherLinear'
MARKERS = {role: 'ma17.column.gather.' + role for role in ('qkv', 'fc1')}
EVENT_APIS = ('cudaEventRecord', 'cudaEventRecordWithFlags', 'cudaStreamWaitEvent',
              'cuEventRecord', 'cuStreamWaitEvent')


def _one(values, message):
    require(len(values) == 1, message)
    return values[0]


def _pairs(index):
    namespace = dict(cache._raw_column_pairs.__globals__, CACHED_COLUMN=COLUMN)
    parser = types.FunctionType(cache._raw_column_pairs.__code__, namespace)
    return parser(index, 'F4-cache', 'ma8-v2')


def _backward(index, call, role, stream):
    work, launches = index.work(call)
    clears = index.clears(call, work, launches, qkv_bias=role == 'qkv')
    kernels = [e for e in work if e['cat'] == 'kernel']
    gemms = sorted([e for e in kernels if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
    rs = [e for e in kernels if 'nccl' in e['name'].lower() and 'ReduceScatter' in e['name']]
    bias = [e for e in kernels if 'reduce_kernel' in e['name']]
    require(len(kernels) == 3 + int(role == 'qkv') and len(gemms) == 2 and len(rs) == 1
            and len(bias) == int(role == 'qkv'), 'MA17 changed cached column backward')
    dgrad, wgrad = gemms
    scatter = rs[0]
    require(dgrad['args']['stream'] == wgrad['args']['stream'] == stream
            and scatter['args']['stream'] != stream, 'MA17 changed column backward streams')
    require(end(dgrad) <= wgrad['ts'] + .01 and end(dgrad) <= scatter['ts'] + .01,
            'MA17 changed dgrad dependency')
    require(end(launches[dgrad['args']['correlation']]) <= launches[scatter['args']['correlation']]['ts'] + .01
            and end(launches[scatter['args']['correlation']]) <= launches[wgrad['args']['correlation']]['ts'] + .01,
            'MA17 changed dgrad/ReduceScatter/wgrad submission order')
    if bias:
        require(bias[0]['args']['stream'] == stream and end(wgrad) <= bias[0]['ts'] + .01,
                'MA17 moved column bias gradient')
    return work, clears


def _forward(index, call, role, marker=None):
    work, launches = index.work(call)
    clears = index.clears(call, work, launches)
    kernels = [e for e in work if e['cat'] == 'kernel']
    gather = _one([e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']],
                  'MA17 requires exactly one column AllGather kernel')
    gemm = _one([e for e in kernels if 'gemm' in e['name'].lower()],
                'MA17 requires exactly one original full GEMM')
    bias = [e for e in kernels if 'elementwise' in e['name'].lower() and 'add' in e['name'].lower()]
    require(len(kernels) == 2 + int(role == 'qkv') and len(bias) == int(role == 'qkv'),
            'MA17 column forward work changed')
    stream = gemm['args']['stream']
    require(end(gather) <= gemm['ts'] + .01, 'MA17 GEMM consumed an unfinished gather')
    require(end(launches[gather['args']['correlation']]) <= launches[gemm['args']['correlation']]['ts'] + .01,
            'MA17 GEMM submitted before gather')
    if bias:
        require(bias[0]['args']['stream'] == stream and end(gemm) <= bias[0]['ts'] + .01,
                'MA17 changed full bias add')
    runtime = [e for e in index.runtime[call['tid']] if contains(call, e)]
    cpu = [e for e in index.cpu[call['tid']] if contains(call, e)]
    pg = [e for e in cpu if 'c10d::' in e['name']]
    events = [e for e in runtime if e['name'] in EVENT_APIS]
    inner_events = []
    if marker is not None:
        require(marker['name'] == MARKERS[role] and contains(call, marker),
                'MA17 marker role or owner mismatch')
        native_work, _ = index.work(marker)
        require(len(native_work) == 1 and native_work[0] is gather,
                'MA17 native marker lacks unique AllGather launch ownership')
        require(gather['args']['stream'] == stream, 'MA17 AllGather is not on the compute stream')
        require(not pg, 'MA17 fell back to ProcessGroup within column forward')
        require(not any(e['name'] == 'record_param_comms' and
                        e.get('args', {}).get('Collective name') == 'wait' for e in cpu),
                'MA17 retained an explicit ProcessGroup Work.wait')
        inner_events = [e for e in events if contains(marker, e)]
        # Preserve any implementation-internal cuBLAS events in the original
        # GEMM; reject a communication handoff outside either native library.
        gemm_scopes = [e for e in cpu if e['name'] in ('aten::matmul', 'aten::mm')
                       and contains(e, launches[gemm['args']['correlation']])]
        require(all(contains(marker, e) or any(contains(g, e) for g in gemm_scopes)
                    for e in events), 'MA17 retained an external stream handoff')
        require(not any('Synchronize' in e['name'] for e in runtime),
                'MA17 inserted a host/device synchronization')
    else:
        require(gather['args']['stream'] != stream, 'MA11 control communication stream changed')
    return dict(work=work, clears=clears, stream=stream, kernel_name=gather['name'],
                event_counts=dict(Counter(e['name'] for e in events)),
                metrics=dict(forward_cpu_us=call['dur'], gather_gpu_us=gather['dur'],
                    full_gemm_gpu_us=gemm['dur'], gather_to_gemm_gap_us=gemm['ts'] - end(gather),
                    pg_cpu_us=sum(e['dur'] for e in pg), event_api_us=sum(e['dur'] for e in events),
                    native_call_cpu_us=marker['dur'] if marker is not None else 0.,
                    native_internal_event_api_us=sum(e['dur'] for e in inner_events)))


def trace_column_direct_records(events, arm, rank=None):
    require(arm in ARMS, 'Unknown MA17 arm')
    markers = [e for e in events if e.get('cat') == 'user_annotation' and e['name'].startswith('ma17.')]
    candidate = arm == ARMS[1]
    index = TraceIndex(events)
    if candidate:
        require(len(markers) == 256 and all(e['name'] in MARKERS.values() for e in markers),
                'MA17 requires exactly 256 native gather markers')
        pairs, heads = _pairs(index)
    else:
        require(not markers and not any(COLUMN in e.get('name', '') for e in events),
                'MA11 control contains MA17 intervention')
        pairs, heads = cache._raw_column_pairs(index, 'F4-cache', 'ma8-v2')
    head_evidence = cache._head_records(index, heads)
    rows, clears = defaultdict(list), Counter()
    used, scopes, streams = set(), set(), set()
    for role, forward, backward in pairs:
        marker = (_one([e for e in markers if contains(forward, e)],
                       'MA17 forward lacks a unique gather marker') if candidate else None)
        if marker is not None:
            require(id(marker) not in scopes, 'MA17 marker attributed twice')
            scopes.add(id(marker))
        record = _forward(index, forward, role, marker)
        bw, bc = _backward(index, backward, role, record['stream'])
        identities = [id(e) for e in record['work'] + bw]
        require(len(set(identities)) == len(identities) and not used.intersection(identities),
                'MA17 device work attributed twice')
        used.update(identities)
        streams.add(record['stream'])
        clears.update(record['clears']); clears.update(bc)
        rows[role].append(record)
    require(len(streams) == 1 and set(rows) == {'qkv', 'fc1'}
            and all(len(v) == 128 for v in rows.values()), 'MA17 column coverage/compute stream changed')
    roles = {}
    for role, items in rows.items():
        counter = Counter()
        for item in items:
            counter.update(item['event_counts'])
        roles[role] = dict(calls=128, event_api_counts=dict(counter),
            allgather_kernel_counts=dict(Counter(i['kernel_name'] for i in items)),
            median_us={k: statistics.median(i['metrics'][k] for i in items) for k in items[0]['metrics']},
            sum_ms={k: sum(i['metrics'][k] for i in items) / 1000 for k in items[0]['metrics']})
    activation = dict(column_calls=256, retained_forward_allgathers=256, backward_allgathers=0,
        retained_reduce_scatters=256, no_gpu_copies=True, original_backward_gemm_order_and_streams=True,
        original_vocabulary_heads=head_evidence, allowed_workspace_memsets=dict(clears),
        attribution='raw_autograd_sequence_and_native_launch_correlation')
    direct = dict(candidate=candidate, calls=256, rank=rank, direct_compute_stream=candidate,
        compute_stream=next(iter(streams)), gemms_per_call=1, roles=roles,
        limitation='Profiler CPU/GPU intervals include rank skew; they are not additive step savings. '
                   'NCCL-internal events and communicator algorithm selection are reported separately.')
    return activation, direct


def trace_composed_records(events, arm, rank=None):
    activation, direct = trace_column_direct_records(events, arm, rank)
    original = baseline.ARMS[1]
    return dict(activation_cache=activation, column_direct_allgather=direct,
                row_backward=baseline.trace_row_backward_records(events, original),
                zero_count=baseline.trace_zero_count_records(events, original, rank),
                attention_softmax=baseline.trace_attention_softmax_records(events, original))
