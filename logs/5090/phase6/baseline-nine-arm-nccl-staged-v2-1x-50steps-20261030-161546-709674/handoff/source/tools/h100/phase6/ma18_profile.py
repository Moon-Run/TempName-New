"""Audit MA18 IPC payload copies, original publication and complete MA11 math.

Copies must be CUDA-API correlated, correctly sized and ordered on the compute
stream. A Python marker alone cannot certify the mechanism. Profiler intervals
include peer waiting and do not predict unprofiled optimizer savings.
"""
from collections import Counter, defaultdict
import statistics
import types

import ma8_profile as cache
import ma11_profile as baseline
from ma8_profile import TraceIndex, contains, end, require

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-ipcag')
COLUMN = '_MA18IpcAllGatherLinear'
MARKERS = {role: 'ma18.column.gather.' + role for role in ('qkv', 'fc1')}
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
            and len(bias) == int(role == 'qkv'), 'MA18 changed cached column backward')
    dgrad, wgrad = gemms
    scatter = rs[0]
    require(dgrad['args']['stream'] == wgrad['args']['stream'] == stream
            and scatter['args']['stream'] != stream, 'MA18 changed column backward streams')
    require(end(dgrad) <= wgrad['ts'] + .01 and end(dgrad) <= scatter['ts'] + .01,
            'MA18 changed dgrad dependency')
    require(end(launches[dgrad['args']['correlation']]) <= launches[scatter['args']['correlation']]['ts'] + .01
            and end(launches[scatter['args']['correlation']]) <= launches[wgrad['args']['correlation']]['ts'] + .01,
            'MA18 changed dgrad/ReduceScatter/wgrad submission order')
    if bias:
        require(bias[0]['args']['stream'] == stream and end(wgrad) <= bias[0]['ts'] + .01,
                'MA18 moved column bias gradient')
    return work, clears


def _forward(index, call, role, marker=None):
    work, launches = index.work(call)
    kernels = [e for e in work if e['cat'] == 'kernel']
    copies = [e for e in work if e['cat'] == 'gpu_memcpy']
    gemm = _one([e for e in kernels if 'gemm' in e['name'].lower()],
                'MA18 requires exactly one original full GEMM')
    bias = [e for e in kernels if 'elementwise' in e['name'].lower() and 'add' in e['name'].lower()]
    stream = gemm['args']['stream']
    runtime = [e for e in index.runtime[call['tid']] if contains(call, e)]
    cpu = [e for e in index.cpu[call['tid']] if contains(call, e)]
    pg = [e for e in cpu if 'c10d::' in e['name']]
    events = [e for e in runtime if e['name'] in EVENT_APIS]
    if marker is None:
        require(not copies, 'MA11 control added payload copies')
        gather = _one([e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']],
                      'MA11 control requires the original NCCL AllGather')
        require(gather['args']['stream'] != stream, 'MA11 control communication stream changed')
        transfer_end = end(gather)
        transport_name = gather['name']
        transport_us, barrier_us = gather['dur'], 0.
    else:
        require(marker['name'] == MARKERS[role] and contains(call, marker),
                'MA18 marker role or owner mismatch')
        native_work, native_launches = index.work(marker)
        barrier = _one([e for e in kernels if 'CudaIpcBarrierAllKernel(' in e['name']],
                       'MA18 requires exactly one original binary IPC barrier')
        require(len(copies) == 9 and len(native_work) == 10 and
                {id(e) for e in native_work} == {id(e) for e in copies + [barrier]},
                'MA18 native marker must own one publication barrier and nine copies')
        require(all(e['args'].get('bytes') == 2097152 and
                    ('DtoD' in e['name'] or 'PtoP' in e['name']) for e in copies),
                'MA18 requires nine 2MiB device-to-device payload copies')
        require(len({e['args']['correlation'] for e in native_work}) == 10,
                'MA18 duplicated native transfer launch correlation')
        require(all('MemcpyAsync' in native_launches[e['args']['correlation']]['name']
                    for e in copies), 'MA18 copies lack asynchronous memcpy API ownership')
        require(all(e['args']['stream'] == stream for e in native_work),
                'MA18 payload and barrier must use the original compute stream')
        ordered = sorted(native_work, key=lambda e: native_launches[e['args']['correlation']]['ts'])
        require(ordered[0]['cat'] == 'gpu_memcpy' and ordered[1] is barrier and
                all(e['cat'] == 'gpu_memcpy' for e in ordered[2:]),
                'MA18 submission must stage, publish, then pull eight sources')
        require(all(end(a) <= b['ts'] + .01 for a, b in zip(ordered, ordered[1:])),
                'MA18 staged data/publication/pull device dependencies changed')
        require(end(native_launches[ordered[-1]['args']['correlation']]) <=
                launches[gemm['args']['correlation']]['ts'] + .01,
                'MA18 full GEMM submitted before final pull')
        require(not pg and not any('nccl' in e['name'].lower() for e in native_work),
                'MA18 fell back to NCCL/ProcessGroup in target gather')
        require(not any(e['name'] == 'record_param_comms' and
                        e.get('args', {}).get('Collective name') == 'wait' for e in cpu),
                'MA18 retained an explicit ProcessGroup Work.wait')
        require(not any('Synchronize' in e['name'] for e in runtime),
                'MA18 inserted a host/device synchronization')
        gemm_scopes = [e for e in cpu if e['name'] in ('aten::matmul', 'aten::mm')
                       and contains(e, launches[gemm['args']['correlation']])]
        require(all(any(contains(g, e) for g in gemm_scopes) for e in events),
                'MA18 inserted a communication stream handoff')
        transfer_end = end(ordered[-1])
        transport_name = 'IPC async memcpy + original binary publication'
        transport_us = sum(e['dur'] for e in copies)
        barrier_us = barrier['dur']
    require(len(kernels) == 2 + int(role == 'qkv') and len(bias) == int(role == 'qkv'),
            'MA18 column forward contains extra kernels or changed full GEMM/bias')
    require(transfer_end <= gemm['ts'] + .01, 'MA18 full GEMM consumed an unfinished gather')
    if bias:
        require(bias[0]['args']['stream'] == stream and end(gemm) <= bias[0]['ts'] + .01,
                'MA18 changed full bias add')
    # Accept copies only after they have been fully attributed and checked.
    clears = index.clears(call, [e for e in work if e['cat'] != 'gpu_memcpy'], launches)
    return dict(work=work, clears=clears, stream=stream, kernel_name=transport_name,
                event_counts=dict(Counter(e['name'] for e in events)),
                metrics=dict(forward_cpu_us=call['dur'], transport_gpu_us=transport_us,
                    publication_gpu_us=barrier_us, copy_count=len(copies),
                    full_gemm_gpu_us=gemm['dur'], gather_to_gemm_gap_us=gemm['ts'] - transfer_end,
                    pg_cpu_us=sum(e['dur'] for e in pg), event_api_us=sum(e['dur'] for e in events),
                    native_call_cpu_us=marker['dur'] if marker is not None else 0.))


def trace_column_ipc_records(events, arm, rank=None):
    require(arm in ARMS, 'Unknown MA18 arm')
    markers = [e for e in events if e.get('cat') == 'user_annotation' and e['name'].startswith('ma18.')]
    candidate = arm == ARMS[1]
    index = TraceIndex(events)
    if candidate:
        require(len(markers) == 256 and all(e['name'] in MARKERS.values() for e in markers),
                'MA18 requires exactly 256 native gather markers')
        pairs, heads = _pairs(index)
    else:
        require(not markers and not any(COLUMN in e.get('name', '') for e in events),
                'MA11 control contains MA18 intervention')
        pairs, heads = cache._raw_column_pairs(index, 'F4-cache', 'ma8-v2')
    head_evidence = cache._head_records(index, heads)
    rows, clears = defaultdict(list), Counter()
    used, scopes, streams = set(), set(), set()
    for role, forward, backward in pairs:
        marker = (_one([e for e in markers if contains(forward, e)],
                       'MA18 forward lacks a unique gather marker') if candidate else None)
        if marker is not None:
            require(id(marker) not in scopes, 'MA18 marker attributed twice')
            scopes.add(id(marker))
        record = _forward(index, forward, role, marker)
        bw, bc = _backward(index, backward, role, record['stream'])
        identities = [id(e) for e in record['work'] + bw]
        require(len(set(identities)) == len(identities) and not used.intersection(identities),
                'MA18 device work attributed twice')
        used.update(identities)
        streams.add(record['stream'])
        clears.update(record['clears']); clears.update(bc)
        rows[role].append(record)
    require(len(streams) == 1 and set(rows) == {'qkv', 'fc1'}
            and all(len(v) == 128 for v in rows.values()), 'MA18 column coverage/compute stream changed')
    roles = {}
    for role, items in rows.items():
        counter = Counter()
        for item in items:
            counter.update(item['event_counts'])
        roles[role] = dict(calls=128, event_api_counts=dict(counter),
            transport_counts=dict(Counter(i['kernel_name'] for i in items)),
            median_us={k: statistics.median(i['metrics'][k] for i in items) for k in items[0]['metrics']},
            sum_ms={k: sum(i['metrics'][k] for i in items) / 1000 for k in items[0]['metrics']})
    activation = dict(column_calls=256, retained_forward_allgathers=256, backward_allgathers=0,
        retained_reduce_scatters=256, no_gpu_copies=not candidate,
        validated_transport_copies=2304 if candidate else 0, original_backward_gemm_order_and_streams=True,
        original_vocabulary_heads=head_evidence, allowed_workspace_memsets=dict(clears),
        attribution='raw_autograd_sequence_and_native_launch_correlation')
    direct = dict(candidate=candidate, calls=256, rank=rank, ipc_dma_transport=candidate,
        native_copy_count=2304 if candidate else 0, publication_barriers=256 if candidate else 0,
        compute_stream=next(iter(streams)), gemms_per_call=1, roles=roles,
        limitation='Profiler CPU/GPU intervals include rank skew; they are not additive step savings. '
                   'Memcpy trace proves the chosen payload path, not net savings or zero SM usage: the barrier uses SMs.')
    return activation, direct


def trace_composed_records(events, arm, rank=None):
    activation, direct = trace_column_ipc_records(events, arm, rank)
    original = baseline.ARMS[1]
    return dict(activation_cache=activation, ipc_allgather=direct,
                row_backward=baseline.trace_row_backward_records(events, original),
                zero_count=baseline.trace_zero_count_records(events, original, rank),
                attention_softmax=baseline.trace_attention_softmax_records(events, original))
