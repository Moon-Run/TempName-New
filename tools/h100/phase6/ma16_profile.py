"""MA16 launch ownership, remote-read dependencies and actual GPU overlap.

All times here are profiler observations, never predicted optimizer savings.
The control keeps the original automatic MA8 events with no MA16 markers.
"""
from collections import Counter, defaultdict
import re
import statistics
import types

import ma8_profile as cache
import ma11_profile as baseline
from ma8_profile import TraceIndex, contains, end, require

ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-localfirst')
COLUMN = '_MA16LocalFirstLinear'
OUTER = re.compile(r'^ma16\.column\.forward\.(qkv|fc1)\.rank([0-7])$')
STAGES = ('gather', 'local', 'wait', 'prefix', 'suffix')


def _one(values, message):
    require(len(values) == 1, message)
    return values[0]


def _overlap(left, right):
    return max(0., min(end(left), end(right)) - max(left['ts'], right['ts']))


def _pairs(index):
    # Parameterize only the expected class name, without modifying trace events
    # or the original parser's globals. Sequence/topology checks remain intact.
    namespace = dict(cache._raw_column_pairs.__globals__, CACHED_COLUMN=COLUMN)
    parser = types.FunctionType(cache._raw_column_pairs.__code__, namespace)
    return parser(index, 'F4-cache', 'ma8-v2')


def _backward(index, call, role, compute_stream):
    work, launches = index.work(call)
    clears = index.clears(call, work, launches, qkv_bias=role == 'qkv')
    kernels = [e for e in work if e['cat'] == 'kernel']
    gemms = sorted([e for e in kernels if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
    rs = [e for e in kernels if 'nccl' in e['name'].lower() and 'ReduceScatter' in e['name']]
    bias = [e for e in kernels if 'reduce_kernel' in e['name']]
    require(len(kernels) == 3 + int(role == 'qkv') and len(gemms) == 2 and len(rs) == 1
            and len(bias) == int(role == 'qkv'), 'MA16 changed cached column backward or restored AllGather')
    dgrad, wgrad = gemms
    scatter = rs[0]
    require(dgrad['args']['stream'] == wgrad['args']['stream'] == compute_stream
            and scatter['args']['stream'] != compute_stream, 'MA16 changed column backward streams')
    require(end(dgrad) <= wgrad['ts'] + .01 and end(dgrad) <= scatter['ts'] + .01,
            'MA16 changed dgrad dependency')
    require(end(launches[dgrad['args']['correlation']]) <= launches[scatter['args']['correlation']]['ts'] + .01
            and end(launches[scatter['args']['correlation']]) <= launches[wgrad['args']['correlation']]['ts'] + .01,
            'MA16 changed dgrad/ReduceScatter/wgrad submission order')
    if bias:
        require(bias[0]['args']['stream'] == compute_stream and end(wgrad) <= bias[0]['ts'] + .01,
                'MA16 changed column bias gradient')
    return work, clears


def _forward(index, call, role, outer, stages):
    match = OUTER.fullmatch(outer['name'])
    require(match is not None and match[1] == role and contains(call, outer),
            'MA16 column marker role/ownership mismatch')
    rank = int(match[2])
    wanted = ['gather', 'local', 'wait'] + (['prefix'] if rank else []) + (['suffix'] if rank < 7 else [])
    selected = {stage: [e for e in stages if e['name'] == 'ma16.column.' + stage and contains(outer, e)]
                for stage in STAGES}
    require(all(len(selected[s]) == int(s in wanted) for s in STAGES),
            'MA16 local/wait/remote stage coverage differs from rank placement')
    scopes = {s: selected[s][0] for s in wanted}
    require(all(end(scopes[a]) <= scopes[b]['ts'] + .01 for a, b in zip(wanted, wanted[1:])),
            'MA16 must submit gather, local GEMM, wait, then remote GEMMs')
    work, launches = index.work(call)
    clears = index.clears(call, work, launches)
    kernels = [e for e in work if e['cat'] == 'kernel']
    owned = {}
    for stage, scope in scopes.items():
        stage_work, _ = index.work(scope)
        stage_kernels = [e for e in stage_work if e['cat'] == 'kernel']
        if stage == 'wait':
            require(not stage_work, 'MA16 wait unexpectedly owns GPU work')
            waits = [e for e in index.runtime[scope['tid']] if contains(scope, e) and
                     e['name'] in ('cudaStreamWaitEvent', 'cuStreamWaitEvent')]
            require(waits, 'MA16 wait marker lacks the actual CUDA stream dependency')
        else:
            owned[stage] = _one(stage_kernels, 'MA16 stage must own exactly one kernel: ' + stage)
    gather, local = owned['gather'], owned['local']
    remote = [owned[s] for s in ('prefix', 'suffix') if s in owned]
    require('nccl' in gather['name'].lower() and 'AllGather' in gather['name'],
            'MA16 gather is not the original NCCL AllGather')
    require(all('gemm' in e['name'].lower() for e in [local] + remote),
            'MA16 local/remote stages must launch GEMMs')
    stream = local['args']['stream']
    require(gather['args']['stream'] != stream and all(e['args']['stream'] == stream for e in remote),
            'MA16 communication and compute streams are not independent')
    require(all(end(gather) <= e['ts'] + .01 for e in remote),
            'MA16 remote GEMM consumed an unfinished gather')
    require(all(end(a) <= b['ts'] + .01 for a, b in zip([local] + remote, remote)),
            'MA16 local/remote GEMM stream order changed')
    extras = [e for e in kernels if id(e) not in {id(v) for v in owned.values()}]
    require(len(extras) == int(role == 'qkv') and
            all('elementwise' in e['name'].lower() and 'add' in e['name'].lower() for e in extras),
            'MA16 column contains unexpected work or changed bias addition')
    if extras:
        require(extras[0]['args']['stream'] == stream and end(remote[-1]) <= extras[0]['ts'] + .01,
                'MA16 full bias add precedes a projection')
    require(len(kernels) == len(owned) + len(extras), 'MA16 kernel ownership duplicated')
    return dict(rank=rank, stream=stream, work=work, clears=clears,
                metrics=dict(allgather_us=gather['dur'], local_gemm_us=local['dur'],
                    remote_gemm_us=sum(e['dur'] for e in remote),
                    local_allgather_overlap_us=_overlap(gather, local),
                    allgather_tail_after_local_us=max(0., end(gather) - end(local)),
                    forward_cpu_us=call['dur'],
                    gpu_envelope_us=max(end(e) for e in kernels) - min(e['ts'] for e in kernels)))


def trace_column_overlap_records(events, arm, rank=None):
    require(arm in ARMS, 'Unknown MA16 arm')
    markers = [e for e in events if e.get('cat') == 'user_annotation' and e['name'].startswith('ma16.')]
    if arm == ARMS[0]:
        require(not markers and not any(COLUMN in e.get('name', '') for e in events),
                'MA11 control contains MA16 intervention')
        activation = cache.trace_activation_cache_records(events, 'F4-cache')
        index = TraceIndex(events)
        pairs, _ = cache._raw_column_pairs(index, 'F4-cache', 'ma8-v2')
        records = defaultdict(list)
        for role, forward, _ in pairs:
            work, _ = index.work(forward)
            ag = _one([e for e in work if e['cat'] == 'kernel' and 'AllGather' in e['name']], 'Missing control gather')
            mm = _one([e for e in work if e['cat'] == 'kernel' and 'gemm' in e['name'].lower()], 'Missing control GEMM')
            records[role].append(dict(allgather_us=ag['dur'], full_gemm_us=mm['dur'],
                                     local_allgather_overlap_us=_overlap(ag, mm)))
        result = dict(candidate=False, overlap_observed=False, calls=256,
                      roles={role: dict(calls=128, overlapping_calls=0,
                          sum_ms={k: sum(v[k] for v in rows) / 1000 for k in rows[0]})
                             for role, rows in records.items()})
        return activation, result

    index = TraceIndex(events)
    pairs, heads = _pairs(index)
    head_evidence = cache._head_records(index, heads)
    outers = [e for e in markers if OUTER.fullmatch(e['name'])]
    stages = [e for e in markers if e['name'] in ['ma16.column.' + s for s in STAGES]]
    require(len(outers) == 256 and len(outers) + len(stages) == len(markers),
            'Missing or unexpected MA16 column profile markers')
    records, clear_counts = defaultdict(list), Counter()
    used, used_scopes, ranks, streams = set(), set(), set(), set()
    for role, forward, backward in pairs:
        outer = _one([e for e in outers if contains(forward, e)], 'MA16 forward lacks unique marker')
        require(id(outer) not in used_scopes, 'MA16 marker attributed twice')
        used_scopes.add(id(outer))
        row = _forward(index, forward, role, outer, stages)
        bw, clears = _backward(index, backward, role, row['stream'])
        identities = [id(e) for e in row['work'] + bw]
        require(len(set(identities)) == len(identities) and not used.intersection(identities),
                'MA16 column device work attributed twice')
        used.update(identities)
        ranks.add(row['rank']); streams.add(row['stream'])
        clear_counts.update(row['clears']); clear_counts.update(clears)
        records[role].append(row['metrics'])
    require(len(ranks) == len(streams) == 1 and (rank is None or ranks == {rank}),
            'MA16 column rank or compute stream changed')
    require(set(records) == {'qkv', 'fc1'} and all(len(rows) == 128 for rows in records.values()),
            'MA16 requires 128 calls of each column role')
    resolved_rank = next(iter(ranks))
    require(len(stages) == 256 * (4 if resolved_rank in (0, 7) else 5),
            'MA16 stage marker escaped its column owner')
    roles = {role: dict(calls=len(rows),
                       overlapping_calls=sum(r['local_allgather_overlap_us'] > .01 for r in rows),
                       median_us={k: statistics.median(r[k] for r in rows) for k in rows[0]},
                       sum_ms={k: sum(r[k] for r in rows) / 1000 for k in rows[0]})
             for role, rows in records.items()}
    # A valid async API trace with zero device intersection is a failed overlap
    # experiment, not evidence that the proposed mechanism has been realized.
    require(all(row['overlapping_calls'] > 0 for row in roles.values()),
            'MA16 async submission produced no observed GPU overlap for one or both column roles')
    activation = dict(column_calls=256, retained_forward_allgathers=256, backward_allgathers=0,
                      retained_reduce_scatters=256, no_gpu_copies=True,
                      original_backward_gemm_order_and_streams=True,
                      original_vocabulary_heads=head_evidence,
                      attribution='raw_autograd_sequence_and_candidate_stage_launch_correlation',
                      allowed_workspace_memsets=dict(clear_counts))
    result = dict(candidate=True, overlap_observed=True, calls=256, rank=resolved_rank,
                  compute_stream=next(iter(streams)),
                  gemms_per_call=2 if resolved_rank in (0, 7) else 3, roles=roles,
                  limitation='Profiler intervals include rank skew and instrumentation; overlap is not net step savings.')
    return activation, result


def trace_composed_records(events, arm, rank=None):
    activation, overlap = trace_column_overlap_records(events, arm, rank)
    original = baseline.ARMS[1]
    return dict(activation_cache=activation, column_overlap=overlap,
                row_backward=baseline.trace_row_backward_records(events, original),
                zero_count=baseline.trace_zero_count_records(events, original, rank),
                attention_softmax=baseline.trace_attention_softmax_records(events, original))
