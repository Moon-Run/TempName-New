"""Audit ABC forward plus actual cache, zero-count, softmax and row backward work."""
from collections import defaultdict
from statistics import median

from rtx5090_loader import load
import op1_abc_profile as projection
import op2_abc_protocol as p

base = load('profile')
require, contains, end = base.require, base.contains, base.end


def row_backward(events, rank):
    # ABC uses GemmRSFunction: bias is outside this Function, unlike the CR3
    # ConsumerGemmRSFunction used by the original MA11 audit.
    index = base.ma8_profile.TraceIndex(events)
    forwards = [e for e in events if e.get('cat') == 'cpu_op' and e.get('name') == 'GemmRSFunction']
    backwards = [e for e in events if e.get('cat') == 'cpu_op' and e.get('name') == 'GemmRSFunctionBackward']
    scopes = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name') in ('scope.attn.flux_tensor', 'scope.mlp.flux_tensor')]
    require(len(forwards) == len(backwards) == len(scopes) == 256, 'OP2 row autograd coverage missing')
    roles, seen, claimed, result = {}, set(), set(), defaultdict(list)
    for call in forwards:
        owners = [e for e in scopes if contains(e, call)]
        require(len(owners) == 1, 'OP2 row forward scope missing')
        sequence = call['args']['Sequence number']
        require(sequence not in roles, 'Duplicate OP2 row sequence')
        roles[sequence] = owners[0]['name'].split('.')[1]
    for call in backwards:
        sequence = call['args']['Sequence number']
        require(sequence in roles and sequence not in seen, 'Unpaired OP2 row backward')
        seen.add(sequence)
        work, launches = index.work(call)
        require(not claimed.intersection(map(id, work)), 'Duplicate OP2 backward work')
        claimed.update(map(id, work))
        clears = index.clears(call, work, launches)
        kernels = [e for e in work if e.get('cat') == 'kernel']
        gathers = [e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        gemms = sorted([e for e in kernels if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
        splits = [e for e in kernels if base.ma8_profile._is_splitk(e)]
        require(len(gathers) == 1 and len(gemms) == 2 and len(kernels) == 3+len(splits),
                'OP2 row backward must retain one gather and two GEMMs')
        gather, (dgrad, wgrad) = gathers[0], gemms
        compute = dgrad['args'].get('stream')
        require(compute is not None and gather['args'].get('stream') == wgrad['args'].get('stream') == compute
                and all(e['args'].get('device') == rank for e in work), 'OP2 row backward device/stream changed')
        maths = base.ma8_profile._row_splitk(index, call, launches, gemms, splits)
        require(end(gather) <= dgrad['ts']+.01 and maths['dgrad_end'] <= wgrad['ts']+.01,
                'OP2 row gather/dgrad/wgrad dependency changed')
        result[roles[sequence]].append(dict(cpu_us=call['dur'], allgather_us=gather['dur'],
            dgrad_us=maths['dgrad_us'], wgrad_us=maths['wgrad_us'],
            splitk_reduction_kernels=len(splits), cublas_4byte_memsets=clears.get('cublas_4byte', 0)))
    require(set(result) == {'attn', 'mlp'} and all(len(v) == 128 for v in result.values()),
            'OP2 row backward role coverage changed')
    return {role: dict(calls=128, allgather_calls=128, dgrad_calls=128, wgrad_calls=128,
        splitk_reduction_kernels=sum(v['splitk_reduction_kernels'] for v in rows),
        allowed_cublas_4byte_memsets=sum(v['cublas_4byte_memsets'] for v in rows),
        median_us={key: median(v[key] for v in rows) for key in ('cpu_us', 'allgather_us', 'dgrad_us', 'wgrad_us')})
        for role, rows in result.items()}


def analyze(events, arm, rank):
    require(arm in p.ARMS and type(rank) is int and 0 <= rank < 8, 'Wrong OP2 profile identity')
    return dict(protocol=p.PROTOCOL, arm=arm, rank=rank, execution_validated=True,
        row_forward=projection.chunk_records(events, arm, rank), row_backward=row_backward(events, rank),
        activation_cache=base.ma8_profile.trace_activation_cache_records(events, 'F4-cache'),
        zero_count=base.ma9_profile.trace_zero_count_records(events, 'F4-zcount', rank),
        attention_softmax=base.ma11_profile.trace_attention_softmax_records(events, 'F4-cache-zcount-csoftmax'))
