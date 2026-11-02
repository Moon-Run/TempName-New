"""Launch-owned OP3 forward, fused consumer and unchanged backward audits."""
from collections import Counter, defaultdict
from statistics import median

from rtx5090_loader import load
import op3_protocol as p
import op2_abc_profile as previous
from op1_a_swizzle import TYPE_NAME

base = load('profile')
require, contains, end = base.require, base.contains, base.end
ENCODERS = {'op3-combined': 'op3_warp_encode_kernel', 'op3-r': 'op1_abc_encode_kernel', 'op3-improved': 'op3_warp_encode_kernel'}
DECODERS = {'op3-combined': 'op3_r_decode_kernel', 'op3-r': 'op3_r_decode_kernel', 'op3-improved': 'op1_abc_decode_kernel'}


def one(events, message):
    require(len(events) == 1, message)
    return events[0]


def before(first, second, message):
    require(end(first) <= second['ts'] + .01, message)


def chunk_records(events, arm, rank, *, role_counts=None, operator=False):
    require(arm in p.ALL_ARMS, 'Unknown OP3 forward arm')
    role_counts = role_counts or {'attn': 128, 'mlp': 128}
    require(set(role_counts) <= {'attn', 'mlp'} and all(type(n) is int and n > 0 for n in role_counts.values()), 'Invalid OP3 profile counts')
    chunks = 1
    ENCODER, DECODER = ENCODERS[arm], DECODERS[arm]
    fused = p.fused(arm)
    index = base.Index(events)
    cpu_by_external = defaultdict(list)
    for event in index.cpu:
        external = event.get('args', {}).get('External id')
        if external is not None:
            cpu_by_external[external].append(event)
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
              and e.get('name') in ('scope.attn.flux_tensor', 'scope.mlp.flux_tensor')]
    require(Counter(e['name'] for e in scopes) == Counter(
        {f'scope.{role}.flux_tensor': count for role, count in role_counts.items()}), 'Incomplete OP3 projection coverage')
    markers = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
               and (e.get('name', '').startswith('op3.') or e.get('name') == 'ablation6.flux_gemm_only')]
    cuda_waits = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                  and e.get('name') == 'cudaStreamWaitEvent']
    streams = defaultdict(list)
    for e in events:
        if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            streams[e.get('args', {}).get('stream')].append(e)
    positions = {}
    for stream, rows in streams.items():
        rows.sort(key=lambda e: e['ts'])
        positions.update({id(e): (stream, i) for i, e in enumerate(rows)})
    claimed, claimed_markers, results = set(), set(), defaultdict(list)
    for scope in scopes:
        role = scope['name'].split('.')[1]
        stages = [e for e in markers if contains(scope, e)]
        expected = Counter({'ablation6.flux_gemm_only': 1})
        expected.update({f'op3.{stage}.{i}': 1
                         for stage in ('encode', 'exchange', 'decode') for i in range(chunks)})
        if chunks > 1:
            expected['op3.join'] = 1
        require(Counter(e['name'] for e in stages) == expected, 'Wrong OP3 chunk/join stage coverage')
        require(not claimed_markers.intersection(map(id, stages)), 'Duplicate OP3 marker ownership')
        claimed_markers.update(map(id, stages))

        def stage(name):
            return one([e for e in stages if e['name'] == name], 'Missing OP3 stage '+name)

        gemm_stage = stage('ablation6.flux_gemm_only')
        gemm_work, _ = index.work(gemm_stage)
        gemm = one([e for e in gemm_work if e['cat'] == 'kernel'], 'Expected one OP3 GEMM')
        require(base.is_basic_gemm(gemm['name']) and TYPE_NAME in gemm['name']
                and '_streamksk_' in gemm['name'].lower(), 'OP3 actual GEMM lacks interleaved Stream-K')
        require(all(e['cat'] in ('kernel', 'gpu_memset') for e in gemm_work), 'Unexpected OP3 GEMM work')
        work, decoders, communication = list(gemm_work), [], []
        for chunk in range(chunks):
            encode = stage(f'op3.encode.{chunk}')
            exchange = stage(f'op3.exchange.{chunk}')
            decode = stage(f'op3.decode.{chunk}')
            before(gemm_stage, encode, 'OP3 encode submitted before GEMM')
            before(encode, exchange, 'OP3 exchange submitted before encode')
            before(exchange, decode, 'OP3 decode submitted before exchange wait')
            encoded, _ = index.work(encode)
            decoded, _ = index.work(decode)
            encoder = one(encoded, 'Expected one direct OP3 encoder launch')
            decoder = one(decoded, 'Expected one OP3 decode/reduce launch')
            require(encoder['cat'] == decoder['cat'] == 'kernel' and ENCODER in encoder['name']
                    and DECODER in decoder['name'], 'Wrong OP3 encoder/decoder')
            if fused:
                kernel_name = ''.join(decoder['name'].split())
                require(any(token in kernel_name for token in
                            ('op3_r_decode_kernel<true>', 'op3_r_decode_kernel<(bool)1>')),
                        'R profile must launch the fused consumer specialization')
            exchanged, _ = index.work(exchange)
            kernels = [e for e in exchanged if e['cat'] == 'kernel']
            require(kernels and all('nccl' in e['name'].lower() for e in kernels)
                    and all(e['cat'] in ('kernel', 'gpu_memcpy', 'gpu_memset') for e in exchanged),
                    'OP3 exchange lacks actual NCCL work')
            calls = [e for e in index.cpu if contains(exchange, e) and e['name'].startswith('c10d::')
                     and 'alltoall' in e['name'].lower().replace('_', '')]
            collective = one(calls, 'Expected one c10d All-to-All per OP3 chunk')
            waits = [e for e in index.cpu if contains(exchange, e)
                     and e['name'] == 'record_param_comms'
                     and e.get('args', {}).get('Collective name') == 'wait']
            waited = one(waits, 'Expected one actual NCCL Work.wait annotation')
            stream_waits = [e for e in cuda_waits if contains(waited, e)]
            require(stream_waits, 'NCCL Work.wait lacks its CUDA stream dependency')
            before(collective, waited, 'NCCL wait precedes collective submission')
            before(waited, decode, 'Decode submission precedes NCCL Work.wait')
            owned, _ = index.work(collective)
            require({id(e) for e in kernels} <= {id(e) for e in owned}, 'Unowned OP3 NCCL kernel')
            before(gemm, encoder, 'OP3 encoder read unfinished GEMM')
            for e in exchanged:
                before(encoder, e, 'OP3 NCCL read unfinished encoded chunk')
                before(e, decoder, 'OP3 decoder read unfinished received chunk')
            if decoders:
                before(decoders[-1], decoder, 'OP3 decoder stream is not ordered')
            decoders.append(decoder)
            communication.append(exchanged)
            work += encoded + exchanged + decoded
        compute = gemm['args'].get('stream')
        decode_streams = {e['args'].get('stream') for e in decoders}
        require(compute is not None and len(decode_streams) == 1
                and ((compute in decode_streams) == (chunks == 1)), 'OP3 decoder stream contract changed')
        if chunks > 1:
            join = stage('op3.join')
            require(not index.work(join)[0], 'OP3 final join must only establish a stream dependency')
            before(stage(f'op3.decode.{chunks-1}'), join, 'OP3 join precedes final decode submission')
        require(all(e.get('args', {}).get('device') == rank for e in work), 'Wrong OP3 CUDA device')
        outer, _ = index.work(scope)
        require({id(e) for e in work} == {id(e) for e in outer}, 'Unregistered OP3 projection work')
        require(not claimed.intersection(map(id, work)), 'Duplicate OP3 work ownership')
        claimed.update(map(id, work))
        cpu = [e for e in index.cpu if contains(scope, e)]
        require(not any(e['name'] in ('aten::cat', 'aten::clone', 'aten::add') for e in cpu),
                'OP3 reintroduced remote concatenation, clone or separate reduction')

        # Only the R decoder includes the consumer. The independent encoder
        # candidate must retain both separately rounded BF16 addition kernels.
        tail = max((e for e in work if e['args'].get('stream') == compute), key=end)
        _, pos = positions[id(tail)]
        successors = []
        if not operator:
            successor_count = 1 if fused else 3
            successors = streams[compute][pos+1:pos+1+successor_count]
            require(len(successors) == successor_count
                    and (fused or all(base._bf16_add(e) for e in successors[:2]))
                    and base._layernorm_backend(successors[-1]) is not None,
                    'OP3 output has the wrong fused/separate bias/residual/LayerNorm consumers')
            for e in decoders:
                before(e, successors[0], 'OP3 consumer raced an unfinished decoder')
            for first, second in zip(successors, successors[1:]):
                before(first, second, 'OP3 consumer order changed')
            for i, e in enumerate(successors):
                external = e.get('args', {}).get('External id')
                owners = [owner for owner in cpu_by_external[external] if external is not None
                          and any(id(w) == id(e) for w in index.work(owner)[0])]
                expected_name = ('aten::add' if not fused and i < 2 else
                    '_OperationFuserAutogradFunction' if base._layernorm_backend(e) == 'transformer_engine'
                    else 'FusedLayerNormAffineFunction')
                require(any(owner['name'] == expected_name for owner in owners), 'OP3 consumer lacks launch owner')
        overlaps = sum(max(0., min(end(decoder), end(comm))-max(decoder['ts'], comm['ts']))
                       for i, decoder in enumerate(decoders) for later in communication[i+1:]
                       for comm in later if comm['cat'] == 'kernel')
        results[role].append(dict(overlap=overlaps,
            decode_to_consumer_us=end(successors[-1] if successors else tail)-min(e['ts'] for e in decoders),
            gemm_us=gemm['dur'], encode_us=encoder['dur'], decode_us=decoder['dur'],
            nccl_us=sum(e['dur'] for wave in communication for e in wave if e['cat'] == 'kernel'),
            cpu_us=scope['dur'], gpu_span_us=end(tail)-gemm['ts'],
            communication_kernels=sum(sum(e['cat'] == 'kernel' for e in wave) for wave in communication)))
    require(claimed_markers == {id(e) for e in markers}, 'Unowned OP3 stage marker')
    actual = {id(e) for e in events if e.get('cat') == 'kernel'
              and any(token in e.get('name', '') for token in
                      (*ENCODERS.values(), *DECODERS.values(), TYPE_NAME))}
    require(actual <= claimed and len(actual) == sum(role_counts.values())*(1+2*chunks), 'Unowned/missing OP3 native work')
    return {role: dict(calls=role_counts[role], chunks_per_call=chunks, fused_consumer=fused,
        consumer_add_kernels=0 if fused else role_counts[role]*2, encode_kernels=role_counts[role]*chunks,
        decode_kernels=role_counts[role]*chunks, all_to_all_calls=role_counts[role]*chunks, independent_decode_stream=chunks > 1,
        original_peer_overlap_preserved=False, gemm_communication_overlap=False,
        calls_with_observed_decode_communication_overlap=sum(v['overlap'] > 0 for v in rows),
        median_decode_communication_overlap_us=median(v['overlap'] for v in rows),
        median_decode_to_consumer_us=median(v['decode_to_consumer_us'] for v in rows),
        median_stage_us={key: median(v[key] for v in rows) for key in
            ('gemm_us', 'encode_us', 'nccl_us', 'decode_us', 'cpu_us', 'gpu_span_us')},
        nccl_kernels=sum(v['communication_kernels'] for v in rows)) for role, rows in results.items()}



def row_backward(events, arm, rank):
    if arm == 'op3-improved':
        return previous.row_backward(events, rank)
    require(p.fused(arm), 'Unknown OP3 backward arm')
    # R owns the consumer: its bias gradient is inside the row Function.
    # Retain the observed PyTorch 2.8 split-K dgrad and ordered same-stream gather.
    index = base.ma8_profile.TraceIndex(events)
    forwards = [e for e in events if e.get('cat') == 'cpu_op' and e.get('name') == 'ConsumerGemmRSFunction']
    backwards = [e for e in events if e.get('cat') == 'cpu_op' and e.get('name') == 'ConsumerGemmRSFunctionBackward']
    scopes = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name') in ('scope.attn.flux_tensor', 'scope.mlp.flux_tensor')]
    require(len(forwards) == len(backwards) == len(scopes) == 256, 'OP3 row autograd coverage missing')
    roles, seen, claimed, result = {}, set(), set(), defaultdict(list)
    for call in forwards:
        owners = [e for e in scopes if contains(e, call)]
        require(len(owners) == 1, 'OP3 row forward scope missing')
        sequence = call['args']['Sequence number']
        require(sequence not in roles, 'Duplicate OP3 row sequence')
        roles[sequence] = owners[0]['name'].split('.')[1]
    for call in backwards:
        sequence = call['args']['Sequence number']
        require(sequence in roles and sequence not in seen, 'Unpaired OP3 row backward')
        seen.add(sequence)
        work, launches = index.work(call)
        require(not claimed.intersection(map(id, work)), 'Duplicate OP3 backward work')
        claimed.update(map(id, work))
        clears = index.clears(call, work, launches)
        kernels = [e for e in work if e.get('cat') == 'kernel']
        gathers = [e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        gemms = sorted([e for e in kernels if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
        splits = [e for e in kernels if base.ma8_profile._is_splitk(e)]
        biases = [e for e in kernels if 'reduce_kernel' in e['name']
                  and not base.ma8_profile._is_splitk(e)]
        require(len(gathers) == len(biases) == 1 and len(gemms) == 2 and len(kernels) == 4+len(splits),
                'OP3 consumer backward must retain one bias reduction, one gather and two GEMMs')
        gather, (dgrad, wgrad) = gathers[0], gemms
        compute = dgrad['args'].get('stream')
        require(compute is not None and gather['args'].get('stream') == wgrad['args'].get('stream') == compute
                and all(e['args'].get('device') == rank for e in work), 'OP3 row backward device/stream changed')
        bias = biases[0]
        bias_launch = launches[bias['args']['correlation']]
        bias_owners = [e for e in index.cpu[call['tid']] if e['name'] == 'aten::sum'
                       and contains(call, e) and contains(e, bias_launch)]
        require(len(bias_owners) == 1 and bias['args'].get('stream') == compute,
                'OP3 bias reduction lacks its original sum owner or changed streams')
        require(end(bias) <= gather['ts']+.01
                and end(bias_launch) <= launches[gather['args']['correlation']]['ts']+.01,
                'OP3 gather moved ahead of the bias reduction')
        maths = base.ma8_profile._row_splitk(index, call, launches, gemms, splits)
        require(end(gather) <= dgrad['ts']+.01 and maths['dgrad_end'] <= wgrad['ts']+.01,
                'OP3 row gather/dgrad/wgrad dependency changed')
        result[roles[sequence]].append(dict(cpu_us=call['dur'], bias_us=bias['dur'], allgather_us=gather['dur'],
            dgrad_us=maths['dgrad_us'], wgrad_us=maths['wgrad_us'],
            splitk_reduction_kernels=len(splits), cublas_4byte_memsets=clears.get('cublas_4byte', 0)))
    require(set(result) == {'attn', 'mlp'} and all(len(v) == 128 for v in result.values()),
            'OP3 row backward role coverage changed')
    return {role: dict(calls=128, bias_reduction_kernels=128, allgather_calls=128, dgrad_calls=128, wgrad_calls=128,
        splitk_reduction_kernels=sum(v['splitk_reduction_kernels'] for v in rows),
        allowed_cublas_4byte_memsets=sum(v['cublas_4byte_memsets'] for v in rows),
        median_us={key: median(v[key] for v in rows) for key in ('cpu_us', 'bias_us', 'allgather_us', 'dgrad_us', 'wgrad_us')})
        for role, rows in result.items()}



def analyze(events, arm, rank):
    require(arm in p.ALL_ARMS and type(rank) is int and 0 <= rank < 8, 'Wrong OP3 profile identity')
    require(not any(token in e.get('name', '') for e in events for token in
        ('CudaIpcBarrier', 'rtx5090_pack_tiles_kernel', 'rtx5090_unpack_tiles_kernel', 'op1_abc.')),
        'Foreign transport or stale OP1 profile markers')
    return dict(protocol=p.PROTOCOL, arm=arm, rank=rank, execution_validated=True,
        row_forward=chunk_records(events, arm, rank), row_backward=row_backward(events, arm, rank),
        activation_cache=base.ma8_profile.trace_activation_cache_records(events, 'F4-cache'),
        zero_count=base.ma9_profile.trace_zero_count_records(events, 'F4-zcount', rank),
        attention_softmax=base.ma11_profile.trace_attention_softmax_records(events, 'F4-cache-zcount-csoftmax'))
