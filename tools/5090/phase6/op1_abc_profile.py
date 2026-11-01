"""Launch-owned evidence for chunk delivery, decoding and model consumption.

Overlap is measured, never assumed or required to be positive. Completion
dependencies are required even when NCCL serializes the attempted pipeline.
"""
from collections import Counter, defaultdict
from statistics import median

from rtx5090_loader import load
import flux_baseline_profile
import op1_abc_protocol as p
from op1_a_swizzle import TYPE_NAME

base = load('profile')
require, contains, end = base.require, base.contains, base.end
ENCODER = 'op1_abc_encode_kernel'
DECODER = 'op1_abc_decode_kernel'


def one(events, message):
    require(len(events) == 1, message)
    return events[0]


def before(first, second, message):
    require(end(first) <= second['ts'] + .01, message)


def chunk_records(events, arm, rank):
    chunks = 1 if arm == 'flux-basic-interleaved' else 4
    index = base.Index(events)
    cpu_by_external = defaultdict(list)
    for event in index.cpu:
        external = event.get('args', {}).get('External id')
        if external is not None:
            cpu_by_external[external].append(event)
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
              and e.get('name') in ('scope.attn.flux_tensor', 'scope.mlp.flux_tensor')]
    require(Counter(e['name'] for e in scopes) == Counter(
        {'scope.attn.flux_tensor': 128, 'scope.mlp.flux_tensor': 128}), 'Incomplete ABC projection coverage')
    markers = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
               and (e.get('name', '').startswith('op1_abc.') or e.get('name') == 'ablation6.flux_gemm_only')]
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
        expected.update({f'op1_abc.{stage}.{i}': 1
                         for stage in ('encode', 'exchange', 'decode') for i in range(chunks)})
        if chunks > 1:
            expected['op1_abc.join'] = 1
        require(Counter(e['name'] for e in stages) == expected, 'Wrong ABC chunk/join stage coverage')
        require(not claimed_markers.intersection(map(id, stages)), 'Duplicate ABC marker ownership')
        claimed_markers.update(map(id, stages))

        def stage(name):
            return one([e for e in stages if e['name'] == name], 'Missing ABC stage '+name)

        gemm_stage = stage('ablation6.flux_gemm_only')
        gemm_work, _ = index.work(gemm_stage)
        gemm = one([e for e in gemm_work if e['cat'] == 'kernel'], 'Expected one ABC GEMM')
        require(base.is_basic_gemm(gemm['name']) and TYPE_NAME in gemm['name']
                and '_streamksk_' in gemm['name'].lower(), 'ABC actual GEMM lacks interleaved Stream-K')
        require(all(e['cat'] in ('kernel', 'gpu_memset') for e in gemm_work), 'Unexpected ABC GEMM work')
        work, decoders, communication = list(gemm_work), [], []
        for chunk in range(chunks):
            encode = stage(f'op1_abc.encode.{chunk}')
            exchange = stage(f'op1_abc.exchange.{chunk}')
            decode = stage(f'op1_abc.decode.{chunk}')
            before(gemm_stage, encode, 'ABC encode submitted before GEMM')
            before(encode, exchange, 'ABC exchange submitted before encode')
            before(exchange, decode, 'ABC decode submitted before exchange wait')
            encoded, _ = index.work(encode)
            decoded, _ = index.work(decode)
            encoder = one(encoded, 'Expected one direct ABC encoder launch')
            decoder = one(decoded, 'Expected one ABC decode/reduce launch')
            require(encoder['cat'] == decoder['cat'] == 'kernel' and ENCODER in encoder['name']
                    and DECODER in decoder['name'], 'Wrong ABC encoder/decoder')
            exchanged, _ = index.work(exchange)
            kernels = [e for e in exchanged if e['cat'] == 'kernel']
            require(kernels and all('nccl' in e['name'].lower() for e in kernels)
                    and all(e['cat'] in ('kernel', 'gpu_memcpy', 'gpu_memset') for e in exchanged),
                    'ABC exchange lacks actual NCCL work')
            calls = [e for e in index.cpu if contains(exchange, e) and e['name'].startswith('c10d::')
                     and 'alltoall' in e['name'].lower().replace('_', '')]
            collective = one(calls, 'Expected one c10d All-to-All per ABC chunk')
            owned, _ = index.work(collective)
            require({id(e) for e in kernels} <= {id(e) for e in owned}, 'Unowned ABC NCCL kernel')
            before(gemm, encoder, 'ABC encoder read unfinished GEMM')
            for e in exchanged:
                before(encoder, e, 'ABC NCCL read unfinished encoded chunk')
                before(e, decoder, 'ABC decoder read unfinished received chunk')
            if decoders:
                before(decoders[-1], decoder, 'ABC decoder stream is not ordered')
            decoders.append(decoder)
            communication.append(exchanged)
            work += encoded + exchanged + decoded
        compute = gemm['args'].get('stream')
        decode_streams = {e['args'].get('stream') for e in decoders}
        require(compute is not None and len(decode_streams) == 1
                and ((compute in decode_streams) == (chunks == 1)), 'ABC decoder stream contract changed')
        if chunks > 1:
            join = stage('op1_abc.join')
            require(not index.work(join)[0], 'ABC final join must only establish a stream dependency')
            before(stage(f'op1_abc.decode.{chunks-1}'), join, 'ABC join precedes final decode submission')
        require(all(e.get('args', {}).get('device') == rank for e in work), 'Wrong ABC CUDA device')
        outer, _ = index.work(scope)
        require({id(e) for e in work} == {id(e) for e in outer}, 'Unregistered ABC projection work')
        require(not claimed.intersection(map(id, work)), 'Duplicate ABC work ownership')
        claimed.update(map(id, work))
        cpu = [e for e in index.cpu if contains(scope, e)]
        require(not any(e['name'] in ('aten::cat', 'aten::clone', 'aten::add') for e in cpu),
                'ABC reintroduced remote concatenation, clone or separate reduction')

        # The returned tensor must be ready for the unchanged two BF16 adds and
        # LayerNorm on the caller stream, including asynchronous B/C decoding.
        tail = max((e for e in work if e['args'].get('stream') == compute), key=end)
        _, pos = positions[id(tail)]
        successors = streams[compute][pos+1:pos+4]
        require(len(successors) == 3 and all(base._bf16_add(e) for e in successors[:2])
                and base._layernorm_backend(successors[2]) is not None,
                'ABC output does not feed the unchanged bias/residual/LayerNorm')
        for e in decoders:
            before(e, successors[0], 'ABC consumer raced an unfinished decoder')
        for first, second in zip(successors, successors[1:]):
            before(first, second, 'ABC consumer order changed')
        for e in successors:
            external = e.get('args', {}).get('External id')
            owners = [owner for owner in cpu_by_external[external] if external is not None
                      and any(id(w) == id(e) for w in index.work(owner)[0])]
            expected_name = ('aten::add' if e in successors[:2] else
                '_OperationFuserAutogradFunction' if base._layernorm_backend(e) == 'transformer_engine'
                else 'FusedLayerNormAffineFunction')
            require(any(owner['name'] == expected_name for owner in owners), 'ABC consumer lacks launch owner')
        overlaps = sum(max(0., min(end(decoder), end(comm))-max(decoder['ts'], comm['ts']))
                       for i, decoder in enumerate(decoders) for later in communication[i+1:]
                       for comm in later if comm['cat'] == 'kernel')
        results[role].append(dict(overlap=overlaps,
            decode_to_consumer_us=end(successors[-1])-min(e['ts'] for e in decoders),
            communication_kernels=sum(sum(e['cat'] == 'kernel' for e in wave) for wave in communication)))
    require(claimed_markers == {id(e) for e in markers}, 'Unowned ABC stage marker')
    actual = {id(e) for e in events if e.get('cat') == 'kernel'
              and any(token in e.get('name', '') for token in (ENCODER, DECODER, TYPE_NAME))}
    require(actual <= claimed and len(actual) == 256*(1+2*chunks), 'Unowned/missing ABC native work')
    return {role: dict(calls=128, chunks_per_call=chunks, encode_kernels=128*chunks,
        decode_kernels=128*chunks, all_to_all_calls=128*chunks, independent_decode_stream=chunks > 1,
        original_peer_overlap_preserved=False, gemm_communication_overlap=False,
        calls_with_observed_decode_communication_overlap=sum(v['overlap'] > 0 for v in rows),
        median_decode_communication_overlap_us=median(v['overlap'] for v in rows),
        median_decode_to_consumer_us=median(v['decode_to_consumer_us'] for v in rows),
        nccl_kernels=sum(v['communication_kernels'] for v in rows)) for role, rows in results.items()}


def analyze(events, arm, rank):
    require(arm in p.ARMS and type(rank) is int and 0 <= rank < 8, 'Wrong ABC profile identity')
    if arm == 'flux-improved':
        result = flux_baseline_profile.analyze(events, arm, rank)
        return dict(result, protocol=p.PROTOCOL, comparator_unchanged=True)
    require(not any(token in e.get('name', '') for e in events for token in
        ('_MA8CachedRawLinear', 'ma11.attention.softmax', 'ma9.optimizer.zero_count',
         'CudaIpcBarrier', 'rtx5090_pack_tiles_kernel', 'rtx5090_unpack_tiles_kernel')),
        'Unrequested mechanism in ABC profile')
    return dict(protocol=p.PROTOCOL, arm=arm, rank=rank, execution_validated=True,
                row_forward=chunk_records(events, arm, rank))
