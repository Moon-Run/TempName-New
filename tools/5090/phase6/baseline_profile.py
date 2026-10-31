"""Launch-correlated mechanism evidence for the nine RTX 5090 arms."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from types import ModuleType

from rtx5090_loader import load

p = load('protocol')
import ma2_ablation_fullstep
import ma4_f_fullstep
import ma4_f_bench
import ma8_profile
import ma9_profile
import ma11_profile as h100_ma11_profile
from ma9_profile import Index, contains, end, require


def _pytorch28_ma11_profile():
    # 2048-element BF16 forward uses PyTorch 2.8's persistent warp kernel.
    # Preserve the entire H100 ownership/order audit, adapting only its exact
    # control-kernel predicate in this private module; do not alter H100 code.
    source = Path(h100_ma11_profile.__file__).read_text()
    old = "'cunn_SoftMaxForwardSmem' in kernel['name']"
    new = "'softmax_warp_forward' in kernel['name']"
    require(source.count(old) == 1, 'MA11 profile source contract changed')
    module = ModuleType('rtx5090_pytorch28_ma11_profile')
    module.__file__ = h100_ma11_profile.__file__
    exec(compile(source.replace(old, new), module.__file__, 'exec'), module.__dict__)
    return module


ma11_profile = _pytorch28_ma11_profile()

PRODUCE = 'rtx5090.transport.produce_pack'
EXCHANGE = 'rtx5090.transport.exchange'
DECODE = 'rtx5090.transport.unpack_decode'
CONSUMER = 'rtx5090.transport.unpack_decode_consumer'
CONFIGURE = 'rtx5090.transport.configure'
PACK_KERNEL = 'rtx5090_pack_tiles_kernel'
UNPACK_KERNEL = 'rtx5090_unpack_tiles_kernel'


def is_local_gemm(name):
    name = name.lower()
    return ('cutlass' in name and
            'flux_bf16_bf16_void_bf16_fp32_fp32_sm80_a100_reducescatter_rcr_gemmv2_false_false_intranode' in name)


def _one(values, message):
    require(len(values) == 1, message)
    return values[0]


def _before(first, second, message):
    require(end(first) <= second['ts'] + .01, message)


def _bf16_add(event):
    return ma4_f_bench.is_add(event) and 'bfloat16' in event['name'].lower()


def _clone_work(event):
    name = event.get('name', '').lower()
    return (event.get('cat') == 'gpu_memcpy' and ('dtod' in name or 'device -> device' in name)
            and event.get('args', {}).get('bytes') == 256*4096*2
            or event.get('cat') == 'kernel' and 'copy' in name and 'bfloat16' in name)


def staged_projection_records(events, arm, rank=None):
    """Prove local producer/pack -> NCCL -> unpack/decoder -> model consumer.

    GPU timestamps alone do not establish ownership: each stage must contain
    the actual runtime launch correlated to its kernels. NCCL-associated
    memcpy/memset is allowed; the original no-copy IPC audit does not apply.
    """
    require(arm in p.ARMS[:-1], 'Unknown staged projection arm')
    fused = arm in p.CR3_FAMILY
    decode_marker = CONSUMER if fused else DECODE
    index = Index(events)
    scopes = [event for event in events if event.get('cat') == 'user_annotation'
              and event.get('ph') == 'X' and event.get('name') in
              ('scope.attn.candidate', 'scope.mlp.candidate')]
    require(Counter(event['name'] for event in scopes) ==
            Counter({'scope.attn.candidate': 128, 'scope.mlp.candidate': 128}),
            'Staged profile must cover both 32-layer/four-microbatch projection roles')
    markers = [event for event in events if event.get('cat') == 'user_annotation'
               and event.get('ph') == 'X' and event.get('name', '').startswith('rtx5090.transport.')]
    expected = Counter({PRODUCE: 256, EXCHANGE: 256, decode_marker: 256})
    if arm == 'a':
        expected[CONFIGURE] = 256
    require(Counter(event['name'] for event in markers) == expected,
            'Missing, extra or wrong staged transport markers')
    device_work = [event for event in events if event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    require(not any('cudaipcbarrier' in event.get('name', '').lower()
                    or 'nvshmem' in event.get('name', '').lower() for event in device_work),
            'Legacy CUDA IPC/NVSHMEM device work is forbidden in staged transport')
    cpu_by_tid, cpu_by_external, stream_events = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in index.cpu:
        cpu_by_tid[event['tid']].append(event)
        external = event.get('args', {}).get('External id')
        if external is not None:
            cpu_by_external[external].append(event)
    for rows in cpu_by_tid.values():
        rows.sort(key=lambda event: event['ts'])
    cpu_starts = {tid: [event['ts'] for event in rows] for tid, rows in cpu_by_tid.items()}
    for event in device_work:
        stream_events[event.get('args', {}).get('stream')].append(event)
    positions = {}
    for stream, rows in stream_events.items():
        rows.sort(key=lambda event: event['ts'])
        positions.update({id(event): (stream, position) for position, event in enumerate(rows)})

    def cpu_inside(scope):
        rows, starts = cpu_by_tid[scope['tid']], cpu_starts.get(scope['tid'], [])
        lo, hi = bisect_left(starts, scope['ts']-.01), bisect_right(starts, end(scope)+.01)
        return [event for event in rows[lo:hi] if contains(scope, event)]

    def actual_cpu_owner(kernel, name):
        candidates = cpu_by_external[kernel.get('args', {}).get('External id')]
        owners = [event for event in candidates if event['name'] == name and
                  any(id(work) == id(kernel) for work in index.work(event)[0])]
        return _one(owners, 'Consumer kernel lacks its actual CPU/launch association: '+name)

    claimed, claimed_markers, compute_streams = set(), set(), set()
    records = defaultdict(list)
    for scope in sorted(scopes, key=lambda event: (event['tid'], event['ts'])):
        stages = [event for event in markers if contains(scope, event)]
        require(not claimed_markers.intersection(map(id, stages)), 'Transport marker attributed to two projections')
        claimed_markers.update(map(id, stages))
        produce = _one([event for event in stages if event['name'] == PRODUCE], 'Missing local producer/pack stage')
        exchange = _one([event for event in stages if event['name'] == EXCHANGE], 'Missing NCCL exchange stage')
        decode = _one([event for event in stages if event['name'] == decode_marker], 'Missing local unpack/decode stage')
        _before(produce, exchange, 'CPU exchange submitted before producer/pack')
        _before(exchange, decode, 'CPU decoder submitted before exchange wait')
        call_name = 'ConsumerGemmRSFunction' if fused else 'GemmRSFunction'
        cpu = cpu_inside(scope)
        call = _one([event for event in cpu if event['name'] == call_name], 'Missing projection autograd call')
        require(all(contains(call, stage) for stage in (produce, exchange, decode)),
                'Staged transport is outside its projection autograd owner')
        configs = [event for event in stages if event['name'] == CONFIGURE]
        require(len(configs) == int(arm == 'a'), 'Wrong per-call native reconfiguration')
        if configs:
            require(contains(call, configs[0]) and not index.work(configs[0])[0],
                    'A reconfiguration must not allocate or initialize CUDA workspace per call')
            _before(configs[0], produce, 'A reconfigured after producer submission')

        produced, _ = index.work(produce)
        producer_kernels = [event for event in produced if event['cat'] == 'kernel']
        gemm = _one([event for event in producer_kernels if is_local_gemm(event['name'])],
                    'Missing registered local V2 GEMM')
        pack = _one([event for event in producer_kernels if PACK_KERNEL in event['name']],
                    'Missing local packet pack kernel')
        require(len(producer_kernels) == 2, 'Unexpected producer kernel or unfused encoding fallback')
        require(all(event['cat'] in ('kernel', 'gpu_memset') for event in produced),
                'Local producer contains an unexpected GPU copy')
        _before(gemm, pack, 'Packet packing started before local GEMM finished')

        exchanged, _ = index.work(exchange)
        comm_kernels = [event for event in exchanged if event['cat'] == 'kernel']
        require(comm_kernels and all('nccl' in event['name'].lower() for event in comm_kernels),
                'Exchange must launch actual NCCL kernels')
        require(all(event['cat'] in ('kernel', 'gpu_memcpy', 'gpu_memset') for event in exchanged),
                'Unknown exchange device work')
        collective_cpu = [event for event in cpu_inside(exchange)
                          if event['name'].startswith('c10d::')
                          and 'alltoall' in event['name'].lower().replace('_', '')]
        collective = _one(collective_cpu, 'Expected one actual c10d All-to-All submission')
        collective_work, _ = index.work(collective)
        require({id(event) for event in comm_kernels} <= {id(event) for event in collective_work},
                'NCCL kernels lack their All-to-All CPU owner')

        decoded, _ = index.work(decode)
        require(len(decoded) == 2 and all(event['cat'] == 'kernel' for event in decoded),
                'Expected one unpack kernel and one original decoder')
        unpack = _one([event for event in decoded if UNPACK_KERNEL in event['name']],
                      'Missing local packet unpack kernel')
        decoder = _one([event for event in decoded if
                        (ma4_f_bench.decoder_matches(event['name'], 'F4') if fused else
                         ma2_ablation_fullstep.decoder_matches(event['name'], p.NATIVE_ARM[arm]))],
                       'Wrong original decoder for the staged arm')
        _before(unpack, decoder, 'Decoder consumed packets before unpack completed')
        compute = (gemm, pack, unpack, decoder)
        stream = gemm.get('args', {}).get('stream')
        require(stream is not None and all(event.get('args', {}).get('stream') == stream for event in compute),
                'Staged compute moved away from its creation stream')
        compute_streams.add(stream)
        for event in produced:
            if event['cat'] == 'gpu_memset':
                require(event.get('args', {}).get('stream') == stream,
                        'Producer workspace clear used a different stream')
                _before(event, gemm, 'Producer workspace clear ran after GEMM started')
        require(all(event.get('args', {}).get('stream') not in (None, stream) for event in comm_kernels),
                'Expected a distinct NCCL communication stream')
        for event in exchanged:
            _before(pack, event, 'NCCL exchange consumed unfinished packed data')
            _before(event, unpack, 'Unpack started before NCCL exchange completed')
        if rank is not None:
            require(all(event.get('args', {}).get('device') == rank
                        for event in (*producer_kernels, *comm_kernels, *decoded)),
                    'Projection device work belongs to a different rank/GPU')

        clone_cpu = [event for event in cpu if event['name'] == 'aten::clone']
        require(len(clone_cpu) == int(arm == 'a'), 'Wrong projection output clone count')
        clone = []
        if clone_cpu:
            require(contains(call, clone_cpu[0]), 'Output clone lacks projection ownership')
            _before(decode, clone_cpu[0], 'A output clone submitted before decoder')
            clone = index.work(clone_cpu[0])[0]
            require(len(clone) == 1 and _clone_work(clone[0]), 'A must retain one BF16 output clone')
        projection_work = produced + exchanged + decoded + clone
        require(not claimed.intersection(map(id, projection_work)), 'Device work attributed to two projections')
        claimed.update(map(id, projection_work))
        outer_work, _ = index.work(scope)
        require({id(event) for event in outer_work} == {id(event) for event in projection_work},
                'Projection contains unregistered CUDA work outside transport stages')

        # Find the real model consumer on the same compute stream, allowing
        # A's preserved output clone before its two BF16 bias/residual adds.
        _, position = positions[id(decoder)]
        sequence = stream_events[stream]
        add_count = 0 if fused else 2
        successors = sequence[position+1:position+2+len(clone)+add_count]
        require(len(successors) == 1+len(clone)+add_count, 'Consumer LayerNorm missing')
        require([id(event) for event in successors[:len(clone)]] == [id(event) for event in clone],
                'A clone does not immediately follow its decoder')
        adds = successors[len(clone):len(clone)+add_count]
        require(all(_bf16_add(event) for event in adds), 'Original BF16 bias/residual additions missing')
        for event in adds:
            actual_cpu_owner(event, 'aten::add')
        norm = successors[-1]
        require(norm['cat'] == 'kernel' and any(token in norm['name'].lower() for token in ('layernorm', 'layer_norm')),
                'Decoder/consumer does not immediately feed LayerNorm')
        actual_cpu_owner(norm, 'FusedLayerNormAffineFunction')
        for first, second in zip([decoder]+successors, successors):
            _before(first, second, 'Projection consumer stream order changed')
        records[scope['name'].split('.')[1]].append(dict(cpu_us=call['dur'], decoder_us=decoder['dur'],
            decoder_start_to_layernorm_end_us=end(norm)-decoder['ts'],
            producer_end_to_layernorm_end_us=end(norm)-end(gemm),
            exchange_kernels=len(comm_kernels),
            exchange_memcpy=sum(event['cat'] == 'gpu_memcpy' for event in exchanged),
            producer_workspace_memsets=sum(event['cat'] == 'gpu_memset' for event in produced)))
    require(len(claimed_markers) == len(markers) and len(compute_streams) == 1,
            'Unowned transport marker or inconsistent compute stream')
    actual_packets = {id(event) for event in device_work if event['cat'] == 'kernel'
                      and (PACK_KERNEL in event['name'] or UNPACK_KERNEL in event['name'])}
    require(actual_packets <= claimed and len(actual_packets) == 512,
            'Unowned or missing packet pack/unpack kernels')
    return {role: dict(projection_calls=128, transport=p.TRANSPORT, local_gemm_kernels=128,
                       packet_pack_kernels=128, packet_unpack_kernels=128, decoder_kernels=128,
                       nccl_alltoall_calls=128, nccl_kernels=sum(row['exchange_kernels'] for row in rows),
                       nccl_memcpy_events=sum(row['exchange_memcpy'] for row in rows),
                       producer_workspace_memsets=sum(row['producer_workspace_memsets'] for row in rows),
                       native_ipc_barriers=0, original_peer_overlap_preserved=False,
                       output_clones=128*int(arm == 'a'), per_call_reconfigurations=128*int(arm == 'a'),
                       consumer_add_kernels=128*(0 if fused else 2), layernorm_successors=128,
                       fused_consumer=fused, compute_stream=next(iter(compute_streams)),
                       **{key: median(row[key] for row in rows) for key in
                          ('cpu_us', 'decoder_us', 'decoder_start_to_layernorm_end_us',
                           'producer_end_to_layernorm_end_us')}) for role, rows in records.items()}


def is_basic_gemm(name):
    """The registered Sm89 GemmV2 implementation is compiled for physical SM120.

    The compile-checked SM120 object retains this exact Flux registry key in
    its cutlass::Kernel2 entry. Arbitrary CUTLASS/CUBLAS GEMMs do not pass.
    """
    name = name.lower()
    registered = 'flux_bf16_bf16_void_bf16_fp32_fp32_sm89_rtx5090_commnone_rcr_gemmv2' in name
    return ('cutlass' in name and registered
            and not any(token in name for token in ('sm90', 'wgmma', 'gmma', 'reducescatter')))


def basic_projection_records(events):
    index = Index(events)
    result, claimed = {}, set()
    for role in ('attn', 'mlp'):
        scopes = [event for event in events if event.get('cat') == 'user_annotation'
                  and event.get('ph') == 'X' and event['name'] == f'scope.{role}.flux_tensor']
        require(len(scopes) == 128, 'Incomplete basic full-model projection coverage: '+role)
        for scope in scopes:
            work, _ = index.work(scope)
            kernels = [event for event in work if event['cat'] == 'kernel']
            require(not claimed.intersection(map(id, kernels)), 'Basic kernels attributed twice')
            claimed.update(map(id, kernels))
            names = [event['name'] for event in kernels]
            cpu = [event for event in index.cpu if contains(scope, event)]
            counts = Counter(event['name'] for event in cpu)
            require(sum('fused_ash_compress_kernel' in name for name in names) == 1
                    and sum('basic_quant_decode_reduce_kernel' in name for name in names) == 1
                    and any('nccl' in name.lower() for name in names),
                    'Missing basic encode/exchange/receiver kernel')
            require(any('alltoall' in event['name'].lower().replace('_', '') for event in cpu),
                    'Basic NCCL All-to-All missing')
            require(not counts['aten::clone'] and not counts['aten::add']
                    and not any('CudaIpcBarrierAllKernel' in name or 'taco_decode_ring_kernel' in name
                                or 'reducescatter' in name.lower().replace('_', '')
                                or 'fused_ash_decompress_kernel' in name for name in names),
                    'Basic receiver changed to GEMM-RS, clone or standalone BF16 fold')
            markers = [event['name'] for event in events if event.get('cat') == 'user_annotation'
                       and contains(scope, event)]
            for marker in ('megatron_taco.encode', 'megatron_taco.exchange',
                           'megatron_taco.decode_reduce', 'ablation6.flux_gemm_only'):
                require(markers.count(marker) == 1, 'Missing basic stage: '+marker)
            require(sum(is_basic_gemm(name) for name in names) == 1,
                    'Expected the registered Sm89/RTX5090 BF16 RCR CUTLASS GemmV2 kernel')
        result[role] = dict(calls=128, route='flux_tensor', receiver='basic_quant_decode_reduce_kernel',
                            encode_kernels=128, decode_kernels=128, separate_bf16_adds=0,
                            gemm_algorithm='Sm89 GemmV2 compiled for SM120')
    return result


def analyze(events, arm, rank):
    require(arm in p.ARMS and 0 <= rank < 8, 'Unknown RTX 5090 profile identity')
    if arm in p.CR3_FAMILY:
        result = dict(row_forward=staged_projection_records(events, arm, rank))
        if arm in ('ma11', 'ma89'):
            result.update(ma11_profile.trace_composed_records(events, p.NATIVE_ARM[arm], rank))
        else:
            cache_arm = 'F4-cache' if arm == 'ma8' else 'F4'
            zero_arm = 'F4-zcount' if arm == 'ma9' else 'F4'
            result.update(activation_cache=ma8_profile.trace_activation_cache_records(events, cache_arm),
                          row_backward=ma8_profile.trace_row_backward_records(events, cache_arm),
                          zero_count=ma9_profile.trace_zero_count_records(events, zero_arm, rank),
                          attention_softmax=ma11_profile.trace_attention_softmax_records(events, 'F4-cache-zcount'))
    elif arm in ('r', 'c', 'a'):
        result = dict(row_forward=staged_projection_records(events, arm, rank))
        require(not any(token in event.get('name', '') for event in events
                        for token in ('_MA8CachedRawLinear', 'ma11.attention.softmax', 'ma9.optimizer.zero_count')),
                'R/C/A unexpectedly installed a later optimization')
    else:
        result = dict(row_forward=basic_projection_records(events))
    return dict(arm=arm, rank=rank, execution_validated=True, **result)
