"""Correlate actual CUDA launches with each projection and transport stage."""
from collections import Counter
from statistics import mean
from rtx5090_loader import load
import open_compress_protocol as p

base = load('profile')
GEMM_STAGE = 'ablation6.flux_gemm_only'


def analyze(events, arm, rank, expected_calls=128):
    p.require(arm in p.ARMS and type(rank) is int and 0 <= rank < 8
              and type(expected_calls) is int and expected_calls > 0, 'Unknown profile identity/count')
    index = base.Index(events)
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
              and e.get('name') in ('scope.attn.flux_tensor', 'scope.mlp.flux_tensor')]
    p.require(Counter(e['name'] for e in scopes) == Counter(
        {'scope.attn.flux_tensor': expected_calls, 'scope.mlp.flux_tensor': expected_calls}), 'Missing projection scopes')
    forbidden = ('taco_decode', 'fused_ash_', 'basic_quant_', 'rtx5090_pack_', 'rtx5090_unpack_',
                 'CudaIpcBarrier', 'nvshmem', 'ma11.attention', 'ma9.optimizer', '_MA8CachedRawLinear')
    p.require(not any(t.lower() in e.get('name', '').lower() for e in events for t in forbidden),
              'TACO or an unrequested mechanism executed')
    claimed, claimed_markers, duration = set(), set(), {'attn': [], 'mlp': []}
    quantized = arm != 'flux-bf16'
    relevant = {GEMM_STAGE, 'open_compress.encode', 'open_compress.exchange',
                'open_compress.decode_reduce', 'flux_baseline.bf16_reduce_scatter'}
    all_markers = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
                   and e.get('name') in relevant]
    waits = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
             and e.get('name') == 'cudaStreamWaitEvent']
    for scope in scopes:
        work, _ = index.work(scope)
        markers = [e for e in all_markers if base.contains(scope, e)]
        expected_names = [GEMM_STAGE] + (['open_compress.encode', 'open_compress.exchange',
            'open_compress.decode_reduce'] if quantized else ['flux_baseline.bf16_reduce_scatter'])
        p.require(Counter(e['name'] for e in markers) == Counter(expected_names), 'Wrong projection stage coverage')
        p.require(not claimed_markers.intersection(map(id, markers)), 'Duplicate stage ownership')
        claimed_markers.update(map(id, markers))

        def marker(name):
            return next(e for e in markers if e['name'] == name)

        def stage(name, match, memory=False):
            owned, _ = index.work(marker(name))
            kernels = [e for e in owned if e.get('cat') == 'kernel']
            p.require(kernels and all(match(e['name']) for e in kernels)
                      and all(e.get('cat') == 'kernel' or memory and
                              e.get('cat') in ('gpu_memcpy', 'gpu_memset') for e in owned),
                      'Wrong stage kernels/work '+name)
            return owned, kernels

        gemm_work, gemms = stage(GEMM_STAGE, base.is_basic_gemm, memory=True)
        p.require(len(gemms) == 1 and '_streamksk_' in gemms[0]['name'].lower(), 'Wrong Flux GEMM')
        gemm = gemms[0]
        # Stream-K's scratch clear is valid; copying a materialized output to
        # simulate sorting is not part of either producer definition.
        p.require(all(e['cat'] != 'gpu_memcpy' for e in gemm_work), 'Unexpected GEMM tensor copy')
        p.require(('Op1AInterleavedSwizzle' in gemm['name']) == (arm == p.ARMS[2]),
                  'Actual GEMM tile swizzle differs from selected arm')
        compute = gemm.get('args', {}).get('stream')
        if quantized:
            encode, _ = stage('open_compress.encode', lambda n: 'oc_int8_encode_kernel' in n)
            exchange, comm_kernels = stage('open_compress.exchange', lambda n: 'nccl' in n.lower(), memory=True)
            decode, _ = stage('open_compress.decode_reduce', lambda n: 'oc_int8_decode_reduce_kernel' in n)
            p.require(len(encode) == len(decode) == 1, 'Unexpected codec launches')
            p.require(compute is not None and all(e.get('args', {}).get('stream') == compute for e in encode+decode),
                      'Encoder/decoder must use the producer CUDA stream')
            chain = [gemm_work, encode, exchange, decode]
            exchange_marker = marker('open_compress.exchange')
        else:
            exchange, comm_kernels = stage('flux_baseline.bf16_reduce_scatter', lambda n: 'nccl' in n.lower(), memory=True)
            chain = [gemm_work, exchange]
            exchange_marker = marker('flux_baseline.bf16_reduce_scatter')
        for name, next_name in zip(expected_names, expected_names[1:]):
            p.require(base.end(marker(name)) <= marker(next_name)['ts']+.01, 'CPU stage submission order changed')
        calls = [e for e in index.cpu if base.contains(exchange_marker, e)
                 and e['name'].startswith('c10d::') and
                 ('alltoall' if quantized else 'reducescatter') in e['name'].lower().replace('_', '')]
        p.require(len(calls) == 1, 'Expected one actual c10d collective')
        collective = calls[0]
        collective_work, _ = index.work(collective)
        p.require({id(e) for e in comm_kernels} <= {id(e) for e in collective_work}, 'Unowned NCCL launch')
        if quantized:
            waiting = [e for e in index.cpu if base.contains(exchange_marker, e)
                       and e['name'] == 'record_param_comms'
                       and e.get('args', {}).get('Collective name') == 'wait']
            p.require(len(waiting) == 1 and any(base.contains(waiting[0], e) for e in waits),
                      'NCCL Work.wait must establish a CUDA stream dependency')
            p.require(base.end(collective) <= waiting[0]['ts']+.01
                      and base.end(waiting[0]) <= marker('open_compress.decode_reduce')['ts']+.01,
                      'Wait does not follow the exchange before decode')
        for before, after in zip(chain, chain[1:]):
            p.require(max(base.end(e) for e in before) <= min(e['ts'] for e in after)+.01,
                      'Consumer ran before producer/communication completed')
        expected = {id(e) for group in chain for e in group}
        p.require({id(e) for e in work} == expected and not claimed.intersection(expected),
                  'Extra/unowned work in projection or duplicate ownership')
        p.require(all(e.get('args', {}).get('device') == rank for e in work), 'Projection ran on wrong GPU')
        cpu_names = [e['name'].lower().replace('_', '') for e in index.cpu if base.contains(scope, e)]
        p.require(not any(('reducescatter' if quantized else 'alltoall') in n for n in cpu_names),
                  'Wrong collective API in projection')
        claimed.update(expected)
        role = 'attn' if '.attn.' in scope['name'] else 'mlp'
        duration[role].append((max(base.end(e) for g in chain for e in g)-gemm['ts'])/1000)
    p.require(claimed_markers == {id(e) for e in all_markers}, 'Unowned projection stage marker')
    owned_types = [e for e in events if e.get('cat') == 'kernel' and
                   (base.is_basic_gemm(e['name']) or 'oc_int8_' in e['name'])]
    p.require(all(id(e) in claimed for e in owned_types), 'Unowned GEMM/codec kernel outside projection')
    return dict(protocol=p.PROTOCOL, arm=arm, rank=rank, execution_validated=True,
        interleaved=arm == p.ARMS[2], arrival=False, selective=False,
        roles={r: dict(calls=len(v), mean_profile_span_ms=mean(v)) for r,v in duration.items()})
