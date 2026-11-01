"""GPU launch-correlated audit of old A/C, standalone quantization and no quantization."""
from collections import Counter
from rtx5090_loader import load
import flux_baseline_protocol as p
from flux_baseline_native import MARKER

base = load('profile')
require = p.require


def bf16_records(events):
    index = base.Index(events)
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
              and e.get('name') in ('scope.attn.flux_tensor', 'scope.mlp.flux_tensor')]
    require(Counter(e['name'] for e in scopes) == Counter(
        {'scope.attn.flux_tensor': 128, 'scope.mlp.flux_tensor': 128}), 'Missing BF16 projection scopes')
    forbidden = ('taco_decode', 'fused_ash_', 'basic_quant_', 'rtx5090_pack_', 'rtx5090_unpack_',
                 'CudaIpcBarrier', 'Op1AInterleavedSwizzle')
    require(not any(any(token in e.get('name', '') for token in forbidden)
                    for e in events if e.get('cat') == 'kernel'), 'Quantized/interleaved work in BF16 arm')
    claimed = set()
    for scope in scopes:
        work, _ = index.work(scope)
        kernels = [e for e in work if e.get('cat') == 'kernel']
        gemms = [e for e in kernels if base.is_basic_gemm(e['name'])]
        communication = [e for e in kernels if 'nccl' in e['name'].lower()]
        require(len(gemms) == 1 and '_streamksk_' in gemms[0]['name'].lower()
                and communication and all(e in gemms or e in communication for e in kernels),
                'Expected original Flux GemmOnly and NCCL kernels only')
        require(not claimed.intersection(map(id, kernels)), 'Duplicate BF16 kernel ownership')
        claimed.update(map(id, kernels))
        cpu = [e for e in index.cpu if base.contains(scope, e)]
        normalized = [e['name'].lower().replace('_', '') for e in cpu]
        require(any('reducescatter' in name for name in normalized)
                and not any('alltoall' in name for name in normalized)
                and not any(e['name'] in ('aten::clone', 'aten::add') for e in cpu),
                'BF16 collective was replaced by codec, clone or tensor fold')
        markers = Counter(e['name'] for e in events if e.get('cat') == 'user_annotation'
                          and base.contains(scope, e))
        require(markers['ablation6.flux_gemm_only'] == 1
                and markers['flux_baseline.bf16_reduce_scatter'] == 1
                and not any(name.startswith('megatron_taco.') for name in markers), 'Wrong BF16 stage markers')
        # Index.work establishes owning launches; timestamps enforce completion before consumption.
        require(all(base.end(gemms[0]) <= e['ts']+.01 for e in communication),
                'BF16 Reduce-Scatter consumed unfinished GEMM output')
    return {role: dict(calls=128, route='flux_tensor', gemm_algorithm='original Sm89 StreamkSK / SM120',
                      reduce_scatter_calls=128, encode_kernels=0, decode_kernels=0,
                      all_to_all_calls=0, quantization='none') for role in ('attn', 'mlp')}


def analyze(events, arm, rank):
    require(arm in p.ARMS and 0 <= rank < 8, 'Unknown four-arm profile')
    require(not any(token in e.get('name', '') for e in events for token in
        ('_MA8CachedRawLinear', 'ma11.attention.softmax', 'ma9.optimizer.zero_count', 'Op1AInterleavedSwizzle')),
        'Unrequested mechanism in four-arm profile')
    if arm in p.NATIVE:
        result = base.staged_projection_records(events, p.NATIVE[arm], rank)
        gemms = [e for e in events if e.get('cat') == 'kernel' and base.is_local_gemm(e.get('name', ''))]
        require(len(gemms) == 256 and all(MARKER in e['name'] for e in gemms),
                'Actual native GEMM does not use the registered original-order variant')
    elif arm == 'flux-basic':
        result = base.basic_projection_records(events)
        gemms = [e for e in events if e.get('cat') == 'kernel' and base.is_basic_gemm(e.get('name', ''))]
        require(len(gemms) == 256 and all('_streamksk_' in e['name'].lower() for e in gemms),
                'Basic Flux GEMM changed')
    else:
        result = bf16_records(events)
    return dict(protocol=p.PROTOCOL, arm=arm, rank=rank, execution_validated=True,
                interleaved=False, arrival=False, selective=False, row_forward=result)
