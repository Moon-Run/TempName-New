"""Real launch-correlated per-rank evidence for the portable TP8/DP2 run."""
from collections import Counter
from ma9_profile import Index, contains, require
import ma11_profile


def dp_gradient_records(events):
    index = Index(events)
    # A DP=1 run cannot pass this gate. Require actual GPU work belonging to
    # the two-member DP collective, in addition to replica gradient hashes.
    dp = [e for e in index.cpu if e['name'] == 'record_param_comms' and
          e.get('args',{}).get('Group size') == 2 and
          e.get('args',{}).get('Collective name') in ('allreduce', 'allreduce_coalesced') and
          e.get('args',{}).get('In msg nelems',0) > 8192]
    require(dp, 'No DP2 gradient AllReduce metadata in profile')
    for scope in dp:
        args=scope['args']
        require(args.get('Process Group Description') in ('DATA_PARALLEL_GROUP','DATA_PARALLEL_GROUP_WITH_CP') and
                args.get('In msg nelems')==args.get('Out msg nelems') and args.get('dtype')=='Float',
                'DP gradient collective group, size or dtype changed')
    require(all(any(e['cat']=='kernel' and 'nccl' in e['name'].lower() and 'AllReduce' in e['name']
                    for e in index.work(scope)[0]) for scope in dp), 'DP collective lacks correlated NCCL kernels')
    return dict(dp_allreduces=len(dp),
                dp_collective_names=dict(Counter(e['args']['Collective name'] for e in dp)),
                dp_gradient_elements=sum(e['args']['In msg nelems'] for e in dp))


def analyze(events, arm, rank):
    result = dict(dp_gradient_records(events),rank=rank)
    if arm == 'ma11':
        result.update(ma11_profile.trace_composed_records(events,'F4-cache-zcount-csoftmax',rank%8))
        return result
    result['roles']=projection_records(events,arm)
    return result


def projection_records(events, arm, expected_calls=64):
    """Production uses 32 layers x 2 microbatches; explicit count also permits
    offline regression against the original four-microbatch single-node traces.
    This helper alone never certifies DP communication or an entire profile.
    """
    require(arm in ('flux-basic','megatron-cuda'), 'Unknown profile arm')
    index=Index(events)
    route = 'flux_tensor' if arm == 'flux-basic' else 'tensor'
    receiver = 'basic_quant_decode_reduce_kernel' if arm == 'flux-basic' else 'fused_ash_decompress_kernel'
    result = {}
    claimed = set()
    for role in ('attn','mlp'):
        scopes = [e for e in events if e.get('cat')=='user_annotation' and e.get('ph')=='X' and
                  e['name']==f'scope.{role}.{route}']
        require(len(scopes)==expected_calls, 'Unexpected local microbatch projection coverage for '+role)
        for scope in scopes:
            work,_ = index.work(scope)
            kernels = [e for e in work if e['cat']=='kernel']
            require(not claimed.intersection(map(id,kernels)), 'Projection kernels attributed twice')
            claimed.update(map(id,kernels))
            names = [e['name'] for e in kernels]
            cpu = [e for e in index.cpu if contains(scope,e)]
            counts = Counter(e['name'] for e in cpu)
            require(sum('fused_ash_compress_kernel' in n for n in names)==1 and
                    sum(receiver in n for n in names)==1 and
                    any('nccl' in n.lower() for n in names), 'Missing standalone encode/exchange/decode kernel')
            require(any('alltoall' in e['name'].lower().replace('_','') for e in cpu), 'NCCL All-to-All missing')
            require(not counts['aten::clone'] and not any('CudaIpcBarrierAllKernel' in n or
                    'taco_decode_ring_kernel' in n or 'reducescatter' in n.lower().replace('_','') for n in names),
                    'Baseline changed to GEMM-RS or clone path')
            require(counts['aten::add'] == (0 if arm=='flux-basic' else 8), 'BF16 receiver fold changed')
            markers = [e['name'] for e in events if e.get('cat')=='user_annotation' and contains(scope,e)]
            for marker in ('megatron_taco.encode','megatron_taco.exchange','megatron_taco.decode_reduce'):
                require(markers.count(marker)==1, 'Missing codec stage: '+marker)
            if arm=='flux-basic':
                require(markers.count('ablation6.flux_gemm_only')==1 and
                        sum(('cutlass' in n.lower() or 'flux_bf16' in n.lower()) and 'sm90' in n.lower() for n in names)==1 and
                        not any('fused_ash_decompress_kernel' in n for n in names), 'Flux GemmOnly/basic receiver changed')
            else:
                # The whole original RowParallelLinear forward is inside this
                # scope. Its source identity is checked by the worker as well.
                require('ablation6.flux_gemm_only' not in markers and
                        not any('basic_quant_decode_reduce_kernel' in n for n in names), 'Native CUDA baseline changed')
        result[role] = dict(calls=expected_calls,route=route,receiver=receiver,encode_kernels=expected_calls,
                           decode_kernels=expected_calls,separate_bf16_adds=0 if arm=='flux-basic' else 8*expected_calls)
    return result
