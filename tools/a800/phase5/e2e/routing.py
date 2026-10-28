"""Explicit nine-way MLP-only phase5 routes."""
POLICIES = ['native', 'original', 'taco_separate', 'remote_bf16', 'interleaved_bf16',
            'remote_all', 'interleaved_all', 'remote_selective', 'interleaved_selective']
ARRIVAL_POLICIES = ['remote_arrival_selective','interleaved_arrival_selective']
NATIVE_QUANT_POLICY = 'native_taco'
EXTRA_POLICIES = ['taco_fused', 'interleaved_remote_arrival_selective', 'interleaved_remote_group_arrival_selective']


def codec_placement(policy):
    assert policy in POLICIES + ARRIVAL_POLICIES + [NATIVE_QUANT_POLICY] + EXTRA_POLICIES
    if policy == NATIVE_QUANT_POLICY: return 'tensor'
    if policy == 'taco_separate': return 'separate'
    if policy == 'taco_fused': return 'fused'
    return 'fused' if policy.endswith(('_all', '_selective')) else 'bf16'


def base_policy(policy):
    if policy.startswith('interleaved_remote_group_'): return 'interleaved_remote_group'
    if policy.startswith('interleaved_remote_'): return 'interleaved_remote'
    return 'remote_first' if policy.startswith('remote_') else 'interleaved' if policy.startswith('interleaved_') else 'original'


def mapping_policy(policy):
    if policy.endswith('_arrival_selective'):
        return policy.removesuffix('_selective')
    return base_policy(policy)


def module_routes(observed, policy, layers=12, model=None):
    assert policy in POLICIES + ARRIVAL_POLICIES + [NATIVE_QUANT_POLICY] + EXTRA_POLICIES
    counts = {'attention': 0, 'mlp': 0}
    model=model or dict(hidden=2048,ffn=8192,sequence=2048,micro_batch=1,tp=4)
    m=model['sequence']*model['micro_batch'];n=model['hidden'];tp=model['tp']
    routes = {}
    for name, shape in observed.items():
        if name.endswith('.self_attention.linear_proj'):
            kind, k = 'attention', model['hidden']
        elif name.endswith('.mlp.linear_fc2'):
            kind, k = 'mlp', model['ffn']
        else:
            raise AssertionError(name)
        assert (shape['M'],shape['N'],shape['K_global'],shape['K_local']) == (m,n,k,k//tp)
        counts[kind] += 1
        routes[name] = 'native' if policy=='native' or policy==NATIVE_QUANT_POLICY and kind=='attention' else policy if kind=='mlp' else 'original'
    assert counts == {'attention':layers, 'mlp':layers}
    return routes


def selection_entry(plan,policy,m,n,k):
    entries=plan['policies'][base_policy(policy)]
    if plan.get('schema')==2:
        return next(s for s in entries if s['shape']==[m,n,k])
    assert plan['shape']==[m,n,k]
    return entries


def logical_bytes(mask,rank,m,n,world):
    bf16=(world-1)*(m//world)*n*2
    nt=(n+127)//128
    if mask is None:return (world-1)*(m//world)*nt*136,bf16
    return bf16+sum(128*(136-2*min(128,n-(t%nt)*128)) for t,v in enumerate(mask[rank]) if v),bf16
