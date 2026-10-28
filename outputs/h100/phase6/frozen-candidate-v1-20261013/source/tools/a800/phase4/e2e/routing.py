"""Check exact MLP-only quantization scope for the frozen TP4 model."""


def codec_placement(policy):
    return {'taco_separate': 'separate', 'taco_fused': 'fused',
            'taco_fused_legacy': 'fused'}.get(policy, 'bf16')


def module_routes(observed, policy, layers=12):
    assert policy in ('original','original_matched','taco_separate','taco_fused','taco_fused_legacy')
    counts = {'attention': 0, 'mlp': 0}
    routes = {}
    for name, shape in observed.items():
        if name.endswith('.self_attention.linear_proj'):
            kind, k = 'attention', 2048
        elif name.endswith('.mlp.linear_fc2'):
            kind, k = 'mlp', 8192
        else:
            raise AssertionError(('Unsupported target module', name))
        assert (shape['M'],shape['N'],shape['K_global'],shape['K_local']) == (2048,2048,k,k//4)
        counts[kind] += 1
        routes[name] = policy if policy.startswith('taco_') and kind == 'mlp' else 'original'
    assert counts == {'attention':layers,'mlp':layers}
    return routes
