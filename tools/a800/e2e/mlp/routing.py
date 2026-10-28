"""Audit the two distinct shapes of the frozen MLP-only model experiment."""


def module_routes(observed, policy, layers=12):
    assert policy in ('native', 'original', 'mlp_remote', 'mlp_remote_arrival')
    counts = {'attention': 0, 'mlp': 0}
    routes = {}
    for name, shape in observed.items():
        if name.endswith('.self_attention.linear_proj'):
            kind, k = 'attention', 2048
        elif name.endswith('.mlp.linear_fc2'):
            kind, k = 'mlp', 8192
        else:
            raise AssertionError(('Unsupported target module', name))
        assert (shape['M'], shape['N'], shape['K_global'], shape['K_local']) == (2048, 2048, k, k//4), (name, shape)
        counts[kind] += 1
        routes[name] = ('native' if policy == 'native' else
                        'remote_arrival' if policy == 'mlp_remote_arrival' and kind == 'mlp' else
                        'remote_first' if policy == 'mlp_remote' and kind == 'mlp' else 'original')
    assert counts == {'attention': layers, 'mlp': layers}, counts
    return routes
