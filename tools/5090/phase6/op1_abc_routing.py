"""Standalone OP1 ABC routing; sparse selection is generated from output shape."""
PROTOCOL = 'rtx5090-phase6-op1-abc-staged-v1'
POLICIES = ('flux-basic-interleaved', 'flux-basic-arrival', 'flux-basic-selective', 'flux-improved')
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def definition(policy):
    require(policy in POLICIES, 'Unknown Flux quant baseline policy')
    native = policy == 'flux-improved'
    return dict(policy=policy, family='candidate' if native else 'flux_tensor',
        scope='attn_mlp', roles=list(SHAPES), implementation='rtx5090-'+policy+'-v1',
        workspace='double' if policy == 'flux-improved' else 'single' if native else 'tensor',
        output_clone=False, base='original', order='original' if native else 'interleaved',
        arrival=policy in POLICIES[1:3], interleaved=not native,
        quantization='selective-tail-1/64' if policy == POLICIES[2] else 'all_remote')


def is_native(policy):
    definition(policy)
    return False


def is_flux_tensor(policy):
    definition(policy)
    return policy in POLICIES[:3]


def is_tensor(policy):
    return is_flux_tensor(policy)


def is_fused_v2(policy):
    return not is_flux_tensor(policy)


def is_original(policy):
    definition(policy)
    return True


def needs_mapping(policy):
    return is_fused_v2(policy)


def module_role(name):
    for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
        if name.endswith(suffix):
            return role
    raise ValueError('Unexpected target projection: '+name)


def enabled(policy, name):
    definition(policy)
    module_role(name)
    return True


def route(policy, name):
    enabled(policy, name)
    return 'flux_tensor' if is_flux_tensor(policy) else 'candidate'


def codec_placement(policy):
    return 'tensor' if is_flux_tensor(policy) else 'fused'


def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Expected 32 layers')
    counts, routes = dict.fromkeys(SHAPES, 0), {}
    for name, shape in observed.items():
        role = module_role(name)
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role]
                and shape['K_local'] == SHAPES[role][2]//8, 'Wrong projection shape')
        counts[role] += 1
        routes[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 projections required')
    return routes


def role_plans(plan, policy):
    definition(policy)
    require(plan['protocol'] == PROTOCOL and plan['calibration_used'] is False
            and set(plan) == {'protocol', 'calibration_used', 'roles'}, 'No calibration inputs permitted')
    return plan['roles']


def selection_entry(*args):
    raise ValueError('ABC derives its sparse H128 mask from shape; no historical table is used')
