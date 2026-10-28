"""Five budget ceilings on one unchanged dual-role selective candidate."""
import hashlib
import json
import random

PROTOCOL = 'h100-gpt67-tp8-ablation5-v1'
DENOMINATORS = (64, 32, 16, 8, 4)
POLICIES = tuple(f'selective_b{d}_attn_mlp' for d in DENOMINATORS)
LABELS = {p: f'1/{d}' for p, d in zip(POLICIES, DENOMINATORS)}
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
PRIMARY = [(p, POLICIES[0]) for p in POLICIES[1:]]
COMPARISONS = [(a, b) for i, a in enumerate(POLICIES) for b in POLICIES[:i]]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def definition(policy):
    require(policy in POLICIES, 'Unknown ablation5 budget')
    d = DENOMINATORS[POLICIES.index(policy)]
    return dict(policy=policy, family='C', scope='attn_mlp', roles=['attn', 'mlp'],
                implementation='frozen-candidate-v1-sm90-gemmv2', workspace='double', output_clone=False,
                base='interleaved', order='arrival', arrival=True, quantization='targeted',
                budget_denominator=d, budget_fraction=1/d, max_tiles_per_source=448//d)


def is_native(policy):
    definition(policy)
    return False


is_fused_v2 = is_native
is_original = is_native


def module_role(name):
    for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
        if name.endswith(suffix):
            return role
    raise ValueError('Unexpected projection: '+name)


def enabled(policy, name):
    definition(policy); module_role(name)
    return True


def route(policy, name):
    enabled(policy, name)
    return 'candidate'


def codec_placement(policy):
    definition(policy)
    return 'fused'


def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Only GPT6.7B/32 layers')
    counts = {r: 0 for r in SHAPES}; routes = {}
    for name, shape in observed.items():
        role = module_role(name); counts[role] += 1
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role] and
                shape['K_local'] == SHAPES[role][2]//8, 'Role/full-shape mismatch')
        routes[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 output projections required')
    return routes


def role_plans(plan, policy):
    require(plan['protocol'] == PROTOCOL, 'Wrong ablation5 plan')
    record = plan['variants'][policy]
    require(record['definition'] == definition(policy) and set(record['roles']) == set(SHAPES), 'Budget/scope changed')
    return record['roles']


def selection_entry(plan, policy, role, m, n, k):
    require(role in SHAPES and [m, n, k] == SHAPES[role], 'Wrong budget role/full shape')
    record = role_plans(plan, policy)[role]
    require(record['role'] == role and record['entry']['shape'] == [m, n, k] and
            record['entry']['K_local'] == k//8 and record['entry_sha256'] == digest(record['entry']),
            'Budget plan shape/hash changed')
    return record


def schedule(blocks=5, seed=20261013, rounds=2):
    require(type(blocks) is int and blocks > 0 and blocks % 5 == 0, 'Blocks must be a positive multiple of 5')
    require(type(rounds) is int and rounds in (1, 2), 'Rounds must be one or two')
    require(seed == 20261013, 'Fixed schedule seed required')
    order = list(POLICIES); random.Random(seed).shuffle(order)
    first = [order[i % 5:]+order[:i % 5] for i in range(blocks)]
    return [dict(round=rep, block=b, index=b*5+i, policy=p, directory=f'round{rep}/window-{b*5+i:03d}-{p}')
            for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]])[:rounds], 1)
            for b, row in enumerate(rows) for i, p in enumerate(row)]
