"""Standalone nine-policy routing, copied into each frozen worker directory."""
import hashlib
import json
import random

PROTOCOL = 'h100-gpt67-tp8-ablation3-v1'
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
SCOPES = ('mlp', 'attn', 'attn_mlp')
PREFIXES = {'A': 'interleaved_all', 'B': 'interleaved_arrival_all',
            'C': 'interleaved_arrival_selective'}
POLICIES = tuple(prefix+'_'+scope for scope in SCOPES for prefix in PREFIXES.values())
_POLICY_META = {prefix+'_'+scope: (family, scope) for scope in SCOPES for family, prefix in PREFIXES.items()}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def policy_id(family, scope):
    require(family in PREFIXES and scope in SCOPES, 'Unknown ablation3 family/scope')
    return PREFIXES[family]+'_'+scope


def definition(policy):
    require(policy in POLICIES, 'Unknown ablation3 policy')
    family, scope = _POLICY_META[policy]
    return dict(policy=policy, family=family, scope=scope,
                roles=['attn', 'mlp'] if scope == 'attn_mlp' else [scope],
                implementation='frozen-candidate-v1-sm90-gemmv2', base='interleaved',
                order='base' if family == 'A' else 'arrival', arrival=family != 'A',
                quantization='targeted' if family == 'C' else 'all_remote')


def is_native(policy):
    definition(policy)
    return False


def module_role(name):
    if name.endswith('.self_attention.linear_proj'):
        return 'attn'
    if name.endswith('.mlp.linear_fc2'):
        return 'mlp'
    raise ValueError('Unexpected target projection: '+name)


def enabled(policy, name):
    return module_role(name) in definition(policy)['roles']


def route(policy, name):
    return 'candidate' if enabled(policy, name) else 'native_bf16'


def codec_placement(policy):
    definition(policy)
    return 'fused'


def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Only GPT6.7B/32 layers')
    counts = {role: 0 for role in SHAPES}
    result = {}
    for name, shape in observed.items():
        role = module_role(name)
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role] and
                shape['K_local'] == SHAPES[role][2]//8, 'Role/full-shape mismatch')
        counts[role] += 1
        result[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'Missing/extra projection modules')
    return result


def role_plans(plan, policy):
    require(plan['protocol'] == PROTOCOL, 'Wrong ablation3 plan protocol')
    variant = plan['variants'][policy]
    require(variant['definition'] == definition(policy), 'Wrong ablation3 plan definition')
    require(set(variant['roles']) == set(definition(policy)['roles']), 'Wrong plan scope')
    return variant['roles']


def selection_entry(plan, policy, role, m, n, k):
    require(role in definition(policy)['roles'] and [m, n, k] == SHAPES[role], 'Wrong selected role/full shape')
    record = role_plans(plan, policy)[role]
    require(record['role'] == role and record['entry']['shape'] == [m, n, k] and
            record['entry']['K_local'] == k//8 and record['entry_sha256'] == digest(record['entry']),
            'Role plan/hash changed')
    return record


def schedule(blocks=9, seed=20261013, rounds=2):
    require(type(blocks) is int and blocks > 0 and blocks % 9 == 0, 'Blocks must be a positive multiple of 9')
    require(type(rounds) is int and rounds in (1, 2), 'Rounds must be one or two')
    require(seed == 20261013, 'Ablation3 fixes schedule seed 20261013')
    order = list(POLICIES); random.Random(seed).shuffle(order)
    first = [order[i % 9:]+order[:i % 9] for i in range(blocks)]
    return [dict(round=rep, block=b, index=b*9+i, policy=p,
                 directory=f'round{rep}/window-{b*9+i:03d}-{p}')
            for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]])[:rounds], 1)
            for b, row in enumerate(rows) for i, p in enumerate(row)]


PRIMARY = [(policy_id(x, s), policy_id(y, s)) for s in SCOPES
           for x, y in (('B', 'A'), ('C', 'B'), ('C', 'A'))]
RANGE_COMPARISONS = [(policy_id(f, x), policy_id(f, y)) for f in PREFIXES
                    for x, y in (('attn_mlp', 'mlp'), ('attn_mlp', 'attn'), ('mlp', 'attn'))]
COMPARISONS = PRIMARY + RANGE_COMPARISONS
