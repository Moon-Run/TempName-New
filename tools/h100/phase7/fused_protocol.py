"""Frozen nine-policy routing for improvements to the original fused path.

All policies share the original selective physical mask and codec. Only the
destination-preserving tile permutation changes among the three orders.
This file is copied as routing.py into a prepared model-worker directory.
"""
import hashlib
import json
import random

PROTOCOL = 'h100-gpt67-tp8-phase7-fused-v1'
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
ORDERS = ('base', 'legacy', 'improved')
SCOPES = ('mlp', 'attn', 'attn_mlp')
POLICIES = tuple('fused_' + order + '_' + scope for scope in SCOPES for order in ORDERS)
_META = {'fused_' + order + '_' + scope: (order, scope) for scope in SCOPES for order in ORDERS}
SCHEDULE_SEED = 20261017


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def policy_id(order, scope):
    require(order in ORDERS and scope in SCOPES, 'Unknown fused order/scope')
    return 'fused_' + order + '_' + scope


def definition(policy):
    require(policy in POLICIES, 'Unknown Phase7 fused policy')
    order, scope = _META[policy]
    return dict(policy=policy, scope=scope,
                roles=['attn', 'mlp'] if scope == 'attn_mlp' else [scope],
                implementation='frozen-candidate-v1-sm90-gemmv2', base='interleaved',
                order=order, arrival=order != 'base', quantization='targeted')


def is_native(policy):
    definition(policy)
    return False


def module_role(name):
    if name.endswith('.self_attention.linear_proj'):
        return 'attn'
    if name.endswith('.mlp.linear_fc2'):
        return 'mlp'
    raise ValueError('Unexpected target projection: ' + name)


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
                shape['K_local'] == SHAPES[role][2] // 8, 'Role/full-shape mismatch')
        counts[role] += 1
        result[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'Missing/extra projection modules')
    return result


def role_plans(plan, policy):
    require(plan['protocol'] == PROTOCOL, 'Wrong Phase7 fused plan protocol')
    variant = plan['variants'][policy]
    require(variant['definition'] == definition(policy), 'Wrong fused plan definition')
    require(set(variant['roles']) == set(definition(policy)['roles']), 'Wrong plan scope')
    return variant['roles']


def selection_entry(plan, policy, role, m, n, k):
    require(role in definition(policy)['roles'] and [m, n, k] == SHAPES[role], 'Wrong selected role/full shape')
    record = role_plans(plan, policy)[role]
    require(record['role'] == role and record['entry']['shape'] == [m, n, k] and
            record['entry']['K_local'] == k // 8 and record['entry_sha256'] == digest(record['entry']),
            'Role plan/hash changed')
    return record


def schedule(blocks=9, seed=SCHEDULE_SEED, rounds=2):
    require(type(blocks) is int and blocks > 0 and blocks % 9 == 0,
            'Blocks must be a positive multiple of nine')
    require(type(rounds) is int and rounds in (1, 2), 'Rounds must be one or two')
    require(seed == SCHEDULE_SEED, 'Phase7 fused fixes schedule seed 20261017')
    order = list(POLICIES)
    random.Random(seed).shuffle(order)
    first = [order[i % 9:] + order[:i % 9] for i in range(blocks)]
    return [dict(round=rep, block=block, index=block * 9 + index, policy=policy,
                 directory=f'round{rep}/window-{block*9+index:03d}-{policy}')
            for rep, rows in enumerate((first, [row[::-1] for row in first[::-1]])[:rounds], 1)
            for block, row in enumerate(rows) for index, policy in enumerate(row)]


PRIMARY = [(policy_id('improved', scope), policy_id('legacy', scope)) for scope in SCOPES]
SECONDARY = [(policy_id(x, scope), policy_id('base', scope))
             for scope in SCOPES for x in ('legacy', 'improved')]
RANGE_COMPARISONS = [(policy_id(order, x), policy_id(order, y)) for order in ORDERS
                     for x, y in (('attn_mlp', 'mlp'), ('attn_mlp', 'attn'), ('mlp', 'attn'))]
COMPARISONS = PRIMARY + SECONDARY + RANGE_COMPARISONS
