"""CPU-only four-scope definitions, also frozen as the worker's routing module."""
import hashlib
import json
import random

PROTOCOL = 'h100-gpt67-tp8-mlp-attention-v1'
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
POLICIES = ('native_taco_attn_mlp', 'native_taco_attn',
            'interleaved_full_attn', 'interleaved_full_attn_mlp')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def definition(policy):
    require(policy in POLICIES, 'Unknown scope policy (legacy policies are unchanged)')
    return dict(policy=policy, roles=['attn', 'mlp'] if policy.endswith('_attn_mlp') else ['attn'],
                implementation='native_tensor_taco' if policy.startswith('native_') else 'frozen_candidate_v2',
                base=None if policy.startswith('native_') else 'interleaved',
                arrival=not policy.startswith('native_'),
                quantization='all_remote' if policy.startswith('native_') else 'targeted')


def is_native(policy):
    return definition(policy)['implementation'] == 'native_tensor_taco'


def module_role(name):
    if name.endswith('.self_attention.linear_proj'):
        return 'attn'
    if name.endswith('.mlp.linear_fc2'):
        return 'mlp'
    raise ValueError('Unexpected target projection: '+name)


def enabled(policy, name):
    return module_role(name) in definition(policy)['roles']


def route(policy, name):
    return ('tensor' if is_native(policy) else 'candidate') if enabled(policy, name) else 'native_bf16'


def codec_placement(policy):
    return 'tensor' if is_native(policy) else 'fused'


def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Only GPT6.7B/32 layers')
    counts = {'attn': 0, 'mlp': 0}
    result = {}
    for name, s in observed.items():
        role = module_role(name)
        require([s[k] for k in ('M', 'N', 'K_global')] == SHAPES[role] and
                s['K_local'] == SHAPES[role][2]//8, 'Role/full-shape mismatch')
        counts[role] += 1
        result[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'Missing/extra projection modules')
    return result


def selection_entry(plan, policy, role, m, n, k):
    require(not is_native(policy) and role in definition(policy)['roles'], 'Unselected/native module requested candidate')
    require(role in SHAPES and [m, n, k] == SHAPES[role], 'Role/full-shape dispatch mismatch')
    record = plan['roles'][role]
    require(record['role'] == role and record['entry']['shape'] == [m, n, k] and
            record['entry']['K_local'] == k//8, 'Wrong role plan')
    require(record['entry_sha256'] == digest(record['entry']), 'Role entry hash changed')
    return record


def role_plans(plan, policy):
    definition(policy)
    return plan['roles']


def schedule(blocks=4, seed=20261013):
    require(type(blocks) is int and blocks > 0 and blocks % 4 == 0, 'Blocks must be a positive multiple of 4')
    require(seed == 20261013, 'Scope protocol fixes schedule seed 20261013')
    order = list(POLICIES); random.Random(seed).shuffle(order)
    first = [order[i % 4:]+order[:i % 4] for i in range(blocks)]
    return [dict(round=rep, block=b, index=b*4+i, policy=p,
                 directory=f'round{rep}/window-{b*4+i:03d}-{p}')
            for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]]), 1)
            for b, row in enumerate(rows) for i, p in enumerate(row)]
