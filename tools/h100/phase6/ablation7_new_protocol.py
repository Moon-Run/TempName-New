"""C(legacy order, 1/16), A(base, full remote), B(improved order, full remote)."""
import hashlib
import json
import random

PROTOCOL = 'h100-gpt67-tp8-ablation7-new-v1'
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
POLICIES = ('interleaved_arrival_selective_b16_attn_mlp', 'interleaved_all_attn_mlp',
            'interleaved_arrival_assignment_all_attn_mlp')
C, A, B = POLICIES
LABELS = dict(zip(POLICIES, ('C：完整版（选择性量化1/16）', 'A：交替＋完整量化',
                           'B：交替＋联合排列到达＋完整量化（新版）')))
PRIMARY = [(B, A)]
COMPARISONS = [(B, A), (C, A), (C, B)]
WARMUP_STEPS = 10
TIMED_STEPS = 50
TIMING = dict(warmup_steps=WARMUP_STEPS, timed_steps=TIMED_STEPS)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def execution_timing():
    return dict(TIMING)


def definition(policy):
    require(policy in POLICIES, 'Unknown ablation7_new policy')
    family = ('C', 'A', 'B')[POLICIES.index(policy)]
    selective = policy == C
    return dict(policy=policy, family=family, scope='attn_mlp', roles=['attn', 'mlp'],
                implementation='frozen-candidate-v1-sm90-gemmv2' if selective else 'a800-fused-v2-cta-v1',
                workspace='double' if selective else 'single', output_clone=not selective,
                base='interleaved', order='base' if policy == A else 'arrival', arrival=policy != A,
                arrival_method='legacy_tail' if selective else 'identity' if policy == A else 'tail_assignment',
                quantization='targeted' if selective else 'all_remote',
                budget_denominator=16 if selective else 1, budget_fraction=1/16 if selective else 1,
                max_tiles_per_source=28 if selective else 448)


def is_native(policy):
    definition(policy)
    return False


is_original = is_native


def is_fused_v2(policy):
    return definition(policy)['family'] in ('A', 'B')


def module_role(name):
    for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
        if name.endswith(suffix):
            return role
    raise ValueError('Unexpected target projection: '+name)


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
    require(layers == 32, 'Ablation7-new fixes GPT6.7B/32 layers')
    counts = {role: 0 for role in SHAPES}; result = {}
    for name, shape in observed.items():
        role = module_role(name); counts[role] += 1
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role] and
                shape['K_local'] == SHAPES[role][2]//8, 'Role/full-shape mismatch')
        result[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 dual-role projections are required')
    return result


def role_plans(plan, policy):
    require(plan['protocol'] == PROTOCOL, 'Wrong ablation7_new plan')
    record = plan['variants'][policy]
    require(record['definition'] == definition(policy) and set(record['roles']) == set(SHAPES),
            'Ablation7-new policy/scope changed')
    return record['roles']


def selection_entry(plan, policy, role, m, n, k):
    require(role in SHAPES and [m, n, k] == SHAPES[role], 'Wrong selected role/full shape')
    record = role_plans(plan, policy)[role]
    require(record['role'] == role and record['entry']['shape'] == [m, n, k] and
            record['entry']['K_local'] == k//8 and record['entry_sha256'] == digest(record['entry']),
            'Candidate shape/hash changed')
    return record


def schedule(blocks=6, seed=20261013, rounds=2):
    require(type(blocks) is int and blocks > 0 and blocks % 3 == 0,
            'Blocks must be a positive multiple of 3')
    require(type(rounds) is int and rounds in (1, 2), 'Rounds must be one or two')
    require(seed == 20261013, 'Ablation7-new fixes schedule seed 20261013')
    order = list(POLICIES); random.Random(seed).shuffle(order)
    first = [order[i % 3:]+order[:i % 3] for i in range(blocks)]
    return [dict(round=rep, block=b, index=b*3+i, policy=p,
                 directory=f'round{rep}/window-{b*3+i:03d}-{p}')
            for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]])[:rounds], 1)
            for b, row in enumerate(rows) for i, p in enumerate(row)]
