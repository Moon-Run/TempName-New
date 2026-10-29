"""Five dual-role implementations, frozen into each ablation4 worker directory."""
import random

PROTOCOL = 'h100-gpt67-tp8-ablation4-v1'
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
POLICIES = ('native_taco_attn_mlp', 'flux_fused_v2_attn_mlp',
            'interleaved_all_attn_mlp', 'interleaved_arrival_all_attn_mlp',
            'interleaved_arrival_selective_attn_mlp')
LABELS = dict(zip(POLICIES, ('Megatron＋量化', 'Flux＋量化（融合v2）',
                           '交替＋全远端量化', '交替＋到达优先＋全远端量化',
                           '交替＋到达优先＋选择性量化')))
PRIMARY = list(zip(POLICIES[1:], POLICIES[:-1]))
COMPARISONS = [(a, b) for i, a in enumerate(POLICIES) for b in POLICIES[:i]]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def definition(policy):
    require(policy in POLICIES, 'Unknown ablation4 policy')
    i = POLICIES.index(policy)
    return dict(policy=policy, family=('native', 'fused_v2', 'A', 'B', 'C')[i],
                scope='attn_mlp', roles=['attn', 'mlp'],
                implementation=('native_megatron_tensor_taco', *(['a800-fused-v2-cta-v1']*3),
                                'frozen-candidate-v1-sm90-gemmv2')[i],
                workspace='tensor' if i == 0 else 'single' if i < 4 else 'double',
                output_clone=0 < i < 4,
                base=None if i == 0 else 'original' if i == 1 else 'interleaved',
                order=('native', 'original', 'base', 'arrival', 'arrival')[i],
                arrival=i >= 3, quantization='targeted' if i == 4 else 'all_remote')


def is_native(policy):
    return definition(policy)['family'] == 'native'


def is_fused_v2(policy):
    return definition(policy)['implementation'] == 'a800-fused-v2-cta-v1'


def is_original(policy):
    return definition(policy)['family'] == 'fused_v2'


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
    return 'tensor' if is_native(policy) else 'candidate'


def codec_placement(policy):
    return 'tensor' if is_native(policy) else 'fused'


def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Ablation4 fixes GPT6.7B/32 layers')
    counts = {role: 0 for role in SHAPES}; result = {}
    for name, shape in observed.items():
        role = module_role(name); counts[role] += 1
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role] and
                shape['K_local'] == SHAPES[role][2]//8, 'Role/full-shape mismatch')
        result[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 dual-role projections are required')
    return result


def role_plans(plan, policy):
    definition(policy)
    require(plan['protocol'] == 'h100-gpt67-tp8-ablation3-v1', 'Wrong shared A/B/C plan protocol')
    # Baselines do not consume these measured masks or reorder tables. The parent
    # records supply shape/hardware provenance only; their plan hashes stay null.
    return plan['parent']['roles'] if is_native(policy) or is_original(policy) else plan['variants'][policy]['roles']


def selection_entry(plan, policy, role, m, n, k):
    require(not is_native(policy) and not is_fused_v2(policy), 'A baseline must not consume a candidate plan')
    require(role in SHAPES and [m, n, k] == SHAPES[role], 'Wrong selected role/full shape')
    record = role_plans(plan, policy)[role]
    require(record['role'] == role and record['entry']['shape'] == [m, n, k] and
            record['entry']['K_local'] == k//8, 'Candidate shape dispatch mismatch')
    return record


def schedule(blocks=5, seed=20261013, rounds=1):
    require(type(blocks) is int and blocks > 0 and blocks % 5 == 0, 'Blocks must be a positive multiple of 5')
    require(type(rounds) is int and rounds in (1, 2), 'Rounds must be one or two')
    require(seed == 20261013, 'Ablation4 fixes schedule seed 20261013')
    order = list(POLICIES); random.Random(seed).shuffle(order)
    first = [order[i % 5:]+order[:i % 5] for i in range(blocks)]
    return [dict(round=rep, block=b, index=b*5+i, policy=p,
                 directory=f'round{rep}/window-{b*5+i:03d}-{p}')
            for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]])[:rounds], 1)
            for b, row in enumerate(rows) for i, p in enumerate(row)]
