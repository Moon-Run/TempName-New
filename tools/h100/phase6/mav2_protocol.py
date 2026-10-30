"""Three full-remote codecs: H8 basic, selective D8/H8, and forced direct D8."""
from itertools import permutations
import random

PROTOCOL = 'h100-mav2-selective-transform-three-arm-v2'
POLICIES = ('basic', 'mav2', 'd8')
LABELS = {'basic': 'Flux＋基础量化 v2（全 H8）', 'mav2': 'Flux＋选择性变换（D8/H8）',
          'd8': 'Flux＋强制直接量化（全 D8）'}
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
TIMING = dict(warmup_steps=10, timed_steps=50)
CODEC = 'mav2-three-arm-h8-selective-d8-fixed136-v2'
COMPARISONS = [('mav2', 'basic'), ('d8', 'basic'), ('d8', 'mav2')]
EQUIVALENCE_PAIRS = []
SCHEDULE_SEED = 20261030
# Pre-registered short-run screening budgets; not a claim about convergence.
QUALITY_BUDGET = {'loss_atol': .05, 'loss_rtol': .05,
                  'grad_norm_atol': .1, 'grad_norm_rtol': .10}
TRANSFORM = {'group_size': 128, 'wire_bytes_per_group': 136,
             'd8_relative_l2': .025, 'd8_max_abs_over_amax': .0625,
             'selection': 'online-per-token-channel-group',
             'fallback': 'H8', 'bf16_fallback': False,
             'policies': {'basic': 'H8', 'mav2': 'D8-or-H8', 'd8': 'D8'},
             'forced_d8_error_guard': False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def definition(policy):
    require(policy in POLICIES, 'Unknown MAV2 policy')
    return dict(policy=policy, family='flux_tensor', scope='attn_mlp', roles=['attn', 'mlp'],
                implementation={'basic': 'flux-sm90-receiver-only-fusion-v2',
                                'mav2': 'flux-sm90-gemm-only-selective-'+CODEC,
                                'd8': 'flux-sm90-gemm-only-forced-d8-'+CODEC}[policy],
                workspace='gemm_output', output_clone=False, base='original', order='gemm_only',
                arrival=False, quantization='all_remote', prefetch=False, graph=False,
                transform=TRANSFORM['policies'][policy])


def is_native(policy):
    definition(policy)
    return False


def is_flux_tensor(policy):
    definition(policy)
    return True


def is_tensor(policy):
    return is_flux_tensor(policy)


def is_fused_v2(policy):
    # All arms use the GemmOnly/tensor frontend, including the basic receiver.
    return not is_flux_tensor(policy)


def is_original(policy):
    return is_fused_v2(policy)


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
    require(layers == 32, 'This experiment fixes 32 layers')
    counts = {r: 0 for r in SHAPES}
    result = {}
    for name, shape in observed.items():
        role = module_role(name)
        counts[role] += 1
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role] and
                shape['K_local'] == SHAPES[role][2]//8, 'Role/full-shape mismatch')
        result[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 projections must run')
    return result


def role_plans(plan, policy):
    definition(policy)
    require(plan['protocol'] == PROTOCOL and plan['calibration_used'] is False,
            'No selective mask or arrival calibration is allowed')
    return plan['roles']


def selection_entry(*args):
    raise ValueError('None of the three arms may consume a selective quantization mask')


def schedule(blocks=6, seed=SCHEDULE_SEED, rounds=2):
    orders = list(permutations(POLICIES))
    require(type(blocks) is int and blocks >= len(orders) and blocks % len(orders) == 0,
            'Use a positive multiple of all six three-arm permutations')
    require(type(rounds) is int and rounds == 2 and seed == SCHEDULE_SEED,
            'Two balanced internal rounds and the fixed schedule seed are required')
    random.Random(seed).shuffle(orders)
    first = [orders[i % len(orders)] for i in range(blocks)]
    return [dict(round=rep, block=b, index=b*len(POLICIES)+i, policy=p,
                 directory=f'round{rep}/window-{b*len(POLICIES)+i:03d}-{p}')
            for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]]), 1)
            for b, row in enumerate(rows) for i, p in enumerate(row)]
