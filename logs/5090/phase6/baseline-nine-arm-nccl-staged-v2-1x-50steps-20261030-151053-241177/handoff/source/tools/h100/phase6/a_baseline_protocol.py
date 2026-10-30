"""Three all-remote TP8 baselines; only the producer-fusion pair is bitwise equivalent."""
import random

PROTOCOL = 'h100-basic-fused-baseline-fullstep-v1'
POLICIES = ('basic', 'improved', 'cuda')
LABELS = {'basic': 'Flux＋基础量化', 'improved': 'Flux+改进量化', 'cuda': 'Flux＋CUDA量化'}
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
TIMING = dict(warmup_steps=10, timed_steps=50)
CODEC = 'upstream-taco-cuda-h128-e4m3-ca717e05-v1'
COMPARISONS = [('improved', 'basic'), ('basic', 'cuda'), ('improved', 'cuda')]
EQUIVALENCE_PAIRS = [['basic', 'improved']]
SCHEDULE_SEED = 20261024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def definition(policy):
    require(policy in POLICIES, 'Unknown basic-fusion comparison policy')
    cuda = policy == 'cuda'
    return dict(policy=policy, family='flux_tensor' if cuda else 'fused_v2',
                scope='attn_mlp', roles=['attn', 'mlp'],
                implementation={'basic': 'a-baseline-producer-fused-staged-decode-v1',
                                'improved': 'a800-fused-v2-cta-v1',
                                'cuda': 'flux-sm90-gemm-only-'+CODEC}[policy],
                workspace='gemm_output' if cuda else 'single', output_clone=not cuda,
                base='original', order='gemm_only' if cuda else 'original', arrival=False,
                quantization='all_remote', prefetch=False, graph=False)


def is_native(policy):
    definition(policy)
    return False


def is_flux_tensor(policy):
    definition(policy)
    return policy == 'cuda'


def is_tensor(policy):
    return is_flux_tensor(policy)


def is_fused_v2(policy):
    # This predicate selects the shared legacy frontend/lifecycle, not a kernel name.
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
    raise ValueError('All three baselines must not consume a selective mask')


def schedule(blocks=6, seed=SCHEDULE_SEED, rounds=2):
    require(type(blocks) is int and blocks >= 6 and blocks % 3 == 0,
            'Use at least six complete blocks, divisible by three')
    require(type(rounds) is int and rounds == 2 and seed == SCHEDULE_SEED,
            'Two balanced rounds and the fixed schedule seed are required')
    order = list(POLICIES)
    random.Random(seed).shuffle(order)
    first = [order[i % 3:]+order[:i % 3] for i in range(blocks)]
    return [dict(round=rep, block=b, index=b*3+i, policy=p,
                 directory=f'round{rep}/window-{b*3+i:03d}-{p}')
            for rep, rows in enumerate((first, [r[::-1] for r in first[::-1]]), 1)
            for b, row in enumerate(rows) for i, p in enumerate(row)]
