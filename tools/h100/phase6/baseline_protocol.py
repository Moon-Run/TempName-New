"""A/B/C from ablation6; two standalone upstream CUDA codec baselines."""
import ablation4_protocol as parent

PROTOCOL = 'h100-gpt67-tp8-upstream-taco-baseline-v1'
NATIVE = 'native_cuda_taco_attn_mlp'
FLUX_TENSOR = 'flux_cuda_taco_attn_mlp'
POLICIES = (NATIVE, FLUX_TENSOR, *parent.POLICIES[2:])
SHAPES = parent.SHAPES
LABELS = {p: parent.LABELS[p] for p in POLICIES[2:]}
LABELS.update({NATIVE: 'Megatron＋独立TACO CUDA量化', FLUX_TENSOR: 'Flux＋独立TACO CUDA量化'})
TIMING = dict(warmup_steps=10, timed_steps=50)
CODEC = 'upstream-taco-cuda-h128-e4m3-ca717e05-v1'
PRIMARY = [(FLUX_TENSOR, NATIVE), (POLICIES[2], FLUX_TENSOR),
           (POLICIES[3], POLICIES[2]), (POLICIES[4], POLICIES[3])]
COMPARISONS = [(a, b) for i, a in enumerate(POLICIES) for b in POLICIES[:i]]


def is_native(policy):
    return policy == NATIVE


def is_flux_tensor(policy):
    return policy == FLUX_TENSOR


def is_tensor(policy):
    return policy in POLICIES[:2]


def is_fused_v2(policy):
    return policy in POLICIES[2:4]


def is_original(policy):
    return False


def needs_mapping(policy):
    return not is_tensor(policy)


def definition(policy):
    parent.require(policy in POLICIES, 'Unknown baseline policy')
    if not is_tensor(policy):
        return parent.definition(policy)  # A/B/C definitions are byte-for-byte equivalent.
    return dict(policy=policy, family='native' if is_native(policy) else 'flux_tensor',
                scope='attn_mlp', roles=['attn', 'mlp'],
                implementation=('native-megatron-' if is_native(policy) else 'flux-sm90-gemm-only-')+CODEC,
                workspace='tensor' if is_native(policy) else 'gemm_output', output_clone=False,
                base=None if is_native(policy) else 'original',
                order='native' if is_native(policy) else 'gemm_only', arrival=False,
                quantization='all_remote', codec=CODEC)


module_role = parent.module_role


def enabled(policy, name):
    definition(policy); module_role(name)
    return True


def route(policy, name):
    enabled(policy, name)
    return 'tensor' if is_native(policy) else 'flux_tensor' if is_flux_tensor(policy) else 'candidate'


def codec_placement(policy):
    return 'tensor' if is_tensor(policy) else 'fused'


def module_routes(observed, policy, layers=32, model=None):
    parent.module_routes(observed, parent.POLICIES[0], layers, model)
    return {name: route(policy, name) for name in observed}


def role_plans(plans, policy):
    definition(policy)
    return parent.role_plans(plans, parent.POLICIES[0] if is_tensor(policy) else policy)


def selection_entry(plans, policy, role, m, n, k):
    parent.require(not is_tensor(policy), 'CUDA baselines must not consume a selective mask')
    return parent.selection_entry(plans, policy, role, m, n, k)


def schedule(blocks=5, seed=20261013, rounds=2):
    names = dict(zip(parent.POLICIES[:2], POLICIES[:2]))
    jobs = parent.schedule(blocks, seed, rounds)
    for job in jobs:
        old = job['policy']
        job['policy'] = names.get(old, old)
        job['directory'] = job['directory'].replace(old, job['policy'])
    return jobs
