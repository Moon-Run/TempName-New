"""A/B/C unchanged; Flux GemmOnly and Megatron share the same Tensor TACO."""
import ablation4_protocol as parent

PROTOCOL = 'h100-gpt67-tp8-ablation6-v1'
FLUX_TENSOR = 'flux_tensor_taco_attn_mlp'
POLICIES = (parent.POLICIES[0], FLUX_TENSOR, *parent.POLICIES[2:])
SHAPES = parent.SHAPES
LABELS = dict(parent.LABELS, **{FLUX_TENSOR: 'Flux GEMM＋朴素量化'})
LABELS.pop(parent.POLICIES[1])
LABELS[POLICIES[0]] = 'Megatron＋朴素量化'
PRIMARY = [(FLUX_TENSOR, POLICIES[0]), (POLICIES[2], FLUX_TENSOR),
           (POLICIES[3], POLICIES[2]), (POLICIES[4], POLICIES[3])]
COMPARISONS = [(a, b) for i, a in enumerate(POLICIES) for b in POLICIES[:i]]
TIMING = dict(warmup_steps=10, timed_steps=50)
NAIVE_CODEC = 'shared-tensor-taco-h128-e4m3-v1'


def is_flux_tensor(policy):
    return policy == FLUX_TENSOR


def is_native(policy):
    return policy == POLICIES[0]


def is_tensor(policy):
    return policy in POLICIES[:2]


def is_fused_v2(policy):
    return policy in POLICIES[2:4]


def is_original(policy):
    return False


def needs_mapping(policy):
    return not is_tensor(policy)


def definition(policy):
    parent.require(policy in POLICIES, 'Unknown ablation6 policy')
    if not is_flux_tensor(policy):
        return parent.definition(policy)
    return dict(policy=policy, family='flux_tensor', scope='attn_mlp', roles=['attn', 'mlp'],
                implementation='flux-gemm-only-sm90-tensor-taco-v1', workspace='gemm_output',
                output_clone=False, base='original', order='gemm_only', arrival=False,
                quantization='all_remote', naive_codec=NAIVE_CODEC)


module_role = parent.module_role


def enabled(policy, name):
    definition(policy); module_role(name)
    return True


def route(policy, name):
    enabled(policy, name)
    return 'flux_tensor' if is_flux_tensor(policy) else parent.route(policy, name)


def codec_placement(policy):
    return 'tensor' if is_tensor(policy) else 'fused'


def module_routes(observed, policy, layers=32, model=None):
    checked = parent.module_routes(observed, POLICIES[0] if is_flux_tensor(policy) else policy, layers, model)
    return {name: route(policy, name) for name in checked}


def role_plans(plans, policy):
    definition(policy)
    return parent.role_plans(plans, POLICIES[0] if is_flux_tensor(policy) else policy)


def selection_entry(plans, policy, role, m, n, k):
    parent.require(not is_tensor(policy), 'Tensor baseline must not consume a selective mask')
    return parent.selection_entry(plans, policy, role, m, n, k)


def schedule(blocks=5, seed=20261013, rounds=2):
    jobs = parent.schedule(blocks, seed, rounds)
    for job in jobs:
        if job['policy'] == parent.POLICIES[1]:
            job['policy'] = FLUX_TENSOR
            job['directory'] = job['directory'].replace(parent.POLICIES[1], FLUX_TENSOR)
    return jobs
