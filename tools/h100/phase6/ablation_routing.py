"""Worker-only routing definitions. Importing this module never imports Torch."""
import json
from pathlib import Path


def definition(policy):
    return json.loads((Path(__file__).parent/'config.json').read_text())['ablation_definitions'][policy]


def codec_placement(policy):
    # Even BF16 uses the frozen mixed-wire receiver with an explicit zero mask.
    # Its enabled instance selects GemmV2; unquantized attention remains V3.
    if policy == 'native_taco':
        return 'tensor'
    return 'bf16' if policy == 'native' else 'fused'


def base_policy(policy):
    return definition(policy)['base']


def selection_entry(plan, policy, m, n, k):
    if policy in ('native', 'native_taco'):
        raise ValueError('Native paths have no candidate tile plan')
    entry = plan['entries'][policy]
    if entry['shape'] != [m, n, k]:
        raise ValueError('Ablation plan/model shape mismatch')
    return entry


def module_routes(observed, policy, layers, model):
    counts = dict(attention=0, mlp=0)
    routes = {}
    for name, shape in observed.items():
        if name.endswith('.self_attention.linear_proj'):
            kind, k = 'attention', model['hidden']
        elif name.endswith('.mlp.linear_fc2'):
            kind, k = 'mlp', model['ffn']
        else:
            raise ValueError(f'Unexpected target module: {name}')
        expected = (model['sequence']*model['micro_batch'], model['hidden'], k, k//model['tp'])
        if tuple(shape[f] for f in ('M', 'N', 'K_global', 'K_local')) != expected:
            raise ValueError(f'Unexpected target shape: {name}')
        counts[kind] += 1
        routes[name] = ('native' if policy == 'native' or policy == 'native_taco' and kind == 'attention'
                        else policy if kind == 'mlp' else 'original')
    if counts != dict(attention=layers, mlp=layers):
        raise ValueError('Missing MLP/attention modules')
    return routes
