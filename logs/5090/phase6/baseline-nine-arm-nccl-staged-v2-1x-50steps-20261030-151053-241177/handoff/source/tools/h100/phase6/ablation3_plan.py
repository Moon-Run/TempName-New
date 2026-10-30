"""Derive all nine physical plans from two shared, independently calibrated parents."""
import copy

from ablation_plan import mask_sha
from common import require, validate_entry
import scope_plan
from ablation3_protocol import PROTOCOL, SHAPES, POLICIES, definition, digest


def derive(parent, policy):
    d = definition(policy)
    roles = {}
    for role in d['roles']:
        p = parent['roles'][role]
        e = {k: copy.deepcopy(p['entry'][k]) for k in
             ('shape', 'M', 'N', 'K_global', 'K_local', 'tiles', 'maps', 'coords', 'base_coords', 'mask')}
        if not d['arrival']:
            e['maps'] = [list(range(512)) for _ in range(8)]
        e['coords'] = [[e['base_coords'][r][i] for i in e['maps'][r]] for r in range(8)]
        if d['quantization'] == 'all_remote':
            e['mask'] = [[int(t//64 != r) for t in range(512)] for r in range(8)]
        e['selected_per_source'] = list(map(sum, e['mask']))
        # Only full-remote mode lifts the sparse budget; local BF16 and all
        # physical-coordinate/window invariants are still checked.
        config = dict(p['config'], fraction=1 if d['quantization'] == 'all_remote' else p['config']['fraction'])
        validate_entry(e, config, SHAPES[role])
        roles[role] = dict(role=role, parent_entry_sha256=p['entry_sha256'],
                           entry=e, entry_sha256=digest(e), mask_sha256=mask_sha(e['mask']))
    return dict(definition=d, roles=roles)


def bundle(parent):
    result = dict(protocol=PROTOCOL, parent=copy.deepcopy(parent), source_only=parent['source_only'],
                  gpu_calibrated=parent['gpu_calibrated'],
                  variants={p: derive(parent, p) for p in POLICIES})
    validate(result)
    return result


def validate(plans, verify_evidence=True):
    require(plans['protocol'] == PROTOCOL and set(plans['variants']) == set(POLICIES), 'Wrong nine-group plans')
    parent = plans['parent']
    require(set(parent['roles']) == set(SHAPES) and plans['source_only'] == parent['source_only'] and
            plans['gpu_calibrated'] == parent['gpu_calibrated'], 'Parent/rehearsal identity changed')
    real = []
    for role, p in parent['roles'].items():
        scope_plan.validate(p, role, allow_test=plans['source_only'], verify_evidence=verify_evidence)
        expected = [[((i % 8+r) % 8)*64+i//8 for i in range(512)] for r in range(8)]
        require(p['entry']['base_coords'] == expected, 'Parent is not the interleaved base')
        if not p['test_only']:
            real.append(p)
    if not plans['source_only']:
        require(plans['gpu_calibrated'] and len(real) == 2, 'Both real shape calibrations required')
    if len(real) == 2:
        require(scope_plan.hardware_identity(real[0]['hardware']) == scope_plan.hardware_identity(real[1]['hardware']),
                'Role hardware/order differs; recalibrate on the same allocation')
    for policy in POLICIES:
        require(plans['variants'][policy] == derive(parent, policy), 'Derived order/mask/role identity changed')
    return plans
