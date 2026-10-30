"""Derive pure scheduling candidates while preserving A's exact quantization plan.

The inherited BF16 calibration records identify the old A build only. Candidate
spans come from a separate probe of the unchanged full-remote codec, never from
those BF16 arrival timestamps.
"""
import copy

import ablation3_plan
import arrival_pair_order as pair_order
import scope_plan
from arrival_op2_protocol import (PROTOCOL, SHAPES, POLICIES, A, S, IDENTITY_SPANS,
                                  definition, digest, execution_timing)
from common import require, validate_entry


def frozen_options(record):
    require('schedule_options' in record, 'Missing frozen schedule options')
    require('arrival_options' not in record, 'Legacy BF16 arrival fit is not part of order-v2')
    return pair_order.validate_spans(record['schedule_options'])


def derive(parent, spans, policy):
    spans = pair_order.validate_spans(spans)
    roles = copy.deepcopy(ablation3_plan.derive(parent, A)['roles'])
    if policy == S:
        for role, record in roles.items():
            mapping = pair_order.role_mapping(spans[role])
            maps, coords, base_coords = (mapping[key] for key in ('maps', 'coords', 'base_coords'))
            entry = record['entry']
            require(entry['base_coords'] == base_coords, 'Pair scheduler requires exact A interleaving')
            entry['maps'], entry['coords'] = copy.deepcopy(maps), copy.deepcopy(coords)
            record['entry_sha256'] = digest(entry)
    for role, record in roles.items():
        validate_entry(record['entry'], dict(parent['roles'][role]['config'], fraction=1), SHAPES[role])
    return dict(definition=definition(policy), roles=roles)


def bundle(parent, source_only=False, verify_evidence=True, spans=None):
    spans = pair_order.validate_spans(IDENTITY_SPANS if spans is None else spans)
    require(source_only == parent['source_only'], 'Cannot promote a source-only parent')
    result = dict(protocol=PROTOCOL, source_only=source_only, gpu_calibrated=not source_only,
                  execution_timing=execution_timing(), parent=copy.deepcopy(parent),
                  schedule_options=spans,
                  schedule_diagnostics={role: pair_order.diagnostics(spans[role]) for role in SHAPES},
                  variants={p: derive(parent, spans, p) for p in POLICIES})
    return validate(result, verify_evidence=verify_evidence)


def validate(plans, verify_evidence=True):
    spans = frozen_options(plans)
    parent, source_only = plans['parent'], plans['source_only']
    require(plans['protocol'] == PROTOCOL and plans['execution_timing'] == execution_timing() and
            set(plans['variants']) == set(POLICIES) and set(parent['roles']) == set(SHAPES),
            'Wrong arrival-op2 plan set/timing')
    require(type(source_only) is bool and source_only == parent['source_only'] and
            plans['gpu_calibrated'] == (not source_only) == parent['gpu_calibrated'],
            'Plan rehearsal identity changed')
    expected_base = [[((i % 8+r) % 8)*64+i//8 for i in range(512)] for r in range(8)]
    identities = []
    for role, record in parent['roles'].items():
        scope_plan.validate(record, role, allow_test=source_only, verify_evidence=verify_evidence)
        require(record['entry']['base_coords'] == expected_base, 'Parent must use interleaved base')
        if not source_only:
            require(not record['test_only'], 'Both original A role provenances required')
            identities.append(scope_plan.hardware_identity(record['hardware']))
        require(plans['schedule_diagnostics'][role] == pair_order.diagnostics(spans[role]),
                'Schedule diagnostic changed')
    if identities:
        require(identities[0] == identities[1], 'Role hardware/order differs')
    for policy in POLICIES:
        require(plans['variants'][policy] == derive(parent, spans, policy),
                'Derived schedule/mask/role identity changed')
    # The only mutable fields in the physical plan are maps/coords and their
    # dependent entry digest. Masks, shapes and parent provenance remain A's.
    a, s = (plans['variants'][p]['roles'] for p in (A, S))
    for role in SHAPES:
        require(a[role]['mask_sha256'] == s[role]['mask_sha256'], 'Codec mask differs from A')
        require({k: v for k, v in a[role]['entry'].items() if k not in ('maps', 'coords')} ==
                {k: v for k, v in s[role]['entry'].items() if k not in ('maps', 'coords')},
                'Scheduling candidate changed a quantization input')
    return plans
