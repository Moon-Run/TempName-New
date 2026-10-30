"""Freeze independent legacy C and residual B orders from one hash-checked calibration."""
import copy
import csv
import math
from pathlib import Path

import ablation5_plan
import arrival_tail
import plan as fitter
import scope_plan
from ablation_plan import resolve_input, mask_sha
from ablation7_protocol import PROTOCOL, SHAPES, POLICIES, C, A, B, definition, digest, execution_timing
from common import read, require, validate_entry

ENTRY_FIELDS = ablation5_plan.ENTRY_FIELDS


def mapping_rows(entry):
    return [dict(rank=rank, index=i, m=t//32, n=t%32, destination=t//64)
            for rank, row in enumerate(entry['base_coords']) for i, t in enumerate(row)]


def measured_orders(parent):
    """Read only registered JSON/CSV, and reproduce legacy C before fitting B."""
    orders = {}
    for role, record in parent['roles'].items():
        origin = record['origin']
        require(origin['kind'] == 'scope_bf16_calibration', 'Both roles need explicit scope BF16 calibration')
        source = Path(origin['source']); used = {}
        def evidence(relative):
            name = str(source/relative)
            require(name in record['inputs'], 'Unregistered calibration input: '+name)
            path = resolve_input(name, record['inputs'][name]); used[name] = record['inputs'][name]
            return path
        done = read(evidence('calibration-complete.json'))
        require(done['protocol'] == scope_plan.PROTOCOL and done['completed'] and done['gpu_calibrated'] and
                done['run_id'] == origin['run_id'] and role in done['roles'] and
                scope_plan.hardware_identity(done['hardware']) == scope_plan.hardware_identity(record['hardware']),
                'Calibration completion identity changed')
        reports = [read(evidence(f'b0-interleaved-instrumented-rank{rank}.json')) for rank in range(8)]
        for rank, report in enumerate(reports):
            require(report['protocol'] == scope_plan.PROTOCOL and report['rank'] == rank and
                    report['run_id'] == origin['run_id'] and report['completed'] and
                    report['world_size'] == 8 and report['policy'] == 'interleaved' and
                    report['sampling'] and not report['smoke'] and
                    report['hardware']['local_rank'] == rank and
                    scope_plan.hardware_identity(report['hardware']) == scope_plan.hardware_identity(record['hardware']),
                    'Mixed/incomplete BF16 calibration observations')
        with evidence(f'mapping-{role}.csv').open() as handle:
            rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        legacy = fitter.fit(reports, rows, record['config'], 'interleaved', SHAPES[role], arrival_order='legacy_tail')
        require(legacy == record['entry'], 'Original C must reproduce the frozen legacy calibration')
        improved = fitter.fit(reports, rows, record['config'], 'interleaved', SHAPES[role], arrival_order='tail_residual')
        require(improved['mask'] == legacy['mask'], 'Improved order changed the calibration mask')
        orders[role] = dict(kind='measured_bf16', entry=improved, entry_sha256=digest(improved), inputs=used,
                            training_passes=[0, 1], heldout_passes=[2])
    return orders


def fixture_orders(parent):
    result = {}
    for role, record in parent['roles'].items():
        entry = copy.deepcopy(record['entry'])
        maps, diagnostics = arrival_tail.fit_order(mapping_rows(entry), {}, 8, record['config']['window'])
        entry.update(maps=maps, coords=copy.deepcopy(entry['base_coords']), arrival_order=diagnostics,
                     changed_per_rank=[0]*8)
        result[role] = dict(kind='cpu_fixture', entry=entry, entry_sha256=digest(entry), inputs={},
                            training_passes=[0, 1], heldout_passes=[2])
    return result


def derive(parent, scores, orders, policy):
    d = definition(policy); roles = {}
    for role in SHAPES:
        original = parent['roles'][role]
        source = orders[role]['entry'] if policy == B else original['entry']
        entry = {key: copy.deepcopy(source[key]) for key in ENTRY_FIELDS}
        if policy == A:
            entry['maps'] = [list(range(512)) for _ in range(8)]
            entry['coords'] = copy.deepcopy(entry['base_coords'])
        entry['mask'] = fitter.selection.select(scores[role]['scores'], 8, 1/16) if policy == C else [
            [int(tile//64 != rank) for tile in range(512)] for rank in range(8)]
        counts = list(map(sum, entry['mask'])); entry['selected_per_source'] = counts
        validate_entry(entry, dict(original['config'], fraction=d['budget_fraction']), SHAPES[role])
        roles[role] = dict(role=role, parent_entry_sha256=original['entry_sha256'], entry=entry,
                           entry_sha256=digest(entry), mask_sha256=mask_sha(entry['mask']),
                           selected_remote_fraction=sum(counts)/3584,
                           wire_bytes_per_source=[14680064-q*128*(256-136) for q in counts])
    return dict(definition=d, roles=roles)


def bundle(parent, source_only=False, verify_evidence=True):
    # Budget replay verifies that pass 0/1 scores reproduce the original 1/64 C.
    budget = ablation5_plan.bundle(parent, source_only, verify_evidence)
    orders = fixture_orders(parent) if source_only else measured_orders(parent)
    result = dict(protocol=PROTOCOL, source_only=source_only, gpu_calibrated=not source_only,
                  execution_timing=execution_timing(), parent=copy.deepcopy(parent), scores=budget['scores'], orders=orders,
                  variants={p: derive(parent, budget['scores'], orders, p) for p in POLICIES})
    return validate(result, verify_evidence=False)


def validate(plans, verify_evidence=True):
    require(plans['protocol'] == PROTOCOL and plans['execution_timing'] == execution_timing() and
            set(plans['variants']) == set(POLICIES) and
            set(plans['orders']) == set(plans['scores']) == set(plans['parent']['roles']) == set(SHAPES),
            'Wrong ablation7 plan set/timing')
    source_only = plans['source_only']; parent = plans['parent']
    require(type(source_only) is bool and source_only == parent['source_only'] and
            plans['gpu_calibrated'] == (not source_only) == parent['gpu_calibrated'], 'Plan rehearsal identity changed')
    identities = []
    expected_base = [[((i % 8+r) % 8)*64+i//8 for i in range(512)] for r in range(8)]
    for role, record in parent['roles'].items():
        scope_plan.validate(record, role, allow_test=source_only, verify_evidence=verify_evidence)
        require(record['entry']['base_coords'] == expected_base, 'Original C must use the interleaved base')
        if not source_only:
            require(not record['test_only'] and record['origin']['kind'] == 'scope_bf16_calibration',
                    'Artificial or frozen-MLP-only evidence cannot execute')
            identities.append(scope_plan.hardware_identity(record['hardware']))
        score = plans['scores'][role]; values = score['scores']; order = plans['orders'][role]
        require(score['training_passes'] == order['training_passes'] == [0, 1] and
                score['heldout_passes'] == order['heldout_passes'] == [2], 'Training/heldout split changed')
        require(len(values) == 8 and all(len(row) == 512 for row in values) and
                all(math.isfinite(x) and x >= 0 for row in values for x in row) and
                score['scores_sha256'] == digest(values), 'Score identity changed')
        kind = 'cpu_fixture' if source_only else 'measured_bf16'
        require(score['kind'] == order['kind'] == kind, 'Synthetic evidence cannot become executable')
        require(fitter.selection.select(values, 8, 1/64) == record['entry']['mask'],
                'Scores no longer reproduce original C mask')
        for evidence in (score, order):
            require((source_only or evidence['inputs']) and
                    all(record['inputs'].get(k) == v for k, v in evidence['inputs'].items()), 'Input evidence hash changed')
        entry = order['entry']; diagnostics = entry['arrival_order']
        validate_entry(entry, record['config'], SHAPES[role])
        require(order['entry_sha256'] == digest(entry) and entry['base_coords'] == expected_base and
                entry['mask'] == record['entry']['mask'], 'Improved order shape/hash/mask changed')
        require(diagnostics['method'] == arrival_tail.METHOD and diagnostics['training_passes'] == [0, 1] and
                diagnostics['heldout_passes'] == [2] and diagnostics['gpu_validated'] is False and
                diagnostics['changed_per_rank'] == [sum(i != j for i, j in enumerate(row)) for row in entry['maps']],
                'Improved order method/diagnostics changed')
    if identities:
        require(identities[0] == identities[1], 'Both role calibrations must use the same GPU order')
    if source_only:
        require(plans['orders'] == fixture_orders(parent), 'CPU rehearsal order changed')
    elif verify_evidence:
        require(plans['scores'] == ablation5_plan.measured_scores(parent), 'Measured calibration scores changed')
        require(plans['orders'] == measured_orders(parent), 'Improved mapping differs from raw-evidence replay')
    for policy in POLICIES:
        require(plans['variants'][policy] == derive(parent, plans['scores'], plans['orders'], policy),
                'Derived C/A/B order/mask/hash changed')
    return plans
