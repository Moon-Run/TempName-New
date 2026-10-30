"""Rebuild nested positive-score masks from the SAME BF16 passes 0/1."""
import copy
import csv
import math
from pathlib import Path

import plan as fitter
import scope_plan
from ablation_plan import resolve_input, mask_sha
from ablation5_protocol import PROTOCOL, POLICIES, SHAPES, definition, digest
from common import read, require, validate_entry

ENTRY_FIELDS = ('shape', 'M', 'N', 'K_global', 'K_local', 'tiles', 'maps', 'coords', 'base_coords')


def training_scores(reports, role):
    """Identical tail scoring to plan.fit; pass 2 never enters these scores."""
    scores = [[0.]*512 for _ in range(8)]
    require(len(reports) == 8, 'Eight calibration ranks required')
    for rank, report in enumerate(reports):
        require(report['rank'] == rank and report['completed'], 'Incomplete calibration rank')
        shape = next(s for s in report['shapes'] if [s[k] for k in ('M', 'N', 'K_global')] == SHAPES[role])
        observations = fitter.arrival.observations(shape, rank, 8, relative=True)
        for tile in range(rank*64, (rank+1)*64):
            if all((p, tile) in observations for p in (0, 1)):
                values = fitter.arrival.priority([observations[(p, tile)] for p in (0, 1)], 8, 'tail')
                for source, value in enumerate(values): scores[source][tile] = value
    return scores


def measured_scores(parent):
    """Read only hash-registered inputs; never execute a returned controller."""
    roles = {}; cached = {}
    for role, record in parent['roles'].items():
        origin = record['origin']
        require(origin['kind'] == 'scope_bf16_calibration', 'Use explicit dual-role scope BF16 calibration')
        source = Path(origin['source']); evidence = {}
        def get(relative):
            name = str(source/relative)
            require(name in record['inputs'], 'Calibration input is not registered: '+name)
            path = resolve_input(name, record['inputs'][name]); evidence[name] = record['inputs'][name]
            return path
        done = read(get('calibration-complete.json'))
        require(done['completed'] and done['gpu_calibrated'] and done['run_id'] == origin['run_id'] and
                scope_plan.hardware_identity(done['hardware']) == scope_plan.hardware_identity(record['hardware']),
                'Calibration identity changed')
        reports = []
        for rank in range(8):
            path = get(f'b0-interleaved-instrumented-rank{rank}.json')
            if path not in cached: cached[path] = read(path)
            row = cached[path]
            require(row['run_id'] == origin['run_id'] and row['world_size'] == 8 and
                    row['policy'] == 'interleaved' and row['sampling'] and not row['smoke'] and
                    scope_plan.hardware_identity(row['hardware']) == scope_plan.hardware_identity(record['hardware']),
                    'Mixed calibration observations')
            reports.append(row)
        with get(f'mapping-{role}.csv').open() as f:
            mapping = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
        # Reproducing the original fit also checks the arrival table and held-out
        # diagnostics. Increasing the budget below never reorders that table.
        reproduced = fitter.fit(reports, mapping, record['config'], 'interleaved', SHAPES[role])
        require(reproduced == record['entry'], 'Cannot reproduce the original 1/64 C plan')
        scores = training_scores(reports, role)
        require(fitter.selection.select(scores, 8, 1/64) == record['entry']['mask'], '1/64 no longer matches C')
        roles[role] = dict(kind='measured_bf16', scores=scores, scores_sha256=digest(scores),
                           inputs=evidence, training_passes=[0, 1], heldout_passes=[2])
    return roles


def derive(parent, scores, policy):
    d = definition(policy); roles = {}
    for role in SHAPES:
        original = parent['roles'][role]
        entry = {k: copy.deepcopy(original['entry'][k]) for k in ENTRY_FIELDS}
        mask = fitter.selection.select(scores[role]['scores'], 8, d['budget_fraction'])
        counts = list(map(sum, mask))
        entry.update(mask=mask, selected_per_source=counts)
        validate_entry(entry, dict(original['config'], fraction=d['budget_fraction']), SHAPES[role])
        roles[role] = dict(role=role, parent_entry_sha256=original['entry_sha256'],
            entry=entry, entry_sha256=digest(entry), mask_sha256=mask_sha(mask),
            selected_remote_fraction=sum(counts)/3584,
            wire_bytes_per_source=[14680064-q*128*(256-136) for q in counts])
    return dict(definition=d, roles=roles)


def bundle(parent, source_only=False, verify_evidence=True):
    for role, record in parent['roles'].items():
        scope_plan.validate(record, role, allow_test=source_only, verify_evidence=verify_evidence)
    if source_only:
        # Artificial scores deliberately reproduce the fixture's existing mask;
        # these cannot become executable measured plans by changing CLI flags.
        scores = {r: dict(kind='cpu_fixture', scores=[[float(x) for x in row] for row in p['entry']['mask']],
                         inputs={}, training_passes=[0, 1], heldout_passes=[2]) for r, p in parent['roles'].items()}
        for s in scores.values(): s['scores_sha256'] = digest(s['scores'])
    else:
        require(not parent['source_only'] and parent['gpu_calibrated'], 'Real parent calibration required')
        scores = measured_scores(parent)
    result = dict(protocol=PROTOCOL, source_only=source_only, gpu_calibrated=not source_only,
                  parent=copy.deepcopy(parent), scores=scores,
                  variants={p: derive(parent, scores, p) for p in POLICIES})
    validate(result, verify_evidence=False)
    return result


def validate(plans, verify_evidence=True):
    require(plans['protocol'] == PROTOCOL and set(plans['variants']) == set(POLICIES) and
            set(plans['scores']) == set(plans['parent']['roles']) == set(SHAPES), 'Wrong budget plan set')
    require(plans['source_only'] == (not plans['gpu_calibrated']), 'Plan rehearsal status changed')
    identities = []
    for role, record in plans['parent']['roles'].items():
        scope_plan.validate(record, role, allow_test=plans['source_only'], verify_evidence=verify_evidence)
        s = plans['scores'][role]; values = s['scores']
        require(s['training_passes'] == [0, 1] and s['heldout_passes'] == [2] and
                len(values) == 8 and all(len(v) == 512 for v in values) and
                all(math.isfinite(x) and x >= 0 for v in values for x in v) and
                s['scores_sha256'] == digest(values), 'Score/split identity changed')
        require(s['kind'] == ('cpu_fixture' if plans['source_only'] else 'measured_bf16'), 'Artificial scores cannot execute')
        if not plans['source_only']:
            require(not plans['parent']['source_only'] and plans['parent']['gpu_calibrated'] and
                    s['inputs'] and not record['test_only'], 'Missing measured score provenance')
            require(all(record['inputs'].get(k) == v for k, v in s['inputs'].items()), 'Score input hash changed')
            identities.append(scope_plan.hardware_identity(record['hardware']))
        require(fitter.selection.select(values, 8, 1/64) == record['entry']['mask'], '1/64 must reproduce original C mask')
    if identities: require(identities[0] == identities[1], 'Both roles must use the same host/GPU order')
    for policy in POLICIES:
        require(plans['variants'][policy] == derive(plans['parent'], plans['scores'], policy), 'Budget mask/order/hash changed')
    return plans
