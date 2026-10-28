"""Separate role/full-shape plans; Attention can only come from new BF16 calibration."""
import copy
from pathlib import Path

from ablation_plan import FREEZE, FREEZE_ID, parent_plan, resolve_input, mask_sha
from common import read, require, sha, validate_entry
from scope_protocol import PROTOCOL, SHAPES, digest


def model_config():
    return copy.deepcopy(read(FREEZE/'manifest.json')['config'])


def validate_hardware(hardware):
    devices = hardware['devices']
    require(hardware['host'] and len(devices) == 8 and len({d['uuid'] for d in devices}) == 8 and
            all('H100' in d['name'] and d['sm_count'] == 132 and d['capability'] == [9, 0] for d in devices),
            'Single-node eight full 132-SM H100s required')


def hardware_identity(hardware):
    return {k: hardware[k] for k in ('host', 'devices')}


def role_record(role, entry, hardware, inputs, origin, test_only=False):
    return dict(protocol=PROTOCOL, role=role, base='interleaved', config=model_config(),
                entry=entry, entry_sha256=digest(entry), mask_sha256=mask_sha(entry['mask']),
                hardware=hardware, inputs=inputs, origin=origin, parent_freeze=FREEZE_ID,
                training_passes=[0, 1], heldout_passes=[2], gpu_calibrated=not test_only, test_only=test_only)


def validate(record, role, allow_test=False, verify_evidence=True):
    require(record['protocol'] == PROTOCOL and record['role'] == role and record['base'] == 'interleaved', 'Wrong role protocol')
    require(record['config'] == model_config() and record['parent_freeze'] == FREEZE_ID, 'Model/source freeze changed')
    require(record['training_passes'] == [0, 1] and record['heldout_passes'] == [2], 'Calibration split changed')
    if record.get('test_only'):
        require(allow_test and not record['gpu_calibrated'] and record['origin']['kind'] == 'cpu_fixture',
                'CPU synthetic plan is permanently non-executable')
    else:
        require(record['gpu_calibrated'] is True and record['inputs'], 'Real H100 BF16 calibration required')
        require(record['origin']['kind'] in (('scope_bf16_calibration',) if role == 'attn' else
                ('scope_bf16_calibration', 'frozen_mlp_calibration')), 'Attention cannot reuse an MLP plan')
        validate_hardware(record['hardware'])
        if verify_evidence:
            for name, value in record['inputs'].items():
                resolve_input(name, value, registered_inputs=record['inputs'])
    entry = record['entry']
    require(entry['shape'] == SHAPES[role] and
            [entry[k] for k in ('M', 'N', 'K_global')] == SHAPES[role], 'Full shape/role mismatch')
    validate_entry(entry, record['config'], SHAPES[role])
    require(record['entry_sha256'] == digest(entry) and record['mask_sha256'] == mask_sha(entry['mask']), 'Plan hash changed')
    return record


def frozen_mlp():
    parent, path, _ = parent_plan()
    return role_record('mlp', copy.deepcopy(parent['policies']['interleaved'][0]), parent['hardware'],
                       dict(parent['inputs'], **{str(path): sha(path)}),
                       dict(kind='frozen_mlp_calibration', parent=str(path), parent_sha256=sha(path)))


def cpu_fixture(role):
    """Independent artificial shapes, never copied/relabeled from an MLP measurement."""
    m, n, k = SHAPES[role]; tiles = 512
    base = [[((i % 8+r) % 8)*64+i//8 for i in range(tiles)] for r in range(8)]
    maps = [list(range(tiles)) for _ in range(8)]
    # Distinct legal permutations make a missing K dispatch observable in CPU/CUDA checks.
    for row in maps:
        a, b = (1, 9) if role == 'attn' else (2, 18)
        row[a], row[b] = row[b], row[a]
    mask = [[int(t == ((r+1) % 8)*64+(1 if role == 'attn' else 2)) for t in range(tiles)] for r in range(8)]
    entry = dict(shape=[m, n, k], M=m, N=n, K_global=k, K_local=k//8, tiles=tiles,
                 maps=maps, base_coords=base, coords=[[base[r][i] for i in maps[r]] for r in range(8)],
                 mask=mask, selected_per_source=list(map(sum, mask)))
    return role_record(role, entry, {}, {}, dict(kind='cpu_fixture', status='pending_real_H100_calibration'), True)


def bundle(attn=None, mlp=None, source_only=False):
    require(attn is not None or source_only, 'Attention needs its own real H100 BF16 calibration; use --source-only for CPU rehearsal')
    records = {'attn': read(attn) if attn else cpu_fixture('attn'),
               'mlp': read(mlp) if mlp else frozen_mlp()}
    for role, record in records.items():
        validate(record, role, allow_test=source_only)
    real = [r for r in records.values() if not r['test_only']]
    if len(real) == 2:
        require(hardware_identity(real[0]['hardware']) == hardware_identity(real[1]['hardware']),
                'Attention/MLP hardware/order differs; recalibrate both on this allocation')
    return dict(protocol=PROTOCOL, roles=records, source_only=source_only,
                gpu_calibrated=not source_only and all(r['gpu_calibrated'] for r in records.values()),
                status='pending_gpu_validation' if not source_only else 'cpu_rehearsal_pending_attention_calibration')
