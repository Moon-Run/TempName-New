"""OP1 evidence checks, sharing unchanged model/numerical checks privately."""
from pathlib import Path
from types import ModuleType, SimpleNamespace

from rtx5090_loader import load
import op1_a_protocol as p
from op1_a_sources import BASIC_ARMS, implementation

base = load('validation')
require = p.require
identity = base.identity


def equivalent(rows, other, smoke=False):
    base.equivalent(rows, other, smoke)
    if smoke:
        require([row['smoke_output_sha256'] for row in rows] ==
                [row['smoke_output_sha256'] for row in other],
                'CTA ordering changed a basic projection output bitwise')


def validate_rca_state(row, arm, worker, names):
    require(arm == 'a' and row['ma2c_arm'] == 'A', 'Wrong OP1 native arm')
    states, workspaces = row['ma2c_codec_state'], row['workspace_checks']
    require(set(states) == names == set(workspaces) and all(
        value == dict(owns_output=True, owns_configuration=True, workspaces=1)
        for value in states.values()), 'OP1 A must own its configuration and independent output')
    pointers = []
    for name, slots in workspaces.items():
        k = 4096 if name.endswith('.self_attention.linear_proj') else 16384
        require(len(slots) == 1 and set(slots[0]) == {'shape', 'packet_ptr'}
                and slots[0]['shape'] == [2048, 4096, k], 'Wrong direct packet workspace')
        pointers.append(slots[0]['packet_ptr'])
    require(all(type(pointer) is int and pointer > 0 for pointer in pointers)
            and len(set(pointers)) == len(pointers), 'Receive packets must remain private per module')


def validate_transport(row, arm, calls, names, layouts):
    require(row['transport'] == p.TRANSPORT, 'Wrong OP1 outer transport')
    if arm in BASIC_ARMS:
        require(row['rtx5090_transport'] == {} and row['native_workspace_bytes'] == 0,
                'Basic codec cannot claim the fused A producer')
        return
    import op1_a_native as native
    records = row['rtx5090_transport']
    require(set(records) == names, 'Missing OP1 per-module transport evidence')
    workspace_bytes, scratch_sets, scratch_owners = 0, {}, {}
    packet = 256*4096//128*136
    splits = [0 if peer == row['rank'] else packet for peer in range(8)]
    for name, slot in records.items():
        role = 'attn' if name.endswith('.self_attention.linear_proj') else 'mlp'
        require('slots' not in slot and slot['protocol'] == native.PROTOCOL
                and slot['native_api'] == 'rtx5090_op1_a_direct_v1'
                and slot['peer_memory'] is False and slot['failed'] is False
                and slot['pending_exchange'] is False and slot['closed'] is False
                and slot['native_phase'] == 0 and slot['device'] == row['rank']
                and type(slot['stream']) is int and slot['stream'] >= 0
                and slot['synchronization'] == 'fixed_stream_producer_NCCL_work_wait_decode_reuse',
                'Incomplete direct producer/exchange/decode lifecycle')
        require(all(slot[key] == calls for key in
                    ('calls', 'producer_calls', 'exchange_calls', 'wait_calls', 'decode_calls', 'initializations'))
                and all(slot[key] == 0 for key in
                        ('pack_calls', 'unpack_calls', 'consumer_calls', 'reconfigure_calls'))
                and slot['configuration_scope'] == 'instance'
                and slot['packet_layout'] == 'compact-peer-soa-h128-v1'
                and slot['output_ownership'] == 'independent'
                and slot['gemm_submission_cache'] is False,
                'Direct A contains missing stages or an unintended optimization')
        require(slot['send_split_bytes'] == splits == slot['receive_split_bytes']
                and slot['send_bytes'] == slot['receive_bytes'] == 7*packet,
                'Direct packet byte budget/self exclusion changed')
        allocated = slot['native_workspace_bytes']
        require(type(allocated) is int and allocated >= 0, 'Missing native workspace accounting')
        workspace_bytes += allocated
        expected_native = dict(protocol_version=2, initializations=calls,
            producer_calls=calls, decode_calls=calls, bias_residual_calls=0,
            phase=0, failed=False, bound_k=512 if role == 'attn' else 2048,
            reconfigure_calls=0, workspace_bytes=allocated)
        require(slot['native'] == expected_native, 'Direct native state disagrees with transport')
        key = (role, slot['device'], slot['stream'])
        pointers = slot['shared_scratch_ptrs']
        require(slot['scratch_policy'] == native.SCRATCH_POLICY
                and set(pointers) == {'tx_wire', 'rx_bf16'}
                and all(type(pointer) is int and pointer > 0 for pointer in pointers.values())
                and len(set(pointers.values())) == 2
                and slot['shared_scratch_bytes'] == 7*packet + 8*256*4096*2,
                'Wrong direct packet scratch layout or memory accounting')
        require(key not in scratch_sets or scratch_sets[key] == pointers,
                'Same-role direct scratch was duplicated across modules')
        scratch_sets[key] = pointers
        for kind, pointer in pointers.items():
            owner = (key, kind)
            require(pointer not in scratch_owners or scratch_owners[pointer] == owner,
                    'Scratch aliases a different role/stream/buffer')
            scratch_owners[pointer] = owner
    require(row['native_workspace_bytes'] == workspace_bytes, 'Native workspace total changed')


def _shared_validator():
    # Execute an isolated copy: historical controller/module globals remain intact.
    # Only basic arm membership changes; every model, gradient, timing, hardware,
    # codec, library hash and no-fallback check remains the baseline check.
    source = Path(base.__file__).read_text()
    source = source.replace("arm == 'flux-basic'", 'arm in p.BASIC_ARMS')
    source = source.replace("arm != 'flux-basic'", 'arm not in p.BASIC_ARMS')
    module = ModuleType('rtx5090_op1_shared_validation')
    module.__file__ = base.__file__
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    values = {name: getattr(load('protocol'), name) for name in dir(load('protocol')) if not name.startswith('__')}
    values.update({name: getattr(p, name) for name in dir(p) if not name.startswith('__')})
    values.update(BASIC_ARMS=BASIC_ARMS, CR3_FAMILY=(), QUANT_SELECTIVE=(), NATIVE_ARM={'a': 'A'})
    module.p = SimpleNamespace(**values)
    module.validate_rca_state = validate_rca_state
    module.validate_transport = validate_transport
    return module


shared = _shared_validator()


def validate_rows(rows, job, plan, worker):
    rows = shared.validate_rows(rows, job, plan, worker)
    expected = implementation(job['arm'])
    require(worker['config']['op1_implementation'] == expected,
            'OP1 worker implementation configuration changed')
    for row in rows:
        require(row['op1_implementation'] == expected, 'Missing/wrong OP1 implementation evidence')
        if job['mode'] == 'smoke':
            names = set(row['target_modules'])
            require(set(row['smoke_output_sha256']) == names
                    and all(isinstance(value, str) and len(value) == 64
                            and all(c in '0123456789abcdef' for c in value)
                            for value in row['smoke_output_sha256'].values()),
                    'Every target projection needs a smoke output fingerprint')
            checks = row['retained_output_checks']
            require(set(checks) == {'attn', 'mlp'} and all(
                value['module'] in names and value['outputs'] == 6
                and value['independent_storage'] is True and value['passed'] is True
                and value['unchanged_after_subsequent_call'] is True
                for value in checks.values()), 'Independent outputs did not survive workspace reuse')
        else:
            require(row['smoke_output_sha256'] == {} and row['retained_output_checks'] == {},
                    'OP1 ownership/hash audits leaked into the measured path')
    return rows
