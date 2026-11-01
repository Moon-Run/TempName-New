"""Strict full-model and distributed transport evidence for the new ABC protocol."""
import inspect
from types import ModuleType, SimpleNamespace
from rtx5090_loader import load
import op1_abc_protocol as p
import flux_baseline_validation as improved

base = load('validation')
require, identity = p.require, base.identity


def expected_chunks(arm, total_groups=8192):
    require(arm in p.BASIC_ARMS, 'Unknown ABC codec arm')
    number = 1 if arm == p.BASIC_ARMS[0] else 4
    count = total_groups // number
    first_quantized = total_groups-total_groups//64 if arm == p.BASIC_ARMS[2] else 0
    return [dict(index=i, begin=i*count, count=count,
                 raw_groups=min(count, max(0, first_quantized-i*count)),
                 peer_bytes=min(count, max(0, first_quantized-i*count))*120+count*136)
            for i in range(number)]


def validate_codec_state(state, arm, calls, rank):
    chunks = expected_chunks(arm)
    arrival, selective = arm != p.BASIC_ARMS[0], arm == p.BASIC_ARMS[2]
    require(type(calls) is int and calls > 0 and type(rank) is int and 0 <= rank < 8,
            'Invalid ABC call/rank identity')
    require(state['arm'] == arm and state['calls'] == calls and state['chunks'] == chunks
            and state['transport'] == 'NCCL-byte-AllToAll'
            and state['arrival_trigger'] == ('NCCL-work-wait-then-CUDA-event' if arrival else 'whole-exchange')
            and state['selection'] == ('last-1/64-H128-groups' if selective else 'all-remote')
            and state['wire_bytes_per_rank'] == 7*sum(chunk['peer_bytes'] for chunk in chunks)
            and state['bf16_wire_bytes_per_rank'] == 7*256*4096*2,
            'ABC chunk layout/arrival/selection byte budget changed')
    require(all(state[key] == calls*len(chunks) for key in ('encode_calls', 'exchange_calls', 'wait_calls', 'decode_calls'))
            and state['join_calls'] == (calls if arrival else 0)
            and state['independent_output'] is True
            and state['fixed_source_order'] == 'rank+1-through-rank'
            and state['packet_layout'] == 'H128-AoS-136-or-BF16-256'
            and state['calibration_used'] is False and state['original_peer_overlap_preserved'] is False
            and state['failed'] is False, 'Missing/failed ABC stages or substituted policy')
    splits = [[0 if peer == rank else chunk['peer_bytes'] for peer in range(8)] for chunk in chunks]
    streams = [state[key] for key in ('caller_stream', 'comm_stream', 'decode_stream')]
    require(state['device'] == rank and state['pending_exchange'] is False
            and all(type(value) is int and value >= 0 for value in streams)
            and len(set(streams)) == (3 if arrival else 1)
            and state['send_split_bytes'] == state['receive_split_bytes'] == splits,
            'ABC stream/dependency/split evidence changed')
    pointers = state['tx_ptrs']+state['rx_ptrs']
    require(len(state['tx_ptrs']) == len(state['rx_ptrs']) == len(chunks)
            and all(type(value) is int and value > 0 for value in pointers)
            and len(set(pointers)) == 2*len(chunks), 'ABC packet buffers alias or are missing')


def validate_transport(row, arm, calls, names, layouts):
    if arm in p.NATIVE:
        improved.validate_transport(row, arm, calls, names, layouts)
        require(row.get('op1_abc_transport', {}) == {}, 'Improved comparator used ABC transport')
        return
    base.validate_transport(row, 'flux-basic', calls, names, layouts)
    states = row['op1_abc_transport']
    require(set(states) == names,
            f'Missing per-module ABC transport evidence: expected {len(names)} modules, '
            f'observed {len(states)}; missing={sorted(names-set(states))}, '
            f'unexpected={sorted(set(states)-names)}')
    pointers = []
    for state in states.values():
        validate_codec_state(state, arm, calls, row['rank'])
        pointers.extend(state['tx_ptrs']+state['rx_ptrs'])
    require(len(set(pointers)) == len(pointers), 'Per-module ABC packet storage aliases another module')


def validate_mechanisms(row, arm, mode, warmup, count, names, worker):
    if arm in p.NATIVE:
        improved.validate_rca_state(row, arm, worker, names)
    else:
        # This still independently verifies the unchanged upstream H128 codec
        # ABI exports. The new direct/chunk APIs are covered by native probes.
        base.validate_mechanisms(row, 'flux-basic', mode, warmup, count, names, worker)
    require(not set(row).intersection({'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state',
        'ma8_activation_setup', 'ma9_zero_count_state'}), 'Unrequested model optimization enabled')


def _shared_validator():
    source = inspect.getsource(base.validate_rows)
    source = source.replace("arm != 'flux-basic'", 'arm in p.NATIVE')
    source = source.replace("arm == 'flux-basic'", 'arm in p.BASIC_ARMS')
    module = ModuleType('rtx5090_op1_abc_shared_validation')
    module.__dict__.update(base.__dict__)
    values = {name: getattr(p, name) for name in dir(p) if not name.startswith('__')}
    values.update(QUANT_SELECTIVE=(), NATIVE_ARM={'flux-improved': 'C'})
    module.p = SimpleNamespace(**values)
    module.validate_transport, module.validate_mechanisms = validate_transport, validate_mechanisms
    exec(compile(source, __file__, 'exec'), module.__dict__)
    return module


shared = _shared_validator()


def validate_rows(rows, job, plan, worker):
    import op1_abc_sources as sources
    rows = shared.validate_rows(rows, job, plan, worker)
    arm, mode = job['arm'], job['mode']
    expected = sources.implementation(arm)
    require(worker['config']['op1_abc_implementation'] == expected
            and worker['selection'] == sources.selection_metadata(),
            'ABC worker implementation changed')
    for row in rows:
        require(row['op1_abc_implementation'] == expected, 'Missing actual ABC implementation record')
        names = set(row['target_modules'])
        require(set(row['selection_checks']) == names, 'Missing projection quantization scope')
        wire = 7*sum(chunk['peer_bytes'] for chunk in expected_chunks(arm)) if arm in p.BASIC_ARMS else 7*8192*136
        for value in row['selection_checks'].values():
            require(value['wire_bytes'] == wire and value['bf16_wire_bytes'] == 7*256*4096*2
                    and value['plan_sha256'] is None and value['mask_sha256'] is None,
                    'Selection or wire budget changed')
        if mode == 'smoke':
            require(set(row['smoke_output_sha256']) == names and all(isinstance(value, str)
                and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                for value in row['smoke_output_sha256'].values()), 'Missing output fingerprints')
            checks = row['retained_output_checks']
            require(set(checks) == {'attn', 'mlp'} and all(value['module'] in names and value['outputs'] == 6
                and value['passed'] and value['independent_storage'] and value['unchanged_after_subsequent_call']
                for value in checks.values()), 'Output ownership/retention failed')
        else:
            require(row['smoke_output_sha256'] == {} and row['retained_output_checks'] == {},
                    'Smoke ownership checks leaked into measured execution')
    return rows
