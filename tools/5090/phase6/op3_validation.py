"""Require actual OP3 transport, consumer and MA8/MA9/MA11 model execution."""
import inspect
from types import ModuleType, SimpleNamespace

from rtx5090_loader import load
import op3_protocol as p
import op2_abc_build as optimizers
import ma11_fullstep

base = load('validation')
require, identity = p.require, base.identity
FIELDS = {'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state'}
MA11_ARM = 'F4-cache-zcount-csoftmax'
CODEC_PROTOCOL = 'rtx5090-op3-direct-h128-v1'
CHUNKS = [dict(index=0, begin=0, count=8192, raw_groups=0, peer_bytes=8192*136)]


def validate_codec_state(state, arm, calls, rank, fused_calls=0):
    require(arm in p.ALL_ARMS and type(calls) is int and calls > 0
            and type(rank) is int and 0 <= rank < 8 and type(fused_calls) is int
            and 0 <= fused_calls <= calls and (p.fused(arm) or fused_calls == 0),
            'Invalid OP3 arm/call/rank/consumer identity')
    require(state['protocol'] == CODEC_PROTOCOL and state['arm'] == arm and state['chunks'] == CHUNKS
            and state['transport'] == 'NCCL-byte-AllToAll' and state['arrival_trigger'] == 'whole-exchange'
            and state['selection'] == 'all-remote' and state['wire_bytes_per_rank'] == 7798784
            and state['bf16_wire_bytes_per_rank'] == 14680064
            and state['packet_layout'] == 'H128-AoS-136' and state['fixed_source_order'] == 'rank+1-through-rank',
            'OP3 packet layout/selection/source order changed')
    counters = dict(calls=calls, encode_calls=calls, exchange_calls=calls, decode_calls=calls,
        wait_calls=calls, join_calls=0, fused_consumer_calls=fused_calls, projection_calls=calls-fused_calls)
    require(all(type(state[key]) is int and state[key] == value for key, value in counters.items()),
            'OP3 stage or fused/projection call counts are stale or incomplete')
    require(state['encoder'] == ('upstream-cta128' if arm == 'op3-r' else 'warp-owned-four-groups')
            and state['decoder'] == ('r-packed-source-prefetch' if p.fused(arm)
                                    else 'basic-ring-bf16-h128-direct-v1')
            and state['consumer_fusion'] is (p.fused(arm))
            and state['independent_output'] is True and state['calibration_used'] is False
            and state['original_peer_overlap_preserved'] is False and state['failed'] is False
            and state['pending_exchange'] is False, 'Wrong/failed OP3 codec or output ownership')
    splits = [[0 if peer == rank else 8192*136 for peer in range(8)]]
    streams = [state[key] for key in ('caller_stream', 'comm_stream', 'decode_stream')]
    require(state['device'] == rank and all(type(value) is int and value >= 0 for value in streams)
            and len(set(streams)) == 1 and state['send_split_bytes'] == state['receive_split_bytes'] == splits,
            'OP3 stream/device/NCCL split contract changed')
    pointers = state['tx_ptrs']+state['rx_ptrs']
    require(len(state['tx_ptrs']) == len(state['rx_ptrs']) == 1
            and all(type(value) is int and value > 0 and value % 8 == 0 for value in pointers)
            and len(set(pointers)) == 2, 'OP3 packet buffers are missing, unaligned or aliased')


def validate_transport(row, arm, calls, names, layouts):
    base.validate_transport(row, 'flux-basic', calls, names, layouts)
    states = row['op3_transport']
    require(set(states) == names,
            f'Missing OP3 transport modules: expected {len(names)}, observed {len(states)}')
    pointers = []
    for state in states.values():
        validate_codec_state(state, arm, calls, row['rank'], fused_calls=calls if p.fused(arm) else 0)
        pointers.extend(state['tx_ptrs']+state['rx_ptrs'])
    require(len(set(pointers)) == len(pointers), 'OP3 packet storage aliases another module')
    consumer = row['op3_consumer_state']
    if p.fused(arm):
        require(set(consumer) == names, 'R consumer fusion is missing target modules')
        bias_ids = []
        for name, value in consumer.items():
            require(type(value['projection_calls']) is int and type(value['bda_calls']) is int
                    and value['projection_calls'] == value['bda_calls'] == calls
                    and type(value['bias_parameter_id']) is int and value['bias_parameter_id'] > 0
                    and value['native'] == states[name], 'R consumer/BDA/native state disagree')
            bias_ids.append(value['bias_parameter_id'])
        require(len(set(bias_ids)) == len(bias_ids), 'R target modules unexpectedly share a bias parameter')
    else:
        require(consumer == {}, 'Encoder candidate unexpectedly enabled consumer fusion')


def validate_mechanisms(row, arm, mode, warmup, count, names, worker):
    # Validate the original standalone codec and model contracts, then separately
    # require the added mechanisms. No global legacy validator is modified.
    base.validate_mechanisms({key: value for key, value in row.items() if key not in FIELDS},
                             'flux-basic', mode, warmup, count, names, worker)
    require(FIELDS <= row.keys(), 'OP3 is missing MA89/MA11 execution evidence')
    steps = warmup+count
    ma11_fullstep.validate_activation_setup(row['ma11_activation_setup'], names)
    ma11_fullstep.validate_zero_count_state(row['ma11_zero_count_state'], MA11_ARM, steps, mode, row['rank'])
    ma11_fullstep.validate_softmax_state(row['ma11_softmax_state'], MA11_ARM, steps, mode)


def _shared_validator():
    code = inspect.getsource(base.validate_rows)
    require(code.count("arm != 'flux-basic'") == 1 and code.count("arm == 'flux-basic'") == 1,
            'Shared full-training admission source changed')
    code = code.replace("arm != 'flux-basic'", 'arm in p.NATIVE')
    code = code.replace("arm == 'flux-basic'", 'arm in p.BASIC_ARMS')
    module = ModuleType('rtx5090_op3_shared_validation')
    module.__dict__.update(base.__dict__)
    values = {name: getattr(p, name) for name in dir(p) if not name.startswith('__')}
    values.update(QUANT_SELECTIVE=(), NATIVE_ARM={})
    module.p = SimpleNamespace(**values)
    module.validate_transport, module.validate_mechanisms = validate_transport, validate_mechanisms
    exec(compile(code, __file__, 'exec'), module.__dict__)
    return module.validate_rows


shared_validate_rows = _shared_validator()


def validate_rows(rows, job, plan, worker):
    import op3_sources as sources
    rows = shared_validate_rows(rows, job, plan, worker)
    arm = job['arm']
    expected = sources.implementation(arm)
    require(worker['selection'] == sources.selection_metadata()
            and worker['definition'] == sources.definition(arm)
            and worker['config']['op3_implementation'] == expected
            and worker['config']['op3_optimization_stack'] == sources.STACK
            and worker['config']['op3_optimization_libraries'] == optimizers.libraries(plan),
            'OP3 worker configuration changed')
    for row in rows:
        require(row['op3_implementation'] == expected and row['op3_optimization_stack'] == sources.STACK
                and row['op3_optimization_libraries'] == optimizers.libraries(plan),
                'OP3 implementation, shared stack or optimization library changed')
        names = set(row['target_modules'])
        require(set(row['selection_checks']) == names and all(value['wire_bytes'] == 7798784
            and value['bf16_wire_bytes'] == 14680064 and value['plan_sha256'] is None
            and value['mask_sha256'] is None for value in row['selection_checks'].values()),
            'OP3 all-remote quantization scope or wire budget changed')
        if job['mode'] == 'smoke':
            require(set(row['smoke_output_sha256']) == names and all(isinstance(value, str)
                and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                for value in row['smoke_output_sha256'].values()), 'Missing OP3 projection fingerprints')
            checks = row['retained_output_checks']
            require(set(checks) == {'attn', 'mlp'} and all(value['module'] in names and value['outputs'] == 6
                and value['passed'] is True and value['independent_storage'] is True
                and value['unchanged_after_subsequent_call'] is True for value in checks.values()),
                'OP3 output retention or ownership failed')
        else:
            require(row['smoke_output_sha256'] == {} and row['retained_output_checks'] == {},
                    'Smoke audit leaked into OP3 measured execution')
    return rows
