"""Reuse frozen training checks while replacing only the codec contract."""
import inspect
from types import ModuleType
from rtx5090_loader import load
import open_compress_protocol as p

base = load('validation')
identity = base.identity


def mechanisms(row, arm, mode, warmup, count, names, worker):
    import open_compress_sources as sources
    calls = (warmup+count)*4
    cfg = worker['config']
    p.require(worker['selection'] == sources.selection_metadata()
              and cfg['open_compress_implementation'] == sources.implementation(arm)
              and row['open_compress_implementation'] == sources.implementation(arm)
              and row['codec_identity'] == cfg['codec_identity']
              and row['codec_library'] == cfg['codec_library']
              and row['codec_validation'] is None
              and row['codec_calls'] == dict.fromkeys(names, calls), 'Codec identity/call coverage changed')
    p.require(set(row['compression_states']) == names, 'Missing per-module compression state')
    for state in row['compression_states'].values():
        p.require(state['calls'] == calls and state['owns_output'] is True, 'Wrong codec ownership/count')
        if arm == 'flux-bf16':
            p.require(state['quantization'] == 'none' and state['collective'] == 'reduce_scatter_tensor'
                      and all(state[k] == 0 for k in ('encode_calls', 'decode_calls', 'all_to_all_calls')),
                      'BF16 control invoked compression')
        else:
            p.require(all(state[k] == calls for k in ('encode_calls', 'decode_calls', 'all_to_all_calls', 'wait_calls'))
                      and state['logical_remote_bytes'] == 7*p.PACKET_BYTES,
                      'INT8 did not execute every encode/exchange/wait/decode')
            p.require(state['protocol'] == 'open-block-int8-soa128-v1' and state['arm'] == arm
                      and state['source_order'] == list(range(8))
                      and state['packet_layout'] == 'peer-soa-q8-scale32' and state['dtype'] == 'uint8'
                      and state['collective'] == 'all_to_all_single' and state['quantization'] == 'all-remote'
                      and state['caller_stream'] == state['current_stream']
                      and state['finite_input_checks'] == (calls if mode == 'smoke' else 0)
                      and all(state[k] is False for k in ('arrival','selective','calibration_used',
                                  'peer_overlap_preserved','failed','pending_exchange')),
                      'INT8 format/lifecycle or mechanism changed')
            splits = [0 if i == row['rank'] else p.PACKET_BYTES for i in range(8)]
            p.require(state['send_split_bytes'] == state['receive_split_bytes'] == splits,
                      'Wrong peer packet split or local contribution quantized')
    p.require(not set(row).intersection({'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state',
                                       'ma8_activation_setup', 'ma9_zero_count_state'}), 'Unrequested mechanism enabled')
    p.require(set(row['selection_checks']) == names, 'Missing projection policy checks')
    for v in row['selection_checks'].values():
        p.require(v['mask_sha256'] is None and v['plan_sha256'] is None
                  and v['wire_bytes'] == (7*256*4096*2 if arm == 'flux-bf16' else 7*p.PACKET_BYTES)
                  and v['bf16_wire_bytes'] == 7*256*4096*2, 'Selective mask or wrong wire accounting')
    if mode == 'smoke':
        p.require(set(row['smoke_output_sha256']) == names, 'Missing output fingerprints')
        checks = row['retained_output_checks']
        p.require(set(checks) == {'attn', 'mlp'} and all(v['outputs'] == 6 and v['passed'] is True
                  and v['independent_storage'] is True and v['unchanged_after_subsequent_call'] is True
                  for v in checks.values()), 'Output overwritten by workspace reuse')
    else:
        p.require(row['smoke_output_sha256'] == row['retained_output_checks'] == {}, 'Smoke audit leaked into timing')


def transport(row, arm, calls, names, layouts):
    p.require(row['transport'] == p.TRANSPORT and row['rtx5090_transport'] == {}
              and row['native_workspace_bytes'] == 0, 'TACO transport/native workspace enabled')


def _validator():
    source = inspect.getsource(base.validate_rows)
    source = source.replace("arm != 'flux-basic'", 'False')
    source = source.replace("arm == 'flux-basic'", 'True')
    module = ModuleType('rtx5090_open_compress_frozen_validation')
    module.__dict__.update(base.__dict__)
    module.p, module.validate_mechanisms, module.validate_transport = p, mechanisms, transport
    exec(compile(source, __file__, 'exec'), module.__dict__)
    return module.validate_rows


validate_rows = _validator()
