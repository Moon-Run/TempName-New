"""Full-step evidence checks for four distinct, explicitly unsorted implementations."""
import inspect
import math
from types import ModuleType, SimpleNamespace
from rtx5090_loader import load
import flux_baseline_protocol as p
import flux_baseline_sources as sources
from nccl_transport import wire_layout

base = load('validation')
require, identity = p.require, base.identity


def validate_rca_state(row, arm, worker, names):
    original = p.NATIVE[arm]
    count = 2 if original == 'c' else 1
    require(row['ma2c_arm'] == original.upper() and set(row['ma2c_codec_state']) == names
            and set(row['workspace_checks']) == names, 'Missing old A/C state')
    require(all(value == dict(owns_output=original == 'c', owns_configuration=original == 'c', workspaces=count)
                for value in row['ma2c_codec_state'].values()), 'Old A/C ownership changed')
    pointers = []
    for name, slots in row['workspace_checks'].items():
        shape = [2048, 4096, 4096 if name.endswith('.self_attention.linear_proj') else 16384]
        require(len(slots) == count and all(set(slot) == {'shape', 'packet_ptr'} and slot['shape'] == shape
                                           for slot in slots), 'Mask survived or workspace shape changed')
        pointers.extend(slot['packet_ptr'] for slot in slots)
    require(all(type(pointer) is int and pointer > 0 for pointer in pointers)
            and len(set(pointers)) == len(pointers), 'Native receive packets must remain private')


def validate_transport(row, arm, calls, names, layouts):
    if arm in p.NATIVE:
        base.validate_transport(row, p.NATIVE[arm], calls, names, layouts)
    else:
        base.validate_transport(row, 'flux-basic', calls, names, layouts)


def validate_mechanisms(row, arm, mode, warmup, count, names, worker):
    calls = (warmup+count)*4
    if arm in p.NATIVE:
        validate_rca_state(row, arm, worker, names)
    elif arm == 'flux-basic':
        base.validate_mechanisms(row, 'flux-basic', mode, warmup, count, names, worker)
    else:
        cfg = worker['config']
        require(row['codec_identity'] == cfg['codec_identity'] and row['codec_library'] is None
                and row['codec_validation'] is None and cfg['codec_library'] is None
                and row['codec_calls'] == dict.fromkeys(names, calls), 'BF16 arm invoked/misreported a codec')
        require(set(row['bf16_collectives']) == names and all(value == dict(
            quantization='none', collective='reduce_scatter_tensor', dtype='bfloat16', calls=calls,
            encode_calls=0, decode_calls=0, all_to_all_calls=0, owns_output=True,
            logical_remote_bytes=7*256*4096*2) for value in row['bf16_collectives'].values()),
            'BF16 Reduce-Scatter path missing or quantization enabled')
        if mode == 'smoke':
            require(all(value['reference'] == 'unquantized_torch_matmul_nccl_reduce_scatter'
                        and value['quantization'] == 'none' for value in row['forward_checks'].values()),
                    'BF16 output was not checked against the unquantized reference')
    require(not set(row).intersection({'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state',
                                     'ma8_activation_setup', 'ma9_zero_count_state'}), 'Unrequested optimization enabled')


def _shared_validator():
    source = inspect.getsource(base.validate_rows)
    source = source.replace("arm != 'flux-basic'", 'arm in p.NATIVE')
    source = source.replace("arm == 'flux-basic'", 'arm in p.BASIC_ARMS')
    module = ModuleType('rtx5090_flux_four_shared_validation')
    module.__dict__.update(base.__dict__)
    values = {name: getattr(p, name) for name in dir(p) if not name.startswith('__')}
    values.update(QUANT_SELECTIVE=(), NATIVE_ARM={arm: value.upper() for arm, value in p.NATIVE.items()})
    module.p = SimpleNamespace(**values)
    module.validate_transport, module.validate_mechanisms = validate_transport, validate_mechanisms
    exec(compile(source, __file__, 'exec'), module.__dict__)
    return module.validate_rows


shared_validate_rows = _shared_validator()


def validate_rows(rows, job, plan, worker):
    rows = shared_validate_rows(rows, job, plan, worker)
    arm, mode = job['arm'], job['mode']
    require(worker['selection'] == sources.selection_metadata()
            and worker['config']['flux_implementation'] == sources.implementation(arm),
            'Calibration input or implementation changed')
    for row in rows:
        require(row['flux_implementation'] == sources.implementation(arm), 'Wrong actual implementation record')
        names = set(row['target_modules'])
        require(set(row['selection_checks']) == names, 'Missing quantization scope checks')
        for value in row['selection_checks'].values():
            require(value['mask_sha256'] is None and value['plan_sha256'] is None
                    and value['wire_bytes'] == (7*256*4096*2 if arm == 'flux-bf16' else 7*8192*136)
                    and value['bf16_wire_bytes'] == 7*256*4096*2, 'Selective mask or remote byte budget changed')
        if mode == 'smoke':
            require(set(row['smoke_output_sha256']) == names and all(isinstance(value, str)
                and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                for value in row['smoke_output_sha256'].values()), 'Missing output fingerprints')
            checks = row['retained_output_checks']
            require(set(checks) == {'attn', 'mlp'} and all(v['module'] in names and v['outputs'] == 6
                and v['passed'] and v['independent_storage'] and v['unchanged_after_subsequent_call']
                for v in checks.values()), 'Output retention failed')
        else:
            require(row['smoke_output_sha256'] == {} and row['retained_output_checks'] == {},
                    'Smoke audit leaked into timing/profile')
        if arm != 'flux-bf16':
            require('bf16_collectives' not in row, 'Quantized arm silently used BF16 Reduce-Scatter')
    return rows
