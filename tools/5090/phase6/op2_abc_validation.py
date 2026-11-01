"""Retain ABC admission and require actual MA8/MA9/MA11 execution in every arm."""
import inspect
from types import ModuleType, SimpleNamespace

from rtx5090_loader import load
import op1_abc_validation as original
import op1_abc_sources as projection
import op2_abc_sources as sources
import op2_abc_protocol as p
import op2_abc_build as optimizers
import ma11_fullstep

base = load('validation')
require, identity = p.require, base.identity
FIELDS = {'ma11_activation_setup', 'ma11_zero_count_state', 'ma11_softmax_state'}
MA11_ARM = 'F4-cache-zcount-csoftmax'


def validate_mechanisms(row, arm, mode, warmup, count, names, worker):
    # The old basic checker deliberately forbids MA11. Feed it only its own
    # fields, then independently require and validate every added mechanism.
    original.validate_mechanisms({key: value for key, value in row.items() if key not in FIELDS},
                                 arm, mode, warmup, count, names, worker)
    require(FIELDS <= row.keys(), 'OP2 is missing MA89/MA11 evidence')
    steps = warmup+count
    ma11_fullstep.validate_activation_setup(row['ma11_activation_setup'], names)
    ma11_fullstep.validate_zero_count_state(row['ma11_zero_count_state'], MA11_ARM, steps, mode, row['rank'])
    ma11_fullstep.validate_softmax_state(row['ma11_softmax_state'], MA11_ARM, steps, mode)


def _shared_validator():
    code = inspect.getsource(base.validate_rows)
    code = code.replace("arm != 'flux-basic'", 'arm in p.NATIVE')
    code = code.replace("arm == 'flux-basic'", 'arm in p.BASIC_ARMS')
    module = ModuleType('rtx5090_op2_shared_validation')
    module.__dict__.update(base.__dict__)
    module.p = SimpleNamespace(**{name: getattr(p, name) for name in dir(p) if not name.startswith('__')},
                               QUANT_SELECTIVE=(), NATIVE_ARM={})
    module.validate_transport = original.validate_transport
    module.validate_mechanisms = validate_mechanisms
    exec(compile(code, __file__, 'exec'), module.__dict__)
    return module.validate_rows


shared_validate_rows = _shared_validator()


def validate_rows(rows, job, plan, worker):
    rows = shared_validate_rows(rows, job, plan, worker)
    arm = job['arm']
    require(worker['selection'] == sources.selection_metadata()
            and worker['config']['op2_implementation'] == sources.implementation(arm)
            and worker['config']['op2_optimization_libraries'] == optimizers.libraries(plan),
            'OP2 worker configuration changed')
    for row in rows:
        require(row['op1_abc_implementation'] == projection.implementation(arm)
                and row['op2_implementation'] == sources.implementation(arm)
                and row['op2_optimization_libraries'] == optimizers.libraries(plan),
                'OP2 implementation or loaded optimization library changed')
        names = set(row['target_modules'])
        require(set(row['selection_checks']) == names, 'Missing OP2 quantization scope')
        wire = 7*sum(chunk['peer_bytes'] for chunk in original.expected_chunks(arm))
        require(all(value['wire_bytes'] == wire and value['bf16_wire_bytes'] == 14680064
                    and value['plan_sha256'] is None and value['mask_sha256'] is None
                    for value in row['selection_checks'].values()), 'OP2 changed ABC quantization')
        if job['mode'] == 'smoke':
            require(set(row['smoke_output_sha256']) == names and all(isinstance(value, str)
                and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                for value in row['smoke_output_sha256'].values()), 'Missing OP2 projection fingerprints')
        else:
            require(row['smoke_output_sha256'] == {} and row['retained_output_checks'] == {},
                    'Smoke audit leaked into OP2 measured execution')
    return rows
