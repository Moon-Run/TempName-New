"""Three full Megatron workers sharing frozen training/backward/optimizer ASTs."""
import hashlib
from pathlib import Path
import shutil

from rtx5090_loader import HERE, H100, REPO, load
from common import read, write, require, sha, replace_once

PROTOCOL = 'rtx5090-open-compress-int8-three-arm-v1'
ARMS = ('flux-bf16', 'flux-int8', 'flux-int8-interleaved')
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
base = load('sources')
p = load('protocol')


def implementation(arm):
    require(arm in ARMS, 'Unknown open compression arm')
    return dict(protocol=PROTOCOL, producer='flux-gemmonly-sm89-v2-sm120', gemm_schedule='StreamkSK',
        quantization='none' if arm == 'flux-bf16' else 'all_remote_block_int8_128_fp32_scale',
        interleaved=arm.endswith('-interleaved'), arrival=False, selective=False, calibration_used=False,
        cta_order='rank-rotated-destination-interleaved' if arm.endswith('-interleaved') else 'original-Flux-GemmOnly',
        may_change_streamk_reduction_assignment=arm.endswith('-interleaved'),
        collective='NCCL-BF16-ReduceScatter' if arm == 'flux-bf16' else 'NCCL-byte-AllToAll',
        receiver='NCCL' if arm == 'flux-bf16' else 'block-int8-decode-source-ordered-bf16-fold',
        owns_output=True, owns_configuration=True, taco_runtime=False, original_peer_overlap_preserved=False)


def definition(arm):
    return dict(policy=arm, family='flux_tensor', scope='attn_mlp', roles=list(SHAPES),
        implementation='rtx5090-open-compress-'+arm, workspace='tensor', output_clone=False,
        base='original', order='interleaved' if arm.endswith('-interleaved') else 'original',
        arrival=False, interleaved=arm.endswith('-interleaved'),
        quantization='none' if arm == 'flux-bf16' else 'all_remote', experiment=implementation(arm))


def selection_metadata():
    return dict(protocol=PROTOCOL, calibration_used=False,
                roles={role: dict(role=role, shape=shape, K_local=shape[2]//8) for role, shape in SHAPES.items()})


def routing_source():
    prefix = f'PROTOCOL = {PROTOCOL!r}\nPOLICIES = {ARMS!r}\nSHAPES = {SHAPES!r}\n'
    definitions = {arm: definition(arm) for arm in ARMS}
    return prefix + f'DEFINITIONS = {definitions!r}\n' + '''
def require(ok, message):
    if not ok: raise ValueError(message)

def definition(policy):
    require(policy in POLICIES, 'Unknown compression policy')
    return DEFINITIONS[policy]

def is_native(policy):
    definition(policy)
    return False

def is_flux_tensor(policy):
    definition(policy)
    return True

is_tensor = is_flux_tensor

def is_fused_v2(policy):
    definition(policy)
    return False

def is_original(policy):
    definition(policy)
    return True

def needs_mapping(policy):
    definition(policy)
    return policy.endswith('-interleaved')

def module_role(name):
    for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
        if name.endswith(suffix): return role
    raise ValueError('Unexpected target projection: '+name)

def enabled(policy, name):
    definition(policy)
    module_role(name)
    return True

def route(policy, name):
    enabled(policy, name)
    return 'flux_tensor'

def codec_placement(policy):
    definition(policy)
    return 'tensor'

def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Expected 32 layers')
    counts, routes = dict.fromkeys(SHAPES, 0), {}
    for name, shape in observed.items():
        role = module_role(name)
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role]
                and shape['K_local'] == SHAPES[role][2]//8, 'Wrong projection shape')
        counts[role] += 1
        routes[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 projections required')
    return routes

def role_plans(plan, policy):
    definition(policy)
    require(plan['protocol'] == PROTOCOL and plan['calibration_used'] is False
            and set(plan) == {'protocol', 'calibration_used', 'roles'}, 'Calibration forbidden')
    return plan['roles']

def selection_entry(*args):
    raise ValueError('This experiment forbids selection masks and arrival tables')
'''


def reference_source(arm='flux-int8'):
    if arm == 'flux-bf16':
        return '# No compression oracle in the BF16 arm.\n'
    return '"""Oracle implementation is independent Torch arithmetic in the hashed codec module."""\nfrom megatron_taco import reference_roundtrip\n'


def codec_source(arm):
    require(arm in ARMS, 'Unknown compression arm')
    if arm == 'flux-bf16':
        return (HERE/'flux_baseline_bf16.py').read_text()
    return (HERE/'open_compress_codec.py').read_text()


def codec_identity(arm):
    return dict(implementation='none-nccl-bf16' if arm == 'flux-bf16' else 'open-block-int8-128-fp32-scale-v1',
        source_sha256=hashlib.sha256(codec_source(arm).encode()).hexdigest(),
        reference_sha256=hashlib.sha256(reference_source(arm).encode()).hexdigest(),
        cuda_source_sha256=None if arm == 'flux-bf16' else sha(HERE/'open_compress_codec.cu'),
        taco_runtime=False)


def taco_source(arm):
    """Legacy adapter class name, with no TACO imports, transforms or oracle."""
    require(arm in ARMS, 'Unknown compression arm')
    return '''from pathlib import Path
import json
import torch
import torch.distributed as dist

class TacoForward:
    def __init__(self, group, m, n, k_local, policy, role):
        from flux_tensor_taco import FluxTensorTaco
        self.role, self.plan_sha256 = role, None
        self.codec = FluxTensorTaco(group, m, n, k_local)
        self.owns_output = True
        self.workspace_info = self.codec.workspace_info

    def forward(self, input_, weight, reduce_scatter_option=None):
        return self.codec.forward(input_, weight)

@torch.no_grad()
def check_taco_output(partial, actual, group, mask=None):
    if mask is not None:
        raise ValueError('No selective quantization in this experiment')
    arm = json.loads((Path(__file__).parent/'config.json').read_text())['arm']
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    bf16 = torch.empty_like(actual)
    dist.reduce_scatter_tensor(bf16, partial.contiguous(), group=group)
    if arm == 'flux-bf16':
        coded = bf16
    else:
        from megatron_taco import reference_roundtrip
        parts = [torch.empty_like(partial) for _ in range(world)]
        dist.all_gather(parts, partial, group=group)
        rows = partial.shape[0]//world
        coded = torch.zeros_like(actual)
        for source in range(world):
            value = parts[source][rank*rows:(rank+1)*rows]
            if source != rank:
                value = reference_roundtrip(value)
            coded = (coded+value).to(torch.bfloat16)
    denominator = max(bf16.float().norm().item(), 1e-20)
    error = dict(max_abs=(actual.float()-bf16.float()).abs().max().item(),
        relative_l2=(actual.float()-bf16.float()).norm().item()/denominator,
        protocol_relative_l2=(actual.float()-coded.float()).norm().item()/denominator,
        budget_vs_bf16=.05, budget_vs_codec=.01,
        reference='torch_bf16_matmul_nccl_rs_and_independent_block_int8_source_fold',
        quantization='none' if arm == 'flux-bf16' else 'all_remote_block_int8')
    error['passed'] = bool(torch.isfinite(actual).all()) and error['relative_l2'] <= .05 and error['protocol_relative_l2'] <= .01
    assert error['passed'], error
    return error
'''


def adapter_source(arm):
    require(arm in ARMS, 'Unknown compression arm')
    return base.adapter_source('flux-basic')


def worker_source(arm):
    require(arm in ARMS, 'Unknown compression arm')
    text = base.worker_source('flux-basic')
    text = replace_once(text, "if is_tensor(POLICY) and MODE == 'smoke':\n"
        '    from basic_quant_validation import validate as validate_codec\n'
        '    import reference as codec_reference\n'
        '    codec_validation = validate_codec(standalone_codec, codec_reference, local_rank)',
        '# Codec correctness is gated by the independent native/operator probe before model smoke.')
    text = replace_once(text, "report['rtx5090_transport'] = ({name: op.codec.transport_state() for name, op in adapter.scope_ops.items()}\n"
        "                             if PLAN['arm'] != 'flux-basic' else {})",
        "report['rtx5090_transport'] = {}\n"
        "report['compression_states'] = {name: op.codec.tensor_codec.state() for name, op in adapter.scope_ops.items()}\n"
        "report['open_compress_implementation'] = PLAN['open_compress_implementation']")
    base.validate_worker(text)
    return text


def configuration(arm, selection, communication, codec=None):
    require(arm in ARMS, 'Unknown compression arm')
    cfg = base.configuration('flux-basic', {}, communication, codec)
    cfg.update(protocol=PROTOCOL, arm=arm, policies=[arm], scope_definition=definition(arm),
        open_compress_implementation=implementation(arm), double_buffered=False,
        role_plan_sha256={arm: {'attn': None, 'mlp': None}},
        scope='RTX5090 TP8 full GPT6.7B step; open block INT8; no arrival or selective quantization',
        codec_identity=codec_identity(arm), codec_library=None if arm == 'flux-bf16' else codec)
    cfg['cases'][0]['name'] = 'rtx5090-open-compress'
    return cfg


def prepare_workers(build, out, plan):
    import open_compress_build as builder
    build, out = Path(build).resolve(), Path(out).resolve()
    builder.verify(build, True)
    workers, selection = {}, selection_metadata()
    for arm in ARMS:
        scripts = out/'workers'/arm/'scripts'
        scripts.mkdir(parents=True)
        runtime = f'TRANSPORT = {p.TRANSPORT!r}\nTRANSPORT_ENV = {p.TRANSPORT_ENV!r}\n'+base.runtime_source()
        generated = dict(worker=worker_source(arm), adapter=adapter_source(arm), taco_support=taco_source(arm),
            runtime=runtime, routing=routing_source(), megatron_taco=codec_source(arm),
            reference=reference_source(arm))
        # Keep reported reference identity exact even for the uncompressed arm.
        reference = generated['reference']
        for name, source in generated.items():
            (scripts/(name+'.py')).write_text(source)
        for source, name in ((H100/'ablation4_protocol.py', 'ablation4_protocol.py'),
                             (H100/'ablation4_runtime.py', 'communication.py'),
                             (H100/'ablation6_flux_tensor.py', 'flux_tensor_taco.py')):
            shutil.copy2(source, scripts/name)
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json', selection)
        codec = None if arm == 'flux-bf16' else builder.codec_library(build)
        cfg = configuration(arm, selection, plan['communication'], codec)
        cfg['codec_identity']['reference_sha256'] = hashlib.sha256(reference.encode()).hexdigest()
        write(scripts/'config.json', cfg)
        write(scripts.parent/'plan.json', plan)
        (scripts/'build').mkdir()
        (scripts/'build'/arm).symlink_to(builder.native_root(build, arm), target_is_directory=True)
        workers[arm] = dict(root=str(scripts), config=cfg, definition=definition(arm),
            libraries=builder.libraries(build, arm), selection=selection,
            files={path.name: sha(path) for path in scripts.iterdir() if path.is_file()},
            plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
