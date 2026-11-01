"""ABC worker overlays; frozen model/backward/optimizer/timing remain unchanged."""
import hashlib
from pathlib import Path
import shutil
from rtx5090_loader import HERE, H100, REPO, load
import op1_abc_protocol as p
import op1_abc_routing as routing
import flux_baseline_sources as original
import op1_abc_native as native

base = load('sources')
read, write, sha, require = p.read, p.write, p.sha, p.require
BASIC_ARMS = routing.POLICIES[:3]


def base_arm(arm):
    require(arm in p.ARMS, 'Unknown ABC arm')
    return 'c' if arm == 'flux-improved' else 'flux-basic'


def implementation(arm):
    base_arm(arm)
    if arm == 'flux-improved':
        return dict(original.implementation(arm), comparison_protocol=p.PROTOCOL)
    return dict(protocol=p.PROTOCOL, original_arm='flux-basic',
        producer='flux-gemmonly-sm89-v2', gemm_schedule='StreamkSK-with-CTA-interleave',
        may_change_streamk_reduction_assignment=True,
        quantization='selective-tail-1/64' if arm == BASIC_ARMS[2] else 'all_remote',
        interleaved=True, arrival=arm != BASIC_ARMS[0], selective=arm == BASIC_ARMS[2],
        calibration_used=False, cta_order='rank-rotated-destination-interleaved',
        decoder='basic-ring-bf16-h128-direct-v1', collective='NCCL-byte-AllToAll',
        chunks=1 if arm == BASIC_ARMS[0] else 4, owns_configuration=True, owns_output=True,
        packet_layout='H128-AoS-136-or-BF16-256', original_peer_overlap_preserved=False,
        arrival_scope='completed-row-chunk' if arm != BASIC_ARMS[0] else 'whole-exchange')


def definition(arm):
    return dict(routing.definition(arm), experiment=implementation(arm))


def selection_metadata():
    return dict(protocol=p.PROTOCOL, calibration_used=False,
        roles={role: dict(role=role, shape=shape, K_local=shape[2]//8)
               for role, shape in routing.SHAPES.items()})


def codec_source(arm):
    require(arm in BASIC_ARMS, 'Only ABC arms have a standalone codec')
    return (HERE/'op1_abc_codec.py').read_text()


def codec_identity(arm):
    value = base.basic_quant_sources.codec_identity()
    value.update(implementation='rtx5090-op1-abc-direct-h128-v1',
        source_sha256=hashlib.sha256(codec_source(arm).encode()).hexdigest(),
        cuda_source_identity=native.identity(), receiver_implementation='basic-ring-bf16-h128-direct-v1')
    return value


def taco_source(arm):
    if arm == 'flux-improved':
        return original.taco_source(arm)
    source = base.taco_source('flux-basic')
    # Reference path constructs the same shape-derived mask independently of
    # CUDA packet layout. Local contributions remain unquantized.
    source = native.replace(source, 'import json\n', 'import json\nimport os\n')
    source = native.replace(source, '    if mask is None:\n        return coded', '''    if os.environ['E2E_POLICY'] == 'flux-basic-selective':
        groups = piece.numel() // 128
        selected = torch.arange(groups, device=piece.device) >= groups - groups // 64
        return torch.where(selected.repeat_interleave(128).reshape_as(piece), coded, piece)
    if mask is None:
        return coded''')
    return source


def adapter_source(arm):
    return original.adapter_source('flux-improved') if arm == 'flux-improved' else base.adapter_source('flux-basic')


def worker_source(arm):
    source = original.worker_source('flux-improved' if arm == 'flux-improved' else 'flux-basic')
    source = native.replace(source, "report['flux_implementation'] = PLAN['flux_implementation']",
                            "report['op1_abc_implementation'] = PLAN['op1_abc_implementation']")
    # Operators are created lazily by the first forward. Snapshot their mutable
    # counters only after warmup AND the measured/profile steps have completed,
    # alongside the existing final native state, before serializing rank JSON.
    anchor = "report['rtx5090_transport'] = "
    final_state = "report['op1_abc_transport'] = {}\n"
    if arm in BASIC_ARMS:
        final_state = """report['op1_abc_transport'] = {
    name: op.codec.tensor_codec.state() for name, op in adapter.scope_ops.items()}
assert set(report['op1_abc_transport']) == set(report['target_modules']), 'Incomplete final ABC transport coverage'
assert all(state['calls'] == (warmup + (1 if MODE == 'profile' else count))*microsteps
           for state in report['op1_abc_transport'].values()), 'Stale final ABC transport counters'
"""
    source = native.replace(source, anchor, final_state + anchor)
    base.validate_worker(source)
    return source


def configuration(arm, selection, communication, codec=None):
    cfg = base.configuration('a' if arm == 'flux-improved' else 'flux-basic', {}, communication, codec)
    cfg.update(protocol=p.PROTOCOL, arm=arm, policies=[arm], scope_definition=definition(arm),
        op1_abc_implementation=implementation(arm), double_buffered=arm == 'flux-improved',
        role_plan_sha256={arm: {'attn': None, 'mlp': None}},
        scope='RTX5090 ABC direct H128 packets and completed-chunk decode; deterministic sparse tail mask')
    if arm in BASIC_ARMS:
        cfg.update(codec_identity=codec_identity(arm), codec_library=codec)
    return cfg


def prepare_workers(build, out, plan):
    import op1_abc_build as builder
    build, out = Path(build), Path(out)
    selection, workers = selection_metadata(), {}
    for arm in p.ARMS:
        scripts = out/'workers'/arm/'scripts'
        scripts.mkdir(parents=True)
        runtime = f'TRANSPORT = {p.TRANSPORT!r}\nTRANSPORT_ENV = {p.TRANSPORT_ENV!r}\n'+base.runtime_source()
        generated = {'worker.py': worker_source(arm), 'adapter.py': adapter_source(arm),
            'taco_support.py': taco_source(arm), 'runtime.py': runtime,
            'routing.py': (HERE/'op1_abc_routing.py').read_text(),
            'megatron_taco.py': codec_source(arm) if arm in BASIC_ARMS else
                (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text().replace('self.world==4', 'self.world in (4,8)')}
        for name, source in generated.items():
            (scripts/name).write_text(source)
        paths = [(H100/'ablation4_protocol.py', 'ablation4_protocol.py'),
            (H100/'ablation4_runtime.py', 'communication.py'),
            (REPO/'tools/a800/phase4/reference.py', 'reference.py')]
        if arm in BASIC_ARMS:
            paths += [(H100/name, name) for name in ('baseline_codec_validation.py', 'basic_quant_validation.py')]
            paths.append((H100/'ablation6_flux_tensor.py', 'flux_tensor_taco.py'))
        for source, name in paths:
            shutil.copy2(source, scripts/name)
        (scripts/'common.py').write_text('def require(ok, message):\n    if not ok: raise ValueError(message)\n')
        write(scripts/'selection-plan.json', selection)
        codec = builder.codec_library(build) if arm in BASIC_ARMS else None
        cfg = configuration(arm, selection, plan['communication'], codec)
        write(scripts/'config.json', cfg)
        write(scripts.parent/'plan.json', plan)
        (scripts/'build').mkdir()
        (scripts/'build'/arm).symlink_to(builder.native_root(build, arm), target_is_directory=True)
        workers[arm] = dict(root=str(scripts), config=cfg, definition=definition(arm),
            libraries=builder.libraries(build, arm), selection=selection,
            files={path.name: sha(path) for path in scripts.iterdir() if path.is_file()},
            plan_sha256=sha(scripts.parent/'plan.json'))
    return workers
