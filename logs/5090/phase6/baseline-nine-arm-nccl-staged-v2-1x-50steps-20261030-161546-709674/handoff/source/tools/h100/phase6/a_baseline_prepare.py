"""Prepare a fresh, fully frozen basic/improved/standalone-CUDA comparison."""
from pathlib import Path
import shutil
import socket
import subprocess
import uuid

import a_baseline_build as fused_build
import a_baseline_protocol as routing
import a_baseline_training_sources as sources
import ablation4_runtime as communication
import ablation6_build as flux_build
import baseline_codec_build as codec_build
from ablation_prepare import megatron_check
from common import HERE, REPO, MEGATRON_COMMIT, read, require, sha, write
from scope_plan import model_config, validate_hardware

SHARED_ENVIRONMENT = {'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'FLUX_GEMM_RS_GEMM_ONLY': '1'}
MODEL_KEYS = ('tp', 'sm_count', 'layers', 'hidden', 'ffn', 'heads', 'sequence',
              'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')


def model():
    source = model_config()
    return {k: source[k] for k in MODEL_KEYS}


def capture_hardware():
    # Called only for a real prepare inside the explicitly launched batch.
    # This is runtime identity capture, not an arrival calibration.
    import torch
    require(torch.cuda.device_count() == 8, 'Expose exactly eight H100 GPUs')
    torch.cuda.set_device(0)
    devices = []
    for i in range(8):
        p = torch.cuda.get_device_properties(i)
        devices.append(dict(name=p.name, uuid=str(p.uuid), sm_count=p.multi_processor_count,
                            capability=[p.major, p.minor]))
    record = dict(host=socket.gethostname(), devices=devices)
    validate_hardware(record)
    require(all(torch.cuda.can_device_access_peer(i, j) for i in range(8) for j in range(8) if i != j),
            'All-to-all local GPU peer access is required')
    return record


def shape_metadata():
    # scope_report consumes these shape records even for nonselective baselines.
    # None is deliberate: these are not a measured mask, ordering or calibration.
    return dict(protocol=routing.PROTOCOL, calibration_used=False,
                roles={role: dict(role=role, entry=dict(shape=shape, K_local=shape[2]//8, mask=None),
                                  entry_sha256=None, mask_sha256=None)
                       for role, shape in routing.SHAPES.items()})


def runtime_root(build_root, flux_root, policy):
    return Path(flux_root)/flux_build.VARIANT if routing.is_flux_tensor(policy) else fused_build.runtime_root(build_root, policy)


def prepare(out, build_root, flux_root, codec_root, megatron, blocks=6, rounds=2,
            nvls='0', source_only=False, scope='both'):
    require(scope == 'both', 'All three arms must cover Attention and MLP together')
    build_root, flux_root, codec_root, megatron = map(lambda p: Path(p).resolve(),
                                                    (build_root, flux_root, codec_root, megatron))
    jobs = routing.schedule(blocks, rounds=rounds)
    f = fused_build.verify(build_root, built=not source_only)
    g = flux_build.verify(flux_root, built=not source_only)
    c = codec_build.verify(codec_root, built=not source_only)
    require(f['source_only'] == g['source_only'] == c['source_only'] == source_only,
            'CPU rehearsals and executable builds cannot be mixed')
    if not source_only:
        megatron_check(megatron)
    hardware = {} if source_only else capture_hardware()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scripts, provenance = out/'scripts', out/'provenance'
    for folder in (scripts, provenance/'controllers', out/'build', out/'results'):
        folder.mkdir(parents=True)
    controllers = {str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}
    for path in controllers:
        shutil.copy2(path, provenance/'controllers'/Path(path).name)
    for name in ('adapter', 'taco', 'worker'):
        target = {'adapter': 'adapter.py', 'taco': 'taco_support.py', 'worker': 'worker.py'}[name]
        fn = getattr(sources, name+'_source')
        (scripts/target).write_text(fn(source_only) if name == 'worker' else fn())
    (scripts/'megatron_taco.py').write_text(sources.codec_source())
    for src, name in ((HERE/'a_baseline_protocol.py', 'routing.py'),
                      (HERE/'ablation6_flux_tensor.py', 'flux_tensor_taco.py'),
                      (HERE/'runtime.py', 'runtime.py'), (HERE/'ablation4_runtime.py', 'communication.py'),
                      (HERE/'baseline_codec_validation.py', 'baseline_codec_validation.py'),
                      (REPO/'tools/a800/phase4/reference.py', 'reference.py')):
        shutil.copy2(src, scripts/name)
    write(scripts/'selection-plan.json', shape_metadata())
    snapshots = {'fused-manifest.json': build_root/'manifest.json',
                 'fused-identity.json': build_root/fused_build.MANIFEST,
                 'gemm-only-manifest.json': flux_root/'manifest.json',
                 'codec-manifest.json': codec_root/'manifest.json'}
    for name, src in snapshots.items():
        shutil.copy2(src, provenance/name)
    m = model()
    comm = communication.capture(nvls)
    codec = sources.codec_identity()
    run_id = uuid.uuid4().hex
    cfg = read(REPO/'tools/a800/phase5/e2e/config.json')
    cfg.update(protocol=routing.PROTOCOL, policies=list(routing.POLICIES), orders=[], repetitions=rounds,
               **routing.TIMING, graph_forward=False, double_buffered=False, ring_reduction=True,
               cases=[dict(name='h100-phase6', **{k: m[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
               common={k: m[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
               role_plan_sha256={p: {r: None for r in routing.SHAPES} for p in routing.POLICIES},
               communication=comm, codec_identity=codec, codec_library=c['library'],
               scope='Three all-remote, original-order baselines on both output projections; no new scheduling, prefetch, graph or submit cache.')
    write(scripts/'config.json', cfg)
    variants = {}
    for p in routing.POLICIES:
        lib = runtime_root(build_root, flux_root, p)
        (out/'build'/p).symlink_to(lib, target_is_directory=True)
        variants[p] = dict(routing.definition(p), root=str(lib), sampling=False,
            libraries={} if source_only else {n: dict(path=str((lib/n).resolve()), sha256=sha(lib/n))
                                              for n in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')})
    info = dict(protocol=routing.PROTOCOL, run_id=run_id, config=m,
                execution_timing=dict(routing.TIMING), comparison_scope='both',
                source_only=source_only, executable=not source_only, gpu_validated=False,
                rounds=rounds, blocks_per_round=blocks, schedule_seed=routing.SCHEDULE_SEED,
                schedule=jobs, timing_windows=len(jobs), variants=variants,
                communication=comm, shared_environment=dict(SHARED_ENVIRONMENT),
                hardware=hardware, calibration_hardware=hardware,
                hardware_identity_kind='runtime identity only; no arrival calibration',
                megatron=str(megatron), megatron_commit=MEGATRON_COMMIT,
                codec_identity=codec, codec_library=c['library'], controllers=controllers,
                build_manifest=dict(path=str(build_root/'manifest.json'), sha256=sha(build_root/'manifest.json')),
                fused_identity=dict(path=str(build_root/fused_build.MANIFEST), sha256=sha(build_root/fused_build.MANIFEST)),
                flux_build_manifest=dict(path=str(flux_root/'manifest.json'), sha256=sha(flux_root/'manifest.json')),
                codec_build_manifest=dict(path=str(codec_root/'manifest.json'), sha256=sha(codec_root/'manifest.json')))
    write(provenance/'preparation.json', {k: info[k] for k in
          ('protocol', 'run_id', 'source_only', 'executable', 'rounds', 'execution_timing', 'comparison_scope',
           'communication', 'shared_environment', 'hardware', 'codec_identity', 'codec_library')})
    write(out/'schedule.json', jobs)
    info['files'] = {str(p.relative_to(out)): sha(p) for folder in (scripts, provenance)
                     for p in folder.rglob('*') if p.is_file()}
    info['files']['schedule.json'] = sha(out/'schedule.json')
    write(out/'submission.json', info)
    return out
