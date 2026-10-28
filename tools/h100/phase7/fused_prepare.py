"""Freeze nine full-model policies using three versions of the fused tile order."""
from pathlib import Path
import shutil
import uuid

from fused_support import HERE7, PHASE6, REPO
import fused_build as builder
from fused_protocol import PROTOCOL, POLICIES, SCHEDULE_SEED, definition, schedule
from ablation_prepare import megatron_check
from common import MEGATRON_COMMIT, read, require, replace_once, sha, write
from scope_plan import model_config
import scope_sources


def worker_source(source_only=False):
    source = scope_sources.worker_source(source_only)
    source = replace_once(source, "os.environ.get('H100_SCOPE_EXECUTE')",
                          "os.environ.get('H100_PHASE7_FUSED_EXECUTE')")
    return replace_once(source, 'GPU worker requires scope --execute',
                        'GPU worker requires phase7 fused --execute')


def controller_paths():
    return [path for path in PHASE6.glob('*.py') if not path.name.startswith('test_')] + \
           [path for path in HERE7.glob('fused*.py') if not path.name.startswith('test_')]


def prepare(out, build_root, megatron, blocks=9, source_only=False, rounds=2):
    build_root, megatron = Path(build_root).resolve(), Path(megatron).resolve()
    jobs = schedule(blocks, rounds=rounds)
    manifest = builder.verify(build_root, built=not source_only)
    plans = read(build_root / 'plans.json')
    require(source_only or not plans['source_only'], 'Cannot promote a CPU rehearsal')
    if not source_only:
        megatron_check(megatron)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scripts, provenance = out / 'scripts', out / 'provenance'
    for folder in (scripts, provenance, out / 'results', out / 'build',
                   provenance / 'controllers' / 'phase6', provenance / 'controllers' / 'phase7'):
        folder.mkdir(parents=True, exist_ok=True)
    controllers = controller_paths()
    for path in controllers:
        family = 'phase6' if path.parent == PHASE6 else 'phase7'
        shutil.copy2(path, provenance / 'controllers' / family / path.name)
    for name in ('adapter.py', 'worker.py', 'megatron_taco.py', 'taco_support.py'):
        shutil.copy2(REPO / 'tools/a800/phase5/e2e' / name, provenance / ('baseline-' + name))
    # Reuse the audited full-model adapter and fused double-buffered wrapper.
    # No import of the standalone Phase7 Tensor/NCCL prototype occurs here.
    (scripts / 'adapter.py').write_text(scope_sources.adapter_source())
    (scripts / 'worker.py').write_text(worker_source(source_only))
    (scripts / 'taco_support.py').write_text(scope_sources.taco_source())
    native = (REPO / 'tools/a800/phase5/e2e/megatron_taco.py').read_text()
    (scripts / 'megatron_taco.py').write_text(replace_once(native, 'self.world==4', 'self.world in (4,8)'))
    for src, dst in ((HERE7 / 'fused_protocol.py', 'routing.py'), (PHASE6 / 'runtime.py', 'runtime.py'),
                     (REPO / 'tools/a800/phase4/reference.py', 'reference.py'),
                     (build_root / 'plans.json', 'selection-plan.json')):
        shutil.copy2(src, scripts / dst)
    shutil.copy2(build_root / 'fused-manifest.json', provenance / 'build-manifest.json')
    config = model_config()
    cfg = read(REPO / 'tools/a800/phase5/e2e/config.json')
    cfg.update(protocol=PROTOCOL, policies=list(POLICIES), orders=[], repetitions=rounds,
               warmup_steps=10, timed_steps=20,
               cases=[dict(name='h100-phase6', **{k: config[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
               common={k: config[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
               graph_forward=False, double_buffered=True, ring_reduction=True,
               role_plan_sha256={policy: {role: record['entry_sha256'] for role, record in data['roles'].items()}
                                 for policy, data in plans['variants'].items()},
               scope='Phase7 fused: identical selective masks/codec, base/legacy/improved order, full optimizer steps.')
    write(scripts / 'config.json', cfg)
    run_id = uuid.uuid4().hex
    write(provenance / 'preparation.json', dict(protocol=PROTOCOL, run_id=run_id,
          source_only=source_only, executable=not source_only, rounds=rounds,
          plans_sha256=sha(build_root / 'plans.json')))
    variants = {}
    build_files = {str(build_root / 'plans.json'): sha(build_root / 'plans.json')}
    for order in builder.ORDERS:
        child_root = build_root / order
        child = read(child_root / 'manifest.json')
        lib = child_root / 'scope-candidate'
        shutil.copy2(child_root / 'manifest.json', provenance / f'build-{order}-manifest.json')
        shutil.copy2(child_root / 'identity.json', provenance / f'build-{order}-identity.json')
        build_files[str(child_root / 'manifest.json')] = sha(child_root / 'manifest.json')
        build_files.update({str(child_root / name): value for name, value in child['files'].items()})
        build_files.update({str(lib / name): value
                            for name, value in child['variants']['scope-candidate']['source_sha256'].items()})
    for policy in POLICIES:
        variant = definition(policy)
        lib = build_root / variant['order'] / 'scope-candidate'
        variant.update(root=str(lib), sampling=False, libraries={})
        (out / 'build' / policy).symlink_to(lib, target_is_directory=True)
        if not source_only:
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                path = (lib / name).resolve()
                variant['libraries'][name] = dict(path=str(path), sha256=sha(path))
        variants[policy] = variant
    hardware = next((record['hardware'] for record in plans['parent']['roles'].values()
                     if not record['test_only']), {})
    info = dict(protocol=PROTOCOL, run_id=run_id, config=config, source_only=source_only,
                executable=not source_only, gpu_validated=False, blocks_per_round=blocks, rounds=rounds,
                schedule_seed=SCHEDULE_SEED, schedule=jobs, timing_windows=len(jobs), variants=variants,
                calibration_hardware=hardware, megatron=str(megatron), megatron_commit=MEGATRON_COMMIT,
                candidate_source_commit=manifest['parent_source_commit'],
                build_manifest=dict(path=str(build_root / 'fused-manifest.json'),
                                    sha256=sha(build_root / 'fused-manifest.json')),
                build_files=build_files, controllers={str(path): sha(path) for path in controllers})
    write(out / 'schedule.json', jobs)
    info['files'] = {str(path.relative_to(out)): sha(path)
                     for folder in (scripts, provenance) for path in folder.rglob('*') if path.is_file()}
    info['files']['schedule.json'] = sha(out / 'schedule.json')
    write(out / 'submission.json', info)
    return out
