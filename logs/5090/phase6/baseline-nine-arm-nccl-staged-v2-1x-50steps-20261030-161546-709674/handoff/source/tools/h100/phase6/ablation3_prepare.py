"""Freeze nine policies, two libraries and all physical plans in a new session."""
from pathlib import Path
import shutil
import uuid

from ablation_prepare import megatron_check
import ablation3_build as builder
import ablation3_sources as sources
import scope_sources
from ablation3_protocol import PROTOCOL, POLICIES, definition, schedule
from scope_plan import model_config
from common import HERE, REPO, MEGATRON_COMMIT, read, require, replace_once, sha, write


def prepare(out, build_root, megatron, blocks=9, source_only=False, rounds=2):
    build_root, megatron = Path(build_root).resolve(), Path(megatron).resolve()
    jobs = schedule(blocks, rounds=rounds)
    m = builder.verify(build_root, built=not source_only)
    plans = read(build_root/'plans.json')
    require(source_only or not plans['source_only'], 'Cannot promote a CPU rehearsal')
    if not source_only:
        megatron_check(megatron)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    scripts = out/'scripts'; scripts.mkdir(); (out/'results').mkdir(); (out/'build').mkdir()
    provenance = out/'provenance'; provenance.mkdir(); (provenance/'controllers').mkdir()
    for path in HERE.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, provenance/'controllers'/path.name)
    for name in ('adapter.py', 'worker.py', 'megatron_taco.py', 'taco_support.py'):
        shutil.copy2(REPO/'tools/a800/phase5/e2e'/name, provenance/('baseline-'+name))
    (scripts/'adapter.py').write_text(scope_sources.adapter_source())
    (scripts/'worker.py').write_text(sources.worker_source(source_only))
    (scripts/'taco_support.py').write_text(sources.taco_source())
    native = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
    (scripts/'megatron_taco.py').write_text(replace_once(native, 'self.world==4', 'self.world in (4,8)'))
    for src, dst in ((HERE/'ablation3_protocol.py', 'routing.py'), (HERE/'runtime.py', 'runtime.py'),
                     (REPO/'tools/a800/phase4/reference.py', 'reference.py'), (build_root/'plans.json', 'selection-plan.json')):
        shutil.copy2(src, scripts/dst)
    shutil.copy2(build_root/'ablation3-manifest.json', provenance/'build-manifest.json')
    c = model_config(); cfg = read(REPO/'tools/a800/phase5/e2e/config.json')
    cfg.update(protocol=PROTOCOL, policies=list(POLICIES), orders=[], repetitions=rounds, warmup_steps=10, timed_steps=20,
               cases=[dict(name='h100-phase6', **{k: c[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
               common={k: c[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
               graph_forward=False, double_buffered=True, ring_reduction=True,
               role_plan_sha256={p: {r: v['entry_sha256'] for r, v in data['roles'].items()}
                                 for p, data in plans['variants'].items()},
               scope='Nine-group full optimizer steps; unselected projections native BF16.')
    write(scripts/'config.json', cfg); run_id = uuid.uuid4().hex
    write(provenance/'preparation.json', dict(protocol=PROTOCOL, run_id=run_id, source_only=source_only,
                                             executable=not source_only, rounds=rounds,
                                             plans_sha256=sha(build_root/'plans.json')))
    variants = {}; build_files = {str(build_root/'plans.json'): sha(build_root/'plans.json')}
    for order in builder.ORDERS:
        root = build_root/order; child = read(root/'manifest.json'); lib = root/'scope-candidate'
        shutil.copy2(root/'manifest.json', provenance/f'build-{order}-manifest.json')
        shutil.copy2(root/'identity.json', provenance/f'build-{order}-identity.json')
        build_files[str(root/'manifest.json')] = sha(root/'manifest.json')
        build_files.update({str(root/p): h for p, h in child['files'].items()})
        build_files.update({str(lib/p): h for p, h in child['variants']['scope-candidate']['source_sha256'].items()})
    for p in POLICIES:
        v = definition(p); lib = build_root/v['order']/'scope-candidate'
        v.update(root=str(lib), sampling=False, libraries={})
        (out/'build'/p).symlink_to(lib, target_is_directory=True)
        if not source_only:
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                path = (lib/name).resolve(); v['libraries'][name] = dict(path=str(path), sha256=sha(path))
        variants[p] = v
    hardware = next((p['hardware'] for p in plans['parent']['roles'].values() if not p['test_only']), {})
    info = dict(protocol=PROTOCOL, run_id=run_id, config=c, source_only=source_only, executable=not source_only,
                gpu_validated=False, rounds=rounds, blocks_per_round=blocks, schedule_seed=20261013,
                schedule=jobs, timing_windows=len(jobs), variants=variants, calibration_hardware=hardware,
                megatron=str(megatron), megatron_commit=MEGATRON_COMMIT, candidate_source_commit=m['parent_source_commit'],
                build_manifest=dict(path=str(build_root/'ablation3-manifest.json'), sha256=sha(build_root/'ablation3-manifest.json')),
                build_files=build_files, controllers={str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'schedule.json', jobs)
    info['files'] = {str(p.relative_to(out)): sha(p) for folder in (scripts, provenance) for p in folder.rglob('*') if p.is_file()}
    info['files']['schedule.json'] = sha(out/'schedule.json')
    write(out/'submission.json', info)
    return out
