"""Freeze four scopes, both role plans and one shared candidate build."""
from pathlib import Path
import shutil
import uuid

from ablation_prepare import megatron_check
from common import HERE, REPO, MEGATRON_COMMIT, read, require, replace_once, sha, write
import scope_build
import scope_sources
from scope_protocol import PROTOCOL, POLICIES, definition, is_native, schedule
from scope_plan import model_config


def prepare(out, build_root, megatron, blocks=4, source_only=False):
    build_root, megatron = Path(build_root).resolve(), Path(megatron).resolve()
    m = scope_build.verify(build_root, built=not source_only, kind='scope_candidate')
    jobs = schedule(blocks)
    if not source_only:
        megatron_check(megatron)
    plans = read(build_root/'plans.json')
    require(source_only or not plans['source_only'], 'Cannot promote CPU synthetic plans')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    scripts = out/'scripts'; scripts.mkdir(); (out/'results').mkdir(); (out/'build').mkdir()
    provenance = out/'provenance'; provenance.mkdir(); (provenance/'controllers').mkdir()
    for path in HERE.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, provenance/'controllers'/path.name)
    for name in ('adapter.py', 'worker.py', 'megatron_taco.py', 'taco_support.py'):
        shutil.copy2(REPO/'tools/a800/phase5/e2e'/name, provenance/('baseline-'+name))
    (scripts/'adapter.py').write_text(scope_sources.adapter_source())
    (scripts/'worker.py').write_text(scope_sources.worker_source(source_only))
    (scripts/'taco_support.py').write_text(scope_sources.taco_source())
    native = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
    (scripts/'megatron_taco.py').write_text(replace_once(native, 'self.world==4', 'self.world in (4,8)'))
    shutil.copy2(HERE/'scope_protocol.py', scripts/'routing.py')
    shutil.copy2(HERE/'runtime.py', scripts/'runtime.py')
    shutil.copy2(REPO/'tools/a800/phase4/reference.py', scripts/'reference.py')
    shutil.copy2(build_root/'plans.json', scripts/'selection-plan.json')
    shutil.copy2(build_root/'manifest.json', provenance/'build-manifest.json')
    shutil.copy2(build_root/'identity.json', provenance/'build-identity.json')
    c = model_config(); cfg = read(REPO/'tools/a800/phase5/e2e/config.json')
    cfg.update(protocol=PROTOCOL, policies=list(POLICIES), orders=[], repetitions=2, warmup_steps=10, timed_steps=20,
               cases=[dict(name='h100-phase6', **{k: c[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
               common={k: c[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
               graph_forward=False, double_buffered=True, ring_reduction=True,
               role_plan_sha256={r: p['entry_sha256'] for r, p in plans['roles'].items()},
               scope='GPT6.7B TP8 Attention/MLP scope comparison, full optimizer steps; unselected modules native BF16.')
    write(scripts/'config.json', cfg)
    run_id = uuid.uuid4().hex
    write(provenance/'preparation.json', dict(protocol=PROTOCOL, run_id=run_id, source_only=source_only,
                                             executable=not source_only, plans_sha256=sha(build_root/'plans.json')))
    lib = build_root/'scope-candidate'
    variants = {}
    for p in POLICIES:
        v = definition(p)
        if not is_native(p):
            v.update(root=str(lib), sampling=False, libraries={})
            (out/'build'/p).symlink_to(lib, target_is_directory=True)
            if not source_only:
                for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                    path = (lib/name).resolve()
                    v['libraries'][name] = dict(path=str(path), sha256=sha(path))
        variants[p] = v
    hardware = next((p['hardware'] for p in plans['roles'].values() if not p['test_only']), {})
    info = dict(protocol=PROTOCOL, run_id=run_id, config=c, source_only=source_only, executable=not source_only,
                gpu_validated=False, blocks_per_round=blocks, schedule_seed=20261013,
                schedule=jobs, timing_windows=len(jobs), variants=variants, calibration_hardware=hardware,
                megatron=str(megatron), megatron_commit=MEGATRON_COMMIT,
                candidate_source_commit=m['parent_source_commit'],
                build_manifest=dict(path=str(build_root/'manifest.json'), sha256=sha(build_root/'manifest.json')),
                build_files={**{str(build_root/p): h for p, h in m['files'].items()},
                             **{str(lib/p): h for p, h in m['variants']['scope-candidate']['source_sha256'].items()}},
                controllers={str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'schedule.json', jobs)
    info['files'] = {str(p.relative_to(out)): sha(p) for folder in (scripts, provenance) for p in folder.rglob('*') if p.is_file()}
    info['files']['schedule.json'] = sha(out/'schedule.json')
    write(out/'submission.json', info)
    return out
