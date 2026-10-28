"""Freeze five double-scope controls, their builds, schedule and communication."""
from pathlib import Path
import shutil
import uuid

import ablation4_build as builder
import ablation4_protocol as routing
import ablation4_runtime as communication
import ablation4_sources as sources
from ablation_timing import execution_timing
from ablation_prepare import megatron_check
from common import HERE, REPO, MEGATRON_COMMIT, read, require, replace_once, sha, write
from scope_plan import model_config


def role_hashes(plans):
    return {p: {} if routing.is_native(p) or routing.is_original(p) else
            {r: record['entry_sha256'] for r, record in routing.role_plans(plans, p).items()}
            for p in routing.POLICIES}


def prepare(out, build_root, megatron, blocks=5, source_only=False, rounds=1, nvls='0'):
    build_root, megatron = Path(build_root).resolve(), Path(megatron).resolve()
    jobs = routing.schedule(blocks, rounds=rounds)
    m = builder.verify(build_root, built=not source_only)
    require(source_only or not m['source_only'], 'Cannot promote a CPU rehearsal')
    if not source_only:
        megatron_check(megatron)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    scripts = out/'scripts'; provenance = out/'provenance'
    for path in (scripts, provenance/'controllers', out/'results', out/'build'):
        path.mkdir(parents=True)
    for path in HERE.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, provenance/'controllers'/path.name)
    for name in ('adapter.py', 'worker.py', 'megatron_taco.py', 'taco_support.py'):
        shutil.copy2(REPO/'tools/a800/phase5/e2e'/name, provenance/('baseline-'+name))
    (scripts/'adapter.py').write_text(sources.adapter_source())
    (scripts/'worker.py').write_text(sources.worker_source(source_only))
    (scripts/'taco_support.py').write_text(sources.taco_source())
    native = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
    (scripts/'megatron_taco.py').write_text(replace_once(native, 'self.world==4', 'self.world in (4,8)'))
    for src, dst in ((HERE/'ablation4_protocol.py', 'routing.py'), (HERE/'runtime.py', 'runtime.py'),
                     (HERE/'ablation4_runtime.py', 'communication.py'),
                     (REPO/'tools/a800/phase4/reference.py', 'reference.py'), (build_root/'plans.json', 'selection-plan.json')):
        shutil.copy2(src, scripts/dst)
    snapshots = {'build-manifest.json': build_root/'ablation4-manifest.json',
                 'candidate-manifest.json': build_root/'candidate/manifest.json',
                 'fused-manifest.json': build_root/'fused/manifest.json'}
    build_files = {str(build_root/'plans.json'): sha(build_root/'plans.json')}
    for name, src in snapshots.items():
        shutil.copy2(src, provenance/name); build_files[str(src)] = sha(src)
    for child in (build_root/'fused', build_root/'candidate'):
        data = read(child/'manifest.json')
        build_files.update({str(child/p): h for p, h in data['files'].items()})
        for name, variant in data['variants'].items():
            build_files.update({str(child/name/p): h for p, h in variant['source_sha256'].items()})
    c = model_config(); plans = read(build_root/'plans.json'); comm = communication.capture(nvls)
    cfg = read(REPO/'tools/a800/phase5/e2e/config.json')
    cfg.update(protocol=routing.PROTOCOL, policies=list(routing.POLICIES), orders=[], repetitions=rounds,
               **execution_timing(), graph_forward=False, double_buffered=True, ring_reduction=True,
               cases=[dict(name='h100-phase6', **{k: c[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
               common={k: c[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
               role_plan_sha256=role_hashes(plans), communication=comm,
               scope='Ablation4: all five implementations quantize both Attention and MLP output projections; full optimizer steps.')
    write(scripts/'config.json', cfg); run_id = uuid.uuid4().hex
    write(provenance/'preparation.json', dict(protocol=routing.PROTOCOL, run_id=run_id, source_only=source_only,
            executable=not source_only, rounds=rounds, communication=comm, execution_timing=execution_timing(),
            plans_sha256=sha(build_root/'plans.json')))
    variants = {}
    for policy in routing.POLICIES:
        v = routing.definition(policy)
        if not routing.is_native(policy):
            lib = builder.runtime_root(build_root, policy)
            (out/'build'/policy).symlink_to(lib, target_is_directory=True)
            v.update(root=str(lib), sampling=False, libraries={})
            if not source_only:
                v['libraries'] = {n: dict(path=str((lib/n).resolve()), sha256=sha(lib/n))
                                  for n in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
        variants[policy] = v
    hardware = next((p['hardware'] for p in plans['parent']['roles'].values() if not p['test_only']), {})
    info = dict(protocol=routing.PROTOCOL, run_id=run_id, config=c, execution_timing=execution_timing(), source_only=source_only,
                executable=not source_only, gpu_validated=False, rounds=rounds, blocks_per_round=blocks,
                schedule_seed=20261013, schedule=jobs, timing_windows=len(jobs), variants=variants,
                communication=comm, calibration_hardware=hardware, megatron=str(megatron),
                megatron_commit=MEGATRON_COMMIT, candidate_source_commit=m['parent_source_commit'],
                build_manifest=dict(path=str(build_root/'ablation4-manifest.json'), sha256=sha(build_root/'ablation4-manifest.json')),
                build_files=build_files, controllers={str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'schedule.json', jobs)
    info['files'] = {str(p.relative_to(out)): sha(p) for folder in (scripts, provenance) for p in folder.rglob('*') if p.is_file()}
    info['files']['schedule.json'] = sha(out/'schedule.json')
    write(out/'submission.json', info)
    return out
