"""Freeze unchanged A/B/C and two standalone upstream CUDA codec baselines."""
from pathlib import Path
import shutil
import uuid

import ablation4_build as parent_build
import ablation4_runtime as communication
import ablation6_build as flux_build
import baseline_codec_build as codec_build
import baseline_protocol as routing
import baseline_sources as sources
from ablation_prepare import megatron_check
from common import HERE, REPO, MEGATRON_COMMIT, read, require, sha, write
from scope_plan import model_config


def role_hashes(plans):
    return {p: {} if routing.is_tensor(p) else
            {r: record['entry_sha256'] for r, record in routing.role_plans(plans,p).items()}
            for p in routing.POLICIES}


def runtime_root(build_root, flux_root, policy):
    return Path(flux_root)/flux_build.VARIANT if routing.is_flux_tensor(policy) else parent_build.runtime_root(build_root,policy)


def prepare(out, build_root, flux_root, megatron, blocks=5, rounds=2, nvls='0', source_only=False, codec_root=None):
    build_root, flux_root, megatron = map(lambda p: Path(p).resolve(), (build_root,flux_root,megatron))
    jobs=routing.schedule(blocks,rounds=rounds)
    m=parent_build.verify(build_root,built=not source_only); fm=flux_build.verify(flux_root,built=not source_only)
    require(codec_root is not None, 'A standalone CUDA codec build is required')
    codec_root=Path(codec_root).resolve(); cm=codec_build.verify(codec_root,built=not source_only)
    require(m['source_only']==fm['source_only']==cm['source_only']==source_only,'Cannot mix real builds and CPU rehearsals')
    if not source_only: megatron_check(megatron)
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    scripts=out/'scripts'; provenance=out/'provenance'
    for folder in (scripts,provenance/'controllers',out/'results',out/'build'): folder.mkdir(parents=True)
    for p in HERE.glob('*.py'):
        if not p.name.startswith('test_'): shutil.copy2(p,provenance/'controllers'/p.name)
    for name in ('adapter','taco','worker'):
        filename={'adapter':'adapter.py','taco':'taco_support.py','worker':'worker.py'}[name]
        function=getattr(sources,name+'_source')
        (scripts/filename).write_text(function(source_only) if name=='worker' else function())
    (scripts/'megatron_taco.py').write_text(sources.codec_source())
    shutil.copy2(HERE/'baseline_codec_validation.py',scripts/'baseline_codec_validation.py')
    for src,name in ((HERE/'baseline_protocol.py','routing.py'), (HERE/'ablation4_protocol.py','ablation4_protocol.py'),
                     (HERE/'ablation6_flux_tensor.py','flux_tensor_taco.py'), (HERE/'runtime.py','runtime.py'),
                     (HERE/'ablation4_runtime.py','communication.py'),
                     (REPO/'tools/a800/phase4/reference.py','reference.py'), (build_root/'plans.json','selection-plan.json')):
        shutil.copy2(src,scripts/name)
    snapshots={'build-manifest.json':build_root/'ablation4-manifest.json',
               'candidate-manifest.json':build_root/'candidate/manifest.json',
               'fused-manifest.json':build_root/'fused/manifest.json','gemm-only-manifest.json':flux_root/'manifest.json',
               'codec-manifest.json':codec_root/'manifest.json'}
    build_files={str(build_root/'plans.json'):sha(build_root/'plans.json')}
    for name,src in snapshots.items(): shutil.copy2(src,provenance/name); build_files[str(src)]=sha(src)
    for child in (build_root/'fused',build_root/'candidate'):
        record=read(child/'manifest.json')
        build_files.update({str(child/p):h for p,h in record['files'].items()})
        for name,v in record['variants'].items(): build_files.update({str(child/name/p):h for p,h in v['source_sha256'].items()})
    build_files.update({str(flux_root/p):h for p,h in fm['files'].items()})
    build_files.update({str(flux_root/flux_build.VARIANT/p):h for p,h in fm['source_sha256'].items()})
    build_files.update({str(codec_root/p):h for p,h in cm['files'].items()})
    model=model_config(); plans=read(build_root/'plans.json'); comm=communication.capture(nvls)
    codec=sources.codec_identity(); run_id=uuid.uuid4().hex
    cfg=read(REPO/'tools/a800/phase5/e2e/config.json')
    cfg.update(protocol=routing.PROTOCOL,policies=list(routing.POLICIES),orders=[],repetitions=rounds,
        **routing.TIMING,graph_forward=False,double_buffered=True,ring_reduction=True,
        cases=[dict(name='h100-phase6',**{k:model[k] for k in ('hidden','ffn','heads','sequence','tp')})],
        common={k:model[k] for k in ('layers','micro_batch','global_batch','seed','token_seed','vocab')},
        role_plan_sha256=role_hashes(plans),communication=comm,codec_identity=codec,codec_library=cm['library'],
        scope='A/B/C unchanged from ablation6; Flux GemmOnly and Megatron share upstream standalone TACO CUDA kernels; no COCCL.')
    write(scripts/'config.json',cfg)
    write(provenance/'preparation.json',dict(protocol=routing.PROTOCOL,run_id=run_id,source_only=source_only,
        executable=not source_only,rounds=rounds,communication=comm,execution_timing=routing.TIMING,
        codec_identity=codec,codec_library=cm['library'],plans_sha256=sha(build_root/'plans.json')))
    variants={}
    for p in routing.POLICIES:
        v=routing.definition(p)
        if not routing.is_native(p):
            lib=runtime_root(build_root,flux_root,p); (out/'build'/p).symlink_to(lib,target_is_directory=True)
            v.update(root=str(lib),sampling=False,libraries={} if source_only else
                {n:dict(path=str((lib/n).resolve()),sha256=sha(lib/n)) for n in ('libflux_cuda.so','libflux_cuda_ths_op.so')})
        variants[p]=v
    hardware=next((p['hardware'] for p in plans['parent']['roles'].values() if not p['test_only']),{})
    info=dict(protocol=routing.PROTOCOL,run_id=run_id,config=model,execution_timing=routing.TIMING,
        codec_identity=codec,codec_library=cm['library'],source_only=source_only,executable=not source_only,gpu_validated=False,
        rounds=rounds,blocks_per_round=blocks,schedule_seed=20261013,schedule=jobs,timing_windows=len(jobs),
        variants=variants,communication=comm,calibration_hardware=hardware,megatron=str(megatron),
        megatron_commit=MEGATRON_COMMIT,candidate_source_commit=m['parent_source_commit'],
        build_manifest=dict(path=str(build_root/'ablation4-manifest.json'),sha256=sha(build_root/'ablation4-manifest.json')),
        flux_build_manifest=dict(path=str(flux_root/'manifest.json'),sha256=sha(flux_root/'manifest.json')),
        codec_build_manifest=dict(path=str(codec_root/'manifest.json'),sha256=sha(codec_root/'manifest.json')),
        build_files=build_files,controllers={str(p):sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'schedule.json',jobs)
    info['files']={str(p.relative_to(out)):sha(p) for folder in (scripts,provenance) for p in folder.rglob('*') if p.is_file()}
    info['files']['schedule.json']=sha(out/'schedule.json'); write(out/'submission.json',info)
    return out
