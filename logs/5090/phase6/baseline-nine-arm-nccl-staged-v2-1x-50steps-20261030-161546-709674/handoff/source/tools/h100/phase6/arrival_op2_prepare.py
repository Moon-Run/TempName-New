"""Freeze a same-codec scheduling selection before independent 10+50 training windows."""
from pathlib import Path
import shutil
import uuid

import arrival_op2_build as builder
import arrival_op2_protocol as routing
import arrival_op2_probe as probe
import arrival_op2_sources as sources
import ablation4_runtime as communication
from ablation_prepare import megatron_check
from common import HERE, REPO, MEGATRON_COMMIT, read, require, replace_once, sha, write
from scope_plan import model_config


def role_hashes(plans):
    return {p: {r: record['entry_sha256'] for r, record in routing.role_plans(plans, p).items()}
            for p in routing.POLICIES}


def prepare(out, build_root, megatron, blocks=6, rounds=2, nvls='0', source_only=False, selection=None):
    build_root, megatron = Path(build_root).resolve(), Path(megatron).resolve()
    jobs = routing.schedule(blocks, rounds=rounds)
    manifest = builder.verify(build_root, built=not source_only)
    require(manifest['source_only'] == source_only, 'Cannot mix executable builds and CPU rehearsals')
    if not source_only:
        megatron_check(megatron)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scripts, provenance = out/'scripts', out/'provenance'
    for folder in (scripts, provenance/'controllers', out/'results', out/'build'):
        folder.mkdir(parents=True)
    if selection is None:
        require(source_only, 'A completed same-codec GPU probe --selection is required before preparation')
        selection = provenance/'selection'/'selection.json'
        selection.parent.mkdir(parents=True)
        probe.make_source_only_selection(selection, build_root)
    selected = probe.selection_verify(selection, build_root)
    require(selected['source_only'] == source_only and selected['gpu_measured'] == (not source_only),
            'Probe/compare source-only identity differs; CPU results cannot select GPU scheduling')
    frozen_selection = provenance/'selection'/'selection.json'
    frozen_selection.parent.mkdir(parents=True, exist_ok=True)
    selection = Path(selection).resolve()
    for relative, expected_sha in selected['evidence_files'].items():
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and '..' not in relative_path.parts,
                'Probe evidence must be relative and remain inside its evidence directory')
        source, target = selection.parent/relative_path, frozen_selection.parent/relative_path
        require(sha(source) == expected_sha, 'Probe evidence changed during preparation')
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    if selection != frozen_selection.resolve():
        shutil.copy2(selection, frozen_selection)
    shutil.copy2(build_root/'arrival-op2-manifest.json', frozen_selection.parent/'arrival-op2-manifest.json')
    template = read(build_root/'plans.json')
    plans = builder.planner.bundle(template['parent'], source_only=source_only,
                                  spans=selected['selected_spans'])
    write(scripts/'selection-plan.json', plans)
    for path in HERE.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, provenance/'controllers'/path.name)
    (scripts/'adapter.py').write_text(sources.adapter_source())
    (scripts/'taco_support.py').write_text(sources.taco_source())
    (scripts/'worker.py').write_text(sources.worker_source(source_only))
    native = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
    (scripts/'megatron_taco.py').write_text(replace_once(native, 'self.world==4', 'self.world in (4,8)'))
    for src, name in ((HERE/'arrival_op2_protocol.py', 'routing.py'), (HERE/'runtime.py', 'runtime.py'),
                      (HERE/'ablation4_runtime.py', 'communication.py'),
                      (REPO/'tools/a800/phase4/reference.py', 'reference.py')):
        shutil.copy2(src, scripts/name)
    snapshots = {'build-manifest.json': 'arrival-op2-manifest.json',
                 'build-plans.json': 'plans.json',
                 'parent-ablation4-manifest.json': 'parent-ablation4-manifest.json',
                 'parent-candidate-manifest.json': 'parent-candidate-manifest.json',
                 'parent-fused-manifest.json': 'parent-fused-manifest.json',
                 'parent-plans.json': 'parent-plans.json',
                 'improved-manifest.json': 'improved/manifest.json'}
    for name, relative in snapshots.items():
        shutil.copy2(build_root/relative, provenance/name)

    options = builder.planner.frozen_options(plans)
    option_fields = dict(schedule_options=options)
    model, timing, comm = model_config(), routing.execution_timing(), communication.capture(nvls)
    budgets = {p: routing.definition(p)['budget_fraction'] for p in routing.POLICIES}
    config = read(REPO/'tools/a800/phase5/e2e/config.json')
    config.update(protocol=routing.PROTOCOL, policies=list(routing.POLICIES), orders=[], repetitions=rounds, **option_fields,
        **timing, execution_timing=timing, graph_forward=False, double_buffered=False, ring_reduction=True,
        cases=[dict(name='h100-phase6', **{k: model[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
        common={k: model[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
        role_plan_sha256=role_hashes(plans), communication=comm, budget_fractions=budgets,
        scope='Arrival-op2: A is pinned all-remote interleaving; S changes only paired-row tile order. '
              'Both use identical codec, CTA decoder, one workspace, clone and both barriers.')
    write(scripts/'config.json', config)
    run_id = uuid.uuid4().hex
    write(provenance/'preparation.json', dict(protocol=routing.PROTOCOL, run_id=run_id, **option_fields,
        source_only=source_only, executable=not source_only, rounds=rounds, communication=comm,
        execution_timing=timing, plans_sha256=sha(scripts/'selection-plan.json'),
        build_plans_sha256=sha(build_root/'plans.json'), selection_sha256=sha(frozen_selection)))

    variants = {}
    for policy in routing.POLICIES:
        lib = builder.runtime_root(build_root, policy)
        (out/'build'/policy).symlink_to(lib, target_is_directory=True)
        variants[policy] = dict(routing.definition(policy), root=str(lib), sampling=False,
            libraries={} if source_only else {name: dict(path=str((lib/name).resolve()), sha256=sha(lib/name))
                for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')})
    build_files = dict(manifest['parent_files'])
    build_files.update({str(build_root/relative): value for relative, value in manifest['files'].items()})
    build_files[str(build_root/'arrival-op2-manifest.json')] = sha(build_root/'arrival-op2-manifest.json')
    improved = read(build_root/'improved/manifest.json')
    build_files.update({str(build_root/'improved'/p): h for p, h in improved['files'].items()})
    for name, variant in improved['variants'].items():
        build_files.update({str(build_root/'improved'/name/p): h for p, h in variant['source_sha256'].items()})
    hardware = next((r['hardware'] for r in plans['parent']['roles'].values() if not r['test_only']), {})
    info = dict(protocol=routing.PROTOCOL, run_id=run_id, config=model, execution_timing=timing, **option_fields,
        source_only=source_only, executable=not source_only, gpu_validated=False,
        rounds=rounds, blocks_per_round=blocks, schedule_seed=20261013, schedule=jobs,
        timing_windows=len(jobs), variants=variants, communication=comm, calibration_hardware=hardware,
        megatron=str(megatron), megatron_commit=MEGATRON_COMMIT,
        candidate_source_commit=manifest['parent_source_commit'], budget_fractions=budgets,
        build_manifest=dict(path=str(build_root/'arrival-op2-manifest.json'), sha256=sha(build_root/'arrival-op2-manifest.json')),
        build_files=build_files, selection_sha256=sha(frozen_selection),
        selection=dict(path='provenance/selection/selection.json', sha256=sha(frozen_selection),
                       selected_spans=selected['selected_spans'], gpu_measured=selected['gpu_measured']),
        controllers={str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'schedule.json', jobs)
    info['files'] = {str(p.relative_to(out)): sha(p) for folder in (scripts, provenance)
                     for p in folder.rglob('*') if p.is_file()}
    info['files']['schedule.json'] = sha(out/'schedule.json')
    write(out/'submission.json', info)
    return out
