"""Freeze budget masks and reuse exactly one existing C runtime for all groups."""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import ablation5_plan as planner
import ablation5_protocol as routing
import ablation4_runtime as communication
import ablation4_sources
import candidate
import scope_build
import scope_sources
from ablation_plan import FREEZE
from ablation_prepare import megatron_check
from ablation_timing import execution_timing
from common import HERE, REPO, MEGATRON_COMMIT, read, require, replace_once, sha, verify_files, write
from scope_plan import model_config


def json_sha(value):
    return hashlib.sha256((json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode()).hexdigest()


def candidate_snapshot(manifest, parent, source_only, built=False):
    require(manifest['protocol'] == scope_build.PROTOCOL and manifest['kind'] == 'scope_candidate' and
            manifest['config'] == model_config() and set(manifest['variants']) == {'scope-candidate'} and
            manifest['source_only'] == manifest['diagnostic_only'] == source_only and
            manifest['files']['plans.json'] == json_sha(parent), 'Wrong shared C build/parent')
    v = manifest['variants']['scope-candidate']
    require(v['role'] == 'candidate' and v['base'] == 'interleaved' and not v['sampling'], 'Sampling/fused baseline cannot replace C')
    freeze = read(FREEZE/'manifest.json'); root = REPO/freeze['candidate']['build']
    require(sha(root/'manifest.json') == freeze['candidate']['manifest_sha256'], 'Frozen C source manifest changed')
    measured = read(root/'manifest.json'); original = measured['variants']['interleaved_arrival_selective']['source_sha256']
    require(manifest['parent_source_commit'] == measured['source_commit'] == freeze['candidate']['source_commit'], 'Wrong C source commit')
    python = 'python/flux/gemm_rs_taco.py'; src = root/'interleaved_arrival_selective'
    verify_files(src, {python: original[python]})
    verify_files(FREEZE/'source', {scope_build.SWIZZLE: freeze['source_files'][scope_build.SWIZZLE]['sha256']})
    entries = {r: p['entry'] for r, p in parent['roles'].items()}
    overrides = {python: candidate.codec_source((src/python).read_text()),
                 scope_build.SWIZZLE: scope_build.swizzle((FREEZE/'source'/scope_build.SWIZZLE).read_text(), entries),
                 'phase6_host.hpp': '#pragma once\n'+scope_build.tables(entries)}
    expected = dict(original, **{p: hashlib.sha256(text.encode()).hexdigest() for p, text in overrides.items()})
    require(v['source_sha256'] == expected, 'C compute/codec/wrapper/arrival sources changed')
    if built: require(manifest['status'] == 'built' and not source_only, 'Real compiled C runtime required')


def prepare_plan(out, build_root, source_only=False):
    build_root = Path(build_root).resolve(); out = Path(out).resolve()
    require(not out.exists(), 'Do not overwrite a budget plan')
    m = scope_build.verify(build_root, built=not source_only, kind='scope_candidate')
    parent = read(build_root/'plans.json'); candidate_snapshot(m, parent, source_only, built=not source_only)
    plan = planner.bundle(parent, source_only)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        f.write(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    return out


def role_hashes(plan):
    return {p: {r: v['entry_sha256'] for r, v in routing.role_plans(plan, p).items()} for p in routing.POLICIES}


def worker_source(source_only=False):
    source = ablation4_sources.worker_source(source_only)
    source = replace_once(source, "os.environ.get('H100_ABLATION4_EXECUTE')", "os.environ.get('H100_ABLATION5_EXECUTE')")
    return replace_once(source, 'GPU worker requires ablation4 --execute', 'GPU worker requires ablation5 --execute')


def prepare(out, build_root, plan_file, megatron, blocks=5, rounds=2, nvls='0', source_only=False):
    jobs = routing.schedule(blocks, rounds=rounds)
    build_root, plan_file, megatron = map(lambda p: Path(p).resolve(), (build_root, plan_file, megatron))
    m = scope_build.verify(build_root, built=not source_only, kind='scope_candidate')
    plan = planner.validate(read(plan_file))
    require(plan['source_only'] == source_only, 'Cannot promote artificial scores to real execution')
    require(plan['parent'] == read(build_root/'plans.json'), 'Budget masks must use this C arrival calibration')
    candidate_snapshot(m, plan['parent'], source_only, built=not source_only)
    if not source_only: megatron_check(megatron)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    scripts = out/'scripts'; provenance = out/'provenance'
    for folder in (scripts, provenance/'controllers', out/'results', out/'build'): folder.mkdir(parents=True)
    for path in HERE.glob('*.py'):
        if not path.name.startswith('test_'): shutil.copy2(path, provenance/'controllers'/path.name)
    (scripts/'adapter.py').write_text(scope_sources.adapter_source())
    (scripts/'taco_support.py').write_text(scope_sources.taco_source())
    (scripts/'worker.py').write_text(worker_source(source_only))
    native = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
    (scripts/'megatron_taco.py').write_text(replace_once(native, 'self.world==4', 'self.world in (4,8)'))
    for src, name in ((HERE/'ablation5_protocol.py', 'routing.py'), (HERE/'runtime.py', 'runtime.py'),
                      (HERE/'ablation4_runtime.py', 'communication.py'),
                      (REPO/'tools/a800/phase4/reference.py', 'reference.py'), (plan_file, 'selection-plan.json')):
        shutil.copy2(src, scripts/name)
    shutil.copy2(build_root/'manifest.json', provenance/'candidate-manifest.json')
    config = model_config(); comm = communication.capture(nvls); run_id = uuid.uuid4().hex
    cfg = read(REPO/'tools/a800/phase5/e2e/config.json')
    cfg.update(protocol=routing.PROTOCOL, policies=list(routing.POLICIES), orders=[], repetitions=rounds,
        **execution_timing(), graph_forward=False, double_buffered=True, ring_reduction=True,
        cases=[dict(name='h100-phase6', **{k: config[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
        common={k: config[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
        role_plan_sha256=role_hashes(plan), communication=comm,
        budget_fractions={p: routing.definition(p)['budget_fraction'] for p in routing.POLICIES},
        scope='Ablation5: same C library/arrival table; only the positive-score budget mask changes.')
    write(scripts/'config.json', cfg)
    write(provenance/'preparation.json', dict(protocol=routing.PROTOCOL, run_id=run_id, source_only=source_only,
        executable=not source_only, rounds=rounds, communication=comm, execution_timing=execution_timing(), plans_sha256=sha(plan_file)))
    lib = build_root/'scope-candidate'
    libraries = {} if source_only else {n: dict(path=str((lib/n).resolve()), sha256=sha(lib/n))
                                       for n in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
    variants = {}
    for policy in routing.POLICIES:
        variants[policy] = dict(routing.definition(policy), root=str(lib), sampling=False, libraries=libraries)
        (out/'build'/policy).symlink_to(lib, target_is_directory=True)
    build_files = {str(build_root/'manifest.json'): sha(build_root/'manifest.json'),
                   **{str(build_root/p): h for p, h in m['files'].items()},
                   **{str(lib/p): h for p, h in m['variants']['scope-candidate']['source_sha256'].items()}}
    hardware = next((p['hardware'] for p in plan['parent']['roles'].values() if not p['test_only']), {})
    info = dict(protocol=routing.PROTOCOL, run_id=run_id, config=config, execution_timing=execution_timing(), source_only=source_only,
        executable=not source_only, gpu_validated=False, rounds=rounds, blocks_per_round=blocks,
        schedule_seed=20261013, schedule=jobs, timing_windows=len(jobs), variants=variants,
        communication=comm, calibration_hardware=hardware, megatron=str(megatron), megatron_commit=MEGATRON_COMMIT,
        candidate_source_commit=m['parent_source_commit'], budget_denominators=list(routing.DENOMINATORS),
        build_manifest=dict(path=str(build_root/'manifest.json'), sha256=sha(build_root/'manifest.json')),
        build_files=build_files, controllers={str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'schedule.json', jobs)
    info['files'] = {str(p.relative_to(out)): sha(p) for folder in (scripts, provenance) for p in folder.rglob('*') if p.is_file()}
    info['files']['schedule.json'] = sha(out/'schedule.json'); write(out/'submission.json', info)
    return out
