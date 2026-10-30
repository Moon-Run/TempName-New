"""Freeze ablation workers, physical masks, schedule, and exact build provenance."""
import argparse
from pathlib import Path
import shutil
import subprocess
import uuid

import candidate
from ablation_plan import PROTOCOL, definitions, schedule, experiment
from common import HERE, REPO, MEGATRON_COMMIT, read, replace_once, require, sha, verify_files, write
from prepare import model_worker


def megatron_check(path):
    require(subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() ==
            MEGATRON_COMMIT, 'Wrong Megatron commit')
    require(not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'],
                                        text=True).strip(), 'Megatron tracked files must be clean')


def taco_source():
    text = (REPO/'tools/a800/phase5/e2e/taco_support.py').read_text()
    text = replace_once(text, 'from routing import base_policy, codec_placement, selection_entry',
                        'from routing import base_policy, codec_placement, selection_entry, definition')
    # Every G group follows precisely the old selective double-buffered route.
    text = text.replace("policy.endswith('_selective')", "policy not in ('native', 'native_taco')")
    text = replace_once(text, '        from flux.gemm_rs_taco import GemmRSTaco\n',
                        "        if policy in ('native', 'native_taco'):\n"
                        "            raise ValueError('Native baseline cannot construct a Flux candidate')\n"
                        "        from flux.gemm_rs_taco import GemmRSTaco\n")
    text = replace_once(text, 'placement=codec_placement(policy),',
                        "placement=codec_placement(policy),\n"
                        "                               quantization_mode=definition(policy)['quantization_mode'],")
    return text


def worker_source():
    text = model_worker()
    # Explicit gate also protects direct worker invocation. Test/dry-run never imports it.
    text = replace_once(text, "local_rank = int(os.environ['LOCAL_RANK'])",
                        "if os.environ.get('H100_ABLATION_EXECUTE') != '1':\n"
                        "    raise RuntimeError('GPU worker requires explicit ablation --execute')\n\n"
                        "local_rank = int(os.environ['LOCAL_RANK'])")
    text = replace_once(text, "report['initial_parameters_sha256'] = fingerprint(raw.named_parameters())",
                        "report['ablation'] = PLAN['ablation_definitions'][POLICY]\n"
                        "report['derived_plan_sha256'] = PLAN['derived_plan_sha256'][POLICY]\n"
                        "report['parent_plan_sha256'] = PLAN['parent_plan_sha256']\n"
                        "report['initial_parameters_sha256'] = fingerprint(raw.named_parameters())")
    return text


def prepare(out, builds, megatron, blocks=10, schedule_seed=20261013, source_only=False):
    builds = Path(builds).resolve()
    data = candidate.verify(builds, built=not source_only)
    protocol_spec = experiment(data['protocol'])
    definitions, schedule = protocol_spec.definitions, protocol_spec.schedule
    c = data['config']
    megatron = Path(megatron).resolve()
    if not source_only:
        megatron_check(megatron)
    out = Path(out).resolve()
    require(type(schedule_seed) is int and 0 <= schedule_seed < 2**32, 'Invalid schedule seed')
    # Validate the schedule before creating any files.
    schedules = [row for b in data['bases'] for row in
                 schedule(b, definitions(b, data['seeds']), blocks, schedule_seed)]
    out.mkdir(parents=True, exist_ok=False)
    (out/'results').mkdir()
    (out/'provenance').mkdir()
    shutil.copy2(builds/'ablation-manifest.json', out/'provenance/ablation-manifest.json')
    if protocol_spec.PROTOCOL != PROTOCOL:
        shutil.copy2(builds/'plans/parent.json', out/'provenance/parent-plan.json')
        (out/'provenance/controllers').mkdir()
        for path in HERE.glob('*.py'):
            if not path.name.startswith('test_'):
                shutil.copy2(path, out/'provenance/controllers'/path.name)
    info = dict(protocol=protocol_spec.PROTOCOL, run_id=uuid.uuid4().hex, config=c, bases=data['bases'],
                seeds=data['seeds'], blocks_per_round=blocks, schedule_seed=schedule_seed,
                schedule=schedules, timing_windows=len(schedules),
                source_only=source_only, executable=not source_only, gpu_validated=False,
                megatron=str(megatron), megatron_commit=MEGATRON_COMMIT,
                parent_freeze=data['parent_freeze'], parent_plan_sha256=data['parent_plan']['sha256'],
                candidate_source=data['candidate_source'], calibration_hardware=data['calibration_hardware'],
                variants={}, files={}, build_files={}, source_manifests={},
                controllers={str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    for base in data['bases']:
        scripts = out/'scripts'/base
        scripts.mkdir(parents=True)
        defs = definitions(base, data['seeds'])
        for name in ('adapter.py', 'megatron_taco.py', 'taco_support.py'):
            shutil.copy2(REPO/'tools/a800/phase5/e2e'/name, scripts/name)
        if protocol_spec.PROTOCOL != PROTOCOL:
            # Exactly the already validated H100 baseline adaptation from prepare.py.
            baseline = scripts/'megatron_taco.py'
            baseline.write_text(replace_once(baseline.read_text(), 'self.world==4', 'self.world in (4,8)'))
            for name in ('adapter.py', 'megatron_taco.py', 'taco_support.py', 'worker.py'):
                shutil.copy2(REPO/'tools/a800/phase5/e2e'/name, out/'provenance'/('baseline-'+name))
        worker = worker_source()
        if source_only and protocol_spec.PROTOCOL != PROTOCOL:
            worker = "raise RuntimeError('Source-only rehearsal worker is permanently non-executable')\n"+worker
        (scripts/'worker.py').write_text(worker)
        (scripts/'taco_support.py').write_text(taco_source())
        shutil.copy2(HERE/'ablation_routing.py', scripts/'routing.py')
        shutil.copy2(HERE/'runtime.py', scripts/'runtime.py')
        shutil.copy2(REPO/'tools/a800/phase4/reference.py', scripts/'reference.py')
        cfg = read(REPO/'tools/a800/phase5/e2e/config.json')
        cfg.update(policies=list(defs), orders=[], repetitions=2, warmup_steps=10, timed_steps=20,
                   cases=[dict(name='h100-phase6', **{k: c[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
                   common={k: c[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')},
                   graph_forward=False, double_buffered=True, ring_reduction=True,
                   ablation_definitions=defs, derived_plan_sha256={}, parent_plan_sha256=info['parent_plan_sha256'],
                   scope='GPT 6.7B TP8 full optimizer step ablation; frozen V2 MLP, V3 attention; BF16 STE.')
        if protocol_spec.PROTOCOL != PROTOCOL:
            cfg['scope'] = ('GPT 6.7B TP8 full optimizer step: native Megatron tensor TACO baseline; '
                            'interleaved G1-G4 share frozen V2 MLP, V3 attention and BF16 STE.')
        selected = dict(protocol=protocol_spec.PROTOCOL, entries={})
        for key, definition in defs.items():
            identity = base+'-'+key
            v = dict(definition, policy=key)
            cfg['derived_plan_sha256'][key] = None
            if key not in ('native', 'native_taco'):
                spec = data['variants'][identity]
                root = builds/spec['build']
                manifest_path = root/'manifest.json'
                m = read(manifest_path)
                library = root/identity
                v.update(root=str(library), role='candidate', sampling=False,
                         derived_plan_sha256=spec['derived_plan_sha256'],
                         source_sha256=m['variants'][identity]['source_sha256'],
                         libraries={})
                info['source_manifests'][str(manifest_path)] = sha(manifest_path)
                shutil.copy2(manifest_path, out/'provenance'/f'{identity}-manifest.json')
                for relative, digest in m['files'].items():
                    info['build_files'][str(root/relative)] = digest
                for relative, digest in m['variants'][identity]['source_sha256'].items():
                    info['build_files'][str(library/relative)] = digest
                if not source_only:
                    for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                        path = (library/name).resolve()
                        require(path.is_file(), f'Missing library: {path}')
                        v['libraries'][name] = dict(path=str(path), sha256=sha(path))
                link = out/'build'/base/key
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(library, target_is_directory=True)
                selected['entries'][key] = read(builds/spec['plan'])['entry']
                cfg['derived_plan_sha256'][key] = spec['derived_plan_sha256']
            info['variants'][identity] = v
        write(scripts/'selection-plan.json', selected)
        write(scripts/'config.json', cfg)
    write(out/'schedule.json', schedules)
    if protocol_spec.PROTOCOL != PROTOCOL:
        write(out/'provenance/preparation.json', dict(protocol=protocol_spec.PROTOCOL, source_only=source_only,
              executable=not source_only, builds=str(builds), builds_sha256=sha(builds/'ablation-manifest.json'),
              run_id=info['run_id']))
    info['files'] = {str(p.relative_to(out)): sha(p) for folder in ('scripts', 'provenance')
                     for p in (out/folder).rglob('*') if p.is_file()}
    info['files']['schedule.json'] = sha(out/'schedule.json')
    write(out/'submission.json', info)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('out', type=Path); p.add_argument('--builds', type=Path, required=True)
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--blocks', type=int, default=10); p.add_argument('--schedule-seed', type=int, default=20261013)
    p.add_argument('--source-only', action='store_true', help='CPU rehearsal only; permanently non-executable')
    a = p.parse_args()
    print(prepare(a.out, a.builds, a.megatron, a.blocks, a.schedule_seed, a.source_only))
