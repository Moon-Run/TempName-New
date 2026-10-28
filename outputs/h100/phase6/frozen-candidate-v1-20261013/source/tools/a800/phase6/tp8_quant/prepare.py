"""Freeze a TP8 six-policy run; adapt validation domains without modifying TP4."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COMMON = HERE.parents[1]/'phase5/e2e'
sys.path.insert(0, str(HERE.parent))
from model_config import DEFAULT_MODEL, read_model, mlp_shape, padded_vocab
POLICIES = ['native', 'original', 'native_taco', 'taco_fused',
            'remote_arrival_selective', 'interleaved_arrival_selective']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def main(args):
    root, candidates, plan_path = args.out.resolve(), args.build.resolve(), args.plan.resolve()
    assert REPO/'logs/a800' in root.parents
    plan = json.loads(plan_path.read_text())
    assert plan['schema'] == 2 and plan['world'] == 8
    profile = getattr(args, 'model_config', None)
    model = read_model(profile, 8) if profile else dict(layers=12,hidden=2048,ffn=8192,
        heads=32,sequence=2048,micro_batch=1,global_batch=4,seed=1234,token_seed=4321,vocab=8192)
    m, n, k = mlp_shape(model)
    assert plan['shape_catalog'] == [[m,n,k]], 'Recalibrate arrival and quantization masks for the requested model'
    for entries in plan['policies'].values():
        assert len(entries)==1 and entries[0]['shape']==[m,n,k] and entries[0]['K_local']==k//8
    assert set(plan['policies']) == {'remote_first', 'interleaved'}
    for name, digest in plan['inputs'].items():
        assert sha(name) == digest, name
    root.mkdir(parents=True, exist_ok=False)
    scripts = root/'scripts'
    scripts.mkdir()
    (root/'results').mkdir()
    artifact = REPO/'outputs/a800/phase6'/root.name
    build = artifact/'build'
    (build/'mapping').mkdir(parents=True, exist_ok=False)
    stock = REPO/'outputs/a800/phase3/three-way-build'
    fused = REPO/'outputs/a800/phase4/taco-fused-warp-20261003'
    manifest = dict(files={}, policies={}, source_manifests={}, scope='Single-node TP8; original frozen fused-v2 baseline and separately built MLP candidates')
    for policy in POLICIES:
        if policy in ('native', 'native_taco'):
            continue
        mapping = policy.removesuffix('_selective') if policy.endswith('_selective') else 'original'
        source = stock/'original' if policy == 'original' else fused/'taco' if policy == 'taco_fused' else candidates/mapping/'taco'
        parent = source.parent
        src = json.loads((parent/'manifest.json').read_text())
        for name, h in src['files'].items():
            assert sha(parent/name) == h, (policy, name)
        if policy == 'taco_fused':
            assert src['placement'] == 'fused' and not src['reorder']
            assert src['original_swizzle_sha256'] == sha(stock/'original/overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp')
        elif policy != 'original':
            assert src['world'] == 8 and src['arrival_plan_sha256'] == sha(plan_path)
            assert src['mapping'] == mapping
        (build/policy).symlink_to(source, target_is_directory=True)
        prefix = source.name+'/'
        manifest['files'].update({policy+'/'+name[len(prefix):]: h for name, h in src['files'].items() if name.startswith(prefix)})
        # Hash the resolved loader paths as well as the original manifest members.
        for name in ['libflux_cuda.so', 'libflux_cuda_ths_op.so', 'python/flux/lib/libflux_cuda.so', 'python/flux/lib/libflux_cuda_ths_op.so']:
            manifest['files'][policy+'/'+name] = sha(source/name)
        manifest['policies'][policy] = dict(library_sha256=sha(source/'libflux_cuda.so'), mapping=mapping)
        manifest['source_manifests'][policy] = dict(path=str(parent/'manifest.json'), sha256=sha(parent/'manifest.json'))
        if policy.endswith('_selective'):
            (build/'mapping'/mapping).symlink_to(parent/'check_mapping')
            manifest['files']['mapping/'+mapping] = sha(parent/'check_mapping')
    (build/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    sources = {}
    def snapshot(source, target):
        sources[str(source)] = sha(source)
        shutil.copy2(source, target)
    for name in ['adapter.py', 'worker.py', 'launch.py', 'summarize.py', 'routing.py', 'taco_support.py', 'megatron_taco.py', 'config.json']:
        snapshot(COMMON/name, scripts/name)
    p = scripts/'megatron_taco.py'
    p.write_text(replace(p.read_text(), 'self.world==4', 'self.world==8'))
    cfg = json.loads((scripts/'config.json').read_text())
    cfg.update(policies=POLICIES, orders=[POLICIES[i:]+POLICIES[:i] for i in range(6)],
               cases=[dict(name=f'l{model["layers"]}-h{n}-s{model["sequence"]}-mb{model["micro_batch"]}-gb{model["global_batch"]}-tp8-quant',
                           hidden=n, ffn=k, heads=model['heads'], sequence=model['sequence'], tp=8)],
               scenario_extension=True, graph_forward=False, double_buffered=True, selective_decoder='legacy',
               base_orders=['remote_first', 'interleaved'],
               acceptance=dict(version='paired-full-step-v2-20261004', target_percent=4.),
               scope='Single-node TP8 compatibility and measurement only; existing phase6 window64/budget1/64 rules, generic decoder and double-buffered workspaces. No kernel tuning or speed admission gate. Synthetic full optimizer steps; no convergence claim.')
    cfg['common'] = {key:model[key] for key in ('layers','micro_batch','global_batch','seed','token_seed','vocab')}
    cfg['expected_padded_vocab_size'] = padded_vocab(model, 8)
    if profile:
        snapshot(profile, scripts/'model-profile.json')
    (scripts/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    for name in ['selection-plan.json', 'arrival-plan.json']:
        shutil.copy2(plan_path, scripts/name)
    p = scripts/'worker.py'
    text = replace(p.read_text(), 'import sys\n', 'import sys\nimport socket\n')
    text = replace(text, "if POLICY not in ('native','native_taco'):\n", """assert args.padded_vocab_size == PLAN['expected_padded_vocab_size']
report.update(host=socket.gethostname(), local_rank=local_rank,
    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
    dp_rank=parallel_state.get_data_parallel_rank(),
    tp_group_ranks=dist.get_process_group_ranks(parallel_state.get_tensor_model_parallel_group()),
    dp_group_ranks=dist.get_process_group_ranks(parallel_state.get_data_parallel_group()))
if POLICY not in ('native','native_taco'):
""")
    p.write_text(text)
    for source, name in [(HERE.parents[1]/'phase5/validate.py', 'validate.py'),
                         (HERE.parents[1]/'phase5/validate_megatron.py', 'validate_megatron.py'),
                         (HERE.parents[1]/'phase4/validate.py', 'validate_separate.py'),
                         (HERE.parents[1]/'phase4/reference.py', 'reference.py')]:
        snapshot(source, scripts/name)
    p = scripts/'validate.py'
    text = replace(p.read_text(), 'world == 4', 'world == 8')
    text = replace(text, 'model_shape or (2048,2048,2048)', 'model_shape or (2048,2048,1024)')
    text = replace(text, '[(512,128,256),(512,136,256),(512,256,256)', '[(1024,128,256),(1024,136,256),(1024,256,256)')
    p.write_text(text)
    p = scripts/'validate_megatron.py'
    text = replace(p.read_text(), "int(os.environ['WORLD_SIZE'])==4", "int(os.environ['WORLD_SIZE'])==8")
    text = replace(text, 'range(4)', 'range(8)', 2)
    text = replace(text, '[(512,128,256),(512,136,256),(512,256,256),(2048,2048,512),(2048,2048,2048)]',
                   repr([(1024,128,256),(1024,136,256),(1024,256,256),(m,n,n//8),(m,n,k//8)]))
    p.write_text(text)
    p = scripts/'validate_separate.py'
    text = replace(p.read_text(), "assert world==4, 'Frozen model-shape preflight is TP4 only'", "assert world==8, 'This frozen model-shape preflight is TP8 only'")
    text = replace(text, '[(2048,2048,512),(2048,2048,2048)]', repr([(m,n,n//8),(m,n,k//8)]))
    p.write_text(text)
    snapshot(COMMON/'verify.py', root/'verify.py')
    p = root/'verify.py'
    text = replace(p.read_text(), 'range(4)', 'range(8)', 2)
    text = replace(text, 'records == 4*windows', 'records == 8*windows')
    text = replace(text, "                model=r['model'];", """                assert r['world_size'] == 8 and r['rank'] == r['local_rank'] == r['tp_rank'] == rank
                assert r['job_id'] == str(info['job_id']) and r['host'] == info['node']
                assert r['dp_rank'] == 0 and r['tp_group_ranks'] == list(range(8)) and r['dp_group_ranks'] == [rank]
                assert r['padded_vocab_size'] == config['expected_padded_vocab_size']
                if r['policy'] not in ('native','native_taco'):
                    for lib in ['libflux_cuda.so','libflux_cuda_ths_op.so']:
                        expected = (Path(info['artifact_root'])/'build'/r['policy']/lib).resolve()
                        assert r[lib]['path'] == str(expected)
                        assert r[lib]['sha256'] == hashlib.sha256(expected.read_bytes()).hexdigest()
                model=r['model'];""")
    text = replace(text, "    if 'native_taco' in info['policies']:\n", """    for name, digest in info['protected_sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    result['single_node_tp8_topology_and_loaded_libraries_checked'] = True
    result['tp4_protected_sources_unchanged'] = True
    if 'native_taco' in info['policies']:
""")
    p.write_text(text)
    snapshot(COMMON/'campaign.py', root/'campaign.py')
    p = root/'campaign.py'
    text = replace(p.read_text(), '--nproc_per_node=4', '--nproc_per_node=8', 2)
    text = replace(text, 'range(4)', 'range(8)', 2)
    text = replace(text, "assert os.environ.get('SLURM_JOB_ID') == str(INFO['job_id'])", """assert os.environ.get('SLURM_JOB_ID') == str(INFO['job_id'])
assert not (ROOT/'campaign-state.json').exists(), 'Use a fresh result directory'
snapshot = subprocess.check_output(['squeue','--steps','-h','-j',str(INFO['job_id']),'-o','%i'],text=True)
allowed = {f"{INFO['job_id']}.batch", f"{INFO['job_id']}.extern", f"{INFO['job_id']}.{os.environ['SLURM_STEP_ID']}"}
assert set(snapshot.split()) <= allowed, snapshot
(ROOT/'initial-slurm-steps.txt').write_text(snapshot)
""")
    text = replace(text, "    for path, digest in INFO['files'].items():", "    for path, digest in INFO['protected_sources'].items():\n        assert sha(Path(path)) == digest, path\n    for path, digest in INFO['files'].items():")
    text = replace(text, "    state.update(status='completed', stage='completed', finished_at=now())\n    save()", """    state.update(status='completed', stage='completed', finished_at=now())
    save()
    execute('verify-and-assess', [PYTHON, ROOT/'assess.py', ROOT],
            dict(base_env(), PYTHONPATH=str(ROOT/'scripts')), ROOT)
""")
    p.write_text(text)
    for source, target in [(HERE/'check_mappings.py', root/'check_mappings.py'),
                           (HERE/'report.py', root/'report.py'),
                           (HERE.parent/'assess.py', root/'assess.py')]:
        snapshot(source, target)
    for p in root.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    archive = REPO/'outputs/a800/archives/tp4-s1024-mb8-20261006/archive-manifest.json'
    # New model runs freeze their own inputs; the old TP4 archive remains an
    # independent historical replay, not a dependency on mutable old scripts.
    protected = {} if profile else json.loads(archive.read_text())['protected_original_files']
    protected.update(sources)
    for name, digest in protected.items():
        assert sha(name) == digest, name
    info = dict(job_id=args.job_id, node=args.node, repo=str(REPO), artifact_root=str(artifact),
                files={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()},
                protected_sources=protected, build_manifest_sha256=sha(build/'manifest.json'),
                policies=POLICIES, mappings=['remote_arrival', 'interleaved_arrival'],
                codec_checks=['remote_arrival', 'interleaved_arrival', 'fused', 'native_taco'],
                repetitions=2, blocks=6, scope=cfg['scope'],
                original_campaign=str(REPO/'outputs/a800/arrival-v2-tp8-20261003'))
    (root/'submission.json').write_text(json.dumps(info, indent=2)+'\n')
    print(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--model-config', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--job-id', type=int, required=True)
    p.add_argument('--node', required=True)
    main(p.parse_args())
