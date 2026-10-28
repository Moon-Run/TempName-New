"""Freeze a TP4/DP2 campaign without rebuilding or editing single-node artifacts."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

from protocol import POLICIES, sha, topology, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0,str(HERE.parent))
from model_config import DEFAULT_MODEL, read_model, mlp_shape


def main(args):
    out = args.out.resolve()
    assert REPO/'logs/a800' in out.parents
    profile = getattr(args,'model_config',None)
    model = read_model(profile,4) if profile else None
    if model and not getattr(args,'source',None):
        raise ValueError('GPT 6.7B DP2 requires --source pointing to a fresh prepared 6.7B TP4 run')
    source = Path(args.source).resolve() if getattr(args,'source',None) else REPO/'logs/a800/phase6/tp4-opt2-s1024-confirm-nodea-20261004'
    base = json.loads((source/'scripts/config.json').read_text())
    if model:
        previous = dict(base['common'],**base['cases'][0])
        assert all(previous[key]==value for key,value in model.items()), 'TP4 source has a different model profile'
        selected=json.loads((source/'scripts/selection-plan.json').read_text())
        assert selected['world']==4
        assert all(any(entry['shape']==mlp_shape(model) for entry in entries) for entries in selected['policies'].values())
    old_submission = json.loads((source/'submission.json').read_text())
    build = Path(old_submission['artifact_root'])/'build'
    manifest = json.loads((build/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(build/name) == digest, name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO.parent/'Megatron-LM', text=True).strip() == '3ea68ad6042cc1204386ae9364358f7c4de1bc37'
    out.mkdir(parents=True, exist_ok=False)
    scripts = out/'scripts'
    scripts.mkdir()
    common = REPO/'tools/a800/phase5/e2e'
    for name in ('adapter.py', 'routing.py', 'taco_support.py', 'megatron_taco.py'):
        shutil.copy2(common/name, scripts/name)
    shutil.copy2(common/'summarize.py', scripts/'decoder_protocol.py')
    shutil.copy2(REPO/'tools/a800/phase4/reference.py', scripts/'reference.py')
    shutil.copy2(source/'scripts/selection-plan.json', scripts/'selection-plan.json')
    shutil.copy2(build/'manifest.json', out/'build-manifest.json')
    for path in HERE.glob('*.py'):
        shutil.copy2(path, scripts/path.name)
    base.update(dp=2, world_size=8, nodes=2, gpus_per_node=4,
                scope='Two nodes, node-local TP4 and cross-node DP2; MLP-only frozen implementations; full optimizer steps including DP gradients; no performance tuning or convergence claim.')
    assert base['policies'] == POLICIES
    cases = []
    for seq in (args.sequences or ([model['sequence']] if model else [1024,2048])):
        cfg = json.loads(json.dumps(base))
        if model:
            assert seq==model['sequence'], 'Prepare/calibrate a separate TP4 source for a different sequence'
            mb=model['micro_batch'];gb=2*model['global_batch']
            hidden,ffn,heads,layers=(model[k] for k in ('hidden','ffn','heads','layers'))
        else:
            mb=8192//seq;gb=mb*2;hidden=2048;ffn=8192;heads=32;layers=12
        cfg['common'].update(micro_batch=mb, global_batch=gb)
        cfg['cases'] = [dict(name=f'l{layers}-h{hidden}-s{seq}-mb{mb}-gb{gb}-tp4-dp2',
                             hidden=hidden, ffn=ffn, heads=heads, sequence=seq, tp=4)]
        name = cfg['cases'][0]['name']
        case = out/name
        case.mkdir()
        write_json(case/'config.json', cfg)
        cases.append(dict(name=name, config=str(case/'config.json')))
    # Codec support only reads execution flags; worker reads its scenario explicitly.
    write_json(scripts/'config.json', base)
    for path in scripts.glob('*.py'):
        ast.parse(path.read_text())
    protected = [REPO/'python/flux/gemm_rs_taco.py', REPO/'src/gemm_rs/taco_runtime.cu',
                 REPO/'src/gemm_rs/taco_runtime.h', REPO/'src/gemm_rs/ths_op/gemm_reduce_scatter.cc']
    protected += list(common.glob('*.py'))
    protected += [REPO/'tools/a800/phase6/prepare_scenario.py']
    spec = dict(repo=str(REPO), build=str(build), jobs=args.jobs,
                nodes=args.nodes, master_addr=args.nodes[0], port_base=args.port_base,
                python=str(REPO.parents[1]/'conda_envs/flux-megatron-a800/bin/python'),
                topology=topology(8,4,4,2), cases=cases, policies=POLICIES,
                original_build_manifest_sha256=sha(build/'manifest.json'),
                protected_sources={str(p): sha(p) for p in protected},
                files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
                calibration=f'Frozen TP4 calibration from {source}; not recalibrated under DP2 load',
                scope=base['scope'])
    write_json(out/'submission.json', spec)
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--jobs', type=int, nargs=2, required=True)
    p.add_argument('--nodes', nargs=2, required=True)
    p.add_argument('--model-config',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--source',type=Path,help='Prepared TP4 run with the same model and fresh plan')
    p.add_argument('--sequences', type=int, nargs='+', choices=[1024,2048])
    p.add_argument('--port-base', type=int, default=29440)
    main(p.parse_args())
