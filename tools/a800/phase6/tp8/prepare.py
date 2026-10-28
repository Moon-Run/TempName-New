"""Freeze supported BF16 hierarchical TP8 comparisons in a fresh directory."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import sys

from protocol import sha,write_json,POLICIES

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]


def main(a):
    out,build=a.out.resolve(),a.build.resolve()
    assert REPO/'logs/a800' in out.parents
    manifest=json.loads((build/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(build/name)==h,name
    out.mkdir(parents=True,exist_ok=False);scripts=out/'scripts';scripts.mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,scripts/p.name)
    shutil.copy2(REPO/'tools/a800/phase6/dp2/probe.py',scripts/'probe.py')
    write_json(out/'build-manifest.json',manifest)
    cases=[]
    for seq in (1024,2048):
        mb=8192//seq;name=f'l12-h2048-s{seq}-mb{mb}-tp8-dp1-bf16'
        case=out/name;case.mkdir()
        c=dict(nodes=2,gpus_per_node=4,tp=8,dp=1,world_size=8,tp_nodes=2,local_tp=4,
               layers=12,hidden=2048,ffn=8192,heads=32,sequence=seq,micro_batch=mb,global_batch=mb,
               seed=1234,token_seed=4321,vocab=8192,padded_vocab_size=8704,
               make_vocab_size_divisible_by=64,microsteps=1,tokens_per_step=8192,
               warmup_steps=10,timed_steps=20,blocks=4,policies=POLICIES,atol=.02,rtol=.02,
               flux_transport='hierarchical',name=name,
               scope='BF16 TP8 transport measurement only. MLP-only node-local base-order variants; no arrival-priority or TACO, no quantization baseline acceptance, no tuning.')
        write_json(case/'config.json',c);cases.append(dict(name=name,config=str(case/'config.json')))
    for p in scripts.glob('*.py'):ast.parse(p.read_text())
    previous=json.loads((REPO/'logs/a800/phase6/tp4-dp2-measure-20261004/submission.json').read_text())
    protected={name:sha(name) for name in previous['protected_sources']}
    for p in (REPO/'tools/a800/phase6/dp2').glob('*.py'):protected[str(p)]=sha(p)
    spec=dict(repo=str(REPO),build=str(build),jobs=a.jobs,nodes=a.nodes,
              master_addr=a.nodes[0],port_base=30440,cases=cases,policies=POLICIES,
              python=str(REPO.parents[1]/'conda_envs/flux-megatron-a800/bin/python'),
              original_build_manifest_sha256=sha(build/'manifest.json'),protected_sources=protected,
              files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
              scope='Supported hierarchical BF16 TP8 measurement, followed separately by unchanged single-node TP4 regression. Not a complete six-policy quantization comparison.')
    write_json(out/'submission.json',spec);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--jobs',type=int,nargs=2,required=True)
    p.add_argument('--nodes',nargs=2,required=True);main(p.parse_args())
