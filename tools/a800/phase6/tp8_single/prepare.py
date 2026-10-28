"""Freeze the existing six-policy TP8 model measurement with MLP-only mappings."""
import argparse,ast,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
SOURCE=REPO/'outputs/a800/arrival-v2-tp8-20261003/artifacts/e2e-scripts'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(args):
 root=args.out.resolve();build=args.build.resolve();manifest=json.loads((build/'manifest.json').read_text())
 for name,h in manifest['files'].items():assert sha(build/name)==h,name
 root.mkdir(parents=True,exist_ok=False);scripts=root/'scripts';scripts.mkdir()
 for name in ['worker.py','adapter.py','launch.py','summarize.py','config.json']:shutil.copy2(SOURCE/name,scripts/name)
 config=json.loads((scripts/'config.json').read_text());config['scope']='Six BF16 policies on single-node TP8, DP1; MLP-only base/arrival order, original Flux attention; no quantization, tuning, speed-based admission or convergence claim.'
 config['cases'][0]['name']='l12-h2048-s2048-mb1-gb4-tp8-bf16'
 (scripts/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 p=scripts/'launch.py';text=p.read_text();old="str({'original':0,'remote_first':1,'interleaved':2,'remote_arrival':3,'interleaved_arrival':4}[policy])"
 assert text.count(old)==1;text=text.replace(old,"str(0 if k==MODEL['hidden'] else {'original':0,'remote_first':1,'interleaved':2,'remote_arrival':3,'interleaved_arrival':4}[policy])")
 p.write_text(text)
 p=scripts/'worker.py';text=p.read_text().replace('import sys\n','import sys\nimport socket\n',1)
 marker="if POLICY != 'native':\n";assert text.count(marker)==1
 audit="""assert args.padded_vocab_size == 9216
report.update(host=socket.gethostname(), local_rank=local_rank,
    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
    dp_rank=parallel_state.get_data_parallel_rank(),
    tp_group_ranks=dist.get_process_group_ranks(parallel_state.get_tensor_model_parallel_group()),
    dp_group_ranks=dist.get_process_group_ranks(parallel_state.get_data_parallel_group()))
"""
 text=text.replace(marker,audit+marker,1);p.write_text(text)
 for name in ['run.py','report.py']:shutil.copy2(HERE/name,root/name)
 for p in root.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 prior=json.loads((REPO/'logs/a800/phase6/tp8-bf16-measure-20261004/submission.json').read_text())
 protected={name:sha(name) for name in prior['protected_sources']}
 protected.update({str(SOURCE/name):sha(SOURCE/name) for name in ['worker.py','adapter.py','launch.py','summarize.py','config.json']})
 spec=dict(job_id=args.job_id,node=args.node,repo=str(REPO),build=str(build),case=config['cases'][0]['name'],python=str(REPO.parents[1]/'conda_envs/flux-megatron-a800/bin/python'),build_manifest_sha256=sha(build/'manifest.json'),protected_sources=protected,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},scope=config['scope'],source_scripts=str(SOURCE),source_plan_sha256=manifest['plan_sha256'])
 (root/'submission.json').write_text(json.dumps(spec,indent=2)+'\n');print(root)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--build',type=Path,required=True);p.add_argument('--job-id',type=int,required=True);p.add_argument('--node',required=True);main(p.parse_args())
