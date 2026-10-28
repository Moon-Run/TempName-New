"""Freeze a validated TP4 runtime and prepare repeat runs from it."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(out,source):
 out,source=out.resolve(),source.resolve();info=json.loads((source/'submission.json').read_text())
 assert json.loads((source/'verification.json').read_text())['completed']
 config=json.loads((source/'scripts/config.json').read_text());assert config['cases'][0]['tp']==4 and config['common']['micro_batch']==config['common']['global_batch']
 assert len(config['policies'])==len(config['orders']) and len(config['policies'])>=6
 out.mkdir(parents=True,exist_ok=False);template=out/'template';template.mkdir()
 for name,h in info['files'].items():
  assert sha(source/name)==h,name
  p=template/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p)
 build=Path(info['artifact_root'])/'build';shutil.copytree(build,out/'artifacts/build',symlinks=False,ignore=shutil.ignore_patterns('__pycache__'))
 # Loader and runtime identity checks must resolve to the same physical library.
 # Only these internal links are restored; no link points to a mutable build.
 for policy in (out/'artifacts/build').iterdir():
  for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
   link=policy/'python/flux/lib'/name
   if link.is_file():
    assert sha(link)==sha(policy/name)
    link.unlink();link.symlink_to('../../../'+name)
 manifest=json.loads((build/'manifest.json').read_text())
 omitted=[name for name in manifest['files'] if '__pycache__' in Path(name).parts or name.endswith('.pyc')]
 manifest['files']={name:h for name,h in manifest['files'].items() if name not in omitted}
 for name,h in manifest['files'].items():assert sha(out/'artifacts/build'/name)==h,name
 if omitted:
  manifest['omitted_regenerable_bytecode']=omitted
  (out/'artifacts/build/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 shutil.copy2(source/'submission.json',out/'source-submission.json')
 for name in ['verification.json','acceptance.json','acceptance.md']:
  if (source/name).exists():shutil.copy2(source/name,out/('source-'+name))
 shutil.copy2(Path(__file__),out/'archive.py')
 protected={}
 for p in [*REPO.glob('src/gemm_rs/taco*'),REPO/'src/gemm_rs/epilogue_evt.hpp',REPO/'src/gemm_rs/ths_op/gemm_reduce_scatter.cc',REPO/'python/flux/gemm_rs_taco.py',*REPO.glob('tools/a800/phase5/e2e/*.py'),*REPO.glob('tools/a800/phase6/*.py')]:
  if p.is_file() and p.resolve()!=Path(__file__).resolve():protected[str(p)]=sha(p)
 for base in ['outputs/a800/phase6/tp4-opt2-final-build-20261004','outputs/a800/phase4/taco-fused-warp-20261003','outputs/a800/phase3/three-way-build']:
  parent=REPO/base;m=json.loads((parent/'manifest.json').read_text()) if (parent/'manifest.json').exists() else None
  if m:
   for name,h in m['files'].items():
    if '__pycache__' not in Path(name).parts and not name.endswith('.pyc'):assert sha(parent/name)==h,name;protected[str(parent/name)]=h
  else:
   for p in parent.glob('*/manifest.json'):
    m=json.loads(p.read_text());protected[str(p)]=sha(p)
    for name,h in m['files'].items():
     if '__pycache__' not in Path(name).parts and not name.endswith('.pyc'):assert sha(p.parent/name)==h,name;protected[str(p.parent/name)]=h
 result=dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),repo=str(REPO),source=str(source),config=config,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},protected_original_files=protected,scope='Independent physical copies of the full TP4 runtime for the archived configuration and frozen scripts. External Megatron commit and Python environment remain pinned by the original protocol.')
 (out/'archive-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(out)
def audit(root,originals=False):
 root=root.resolve();m=json.loads((root/'archive-manifest.json').read_text())
 for name,h in m['files'].items():assert sha(root/name)==h,name
 for p in root.rglob('*'):
  if p.is_symlink():assert p.resolve().is_relative_to(root),p
 if originals:
  for name,h in m['protected_original_files'].items():assert sha(name)==h,name
 print('TP4 archive verified:',len(m['files']),'files; originals checked:',originals)
 return m
def prepare(root,out,job):
 root,out=root.resolve(),out.resolve();m=audit(root);assert not out.exists()
 shutil.copytree(root/'template',out);(out/'results').mkdir()
 info=json.loads((root/'source-submission.json').read_text());info.update(job_id=job,repo=m['repo'],artifact_root=str(root/'artifacts'),archive_root=str(root),source_campaign=info['original_campaign'],original_campaign=str(out))
 info['files']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
 info['build_manifest_sha256']=sha(root/'artifacts/build/manifest.json')
 (out/'submission.json').write_text(json.dumps(info,indent=2)+'\n');print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
 c=sub.add_parser('create');c.add_argument('out',type=Path);c.add_argument('--source',type=Path,required=True)
 c=sub.add_parser('audit');c.add_argument('archive',type=Path);c.add_argument('--originals',action='store_true')
 c=sub.add_parser('prepare');c.add_argument('archive',type=Path);c.add_argument('out',type=Path);c.add_argument('--job-id',type=int,required=True)
 a=p.parse_args()
 if a.action=='create':create(a.out,a.source)
 elif a.action=='audit':audit(a.archive,a.originals)
 else:prepare(a.archive,a.out,a.job_id)
