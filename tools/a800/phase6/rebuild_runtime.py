"""Clone a frozen candidate and rebuild only its decoder/runtime, never in place."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(base,out):
    repo=Path(__file__).resolve().parents[3]
    base,out=base.resolve(),out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    policies = [name for name in ('remote_arrival','interleaved_arrival',
                                  'interleaved_remote_arrival','interleaved_remote_group_arrival')
                if (base/name/'manifest.json').is_file()]
    assert policies, 'No frozen candidate manifests found'
    for policy in policies:
        old,new=base/policy,out/policy
        m=json.loads((old/'manifest.json').read_text())
        for name,h in m['files'].items():assert sha(old/name)==h,name
        # These affect the copied GEMM and C++ object ABI. Require them to match
        # exactly; a change needs the full phase5 builder instead.
        for name in ('epilogue_evt.hpp','taco_runtime.h','taco_codec.cuh','ths_op/gemm_reduce_scatter.cc'):
            assert sha(repo/'src/gemm_rs'/name)==sha(old/'taco/overlay/gemm_rs'/name),name
        shutil.copytree(old,new,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(repo/'src/gemm_rs/taco_runtime.cu',new/'taco/overlay/gemm_rs/taco_runtime.cu')
        shutil.copy2(repo/'python/flux/gemm_rs_taco.py',new/'taco/python/flux/gemm_rs_taco.py')
        for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            p=new/'taco/python/flux/lib'/lib
            p.unlink();p.symlink_to(new/'taco'/lib)
        commands=[dict(args=[s.replace(str(old),str(new)) for s in c['args']],
                       cwd=c['cwd'].replace(str(old),str(new))) for c in m['commands']]
        for c in commands:
            args=c['args']
            if '-o' in args and Path(args[args.index('-o')+1]).name in ('taco_runtime.o','device_link.o','libflux_cuda.so','libflux_cuda_ths_op.so'):
                subprocess.run(args,cwd=c['cwd'],check=True)
        m['commands']=commands
        m['runtime_parent_manifest']=dict(path=str(old/'manifest.json'),sha256=sha(old/'manifest.json'))
        m['inputs'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
        m['inputs'][str(repo/'src/gemm_rs/taco_runtime.cu')]=sha(repo/'src/gemm_rs/taco_runtime.cu')
        m['inputs'][str(repo/'python/flux/gemm_rs_taco.py')]=sha(repo/'python/flux/gemm_rs_taco.py')
        m['files']={str(p.relative_to(new)):sha(p) for p in new.rglob('*')
                    if p.is_file() and not p.is_symlink() and p.name not in ('manifest.json','build-progress.json')}
        (new/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        print('RUNTIME REBUILD COMPLETE',policy,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('base',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.base,a.out)
