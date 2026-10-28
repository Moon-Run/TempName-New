"""Clone unchanged GPU mappings and bind a different runtime-only physical mask."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    base,out,plan_path = args.base.resolve(),args.out.resolve(),args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    out.mkdir(parents=True,exist_ok=False)
    for policy,key in [('remote_arrival','remote_first'),('interleaved_arrival','interleaved')]:
        old,new = base/policy,out/policy
        m = json.loads((old/'manifest.json').read_text())
        for name,digest in m['files'].items(): assert sha(old/name) == digest,name
        previous = json.loads((old/'joint-plan.json').read_text())
        assert previous['world'] == plan['world'] == 8
        for field in ('shape','maps','coords','base_coords'):
            assert previous['policies'][key][0][field] == plan['policies'][key][0][field],field
        shutil.copytree(old,new,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(plan_path,new/'joint-plan.json')
        for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            link = new/'taco/python/flux/lib'/name
            link.unlink();link.symlink_to(new/'taco'/name)
        m['arrival_plan_sha256'] = sha(plan_path)
        m['mask_only_parent'] = dict(path=str(old/'manifest.json'),sha256=sha(old/'manifest.json'))
        m['mask_bound_without_recompile'] = True
        m['commands'] = [dict(args=[v.replace(str(old),str(new)) for v in c['args']],
                              cwd=c['cwd'].replace(str(old),str(new))) for c in m['commands']]
        for p in (plan_path,Path(__file__).resolve()): m['inputs'][str(p)] = sha(p)
        m['files'] = {str(p.relative_to(new)):sha(p) for p in new.rglob('*')
                     if p.is_file() and not p.is_symlink() and p.name not in ('manifest.json','build-progress.json')}
        (new/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('base',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--plan',type=Path,required=True)
    main(p.parse_args())
