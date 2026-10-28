"""Build the R1,L,R2,R3 instrumented control in a fresh isolated directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out,group_first=False):
    repo=Path(__file__).resolve().parents[3]
    parent=repo/'outputs/a800/phase3/mechanism-build'
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha(parent/name)==h,name
    old=parent/'interleaved';out=out.resolve()
    shutil.copytree(old,out,symlinks=True)
    header=out/'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
    text=header.read_text()
    anchor='    int destination = (tile_idx % local_world_size + local_rank) % local_world_size;'
    assert text.count(anchor)==1
    slot_rule='slot = (slot + 1) % local_world_size;' if group_first else 'if (slot < 2) slot ^= 1;'
    text=text.replace(anchor,'    int slot = tile_idx % local_world_size;\n    '+slot_rule+'\n    int destination = (slot + local_rank) % local_world_size;')
    header.write_text(text)
    commands=[]
    for c in manifest['commands']:
        if '-o' not in c['args']:continue
        target=c['args'][c['args'].index('-o')+1]
        if not target.startswith(str(old)+'/'):continue
        cmd=[s.replace(str(old),str(out)) for s in c['args']]
        commands.append(dict(args=cmd,cwd=c['cwd']))
        subprocess.run(cmd,cwd=c['cwd'],check=True)
    for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
        p=out/'python/flux/lib'/name;p.unlink();p.symlink_to(out/name)
    checker=(repo/'tools/a800/e2e/scale/check_mapping.cu').read_text()
    checker=checker.replace('int dest=(i%tp+rank)%tp,within=i/tp;',
        'int slot=i%tp;'+('slot=(slot+1)%tp;' if group_first else 'if(slot<2)slot^=1;')+'int dest=(slot+rank)%tp,within=i/tp;')
    (out/'check_mapping.cu').write_text(checker)
    cmd=next(c['args'] for c in commands if c['args'][0].endswith('/nvcc') and '-c' in c['args'])
    cmd=cmd[:cmd.index('-c')]+[str(out/'check_mapping.cu'),'-o',str(out/'check_mapping')]
    subprocess.run(cmd,check=True)
    commands.append(dict(args=cmd,cwd=str(repo)))
    result=dict(policy='interleaved_remote_group' if group_first else 'interleaved_remote',cycle='R1,R2,R3,L' if group_first else 'R1,L,R2,R3',parent_manifest_sha256=sha(parent/'manifest.json'),
                commands=commands,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and not p.is_symlink()})
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path);p.add_argument('--group-first',action='store_true')
    a=p.parse_args();main(a.out,a.group_first)
