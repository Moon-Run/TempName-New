"""Build exact-shape MLP arrival tables for three independently calibrated bases."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('phase4_build',HERE.parent/'phase4/build.py')
compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler)
spec=importlib.util.spec_from_file_location('phase5_build',HERE.parent/'phase5/build.py')
phase5=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase5)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def tables(entries,device=False):
    qualifier='static __device__ __constant__' if device else 'static const'
    field='maps' if device else 'coords'
    prefix='phase6_device' if device else 'phase6_host'
    text='#pragma once\n'
    assert sum(2*4*s['tiles'] for s in entries)<=64*1024
    for i,s in enumerate(entries):
        values=[v for row in s[field] for v in row]
        assert all(0<=v<65536 for v in values)
        text+=f'{qualifier} unsigned short {prefix}_{i}[] = {{'+','.join(map(str,values))+'};\n'
    text+=('__device__ __forceinline__ ' if device else 'inline ')+f'int {prefix}_lookup(int tm,int tn,int k,int rank,int idx) {{\n'
    for i,s in enumerate(entries):
        text+=f"if(tm=={s['M']//128} && tn=={(s['N']+127)//128} && k=={s['K_local']} && rank>=0 && rank<4 && idx>=0 && idx<{s['tiles']})return {prefix}_{i}[rank*{s['tiles']}+idx];\n"
    return text+'return -1;\n}\n'


def patch(old,base,entries):
    text=phase5.patch_swizzle(old,'remote_first' if base=='remote_first' else 'interleaved')
    predicate=' || '.join(f"(problem_size_.m()=={s['M']} && problem_size_.n()=={s['N']} && problem_size_.k()=={s['K_local']})" for s in entries)
    old_pred='phase5_mlp = local_world_size == 4 && problem_size_.m() == 2048 &&\n                   problem_size_.n() == 2048 && problem_size_.k() == 2048;'
    assert text.count(old_pred)==1
    text=text.replace(old_pred,f'phase5_mlp = local_world_size == 4 && ({predicate});\n      phase6_local_k = problem_size_.k();')
    text=text.replace('  bool phase5_mlp = false;','  bool phase5_mlp = false;\n  int phase6_local_k = 0;')
    if base in ('interleaved_remote','interleaved_remote_group'):
        anchor='      int destination = (tile_idx % local_world_size + local_rank) % local_world_size;'
        assert text.count(anchor)==1
        rule='slot = (slot + 1) % local_world_size;' if base=='interleaved_remote_group' else 'if (slot < 2) slot ^= 1;'
        text=text.replace(anchor,'      int slot = tile_idx % local_world_size;\n      '+rule+'\n      int destination = (slot + local_rank) % local_world_size;')
    text=text.replace('#pragma once','#pragma once\n#ifdef __CUDACC__\n'+tables(entries,True)+
                      '#else\ninline int phase6_device_lookup(int,int,int,int,int){return -1;}\n#endif',1)
    anchor='    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
    assert text.count(anchor)==1
    return text.replace(anchor,'''    if (phase5_mlp) {
      int mapped = phase6_device_lookup(ThreadblockSwizzleStreamK::tiled_shape().m(),
          ThreadblockSwizzleStreamK::tiled_shape().n(), phase6_local_k, local_rank, tile_idx);
      if (mapped >= 0) tile_idx = mapped;
    }
'''+anchor)


def main(plan_path,out,bases=None):
    plan_path,out=plan_path.resolve(),out.resolve()
    plan=json.loads(plan_path.read_text());assert plan['schema']==2 and plan['world']==4
    bases = bases or list(plan['policies'])
    assert len(set(bases)) == len(bases) and set(bases) <= set(plan['policies'])
    for p,h in plan['inputs'].items():assert sha(Path(p))==h,p
    out.mkdir(parents=True,exist_ok=False)
    for base,entries in plan['policies'].items():
        if base not in bases:
            continue
        name='remote_arrival' if base=='remote_first' else base+'_arrival'
        folder=out/name
        compiler.main(folder,'fused',lambda old:patch(old,base,entries),name)
        (folder/'taco/overlay/phase6_host.hpp').write_text(tables(entries))
        shutil.copy2(plan_path,folder/'joint-plan.json')
        m=json.loads((folder/'manifest.json').read_text())
        cmd=next(c['args'] for c in m['commands'] if c['args'][0].endswith('/nvcc') and '-c' in c['args'])
        cmd=cmd[:cmd.index('-c')]+[str(HERE/'check_mapping.cu'),'-o',str(folder/'check_mapping')]
        subprocess.run(cmd,check=True)
        m['commands'].append(dict(args=cmd,cwd=str(REPO)))
        m['arrival_plan_sha256']=sha(plan_path);m['arrival_base']=base
        m['inputs'][str(plan_path)]=sha(plan_path)
        for p in (HERE/'build.py',HERE/'check_mapping.cu'):m['inputs'][str(p)]=sha(p)
        for p in (folder/'check_mapping',folder/'joint-plan.json',folder/'taco/overlay/phase6_host.hpp'):
            m['files'][str(p.relative_to(folder))]=sha(p)
        (folder/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('plan',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--bases',nargs='+',choices=['remote_first','interleaved','interleaved_remote','interleaved_remote_group'])
    a=p.parse_args();main(a.plan,a.out,a.bases)
