"""Serially build only the two arrival overlays; freeze the three baseline libraries."""
import difflib,hashlib,json,shlex,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SOURCE=ROOT/'outputs/a800/gemm_rs_validation/source'
PARENT=ROOT/'outputs/a800/phase3/three-way-build'
OUT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'outputs/a800/arrival/build'
PLAN=Path(sys.argv[1]).resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads(PLAN.read_text());assert plan['world_size']==4;assert plan.get('schema_version')==2, 'Refit calibration with the audited planner into a new plan'
assert plan['planner_sha256']==sha(HERE/'plan.py'), 'Planner changed after fitting'
parent=json.loads((PARENT/'manifest.json').read_text())
for name,digest in parent['files'].items():assert sha(PARENT/name)==digest,name
for name,digest in plan['inputs'].items():assert sha(Path(name))==digest,name
OUT.mkdir(exist_ok=False);(OUT/'mapping').mkdir()
shutil.copy2(PLAN,OUT/'plan.json')
manifest=dict(plan_schema_version=plan['schema_version'],score_mode=plan['score_mode'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),plan_sha256=sha(PLAN),parent_manifest_sha256=sha(PARENT/'manifest.json'),files={},commands=[],policies={})
FLAGS=['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-gencode=arch=compute_80,code=[sm_80,compute_80]']
NVCC='/data/apps/cuda/12.8/bin/nvcc'
def run(args,cwd=ROOT):
 args=list(map(str,args));manifest['commands'].append(dict(args=args,cwd=str(cwd)));print(shlex.join(args),flush=True);subprocess.run(args,cwd=cwd,check=True)
def arrays(shapes,device=False,coord=False):
 if device:
  shapes=[s for s in shapes if not s.get('identity_fallback',False)]
  assert sum(len(r)*2 for s in shapes for r in s['maps'])<=64*1024, 'Constant table exceeds 64 KiB'
 qualifier='static __device__ __constant__' if device else 'static const'
 text='#pragma once\n'
 for i,s in enumerate(shapes):
  data=s['coords' if coord else 'maps'];flat=[x for row in data for x in row]
  assert all(0<=x<=65535 for x in flat), 'Tile indices exceed uint16 table capacity'
  text+=f'{qualifier} unsigned short arrival_{i}[{len(flat)}] = {{'+','.join(map(str,flat))+'};\n'
 name='flux_arrival_index' if device else 'flux_arrival_host_coord' if coord else 'flux_arrival_host_index'
 text+= ('__device__ __forceinline__ ' if device else 'inline ')+f'int {name}(int tm,int tn,int k,int tp,int rank,int idx) {{\n'
 for i,s in enumerate(shapes):
  text+=f"if(tm=={s['M']//128} && tn=={(s['N']+127)//128} && k=={s['K_local']} && tp==4 && rank>=0 && rank<4 && idx>=0 && idx<{s['tiles']})return arrival_{i}[rank*{s['tiles']}+idx];\n"
 text+='return -1;\n}\n'
 return text
for policy in ['original','remote_first','interleaved','remote_arrival','interleaved_arrival']:
 base=plan['policies'][policy]['base'] if policy in plan['policies'] else policy
 dest=OUT/policy
 shutil.copytree(PARENT/base,dest,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.o','check_mapping'))
 overlay=dest/'overlay';header=overlay/'gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
 if policy in plan['policies']:
  shapes=plan['policies'][policy]['shapes'];old=header.read_text()
  (overlay/'arrival_table.hpp').write_text(arrays(shapes,device=True))
  host=arrays(shapes).replace('arrival_', 'host_map_').replace('flux_host_map_host_index','flux_arrival_host_index')
  coord=arrays(shapes,coord=True).replace('arrival_', 'host_coord_').replace('flux_host_coord_host_coord','flux_arrival_host_coord')
  (overlay/'arrival_host.hpp').write_text(host+'\n'+coord)
  modified=old.replace('#pragma once','#pragma once\n#include <stdexcept>\n#include "arrival_table.hpp"',1)
  modified=modified.replace('  int local_rank;', '  int arrival_local_k;\n  int local_rank;',1)
  anchor='    const char *local_rank_str'
  modified=modified.replace(anchor,'    arrival_local_k = problem_size_.k();\n'+anchor,1)
  marker='      local_world_size = atoi(local_world_size_str);'
  cond=' || '.join(f'(problem_size_.m()=={s["M"]} && problem_size_.n()=={s["N"]} && problem_size_.k()=={s["K_local"]})' for s in shapes)
  modified=modified.replace(marker,marker+f'\n      if(local_world_size!=4 || tile_size_.m()!=128 || tile_size_.n()!=128 || tile_size_.k()!=32 || !({cond})) throw std::runtime_error("Unsupported arrival shape/TP");')
  marker='    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
  modified=modified.replace(marker,'    int arrival_idx = flux_arrival_index(ThreadblockSwizzleStreamK::tiled_shape().m(), ThreadblockSwizzleStreamK::tiled_shape().n(), arrival_local_k, local_world_size, local_rank, tile_idx);\n    if(arrival_idx >= 0) tile_idx = arrival_idx;\n'+marker,1)
  header.write_text(modified)
  (dest/'swizzle.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),modified.splitlines(True))))
  includes=[overlay,SOURCE/'src',SOURCE/'include',SOURCE/'build',SOURCE/'3rdparty/cutlass/include',SOURCE/'3rdparty/cutlass/tools/util/include',SOURCE/'3rdparty/cutlass/tools/library/include',SOURCE/'3rdparty/cutlass/tools/profiler/include']
  reg=next((SOURCE/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
  work=SOURCE/'build/src/cuda';obj=dest/'registration.o';dlink=dest/'device_link.o';lib=dest/'libflux_cuda.so'
  run([NVCC,*FLAGS,*['-I'+str(p) for p in includes],'-c',reg,'-o',obj])
  objects=[obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve() for p in shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())]
  run([NVCC,*FLAGS,'-shared','-dlink',*objects,'-o',dlink,'-L/data/apps/cuda/12.8/lib64','-lcudart_static','-lcudadevrt','-ldl','-lrt','-lpthread'])
  link=shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
  for i,arg in enumerate(link):
   if arg.endswith('.o'):link[i]=str(obj if 'registers/flux_bf16_bf16_void' in arg else dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
  link[link.index('-o')+1]=str(lib);run(link,work)
 else:
  (overlay/'arrival_host.hpp').write_text('inline int flux_arrival_host_index(int,int,int,int,int,int){return -1;}\ninline int flux_arrival_host_coord(int,int,int,int,int,int){return -1;}\n')
 for name in ['libflux_cuda.so','libflux_cuda_ths_op.so']:
  link=dest/'python/flux/lib'/name;link.unlink();link.symlink_to(dest/name)
 includes=[overlay,SOURCE/'src',SOURCE/'include',SOURCE/'build',SOURCE/'3rdparty/cutlass/include',SOURCE/'3rdparty/cutlass/tools/util/include']
 run([NVCC,*FLAGS,*['-I'+str(p) for p in includes],HERE/'check_mapping.cu','-o',OUT/'mapping'/f'check_mapping-{policy}'])
 manifest['policies'][policy]=dict(base=base,library_sha256=sha(dest/'libflux_cuda.so'),swizzle_sha256=sha(header))
for p in OUT.rglob('*'):
 if p.is_file():manifest['files'][str(p.relative_to(OUT))]=sha(p)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
mapping_files={str(p):sha(p) for p in (OUT/'mapping').glob('check_mapping-*')}
(OUT/'mapping/manifest.json').write_text(json.dumps(dict(files=mapping_files),indent=2)+'\n')
print('ARRIVAL BUILD COMPLETE',flush=True)
