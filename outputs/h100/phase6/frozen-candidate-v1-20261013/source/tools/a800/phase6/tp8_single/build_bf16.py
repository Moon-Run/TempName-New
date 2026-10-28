"""Restrict existing TP8 BF16 order/arrival maps to MLP; no kernel tuning."""
import argparse,hashlib,importlib.util,json,shlex,shutil,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]
SOURCE=REPO/'outputs/a800/gemm_rs_validation/source'
PARENT=REPO/'outputs/a800/phase3/three-way-build'
ARRIVAL=REPO/'outputs/a800/arrival-v2-tp8-20261003/artifacts/build'
POLICIES=['original','remote_first','interleaved','remote_arrival','interleaved_arrival']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
 out=out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'mapping').mkdir()
 for parent in [PARENT,ARRIVAL]:
  for name,h in json.loads((parent/'manifest.json').read_text())['files'].items():assert sha(parent/name)==h,name
 module=importlib.util.spec_from_file_location('mlp_shape_gate',REPO/'tools/a800/phase5/build.py')
 patcher=importlib.util.module_from_spec(module);module.loader.exec_module(patcher)
 plan=json.loads((ARRIVAL/'plan.json').read_text());assert plan['world_size']==8
 manifest=dict(scope='Existing BF16 TP8 maps, restricted to MLP M2048/N2048/K_local1024; attention K_local256 original; no tuning.',plan_sha256=sha(ARRIVAL/'plan.json'),parent_manifest_sha256=sha(PARENT/'manifest.json'),arrival_manifest_sha256=sha(ARRIVAL/'manifest.json'),commands=[],policies={})
 shutil.copy2(ARRIVAL/'plan.json',out/'plan.json')
 nvcc='/data/apps/cuda/12.8/bin/nvcc';flags=['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-gencode=arch=compute_80,code=[sm_80,compute_80]']
 def run(args,cwd=REPO):
  args=list(map(str,args));manifest['commands'].append(dict(args=args,cwd=str(cwd)));print('BUILD',args[-1],flush=True);subprocess.run(args,cwd=cwd,check=True)
 for policy in POLICIES:
  dest=out/policy;shutil.copytree(PARENT/'original',dest,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.o','check_mapping'))
  overlay=dest/'overlay';header=overlay/'gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
  inc=[overlay,SOURCE/'src',SOURCE/'include',SOURCE/'build',SOURCE/'3rdparty/cutlass/include',SOURCE/'3rdparty/cutlass/tools/util/include',SOURCE/'3rdparty/cutlass/tools/library/include',SOURCE/'3rdparty/cutlass/tools/profiler/include']
  if policy!='original':
   base={'remote_arrival':'remote_first','interleaved_arrival':'interleaved'}.get(policy,policy)
   text=patcher.patch_swizzle(header.read_text(),base)
   gate='phase5_mlp = local_world_size == 4 && problem_size_.m() == 2048 &&\n                   problem_size_.n() == 2048 && problem_size_.k() == 2048;'
   assert text.count(gate)==1
   text=text.replace(gate,'phase5_mlp = local_world_size == 8 && problem_size_.m() == 2048 &&\n                   problem_size_.n() == 2048 && problem_size_.k() == 1024;')
   if policy.endswith('_arrival'):
    entry=next(s for s in plan['policies'][policy]['shapes'] if [s['M'],s['N'],s['K_global']]==[2048,2048,8192])
    assert entry['K_local']==1024 and len(entry['maps'])==8
    for name in ['arrival_table.hpp','arrival_host.hpp']:shutil.copy2(ARRIVAL/policy/'overlay'/name,overlay/name)
    text=text.replace('#pragma once','#pragma once\n#include "arrival_table.hpp"',1)
    marker='    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
    assert text.count(marker)==1
    text=text.replace(marker,'    if (phase5_mlp) {\n      int mapped = flux_arrival_index(ThreadblockSwizzleStreamK::tiled_shape().m(), ThreadblockSwizzleStreamK::tiled_shape().n(), 1024, local_world_size, local_rank, tile_idx);\n      if (mapped >= 0) tile_idx = mapped;\n    }\n'+marker,1)
   header.write_text(text)
   reg=next((SOURCE/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
   work=SOURCE/'build/src/cuda';obj=dest/'registration.o';dlink=dest/'device_link.o';lib=dest/'libflux_cuda.so'
   if lib.is_symlink():lib.unlink()
   run([nvcc,*flags,*['-I'+str(p) for p in inc],'-c',reg,'-o',obj])
   objects=[obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve() for p in shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())]
   run([nvcc,*flags,'-shared','-dlink',*objects,'-o',dlink,'-L/data/apps/cuda/12.8/lib64','-lcudart_static','-lcudadevrt','-ldl','-lrt','-lpthread'])
   link=shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
   for i,arg in enumerate(link):
    if arg.endswith('.o'):link[i]=str(obj if 'registers/flux_bf16_bf16_void' in arg else dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
   link[link.index('-o')+1]=str(lib);run(link,work)
  if not policy.endswith('_arrival'):(overlay/'arrival_host.hpp').write_text('inline int flux_arrival_host_index(int,int,int,int,int,int){return -1;}\ninline int flux_arrival_host_coord(int,int,int,int,int,int){return -1;}\n')
  for name in ['libflux_cuda.so','libflux_cuda_ths_op.so']:
   p=dest/'python/flux/lib'/name;p.unlink();p.symlink_to(dest/name)
  run([nvcc,*flags,*['-I'+str(p) for p in inc],REPO/'tools/a800/arrival8/check_mapping.cu','-o',out/'mapping'/f'check_mapping-{policy}'])
  manifest['policies'][policy]=dict(library_sha256=sha(dest/'libflux_cuda.so'),swizzle_sha256=sha(header))
 assert sha(out/'original/libflux_cuda.so')==sha(PARENT/'original/libflux_cuda.so')
 for parent in [PARENT,ARRIVAL]:
  for name,h in json.loads((parent/'manifest.json').read_text())['files'].items():assert sha(parent/name)==h,name
 manifest['files']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
 manifest['builder_sha256']=sha(Path(__file__))
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (out/'mapping/manifest.json').write_text(json.dumps(dict(files={str(p):sha(p) for p in (out/'mapping').glob('check_mapping-*')}),indent=2)+'\n')
 print('SINGLE NODE TP8 MLP-ONLY BF16 BUILD COMPLETE',out,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);main(p.parse_args().out)
