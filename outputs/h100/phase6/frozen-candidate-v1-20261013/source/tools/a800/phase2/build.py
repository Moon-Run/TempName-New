"""Recompile one registration unit and relink an isolated library; original build stays intact."""
from pathlib import Path
import difflib,json,os,shlex,shutil,subprocess
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'outputs/a800/gemm_rs_validation/source'
OUT=ROOT/'outputs/a800/phase2/build'
OUT.mkdir(parents=True,exist_ok=True)
OVER=OUT/'overlay';(OVER/'gemm_rs').mkdir(parents=True,exist_ok=True)
source=BASE/'src/gemm_rs/epilogue_evt.hpp'
s=source.read_text();old=s
s=s.replace('#pragma once','#pragma once\n#include "sampler.h"',1)
s=s.replace('    bool has_barrier_ptr;','    Phase2Config phase2;\n    bool has_barrier_ptr;',1)
s=s.replace('  static constexpr Params\n  to_underlying_arguments','  static Params\n  to_underlying_arguments',1)
s=s.replace('    int col_stride[MAX_RANK_SIZE];','    int col_stride[MAX_RANK_SIZE];\n    unsigned p2_step_mask = 0;',1)
s=s.replace('    params.dAux = args.dAux;','    params.phase2 = phase2_config();\n    params.dAux = args.dAux;',1)
needle='    CUTLASS_DEVICE void\n    end_step(int step_idx) {\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''      bool p2 = !kPcieMode && params.phase2.mode && params.phase2.stride > 0 &&
                tile_idx < P2_TILES && step_idx < P2_FRAGMENTS &&
                tile_idx % params.phase2.stride == params.phase2.offset;
      unsigned long long* rec = nullptr;
      if(p2){
        p2_step_mask |= 1u << step_idx;
        // All threads have completed this fragment's conversion before the ready marker.
        __syncthreads();
        if(thread_idx==0){
          rec=params.phase2.producer+(tile_idx*P2_FRAGMENTS+step_idx)*P2_FIELDS;
          rec[0]=phase2_clock(); rec[2]=ThreadblockShape::kM; rec[3]=ThreadblockShape::kN;
          rec[4]=size<3>(tC_cAux); rec[5]=dst_rank; rec[6]=blockIdx.x;
          atomicAdd(rec+7,1ull);
        }
      }
''',1)
needle='''      }
    }

    ////
    CUTLASS_DEVICE void
    end_epilogue()'''
assert s.count(needle)==1
s=s.replace(needle,'''      }
      if(p2 && params.phase2.mode==1){
        __syncthreads();
        if(thread_idx==0)rec[1]=phase2_clock();
      }
    }

    ////
    CUTLASS_DEVICE void
    end_epilogue()''',1)
s=s.replace('    end_epilogue() {', '    end_epilogue() {\n      if(p2_step_mask && params.phase2.mode==2){\n        // One system fence per epilogue callback, not per fragment. Separate Stream-K\n        // reduction callbacks publish their own fragment; no tile is declared complete here.\n        __threadfence_system();\n        __syncthreads();\n        if(thread_idx==0){\n          auto stamp=phase2_clock();\n          for(int f=0;f<P2_FRAGMENTS;++f)if(p2_step_mask & (1u<<f)){\n            auto rec=params.phase2.producer+(tile_idx*P2_FRAGMENTS+f)*P2_FIELDS;\n            rec[1]=stamp;\n            phase2_publish(params.phase2.flags[dst_rank]+params.rank*P2_SLOTS+\n                           tile_idx*P2_FRAGMENTS+f,params.phase2.epoch);\n          }\n        }\n      }\n',1)
(OVER/'gemm_rs/epilogue_evt.hpp').write_text(s)
(OUT/'instrumentation.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/src/gemm_rs/epilogue_evt.hpp',tofile='b/src/gemm_rs/epilogue_evt.hpp')))
TOOLS=ROOT/'tools/a800/phase2'
for name in ['sampler.h','sampler.cu']:shutil.copy2(TOOLS/name,OVER/name)
NVCC='/data/apps/cuda/12.8/bin/nvcc'
reg=next((BASE/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
obj=OUT/'registration.o';helper=OUT/'sampler.o';dlink=OUT/'device_link.o'
includes=[OVER,BASE/'src',BASE/'include',BASE/'build',BASE/'3rdparty/cutlass/include',BASE/'3rdparty/cutlass/tools/util/include',BASE/'3rdparty/cutlass/tools/library/include',BASE/'3rdparty/cutlass/tools/profiler/include']
flags=['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-gencode=arch=compute_80,code=[sm_80,compute_80]']
cmds=[]
def run(args,cwd=None):
 cmds.append({'args':list(map(str,args)),'cwd':str(cwd or ROOT)})
 print(shlex.join(list(map(str,args))),flush=True)
 subprocess.run(list(map(str,args)),cwd=cwd or ROOT,check=True)
run([NVCC,*flags,*['-I'+str(p) for p in includes],'-c',reg,'-o',obj])
run([NVCC,*flags,'-I'+str(OVER),'-c',OVER/'sampler.cu','-o',helper])
work=BASE/'build/src/cuda'
objects=shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())
objects=[obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve() for p in objects]
objects.append(helper)
run([NVCC,*flags,'-shared','-dlink',*objects,'-o',dlink,'-L/data/apps/cuda/12.8/lib64','-lcudart_static','-lcudadevrt','-ldl','-lrt','-lpthread'])
link=shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
version=(BASE/'src/cuda/version.ld').read_text().replace('global:','global:\n    phase2_*;')
(OUT/'version.ld').write_text(version)
for i,arg in enumerate(link):
 if arg.endswith('.o'):
  link[i]=str(obj if 'registers/flux_bf16_bf16_void' in arg else dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
 if '--version-script=' in arg:link[i]='-Wl,--version-script='+str(OUT/'version.ld')
link[link.index('-o')+1]=str(OUT/'libflux_cuda.so');link.append(str(helper))
run(link,cwd=work)
package=OUT/'python/flux'
shutil.copytree(BASE/'python/flux',package,ignore=shutil.ignore_patterns('lib','include','__pycache__'),dirs_exist_ok=True)
(package/'lib').mkdir(exist_ok=True)
for name,origin in [('libflux_cuda.so',OUT/'libflux_cuda.so'),('libflux_cuda_ths_op.so',BASE/'build/lib/libflux_cuda_ths_op.so')]:
 p=package/'lib'/name
 if not p.exists():p.symlink_to(origin)
for p in (BASE/'python').glob('flux_ths_pybind*.so'):
 target=OUT/'python'/p.name
 if not target.exists():target.symlink_to(p)
(OUT/'commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
print('PHASE2 BUILD COMPLETE',flush=True)
