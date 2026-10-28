"""Build static tile-order variants from the uninstrumented phase1 objects."""
from pathlib import Path
import difflib,hashlib,json,shlex,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'outputs/a800/gemm_rs_validation/source'
OUT=ROOT/'outputs/a800/phase3/build';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=BASE/'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
old=SOURCE.read_text()
needle='''    int m = (coord.m() + tiled_m / local_world_size * (local_rank)) % tiled_m;
    coord.m() = m;'''
assert old.count(needle)==1
variants={
 'original':needle,
 'original_matched':needle,
 'remote_first':'''    // Cyclic rank+1 partition offset. Column/cohort rasters can revisit local
    // tiles before exhausting remote tiles; this is not strict remote-first.
    int m = (coord.m() + tiled_m / local_world_size * (local_rank + 1)) % tiled_m;
    coord.m() = m;''',
 'interleaved':'''    // Equal, block-aligned partitions; tested only for M=N=4096 and TP=2.
    // Logical tiles alternate destination partitions. Preserve row-major traversal
    // within each partition. This deliberately changes the original locality pattern.
    int tiled_n = ThreadblockSwizzleStreamK::tiled_shape().n();
    int destination = (tile_idx % local_world_size + local_rank) % local_world_size;
    int within_partition = tile_idx / local_world_size;
    coord.m() = destination * (tiled_m / local_world_size) + within_partition / tiled_n;
    coord.n() = within_partition % tiled_n;'''}
NVCC='/data/apps/cuda/12.8/bin/nvcc'
FLAGS=['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-gencode=arch=compute_80,code=[sm_80,compute_80]']
reg=next((BASE/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
work=BASE/'build/src/cuda'
all_commands=json.loads((OUT/'commands.json').read_text()) if (OUT/'commands.json').exists() else []
manifest=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else {}
def run(args,cwd=None):
 args=list(map(str,args));all_commands.append(dict(args=args,cwd=str(cwd or ROOT)))
 print(shlex.join(args),flush=True);subprocess.run(args,cwd=cwd or ROOT,check=True)
for name,body in variants.items():
 if len(sys.argv)>1 and name not in sys.argv[1:]:continue
 dest=OUT/name;overlay=dest/'overlay';(overlay/'gemm_rs/tile_scheduler').mkdir(parents=True,exist_ok=True)
 text=old.replace(needle,body)
 (overlay/'gemm_rs/tile_scheduler/threadblock_swizzle.hpp').write_text(text)
 (dest/'swizzle.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='a/src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp',tofile='b/src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp')))
 includes=[overlay,BASE/'src',BASE/'include',BASE/'build',BASE/'3rdparty/cutlass/include',BASE/'3rdparty/cutlass/tools/util/include',BASE/'3rdparty/cutlass/tools/library/include',BASE/'3rdparty/cutlass/tools/profiler/include']
 inc=['-I'+str(p) for p in includes]
 obj=dest/'registration.o';dlink=dest/'device_link.o';lib=dest/'libflux_cuda.so'
 if name=='original':
  if not lib.exists():lib.symlink_to(BASE/'build/lib/libflux_cuda.so')
 else:
  run([NVCC,*FLAGS,*inc,'-c',reg,'-o',obj])
  objects=shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())
  objects=[obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve() for p in objects]
  run([NVCC,*FLAGS,'-shared','-dlink',*objects,'-o',dlink,'-L/data/apps/cuda/12.8/lib64','-lcudart_static','-lcudadevrt','-ldl','-lrt','-lpthread'])
  link=shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
  for i,arg in enumerate(link):
   if arg.endswith('.o'):link[i]=str(obj if 'registers/flux_bf16_bf16_void' in arg else dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
  link[link.index('-o')+1]=str(lib);run(link,cwd=work)
 package=dest/'python/flux'
 shutil.copytree(BASE/'python/flux',package,ignore=shutil.ignore_patterns('lib','include','__pycache__'),dirs_exist_ok=True)
 (package/'lib').mkdir(exist_ok=True)
 for lname,origin in [('libflux_cuda.so',lib),('libflux_cuda_ths_op.so',BASE/'build/lib/libflux_cuda_ths_op.so')]:
  p=package/'lib'/lname
  if not p.exists():p.symlink_to(origin)
 for p in (BASE/'python').glob('flux_ths_pybind*.so'):
  target=dest/'python'/p.name
  if not target.exists():target.symlink_to(p)
 run([NVCC,*FLAGS,*inc,ROOT/'tools/a800/phase3/check_mapping.cu','-o',dest/'check_mapping'])
 manifest[name]=dict(library=str(lib),sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                     swizzle_sha256=hashlib.sha256(text.encode()).hexdigest(),sampling=False)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'commands.json').write_text(json.dumps(all_commands,indent=2)+'\n')
print('PHASE3 BUILD COMPLETE',flush=True)
