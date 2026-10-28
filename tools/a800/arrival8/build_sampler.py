"""Serial isolated builds for the two instrumented TP8 policies."""
import difflib,hashlib,json,shlex,shutil,subprocess
from pathlib import Path
from instrument import instrument_epilogue
ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'outputs/a800/gemm_rs_validation/source'
PARENT=ROOT/'outputs/a800/phase3/three-way-build'
OUT=ROOT/'outputs/a800/arrival8/mechanism-build'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parent=json.loads((PARENT/'manifest.json').read_text())
for name,digest in parent['files'].items():assert sha(PARENT/name)==digest,name
OUT.mkdir(exist_ok=False)
source=BASE/'src/gemm_rs/epilogue_evt.hpp';original=source.read_text();modified=instrument_epilogue(original)
(OUT/'instrumentation.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/src/gemm_rs/epilogue_evt.hpp',tofile='b/src/gemm_rs/epilogue_evt.hpp')))
for name in ('sampler.h','sampler.cu'):shutil.copy2(TOOLS/name,OUT/name)
NVCC='/data/apps/cuda/12.8/bin/nvcc'
FLAGS=['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-gencode=arch=compute_80,code=[sm_80,compute_80]']
manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),parent_build=str(PARENT),parent_manifest_sha256=sha(PARENT/'manifest.json'),config_sha256=sha(TOOLS/'config.json'),policies={},commands=[],files={})
def run(args,cwd=ROOT):
 args=list(map(str,args));manifest['commands'].append(dict(args=args,cwd=str(cwd)))
 print(shlex.join(args),flush=True);subprocess.run(args,cwd=cwd,check=True)
helper=OUT/'sampler.o';run([NVCC,*FLAGS,'-I'+str(OUT),'-c',OUT/'sampler.cu','-o',helper])
version=(BASE/'src/cuda/version.ld').read_text().replace('global:','global:\n    mech_*;')
(OUT/'version.ld').write_text(version)
reg=next((BASE/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
work=BASE/'build/src/cuda'
for policy in ('remote_first','interleaved'):
 dest=OUT/policy;shutil.copytree(PARENT/policy,dest,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
 overlay=dest/'overlay';(overlay/'gemm_rs/epilogue_evt.hpp').write_text(modified)
 includes=[overlay,OUT,BASE/'src',BASE/'include',BASE/'build',BASE/'3rdparty/cutlass/include',BASE/'3rdparty/cutlass/tools/util/include',BASE/'3rdparty/cutlass/tools/library/include',BASE/'3rdparty/cutlass/tools/profiler/include']
 obj=dest/'registration.o';dlink=dest/'device_link.o';lib=dest/'libflux_cuda.so'
 run([NVCC,*FLAGS,*['-I'+str(p) for p in includes],'-c',reg,'-o',obj])
 objects=shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())
 objects=[obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve() for p in objects]+[helper]
 run([NVCC,*FLAGS,'-shared','-dlink',*objects,'-o',dlink,'-L/data/apps/cuda/12.8/lib64','-lcudart_static','-lcudadevrt','-ldl','-lrt','-lpthread'])
 link=shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
 for i,arg in enumerate(link):
  if arg.endswith('.o'):link[i]=str(obj if 'registers/flux_bf16_bf16_void' in arg else dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
  if '--version-script=' in arg:link[i]='-Wl,--version-script='+str(OUT/'version.ld')
 link[link.index('-o')+1]=str(lib);link.append(str(helper));run(link,work)
 for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
  p=dest/'python/flux/lib'/name;p.unlink();p.symlink_to(dest/name)
 manifest['policies'][policy]=dict(library=str(lib),sampling=True,sha256=sha(lib),swizzle_sha256=sha(overlay/'gemm_rs/tile_scheduler/threadblock_swizzle.hpp'))
for p in OUT.rglob('*'):
 if p.is_file() and not p.is_symlink():manifest['files'][str(p.relative_to(OUT))]=sha(p)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('MECHANISM BUILD COMPLETE',flush=True)
