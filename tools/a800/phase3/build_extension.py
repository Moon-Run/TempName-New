"""Freeze the already matched, uninstrumented libraries; compile expanded GPU mapping checks."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'outputs/a800/gemm_rs_validation/source'
OLD=ROOT/'outputs/a800/phase3/build'
OUT=ROOT/'outputs/a800/phase3/extension-build'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
previous=json.loads((OLD/'manifest.json').read_text())
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'sampling':False,'policies':{},'files':{},'commands':[]}
for policy,variant in [('original','original_matched'),('remote_first','remote_first')]:
 src=OLD/variant;dest=OUT/policy;dest.mkdir(exist_ok=True)
 assert sha(src/'libflux_cuda.so')==previous[variant]['sha256']
 header=src/'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
 assert sha(header)==previous[variant]['swizzle_sha256']
 shutil.copytree(src/'overlay',dest/'overlay',dirs_exist_ok=True)
 shutil.copy2(src/'swizzle.patch',dest/'swizzle.patch')
 package=dest/'python/flux'
 shutil.copytree(BASE/'python/flux',package,ignore=shutil.ignore_patterns('lib','include','__pycache__'),dirs_exist_ok=True)
 (package/'lib').mkdir(exist_ok=True)
 for name,origin in [('libflux_cuda.so',src/'libflux_cuda.so'),('libflux_cuda_ths_op.so',BASE/'build/lib/libflux_cuda_ths_op.so')]:
  shutil.copy2(origin,dest/name)
  link=package/'lib'/name
  if not link.is_symlink():link.symlink_to(dest/name)
 for origin in (BASE/'python').glob('flux_ths_pybind*.so'):shutil.copy2(origin,dest/'python'/origin.name)
 includes=[dest/'overlay',BASE/'src',BASE/'include',BASE/'build',BASE/'3rdparty/cutlass/include',BASE/'3rdparty/cutlass/tools/util/include']
 cmd=['/data/apps/cuda/12.8/bin/nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_80,code=[sm_80,compute_80]',
      *['-I'+str(p) for p in includes],str(ROOT/'tools/a800/phase3/check_mapping_extension.cu'),'-o',str(dest/'check_mapping')]
 manifest['commands'].append(cmd);subprocess.run(cmd,check=True)
 manifest['policies'][policy]=dict(previous[variant],variant=variant,library=str(dest/'libflux_cuda.so'))
for p in OUT.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.name!='manifest.json':manifest['files'][str(p.relative_to(OUT))]=sha(p)
manifest['mapping_source_sha256']=sha(ROOT/'tools/a800/phase3/check_mapping_extension.cu')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('EXTENSION BUILD VERIFIED',OUT)
