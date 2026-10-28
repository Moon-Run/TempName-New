"""Isolated MLP-only BF16 mappings for the existing hierarchical TP8 adapter.

No tuning: retain stock kernels, wrapper, tile geometry and scheduling. Only
extend the shape predicate to the hierarchical local GEMM M4096/N2048/K1024.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[4]
SOURCE=REPO/'outputs/a800/gemm_rs_validation/source'
PARENT=REPO/'outputs/a800/phase3/three-way-build'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'mapping').mkdir()
    parent=json.loads((PARENT/'manifest.json').read_text())
    for name,h in parent['files'].items():assert sha(PARENT/name)==h,name
    module=importlib.util.spec_from_file_location('mlp_shapes',REPO/'tools/a800/phase5/build.py')
    patcher=importlib.util.module_from_spec(module);module.loader.exec_module(patcher)
    manifest=dict(scope='BF16 only, MLP-only node-local mapping in hierarchical TP8; no arrival or quantization',
                  parent_manifest_sha256=sha(PARENT/'manifest.json'),policies={},commands=[])
    nvcc='/data/apps/cuda/12.8/bin/nvcc'
    flags=['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda',
           '--expt-relaxed-constexpr','-Xcompiler=-fPIC','-gencode=arch=compute_80,code=[sm_80,compute_80]']
    def run(args,cwd=REPO):
        args=list(map(str,args));manifest['commands'].append(dict(args=args,cwd=str(cwd)))
        print('BUILD',args[-1],flush=True);subprocess.run(args,cwd=cwd,check=True)
    for policy in ('original','remote_first','interleaved'):
        dest=out/policy
        shutil.copytree(PARENT/'original',dest,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.o','check_mapping'))
        overlay=dest/'overlay';header=overlay/'gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
        inc=[overlay,SOURCE/'src',SOURCE/'include',SOURCE/'build',SOURCE/'3rdparty/cutlass/include',
             SOURCE/'3rdparty/cutlass/tools/util/include',SOURCE/'3rdparty/cutlass/tools/library/include',
             SOURCE/'3rdparty/cutlass/tools/profiler/include']
        if policy!='original':
            text=patcher.patch_swizzle(header.read_text(),policy)
            before='problem_size_.m() == 2048 &&\n                   problem_size_.n() == 2048 && problem_size_.k() == 2048;'
            assert before in text
            text=text.replace(before,'problem_size_.m() == 4096 &&\n                   problem_size_.n() == 2048 && problem_size_.k() == 1024;')
            header.write_text(text)
            reg=next((SOURCE/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
            work=SOURCE/'build/src/cuda';obj=dest/'registration.o';dlink=dest/'device_link.o';lib=dest/'libflux_cuda.so'
            if lib.is_symlink():lib.unlink()
            run([nvcc,*flags,*['-I'+str(p) for p in inc],'-c',reg,'-o',obj])
            objects=[obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve()
                     for p in shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())]
            run([nvcc,*flags,'-shared','-dlink',*objects,'-o',dlink,'-L/data/apps/cuda/12.8/lib64',
                 '-lcudart_static','-lcudadevrt','-ldl','-lrt','-lpthread'])
            link=shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
            for i,arg in enumerate(link):
                if arg.endswith('.o'):
                    link[i]=str(obj if 'registers/flux_bf16_bf16_void' in arg else dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
            link[link.index('-o')+1]=str(lib);run(link,work)
        for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            link=dest/'python/flux/lib'/lib;link.unlink();link.symlink_to(dest/lib)
        run([nvcc,*flags,*['-I'+str(p) for p in inc],REPO/'tools/a800/e2e/scale/check_mapping.cu',
             '-o',out/'mapping'/f'check_mapping-{policy}'])
        manifest['policies'][policy]=dict(library_sha256=sha(dest/'libflux_cuda.so'),swizzle_sha256=sha(header))
    for name,h in parent['files'].items():assert sha(PARENT/name)==h,name
    manifest['files']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    manifest['builder_sha256']=sha(Path(__file__))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('TP8 BF16 MLP-ONLY BUILD COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path);main(p.parse_args().out)
