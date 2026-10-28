"""Build TACO in isolation from frozen BF16 objects; no GPU allocation needed."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT/'outputs/a800/gemm_rs_validation/source'
PARENT = ROOT/'outputs/a800/phase3/three-way-build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out, placement='fused', swizzle_transform=None, mapping_name='original'):
    assert placement in ('fused','separate')
    out = out.resolve()
    manifest = dict(policy='original_taco_all_remote', placement=placement, codec='E4M3, adaptive scale + normalized H128, two FP32 scales',
                    reorder=mapping_name!='original', mapping=mapping_name,
                    selective_quantization_supported=True, gpu_validated=False, files={}, commands=[], inputs={})
    parent = json.loads((PARENT/'manifest.json').read_text())
    for name, digest in parent['files'].items():
        assert sha(PARENT/name) == digest, name
    out.mkdir(parents=True, exist_ok=False)
    dest = out/'taco'
    shutil.copytree(PARENT/'original', dest, symlinks=True, ignore=shutil.ignore_patterns('*.o','__pycache__','check_mapping'))
    for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
        if (dest/name).is_symlink():
            (dest/name).unlink()
    overlay = dest/'overlay'
    for name in ('epilogue_evt.hpp','taco_runtime.h','taco_codec.cuh','taco_runtime.cu','ths_op/gemm_reduce_scatter.cc'):
        source = ROOT/'src/gemm_rs'/name
        target = overlay/'gemm_rs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest['inputs'][str(source)] = sha(source)
    manifest['inputs'][str(PARENT/'manifest.json')] = sha(PARENT/'manifest.json')
    for name in ('taco.cu','taco.h'):
        p = ROOT.parent/'COCCL_TACO/src/device/compress/taco'/name
        manifest['inputs'][str(p)] = sha(p)
    shutil.copy2(HERE/'TACO_LICENSE.txt', out/'TACO_LICENSE.txt')
    shutil.copy2(ROOT/'python/flux/gemm_rs_taco.py', dest/'python/flux/gemm_rs_taco.py')
    manifest['inputs'][str(ROOT/'python/flux/gemm_rs_taco.py')] = sha(ROOT/'python/flux/gemm_rs_taco.py')
    # No policy-specific swizzle: it must remain byte-identical to original Flux.
    swizzle = 'gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
    assert sha(overlay/swizzle) == sha(PARENT/'original/overlay'/swizzle)
    manifest['original_swizzle_sha256'] = sha(overlay/swizzle)
    if swizzle_transform is not None:
        (overlay/swizzle).write_text(swizzle_transform((overlay/swizzle).read_text()))
    manifest['swizzle_sha256'] = sha(overlay/swizzle)

    def run(args, cwd=ROOT):
        args = list(map(str, args))
        manifest['commands'].append(dict(args=args, cwd=str(cwd)))
        (out/'build-progress.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(shlex.join(args), flush=True)
        subprocess.run(args, cwd=cwd, check=True)

    nvcc = '/data/apps/cuda/12.8/bin/nvcc'
    flags = ['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda','--expt-relaxed-constexpr',
             '-Xcompiler=-fPIC','-DFLUX_TACO_BASELINE','-gencode=arch=compute_80,code=[sm_80,compute_80]']
    if placement == 'separate':
        flags += ['-DFLUX_TACO_SEPARATE']
    includes = [overlay, BASE/'src', BASE/'include', BASE/'build', BASE/'3rdparty/cutlass/include',
                BASE/'3rdparty/cutlass/tools/util/include', BASE/'3rdparty/cutlass/tools/library/include',
                BASE/'3rdparty/cutlass/tools/profiler/include']
    inc = ['-I'+str(p) for p in includes]
    # The frozen CUTLASS header has TensorRef aliases requiring -fpermissive
    # with GCC host-only compilation (the nvcc compilation does not need it).
    run(['/usr/bin/g++','-std=c++17','-fpermissive','-O2','-I/data/apps/cuda/12.8/include',*inc,HERE/'check_layout.cc','-o',out/'check_layout'])
    run([out/'check_layout'])
    helper = dest/'taco_runtime.o'
    run([nvcc,*flags,*inc,'-c',overlay/'gemm_rs/taco_runtime.cu','-o',helper])
    registration = next((BASE/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
    obj, dlink = dest/'registration.o', dest/'device_link.o'
    run([nvcc,*flags,*inc,'-Xptxas=-v','-c',registration,'-o',obj])
    work = BASE/'build/src/cuda'
    objects = [obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve()
               for p in shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())]+[helper]
    run([nvcc,*flags,'-shared','-dlink',*objects,'-o',dlink,'-L/data/apps/cuda/12.8/lib64',
         '-lcudart_static','-lcudadevrt','-ldl','-lrt','-lpthread'])
    version = out/'version.ld'
    version.write_text((BASE/'src/cuda/version.ld').read_text().replace('global:', 'global:\n    taco_*;'))
    link = shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
    for i, arg in enumerate(link):
        if arg.endswith('.o'):
            link[i] = str(obj if 'registers/flux_bf16_bf16_void' in arg else dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
        if '--version-script=' in arg:
            link[i] = '-Wl,--version-script='+str(version)
    link[link.index('-o')+1] = str(dest/'libflux_cuda.so')
    link.append(str(helper))
    run(link, work)

    entries = json.loads((BASE/'build/compile_commands.json').read_text())
    entry = next(e for e in entries if e['file'].endswith('/gemm_rs/ths_op/gemm_reduce_scatter.cc'))
    compile_args = shlex.split(entry['command'])
    compile_args[1:1] = ['-DFLUX_TACO_BASELINE','-I'+str(overlay)]
    if placement == 'separate':
        compile_args.insert(1, '-DFLUX_TACO_SEPARATE')
    compile_args[compile_args.index('-c')+1] = str(overlay/'gemm_rs/ths_op/gemm_reduce_scatter.cc')
    wrapper = dest/'gemm_reduce_scatter.cc.o'
    compile_args[compile_args.index('-o')+1] = str(wrapper)
    run(compile_args, Path(entry['directory']))
    work = BASE/'build/src/ths_op'
    link = shlex.split((work/'CMakeFiles/flux_cuda_ths_op.dir/link.txt').read_text())
    ipc = ROOT/'outputs/a800/phase3/ipc-fix-build/flux_shm.cc.o'
    assert ipc.is_file()
    manifest['inputs'][str(ipc)] = sha(ipc)
    for i, arg in enumerate(link):
        if arg.endswith('.o'):
            link[i] = str(wrapper if arg.endswith('/gemm_reduce_scatter.cc.o') else ipc if arg.endswith('/flux_shm.cc.o') else (work/arg).resolve())
        if arg == '../../lib/libflux_cuda.so':
            link[i] = str(dest/'libflux_cuda.so')
    link[link.index('-o')+1] = str(dest/'libflux_cuda_ths_op.so')
    run(link, work)
    for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
        p = dest/'python/flux/lib'/name
        p.unlink()
        p.symlink_to(dest/name)
    for name, digest in parent['files'].items():
        assert sha(PARENT/name) == digest, ('Parent artifact changed', name)
    for p in out.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.name not in ('manifest.json','build-progress.json'):
            manifest['files'][str(p.relative_to(out))] = sha(p)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('TACO COMPILE/LINK COMPLETE (GPU validation still required)', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/a800/phase4/taco-build')
    parser.add_argument('--placement', choices=['fused','separate'], default='fused')
    args=parser.parse_args()
    main(args.out, args.placement)
