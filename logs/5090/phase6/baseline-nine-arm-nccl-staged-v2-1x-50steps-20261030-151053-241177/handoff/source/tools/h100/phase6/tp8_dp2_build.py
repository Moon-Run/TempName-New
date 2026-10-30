"""Compile the immutable portable sources locally; never downloads dependencies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tp8_dp2_protocol import require, sha, read, write

EXPORTS = ('taco_ma4f_protocol_version','taco_ma4f_decode_reduce',
           'ma9_zero_count_protocol','ma9_zero_count_chunk_size','ma9_zero_count_cuda',
           'ma11_softmax_protocol','ma11_softmax_forward_cuda','ma11_softmax_backward_cuda')


def artifact_paths(root):
    result = []
    for arm in ('ma11','flux-basic'):
        result += [root/arm/'python/flux/lib'/name for name in ('libflux_cuda.so','libflux_cuda_ths_op.so')]
        extension = list((root/arm/'python').glob('flux_ths_pybind*.so'))
        require(len(extension) == 1, 'Exactly one Flux Python extension required: '+arm)
        result += extension
        result += [p for p in (root/arm/'python').rglob('*.py') if '__pycache__' not in p.parts]
    result += [root/'codec-basic/libbasic_quant.so', root/'codec-cuda/libbaseline_taco.so']
    result += list((root/'megatron').rglob('*.py'))
    helpers=list((root/'megatron/megatron/core/datasets').glob('helpers*.so'))
    require(len(helpers)==1,'Per-node Megatron dataset helper is missing')
    result += helpers
    return result


def verify(root, package_id):
    root = Path(root)
    record = read(root/'build.json')
    require(record['status'] == 'built' and record['package_id'] == package_id, 'Incomplete or stale build')
    for name,value in record['files'].items():
        require(sha(root/name) == value, 'Built artifact changed: '+name)
    require(set(record['files']) == {str(p.relative_to(root)) for p in artifact_paths(root)}, 'Build artifact set changed')
    return record


def build(package, root, cuda_home, jobs, package_id):
    import torch
    package,root,cuda_home = map(Path,(package,root,cuda_home))
    identity = dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                    nvcc_sha256=sha(cuda_home/'bin/nvcc'), recipe_sha256=sha(__file__))
    if root.exists():
        record = verify(root, package_id)
        require(record['environment'] == identity, 'Build Python/compiler environment changed')
        return record
    require(jobs > 0, 'Positive compiler jobs required')
    root.mkdir(parents=True)
    record = dict(status='building',package_id=package_id,environment=identity,commands=[],gpu_validated=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home), CUDACXX=str(cuda_home/'bin/nvcc'),
               PATH=f'{Path(sys.executable).parent}:{cuda_home}/bin:'+os.environ['PATH'],
               TORCH_CUDA_ARCH_LIST='9.0',FLUX_FORCE_BUILD='1',FLUX_SHM_USE_NVSHMEM='0',MAX_JOBS=str(jobs),
               CMAKE_BUILD_PARALLEL_LEVEL=str(jobs),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    def run(command,cwd):
        command=list(map(str,command))
        record['commands'].append(dict(command=command,cwd=str(cwd)))
        write(root/'build.json',record)
        print('BUILD', ' '.join(command),flush=True)
        with (root/'compile.log').open('a') as log:
            log.write('\nCOMMAND '+repr(command)+'\n');log.flush()
            subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        # Megatron's initializer compiles dataset helpers on global rank 0.
        # The containers have independent filesystems: prebuild on BOTH nodes.
        shutil.copytree(package/'deps/Megatron-LM',root/'megatron')
        run(['make','-C',root/'megatron/megatron/core/datasets'],root)
        nccl = root/'nccl'
        shutil.copytree(package/'deps/nccl', nccl)
        run(['make',f'-j{jobs}','src.build','NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90'],nccl)
        nccl = nccl/'build'
        require((nccl/'include/nccl.h').is_file() and (nccl/'lib/libnccl_static.a').is_file(), 'NCCL development build missing')
        env['NCCL_ROOT']=str(nccl)
        for arm in ('ma11','flux-basic'):
            src = root/arm
            shutil.copytree(package/'native'/arm,src,symlinks=True)
            link = src/'3rdparty/cutlass'
            link.unlink()
            link.symlink_to(os.path.relpath(package/'deps/cutlass',link.parent),target_is_directory=True)
            run(['cmake','-S',src,'-B',src/'build','-DCUDAARCHS=90','-DGPU_SM_CORES=132',
                 '-DENABLE_NVSHMEM=OFF','-DWITH_PROTOBUF=OFF','-DBUILD_TEST=OFF','-DCMAKE_BUILD_TYPE=Release',
                 f'-DPYTHON_EXECUTABLE={sys.executable}',f'-DPython_EXECUTABLE={sys.executable}',
                 f'-DCMAKE_CXX_FLAGS=-I{nccl}/include',f'-DCMAKE_CUDA_FLAGS=-I{nccl}/include',
                 f'-DCMAKE_INSTALL_PREFIX={src}/python/flux'],src)
            if arm == 'ma11':
                regs = list((src/'build/src/gemm_rs/registers').glob('*sm80*.cu'))
                require(len(regs)==1, 'Expected one frozen V2 registration')
                regs[0].write_text(regs[0].read_text()+'''
extern "C" __attribute__((visibility("default"))) int h100_v2_occupancy() {
  using Device = decltype(GemmDevice_0().gemm_device());
  return Device::maximum_active_blocks();
}
''')
            else:
                regs = list((src/'build/src/comm_none/registers').glob('*.cu'))
                require(len(regs)==1 and 'sm90' in regs[0].name.lower(), 'Expected one native SM90 GemmOnly registration')
            run(['cmake','--build',src/'build','--parallel',jobs],src)
            run(['cmake','--install',src/'build'],src)
            run([sys.executable,'setup.py','build_ext','--inplace'],src)
            for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                (src/lib).symlink_to(Path('python/flux/lib')/lib)
            if arm == 'ma11':
                output = subprocess.check_output(['nm','-D','--defined-only','--format=posix',str(src/'libflux_cuda.so')],text=True)
                symbols = {line.split()[0].split('@')[0] for line in output.splitlines() if line.strip()}
                require(set(EXPORTS) <= symbols, 'MA11 native exports missing: '+repr(set(EXPORTS)-symbols))
        for name,source,lib in [('codec-basic','basic_quant.cu','libbasic_quant.so'),
                                ('codec-cuda','baseline_codec.cu','libbaseline_taco.so')]:
            src = root/name
            shutil.copytree(package/'native'/name,src)
            command = [cuda_home/'bin/nvcc','-std=c++17','-O3','-shared','-Xcompiler=-fPIC','-lineinfo',
                       '-gencode=arch=compute_90,code=sm_90','--cudart=shared',src/source,'-o',src/lib]
            host = os.environ.get('CUDAHOSTCXX') or os.environ.get('CXX')
            if host: command += ['-ccbin',host]
            run(command,src)
        record.update(status='built',files={str(p.relative_to(root)):sha(p) for p in artifact_paths(root)})
        write(root/'build.json',record)
        return verify(root,package_id)
    except BaseException as error:
        record.update(status='failed',error=str(error));write(root/'build.json',record);raise
