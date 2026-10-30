"""Build a separate native SM90 Flux GemmOnly library; never alter A/B/C."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import build as parent
from common import HERE, REPO, read, require, replace_once, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-gpt67-tp8-ablation6-gemm-only-v1'
VARIANT = 'flux-gemm-only'


def prepare(out, source_only=False):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    src = out/VARIANT; parent.snapshot(src)
    p = src/'src/CMakeLists.txt'
    allowed = {'add_subdirectory(comm_none)', 'add_subdirectory(cuda)', 'add_subdirectory(ths_op)'}
    p.write_text('\n'.join(line for line in p.read_text().split('\n')
                           if not line.strip().startswith('add_subdirectory(') or line.strip() in allowed))
    p = src/'src/generator/CMakeLists.txt'
    p.write_text('\n'.join(line for line in p.read_text().split('\n')
                           if not line.startswith('add_executable(') or line.startswith('add_executable(gen_comm_none ')))
    shutil.copy2(HERE/'ablation6_gemm_only_generator.cc', src/'src/generator/gen_comm_none.cc')
    p = src/'src/comm_none/CMakeLists.txt'
    p.write_text(replace_once(p.read_text(), 'file(GLOB TUNING_CONFIGS tuning_config/*.cu)', 'set(TUNING_CONFIGS)'))
    p = src/'setup.py'
    p.write_text(replace_once(p.read_text(), 'setup_requires=["torch", "cmake", "packaging"],',
                             'setup_requires=["torch", "packaging"],'))
    p = src/'CMakeLists.txt'
    p.write_text(replace_once(p.read_text(), 'add_subdirectory(src)',
                             'include_directories("$ENV{NCCL_ROOT}/include")\nadd_subdirectory(src)'))
    record = dict(protocol=PROTOCOL, kind='flux_gemm_only', status='prepared', source_only=source_only,
        diagnostic_only=source_only, gpu_validated=False, config=model_config(), variant=VARIANT,
        arch='sm90', algorithm='native Hopper GemmV3 / CommNone / BF16-RCR / no TACO or ReduceScatter',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        inputs={str(p): sha(p) for p in (HERE/'ablation6_build.py', HERE/'ablation6_gemm_only_generator.cc')},
        submodules={n:subprocess.check_output(['git','-C',str(REPO/'3rdparty'/n),'rev-parse','HEAD'],text=True).strip()
                    for n in ('cutlass','nccl')},
        source_sha256={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file() and not p.is_symlink()}, files={})
    write(out/'manifest.json', record)
    verify(out)
    return out


def validate_snapshot(record, built=False):
    require(record['protocol'] == PROTOCOL and record['kind'] == 'flux_gemm_only' and
            record['variant'] == VARIANT and record['arch'] == 'sm90' and record['config'] == model_config() and
            record['source_only'] == record['diagnostic_only'], 'Wrong GemmOnly build/model')
    require(record['source_sha256']['src/generator/gen_comm_none.cc'] == sha(HERE/'ablation6_gemm_only_generator.cc'),
            'GemmOnly registry changed')
    # These files define the actual GEMM and exported frontend; the build only
    # narrows registration/CMake and leaves their upstream numerical code intact.
    for name in ('src/comm_none/ths_op/gemm_only.cc', 'src/comm_none/gemm_v3_comm_none.hpp',
                 'src/pybind/gemm_only.cc', 'python/flux/cpp_mod.py'):
        require(record['source_sha256'][name] == sha(REPO/name), 'Native GemmOnly source changed: '+name)
    if built:
        require(record['status'] == 'built' and not record['source_only'] and
                all(VARIANT+'/'+n in record['files'] for n in ('libflux_cuda.so','libflux_cuda_ths_op.so')),
                'Real compiled GemmOnly build required')


def verify(root, built=False):
    root = Path(root); record = read(root/'manifest.json')
    validate_snapshot(record, built)
    verify_files(root/VARIANT, record['source_sha256']); verify_files(root, record['files'])
    return record


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    record = verify(root); src = root/VARIANT
    require(jobs > 0 and (cuda_home/'bin/nvcc').is_file(), 'CUDA Toolkit and positive jobs required')
    for n, commit in record['submodules'].items():
        require(subprocess.check_output(['git','-C',str(REPO/'3rdparty'/n),'rev-parse','HEAD'],text=True).strip()==commit,
                'Build submodule changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home), CUDACXX=str(cuda_home/'bin/nvcc'),
        PATH=f'{Path(sys.executable).parent}:{cuda_home}/bin:'+os.environ['PATH'], TORCH_CUDA_ARCH_LIST='9.0',
        FLUX_FORCE_BUILD='1', FLUX_SHM_USE_NVSHMEM='0', MAX_JOBS=str(jobs),
        CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), PYTHONNOUSERSITE='1')
    record.update(status='compiling', commands=[])
    def run(cmd, cwd=src):
        cmd=list(map(str,cmd)); record['commands'].append(dict(args=cmd,cwd=str(cwd))); write(root/'manifest.json',record)
        with (root/'compile.log').open('a') as log:
            log.write('\nCOMMAND '+repr(cmd)+'\n'); log.flush()
            subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if nccl_root is None:
            nccl=root/'nccl-source'
            if not nccl.exists(): shutil.copytree(REPO/'3rdparty/nccl',nccl,ignore=shutil.ignore_patterns('.git','build'))
            run(['make',f'-j{jobs}','src.build','NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90'],nccl)
            nccl_root=nccl/'build'
        nccl_root=Path(nccl_root).resolve()
        require((nccl_root/'include/nccl.h').is_file() and (nccl_root/'lib/libnccl_static.a').is_file(), 'NCCL development files missing')
        env['NCCL_ROOT']=str(nccl_root)
        record['nccl_files']={str(p):sha(p) for p in (nccl_root/'include/nccl.h',nccl_root/'lib/libnccl_static.a')}
        run(['cmake','-S',src,'-B',src/'build','-DCUDAARCHS=90','-DGPU_SM_CORES=132',
             '-DENABLE_NVSHMEM=OFF','-DWITH_PROTOBUF=OFF','-DBUILD_TEST=OFF','-DCMAKE_BUILD_TYPE=Release',
             f'-DPYTHON_EXECUTABLE={sys.executable}',f'-DPython_EXECUTABLE={sys.executable}',
             f'-DCMAKE_CXX_FLAGS=-I{nccl_root}/include',f'-DCMAKE_CUDA_FLAGS=-I{nccl_root}/include',
             f'-DCMAKE_INSTALL_PREFIX={src}/python/flux'])
        registrations=list((src/'build/src/comm_none/registers').glob('*.cu'))
        require(len(registrations)==1 and 'sm90' in registrations[0].name.lower(), 'Expected one native BF16 SM90 GemmOnly registry')
        run(['cmake','--build',src/'build','--parallel',jobs]); run(['cmake','--install',src/'build'])
        run([sys.executable,'setup.py','build_ext','--inplace'])
        for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            if not (src/name).is_symlink(): (src/name).symlink_to(Path('python/flux/lib')/name)
        for p in list((src/'python').rglob('*'))+[src/'libflux_cuda.so',src/'libflux_cuda_ths_op.so']:
            if p.is_file() and '__pycache__' not in p.parts: record['files'][str(p.relative_to(root))]=sha(p)
        record.update(status='built',gpu_validated=False)
    except BaseException:
        record['status']='compile_failed'; raise
    finally:
        write(root/'manifest.json',record)
