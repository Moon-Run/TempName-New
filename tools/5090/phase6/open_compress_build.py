"""Isolated SM120 GemmOnly builds and open INT8 codec; no TACO build or runtime."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from rtx5090_loader import HERE, REPO
import op1_a_swizzle
from common import MEGATRON_COMMIT, read, write, require, sha, verify_files, replace_once

PROTOCOL = 'rtx5090-open-compress-native-sm120-v1'
MANIFEST = 'open-compress-build.json'
ARMS = ('flux-bf16', 'flux-int8', 'flux-int8-interleaved')
VARIANTS = ('flux-original', 'flux-interleaved')


def native_root(root, arm):
    require(arm in ARMS, 'Unknown open compression arm')
    return Path(root).resolve()/('flux-interleaved' if arm.endswith('-interleaved') else 'flux-original')


def libraries(root, arm):
    native = native_root(root, arm)
    return {name: dict(path=str((native/name).resolve()), sha256=sha(native/name))
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}


def codec_library(root):
    path = Path(root).resolve()/'codec/libopen_compress.so'
    return dict(path=str(path), sha256=sha(path))


def inputs():
    names = ('open_compress_build.py', 'open_compress_codec.cu', 'open_compress_codec.py',
             'gemm_only_generator.cc', 'op1_a_swizzle.py', 'op1_a_swizzle.hpp', 'op1_a_swizzle_probe.cuh')
    return {name: sha(HERE/name) for name in names}


def dependency_identity():
    result = {}
    for name in ('cutlass', 'nccl'):
        path = REPO/'3rdparty'/name
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain',
                                        '--untracked-files=no'], text=True)
        require(not dirty.strip(), 'Tracked dependency changes are not allowed: '+name)
        result[name] = dict(commit=revision, tracked_clean=True)
    return result


def source_overrides(source):
    """Narrow ordinary Flux source to the existing RTX5090 GemmOnly registration."""
    source = Path(source)
    allowed = {'add_subdirectory(comm_none)', 'add_subdirectory(cuda)', 'add_subdirectory(ths_op)'}
    cmake = '\n'.join(line for line in (source/'src/CMakeLists.txt').read_text().split('\n')
                      if not line.strip().startswith('add_subdirectory(') or line.strip() in allowed)
    generator = '\n'.join(line for line in (source/'src/generator/CMakeLists.txt').read_text().split('\n')
                          if not line.startswith('add_executable(') or line.startswith('add_executable(gen_comm_none '))
    return {'src/CMakeLists.txt': cmake, 'src/generator/CMakeLists.txt': generator,
        'src/generator/gen_comm_none.cc': (HERE/'gemm_only_generator.cc').read_text(),
        'src/comm_none/CMakeLists.txt': replace_once((source/'src/comm_none/CMakeLists.txt').read_text(),
            'file(GLOB TUNING_CONFIGS tuning_config/*.cu)', 'set(TUNING_CONFIGS)'),
        'setup.py': replace_once((source/'setup.py').read_text(),
            'setup_requires=["torch", "cmake", "packaging"],', 'setup_requires=["torch", "packaging"],'),
        'CMakeLists.txt': replace_once((source/'CMakeLists.txt').read_text(), 'add_subdirectory(src)',
            'include_directories("$ENV{NCCL_ROOT}/include")\nadd_subdirectory(src)')}


def _snapshot(destination):
    ignore = shutil.ignore_patterns('.git', '__pycache__', '*.so', '*.so.*', '*.o', '*.a', '*.pyc', 'build', '*.egg-info')
    for name in ('src', 'include', 'cmake', 'python'):
        shutil.copytree(REPO/name, destination/name, ignore=ignore)
    for name in ('CMakeLists.txt', 'setup.py', 'LICENSE', 'README.md'):
        shutil.copy2(REPO/name, destination/name)
    (destination/'3rdparty').mkdir()
    require((REPO/'3rdparty/cutlass/include/cutlass/cutlass.h').is_file(), 'Initialize CUTLASS submodule')
    require((REPO/'3rdparty/nccl/src/nccl.h.in').is_file(), 'Initialize NCCL submodule')
    (destination/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)


def prepare(root):
    root = Path(root).resolve()
    require(not root.exists(), 'Use a fresh open compression build directory')
    source_inputs = inputs()
    dependencies = dependency_identity()
    root.mkdir(parents=True)
    variants = {}
    for name in VARIANTS:
        destination = root/name
        _snapshot(destination)
        updates = source_overrides(destination)
        if name == 'flux-interleaved':
            updates.update(op1_a_swizzle.overrides(destination))
        for path, content in updates.items():
            (destination/path).write_text(content)
        variants[name] = dict(source_sha256={str(path.relative_to(destination)): sha(path)
            for path in sorted(destination.rglob('*')) if path.is_file() and not path.is_symlink()},
            modified_files=sorted(updates))
    (root/'codec').mkdir()
    shutil.copy2(HERE/'open_compress_codec.cu', root/'codec/open_compress_codec.cu')
    write(root/MANIFEST, dict(protocol=PROTOCOL, status='prepared', actual_arch='sm120', gpu_validated=False,
        arms=list(ARMS), variants=variants, inputs=source_inputs, files={}, registrations={},
        source_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        submodules=dependencies,
        codec_source_sha256=sha(root/'codec/open_compress_codec.cu'),
        taco_build_required=False, taco_runtime_required=False, peer_memory_required=False,
        gemm_schedule='Sm89/RTX5090 GemmV2 StreamkSK BF16-RCR 128x128x32 stage3; SM120 cubin'))
    verify(root)
    return root


def verify(root, built=False):
    root = Path(root).resolve()
    record = read(root/MANIFEST)
    require(record['protocol'] == PROTOCOL and record['actual_arch'] == 'sm120'
            and record['arms'] == list(ARMS) and set(record['variants']) == set(VARIANTS),
            'Wrong open compression build identity')
    require(record['inputs'] == inputs(), 'Open compression build inputs changed')
    require(record['submodules'] == dependency_identity(), 'Dependency commit/worktree changed')
    require(record['taco_build_required'] is False and record['taco_runtime_required'] is False,
            'Open compression must not depend on TACO')
    require(sha(root/'codec/open_compress_codec.cu') == record['codec_source_sha256']
            == record['inputs']['open_compress_codec.cu'], 'Codec source changed')
    for name in VARIANTS:
        link = root/name/'3rdparty/cutlass'
        require(link.is_symlink() and link.resolve() == (REPO/'3rdparty/cutlass').resolve(),
                'CUTLASS dependency symlink changed')
        verify_files(root/name, record['variants'][name]['source_sha256'])
    verify_files(root, record['files'])
    if built:
        require(record['status'] == 'built' and set(record['registrations']) == set(VARIANTS),
                'Compile both GemmOnly variants before use')
        for name in VARIANTS:
            item = record['registrations'][name]
            text = (root/item['path']).read_text()
            require(sha(root/item['path']) == item['sha256'], 'Changed actual GemmOnly registration')
            original = op1_a_swizzle.original_registration(text) if name == 'flux-interleaved' else text
            require(hashlib.sha256(original.encode()).hexdigest() == item['generated_sha256']
                    and '_streamksk_' in original.lower() and '_streamkdp_' not in original.lower(),
                    'Original StreamK registration changed')
            require(len(list((root/name/'python').glob('flux_ths_pybind*.so'))) == 1,
                    'Missing Flux Python extension')
        require('codec/libopen_compress.so' in record['files']
                and any(name.startswith('megatron/') for name in record['files']), 'Missing codec/Megatron build')
        for name, expected in record.get('nccl_files', {}).items():
            require(sha(name) == expected, 'NCCL dependency changed')
        from native_exports import verify_native_exports
        verify_native_exports(root/'flux-interleaved/libflux_cuda.so', op1_a_swizzle.EXPORTS)
        for arm in ARMS:
            libraries(root, arm)
        codec_library(root)
    return record


def environment_identity(cuda_home):
    import torch
    nvcc = Path(cuda_home)/'bin/nvcc'
    require(nvcc.is_file(), 'CUDA Toolkit nvcc is required')
    codes = subprocess.check_output([str(nvcc), '--list-gpu-code'], text=True)
    require('sm_120' in codes.split(), 'CUDA Toolkit must support sm_120 (CUDA >= 12.8)')
    runtime = tuple(int(x) for x in (torch.version.cuda or '0.0').split('.')[:2])
    arch_flags = torch._C._cuda_getArchFlags() if hasattr(torch._C, '_cuda_getArchFlags') else ''
    require(runtime >= (12, 8) and 'sm_120' in (arch_flags or '').split(),
            'Use a PyTorch CUDA >= 12.8 build containing SM120 kernels')
    return dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), nvcc_sha256=sha(nvcc))


def megatron_check(path):
    require(subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            == MEGATRON_COMMIT, 'Wrong Megatron commit')
    require(not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain',
                                        '--untracked-files=no'], text=True).strip(),
            'Megatron tracked files must match the pinned commit')


def archive_megatron(repo, target):
    """Freeze the pinned dependency without importing historical campaign builders."""
    target = Path(target)
    data = subprocess.check_output(['git', '-C', str(repo), 'archive', '--format=tar', MEGATRON_COMMIT])
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as handle:
        members = handle.getmembers()
        require(all(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                    and (member.isdir() or member.isfile() or member.issym()) for member in members),
                'Unexpected Megatron archive member')
        for member in members:
            if member.issym():
                require(not Path(member.linkname).is_absolute() and
                        (target/Path(member.name).parent/member.linkname).resolve().is_relative_to(target.resolve()),
                        'Megatron archive symlink escapes snapshot')
        handle.extractall(target, members=members)
    return dict(commit=MEGATRON_COMMIT, archive_sha256=hashlib.sha256(data).hexdigest(), paths=[])


def compile_build(root, megatron, cuda_home, jobs=2):
    root, megatron, cuda_home = map(lambda path: Path(path).resolve(), (root, megatron, cuda_home))
    record = verify(root)
    require(record['status'] == 'prepared' and type(jobs) is int and jobs > 0, 'Compile a fresh prepared build')
    megatron_check(megatron)
    environment = environment_identity(cuda_home)
    require(record['submodules'] == dependency_identity(), 'Build dependency changed')
    record.update(status='building', environment=environment, commands=[])
    write(root/MANIFEST, record)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home), CUDACXX=str(cuda_home/'bin/nvcc'),
        PATH=f'{Path(sys.executable).parent}:{cuda_home}/bin:' + os.environ.get('PATH', ''),
        TORCH_CUDA_ARCH_LIST='12.0', FLUX_FORCE_BUILD='1', FLUX_SHM_USE_NVSHMEM='0',
        MAX_JOBS=str(jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('FLUX_FORCE_NVLINK', None)
    (root/'compiler-tmp').mkdir()
    env['TMPDIR'] = str(root/'compiler-tmp')

    def run(command, cwd=root):
        command = list(map(str, command))
        record['commands'].append(dict(args=command, cwd=str(cwd)))
        write(root/MANIFEST, record)
        with (root/'compile.log').open('a') as output:
            output.write('\nCOMMAND '+repr(command)+'\n')
            output.flush()
            subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)

    try:
        record['megatron'] = archive_megatron(megatron, root/'megatron')
        run(['make', '-C', root/'megatron/megatron/core/datasets'])
        nccl = root/'nccl-source'
        shutil.copytree(REPO/'3rdparty/nccl', nccl, ignore=shutil.ignore_patterns('.git', 'build'))
        run(['make', f'-j{jobs}', 'src.build', 'NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120'], nccl)
        nccl = nccl/'build'
        require((nccl/'lib/libnccl_static.a').is_file(), 'NCCL static build missing')
        env['NCCL_ROOT'] = str(nccl)
        record['nccl_files'] = {str(nccl/name): sha(nccl/name) for name in ('include/nccl.h', 'lib/libnccl_static.a')}
        for name in VARIANTS:
            source = root/name
            run(['cmake', '-S', source, '-B', source/'build', '-DCUDAARCHS=120', '-DGPU_SM_CORES=170',
                 '-DENABLE_NVSHMEM=OFF', '-DWITH_PROTOBUF=OFF', '-DBUILD_TEST=OFF', '-DCMAKE_BUILD_TYPE=Release',
                 f'-DPYTHON_EXECUTABLE={sys.executable}', f'-DPython_EXECUTABLE={sys.executable}',
                 f'-DCMAKE_CXX_FLAGS=-I{nccl}/include', f'-DCMAKE_CUDA_FLAGS=-I{nccl}/include',
                 f'-DCMAKE_INSTALL_PREFIX={source}/python/flux'])
            registrations = list((source/'build/src/comm_none/registers').glob('*.cu'))
            require(len(registrations) == 1 and 'sm89' in registrations[0].name.lower(), 'Expected one Sm89 GemmOnly registration')
            path = registrations[0]
            original = path.read_text()
            require('_streamksk_' in original.lower() and '_streamkdp_' not in original.lower(), 'Preserve original Flux StreamkSK')
            if name == 'flux-interleaved':
                path.write_text(op1_a_swizzle.registration_source(original))
            record['registrations'][name] = dict(path=str(path.relative_to(root)), sha256=sha(path),
                generated_sha256=hashlib.sha256(original.encode()).hexdigest())
            run(['cmake', '--build', source/'build', '--parallel', jobs])
            run(['cmake', '--install', source/'build'])
            run([sys.executable, 'setup.py', 'build_ext', '--inplace'], source)
            for library in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                (source/library).symlink_to(Path('python/flux/lib')/library)
            for path in list((source/'python').rglob('*')) + [source/'libflux_cuda.so', source/'libflux_cuda_ths_op.so', path]:
                if path.is_file() and '__pycache__' not in path.parts:
                    record['files'][str(path.relative_to(root))] = sha(path)
        run([cuda_home/'bin/nvcc', '-std=c++17', '-O3', '-shared', '-Xcompiler=-fPIC',
             '-gencode=arch=compute_120,code=sm_120', root/'codec/open_compress_codec.cu',
             '-o', root/'codec/libopen_compress.so'])
        record['files']['codec/libopen_compress.so'] = sha(root/'codec/libopen_compress.so')
        record['files'].update({str(path.relative_to(root)): sha(path) for path in (root/'megatron').rglob('*') if path.is_file()})
        require(len(list((root/'megatron/megatron/core/datasets').glob('helpers*.so'))) == 1, 'Missing Megatron dataset helper')
        record['status'] = 'built'
        write(root/MANIFEST, record)
        return verify(root, True)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(root/MANIFEST, record)
        raise
