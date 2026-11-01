"""Isolated old A/C ablations; common GemmOnly, codec and Megatron remain frozen."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from rtx5090_loader import HERE, REPO, load
import flux_baseline_native as native
from ma2_ablation_build import copy_registered
from native_exports import verify_native_exports

baseline = load('build')
import flux_baseline_protocol as p
read, write, require, sha = p.read, p.write, p.require, p.sha
PROTOCOL = 'rtx5090-flux-quant-native-sm120-v1'
MANIFEST = 'flux-build.json'
ARMS = p.ARMS
NEW_ARMS = ('a', 'c')
ALIASES = {}
OCCUPANCY_EXPORT = '''\nextern "C" __attribute__((visibility("default"))) int h100_v2_occupancy() {
  using Device = decltype(GemmDevice_0().gemm_device());
  return Device::maximum_active_blocks();
}
'''


def _inputs():
    return dict(recipe_sha256=sha(__file__), native=native.identity())



def base_root(root):
    return Path(read(Path(root)/MANIFEST)['baseline_root'])


def native_root(root, arm):
    require(arm in (*ARMS, *NEW_ARMS), 'Unknown Flux baseline arm: '+str(arm))
    root = Path(root).resolve()
    if arm in p.BASIC_ARMS:
        return baseline.native_root(base_root(root), 'flux-basic')
    return root/p.NATIVE.get(arm, arm)



def libraries(root, arm):
    source = native_root(root, arm)
    return {name: dict(path=str((source/name).resolve()), sha256=sha(source/name))
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}


def _sources(base, arm):
    require(arm in NEW_ARMS, 'Expected old A/C native source')
    source = baseline.native_root(base, arm)
    hashes = read(base/'selective/manifest.json')['variants'][arm]['source_sha256']
    updates = native.overrides(source, arm)
    expected = dict(hashes, **{name: hashlib.sha256(content.encode()).hexdigest()
                              for name, content in updates.items()})
    return source, hashes, updates, expected



def _verify_files(root, files):
    for name, expected in files.items():
        path = Path(root)/name
        require(path.is_file() and sha(path) == expected, 'Missing/changed Flux baseline build input: ' + str(path))


def prepare(root, baseline_root):
    root, base = Path(root).resolve(), Path(baseline_root).resolve()
    require(not root.exists(), 'Use a fresh Flux baseline build directory')
    # Preparation is also usable for CPU-only source inspection. Compilation
    # and execution explicitly require the baseline's complete built evidence.
    old = baseline.verify(base)
    root.mkdir(parents=True)
    variants = {}
    for arm in NEW_ARMS:
        source, hashes, updates, expected = _sources(base, arm)
        destination = root/arm
        copy_registered(source, destination, hashes)
        for name, content in updates.items():
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        variants[arm] = dict(source_sha256=expected, modified_files=sorted(updates),
                             parent_source_sha256=hashes)
    write(root/MANIFEST, dict(protocol=PROTOCOL, status='prepared', actual_arch='sm120',
          baseline_root=str(base), baseline_manifest_sha256=sha(base/'baseline-build.json'),
          baseline_protocol=old['protocol'], submodules=old['submodules'], arms=list(ARMS),
          inputs=_inputs(), variants=variants, gpu_validated=False,
          ordering='original-Flux-rank-offset', all_remote=True,
          mathematical_reference='independent BF16 and all-remote codec reference; reduction changes permitted',
          files={}))
    verify(root)
    return root


def verify(root, built=False):
    root = Path(root).resolve()
    record = read(root/MANIFEST)
    require(record['protocol'] == PROTOCOL and record['actual_arch'] == 'sm120'
            and tuple(record['arms']) == ARMS and record['gpu_validated'] is False,
            'Wrong Flux baseline build protocol/architecture/arms')
    require(record['inputs'] == _inputs(), 'Flux baseline recipe/native/swizzle inputs changed')
    require(record['ordering'] == 'original-Flux-rank-offset' and record['all_remote'] is True,
            'Ordering/quantization contract changed')
    base = Path(record['baseline_root'])
    require(base.is_absolute() and record['baseline_manifest_sha256'] == sha(base/'baseline-build.json'),
            'Parent nine-arm build identity changed')
    old = baseline.verify(base, built)
    require(record['baseline_protocol'] == old['protocol'] and record['submodules'] == old['submodules'],
            'Parent baseline protocol/dependencies changed')
    require(set(record['variants']) == set(NEW_ARMS), 'Incomplete Flux baseline variants')
    for arm in NEW_ARMS:
        _, hashes, updates, expected = _sources(base, arm)
        require(record['variants'][arm] == dict(source_sha256=expected, modified_files=sorted(updates),
                                               parent_source_sha256=hashes),
                'Unregistered Flux baseline source transformation: ' + arm)
        _verify_files(root/arm, expected)
    _verify_files(root, record['files'])
    if built:
        require(record['status'] == 'built', 'Flux baseline build incomplete')
        require(record.get('environment') == old.get('environment'), 'Flux baseline and baseline Python/CUDA ABI differ')
        require(set(record.get('registrations', {})) == set(NEW_ARMS), 'Missing Flux baseline compiled registrations')
        for arm in NEW_ARMS:
            source = native_root(root, arm)
            registered = record['registrations'][arm]
            path = root/registered['path']
            require(path.is_relative_to(source/'build/src') and sha(path) == registered['sha256'],
                    'Flux baseline generated registration identity changed: ' + arm)
            transformed = path.read_text()
            require(transformed.endswith(OCCUPANCY_EXPORT), 'Missing exact-kernel occupancy export')
            original = native.original_registration(transformed[:-len(OCCUPANCY_EXPORT)])
            require(hashlib.sha256(original.encode()).hexdigest() == registered['generated_sha256'],
                    'Flux baseline registration has unregistered changes: ' + arm)
            require(len(list((source/'python').glob('flux_ths_pybind*.so'))) == 1,
                    'Missing Flux baseline Python extension: ' + arm)
            require(all(str((source/name).relative_to(root)) in record['files']
                        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')),
                    'Missing Flux baseline library hashes: ' + arm)
            libraries(root, arm)
        import local_sources
        for arm in NEW_ARMS:
            require((root/arm/'check_mapping').is_file() and arm+'/check_mapping' in record['files'],
                    'Missing original-order mapping executable')
            verify_native_exports(root/arm/'libflux_cuda.so', ('h100_v2_occupancy',) + local_sources.NATIVE_EXPORTS)
        for name, expected in record.get('nccl_files', {}).items():
            require(sha(name) == expected, 'NCCL build dependency changed: ' + name)
    return record


def compile_build(root, cuda_home, jobs=2):
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    record = verify(root)
    require(record['status'] == 'prepared' and type(jobs) is int and jobs > 0,
            'Compile a fresh prepared Flux baseline build with positive jobs')
    old = baseline.verify(base_root(root), True)
    environment = baseline.environment_identity(cuda_home)
    require(environment == old['environment'], 'Use the same Python/CUDA/ABI as the baseline build')
    for name, revision in record['submodules'].items():
        require(subprocess.check_output(['git', '-C', str(REPO/'3rdparty'/name), 'rev-parse', 'HEAD'],
                                        text=True).strip() == revision,
                'Build dependency revision changed: ' + name)
    record.update(status='building', environment=environment, commands=[], registrations={})
    write(root/MANIFEST, record)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home),
               CUDACXX=str(cuda_home/'bin/nvcc'),
               PATH=f'{Path(sys.executable).parent}:{cuda_home}/bin:' + os.environ.get('PATH', ''),
               TORCH_CUDA_ARCH_LIST='12.0', FLUX_FORCE_BUILD='1', FLUX_SHM_USE_NVSHMEM='0',
               MAX_JOBS=str(jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(jobs),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('FLUX_FORCE_NVLINK', None)
    temporary = root/'compiler-tmp'
    temporary.mkdir()
    env['TMPDIR'] = str(temporary)

    def run(arguments, cwd=root):
        arguments = list(map(str, arguments))
        record['commands'].append(dict(args=arguments, cwd=str(cwd)))
        write(root/MANIFEST, record)
        with (root/'compile.log').open('a') as output:
            output.write('\nCOMMAND ' + repr(arguments) + '\n')
            output.flush()
            subprocess.run(arguments, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)

    try:
        nccl = base_root(root)/'nccl-source/build'
        if not all((nccl/name).is_file() for name in ('include/nccl.h', 'lib/libnccl_static.a')):
            source = root/'nccl-source'
            shutil.copytree(REPO/'3rdparty/nccl', source, ignore=shutil.ignore_patterns('.git', 'build'))
            run(['make', f'-j{jobs}', 'src.build', 'NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120'], source)
            nccl = source/'build'
        require(all((nccl/name).is_file() for name in ('include/nccl.h', 'lib/libnccl_static.a')),
                'NCCL static build missing')
        env['NCCL_ROOT'] = str(nccl)
        record['nccl_files'] = {str(nccl/name): sha(nccl/name)
                                for name in ('include/nccl.h', 'lib/libnccl_static.a')}
        for arm in NEW_ARMS:
            source = native_root(root, arm)
            print('Compile RTX5090 Flux baseline variant:', arm, flush=True)
            run(['cmake', '-S', source, '-B', source/'build', '-DCUDAARCHS=120', '-DGPU_SM_CORES=170',
                 '-DENABLE_NVSHMEM=OFF', '-DWITH_PROTOBUF=OFF', '-DBUILD_TEST=OFF', '-DCMAKE_BUILD_TYPE=Release',
                 f'-DPYTHON_EXECUTABLE={sys.executable}', f'-DPython_EXECUTABLE={sys.executable}',
                 f'-DCMAKE_CXX_FLAGS=-I{nccl}/include', f'-DCMAKE_CUDA_FLAGS=-I{nccl}/include',
                 f'-DCMAKE_INSTALL_PREFIX={source}/python/flux'])
            operator = 'gemm_rs'
            registrations = list((source/'build/src'/operator/'registers').glob('*.cu'))
            architecture = 'sm80'
            require(len(registrations) == 1 and architecture in registrations[0].name.lower(),
                    'Wrong Flux baseline V2 registration: ' + arm)
            registration = registrations[0]
            original = registration.read_text()
            transformed = native.registration_source(original) + OCCUPANCY_EXPORT
            registration.write_text(transformed)
            record['registrations'][arm] = dict(path=str(registration.relative_to(root)),
                                                generated_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                                sha256=sha(registration))
            run(['cmake', '--build', source/'build', '--parallel', jobs])
            run(['cmake', '--install', source/'build'])
            run([sys.executable, 'setup.py', 'build_ext', '--inplace'], source)
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                (source/name).symlink_to(Path('python/flux/lib')/name)
            if arm in NEW_ARMS:
                includes = [source, source/'src', source/'include', source/'build',
                            source/'3rdparty/cutlass/include', source/'3rdparty/cutlass/tools/util/include']
                run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                     '--expt-extended-lambda', '-gencode=arch=compute_120,code=sm_120',
                     *['-I'+str(path) for path in includes], source/'check_mapping.cu',
                     '-ldl', '-o', source/'check_mapping'])
            paths = [path for path in (source/'python').rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts]
            paths += [source/'libflux_cuda.so', source/'libflux_cuda_ths_op.so', registration]
            if arm in NEW_ARMS:
                paths.append(source/'check_mapping')
            record['files'].update({str(path.relative_to(root)): sha(path) for path in paths})
        record['status'] = 'built'
        write(root/MANIFEST, record)
        return verify(root, True)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(root/MANIFEST, record)
        raise
