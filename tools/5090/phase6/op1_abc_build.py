"""Verified composition of interleaved Flux, unchanged improved Flux, and ABC codec.

All generated files live in a fresh directory. Parent experiments remain
read-only; reusing a Flux build also reuses its exact baseline dependency.
"""
import os
from pathlib import Path
import subprocess

from rtx5090_loader import HERE, load
import op1_a_build as sorted_build
import flux_baseline_build as improved_build
import op1_abc_native as native
from native_exports import verify_native_exports

base = load('build')
p = load('protocol')
read, write, sha, require = p.read, p.write, p.sha, p.require
PROTOCOL = 'rtx5090-op1-abc-native-sm120-v1'
MANIFEST = 'op1-abc-build.json'
ARMS = ('flux-basic-interleaved', 'flux-basic-arrival', 'flux-basic-selective', 'flux-improved')
LIBRARY = 'libop1_abc.so'


def inputs():
    return dict(recipe_sha256=sha(__file__), native=native.identity())


def base_root(root):
    return Path(read(Path(root)/MANIFEST)['baseline_root'])


def flux_root(root):
    return Path(read(Path(root)/MANIFEST)['flux_root'])


def native_root(root, arm):
    require(arm in ARMS, 'Unknown ABC arm: '+str(arm))
    root = Path(root).resolve()
    return (improved_build.native_root(flux_root(root), arm) if arm == 'flux-improved'
            else sorted_build.native_root(root/'interleaved', 'flux-basic-interleaved'))


def libraries(root, arm):
    native = native_root(root, arm)
    return {name: dict(path=str((native/name).resolve()), sha256=sha(native/name))
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}


def codec_library(root):
    path = Path(root).resolve()/'codec'/LIBRARY
    return dict(path=str(path), sha256=sha(path))


def prepare(root, baseline_root, flux_root=None):
    root, parent = Path(root).resolve(), Path(baseline_root).resolve()
    require(not root.exists(), 'Use a fresh ABC build directory')
    baseline = base.verify(parent)
    reused = flux_root is not None
    flux = Path(flux_root).resolve() if reused else root/'flux'
    if reused:
        improved_build.verify(flux, True)
        require(improved_build.base_root(flux) == parent,
                'Reused Flux build must have the same baseline dependency')
    root.mkdir(parents=True)
    sorted_build.prepare(root/'interleaved', parent)
    if not reused:
        improved_build.prepare(flux, parent)
    codec = root/'codec'
    codec.mkdir()
    for name, content in native.source_files().items():
        (codec/name).write_text(content)
    write(root/MANIFEST, dict(protocol=PROTOCOL, status='prepared', actual_arch='sm120',
        arms=list(ARMS), baseline_root=str(parent), baseline_manifest_sha256=sha(parent/'baseline-build.json'),
        baseline_protocol=baseline['protocol'], flux_root=str(flux), reused_flux=reused,
        sorted_manifest_sha256=sha(root/'interleaved'/sorted_build.MANIFEST),
        flux_manifest_sha256=sha(flux/improved_build.MANIFEST), inputs=inputs(),
        files={str(path.relative_to(root)): sha(path) for path in codec.iterdir()},
        gpu_validated=False))
    verify(root)
    return root


def verify(root, built=False):
    root = Path(root).resolve()
    record = read(root/MANIFEST)
    require(record['protocol'] == PROTOCOL and record['actual_arch'] == 'sm120'
            and record['arms'] == list(ARMS) and record['gpu_validated'] is False,
            'Wrong ABC build protocol/architecture/arms')
    require(record['inputs'] == inputs(), 'ABC codec/build inputs changed')
    parent = Path(record['baseline_root'])
    require(parent.is_absolute() and record['baseline_manifest_sha256'] == sha(parent/'baseline-build.json'),
            'ABC parent baseline manifest changed')
    baseline = base.verify(parent, built)
    require(record['baseline_protocol'] == baseline['protocol'], 'Wrong ABC parent protocol')
    require(record['sorted_manifest_sha256'] == sha(root/'interleaved'/sorted_build.MANIFEST)
            and record['flux_manifest_sha256'] == sha(flux_root(root)/improved_build.MANIFEST),
            'ABC dependency manifest changed')
    sorted_build.verify(root/'interleaved', built)
    improved_build.verify(flux_root(root), built or record['reused_flux'])
    require(sorted_build.base_root(root/'interleaved') == parent
            and improved_build.base_root(flux_root(root)) == parent, 'ABC dependency roots disagree')
    expected = {'codec/'+name for name in native.source_files()}
    if record['status'] == 'built':
        expected.add('codec/'+LIBRARY)
    require(set(record['files']) == expected, 'ABC codec manifest is incomplete')
    for name, digest in record['files'].items():
        require((root/name).is_file() and sha(root/name) == digest, 'ABC codec source/binary changed: '+name)
    require(all((root/'codec'/name).read_text() == content
                for name, content in native.source_files().items()),
            'ABC codec is not the registered implementation')
    if built:
        require(record['status'] == 'built' and record['environment'] == baseline['environment'],
                'ABC build incomplete or Python/CUDA ABI differs')
        require(record['codec_library'] == codec_library(root), 'ABC codec library identity changed')
        verify_native_exports(root/'codec'/LIBRARY, ('op1_abc_encode', 'op1_abc_decode',
            'baseline_taco_encode', 'baseline_taco_decode', 'basic_quant_decode_reduce', 'baseline_taco_error'))
    return record


def compile_build(root, cuda_home, jobs=2):
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    record = verify(root)
    require(record['status'] == 'prepared' and type(jobs) is int and jobs > 0,
            'Compile a fresh ABC build with positive jobs')
    parent = base.verify(base_root(root), True)
    environment = base.environment_identity(cuda_home)
    require(environment == parent['environment'], 'ABC must use the baseline Python/CUDA ABI')
    record.update(status='building', environment=environment)
    write(root/MANIFEST, record)
    try:
        sorted_build.compile_build(root/'interleaved', cuda_home, jobs)
        if not record['reused_flux']:
            improved_build.compile_build(flux_root(root), cuda_home, jobs)
        codec = root/'codec'
        command = [str(cuda_home/'bin/nvcc'), '-std=c++17', '-O3', '-shared', '-Xcompiler=-fPIC',
                   '-lineinfo', '-gencode=arch=compute_120,code=sm_120', '--cudart=shared',
                   str(codec/'op1_abc.cu'), '-o', str(codec/LIBRARY)]
        host = os.environ.get('CUDAHOSTCXX') or os.environ.get('CXX')
        if host:
            command += ['-ccbin', host]
        record['codec_command'] = command
        with (codec/'compile.log').open('x') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
        record.update(status='built', codec_library=codec_library(root),
            sorted_manifest_sha256=sha(root/'interleaved'/sorted_build.MANIFEST),
            flux_manifest_sha256=sha(flux_root(root)/improved_build.MANIFEST))
        record['files']['codec/'+LIBRARY] = sha(codec/LIBRARY)
        write(root/MANIFEST, record)
        return verify(root, True)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(root/MANIFEST, record)
        raise
