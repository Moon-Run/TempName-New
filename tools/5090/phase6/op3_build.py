"""Compile only the two OP3 codec paths; keep the verified ABC GEMM read-only."""
from pathlib import Path
import os
import subprocess

from rtx5090_loader import load
import op1_abc_build as parent
import op3_native as native
from native_exports import verify_native_exports

p = load('protocol')
read, write, require, sha = p.read, p.write, p.require, p.sha
PROTOCOL = 'rtx5090-op3-codec-sm120-v1'
MANIFEST = 'op3-build.json'
LIBRARY = 'libop3_codec.so'
ARMS = ('op3-r', 'op3-improved')


def abc_root(root):
    return Path(read(Path(root)/MANIFEST)['abc_root'])


def base_root(root):
    return parent.base_root(abc_root(root))


def native_root(root, arm):
    require(arm in ARMS, 'Unknown OP3 arm')
    return parent.native_root(abc_root(root), 'flux-basic-interleaved')


def libraries(root, arm):
    require(arm in ARMS, 'Unknown OP3 arm')
    return parent.libraries(abc_root(root), 'flux-basic-interleaved')


def codec_library(root):
    library = Path(root).resolve()/'codec'/LIBRARY
    return dict(path=str(library), sha256=sha(library))


def inputs():
    return dict(recipe_sha256=sha(__file__), native=native.identity())


def prepare(root, abc_build):
    root, abc = Path(root).resolve(), Path(abc_build).resolve()
    require(not root.exists(), 'Use a fresh OP3 build directory')
    old = parent.verify(abc)
    (root/'codec').mkdir(parents=True)
    files = native.source_files()
    for name, content in files.items():
        (root/'codec'/name).write_text(content)
    write(root/MANIFEST, dict(protocol=PROTOCOL, status='prepared', actual_arch='sm120',
        arms=list(ARMS), abc_root=str(abc), abc_manifest_sha256=sha(abc/parent.MANIFEST),
        abc_protocol=old['protocol'], inputs=inputs(), gpu_validated=False,
        files={'codec/'+name: sha(root/'codec'/name) for name in files}))
    verify(root)
    return root


def verify(root, built=False):
    root = Path(root).resolve()
    record = read(root/MANIFEST)
    require(record['protocol'] == PROTOCOL and record['actual_arch'] == 'sm120'
            and record['arms'] == list(ARMS) and record['gpu_validated'] is False,
            'Wrong OP3 codec architecture/protocol/arms')
    require(record['inputs'] == inputs(), 'OP3 source recipe changed')
    abc = abc_root(root)
    require(abc.is_absolute() and record['abc_manifest_sha256'] == sha(abc/parent.MANIFEST),
            'OP3 parent ABC manifest changed')
    old = parent.verify(abc, built)
    require(record['abc_protocol'] == old['protocol'], 'OP3 parent protocol changed')
    expected = {'codec/'+name for name in native.source_files()}
    if record['status'] == 'built':
        expected.update({'codec/'+LIBRARY, 'codec/compiled.sass'})
    require(set(record['files']) == expected, 'OP3 codec file registry incomplete')
    for name, digest in record['files'].items():
        require((root/name).is_file() and sha(root/name) == digest, 'OP3 codec source/binary changed: '+name)
    require(all((root/'codec'/name).read_text() == content for name, content in native.source_files().items()),
            'Unregistered OP3 native source')
    if built:
        require(record['status'] == 'built' and record['environment'] == old['environment']
                and record['codec_library'] == codec_library(root), 'OP3 build incomplete or ABI differs')
        verify_native_exports(root/'codec'/LIBRARY, native.EXPORTS)
    return record


def compile_build(root, cuda_home, jobs=2):
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    record = verify(root)
    require(record['status'] == 'prepared' and type(jobs) is int and jobs > 0, 'Compile a fresh OP3 build')
    old = parent.verify(abc_root(root), True)
    environment = load('build').environment_identity(cuda_home)
    require(environment == old['environment'], 'Use the same Python/CUDA ABI as the ABC build')
    directory = root/'codec'
    command = [str(cuda_home/'bin/nvcc'), '-std=c++17', '-O3', '--shared', '-Xcompiler=-fPIC',
        '-lineinfo', '-gencode=arch=compute_120,code=sm_120', '--cudart=shared',
        str(directory/'op3.cu'), '-o', str(directory/LIBRARY)]
    host = os.environ.get('CUDAHOSTCXX') or os.environ.get('CXX')
    if host:
        command += ['-ccbin', host]
    record.update(status='building', environment=environment, command=command)
    write(root/MANIFEST, record)
    try:
        with (directory/'compile.log').open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        with (directory/'compiled.sass').open('x') as output:
            subprocess.run([str(cuda_home/'bin/cuobjdump'), '--dump-sass', str(directory/LIBRARY)],
                           check=True, stdout=output, stderr=subprocess.STDOUT)
        record.update(status='built', codec_library=codec_library(root))
        for name in (LIBRARY, 'compiled.sass'):
            record['files']['codec/'+name] = sha(directory/name)
        write(root/MANIFEST, record)
        return verify(root, True)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(root/MANIFEST, record)
        raise
