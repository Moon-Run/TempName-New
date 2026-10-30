"""Pin upstream kernels and build a standalone CUDA C ABI library; no GPU use."""
from pathlib import Path
import os
import shutil
import subprocess

from common import HERE, read, require, sha, verify_files, write

PROTOCOL = 'taco-upstream-standalone-cuda-v1'
UPSTREAM = HERE/'baseline_upstream'
UPSTREAM_COMMIT = 'ca717e05b6dd6d21fff4d2617d209c95be63f3ef'
UPSTREAM_SHA = '15a9fa3b41f59333c03170a886d1c5d596415cebbc3642447cfcfdd93dffad8e'
LIBRARY = 'libbaseline_taco.so'


def kernel_source():
    record = read(UPSTREAM/'provenance.json')
    require(record['commit'] == UPSTREAM_COMMIT and sha(UPSTREAM/'taco.cu') == UPSTREAM_SHA,
            'Pinned upstream TACO changed')
    verify_files(UPSTREAM, record['files'])
    text = (UPSTREAM/'taco.cu').read_text()
    return text[text.index('__device__ __forceinline__ float safe_divide('):
                text.index('template<int GROUP_SIZE>\nvoid launch_compress_template(')]


def source_identity():
    kernel_source()
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(UPSTREAM.iterdir())} | {
        'baseline_codec.cu': sha(HERE/'baseline_codec.cu')}


def prepare(out, source_only=False):
    root = Path(out).resolve(); root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(UPSTREAM, root/'upstream')
    shutil.copy2(HERE/'baseline_codec.cu', root/'baseline_codec.cu')
    (root/'upstream_kernels.cuh').write_text(kernel_source())
    write(root/'manifest.json', dict(protocol=PROTOCOL, status='prepared', arch='sm90',
        source_only=source_only, upstream_commit=UPSTREAM_COMMIT, source_identity=source_identity(),
        files={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()},
        library=None, gpu_validated=False))
    return root


def validate_snapshot(m, built=False):
    require(m['protocol'] == PROTOCOL and m['arch'] == 'sm90' and
            m['upstream_commit'] == UPSTREAM_COMMIT and m['source_identity'] == source_identity(),
            'Standalone CUDA codec provenance changed')
    if built:
        require(m['status'] == 'built' and not m['source_only'] and
                set(m['library']) == {'path', 'sha256'}, 'CUDA codec has not been built')
    return m


def verify(root, built=False):
    root = Path(root); m = validate_snapshot(read(root/'manifest.json'), built)
    verify_files(root, m['files'])
    require((root/'upstream_kernels.cuh').read_text() == kernel_source(), 'Upstream kernel body modified')
    if built:
        require(Path(m['library']['path']) == root.resolve()/LIBRARY and
                sha(root/LIBRARY) == m['library']['sha256'], 'Codec binary changed')
    return m


def compile_build(root, cuda_home):
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    m = verify(root)
    require(not m['source_only'] and m['status'] == 'prepared', 'Compile requires a fresh executable build')
    nvcc = cuda_home/'bin/nvcc'
    require(nvcc.is_file(), 'CUDA Toolkit missing')
    cmd = [str(nvcc), '-std=c++17', '-O3', '-shared', '-Xcompiler=-fPIC', '-lineinfo',
           '-gencode=arch=compute_90,code=sm_90', '--cudart=shared',
           str(root/'baseline_codec.cu'), '-o', str(root/LIBRARY)]
    host = os.environ.get('CUDAHOSTCXX') or os.environ.get('CXX')
    if host: cmd += ['-ccbin', host]
    m.update(status='building', command=cmd, cuda_home=str(cuda_home))
    write(root/'manifest.json', m)
    try:
        with (root/'compile.log').open('x') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        m.update(status='built', library=dict(path=str(root/LIBRARY), sha256=sha(root/LIBRARY)),
                 compiler_sha256=sha(nvcc))
        m['files'][LIBRARY] = sha(root/LIBRARY)
        write(root/'manifest.json', m)
        return verify(root, built=True)
    except BaseException:
        m['status'] = 'failed'; write(root/'manifest.json', m); raise
