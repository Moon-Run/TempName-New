"""Small isolated MA9/MA11 libraries; reuse the existing ABC GEMM/codec build."""
import os
from pathlib import Path
import subprocess

from rtx5090_loader import HERE, H100, load
from native_exports import verify_native_exports

p = load('protocol')
compat = load('ma11_compat')
require, read, write, sha = p.require, p.read, p.write, p.sha
PROTOCOL = 'rtx5090-op2-ma9-standalone-sm120-v1'
HEADER = H100/'ma9_zero_count.cuh'
LIBRARY = 'libop2_ma9_zero_count.so'
EXPORTS = ('ma9_zero_count_protocol', 'ma9_zero_count_chunk_size', 'ma9_zero_count_cuda')


def verify(record, root, torch_version):
    root = Path(root).resolve()
    require(record['protocol'] == PROTOCOL and record['torch'] == torch_version == compat.TORCH
            and record['architecture'] == 'sm_120' and record['gpu_validated'] is False,
            'Wrong OP2 MA9 library environment')
    directory = root/'ma9-compat'
    require(record['source_sha256'] == sha(HEADER) == sha(directory/HEADER.name)
            and record['builder_sha256'] == sha(__file__)
            and (directory/'ma9.cu').read_text() == '#include "ma9_zero_count.cuh"\n',
            'OP2 MA9 source or recipe changed')
    library = directory/LIBRARY
    require(record['library'] == str(library) and record['library_sha256'] == sha(library)
            and record['compile_log_sha256'] == sha(directory/'compile.log')
            and read(directory/'manifest.json') == record, 'OP2 MA9 compile evidence changed')
    verify_native_exports(library, EXPORTS)
    return record


def prepare(root, cuda_home, torch_version):
    require(torch_version == compat.TORCH, 'OP2 requires the verified NGC 25.06 PyTorch')
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    directory = root/'ma9-compat'
    directory.mkdir(parents=True, exist_ok=False)
    (directory/HEADER.name).write_bytes(HEADER.read_bytes())
    (directory/'ma9.cu').write_text('#include "ma9_zero_count.cuh"\n')
    nvcc, library = cuda_home/'bin/nvcc', directory/LIBRARY
    command = [str(nvcc), '--shared', '--cudart=shared', '-O3', '-std=c++17',
               '-Xcompiler=-fPIC', '-arch=sm_120', str(directory/'ma9.cu'), '-o', str(library)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=600, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=False)
    (directory/'compile.log').write_text('COMMAND '+repr(command)+'\n'+result.stdout)
    require(result.returncode == 0 and library.is_file(), 'OP2 MA9 compile failed; see '+str(directory/'compile.log'))
    record = dict(protocol=PROTOCOL, torch=torch_version, architecture='sm_120', gpu_validated=False,
        source_sha256=sha(HEADER), builder_sha256=sha(__file__), nvcc_sha256=sha(nvcc),
        library=str(library), library_sha256=sha(library), compile_log_sha256=sha(directory/'compile.log'))
    write(directory/'manifest.json', record)
    return verify(record, root, torch_version)


def libraries(plan):
    return {name: dict(path=plan[key]['library'], sha256=plan[key]['library_sha256'])
            for name, key in (('ma9', 'ma9_compat'), ('ma11', 'ma11_compat'))}


def verify_all(plan, root):
    verify(plan['ma9_compat'], root, plan['environment']['torch'])
    compat.verify(plan['ma11_compat'], root, plan['environment']['torch'])
