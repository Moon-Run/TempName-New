"""Build and verify the small PyTorch-2.8-specific MA11 CUDA library."""
from pathlib import Path
import json
import os
import subprocess

from rtx5090_loader import HERE, load

p = load('protocol')
require, sha = p.require, p.sha
PROTOCOL = 'rtx5090-ma11-pytorch-280-softmax-v1'
TORCH = '2.8.0a0+5228986c39.nv25.06'
SOURCE = HERE/'ma11_softmax_5090.cu'
LIBRARY = 'libma11_softmax_5090.so'
EXPORTS = ('ma11_softmax_protocol', 'ma11_softmax_forward_cuda', 'ma11_softmax_backward_cuda')


def verify(record, root, torch_version):
    root = Path(root).resolve()
    require(record.get('protocol') == PROTOCOL and torch_version == record.get('torch') == TORCH,
            'Wrong MA11 PyTorch softmax compatibility protocol/environment')
    require(record.get('source_sha256') == sha(SOURCE) and record.get('builder_sha256') == sha(__file__),
            'MA11 compatibility source or builder changed')
    require(record.get('architecture') == 'sm_120' and record.get('gpu_validated') is False,
            'Invalid MA11 compilation identity')
    library = (root/'ma11-compat'/LIBRARY).resolve()
    require(record.get('library') == str(library) and library.is_file()
            and record.get('library_sha256') == sha(library), 'MA11 compatibility library changed')
    manifest, log = root/'ma11-compat/manifest.json', root/'ma11-compat/compile.log'
    require(manifest.is_file() and json.loads(manifest.read_text()) == record and log.is_file()
            and record.get('compile_log_sha256') == sha(log), 'MA11 compile evidence changed')
    return record


def prepare(root, cuda_home, torch_version):
    require(torch_version == TORCH, 'MA11 compatibility library requires the frozen NGC 25.06 PyTorch')
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    nvcc = cuda_home/'bin/nvcc'
    require(nvcc.is_file(), 'CUDA Toolkit nvcc missing: '+str(nvcc))
    directory = root/'ma11-compat'
    directory.mkdir(parents=True, exist_ok=False)
    library = directory/LIBRARY
    command = [str(nvcc), '--shared', '--cudart=shared', '-O3', '-std=c++17', '-Xcompiler=-fPIC', '-arch=sm_120',
               str(SOURCE), '-o', str(library)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=600, check=False)
    (directory/'compile.log').write_text('COMMAND '+repr(command)+'\n'+result.stdout)
    require(result.returncode == 0 and library.is_file(),
            'MA11 compatibility compilation failed; see '+str(directory/'compile.log'))
    output = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    symbols = {line.split()[-1] for line in output.splitlines() if line.split()}
    require(set(EXPORTS) <= symbols, 'MA11 compatibility library is missing native exports')
    record = dict(protocol=PROTOCOL, torch=torch_version, source_sha256=sha(SOURCE),
                  builder_sha256=sha(__file__), nvcc_sha256=sha(nvcc),
                  library=str(library), library_sha256=sha(library),
                  compile_log_sha256=sha(directory/'compile.log'),
                  architecture='sm_120', gpu_validated=False)
    (directory/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    return verify(record, root, torch_version)
