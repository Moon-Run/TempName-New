"""Freeze all three existing CUDA variants with the validated IPC wrapper."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs/a800/gemm_rs_validation/source'
FIXED = ROOT / 'outputs/a800/phase3/ipc-fix-build'
OLD = ROOT / 'outputs/a800/phase3/build'
OUT = ROOT / 'outputs/a800/phase3/three-way-build'
CONFIG = ROOT / 'tools/a800/phase3/three_way.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parent = json.loads((FIXED / 'manifest.json').read_text())
old = json.loads((OLD / 'manifest.json').read_text())
for name, expected in parent['files'].items():
    assert sha(FIXED / name) == expected, name
interleaved = OLD / 'interleaved'
assert sha(interleaved / 'libflux_cuda.so') == old['interleaved']['sha256']
assert sha(interleaved / 'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp') == old['interleaved']['swizzle_sha256']
OUT.mkdir(exist_ok=False)
shutil.copy2(FIXED / 'ipc-fix.patch', OUT / 'ipc-fix.patch')
manifest = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                sampling=False, config_sha256=sha(CONFIG), parent_build=str(FIXED),
                parent_manifest_sha256=sha(FIXED / 'manifest.json'),
                original_variants_manifest_sha256=sha(OLD / 'manifest.json'),
                mapping_source_sha256=sha(ROOT / 'tools/a800/phase3/check_mapping_extension.cu'),
                policies={}, files={}, commands=[])
for policy in ('original', 'remote_first', 'interleaved'):
    dest = OUT / policy
    shutil.copytree(FIXED / ('original' if policy == 'interleaved' else policy), dest,
                    symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    if policy == 'interleaved':
        shutil.copy2(interleaved / 'libflux_cuda.so', dest / 'libflux_cuda.so')
        shutil.copy2(interleaved / 'swizzle.patch', dest / 'swizzle.patch')
        shutil.copytree(interleaved / 'overlay', dest / 'overlay', dirs_exist_ok=True)
        metadata = dict(old[policy], variant=policy)
    else:
        metadata = parent['policies'][policy]
    assert sha(dest / 'libflux_cuda.so') == metadata['sha256']
    assert sha(dest / 'libflux_cuda_ths_op.so') == sha(FIXED / 'original/libflux_cuda_ths_op.so')
    for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
        link = dest / 'python/flux/lib' / name
        assert link.is_symlink()
        link.unlink()
        link.symlink_to(dest / name)
    includes = [dest / 'overlay', BASE / 'src', BASE / 'include', BASE / 'build',
                BASE / '3rdparty/cutlass/include', BASE / '3rdparty/cutlass/tools/util/include']
    cmd = ['/data/apps/cuda/12.8/bin/nvcc', '-std=c++17', '-O3', '-DNDEBUG',
           '-gencode=arch=compute_80,code=[sm_80,compute_80]',
           *['-I'+str(p) for p in includes], str(ROOT / 'tools/a800/phase3/check_mapping_extension.cu'),
           '-o', str(dest / 'check_mapping')]
    manifest['commands'].append(cmd)
    subprocess.run(cmd, check=True)
    manifest['policies'][policy] = dict(metadata, library=str(dest / 'libflux_cuda.so'))
for path in OUT.rglob('*'):
    if path.is_file() and not path.is_symlink() and path.name != 'manifest.json':
        manifest['files'][str(path.relative_to(OUT))] = sha(path)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('THREE-WAY BUILD VERIFIED', OUT)
