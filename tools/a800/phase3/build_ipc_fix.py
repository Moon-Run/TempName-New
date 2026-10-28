"""Build the IPC wrapper fix without changing any frozen CUDA experiment library."""
import difflib
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs/a800/gemm_rs_validation/source'
OLD = ROOT / 'outputs/a800/phase3/extension-build'
OUT = ROOT / 'outputs/a800/phase3/ipc-fix-build'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

previous = json.loads((OLD / 'manifest.json').read_text())
for name, expected in previous['files'].items():
    assert sha(OLD / name) == expected, name
OUT.mkdir(exist_ok=False)
source = OUT / 'flux_shm.cc'
shutil.copy2(ROOT / 'src/ths_op/flux_shm.cc', source)
original = BASE / 'src/ths_op/flux_shm.cc'
(OUT / 'ipc-fix.patch').write_text(''.join(difflib.unified_diff(
    original.read_text().splitlines(True), source.read_text().splitlines(True),
    fromfile='a/src/ths_op/flux_shm.cc', tofile='b/src/ths_op/flux_shm.cc')))
commands = []

def run(args, cwd):
    commands.append(dict(args=list(map(str, args)), cwd=str(cwd)))
    print(shlex.join(list(map(str, args))), flush=True)
    subprocess.run(args, cwd=cwd, check=True)

entries = json.loads((BASE / 'build/compile_commands.json').read_text())
entry = next(e for e in entries if e['file'] == str(original))
compile_args = shlex.split(entry['command'])
compile_args[compile_args.index('-c') + 1] = str(source)
obj = OUT / 'flux_shm.cc.o'
compile_args[compile_args.index('-o') + 1] = str(obj)
run(compile_args, entry['directory'])
work = BASE / 'build/src/ths_op'
link = shlex.split((work / 'CMakeFiles/flux_cuda_ths_op.dir/link.txt').read_text())
link[link.index('-o') + 1] = str(OUT / 'libflux_cuda_ths_op.so')
old_object = 'CMakeFiles/flux_cuda_ths_op.dir/flux_shm.cc.o'
assert link.count(old_object) == 1
link[link.index(old_object)] = str(obj)
run(link, work)
manifest = dict(previous, parent_build=str(OLD), parent_manifest_sha256=sha(OLD / 'manifest.json'),
                ipc_fix_source_sha256=sha(source), commands=commands, policies={}, files={})
for policy, metadata in previous['policies'].items():
    dest = OUT / policy
    shutil.copytree(OLD / policy, dest, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(OUT / 'libflux_cuda_ths_op.so', dest / 'libflux_cuda_ths_op.so')
    for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
        symlink = dest / 'python/flux/lib' / name
        assert symlink.is_symlink()
        symlink.unlink()
        symlink.symlink_to(dest / name)
    assert sha(dest / 'libflux_cuda.so') == metadata['sha256']
    manifest['policies'][policy] = dict(metadata, library=str(dest / 'libflux_cuda.so'))
for path in OUT.rglob('*'):
    if path.is_file() and not path.is_symlink() and path.name != 'manifest.json':
        manifest['files'][str(path.relative_to(OUT))] = sha(path)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('IPC FIX BUILD VERIFIED', OUT, flush=True)
