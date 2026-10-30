#!/usr/bin/env python3
"""Configure an inherited-package venv inside NVIDIA PyTorch 25.06, without training."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse only checked source operations and command logging. No H100 package,
# Python, CUDA, Apex installation or H100 GPU probe is invoked.
import setup_h100 as sources

PROTOCOL = 'rtx5090-ngc-25.06-environment-v1'
EXPECTED_TORCH = '2.8.0a0+5228986c39.nv25.06'
MEGATRON_COMMIT = sources.recovery.MEGATRON_COMMIT
OUTPUT = REPO/'outputs/5090/env-setup'
EXTRAS = {
    'packaging': 'packaging', 'pybind11': 'pybind11', 'numpy': 'numpy',
    'einops': 'einops', 'regex': 'regex', 'psutil': 'psutil', 'yaml': 'PyYAML',
    'tqdm': 'tqdm', 'six': 'six', 'setuptools': 'setuptools', 'wheel': 'wheel',
}
SYSTEM_PACKAGES = ('build-essential', 'git', 'cmake', 'ninja-build', 'binutils',
                   'pkg-config', 'libnuma-dev', 'libibverbs1', 'ibverbs-providers',
                   'libibverbs-dev', 'python3.12-dev', 'python3.12-venv')
write = sources.write
require = sources.require
LiveRunner = sources.LiveRunner


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python', type=Path,
                        default=Path(getattr(sys, '_base_executable', sys.executable)),
                        help='Image Python, not a freshly installed interpreter')
    parser.add_argument('--prefix', type=Path, default=REPO.parent/'venvs/tempname-5090')
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--state-dir', type=Path, default=REPO.parent/'.5090-env')
    parser.add_argument('--cuda-home', type=Path, default=Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')))
    parser.add_argument('--skip-gpu-check', action='store_true', help='Configure software only; explicitly record GPU checks as pending')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without subprocesses, installation or GPU queries')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().absolute() if name == 'base_python' else value.expanduser().resolve())
    for path in (args.prefix, args.state_dir, args.megatron):
        require(path != REPO and not path.is_relative_to(REPO), 'Keep environment/cache/Megatron outside TempName-New')
    require(not args.prefix.is_relative_to(args.megatron) and not args.megatron.is_relative_to(args.prefix),
            'Megatron and the venv must be separate directories')
    require(not args.state_dir.is_relative_to(args.prefix) and not args.prefix.is_relative_to(args.state_dir),
            'State directory and venv must be separate directories')
    require(not args.state_dir.is_relative_to(args.megatron) and not args.megatron.is_relative_to(args.state_dir),
            'State directory and Megatron must be separate directories')
    return args


def plan(args):
    return dict(protocol=PROTOCOL, image='nvcr.io/nvidia/pytorch:25.06-py3',
                base_python=str(args.base_python), python=str(args.prefix/'bin/python'),
                inherited_system_packages=True, expected_python='3.12.x', expected_torch=EXPECTED_TORCH,
                expected_cuda='12.9', cuda_home=str(args.cuda_home),
                preserve_image_torch_cuda_apex=True, install_torch=False, install_cuda=False,
                install_apex=False, megatron=str(args.megatron), megatron_commit=MEGATRON_COMMIT,
                existing_repositories_required=True, missing_only_python_packages=list(EXTRAS.values()),
                state_dir=str(args.state_dir), gpu_check=not args.skip_gpu_check,
                gpu_check_kind='8xRTX5090 SM120, Apex kernels, NCCL collectives; no P2P requirement',
                generated_environment=str(OUTPUT/'env.sh'), training_started=False)


def clean_environment(python, cuda_home=None):
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_DEFAULT_TIMEOUT='120')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(key, None)
    paths = [str(Path(python).parent)]
    if cuda_home:
        paths.append(str(Path(cuda_home)/'bin'))
        env.update(CUDA_HOME=str(cuda_home), CUDACXX=str(Path(cuda_home)/'bin/nvcc'))
        env['LD_LIBRARY_PATH'] = str(Path(cuda_home)/'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = ':'.join(paths) + ':' + env.get('PATH', '/usr/bin:/bin')
    return env


def capture(command, env=None, timeout=120):
    result = subprocess.run(list(map(str, command)), env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    require(result.returncode == 0, 'Command failed: '+shlex.join(list(map(str, command)))+'\n'+result.stderr)
    return result.stdout


STACK_PROBE = r'''
import hashlib, importlib, importlib.metadata as metadata, json, sys, sysconfig
import torch
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
protected = {}
for name in ('torch', 'torch._C', 'apex', 'apex_C', 'amp_C', 'fused_layer_norm_cuda'):
    module = importlib.import_module(name)
    filename = module.__file__
    protected[name] = dict(path=filename, sha256=sha(filename))
versions = {}
for dist in ('torch', 'apex', 'numpy', 'transformer-engine', 'pip', 'setuptools',
             'wheel', 'packaging', 'pybind11', 'einops', 'regex', 'psutil', 'PyYAML', 'tqdm', 'six'):
    try: versions[dist] = metadata.version(dist)
    except metadata.PackageNotFoundError: versions[dist] = None
print('RTX5090_JSON:'+json.dumps(dict(python=list(sys.version_info[:3]), executable=sys.executable,
    prefix=sys.prefix, base_prefix=sys.base_prefix, include=sysconfig.get_path('include'),
    torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
    nccl=list(torch.cuda.nccl.version()), cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    arch_flags=torch._C._cuda_getArchFlags(), protected_modules=protected, versions=versions)))
'''


def json_probe(python, code, env):
    output = capture([python, '-B', '-c', code], env)
    lines = [line[len('RTX5090_JSON:'):] for line in output.splitlines() if line.startswith('RTX5090_JSON:')]
    require(len(lines) == 1, 'Python probe did not return one environment record')
    return json.loads(lines[0])


def inventory(python, cuda_home):
    env = dict(clean_environment(python, cuda_home), CUDA_VISIBLE_DEVICES='')
    return json_probe(python, STACK_PROBE, env)


def validate_stack(value):
    require(value['python'][:2] == [3, 12] and value['torch'] == EXPECTED_TORCH
            and value['cuda'] == '12.9',
            'Use the existing NGC 25.06 Python 3.12 / PyTorch '+EXPECTED_TORCH+' / CUDA 12.9 stack; '
            'this script will not replace Python, PyTorch, CUDA or Apex')
    require('sm_120' in value['arch_flags'].split(), 'The image PyTorch build lacks SM120 kernels')
    require(set(value['protected_modules']) == {'torch', 'torch._C', 'apex', 'apex_C', 'amp_C', 'fused_layer_norm_cuda'},
            'The image Apex extensions must be available before setup')


def core_identity(value):
    return {name: value[name] for name in ('python', 'base_prefix', 'torch', 'cuda', 'cudnn', 'nccl',
                                          'cxx11_abi', 'arch_flags', 'protected_modules')}


def validate_toolkit(root):
    nvcc = Path(root)/'bin/nvcc'
    require(nvcc.is_file(), 'Image nvcc missing: '+str(nvcc))
    text = capture([nvcc, '--version'])
    require(re.search(r'\brelease 12\.9[,\s]', text) is not None, 'NGC 25.06 setup expects its existing CUDA 12.9 Toolkit')
    require('sm_120' in capture([nvcc, '--list-gpu-code']).split(), 'nvcc does not support SM120')
    require((root/'include/cuda_runtime.h').is_file() and (root/'lib64/libcudart.so').is_file(),
            'Incomplete CUDA Toolkit headers/runtime')
    return dict(root=str(root), nvcc=text.strip(), nvcc_sha256=sha(nvcc))


def check_host(args):
    release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    require(release.get('ID', '').strip('"') == 'ubuntu' and
            release.get('VERSION_ID', '').strip('"') == '24.04' and platform.machine() == 'x86_64',
            'Run inside the NGC 25.06 Ubuntu 24.04 x86_64 container')
    require(args.base_python.is_file() and shutil.which('git'), 'Image Python and Git must be available')
    require((REPO/'.git').exists() and (args.megatron/'.git').exists(),
            'Use the existing sibling checkouts hjr/TempName-New and hjr/Megatron-LM; '
            'select the actual checkout with --megatron')


def ensure_sources(args, runner):
    # User already cloned both repositories. Never create/replace either one.
    require((args.megatron/'.git').exists(), 'Existing Megatron Git checkout required')
    sources.check_clean_checkout(args.megatron)
    before = capture(sources.recovery.git_command(args.megatron, 'rev-parse', 'HEAD')).strip()
    sources.ensure_megatron(args.megatron, runner)
    sources.ensure_submodules(runner)
    sources.software_sources_check()
    return dict(megatron_before=before, megatron_after=MEGATRON_COMMIT,
                megatron=str(args.megatron), repositories_recloned=False)


def install_system(runner):
    missing = [package for package in SYSTEM_PACKAGES if not sources.recovery.package_installed(package)]
    if not missing:
        return []
    require(os.geteuid() == 0 and shutil.which('apt-get'),
            'Missing system packages require container root: '+', '.join(missing))
    env = dict(os.environ, DEBIAN_FRONTEND='noninteractive', LC_ALL='C')
    command = ['apt-get', '-o', 'Dpkg::Lock::Timeout=120', '-o', 'Acquire::Retries=3']
    runner.run(command+['update'], env)
    simulated = capture(command+['-s', 'install', '--no-install-recommends', *missing], env, timeout=300)
    sources.recovery.validate_apt_plan(simulated)
    require(not any(line.startswith('Inst ') and line.split()[1].startswith(('cuda-', 'nvidia-'))
                    for line in simulated.splitlines()), 'Unexpected CUDA/driver package in APT dependency plan')
    (runner.root/'apt-plan.txt').write_text(simulated)
    runner.run(command+['install', '--no-install-recommends', '-y', *missing], env)
    return missing


def ensure_venv(args, base, runner):
    key = hashlib.sha256(str(args.prefix).encode()).hexdigest()[:16]
    receipt = args.state_dir/('venv-'+key+'.json')
    expected = dict(protocol=PROTOCOL, prefix=str(args.prefix), base_python=str(args.base_python),
                    core=core_identity(base), system_site_packages=True)
    python = args.prefix/'bin/python'
    if args.prefix.exists():
        require(receipt.is_file() and read(receipt) == expected,
                'Existing venv is unmanaged or belongs to a different image; preserved. Select a fresh --prefix')
        require(python.is_file() and (args.prefix/'pyvenv.cfg').is_file(),
                'Interrupted venv is preserved; select a fresh --prefix')
    else:
        write(receipt, expected)
        # Inherit image pip as well: no pip/setuptools bootstrap or upgrades.
        runner.run([args.base_python, '-m', 'venv', '--system-site-packages', '--without-pip', args.prefix],
                   clean_environment(args.base_python, args.cuda_home))
    require('include-system-site-packages = true' in (args.prefix/'pyvenv.cfg').read_text().lower(),
            'The managed venv must inherit the image packages')
    value = inventory(python, args.cuda_home)
    require(Path(value['prefix']).resolve() == args.prefix and core_identity(value) == core_identity(base),
            'Venv shadowed or changed the image PyTorch/CUDA/Apex stack')
    return python


def install_missing_python(python, cuda_home, runner):
    env = dict(clean_environment(python, cuda_home), CUDA_VISIBLE_DEVICES='')
    code = ('import importlib.util,json\nmodules='+repr(list(EXTRAS))+'\n'
            'print("RTX5090_JSON:"+json.dumps({name:importlib.util.find_spec(name) is not None for name in modules}))')
    found = json_probe(python, code, env)
    wanted = [package for name, package in EXTRAS.items() if not found[name]]
    if wanted:
        command = [python, '-m', 'pip', 'install', '--no-deps', '--disable-pip-version-check']
        constraints = Path('/etc/pip/constraint.txt')
        if constraints.is_file():
            command += ['--constraint', constraints]
        runner.run(command+wanted, env)
    # Import failures are not treated as permission to upgrade image packages.
    code = ('import importlib\nfor name in '+repr(list(EXTRAS))+': importlib.import_module(name)\n'
            'from apex.optimizers import FusedAdam\nfrom apex.normalization import FusedLayerNorm\n'
            'print("Python dependencies and inherited Apex: OK")')
    runner.run([python, '-c', code], env)
    return wanted


def compile_probe(python, args, runner):
    env = dict(clean_environment(python, args.cuda_home), CUDA_VISIBLE_DEVICES='')
    temp = runner.root/'compiler-tmp'
    temp.mkdir()
    env['TMPDIR'] = str(temp)
    source = runner.root/'sm120-probe.cu'
    source.write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n'
                      '__global__ void probe(__nv_bfloat16* x) { x[threadIdx.x] = __float2bfloat16(1.f); }\n')
    output = runner.root/'sm120-probe.o'
    runner.run([args.cuda_home/'bin/nvcc', '-std=c++17', '-arch=sm_120', '-c', source, '-o', output], env)
    require(output.is_file() and output.stat().st_size > 0, 'SM120 compilation produced no object')
    return dict(object=str(output), sha256=sha(output), gpu_used=False)


def verify_megatron(python, args, runner):
    env = dict(clean_environment(python, args.cuda_home), CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(args.megatron))
    code = '''import ctypes, pathlib, torch
import apex_C, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from pretrain_gpt import model_provider
import megatron.core
expected = pathlib.Path(EXPECTED_MEGATRON).resolve()
assert pathlib.Path(megatron.core.__file__).resolve().is_relative_to(expected), 'Wrong Megatron import location'
ctypes.CDLL('libibverbs.so.1')
print('Pinned Megatron training imports and inherited Apex: OK; no training started')
'''.replace('EXPECTED_MEGATRON', repr(str(args.megatron)))
    runner.run([python, '-c', code], env)
    # Compile a fresh helper outside the user's checkout; an old Python 3.10
    # helper left in Megatron cannot stand in for this Python 3.12 check.
    helper = runner.root/'dataset-helper'
    helper.mkdir()
    for name in ('Makefile', 'helpers.cpp'):
        shutil.copy2(args.megatron/'megatron/core/datasets'/name, helper/name)
    runner.run(['make', '-C', helper], env)
    outputs = list(helper.glob('helpers_cpp*.so'))
    require(len(outputs) == 1, 'Dataset helper did not compile for the selected Python')
    runner.run([python, '-c', 'import helpers_cpp; print("Megatron dataset helper: OK")'],
               dict(env, PYTHONPATH=str(helper)))
    return dict(path=str(outputs[0]), sha256=sha(outputs[0]), source_checkout_modified=False)


def verify_gpu(python, args, runner):
    out = runner.root/'gpu-preflight'
    env = clean_environment(python, args.cuda_home)
    runner.run([python, REPO/'tools/5090/phase6/baseline.py', '--execute', '--preflight-only',
                '--python', python, '--cuda-home', args.cuda_home, '--build-cuda-home', args.cuda_home,
                '--megatron', args.megatron, '--out', out], env)
    report = read(out/'environment-preflight.json')
    require(read(out/'batch-state.json')['status'] == 'preflight_completed'
            and report['completed'] is True and report['collective_probe_passed'] is True,
            'The eight-rank CUDA/Apex/NCCL preflight did not complete')
    return dict(report=str(out/'environment-preflight.json'), sha256=sha(out/'environment-preflight.json'),
                peer_access=report['peer_access'], collective_probe_passed=True)


def environment_text(args, python):
    values = dict(RTX5090_SETUP_PYTHON=str(python), RTX5090_SETUP_CUDA_HOME=str(args.cuda_home),
                  RTX5090_SETUP_MEGATRON=str(args.megatron), RTX5090_SETUP_PREFIX=str(args.prefix))
    lines = ['# Generated by env/setup_5090.sh after successful validation.']
    lines += ['export '+name+'='+shlex.quote(value) for name, value in values.items()]
    # Per-process exact checkout trust for root containers and mounted user files.
    lines += ['export GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-0}"']
    for path in (REPO, REPO/'3rdparty/cutlass', REPO/'3rdparty/nccl', args.megatron):
        lines += ['export "GIT_CONFIG_KEY_${GIT_CONFIG_COUNT}=safe.directory"',
                  'export "GIT_CONFIG_VALUE_${GIT_CONFIG_COUNT}="'+shlex.quote(str(path)),
                  'GIT_CONFIG_COUNT=$((GIT_CONFIG_COUNT + 1))']
    return '\n'.join(lines)+'\n'


def execute(args):
    check_host(args)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir/'setup.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another 5090 setup is using '+str(args.state_dir)) from None
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    work = OUTPUT/'runs'/stamp
    work.mkdir(parents=True)
    runner = LiveRunner(work)
    result = dict(plan(args), status='running', software_ready=False, gpu_checked=False, commands=runner.commands)

    def stage(name):
        result['stage'] = name
        write(work/'report.json', result)
        print('STAGE:', name, flush=True)

    try:
        print('Setup log:', work/'install.log', flush=True)
        stage('validate-existing-image')
        base = inventory(args.base_python, args.cuda_home)
        validate_stack(base)
        result['image_before'] = base
        result['toolkit'] = validate_toolkit(args.cuda_home)
        sources.add_git_trust([REPO, REPO/'3rdparty/cutlass', REPO/'3rdparty/nccl', args.megatron])
        stage('existing-source-checkouts')
        result['sources'] = ensure_sources(args, runner)
        stage('missing-system-tools')
        result['installed_system_packages'] = install_system(runner)
        # APT must not have replaced the selected image runtime either.
        require(core_identity(inventory(args.base_python, args.cuda_home)) == core_identity(base),
                'Image core runtime changed during system dependency installation')
        stage('inherited-package-venv')
        python = ensure_venv(args, base, runner)
        stage('missing-python-dependencies')
        result['installed_python_packages'] = install_missing_python(python, args.cuda_home, runner)
        after = inventory(python, args.cuda_home)
        require(core_identity(after) == core_identity(base), 'Image PyTorch/CUDA/Apex changed or was shadowed')
        result['environment'] = after
        stage('sm120-compile-check')
        result['compile_probe'] = compile_probe(python, args, runner)
        stage('pinned-megatron-imports-and-helper')
        result['dataset_helper'] = verify_megatron(python, args, runner)
        if not args.skip_gpu_check:
            stage('eight-rank-cuda-apex-nccl-preflight')
            result['gpu_preflight'] = verify_gpu(python, args, runner)
            result['gpu_checked'] = True
        stage('save-environment')
        target = OUTPUT/'env.sh'
        temporary = target.with_suffix('.sh.tmp')
        temporary.write_text(environment_text(args, python))
        temporary.replace(target)
        result.update(status='completed', stage='done', software_ready=True,
                      gpu_checks_pending=args.skip_gpu_check, image_stack_preserved=True)
        print('Environment ready. source '+shlex.quote(str(REPO/'env/activate_5090.sh')), flush=True)
        print('run_baseline.sh automatically reads the configured paths. Training was not started.', flush=True)
        if args.skip_gpu_check:
            print('GPU/Apex-kernel/NCCL checks were skipped; they remain mandatory before training.', flush=True)
    except BaseException as error:
        result.update(status='failed', software_ready=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(work/'report.json', result)
        write(OUTPUT/'last-report.json', result)
        print('Setup report:', work/'report.json', flush=True)
        lock.close()


def main(argv=None):
    args = arguments(argv)
    print(json.dumps(plan(args), indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print('PLAN ONLY: no installs, writes, subprocesses or GPU queries.')
        return
    execute(args)


if __name__ == '__main__':
    main()
