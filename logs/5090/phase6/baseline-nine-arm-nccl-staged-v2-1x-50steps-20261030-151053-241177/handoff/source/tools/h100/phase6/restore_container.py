#!/usr/bin/env python3
"""Recover a recreated H100 container while reusing the persistent conda prefix.

Default: inspect and print a plan, without installs/files/GPU queries.
--execute installs missing tools and packages, validates the environment and
writes env.sh plus run_a_baseline.sh. It never starts the training experiment.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_PYTHON = Path('/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python')
MEGATRON_COMMIT = '3ea68ad6042cc1204386ae9364358f7c4de1bc37'
APEX_COMMIT = 'e13873debc4699d39c6861074b9a3b2a02327f92'
PINS = {'setuptools': '75.8.0', 'wheel': '0.45.1', 'packaging': '24.2',
        'numpy': '1.26.4', 'einops': '0.8.1', 'regex': '2024.11.6',
        'psutil': '6.1.1', 'PyYAML': '6.0.2', 'tqdm': '4.67.1', 'six': '1.17.0',
        'pybind11': '2.13.6', 'ninja': '1.11.1.3', 'cmake': '3.30.0'}
SYSTEM_PACKAGES = ('build-essential', 'gcc-11', 'g++-11', 'make', 'git',
                   'curl', 'ca-certificates', 'pkg-config', 'binutils')
TOOLKITS = {'12.4': 'cuda-toolkit-12-4', '12.8': 'cuda-toolkit-12-8'}
DRIVER_PACKAGE = re.compile(r'^(?:cuda(?:-[0-9]+(?:-[0-9]+)?)?|cuda-runtime(?:-.*)?|cuda-drivers(?:-.*)?|cuda-compat(?:-.*)?|nvidia-(?:driver|dkms|kernel|utils|headless|open|modprobe|persistenced|fabricmanager|settings|compute-utils)(?:-.*)?|libnvidia-.*|linux-modules-nvidia-.*)$')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def capture(command, env=None, timeout=120):
    return subprocess.run(list(map(str, command)), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def clean_environment(python, cuda=None):
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               DEBUG='0', MAKEFLAGS='', MAX_JOBS='2', CUDA_DEVICE_MAX_CONNECTIONS='1')
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE'):
        env.pop(name, None)
    paths = [str(Path(python).parent)]
    if cuda:
        paths.append(str(Path(cuda)/'bin'))
        env.update(CUDA_HOME=str(cuda), CUDACXX=str(Path(cuda)/'bin/nvcc'))
        env['LD_LIBRARY_PATH'] = str(Path(cuda)/'lib64') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['PATH'] = ':'.join(paths) + ':' + env.get('PATH', '/usr/bin:/bin')
    env.update(CC='/usr/bin/gcc-11', CXX='/usr/bin/g++-11', CUDAHOSTCXX='/usr/bin/g++-11')
    return env


def python_inventory(python):
    require(Path(python).is_file(), 'Persistent Python missing: '+str(python)+'. Mount the original conda prefix or pass --python; no environment is deleted/recreated.')
    source = '''import importlib.metadata as m, json, sys
names = '''+repr(['torch', 'pip', *PINS])+'''
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps(dict(python=list(sys.version_info[:3]), prefix=sys.prefix, executable=sys.executable, versions=versions)))
'''
    result = capture([python, '-c', source], clean_environment(python))
    require(result.returncode == 0, 'Persistent Python cannot run: '+result.stderr)
    return json.loads(result.stdout)


def toolkit(root, version):
    root = Path(root)
    if not (root/'bin/nvcc').is_file():
        return None
    result = capture([root/'bin/nvcc', '--version'])
    if result.returncode or not re.search(r'\brelease '+re.escape(version)+r'[,\s]', result.stdout):
        return None
    if not (root/'include/cuda_runtime.h').is_file() or not (root/'lib64/libcudart.so').is_file():
        return None
    # Installing another toolkit may repoint /usr/local/cuda. Bind to the real
    # versioned root now, so the selected 12.4 runtime cannot silently become 12.8.
    return root.resolve()


def find_toolkit(version, requested=None, python=DEFAULT_PYTHON):
    if requested is not None:
        return toolkit(requested, version)
    candidates = [Path('/usr/local')/('cuda-'+version), Path('/opt')/('cuda-'+version),
                  Path('/workspace')/('cuda-'+version), Path('/usr/local/cuda'), Path(python).parent.parent]
    found = shutil.which('nvcc')
    if found:
        candidates.append(Path(found).resolve().parent.parent)
    for root in candidates:
        good = toolkit(root, version)
        if good is not None:
            return good
    return None


def apt_repository(os_release=None, machine=None):
    if os_release is None:
        os_release = {}
        for line in Path('/etc/os-release').read_text().splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                os_release[key] = value.strip('"')
    machine = machine or platform.machine()
    require(machine == 'x86_64' and os_release.get('ID') == 'ubuntu' and
            os_release.get('VERSION_ID') in ('20.04', '22.04', '24.04'),
            'Automatic system recovery supports Ubuntu 20.04/22.04/24.04 x86_64; install the required toolkits/compiler manually on this OS and pass their paths.')
    return 'ubuntu'+os_release['VERSION_ID'].replace('.', '')


def package_installed(name):
    if not shutil.which('dpkg-query'):
        return False
    result = capture(['dpkg-query', '-W', '-f=${Status}', name])
    return result.returncode == 0 and result.stdout.strip() == 'install ok installed'


def installation_plan(inventory, runtime, compiler, repair=False):
    require(inventory['python'] == [3, 10, 20],
            'Expected persistent Python 3.10.20, got '+str(inventory['python'])+'. Select the original --python; this script does not replace Python.')
    versions = inventory['versions']
    mismatches = {k: [versions.get(k), v] for k, v in PINS.items() if versions.get(k) not in (None, v)}
    torch_version = versions.get('torch')
    if torch_version not in (None, '2.6.0+cu124'):
        mismatches['torch'] = [torch_version, '2.6.0+cu124']
    require(repair or not mismatches,
            'Existing package versions differ from the recorded environment: '+json.dumps(mismatches)+
            '. Select the correct --python, or use --repair-python to explicitly restore pinned package versions.')
    return dict(system_packages=[p for p in SYSTEM_PACKAGES if not package_installed(p)],
                toolkit_packages=[TOOLKITS[v] for v, home in [('12.4', runtime), ('12.8', compiler)] if home is None],
                python_packages=[k+'=='+v for k, v in PINS.items() if versions.get(k) != v],
                install_torch=torch_version != '2.6.0+cu124',
                ensure_pip=versions.get('pip') is None)


class Runner:
    def __init__(self, root):
        self.root = Path(root)
        self.commands = []

    def run(self, command, env=None, cwd=None):
        command = list(map(str, command))
        self.commands.append(dict(argv=command, cwd=str(cwd) if cwd else None))
        print('+ '+shlex.join(command), flush=True)
        with (self.root/'install.log').open('a') as log:
            log.write('\nCOMMAND '+shlex.join(command)+'\n')
            log.flush()
            completed = subprocess.run(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        require(completed.returncode == 0, 'Command failed ('+str(completed.returncode)+'); see '+str(self.root/'install.log'))


def validate_apt_plan(text):
    for line in text.splitlines():
        if line.startswith(('Remv ', 'Purg ')):
            raise RuntimeError('APT proposes removing packages; refusing this recovery transaction: '+line)
        if line.startswith('Inst '):
            package = line.split()[1].split(':')[0]
            require(not DRIVER_PACKAGE.match(package), 'APT proposes a driver/runtime meta-package: '+package)


def install_system(plan, runner):
    wanted = plan['system_packages']+plan['toolkit_packages']
    if not wanted:
        return
    require(os.geteuid() == 0 and shutil.which('apt-get'), 'Run --execute as container root to install missing system packages')
    distro = apt_repository()
    env = dict(os.environ, DEBIAN_FRONTEND='noninteractive', LC_ALL='C')
    apt = ['apt-get', '-o', 'Dpkg::Lock::Timeout=60', '-o', 'Acquire::Retries=3']
    runner.run(apt+['update'], env)
    # Bootstrap HTTPS tooling from the container's existing OS repositories.
    bootstrap = [p for p in ('ca-certificates', 'curl') if p in plan['system_packages']]
    if bootstrap:
        simulation = capture(apt+['-s', 'install', '--no-install-recommends', *bootstrap], env)
        require(simulation.returncode == 0, simulation.stderr)
        validate_apt_plan(simulation.stdout)
        runner.run(apt+['install', '-y', '--no-install-recommends', *bootstrap], env)
    if plan['toolkit_packages']:
        unavailable = []
        for name in plan['toolkit_packages']:
            policy = capture(['apt-cache', 'policy', name], env)
            if not re.search(r'Candidate:\s+(?!\(none\))\S+', policy.stdout):
                unavailable.append(name)
        if unavailable:
            keyring = runner.root/'cuda-keyring_1.1-1_all.deb'
            url = 'https://developer.download.nvidia.com/compute/cuda/repos/'+distro+'/x86_64/cuda-keyring_1.1-1_all.deb'
            runner.run(['curl', '--fail', '--location', '--retry', '3', '--proto', '=https', '--tlsv1.2', '-o', keyring, url], env)
            runner.run(['dpkg', '-i', keyring], env)
            runner.run(apt+['update'], env)
    simulation = capture(apt+['-s', 'install', '--no-install-recommends', *wanted], env)
    (runner.root/'apt-plan.txt').write_text(simulation.stdout+simulation.stderr)
    require(simulation.returncode == 0, 'Requested toolkit/compiler packages are unavailable; see apt-plan.txt. CUDA12.4 packages may require an Ubuntu22.04 image. No alternate CUDA version was substituted.')
    validate_apt_plan(simulation.stdout)
    runner.run(apt+['install', '-y', '--no-install-recommends', *wanted], env)


def git_command(path, *args):
    # Trust only the user-selected exact checkout, without changing global config.
    return ['git', '-c', 'safe.directory='+str(Path(path).resolve()), '-C', str(path), *args]


def check_checkout(path, commit):
    result = capture(git_command(path, 'rev-parse', 'HEAD'))
    require(result.returncode == 0 and result.stdout.strip() == commit, 'Source commit mismatch: '+str(path))
    result = capture(git_command(path, 'status', '--porcelain', '--untracked-files=no'), timeout=300)
    require(result.returncode == 0 and not result.stdout.strip(), 'Tracked source modifications at '+str(path)+'; preserve them and select a clean checkout')


def recover_sources(args, runner, env):
    require((REPO/'.git').exists(), 'Run from a complete TempName git checkout')
    for name in ('cutlass', 'nccl'):
        relative = '3rdparty/'+name
        path = REPO/relative
        row = capture(git_command(REPO, 'ls-tree', 'HEAD', relative))
        require(row.returncode == 0 and row.stdout.startswith('160000'), 'Missing pinned submodule: '+relative)
        commit = row.stdout.split()[2]
        if not (path/'.git').exists():
            runner.run(git_command(REPO, 'submodule', 'update', '--init', '--recursive', '--', relative), env)
        check_checkout(path, commit)
    if not args.megatron.exists():
        clone_pinned('https://github.com/NVIDIA/Megatron-LM.git', args.megatron, MEGATRON_COMMIT, runner, env)
    check_checkout(args.megatron, MEGATRON_COMMIT)
    # Verify the sources used by this ablation, without importing Torch or building.
    sys.path.insert(0, str(HERE))
    import a_baseline_build
    import baseline_codec_build
    a_baseline_build._frozen_sources()
    baseline_codec_build.source_identity()


def clone_pinned(url, target, commit, runner, env):
    require(not Path(target).exists(), 'Use a new source directory: '+str(target))
    runner.run(['git', 'init', target], env)
    runner.run(git_command(target, 'remote', 'add', 'origin', url), env)
    runner.run(git_command(target, 'fetch', '--depth', '1', 'origin', commit), env)
    runner.run(git_command(target, 'checkout', '--detach', 'FETCH_HEAD'), env)
    check_checkout(target, commit)


IMPORT_PROBE = r'''import importlib, json, sys, torch, numpy
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torch.version.cuda == '12.4', torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is False
assert numpy.__version__ == '1.26.4'
for name in ('einops', 'regex', 'psutil', 'yaml', 'tqdm', 'six', 'packaging', 'pybind11'):
    importlib.import_module(name)
print(json.dumps(dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                     abi=torch._C._GLIBCXX_USE_CXX11_ABI, numpy=numpy.__version__)))
'''
APEX_PROBE = '''import torch, apex_C, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
print('Apex compiled extensions: OK')
'''
GPU_PROBE = '''import ctypes, json, torch
ctypes.CDLL('libcuda.so.1')
assert torch.cuda.device_count() == 8, 'Expose exactly eight GPUs with --gpus all'
torch.cuda.set_device(0)
devices=[]
for i in range(8):
    p=torch.cuda.get_device_properties(i)
    assert 'H100' in p.name and (p.major,p.minor)==(9,0) and p.multi_processor_count==132
    devices.append(dict(name=p.name,uuid=str(p.uuid),sm_count=p.multi_processor_count))
assert all(torch.cuda.can_device_access_peer(i,j) for i in range(8) for j in range(8) if i!=j)
print(json.dumps(devices))
'''


def restore_python(args, plan, runner, env):
    if plan['ensure_pip']:
        runner.run([args.python, '-m', 'ensurepip'], env)
    if plan['install_torch']:
        runner.run([args.python, '-m', 'pip', 'install', 'torch==2.6.0',
                    '--index-url', 'https://download.pytorch.org/whl/cu124'], env)
    if plan['python_packages']:
        runner.run([args.python, '-m', 'pip', 'install', '--no-deps', *plan['python_packages']], env)
    runner.run([args.python, '-m', 'pip', 'check'], env)
    runner.run([args.python, '-c', IMPORT_PROBE], dict(env, CUDA_VISIBLE_DEVICES=''))
    apex = capture([args.python, '-c', APEX_PROBE], dict(env, CUDA_VISIBLE_DEVICES=''))
    (runner.root/'apex-import-before.txt').write_text(apex.stdout+apex.stderr)
    if apex.returncode == 0 and not plan['install_torch']:
        return 'reused existing compiled Apex'
    src = runner.root/'apex-source'
    clone_pinned('https://github.com/NVIDIA/apex.git', src, APEX_COMMIT, runner, env)
    setup = src/'setup.py'
    text = setup.read_text()
    anchor = '    ext_modules=ext_modules,'
    require(text.count(anchor) == 1, 'Pinned Apex setup.py no longer matches the minimal-extension patch')
    setup.write_text(text.replace(anchor, '    ext_modules=[ext for ext in ext_modules if ext.name in {"apex_C", "amp_C", "fused_layer_norm_cuda"}],'))
    build_env = dict(env, CUDA_VISIBLE_DEVICES='', TORCH_CUDA_ARCH_LIST='9.0')
    runner.run([args.python, '-m', 'pip', 'install', '-v', '--no-deps', '--no-build-isolation',
                '--config-settings', '--build-option=--cpp_ext',
                '--config-settings', '--build-option=--cuda_ext', src], build_env)
    runner.run([args.python, '-c', APEX_PROBE], dict(env, CUDA_VISIBLE_DEVICES=''))
    return 'rebuilt three Apex extensions from '+APEX_COMMIT


def write_launch_files(args, runtime, compiler):
    variables = dict(PYTHONNOUSERSITE='1', CC='/usr/bin/gcc-11', CXX='/usr/bin/g++-11',
                     CUDAHOSTCXX='/usr/bin/g++-11', DEBUG='0', MAKEFLAGS='', MAX_JOBS='2',
                     CUDA_DEVICE_MAX_CONNECTIONS='1', NCCL_NVLS_ENABLE='0',
                     CUDA_HOME=str(runtime), CUDACXX=str(runtime/'bin/nvcc'),
                     H100_RECOVERY_PYTHON=str(args.python), H100_RECOVERY_BUILD_CUDA=str(compiler))
    lines = ['# Generated by restore_container.py; source before a direct launch.',
             'unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE']
    lines += ['export '+k+'='+shlex.quote(v) for k,v in variables.items()]
    lines += ['export PATH='+shlex.quote(str(args.python.parent)+':'+str(runtime/'bin'))+':"${PATH}"',
              'export LD_LIBRARY_PATH='+shlex.quote(str(runtime/'lib64'))+'${LD_LIBRARY_PATH:+:"$LD_LIBRARY_PATH"}']
    # The selected roots may be bind-mounted and owned by another UID. Scope trust
    # to these exact paths, propagated to child git commands, never safe.directory=*.
    lines += ['export GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-0}"']
    for path in (REPO, REPO/'3rdparty/cutlass', REPO/'3rdparty/nccl', args.megatron):
        lines += ['export "GIT_CONFIG_KEY_${GIT_CONFIG_COUNT}=safe.directory"',
                  'export "GIT_CONFIG_VALUE_${GIT_CONFIG_COUNT}='+str(path).replace('\\','\\\\').replace('"','\\"').replace('$','\\$').replace('`','\\`')+'"',
                  'GIT_CONFIG_COUNT=$((GIT_CONFIG_COUNT + 1))']
    (args.out/'env.sh').write_text('\n'.join(lines)+'\n')
    command = [str(args.python), str(HERE/'detach_a_baseline.py'), '--build-cuda-home', str(compiler),
               '--cuda-home', str(runtime), '--megatron', str(args.megatron)]
    wrapper = '#!/usr/bin/env bash\nset -euo pipefail\nsource '+shlex.quote(str(args.out/'env.sh'))+'\ncd '+shlex.quote(str(REPO))+'\nexec '+shlex.join(command)+' "$@"\n'
    path = args.out/'run_a_baseline.sh'
    path.write_text(wrapper)
    path.chmod(0o755)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    p.add_argument('--cuda-home', type=Path, help='Existing CUDA12.4 root; otherwise autodetect/install')
    p.add_argument('--build-cuda-home', type=Path, help='Existing CUDA12.8 root; otherwise autodetect/install')
    p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--out', type=Path, default=REPO/'outputs/h100/container-recovery')
    p.add_argument('--repair-python', action='store_true', help='Also restore mismatched Python packages to the recorded pins')
    p.add_argument('--skip-gpu-check', action='store_true', help='Validate software only; GPU readiness remains unverified')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().absolute())
    return args


def main(argv=None):
    args = arguments(argv)
    inventory = python_inventory(args.python)
    runtime = find_toolkit('12.4', args.cuda_home, args.python)
    compiler = find_toolkit('12.8', args.build_cuda_home, args.python)
    for requested, found, label in ((args.cuda_home, runtime, '12.4'), (args.build_cuda_home, compiler, '12.8')):
        require(requested is None or found is not None,
                'Explicit CUDA'+label+' path is incomplete or has the wrong nvcc version: '+str(requested)+'. Omit this option to install at /usr/local/cuda-'+label)
    plan = installation_plan(inventory, runtime, compiler, args.repair_python)
    print(json.dumps(dict(python=str(args.python), runtime_cuda=str(runtime) if runtime else 'missing',
                         build_cuda=str(compiler) if compiler else 'missing', plan=plan), indent=2))
    if not args.execute:
        print('PLAN ONLY: no installation, files or GPU queries. Use --execute inside the new container.')
        return plan
    if plan['system_packages'] or plan['toolkit_packages']:
        require(os.geteuid() == 0, 'Use container root for system installation')
        apt_repository()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    work = args.out/'runs'/stamp
    work.mkdir(parents=True, exist_ok=False)
    runner = Runner(work)
    report = dict(status='running', python_before=inventory, plan=plan, commands=runner.commands,
                  started_at=stamp, training_started=False, gpu_checked=False)
    try:
        install_system(plan, runner)
        runtime = toolkit(runtime, '12.4') if runtime else find_toolkit('12.4', python=args.python)
        compiler = toolkit(compiler, '12.8') if compiler else find_toolkit('12.8', python=args.python)
        require(runtime is not None and compiler is not None, 'CUDA toolkit installation did not provide complete 12.4 and 12.8 roots')
        env = clean_environment(args.python, runtime)
        for command in (['/usr/bin/gcc-11', '--version'], ['/usr/bin/g++-11', '--version'],
                        [runtime/'bin/nvcc', '--version'], [compiler/'bin/nvcc', '--version']):
            runner.run(command, env)
        report['apex'] = restore_python(args, plan, runner, env)
        for command in (['cmake', '--version'], ['ninja', '--version']):
            runner.run(command, env)
        recover_sources(args, runner, env)
        probe = work/'compile-probe.cu'
        probe.write_text('#include <cuda_runtime.h>\n#include <cuda_fp8.h>\n#include <cublas_v2.h>\n__global__ void probe(float* x) { x[threadIdx.x] = 1.0f; }\n')
        runner.run([compiler/'bin/nvcc', '-std=c++17', '-arch=sm_90', '-ccbin', '/usr/bin/g++-11',
                    '-c', probe, '-o', work/'compile-probe.o'], dict(env, CUDA_VISIBLE_DEVICES=''))
        if not args.skip_gpu_check:
            require(shutil.which('nvidia-smi'), 'nvidia-smi missing: recreate the container with GPU/driver injection (--gpus all). Do not install a host driver in this container.')
            runner.run(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv'], env)
            runner.run([args.python, '-c', GPU_PROBE], env)
            report['gpu_checked'] = True
        report.update(status='completed', software_ready=True, runtime_cuda=str(runtime), build_cuda=str(compiler),
                      python_after=python_inventory(args.python))
        write_launch_files(args, runtime, compiler)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', software_ready=False)
        raise
    finally:
        for path in (work/'report.json', args.out/'last-report.json'):
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('Recovery completed. Log: '+str(work/'install.log'))
    print('Run one comparison: '+shlex.join(['bash', str(args.out/'run_a_baseline.sh'), '--trials', '1', '--execute']))
    return report


if __name__ == '__main__':
    main()
