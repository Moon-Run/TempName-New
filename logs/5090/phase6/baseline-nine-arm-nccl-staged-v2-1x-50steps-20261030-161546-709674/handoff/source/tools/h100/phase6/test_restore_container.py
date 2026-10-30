"""CPU-only recovery tests: no installs, network, CUDA imports, or GPU queries."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import restore_container as recovery


def inventory():
    return dict(python=[3, 10, 20], prefix='/persistent/conda',
                executable='/persistent/conda/bin/python',
                versions=dict(recovery.PINS, torch='2.6.0+cu124', pip='26.2.1'))


def completed(stdout='', stderr='', code=0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


class FakeRunner:
    def __init__(self, root):
        self.root = Path(root)
        self.commands = []

    def run(self, command, env=None, cwd=None):
        self.commands.append(dict(argv=list(map(str, command)), env=env, cwd=cwd))


class InventoryTests(unittest.TestCase):
    def test_matching_persistent_environment_has_no_install_actions(self):
        with mock.patch.object(recovery, 'package_installed', return_value=True):
            plan = recovery.installation_plan(inventory(), Path('/cuda124'), Path('/cuda128'))
        self.assertEqual(plan['system_packages'], [])
        self.assertEqual(plan['toolkit_packages'], [])
        self.assertEqual(plan['python_packages'], [])
        self.assertFalse(plan['install_torch'])
        self.assertFalse(plan['ensure_pip'])

    def test_missing_only_128_requests_toolkit_not_driver_or_python_reinstall(self):
        with mock.patch.object(recovery, 'package_installed', return_value=True):
            plan = recovery.installation_plan(inventory(), Path('/cuda124'), None)
        self.assertEqual(plan['toolkit_packages'], ['cuda-toolkit-12-8'])
        self.assertFalse(plan['python_packages'])
        self.assertFalse(plan['install_torch'])
        self.assertTrue(all(not recovery.DRIVER_PACKAGE.match(name)
                            for name in plan['toolkit_packages']))

    def test_version_mismatch_requires_explicit_repair_python(self):
        found = inventory()
        found['versions'].update(torch='2.7.0', numpy='2.0.0')
        with mock.patch.object(recovery, 'package_installed', return_value=True):
            with self.assertRaisesRegex(RuntimeError, '--repair-python'):
                recovery.installation_plan(found, Path('/cuda124'), Path('/cuda128'))
            plan = recovery.installation_plan(found, Path('/cuda124'), Path('/cuda128'), repair=True)
        self.assertTrue(plan['install_torch'])
        self.assertEqual(plan['python_packages'], ['numpy==1.26.4'])
        found['python'] = [3, 12, 0]
        with self.assertRaisesRegex(RuntimeError, 'does not replace Python'):
            recovery.installation_plan(found, None, None, repair=True)

    def test_missing_packages_are_filled_without_replacing_present_packages(self):
        found = inventory()
        found['versions'].update(cmake=None, ninja=None, pip=None)
        with mock.patch.object(recovery, 'package_installed', return_value=True):
            plan = recovery.installation_plan(found, Path('/cuda124'), Path('/cuda128'))
        self.assertEqual(set(plan['python_packages']), {'cmake==3.30.0', 'ninja==1.11.1.3'})
        self.assertTrue(plan['ensure_pip'])
        self.assertFalse(plan['install_torch'])

    def test_working_compiled_apex_and_torch_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner(tmp)
            args = types.SimpleNamespace(python=Path('/persistent/conda/bin/python'))
            plan = dict(ensure_pip=False, install_torch=False, python_packages=[])
            with mock.patch.object(recovery, 'capture', return_value=completed('Apex OK')) as probe, \
                 mock.patch.object(recovery, 'clone_pinned', side_effect=AssertionError('Apex cloned')):
                result = recovery.restore_python(args, plan, runner, {'PATH': '/usr/bin:/bin'})
            self.assertEqual(result, 'reused existing compiled Apex')
            self.assertFalse(any('install' in row['argv'] for row in runner.commands))
            self.assertEqual(probe.call_args.kwargs, {})
            self.assertEqual(probe.call_args.args[1]['CUDA_VISIBLE_DEVICES'], '')

    def test_repairing_torch_rebuilds_apex_even_if_old_extensions_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner(tmp)
            args = types.SimpleNamespace(python=Path('/persistent/conda/bin/python'))
            plan = dict(ensure_pip=False, install_torch=True, python_packages=[])

            def clone(url, target, commit, runner_, env):
                self.assertEqual(commit, recovery.APEX_COMMIT)
                target.mkdir()
                (target/'setup.py').write_text('setup(\n    ext_modules=ext_modules,\n)\n')

            with mock.patch.object(recovery, 'capture', return_value=completed('Old Apex imports')), \
                 mock.patch.object(recovery, 'clone_pinned', side_effect=clone) as cloned:
                result = recovery.restore_python(args, plan, runner, {'PATH': '/usr/bin:/bin'})
            cloned.assert_called_once()
            self.assertIn('rebuilt three Apex extensions', result)
            installs = [row for row in runner.commands if 'install' in row['argv']]
            self.assertEqual(len(installs), 2)
            self.assertIn('torch==2.6.0', installs[0]['argv'])
            self.assertIn('--no-deps', installs[1]['argv'])
            self.assertEqual(installs[1]['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(installs[1]['env']['TORCH_CUDA_ARCH_LIST'], '9.0')


class ToolkitAndAptTests(unittest.TestCase):
    def test_toolkit_requires_matching_nvcc_header_and_shared_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'bin').mkdir()
            (root/'include').mkdir()
            (root/'lib64').mkdir()
            (root/'bin/nvcc').write_text('fixture, not executed')
            with mock.patch.object(recovery, 'capture', return_value=completed('Cuda compilation tools, release 12.8, V12.8.93')):
                self.assertIsNone(recovery.toolkit(root, '12.8'))
                (root/'include/cuda_runtime.h').touch()
                (root/'lib64/libcudart.so').touch()
                self.assertEqual(recovery.toolkit(root, '12.8'), root)
                self.assertIsNone(recovery.toolkit(root, '12.4'))
            with mock.patch.object(recovery, 'capture', return_value=completed(code=1)):
                self.assertIsNone(recovery.toolkit(root, '12.8'))

    def test_explicit_toolkit_path_does_not_silently_substitute_another(self):
        with mock.patch.object(recovery, 'toolkit', return_value=None) as probe, \
             mock.patch.object(recovery.shutil, 'which', side_effect=AssertionError('fallback search')):
            self.assertIsNone(recovery.find_toolkit('12.8', Path('/selected/cuda')))
        probe.assert_called_once_with(Path('/selected/cuda'), '12.8')

    def test_resolved_toolkit_survives_generic_symlink_repointing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version in ('12.4', '12.8'):
                target = root/('cuda-'+version)
                for name in ('bin/nvcc', 'include/cuda_runtime.h', 'lib64/libcudart.so'):
                    path = target/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            alias = root/'cuda'
            alias.symlink_to(root/'cuda-12.4', target_is_directory=True)

            def capture(command, env=None, timeout=120):
                version = Path(command[0]).resolve().parents[1].name.removeprefix('cuda-')
                return completed('Cuda compilation tools, release '+version+', V'+version+'.99')

            with mock.patch.object(recovery, 'capture', side_effect=capture):
                runtime = recovery.toolkit(alias, '12.4')
                self.assertEqual(runtime, root/'cuda-12.4')
                alias.unlink()
                alias.symlink_to(root/'cuda-12.8', target_is_directory=True)
                self.assertEqual(recovery.toolkit(runtime, '12.4'), root/'cuda-12.4')
                self.assertEqual(recovery.toolkit(alias, '12.8'), root/'cuda-12.8')

    def test_supported_apt_repository_and_unsupported_platform_rejection(self):
        self.assertEqual(recovery.apt_repository({'ID': 'ubuntu', 'VERSION_ID': '22.04'}, 'x86_64'), 'ubuntu2204')
        for os_release, machine in [({'ID': 'debian', 'VERSION_ID': '12'}, 'x86_64'),
                                    ({'ID': 'ubuntu', 'VERSION_ID': '22.04'}, 'aarch64')]:
            with self.subTest(os=os_release, machine=machine), self.assertRaises(RuntimeError):
                recovery.apt_repository(os_release, machine)

    def test_apt_rejects_removal_and_driver_installation(self):
        for action in ('Remv old-tool', 'Purg old-tool'):
            with self.subTest(action=action), self.assertRaises(RuntimeError):
                recovery.validate_apt_plan(action+' [1.0]\n')
        blocked = ['cuda', 'cuda-12-8', 'cuda-runtime-12-8', 'cuda-drivers', 'cuda-drivers-570',
                   'cuda-compat-12-8', 'nvidia-driver-570', 'nvidia-dkms-570',
                   'nvidia-kernel-common-570', 'linux-modules-nvidia-570-generic',
                   'libnvidia-compute-570', 'libnvidia-gl-570', 'nvidia-utils-570',
                   'nvidia-headless-570', 'nvidia-open']
        for package in blocked:
            with self.subTest(package=package), self.assertRaises(RuntimeError):
                recovery.validate_apt_plan('Inst '+package+':amd64 (1.0 Ubuntu [amd64])\n')
        recovery.validate_apt_plan('Inst cuda-toolkit-12-8 (12.8.1 NVIDIA [amd64])\n'
                                   'Inst cuda-nvcc-12-8 (12.8.93 NVIDIA [amd64])\n'
                                   'Inst cuda-cudart-12-8 (12.8.90 NVIDIA [amd64])\n'
                                   'Inst nsight-compute-2025.1 (2025.1 NVIDIA [amd64])\n'
                                   'Inst gcc-11 (11.4.0 Ubuntu [amd64])\n')

    def test_dangerous_apt_simulation_prevents_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner(tmp)
            plan = dict(system_packages=[], toolkit_packages=['cuda-toolkit-12-8'])
            def capture(command, env=None, timeout=120):
                if command[0] == 'apt-cache':
                    return completed('Candidate: 12.8.1-1\n')
                return completed('Remv libexisting [1.0]\nInst cuda-toolkit-12-8 (12.8.1)\n')
            with mock.patch.object(recovery.os, 'geteuid', return_value=0), \
                 mock.patch.object(recovery.shutil, 'which', return_value='/usr/bin/apt-get'), \
                 mock.patch.object(recovery, 'apt_repository', return_value='ubuntu2204'), \
                 mock.patch.object(recovery, 'capture', side_effect=capture):
                with self.assertRaisesRegex(RuntimeError, 'removing'):
                    recovery.install_system(plan, runner)
            self.assertFalse(any('install' in row['argv'] for row in runner.commands))
            self.assertTrue((Path(tmp)/'apt-plan.txt').is_file())


class ExecutionTests(unittest.TestCase):
    def test_default_preview_has_no_install_files_or_gpu_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'must-not-exist'
            with mock.patch.object(recovery, 'python_inventory', return_value=inventory()), \
                 mock.patch.object(recovery, 'find_toolkit', side_effect=[Path('/cuda124'), None]), \
                 mock.patch.object(recovery, 'package_installed', return_value=True), \
                 mock.patch.object(recovery, 'Runner', side_effect=AssertionError('runner')), \
                 mock.patch.object(recovery, 'install_system', side_effect=AssertionError('install')), \
                 mock.patch.object(recovery, 'capture', side_effect=AssertionError('GPU or process query')), \
                 contextlib.redirect_stdout(io.StringIO()) as stream:
                plan = recovery.main(['--out', str(out)])
            self.assertFalse(out.exists())
            self.assertEqual(plan['toolkit_packages'], ['cuda-toolkit-12-8'])
            self.assertIn('PLAN ONLY', stream.getvalue())

    def test_recovery_failure_preserves_log_and_never_emits_launch_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'recovery'
            with mock.patch.object(recovery, 'python_inventory', return_value=inventory()), \
                 mock.patch.object(recovery, 'find_toolkit', side_effect=[Path('/cuda124'), Path('/cuda128')]), \
                 mock.patch.object(recovery, 'package_installed', return_value=True), \
                 mock.patch.object(recovery, 'install_system', side_effect=RuntimeError('controlled failure')), \
                 mock.patch.object(recovery, 'recover_sources', side_effect=AssertionError('later stage')), \
                 mock.patch.object(recovery, 'write_launch_files', side_effect=AssertionError('launch wrapper')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'controlled failure'):
                    recovery.main(['--out', str(out), '--execute'])
            report = json.loads((out/'last-report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['training_started'])
            self.assertFalse(report['software_ready'])
            self.assertFalse((out/'run_a_baseline.sh').exists())
            self.assertFalse((out/'env.sh').exists())

    def test_generated_shell_preserves_literal_paths_and_restores_build_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root/"repo 'quote' $(touch INJECTED_REPO)"
            repo.mkdir()
            out = root/"recovery 'quoted'"
            out.mkdir()
            python = root/"conda $(touch INJECTED_PY) 'quoted'"/'bin/python'
            python.parent.mkdir(parents=True)
            python.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                              'print(json.dumps(dict(argv=sys.argv[1:],env=dict(os.environ))))\n')
            python.chmod(0o755)
            runtime = root/"cuda124 $(touch INJECTED_RT) 'quoted'"
            compiler = root/'cuda128 `touch INJECTED_COMPILER`'
            megatron = root/"Megatron $HOME `touch INJECTED_MEGATRON` 'quoted'"
            args = types.SimpleNamespace(out=out, python=python, megatron=megatron)
            with mock.patch.object(recovery, 'REPO', repo), mock.patch.object(recovery, 'HERE', repo/'tools/h100/phase6'):
                recovery.write_launch_files(args, runtime, compiler)
            env = dict(os.environ, PATH='/usr/bin:/bin', PYTHONPATH='/BAD', PYTHONHOME='/BAD',
                       PYTHONOPTIMIZE='1', DEBUG='release', MAKEFLAGS='-j99',
                       LD_LIBRARY_PATH='/existing libs', GIT_CONFIG_COUNT='1',
                       GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0='/prior')
            result = subprocess.run(['/bin/bash', str(out/'run_a_baseline.sh'), '--trials', '1',
                                     '--out', 'literal argument', '--execute'],
                                    env=env, cwd=root, text=True, capture_output=True, check=True)
            actual = json.loads(result.stdout)
            self.assertEqual(actual['argv'], [str(repo/'tools/h100/phase6/detach_a_baseline.py'),
                '--build-cuda-home', str(compiler), '--cuda-home', str(runtime), '--megatron', str(megatron),
                '--trials', '1', '--out', 'literal argument', '--execute'])
            values = actual['env']
            self.assertEqual(values['CC'], '/usr/bin/gcc-11')
            self.assertEqual(values['CXX'], '/usr/bin/g++-11')
            self.assertEqual(values['CUDAHOSTCXX'], '/usr/bin/g++-11')
            self.assertEqual(values['DEBUG'], '0')
            self.assertEqual(values['MAKEFLAGS'], '')
            self.assertEqual(values['CUDA_HOME'], str(runtime))
            self.assertEqual(values['LD_LIBRARY_PATH'], str(runtime/'lib64')+':/existing libs')
            self.assertEqual(values['PATH'].split(':')[:2], [str(python.parent), str(runtime/'bin')])
            self.assertEqual(values['GIT_CONFIG_COUNT'], '5')
            self.assertEqual(values['GIT_CONFIG_VALUE_4'], str(megatron))
            self.assertTrue(all(k not in values for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE')))
            self.assertFalse(list(root.rglob('INJECTED_*')))


if __name__ == '__main__':
    unittest.main()
