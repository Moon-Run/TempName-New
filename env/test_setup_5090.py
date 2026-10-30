"""CPU-only setup checks: no downloads, package installs, CUDA or GPU queries."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import setup_5090 as setup


def stack():
    return dict(python=[3, 12, 3], executable='/image/bin/python', prefix='/image', base_prefix='/image',
                include='/image/include', torch=setup.EXPECTED_TORCH, cuda='12.9', cudnn=91002,
                nccl=[2, 27, 3], cxx11_abi=True, arch_flags='sm_80 sm_90 sm_120 compute_120',
                protected_modules={name: dict(path='/image/'+name, sha256='a'*64)
                                   for name in ('torch', 'torch._C', 'apex', 'apex_C', 'amp_C', 'fused_layer_norm_cuda')},
                versions={})


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.commands = []

    def run(self, command, env=None, cwd=None):
        self.commands.append(list(map(str, command)))


class Plans(unittest.TestCase):
    def test_uses_existing_ngc_stack_and_sibling_source(self):
        args = setup.arguments(['--dry-run'])
        value = setup.plan(args)
        self.assertEqual(args.megatron, setup.REPO.parent/'Megatron-LM')
        self.assertEqual(args.prefix, setup.REPO.parent/'venvs/tempname-5090')
        self.assertTrue(value['inherited_system_packages'])
        self.assertTrue(value['existing_repositories_required'])
        self.assertEqual(value['expected_torch'], '2.8.0a0+5228986c39.nv25.06')
        self.assertFalse(any(value[key] for key in ('install_torch', 'install_cuda', 'install_apex', 'training_started')))

    def test_preview_does_not_execute_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(setup, 'execute', side_effect=AssertionError('execution forbidden')), \
                 patch.object(setup, 'capture', side_effect=AssertionError('query forbidden')), \
                 contextlib.redirect_stdout(io.StringIO()):
                setup.main(['--dry-run', '--state-dir', str(root/'state'), '--prefix', str(root/'venv')])
            self.assertEqual(list(root.iterdir()), [])
            process = subprocess.run(['bash', str(Path(setup.__file__).with_suffix('.sh')), '--dry-run',
                                      '--state-dir', str(root/'state')], capture_output=True, text=True,
                                     env=dict(os.environ, RTX5090_BASE_PYTHON=sys.executable))
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('PLAN ONLY', process.stdout)
            self.assertFalse((root/'state').exists())

    def test_overlapping_source_and_environment_paths_rejected(self):
        for flags in (['--prefix', str(setup.REPO/'env')],
                      ['--prefix', '/tmp/a', '--state-dir', '/tmp/a/cache'],
                      ['--prefix', '/tmp/a', '--megatron', '/tmp/a/source'],
                      ['--state-dir', '/tmp/a', '--megatron', '/tmp/a/source']):
            with self.assertRaises(RuntimeError):
                setup.arguments(flags)

    def test_wrong_image_or_missing_arch_rejected_without_repair(self):
        setup.validate_stack(stack())
        for key, value in (('torch', '2.6.0+cu124'), ('cuda', '12.8'),
                           ('python', [3, 10, 20]), ('arch_flags', 'sm_90')):
            record = stack()
            record[key] = value
            with self.assertRaises(RuntimeError):
                setup.validate_stack(record)


class VenvAndPackages(unittest.TestCase):
    def test_real_venv_inherits_packages_and_owned_repeat_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args = SimpleNamespace(prefix=root/'venv', state_dir=root/'state',
                                   base_python=Path(sys.executable), cuda_home=root/'cuda')
            runner = setup.LiveRunner(root)
            base = stack()
            inherited = dict(base, prefix=str(args.prefix))
            with patch.object(setup, 'inventory', return_value=inherited):
                python = setup.ensure_venv(args, base, runner)
                self.assertTrue(python.is_file())
                self.assertIn('--system-site-packages', runner.commands[0]['argv'])
                self.assertIn('--without-pip', runner.commands[0]['argv'])
                count = len(runner.commands)
                setup.ensure_venv(args, base, runner)
                self.assertEqual(len(runner.commands), count)
                other = copy.deepcopy(base)
                other['protected_modules']['amp_C']['sha256'] = 'b'*64
                with self.assertRaisesRegex(RuntimeError, 'different image'):
                    setup.ensure_venv(args, other, runner)
            cfg = (args.prefix/'pyvenv.cfg').read_text()
            self.assertIn('include-system-site-packages = true', cfg)

    def test_unmanaged_venv_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(prefix=root/'venv', state_dir=root/'state',
                                   base_python=Path(sys.executable), cuda_home=root/'cuda')
            args.prefix.mkdir()
            marker = args.prefix/'user-file'
            marker.write_text('preserve')
            runner = Recorder(root)
            with self.assertRaisesRegex(RuntimeError, 'unmanaged'):
                setup.ensure_venv(args, stack(), runner)
            self.assertEqual(marker.read_text(), 'preserve')
            self.assertEqual(runner.commands, [])

    def test_pip_only_installs_missing_extras_without_dependency_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Recorder(tmp)
            found = dict.fromkeys(setup.EXTRAS, True)
            found['pybind11'] = False
            with patch.object(setup, 'json_probe', return_value=found):
                installed = setup.install_missing_python(Path('/fixture/python'), Path('/fixture/cuda'), runner)
            self.assertEqual(installed, ['pybind11'])
            command = runner.commands[0]
            self.assertEqual(command[:4], ['/fixture/python', '-m', 'pip', 'install'])
            self.assertIn('--no-deps', command)
            self.assertNotIn('--upgrade', command)
            self.assertEqual(command[-1], 'pybind11')
            self.assertFalse(set(command).intersection({'torch', 'apex', 'transformer-engine'}))

    def test_complete_image_needs_no_pip_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Recorder(tmp)
            with patch.object(setup, 'json_probe', return_value=dict.fromkeys(setup.EXTRAS, True)):
                self.assertEqual(setup.install_missing_python(Path('/python'), Path('/cuda'), runner), [])
            self.assertFalse(any('pip' in command for command in runner.commands))

    def test_system_install_cannot_introduce_driver_or_remove_packages(self):
        for text in ('Inst cuda-drivers (1 archive)\n', 'Remv python3 [3.12]\n'):
            with tempfile.TemporaryDirectory() as tmp:
                runner = Recorder(tmp)
                with patch.object(setup.sources.recovery, 'package_installed', return_value=False), \
                     patch.object(setup.os, 'geteuid', return_value=0), \
                     patch.object(setup.shutil, 'which', return_value='/usr/bin/apt-get'), \
                     patch.object(setup, 'capture', return_value=text):
                    with self.assertRaises(RuntimeError):
                        setup.install_system(runner)
                self.assertEqual(len(runner.commands), 1)
                self.assertIn('update', runner.commands[0])


class ExistingSources(unittest.TestCase):
    def fixture(self, root):
        repo = root/'Megatron-LM'
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), '-c', 'user.name=CPU Fixture',
                '-c', 'user.email=fixture@example.invalid', *args], text=True, stderr=subprocess.DEVNULL).strip()
        git('init')
        (repo/'source.py').write_text('first\n')
        git('add', 'source.py')
        git('commit', '-m', 'first')
        target = git('rev-parse', 'HEAD')
        (repo/'source.py').write_text('second\n')
        git('commit', '-am', 'second')
        return repo, target, git

    def test_existing_clean_clone_is_pinned_without_recloning(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            repo, target, git = self.fixture(root)
            args = SimpleNamespace(megatron=repo)
            runner = setup.LiveRunner(root)
            with patch.object(setup, 'MEGATRON_COMMIT', target), \
                 patch.object(setup.sources.recovery, 'MEGATRON_COMMIT', target), \
                 patch.object(setup.sources, 'ensure_submodules'), patch.object(setup.sources, 'software_sources_check'):
                record = setup.ensure_sources(args, runner)
            self.assertEqual(git('rev-parse', 'HEAD'), target)
            self.assertFalse(record['repositories_recloned'])
            self.assertFalse(any('clone' in entry['argv'] or 'fetch' in entry['argv'] for entry in runner.commands))

    def test_dirty_clone_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, target, git = self.fixture(Path(tmp))
            before = git('rev-parse', 'HEAD')
            (repo/'source.py').write_text('user changes\n')
            runner = Recorder(tmp)
            with self.assertRaisesRegex(RuntimeError, 'Tracked edits'):
                setup.ensure_sources(SimpleNamespace(megatron=repo), runner)
            self.assertEqual(git('rev-parse', 'HEAD'), before)
            self.assertEqual((repo/'source.py').read_text(), 'user changes\n')
            self.assertEqual(runner.commands, [])

    def test_missing_clone_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)/'Megatron-LM'
            with self.assertRaisesRegex(RuntimeError, 'Existing Megatron'):
                setup.ensure_sources(SimpleNamespace(megatron=repo), Recorder(tmp))
            self.assertFalse(repo.exists())


class Integration(unittest.TestCase):
    def test_generated_shell_quotes_paths_and_trusts_only_exact_checkouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(prefix=root/'venv with spaces', cuda_home=root/'cuda',
                                   megatron=root/'literal$(touch injected)')
            envfile = root/'env.sh'
            envfile.write_text(setup.environment_text(args, Path(sys.executable)))
            command = ['bash', '-c', 'source "$1"; "$2" -c \'import json,os; print(json.dumps(dict(os.environ)))\'',
                       'fixture', str(envfile), sys.executable]
            value = json.loads(subprocess.check_output(command, cwd=root, text=True,
                                                      env=dict(os.environ, GIT_CONFIG_COUNT='0')))
            self.assertEqual(value['RTX5090_SETUP_MEGATRON'], str(args.megatron))
            self.assertEqual(value['GIT_CONFIG_VALUE_3'], str(args.megatron))
            self.assertFalse((root/'injected').exists())
            self.assertNotIn('safe.directory=*', envfile.read_text())

    def pipeline(self, root, *, skip=False, fail_gpu=False):
        repo = root/'repo'
        repo.mkdir()
        output = repo/'outputs/5090/env-setup'
        args = SimpleNamespace(prefix=root/'venv', base_python=Path('/image/python'), megatron=root/'Megatron-LM',
                               state_dir=root/'state', cuda_home=Path('/image/cuda'), skip_gpu_check=skip)
        gpu = RuntimeError('NCCL probe failed') if fail_gpu else dict(collective_probe_passed=True)
        with patch.object(setup, 'REPO', repo), patch.object(setup, 'OUTPUT', output), \
             patch.object(setup, 'check_host'), patch.object(setup, 'LiveRunner', Recorder), \
             patch.object(setup, 'inventory', return_value=stack()), patch.object(setup, 'validate_toolkit', return_value={}), \
             patch.object(setup.sources, 'add_git_trust'), patch.object(setup, 'ensure_sources', return_value={}), \
             patch.object(setup, 'install_system', return_value=[]), \
             patch.object(setup, 'ensure_venv', return_value=args.prefix/'bin/python'), \
             patch.object(setup, 'install_missing_python', return_value=[]), \
             patch.object(setup, 'compile_probe', return_value={'gpu_used': False}), \
             patch.object(setup, 'verify_megatron', return_value={}), \
             patch.object(setup, 'verify_gpu', side_effect=gpu if fail_gpu else None, return_value=gpu) as probe, \
             contextlib.redirect_stdout(io.StringIO()):
            if fail_gpu:
                with self.assertRaisesRegex(RuntimeError, 'NCCL probe failed'):
                    setup.execute(args)
            else:
                setup.execute(args)
        return setup.read(output/'last-report.json'), (output/'env.sh').is_file(), probe.call_count

    def test_ready_environment_only_published_after_all_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, published, probes = self.pipeline(Path(tmp))
            self.assertEqual(record['status'], 'completed')
            self.assertTrue(record['software_ready'] and record['gpu_checked'] and published)
            self.assertEqual(probes, 1)
            self.assertFalse(record['training_started'])

    def test_skipped_gpu_checks_and_failed_gpu_checks_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, published, probes = self.pipeline(Path(tmp), skip=True)
            self.assertTrue(record['software_ready'] and record['gpu_checks_pending'] and published)
            self.assertFalse(record['gpu_checked'])
            self.assertEqual(probes, 0)
        with tempfile.TemporaryDirectory() as tmp:
            record, published, probes = self.pipeline(Path(tmp), fail_gpu=True)
            self.assertEqual(record['status'], 'failed')
            self.assertFalse(record['software_ready'] or record['gpu_checked'] or published)

    def test_launcher_reads_setup_paths_and_respects_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root/'tools/5090/phase6'
            scripts.mkdir(parents=True)
            shutil.copy2(setup.REPO/'tools/5090/phase6/run_baseline.sh', scripts/'run_baseline.sh')
            (scripts/'baseline.py').write_text('import json,sys; print(json.dumps(sys.argv[1:]))\n')
            envfile = root/'outputs/5090/env-setup/env.sh'
            envfile.parent.mkdir(parents=True)
            envfile.write_text('export RTX5090_SETUP_PYTHON='+setup.shlex.quote(sys.executable)+'\n'
                               'export RTX5090_SETUP_CUDA_HOME=/configured/cuda\n'
                               'export RTX5090_SETUP_MEGATRON=/configured/Megatron-LM\n')
            env = {k: v for k, v in os.environ.items() if not k.startswith('RTX5090_') and k != 'CUDA_HOME'}
            result = subprocess.check_output(['bash', str(scripts/'run_baseline.sh')], env=env, text=True)
            self.assertEqual(json.loads(result), ['--cuda-home', '/configured/cuda', '--build-cuda-home',
                                                  '/configured/cuda', '--megatron', '/configured/Megatron-LM'])
            result = subprocess.check_output(['bash', str(scripts/'run_baseline.sh'), '--megatron', '/cli/Megatron-LM'],
                                             env=dict(env, RTX5090_CUDA_HOME='/explicit/cuda'), text=True)
            values = json.loads(result)
            self.assertEqual(values[1], '/explicit/cuda')
            self.assertEqual(values[-2:], ['--megatron', '/cli/Megatron-LM'])


if __name__ == '__main__':
    unittest.main()
