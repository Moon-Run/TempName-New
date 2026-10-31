"""CPU-only integration checks; no GPU timing data is synthesized as results."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rtx5090_loader import HERE, REPO, load


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launcher = module('rtx5090_controller_test', 'baseline.py')
preflight = module('rtx5090_preflight_test', 'baseline_preflight.py')


class Launcher(unittest.TestCase):
    def test_ma11_gpu_probe_evidence_rejects_wrong_device_or_numerical_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compat = dict(protocol='CPU-FIXTURE', torch='2.8.0+CPU-FIXTURE')
            hardware = dict(devices=[dict(uuid='GPU-0')], environment=dict(cuda='CPU-CUDA'))
            path = root/'ma11-compat/self-test/probe-result.json'
            result = dict(protocol=compat['protocol'], completed=True, performance_measured=False,
                          gpu=0, uuid='GPU-0', torch=compat['torch'], cuda='CPU-CUDA', library=compat,
                          self_test=dict(passed=True, cases=6, forward_bitwise=True,
                                         backward_bitwise=True, rng_unchanged=True))
            launcher.write(path, result)
            evidence = dict(result_sha256=launcher.sha(path))
            launcher.verify_ma11_probe(root, compat, evidence, hardware)
            for update in ({'uuid': 'wrong'}, {'completed': False},
                           {'self_test': dict(passed=False)}):
                changed = dict(result, **update)
                launcher.write(path, changed)
                with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'did not pass'):
                    launcher.verify_ma11_probe(root, compat, dict(result_sha256=launcher.sha(path)), hardware)
            with self.assertRaisesRegex(ValueError, 'Missing/changed'):
                launcher.verify_ma11_probe(root, compat, evidence, hardware)

    def test_ma11_library_is_injected_only_for_the_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launcher.arguments(['--out', str(root/'out')])
            plan = dict(run_id='CPU', cuda_visible_devices='0,1,2,3,4,5,6,7',
                        communication=launcher.capture_communication(),
                        ma11_compat=dict(library='/CPU-FIXTURE/ma11.so'))
            for arm in load('protocol').ARMS:
                job = dict(arm=arm, mode='smoke', trial=1, block=0, index=0)
                env = launcher.worker_environment(args, root/'build', root/'scripts', job, root/'results', plan)
                if arm == 'ma11':
                    self.assertEqual(env['RTX5090_MA11_SOFTMAX_LIBRARY'], '/CPU-FIXTURE/ma11.so')
                else:
                    self.assertNotIn('RTX5090_MA11_SOFTMAX_LIBRARY', env)

    def test_new_batches_default_to_sibling_data_repository_without_moving_old_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root/'TempName-New'
            old = repo/'logs/5090/phase6/existing-batch'
            old.mkdir(parents=True)
            (old/'report.json').write_text('existing result\n')
            with patch.object(launcher, 'REPO', repo):
                for mode in ([], ['--preflight-only'], ['--transport-smoke-only']):
                    args = launcher.arguments([*mode, '--baseline-build', str(old/'build')])
                    self.assertEqual(args.out.parent, root/'Load-5090/logs/5090/phase6')
                    self.assertTrue(args.out.name.startswith('baseline-nine-arm-nccl-staged-v2-1x-50steps-'))
                    self.assertEqual(args.baseline_build, old/'build')
            self.assertEqual((old/'report.json').read_text(), 'existing result\n')
            self.assertFalse((root/'Load-5090').exists())

    def test_explicit_output_and_historical_report_paths_remain_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp)/'TempName-New/logs/5090/phase6/old-batch'
            for mode in ([], ['--report']):
                args = launcher.arguments([*mode, '--out', str(old)])
                self.assertEqual(args.out, old)

    def test_transport_smoke_shell_preview_does_not_read_or_repair_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(os.environ, RTX5090_PYTHON=sys.executable)
            result = subprocess.run(['bash', str(HERE/'run_transport_smoke.sh'),
                                     '--baseline-build', str(root/'missing-build'), '--out', str(root/'out')],
                                    cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout[result.stdout.index('{'):])
            self.assertTrue(plan['transport_smoke_only'])
            self.assertEqual(plan['formal_optimizer_steps'], 0)
            self.assertFalse((root/'out').exists())
            self.assertFalse((root/'missing-build').exists())

    def test_reused_legacy_build_is_repaired_and_verified_in_one_launch(self):
        import repair_native_rank as repair
        builder = load('build')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launcher.arguments(['--out', str(root/'out')])
            environment = {'CPU_FIXTURE': True}
            receipt = dict(status='repaired', backup='CPU-FIXTURE-backup')
            for identity in repair.LEGACY_FRONTEND_SHA256:
                launcher.write(root/'build/baseline-build.json', dict(environment=environment,
                               transport_inputs={'nccl_transport.py': identity}))
                stages = []
                with patch.object(repair, 'repair', return_value=receipt) as migrate, \
                     patch.object(builder, 'environment_identity', return_value=environment), \
                     patch.object(builder, 'verify', return_value={'CPU_FIXTURE': True}) as verify:
                    launcher.reuse_build(args, root/'build', stages.append)
                self.assertEqual(stages, ['repair-transport-frontend', 'verify-build'])
                migrate.assert_called_once_with(root/'build')
                verify.assert_called_once_with(root/'build', True)
                self.assertEqual(launcher.read(args.out/'build-frontend-repair.json'), receipt)

    def test_changed_environment_and_unknown_build_are_not_repaired(self):
        import repair_native_rank as repair
        builder = load('build')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launcher.arguments(['--out', str(root/'out')])
            for identity in (repair.LEGACY_SHA256, 'unknown-frontend'):
                launcher.write(root/'build/baseline-build.json', dict(environment={'version': 'old'},
                               transport_inputs={'nccl_transport.py': identity}))
                with patch.object(repair, 'repair') as migrate, \
                     patch.object(builder, 'environment_identity', return_value={'version': 'new'}), \
                     patch.object(builder, 'verify') as verify:
                    with self.assertRaisesRegex(ValueError, 'environment changed'):
                        launcher.reuse_build(args, root/'build', lambda stage: None)
                    migrate.assert_not_called()
                    verify.assert_not_called()
            with patch.object(repair, 'repair') as migrate, \
                 patch.object(builder, 'environment_identity', return_value={'version': 'old'}), \
                 patch.object(builder, 'verify', side_effect=ValueError('unknown build')):
                with self.assertRaisesRegex(ValueError, 'unknown build'):
                    launcher.reuse_build(args, root/'build', lambda stage: None)
                migrate.assert_not_called()

    def test_source_failure_stops_before_hardware_or_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launcher.arguments(['--out', str(root/'out'), '--execute'])
            launcher.write(args.out/'batch-state.json', dict(status='launching'))
            with patch.object(launcher, 'REPO', root), patch.object(launcher, 'verify_controllers'), \
                 patch.object(load('sources'), 'worker_source', side_effect=ValueError('bad worker')), \
                 patch.object(launcher, 'hardware_preflight') as hardware:
                with self.assertRaisesRegex(ValueError, 'bad worker'):
                    launcher.supervise(args)
                hardware.assert_not_called()
            self.assertEqual(launcher.read(args.out/'batch-state.json')['stage'], 'worker-source-preflight')

    def test_smoke_prints_underlying_rank_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank in (0, 7):
                launcher.write(root/f'rank{rank}.json', dict(completed=False, error='AttributeError: missing rank'))
            output = io.StringIO()
            with redirect_stderr(output):
                launcher.print_smoke_failures(root)
            self.assertIn('rank0,rank7: AttributeError: missing rank', output.getvalue())
            self.assertEqual(output.getvalue().count('missing rank'), 1)

    def test_shell_preview_has_no_external_work_and_no_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root/'nvidia-smi'
            probe.write_text('#!/bin/sh\nexit 99\n')
            probe.chmod(0o755)
            env = dict(os.environ, RTX5090_PYTHON=sys.executable, PATH=str(root)+':'+os.environ['PATH'])
            output = root/'batch'
            result = subprocess.run(['bash', str(HERE/'run_baseline.sh'), '--out', str(output)],
                                    cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout[result.stdout.index('{'):])
            self.assertEqual(plan['arms'], list(load('protocol').ARMS))
            self.assertEqual((plan['timing_windows'], plan['formal_optimizer_steps']), (162, 8100))
            self.assertEqual(plan['model']['sm_count'], 170)
            self.assertEqual(plan['transport'], load('protocol').TRANSPORT)
            self.assertFalse(plan['peer_access_required'])
            self.assertEqual(plan['transport_environment']['NCCL_P2P_DISABLE'], '1')
            self.assertFalse(output.exists())

    def test_invalid_or_incomplete_schedule_rejected(self):
        for arguments in (['--blocks', '8'], ['--blocks', '0'], ['--report'],
                          ['--report', '--out', '/tmp/x', '--execute']):
            with self.assertRaises(ValueError):
                launcher.arguments(arguments)

    def test_preflight_records_missing_peers_but_still_requires_correct_sku(self):
        devices = [dict(name='NVIDIA GeForce RTX 5090', uuid=str(i), sm_count=170, capability=[12, 0])
                   for i in range(8)]
        peers = [[True]*8 for _ in range(8)]
        preflight.validate_devices(devices, peers)
        peers[7][0] = False
        self.assertEqual(preflight.validate_devices(devices, peers), [(7, 0)])
        peers = [[i == j for j in range(8)] for i in range(8)]
        self.assertEqual(len(preflight.validate_devices(devices, peers)), 56)
        devices[1]['sm_count'] = 132
        with self.assertRaisesRegex(RuntimeError, '170-SM'):
            preflight.validate_devices(devices, peers)

    def test_preflight_preserves_device_memory_evidence_when_later_probe_fails(self):
        observed = []
        def memory(device):
            observed.append(device)
            return (1000+device, 32000)
        cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 8,
            get_device_properties=lambda device: SimpleNamespace(
                name='NVIDIA GeForce RTX 5090', uuid=f'GPU-{device}', major=12, minor=0,
                multi_processor_count=170, total_memory=33000),
            can_device_access_peer=lambda source, destination: source == destination,
            mem_get_info=memory, memory_allocated=lambda device: 16*device,
            memory_reserved=lambda device: 32*device)
        torch = SimpleNamespace(version=SimpleNamespace(cuda='12.9'), cuda=cuda)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'preflight.json'
            log, errors = io.StringIO(), io.StringIO()
            with patch.dict(sys.modules, {'torch': torch}), \
                 patch.object(preflight, 'toolkit', return_value={'version': '12.9'}), \
                 patch.object(preflight.importlib, 'import_module', side_effect=RuntimeError('Apex unavailable')), \
                 redirect_stdout(log), redirect_stderr(errors):
                with self.assertRaisesRegex(RuntimeError, 'Apex unavailable'):
                    preflight.main(['--out', str(output), '--cuda-home', '/CPU-FIXTURE',
                                    '--build-cuda-home', '/CPU-FIXTURE'])
            evidence = json.loads(output.read_text())
        self.assertEqual(observed, list(range(8)))
        self.assertFalse(evidence['completed'])
        self.assertFalse(evidence['environment_probe_passed'])
        self.assertEqual(evidence['error'], 'RuntimeError: Apex unavailable')
        self.assertEqual(json.loads(errors.getvalue()), evidence)
        self.assertEqual(json.loads(log.getvalue())['devices'], evidence['devices'])
        for device, row in enumerate(evidence['devices']):
            self.assertEqual(row['cuda_device'], device)
            self.assertEqual(row['uuid'], f'GPU-{device}')
            self.assertEqual(row['free_memory_bytes'], 1000+device)
            self.assertEqual(row['used_memory_bytes'], 31000-device)
            self.assertEqual(row['cuda_total_memory_bytes'], 32000)
            self.assertEqual(row['total_memory_bytes'], 33000)
            self.assertEqual(row['torch_allocated_bytes'], 16*device)
            self.assertEqual(row['torch_reserved_bytes'], 32*device)

    def test_invalid_driver_memory_observation_is_not_reported_as_capacity(self):
        for free, total in ((-1, 32000), (33000, 32000), (0, 0)):
            torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _: (free, total)))
            with self.subTest(free=free, total=total), self.assertRaisesRegex(RuntimeError, 'memory observation'):
                preflight.memory_state(torch, 0)

    def test_worker_environment_routes_all_nine_arms_and_drops_topology_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launcher.arguments(['--out', str(root/'out')])
            plan = dict(run_id='CPU', cuda_visible_devices='0,1,2,3,4,5,6,7',
                        communication=launcher.capture_communication())
            with patch.dict(os.environ, {'FLUX_FORCE_NVLINK': '1', 'H100_SM_COUNT': '132'}):
                for arm in load('protocol').ARMS:
                    job = dict(arm=arm, mode='timing', trial=1, block=0, index=0)
                    env = launcher.worker_environment(args, root/'build', root/'scripts', job, root/'results', plan)
                    self.assertNotIn('FLUX_FORCE_NVLINK', env)
                    self.assertEqual(env['NCCL_P2P_DISABLE'], '1')
                    self.assertEqual(env['NCCL_SHM_DISABLE'], '0')
                    self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
                    self.assertEqual(env['RTX5090_TRANSPORT'], load('protocol').TRANSPORT)
                    self.assertEqual(env['H100_SM_COUNT'], '170')
                    self.assertEqual(env['E2E_CASE'], 'rtx5090-phase6')
                    self.assertEqual(env['E2E_POLICY'], load('protocol').POLICY[arm])
                    if arm in ('r', 'c', 'a'):
                        self.assertEqual(env['H100_MA2C_ARM'], arm.upper())
                    if arm == 'ma8':
                        self.assertEqual(env['H100_MA8_ARM'], 'F4-cache')
                    if arm == 'ma9':
                        self.assertEqual(env['H100_MA9_ARM'], 'F4-zcount')

    def test_preflight_failure_is_persisted_and_never_starts_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launcher.arguments(['--out', str(root/'out'), '--execute', '--preflight-only'])
            launcher.write(args.out/'batch-state.json', dict(status='launching'))
            with patch.object(launcher, 'REPO', root), patch.object(launcher, 'verify_controllers'), \
                 patch.object(launcher, 'hardware_preflight', side_effect=RuntimeError('NCCL validation failed')):
                with self.assertRaisesRegex(RuntimeError, 'NCCL validation failed'):
                    launcher.supervise(args)
            state = launcher.read(args.out/'batch-state.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stage'], 'hardware-preflight')
            self.assertFalse((args.out/'build').exists())

    def test_non_p2p_communication_overrides_only_required_transport_settings(self):
        with patch.dict(os.environ, {'NCCL_P2P_DISABLE': '0', 'NCCL_SHM_DISABLE': '1',
                                     'NCCL_NVLS_ENABLE': '1', 'NCCL_SOCKET_IFNAME': 'eth0'}):
            record = launcher.capture_communication()
        self.assertEqual(record['environment']['NCCL_SOCKET_IFNAME'], 'eth0')
        for key, value in load('protocol').TRANSPORT_ENV.items():
            self.assertEqual(record['environment'][key], value)

    def test_environment_only_success_cannot_hide_distributed_probe_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = launcher.arguments(['--out', tmp, '--execute', '--preflight-only'])

            def fake_run(command, cwd, env, log, timeout, peer):
                if 'baseline_preflight.py' in str(command[1]):
                    launcher.write(Path(tmp)/'environment-preflight.json', dict(
                        completed=False, environment_probe_passed=True, collective_probe_passed=False))
                else:
                    raise RuntimeError('NCCL worker failed')

            with patch.object(launcher, 'gpu_idle'), patch('tp8_dp2_train.run_process', side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, 'NCCL worker failed'):
                    launcher.hardware_preflight(args)
            evidence = launcher.read(Path(tmp)/'environment-preflight.json')
            self.assertFalse(evidence['completed'])
            self.assertFalse(evidence['collective_probe_passed'])
            self.assertTrue(evidence['environment_probe_passed'])

    def test_missing_mapping_or_profile_evidence_cannot_pass_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(trials=1, blocks=18, build=dict(root=str(root/'build'), manifest_sha256='a'*64))
            launcher.write(root/'mapping-summary.json', dict(completed=True, evidence_files={},
                                                            build_manifest_sha256='a'*64))
            with self.assertRaisesRegex(ValueError, 'mapping checks'):
                launcher.mapping_evidence(root, plan)
            with self.assertRaisesRegex(ValueError, '72 profile traces'):
                launcher.profile_evidence(root, plan, dict(evidence_files={}), recompute=False)


if __name__ == '__main__':
    unittest.main()
