"""CPU-only diagnostics tests: incomplete environments must still yield honest reports."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import probe_tp8_dp2 as probe


class EnvironmentProbeTests(unittest.TestCase):
    def test_missing_command_is_recorded(self):
        result = probe.run_command(['/definitely-not-installed/tp8-dp2-tool'])
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['returncode'])

    def test_timed_out_child_is_terminated(self):
        start = time.monotonic()
        result = probe.run_command([sys.executable, '-I', '-c', 'import time; time.sleep(60)'], timeout=.1)
        self.assertEqual(result['status'], 'timeout')
        self.assertLess(result['returncode'], 0)
        self.assertLess(time.monotonic()-start, 5)

    def test_recovery_paths_are_read_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = root/'outputs/h100/container-recovery/env.sh'
            env.parent.mkdir(parents=True)
            sentinel = root/'must-not-be-created'
            env.write_text('export H100_RECOVERY_PYTHON="/path with spaces/bin/python"\n'
                f'export CUDA_HOME=$(touch {sentinel})\n'
                'export H100_RECOVERY_BUILD_CUDA=/usr/local/cuda-12.8\n')
            self.assertEqual(probe.recovery_values(root), {
                'H100_RECOVERY_PYTHON': '/path with spaces/bin/python',
                'H100_RECOVERY_BUILD_CUDA': '/usr/local/cuda-12.8'})
            self.assertFalse(sentinel.exists())

    def test_virtualenv_python_symlink_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp)/'venv/bin/python'
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            self.assertEqual(probe.python_candidates(python, {}), [str(python)])

    def test_loopback_and_link_local_are_not_candidate_addresses(self):
        value = [{'ifname': 'lo', 'operstate': 'UNKNOWN', 'addr_info': [
            {'local': '127.0.0.1'}, {'local': '::1'}]},
            {'ifname': 'eth0', 'operstate': 'UP', 'addr_info': [
             {'local': 'fe80::1234'}, {'local': '169.254.10.20'},
             {'local': '10.0.0.74', 'family': 'inet', 'scope': 'global', 'prefixlen': 24}]}]
        addresses = probe.network_addresses({'stdout': json.dumps(value)})
        self.assertEqual([v['address'] for v in addresses], ['10.0.0.74'])
        self.assertEqual(addresses[0]['interface'], 'eth0')
        self.assertEqual(probe.network_addresses({'status': 'unavailable'}), [])

    def test_visible_rdma_sysfs_does_not_imply_container_device_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            port = root/'sys/mlx5_0/ports/1'
            port.mkdir(parents=True)
            (port/'state').write_text('4: ACTIVE\n')
            (port/'rate').write_text('200 Gb/sec\n')
            (port/'link_layer').write_text('Ethernet\n')
            result = probe.rdma_devices(root/'sys', root/'missing-dev')
            self.assertEqual(result['device_nodes'], [])
            self.assertEqual(result['sysfs_devices'][0]['ports'][0]['state'], '4: ACTIVE')
            self.assertIn('have not been tested', result['note'])

    def test_cpu_torch_and_mig_are_not_validated_as_eight_full_h100s(self):
        self.assertFalse(any(probe.local_gpu_checks({}).values()))
        info = dict(torch='2.6.0+cu124', cuda_runtime='12.4',
            devices=[dict(name='NVIDIA H100', capability=[9, 0], sm_count=16)]*8,
            local_peer_access=[[True]*8 for _ in range(8)])
        self.assertFalse(probe.local_gpu_checks(info)['eight_full_h100'])
        info['devices'] = [dict(name='NVIDIA H100', capability=[9, 0], sm_count=132)]*8
        self.assertTrue(all(probe.local_gpu_checks(info).values()))
        info['local_peer_access'][1][7] = False
        self.assertFalse(probe.local_gpu_checks(info)['all_local_peer_pairs'])

    def test_cuda_training_environment_is_selected_over_cpu_torch(self):
        cpu = dict(import_ok=True, torch='2.6.0', cuda_runtime=None, cuda_available=False, devices=[])
        gpu = dict(import_ok=True, torch='2.6.0+cu124', cuda_runtime='12.4', cuda_available=True,
            devices=[dict(name='NVIDIA H100', capability=[9, 0], sm_count=132)]*8,
            local_peer_access=[[True]*8 for _ in range(8)])
        records = [dict(status='ok', returncode=0, stdout='a library warning\n'+probe.PREFIX+json.dumps(v))
                   for v in (cpu, gpu)]
        with patch.object(probe, 'run_command', side_effect=records):
            result = probe.probe_python([sys.executable, '/bin/sh'], 1)
        self.assertEqual(result['selected']['path'], '/bin/sh')
        self.assertTrue(all(result['selected']['checks'].values()))

    def test_skip_torch_collects_failures_and_preserves_existing_output(self):
        def unavailable(command, *args, **kwargs):
            return dict(command=command, status='unavailable', error='test fixture', returncode=None)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'probe'
            with patch.object(probe, 'run_command', side_effect=unavailable), \
                 patch.object(probe, 'probe_python') as torch, \
                 patch.object(probe, 'rdma_devices', return_value={'sysfs_devices': [], 'device_nodes': []}), \
                 contextlib.redirect_stdout(io.StringIO()):
                probe.main(['--label', 'gxn74', '--out', str(out), '--skip-torch'])
                torch.assert_not_called()
                saved = (out/'report.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    probe.main(['--label', 'gxn74', '--out', str(out), '--skip-torch'])
                self.assertEqual((out/'report.json').read_bytes(), saved)
            result = json.loads(saved)
            self.assertFalse(result['cross_node_connectivity_tested'])
            self.assertFalse(result['nccl_collectives_tested'])
            self.assertFalse(result['training_started'])
            self.assertIn('查询失败', (out/'summary.txt').read_text())
            self.assertTrue(all('status' not in row['command'] for row in result['commands'].values()))

    def test_wrapper_runs_from_a_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='tp8 dp2 probe ') as temp:
            scripts = Path(temp)/'tools/h100/phase6'
            scripts.mkdir(parents=True)
            for name in ('probe_tp8_dp2.py', 'probe_tp8_dp2.sh'):
                (scripts/name).write_bytes((Path(probe.__file__).parent/name).read_bytes())
            result = subprocess.run(['bash', str(scripts/'probe_tp8_dp2.sh'), '--help'],
                cwd='/tmp', capture_output=True, text=True, timeout=15,
                env=__import__('os').environ | {'TP8_DP2_PROBE_PYTHON': sys.executable})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--label', result.stdout)
            self.assertFalse((Path(temp)/'outputs').exists())


if __name__ == '__main__':
    unittest.main()
