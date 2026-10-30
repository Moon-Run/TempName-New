"""CPU contracts for the two-node link preflight; synthetic records are not GPU evidence."""
import contextlib
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import check_tp8_dp2_link as link


def fixture():
    tp, dp = link.groups()
    return [dict(protocol=link.PROTOCOL, run_id='cpu-fixture', worker_sha256='a'*64,
        rank=r, node_rank=r//8, local_rank=r % 8, host='synthetic-node-'+str(r//8), pid=10000+r,
        torch='2.6.0+cu124', cuda='12.4', nccl=[2, 21, 5],
        hca=link.DEFAULT_HCA, socket_ifname='bond0', device_count=8,
        gpu_uuid='synthetic-gpu-'+str(r), gpu_name='NVIDIA H100', sm_count=132, capability=[9, 0],
        local_peer_access=True, tp_members=tp[r//8], dp_members=dp[r % 8],
        completed=True, checks=dict(tp_allgather=True, tp_reduce_scatter=True,
                                    dp_allreduce_small=True, dp_allreduce_large=True),
        large_elements=link.LARGE_ELEMENTS, large_iterations=link.ITERATIONS) for r in range(16)]


class LinkTests(unittest.TestCase):
    def test_groups_stay_local_for_tp_and_pair_across_nodes_for_dp(self):
        tp, dp = link.groups()
        self.assertEqual(tp, [list(range(8)), list(range(8, 16))])
        self.assertEqual(dp, [[i, i+8] for i in range(8)])
        self.assertEqual(link.validate_results(fixture()), fixture())

    def test_rank_gpu_node_and_software_drift_are_rejected(self):
        changes = [('gpu_uuid', 'synthetic-gpu-0'), ('host', 'synthetic-node-0'),
                   ('local_rank', 8), ('dp_members', [0, 1]), ('worker_sha256', 'b'*64),
                   ('nccl', [2, 22, 0]), ('local_peer_access', False), ('sm_count', 16)]
        for key, value in changes:
            with self.subTest(key=key):
                rows = fixture()
                rows[8][key] = value
                with self.assertRaises(ValueError):
                    link.validate_metadata(rows)
        with self.assertRaises(ValueError):
            link.validate_metadata(fixture()[:8])

    def test_metadata_alone_cannot_pass_collective_validation(self):
        for field in ('tp_allgather', 'tp_reduce_scatter', 'dp_allreduce_small', 'dp_allreduce_large'):
            rows = fixture()
            rows[15]['checks'][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                link.validate_results(rows)
        rows = fixture()
        rows[8]['large_iterations'] = 1
        with self.assertRaisesRegex(ValueError, 'large-message'):
            link.validate_results(rows)

    def test_ib_selection_is_not_proof_of_actual_ib_channels(self):
        selection = 'NCCL INFO NET/IB : Using [0]mlx5_1:1/IB\nNCCL INFO Using network IB\n'
        self.assertFalse(link.transport_evidence(selection)['ib_channels_observed'])
        actual = selection+'NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA\n'
        result = link.transport_evidence(actual)
        self.assertTrue(result['ib_channels_observed'])
        self.assertTrue(result['gdrdma_channels_observed'])
        self.assertFalse(result['socket_channels_observed'])
        result = link.transport_evidence(actual+'Channel 01 : 1 -> 0 via NET/Socket/0\n')
        self.assertTrue(result['socket_channels_observed'])

    def test_exact_hca_filter_and_ib_only_environment(self):
        args = link.arguments(['--node-rank', '1'])
        env = link.settings(args)
        self.assertEqual(env['NCCL_NET'], 'IB')
        self.assertEqual(env['NCCL_IB_DISABLE'], '0')
        self.assertEqual(env['NCCL_IB_HCA'], '='+link.DEFAULT_HCA)
        self.assertEqual(env['NCCL_SOCKET_IFNAME'], '=bond0')
        for value in ('mlx5_1', 'mlx5_1:0', 'mlx5_1:1,mlx5_1:1', '^mlx5_1:1', 'mlx5_1:1,'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                link.parse_hca(value)

    def test_preview_does_not_create_output_or_launch_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'not-created'
            with patch.object(link, 'execute') as run, patch.object(link, 'local_requirements') as checks, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                link.main(['--node-rank', '0', '--out', str(out)])
            run.assert_not_called()
            checks.assert_not_called()
            self.assertFalse(out.exists())
            self.assertIn('PLAN ONLY', output.getvalue())

    def test_cli_preserves_python_path_and_forwards_custom_deadlines(self):
        args = link.arguments(['--node-rank', '1', '--python', '/training env/bin/python',
                               '--timeout', '300', '--process-timeout', '800'])
        command = link.command(args, Path('/frozen scripts/worker.py'))
        self.assertEqual(command[0], '/training env/bin/python')
        self.assertIn('--nnodes=2', command)
        self.assertIn('--nproc-per-node=8', command)
        self.assertNotIn('--standalone', command)
        forwarded = link.arguments(command[command.index('--worker'):])
        self.assertEqual(forwarded.timeout, 300)
        self.assertEqual(forwarded.process_timeout, 800)
        self.assertEqual(forwarded.node_rank, 1)
        with self.assertRaisesRegex(ValueError, '--python'):
            link.arguments(['--node-rank', '0', '--execute'])

    def test_memlock_is_set_for_child_inheritance_and_failure_is_actionable(self):
        with patch.object(resource, 'getrlimit', side_effect=[(65536, 65536), (-1, -1)]), \
             patch.object(resource, 'setrlimit') as set_limit:
            self.assertEqual(link.ensure_memlock()['after'], [-1, -1])
            set_limit.assert_called_once_with(resource.RLIMIT_MEMLOCK, (-1, -1))
        with patch.object(resource, 'getrlimit', return_value=(65536, 65536)), \
             patch.object(resource, 'setrlimit', side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(RuntimeError, 'ulimit -l unlimited'):
                link.ensure_memlock()

    def _execute_fixture(self, root, transport):
        args = link.arguments(['--node-rank', '0', '--python', sys.executable,
                               '--out', str(root), '--run-id', 'cpu-fixture', '--execute'])

        def fake_popen(command, **kwargs):
            rows = fixture()
            for row in rows:
                row['worker_sha256'] = link.sha(root/'link-worker.py')
            link.write(root/'all-ranks.json', rows)
            for row in rows[:8]:
                link.write(root/f'rank{row["rank"]}.json', row)
                (root/f'nccl.synthetic.{row["pid"]}.log').write_text(transport)
            process = unittest.mock.Mock()
            process.wait.return_value = 0
            return process

        with patch.object(link, 'local_requirements', return_value={'cpu_fixture_only': True}), \
             patch.object(link.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'fixture-head\n', '')), \
             patch.object(link.subprocess, 'Popen', side_effect=fake_popen), \
             contextlib.redirect_stdout(io.StringIO()):
            return link.execute(args)

    def test_controller_requires_actual_ib_evidence_after_collectives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'missing-transport'
            with self.assertRaisesRegex(ValueError, 'IB channel evidence'):
                self._execute_fixture(root, 'NCCL INFO Using network IB\n')
            state = json.loads((root/'state.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertTrue(state['collectives_passed'])
            self.assertFalse(state['ib_transport_validated'])

    def test_controller_finishes_local_validation_and_never_claims_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'valid'
            state = self._execute_fixture(root, 'Channel 00/0 : 0 -> 1 via NET/IB/0/GDRDMA\n')
            self.assertEqual(state['status'], 'completed')
            self.assertTrue(state['ib_transport_validated'])
            self.assertFalse(state['training_started'])
            self.assertEqual(state['local_rank_logs_validated'], 8)
            with self.assertRaises(FileExistsError):
                self._execute_fixture(root, '')


if __name__ == '__main__':
    unittest.main()
