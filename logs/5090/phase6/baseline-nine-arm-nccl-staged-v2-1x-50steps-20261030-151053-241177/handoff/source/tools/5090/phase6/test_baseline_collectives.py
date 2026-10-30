"""CPU-only probe contract tests; fixture records are not GPU evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rtx5090_loader import HERE, load

c = load('collectives')


def fixture():
    environment = dict(python='/CPU-FIXTURE/python', torch='2.8.0a0+CPU-FIXTURE', cuda='12.9', nccl=[2, 27, 3])
    hardware = dict(host='CPU-FIXTURE', devices=[dict(uuid=f'CPU-FIXTURE-{rank}') for rank in range(8)],
                    environment=environment)
    communication = dict(environment=dict(c.REQUIRED_ENV), config_files={}, nvls_request='0')
    rows = [dict(protocol=c.PROTOCOL, run_id='CPU-FIXTURE', completed=True, rank=rank, local_rank=rank,
                 world=8, host=hardware['host'], gpu_uuid=hardware['devices'][rank]['uuid'],
                 environment=environment, communication=communication, iterations=c.ITERATIONS,
                 non_default_stream=True, async_work_waits=c.ITERATIONS*len(c.COLLECTIVES),
                 collectives=dict.fromkeys(c.COLLECTIVES, True),
                 input_split_sizes=[c.packet_bytes(rank, dst) for dst in range(8)],
                 output_split_sizes=[c.packet_bytes(src, rank) for src in range(8)]) for rank in range(8)]
    return rows, 'CPU-FIXTURE', hardware, communication


class Collectives(unittest.TestCase):
    def test_packet_splits_cover_all_directed_links_without_self_transfer(self):
        lengths = [[c.packet_bytes(src, dst) for dst in range(8)] for src in range(8)]
        self.assertEqual(sum(lengths[i][i] for i in range(8)), 0)
        self.assertEqual(sum(value > 0 for row in lengths for value in row), 56)
        self.assertGreater(len({value for row in lengths for value in row if value}), 1)
        self.assertEqual(sum(sum(row) for row in lengths),
                         sum(sum(lengths[src][dst] for src in range(8)) for dst in range(8)))

    def test_complete_probe_records_validate(self):
        args = fixture()
        self.assertEqual(len(c.validate_records(*args)), 8)

    def test_missing_failed_wrong_stream_and_wrong_transport_records_rejected(self):
        args = fixture()
        with self.assertRaises(RuntimeError):
            c.validate_records(args[0][:-1], *args[1:])
        mutations = [lambda row: row.update(completed=False),
                     lambda row: row.update(gpu_uuid='WRONG'),
                     lambda row: row.update(non_default_stream=False),
                     lambda row: row.update(async_work_waits=0),
                     lambda row: row['collectives'].update(all_to_all_bytes=False),
                     lambda row: row['input_split_sizes'].__setitem__(0, 123),
                     lambda row: row['communication']['environment'].update(NCCL_P2P_DISABLE='0')]
        for mutation in mutations:
            rows, run_id, hardware, communication = copy.deepcopy(args)
            mutation(rows[0])
            with self.assertRaises(RuntimeError):
                c.validate_records(rows, run_id, hardware, communication)

    def test_direct_execution_requires_explicit_gate_before_cuda_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            configuration = Path(tmp)/'comm.json'
            configuration.write_text(json.dumps(dict(environment=c.REQUIRED_ENV)))
            env = dict(os.environ)
            env.pop('RTX5090_TRANSPORT_PROBE_EXECUTE', None)
            result = subprocess.run([sys.executable, '-I', '-B', str(HERE/'baseline_collectives.py'),
                                     '--out', tmp, '--run-id', 'CPU-FIXTURE', '--communication', str(configuration)],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires explicit controller', result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
