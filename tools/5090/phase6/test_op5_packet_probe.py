"""CPU checks for the real packet loop and mandatory OP5 v2 probe evidence."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import op5_probe as probe


class Tensor:
    def __init__(self, values, *, storage=None, start=0, stop=None):
        self.storage = list(values) if storage is None else storage
        self.start = start
        self.stop = len(self.storage) if stop is None else stop

    @property
    def values(self):
        return self.storage[self.start:self.stop]

    def __getitem__(self, item):
        return Tensor([], storage=self.storage, start=self.start+item.start, stop=self.start+item.stop)

    def __mul__(self, other):
        return Tensor(x*other for x in self.values)

    def __add__(self, other):
        return Tensor(a+b for a, b in zip(self.values, other.values)) if isinstance(other, Tensor) else Tensor(x+other for x in self.values)

    def __floordiv__(self, other):
        return Tensor(x//other for x in self.values)

    def __mod__(self, other):
        return Tensor(x % other for x in self.values)

    def to(self, dtype):
        return self

    def copy_(self, other):
        if self.stop-self.start != len(other.values):
            raise AssertionError('copy shape changed')
        self.storage[self.start:self.stop] = other.values

    def fill_(self, value):
        self.storage[self.start:self.stop] = [value]*(self.stop-self.start)


class Torch:
    uint8 = 'uint8'

    @staticmethod
    def equal(a, b):
        return a.values == b.values


class Distributed:
    def __init__(self, rank, index, bad=None):
        self.rank, self.index, self.bad = rank, index, bad
        self.calls, self.waits = [], 0

    def all_to_all_single(self, destination, source, **kwargs):
        epoch = len(self.calls)
        splits = probe.packet_splits(self.rank)
        if kwargs != dict(output_split_sizes=splits, input_split_sizes=splits, async_op=True):
            raise AssertionError('Wrong All-to-All invocation')
        peers = [peer for peer in range(8) if peer != self.rank]
        count = probe.SETTINGS['alltoall']['peer_bytes']
        for slot, peer in enumerate(peers):
            actual = source[slot*count:(slot+1)*count]
            expected = probe.packet_pattern(Torch, self.index, self.rank, peer, epoch)
            if not Torch.equal(actual, expected):
                raise AssertionError('Wrong send routing/pattern')
        self.calls.append((id(source), id(destination), kwargs))
        if self.bad == 'missing-work':
            return None
        owner = self
        class Work:
            def wait(self):
                owner.waits += 1
                for slot, peer in enumerate(peers):
                    actual_epoch = 0 if owner.bad == 'stale' else epoch
                    actual_peer = owner.rank if owner.bad == 'wrong-peer' else peer
                    expected = probe.packet_pattern(Torch, owner.index, actual_peer, owner.rank, actual_epoch)
                    destination[slot*count:(slot+1)*count].copy_(expected)
                if owner.bad == 'partial':
                    destination.storage[-1] = 255
        return Work()


def small_settings():
    value = copy.deepcopy(probe.SETTINGS)
    value['alltoall'].update(peer_bytes=16, wire_bytes=112)
    return value


def complete_record(rank):
    peers = [peer for peer in range(8) if peer != rank]
    return dict(settings=dict(probe.SETTINGS['alltoall']), input_split_sizes=probe.packet_splits(rank),
        output_split_sizes=probe.packet_splits(rank), iterations=probe.SETTINGS['correctness_iterations'],
        verified_peers_per_iteration=[peers]*probe.SETTINGS['correctness_iterations'],
        bitwise_correct=True, non_default_stream=True, reused_buffers=True, delayed_rank=7,
        async_work_waits=sum(probe.SETTINGS[k] for k in ('correctness_iterations', 'warmup', 'timed_calls', 'profile_calls')),
        profile_complete=True, gpu_us=300., wall_us=310.)


def connection_log(all_pairs=True):
    return '\n'.join(f'host:1:2 [{source}] NCCL INFO Channel {channel:02d} : '
        f'{source}[{source}] -> {dest}[{dest}] via SHM/direct/CE'
        for channel in range(16) for source in range(8) for dest in range(8)
        if source != dest and (all_pairs or dest == (source+1)%8))


def packet_trace(rank):
    return [dict(cat='kernel', name='ncclDevKernel_SendRecv', ts=100*i, dur=50,
        args={'device': rank, 'dtype': 'Byte', 'Group size': 8, 'Collective name': 'all_to_allv',
              'In msg nelems': 7798784, 'Out msg nelems': 7798784,
              'In split size': json.dumps(probe.packet_splits(rank)),
              'Out split size': json.dumps(probe.packet_splits(rank))})
            for i in range(probe.SETTINGS['profile_calls'])]


class PacketLoop(unittest.TestCase):
    def test_real_shape_matches_failed_native_packet(self):
        self.assertEqual(probe.SETTINGS['alltoall']['peer_bytes'], 1114112)
        self.assertEqual(probe.SETTINGS['alltoall']['wire_bytes'], 7798784)
        for rank in range(8):
            splits = probe.packet_splits(rank)
            self.assertEqual(sum(splits), 7798784)
            self.assertEqual(splits[rank], 0)
            self.assertEqual(sum(size != 0 for size in splits), 7)

    def test_all_ranks_check_each_direction_and_wait_before_reading_reused_buffers(self):
        with patch.object(probe, 'SETTINGS', small_settings()):
            for rank in range(8):
                index, source, dest = Tensor(range(16)), Tensor([0]*112), Tensor([0]*112)
                dist, stages, delays = Distributed(rank, index), [], []
                exchange, waits, record = probe.packet_check(Torch, dist, rank, index, source, dest, stages.append, sleep=delays.append)
                self.assertEqual(waits(), 3)
                self.assertEqual(dist.waits, 3)
                self.assertEqual(len({(a, b) for a, b, _ in dist.calls}), 1)
                self.assertEqual(record['verified_peers_per_iteration'], [[r for r in range(8) if r != rank]]*3)
                self.assertEqual(delays, [.01] if rank == 7 else [])
                self.assertEqual(stages, [f'alltoall-correctness-{i}' for i in range(3)])

    def test_missing_wait_result_stale_routes_and_partial_copy_fail(self):
        with patch.object(probe, 'SETTINGS', small_settings()):
            for bad in ('missing-work', 'stale', 'wrong-peer', 'partial'):
                index, source, dest = Tensor(range(16)), Tensor([0]*112), Tensor([0]*112)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    probe.packet_check(Torch, Distributed(0, index, bad), 0, index, source, dest, lambda stage: None)

    def test_records_cannot_omit_real_size_coverage_waits_or_profile(self):
        for rank in range(8):
            probe.check_packet_record(complete_record(rank), rank)
        for key, value in (('settings', {}), ('input_split_sizes', [0]*8), ('iterations', 2),
                           ('verified_peers_per_iteration', [[1]]*3), ('async_work_waits', 3),
                           ('profile_complete', False), ('bitwise_correct', False), ('gpu_us', float('nan'))):
            bad = complete_record(0)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.check_packet_record(bad, 0)
        with self.assertRaises(ValueError):
            probe.check_packet_record(None, 0)

    def test_ring_connections_cannot_prove_all_to_all_coverage(self):
        self.assertEqual(probe.transport_log(connection_log(), require_alltoall=True)['directed_peer_pairs'], 56)
        with self.assertRaisesRegex(ValueError, 'All-to-All'):
            probe.transport_log(connection_log(False), require_alltoall=True)

    def test_profile_requires_actual_native_byte_payload_and_zero_self_split(self):
        self.assertEqual(probe.packet_profile_evidence(packet_trace(0), 0)['wire_bytes'], 7798784)
        for field, value in (('dtype', 'BFloat16'), ('In msg nelems', 256),
                             ('Out split size', '[1, 1, 1, 1, 1, 1, 1, 1]')):
            events = packet_trace(0)
            events[0]['args'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'payload/splits'):
                probe.packet_profile_evidence(events, 0)


class Evidence(unittest.TestCase):
    def fixture(self, root):
        environment = dict(torch='fixture', cuda='12.9', nccl=[2, 27, 3], python='/cpu/python')
        communication = dict(environment=dict(probe.OVERRIDES, NCCL_DEBUG='WARN'), config_files={})
        hardware = dict(completed=True, host='cpu-fixture', environment=environment,
            communication=communication, cuda_visible_devices='0,1,2,3,4,5,6,7',
            devices=[dict(uuid=str(rank), name='RTX 5090', capability=[12, 0], sm_count=170, cuda_device=rank) for rank in range(8)])
        probe.p.write(root/'environment-preflight.json', hardware)
        probe.p.write(root/'preflight-communication.json', communication)
        env = dict(communication['environment'], **probe.DEBUG)
        probe.p.write(root/'command.json', dict(protocol=probe.PROTOCOL, run_id='fixture', settings=probe.SETTINGS, communication_environment=env))
        (root/'process.log').write_text(connection_log())
        for rank in range(8):
            record = dict(protocol=probe.PROTOCOL, run_id='fixture', rank=rank, completed=True,
                settings=probe.SETTINGS, gpu_uuid=str(rank), host='cpu-fixture', environment=environment,
                communication_environment=env, config_files={}, bitwise_correct=True, non_default_stream=True,
                iterations=3, profile_complete=True, gpu_us=100., wall_us=110., alltoall=complete_record(rank))
            probe.p.write(root/f'rank{rank}.json', record)
            for suffix in ('profile', 'alltoall-profile'):
                with gzip.open(root/f'rank{rank}-{suffix}.json.gz', 'wt') as stream:
                    json.dump({'traceEvents': packet_trace(rank) if suffix == 'alltoall-profile' else []}, stream)

    def test_packet_files_and_all_eight_records_are_mandatory_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe.op3.base, 'collective_evidence'), \
                patch.object(probe, 'copy_evidence', return_value={'gather': True}), \
                patch.object(probe, 'sendrecv_evidence', return_value={'sendrecv': True}):
            root = Path(tmp)
            self.fixture(root)
            result = probe.evidence(root)
            self.assertEqual(result['alltoall']['directed_links'], 56)
            self.assertEqual(len(result['alltoall']['profiles']), 8)
            self.assertEqual(result['alltoall']['gpu_us'], 300.)
            for rank in range(8):
                name = f'rank{rank}-alltoall-profile.json.gz'
                self.assertEqual(result['evidence_files'][name], probe.p.sha(root/name))
            record = probe.p.read(root/'rank7.json')
            old = record.pop('alltoall')
            probe.p.write(root/'rank7.json', record)
            with self.assertRaisesRegex(ValueError, 'All-to-All evidence'):
                probe.evidence(root)
            record['alltoall'] = old
            probe.p.write(root/'rank7.json', record)
            (root/'rank7-alltoall-profile.json.gz').unlink()
            with self.assertRaises(FileNotFoundError):
                probe.evidence(root)


if __name__ == '__main__':
    unittest.main()
