"""CPU-only smoke contract checks; fixture records are not GPU evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


s = module('rtx5090_local_smoke_tested', HERE/'local_smoke.py')
t = module('rtx5090_local_smoke_transport', HERE/'nccl_transport.py')


def shared_scratch_fixture(arm, state, reference):
    count = 1 if arm in ('a', 'r') else 2
    pointers = [[dict(shared={key: 1000+index*10+offset for offset, key in enumerate(s.SHARED_BUFFERS)},
                      packet=2000+instance*10+index, mask=3000+instance*10+index if count == 2 else None,
                      native=4000+instance*10+index)
                 for index in range(count)] for instance in range(2)]

    def lifecycle(instance, calls):
        result = copy.deepcopy(state)
        slots = result.get('slots', [result])
        for index, slot in enumerate(slots):
            expected = calls if count == 1 else (calls+1-index)//2
            slot.update({key: expected for key in ('calls', 'producer_calls', 'pack_calls', 'exchange_calls',
                                                   'wait_calls', 'unpack_calls', 'decode_calls')})
            slot.update(closed=False, consumer_calls=0, initializations=1 if arm in s.FUSED else expected,
                        reconfigure_calls=expected if arm == 'a' else 0, scratch_policy=s.SCRATCH_POLICY,
                        shared_scratch_bytes=123456, shared_scratch_ptrs=pointers[instance][index]['shared'])
        return result

    return dict(passed=True, scratch_policy=s.SCRATCH_POLICY, instances=2, slots_per_instance=count,
                pointers=pointers, calls_before_close=[4, 4], async_burst_calls=6,
                repeated_bitwise=True, retained_outputs=8,
                retained_bitwise=True, output_storage_independent=True, survivor_after_close_bitwise=True,
                reference_checks=[dict(reference), dict(reference)],
                before_close=[lifecycle(0, 4), lifecycle(1, 4)], survivor_after_close=lifecycle(1, 5))


def rows_for(arm):
    """Structural CPU fixture, tagged by its path/UUID; never stored in logs/."""
    rows = []
    for rank in range(8):
        roles = {}
        for role, k in s.ROLES.items():
            one = dict(passed=True, relative_l2=.01, protocol_relative_l2=.001, max_abs=.1,
                       budget_vs_bf16=.05, budget_vs_codec=.01)
            cases = {case: dict(one) for case in (*s.CASES, 'after_delayed_reuse')}
            wire = dict(passed=True, cross_rank_payload_identity=True,
                        pack=dict(passed=True, physical_tiles=448, bytes=7*1234),
                        unpack=dict(passed=True, physical_tiles=448, bytes=7*1234),
                        sent={str(peer): dict(bytes=1234, sha256='0'*64) for peer in range(8) if peer != rank},
                        received={str(peer): dict(bytes=1234, sha256='0'*64) for peer in range(8) if peer != rank})
            count = 12 if arm in s.FUSED else 10
            slots = []
            for index in range(1 if arm in ('a', 'r') else 2):
                calls = count if arm in ('a', 'r') else count//2
                slots.append(dict(protocol=s.TRANSPORT_PROTOCOL, failed=False, peer_memory=False,
                                  pending_exchange=False, native_phase=0, calls=calls, producer_calls=calls,
                                  pack_calls=calls, exchange_calls=calls, wait_calls=calls, unpack_calls=calls,
                                  decode_calls=calls, consumer_calls=int(arm in s.FUSED and index == 1),
                                  initializations=1 if arm in s.FUSED else calls,
                                  reconfigure_calls=calls if arm == 'a' else 0,
                                  send_bytes=7*1234, receive_bytes=7*1234, native_workspace_bytes=4096))
            state = slots[0] if len(slots) == 1 else dict(protocol=s.TRANSPORT_PROTOCOL, failed=False, slots=slots)
            roles[role] = dict(passed=True, k_local=k, cases=cases, wire=wire,
                               consumer=dict(passed=True, bitwise_equal=True, independent_storage=True)
                               if arm in s.FUSED else None,
                               retained_outputs=dict(passed=True, count=6, reuse_calls=7,
                                                     independent_storage=True, unchanged_after_subsequent_calls=True),
                               transport=state, shared_scratch=shared_scratch_fixture(arm, state, one))
        rows.append(dict(protocol=s.PROTOCOL, transport_protocol=s.TRANSPORT_PROTOCOL, completed=True,
                         arm=arm, rank=rank, gpu_uuid=f'CPU-FIXTURE-{rank}', build_manifest_sha256='a'*64, roles=roles))
    return rows


class Smoke(unittest.TestCase):
    def test_preview_does_not_import_torch_create_files_or_require_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run([sys.executable, '-B', str(HERE/'local_smoke.py'),
                                     '--build', str(root/'missing-build'), '--arm', 'ma11', '--out', str(root/'out')],
                                    text=True, capture_output=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertTrue(plan['preview'])
            self.assertFalse(plan['performance_measured'])
            self.assertEqual(plan['roles'], {'attn': 512, 'mlp': 2048})
            self.assertFalse((root/'out').exists())

    def test_gpu_execution_requires_explicit_eight_rank_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env.pop('WORLD_SIZE', None)
            result = subprocess.run([sys.executable, '-B', str(HERE/'local_smoke.py'),
                                     '--build', tmp, '--arm', 'a', '--out', str(Path(tmp)/'out'), '--execute'],
                                    text=True, capture_output=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('eight local ranks', result.stderr)
            self.assertNotIn('No module named', result.stderr)

    def test_cpu_packet_bit_pattern_uses_rows_and_preserves_scale_bits(self):
        bf16, packets = bytearray(8*s.CHUNK_BYTES), bytearray(8*s.PACKET_BYTES)
        peer, tile = 3, 33
        expected_bf16, expected_payload, expected_qs, expected_as = [], [], [], []
        for row in range(128):
            plain = bytes([row, 255-row])*128
            offset = peer*s.CHUNK_BYTES+((128+row)*4096+128)*2
            bf16[offset:offset+256] = plain
            expected_bf16.append(plain)
            payload = bytes([(row*3) % 256])*128
            group = (128+row)*32+1
            base = peer*s.PACKET_BYTES
            packets[base+group*128:base+(group+1)*128] = payload
            expected_payload.append(payload)
            qs, adaptive = struct.pack('<I', 0x80000000+row), struct.pack('<I', 0x7f800001+row)
            packets[base+8192*128+group*4:base+8192*128+(group+1)*4] = qs
            packets[base+8192*132+group*4:base+8192*132+(group+1)*4] = adaptive
            expected_qs.append(qs)
            expected_as.append(adaptive)
        self.assertEqual(s.tile_bytes(bf16, packets, peer, tile, False), b''.join(expected_bf16))
        self.assertEqual(s.tile_bytes(bf16, packets, peer, tile, True),
                         b''.join(expected_payload+expected_qs+expected_as))
        mask = [[0]*512 for _ in range(8)]
        rank = 1
        mask[rank][peer*64+tile] = 1
        layout = t.wire_layout(mask, rank)
        wire = bytearray(layout['tx_bytes'])
        offset = layout['tx_offsets'][peer*64+tile]
        wire[offset:offset+17408] = b''.join(expected_payload+expected_qs+expected_as)
        result = s.check_wire_bytes(bf16, packets, wire, layout['tx_offsets'], layout['tx_splits'], mask, rank)
        self.assertEqual(result['physical_tiles'], 448)
        wire[offset+16384] ^= 1
        with self.assertRaisesRegex(RuntimeError, 'Lossless packet copy mismatch'):
            s.check_wire_bytes(bf16, packets, wire, layout['tx_offsets'], layout['tx_splits'], mask, rank)

    def test_complete_reports_and_corrupt_payload_or_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in s.ARMS:
                rows = rows_for(arm)
                for rank, row in enumerate(rows):
                    s.write(root/f'rank{rank}.json', row)
                result = s.validate_reports(root, arm, 'a'*64)
                self.assertTrue(result['completed'])
                self.assertFalse(result['performance_measured'])
                changed = copy.deepcopy(rows[0])
                changed['roles']['attn']['wire']['received']['1']['sha256'] = 'bad'
                s.write(root/'rank0.json', changed)
                with self.assertRaisesRegex(RuntimeError, 'Cross-rank wire hash'):
                    s.validate_reports(root, arm, 'a'*64)
                changed = copy.deepcopy(rows[0])
                state = changed['roles']['mlp']['transport']
                state.get('slots', [state])[0]['pending_exchange'] = True
                s.write(root/'rank0.json', changed)
                with self.assertRaisesRegex(RuntimeError, 'Unfinished'):
                    s.validate_reports(root, arm, 'a'*64)

    def test_reports_require_cross_instance_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = rows_for('ma11')
            for rank, row in enumerate(rows):
                s.write(root/f'rank{rank}.json', row)
            del rows[0]['roles']['attn']['shared_scratch']
            s.write(root/'rank0.json', rows[0])
            with self.assertRaisesRegex(RuntimeError, 'Missing shared scratch GPU evidence'):
                s.validate_reports(root, 'ma11', 'a'*64)

    def test_cross_instance_evidence_rejects_aliases_stale_outputs_and_incomplete_cleanup(self):
        for arm in s.ARMS:
            record = rows_for(arm)[0]['roles']['attn']['shared_scratch']
            for label in ('not_shared', 'private_aliased', 'output_changed', 'close_invalidated',
                          'pending_survivor', 'missing_call', 'bad_reference'):
                with self.subTest(arm=arm, fault=label):
                    changed = copy.deepcopy(record)
                    if label == 'not_shared':
                        changed['pointers'][1][0]['shared']['tx_bf16'] += 100
                    elif label == 'private_aliased':
                        changed['pointers'][1][0]['packet'] = changed['pointers'][0][0]['packet']
                    elif label == 'output_changed':
                        changed['retained_bitwise'] = False
                    elif label == 'close_invalidated':
                        changed['survivor_after_close_bitwise'] = False
                    elif label == 'bad_reference':
                        changed['reference_checks'][0]['protocol_relative_l2'] = .02
                    else:
                        state = changed['survivor_after_close']
                        slot = state.get('slots', [state])[0]
                        slot['pending_exchange' if label == 'pending_survivor' else 'decode_calls'] = True
                    with self.assertRaises(RuntimeError):
                        s.validate_shared_scratch(changed, arm)


if __name__ == '__main__':
    unittest.main()
