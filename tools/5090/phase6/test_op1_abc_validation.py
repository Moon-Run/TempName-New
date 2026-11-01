"""CPU-only admission regression fixtures; never exported as GPU evidence."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import op1_abc_protocol as p
import op1_abc_sources as sources
import op1_abc_validation as validation
import op1_abc_probe as probe
from test_flux_baseline import fixture as previous_rows
from rtx5090_loader import REPO, load


def state(arm, rank=0, calls=240, offset=0):
    chunks = validation.expected_chunks(arm)
    arrival = arm != p.BASIC_ARMS[0]
    splits = [[0 if peer == rank else c['peer_bytes'] for peer in range(8)] for c in chunks]
    return dict(arm=arm, calls=calls, chunks=chunks, transport='NCCL-byte-AllToAll',
        arrival_trigger='NCCL-work-wait-then-CUDA-event' if arrival else 'whole-exchange',
        selection='last-1/64-H128-groups' if arm == p.BASIC_ARMS[2] else 'all-remote',
        wire_bytes_per_rank=7*sum(c['peer_bytes'] for c in chunks), bf16_wire_bytes_per_rank=14680064,
        encode_calls=calls*len(chunks), exchange_calls=calls*len(chunks), wait_calls=calls*len(chunks),
        decode_calls=calls*len(chunks), join_calls=calls if arrival else 0,
        independent_output=True, fixed_source_order='rank+1-through-rank',
        packet_layout='H128-AoS-136-or-BF16-256', calibration_used=False,
        original_peer_overlap_preserved=False, failed=False, pending_exchange=False,
        device=rank, caller_stream=100, comm_stream=200 if arrival else 100,
        decode_stream=300 if arrival else 100, send_split_bytes=splits, receive_split_bytes=copy.deepcopy(splits),
        tx_ptrs=[offset+1000+i*128 for i in range(len(chunks))],
        rx_ptrs=[offset+2000+i*128 for i in range(len(chunks))])


def fixture(arm, mode='timing'):
    rows, job, plan, worker = previous_rows('flux-improved' if arm == 'flux-improved' else 'flux-basic', mode)
    job['arm'] = arm
    plan['protocol'] = p.PROTOCOL
    worker['selection'] = sources.selection_metadata()
    worker['config'] = sources.configuration(arm, worker['selection'], plan['communication'],
                                            worker['config'].get('codec_library'))
    worker['definition'] = sources.definition(arm)
    for row in rows:
        row.update(protocol=p.PROTOCOL, arm=arm, policy=arm, scope_definition=worker['definition'],
                   role_plan_sha256=worker['config']['role_plan_sha256'],
                   op1_abc_implementation=sources.implementation(arm), op1_abc_transport={})
        row.pop('flux_implementation')
        if arm in p.BASIC_ARMS:
            calls = (row['warmup_steps']+len(row['losses']))*4
            row['codec_identity'] = worker['config']['codec_identity']
            row['op1_abc_transport'] = {name: state(arm, row['rank'], calls, i*10000)
                                       for i, name in enumerate(row['target_modules'])}
            wire_bytes = next(iter(row['op1_abc_transport'].values()))['wire_bytes_per_rank']
            for check in row['selection_checks'].values():
                check['wire_bytes'] = wire_bytes
    return rows, job, plan, worker


def probe_rows(arm):
    rows = []
    for rank in range(8):
        cases = {}
        for case in probe.CASES:
            checks = dict(passed=True, budget_vs_bf16=.05, budget_vs_codec=.01,
                          relative_l2=.001, protocol_relative_l2=.0001, max_abs=.01)
            wire = {direction: {str(peer): dict(bytes=state(arm)['wire_bytes_per_rank']//7,
                                               sha256=f'{min(rank, peer):01x}{max(rank, peer):01x}'+'a'*62)
                                for peer in range(8) if peer != rank}
                    for direction in ('sent', 'received')}
            wire.update(upstream_packet_bitwise=True, upstream_decode_fold_bitwise=True)
            cases[case] = dict(reference=checks, wire=wire, input_sha256='a'*64,
                               weight_sha256='b'*64, output_sha256='c'*64)
        expected = [[((i%8+rank)%8)*2+(i//8)//32, (i//8)%32] for i in range(512)]
        # The last two columns represent the original permutation. Any complete
        # distinct mapping suffices here; the production GPU probe computes it.
        mapping = [pair+[(i+1)//32 % 16, (i+1)%32] for i, pair in enumerate(expected)]
        row = dict(protocol=probe.PROTOCOL, completed=True, rank=rank, arm=arm,
            build_manifest_sha256='a'*64, entrypoint_sha256=probe.sha(probe.__file__),
            reference_sha256=probe.sha(REPO/'tools/a800/phase4/reference.py'),
            probe_stream=100, default_stream=0,
            mapping={role: copy.deepcopy(mapping) for role in probe.ROLES},
            roles={role: dict(k_local=k, cases=copy.deepcopy(cases), retained_outputs=6, reuse_calls=7,
                       retention_passed=True, independent_storage=True, state=state(arm, rank, 10),
                       delayed_reference=copy.deepcopy(checks))
                   for role, k in probe.ROLES.items()})
        rows.append(row)
    return rows


class Evidence(unittest.TestCase):
    def test_all_arms_modes_keep_full_training_admission(self):
        for arm in p.ARMS:
            for mode in ('smoke', 'profile', 'timing'):
                with self.subTest(arm=arm, mode=mode):
                    self.assertEqual(len(validation.validate_rows(*fixture(arm, mode))), 8)

    def test_actual_completion_and_private_storage_are_required(self):
        for arm in p.BASIC_ARMS:
            for mutate in (
                lambda s: s.update(wait_calls=s['wait_calls']-1),
                lambda s: s.update(pending_exchange=True),
                lambda s: s.update(device=7),
                lambda s: s.update(fixed_source_order='arrival-order'),
                lambda s: s.update(encode_calls=0),
                lambda s: s['send_split_bytes'][0].__setitem__(0, 1),
                lambda s: s['rx_ptrs'].__setitem__(0, s['tx_ptrs'][0]),
                lambda s: s.update(wire_bytes_per_rank=14680064),
            ):
                rows, job, plan, worker = fixture(arm)
                mutate(next(iter(rows[0]['op1_abc_transport'].values())))
                with self.subTest(arm=arm, mutation=mutate), self.assertRaises(ValueError):
                    validation.validate_rows(rows, job, plan, worker)
        rows, job, plan, worker = fixture(p.BASIC_ARMS[1])
        values = list(rows[0]['op1_abc_transport'].values())
        values[1]['tx_ptrs'][0] = values[0]['tx_ptrs'][0]
        with self.assertRaisesRegex(ValueError, 'another module'):
            validation.validate_rows(rows, job, plan, worker)

    def test_arrival_requires_distinct_streams_and_selective_budget(self):
        for arm in p.BASIC_ARMS[1:]:
            values = state(arm)
            values['decode_stream'] = values['caller_stream']
            with self.assertRaisesRegex(ValueError, 'stream'):
                validation.validate_codec_state(values, arm, 240, 0)
        values = state(p.BASIC_ARMS[2])
        self.assertEqual(values['wire_bytes_per_rank'], 14572544)
        self.assertEqual(sum(c['count']-c['raw_groups'] for c in values['chunks']), 128)
        values['chunks'][-1]['raw_groups'] = 0
        with self.assertRaisesRegex(ValueError, 'layout'):
            validation.validate_codec_state(values, p.BASIC_ARMS[2], 240, 0)

    def test_frozen_input_and_no_fallback_checks_are_not_bypassed(self):
        for change in ('fallback_calls', 'losses', 'target_modules'):
            rows, job, plan, worker = fixture(p.BASIC_ARMS[2])
            rows[0][change] = 1 if change == 'fallback_calls' else [float('nan')] if change == 'losses' else []
            with self.assertRaises(ValueError):
                validation.validate_rows(rows, job, plan, worker)
        self.assertEqual(load('validation').p.PROTOCOL, load('protocol').PROTOCOL)

    def test_distributed_probe_rejects_corrupt_wire_reference_or_mapping(self):
        for arm in p.BASIC_ARMS:
            rows = probe_rows(arm)
            self.assertTrue(probe.check_basic_rows(rows, arm, 'a'*64)['completed'])
            for mutate in (
                lambda r: r[0]['roles']['attn']['cases']['random']['wire']['received']['1'].update(sha256='f'*64),
                lambda r: r[0]['roles']['mlp']['cases']['zero']['reference'].update(protocol_relative_l2=.011),
                lambda r: r[0]['roles']['mlp']['cases']['random']['wire'].update(upstream_packet_bitwise=False),
                lambda r: r[0]['mapping']['attn'].__setitem__(0, [99, 0, 0, 0]),
                lambda r: r[0]['roles']['attn'].update(retention_passed=False),
            ):
                bad = copy.deepcopy(rows)
                mutate(bad)
                with self.assertRaises(ValueError):
                    probe.check_basic_rows(bad, arm, 'a'*64)

    def test_improved_probe_executes_composite_builder_adapter(self):
        module = probe.improved_probe()
        reached = []
        def verify(root, built):
            reached.append(('verify', root, built))
        def native(root, arm):
            reached.append(('native', root, arm))
            raise RuntimeError('CPU reached isolated composite builder')
        module._abc_builder = lambda: SimpleNamespace(verify=verify, native_root=native)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'op1-abc-build.json').write_text('{}')
            args = SimpleNamespace(build=root, out=root/'probe', arm='c')
            env = dict(WORLD_SIZE='8', LOCAL_WORLD_SIZE='8', LOCAL_RANK='0', NCCL_P2P_DISABLE='1',
                       FLUX_GEMM_RS_GEMM_ONLY='1')
            with patch.dict(os.environ, env), self.assertRaisesRegex(RuntimeError, 'isolated composite builder'):
                module.run(args)
            self.assertEqual(reached, [('verify', root, True), ('native', root, 'c')])
            self.assertFalse(p.read(args.out/'rank0.json')['completed'])


if __name__ == '__main__':
    unittest.main()
