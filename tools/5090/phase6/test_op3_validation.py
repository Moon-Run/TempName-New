"""CPU admission regressions for the distinct OP3 transport and consumer paths."""
import copy
import itertools
import unittest

from rtx5090_loader import load
import op3_protocol as p
import op3_sources as sources
import op3_validation as validation
import op2_abc_build as optimizers
from test_flux_baseline import fixture as basic_fixture
from test_baseline_workers import fixture as model_fixture


def state(arm, rank=0, calls=240, offset=0, fused_calls=None):
    if fused_calls is None:
        fused_calls = calls if arm == 'op3-r' else 0
    splits = [[0 if peer == rank else 8192*136 for peer in range(8)]]
    return dict(protocol=validation.CODEC_PROTOCOL, arm=arm, calls=calls, chunks=copy.deepcopy(validation.CHUNKS),
        transport='NCCL-byte-AllToAll', arrival_trigger='whole-exchange', selection='all-remote',
        wire_bytes_per_rank=7798784, bf16_wire_bytes_per_rank=14680064, encode_calls=calls,
        exchange_calls=calls, decode_calls=calls, wait_calls=calls, join_calls=0,
        fused_consumer_calls=fused_calls, projection_calls=calls-fused_calls, independent_output=True,
        fixed_source_order='rank+1-through-rank', pending_exchange=False, device=rank,
        caller_stream=100, comm_stream=100, decode_stream=100,
        send_split_bytes=splits, receive_split_bytes=copy.deepcopy(splits),
        tx_ptrs=[offset+1024], rx_ptrs=[offset+2048], packet_layout='H128-AoS-136', calibration_used=False,
        encoder='upstream-cta128' if arm == 'op3-r' else 'warp-owned-four-groups',
        decoder='r-packed-source-prefetch' if arm == 'op3-r' else 'basic-ring-bf16-h128-direct-v1',
        consumer_fusion=arm == 'op3-r', original_peer_overlap_preserved=False, failed=False)


def fixture(arm, mode='timing'):
    rows, job, plan, worker = basic_fixture('flux-basic', mode)
    shared_rows = model_fixture('ma11', mode)[0]
    job['arm'] = arm
    plan.update(protocol=p.PROTOCOL, ma9_compat=dict(library='/CPU/ma9.so', library_sha256='a'*64),
                ma11_compat=dict(library='/CPU/ma11.so', library_sha256='b'*64))
    worker['selection'], worker['definition'] = sources.selection_metadata(), sources.definition(arm)
    worker['config'] = sources.configuration(arm, worker['selection'], plan['communication'],
                                            worker['config']['codec_library'], plan)
    for row, shared in zip(rows, shared_rows):
        row.pop('flux_implementation')
        row.update(protocol=p.PROTOCOL, arm=arm, policy=arm, scope_definition=worker['definition'],
            role_plan_sha256=worker['config']['role_plan_sha256'], codec_identity=worker['config']['codec_identity'],
            op3_implementation=sources.implementation(arm), op3_optimization_stack=copy.deepcopy(sources.STACK),
            op3_optimization_libraries=optimizers.libraries(plan), op3_consumer_state={},
            **{field: copy.deepcopy(shared[field]) for field in validation.FIELDS})
        calls = (row['warmup_steps']+len(row['losses']))*4
        row['op3_transport'] = {name: state(arm, row['rank'], calls, i*10000)
                                for i, name in enumerate(row['target_modules'])}
        if arm == 'op3-r':
            row['op3_consumer_state'] = {name: dict(projection_calls=calls, bda_calls=calls,
                bias_parameter_id=i+100, native=copy.deepcopy(row['op3_transport'][name]))
                for i, name in enumerate(row['target_modules'])}
    return rows, job, plan, worker


class CodecAdmission(unittest.TestCase):
    def test_probe_raw_and_fused_counters_use_their_actual_boundaries(self):
        validation.validate_codec_state(state('op3-r', calls=10, fused_calls=0), 'op3-r', 10, 0)
        validation.validate_codec_state(state('op3-r', calls=13, fused_calls=7), 'op3-r', 13, 0, fused_calls=7)
        validation.validate_codec_state(state('op3-improved', calls=10), 'op3-improved', 10, 0)
        with self.assertRaises(ValueError):
            validation.validate_codec_state(state('op3-r', calls=13, fused_calls=7), 'op3-r', 13, 0)

    def test_rejects_wrong_algorithm_packet_order_and_completion(self):
        for arm in p.ARMS:
            for mutate in (
                lambda s: s.update(encoder='old-encoder'),
                lambda s: s.update(decoder='other-decoder'),
                lambda s: s.update(selection='last-1/64-H128-groups'),
                lambda s: s.update(fixed_source_order='arrival-order'),
                lambda s: s.update(wait_calls=0),
                lambda s: s.update(pending_exchange=True),
                lambda s: s.update(decode_stream=101),
                lambda s: s.update(join_calls=True),
                lambda s: s.update(consumer_fusion=arm != 'op3-r'),
                lambda s: s['tx_ptrs'].__setitem__(0, s['rx_ptrs'][0]),
                lambda s: s['receive_split_bytes'][0].__setitem__(0, 8192*136),
                lambda s: s['chunks'][0].update(raw_groups=1),
            ):
                sample = state(arm)
                mutate(sample)
                with self.subTest(arm=arm, mutation=mutate), self.assertRaises(ValueError):
                    validation.validate_codec_state(sample, arm, 240, 0, fused_calls=240 if arm == 'op3-r' else 0)


class ModelAdmission(unittest.TestCase):
    def test_two_arms_all_modes_retain_full_model_admission(self):
        for arm, mode in itertools.product(p.ARMS, ('smoke', 'profile', 'timing')):
            with self.subTest(arm=arm, mode=mode):
                self.assertEqual(len(validation.validate_rows(*fixture(arm, mode))), 8)
        self.assertEqual(load('validation').p.PROTOCOL, load('protocol').PROTOCOL)

    def test_r_consumers_must_match_each_native_instance_and_bda(self):
        for mutate in (
            lambda row: row.update(op3_consumer_state={}),
            lambda row: next(iter(row['op3_consumer_state'].values())).update(bda_calls=0),
            lambda row: next(iter(row['op3_consumer_state'].values())).update(native={}),
            lambda row: next(iter(row['op3_transport'].values())).update(fused_consumer_calls=0),
            lambda row: list(row['op3_consumer_state'].values())[1].update(
                bias_parameter_id=list(row['op3_consumer_state'].values())[0]['bias_parameter_id']),
        ):
            rows, job, plan, worker = fixture('op3-r')
            mutate(rows[0])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validation.validate_rows(rows, job, plan, worker)
        rows, job, plan, worker = fixture('op3-improved')
        rows[0]['op3_consumer_state'] = {'unexpected': {}}
        with self.assertRaisesRegex(ValueError, 'unexpectedly enabled'):
            validation.validate_rows(rows, job, plan, worker)

    def test_packet_buffers_are_private_across_all_64_modules(self):
        rows, job, plan, worker = fixture('op3-improved')
        values = list(rows[0]['op3_transport'].values())
        values[1]['tx_ptrs'] = values[0]['tx_ptrs'][:]
        with self.assertRaisesRegex(ValueError, 'another module'):
            validation.validate_rows(rows, job, plan, worker)
        rows, job, plan, worker = fixture('op3-r')
        rows[0]['op3_transport'] = {}
        with self.assertRaisesRegex(ValueError, 'transport modules'):
            validation.validate_rows(rows, job, plan, worker)

    def test_shared_optimizations_and_loaded_libraries_are_required(self):
        for arm in p.ARMS:
            for mutate in (
                lambda row: row['ma11_activation_setup'].update(installed=False),
                lambda row: row['ma11_zero_count_state'].update(calls=1),
                lambda row: row['ma11_softmax_state'].update(forward_audits=0),
                lambda row: row['op3_optimization_libraries']['ma9'].update(sha256='f'*64),
                lambda row: row.update(op3_optimization_stack={}),
                lambda row: row.pop('ma11_softmax_state'),
            ):
                rows, job, plan, worker = fixture(arm, 'smoke')
                mutate(rows[0])
                with self.subTest(arm=arm, mutation=mutate), self.assertRaises(ValueError):
                    validation.validate_rows(rows, job, plan, worker)

    def test_base_workload_numerical_and_output_gates_remain_active(self):
        for mutate in (
            lambda row: row.update(fallback_calls=1),
            lambda row: row['losses'].__setitem__(0, float('nan')),
            lambda row: row.update(target_modules=[]),
            lambda row: row.update(parameter_trajectory=[]),
            lambda row: row['retained_output_checks']['attn'].update(outputs=5),
            lambda row: row['selection_checks'][next(iter(row['selection_checks']))].update(wire_bytes=14680064),
            lambda row: row.update(smoke_output_sha256={}),
        ):
            rows, job, plan, worker = fixture('op3-r', 'smoke')
            mutate(rows[0])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validation.validate_rows(rows, job, plan, worker)


if __name__ == '__main__':
    unittest.main()
