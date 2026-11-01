"""Synthetic OP1 contracts; none of these fixtures are GPU measurements."""
import copy
from contextlib import ExitStack, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import op1_a_native as native
import op1_a_protocol as p
import op1_a_sources as s
import op1_a_validation as v
import op1_a_profile as profile
from test_baseline_workers import fixture as old_rows
from test_baseline_profile import fixture as old_trace, find
from rtx5090_loader import load


def fixture(arm, mode='timing'):
    rows, job, plan, worker = old_rows(s.base_arm(arm), mode)
    job['arm'] = arm
    worker['config'] = s.configuration(arm, worker['selection'], plan['communication'],
                                       worker['config'].get('codec_library'))
    worker['definition'] = s.definition(arm)
    for row in rows:
        row.update(protocol=p.PROTOCOL, arm=arm, scope_definition=worker['definition'],
                   op1_implementation=s.implementation(arm),
                   smoke_output_sha256={}, retained_output_checks={})
        names = row['target_modules']
        if mode == 'smoke':
            row['smoke_output_sha256'] = dict.fromkeys(names, 'a'*64)
            for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
                row['retained_output_checks'][role] = dict(
                    module=next(name for name in names if name.endswith(suffix)), outputs=6,
                    independent_storage=True, unchanged_after_subsequent_call=True, passed=True)
        if arm == 'a':
            for state in row['ma2c_codec_state'].values():
                state.update(owns_output=True, owns_configuration=True)
            for slot in row['rtx5090_transport'].values():
                slot.update(protocol=native.PROTOCOL, native_api=native.NATIVE_API,
                            pack_calls=0, unpack_calls=0, reconfigure_calls=0,
                            configuration_scope='instance', packet_layout=native.PACKET_LAYOUT,
                            output_ownership='independent', gemm_submission_cache=False,
                            scratch_policy=native.SCRATCH_POLICY,
                            shared_scratch_ptrs={key: slot['shared_scratch_ptrs'][key]
                                                 for key in ('tx_wire', 'rx_bf16')},
                            shared_scratch_bytes=7*native.PACKET_BYTES+8*256*4096*2)
                slot['native'].update(protocol_version=2, reconfigure_calls=0)
    return rows, job, plan, worker


def direct_trace():
    events = old_trace('a', te_final=True, memcpy=True, multiple_nccl=True)
    old = load('profile')
    remove = [event for event in events if event['cat'] == 'kernel' and
              (old.PACK_KERNEL in event['name'] or old.UNPACK_KERNEL in event['name'])
              or event['cat'] == 'gpu_memcpy' and 'DtoD' in event['name']]
    correlations = {event['args']['correlation'] for event in remove}
    removed_ids = set(map(id, remove))
    events = [event for event in events if id(event) not in removed_ids
              and not (event['cat'] == 'cuda_runtime' and event['args']['correlation'] in correlations)
              and event['name'] not in (old.CONFIGURE, 'aten::clone')]
    for event in events:
        if event['name'] == old.PRODUCE:
            event['name'] = 'rtx5090.transport.produce_direct'
        elif event['name'] == old.DECODE:
            event['name'] = 'rtx5090.transport.decode_direct'
    return events


def basic_trace(sorted_ctas=False):
    old = load('profile')
    events = old_trace('r')
    remove = [event for event in events if event['cat'] == 'kernel' and old.UNPACK_KERNEL in event['name']]
    correlations = {event['args']['correlation'] for event in remove}
    removed_ids = set(map(id, remove))
    events = [event for event in events if id(event) not in removed_ids
              and not (event['cat'] == 'cuda_runtime' and event['args']['correlation'] in correlations)]
    for event in events:
        name = event['name']
        if name.startswith('scope.'):
            event['name'] = name.replace('.candidate', '.flux_tensor')
        elif name == old.PRODUCE:
            event['name'] = 'ablation6.flux_gemm_only'
        elif name == old.EXCHANGE:
            event['name'] = 'megatron_taco.exchange'
        elif name == old.DECODE:
            event['name'] = 'megatron_taco.decode_reduce'
        elif old.PACK_KERNEL in name:
            event['name'] = 'fused_ash_compress_kernel<__nv_bfloat16, 128>(...)'
        elif 'taco_ma2_decode' in name:
            event['name'] = 'basic_quant_decode_reduce_kernel(...)'
        elif old.is_local_gemm(name):
            event['name'] = ('cutlass::Kernel2<flux_bf16_bf16_void_bf16_fp32_fp32_sm89_rtx5090_'
                             'commnone_rcr_gemmv2_false_nil_64x64x32_16x8x16_streamksk_nil_'
                             '128x128x32_gemmstreamk_3_rasteralongn' +
                             ('_Op1AInterleavedSwizzle' if sorted_ctas else '') + '>')
    for number in range(256):
        events.append(dict(cat='user_annotation', name='megatron_taco.encode', ph='X',
                           ts=number*100+11, dur=4, tid=1, pid=100, args={}))
    return events


class WorkerContracts(unittest.TestCase):
    def test_generated_sources_preserve_frozen_training_and_compile(self):
        for arm in p.ARMS:
            for generator in (s.worker_source, s.taco_source, s.adapter_source):
                compile(generator(arm), arm+'/'+generator.__name__, 'exec')
            s.base.validate_worker(s.worker_source(arm))
        self.assertEqual(s.adapter_source('flux-basic'), s.adapter_source('flux-basic-interleaved'))
        self.assertEqual(s.taco_source('flux-basic'), s.taco_source('flux-basic-interleaved'))
        self.assertEqual(s.implementation('flux-basic')['gemm_schedule'], 'StreamkSK')
        self.assertEqual(s.implementation('flux-basic-interleaved')['gemm_schedule'], 'StreamkSK')
        self.assertFalse(s.implementation('flux-basic')['may_change_streamk_reduction_assignment'])
        self.assertTrue(s.implementation('flux-basic-interleaved')['may_change_streamk_reduction_assignment'])

    def test_all_three_modes_and_arms_have_complete_evidence(self):
        for arm in p.ARMS:
            for mode in ('timing', 'profile', 'smoke'):
                with self.subTest(arm=arm, mode=mode):
                    rows, job, plan, worker = fixture(arm, mode)
                    self.assertEqual(len(v.validate_rows(rows, job, plan, worker)), 8)

    def test_direct_path_rejects_old_stages_aliases_or_missing_wait(self):
        mutations = [lambda row, slot: slot.update(pack_calls=slot['calls']),
                     lambda row, slot: slot.update(unpack_calls=slot['calls']),
                     lambda row, slot: slot.update(reconfigure_calls=slot['calls']),
                     lambda row, slot: slot.update(wait_calls=slot['calls']-1),
                     lambda row, slot: slot.update(gemm_submission_cache=True),
                     lambda row, slot: slot.update(shared_scratch_bytes=0),
                     lambda row, slot: slot['native'].update(protocol_version=1),
                     lambda row, slot: row['ma2c_codec_state'][next(iter(row['ma2c_codec_state']))].update(owns_output=False)]
        for mutate in mutations:
            rows, job, plan, worker = fixture('a')
            mutate(rows[0], next(iter(rows[0]['rtx5090_transport'].values())))
            with self.assertRaises(ValueError):
                v.validate_rows(rows, job, plan, worker)

    def test_ownership_audit_and_projection_hash_are_required(self):
        for arm in p.ARMS:
            for field in ('smoke_output_sha256', 'retained_output_checks'):
                rows, job, plan, worker = fixture(arm, 'smoke')
                rows[0][field] = {}
                with self.subTest(arm=arm, field=field), self.assertRaises(ValueError):
                    v.validate_rows(rows, job, plan, worker)

    def test_optional_equivalence_audit_detects_projection_differences(self):
        a = fixture('flux-basic', 'smoke')[0]
        b = copy.deepcopy(a)
        v.equivalent(a, b, True)
        b[0]['smoke_output_sha256'][next(iter(b[0]['smoke_output_sha256']))] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'output bitwise'):
            v.equivalent(a, b, True)

    def test_original_validator_globals_are_untouched(self):
        self.assertEqual(load('validation').p.PROTOCOL, load('protocol').PROTOCOL)
        self.assertEqual(len(load('validation').p.ARMS), 9)


class DirectProfiles(unittest.TestCase):
    def test_both_basic_arms_require_their_actual_kernel_type(self):
        for arm, sorted_ctas in (('flux-basic', False), ('flux-basic-interleaved', True)):
            result = profile.analyze(basic_trace(sorted_ctas), arm, 0)
            self.assertTrue(result['row_forward']['attn']['swizzle_verified_from_kernel_type'])
            with self.assertRaisesRegex(ValueError, 'kernel swizzle'):
                profile.analyze(basic_trace(not sorted_ctas), arm, 0)

    def test_dp_schedule_cannot_silently_substitute_for_original_sk(self):
        for arm, sorted_ctas in (('flux-basic', False), ('flux-basic-interleaved', True)):
            events = basic_trace(sorted_ctas)
            for event in events:
                if event['cat'] == 'kernel':
                    event['name'] = event['name'].replace('_streamksk_', '_streamkdp_')
            with self.assertRaisesRegex(ValueError, 'original StreamkSK'):
                profile.analyze(events, arm, 0)

    def test_actual_direct_dependencies_and_all_consumers(self):
        result = profile.analyze(direct_trace(), 'a', 0)
        self.assertTrue(result['execution_validated'])
        for role in result['row_forward'].values():
            self.assertEqual(role['packet_pack_kernels'], 0)
            self.assertEqual(role['packet_unpack_kernels'], 0)
            self.assertEqual(role['output_clones'], 0)
            self.assertEqual(role['per_call_reconfigurations'], 0)
            self.assertEqual(role['consumer_add_kernels'], 256)
        self.assertEqual(result['row_forward']['mlp']['layernorm_backends'],
                         {'apex': 124, 'transformer_engine': 4})

    def test_early_exchange_unowned_work_and_wrong_decoder_are_rejected(self):
        mutations = [lambda events: find(events, 'nccl').update(ts=1000003),
                     lambda events: find(events, 'taco_decode_ring').update(ts=1000010),
                     lambda events: find(events, 'taco_decode_ring').update(name='taco_ma2_decode_kernel(...)'),
                     lambda events: find(events, 'taco_decode_ring')['args'].update(stream=99),
                     lambda events: find(events, 'c10d::alltoall', 'cpu_op').update(name='unrelated')]
        for mutate in mutations:
            events = direct_trace()
            mutate(events)
            with self.assertRaises(ValueError):
                profile.analyze(events, 'a', 0)
        events = direct_trace()
        duplicate = copy.deepcopy(find(events, 'cutlass'))
        duplicate['name'] = 'rtx5090_pack_tiles_kernel(...)'
        events.append(duplicate)
        with self.assertRaisesRegex(ValueError, 'only launch'):
            profile.analyze(events, 'a', 0)

    def test_historical_a_trace_cannot_pass_op1(self):
        with self.assertRaises(ValueError):
            profile.analyze(old_trace('a'), 'a', 0)
        self.assertEqual(load('profile').staged_projection_records(old_trace('a'), 'a', 0)
                         ['attn']['output_clones'], 128)


class Integration(unittest.TestCase):
    def fake_build(self, root):
        import op1_a_build as builder
        build, original = root/'build', root/'baseline'
        selection = fixture('a')[3]['selection']
        p.write(original/'selective/selection-plan.json', selection)
        codec = original/'codec-basic/libbasic_quant.so'
        codec.parent.mkdir(parents=True)
        codec.write_bytes(b'CPU-FIXTURE codec, not a native library')
        env = fixture('a')[2]['environment']
        p.write(build/'op1-build.json', dict(baseline_root=str(original), environment=env))
        for arm in p.ARMS:
            native_dir = builder.native_root(build, arm)
            native_dir.mkdir(parents=True, exist_ok=True)
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                (native_dir/name).write_bytes((arm+'/'+name+'/CPU-FIXTURE').encode())
        return build, original, env

    def test_prepare_three_workers_and_actual_environment_routing(self):
        import op1_a as controller
        import op1_a_build as builder
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build, original, env = self.fake_build(root)
            args = controller.arguments(['--out', str(root/'out'), '--op1-build', str(build)])
            plan = dict(protocol=p.PROTOCOL, transport=p.TRANSPORT, environment=env, run_id='CPU',
                        communication=controller.base.capture_communication(),
                        cuda_visible_devices='0,1,2,3,4,5,6,7')
            workers = s.prepare_workers(build, args.out, plan)
            self.assertEqual(set(workers), set(p.ARMS))
            self.assertEqual({path.name for path in (args.out/'workers').iterdir()}, set(p.ARMS))
            for arm in p.ARMS:
                worker = workers[arm]
                scripts = controller.base.check_worker(args.out, arm, worker)
                config = p.read(scripts/'config.json')
                self.assertEqual(config['protocol'], p.PROTOCOL)
                self.assertEqual(config['arm'], arm)
                self.assertEqual((scripts/'build'/p.POLICY[arm]).resolve(), builder.native_root(build, arm))
                self.assertEqual(config['scope_definition'], s.definition(arm))
                self.assertEqual(worker['libraries'], builder.libraries(build, arm))
                job = dict(arm=arm, mode='timing', trial=1, block=0, index=0)
                runtime = controller.worker_environment(args, build, scripts, job, root/'results', plan)
                self.assertEqual(runtime['E2E_POLICY'], p.POLICY[arm])
                self.assertEqual(runtime['E2E_CASE'], 'rtx5090-phase6')
                self.assertEqual(runtime['E2E_BUILD'], str(scripts/'build'))
                self.assertIn(str(builder.native_root(build, arm)/'python'), runtime['PYTHONPATH'].split(':'))
                self.assertIn(str(original/'megatron'), runtime['PYTHONPATH'].split(':'))
                self.assertEqual(runtime['RTX5090_BASELINE_EXECUTE'], '1')
                self.assertEqual(runtime['RTX5090_OP1_EXECUTE'], '1')
                for key, value in p.TRANSPORT_ENV.items():
                    self.assertEqual(runtime[key], value)
            old = controller.worker_environment(args, build, args.out/'workers/a/scripts',
                dict(arm='a', mode='smoke', trial=0, block=-1, index=-1), root/'old-a', plan, 'old-a')
            self.assertIn(str(load('build').native_root(original, 'a')/'python'), old['PYTHONPATH'].split(':'))

    def test_supervisor_runs_all_gates_before_balanced_timing(self):
        import op1_a as controller
        import op1_a_build as builder
        import tp8_dp2_train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build, original, environment = self.fake_build(root)
            args = controller.arguments(['--out', str(root/'out'), '--op1-build', str(build), '--foreground'])
            p.write(args.out/'batch-state.json', dict(protocol=p.PROTOCOL, status='CPU-FIXTURE'))
            communication = controller.base.capture_communication()
            hardware = dict(completed=True, host='CPU-FIXTURE', environment=environment,
                devices=fixture('a')[0][0]['hardware']['devices'], communication=communication,
                cuda_visible_devices='0,1,2,3,4,5,6,7')
            order = []

            def hardware_preflight(_args):
                p.write(args.out/'environment-preflight.json', hardware)
                order.append('hardware')
                return hardware

            def run_process(command, cwd, runtime, log, timeout, peer):
                dest = Path(runtime['E2E_PROCESS_OUT'])
                job = p.read(dest/'command.json')['job']
                order.append(job['mode'] + ':' + job['arm'])
                plan = p.read(args.out/'plan.json')
                workers = p.read(args.out/'workers.json')
                worker = workers[job['arm']]
                rows = fixture(job['arm'], job['mode'])[0]
                for rank, row in enumerate(rows):
                    row.update(run_id=plan['run_id'], trial=job['trial'], block=job['block'],
                        window=job['index'], worker_config_sha256=worker['files']['config.json'],
                        communication=communication, scope_definition=worker['definition'],
                        role_plan_sha256=worker['config']['role_plan_sha256'], **worker['libraries'])
                    if job['arm'] in s.BASIC_ARMS:
                        row.update(codec_identity=worker['config']['codec_identity'],
                                   codec_library=worker['config']['codec_library'])
                        # The fixture's native codec self-test record carries library identity.
                        # Its separately audited verification is mocked below; all row gates run.
                    p.write(dest/f'rank{rank}.json', row)
                    if job['mode'] == 'profile':
                        with gzip.open(dest/f'rank{rank}-profile.json.gz', 'wt') as handle:
                            json.dump(dict(traceEvents=[]), handle)
                Path(log).write_text('CPU-FIXTURE: no worker or GPU was launched\n')

            def analyze(events, arm, rank):
                return dict(protocol=p.PROTOCOL, arm=arm, rank=rank, execution_validated=True,
                            row_forward={'CPU-FIXTURE': True})

            def replay(directory, recompute_profiles):
                self.assertFalse(recompute_profiles)
                return p.report(p.read(directory/'samples.json'), args.trials, args.blocks)

            with ExitStack() as stack:
                stack.enter_context(patch.object(controller, 'REPO', root/'repo'))
                stack.enter_context(patch.object(controller.base, 'verify_controllers'))
                stack.enter_context(patch.object(controller.base, 'gpu_idle'))
                stack.enter_context(patch.object(controller.base, 'hardware_preflight', side_effect=hardware_preflight))
                stack.enter_context(patch.object(builder, 'verify'))
                stack.enter_context(patch.object(load('build'), 'environment_identity', return_value=environment))
                stack.enter_context(patch.object(controller, 'native_checks', side_effect=lambda *a: order.append('native')))
                stack.enter_context(patch.object(tp8_dp2_train, 'run_process', side_effect=run_process))
                stack.enter_context(patch.object(profile, 'analyze', side_effect=analyze))
                stack.enter_context(patch.object(v.shared.basic_quant_validation, 'verify'))
                stack.enter_context(patch.object(controller, 'replay', side_effect=replay))
                stack.enter_context(redirect_stdout(io.StringIO()))
                controller.supervise(args)
            state = p.read(args.out/'batch-state.json')
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['timing_windows'], 18)
            self.assertEqual(state['formal_optimizer_steps'], 900)
            self.assertEqual(order[:2], ['hardware', 'native'])
            self.assertEqual(order[2:5], ['smoke:'+arm for arm in p.ARMS])
            self.assertEqual(order[5:8], ['profile:'+arm for arm in p.ARMS])
            self.assertEqual(order[8:], ['timing:'+job['arm'] for job in p.schedule()])
            self.assertEqual(p.read(args.out/'preflight.json'),
                             dict(completed=True, arms=dict.fromkeys(p.ARMS, True)))


if __name__ == '__main__':
    unittest.main()
