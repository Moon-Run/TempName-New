"""CPU-only contract checks; synthetic measurements remain in temporary folders."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ma2_bench as bench
import ma2_fullstep as full


def fixture_plan():
    return dict(protocol=full.PROTOCOL, run_id='CPU-only-fixture',
        trial_ids={'1': 'trial1', '2': 'trial2'}, trials=2, blocks=12,
        model=full.MODEL, roles=bench.ROLES, arms=list(bench.ARMS), schedule=full.expected_schedule(2, 12),
        warmup_steps=10, timed_steps=50, source_files={}, shared_environment=full.SHARED_ENVIRONMENT,
        communication={'NCCL_NVLS_ENABLE': '0', 'NCCL_DEBUG': 'WARN'},
        communication_record={'environment': {'NCCL_NVLS_ENABLE': '0', 'NCCL_DEBUG': 'WARN'},
                              'nvls_request': '0', 'config_files': {}},
        target_percent=full.TARGET_PERCENT, acceptance_rule=full.ACCEPTANCE_RULE,
        libraries={arm: {'files': {
            f'/synthetic/{arm}/{name}': arm+name for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}}
            for arm in bench.ARMS})


def fixture_rows(plan, job, plan_hash):
    arm = job['arm']
    count = 1 if job['mode'] == 'profile' else 3 if job['mode'] == 'smoke' else 50
    shapes = {f'decoder.layers.{layer}.{suffix}': dict(M=2048, N=4096, K_local=k, K_global=k*8)
              for layer in range(32) for suffix, k in (('self_attention.linear_proj', 512), ('mlp.linear_fc2', 2048))}
    modules = list(shapes)
    rows = []
    for rank in range(8):
        row = dict(completed=True, passed=True, rank=rank, local_rank=rank, protocol=full.PROTOCOL,
            policy=full.POLICY, mode=job['mode'], ma2_arm=arm, world_size=8,
            run_id=plan['trial_ids'].get(str(job['trial']), plan['run_id']), ma2_plan_sha256=plan_hash,
            ma2_shared_environment=plan['shared_environment'], ma2_protocol=bench.SOURCE_PROTOCOL,
            tp_rank=rank, dp_rank=0, tp_group_ranks=list(range(8)), dp_group_ranks=[rank],
            block=job['block'], window=job['index'], scope_definition=full.routing.definition(full.POLICY),
            role_plan_sha256={full.POLICY: {'attn': None, 'mlp': None}},
            ma2_codec_state={name: dict(owns_output=False, workspaces=1) for name in modules},
            workspace_checks={name: [dict(packet_ptr=1000+i, shape=full.routing.SHAPES[full.routing.module_role(name)])]
                              for i, name in enumerate(modules)},
            communication=plan['communication_record'], sampling=False, profiling=job['mode'] == 'profile',
            fallback_calls=0, skips=0, timing_protocol='full-step-one-clear-v2',
            warmup_steps=2 if job['mode'] == 'smoke' else 10, losses=[1.]*count, grad_norms=[.1]*count,
            tokens_per_optimizer_step=8192, padded_vocab_size=33792, target_modules=modules,
            replaced_modules=64, independent_scope_instances=64,
            timed_calls={name: count*4 for name in modules}, post_audit_routes={name: 'candidate' for name in modules},
            warmup_calls={name: (2 if job['mode'] == 'smoke' else 10)*4 for name in modules},
            module_policies={name: 'candidate' for name in modules}, enabled_modules=sorted(modules),
            forward_preserved={name: False for name in modules}, codec_placements={name: 'fused' for name in modules},
            model=dict(name='h100-phase6', **{k: v for k, v in full.MODEL.items() if k != 'sm_count'}),
            observed_shapes=shapes, gpu_uuid=str(rank),
            hardware={'host': 'CPU-fixture', 'local_rank': rank, 'devices': [dict(name='H100 synthetic', uuid=str(i),
                       sm_count=132, capability=[9, 0]) for i in range(8)]},
            torch='synthetic', cuda='synthetic', nccl=[0, 0], python='synthetic',
            initial_parameters_sha256=str(rank), initial_rng_sha256=str(rank), tokens_sha256='synthetic',
            timed_steps=count, elapsed_seconds={'A': .5, 'P': .49, 'V': .48}[arm]*count,
            forward_checks={name: dict(passed=True, relative_l2=.025, protocol_relative_l2=.0002,
                           budget_vs_bf16=.05, budget_vs_codec=.01) for name in modules})
        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            row[name] = dict(path=f'/synthetic/{arm}/{name}', sha256=arm+name)
        rows.append(row)
    return rows


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ma2-fullstep-CPU-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = fixture_plan()
        bench.write(self.root/'plan.json', self.plan)
        self.job = dict(self.plan['schedule'][0], mode='timing')
        self.rows = fixture_rows(self.plan, self.job, bench.sha(self.root/'plan.json'))

    def window(self):
        with patch.object(full, 'read', side_effect=lambda path: self.rows[int(path.stem[4:])]):
            return full.window(self.root, self.plan, self.job)

    def test_registered_single_workspace_path_is_accepted(self):
        self.assertEqual(len(self.window()), 8)

    def test_rejects_aliased_or_double_workspace(self):
        name = self.rows[0]['target_modules'][1]
        self.rows[0]['workspace_checks'][name][0]['packet_ptr'] = 1000
        with self.assertRaisesRegex(ValueError, 'Aliased'): self.window()
        self.rows[0]['workspace_checks'][name][0]['packet_ptr'] = 1001
        self.rows[0]['ma2_codec_state'][name]['workspaces'] = 2
        with self.assertRaisesRegex(ValueError, 'workspace count'): self.window()

    def test_wrong_arm_environment_library_and_skipped_updates_fail(self):
        saved = copy.deepcopy(self.rows)
        for key, value, pattern in (('ma2_arm', 'bad', 'rank record'),
                                    ('ma2_shared_environment', {}, 'runtime settings'),
                                    ('skips', 1, 'skipped updates'),
                                    ('libflux_cuda.so', {'path': '/wrong', 'sha256': 'bad'}, 'runtime differs')):
            with self.subTest(key=key):
                self.rows = copy.deepcopy(saved)
                self.rows[0][key] = value
                with self.assertRaisesRegex(ValueError, pattern): self.window()


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ma2-report-CPU-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = fixture_plan()
        bench.write(self.root/'plan.json', self.plan)
        self.hash = bench.sha(self.root/'plan.json')

    def report(self, transform=None):
        reference = fixture_rows(self.plan, dict(self.plan['schedule'][0], mode='timing'), self.hash)
        def window(root, plan, job):
            rows = fixture_rows(plan, job, self.hash)
            if transform: transform(rows, job)
            return rows
        gate = dict(files={}, rank_identities=full.identity(reference))
        with patch.object(full, 'verify', return_value=self.plan), patch.object(full, 'preflight', return_value=gate), \
             patch.object(full, 'window', side_effect=window):
            return full.report(self.root, save=False)

    def test_both_independent_trials_require_target_and_packed_increment(self):
        result = self.report()
        self.assertTrue(result['target_validated'])
        self.assertEqual(result['formal_optimizer_steps'], 3600)
        self.assertEqual(result['timing_rank_records'], 576)
        self.assertEqual(len(result['all_samples']), 72)
        self.assertEqual(set(result['comparisons']), {'compare1', 'compare2'})
        def same_as_ma1(rows, job):
            if job['trial'] == 2 and job['arm'] == 'V':
                for row in rows: row['elapsed_seconds'] = .49*50
        result = self.report(same_as_ma1)
        self.assertTrue(result['one_percent_point_target_validated'])
        self.assertFalse(result['packed_benefit_validated'])
        self.assertFalse(result['target_validated'])
        self.assertTrue(result['comparisons']['compare1']['target_validated'])
        self.assertFalse(result['comparisons']['compare2']['target_validated'])

    def test_gradient_mismatch_cannot_hide_behind_equal_losses(self):
        def change(rows, job):
            if job['arm'] == 'V': rows[7]['grad_norms'][0] += .0000001
        with self.assertRaisesRegex(ValueError, 'trajectories differ'): self.report(change)

    def test_markdown_keeps_negative_or_unproven_results_visible(self):
        result = self.report()
        text = full.report_markdown(result)
        self.assertIn('compare1', text)
        self.assertIn('compare2', text)
        self.assertIn('V/P', text)
        self.assertIn('3600', text)
        self.assertIn('95% CI', text)
        self.assertIn('点估计 ≥1%', text)
        self.assertIn('CI 下限 ≥1%', text)

    def test_offline_verification_rejects_omitted_window(self):
        self.plan['schedule'].pop()
        (self.root/'plan.json').unlink()
        bench.write(self.root/'plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'balanced blocks'): full.verify(self.root, offline=True)


def trace_fixture(arm):
    events = []
    decoder = {'A': 'taco_decode_ring_kernel(X)', 'P': 'void taco_ma1_decode_kernel<true>(X)',
               'V': 'taco_ma2_decode_kernel(X)'}[arm]
    for role_index, role in enumerate(('attn', 'mlp')):
        for index in range(128):
            external = role_index*128+index
            start = external*20
            events += [dict(cat='user_annotation', name=f'scope.{role}.candidate', tid=1, ts=start, dur=19),
                       dict(cat='cpu_op', name='GemmRSFunction', tid=1, ts=start+1, dur=17,
                            args={'External id': external}),
                       dict(cat='cpu_op', name='aten::clone', tid=1, ts=start+12, dur=1)]
            for offset, name in ((2, 'flux_bf16_reducescatter'), (4, 'CudaIpcBarrierAllKernel'),
                                 (6, decoder), (8, 'CudaIpcBarrierAllKernel')):
                events.append(dict(cat='kernel', name=name, ts=start+offset, dur=1,
                                   args={'External id': external, 'stream': 7}))
    return events


class ProfileTests(unittest.TestCase):
    def test_every_projection_has_expected_decoder_barriers_and_clone(self):
        for arm in bench.ARMS:
            result = full.trace_projection_records(trace_fixture(arm), arm)
            self.assertEqual(result['attn']['projection_calls'], 128)
            self.assertEqual(result['mlp']['barrier_calls'], 256)

    def test_wrong_decoder_missing_clone_and_barrier_order_fail(self):
        with self.assertRaisesRegex(ValueError, 'decoder'): full.trace_projection_records(trace_fixture('P'), 'V')
        events = trace_fixture('V')
        events.pop(2)
        with self.assertRaisesRegex(ValueError, 'clone'): full.trace_projection_records(events, 'V')
        events = trace_fixture('V')
        events[5]['args']['stream'] = 9
        with self.assertRaisesRegex(ValueError, 'ordering'): full.trace_projection_records(events, 'V')

    def test_generated_worker_is_valid_and_records_actual_ownership(self):
        worker = full.worker_source()
        ast.parse(worker)
        self.assertIn("report['ma2_codec_state']", worker)
        self.assertIn("H100_MA2_PROTOCOL", worker)
        self.assertNotIn('MEGATRON_OP3_GUARD', worker)

    def test_old_ma1_and_new_packed_kernels_cannot_be_confused(self):
        for arm, name in [('A', 'taco_decode_ring_kernel(X)'),
                          ('P', 'void taco_ma1_decode_kernel<true>(X)'),
                          ('V', 'taco_ma2_decode_kernel(X)')]:
            for candidate in bench.ARMS:
                self.assertEqual(full.decoder_matches(name, candidate), candidate == arm)
        self.assertFalse(full.decoder_matches('void taco_ma1_decode_kernel<false>(X)', 'P'))

    def test_gpu_annotation_mirrors_do_not_duplicate_cpu_projection_scopes(self):
        events = trace_fixture('V')
        mirrors = [dict(event, cat='gpu_user_annotation', tid=7) for event in events
                   if event.get('cat') == 'user_annotation']
        result = full.trace_projection_records(events+mirrors, 'V')
        self.assertEqual(result['attn']['projection_calls'], 128)


class AcceptanceTests(unittest.TestCase):
    def estimates(self, main=1.2, lower=.2, increment=.3, increment_lower=.01, control=.5):
        return {'V/A': dict(percent=main, ci95=[lower, 2.]),
                'V/P': dict(percent=increment, ci95=[increment_lower, 1.]),
                'P/A': dict(percent=control, ci95=[-.1, 1.])}

    def test_point_target_is_separate_from_strict_confidence_target(self):
        result = full.comparison_acceptance(self.estimates(main=1., lower=.01))
        self.assertTrue(result['target_validated'])
        self.assertTrue(result['one_percent_point_target_validated'])
        self.assertFalse(result['strict_one_percent_confidence_validated'])
        strict = full.comparison_acceptance(self.estimates(main=1.2, lower=1.))
        self.assertTrue(strict['strict_one_percent_confidence_validated'])

    def test_sub_one_percent_or_zero_lower_bound_fails_the_point_target(self):
        for estimates in (self.estimates(main=.999), self.estimates(lower=0.),
                          self.estimates(increment_lower=0.), self.estimates(increment=0.)):
            self.assertFalse(full.comparison_acceptance(estimates)['target_validated'])

    def test_negative_ma1_control_does_not_gate_new_mechanism(self):
        result = full.comparison_acceptance(self.estimates(control=-.3))
        self.assertTrue(result['target_validated'])
        self.assertFalse(result['ma1_control_passed'])

    def test_nonfinite_or_reversed_interval_is_rejected(self):
        for estimates in (self.estimates(main=float('nan')), self.estimates(lower=3.)):
            with self.assertRaisesRegex(ValueError, 'confidence interval'):
                full.comparison_acceptance(estimates)


if __name__ == '__main__':
    unittest.main()
