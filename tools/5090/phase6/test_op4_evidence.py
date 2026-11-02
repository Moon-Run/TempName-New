"""CPU-only OP4 evidence admission; synthetic records are never GPU results."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import op4_evidence as evidence
from rtx5090_loader import REPO


def communication(candidate=False):
    env = {'NCCL_P2P_DISABLE': '1', 'NCCL_SHM_DISABLE': '0',
           'NCCL_NVLS_ENABLE': '0', 'CUDA_DEVICE_MAX_CONNECTIONS': '1'}
    if candidate:
        env.update(NCCL_MIN_NCHANNELS='16', NCCL_MAX_NCHANNELS='16')
    return {'environment': env, 'world_size': 8, 'backend': 'nccl'}


def trace(channels=4, rank=0):
    """One step's communication, with independent launch and device records."""
    events = []
    for i in range(524):
        correlation = 1000+i
        events.append(dict(cat='cuda_runtime', name='cudaLaunchKernel', ph='X',
            ts=i*100, dur=1, args={'correlation': correlation}))
        events.append(dict(cat='kernel', name='ncclDevKernel_AllGather_RING_LL',
            ph='X', ts=i*100+5, dur=70, args={
                'correlation': correlation, 'device': rank, 'stream': 7,
                'Group size': 8, 'dtype': 'BFloat16', 'Collective name': '_allgather_base',
                'In msg nelems': 1048576, 'Out msg nelems': 8388608,
                'Process Group Description': 'TENSOR_MODEL_PARALLEL_GROUP',
                'Process Group Ranks': '[0, 1, 2, 3, 4, 5, 6, 7]',
                'grid': [channels, 1, 1], 'block': [544, 1, 1], 'shared memory': 0}))
    for i, kind in enumerate(['SendRecv']*256 + ['ReduceScatter_RING_SIMPLE']*260
                             + ['AllReduce_RING_LL']*14):
        events.append(dict(cat='kernel', name='ncclDevKernel_'+kind, ph='X',
            ts=60000+i*100, dur=50, args={'device': rank, 'stream': 9,
                'grid': [8, 1, 1], 'block': [512, 1, 1], 'shared memory': 0,
                'Group size': 8, 'dtype': 'BFloat16'}))
    return events


def rows(mode, candidate=False):
    """Minimal post-OP3-replay records for the cross-arm equality boundary."""
    output = []
    for rank in range(8):
        row = dict(rank=rank, gpu_uuid=f'cpu-fixture-{rank}', torch='2.8-fixture',
            cuda='12.9', nccl=[2, 27, 3], python='/cpu/python',
            hardware={'host': 'CPU-FIXTURE', 'rank': rank}, model={'layers': 32},
            transport='nccl-staged', initial_parameters_sha256='a'*64,
            initial_rng_sha256='b'*64, tokens_sha256='c'*64,
            parameter_count_local=849207296, padded_vocab_size=33792,
            communication=communication(candidate), losses=[1.0, 0.5, 0.25],
            grad_norms=[2.0, 1.0, 0.5], final_parameters_sha256='d'*64,
            initial_loss=2.0, warmup_last_loss=1.5, skips=0, seed=42,
            cpu_affinity_policy='inherit', cpu_affinity=[0, 1, 2, 3], precision='bf16',
            target_modules=['decoder.layers.0.self_attention.linear_proj'],
            op3_implementation={'encoder': 'warp-owned-four-groups'},
            op3_optimization_stack={'ma8': True, 'ma9': True, 'ma11': True},
            codec_identity={'implementation': 'op3-fixture', 'source_sha256': '0'*64})
        row.update(trainable_parameters={'weight': {'shape': [2], 'dtype': 'bf16'}},
            parameter_trajectory=[{'step': 0, 'gradients': {'weight': 'e'*64},
                                   'parameters': {'weight': 'f'*64}}] if mode == 'smoke' else [],
            smoke_output_sha256={'projection': '1'*64} if mode == 'smoke' else {})
        output.append(row)
    return output


class ProfileAdmission(unittest.TestCase):
    def test_detects_channel_grid_without_inferring_protocol_from_entry_name(self):
        for channels in (4, 16):
            result = evidence.profile_evidence(trace(channels), 0, channels)
            self.assertEqual(result['allgather_calls'], 524)
            self.assertEqual(result['allgather_grid_channels'], channels)
            self.assertFalse(result['algorithm_protocol_inferred_from_kernel_name'])
            self.assertNotIn('allgather_protocol', result)
            self.assertAlmostEqual(result['allgather_total_ms'], 36.68)
            self.assertEqual(sum(x['calls'] for x in result['other_nccl_launches']), 530)

    def test_actual_historical_rank0_trace_has_four_channels_not_sixteen(self):
        path = (REPO.parent/'Load-5090/logs/5090/phase6/'
                'op3-two-arm-v1-2x-50steps-20261102-022324-519378/'
                'results/profile-op3-r/rank0-profile.json.gz')
        if not path.is_file():
            self.skipTest('Historical RTX 5090 rank0 trace is unavailable')
        with gzip.open(path, 'rt') as handle:
            events = json.load(handle)['traceEvents']
        record = evidence.profile_evidence(events, 0, 4)
        self.assertEqual(record['allgather_calls'], 524)
        self.assertGreater(record['allgather_total_ms'], 0)
        self.assertFalse(record['algorithm_protocol_inferred_from_kernel_name'])
        with self.assertRaisesRegex(ValueError, 'required 16 channel grid'):
            evidence.profile_evidence(events, 0, 16)

    def test_missing_gather_or_projection_exchange_is_rejected(self):
        for name in ('AllGather', 'SendRecv'):
            events = trace()
            events.remove(next(e for e in events if e['cat'] == 'kernel' and name in e['name']))
            with self.subTest(name=name), self.assertRaises(ValueError):
                evidence.profile_evidence(events, 0, 4)

    def test_changed_entry_kernel_is_rejected_without_claiming_actual_protocol(self):
        for name in ('ncclDevKernel_AllGather_RING_LL128',
                     'ncclDevKernel_AllGather_TREE_SIMPLE', 'ncclDevKernel_AllGather_RING_SIMPLE'):
            events = trace(16)
            events[1]['name'] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unexpected NCCL AllGather entry'):
                evidence.profile_evidence(events, 0, 16)

    def test_unchanged_or_partially_applied_channel_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'required 16 channel grid'):
            evidence.profile_evidence(trace(4), 0, 16)
        for grid in ([4, 1, 1], [16, 2, 1], [8, 1, 1]):
            events = trace(16)
            events[1047]['args']['grid'] = grid
            with self.subTest(grid=grid), self.assertRaisesRegex(ValueError, 'required 16 channel grid'):
                evidence.profile_evidence(events, 0, 16)

    def test_payload_device_and_process_group_are_bound_to_tp8(self):
        changes = {'device': 1, 'dtype': 'Float', 'Group size': 4,
            'In msg nelems': 524288, 'Out msg nelems': 4194304,
            'Collective name': 'all_gather', 'Process Group Description': 'WORLD',
            'Process Group Ranks': '[7, 6, 5, 4, 3, 2, 1, 0]'}
        for key, value in changes.items():
            events = trace()
            events[1]['args'][key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'TP group or BF16 payload'):
                evidence.profile_evidence(events, 0, 4)

    def test_actual_launch_is_required_unique_and_precedes_gpu_work(self):
        for kind in ('missing', 'duplicate', 'shared', 'late'):
            events = trace()
            if kind == 'missing':
                events.pop(0)
            elif kind == 'duplicate':
                events.append(copy.deepcopy(events[0]))
            elif kind == 'shared':
                events[3]['args']['correlation'] = events[1]['args']['correlation']
            else:
                events[0]['ts'] = events[1]['ts']+10
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evidence.profile_evidence(events, 0, 4)

    def test_nonfinite_or_nonpositive_time_cannot_enter_diagnostics(self):
        for position, key, value in ((1, 'dur', 0), (1, 'dur', float('inf')),
                                     (1, 'ts', float('inf')), (0, 'ts', float('-inf'))):
            events = trace()
            events[position][key] = value
            with self.subTest(position=position, field=key, value=value), self.assertRaises(ValueError):
                evidence.profile_evidence(events, 0, 4)


class HistoricalRecordAdmission(unittest.TestCase):
    def test_all_modes_compare_without_mutating_rank_evidence(self):
        for mode in ('smoke', 'profile', 'timing'):
            control, candidate = rows(mode), rows(mode, True)
            saved = copy.deepcopy((control, candidate))
            result = evidence.compare_rows(control, candidate, mode)
            self.assertTrue(result['numerical_trajectory_identical'])
            self.assertEqual(result['full_parameter_trajectory_checked'], mode == 'smoke')
            self.assertEqual((control, candidate), saved)

    def test_environment_change_or_global_protocol_override_is_rejected(self):
        for side, key, value in ((0, 'NCCL_PROTO', 'LL'), (1, 'NCCL_PROTO', 'Simple'),
                                (1, 'NCCL_SHM_DISABLE', '1'),
                                (1, 'CUDA_DEVICE_MAX_CONNECTIONS', '8')):
            pair = [communication(), communication(True)]
            pair[side]['environment'][key] = value
            with self.subTest(side=side, field=key), self.assertRaises(ValueError):
                evidence.compare_communication(*pair)
        for key in ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_TUNER_PLUGIN', 'NCCL_MIN_NRINGS',
                    'NCCL_MAX_NRINGS', 'NCCL_MIN_CTAS', 'NCCL_MAX_CTAS'):
            pair = [communication(), communication(True)]
            for item in pair:
                item['environment'][key] = 'foreign-override'
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'hidden protocol, algorithm or channel override'):
                evidence.compare_communication(*pair)

    def test_exact_two_channel_overrides_are_required_only_on_candidate(self):
        for key in ('NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS'):
            for side, value in ((0, '16'), (1, '8'), (1, None)):
                pair = [communication(), communication(True)]
                if value is None:
                    pair[side]['environment'].pop(key)
                else:
                    pair[side]['environment'][key] = value
                with self.subTest(field=key, side=side, value=value), self.assertRaises(ValueError):
                    evidence.compare_communication(*pair)

    def test_rank_coverage_order_and_initial_identity_must_match(self):
        control, candidate = rows('timing'), rows('timing', True)
        with self.assertRaises(ValueError):
            evidence.compare_rows(control, candidate[:-1], 'timing')
        with self.assertRaises(ValueError):
            evidence.compare_rows(control, list(reversed(candidate)), 'timing')
        for key in ('gpu_uuid', 'torch', 'initial_parameters_sha256', 'initial_rng_sha256',
                    'tokens_sha256', 'model'):
            candidate = rows('timing', True)
            candidate[3][key] = 'foreign-identity'
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'hardware/software/model'):
                evidence.compare_rows(control, candidate, 'timing')

    def test_every_formal_loss_gradient_and_final_parameter_hash_must_match(self):
        for key in ('losses', 'grad_norms', 'final_parameters_sha256'):
            control, candidate = rows('timing'), rows('timing', True)
            if key == 'final_parameters_sha256':
                candidate[7][key] = '0'*64
            else:
                candidate[7][key][-1] += 1e-12
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'loss, gradients or parameter'):
                evidence.compare_rows(control, candidate, 'timing')

    def test_smoke_checks_full_gradient_update_and_projection_bytes(self):
        for kind in ('gradient', 'update', 'parameter_universe', 'projection'):
            control, candidate = rows('smoke'), rows('smoke', True)
            if kind in ('gradient', 'update'):
                field = 'gradients' if kind == 'gradient' else 'parameters'
                candidate[4]['parameter_trajectory'][0][field]['weight'] = '9'*64
            elif kind == 'parameter_universe':
                candidate[4]['trainable_parameters']['weight']['shape'] = [3]
            else:
                candidate[4]['smoke_output_sha256']['projection'] = '9'*64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evidence.compare_rows(control, candidate, 'smoke')

    def test_affinity_and_optimization_stack_cannot_be_hidden_in_timing_gain(self):
        for key, value in (('cpu_affinity', [0]), ('cpu_affinity_policy', 'single-core'),
                           ('op3_optimization_stack', {}), ('precision', 'fp16'),
                           ('codec_identity', {'source_sha256': 'changed'})):
            control, candidate = rows('timing'), rows('timing', True)
            candidate[0][key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'CPU affinity or the OP3'):
                evidence.compare_rows(control, candidate, 'timing')


class PersistedCandidateAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='op4-evidence-')
        self.addCleanup(self.temp.cleanup)
        self.roots = [Path(self.temp.name)/side for side in ('reference', 'candidate')]
        for candidate, root in enumerate(self.roots):
            plan = dict(protocol=evidence.p.PROTOCOL, arms=[evidence.p.COMBINED],
                trials=1, blocks=1, operator_only=False, communication=communication(bool(candidate)),
                model={'layers': 32}, environment={'torch': '2.8-fixture'},
                cuda_visible_devices='0,1,2,3,4,5,6,7',
                build=dict(root='/cpu/same-build', manifest_sha256='2'*64))
            evidence.p.write(root/'plan.json', plan)
            for job in evidence.p.jobs(1, 1):
                for rank, row in enumerate(rows(job['mode'], bool(candidate))):
                    evidence.p.write(root/f"results/{job['directory']}/rank{rank}.json", row)
            for rank in range(8):
                self.write_trace(candidate, rank, trace(16 if candidate else 4, rank))
            operator_files = {}
            for rank in range(8):
                name = f'rank{rank}.json'
                evidence.p.write(root/'operator'/name, dict(rank=rank, completed=True,
                    protocol='rtx5090-op3-combined-operator-v1', arm=evidence.p.COMBINED,
                    gpu_uuid=f'cpu-fixture-{rank}', torch='2.8-fixture', cuda='12.9', nccl=[2, 27, 3],
                    settings=dict(warmup=20, iterations=100, windows=1, profile_calls=10, gradient_steps=3),
                    entrypoint_sha256='3'*64,
                    libraries={'libflux.so': {'path': '/cpu/build/libflux.so', 'sha256': '7'*64}},
                    codec_library={'path': '/cpu/build/codec.so', 'sha256': '8'*64},
                    roles={role: dict(shape=[2048, 4096, k], gradient_checks=[dict(step=0,
                        output_sha256='4'*64, gradients={'input': '5'*64}, updates={'input': '6'*64})])
                        for role, k in (('attn', 512), ('mlp', 2048))}))
                operator_files[name] = evidence.p.sha(root/'operator'/name)
                for role in ('attn', 'mlp'):
                    name = f'{role}-rank{rank}-profile.json.gz'
                    kernels = [e for e in trace(16 if candidate else 4, rank)
                               if e['cat'] == 'kernel' and 'SendRecv' in e['name']][:10]
                    with gzip.open(root/'operator'/name, 'wt') as handle:
                        json.dump({'traceEvents': kernels}, handle)
                    operator_files[name] = evidence.p.sha(root/'operator'/name)
            operator_report = dict(protocol='rtx5090-op3-combined-operator-v1', completed=True,
                arm=evidence.p.COMBINED, settings=dict(warmup=20, iterations=100, windows=1,
                    profile_calls=10, gradient_steps=3), operator_gradients_validated=True,
                source_order_preserved=True, timed_arms=[evidence.p.COMBINED], full_step_measured=False,
                roles={role: {'gpu_us': {'mean': 700.0}} for role in ('attn', 'mlp')},
                evidence_files=operator_files)
            evidence.p.write(root/'operator/report.json', operator_report)
            profile_files = {f'results/profile-{evidence.p.COMBINED}/rank{rank}-profile.json.gz':
                             evidence.p.sha(root/f'results/profile-{evidence.p.COMBINED}/rank{rank}-profile.json.gz')
                             for rank in range(8)}
            evidence.p.write(root/'profile-summary.json', dict(protocol=evidence.p.PROTOCOL,
                completed=True, execution_validated=True, evidence_files=profile_files))
            worker_files = {f"results/{job['directory']}/rank{rank}.json":
                            evidence.p.sha(root/f"results/{job['directory']}/rank{rank}.json")
                            for job in evidence.p.jobs(1, 1) for rank in range(8)}
            evidence.p.write(root/'report.json', dict(protocol=evidence.p.PROTOCOL, completed=True,
                arms=[evidence.p.COMBINED], trials=1, blocks=1, correctness_passed=True,
                execution_path_validated=True, worker_records=worker_files,
                plan_sha256=evidence.p.sha(root/'plan.json'),
                profile_summary_sha256=evidence.p.sha(root/'profile-summary.json'),
                operator_report_sha256=evidence.p.sha(root/'operator/report.json'),
                comparisons={'compare1': {'arms': {evidence.p.COMBINED: {'mean_ms': 850.195}}}}))

    def audit(self):
        return evidence.audit_candidate(self.roots[1], self.roots[0])

    def write_trace(self, side, rank, events):
        path = self.roots[side]/f'results/profile-{evidence.p.COMBINED}/rank{rank}-profile.json.gz'
        with gzip.open(path, 'wt') as handle:
            json.dump({'traceEvents': events}, handle)

    def test_candidate_binds_historical_records_without_claiming_concurrent_control(self):
        result = self.audit()
        self.assertTrue(result['passed'])
        self.assertTrue(result['full_step_completed'])
        self.assertTrue(result['historical_numerical_regression_checked'])
        self.assertFalse(result['full_training_cross_arm_comparison'])
        self.assertFalse(result['reference_binary_revalidated'])
        self.assertFalse(result['concurrent_control_measured'])
        self.assertFalse(result['profile_durations_are_formal_speedups'])
        self.assertEqual(result['historical_full_step_ms'], 850.195)
        self.assertEqual(len(result['evidence_files']), 118)
        self.assertEqual(len(result['profiles']), 8)
        self.assertEqual({x['mode'] for x in result['numerical_checks']}, {'smoke', 'profile', 'timing'})
        self.assertTrue(all(len(digest) == 64 for digest in result['evidence_files'].values()))

    def test_non_allgather_entry_payload_or_group_change_is_rejected(self):
        for key, value in (('name', 'ncclDevKernel_ReduceScatter_RING_LL'),
                           ('In msg nelems', 123), ('Process Group Ranks', '[0, 1]')):
            events = trace(16, 7)
            target = next(e for e in events if 'ReduceScatter' in e['name'])
            (target if key == 'name' else target['args'])[key] = value
            self.write_trace(1, 7, events)
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'Non-AllGather NCCL'):
                self.audit()

    def test_non_allgather_geometry_may_change_but_is_reported(self):
        events = trace(16, 7)
        for event in events:
            if event['cat'] == 'kernel' and 'AllGather' not in event['name']:
                event['args'].update(grid=[16, 1, 1], block=[256, 1, 1], **{'shared memory': 1024})
        self.write_trace(1, 7, events)
        result = self.audit()
        self.assertTrue(result['other_collective_launch_geometry_may_change'])
        profile = result['profiles'][7]
        self.assertEqual(profile['reference']['other_nccl_payloads'],
                         profile['candidate']['other_nccl_payloads'])
        self.assertNotEqual(profile['reference']['other_nccl_launches'],
                            profile['candidate']['other_nccl_launches'])

    def test_non_allgather_call_count_change_is_rejected(self):
        events = trace(16, 7)
        events.append(copy.deepcopy(next(e for e in events if 'AllReduce' in e['name'])))
        self.write_trace(1, 7, events)
        with self.assertRaisesRegex(ValueError, 'Non-AllGather NCCL'):
            self.audit()

    def test_incomplete_child_or_different_environment_is_rejected(self):
        path = self.roots[1]/'plan.json'
        original = evidence.p.read(path)
        for key, value in (('blocks', 2), ('environment', {'torch': 'different-version'})):
            evidence.p.write(path, dict(original, **{key: value}))
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.audit()

    def test_valid_rebuild_paths_may_differ_but_source_mechanisms_must_match(self):
        path = self.roots[1]/'plan.json'
        evidence.p.write(path, dict(evidence.p.read(path),
            build=dict(root='/cpu/new-build', manifest_sha256='9'*64)))
        path = self.roots[1]/'operator/rank0.json'
        record = evidence.p.read(path)
        record['codec_library'] = {'path': '/cpu/new-build/codec.so', 'sha256': '9'*64}
        evidence.p.write(path, record)
        report_path = self.roots[1]/'operator/report.json'
        report = evidence.p.read(report_path)
        report['evidence_files']['rank0.json'] = evidence.p.sha(path)
        evidence.p.write(report_path, report)
        result = self.audit()
        self.assertFalse(result['build_paths_equal'])
        self.assertEqual(result['candidate_build']['root'], '/cpu/new-build')
        self.assertFalse(result['reference_binary_revalidated'])
        self.assertEqual(result['operator_binary_records'][0]['candidate']['codec_library']['path'],
                         '/cpu/new-build/codec.so')
        path = self.roots[1]/'results/smoke-op3-combined/rank0.json'
        record = evidence.p.read(path)
        record['codec_identity']['source_sha256'] = '7'*64
        evidence.p.write(path, record)
        with self.assertRaisesRegex(ValueError, 'OP3 mechanism stack'):
            self.audit()

    def test_operator_only_does_not_read_fullstep_records_or_require_allgather(self):
        path = self.roots[1]/'plan.json'
        evidence.p.write(path, dict(evidence.p.read(path), operator_only=True))
        for root in self.roots:
            shutil.rmtree(root/'results')
        result = self.audit()
        self.assertTrue(result['passed'])
        self.assertTrue(result['operator_only'])
        self.assertFalse(result['full_step_completed'])
        self.assertFalse(result['historical_numerical_regression_checked'])
        self.assertFalse(result['reference_binary_revalidated'])
        self.assertEqual(result['profiles'], [])
        self.assertEqual(result['numerical_checks'], [])
        self.assertNotIn('historical_full_step_ms', result)
        self.assertEqual(len(result['operator_profiles']['candidate']), 16)
        self.assertTrue(all(item['sendrecv_calls'] == 10 and item['diagnostic_only']
                            and not item['allgather_grid_checked']
                            for item in result['operator_profiles']['candidate']))

    def test_missing_reference_or_reference_hash_tampering_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing reference evidence'):
            evidence.audit_candidate(self.roots[1], self.roots[0]/'missing')
        name = 'results/smoke-op3-combined/rank7.json'
        path = self.roots[0]/name
        record = evidence.p.read(path)
        record['initial_loss'] += 1
        evidence.p.write(path, record)
        with self.assertRaisesRegex(ValueError, 'Changed reference evidence SHA256'):
            self.audit()

    def test_foreign_or_incomplete_reference_report_is_rejected(self):
        path = self.roots[0]/'report.json'
        original = evidence.p.read(path)
        for key, value in (('protocol', 'foreign-protocol'), ('completed', False),
                           ('arms', ['op3-r']), ('blocks', 2), ('correctness_passed', False)):
            evidence.p.write(path, dict(original, **{key: value}))
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'historical combined full-step report'):
                self.audit()

    def test_historical_manifest_must_cover_every_rank(self):
        path = self.roots[0]/'report.json'
        record = evidence.p.read(path)
        record['worker_records'].pop('results/smoke-op3-combined/rank7.json')
        evidence.p.write(path, record)
        with self.assertRaisesRegex(ValueError, 'reference evidence manifest'):
            self.audit()

    def test_operator_only_still_rejects_historical_operator_hash_tampering(self):
        path = self.roots[1]/'plan.json'
        evidence.p.write(path, dict(evidence.p.read(path), operator_only=True))
        path = self.roots[0]/'operator/rank7.json'
        record = evidence.p.read(path)
        record['arm'] = 'op3-r'
        evidence.p.write(path, record)
        with self.assertRaisesRegex(ValueError, 'Changed reference evidence SHA256'):
            self.audit()

    def test_operator_only_rejects_changed_hardware_or_numerical_record(self):
        path = self.roots[1]/'plan.json'
        evidence.p.write(path, dict(evidence.p.read(path), operator_only=True))
        rank_path = self.roots[1]/'operator/rank0.json'
        original = evidence.p.read(rank_path)
        report_path = self.roots[1]/'operator/report.json'
        for kind in ('hardware', 'gradient'):
            record = copy.deepcopy(original)
            if kind == 'hardware':
                record['gpu_uuid'] = 'different-device'
            else:
                record['roles']['attn']['gradient_checks'][0]['gradients']['input'] = '0'*64
            evidence.p.write(rank_path, record)
            report = evidence.p.read(report_path)
            report['evidence_files']['rank0.json'] = evidence.p.sha(rank_path)
            evidence.p.write(report_path, report)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Historical/candidate operator'):
                self.audit()

    def test_full_report_must_bind_operator_report_hash(self):
        path = self.roots[0]/'report.json'
        record = evidence.p.read(path)
        record.pop('operator_report_sha256')
        evidence.p.write(path, record)
        with self.assertRaisesRegex(ValueError, 'not bound to its operator report'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
