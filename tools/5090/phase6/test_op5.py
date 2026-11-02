"""CPU admission and orchestration checks; no synthetic record is a GPU result."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import op5
import op5_evidence as evidence
import op5_probe as probe
import op3_protocol as p
from test_op4_evidence import rows

HISTORY_MS = 832.2487896867096


def samples(ms=780., blocks=1):
    return [dict(job, ms_per_step=ms) for job in op5.schedule(1, blocks)]


def ce_trace(calls=10, rank=0):
    result = []
    for i in range(calls):
        result.append(dict(cat='kernel', name='ncclDevKernel_AllGather_RING_LL', ts=100*i, dur=80,
                           args={'device': rank, 'grid': [16, 1, 1]}))
        for j in range(7):
            result.append(dict(cat='gpu_memcpy', name='Memcpy HtoD (Host -> Device)',
                ts=100*i+1+j*10, dur=3, args={'device': rank, 'bytes': 2*1048576}))
    return result


def ce_log():
    return '\n'.join(f'host:123:456 [{rank}] NCCL INFO Channel {channel:02d} : '
        f'{rank}[{rank}] -> {(rank+1)%8}[{(rank+1)%8}] via SHM/direct/CE'
        for channel in range(16) for rank in range(8))


class LaunchAndReporting(unittest.TestCase):
    def test_target_uses_op4_and_no_single_window_causal_claim(self):
        for ms, hit in ((HISTORY_MS-50, True), (HISTORY_MS-49.999, False), (900., False)):
            result = op5.report(samples(ms), 1, 1, historical_reference_ms=HISTORY_MS)
            self.assertEqual(result['historical_target_observed'], hit)
            self.assertFalse(result['target_validated'])
            self.assertFalse(result['concurrent_control'])
            self.assertAlmostEqual(result['historical_target_ms'], 782.2487896867096)
            self.assertEqual(result['formal_optimizer_steps'], 50)
        slow = samples(blocks=2)
        slow[1]['ms_per_step'] = 1000
        result = op5.report(slow, 1, 2, historical_reference_ms=HISTORY_MS)
        self.assertFalse(result['historical_target_observed'])
        self.assertEqual(result['all_samples'], slow)
        operators = [dict(op5.schedule()[0], roles={})]
        result = op5.report([], 1, 1, operators, historical_reference_ms=HISTORY_MS)
        self.assertFalse(result['historical_target_observed'])
        self.assertFalse(result['full_step_completed'])
        self.assertEqual(result['formal_optimizer_steps'], 0)
        self.assertIn('未执行完整模型训练', op5.markdown(result))
        with self.assertRaises(ValueError):
            op5.report([], 1, 1, historical_reference_ms=HISTORY_MS)
        with self.assertRaises(ValueError):
            op5.report(samples(float('nan')), 1, 1, historical_reference_ms=HISTORY_MS)

    def test_preview_needs_no_gpu_or_build_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for operator in ([], ['--operator-only']):
                out = Path(tmp)/'not-created'
                run = subprocess.run(['bash', str(op5.HERE/'run_op5.sh'), '--out', str(out), *operator],
                    capture_output=True, text=True, env=dict(os.environ, RTX5090_PYTHON=sys.executable))
                self.assertEqual(run.returncode, 0, run.stderr)
                value = json.loads(run.stdout[run.stdout.index('{'):])
                self.assertEqual(value['nccl_overrides']['candidate'], evidence.OVERRIDES)
                self.assertEqual(value['reference_gpu_jobs'], 0)
                self.assertEqual(value['formal_optimizer_steps'], 0 if operator else 50)
                self.assertFalse(out.exists())
        with self.assertRaises(ValueError):
            op5.arguments(['--execute'])

    def test_hidden_channel_or_shm_configuration_cannot_leak(self):
        env = op5.child_environment('candidate', {})
        self.assertTrue(all(env[k] == v for k, v in evidence.OVERRIDES.items()))
        self.assertEqual(env['NCCL_P2P_LL_THRESHOLD'], '2097152')
        self.assertNotIn('NCCL_PROTO', env)
        self.assertNotIn('NCCL_DEBUG_SUBSYS', env)
        for key in op5.EXTERNAL_OVERRIDES:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                op5.check_overrides({key: '0'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'nccl.conf'
            path.write_text('# test\nNCCL_SHM_MEMCPY_MODE = 3\n')
            with self.assertRaisesRegex(ValueError, 'NCCL_SHM_MEMCPY_MODE'):
                op5.check_overrides({'NCCL_CONF_FILE': str(path)})

    def test_child_and_supervisor_keep_build_mode_and_reference(self):
        args = op5.arguments(['--op3-build', '/example/build', '--operator-only', '--blocks', '2'])
        child = op5.child_command(args, Path('/example/child'), args.op3_build)
        self.assertEqual(child[child.index('--blocks')+1], '1')
        self.assertIn('--operator-only', child)
        self.assertIn('op3.py', Path(child[1]).name)
        supervisor = op5.supervisor_command(args)
        self.assertEqual(Path(supervisor[1]).name, 'op5.py')
        self.assertIn('--operator-only', supervisor)
        self.assertIn('--reference-batch', supervisor)


class NumericalEvidence(unittest.TestCase):
    def pair(self, mode):
        reference, candidate = rows(mode, True), rows(mode, True)
        for row in candidate:
            row['communication']['environment'].update(evidence.INTERVENTION_ENV)
        return reference, candidate

    def test_transport_is_only_permitted_identity_change(self):
        a, b = self.pair('smoke')
        saved = copy.deepcopy((a, b))
        evidence.previous.compare_rows(a, b, 'smoke', communication_check=evidence.compare_communication)
        self.assertEqual((a, b), saved)
        for field, value in (('NCCL_MIN_NCHANNELS', '32'), ('NCCL_SHM_MEMCPY_MODE', '3'),
                             ('NCCL_ALGO', 'Ring'), ('NCCL_P2P_LL_THRESHOLD', '16384')):
            bad = copy.deepcopy(b)
            bad[0]['communication']['environment'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evidence.previous.compare_rows(a, bad, 'smoke', communication_check=evidence.compare_communication)

    def test_failed_v1_transport_cannot_be_admitted_as_v2(self):
        a, b = self.pair('smoke')
        for row in b:
            row['communication']['environment'].pop('NCCL_P2P_LL_THRESHOLD')
        with self.assertRaisesRegex(ValueError, 'NCCL_P2P_LL_THRESHOLD'):
            evidence.previous.compare_rows(a, b, 'smoke', communication_check=evidence.compare_communication)
        self.assertIn('p2p-ll-single-v2', op5.PROTOCOL)
        self.assertIn('p2p-ll-single-v2', op5.arguments([]).out.name)

    def test_original_loss_and_parameter_gates_still_apply(self):
        for mode in ('smoke', 'profile', 'timing'):
            for field, value in (('losses', [3., 2., 1.]), ('final_parameters_sha256', '0'*64)):
                a, b = self.pair(mode)
                b[0][field] = value
                with self.subTest(mode=mode, field=field), self.assertRaises(ValueError):
                    evidence.previous.compare_rows(a, b, mode, communication_check=evidence.compare_communication)
        a, b = self.pair('smoke')
        b[0]['parameter_trajectory'][0]['gradients']['weight'] = '0'*64
        result = evidence.previous.compare_rows(a, b, 'smoke', communication_check=evidence.compare_communication)
        self.assertFalse(result['historical_gradient_hash_equality_required'])
        self.assertTrue(result['historical_gradient_hash_differences'])

    def test_real_op4_uses_both_windows_without_old_binary_replay(self):
        if not (op5.DEFAULT_REFERENCE/'report.json').exists():
            self.skipTest('Archived OP4 batch unavailable')
        result = evidence.reference_record(op5.DEFAULT_REFERENCE)
        self.assertEqual(result['timing_windows'], 2)
        self.assertEqual(result['all_ms'], [833.4861183352768, 831.0114610381424])
        self.assertEqual(result['historical_mean_ms'], HISTORY_MS)
        self.assertEqual(result['reference_gpu_jobs'], 0)
        self.assertFalse(result['historical_binaries_revalidated'])


class ActualPathAdmission(unittest.TestCase):
    def test_sendrecv_rejects_contained_and_cross_boundary_ce_activity(self):
        events = [dict(cat='kernel', name='ncclDevKernel_SendRecv', ts=100., dur=50., args={'device': 0})]
        result = evidence.sendrecv_evidence(events, 0, 1)
        self.assertFalse(result['actual_protocol_inferred_from_kernel_name'])
        self.assertIn('inference', result['ll_selection_basis'])
        self.assertIn('alone do not identify', result['limitation'])
        for ts, dur, reject in ((110., 1., True), (90., 20., True), (140., 20., True),
                                (90., 10., False), (150., 10., False)):
            copied = events+[dict(cat='gpu_memcpy', name='Memcpy HtoD', ts=ts, dur=dur,
                                 args={'device': 0, 'bytes': 4096})]
            with self.subTest(ts=ts):
                if reject:
                    with self.assertRaisesRegex(ValueError, 'overlaps HtoD'):
                        evidence.sendrecv_evidence(copied, 0, 1)
                else:
                    evidence.sendrecv_evidence(copied, 0, 1)
        with self.assertRaisesRegex(ValueError, 'Missing OP5 SendRecv'):
            evidence.sendrecv_evidence(events, 0, 2)
        with self.assertRaisesRegex(ValueError, 'Invalid OP5 SendRecv'):
            evidence.sendrecv_evidence(events, 1, 1)

    def test_full_model_htod_overlap_is_diagnostic_without_copy_owner(self):
        events = [dict(cat='kernel', name='ncclDevKernel_SendRecv', ts=100., dur=50., args={'device': 0}),
                  dict(cat='gpu_memcpy', name='Memcpy HtoD', ts=90., dur=20.,
                       args={'device': 0, 'bytes': 65536})]
        result = evidence.sendrecv_evidence(events, 0, 1, isolated=False)
        self.assertFalse(result['no_htod_overlap_required'])
        self.assertEqual(result['overlapping_htod_copies'], 1)
        self.assertEqual(result['per_call'][0]['overlapping_htod_bytes'], 65536)
        self.assertIn('does not establish copy ownership', result['limitation'])

    def test_every_gather_requires_bounded_contained_htod_activity(self):
        result = evidence.copy_evidence(ce_trace(), 0, 10)
        self.assertEqual(result['ce_covered_calls'], 10)
        self.assertEqual(result['htod_bytes'], 10*14*1048576)
        for change in ('missing', 'short', 'outside', 'wrong_device', 'old_grid', 'negative_duration'):
            events = ce_trace()
            if change == 'missing':
                events.pop()
                events.pop()
            elif change == 'short':
                events[1]['args']['bytes'] = 1
                events[2]['args']['bytes'] = 1
            elif change == 'outside':
                events[1]['ts'] = -5
                events[2]['ts'] = -5
            elif change == 'wrong_device':
                events[1]['args']['device'] = 1
                events[2]['args']['device'] = 1
            elif change == 'old_grid':
                events[0]['args']['grid'] = [4, 1, 1]
            else:
                events[1]['dur'] = -1
            with self.subTest(change=change), self.assertRaises(ValueError):
                evidence.copy_evidence(events, 0, 10)

    def test_payload_range_keeps_actual_per_call_bytes(self):
        for total, passes in ((evidence.MIN_GATHER_HTOD_BYTES-1, False),
                              (evidence.MIN_GATHER_HTOD_BYTES, True),
                              (evidence.GATHER_RECEIVED_BYTES, True),
                              (evidence.MAX_GATHER_HTOD_BYTES, True),
                              (evidence.MAX_GATHER_HTOD_BYTES+1, False)):
            events = ce_trace(1)
            for event in events[1:]:
                event['args']['bytes'] = 1
            events[1]['args']['bytes'] = total-6
            with self.subTest(total=total):
                if passes:
                    result = evidence.copy_evidence(events, 0, 1)
                    self.assertEqual(result['contained_htod_bytes_per_call'], [total])
                    self.assertEqual(result['htod_bytes'], total)
                    self.assertIn('does not prove', result['attribution_limitation'])
                else:
                    with self.assertRaises(ValueError):
                        evidence.copy_evidence(events, 0, 1)

    def test_cross_boundary_copy_is_excluded_not_counted_as_contained(self):
        events = ce_trace(1)
        events.append(dict(cat='gpu_memcpy', name='Memcpy HtoD', ts=79, dur=2,
                           args={'device': 0, 'bytes': 65536}))
        result = evidence.copy_evidence(events, 0, 1)
        self.assertEqual(result['contained_htod_bytes_per_call'], [evidence.GATHER_RECEIVED_BYTES])
        self.assertEqual(result['excluded_cross_boundary_copies_per_call'], [1])

    def test_archived_op5_probe_traces_fit_relaxed_gate(self):
        root = (op5.REPO.parent/'Load-5090/logs/5090/phase6/'
                'op5-shm-recv-ce-single-v1-1x-50steps-20261102-122453-878129/transport-probe')
        if not (root/'rank0-profile.json.gz').exists():
            self.skipTest('Archived OP5 probe unavailable')
        import gzip
        for rank in range(8):
            with gzip.open(root/f'rank{rank}-profile.json.gz', 'rt') as stream:
                result = evidence.copy_evidence(json.load(stream)['traceEvents'], rank, 10)
            self.assertEqual(result['ce_covered_calls'], 10)
        self.assertEqual(probe.transport_log((root/'process.log').read_text())['transport'], 'SHM/direct/CE')

    def test_transport_flags_alone_and_incomplete_logs_are_rejected(self):
        result = probe.transport_log(ce_log())
        self.assertEqual(len(result['connections']), 16*8)
        for text in ('NCCL_SHM_USE_CUDA_MEMCPY=1', ce_log().replace('SHM/direct/CE', 'SHM/direct/direct'),
                     ce_log().replace('SHM/direct/CE', 'NET/Socket'), '\n'.join(ce_log().splitlines()[:-1])):
            with self.assertRaises(ValueError):
                probe.transport_log(text)

    def test_op4_trace_cannot_be_mislabeled_as_op5(self):
        path = op5.DEFAULT_REFERENCE/'compare1/block00-candidate/results/profile-op3-combined/rank0-profile.json.gz'
        if not path.exists():
            self.skipTest('Archived trace unavailable')
        import gzip
        with gzip.open(path, 'rt') as stream:
            events = json.load(stream)['traceEvents']
        with self.assertRaisesRegex(ValueError, 'HtoD'):
            evidence.copy_evidence(events, 0, 524)


class Supervision(unittest.TestCase):
    def run_fixture(self, root, fail=None, blocks=1):
        args = op5.arguments(['--out', str(root), '--execute', '--op3-build', str(root/'build'), '--blocks', str(blocks)])
        (root/'handoff').mkdir()
        p.write(root/'batch-state.json', dict(status='launching'))
        calls = []
        def run(command, env, log):
            calls.append((command, env))
            probe_call = Path(command[1]).name == 'op5_probe.py'
            if fail == ('probe' if probe_call else 'child'):
                raise ValueError('injected failure')
            out = Path(command[command.index('--out')+1])
            out.mkdir(parents=True)
            if probe_call:
                p.write(out/'report.json', {'completed': True})
            else:
                p.write(out/'plan.json', dict(build={'root': str(root/'build')}, communication={}))
                p.write(out/'environment-preflight.json', {})
        result = op5.report(samples(blocks=blocks), 1, blocks, historical_reference_ms=HISTORY_MS)
        error = None
        with ExitStack() as stack:
            stack.enter_context(patch.object(op5, 'REPO', root))
            stack.enter_context(patch.object(op5, 'check_overrides'))
            stack.enter_context(patch.object(op5.op3.base, 'verify_controllers'))
            stack.enter_context(patch.object(op5, 'reference_record', return_value={'historical_mean_ms': HISTORY_MS}))
            stack.enter_context(patch.object(op5, 'run_child', side_effect=run))
            stack.enter_context(patch.object(op5, 'child_result', return_value={}))
            stack.enter_context(patch.object(op5.op5_probe, 'evidence', return_value={'completed': True}))
            stack.enter_context(patch.object(op5.op5_probe, 'check_training_identity'))
            stack.enter_context(patch.object(evidence, 'audit_candidate',
                **({'side_effect': ValueError('injected audit failure')} if fail == 'audit' else {'return_value': {'passed': True}})))
            stack.enter_context(patch.object(op5, 'replay', return_value=result))
            try:
                with redirect_stdout(io.StringIO()):
                    op5.supervise(args)
            except ValueError as exc:
                error = exc
        return calls, error

    def test_probe_precedes_all_candidate_windows_and_shares_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, error = self.run_fixture(root, blocks=2)
            self.assertIsNone(error)
            self.assertEqual([Path(cmd[1]).name for cmd, _ in calls], ['op5_probe.py', 'op3.py', 'op3.py'])
            for _, env in calls:
                self.assertTrue(all(env.get(k) == v for k, v in evidence.OVERRIDES.items()))
            self.assertEqual(p.read(root/'batch-state.json')['status'], 'completed')
            self.assertFalse(p.read(root/'report.json')['target_validated'])

    def test_failures_never_emit_success_or_continue_past_probe(self):
        for stage in ('probe', 'child', 'audit'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, error = self.run_fixture(root, fail=stage)
                self.assertIsNotNone(error)
                self.assertEqual(p.read(root/'batch-state.json')['status'], 'failed')
                self.assertFalse((root/'report.json').exists())
                self.assertEqual(len(calls), 1 if stage == 'probe' else 2)


class Replay(unittest.TestCase):
    def test_completed_report_is_bound_to_probe_and_candidate_audit(self):
        for operator_only in (False, True):
            with self.subTest(operator_only=operator_only), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                reference = dict(root=str(root/'historical'), historical_mean_ms=HISTORY_MS)
                job = op5.schedule()[0]
                plan = dict(protocol=op5.PROTOCOL, run_id='cpu-fixture', model=p.MODEL,
                    nccl_overrides=op5.OVERRIDES, target_ms=50., trials=1, blocks=1,
                    operator_only=operator_only, schedule=[job], reference=reference)
                probe_root = root/'transport-probe'
                p.write(probe_root/'report.json', {'completed': True})
                plan['transport_probe_sha256'] = p.sha(probe_root/'report.json')
                p.write(root/'plan.json', plan)
                p.write(root/'candidate-evidence.json', {job['directory']: {'passed': True}})
                child = root/job['directory']
                p.write(child/'plan.json', dict(operator_only=operator_only,
                    build={'root': '/cpu/build'}, model=p.MODEL, communication={}, environment={},
                    cuda_visible_devices='0,1,2,3,4,5,6,7'))
                p.write(child/'environment-preflight.json', {})
                p.write(child/('operator/report.json' if operator_only else 'report.json'), {})
                p.write(child.with_name(child.name+'-launch.json'), dict(job=job, nccl_overrides=op5.OVERRIDES['candidate']))
                stack.enter_context(patch.object(op5.op3.base, 'verify_controllers'))
                stack.enter_context(patch.object(op5, 'reference_record', return_value=reference))
                stack.enter_context(patch.object(op5.op5_probe, 'evidence', return_value={'completed': True}))
                stack.enter_context(patch.object(op5.op5_probe, 'check_training_identity'))
                stack.enter_context(patch.object(op5, 'child_result', return_value=(
                    {'roles': {}} if operator_only else {'all_samples': [{'ms_per_step': 780.}]})))
                stack.enter_context(patch.object(op5, 'training_identity', return_value=['frozen']))
                stack.enter_context(patch.object(evidence, 'audit_candidate', return_value={'passed': True}))
                result = op5.replay(root)
                self.assertEqual(result['formal_optimizer_steps'], 0 if operator_only else 50)
                self.assertTrue(result['receive_ce_probe_validated'])
                self.assertFalse(result['target_validated'])
                p.write(root/'candidate-evidence.json', {job['directory']: {'passed': False}})
                with self.assertRaisesRegex(ValueError, 'candidate evidence differs'):
                    op5.replay(root)
                p.write(root/'candidate-evidence.json', {job['directory']: {'passed': True}})
                p.write(probe_root/'report.json', {'completed': False})
                with self.assertRaisesRegex(ValueError, 'Transport probe changed'):
                    op5.replay(root)


if __name__ == '__main__':
    unittest.main()
