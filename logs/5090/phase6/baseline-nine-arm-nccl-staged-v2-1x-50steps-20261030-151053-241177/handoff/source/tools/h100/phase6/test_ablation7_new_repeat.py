"""CPU-only tests of four independent assignment-based ablation7 comparisons."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation7_new_protocol as protocol
import detach_ablation7_new as batch
from common import read, sha, write


class BatchTests(unittest.TestCase):
    summary = {role: dict(method='receiver-tail-assignment-v3', training_passes=[0, 1],
                         heldout_passes=[2], changed_per_rank=[0]*8, identical_to_A=True,
                         fitted_entry_sha256='same-'+role, accepted_windows=0,
                         arrival_options={'uncertainty_multiplier': 1.0}, exploratory=True)
               for role in ('attn', 'mlp')}

    def summary_for(self, multiplier):
        summary = copy.deepcopy(self.summary)
        for metadata in summary.values():
            metadata.update(arrival_options={'uncertainty_multiplier': multiplier}, exploratory=multiplier < 4)
        return summary

    def fake_invoke(self, calls, mutate=None, multiplier=1.0):
        """Synthesize child reports without launching workers or compiling CUDA."""
        def invoke(cmd, log, env):
            action, root = cmd[2], Path(cmd[3])
            self.assertEqual(Path(cmd[1]).name, 'ablation7_new.py')
            self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
            calls.append(log.stem)
            if action == 'prepare':
                root.mkdir()
                info = dict(protocol=protocol.PROTOCOL, run_id='ablation7-new/'+root.name,
                    executable=True, source_only=False, execution_timing=dict(batch.TIMING),
                    arrival_options={'uncertainty_multiplier': multiplier},
                    rounds=2, blocks_per_round=6, timing_windows=36,
                    schedule=protocol.schedule(6, rounds=2),
                    communication=dict(environment={'NCCL_NVLS_ENABLE': '0'}, nvls_request='0'),
                    budget_fractions={p: protocol.definition(p)['budget_fraction'] for p in protocol.POLICIES},
                    variants={p: protocol.definition(p) for p in protocol.POLICIES},
                    config={}, calibration_hardware={}, megatron='same', megatron_commit='same',
                    candidate_source_commit='same', build_manifest={}, build_files={}, controllers={}, files={})
                write(root/'submission.json', info)
            elif action == 'run' and '--execute' in cmd:
                info = read(root/'submission.json')
                write(root/'all-state.json', dict(status='completed'))
                gate = dict(completed=True, protocol=protocol.PROTOCOL, execution_timing=dict(batch.TIMING),
                    arrival_options=copy.deepcopy(info['arrival_options']),
                    submission_sha256=sha(root/'submission.json'), communication=info['communication'])
                write(root/'preflight.json', gate)
                result = dict(protocol=protocol.PROTOCOL, run_id=info['run_id'], completed=True,
                    correctness_passed=True, execution_timing=dict(batch.TIMING),
                    arrival_options=copy.deepcopy(info['arrival_options']),
                    formal_optimizer_steps=1800, timing_windows=36, all_samples=[{}]*36,
                    round_count=2, blocks_per_round=6, timing_rank_records=288,
                    preflight_rank_records=48, profile_traces=24, candidate_shape_mappings=6,
                    communication=info['communication'], measurement_identity={'same_runtime': True},
                    role_plan_sha256={'frozen_plans': True}, arrival_schedule=self.summary_for(multiplier),
                    evidence=dict(preflight_sha256=sha(root/'preflight.json')))
                write(root/'report.json', result)
            if mutate:
                mutate(log.stem, root, cmd)
        return invoke

    def write_reused_build(self, root, parent=None, multiplier=1.0):
        root.mkdir(parents=True, exist_ok=True)
        options = {'uncertainty_multiplier': multiplier}
        write(root/'ablation7-new-manifest.json',
              dict(parent_build=str(parent or root.parent/'parent'), protocol=protocol.PROTOCOL,
                   source_only=False, arrival_options=options))
        write(root/'plans.json', dict(arrival_options=options))

    def args(self, root, build=False, multiplier=1.0, trials=None):
        flags = ['--out', str(root), '--arrival-margin-multiplier', str(multiplier)]
        if trials is not None:
            flags += ['--trials', str(trials)]
        if build:
            nccl = root/'nccl'
            for name in ('include/nccl.h', 'lib/libnccl_static.a'):
                path = nccl/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic')
            flags += ['--nccl-root', str(nccl)]
        else:
            reused = root/'reused-build'
            self.write_reused_build(reused, multiplier=multiplier)
            flags += ['--ablation7-build', str(reused)]
        return batch.arguments(flags)

    def run_fake(self, args, calls, mutate=None):
        with mock.patch.object(batch.launcher, 'invoke',
                               side_effect=self.fake_invoke(calls, mutate, args.arrival_margin_multiplier)), \
                mock.patch.object(batch, 'order_summary', return_value=self.summary_for(args.arrival_margin_multiplier)), \
                contextlib.redirect_stdout(io.StringIO()):
            return batch.supervise(args)

    def test_preview_is_read_only_and_has_four_separate_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(os, 'fork', side_effect=AssertionError('fork')), \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('process')), \
                mock.patch.object(batch, 'resolve_nccl', side_effect=AssertionError('NCCL lookup')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(tmp)/'new'
            batch.main(['--out', str(root)])
            self.assertFalse(root.exists())
            self.assertIn('7200 formal steps', output.getvalue())
            self.assertIn('Arrival margin: 1 epsilon', output.getvalue())
            self.assertIn('exploratory=True', output.getvalue())
            steps = batch.commands(batch.arguments(['--out', str(root)]))
            self.assertEqual([label for label, _ in steps if label.endswith('-controller')],
                             [f'compare{i}-controller' for i in range(1, 5)])
            self.assertEqual([label for label, _ in steps].count('compile'), 1)
            for label, cmd in steps:
                self.assertEqual(Path(cmd[1]).name, 'ablation7_new.py')
                if label.endswith('-prepare') and label != 'build-prepare':
                    self.assertEqual(Path(cmd[3]).parent, root)
                    for flag, value in (('--blocks', '6'), ('--rounds', '2'), ('--nvls-enable', '0')):
                        self.assertEqual(cmd[cmd.index(flag)+1], value)

    def test_all_four_reports_in_order_build_once_and_identity_still_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            result = self.run_fake(self.args(root, build=True), calls)
            expected = ['build-prepare', 'compile']
            for i in range(1, 5):
                expected += [f'compare{i}-{stage}' for stage in ('prepare', 'dry-run', 'controller', 'report')]
            self.assertEqual(calls, expected)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual((result['timing_windows'], result['timing_rank_records'],
                              result['formal_optimizer_steps']), (144, 1152, 7200))
            self.assertEqual(result['arrival_schedule'], self.summary)
            self.assertTrue(result['identity_fallback'])
            self.assertEqual(result['arrival_options'], {'uncertainty_multiplier': 1.0})
            self.assertEqual(result['arrival_method'], 'receiver-tail-assignment-v3')
            self.assertTrue(result['exploratory'])
            for trial in ('compare1', 'compare2', 'compare3', 'compare4'):
                record = result['comparisons'][trial]
                self.assertEqual(record['status'], 'completed')
                self.assertEqual(record['report_sha256'], sha(root/trial/'report.json'))

    def test_two_trial_preview_is_read_only_and_omits_later_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(os, 'fork', side_effect=AssertionError('fork')), \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('process')), \
                mock.patch.object(batch, 'resolve_nccl', side_effect=AssertionError('NCCL lookup')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(tmp)/'new'
            flags = ['--out', str(root), '--trials', '2']
            batch.main(flags)
            self.assertFalse(root.exists())
            self.assertIn('72 formal windows / 576 rank records / 3600 formal steps', output.getvalue())
            self.assertNotIn('compare3', output.getvalue())
            self.assertNotIn('compare4', output.getvalue())
            steps = batch.commands(batch.arguments(flags))
            self.assertEqual([label for label, _ in steps if label.endswith('-controller')],
                             ['compare1-controller', 'compare2-controller'])
            for label, command in steps:
                if label.endswith('-prepare') and label != 'build-prepare':
                    self.assertEqual(command[command.index('--blocks')+1], '6')
                    self.assertEqual(command[command.index('--rounds')+1], '2')

    def test_two_trials_build_once_and_complete_exactly_two_strict_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            result = self.run_fake(self.args(root, build=True, trials=2), calls)
            expected = ['build-prepare', 'compile']
            for i in (1, 2):
                expected += [f'compare{i}-{stage}' for stage in ('prepare', 'dry-run', 'controller', 'report')]
            self.assertEqual(calls, expected)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['order'], ['compare1', 'compare2'])
            self.assertEqual(set(result['comparisons']), {'compare1', 'compare2'})
            self.assertEqual((result['timing_windows'], result['timing_rank_records'],
                              result['formal_optimizer_steps']), (72, 576, 3600))
            self.assertEqual(result['expected_timing_windows'], 72)
            self.assertEqual(result['expected_formal_optimizer_steps'], 3600)
            self.assertEqual(result['execution_timing'], {'warmup_steps': 10, 'timed_steps': 50})
            for trial in ('compare1', 'compare2'):
                record = result['comparisons'][trial]
                self.assertEqual(record['status'], 'completed')
                self.assertEqual(record['report_sha256'], sha(root/trial/'report.json'))
            self.assertFalse((root/'compare3').exists())
            self.assertFalse((root/'compare4').exists())

    def test_two_trials_reject_second_controller_failure_or_incomplete_report(self):
        for failure in ('controller', 'report'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(stage, folder, command):
                    if stage != 'compare2-'+failure:
                        return
                    if failure == 'controller':
                        raise subprocess.CalledProcessError(1, command)
                    result = read(folder/'report.json')
                    result['all_samples'] = result['all_samples'][:-1]
                    write(folder/'report.json', result)
                expected_error = subprocess.CalledProcessError if failure == 'controller' else ValueError
                with self.assertRaises(expected_error):
                    self.run_fake(self.args(root, trials=2), calls, corrupt)
                self.assertEqual(calls[-1], 'compare2-'+failure)
                self.assertEqual(read(root/'batch-state.json')['status'], 'failed')
                self.assertFalse((root/'compare3').exists())

    def test_trial_counts_default_to_four_and_name_outputs_without_accepting_other_counts(self):
        default = batch.arguments([])
        self.assertEqual(default.trials, 4)
        self.assertEqual(batch.trial_names(default), ('compare1', 'compare2', 'compare3', 'compare4'))
        self.assertIn('ablation7-new-4x-', default.out.name)
        twice = batch.arguments(['--trials', '2'])
        self.assertEqual(batch.trial_names(twice), ('compare1', 'compare2'))
        self.assertIn('ablation7-new-2x-', twice.out.name)
        for count in ('0', '1', '3', '5', '-1', '2.5'):
            with self.subTest(count=count), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                batch.arguments(['--trials', count])
            self.assertEqual(raised.exception.code, 2)

    def test_default_margin_one_and_explicit_four_are_forwarded_to_build(self):
        for flags, expected in (([], 1.0), (['--arrival-margin-multiplier', '4'], 4.0)):
            with self.subTest(expected=expected):
                args = batch.arguments(flags)
                self.assertEqual(args.arrival_margin_multiplier, expected)
                self.assertEqual(batch.requested_options(args), {'uncertainty_multiplier': expected})
                build_command = dict(batch.commands(args))['build-prepare']
                self.assertEqual(build_command[build_command.index('--arrival-margin-multiplier')+1], str(expected))

    def test_explicit_four_keeps_four_runs_and_is_recorded_as_nonexploratory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            result = self.run_fake(self.args(root, multiplier=4.0), calls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['arrival_options'], {'uncertainty_multiplier': 4.0})
            self.assertFalse(result['exploratory'])
            self.assertEqual(result['arrival_schedule'], self.summary_for(4.0))
            self.assertEqual(sum(stage.endswith('-controller') for stage in calls), 4)

    def test_invalid_margin_is_rejected_before_any_launch(self):
        for multiplier in ('0', '6.1', 'nan', 'inf'):
            with self.subTest(multiplier=multiplier), \
                    mock.patch.object(batch.launcher, 'detach', side_effect=AssertionError('detach')), \
                    self.assertRaises(ValueError):
                batch.arguments(['--arrival-margin-multiplier', multiplier])

    def test_failed_controller_stops_before_next_prepare_including_fourth(self):
        for index in (2, 4):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def fail(stage, folder, cmd):
                    if stage == f'compare{index}-controller':
                        raise subprocess.CalledProcessError(1, cmd)
                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_fake(self.args(root), calls, fail)
                self.assertEqual(calls[-1], f'compare{index}-controller')
                self.assertNotIn(f'compare{index}-report', calls)
                self.assertFalse((root/f'compare{index+1}').exists())
                self.assertEqual(read(root/'batch-state.json')['status'], 'failed')

    def test_invalid_fourth_report_never_marks_batch_complete(self):
        changes = (
            {'execution_timing': {'warmup_steps': 10, 'timed_steps': 60}},
            {'role_plan_sha256': {'changed': True}},
            {'measurement_identity': {'changed': True}},
            {'preflight_rank_records': 40},
            {'formal_optimizer_steps': 2160},
            {'evidence': {'preflight_sha256': 'stale'}},
            {'run_id': 'another-comparison'},
            {'arrival_schedule': {'changed': True}},
            {'arrival_options': {'uncertainty_multiplier': 4.0}},
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(stage, folder, cmd):
                    if stage == 'compare4-report':
                        write(folder/'report.json', dict(read(folder/'report.json'), **change))
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(root), calls, corrupt)
                self.assertEqual(calls[-1], 'compare4-report')
                self.assertEqual(read(root/'batch-state.json')['status'], 'failed')

    def test_fourth_run_is_distinct_and_uses_same_inputs(self):
        changes = (
            ('run_id', 'ablation7-new/compare2'),
            ('budget_fractions', {}),
            ('arrival_options', {'uncertainty_multiplier': 4.0}),
            ('build_files', {'drift': 'x'}),
            ('controllers', {'drift': 'x'}),
            ('files', {'scripts/worker.py': 'changed'}),
            ('protocol', 'h100-gpt67-tp8-ablation7-v1'),
            ('variants', {p: dict(protocol.definition(p), arrival_method='tail_residual')
                          for p in protocol.POLICIES}),
        )
        for key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(stage, folder, cmd):
                    if stage == 'compare4-prepare':
                        write(folder/'submission.json', dict(read(folder/'submission.json'), **{key: value}))
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(root), calls, corrupt)
                self.assertNotIn('compare4-controller', calls)
                self.assertEqual(read(root/'batch-state.json')['status'], 'failed')

    def test_fourth_preflight_margin_drift_is_rejected_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def corrupt(stage, folder, cmd):
                if stage == 'compare4-report':
                    gate = read(folder/'preflight.json')
                    gate['arrival_options'] = {'uncertainty_multiplier': 4.0}
                    write(folder/'preflight.json', gate)
                    result = read(folder/'report.json')
                    result['evidence']['preflight_sha256'] = sha(folder/'preflight.json')
                    write(folder/'report.json', result)
            with self.assertRaisesRegex(ValueError, 'arrival margin'):
                self.run_fake(self.args(root), calls, corrupt)
            self.assertEqual(calls[-1], 'compare4-report')
            self.assertEqual(read(root/'batch-state.json')['status'], 'failed')

    def test_build_failure_stops_before_any_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def fail(stage, folder, cmd):
                if stage == 'compile':
                    raise subprocess.CalledProcessError(1, cmd)
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_fake(self.args(root, build=True), calls, fail)
            self.assertEqual(calls, ['build-prepare', 'compile'])
            self.assertFalse((root/'compare1').exists())

    def test_reuse_omits_compilation_and_nccl_but_keeps_four_preflights(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            with mock.patch.object(batch, 'read', wraps=read) as reads:
                self.run_fake(self.args(Path(tmp)), calls)
            self.assertNotIn('compile', calls)
            self.assertEqual(sum(s.endswith('-controller') for s in calls), 4)
            self.assertFalse(any(str(call.args[0]).endswith('fused/manifest.json')
                                 for call in reads.call_args_list))

    def test_recorded_nccl_files_are_verified_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root, build=True)
            nccl = args.nccl_root
            args.build = root/'parent'
            (args.build/'fused').mkdir(parents=True)
            write(args.build/'fused/manifest.json', dict(nccl_files={
                str(nccl/name): sha(nccl/name) for name in ('include/nccl.h', 'lib/libnccl_static.a')}))
            args.nccl_root = None
            batch.resolve_nccl(args)
            self.assertEqual(args.nccl_root, nccl)
            (nccl/'lib/libnccl_static.a').write_text('changed')
            args.nccl_root = None
            calls = []
            with self.assertRaises(ValueError):
                self.run_fake(args, calls)
            self.assertEqual(calls, [])
            self.assertEqual(read(root/'batch-state.json')['status'], 'failed')

    def test_fourth_arrival_order_drift_stops_before_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def order_summary(folder):
                summary = copy.deepcopy(self.summary)
                if folder.name == 'compare4':
                    summary['attn'].update(changed_per_rank=[2]+[0]*7, identical_to_A=False)
                return summary
            with mock.patch.object(batch.launcher, 'invoke', side_effect=self.fake_invoke(calls)), \
                    mock.patch.object(batch, 'order_summary', side_effect=order_summary), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                batch.supervise(self.args(root))
            self.assertEqual(calls[-1], 'compare4-dry-run')
            self.assertNotIn('compare4-controller', calls)
            self.assertEqual(read(root/'batch-state.json')['status'], 'failed')

    def plan(self, changed=False, multiplier=1.0):
        coords = [[[0, 0], [0, 1], [1, 0], [1, 1]] for _ in range(8)]
        plan = dict(protocol=protocol.PROTOCOL, variants={}, orders={})
        if multiplier is not None:
            plan['arrival_options'] = {'uncertainty_multiplier': multiplier}
        for policy in protocol.POLICIES:
            roles = {}
            for role in protocol.SHAPES:
                role_coords = copy.deepcopy(coords)
                if changed and policy == protocol.B and role == 'attn':
                    role_coords[0][:2] = role_coords[0][:2][::-1]
                roles[role] = dict(entry=dict(coords=role_coords))
            plan['variants'][policy] = dict(definition=protocol.definition(policy), roles=roles)
        for role in protocol.SHAPES:
            selected = changed and role == 'attn'
            diagnostic = dict(method='receiver-tail-assignment-v2', identity_fallback=not selected,
                              accepted_windows=int(selected), max_displacement=1,
                              groups=[dict(status='candidate' if selected else 'identity')])
            if multiplier is not None:
                diagnostic.update(method='receiver-tail-assignment-v3',
                                  arrival_options={'uncertainty_multiplier': multiplier},
                                  exploratory=multiplier < 4, conditional_error_multiplier=4.0,
                                  accepted_windows_with_positive_conservative_bound=int(selected))
                if selected:
                    diagnostic['groups'][0]['candidate'] = dict(conservative_gain_lower_bound_us=[0.5, 0.6])
            plan['orders'][role] = dict(entry=dict(arrival_order=diagnostic), entry_sha256='fitted-'+role,
                                       training_passes=[0, 1], heldout_passes=[2])
        return plan

    def test_order_summary_reads_physical_order_and_keeps_identity_visible(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'scripts').mkdir()
                write(root/'scripts/selection-plan.json', self.plan(changed))
                summary = batch.order_summary(root)
                self.assertEqual(summary['attn']['changed_per_rank'], [2 if changed else 0]+[0]*7)
                self.assertEqual(summary['attn']['identical_to_A'], not changed)
                self.assertEqual(summary['attn']['accepted_windows'], int(changed))
                self.assertEqual(summary['attn']['method'], 'receiver-tail-assignment-v3')
                self.assertEqual(summary['attn']['arrival_options'], {'uncertainty_multiplier': 1.0})
                self.assertTrue(summary['mlp']['identical_to_A'])

    def test_order_summary_rejects_old_method_or_changed_holdout(self):
        for change in ('method', 'legacy_method', 'training', 'heldout'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'scripts').mkdir()
                plan = self.plan()
                order = plan['orders']['attn']
                if change == 'method':
                    order['entry']['arrival_order']['method'] = 'tail_residual'
                elif change == 'legacy_method':
                    order['entry']['arrival_order']['method'] = 'receiver-tail-assignment-v2'
                elif change == 'training':
                    order['training_passes'] = [0, 2]
                else:
                    order['heldout_passes'] = [1]
                write(root/'scripts/selection-plan.json', plan)
                with self.assertRaises(ValueError):
                    batch.order_summary(root)

    def test_legacy_order_summary_remains_readable_without_configured_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scripts').mkdir()
            write(root/'scripts/selection-plan.json', self.plan(multiplier=None))
            summary = batch.order_summary(root)
            self.assertTrue(all(record['method'] == 'receiver-tail-assignment-v2' for record in summary.values()))
            self.assertTrue(all('arrival_options' not in record for record in summary.values()))

    def test_main_rejects_existing_output_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root/'parent'
            parent.mkdir()
            write(parent/'ablation4-manifest.json', {})
            reused = root/'built-new'
            self.write_reused_build(reused, parent)
            output = root/'existing'
            output.mkdir()
            evidence = output/'batch-state.json'
            evidence.write_text('preserve existing evidence\n')
            with mock.patch.object(batch, 'supervise', side_effect=AssertionError('supervisor')), \
                    mock.patch.object(batch.launcher, 'detach', side_effect=AssertionError('detach')), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                batch.main(['--out', str(output), '--build', str(parent), '--ablation7-build', str(reused),
                            '--python', sys.executable, '--execute'])
            self.assertEqual(evidence.read_text(), 'preserve existing evidence\n')

    def test_execute_dispatches_detached_supervisor_and_foreground_is_explicit(self):
        for foreground in (False, True):
            with self.subTest(foreground=foreground), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parent = root/'parent'
                parent.mkdir()
                write(parent/'ablation4-manifest.json', {})
                reused = root/'built-new'
                self.write_reused_build(reused, parent)
                output = root/'fresh'
                flags = ['--out', str(output), '--build', str(parent), '--ablation7-build', str(reused),
                         '--python', sys.executable, '--execute']
                if foreground:
                    flags += ['--foreground']
                with mock.patch.object(batch, 'supervise') as supervise, \
                        mock.patch.object(batch.launcher, 'detach', return_value={'ready': True}) as detach, \
                        contextlib.redirect_stdout(io.StringIO()):
                    batch.main(flags)
                self.assertTrue((output/'handoff/launch.json').is_file())
                if foreground:
                    supervise.assert_called_once()
                    detach.assert_not_called()
                else:
                    supervise.assert_not_called()
                    detach.assert_called_once()
                    self.assertIs(detach.call_args.kwargs['supervisor'], supervise)

    def test_reused_build_must_be_new_protocol_and_same_parent(self):
        cases = (
            ('ablation7-manifest.json', None, None),
            ('ablation7-new-manifest.json', 'h100-gpt67-tp8-ablation7-v1', None),
            ('ablation7-new-manifest.json', protocol.PROTOCOL, '/wrong-parent'),
        )
        for name, proto, parent_override in cases:
            with self.subTest(name=name, proto=proto, parent_override=parent_override), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parent = root/'parent'
                parent.mkdir()
                write(parent/'ablation4-manifest.json', {})
                reused = root/'built'
                reused.mkdir()
                write(reused/name, dict(parent_build=parent_override or str(parent), protocol=proto,
                                       source_only=False, arrival_options={'uncertainty_multiplier': 1.0}))
                write(reused/'plans.json', dict(arrival_options={'uncertainty_multiplier': 1.0}))
                output = root/'fresh'
                with mock.patch.object(batch.launcher, 'detach', side_effect=AssertionError('detach')), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises((ValueError, FileNotFoundError)):
                    batch.main(['--out', str(output), '--build', str(parent), '--ablation7-build', str(reused),
                                '--python', sys.executable, '--execute'])
                self.assertFalse(output.exists())

    def test_reuse_rejects_legacy_or_different_options_before_output_creation(self):
        cases = (
            ('legacy', None, None),
            ('different_margin', 4.0, 4.0),
            ('legacy_plan', 1.0, None),
            ('manifest_plan_disagree', 1.0, 4.0),
        )
        for label, manifest_multiplier, plan_multiplier in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parent = root/'parent'
                parent.mkdir()
                write(parent/'ablation4-manifest.json', {})
                reused = root/'built'
                reused.mkdir()
                manifest = dict(parent_build=str(parent), protocol=protocol.PROTOCOL, source_only=False)
                plans = {}
                if manifest_multiplier is not None:
                    manifest['arrival_options'] = {'uncertainty_multiplier': manifest_multiplier}
                if plan_multiplier is not None:
                    plans['arrival_options'] = {'uncertainty_multiplier': plan_multiplier}
                write(reused/'ablation7-new-manifest.json', manifest)
                write(reused/'plans.json', plans)
                output = root/'fresh'
                with mock.patch.object(batch.launcher, 'detach', side_effect=AssertionError('detach')), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'arrival margin'):
                    batch.main(['--out', str(output), '--build', str(parent), '--ablation7-build', str(reused),
                                '--python', sys.executable, '--execute'])
                self.assertFalse(output.exists())

    def test_supervisor_rechecks_reuse_options_before_any_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            manifest = read(args.ablation7_build/'ablation7-new-manifest.json')
            manifest['arrival_options'] = {'uncertainty_multiplier': 4.0}
            write(args.ablation7_build/'ablation7-new-manifest.json', manifest)
            calls = []
            with self.assertRaisesRegex(ValueError, 'arrival margin'):
                self.run_fake(args, calls)
            self.assertEqual(calls, [])
            self.assertEqual(read(root/'batch-state.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
