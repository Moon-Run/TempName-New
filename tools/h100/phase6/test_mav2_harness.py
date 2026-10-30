"""CPU protocol/admission tests; fixture records are explicitly not GPU results."""
import ast
import contextlib
import copy
import gzip
import io
from itertools import permutations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation6_build as flux_builder
import mav2 as cli
import mav2_build as codec_builder
import mav2_prepare as preparation
import mav2_protocol as routing
import mav2_report as report
import mav2_sources as sources
import scope_run
import test_basic_quant as fixture
from common import read, write


def profile_fixture(policy):
    events = fixture.profile_fixture('basic')
    if policy != 'basic':
        changes = {'fused_ash_compress_kernel': 'mav2_encode_kernel',
                   'basic_quant_decode_reduce_kernel': 'mav2_decode_reduce_kernel'}
        for event in events:
            event['name'] = changes.get(event['name'], event['name'])
    return events


def make_window(root, info, job):
    # Reuse the older integration fixture for the unchanged model/TP contract;
    # rewrite only new policy, numerical and observed transform evidence.
    with mock.patch.object(fixture, 'sp', routing):
        fixture.make_window(root, info, dict(job, policy='basic'))
    folder = root/'results'/job['directory']
    write(folder/'command.json', dict(job, scope=routing.definition(job['policy']),
          communication=info['communication'], shared_environment=preparation.SHARED_ENVIRONMENT))
    for rank in range(8):
        row = read(folder/f'rank{rank}.json')
        policy, mode = job['policy'], job['mode']
        row.update(policy=policy, scope_definition=routing.definition(policy),
                   # Keep both positive and negative speed results in one full report.
                   elapsed_seconds=row['elapsed_seconds']*{'basic': 1., 'mav2': .9, 'd8': 1.1}[policy],
                   losses=[1.01 if policy != 'basic' else 1.]*len(row['losses']),
                   transform_diagnostics={})
        row.update(info['variants'][policy]['libraries'])
        if mode in ('smoke', 'profile'):
            total = 7*256*32
            direct = {'basic': 0, 'mav2': 100, 'd8': total}[policy]
            row['transform_diagnostics'] = {name: dict(policy=policy, calls=calls, sampled_calls=1,
                total=total, d8=direct, h8=total-direct, nonfinite=0, invalid_tags=0,
                bytes_per_group=136, selector={'relative_l2': .025, 'max_abs_over_amax': .0625},
                quantized_remote_fraction=1.0, d8_fraction=direct/total,
                quality_selection=policy == 'mav2', h8_fallback_allowed=policy == 'mav2')
                for name, calls in row['codec_calls'].items()}
        write(folder/f'rank{rank}.json', row)
        if mode == 'profile':
            with gzip.open(folder/f'rank{rank}-profile.json.gz', 'wt') as handle:
                json.dump(dict(traceEvents=profile_fixture(policy)), handle)


class ProtocolTests(unittest.TestCase):
    def test_balance_and_full_remote_contract(self):
        jobs = routing.schedule()
        self.assertEqual(len(jobs), 36)
        blocks = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b]
                  for r in (1, 2) for b in range(6)]
        self.assertEqual(blocks[6:], [x[::-1] for x in blocks[:6][::-1]])
        self.assertEqual({tuple(b) for b in blocks[:6]}, set(permutations(routing.POLICIES)))
        for position in range(len(routing.POLICIES)):
            for policy in routing.POLICIES:
                self.assertEqual(sum(b[position] == policy for b in blocks), 4)
        with self.assertRaises(ValueError): routing.schedule(blocks=8)
        for policy in routing.POLICIES:
            d = routing.definition(policy)
            self.assertEqual(d['quantization'], 'all_remote')
            self.assertFalse(d['arrival'] or d['prefetch'] or d['graph'])
        self.assertFalse(routing.TRANSFORM['bf16_fallback'])
        self.assertEqual(routing.EQUIVALENCE_PAIRS, [])
        with self.assertRaises(ValueError): routing.selection_entry(None)

    def test_real_source_generation_and_candidate_oracle(self):
        for fn in (sources.adapter_source, sources.taco_source, sources.codec_source, sources.worker_source):
            ast.parse(fn())
        source = sources.taco_source()
        self.assertIn('from mav2_reference import roundtrip as selective_roundtrip', source)
        self.assertIn("'d8': direct_roundtrip", source)
        self.assertIn('budget_vs_bf16=.05,budget_vs_codec=.01', source)
        worker = sources.worker_source()
        self.assertIn('H100_MAV2_EXECUTE', worker)
        self.assertIn("if MODE in ('smoke', 'profile') else {}", worker)
        self.assertLess(worker.index('torch.cuda.set_device(local_rank)'), worker.index('initialize_megatron()'))

    def test_profile_requires_declared_native_paths(self):
        for policy in routing.POLICIES:
            self.assertEqual(report.profile_roles(profile_fixture(policy), policy)['attn']['calls'], 128)
            other = 'mav2' if policy == 'basic' else 'basic'
            with self.assertRaises(ValueError): report.profile_roles(profile_fixture(other), policy)
            for removed in ('ncclDevKernel_SendRecv', 'flux_bf16_sm90_commnone',
                            'mav2_encode_kernel' if policy != 'basic' else 'fused_ash_compress_kernel'):
                with self.assertRaises(ValueError):
                    report.profile_roles([e for e in profile_fixture(policy) if e['name'] != removed], policy)

    def test_real_projection_oracle_routes_each_policy(self):
        tree = ast.parse(sources.taco_source())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                        node.name == 'masked_roundtrip')
        namespace = {'os': os, 'roundtrip': lambda x: ('H8', {}),
                     'selective_roundtrip': lambda x: ('selected', {}),
                     'direct_roundtrip': lambda x: ('D8', {})}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<generated-oracle>', 'exec'), namespace)
        for policy, expected in [('basic', 'H8'), ('mav2', 'selected'), ('d8', 'D8')]:
            with mock.patch.dict(os.environ, E2E_POLICY=policy):
                self.assertEqual(namespace['masked_roundtrip'](object(), None, 0, 1, 2048, 4096, 8), expected)

    def test_quality_accepts_rounding_but_rejects_degradation(self):
        base = [dict(losses=[1., .00001], grad_norms=[1., .5], initial_loss=5., warmup_last_loss=2.) for _ in range(8)]
        changed = copy.deepcopy(base)
        changed[0]['losses'][-1] = .001
        good = report.quality_pair(base, changed)
        self.assertTrue(good['passed'])
        self.assertEqual(good['ranks'][0]['fields']['losses']['final_ratio'], 100.)
        changed[0]['grad_norms'][0] = 2.
        self.assertFalse(report.quality_pair(base, changed)['passed'])
        changed[0]['grad_norms'][0] = float('nan')
        with self.assertRaises(ValueError): report.quality_pair(base, changed)

    def test_returned_profile_failure_reports_actual_worst_point(self):
        # rank0 values from the 20261031-010411 returned H100 profile.
        base = [dict(losses=[8.973113059997559], grad_norms=[0.2716255507018137],
                     initial_loss=11.255111694335938, warmup_last_loss=9.039176940917969)]*8
        candidate = [dict(losses=[13.185644149780273], grad_norms=[71.15936875576364],
                          initial_loss=11.255816459655762, warmup_last_loss=8.886245727539062)]*8
        quality = report.quality_pair(base, candidate)
        self.assertFalse(quality['passed'])
        field = quality['ranks'][0]['fields']['grad_norms']
        self.assertEqual(field['failed_points'], 1)
        self.assertAlmostEqual(field['max_budget_fraction'], 557.4576821450204)
        message = report.quality_failure_summary({'mav2': {'profile': quality}})
        self.assertIn('mav2/profile/grad_norms: rank=0, index=0', message)
        self.assertIn('candidate=71.1594', message)
        self.assertNotIn('initial_loss:', message)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='mav2-harness-CPU-FIXTURE-')
        cls.root = Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.flux = flux_builder.prepare(cls.root/'flux', source_only=True)
            cls.codec = codec_builder.prepare(cls.root/'codec', source_only=True)
            cls.compare = preparation.prepare(cls.root/'compare', cls.flux, cls.codec,
                                              Path('/absent'), source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_frozen_inputs_preview_and_source_only_rejection(self):
        with mock.patch.object(preparation, 'capture_hardware', side_effect=AssertionError('GPU query')), \
                contextlib.redirect_stdout(io.StringIO()):
            info = report.verify_inputs(self.compare)
            self.assertEqual(len(cli.run(self.compare)), 42)
            self.assertEqual(list(scope_run.mapping_jobs(self.compare, info, routing)), [])
            with self.assertRaises(ValueError): cli.run(self.compare, execute=True)
        result = subprocess.run([sys.executable, str(self.compare/'scripts/worker.py')],
                                capture_output=True, text=True, env=dict(os.environ, H100_MAV2_EXECUTE='1'))
        self.assertIn('Source-only', result.stderr)
        names = {p.name for p in preparation.controller_paths()}
        self.assertTrue({'mav2_reference.py', 'mav2_validation.py', 'mav2_codec.py',
                         'mav2_report.py', 'detach_mav2.py'}.issubset(names))

    def test_threshold_and_source_drift_rejected(self):
        path, saved = self.compare/'submission.json', read(self.compare/'submission.json')
        try:
            write(path, dict(saved, quality_budget=dict(saved['quality_budget'], loss_atol=999)))
            with self.assertRaises(ValueError): report.verify_inputs(self.compare)
        finally:
            write(path, saved)
        try:
            write(path, dict(saved, allow_quality_failure=True))
            with self.assertRaises(ValueError): report.verify_inputs(self.compare)
        finally:
            write(path, saved)
        path = self.compare/'scripts/mav2_reference.py'
        saved = path.read_text()
        try:
            path.write_text(saved+'\n# changed\n')
            with self.assertRaises(ValueError): report.verify_inputs(self.compare)
        finally:
            path.write_text(saved)

    def test_worker_validation_dependency_closure(self):
        scripts = self.compare/'scripts'
        # Isolated interpreter has no phase6 source directory on sys.path.
        code = ('import sys; sys.path.insert(0, sys.argv[1]); '
                'import mav2_validation, mav2_reference, megatron_taco; '
                'print(mav2_validation.__file__)')
        proc = subprocess.run([sys.executable, '-I', '-c', code, str(scripts)],
                              cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(str(scripts), proc.stdout)

    def test_explicit_timing_rejects_failed_preflight_before_worker(self):
        root = preparation.prepare(self.root/'failed-gate', self.flux, self.codec,
                                   Path('/absent'), source_only=True)
        info = read(root/'submission.json')
        info.update(executable=True, source_only=False)
        from common import sha
        write(root/'preflight.json', dict(completed=True, correctness_passed=False,
              quality_within_pairs=False, execution_path_validated=True,
              allow_quality_failure=False, quality_by_candidate={'mav2': False, 'd8': False},
              submission_sha256=sha(root/'submission.json')))
        with mock.patch.object(report, 'verify_inputs', return_value=info), \
                mock.patch.object(cli, 'run_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'passing numerical/path preflight'):
                cli.run(root, stage='timing', execute=True)
            worker.assert_not_called()

    def test_opt_in_runs_all_windows_and_keeps_preflight_failure_in_final_report(self):
        root = preparation.prepare(self.root/'QUALITY-FAILURE-CPU-FIXTURE', self.flux, self.codec,
                                   Path('/absent'), source_only=True, allow_quality_failure=True)
        info = report.verify_inputs(root)
        info.update(executable=True, source_only=False, hardware=fixture.HARDWARE,
                    calibration_hardware=fixture.HARDWARE, codec_library=fixture.LIBRARY)
        for policy, variant in info['variants'].items():
            variant['libraries'] = {name: dict(path='/CPU-FIXTURE/'+policy+'/'+name, sha256='f'*64)
                                    for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
        write(root/'submission.json', info)
        write(root/'start-gpu-check.json', dict(passed=True, run_id=info['run_id'], synthetic=True))
        generated = self.root/'mock-worker-output'
        generated.mkdir()
        (generated/'scripts').symlink_to(root/'scripts', target_is_directory=True)
        seen = []

        def worker(command, megatron, env, log, timeout):
            folder = Path(env['E2E_PROCESS_OUT'])
            job = read(folder/'command.json')
            seen.append(job['directory'])
            make_window(generated, info, job)
            shutil.copytree(generated/'results'/job['directory'], folder, dirs_exist_ok=True)
            if job['mode'] == 'profile' and job['policy'] == 'mav2':
                for rank in range(8):
                    path = folder/f'rank{rank}.json'
                    write(path, dict(read(path), grad_norms=[71.15936875576364]))

        with mock.patch.object(report, 'verify_inputs', return_value=info), \
                mock.patch.object(report.baseline_codec_validation, 'verify'), \
                mock.patch.object(cli, 'check_start'), \
                mock.patch.object(cli, 'environment', return_value={}), \
                mock.patch.object(cli, 'run_worker', side_effect=worker), \
                contextlib.redirect_stdout(io.StringIO()):
            result = cli.run(root, execute=True)
            self.assertEqual(len(seen), 42)
            self.assertEqual(len(result['all_samples']), 36)
            self.assertFalse(result['correctness_passed'] or result['quality_within_pairs'])
            self.assertEqual(result['quality_by_candidate'], {'mav2': False, 'd8': True})
            self.assertTrue(all(q['passed'] for q in result['quality']))
            self.assertFalse(result['candidate_acceptance']['mav2']['benefit_confirmed'])
            self.assertFalse(result['comparison_acceptance']['d8/mav2']['quality_passed'])
            self.assertEqual(read(root/'all-state.json')['status'], 'completed_with_quality_failure')
            self.assertIn('质量门禁未通过', (root/'report.md').read_text())
            gate = read(root/'preflight.json')
            report.admit_preflight(root, info, gate)
            with self.assertRaises(ValueError):
                report.admit_preflight(root, dict(info, allow_quality_failure=False), gate)
            with self.assertRaises(ValueError):
                report.admit_preflight(root, info, dict(gate, execution_path_validated=False))
            # Opt-in never permits missing codec evidence or nonfinite trajectories.
            with mock.patch.object(report.baseline_codec_validation, 'verify',
                                   side_effect=ValueError('codec evidence rejected')):
                with self.assertRaisesRegex(ValueError, 'codec evidence rejected'):
                    report.preflight(root, offline=True)
            path = root/'results/smoke-mav2/rank0.json'
            row = read(path)
            write(path, dict(row, losses=[float('nan')]*len(row['losses'])))
            with self.assertRaises(ValueError): report.preflight(root, offline=True)

    def test_forced_d8_sample_rejects_h8_or_selector_fallback(self):
        root = preparation.prepare(self.root/'d8-observed-route', self.flux, self.codec,
                                   Path('/absent'), source_only=True)
        info = read(root/'submission.json')
        info.update(executable=True, source_only=False, hardware=fixture.HARDWARE,
                    calibration_hardware=fixture.HARDWARE, codec_library=fixture.LIBRARY)
        for policy, variant in info['variants'].items():
            variant['libraries'] = {name: dict(path='/CPU-FIXTURE/'+policy+'/'+name, sha256='f'*64)
                                    for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
        write(root/'submission.json', info)
        job = next(j for j in scope_run.jobs(info, 'all', routing) if j['policy'] == 'd8' and j['mode'] == 'smoke')
        make_window(root, info, job)
        folder = root/'results'/job['directory']
        path = folder/'rank0.json'
        saved = read(path)
        with mock.patch.object(report.baseline_codec_validation, 'verify'):
            report.window(root, folder, 'd8', 'smoke')
            for corruption in ('h8', 'selector', 'policy'):
                row = copy.deepcopy(saved)
                sample = next(iter(row['transform_diagnostics'].values()))
                if corruption == 'h8':
                    sample.update(d8=sample['total']-1, h8=1, d8_fraction=(sample['total']-1)/sample['total'])
                elif corruption == 'selector':
                    sample['quality_selection'] = True
                else:
                    sample['policy'] = 'mav2'
                write(path, row)
                with self.assertRaises(ValueError): report.window(root, folder, 'd8', 'smoke')
            write(path, saved)

    def test_complete_report_keeps_negative_result_and_quality_failure(self):
        root = preparation.prepare(self.root/'SYNTHETIC-NO-GPU', self.flux, self.codec,
                                   Path('/absent'), source_only=True)
        info = read(root/'submission.json')
        info.update(executable=True, source_only=False, hardware=fixture.HARDWARE,
                    calibration_hardware=fixture.HARDWARE, codec_library=fixture.LIBRARY)
        for policy, variant in info['variants'].items():
            variant['libraries'] = {name: dict(path='/CPU-FIXTURE/'+policy+'/'+name, sha256='f'*64)
                                    for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
        write(root/'submission.json', info)
        write(root/'start-gpu-check.json', dict(passed=True, run_id=info['run_id'], synthetic=True))
        for job in scope_run.jobs(info, 'all', routing): make_window(root, info, job)
        # Native codec validation has its own tests. Only build admission and its
        # GPU-only evidence verifier are bypassed here, never training/path gates.
        with mock.patch.object(report, 'verify_inputs', return_value=info), \
                mock.patch.object(report.baseline_codec_validation, 'verify') as codec_verify:
            gate = report.preflight(root, offline=True)
            self.assertTrue(gate['quality_within_pairs'])
            self.assertFalse(gate['bitwise_equal_within_pairs'])
            self.assertEqual(codec_verify.call_count, 24)
            result = report.report(root, offline=True)
            self.assertEqual(result['formal_optimizer_steps'], 1800)
            self.assertEqual(result['timing_rank_records'], 288)
            self.assertEqual(result['preflight_rank_records'], 48)
            self.assertEqual(result['profile_traces'], 24)
            self.assertEqual(result['transform_summary']['mav2']['attn']['d8'], 8*32*100)
            self.assertEqual(result['transform_summary']['d8']['attn']['d8_fraction'], 1.)
            self.assertTrue(result['completed'] and result['quality_within_pairs'])
            self.assertTrue(result['expected_order_point_estimate'] and result['expected_order_confirmed'])
            self.assertEqual(set(result['comparison_acceptance']), {'mav2/basic', 'd8/basic', 'd8/mav2'})
            self.assertTrue(result['candidate_acceptance']['mav2']['benefit_confirmed'])
            self.assertFalse(result['candidate_acceptance']['d8']['positive_point_estimate'])
            self.assertEqual(result, report.report(root, offline=True, save=False))
            job = next(j for j in info['schedule'] if j['policy'] == 'd8')
            path = root/'results'/job['directory']/'rank0.json'
            row = read(path)
            write(path, dict(row, grad_norms=[2.]*50))
            failed = report.report(root, offline=True)
            self.assertFalse(failed['quality_within_pairs'] or failed['correctness_passed'])
            self.assertEqual(failed['quality_by_candidate'], {'mav2': True, 'd8': False})
            self.assertTrue(failed['candidate_acceptance']['mav2']['benefit_confirmed'])
            self.assertFalse(failed['comparison_acceptance']['d8/mav2']['quality_passed'])
            self.assertEqual(len(failed['all_samples']), 36)
            write(path, dict(row, codec_calls={}))
            with self.assertRaises(ValueError): report.report(root, offline=True)


if __name__ == '__main__':
    unittest.main()
