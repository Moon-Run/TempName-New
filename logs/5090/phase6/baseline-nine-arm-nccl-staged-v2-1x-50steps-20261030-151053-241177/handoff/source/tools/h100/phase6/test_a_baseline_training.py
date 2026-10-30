"""CPU-only admission, source routing and synthetic report rejection checks.

The synthetic records never pass executable build admission and are not GPU data.
"""
import ast
import contextlib
import copy
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import a_baseline as cli
import a_baseline_build as fused_builder
import a_baseline_prepare as preparation
import a_baseline_protocol as sp
import a_baseline_report as report
import a_baseline_training_sources as sources
import ablation6_build as flux_builder
import baseline_codec_build as codec_builder
import baseline_codec_validation as validator
import scope_run
import test_baseline as previous_tests
from common import read, sha, write

LIBRARY = dict(path='/CPU-FIXTURE/libbaseline_taco.so', sha256='a'*64)
HARDWARE = dict(host='CPU-FIXTURE', devices=[dict(name='NVIDIA H100 CPU FIXTURE', uuid=f'fixture-{r}',
                sm_count=132, capability=[9, 0]) for r in range(8)])


def validation_record():
    return dict(passed=True, synthetic=False, library=LIBRARY, retained_outputs_unchanged=True,
                cases=[dict(case=case, shape=[1792, 4096] if case == 'full_shape' else [56, 128], chunks=7,
                            relative_l2=.02, protocol_relative_l2=.001, packet_relative_l2=.001,
                            budget_vs_bf16=.05, budget_vs_codec=.01, budget_vs_packet=.005,
                            finite=True, passed=True) for case in validator.CASES])


def profile_fixture(policy):
    events = []
    for number in range(256):
        role = 'attn' if number < 128 else 'mlp'
        route = 'flux_tensor' if policy == 'cuda' else 'candidate'
        base = number*1000
        scope = dict(cat='user_annotation', ph='X', name=f'scope.{role}.{route}',
                     pid=1, tid=1, ts=base, dur=100)
        events += [scope, dict(scope, cat='gpu_user_annotation', pid=0, tid=7)]
        def kernel(name, i):
            correlation = number*20+i+1
            events.append(dict(cat='cuda_runtime', ph='X', name='cudaLaunchKernel', pid=1, tid=1,
                               ts=base+5+i*8, dur=1, args={'correlation': correlation}))
            events.append(dict(cat='kernel', ph='X', name=name, ts=base+200+i*8, dur=2,
                               args={'correlation': correlation, 'stream': 7}))
        if policy == 'cuda':
            names = ['flux_bf16_sm90_commnone', 'fused_ash_compress_kernel',
                     'ncclDevKernel_SendRecv', 'fused_ash_decompress_kernel']
            for name in ('ablation6.flux_gemm_only', 'megatron_taco.encode', 'megatron_taco.exchange', 'megatron_taco.decode_reduce'):
                events.append(dict(cat='user_annotation', ph='X', name=name, pid=1, tid=1, ts=base+2, dur=75))
            events.append(dict(cat='cpu_op', ph='X', name='c10d::alltoall_base_', pid=1, tid=1, ts=base+25, dur=3))
        else:
            names = ['flux_bf16_sm80_reducescatter', 'CudaIpcBarrierAllKernel']
            names += ['taco_basic_decode_kernel', 'taco_basic_reduce_kernel'] if policy == 'basic' else ['taco_decode_ring_kernel']
            names += ['CudaIpcBarrierAllKernel']
            events.append(dict(cat='cpu_op', ph='X', name='aten::clone', pid=1, tid=1, ts=base+75, dur=10))
        for i, name in enumerate(names):
            kernel(name, i)
    return events


def make_window(root, info, job):
    policy, mode = job['policy'], job['mode']
    folder = root/'results'/job['directory']
    folder.mkdir(parents=True)
    cfg = read(root/'scripts/config.json')
    count = cfg['smoke_steps'] if mode == 'smoke' else 1 if mode == 'profile' else cfg['timed_steps']
    warmup = cfg['smoke_warmup_steps'] if mode == 'smoke' else cfg['warmup_steps']
    observed = {f'layers.{i}.{suffix}': dict(M=2048, N=4096, K_global=k, K_local=k//8)
                for i in range(32) for suffix, k in [('self_attention.linear_proj', 4096), ('mlp.linear_fc2', 16384)]}
    routes = sp.module_routes(observed, policy)
    names = sorted(routes)
    write(folder/'command.json', dict(job, scope=sp.definition(policy), communication=info['communication']))
    for rank in range(8):
        hardware = dict(HARDWARE, local_rank=rank)
        row = dict(protocol=sp.PROTOCOL, completed=True, passed=True, skips=0, fallback_calls=0,
                   run_id=info['run_id'], policy=policy, mode=mode, scope_definition=sp.definition(policy),
                   role_plan_sha256=cfg['role_plan_sha256'], rank=rank, local_rank=rank, tp_rank=rank, dp_rank=0,
                   tp_group_ranks=list(range(8)), dp_group_ranks=[rank], world_size=8, hardware=hardware,
                   gpu_uuid=hardware['devices'][rank]['uuid'], model=dict(cfg['common'], **cfg['cases'][0]),
                   padded_vocab_size=33792, block=job['block'], window=job['index'],
                   timing_protocol='full-step-one-clear-v2', sampling=False, profiling=mode == 'profile',
                   losses=[1.]*count, grad_norms=[1.]*count, initial_loss=1., warmup_last_loss=1.,
                   warmup_steps=warmup, timed_steps=count,
                   # Deliberately contradict the hoped-for performance order.
                   elapsed_seconds=count*{'basic': .6, 'improved': .7, 'cuda': .5}[policy]*(1 if rank == 7 else .9),
                   observed_shapes=observed, module_policies=routes, post_audit_routes=routes,
                   target_modules=names, warmup_calls={n: warmup*4 for n in names}, timed_calls={n: count*4 for n in names},
                   enabled_modules=names, replaced_modules=64, independent_scope_instances=64,
                   forward_preserved={n: False for n in names},
                   codec_placements={n: 'tensor' if policy == 'cuda' else 'fused' for n in names},
                   selection_checks={}, forward_checks={}, workspace_checks={},
                   initial_parameters_sha256='init'+str(rank), initial_rng_sha256='rng'+str(rank), tokens_sha256='tokens',
                   parameter_count_local=123, torch='CPU-FIXTURE', cuda='CPU-FIXTURE', nccl='CPU-FIXTURE', python='CPU-FIXTURE',
                   communication=info['communication'], shared_environment=preparation.SHARED_ENVIRONMENT,
                   codec_identity=info['codec_identity'], codec_library=LIBRARY if policy == 'cuda' else None,
                   codec_calls={n: (warmup+count)*4 for n in names} if policy == 'cuda' else {},
                   codec_validation=validation_record() if policy == 'cuda' and mode == 'smoke' else None,
                   smoke_output_sha256={n: 'a'*64 for n in names} if mode == 'smoke' else {},
                   retained_output_checks={})
        row.update(info['variants'][policy]['libraries'])
        for i, name in enumerate(names):
            role = sp.module_role(name)
            row['selection_checks'][name] = dict(role=role, shape=sp.SHAPES[role], bf16_wire_bytes=7*256*4096*2,
                                                mask_sha256=None, plan_sha256=None, wire_bytes=7*256*32*136)
            row['forward_checks'][name] = dict(passed=True, max_abs=0., relative_l2=0., protocol_relative_l2=0.,
                                              budget_vs_bf16=.05, budget_vs_codec=.01)
            field = 'gemm_output_ptr' if policy == 'cuda' else 'packet_ptr'
            row['workspace_checks'][name] = [dict(shape=sp.SHAPES[role], **{field: 1000+i})]
        if mode == 'smoke':
            row['retained_output_checks'] = {r: dict(module=next(n for n in names if sp.module_role(n) == r), outputs=6,
                independent_storage=True, unchanged_after_subsequent_call=True, passed=True) for r in sp.SHAPES}
        write(folder/f'rank{rank}.json', row)
        if mode == 'profile':
            with gzip.open(folder/f'rank{rank}-profile.json.gz', 'wt') as handle:
                json.dump(dict(traceEvents=profile_fixture(policy)), handle)


def make_mappings(root):
    for policy in ('basic', 'improved'):
        for role in sp.SHAPES:
            with (root/'results'/f'mapping-{policy}-{role}.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=['rank', 'index', 'm', 'n', 'destination', 'base_m', 'base_n', 'sm_occupancy'])
                writer.writeheader()
                for rank in range(8):
                    for i in range(512):
                        m = (i//32+2*rank)%16
                        writer.writerow(dict(rank=rank, index=i, m=m, n=i%32, destination=m//2,
                                             base_m=i//32, base_n=i%32, sm_occupancy=2))


class ProtocolTests(unittest.TestCase):
    def test_balance_same_scope_no_optional_optimizations(self):
        jobs = sp.schedule()
        self.assertEqual(len(jobs), 36)
        rows = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b] for r in (1, 2) for b in range(6)]
        self.assertEqual(rows[6:], [r[::-1] for r in rows[:6][::-1]])
        for position in range(3):
            self.assertEqual({p: sum(row[position] == p for row in rows[:6]) for p in sp.POLICIES},
                             {p: 2 for p in sp.POLICIES})
        for p in sp.POLICIES:
            definition = sp.definition(p)
            self.assertEqual(definition['roles'], ['attn', 'mlp'])
            self.assertEqual(definition['quantization'], 'all_remote')
            self.assertFalse(definition['arrival'] or definition['prefetch'] or definition['graph'])
        for n in (0, 3, 5, 7):
            with self.assertRaises(ValueError): sp.schedule(n)
        with self.assertRaises(ValueError): sp.selection_entry(None)

    def test_generated_sources_and_gpu_binding(self):
        for fn in (sources.adapter_source, sources.taco_source, sources.codec_source, sources.worker_source):
            ast.parse(fn())
        text = sources.worker_source()
        self.assertLess(text.index('torch.cuda.set_device(local_rank)'), text.index('initialize_megatron()'))
        self.assertIn('H100_A_BASELINE_EXECUTE', text)
        self.assertIn('retained_output_checks', text)
        self.assertNotIn('ma1_decode', sources.taco_source())
        self.assertIn('GemmRSTaco(group, m, n, k_local', sources.taco_source())

    def test_actual_generated_adapter_before_and_after_audit(self):
        # Reuse the existing CPU fake-Megatron executor, not a replica of the adapter.
        with mock.patch.object(previous_tests, 'sp', sp), mock.patch.object(previous_tests, 'sources', sources):
            previous_tests.ContractTests.test_real_generated_adapter_routes_and_shared_codec_calls_survive_audit(self)

    def test_retained_output_probe_rejects_alias_and_mutation(self):
        tree = ast.parse(sources.adapter_source())
        method = next(x for x in next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Adapter').body
                      if isinstance(x, ast.FunctionDef) and x.name == 'record_smoke_output')
        class Value:
            def __init__(self, data, ptr): self.data, self.ptr = data, ptr
            def detach(self): return self
            def clone(self): return Value(list(self.data), self.ptr+100)
            def data_ptr(self): return self.ptr
            def contiguous(self): return self
            def view(self, dtype): return self
            def cpu(self): return self
            def numpy(self): return self
            def tobytes(self): return bytes(self.data)
        ns = dict(hashlib=hashlib, module_role=sp.module_role,
                  torch=types.SimpleNamespace(uint8=object(), equal=lambda a, b: a.data == b.data))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<probe>', 'exec'), ns)
        for corrupt in ('none', 'alias', 'mutate'):
            owner = types.SimpleNamespace(smoke_outputs={}, retention_targets={}, retained_outputs={}, retention_checks={})
            values = [Value([i], 1 if corrupt == 'alias' else i+1) for i in range(6)]
            for value in values: ns['record_smoke_output'](owner, 'x.mlp.linear_fc2', value)
            if corrupt == 'mutate': values[0].data[0] = 7
            if corrupt == 'none':
                ns['record_smoke_output'](owner, 'x.mlp.linear_fc2', Value([8], 8))
                self.assertTrue(owner.retention_checks['mlp']['passed'])
            else:
                with self.assertRaises(AssertionError): ns['record_smoke_output'](owner, 'x.mlp.linear_fc2', Value([8], 8))

    def test_true_profiles_and_gpu_mirrors(self):
        for p in sp.POLICIES:
            result = report.profile_roles(profile_fixture(p), p)
            self.assertEqual(result['attn']['calls'], 128)
            self.assertEqual(result['mlp']['barriers'], 0 if p == 'cuda' else 256)
        for p, remove in [('basic', 'taco_basic_reduce_kernel'), ('improved', 'aten::clone'),
                          ('cuda', 'fused_ash_compress_kernel'), ('cuda', 'ncclDevKernel_SendRecv')]:
            events = [e for e in profile_fixture(p) if e['name'] != remove]
            with self.assertRaises(ValueError): report.profile_roles(events, p)
        events = profile_fixture('basic')
        for event in events:
            if event['name'] == 'taco_basic_decode_kernel': event['name'] = 'taco_decode_ring_kernel'
        with self.assertRaises(ValueError): report.profile_roles(events, 'basic')

    def test_cli_build_action_contract(self):
        for name in ('fused', 'flux', 'codec'):
            self.assertTrue(cli.arguments([name+'-prepare', '/tmp/fixture', '--source-only']).source_only)
            self.assertEqual(cli.arguments([name+'-compile', '/tmp/fixture', '--cuda-home', '/cuda']).cuda_home, Path('/cuda'))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='a-baseline-CPU-FIXTURE-')
        cls.root = Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.fused = fused_builder.prepare(cls.root/'fused', source_only=True)
            cls.flux = flux_builder.prepare(cls.root/'flux', source_only=True)
            cls.codec = codec_builder.prepare(cls.root/'codec', source_only=True)
            cls.compare = preparation.prepare(cls.root/'compare', cls.fused, cls.flux, cls.codec, Path('/absent'), source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_source_only_preview_and_execution_rejection(self):
        with mock.patch.object(preparation, 'capture_hardware', side_effect=AssertionError('GPU query')), \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('launched')), \
                mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('GPU query')), \
                contextlib.redirect_stdout(io.StringIO()):
            info = report.verify_inputs(self.compare)
            self.assertEqual(len(cli.run(self.compare)), 42)
            self.assertEqual(len(list(scope_run.mapping_jobs(self.compare, info, sp))), 4)
            with self.assertRaises(ValueError): cli.run(self.compare, execute=True)
        result = subprocess.run([sys.executable, str(self.compare/'scripts/worker.py')], capture_output=True, text=True,
                                env=dict(os.environ, H100_A_BASELINE_EXECUTE='1'))
        self.assertIn('Source-only', result.stderr)

    def test_reject_duration_scope_identity_and_code_drift(self):
        path, saved = self.compare/'submission.json', read(self.compare/'submission.json')
        for key, value in [('execution_timing', dict(warmup_steps=10, timed_steps=20)),
                           ('comparison_scope', 'mlp'), ('codec_identity', {}), ('source_only', False)]:
            try:
                write(path, dict(saved, **{key: value}))
                with self.assertRaises(ValueError): report.verify_inputs(self.compare)
            finally: write(path, saved)
        script = self.compare/'scripts/worker.py'
        text = script.read_text()
        try:
            script.write_text(text+'# changed\n')
            with self.assertRaises(ValueError): report.verify_inputs(self.compare)
        finally: script.write_text(text)

    def test_complete_negative_order_report_and_rejection_gates(self):
        root = preparation.prepare(self.root/'SYNTHETIC-NO-GPU', self.fused, self.flux, self.codec, Path('/absent'), source_only=True)
        info = read(root/'submission.json')
        info.update(executable=True, source_only=False, hardware=HARDWARE, calibration_hardware=HARDWARE, codec_library=LIBRARY)
        for p, variant in info['variants'].items():
            variant['libraries'] = {n: dict(path='/CPU-FIXTURE/'+p+'/'+n, sha256='f'*64)
                                    for n in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
        write(root/'submission.json', info)
        write(root/'start-gpu-check.json', dict(passed=True, run_id=info['run_id'], synthetic=True))
        for job in scope_run.jobs(info, 'all', sp): make_window(root, info, job)
        make_mappings(root)
        # Only executable-build admission is replaced. Every data/profile/numerical gate runs.
        with mock.patch.object(report, 'verify_inputs', return_value=info):
            gate = report.preflight(root, offline=True)
            result = report.report(root, offline=True)
            self.assertTrue(result['completed'] and result['correctness_passed'])
            self.assertFalse(result['expected_order_point_estimate'] or result['expected_order_confirmed'])
            self.assertEqual(result['formal_optimizer_steps'], 1800)
            self.assertEqual(result['timing_rank_records'], 288)
            self.assertEqual(result['candidate_shape_mappings'], 4)
            self.assertEqual(result['rounds']['combined']['mean_ms']['improved'], 700.)
            self.assertEqual(result['rounds']['combined']['mean_final_loss']['cuda'], 1.)
            job = next(j for j in info['schedule'] if j['policy'] == 'cuda')
            path = root/'results'/job['directory']/'rank0.json'
            original = read(path)
            for key, value in [('losses', [1.]*20), ('timed_steps', 20), ('codec_calls', {}),
                               ('codec_library', None), ('shared_environment', {}), ('rank', 1)]:
                write(path, dict(original, **{key: value}))
                with self.assertRaises(ValueError): report.window(root, path.parent, 'cuda', 'timing', job['block'], job['index'])
            write(path, original)
            job = next(j for j in info['schedule'] if j['policy'] == 'improved')
            path = root/'results'/job['directory']/'rank0.json'
            original = read(path)
            changed = dict(original, grad_norms=[2.]*50)
            write(path, changed)
            with self.assertRaisesRegex(ValueError, 'trajectory differs'): report.report(root, offline=True)
            write(path, original)
            path.unlink()
            with self.assertRaises(FileNotFoundError): report.report(root, offline=True)
        with self.assertRaises(ValueError): report.verify_inputs(root, offline=True)


if __name__ == '__main__':
    unittest.main()
