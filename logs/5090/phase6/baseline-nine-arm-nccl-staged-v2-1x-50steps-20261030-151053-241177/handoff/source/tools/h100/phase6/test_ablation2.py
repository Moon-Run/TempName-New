"""CPU regressions; synthetic evidence lives only in temporary directories."""
import ast
import contextlib
import copy
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import ablation2
import ablation2_guard as guard
import ablation2_plan as spec
import ablation2_report as stats
import ablation_plan as legacy
import ablation_prepare as prep
import ablation_report as evidence
import ablation_routing as routing
import ablation_run as runner
import candidate
from common import REPO, read, sha, write


class ProtocolTests(unittest.TestCase):
    def test_start_check_is_once_only_and_busy_fails_without_wait(self):
        parent, _, _ = legacy.parent_plan()
        info = dict(run_id='CPU-FIXTURE', calibration_hardware=parent['hardware'])
        rows = '\n'.join(f'{i}, GPU-{d["uuid"].removeprefix("GPU-")}, NVIDIA H100, 100, 0'
                         for i, d in enumerate(parent['hardware']['devices']))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(guard.socket, 'gethostname', return_value=parent['hardware']['host']), \
                mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(8)))}), \
                mock.patch.object(guard.subprocess, 'check_output', side_effect=[rows, '']) as query:
            root = Path(tmp)
            self.assertTrue(guard.check_start(root, info)['passed'])
            self.assertEqual(query.call_count, 2)
            with self.assertRaisesRegex(ValueError, 'already started'): guard.check_start(root, info)
            self.assertEqual(query.call_count, 2)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(guard.socket, 'gethostname', return_value=parent['hardware']['host']), \
                mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(8)))}), \
                mock.patch.object(guard.subprocess, 'check_output', side_effect=[rows, 'GPU-0, 123, training, 2000']):
            with self.assertRaisesRegex(ValueError, 'busy/unavailable'): guard.check_start(Path(tmp), info)
            self.assertFalse(read(Path(tmp)/'start-gpu-check.json')['passed'])

    def test_balanced_fifty_windows_and_exact_reverse(self):
        defs = spec.definitions()
        self.assertEqual(tuple(defs), spec.POLICIES)
        jobs = spec.schedule('interleaved', defs)
        self.assertEqual(len(jobs), 50)
        rows = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b]
                for r in (1, 2) for b in range(5)]
        self.assertEqual(rows[5:], [row[::-1] for row in rows[:5][::-1]])
        for pos in range(5):
            self.assertEqual({row[pos] for row in rows[:5]}, set(spec.POLICIES))
        for row in rows:
            self.assertEqual(set(row), set(spec.POLICIES))
        self.assertEqual(len(spec.schedule('interleaved', defs, blocks=10)), 100)
        for n in (0, 1, 4, 6, -5):
            with self.assertRaises(ValueError): spec.schedule('interleaved', defs, blocks=n)
        with self.assertRaises(ValueError): spec.definitions('remote_first')
        with self.assertRaises(ValueError): spec.definitions(seeds=[17, 29, 43])
        with self.assertRaises(ValueError): spec.schedule('interleaved', defs, seed=123)
        self.assertEqual(len(legacy.schedule('interleaved', legacy.definitions('interleaved'))), 200)

    def test_frozen_masks_arrival_and_compute_route(self):
        parent, _, _ = legacy.parent_plan()
        entries = {k: spec.derive(parent, d) for k, d in spec.definitions().items() if k != 'native_taco'}
        self.assertEqual(entries['G3']['selected_per_source'], [7, 7, 7, 0, 7, 7, 7, 3])
        self.assertEqual(entries['G3']['mask'], entries['G4']['mask'])
        self.assertEqual(entries['G1']['mask'], entries['G2']['mask'])
        self.assertFalse(any(map(any, entries['G1']['mask'])))
        self.assertEqual(entries['G1']['coords'], entries['G3']['coords'])
        self.assertEqual(entries['G2']['coords'], entries['G4']['coords'])
        self.assertNotEqual(entries['G1']['coords'], entries['G2']['coords'])
        self.assertEqual({routing.codec_placement(k) for k in entries}, {'fused'})
        self.assertEqual(routing.codec_placement('native_taco'), 'tensor')
        with self.assertRaises(ValueError): routing.selection_entry({}, 'native_taco', 2048, 4096, 16384)

    def test_native_candidate_constructor_rejected_before_flux_import(self):
        torch = types.ModuleType('torch'); torch.no_grad = lambda: lambda f: f
        dist = types.ModuleType('torch.distributed'); torch.distributed = dist
        reference = types.ModuleType('reference'); reference.roundtrip = lambda x: x
        with mock.patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist, 'reference': reference,
                                          'routing': routing, 'flux': None}):
            ns = {}
            exec(prep.taco_source(), ns)
            with self.assertRaisesRegex(ValueError, 'Native baseline'):
                ns['TacoForward'](None, 2048, 4096, 2048, 'native_taco')

    def test_native_adapter_preserves_forward_and_only_changes_mlp_communication(self):
        # Execute the real adapter with CPU stubs, including its copied forward globals.
        calls = []
        ns = {'reduce_scatter_to_sequence_parallel_region': lambda x: calls.append('native-rs') or x}
        exec('def forward(self, x):\n    return reduce_scatter_to_sequence_parallel_region(x)\n', ns)
        class Linear:
            forward = ns['forward']
            input_is_parallel = sequence_parallel = True
            explicit_expert_comm = gradient_accumulation_fusion = False
            config = types.SimpleNamespace(_cpu_offloading_context=None)
            weight = types.SimpleNamespace(dtype='bf16')
            output_size = 4096
            def __init__(self, k):
                self.input_size = k; self.input_size_per_partition = k//8
        class Model:
            config = types.SimpleNamespace(num_layers=1)
            def named_modules(self): return [('x.self_attention.linear_proj', attention), ('x.mlp.linear_fc2', mlp)]
        attention, mlp = Linear(4096), Linear(16384)
        torch = types.ModuleType('torch'); torch.bfloat16 = 'bf16'; torch.float32 = 'fp32'
        torch.autograd = types.SimpleNamespace(Function=object)
        state = types.SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
        tensor = types.SimpleNamespace(TensorTacoRS=lambda *a: types.SimpleNamespace(wire_bytes_per_rank=1, bf16_wire_bytes_per_rank=2),
                                       QuantizedRS=types.SimpleNamespace(apply=lambda x, codec: calls.append('tensor-taco') or x))
        modules = {'torch': torch, 'megatron': types.ModuleType('megatron'),
                   'megatron.core': types.SimpleNamespace(parallel_state=state),
                   'megatron.core.tensor_parallel': types.ModuleType('tp'),
                   'megatron.core.tensor_parallel.layers': types.SimpleNamespace(RowParallelLinear=Linear),
                   'megatron.core.tensor_parallel.mappings': types.SimpleNamespace(_gather_along_first_dim=None),
                   'taco_support': types.SimpleNamespace(TacoForward=lambda *a: self.fail('Flux constructor called'), check_taco_output=None),
                   'routing': routing, 'megatron_taco': tensor, 'flux': None}
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {'E2E_MODE': 'timing'}):
            scope = {}
            exec((REPO/'tools/a800/phase5/e2e/adapter.py').read_text(), scope)
            adapter = scope['Adapter'](Model(), 'native_taco', [[2048, 4096, 4096], [2048, 4096, 16384]])
            adapter.audit = False
            input_ = types.SimpleNamespace(shape=(2048, 1, 2048))
            attention.forward(input_); mlp.forward(input_)
            adapter.stop_audit()
            self.assertIs(mlp.forward.__func__.__code__, Linear.forward.__code__)
            self.assertIs(attention.forward.__func__, Linear.forward)
            self.assertEqual(calls, ['native-rs', 'tensor-taco'])
            self.assertIsNone(adapter.flux)

    def test_paired_speedups_best_provenance_and_increments(self):
        values = {k: [100., 200., 300., 400., 500.] for k in spec.POLICIES}
        values['G1'] = [50., 200., 100., 400., 250.]
        blocks = [dict(round=2, block=i) for i in range(5)]
        s = stats.summarize(values, blocks)
        self.assertEqual(s['speedups']['G1']['median'], 2.)
        self.assertEqual(s['speedups']['G1']['best'], dict(ratio=3., round=2, block=2, native_taco_ms=300., group_ms=100.))
        self.assertEqual(len(s['comparisons']), 9)
        expected = 100*(1-math.exp(sum(math.log(x/y) for x, y in zip(values['G1'], values['native_taco']))/5))
        self.assertAlmostEqual(s['comparisons']['G1/native_taco']['percent'], expected)
        self.assertEqual(s['comparisons']['G4/G1']['blocks'], 5)
        self.assertLess(s['comparisons']['G4/G1']['ci95'][1], 0)

    def test_uniform_non_iterated_whole_block_filter_and_empty_case(self):
        values = {k: [100., 100., 100., 100., 100.] for k in spec.POLICIES}
        values['G1'][0] = 111.; values['G4'][1] = 150.
        blocks = [dict(round=1, block=i) for i in range(5)]
        s = stats.sensitivity(values, blocks)
        self.assertEqual(s['remaining_blocks'], 3)
        self.assertEqual([e['block'] for e in s['excluded']], [0, 1])
        self.assertEqual([list(e['triggers']) for e in s['excluded']], [['G1'], ['G4']])
        self.assertEqual({v['blocks'] for v in s['summary']['comparisons'].values()}, {3})
        for i, k in enumerate(spec.POLICIES):
            values[k] = [120. if j == i else 100. for j in range(5)]
        empty = stats.sensitivity(values, blocks)['summary']
        self.assertEqual(empty['status'], '无可用配对块'); self.assertEqual(empty['comparisons'], {})
        single = stats.summarize({k: [100.] for k in spec.POLICIES}, blocks[:1])
        self.assertIsNone(single['comparisons']['G4/G1']['ci95'])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='h100-ablation2-CPU-FIXTURE-')
        cls.base = Path(cls.tmp.name)
        cls.builds = candidate.prepare(cls.base/'builds', seeds=(), bases=spec.BASES, protocol=spec.PROTOCOL)
        cls.root = prep.prepare(cls.base/'compare', cls.builds, Path('/absent'), blocks=5, source_only=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_only_four_builds_and_native_has_no_plan_or_library(self):
        info = candidate.verify(self.builds)
        self.assertEqual(len(list((self.builds/'builds').iterdir())), 4)
        self.assertEqual(len(list((self.builds/'plans').iterdir())), 5)
        compare = evidence.verify_inputs(self.root)
        self.assertNotIn('root', compare['variants']['interleaved-native_taco'])
        self.assertNotIn('native_taco', read(self.root/'scripts/interleaved/selection-plan.json')['entries'])
        native = (self.root/'scripts/interleaved/megatron_taco.py').read_text()
        self.assertIn('self.world in (4,8)', native)
        source = (REPO/'tools/a800/phase5/e2e/megatron_taco.py').read_text()
        self.assertEqual(native, source.replace('self.world==4', 'self.world in (4,8)'))
        for k in spec.POLICIES[1:]:
            v = info['variants']['interleaved-'+k]
            m = read(self.builds/v['build']/'manifest.json')['variants']['interleaved-'+k]
            self.assertLessEqual(set(m['source_changes']), {'phase6_host.hpp', 'python/flux/gemm_rs_taco.py',
                                                          'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'})

    def test_dry_run_and_direct_worker_cannot_launch_gpu_or_promote_source_only(self):
        with mock.patch.object(runner.subprocess, 'run', side_effect=AssertionError('GPU launched')), \
                mock.patch.object(runner.subprocess, 'check_output', side_effect=AssertionError('GPU queried')):
            with contextlib.redirect_stdout(io.StringIO()): jobs = runner.run(self.root)
            self.assertEqual(len(jobs), 60)
            self.assertEqual(sum(j['mode'] == 'timing' for j in jobs), 50)
            with self.assertRaisesRegex(ValueError, 'non-executable'): runner.run(self.root, execute=True)
        worker = self.root/'scripts/interleaved/worker.py'
        p = subprocess.run([sys.executable, str(worker)], env=dict(os.environ, H100_ABLATION_EXECUTE='1'), capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0); self.assertIn('permanently non-executable', p.stderr)
        original = read(self.root/'submission.json')
        try:
            write(self.root/'submission.json', dict(original, source_only=False, executable=True))
            with self.assertRaisesRegex(ValueError, 'Preparation mode changed'): evidence.verify_inputs(self.root)
        finally: write(self.root/'submission.json', original)
        self.assertNotIn('torch', sys.modules)

    def test_bad_hashes_and_unbuilt_reuse_rejected(self):
        for path in (self.root/'schedule.json', self.root/'scripts/interleaved/worker.py',
                     self.root/'scripts/interleaved/megatron_taco.py'):
            original = path.read_bytes()
            try:
                path.write_bytes(original+b'\n')
                with self.assertRaises(ValueError): evidence.verify_inputs(self.root)
            finally: path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'Unbuilt'):
            prep.prepare(self.base/'unbuilt', self.builds, Path('/absent'), blocks=5)
        with self.assertRaisesRegex(ValueError, 'complete local builds'):
            ablation2.reuse(self.base/'reuse-unbuilt', self.builds)
        self.assertFalse((self.base/'reuse-unbuilt').exists())

    def test_reuse_checks_only_four_builds_and_never_changes_sources(self):
        source = candidate.prepare(self.base/'CPU-FIXTURE-built', seeds=(), bases=spec.BASES, protocol=spec.PROTOCOL)
        old = read(source/'ablation-manifest.json')
        old['protocol'] = legacy.PROTOCOL
        old['variants']['remote_first-G5-s17'] = dict(build='DOES-NOT-EXIST')
        originals = {}
        for key in spec.POLICIES[1:]:
            identity = 'interleaved-'+key
            v = old['variants'][identity]
            root = source/v['build']
            m = read(root/'manifest.json'); m.update(protocol=legacy.PROTOCOL, status='built')
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping'):
                path = root/identity/name
                path.write_text('CPU TEST FIXTURE, NOT AN EXECUTABLE')
                m['files'][f'{identity}/{name}'] = sha(path)
            write(root/'manifest.json', m)
            originals[str(root/'manifest.json')] = sha(root/'manifest.json')
        write(source/'ablation-manifest.json', old)
        originals[str(source/'ablation-manifest.json')] = sha(source/'ablation-manifest.json')
        target = ablation2.reuse(self.base/'CPU-FIXTURE-reuse', source)
        result = candidate.verify(target, built=True)
        self.assertEqual(set(result['variants']), {'interleaved-'+k for k in spec.POLICIES})
        self.assertFalse(result['reuse']['timing_reused'])
        self.assertEqual(originals, {p: sha(p) for p in originals})
        v = result['variants']['interleaved-G1']
        library = Path(v['build'])/'interleaved-G1/libflux_cuda.so'
        library.write_text('changed')
        with self.assertRaises(ValueError): candidate.verify(target, built=True)

    def test_synthetic_full_evidence_report_and_failure_gates(self):
        root = self.base/'SYNTHETIC-EVIDENCE'
        prep.prepare(root, self.builds, Path('/absent'), blocks=5, source_only=True)
        # Explicit offline fixtures, never executable libraries; do not persist them.
        info = read(root/'submission.json'); info.update(source_only=False, executable=True)
        seal = read(root/'provenance/preparation.json'); seal.update(source_only=False, executable=True)
        write(root/'provenance/preparation.json', seal)
        for key in spec.POLICIES[1:]:
            identity = 'interleaved-'+key
            v = info['variants'][identity]
            v['libraries'] = {name: dict(path='/CPU-FIXTURE/'+name, sha256='f'*64)
                              for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
            path = root/'provenance'/f'{identity}-manifest.json'
            m = read(path); m['status'] = 'built'; write(path, m)
        info['files'] = {name: sha(root/name) for name in info['files']}
        write(root/'submission.json', info)
        write(root/'start-gpu-check.json', dict(passed=True, run_id=info['run_id'], synthetic=True))
        for job in runner.jobs(info, 'all'): make_window(root, info, job)
        entries = read(root/'scripts/interleaved/selection-plan.json')['entries']
        for key, entry in entries.items():
            with (root/'results/interleaved'/f'mapping-{key}.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['rank', 'index', 'm', 'n', 'destination', 'sm_occupancy'])
                writer.writeheader()
                for rank, row in enumerate(entry['coords']):
                    for i, t in enumerate(row):
                        writer.writerow(dict(rank=rank, index=i, m=t//32, n=t%32, destination=t//64, sm_occupancy=2))
        evidence.preflight(root, offline=True)
        result = stats.report(root, offline=True)
        self.assertEqual(len(result['all_samples']), 50)
        self.assertEqual(result['timing_rank_records'], 400)
        self.assertEqual(result['profile_traces'], 40)
        native = [s['ms_per_step'] for s in result['all_samples'] if s['policy'] == 'native_taco']
        self.assertEqual(native, [2000, 1000, 1000, 1000, 1000]*2)
        self.assertEqual(result['sensitivity']['1']['remaining_blocks'], 4)
        self.assertEqual(result['rounds']['combined']['blocks'], 10)
        self.assertIn('最佳单块观测', (root/'report.md').read_text())
        # Incorrect masks, native routing, rank completeness, fingerprints and sampling fail.
        native_job = next(j for j in runner.jobs(info, 'timing') if j['policy'] == 'native_taco')
        path = root/'results'/native_job['directory']/'rank7.json'
        original = read(path)
        for mutation in ({'fallback_calls': 1}, {'sampling': True}, {'native_linear_forward_preserved': False},
                         {'flux_library_loaded': True}, {'derived_plan_sha256': 'bad'}, {'replaced_modules': 64}):
            write(path, dict(original, **mutation))
            with self.assertRaises(ValueError): evidence.window(root, path.parent, 'interleaved', 'native_taco', 'timing', native_job['block'], native_job['index'])
        write(path, dict(original, tokens_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'initialization/RNG/tokens'): stats.report(root, offline=True)
        path.unlink()
        with self.assertRaises(FileNotFoundError): stats.report(root, offline=True)
        write(path, original)
        profile = root/'results/interleaved/profile-G1/rank0-profile.json.gz'
        original_trace = profile.read_bytes()
        with gzip.open(profile, 'wt') as f:
            json.dump(dict(traceEvents=[dict(cat='kernel', name='flux_bf16_sm90')]), f)
        with self.assertRaisesRegex(ValueError, 'V2 MLP'): evidence.preflight(root, offline=True)
        profile.write_bytes(original_trace)
        path = root/'results/interleaved/smoke-G3/rank0.json'
        original = read(path); changed = copy.deepcopy(original)
        next(iter(changed['selection_checks'].values()))['mask_sha256'] = 'bad'
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'mask/wire'): evidence.preflight(root, offline=True)
        write(path, original)


def make_window(root, info, job):
    key, mode = job['policy'], job['mode']
    folder = root/'results'/job['directory']; folder.mkdir(parents=True)
    cfg = read(root/'scripts/interleaved/config.json')
    count = cfg['smoke_steps'] if mode == 'smoke' else 1 if mode == 'profile' else 20
    warmup = cfg['smoke_warmup_steps'] if mode == 'smoke' else 10
    model = dict(cfg['common'], **cfg['cases'][0])
    observed = {f'layers.{i}.{name}': dict(M=2048, N=4096, K_global=k, K_local=k//8)
                for i in range(32) for name, k in [('self_attention.linear_proj', 4096), ('mlp.linear_fc2', 16384)]}
    v = info['variants']['interleaved-'+key]
    mask = None if key == 'native_taco' else read(root/'scripts/interleaved/selection-plan.json')['entries'][key]['mask']
    for rank in range(8):
        hardware = dict(info['calibration_hardware'], local_rank=rank)
        r = dict(completed=True, passed=True, skips=0, fallback_calls=0, run_id=info['run_id'], policy=key, mode=mode,
                 ablation=spec.definitions()[key], derived_plan_sha256=v.get('derived_plan_sha256'),
                 parent_plan_sha256=info['parent_plan_sha256'], rank=rank, local_rank=rank, tp_rank=rank, dp_rank=0,
                 tp_group_ranks=list(range(8)), dp_group_ranks=[rank], world_size=8, hardware=hardware,
                 gpu_uuid=hardware['devices'][rank]['uuid'], model=model, block=job['block'], window=job['index'],
                 timing_protocol='full-step-one-clear-v2', sampling=False, profiling=mode == 'profile',
                 losses=[1.]*count, grad_norms=[1.]*count, initial_loss=1., warmup_steps=warmup, timed_steps=count,
                 elapsed_seconds=(20 if rank == 7 else 10)*(2 if key == 'native_taco' and job['block'] == 0 else 1),
                 module_policies=routing.module_routes(observed, key, 32, model), observed_shapes=observed,
                 target_modules=list(observed), warmup_calls={n: warmup*4 for n in observed},
                 replaced_modules=32 if key == 'native_taco' else 64, flux_library_loaded=False,
                 native_linear_forward_preserved=key == 'native_taco', selection_checks={}, codec_placements={}, forward_checks={},
                 initial_parameters_sha256='init'+str(rank), initial_rng_sha256='rng'+str(rank), tokens_sha256='tokens',
                 parameter_count_local=123, padded_vocab_size=33792, torch='CPU-FIXTURE', cuda='CPU-FIXTURE',
                 nccl='CPU-FIXTURE', python='CPU-FIXTURE')
        if key != 'native_taco': r.update(v['libraries'])
        for n in observed:
            mlp = n.endswith('.mlp.linear_fc2')
            r['codec_placements'][n] = ('tensor' if mlp else 'native') if key == 'native_taco' else ('fused' if mlp else 'bf16')
            if mlp:
                r['selection_checks'][n] = dict(mask_sha256=legacy.mask_sha(mask) if mask else None,
                    wire_bytes=7*256*4096*2+sum(mask[rank])*128*(136-256) if mask else 7*256*32*136,
                    bf16_wire_bytes=7*256*4096*2)
            if mlp or key != 'native_taco':
                r['forward_checks'][n] = dict(max_abs=0., relative_l2=0., passed=True,
                    protocol_relative_l2=0., budget_vs_bf16=.05, budget_vs_codec=.01)
        write(folder/f'rank{rank}.json', r)
        if mode == 'profile':
            names = ['native_gemm', 'ncclDevKernel_AllToAll'] if key == 'native_taco' else ['flux_bf16_sm80', 'flux_bf16_sm90', 'taco_decode_ring_kernel<false>']
            events = [dict(cat='kernel', name=n) for n in names]
            if key == 'native_taco':
                events += [dict(cat='user_annotation', name='megatron_taco.'+n) for n in ('encode', 'exchange', 'decode_reduce')]
            with gzip.open(folder/f'rank{rank}-profile.json.gz', 'wt') as f:
                json.dump(dict(traceEvents=events), f)


if __name__ == '__main__': unittest.main()
