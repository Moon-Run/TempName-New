"""CPU-only regression tests. Synthetic records are never GPU evidence."""
import ast
import contextlib
import copy
import csv
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import ablation_plan as ap
import ablation_prepare as prep
import ablation_report as ar
import ablation_routing as routing
import ablation_run as runner
import candidate
from common import HERE, REPO, read, sha, write
from report import paired


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent, _, cls.evidence = ap.parent_plan()

    def test_eight_groups_and_three_preregistered_random_seeds(self):
        for base in ap.BASES:
            defs = ap.definitions(base)
            self.assertEqual({d['group'] for d in defs.values()}, {'native', *[f'G{i}' for i in range(7)]})
            self.assertEqual(len(defs), 10)
            self.assertEqual(defs['G0']['order'], 'original')
            self.assertEqual(defs['G0']['quantization_mode'], 'bf16')
            self.assertNotEqual(defs['native']['implementation'], defs['G0']['implementation'])
        for seeds in ([1], [1, 1, 2], [-1, 2, 3]):
            with self.assertRaises(ValueError): ap.definitions('remote_first', seeds)

    def test_arrival_pairs_keep_identical_physical_masks(self):
        for base in ap.BASES:
            defs = ap.definitions(base)
            entries = {k: ap.derive(self.parent, d) for k, d in defs.items() if k != 'native'}
            self.assertEqual(entries['G3']['mask'], entries['G4']['mask'])
            self.assertEqual(entries['G1']['mask'], entries['G2']['mask'])
            self.assertNotEqual(entries['G3']['coords'], entries['G4']['coords'])
            for key in ('G2', 'G4', 'G5-s17', 'G5-s29', 'G5-s43', 'G6'):
                self.assertEqual(entries[key]['maps'], entries['G4']['maps'])

    def test_g0_recovers_measured_streamk_original_order_for_both_bases(self):
        a, b = [ap.derive(self.parent, ap.definitions(base)['G0']) for base in ap.BASES]
        self.assertEqual(a, b)
        remote = self.parent['policies']['remote_first'][0]['base_coords']
        for row, expected in zip(a['coords'], remote):
            self.assertEqual([(t+64) % 512 for t in row], expected)

    def test_random_matches_each_actual_source_destination_count(self):
        for base in ap.BASES:
            defs = ap.definitions(base)
            target = ap.derive(self.parent, defs['G4'])
            masks = []
            for seed in (17, 29, 43):
                e = ap.derive(self.parent, defs[f'G5-s{seed}'])
                self.assertEqual(e, ap.derive(self.parent, defs[f'G5-s{seed}']))
                self.assertEqual(ap.counts(e['mask']), ap.counts(target['mask']))
                masks.append(e['mask_sha256'])
            self.assertEqual(len(set(masks)), 3)

    def test_explicit_bf16_and_all_remote_not_budget_extremes(self):
        for base in ap.BASES:
            defs = ap.definitions(base)
            for key in ('G0', 'G1', 'G2'):
                self.assertEqual(ap.derive(self.parent, defs[key])['selected_per_source'], [0]*8)
            self.assertEqual(ap.derive(self.parent, defs['G6'])['selected_per_source'], [448]*8)
        self.assertEqual(self.parent['config']['fraction'], 1/64)

    def test_corrupt_permutation_and_local_masks_rejected(self):
        d = ap.definitions('remote_first')['G4']; source = self.parent['policies']['remote_first'][0]
        e = ap.derive(self.parent, d)
        e['maps'][0][0], e['maps'][0][400] = e['maps'][0][400], e['maps'][0][0]
        with self.assertRaises(ValueError): ap.validate(e, self.parent['config'], d, source)
        e = ap.derive(self.parent, d); e['mask'][0][0] = 1
        with self.assertRaises(ValueError): ap.validate(e, self.parent['config'], d, source)

    def test_reject_unmeasured_or_a800_parent_and_changed_evidence(self):
        for change in ({'arch': 'sm80'}, {'gpu_calibrated': False}, {'test_only': True}, {'training_passes': [0, 2]}):
            p = dict(self.parent, **change)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'plan.json'; write(path, p)
                with self.assertRaises(ValueError): ap.parent_plan(path)
        with self.assertRaises(ValueError): ap.resolve_input('/no-such-input', '0'*64)
        self.assertEqual(len(self.evidence), 24)

    def test_schedule_is_frozen_paired_reversed_and_not_72_windows(self):
        defs = ap.definitions('remote_first')
        jobs = ap.schedule('remote_first', defs)
        self.assertEqual(len(jobs), 200)
        first = [j['policy'] for j in jobs if j['round'] == 1]
        second = [j['policy'] for j in jobs if j['round'] == 2]
        self.assertEqual(first, list(reversed(second)))
        for rep in (1, 2):
            for block in range(10):
                self.assertEqual({j['policy'] for j in jobs if j['round'] == rep and j['block'] == block}, set(defs))
        self.assertEqual(jobs, ap.schedule('remote_first', defs))
        self.assertNotEqual(jobs, ap.schedule('remote_first', defs, seed=99))

    def test_routing_keeps_native_and_candidate_attention_mlp_distinct(self):
        model = self.parent['config']
        observed = {}
        for i in range(32):
            for name, k in (('self_attention.linear_proj', 4096), ('mlp.linear_fc2', 16384)):
                observed[f'layers.{i}.{name}'] = dict(M=2048, N=4096, K_global=k, K_local=k//8)
        for key in ap.definitions('remote_first'):
            routes = routing.module_routes(observed, key, 32, model)
            self.assertEqual(set(routes.values()), {'native'} if key == 'native' else {'original', key})
            self.assertEqual(routing.codec_placement(key), 'bf16' if key == 'native' else 'fused')

    def test_worker_and_codec_overlay_preserves_common_implementation(self):
        worker = prep.worker_source(); ast.parse(worker)
        self.assertLess(worker.index('H100_ABLATION_EXECUTE'), worker.index('import torch'))
        self.assertIn("report['ablation']", worker)
        self.assertIn('clear_gradients()', worker)
        source = prep.taco_source(); ast.parse(source)
        self.assertIn('GemmRSTacoDoubleBuffered', source)
        self.assertIn("quantization_mode=definition(policy)['quantization_mode']", source)
        frozen = ap.FREEZE/'source/python/flux/gemm_rs_taco.py'
        patched = candidate.codec_source(frozen.read_text()); ast.parse(patched)
        self.assertIn('if not self.owns_configuration:', patched)
        # No change to the forward, decoder, workspace rotation or backward.
        self.assertEqual(patched[patched.index('    def forward('):], frozen.read_text()[frozen.read_text().index('    def forward('):])

    def test_g5_summary_is_within_block_geometric_mean(self):
        values = {key: [100., 200.] for key in ap.definitions('remote_first')}
        values['G5-s17'] = [50., 100.]
        values['G5-s43'] = [200., 400.]
        with mock.patch.object(ar, 'paired', side_effect=lambda x: paired(x, resamples=100)):
            result = ar.summarize(values, [17, 29, 43])
        for actual, expected in zip(result['ms_per_step']['G5-seed-mean'], [100., 200.]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result['comparisons']['G5-seed-mean/native']['blocks'], 2)
        self.assertAlmostEqual(result['comparisons']['G4/G5-seed-mean']['percent'], 0.)

    def test_independent_session_aggregate_rejects_changed_parent_and_duplicates(self):
        base = dict(run_id='a', config={}, seeds=[17, 29, 43], bases={}, parent_plan_sha256='x',
                    parent_freeze=ap.FREEZE_ID, source_hashes={})
        with mock.patch.object(ar, 'report', side_effect=[base, dict(base)]):
            with self.assertRaisesRegex(ValueError, 'Duplicate'): ar.aggregate(['a', 'b'], '/unused')
        with mock.patch.object(ar, 'report', side_effect=[base, dict(base, run_id='b', parent_plan_sha256='y')]):
            with self.assertRaisesRegex(ValueError, 'Different frozen'): ar.aggregate(['a', 'b'], '/unused')

    def test_aggregate_rejects_changed_measurement_environment_or_worker(self):
        identity = dict(fingerprints=[['torch-version', 'cuda-version']],
                        worker_files={'scripts/remote_first/worker.py': 'worker-hash'},
                        libraries={'remote_first-G0': {'libflux_cuda.so': 'binary-hash'}},
                        controllers={'ablation_run.py': 'runner-hash'})
        base = dict(run_id='a', config={}, seeds=[17, 29, 43], bases={},
                    parent_plan_sha256='plan', parent_freeze=ap.FREEZE_ID,
                    source_hashes={'same-cuda-source': 'source-hash'}, measurement_identity=identity)
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp)/name for name in ('a', 'b')]
            for root in roots:
                root.mkdir(); write(root/'report.json', {})
            for field in identity:
                changed = copy.deepcopy(base)
                changed['run_id'] = 'b'
                changed['measurement_identity'][field] = {'different': field}
                with self.subTest(field=field), mock.patch.object(ar, 'report', side_effect=[base, changed]):
                    with self.assertRaisesRegex(ValueError, 'measurement environment'):
                        ar.aggregate(roots, Path(tmp)/f'{field}.json')

    def test_aggregate_accepts_same_measurement_identity_in_independent_sessions(self):
        identity = dict(fingerprints=[['same-runtime']], worker_files={'worker.py': 'same'},
                        libraries={'G0': {'libflux_cuda.so': 'same'}}, controllers={'run.py': 'same'})
        base = dict(run_id='a', config={}, seeds=[17, 29, 43], bases={},
                    parent_plan_sha256='plan', parent_freeze=ap.FREEZE_ID,
                    source_hashes={}, measurement_identity=identity)
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp)/name for name in ('a', 'b')]
            for root in roots:
                root.mkdir(); write(root/'report.json', {})
            with mock.patch.object(ar, 'report', side_effect=[base, dict(base, run_id='b')]):
                result = ar.aggregate(roots, Path(tmp)/'aggregate.json')
            self.assertEqual(result['measurement_identity'], identity)

    def test_taco_forward_passes_explicit_mode_and_mask_for_every_group(self):
        class FakeCodec:
            owns_configuration = True
            def __init__(self, *args, **kwargs): self.kwargs = kwargs
        torch = types.ModuleType('torch'); dist = types.ModuleType('torch.distributed')
        torch.distributed = dist; torch.no_grad = lambda: (lambda f: f); dist.get_world_size = lambda g: 8
        reference = types.ModuleType('reference'); reference.roundtrip = lambda x: x
        flux = types.ModuleType('flux'); codec = types.ModuleType('flux.gemm_rs_taco')
        codec.GemmRSTaco = lambda *a, **kw: self.fail('Expected double-buffered candidate for ALL G groups')
        codec.GemmRSTacoDoubleBuffered = FakeCodec
        modules = {'torch': torch, 'torch.distributed': dist, 'reference': reference,
                   'routing': routing, 'flux': flux, 'flux.gemm_rs_taco': codec}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, modules):
            path = Path(tmp)
            defs = ap.definitions('remote_first')
            write(path/'config.json', dict(double_buffered=True))
            entries = {k: ap.derive(self.parent, d) for k, d in defs.items() if k != 'native'}
            write(path/'selection-plan.json', dict(entries=entries))
            ns = {'__file__': str(path/'taco_support.py')}
            with mock.patch.object(routing, 'definition', side_effect=lambda p: defs[p]):
                exec(compile(prep.taco_source(), ns['__file__'], 'exec'), ns)
                for key, entry in entries.items():
                    op = ns['TacoForward'](None, 2048, 4096, 2048, key)
                    self.assertEqual(op.codec.kwargs['selected'], entry['mask'])
                    self.assertEqual(op.codec.kwargs['quantization_mode'], defs[key]['quantization_mode'])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='h100-ablation-cpu-')
        cls.builds = candidate.prepare(Path(cls.tmp.name)/'builds')
        cls.root = prep.prepare(Path(cls.tmp.name)/'compare', cls.builds, Path('/absent-megatron'),
                                blocks=2, source_only=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_independent_sources_preserve_frozen_compute_and_v2_route(self):
        info = candidate.verify(self.builds)
        self.assertEqual(len(info['variants']), 20)
        for identity, v in info['variants'].items():
            if v['group'] == 'native': continue
            m = read(self.builds/v['build']/'manifest.json')['variants'][identity]
            self.assertLessEqual(set(m['source_changes']), {'phase6_host.hpp', 'python/flux/gemm_rs_taco.py',
                                                          'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'})
            source = self.builds/v['build']/identity/'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
            self.assertIn('if (owned_taco.enabled)', source.read_text())
            self.assertEqual(sha(source), m['parent_source_sha256'][str(source.relative_to(self.builds/v['build']/identity))])

    def test_dry_run_never_imports_torch_or_launches_worker_and_cannot_execute_rehearsal(self):
        with mock.patch.object(runner.subprocess, 'run', side_effect=AssertionError('worker launched')):
            with contextlib.redirect_stdout(io.StringIO()): jobs = runner.run(self.root)
            self.assertEqual(len(jobs), 120)  # 40 preflight + 80 timing, two bases.
            with self.assertRaisesRegex(ValueError, 'non-executable'): runner.run(self.root, execute=True)
        self.assertNotIn('torch', sys.modules)
        self.assertFalse((self.root/'all-state.json').exists())

    def test_changed_frozen_worker_plan_or_schedule_stops_dry_run(self):
        for path in (self.root/'scripts/remote_first/worker.py', self.root/'scripts/remote_first/selection-plan.json',
                     self.root/'schedule.json'):
            original = path.read_bytes()
            try:
                path.write_bytes(original+b'\n')
                with self.assertRaises(ValueError): ar.verify_inputs(self.root)
            finally: path.write_bytes(original)

    def test_changed_build_source_rejected(self):
        v = candidate.verify(self.builds)['variants']['remote_first-G0']
        path = self.builds/v['build']/'remote_first-G0/phase6_host.hpp'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            with self.assertRaises(ValueError): candidate.verify(self.builds)
        finally: path.write_bytes(original)

    def test_real_prepare_requires_built_libraries(self):
        with self.assertRaisesRegex(ValueError, 'Unbuilt'):
            prep.prepare(Path(self.tmp.name)/'not-runnable', self.builds, Path('/absent'))

    def test_complete_synthetic_evidence_gates_and_all_sample_report(self):
        # CPU fixtures deliberately use fake libraries and records. They live only
        # in this TemporaryDirectory and cannot become runnable build artifacts.
        root = Path(self.tmp.name)/'synthetic'
        prep.prepare(root, self.builds, Path('/absent'), blocks=2, source_only=True)
        info = read(root/'submission.json')
        info.update(source_only=False, executable=True)
        for v in info['variants'].values():
            if v['group'] != 'native':
                v['libraries'] = {name: dict(path='/CPU-FIXTURE/'+name, sha256='f'*64)
                                  for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
        write(root/'submission.json', info)
        for job in runner.jobs(info, 'all'):
            make_window(root, info, job)
        for base in info['bases']:
            entries = read(root/'scripts'/base/'selection-plan.json')['entries']
            for key, e in entries.items():
                path = root/'results'/base/f'mapping-{key}.csv'
                with path.open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=['rank', 'index', 'm', 'n', 'destination', 'sm_occupancy'])
                    writer.writeheader()
                    for rank, row in enumerate(e['coords']):
                        for i, t in enumerate(row):
                            writer.writerow(dict(rank=rank, index=i, m=t//32, n=t%32, destination=t//64, sm_occupancy=2))
        gate = ar.preflight(root, offline=True)
        self.assertTrue(gate['completed'])
        with mock.patch.object(ar, 'paired', side_effect=lambda x: paired(x, resamples=100)):
            result = ar.report(root, offline=True)
        self.assertEqual(len(result['all_samples']), 80)
        # Slow native block is retained; the max (rank7), not rank0 or median, is used.
        samples = [s for s in result['all_samples'] if s['base'] == 'remote_first' and s['policy'] == 'native']
        self.assertEqual([s['ms_per_step'] for s in samples], [11000, 1000, 11000, 1000])
        self.assertIsNone(result['acceptance_threshold'])
        self.assertEqual(result['measurement_identity']['fingerprints'], read(root/'preflight.json')['fingerprints'])
        self.assertEqual(result['measurement_identity']['worker_files']['scripts/remote_first/worker.py'],
                         sha(root/'scripts/remote_first/worker.py'))
        self.assertEqual(result['measurement_identity']['libraries']['remote_first-G0']['libflux_cuda.so'], 'f'*64)
        self.assertIn('G4/G5-seed-mean', result['bases']['remote_first']['combined']['comparisons'])
        job = info['schedule'][0]; path = root/'results'/job['directory']/'rank7.json'
        original = read(path)
        for mutation in ({'fallback_calls': 1}, {'timed_steps': 19}, {'derived_plan_sha256': 'bad'},
                         {'ablation': {}}, {'sampling': True}, {'module_policies': {}}):
            write(path, dict(original, **mutation))
            with self.assertRaises(ValueError): ar.window(root, path.parent, job['base'], job['policy'], 'timing', job['block'], job['index'])
        write(path, original)
        path.unlink()
        with self.assertRaises(FileNotFoundError): ar.window(root, path.parent, job['base'], job['policy'], 'timing', job['block'], job['index'])
        # A BF16 profile accidentally routing MLP to V3 must fail.
        profile = root/'results/remote_first/profile-G0/rank0-profile.json.gz'
        with gzip.open(profile, 'wt') as f:
            json.dump(dict(traceEvents=[dict(cat='kernel', name='flux_bf16_sm90')]), f)
        with self.assertRaisesRegex(ValueError, 'V2 MLP'): ar.preflight(root, offline=True)


def make_window(root, info, job):
    base, key, mode = (job[k] for k in ('base', 'policy', 'mode'))
    directory = root/'results'/job['directory']; directory.mkdir(parents=True)
    c = info['config']; v = info['variants'][base+'-'+key]
    cfg = read(root/'scripts'/base/'config.json')
    count = cfg['smoke_steps'] if mode == 'smoke' else 1 if mode == 'profile' else 20
    warmup = cfg['smoke_warmup_steps'] if mode == 'smoke' else 10
    model = dict(cfg['common'], **cfg['cases'][0])
    observed = {}
    for i in range(32):
        for name, k in (('self_attention.linear_proj', 4096), ('mlp.linear_fc2', 16384)):
            observed[f'layers.{i}.{name}'] = dict(M=2048, N=4096, K_global=k, K_local=k//8)
    routes = routing.module_routes(observed, key, 32, model)
    mask = None if key == 'native' else read(root/'scripts'/base/'selection-plan.json')['entries'][key]['mask']
    for rank in range(8):
        hardware = dict(info['calibration_hardware'], local_rank=rank)
        r = dict(completed=True, passed=True, skips=0, fallback_calls=0, run_id=info['run_id'], policy=key, mode=mode,
                 ablation=ap.definitions(base)[key], derived_plan_sha256=v.get('derived_plan_sha256'),
                 parent_plan_sha256=info['parent_plan_sha256'], rank=rank, local_rank=rank, tp_rank=rank, dp_rank=0,
                 tp_group_ranks=list(range(8)), dp_group_ranks=[rank], world_size=8, hardware=hardware,
                 gpu_uuid=hardware['devices'][rank]['uuid'], model=model, block=job['block'], window=job['index'],
                 timing_protocol='full-step-one-clear-v2', sampling=False, profiling=mode == 'profile',
                 losses=[1.]*count, grad_norms=[1.]*count, initial_loss=1., warmup_steps=warmup, timed_steps=count,
                 elapsed_seconds=(20 if rank == 7 else 10)*(11 if key == 'native' and job['block'] == 0 else 1),
                 module_policies=routes, observed_shapes=observed, target_modules=list(observed),
                 warmup_calls={n: warmup*4 for n in observed}, replaced_modules=0 if key == 'native' else 64,
                 flux_library_loaded=False, selection_checks={}, codec_placements={}, forward_checks={},
                 initial_parameters_sha256='init'+str(rank), initial_rng_sha256='rng'+str(rank), tokens_sha256='tokens',
                 parameter_count_local=123, padded_vocab_size=32768, torch='CPU-FIXTURE', cuda='CPU-FIXTURE',
                 nccl='CPU-FIXTURE', python='CPU-FIXTURE')
        if key != 'native':
            r.update(v['libraries'])
            bf16 = 7*256*4096*2
            for n in observed:
                mlp = n.endswith('.mlp.linear_fc2')
                r['codec_placements'][n] = 'fused' if mlp else 'bf16'
                if mlp:
                    r['selection_checks'][n] = dict(mask_sha256=ap.mask_sha(mask),
                        wire_bytes=bf16+sum(mask[rank])*128*(136-256), bf16_wire_bytes=bf16)
                r['forward_checks'][n] = dict(max_abs=0., relative_l2=0., passed=True,
                    protocol_relative_l2=0., budget_vs_bf16=.05, budget_vs_codec=.01)
        write(directory/f'rank{rank}.json', r)
        if mode == 'profile':
            names = ['native_gemm'] if key == 'native' else ['flux_bf16_sm80', 'flux_bf16_sm90', 'taco_decode_ring_kernel<false>']
            with gzip.open(directory/f'rank{rank}-profile.json.gz', 'wt') as f:
                json.dump(dict(traceEvents=[dict(cat='kernel', name=n) for n in names]), f)


if __name__ == '__main__': unittest.main()
