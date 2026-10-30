"""CPU tests only. Simulated GPU records are temporary fixtures, never evidence."""
import ast
import contextlib
import copy
import csv
import gzip
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

import scope_protocol as sp
import scope_plan as plans
import scope_build as builder
import scope_calibrate as calibration
import scope_sources as sources
import scope_prepare as preparation
import scope_run as runner
import scope_report as report
from common import read, write, sha, REPO
from ablation_plan import mask_sha
import plan as fitter
from test_protocol import mapping, reports


class ProtocolTests(unittest.TestCase):
    def test_four_scopes_balanced_32_windows_and_reverse(self):
        jobs = sp.schedule(); self.assertEqual(len(jobs), 32)
        rows = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b] for r in (1, 2) for b in range(4)]
        self.assertEqual(rows[4:], [r[::-1] for r in rows[:4][::-1]])
        for i in range(4): self.assertEqual({r[i] for r in rows[:4]}, set(sp.POLICIES))
        for p in sp.POLICIES:
            self.assertEqual(sp.route(p, 'x.mlp.linear_fc2') == 'native_bf16', p.endswith('_attn'))
        with self.assertRaises(ValueError): sp.definition('native_taco')
        for n in (0, 1, 3, 5):
            with self.assertRaises(ValueError): sp.schedule(n)
        self.assertEqual(len(sp.schedule(8)), 64)

    def test_exact_role_shape_local_mask_budget_and_no_synthetic_promotion(self):
        bundle = plans.bundle(source_only=True)
        attn = bundle['roles']['attn']; mlp = bundle['roles']['mlp']
        self.assertEqual(attn['entry']['K_local'], 512); self.assertEqual(mlp['entry']['K_local'], 2048)
        with self.assertRaisesRegex(ValueError, 'own real H100'): plans.bundle()
        with self.assertRaises(ValueError): plans.validate(attn, 'attn')
        with self.assertRaises(ValueError): plans.validate(mlp, 'attn')
        for change in ('local', 'budget', 'hash'):
            broken = copy.deepcopy(attn)
            if change == 'local': broken['entry']['mask'][0][0] = 1
            elif change == 'budget': broken['entry']['mask'][0][64:80] = [1]*16
            else: broken['entry_sha256'] = 'wrong'
            with self.assertRaises(ValueError): plans.validate(broken, 'attn', allow_test=True)
        for role, shape in sp.SHAPES.items():
            self.assertEqual(sp.selection_entry(bundle, 'interleaved_full_attn_mlp', role, *shape)['role'], role)
        with self.assertRaises(ValueError): sp.selection_entry(bundle, 'interleaved_full_attn', 'mlp', *sp.SHAPES['mlp'])
        with self.assertRaises(ValueError): sp.selection_entry(bundle, 'interleaved_full_attn_mlp', 'attn', *sp.SHAPES['mlp'])

    def test_attention_fitter_uses_full_shape_without_changing_model(self):
        c = plans.model_config(); rows = mapping(c, 'interleaved')
        rs = reports(c)
        for r in rs: r['shapes'][0]['K_global'] = 4096
        first = fitter.fit(rs, rows, c, 'interleaved', sp.SHAPES['attn'])
        self.assertEqual(first['shape'], [2048, 4096, 4096]); self.assertEqual(c['ffn'], 16384)
        for r in rs:
            for s in r['shapes'][0]['samples']:
                if s['mode'] == 'receiver_sparse' and s['repeat'] >= 128:
                    for e in s['receiver']: e['observed_ns'] += 333*e['source']
        second = fitter.fit(rs, rows, c, 'interleaved', sp.SHAPES['attn'])
        for field in ('maps', 'coords', 'mask'): self.assertEqual(first[field], second[field])
        with self.assertRaises(StopIteration): fitter.fit(rs, rows, c, 'interleaved', sp.SHAPES['mlp'])

    def test_calibration_rank_shape_pass_and_library_gates(self):
        c = plans.model_config(); hardware = plans.frozen_mlp()['hardware']
        with tempfile.TemporaryDirectory(prefix='scope-calibration-CPU-FIXTURE-') as tmp:
            root = Path(tmp); lib = root/'build/scope-sampler'; lib.mkdir(parents=True)
            hashes = {}
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                (lib/name).write_text('CPU FIXTURE, NOT AN EXECUTABLE'); hashes[name] = sha(lib/name)
            p = dict(run_id='CPU-FIXTURE', roles=['attn'], config=c, build=str(root/'build'))
            rs = reports(c)
            for rank, r in enumerate(rs):
                r.update(protocol=sp.PROTOCOL,run_id=p['run_id'],roles=['attn'],model_config=c,world_size=8,
                         policy='interleaved',kind='instrumented',sampling=True,smoke=False,
                         hardware=dict(hardware,local_rank=rank),library_hashes=hashes)
                s = r['shapes'][0]; s.update(K_global=4096,correctness=True,fragment_count=8,kernel=dict(name='flux_bf16_sm80'))
                for sample in s['samples']: sample['correctness']=True
                write(root/f'b0-interleaved-instrumented-rank{rank}.json',r)
            self.assertEqual(len(calibration.calibration_records(root,p)),8)
            path=root/'b0-interleaved-instrumented-rank7.json'; original=read(path)
            for failure in ('shape','pass','library','kernel'):
                changed=copy.deepcopy(original)
                if failure=='shape': changed['shapes'][0]['K_global']=16384
                elif failure=='pass': changed['shapes'][0]['samples'].pop()
                elif failure=='kernel': changed['shapes'][0]['kernel']['name']='flux_bf16_sm90'
                else: changed['library_hashes']['libflux_cuda.so']='bad'
                write(path,changed)
                with self.assertRaises(ValueError): calibration.calibration_records(root,p)
            path.unlink()
            with self.assertRaises(FileNotFoundError): calibration.calibration_records(root,p)

    def test_same_scope_baselines_best_paired_block_and_net_effects(self):
        values = {p: [100., 100., 100., 100.] for p in sp.POLICIES}
        values['native_taco_attn_mlp'] = [300., 200., 100., 400.]
        values['interleaved_full_attn'] = [50., 100., 25., 100.]
        s = report.summarize(values, [dict(round=1, block=i) for i in range(4)])
        self.assertEqual(s['speedups']['interleaved_full_attn']['median'], 1.5)
        self.assertEqual(s['speedups']['interleaved_full_attn_mlp']['median'], 2.5)
        self.assertEqual(s['speedups']['interleaved_full_attn']['best']['block'], 2)
        self.assertEqual(len(s['comparisons']), 4)
        self.assertLess(s['comparisons']['native_taco_attn_mlp/native_taco_attn']['percent'], 0)

    def test_whole_block_filter_shared_across_all_four_comparisons(self):
        blocks = [dict(round=1, block=i) for i in range(4)]
        values = {p: [100.]*4 for p in sp.POLICIES}; values[sp.POLICIES[2]][1] = 111.
        s = report.sensitivity(values, blocks)
        self.assertEqual(s['remaining_blocks'], 3); self.assertEqual(s['excluded'][0]['block'], 1)
        self.assertEqual({v['blocks'] for v in s['summary']['comparisons'].values()}, {3})
        for i, p in enumerate(sp.POLICIES): values[p] = [120. if j == i else 100. for j in range(4)]
        self.assertEqual(report.sensitivity(values, blocks)['summary']['status'], '无可用配对块')
        s = report.summarize({p: [1.] for p in sp.POLICIES}, blocks[:1])
        self.assertIsNone(next(iter(s['comparisons'].values()))['ci95'])

    def test_profile_role_kernel_association_rejects_v3_and_wrong_scope(self):
        events = profile_fixture('interleaved_full_attn')
        self.assertEqual(report.profile_roles(events, 'interleaved_full_attn')['mlp']['route'], 'native_bf16')
        broken = copy.deepcopy(events)
        for e in broken:
            if e['name'] == 'flux_bf16_sm80': e['name'] = 'flux_bf16_sm90'
        with self.assertRaises(ValueError): report.profile_roles(broken, 'interleaved_full_attn')
        with self.assertRaises(ValueError): report.profile_roles(events, 'interleaved_full_attn_mlp')

    def test_profile_gpu_annotation_mirrors_do_not_count_as_cpu_calls(self):
        for policy in sp.POLICIES:
            with self.subTest(policy=policy):
                events = profile_fixture(policy)
                expected = report.profile_roles(events, policy)
                cpu_scopes = [e for e in events if e.get('cat') == 'user_annotation']
                mirrors = [dict(e, cat='gpu_user_annotation', pid=2, tid=7, ts=e['ts'] + 10000)
                           for e in cpu_scopes]
                self.assertEqual(report.profile_roles(events + mirrors, policy), expected)
                self.assertEqual({v['calls'] for v in expected.values()}, {128})
                missing_cpu = [e for e in events if e is not cpu_scopes[0]]
                with self.assertRaisesRegex(ValueError, 'Missing profile scopes'):
                    report.profile_roles(missing_cpu + mirrors, policy)
                with self.assertRaisesRegex(ValueError, 'Missing profile scopes'):
                    report.profile_roles(events + [cpu_scopes[0]] + mirrors, policy)


class AdapterTests(unittest.TestCase):
    def test_real_adapter_before_after_stop_audit_for_all_scopes(self):
        for policy in sp.POLICIES:
            for mode in ('timing', 'profile'):
                with self.subTest(policy=policy, mode=mode): self.exercise(policy, mode)

    def exercise(self, policy, mode):
        calls = []
        ns = {'reduce_scatter_to_sequence_parallel_region': lambda x: calls.append(('native_bf16', x.role)) or x}
        exec('def forward(self,x):\n    return reduce_scatter_to_sequence_parallel_region(x),None\n', ns)
        class Value:
            dtype = 'bf16'
            def __init__(self, role): self.role = role; self.shape = (2048, 1, sp.SHAPES[role][2]//8)
            def stride(self): return (self.shape[2], self.shape[2], 1)
        class Linear:
            forward = ns['forward']; input_is_parallel = sequence_parallel = skip_bias_add = True
            explicit_expert_comm = gradient_accumulation_fusion = False
            config = types.SimpleNamespace(_cpu_offloading_context=None); output_size = 4096; bias = None
            def __init__(self, role):
                self.input_size = sp.SHAPES[role][2]; self.input_size_per_partition = self.input_size//8
                self.weight = types.SimpleNamespace(dtype='bf16', shape=(4096, self.input_size//8))
        modules = {f'layer.{i}.{suffix}': Linear(role) for i in range(32)
                   for role, suffix in [('attn', 'self_attention.linear_proj'), ('mlp', 'mlp.linear_fc2')]}
        model = types.SimpleNamespace(config=types.SimpleNamespace(num_layers=32), named_modules=lambda: modules.items())
        torch = types.ModuleType('torch'); torch.bfloat16 = 'bf16'; torch.float32 = 'fp32'; torch.autograd = types.SimpleNamespace(Function=object)
        torch.profiler = types.SimpleNamespace(record_function=lambda name: contextlib.nullcontext())
        state = types.SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
        class Candidate:
            owns_output = True
            def __init__(self, group, m, n, k, p, role):
                assert k == sp.SHAPES[role][2]//8
                self.plan_sha256 = role+'-plan'
                self.codec = types.SimpleNamespace(placement='fused', selection_sha256=role+'-mask', wire_bytes_per_rank=1, bf16_wire_bytes_per_rank=2)
                self.workspace_info = [dict(shape=sp.SHAPES[role],mask_sha256=role+'-mask',packet_ptr=id(self)*10+i,mask_ptr=id(self)*10+i+2) for i in range(2)]
            def forward(self, x, w): calls.append(('candidate', x.role)); return x
        tensor = types.SimpleNamespace(TensorTacoRS=lambda *a: types.SimpleNamespace(wire_bytes_per_rank=1, bf16_wire_bytes_per_rank=2),
            QuantizedRS=types.SimpleNamespace(apply=lambda x, codec: calls.append(('tensor', x.role)) or x))
        flux = types.SimpleNamespace(init_flux_shm=lambda *a: None, ReduceScatterOption=type('Options', (), {}),
                                   RingMode=types.SimpleNamespace(Ring1D=1), GemmRS=lambda *a, **k: self.fail('Unexpected fallback'))
        mocks = {'torch': torch, 'megatron': types.ModuleType('megatron'), 'megatron.core': types.SimpleNamespace(parallel_state=state),
                 'megatron.core.tensor_parallel': types.ModuleType('tp'),
                 'megatron.core.tensor_parallel.layers': types.SimpleNamespace(RowParallelLinear=Linear),
                 'megatron.core.tensor_parallel.mappings': types.SimpleNamespace(_gather_along_first_dim=None),
                 'taco_support': types.SimpleNamespace(TacoForward=Candidate, check_taco_output=None),
                 'megatron_taco': tensor, 'routing': sp, 'flux': None if sp.is_native(policy) else flux}
        with mock.patch.dict(sys.modules, mocks), mock.patch.dict(os.environ, {'E2E_MODE': mode}):
            scope = {}; exec(sources.adapter_source(), scope)
            scope['GemmRSFunction'] = types.SimpleNamespace(apply=lambda x, w, op, option: op.forward(x, w))
            adapter = scope['Adapter'](model, policy, list(sp.SHAPES.values()))
            for name, module in modules.items(): module.forward(Value(sp.module_role(name)))
            expected = [(sp.route(policy,n), sp.module_role(n)) for n in modules]
            self.assertEqual(calls, expected)
            adapter.stop_audit(); calls.clear()
            for _ in range(2):
                for name, module in modules.items(): module.forward(Value(sp.module_role(name)))
            self.assertEqual(calls, expected*2)
            self.assertEqual(set(adapter.timed_calls.values()), {2})
            self.assertEqual(adapter.actual_routes, adapter.post_audit_routes)
            self.assertEqual(len(adapter.scope_ops), 32*len(sp.definition(policy)['roles']))
            self.assertEqual(adapter.forward_preserved, {n: sp.route(policy,n) != 'candidate' for n in modules})
            if sp.is_native(policy): self.assertIsNone(adapter.flux)

    def test_candidate_double_buffers_receive_independent_role_masks(self):
        class Codec:
            owns_configuration = True
            def __init__(self, group, m, n, k, **kwargs):
                self.kwargs = kwargs
                self.slots = [types.SimpleNamespace(m=m,n=n,k_local=k,rank=0,selection_sha256=mask_sha(kwargs['selected']),
                    peers=[types.SimpleNamespace(data_ptr=lambda:123)],selected=types.SimpleNamespace(data_ptr=lambda:456)) for _ in range(2)]
        torch = types.ModuleType('torch'); torch.no_grad = lambda: lambda f: f
        dist = types.ModuleType('torch.distributed'); dist.get_world_size = lambda g: 8; torch.distributed = dist
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {
                'torch': torch, 'torch.distributed': dist, 'routing': sp,
                'reference': types.SimpleNamespace(roundtrip=lambda x: x),
                'flux': types.ModuleType('flux'), 'flux.gemm_rs_taco': types.SimpleNamespace(GemmRSTacoDoubleBuffered=Codec)}):
            path = Path(tmp); bundle = dict(roles={r: plans.cpu_fixture(r) for r in sp.SHAPES})
            write(path/'selection-plan.json', bundle)
            scope = {'__file__': str(path/'taco_support.py')}; exec(sources.taco_source(), scope)
            ops = [scope['TacoForward'](None, 2048, 4096, k//8, 'interleaved_full_attn_mlp', role)
                   for role, (_, _, k) in sp.SHAPES.items()]
            self.assertIsNot(ops[0].codec, ops[1].codec)
            for role, op in zip(sp.SHAPES, ops):
                self.assertEqual(op.codec.kwargs['selected'], bundle['roles'][role]['entry']['mask'])
                self.assertEqual(op.codec.kwargs['quantization_mode'], 'targeted')
            with self.assertRaises(ValueError): scope['TacoForward'](None, 2048, 4096, 512, 'native_taco_attn', 'attn')


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='scope-CPU-FIXTURE-'); cls.base = Path(cls.tmp.name)
        cls.build = builder.prepare(cls.base/'build', source_only=True)
        cls.compare = preparation.prepare(cls.base/'compare', cls.build, Path('/absent'), source_only=True)
        cls.sampler = builder.prepare(cls.base/'sampler', sampler=True, roles=('attn', 'mlp'))
        cls.cal = calibration.prepare(cls.base/'cal', cls.sampler, source_only=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_dry_runs_do_not_import_torch_query_or_launch_gpu(self):
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('launched')), \
                mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('queried')):
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = runner.run(self.compare); maps, _ = calibration.run(self.cal)
            self.assertEqual(len(jobs), 40); self.assertEqual(set(maps), {'attn', 'mlp'})
            info = report.verify_inputs(self.compare)
            mapping_jobs = list(runner.mapping_jobs(self.compare, info))
            self.assertEqual(len(mapping_jobs), 3)
            self.assertEqual(mapping_jobs[0][2][1:4], ['2048', '4096', '4096'])
            self.assertEqual(mapping_jobs[2][2][1:4], ['2048', '4096', '16384'])
            with self.assertRaises(ValueError): runner.run(self.compare, execute=True)
            with self.assertRaises(ValueError): calibration.run(self.cal, execute=True)
        self.assertNotIn('torch', sys.modules)

    def test_source_only_worker_and_flags_cannot_be_promoted(self):
        for root, script, metadata in [(self.compare, 'scripts/worker.py', 'submission.json'), (self.cal, 'worker.py', 'plan.json')]:
            p = subprocess.run([sys.executable, str(root/script)], env=dict(os.environ, H100_SCOPE_EXECUTE='1'), capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0); self.assertIn('Source-only', p.stderr)
            original = read(root/metadata)
            try:
                write(root/metadata, dict(original, source_only=False, executable=True))
                with self.assertRaises(ValueError): (report.verify_inputs if metadata == 'submission.json' else calibration.verify)(root)
            finally: write(root/metadata, original)
        with self.assertRaises(ValueError): calibration.fit(self.cal, self.base/'forbidden-fit')

    def test_plan_scope_worker_hashes_and_sampler_timing_rejected(self):
        for path in [self.compare/'scripts/adapter.py', self.compare/'scripts/selection-plan.json', self.compare/'schedule.json']:
            original = path.read_bytes()
            try:
                path.write_bytes(original+b'\n')
                with self.assertRaises(ValueError): report.verify_inputs(self.compare)
            finally: path.write_bytes(original)
        path = self.cal/'controllers/scope_protocol.py'; original = path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            with self.assertRaises(ValueError): calibration.verify(self.cal)
        finally: path.write_bytes(original)
        with self.assertRaises(ValueError): preparation.prepare(self.base/'bad-sampler', self.sampler, Path('/absent'), source_only=True)
        info = read(self.compare/'submission.json')
        try:
            changed = copy.deepcopy(info); changed['variants']['interleaved_full_attn']['root'] = '/different'
            write(self.compare/'submission.json', changed)
            with self.assertRaises(ValueError): report.verify_inputs(self.compare)
        finally: write(self.compare/'submission.json', info)

    def test_builds_preserve_compute_and_shape_dispatch(self):
        m = builder.verify(self.build); v = m['variants']['scope-candidate']
        self.assertEqual(set(v['source_changes']), {'phase6_host.hpp', builder.SWIZZLE, 'python/flux/gemm_rs_taco.py'})
        table = (self.build/'scope-candidate/phase6_host.hpp').read_text()
        self.assertIn('if(k==512)', table); self.assertIn('if(k==2048)', table)
        self.assertEqual(m['config']['ffn'], 16384)
        with self.assertRaises(ValueError): builder.verify(self.build, built=True)

    def test_synthetic_complete_report_and_failure_paths(self):
        root = preparation.prepare(self.base/'SYNTHETIC-ONLY', self.build, Path('/absent'), source_only=True)
        info = make_executable_fixture(root)
        for job in runner.jobs(info, 'all'): make_window(root, info, job)
        role_plans = read(root/'scripts/selection-plan.json')['roles']
        for policy, role, _ in runner.mapping_jobs(root, info):
            with (root/'results'/f'mapping-{policy}-{role}.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['rank','index','m','n','destination','sm_occupancy']); writer.writeheader()
                for rank, coords in enumerate(role_plans[role]['entry']['coords']):
                    for i, t in enumerate(coords): writer.writerow(dict(rank=rank,index=i,m=t//32,n=t%32,destination=t//64,sm_occupancy=2))
        report.preflight(root, offline=True); result = report.report(root, offline=True)
        self.assertEqual(result['timing_rank_records'], 256); self.assertEqual(len(result['all_samples']), 32)
        self.assertEqual(result['preflight_rank_records'], 64); self.assertEqual(result['profile_traces'], 32)
        self.assertEqual(result['rounds']['combined']['blocks'], 8)
        self.assertEqual(result['sensitivity']['1']['remaining_blocks'], 3)
        job = info['schedule'][0]; path = root/'results'/job['directory']/'rank7.json'; original = read(path)
        for mutate in ({'post_audit_routes': {}}, {'timed_calls': {}}, {'scope_definition': {}}, {'sampling': True}):
            write(path, dict(original, **mutate))
            with self.assertRaises(ValueError): report.window(root, path.parent, job['policy'], 'timing', job['block'], job['index'])
        write(path, dict(original, tokens_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'Timing initialization'): report.report(root, offline=True)
        path.unlink()
        with self.assertRaises(FileNotFoundError): report.report(root, offline=True)
        write(path, original)
        changed = copy.deepcopy(original)
        slots = list(changed['workspace_checks'].values())
        slots[1][0]['packet_ptr'] = slots[0][0]['packet_ptr']
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'alias'):
            report.window(root, path.parent, job['policy'], 'timing', job['block'], job['index'])
        write(path, original)
        smoke = root/'results/smoke-interleaved_full_attn_mlp/rank0.json'; original = read(smoke)
        changed = copy.deepcopy(original)
        attn = next(n for n in changed['selection_checks'] if n.endswith('.self_attention.linear_proj'))
        changed['selection_checks'][attn]['shape'] = sp.SHAPES['mlp']
        write(smoke, changed)
        with self.assertRaises(ValueError): report.preflight(root, offline=True)
        write(smoke, original)


def make_executable_fixture(root):
    """Forge only temporary CPU fixtures; there are no executable libraries here."""
    info = read(root/'submission.json'); info.update(source_only=False, executable=True)
    bundle = read(root/'scripts/selection-plan.json'); bundle.update(source_only=False, gpu_calibrated=True)
    attn = bundle['roles']['attn']; attn.update(test_only=False, gpu_calibrated=True, hardware=info['calibration_hardware'],
        origin=dict(kind='scope_bf16_calibration'), inputs={'/CPU-FIXTURE':'f'*64})
    write(root/'scripts/selection-plan.json', bundle)
    seal = read(root/'provenance/preparation.json'); seal.update(source_only=False, executable=True, plans_sha256=sha(root/'scripts/selection-plan.json'))
    write(root/'provenance/preparation.json', seal)
    m = read(root/'provenance/build-manifest.json'); m.update(status='built', source_only=False, diagnostic_only=False)
    write(root/'provenance/build-manifest.json', m); info['build_manifest']['sha256'] = sha(root/'provenance/build-manifest.json')
    for policy in sp.POLICIES:
        if not sp.is_native(policy):
            info['variants'][policy]['libraries'] = {n: dict(path='/CPU-FIXTURE/'+n,sha256='f'*64)
                for n in ('libflux_cuda.so','libflux_cuda_ths_op.so')}
    info['files'] = {p: sha(root/p) for p in info['files']}; write(root/'submission.json', info)
    write(root/'start-gpu-check.json', dict(passed=True,run_id=info['run_id'],synthetic=True))
    return info


def profile_fixture(policy):
    events = []
    for ri, role in enumerate(sp.SHAPES):
        route = ('tensor' if sp.is_native(policy) else 'candidate') if role in sp.definition(policy)['roles'] else 'native_bf16'
        names = ['flux_bf16_sm80','taco_decode_ring_kernel'] if route == 'candidate' else ['native_gemm']
        for i in range(128):
            ts = ri*1000000+i*1000
            events.append(dict(ph='X',name=f'scope.{role}.{route}',cat='user_annotation',pid=1,tid=1,ts=ts,dur=100))
            for j, name in enumerate(names):
                corr = ri*10000+i*10+j
                events += [dict(ph='X',name='cudaLaunchKernel',cat='cuda_runtime',pid=1,tid=1,ts=ts+1+j,dur=1,args={'correlation':corr}),
                           dict(ph='X',name=name,cat='kernel',pid=2,tid=7,ts=ts+10000,dur=1,args={'correlation':corr})]
    if sp.is_native(policy):
        events += [dict(name='megatron_taco.'+n) for n in ('encode','exchange','decode_reduce')]+[dict(name='c10d::alltoall_base_')]
    return events


def make_window(root, info, job):
    policy, mode = job['policy'], job['mode']; folder = root/'results'/job['directory']; folder.mkdir(parents=True)
    cfg = read(root/'scripts/config.json'); bundle = read(root/'scripts/selection-plan.json')
    count = cfg['smoke_steps'] if mode == 'smoke' else 1 if mode == 'profile' else cfg['timed_steps']
    warmup = cfg['smoke_warmup_steps'] if mode == 'smoke' else cfg['warmup_steps']
    observed = {f'layers.{i}.{suffix}':dict(M=2048,N=4096,K_global=k,K_local=k//8) for i in range(32)
                for suffix,k in [('self_attention.linear_proj',4096),('mlp.linear_fc2',16384)]}
    routes = sp.module_routes(observed, policy); selected = {n for n,v in routes.items() if v!='native_bf16'}
    for rank in range(8):
        hardware = dict(info['calibration_hardware'],local_rank=rank)
        r = dict(protocol=sp.PROTOCOL,completed=True,passed=True,skips=0,fallback_calls=0,run_id=info['run_id'],policy=policy,mode=mode,
                 scope_definition=sp.definition(policy),role_plan_sha256=cfg['role_plan_sha256'],rank=rank,local_rank=rank,tp_rank=rank,dp_rank=0,
                 tp_group_ranks=list(range(8)),dp_group_ranks=[rank],world_size=8,hardware=hardware,gpu_uuid=hardware['devices'][rank]['uuid'],
                 model=dict(cfg['common'],**cfg['cases'][0]),padded_vocab_size=33792,block=job['block'],window=job['index'],
                 timing_protocol='full-step-one-clear-v2',sampling=False,profiling=mode=='profile',losses=[1.]*count,grad_norms=[1.]*count,
                 initial_loss=1.,warmup_steps=warmup,timed_steps=count,elapsed_seconds=(count if rank==7 else count/2)*(2 if job['block']==0 and policy==sp.POLICIES[0] else 1),
                 observed_shapes=observed,module_policies=routes,post_audit_routes=routes,target_modules=list(routes),warmup_calls={n:warmup*4 for n in routes},
                 timed_calls={n:count*4 for n in routes},enabled_modules=sorted(selected),replaced_modules=len(selected),independent_scope_instances=len(selected),
                 forward_preserved={n:v not in ('candidate','flux_tensor') for n,v in routes.items()},codec_placements={n:{'native_bf16':'native','tensor':'tensor','candidate':'fused','flux_tensor':'tensor'}[v] for n,v in routes.items()},
                 flux_library_loaded=False,native_linear_forward_preserved=sp.is_native(policy),selection_checks={},forward_checks={},workspace_checks={},
                 initial_parameters_sha256='init'+str(rank),initial_rng_sha256='rng'+str(rank),tokens_sha256='tokens',parameter_count_local=123,
                 torch='CPU-FIXTURE',cuda='CPU-FIXTURE',nccl='CPU-FIXTURE',python='CPU-FIXTURE')
        if not sp.is_native(policy):r.update(info['variants'][policy]['libraries'])
        for name in selected:
            role=sp.module_role(name); p=bundle['roles'][role]; mask=p['entry']['mask']
            r['selection_checks'][name]=dict(role=role,shape=sp.SHAPES[role],bf16_wire_bytes=7*256*4096*2,
                mask_sha256=None if sp.is_native(policy) else mask_sha(mask),plan_sha256=None if sp.is_native(policy) else p['entry_sha256'],
                wire_bytes=7*256*32*136 if sp.is_native(policy) else 7*256*4096*2+sum(mask[rank])*128*(136-256))
            r['forward_checks'][name]=dict(passed=True,max_abs=0.,relative_l2=0.,protocol_relative_l2=0.,budget_vs_bf16=.05,budget_vs_codec=.01)
            if not sp.is_native(policy):
                i = len(r['workspace_checks'])*10+100
                r['workspace_checks'][name] = [dict(shape=sp.SHAPES[role],mask_sha256=p['mask_sha256'],packet_ptr=i+j,mask_ptr=i+j+2) for j in range(2)]
        write(folder/f'rank{rank}.json',r)
        if mode=='profile':
            with gzip.open(folder/f'rank{rank}-profile.json.gz','wt') as f:json.dump(dict(traceEvents=profile_fixture(policy)),f)


if __name__ == '__main__': unittest.main()
