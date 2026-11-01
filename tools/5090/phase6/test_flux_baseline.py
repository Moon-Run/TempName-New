"""CPU fixtures and source checks; never reported as GPU measurements."""
import ast
from collections import Counter
import copy
from contextlib import redirect_stdout, ExitStack
import gzip
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from rtx5090_loader import HERE, REPO, load
import flux_baseline as controller
import flux_baseline_build as builder
import flux_baseline_native as native
import flux_baseline_probe as probe
import flux_baseline_protocol as p
import flux_baseline_profile as profile
import flux_baseline_routing as routing
import flux_baseline_sources as sources
import flux_baseline_validation as validation
from nccl_transport import wire_layout, PACKET_BYTES_PER_SOURCE
from test_baseline_workers import fixture as old_rows
from test_baseline_profile import fixture as old_trace
from test_op1_a_workers import basic_trace


def fixture(arm, mode='timing'):
    rows, job, plan, worker = old_rows(sources.base_arm(arm), mode)
    job['arm'] = arm
    worker['selection'] = sources.selection_metadata()
    worker['config'] = sources.configuration(arm, worker['selection'], plan['communication'],
                                            worker['config'].get('codec_library'))
    worker['definition'] = sources.definition(arm)
    plan['protocol'] = p.PROTOCOL
    for row in rows:
        row.update(protocol=p.PROTOCOL, arm=arm, policy=arm, scope_definition=worker['definition'],
                   role_plan_sha256=worker['config']['role_plan_sha256'],
                   flux_implementation=sources.implementation(arm),
                   smoke_output_sha256={}, retained_output_checks={})
        names = row['target_modules']
        row['selection_checks'] = {name: dict(mask_sha256=None, plan_sha256=None,
            wire_bytes=7*256*4096*2 if arm == 'flux-bf16' else 7*8192*136,
            bf16_wire_bytes=7*256*4096*2) for name in names}
        if mode == 'smoke':
            row['smoke_output_sha256'] = dict.fromkeys(names, 'a'*64)
            for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
                row['retained_output_checks'][role] = dict(module=next(n for n in names if n.endswith(suffix)),
                    outputs=6, passed=True, independent_storage=True, unchanged_after_subsequent_call=True)
        if arm in p.BASIC_ARMS:
            row['codec_identity'] = worker['config']['codec_identity']
        if arm == 'flux-improved':
            for slots in row['workspace_checks'].values():
                for slot in slots:
                    slot.pop('mask_ptr'); slot.pop('mask_sha256')
            layout = wire_layout(None, row['rank'])
            for record in row['rtx5090_transport'].values():
                for slot in record['slots']:
                    slot.update(send_split_bytes=layout['tx_splits'], receive_split_bytes=layout['rx_splits'],
                        send_bytes=layout['tx_bytes'], receive_bytes=layout['rx_bytes'],
                        shared_scratch_bytes=2*8*256*4096*2+8*PACKET_BYTES_PER_SOURCE+layout['tx_bytes']+layout['rx_bytes'])
        if arm == 'flux-bf16':
            row.update(codec_library=None, codec_validation=None)
            calls = (row['warmup_steps']+len(row['losses']))*4
            row['bf16_collectives'] = {name: dict(quantization='none', collective='reduce_scatter_tensor',
                dtype='bfloat16', calls=calls, encode_calls=0, decode_calls=0, all_to_all_calls=0,
                owns_output=True, logical_remote_bytes=7*256*4096*2) for name in names}
            for check in row['forward_checks'].values():
                check.update(reference='unquantized_torch_matmul_nccl_reduce_scatter', quantization='none')
    return rows, job, plan, worker


def bf16_trace():
    events = basic_trace()
    discarded = [e for e in events if e.get('cat') == 'kernel' and
                 any(token in e['name'] for token in ('fused_ash_', 'basic_quant_'))]
    correlations = {e['args']['correlation'] for e in discarded}
    ids = set(map(id, discarded))
    events = [e for e in events if id(e) not in ids and not
              (e.get('cat') == 'cuda_runtime' and e['args'].get('correlation') in correlations)
              and e['name'] not in ('megatron_taco.encode', 'megatron_taco.decode_reduce')]
    for event in events:
        if event['name'] == 'megatron_taco.exchange':
            event['name'] = 'flux_baseline.bf16_reduce_scatter'
        elif 'alltoall' in event['name'].lower().replace('_', ''):
            event['name'] = 'c10d::reduce_scatter_tensor'
    return events


class Protocol(unittest.TestCase):
    def test_balanced_four_arm_schedule(self):
        for trials, blocks in ((1, 8), (2, 16)):
            jobs = p.schedule(trials, blocks)
            self.assertEqual(len(jobs), trials*blocks*4)
            for trial in range(1, trials+1):
                own = [j for j in jobs if j['trial'] == trial]
                orders = [tuple(j['arm'] for j in own if j['block'] == block) for block in range(blocks)]
                for position in range(4):
                    self.assertEqual(Counter(order[position] for order in orders), Counter(dict.fromkeys(p.ARMS, blocks//4)))
                for a, b in itertools.combinations(p.ARMS, 2):
                    self.assertEqual(sum(order.index(a) < order.index(b) for order in orders), blocks//2)
        for blocks in (0, 6, 9):
            with self.assertRaises(ValueError): p.schedule(1, blocks)

    def test_all_pairs_and_negative_gains_are_preserved(self):
        samples = [dict(job, ms_per_step=1000+p.ARMS.index(job['arm'])*-10,
                   peak_allocated_gib=17, peak_reserved_gib=18, native_workspace_gib=1,
                   peak_allocated_plus_taco_workspace_gib=18) for job in p.schedule()]
        result = p.report(samples)
        self.assertEqual(result['timing_windows'], 32)
        self.assertEqual(result['formal_optimizer_steps'], 1600)
        self.assertEqual(len(result['comparisons']['compare1']['pairs']), 6)
        self.assertLess(result['comparisons']['compare1']['arms']['flux-improved']['reduction_vs_bf16_percent'], 0)
        self.assertIn('无量化', p.markdown(result))
        with self.assertRaises(ValueError): p.report(samples[:-1])
        bad = copy.deepcopy(samples); bad[0]['ms_per_step'] = float('nan')
        with self.assertRaises(ValueError): p.report(bad)

    def test_routing_rejects_calibration_and_selection(self):
        for arm in p.ARMS:
            definition = routing.definition(arm)
            self.assertFalse(definition['arrival'])
            self.assertFalse(definition['interleaved'])
            self.assertEqual(definition['quantization'], 'none' if arm == 'flux-bf16' else 'all_remote')
            routing.role_plans(sources.selection_metadata(), arm)
        with self.assertRaises(ValueError): routing.selection_entry()
        bad = sources.selection_metadata(); bad['mask'] = []
        with self.assertRaises(ValueError): routing.role_plans(bad, 'flux-improved')


class Workers(unittest.TestCase):
    def test_bf16_runtime_only_submits_reduce_scatter_and_returns_owned_outputs(self):
        calls = []
        class Tensor:
            dtype, is_cuda, device = 'bf16', True, 'cuda:0'
            def __init__(self, shape): self.shape = shape
            def is_contiguous(self): return True
        torch = ModuleType('torch')
        dist = ModuleType('torch.distributed')
        torch.bfloat16 = 'bf16'
        torch.autograd = SimpleNamespace(Function=object)
        torch.empty = lambda shape, **kw: Tensor(shape)
        dist.get_rank = lambda group: 0
        dist.get_world_size = lambda group: 8
        dist.get_backend = lambda group: 'nccl'
        dist.reduce_scatter_tensor = lambda output, partial, **kw: calls.append((output, partial, kw['group']))
        torch.distributed = dist
        module = ModuleType('cpu_fixture_bf16')
        with patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist}), patch.dict(os.environ, {'E2E_MODE': 'timing'}):
            exec(compile(sources.codec_source('flux-bf16'), 'bf16_cpu_fixture', 'exec'), module.__dict__)
            op = module.TensorTacoRS('group', 2048, 4096)
            first, second = op.reduce(Tensor((2048, 4096))), op.reduce(Tensor((2048, 4096)))
            self.assertIsNot(first, second)
            self.assertEqual(len(calls), 2)
            self.assertEqual(first.shape, (256, 4096))
            self.assertEqual(op.state()['encode_calls'], 0)
            self.assertEqual(op.state()['calls'], 2)
            with self.assertRaises(ValueError): op.reduce(Tensor((256, 4096)))

    def test_native_c_probe_uses_double_buffers_without_a_mask(self):
        frontend = SimpleNamespace(GemmRSTaco=lambda *a, **kw: self.fail('single slot selected'),
            GemmRSTacoDoubleBuffered=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('double slot selected')))
        dist = SimpleNamespace(get_rank=lambda: 0, group=SimpleNamespace(WORLD='group'))
        with self.assertRaisesRegex(RuntimeError, 'double slot selected'):
            probe.probe.role_probe(None, dist, frontend, None, 'c', 'attn', None)
        self.assertEqual(probe.probe.FUSED, ())
    def test_generated_workers_preserve_training_and_timing(self):
        for arm in p.ARMS:
            for generate in (sources.worker_source, sources.adapter_source, sources.taco_source):
                compile(generate(arm), arm+'/'+generate.__name__, 'exec')
            sources.base.validate_worker(sources.worker_source(arm))
        c = sources.taco_source('flux-improved')
        tree = ast.parse(c)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TacoForward')
        init = ast.unparse(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
        self.assertIn('GemmRSTacoDoubleBuffered', init)
        self.assertNotIn('selection_entry', init)
        self.assertNotIn('selected=', init)
        self.assertIn("quantization_mode='all_remote'", init)

    def test_all_arms_and_modes_validate(self):
        for arm in p.ARMS:
            for mode in ('smoke', 'profile', 'timing'):
                with self.subTest(arm=arm, mode=mode):
                    rows, job, plan, worker = fixture(arm, mode)
                    self.assertEqual(len(validation.validate_rows(rows, job, plan, worker)), 8)

    def test_mask_wrong_ownership_wrong_wire_and_missing_wait_rejected(self):
        mutations = [lambda row: next(iter(row['selection_checks'].values())).update(mask_sha256='a'*64),
            lambda row: next(iter(row['selection_checks'].values())).update(wire_bytes=1),
            lambda row: next(iter(row['workspace_checks'].values()))[0].update(mask_ptr=1),
            lambda row: next(iter(row['ma2c_codec_state'].values())).update(owns_output=False),
            lambda row: next(iter(row['rtx5090_transport'].values()))['slots'][0].update(wait_calls=0)]
        for mutate in mutations:
            rows, job, plan, worker = fixture('flux-improved')
            mutate(rows[0])
            with self.assertRaises(ValueError): validation.validate_rows(rows, job, plan, worker)

    def test_bf16_cannot_claim_quantized_work_or_codec_reference(self):
        for mutate in (lambda row: row.update(codec_library={'fake': True}),
                       lambda row: next(iter(row['bf16_collectives'].values())).update(encode_calls=1),
                       lambda row: next(iter(row['forward_checks'].values())).update(quantization='all_remote')):
            rows, job, plan, worker = fixture('flux-bf16', 'smoke')
            mutate(rows[0])
            with self.assertRaises(ValueError): validation.validate_rows(rows, job, plan, worker)

    def test_reduction_differences_are_observations_not_equivalence_gates(self):
        refs = {(1, 0, 'timing'): {arm: fixture(arm)[0] for arm in p.ARMS}}
        refs[(1, 0, 'timing')]['flux-improved'][0]['losses'][0] += .1
        result = controller.numerical_comparison(refs)
        self.assertEqual(len(result), 6)
        self.assertAlmostEqual(result['flux-improved/flux-bf16']['max_loss_abs_difference'], .1)
        self.assertFalse(result['flux-improved/flux-bf16']['equivalence_required'])


class Profiles(unittest.TestCase):
    def test_a_c_require_original_order_kernel_and_old_decoder(self):
        for arm, old in p.NATIVE.items():
            events = old_trace(old, te_final=True, memcpy=True, multiple_nccl=True)
            with self.assertRaises(ValueError): profile.analyze(events, arm, 0)
            for event in events:
                if load('profile').is_local_gemm(event['name']): event['name'] += '_'+native.MARKER
            self.assertTrue(profile.analyze(events, arm, 0)['execution_validated'])

    def test_basic_and_bf16_are_distinct_actual_paths(self):
        quantized, bf16 = basic_trace(), bf16_trace()
        self.assertTrue(profile.analyze(quantized, 'flux-basic', 0)['execution_validated'])
        self.assertTrue(profile.analyze(bf16, 'flux-bf16', 0)['execution_validated'])
        with self.assertRaises(ValueError): profile.analyze(quantized, 'flux-bf16', 0)
        with self.assertRaises(ValueError): profile.analyze(bf16, 'flux-basic', 0)


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='flux-four-build-test-')
        cls.base, cls.root = Path(cls.temporary.name)/'base', Path(cls.temporary.name)/'new'
        builder.baseline.prepare(cls.base)
        cls.before = builder.sha(cls.base/'baseline-build.json')
        builder.prepare(cls.root, cls.base)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_native_origins_and_no_interleave_or_arrival_table(self):
        builder.verify(self.root)
        self.assertEqual(builder.sha(self.base/'baseline-build.json'), self.before)
        for arm in ('a', 'c'):
            new, old = self.root/arm, self.base/'selective'/arm
            self.assertEqual((new/native.SWIZZLE).read_bytes(), (REPO/native.SWIZZLE).read_bytes())
            self.assertNotIn('phase6_device_lookup', (new/native.SWIZZLE).read_text())
            self.assertNotIn('phase6_host_lookup', (new/'check_mapping.cu').read_text())
            for name in ('taco_runtime.cu', 'taco_codec.cuh', 'epilogue_evt.hpp'):
                self.assertEqual((new/'src/gemm_rs'/name).read_bytes(), (old/'src/gemm_rs'/name).read_bytes())
            self.assertIn('#define RTX5090_LOCAL_SELECTIVE 0', (new/'src/gemm_rs/local_transport.cu').read_text())
            self.assertIn('rtx5090_pack_tiles_kernel', (new/'src/gemm_rs/local_transport.cu').read_text())
        self.assertEqual(builder.native_root(self.root, 'flux-basic'), builder.native_root(self.root, 'flux-bf16'))

    def test_unbuilt_and_changed_source_rejected(self):
        with self.assertRaises(ValueError): builder.verify(self.root, True)
        path = self.root/'c'/native.SWIZZLE
        old = path.read_bytes()
        try:
            path.write_bytes(old+b'\n// changed\n')
            with self.assertRaises(ValueError): builder.verify(self.root)
        finally: path.write_bytes(old)

    def test_real_worker_packages_and_library_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            library = self.base/'codec-basic/libbasic_quant.so'
            library.write_bytes(b'CPU fixture only')
            plan = dict(run_id='CPU', communication=controller.base.capture_communication(),
                        cuda_visible_devices='0,1,2,3,4,5,6,7', environment={})
            with patch.object(builder, 'libraries', return_value={}):
                workers = sources.prepare_workers(self.root, out, plan)
            args = controller.arguments(['--out', str(out)])
            for arm in p.ARMS:
                scripts = out/'workers'/arm/'scripts'
                job = dict(arm=arm, mode='timing', trial=1, block=0, index=0)
                env = controller.worker_environment(args, self.root, scripts, job, out/'results', plan)
                self.assertEqual(env['E2E_POLICY'], arm)
                self.assertEqual(env['H100_MA2C_ARM'], p.NATIVE.get(arm, 'a').upper())
                self.assertIn(str(builder.native_root(self.root, arm)/'python'), env['PYTHONPATH'])
                self.assertEqual((scripts/'build'/arm).resolve(), builder.native_root(self.root, arm))
                self.assertEqual(json.loads((scripts/'selection-plan.json').read_text()), sources.selection_metadata())
                if arm == 'flux-bf16':
                    text = (scripts/'megatron_taco.py').read_text()
                    self.assertNotIn('all_to_all_single', text)
                    self.assertNotIn('ctypes', text)
                    self.assertIsNone(workers[arm]['config']['codec_library'])


class Launcher(unittest.TestCase):
    def test_complete_four_arm_controller_flow(self):
        import tp8_dp2_train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build, original = root/'build', root/'original'
            environment = fixture('flux-fused')[2]['environment']
            p.write(build/'flux-build.json', dict(baseline_root=str(original), environment=environment))
            codec = original/'codec-basic/libbasic_quant.so'
            codec.parent.mkdir(parents=True)
            codec.write_bytes(b'CPU-FIXTURE ONLY')
            for arm in p.ARMS:
                native_dir = builder.native_root(build, arm)
                native_dir.mkdir(parents=True, exist_ok=True)
                for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                    (native_dir/name).write_bytes(b'CPU-FIXTURE ONLY')
            args = controller.arguments(['--out', str(root/'out'), '--flux-build', str(build)])
            p.write(args.out/'batch-state.json', dict(protocol=p.PROTOCOL, status='CPU-FIXTURE'))
            communication = controller.base.capture_communication()
            hardware = dict(completed=True, host='CPU-FIXTURE', environment=environment,
                devices=fixture('flux-fused')[0][0]['hardware']['devices'], communication=communication,
                cuda_visible_devices='0,1,2,3,4,5,6,7')
            order = []
            def hardware_preflight(_args):
                p.write(args.out/'environment-preflight.json', hardware)
                order.append('hardware')
                return hardware
            def run_process(command, cwd, runtime, log, timeout, peer):
                dest = Path(runtime['E2E_PROCESS_OUT'])
                job = p.read(dest/'command.json')['job']
                order.append(job['mode']+':'+job['arm'])
                plan, workers = p.read(args.out/'plan.json'), p.read(args.out/'workers.json')
                worker = workers[job['arm']]
                rows = fixture(job['arm'], job['mode'])[0]
                for rank, row in enumerate(rows):
                    row.update(run_id=plan['run_id'], trial=job['trial'], block=job['block'],
                        window=job['index'], worker_config_sha256=worker['files']['config.json'],
                        communication=communication, scope_definition=worker['definition'],
                        role_plan_sha256=worker['config']['role_plan_sha256'], **worker['libraries'])
                    if job['arm'] in p.BASIC_ARMS:
                        row.update(codec_identity=worker['config']['codec_identity'], codec_library=worker['config']['codec_library'])
                    p.write(dest/f'rank{rank}.json', row)
                    if job['mode'] == 'profile':
                        with gzip.open(dest/f'rank{rank}-profile.json.gz', 'wt') as handle:
                            json.dump(dict(traceEvents=[]), handle)
                Path(log).write_text('CPU fixture only: no GPU workers launched\n')
            def replay(directory, recompute_profiles):
                self.assertFalse(recompute_profiles)
                return p.report(p.read(directory/'samples.json'), args.trials, args.blocks)
            with ExitStack() as stack:
                for context in (
                    patch.object(controller, 'REPO', root/'repo'), patch.object(controller.base, 'verify_controllers'),
                    patch.object(controller.base, 'gpu_idle'), patch.object(builder, 'verify'),
                    patch.object(controller.base, 'hardware_preflight', side_effect=hardware_preflight),
                    patch.object(load('build'), 'environment_identity', return_value=environment),
                    patch.object(controller, 'native_checks', side_effect=lambda *a: order.append('native')),
                    patch.object(tp8_dp2_train, 'run_process', side_effect=run_process),
                    patch.object(profile, 'analyze', side_effect=lambda events, arm, rank: dict(
                        protocol=p.PROTOCOL, arm=arm, rank=rank, execution_validated=True)),
                    patch.object(validation.base.basic_quant_validation, 'verify'),
                    patch.object(controller, 'replay', side_effect=replay), redirect_stdout(io.StringIO())):
                    stack.enter_context(context)
                controller.supervise(args)
            state = p.read(args.out/'batch-state.json')
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['timing_windows'], 32)
            self.assertEqual(state['formal_optimizer_steps'], 1600)
            self.assertEqual(order[:2], ['hardware', 'native'])
            self.assertEqual(order[2:6], ['smoke:'+arm for arm in p.ARMS])
            self.assertEqual(order[6:10], ['profile:'+arm for arm in p.ARMS])
            self.assertEqual(order[10:], ['timing:'+job['arm'] for job in p.schedule()])

    def test_preview_without_build_gpu_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'absent'
            for flags in ([], ['--preflight-only'], ['--transport-smoke-only']):
                result = subprocess.run(['bash', str(HERE/'run_baseline_flux.sh'), '--out', str(out),
                    '--baseline-build', str(Path(tmp)/'missing'), *flags], cwd=tmp,
                    env=dict(os.environ, RTX5090_PYTHON=sys.executable), text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout[result.stdout.index('{'):])
                self.assertEqual(plan['arms'], list(p.ARMS))
                self.assertEqual(plan['formal_optimizer_steps'], 0 if flags else 1600)
                self.assertFalse(out.exists())

    def test_bad_flags_rejected_before_work(self):
        for flags in (['--blocks', '6'], ['--report'], ['--inside'], ['--execute', '--report', '--out', '/tmp/none']):
            with self.assertRaises(ValueError): controller.arguments(flags)

    def test_source_error_stops_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = controller.arguments(['--execute', '--out', str(root)])
            p.write(root/'batch-state.json', {})
            with patch.object(controller, 'REPO', root), patch.object(controller.base, 'verify_controllers'), \
                 patch.object(sources, 'worker_source', side_effect=ValueError('source gate')), \
                 patch.object(controller.base, 'hardware_preflight') as gpu:
                with self.assertRaisesRegex(ValueError, 'source gate'): controller.supervise(args)
                gpu.assert_not_called()
            self.assertEqual(p.read(root/'batch-state.json')['status'], 'failed')


if __name__ == '__main__': unittest.main()
