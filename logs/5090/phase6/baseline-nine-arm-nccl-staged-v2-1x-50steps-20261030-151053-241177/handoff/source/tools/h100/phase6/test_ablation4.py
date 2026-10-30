"""CPU-only checks; simulated execution evidence never leaves temporary folders."""
import ast
import contextlib
import copy
import csv
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import ablation4 as cli
import ablation4_build as builder
import ablation4_prepare as preparation
import ablation4_protocol as sp
import ablation4_report as report
import ablation4_run as runner
import ablation4_runtime as communication
import ablation4_sources as sources
import scope_build
import scope_calibrate
import test_scope as scope_tests
from common import read, write, sha


class ProtocolTests(unittest.TestCase):
    def test_five_groups_and_balanced_reversed_schedule(self):
        self.assertEqual(len(sp.POLICIES), 5)
        self.assertEqual(len(sp.COMPARISONS), 10)
        self.assertEqual(len(sp.PRIMARY), 4)
        jobs = sp.schedule(rounds=2)
        rows = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b]
                for r in (1, 2) for b in range(5)]
        self.assertEqual(rows[5:], [r[::-1] for r in rows[:5][::-1]])
        for i in range(5): self.assertEqual({r[i] for r in rows[:5]}, set(sp.POLICIES))
        self.assertEqual(sp.schedule(), [j for j in jobs if j['round'] == 1])
        for p in sp.POLICIES:
            self.assertEqual(sp.definition(p)['roles'], ['attn', 'mlp'])
        self.assertEqual([sp.definition(p)['order'] for p in sp.POLICIES], ['native', 'original', 'base', 'arrival', 'arrival'])
        self.assertEqual([sp.definition(p)['quantization'] for p in sp.POLICIES], ['all_remote']*4+['targeted'])
        self.assertEqual([sp.is_fused_v2(p) for p in sp.POLICIES], [False, True, True, True, False])
        self.assertEqual([sp.definition(p)['workspace'] for p in sp.POLICIES], ['tensor', 'single', 'single', 'single', 'double'])
        for n in (0, 4, 6, True):
            with self.assertRaises(ValueError): sp.schedule(n)
        for n in (0, 3, True):
            with self.assertRaises(ValueError): sp.schedule(rounds=n)

    def test_communication_freezing_and_config_drift(self):
        env = {'PATH': '/bin', 'NCCL_NET': 'IB', 'NCCL_DEBUG': 'TRACE', 'NCCL_NVLS_ENABLE': '1',
               'NCCL_DEBUG_FILE': '/tmp/old', 'GLOO_SOCKET_IFNAME': 'bond0'}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(communication, 'config_files', return_value={'/etc/nccl.conf': None}):
            record = communication.capture()
            self.assertEqual(record['environment']['NCCL_NVLS_ENABLE'], '0')
            self.assertEqual(record['environment']['NCCL_NET'], 'IB')
            self.assertEqual(record['environment']['NCCL_DEBUG'], 'WARN')
            self.assertNotIn('NCCL_DEBUG_FILE', record['environment'])
            actual = communication.apply(dict(env, NCCL_IB_DISABLE='1'), record)
            self.assertNotIn('NCCL_IB_DISABLE', actual)
            with cli.communication_environment(record):
                self.assertEqual(communication.verify(record), record)
                with mock.patch.object(communication, 'config_files', return_value={'/etc/nccl.conf': 'changed'}):
                    with self.assertRaises(RuntimeError): communication.verify(record)
                os.environ['NCCL_NVLS_ENABLE'] = '1'
                with self.assertRaises(RuntimeError): communication.verify(record)
            self.assertEqual(dict(os.environ), env)
            self.assertNotIn('NCCL_NVLS_ENABLE', communication.capture('auto')['environment'])

    def test_adapter_routes_survive_audit_for_all_five_groups(self):
        # Exercise the real generated adapter. The fixture's candidate owns two
        # buffers; adapt only its mock to the original fused-v2 single workspace.
        test_source = sources.adapter_source()
        helper_source = __import__('inspect').getsource(scope_tests.AdapterTests.exercise)
        helper_source = __import__('textwrap').dedent(helper_source).replace('owns_output = True', 'owns_output = not sp.is_fused_v2(policy)')
        anchor = "            self.plan_sha256 = role+'-plan'"
        helper_source = helper_source.replace(anchor, anchor+"\n            if sp.is_fused_v2(policy): self.plan_sha256 = None")
        anchor = '        def forward(self, x, w):'
        helper_source = helper_source.replace(anchor, "            if sp.is_fused_v2(policy): self.workspace_info = [dict(shape=sp.SHAPES[role], packet_ptr=id(self))]\n"+anchor)
        ns = dict(scope_tests.__dict__, sp=sp, sources=types.SimpleNamespace(adapter_source=lambda: test_source))
        exec(helper_source, ns)
        for p in sp.POLICIES:
            for mode in ('timing', 'profile'):
                with self.subTest(policy=p, mode=mode): ns['exercise'](self, p, mode)

    def test_real_taco_dispatch_uses_original_codec_for_both_roles(self):
        class Codec:
            def __init__(self, group, m, n, k, **kwargs):
                self.args = (m, n, k, kwargs); self.rank = 0
                self.peers = [types.SimpleNamespace(data_ptr=lambda: id(self))]
        torch = types.ModuleType('torch'); torch.no_grad = lambda: lambda f: f
        dist = types.ModuleType('torch.distributed'); dist.get_world_size = lambda g: 8; torch.distributed = dist
        mocks = {'torch': torch, 'torch.distributed': dist, 'routing': sp,
                 'reference': types.SimpleNamespace(roundtrip=lambda x: x), 'flux': types.ModuleType('flux'),
                 'flux.gemm_rs_taco': types.SimpleNamespace(GemmRSTaco=Codec)}
        with mock.patch.dict(sys.modules, mocks):
            ns = {}; exec(sources.taco_source(), ns)
            ops = []
            for policy in sp.POLICIES[1:4]:
                for role, (m, n, k) in sp.SHAPES.items():
                    op = ns['TacoForward'](None, m, n, k//8, policy, role); ops.append(op)
                    self.assertEqual(op.codec.args, (m, n, k//8, dict(placement='fused')))
                    self.assertFalse(op.owns_output); self.assertIsNone(op.plan_sha256)
                    self.assertEqual(op.workspace_info, [dict(shape=[m, n, k], packet_ptr=id(op.codec))])
            self.assertIsNot(ops[0].codec, ops[1].codec)
        node = next(n for n in ast.parse(sources.adapter_source()).body if isinstance(n, ast.ClassDef) and n.name == 'GemmRSFunction')
        namespace = dict(torch=types.SimpleNamespace(autograd=types.SimpleNamespace(Function=object)),
                         parallel_state=types.SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-autograd-adapter>', 'exec'), namespace)
        class Value:
            shape = (2048, 1, 512)
            def __init__(self): self.clones = 0
            def reshape(self, *shape): return self
            def contiguous(self): return self
            def clone(self): self.clones += 1; return self
        for policy in sp.POLICIES[1:]:
            value = Value()
            op = types.SimpleNamespace(owns_output=not sp.is_fused_v2(policy), forward=lambda *a, **kw: value)
            namespace['GemmRSFunction'].forward(types.SimpleNamespace(save_for_backward=lambda *a: None),
                value, types.SimpleNamespace(shape=(4096, 512)), op, None)
            self.assertEqual(value.clones, int(sp.is_fused_v2(policy)))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='ablation4-CPU-FIXTURE-'); cls.root = Path(cls.tmp.name)
        cls.build = builder.prepare(cls.root/'build', source_only=True)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, Path('/absent'), source_only=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_dry_run_is_cpu_only_and_rehearsal_cannot_execute(self):
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('launched')), \
                mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('queried')), contextlib.redirect_stdout(io.StringIO()):
            jobs = runner.run(self.compare)
            self.assertEqual(len(jobs), 35)
            self.assertEqual(len(list(runner.mapping_jobs(self.compare, report.verify_inputs(self.compare)))), 8)
            with self.assertRaises(ValueError): runner.run(self.compare, execute=True)
        self.assertNotIn('torch', sys.modules)
        for p in (self.compare/'scripts').glob('*.py'): ast.parse(p.read_text())
        result = subprocess.run([sys.executable, str(self.compare/'scripts/worker.py')], capture_output=True, text=True,
                                env=dict(os.environ, H100_ABLATION4_EXECUTE='1'))
        self.assertNotEqual(result.returncode, 0); self.assertIn('Source-only', result.stderr)
        with self.assertRaisesRegex(ValueError, 'both explicit'): builder.prepare(self.root/'missing-plans')

    def test_build_identity_cannot_label_candidate_as_original_fused(self):
        m = builder.verify(self.build)
        plans = read(self.build/'plans.json'); candidate = read(self.build/'candidate/manifest.json')
        fused = read(self.build/'fused/manifest.json')
        for field, value in (('base', 'interleaved'), ('implementation', 'candidate'), ('sampling', True)):
            changed = copy.deepcopy(fused); changed['variants']['taco_fused'][field] = value
            with self.assertRaises(ValueError): builder.validate_snapshots(m, plans, candidate, changed)
        original = fused['variants']['taco_fused']['source_sha256']
        for name in read(builder.HERE/'fused_v2/provenance.json')['files']:
            changed, record = copy.deepcopy(fused), copy.deepcopy(m)
            for policy, variant in builder.FUSED_NAMES.items():
                changed['variants'][variant]['source_sha256'][name] = 'f'*64
                record['fused_source_sha256'][policy][name] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'Pinned fused-v2'): builder.validate_snapshots(record, plans, candidate, changed)
        self.assertEqual(original, m['fused_source_sha256'][sp.POLICIES[1]])
        for policy in sp.POLICIES[2:4]:
            other = fused['variants'][builder.FUSED_NAMES[policy]]['source_sha256']
            self.assertEqual({p for p in original if original[p] != other[p]}, builder.ORDER_FILES)
        for policy in sp.POLICIES[2:]:
            self.assertEqual(len(sp.role_plans(plans, policy)), 2)
        self.assertEqual(sp.role_plans(plans, sp.POLICIES[3])['attn']['entry']['coords'],
                         sp.role_plans(plans, sp.POLICIES[4])['attn']['entry']['coords'])

    def test_calibration_freezes_same_communication_before_torch(self):
        sampler = scope_build.prepare(self.root/'sampler', sampler=True, roles=('attn', 'mlp'))
        cal = cli.calibration_prepare(self.root/'cal', sampler, source_only=True)
        plan = scope_calibrate.verify(cal)
        self.assertEqual(plan['communication'], read(cal/'config.json')['communication'])
        source = (cal/'worker.py').read_text(); ast.parse(source)
        self.assertLess(source.index('communication = verify_communication'), source.index('import torch'))
        with self.assertRaises(ValueError): scope_calibrate.run(cal, execute=True)

    def test_compile_snapshot_remains_verifiable_after_failure_and_retry(self):
        root = builder.prepare(self.root/'mock-compile', source_only=True)
        calls = []
        def compiler(child, *args):
            calls.append(child.name); manifest = read(child/'manifest.json')
            manifest['status'] = 'compile_failed' if calls == ['fused', 'candidate'] else 'built'
            write(child/'manifest.json', manifest)
            if manifest['status'] == 'compile_failed': raise RuntimeError('simulated linker failure')
            for name in manifest['variants']:
                for artifact in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping'):
                    (child/name/artifact).write_text('CPU FIXTURE, NOT A BINARY')
        with mock.patch.object(builder.build, 'compile_build', side_effect=compiler):
            with self.assertRaisesRegex(RuntimeError, 'simulated'): builder.compile_build(root, Path('/unused'))
            builder.verify(root)
            builder.compile_build(root, Path('/unused'))
        self.assertEqual(calls, ['fused', 'candidate', 'candidate'])
        builder.verify(root)
        with self.assertRaises(ValueError): builder.verify(root, built=True)

    def test_complete_reports_and_failure_paths(self):
        for rounds in (1, 2):
            root = preparation.prepare(self.root/f'SYNTHETIC-NOT-GPU-{rounds}', self.build, Path('/absent'), source_only=True, rounds=rounds)
            info = executable_fixture(root); make_evidence(root, info)
            result = report.report(root, offline=True)
            self.assertTrue(result['completed']); self.assertEqual(result['round_count'], rounds)
            self.assertEqual(result['timing_windows'], 25*rounds)
            self.assertEqual(result['timing_rank_records'], 200*rounds)
            self.assertEqual(result['execution_timing'], dict(warmup_steps=10, timed_steps=60))
            self.assertEqual(result['formal_optimizer_steps'], 1500*rounds)
            self.assertEqual(result['preflight_rank_records'], 80)
            self.assertEqual(result['profile_traces'], 40); self.assertEqual(result['candidate_shape_mappings'], 8)
            self.assertEqual(len(result['rounds']['combined']['comparisons']), 10)
            self.assertEqual(result['sensitivity']['1']['remaining_blocks'], 4)
            self.assertEqual(result['rounds']['1']['mean_ms'][sp.POLICIES[0]], 1200.)
            if rounds == 1: self.assertEqual(result['rounds']['combined'], result['rounds']['1'])
            job = next(j for j in info['schedule'] if sp.is_fused_v2(j['policy']))
            path = root/'results'/job['directory']/'rank7.json'; original = read(path)
            self.assertEqual(len(original['losses']), 60)
            for field, value in (('communication', {}), ('timed_calls', {}), ('sampling', True),
                                 ('losses', original['losses'][:20]), ('timed_steps', 20)):
                write(path, dict(original, **{field: value}))
                with self.assertRaises(ValueError): report.window(root, path.parent, job['policy'], 'timing', job['block'], job['index'])
            changed = copy.deepcopy(original); checks = list(changed['workspace_checks'].values())
            checks[1][0]['packet_ptr'] = checks[0][0]['packet_ptr']; write(path, changed)
            with self.assertRaisesRegex(ValueError, 'alias'): report.window(root, path.parent, job['policy'], 'timing', job['block'], job['index'])
            path.unlink()
            with self.assertRaises(FileNotFoundError): report.report(root, offline=True)
            write(path, original)
            extra = root/'results/round3/window-000-extra'; extra.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Missing or extra'): report.report(root, offline=True)
            extra.rmdir(); extra.parent.rmdir()


def executable_fixture(root):
    """Mock metadata only: generated worker retains its source-only hard stop."""
    info = read(root/'submission.json'); info.update(source_only=False, executable=True)
    plans = read(root/'scripts/selection-plan.json'); plans.update(source_only=False, gpu_calibrated=True)
    plans['parent'].update(source_only=False, gpu_calibrated=True)
    for record in plans['parent']['roles'].values():
        record.update(test_only=False, gpu_calibrated=True, hardware=info['calibration_hardware'],
                      origin=dict(kind='scope_bf16_calibration'), inputs={'/CPU-FIXTURE': 'f'*64})
    write(root/'scripts/selection-plan.json', plans); digest = sha(root/'scripts/selection-plan.json')
    seal = read(root/'provenance/preparation.json'); seal.update(source_only=False, executable=True, plans_sha256=digest)
    write(root/'provenance/preparation.json', seal)
    candidate = read(root/'provenance/candidate-manifest.json'); candidate.update(source_only=False, diagnostic_only=False)
    candidate['files']['plans.json'] = hashlib.sha256((json.dumps(plans['parent'], indent=2, ensure_ascii=False)+'\n').encode()).hexdigest()
    write(root/'provenance/candidate-manifest.json', candidate)
    m = read(root/'provenance/build-manifest.json'); m.update(source_only=False, diagnostic_only=False)
    m['files']['plans.json'] = digest; m['candidate_manifest_sha256'] = sha(root/'provenance/candidate-manifest.json')
    write(root/'provenance/build-manifest.json', m); info['build_manifest']['sha256'] = sha(root/'provenance/build-manifest.json')
    for name in ('fused-manifest.json', 'candidate-manifest.json'):
        path = root/'provenance'/name; child = read(path)
        child.update(status='built', source_only=False, diagnostic_only=False); write(path, child)
    # Updating the candidate's compiled status changes its snapshot hash.
    m['candidate_manifest_sha256'] = sha(root/'provenance/candidate-manifest.json')
    write(root/'provenance/build-manifest.json', m); info['build_manifest']['sha256'] = sha(root/'provenance/build-manifest.json')
    for p, v in info['variants'].items():
        if not sp.is_native(p):
            v['libraries'] = {name: dict(path='/CPU-FIXTURE/'+v['order']+'/'+name, sha256=('a' if v['order']=='base' else 'b')*64)
                              for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
    info['files'] = {p: sha(root/p) for p in info['files']}; write(root/'submission.json', info)
    write(root/'start-gpu-check.json', dict(passed=True, run_id=info['run_id'], synthetic=True))
    return info


def make_evidence(root, info):
    for job in runner.jobs(info, 'all'):
        original_read = scope_tests.read
        def per_policy(path):
            data = original_read(path)
            return dict(roles=sp.role_plans(data, job['policy'])) if Path(path).name == 'selection-plan.json' else data
        with mock.patch.object(scope_tests, 'sp', sp), mock.patch.object(scope_tests, 'read', side_effect=per_policy):
            scope_tests.make_window(root, info, job)
        folder = root/'results'/job['directory']
        write(folder/'command.json', dict(job, scope=sp.definition(job['policy']), communication=info['communication']))
        for rank in range(8):
            path = folder/f'rank{rank}.json'; row = read(path); row['communication'] = info['communication']
            if sp.is_fused_v2(job['policy']):
                for c in row['selection_checks'].values(): c.update(mask_sha256=None, plan_sha256=None, wire_bytes=7*256*32*136)
                row['workspace_checks'] = {n: [dict(shape=s[0]['shape'], packet_ptr=s[0]['packet_ptr'])] for n, s in row['workspace_checks'].items()}
            write(path, row)
    plans = read(root/'scripts/selection-plan.json')
    for policy, role, _ in runner.mapping_jobs(root, info):
        original = sp.is_original(policy)
        coords = [[((i//32+2*r)%16)*32+i%32 for i in range(512)] for r in range(8)] if original else sp.role_plans(plans, policy)[role]['entry']['coords']
        with (root/'results'/f'mapping-{policy}-{role}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['rank', 'index', 'm', 'n', 'destination', 'sm_occupancy', 'base_m', 'base_n']); writer.writeheader()
            for r, row in enumerate(coords):
                for i, tile in enumerate(row):
                    writer.writerow(dict(rank=r, index=i, m=tile//32, n=tile%32, destination=tile//64, sm_occupancy=2, base_m=i//32, base_n=i%32))
    report.preflight(root, offline=True)


if __name__ == '__main__': unittest.main()
