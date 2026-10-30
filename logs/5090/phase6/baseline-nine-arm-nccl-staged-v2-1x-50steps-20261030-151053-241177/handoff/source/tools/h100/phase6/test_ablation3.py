"""CPU-only nine-group tests. All simulated GPU evidence stays in temp directories."""
import contextlib
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import ablation3 as cli
import ablation3_protocol as sp
import ablation3_plan as planner
import ablation3_build as builder
import ablation3_prepare as preparation
import ablation3_sources as sources
import ablation3_run as runner
import ablation3_report as report
import scope_plan
import test_scope as scope_tests
from common import read, write, sha
from ablation_plan import mask_sha


def parent_fixture():
    return dict(source_only=True, gpu_calibrated=False, roles={r: scope_plan.cpu_fixture(r) for r in sp.SHAPES})


class ProtocolTests(unittest.TestCase):
    def test_nine_policy_schedule_and_scope_coverage(self):
        jobs = sp.schedule(); self.assertEqual(len(jobs), 162)
        self.assertEqual(len(set(sp.POLICIES)), 9)
        self.assertEqual(sum(len(sp.definition(p)['roles']) for p in sp.POLICIES), 12)
        rows = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b] for r in (1, 2) for b in range(9)]
        self.assertEqual(rows[9:], [row[::-1] for row in rows[:9][::-1]])
        for i in range(9): self.assertEqual({row[i] for row in rows[:9]}, set(sp.POLICIES))
        for blocks in (0, 3, 4, 5, 10):
            with self.assertRaises(ValueError): sp.schedule(blocks)
        self.assertEqual(len(sp.schedule(18)), 324)
        with self.assertRaises(ValueError): sp.definition('native_taco')

    def test_single_round_preserves_first_round_and_rejects_invalid_counts(self):
        for blocks in (9, 18):
            single = sp.schedule(blocks, rounds=1)
            self.assertEqual(single, [job for job in sp.schedule(blocks) if job['round'] == 1])
            self.assertEqual(len(single), 9*blocks)
            self.assertEqual({job['round'] for job in single}, {1})
        for rounds in (0, 3, -1, True, False, 1.0, '1', None):
            with self.subTest(rounds=rounds), self.assertRaisesRegex(ValueError, 'Rounds'):
                sp.schedule(rounds=rounds)

    def test_prepare_cli_rounds_and_default(self):
        for options, count in (([], 2), (['--rounds', '1'], 1), (['--rounds', '2'], 2)):
            argv = ['ablation3.py', 'prepare', '/unused/compare', '--build', '/unused/build',
                    '--megatron', '/unused/megatron', '--source-only', *options]
            with self.subTest(rounds=count), mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(preparation, 'prepare') as prepare, contextlib.redirect_stdout(io.StringIO()):
                cli.main()
                prepare.assert_called_once_with(Path('/unused/compare'), Path('/unused/build'),
                                                Path('/unused/megatron'), 9, True, rounds=count)
        with mock.patch.object(sys, 'argv', argv[:-2] + ['--rounds', '3']), \
                mock.patch.object(preparation, 'prepare') as prepare, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main()
        prepare.assert_not_called()

    def test_all_remote_and_sparse_masks_are_independent_of_order(self):
        p = planner.bundle(parent_fixture())
        for scope in sp.SCOPES:
            a, b, c = [p['variants'][sp.policy_id(f, scope)]['roles'] for f in sp.PREFIXES]
            for role in a:
                ae, be, ce = [v[role]['entry'] for v in (a, b, c)]
                self.assertEqual(ae['mask'], be['mask'])
                self.assertEqual(be['maps'], ce['maps']); self.assertEqual(be['coords'], ce['coords'])
                self.assertEqual(ae['maps'], [list(range(512)) for _ in range(8)])
                self.assertEqual(ae['selected_per_source'], [448]*8)
                self.assertTrue(all(q <= 7 for q in ce['selected_per_source']))
                self.assertNotEqual(ae['maps'], be['maps'])
                for r in range(8): self.assertFalse(any(ae['mask'][r][r*64:(r+1)*64]))
        for role in sp.SHAPES:
            for f in sp.PREFIXES:
                single = p['variants'][sp.policy_id(f, role)]['roles'][role]
                both = p['variants'][sp.policy_id(f, 'attn_mlp')]['roles'][role]
                self.assertEqual(single, both)

    def test_semantic_plan_tampering_rejected_even_with_updated_hashes(self):
        p = planner.bundle(parent_fixture()); key = sp.policy_id('B', 'attn')
        for change in ('local', 'sparse_full', 'order', 'shape'):
            q = copy.deepcopy(p); r = q['variants'][key]['roles']['attn']; e = r['entry']
            if change == 'local': e['mask'][0][0] = 1
            elif change == 'sparse_full': e['mask'][0][64] = 0
            elif change == 'order': e['maps'] = [list(range(512)) for _ in range(8)]; e['coords'] = copy.deepcopy(e['base_coords'])
            else: e['shape'] = sp.SHAPES['mlp']
            r.update(entry_sha256=sp.digest(e), mask_sha256=mask_sha(e['mask']))
            with self.assertRaises(ValueError): planner.validate(q)
        with self.assertRaises(ValueError): sp.selection_entry(p, key, 'mlp', *sp.SHAPES['mlp'])
        with self.assertRaises(ValueError): sp.selection_entry(p, key, 'attn', *sp.SHAPES['mlp'])
        changed = copy.deepcopy(p); changed.update(source_only=False, gpu_calibrated=True)
        with self.assertRaises(ValueError): planner.validate(changed)

    def test_18_comparisons_pair_best_and_negative_gains(self):
        self.assertEqual(len(sp.PRIMARY), 9); self.assertEqual(len(sp.RANGE_COMPARISONS), 9)
        values = {p: [100.]*3 for p in sp.POLICIES}
        a, b, c = [sp.policy_id(f, 'mlp') for f in sp.PREFIXES]
        values[a] = [100., 300., 200.]; values[b] = [50., 150., 50.]; values[c] = [200., 200., 200.]
        blocks = [dict(round=1, block=i) for i in range(3)]
        s = report.summarize(values, blocks)
        self.assertEqual(len(s['comparisons']), 18)
        self.assertEqual(s['speedups'][b+'/'+a]['median'], 2.)
        self.assertEqual(s['speedups'][b+'/'+a]['best']['block'], 2)
        self.assertEqual(s['speedups'][b+'/'+a]['best']['reference_ms'], 200.)
        self.assertLess(s['comparisons'][c+'/'+b]['percent'], 0)
        expected = 100*(1-math.exp(sum(math.log(x/y) for x,y in zip(values[b],values[a]))/3))
        self.assertAlmostEqual(s['comparisons'][b+'/'+a]['percent'], expected)

    def test_whole_nine_group_filter_and_zero_one_blocks(self):
        blocks = [dict(round=1, block=i) for i in range(9)]
        values = {p: [100.]*9 for p in sp.POLICIES}; values[sp.POLICIES[-1]][3] = 111.
        s = report.sensitivity(values, blocks)
        self.assertEqual(s['remaining_blocks'], 8); self.assertEqual(s['excluded'][0]['block'], 3)
        self.assertEqual({v['blocks'] for v in s['summary']['comparisons'].values()}, {8})
        for i,p in enumerate(sp.POLICIES): values[p] = [120. if i==j else 100. for j in range(9)]
        self.assertEqual(report.sensitivity(values, blocks)['summary']['status'], '无可用配对块')
        one = report.summarize({p:[1.] for p in sp.POLICIES}, blocks[:1])
        self.assertTrue(all(v['ci95'] is None for v in one['comparisons'].values()))


class AdapterTests(unittest.TestCase):
    def test_real_adapter_nine_routes_before_after_audit(self):
        helper = scope_tests.AdapterTests()
        with mock.patch.object(scope_tests, 'sp', sp):
            for policy in sp.POLICIES:
                for mode in ('timing', 'profile'):
                    with self.subTest(policy=policy, mode=mode): helper.exercise(policy, mode)

    def test_codec_receives_correct_full_or_sparse_physical_mask(self):
        class Codec:
            owns_configuration = True
            def __init__(self, group, m, n, k, **kwargs):
                self.kwargs = kwargs
                self.slots = [types.SimpleNamespace(m=m,n=n,k_local=k,rank=0,selection_sha256=mask_sha(kwargs['selected']),
                    peers=[types.SimpleNamespace(data_ptr=lambda:123)],selected=types.SimpleNamespace(data_ptr=lambda:456)) for _ in range(2)]
        torch = types.ModuleType('torch'); torch.no_grad = lambda: lambda f: f
        dist = types.ModuleType('torch.distributed'); dist.get_world_size = lambda g: 8; torch.distributed = dist
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {
                'torch':torch, 'torch.distributed':dist, 'routing':sp,
                'reference':types.SimpleNamespace(roundtrip=lambda x:x),
                'flux':types.ModuleType('flux'), 'flux.gemm_rs_taco':types.SimpleNamespace(GemmRSTacoDoubleBuffered=Codec)}):
            root=Path(tmp); plans=planner.bundle(parent_fixture()); write(root/'selection-plan.json',plans)
            ns={'__file__':str(root/'taco_support.py')}; exec(sources.taco_source(),ns)
            for p in sp.POLICIES:
                for role in sp.definition(p)['roles']:
                    m,n,k=sp.SHAPES[role]; op=ns['TacoForward'](None,m,n,k//8,p,role)
                    r=sp.role_plans(plans,p)[role]
                    self.assertEqual(op.codec.kwargs['selected'],r['entry']['mask'])
                    self.assertEqual(op.codec.kwargs['quantization_mode'],sp.definition(p)['quantization'])
                    self.assertEqual(op.plan_sha256,r['entry_sha256'])
                    self.assertEqual(sum(op.codec.kwargs['selected'][0]),448 if sp.definition(p)['family']!='C' else 1)

    def test_real_worker_gate_precedes_torch_import(self):
        worker=sources.worker_source()
        self.assertLess(worker.index('H100_ABLATION3_EXECUTE'),worker.index('import torch'))
        self.assertNotIn('H100_SCOPE_EXECUTE',worker)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'worker.py';path.write_text(worker)
            env={k:v for k,v in os.environ.items() if k!='H100_ABLATION3_EXECUTE'}
            p=subprocess.run([sys.executable,str(path)],env=env,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('requires ablation3 --execute',p.stderr)
            self.assertNotIn('No module named',p.stderr)

    def test_role_profiles_reject_cross_scope_and_count_only_cpu_spans(self):
        for p in sp.POLICIES:
            events=profile_fixture(p);expected=report.profile_roles(events,p)
            self.assertEqual({v['calls'] for v in expected.values()},{128})
            mirrors=[dict(e,cat='gpu_user_annotation') for e in events if e.get('cat')=='user_annotation']
            self.assertEqual(report.profile_roles(events+mirrors,p),expected)
        events=profile_fixture(sp.policy_id('A','mlp'))
        with self.assertRaises(ValueError):report.profile_roles(events,sp.policy_id('A','attn'))
        for e in events:
            if e.get('name')=='flux_bf16_sm80':e['name']='flux_bf16_sm90'
        with self.assertRaises(ValueError):report.profile_roles(events,sp.policy_id('A','mlp'))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='ablation3-CPU-FIXTURE-');cls.root=Path(cls.tmp.name)
        cls.build=builder.prepare(cls.root/'build',source_only=True)
        cls.compare=preparation.prepare(cls.root/'compare',cls.build,Path('/absent'),source_only=True)
        cls.single=preparation.prepare(cls.root/'single',cls.build,Path('/absent'),source_only=True,rounds=1)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_prepare_dry_run_cli_and_compile_admission_without_gpu(self):
        with mock.patch.object(subprocess,'run',side_effect=AssertionError('GPU/process launch')), \
             mock.patch.object(subprocess,'check_output',side_effect=AssertionError('GPU query')):
            with contextlib.redirect_stdout(io.StringIO()):
                jobs=runner.run(self.compare)
                with mock.patch.object(sys,'argv',['ablation3.py','run',str(self.compare)]):cli.main()
            self.assertEqual(len(jobs),180)
            info=report.verify_inputs(self.compare)
            self.assertEqual(len(list(runner.mapping_jobs(self.compare,info))),12)
            with self.assertRaises(ValueError):runner.run(self.compare,execute=True)
        self.assertNotIn('torch',sys.modules)
        with self.assertRaises(ValueError):builder.verify(self.build,built=True)
        m=builder.verify(self.build)
        a,b=[m['builds'][order]['source_sha256'] for order in builder.ORDERS]
        self.assertEqual({k for k in a if a[k]!=b[k]}, {'phase6_host.hpp',builder.scope_build.SWIZZLE})

    def test_single_round_freezes_81_windows_and_keeps_full_preflight(self):
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('GPU/process launch')), \
                mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('GPU query')):
            info = report.verify_inputs(self.single)
            self.assertEqual(info['rounds'], 1)
            self.assertEqual(info['timing_windows'], 81)
            self.assertEqual(read(self.single/'scripts/config.json')['repetitions'], 1)
            self.assertEqual(read(self.single/'provenance/preparation.json')['rounds'], 1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(len(runner.run(self.single)), 99)
                self.assertEqual(len(runner.run(self.single, stage='preflight')), 18)
                timing = runner.run(self.single, stage='timing')
            self.assertEqual(len(timing), 81)
            self.assertEqual({job['round'] for job in timing}, {1})
            self.assertEqual(len(list(runner.mapping_jobs(self.single, info))), 12)
            with self.assertRaises(ValueError): runner.run(self.single, execute=True)
        self.assertNotIn('torch', sys.modules)

    def test_round_count_and_schedule_tampering_rejected_even_with_new_hashes(self):
        root = self.single
        paths = [root/name for name in ('submission.json', 'provenance/preparation.json',
                                        'scripts/config.json', 'schedule.json')]
        originals = {p: p.read_bytes() for p in paths}
        for mutation in ('submission', 'seal', 'config', 'missing_round_metadata', 'missing_window', 'extra_round'):
            with self.subTest(mutation=mutation):
                try:
                    info = read(root/'submission.json')
                    if mutation == 'submission': info['rounds'] = 2
                    elif mutation in ('seal', 'missing_round_metadata'):
                        seal = read(root/'provenance/preparation.json')
                        if mutation == 'seal': seal['rounds'] = 2
                        else:
                            info.pop('rounds'); seal.pop('rounds')
                        write(root/'provenance/preparation.json', seal)
                    elif mutation == 'config':
                        cfg = read(root/'scripts/config.json'); cfg['repetitions'] = 2
                        write(root/'scripts/config.json', cfg)
                    else:
                        info['schedule'] = info['schedule'][:-1] if mutation == 'missing_window' else sp.schedule()
                        info['timing_windows'] = len(info['schedule'])
                        write(root/'schedule.json', info['schedule'])
                    info['files'] = {p: sha(root/p) for p in info['files']}
                    write(root/'submission.json', info)
                    with self.assertRaises(ValueError): report.verify_inputs(root)
                finally:
                    for path, original in originals.items(): path.write_bytes(original)

    def test_legacy_two_round_metadata_still_requires_both_rounds(self):
        root = self.compare
        paths = [root/name for name in ('submission.json', 'provenance/preparation.json', 'schedule.json')]
        originals = {p: p.read_bytes() for p in paths}
        try:
            info = read(root/'submission.json'); info.pop('rounds')
            seal = read(root/'provenance/preparation.json'); seal.pop('rounds')
            write(root/'provenance/preparation.json', seal)
            info['files']['provenance/preparation.json'] = sha(root/'provenance/preparation.json')
            write(root/'submission.json', info)
            verified = report.verify_inputs(root)
            self.assertEqual(verified.get('rounds', 2), 2)
            self.assertEqual(verified['timing_windows'], 162)
            # An unfinished old two-round run cannot be relabelled as a completed single round.
            info.update(rounds=1, schedule=sp.schedule(rounds=1), timing_windows=81)
            write(root/'schedule.json', info['schedule'])
            info['files']['schedule.json'] = sha(root/'schedule.json')
            write(root/'submission.json', info)
            with self.assertRaisesRegex(ValueError, 'Frozen round count'): report.verify_inputs(root)
        finally:
            for path, original in originals.items(): path.write_bytes(original)

    def test_frozen_input_and_library_order_mismatch_rejected(self):
        for name in ('scripts/taco_support.py','scripts/selection-plan.json','schedule.json'):
            p=self.compare/name;original=p.read_bytes()
            try:
                p.write_bytes(original+b'\n')
                with self.assertRaises(ValueError):report.verify_inputs(self.compare)
            finally:p.write_bytes(original)
        path=self.compare/'submission.json';original=read(path)
        try:
            changed=copy.deepcopy(original);p=sp.policy_id('A','mlp')
            changed['variants'][p]['root']=changed['variants'][sp.policy_id('B','mlp')]['root'];write(path,changed)
            with self.assertRaises(ValueError):report.verify_inputs(self.compare)
        finally:write(path,original)

    def test_source_only_worker_stays_disabled(self):
        p=subprocess.run([sys.executable,str(self.compare/'scripts/worker.py')],
                         env=dict(os.environ,H100_ABLATION3_EXECUTE='1'),capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('Source-only',p.stderr)
        path=self.compare/'submission.json';original=read(path)
        try:
            write(path,dict(original,source_only=False,executable=True))
            with self.assertRaises(ValueError):report.verify_inputs(self.compare)
        finally:write(path,original)

    def test_complete_synthetic_180_window_report_and_failure_paths(self):
        root=preparation.prepare(self.root/'SYNTHETIC-NOT-GPU',self.build,Path('/absent'),source_only=True)
        info=executable_fixture(root)
        make_evidence(root, info)
        result=report.report(root,offline=True)
        self.assertEqual(result['round_count'], 2)
        self.assertEqual(result['timing_rank_records'],1296);self.assertEqual(result['preflight_rank_records'],144)
        self.assertEqual(result['profile_traces'],72);self.assertEqual(result['candidate_shape_mappings'],12)
        self.assertEqual(len(result['all_samples']),162);self.assertEqual(result['rounds']['combined']['blocks'],18)
        self.assertEqual(len(result['rounds']['1']['comparisons']),18)
        self.assertEqual(result['sensitivity']['1']['remaining_blocks'],8)
        job=info['schedule'][0];path=root/'results'/job['directory']/'rank7.json';original=read(path)
        for mutate in ({'post_audit_routes':{}},{'timed_calls':{}},{'scope_definition':{}},{'sampling':True}):
            write(path,dict(original,**mutate))
            with self.assertRaises(ValueError):report.window(root,path.parent,job['policy'],'timing',job['block'],job['index'])
        write(path,original)
        changed=copy.deepcopy(original);name=next(iter(changed['selection_checks']))
        changed['selection_checks'][name]['wire_bytes']=1;write(path,changed)
        with self.assertRaisesRegex(ValueError,'wire bytes'):report.window(root,path.parent,job['policy'],'timing',job['block'],job['index'])
        changed=copy.deepcopy(original);slots=list(changed['workspace_checks'].values());slots[1][0]['packet_ptr']=slots[0][0]['packet_ptr'];write(path,changed)
        with self.assertRaisesRegex(ValueError,'alias'):report.window(root,path.parent,job['policy'],'timing',job['block'],job['index'])
        write(path,dict(original,tokens_sha256='changed'))
        with self.assertRaisesRegex(ValueError,'Timing initialization'):report.report(root,offline=True)
        path.unlink()
        with self.assertRaises(FileNotFoundError):report.report(root,offline=True)
        write(path,original)

    def test_complete_single_round_report_and_missing_extra_evidence(self):
        root = preparation.prepare(self.root/'SINGLE-SYNTHETIC-NOT-GPU', self.build, Path('/absent'),
                                   source_only=True, rounds=1)
        info = executable_fixture(root)
        make_evidence(root, info)
        result = report.report(root, offline=True)
        self.assertTrue(result['completed'])
        self.assertEqual(result['round_count'], 1)
        self.assertEqual(result['blocks_per_round'], 9)
        self.assertEqual(result['timing_windows'], 81)
        self.assertEqual(result['timing_rank_records'], 648)
        self.assertEqual(result['preflight_rank_records'], 144)
        self.assertEqual(result['profile_traces'], 72)
        self.assertEqual(result['candidate_shape_mappings'], 12)
        self.assertEqual(len(result['all_samples']), 81)
        self.assertEqual(len(result['evidence']['timing_files']), 648)
        self.assertEqual(set(result['rounds']), {'1', 'combined'})
        self.assertEqual(set(result['sensitivity']), {'1'})
        self.assertEqual(result['rounds']['combined'], result['rounds']['1'])
        self.assertEqual(result['rounds']['1']['blocks'], 9)
        self.assertEqual(len(result['rounds']['1']['comparisons']), 18)
        markdown = (root/'report.md').read_text()
        self.assertIn('本次1轮', markdown)
        self.assertIn('没有第二轮反转验证', markdown)
        self.assertNotIn('| 2 |', markdown)
        self.assertFalse((root/'results/round2').exists())

        extra = root/'results/round2/window-000-unrequested'; extra.mkdir(parents=True)
        try:
            with self.assertRaisesRegex(ValueError, 'extra timing window'): report.report(root, offline=True)
        finally:
            extra.rmdir(); extra.parent.rmdir()
        last = root/'results'/info['schedule'][-1]['directory']
        renamed = last.with_name('hidden-last-window'); last.rename(renamed)
        try:
            with self.assertRaisesRegex(ValueError, 'Missing or extra timing'): report.report(root, offline=True)
        finally: renamed.rename(last)
        rank = last/'rank7.json'; original = rank.read_bytes(); rank.unlink()
        try:
            with self.assertRaises(FileNotFoundError): report.report(root, offline=True)
        finally: rank.write_bytes(original)


def profile_fixture(policy):
    with mock.patch.object(scope_tests,'sp',sp):return scope_tests.profile_fixture(policy)


def make_window(root,info,job):
    # Reuse the existing mock model/trace writer; provide this policy's role plans.
    original_read=scope_tests.read
    def per_policy(path):
        data=original_read(path)
        if Path(path).name=='selection-plan.json':return dict(roles=sp.role_plans(data,job['policy']))
        return data
    with mock.patch.object(scope_tests,'sp',sp),mock.patch.object(scope_tests,'read',side_effect=per_policy):
        scope_tests.make_window(root,info,job)


def make_evidence(root, info):
    """Complete simulated evidence under a temporary directory, never a GPU run."""
    for job in runner.jobs(info, 'all'): make_window(root, info, job)
    plans = read(root/'scripts/selection-plan.json')
    for policy, role, _ in runner.mapping_jobs(root, info):
        coords = sp.role_plans(plans, policy)[role]['entry']['coords']
        with (root/'results'/f'mapping-{policy}-{role}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['rank', 'index', 'm', 'n', 'destination', 'sm_occupancy'])
            writer.writeheader()
            for rank, row in enumerate(coords):
                for i, tile in enumerate(row):
                    writer.writerow(dict(rank=rank, index=i, m=tile//32, n=tile%32,
                                         destination=tile//64, sm_occupancy=2))
    report.preflight(root, offline=True)


def executable_fixture(root):
    """Temporary mock records only; the generated worker remains permanently disabled."""
    info=read(root/'submission.json');info.update(source_only=False,executable=True)
    plans=read(root/'scripts/selection-plan.json');parent=plans['parent'];parent.update(source_only=False,gpu_calibrated=True)
    for record in parent['roles'].values():
        record.update(test_only=False,gpu_calibrated=True,hardware=info['calibration_hardware'],
                      origin=dict(kind='scope_bf16_calibration'),inputs={'/CPU-FIXTURE':'f'*64})
    plans.update(source_only=False,gpu_calibrated=True,parent=parent)
    write(root/'scripts/selection-plan.json',plans)
    seal=read(root/'provenance/preparation.json');seal.update(source_only=False,executable=True,plans_sha256=sha(root/'scripts/selection-plan.json'))
    write(root/'provenance/preparation.json',seal)
    m=read(root/'provenance/build-manifest.json');m.update(source_only=False,diagnostic_only=False)
    m['files']['plans.json']=seal['plans_sha256'];write(root/'provenance/build-manifest.json',m)
    info['build_manifest']['sha256']=sha(root/'provenance/build-manifest.json')
    for order in builder.ORDERS:
        path=root/'provenance'/f'build-{order}-manifest.json';child=read(path)
        child.update(status='built',source_only=False,diagnostic_only=False);write(path,child)
        path=root/'provenance'/f'build-{order}-identity.json';identity=read(path);identity['source_only']=False;write(path,identity)
    for p,v in info['variants'].items():
        v['libraries']={name:dict(path='/CPU-FIXTURE/'+v['order']+'/'+name,sha256=('a' if v['order']=='base' else 'b')*64)
                        for name in ('libflux_cuda.so','libflux_cuda_ths_op.so')}
    info['files']={p:sha(root/p) for p in info['files']};write(root/'submission.json',info)
    write(root/'start-gpu-check.json',dict(passed=True,run_id=info['run_id'],synthetic=True))
    return info


if __name__=='__main__':unittest.main()
