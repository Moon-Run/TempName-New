"""CPU-only budget, shared-runtime, strict-report and detached-launch checks."""
import ast
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import ablation5 as cli
import ablation5_plan as planner
import ablation5_prepare as preparation
import ablation5_protocol as sp
import ablation5_report as report
import detach_ablation5_run_two as detached
import scope_build
import scope_plan
import scope_sources
import test_scope as fixtures
from ablation_plan import mask_sha
from common import read, write, sha


def parent_fixture():
    return dict(protocol=scope_build.PROTOCOL, roles={r: scope_plan.cpu_fixture(r) for r in sp.SHAPES},
                source_only=True, gpu_calibrated=False)


class BudgetTests(unittest.TestCase):
    def test_schedule_and_fixed_candidate_definition(self):
        jobs = sp.schedule(); self.assertEqual(len(jobs), 50)
        rows = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b] for r in (1, 2) for b in range(5)]
        self.assertEqual(rows[5:], [row[::-1] for row in rows[:5][::-1]])
        for i in range(5): self.assertEqual({row[i] for row in rows[:5]}, set(sp.POLICIES))
        for p, limit in zip(sp.POLICIES, (7, 14, 28, 56, 112)):
            d = sp.definition(p)
            self.assertEqual(d['max_tiles_per_source'], limit)
            self.assertEqual((d['order'], d['workspace'], d['quantization']), ('arrival', 'double', 'targeted'))
        for blocks in (0, 4, True):
            with self.assertRaises(ValueError): sp.schedule(blocks)

    def test_nested_caps_positive_only_ties_local_exclusion_and_tampering(self):
        parent = parent_fixture(); scores = {}
        for role, r in parent['roles'].items():
            values = [[1. if t//64 != src else 100. for t in range(512)] for src in range(8)]
            r['entry']['mask'] = planner.fitter.selection.select(values, 8, 1/64)
            r['entry']['selected_per_source'] = [7]*8
            r['entry_sha256'] = sp.digest(r['entry']); r['mask_sha256'] = mask_sha(r['entry']['mask'])
            scores[role] = dict(kind='cpu_fixture', scores=values, scores_sha256=sp.digest(values), inputs={},
                                training_passes=[0, 1], heldout_passes=[2])
        plans = dict(protocol=sp.PROTOCOL, source_only=True, gpu_calibrated=False, parent=parent, scores=scores,
                     variants={p: planner.derive(parent, scores, p) for p in sp.POLICIES})
        planner.validate(plans)
        previous = None
        for p, limit in zip(sp.POLICIES, (7, 14, 28, 56, 112)):
            r = sp.role_plans(plans, p)['attn']; e = r['entry']
            self.assertEqual(e['selected_per_source'], [limit]*8)
            self.assertEqual(e['maps'], parent['roles']['attn']['entry']['maps'])
            self.assertEqual(r['wire_bytes_per_source'], [14680064-limit*128*120]*8)
            for src, row in enumerate(e['mask']):
                self.assertFalse(any(row[src*64:(src+1)*64]))
                self.assertEqual([t for t, selected in enumerate(row) if selected], [t for t in range(512) if t//64 != src][:limit])
                if previous is not None: self.assertTrue(all(not old or new for old, new in zip(previous[src], row)))
            previous = e['mask']
        changed = copy.deepcopy(plans); r = changed['variants'][sp.POLICIES[-1]]['roles']['attn']
        r['entry']['mask'][0][64] = 0; r['entry']['mask'][0][511] = 1
        r['entry_sha256'] = sp.digest(r['entry']); r['mask_sha256'] = mask_sha(r['entry']['mask'])
        with self.assertRaisesRegex(ValueError, 'mask/order/hash'): planner.validate(changed)
        sparse = planner.bundle(parent_fixture(), source_only=True)
        self.assertEqual(sp.role_plans(sparse, sp.POLICIES[0])['attn']['entry']['mask'],
                         sp.role_plans(sparse, sp.POLICIES[-1])['attn']['entry']['mask'])

    def test_heldout_observations_do_not_change_scores(self):
        reports = []
        for rank in range(8):
            samples = []
            for repeat in range(192):
                tile = rank*64+repeat%64
                receiver = [dict(tile=tile, source=src, fragment=f,
                                 observed_ns=1000.+100*(src == (rank+1)%8)) for src in range(8) for f in range(8)]
                samples.append(dict(mode='receiver_sparse', offset=repeat%64, repeat=repeat, receiver=receiver,
                                    max_poll_cycle_ns=1., local_markers=[100., 200.]))
            reports.append(dict(rank=rank, completed=True, shapes=[dict(M=2048, N=4096, K_global=4096,
                                    partition_tiles=64, samples=samples)]))
        scores = planner.training_scores(reports, 'attn')
        for r in reports:
            for s in r['shapes'][0]['samples'][128:]:
                for value in s['receiver']: value['observed_ns'] += 100000.*value['source']
        self.assertEqual(scores, planner.training_scores(reports, 'attn'))

    def test_generated_adapter_uses_actual_budget_mask_and_double_buffers(self):
        class Codec:
            owns_configuration = True
            def __init__(self, group, m, n, k, **kwargs):
                self.kwargs = kwargs
                self.slots = [types.SimpleNamespace(m=m,n=n,k_local=k,rank=0,
                    selection_sha256=mask_sha(kwargs['selected']), peers=[types.SimpleNamespace(data_ptr=lambda:123)],
                    selected=types.SimpleNamespace(data_ptr=lambda:456)) for _ in range(2)]
        torch = types.ModuleType('torch'); torch.no_grad = lambda: lambda f:f
        dist = types.ModuleType('torch.distributed'); dist.get_world_size = lambda group:8; torch.distributed = dist
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(sys.modules, {
            'torch':torch,'torch.distributed':dist,'routing':sp,'reference':types.SimpleNamespace(roundtrip=None),
            'flux':types.ModuleType('flux'),'flux.gemm_rs_taco':types.SimpleNamespace(GemmRSTacoDoubleBuffered=Codec)}):
            root = Path(temp); plans = planner.bundle(parent_fixture(), source_only=True)
            # Distinct dense masks detect accidentally routing all five budgets
            # to the original 1/64 mask; also exercise the 112-tile constructor.
            for role, record in plans['parent']['roles'].items():
                values = [[float(t//64 != src) for t in range(512)] for src in range(8)]
                record['entry']['mask'] = planner.fitter.selection.select(values,8,1/64)
                record['entry']['selected_per_source'] = [7]*8
                record['entry_sha256'] = sp.digest(record['entry']); record['mask_sha256'] = mask_sha(record['entry']['mask'])
                plans['scores'][role].update(scores=values,scores_sha256=sp.digest(values))
            plans['variants'] = {p:planner.derive(plans['parent'],plans['scores'],p) for p in sp.POLICIES}
            planner.validate(plans); write(root/'selection-plan.json', plans)
            ns = {'__file__':str(root/'taco_support.py')}; exec(scope_sources.taco_source(), ns)
            for p in sp.POLICIES:
                for role,(m,n,k) in sp.SHAPES.items():
                    op = ns['TacoForward'](None,m,n,k//8,p,role)
                    self.assertEqual(op.codec.kwargs, dict(placement='fused', selected=sp.role_plans(plans,p)[role]['entry']['mask'],
                                                         quantization_mode='targeted'))
                    self.assertTrue(op.owns_output); self.assertEqual(len(op.workspace_info),2)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='ablation5-CPU-FIXTURE-'); cls.root = Path(cls.tmp.name)
        for role in sp.SHAPES: write(cls.root/f'{role}.json', scope_plan.cpu_fixture(role))
        cls.build = scope_build.prepare(cls.root/'build', cls.root/'attn.json', cls.root/'mlp.json', source_only=True)
        cls.plan = preparation.prepare_plan(cls.root/'budget-plan.json', cls.build, source_only=True)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, cls.plan, Path('/absent'), source_only=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_dry_run_shared_library_and_nonexecutable_rehearsal(self):
        with mock.patch.object(subprocess,'run',side_effect=AssertionError('GPU/process')), \
                mock.patch.object(subprocess,'check_output',side_effect=AssertionError('GPU query')), contextlib.redirect_stdout(io.StringIO()):
            jobs = cli.run(self.compare)
            info = report.verify_inputs(self.compare)
            self.assertEqual(len(jobs),60); self.assertEqual(len(list(cli.mapping_jobs(self.compare,info))),10)
            self.assertEqual(len({v['root'] for v in info['variants'].values()}),1)
            with self.assertRaises(ValueError): cli.run(self.compare,execute=True)
        result = subprocess.run([sys.executable,str(self.compare/'scripts/worker.py')],capture_output=True,text=True,
                                env=dict(os.environ,H100_ABLATION5_EXECUTE='1'))
        self.assertIn('Source-only',result.stderr)
        for p in (self.compare/'scripts').glob('*.py'): ast.parse(p.read_text())

    def test_budget_library_and_worker_tampering_are_rejected(self):
        path = self.compare/'submission.json'; original = read(path)
        try:
            altered = copy.deepcopy(original); altered['variants'][sp.POLICIES[-1]]['root'] = '/wrong-library'
            write(path,altered)
            with self.assertRaises(ValueError): report.verify_inputs(self.compare)
        finally: write(path,original)
        m = read(self.build/'manifest.json'); parent = read(self.build/'plans.json')
        m['variants']['scope-candidate']['source_sha256']['src/gemm_rs/taco_codec.cuh'] = 'f'*64
        with self.assertRaisesRegex(ValueError,'sources changed'): preparation.candidate_snapshot(m,parent,True)

    def test_complete_synthetic_report_and_missing_or_invalid_evidence(self):
        root = preparation.prepare(self.root/'SYNTHETIC-NO-GPU',self.build,self.plan,Path('/absent'),source_only=True)
        info = read(root/'submission.json'); info.update(executable=True,source_only=False,
            calibration_hardware=scope_plan.frozen_mlp()['hardware'])
        for v in info['variants'].values():
            v['libraries'] = {name:dict(path='/CPU-FIXTURE/'+name,sha256='a'*64)
                              for name in ('libflux_cuda.so','libflux_cuda_ths_op.so')}
        write(root/'submission.json',info)
        write(root/'start-gpu-check.json',dict(passed=True,run_id=info['run_id'],synthetic=True))
        plans = read(root/'scripts/selection-plan.json')
        original_profile = fixtures.profile_fixture
        def profile(policy):
            events = original_profile(policy)
            for event in events:
                if event.get('name') == 'taco_decode_ring_kernel': event['name'] = 'taco_decode_ring_kernel<false>(int)'
            return events
        for job in cli.jobs(info,'all'):
            original_read = fixtures.read
            def per_policy(path):
                data = original_read(path)
                return dict(roles=sp.role_plans(data,job['policy'])) if Path(path).name=='selection-plan.json' else data
            with mock.patch.object(fixtures,'sp',sp), mock.patch.object(fixtures,'read',side_effect=per_policy), \
                    mock.patch.object(fixtures,'profile_fixture',side_effect=profile):
                fixtures.make_window(root,info,job)
            folder = root/'results'/job['directory']
            write(folder/'command.json',dict(job,scope=sp.definition(job['policy']),communication=info['communication']))
            for rank in range(8):
                path = folder/f'rank{rank}.json'; row = read(path); row['communication'] = info['communication']; write(path,row)
        for p, role, _ in cli.mapping_jobs(root,info):
            with (root/'results'/f'mapping-{p}-{role}.csv').open('w') as f:
                writer = csv.DictWriter(f,fieldnames=['rank','index','m','n','destination','sm_occupancy']); writer.writeheader()
                for rank,coords in enumerate(sp.role_plans(plans,p)[role]['entry']['coords']):
                    for i,tile in enumerate(coords): writer.writerow(dict(rank=rank,index=i,m=tile//32,n=tile%32,destination=tile//64,sm_occupancy=2))
        # Only substitute admission for the synthetic, permanently disabled worker.
        # Real window/profile/mapping/numerical/pairing/hash/report code runs below.
        with mock.patch.object(report,'verify_inputs',return_value=info):
            report.preflight(root,offline=True); result = report.report(root,offline=True)
            self.assertEqual(result['timing_windows'],50); self.assertEqual(result['timing_rank_records'],400)
            self.assertEqual(result['candidate_shape_mappings'],10)
            self.assertEqual(result['execution_timing'],dict(warmup_steps=10,timed_steps=60))
            self.assertEqual(result['formal_optimizer_steps'],3000)
            self.assertEqual(result['rounds']['1']['mean_ms'][sp.POLICIES[0]],1200.)
            self.assertEqual(len(result['rounds']['combined']['comparisons']),10)
            self.assertEqual(result['rounds']['combined']['mean_loss'][sp.POLICIES[0]],1.)
            self.assertEqual(result['rounds']['combined']['mean_final_loss'][sp.POLICIES[0]],1.)
            job = info['schedule'][0]; path = root/'results'/job['directory']/'rank7.json'; saved = path.read_bytes()
            self.assertEqual(len(read(path)['losses']),60)
            changed = read(path); changed['losses'] = changed['losses'][:20]; write(path,changed)
            with self.assertRaisesRegex(ValueError,'trajectory'): report.window(root,path.parent,job['policy'],'timing',job['block'],job['index'])
            path.write_bytes(saved)
            changed = read(path); changed['losses'][-1] = 9.; write(path,changed)
            with self.assertRaisesRegex(ValueError,'TP loss'): report.window(root,path.parent,job['policy'],'timing',job['block'],job['index'])
            path.unlink()
            with self.assertRaises(FileNotFoundError): report.report(root,offline=True)
            path.write_bytes(saved)
            extra = root/'results/round3/window-000-extra'; extra.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'extra'): report.report(root,offline=True)


class LauncherTests(unittest.TestCase):
    def test_real_double_fork_redirects_descriptors_and_reports_pid1_before_work(self):
        with tempfile.TemporaryDirectory(prefix='ablation5-detach-CPU-') as tmp:
            args = detached.arguments(['--out',tmp]); root = Path(tmp); (root/'handoff').mkdir()
            write(root/'batch-state.json',dict(status='launching'))
            def probe(_):
                write(root/'probe.json',dict(pid=os.getpid(),ppid=os.getppid(),
                    descriptors=[os.readlink(f'/proc/self/fd/{fd}') for fd in (0,1,2)]))
            with mock.patch.object(detached,'supervise',side_effect=probe):
                identity = detached.detach(args)
                deadline = time.monotonic()+5
                while not (root/'probe.json').exists() and time.monotonic()<deadline: time.sleep(.02)
                self.assertTrue((root/'probe.json').exists())
                probe_result = read(root/'probe.json')
            self.assertEqual(identity['pid'],probe_result['pid'])
            self.assertEqual(identity['ppid'],1); self.assertEqual(probe_result['ppid'],1)
            self.assertEqual(probe_result['descriptors'][0],'/dev/null')
            self.assertEqual(probe_result['descriptors'][1:],[str(root/'handoff/container-supervisor.log')]*2)

    def test_plan_only_and_exact_two_run_commands(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(os,'fork',side_effect=AssertionError('forked')), \
                mock.patch.object(subprocess,'run',side_effect=AssertionError('launched')), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)/'new'; detached.main(['--out',str(root)])
            self.assertFalse(root.exists())
            cmds = detached.commands(detached.arguments(['--out',str(root)]))
            self.assertEqual(len(cmds),9)
            self.assertEqual([name for name,_ in cmds][-4:],['compare1-controller','compare1-report','compare2-controller','compare2-report'])
            for name,cmd in cmds:
                if name.endswith('prepare'):
                    for flag,value in (('--blocks','5'),('--rounds','2'),('--nvls-enable','0')):
                        self.assertEqual(cmd[cmd.index(flag)+1],value)

    def test_failure_stops_second_run_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = detached.arguments(['--out',tmp]); calls=[]
            info = dict(run_id='one')
            def invoke(cmd,log,env):
                calls.append(log.name)
                if log.name=='compare1-controller.log': raise subprocess.CalledProcessError(1,cmd)
            with mock.patch.object(detached,'invoke',side_effect=invoke), \
                    mock.patch.object(detached,'check_inputs',side_effect=[info,dict(run_id='two')]), \
                    mock.patch('ablation4_repeat.same_inputs'), self.assertRaises(subprocess.CalledProcessError):
                detached.supervise(args)
            self.assertNotIn('compare2-controller.log',calls)
            self.assertEqual(read(Path(tmp)/'batch-state.json')['status'],'failed')


if __name__ == '__main__': unittest.main()
