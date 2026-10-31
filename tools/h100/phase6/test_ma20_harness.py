"""MA20 worker identity, physical mask evidence, and guarded paired execution."""
import argparse
from collections import Counter
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import detach_ma20 as launcher
import ma20_experiment as experiment
import ma20_plan as planner
import ma20_worker as worker
import ma2_ablation_build
from common import REPO, read, sha, write


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selection,_,_=ma2_ablation_build.canonical()
        cls.bundle=planner.bundle(selection)

    def test_original_training_loop_and_outside_timing_device_readback(self):
        source=worker.worker_source()
        self.assertTrue(worker.validate_sources(source))
        for old,new in [('start=time.perf_counter()', 'start=time.monotonic()'),
                        ('successful, grad_norm, _ = optimizer.step()',
                         'successful, grad_norm, _ = optimizer.bad_step()')]:
            with self.assertRaises(ValueError):worker.validate_sources(source.replace(old,new))
        line="report['ma20_device_masks'] = device_masks(adapter)\n"
        with self.assertRaisesRegex(ValueError,'after the measured'):
            worker.validate_sources(source.replace(line,'').replace('if MODE==\'profile\':',line+'if MODE==\'profile\':'))

    def test_actual_device_mask_is_read_not_cpu_cached_count(self):
        mask=self.bundle['configurations']['Q28']['roles']['attn']['entry']['mask']
        class Value:
            shape=(8,512);dtype='torch.uint8';is_cuda=True
            def detach(self):return self
            def cpu(self):return self
            def contiguous(self):return self
            def tolist(self):return mask
        record=self.bundle['configurations']['Q28']['roles']['attn']
        slot=types.SimpleNamespace(selected=Value(),selection_sha256=record['mask_sha256'])
        adapter=types.SimpleNamespace(scope_ops={'module':types.SimpleNamespace(codec=types.SimpleNamespace(slots=[slot,slot]))})
        self.assertEqual(worker.device_masks(adapter)['module'][0]['selected_per_source'],[28]*8)
        slot.selected.is_cuda=False
        with self.assertRaisesRegex(ValueError,'actual CUDA mask'):
            worker.device_masks(adapter)
        slot.selected.is_cuda=True
        slot.selection_sha256=self.bundle['configurations']['MA11']['roles']['attn']['mask_sha256']
        with self.assertRaisesRegex(ValueError,'actual device mask'):
            worker.device_masks(adapter)

    def test_report_rejects_old_mask_even_when_nominal_arm_is_candidate(self):
        names=['decoder.layers.0.self_attention.linear_proj','decoder.layers.0.mlp.linear_fc2']
        records=self.bundle['configurations']['Q28']['roles']
        row=dict(target_modules=names,ma20_device_masks={})
        for name,role in zip(names,('attn','mlp')):
            record=records[role]
            row['ma20_device_masks'][name]=[dict(slot=i,mask_sha256=record['mask_sha256'],
                selected_per_source=[28]*8,wire_bytes_per_source=[14680064-15360*28]*8) for i in range(2)]
        experiment.validate_device_masks(row,records)
        row['ma20_device_masks'][names[0]][1]['selected_per_source']=[7]*8
        with self.assertRaisesRegex(ValueError,'device mask/count/bytes'):
            experiment.validate_device_masks(row,records)

    def test_schedule_is_complete_paired_and_counterbalanced(self):
        for trials in (1,2):
            jobs=experiment.schedule('Q14',trials)
            self.assertEqual(len(jobs),16*trials)
            for trial in range(1,trials+1):
                selected=[j for j in jobs if j['trial']==trial]
                self.assertEqual(Counter(j['arm'] for j in selected),{'MA11':8,'Q14':8})
                self.assertEqual(Counter(j['arm'] for j in selected if j['index']%2==0),{'MA11':4,'Q14':4})
            self.assertEqual(len({(j['trial'],j['directory']) for j in jobs}),16*trials)

    def test_prepare_routes_candidate_mask_and_frozen_worker_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);bank=base/'bank';bank.mkdir();root=base/'fullstep'
            write(bank/'ma20-plan.json',self.bundle)
            write(bank/'tuning.json',experiment.full.bench.validate_tuning(None))
            write(bank/experiment.builder.MANIFEST,{})
            decision=dict(completed=True,candidate='Q28',bundle_sha256=sha(bank/'ma20-plan.json'),
                          build_manifest_sha256=sha(bank/experiment.builder.MANIFEST))
            selection=base/'selection.json';write(selection,decision)
            args=argparse.Namespace(root=root,build=bank,selection=selection,candidate='Q28',
                                    megatron=base/'megatron',trials=1)
            libs=lambda bank,arm:dict(root='/synthetic/'+arm,files={})
            with patch.object(experiment.builder,'verify'),patch.object(experiment,'libraries',side_effect=libs), \
                 patch.object(experiment.full,'snapshot_megatron',return_value={}), \
                 patch('ma20_probe.verify_selection',return_value=decision):
                experiment.prepare(args)
            plan=experiment.verify(root,offline=True)
            for arm in ('MA11','Q28'):
                scripts=root/'arms'/arm/'scripts'
                selected=read(scripts/'selection-plan.json')
                self.assertEqual(selected,planner.selection_for(self.bundle,arm))
                inner=read(scripts.parent/'plan.json')
                env=experiment.full.bench.environment(scripts.parent,inner,experiment.NATIVE_ARM,
                                                       Path(sys.executable),None)
                self.assertEqual(env['H100_MA11_ARM'],experiment.NATIVE_ARM)
                self.assertEqual(env['CUDA_DEVICE_MAX_CONNECTIONS'],'1')
                script='''import importlib.util,sys
from pathlib import Path
p=Path(sys.argv[1]);sys.path.insert(0,str(p))
import ma20_worker
s=importlib.util.spec_from_file_location('operator',p/'operator-controller.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert m.ARMS[-1]=='F4-cache-zcount-csoftmax'
'''
                done=subprocess.run([sys.executable,'-I','-c',script,str(scripts)],capture_output=True,text=True)
                self.assertEqual(done.returncode,0,done.stderr)
            self.assertEqual(plan['arms'],['MA11','Q28'])
            (root/'arms/Q28/scripts/selection-plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Frozen worker source changed'):
                experiment.verify(root,offline=True)

    def test_preview_is_read_only_and_does_not_import_probe_or_query_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'never-created'
            with patch.object(launcher,'controller_hashes',side_effect=AssertionError('execute path')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                launcher.main(['--out',str(out)])
            self.assertFalse(out.exists())
            self.assertIn('PLAN ONLY',output.getvalue())
            self.assertIn('--stage profile',output.getvalue())

    def test_reuse_preview_only_launches_fullstep_for_the_manual_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            old=Path(temp)/'old';out=Path(temp)/'new'
            argv=['--out',str(out),'--reuse-batch',str(old),'--candidate','Q112','--reuse-ma11']
            args=launcher.arguments(argv)
            self.assertEqual(args.ma20_build,old/'build')
            self.assertEqual(launcher.commands(args),[])
            commands=launcher.commands(args,'Q112')
            self.assertEqual([name for name,_ in commands],
                             ['fullstep-prepare','fullstep-preflight','fullstep-profile','fullstep-timing'])
            prepare=commands[0][1]
            self.assertIn('--manual-candidate',prepare)
            self.assertEqual(prepare[prepare.index('--reuse-ma11-from')+1],str(old/'fullstep'))
            self.assertEqual(prepare[prepare.index('--selection')+1],str(old/'operator/selection.json'))
            with patch.object(launcher,'controller_hashes',side_effect=AssertionError('execute path')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                launcher.main(argv)
            self.assertIn('Run only Q112',output.getvalue())
            self.assertFalse(out.exists())
            self.assertFalse(old.exists())

    def test_reuse_flags_cannot_silently_skip_normal_selection(self):
        for argv,message in [(['--candidate','Q112'],'--candidate requires'),
                             (['--reuse-ma11'],'--reuse-ma11 requires'),
                             (['--reuse-batch','/tmp/old','--ma11-build','/tmp/bank'],'cannot use'),
                             (['--reuse-batch','/tmp/old','--out','/tmp/old/nested'],'outside')]:
            with self.subTest(argv=argv),self.assertRaisesRegex(ValueError,message):
                launcher.arguments(argv)
        decision=dict(completed=True,candidate='Q0')
        with self.assertRaisesRegex(ValueError,'must nominate'):
            experiment.candidate_selection(decision,'Q112')
        self.assertEqual(experiment.candidate_selection(decision,'Q112',manual=True),
                         dict(mode='manual',operator_selected_candidate='Q0',operator_confirmation_passed=False))

    def test_manual_prepare_keeps_q0_evidence_and_runs_only_q112(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);bank=base/'bank';bank.mkdir();root=base/'fullstep'
            write(bank/'ma20-plan.json',self.bundle)
            write(bank/'tuning.json',experiment.full.bench.validate_tuning(None))
            write(bank/experiment.builder.MANIFEST,{})
            decision=dict(completed=True,candidate='Q0',bundle_sha256=sha(bank/'ma20-plan.json'),
                          build_manifest_sha256=sha(bank/experiment.builder.MANIFEST))
            selection=base/'selection.json';write(selection,decision);original=selection.read_bytes()
            args=argparse.Namespace(root=root,build=bank,selection=selection,candidate='Q112',
                megatron=base/'megatron',trials=1,manual_candidate=True,reuse_ma11_from=base/'old/fullstep')
            with patch.object(experiment.builder,'verify'), \
                 patch.object(experiment,'libraries',side_effect=lambda bank,arm:dict(root='/synthetic/'+arm,files={})), \
                 patch.object(experiment.full,'snapshot_megatron',return_value={}), \
                 patch.object(experiment,'reuse_ma11',return_value=dict(files={})), \
                 patch('ma20_probe.verify_selection',return_value=decision):
                experiment.prepare(args)
            plan=experiment.verify(root,offline=True)
            self.assertEqual(plan['arms'],['Q112'])
            self.assertEqual(plan['candidate_selection'],dict(mode='manual',operator_selected_candidate='Q0',
                                                             operator_confirmation_passed=False))
            self.assertEqual(read(root/'operator-selection.json'),decision)
            self.assertEqual(selection.read_bytes(),original)
            self.assertFalse((root/'arms/MA11').exists())
            actual=read(root/'arms/Q112/scripts/selection-plan.json')
            self.assertEqual(actual,planner.selection_for(self.bundle,'Q112'))
            for stage,count in (('preflight',1),('profile',1),('timing',8)):
                work=list(experiment.jobs(plan,stage))
                self.assertEqual(len(work),count)
                self.assertEqual({j['arm'] for j in work},{'Q112'})
            plan['candidate_selection']['operator_confirmation_passed']=True
            write(root/'plan.json',plan)
            with self.assertRaisesRegex(ValueError,'provenance changed'):
                experiment.verify(root,offline=True)

    def test_historical_comparison_never_bootstraps_or_claims_validation(self):
        values={'MA11':[500.]*8,'Q112':[450.]*8}
        with patch.object(experiment.full.bench,'paired',side_effect=AssertionError('not paired')):
            result=experiment.timing_comparison(values,'Q112',historical=True)
        self.assertAlmostEqual(result['descriptive_percent'],10.)
        self.assertNotIn('paired',result)
        self.assertFalse(result['positive_increment'])

    def test_single_candidate_keeps_operator_selection_and_has_no_baseline(self):
        args=launcher.arguments(['--candidate-only'])
        self.assertIn('operator-calibrate',dict(launcher.commands(args)))
        self.assertIn('--candidate-only',dict(launcher.commands(args,'Q28'))['fullstep-prepare'])
        with self.assertRaisesRegex(ValueError,'does not borrow'):
            launcher.arguments(['--candidate-only','--reuse-ma11','--reuse-batch','/tmp/old','--candidate','Q28'])
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);bank=base/'bank';bank.mkdir();root=base/'fullstep'
            write(bank/'ma20-plan.json',self.bundle)
            write(bank/'tuning.json',experiment.full.bench.validate_tuning(None))
            write(bank/experiment.builder.MANIFEST,{})
            decision=dict(completed=True,candidate='Q28',bundle_sha256=sha(bank/'ma20-plan.json'),
                          build_manifest_sha256=sha(bank/experiment.builder.MANIFEST))
            selection=base/'selection.json';write(selection,decision)
            args=argparse.Namespace(root=root,build=bank,selection=selection,candidate='Q28',
                                    megatron=base/'megatron',trials=2,candidate_only=True)
            with patch.object(experiment.builder,'verify'), \
                 patch.object(experiment,'libraries',side_effect=lambda bank,arm:dict(root='/CPU-FIXTURE/'+arm,files={})), \
                 patch.object(experiment.full,'snapshot_megatron',return_value={}), \
                 patch('ma20_probe.verify_selection',return_value=decision):
                experiment.prepare(args)
            plan=experiment.verify(root,offline=True)
            self.assertEqual(plan['comparison_mode'],'candidate_only')
            self.assertEqual(plan['arms'],['Q28'])
            self.assertFalse((root/'arms/MA11').exists())
            self.assertFalse((root/'reused-ma11.json').exists())
            for stage,count in [('preflight',1),('profile',1),('timing',16)]:
                work=list(experiment.jobs(plan,stage))
                self.assertEqual(len(work),count)
                self.assertEqual({job['arm'] for job in work},{'Q28'})

    def test_single_candidate_report_has_no_speedup_or_borrowed_ma11(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            plan=dict(run_id='CPU-FIXTURE',candidate='Q28',arms=['Q28'],trials=1,
                      schedule=experiment.schedule('Q28',candidate_only=True),
                      comparison_mode='candidate_only',arm_roots={'Q28':str(root/'Q28')})
            write(root/'operator-selection.json',{})
            write(root/'ma20-plan.json',self.bundle)
            rows=[dict(elapsed_seconds=22.5,losses=[1.]*50,grad_norms=[2.]*50) for _ in range(8)]
            with patch.object(experiment,'verify',return_value=plan), \
                 patch.object(experiment,'preflight',return_value=dict(files={},rank_identities=[])), \
                 patch.object(experiment,'profile_summary',return_value=dict(execution_validated=True)), \
                 patch.object(experiment,'window',return_value=rows), \
                 patch.object(experiment,'validate_operator_identity'), \
                 patch.object(experiment.full,'identity',return_value=[]), \
                 patch.object(experiment.full.bench,'paired',side_effect=AssertionError('not paired')):
                result=experiment.report(root)
            self.assertEqual(result['formal_optimizer_steps'],400)
            self.assertEqual(result['timing_rank_records'],64)
            self.assertEqual(result['executed_arms'],['Q28'])
            self.assertNotIn('reused_ma11',result)
            self.assertFalse(result['performance_increment_over_MA11_validated'])
            self.assertEqual(result['paired_trajectory_drift'],[])
            self.assertEqual(result['comparisons']['compare1']['mean_ms'],{'Q28':450.})
            self.assertNotIn('paired',result['comparisons']['compare1'])
            self.assertNotIn('MA11 ms',(root/'report.md').read_text())

    def test_candidate_only_report_counts_only_new_work_and_labels_old_ma11(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);subroot=root/'Q112';subroot.mkdir()
            plan=dict(run_id='new',candidate='Q112',arms=['Q112'],trials=1,
                      schedule=experiment.schedule('Q112',candidate_only=True),
                      comparison_mode='historical_ma11',arm_roots={'Q112':str(subroot)},
                      candidate_selection=dict(mode='manual',operator_selected_candidate='Q0',
                                               operator_confirmation_passed=False),
                      operator_selection_source='/old/operator/selection.json',reused_ma11_sha256='fixture')
            write(root/'operator-selection.json',{})
            write(root/'ma20-plan.json',self.bundle)
            write(root/'reused-ma11.json',dict(source='/old/fullstep',run_id='old',
                  samples=[dict(trial=1,ms_per_step=500.) for _ in range(8)],
                  formal_optimizer_steps=400,timing_rank_records=64))
            rows=[dict(elapsed_seconds=22.5,losses=[1.]*50,grad_norms=[2.]*50) for _ in range(8)]
            with patch.object(experiment,'verify',return_value=plan), \
                 patch.object(experiment,'preflight',return_value=dict(files={},rank_identities=[])), \
                 patch.object(experiment,'profile_summary',return_value=dict(execution_validated=True)), \
                 patch.object(experiment,'window',return_value=rows), \
                 patch.object(experiment,'validate_operator_identity'), \
                 patch.object(experiment.full,'identity',return_value=[]), \
                 patch.object(experiment.full.bench,'paired',side_effect=AssertionError('not paired')):
                result=experiment.report(root)
            self.assertEqual(result['formal_optimizer_steps'],400)
            self.assertEqual(result['timing_rank_records'],64)
            self.assertEqual(result['executed_arms'],['Q112'])
            self.assertEqual(result['reused_ma11']['formal_optimizer_steps'],400)
            self.assertEqual(result['paired_trajectory_drift'],[])
            self.assertFalse(result['performance_increment_over_MA11_validated'])
            self.assertIn('MA11 times are historical',(root/'report.md').read_text())
            self.assertNotIn('Paired speedup',(root/'report.md').read_text())

    def test_normal_two_trial_report_retains_paired_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            plan=dict(run_id='new',candidate='Q112',arms=['MA11','Q112'],trials=2,
                      schedule=experiment.schedule('Q112',trials=2),
                      arm_roots={arm:str(root/arm) for arm in ('MA11','Q112')})
            write(root/'operator-selection.json',{})
            write(root/'ma20-plan.json',self.bundle)
            def rows(root,plan,job):
                return [dict(elapsed_seconds=25. if job['arm']=='MA11' else 22.5,
                             losses=[1.]*50,grad_norms=[2.]*50,ma11_final_parameters_sha256='fixture')
                        for _ in range(8)]
            with patch.object(experiment,'verify',return_value=plan), \
                 patch.object(experiment,'preflight',return_value=dict(files={},rank_identities=[])), \
                 patch.object(experiment,'profile_summary',return_value=dict(execution_validated=True)), \
                 patch.object(experiment,'window',side_effect=rows), \
                 patch.object(experiment,'validate_operator_identity'), \
                 patch.object(experiment.full,'identity',return_value=[]):
                result=experiment.report(root)
            self.assertEqual(result['formal_optimizer_steps'],1600)
            self.assertEqual(len(result['paired_trajectory_drift']),16)
            self.assertTrue(result['performance_increment_over_MA11_validated'])
            self.assertNotIn('reused_ma11',result)
            self.assertTrue(all('paired' in c for c in result['comparisons'].values()))
            self.assertIn('Paired speedup',(root/'report.md').read_text())

    def test_reused_ma11_requires_complete_unchanged_raw_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bank=root/'bank';source=root/'old';subroot=source/'arms/MA11'
            bank.mkdir();(subroot/'scripts').mkdir(parents=True)
            decision=dict(completed=True,candidate='Q0')
            write(bank/'ma20-plan.json',self.bundle);write(source/'ma20-plan.json',self.bundle)
            write(source/'operator-selection.json',decision)
            (subroot/'scripts/worker.py').write_text(worker.worker_source())
            libs=dict(root='/synthetic/MA11',files={})
            write(subroot/'plan.json',dict(selection=planner.selection_for(self.bundle,'MA11'),
                  libraries={experiment.NATIVE_ARM:libs},
                  source_files={'scripts/worker.py':sha(subroot/'scripts/worker.py')}))
            plan=dict(protocol=experiment.PROTOCOL,run_id='old',arms=['MA11','Q0'],candidate='Q0',
                      trials=1,schedule=experiment.schedule('Q0'),blocks=8,timed_steps=50,warmup_steps=10,
                      bundle_sha256=sha(bank/'ma20-plan.json'),selection_sha256=sha(source/'operator-selection.json'),
                      arm_roots={'MA11':str(subroot)},arm_plan_sha256={'MA11':sha(subroot/'plan.json')})
            write(source/'plan.json',plan)
            result=dict(protocol=experiment.PROTOCOL,run_id='old',completed=True,formal_optimizer_steps=800,
                        timing_rank_records=128,all_samples=[],files={})
            for job in plan['schedule']:
                if job['arm']!='MA11':continue
                result['all_samples'].append(dict(job,ms_per_step=500.))
                for rank in range(8):
                    path=subroot/'results'/job['directory']/f'rank{rank}.json'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    write(path,dict(elapsed_seconds=25.))
                    result['files'][str(path)]=sha(path)
            write(source/'report.json',result)
            def rows(root,plan,job):
                return [read(subroot/'results'/job['directory']/f'rank{rank}.json') for rank in range(8)]
            with patch.object(experiment,'libraries',return_value=libs), \
                 patch.object(experiment,'window',side_effect=rows), \
                 patch.object(experiment,'validate_operator_identity'), \
                 patch.object(experiment.full,'identity',return_value=[]):
                snapshot=experiment.reuse_ma11(source,bank,self.bundle,decision,1)
                self.assertEqual(len(snapshot['samples']),8)
                self.assertEqual(snapshot['timing_rank_records'],64)
                path=Path(next(iter(result['files'])))
                raw=path.read_bytes()
                write(path,dict(elapsed_seconds=26.))
                with self.assertRaisesRegex(ValueError,'timing differs'):
                    experiment.reuse_ma11(source,bank,self.bundle,decision,1)
                path.write_bytes(raw)
                path.write_bytes(raw+b'\n')
                with self.assertRaisesRegex(ValueError,'rank record changed'):
                    experiment.reuse_ma11(source,bank,self.bundle,decision,1)
                path.write_bytes(raw)
                result['completed']=False;write(source/'report.json',result)
                with self.assertRaisesRegex(ValueError,'report is incomplete'):
                    experiment.reuse_ma11(source,bank,self.bundle,decision,1)

    def test_no_candidate_stops_without_fullstep_or_success_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'handoff').mkdir();(out/'operator').mkdir()
            write(out/'handoff/controllers.json',{})
            write(out/'operator/selection.json',dict(completed=True,candidate=None))
            args=launcher.arguments(['--out',str(out),'--ma20-build',str(out/'bank')])
            calls=[]
            with patch.object(launcher.launcher,'invoke',side_effect=lambda cmd,*a:calls.append(cmd)), \
                 patch('ma20_probe.verify_selection',return_value=dict(completed=True,candidate=None)), \
                 contextlib.redirect_stdout(io.StringIO()):
                state=launcher.supervise(args)
            self.assertEqual(state['stage'],'no_candidate')
            self.assertFalse(state['fullstep_executed'])
            self.assertFalse(state['increment_over_MA11_validated'])
            self.assertFalse(any('ma20_experiment.py' in ' '.join(c) for c in calls))

    def test_changed_numeric_trajectories_are_reported_as_drift(self):
        value=experiment.trajectory_drift([1.,2.],[1.1,1.9])
        self.assertAlmostEqual(value['max_abs'],.1)
        self.assertGreater(value['relative_l2'],0)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            experiment.trajectory_drift([1.,2.],[1.])

    def test_operator_and_fullstep_hardware_software_cannot_drift(self):
        hardware=dict(host='fixture',devices=[dict(uuid=f'gpu-{i}',name='H100',sm_count=132,
                                                  capability=[9,0]) for i in range(8)],local_rank=0)
        row=dict(hardware=hardware,torch='2.7',cuda='12.4',nccl=[2,23,4],
                 python='/same/interpreter',ma20_python_version='3.12 fixture build')
        selection=dict(hardware=dict(host='fixture',devices=[{k:d[k] for k in ('uuid','name','sm_count')}
                       for d in hardware['devices']]),
                       software=dict(torch='2.7',cuda='12.4',nccl=[2,23,4],python='3.12 fixture build'))
        experiment.validate_operator_identity(row,selection)
        for key,value in (('torch','different'),('cuda','different'),('nccl',[2,99]),
                          ('ma20_python_version','different')):
            changed=copy.deepcopy(row);changed[key]=value
            with self.assertRaisesRegex(ValueError,'Torch/CUDA/NCCL/Python'):
                experiment.validate_operator_identity(changed,selection)
        for change in ('host','rank-order'):
            changed=copy.deepcopy(row)
            if change=='host':changed['hardware']['host']='different'
            else:changed['hardware']['devices'].reverse()
            with self.assertRaisesRegex(ValueError,'ordered GPU identity'):
                experiment.validate_operator_identity(changed,selection)

    def test_producer_barrier_and_complete_path_keep_original_trace_checks(self):
        from test_ma4_f_harness import profile_fixture
        events=profile_fixture('F4')
        for event in events:
            if event['cat']=='kernel' and 'flux_bf16' in event['name']:
                event['args'].update(grid=[512,1,1],block=[128,1,1])
            elif event['cat']=='kernel' and 'decode_consumer' in event['name']:
                event['args'].update(grid=[2048,1,1],block=[128,1,1])
        tuning=experiment.full.bench.validate_tuning(None)
        original=experiment.full._trace_forward(events,experiment.NATIVE_ARM,tuning)
        measured=experiment.trace_forward_records(events,experiment.NATIVE_ARM,tuning)
        extra={'producer_us','barrier_us','producer_start_to_layernorm_end_us','path_distribution_us'}
        for role in ('attn','mlp'):
            self.assertEqual({k:v for k,v in measured[role].items() if k not in extra},original[role])
            self.assertEqual(measured[role]['producer_us'],2)
            self.assertEqual(measured[role]['barrier_us'],1)
            self.assertEqual(measured[role]['producer_start_to_layernorm_end_us'],10)
            self.assertEqual(measured[role]['path_distribution_us']['producer_us'],
                             dict(median=2,p95=2,maximum=2,sum=256))
        broken=copy.deepcopy(events)
        next(e for e in broken if e['cat']=='kernel' and 'CudaIpcBarrier' in e['name'])['args']['stream']=99
        with self.assertRaises(ValueError):
            experiment.trace_forward_records(broken,experiment.NATIVE_ARM,tuning)
        broken=[e for e in events if not (e['cat']=='kernel' and 'CudaIpcBarrier' in e['name'])]
        with self.assertRaises(ValueError):
            experiment.trace_forward_records(broken,experiment.NATIVE_ARM,tuning)

    def test_original_full_window_validator_replays_historical_rank_schema(self):
        # Optional local regression against real MA11 records. The copied records
        # are a CPU validation fixture, never new MA20 performance evidence.
        candidates=sorted((REPO.parent/'TempName/logs/h100/phase6').glob('ma11*/fullstep/plan.json'))
        if not candidates:self.skipTest('Historical sibling MA11 rank records unavailable')
        original=candidates[0].parent
        inner=read(original/'plan.json')
        job=next(dict(j,mode='timing') for j in inner['schedule'] if j['arm']==experiment.NATIVE_ARM)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);subroot=root/'arms/MA11';subroot.mkdir(parents=True)
            write(subroot/'plan.json',inner)
            dest=subroot/'results'/job['directory'];dest.mkdir(parents=True)
            records=inner['selection']['variants'][experiment.POLICY]['roles']
            for rank in range(8):
                row=read(original/'results'/job['directory']/f'rank{rank}.json')
                row.update(ma11_plan_sha256=sha(subroot/'plan.json'),ma20_arm='MA11',
                           ma20_bundle_sha256='fixture',ma20_device_masks={})
                for name in row['target_modules']:
                    role=experiment.full.routing.module_role(name);record=records[role]
                    counts=list(map(sum,record['entry']['mask']))
                    row['ma20_device_masks'][name]=[dict(slot=i,mask_sha256=record['mask_sha256'],
                        selected_per_source=counts,wire_bytes_per_source=[14680064-15360*q for q in counts]) for i in range(2)]
                write(dest/f'rank{rank}.json',row)
            plan=dict(arm_roots={'MA11':str(subroot)},bundle_sha256='fixture')
            actual=experiment.window(root,plan,dict(job,arm='MA11'))
            self.assertEqual(len(actual),8)
            row=read(dest/'rank3.json');row['losses']=row['losses'][:20];write(dest/'rank3.json',row)
            with self.assertRaisesRegex(ValueError,'trajectory'):
                experiment.window(root,plan,dict(job,arm='MA11'))


if __name__=='__main__':unittest.main()
