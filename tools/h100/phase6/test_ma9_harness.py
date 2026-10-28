"""MA9 controller contracts; all synthetic timing values are validation fixtures."""
import argparse
import ast
from collections import Counter
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import ma9_bench as bench
import ma9_fullstep as full
from test_ma2_cr2 import fixture_plan


class ScheduleAndFrozenSources(unittest.TestCase):
    def test_candidate_startup_with_lazy_empty_scope_ops_and_untouched_control(self):
        import ctypes
        worker=ast.parse(full.worker_source())
        branch=next(n for n in worker.body if isinstance(n,ast.If) and
                    ast.unparse(n.test)=="os.environ['H100_MA9_ARM'] == 'F4-zcount'")
        statements=compile(ast.Module(body=[branch],type_ignores=[]),'startup','exec')
        calls=[]; library=object(); optimizer=object(); tracker=object()
        module=types.SimpleNamespace(install_zero_count=lambda *args:(calls.append(args) or tracker))
        namespace=dict(os=os,Path=Path,POLICY='selective',MODE='smoke',optimizer=optimizer,
                       adapter=types.SimpleNamespace(scope_ops={}),ma9_zero_tracker=None)
        with patch.dict(sys.modules,ma9_adapter=module), \
             patch.dict(os.environ,H100_MA9_ARM='F4-zcount',E2E_BUILD='/tmp/ma9-build'), \
             patch.object(ctypes,'CDLL',return_value=library) as load:
            exec(statements,namespace)
            self.assertEqual(calls,[(optimizer,library,'smoke')])
            self.assertIs(namespace['ma9_zero_tracker'],tracker)
            load.assert_called_once_with('/tmp/ma9-build/selective/libflux_cuda.so')
            calls.clear();load.reset_mock();os.environ['H100_MA9_ARM']='F4'
            namespace['ma9_zero_tracker']=None
            exec(statements,namespace)
            self.assertEqual(calls,[])
            load.assert_not_called()
            self.assertIsNone(namespace['ma9_zero_tracker'])

    def test_baseline_guard_rejects_model_step_and_timing_changes(self):
        adapter,taco,worker=full.adapter_source(),full.taco_source(),full.worker_source()
        full.validate_baseline_sources(adapter,taco,worker)
        with self.assertRaisesRegex(ValueError,'adapter'):
            full.validate_baseline_sources(adapter+'\n',taco,worker)
        with self.assertRaisesRegex(ValueError,'training-step'):
            full.validate_baseline_sources(adapter,taco,worker.replace('successful, grad_norm, _ = optimizer.step()',
                'successful, grad_norm, _ = optimizer.changed_step()'))
        with self.assertRaisesRegex(ValueError,'timing/profile'):
            full.validate_baseline_sources(adapter,taco,worker.replace('start=time.perf_counter()',
                'start=time.monotonic()'))

    def test_balanced_paired_blocks_and_independent_trials(self):
        for trials in (1, 2):
            jobs = full.expected_schedule(trials, 8)
            self.assertEqual(len(jobs), trials * 16)
            for trial in range(1, trials + 1):
                trial_jobs = [j for j in jobs if j['trial'] == trial]
                self.assertEqual(Counter(j['arm'] for j in trial_jobs), {a:8 for a in bench.ARMS})
                for half in (range(4), range(4, 8)):
                    orders = Counter(tuple(j['arm'] for j in trial_jobs if j['block'] == b) for b in half)
                    self.assertEqual(orders, {bench.ARMS:2, bench.ARMS[::-1]:2})
        for invalid in (0, 4, 9, True):
            with self.assertRaises(ValueError):
                bench.schedule(invalid)

    def test_generated_sources_parse_and_keep_parameter_audit_outside_timing(self):
        for source in (full.worker_source(), full.adapter_source(), full.taco_source()):
            ast.parse(source)
        self.assertIn("audit_parameters = MODE == 'smoke'", full.worker_source())
        self.assertIn("report['ma9_zero_count_state']", full.worker_source())
        self.assertNotIn('H100_MA9_ARM', full.adapter_source())
        self.assertIn('H100_MA4_F_ARM', full.adapter_source())
        full.validate_baseline_sources(full.adapter_source(), full.taco_source(), full.worker_source())

    def test_frozen_operator_import_is_self_contained(self):
        with tempfile.TemporaryDirectory() as temp:
            scripts = Path(temp)/'scripts'
            scripts.mkdir()
            bench.freeze_helpers(scripts)
            shutil.copy2(bench.__file__, scripts/'bench.py')
            code = ('import sys;sys.path.insert(0,sys.argv[1]);import bench;'
                    'assert bench.ARMS==("F4","F4-zcount");assert len(bench.schedule())==16')
            result = subprocess.run([sys.executable, '-I', '-c', code, str(scripts)],
                                    cwd=scripts, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_freezes_adapters_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root, build = Path(temp)/'fullstep', Path(temp)/'build'
            build.mkdir()
            bench.write(build/'selection-plan.json', fixture_plan()['selection'])
            args = argparse.Namespace(root=root, build=build, megatron=Path(temp)/'megatron', trials=1, blocks=8)
            libs = {a:dict(root='/synthetic/'+a, files={}) for a in bench.ARMS}
            with patch.object(bench, 'build_identity', return_value={'tuning':bench.validate_tuning(None)}), \
                 patch.object(bench, 'libraries', return_value=libs), \
                 patch.object(full, 'snapshot_megatron', return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            plan = full.verify(root, offline=True)
            self.assertEqual(len(plan['schedule']), 16)
            self.assertEqual(plan['arms'], ['F4', 'F4-zcount'])
            (root/'scripts/ma9_adapter.py').write_text('# changed adapter\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                full.verify(root, offline=True)

class FullModelEvidenceTests(unittest.TestCase):
    def setUp(self):
        from test_ma2_cr2 import fixture_rows
        from test_ma4_f_harness import fingerprints
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.plan=fixture_plan();self.plan.pop('target_percent')
        tuning=bench.validate_tuning(None)
        self.plan.update(protocol=full.PROTOCOL,trials=1,blocks=8,zero_count_policy='fused_optimizer_zero_count',
            trial_ids={'1':'trial1'},arms=list(bench.ARMS),schedule=full.expected_schedule(1,8),
            target_ms=None,acceptance_rule=full.ACCEPTANCE_RULE,tuning=tuning,build_identity={'tuning':tuning},
            libraries={a:{'files':{f'/synthetic/{a}/{n}':a+n for n in ('libflux_cuda.so','libflux_cuda_ths_op.so')}} for a in bench.ARMS})
        bench.write(self.root/'plan.json',self.plan)
        scripts=self.root/'scripts';scripts.mkdir()
        for name,source in [('adapter',full.adapter_source()),('taco_support',full.taco_source()),('worker',full.worker_source())]:
            (scripts/(name+'.py')).write_text(source)
        self.digest=bench.sha(self.root/'plan.json')
        self.cr2_rows,self.fingerprints=fixture_rows,fingerprints

    def rows(self,job,means=None):
        arm=job['arm'];rows=self.cr2_rows(self.plan,dict(job,arm='CR2'),self.digest)
        for row in rows:
            for key in list(row):
                if key.startswith('cr2_'):row['ma9_'+key[4:]]=row.pop(key)
            row.update(protocol=full.PROTOCOL,ma9_protocol=bench.SOURCE_PROTOCOL,ma9_arm=arm,
                ma9_native_tuning={name:bench.expected_tuning(self.plan,arm) for name in row['ma9_codec_state']},
                ma9_final_parameters_sha256='c'*64,ma9_consumer_state={},ma9_zero_count_state={})
            universe={name+suffix:dict(shape=[4096],dtype='torch.bfloat16')
                      for name in row['target_modules'] for suffix in ('.weight','.bias')}
            universe.update({f'other.parameter{i}':dict(shape=[1],dtype='torch.bfloat16') for i in range(389-len(universe))})
            row['ma9_trainable_parameters']=universe
            row['ma9_parameter_trajectory']=self.fingerprints(universe) if job['mode']=='smoke' else []
            for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                row[name]=dict(path=f'/synthetic/{arm}/{name}',sha256=arm+name)
            for i,(name,states) in enumerate(row['ma9_submission_state'].items()):
                calls=sum(s['successful_calls'] for s in states)
                row['ma9_consumer_state'][name]=dict(projection_calls=calls,bda_calls=calls,bias_parameter_id=100+i,
                    native=[dict(s,bias_residual_calls=s['successful_calls']) for s in states])
            row['ma9_zero_count_state']=zero_state_fixture(arm,job['mode'],row['rank'])
            row['peak_allocated_gib']=40.0
            row['peak_reserved_gib']=42.0
            row['elapsed_seconds']=(means or {'F4':501.,'F4-zcount':499.})[arm]*len(row['losses'])/1000
        return rows

    def test_fullstep_window_checks_pending_work_and_every_module(self):
        job=dict(self.plan['schedule'][0],arm='F4-zcount',mode='timing')
        rows=self.rows(job)
        with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
            self.assertEqual(len(full.window(self.root,self.plan,job)),8)
            rows[0]['ma9_zero_count_state']['calls']=1
            with self.assertRaisesRegex(ValueError,'execution or buffer reuse'):
                full.window(self.root,self.plan,job)

    def test_all_parameter_gradient_and_update_fingerprints_are_compared(self):
        rows={a:self.rows(dict(arm=a,mode='smoke',trial=0,block=-1,index=-1,directory='smoke-'+a)) for a in bench.ARMS}
        with patch.object(full,'verify',return_value=self.plan), \
             patch.object(full,'window',side_effect=lambda root,plan,job:rows[job['arm']]), \
             patch.object(full,'_preflight',return_value={'ok':True}):
            self.assertEqual(full.preflight(self.root),{'ok':True})
            next(iter(rows['F4-zcount'][7]['ma9_parameter_trajectory'][1]['gradients'].values()))['sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'gradients or updates'):
                full.preflight(self.root)

    def test_negative_performance_is_retained_and_final_parameter_drift_rejected(self):
        means={'F4':501.,'F4-zcount':510.}
        ref=self.rows(dict(self.plan['schedule'][0],mode='timing'),means)
        def rows(root,plan,job):return self.rows(job,means)
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=rows), \
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))):
            result=full.report(self.root,save=False)
            self.assertEqual(result['formal_optimizer_steps'],800)
            self.assertEqual(len(result['all_samples']),16)
            self.assertEqual(result['increment_over_F4_validated'],{'F4-zcount':False})
            self.assertTrue(result['target_evaluated'])
            self.assertFalse(result['target_validated'])
            self.assertEqual(result['target_percent'],2.0)
        def wrong(root,plan,job):
            result=self.rows(job,means)
            if job['arm']=='F4-zcount':result[7]['ma9_final_parameters_sha256']='f'*64
            return result
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=wrong), \
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))):
            with self.assertRaisesRegex(ValueError,'Final paired model parameters differ'):
                full.report(self.root,save=False)


def zero_state_fixture(arm,mode,rank):
    if arm=='F4': return {'installed':False}
    calls={'timing':60,'smoke':5,'profile':11}[mode]
    return dict(protocol=full.ma9_adapter.PROTOCOL,installed=True,calls=calls,kernel_calls=2*calls,
        initializations=1,selected_tensors=389 if rank==0 else 194,gradient_elements=1000000,
        chunks=40,failed=False,audits=calls if mode=='smoke' else 0,
        self_test=dict(passed=True,cases=6,ordered_above_2pow24=True,
                       nan_inf_subnormal_signed_zero=True) if mode=='smoke' else None)


class AcceptanceTests(unittest.TestCase):
    def result(self,point,low,high):
        return full.comparison_acceptance({'F4-zcount/F4':dict(percent=point,ci95=[low,high])},
                                         {'F4':500,'F4-zcount':490})

    def test_two_percent_point_and_confidence_are_separate(self):
        result=self.result(2.1,1.7,2.5)
        self.assertTrue(result['increment_over_F4_validated']['F4-zcount'])
        self.assertTrue(result['target_point_observed'])
        self.assertFalse(result['target_validated'])
        self.assertTrue(self.result(2.2,2.0,2.4)['target_validated'])
        self.assertFalse(self.result(.2,-.1,.5)['increment_over_F4_validated']['F4-zcount'])

    def test_every_trial_must_pass(self):
        good=self.result(2.3,2.1,2.5)
        insufficient=self.result(2.3,1.9,2.7)
        result=full.aggregate_acceptance({'compare1':good,'compare2':insufficient})
        self.assertTrue(result['target_point_observed'])
        self.assertFalse(result['target_validated'])
        self.assertTrue(full.aggregate_acceptance({'compare1':good})['target_validated'])

    def test_invalid_statistics_are_rejected(self):
        for point,low,high in ((1,2,1),(float('nan'),0,2)):
            with self.assertRaises(ValueError):self.result(point,low,high)
        with self.assertRaises(ValueError):full.comparison_acceptance({}, {'F4':500,'F4-zcount':490})


if __name__ == '__main__':
    unittest.main()
