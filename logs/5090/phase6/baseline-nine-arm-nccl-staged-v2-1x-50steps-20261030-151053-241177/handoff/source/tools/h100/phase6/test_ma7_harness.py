"""MA7 controller contracts; all synthetic timing values are validation fixtures."""
import argparse
import ast
from collections import Counter
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ma7_bench as bench
import ma7_fullstep as full
from test_ma2_cr2 import fixture_plan


class ScheduleAndFrozenSources(unittest.TestCase):
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
        self.assertIn("report['ma7_backward_state']", full.worker_source())
        self.assertIn('H100_MA7_ARM', full.adapter_source())
        self.assertNotIn('H100_MA4_F_ARM', full.adapter_source())

    def test_frozen_operator_import_is_self_contained(self):
        with tempfile.TemporaryDirectory() as temp:
            scripts = Path(temp)/'scripts'
            scripts.mkdir()
            bench.freeze_helpers(scripts)
            shutil.copy2(bench.__file__, scripts/'bench.py')
            code = ('import sys;sys.path.insert(0,sys.argv[1]);import bench;'
                    'assert bench.ARMS==("F4","F4-wgrad");assert len(bench.schedule())==16')
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
            self.assertEqual(plan['arms'], ['F4', 'F4-wgrad'])
            (root/'scripts/ma7_adapter.py').write_text('# changed adapter\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                full.verify(root, offline=True)


class AcceptanceTests(unittest.TestCase):
    def test_positive_point_with_interval_crossing_zero_is_not_accepted(self):
        result = full.comparison_acceptance({'F4-wgrad/F4':dict(percent=.2, ci95=[-.1, .5])},
                                            {'F4':500, 'F4-wgrad':499})
        self.assertEqual(result['increment_over_F4_validated'], {'F4-wgrad':False})
        self.assertFalse(result['target_validated'])

    def test_every_independent_trial_must_pass(self):
        good = dict(increment_over_F4_validated={'F4-wgrad':True})
        bad = dict(increment_over_F4_validated={'F4-wgrad':False})
        result = full.aggregate_acceptance({'compare1':good, 'compare2':bad})
        self.assertEqual(result['increment_over_F4_validated'], {'F4-wgrad':False})
        self.assertTrue(full.aggregate_acceptance({'compare1':good})['increment_over_F4_validated']['F4-wgrad'])

    def test_bad_intervals_and_missing_arms_are_rejected(self):
        with self.assertRaises(ValueError):
            full.comparison_acceptance({'F4-wgrad/F4':dict(percent=1,ci95=[2,1])}, {'F4':500,'F4-wgrad':495})
        with self.assertRaises(ValueError):
            full.comparison_acceptance({}, {'F4':500,'F4-wgrad':495})


def backward_state_fixture(arm,calls,training=False):
    deferred=calls if arm=='F4-wgrad' else 0
    return dict(protocol=bench.ma7_adapter.PROTOCOL,arm=arm,backward_schedule='independent_weight_stream',
        forward_calls=calls,backward_calls=calls,baseline_backward_calls=calls if arm=='F4' else 0,
        deferred_wgrad_calls=deferred,cuda_wgrad_calls=deferred,cpu_wgrad_calls=0,
        main_grad_accumulations=deferred if training else 0,standalone_grad_assignments=0 if training else deferred,
        drained_wgrad_calls=deferred,drain_calls=calls,pending_wgrad=0,peak_pending_wgrad=int(deferred>0),
        async_gather_calls=0,wait_calls=0,failed=False,active=False)


def profile_fixture(arm):
    from test_ma6_harness import profile_fixture as baseline_fixture
    events=baseline_fixture('F4')
    for i in range(256):
        start=30000+i*100;external=20000+i;corr=40000+i*10
        if arm=='F4-wgrad':
            weight=next(e for e in events if e.get('cat')=='kernel' and e['args']['correlation']==corr+3)
            weight['args']['stream']=17;weight['dur']=12
            events.extend([
                dict(cat='user_annotation',name='ma7.wgrad',ts=start+9,dur=5,tid=1,pid=1),
                dict(cat='cuda_runtime',name='cudaStreamWaitEvent',ts=start+9.1,dur=.1,tid=1,pid=1,
                     args={'correlation':corr+8,'External id':external}),
                dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=start+12,dur=.1,tid=1,pid=1,
                     args={'correlation':corr+4,'External id':external}),
                dict(cat='kernel',name='vectorized_elementwise_add_kernel',ts=start+102,dur=2,
                     args={'correlation':corr+4,'External id':external,'stream':17}),
            ])
        events.extend([
            dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=start+27,dur=.1,tid=1,pid=1,
                 args={'correlation':corr+5,'External id':99999+i}),
            dict(cat='kernel',name='upstream_backward_kernel',ts=start+96,dur=4,
                 args={'correlation':corr+5,'External id':99999+i,'stream':7}),
        ])
        if i%64==63:
            events.append(dict(cat='user_annotation',name='ma7.drain',ts=start+40,dur=4,tid=1,pid=1))
            if arm=='F4-wgrad':
                events.append(dict(cat='cuda_runtime',name='cudaStreamWaitEvent',ts=start+41,dur=.1,tid=1,pid=1,
                                   args={'correlation':110000+i,'External id':120000+i}))
    return events


class BackwardEvidenceTests(unittest.TestCase):
    def test_only_fully_drained_expected_gpu_work_is_accepted(self):
        for arm in bench.ARMS:
            for training in (False,True):
                state=backward_state_fixture(arm,3,training)
                bench.validate_backward_state(state,arm,3,training=training)
                state['pending_wgrad']=1
                with self.assertRaises(ValueError):
                    bench.validate_backward_state(state,arm,3,training=training)
        state=backward_state_fixture('F4-wgrad',3,True)
        state['main_grad_accumulations']=2
        with self.assertRaises(ValueError):
            bench.validate_backward_state(state,'F4-wgrad',3,training=True)

    def test_actual_stream_and_microbatch_path_and_overlap(self):
        for arm in bench.ARMS:
            result=full.trace_backward_records(profile_fixture(arm),arm)
            self.assertEqual(result['attn']['backward_calls'],128)
            self.assertEqual(result['attn']['microbatch_drains'],4)
            self.assertEqual(result['attn']['separate_weight_stream'],arm=='F4-wgrad')
            self.assertEqual(result['attn']['overlap_calls'],128 if arm=='F4-wgrad' else 0)
            self.assertEqual(result['attn']['median_us']['wgrad_main_continuation_overlap_us'],4 if arm=='F4-wgrad' else 0)

    def test_missing_drain_submission_wait_and_wrong_stream_are_rejected(self):
        events=profile_fixture('F4-wgrad')
        events.remove(next(e for e in events if e.get('name')=='ma7.drain'))
        with self.assertRaisesRegex(ValueError,'four microbatch'):
            full.trace_backward_records(events,'F4-wgrad')
        events=profile_fixture('F4-wgrad')
        events.remove(next(e for e in events if e.get('name')=='cudaStreamWaitEvent'))
        with self.assertRaisesRegex(ValueError,'exactly once'):
            full.trace_backward_records(events,'F4-wgrad')
        events=profile_fixture('F4-wgrad')
        next(e for e in events if e.get('name')=='wgrad_gemm')['args']['stream']=7
        with self.assertRaisesRegex(ValueError,'separate ordered CUDA stream'):
            full.trace_backward_records(events,'F4-wgrad')

    def test_missing_drain_wait_and_multiple_weight_streams_are_rejected(self):
        events=profile_fixture('F4-wgrad')
        events.remove(next(e for e in events if e.get('name')=='cudaStreamWaitEvent' and e['args']['correlation']>100000))
        with self.assertRaisesRegex(ValueError,'one candidate join'):
            full.trace_backward_records(events,'F4-wgrad')
        events=profile_fixture('F4-wgrad')
        for event in events:
            if event.get('cat')=='kernel' and event['args'].get('correlation') in (40003,40004):
                event['args']['stream']=18
        with self.assertRaisesRegex(ValueError,'share one CUDA stream'):
            full.trace_backward_records(events,'F4-wgrad')

    def test_overlap_is_diagnostic_and_not_a_path_gate(self):
        events=profile_fixture('F4-wgrad')
        for e in events:
            if e.get('name')=='upstream_backward_kernel':e['ts']+=20
        result=full.trace_backward_records(events,'F4-wgrad')
        self.assertEqual(result['attn']['overlap_calls'],0)


class FullModelEvidenceTests(unittest.TestCase):
    def setUp(self):
        from test_ma2_cr2 import fixture_rows
        from test_ma4_f_harness import fingerprints
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.plan=fixture_plan();self.plan.pop('target_percent')
        tuning=bench.validate_tuning(None)
        self.plan.update(protocol=full.PROTOCOL,trials=1,blocks=8,backward_schedule='independent_weight_stream',
            trial_ids={'1':'trial1'},arms=list(bench.ARMS),schedule=full.expected_schedule(1,8),
            target_ms=None,acceptance_rule=full.ACCEPTANCE_RULE,tuning=tuning,build_identity={'tuning':tuning},
            libraries={a:{'files':{f'/synthetic/{a}/{n}':a+n for n in ('libflux_cuda.so','libflux_cuda_ths_op.so')}} for a in bench.ARMS})
        bench.write(self.root/'plan.json',self.plan)
        self.digest=bench.sha(self.root/'plan.json')
        self.cr2_rows,self.fingerprints=fixture_rows,fingerprints

    def rows(self,job,means=None):
        arm=job['arm'];rows=self.cr2_rows(self.plan,dict(job,arm='CR2'),self.digest)
        for row in rows:
            for key in list(row):
                if key.startswith('cr2_'):row['ma7_'+key[4:]]=row.pop(key)
            row.update(protocol=full.PROTOCOL,ma7_protocol=bench.SOURCE_PROTOCOL,ma7_arm=arm,
                ma7_native_tuning={name:bench.expected_tuning(self.plan,arm) for name in row['ma7_codec_state']},
                ma7_final_parameters_sha256='c'*64,ma7_consumer_state={},ma7_backward_state={})
            universe={name+suffix:dict(shape=[4096],dtype='torch.bfloat16')
                      for name in row['target_modules'] for suffix in ('.weight','.bias')}
            universe.update({f'other.parameter{i}':dict(shape=[1],dtype='torch.bfloat16') for i in range(389-len(universe))})
            row['ma7_trainable_parameters']=universe
            row['ma7_parameter_trajectory']=self.fingerprints(universe) if job['mode']=='smoke' else []
            for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                row[name]=dict(path=f'/synthetic/{arm}/{name}',sha256=arm+name)
            for i,(name,states) in enumerate(row['ma7_submission_state'].items()):
                calls=sum(s['successful_calls'] for s in states)
                row['ma7_consumer_state'][name]=dict(projection_calls=calls,bda_calls=calls,bias_parameter_id=100+i,
                    native=[dict(s,bias_residual_calls=s['successful_calls']) for s in states])
                row['ma7_backward_state'][name]=backward_state_fixture(arm,calls,True)
            row['peak_allocated_gib']=40.0
            row['peak_reserved_gib']=42.0
            row['elapsed_seconds']=(means or {'F4':501.,'F4-wgrad':499.})[arm]*len(row['losses'])/1000
        return rows

    def test_fullstep_window_checks_pending_work_and_every_module(self):
        job=dict(self.plan['schedule'][0],arm='F4-wgrad',mode='timing')
        rows=self.rows(job)
        with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
            self.assertEqual(len(full.window(self.root,self.plan,job)),8)
            next(iter(rows[0]['ma7_backward_state'].values()))['pending_wgrad']=1
            with self.assertRaisesRegex(ValueError,'backward schedule'):
                full.window(self.root,self.plan,job)

    def test_all_parameter_gradient_and_update_fingerprints_are_compared(self):
        rows={a:self.rows(dict(arm=a,mode='smoke',trial=0,block=-1,index=-1,directory='smoke-'+a)) for a in bench.ARMS}
        with patch.object(full,'verify',return_value=self.plan), \
             patch.object(full,'window',side_effect=lambda root,plan,job:rows[job['arm']]), \
             patch.object(full,'_preflight',return_value={'ok':True}):
            self.assertEqual(full.preflight(self.root),{'ok':True})
            next(iter(rows['F4-wgrad'][7]['ma7_parameter_trajectory'][1]['gradients'].values()))['sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'gradients or updates'):
                full.preflight(self.root)

    def test_negative_performance_is_retained_and_final_parameter_drift_rejected(self):
        means={'F4':501.,'F4-wgrad':510.}
        ref=self.rows(dict(self.plan['schedule'][0],mode='timing'),means)
        def rows(root,plan,job):return self.rows(job,means)
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=rows), \
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))):
            result=full.report(self.root,save=False)
            self.assertEqual(result['formal_optimizer_steps'],800)
            self.assertEqual(len(result['all_samples']),16)
            self.assertEqual(result['increment_over_F4_validated'],{'F4-wgrad':False})
            self.assertFalse(result['target_evaluated'])
        def wrong(root,plan,job):
            result=self.rows(job,means)
            if job['arm']=='F4-wgrad':result[7]['ma7_final_parameters_sha256']='f'*64
            return result
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=wrong), \
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))):
            with self.assertRaisesRegex(ValueError,'Final paired model parameters differ'):
                full.report(self.root,save=False)


if __name__ == '__main__':
    unittest.main()
