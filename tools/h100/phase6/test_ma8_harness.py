"""MA8 controller contracts; all synthetic timing values are validation fixtures."""
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

import ma8_bench as bench
import ma8_fullstep as full
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
        self.assertIn("report['ma8_activation_setup']", full.worker_source())
        self.assertNotIn('ma8_cache_tracker', full.worker_source())
        self.assertNotIn('ma8_activation_cache_state', full.worker_source())
        self.assertIn('H100_MA8_ARM', full.adapter_source())
        self.assertNotIn('H100_MA4_F_ARM', full.adapter_source())

    def test_baseline_setup_never_installs_and_step_functions_are_unchanged(self):
        from types import SimpleNamespace
        setup=ast.parse(full.activation_setup_source())
        # Execute the exact generated branch with setup functions replaced by
        # observers; no model or torch import is involved.
        branch=compile(ast.Module(body=setup.body[1:], type_ignores=[]), '<MA8 setup branch>', 'exec')
        for arm in bench.ARMS:
            actions=[]
            adapter=SimpleNamespace(
                install_activation_cache=lambda model, value: actions.append(('install',value)) or {'arm':value},
                inspect_activation_setup=lambda model, value: actions.append(('inspect',value)) or {'arm':value})
            scope=dict(ma8_adapter=adapter,os=SimpleNamespace(environ={'H100_MA8_ARM':arm}),raw=object())
            exec(branch,scope)
            self.assertEqual(actions,[('install' if arm=='F4-cache' else 'inspect',arm)])
        baseline=ast.parse(full._consumer_worker_source())
        candidate=ast.parse(full.worker_source())
        functions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body
                               if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertEqual(functions(baseline),functions(candidate))
        original_source=full.f4_adapter.consumer_autograd_source()
        actual_source=full.adapter_source()
        original=ast.parse(original_source).body[0]
        actual=next(node for node in ast.parse(actual_source).body
                    if isinstance(node,ast.ClassDef) and node.name=='ConsumerGemmRSFunction')
        self.assertEqual(ast.get_source_segment(original_source,original),ast.get_source_segment(actual_source,actual))

    def test_frozen_operator_import_is_self_contained(self):
        with tempfile.TemporaryDirectory() as temp:
            scripts = Path(temp)/'scripts'
            scripts.mkdir()
            bench.freeze_helpers(scripts)
            shutil.copy2(bench.__file__, scripts/'bench.py')
            code = ('import sys;sys.path.insert(0,sys.argv[1]);import bench;'
                    'assert bench.ARMS==("F4","F4-cache");assert len(bench.schedule())==16')
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
            self.assertEqual(plan['arms'], ['F4', 'F4-cache'])
            (root/'scripts/ma8_adapter.py').write_text('# changed adapter\n')
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
        self.plan.update(protocol=full.PROTOCOL,trials=1,blocks=8,activation_policy='retain_owned_column_input_uninstrumented',
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
                if key.startswith('cr2_'):row['ma8_'+key[4:]]=row.pop(key)
            row.update(protocol=full.PROTOCOL,ma8_protocol=bench.SOURCE_PROTOCOL,ma8_arm=arm,
                ma8_native_tuning={name:bench.expected_tuning(self.plan,arm) for name in row['ma8_codec_state']},
                ma8_final_parameters_sha256='c'*64,ma8_consumer_state={})
            row['ma8_activation_setup']=activation_setup_fixture(arm,row['target_modules'])
            universe={name+suffix:dict(shape=[4096],dtype='torch.bfloat16')
                      for name in row['target_modules'] for suffix in ('.weight','.bias')}
            universe.update({f'other.parameter{i}':dict(shape=[1],dtype='torch.bfloat16') for i in range(389-len(universe))})
            row['ma8_trainable_parameters']=universe
            row['ma8_parameter_trajectory']=self.fingerprints(universe) if job['mode']=='smoke' else []
            for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                row[name]=dict(path=f'/synthetic/{arm}/{name}',sha256=arm+name)
            for i,(name,states) in enumerate(row['ma8_submission_state'].items()):
                calls=sum(s['successful_calls'] for s in states)
                row['ma8_consumer_state'][name]=dict(projection_calls=calls,bda_calls=calls,bias_parameter_id=100+i,
                    native=[dict(s,bias_residual_calls=s['successful_calls']) for s in states])
            row['peak_allocated_gib']=40.0
            row['peak_reserved_gib']=42.0
            row['elapsed_seconds']=(means or {'F4':501.,'F4-cache':499.})[arm]*len(row['losses'])/1000
        return rows

    def test_fullstep_window_checks_setup_identity_and_rejects_v1_counters(self):
        job=dict(self.plan['schedule'][0],arm='F4-cache',mode='timing')
        rows=self.rows(job)
        with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
            self.assertEqual(len(full.window(self.root,self.plan,job)),8)
            rows[0]['ma8_activation_setup']['installed']=False
            with self.assertRaisesRegex(ValueError,'setup-only'):
                full.window(self.root,self.plan,job)
            rows[0]['ma8_activation_setup']['installed']=True
            rows[0]['ma8_activation_cache_state']={}
            with self.assertRaisesRegex(ValueError,'per-call counter'):
                full.window(self.root,self.plan,job)

    def test_all_parameter_gradient_and_update_fingerprints_are_compared(self):
        rows={a:self.rows(dict(arm=a,mode='smoke',trial=0,block=-1,index=-1,directory='smoke-'+a)) for a in bench.ARMS}
        with patch.object(full,'verify',return_value=self.plan), \
             patch.object(full,'window',side_effect=lambda root,plan,job:rows[job['arm']]), \
             patch.object(full,'_preflight',return_value={'ok':True}):
            self.assertEqual(full.preflight(self.root),{'ok':True})
            next(iter(rows['F4-cache'][7]['ma8_parameter_trajectory'][1]['gradients'].values()))['sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'gradients or updates'):
                full.preflight(self.root)

    def test_negative_performance_is_retained_and_final_parameter_drift_rejected(self):
        means={'F4':501.,'F4-cache':510.}
        ref=self.rows(dict(self.plan['schedule'][0],mode='timing'),means)
        def rows(root,plan,job):return self.rows(job,means)
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=rows), \
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))):
            result=full.report(self.root,save=False)
            self.assertEqual(result['formal_optimizer_steps'],800)
            self.assertEqual(len(result['all_samples']),16)
            self.assertEqual(result['increment_over_F4_validated'],{'F4-cache':False})
            self.assertTrue(result['target_evaluated'])
            self.assertFalse(result['target_validated'])
            self.assertEqual(result['target_percent'],2.0)
        def wrong(root,plan,job):
            result=self.rows(job,means)
            if job['arm']=='F4-cache':result[7]['ma8_final_parameters_sha256']='f'*64
            return result
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=wrong), \
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))):
            with self.assertRaisesRegex(ValueError,'Final paired model parameters differ'):
                full.report(self.root,save=False)


def activation_setup_fixture(arm,target_modules):
    roles={name.replace('.self_attention.linear_proj', '.self_attention.linear_qkv')
           .replace('.mlp.linear_fc2', '.mlp.linear_fc1'):
           'qkv' if name.endswith('.self_attention.linear_proj') else 'fc1'
           for name in target_modules}
    return dict(protocol=full.ma8_adapter.PROTOCOL,arm=arm,installed=arm=='F4-cache',module_roles=roles,
        function='_MA8CachedRawLinear' if arm=='F4-cache' else 'LinearWithGradAccumulationAndAsyncCommunication',
        source_sha256=dict(class_source=full.ma8_adapter.CLASS_SHA256,
                           column_forward=full.ma8_adapter.COLUMN_FORWARD_SHA256,
                           linear_wrapper=full.ma8_adapter.LINEAR_WRAPPER_SHA256))


class AcceptanceTests(unittest.TestCase):
    def result(self,point,low,high):
        return full.comparison_acceptance({'F4-cache/F4':dict(percent=point,ci95=[low,high])},
                                         {'F4':500,'F4-cache':490})

    def test_two_percent_point_and_confidence_are_separate(self):
        result=self.result(2.1,1.7,2.5)
        self.assertTrue(result['increment_over_F4_validated']['F4-cache'])
        self.assertTrue(result['target_point_observed'])
        self.assertFalse(result['target_validated'])
        self.assertTrue(self.result(2.2,2.0,2.4)['target_validated'])
        self.assertFalse(self.result(.2,-.1,.5)['increment_over_F4_validated']['F4-cache'])

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
        with self.assertRaises(ValueError):full.comparison_acceptance({}, {'F4':500,'F4-cache':490})


if __name__ == '__main__':
    unittest.main()
