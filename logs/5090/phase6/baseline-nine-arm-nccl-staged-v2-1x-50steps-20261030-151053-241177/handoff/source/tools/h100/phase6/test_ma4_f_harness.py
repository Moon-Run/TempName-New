"""Consumer experiment CPU contracts; synthetic timings are never GPU results."""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ma4_f_bench as bench
import ma4_f_fullstep as full
import detach_ma4_f as launcher
from test_ma2_cr2 import fixture_plan as cr2_plan, fixture_rows as cr2_rows


def plan_fixture(trials=1):
    plan=cr2_plan();plan.pop('target_percent')
    plan.update(protocol=full.PROTOCOL,trials=trials,trial_ids={str(i+1):f'trial{i+1}' for i in range(trials)},
                arms=list(bench.ARMS),schedule=full.expected_schedule(trials,8),target_ms=None,
                acceptance_rule=full.ACCEPTANCE_RULE,mechanism=bench.MECHANISM,build_identity={'mechanism':bench.MECHANISM},
                libraries={arm:{'files':{f'/synthetic/{arm}/{name}':arm+name for name in
                    ('libflux_cuda.so','libflux_cuda_ths_op.so')}} for arm in bench.ARMS})
    return plan


def fingerprints(universe=None):
    universe=universe or {'layer.bias':dict(shape=[4096],dtype='torch.bfloat16')}
    return [dict(step=i,gradients={name:dict(shape=v['shape'],dtype='torch.float32',sha256='a'*64) for name,v in universe.items()},
                 parameters={name:dict(v,sha256='b'*64) for name,v in universe.items()}) for i in range(3)]


def rows_fixture(plan,job,digest,means=None):
    rows=cr2_rows(plan,dict(job,arm='CR2'),digest);arm=job['arm']
    for row in rows:
        for key in list(row):
            if key.startswith('cr2_'):row['ma4_f_'+key[4:]]=row.pop(key)
        row.update(protocol=full.PROTOCOL,ma4_f_protocol=bench.SOURCE_PROTOCOL,ma4_f_arm=arm,
            ma4_f_native_mechanism={name:bench.expected_mechanism(plan,arm) for name in row['ma4_f_codec_state']},
            ma4_f_final_parameters_sha256='c'*64,ma4_f_consumer_state={})
        universe={name+suffix:dict(shape=[4096],dtype='torch.bfloat16') for name in row['target_modules'] for suffix in ('.weight','.bias')}
        row['ma4_f_trainable_parameters']=universe
        row['ma4_f_parameter_trajectory']=fingerprints(universe) if job['mode']=='smoke' else []
        for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            row[name]=dict(path=f'/synthetic/{arm}/{name}',sha256=arm+name)
        if arm=='F4':
            for i,(name,states) in enumerate(row['ma4_f_submission_state'].items()):
                row['ma4_f_consumer_state'][name]=dict(projection_calls=sum(s['successful_calls'] for s in states),
                    bda_calls=sum(s['successful_calls'] for s in states),bias_parameter_id=100+i,
                    native=[dict(s,bias_residual_calls=s['successful_calls']) for s in states])
        row['elapsed_seconds']=(means or {'F4':499.,'CR2':511.})[arm]*len(row['losses'])/1000
    return rows


def profile_fixture(arm,full_model=True):
    events=[];count=256 if full_model else 4
    for i in range(count):
        start=i*40;role='attn' if i<128 else 'mlp'
        if full_model:
            events += [dict(cat='user_annotation',name=f'scope.{role}.candidate',ts=start,dur=35,tid=1,pid=1),
                       dict(cat='cpu_op',name='ConsumerGemmRSFunction' if arm=='F4' else 'GemmRSFunction',ts=start+1,dur=22,
                            tid=1,pid=1,args={'External id':i})]
        else:events.append(dict(cat='user_annotation',name=f'ma4f.attn.{i}',ts=start,dur=35,tid=1,pid=1))
        chain=[('flux_bf16_reducescatter',10,2),('CudaIpcBarrierAllKernel',12,1),
               ('taco_ma4f_decode_consumer_kernel' if arm=='F4' else 'taco_ma2c_decode_kernel',13,5 if arm=='F4' else 2)]
        if arm=='CR2':chain += [('elementwise_add_kernel',15,1),('elementwise_add_kernel',16,1)]
        for j,(name,offset,duration) in enumerate(chain):
            corr=i*10+j
            external=i if j<3 or not full_model else 20000+i*10+j
            host_offset=2+j if j<3 or not full_model else 24+j
            events += [dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=start+host_offset,dur=.1,tid=1,pid=1,args={'correlation':corr}),
                       dict(cat='kernel',name=name,ts=start+(100 if full_model else 0)+offset,dur=duration,
                            args={'stream':7,'correlation':corr,'External id':external})]
        if full_model:
            events += [dict(cat='kernel',name='LayerNormForward',ts=start+100+(18 if arm=='F4' else 17),dur=2,
                            args={'stream':7,'External id':10000+i}),
                       dict(cat='cpu_op',name='FusedLayerNormAffineFunction',ts=start+32,dur=2,pid=1,tid=1,
                            args={'External id':10000+i})]
    return events


class ConsumerProtocolTests(unittest.TestCase):
    def test_schedule_preview_and_no_c_r(self):
        for trials in (1,2):
            jobs=full.expected_schedule(trials,8)
            self.assertEqual(len(jobs),trials*16)
            for trial in range(1,trials+1):
                self.assertEqual(Counter(j['arm'] for j in jobs if j['trial']==trial),{'F4':8,'CR2':8})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'absent'
            with contextlib.redirect_stdout(io.StringIO()) as out:launcher.main(['--out',str(root)])
            self.assertIn('800 formal optimizer steps',out.getvalue());self.assertFalse(root.exists())
            self.assertEqual(len(launcher.commands(launcher.arguments(['--out',str(root)]))),8)
            self.assertEqual(len(launcher.commands(launcher.arguments(['--out',str(root),'--ma4-f-build',str(root)]))),6)

    def test_sources_and_strong_gradient_gate_are_outside_timing(self):
        for source in (full.worker_source(),full.adapter_source(),full.taco_source()):ast.parse(source)
        worker=full.worker_source()
        self.assertIn("audit_parameters = MODE == 'smoke'",worker)
        self.assertIn("getattr(parameter, 'main_grad', None)",worker)
        self.assertIn("report['ma4_f_parameter_trajectory']",worker)
        self.assertNotIn('graph_preparation',worker)
        universe={'layer.bias':dict(shape=[4096],dtype='torch.bfloat16')}
        full.validate_parameter_trajectory(fingerprints(),universe)
        bad=fingerprints();bad[1]['gradients']={}
        with self.assertRaises(ValueError):full.validate_parameter_trajectory(bad,universe)
        changed=dict(universe,missing_weight=dict(shape=[4096],dtype='torch.bfloat16'))
        with self.assertRaises(ValueError):full.validate_parameter_trajectory(fingerprints(),changed)

    def test_frozen_operator_import_without_repository_pythonpath(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'operator/scripts';root.mkdir(parents=True);bench.freeze_helpers(root)
            (root/'bench.py').write_bytes(Path(bench.__file__).read_bytes())
            code='import sys,runpy;sys.path.insert(0,sys.argv[1]);m=runpy.run_path(sys.argv[1]+"/bench.py");import ma4_f_adapter;assert m["ARMS"]==("F4","CR2");assert ma4_f_adapter.consumer_autograd_source()'
            result=subprocess.run([sys.executable,'-I','-c',code,str(root)],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_fullstep_preparation_freezes_base_overlay_and_runtime_helpers(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);build=parent/'build';build.mkdir()
            selection=cr2_plan()['selection'];bench.write(build/'selection-plan.json',selection)
            identities={arm:dict(root='/synthetic/'+arm,files={}) for arm in bench.ARMS}
            root=parent/'training'
            args=argparse.Namespace(root=root,build=build,megatron=parent/'megatron',trials=1,blocks=8)
            with patch.object(bench,'build_identity',return_value={'mechanism':bench.MECHANISM}),\
                 patch.object(bench,'libraries',return_value=identities),\
                 patch.object(full,'snapshot_megatron',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            plan=full.verify(root,offline=True)
            self.assertEqual(plan['arms'],['F4','CR2']);self.assertEqual(plan['trials'],1)
            self.assertEqual(len(plan['schedule']),16)
            for name in ('ma4_f_fullstep_base.py','ma4_f_bench_base.py','ma4_f_adapter.py','ma4_f_bench.py'):
                self.assertIn('scripts/'+name,plan['source_files'])
            (root/'scripts/ma4_f_fullstep_base.py').write_text('changed dependency')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):full.verify(root,offline=True)

    def test_profile_counts_and_complete_consumer_chain_include_fused_work(self):
        for arm in bench.ARMS:
            result=bench.trace_path(profile_fixture(arm,False),arm,'attn')
            self.assertEqual(result['consumer_add_kernels_per_call'],0 if arm=='F4' else 2)
            result=full.trace_projection_records(profile_fixture(arm),arm,mechanism=bench.MECHANISM)
            self.assertEqual(sum(v['projection_calls'] for v in result.values()),256)
            self.assertEqual(result['attn']['decoder_start_to_layernorm_end_us'],7 if arm=='F4' else 6)
            self.assertEqual(result['attn']['gemm_end_to_layernorm_end_us'],8 if arm=='F4' else 7)
        events=profile_fixture('CR2')
        events.remove(next(e for e in events if bench.is_add(e)))
        with self.assertRaises(ValueError):full.trace_projection_records(events,'CR2',mechanism=bench.MECHANISM)

    def test_old_decoder_and_missing_layernorm_are_rejected(self):
        events=profile_fixture('F4')
        next(e for e in events if e.get('cat')=='kernel' and 'ma4f_decode' in e['name'])['name']='taco_ma2c_decode_kernel'
        with self.assertRaises(ValueError):full.trace_projection_records(events,'F4',mechanism=bench.MECHANISM)

    def test_extra_projection_kernel_copy_and_missing_norm_cpu_identity_fail(self):
        for extra in (dict(cat='kernel',name='unexpected_work',ts=111,dur=.1,args={'stream':7,'External id':0}),
                      dict(cat='gpu_memcpy',name='Memcpy DtoD',ts=118,dur=.1,args={'stream':7,'External id':99999}),
                      dict(cat='cpu_op',name='aten::clone',ts=3,dur=.1,tid=1,pid=1,args={})):
            events=profile_fixture('F4')+[extra]
            if extra['cat']=='gpu_memcpy':
                next(e for e in events if e.get('name')=='LayerNormForward')['ts']+=1
            with self.assertRaises(ValueError):full.trace_projection_records(events,'F4',mechanism=bench.MECHANISM)
        events=profile_fixture('F4')
        next(e for e in events if e.get('name')=='FusedLayerNormAffineFunction')['name']='unrelated_cpu_function'
        with self.assertRaises(ValueError):full.trace_projection_records(events,'F4',mechanism=bench.MECHANISM)
        events=profile_fixture('F4')
        next(e for e in events if e.get('name')=='LayerNormForward')['name']='unrelated_kernel'
        with self.assertRaises(ValueError):full.trace_projection_records(events,'F4',mechanism=bench.MECHANISM)


class ConsumerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=plan_fixture();bench.write(self.root/'plan.json',self.plan)
        self.digest=bench.sha(self.root/'plan.json')

    def test_native_fallback_and_parameter_checks_are_enforced(self):
        for arm in bench.ARMS:
            job=dict(self.plan['schedule'][0],arm=arm,mode='timing');rows=rows_fixture(self.plan,job,self.digest)
            with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
                self.assertEqual(len(full.window(self.root,self.plan,job)),8)
        job=dict(self.plan['schedule'][0],mode='timing');rows=rows_fixture(self.plan,job,self.digest)
        next(iter(rows[0]['ma4_f_consumer_state'].values()))['native'][0]['bias_residual_calls']-=1
        with patch.object(full,'read',side_effect=lambda p:rows[int(p.stem[4:])]):
            with self.assertRaises(ValueError):full.window(self.root,self.plan,job)

    def test_every_parameter_gradient_and_update_is_paired(self):
        by_arm={arm:rows_fixture(self.plan,dict(arm=arm,mode='smoke',trial=0,block=-1,index=-1,directory='smoke-'+arm),self.digest) for arm in bench.ARMS}
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=lambda root,plan,job:by_arm[job['arm']]),patch.object(full,'_preflight',return_value={'ok':True}):
            self.assertEqual(full.preflight(self.root),{'ok':True})
            next(iter(by_arm['F4'][0]['ma4_f_parameter_trajectory'][1]['gradients'].values()))['sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'gradients or updates'):full.preflight(self.root)

    def test_negative_fullstep_result_preserves_all_windows(self):
        def rows(root,plan,job):return rows_fixture(plan,job,self.digest,{'F4':520.,'CR2':511.})
        reference=rows_fixture(self.plan,dict(self.plan['schedule'][0],mode='timing'),self.digest)
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'window',side_effect=rows),\
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(reference))):
            result=full.report(self.root,save=False)
        self.assertEqual(result['formal_optimizer_steps'],800);self.assertEqual(len(result['all_samples']),16)
        self.assertFalse(result['target_evaluated']);self.assertFalse(result['increment_over_CR2_validated'])
        self.assertIn('520.000',full.report_markdown(result))


if __name__=='__main__':unittest.main()
