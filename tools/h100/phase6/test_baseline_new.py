"""CPU regressions. Synthetic worker records never represent GPU measurements."""
import ast
from collections import Counter
import contextlib
import copy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import baseline_new as campaign
import baseline_new_protocol as p
import baseline_new_sources as sources
import baseline_new_build as builder
import ma8_adapter
import ma9_adapter
import ma11_adapter
from test_ma2_cr2 import fixture_plan
from test_tp8_dp2_campaign import codec_evidence


def fixture_worker(arm):
    selection=fixture_plan()['selection']
    codec={'path':'/CPU-FIXTURE/codec.so','sha256':'a'*64} if arm not in p.SELECTIVE else None
    config=sources.configuration(arm,selection,{'CPU_FIXTURE':True},codec)
    return dict(config=config,definition=sources.routing(arm).definition(p.POLICY[arm]),
                files={'config.json':'b'*64},libraries={})


def fixture_rows(job,plan,worker):
    arm,mode=job['arm'],job['mode']
    warmup,count=(2,3) if mode=='smoke' else (10,1 if mode=='profile' else 50)
    calls=(warmup+count)*4
    names=[f'decoder.layers.{i}.{s}' for i in range(32) for s in ('self_attention.linear_proj','mlp.linear_fc2')]
    parameter_names=[n+s for n in names for s in ('.weight','.bias')]+[f'other{i}' for i in range(261)]
    universe={n:dict(shape=[1],dtype='torch.float32') for n in parameter_names}
    parameters={n:dict(v,sha256='0'*64) for n,v in universe.items()}
    trajectory=[dict(step=i,gradients=parameters,parameters=parameters) for i in range(3)] if mode=='smoke' else []
    route='candidate' if arm in p.SELECTIVE else 'flux_tensor' if arm=='flux-basic' else 'tensor'
    rows=[]
    for rank in range(8):
        elapsed=count*(.65 if arm=='ma11' else .5)  # MA11 slower: finish all windows anyway.
        row=dict(protocol=p.PROTOCOL,run_id=plan['run_id'],arm=arm,mode=mode,rank=rank,
            block=job['block'],window=job['index'],trial=job['trial'],worker_config_sha256=worker['files']['config.json'],
            completed=True,passed=True,skips=0,world_size=8,local_rank=rank,tp_rank=rank,dp_rank=0,
            tp_group_ranks=list(range(8)),dp_group_ranks=[rank],model=p.MODEL,precision='bf16',
            torch='2.6.0+cu124',cuda='12.4',nccl=[2,21,5],policy=p.POLICY[arm],communication=plan['communication'],
            scope_definition=worker['definition'],role_plan_sha256=worker['config']['role_plan_sha256'],
            sampling=False,profiling=mode=='profile',tokens_per_optimizer_step=8192,warmup_steps=warmup,
            timing_protocol='full-step-one-clear-v2',losses=[1.]*count,grad_norms=[2.]*count,
            target_modules=names,timed_calls=dict.fromkeys(names,count*4),target_calls=dict.fromkeys(names,warmup*4),
            warmup_calls=dict.fromkeys(names,warmup*4),
            enabled_modules=names,independent_scope_instances=64,replaced_modules=64,fallback_calls=0,
            module_policies=dict.fromkeys(names,route),post_audit_routes=dict.fromkeys(names,route),
            forward_checks={n:dict(passed=True,budget_vs_bf16=.05,budget_vs_codec=.01,max_abs=0.,
                                  relative_l2=0.,protocol_relative_l2=0.) for n in names} if mode=='smoke' else {},
            trainable_parameters=universe,parameter_trajectory=trajectory,final_parameters_sha256='c'*64,
            peak_allocated_gib=20.,peak_reserved_gib=21.,timed_steps=count,elapsed_seconds=elapsed,
            ms_per_step=elapsed*1000/count,gpu_uuid=f'CPU-FIXTURE-{rank}',python='/CPU-FIXTURE/python',
            hardware={'CPU_SIMULATION':True},initial_parameters_sha256='d'*64,initial_rng_sha256='e'*64,
            tokens_sha256='f'*64,parameter_count_local=849207296,padded_vocab_size=33792)
        row.update(worker['libraries'])
        if arm in p.SELECTIVE:
            slots=[dict(protocol_version=1,failed=False,slot=i,bound_k=512,local_packet_ptr=100+i,
                        successful_calls=calls//2,bias_residual_calls=calls//2) for i in range(2)]
            prefix='ma4_f' if arm=='cr3' else 'ma11'
            row[prefix+'_consumer_state']={n:dict(projection_calls=calls,bda_calls=calls,bias_parameter_id=1,
                                                 native=slots) for n in names}
            row[prefix+'_submission_state']={n:slots for n in names}
            if arm!='cr3':
                roles={n.replace('.self_attention.linear_proj','.self_attention.linear_qkv')
                       .replace('.mlp.linear_fc2','.mlp.linear_fc1'):
                       'qkv' if n.endswith('.self_attention.linear_proj') else 'fc1' for n in names}
                row['ma11_activation_setup']=dict(protocol=ma8_adapter.PROTOCOL,arm='F4-cache',installed=True,
                    function='_MA8CachedRawLinear',module_roles=roles,source_sha256=dict(
                    class_source=ma8_adapter.CLASS_SHA256,column_forward=ma8_adapter.COLUMN_FORWARD_SHA256,
                    linear_wrapper=ma8_adapter.LINEAR_WRAPPER_SHA256))
                row['ma11_zero_count_state']=dict(protocol=ma9_adapter.PROTOCOL,installed=True,failed=False,
                    calls=warmup+count,kernel_calls=2*(warmup+count),initializations=1,
                    selected_tensors=389 if rank==0 else 194,gradient_elements=100,chunks=1,
                    audits=warmup+count if mode=='smoke' else 0,self_test=dict(passed=True,cases=6,
                    ordered_above_2pow24=True,nan_inf_subnormal_signed_zero=True) if mode=='smoke' else None)
                row['ma11_softmax_state']={'installed':False} if arm=='ma89' else dict(
                    protocol=ma11_adapter.PROTOCOL,installed=True,shape=list(ma11_adapter.SHAPE),
                    original_class_sha256=ma11_adapter.CLASS_SHA256,original_mask_sha256=ma11_adapter.MASK_SHA256,
                    modules=[f'decoder.layers.{i}.self_attention.core_attention.scale_mask_softmax' for i in range(32)],
                    forward_audits=640 if mode=='smoke' else 0,backward_audits=640 if mode=='smoke' else 0,
                    self_test=dict(passed=True,cases=6,forward_bitwise=True,backward_bitwise=True,
                                   rng_unchanged=True) if mode=='smoke' else None)
        else:
            row.update(codec_identity=worker['config']['codec_identity'],codec_library=worker['config']['codec_library'],
                       codec_calls=dict.fromkeys(names,calls),codec_validation=codec_evidence(worker['config']['codec_library'],rank)
                       if mode=='smoke' else None,flux_library_loaded=False,native_linear_forward_preserved=True,
                       retained_output_checks={r:dict(passed=True,independent_storage=True,
                            unchanged_after_subsequent_call=True) for r in ('attn','mlp')})
        if mode=='profile':
            for key in ('timed_steps','elapsed_seconds','ms_per_step'):row.pop(key)
        rows.append(row)
    return rows


class Contracts(unittest.TestCase):
    def test_five_arm_position_pair_and_carryover_balance(self):
        jobs=p.schedule(2,20)
        for trial in (1,2):
            rows=[j for j in jobs if j['trial']==trial]
            self.assertEqual(Counter(j['arm'] for j in rows),dict.fromkeys(p.ARMS,20))
            self.assertEqual(Counter((j['arm'],j['index']%5) for j in rows),
                             {(a,i):4 for a in p.ARMS for i in range(5)})
            order=[[j['arm'] for j in rows if j['block']==b] for b in range(20)]
            for a in p.ARMS:
                for b in p.ARMS:
                    if a!=b:self.assertEqual(sum(o.index(a)<o.index(b) for o in order),10)
            self.assertEqual(set(Counter(pair for o in order for pair in zip(o,o[1:])).values()),{4})
        with self.assertRaises(ValueError):p.schedule(1,8)

    def test_sources_preserve_all_training_arithmetic_and_timing(self):
        for arm in p.ARMS:
            text=sources.worker_source(arm);sources.validate_worker(text)
            self.assertNotIn('operator-controller.py',text)
            self.assertIn("args.data_parallel_size == 1",text)
            with self.assertRaises(ValueError):sources.validate_worker(text.replace('time.perf_counter()','time.monotonic()'))
        cr3=sources.worker_source('cr3')
        self.assertNotIn('install_activation_cache',cr3)
        self.assertNotIn('install_zero_count',cr3)
        self.assertNotIn('install_softmax',cr3)
        self.assertEqual(sources.worker_source('ma11'),sources.worker_source('ma89'))

    def test_workers_reject_unapproved_start_before_torch_import(self):
        for arm in p.ARMS:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'worker.py';path.write_text(sources.worker_source(arm))
                env=dict(os.environ);env.pop('H100_BASELINE_NEW_EXECUTE',None)
                result=subprocess.run([sys.executable,'-I','-B',str(path)],env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('GPU worker requires',result.stderr)
                self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_all_modes_and_arms_accept_complete_records(self):
        for arm in p.ARMS:
            worker=fixture_worker(arm);plan=dict(run_id='CPU',communication=worker['config']['communication'])
            for mode in ('smoke','timing','profile'):
                job=dict(arm=arm,mode=mode,block=0,index=0,trial=1)
                rows=fixture_rows(job,plan,worker)
                self.assertEqual(len(p.validate_rows(rows,job,plan,worker)),8)

    def test_bad_samples_audits_and_control_wrappers_rejected(self):
        for arm in p.ARMS:
            worker=fixture_worker(arm);plan=dict(run_id='CPU',communication=worker['config']['communication'])
            job=dict(arm=arm,mode='timing',block=0,index=0,trial=1)
            original=fixture_rows(job,plan,worker)
            mutations=[lambda r:r.update(elapsed_seconds=float('nan')),lambda r:r.update(skips=1),
                       lambda r:r.update(dp_rank=1),lambda r:r.update(timed_steps=20),
                       lambda r:r.update(forward_checks={'bad':{'passed':True}}),
                       lambda r:r.update(parameter_trajectory=[{}]),lambda r:r.update(module_policies={})]
            if arm=='cr3':mutations.append(lambda r:r.update(ma11_activation_setup={'installed':True}))
            if arm=='ma89':mutations.append(lambda r:r.update(ma11_softmax_state={'installed':True}))
            for mutate in mutations:
                rows=copy.deepcopy(original);mutate(rows[0])
                with self.assertRaises((ValueError,KeyError)):p.validate_rows(rows,job,plan,worker)
            with self.assertRaises(ValueError):p.validate_rows(original[:-1],job,plan,worker)

    def test_selective_equivalence_includes_gradients_and_final_weights(self):
        w=fixture_worker('ma11');plan=dict(run_id='CPU',communication=w['config']['communication'])
        rows=fixture_rows(dict(arm='ma11',mode='smoke',block=-1,index=0,trial=0),plan,w)
        p.equivalent(rows,copy.deepcopy(rows),True)
        for key in ('losses','grad_norms','final_parameters_sha256','parameter_trajectory'):
            altered=copy.deepcopy(rows);altered[0][key]=None
            with self.assertRaises(ValueError):p.equivalent(rows,altered,True)

    def test_rank_max_and_negative_report_all_samples(self):
        samples=[dict(j,ms_per_step=650. if j['arm']=='ma11' else 500.,
                      peak_allocated_gib=20.,peak_reserved_gib=21.) for j in p.schedule()]
        r=p.report(samples)
        self.assertEqual(r['formal_optimizer_steps'],2500)
        self.assertEqual(r['timing_rank_records'],400)
        self.assertAlmostEqual(r['comparisons']['compare1']['arms']['cr3']['ma11_mean_reduction_percent'],-30.)
        self.assertEqual(len(r['comparisons']['compare1']['pairs']),10)
        self.assertIn('MA11 / MA89',p.markdown(r))
        with self.assertRaises(ValueError):p.report(samples[:-1])
        rows=[dict(elapsed_seconds=t,peak_allocated_gib=20.,peak_reserved_gib=21.) for t in range(20,28)]
        self.assertEqual(p.summarize(rows,samples[0])['ms_per_step'],540.)

    def test_preview_no_writes_no_gpu_and_native_path_isolation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'gpu_idle',side_effect=AssertionError('GPU queried')):
            root=Path(tmp)/'not-created'
            with contextlib.redirect_stdout(io.StringIO()):campaign.main(['--out',str(root)])
            self.assertFalse(root.exists())
            a=campaign.arguments(['--out',str(root)])
            plan={'run_id':'CPU','communication':campaign.capture_communication()}
            job=dict(arm='megatron-cuda',mode='timing',trial=1,block=0,index=0)
            env=campaign.worker_environment(a,root,root/'scripts',job,root/'job',plan)
            self.assertNotIn('/flux',env['PYTHONPATH'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1,2,3,4,5,6,7')
            self.assertEqual(env['NCCL_NVLS_ENABLE'],'0')

    def test_profile_dispatch_keeps_original_consumer_contract(self):
        import baseline_new_profile as profiles
        with patch.object(profiles.ma4_f_fullstep,'trace_projection_records',return_value={}) as row, \
             patch.object(profiles.ma11_profile,'trace_composed_records',return_value={}):
            for arm in ('ma11','ma89'):
                self.assertTrue(profiles.analyze([],arm,0)['execution_validated'])
                row.assert_called_with([],'F4',profiles.ma4_f_bench.MECHANISM)
        for arm in p.ARMS:
            with self.assertRaises(ValueError):profiles.analyze([],arm,0)

    def test_freeze_failure_is_recorded_before_background_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cuda=root/'cuda';(cuda/'bin').mkdir(parents=True);(cuda/'lib64').mkdir()
            (cuda/'bin/nvcc').write_text('CPU FIXTURE')
            out=root/'run'
            with patch.object(campaign,'controller_files',side_effect=ValueError('INJECTED SOURCE FAILURE')), \
                 patch.object(campaign,'detach') as detach,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'INJECTED SOURCE'):
                    campaign.main(['--out',str(out),'--python',sys.executable,'--megatron',str(root),
                                   '--cuda-home',str(cuda),'--build-cuda-home',str(cuda),'--execute'])
                detach.assert_not_called()
            self.assertEqual(p.read(out/'batch-state.json')['status'],'failed')


class NativeSources(unittest.TestCase):
    def test_real_prepare_exact_variants_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=builder.prepare(Path(tmp)/'build')
            self.assertEqual(tuple(p.read(root/'selective/manifest.json')['variants']),p.SELECTIVE)
            self.assertFalse((root/'selective/cr3/src/gemm_rs/ma11_softmax.cuh').exists())
            self.assertFalse((root/'selective/ma89/src/gemm_rs/ma11_softmax.cuh').exists())
            self.assertTrue((root/'selective/ma11/src/gemm_rs/ma11_softmax.cuh').is_file())
            path=root/'selective/cr3/src/gemm_rs/taco_runtime.h'
            path.write_text(path.read_text()+'\n// changed\n')
            with self.assertRaises(ValueError):builder.verify(root)


def simulated_campaign(tmp, fail=None, bad_profile=False):
    root=tmp/'run';(root/'handoff').mkdir(parents=True)
    p.write(root/'batch-state.json',dict(status='launching'))
    p.write(root/'handoff/controllers.json',{})
    a=campaign.arguments(['--out',str(root)])
    started=[]
    def fake_build(build,megatron,cuda,jobs):
        (build/'megatron').mkdir(parents=True)
        p.write(build/'selective/selection-plan.json',fixture_plan()['selection'])
        for arm in p.ARMS:
            if arm=='megatron-cuda':continue
            src=builder.native_root(build,arm)
            for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                target=src/'python/flux/lib'/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text('CPU BINARY FIXTURE '+arm);(src/name).symlink_to(target)
        for name in ('codec-basic/libbasic_quant.so','codec-cuda/libbaseline_taco.so'):
            target=build/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('CPU CODEC FIXTURE')
        record=dict(status='built',files={});p.write(build/'baseline-build.json',record);return record
    def fake_run(command,cwd,env,log,timeout,peer):
        Path(log).write_text('CPU process simulation\n')
        if 'E2E_PROCESS_OUT' not in env:return
        dest=Path(env['E2E_PROCESS_OUT']);job=p.read(dest/'command.json')['job'];started.append(job)
        if job['directory']==fail:raise RuntimeError('INJECTED WORKER FAILURE')
        plan=p.read(root/'plan.json');worker=p.read(root/'workers.json')[job['arm']]
        for rank,row in enumerate(fixture_rows(job,plan,worker)):
            p.write(dest/f'rank{rank}.json',row)
            if job['mode']=='profile':
                with gzip.open(dest/f'rank{rank}-profile.json.gz','wt') as f:json.dump({'traceEvents':[]},f)
    def fake_profile(events,arm,rank):
        if bad_profile and arm=='cr3':raise ValueError('INJECTED PROFILE FAILURE')
        return dict(arm=arm,rank=rank,execution_validated=True,CPU_SIMULATION=True)
    with patch.object(campaign,'gpu_idle'),patch.object(builder,'prepare',side_effect=lambda _:None), \
         patch.object(builder,'compile_build',side_effect=fake_build),patch.object(campaign,'run_process',side_effect=fake_run), \
         patch.object(campaign.profiler,'analyze',side_effect=fake_profile),contextlib.redirect_stdout(io.StringIO()):
        campaign.supervise(a)
    return root,started


class Campaign(unittest.TestCase):
    def test_all_fifty_windows_and_profiles_complete_despite_negative_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,started=simulated_campaign(Path(tmp))
            self.assertEqual(Counter(j['mode'] for j in started),dict(smoke=5,timing=50,profile=5))
            self.assertEqual(p.read(root/'batch-state.json')['status'],'completed')
            report=p.read(root/'report.json')
            self.assertEqual(len(report['worker_records']),480)
            self.assertLess(report['comparisons']['compare1']['pairs']['ma11/cr3']['percent'],0)
            self.assertEqual(campaign.replay(root),report)
            path=root/'results/profile-cr3/rank0-profile.json.gz';path.write_bytes(b'changed')
            with self.assertRaises(ValueError):campaign.replay(root)

    def test_smoke_failure_stops_before_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'INJECTED WORKER'):
                simulated_campaign(Path(tmp),fail='smoke-cr3')
            root=Path(tmp)/'run'
            self.assertEqual(p.read(root/'batch-state.json')['status'],'failed')
            self.assertFalse((root/'samples.json').exists())
            self.assertFalse((root/'report.json').exists())

    def test_profile_failure_cannot_produce_completed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'INJECTED PROFILE'):
                simulated_campaign(Path(tmp),bad_profile=True)
            root=Path(tmp)/'run'
            self.assertEqual(len(p.read(root/'samples.json')),50)
            self.assertEqual(p.read(root/'batch-state.json')['status'],'failed')
            self.assertFalse((root/'report.json').exists())


if __name__=='__main__':unittest.main()
