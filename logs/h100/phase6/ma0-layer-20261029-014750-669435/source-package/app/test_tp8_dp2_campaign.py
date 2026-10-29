"""CPU-only two-controller simulation. Workers/build/profile are explicit fakes.

Exercises the real controller, coordinator, rank validation and final reporting
through every stage. Temporary outputs are synthetic and never GPU evidence.
"""
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import tp8_dp2_protocol as protocol
import tp8_dp2_train as train
import tp8_dp2_build as builder
import tp8_dp2_profile as profiler
import check_tp8_dp2_link as link
try:
    import communication
except ModuleNotFoundError:
    import ablation4_runtime as communication


class MemoryStore:
    def __init__(self):
        self.values={};self.lock=threading.Lock()
    def set(self,key,value):
        with self.lock: self.values[key]=value.encode() if isinstance(value,str) else value
    def check(self,keys):
        with self.lock: return all(k in self.values for k in keys)
    def get(self,key):
        with self.lock: return self.values[key]


def codec_evidence(library,rank):
    cases=('zeros','small','outliers','full_shape')
    return dict(passed=True,synthetic=False,retained_outputs_unchanged=True,library=library,
        cases=[dict(case=c,shape=[1792,4096] if c=='full_shape' else [56,128],chunks=7,
                    passed=True,finite=True,relative_l2=0.,protocol_relative_l2=0.,packet_relative_l2=0.,
                    budget_vs_bf16=.05,budget_vs_codec=.01,budget_vs_packet=.005) for c in cases],
        receiver_fusion=dict(passed=True,synthetic=False,rank=rank,retained_outputs_unchanged=True,
            cases=[dict(case=c,rank=rank,shape=[256,4096] if c=='full_shape' else [8,128],bitwise_equal=True) for c in cases]))


def worker_row(rank,job,cfg,run_id):
    arm,mode=job['arm'],job['mode'];tp,dp=rank%8,rank//8
    warmup,count=(2,3) if mode=='smoke' else (10,1 if mode=='profile' else 50)
    names=[f'decoder.layers.{i}.{r}' for i in range(32) for r in ('self_attention.linear_proj','mlp.linear_fc2')]
    parameters={f'cpu_fixture_parameter{i}':dict(shape=[1],dtype='torch.float32',sha256='0'*64) for i in range(389)}
    row=dict(completed=True,passed=True,skips=0,protocol=protocol.PROTOCOL,package_id=cfg['package_id'],
        arm=arm,mode=mode,rank=rank,block=job['block'],window=job['index'],run_id=run_id,world_size=16,
        local_rank=tp,tp_rank=tp,dp_rank=dp,tp_group_ranks=list(range(dp*8,dp*8+8)),dp_group_ranks=[tp,tp+8],
        model=protocol.MODEL,microsteps=2,sample_ids=[dp*2,dp*2+1],tokens_per_optimizer_step=8192,
        warmup_steps=warmup,timed_steps=count,elapsed_seconds=31. if arm=='ma11' else 25.,
        losses=[1.]*count,grad_norms=[2.]*count,target_modules=names,timed_calls={n:count*2 for n in names},
        forward_checks={n:dict(passed=True) for n in names} if mode=='smoke' else {},fallback_calls=0,
        peak_allocated_gib=20.,peak_reserved_gib=30.,gpu_uuid=f'CPU-FIXTURE-{rank}',hardware=dict(host=f'CPU-NODE-{dp}'),
        torch='2.6.0+cu124',cuda='12.4',nccl=[2,21,5],python=f'/cpu-fixture-node{dp}/python',
        initial_parameters_sha256=f'initial-{tp}',final_parameters_sha256=f'final-{tp}',
        initial_rng_sha256=f'rng-{tp}',tokens_sha256=f'partition-{dp}',global_tokens_sha256='CPU-GLOBAL-BATCH',
        parameter_count_local=849207296,padded_vocab_size=33792,communication=cfg['communication'],
        parameter_trajectory=[dict(step=i,gradients=parameters,parameters=parameters) for i in range(3)] if mode=='smoke' else [],
        ma11_activation_setup=dict(installed=True,module_roles=names),
        ma11_zero_count_state=dict(installed=True,failed=False,calls=warmup+count,kernel_calls=2*(warmup+count),
            initializations=1,selected_tensors=389 if tp==0 else 194,audits=warmup+count if mode=='smoke' else 0,
            self_test=dict(passed=True) if mode=='smoke' else None),
        ma11_softmax_state=dict(installed=True,modules=list(range(32)),shape=[1,4,2048,2048],
            forward_audits=320 if mode=='smoke' else 0,backward_audits=320 if mode=='smoke' else 0,
            self_test=dict(passed=True) if mode=='smoke' else None),
        ma11_consumer_state={n:dict(projection_calls=2*(warmup+count),bda_calls=2*(warmup+count)) for n in names},
        codec_calls={n:2*(warmup+count) for n in names},codec_library=cfg.get('codec_library'),
        codec_validation=codec_evidence(cfg.get('codec_library'),tp) if mode=='smoke' else None,
        native_linear_forward_preserved=True,flux_library_loaded=False)
    return row


def run_campaign(temp,trials=1,blocks=12,fail=None,fail_analysis=False):
    package=temp/'package';package.mkdir()
    (package/'app').mkdir();protocol.write(package/'app/model-arguments.json',[])
    for arm in protocol.ARMS:
        p=package/'workers'/arm;p.mkdir(parents=True)
        (p/'worker.py').write_text('# CPU fixture; never executed.\n')
        protocol.write(p/'config.json',dict(arm=arm))
    toolkit=temp/'cuda';(toolkit/'bin').mkdir(parents=True);(toolkit/'bin/nvcc').write_text('CPU fixture')
    info=dict(protocol=protocol.PROTOCOL,model=protocol.MODEL,arms=list(protocol.ARMS),symlinks={},
              files={str(p.relative_to(package)):protocol.sha(p) for p in package.rglob('*') if p.is_file()})
    info['package_id']=protocol.digest(info);protocol.write(package/'package.json',info)
    args=[train.parser(['--node-rank',str(r),'--python',sys.executable,'--package',str(package),
                       '--cuda-home',str(toolkit),'--run-id','CPU-FIXTURE','--trials',str(trials),
                       '--blocks',str(blocks),'--execute','--inside']) for r in range(2)]
    store=MemoryStore();started={0:[],1:[]};errors={}
    def coordinator(a):
        value=object.__new__(original_coordinator);value.rank=a.node_rank;value.timeout=10;value.store=store
        return value
    def fake_build(package,root,cuda,jobs,package_id):
        files={}
        for name in ['ma11/python/flux/lib/libflux_cuda.so','ma11/python/flux/lib/libflux_cuda_ths_op.so',
                     'flux-basic/python/flux/lib/libflux_cuda.so','flux-basic/python/flux/lib/libflux_cuda_ths_op.so',
                     'codec-basic/libbasic_quant.so','codec-cuda/libbaseline_taco.so']:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU BINARY FIXTURE '+name)
            files[name]=protocol.sha(p)
        (root/'megatron').mkdir()
        record=dict(status='built',package_id=package_id,files=files,environment={'CPU_SIMULATION':True})
        protocol.write(root/'build.json',record);return record
    def fake_verify(root,package_id):
        record=protocol.read(root/'build.json')
        assert record['package_id']==package_id
        for name,sha in record['files'].items(): assert protocol.sha(root/name)==sha
        return record
    def fake_run(command,cwd,env,log,timeout,coordinator):
        coordinator.check();command=list(map(str,command));Path(log).write_text('CPU process simulation\n')
        if '--out' in command:
            out=Path(command[command.index('--out')+1])
            protocol.write(out/'state.json',dict(status='completed',CPU_SIMULATION=True))
            return
        dest=Path(env['E2E_PROCESS_OUT']);scripts=Path(command[command.index('--max-restarts=0')+1]).parent
        cfg=protocol.read(scripts/'config.json');job=protocol.read(dest/'command.json')['job']
        started[coordinator.rank].append(job['directory'])
        if coordinator.rank==1 and job['directory']==fail: raise RuntimeError('INJECTED WORKER FAILURE')
        for rank in range(coordinator.rank*8,coordinator.rank*8+8):
            row=worker_row(rank,job,cfg,env['H100_RUN_ID'])
            if job['mode']=='profile':
                # The real profile worker reports no wall-clock timing fields.
                row.pop('timed_steps');row.pop('elapsed_seconds')
            row['worker_config_sha256']=protocol.sha(scripts/'config.json')
            if job['arm']!='megatron-cuda':
                for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                    p=(Path(env['E2E_BUILD'])/env['E2E_POLICY']/('python/flux/lib/'+name)).resolve()
                    row[name]=dict(path=str(p),sha256=protocol.sha(p))
            protocol.write(dest/f'rank{rank}.json',row)
            if job['mode']=='profile':
                with gzip.open(dest/f'rank{rank}-profile.json.gz','wt') as f:
                    json.dump({'traceEvents':[dict(CPU_SIMULATION=True,rank=rank,arm=job['arm'])]},f)
    def fake_analyze(events,arm,rank):
        assert events==[dict(CPU_SIMULATION=True,rank=rank,arm=arm)]
        if fail_analysis and arm=='flux-basic' and rank==8:
            raise ValueError('INJECTED PROFILE ANALYSIS FAILURE')
        return dict(CPU_SIMULATION=True,rank=rank)
    def target(a):
        try: train.execute(a,info)
        except BaseException as error: errors[a.node_rank]=str(error)
    torch=types.SimpleNamespace(__version__='2.6.0+cu124',version=types.SimpleNamespace(cuda='12.4'),
        cuda=types.SimpleNamespace(nccl=types.SimpleNamespace(version=lambda:(2,21,5))))
    sleep=time.sleep
    # Real collectives, GPUs, subprocess launch and compilation are not invoked.
    with mock.patch.dict(os.environ,{},clear=False),mock.patch.dict(sys.modules,{'torch':torch,'communication':communication}), \
         mock.patch.object(train,'Coordinator',side_effect=coordinator), \
         mock.patch.object(link,'local_requirements',return_value={'CPU_SIMULATION':True}), \
         mock.patch.object(builder,'build',side_effect=fake_build),mock.patch.object(builder,'verify',side_effect=fake_verify), \
         mock.patch.object(train,'run_process',side_effect=fake_run),mock.patch.object(profiler,'analyze',side_effect=fake_analyze), \
         mock.patch.object(time,'sleep',side_effect=lambda seconds:sleep(min(seconds,.001))), \
         contextlib.redirect_stdout(io.StringIO()):
        threads=[threading.Thread(target=target,args=(a,),daemon=True) for a in args]
        for thread in threads: thread.start()
        for thread in threads: thread.join(30)
        if any(t.is_alive() for t in threads):
            store.set('abort','CPU test timed out')
            for thread in threads: thread.join(3)
            raise AssertionError('Controller synchronization did not terminate')
    return args,started,errors


original_coordinator=train.Coordinator


class CampaignTests(unittest.TestCase):
    def test_complete_default_campaign_both_nodes_and_negative_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            args,started,errors=run_campaign(Path(temp))
            self.assertEqual(errors,{})
            self.assertEqual(started[0],started[1])
            self.assertEqual(len(started[0]),42)
            self.assertEqual(started[0][3:6],['profile-'+a for a in protocol.ARMS])
            for a in args:
                self.assertEqual(protocol.read(a.out/'state.json')['status'],'completed')
                result=protocol.read(a.out/'report.json')
                self.assertEqual(result['timing_rank_records'],576)
                self.assertEqual(len(result['all_samples']),36)
                self.assertLess(result['comparisons']['compare1']['comparisons']['ma11/flux-basic']['percent'],0)

    def test_two_trials_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            args,started,errors=run_campaign(Path(temp),trials=2,blocks=6)
            self.assertEqual(errors,{})
            result=protocol.read(args[0].out/'report.json')
            self.assertEqual(set(result['comparisons']),{'compare1','compare2'})
            self.assertEqual(len(result['all_samples']),36)

    def test_smoke_or_profile_failure_aborts_peer_before_any_timing(self):
        for job in ('smoke-flux-basic','profile-flux-basic'):
            with self.subTest(job=job),tempfile.TemporaryDirectory() as temp:
                args,started,errors=run_campaign(Path(temp),fail=job)
                self.assertEqual(set(errors),{0,1})
                for a in args:
                    self.assertEqual(protocol.read(a.out/'state.json')['status'],'failed')
                    self.assertFalse((a.out/'report.json').exists())
                    self.assertFalse(any(n.startswith('compare') for n in started[a.node_rank]))

    def test_timing_failure_never_publishes_a_complete_report(self):
        with tempfile.TemporaryDirectory() as temp:
            args,started,errors=run_campaign(Path(temp),fail=protocol.schedule()[1]['directory'])
            self.assertEqual(set(errors),{0,1})
            for a in args:
                self.assertFalse((a.out/'report.json').exists())
                progress=protocol.read(a.out/'timing-progress.json')
                self.assertFalse(progress['completed'])
                self.assertEqual(len(progress['samples']),1)

    def test_profile_parser_failure_aborts_peer_before_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            args,started,errors=run_campaign(Path(temp),fail_analysis=True)
            self.assertEqual(set(errors),{0,1})
            for a in args:
                self.assertEqual(protocol.read(a.out/'state.json')['status'],'failed')
                self.assertFalse((a.out/'report.json').exists())
                self.assertFalse(any(n.startswith('compare') for n in started[a.node_rank]))


if __name__=='__main__': unittest.main()
