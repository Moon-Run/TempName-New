"""CPU-only admission, real adapter dispatch, synthetic evidence and launch gates."""
import ast
import contextlib
import copy
import csv
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import ablation4_build
import ablation4_protocol
import ablation6 as cli
import ablation6_build as builder
import ablation6_prepare as preparation
import ablation6_protocol as sp
import ablation6_report as report
import ablation6_sources as sources
import detach_ablation6_run_two as launcher
import scope_plan
import test_scope as fixtures
from common import HERE, read, sha, write


class ProtocolTests(unittest.TestCase):
    def test_real_generated_adapter_routes_and_shared_codec_calls_survive_audit(self):
        import inspect
        import textwrap
        helper=textwrap.dedent(inspect.getsource(fixtures.AdapterTests.exercise))
        helper=helper.replace('    class Candidate:', '''    class SharedCodec:
        def __init__(self,*args):
            self.calls=0;self.wire_bytes_per_rank=1;self.bf16_wire_bytes_per_rank=2
    def quantized(x,codec):
        codec.calls+=1;calls.append(('tensor',x.role));return x
    class Candidate:''')
        helper=helper.replace('owns_output = True','owns_output = not sp.is_fused_v2(policy)')
        helper=helper.replace("        def forward(self, x, w): calls.append(('candidate', x.role)); return x", '''            if sp.is_fused_v2(policy):
                self.plan_sha256=None
                self.workspace_info=[dict(shape=sp.SHAPES[role],packet_ptr=id(self))]
            if sp.is_flux_tensor(policy):
                self.plan_sha256=None
                self.codec=types.SimpleNamespace(placement='tensor',selection_sha256=None,
                    wire_bytes_per_rank=1,bf16_wire_bytes_per_rank=2,tensor_codec=SharedCodec())
                self.workspace_info=[dict(shape=sp.SHAPES[role],gemm_output_ptr=id(self))]
        def forward(self,x,w):
            if sp.is_flux_tensor(policy):self.codec.tensor_codec.calls+=1
            calls.append(('flux_tensor' if sp.is_flux_tensor(policy) else 'candidate',x.role));return x''')
        helper=helper.replace('TensorTacoRS=lambda *a: types.SimpleNamespace(wire_bytes_per_rank=1, bf16_wire_bytes_per_rank=2)',
                              'TensorTacoRS=SharedCodec')
        helper=helper.replace("apply=lambda x, codec: calls.append(('tensor', x.role)) or x",'apply=quantized')
        helper=helper.replace("sp.route(policy,n) != 'candidate'","sp.route(policy,n) not in ('candidate','flux_tensor')")
        helper += "\n    if sp.is_tensor(policy):self.assertEqual(set(adapter.naive_codec_calls().values()),{3})\n"
        ns=dict(fixtures.__dict__,sp=sp,sources=sources);exec(helper,ns)
        for p in sp.POLICIES:
            for mode in ('timing','profile'):
                with self.subTest(policy=p,mode=mode):ns['exercise'](self,p,mode)

    def test_five_policies_keep_abc_and_balance_fifty_windows(self):
        jobs=sp.schedule();self.assertEqual(len(jobs),50)
        rows=[[j['policy'] for j in jobs if j['round']==r and j['block']==b] for r in (1,2) for b in range(5)]
        self.assertEqual(rows[5:],[r[::-1] for r in rows[:5][::-1]])
        for i in range(5):self.assertEqual({r[i] for r in rows[:5]},set(sp.POLICIES))
        for p in sp.POLICIES[2:]:self.assertEqual(sp.definition(p),ablation4_protocol.definition(p))
        self.assertEqual([sp.needs_mapping(p) for p in sp.POLICIES],[False,False,True,True,True])
        self.assertEqual(sp.TIMING,dict(warmup_steps=10,timed_steps=50))

    def test_generated_workers_are_valid_and_codec_has_one_shared_implementation(self):
        for fn in (sources.naive_source,sources.taco_source,sources.adapter_source,sources.worker_source):ast.parse(fn())
        code=sources.naive_source()
        self.assertIn('from reference import hadamard, decode',code)
        self.assertIn('dist.all_to_all_single',code)
        self.assertIn("self.calls += 1",code)
        self.assertIn('op.codec.tensor_codec if is_flux_tensor',sources.adapter_source())
        self.assertIn('type(codec) is TensorTacoRS',sources.adapter_source())
        events=profile_fixture(sp.FLUX_TENSOR)
        for event in events:
            if event.get('name')=='flux_bf16_sm90_commnone':event['name']='cutlass_Sm90_GemmReduceScatter'
        with self.assertRaisesRegex(ValueError,'extra ReduceScatter'):report.profile_roles(events,sp.FLUX_TENSOR)

    def test_flux_only_gemm_then_exact_shared_codec_without_extra_collective(self):
        calls=[]
        class Tensor:
            dtype='bf16'
            def __init__(self,shape):self.shape=shape
            def data_ptr(self):return 1234
        class Codec:
            world=8;wire_bytes_per_rank=10;bf16_wire_bytes_per_rank=20
            def __init__(self,*args):self.calls=0
            def reduce(self,partial):
                calls.append(('shared_codec',partial));self.calls+=1;return object()
        class Gemm:
            def __init__(self,*args,**kwargs):
                calls.append(('constructor',args,kwargs))
            def forward(self,x,w,**kwargs):calls.append(('gemm_only',x,w,kwargs))
        torch=types.ModuleType('torch');torch.bfloat16='bf16';torch.empty=lambda shape,**kwargs:Tensor(shape)
        ns={}
        with mock.patch.dict(sys.modules,{'torch':torch,'flux':types.SimpleNamespace(GemmOnly=Gemm),
                                         'megatron_taco':types.SimpleNamespace(TensorTacoRS=Codec)}),mock.patch.dict(os.environ,{'E2E_MODE':'timing'}):
            exec((HERE/'ablation6_flux_tensor.py').read_text(),ns)
            op=ns['FluxTensorTaco'](None,2048,4096,512);x=Tensor((2048,512));w=Tensor((4096,512))
            one=op.forward(x,w);two=op.forward(x,w)
        self.assertEqual([v[0] for v in calls],['constructor','gemm_only','shared_codec','gemm_only','shared_codec'])
        self.assertEqual(calls[0][2],dict(transpose_weight=False,use_fp8_gemm=False))
        self.assertIs(calls[1][3]['output_buf'],calls[2][1]);self.assertIsNot(one,two)
        self.assertEqual(op.tensor_codec.calls,2)


def profile_fixture(policy):
    with mock.patch.object(fixtures,'sp',sp):events=fixtures.profile_fixture(policy)
    if sp.is_flux_tensor(policy):
        events=[e for e in events if e.get('name')!='taco_decode_ring_kernel']
        for e in events:
            if e.get('name','').startswith('scope.'):e['name']=e['name'].replace('.candidate','.flux_tensor')
            if e.get('name')=='flux_bf16_sm80':e['name']='flux_bf16_sm90_commnone'
    if sp.is_tensor(policy):
        # Actual worker creates four calls per each of 64 projections.
        names=['megatron_taco.'+n for n in ('encode','exchange','decode_reduce')]
        events=[e for e in events if e.get('name') not in names]
        for n in names+(['ablation6.flux_gemm_only'] if sp.is_flux_tensor(policy) else []):
            events += [dict(name=n,ph='X',cat='user_annotation',pid=1,tid=1,ts=i*1000,dur=1) for i in range(256)]
        events.append(dict(name='c10d::alltoall_base_'))
    return events


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='ablation6-CPU-FIXTURE-');cls.root=Path(cls.tmp.name)
        cls.parent=ablation4_build.prepare(cls.root/'parent',source_only=True)
        cls.flux=builder.prepare(cls.root/'flux',source_only=True)
        cls.compare=preparation.prepare(cls.root/'compare',cls.parent,cls.flux,Path('/absent'),source_only=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_cpu_preparation_dry_run_and_nonexecutable_worker(self):
        with mock.patch.object(subprocess,'run',side_effect=AssertionError('launched')),mock.patch.object(subprocess,'check_output',side_effect=AssertionError('GPU query')),contextlib.redirect_stdout(io.StringIO()):
            jobs=cli.run(self.compare);info=report.verify_inputs(self.compare)
            self.assertEqual(len(jobs),60);self.assertEqual(len(list(cli.mapping_jobs(self.compare,info))),6)
            with self.assertRaises(ValueError):cli.run(self.compare,execute=True)
        for p in (self.compare/'scripts').glob('*.py'):ast.parse(p.read_text())
        result=subprocess.run([sys.executable,str(self.compare/'scripts/worker.py')],capture_output=True,text=True,
                              env=dict(os.environ,H100_ABLATION6_EXECUTE='1'))
        self.assertIn('Source-only',result.stderr)
        cm=(self.flux/builder.VARIANT/'src/CMakeLists.txt').read_text()
        self.assertIn('add_subdirectory(comm_none)',cm);self.assertNotIn('add_subdirectory(gemm_rs)',cm)

    def test_wrong_duration_or_codec_or_build_is_rejected(self):
        path=self.compare/'submission.json';saved=read(path)
        for field,value in (('execution_timing',dict(warmup_steps=10,timed_steps=60)),('naive_codec_identity',{})):
            try:
                write(path,dict(saved,**{field:value}))
                with self.assertRaises(ValueError):report.verify_inputs(self.compare)
            finally:write(path,saved)
        m=read(self.flux/'manifest.json');m['source_sha256']['src/comm_none/ths_op/gemm_only.cc']='f'*64
        with self.assertRaisesRegex(ValueError,'source changed'):builder.validate_snapshot(m)

    def test_strict_profiles_reject_fused_codec_or_missing_naive_calls(self):
        for p in sp.POLICIES:report.profile_roles(profile_fixture(p),p)
        events=profile_fixture(sp.FLUX_TENSOR)
        with self.assertRaises(ValueError):report.profile_roles(events+[dict(name='taco_decode_ring_kernel')],sp.FLUX_TENSOR)
        changed=[e for e in events if e.get('name')!='ablation6.flux_gemm_only']
        with self.assertRaisesRegex(ValueError,'GemmOnly'):report.profile_roles(changed,sp.FLUX_TENSOR)
        changed=[e for e in events if e.get('name')!='megatron_taco.encode']
        with self.assertRaises(ValueError):report.profile_roles(changed,sp.FLUX_TENSOR)

    def test_full_fifty_step_report_and_truncation_codec_route_gates(self):
        root=preparation.prepare(self.root/'SYNTHETIC-NO-GPU',self.parent,self.flux,Path('/absent'),source_only=True)
        info=read(root/'submission.json');info.update(executable=True,source_only=False,calibration_hardware=scope_plan.frozen_mlp()['hardware'])
        for p,v in info['variants'].items():
            if not sp.is_native(p):v['libraries']={n:dict(path='/CPU-FIXTURE/'+p+'/'+n,sha256='f'*64) for n in ('libflux_cuda.so','libflux_cuda_ths_op.so')}
        write(root/'submission.json',info);write(root/'start-gpu-check.json',dict(passed=True,run_id=info['run_id'],synthetic=True))
        for job in cli.jobs(info,'all'):
            p=job['policy'];original_read=fixtures.read
            def per_policy(path):
                result=original_read(path)
                return dict(roles=sp.role_plans(result,p)) if Path(path).name=='selection-plan.json' else result
            with mock.patch.object(fixtures,'sp',sp),mock.patch.object(fixtures,'read',side_effect=per_policy):
                fixtures.make_window(root,info,job)
            folder=root/'results'/job['directory'];write(folder/'command.json',dict(job,scope=sp.definition(p),communication=info['communication']))
            for rank in range(8):
                path=folder/f'rank{rank}.json';row=read(path);row.update(communication=info['communication'],naive_codec_identity=info['naive_codec_identity'])
                row['naive_codec_calls']={n:(row['warmup_steps']+len(row['losses']))*4 for n in row['enabled_modules']} if sp.is_tensor(p) else {}
                if sp.is_fused_v2(p) or sp.is_flux_tensor(p):
                    for c in row['selection_checks'].values():c.update(mask_sha256=None,plan_sha256=None,wire_bytes=7*256*32*136)
                    field='gemm_output_ptr' if sp.is_flux_tensor(p) else 'packet_ptr'
                    row['workspace_checks']={n:[dict(shape=s[0]['shape'],**{field:s[0]['packet_ptr']})] for n,s in row['workspace_checks'].items()}
                write(path,row)
                if job['mode']=='profile':
                    with gzip.open(folder/f'rank{rank}-profile.json.gz','wt') as f:json.dump(dict(traceEvents=profile_fixture(p)),f)
        plans=read(root/'scripts/selection-plan.json')
        for p,role,_ in cli.mapping_jobs(root,info):
            with (root/'results'/f'mapping-{p}-{role}.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['rank','index','m','n','destination','sm_occupancy']);writer.writeheader()
                for rank,coords in enumerate(sp.role_plans(plans,p)[role]['entry']['coords']):
                    for i,tile in enumerate(coords):writer.writerow(dict(rank=rank,index=i,m=tile//32,n=tile%32,destination=tile//64,sm_occupancy=2))
        # Synthetic evidence exercises all report gates except GPU build admission;
        # source-only worker code stays permanently disabled.
        with mock.patch.object(report,'verify_inputs',return_value=info):
            report.preflight(root,offline=True);result=report.report(root,offline=True)
            self.assertTrue(result['completed']);self.assertEqual(result['formal_optimizer_steps'],2500)
            self.assertEqual(result['candidate_shape_mappings'],6);self.assertEqual(result['timing_rank_records'],400)
            self.assertEqual(result['rounds']['1']['mean_ms'][sp.POLICIES[0]],1200.)
            self.assertEqual(result['rounds']['combined']['mean_loss'][sp.FLUX_TENSOR],1.)
            job=next(j for j in info['schedule'] if j['policy']==sp.FLUX_TENSOR)
            path=root/'results'/job['directory']/'rank0.json';saved=read(path)
            for field,value in (('losses',[1.]*20),('timed_steps',60),('naive_codec_calls',{}),('naive_codec_identity',{})):
                write(path,dict(saved,**{field:value}))
                with self.assertRaises(ValueError):report.window(root,path.parent,sp.FLUX_TENSOR,'timing',job['block'],job['index'])
            write(path,saved);path.unlink()
            with self.assertRaises(FileNotFoundError):report.report(root,offline=True)


class LauncherTests(unittest.TestCase):
    def test_two_reports_complete_in_order_and_first_old_duration_stops_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            for old_duration in (False,True):
                root=Path(tmp)/str(old_duration);root.mkdir()
                args=launcher.arguments(['--out',str(root),'--flux-build','/existing-flux']);calls=[]
                def invoke(cmd,log,env):
                    action=cmd[2];folder=Path(cmd[3]);calls.append((action,folder.name,'--execute' in cmd))
                    if action=='prepare':
                        folder.mkdir();codec=dict(implementation=sp.NAIVE_CODEC)
                        info=dict(protocol=sp.PROTOCOL,run_id=folder.name,executable=True,source_only=False,
                            execution_timing=sp.TIMING,rounds=2,blocks_per_round=5,timing_windows=50,schedule=sp.schedule(),
                            communication=dict(environment={'NCCL_NVLS_ENABLE':'0'}),naive_codec_identity=codec,
                            config={},variants={},calibration_hardware={},megatron='same',megatron_commit='same',
                            candidate_source_commit='same',build_manifest={},flux_build_manifest={},build_files={},controllers={},files={})
                        write(folder/'submission.json',info)
                    elif action=='run' and '--execute' in cmd:
                        info=read(folder/'submission.json');write(folder/'all-state.json',dict(status='completed'))
                        write(folder/'preflight.json',dict(completed=True,submission_sha256=sha(folder/'submission.json'),
                            communication=info['communication'],naive_codec_identity=info['naive_codec_identity']))
                        write(folder/'report.json',dict(protocol=sp.PROTOCOL,run_id=info['run_id'],completed=True,correctness_passed=True,
                            execution_timing=sp.TIMING,formal_optimizer_steps=2500,timing_windows=50,all_samples=[{}]*50,
                            round_count=2,blocks_per_round=5,timing_rank_records=400,preflight_rank_records=80,profile_traces=40,
                            candidate_shape_mappings=6,communication=info['communication'],naive_codec_identity=info['naive_codec_identity'],
                            measurement_identity={},evidence=dict(preflight_sha256=sha(folder/'preflight.json'))))
                    elif action=='report' and folder.name=='compare1' and old_duration:
                        result=read(folder/'report.json');result['execution_timing']=dict(warmup_steps=10,timed_steps=60)
                        write(folder/'report.json',result)
                with mock.patch.object(launcher.launcher,'invoke',side_effect=invoke),contextlib.redirect_stdout(io.StringIO()):
                    if old_duration:
                        with self.assertRaises(ValueError):launcher.supervise(args)
                        self.assertNotIn(('run','compare2',True),calls)
                    else:
                        result=launcher.supervise(args);self.assertEqual(result['formal_optimizer_steps'],5000)
                        self.assertEqual(calls[-4:],[('run','compare1',True),('report','compare1',False),('run','compare2',True),('report','compare2',False)])

    def test_preview_exact_order_and_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(os,'fork',side_effect=AssertionError('fork')), \
                mock.patch.object(subprocess,'run',side_effect=AssertionError('launched')),contextlib.redirect_stdout(io.StringIO()) as output:
            root=Path(tmp)/'new';launcher.main(['--out',str(root)])
            self.assertFalse(root.exists());self.assertIn('50 timed steps',output.getvalue())
            steps=launcher.commands(launcher.arguments(['--out',str(root)]))
            self.assertEqual(len(steps),10);self.assertEqual([x[0] for x in steps[:2]],['build-prepare','compile'])
            self.assertEqual([x[0] for x in steps[-4:]],['compare1-controller','compare1-report','compare2-controller','compare2-report'])

    def test_build_failure_stops_all_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=launcher.arguments(['--out',tmp,'--nccl-root','/fake-nccl']);calls=[]
            def fail(cmd,log,env):
                calls.append(log.stem)
                if log.stem=='compile':raise subprocess.CalledProcessError(1,cmd)
            with mock.patch.object(launcher.launcher,'invoke',side_effect=fail),self.assertRaises(subprocess.CalledProcessError):launcher.supervise(args)
            self.assertEqual(calls,['build-prepare','compile']);self.assertEqual(read(Path(tmp)/'batch-state.json')['status'],'failed')


if __name__=='__main__':unittest.main()
