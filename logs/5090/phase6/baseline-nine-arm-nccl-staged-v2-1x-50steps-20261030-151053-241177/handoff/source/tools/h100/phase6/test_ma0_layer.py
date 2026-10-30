"""CPU regressions: scopes, source startup, rank aggregation and full campaign.

Controller fixtures simulate worker results; they are NOT H100 measurements.
"""
import ast
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ma0_layer as campaign
import ma0_layer_protocol as p
import ma0_layer_sources as sources
import ma0_layer_worker as worker
from test_ma2_cr2 import fixture_plan
from test_tp8_dp2_package import fixture


def rows_for(job,plan):
    _,rows = fixture(job['arm'],'smoke')
    rows=rows[:8]
    for row in rows:
        rank=row['rank']
        row.update(protocol=p.PROTOCOL,run_id=plan['run_id'],mode=job['mode'],block=job['block'],window=job['index'],
            world_size=8,dp_group_ranks=[rank],torch='CPU-FIXTURE',cuda='CPU-FIXTURE',nccl=[],python=sys.executable,
            initial_rng_sha256=f'rng{rank}',retained_output_checks={'attn':{'passed':True},'mlp':{'passed':True}},
            instrumented=False)
        if job['mode'] == 'timing':
            duration=2.0 if job['arm']=='ma11' else 1.0  # Negative performance is still a completed test.
            row.update(weights_unchanged=True,instrumentation='cuda_events_no_profiler',
                warmup_iterations=plan['warmup'],iterations=plan['iterations'],forward_checks={},mechanism_validated=True,
                regions={f'{role}/{layer}':dict(input_sha256='input',grad_output_sha256='grad',weights_sha256=f'w{rank}',
                    boundary_equivalent=True,finite_gradients=True,gradient_parameters=6,
                    samples=[dict(forward_ms=duration,backward_ms=duration,total_ms=2*duration)
                             for _ in range(plan['iterations'])]) for role in p.ROLES for layer in range(32)})
    return rows


class LayerContracts(unittest.TestCase):
    def test_scope_contains_norm_bias_residual_and_restores_on_failure(self):
        class Identity:
            def __call__(self,x): return x
        class Layer:
            pre_mlp_layernorm = staticmethod(lambda x:x*2)
            def _forward_attention(self,x,attention_mask):
                residual=x+3
                return self.pre_mlp_layernorm(residual),residual,None
            def _forward_mlp(self,x,residual): return x*5+residual
        layer=Layer();norm=layer.pre_mlp_layernorm
        whole=layer._forward_attention(7,None)
        expected=layer._forward_mlp(whole[0],whole[1])
        with worker.attention_boundary(layer,Identity()):
            attention=worker.invoke(layer,'attention',7,None)
        self.assertIs(layer.pre_mlp_layernorm,norm)
        self.assertEqual(worker.invoke(layer,'mlp',attention,None),expected)
        with self.assertRaises(RuntimeError):
            with worker.attention_boundary(layer,Identity()): raise RuntimeError('fixture')
        self.assertIs(layer.pre_mlp_layernorm,norm)

    def test_event_order_has_no_sync_between_forward_and_backward(self):
        calls=[]
        class Event:
            def __init__(self,name): self.name=name
            def record(self): calls.append(self.name)
            def elapsed_time(self,other): return 2 if self.name=='start' and other.name=='end' else 1
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:calls.append('sync')),
                              autograd=SimpleNamespace(backward=lambda out,grad:calls.append('backward')))
        dist=SimpleNamespace(barrier=lambda:calls.append('barrier'))
        with patch.object(worker,'clear_region',side_effect=lambda *a:calls.append('clear')), \
             patch.object(worker,'invoke',side_effect=lambda *a:calls.append('forward') or 1):
            result,_=worker.one_iteration(torch,dist,None,'mlp',None,None,None,[],
                                          [Event(n) for n in ('start','middle','end')])
        self.assertEqual(calls,['clear','barrier','sync','start','forward','middle','backward','end','sync'])
        self.assertEqual(result,dict(forward_ms=1,backward_ms=1,total_ms=2))

    def test_rank_maxima_do_not_add_different_ranks(self):
        job=p.schedule()[0];plan=dict(run_id='test',warmup=2,iterations=5)
        rows=rows_for(job,plan)
        for i,row in enumerate(rows):
            for region in row['regions'].values():
                region['samples']=[dict(forward_ms=3 if i==0 else 1,backward_ms=1 if i==0 else 3,total_ms=4)]*5
        p.validate_rows(rows,job,plan)
        region=p.summarize_window(rows)['mlp/0']
        self.assertEqual(region,dict(forward_ms=3,backward_ms=3,total_ms=4))

    def test_missing_layer_rank_nan_and_mislabeled_worker_fail(self):
        job=p.schedule()[0];plan=dict(run_id='test',warmup=2,iterations=5)
        for mutate in (lambda rows:rows.pop(),lambda rows:rows[0]['regions'].pop('mlp/31'),
                       lambda rows:rows[0].update(world_size=16),
                       lambda rows:rows[0]['regions']['attention/0']['samples'][0].update(forward_ms=float('nan'))):
            rows=rows_for(job,plan);mutate(rows)
            with self.assertRaises(ValueError): p.validate_rows(rows,job,plan)

    def test_source_compiles_and_uses_single_node_original_worker(self):
        for arm in p.ARMS:
            text=sources.worker_source(arm)
            ast.parse(text);compile(text,arm,'exec')
            self.assertIn('data_parallel_size == 1',text)
            self.assertIn('H100_MA0_LAYER_EXECUTE',text)
            self.assertLess(text.index('result = measure_layers(globals())'),text.index("report['timing_protocol']"))
            self.assertNotIn('operator-controller.py',text)
            self.assertIn('successful, grad_norm, _ = optimizer.step()',text)
            # Execute the REAL generated startup through the first Torch import.
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'worker.py';path.write_text(text)
                script="""import builtins,os,runpy,sys
os.environ.update(H100_MA0_LAYER_EXECUTE='1',LOCAL_RANK='0',WORLD_SIZE='8')
os.sched_setaffinity=lambda *a: None
original=builtins.__import__
def block(name,*a,**k):
 if name=='torch': print('GPU_BOUNDARY');raise SystemExit(0)
 return original(name,*a,**k)
builtins.__import__=block
runpy.run_path(sys.argv[1],run_name='__main__')
"""
                result=subprocess.run([sys.executable,'-I','-c',script,str(path)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('GPU_BOUNDARY',result.stdout)

    def test_preview_does_not_create_out_or_import_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'uncreated'
            with contextlib.redirect_stdout(io.StringIO()): campaign.main(['--out',str(root)])
            self.assertFalse(root.exists())

    def test_paired_gain_and_slowdown(self):
        result=p.comparison([2]*8,[1]*8)
        self.assertEqual(result['percent'],50)
        self.assertEqual(result['ci95'],[50,50])
        self.assertLess(p.comparison([1]*8,[2]*8)['percent'],0)


class Campaign(unittest.TestCase):
    def run_fixture(self,fail=False):
        import tp8_dp2_build,tp8_dp2_protocol,tp8_dp2_train
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'package';build=root/'build';out=root/'out'
            for arm in p.ARMS:
                dest=package/'workers'/arm;dest.mkdir(parents=True)
                (dest/'common.py').write_text('# CPU fixture\n')
                parent=sources.ma11_fullstep if arm=='ma11' else sources.basic_quant_sources
                (dest/'adapter.py').write_text(parent.adapter_source())
                (dest/'taco_support.py').write_text(parent.taco_source())
                if arm=='ma11':
                    for name in ('ma8_adapter.py','ma9_adapter.py','ma11_adapter.py'):
                        (dest/name).write_bytes((sources.HERE/name).read_bytes())
                else:
                    (dest/'megatron_taco.py').write_text(parent.codec_source())
                (build/arm).mkdir(parents=True)
            p.write(package/'plans/selection-plan.json',fixture_plan()['selection'])
            codec=build/'codec-basic/libbasic_quant.so';codec.parent.mkdir();codec.write_bytes(b'CPU FIXTURE')
            (build/'megatron').mkdir()
            (out/'handoff').mkdir(parents=True)
            p.write(out/'handoff/controllers.json',{})
            p.write(out/'batch-state.json',dict(status='launching'))
            args=campaign.arguments(['--out',str(out),'--package',str(package),'--build-root',str(build),
                                     '--warmup','2','--iterations','5','--foreground','--execute'])
            started=[]
            def fake_process(command,cwd,env,log,timeout,peer):
                job=dict(arm=p.read(Path(env['PYTHONPATH'].split(':')[0])/'config.json')['arm'],
                         mode=env['E2E_MODE'],block=int(env['E2E_BLOCK']),index=int(env['E2E_WINDOW']))
                started.append(job)
                plan=p.read(out/'plan.json');rows=rows_for(job,plan)
                if fail and job['arm']=='ma11': rows[0]['forward_checks']={}
                for row in rows: p.write(Path(env['E2E_PROCESS_OUT'])/f'rank{row["rank"]}.json',row)
            # Native compilation and GPU processes are the only simulated work.
            # Real worker generation, identity gates and report flow execute.
            def fast_paired(a,b):
                percent=100*(1-sum(a)/sum(b))
                return dict(percent=percent,ci95=[percent,percent],paired_blocks=len(a))
            with patch.object(tp8_dp2_protocol,'verify_package',return_value={'package_id':'CPU'}), \
                 patch.object(tp8_dp2_protocol,'build_package_id',return_value='CPU'), \
                 patch.object(tp8_dp2_build,'build'),patch.object(tp8_dp2_build,'verify'), \
                 patch.object(campaign,'gpu_idle'),patch.object(tp8_dp2_train,'run_process',side_effect=fake_process), \
                 patch.object(p,'paired',side_effect=fast_paired),contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(ValueError): campaign.execute(args)
                else: campaign.execute(args)
            result=p.read(out/'batch-state.json')
            if not fail:
                output=p.read(out/'report.json')
                self.assertTrue(output['completed'])
                self.assertLess(output['roles']['mlp']['total_ms']['percent'],0)
                self.assertEqual(len(output['layers']),64)
                self.assertIn('backward',(out/'report.md').read_text())
                for arm in p.ARMS:
                    cfg=p.read(out/'workers'/arm/'scripts/config.json')
                    self.assertEqual(cfg['cases'][0]['dp'],1)
                    self.assertEqual(cfg['protocol'],p.PROTOCOL)
            return started,result

    def test_full_campaign_completes_all_blocks_despite_negative_gain(self):
        started,state=self.run_fixture()
        self.assertEqual(state['status'],'completed')
        self.assertEqual(len(started),18)
        self.assertEqual([j['mode'] for j in started[:2]],['smoke','smoke'])
        self.assertEqual([(j['block'],j['arm']) for j in started[2:]],
                         [(j['block'],j['arm']) for j in p.schedule()])

    def test_smoke_failure_stops_before_all_timing(self):
        started,state=self.run_fixture(fail=True)
        self.assertEqual(state['status'],'failed')
        self.assertEqual(len(started),2)


if __name__ == '__main__': unittest.main()
