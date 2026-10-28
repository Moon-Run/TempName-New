"""CPU tests for unchanged controls, codec provenance and experiment admission.

Synthetic rank/trace fixtures test report rejection; they are not GPU results.
"""
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
import ablation6_build as builder
import ablation6_sources
import baseline as cli
import baseline_codec_build as codec_builder
import baseline_codec_validation as validator
import baseline_prepare as preparation
import baseline_protocol as sp
import baseline_report as report
import baseline_sources as sources
import detach_baseline as launcher
import scope_plan
import test_ablation6 as old_tests
import test_scope as fixtures
from common import HERE, read, sha, write

LIBRARY = dict(path='/CPU-FIXTURE/libbaseline_taco.so', sha256='a'*64)


def validation_record():
    return dict(passed=True, synthetic=False, library=LIBRARY, retained_outputs_unchanged=True,
        cases=[dict(case=case, shape=[1792,4096] if case=='full_shape' else [56,128],
                    chunks=7, relative_l2=.02, protocol_relative_l2=.001, packet_relative_l2=.001,
                    budget_vs_bf16=.05, budget_vs_codec=.01, budget_vs_packet=.005,
                    finite=True, passed=True) for case in validator.CASES])


def profile_fixture(policy):
    with mock.patch.object(old_tests, 'sp', sp):
        events = old_tests.profile_fixture(policy)
    if sp.is_tensor(policy):
        for name in ('fused_ash_compress_kernel', 'fused_ash_decompress_kernel'):
            events += [dict(cat='kernel', name=name, ph='X') for _ in range(256)]
    return events


class ContractTests(unittest.TestCase):
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
        helper += "\n    if sp.is_tensor(policy):self.assertEqual(set(adapter.codec_calls().values()),{3})\n"
        ns=dict(fixtures.__dict__,sp=sp,sources=sources);exec(helper,ns)
        for p in sp.POLICIES:
            for mode in ('timing','profile'):
                with self.subTest(policy=p,mode=mode):ns['exercise'](self,p,mode)

    def test_control_definitions_schedule_and_implementation_unchanged(self):
        jobs=sp.schedule()
        self.assertEqual(len(jobs),50)
        for p in sp.POLICIES[2:]:
            self.assertEqual(sp.definition(p),ablation4_protocol.definition(p))
        for a,b in zip(jobs,ablation4_protocol.schedule(rounds=2)):
            if a['policy'] in sp.POLICIES[2:]:self.assertEqual(a,b)
        self.assertEqual(sources.taco_source(),ablation6_sources.taco_source())
        self.assertEqual(sources.adapter_source().replace('codec_calls','naive_codec_calls'),
                         ablation6_sources.adapter_source())
        for fn in (sources.codec_source,sources.taco_source,sources.adapter_source,sources.worker_source):
            ast.parse(fn())

    def test_upstream_device_code_is_verbatim_and_fixed(self):
        text=(codec_builder.UPSTREAM/'taco.cu').read_text()
        extracted=codec_builder.kernel_source()
        self.assertEqual(text.count(extracted),1)
        self.assertIn('fused_ash_compress_kernel',extracted)
        self.assertIn('fused_ash_decompress_kernel',extracted)
        self.assertNotIn('encode_warp',extracted)
        with tempfile.TemporaryDirectory() as tmp:
            root=codec_builder.prepare(Path(tmp)/'codec',source_only=True)
            codec_builder.verify(root)
            (root/'upstream_kernels.cuh').write_text(extracted+'// drift\n')
            with self.assertRaises(ValueError):codec_builder.verify(root)

    def test_codec_validation_rejects_missing_cases_relaxed_limits_and_wrong_library(self):
        validator.verify(validation_record(),LIBRARY)
        for change in ('missing_case','relaxed_limit','exceeded_limit','nan','library','retention','synthetic'):
            row=validation_record()
            if change=='missing_case':row['cases'].pop()
            if change=='relaxed_limit':row['cases'][0]['budget_vs_codec']=.02
            if change=='exceeded_limit':row['cases'][0]['protocol_relative_l2']=.011
            if change=='nan':row['cases'][0]['packet_relative_l2']=float('nan')
            if change=='library':row['library']=dict(LIBRARY,sha256='b'*64)
            if change=='retention':row['retained_outputs_unchanged']=False
            if change=='synthetic':row['synthetic']=True
            with self.subTest(change=change),self.assertRaises(ValueError):validator.verify(row,LIBRARY)

    def test_profiles_require_actual_cuda_kernels_not_just_labels(self):
        for p in sp.POLICIES:report.profile_roles(profile_fixture(p),p)
        for p in sp.POLICIES[:2]:
            for name in ('fused_ash_compress_kernel','fused_ash_decompress_kernel'):
                events=[e for e in profile_fixture(p) if name not in e.get('name','')]
                with self.subTest(policy=p,kernel=name),self.assertRaises(ValueError):report.profile_roles(events,p)
            with self.assertRaises(ValueError):
                report.profile_roles(profile_fixture(p)+[dict(cat='kernel',name='taco_decode_ring_kernel')],p)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='baseline-CPU-FIXTURE-');cls.root=Path(cls.tmp.name)
        cls.parent=ablation4_build.prepare(cls.root/'parent',source_only=True)
        cls.flux=builder.prepare(cls.root/'flux',source_only=True)
        cls.codec=codec_builder.prepare(cls.root/'codec',source_only=True)
        cls.compare=preparation.prepare(cls.root/'compare',cls.parent,cls.flux,Path('/absent'),
                                        source_only=True,codec_root=cls.codec)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_dry_run_and_source_only_execution_rejected(self):
        with mock.patch.object(subprocess,'run',side_effect=AssertionError('launched')), \
                mock.patch.object(subprocess,'check_output',side_effect=AssertionError('GPU query')), \
                contextlib.redirect_stdout(io.StringIO()):
            info=report.verify_inputs(self.compare)
            self.assertEqual(len(cli.run(self.compare)),60)
            self.assertEqual(len(list(cli.mapping_jobs(self.compare,info))),6)
            with self.assertRaises(ValueError):cli.run(self.compare,execute=True)
        for p in (self.compare/'scripts').glob('*.py'):ast.parse(p.read_text())
        result=subprocess.run([sys.executable,str(self.compare/'scripts/worker.py')],capture_output=True,text=True,
                              env=dict(os.environ,H100_BASELINE_EXECUTE='1'))
        self.assertIn('Source-only',result.stderr)

    def test_reject_wrong_codec_and_duration(self):
        path=self.compare/'submission.json';saved=read(path)
        for key,value in (('codec_identity',{}),('codec_library',LIBRARY),
                          ('execution_timing',dict(warmup_steps=10,timed_steps=20))):
            try:
                write(path,dict(saved,**{key:value}))
                with self.assertRaises(ValueError):report.verify_inputs(self.compare)
            finally:write(path,saved)


    def test_full_fifty_step_report_and_truncation_codec_route_gates(self):
        root=preparation.prepare(self.root/'SYNTHETIC-NO-GPU',self.parent,self.flux,Path('/absent'),source_only=True,codec_root=self.codec)
        info=read(root/'submission.json');info.update(executable=True,source_only=False,calibration_hardware=scope_plan.frozen_mlp()['hardware'],codec_library=LIBRARY)
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
                path=folder/f'rank{rank}.json';row=read(path);row.update(communication=info['communication'],codec_identity=info['codec_identity'])
                row['codec_library']=LIBRARY if sp.is_tensor(p) else None
                row['codec_validation']=validation_record() if sp.is_tensor(p) and job['mode']=='smoke' else None
                row['codec_calls']={n:(row['warmup_steps']+len(row['losses']))*4 for n in row['enabled_modules']} if sp.is_tensor(p) else {}
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
            for field,value in (('losses',[1.]*20),('timed_steps',60),('codec_calls',{}),('codec_identity',{})):
                write(path,dict(saved,**{field:value}))
                with self.assertRaises(ValueError):report.window(root,path.parent,sp.FLUX_TENSOR,'timing',job['block'],job['index'])
            write(path,saved);path.unlink()
            with self.assertRaises(FileNotFoundError):report.report(root,offline=True)

class LauncherTests(unittest.TestCase):
    def test_read_only_preview_and_sequential_three_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'must-not-exist'
            with mock.patch.object(os,'fork',side_effect=AssertionError('fork')), \
                    mock.patch.object(subprocess,'run',side_effect=AssertionError('subprocess')), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                launcher.main(['--out',str(root)])
            self.assertFalse(root.exists())
            self.assertIn('7500 formal steps',output.getvalue())
            args=launcher.arguments(['--out',str(root),'--flux-build','/old-flux'])
            names=[label for label,_ in launcher.commands(args)]
            self.assertEqual(names[:2],['codec-prepare','codec-compile'])
            self.assertEqual(names[2:],[trial+'-'+stage for trial in launcher.TRIALS
                for stage in ('prepare','dry-run','controller','report')])

    def test_first_failure_stops_later_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=launcher.arguments(['--out',tmp,'--flux-build','/existing','--codec-build','/existing'])
            with mock.patch.object(launcher.launcher,'invoke',side_effect=RuntimeError('failed')) as invoke:
                with self.assertRaises(RuntimeError):launcher.supervise(args)
            self.assertEqual(invoke.call_count,1)
            self.assertEqual(read(Path(tmp)/'batch-state.json')['status'],'failed')


if __name__=='__main__':unittest.main()
