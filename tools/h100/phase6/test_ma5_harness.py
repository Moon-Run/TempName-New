"""CPU-only MA5 evidence contracts; fixture timings are synthetic, never GPU claims."""
import argparse
import ast
from collections import Counter
import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ma5_bench as bench
import ma5_fullstep as full
from test_ma2_cr2 import fixture_plan as cr2_plan, fixture_rows as cr2_rows, submission_fixture, trace_fixture

TUNING = dict(decoder_threads=256,decoder_blocks=528,gemm_sms=-1,gemm_tile_k=32)


def fixture_plan(trials=1):
    plan=cr2_plan()
    plan.pop('target_percent')
    plan.update(protocol=full.PROTOCOL,trials=trials,trial_ids={str(i+1):f'trial{i+1}' for i in range(trials)},
                arms=list(bench.ARMS),schedule=full.expected_schedule(trials,8),target_ms=full.TARGET_MS,
                acceptance_rule=full.ACCEPTANCE_RULE,tuning=dict(TUNING),build_identity={'tuning':dict(TUNING)},
                libraries={arm:{'files':{f'/synthetic/{arm}/{name}':arm+name for name in
                    ('libflux_cuda.so','libflux_cuda_ths_op.so')}} for arm in bench.ARMS})
    return plan


def fixture_rows(plan,job,digest,means=None):
    oldjob=dict(job,arm='CR2')
    rows=cr2_rows(plan,oldjob,digest)
    arm=job['arm']
    for row in rows:
        for key in list(row):
            if key.startswith('cr2_'):
                row['ma5_'+key[4:]]=row.pop(key)
        row.update(protocol=full.PROTOCOL,ma5_protocol=bench.SOURCE_PROTOCOL,ma5_arm=arm,
                   ma5_native_tuning={name:bench.expected_tuning(plan,arm) for name in row['ma5_codec_state']})
        for name in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            row[name]=dict(path=f'/synthetic/{arm}/{name}',sha256=arm+name)
        row['elapsed_seconds']=(means or {'T':499.,'CR2':511.})[arm]*len(row['losses'])/1000
    return rows


def kernel_args(arm,tuning=TUNING):
    expected=bench.expected_tuning({'tuning':tuning},arm)
    return dict(grid=[expected['decoder_blocks'],1,1],block=[expected['decoder_threads'],1,1])


def gemm_name(tile=32):
    return f'flux_bf16_reducescatter_64x64x{tile}_16x8x16_streamksk_nil_128x128x{tile}_gemmstreamk_3_rasterheuristic'


def projection_trace(arm,tuning=TUNING):
    events=trace_fixture('CR2')
    expected=bench.expected_tuning({'tuning':tuning},arm)
    for event in events:
        if event['cat']!='kernel':continue
        if 'flux_bf16' in event['name']:
            event['name']=gemm_name(expected['gemm_tile_k'])
            event['args'].update(grid=[512,1,1],block=[128,1,1])
        elif 'decode' in event['name']:
            event['name']='taco_ma5_decode_kernel(X)' if arm=='T' else 'taco_ma2c_decode_kernel(X)'
            event['args'].update(kernel_args(arm,tuning))
    return events


def operator_trace(arm,tuning=TUNING):
    events=[];expected=bench.expected_tuning({'tuning':tuning},arm)
    for i in range(4):
        start=i*100
        events.append(dict(cat='user_annotation',name=f'ma5.attn.{i}',ph='X',pid=1,tid=1,ts=start,dur=40))
        for j,name in enumerate((gemm_name(expected['gemm_tile_k']),'CudaIpcBarrierAllKernel',
                                'taco_ma5_decode_kernel(X)' if arm=='T' else 'taco_ma2c_decode_kernel(X)')):
            correlation=i*3+j
            events.append(dict(cat='cuda_runtime',name='cudaLaunchKernel',pid=1,tid=1,ts=start+2+j*2,dur=1,
                               args={'correlation':correlation}))
            args=dict(correlation=correlation,stream=7,grid=[512,1,1],block=[128,1,1])
            if j==2:args.update(kernel_args(arm,tuning))
            events.append(dict(cat='kernel',name=name,ts=start+10+j*5,dur=3,args=args))
    return events


class ProtocolTests(unittest.TestCase):
    def test_one_or_two_balanced_trials(self):
        for trials in (1,2):
            jobs=full.expected_schedule(trials,8)
            self.assertEqual(len(jobs),trials*16)
            for trial in range(1,trials+1):
                selected=[j for j in jobs if j['trial']==trial]
                self.assertEqual(Counter(j['arm'] for j in selected),dict.fromkeys(bench.ARMS,8))
                for half in (selected[:8],selected[8:]):
                    for arm in bench.ARMS:
                        self.assertEqual(Counter(j['index']%2 for j in half if j['arm']==arm),{0:2,1:2})
        self.assertEqual({j['arm'] for j in full.jobs(fixture_plan(),'all')},set(bench.ARMS))
        with self.assertRaises(ValueError):bench.schedule(4)

    def test_tuning_contract(self):
        for threads in (64,128,256,512):
            for tile in (32,64):
                bench.validate_tuning(dict(TUNING,decoder_threads=threads,decoder_blocks=1,gemm_tile_k=tile))
        for change in ({'decoder_threads':96},{'decoder_blocks':0},{'decoder_blocks':1025},
                       {'gemm_sms':127},{'gemm_tile_k':16}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                bench.validate_tuning(dict(TUNING,**change))
        self.assertEqual(bench.expected_tuning({'tuning':TUNING},'CR2'),
            dict(decoder_threads=128,decoder_blocks=2048,gemm_sms=-1,gemm_tile_k=32))

    def test_absolute_target_and_increment_are_separate(self):
        good={'T/CR2':dict(percent=1.,ci95=[.2,1.5])}
        result=full.comparison_acceptance(good,{'T':510.,'CR2':515.})
        self.assertTrue(result['increment_over_CR2_validated']);self.assertFalse(result['target_validated'])
        result=full.comparison_acceptance({'T/CR2':dict(percent=-1.,ci95=[-2.,-.2])},{'T':499.,'CR2':495.})
        self.assertTrue(result['absolute_500ms_observed']);self.assertFalse(result['target_validated'])
        passed=full.comparison_acceptance(good,{'T':500.,'CR2':511.})
        self.assertTrue(passed['target_validated'])
        self.assertFalse(full.aggregate_acceptance({'compare1':passed,'compare2':result})['target_validated'])
        self.assertTrue(full.aggregate_acceptance({'compare1':passed})['target_validated'])
        with self.assertRaises(ValueError):full.aggregate_acceptance({})

    def test_generated_sources_parse_and_both_arms_record_native_cache(self):
        for source in (full.worker_source(),full.taco_source(),full.adapter_source()):ast.parse(source)
        source=full.worker_source()
        self.assertIn('H100_MA5_EXECUTE',source)
        self.assertIn('op.codec.cr2_submission_state()',source)
        self.assertIn('ma5_native_tuning',source)
        self.assertNotIn("=='CR2'",source)
        for arm in bench.ARMS:
            states=submission_fixture('attn',[101,102],12)
            bench.validate_submission(states,arm,'attn',[101,102],minimum_calls=1)
            with self.assertRaises(ValueError):bench.validate_submission([],arm,'attn',[101,102])

    def test_operator_profiles_validate_launch_and_true_tile(self):
        for arm in bench.ARMS:
            for tile in (32,64):
                tuning=dict(TUNING,gemm_tile_k=tile)
                result=bench.trace_path(operator_trace(arm,tuning),arm,'attn',tuning=tuning)
                self.assertTrue(result['execution_path_validated'])
                events=operator_trace(arm,tuning)
                next(e for e in events if e['cat']=='kernel' and 'decode' in e['name'])['args']['grid']=[1,1,1]
                with self.assertRaisesRegex(ValueError,'grid/block'):bench.trace_path(events,arm,'attn',tuning=tuning)
        events=operator_trace('T')
        next(e for e in events if e['cat']=='kernel' and 'flux_bf16' in e['name'])['name']=gemm_name(64)
        with self.assertRaisesRegex(ValueError,'K tile'):bench.trace_path(events,'T','attn',tuning=TUNING)

    def test_training_profiles_validate_all_256_projections(self):
        for arm in bench.ARMS:
            result=full.trace_projection_records(projection_trace(arm),arm,tuning=TUNING)
            self.assertEqual(sum(v['projection_calls'] for v in result.values()),256)
            self.assertEqual(sum(v['output_slice_view_count'] for v in result.values()),0)
            events=projection_trace(arm)
            next(e for e in events if e['cat']=='kernel' and 'decode' in e['name'])['args']['block']=[64,1,1]
            with self.assertRaisesRegex(ValueError,'grid/block'):
                full.trace_projection_records(events,arm,tuning=TUNING)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.plan=fixture_plan()
        bench.write(self.root/'plan.json',self.plan);self.digest=bench.sha(self.root/'plan.json')

    def test_both_arms_require_tuning_and_cache_counters(self):
        for arm in bench.ARMS:
            job=dict(self.plan['schedule'][0],arm=arm,mode='timing')
            rows=fixture_rows(self.plan,job,self.digest)
            with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
                self.assertEqual(len(full.window(self.root,self.plan,job)),8)
                rows[0]['ma5_submission_state']={}
                with self.assertRaisesRegex(ValueError,'cached submission'):full.window(self.root,self.plan,job)
            rows=fixture_rows(self.plan,job,self.digest)
            rows[0]['ma5_native_tuning'][next(iter(rows[0]['ma5_native_tuning']))]['decoder_blocks']=1
            with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
                with self.assertRaisesRegex(ValueError,'tuning'):full.window(self.root,self.plan,job)

    def result(self,means=None,mutate=None,trials=1):
        plan=fixture_plan(trials)
        reference=fixture_rows(plan,dict(plan['schedule'][0],mode='timing'),self.digest)
        def window(root,plan,job):
            rows=fixture_rows(plan,job,self.digest,means)
            if mutate:mutate(rows,job)
            return rows
        with patch.object(full,'verify',return_value=plan),patch.object(full,'window',side_effect=window), \
             patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(reference))):
            return full.report(self.root,save=False)

    def test_slower_candidate_is_complete_with_every_window(self):
        result=self.result(means={'T':520.,'CR2':511.})
        self.assertTrue(result['completed']);self.assertFalse(result['target_validated'])
        self.assertEqual(result['formal_optimizer_steps'],800)
        self.assertEqual(result['timing_rank_records'],128)
        self.assertEqual(len(result['all_samples']),16)
        self.assertEqual(result['equivalence_pairs'],[['T','CR2']])
        self.assertEqual(result['comparisons']['compare1']['rounds']['combined']['median_ms']['T'],520.)
        self.assertIn('520.000',full.report_markdown(result))

    def test_two_trials_and_target(self):
        result=self.result(trials=2)
        self.assertTrue(result['target_validated']);self.assertFalse(result['exploratory'])
        self.assertEqual(result['formal_optimizer_steps'],1600)
        self.assertEqual(set(result['comparisons']),{'compare1','compare2'})

    def test_gradient_drift_rejected(self):
        def mutate(rows,job):
            if job['arm']=='T':rows[7]['grad_norms'][0]+=.00001
        with self.assertRaisesRegex(ValueError,'trajectories'):self.result(mutate=mutate)

    def test_missing_rank_rejected(self):
        job=dict(self.plan['schedule'][0],mode='timing')
        rows=fixture_rows(self.plan,job,self.digest)[:7]
        with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
            with self.assertRaises(IndexError):full.window(self.root,self.plan,job)


class PreparationTests(unittest.TestCase):
    def test_frozen_pair_workers_and_one_trial_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);build=base/'build';build.mkdir()
            bench.write(build/'selection-plan.json',cr2_plan()['selection'])
            identities={arm:dict(root='/synthetic/'+arm,files={}) for arm in bench.ARMS}
            provenance=dict(tuning=dict(TUNING))
            micro=base/'operator'
            args=argparse.Namespace(root=micro,build=build,blocks=8,warmup=100,iterations=200,
                                    reuse=2,seeds=[817,919])
            with patch.object(bench,'build_identity',return_value=provenance), \
                 patch.object(bench,'libraries',return_value=identities),contextlib.redirect_stdout(io.StringIO()):
                bench.prepare(args)
            plan=bench.verify(micro,offline=True)
            self.assertEqual(plan['arms'],['T','CR2'])
            self.assertEqual(len(plan['schedule']),16)
            self.assertEqual((micro/'scripts/bench.py').read_bytes(),Path(bench.__file__).read_bytes())
            train=base/'training'
            args=argparse.Namespace(root=train,build=build,megatron=base/'megatron',trials=1,blocks=8)
            with patch.object(bench,'build_identity',return_value=provenance), \
                 patch.object(bench,'libraries',return_value=identities), \
                 patch.object(full,'snapshot_megatron',return_value={'fixture':'CPU-only'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            plan=full.verify(train,offline=True)
            self.assertEqual(plan['trials'],1)
            self.assertEqual(set(plan['trial_ids']),{'1'})
            self.assertEqual(len(plan['schedule']),16)
            for source in ('worker.py','taco_support.py','adapter.py','operator-controller.py'):
                ast.parse((train/'scripts'/source).read_text())
            with (train/'scripts/operator-controller.py').open('a') as handle:handle.write('\n# mutation\n')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):full.verify(train,offline=True)


if __name__=='__main__':unittest.main()
