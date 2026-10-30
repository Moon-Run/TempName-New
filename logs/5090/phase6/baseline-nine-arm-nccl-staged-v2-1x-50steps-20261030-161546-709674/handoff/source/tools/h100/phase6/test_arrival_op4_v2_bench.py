"""Synthetic CPU fixtures only: they cannot become GPU experiment evidence."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import arrival_op4_v2_bench as bench
import arrival_op4_v2_plan as planmod


def state(arm,phase):
    return dict(protocol=bench.SOURCE_PROTOCOL,arm=arm,phase=phase,
        mask_sha256=planmod.mask_sha256(arm,phase),mask_kind='closed' if arm[0]=='C' else 'dispersed',
        reduction='merged' if arm[1]=='M' else 'per_source',inverse_h128_per_rank=1024 if arm=='CM' else 7168,
        selected_per_source=56,selected_per_link=8,wire_bytes_per_rank=13819904)


def events(arm):
    result=[]
    decoder='taco_closure_decode_merged(' if arm[1]=='M' else 'taco_closure_decode_per_source('
    for k in range(4):
        result.append(dict(cat='cpu_op',name='aten::clone'))
        for i,name in enumerate(('flux_bf16_reducescatter','CudaIpcBarrierAllKernel',decoder,'CudaIpcBarrierAllKernel')):
            result.append(dict(cat='kernel',name=name,ts=k*100+i*10,dur=5,args={'stream':7}))
    return result


class ProtocolTests(unittest.TestCase):
    def test_balanced_phases_positions_complete_blocks(self):
        for blocks in (16,32):
            schedule=bench.schedule(blocks)
            self.assertEqual(len(schedule),blocks*4)
            for arm in bench.ARMS:
                rows=[x for x in schedule if x['arm']==arm]
                self.assertEqual([sum(r['phase']==p for r in rows) for p in range(8)],[blocks//8]*8)
                self.assertEqual([sum(r['index']%4==p for r in rows) for p in range(4)],[blocks//4]*4)
        for blocks in (4,8,12,0):
            with self.assertRaises(ValueError): bench.schedule(blocks)

    def test_runtime_state_matches_every_mask(self):
        for arm in bench.ARMS:
            for phase in range(8): bench.validate_state(state(arm,phase),arm,phase)
        bad=state('CM',3); bad['wire_bytes_per_rank']-=1
        with self.assertRaisesRegex(ValueError,'budget'): bench.validate_state(bad,'CM',3)

    def test_all_paths_keep_two_barriers_clone(self):
        for arm in bench.ARMS:
            r=bench.trace_path(events(arm),arm)
            self.assertEqual(r['barrier_count'],8); self.assertEqual(r['clone_count'],4)
            self.assertTrue(r['inverse_count_is_analytic'])
        with self.assertRaisesRegex(ValueError,'path'): bench.trace_path(events('SP'),'CM')
        reduced=events('CM'); reduced.pop(1)
        with self.assertRaisesRegex(ValueError,'path'): bench.trace_path(reduced,'CM')

    def test_phase_environment_and_venv_path(self):
        root=Path('/tmp/closure-preview')
        config=dict(libraries={a:{'root':f'/test/{a}'} for a in bench.ARMS},communication={},shared_environment={})
        env=bench.environment(root,config,'CM',Path('/venv/bin/python'),None,phase=7)
        self.assertEqual(env['H100_CLOSURE_PHASE'],'7')
        self.assertTrue(env['PATH'].startswith('/venv/bin:'))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='closure-synthetic-cpu-'); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        # Four blocks suffice for report unit tests; execution verify only accepts 16+.
        schedule=[]
        for block in range(4):
            for index,arm in enumerate(bench.ARMS):
                schedule.append(dict(block=block,phase=block%2,arm=arm,index=block*4+index,
                                     directory=f'block{block:02d}-{arm}'))
        self.plan=dict(protocol=bench.PROTOCOL,run_id='CPU-ONLY',schedule=schedule,
            quantization_protocol=planmod.protocol_record(),same_mask_sample_l2_budget=.005,
            shared_environment={},communication={},roles=bench.ROLES,seeds=[17,19],reuse=2,
            retained_output_calls=6,slow_rank_stress={'ranks':[0,7],'cycles':2000000},warmup=1,iterations=2)
        bench.write(self.root/'plan.json',self.plan)
        for job in schedule:
            folder=self.root/'results'/job['directory']; folder.mkdir(parents=True)
            arm=job['arm']; phase=job['phase']
            for rank in range(8):
                checks=[]
                correctness=dict(passed=True,budget_vs_bf16=.05,budget_vs_codec=.01,
                                 relative_l2=.01,protocol_relative_l2=.0001)
                for epoch in range(2):
                    for seed in self.plan['seeds']:
                        delayed,cycles=bench.correctness_delay(self.plan,epoch,seed)
                        sample=[1.0 if arm[0]=='S' else .9]*512
                        checks.append(dict(epoch=epoch,seed=seed,delayed_rank=delayed,delay_cycles=cycles,
                            inputs_sha256=f'i{rank}{epoch}{seed}',weights_sha256=f'w{rank}{epoch}{seed}',
                            output_sha256=f'can-differ-{arm}',correctness=correctness,
                            packet_check=dict(passed=True,budget=.005,relative_l2=.001,
                                sample=sample,sample_elements=512,sample_pattern='multiplicative-512-v1',
                                actual_packet_sha256=f'packet-{arm[0]}-{phase}')))
                burst=[dict(generation=g,inputs_sha256=f'i{g}',weights_sha256=f'w{g}',
                            output_sha256=f'o{arm}{g}',correctness=correctness) for g in range(6)]
                row=dict(completed=True,protocol=bench.PROTOCOL,run_id='CPU-ONLY',
                    plan_sha256=bench.sha(self.root/'plan.json'),rank=rank,job=job,profiling=False,sampling=False,
                    execution_environment=dict(H100_CLOSURE_ARM=arm,H100_CLOSURE_PHASE=str(phase),H100_CLOSURE_PROTOCOL=bench.SOURCE_PROTOCOL),
                    hardware={'synthetic':True},software={'synthetic':True},roles={})
                for role,shape in bench.ROLES.items():
                    row['roles'][role]=dict(shape=shape,checks=checks,retained_output_checks=burst,
                        owns_output=False,workspace_count=1,packet_pointers=[123],closure_state=state(arm,phase),
                        timing=dict(iterations=2,warmup=1,wall_us=80 if arm=='CM' else 100))
                bench.write(folder/f'rank{rank}.json',row)
        patcher=patch.object(bench,'verify',return_value=self.plan); patcher.start(); self.addCleanup(patcher.stop)

    def change(self,arm,fn):
        job=next(j for j in self.plan['schedule'] if j['arm']==arm)
        path=self.root/'results'/job['directory']/'rank7.json'; row=bench.read(path)
        fn(row['roles']['attn']); path.write_text(json.dumps(row))

    def test_cross_mask_difference_accepted_and_result_not_training(self):
        result=bench.report(self.root,save=False)
        self.assertTrue(result['precision_control_passed'] and result['budget_matched'])
        self.assertFalse(result['target_validated'])
        self.assertAlmostEqual(result['roles']['attn']['comparisons']['CM/SM']['percent'],20)
        self.assertEqual(result['same_mask_max_sample_relative_l2'],0)

    def test_same_mask_changed_payload_rejected(self):
        self.change('CM',lambda r:r['checks'][0]['packet_check'].update(actual_packet_sha256='changed'))
        with self.assertRaisesRegex(ValueError,'encoded payload'): bench.report(self.root,save=False)

    def test_same_mask_large_output_error_rejected(self):
        self.change('CM',lambda r:r['checks'][0]['packet_check'].update(sample=[.7]*512))
        with self.assertRaisesRegex(ValueError,'sample error'): bench.report(self.root,save=False)

    def test_phase_or_work_drift_rejected(self):
        self.change('CM',lambda r:r['closure_state'].update(inverse_h128_per_rank=7168))
        with self.assertRaisesRegex(ValueError,'budget'): bench.report(self.root,save=False)

    def test_packet_oracle_failure_rejected(self):
        self.change('CM',lambda r:r['checks'][0]['packet_check'].update(relative_l2=.02))
        with self.assertRaisesRegex(ValueError,'oracle'): bench.report(self.root,save=False)

    def test_changed_inputs_rejected(self):
        self.change('CP',lambda r:r['checks'][0].update(inputs_sha256='different'))
        with self.assertRaisesRegex(ValueError,'Inputs'): bench.report(self.root,save=False)

    def test_missing_rank_rejected(self):
        (self.root/'results'/self.plan['schedule'][0]['directory']/'rank0.json').unlink()
        with self.assertRaises(FileNotFoundError): bench.report(self.root,save=False)

    def test_offline_recompute_preserves_report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report=bench.report(self.root)
            self.assertEqual(bench.report(self.root,offline=True),report)
        path=self.root/'report.json'; bad=dict(report,budget_matched=False); path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'Saved operator'): bench.report(self.root,offline=True)
        self.assertEqual(bench.read(path),bad)

    def test_negative_micro_savings_diagnostic_only(self):
        samples=[dict(block=b,arm=a,rankmax_wall_us={r:110 if a=='CM' else 100 for r in bench.ROLES})
                 for b in range(4) for a in bench.ARMS]
        self.assertLess(bench.savings_estimate(samples)['mean_ms'],0)


if __name__=='__main__': unittest.main()
