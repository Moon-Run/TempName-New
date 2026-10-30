"""CPU contracts for the four-arm CR2 host submission experiment; no GPU performance claim."""
import ast
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ma2_cr2_bench as bench
import ma2_cr2_fullstep as full
import ma2_cr2_build as builder
import ma2_cr2_sources as source
import detach_ma2_cr2 as launcher
from common import HERE


def submission_fixture(role, pointers, calls):
    return [dict(slot=i, protocol_version=1, initializations=int(calls > 0),
                 successful_calls=(calls + 1 - i)//2, bound_k=bench.ROLES[role][2],
                 local_packet_ptr=p, failed=False) for i,p in enumerate(pointers)]

def fixture_plan():
    return dict(protocol=full.PROTOCOL, run_id='CPU-only-fixture',
        trial_ids={'1': 'trial1', '2': 'trial2'}, trials=2, blocks=8,
        selection=builder.canonical()[0], model=full.MODEL, roles=bench.ROLES, arms=list(bench.ARMS), schedule=full.expected_schedule(2, 8),
        warmup_steps=10, timed_steps=50, source_files={}, shared_environment=full.SHARED_ENVIRONMENT,
        communication={'NCCL_NVLS_ENABLE': '0', 'NCCL_DEBUG': 'WARN'},
        communication_record={'environment': {'NCCL_NVLS_ENABLE': '0', 'NCCL_DEBUG': 'WARN'},
                              'nvls_request': '0', 'config_files': {}},
        target_percent=full.TARGET_PERCENT, acceptance_rule=full.ACCEPTANCE_RULE,
        libraries={arm: {'files': {
            f'/synthetic/{arm}/{name}': arm+name for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}}
            for arm in bench.ARMS})


def fixture_rows(plan, job, plan_hash):
    arm = job['arm']
    count = 1 if job['mode'] == 'profile' else 3 if job['mode'] == 'smoke' else 50
    shapes = {f'decoder.layers.{layer}.{suffix}': dict(M=2048, N=4096, K_local=k, K_global=k*8)
              for layer in range(32) for suffix, k in (('self_attention.linear_proj', 512), ('mlp.linear_fc2', 2048))}
    modules = list(shapes)
    rows = []
    for rank in range(8):
        row = dict(completed=True, passed=True, rank=rank, local_rank=rank, protocol=full.PROTOCOL,
            policy=full.policy_for_arm(arm), mode=job['mode'], cr2_arm=arm, world_size=8,
            run_id=plan['trial_ids'].get(str(job['trial']), plan['run_id']), cr2_plan_sha256=plan_hash,
            cr2_shared_environment=plan['shared_environment'], cr2_protocol=bench.SOURCE_PROTOCOL,
            tp_rank=rank, dp_rank=0, tp_group_ranks=list(range(8)), dp_group_ranks=[rank],
            block=job['block'], window=job['index'], scope_definition=full.routing.definition(full.policy_for_arm(arm)),
            role_plan_sha256=full.role_hashes(plan['selection']),
            cr2_codec_state={name: dict(owns_output=arm != 'A', owns_configuration=arm != 'A',
                                       workspaces=2 if bench.family(arm)=='selective' else 1) for name in modules},
            workspace_checks={name: [dict(packet_ptr=1000+i, shape=full.routing.SHAPES[full.routing.module_role(name)])]
                              for i, name in enumerate(modules)},
            communication=plan['communication_record'], sampling=False, profiling=job['mode'] == 'profile',
            fallback_calls=0, skips=0, timing_protocol='full-step-one-clear-v2',
            warmup_steps=2 if job['mode'] == 'smoke' else 10, losses=[1.]*count, grad_norms=[.1]*count,
            tokens_per_optimizer_step=8192, padded_vocab_size=33792, target_modules=modules,
            replaced_modules=64, independent_scope_instances=64,
            timed_calls={name: count*4 for name in modules}, post_audit_routes={name: 'candidate' for name in modules},
            warmup_calls={name: (2 if job['mode'] == 'smoke' else 10)*4 for name in modules},
            module_policies={name: 'candidate' for name in modules}, enabled_modules=sorted(modules),
            forward_preserved={name: False for name in modules}, codec_placements={name: 'fused' for name in modules},
            model=dict(name='h100-phase6', **{k: v for k, v in full.MODEL.items() if k != 'sm_count'}),
            observed_shapes=shapes, gpu_uuid=str(rank),
            hardware={'host': 'CPU-fixture', 'local_rank': rank, 'devices': [dict(name='H100 synthetic', uuid=str(i),
                       sm_count=132, capability=[9, 0]) for i in range(8)]},
            torch='synthetic', cuda='synthetic', nccl=[0, 0], python='synthetic',
            initial_parameters_sha256=str(rank), initial_rng_sha256=str(rank), tokens_sha256='synthetic',
            timed_steps=count, elapsed_seconds={'CR2': .46, 'R': .48, 'C': .475, 'CR': .476}[arm]*count,
            forward_checks={name: dict(passed=True, relative_l2=.025, protocol_relative_l2=.0002,
                           budget_vs_bf16=.05, budget_vs_codec=.01) for name in modules})
        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            row[name] = dict(path=f'/synthetic/{arm}/{name}', sha256=arm+name)
        if bench.family(arm)=='selective':
            row['losses']=[2.]*count;row['grad_norms']=[.2]*count
            row['workspace_checks']={name:[dict(packet_ptr=1000+2*i+j,mask_ptr=5000+2*i+j,
                mask_sha256=plan['selection']['variants'][full.C_POLICY]['roles'][full.routing.module_role(name)]['mask_sha256'],
                shape=full.routing.SHAPES[full.routing.module_role(name)]) for j in range(2)] for i,name in enumerate(modules)}
        row['cr2_submission_state'] = {name: submission_fixture(full.routing.module_role(name),
            [s['packet_ptr'] for s in slots], (row['warmup_steps']+count)*4)
            for name,slots in row['workspace_checks'].items()} if arm == 'CR2' else {}
        rows.append(row)
    return rows

def trace_fixture(arm):
    events = []
    decoder = {'CR2': 'taco_ma2c_decode_kernel(X)', 'C': 'taco_decode_ring_kernel<false>(X)', 'CR':'taco_ma2c_decode_kernel(X)',
               'R': 'taco_ma2_decode_kernel(X)'}[arm]
    for role_index, role in enumerate(('attn', 'mlp')):
        for index in range(128):
            external = role_index*128+index
            start = external*20
            events += [dict(cat='user_annotation', name=f'scope.{role}.candidate', tid=1, ts=start, dur=19),
                       dict(cat='cpu_op', name='GemmRSFunction', tid=1, ts=start+1, dur=17,
                            args={'External id': external})]
            if arm == 'A':
                events.append(dict(cat='cpu_op', name='aten::clone', tid=1, ts=start+12, dur=1))
            for offset, name in ((2, 'flux_bf16_reducescatter'), (4, 'CudaIpcBarrierAllKernel'),
                                 (6, decoder), (8, 'CudaIpcBarrierAllKernel')):
                if offset==8 and arm in ('CR2','CR','C'): continue
                events.append(dict(cat='kernel', name=name, ts=start+offset, dur=1,
                                   args={'External id': external, 'stream': 7}))
    return events


class ProtocolTests(unittest.TestCase):
    def test_balanced_two_trials(self):
        jobs=full.expected_schedule(2,8)
        self.assertEqual(len(jobs),64)
        for trial in (1,2):
            selected=[j for j in jobs if j['trial']==trial]
            self.assertEqual(Counter(j['arm'] for j in selected),{a:8 for a in bench.ARMS})
            for arm in bench.ARMS:
                self.assertEqual(Counter(j['index']%4 for j in selected if j['arm']==arm),{i:2 for i in range(4)})
        self.assertEqual(launcher.FORMAL_STEPS,3200)
        with self.assertRaises(ValueError):bench.schedule(12)

    def test_selection_rejects_new_mask_or_arrival(self):
        plan=builder.canonical()[0]
        bench.verify_selection(plan)
        self.assertEqual([sum(map(sum,plan['variants'][builder.C_POLICY]['roles'][r]['entry']['mask'])) for r in ('attn','mlp')],[53,50])
        for key in ('mask','coords'):
            changed=copy.deepcopy(plan)
            changed['variants'][builder.C_POLICY]['roles']['attn']['entry'][key][0][0]^=1
            with self.assertRaises(ValueError):bench.verify_selection(changed)

    def test_target_requires_one_percent_with_positive_confidence(self):
        estimates={a+'/'+b:dict(percent=.5,ci95=[.1,.9]) for a,b in bench.COMPARISONS}
        self.assertFalse(full.comparison_acceptance(estimates)['target_validated'])
        for k in ('CR2/C','CR2/R'):
            good=copy.deepcopy(estimates);good[k]=dict(percent=1.,ci95=[.1,1.2])
            self.assertTrue(full.comparison_acceptance(good)['target_validated'])
            good[k]['ci95'][0]=0
            self.assertFalse(full.comparison_acceptance(good)['target_validated'])

    def test_two_trials_must_beat_the_same_control(self):
        estimates={a+'/'+b:dict(percent=.5,ci95=[.1,.9]) for a,b in bench.COMPARISONS}
        comparisons={}
        for key in ('CR2/C','CR2/R'):
            current=copy.deepcopy(estimates);current[key]=dict(percent=1.5,ci95=[1.1,1.8])
            comparisons[key]=full.comparison_acceptance(current)
            self.assertTrue(comparisons[key]['target_validated'])
        result=full.aggregate_acceptance(comparisons)
        self.assertFalse(result['target_validated'])
        self.assertFalse(result['strict_one_percent_confidence_validated'])
        result=full.aggregate_acceptance({'one':comparisons['CR2/C'],'two':comparisons['CR2/C']})
        self.assertTrue(result['target_validated'])
        self.assertFalse(result['beats_both_one_percent_validated'])

    def test_generated_workers_have_correct_ownership_and_no_grad_adapter(self):
        for text in (full.worker_source(),full.taco_source(),full.adapter_source()): ast.parse(text)
        self.assertIn('GemmRSTacoDoubleBuffered',full.taco_source())
        self.assertIn("self.owns_configuration = self.codec.owns_configuration",full.taco_source())
        self.assertIn('H100_CR2_EXECUTE',full.worker_source())
        self.assertNotIn('H100_MA2R_',full.worker_source())

    def test_launcher_preview_is_fresh_four_arm_no_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'untouched'
            args=launcher.arguments(['--out',str(root)])
            tasks=launcher.commands(args)
            self.assertEqual(len(tasks),8)
            command=next(c for label,c in tasks if label=='fullstep-prepare')
            self.assertEqual(command[command.index('--trials')+1],'2')
            self.assertEqual(command[command.index('--blocks')+1],'8')
            self.assertFalse(root.exists())

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=fixture_plan()
        bench.write(self.root/'plan.json',self.plan);self.hash=bench.sha(self.root/'plan.json')

    def test_all_four_workspace_layouts_validate_and_masks_are_enforced(self):
        for arm in bench.ARMS:
            job=dict(self.plan['schedule'][0],arm=arm,mode='timing')
            rows=fixture_rows(self.plan,job,self.hash)
            with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
                self.assertEqual(len(full.window(self.root,self.plan,job)),8)
                if arm in ('CR2','CR','C'):
                    first=next(iter(rows[0]['workspace_checks'].values()))[0]
                    first['mask_sha256']='bad'
                    with self.assertRaisesRegex(ValueError,'mask'):full.window(self.root,self.plan,job)

    def result(self, transform=None):
        ref=fixture_rows(self.plan,dict(self.plan['schedule'][0],mode='timing'),self.hash)
        def window(root,plan,job):
            rows=fixture_rows(plan,job,self.hash)
            if transform:transform(rows,job)
            return rows
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))),patch.object(full,'window',side_effect=window):
            return full.report(self.root,save=False)

    def test_two_numerical_families_are_valid_but_not_cross_family_equal(self):
        result=self.result()
        self.assertTrue(result['target_validated'])
        self.assertTrue(result['identical_trajectories_within_pairs'])
        self.assertNotIn('identical_trajectories',result)
        self.assertEqual(result['formal_optimizer_steps'],3200)
        self.assertEqual(result['timing_rank_records'],512)
        self.assertIn('CR2/C',full.report_markdown(result))

    def test_pair_gradient_drift_is_rejected(self):
        def change(rows,job):
            if job['arm']=='CR':rows[7]['grad_norms'][0]+=.00001
        with self.assertRaisesRegex(ValueError,'trajectories'):self.result(change)

    def test_second_trial_cannot_be_hidden_by_first(self):
        def no_increment(rows,job):
            if job['trial']==2 and job['arm']=='CR2':
                for row in rows:row['elapsed_seconds']=.48*50
        result=self.result(no_increment)
        self.assertTrue(result['comparisons']['compare1']['target_validated'])
        self.assertFalse(result['comparisons']['compare2']['target_validated'])
        self.assertFalse(result['target_validated'])

    def test_loss_summary_uses_all_fifty_formal_steps(self):
        def change(rows,job):
            for row in rows:row['losses']=[10.]*10+[1.]*40
        result=self.result(change)
        self.assertEqual(result['numeric_summary']['CR']['mean_loss'],2.8)
        self.assertEqual(result['numeric_summary']['CR']['final_loss'],1.)

    def test_saved_preflight_checks_equivalence_within_families(self):
        for job in full.jobs(self.plan,'preflight'):
            folder=self.root/'results'/job['directory'];folder.mkdir(parents=True)
            for rank,row in enumerate(fixture_rows(self.plan,job,self.hash)):bench.write(folder/f'rank{rank}.json',row)
        with patch.object(full,'verify',return_value=self.plan):
            self.assertTrue(full.preflight(self.root)['identical_trajectories_within_pairs'])

    def test_fullstep_rejects_counter_drift_wrong_k_rebind_and_pointer(self):
        job=dict(self.plan['schedule'][0],arm='CR2',mode='timing')
        for key,value in [('successful_calls',0),('bound_k',2048),('initializations',2),('local_packet_ptr',1),('failed',True)]:
            rows=fixture_rows(self.plan,job,self.hash)
            rows[0]['cr2_submission_state']['decoder.layers.0.self_attention.linear_proj'][0][key]=value
            with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
                with self.assertRaises(ValueError): full.window(self.root,self.plan,job)

class ProfileTests(unittest.TestCase):
    def test_four_arm_kernel_barrier_clone_paths(self):
        for arm in bench.ARMS:
            r=full.trace_projection_records(trace_fixture(arm),arm)
            self.assertEqual(r['attn']['output_clones'],128*bench.clone_per_call(arm))
            self.assertEqual(r['mlp']['barrier_calls'],128*bench.barriers_per_call(arm))

    def test_wrong_kernel_extra_clone_or_missing_barrier_is_rejected(self):
        for source_arm,target in [('C','CR2'),('R','CR'),('CR','R')]:
            with self.assertRaises(ValueError):full.trace_projection_records(trace_fixture(source_arm),target)
        t=trace_fixture('CR');t.append(dict(cat='cpu_op',name='aten::clone',tid=1,ts=12,dur=1))
        with self.assertRaises(ValueError):full.trace_projection_records(t,'CR')

    def test_cr2_profile_rejects_legacy_output_views(self):
        for name in ('aten::slice','aten::view'):
            t=trace_fixture('CR2');t.append(dict(cat='cpu_op',name=name,tid=1,ts=12,dur=1))
            with self.assertRaisesRegex(ValueError,'slice/view'):full.trace_projection_records(t,'CR2')
            self.assertEqual(full.trace_projection_records(t,'CR')['attn']['output_slice_view_count'],1)

    def test_cr2_profile_keeps_common_autograd_reshape_views(self):
        t=trace_fixture('CR2')
        for call in [e for e in t if e.get('name')=='GemmRSFunction']:
            for offset in (0,14):
                t += [dict(cat='cpu_op',name='aten::reshape',tid=1,ts=call['ts']+offset,dur=2),
                      dict(cat='cpu_op',name='aten::view',tid=1,ts=call['ts']+offset+.5,dur=1)]
        summary=full.trace_projection_records(t,'CR2')
        self.assertEqual(summary['attn']['adapter_view_count'],256)
        self.assertEqual(summary['attn']['output_slice_view_count'],0)
        # A native view outside either reshape is still forbidden.
        t.append(dict(cat='cpu_op',name='aten::view',tid=1,ts=8,dur=1))
        with self.assertRaisesRegex(ValueError,'slice/view'):full.trace_projection_records(t,'CR2')

    def test_gpu_mirrors_do_not_duplicate_cpu_scopes(self):
        t=trace_fixture('CR')
        t += [dict(e,cat='gpu_user_annotation') for e in t if e['cat']=='user_annotation']
        self.assertEqual(full.trace_projection_records(t,'CR')['attn']['projection_calls'],128)

class ReceiverTests(unittest.TestCase):
    def test_physical_mask_index_covers_every_destination_and_source(self):
        plan=builder.canonical()[0]
        for role in ('attn','mlp'):
            mask=plan['variants'][builder.C_POLICY]['roles'][role]['entry']['mask']
            for rank in range(8):
                for group in range(8192):
                    row=group//32
                    index=(rank*2+row//128)*32+group%32
                    scalar=((rank*256+row)//128)*32+group%32
                    self.assertEqual(index,scalar)
                    self.assertFalse(mask[rank][index])

    def test_source_pipeline_never_reads_unselected_fp8_slots(self):
        for rank in range(8):
            for bits in range(256):
                selected=[bool(bits&(1<<i)) and i!=rank for i in range(8)]
                source_order=[(rank+i)&7 for i in range(1,9)]
                reads=[('fp8' if selected[src] else 'bf16',src) for src in source_order]
                self.assertEqual(reads[-1],('bf16',rank))
                self.assertEqual(len(set(src for _,src in reads)),8)
        header=(HERE/'ma2_c_decode.cuh').read_text()
        self.assertIn('if (quantized)',header)
        self.assertIn('if (!any_quant)',header)
        self.assertIn('taco_ma2::fold',header)
        self.assertNotIn('__syncthreads',header)


class OperatorTests(unittest.TestCase):
    def fixture(self):
        import test_ma2r_bench as old
        # The old helper produces raw records; replace metadata to model the
        # two numerical families independently of production grouping logic.
        plan=old.fixture_plan()
        plan.update(protocol=bench.PROTOCOL,arms=list(bench.ARMS),arm_labels=bench.LABELS,
            selection=builder.canonical()[0],schedule=bench.schedule(8),
            libraries={a:dict(root='/synthetic/'+a,files={'/synthetic/'+a+'/libflux_cuda.so':a}) for a in bench.ARMS})
        records={}
        for job in bench.stage_jobs('preflight'):
            for rank in range(8):
                old_job=dict(job,arm='R')
                old_plan=old.fixture_plan()
                row=old.fixture_row(old_plan,old_job,rank)
                row.update(protocol=bench.PROTOCOL,job=job,run_id=plan['run_id'],
                    execution_environment=bench.execution_environment(plan,job['arm']),
                    loaded_libraries=plan['libraries'][job['arm']]['files'])
                selective=job['arm'] in ('CR2','CR','C')
                for role,data in row['roles'].items():
                    rec=plan['selection']['variants'][builder.C_POLICY]['roles'][role]
                    data['runtime_state']=dict(arm=job['arm'],workspace_count=2 if selective else 1,
                        output_clone=job['arm']=='A',owns_configuration=job['arm']!='A',placement='fused',
                        mask_sha256=rec['mask_sha256'] if selective else None,
                        wire_bytes_per_rank=14680064-15360*sum(rec['entry']['mask'][rank]) if selective else 7798784,
                        protocol_version=int(job['arm']!='A'))
                    data['packet_pointers']=[1000,2000] if selective else [1000]
                    calls=2*len(bench.check_indices(plan))+plan['retained_output_calls']+plan['instance_isolation_calls']//2
                    data['submission_before']=submission_fixture(role,data['packet_pointers'],0) if job['arm']=='CR2' else []
                    data['submission_after']=submission_fixture(role,data['packet_pointers'],calls) if job['arm']=='CR2' else []
                    for c in data['checks']+data['retained_output_checks']+data['instance_isolation_checks']:
                        c['output_sha256']=('c' if selective else 'f')*64
                records[f"{job['directory']}/rank{rank}"]=row
        return plan,records

    def validate(self,plan,records):
        with patch.object(bench,'read',side_effect=lambda path:records[str(path.parent.name)+'/'+path.stem]), \
             patch.object(bench,'sha',return_value='1'*64):
            return bench.validate_rows(Path('/unused'),plan,bench.stage_jobs('preflight'),'preflight')

    def test_operator_families_allow_distinct_quantization(self):
        plan,records=self.fixture();self.validate(plan,records)

    def test_operator_rejects_changed_output_inside_equivalent_pair(self):
        for arm in ('C','CR'):
            plan,records=self.fixture()
            records[f'preflight-{arm}/rank7']['roles']['mlp']['checks'][0]['output_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'bitwise'):self.validate(plan,records)

    def test_operator_inputs_must_match_across_all_four_arms(self):
        plan,records=self.fixture()
        records['preflight-R/rank0']['roles']['attn']['checks'][0]['inputs_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'inputs or weights'):self.validate(plan,records)

    def test_operator_rejects_mask_drift_and_workspace_swap(self):
        for field,value in [('mask_sha256','bad'),('workspace_count',1),('output_clone',True)]:
            plan,records=self.fixture()
            records['preflight-CR/rank0']['roles']['attn']['runtime_state'][field]=value
            with self.assertRaisesRegex(ValueError,'Runtime lifecycle'):self.validate(plan,records)

    def test_operator_requires_actual_cached_submissions_and_fixed_slots(self):
        for field,value in [('successful_calls',0),('initializations',2),('local_packet_ptr',7),('slot',1),('failed',True)]:
            plan,records=self.fixture()
            records['preflight-CR2/rank0']['roles']['attn']['submission_after'][0][field]=value
            with self.assertRaises(ValueError):self.validate(plan,records)

    def test_operator_rejects_missing_or_extra_successful_calls(self):
        for delta in (-2,2):
            plan,records=self.fixture()
            state=records['preflight-CR2/rank0']['roles']['attn']['submission_after']
            for item in state:item['successful_calls']+=delta//2
            with self.assertRaisesRegex(ValueError,'executed calls'):self.validate(plan,records)

    def test_operator_profiles_match_actual_single_vs_double_workspace_paths(self):
        import test_ma2r_bench as old
        decoders={'C':'void taco_decode_ring_kernel<false>(X)','CR':'taco_ma2c_decode_kernel(X)',
                  'CR2':'taco_ma2c_decode_kernel(X)'}
        for arm in bench.ARMS:
            t=old.trace_fixture('A' if arm=='A' else 'R')
            for e in t:
                e['name']=e.get('name','').replace('ma2r.','cr2.')
                if arm in decoders and 'taco_ma2_decode_kernel' in e['name']:e['name']=decoders[arm]
            if arm in decoders:
                barriers=[e for e in t if e.get('cat')=='kernel' and 'CudaIpcBarrierAllKernel' in e['name']]
                skip={e['args']['correlation'] for e in barriers[1::2]}
                t=[e for e in t if e.get('args',{}).get('correlation') not in skip]
            result=bench.trace_path(t,arm,'attn')
            self.assertEqual(result['barrier_count'],4*bench.barriers_per_call(arm))
            self.assertEqual(result['clone_count'],4*bench.clone_per_call(arm))

class MixedNumericsTests(unittest.TestCase):
    def test_all_source_masks_preserve_per_source_bf16_rounding(self):
        from test_ma2_decode import packed_decode,pack_payload,packed_local,unpack_local
        from test_arrival_op3_v2_decode import decode,warp_stages,bf16,bits,f32
        rng=random.Random(20261023)
        payloads=[[rng.randrange(256) for _ in range(128)] for _ in range(8)]
        raw=[(pack_payload(v),f32(.013*(i+1)),f32(.7*(i+1))) for i,v in enumerate(payloads)]
        bf=[[bf16(rng.uniform(-2,2)) for _ in range(128)] for _ in range(8)]
        original=[decode(v,raw[i][1],raw[i][2],warp_stages) for i,v in enumerate(payloads)]
        packed=[packed_decode(packet) for packet in raw]
        for rank in range(8):
            for mask in range(256):
                a=[0.]*128;b=[0.]*128
                for src in [(rank+i)&7 for i in range(1,9)]:
                    quant=src!=rank and bool(mask&(1<<src))
                    old=original[src] if quant else bf[src]
                    new=packed[src] if quant else unpack_local(packed_local(bf[src]))
                    a=[bf16(f32(x+bf16(y))) for x,y in zip(a,old)]
                    b=[bf16(f32(x+bf16(y))) for x,y in zip(b,new)]
                self.assertEqual(list(map(bits,a)),list(map(bits,b)))

if __name__=='__main__':unittest.main()
