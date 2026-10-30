"""CPU contracts for the four-arm G host submission experiment; no GPU performance claim."""
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

import ma4_g_bench as bench
import ma4_g_fullstep as full
import ma4_g_build as builder
import ma4_g_sources as source
import detach_ma4_g as launcher
from common import HERE


def cr2_fixture(role, pointers, calls):
    return [dict(slot=i, protocol_version=1, initializations=int(calls > 0),
                 successful_calls=(calls + 1 - i)//2, bound_k=bench.ROLES[role][2],
                 local_packet_ptr=p, failed=False) for i,p in enumerate(pointers)]

def submission_fixture(role, pointers, calls, arm='G'):
    if arm == 'CR2':
        return cr2_fixture(role, pointers, calls)
    return [dict(slot=i,protocol_version=1,graph_build_count=1,graph_launch_count=(calls+1-i)//2,
        gemm_param_update_count=(calls+1-i)//2,decoder_param_update_count=(calls+1-i)//2,
        node_count=4,kernel_node_count=3,memset_node_count=1,bound_k=bench.ROLES[role][2],
        local_packet_ptr=pointer,failed=False,workspace_ptr=pointer*20,device_index=0,stream_id=0,graph_build_host_ns=1000,
        **{key:pointer*10+j for j,key in enumerate(bench.GRAPH_IDENTITIES)}) for i,pointer in enumerate(pointers)]


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
            policy=full.policy_for_arm(arm), mode=job['mode'], ma4_g_arm=arm, world_size=8,
            run_id=plan['trial_ids'].get(str(job['trial']), plan['run_id']), ma4_g_plan_sha256=plan_hash,
            ma4_g_shared_environment=plan['shared_environment'], ma4_g_protocol=bench.SOURCE_PROTOCOL,
            tp_rank=rank, dp_rank=0, tp_group_ranks=list(range(8)), dp_group_ranks=[rank],
            block=job['block'], window=job['index'], scope_definition=full.routing.definition(full.policy_for_arm(arm)),
            role_plan_sha256=full.role_hashes(plan['selection']),
            ma4_g_codec_state={name: dict(owns_output=arm != 'A', owns_configuration=arm != 'A',
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
            timed_steps=count, elapsed_seconds={'G': .46, 'R': .48, 'C': .475, 'CR2': .470}[arm]*count,
            forward_checks={name: dict(passed=True, relative_l2=.025, protocol_relative_l2=.0002,
                           budget_vs_bf16=.05, budget_vs_codec=.01) for name in modules})
        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            row[name] = dict(path=f'/synthetic/{arm}/{name}', sha256=arm+name)
        if bench.family(arm)=='selective':
            row['losses']=[2.]*count;row['grad_norms']=[.2]*count
            row['workspace_checks']={name:[dict(packet_ptr=1000+2*i+j,mask_ptr=5000+2*i+j,
                mask_sha256=plan['selection']['variants'][full.C_POLICY]['roles'][full.routing.module_role(name)]['mask_sha256'],
                shape=full.routing.SHAPES[full.routing.module_role(name)]) for j in range(2)] for i,name in enumerate(modules)}
        row['ma4_g_submission_state'] = {name: submission_fixture(full.routing.module_role(name),
            [s['packet_ptr'] for s in slots], (row['warmup_steps']+count)*4, arm)
            for name,slots in row['workspace_checks'].items()} if arm in ('G','CR2') else {}
        row['ma4_g_first_warmup_seconds']=1.
        row['ma4_g_final_parameters_sha256']=('c' if bench.family(arm)=='selective' else 'f')*64
        row['ma4_g_preparation']=dict(modules=sorted(modules),graph_exec_count=128,host_seconds=1.,
            before={name:submission_fixture(full.routing.module_role(name),[v['packet_ptr'] for v in slots],0)
                    for name,slots in row['workspace_checks'].items()}) if arm=='G' else None
        if arm=='G':
            for mapping in (row['ma4_g_submission_state'],row['ma4_g_preparation']['before']):
                for slots in mapping.values():
                    for slot in slots:slot['device_index']=rank
        rows.append(row)
    return rows

def trace_fixture(arm):
    events = []
    decoder = {'G': 'taco_ma2c_decode_kernel(X)', 'C': 'taco_decode_ring_kernel<false>(X)', 'CR2':'taco_ma2c_decode_kernel(X)',
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
                if offset==8 and arm in ('G','CR2','C'): continue
                events.append(dict(cat='kernel', name=name, ts=start+offset, dur=1,
                                   args={'External id': external, 'stream': 7}))
    return events


class ProtocolTests(unittest.TestCase):
    def test_balanced_four_arm_two_trial_schedule(self):
        jobs=full.expected_schedule(2,8)
        self.assertEqual(len(jobs),64)
        for trial in (1,2):
            selected=[j for j in jobs if j['trial']==trial]
            self.assertEqual(Counter(j['arm'] for j in selected),dict.fromkeys(bench.ARMS,8))
            for arm in bench.ARMS:
                self.assertEqual(Counter(j['index']%4 for j in selected if j['arm']==arm),dict.fromkeys(range(4),2))
        self.assertEqual(launcher.FORMAL_STEPS,3200)
        self.assertEqual(bench.COMPARISONS,(('G','C'),('G','R'),('G','CR2'),('CR2','C'),('CR2','R'),('R','C')))

    def test_two_percent_requires_both_controls_and_reports_increment_separately(self):
        estimates={a+'/'+b:dict(percent=2.5,ci95=[.1,3.]) for a,b in bench.COMPARISONS}
        self.assertTrue(full.comparison_acceptance(estimates)['target_validated'])
        for key in ('G/C','G/R'):
            changed=copy.deepcopy(estimates);changed[key]['percent']=1.99
            self.assertFalse(full.comparison_acceptance(changed)['target_validated'])
            changed=copy.deepcopy(estimates);changed[key]['ci95'][0]=0.
            self.assertFalse(full.comparison_acceptance(changed)['target_validated'])
        changed=copy.deepcopy(estimates);changed['G/CR2']['ci95'][0]=-.1
        self.assertTrue(full.comparison_acceptance(changed)['target_validated'])
        self.assertFalse(full.comparison_acceptance(changed)['increment_over_CR2_validated'])

    def test_trials_cannot_switch_control_or_hide_negative_increment(self):
        estimates={a+'/'+b:dict(percent=2.5,ci95=[2.1,3.]) for a,b in bench.COMPARISONS}
        first=full.comparison_acceptance(estimates)
        estimates['G/R']['percent']=1.
        second=full.comparison_acceptance(estimates)
        result=full.aggregate_acceptance({'first':first,'second':second})
        self.assertFalse(result['target_validated'])
        self.assertTrue(result['two_percent_over_C_validated'])
        self.assertFalse(result['two_percent_over_R_validated'])

    def test_generated_workers_prepare_before_warmup_and_keep_controls(self):
        for source in (full.worker_source(),full.adapter_source(),full.taco_source()):ast.parse(source)
        worker=full.worker_source(); adapter=full.adapter_source()
        self.assertLess(worker.index('adapter.prepare_graphs()'),worker.index('for _ in range(warmup)'))
        self.assertIn('op.codec.cr2_submission_state()',worker)
        self.assertIn("report['ma4_g_final_parameters_sha256'] = fingerprint(raw.named_parameters())",worker)
        self.assertIn('adapter.prepared_ops.get(name) or TacoForward(',adapter)
        self.assertIn("order = sorted(self.modules)",adapter)
        self.assertIn('torch.distributed.barrier(group=group)',adapter)

    def test_read_only_preview_and_new_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'
            args=launcher.arguments(['--out',str(out)])
            commands=launcher.commands(args)
            self.assertEqual(len(commands),8)
            self.assertTrue(all('ma4_g_' in ' '.join(command) for _,command in commands))
            self.assertFalse(out.exists())


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=fixture_plan()
        bench.write(self.root/'plan.json',self.plan);self.hash=bench.sha(self.root/'plan.json')

    def test_all_arms_validate_and_graph_counters_fail_closed(self):
        for arm in bench.ARMS:
            job=dict(self.plan['schedule'][0],arm=arm,mode='timing')
            rows=fixture_rows(self.plan,job,self.hash)
            with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
                self.assertEqual(len(full.window(self.root,self.plan,job)),8)
        job=dict(self.plan['schedule'][0],arm='G',mode='timing')
        for field,value in [('graph_launch_count',0),('graph_build_count',2),('gemm_param_update_count',0),
                            ('decoder_param_update_count',0),('node_count',3),('bound_k',2048),('failed',True)]:
            rows=fixture_rows(self.plan,job,self.hash)
            next(iter(rows[0]['ma4_g_submission_state'].values()))[0][field]=value
            with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
                with self.assertRaises(ValueError):full.window(self.root,self.plan,job)

    def test_graph_preparation_cannot_advance_slot_or_share_resources(self):
        job=dict(self.plan['schedule'][0],arm='G',mode='timing')
        for changed_field in ('graph_exec_count','before'):
            rows=fixture_rows(self.plan,job,self.hash)
            if changed_field=='before':next(iter(rows[0]['ma4_g_preparation']['before'].values()))[0]['graph_launch_count']=1
            else:rows[0]['ma4_g_preparation']['graph_exec_count']=64
            with patch.object(full,'read',side_effect=lambda path:rows[int(path.stem[4:])]):
                with self.assertRaises(ValueError):full.window(self.root,self.plan,job)

    def result(self,transform=None):
        ref=fixture_rows(self.plan,dict(self.plan['schedule'][0],mode='timing'),self.hash)
        def window(root,plan,job):
            rows=fixture_rows(plan,job,self.hash)
            if transform:transform(rows,job)
            return rows
        with patch.object(full,'verify',return_value=self.plan),patch.object(full,'preflight',return_value=dict(files={},rank_identities=full.identity(ref))),patch.object(full,'window',side_effect=window):
            return full.report(self.root,save=False)

    def test_six_comparisons_and_all_windows_preserved_on_slow_graph(self):
        def slow(rows,job):
            if job['arm']=='G':
                for row in rows:row['elapsed_seconds']=.6*50
        result=self.result(slow)
        self.assertFalse(result['target_validated'])
        self.assertEqual(len(result['all_samples']),64)
        self.assertEqual(result['formal_optimizer_steps'],3200)
        self.assertEqual(result['timing_rank_records'],512)
        for value in result['comparisons'].values():self.assertEqual(len(value['rounds']['combined']['comparisons']),6)
        self.assertIn('G/CR2',full.report_markdown(result))

    def test_final_parameter_drift_is_rejected(self):
        def drift(rows,job):
            if job['arm']=='G':rows[0]['ma4_g_final_parameters_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'trajectories'):self.result(drift)


def graph_trace(full_model=True, graph_only_ids=False):
    events=trace_fixture('G')
    calls=[e for e in events if e.get('name')=='GemmRSFunction']
    for call in calls:
        eid=call['args']['External id']; start=call['ts']
        # CPU updates precede one replay; GPU nodes need no individual launch.
        for offset,name in ((0,'cudaGraphExecKernelNodeSetParams'),(.2,'cudaGraphExecKernelNodeSetParams'),(.4,'cudaGraphLaunch')):
            args={'correlation':10000+eid,'graph id':eid//2}
            events.append(dict(cat='cuda_runtime',name=name,ts=start+offset,dur=.1,tid=1,args=args))
        kernels=[e for e in events if e.get('cat')=='kernel' and e['args'].get('External id')==eid]
        reset=dict(cat='gpu_memset',name='Memset (Device)',ts=start+.8,dur=.1,args={'stream':7})
        for index,node in enumerate([reset,*kernels]):
            node['args'].update({'correlation':10000+eid,'graph id':eid//2,'graph node id':index+1})
            if graph_only_ids:
                node['args'].pop('External id',None);node['args'].pop('correlation',None)
        events.append(reset)
    if not full_model:
        selected=calls[:4]; ends=selected[-1]['ts']+selected[-1]['dur']
        events=[e for e in events if e['ts']<=ends]
        for i,call in enumerate(selected):
            events.append(dict(cat='user_annotation',name=f'ma4g.attn.{i}',ph='X',tid=1,ts=call['ts'],dur=call['dur'],
                args={'External id':call['args']['External id']}))
    return events


def graph_trace_kineto26(*, full_model=True, association='correlation'):
    # PyTorch v2.6.0 pins Kineto 338140f58a28d599da3434ced4fd2d75dd1a213d:
    # kernel/memset export device/context/stream/correlation, without graph IDs.
    # These are schema fixtures, not evidence of an H100 capture/replay run.
    events = graph_trace(full_model=full_model)
    for event in events:
        args = event.get('args', {})
        args.pop('graph id', None)
        args.pop('graph node id', None)
        if event.get('cat') in ('kernel','gpu_memset'):
            args.update(device=0, context=1)
            if association == 'correlation':
                args.pop('External id', None)
            elif association == 'external':
                args['External id'] = args.pop('correlation') - 10000
            elif association == 'missing':
                args.pop('External id', None)
                args.pop('correlation', None)
            else:
                raise ValueError('Unknown fixture association')
    return events


class ProfileTests(unittest.TestCase):
    def test_graph_launch_correlates_multiple_nodes_without_individual_launches(self):
        for only_ids in (False,True):
            result=full.trace_projection_records(graph_trace(graph_only_ids=only_ids),'G')
            self.assertEqual(result['attn']['graph_launch_count'],128)
            self.assertEqual(result['mlp']['workspace_reset_count'],128)
            self.assertTrue(result['attn']['graph_node_association_validated'])
            operator=bench.trace_path(graph_trace(full_model=False,graph_only_ids=only_ids),'G','attn')
            self.assertEqual(operator['graph_launch_count'],4)

    def test_kineto26_without_graph_metadata_accepts_correlation_or_external_id(self):
        for association in ('correlation','external'):
            with self.subTest(association=association):
                events=graph_trace_kineto26(association=association)
                self.assertTrue(all('graph id' not in e.get('args',{}) and
                                    'graph node id' not in e.get('args',{}) for e in events))
                result=full.trace_projection_records(events,'G')
                self.assertEqual(result['attn']['graph_launch_count'],128)
                self.assertEqual(result['mlp']['workspace_reset_count'],128)
                operator=bench.trace_path(graph_trace_kineto26(full_model=False,association=association),'G','attn')
                self.assertEqual(operator['graph_launch_count'],4)

    def test_kineto26_missing_all_associations_is_rejected(self):
        with self.assertRaises(ValueError):
            full.trace_projection_records(graph_trace_kineto26(association='missing'),'G')
        with self.assertRaises(ValueError):
            bench.trace_path(graph_trace_kineto26(full_model=False,association='missing'),'G','attn')

    def test_graph_missing_reset_eager_fallback_or_param_update_rejected(self):
        for remove in ('gpu_memset','cudaGraphExecKernelNodeSetParams'):
            events=graph_trace()
            index=next(i for i,e in enumerate(events) if e.get('cat')==remove or e.get('name')==remove)
            events.pop(index)
            with self.assertRaises(ValueError):full.trace_projection_records(events,'G')
        events=graph_trace();first=next(e for e in events if e.get('name')=='cudaGraphLaunch')
        events.append(dict(first,name='cudaLaunchKernel'))
        with self.assertRaisesRegex(ValueError,'eager'):full.trace_projection_records(events,'G')

    def test_controls_keep_their_original_eager_paths(self):
        for arm in ('CR2','C','R'):
            result=full.trace_projection_records(trace_fixture(arm),arm)
            self.assertEqual(result['attn']['graph_launch_count'],0)


class OperatorEvidenceTests(unittest.TestCase):
    def fixture(self):
        import test_ma2_cr2 as old
        old_plan, old_records=old.OperatorTests().fixture()
        def convert(value):
            if isinstance(value,dict):return {convert(k):convert(v) for k,v in value.items()}
            if isinstance(value,list):return [convert(v) for v in value]
            if isinstance(value,str):
                if value in ('CR2','CR'):return {'CR2':'G','CR':'CR2'}[value]
                return value.replace('ma2-cr2','ma4-g').replace('H100_CR2_','H100_MA4_G_').replace('preflight-CR2','preflight-__G__').replace('preflight-CR','preflight-CR2').replace('preflight-__G__','preflight-G').replace('/CR2/','/__G__/').replace('/CR/','/CR2/').replace('/__G__/','/G/').replace('/CR2','/__G__').replace('/CR','/CR2').replace('/__G__','/G')
            return value
        plan,records=convert(old_plan),convert(old_records)
        plan.update(protocol=bench.PROTOCOL,arms=list(bench.ARMS),arm_labels=bench.LABELS,schedule=bench.schedule(8),inplace_update_calls=4,
            libraries={a:dict(root='/synthetic/'+a,files={'/synthetic/'+a+'/libflux_cuda.so':a}) for a in bench.ARMS})
        for job in bench.stage_jobs('preflight'):
            for rank in range(8):
                row=records[f"{job['directory']}/rank{rank}"]
                row.update(job=job,execution_environment=bench.execution_environment(plan,job['arm']),loaded_libraries=plan['libraries'][job['arm']]['files'])
                for role,data in row['roles'].items():
                    arm=job['arm'];calls=2*len(bench.check_indices(plan))+plan['retained_output_calls']+plan['instance_isolation_calls']//2+4
                    data['submission_before']=submission_fixture(role,data['packet_pointers'],0,arm) if arm in ('G','CR2') else []
                    data['submission_after']=submission_fixture(role,data['packet_pointers'],calls,arm) if arm in ('G','CR2') else []
                    data['graph_preparation']=dict(graph_exec_count=2,host_seconds=.1) if arm=='G' else None
                    for i,check in enumerate(data['retained_output_checks']):
                        check.update(input_ptr=100+i,weight_ptr=200+i,output_ptr=300+i)
                    reference=data['checks'][0]['correctness']
                    data['inplace_update_checks']=[dict(generation=i,input_ptr=11,weight_ptr=12,correctness=reference,
                        inputs_sha256=str(i+1)*64,weights_sha256=str(i+2)*64,
                        output_sha256=('c' if bench.family(arm)=='selective' else 'f')*64) for i in range(4)]
        return plan,records

    def validate(self,plan,records):
        with patch.object(bench,'read',side_effect=lambda path:records[str(path.parent.name)+'/'+path.stem]),patch.object(bench,'sha',return_value='1'*64):
            return bench.validate_rows(Path('/unused'),plan,bench.stage_jobs('preflight'),'preflight')

    def test_inplace_and_dynamic_cases_match_selective_controls(self):
        plan,records=self.fixture();self.validate(plan,records)
        for arm in ('G','CR2','C'):
            plan,records=self.fixture()
            records[f'preflight-{arm}/rank7']['roles']['mlp']['inplace_update_checks'][0]['output_sha256']='a'*64
            with self.assertRaisesRegex(ValueError,'bitwise'):self.validate(plan,records)

    def test_inplace_stress_requires_same_storage_and_changed_content(self):
        for key,value in [('input_ptr',99),('weight_ptr',99),('weights_sha256','2'*64),('inputs_sha256','1'*64)]:
            plan,records=self.fixture()
            records['preflight-G/rank0']['roles']['attn']['inplace_update_checks'][1][key]=value
            with self.assertRaisesRegex(ValueError,'in-place'):self.validate(plan,records)

    def test_dynamic_address_evidence_cannot_alias(self):
        for key in ('input_ptr','weight_ptr','output_ptr'):
            plan,records=self.fixture()
            checks=records['preflight-G/rank0']['roles']['attn']['retained_output_checks']
            checks[1][key]=checks[0][key]
            with self.assertRaisesRegex(ValueError,'independent input'):self.validate(plan,records)

    def test_operator_rejects_missing_graph_replay_and_cr2_submission(self):
        for arm,key in [('G','graph_launch_count'),('CR2','successful_calls')]:
            plan,records=self.fixture()
            records[f'preflight-{arm}/rank0']['roles']['attn']['submission_after'][0][key]-=1
            with self.assertRaises(ValueError):self.validate(plan,records)


if __name__=='__main__':unittest.main()
