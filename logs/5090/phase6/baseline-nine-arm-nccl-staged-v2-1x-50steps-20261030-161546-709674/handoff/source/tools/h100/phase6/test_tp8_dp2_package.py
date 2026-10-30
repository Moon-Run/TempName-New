"""CPU regressions for source export, real DP admission, and profile adaptation."""
import ast
import copy
import inspect
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import tp8_dp2_protocol as protocol
import tp8_dp2_sources as sources
import tp8_dp2_train as train
import test_ma8_profile
import test_ma11_profile


def fixture(arm='ma11',mode='smoke'):
    job=dict(arm=arm,mode=mode,block=-1,index=0)
    count=3 if mode=='smoke' else 1 if mode=='profile' else 50
    warmup=2 if mode=='smoke' else 10
    targets=[f'projection{i}' for i in range(64)]
    parameters={f'parameter{i}':dict(shape=[2],dtype='torch.float32',sha256='a'*64) for i in range(389)}
    trajectory=[dict(step=s,gradients=parameters,parameters=parameters) for s in range(3)] if mode=='smoke' else []
    rows=[]
    for rank in range(16):
        tp,dp=rank%8,rank//8
        row=dict(completed=True,passed=True,skips=0,protocol=protocol.PROTOCOL,package_id='package',
            arm=arm,mode=mode,rank=rank,block=-1,window=0,world_size=16,local_rank=tp,tp_rank=tp,dp_rank=dp,
            tp_group_ranks=list(range(dp*8,dp*8+8)),dp_group_ranks=[tp,tp+8],model=protocol.MODEL,
            microsteps=2,sample_ids=[dp*2,dp*2+1],tokens_per_optimizer_step=8192,warmup_steps=warmup,
            losses=[1.]*count,grad_norms=[2.]*count,target_modules=targets,
            timed_calls={n:count*2 for n in targets},fallback_calls=0,
            forward_checks={n:dict(passed=True) for n in targets} if mode=='smoke' else {},
            peak_allocated_gib=20.,peak_reserved_gib=30.,
            timed_steps=count,elapsed_seconds=30.,gpu_uuid=f'gpu{rank}',hardware={'host':f'node{dp}'},
            global_tokens_sha256='global',tokens_sha256=f'part{dp}',initial_parameters_sha256=f'initial{tp}',
            final_parameters_sha256=f'final{tp}',parameter_trajectory=trajectory,
            ma11_activation_setup=dict(installed=True,module_roles=targets),
            ma11_zero_count_state=dict(installed=True,failed=False,calls=warmup+count,kernel_calls=2*(warmup+count),
                initializations=1,selected_tensors=389 if tp==0 else 194,audits=warmup+count if mode=='smoke' else 0,
                self_test={'passed':True} if mode=='smoke' else None),
            ma11_softmax_state=dict(installed=True,modules=list(range(32)),shape=[1,4,2048,2048],
                forward_audits=320 if mode=='smoke' else 0,backward_audits=320 if mode=='smoke' else 0,
                self_test={'passed':True} if mode=='smoke' else None),
            ma11_consumer_state={n:dict(projection_calls=2*(warmup+count),bda_calls=2*(warmup+count)) for n in targets},
            codec_calls={n:2*(warmup+count) for n in targets},
            codec_validation=dict(passed=True,synthetic=False) if mode=='smoke' else None,
            native_linear_forward_preserved=True,flux_library_loaded=False)
        rows.append(copy.deepcopy(row))
    return job,rows


def profile_module(name):
    module=types.ModuleType(name)
    exec(compile(sources.profile_source(name+'.py'),name,'exec'),module.__dict__)
    return module


class Contracts(unittest.TestCase):
    def test_balanced_paired_schedule(self):
        from collections import Counter
        schedule=protocol.schedule()
        self.assertEqual(len(schedule),36)
        orders=[tuple(x['arm'] for x in schedule[i:i+3]) for i in range(0,36,3)]
        self.assertEqual(set(Counter(orders).values()),{2})
        self.assertEqual(len(set(orders)),6)
        for i in range(3):
            self.assertEqual(Counter(order[i] for order in orders),dict.fromkeys(protocol.ARMS,4))

    def test_no_performance_gain_still_complete(self):
        samples=[dict(j,ms_per_step=600. if j['arm']=='ma11' else 500.) for j in protocol.schedule()]
        result=protocol.report(samples,1,12,'package')
        self.assertTrue(result['completed'])
        self.assertEqual(result['timing_rank_records'],576)
        self.assertLess(result['comparisons']['compare1']['comparisons']['ma11/flux-basic']['percent'],0)
        with self.assertRaisesRegex(ValueError,'Missing/reordered'):
            protocol.report(samples[:-1],1,12,'package')

    def test_all_arms_and_tp_rank_zero_on_second_node(self):
        for arm in protocol.ARMS:
            for mode in ('smoke','timing','profile'):
                job,rows=fixture(arm,mode)
                self.assertEqual(len(protocol.validate_rows(rows,job,'package')),16)

    def test_smoke_requires_all_projection_checks_and_timing_forbids_them(self):
        for arm in protocol.ARMS:
            for mode in ('smoke','timing','profile'):
                job,rows=fixture(arm,mode)
                rows[8]['forward_checks']=({} if mode=='smoke' else {'projection0':dict(passed=True)})
                with self.subTest(arm=arm,mode=mode),self.assertRaises(ValueError):
                    protocol.validate_rows(rows,job,'package')

    def test_reject_wrong_group_batch_or_missing_rank(self):
        for field,value in [('world_size',8),('tp_group_ranks',list(range(16))),('dp_group_ranks',[8]),
                            ('microsteps',4),('sample_ids',[0,1]),('final_parameters_sha256','wrong')]:
            job,rows=fixture()
            rows[8][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                protocol.validate_rows(rows,job,'package')
        job,rows=fixture()
        with self.assertRaises(ValueError): protocol.validate_rows(rows[:-1],job,'package')

    def test_reject_dp_gradient_divergence_and_duplicate_inputs(self):
        job,rows=fixture()
        rows[8]['parameter_trajectory'][1]['gradients']['parameter2']['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'gradients/updates'):
            protocol.validate_rows(rows,job,'package')
        job,rows=fixture()
        for row in rows: row['tokens_sha256']='same'
        with self.assertRaisesRegex(ValueError,'distinct'):
            protocol.validate_rows(rows,job,'package')

    def test_second_node_zero_count_uses_tp_rank(self):
        job,rows=fixture()
        rows[8]['ma11_zero_count_state']['selected_tensors']=194
        with self.assertRaisesRegex(ValueError,'MA8/MA9'):
            protocol.validate_rows(rows,job,'package')

    def test_worker_shards_global_batch_and_keeps_native_dp(self):
        for arm in protocol.ARMS:
            source=sources.worker_source(arm)
            ast.parse(source)
            self.assertIn('assert args.data_parallel_size == 2',source)
            self.assertIn('all_tokens = all_tokens[sample_start:sample_start+microsteps]',source)
            self.assertIn('config.finalize_model_grads_func = finalize_model_grads',source)
            self.assertIn('num_microbatches=microsteps',source)
            self.assertNotIn('operator-controller.py',source)
            self.assertNotIn('check_topology(WORLD)',source)
            self.assertNotIn("HERE.parent/'plan.json'",source)
            self.assertIn("report['final_parameters_sha256']",source)
            self.assertIn('parameter_fingerprints(True)',source)
        arguments=sources.model_arguments()
        self.assertEqual(arguments[arguments.index('--tensor-model-parallel-size')+1],'8')
        self.assertEqual(arguments[arguments.index('--global-batch-size')+1],'4')

    def test_package_hash_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'source.py').write_text('pass\n')
            (root/'link').symlink_to('source.py')
            info=dict(protocol=protocol.PROTOCOL,model=protocol.MODEL,arms=list(protocol.ARMS),
                      files={'source.py':protocol.sha(root/'source.py')},symlinks={'link':'source.py'})
            info['package_id']=protocol.digest(info);protocol.write(root/'package.json',info)
            protocol.verify_package(root)
            (root/'source.py').write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'file changed'): protocol.verify_package(root)
            (root/'source.py').write_text('pass\n')
            (root/'link').unlink();(root/'link').symlink_to('../outside')
            info['symlinks']['link']='../outside'
            info.pop('package_id');info['package_id']=protocol.digest(info);protocol.write(root/'package.json',info)
            with self.assertRaisesRegex(ValueError,'escapes'): protocol.verify_package(root)

    def test_preview_preserves_python_environment_path(self):
        args=train.parser(['--node-rank','1','--python','/opt/conda/envs/test/bin/python'])
        self.assertEqual(str(args.python),'/opt/conda/envs/test/bin/python')
        with self.assertRaises(ValueError):
            train.parser(['--node-rank','1','--python','python','--master-port','29580'])


class ProfileAdaptation(unittest.TestCase):
    def test_all_ma11_mechanisms_and_dp_coalescing_together(self):
        import tp8_dp2_profile
        from test_ma9_profile import fixture as zero_fixture
        from test_ma89_profile import _shift
        from test_tp8_dp2_profile import dp_fixture
        adapted_cache=profile_module('ma8_profile')
        with mock.patch.dict(sys.modules,{'ma8_profile':adapted_cache}):
            adapted_attention=profile_module('ma11_profile')
        namespace={'profile':adapted_cache}
        exec(inspect.getsource(test_ma8_profile.fixture).replace('range(256 if row else 260)',
                                                               'range(128 if row else 130)'),namespace)
        columns=namespace['fixture']
        namespace={'profile':adapted_attention}
        exec(inspect.getsource(test_ma11_profile.fixture).replace('range(4)','range(2)'),namespace)
        events=(columns(True)+_shift(columns(True,True),300000,100000)+
                _shift(zero_fixture(True,0),600000,200000)+
                _shift(namespace['fixture'](True),1000000,1000000)+
                _shift(dp_fixture(),3000000,3000000))
        with mock.patch.object(tp8_dp2_profile,'ma11_profile',adapted_attention):
            result=tp8_dp2_profile.analyze(events,'ma11',8)
        self.assertEqual(result['activation_cache']['column_calls'],128)
        self.assertEqual(result['zero_count']['eligible_parameter_count'],389)
        self.assertEqual(result['attention_softmax']['forward_calls'],64)
        self.assertEqual(result['dp_collective_names'],{'allreduce_coalesced':1})

    def test_two_microbatch_cache_and_original_rows(self):
        profile=profile_module('ma8_profile')
        text=inspect.getsource(test_ma8_profile.fixture).replace('range(256 if row else 260)','range(128 if row else 130)')
        namespace={'profile':profile};exec(text,namespace)
        result=profile.trace_activation_cache_records(namespace['fixture'](True),'F4-cache')
        self.assertEqual(result['column_calls'],128)
        self.assertEqual(result['original_vocabulary_heads']['calls'],2)
        self.assertEqual(result['roles']['qkv']['allowed_workspace_memsets']['qkv_bias_48byte'],64)
        row=profile.trace_row_backward_records(namespace['fixture'](True,True),'F4-cache')
        self.assertEqual(row['attn']['backward_calls'],64)
        with self.assertRaises(ValueError):
            profile.trace_activation_cache_records(test_ma8_profile.fixture(True),'F4-cache')

    def test_attention_counts_keep_ownership_and_allow_dp_scaling(self):
        profile=profile_module('ma11_profile')
        text=inspect.getsource(test_ma11_profile.fixture).replace('range(4)','range(2)')
        namespace={'profile':profile};exec(text,namespace)
        events=namespace['fixture'](True)
        result=profile.trace_attention_softmax_records(events,profile.ARMS[1])
        self.assertEqual(result['microbatches'],2)
        self.assertEqual(result['softmax_compute_kernel_calls'],128)
        events.remove(next(e for e in events if e['cat']=='kernel' and profile.FORWARD_KERNEL in e['name']))
        with self.assertRaisesRegex(ValueError,'coverage'):
            profile.trace_attention_softmax_records(events,profile.ARMS[1])

    def test_dp_profile_requires_real_allreduce_kernels(self):
        import tp8_dp2_profile
        events=[dict(cat='cpu_op',name='record_param_comms',ts=0,dur=10,tid=1,
                     args={'Group size':2,'Collective name':'allreduce','In msg nelems':1000000,
                           'Out msg nelems':1000000,'Process Group Description':'DATA_PARALLEL_GROUP_WITH_CP','dtype':'Float'})]
        with self.assertRaisesRegex(ValueError,'correlated NCCL'):
            tp8_dp2_profile.analyze(events,'ma11',8)


class Processes(unittest.TestCase):
    def test_last_ack_never_reads_a_closed_store(self):
        coordinator=object.__new__(train.Coordinator)
        coordinator.rank=1;coordinator.timeout=1;coordinator.store=mock.Mock()
        coordinator.finish()
        coordinator.store.set.assert_called_once_with('node1-final-state-written','done')
        coordinator.store.get.assert_not_called()
        coordinator.store.check.assert_not_called()

    def test_peer_failure_terminates_own_child(self):
        class Peer:
            def check(self): raise RuntimeError('peer failed')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'peer failed'):
                train.run_process([sys.executable,'-c','import time;time.sleep(30)'],temp,dict(os.environ),
                                  Path(temp)/'process.log',10,Peer())

    def test_failed_child_cannot_be_marked_complete(self):
        peer=mock.Mock()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'Child exited 7'):
                train.run_process([sys.executable,'-c','raise SystemExit(7)'],temp,dict(os.environ),
                                  Path(temp)/'process.log',10,peer)


if __name__=='__main__': unittest.main()
