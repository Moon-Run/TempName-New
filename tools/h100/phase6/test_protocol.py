"""CPU-only protocol regressions. Fixtures are synthetic, never GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build
import calibrate
import common
import plan
import prepare
import report
import run


def fixture_config(world=4):
    c=common.config(common.HERE/'config.json')
    c.update(tp=world,sequence=2048,hidden=128 if world==4 else 256,
             ffn=512 if world==4 else 1024,heads=world,layers=2,window=2,fraction=.25)
    return c


def mapping(c,base):
    m,n,k=common.shape(c);tm,tn=m//128,n//128;world=c['tp'];rows=[]
    for rank in range(world):
        for i in range(tm*tn):
            bm,bn=divmod(i,tn)
            if base=='remote_first':x,y=(bm+tm//world*(rank+1))%tm,bn
            else:
                slot=i%world
                if base=='interleaved_remote' and slot<2:slot^=1
                if base=='interleaved_remote_group':slot=(slot+1)%world
                dst=(slot+rank)%world;within=i//world
                x,y=dst*(tm//world)+within//tn,within%tn
            rows.append(dict(rank=rank,index=i,m=x,n=y,base_m=bm,base_n=bn,destination=x//(tm//world)))
    return rows


def reports(c,heldout_shift=0):
    m,n,k=common.shape(c);world=c['tp'];per=m//128*(n//128)//world;result=[]
    for rank in range(world):
        samples=[]
        for p in range(3):
            for offset in range(per):
                tile=rank*per+offset;late=(rank+1+(heldout_shift if p==2 else 0))%world
                receiver=[dict(source=s,tile=tile,fragment=f,observed_ns=1000000+(5000+100*tile if s==late else 1000)+f)
                          for s in range(world) for f in range(8)]
                samples += [dict(mode='off',repeat=p*per+offset,forward_us=10.),
                    dict(mode='receiver_sparse',repeat=p*per+offset,offset=offset,forward_us=12.,receiver=receiver,
                         max_poll_cycle_ns=10,observer_status=[0,0,0,8*world],local_markers=[1000000,1100000])]
        result.append(dict(rank=rank,completed=True,shapes=[dict(M=m,N=n,K_global=k,partition_tiles=per,samples=samples)]))
    return result


class ProtocolTests(unittest.TestCase):
    def test_shape_gates(self):
        c=common.config(common.HERE/'config.json')
        for change in ({'tp':2},{'sm_count':108},{'sequence':7},{'fraction':float('nan')},{'global_batch':3,'micro_batch':2}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'c.json';common.write(p,dict(c,**change))
                with self.assertRaises(ValueError):common.config(p)

    def test_all_bases_and_worlds_preserve_destinations_budget_and_local_bf16(self):
        for world in (4,8):
            c=fixture_config(world)
            for base in common.BASES:
                with self.subTest(world=world,base=base):
                    e=plan.fit(reports(c),mapping(c,base),c,base)
                    common.validate_entry(e,c)
                    self.assertGreater(sum(e['selected_per_source']),0)
                    self.assertEqual(e['training_valid_tiles'],e['tiles'])

    def test_heldout_does_not_train_order_or_mask(self):
        c=fixture_config();rows=mapping(c,'remote_first')
        a=plan.fit(reports(c),rows,c,'remote_first')
        b=plan.fit(reports(c,heldout_shift=1),rows,c,'remote_first')
        for field in ('maps','coords','mask'):self.assertEqual(a[field],b[field])
        self.assertNotEqual(a['heldout_near_critical_fraction'],b['heldout_near_critical_fraction'])

    def test_bad_training_observation_does_not_get_quantized(self):
        c=fixture_config();rs=reports(c)
        # Receiver 0 tile 0 has incomplete/invalid observations on pass zero.
        rs[0]['shapes'][0]['samples'][1]['receiver'][0]['observed_ns']=-1
        e=plan.fit(rs,mapping(c,'remote_first'),c,'remote_first')
        self.assertTrue(all(row[0]==0 for row in e['mask']))
        self.assertEqual(e['training_valid_tiles'],e['tiles']-1)

    def test_plan_rejects_cross_destination_permutation_and_local_quantization(self):
        c=fixture_config();e=plan.fit(reports(c),mapping(c,'remote_first'),c,'remote_first')
        bad=copy.deepcopy(e);bad['mask'][0][0]=1
        with self.assertRaises(ValueError):common.validate_entry(bad,c)
        bad=copy.deepcopy(e);bad['maps'][0][0],bad['maps'][0][-1]=bad['maps'][0][-1],bad['maps'][0][0]
        bad['coords'][0]=[bad['base_coords'][0][i] for i in bad['maps'][0]]
        with self.assertRaises(ValueError):common.validate_entry(bad,c)

    def test_stale_a800_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'plan.json';common.write(p,dict(schema=2,world=8,policies={}))
            with self.assertRaises(ValueError):common.plan(p,fixture_config())

    def test_incomplete_and_diagnostic_builds_cannot_be_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for fields in (dict(status='prepared'),dict(status='built',diagnostic_only=True)):
                common.write(root/'manifest.json',dict(protocol=common.PROTOCOL,arch='sm90',files={},**fields))
                with self.assertRaises(ValueError):common.manifest(root)

    def test_balanced_two_rounds_and_complete_optimizer_commands(self):
        c=fixture_config();c['bases']=list(common.BASES);ps=common.policies(c);rounds=common.orders(c)
        self.assertEqual(len(ps),8)
        self.assertEqual(rounds[1],[list(reversed(x)) for x in reversed(rounds[0])])
        for r in rounds:
            for position in range(len(ps)):
                self.assertEqual(sorted(row[position] for row in r),sorted(ps))
        jobs=list(run.schedule(dict(config=c,policies=ps),'all'))
        self.assertEqual(sum(x[0]=='timing' for x in jobs),2*len(ps)**2)
        self.assertIn('--sequence-parallel',run.arguments(c))

    def test_worker_adaptation_has_no_slurm_or_a800_device_gate(self):
        for source in (prepare.model_worker(),calibrate.worker_source()):
            compile(source,'frozen-worker','exec')
            self.assertNotIn('SLURM_JOB_ID',source)
            self.assertNotIn("'A800'",source)
            self.assertIn('check_topology',source)
        sampler=calibrate.worker_source()
        self.assertLess(sampler.index('torch.cuda.set_device(local)'),sampler.index('import flux'))

    def test_no_unused_device_allocation_at_library_load(self):
        source=(common.REPO/'src/gemm_rs/tile_scheduler/threadblock_swizzle_pcie.hpp').read_text()
        fixed=build.remove_unused_pcie_initializer(source)
        self.assertNotIn('cudaMalloc',fixed)
        self.assertNotIn('segments_global_device',fixed)

    def test_wrapper_routes_all_layout_and_barrier_decisions_together(self):
        source=(common.REPO/'src/gemm_rs/ths_op/gemm_reduce_scatter.cc').read_text()
        adapted=build.wrapper(source)
        self.assertIn('if (owned_taco.enabled)',adapted)
        self.assertIn('if (phase6_arch() == _Sm90{} and nnodes == 1)',adapted)
        self.assertIn('group_barrier.barrier_all(stream)',adapted)
        self.assertIn('get_arch() == _Sm90{}',adapted)  # physical device gate
        self.assertIn('return get_arch();',adapted)   # native attention dispatch

    def test_baseline_swizzle_is_unchanged_and_tables_are_bounded(self):
        src=(common.REPO/'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp').read_text()
        c=fixture_config()
        self.assertEqual(build.swizzle(src,c,'original'),src)
        e=plan.fit(reports(c),mapping(c,'remote_first'),c,'remote_first')
        self.assertIn('phase6_device_lookup',build.swizzle(src,c,'remote_first',e))
        huge=dict(e,maps=[[0]*40000]*c['tp'])
        with self.assertRaises(ValueError):build.table(huge,c['tp'],True)

    def test_sampler_adapts_current_epilogue_with_taco_methods(self):
        source=(common.REPO/'src/gemm_rs/epilogue_evt.hpp').read_text()
        instrumented=build.sampler_epilogue(source)
        self.assertIn('params.mech = mech_config();',instrumented)
        self.assertIn('mech_publish(',instrumented)
        self.assertIn('taco_stage_step(int step_idx)',instrumented)
        self.assertNotIn('static constexpr Params\n  to_underlying_arguments',instrumented)

    def test_paired_statistics_do_not_use_ratio_of_unpaired_medians(self):
        import math
        stats=report.paired([math.log(.9)]*6,resamples=1000)
        self.assertAlmostEqual(stats['percent'],10.)
        self.assertTrue(all(abs(x-10)<1e-10 for x in stats['ci95']))
        with self.assertRaises(ValueError):report.paired([0.])

    def test_prepare_and_dry_run_freeze_inputs_without_launching_workers(self):
        c=fixture_config();ps=common.policies(c)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';candidate=root/'candidate'
            measured=root/'plan.json';common.write(measured,dict(test_only=True))
            manifests={}
            for folder,kind in ((base,'baselines'),(candidate,'candidates')):
                folder.mkdir();m=dict(config=c,kind=kind,variants={},files={},plan_sha256=common.sha(measured))
                names=['original','taco_fused'] if kind=='baselines' else ps[4:]
                for name in names:
                    path=folder/name;path.mkdir();(path/'python').mkdir()
                    role='hopper' if name=='original' else 'fused' if name=='taco_fused' else 'candidate'
                    order='original' if kind=='baselines' else c['bases'][ps[4:].index(name)]
                    m['variants'][name]=dict(role=role,base=order,sampling=False)
                    if name=='taco_fused':
                        m['variants'][name]['implementation']='a800-fused-v2-cta-v1'
                    for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                        p=path/lib;p.write_bytes(b'CPU unit-test placeholder, not a loadable library')
                        m['files'][str(p.relative_to(folder))]=common.sha(p)
                common.write(folder/'manifest.json',m);manifests[folder]=m
            def git_output(args,**kwargs):
                return common.MEGATRON_COMMIT+'\n' if 'rev-parse' in args else ''
            # Mock only measured/build admission; separate tests reject synthetic
            # plans in the real admission path. These artifacts live only in tmp.
            with mock.patch.object(prepare,'manifest',side_effect=lambda p,k:manifests[p]), \
                 mock.patch.object(prepare,'plan',return_value={'hardware':{'host':'unit-test-only'}}), \
                 mock.patch.object(subprocess_module(),'check_output',side_effect=git_output), \
                 mock.patch.object(subprocess_module(),'run',side_effect=AssertionError('Dry run launched a process')):
                import contextlib,io
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare.prepare(root/'run',base,candidate,measured,root/'Megatron-LM')
                    run.main(root/'run','all',False,Path('/unused/python'),None)
                info=common.read(root/'run/submission.json')
                self.assertEqual(info['policies'],ps)
                self.assertEqual(info['variants']['taco_fused']['implementation'],'a800-fused-v2-cta-v1')
                common.verify_files(root/'run',info['files'])
                self.assertFalse((root/'run/all-state.json').exists())


def subprocess_module():
    import subprocess
    return subprocess


if __name__=='__main__':unittest.main()
