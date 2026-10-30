#!/usr/bin/env python3
"""MA7 paired F4 consumer checks and forward-only microbenchmarks.

The unchanged MA4-F implementation is frozen as a source template. Both arms retain identical native forward sources.
Full-step measurements evaluate the independent weight-gradient stream.
"""
from pathlib import Path
import ast
import itertools
import random


def registered_replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise ValueError('MA7 registered source anchor changed: '+old[:100])
    return source.replace(old,new)


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / 'ma7_bench_base.py'
if not TEMPLATE.is_file():
    TEMPLATE = HERE / 'ma4_f_bench.py'
_source = TEMPLATE.read_text()
_source = _source[:_source.index("\nif __name__=='__main__':")]
_source = registered_replace(_source,'H100_MA4_F_', 'H100_MA7_',4)
_source = registered_replace(_source,'import ma4_f_build as builder', 'import ma7_build as builder')
_source = registered_replace(_source,'import ma4_f_adapter\n', 'import ma7_adapter as ma4_f_adapter\n')
_source = registered_replace(_source,"job['arm']=='F4'", "job['arm'] in ARMS",2)
_source = registered_replace(_source,"equivalence_pairs=[['F4','CR2']]", 'equivalence_pairs=EQUIVALENCE_PAIRS',2)
_source = registered_replace(_source,"mechanism=MECHANISM, roles=ROLES", "mechanism=MECHANISM, tuning=identity['tuning'], roles=ROLES")
_source = registered_replace(_source,'native_mechanism(op,job[\'arm\'],plan)', "actual_tuning=native_tuning(op,job['arm'],plan)")
_source = registered_replace(_source,"result['roles'][role]=dict(shape=shape", "result['roles'][role]=dict(native_tuning=actual_tuning, backward_state=dict(backward_state), shape=shape")
_source = registered_replace(_source,'def get_tensor_model_parallel_world_size():return 8',
    'def get_tensor_model_parallel_world_size():return 8\n        @staticmethod\n'
    '        def get_tensor_model_parallel_group():return group')
_source = registered_replace(_source,"gradients=[]\n        if args.preflight:",
    "gradients=[]\n        backward_state=ma4_f_adapter.new_backward_state(job['arm'])\n        if args.preflight:")
_source = registered_replace(_source, "mechanism=MECHANISM, tuning=identity['tuning'], roles=ROLES",
    "mechanism=MECHANISM, tuning=identity['tuning'], backward_schedule='independent_weight_stream', roles=ROLES")
_source = registered_replace(_source,'fused_fn.apply(*left,OpAdapter(op),None)',
                         "namespace['consumer_apply'](*left,OpAdapter(op),None,backward_state)")
_source = registered_replace(_source,'output.backward(upstream);expected.backward(upstream)',
    "output.backward(upstream);namespace['drain_backward_state'](backward_state);expected.backward(upstream)")
_source = registered_replace(_source,'OpAdapter(op)', 'candidate_adapter',2)
_source = registered_replace(_source,'OpAdapter(reference)', 'reference_adapter')
_source = registered_replace(_source,'shape=ROLES[role];op,reference=new_op(role),new_op(role)',
    'shape=ROLES[role];op,reference=new_op(role),new_op(role)\n'
    '        candidate_adapter,reference_adapter=OpAdapter(op),OpAdapter(reference)')
_source = registered_replace(_source,"comparisons={'F4/CR2':paired(values['F4'],values['CR2'])}",
                         "comparisons={c+'/'+b:paired(values[c],values[b]) for c,b in COMPARISONS}")
exec(compile(_source, str(TEMPLATE), 'exec'), globals())
__doc__ = 'MA7 F4 / F4-wgrad consumer correctness and forward-only operator timings.'

import ma7_adapter
import ma7_sources

SOURCE_PROTOCOL = 'h100-ma7-f4-wgrad-v1'
PROTOCOL = SOURCE_PROTOCOL + '-operator'
ARMS = ('F4', 'F4-wgrad')
MAIN_COMPARISONS = (('F4-wgrad', 'F4'),)
COMPARISONS = MAIN_COMPARISONS
EQUIVALENCE_PAIRS = [list(pair) for pair in COMPARISONS]
LABELS = {'F4':'Unchanged CR3 native forward and original backward',
          'F4-wgrad':'Identical CR3 forward; deferred weight gradient on an independent CUDA stream'}
_verify, _validate_rows, _freeze_helpers, _report = verify, validate_rows, freeze_helpers, report


def schedule(blocks=8):
    require(type(blocks) is int and blocks >= 8 and blocks % 8 == 0,
            'Use complete multiples of eight paired blocks')
    first = [list(ARMS[b % 2:] + ARMS[:b % 2]) for b in range(blocks // 2)]
    return [dict(block=b, arm=arm, index=b*2+i, directory=f'block{b:02d}-{arm}')
            for b, order in enumerate(first + [r[::-1] for r in first[::-1]]) for i, arm in enumerate(order)]


def expected_tuning(plan,arm):
    family(arm)
    tuning=ma7_sources.tuning_config(plan['tuning'])
    return {role:dict(config,receiver_variant=0,native_protocol=1,
                      packed_mask_bound=False) for role,config in tuning.items()}


def validate_tuning(value):
    return ma7_sources.tuning_config(value)


def validate_backward_schedule(plan):
    require(plan.get('backward_schedule') == 'independent_weight_stream' and 'gather_schedule' not in plan,
            'MA7 requires the independent weight-gradient stream experiment')


def native_tuning(op,arm,plan):
    expected=expected_tuning(plan,arm)
    require(len(op.slots)==2,'Missing original F4 receiver slots')
    for slot in op.slots:
        fn=slot.lib.taco_ma4f_protocol_version;fn.argtypes=[];fn.restype=ctypes.c_int
        require(fn()==1,'Missing native fused consumer')
        state=slot.cr2_submission_state()
        require(state['bound_k'] in (512,2048),'Wrong immutable K')
        require(not hasattr(slot.lib,'taco_ma6_protocol_version'),
                'MA7 must use unchanged F4 native forward sources in both arms')
    return expected


def _repository_build_identity(build):
    """Called only by the clean repository-side verification subprocess."""
    import ma7_build as builder
    root=Path(build).resolve();record=builder.verify(root,built=True);manifest=read(root/'manifest.json')
    require(record['protocol']==SOURCE_PROTOCOL and tuple(manifest['variants'])==ARMS,'Wrong MA7 build')
    return dict(builder_root=str(Path(builder.__file__).resolve().parent),
        manifest_sha256=sha(root/'manifest.json'),ma7_manifest_sha256=sha(root/builder.MANIFEST),
        builder_inputs=record['inputs'],mechanism=MECHANISM,tuning=validate_tuning(record['tuning']),
        source_files={str((root/arm/name).resolve()):digest for arm in ARMS
                      for name,digest in manifest['variants'][arm]['source_sha256'].items()})


def build_identity(build,builder_root=None):
    # Frozen workers have already imported scripts/common.py and source
    # helpers. Changing sys.path cannot replace those cached modules: importing
    # a repository builder here would give it the frozen HERE/REPO instead.
    # Keep the namespaces separate without weakening online provenance checks.
    import subprocess
    directory=Path(builder_root if builder_root is not None else HERE).resolve()
    require((directory/'ma7_bench.py').is_file() and (directory/'ma7_build.py').is_file(),
            'MA7 repository builder/controller missing: '+str(directory))
    script='''import hashlib, json, sys
from pathlib import Path
directory = Path(sys.argv[1]).resolve()
controller = directory / 'ma7_bench.py'
if hashlib.sha256(controller.read_bytes()).hexdigest() != sys.argv[3]:
    raise RuntimeError('Repository MA7 controller differs from the frozen worker')
sys.path.insert(0, str(directory))
import ma7_bench
if Path(ma7_bench.__file__).resolve() != controller:
    raise RuntimeError('MA7 repository controller resolved to another directory')
print(json.dumps(ma7_bench._repository_build_identity(Path(sys.argv[2]))))
'''
    checked=subprocess.run([sys.executable,'-I','-c',script,str(directory),
                            str(Path(build).resolve()),sha(Path(__file__))],
                           capture_output=True,text=True,check=False)
    require(checked.returncode==0,
            'MA7 isolated build verification failed:\n'+checked.stderr.strip())
    result=json.loads(checked.stdout)
    require(result['builder_root']==str(directory),'MA7 builder root changed during verification')
    return result


def freeze_helpers(scripts):
    _freeze_helpers(scripts)
    for source,name in [(TEMPLATE,'ma7_bench_base.py'),(Path(ma7_adapter.__file__),'ma7_adapter.py'),
                        (Path(ma7_sources.__file__),'ma7_sources.py'),
                        *[(HERE/name,name) for name in ('common.py','ma4_f_sources.py','ma2_cr2_sources.py')]]:
        shutil.copy2(source,scripts/name)


def verify(root,offline=False):
    plan=_verify(root,offline)
    validate_backward_schedule(plan)
    require(plan['tuning']==plan['build_identity']['tuning'],'Receiver plan/build mismatch')
    validate_tuning(plan['tuning'])
    for current,name in [(TEMPLATE,'ma7_bench_base.py'),(base_path,'ma4_f_bench_base.py'),
                          (HERE/'ma4_f_adapter.py','ma4_f_adapter.py'),
                          (Path(ma7_adapter.__file__),'ma7_adapter.py'),
                          (Path(ma7_sources.__file__),'ma7_sources.py'),
                          *[(HERE/name,name) for name in ('common.py','ma4_f_sources.py','ma2_cr2_sources.py')]]:
        require(sha(current)==plan['source_files']['scripts/'+name],'Changed controller dependency: '+name)
    return plan


def forward_consumer(op,arm,x,w,bias,residual):
    family(arm)
    return op.forward_consumer(x,w,bias,residual)


def validate_consumer(states,submissions,arm):
    family(arm)
    require(len(states)==len(submissions)==2,'Missing fused slots')
    for state,submission in zip(states,submissions):
        require(state['protocol_version']==1 and state['failed'] is False and
                state['slot']==submission['slot'] and state['bound_k']==submission['bound_k'] and
                state['local_packet_ptr']==submission['local_packet_ptr'] and
                state['successful_calls']==state['bias_residual_calls']==submission['successful_calls'],
                'Fused consumer fallback/counter drift')


def validate_backward_state(state,arm,calls,*,training=False):
    family(arm)
    deferred=calls if arm=='F4-wgrad' else 0
    require(state==dict(protocol=ma7_adapter.PROTOCOL,arm=arm,
        backward_schedule='independent_weight_stream',forward_calls=calls,backward_calls=calls,
        baseline_backward_calls=calls if arm=='F4' else 0,
        deferred_wgrad_calls=deferred,cuda_wgrad_calls=deferred,cpu_wgrad_calls=0,
        main_grad_accumulations=deferred if training else 0,
        standalone_grad_assignments=0 if training else deferred,
        drained_wgrad_calls=deferred,drain_calls=calls,pending_wgrad=0,
        peak_pending_wgrad=int(deferred>0),async_gather_calls=0,wait_calls=0,
        failed=False,active=False),'Wrong deferred backward schedule identity or coverage')


def validate_rows(root,plan,jobs,stage,reference=None):
    result=_validate_rows(root,plan,jobs,stage,reference)
    for job in jobs:
        for rank in range(8):
            row=read(Path(root)/'results'/job['directory']/f'rank{rank}.json')
            for value in row['roles'].values():
                require(value['native_tuning']==expected_tuning(plan,job['arm']),'Wrong native receiver evidence')
                validate_backward_state(value['backward_state'],job['arm'],3 if stage=='preflight' else 0)
    return result


def decoder_name(arm,role,tuning):
    family(arm);validate_tuning(tuning)
    require(role in ROLES,'Unknown decoder role')
    return 'taco_ma4f_decode_consumer_kernel'


def decoder_matches(name,arm,role=None,tuning=None):
    if role is None:
        family(arm)
        return decoder_name(arm,next(iter(ROLES)),validate_tuning(None)) in name
    return decoder_name(arm,role,tuning) in name


def validate_launch(gemm,decoder,arm,role,tuning):
    expected=expected_tuning({'tuning':tuning},arm)[role]
    require(decoder_matches(decoder['name'],arm,role,tuning),'Wrong role decoder kernel')
    require(gemm['args']['grid']==[512,1,1] and gemm['args']['block']==[128,1,1], 'GEMM launch changed')
    require(decoder['args']['grid']==[expected['decoder_blocks'],1,1] and
            decoder['args']['block']==[expected['decoder_threads'],1,1],'Actual decoder geometry changed')
    return dict(gemm_grid=gemm['args']['grid'],gemm_block=gemm['args']['block'],
                decoder_grid=decoder['args']['grid'],decoder_block=decoder['args']['block'],
                decoder_rows=expected['decoder_rows'],receiver_variant=expected['receiver_variant'],
                decoder_kernel=decoder_name(arm,role,tuning))


def trace_path(events,arm,role,calls=4,tuning=None):
    tuning=validate_tuning(tuning)
    kernels=[e for e in events if e.get('cat')=='kernel']
    runtime=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')]
    durations=[];used=set();launches=[]
    for i in range(calls):
        markers=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')==f'ma4f.{role}.{i}']
        require(len(markers)==1,'Missing consumer marker');marker=markers[0]
        inside=lambda e:e.get('pid')==marker.get('pid') and e.get('tid')==marker.get('tid') and marker['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=marker['ts']+marker['dur']+.01
        correlations={e.get('args',{}).get('correlation') for e in runtime if inside(e)}-{None}
        require(not any(e.get('cat')=='gpu_memcpy' and e.get('args',{}).get('correlation') in correlations for e in events),'Unexpected consumer memcpy')
        path=sorted((e for e in kernels if e.get('args',{}).get('correlation') in correlations),key=lambda e:e['ts'])
        require(len(path)==3 and 'flux_bf16' in path[0]['name'] and 'reducescatter' in path[0]['name'] and
                'CudaIpcBarrierAllKernel' in path[1]['name'],'Wrong producer/barrier/consumer path')
        launches.append(validate_launch(path[0],path[2],arm,role,tuning))
        require(len({e.get('args',{}).get('stream') for e in path})==1 and
                all(a['ts']+a['dur']<=b['ts']+.01 for a,b in zip(path,path[1:])),'Consumer stream order changed')
        require(not used & {id(e) for e in path},'Repeated GPU assignment');used.update(map(id,path))
        require(not any(e.get('cat')=='cpu_op' and e.get('name') in ('aten::clone','aten::copy_') and inside(e) for e in events),'Unexpected consumer clone')
        durations.append(dict(cpu_us=marker['dur'],decoder_us=path[2]['dur'],chain_us=path[-1]['ts']+path[-1]['dur']-path[0]['ts']))
    require(len(used)==len(kernels),'Unattributed operator kernels')
    return dict(calls=calls,execution_path_validated=True,consumer_add_kernels_per_call=0,fused_consumer=True,
        launch=launches[0],median_us={k:statistics.median(v[k] for v in durations) for k in durations[0]})


def profile_summary(root,offline=False):
    root=Path(root).resolve();plan=verify(root,offline);timed=report(root,save=False,offline=offline)
    require(read(root/'report.json')==timed,'Operator report changed')
    gate=preflight(root,save=False,offline=offline)
    _,_,files,_=validate_rows(root,plan,stage_jobs('profile'),'profile',(timed['measurement_identity'],dict(gate['tensor_fingerprints'])))
    profiles={}
    for arm in ARMS:
        for rank in range(8):
            for role in ROLES:
                path=root/'results'/f'profile-{arm}'/f'rank{rank}-{role}-profile.json.gz'
                with gzip.open(path,'rt') as handle:profiles[f'{arm}/rank{rank}/{role}']=trace_path(json.load(handle)['traceEvents'],arm,role,tuning=plan['tuning'])
                files[str(path.relative_to(root))]=sha(path)
    return dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,execution_path_validated=True,
        target_validated=False,profiles=profiles,evidence_files=files,
        limitation='Forward-only profiles; the distinct backward DAG and stream execution are checked in fullstep.')


def report(root,save=True,offline=False):
    result=_report(root,save=False,offline=offline)
    result['backward_schedule']=verify(root,offline)['backward_schedule']
    result['limitation']='Forward-only complete projection+bias+residual API. End-to-end benefit requires the independent full-step comparisons; the candidate changes only the weight-gradient dependency and stream.'
    if save:save_unchanged(Path(root)/'report.json',result);print(json.dumps(result['roles'],indent=2))
    return result


if __name__=='__main__':main()
