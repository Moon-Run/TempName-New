#!/usr/bin/env python3
"""MA4-F: the complete projection + bias + residual path versus frozen CR2.

The existing CR2 controller supplies unchanged I/O, process supervision and
paired statistics. This controller adds consumer-specific evidence, not another
copy of the projection-only benchmark.
"""
import argparse
import ast
import ctypes
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
base_path = HERE/'ma4_f_bench_base.py'
if not base_path.is_file():
    base_path = HERE/'ma2_cr2_bench.py'
spec = importlib.util.spec_from_file_location('_ma4_f_cr2_helpers', base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
read, write, sha, require, save_unchanged = base.read, base.write, base.sha, base.require, base.save_unchanged
paired, run_worker = base.paired, base.run_worker
SOURCE_PROTOCOL = 'h100-ma4-f-v1'
PROTOCOL = SOURCE_PROTOCOL + '-operator'
ARMS = ('F4', 'CR2')
ROLES = base.ROLES
COMPARISONS = (('F4', 'CR2'),)
LABELS = {'F4': 'CR2 decoder with ordered BF16 bias and residual consumption',
          'CR2': 'Unchanged CR2 followed by eager bias and residual additions'}
C_POLICY, BOOTSTRAP_SEED = base.C_POLICY, base.BOOTSTRAP_SEED
MECHANISM = dict(fusion='decoder_bias_residual', dropout=0.0, ordered_bf16_rounding=True)
verify_selection = base.verify_selection


def family(arm):
    require(arm in ARMS, 'Unknown MA4-F arm')
    return 'selective'


def validate_mechanism(value):
    require(value == MECHANISM, 'MA4-F consumer semantics changed')
    return value


def expected_mechanism(plan, arm):
    validate_mechanism(plan['mechanism']); family(arm)
    return dict(MECHANISM, fused=arm == 'F4')


def native_mechanism(op, arm, plan):
    expected = expected_mechanism(plan, arm)
    if arm == 'F4':
        for slot in op.slots:
            fn = slot.lib.taco_ma4f_protocol_version
            fn.argtypes = []; fn.restype = ctypes.c_int
            require(fn() == 1, 'Wrong native consumer protocol')
    return expected


def schedule(blocks=8):
    require(type(blocks) is int and blocks >= 8 and blocks % 8 == 0, 'Use complete multiples of eight paired blocks')
    first = [list(ARMS[b % 2:]+ARMS[:b % 2]) for b in range(blocks//2)]
    return [dict(block=b, arm=arm, index=b*2+i, directory=f'block{b:02d}-{arm}')
            for b, order in enumerate(first+[r[::-1] for r in first[::-1]]) for i, arm in enumerate(order)]


def stage_jobs(stage):
    require(stage in ('preflight', 'profile'), 'Unknown stage')
    return [dict(arm=arm, block=-1, index=i, directory=f'{stage}-{arm}') for i, arm in enumerate(ARMS)]


def libraries(build):
    result = {}
    for arm in ARMS:
        root = (Path(build)/arm).resolve()
        paths = [root/'python/flux/lib'/name for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')]
        paths += sorted((root/'python').glob('flux_ths_pybind*.so'))
        require(len(paths) >= 3 and all(p.is_file() for p in paths), 'Missing linked MA4-F runtime: '+arm)
        mapping = root/'check_mapping'
        require(mapping.is_file(), 'Missing mapping executable')
        result[arm] = dict(root=str(root), files={str(p.resolve()): sha(p) for p in paths},
                           mapping=dict(path=str(mapping), sha256=sha(mapping)))
    return result


def build_identity(build, builder_root=None):
    if builder_root is not None:
        sys.path.insert(1, str(builder_root))
    import ma4_f_build as builder
    root = Path(build).resolve()
    record = builder.verify(root, built=True)
    manifest = read(root/'manifest.json')
    require(record['protocol'] == SOURCE_PROTOCOL and tuple(manifest['variants']) == ARMS,
            'Wrong MA4-F build')
    return dict(builder_root=str(Path(builder.__file__).resolve().parent),
                manifest_sha256=sha(root/'manifest.json'), ma4_f_manifest_sha256=sha(root/builder.MANIFEST),
                builder_inputs=record['inputs'], mechanism=MECHANISM,
                source_files={str((root/arm/name).resolve()): digest for arm in ARMS
                              for name, digest in manifest['variants'][arm]['source_sha256'].items()})


def freeze_helpers(scripts):
    shutil.copy2(base_path, scripts/'ma4_f_bench_base.py')
    shutil.copy2(HERE/'ma4_f_adapter.py', scripts/'ma4_f_adapter.py')
    original = (REPO/'tools/a800/phase5/e2e/adapter.py').read_text()
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.ClassDef) and n.name == 'GemmRSFunction')
    (scripts/'control_autograd.py').write_text(ast.get_source_segment(original, node)+'\n')
    source = (REPO/'tools/a800/phase5/e2e/taco_support.py').read_text()
    before = source[:source.index('\nclass TacoForward:')]
    before = '\n'.join(line for line in before.splitlines() if not line.startswith('from routing import'))+'\n'
    (scripts/'codec_reference.py').write_text(before+source[source.index('\n@torch.no_grad()\ndef check_taco_output'):])
    shutil.copy2(REPO/'tools/a800/phase4/reference.py', scripts/'reference.py')


def prepare(args):
    root, build = args.root.resolve(), args.build.resolve()
    identity, libs = build_identity(build), libraries(build)
    selection = read(build/'selection-plan.json'); base.verify_selection(selection)
    jobs = schedule(args.blocks)
    require(args.warmup >= 1 and args.iterations >= 2, 'Invalid operator timing length')
    root.mkdir(parents=True, exist_ok=False)
    scripts = root/'scripts'; scripts.mkdir(); (root/'results').mkdir()
    shutil.copy2(__file__, scripts/'bench.py'); freeze_helpers(scripts)
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex, arms=list(ARMS), arm_labels=LABELS,
                build=str(build), build_identity=identity, libraries=libs, selection=selection,
                mechanism=MECHANISM, roles=ROLES, schedule=jobs, warmup=args.warmup,
                iterations=args.iterations, cases=['zero','small','outlier','random'], seeds=[817,919],
                retained_output_calls=6, gradient_update_steps=3, special_cases=['rounding_witness','tiny'],
                shared_environment={'CUDA_DEVICE_MAX_CONNECTIONS':'1','FLUX_GEMM_RS_GEMM_ONLY':'1'},
                communication={'NCCL_NVLS_ENABLE':'0','NCCL_DEBUG':'WARN','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'},
                source_files={str(p.relative_to(root)):sha(p) for p in scripts.iterdir()},
                target_validated=False)
    write(root/'plan.json',plan); print(root)


def verify(root, offline=False):
    root=Path(root).resolve(); plan=read(root/'plan.json')
    require(plan['protocol']==PROTOCOL and plan['arms']==list(ARMS) and plan['arm_labels']==LABELS,
            'Wrong MA4-F operator plan')
    require(plan['roles']==ROLES and set(plan['libraries'])==set(ARMS), 'Wrong operator coverage')
    require(plan['schedule']==schedule(len(plan['schedule'])//2), 'Changed paired schedule')
    require(plan['cases']==['zero','small','outlier','random'] and plan['seeds']==[817,919] and
            plan['retained_output_calls']==6 and plan['gradient_update_steps']==3 and
            plan['special_cases']==['rounding_witness','tiny'], 'Incomplete consumer correctness plan')
    validate_mechanism(plan['mechanism']); base.verify_selection(plan['selection'])
    for name,digest in plan['source_files'].items():
        require(sha(root/name)==digest,'Frozen consumer helper changed: '+name)
    if not offline:
        require(sha(__file__)==plan['source_files']['scripts/bench.py'], 'Consumer controller changed')
        require(build_identity(plan['build'],plan['build_identity']['builder_root'])==plan['build_identity'], 'Build changed')
        require(libraries(plan['build'])==plan['libraries'],'Linked libraries changed')
    return plan


def execution_environment(plan, arm):
    family(arm)
    return dict(plan['communication'],**plan['shared_environment'],H100_MA4_F_ARM=arm,H100_MA4_F_PROTOCOL=SOURCE_PROTOCOL)


def environment(root, plan, arm, python, cuda_home, profile=False):
    root=Path(root); lib=Path(plan['libraries'][arm]['root'])
    env={k:v for k,v in os.environ.items() if not k.startswith(('NCCL_','TORCH_NCCL_','H100_MA','H100_CR2_',
                                                              'H100_OP3_','H100_OP4_','H100_CLOSURE_'))}
    env.update(execution_environment(plan,arm),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1',PYTHONPATH=f'{root}/scripts:{lib}/python',H100_MA4_F_BENCH_EXECUTE='1')
    for name in ('FLUX_FORCE_NVLINK','FLUX_TUNE_CONFIG_FILE','PYTHONOPTIMIZE'):env.pop(name,None)
    env['PATH']=str(Path(python).parent)+':'+env.get('PATH','')
    ld=[str(lib/'python/flux/lib'),str(lib),str(lib/'build/lib')]
    if cuda_home:
        env['CUDA_HOME']=str(cuda_home);env['PATH']=str(Path(cuda_home)/'bin')+':'+env['PATH'];ld.append(str(Path(cuda_home)/'lib64'))
    env['LD_LIBRARY_PATH']=':'.join(ld)+':'+env.get('LD_LIBRARY_PATH','')
    return env


def mapping_evidence(root, plan):
    import csv
    files={}; reference={}
    for arm in ARMS:
        for role in ROLES:
            p=Path(root)/'results'/f'mapping-{arm}-{role}.csv'
            with p.open() as handle:rows=[{k:int(v) for k,v in row.items()} for row in csv.DictReader(handle)]
            require(len(rows)==4096 and {r['rank'] for r in rows}==set(range(8)), 'Incomplete mapping')
            for rank in range(8):
                values=sorted((r for r in rows if r['rank']==rank),key=lambda r:r['index'])
                require([r['index'] for r in values]==list(range(512)) and
                        all(r['destination']==r['m']//2 and r['sm_occupancy']>0 for r in values), 'Invalid mapping or occupancy')
                require([r['m']*32+r['n'] for r in values]==plan['selection']['variants'][C_POLICY]['roles'][role]['entry']['coords'][rank],
                        'Frozen selective mapping changed')
            reference.setdefault(role,rows);require(rows==reference[role], 'Candidate/control mapping or occupancy differs')
            files[str(p.relative_to(root))]=sha(p)
    return files


def run(args):
    root=args.root.resolve(); plan=verify(root); python=args.python.expanduser().absolute()
    jobs=stage_jobs('profile') if args.profile else stage_jobs('preflight')+plan['schedule']
    prefix=[str(python),'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(root/'scripts/bench.py'),'worker',str(root)]
    if not args.execute:
        print(json.dumps(dict(dry_run=True,jobs=jobs,command=prefix),indent=2));return
    write(root/('profile-started.json' if args.profile else 'timing-started.json'),dict(run_id=plan['run_id'],jobs=jobs))
    if args.profile:require(report(root,save=False)['completed'],'Unprofiled operator report required')
    else:
        for arm in ARMS:
            lib=plan['libraries'][arm]
            flux=next(p for p in lib['files'] if Path(p).name=='libflux_cuda.so')
            for role,(m,n,k) in ROLES.items():
                with (root/'results'/f'mapping-{arm}-{role}.csv').open('x') as handle:
                    run_worker([lib['mapping']['path'],str(m),str(n),str(k*8),'8',flux],root,
                               environment(root,plan,arm,python,args.cuda_home),handle,args.worker_timeout)
    for job in jobs:
        verify(root)
        pre=job['directory'].startswith('preflight-')
        if not args.profile and not pre and job['index']==0:preflight(root)
        dest=root/'results'/job['directory'];dest.mkdir()
        command=prefix+['--job',json.dumps(job,sort_keys=True)]+(['--profile'] if args.profile else ['--preflight'] if pre else [])
        write(dest/'command.json',dict(command=command,job=job))
        try:
            with (dest/'process.log').open('x') as log:
                run_worker(command,root,environment(root,plan,job['arm'],python,args.cuda_home),log,args.worker_timeout)
        except BaseException as error:
            write(dest/'failed.json',dict(status='failed',retry=False,job=job,
                                         exception=type(error).__name__,message=str(error)))
            raise
    if args.profile:save_unchanged(root/'profile-summary.json',profile_summary(root))
    else:report(root)


def validate_submission(states, arm, role, pointers, *, minimum_calls=0):
    family(arm)
    require(len(states)==len(pointers)==2,'Need independent double workspaces')
    for i,(state,pointer) in enumerate(zip(states,pointers)):
        calls=state['successful_calls']
        require(type(calls)is int and calls>=minimum_calls and state['protocol_version']==1 and
                state['initializations']==int(calls>0) and state['slot']==i and state['bound_k']==ROLES[role][2]
                and state['local_packet_ptr']==pointer and state['failed'] is False,'Wrong CR2 submission state')
    require(abs(states[0]['successful_calls']-states[1]['successful_calls'])<=1,'Unbalanced slots')


def validate_consumer(states, submissions, arm):
    if arm=='CR2':require(states==[], 'CR2 unexpectedly fused consumers');return
    require(len(states)==len(submissions)==2,'Missing fused-consumer slots')
    for state,submission in zip(states,submissions):
        require(state['protocol_version']==1 and state['failed'] is False and
                state['slot']==submission['slot'] and state['bound_k']==submission['bound_k'] and
                state['local_packet_ptr']==submission['local_packet_ptr'] and
                state['successful_calls']==state['bias_residual_calls']==submission['successful_calls'],
                'Consumer fallback or counter drift')


def forward_consumer(op, arm, x, w, bias, residual):
    return op.forward_consumer(x,w,bias,residual) if arm=='F4' else residual+(op.forward(x,w)+bias)


def worker(args):
    require(os.environ.get('H100_MA4_F_BENCH_EXECUTE')=='1','Use run --execute')
    root=args.root.resolve();plan=verify(root);job=json.loads(args.job)
    allowed=stage_jobs('profile' if args.profile else 'preflight') if args.profile or args.preflight else plan['schedule']
    require(job in allowed and not(args.profile and args.preflight),'Unregistered worker')
    # Preserve the launcher's CPU set for this rank and its helper threads.
    cpu_affinity=sorted(os.sched_getaffinity(0))
    import torch
    import torch.distributed as dist
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    require(torch.cuda.device_count()==8,'Need eight GPUs')
    dist.init_process_group('nccl');group=dist.group.WORLD
    require(dist.get_world_size()==8,'Need TP8')
    devices=[dict(name=torch.cuda.get_device_name(i),uuid=str(torch.cuda.get_device_properties(i).uuid),
                  sm_count=torch.cuda.get_device_properties(i).multi_processor_count) for i in range(8)]
    require(all('H100' in d['name'] and d['sm_count']==132 for d in devices),'Need eight full H100s')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    import flux
    from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
    from codec_reference import check_taco_output
    import ma4_f_adapter
    flux.init_flux_shm(group)
    mapped={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.strip()}
    libs=plan['libraries'][job['arm']]['files']
    require(all(p in mapped and sha(p)==digest for p,digest in libs.items()),'Loaded consumer runtime changed')
    def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    def bitwise(a,b):
        require(a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().contiguous().view(torch.uint8),
                b.detach().contiguous().view(torch.uint8)), 'Consumer/gradient/update bytes differ')
    def new_op(role):
        m,n,k=ROLES[role];mask=plan['selection']['variants'][C_POLICY]['roles'][role]['entry']['mask']
        return GemmRSTacoDoubleBuffered(group,m,n,k,placement='fused',selected=mask,quantization_mode='targeted')
    def extra(role,seed):
        m,n,k=ROLES[role];gen=torch.Generator(device='cuda').manual_seed(seed)
        return ((torch.randn((n,),generator=gen,device='cuda')*.125+.03125).bfloat16(),
                (torch.randn((m//8,n),generator=gen,device='cuda')*.25+.0625).bfloat16())
    result=dict(protocol=PROTOCOL,run_id=plan['run_id'],plan_sha256=sha(root/'plan.json'),job=job,rank=rank,
                completed=True,preflight=args.preflight,profiling=args.profile,roles={},loaded_libraries=libs,
                hardware=dict(host=os.uname().nodename,devices=devices),
                software=dict(torch=torch.__version__,cuda=torch.version.cuda,nccl=list(torch.cuda.nccl.version())))
    result.update(cpu_affinity_policy='inherit',cpu_affinity=cpu_affinity)
    def gather(value):
        out=torch.empty((value.shape[0]*8,*value.shape[1:]),device=value.device,dtype=value.dtype)
        dist.all_gather_into_tensor(out,value.contiguous(),group=group);return out
    class Parallel:
        @staticmethod
        def get_tensor_model_parallel_world_size():return 8
    namespace=dict(torch=torch,parallel_state=Parallel,_gather_along_first_dim=gather)
    exec(compile((root/'scripts/control_autograd.py').read_text(),'<CR2-autograd>','exec'),namespace)
    exec(compile(ma4_f_adapter.consumer_autograd_source(),'<consumer-autograd>','exec'),namespace)
    control_fn, fused_fn=namespace['GemmRSFunction'],namespace['ConsumerGemmRSFunction']
    class OpAdapter:
        owns_output=True
        def __init__(self,codec):self.codec=codec
        def forward(self,x,w,reduce_scatter_option=None):return self.codec.forward(x,w)
        def forward_consumer(self,x,w,b,r,reduce_scatter_option=None):return self.codec.forward_consumer(x,w,b,r)
    for index,role in enumerate(ROLES):
        shape=ROLES[role];op,reference=new_op(role),new_op(role)
        native_mechanism(op,job['arm'],plan)
        mask_sha=plan['selection']['variants'][C_POLICY]['roles'][role]['mask_sha256']
        require(all(s.selection_sha256==mask_sha for s in op.slots+reference.slots),'Runtime mask changed')
        checks=[];calls=0
        with torch.no_grad():
            for seed in plan['seeds']:
                for case in plan['cases']:
                    x,w=base.make_inputs(torch,shape,seed+rank+index*10000,case);bias,residual=extra(role,seed+rank+91)
                    y=forward_consumer(op,job['arm'],x,w,bias,residual);calls+=1
                    raw=reference.forward(x,w)
                    correctness=check_taco_output(x@w.t(),raw,group,mask=base.reference_mask(reference))
                    bitwise(y,residual+(raw+bias))
                    checks.append(dict(seed=seed,case=case,correctness=correctness,
                        inputs=[tensor_hash(t) for t in (x,w,bias,residual)],output=tensor_hash(y)))
            for case in plan['special_cases']:
                m,n,k=shape;x=torch.zeros((m,k),device='cuda',dtype=torch.bfloat16)
                w=torch.zeros((n,k),device='cuda',dtype=torch.bfloat16)
                if case=='rounding_witness':
                    x[rank*(m//8):(rank+1)*(m//8),0]=1;w[:,0]=1
                    bias=torch.full((n,),1/256,device='cuda',dtype=torch.bfloat16)
                    residual=torch.full((m//8,n),-1,device='cuda',dtype=torch.bfloat16)
                else:
                    tiny=torch.finfo(torch.bfloat16).tiny
                    bias=torch.full((n,),tiny/128,device='cuda',dtype=torch.bfloat16)
                    residual=torch.full((m//8,n),tiny,device='cuda',dtype=torch.bfloat16)
                y=forward_consumer(op,job['arm'],x,w,bias,residual);calls+=1
                raw=reference.forward(x,w)
                correctness=check_taco_output(x@w.t(),raw,group,mask=base.reference_mask(reference))
                bitwise(y,residual+(raw+bias))
                if case=='rounding_witness':require(bool((raw==1).all()) and bool((y==0).all()),'BF16 two-rounding witness failed')
                checks.append(dict(seed=0,case=case,correctness=correctness,
                    inputs=[tensor_hash(t) for t in (x,w,bias,residual)],output=tensor_hash(y)))
            retained=[]
            for generation in range(6):
                x,w=base.make_inputs(torch,shape,8171+generation*100+rank,'random');bias,residual=extra(role,9911+generation+rank)
                if rank in (0,7) and generation%2==rank//7:torch.cuda._sleep(2000000)
                y=forward_consumer(op,job['arm'],x,w,bias,residual);calls+=1
                retained.append((x,w,bias,residual,y))
            torch.cuda.synchronize();base.require_independent_outputs([v[-1] for v in retained])
            retained_checks=[]
            for x,w,bias,residual,y in retained:
                expected=residual+(reference.forward(x,w)+bias)
                bitwise(y,expected)
                retained_checks.append(dict(inputs=[tensor_hash(t) for t in (x,w,bias,residual)],output=tensor_hash(y),
                    pointers=[t.data_ptr() for t in (x,w,bias,residual,y)]))
            del retained
        gradients=[]
        if args.preflight:
            x,w=base.make_inputs(torch,shape,4311+rank,'random');bias,residual=extra(role,5711+rank)
            inputs=[x.reshape(shape[0],1,shape[2]),w,bias,residual.reshape(shape[0]//8,1,shape[1])]
            left=[v.detach().clone().requires_grad_(True) for v in inputs]
            right=[v.detach().clone().requires_grad_(True) for v in inputs]
            for step in range(3):
                if job['arm']=='F4':output=fused_fn.apply(*left,OpAdapter(op),None)
                else:output=left[3]+(control_fn.apply(left[0],left[1],OpAdapter(op),None)+left[2])
                calls+=1
                expected=right[3]+(control_fn.apply(right[0],right[1],OpAdapter(reference),None)+right[2])
                bitwise(output,expected)
                upstream=(torch.arange(output.numel(),device='cuda').reshape_as(output)%13-6).bfloat16()*.03125
                output.backward(upstream);expected.backward(upstream)
                record=dict(step=step,gradients={},updated={})
                with torch.no_grad():
                    for name,a,b in zip(('input','weight','bias','residual'),left,right):
                        bitwise(a.grad,b.grad)
                        record['gradients'][name]=tensor_hash(a.grad)
                        a.add_(a.grad,alpha=-.001);b.add_(b.grad,alpha=-.001)
                        bitwise(a,b)
                        record['updated'][name]=tensor_hash(a)
                        a.grad=None;b.grad=None
                gradients.append(record)
        timing=None;final_output_check=None
        if not args.preflight:
            with torch.no_grad():
                x,w=base.make_inputs(torch,shape,7191+rank,'random');bias,residual=extra(role,8411+rank)
                for _ in range(plan['warmup']):y=forward_consumer(op,job['arm'],x,w,bias,residual)
                calls+=plan['warmup'];dist.barrier();torch.cuda.synchronize()
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        for i in range(4):
                            with torch.profiler.record_function(f'ma4f.{role}.{i}'):
                                y=forward_consumer(op,job['arm'],x,w,bias,residual)
                            torch.cuda.synchronize()
                    calls+=4
                    path=root/'results'/job['directory']/f'rank{rank}-{role}-profile.json'
                    profiler.export_chrome_trace(str(path))
                    with gzip.open(str(path)+'.gz','wb') as handle:handle.write(path.read_bytes())
                    path.unlink()
                else:
                    started=time.perf_counter()
                    for _ in range(plan['iterations']):y=forward_consumer(op,job['arm'],x,w,bias,residual)
                    torch.cuda.synchronize();calls+=plan['iterations']
                    timing=dict(wall_us=(time.perf_counter()-started)*1e6/plan['iterations'])
                expected=residual+(reference.forward(x,w)+bias)
                bitwise(y,expected)
                final_output_check=dict(passed=True,output=tensor_hash(y),reference=tensor_hash(expected),
                                        inputs=[tensor_hash(t) for t in (x,w,bias,residual)])
        submissions=op.cr2_submission_state();consumer=op.ma4_consumer_state() if job['arm']=='F4' else []
        pointers=[slot.peers[rank].data_ptr() for slot in op.slots]
        validate_submission(submissions,job['arm'],role,pointers,minimum_calls=1)
        validate_consumer(consumer,submissions,job['arm'])
        require(sum(s['successful_calls'] for s in submissions)==calls,'Lost consumer calls')
        result['roles'][role]=dict(shape=shape,checks=checks,retained=retained_checks,gradients=gradients,
                                  expected_calls=calls,submission=submissions,consumer=consumer,
                                  packet_pointers=pointers,mask_sha256=mask_sha,timing=timing,
                                  final_output_check=final_output_check)
        del op,reference
    write(root/'results'/job['directory']/f'rank{rank}.json',result)
    dist.barrier();dist.destroy_process_group()


def validate_rows(root,plan,jobs,stage,reference=None):
    identity,fingerprints=(None,{}) if reference is None else reference
    samples=[];files={}
    for job in jobs:
        rows=[]
        for rank in range(8):
            path=Path(root)/'results'/job['directory']/f'rank{rank}.json';row=read(path)
            require(row['protocol']==PROTOCOL and row['run_id']==plan['run_id'] and row['plan_sha256']==sha(Path(root)/'plan.json')
                    and row['rank']==rank and row['job']==job and row['completed'] and row['preflight']==(stage=='preflight')
                    and row['profiling']==(stage=='profile'),'Wrong consumer rank record')
            require(row['loaded_libraries']==plan['libraries'][job['arm']]['files'],'Wrong loaded library')
            current=dict(hardware=row['hardware'],software=row['software']);identity=current if identity is None else identity
            require(identity==current,'Operator hardware/software drift')
            require(set(row['roles'])==set(ROLES),'Missing role')
            for role,value in row['roles'].items():
                require(value['shape']==ROLES[role] and len(value['checks'])==10 and len(value['retained'])==6,'Missing consumer cases')
                require(value['mask_sha256']==plan['selection']['variants'][C_POLICY]['roles'][role]['mask_sha256'],'Wrong runtime mask')
                validate_submission(value['submission'],job['arm'],role,value['packet_pointers'],minimum_calls=1)
                validate_consumer(value['consumer'],value['submission'],job['arm'])
                expected=16+(3 if stage=='preflight' else plan['warmup']+(4 if stage=='profile' else plan['iterations']))
                require(value['expected_calls']==expected==sum(s['successful_calls'] for s in value['submission']),'Wrong consumer coverage')
                require(len(value['gradients'])==(3 if stage=='preflight' else 0),'Missing gradient/update steps')
                final=value['final_output_check']
                if stage=='preflight':require(final is None,'Unexpected timed output check in preflight')
                else:require(final['passed'] and final['output']==final['reference'],'Final timed/profile output changed')
                for check in value['checks']:base.validate_correctness(check['correctness'])
                for column in range(5):require(len({r['pointers'][column] for r in value['retained']})==6,'Dynamic tensors or outputs alias')
                evidence=(value['checks'],[{k:v for k,v in x.items() if k!='pointers'} for x in value['retained']])
                # Numeric reference diagnostics may vary slightly between identical
                # libraries; compare actual input/output bytes and gradient bytes.
                hashes=[[[x['seed'],x['case'],x['inputs'],x['output']] for x in value['checks']],evidence[1]]
                key=f'{rank}/{role}';fingerprints.setdefault(key,hashes)
                require(fingerprints[key]==hashes,'F4/CR2 consumed output is not bitwise identical')
                if stage=='preflight':
                    key+=':gradients';fingerprints.setdefault(key,value['gradients'])
                    require(fingerprints[key]==value['gradients'],'F4/CR2 gradient/update bytes differ')
                if stage=='timing':require(math.isfinite(value['timing']['wall_us']) and value['timing']['wall_us']>0,'Invalid timing')
            files[str(path.relative_to(root))]=sha(path);rows.append(row)
        if stage=='timing':samples.append(dict(job,rankmax_wall_us={role:max(r['roles'][role]['timing']['wall_us'] for r in rows) for role in ROLES}))
    return identity,fingerprints,files,samples


def preflight(root,save=True,offline=False):
    root=Path(root).resolve();plan=verify(root,offline)
    identity,hashes,files,_=validate_rows(root,plan,stage_jobs('preflight'),'preflight')
    files.update(mapping_evidence(root,plan))
    result=dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,correctness_passed=True,
                bitwise_equal_within_pairs=True,equivalence_pairs=[['F4','CR2']],consumer_gradients_validated=True,
                measurement_identity=identity,tensor_fingerprints=hashes,evidence_files=files,target_validated=False)
    if save:save_unchanged(root/'preflight.json',result)
    return result


def report(root,save=True,offline=False):
    root=Path(root).resolve();plan=verify(root,offline);gate=preflight(root,save=False,offline=offline)
    require(read(root/'preflight.json')==gate,'Preflight changed')
    identity,_,files,samples=validate_rows(root,plan,plan['schedule'],'timing',(gate['measurement_identity'],dict(gate['tensor_fingerprints'])))
    files.update(gate['evidence_files']);roles={}
    for role in ROLES:
        values={arm:[s['rankmax_wall_us'][role] for s in sorted(samples,key=lambda s:s['block']) if s['arm']==arm] for arm in ARMS}
        roles[role]=dict(mean_us={arm:statistics.mean(v) for arm,v in values.items()},comparisons={'F4/CR2':paired(values['F4'],values['CR2'])})
    result=dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,correctness_passed=True,
                bitwise_equal_within_pairs=True,equivalence_pairs=[['F4','CR2']],consumer_gradients_validated=True,
                roles=roles,all_samples=samples,measurement_identity=identity,evidence_files=files,target_validated=False,
                limitation='Complete projection+bias+residual API; operator speedups do not prove end-to-end speedup.')
    if save:save_unchanged(root/'report.json',result);print(json.dumps(roles,indent=2))
    return result


def decoder_matches(name,arm):
    return ('taco_ma4f_decode' in name) if arm=='F4' else ('taco_ma2c_decode' in name)


def is_add(event):
    name=event.get('name','').lower()
    return event.get('cat')=='kernel' and 'elementwise' in name and 'add' in name


def trace_path(events,arm,role,calls=4):
    family(arm);kernels=[e for e in events if e.get('cat')=='kernel']
    runtime=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')]
    durations=[];used=set()
    for i in range(calls):
        markers=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')==f'ma4f.{role}.{i}']
        require(len(markers)==1,'Missing consumer marker');marker=markers[0]
        inside=lambda e:e.get('pid')==marker.get('pid') and e.get('tid')==marker.get('tid') and marker['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=marker['ts']+marker['dur']+.01
        correlations={e.get('args',{}).get('correlation') for e in runtime if inside(e)}-{None}
        require(not any(e.get('name','').startswith(('cudaMemcpy','cuMemcpy')) and inside(e) for e in runtime) and
                not any(e.get('cat')=='gpu_memcpy' and e.get('args',{}).get('correlation') in correlations for e in events),
                'Unexpected consumer memcpy')
        path=sorted((e for e in kernels if e.get('args',{}).get('correlation') in correlations),key=lambda e:e['ts'])
        require(len(path)==(3 if arm=='F4' else 5),'Consumer chain kernel count changed')
        require('flux_bf16' in path[0]['name'] and 'reducescatter' in path[0]['name'] and
                'CudaIpcBarrierAllKernel' in path[1]['name'] and decoder_matches(path[2]['name'],arm), 'Wrong producer/publication/consumer path')
        require(all(is_add(e) for e in path[3:]),'CR2 eager bias/residual additions missing')
        require(len({e.get('args',{}).get('stream') for e in path})==1 and all(a['ts']+a['dur']<=b['ts']+.01 for a,b in zip(path,path[1:])), 'Consumer dependency order changed')
        require(not used & {id(e) for e in path},'Repeated kernel assignment');used.update(map(id,path))
        require(not any(e.get('cat')=='cpu_op' and e.get('name') in ('aten::clone','aten::copy_') and inside(e) for e in events),'Unexpected clone/copy')
        durations.append(dict(cpu_us=marker['dur'],decoder_us=path[2]['dur'],chain_us=path[-1]['ts']+path[-1]['dur']-path[0]['ts']))
    require(len(used)==len(kernels),'Unattributed operator kernels')
    return dict(calls=calls,execution_path_validated=True,consumer_add_kernels_per_call=0 if arm=='F4' else 2,
                fused_consumer=arm=='F4',median_us={k:statistics.median(v[k] for v in durations) for k in durations[0]})


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
                with gzip.open(path,'rt') as handle:profiles[f'{arm}/rank{rank}/{role}']=trace_path(json.load(handle)['traceEvents'],arm,role)
                files[str(path.relative_to(root))]=sha(path)
    return dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,execution_path_validated=True,
                target_validated=False,profiles=profiles,evidence_files=files)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('root',type=Path);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--blocks',type=int,default=8);p.add_argument('--warmup',type=int,default=100);p.add_argument('--iterations',type=int,default=200)
    p=sub.add_parser('run');p.add_argument('root',type=Path);p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--cuda-home',type=Path)
    p.add_argument('--profile',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--worker-timeout',type=int,default=1800)
    p=sub.add_parser('worker');p.add_argument('root',type=Path);p.add_argument('--job',required=True);p.add_argument('--profile',action='store_true');p.add_argument('--preflight',action='store_true')
    for action in ('report','profile-report'):
        p=sub.add_parser(action);p.add_argument('root',type=Path);p.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.action=='report':report(args.root,offline=args.offline)
    elif args.action=='profile-report':save_unchanged(args.root/'profile-summary.json',profile_summary(args.root,args.offline))
    else:globals()[args.action](args)


if __name__=='__main__':main()
