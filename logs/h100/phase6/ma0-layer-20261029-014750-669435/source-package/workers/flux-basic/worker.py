"""One fresh Megatron process group: model validation or a full-step timing window."""
import functools
import gzip
import tempfile
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

if os.environ.get('H100_TP8_DP2_EXECUTE') != '1':
    raise RuntimeError('GPU worker requires basic-quant --execute')
local_rank = int(os.environ['LOCAL_RANK'])
WORLD = int(os.environ['WORLD_SIZE'])
allowed_cpus = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed_cpus[local_rank * len(allowed_cpus) // 8]})
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'config.json').read_text())
CASE = next(c for c in PLAN['cases'] if c['name'] == os.environ['E2E_CASE'])
SPEC = dict(PLAN,model=dict(PLAN['common'],**CASE),shapes=[[CASE['sequence']*PLAN['common']['micro_batch'],CASE['hidden'],CASE['hidden']],[CASE['sequence']*PLAN['common']['micro_batch'],CASE['hidden'],CASE['ffn']]])
assert WORLD == 16 and CASE['tp'] == 8
POLICY = os.environ['E2E_POLICY']
MODE = os.environ['E2E_MODE']
OUT = Path(os.environ['E2E_PROCESS_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
assert os.environ.get('H100_RUN_ID') and POLICY in SPEC['policies']
from communication import verify as verify_communication
communication = verify_communication(PLAN['communication'])
torch.cuda.set_device(local_rank)
from routing import is_tensor
import megatron_taco as standalone_codec
codec_validation = None
if is_tensor(POLICY) and MODE == 'smoke':
    from basic_quant_validation import validate as validate_codec
    import reference as codec_reference
    codec_validation = validate_codec(standalone_codec, codec_reference, local_rank)

assert torch.cuda.device_count() == 8
from runtime import check_topology
hardware = check_topology(8)
assert all(torch.cuda.can_device_access_peer(i,j) for i in range(8) for j in range(8) if i != j)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_timers
from megatron.training.training import setup_model_and_optimizer
from megatron.core.enums import ModelType
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.utils import get_model_config
from megatron.training.utils import unwrap_model
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from pretrain_gpt import model_provider, loss_func
from adapter import Adapter
from routing import module_routes, definition, is_native, enabled

initialize_megatron()
args = get_args()
assert args.tensor_model_parallel_size == 8 and args.sequence_parallel
assert args.pipeline_model_parallel_size == args.context_parallel_size == 1
assert args.data_parallel_size == 2
assert parallel_state.get_tensor_model_parallel_rank() == local_rank
assert parallel_state.get_data_parallel_rank() == dist.get_rank() // 8
assert dist.get_process_group_ranks(parallel_state.get_tensor_model_parallel_group()) == list(range(dist.get_rank()//8*8, dist.get_rank()//8*8+8))
assert dist.get_process_group_ranks(parallel_state.get_data_parallel_group()) == [local_rank, local_rank+8]
for arg,key in [('num_layers','layers'),('hidden_size','hidden'),('ffn_hidden_size','ffn'),('num_attention_heads','heads'),('seq_length','sequence'),('micro_batch_size','micro_batch'),('global_batch_size','global_batch')]:
    assert getattr(args,arg)==SPEC['model'][key],arg
assert args.bf16 and not args.gradient_accumulation_fusion
assert not args.overlap_grad_reduce and not args.overlap_param_gather
model, optimizer, scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
config = get_model_config(model[0])
config.grad_scale_func = optimizer.scale_loss
config.finalize_model_grads_func = finalize_model_grads
config.timers = get_timers()
raw = unwrap_model(model)[0]
model[0].train()
adapter = Adapter(raw, POLICY, SPEC['shapes'], SPEC['ring_reduction'])
rank = dist.get_rank()
report = dict(case=CASE['name'],model=SPEC['model'],world_size=WORLD,policy=POLICY, mode=MODE, rank=rank, run_id=os.environ['H100_RUN_ID'],hardware=hardware,
    torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
    python=sys.executable, gpu_uuid=str(torch.cuda.get_device_properties(local_rank).uuid),
    padded_vocab_size=args.padded_vocab_size, scope=SPEC['scope'], sampling=False,
    profiling=MODE == 'profile', block=int(os.environ.get('E2E_BLOCK','-1')),
    window=int(os.environ.get('E2E_WINDOW','-1')), losses=[], grad_norms=[], skips=0)
if not is_native(POLICY):
    for lib in ['libflux_cuda.so', 'libflux_cuda_ths_op.so']:
        expected = (Path(os.environ['E2E_BUILD']) / POLICY / lib).resolve()
        loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                  if line.endswith('/'+lib)}
        assert loaded and all(Path(p).resolve() == expected for p in loaded), (loaded, expected)
        report[lib] = dict(path=str(expected), sha256=hashlib.sha256(expected.read_bytes()).hexdigest())
else:
    assert not any('/libflux_cuda' in line for line in Path('/proc/self/maps').read_text().splitlines())
    report['flux_library_loaded']=False

def fingerprint(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

report.update(local_rank=local_rank,
    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
    dp_rank=parallel_state.get_data_parallel_rank(),
    tp_group_ranks=dist.get_process_group_ranks(parallel_state.get_tensor_model_parallel_group()),
    dp_group_ranks=dist.get_process_group_ranks(parallel_state.get_data_parallel_group()))
report['communication'] = communication
report['codec_identity'] = dict(PLAN['codec_identity'],
    source_sha256=hashlib.sha256((HERE/'megatron_taco.py').read_bytes()).hexdigest(),
    reference_sha256=hashlib.sha256((HERE/'reference.py').read_bytes()).hexdigest())
report['protocol'] = PLAN['protocol']
report['package_id'] = PLAN['package_id']
report['arm'] = PLAN['arm']
report['worker_config_sha256'] = hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()
report['scope_definition'] = definition(POLICY)
report['role_plan_sha256'] = PLAN['role_plan_sha256']
report['initial_parameters_sha256'] = fingerprint(raw.named_parameters())
report['seed'] = args.seed
report['precision'] = 'bf16'
report['parameter_count_local'] = sum(p.numel() for p in raw.parameters())
report['initial_rng_sha256'] = fingerprint([('cpu',torch.get_rng_state()),
    ('cuda',torch.cuda.get_rng_state()), *sorted(get_cuda_rng_tracker().get_states().items())])
microsteps = SPEC['model']['global_batch'] // (SPEC['model']['micro_batch'] * 2)
assert microsteps == 2
seq = SPEC['model']['sequence']
generator = torch.Generator(device='cuda').manual_seed(SPEC['model']['token_seed'])
all_tokens = torch.randint(0, SPEC['model']['vocab'], (microsteps*2,SPEC['model']['micro_batch'],seq+1),
                          generator=generator, device='cuda')
report['global_tokens_sha256'] = fingerprint([('tokens',all_tokens)])
sample_start = parallel_state.get_data_parallel_rank() * microsteps
report['sample_ids'] = list(range(sample_start, sample_start+microsteps))
report['microsteps'] = microsteps
all_tokens = all_tokens[sample_start:sample_start+microsteps]
positions = torch.arange(seq, device='cuda').unsqueeze(0).expand(SPEC['model']['micro_batch'],-1).contiguous()
attention_mask = torch.triu(torch.ones((1,1,seq,seq), dtype=torch.bool, device='cuda'), diagonal=1)
loss_mask = torch.ones((SPEC['model']['micro_batch'],seq), dtype=torch.float32, device='cuda')
batches = [dict(tokens=t[:,:-1].contiguous(), labels=t[:,1:].contiguous(),
                position_ids=positions, attention_mask=attention_mask, loss_mask=loss_mask) for t in all_tokens]
report['tokens_sha256'] = fingerprint([('tokens',all_tokens)])
report['tokens_per_optimizer_step'] = SPEC['model']['global_batch'] * seq
forward_backward = get_forward_backward_func()


def forward_step(iterator, current_model):
    batch = next(iterator)
    result = current_model(batch['tokens'], batch['position_ids'], batch['attention_mask'],
                           labels=batch['labels'], loss_mask=batch['loss_mask'])
    return result, functools.partial(loss_func, batch['loss_mask'])


def forward_backward_step():
    return forward_backward(forward_step_func=forward_step, data_iterator=iter(batches),
        model=model, num_microbatches=microsteps, seq_length=seq,
        micro_batch_size=SPEC['model']['micro_batch'], decoder_seq_length=None, forward_only=False)


def update():
    successful, grad_norm, _ = optimizer.step()
    scheduler.step(increment=SPEC['model']['global_batch'])
    # Both values are produced by the native optimizer. No extra GPU synchronizations here.
    return successful, grad_norm


consumer_smoke_step = 0
consumer_parameter_trajectory = []
consumer_parameter_universe = {name:dict(shape=list(parameter.shape),dtype=str(parameter.dtype))
    for name,parameter in raw.named_parameters() if parameter.requires_grad}
def parameter_fingerprints(gradients=False):
    result = {}
    for name, parameter in raw.named_parameters():
        if not parameter.requires_grad:
            continue
        value = getattr(parameter, 'main_grad', None) if gradients else parameter
        if gradients and value is None:
            value = parameter.grad
        assert value is not None, 'Missing trainable gradient: '+name
        result[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
    assert set(result) == set(consumer_parameter_universe)
    return result

def step():
    global consumer_smoke_step
    losses = forward_backward_step()
    audit_parameters = MODE == 'smoke' and consumer_smoke_step >= SPEC['smoke_warmup_steps']
    gradients = parameter_fingerprints(True) if audit_parameters else None
    successful, grad_norm = update()
    if audit_parameters:
        consumer_parameter_trajectory.append(dict(step=len(consumer_parameter_trajectory),
            gradients=gradients, parameters=parameter_fingerprints()))
    consumer_smoke_step += 1
    # One clear per complete optimizer step, including the final timed step.
    clear_gradients()
    return losses, successful, grad_norm


def clear_gradients():
    for chunk in model:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()


def loss_number(losses):
    numerator = sum(d['lm loss'][0] for d in losses)
    denominator = sum(d['lm loss'][1] for d in losses)
    return (numerator / denominator).item()


clear_gradients()
report['timing_protocol'] = 'full-step-one-clear-v2'
assert MODE in ('smoke','timing','profile')
# Loss is collected from the actual warmup trajectory, avoiding a separate-run initial loss.
warmup = SPEC['smoke_warmup_steps'] if MODE=='smoke' else SPEC['warmup_steps']
count = SPEC['smoke_steps'] if MODE=='smoke' else SPEC['timed_steps']
warmup_losses = []
for _ in range(warmup):
    losses, success, grad_norm = step()
    assert success and math.isfinite(float(grad_norm))
    warmup_losses.append(losses)
report['initial_loss'] = loss_number(warmup_losses[0])
report['warmup_last_loss'] = loss_number(warmup_losses[-1])
report['warmup_steps'] = warmup
report['observed_shapes'] = adapter.observed.copy()
report['module_policies'] = module_routes(adapter.observed, POLICY, SPEC['model']['layers'],SPEC['model'])
report['warmup_calls'] = adapter.calls.copy()
assert all(c == warmup*microsteps for c in adapter.calls.values())
adapter.stop_audit()
dist.barrier();torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
if MODE=='profile':
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        losses,success,grad_norm=step()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(prefix='flux-e2e-scale-trace-',dir='/tmp') as temp:
        path=Path(temp)/'trace.json'
        profiler.export_chrome_trace(str(path))
        with gzip.open(OUT/f'rank{rank}-profile.json.gz','wb') as target:target.write(path.read_bytes())
    report.update(passed=bool(success),losses=[loss_number(losses)],grad_norms=[float(grad_norm)])
else:
    successes=[];window_losses=[];window_grads=[]
    start=time.perf_counter()
    for _ in range(count):
        losses,success,grad_norm=step()
        successes.append(success);window_losses.append(losses);window_grads.append(grad_norm)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    report.update(elapsed_seconds=elapsed,timed_steps=count,ms_per_step=elapsed*1000/count,
        tokens_per_second=count*report['tokens_per_optimizer_step']/elapsed,
        losses=[loss_number(x) for x in window_losses],grad_norms=[float(g) for g in window_grads],skips=sum(not s for s in successes),
        passed=all(successes),timed_target_calls_expected=count*microsteps*len(adapter.modules))
report['forward_checks']=adapter.forward_checks
report['codec_placements']=adapter.placements
report['selection_checks']=adapter.selection_checks
if is_native(POLICY):
    from megatron.core.tensor_parallel.layers import RowParallelLinear
    assert len(adapter.native_quant_forwards)==SPEC['model']['layers']*len(definition(POLICY)['roles'])
    assert all(method.__func__.__code__ is RowParallelLinear.forward.__code__ for method in adapter.native_quant_forwards.values())
    report['native_linear_forward_preserved']=True
report['codec_calls'] = adapter.codec_calls()
report['codec_library'] = standalone_codec.loaded_library()
report['codec_validation'] = codec_validation
report['smoke_output_sha256'] = adapter.smoke_outputs
report['retained_output_checks'] = adapter.retention_checks
if MODE == 'smoke':
    assert set(adapter.smoke_outputs) == set(adapter.scope_ops)
    assert set(adapter.retention_checks) == {'attn', 'mlp'}
else:
    assert not adapter.smoke_outputs and not adapter.retention_checks
report['shared_environment'] = {key: os.environ[key] for key in
    ('CUDA_DEVICE_MAX_CONNECTIONS', 'FLUX_GEMM_RS_GEMM_ONLY')}


report['post_audit_routes'] = adapter.post_audit_routes
report['timed_calls'] = adapter.timed_calls
report['forward_preserved'] = adapter.forward_preserved
report['enabled_modules'] = sorted(adapter.scope_ops)
report['independent_scope_instances'] = len({id(op) for op in adapter.scope_ops.values()})
report['workspace_checks'] = adapter.workspace_checks
assert all(n == (1 if MODE=='profile' else count)*microsteps for n in adapter.timed_calls.values())
assert adapter.post_audit_routes == report['module_policies'] == adapter.actual_routes
report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
report['passed'] = report['passed'] and all(math.isfinite(x) for x in report['losses']+report['grad_norms']+[report['initial_loss']])
report.update(completed=True,observed_shapes=adapter.observed,target_modules=list(adapter.modules),
    target_calls=adapter.calls,fallback_calls=0,
    replaced_modules=len(adapter.scope_ops))
report['parameter_trajectory'] = consumer_parameter_trajectory
report['final_parameters_sha256'] = fingerprint(raw.named_parameters())
(OUT/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
dist.barrier()
all_passed=torch.tensor(int(report['passed']),device='cuda',dtype=torch.int)
dist.all_reduce(all_passed,op=dist.ReduceOp.MIN)
assert all_passed.item(),f'{CASE["name"]} {POLICY} {MODE}: non-finite result or skipped update'
if rank==0:print('SCALE_PROCESS_COMPLETE',CASE['name'],POLICY,MODE,report.get('ms_per_step'),flush=True)
dist.destroy_process_group()
