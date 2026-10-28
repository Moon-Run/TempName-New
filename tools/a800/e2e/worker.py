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

local_rank = int(os.environ['LOCAL_RANK'])
allowed_cpus = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed_cpus[local_rank * len(allowed_cpus) // 4]})
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'config.json').read_text())
POLICY = os.environ['E2E_POLICY']
MODE = os.environ['E2E_MODE']
OUT = Path(os.environ['E2E_PROCESS_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
assert os.environ.get('SLURM_JOB_ID') and POLICY in SPEC['policies']
torch.cuda.set_device(local_rank)
assert torch.cuda.device_count() == 4
assert all('A800' in torch.cuda.get_device_name(i) for i in range(4))
assert all(torch.cuda.can_device_access_peer(i,j) for i in range(4) for j in range(4) if i != j)
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

initialize_megatron()
args = get_args()
assert args.tensor_model_parallel_size == 4 and args.sequence_parallel
assert args.pipeline_model_parallel_size == args.context_parallel_size == args.data_parallel_size == 1
for arg,key in [('num_layers','layers'),('hidden_size','hidden'),('ffn_hidden_size','ffn'),('num_attention_heads','heads'),('seq_length','sequence'),('micro_batch_size','micro_batch'),('global_batch_size','global_batch')]:
    assert getattr(args,arg)==SPEC['model'][key],arg
is_fp32_calibration = MODE == 'calibrate' and os.environ.get('E2E_FP32') == '1'
assert args.bf16 != is_fp32_calibration and not args.gradient_accumulation_fusion
assert not args.overlap_grad_reduce and not args.overlap_param_gather
model, optimizer, scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
config = get_model_config(model[0])
config.grad_scale_func = optimizer.scale_loss
config.finalize_model_grads_func = finalize_model_grads
config.timers = get_timers()
raw = unwrap_model(model)[0]
model[0].train()
initial_parameters = {name:p.detach().cpu().clone() for name,p in raw.named_parameters()} if MODE=='calibrate' else None
if is_fp32_calibration:
    native_reference = torch.load(Path(os.environ['E2E_REFERENCE'])/f'rank{dist.get_rank()}.pt', map_location='cpu', weights_only=False)
    with torch.no_grad():
        for name,p in raw.named_parameters():
            p.copy_(native_reference['initial_parameters'][name])
    del native_reference
adapter = Adapter(raw, POLICY, SPEC['shapes'], SPEC.get('ring_reduction',False), os.environ.get('E2E_CONTROL_BACKWARD')=='1')
rank = dist.get_rank()
report = dict(policy=POLICY, mode=MODE, rank=rank, job_id=os.environ['SLURM_JOB_ID'],
    torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
    python=sys.executable, gpu_uuid=str(torch.cuda.get_device_properties(local_rank).uuid),
    padded_vocab_size=args.padded_vocab_size, scope=SPEC['scope'], sampling=False,
    profiling=MODE == 'profile', block=int(os.environ.get('E2E_BLOCK','-1')),
    window=int(os.environ.get('E2E_WINDOW','-1')), losses=[], grad_norms=[], skips=0)
if POLICY != 'native':
    for lib in ['libflux_cuda.so', 'libflux_cuda_ths_op.so']:
        expected = (Path(os.environ['E2E_BUILD']) / POLICY / lib).resolve()
        loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                  if line.endswith('/'+lib)}
        assert loaded and all(Path(p).resolve() == expected for p in loaded), (loaded, expected)
        report[lib] = dict(path=str(expected), sha256=hashlib.sha256(expected.read_bytes()).hexdigest())

def fingerprint(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

report['initial_parameters_sha256'] = fingerprint((name,p.bfloat16() if is_fp32_calibration else p) for name,p in raw.named_parameters())
report['seed'] = args.seed
report['precision'] = 'fp32' if is_fp32_calibration else 'bf16'
report['control_backward'] = os.environ.get('E2E_CONTROL_BACKWARD')=='1'
report['initial_rng_sha256'] = fingerprint([('cpu',torch.get_rng_state()),
    ('cuda',torch.cuda.get_rng_state()), *sorted(get_cuda_rng_tracker().get_states().items())])
microsteps = SPEC['model']['global_batch'] // SPEC['model']['micro_batch']
seq = SPEC['model']['sequence']
generator = torch.Generator(device='cuda').manual_seed(SPEC['model']['token_seed'])
all_tokens = torch.randint(0, SPEC['model']['vocab'], (microsteps,1,seq+1),
                          generator=generator, device='cuda')
positions = torch.arange(seq, device='cuda').unsqueeze(0)
attention_mask = torch.triu(torch.ones((1,1,seq,seq), dtype=torch.bool, device='cuda'), diagonal=1)
loss_mask = torch.ones((1,seq), dtype=torch.float32, device='cuda')
batches = [dict(tokens=t[:,:-1].contiguous(), labels=t[:,1:].contiguous(),
                position_ids=positions, attention_mask=attention_mask, loss_mask=loss_mask) for t in all_tokens]
report['tokens_sha256'] = fingerprint([('tokens',all_tokens)])
report['tokens_per_optimizer_step'] = microsteps * seq
forward_backward = get_forward_backward_func()


def forward_step(iterator, current_model):
    batch = next(iterator)
    result = current_model(batch['tokens'], batch['position_ids'], batch['attention_mask'],
                           labels=batch['labels'], loss_mask=batch['loss_mask'])
    return result, functools.partial(loss_func, batch['loss_mask'])


def forward_backward_step():
    for chunk in model:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()
    return forward_backward(forward_step_func=forward_step, data_iterator=iter(batches),
        model=model, num_microbatches=microsteps, seq_length=seq,
        micro_batch_size=1, decoder_seq_length=None, forward_only=False)


def update():
    successful, grad_norm, _ = optimizer.step()
    scheduler.step(increment=SPEC['model']['global_batch'])
    # Both values are produced by the native optimizer. No extra GPU synchronizations here.
    return successful, grad_norm


def step():
    losses = forward_backward_step()
    successful, grad_norm = update()
    return losses, successful, grad_norm


def clear_gradients():
    for chunk in model:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()


def error_metrics(value, reference):
    value, reference = value.detach().float().cpu(), reference.detach().float().cpu()
    assert value.shape == reference.shape
    delta = value - reference
    denom = max(reference.norm().item(), math.sqrt(reference.numel()) * SPEC['budgets']['near_zero_rms_floor'])
    return dict(max_abs=delta.abs().max().item(), relative_l2=delta.norm().item()/denom,
                reference_rms=reference.square().mean().sqrt().item(),
                finite=bool(torch.isfinite(value).all() and torch.isfinite(reference).all()))


def compare(value, reference, category, key=None):
    budget = SPEC['budgets'][category]
    if key is not None:
        if key.startswith('shape/'):
            budget = SPEC.get('shape_budgets',{}).get(key[6:],budget)
        else:
            budget = SPEC.get('tensor_budgets',{}).get(key,budget)
    metrics = error_metrics(value, reference)
    close = torch.allclose(value.float().cpu(), reference.float().cpu(),
                           atol=budget['atol'], rtol=budget['rtol'])
    metrics['passed'] = metrics['finite'] and close and metrics['relative_l2'] <= budget['relative_l2']
    return metrics


def loss_number(losses):
    numerator = sum(d['lm loss'][0] for d in losses)
    denominator = sum(d['lm loss'][1] for d in losses)
    return (numerator / denominator).item()


if MODE in ('validate','calibrate'):
    reference_dir = Path(os.environ['E2E_REFERENCE'])
    reference_path = reference_dir / f'rank{rank}.pt'
    tensors = {}
    hooks = []
    # Capture the first microstep of all eight replaced modules and local output-head logits.
    for name, module in adapter.modules.items():
        def hook(module, inputs, result, name=name):
            key = 'layer_output/'+name
            if key in tensors:
                return
            tensors[key] = result[0].detach().cpu().clone()
            inputs[0].register_hook(lambda grad, name=name: tensors.__setitem__(
                'input_gradient/'+name, grad.detach().cpu().clone()))
        hooks.append(module.register_forward_hook(hook))
    def logits_hook(module, inputs, result):
        if 'logits/output_head' not in tensors:
            tensors['logits/output_head'] = result[0].detach().cpu().clone()
    hooks.append(raw.output_layer.register_forward_hook(logits_hook))
    # An independent FP32 GEMM + RS subset checks the actual model-layer shapes.
    shape_checks = []
    if POLICY != 'native' or MODE=='calibrate':
        if POLICY != 'native':
            import flux
        generator = torch.Generator(device='cuda').manual_seed(9000+rank+args.seed)
        for m,n,k in SPEC['shapes']:
            a = (torch.randn((m,k//4),device='cuda',generator=generator)*k**-0.25).bfloat16()
            b = (torch.randn((n,k//4),device='cuda',generator=generator)*k**-0.25).bfloat16()
            ref = torch.empty((m//4,n),device='cuda')
            dist.reduce_scatter_tensor(ref,a.float()@b.float().t(),group=parallel_state.get_tensor_model_parallel_group())
            if POLICY != 'native':
                op = flux.GemmRS(parallel_state.get_tensor_model_parallel_group(),1,m,n,torch.bfloat16,
                                torch.bfloat16,transpose_weight=False,fuse_reduction=False,ring_reduction=SPEC.get('ring_reduction',False))
                got = op.forward(a,b,reduce_scatter_option=adapter.option).reshape_as(ref).clone()
                del op
            else:
                got = torch.empty_like(ref,dtype=torch.bfloat16)
                dist.reduce_scatter_tensor(got,a@b.t(),group=parallel_state.get_tensor_model_parallel_group())
            shape_checks.append(dict(shape=[m,n,k],**compare(got,ref,'layer_fp32',key='shape/'+','.join(map(str,[m,n,k])))))
    for iteration in range(SPEC['validation_steps']):
        losses = forward_backward_step()
        if iteration == 0:
            for name,p in raw.named_parameters():
                tensors['parameter_gradient/'+name] = p.main_grad.detach().cpu().clone()
        successful, grad_norm = update()
        report['skips'] += not successful
        report['losses'].append(loss_number(losses))
        report['grad_norms'].append(float(grad_norm))
        if iteration == 0:
            for name,p in raw.named_parameters():
                tensors['parameter_after_step/'+name] = p.detach().cpu().clone()
            for hook in hooks:
                hook.remove()
    clear_gradients()
    report['shape_fp32_checks'] = shape_checks
    report['finite_training'] = all(math.isfinite(x) for x in report['losses']+report['grad_norms']) and report['skips']==0
    if os.environ.get('E2E_WRITE_REFERENCE') == '1':
        assert POLICY == 'native' and not reference_path.exists()
        reference_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(tensors=tensors, report=report, initial_parameters=initial_parameters), reference_path)
        report['checks'] = {k: dict(finite=bool(torch.isfinite(v).all()), passed=bool(torch.isfinite(v).all())) for k,v in tensors.items()}
        report['loss_differences'] = [0.0]*SPEC['validation_steps']
    else:
        reference = torch.load(reference_path,map_location='cpu',weights_only=False)
        assert set(tensors) == set(reference['tensors'])
        keys = ['initial_parameters_sha256','tokens_sha256'] if MODE=='calibrate' else ['initial_parameters_sha256','initial_rng_sha256','tokens_sha256']
        for key in keys:
            assert report[key] == reference['report'][key], key
        report['checks'] = {key: compare(value, reference['tensors'][key], key.split('/')[0], key=key) for key,value in tensors.items()}
        report['loss_differences'] = [abs(a-b) for a,b in zip(report['losses'],reference['report']['losses'])]
    report['passed'] = report['finite_training'] and all(c['passed'] for c in report['checks'].values()) and all(c['passed'] for c in shape_checks) and max(report['loss_differences']) <= SPEC['budgets']['loss_absolute']
    if MODE=='calibrate':
        report['v1_thresholds_passed'] = report['passed']
        report['passed'] = report['finite_training'] and all(c['finite'] for c in report['checks'].values())
elif MODE in ('pilot','timing','profile'):
    warmup = SPEC['pilot_warmup_steps'] if MODE == 'pilot' else SPEC['warmup_steps']
    count = SPEC['pilot_timed_steps'] if MODE == 'pilot' else SPEC['timed_steps']
    for _ in range(warmup):
        _, success, _ = step()
        assert success
    report['warmup_steps'] = warmup
    report['observed_shapes'] = adapter.observed.copy()
    report['warmup_calls'] = adapter.calls.copy()
    assert all(c == warmup * microsteps for c in adapter.calls.values())
    adapter.stop_audit()
    dist.barrier();torch.cuda.synchronize()
    if MODE == 'profile':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            step()
            clear_gradients()
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory(prefix='flux-e2e-trace-',dir='/tmp') as temp:
            path = Path(temp)/'trace.json'
            profiler.export_chrome_trace(str(path))
            with gzip.open(OUT/f'rank{rank}-profile.json.gz','wb') as target:
                target.write(path.read_bytes())
        report['passed'] = True
    else:
        successful_steps = []
        window_losses = []
        window_grads = []
        start = time.perf_counter()
        for _ in range(count):
            losses, success, grad_norm = step()
            successful_steps.append(success)
            window_losses.append(losses)
            window_grads.append(float(grad_norm))
        clear_gradients()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        report['losses'] = [loss_number(losses) for losses in window_losses]
        report['grad_norms'] = window_grads
        report['skips'] = sum(not success for success in successful_steps)
        assert all(math.isfinite(x) for x in report['losses']+window_grads)
        report.update(elapsed_seconds=elapsed, timed_steps=count, ms_per_step=elapsed*1000/count,
            tokens_per_second=count*report['tokens_per_optimizer_step']/elapsed,
            passed=all(successful_steps), timed_target_calls_expected=count*microsteps*len(adapter.modules))
else:
    raise ValueError(MODE)
report.update(completed=True, observed_shapes=adapter.observed, target_modules=list(adapter.modules),
              target_calls=adapter.calls, fallback_calls=0, replaced_modules=0 if POLICY=='native' else len(adapter.modules))
(OUT/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
# Persist every rank before propagating a numerical rejection to torchrun.
dist.barrier()
all_passed = torch.tensor(int(report['passed']),device='cuda',dtype=torch.int)
dist.all_reduce(all_passed,op=dist.ReduceOp.MIN)
assert all_passed.item(), f'{POLICY} {MODE} failed; see {OUT}/rank*.json'
if rank == 0:
    print('E2E_PROCESS_PASS', POLICY, MODE, report.get('ms_per_step'), flush=True)
dist.destroy_process_group()
