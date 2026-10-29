"""Complete-sublayer microbenchmark, loaded after original Megatron setup.

No module backward hooks, extra autograd Functions, profiler, or optimizer
updates in measured intervals. Both arms call the same TransformerLayer APIs.
"""
from contextlib import contextmanager


@contextmanager
def attention_boundary(layer, identity):
    """Move the original pre-MLP norm to the MLP region, once, outside timing."""
    norm = layer.pre_mlp_layernorm
    layer.pre_mlp_layernorm = identity
    try:
        yield
    finally:
        layer.pre_mlp_layernorm = norm


def invoke(layer, role, hidden, mask):
    if role == 'attention':
        normalized, residual, context = layer._forward_attention(hidden, attention_mask=mask)
        if normalized is not residual or context is not None:
            raise ValueError('Expected GPT self-attention with identity pre-MLP norm and no cross attention')
        return residual
    return layer._forward_mlp(layer.pre_mlp_layernorm(hidden), hidden)


def parameters(layer, role):
    modules = ((layer.input_layernorm, layer.self_attention) if role == 'attention'
               else (layer.pre_mlp_layernorm, layer.mlp))
    return [(f'{i}.{name}',param) for i,module in enumerate(modules)
            for name,param in module.named_parameters() if param.requires_grad]


def clear_region(params, hidden):
    hidden.grad = None
    for _,param in params:
        param.grad = None
        if hasattr(param, 'main_grad'):
            param.main_grad.zero_()
        if hasattr(param, 'grad_added_to_main_grad'):
            param.grad_added_to_main_grad = False


def one_iteration(torch, dist, layer, role, hidden, grad, mask, params, events):
    clear_region(params, hidden)
    # Synchronization is OUTSIDE the recorded interval. All ranks use the
    # same collective order; no sync or scalar read divides fwd and bwd.
    dist.barrier()
    torch.cuda.synchronize()
    start, middle, finish = events
    start.record()
    output = invoke(layer, role, hidden, mask)
    middle.record()
    torch.autograd.backward(output, grad)
    finish.record()
    torch.cuda.synchronize()
    return dict(forward_ms=start.elapsed_time(middle), backward_ms=middle.elapsed_time(finish),
                total_ms=start.elapsed_time(finish)), output


def run(namespace):
    from contextlib import nullcontext
    torch, dist = namespace['torch'], namespace['dist']
    raw, adapter, plan, report = (namespace[k] for k in ('raw','adapter','PLAN','report'))
    fingerprint = namespace['fingerprint']
    layers = list(raw.decoder.layers)
    assert len(layers) == 32 and not raw.config.recompute_granularity
    assert not raw.config.gradient_accumulation_fusion and not raw.config.defer_embedding_wgrad_compute
    # Initialize every original projection/workspace using a real model pass,
    # with no optimizer update and no extra smoke checks in this worker.
    namespace['clear_gradients']()
    namespace['forward_backward_step']()
    namespace['clear_gradients']()
    adapter.stop_audit()
    settings = plan['layer_measurement']
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    regions = {}
    # Counterbalance traversal independently of the AB/BA arm order.
    order = list(range(32)) if report['block'] % 2 == 0 else list(reversed(range(32)))
    roles = ('attention','mlp') if report['block'] % 2 == 0 else ('mlp','attention')
    for number in order:
        layer = layers[number]
        for role in roles:
            generator = torch.Generator(device='cuda').manual_seed(90210+number*31+report['rank'])
            hidden = torch.randn((256,1,4096),device='cuda',dtype=torch.bfloat16,generator=generator).requires_grad_()
            grad = torch.randn(hidden.shape,device='cuda',dtype=torch.bfloat16,generator=generator)/1024
            params = parameters(layer,role)
            meta = dict(input_sha256=fingerprint([('input',hidden)]),
                        grad_output_sha256=fingerprint([('grad',grad)]), weights_sha256=fingerprint(params))
            # Verify the boundary against the SAME original methods before
            # timing. Moving pre_mlp_layernorm must not change the block output.
            # Retain grad mode: MA11's original fused consumer requires it.
            normal, residual, context = layer._forward_attention(hidden, attention_mask=namespace['attention_mask'])
            expected = residual if role == 'attention' else layer._forward_mlp(normal,residual)
            expected = expected.detach().clone()
            if role == 'mlp':
                # Feed the actual attention output as the MLP boundary input.
                # It is arm-dependent, so do NOT use it for the timed workload.
                check_input = residual.detach().requires_grad_()
                actual = invoke(layer,role,check_input,namespace['attention_mask'])
            else:
                with attention_boundary(layer,torch.nn.Identity()):
                    actual = invoke(layer,role,hidden,namespace['attention_mask'])
            assert torch.equal(actual.detach().view(torch.uint8),expected.view(torch.uint8))
            del normal,residual,expected,actual
            if role == 'mlp': del check_input
            boundary = attention_boundary(layer,torch.nn.Identity()) if role == 'attention' else nullcontext()
            samples = []
            with boundary:
                for i in range(settings['warmup']+settings['iterations']):
                    value,output = one_iteration(torch,dist,layer,role,hidden,grad,
                                                namespace['attention_mask'],params,events)
                    if i >= settings['warmup']: samples.append(value)
                    # Last-iteration checks are outside CUDA event boundaries.
                    if i == settings['warmup']+settings['iterations']-1:
                        assert hidden.grad is not None and torch.isfinite(hidden.grad).all().item()
                        assert torch.isfinite(output).all().item()
                        for _,param in params:
                            g = getattr(param,'main_grad',None)
                            if g is None: g = param.grad
                            assert g is not None and torch.isfinite(g).all().item()
                    del output
            regions[f'{role}/{number}'] = dict(meta,samples=samples,boundary_equivalent=True,
                                             finite_gradients=True,gradient_parameters=len(params))
            del hidden,grad
            if report['rank'] == 0: print('LAYER_COMPLETE',role,number,flush=True)
    unchanged = fingerprint(raw.named_parameters()) == report['initial_parameters_sha256']
    assert unchanged and not adapter.forward_checks
    if plan['arm'] == 'ma11':
        assert namespace['ma11_activation_setup']['installed']
        state = namespace['ma11_softmax_tracker'].state()
        assert state['installed'] and state['forward_audits'] == state['backward_audits'] == 0
        assert namespace['ma11_zero_tracker'].state()['calls'] == 0
        consumer = adapter.consumer_state()
        assert len(consumer) == 64 and all(v['projection_calls'] == v['bda_calls'] for v in consumer.values())
    else:
        assert namespace['codec_validation'] is None
        assert len(adapter.codec_calls()) == 64
    report.update(completed=True,passed=True,regions=regions,weights_unchanged=unchanged,
                  mechanism_validated=True,forward_checks=adapter.forward_checks,
                  instrumentation='cuda_events_no_profiler',warmup_iterations=settings['warmup'],
                  iterations=settings['iterations'])
    return report
