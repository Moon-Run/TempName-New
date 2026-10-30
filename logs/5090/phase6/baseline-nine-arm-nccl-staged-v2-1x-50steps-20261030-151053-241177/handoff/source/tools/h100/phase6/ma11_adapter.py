"""Instance-local MA11 masked softmax. Original MA89 runs no MA11 wrapper.

Fixed BF16 [1,4,2048,2048] attention, with the actual broadcast boolean mask.
Native arithmetic is gated against the installed ATen implementation on GPU.
No RNG consumption, per-layer queue, host tensor metadata array or collectives.
"""
import ctypes
import hashlib
import inspect
import textwrap
import types
from contextlib import nullcontext

PROTOCOL = 'h100-ma11-causal-softmax-v1'
CLASS_SHA256 = 'eaeb34d713feb4f22246a07609c4f3306aa6ca7d4739cb22d36b63cebced1b44'
MASK_SHA256 = 'd3b0933503154c4d8f9386655c178ed0c882abfc59165ce94fda23f9e27ed5b3'
SUFFIX = '.self_attention.core_attention.scale_mask_softmax'
SHAPE = (1, 4, 2048, 2048)


def require_source(value, expected):
    source = textwrap.dedent(inspect.getsource(value)).strip() + '\n'
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError('MA11 requires the pinned original softmax/mask source')


class NativeSoftmax:
    def __init__(self, torch, library):
        self.torch, self.library = torch, library
        protocol = library.ma11_softmax_protocol
        protocol.argtypes, protocol.restype = [], ctypes.c_int
        if protocol() != 11:
            raise RuntimeError('Wrong MA11 native protocol')
        self.fwd, self.bwd = library.ma11_softmax_forward_cuda, library.ma11_softmax_backward_cuda
        self.fwd.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
        self.bwd.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
        self.fwd.restype = self.bwd.restype = ctypes.c_int

    def validate(self, value, mask):
        torch = self.torch
        if (tuple(value.shape) != SHAPE or value.dtype != torch.bfloat16 or not value.is_cuda
                or not value.is_contiguous() or mask is None or tuple(mask.shape) != (1, 1, 2048, 2048)
                or mask.dtype != torch.bool or not mask.is_contiguous() or mask.device != value.device
                or value.device.index != torch.cuda.current_device()):
            raise RuntimeError('MA11 supports only fixed contiguous BF16 attention and broadcast bool mask')

    def forward(self, value, mask):
        self.validate(value, mask)
        result = self.torch.empty_like(value)
        code = self.fwd(value.data_ptr(), mask.data_ptr(), result.data_ptr(), 8192, 2048,
                        self.torch.cuda.current_stream(value.device).cuda_stream)
        if code:
            raise RuntimeError('MA11 forward CUDA launch failed: ' + str(code))
        return result

    def backward(self, grad, result, mask):
        grad = grad.contiguous()
        self.validate(grad, mask)
        value = self.torch.empty_like(result)
        code = self.bwd(grad.data_ptr(), result.data_ptr(), mask.data_ptr(), value.data_ptr(),
                        8192, 2048, self.torch.cuda.current_stream(result.device).cuda_stream)
        if code:
            raise RuntimeError('MA11 backward CUDA launch failed: ' + str(code))
        return value


def exact_equal(torch, left, right):
    # Include the sign of zero; no allclose or relaxed fingerprint tolerance.
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def native_self_test(torch, native):
    """Actual ATen oracle, deterministic full-size patterns, before training only."""
    index = torch.arange(2048, device='cuda', dtype=torch.float32)
    base = (index[:, None] * .03125 - index[None, :] * .0625).remainder(17) - 8
    causal = index[None, :] > index[:, None]
    # Exercise the exact finite sentinel, ties, all masked rows and noncausal holes.
    masks = [causal, torch.zeros_like(causal), torch.ones_like(causal),
             ((index[:, None] + index[None, :]).remainder(7) == 0),
             causal, torch.zeros_like(causal)]
    for case, mask2d in enumerate(masks):
        mask = mask2d.reshape(1, 1, 2048, 2048).contiguous()
        value = base[None, None].expand(SHAPE).contiguous().to(torch.bfloat16)
        if case == 1:
            value.zero_()
        elif case == 2:
            value.fill_(-10000)
        elif case == 4:
            # The finite masked value can exceed an unmasked score: -inf is invalid.
            value.fill_(-20000)
        elif case == 5:
            # BF16 subnormal probabilities must survive, including in backward.
            value.fill_(-90)
            value[..., 0] = 0
        grad = (value.float().remainder(5) * .125 - .25).to(torch.bfloat16)
        ref_input = value.clone().requires_grad_()
        ref = ref_input.masked_fill(mask, -10000).softmax(-1)
        ref_grad, = torch.autograd.grad(ref, ref_input, grad)
        out = native.forward(value, mask)
        derivative = native.backward(grad, out, mask)
        if not exact_equal(torch, ref, out) or not exact_equal(torch, ref_grad, derivative):
            raise RuntimeError('MA11 native arithmetic differs from installed ATen, case ' + str(case))
    torch.cuda.synchronize()
    return dict(passed=True, cases=6, forward_bitwise=True, backward_bitwise=True, rng_unchanged=True)


def make_function(torch, native, mode, evidence):
    class MA11MaskedSoftmax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, mask):
            with (torch.autograd.profiler.record_function('ma11.attention.softmax.forward')
                  if mode == 'profile' else nullcontext()):
                result = native.forward(value, mask)
            ctx.save_for_backward(result, mask)
            if mode == 'smoke':
                reference = value.masked_fill(mask, -10000).softmax(-1)
                if not exact_equal(torch, result, reference):
                    raise RuntimeError('MA11 full-model softmax forward differs')
                evidence['forward_audits'] += 1
            return result

        @staticmethod
        def backward(ctx, grad):
            result, mask = ctx.saved_tensors
            with (torch.autograd.profiler.record_function('ma11.attention.softmax.backward')
                  if mode == 'profile' else nullcontext()):
                derivative = native.backward(grad, result, mask)
            if mode == 'smoke':
                reference = torch.ops.aten._softmax_backward_data(
                    grad.contiguous(), result, -1, torch.bfloat16).masked_fill(mask, 0)
                if not exact_equal(torch, derivative, reference):
                    raise RuntimeError('MA11 full-model softmax backward differs')
                evidence['backward_audits'] += 1
            return derivative, None
    return MA11MaskedSoftmax


class Tracker:
    def __init__(self, metadata, evidence):
        self.metadata, self.evidence = metadata, evidence

    def state(self):
        return dict(self.metadata, **self.evidence)


def install_softmax(model, library, mode):
    import torch
    from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.utils import attention_mask_func
    if mode not in ('smoke', 'timing', 'profile') or hasattr(model, '_ma11_softmax'):
        raise RuntimeError('Invalid or duplicate MA11 installation')
    if not torch.__version__.split('+')[0] == '2.6.0':
        raise RuntimeError('MA11 requires the pinned PyTorch 2.6.0 softmax dispatch')
    require_source(FusedScaleMaskSoftmax, CLASS_SHA256)
    require_source(attention_mask_func, MASK_SHA256)
    config = model.config
    if (not model.training or config.num_layers != 32 or config.hidden_size != 4096
            or config.tensor_model_parallel_size != 8 or config.pipeline_model_parallel_size != 1
            or config.context_parallel_size != 1 or config.attention_dropout != 0
            or config.apply_query_key_layer_scaling or config.attention_softmax_in_fp32
            or config.masked_softmax_fusion or config.recompute_granularity is not None
            or config.enable_cuda_graph or config.external_cuda_graph):
        raise RuntimeError('MA11 requires the original MA89 model/attention configuration')
    selected = [(name, module) for name, module in model.named_modules() if name.endswith(SUFFIX)]
    if len(selected) != 32:
        raise RuntimeError('MA11 must cover exactly 32 attention softmax modules')
    for name, module in selected:
        if (type(module) is not FusedScaleMaskSoftmax or not module.training
                or not module.input_in_bf16 or module.input_in_fp16 or module.softmax_in_fp32
                or module.scale is not None or module.scaled_masked_softmax_fusion
                or module.attn_mask_type != AttnMaskType.causal
                or module.mask_func is not attention_mask_func or 'forward' in module.__dict__):
            raise RuntimeError('Incompatible original attention module: ' + name)
    native = NativeSoftmax(torch, library)
    evidence = dict(forward_audits=0, backward_audits=0,
                    self_test=native_self_test(torch, native) if mode == 'smoke' else None)
    function = make_function(torch, native, mode, evidence)
    def forward(self, value, mask):
        return function.apply(value, mask)
    # Installation is transactional: validate all modules and GPU arithmetic first.
    for _, module in selected:
        module.forward = types.MethodType(forward, module)
    tracker = Tracker(dict(protocol=PROTOCOL, installed=True, modules=[n for n, _ in selected],
                           shape=list(SHAPE), original_class_sha256=CLASS_SHA256,
                           original_mask_sha256=MASK_SHA256), evidence)
    model._ma11_softmax = tracker
    return tracker
