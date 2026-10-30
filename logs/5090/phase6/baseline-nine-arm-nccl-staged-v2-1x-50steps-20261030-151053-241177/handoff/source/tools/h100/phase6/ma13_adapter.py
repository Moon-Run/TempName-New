"""MA13 changes only the MA11 forward kernel; use its exact backward and guards.

The original installation function is cloned once with instance-local globals.
No global Megatron or MA11 function is patched. Timing has no extra oracle calls.
"""
import ctypes
import types
from contextlib import nullcontext

import ma11_adapter as reference

PROTOCOL = 'h100-ma13-resident-softmax-v1'
CLASS_SHA256, MASK_SHA256 = reference.CLASS_SHA256, reference.MASK_SHA256
SHAPE, SUFFIX = reference.SHAPE, reference.SUFFIX
exact_equal = reference.exact_equal


class NativeSoftmax(reference.NativeSoftmax):
    def __init__(self, torch, library):
        # Preserve the old backward callable, validation, output allocation,
        # contiguous-gradient handling and current-stream submission literally.
        super().__init__(torch, library)
        protocol = library.ma13_softmax_protocol
        protocol.argtypes, protocol.restype = [], ctypes.c_int
        if protocol() != 13:
            raise RuntimeError('Wrong MA13 native protocol')
        self.fwd = library.ma13_softmax_forward_cuda
        self.fwd.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
        self.fwd.restype = ctypes.c_int


def native_self_test(torch, native):
    """Smoke only: ATen and the actual original MA11 forward, with byte equality."""
    original = reference.NativeSoftmax(torch, native.library)

    class Audited:
        def forward(self, value, mask):
            result = native.forward(value, mask)
            if not exact_equal(torch, result, original.forward(value, mask)):
                raise RuntimeError('MA13 native forward differs from original MA11')
            return result

        def backward(self, *args):
            result = native.backward(*args)
            if not exact_equal(torch, result, original.backward(*args)):
                raise RuntimeError('MA13 native backward differs from original MA11')
            return result

    oracle = Audited()
    reference.native_self_test(torch, oracle)
    index = torch.arange(2048, device='cuda', dtype=torch.float32)
    causal = (index[None, :] > index[:, None]).reshape(1, 1, 2048, 2048)
    gradient = ((index.remainder(13) - 6) * .125).to(torch.bfloat16)
    gradient = gradient[None, None, None, :].expand(SHAPE).contiguous()
    for case in range(10):
        value = torch.zeros(SHAPE, device='cuda', dtype=torch.bfloat16)
        mask = torch.zeros_like(causal)
        if case == 0:  # Both signs of zero, exactly equal maxima.
            value[..., 1::2] = -0.0
        elif case == 1:
            value[..., 0] = float('inf')
        elif case == 2:
            value.fill_(float('-inf'))
        elif case == 3:
            value[..., 0] = float('nan')
        elif case == 4:
            value.fill_(float('nan'))
            mask = torch.ones_like(causal)  # NaNs overwritten by finite sentinel.
        elif case == 5:
            value.fill_(-9984)
            value[..., 0] = -9920
            mask = causal
        elif case == 6:
            value.fill_(-104)
            value[..., 0] = 0  # Boundary near FP32 exp underflow.
        elif case == 7:
            value.fill_(torch.finfo(torch.bfloat16).max)
            value[..., 1::2] = -torch.finfo(torch.bfloat16).max
        elif case == 8:
            value.copy_((index.remainder(17) * .00390625)[None, None, None, :])
            mask = causal
        else:
            value[..., 0] = torch.finfo(torch.bfloat16).tiny
            value[..., 1] = -torch.finfo(torch.bfloat16).tiny
        ref_input = value.clone().requires_grad_()
        expected = ref_input.masked_fill(mask, -10000).softmax(-1)
        expected_grad, = torch.autograd.grad(expected, ref_input, gradient)
        actual = oracle.forward(value, mask)
        actual_grad = oracle.backward(gradient, actual, mask)
        # NaN payloads are not an expanded ATen contract for MA11. For these
        # three newly added nonfinite cases the oracle above still demands
        # byte-exact MA13/MA11 equality in BOTH directions. Keep ATen byte
        # equality for all original six and all seven new finite-result cases.
        if case not in (1, 2, 3) and (not exact_equal(torch, actual, expected) or
                                    not exact_equal(torch, actual_grad, expected_grad)):
            raise RuntimeError('MA13 native arithmetic differs from ATen, extra case ' + str(case))
    torch.cuda.synchronize()
    return dict(passed=True, cases=16, forward_bitwise=True, backward_bitwise=True,
                rng_unchanged=True, ma11_forward_bitwise=True,
                ma11_backward_bitwise=True, aten_bitwise_cases=13, nonfinite_ma11_cases=3)


def make_function(torch, native, mode, evidence):
    class MA13MaskedSoftmax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, mask):
            with (torch.autograd.profiler.record_function('ma13.attention.softmax.forward')
                  if mode == 'profile' else nullcontext()):
                result = native.forward(value, mask)
            ctx.save_for_backward(result, mask)
            if mode == 'smoke':
                expected = value.masked_fill(mask, -10000).softmax(-1)
                if not exact_equal(torch, result, expected):
                    raise RuntimeError('MA13 full-model softmax forward differs')
                evidence['forward_audits'] += 1
            return result

        @staticmethod
        def backward(ctx, grad):
            result, mask = ctx.saved_tensors
            with (torch.autograd.profiler.record_function('ma13.attention.softmax.backward')
                  if mode == 'profile' else nullcontext()):
                derivative = native.backward(grad, result, mask)
            if mode == 'smoke':
                expected = torch.ops.aten._softmax_backward_data(
                    grad.contiguous(), result, -1, torch.bfloat16).masked_fill(mask, 0)
                if not exact_equal(torch, derivative, expected):
                    raise RuntimeError('MA13 full-model softmax backward differs')
                evidence['backward_audits'] += 1
            return derivative, None
    return MA13MaskedSoftmax


# Reuse every original fixed-shape/config/source guard and transactional module
# installation. Both arms install once, after common MA8/MA9 setup. This clone
# changes only its explicit dependencies, never reference.install_softmax.
_globals = dict(reference.install_softmax.__globals__)
_globals.update(PROTOCOL=PROTOCOL, NativeSoftmax=NativeSoftmax,
                native_self_test=native_self_test, make_function=make_function)
install_softmax = types.FunctionType(reference.install_softmax.__code__, _globals,
                                    'install_softmax', reference.install_softmax.__defaults__)
