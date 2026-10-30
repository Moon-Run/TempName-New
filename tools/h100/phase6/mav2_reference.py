"""Independent tensor D8/H8 oracle; validation only, never the timed codec."""
import math
import torch

RELATIVE_L2 = .025
MAX_ABS_OVER_AMAX = .0625


def _groups(x):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[1] % 128:
        raise ValueError('Expected nonempty BF16 rows of complete H128 groups')
    if not bool(torch.isfinite(x).all()): raise ValueError('Nonfinite codec input')
    return x.float().reshape(x.shape[0], x.shape[1]//128, 128)


def _direct_quantize(x):
    g = _groups(x)
    amax = g.abs().amax(-1)
    # A Python scalar divisor makes PyTorch CUDA multiply by a rounded reciprocal.
    # That one-ULP scale change can flip E4M3 midpoint rounding. A device tensor
    # divisor selects FP32 division, matching the encoder's explicit __fdiv_rn.
    fp8_max = torch.full_like(amax, 448.0)
    scale = torch.where(amax == 0, torch.ones_like(amax), (amax / fp8_max).clamp_min(2.**-149))
    payload = (g / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    restored = (payload.float()*scale.unsqueeze(-1)).to(torch.bfloat16)
    return g, amax, payload, scale, restored


def direct(x):
    g, amax, payload, scale, restored = _direct_quantize(x)
    # Normalize operands before subtracting/squaring: no tiny-value underflow or large-value overflow.
    denominator = torch.where(amax == 0, torch.ones_like(amax), amax).unsqueeze(-1)
    unit = g/denominator
    errors = restored.float()/denominator-unit
    energy, squared = unit.square().sum(-1), errors.square().sum(-1)
    peak = errors.abs().amax(-1)
    selected = (squared <= RELATIVE_L2**2*energy) & (peak <= MAX_ABS_OVER_AMAX)
    return payload, scale, restored.reshape_as(x), selected


def direct_roundtrip(x):
    """The formal all-D8 reference has no quality selection and never calls H8."""
    _, _, _, _, restored = _direct_quantize(x)
    return restored.reshape_as(x), dict(coefficients=x.numel(), saturated=0,
                                       d8=x.numel()//128, h8=0)


def hadamard(x):
    y = x.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pair = y.reshape(*y.shape[:-1], -1, 2, stride)
        left, right = pair[..., 0, :], pair[..., 1, :]
        y = torch.stack((left+right, left-right), dim=-2).reshape_as(y)
    return y/math.sqrt(128)


def h8(x):
    g = _groups(x)
    adaptive = (448*(g.square().mean(-1)+1e-6).clamp_min(1e-12).rsqrt()).clamp(1e-3, 1e3)
    transformed = hadamard(g*adaptive.unsqueeze(-1))
    qs = (transformed.abs().amax(-1).clamp_min(1e-12)/448).clamp(1e-12, 1e6)
    q = (transformed/qs.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
    return (hadamard(q.float()*qs.unsqueeze(-1))/adaptive.unsqueeze(-1)).reshape_as(x).to(torch.bfloat16)


def roundtrip(x):
    _, _, d8, selected = direct(x)
    h = h8(x)
    result = torch.where(selected.unsqueeze(-1), d8.reshape(*selected.shape, 128),
                         h.reshape(*selected.shape, 128)).reshape_as(x)
    return result, dict(coefficients=x.numel(), saturated=0,
                       d8=int(selected.sum()), h8=selected.numel()-int(selected.sum()))


def decode_packets(packets, rows, n):
    """Interpret actual packet bytes independently, including the negative mode tag."""
    elements = rows*n; groups = elements//128; chunks = packets.shape[0]
    if packets.dtype != torch.uint8 or tuple(packets.shape) != (chunks, groups*136):
        raise ValueError('Invalid MAV2 packet shape')
    q = packets[:, :elements].contiguous().view(torch.float8_e4m3fn).float().reshape(chunks, groups, 128)
    qs = packets[:, elements:elements+groups*4].contiguous().view(torch.float32)
    tag = packets[:, elements+groups*4:].contiguous().view(torch.float32)
    if not bool((torch.isfinite(qs) & (qs > 0) & torch.isfinite(tag) & ((tag == -1) | (tag > 0))).all()):
        raise ValueError('Invalid scale or mode tag')
    values = q*qs.unsqueeze(-1)
    adaptive = torch.where(tag > 0, tag, torch.ones_like(tag))
    transformed = hadamard(values)/adaptive.unsqueeze(-1)
    return torch.where((tag == -1).unsqueeze(-1), values, transformed).to(torch.bfloat16).reshape(chunks, rows, n)
