"""Independent tensor reference for the fixed TACO E4M3/128 wire protocol."""
import math
import torch
import torch.nn.functional as F


def hadamard(x):
    if x.shape[-1] != 128:
        raise ValueError('Fixed 128-element groups only')
    y = x.float().clone()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        view = y.reshape(*y.shape[:-1], -1, 2, stride)
        left, right = view[..., 0, :].clone(), view[..., 1, :].clone()
        y = torch.stack((left+right, left-right), dim=-2).reshape_as(y)
    return y / math.sqrt(128)


def encode(x):
    if x.ndim != 2 or x.shape[1] <= 0 or not torch.isfinite(x).all():
        raise ValueError('Expected a finite nonempty row matrix')
    n = x.shape[1]
    nt = (n+127)//128
    groups = F.pad(x.float(), (0, nt*128-n)).reshape(x.shape[0], nt, 128)
    counts = torch.full((nt,), 128., device=x.device)
    counts[-1] = n-(nt-1)*128
    variance = groups.square().sum(-1)/counts + 1e-6
    adaptive = (448*variance.clamp_min(1e-12).rsqrt()).clamp(1e-3, 1e3)
    transformed = hadamard(groups*adaptive.unsqueeze(-1))
    quant = (transformed.abs().amax(-1).clamp_min(1e-12)/448).clamp(1e-12, 1e6)
    scaled = transformed/quant.unsqueeze(-1)
    saturated = (scaled.abs()>448).sum().item()
    payload = scaled.clamp(-448,448).to(torch.float8_e4m3fn)
    return payload, quant, adaptive, dict(saturated=saturated, coefficients=payload.numel())


def decode(payload, quant, adaptive, n):
    # Decode ALL padded transform coefficients before cropping physical N.
    x = hadamard(payload.float()*quant.unsqueeze(-1))/adaptive.unsqueeze(-1)
    return x.reshape(x.shape[0], -1)[:, :n].to(torch.bfloat16)


def roundtrip(x):
    q, qs, adaptive, stats = encode(x)
    return decode(q, qs, adaptive, x.shape[1]), stats


def wire_bytes(m, n, world):
    if world not in (2,4,8) or m <= 0 or m % (128*world) or n <= 0 or n % 8:
        raise ValueError('Unsupported shape/TP')
    groups=(m//world)*((n+127)//128)
    return dict(per_source=groups*136, per_rank_remote=(world-1)*groups*136,
                per_rank_bf16_remote=(world-1)*(m//world)*n*2,
                allocated_packet_bytes=world*groups*136)
