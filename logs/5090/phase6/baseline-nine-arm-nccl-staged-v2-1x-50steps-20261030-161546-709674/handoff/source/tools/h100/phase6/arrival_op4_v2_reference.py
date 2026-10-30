"""Independent FP64 matrix oracle for receiver-group closure quantization.

This intentionally does not copy the CUDA butterfly or its summation order.
It accepts each arm's explicit mask and never expects cross-mask bitwise equality.
"""
import math
import os
from functools import lru_cache

try:
    import closure_plan as plan
except ModuleNotFoundError:
    import arrival_op4_v2_plan as plan


@lru_cache(maxsize=8)
def hadamard_matrix(torch, device):
    # Orthogonal symmetric Sylvester matrix from parity, not a butterfly loop.
    matrix = [[(-1.0 if (i & j).bit_count() & 1 else 1.0) / math.sqrt(128)
               for j in range(128)] for i in range(128)]
    return torch.tensor(matrix, dtype=torch.float64, device=device)


def validate_metadata(torch, payload, quant, adaptive):
    valid = (bool(torch.isfinite(payload.float()).all()) and
             bool(torch.isfinite(quant).all()) and bool(torch.isfinite(adaptive).all()) and
             bool(((quant >= 1e-12) & (quant <= 1e6)).all()) and
             bool(((adaptive >= 1e-3) & (adaptive <= 1e3)).all()))
    if not valid:
        raise ValueError('Nonfinite FP8 or scales outside the registered encoder domain')


def dense_decode(torch, payload, quant, adaptive, h):
    validate_metadata(torch, payload, quant, adaptive)
    # Apply each source's own q and a independently; no common-scale assumption.
    return ((payload.double() * quant.double().unsqueeze(-1)) @ h) / adaptive.double().unsqueeze(-1)


def group_selector(torch, arm, phase, source, destination, device):
    # Destination owns two 128-row tiles by 32 column tiles.
    flags = [plan.selected(arm, phase, source, destination, within) for within in range(64)]
    return torch.tensor(flags, dtype=torch.bool, device=device).reshape(2,32).repeat_interleave(128,0)


def normalized_metrics(torch, actual, oracle, bf16):
    finite = all(bool(torch.isfinite(v).all()) for v in (actual, oracle, bf16))
    if not finite:
        raise ValueError('Nonfinite reference or output')
    denominator = max(float(bf16.double().norm()), 1e-20)
    delta = actual.double() - bf16.double()
    protocol_delta = actual.double() - oracle.double()
    tile_error = None
    if tuple(actual.shape) == (256,4096):
        tiles = delta.reshape(2,128,32,128).permute(0,2,1,3).reshape(64,-1)
        base_tiles = bf16.double().reshape(2,128,32,128).permute(0,2,1,3).reshape(64,-1)
        tile_denominator = base_tiles.norm(dim=1).clamp_min(denominator/8 * 1e-3)
        tile_error = float((tiles.norm(dim=1)/tile_denominator).max())
    return dict(max_abs=float(delta.abs().max()), relative_l2=float(delta.norm())/denominator,
                protocol_relative_l2=float(protocol_delta.norm())/denominator,
                max_tile_relative_l2=tile_error,
                absolute_error_p99=float(torch.quantile(delta.abs().flatten(),.99)),
                budget_vs_bf16=.05, budget_vs_codec=.01,
                rounding='FP32 source accumulation; one BF16 output; FP64 matrix oracle')


def _parts(partial, group):
    import torch
    import torch.distributed as dist
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if world != 8 or tuple(partial.shape) != (2048,4096):
        raise ValueError('Closure reference requires exact TP8/M2048/N4096')
    parts = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(parts, partial, group=group)
    return rank, parts


def check_taco_output(partial, actual, group, mask=None):
    """Same legacy adapter signature, new explicit reduction semantics."""
    import torch
    from reference import encode
    arm, phase = os.environ['H100_CLOSURE_ARM'], int(os.environ['H100_CLOSURE_PHASE'])
    if mask is not None:
        supplied = mask.detach().cpu().tolist() if hasattr(mask, 'detach') else mask
        if supplied != plan.mask(arm, phase):
            raise ValueError('External mask differs from the frozen closure protocol')
    with torch.no_grad():
        rank, parts = _parts(partial, group)
        h = hadamard_matrix(torch, actual.device)
        oracle = torch.zeros((256,32,128), dtype=torch.float64, device=actual.device)
        bf16 = torch.zeros_like(actual)
        for i in range(1,9):
            source = (rank+i)%8
            piece = parts[source][rank*256:(rank+1)*256]
            bf16 = (bf16+piece).to(torch.bfloat16)  # Established BF16 ring comparator.
            flags = group_selector(torch, arm, phase, source, rank, actual.device)
            values = piece.double().reshape(256,32,128)
            if bool(flags.any()):
                payload, quant, adaptive, _ = encode(piece)
                decoded = dense_decode(torch, payload, quant, adaptive, h)
                values = torch.where(flags.unsqueeze(-1), decoded, values)
            oracle += values
        oracle = oracle.reshape_as(actual).to(torch.bfloat16)
        error = normalized_metrics(torch, actual, oracle, bf16)
        error['passed'] = error['relative_l2'] <= .05 and error['protocol_relative_l2'] <= .01
        if not error['passed']:
            raise ValueError(str(error))
        return error


def check_actual_packets(op, partial, actual, group):
    """Read the actual selected payloads and reconstruct with a dense FP64 H.

    The producer's non-selected packet locations are deliberately never read.
    The independent GEMM/encode reference remains a separate mandatory check.
    """
    import hashlib
    import torch
    arm, phase = os.environ['H100_CLOSURE_ARM'], int(os.environ['H100_CLOSURE_PHASE'])
    with torch.no_grad():
        rank, parts = _parts(partial, group)
        h = hadamard_matrix(torch, actual.device)
        oracle = torch.zeros((256,32,128), dtype=torch.float64, device=actual.device)
        magnitude = 0.0
        packet_hash = hashlib.sha256()
        source_bytes, groups = 8192*136, 8192
        packet = op.peers[rank]
        for i in range(1,9):
            source = (rank+i)%8
            piece = parts[source][rank*256:(rank+1)*256].double().reshape(256,32,128)
            magnitude += float(piece.norm())
            flags = group_selector(torch, arm, phase, source, rank, actual.device)
            if bool(flags.any()):
                slot = packet[source*source_bytes:(source+1)*source_bytes]
                payload = slot[:groups*128].reshape(256,32,128)[flags].contiguous().view(torch.float8_e4m3fn)
                quant = slot[groups*128:groups*132].view(torch.float32).reshape(256,32)[flags]
                adaptive = slot[groups*132:groups*136].view(torch.float32).reshape(256,32)[flags]
                decoded = dense_decode(torch, payload, quant, adaptive, h)
                piece[flags] = decoded
                for tensor in (payload, quant, adaptive):
                    packet_hash.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            oracle += piece
        oracle = oracle.reshape_as(actual)
        finite = bool(torch.isfinite(actual).all())
        # One final BF16 rounding has a <=~0.4% relative componentwise envelope.
        # The denominator accounts for cancellation, rather than dividing by zero output.
        scale = max(float(oracle.norm()), magnitude*1e-3, 1e-20)
        relative = float((actual.double()-oracle).norm())/scale
        if not finite or relative > .005:
            raise ValueError(f'Actual packet FP64 oracle failed: {relative}')
        flat = actual.flatten()
        indices = (torch.arange(512, device=flat.device) * 2654435761 + 12345) % flat.numel()
        sample = flat[indices].float().cpu().tolist()
        return dict(passed=True, relative_l2=relative, budget=.005,
                    actual_packet_sha256=packet_hash.hexdigest(),
                    sample=sample, sample_pattern='multiplicative-512-v1', sample_elements=len(sample),
                    oracle_norm=float(oracle.norm()), source_norm_sum=magnitude,
                    input_reference='actual packets plus independent BF16 GEMM partials')


def paired_sample_error(first, second):
    """An additional same-mask cross-process check, not a substitute for the full oracle."""
    if len(first) != len(second) or not first or not all(math.isfinite(x) for x in first+second):
        raise ValueError('Missing or nonfinite paired output sample')
    numerator = math.sqrt(sum((x-y)**2 for x,y in zip(first,second)))
    denominator = max(math.sqrt(sum(x*x for x in first)), math.sqrt(sum(x*x for x in second)),1e-20)
    return numerator/denominator
