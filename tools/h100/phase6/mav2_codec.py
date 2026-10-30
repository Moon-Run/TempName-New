"""Shared H8/selective/D8 runtime; all remote groups stay FP8 and 136 bytes."""
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

D8_REL_L2 = .025
D8_MAX_ABS = .0625
DIRECT_TAG = -1.0
POLICIES = ('basic', 'mav2', 'd8')
_LIB = None
_IDENTITY = None


def load_library(record=None):
    global _LIB, _IDENTITY
    if _LIB is not None:
        if record is not None and record != _IDENTITY:
            raise RuntimeError('Cannot substitute a different codec library')
        return _LIB
    if record is None:
        record = json.loads((Path(__file__).parent/'config.json').read_text())['codec_library']
    path = Path(record['path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise RuntimeError('Standalone TACO binary hash mismatch')
    lib = ctypes.CDLL(str(path))
    for name in ('baseline_taco_encode', 'baseline_taco_decode'):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p]
        fn.restype = ctypes.c_int
    lib.basic_quant_decode_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_int, ctypes.c_int64, ctypes.c_void_p]
    lib.basic_quant_decode_reduce.restype = ctypes.c_int
    lib.mav2_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
    lib.mav2_encode.restype = ctypes.c_int
    lib.mav2_decode.argtypes = lib.baseline_taco_decode.argtypes
    lib.mav2_decode.restype = ctypes.c_int
    lib.mav2_decode_reduce.argtypes = lib.basic_quant_decode_reduce.argtypes
    lib.mav2_decode_reduce.restype = ctypes.c_int
    lib.baseline_taco_error.argtypes = [ctypes.c_int]
    lib.baseline_taco_error.restype = ctypes.c_char_p
    _LIB, _IDENTITY = lib, dict(path=str(path), sha256=record['sha256'])
    return lib


def loaded_library():
    return _IDENTITY


def _call(name, input_, output, chunks, elements):
    lib = load_library()
    with torch.cuda.device(input_.device):
        code = getattr(lib, name)(input_.data_ptr(), output.data_ptr(), chunks, elements,
                                 torch.cuda.current_stream(input_.device).cuda_stream)
    if code:
        raise RuntimeError(lib.baseline_taco_error(code).decode())


def encoder_mode(policy, force_mode=-1):
    """Freeze each formal arm; only MAV2 exposes explicit smoke mode overrides."""
    if policy not in POLICIES or type(force_mode) is not int or force_mode not in (-1, 0, 1):
        raise ValueError("Invalid MAV2 codec policy/mode")
    if policy != 'mav2' and force_mode != -1:
        raise ValueError('Formal basic/D8 policies cannot be overridden by diagnostic modes')
    return None if policy == 'basic' else 1 if policy == 'd8' else force_mode


def pack_rows(x, destinations, policy=None, force_mode=-1):
    policy = policy or os.environ.get("E2E_POLICY", "mav2")
    mode = encoder_mode(policy, force_mode)
    if (not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.ndim != 2 or
            destinations <= 0 or x.shape[0] % destinations or x.shape[1] % 128):
        raise ValueError('Standalone upstream codec requires contiguous BF16 full H128 groups')
    elements = x.numel() // destinations
    packets = torch.empty((destinations, elements + elements//128*8), dtype=torch.uint8, device=x.device)
    if policy == 'basic':
        _call('baseline_taco_encode', x, packets, destinations, elements)
    else:
        lib = load_library()
        with torch.cuda.device(x.device):
            code = lib.mav2_encode(x.data_ptr(), packets.data_ptr(), destinations, elements, mode,
                                  torch.cuda.current_stream(x.device).cuda_stream)
        if code: raise RuntimeError(lib.baseline_taco_error(code).decode())
    return packets


def unpack_rows(packets, rows, n, policy=None):
    policy = policy or os.environ.get("E2E_POLICY", "mav2")
    if policy not in POLICIES: raise ValueError("Unknown MAV2 policy")
    elements = rows*n
    if (not packets.is_cuda or packets.dtype != torch.uint8 or not packets.is_contiguous() or
            packets.ndim != 2 or rows <= 0 or n <= 0 or n % 128 or
            packets.shape[1] != elements + elements//128*8):
        raise ValueError('Invalid upstream TACO packet shape')
    output = torch.empty((packets.shape[0], rows, n), dtype=torch.bfloat16, device=packets.device)
    _call('baseline_taco_decode' if policy == 'basic' else 'mav2_decode',
          packets, output, packets.shape[0], elements)
    return output


def decode_reduce(packets, local, rank, policy=None):
    policy = policy or os.environ.get("E2E_POLICY", "mav2")
    if policy not in POLICIES: raise ValueError("Unknown MAV2 policy")
    if (type(rank) is not int or not 0 <= rank < 8 or not local.is_cuda or
            local.dtype != torch.bfloat16 or not local.is_contiguous() or local.ndim != 2 or
            not packets.is_cuda or packets.device != local.device or packets.dtype != torch.uint8 or
            not packets.is_contiguous() or local.shape[0] <= 0 or local.shape[1] <= 0 or
            local.shape[1] % 128 or tuple(packets.shape) != (7, local.numel()//128*136)):
        raise ValueError('Basic receiver requires seven source-ordered packets and contiguous local BF16')
    result = torch.empty_like(local)
    lib = load_library()
    with torch.cuda.device(local.device):
        fn = lib.basic_quant_decode_reduce if policy == 'basic' else lib.mav2_decode_reduce
        code = fn(packets.data_ptr(), local.data_ptr(), result.data_ptr(),
            rank, local.numel(), torch.cuda.current_stream(local.device).cuda_stream)
    if code:
        raise RuntimeError(lib.baseline_taco_error(code).decode())
    return result


class TensorTacoRS:
    """Adapter-compatible name; implementation is the pinned CUDA codec."""
    def __init__(self, group, m, n):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        assert self.world == 8 and m % (128*self.world) == 0 and n % 128 == 0
        self.m, self.n, self.rows = m, n, m//self.world
        self.packet_bytes = self.rows*(n//128)*136
        self.splits = [0 if r == self.rank else self.packet_bytes for r in range(self.world)]
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.calls = 0
        self._diagnostic_packet = None
        self._diagnostic_enabled = os.environ.get("E2E_MODE") in ("smoke", "profile")
        self.policy = os.environ['E2E_POLICY']
        if self.policy not in POLICIES:
            raise ValueError('Unknown receiver-fusion policy')
        self.selection_sha256 = self.selection_cpu = None
        self.wire_bytes_per_rank = (self.world-1)*self.packet_bytes
        self.bf16_wire_bytes_per_rank = (self.world-1)*self.rows*n*2
        load_library()

    def scope(self, name):
        return torch.profiler.record_function('megatron_taco.'+name) if self.profile else nullcontext()

    def reduce(self, partial):
        self.calls += 1
        shape = partial.shape
        assert partial.is_cuda and partial.dtype == torch.bfloat16 and partial.numel() == self.m*self.n
        pieces = partial.reshape(self.world, self.rows, self.n)
        with self.scope('encode'):
            remote = torch.cat([pieces[d] for d in range(self.world) if d != self.rank], dim=0)
            packets = pack_rows(remote, self.world-1, self.policy)
            if self._diagnostic_enabled and self._diagnostic_packet is None:
                self._diagnostic_packet = packets
        with self.scope('exchange'):
            received = torch.empty_like(packets)
            dist.all_to_all_single(received.view(-1), packets.view(-1),
                output_split_sizes=self.splits, input_split_sizes=self.splits, group=self.group)
        with self.scope('decode_reduce'):
            return decode_reduce(received, pieces[self.rank], self.rank, self.policy).reshape(
                shape[0]//self.world, *shape[1:])

    def transform_diagnostics(self):
        if not self._diagnostic_enabled: return None
        if self._diagnostic_packet is None: raise RuntimeError('No observed projection packet')
        record = packet_diagnostics(self._diagnostic_packet, self.rows * self.n)
        if self.policy == 'd8' and (record['d8'] != record['total'] or record['h8'] != 0):
            raise RuntimeError('Formal D8 arm used a different representation')
        record.update(policy=self.policy, calls=self.calls, sampled_calls=1,
                      sampling='first real projection call; diagnostics read after window',
                      quantized_remote_fraction=1.0, quality_selection=self.policy == 'mav2',
                      h8_fallback_allowed=self.policy == 'mav2')
        return record


def packet_diagnostics(packets, elements):
    groups = elements // 128
    qs = packets[:, elements:elements+groups*4].contiguous().view(torch.float32)
    tags = packets[:, elements+groups*4:].contiguous().view(torch.float32)
    finite = torch.isfinite(qs) & torch.isfinite(tags)
    valid = finite & (qs > 0) & ((tags == DIRECT_TAG) | (tags > 0))
    if not bool(valid.all()): raise RuntimeError('Invalid D8/H8 scale or mode tag')
    direct = int((tags == DIRECT_TAG).sum())
    return dict(total=tags.numel(), d8=direct, h8=tags.numel()-direct, nonfinite=0,
                invalid_tags=0, bytes_per_group=136, selector=dict(relative_l2=D8_REL_L2,
                max_abs_over_amax=D8_MAX_ABS), d8_fraction=direct/tags.numel())



class QuantizedRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, codec):
        ctx.group = codec.group
        return codec.reduce(partial)

    @staticmethod
    def backward(ctx, grad_output):
        from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
        return _gather_along_first_dim(grad_output.contiguous(), ctx.group), None
