"""Standalone upstream TACO CUDA codec + NCCL All-to-All + BF16 reduction.

Frozen as megatron_taco.py for both baselines. A/B/C never load this library.
No JIT, GEMM fusion, warp rewrite, persistent packet pool or fused reduction.
"""
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

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


def pack_rows(x, destinations):
    if (not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.ndim != 2 or
            destinations <= 0 or x.shape[0] % destinations or x.shape[1] % 128):
        raise ValueError('Standalone upstream codec requires contiguous BF16 full H128 groups')
    elements = x.numel() // destinations
    packets = torch.empty((destinations, elements + elements//128*8), dtype=torch.uint8, device=x.device)
    _call('baseline_taco_encode', x, packets, destinations, elements)
    return packets


def unpack_rows(packets, rows, n):
    elements = rows*n
    if (not packets.is_cuda or packets.dtype != torch.uint8 or not packets.is_contiguous() or
            packets.ndim != 2 or rows <= 0 or n <= 0 or n % 128 or
            packets.shape[1] != elements + elements//128*8):
        raise ValueError('Invalid upstream TACO packet shape')
    output = torch.empty((packets.shape[0], rows, n), dtype=torch.bfloat16, device=packets.device)
    _call('baseline_taco_decode', packets, output, packets.shape[0], elements)
    return output


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
            packets = pack_rows(remote, self.world-1)
        with self.scope('exchange'):
            received = torch.empty_like(packets)
            dist.all_to_all_single(received.view(-1), packets.view(-1),
                output_split_sizes=self.splits, input_split_sizes=self.splits, group=self.group)
        with self.scope('decode_reduce'):
            decoded = unpack_rows(received, self.rows, self.n)
            result = torch.zeros_like(pieces[self.rank])
            for i in range(1, self.world+1):
                src = (self.rank+i) % self.world
                value = pieces[src] if src == self.rank else decoded[src-int(src > self.rank)]
                result = (result+value).to(torch.bfloat16)
        return result.reshape(shape[0]//self.world, *shape[1:])


class QuantizedRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, codec):
        ctx.group = codec.group
        return codec.reduce(partial)

    @staticmethod
    def backward(ctx, grad_output):
        from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
        return _gather_along_first_dim(grad_output.contiguous(), ctx.group), None
