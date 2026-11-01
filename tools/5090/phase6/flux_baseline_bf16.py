"""Unquantized GemmOnly output followed by NCCL BF16 Reduce-Scatter.

The adapter-compatible class name does not imply that a codec is present.
No codec library, encoder, decoder, packet buffer or all-to-all is used.
"""
from contextlib import nullcontext
import os
import torch
import torch.distributed as dist


def loaded_library():
    return None


class TensorTacoRS:
    def __init__(self, group, m, n):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if self.world != 8 or (m, n) != (2048, 4096) or dist.get_backend(group) != 'nccl':
            raise ValueError('BF16 baseline requires exact NCCL TP8 shapes')
        self.m, self.n, self.rows = m, n, m//8
        self.calls = 0
        self.selection_sha256 = self.selection_cpu = None
        # Logical remote contribution volume; this is not measured NCCL bus traffic.
        self.wire_bytes_per_rank = self.bf16_wire_bytes_per_rank = 7*self.rows*n*2
        self.profile = os.environ.get('E2E_MODE') == 'profile'

    def reduce(self, partial):
        if partial.dtype != torch.bfloat16 or not partial.is_cuda or not partial.is_contiguous() \
                or tuple(partial.shape) != (self.m, self.n):
            raise ValueError('Expected contiguous full BF16 partial output')
        output = torch.empty((self.rows, self.n), dtype=partial.dtype, device=partial.device)
        with torch.profiler.record_function('flux_baseline.bf16_reduce_scatter') if self.profile else nullcontext():
            dist.reduce_scatter_tensor(output, partial, group=self.group)
        self.calls += 1
        return output

    def state(self):
        return dict(quantization='none', collective='reduce_scatter_tensor', dtype='bfloat16',
                    calls=self.calls, encode_calls=0, decode_calls=0, all_to_all_calls=0,
                    owns_output=True, logical_remote_bytes=self.wire_bytes_per_rank)


class QuantizedRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, codec):
        ctx.group = codec.group
        return codec.reduce(partial)

    @staticmethod
    def backward(ctx, grad_output):
        from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
        return _gather_along_first_dim(grad_output.contiguous(), ctx.group), None
