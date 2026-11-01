"""Two OP3 single-exchange paths; immutable instance and independent outputs."""
from contextlib import nullcontext
import ctypes
import os
import threading

import torch
import torch.distributed as dist
import op3_reference_codec as reference

ARMS = ('op3-r', 'op3-improved')
PROTOCOL = 'rtx5090-op3-direct-h128-v1'
_BOUND = False
_LOCKS = {}
_POISONED = set()
_RETAINED = []


def load_library(record=None):
    global _BOUND
    lib = reference.load_library(record)
    if not _BOUND:
        lib.op3_r_decode.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_void_p]
        lib.op3_r_decode.restype = ctypes.c_int
        lib.op3_encode_warp.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*2 + [ctypes.c_void_p]
        lib.op3_encode_warp.restype = ctypes.c_int
        _BOUND = True
    return lib


loaded_library = reference.loaded_library
pack_rows = reference.pack_rows
unpack_rows = reference.unpack_rows
decode_reduce = reference.decode_reduce


def chunk_plan(total_groups, arm):
    if arm not in ARMS or total_groups != 8192:
        raise ValueError('OP3 requires one full TP8 2048x4096 projection')
    return [dict(index=0, begin=0, count=8192, raw_groups=0, peer_bytes=8192*136)]


def _shape(local, rank, begin, count, raw_groups, arm):
    if (arm not in ARMS or local.shape != (256, 4096) or begin != 0 or count != 8192
            or raw_groups != 0 or type(rank) is not int or not 0 <= rank < 8):
        raise ValueError('OP3 requires its frozen full projection and whole quantized packet')


def _consumer(local, bias, residual):
    if (bias is None) != (residual is None):
        raise ValueError('Consumer bias and residual must be supplied together')
    if bias is None:
        return
    if (bias.shape != (4096,) or residual.shape != local.shape
            or any(x.device != local.device or x.dtype != torch.bfloat16
                   or not x.is_contiguous() or x.data_ptr() % 8 for x in (bias, residual))):
        raise ValueError('Consumer requires contiguous BF16 bias[4096] and residual[256,4096]')


def _check(code):
    if code:
        raise RuntimeError(load_library().baseline_taco_error(code).decode())


def encode_chunk(partial, wire, rank, begin, count, raw_groups, arm=None):
    arm = arm or os.environ['E2E_POLICY']
    if partial.shape != (2048, 4096) or not partial.is_contiguous():
        raise ValueError('OP3 encoder requires contiguous BF16 partial[2048,4096]')
    local = partial.reshape(8, 256, 4096)[rank]
    _shape(local, rank, begin, count, raw_groups, arm)
    reference._validate_chunk(wire, local, rank, begin, count, raw_groups)
    if arm == ARMS[0]:
        reference.encode_chunk(partial, wire, rank, begin, count, raw_groups)
    else:
        _check(load_library().op3_encode_warp(partial.data_ptr(), wire.data_ptr(), rank, count,
            torch.cuda.current_stream(partial.device).cuda_stream))


def decode_chunk(wire, local, output, rank, begin, count, raw_groups,
                 arm=None, bias=None, residual=None):
    arm = arm or os.environ['E2E_POLICY']
    _shape(local, rank, begin, count, raw_groups, arm)
    reference._validate_chunk(wire, local, rank, begin, count, raw_groups)
    _consumer(local, bias, residual)
    if (output.dtype != local.dtype or output.device != local.device
            or output.shape != local.shape or not output.is_contiguous()
            or output.data_ptr() == local.data_ptr()
            or (residual is not None and output.data_ptr() == residual.data_ptr())):
        raise ValueError('OP3 decoder requires an independently owned output')
    if arm == ARMS[1]:
        if bias is not None:
            raise ValueError('The independent encoder candidate does not fuse the consumer')
        reference.decode_chunk(wire, local, output, rank, begin, count, raw_groups)
    else:
        _check(load_library().op3_r_decode(wire.data_ptr(), local.data_ptr(), output.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            residual.data_ptr() if residual is not None else None, rank, count,
            torch.cuda.current_stream(local.device).cuda_stream))


class TensorTacoRS:
    """Persistent packets, one stream and one NCCL collective per projection.

    R consumes the full completed exchange via packed-source prefetch. Its
    optional consumer arguments implement CR3's two ordered BF16 additions.
    The independent candidate changes the encoder only. Each call owns a fresh
    output, so later submissions cannot overwrite a saved autograd activation.
    """
    def __init__(self, group, m, n):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.arm = os.environ['E2E_POLICY']
        if self.world != 8 or (m, n) != (2048, 4096) or self.arm not in ARMS:
            raise ValueError('Unsupported OP3 TP8 projection')
        self.m, self.n, self.rows = m, n, m//8
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.chunks = chunk_plan(8192, self.arm)
        self.tx = [torch.empty(7*8192*136, dtype=torch.uint8, device=self.device)]
        self.rx = [torch.empty_like(self.tx[0])]
        self.splits = [[0 if peer == self.rank else 8192*136 for peer in range(8)]]
        self.wire_bytes_per_rank = 7*8192*136
        self.bf16_wire_bytes_per_rank = 7*256*4096*2
        self.selection_cpu = self.selection_sha256 = None
        self.calls = self.encode_calls = self.exchange_calls = self.decode_calls = 0
        self.wait_calls = self.fused_consumer_calls = self.projection_calls = 0
        self.join_calls = 0
        self.arrival = False
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self._caller = None
        self._failed = self._pending = False
        self._pipeline_key = (id(group), self.device.index)
        self._submit_lock = _LOCKS.setdefault(self._pipeline_key, threading.Lock())
        load_library()

    def scope(self, name):
        return torch.profiler.record_function('op3.'+name) if self.profile else nullcontext()

    def reduce(self, partial, bias=None, residual=None):
        if not self._submit_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent OP3 submissions to one process group are unsupported')
        try:
            return self._reduce(partial, bias, residual)
        finally:
            self._submit_lock.release()

    def _reduce(self, partial, bias, residual):
        if (self._failed or self._pipeline_key in _POISONED
                or partial.dtype != torch.bfloat16 or partial.device != self.device
                or partial.shape != (self.m, self.n) or not partial.is_contiguous()):
            raise ValueError('Wrong OP3 input or poisoned submission')
        caller = torch.cuda.current_stream(self.device)
        if self._caller is None:
            self._caller = caller.cuda_stream
        if self._caller != caller.cuda_stream:
            raise ValueError('Persistent OP3 packets require the original caller stream')
        local = partial.reshape(8, self.rows, self.n)[self.rank]
        _consumer(local, bias, residual)
        if bias is not None and self.arm != ARMS[0]:
            raise ValueError('Only the R arm enables the fused consumer')
        output = torch.empty_like(local)
        self.calls += 1
        self._pending = True
        try:
            with self.scope('encode.0'):
                encode_chunk(partial, self.tx[0], self.rank, 0, 8192, 0, arm=self.arm)
                self.encode_calls += 1
            with self.scope('exchange.0'):
                work = dist.all_to_all_single(self.rx[0], self.tx[0],
                    output_split_sizes=self.splits[0], input_split_sizes=self.splits[0],
                    group=self.group, async_op=True)
                # Work.wait establishes completion on the current CUDA stream.
                # The R loads must never observe an enqueue-only NCCL event.
                work.wait()
                self.wait_calls += 1
                self.exchange_calls += 1
            with self.scope('decode.0'):
                decode_chunk(self.rx[0], local, output, self.rank, 0, 8192, 0,
                             arm=self.arm, bias=bias, residual=residual)
                self.decode_calls += 1
                self.fused_consumer_calls += int(bias is not None)
                self.projection_calls += int(bias is None)
            self._pending = False
            return output
        except BaseException:
            self._failed = True
            _POISONED.add(self._pipeline_key)
            _RETAINED.append((self, partial, output, bias, residual))
            raise

    def state(self):
        return dict(protocol=PROTOCOL, arm=self.arm, calls=self.calls, chunks=self.chunks,
            transport='NCCL-byte-AllToAll', arrival_trigger='whole-exchange', selection='all-remote',
            wire_bytes_per_rank=self.wire_bytes_per_rank, bf16_wire_bytes_per_rank=self.bf16_wire_bytes_per_rank,
            encode_calls=self.encode_calls, exchange_calls=self.exchange_calls, decode_calls=self.decode_calls,
            fused_consumer_calls=self.fused_consumer_calls, projection_calls=self.projection_calls,
            join_calls=0, wait_calls=self.wait_calls, independent_output=True,
            fixed_source_order='rank+1-through-rank', pending_exchange=self._pending,
            device=self.device.index, caller_stream=self._caller, comm_stream=self._caller,
            decode_stream=self._caller, send_split_bytes=self.splits, receive_split_bytes=self.splits,
            tx_ptrs=[x.data_ptr() for x in self.tx], rx_ptrs=[x.data_ptr() for x in self.rx],
            packet_layout='H128-AoS-136', calibration_used=False,
            encoder='upstream-cta128' if self.arm == ARMS[0] else 'warp-owned-four-groups',
            decoder='r-packed-source-prefetch' if self.arm == ARMS[0] else 'basic-ring-bf16-h128-direct-v1',
            consumer_fusion=self.arm == ARMS[0], original_peer_overlap_preserved=False,
            failed=self._failed)

    def debug_packets(self):
        return [dict(tx=self.tx[0].view(7, 8192*136), rx=self.rx[0].view(7, 8192*136))]

    def close(self):
        torch.cuda.synchronize(self.device)


class QuantizedRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, codec):
        ctx.group = codec.group
        return codec.reduce(partial)

    @staticmethod
    def backward(ctx, grad_output):
        from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
        return _gather_along_first_dim(grad_output.contiguous(), ctx.group), None
