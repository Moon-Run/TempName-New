"""5090 direct H128 packets, completion-driven chunk decoding, sparse tail coding."""
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading

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
    lib.basic_quant_decode_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_int, ctypes.c_int64, ctypes.c_void_p]
    lib.basic_quant_decode_reduce.restype = ctypes.c_int
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


def decode_reduce(packets, local, rank):
    if (type(rank) is not int or not 0 <= rank < 8 or not local.is_cuda or
            local.dtype != torch.bfloat16 or not local.is_contiguous() or local.ndim != 2 or
            not packets.is_cuda or packets.device != local.device or packets.dtype != torch.uint8 or
            not packets.is_contiguous() or local.shape[0] <= 0 or local.shape[1] <= 0 or
            local.shape[1] % 128 or tuple(packets.shape) != (7, local.numel()//128*136)):
        raise ValueError('Basic receiver requires seven source-ordered packets and contiguous local BF16')
    result = torch.empty_like(local)
    lib = load_library()
    with torch.cuda.device(local.device):
        code = lib.basic_quant_decode_reduce(packets.data_ptr(), local.data_ptr(), result.data_ptr(),
            rank, local.numel(), torch.cuda.current_stream(local.device).cuda_stream)
    if code:
        raise RuntimeError(lib.baseline_taco_error(code).decode())
    return result


ARMS = ('flux-basic-interleaved', 'flux-basic-arrival', 'flux-basic-selective')
_PIPELINES = {}
_POISONED_PIPELINES = set()
_POISONED_REFERENCES = []
_SUBMISSION_LOCKS = {}
_ABC_BOUND = False
_upstream_load_library = load_library


def load_library(record=None):
    global _ABC_BOUND
    lib = _upstream_load_library(record)
    if not _ABC_BOUND:
        ints = [ctypes.c_int] * 5
        lib.op1_abc_encode.argtypes = [ctypes.c_void_p] * 2 + ints + [ctypes.c_void_p]
        lib.op1_abc_decode.argtypes = [ctypes.c_void_p] * 3 + ints + [ctypes.c_void_p]
        lib.op1_abc_encode.restype = lib.op1_abc_decode.restype = ctypes.c_int
        _ABC_BOUND = True
    return lib


def chunk_plan(total_groups, arm):
    if arm not in ARMS or type(total_groups) is not int or total_groups < 64 or total_groups % 64:
        raise ValueError('Expected a supported arm and a multiple of 64 H128 output groups')
    chunks = 1 if arm == ARMS[0] else 4
    count = total_groups // chunks
    first_quantized = total_groups - total_groups // 64 if arm == ARMS[2] else 0
    result = []
    for index in range(chunks):
        begin = index * count
        raw = min(count, max(0, first_quantized - begin))
        result.append(dict(index=index, begin=begin, count=count, raw_groups=raw,
                           peer_bytes=raw*256+(count-raw)*136))
    return result


def _validate_chunk(wire, local, rank, begin, count, raw_groups):
    if (type(rank) is not int or not 0 <= rank < 8 or local.dtype != torch.bfloat16
            or not local.is_cuda or not local.is_contiguous() or local.numel() % 128
            or not all(type(x) is int for x in (begin, count, raw_groups))
            or not 0 <= begin < local.numel()//128 or count <= 0
            or begin+count > local.numel()//128 or not 0 <= raw_groups <= count
            or wire.dtype != torch.uint8 or wire.device != local.device or not wire.is_contiguous()
            or wire.numel() != 7*(raw_groups*256+(count-raw_groups)*136)):
        raise ValueError('Invalid direct-wire H128 chunk')


def encode_chunk(partial, wire, rank, begin, count, raw_groups):
    if partial.ndim != 2 or partial.shape[0] % 8 or not partial.is_contiguous():
        raise ValueError('Encode requires the full contiguous TP8 partial')
    local = partial.reshape(8, -1)[rank]
    _validate_chunk(wire, local, rank, begin, count, raw_groups)
    code = load_library().op1_abc_encode(partial.data_ptr(), wire.data_ptr(), rank,
        local.numel()//128, begin, count, raw_groups, torch.cuda.current_stream(partial.device).cuda_stream)
    if code:
        raise RuntimeError(load_library().baseline_taco_error(code).decode())


def decode_chunk(wire, local, output, rank, begin, count, raw_groups):
    _validate_chunk(wire, local, rank, begin, count, raw_groups)
    if (output.dtype != local.dtype or output.device != local.device
            or output.shape != local.shape or not output.is_contiguous()
            or output.data_ptr() == local.data_ptr()):
        raise ValueError('Decode requires independently owned contiguous output')
    code = load_library().op1_abc_decode(wire.data_ptr(), local.data_ptr(), output.data_ptr(), rank,
        local.numel()//128, begin, count, raw_groups, torch.cuda.current_stream(local.device).cuda_stream)
    if code:
        raise RuntimeError(load_library().baseline_taco_error(code).decode())


class TensorTacoRS:
    """Same H128 arithmetic; completed NCCL row chunks become decodable immediately.

    A sends the complete tensor once. B/C use four ordered NCCL calls and CUDA
    events, never cross-device pointers or host polling. C compresses just the
    final 1/64 H128 groups of each destination; its mask is deterministic.
    """
    def __init__(self, group, m, n):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.arm = os.environ['E2E_POLICY']
        if self.world != 8 or m % (8*128) or n % 128 or self.arm not in ARMS:
            raise ValueError('Unsupported TP8 ABC projection')
        self.m, self.n, self.rows = m, n, m//8
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.chunks = chunk_plan(self.rows*n//128, self.arm)
        self.tx = [torch.empty(7*c['peer_bytes'], dtype=torch.uint8, device=self.device) for c in self.chunks]
        self.rx = [torch.empty_like(t) for t in self.tx]
        self.splits = [[0 if peer == self.rank else c['peer_bytes'] for peer in range(8)] for c in self.chunks]
        self.calls = self.encode_calls = self.exchange_calls = self.decode_calls = self.join_calls = self.wait_calls = 0
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.arrival = self.arm != ARMS[0]
        self.selection_cpu = None
        self.selection_sha256 = None
        self.wire_bytes_per_rank = sum(t.numel() for t in self.tx)
        self.bf16_wire_bytes_per_rank = 7*self.rows*n*2
        self._caller = None
        self._failed = False
        self._pending = False
        self._pipeline_key = (id(group), self.device.index)
        self._submit_lock = _SUBMISSION_LOCKS.setdefault(self._pipeline_key, threading.Lock())
        if self.arrival:
            key = self._pipeline_key
            if key not in _PIPELINES:
                _PIPELINES[key] = (group, torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device))
            _, self.comm_stream, self.decode_stream = _PIPELINES[key]
            self.encoded = [torch.cuda.Event() for _ in self.chunks]
            self.received = [torch.cuda.Event() for _ in self.chunks]
            self.finished = torch.cuda.Event()
        load_library()

    def scope(self, name):
        return torch.profiler.record_function('op1_abc.'+name) if self.profile else nullcontext()

    def reduce(self, partial):
        if not self._submit_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent ABC submissions to one process group are unsupported')
        try:
            return self._reduce(partial)
        finally:
            self._submit_lock.release()

    def _reduce(self, partial):
        if (self._failed or self._pipeline_key in _POISONED_PIPELINES
                or partial.dtype != torch.bfloat16 or partial.device != self.device
                or partial.shape != (self.m, self.n) or not partial.is_contiguous()):
            raise ValueError('Wrong ABC input or poisoned codec')
        caller = torch.cuda.current_stream(self.device)
        if self._caller is None:
            self._caller = caller.cuda_stream
        if self._caller != caller.cuda_stream:
            raise ValueError('Persistent ABC buffers require the original caller stream')
        self.calls += 1
        local = partial.reshape(8, self.rows, self.n)[self.rank]
        output = torch.empty_like(local)
        self._pending = True
        try:
            for i, chunk in enumerate(self.chunks):
                args = (self.rank, chunk['begin'], chunk['count'], chunk['raw_groups'])
                with self.scope(f'encode.{i}'):
                    encode_chunk(partial, self.tx[i], *args)
                    self.encode_calls += 1
                    if self.arrival:
                        self.encoded[i].record(caller)
                with self.scope(f'exchange.{i}'):
                    stream = self.comm_stream if self.arrival else caller
                    with torch.cuda.stream(stream):
                        if self.arrival:
                            stream.wait_event(self.encoded[i])
                        work = dist.all_to_all_single(self.rx[i], self.tx[i],
                            output_split_sizes=self.splits[i], input_split_sizes=self.splits[i],
                            group=self.group, async_op=True)
                        # This inserts the NCCL completion dependency on this
                        # CUDA stream; an enqueue-time event alone is insufficient.
                        work.wait()
                        self.wait_calls += 1
                        self.exchange_calls += 1
                        if self.arrival:
                            self.received[i].record(stream)
                with self.scope(f'decode.{i}'):
                    stream = self.decode_stream if self.arrival else caller
                    with torch.cuda.stream(stream):
                        if self.arrival:
                            stream.wait_event(self.received[i])
                        decode_chunk(self.rx[i], local, output, *args)
                        self.decode_calls += 1
            if self.arrival:
                with self.scope('join'):
                    self.finished.record(self.decode_stream)
                    caller.wait_event(self.finished)
                    self.join_calls += 1
            self._pending = False
            return output
        except BaseException:
            self._failed = True
            _POISONED_PIPELINES.add(self._pipeline_key)
            # Side-stream kernels may still reference these tensors on failure.
            self._failed_refs = (partial, output)
            _POISONED_REFERENCES.append((self, partial, output))
            raise

    def state(self):
        return dict(arm=self.arm, calls=self.calls, chunks=self.chunks,
            transport='NCCL-byte-AllToAll', arrival_trigger='NCCL-work-wait-then-CUDA-event' if self.arrival else 'whole-exchange',
            selection='last-1/64-H128-groups' if self.arm == ARMS[2] else 'all-remote',
            wire_bytes_per_rank=self.wire_bytes_per_rank, bf16_wire_bytes_per_rank=self.bf16_wire_bytes_per_rank,
            encode_calls=self.encode_calls, exchange_calls=self.exchange_calls, decode_calls=self.decode_calls,
            join_calls=self.join_calls, independent_output=True, fixed_source_order='rank+1-through-rank',
            wait_calls=self.wait_calls, pending_exchange=self._pending,
            device=self.device.index, caller_stream=self._caller,
            comm_stream=self.comm_stream.cuda_stream if self.arrival else self._caller,
            decode_stream=self.decode_stream.cuda_stream if self.arrival else self._caller,
            send_split_bytes=self.splits, receive_split_bytes=self.splits,
            tx_ptrs=[x.data_ptr() for x in self.tx], rx_ptrs=[x.data_ptr() for x in self.rx],
            packet_layout='H128-AoS-136-or-BF16-256', calibration_used=False,
            original_peer_overlap_preserved=False, failed=self._failed)

    def debug_packets(self):
        """Probe-only view; the caller synchronizes before reading on the CPU."""
        return [dict(tx=tx.view(7, c['peer_bytes']), rx=rx.view(7, c['peer_bytes']))
                for c, tx, rx in zip(self.chunks, self.tx, self.rx)]

    def close(self):
        # Explicit probe teardown only; full-model timing never calls close.
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
