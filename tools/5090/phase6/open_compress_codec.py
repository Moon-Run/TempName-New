# SPDX-License-Identifier: Apache-2.0
"""Independent block-absmax INT8 codec, NCCL exchange, ordered BF16 reduction.

The adapter-compatible TensorTacoRS name is only an interface inherited from
Flux's experiment harness. No TACO library, arithmetic or rotation is used.
Finite, contiguous BF16 inputs are required. NaN/Inf checks belong to smoke
validation, never an implicit host synchronization in a timed projection.
"""
from contextlib import nullcontext
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading

try:
    import torch
    import torch.distributed as dist
except ImportError:  # Protocol/reference unit tests require only stdlib.
    torch = dist = None

PROTOCOL = 'open-block-int8-soa128-v1'
ARMS = ('flux-int8', 'flux-int8-interleaved')
BLOCK = 128
ELEMENTS = 256 * 4096
GROUPS = ELEMENTS // BLOCK
PEER_BYTES = ELEMENTS + GROUPS * 4
WIRE_BYTES = 7 * PEER_BYTES
BF16_REMOTE_BYTES = 7 * ELEMENTS * 2
_LIB = _IDENTITY = None
_LOCKS = {}
_POISONED = set()
_RETAINED = []


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def reference_block(values):
    """Small stdlib FP32 oracle: return (signed INT8 values, FP32 scale)."""
    if len(values) != BLOCK or any(not math.isfinite(v) for v in values):
        raise ValueError('One finite 128-element block is required')
    values = [_f32(v) for v in values]
    maximum = max(map(abs, values))
    scale = _f32(maximum / 127) if maximum else 1.0
    return [max(-127, min(127, round(_f32(v / scale)))) for v in values], scale


def bf16_round(value):
    """Independent round-to-nearest-even BF16 scalar reference."""
    bits = struct.unpack('<I', struct.pack('<f', _f32(value)))[0]
    if bits & 0x7f800000 != 0x7f800000:
        bits += 0x7fff + ((bits >> 16) & 1)
    return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]


def reference_block_roundtrip(values):
    quantized, scale = reference_block(values)
    return [bf16_round(_f32(q * scale)) for q in quantized]


def splits_for_rank(rank):
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('TP8 rank must be an integer in [0,8)')
    return [0 if peer == rank else PEER_BYTES for peer in range(8)]


def load_library(record=None):
    global _LIB, _IDENTITY
    if record is None and _LIB is not None:
        return _LIB
    if record is None:
        config = Path(__file__).with_name('config.json')
        if config.is_file():
            record = json.loads(config.read_text())['codec_library']
        else:
            path = Path(os.environ['OPEN_COMPRESS_LIBRARY']).resolve()
            record = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path = Path(record['path']).resolve()
    identity = dict(path=str(path), sha256=record['sha256'])
    if _LIB is not None:
        if identity != _IDENTITY:
            raise RuntimeError('Cannot replace the loaded INT8 codec library')
        return _LIB
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise RuntimeError('INT8 codec binary hash mismatch')
    lib = ctypes.CDLL(str(path))
    lib.oc_abi_version.argtypes, lib.oc_abi_version.restype = [], ctypes.c_int
    if lib.oc_abi_version() != 1:
        raise RuntimeError('Unsupported INT8 codec ABI')
    lib.oc_error.argtypes, lib.oc_error.restype = [ctypes.c_int], ctypes.c_char_p
    lib.oc_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.oc_decode.argtypes = [ctypes.c_void_p] * 3
    lib.oc_decode_reduce.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
    for name in ('oc_encode', 'oc_decode', 'oc_decode_reduce'):
        getattr(lib, name).restype = ctypes.c_int
    _LIB, _IDENTITY = lib, identity
    return lib


def loaded_library():
    return _IDENTITY


def _tensor(value, shape, dtype, device=None):
    if (not value.is_cuda or value.dtype != dtype or tuple(value.shape) != shape
            or not value.is_contiguous() or (device is not None and value.device != device)):
        raise ValueError('Invalid contiguous CUDA tensor shape/dtype/device')


def _wire(wire, device=None):
    if (not wire.is_cuda or wire.dtype != torch.uint8 or wire.numel() != WIRE_BYTES
            or not wire.is_contiguous() or wire.data_ptr() % 4
            or (device is not None and wire.device != device)):
        raise ValueError('Expected aligned contiguous uint8 remote packet buffer')


def _call(name, device, *arguments):
    lib = load_library()
    with torch.cuda.device(device):
        result = getattr(lib, name)(*arguments, torch.cuda.current_stream(device).cuda_stream)
    if result:
        raise RuntimeError(lib.oc_error(result).decode())


def encode(partial, wire=None, rank=None):
    rank = dist.get_rank() if rank is None else rank
    splits_for_rank(rank)
    _tensor(partial, (2048, 4096), torch.bfloat16)
    if wire is None:
        wire = torch.empty(WIRE_BYTES, dtype=torch.uint8, device=partial.device)
    _wire(wire, partial.device)
    _call('oc_encode', partial.device, partial.data_ptr(), wire.data_ptr(), rank)
    return wire


def decode(wire, output=None):
    _wire(wire)
    if output is None:
        output = torch.empty((7, 256, 4096), dtype=torch.bfloat16, device=wire.device)
    _tensor(output, (7, 256, 4096), torch.bfloat16, wire.device)
    _call('oc_decode', wire.device, wire.data_ptr(), output.data_ptr())
    return output


def decode_reduce(wire, local, output=None, rank=None):
    rank = dist.get_rank() if rank is None else rank
    splits_for_rank(rank)
    if tuple(local.shape) == (2048, 4096):
        local = local.view(8, 256, 4096)[rank]
    _tensor(local, (256, 4096), torch.bfloat16)
    _wire(wire, local.device)
    if output is None:
        output = torch.empty_like(local)
    _tensor(output, (256, 4096), torch.bfloat16, local.device)
    if output.untyped_storage().data_ptr() in (
            local.untyped_storage().data_ptr(), wire.untyped_storage().data_ptr()):
        raise ValueError('Decoder output must own independent storage')
    _call('oc_decode_reduce', local.device, wire.data_ptr(), local.data_ptr(), output.data_ptr(), rank)
    return output


def reference_quantize(piece):
    """Independent Torch FP32 oracle, arbitrary whole 128-value groups."""
    if piece.dtype != torch.bfloat16 or not piece.is_contiguous() or piece.numel() % BLOCK:
        raise ValueError('Reference requires contiguous BF16 whole blocks')
    groups = piece.float().view(-1, BLOCK)
    maxima = groups.abs().amax(dim=1)
    # Use tensor/tensor division: ATen may turn division by a Python scalar
    # into multiplication by a rounded reciprocal, changing scale bytes.
    scales = torch.where(maxima == 0, torch.ones_like(maxima),
                         maxima / torch.full_like(maxima, 127.0))
    quantized = torch.round(groups / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized, scales


def reference_roundtrip(piece):
    quantized, scales = reference_quantize(piece)
    return (quantized.float() * scales[:, None]).to(torch.bfloat16).reshape(piece.shape)


def reference_packets(partial, rank):
    """Torch oracle for exact outgoing peer-order bytes, including FP32 scales."""
    splits_for_rank(rank)
    _tensor(partial, (2048, 4096), torch.bfloat16)
    packets = []
    for destination in range(8):
        if destination == rank:
            continue
        quantized, scales = reference_quantize(partial.view(8, 256, 4096)[destination])
        packets.append(torch.cat((quantized.reshape(-1).view(torch.uint8), scales.view(torch.uint8))))
    return torch.stack(packets).view(-1)


def reference_decode(wire):
    _wire(wire)
    packets = wire.view(7, PEER_BYTES)
    outputs = []
    for slot in range(7):
        quantized = packets[slot, :ELEMENTS].view(torch.int8).view(GROUPS, BLOCK)
        scales = packets[slot, ELEMENTS:].view(torch.float32)
        outputs.append((quantized.float() * scales[:, None]).to(torch.bfloat16).view(256, 4096))
    return torch.stack(outputs)


def reference_reduce(wire, local, rank):
    if tuple(local.shape) == (2048, 4096):
        local = local.view(8, 256, 4096)[rank]
    decoded = reference_decode(wire)
    output = torch.zeros_like(local)
    for source in range(8):
        value = local if source == rank else decoded[source - int(source > rank)]
        output = (output.float() + value.float()).to(torch.bfloat16)
    return output


class TensorTacoRS:
    """Persistent packets on one stream, one NCCL exchange, owned output.

    The submission lock covers each process group on a device; one instance
    must keep its original CUDA stream. Work.wait() inserts the dependency
    required before decoder submission and subsequent packet reuse.
    """
    owns_output = owns_configuration = True
    selection_cpu = selection_sha256 = None

    def __init__(self, group, m, n):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.arm = os.environ['E2E_POLICY']
        if self.world != 8 or (m, n) != (2048, 4096) or self.arm not in ARMS or dist.get_backend(group) != 'nccl':
            raise ValueError('INT8 compression requires its TP8 NCCL projection and policy')
        self.m, self.n, self.rows = m, n, 256
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.tx = torch.empty(WIRE_BYTES, dtype=torch.uint8, device=self.device)
        self.rx = torch.empty_like(self.tx)
        self.splits = splits_for_rank(self.rank)
        self.packet_bytes = PEER_BYTES
        self.wire_bytes_per_rank = WIRE_BYTES
        self.bf16_wire_bytes_per_rank = BF16_REMOTE_BYTES
        self.calls = self.encode_calls = self.decode_calls = self.all_to_all_calls = self.wait_calls = 0
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.check_finite = os.environ.get('E2E_MODE') == 'smoke'
        self.finite_input_checks = 0
        self._stream = None
        self._failed = self._pending = False
        self._key = (id(group), self.device.index)
        self._lock = _LOCKS.setdefault(self._key, threading.Lock())
        load_library()

    def scope(self, name):
        return torch.profiler.record_function('open_compress.' + name) if self.profile else nullcontext()

    def reduce(self, partial):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent codec submissions on a process group are unsupported')
        try:
            return self._reduce(partial)
        finally:
            self._lock.release()

    def _reduce(self, partial):
        _tensor(partial, (self.m, self.n), torch.bfloat16, self.device)
        if self.check_finite:
            if not bool(torch.isfinite(partial).all()):
                raise ValueError('INT8 smoke encountered a nonfinite GEMM partial')
            self.finite_input_checks += 1
        if self._failed or self._key in _POISONED:
            raise RuntimeError('INT8 process-group submission state is poisoned')
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if self._stream is None:
            self._stream = stream
        if stream != self._stream:
            raise ValueError('Persistent INT8 packets require their original caller CUDA stream')
        output = torch.empty((self.rows, self.n), dtype=torch.bfloat16, device=self.device)
        self._pending = True
        try:
            with self.scope('encode'):
                encode(partial, self.tx, self.rank)
                self.encode_calls += 1
            with self.scope('exchange'):
                work = dist.all_to_all_single(self.rx, self.tx, output_split_sizes=self.splits,
                    input_split_sizes=self.splits, group=self.group, async_op=True)
                self.all_to_all_calls += 1
                work.wait()
                self.wait_calls += 1
            with self.scope('decode_reduce'):
                decode_reduce(self.rx, partial, output, self.rank)
                self.decode_calls += 1
            self.calls += 1
            self._pending = False
            return output
        except BaseException:
            self._failed = True
            _POISONED.add(self._key)
            _RETAINED.append((self, partial, output))
            raise

    def state(self):
        return dict(protocol=PROTOCOL, arm=self.arm, calls=self.calls,
            encode_calls=self.encode_calls, decode_calls=self.decode_calls,
            all_to_all_calls=self.all_to_all_calls, exchange_calls=self.all_to_all_calls,
            wait_calls=self.wait_calls, logical_remote_bytes=WIRE_BYTES,
            finite_input_checks=self.finite_input_checks,
            wire_bytes_per_rank=WIRE_BYTES, bf16_wire_bytes_per_rank=BF16_REMOTE_BYTES,
            send_split_bytes=list(self.splits), receive_split_bytes=list(self.splits),
            owns_output=True, owns_configuration=True, quantization='all-remote',
            codec='symmetric-absmax-int8-block128-fp32-scale', packet_layout='peer-soa-q8-scale32',
            collective='all_to_all_single', dtype='uint8', source_order=list(range(8)),
            caller_stream=self._stream, current_stream=torch.cuda.current_stream(self.device).cuda_stream,
            packet_storage_bytes=2 * WIRE_BYTES, native_workspace_bytes=0,
            tx_ptr=self.tx.data_ptr(), rx_ptr=self.rx.data_ptr(),
            arrival=False, selective=False, calibration_used=False, peer_overlap_preserved=False,
            pending_exchange=self._pending, failed=self._failed)

    def debug_packets(self):
        return dict(tx=self.tx.view(7, PEER_BYTES), rx=self.rx.view(7, PEER_BYTES))

    def close(self):
        torch.cuda.synchronize(self.device)


if torch is not None:
    class QuantizedRS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, partial, codec):
            ctx.group = codec.group
            return codec.reduce(partial)

        @staticmethod
        def backward(ctx, grad_output):
            from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
            return _gather_along_first_dim(grad_output.contiguous(), ctx.group), None
