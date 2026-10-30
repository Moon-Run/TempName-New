"""RTX5090 local TACO staging followed by NCCL transport and local decoding.

This file is installed as ``flux/gemm_rs_taco.py`` by the isolated builder.
No foreign GPU pointer or CUDA IPC allocation is created. ProcessGroupNCCL
orders producer work on the calling CUDA stream before its communication, and
``Work.wait()`` inserts the completion dependency onto that same stream before
unpacking/decoding. Slots are reused only on their fixed creation stream.
"""
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading

PROTOCOL = 'rtx5090-nccl-staged-v2'
NATIVE_ARM = '__RTX5090_NATIVE_ARM__'
NATIVE_ARMS = ('a', 'r', 'c', 'cr3', 'ma89', 'ma11')
STATE_FIELDS = ('protocol_version', 'initializations', 'producer_calls', 'decode_calls',
                'bias_residual_calls', 'phase', 'failed', 'bound_k')
TILE_BF16_BYTES = 128 * 128 * 2
TILE_PACKET_BYTES = 128 * 136
PACKET_BYTES_PER_SOURCE = 256 * 32 * 136


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_mask(selected):
    """Immutable CPU wire definition; no GPU or Torch import is needed."""
    if selected is None:
        return None
    if hasattr(selected, 'detach'):
        selected = selected.detach().cpu().tolist()
    elif hasattr(selected, 'tolist'):
        selected = selected.tolist()
    require(len(selected) == 8 and all(len(row) == 512 for row in selected),
            'Selection mask must have shape [8,512]')
    require(all(value in (0, 1) for row in selected for value in row), 'Selection mask must be binary')
    result = [[int(value) for value in row] for row in selected]
    require(all(not any(result[src][src*64:(src+1)*64]) for src in range(8)),
            'Local contributions must remain BF16')
    return result


def wire_layout(selected, rank):
    """Compact byte splits plus offsets, ordered by destination/source rank.

    A selected physical tile contributes 128 packed H128 groups; an unselected
    tile contributes 128x128 BF16 values. Self data never enters the exchange.
    """
    require(type(rank) is int and 0 <= rank < 8, 'Expected a TP8 rank')
    mask = canonical_mask(selected)
    tx_offsets, rx_offsets = [-1]*512, [-1]*512
    tx_splits, rx_splits = [0]*8, [0]*8
    tx_bytes = rx_bytes = 0
    for peer in range(8):
        if peer == rank:
            continue
        for tile in range(64):
            tx_offsets[peer*64+tile] = tx_bytes
            rx_offsets[peer*64+tile] = rx_bytes
            tx_quant = mask is None or bool(mask[rank][peer*64+tile])
            rx_quant = mask is None or bool(mask[peer][rank*64+tile])
            tx_size = TILE_PACKET_BYTES if tx_quant else TILE_BF16_BYTES
            rx_size = TILE_PACKET_BYTES if rx_quant else TILE_BF16_BYTES
            tx_splits[peer] += tx_size
            rx_splits[peer] += rx_size
            tx_bytes += tx_size
            rx_bytes += rx_size
    return dict(tx_offsets=tx_offsets, rx_offsets=rx_offsets,
                tx_splits=tx_splits, rx_splits=rx_splits, tx_bytes=tx_bytes, rx_bytes=rx_bytes)


class _NativeRuntime:
    """Checked C ABI. All communication buffers remain owned by Python."""
    def __init__(self, library, torch, rank, k_local, cache, selected, buffers, layout, stream):
        self.rank = rank
        self.torch, self.device = torch, torch.cuda.current_device()
        require(self.device == self.rank, 'Bind each local torchrun rank to its matching CUDA device')
        self.lib = ctypes.CDLL(str(library))
        self.handle = ctypes.c_void_p()
        self._closed = False
        signatures = {
            'rtx5090_local_protocol_version': ([], ctypes.c_int),
            'rtx5090_local_last_error': ([], ctypes.c_char_p),
            'rtx5090_local_create': ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int] + [ctypes.c_void_p]*9 +
                                    [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p], ctypes.c_int),
            'rtx5090_local_produce': ([ctypes.c_void_p]*4, ctypes.c_int),
            'rtx5090_local_decode': ([ctypes.c_void_p]*5, ctypes.c_int),
            'rtx5090_local_state': ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int], ctypes.c_int),
            'rtx5090_local_reconfigure': ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
            'rtx5090_local_reconfiguration_count': ([ctypes.c_void_p], ctypes.c_int64),
            'rtx5090_local_workspace_bytes': ([ctypes.c_void_p], ctypes.c_int64),
            'rtx5090_local_destroy': ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        require(self.lib.rtx5090_local_protocol_version() == 1, 'Wrong local TACO native protocol')
        pointers = [selected.data_ptr() if selected is not None else None]
        pointers += [buffers[key].data_ptr() for key in
                     ('tx_bf16', 'tx_packets', 'rx_bf16', 'rx_packets',
                      'tx_offsets', 'rx_offsets', 'tx_wire', 'rx_wire')]
        self._call('rtx5090_local_create', ctypes.byref(self.handle), rank, k_local, int(cache),
                   *pointers, layout['tx_bytes'], layout['rx_bytes'], stream)

    def _call(self, name, *arguments):
        code = getattr(self.lib, name)(*arguments)
        if code:
            detail = self.lib.rtx5090_local_last_error()
            detail = detail.decode(errors='replace') if detail else 'no native detail'
            raise RuntimeError(f'{name} failed with native/CUDA error code {code}: {detail}')

    def reconfigure(self, stream):
        self._call('rtx5090_local_reconfigure', self.handle, stream)

    def produce(self, input_, weight, stream):
        self._call('rtx5090_local_produce', self.handle, input_.data_ptr(), weight.data_ptr(), stream)

    def decode(self, output, bias, residual, stream):
        self._call('rtx5090_local_decode', self.handle, output.data_ptr(),
                   bias.data_ptr() if bias is not None else None,
                   residual.data_ptr() if residual is not None else None, stream)

    def state(self):
        if self._closed:
            return dict(self._last_state)
        values = (ctypes.c_int64 * len(STATE_FIELDS))()
        self._call('rtx5090_local_state', self.handle, values, len(values))
        result = dict(zip(STATE_FIELDS, values))
        result['failed'] = bool(result['failed'])
        result['reconfigure_calls'] = self.lib.rtx5090_local_reconfiguration_count(self.handle)
        require(result['reconfigure_calls'] >= 0, 'Cannot read native reconfiguration counter')
        result['workspace_bytes'] = self.lib.rtx5090_local_workspace_bytes(self.handle)
        require(result['workspace_bytes'] >= 0, 'Cannot read native workspace size')
        return result

    def close(self):
        if not self._closed and self.handle.value:
            self._last_state = self.state()
            with self.torch.cuda.device(self.device):
                self._call('rtx5090_local_destroy', self.handle)
            self.handle = ctypes.c_void_p()
            self._closed = True


class _Exchange:
    """Fixed-stream single-flight submission, independently testable on CPU."""
    def __init__(self, native, torch, dist, buffers, layout, group, device, stream,
                 *, reconfigure=False, profile=False):
        self.native, self.torch, self.dist = native, torch, dist
        self.buffers, self.layout, self.group = buffers, layout, group
        self.device, self.stream = device, stream
        self.reconfigure, self.profile = reconfigure, profile
        self.lock = threading.Lock()
        self.failed = self.closed = False
        self.inflight = None
        self.calls = self.exchange_calls = self.wait_calls = 0

    def scope(self, name):
        return self.torch.profiler.record_function('rtx5090.transport.'+name) if self.profile else nullcontext()

    def run(self, input_, weight, output, bias=None, residual=None):
        if not self.lock.acquire(blocking=False):
            self.failed = True
            raise RuntimeError('Concurrent or reentrant NCCL TACO submission')
        try:
            require(not self.failed and not self.closed, 'Cannot reuse failed/closed NCCL TACO staging')
            require(self.torch.cuda.current_device() == self.device and
                    self.torch.cuda.current_stream().cuda_stream == self.stream,
                    'NCCL TACO staging requires its creation device and CUDA stream')
            require(not self.torch.cuda.is_current_stream_capturing(), 'CUDA graph capture is unsupported')
            current = self.torch.cuda.current_stream()
            for tensor in (input_, weight, output, bias, residual):
                if tensor is not None:
                    tensor.record_stream(current)
            if self.reconfigure:
                with self.scope('configure'):
                    self.native.reconfigure(self.stream)
            with self.scope('produce_pack'):
                self.native.produce(input_, weight, self.stream)
            with self.scope('exchange'):
                self.inflight = self.dist.all_to_all_single(
                    self.buffers['rx_wire'], self.buffers['tx_wire'],
                    output_split_sizes=self.layout['rx_splits'],
                    input_split_sizes=self.layout['tx_splits'], group=self.group, async_op=True)
                require(self.inflight is not None, 'NCCL asynchronous exchange returned no Work')
                self.exchange_calls += 1
                # CUDA Work.wait() orders the current CUDA stream after NCCL;
                # it is not a CPU/device-wide synchronize or an all-rank barrier.
                require(self.inflight.wait() is not False, 'NCCL exchange did not complete successfully')
                self.wait_calls += 1
                self.inflight = None
            with self.scope('unpack_decode_consumer' if bias is not None else 'unpack_decode'):
                self.native.decode(output, bias, residual, self.stream)
            self.calls += 1
            return output
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def state(self):
        native = self.native.state()
        return dict(protocol=PROTOCOL, native_api='rtx5090_local_v1', calls=self.calls,
                    producer_calls=native['producer_calls'], pack_calls=native['producer_calls'],
                    exchange_calls=self.exchange_calls, wait_calls=self.wait_calls,
                    unpack_calls=native['decode_calls'], decode_calls=native['decode_calls'],
                    consumer_calls=native['bias_residual_calls'], initializations=native['initializations'],
                    native_workspace_bytes=native['workspace_bytes'],
                    reconfigure_calls=native['reconfigure_calls'],
                    configuration_scope='per_call' if self.reconfigure else 'instance',
                    failed=bool(self.failed or native['failed']), closed=self.closed,
                    pending_exchange=self.inflight is not None, native_phase=native['phase'],
                    device=self.device, stream=self.stream, peer_memory=False,
                    send_split_bytes=list(self.layout['tx_splits']), receive_split_bytes=list(self.layout['rx_splits']),
                    send_bytes=self.layout['tx_bytes'], receive_bytes=self.layout['rx_bytes'],
                    synchronization='fixed_stream_producer_NCCL_work_wait_decode_reuse', native=native)

    def close(self):
        require(self.lock.acquire(blocking=False), 'Cannot close an active NCCL TACO submission')
        try:
            if self.closed:
                return
            require(self.inflight is None, 'Cannot close pending NCCL TACO work')
            self.native.close()
            self.closed = True
        finally:
            self.lock.release()


class GemmRSTaco:
    def __init__(self, group, m, n, k_local, placement=None, selected=None, graph=False,
                 _defer_reuse_barrier=False, quantization_mode=None):
        import torch
        import torch.distributed as dist
        require(NATIVE_ARM in NATIVE_ARMS, 'Builder must bind the native arm before installing this frontend')
        require(placement in (None, 'fused') and not graph, 'Local NCCL TACO requires eager fused placement')
        require(os.environ.get('NCCL_P2P_DISABLE') == '1', 'NCCL_P2P_DISABLE=1 is required by this protocol')
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        require((self.world, m, n) == (8, 2048, 4096) and k_local in (512, 2048),
                'Local TACO requires exact TP8 attention/MLP shapes')
        require(dist.get_backend(group) == 'nccl', 'Use an NCCL TP process group')
        require(int(os.environ.get('LOCAL_WORLD_SIZE', '-1')) == 8 and
                int(os.environ.get('LOCAL_RANK', '-1')) == self.rank and torch.cuda.device_count() == 8,
                'Expected one complete local TP8 group; subgroup swizzle ranks are unsupported')
        self.torch, self.device = torch, torch.cuda.current_device()
        self.stream = torch.cuda.current_stream().cuda_stream
        prop = torch.cuda.get_device_properties(self.device)
        require((prop.major, prop.minor, prop.multi_processor_count) == (12, 0, 170), 'Expected full RTX5090 SM120')
        require(not torch.cuda.is_current_stream_capturing(), 'Cannot initialize local TACO during graph capture')
        self.m, self.n, self.k_local = m, n, k_local
        self.placement, self.graph_enabled, self.graph = 'fused', False, None
        self.quantization_mode = quantization_mode or 'all_remote'
        self.owns_output = self.owns_configuration = NATIVE_ARM != 'a'
        self.mask = canonical_mask(selected)
        require((self.mask is None) == (NATIVE_ARM in ('a', 'r')), 'Native arm/selection family mismatch')
        require(self.quantization_mode == ('all_remote' if self.mask is None else 'targeted'),
                'Native arm/quantization mode mismatch')
        self.selection_cpu = None if self.mask is None else torch.tensor(self.mask, dtype=torch.uint8)
        self.selection_sha256 = None if self.mask is None else hashlib.sha256(bytes(x for row in self.mask for x in row)).hexdigest()
        self.selected = None if self.mask is None else self.selection_cpu.to(device=self.device)
        layout = wire_layout(self.mask, self.rank)
        identity = hashlib.sha256(json.dumps(dict(protocol=PROTOCOL, arm=NATIVE_ARM, m=m, n=n, k=k_local,
                                                  mask=self.mask), sort_keys=True).encode()).hexdigest()
        digests = [None]*8
        dist.all_gather_object(digests, identity, group=group)
        require(all(value == identity for value in digests), 'Ranks disagree on native arm, shape or packet schema')
        self.buffers = {
            key: torch.empty((8, 256, 4096), dtype=torch.bfloat16, device=self.device)
            for key in ('tx_bf16', 'rx_bf16')}
        self.buffers.update({
            key: torch.empty((8, PACKET_BYTES_PER_SOURCE), dtype=torch.uint8, device=self.device)
            for key in ('tx_packets', 'rx_packets')})
        for prefix in ('tx', 'rx'):
            self.buffers[prefix+'_offsets'] = torch.tensor(layout[prefix+'_offsets'], dtype=torch.int64, device=self.device)
            self.buffers[prefix+'_wire'] = torch.empty((layout[prefix+'_bytes'],), dtype=torch.uint8, device=self.device)
        self.packet_bytes_per_source = PACKET_BYTES_PER_SOURCE
        self._wire_bytes_per_rank = layout['tx_bytes']
        library = (Path(__file__).parent/'lib/libflux_cuda.so').resolve()
        loaded = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                  if line.endswith('/libflux_cuda.so')}
        require(loaded == {library}, 'Use a fresh process with only this isolated Flux library')
        native = _NativeRuntime(library, torch, self.rank, k_local, NATIVE_ARM in ('cr3', 'ma89', 'ma11'),
                                self.selected, self.buffers, layout, self.stream)
        self.lib = native.lib
        self._exchange = _Exchange(native, torch, dist, self.buffers, layout, group, self.device, self.stream,
                                   reconfigure=NATIVE_ARM == 'a', profile=os.environ.get('E2E_MODE') == 'profile')

    def _forward(self, input_, weight, bias=None, residual=None):
        tensors = [(input_, (self.m, self.k_local)), (weight, (self.n, self.k_local))]
        if bias is not None or residual is not None:
            require(bias is not None and residual is not None and NATIVE_ARM in ('cr3', 'ma89', 'ma11'),
                    'Consumer fusion requires both tensors and a CR3-family native arm')
            tensors += [(bias, (4096,)), (residual, (256, 4096))]
        try:
            for tensor, shape in tensors:
                require(tensor.is_cuda and tensor.device.index == self.device and
                        tensor.dtype == self.torch.bfloat16 and tuple(tensor.shape) == shape and
                        tensor.is_contiguous(), 'Expected exact contiguous CUDA BF16 tensor shape '+repr(shape))
                require(not self.torch.is_grad_enabled() or not tensor.requires_grad,
                        'Use the registered autograd adapter or no_grad for local TACO')
            output = self.torch.empty((256, 4096), dtype=self.torch.bfloat16, device=self.device)
            return self._exchange.run(input_, weight, output, bias, residual)
        except BaseException:
            self._exchange.failed = True
            raise

    def forward(self, input_, weight):
        return self._forward(input_, weight)

    def forward_consumer(self, input_, weight, bias, residual):
        return self._forward(input_, weight, bias, residual)

    def local_workspace_info(self):
        result = dict(shape=[self.m, self.n, self.k_local*self.world],
                      packet_ptr=self.buffers['rx_packets'].data_ptr())
        if self.selected is not None:
            result.update(mask_sha256=self.selection_sha256, mask_ptr=self.selected.data_ptr())
        return result

    def transport_state(self):
        return self._exchange.state()

    def cr2_submission_state(self):
        native = self._exchange.native.state()
        return dict(protocol_version=native['protocol_version'], native_api='rtx5090_local_v1',
                    transport_protocol=PROTOCOL, initializations=native['initializations'],
                    successful_calls=native['decode_calls'], bound_k=native['bound_k'],
                    local_packet_ptr=self.buffers['rx_packets'].data_ptr(),
                    failed=bool(native['failed'] or self._exchange.failed))

    def ma4_consumer_state(self):
        return dict(self.cr2_submission_state(),
                    bias_residual_calls=self._exchange.native.state()['bias_residual_calls'])

    @property
    def wire_bytes_per_rank(self):
        return self._wire_bytes_per_rank

    @property
    def bf16_wire_bytes_per_rank(self):
        return 7*256*4096*2

    def close(self):
        self._exchange.close()

    def __del__(self):
        exchange = getattr(self, '_exchange', None)
        if exchange is not None and not exchange.failed and exchange.inflight is None:
            try:
                exchange.close()
            except BaseException:
                pass  # Interpreter/CUDA shutdown must not replace an original exception.


class GemmRSTacoDoubleBuffered:
    """Two local staging slots; no cross-rank reuse rendezvous is required."""
    def __init__(self, *args, **kwargs):
        require(NATIVE_ARM in ('c', 'cr3', 'ma89', 'ma11'), 'Double staging is only defined for selective arms')
        self.slots = [GemmRSTaco(*args, **kwargs) for _ in range(2)]
        self.next_slot = 0
        self._failed = False
        self._lock = threading.Lock()
        for name in ('selection_cpu', 'selection_sha256', 'selected', 'placement', 'owns_configuration',
                     'owns_output', 'wire_bytes_per_rank', 'bf16_wire_bytes_per_rank', 'group'):
            setattr(self, name, getattr(self.slots[0], name))

    def _run(self, method, *args):
        if not self._lock.acquire(blocking=False):
            self._failed = True
            raise RuntimeError('Concurrent or reentrant double staging submission')
        try:
            require(not self._failed, 'Cannot reuse failed double staging')
            result = getattr(self.slots[self.next_slot], method)(*args)
            self.next_slot ^= 1
            return result
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()

    def forward(self, input_, weight):
        return self._run('forward', input_, weight)

    def forward_consumer(self, input_, weight, bias, residual):
        return self._run('forward_consumer', input_, weight, bias, residual)

    def local_workspace_info(self):
        return [slot.local_workspace_info() for slot in self.slots]

    def transport_state(self):
        return dict(protocol=PROTOCOL, failed=self._failed, next_slot=self.next_slot,
                    slots=[dict(slot=index, **slot.transport_state()) for index, slot in enumerate(self.slots)])

    def cr2_submission_state(self):
        return [dict(slot=index, **slot.cr2_submission_state()) for index, slot in enumerate(self.slots)]

    def ma4_consumer_state(self):
        return [dict(slot=index, **slot.ma4_consumer_state()) for index, slot in enumerate(self.slots)]

    def close(self):
        require(self._lock.acquire(blocking=False), 'Cannot close an active double staging submission')
        try:
            for slot in self.slots:
                slot.close()
        finally:
            self._lock.release()


def frontend_source(arm):
    """Builder helper; the installed file contains no repository import."""
    require(arm in NATIVE_ARMS, 'Unknown native arm')
    source = Path(__file__).read_text()
    marker = '\nNATIVE_ARM = ' + repr('__RTX5090_NATIVE_ARM__') + '\n'
    require(source.count(marker) == 1, 'Native arm source marker changed')
    return source.replace(marker, '\nNATIVE_ARM = '+repr(arm)+'\n')
