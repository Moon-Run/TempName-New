"""Experimental all/selective remote TACO, built by tools/a800/phase4/build.py.

BF16 GEMM, original Flux swizzle, fixed E4M3/128 adaptive-Hadamard codec.
One instance owns exact-shape IPC packets and is bound to its creation stream.
All ranks must construct and call instances in the same order. Fixed-shape graph
replay is opt-in for the instance-owned TP4 implementation.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import threading

import torch
import torch.distributed as dist

from . import cpp_mod


class GemmRSTaco:
    def __init__(self, group, m, n, k_local, placement=None, selected=None, graph=False,
                 _defer_reuse_barrier=False):
        if placement not in (None, 'fused', 'separate'):
            raise ValueError('TACO placement must be fused or separate')
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if graph and self.world != 4:
            raise ValueError('TACO graph replay is validated for TP4 only')
        if self.world not in (2, 4, 8) or m <= 0 or m % (128*self.world) or n <= 0 or n % 8 or k_local <= 0 or k_local % 32:
            raise ValueError('TACO requires TP2/4/8, M%(128*TP)=0, N%8=0, K_local%32=0')
        if torch.cuda.get_device_capability() != (8, 0):
            raise ValueError('This TACO baseline supports SM80 only')
        if (int(os.environ.get('LOCAL_WORLD_SIZE', '-1')) != self.world or
                int(os.environ.get('LOCAL_RANK', '-1')) != self.rank or
                torch.cuda.device_count() != self.world):
            raise ValueError('TACO requires one complete local torchrun group; subgroup swizzle ranks are unsupported')
        if not all(torch.cuda.can_device_access_peer(i,j) for i in range(self.world)
                   for j in range(self.world) if i != j):
            raise ValueError('TACO requires peer access between all local devices')
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.current_stream().cuda_stream
        self.m, self.n, self.k_local = m, n, k_local
        self._lock = threading.Lock()
        self.graph_enabled = graph
        self.graph = None
        library = (Path(__file__).parent/'lib/libflux_cuda.so').resolve()
        loaded = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                  if line.endswith('/libflux_cuda.so')}
        if loaded != {library}:
            raise RuntimeError('TACO requires a fresh process using only its isolated Flux build')
        self.lib = ctypes.CDLL(str(library))
        try:
            self.lib.taco_configure.argtypes = [ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_int]*4
            self.lib.taco_configure.restype = ctypes.c_int
            self.lib.taco_reset.argtypes = []
            self.lib.taco_reset.restype = None
            self.lib.taco_placement.argtypes = []
            self.lib.taco_placement.restype = ctypes.c_int
        except AttributeError as e:
            raise RuntimeError('Build the TACO baseline with tools/a800/phase4/build.py first') from e
        self.placement = {1: 'fused', 2: 'separate'}.get(self.lib.taco_placement())
        if self.placement is None or placement is not None and self.placement != placement:
            raise RuntimeError('Loaded TACO library does not match the requested codec placement')
        self.selected = None
        self.selection_cpu = None
        self.selection_sha256 = None
        self.quant_tiles = None
        self.quant_tile_count = None
        if selected is not None:
            if self.placement != 'fused':
                raise ValueError('Selective TACO requires epilogue-fused encoding')
            mask = torch.as_tensor(selected, device='cpu').clone().contiguous()
            tiles = (m//128)*((n+127)//128)
            if tuple(mask.shape) != (self.world, tiles) or not bool(((mask==0)|(mask==1)).all()):
                raise ValueError('Selection must be a binary [world, physical_tiles] mask')
            for src in range(self.world):
                if bool(mask[src,src*(tiles//self.world):(src+1)*(tiles//self.world)].any()):
                    raise ValueError('Local contributions must remain BF16')
            mask = mask.to(torch.uint8)
            digest = hashlib.sha256(bytes(mask.flatten().tolist())).hexdigest()
            digests = [None]*self.world
            dist.all_gather_object(digests, digest, group=group)
            if len(set(digests)) != 1:
                raise ValueError('All ranks must use the same immutable selection mask')
            self.selection_cpu = mask
            self.selection_sha256 = digest
            self.selected = mask.to(device=self.device)
            self.lib.taco_configure_selective.argtypes = [ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_int]*4 + [ctypes.c_void_p]
            self.lib.taco_configure_selective.restype = ctypes.c_int
            if hasattr(self.lib, 'taco_configure_selective_compact'):
                per_rank = tiles // self.world
                local_selected = mask[:,self.rank*per_rank:(self.rank+1)*per_rank].bool().any(dim=0)
                indices = local_selected.nonzero().flatten().to(torch.int32).contiguous()
                self.quant_tile_count = indices.numel()
                self.quant_tiles = indices.to(device=self.device)
                self.lib.taco_configure_selective_compact.argtypes = (
                    [ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_int]*4 +
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int])
                self.lib.taco_configure_selective_compact.restype = ctypes.c_int
        cpp_mod.init_flux_shm(group)
        groups = (m//self.world)*((n+127)//128)
        self.packet_bytes_per_source = groups*136
        # Source slots are disjoint at each destination. The local slot is unused.
        self.peers = cpp_mod.create_tensor_list([self.world*self.packet_bytes_per_source], torch.uint8,
                                               group, False, True)
        self.pointers = (ctypes.c_void_p*self.world)(*[p.data_ptr() for p in self.peers])
        self.owns_configuration = hasattr(self.lib, 'taco_instance_config_supported')
        if graph and not hasattr(self.lib, 'taco_allow_capture'):
            raise RuntimeError('Graph replay requires the instance-owned TACO build')
        if _defer_reuse_barrier and (graph or not hasattr(self.lib, 'taco_defer_reuse_barrier')):
            raise RuntimeError('Deferred reuse needs the double-buffered eager TACO implementation')
        if self.owns_configuration:
            if self.quant_tile_count is not None:
                code = self.lib.taco_configure_selective_compact(
                    self.pointers, self.rank, self.world, m, n, self.selected.data_ptr(),
                    self.quant_tiles.data_ptr(), self.quant_tile_count)
            else:
                code = (self.lib.taco_configure(self.pointers, self.rank, self.world, m, n)
                    if self.selected is None else
                    self.lib.taco_configure_selective(self.pointers, self.rank, self.world, m, n,
                                                      self.selected.data_ptr()))
            if code:
                raise RuntimeError(f'TACO instance configuration rejected: CUDA error {code}')
            if graph:
                self.lib.taco_allow_capture()
            if _defer_reuse_barrier:
                self.lib.taco_defer_reuse_barrier()
        try:
            self.op = cpp_mod.GemmRS(group, 1, m, n, torch.bfloat16, torch.bfloat16,
                                    transpose_weight=False, fuse_reduction=False, ring_reduction=True)
        finally:
            if self.owns_configuration:
                self.lib.taco_reset()
        self.option = cpp_mod.ReduceScatterOption()
        for key, value in dict(use_1d_ring=True, use_p2p_read=True, use_cudaMemcpyAsync=False,
                               use_gemmk=False, per_tile_flags=False, use_barrier_queue=False,
                               num_blocks=6, n_split=1).items():
            setattr(self.option, key, value)
        self.option.ring_mode = cpp_mod.RingMode.Ring1D

    def forward(self, input, weight):
        if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
            raise ValueError('Forward-only codec; use no_grad or an explicit autograd adapter')
        if self.graph_enabled:
            return self._graph_forward(input, weight)
        if self.owns_configuration:
            # C++ validates tensor layout, device, stream and capture, and scopes
            # the immutable config to this instance. Output owns its allocation.
            return self.op.forward(input, weight, reduce_scatter_option=self.option)
        if torch.cuda.current_device() != self.device or torch.cuda.current_stream().cuda_stream != self.stream:
            raise RuntimeError('TACO instance must be used on its creation device and stream')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('CUDA graph capture is not validated for this TACO baseline')
        for tensor, shape in ((input, (self.m, self.k_local)), (weight, (self.n, self.k_local))):
            if not tensor.is_cuda or tensor.device.index != self.device or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise ValueError(f'Expected contiguous CUDA BF16 tensor of shape {shape}')
            if tensor.requires_grad:
                raise ValueError('Forward-only baseline; supply detached tensors or an explicit autograd adapter')
        with self._lock:
            if self.selected is None:
                code = self.lib.taco_configure(self.pointers, self.rank, self.world, self.m, self.n)
            else:
                code = self.lib.taco_configure_selective(self.pointers, self.rank, self.world, self.m, self.n,
                                                        self.selected.data_ptr())
            if code:
                raise RuntimeError(f'TACO configuration rejected: CUDA error {code}')
            try:
                return self.op.forward(input, weight, reduce_scatter_option=self.option).reshape(self.m//self.world, self.n)
            finally:
                self.lib.taco_reset()

    def _graph_forward(self, input, weight):
        # Only this exact MLP forward is captured. Input copying, replay and
        # output ownership are timed; all other training/attention remains eager.
        if torch.cuda.current_device() != self.device or torch.cuda.current_stream().cuda_stream != self.stream:
            raise RuntimeError('Graph replay must use the instance creation stream')
        if (tuple(input.shape) != (self.m, self.k_local) or not input.is_contiguous() or
                input.dtype != torch.bfloat16 or not input.is_cuda or input.device.index != self.device):
            raise ValueError('Graph input shape/layout changed')
        if (tuple(weight.shape) != (self.n, self.k_local) or not weight.is_contiguous() or
                weight.dtype != torch.bfloat16 or weight.device != input.device):
            raise ValueError('Graph weight shape/layout changed')
        with self._lock:
            if self.graph is None:
                self.graph_input = torch.empty_like(input)
                self.graph_weight = weight
                self.graph_weight_ptr = weight.data_ptr()
                self.graph_input.copy_(input)
                for _ in range(2):
                    self.op.forward(self.graph_input, weight, reduce_scatter_option=self.option)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.graph_output = self.op.forward(self.graph_input, weight,
                                                        reduce_scatter_option=self.option)
                self.graph = graph
            else:
                if weight.data_ptr() != self.graph_weight_ptr:
                    raise ValueError('Graph weight storage changed')
                self.graph_input.copy_(input)
            self.graph.replay()
            return self.graph_output.clone()

    @property
    def wire_bytes_per_rank(self):
        if self.selection_cpu is not None:
            total = self.bf16_wire_bytes_per_rank
            nt = (self.n+127)//128
            for tile in self.selection_cpu[self.rank].nonzero().flatten().tolist():
                valid = min(128, self.n-(tile%nt)*128)
                total += 128*(136-2*valid)
            return total
        return (self.world-1)*self.packet_bytes_per_source

    @property
    def bf16_wire_bytes_per_rank(self):
        return (self.world-1)*(self.m//self.world)*self.n*2


class GemmRSTacoDoubleBuffered:
    """Alternate TWO complete IPC workspaces, not just the FP8 packets.

    For invocation k+2 to overwrite slot k, it must first pass invocation k+1's
    post-GEMM all-rank barrier. Every rank enters that barrier only after its
    decode of k has completed on the same stream. Thus both BF16 source slots
    and FP8 packets from k are no longer read. This preserves the publication
    barrier on every call while avoiding a second all-rank rendezvous after DQ.
    All ranks must call this wrapper in the same order on one fixed stream.
    """
    def __init__(self, *args, **kwargs):
        self.slots = [GemmRSTaco(*args, **kwargs, _defer_reuse_barrier=True) for _ in range(2)]
        self.next_slot = 0
        self._lock = threading.Lock()
        for name in ('selection_cpu','selection_sha256','selected','placement',
                     'owns_configuration','wire_bytes_per_rank','bf16_wire_bytes_per_rank','group'):
            setattr(self,name,getattr(self.slots[0],name))

    def forward(self, input, weight):
        with self._lock:
            output = self.slots[self.next_slot].forward(input, weight)
            self.next_slot ^= 1
            return output
