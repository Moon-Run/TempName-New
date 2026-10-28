"""Experimental all-remote TACO baseline, built by tools/a800/phase4/build.py.

BF16 GEMM, original Flux swizzle, fixed E4M3/128 adaptive-Hadamard codec.
One instance owns exact-shape IPC packets and is bound to its creation stream.
All ranks must construct and call instances in the same order. No graph capture.
"""
import ctypes
import os
from pathlib import Path
import threading

import torch
import torch.distributed as dist

from . import cpp_mod


class GemmRSTaco:
    def __init__(self, group, m, n, k_local, placement=None):
        if placement not in (None, 'fused', 'separate'):
            raise ValueError('TACO placement must be fused or separate')
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
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
        cpp_mod.init_flux_shm(group)
        groups = (m//self.world)*((n+127)//128)
        self.packet_bytes_per_source = groups*136
        # Source slots are disjoint at each destination. The local slot is unused.
        self.peers = cpp_mod.create_tensor_list([self.world*self.packet_bytes_per_source], torch.uint8,
                                               group, False, True)
        self.pointers = (ctypes.c_void_p*self.world)(*[p.data_ptr() for p in self.peers])
        self.op = cpp_mod.GemmRS(group, 1, m, n, torch.bfloat16, torch.bfloat16,
                                transpose_weight=False, fuse_reduction=False, ring_reduction=True)
        self.option = cpp_mod.ReduceScatterOption()
        for key, value in dict(use_1d_ring=True, use_p2p_read=True, use_cudaMemcpyAsync=False,
                               use_gemmk=False, per_tile_flags=False, use_barrier_queue=False,
                               num_blocks=6, n_split=1).items():
            setattr(self.option, key, value)
        self.option.ring_mode = cpp_mod.RingMode.Ring1D

    def forward(self, input, weight):
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
            code = self.lib.taco_configure(self.pointers, self.rank, self.world, self.m, self.n)
            if code:
                raise RuntimeError(f'TACO configuration rejected: CUDA error {code}')
            try:
                return self.op.forward(input, weight, reduce_scatter_option=self.option).reshape(self.m//self.world, self.n)
            finally:
                self.lib.taco_reset()

    @property
    def wire_bytes_per_rank(self):
        return (self.world-1)*self.packet_bytes_per_source

    @property
    def bf16_wire_bytes_per_rank(self):
        return (self.world-1)*(self.m//self.world)*self.n*2
