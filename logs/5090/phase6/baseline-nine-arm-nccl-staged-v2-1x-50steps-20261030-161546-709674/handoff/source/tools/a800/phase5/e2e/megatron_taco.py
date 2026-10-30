"""Unoptimized tensor TACO RS baseline; no Flux import or GEMM replacement."""
from contextlib import nullcontext
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from reference import hadamard, decode


def pack_rows(x, destinations):
    n=x.shape[1];nt=(n+127)//128
    groups=F.pad(x.float(),(0,nt*128-n)).reshape(x.shape[0],nt,128)
    counts=torch.full((nt,),128.,device=x.device);counts[-1]=n-(nt-1)*128
    adaptive=(448*(groups.square().sum(-1)/counts+1e-6).clamp_min(1e-12).rsqrt()).clamp(1e-3,1e3)
    transformed=hadamard(groups*adaptive.unsqueeze(-1))
    quant=(transformed.abs().amax(-1).clamp_min(1e-12)/448).clamp(1e-12,1e6)
    payload=(transformed/quant.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
    # No diagnostic .item()/finite checks in the timed encoding path.
    return torch.cat([t.contiguous().view(torch.uint8).reshape(destinations,-1)
                      for t in (payload,quant,adaptive)],dim=1).contiguous()


def unpack_rows(packets, rows, n):
    nt=(n+127)//128;groups=rows*nt;count=packets.shape[0]
    q=packets[:,:groups*128].contiguous().view(torch.float8_e4m3fn).reshape(count*rows,nt,128)
    qs=packets[:,groups*128:groups*132].contiguous().view(torch.float32).reshape(count*rows,nt)
    adaptive=packets[:,groups*132:].contiguous().view(torch.float32).reshape(count*rows,nt)
    return decode(q,qs,adaptive,n).reshape(count,rows,n)


class TensorTacoRS:
    def __init__(self,group,m,n):
        self.group=group;self.rank=dist.get_rank(group);self.world=dist.get_world_size(group)
        assert self.world==4 and m%(128*self.world)==0 and n%8==0
        self.m,self.n,self.rows=m,n,m//self.world
        self.packet_bytes=self.rows*((n+127)//128)*136
        self.splits=[0 if r==self.rank else self.packet_bytes for r in range(self.world)]
        self.profile=os.environ.get('E2E_MODE')=='profile'
        self.selection_sha256=None
        self.selection_cpu=None
        self.wire_bytes_per_rank=(self.world-1)*self.packet_bytes
        self.bf16_wire_bytes_per_rank=(self.world-1)*self.rows*n*2

    def scope(self,name):
        return torch.profiler.record_function('megatron_taco.'+name) if self.profile else nullcontext()

    def reduce(self,partial):
        shape=partial.shape
        assert partial.is_cuda and partial.dtype==torch.bfloat16 and partial.numel()==self.m*self.n
        flat=partial.reshape(self.m,self.n)
        pieces=flat.reshape(self.world,self.rows,self.n)
        with self.scope('encode'):
            remote=torch.cat([pieces[d] for d in range(self.world) if d!=self.rank],dim=0)
            packets=pack_rows(remote,self.world-1)
        with self.scope('exchange'):
            received=torch.empty_like(packets)
            dist.all_to_all_single(received.view(-1),packets.view(-1),
                output_split_sizes=self.splits,input_split_sizes=self.splits,group=self.group)
        with self.scope('decode_reduce'):
            decoded=unpack_rows(received,self.rows,self.n)
            result=torch.zeros_like(pieces[self.rank])
            for i in range(1,self.world+1):
                src=(self.rank+i)%self.world
                value=pieces[src] if src==self.rank else decoded[src-int(src>self.rank)]
                result=(result+value).to(torch.bfloat16)
        return result.reshape(shape[0]//self.world,*shape[1:])


class QuantizedRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,partial,codec):
        ctx.group=codec.group
        return codec.reduce(partial)

    @staticmethod
    def backward(ctx,grad_output):
        # Same gather as Megatron's original sequence-parallel RS backward.
        from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
        return _gather_along_first_dim(grad_output.contiguous(),ctx.group),None
