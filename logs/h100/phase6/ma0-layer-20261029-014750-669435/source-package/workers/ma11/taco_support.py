"""Mixed-wire forward reference; backward remains the common BF16 approximation."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from reference import roundtrip
from routing import is_native, is_fused_v2, selection_entry, definition


def masked_roundtrip(piece, mask, src, dst, m, n, world):
    coded, _ = roundtrip(piece)
    if mask is None: return coded
    tm, nt = m//128, (n+127)//128
    selected = mask[src].reshape(tm,nt)[dst*(tm//world):(dst+1)*(tm//world)]
    selected = selected.repeat_interleave(128,0).repeat_interleave(128,1)[:,:n].to(piece.device).bool()
    return torch.where(selected, coded, piece)


class TacoForward:
    def __init__(self, group, m, n, k_local, policy, role):
        if is_native(policy):
            raise ValueError('Native scope cannot construct a Flux candidate')
        if is_fused_v2(policy):
            from flux.gemm_rs_taco import GemmRSTaco
            self.role = role
            self.plan_sha256 = None
            self.codec = GemmRSTaco(group, m, n, k_local, placement='fused')
            raise RuntimeError('MA4_F requires the selective CR2 policy')
            self.workspace_info = [dict(shape=[m,n,k_local*dist.get_world_size(group)],
                packet_ptr=self.codec.peers[self.codec.rank].data_ptr())]
            return
        plan = json.loads((Path(__file__).parent/'selection-plan.json').read_text())
        record = selection_entry(plan, policy, role, m, n, k_local*dist.get_world_size(group))
        from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
        self.role = role
        self.plan_sha256 = record['entry_sha256']
        self.codec = GemmRSTacoDoubleBuffered(group,m,n,k_local,placement='fused',
            selected=record['entry']['mask'],quantization_mode=definition(policy)['quantization'])
        self.owns_configuration = self.codec.owns_configuration
        self.owns_output = self.codec.owns_configuration
        if not self.owns_output:
            raise RuntimeError('Scope requires the frozen instance-owned V2 path')
        self.workspace_info = [dict(shape=[slot.m,slot.n,slot.k_local*dist.get_world_size(group)],
            mask_sha256=slot.selection_sha256, packet_ptr=slot.peers[slot.rank].data_ptr(),
            mask_ptr=slot.selected.data_ptr()) for slot in self.codec.slots]
        if len(self.workspace_info) != 2 or any(s['shape'] != [m,n,k_local*dist.get_world_size(group)]
                                               for s in self.workspace_info):
            raise RuntimeError('Double workspace/full-shape mismatch')

    def forward(self,input_,weight,reduce_scatter_option=None):
        if self.owns_output:
            return self.codec.forward(input_, weight)
        return self.codec.forward(input_.detach(),weight.detach())


    def forward_consumer(self, input_, weight, bias, residual, reduce_scatter_option=None):
        if not self.owns_output or not self.owns_configuration:
            raise RuntimeError('MA4-F requires the owned CR2 output/configuration')
        return self.codec.forward_consumer(input_, weight, bias, residual)


@torch.no_grad()
def check_taco_output(partial, actual, group, mask=None):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    parts = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(parts, partial, group=group)
    rows = partial.shape[0]//world
    bf16, coded = torch.zeros_like(actual), torch.zeros_like(actual)
    for i in range(1,world+1):
        src = (rank+i)%world
        piece = parts[src][rank*rows:(rank+1)*rows]
        bf16 = (bf16+piece).to(torch.bfloat16)
        if src != rank:
            piece = masked_roundtrip(piece,mask,src,rank,*partial.shape,world)
        coded = (coded+piece).to(torch.bfloat16)
    denominator = max(bf16.float().norm().item(),1e-20)
    delta = actual.float()-bf16.float()
    error = dict(max_abs=delta.abs().max().item(),relative_l2=delta.norm().item()/denominator,
                 protocol_relative_l2=(actual.float()-coded.float()).norm().item()/denominator,
                 budget_vs_bf16=.05,budget_vs_codec=.01)
    error['passed'] = bool(torch.isfinite(actual).all()) and error['relative_l2']<=.05 and error['protocol_relative_l2']<=.01
    assert error['passed'], error
    return error


@torch.no_grad()
def check_consumer_output(partial, actual, group, mask, bias, residual):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    parts = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(parts, partial, group=group)
    rows = partial.shape[0]//world
    bf16, coded = torch.zeros_like(actual), torch.zeros_like(actual)
    for i in range(1,world+1):
        src = (rank+i)%world
        piece = parts[src][rank*rows:(rank+1)*rows]
        bf16 = (bf16+piece).to(torch.bfloat16)
        if src != rank:
            piece = masked_roundtrip(piece,mask,src,rank,*partial.shape,world)
        coded = (coded+piece).to(torch.bfloat16)
    # Apply the same two BF16 roundings after the ordered codec reference.
    bf16 = residual + (bf16 + bias)
    coded = residual + (coded + bias)
    denominator = max(bf16.float().norm().item(),1e-20)
    delta = actual.float()-bf16.float()
    error = dict(max_abs=delta.abs().max().item(),relative_l2=delta.norm().item()/denominator,
                 protocol_relative_l2=(actual.float()-coded.float()).norm().item()/denominator,
                 budget_vs_bf16=.05,budget_vs_codec=.01)
    error['passed'] = bool(torch.isfinite(actual).all()) and error['relative_l2']<=.05 and error['protocol_relative_l2']<=.01
    assert error['passed'], error
    return error
