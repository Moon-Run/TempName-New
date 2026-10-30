"""Mixed-wire forward reference; backward remains the common BF16 approximation."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from reference import roundtrip
from routing import base_policy, codec_placement, selection_entry


def masked_roundtrip(piece, mask, src, dst, m, n, world):
    coded, _ = roundtrip(piece)
    if mask is None: return coded
    tm, nt = m//128, (n+127)//128
    selected = mask[src].reshape(tm,nt)[dst*(tm//world):(dst+1)*(tm//world)]
    selected = selected.repeat_interleave(128,0).repeat_interleave(128,1)[:,:n].to(piece.device).bool()
    return torch.where(selected, coded, piece)


class TacoForward:
    def __init__(self, group, m, n, k_local, policy):
        from flux.gemm_rs_taco import GemmRSTaco
        selected = None
        config = json.loads((Path(__file__).parent/'config.json').read_text())
        if config.get('double_buffered') and policy.endswith('_selective'):
            from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered as GemmRSTaco
        if policy.endswith('_selective'):
            plan = json.loads((Path(__file__).parent/'selection-plan.json').read_text())
            selected = selection_entry(plan,policy,m,n,k_local*dist.get_world_size(group))['mask']
        self.codec = GemmRSTaco(group,m,n,k_local,placement=codec_placement(policy),
                               **({'selected':selected} if selected is not None else {}),
                               **({'graph': True} if config.get('graph_forward') and policy.endswith('_selective') else {}))
        self.owns_output = getattr(self.codec, 'owns_configuration', False)

    def forward(self,input_,weight,reduce_scatter_option=None):
        if self.owns_output:
            return self.codec.forward(input_, weight)
        return self.codec.forward(input_.detach(),weight.detach())


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
