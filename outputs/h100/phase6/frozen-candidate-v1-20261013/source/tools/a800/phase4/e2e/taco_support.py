"""MLP communication codec adapter and independent forward error checks.

The surrounding GemmRSFunction retains the same BF16 backward for every policy.
For the quantized forward this is a straight-through approximation, not an
exact derivative of rounding/scale selection. No codec runs in attention.
"""
import torch
import torch.distributed as dist
from reference import roundtrip


class TacoForward:
    def __init__(self, group, m, n, k_local, placement):
        from flux.gemm_rs_taco import GemmRSTaco
        self.codec = GemmRSTaco(group, m, n, k_local, placement=placement)

    def forward(self, input_, weight, reduce_scatter_option=None):
        # The explicit autograd Function owns backward, so satisfy the codec's
        # forward-only API without detaching the surrounding model graph.
        return self.codec.forward(input_.detach(), weight.detach())


@torch.no_grad()
def check_taco_output(partial, actual, group):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    all_parts = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(all_parts, partial, group=group)
    rows = partial.shape[0]//world
    bf16 = torch.zeros_like(actual)
    coded = torch.zeros_like(actual)
    for i in range(1, world+1):
        src = (rank+i)%world
        piece = all_parts[src][rank*rows:(rank+1)*rows]
        bf16 = (bf16+piece).to(torch.bfloat16)
        if src != rank:
            piece, _ = roundtrip(piece)
        coded = (coded+piece).to(torch.bfloat16)
    denominator = max(bf16.float().norm().item(), 1e-20)
    delta = actual.float()-bf16.float()
    error = dict(max_abs=delta.abs().max().item(), relative_l2=delta.norm().item()/denominator,
                 protocol_relative_l2=(actual.float()-coded.float()).norm().item()/denominator,
                 budget_vs_bf16=.05, budget_vs_codec=.01)
    error['passed'] = bool(torch.isfinite(actual).all()) and error['relative_l2'] <= .05 and error['protocol_relative_l2'] <= .01
    assert error['passed'], error
    return error
