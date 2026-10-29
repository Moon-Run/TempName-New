"""Worker module: local Flux GEMM only, followed by the shared TensorTacoRS.

No GemmRS, BF16 reduce-scatter or fused codec runs before Tensor TACO.
"""
from contextlib import nullcontext
import os

import torch
from megatron_taco import TensorTacoRS


class FluxTensorTaco:
    placement = 'tensor'
    owns_configuration = True
    selection_sha256 = None
    selection_cpu = None

    def __init__(self, group, m, n, k_local):
        from flux import GemmOnly
        if not callable(GemmOnly):
            raise RuntimeError('Ablation6 requires its separate GemmOnly build; existing GEMM-RS libraries cannot substitute')
        self.tensor_codec = TensorTacoRS(group, m, n)
        self.gemm = GemmOnly(torch.bfloat16, torch.bfloat16, torch.bfloat16,
                             transpose_weight=False, use_fp8_gemm=False)
        self.partial = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        self.m, self.n, self.k_local = m, n, k_local
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.wire_bytes_per_rank = self.tensor_codec.wire_bytes_per_rank
        self.bf16_wire_bytes_per_rank = self.tensor_codec.bf16_wire_bytes_per_rank
        self.workspace_info = [dict(shape=[m, n, k_local*self.tensor_codec.world], gemm_output_ptr=self.partial.data_ptr())]

    def forward(self, input_, weight):
        assert input_.shape == (self.m, self.k_local) and weight.shape == (self.n, self.k_local)
        assert input_.dtype == weight.dtype == torch.bfloat16
        with torch.profiler.record_function('ablation6.flux_gemm_only') if self.profile else nullcontext():
            self.gemm.forward(input_, weight, output_buf=self.partial, fast_accum=False)
        # Both baselines call this exact class, including pack_rows, all_to_all_single,
        # unpack_rows and source-ordered BF16 accumulation. The returned result is new.
        return self.tensor_codec.reduce(self.partial)
