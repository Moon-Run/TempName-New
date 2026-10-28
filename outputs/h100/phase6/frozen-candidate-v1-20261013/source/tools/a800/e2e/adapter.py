"""Restricted BF16 RowParallelLinear GEMM-RS integration for the frozen GPT trial.

Only attention.linear_proj and mlp.linear_fc2 forward paths are replaced.
Backward keeps the native sequence AllGather and local dgrad/wgrad GEMMs.
Unsupported shapes/configurations fail before timing rather than silently falling back.
"""
import types
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim


class GemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, op, option):
        ctx.save_for_backward(input_, weight)
        shape = input_.shape
        flat = input_.reshape(-1, shape[-1]).contiguous()
        # Give the autograd graph its own output lifetime, independent of Flux workspaces.
        output = op.forward(flat, weight, reduce_scatter_option=option).clone()
        return output.reshape(shape[0] // parallel_state.get_tensor_model_parallel_world_size(),
                              shape[1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        full_grad = _gather_along_first_dim(grad_output.contiguous())
        grad_input = full_grad.matmul(weight)
        grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
            input_.reshape(-1, input_.shape[-1]))
        return grad_input, grad_weight, None, None


class NativeRSControl:
    def forward(self, input_, weight, reduce_scatter_option=None):
        partial = input_.matmul(weight.t())
        output = torch.empty((input_.shape[0] // 4, weight.shape[0]), device=input_.device, dtype=input_.dtype)
        torch.distributed.reduce_scatter_tensor(output, partial,
            group=parallel_state.get_tensor_model_parallel_group())
        return output


class Adapter:
    def __init__(self, model, policy, shapes, ring_reduction=False, control_backward=False):
        self.policy = policy
        self.ring_reduction = ring_reduction
        self.control_backward = control_backward
        self.option = None
        self.allowed = {tuple(s) for s in shapes}
        self.modules = {}
        self.calls = {}
        self.audit = True
        self.observed = {}
        self.flux = None
        if policy != 'native':
            import flux
            self.flux = flux
            flux.init_flux_shm(parallel_state.get_tensor_model_parallel_group())
            self.option = flux.ReduceScatterOption()
            for key, value in dict(use_1d_ring=True, use_p2p_read=True, use_cudaMemcpyAsync=False,
                                   use_gemmk=False, per_tile_flags=False, use_barrier_queue=False,
                                   num_blocks=6, n_split=1).items():
                setattr(self.option, key, value)
            self.option.ring_mode = flux.RingMode.Ring1D
        for name, module in model.named_modules():
            if not isinstance(module, RowParallelLinear):
                continue
            assert name.endswith(('.self_attention.linear_proj', '.mlp.linear_fc2')), name
            assert module.input_is_parallel and module.sequence_parallel
            assert not module.explicit_expert_comm and not module.gradient_accumulation_fusion
            assert module.weight.dtype == torch.bfloat16 or (policy == 'native' and module.weight.dtype == torch.float32)
            assert module.config._cpu_offloading_context is None
            self.modules[name] = module
            self.calls[name] = 0
            native = module.forward
            module.forward = types.MethodType(self.make_forward(name, native), module)
        assert len(self.modules) == 8, list(self.modules)

    def make_forward(self, name, native):
        adapter = self
        op = None
        expected_shape = None

        def forward(module, input_):
            nonlocal op, expected_shape
            if adapter.audit:
                shape = (input_.shape[0] * input_.shape[1], module.output_size, module.input_size)
                assert shape in adapter.allowed, (name, shape)
                assert input_.dtype == module.weight.dtype
                assert input_.shape[-1] == module.input_size_per_partition
                adapter.calls[name] += 1
                adapter.observed[name] = dict(M=shape[0], N=shape[1], K_global=shape[2],
                    input_shape=list(input_.shape), input_stride=list(input_.stride()),
                    weight_shape=list(module.weight.shape), K_local=input_.shape[-1])
            if adapter.policy == 'native' and not adapter.control_backward:
                return native(input_)
            if op is None:
                expected_shape = tuple(input_.shape)
                op = NativeRSControl() if adapter.control_backward else adapter.flux.GemmRS(parallel_state.get_tensor_model_parallel_group(), 1,
                    input_.shape[0] * input_.shape[1], module.output_size,
                    torch.bfloat16, torch.bfloat16, transpose_weight=False,
                    fuse_reduction=False, ring_reduction=adapter.ring_reduction)
            assert tuple(input_.shape) == expected_shape
            output = GemmRSFunction.apply(input_, module.weight, op, adapter.option)
            if module.skip_bias_add:
                return output, module.bias
            return output + module.bias if module.bias is not None else output, None
        return forward

    def stop_audit(self):
        self.audit = False
        # Restore the exact original forward for the native timing baseline.
        if self.policy == 'native':
            for module in self.modules.values():
                module.forward = types.MethodType(RowParallelLinear.forward, module)
