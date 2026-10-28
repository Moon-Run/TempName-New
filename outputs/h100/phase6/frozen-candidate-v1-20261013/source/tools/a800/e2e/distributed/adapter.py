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
        output = torch.empty((input_.shape[0] // parallel_state.get_tensor_model_parallel_world_size(), weight.shape[0]), device=input_.device, dtype=input_.dtype)
        torch.distributed.reduce_scatter_tensor(output, partial,
            group=parallel_state.get_tensor_model_parallel_group())
        return output


class Adapter:
    def __init__(self, model, policy, shapes, ring_reduction=False, control_backward=False, topology=None, validate=False):
        self.policy = policy
        self.topology = topology
        self.validate = validate
        self.checks = {}
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.local_group = self.tp_group
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
            # All world ranks create subgroups in the same order, even with multiple DP replicas.
            if topology['tp_nodes'] > 1:
                for base in range(0, topology['world_size'], topology['gpus_per_node']):
                    ranks = list(range(base, base + topology['gpus_per_node']))
                    group = torch.distributed.new_group(ranks)
                    if torch.distributed.get_rank() in ranks:
                        self.local_group = group
            flux.init_flux_shm(self.local_group)
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
        assert len(self.modules) == 2 * model.config.num_layers, list(self.modules)

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
                if adapter.control_backward:
                    op = NativeRSControl()
                else:
                    c = adapter.topology
                    local_op = adapter.flux.GemmRS(adapter.local_group, 1,
                        input_.shape[0] * input_.shape[1] // c['tp_nodes'], module.output_size,
                        torch.bfloat16, torch.bfloat16, transpose_weight=False,
                        fuse_reduction=False, ring_reduction=adapter.ring_reduction)
                    op = HierarchicalRS(local_op, adapter.tp_group, c) if c['tp_nodes'] > 1 else local_op
            assert tuple(input_.shape) == expected_shape
            output = GemmRSFunction.apply(input_, module.weight, op, adapter.option)
            if adapter.validate and name not in adapter.checks and adapter.policy != 'native':
                flat = input_.detach().reshape(-1, input_.shape[-1]).contiguous()
                reference = NativeRSControl().forward(flat, module.weight.detach()).reshape_as(output)
                delta = (output.detach().float() - reference.float())
                c = adapter.topology
                passed = torch.allclose(output.detach(), reference, atol=c['atol'], rtol=c['rtol'])
                adapter.checks[name] = dict(passed=passed, max_abs=delta.abs().max().item(),
                    relative_l2=(delta.norm()/reference.float().norm().clamp_min(1e-12)).item())
                ok = torch.tensor(int(passed), device=input_.device)
                torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
                assert ok.item(), (name, adapter.checks[name])
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


class HierarchicalRS:
    """Node-local GEMM-RS followed by a ring exchange over the full TP group.

    Node-major contiguous TP ranks are required and checked by the launcher config.
    Each local output is cloned before the Flux workspace can be reused.
    """
    def __init__(self, local_op, tp_group, topology):
        self.local_op = local_op
        self.group = tp_group
        self.nodes = topology['tp_nodes']
        self.local_tp = topology['local_tp']
        self.rank = torch.distributed.get_rank(tp_group)
        self.node = self.rank // self.local_tp
        self.world = torch.distributed.get_world_size(tp_group)
        self.base = torch.distributed.get_rank() - self.rank

    def forward(self, input_, weight, reduce_scatter_option=None):
        assert input_.shape[0] % self.nodes == 0
        outputs = []
        for chunk in input_.chunk(self.nodes, dim=0):
            chunk = chunk.contiguous()
            outputs.append(self.local_op.forward(chunk, weight,
                reduce_scatter_option=reduce_scatter_option).clone())
            # GemmRS synchronizes before final local reduction. All peers must
            # finish consuming that workspace before another GEMM overwrites it.
            self.local_op.forward_barrier(chunk, weight)
        result = outputs[self.node]
        for distance in range(1, self.nodes):
            send_node = (self.node + distance) % self.nodes
            send_rank = self.base + (self.rank + distance*self.local_tp) % self.world
            recv_rank = self.base + (self.rank - distance*self.local_tp) % self.world
            received = torch.empty_like(result)
            requests = torch.distributed.batch_isend_irecv([
                torch.distributed.P2POp(torch.distributed.isend, outputs[send_node], send_rank, self.group),
                torch.distributed.P2POp(torch.distributed.irecv, received, recv_rank, self.group)])
            for request in requests:
                request.wait()
            result.add_(received)
        return result
