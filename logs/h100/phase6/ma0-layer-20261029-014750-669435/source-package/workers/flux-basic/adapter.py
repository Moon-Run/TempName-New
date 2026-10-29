"""Restricted BF16 RowParallelLinear GEMM-RS integration for the frozen GPT trial.

Only attention.linear_proj and mlp.linear_fc2 forward paths are replaced.
Backward keeps the native sequence AllGather and local dgrad/wgrad GEMMs.
Unsupported shapes/configurations fail before timing rather than silently falling back.
"""
import types
import hashlib
import os
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
from taco_support import TacoForward, check_taco_output
from routing import codec_placement, enabled, is_native, is_fused_v2, is_flux_tensor, is_tensor, module_role, route
from megatron_taco import TensorTacoRS, QuantizedRS


class GemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, op, option):
        ctx.save_for_backward(input_, weight)
        shape = input_.shape
        flat = input_.reshape(-1, shape[-1]).contiguous()
        # Give the autograd graph its own output lifetime, independent of Flux workspaces.
        output = op.forward(flat, weight, reduce_scatter_option=option)
        if not getattr(op, 'owns_output', False):
            output = output.clone()
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
        self.forward_checks = {}
        self.placements = {}
        self.selection_checks = {}
        self.native_quant_forwards = {}
        self.smoke_outputs = {}
        self.retained_outputs = {}
        self.retention_targets = {}
        self.retention_checks = {}
        self.scope_ops = {}
        self.actual_routes = {}
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.flux = None
        if not is_native(policy):
            import flux
            self.flux = flux
            if not is_flux_tensor(policy):
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
        assert len(self.modules) == 2 * model.config.num_layers, list(self.modules)

    def make_forward(self, name, native):
        adapter = self
        op = None
        expected_shape = None
        native_quant = None

        def forward(module, input_):
            nonlocal op, expected_shape, native_quant
            if adapter.audit:
                shape = (input_.shape[0] * input_.shape[1], module.output_size, module.input_size)
                assert shape in adapter.allowed, (name, shape)
                assert input_.dtype == module.weight.dtype
                assert input_.shape[-1] == module.input_size_per_partition
                adapter.calls[name] += 1
                adapter.observed[name] = dict(M=shape[0], N=shape[1], K_global=shape[2],
                    input_shape=list(input_.shape), input_stride=list(input_.stride()),
                    weight_shape=list(module.weight.shape), K_local=input_.shape[-1])
            if not enabled(adapter.policy, name):
                adapter.actual_routes[name] = 'native_bf16'
                adapter.placements[name] = 'native'
                return native(input_)
            if is_native(adapter.policy):
                if native_quant is None:
                    assert native.__func__ is RowParallelLinear.forward
                    op=TensorTacoRS(parallel_state.get_tensor_model_parallel_group(),
                                   input_.shape[0]*input_.shape[1],module.output_size)
                    adapter.scope_ops[name] = op
                    adapter.actual_routes[name] = 'tensor'
                    def quantized_rs(partial):
                        result=QuantizedRS.apply(partial,op)
                        if os.environ.get('E2E_MODE')=='smoke' and name not in adapter.forward_checks:
                            adapter.forward_checks[name]=check_taco_output(
                                partial.reshape(-1,partial.shape[-1]),result.reshape(-1,result.shape[-1]),op.group)
                        return result
                    # Copy only the function globals for this module: preserve native
                    # GEMM/bias/linear backward and avoid a process-global monkeypatch.
                    globals_=dict(native.__func__.__globals__,reduce_scatter_to_sequence_parallel_region=quantized_rs)
                    function=types.FunctionType(native.__func__.__code__,globals_,native.__func__.__name__,
                                                native.__func__.__defaults__,native.__func__.__closure__)
                    native_quant=types.MethodType(function,module)
                    adapter.native_quant_forwards[name]=native_quant
                    adapter.placements[name]='tensor'
                    adapter.selection_checks[name]=dict(mask_sha256=None,wire_bytes=op.wire_bytes_per_rank,
                                                        bf16_wire_bytes=op.bf16_wire_bytes_per_rank)
                return native_quant(input_)
            if op is None:
                expected_shape = tuple(input_.shape)
                if enabled(adapter.policy, name) and not is_native(adapter.policy):
                    placement = codec_placement(adapter.policy)
                    op = TacoForward(parallel_state.get_tensor_model_parallel_group(),
                        input_.shape[0] * input_.shape[1], module.output_size,
                        input_.shape[-1], adapter.policy, module_role(name))
                    adapter.scope_ops[name] = op
                    adapter.actual_routes[name] = route(adapter.policy, name)
                    adapter.placements[name] = op.codec.placement
                    adapter.selection_checks[name] = dict(mask_sha256=getattr(op.codec,'selection_sha256',None),
                        wire_bytes=op.codec.wire_bytes_per_rank,bf16_wire_bytes=op.codec.bf16_wire_bytes_per_rank)
                else:
                    raise RuntimeError('Unexpected scope fallback')
                    op = adapter.flux.GemmRS(parallel_state.get_tensor_model_parallel_group(), 1,
                        input_.shape[0] * input_.shape[1], module.output_size,
                        torch.bfloat16, torch.bfloat16, transpose_weight=False,
                        fuse_reduction=False, ring_reduction=adapter.ring_reduction)
                    adapter.placements[name] = 'bf16'
            assert tuple(input_.shape) == expected_shape
            output = GemmRSFunction.apply(input_, module.weight, op, adapter.option)
            if os.environ.get('E2E_MODE') == 'smoke' and name not in adapter.forward_checks:
                with torch.no_grad():
                    flat = input_.reshape(-1, input_.shape[-1])
                    partial = flat.matmul(module.weight.t())
                    ref = torch.empty_like(output).reshape(-1, output.shape[-1])
                    torch.distributed.reduce_scatter_tensor(ref, partial, group=parallel_state.get_tensor_model_parallel_group())
                    actual = output.reshape_as(ref)
                    if adapter.placements[name] in ('fused','separate','tensor'):
                        adapter.forward_checks[name] = check_taco_output(partial, actual,
                            parallel_state.get_tensor_model_parallel_group(), getattr(op.codec,'selection_cpu',None))
                    else:
                        torch.testing.assert_close(actual, ref, atol=.02, rtol=.02)
                        delta = actual.float() - ref.float()
                        adapter.forward_checks[name] = dict(max_abs=delta.abs().max().item(), relative_l2=(delta.norm()/ref.float().norm()).item())

            if os.environ.get('E2E_MODE') == 'smoke':
                adapter.record_smoke_output(name, output)
            if module.skip_bias_add:
                return output, module.bias
            return output + module.bias if module.bias is not None else output, None
        return forward

    def record_smoke_output(self, name, output):
        # Only smoke workers hash or retain tensors. There are no extra forward calls.
        if name not in self.smoke_outputs:
            data = output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            self.smoke_outputs[name] = hashlib.sha256(data).hexdigest()
        role = module_role(name)
        if role not in self.retention_targets:
            self.retention_targets[role] = name
        if self.retention_targets[role] != name or role in self.retention_checks:
            return
        retained = self.retained_outputs.setdefault(role, [])
        if len(retained) < 6:
            retained.append((output.detach(), output.detach().clone()))
            return
        # A seventh real model call has now reused the internal workspace.
        pointers = [value.data_ptr() for value, _ in retained]
        assert len(set(pointers)) == 6
        assert all(torch.equal(value, reference) for value, reference in retained)
        self.retention_checks[role] = dict(module=name, outputs=6,
            independent_storage=True, unchanged_after_subsequent_call=True, passed=True)
        del self.retained_outputs[role]

    def stop_audit(self):
        self.audit = False
        expected = {name: route(self.policy,name) for name in self.modules}
        assert self.actual_routes == expected
        assert set(self.scope_ops) == {name for name in self.modules if enabled(self.policy,name)}
        assert len({id(op) for op in self.scope_ops.values()}) == len(self.scope_ops)
        self.post_audit_routes = {}
        self.forward_preserved = {}
        self.timed_calls = {name:0 for name in self.modules}
        self.workspace_checks = {} if is_native(self.policy) else {name:op.workspace_info for name,op in self.scope_ops.items()}
        for field in ('packet_ptr','mask_ptr','gemm_output_ptr'):
            pointers = [slot[field] for slots in self.workspace_checks.values() for slot in slots if field in slot]
            assert all(p > 0 for p in pointers) and len(pointers) == len(set(pointers))
        for name,module in self.modules.items():
            if not enabled(self.policy,name):
                target = types.MethodType(RowParallelLinear.forward,module)
                actual = 'native_bf16'
            elif is_native(self.policy):
                target = self.native_quant_forwards[name]
                actual = 'tensor'
            else:
                target = module.forward
                assert self.scope_ops[name].owns_output == (not is_fused_v2(self.policy))
                actual = route(self.policy, name)
            self.forward_preserved[name] = target.__func__.__code__ is RowParallelLinear.forward.__code__
            assert self.forward_preserved[name] == (actual not in ('candidate', 'flux_tensor'))
            self.post_audit_routes[name] = actual
            if enabled(self.policy,name):
                op = self.scope_ops[name]
                self.selection_checks[name].update(role=module_role(name),
                    shape=[self.observed[name][k] for k in ('M','N','K_global')],
                    plan_sha256=None if is_native(self.policy) else op.plan_sha256)
            def timed(module,input_,_target=target,_name=name,_role=module_role(name),_route=actual):
                self.timed_calls[_name] += 1
                if self.profile:
                    with torch.profiler.record_function('scope.'+_role+'.'+_route):
                        return _target(input_)
                return _target(input_)
            module.forward = types.MethodType(timed,module)
        assert self.post_audit_routes == expected

    def codec_calls(self):
        if not is_tensor(self.policy):
            return {}
        codecs = {name: op.codec.tensor_codec if is_flux_tensor(self.policy) else op
                  for name, op in self.scope_ops.items()}
        assert all(type(codec) is TensorTacoRS for codec in codecs.values())
        return {name: codec.calls for name, codec in codecs.items()}
