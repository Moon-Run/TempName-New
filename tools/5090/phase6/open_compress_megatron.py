"""Native Megatron row projections with a private communication-boundary adapter.

RowParallelLinear.forward keeps its original code object. The BF16 arm invokes
its original ReduceScatter function; the INT8 arm changes only the private
ReduceScatter autograd forward implementation and reuses its exact backward.
No process-global Megatron function is replaced.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import types

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel import layers, mappings

ARMS = ('megatron-bf16', 'megatron-int8')


def _clone(function, replacements):
    result = types.FunctionType(function.__code__, dict(function.__globals__, **replacements),
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def native_sources():
    return {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (('layers.py', layers), ('mappings.py', mappings))}


class NativeBF16Reduction:
    """Count the original Megatron collective; allocate no codec workspace."""
    owns_output = owns_configuration = True
    selection_cpu = selection_sha256 = None

    def __init__(self, group):
        self.group = group
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.calls = 0
        self.profile = False
        self.wire_bytes_per_rank = self.bf16_wire_bytes_per_rank = 7*256*4096*2

    def state(self):
        return dict(quantization='none', collective='reduce_scatter_tensor', dtype='bfloat16',
                    calls=self.calls, encode_calls=0, decode_calls=0, all_to_all_calls=0,
                    owns_output=True, logical_remote_bytes=self.wire_bytes_per_rank)


class MegatronProjection:
    placement = 'tensor'
    owns_output = owns_configuration = True
    selection_cpu = selection_sha256 = None

    def __init__(self, group, m, n, k_local, arm, module=None, retain_partial=False):
        if arm not in ARMS or (m, n) != (2048, 4096) or k_local not in (512, 2048):
            raise ValueError('Expected a fixed TP8 Megatron projection')
        if dist.get_world_size(group) != 8 or dist.get_backend(group) != 'nccl':
            raise ValueError('Native projection requires the TP8 NCCL group')
        if group is not parallel_state.get_tensor_model_parallel_group():
            raise ValueError('Megatron and codec tensor-parallel groups differ')
        self.group, self.arm = group, arm
        self.m, self.n, self.k_local = m, n, k_local
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.retain_partial, self.partial = retain_partial, None
        self.calls, self.linear_calls = 0, 0
        self.audit_callback = None
        self.plan_sha256 = None
        self.workspace_info = []
        self.source_sha256 = native_sources()
        config = Path(__file__).with_name('config.json')
        if config.is_file():
            expected = json.loads(config.read_text()).get('native_source_sha256')
            if expected != self.source_sha256:
                raise ValueError('Loaded Megatron sources differ from the frozen build')
        self._probe_owned = module is None
        if module is None:
            from megatron.core.model_parallel_config import ModelParallelConfig
            config = ModelParallelConfig(tensor_model_parallel_size=8, sequence_parallel=True,
                params_dtype=torch.bfloat16, perform_initialization=False,
                gradient_accumulation_fusion=False, use_cpu_initialization=False)
            module = layers.RowParallelLinear(k_local*8, n, config=config,
                init_method=lambda value: value, bias=False, input_is_parallel=True, skip_bias_add=True)
        self.module = module
        if not isinstance(module, layers.RowParallelLinear):
            raise ValueError('Expected the pinned native RowParallelLinear class')
        if not module.input_is_parallel or not module.sequence_parallel or module.explicit_expert_comm \
                or module.gradient_accumulation_fusion or module.config._cpu_offloading_context is not None:
            raise ValueError('Unsupported native projection configuration')
        self._linear = layers.linear_with_grad_accumulation_and_async_allreduce
        self._linear_backward = layers.LinearWithGradAccumulationAndAsyncCommunication.backward
        self._native_rs = layers.RowParallelLinear.forward.__globals__['reduce_scatter_to_sequence_parallel_region']
        self._native_rs_class = mappings._ReduceScatterToSequenceParallelRegion
        self._native_rs_backward = self._native_rs_class.backward
        if arm == 'megatron-int8':
            from megatron_taco import TensorTacoRS
            self.tensor_codec = TensorTacoRS(group, m, n)
            # Clone only the upstream forward's globals. Its ctx fields and
            # argument signature remain upstream, and backward is the exact
            # original function object, including group/split/buffer handling.
            replacement = _clone(self._native_rs_class.forward,
                                  {'_reduce_scatter_along_first_dim': self._compressed_reduce})
            self._compressed_class = type('OpenCompressNativeReduceScatter', (torch.autograd.Function,),
                {'forward': staticmethod(replacement), 'backward': staticmethod(self._native_rs_backward)})
        else:
            self.tensor_codec = NativeBF16Reduction(group)
            self._compressed_class = None
        self.wire_bytes_per_rank = self.tensor_codec.wire_bytes_per_rank
        self.bf16_wire_bytes_per_rank = self.tensor_codec.bf16_wire_bytes_per_rank
        original = layers.RowParallelLinear.forward
        function = _clone(original, {
            'linear_with_grad_accumulation_and_async_allreduce': self._linear_forward,
            'reduce_scatter_to_sequence_parallel_region': self._reduce})
        self.native_forward = types.MethodType(function, module)

    @property
    def codec(self):
        # Shared worker reporting ABI: op.codec.tensor_codec.
        return self

    def _linear_forward(self, *args, **kwargs):
        if kwargs.get('gradient_accumulation_fusion') or kwargs.get('allreduce_dgrad') \
                or kwargs.get('sequence_parallel') or kwargs.get('bias') is not None:
            raise ValueError('Native row linear configuration changed')
        with torch.profiler.record_function('open_compress.megatron_gemm') if self.profile else nullcontext():
            output = self._linear(*args, **kwargs)
        self.linear_calls += 1
        if self.retain_partial:
            self.partial = output.reshape(self.m, self.n)
        return output

    def _compressed_reduce(self, partial, group=None, input_split_sizes=None, use_global_buffer=False):
        if (group is not None and group is not self.group) or input_split_sizes is not None or use_global_buffer:
            raise ValueError('Only the native equal-split, owned-output TP8 boundary is supported')
        output = self.tensor_codec.reduce(partial.reshape(self.m, self.n))
        return output.reshape(partial.shape[0]//8, *partial.shape[1:])

    def _reduce(self, partial):
        if self.arm == 'megatron-bf16':
            with torch.profiler.record_function('open_compress.megatron_bf16_reduce_scatter') if self.profile else nullcontext():
                output = self._native_rs(partial)
            self.tensor_codec.calls += 1
        else:
            self.tensor_codec.profile = self.profile
            output = self._compressed_class.apply(partial, self.group, None, False)
        if self.audit_callback is not None:
            self.audit_callback(partial, output)
        return output

    def forward_module(self, input_):
        if tuple(input_.shape) != (self.m, 1, self.k_local) or input_.dtype != torch.bfloat16 \
                or not self.module.weight.requires_grad:
            raise ValueError('Native training path requires fixed BF16 inputs and trainable weights')
        output = self.native_forward(input_)
        self.calls += 1
        return output

    def _bind_probe_weight(self, weight):
        if tuple(weight.shape) != (self.n, self.k_local) or weight.dtype != torch.bfloat16:
            raise ValueError('Wrong native operator weight')
        # Parameter wraps the provided storage; no GEMM or weight copy occurs.
        if self.module.weight.data_ptr() != weight.data_ptr():
            self.module.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, input_, weight, reduce_scatter_option=None):
        self._bind_probe_weight(weight)
        output, bias = self.forward_module(input_.reshape(self.m, 1, self.k_local))
        if bias is not None:
            raise ValueError('Standalone native probe has no bias')
        return output.reshape(self.m//8, self.n)

    def gemm_only(self, input_, weight):
        self._bind_probe_weight(weight)
        return self._linear_forward(input=input_.reshape(self.m, 1, self.k_local), weight=self.module.weight,
            bias=None, gradient_accumulation_fusion=False, allreduce_dgrad=False,
            sequence_parallel=False, grad_output_buffer=None)

    def state(self):
        return dict(arm=self.arm, calls=self.calls, linear_calls=self.linear_calls,
            native_forward_preserved=self.native_forward.__func__.__code__ is layers.RowParallelLinear.forward.__code__,
            native_linear_backward_preserved=self._linear is layers.linear_with_grad_accumulation_and_async_allreduce
                and self._linear_backward is layers.LinearWithGradAccumulationAndAsyncCommunication.backward,
            native_rs_backward_preserved=self._native_rs_backward is mappings._ReduceScatterToSequenceParallelRegion.backward
                and (self._compressed_class is None or self._compressed_class.backward is self._native_rs_backward),
            original_rs_forward=self.arm == 'megatron-bf16', private_function_globals=True,
            source_sha256=dict(self.source_sha256), flux_loaded=False)

    def close(self):
        """Release standalone probe cycles before measuring another role."""
        if not self._probe_owned:
            raise ValueError('A live training module must retain its native adapter')
        self.module._forward_impl = self._linear
        self.native_forward = self._compressed_class = None
        self.partial = self.audit_callback = self.module = None


class Adapter:
    """Instrument native modules while retaining their original linear code."""
    def __init__(self, model, policy, shapes, ring_reduction=False, control_backward=False):
        if policy not in ARMS:
            raise ValueError('Unsupported native adapter policy')
        if ring_reduction:
            raise ValueError('Native Megatron requires ring_reduction=False; this flag configures Flux GemmRS, not NCCL')
        if control_backward:
            raise ValueError('Native Megatron must retain its original backward')
        self.policy, self.allowed = policy, {tuple(shape) for shape in shapes}
        self.audit = True
        self.profile = os.environ.get('E2E_MODE') == 'profile'
        self.option = None
        self.modules, self.calls, self.observed = {}, {}, {}
        self.forward_checks, self.placements, self.selection_checks = {}, {}, {}
        self.native_quant_forwards, self.scope_ops, self.actual_routes = {}, {}, {}
        self.smoke_outputs, self.retained_outputs, self.retention_targets, self.retention_checks = {}, {}, {}, {}
        for name, module in model.named_modules():
            if not isinstance(module, layers.RowParallelLinear):
                continue
            if not name.endswith(('.self_attention.linear_proj', '.mlp.linear_fc2')):
                raise ValueError('Unexpected native target '+name)
            if module.forward.__func__ is not layers.RowParallelLinear.forward:
                raise ValueError('Native module forward was already replaced')
            self.modules[name], self.calls[name] = module, 0
            module.forward = types.MethodType(self.make_forward(name), module)
        if len(self.modules) != 2*model.config.num_layers:
            raise ValueError('Expected all 64 target native projections')

    def make_forward(self, name):
        from routing import route, module_role
        def forward(module, input_):
            shape = (input_.shape[0]*input_.shape[1], module.output_size, module.input_size)
            if self.audit:
                if shape not in self.allowed or input_.shape[-1] != module.input_size_per_partition:
                    raise ValueError('Native target shape changed')
                self.calls[name] += 1
                self.observed[name] = dict(M=shape[0],N=shape[1],K_global=shape[2],
                    input_shape=list(input_.shape),input_stride=list(input_.stride()),
                    weight_shape=list(module.weight.shape),K_local=input_.shape[-1])
            if name not in self.scope_ops:
                op = MegatronProjection(parallel_state.get_tensor_model_parallel_group(),shape[0],shape[1],
                    input_.shape[-1],self.policy,module=module)
                self.scope_ops[name] = op
                self.native_quant_forwards[name] = op.native_forward
                self.actual_routes[name] = route(self.policy,name)
                self.placements[name] = 'native' if self.policy == 'megatron-bf16' else 'tensor'
                self.selection_checks[name] = dict(mask_sha256=None,plan_sha256=None,
                    wire_bytes=op.wire_bytes_per_rank,bf16_wire_bytes=op.bf16_wire_bytes_per_rank)
            op = self.scope_ops[name]
            if os.environ.get('E2E_MODE') == 'smoke' and name not in self.forward_checks:
                from taco_support import check_taco_output
                def check(partial, actual):
                    with torch.no_grad():
                        independent = input_.reshape(-1,input_.shape[-1]).matmul(module.weight.t())
                        self.forward_checks[name] = check_taco_output(independent,
                            actual.reshape(-1,actual.shape[-1]),op.group)
                op.audit_callback = check
            output = op.forward_module(input_)
            op.audit_callback = None
            if os.environ.get('E2E_MODE') == 'smoke':
                self.record_smoke_output(name,output[0])
            return output
        return forward

    def record_smoke_output(self, name, output):
        from routing import module_role
        if name not in self.smoke_outputs:
            self.smoke_outputs[name] = hashlib.sha256(output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        role = module_role(name)
        self.retention_targets.setdefault(role,name)
        if self.retention_targets[role] != name or role in self.retention_checks:
            return
        retained = self.retained_outputs.setdefault(role,[])
        if len(retained) < 6:
            retained.append((output.detach(),output.detach().clone()))
            return
        if len({value.data_ptr() for value,_ in retained}) != 6 or not all(torch.equal(value,copy) for value,copy in retained):
            raise ValueError('Native output lifetime/ownership failed')
        self.retention_checks[role] = dict(module=name,outputs=6,independent_storage=True,
            unchanged_after_subsequent_call=True,passed=True)
        del self.retained_outputs[role]

    def stop_audit(self):
        from routing import route,module_role
        self.audit = False
        expected = {name:route(self.policy,name) for name in self.modules}
        if self.actual_routes != expected or set(self.scope_ops) != set(self.modules):
            raise ValueError('Native projection coverage is incomplete')
        self.post_audit_routes, self.forward_preserved = dict(expected), {}
        self.timed_calls = dict.fromkeys(self.modules,0)
        self.workspace_checks = {}
        for name,module in self.modules.items():
            target = module.forward
            op = self.scope_ops[name]
            self.forward_preserved[name] = op.native_forward.__func__.__code__ is layers.RowParallelLinear.forward.__code__
            if not self.forward_preserved[name]:
                raise ValueError('Native forward code changed')
            self.selection_checks[name].update(role=module_role(name),
                shape=[self.observed[name][k] for k in ('M','N','K_global')])
            def timed(module,input_,_target=target,_name=name,_role=module_role(name),_route=expected[name]):
                self.timed_calls[_name] += 1
                with torch.profiler.record_function('scope.'+_role+'.'+_route) if self.profile else nullcontext():
                    return _target(input_)
            module.forward = types.MethodType(timed,module)

    def codec_calls(self):
        return {name:op.tensor_codec.calls for name,op in self.scope_ops.items()}
