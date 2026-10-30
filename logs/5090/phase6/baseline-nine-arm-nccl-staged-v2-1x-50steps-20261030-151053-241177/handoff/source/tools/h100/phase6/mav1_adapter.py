"""Instance-only native linear plans over the complete, unchanged MA11 model.

The original MA4-F outer row wrapper (residual ownership, smoke checks, and
profile markers) is retained. Its Python Function entry dispatches directly to
a C++ autograd Function; neither native backward calls back into Python. Plans
own metadata/handles, while every Function call owns its saved activation.
"""
import hashlib
import inspect
import math
import types

import ma4_f_adapter
import ma8_adapter as cache

PROTOCOL = 'h100-mav1-native-linear-v1'
WORLD_SIZE = 8
LOCAL_SHAPE = (256, 1, 4096)
OUTPUT_WIDTHS = {'qkv': 1536, 'fc1': 2048}
ROW_WIDTHS = {'attn': 512, 'fc2': 2048}
SUFFIXES = (('.self_attention.linear_qkv', 'qkv'), ('.mlp.linear_fc1', 'fc1'),
            ('.self_attention.linear_proj', 'attn'), ('.mlp.linear_fc2', 'fc2'))
SELF_TEST_CASES = 8
WGRAD_PREPARE_SHA256 = '79c20a0283c05c6e020d7d9f2c617202936d5b34c4d0000fb9b88f9f1347bf1f'
ROW_GATHER_SHA256 = '34b838bf32851b7aa0e9ef325e476f696f811d7b542b03701035ae3d1e571714'


def require(condition, message):
    if not condition:
        raise RuntimeError('MAV1 ' + message)


def module_roles():
    return {f'decoder.layers.{index}{suffix}': role for index in range(32)
            for suffix, role in SUFFIXES}


def plan_ids():
    roles = module_roles()
    names = [name for name, role in roles.items() if role in OUTPUT_WIDTHS]
    names += [name for name, role in roles.items() if role in ROW_WIDTHS]
    return {name: index for index, name in enumerate(names)}


def _cell(value):
    return (lambda: value).__closure__[0]


def clone_row_forward(function, dispatch, op, shape):
    """Change only Function dispatch and the already-bound original row op.

    New closure cells prevent mutating the original callable, which remains an
    independent oracle. Original audit/consumer/profile handling stays literal.
    """
    require(function.__closure__ is not None, 'row wrapper has no closure')
    closure = dict(zip(function.__code__.co_freevars, function.__closure__))
    require({'op', 'expected_shape', 'adapter', 'name'} <= set(closure),
            'original row closure changed')
    require(closure['op'].cell_contents is None and
            closure['expected_shape'].cell_contents is None,
            'install before the first model forward')
    closure['op'], closure['expected_shape'] = _cell(op), _cell(shape)
    namespace = dict(function.__globals__, ConsumerGemmRSFunction=dispatch)
    result = types.FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, tuple(closure[key] for key in
                                                           function.__code__.co_freevars))
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def exact_equal(torch, left, right):
    return (tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype
            and torch.equal(left.detach().contiguous().view(torch.uint8),
                            right.detach().contiguous().view(torch.uint8)))


def compare_autograd(torch, reference, candidate, inputs, gradient):
    """Byte oracle on private leaves: never executes the model's DDP hooks."""
    results = []
    for function in (reference, candidate):
        leaves = tuple(value.detach().clone().requires_grad_(True) if value is not None
                       else None for value in inputs)
        output = function(*leaves)
        active = tuple(value for value in leaves if value is not None)
        gradients = torch.autograd.grad(output, active, gradient)
        require(all(value.grad is None for value in active),
                'standalone oracle unexpectedly accumulated leaf gradients')
        results.append((output.detach(), *gradients))
    require(len(results[0]) == len(results[1]) and
            all(exact_equal(torch, a, b) for a, b in zip(*results)),
            'native forward/input/weight/bias/residual gradient differs bytewise')


def deterministic_tensor(torch, shape, device, salt=0):
    values = torch.arange(math.prod(shape), dtype=torch.int32, device=device)
    return ((values.add(salt).remainder(29).float() - 14) / 256).to(
        torch.bfloat16).reshape(shape)


def retain_row_storage(plan, slots):
    """Transfer strong Tensor aliases to the plan saved by native autograd.

    Keeping Python codec objects on the tracker is not enough: an output's
    autograd context can outlive the model/tracker that created it.
    """
    require(len(slots) == 2, 'row storage requires two complete slots')
    buffers = []
    for slot in slots:
        require(len(slot.peers) == WORLD_SIZE and slot.selected is not None,
                'row IPC packet or selection storage is incomplete')
        buffers.extend(slot.peers)
        buffers.append(slot.selected)
        if slot.quant_tiles is not None:
            buffers.append(slot.quant_tiles)
    plan.retain_storage(buffers)


def native_self_test(torch, extension, group, rank, taco_factory, policy,
                     column_reference, row_reference, option):
    """Actual GPU byte comparisons, separate plans/storage, no training RNG.

    Each role runs a contiguous and a noncontiguous output-gradient case. The
    test plans never enter the model tracker or its MA4-F slot counters.
    """
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    device = torch.device('cuda', torch.cuda.current_device())
    cases = 0
    for role in ('qkv', 'fc1', 'attn', 'fc2'):
        if role in OUTPUT_WIDTHS:
            width = OUTPUT_WIDTHS[role]
            plan = extension.Mav1ColumnPlan(group, width, False, 1000 + cases)
            inputs = (deterministic_tensor(torch, LOCAL_SHAPE, device, rank),
                      deterministic_tensor(torch, (width, 4096), device, 3),
                      deterministic_tensor(torch, (width,), device, 5) if role == 'qkv' else None)
            reference = lambda x, w, b: column_reference.apply(
                x, w, b, False, False, True, None, None)
            candidate = plan.apply
            output_shape = (2048, 1, width)
            owners = ()
        else:
            width = ROW_WIDTHS[role]
            module_role = 'attn' if role == 'attn' else 'mlp'
            oracle_op = taco_factory(group, 2048, 4096, width, policy, module_role)
            candidate_op = taco_factory(group, 2048, 4096, width, policy, module_role)
            slots = candidate_op.codec.slots
            plan = extension.Mav1RowPlan(group, slots[0].op, slots[1].op,
                                         slots[0].option, width, False, 1000 + cases)
            retain_row_storage(plan, slots)
            inputs = (deterministic_tensor(torch, (2048, 1, width), device, rank),
                      deterministic_tensor(torch, (4096, width), device, 3),
                      deterministic_tensor(torch, (4096,), device, 5),
                      deterministic_tensor(torch, LOCAL_SHAPE, device, rank + 7))
            reference = lambda x, w, b, r: row_reference.apply(x, w, b, r, oracle_op, option)
            candidate = plan.apply
            output_shape = LOCAL_SHAPE
            # Keep all Python IPC owners alive through both backward calls.
            owners = (oracle_op, candidate_op)
        for strided in (False, True):
            if strided:
                gradient = deterministic_tensor(torch,
                    (*output_shape[:-1], output_shape[-1] * 2), device, 11)[..., ::2]
                require(not gradient.is_contiguous(), 'strided gradient self-test lost its stride')
            else:
                gradient = deterministic_tensor(torch, output_shape, device, 11)
            compare_autograd(torch, reference, candidate, inputs, gradient)
            cases += 1
        require(list(plan.state())[1:3] == [2, 2], 'self-test did not execute both native paths')
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        del candidate, reference, plan, inputs, gradient, owners
        if role in ROW_WIDTHS:
            del slots, oracle_op, candidate_op
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and
            torch.equal(gpu_rng, torch.cuda.get_rng_state()), 'self-test changed training RNG')
    return dict(passed=True, cases=cases, roles=['qkv', 'fc1', 'attn', 'fc2'],
                strided_gradients=True, private_leaves=True, rng_unchanged=True)


class RowDispatch:
    def __init__(self, tracker, plan, op, option):
        self.tracker, self.plan, self.op, self.option = tracker, plan, op, option

    def apply(self, input_, weight, bias, residual, op, option):
        try:
            require(op is self.op and option is self.option, 'row operation/option changed')
            return self.plan.apply(input_, weight, bias, residual)
        except BaseException:
            self.tracker.failed = True
            raise


def column_forward(tracker, plan, skip_bias_add):
    def forward(module, input_, weight=None, runtime_gather_output=None):
        try:
            require(module.training and weight is None and runtime_gather_output in (None, False),
                    'column runtime weight/output override or eval is unsupported')
            output = plan.apply(input_, module.weight, None if skip_bias_add else module.bias)
            return output, module.bias if skip_bias_add else None
        except BaseException:
            tracker.failed = True
            raise
    return forward


class Tracker:
    def __init__(self, rank, mode, identities):
        self.rank, self.mode, self.identities = rank, mode, identities
        self.plans, self.owners, self.self_test = {}, [], None
        self.failed = False

    def state(self):
        require(not self.failed, 'cannot report a failed native installation')
        modules = {}
        totals = dict(column_forward=0, column_backward=0, row_forward=0, row_backward=0)
        for name, item in self.plans.items():
            native = list(item['plan'].state())
            require(len(native) == 6 and native[0] == 1 and native[3] == 0,
                    'native state ABI or failure: ' + name)
            role = item['role']
            kind = 'column' if role in OUTPUT_WIDTHS else 'row'
            totals[kind + '_forward'] += native[1]
            totals[kind + '_backward'] += native[2]
            modules[name] = dict(role=role, plan_id=item['plan_id'],
                                 forward_calls=native[1], backward_calls=native[2],
                                 native=native)
        return dict(protocol=PROTOCOL, installed=True, failed=False, mode=self.mode,
                    rank=self.rank, world_size=WORLD_SIZE, plan_count=len(modules),
                    modules=modules, totals=totals, source_sha256=self.identities,
                    self_test=self.self_test, per_call_audits=0,
                    retains_original_process_group=True, native_backward=True)


def validate_state(state, steps, mode, rank=None):
    require(type(steps) is int and steps > 0 and mode in ('smoke', 'timing', 'profile'),
            'invalid native state schedule')
    require(isinstance(state, dict) and state.get('protocol') == PROTOCOL
            and state.get('installed') is True and state.get('failed') is False
            and state.get('mode') == mode and state.get('world_size') == WORLD_SIZE
            and state.get('plan_count') == 128 and state.get('per_call_audits') == 0
            and state.get('retains_original_process_group') is True
            and state.get('native_backward') is True
            and type(state.get('rank')) is int and 0 <= state['rank'] < WORLD_SIZE
            and (rank is None or state['rank'] == rank), 'native state identity or mode mismatch')
    require(state.get('source_sha256') == source_identities(), 'native source identities changed')
    require(state.get('totals') == {key: steps * 256 for key in
            ('column_forward', 'column_backward', 'row_forward', 'row_backward')},
            'native execution totals changed')
    modules, ids = state.get('modules', {}), plan_ids()
    require(set(modules) == set(module_roles()), 'native module coverage changed')
    for name, role in module_roles().items():
        item = modules[name]
        native = item.get('native', [])
        width = OUTPUT_WIDTHS.get(role, ROW_WIDTHS.get(role))
        require(item.get('role') == role and item.get('plan_id') == ids[name]
                and item.get('forward_calls') == steps * 4
                and item.get('backward_calls') == steps * 4
                and native == [1, steps * 4, steps * 4, 0,
                               -1 if role in OUTPUT_WIDTHS else (steps * 4) % 2, width],
                'native module execution mismatch: ' + name)
    expected_test = dict(passed=True, cases=SELF_TEST_CASES,
                         roles=['qkv', 'fc1', 'attn', 'fc2'], strided_gradients=True,
                         private_leaves=True, rng_unchanged=True)
    require(state.get('self_test') == (expected_test if mode == 'smoke' else None),
            'missing byte self-test or self-test leaked into timing/profile')
    return True


def source_identities():
    return dict(column_class=cache.CLASS_SHA256,
                column_forward=cache.COLUMN_FORWARD_SHA256,
                linear_wrapper=cache.LINEAR_WRAPPER_SHA256,
                wgrad_prepare=WGRAD_PREPARE_SHA256, row_gather=ROW_GATHER_SHA256,
                row_class=hashlib.sha256(cache.normalized(
                    ma4_f_adapter.consumer_autograd_source()).encode()).hexdigest())


def validate_extension(extension):
    require(hasattr(extension, 'Mav1ColumnPlan') and hasattr(extension, 'Mav1RowPlan')
            and callable(getattr(extension, 'mav1_native_protocol', None)),
            'candidate native extension is missing')
    require(extension.mav1_native_protocol() == 1, 'candidate native ABI version changed')


def install_native_linear(model, adapter, mode):
    import torch
    import flux_ths_pybind as extension
    from megatron.core.tensor_parallel import layers

    require(mode in ('smoke', 'timing', 'profile'), 'invalid execution mode')
    require(not hasattr(model, '_mav1_native_linear'), 'duplicate installation')
    require(torch.__version__.split('+')[0] == '2.6.0', 'requires pinned PyTorch 2.6.0')
    validate_extension(extension)
    config = model.config
    require(model.training and config.num_layers == 32 and config.hidden_size == 4096
            and config.tensor_model_parallel_size == 8 and config.pipeline_model_parallel_size == 1
            and config.context_parallel_size == 1 and config.sequence_parallel
            and not config.gradient_accumulation_fusion and config.recompute_granularity is None
            and not config.enable_cuda_graph and not config.external_cuda_graph
            and not getattr(config, 'cpu_offloading', False) and not config.defer_embedding_wgrad_compute
            and config.hidden_dropout == 0 and config.attention_dropout == 0,
            'requires frozen MA11 TP8/PP1/CP1 training')
    roles, ids = module_roles(), plan_ids()
    common = getattr(model, '_ma8_activation_setup', {})
    require(common.get('installed') is True and common.get('protocol') == cache.PROTOCOL
            and common.get('arm') == 'F4-cache' and common.get('module_roles') ==
            {name: role for name, role in roles.items() if role in OUTPUT_WIDTHS},
            'requires complete original MA8 cache')
    require(adapter._consumer_enabled and not adapter._consumer_failed
            and not adapter.scope_ops and set(adapter.modules) ==
            {name for name, role in roles.items() if role in ROW_WIDTHS},
            'requires fresh complete MA4-F consumer installation')
    for function, expected in (
            (layers.LinearWithGradAccumulationAndAsyncCommunication, cache.CLASS_SHA256),
            (layers.ColumnParallelLinear.forward, cache.COLUMN_FORWARD_SHA256),
            (layers.linear_with_grad_accumulation_and_async_allreduce, cache.LINEAR_WRAPPER_SHA256),
            (layers.prepare_input_tensors_for_wgrad_compute, WGRAD_PREPARE_SHA256)):
        cache.require_source(inspect.getsource(function), expected)
    modules = dict(model.named_modules())
    selected = []
    column_reference, row_reference, taco_factory = None, None, None
    for name, role in roles.items():
        require(name in modules, 'missing selected module: ' + name)
        module = modules[name]
        function = getattr(module.forward, '__func__', None)
        require(function is not None and module.training and module.weight.requires_grad
                and module.weight.dtype == torch.bfloat16 and module.weight.is_cuda
                and module.weight.is_contiguous(), 'invalid module or weight: ' + name)
        if role in OUTPUT_WIDTHS:
            require(type(module) is layers.ColumnParallelLinear and module.sequence_parallel
                    and not module.allreduce_dgrad and not module.gradient_accumulation_fusion
                    and not module.explicit_expert_comm and not module.disable_grad_reduce
                    and not module.gather_output and tuple(module.weight.shape) ==
                    (OUTPUT_WIDTHS[role], 4096) and
                    function.__code__ is layers.ColumnParallelLinear.forward.__code__,
                    'original column wrapper changed: ' + name)
            wrapper = function.__globals__.get('linear_with_grad_accumulation_and_async_allreduce')
            require(wrapper is not None and wrapper.__code__ is
                    layers.linear_with_grad_accumulation_and_async_allreduce.__code__,
                    'original cached linear wrapper changed')
            cached = wrapper.__globals__.get('LinearWithGradAccumulationAndAsyncCommunication')
            require(cached is not None and cached.__name__ == '_MA8CachedRawLinear',
                    'column cache route changed')
            column_reference = cached
        else:
            require(adapter.modules[name] is module and module.sequence_parallel
                    and module.input_is_parallel and module.skip_bias_add
                    and tuple(module.weight.shape) == (4096, ROW_WIDTHS[role])
                    and module.bias is not None and tuple(module.bias.shape) == (4096,),
                    'original row module changed: ' + name)
            reference = function.__globals__.get('ConsumerGemmRSFunction')
            require(reference is not None and cache.normalized(inspect.getsource(reference)) ==
                    cache.normalized(ma4_f_adapter.consumer_autograd_source()),
                    'original row autograd source changed')
            require(taco_factory is None or taco_factory is function.__globals__['TacoForward'],
                    'multiple row factories')
            row_reference, taco_factory = reference, function.__globals__['TacoForward']
        selected.append((name, role, module, function))
    cache.require_source(inspect.getsource(row_reference.backward.__globals__[
        '_gather_along_first_dim']), ROW_GATHER_SHA256)
    group, rank = layers.get_tensor_model_parallel_group(), layers.get_tensor_model_parallel_rank()
    require(layers.get_tensor_model_parallel_world_size() == WORLD_SIZE and
            torch.distributed.get_world_size(group) == WORLD_SIZE and
            torch.distributed.get_rank(group) == rank, 'original TP group changed')
    tracker = Tracker(rank, mode, source_identities())
    if mode == 'smoke':
        tracker.self_test = native_self_test(torch, extension, group, rank, taco_factory,
            adapter.policy, column_reference, row_reference, adapter.option)
    pending, row_owners = [], []
    # Constructors/IPC registration are collective. Every rank uses the same
    # layer/role order before any per-instance callable is changed.
    for name, role, module, function in selected:
        if role in OUTPUT_WIDTHS:
            plan = extension.Mav1ColumnPlan(group, OUTPUT_WIDTHS[role], mode == 'profile', ids[name])
            forward = column_forward(tracker, plan, module.skip_bias_add)
        else:
            width = ROW_WIDTHS[role]
            op = taco_factory(group, 2048, 4096, width, adapter.policy,
                              'attn' if role == 'attn' else 'mlp')
            slots = op.codec.slots
            require(len(slots) == 2 and op.owns_output and op.owns_configuration,
                    'row requires original independent double workspaces')
            plan = extension.Mav1RowPlan(group, slots[0].op, slots[1].op,
                                         slots[0].option, width, mode == 'profile', ids[name])
            retain_row_storage(plan, slots)
            dispatch = RowDispatch(tracker, plan, op, adapter.option)
            forward = clone_row_forward(function, dispatch, op, (2048, 1, width))
            row_owners.append((name, op))
            tracker.owners.append(op)
        tracker.plans[name] = dict(role=role, plan_id=ids[name], plan=plan)
        pending.append((module, forward))
    for name, op in row_owners:
        adapter.scope_ops[name] = op
        adapter.actual_routes[name] = 'candidate'
        adapter.placements[name] = op.codec.placement
        adapter.selection_checks[name] = dict(mask_sha256=getattr(op.codec, 'selection_sha256', None),
            wire_bytes=op.codec.wire_bytes_per_rank, bf16_wire_bytes=op.codec.bf16_wire_bytes_per_rank)
    for module, forward in pending:
        module.forward = types.MethodType(forward, module)
    model._mav1_native_linear = tracker
    return tracker
