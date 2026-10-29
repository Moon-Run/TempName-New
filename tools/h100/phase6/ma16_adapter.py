"""MA16: hide column forward AllGather behind its already-local row GEMM.

The complete rank-ordered gather is retained exactly as in MA8. Only the
forward dependency graph changes; the pinned MA8 backward is copied verbatim.
There is one collective and no auxiliary compute stream, gradient queue or
packed remote-input copy. Smoke rejects any cuBLAS M-split arithmetic drift.
"""
import ast
import hashlib
import inspect
import types
from contextlib import nullcontext

import ma8_adapter as cache

PROTOCOL = 'h100-ma16-local-first-v1'
LOCAL_SHAPE = (256, 1, 4096)
WORLD_SIZE = 8
OUTPUT_WIDTHS = {'qkv': 1536, 'fc1': 2048}
CLASS_SHA256 = '932e3c1b3d29654ad4b43d5d5294deb36984cde2658a7c1b41ccdc62fcdd180a'
BACKWARD_SHA256 = '73a5045f05a4ead1b9168687d911dd3d4704ee995c2d72bd3a6756b835f6dcd0'


def require(condition, message):
    if not condition:
        raise RuntimeError('MA16 ' + message)


def local_first_class_source(source):
    """Replace only the MA8 forward gather/GEMM; preserve its backward AST."""
    source = cache.cached_class_source(source)
    source = cache.replace_once(source, 'class _MA8CachedRawLinear(',
                                'class _MA16LocalFirstLinear(')
    begin = source.index('        if sequence_parallel:\n')
    end = source.index('        if bias is not None:\n', begin)
    source = source[:begin] + (
        "        if not sequence_parallel:\n"
        "            raise RuntimeError('MA16 requires sequence parallel column forward')\n"
        "        total_input, output = _ma16_projection(input, weight)\n"
        "        ctx.save_for_backward(input, weight, total_input)\n"
    ) + source[end:]
    return source


def backward_sha256(source):
    tree = ast.parse(source)
    method = next(node for node in tree.body[0].body
                  if isinstance(node, ast.FunctionDef) and node.name == 'backward')
    return hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()


def local_first_matmul(torch, input, weight, gather, group, rank, world_size,
                       marker=lambda stage: nullcontext()):
    """Algorithm shared by CUDA execution and small CPU dependency tests.

    ``wait`` is intentionally after local GEMM submission. A CUDA NCCL Work
    wait enqueues the gather dependency on the current compute stream. Remote
    rows are never read before it, including the rank's own gathered slice.
    """
    require(0 <= rank < world_size, 'invalid tensor parallel rank')
    rows = input.shape[0]
    total_shape = (rows * world_size, *input.shape[1:])
    total_input = torch.empty(total_shape, dtype=input.dtype, device=input.device)
    output = torch.empty((*total_shape[:-1], weight.shape[0]),
                         dtype=input.dtype, device=input.device)
    start, stop = rank * rows, (rank + 1) * rows
    transposed_weight = weight.t()
    with marker('gather'):
        handle = gather(total_input, input, group=group, async_op=True)
    with marker('local'):
        torch.matmul(input, transposed_weight, out=output[start:stop])
    with marker('wait'):
        handle.wait()
    if start:
        with marker('prefix'):
            torch.matmul(total_input[:start], transposed_weight, out=output[:start])
    if stop < total_shape[0]:
        with marker('suffix'):
            torch.matmul(total_input[stop:], transposed_weight, out=output[stop:])
    return total_input, output


class LocalFirstProjection:
    def __init__(self, torch, name, role, rank, group, gather, mode):
        self.torch, self.name, self.role = torch, name, role
        self.rank, self.group, self.gather, self.mode = rank, group, gather, mode
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.current_stream().cuda_stream
        self.calls = self.forward_audits = self.local_input_audits = 0
        self.failed = self.active = False

    def marker(self, stage):
        return (self.torch.autograd.profiler.record_function(
            ('ma16.column.forward.' + self.role + '.rank' + str(self.rank)
             if stage == 'forward' else 'ma16.column.' + stage))
            if self.mode == 'profile' else nullcontext())

    def __call__(self, input, weight):
        torch = self.torch
        try:
            require(not self.failed and not self.active, 'failed or reentrant column invocation')
            self.active = True
            require(input.is_cuda and weight.is_cuda and input.device == weight.device and
                    input.device.index == self.device == torch.cuda.current_device() and
                    torch.cuda.current_stream().cuda_stream == self.stream,
                    'column device or compute stream changed')
            require(not torch.cuda.is_current_stream_capturing(), 'CUDA Graph is unsupported')
            require(tuple(input.shape) == LOCAL_SHAPE and input.is_contiguous() and
                    input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and
                    weight.is_contiguous() and tuple(weight.shape) ==
                    (OUTPUT_WIDTHS[self.role], LOCAL_SHAPE[-1]),
                    'requires pinned contiguous BF16 column input and weight')
            with self.marker('forward'):
                total_input, output = local_first_matmul(
                    torch, input, weight, self.gather, self.group,
                    self.rank, WORLD_SIZE, self.marker)
            if self.mode == 'smoke':
                # Verify both the rank placement and changed M-shape arithmetic.
                # This synchronization and full GEMM never run in timing/profile.
                local = total_input[self.rank * 256:(self.rank + 1) * 256]
                require(torch.equal(local.view(torch.uint8), input.view(torch.uint8)),
                        'gathered local rows differ from the actual local input')
                reference = torch.matmul(total_input, weight.t())
                require(torch.equal(output.view(torch.uint8), reference.view(torch.uint8)),
                        'split column forward differs bytewise from original full GEMM')
                self.local_input_audits += 1
                self.forward_audits += 1
            self.calls += 1
            return total_input, output
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False

    def state(self):
        require(not self.failed and not self.active, 'cannot report failed/active projection')
        return dict(role=self.role, rank=self.rank, calls=self.calls,
                    forward_audits=self.forward_audits,
                    local_input_audits=self.local_input_audits,
                    gemms_per_call=2 if self.rank in (0, 7) else 3,
                    device=self.device, stream=self.stream,
                    failed=self.failed, active=self.active)


class Tracker:
    def __init__(self, projections, source, original_backward):
        self.projections = projections
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.original_backward = original_backward

    def state(self):
        return dict(protocol=PROTOCOL, installed=True, local_shape=list(LOCAL_SHAPE),
                    world_size=WORLD_SIZE, source_sha256=self.source_sha256,
                    backward_sha256=self.original_backward,
                    modules={name: projection.state()
                             for name, projection in self.projections.items()})


def validate_state(state, steps, mode, rank=None):
    """Offline validator shared by frozen controller and worker tests."""
    require(mode in ('smoke', 'timing', 'profile') and isinstance(steps, int) and steps > 0,
            'invalid state-validation schedule')
    require(isinstance(state, dict) and state.get('protocol') == PROTOCOL and
            state.get('installed') is True and state.get('local_shape') == list(LOCAL_SHAPE) and
            state.get('world_size') == WORLD_SIZE and state.get('source_sha256') == CLASS_SHA256 and
            state.get('backward_sha256') == BACKWARD_SHA256, 'local-first identity mismatch')
    modules = state.get('modules', {})
    expected = {f'decoder.layers.{index}{suffix}': role for index in range(32)
                for suffix, role in cache.SUFFIX_ROLES.items()}
    require(set(modules) == set(expected), 'local-first module coverage mismatch')
    calls = steps * 4
    audits = calls if mode == 'smoke' else 0
    devices, streams, ranks = set(), set(), set()
    for name, role in expected.items():
        item = modules[name]
        actual_rank = item.get('rank')
        require(isinstance(actual_rank, int) and 0 <= actual_rank < WORLD_SIZE and
                (rank is None or actual_rank == rank) and item.get('role') == role and
                item.get('calls') == calls and item.get('forward_audits') == audits and
                item.get('local_input_audits') == audits and
                item.get('gemms_per_call') == (2 if actual_rank in (0, 7) else 3) and
                item.get('failed') is False and item.get('active') is False and
                isinstance(item.get('device'), int) and 0 <= item['device'] < WORLD_SIZE and
                isinstance(item.get('stream'), int) and item['stream'] >= 0,
                'invalid local-first execution evidence: ' + name)
        devices.add(item['device'])
        streams.add(item['stream'])
        ranks.add(actual_rank)
    require(len(devices) == len(streams) == len(ranks) == 1,
            'local-first modules changed rank/device/compute stream')
    return True


def install_local_first(model, mode):
    """Install after the common MA8 cache, without changing global Megatron."""
    import torch
    from megatron.core.tensor_parallel import layers
    require(mode in ('smoke', 'timing', 'profile'), 'invalid execution mode')
    require(not hasattr(model, '_ma16_local_first'), 'duplicate installation')
    require(torch.__version__.split('+')[0] == '2.6.0', 'requires pinned PyTorch 2.6.0')
    common = getattr(model, '_ma8_activation_setup', None)
    require(isinstance(common, dict) and common.get('protocol') == cache.PROTOCOL and
            common.get('installed') is True and common.get('arm') == 'F4-cache',
            'requires the original installed MA8 activation cache')
    config = model.config
    require(model.training and config.num_layers == 32 and config.hidden_size == 4096 and
            config.tensor_model_parallel_size == 8 and config.pipeline_model_parallel_size == 1 and
            config.context_parallel_size == 1 and config.sequence_parallel and
            not config.gradient_accumulation_fusion and config.recompute_granularity is None and
            not config.enable_cuda_graph and not config.external_cuda_graph and
            not getattr(config, 'cpu_offloading', False) and not config.defer_embedding_wgrad_compute,
            'requires frozen TP8/PP1/CP1 training and original activation-cache configuration')
    source = cache.require_source(inspect.getsource(
        layers.LinearWithGradAccumulationAndAsyncCommunication), cache.CLASS_SHA256)
    cache.require_source(inspect.getsource(layers.ColumnParallelLinear.forward),
                         cache.COLUMN_FORWARD_SHA256)
    cache.require_source(inspect.getsource(layers.linear_with_grad_accumulation_and_async_allreduce),
                         cache.LINEAR_WRAPPER_SHA256)
    candidate_source = local_first_class_source(source)
    original_backward = backward_sha256(cache.cached_class_source(source))
    require(backward_sha256(candidate_source) == original_backward == BACKWARD_SHA256 and
            hashlib.sha256(candidate_source.encode()).hexdigest() == CLASS_SHA256,
            'MA8 backward or registered candidate source changed')
    rank = layers.get_tensor_model_parallel_rank()
    group = layers.get_tensor_model_parallel_group()
    require(layers.get_tensor_model_parallel_world_size() == WORLD_SIZE and 0 <= rank < WORLD_SIZE,
            'requires the original eight-rank tensor parallel group')
    role_map = common.get('module_roles', {})
    require(len(role_map) == 64 and list(role_map.values()).count('qkv') == 32 and
            list(role_map.values()).count('fc1') == 32, 'requires all 64 MA8 column modules')
    modules = dict(model.named_modules())
    projections, prepared = {}, []
    code = compile(candidate_source, '<ma16-local-first-pinned-megatron>', 'exec')
    for name, role in role_map.items():
        module = modules[name]
        current = getattr(module.forward, '__func__', None)
        require(type(module) is layers.ColumnParallelLinear and current is not None and
                current.__code__ is layers.ColumnParallelLinear.forward.__code__,
                'cached column wrapper changed: ' + name)
        wrapper = current.__globals__.get('linear_with_grad_accumulation_and_async_allreduce')
        require(wrapper is not None and wrapper.__code__ is
                layers.linear_with_grad_accumulation_and_async_allreduce.__code__,
                'cached linear wrapper changed: ' + name)
        cached = wrapper.__globals__.get('LinearWithGradAccumulationAndAsyncCommunication')
        require(cached is not None and cached.__name__ == '_MA8CachedRawLinear' and
                module.training and module.sequence_parallel and not module.allreduce_dgrad and
                not module.gradient_accumulation_fusion and not module.explicit_expert_comm and
                not module.disable_grad_reduce and not module.gather_output and
                module.weight is not None and module.weight.requires_grad and
                module.weight.dtype == torch.bfloat16 and module.weight.is_cuda and
                module.weight.is_contiguous() and tuple(module.weight.shape) ==
                (OUTPUT_WIDTHS[role], LOCAL_SHAPE[-1]), 'incompatible cached column: ' + name)
        projection = LocalFirstProjection(torch, name, role, rank, group,
                                          layers.dist_all_gather_func, mode)
        namespace = dict(vars(layers), _ma16_projection=projection)
        exec(code, namespace)
        new_wrapper = cache._clone_function(layers.linear_with_grad_accumulation_and_async_allreduce,
            {'LinearWithGradAccumulationAndAsyncCommunication': namespace['_MA16LocalFirstLinear']})
        forward = cache._clone_function(layers.ColumnParallelLinear.forward,
            {'linear_with_grad_accumulation_and_async_allreduce': new_wrapper})
        projections[name] = projection
        prepared.append((module, forward))
    # No instance is changed until every source, module and role has passed.
    for module, forward in prepared:
        module.forward = types.MethodType(forward, module)
    tracker = Tracker(projections, candidate_source, original_backward)
    model._ma16_local_first = tracker
    return tracker
