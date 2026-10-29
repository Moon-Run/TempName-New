"""MA8: retain the forward sequence gather for QKV/FC1 weight gradients.

The pinned Megatron methods supply all arithmetic. The candidate changes only
the ownership of the forward gather output and its reuse in backward. No global
Megatron symbol is replaced; output-layer and row-parallel paths stay original.
"""
import hashlib
import inspect
import textwrap
import types

PROTOCOL = 'h100-ma8-f4-activation-cache-v2'
ARMS = ('F4', 'F4-cache')
CLASS_SHA256 = '0f1975a4582b1e31ce8d5a5a9189f820cfb8a35abb91a025e8c330a198258bea'
# Pin normalized inspect.getsource(), retaining the method's first-line indent
# before dedenting. ast.get_source_segment() drops only that first-line indent.
COLUMN_FORWARD_SHA256 = '9ee66b4f84be3e67664610ec0311e26e8a8b2f463a0088c6a242178ba0f5bd82'
LINEAR_WRAPPER_SHA256 = '000d20605d4be27b7dd5fe1361e3116bfa5df2d4e8f4976b7729967c5c42433b'
SUFFIX_ROLES = {'.self_attention.linear_qkv': 'qkv', '.mlp.linear_fc1': 'fc1'}


def normalized(source):
    return textwrap.dedent(source).strip() + '\n'


def require_source(source, expected):
    source = normalized(source)
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError('MA8 requires the pinned, unmodified Megatron source: '
                           f'expected SHA256 {expected}, got {actual}')
    return source


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('MA8 source anchor changed: ' + old[:100])
    return source.replace(old, new)


def cached_class_source(source):
    """Produce a bounded source change; original backward math remains literal."""
    source = require_source(source, CLASS_SHA256)
    source = replace_once(source,
        'class LinearWithGradAccumulationAndAsyncCommunication(', 'class _MA8CachedRawLinear(')
    source = replace_once(source, '        ctx.save_for_backward(input, weight)\n', '')
    source = replace_once(source,
        '            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")',
        '            all_gather_buffer = torch.empty(dim_size, dtype=input.dtype, device=input.device)')
    source = replace_once(source, '        output = torch.matmul(total_input, weight.t())',
        '        ctx.save_for_backward(input, weight, total_input)\n'
        '        output = torch.matmul(total_input, weight.t())')
    source = replace_once(source, '        input, weight = ctx.saved_tensors',
        '        input, weight, retained_input = ctx.saved_tensors')
    start = source.index('        if wgrad_compute:\n            if ctx.sequence_parallel:')
    end = source.index('        grad_input = grad_output.matmul(weight)', start)
    source = source[:start] + '        if wgrad_compute:\n            total_input = retained_input\n' + source[end:]
    source = replace_once(source,
        '        if ctx.sequence_parallel and wgrad_compute:\n            handle.wait()\n\n', '')
    return source


def _clone_function(function, replacements):
    """Clone globals once at setup; execute the original code object verbatim."""
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    clone = types.FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = dict(function.__annotations__)
    clone.__dict__.update(function.__dict__)
    clone.__qualname__ = function.__qualname__
    clone.__module__ = function.__module__
    # The original linear wrapper reads/writes its own .warned attribute.
    # Bind that name to the clone so even warning bookkeeping is isolated.
    namespace[function.__name__] = clone
    return clone


def _validated_setup(model, arm):
    import torch
    from megatron.core.tensor_parallel import layers

    if arm not in ARMS:
        raise RuntimeError('MA8 arm is invalid')
    config = model.config
    if (not model.training or config.num_layers != 32 or config.hidden_size != 4096 or
            config.tensor_model_parallel_size != 8 or config.pipeline_model_parallel_size != 1 or
            not config.sequence_parallel or config.gradient_accumulation_fusion or
            config.recompute_granularity is not None or config.enable_cuda_graph or
            config.external_cuda_graph or getattr(config, 'cpu_offloading', False) or
            config.defer_embedding_wgrad_compute):
        raise RuntimeError('MA8 requires frozen TP8/PP1 BF16 training without recompute or offload')
    reference = layers.LinearWithGradAccumulationAndAsyncCommunication
    source = require_source(inspect.getsource(reference), CLASS_SHA256)
    original_forward = layers.ColumnParallelLinear.forward
    require_source(inspect.getsource(original_forward), COLUMN_FORWARD_SHA256)
    original_wrapper = layers.linear_with_grad_accumulation_and_async_allreduce
    require_source(inspect.getsource(original_wrapper), LINEAR_WRAPPER_SHA256)
    selected, roles = [], {}
    for name, module in model.named_modules():
        matches = [role for suffix, role in SUFFIX_ROLES.items() if name.endswith(suffix)]
        if not matches:
            continue
        role = matches[0]
        if (type(module) is not layers.ColumnParallelLinear or not module.training or
                not module.sequence_parallel or module.allreduce_dgrad or
                module.gradient_accumulation_fusion or module.explicit_expert_comm or
                module.disable_grad_reduce or module.gather_output or
                module.weight is None or not module.weight.requires_grad or
                module.weight.dtype != torch.bfloat16 or not module.weight.is_cuda or
                not module.weight.is_contiguous() or
                tuple(module.weight.shape) != (1536 if role == 'qkv' else 2048, 4096) or
                getattr(module.forward, '__func__', None) is not original_forward):
            raise RuntimeError('MA8 column module is incompatible: ' + name)
        roles[name] = role
        selected.append(module)
    if len(roles) != 64 or list(roles.values()).count('qkv') != 32 or list(roles.values()).count('fc1') != 32:
        raise RuntimeError('MA8 must cover all 32 QKV and 32 FC1 modules')
    metadata = dict(protocol=PROTOCOL, arm=arm, installed=False,
        function='LinearWithGradAccumulationAndAsyncCommunication', module_roles=roles,
        source_sha256=dict(class_source=CLASS_SHA256, column_forward=COLUMN_FORWARD_SHA256,
                           linear_wrapper=LINEAR_WRAPPER_SHA256))
    return metadata, selected, layers, source, original_forward, original_wrapper


def inspect_activation_setup(model, arm='F4'):
    """Validate once and return JSON metadata; do not mutate any model/function."""
    return _validated_setup(model, arm)[0]


def install_activation_cache(model, arm='F4-cache'):
    """Install the raw candidate only; no added work runs on each invocation.

    F4 is never installed or wrapped. Both cloned Python functions retain their
    original code objects. Only their global references route QKV/FC1 to the
    retained-input Function; the global Megatron functions and LM head remain
    unchanged. All validation is performed here before any instance mutation.
    """
    if arm != 'F4-cache':
        raise RuntimeError('MA8 installation is candidate-only; inspect F4 without modifying it')
    if hasattr(model, '_ma8_activation_setup'):
        raise RuntimeError('MA8 adapter was installed twice')
    metadata, selected, layers, source, original_forward, original_wrapper = _validated_setup(model, arm)
    namespace = dict(vars(layers))
    exec(compile(cached_class_source(source), '<ma8-cached-pinned-megatron>', 'exec'), namespace)
    cached = namespace['_MA8CachedRawLinear']
    wrapper = _clone_function(original_wrapper,
        {'LinearWithGradAccumulationAndAsyncCommunication': cached})
    forward = _clone_function(original_forward,
        {'linear_with_grad_accumulation_and_async_allreduce': wrapper})
    for module in selected:
        module.forward = types.MethodType(forward, module)
    metadata.update(installed=True, function=cached.__name__)
    model._ma8_activation_setup = metadata
    return metadata
