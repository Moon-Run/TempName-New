"""MA15: replay the unchanged, communication-free MA11 attention DAG.

Each of the 32 cores owns one forward/backward graph pair and a private pool.
There is no graph around Flux, collectives, DDP, or the optimizer. Copies,
clones, stream checks and graph replay remain inside the measured path.
"""
import os
import types
from contextlib import nullcontext

import ma11_adapter as reference

PROTOCOL = 'h100-ma15-local-attention-graph-v1'
CORE_SHA256 = '4e0d8275ba546bbc8289054d6d8cf56e6253f990e62c7fba38f3b02a2e77cccf'
SUFFIX = '.self_attention.core_attention'
QKV_SHAPE = (2048, 1, 4, 128)
MASK_SHAPE = (1, 1, 2048, 2048)
OUTPUT_SHAPE = (2048, 1, 512)
MODULES = [f'decoder.layers.{i}{SUFFIX}' for i in range(32)]
PROFILER_ENVIRONMENT = {'TEARDOWN_CUPTI': '0', 'DISABLE_CUPTI_LAZY_REINIT': '1'}


def require(condition, message):
    if not condition:
        raise RuntimeError('MA15 ' + message)


def layout(tensor):
    # Allocation addresses can change. Layout and pointer alignment cannot:
    # changing either can change the cuBLAS dispatch used by the MA11 oracle.
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
            str(tensor.device), tensor.data_ptr() % 256)


def prime_profiler(torch, mode):
    """Attach CUPTI before any graph exists, only in the profile subprocess.

    Late attachment can report graph memcpy/memset nodes as kernels. Keep
    CUPTI attached between this setup session and the original measured
    profile session, using PyTorch 2.6's CUDA Graph teardown workaround.
    https://github.com/pytorch/pytorch/blob/v2.6.0/torch/profiler/profiler.py
    """
    require(mode in ('smoke', 'timing', 'profile'), 'invalid profiler setup mode')
    if mode != 'profile':
        return {'primed': False}
    activities = (torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
    require(activities[1] in torch.profiler.supported_activities(),
            'requires CUPTI GPU profiling before graph capture')
    os.environ.update(PROFILER_ENVIRONMENT)
    probe = torch.empty(1, dtype=torch.int64, device='cuda')
    with torch.profiler.profile(activities=activities):
        probe.zero_()
        torch.cuda.synchronize()
    return dict(primed=True, sessions=1, before_capture=True,
                environment=dict(PROFILER_ENVIRONMENT))


class InvocationGuard:
    """One live forward per core; errors permanently invalidate the instance."""

    def __init__(self):
        self.phase = 'idle'
        self.device = self.stream = self.signature = None
        self.forward_replays = self.backward_replays = 0

    def fail(self):
        self.phase = 'failed'

    def begin(self, device, stream, signature):
        try:
            require(self.phase == 'idle', 'requires one forward followed by one backward')
            if self.signature is None:
                self.device, self.stream, self.signature = device, stream, signature
            require((device, stream, signature) ==
                    (self.device, self.stream, self.signature),
                    'input device, stream or layout changed')
            self.phase = 'live'
        except Exception:
            self.fail()
            raise

    def check_backward(self, device, stream):
        try:
            require(self.phase == 'live', 'backward has no unique live forward')
            require((device, stream) == (self.device, self.stream),
                    'backward device or stream changed')
        except Exception:
            self.fail()
            raise

    def finish_forward(self):
        require(self.phase == 'live', 'invalid forward completion')
        self.forward_replays += 1

    def finish_backward(self):
        require(self.phase == 'live', 'invalid backward completion')
        self.backward_replays += 1
        self.phase = 'idle'


class AttentionGraph:
    def __init__(self, torch, name, core, native, mode):
        self.torch, self.name, self.core, self.native, self.mode = torch, name, core, native, mode
        self.eager = core.forward
        self.guard = InvocationGuard()
        self.initializations = self.forward_audits = self.backward_audits = 0
        self.rng_unchanged = False
        self.forward_graph = self.backward_graph = None
        self.input_layouts = None
        self.capture_stream = None

    def marker(self, direction):
        return (self.torch.autograd.profiler.record_function('ma15.attention.graph.' + direction)
                if self.mode == 'profile' else nullcontext())

    def validate_inputs(self, values, mask):
        torch = self.torch
        require(not torch.cuda.is_current_stream_capturing(), 'cannot nest graph capture')
        require(len(values) == 3 and all(tuple(x.shape) == QKV_SHAPE and
                x.dtype == torch.bfloat16 and x.is_cuda and x.requires_grad for x in values),
                'requires the original BF16 Q/K/V training tensors')
        device = values[0].device
        require(device.index == torch.cuda.current_device() and
                all(x.device == device for x in values), 'Q/K/V device changed')
        require(mask is not None and tuple(mask.shape) == MASK_SHAPE and
                mask.dtype == torch.bool and mask.device == device and mask.is_contiguous(),
                'requires the original contiguous boolean attention mask')
        # The pinned non-GQA QKV split is strided, not contiguous. Do not silently
        # make it contiguous and thereby benchmark different GEMM dispatches.
        require(all(tuple(x.stride()) == (1536, 1536, 384, 1) and
                    x.data_ptr() % 256 == 0 for x in values),
                'requires the pinned QKV split strides and alignment')
        return device, torch.cuda.current_stream(device).cuda_stream

    def initialize(self, values, mask):
        torch = self.torch
        require(self.initializations == 0 and self.forward_graph is None,
                'graph initialization cannot be retried')
        device = values[0].device
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        self.input_layouts = [dict(shape=list(x.shape), strides=list(x.stride()),
                                  dtype=str(x.dtype)) for x in values]
        self.static_inputs = tuple(torch.empty_strided(x.shape, x.stride(),
            device=device, dtype=x.dtype).copy_(x.detach()).requires_grad_(True) for x in values)
        self.static_mask = torch.empty_like(mask).copy_(mask)
        softmax = self.core.scale_mask_softmax
        original_softmax = softmax.forward
        quiet_function = reference.make_function(torch, self.native, 'timing', {})

        def quiet_forward(_self, scores, actual_mask):
            return quiet_function.apply(scores, actual_mask)

        self.capture_stream = torch.cuda.Stream(device=device)
        caller = torch.cuda.current_stream(device)
        self.capture_stream.wait_stream(caller)
        try:
            # The original smoke softmax performs equality checks that synchronize;
            # profile annotations are also CPU-only. Neither belongs in capture.
            # All numerical kernels still come from the original MA11 function.
            softmax.forward = types.MethodType(quiet_forward, softmax)
            with torch.cuda.stream(self.capture_stream), torch.enable_grad():
                for _ in range(3):
                    output = self.eager(*self.static_inputs, self.static_mask)
                    gradient = torch.ones_like(output)
                    torch.autograd.grad(output, self.static_inputs, gradient)
                self.static_gradient = torch.empty_like(output).zero_()
                del output, gradient
            caller.wait_stream(self.capture_stream)
            torch.cuda.synchronize(device)
            self.forward_graph, self.backward_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.enable_grad():
                with torch.cuda.graph(self.forward_graph, stream=self.capture_stream):
                    self.static_output = self.eager(*self.static_inputs, self.static_mask)
                # The pool is shared only within this pair. Replay order is always
                # forward then backward, enforced before any storage is overwritten.
                with torch.cuda.graph(self.backward_graph, pool=self.forward_graph.pool(),
                                      stream=self.capture_stream):
                    self.static_derivatives = torch.autograd.grad(
                        self.static_output, self.static_inputs, self.static_gradient)
            caller.wait_stream(self.capture_stream)
            require(torch.equal(cpu_rng, torch.get_rng_state()) and
                    torch.equal(cuda_rng, torch.cuda.get_rng_state(device)),
                    'local graph initialization changed RNG state')
            self.rng_unchanged = True
            self.initializations = 1
        finally:
            softmax.forward = original_softmax

    def forward(self, values, mask):
        try:
            device, stream = self.validate_inputs(values, mask)
            self.guard.begin(device.index, stream, tuple(layout(x) for x in (*values, mask)))
            if self.initializations == 0:
                self.initialize(values, mask)
            with self.marker('forward'):
                for target, value in zip(self.static_inputs, values):
                    target.copy_(value)
                self.static_mask.copy_(mask)
                self.forward_graph.replay()
                # A caller may retain the output after the next replay. Return
                # owned storage rather than exposing the reusable graph surface.
                output = self.static_output.detach().clone()
            self.guard.finish_forward()
            return output
        except Exception:
            self.guard.fail()
            raise

    def backward(self, gradient):
        torch = self.torch
        try:
            require(not torch.is_grad_enabled(), 'higher-order gradients are unsupported')
            require(gradient is not None and tuple(gradient.shape) == OUTPUT_SHAPE and
                    gradient.dtype == torch.bfloat16 and gradient.is_cuda and
                    gradient.device.index == torch.cuda.current_device(),
                    'invalid attention output gradient')
            self.guard.check_backward(gradient.device.index,
                                      torch.cuda.current_stream(gradient.device).cuda_stream)
            require(not torch.cuda.is_current_stream_capturing(), 'cannot nest graph capture')
            with self.marker('backward'):
                self.static_gradient.copy_(gradient)
                self.backward_graph.replay()
                derivatives = tuple(x.detach().clone() for x in self.static_derivatives)
            # The caller must finish its smoke oracle before this slot becomes idle.
            return derivatives
        except Exception:
            self.guard.fail()
            raise

    def state(self):
        return dict(initializations=self.initializations, phase=self.guard.phase,
                    forward_replays=self.guard.forward_replays,
                    backward_replays=self.guard.backward_replays,
                    forward_audits=self.forward_audits, backward_audits=self.backward_audits,
                    rng_unchanged=self.rng_unchanged, inputs=self.input_layouts,
                    private_graph_pair=True, owned_outputs=True)


def make_function(torch, slot):
    class MA15AttentionGraph(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, mask):
            try:
                values = (query, key, value)
                result = slot.forward(values, mask)
                # Preserve autograd's in-place modification detection, even though
                # the graph uses private copies rather than these tensor addresses.
                ctx.save_for_backward(query, key, value, mask)
                ctx.reference = None
                if slot.mode == 'smoke':
                    with torch.enable_grad():
                        oracle_inputs = tuple(x.detach().requires_grad_(True) for x in values)
                        expected = slot.eager(*oracle_inputs, mask)
                    require(reference.exact_equal(torch, result, expected),
                            'graph forward differs from eager MA11')
                    slot.forward_audits += 1
                    ctx.reference = (oracle_inputs, expected)
                return result
            except Exception:
                slot.guard.fail()
                raise

        @staticmethod
        def backward(ctx, gradient):
            try:
                # Reading saved_tensors checks versions before graph replay.
                _ = ctx.saved_tensors
                derivatives = slot.backward(gradient)
                if ctx.reference is not None:
                    oracle_inputs, expected = ctx.reference
                    with torch.enable_grad():
                        oracle_gradients = torch.autograd.grad(expected, oracle_inputs, gradient)
                    require(all(reference.exact_equal(torch, left, right)
                                for left, right in zip(derivatives, oracle_gradients)),
                            'graph Q/K/V gradients differ from eager MA11')
                    slot.backward_audits += 1
                    ctx.reference = None
                slot.guard.finish_backward()
                return (*derivatives, None)
            except Exception:
                slot.guard.fail()
                raise
    return MA15AttentionGraph


class Tracker:
    def __init__(self, slots, profiler_setup):
        self.slots, self.profiler_setup = slots, profiler_setup

    def state(self):
        return dict(protocol=PROTOCOL, installed=True, original_core_sha256=CORE_SHA256,
                    modules={name: slot.state() for name, slot in self.slots.items()},
                    scope='local_attention_only', communication_captured=False,
                    optimizer_captured=False, profiler_setup=self.profiler_setup)


def install_attention_graphs(model, library, mode):
    import torch
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    require(mode in ('smoke', 'timing', 'profile') and not hasattr(model, '_ma15_graphs'),
            'invalid or duplicate graph installation')
    require(torch.__version__.split('+')[0] == '2.6.0', 'requires pinned PyTorch 2.6.0')
    require(not torch.is_autocast_enabled(), 'autocast is outside the fixed BF16 contract')
    require(hasattr(model, '_ma11_softmax'), 'requires the complete original MA11 installation')
    reference.require_source(DotProductAttention, CORE_SHA256)
    config = model.config
    require(model.training and config.num_layers == 32 and config.hidden_size == 4096 and
            config.tensor_model_parallel_size == 8 and config.pipeline_model_parallel_size == 1 and
            config.context_parallel_size == 1 and config.sequence_parallel and
            config.attention_dropout == 0 and config.recompute_granularity is None and
            not config.enable_cuda_graph and not config.external_cuda_graph,
            'requires the fixed MA11 training configuration')
    selected = [(name, core) for name, core in model.named_modules() if name.endswith(SUFFIX)]
    require(len(selected) == 32 and {name for name, _ in selected} == set(MODULES),
            'must cover exactly the original 32 attention cores')
    for name, core in selected:
        require(type(core) is DotProductAttention and core.training and
                core.num_attention_heads_per_partition == 4 and
                core.num_query_groups_per_partition == 4 and
                core.hidden_size_per_partition == 512 and
                core.attention_dropout.p == 0 and not list(core.parameters()) and
                'forward' not in core.__dict__, 'incompatible original attention core: ' + name)
        require(all(not sub._forward_hooks and not sub._forward_pre_hooks and
                    not sub._backward_hooks and not sub._backward_pre_hooks
                    for sub in core.modules()),
                'Python module hooks cannot be omitted by replay: ' + name)
    native = reference.NativeSoftmax(torch, library)
    profiler_setup = prime_profiler(torch, mode)
    slots = {name: AttentionGraph(torch, name, core, native, mode) for name, core in selected}

    def replacement(slot):
        function = make_function(torch, slot)

        def forward(_self, query, key, value, attention_mask, attn_mask_type=None,
                    attention_bias=None, packed_seq_params=None):
            try:
                require(torch.is_grad_enabled() and _self.training,
                        'requires an ordinary training forward')
                require(not torch.is_autocast_enabled(), 'autocast state changed')
                require(attn_mask_type in (None, AttnMaskType.causal) and
                        attention_bias is None and packed_seq_params is None,
                        'unsupported attention mask, bias or packed sequence')
                return function.apply(query, key, value, attention_mask)
            except Exception:
                slot.guard.fail()
                raise
        return forward

    for name, core in selected:
        core.forward = types.MethodType(replacement(slots[name]), core)
    tracker = Tracker(slots, profiler_setup)
    model._ma15_graphs = tracker
    return tracker


def validate_state(state, steps, mode):
    require(type(steps) is int and steps > 0 and mode in ('smoke', 'timing', 'profile'),
            'invalid graph validation window')
    require(state.get('protocol') == PROTOCOL and state.get('installed') is True and
            state.get('original_core_sha256') == CORE_SHA256 and
            state.get('scope') == 'local_attention_only' and
            state.get('communication_captured') is False and
            state.get('optimizer_captured') is False, 'invalid graph implementation identity')
    expected_profiler = (dict(primed=True, sessions=1, before_capture=True,
                             environment=PROFILER_ENVIRONMENT)
                         if mode == 'profile' else {'primed': False})
    require(state.get('profiler_setup') == expected_profiler,
            'CUPTI must attach before graph capture only in the profile subprocess')
    modules = state.get('modules', {})
    require(set(modules) == set(MODULES), 'graph module coverage changed')
    expected_inputs = [dict(shape=list(QKV_SHAPE), strides=[1536, 1536, 384, 1],
                            dtype='torch.bfloat16')] * 3
    for name, item in modules.items():
        require(item.get('initializations') == 1 and item.get('phase') == 'idle' and
                item.get('forward_replays') == item.get('backward_replays') == steps * 4 and
                item.get('rng_unchanged') is True and item.get('private_graph_pair') is True and
                item.get('owned_outputs') is True and item.get('inputs') == expected_inputs,
                'missing, failed, stale or overlapping graph replay: ' + name)
        expected_audits = steps * 4 if mode == 'smoke' else 0
        require(item.get('forward_audits') == item.get('backward_audits') == expected_audits,
                'eager graph audit missing or leaked into timing: ' + name)
