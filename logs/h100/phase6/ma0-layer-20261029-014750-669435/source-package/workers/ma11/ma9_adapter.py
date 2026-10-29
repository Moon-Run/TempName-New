"""Candidate-only exact optimizer zero-statistic fusion; never install on F4.

The model, gradient norm, clipping, Adam and returned zero statistic are retained.
Only ChainedOptimizer.count_zeros receives an instance-local function namespace.
"""
import ctypes
import hashlib
import inspect
import textwrap
import types
from contextlib import nullcontext

PROTOCOL = 'h100-ma9-exact-zero-count-v1'
CHUNK_SIZE = 32768
# Generated from full indented source lines, including decorators, exactly as
# inspect.getsource returns them. Runtime source drift is an error.
SOURCE_HASHES = {
    'count_zeros_fp32': '1f8f96762e2f6a9c85a56ffb0ae8003eb0db9ddc03c7241a61ccde3fa337c247',
    'count_zeros': 'ebb56da4cfd6945b7884b2082f48c476fbe60cdf5ec8a7374de16177ba6c1a2a',
    'step': '717b7495c1f181f7a83e651c5a1d992997ba355a71937483d7de42d3db2866e5',
}


def normalized(source):
    return textwrap.dedent(source).strip() + '\n'


def require_source(function, name):
    actual = hashlib.sha256(normalized(inspect.getsource(function)).encode()).hexdigest()
    if actual != SOURCE_HASHES[name]:
        raise RuntimeError(f'MA9 requires original {name}: expected {SOURCE_HASHES[name]}, got {actual}')


def select_gradients(parameters, helpers, torch):
    """Use the pinned reference's filters and parameter order, including TP replicas."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    selected = []
    data_parallel_group = None
    for param in parameters:
        present = hasattr(param, 'grad') and param.grad is not None
        shared = helpers.param_is_not_shared(param)
        duplicate = helpers.param_is_not_tensor_parallel_duplicate(param)
        if present and shared and duplicate:
            grad_obj = param.grad
            data_parallel_group = helpers.get_data_parallel_group_if_dtensor(grad_obj, data_parallel_group)
            grad = helpers.to_local_if_dtensor(grad_obj).detach()
            if (grad.dtype != torch.float32 or not grad.is_cuda or not grad.is_contiguous()
                    or grad.device != torch.device('cuda', torch.cuda.current_device())):
                raise RuntimeError('MA9 supports only original contiguous local CUDA FP32 gradients')
            selected.append(grad)
    if data_parallel_group is not None:
        raise RuntimeError('MA9 frozen experiment excludes DTensor/FSDP')
    return selected


class NativeCounter:
    def __init__(self, torch, library):
        self.torch, self.library = torch, library
        for name, expected in [('ma9_zero_count_protocol', 9), ('ma9_zero_count_chunk_size', CHUNK_SIZE)]:
            fn = getattr(library, name)
            fn.argtypes, fn.restype = [], ctypes.c_int
            if fn() != expected:
                raise RuntimeError('Wrong MA9 native identity: ' + name)
        self.launch = library.ma9_zero_count_cuda
        self.launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.launch.restype = ctypes.c_int
        self.signature = None
        self.initializations = 0
        self.calls = 0

    def local(self, gradients):
        torch = self.torch
        signature = tuple((g.data_ptr(), g.numel(), str(g.dtype), str(g.device), g.is_contiguous())
                          for g in gradients)
        device = torch.device('cuda', torch.cuda.current_device())
        if any(g.dtype != torch.float32 or g.device != device or not g.is_contiguous() for g in gradients):
            raise RuntimeError('Invalid native gradient layout/dtype/device')
        stream = torch.cuda.current_stream(device)
        if self.signature is None:
            rows, chunks = [], 0
            for grad in gradients:
                rows.append([grad.data_ptr(), grad.numel(), chunks])
                chunks += (grad.numel() + CHUNK_SIZE - 1) // CHUNK_SIZE
            if len(rows) >= 2**31 or chunks >= 2**31:
                raise RuntimeError('MA9 descriptor count exceeds ABI')
            self.descriptors = torch.tensor(rows, dtype=torch.int64, device=device).reshape(-1, 3)
            self.partials = torch.empty(chunks, dtype=torch.int64, device=device)
            self.counts = torch.empty(len(rows), dtype=torch.int64, device=device)
            self.total = torch.empty(1, dtype=torch.float32, device=device)
            self.signature, self.chunks = signature, chunks
            self.stream_id = stream.cuda_stream
            self.initializations += 1
        if signature != self.signature or stream.cuda_stream != self.stream_id:
            raise RuntimeError('MA9 gradient storage/order or current stream changed; refusing stale pointers')
        status = self.launch(self.descriptors.data_ptr(), len(gradients), self.chunks,
                             self.partials.data_ptr(), self.counts.data_ptr(), self.total.data_ptr(),
                             stream.cuda_stream)
        if status:
            raise RuntimeError(f'MA9 zero-count CUDA launch failed: {status}')
        self.calls += 1
        return self.total


def native_self_test(torch, library):
    """Runs on GPU during smoke only, before training; no synthetic performance claim."""
    values = torch.tensor([0., -0., 1., -1., float('nan'), float('inf'), -float('inf')], device='cuda')
    subnormal = torch.tensor([1, -2147483647], dtype=torch.int32, device='cuda').view(torch.float32)
    large = torch.zeros(2**24 + 1, device='cuda')
    cases = [[], [torch.empty(0, device='cuda')], [values, subnormal],
             [torch.zeros(CHUNK_SIZE + 1, device='cuda'), torch.ones(17, device='cuda')],
             [large, torch.zeros(1, device='cuda'), torch.zeros(1, device='cuda'), torch.zeros(1, device='cuda')],
             [torch.empty(0, device='cuda'), values, torch.empty(0, device='cuda')]]
    for gradients in cases:
        native = NativeCounter(torch, library)
        actual = native.local(gradients)
        expected = torch.tensor([0.], device='cuda')
        counts = []
        for grad in gradients:
            count = grad.numel() - torch.count_nonzero(grad)
            counts.append(count)
            expected = count + expected
        if not torch.equal(actual, expected) or (counts and not torch.equal(native.counts, torch.stack(counts))):
            raise RuntimeError('MA9 native exact integer/ordered FP32 self-test failed')
        # Every launch must overwrite its scratch and output, including empty input.
        if not torch.equal(native.local(gradients), expected):
            raise RuntimeError('MA9 scratch reuse failed')
    return dict(passed=True, cases=len(cases), ordered_above_2pow24=True,
                nan_inf_subnormal_signed_zero=True)


class ZeroCountTracker:
    def __init__(self, torch, helpers, library, mode):
        if mode not in ('smoke', 'timing', 'profile'):
            raise RuntimeError('Invalid MA9 worker mode')
        self.torch, self.helpers, self.mode = torch, helpers, mode
        self.native = NativeCounter(torch, library)
        self.failed = False
        self.calls = self.audits = 0
        self.self_test = native_self_test(torch, library) if mode == 'smoke' else None

    def count(self, parameters, grad_stats_parallel_group, use_decoupled_grad=False):
        if self.failed or use_decoupled_grad:
            raise RuntimeError('MA9 failed or unsupported precision-aware optimizer')
        torch = self.torch
        scope = (torch.autograd.profiler.record_function('ma9.optimizer.zero_count')
                 if self.mode == 'profile' else nullcontext())
        try:
            with scope:
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                else:
                    parameters = list(parameters)
                gradients = select_gradients(parameters, self.helpers, torch)
                total = self.native.local(gradients)
                torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM,
                                             group=grad_stats_parallel_group)
                value = total.item()
                if self.mode == 'smoke':
                    counts = (torch.stack([g.numel() - torch.count_nonzero(g) for g in gradients])
                              if gradients else torch.empty(0, dtype=torch.int64, device=total.device))
                    if not torch.equal(counts, self.native.counts):
                        raise RuntimeError('MA9 per-parameter zero counts differ')
                    reference = self.helpers.count_zeros_fp32(parameters, grad_stats_parallel_group,
                                                            use_decoupled_grad=False)
                    if value != reference:
                        raise RuntimeError('MA9 original reduced statistic differs')
                    self.audits += 1
                self.calls += 1
                return value
        except BaseException:
            self.failed = True
            raise

    def state(self):
        if self.failed:
            raise RuntimeError('MA9 evidence requested after failure')
        return dict(protocol=PROTOCOL, installed=True, calls=self.calls, audits=self.audits,
                    initializations=self.native.initializations, kernel_calls=2*self.native.calls,
                    selected_tensors=len(self.native.signature or ()),
                    gradient_elements=sum(x[1] for x in (self.native.signature or ())),
                    chunks=getattr(self.native, 'chunks', 0), failed=False, self_test=self.self_test)


def install_zero_count(optimizer, library, mode):
    import torch
    from megatron.core.optimizer import clip_grads, optimizer as optimizer_module
    cls = optimizer_module.ChainedOptimizer
    if type(optimizer) is not cls or 'count_zeros' in optimizer.__dict__:
        raise RuntimeError('MA9 requires untouched ChainedOptimizer instance')
    if (not optimizer.grads_states_parallel_group_is_shared() or
            optimizer.config.use_precision_aware_optimizer):
        raise RuntimeError('MA9 requires the original shared TP8 statistics group and FP32 optimizer')
    require_source(clip_grads.count_zeros_fp32, 'count_zeros_fp32')
    require_source(cls.count_zeros, 'count_zeros')
    require_source(cls.step, 'step')
    original = inspect.unwrap(cls.count_zeros)
    if original.__globals__['count_zeros_fp32'] is not clip_grads.count_zeros_fp32:
        raise RuntimeError('MA9 refuses a previously modified statistics function')
    tracker = ZeroCountTracker(torch, clip_grads, library, mode)
    namespace = dict(original.__globals__)
    namespace['count_zeros_fp32'] = tracker.count
    method = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    method.__kwdefaults__ = original.__kwdefaults__
    optimizer.count_zeros = types.MethodType(torch.no_grad()(method), optimizer)
    return tracker
