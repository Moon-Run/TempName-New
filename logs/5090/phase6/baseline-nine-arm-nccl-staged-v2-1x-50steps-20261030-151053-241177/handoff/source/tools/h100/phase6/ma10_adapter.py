"""Candidate-only bounded commits of existing BF16 leaf gradients to main_grad.

The original GEMMs, autograd return values and four microbatch additions remain.
Prepare before DDP construction, bind immediately afterwards. The temporary DDP
factory replacement is restored before any forward executes. No old hook is
removed through private PyTorch APIs and no hook is stacked on top of it.
"""
import bisect
import ctypes
import functools
import hashlib
import inspect
import textwrap
from contextlib import nullcontext

PROTOCOL = 'h100-ma10-bounded-gradient-commit-v1'
CHUNK_SIZE = 32768
TENSOR_LIMIT = 32
BYTE_LIMIT = 64 * 1024 * 1024
PARAMETER_COUNT = 389
SOURCE_HASHES = {
    '_make_backward_post_hook': 'affcf7b965b667c1d2ddb5fc76ba4c1f25bdf6ea770aead14f5e1e4b658d87ed',
    'backward_step': 'c251a17b731002c140b9b94a3ad1d04d67a6635387a75292752a0d0cfd3611ac',
}


def normalized(source):
    return textwrap.dedent(source).strip() + '\n'


def require_source(function, name):
    actual = hashlib.sha256(normalized(inspect.getsource(function)).encode()).hexdigest()
    if actual != SOURCE_HASHES[name]:
        raise RuntimeError(f'MA10 requires original {name}: expected {SOURCE_HASHES[name]}, got {actual}')


class NativeCommit:
    def __init__(self, torch, library):
        self.torch, self.library, self.calls = torch, library, 0
        for name, expected in [('ma10_gradient_batching_protocol', 10),
                               ('ma10_gradient_batching_chunk_size', CHUNK_SIZE),
                               ('ma10_gradient_batching_max_tensors', TENSOR_LIMIT)]:
            fn = getattr(library, name)
            fn.argtypes, fn.restype = [], ctypes.c_int
            if fn() != expected:
                raise RuntimeError('Wrong MA10 native identity: ' + name)
        self.launch = library.ma10_gradient_batching_cuda
        self.launch.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                               ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_void_p]
        self.launch.restype = ctypes.c_int

    def commit(self, targets, sources, stream):
        n = len(targets)
        if not 1 <= n <= TENSOR_LIMIT or len(sources) != n:
            raise RuntimeError('Invalid MA10 batch length')
        pointers = (ctypes.c_uint64 * n)(*(t.data_ptr() for t in targets))
        inputs = (ctypes.c_uint64 * n)(*(s.data_ptr() for s in sources))
        sizes = (ctypes.c_int64 * n)(*(t.numel() for t in targets))
        status = self.launch(pointers, inputs, sizes, n, stream.cuda_stream)
        if status:
            raise RuntimeError(f'MA10 gradient batch CUDA launch failed: {status}')
        self.calls += 1


def _equal_bits(torch, actual, expected):
    # NaN payloads are not part of the finite-training contract; all finite
    # values, infinities and both zero signs are compared as exact FP32 bits.
    equal = actual.view(torch.int32) == expected.view(torch.int32)
    equal = equal | (torch.isnan(actual) & torch.isnan(expected))
    return bool(equal.all().item())


def native_self_test(torch, library):
    """CUDA-only smoke gate, never part of formal timing or profile windows."""
    native = NativeCommit(torch, library)
    stream = torch.cuda.current_stream()
    bf16_bits = torch.tensor([0, -32768, 1, -32767, 16256, -16512,
                              32640, -128, 32704], dtype=torch.int16, device='cuda')
    source = bf16_bits.view(torch.bfloat16)
    target = torch.tensor([-0., 0., 0., 0., 2**24, -(2**24), 1., -1., 0.], device='cuda')
    # Deterministic inputs preserve the candidate's CPU/CUDA RNG identities.
    def values(n, source=False):
        x = (torch.arange(n, device='cuda', dtype=torch.float32).remainder(37) - 18) / 16
        return x.to(torch.bfloat16) if source else x
    cases = [([target], [source]),
             ([values(CHUNK_SIZE + 3)], [values(CHUNK_SIZE + 3, True)]),
             ([values(i + 1) for i in range(TENSOR_LIMIT)],
              [values(i + 1, True) for i in range(TENSOR_LIMIT)])]
    checks = 0
    for targets, sources in cases:
        for _ in range(4):
            expected = [target.clone().add_(source) for target, source in zip(targets, sources)]
            native.commit(targets, sources, stream)
            if not all(_equal_bits(torch, target, ref) for target, ref in zip(targets, expected)):
                raise RuntimeError('MA10 native BF16-to-FP32 accumulation self-test failed')
            checks += 1
    return dict(passed=True, cases=checks, signed_zero_subnormal_nan_inf=True,
                scratch_free=True, repeated_ordered_additions=True)


class GradientBatcher:
    def __init__(self, torch, mode, ddp_class, schedules, original_factory, original_backward):
        if mode not in ('smoke', 'timing', 'profile'):
            raise RuntimeError('Invalid MA10 worker mode')
        self.torch, self.mode, self.ddp_class, self.schedules = torch, mode, ddp_class, schedules
        self.original_factory, self.original_backward = original_factory, original_backward
        self.registrations, self.pending, self.seen = {}, [], set()
        self.bound = self.active = self.failed = False
        self.original_factory_restored = False
        self.microbatches = self.hooks = self.tensor_commits = self.flushes = self.audits = 0
        self.pending_bytes = self.max_pending_bytes = self.max_pending_tensors = 0
        self.largest_source_bytes = 0
        self.self_test = None

    def restore_factory(self):
        if self.ddp_class._make_backward_post_hook is not self.factory:
            self.failed = True
            raise RuntimeError('MA10 temporary DDP factory was replaced by another installer')
        self.ddp_class._make_backward_post_hook = self.original_factory
        self.original_factory_restored = True

    def make_hook(self, owner, param):
        if self.bound or id(param) in self.registrations:
            raise RuntimeError('MA10 duplicate parameter hook or late DDP construction')
        if (owner.ddp_config.overlap_grad_reduce or owner.ddp_config.use_distributed_optimizer
                or owner.ddp_config.overlap_param_gather):
            raise RuntimeError('MA10 requires original non-overlapped, non-distributed optimizer DDP')
        self.registrations[id(param)] = (owner, param)

        def hook(*unused):
            try:
                self.enqueue(owner, param)
            except BaseException:
                self.failed = True
                raise
        return hook

    def bind(self, model, library, expected_parameters=PARAMETER_COUNT):
        """Bind after DDP construction. Production fixes expected_parameters=389."""
        self.restore_factory()
        try:
            if self.bound or self.failed or len(model) != 1 or type(model[0]) is not self.ddp_class:
                raise RuntimeError('MA10 requires one fresh original DDP model')
            owner = model[0]
            config = owner.config
            if (config.gradient_accumulation_fusion or config.sequence_parallel is not True
                    or config.tensor_model_parallel_size != 8
                    or config.pipeline_model_parallel_size != 1
                    or getattr(config, 'context_parallel_size', 1) != 1
                    or getattr(config, 'recompute_granularity', None) is not None):
                raise RuntimeError('Unsupported MA10 model/parallel/recomputation configuration')
            parameters = [p for p in owner.module.parameters() if p.requires_grad]
            if (len(parameters) != expected_parameters or
                    set(map(id, parameters)) != set(self.registrations) or
                    any(item[0] is not owner for item in self.registrations.values())):
                raise RuntimeError('MA10 DDP hook coverage differs from complete parameter universe')
            torch = self.torch
            self.device = torch.device('cuda', torch.cuda.current_device())
            self.stream = torch.cuda.current_stream(self.device)
            self.parameter_slots = {}
            ranges = []
            for p in parameters:
                if p not in owner.param_to_bucket_group:
                    raise RuntimeError('MA10 parameter is not owned by original DDP gradient buffers')
                target = p.main_grad
                if (p.dtype != torch.bfloat16 or p.device != self.device or not p.is_contiguous()
                        or target.dtype != torch.float32 or target.device != self.device
                        or not target.is_contiguous() or target.shape != p.shape or p.numel() == 0
                        or p.grad is not None or p.grad_added_to_main_grad
                        or getattr(p, 'zero_out_wgrad', False)):
                    raise RuntimeError('Unsupported MA10 parameter or main_grad layout/state')
                self.parameter_slots[id(p)] = (target, target.data_ptr(), tuple(p.shape), p.numel())
                ranges.append((target.data_ptr(), target.data_ptr() + target.numel() * 4))
                self.largest_source_bytes = max(self.largest_source_bytes, p.numel() * 2)
            ranges.sort()
            if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
                raise RuntimeError('MA10 main_grad destinations overlap')
            self.target_ranges = ranges
            self.target_starts = [begin for begin, _ in ranges]
            self.native = NativeCommit(torch, library)
            if self.mode == 'smoke':
                self.self_test = native_self_test(torch, library)
            if self.schedules.backward_step is not self.original_backward:
                raise RuntimeError('MA10 backward boundary was modified before installation')

            @functools.wraps(self.original_backward)
            def backward(*args, **kwargs):
                try:
                    return self.run_backward(*args, **kwargs)
                except BaseException:
                    self.failed = True
                    raise

            self.bound = True
            self.schedules.backward_step = backward
            self.backward_wrapper = backward
            return self
        except BaseException:
            self.failed = True
            raise

    def enqueue(self, owner, param):
        torch = self.torch
        if self.failed or not self.bound or not self.active:
            raise RuntimeError('MA10 gradient outside active microbatch or after failure')
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError('MA10 gradient arrived on a different CUDA stream')
        if id(param) in self.seen:
            raise RuntimeError('MA10 duplicate gradient within one microbatch')
        target, pointer, shape, size = self.parameter_slots[id(param)]
        source = param.grad
        if (source is None or not param.requires_grad or param.grad_added_to_main_grad
                or getattr(param, 'zero_out_wgrad', False)
                or source.dtype != torch.bfloat16 or source.device != self.device
                or not source.is_contiguous() or tuple(source.shape) != shape
                or param.main_grad is not target or target.data_ptr() != pointer
                or target.dtype != torch.float32 or target.device != self.device
                or tuple(target.shape) != shape or not target.is_contiguous()
                or param not in owner.param_to_bucket_group):
            raise RuntimeError('MA10 missing/replaced/unsupported gradient; no fallback is permitted')
        count_bytes = size * 2
        begin, end = source.data_ptr(), source.data_ptr() + count_bytes
        after = bisect.bisect_left(self.target_starts, end)
        if after and self.target_ranges[after - 1][1] > begin:
            raise RuntimeError('MA10 gradient source overlaps an accumulation destination')
        if self.pending and (len(self.pending) == TENSOR_LIMIT or
                             self.pending_bytes + count_bytes > BYTE_LIMIT):
            self.flush()
        self.pending.append((target, source))  # Strong ownership survives param.grad=None.
        self.pending_bytes += count_bytes
        self.max_pending_bytes = max(self.max_pending_bytes, self.pending_bytes)
        self.max_pending_tensors = max(self.max_pending_tensors, len(self.pending))
        self.seen.add(id(param))
        self.hooks += 1
        param.grad = None
        if len(self.pending) == TENSOR_LIMIT or self.pending_bytes >= BYTE_LIMIT:
            self.flush()

    def flush(self):
        if self.failed:
            raise RuntimeError('MA10 flush after failure')
        if not self.pending:
            return
        torch = self.torch
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError('MA10 commit stream changed')
        targets, sources = map(list, zip(*self.pending))
        expected = ([target.clone().add_(source) for target, source in zip(targets, sources)]
                    if self.mode == 'smoke' else None)
        scope = (torch.autograd.profiler.record_function(f'ma10.gradient_batch.flush:{len(targets)}')
                 if self.mode == 'profile' else nullcontext())
        with scope:
            self.native.commit(targets, sources, self.stream)
        # Sources were allocated/produced and are consumed on the SAME original
        # autograd CUDA stream. The caching allocator observes stream order; no
        # auxiliary stream or record_stream call is needed before dropping refs.
        if expected is not None:
            if not all(_equal_bits(torch, target, ref) for target, ref in zip(targets, expected)):
                raise RuntimeError('MA10 smoke commit differs from original per-tensor add_')
            self.audits += 1
        self.tensor_commits += len(targets)
        self.flushes += 1
        self.pending.clear()
        self.pending_bytes = 0

    def run_backward(self, *args, **kwargs):
        if self.failed or not self.bound or self.active or self.pending or self.seen:
            raise RuntimeError('MA10 invalid microbatch boundary')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('MA10 does not support CUDA Graph capture')
        scope = (self.torch.autograd.profiler.record_function('ma10.microbatch.backward')
                 if self.mode == 'profile' else nullcontext())
        self.active = True
        with scope:
            value = self.original_backward(*args, **kwargs)
            if self.seen != set(self.parameter_slots):
                raise RuntimeError('MA10 microbatch did not produce every trainable gradient exactly once')
            self.flush()
        self.active = False
        self.seen.clear()
        self.microbatches += 1
        return value

    def state(self):
        if self.failed or not self.bound or self.active or self.pending or self.seen:
            raise RuntimeError('MA10 state requested with failed/uncommitted gradients')
        if self.schedules.backward_step is not self.backward_wrapper:
            raise RuntimeError('MA10 microbatch boundary was replaced after binding')
        return dict(protocol=PROTOCOL, installed=True, bound=True, failed=False,
                    parameter_count=len(self.parameter_slots), microbatches=self.microbatches,
                    hooks=self.hooks, tensor_commits=self.tensor_commits, flushes=self.flushes,
                    kernel_calls=self.native.calls, audits=self.audits,
                    pending_tensors=0, pending_bytes=0, max_pending_tensors=self.max_pending_tensors,
                    max_pending_bytes=self.max_pending_bytes, largest_source_bytes=self.largest_source_bytes,
                    byte_limit=BYTE_LIMIT, tensor_limit=TENSOR_LIMIT, source_dtype='torch.bfloat16',
                    target_dtype='torch.float32', original_factory_restored=self.original_factory_restored,
                    self_test=self.self_test)


def prepare_gradient_batching(mode):
    """Candidate-only call before setup_model_and_optimizer creates any DDP hooks."""
    import torch
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
    from megatron.core.pipeline_parallel import schedules
    from megatron.core import parallel_state
    cls = DistributedDataParallel
    original_factory, original_backward = cls._make_backward_post_hook, schedules.backward_step
    require_source(original_factory, '_make_backward_post_hook')
    require_source(original_backward, 'backward_step')
    if (parallel_state.get_tensor_model_parallel_world_size() != 8 or
            parallel_state.get_pipeline_model_parallel_world_size() != 1 or
            parallel_state.get_data_parallel_world_size() != 1):
        raise RuntimeError('MA10 requires original TP8, PP1, DP1 topology')
    tracker = GradientBatcher(torch, mode, cls, schedules, original_factory, original_backward)

    def factory(owner, param):
        return tracker.make_hook(owner, param)

    tracker.factory = factory
    cls._make_backward_post_hook = factory
    return tracker


def install_gradient_batching(tracker, model, library, mode):
    if mode != tracker.mode:
        tracker.restore_factory()
        raise RuntimeError('MA10 prepare/bind worker mode changed')
    return tracker.bind(model, library, expected_parameters=PARAMETER_COUNT)
