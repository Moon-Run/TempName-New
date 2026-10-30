"""Fixed-storage master-to-model publication at the original optimizer boundary.

Only child optimizer instance methods are replaced. Every child's Adam update
still precedes its own publication. No backward hooks, queues, graph replay or
per-step tensor descriptor construction are installed. The frozen experiment
does not load checkpoints, replace Parameter.data, resize, or offload parameters.
Detached storage aliases prevent dangling descriptor addresses; state() verifies
every original object and allocation before accepting a completed window.
"""
import ctypes
import hashlib
import inspect
import textwrap
import types
from contextlib import nullcontext

PROTOCOL = 'h100-ma19-fixed-parameter-publish-v1'
CHUNK_SIZE = 32768
THREADS = 256
PARAMETER_COUNT = 389
ELEMENT_COUNT = 849207296
BLOCK_COUNT = 26146
MARKER = 'ma19.optimizer.parameter_publish'
SOURCE_HASHES = {
    '_multi_tensor_copy_this_to_that': '2788b0572db3ae06792290baeaf36b75a6ab33ab539a019e88817597adcac75c',
    '_get_model_and_main_params_data_float16': 'f89ecbf229880a167983ec78017c2080993eb945d20da0f9a9cc4581c089a3d5',
    '_copy_main_params_to_model_params': '6c28b071f2b1d890045da8682a58cd68c3bb80ada18ef9f92549df4d438801a0',
}


def require_source(function, name):
    source = textwrap.dedent(inspect.getsource(function)).strip() + '\n'
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_HASHES[name]:
        raise RuntimeError('MA19 requires pinned optimizer source: ' + name)


def descriptor_rows(pairs):
    rows, chunks = [], []
    for index, (source, target) in enumerate(pairs):
        elements = source.numel()
        if elements <= 0 or elements != target.numel():
            raise RuntimeError('MA19 requires equal nonempty source and target tensors')
        rows.append([source.data_ptr(), target.data_ptr(), elements])
        chunks.extend([index, offset] for offset in range(0, elements, CHUNK_SIZE))
    if not rows or len(chunks) >= 2**31:
        raise RuntimeError('MA19 descriptor/block count is outside the native ABI')
    return rows, chunks


def tensor_identity(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device), tensor.storage_offset())


def exact_equal(torch, left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


class NativePlan:
    def __init__(self, torch, library, pairs):
        self.torch, self.library = torch, library
        self.pairs = tuple(pairs)
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.current_stream().cuda_stream
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('MA19 cannot initialize during CUDA Graph capture')
        for source, target in self.pairs:
            if (source.dtype != torch.float32 or target.dtype != torch.bfloat16 or
                    not source.is_cuda or not target.is_cuda or
                    source.device.index != self.device or target.device != source.device or
                    not source.is_contiguous() or not target.is_contiguous() or
                    tuple(source.shape) != tuple(target.shape)):
                raise RuntimeError('MA19 requires fixed contiguous CUDA FP32/BF16 parameter pairs')
        self.identities = tuple((tensor_identity(s), tensor_identity(t)) for s, t in self.pairs)
        # Aliases keep the actual allocations alive even if an unsupported user
        # rebind happens. Verification then rejects the window, without a UAF.
        self.storage_aliases = tuple((s.detach(), t.detach()) for s, t in self.pairs)
        rows, chunks = descriptor_rows(self.storage_aliases)
        intervals = sorted([(t.data_ptr(), t.data_ptr() + 2 * t.numel()) for _, t in self.pairs] +
                           [(s.data_ptr(), s.data_ptr() + 4 * s.numel()) for s, _ in self.pairs])
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise RuntimeError('MA19 source/destination allocations overlap')
        for name, expected in [('ma19_copy_protocol', 19), ('ma19_copy_chunk_size', CHUNK_SIZE)]:
            fn = getattr(library, name)
            fn.argtypes, fn.restype = [], ctypes.c_int
            if fn() != expected:
                raise RuntimeError('Wrong MA19 native protocol: ' + name)
        self.launch = library.ma19_copy_cuda
        self.launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.launch.restype = ctypes.c_int
        self.descriptors = torch.tensor(rows, dtype=torch.int64, device=self.pairs[0][0].device)
        self.chunks = torch.tensor(chunks, dtype=torch.int64, device=self.pairs[0][0].device)
        self.blocks = len(chunks)
        self.elements = sum(row[2] for row in rows)
        self.calls = 0
        self.failed = False

    def publish(self):
        if self.failed:
            raise RuntimeError('MA19 plan is failed; refusing retry')
        try:
            if (self.torch.cuda.current_device() != self.device or
                    self.torch.cuda.current_stream().cuda_stream != self.stream or
                    self.torch.cuda.is_current_stream_capturing()):
                raise RuntimeError('MA19 device/stream changed or CUDA Graph capture requested')
            status = self.launch(self.descriptors.data_ptr(), self.chunks.data_ptr(),
                                 self.blocks, self.stream)
            if status:
                raise RuntimeError('MA19 parameter publication CUDA launch failed: ' + str(status))
            self.calls += 1
        except BaseException:
            self.failed = True
            raise

    def verify_storage(self):
        actual = tuple((tensor_identity(s), tensor_identity(t)) for s, t in self.pairs)
        aliases = tuple((s.data_ptr(), t.data_ptr()) for s, t in self.storage_aliases)
        expected_aliases = tuple((s[1], t[1]) for s, t in self.identities)
        if self.failed or actual != self.identities or aliases != expected_aliases:
            self.failed = True
            raise RuntimeError('MA19 parameter storage/layout changed; refusing completed evidence')

    def state(self):
        self.verify_storage()
        return dict(tensors=len(self.pairs), elements=self.elements, blocks=self.blocks,
                    calls=self.calls, initializations=1, failed=self.failed,
                    device=self.device, stream=self.stream, storage_verified=True,
                    descriptor_bytes=self.descriptors.numel() * 8 + self.chunks.numel() * 8)


def native_self_test(torch, library):
    """Actual CUDA copy_ oracle: every BF16 midpoint and adjacent FP32 bit.

    Includes both signs, normal/subnormal boundaries, overflow, infinities and
    all exponent-255 payload classes. No RNG is consumed.
    """
    top = torch.arange(65536, dtype=torch.int64, device='cuda') << 16
    bits = torch.cat([top + offset for offset in (0, 0x7fff, 0x8000, 0x8001)]).to(torch.int32)
    source = bits.view(torch.float32)
    sizes = [1, CHUNK_SIZE - 1, CHUNK_SIZE + 1, source.numel() - 2 * CHUNK_SIZE - 1]
    # Both layouts see ALL midpoint/adjacent patterns. In particular, testing
    # only split slices would put most midpoint values in the scalar fallback.
    for pieces in ([source], list(source.split(sizes))):
        targets = [torch.empty_like(value, dtype=torch.bfloat16) for value in pieces]
        plan = NativePlan(torch, library, list(zip(pieces, targets)))
        for _ in range(2):
            plan.publish()
            references = [torch.empty_like(target).copy_(value) for value, target in zip(pieces, targets)]
            if not all(exact_equal(torch, target, expected) for target, expected in zip(targets, references)):
                raise RuntimeError('MA19 native FP32-to-BF16 conversion differs from torch.copy_')
        plan.verify_storage()
    torch.cuda.synchronize()
    return dict(passed=True, patterns=source.numel(), repetitions=2,
                layouts=['aligned_full', 'unaligned_split'], bitwise=True,
                signed_zero_subnormal_nan_inf=True, rng_unchanged=True)


class ParameterPublisher:
    def __init__(self, torch, owner, original, native, mode):
        self.torch, self.owner, self.original, self.native, self.mode = torch, owner, original, native, mode
        self.audits = 0
        self.failed = False
        self.bound = None
        self.model_groups = owner.float16_groups
        self.master_groups = owner.fp32_from_float16_groups

    def publish(self):
        if self.failed:
            raise RuntimeError('MA19 publisher is failed; refusing retry')
        try:
            with (self.torch.autograd.profiler.record_function(MARKER)
                  if self.mode == 'profile' else nullcontext()):
                self.native.publish()
            if self.mode == 'smoke':
                # Run the *actual original method*, then compare every byte to
                # the candidate result. Smoke-only copies never enter timing.
                observed = [target.detach().clone() for _, target in self.native.pairs]
                self.original()
                if not all(exact_equal(self.torch, saved, target)
                           for saved, (_, target) in zip(observed, self.native.pairs)):
                    raise RuntimeError('MA19 published model weights differ from original copy_')
                self.audits += 1
        except BaseException:
            self.failed = True
            raise

    def state(self):
        if (self.failed or self.owner.float16_groups is not self.model_groups or
                self.owner.fp32_from_float16_groups is not self.master_groups or
                self.owner._copy_main_params_to_model_params is not self.bound or
                len(self.model_groups) != len(self.master_groups) or any(
                    len(m) != len(s) for m, s in zip(self.model_groups, self.master_groups))):
            raise RuntimeError('MA19 optimizer binding changed or failed')
        current = tuple((master, model) for models, masters in
                        zip(self.model_groups, self.master_groups) for model, master in zip(models, masters))
        if len(current) != len(self.native.pairs) or any(
                s is not old_s or t is not old_t for (s, t), (old_s, old_t) in zip(current, self.native.pairs)):
            raise RuntimeError('MA19 optimizer parameter order/identity changed')
        return dict(self.native.state(), audits=self.audits)


class Tracker:
    def __init__(self, publishers, self_test):
        self.publishers, self.self_test = publishers, self_test

    def state(self):
        children = [publisher.state() for publisher in self.publishers]
        return dict(protocol=PROTOCOL, installed=True, failed=False, chunk_size=CHUNK_SIZE,
                    threads=THREADS, source_sha256=SOURCE_HASHES, children=children,
                    child_count=len(children), tensors=sum(c['tensors'] for c in children),
                    elements=sum(c['elements'] for c in children),
                    calls=sum(c['calls'] for c in children), audits=sum(c['audits'] for c in children),
                    storage_verified=True, self_test=self.self_test)


def install_parameter_publish(optimizer, library, mode):
    import torch
    from megatron.core.optimizer import optimizer as module
    if mode not in ('smoke', 'timing', 'profile') or hasattr(optimizer, '_ma19_parameter_publish'):
        raise RuntimeError('Invalid or duplicate MA19 installation')
    if torch.__version__.split('+')[0] != '2.6.0':
        raise RuntimeError('MA19 requires pinned PyTorch 2.6.0')
    cls = module.Float16OptimizerWithFloat16Params
    require_source(module._multi_tensor_copy_this_to_that, '_multi_tensor_copy_this_to_that')
    for name in ('_get_model_and_main_params_data_float16', '_copy_main_params_to_model_params'):
        require_source(getattr(cls, name), name)
    if type(optimizer) is module.ChainedOptimizer:
        children = list(optimizer.chained_optimizers)
    else:
        children = [optimizer]
    if not children or any(type(child) is not cls for child in children):
        raise RuntimeError('MA19 requires original BF16 child optimizers')
    publishers = []
    for child in children:
        if (child.is_stub_optimizer or not child.config.bf16 or child._dummy_overflow_buf is not None or
                child.grad_scaler is not None or any(child.fp32_from_fp32_groups) or
                '_copy_main_params_to_model_params' in child.__dict__ or
                len(child.float16_groups) != len(child.fp32_from_float16_groups) or
                any(len(m) != len(s) for m, s in zip(child.float16_groups, child.fp32_from_float16_groups))):
            raise RuntimeError('MA19 requires frozen BF16 master/model optimizer groups')
        pairs = [(master, model) for models, masters in
                 zip(child.float16_groups, child.fp32_from_float16_groups)
                 for model, master in zip(models, masters)]
        if not pairs:
            raise RuntimeError('MA19 does not support empty optimizer children')
        native = NativePlan(torch, library, pairs)
        publishers.append(ParameterPublisher(torch, child, child._copy_main_params_to_model_params, native, mode))
    if (sum(len(p.native.pairs) for p in publishers) != PARAMETER_COUNT or
            sum(p.native.elements for p in publishers) != ELEMENT_COUNT or
            sum(p.native.blocks for p in publishers) != BLOCK_COUNT or len(publishers) != 1):
        raise RuntimeError('MA19 fixed model parameter inventory/optimizer grouping differs')
    self_test = native_self_test(torch, library) if mode == 'smoke' else None
    # Transactional installation: all objects and CUDA arithmetic are validated
    # before replacing any optimizer method.
    for publisher in publishers:
        def publish(owner, _publisher=publisher):
            return _publisher.publish()
        publisher.bound = types.MethodType(publish, publisher.owner)
        publisher.owner._copy_main_params_to_model_params = publisher.bound
    tracker = Tracker(publishers, self_test)
    optimizer._ma19_parameter_publish = tracker
    return tracker


def validate_state(state, steps, mode):
    def require(condition, message):
        if not condition:
            raise RuntimeError('MA19 evidence invalid: ' + message)
    require(mode in ('smoke', 'timing', 'profile') and isinstance(steps, int) and steps > 0, 'mode/steps')
    require(state.get('protocol') == PROTOCOL and state.get('installed') is True and
            state.get('failed') is False and state.get('storage_verified') is True, 'identity/storage')
    require(state.get('chunk_size') == CHUNK_SIZE and state.get('threads') == THREADS and
            state.get('source_sha256') == SOURCE_HASHES, 'source/geometry')
    require(state.get('tensors') == PARAMETER_COUNT and state.get('elements') == ELEMENT_COUNT and
            state.get('child_count') == 1 and len(state.get('children', [])) == 1, 'inventory')
    require(state.get('calls') == steps and state.get('audits') == (steps if mode == 'smoke' else 0), 'calls/audits')
    child = state['children'][0]
    require(child.get('calls') == steps and child.get('audits') == state['audits'] and
            child.get('tensors') == PARAMETER_COUNT and child.get('elements') == ELEMENT_COUNT and
            child.get('blocks') == BLOCK_COUNT and child.get('initializations') == 1 and
            child.get('failed') is False and child.get('storage_verified') is True and
            child.get('descriptor_bytes') == PARAMETER_COUNT * 24 + BLOCK_COUNT * 16, 'child')
    if mode == 'smoke':
        test = state.get('self_test') or {}
        require(test.get('passed') is True and test.get('patterns') == 262144 and
                test.get('repetitions') == 2 and test.get('bitwise') is True and
                test.get('layouts') == ['aligned_full', 'unaligned_split'] and
                test.get('signed_zero_subnormal_nan_inf') is True, 'conversion self-test')
    else:
        require(state.get('self_test') is None, 'oracle ran in measured process')
    return state
