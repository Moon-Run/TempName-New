"""MA17: submit immediately consumed column AllGather on the compute stream.

The original MA8 allocation, complete GEMM, bias and backward stay literal.
Only these synchronous forward gathers use one dedicated native NCCL
communicator. Original asynchronous backward collectives keep ProcessGroupNCCL.
"""
import ast
import ctypes
import hashlib
import inspect
import os
from pathlib import Path
import types
from contextlib import nullcontext

import ma8_adapter as cache

PROTOCOL = 'h100-ma17-direct-allgather-v1'
LOCAL_SHAPE = (256, 1, 4096)
WORLD_SIZE = 8
OUTPUT_WIDTHS = {'qkv': 1536, 'fc1': 2048}
CLASS_SHA256 = 'e051f4f0c4a90c8869f8331f55acef9e610f930c5caeb407314ff3446dc769ac'
BACKWARD_SHA256 = '73a5045f05a4ead1b9168687d911dd3d4704ee995c2d72bd3a6756b835f6dcd0'
NCCL_BFLOAT16 = 9
NCCL_UNIQUE_ID_BYTES = 128


def require(condition, message):
    if not condition:
        raise RuntimeError('MA17 ' + message)


def direct_allgather_class_source(source):
    source = cache.cached_class_source(source)
    source = cache.replace_once(source, 'class _MA8CachedRawLinear(',
                                'class _MA17DirectAllGatherLinear(')
    return cache.replace_once(source,
        '            dist_all_gather_func(all_gather_buffer, input, group=get_tensor_model_parallel_group())',
        '            _ma17_gather(all_gather_buffer, input, group=get_tensor_model_parallel_group())')


def backward_sha256(source):
    tree = ast.parse(source)
    method = next(node for node in tree.body[0].body
                  if isinstance(node, ast.FunctionDef) and node.name == 'backward')
    return hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()


def module_roles():
    return {f'decoder.layers.{index}{suffix}': role for index in range(32)
            for suffix, role in cache.SUFFIX_ROLES.items()}


def _loaded_library_path(stem, maps=None):
    """Select one already mapped shared object, never search loader paths."""
    maps = Path('/proc/self/maps').read_text() if maps is None else maps
    found = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = fields[5]
        name = Path(path.removesuffix(' (deleted)')).name
        if name == stem or name.startswith(stem + '.'):
            require(not path.endswith(' (deleted)'), 'mapped library was deleted: ' + stem)
            found.add(str(Path(path).resolve(strict=True)))
    require(len(found) == 1, 'requires exactly one already-loaded shared library: ' + stem)
    return Path(next(iter(found)))


def loaded_nccl_path(maps=None):
    return _loaded_library_path('libnccl.so', maps)


def loaded_torch_cuda_path(maps=None):
    return _loaded_library_path('libtorch_cuda.so', maps)


def verify_torch_nccl_linkage(nccl_library):
    """Verify libtorch_cuda resolves the actual same NCCL entry points.

    torch.cuda.nccl.version() reports build headers, so version equality alone
    cannot establish linkage. dlsym on an existing libtorch_cuda handle searches
    its dependency tree. Missing or different symbols reject static/ambiguous
    deployments instead of silently binding an unrelated NCCL installation.
    """
    path = loaded_torch_cuda_path()
    handle = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    addresses = {}
    for name in ('ncclAllGather', 'ncclGetVersion'):
        try:
            direct = ctypes.cast(getattr(nccl_library, name), ctypes.c_void_p).value
            dependency = ctypes.cast(getattr(handle, name), ctypes.c_void_p).value
        except (AttributeError, TypeError) as error:
            raise RuntimeError('MA17 cannot verify dynamically linked PyTorch NCCL: ' + name) from error
        require(direct is not None and direct == dependency,
                'PyTorch resolves a different NCCL implementation: ' + name)
        addresses[name] = dict(direct=direct, torch_dependency=dependency)
    return dict(linkage_verified=True, torch_cuda_library_path=str(path),
                linkage_symbols=list(addresses), linkage_symbol_addresses=addresses)


class UniqueId(ctypes.Structure):
    _fields_ = [('internal', ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


class NativeAllGather:
    """Own one blocking NCCL communicator and no CUDA stream or tensor buffer.

    No destructor calls NCCL. Explicit close synchronizes before destroy;
    detected failures abort, and the enclosing launcher supplies a hard timeout.
    """
    def __init__(self, torch, group, rank):
        self.torch, self.group, self.rank = torch, group, rank
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.current_stream().cuda_stream
        self.calls = self.async_checks = 0
        self.failed = self.closed = False
        self.comm = ctypes.c_void_p()
        self.memory_before = tuple(torch.cuda.mem_get_info(self.device))
        path = loaded_nccl_path()
        require(hasattr(os, 'RTLD_NOLOAD'), 'requires Linux RTLD_NOLOAD')
        self.library_path = str(path)
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        self.library_sha256 = digest.hexdigest()
        self.library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        self.linkage = verify_torch_nccl_linkage(self.library)
        signatures = {
            'ncclGetVersion': ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            'ncclGetUniqueId': ([ctypes.POINTER(UniqueId)], ctypes.c_int),
            'ncclCommInitRank': ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                  UniqueId, ctypes.c_int], ctypes.c_int),
            'ncclAllGather': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
            'ncclCommGetAsyncError': ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            'ncclCommDestroy': ([ctypes.c_void_p], ctypes.c_int),
            'ncclCommAbort': ([ctypes.c_void_p], ctypes.c_int),
            'ncclGetErrorString': ([ctypes.c_int], ctypes.c_char_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.library, name)
            fn.argtypes, fn.restype = args, result
        version = ctypes.c_int()
        self.check(self.library.ncclGetVersion(ctypes.byref(version)), 'get version')
        self.version = version.value
        self.torch_version = tuple(torch.cuda.nccl.version())
        require(len(self.torch_version) == 3 and self.version ==
                self.torch_version[0] * 10000 + self.torch_version[1] * 100 + self.torch_version[2],
                'loaded NCCL differs from PyTorch compiled NCCL version')
        require(self.version >= 21000, 'requires NCCL BF16 support')
        self.group_ranks = list(torch.distributed.get_process_group_ranks(group))
        require(len(self.group_ranks) == WORLD_SIZE and len(set(self.group_ranks)) == WORLD_SIZE
                and torch.distributed.get_world_size() == WORLD_SIZE
                and torch.distributed.get_world_size(group) == WORLD_SIZE
                and torch.distributed.get_rank(group) == rank
                and torch.distributed.get_rank() == self.group_ranks[rank],
                'tensor parallel rank mapping or single-node TP8 world changed')
        # All peers must enter the dedicated communicator with the same ABI.
        identities = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(identities,
            (self.version, self.library_sha256, self.group_ranks), group=group)
        require(all(item == identities[0] for item in identities),
                'ranks loaded different NCCL libraries or TP mappings')
        uid = UniqueId()
        if rank == 0:
            self.check(self.library.ncclGetUniqueId(ctypes.byref(uid)), 'get unique ID')
        payload = [bytes(uid) if rank == 0 else None]
        torch.distributed.broadcast_object_list(payload, src=self.group_ranks[0], group=group,
            device=torch.device('cuda', self.device))
        require(type(payload[0]) is bytes and len(payload[0]) == NCCL_UNIQUE_ID_BYTES,
                'invalid NCCL unique-ID broadcast')
        uid = UniqueId.from_buffer_copy(payload[0])
        self.check(self.library.ncclCommInitRank(ctypes.byref(self.comm), WORLD_SIZE, uid, rank),
                   'initialize communicator')
        require(self.comm.value is not None, 'NCCL returned a null communicator')
        # Drain setup collectives on the original ProcessGroup before using the
        # new communicator. This synchronization is outside every timed step.
        torch.cuda.synchronize(self.device)
        self.memory_after = tuple(torch.cuda.mem_get_info(self.device))

    def check(self, status, operation):
        if status != 0:
            self.failed = True
            detail = self.library.ncclGetErrorString(status)
            if self.comm.value:
                self.library.ncclCommAbort(self.comm)
                self.closed = True
            raise RuntimeError(f'MA17 NCCL {operation} failed ({status}): {detail!r}')

    def launch(self, output, input):
        require(not self.failed and not self.closed, 'cannot launch a failed/closed communicator')
        self.check(self.library.ncclAllGather(input.data_ptr(), output.data_ptr(), input.numel(),
                   NCCL_BFLOAT16, self.comm, self.stream), 'allgather')
        self.calls += 1

    def state(self):
        require(not self.failed and not self.closed, 'cannot report a failed/closed communicator')
        # Reporting is outside the original measured loop. Completion plus an
        # explicit asynchronous-error query is required before evidence is used.
        self.torch.cuda.synchronize(self.device)
        error = ctypes.c_int()
        self.check(self.library.ncclCommGetAsyncError(self.comm, ctypes.byref(error)), 'query async error')
        self.check(error.value, 'asynchronous allgather')
        self.async_checks += 1
        return dict(library_path=self.library_path, library_sha256=self.library_sha256,
                    version=self.version, torch_version=list(self.torch_version),
                    group_ranks=self.group_ranks, rank=self.rank, device=self.device,
                    stream=self.stream, calls=self.calls, initializations=1,
                    async_checks=self.async_checks, async_error=0,
                    device_memory_before=list(self.memory_before),
                    device_memory_after=list(self.memory_after),
                    device_memory_after_work=list(self.torch.cuda.mem_get_info(self.device)),
                    **self.linkage,
                    failed=False, closed=False, datatype=NCCL_BFLOAT16)

    def close(self):
        if self.closed:
            return
        self.torch.cuda.synchronize(self.device)
        self.check(self.library.ncclCommDestroy(self.comm), 'destroy communicator')
        self.closed = True


class Tracker:
    def __init__(self, torch, group, rank, mode, native, gather, source):
        self.torch, self.group, self.rank, self.mode = torch, group, rank, mode
        self.native, self.gather = native, gather
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.device, self.stream = native.device, native.stream
        self.names = tuple(module_roles())
        self.modules = {name: dict(role=role, rank=rank, calls=0, audits=0)
                        for name, role in module_roles().items()}
        self.calls = 0
        self.failed = self.active = False

    def bound_gather(self, name):
        role = self.modules[name]['role']

        def gather(output, input, group=None):
            torch = self.torch
            try:
                require(not self.failed and not self.active, 'failed or reentrant gather')
                self.active = True
                require(group is self.group and name == self.names[self.calls % len(self.names)],
                        'column collective rank-order schedule changed')
                require(torch.cuda.current_device() == self.device and
                        torch.cuda.current_stream().cuda_stream == self.stream and
                        not torch.cuda.is_current_stream_capturing(),
                        'column compute stream/device changed or graph capture requested')
                require(input.is_cuda and output.is_cuda and input.device == output.device and
                        input.device.index == self.device and input.dtype == output.dtype == torch.bfloat16
                        and tuple(input.shape) == LOCAL_SHAPE
                        and tuple(output.shape) == (LOCAL_SHAPE[0] * WORLD_SIZE, *LOCAL_SHAPE[1:])
                        and input.is_contiguous() and output.is_contiguous()
                        and input.data_ptr() != output.data_ptr(),
                        'requires disjoint original contiguous BF16 gather tensors')
                marker = (torch.autograd.profiler.record_function('ma17.column.gather.' + role)
                          if self.mode == 'profile' else nullcontext())
                with marker:
                    self.native.launch(output, input)
                if self.mode == 'smoke':
                    # The original PG communicator waits for the compute-stream
                    # native collective before its oracle, preserving common
                    # cross-communicator launch order on every rank.
                    reference = torch.empty_like(output)
                    self.gather(reference, input, group=group)
                    require(torch.equal(output.view(torch.uint8), reference.view(torch.uint8)),
                            'native gather differs bytewise from original ProcessGroup gather')
                    self.modules[name]['audits'] += 1
                self.modules[name]['calls'] += 1
                self.calls += 1
            except BaseException:
                self.failed = True
                raise
            finally:
                self.active = False
        return gather

    def state(self):
        require(not self.failed and not self.active, 'cannot report failed/active gather tracker')
        return dict(protocol=PROTOCOL, installed=True, local_shape=list(LOCAL_SHAPE),
                    world_size=WORLD_SIZE, source_sha256=self.source_sha256,
                    backward_sha256=BACKWARD_SHA256, calls=self.calls, failed=False, active=False,
                    native=self.native.state(), modules={name: dict(row) for name, row in self.modules.items()})

    def close(self):
        require(not self.active, 'cannot close an active gather')
        self.native.close()


def validate_state(state, steps, mode, rank=None):
    require(mode in ('smoke', 'timing', 'profile') and type(steps) is int and steps > 0,
            'invalid state-validation schedule')
    require(isinstance(state, dict) and state.get('protocol') == PROTOCOL and
            state.get('installed') is True and state.get('local_shape') == list(LOCAL_SHAPE)
            and state.get('world_size') == WORLD_SIZE and state.get('source_sha256') == CLASS_SHA256
            and state.get('backward_sha256') == BACKWARD_SHA256 and state.get('failed') is False
            and state.get('active') is False and state.get('calls') == steps * 256,
            'direct-allgather identity or schedule mismatch')
    native = state.get('native', {})
    actual_rank = native.get('rank')
    version = native.get('torch_version', [])
    require(type(actual_rank) is int and 0 <= actual_rank < WORLD_SIZE
            and (rank is None or rank == actual_rank) and native.get('calls') == steps * 256
            and native.get('initializations') == 1 and native.get('async_checks', 0) > 0
            and native.get('async_error') == 0 and native.get('failed') is False
            and native.get('closed') is False and native.get('datatype') == NCCL_BFLOAT16
            and type(native.get('device')) is int and 0 <= native['device'] < WORLD_SIZE
            and type(native.get('stream')) is int and native['stream'] >= 0
            and len(version) == 3 and all(type(v) is int for v in version)
            and native.get('version') == version[0] * 10000 + version[1] * 100 + version[2]
            and native['version'] >= 21000 and sorted(native.get('group_ranks', [])) == list(range(8))
            and Path(native.get('library_path', '')).is_absolute()
            and len(native.get('library_sha256', '')) == 64
            and all(c in '0123456789abcdef' for c in native['library_sha256']),
            'native communicator provenance or completed execution mismatch')
    symbols = ('ncclAllGather', 'ncclGetVersion')
    addresses = native.get('linkage_symbol_addresses', {})
    require(native.get('linkage_verified') is True
            and Path(native.get('torch_cuda_library_path', '')).is_absolute()
            and native.get('linkage_symbols') == list(symbols)
            and set(addresses) == set(symbols)
            and all(type(addresses[name].get('direct')) is int
                    and addresses[name]['direct'] > 0
                    and addresses[name]['direct'] == addresses[name].get('torch_dependency')
                    for name in symbols), 'unverified or changed dynamic PyTorch NCCL linkage')
    require(all(isinstance(native.get(key), list) and len(native[key]) == 2
                and all(type(v) is int and v >= 0 for v in native[key])
                and native[key][0] <= native[key][1]
                for key in ('device_memory_before', 'device_memory_after', 'device_memory_after_work')),
            'missing device-memory observations around communicator initialization')
    modules = state.get('modules', {})
    require(set(modules) == set(module_roles()), 'direct-allgather module coverage mismatch')
    for name, role in module_roles().items():
        row = modules[name]
        require(row.get('role') == role and row.get('rank') == actual_rank
                and row.get('calls') == steps * 4
                and row.get('audits') == (steps * 4 if mode == 'smoke' else 0),
                'direct-allgather execution or audit count mismatch: ' + name)
    return True


def install_direct_allgather(model, mode):
    import torch
    from megatron.core.tensor_parallel import layers
    require(mode in ('smoke', 'timing', 'profile'), 'invalid execution mode')
    require(not hasattr(model, '_ma17_direct_allgather'), 'duplicate installation')
    require(torch.__version__.split('+')[0] == '2.6.0', 'requires pinned PyTorch 2.6.0')
    common = getattr(model, '_ma8_activation_setup', None)
    require(isinstance(common, dict) and common.get('protocol') == cache.PROTOCOL
            and common.get('installed') is True and common.get('arm') == 'F4-cache'
            and common.get('module_roles') == module_roles(), 'requires complete original MA8 cache')
    config = model.config
    require(model.training and config.num_layers == 32 and config.hidden_size == 4096
            and config.tensor_model_parallel_size == 8 and config.pipeline_model_parallel_size == 1
            and config.context_parallel_size == 1 and config.sequence_parallel
            and not config.gradient_accumulation_fusion and config.recompute_granularity is None
            and not config.enable_cuda_graph and not config.external_cuda_graph
            and not getattr(config, 'cpu_offloading', False) and not config.defer_embedding_wgrad_compute,
            'requires frozen TP8/PP1/CP1 training configuration')
    source = cache.require_source(inspect.getsource(layers.LinearWithGradAccumulationAndAsyncCommunication),
                                  cache.CLASS_SHA256)
    cache.require_source(inspect.getsource(layers.ColumnParallelLinear.forward), cache.COLUMN_FORWARD_SHA256)
    cache.require_source(inspect.getsource(layers.linear_with_grad_accumulation_and_async_allreduce),
                         cache.LINEAR_WRAPPER_SHA256)
    candidate = direct_allgather_class_source(source)
    require(hashlib.sha256(candidate.encode()).hexdigest() == CLASS_SHA256
            and backward_sha256(candidate) == backward_sha256(cache.cached_class_source(source))
            == BACKWARD_SHA256, 'registered candidate source or original backward changed')
    rank, group = layers.get_tensor_model_parallel_rank(), layers.get_tensor_model_parallel_group()
    require(layers.get_tensor_model_parallel_world_size() == WORLD_SIZE and 0 <= rank < WORLD_SIZE,
            'requires original eight-rank tensor parallel group')
    modules, selected = dict(model.named_modules()), []
    for name, role in module_roles().items():
        module = modules[name]
        current = getattr(module.forward, '__func__', None)
        require(type(module) is layers.ColumnParallelLinear and current is not None
                and current.__code__ is layers.ColumnParallelLinear.forward.__code__,
                'cached column wrapper changed: ' + name)
        wrapper = current.__globals__.get('linear_with_grad_accumulation_and_async_allreduce')
        require(wrapper is not None and wrapper.__code__ is
                layers.linear_with_grad_accumulation_and_async_allreduce.__code__,
                'cached linear wrapper changed: ' + name)
        cached = wrapper.__globals__.get('LinearWithGradAccumulationAndAsyncCommunication')
        require(cached is not None and cached.__name__ == '_MA8CachedRawLinear'
                and module.training and module.sequence_parallel and not module.allreduce_dgrad
                and not module.gradient_accumulation_fusion and not module.explicit_expert_comm
                and not module.disable_grad_reduce and not module.gather_output
                and module.weight is not None and module.weight.requires_grad
                and module.weight.dtype == torch.bfloat16 and module.weight.is_cuda
                and module.weight.is_contiguous()
                and tuple(module.weight.shape) == (OUTPUT_WIDTHS[role], LOCAL_SHAPE[-1]),
                'incompatible cached column: ' + name)
        selected.append((name, module))
    native = NativeAllGather(torch, group, rank)
    tracker = Tracker(torch, group, rank, mode, native, layers.dist_all_gather_func, candidate)
    prepared = []
    try:
        code = compile(candidate, '<ma17-direct-allgather-pinned-megatron>', 'exec')
        for name, module in selected:
            namespace = dict(vars(layers), _ma17_gather=tracker.bound_gather(name))
            exec(code, namespace)
            wrapper = cache._clone_function(layers.linear_with_grad_accumulation_and_async_allreduce,
                {'LinearWithGradAccumulationAndAsyncCommunication': namespace['_MA17DirectAllGatherLinear']})
            forward = cache._clone_function(layers.ColumnParallelLinear.forward,
                {'linear_with_grad_accumulation_and_async_allreduce': wrapper})
            prepared.append((module, forward))
    except BaseException:
        native.close()
        raise
    for module, forward in prepared:
        module.forward = types.MethodType(forward, module)
    model._ma17_direct_allgather = tracker
    return tracker
