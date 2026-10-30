"""MA18: fixed-size copy-engine IPC transport for column-forward AllGather.

The original MA8 allocation, complete GEMM, bias and backward stay literal.
Two sender slots and the original binary publication barrier protect payloads.
Original backward collectives keep ProcessGroupNCCL.
"""
import ast
import ctypes
import hashlib
import inspect
import os
import socket
import types
from contextlib import nullcontext

import ma8_adapter as cache

PROTOCOL = 'h100-ma18-ipc-allgather-v1'
LOCAL_SHAPE = (256, 1, 4096)
WORLD_SIZE = 8
OUTPUT_WIDTHS = {'qkv': 1536, 'fc1': 2048}
# Only the class name and forward gather binding differ from the pinned MA8.
CLASS_SHA256 = 'ef46767fa08790a5eae941581bc493dc33549b3419c73cf87dedf0d908e128ad'
BACKWARD_SHA256 = '73a5045f05a4ead1b9168687d911dd3d4704ee995c2d72bd3a6756b835f6dcd0'
STAGING_BYTES = 2 * 1024 * 1024
SLOTS = 2
SELF_TEST_CALLS = 20


def require(condition, message):
    if not condition:
        raise RuntimeError('MA18 ' + message)


def ipc_allgather_class_source(source):
    source = cache.cached_class_source(source)
    source = cache.replace_once(source, 'class _MA8CachedRawLinear(',
                                'class _MA18IpcAllGatherLinear(')
    return cache.replace_once(source,
        '            dist_all_gather_func(all_gather_buffer, input, group=get_tensor_model_parallel_group())',
        '            _ma18_gather(all_gather_buffer, input, group=get_tensor_model_parallel_group())')


def backward_sha256(source):
    tree = ast.parse(source)
    method = next(node for node in tree.body[0].body
                  if isinstance(node, ast.FunctionDef) and node.name == 'backward')
    return hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()


def module_roles():
    return {f'decoder.layers.{index}{suffix}': role for index in range(32)
            for suffix, role in cache.SUFFIX_ROLES.items()}


class NativeAllGather:
    """Two rank-owned IPC sender slots, one original binary publication barrier.

    All operations use one fixed compute stream. Barrier k+1 establishes that
    every rank completed its reads of slot k before any rank overwrites k+2.
    Outputs are the original independent MA8 allocations, never staging views.
    """
    def __init__(self, torch, library, group, rank, mode):
        import flux
        self.torch, self.library, self.group, self.rank = torch, library, group, rank
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.current_stream().cuda_stream
        self.calls = 0
        self.failed = self.closed = False
        self.self_test = None
        self.memory_before = tuple(torch.cuda.mem_get_info(self.device))
        self.group_ranks = list(torch.distributed.get_process_group_ranks(group))
        require(torch.cuda.device_count() == WORLD_SIZE
                and len(self.group_ranks) == WORLD_SIZE
                and sorted(self.group_ranks) == list(range(WORLD_SIZE))
                and torch.distributed.get_world_size() == WORLD_SIZE
                and torch.distributed.get_world_size(group) == WORLD_SIZE
                and torch.distributed.get_rank(group) == rank
                and torch.distributed.get_rank() == self.group_ranks[rank]
                and self.device == self.group_ranks[rank],
                'requires single-node TP8 and the original group rank mapping')
        require(all(peer == self.device or torch.cuda.can_device_access_peer(self.device, peer)
                    for peer in range(WORLD_SIZE)), 'requires peer access to all seven GPUs')
        identities = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(identities,
            (socket.gethostname(), os.environ.get('CUDA_VISIBLE_DEVICES'), self.group_ranks,
             self.device), group=group)
        require(all(item[:3] == identities[0][:3] for item in identities)
                and [item[3] for item in identities] == self.group_ranks,
                'IPC requires one host, identical visible-device ordering and eight distinct rank devices')
        signatures = {
            'ma18_ipc_protocol': ([], ctypes.c_int),
            'ma18_ipc_stage_cuda': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
                                    ctypes.c_void_p], ctypes.c_int),
            'ma18_ipc_pull_cuda': ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(library, name)
            fn.argtypes, fn.restype = args, result
        require(library.ma18_ipc_protocol() == 18,
                'requires MA18 native APIs and a Flux build with NVSHMEM disabled')
        self.peers = [flux.create_tensor_list(list(LOCAL_SHAPE), torch.bfloat16, group,
                                               ring_mode=False, init_zero=False)
                      for _ in range(SLOTS)]
        require(all(len(slot) == WORLD_SIZE and
                    len({value.data_ptr() for value in slot}) == WORLD_SIZE
                    for slot in self.peers)
                and len({value.data_ptr() for slot in self.peers for value in slot}) == SLOTS * WORLD_SIZE,
                'IPC setup returned missing peers or aliased sender slots')
        self.pointer_tables = [(ctypes.c_void_p * WORLD_SIZE)(*(value.data_ptr() for value in slot))
                               for slot in self.peers]
        self.barrier = flux.GroupBarrier(group, ring_mode=False)
        # GroupBarrier zeros its local flag buffer on the compute stream. Wait
        # for all ranks' initialization before any peer may publish to it.
        torch.cuda.synchronize(self.device)
        torch.distributed.barrier(group=group)
        torch.cuda.synchronize(self.device)
        self.memory_after = tuple(torch.cuda.mem_get_info(self.device))
        if mode == 'smoke':
            self.self_test = self._self_test()

    def check(self, status, operation):
        if status:
            self.failed = True
            raise RuntimeError(f'MA18 CUDA {operation} submission failed: {status}')

    def launch(self, output, input):
        require(not self.failed and not self.closed, 'cannot launch failed/closed IPC transport')
        try:
            slot = self.calls % SLOTS
            self.check(self.library.ma18_ipc_stage_cuda(self.peers[slot][self.rank].data_ptr(),
                input.data_ptr(), STAGING_BYTES, self.stream), 'stage')
            self.barrier.barrier_all(self.stream)
            self.check(self.library.ma18_ipc_pull_cuda(output.data_ptr(), self.pointer_tables[slot],
                self.rank, STAGING_BYTES, self.stream), 'pull')
            self.calls += 1
        except BaseException:
            self.failed = True
            raise

    def _self_test(self):
        torch = self.torch
        values = [torch.full(LOCAL_SHAPE, self.rank * 32 + index, dtype=torch.bfloat16,
                             device=torch.device('cuda', self.device))
                  for index in range(SELF_TEST_CALLS)]
        outputs = [torch.empty((LOCAL_SHAPE[0] * WORLD_SIZE, *LOCAL_SHAPE[1:]),
                               dtype=torch.bfloat16, device=values[0].device)
                   for _ in values]
        torch.cuda.synchronize(self.device)
        torch.distributed.barrier(group=self.group)
        # Queue all calls before examining a single output. Per-call allclose or
        # NCCL oracle would synchronize ranks and could hide slot reuse races.
        for index, (input, output) in enumerate(zip(values, outputs)):
            if self.rank == index % WORLD_SIZE:
                torch.cuda._sleep(200000)
            self.launch(output, input)
        torch.cuda.synchronize(self.device)
        for index, output in enumerate(outputs):
            reference = torch.cat([torch.full_like(values[0], peer * 32 + index)
                                   for peer in range(WORLD_SIZE)])
            require(torch.equal(output.view(torch.uint8), reference.view(torch.uint8)),
                    'queued delayed-rank IPC self-test or retained output failed')
        torch.distributed.barrier(group=self.group)
        torch.cuda.synchronize(self.device)
        require(SELF_TEST_CALLS % SLOTS == 0, 'self-test must end on the original slot parity')
        self.calls = 0
        return dict(passed=True, calls=SELF_TEST_CALLS, rank_delays=True,
                    retained_outputs=True, all_bytes=True, queued_before_audit=True)

    def state(self):
        require(not self.failed and not self.closed, 'cannot report failed/closed IPC transport')
        self.torch.cuda.synchronize(self.device)
        return dict(rank=self.rank, device=self.device, stream=self.stream,
                    group_ranks=self.group_ranks, calls=self.calls, initializations=1,
                    slots=SLOTS, staging_bytes=STAGING_BYTES,
                    local_staging_bytes=SLOTS * STAGING_BYTES, copies_per_call=9,
                    publication_barriers_per_call=1, native_protocol=18,
                    copy_order='cyclic_sources_rank_ordered_destination',
                    failed=False, closed=False, self_test=self.self_test,
                    device_memory_before=list(self.memory_before),
                    device_memory_after=list(self.memory_after),
                    device_memory_after_work=list(self.torch.cuda.mem_get_info(self.device)))

    def close(self):
        if self.closed:
            return
        self.torch.cuda.synchronize(self.device)
        self.torch.distributed.barrier(group=self.group)
        self.torch.cuda.synchronize(self.device)
        # Fresh-process workers own the imported mappings until process exit.
        # Retain exporter allocations as well: freeing an exporter before a
        # remote cudaIpcCloseMemHandle would violate the CUDA IPC lifetime rule.
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
                marker = (torch.autograd.profiler.record_function('ma18.column.gather.' + role)
                          if self.mode == 'profile' else nullcontext())
                with marker:
                    self.native.launch(output, input)
                if self.mode == 'smoke':
                    # Oracle is smoke-only; queued reuse is separately tested at setup.
                    reference = torch.empty_like(output)
                    self.gather(reference, input, group=group)
                    require(torch.equal(output.view(torch.uint8), reference.view(torch.uint8)),
                            'IPC gather differs bytewise from original ProcessGroup gather')
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
            'ipc-allgather identity or schedule mismatch')
    native = state.get('native', {})
    actual_rank = native.get('rank')
    require(type(actual_rank) is int and 0 <= actual_rank < WORLD_SIZE
            and (rank is None or rank == actual_rank) and native.get('calls') == steps * 256
            and native.get('initializations') == 1 and native.get('slots') == SLOTS
            and native.get('staging_bytes') == STAGING_BYTES
            and native.get('local_staging_bytes') == SLOTS * STAGING_BYTES
            and native.get('copies_per_call') == 9
            and native.get('publication_barriers_per_call') == 1
            and native.get('native_protocol') == 18
            and native.get('copy_order') == 'cyclic_sources_rank_ordered_destination'
            and native.get('failed') is False and native.get('closed') is False
            and type(native.get('device')) is int and 0 <= native['device'] < WORLD_SIZE
            and type(native.get('stream')) is int and native['stream'] >= 0
            and sorted(native.get('group_ranks', [])) == list(range(WORLD_SIZE))
            and native['device'] == native['group_ranks'][actual_rank],
            'native IPC transport identity or completed execution mismatch')
    expected_test = (dict(passed=True, calls=SELF_TEST_CALLS, rank_delays=True,
                         retained_outputs=True, all_bytes=True, queued_before_audit=True)
                     if mode == 'smoke' else None)
    require(native.get('self_test') == expected_test, 'missing queued reuse audit or audit leaked into timing')
    require(all(isinstance(native.get(key), list) and len(native[key]) == 2
                and all(type(v) is int and v >= 0 for v in native[key])
                and native[key][0] <= native[key][1]
                for key in ('device_memory_before', 'device_memory_after', 'device_memory_after_work')),
            'missing device-memory observations around IPC initialization')
    modules = state.get('modules', {})
    require(set(modules) == set(module_roles()), 'ipc-allgather module coverage mismatch')
    for name, role in module_roles().items():
        row = modules[name]
        require(row.get('role') == role and row.get('rank') == actual_rank
                and row.get('calls') == steps * 4
                and row.get('audits') == (steps * 4 if mode == 'smoke' else 0),
                'ipc-allgather execution or audit count mismatch: ' + name)
    return True


def install_ipc_allgather(model, library, mode):
    import torch
    from megatron.core.tensor_parallel import layers
    require(mode in ('smoke', 'timing', 'profile'), 'invalid execution mode')
    require(not hasattr(model, '_ma18_ipc_allgather'), 'duplicate installation')
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
    candidate = ipc_allgather_class_source(source)
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
    native = NativeAllGather(torch, library, group, rank, mode)
    tracker = Tracker(torch, group, rank, mode, native, layers.dist_all_gather_func, candidate)
    prepared = []
    try:
        code = compile(candidate, '<ma18-ipc-allgather-pinned-megatron>', 'exec')
        for name, module in selected:
            namespace = dict(vars(layers), _ma18_gather=tracker.bound_gather(name))
            exec(code, namespace)
            wrapper = cache._clone_function(layers.linear_with_grad_accumulation_and_async_allreduce,
                {'LinearWithGradAccumulationAndAsyncCommunication': namespace['_MA18IpcAllGatherLinear']})
            forward = cache._clone_function(layers.ColumnParallelLinear.forward,
                {'linear_with_grad_accumulation_and_async_allreduce': wrapper})
            prepared.append((module, forward))
    except BaseException:
        native.close()
        raise
    for module, forward in prepared:
        module.forward = types.MethodType(forward, module)
    model._ma18_ipc_allgather = tracker
    return tracker
