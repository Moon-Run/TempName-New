"""MA7: defer the unchanged BF16 weight GEMM to one auxiliary stream.

The frozen non-overlap Megatron DDP path accumulates BF16 weight gradients into
its original FP32 main_grad. A public schedule boundary joins once per backward
microbatch, before finalize_model_grads and every optimizer operation.
"""
import inspect

import ma4_f_adapter as baseline

PROTOCOL = 'h100-ma7-f4-wgrad-v1'
BACKWARD_SCHEDULE = 'independent_weight_stream'
ARMS = ('F4', 'F4-wgrad')


class BackwardState(dict):
    def __init__(self, arm, allow_cpu=False):
        if arm not in ARMS:
            raise ValueError('Unknown MA7 arm: ' + str(arm))
        super().__init__(protocol=PROTOCOL, arm=arm, backward_schedule=BACKWARD_SCHEDULE,
            forward_calls=0, backward_calls=0, baseline_backward_calls=0,
            deferred_wgrad_calls=0, cuda_wgrad_calls=0, cpu_wgrad_calls=0,
            main_grad_accumulations=0, standalone_grad_assignments=0,
            drained_wgrad_calls=0, drain_calls=0, pending_wgrad=0,
            peak_pending_wgrad=0, async_gather_calls=0, wait_calls=0,
            failed=False, active=False)
        self._arm = arm
        self._allow_cpu = bool(allow_cpu)
        self._op = None
        self._weight = None
        self._weight_ptr = None
        self._main_grad = None
        self._main_grad_ptr = None
        self._queue = None
        self._training = False
        self._in_microbatch = False


def new_backward_state(arm, allow_cpu=False):
    return BackwardState(arm, allow_cpu)


SUPPORT_SOURCE = '''
def _ma7_poison(state, message):
    state['failed'] = True
    if state._queue is not None:
        state._queue.poison()
    raise RuntimeError(message)


def _ma7_validate_state(state, op=None):
    expected = {'protocol', 'arm', 'backward_schedule', 'forward_calls', 'backward_calls',
        'baseline_backward_calls', 'deferred_wgrad_calls', 'cuda_wgrad_calls',
        'cpu_wgrad_calls', 'main_grad_accumulations', 'standalone_grad_assignments',
        'drained_wgrad_calls', 'drain_calls', 'pending_wgrad', 'peak_pending_wgrad',
        'async_gather_calls', 'wait_calls', 'failed', 'active'}
    if (not isinstance(state, dict) or set(state) != expected or
            state['protocol'] != PROTOCOL or state['arm'] not in ARMS or
            getattr(state, '_arm', None) != state['arm'] or
            state['backward_schedule'] != BACKWARD_SCHEDULE or
            state['async_gather_calls'] != 0 or state['wait_calls'] != 0):
        if isinstance(state, dict): state['failed'] = True
        raise RuntimeError('Invalid MA7 backward state')
    if state['failed'] or state['active']:
        _ma7_poison(state, 'MA7 cannot retry or reenter a failed backward')
    if op is not None:
        if state._op is None:
            state._op = op
        elif state._op is not op:
            _ma7_poison(state, 'MA7 backward state belongs to another projection instance')


def _ma7_bind_weight(state, weight):
    if (not weight.is_leaf or not weight.requires_grad or not weight.is_contiguous() or
            weight.dtype not in (torch.bfloat16, torch.float32) or
            (not weight.is_cuda and not state._allow_cpu)):
        _ma7_poison(state, 'MA7 requires a CUDA contiguous leaf weight (CPU is test-only)')
    main_grad = getattr(weight, 'main_grad', None)
    if main_grad is not None and (main_grad.shape != weight.shape or
            main_grad.dtype != torch.float32 or main_grad.device != weight.device or
            not main_grad.is_contiguous() or main_grad.requires_grad):
        _ma7_poison(state, 'MA7 requires the original contiguous FP32 main_grad buffer')
    if state._weight is None:
        state._weight = weight
        state._weight_ptr = weight.data_ptr()
        state._main_grad = main_grad
        state._main_grad_ptr = None if main_grad is None else main_grad.data_ptr()
    elif (state._weight is not weight or state._weight_ptr != weight.data_ptr() or
            state._main_grad is not main_grad or
            state._main_grad_ptr != (None if main_grad is None else main_grad.data_ptr())):
        _ma7_poison(state, 'MA7 weight or main_grad ownership/storage changed')
    if state._training and (main_grad is None or weight.dtype != torch.bfloat16 or
            getattr(weight, 'zero_out_wgrad', False)):
        _ma7_poison(state, 'MA7 training requires the frozen BF16 weight and native main_grad')


class DeferredWgradQueue:
    """One stream, at most one contribution per module and 64 per microbatch.

    Strong input/full_grad/result references remain alive until the microbatch
    stream join is enqueued. record_stream additionally protects their allocator
    lifetime when those references are released before GPU completion.
    """
    def __init__(self, max_pending=64):
        self.max_pending = max_pending
        self.states = []
        self.pending = []
        self.failed = False
        self.device = None
        self.stream = None
        self.main_stream = None

    def register(self, state, weight):
        if self.failed:
            _ma7_poison(state, 'MA7 queue is poisoned')
        if state._queue not in (None, self):
            _ma7_poison(state, 'MA7 state already belongs to another queue')
        if self.device is None:
            self.device = weight.device
            if weight.is_cuda and state['arm'] == 'F4-wgrad':
                self.main_stream = torch.cuda.current_stream(weight.device)
                self.stream = torch.cuda.Stream(device=weight.device)
        if self.device != weight.device:
            _ma7_poison(state, 'MA7 queue cannot cross devices')
        if all(existing is not state for existing in self.states):
            self.states.append(state)
        state._queue = self

    def poison(self):
        self.failed = True
        for state in self.states:
            state['failed'] = True

    def submit(self, state, input_, full_grad, weight):
        _ma7_bind_weight(state, weight)
        if (self.failed or state['pending_wgrad'] != 0 or
                len(self.pending) >= self.max_pending or
                (state._training and not state._in_microbatch)):
            _ma7_poison(state, 'MA7 queue boundary or bounded capacity violated')
        if input_.device != self.device or full_grad.device != self.device:
            _ma7_poison(state, 'MA7 queue inputs changed device')
        if state._main_grad is not None and weight.grad is not None:
            _ma7_poison(state, 'MA7 deferred weight unexpectedly has a native .grad')
        try:
            with torch.autograd.profiler.record_function('ma7.wgrad'):
                if weight.is_cuda:
                    current = torch.cuda.current_stream(weight.device)
                    if current != self.main_stream:
                        _ma7_poison(state, 'MA7 backward moved to another main stream')
                    # Called AFTER dgrad submission. Only aux waits here; the
                    # main stream never waits for wgrad inside Consumer.backward.
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_stream(current)
                        input_.record_stream(self.stream)
                        full_grad.record_stream(self.stream)
                        weight.record_stream(self.stream)
                        grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
                            input_.reshape(-1, input_.shape[-1]))
                        if state._main_grad is not None:
                            state._main_grad.record_stream(self.stream)
                            state._main_grad.add_(grad_weight)
                        grad_weight.record_stream(self.stream)
                    state['cuda_wgrad_calls'] += 1
                else:
                    if not state._allow_cpu:
                        _ma7_poison(state, 'MA7 CPU execution is test-only')
                    grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
                        input_.reshape(-1, input_.shape[-1]))
                    if state._main_grad is not None:
                        state._main_grad.add_(grad_weight)
                    state['cpu_wgrad_calls'] += 1
            self.pending.append((state, input_, full_grad, weight, grad_weight))
            state['deferred_wgrad_calls'] += 1
            state['pending_wgrad'] += 1
            state['peak_pending_wgrad'] = max(state['peak_pending_wgrad'], state['pending_wgrad'])
            if state._main_grad is not None:
                state['main_grad_accumulations'] += 1
        except BaseException:
            self.poison()
            raise

    def drain(self):
        if self.failed:
            raise RuntimeError('MA7 cannot drain a poisoned queue')
        try:
            if self.pending and self.stream is not None:
                current = torch.cuda.current_stream(self.device)
                if current != self.main_stream:
                    raise RuntimeError('MA7 drain moved to another main stream')
                current.wait_stream(self.stream)
            for state, input_, full_grad, weight, grad_weight in self.pending:
                _ma7_bind_weight(state, weight)
                if state._main_grad is None:
                    # Operator preflight uses ordinary leaf gradients. Its
                    # explicit drain is part of the tested backward boundary.
                    if grad_weight.is_cuda:
                        grad_weight.record_stream(self.main_stream)
                    if weight.grad is None:
                        weight.grad = grad_weight
                    else:
                        weight.grad.add_(grad_weight)
                    state['standalone_grad_assignments'] += 1
                state['drained_wgrad_calls'] += 1
                state['pending_wgrad'] -= 1
            self.pending.clear()
        except BaseException:
            self.poison()
            raise


def drain_backward_state(state):
    """Standalone operator boundary; training uses one shared queue join."""
    _ma7_validate_state(state)
    if state._training or state['backward_calls'] != state['drain_calls'] + 1:
        _ma7_poison(state, 'MA7 standalone drain has no exactly-once backward')
    with torch.autograd.profiler.record_function('ma7.drain'):
        if state._queue is not None:
            state._queue.drain()
        state['drain_calls'] += 1


class ConsumerGemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, residual, op, option, backward_state):
        state = backward_state
        _ma7_validate_state(state, op)
        _ma7_bind_weight(state, weight)
        if state['pending_wgrad']:
            _ma7_poison(state, 'MA7 requires the previous backward drain before another forward')
        if state['arm'] == 'F4-wgrad' and state._queue is None:
            queue = DeferredWgradQueue(max_pending=1)
            queue.register(state, weight)
        ctx.ma7_state = state
        ctx.ma7_backward_done = False
        try:
            output = _MA7ReferenceConsumerGemmRSFunction.forward(
                ctx, input_, weight, bias, residual, op, option)
            state['forward_calls'] += 1
            return output
        except BaseException:
            state['failed'] = True
            if state._queue is not None: state._queue.poison()
            raise

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.ma7_state
        _ma7_validate_state(state)
        if ctx.ma7_backward_done:
            _ma7_poison(state, 'MA7 does not support repeated backward')
        if torch.is_grad_enabled():
            _ma7_poison(state, 'MA7 does not support higher-order backward')
        state['active'] = True
        try:
            if state['arm'] == 'F4':
                result = _MA7ReferenceConsumerGemmRSFunction.backward(ctx, grad_output)
                state['baseline_backward_calls'] += 1
            else:
                input_, weight = ctx.saved_tensors
                grad_bias = grad_output.sum_to_size(ctx.bias_shape)
                grad_residual = grad_output
                full_grad = _gather_along_first_dim(grad_output.contiguous())
                grad_input = full_grad.matmul(weight)
                state._queue.submit(state, input_, full_grad, weight)
                result = (grad_input, None, grad_bias, grad_residual, None, None)
            state['backward_calls'] += 1
            ctx.ma7_backward_done = True
            return (*result, None)
        except BaseException:
            state['failed'] = True
            if state._queue is not None: state._queue.poison()
            raise
        finally:
            state['active'] = False


def consumer_apply(input_, weight, bias, residual, op, option, state):
    return ConsumerGemmRSFunction.apply(input_, weight, bias, residual, op, option, state)


'''


ADAPTER_METHODS = '''
    def install_wgrad_schedule(self, schedules, config, model):
        if getattr(self, '_ma7_schedule_installed', False):
            raise RuntimeError('MA7 schedule cannot be installed twice')
        if (len(model) != 1 or len(self._consumer_contexts) != 64 or
                getattr(config, 'gradient_accumulation_fusion', True) or
                not getattr(config, 'sequence_parallel', False) or
                getattr(config, 'pipeline_model_parallel_size', None) != 1 or
                getattr(config, 'context_parallel_size', None) != 1 or
                parallel_state.get_tensor_model_parallel_world_size() != 8 or
                parallel_state.get_data_parallel_world_size() != 1):
            raise RuntimeError('MA7 requires frozen TP8, DP1/PP1/CP1, non-fused gradient training')
        ddp = getattr(model[0], 'ddp_config', None)
        if (ddp is None or getattr(ddp, 'overlap_grad_reduce', True) or
                getattr(ddp, 'overlap_param_gather', True)):
            raise RuntimeError('MA7 requires the original non-overlap Megatron DDP')
        if getattr(schedules.backward_step, '_ma7_wrapped', False):
            raise RuntimeError('MA7 schedule is already owned by another adapter')
        finalize = config.finalize_model_grads_func
        if not callable(finalize):
            raise RuntimeError('MA7 requires the original finalize_model_grads boundary')
        queue = DeferredWgradQueue(max_pending=64)
        states = []
        for name, consumer in self._consumer_contexts.items():
            state = consumer['backward']
            state._training = True
            weight = self.modules[name].weight
            _ma7_bind_weight(state, weight)
            queue.register(state, weight)
            states.append(state)
        if len({id(state._weight) for state in states}) != 64:
            queue.poison()
            raise RuntimeError('MA7 does not support shared projection weights')
        original = schedules.backward_step
        adapter = self
        def backward_step(*args, **kwargs):
            try:
                if queue.failed or queue.pending or any(s._in_microbatch for s in states):
                    raise RuntimeError('MA7 backward microbatch boundary is unfinished or reentrant')
                before = {id(s): s['backward_calls'] for s in states}
                for state in states:
                    _ma7_validate_state(state)
                    _ma7_bind_weight(state, state._weight)
                    if state['drain_calls'] != state['backward_calls']:
                        raise RuntimeError('MA7 previous microbatch was not drained')
                    state._in_microbatch = True
                result = original(*args, **kwargs)
                with torch.autograd.profiler.record_function('ma7.drain'):
                    if any(s['backward_calls'] != before[id(s)] + 1 for s in states):
                        raise RuntimeError('MA7 requires exactly one backward per target per microbatch')
                    queue.drain()
                    for state in states:
                        state['drain_calls'] += 1
                        state._in_microbatch = False
                return result
            except BaseException:
                queue.poison()
                adapter._consumer_failed = True
                raise
        backward_step._ma7_wrapped = True
        def finalize_checked(*args, **kwargs):
            if (queue.failed or queue.pending or any(s._in_microbatch or s['pending_wgrad'] or
                    s['drain_calls'] != s['backward_calls'] for s in states)):
                queue.poison()
                raise RuntimeError('MA7 finalization crossed an unfinished backward')
            return finalize(*args, **kwargs)
        schedules.backward_step = backward_step
        config.finalize_model_grads_func = finalize_checked
        self._ma7_queue = queue
        self._ma7_schedule_installed = True

    def ma7_backward_state(self):
        if self._consumer_failed or not getattr(self, '_ma7_schedule_installed', False):
            raise RuntimeError('MA7 requires a healthy installed schedule')
        result = {}
        for name, consumer in self._consumer_contexts.items():
            state = consumer['backward']
            _ma7_validate_state(state)
            _ma7_bind_weight(state, self.modules[name].weight)
            if state['pending_wgrad'] or state._in_microbatch:
                raise RuntimeError('MA7 evidence requested before microbatch drain')
            result[name] = dict(state)
        return result
'''


def consumer_autograd_source():
    original = baseline.replace_once(baseline.consumer_autograd_source(),
        'class ConsumerGemmRSFunction(', 'class _MA7ReferenceConsumerGemmRSFunction(')
    identity = ('PROTOCOL = ' + repr(PROTOCOL) + '\nARMS = ' + repr(ARMS) +
                '\nBACKWARD_SCHEDULE = ' + repr(BACKWARD_SCHEDULE) + '\n\n')
    return original + identity + inspect.getsource(BackwardState) + '\n\n' + inspect.getsource(new_backward_state) + '\n' + SUPPORT_SOURCE


def adapter_source(source):
    source = baseline.replace_once(source, baseline.consumer_autograd_source(), consumer_autograd_source())
    source = baseline.replace_once(source,
        "        self._consumer_enabled = os.environ.get('H100_MA4_F_ARM') == 'F4'",
        """        self._ma7_arm = os.environ.get('H100_MA7_ARM')
        if self._ma7_arm not in ARMS:
            raise RuntimeError('MA7 requires an explicit valid experiment arm')
        self._consumer_enabled = True""")
    source = baseline.replace_once(source,
        '                         bias_parameter_id=id(module.bias))',
        "                         bias_parameter_id=id(module.bias),\n"
        "                         backward=new_backward_state(self._ma7_arm))")
    source = baseline.replace_once(source,
        'output = ConsumerGemmRSFunction.apply(input_, module.weight, module.bias, residual, op, adapter.option)',
        "output = consumer_apply(input_, module.weight, module.bias, residual, op, adapter.option, consumer['backward'])")
    return source + '\n' + ADAPTER_METHODS


def taco_source(source):
    return baseline.taco_source(source)
