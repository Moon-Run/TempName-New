"""MA6-v3 receiver experiments share the exact original MA4-F backward.

Only native forward mask representation or warp mapping differs between arms.
Counters retain explicit zero asynchronous-gather coverage so a v2 backward
cannot silently become part of a receiver comparison.
"""
import inspect

import ma4_f_adapter as baseline

PROTOCOL = 'h100-ma6-f4-v3'
BACKWARD_SCHEDULE = 'baseline'
ARMS = ('F4', 'F4-mask', 'F4-layout')


class BackwardState(dict):
    """One module's counters and immutable native projection ownership."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('Unknown MA6 arm: ' + str(arm))
        super().__init__(protocol=PROTOCOL, arm=arm, gather_schedule=BACKWARD_SCHEDULE, forward_calls=0,
                         backward_calls=0, baseline_backward_calls=0,
                         async_gather_calls=0, wait_calls=0,
                         failed=False, active=False)
        self._ma6_arm = arm
        self._ma6_op_id = None


def new_backward_state(arm):
    return BackwardState(arm)


SUPPORT_SOURCE = '''
def _ma6_validate_state(state, op=None):
    expected = {'protocol', 'arm', 'gather_schedule', 'forward_calls', 'backward_calls',
                'baseline_backward_calls', 'async_gather_calls', 'wait_calls',
                'failed', 'active'}
    if (not isinstance(state, dict) or set(state) != expected or
            state['protocol'] != PROTOCOL or state['arm'] not in ARMS or
            not hasattr(state, '_ma6_op_id') or
            getattr(state, '_ma6_arm', None) != state['arm'] or
            state['gather_schedule'] != BACKWARD_SCHEDULE or
            state['async_gather_calls'] != 0 or state['wait_calls'] != 0):
        raise RuntimeError('Invalid MA6 backward state')
    if state['failed'] or state['active']:
        state['failed'] = True
        raise RuntimeError('MA6 cannot retry or reenter a failed backward')
    if op is not None:
        if state._ma6_op_id is None:
            state._ma6_op_id = id(op)
        elif state._ma6_op_id != id(op):
            state['failed'] = True
            raise RuntimeError('MA6 backward state belongs to another projection instance')


class ConsumerGemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, residual, op, option, backward_state):
        _ma6_validate_state(backward_state, op)
        ctx.ma6_state = backward_state
        ctx.ma6_backward_done = False
        try:
            output = _MA6ReferenceConsumerGemmRSFunction.forward(
                ctx, input_, weight, bias, residual, op, option)
            backward_state['forward_calls'] += 1
            return output
        except BaseException:
            backward_state['failed'] = True
            raise

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.ma6_state
        _ma6_validate_state(state)
        if ctx.ma6_backward_done:
            state['failed'] = True
            raise RuntimeError('MA6 does not support repeated backward on a consumer output')
        state['active'] = True
        try:
            # All arms call the same original bias -> gather -> dgrad -> wgrad.
            result = _MA6ReferenceConsumerGemmRSFunction.backward(ctx, grad_output)
            state['baseline_backward_calls'] += 1
            state['backward_calls'] += 1
            ctx.ma6_backward_done = True
            return (*result, None)
        except BaseException:
            state['failed'] = True
            raise
        finally:
            state['active'] = False


'''


ADAPTER_METHODS = '''
    def ma6_backward_state(self):
        if self._consumer_failed:
            raise RuntimeError('MA6 cannot report a failed consumer adapter')
        result = {}
        for name, consumer in self._consumer_contexts.items():
            state = consumer['backward']
            _ma6_validate_state(state)
            if state['arm'] != self._ma6_arm:
                raise RuntimeError('MA6 backward arm identity changed')
            result[name] = dict(state)
        return result
'''


def consumer_autograd_source():
    """Standalone source; callers supply torch and the original TP helper."""
    original = baseline.consumer_autograd_source()
    original = baseline.replace_once(original, 'class ConsumerGemmRSFunction(',
                                     'class _MA6ReferenceConsumerGemmRSFunction(')
    identity = ('PROTOCOL = ' + repr(PROTOCOL) + '\nARMS = ' + repr(ARMS) +
                '\nBACKWARD_SCHEDULE = ' + repr(BACKWARD_SCHEDULE) + '\n\n')
    return (original + identity + inspect.getsource(BackwardState) + '\n\n' +
            inspect.getsource(new_backward_state) + '\n' + SUPPORT_SOURCE)


def adapter_source(source):
    """Apply to the complete already-generated MA4-F adapter, exactly once."""
    source = baseline.replace_once(source, baseline.consumer_autograd_source(),
                                   consumer_autograd_source())
    source = baseline.replace_once(source,
        "        self._consumer_enabled = os.environ.get('H100_MA4_F_ARM') == 'F4'",
        """        self._ma6_arm = os.environ.get('H100_MA6_ARM')
        if self._ma6_arm not in ARMS:
            raise RuntimeError('MA6 requires an explicit valid experiment arm')
        self._consumer_enabled = True""")
    source = baseline.replace_once(source,
        "                         bias_parameter_id=id(module.bias))",
        "                         bias_parameter_id=id(module.bias),\n"
        "                         backward=new_backward_state(self._ma6_arm))")
    source = baseline.replace_once(source,
        'output = ConsumerGemmRSFunction.apply(input_, module.weight, module.bias, residual, op, adapter.option)',
        "output = ConsumerGemmRSFunction.apply(input_, module.weight, module.bias, residual, op, adapter.option, consumer['backward'])")
    return source + '\n' + ADAPTER_METHODS


def taco_source(source):
    """The native consumer interface remains exactly MA4-F's interface."""
    return baseline.taco_source(source)
