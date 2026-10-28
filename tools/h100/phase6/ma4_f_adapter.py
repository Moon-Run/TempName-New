"""Per-layer residual binding and explicit autograd for MA4-F consumer fusion.

The frozen TransformerLayer implementation is called unchanged. Instance wrappers
bind each phase's residual and replace only its already-consumed BDA output.
"""
def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('MA4-F source contract changed: ' + old[:100])
    return source.replace(old, new)


def consumer_autograd_source():
    return '''class ConsumerGemmRSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, residual, op, option):
        if input_.ndim != 3:
            raise RuntimeError('MA4-F requires a three-dimensional projection input')
        expected = (input_.shape[0] // parallel_state.get_tensor_model_parallel_world_size(),
                    input_.shape[1], weight.shape[0])
        if (tuple(residual.shape) != expected or not residual.is_contiguous() or
                residual.dtype != input_.dtype or tuple(bias.shape) != (weight.shape[0],)):
            raise RuntimeError('MA4-F residual shape/dtype/layout differs from the frozen consumer')
        ctx.save_for_backward(input_, weight)
        ctx.bias_shape = bias.shape
        shape = input_.shape
        flat = input_.reshape(-1, shape[-1]).contiguous()
        output = op.forward_consumer(flat, weight, bias, residual.reshape(-1, residual.shape[-1]),
                                     reduce_scatter_option=option)
        if output._base is not None or output.numel() != residual.numel():
            raise RuntimeError('MA4-F native consumer must return an independent complete output')
        # Native output is a fresh allocation. Equal-numel resize changes only
        # its metadata and avoids a new MakeViewlessTensor path after MLP BDA.
        return output.resize_(residual.shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        # The original bias AddBackward unbroadcast precedes linear backward.
        grad_bias = grad_output.sum_to_size(ctx.bias_shape)
        grad_residual = grad_output
        # Preserve the baseline linear backward, including its sequence gather.
        full_grad = _gather_along_first_dim(grad_output.contiguous())
        grad_input = full_grad.matmul(weight)
        grad_weight = full_grad.reshape(-1, full_grad.shape[-1]).t().matmul(
            input_.reshape(-1, input_.shape[-1]))
        return grad_input, grad_weight, grad_bias, grad_residual, None, None


'''


def taco_source(source):
    marker = '\n\n@torch.no_grad()\ndef check_taco_output('
    method = '''
    def forward_consumer(self, input_, weight, bias, residual, reduce_scatter_option=None):
        if not self.owns_output or not self.owns_configuration:
            raise RuntimeError('MA4-F requires the owned CR2 output/configuration')
        return self.codec.forward_consumer(input_, weight, bias, residual)
'''
    source = replace_once(source, marker, '\n' + method + marker)
    reference = source[source.index('@torch.no_grad()\ndef check_taco_output('):]
    reference = replace_once(reference,
        'def check_taco_output(partial, actual, group, mask=None):',
        'def check_consumer_output(partial, actual, group, mask, bias, residual):')
    reference = replace_once(reference, '    denominator = max(bf16.float().norm().item(),1e-20)', '''    # Apply the same two BF16 roundings after the ordered codec reference.
    bf16 = residual + (bf16 + bias)
    coded = residual + (coded + bias)
    denominator = max(bf16.float().norm().item(),1e-20)''')
    return source + '\n\n' + reference


ADAPTER_METHODS = '''
    def _install_consumers(self, model):
        self._consumer_enabled = os.environ.get('H100_MA4_F_ARM') == 'F4'
        self._consumer_failed = False
        self._consumer_contexts = {}
        if not self._consumer_enabled:
            return
        config = model.config
        if (config.hidden_dropout != 0 or config.attention_dropout != 0 or
                config.bias_dropout_fusion or config.recompute_granularity is not None or
                config.enable_cuda_graph or config.external_cuda_graph or
                getattr(config, 'fp32_residual_connection', False) or not model.training):
            raise RuntimeError('MA4-F requires frozen BF16 training, zero dropout, no recompute or CUDA graphs')
        layers = dict(model.named_modules())
        for name, module in self.modules.items():
            if (not module.skip_bias_add or module.bias is None or
                    module.bias.dtype != torch.bfloat16 or not module.bias.is_contiguous() or
                    tuple(module.bias.shape) != (4096,) or
                    not getattr(module.bias, 'sequence_parallel', False)):
                raise RuntimeError('MA4-F requires the original sequence-parallel BF16 projection bias')
            attention = name.endswith('.self_attention.linear_proj')
            suffix = '.self_attention.linear_proj' if attention else '.mlp.linear_fc2'
            layer = layers[name[:-len(suffix)]]
            if (layer.recompute_input_layernorm or layer.recompute_pre_mlp_layernorm or layer.recompute_mlp or
                    layer.hidden_dropout != 0):
                raise RuntimeError('MA4-F does not support layer recomputation or dropout')
            state = dict(residual=None, output=None, residual_version=None, projection_calls=0, bda_calls=0,
                         bias_parameter_id=id(module.bias))
            self._consumer_contexts[name] = state
            phase = '_forward_attention' if attention else '_forward_mlp'
            original = getattr(layer, phase)
            wrapper = self._consumer_phase(original, state, 0 if attention else 1,
                                           'hidden_states' if attention else 'residual')
            setattr(layer, phase, types.MethodType(wrapper, layer))
            setattr(layer, 'self_attn_bda' if attention else 'mlp_bda', self._consumer_bda(state))

    def _consumer_phase(self, original, state, index, keyword):
        adapter = self
        def forward(layer, *args, **kwargs):
            if adapter._consumer_failed or state['residual'] is not None:
                raise RuntimeError('MA4-F consumer phase is failed or reentrant')
            before = state['projection_calls']
            try:
                if not layer.training or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
                    raise RuntimeError('MA4-F requires the frozen grad-enabled training phase')
                residual = args[index] if len(args) > index else kwargs[keyword]
                state['residual_version'] = residual._version
                state['residual'] = residual
                result = original(*args, **kwargs)
                if (state['projection_calls'] != before + 1 or state['bda_calls'] != state['projection_calls'] or
                        state['output'] is not None):
                    raise RuntimeError('MA4-F phase did not consume exactly one fused projection')
                return result
            except BaseException:
                adapter._consumer_failed = True
                raise
            finally:
                state['residual'] = None
                state['residual_version'] = None
                state['output'] = None
        return forward

    def _consumer_bda(self, state):
        def consume(output_with_bias, residual, prob):
            output, bias = output_with_bias
            if (prob != 0 or bias is not None or residual is not state['residual'] or
                    residual._version != state['residual_version'] or
                    output is not state['output'] or output is None):
                raise RuntimeError('MA4-F BDA ownership, residual identity or dropout changed')
            state['output'] = None
            state['bda_calls'] += 1
            return output
        def factory(training, fused):
            if not training or fused:
                raise RuntimeError('MA4-F requires the frozen unfused training BDA interface')
            return consume
        return factory

    def consumer_state(self):
        if not self._consumer_enabled:
            return {}
        if self._consumer_failed:
            raise RuntimeError('MA4-F consumer evidence requested after a failed forward')
        result = {}
        for name, state in self._consumer_contexts.items():
            if state['residual'] is not None or state['output'] is not None:
                raise RuntimeError('MA4-F retains an unfinished consumer phase')
            result[name] = dict(projection_calls=state['projection_calls'], bda_calls=state['bda_calls'],
                bias_parameter_id=state['bias_parameter_id'], native=self.scope_ops[name].codec.ma4_consumer_state())
        return result
'''


def adapter_source(source):
    source = replace_once(source, 'from taco_support import TacoForward, check_taco_output',
                          'from taco_support import TacoForward, check_taco_output, check_consumer_output')
    source = replace_once(source, 'class Adapter:', consumer_autograd_source() + 'class Adapter:')
    source = replace_once(source, '        assert len(self.modules) == 2 * model.config.num_layers, list(self.modules)',
                          '        assert len(self.modules) == 2 * model.config.num_layers, list(self.modules)\n'
                          '        self._install_consumers(model)')
    source = replace_once(source, '            output = GemmRSFunction.apply(input_, module.weight, op, adapter.option)', '''            consumer = adapter._consumer_contexts.get(name)
            if consumer is None:
                output = GemmRSFunction.apply(input_, module.weight, op, adapter.option)
            else:
                residual = consumer['residual']
                if (residual is None or consumer['output'] is not None or
                        id(module.bias) != consumer['bias_parameter_id']):
                    raise RuntimeError('MA4-F projection has no matching residual phase or bias parameter')
                output = ConsumerGemmRSFunction.apply(input_, module.weight, module.bias, residual, op, adapter.option)
                consumer['projection_calls'] += 1
                consumer['output'] = output''')
    source = replace_once(source, '''                        adapter.forward_checks[name] = check_taco_output(partial, actual,
                            parallel_state.get_tensor_model_parallel_group(), getattr(op.codec,'selection_cpu',None))''', '''                        if consumer is None:
                            adapter.forward_checks[name] = check_taco_output(partial, actual,
                                parallel_state.get_tensor_model_parallel_group(), getattr(op.codec,'selection_cpu',None))
                        else:
                            adapter.forward_checks[name] = check_consumer_output(partial, actual,
                                parallel_state.get_tensor_model_parallel_group(), getattr(op.codec,'selection_cpu',None),
                                module.bias, residual.reshape_as(actual))''')
    source = replace_once(source, '            if module.skip_bias_add:\n                return output, module.bias',
                          '            if consumer is not None:\n                return output, None\n'
                          '            if module.skip_bias_add:\n                return output, module.bias')
    return source + '\n' + ADAPTER_METHODS
