"""Execute the native-boundary adapter against independent CPU call sentinels."""
from contextlib import contextmanager, nullcontext
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import open_compress_sources as sources


class Tensor:
    dtype = 'bf16'
    requires_grad = True
    def __init__(self, shape, tag=None):
        self.shape, self.tag = tuple(shape), object() if tag is None else tag
    def reshape(self, *shape):
        if -1 in shape:
            known = math.prod(d for d in shape if d != -1)
            shape = tuple(math.prod(self.shape)//known if d == -1 else d for d in shape)
        return Tensor(shape,self.tag)
    def data_ptr(self):
        return id(self.tag)
    def stride(self):
        return tuple(math.prod(self.shape[i+1:]) for i in range(len(self.shape)))


@contextmanager
def runtime_fixture():
    calls, group = [], object()
    torch = ModuleType('torch')
    class Function:
        @classmethod
        def apply(cls,*args):
            ctx = SimpleNamespace()
            result = cls.forward(ctx,*args)
            calls.append(('autograd',cls,ctx))
            return result
    torch.autograd = SimpleNamespace(Function=Function)
    torch.bfloat16 = 'bf16'
    torch.nn = SimpleNamespace(Parameter=lambda value,requires_grad=True:value)
    torch.profiler = SimpleNamespace(record_function=lambda name:nullcontext())
    dist = ModuleType('torch.distributed')
    dist.get_world_size,dist.get_rank,dist.get_backend = lambda g:8, lambda g:0, lambda g:'nccl'
    torch.distributed = dist
    core = ModuleType('megatron.core')
    core.parallel_state = SimpleNamespace(get_tensor_model_parallel_group=lambda:group)
    layers = ModuleType('megatron.core.tensor_parallel.layers')
    mappings = ModuleType('megatron.core.tensor_parallel.mappings')
    package = ModuleType('megatron.core.tensor_parallel')
    package.layers,package.mappings = layers,mappings
    def original_linear(**kwargs):
        calls.append(('linear',kwargs))
        return Tensor((2048,1,4096))
    def original_reduce(input_, group=None, input_split_sizes=None, use_global_buffer=False):
        calls.append(('reduce',input_,group,input_split_sizes,use_global_buffer))
        return Tensor((256,1,4096))
    def original_gather(input_,group=None,input_split_sizes=None,use_global_buffer=False):
        calls.append(('gather',input_,group,input_split_sizes,use_global_buffer))
        return Tensor((2048,1,4096))
    maps = dict(Function=Function,_reduce_scatter_along_first_dim=original_reduce,
                _gather_along_first_dim=original_gather)
    exec('''class OriginalReduceScatter(Function):
    @staticmethod
    def forward(ctx,input_,group=None,input_split_sizes=None,use_global_buffer=False):
        ctx.group,ctx.input_split_sizes,ctx.use_global_buffer=group,input_split_sizes,use_global_buffer
        return _reduce_scatter_along_first_dim(input_,group,input_split_sizes,use_global_buffer)
    @staticmethod
    def backward(ctx,gradient):
        return _gather_along_first_dim(gradient,ctx.group,ctx.input_split_sizes,ctx.use_global_buffer),None,None,None
''',maps)
    native_rs = maps['OriginalReduceScatter']
    def original_rs(value):
        return native_rs.apply(value,None,None,False)
    def original_backward(ctx,gradient):
        calls.append(('linear-backward',ctx,gradient))
    namespace = dict(linear_with_grad_accumulation_and_async_allreduce=original_linear,
                     reduce_scatter_to_sequence_parallel_region=original_rs)
    exec('''class Row:
    def forward(self,input_):
        self._forward_impl=linear_with_grad_accumulation_and_async_allreduce
        partial=self._forward_impl(input=input_,weight=self.weight,bias=None,
            gradient_accumulation_fusion=False,allreduce_dgrad=False,sequence_parallel=False,grad_output_buffer=None)
        output=reduce_scatter_to_sequence_parallel_region(partial)
        return output,self.bias
''',namespace)
    layers.RowParallelLinear = namespace['Row']
    layers.linear_with_grad_accumulation_and_async_allreduce = original_linear
    layers.LinearWithGradAccumulationAndAsyncCommunication = SimpleNamespace(backward=original_backward)
    mappings._ReduceScatterToSequenceParallelRegion = native_rs
    codec = ModuleType('megatron_taco')
    class Codec:
        def __init__(self,g,m,n):
            self.group,self.calls,self.profile=g,0,False
            self.wire_bytes_per_rank,self.bf16_wire_bytes_per_rank=7569408,14680064
        def reduce(self,value):
            calls.append(('int8',value))
            self.calls+=1
            return Tensor((256,4096))
    codec.TensorTacoRS=Codec
    modules={'torch':torch,'torch.distributed':dist,'megatron.core':core,
        'megatron.core.tensor_parallel':package,'megatron.core.tensor_parallel.layers':layers,
        'megatron.core.tensor_parallel.mappings':mappings,'megatron_taco':codec}
    with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules):
        for name,module in (('layers',layers),('mappings',mappings)):
            path=Path(directory)/(name+'.py');path.write_text('# independent CPU fixture\n')
            module.__file__=str(path)
        path=Path(__file__).with_name('open_compress_megatron.py')
        spec=importlib.util.spec_from_file_location('cpu_native_projection_fixture',path)
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        def row():
            module=layers.RowParallelLinear()
            module.input_is_parallel=module.sequence_parallel=True
            module.explicit_expert_comm=module.gradient_accumulation_fusion=False
            module.config=SimpleNamespace(_cpu_offloading_context=None)
            module.weight=Tensor((4096,512));module.bias=None
            module.output_size,module.input_size,module.input_size_per_partition=4096,4096,512
            return module
        yield SimpleNamespace(runtime=runtime,calls=calls,group=group,layers=layers,mappings=mappings,
                              row=row,original_linear=original_linear,original_reduce=original_reduce,namespace=namespace)


class NativeBoundary(unittest.TestCase):
    def test_bf16_executes_original_linear_and_reduce_without_codec(self):
        with runtime_fixture() as f:
            module=f.row()
            op=f.runtime.MegatronProjection(f.group,2048,4096,512,'megatron-bf16',module,True)
            output,bias=op.forward_module(Tensor((2048,1,512)))
            self.assertEqual(output.shape,(256,1,4096))
            self.assertIsNone(bias)
            self.assertEqual([call[0] for call in f.calls],['linear','reduce','autograd'])
            self.assertEqual(op.tensor_codec.state()['encode_calls'],0)
            self.assertEqual(op.tensor_codec.calls,1)
            self.assertIs(op.native_forward.__func__.__code__,f.layers.RowParallelLinear.forward.__code__)
            self.assertIs(f.namespace['linear_with_grad_accumulation_and_async_allreduce'],f.original_linear)
            self.assertEqual(op.partial.shape,(2048,4096))
            self.assertTrue(all(op.state()[k] for k in ('native_forward_preserved','native_linear_backward_preserved',
                                                       'native_rs_backward_preserved','original_rs_forward')))

    def test_int8_replaces_only_rs_forward_and_reuses_exact_upstream_backward(self):
        with runtime_fixture() as f:
            original_globals=dict(f.mappings._ReduceScatterToSequenceParallelRegion.forward.__globals__)
            op=f.runtime.MegatronProjection(f.group,2048,4096,512,'megatron-int8',f.row(),True)
            op.forward_module(Tensor((2048,1,512)))
            self.assertEqual([call[0] for call in f.calls],['linear','int8','autograd'])
            self.assertIs(op._compressed_class.backward,f.mappings._ReduceScatterToSequenceParallelRegion.backward)
            self.assertEqual(f.mappings._ReduceScatterToSequenceParallelRegion.forward.__globals__,original_globals)
            _,cls,ctx=f.calls[-1]
            gradient=Tensor((256,1,4096))
            cls.backward(ctx,gradient)
            self.assertEqual(f.calls[-1],('gather',gradient,f.group,None,False))
            self.assertFalse(op.state()['original_rs_forward'])
            self.assertTrue(op.state()['native_linear_backward_preserved'])

    def test_private_instances_do_not_cross_route_and_reject_changed_rs_policy(self):
        with runtime_fixture() as f:
            first=f.runtime.MegatronProjection(f.group,2048,4096,512,'megatron-bf16',f.row())
            second=f.runtime.MegatronProjection(f.group,2048,4096,512,'megatron-int8',f.row())
            first.forward_module(Tensor((2048,1,512)))
            second.forward_module(Tensor((2048,1,512)))
            first.forward_module(Tensor((2048,1,512)))
            self.assertEqual([c[0] for c in f.calls if c[0] in ('reduce','int8')],['reduce','int8','reduce'])
            self.assertIsNone(first.partial)
            with self.assertRaises(ValueError): second._compressed_reduce(Tensor((2048,1,4096)),input_split_sizes=[256]*8)
            with self.assertRaises(ValueError): second._compressed_reduce(Tensor((2048,1,4096)),use_global_buffer=True)
            with self.assertRaises(ValueError): second._compressed_reduce(Tensor((2048,1,4096)),group=object())

    def test_generated_workers_record_real_native_identity_and_never_import_flux(self):
        for arm in sources.MEGATRON_ARMS:
            worker=sources.worker_source(arm)
            self.assertIn("report['native_execution']",worker)
            self.assertIn("report['flux_library_loaded']=False",worker)
            self.assertEqual(sources.adapter_source(arm),'from open_compress_megatron import Adapter\n')
        runtime=Path(__file__).with_name('open_compress_megatron.py').read_text()
        self.assertNotIn('import flux',runtime)

    def test_native_adapter_covers_64_modules_and_retains_native_forward_after_audit(self):
        routing=ModuleType('routing')
        exec(sources.routing_source(),routing.__dict__)
        for arm in sources.MEGATRON_ARMS:
            with self.subTest(arm=arm),runtime_fixture() as f,patch.dict(sys.modules,{'routing':routing}), \
                 patch.dict(os.environ,{'E2E_MODE':'timing'}):
                rows=[(f'decoder.layers.{i}{suffix}',f.row()) for i in range(32)
                      for suffix in ('.self_attention.linear_proj','.mlp.linear_fc2')]
                for name,module in rows:
                    if name.endswith('.mlp.linear_fc2'):
                        module.input_size,module.input_size_per_partition=16384,2048
                        module.weight=Tensor((4096,2048))
                model=SimpleNamespace(named_modules=lambda:iter(rows),config=SimpleNamespace(num_layers=32))
                adapter=f.runtime.Adapter(model,arm,list(sources.SHAPES.values()))
                for name,module in rows:
                    module.forward(Tensor((2048,1,module.input_size_per_partition)))
                self.assertEqual(len(adapter.scope_ops),64)
                self.assertTrue(all(value == 1 for value in adapter.codec_calls().values()))
                adapter.stop_audit()
                for name,module in rows:
                    module.forward(Tensor((2048,1,module.input_size_per_partition)))
                self.assertTrue(all(adapter.forward_preserved.values()))
                self.assertEqual(adapter.calls,dict.fromkeys(adapter.modules,1))
                self.assertEqual(adapter.timed_calls,dict.fromkeys(adapter.modules,1))
                self.assertEqual(adapter.codec_calls(),dict.fromkeys(adapter.modules,2))
                self.assertEqual(adapter.post_audit_routes,adapter.actual_routes)


if __name__ == '__main__':
    unittest.main()
