"""CPU checks of the actual generated training/build boundary, never GPU evidence."""
import ast
import hashlib
import os
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import open_compress_build as build
import open_compress_sources as sources


class Workers(unittest.TestCase):
    def test_generated_adapter_imports_and_initializes_all_three_policies_without_p2p_api(self):
        torch = ModuleType('torch')
        torch.bfloat16, torch.float32 = object(), object()
        torch.autograd = SimpleNamespace(Function=object)
        torch.no_grad = lambda: lambda fn: fn
        torch.distributed = ModuleType('torch.distributed')
        core = ModuleType('megatron.core')
        core.parallel_state = SimpleNamespace(get_tensor_model_parallel_group=lambda: 'TP8')
        layers = ModuleType('megatron.core.tensor_parallel.layers')
        class Row:
            input_is_parallel = sequence_parallel = True
            explicit_expert_comm = gradient_accumulation_fusion = False
            config = SimpleNamespace(_cpu_offloading_context=None)
            weight = SimpleNamespace(dtype=torch.bfloat16)
            def forward(self, value): return value
        layers.RowParallelLinear = Row
        mappings = ModuleType('megatron.core.tensor_parallel.mappings')
        mappings._gather_along_first_dim = lambda value: value
        codec = ModuleType('megatron_taco')
        codec.TensorTacoRS = codec.QuantizedRS = object
        routing = ModuleType('routing')
        exec(sources.routing_source(), routing.__dict__)
        support = ModuleType('taco_support')
        # Deliberately no GemmRS, ReduceScatterOption, RingMode or init_flux_shm.
        flux = ModuleType('flux')
        modules = {'torch': torch, 'torch.distributed': torch.distributed, 'megatron.core': core,
            'megatron.core.tensor_parallel.layers': layers, 'megatron.core.tensor_parallel.mappings': mappings,
            'megatron_taco': codec, 'routing': routing, 'taco_support': support, 'flux': flux}
        with patch.dict(sys.modules, modules):
            exec(sources.taco_source('flux-int8'), support.__dict__)
            for arm in sources.ARMS:
                module = ModuleType('generated_adapter')
                exec(sources.adapter_source(arm), module.__dict__)
                rows = [(f'decoder.layers.{i}{suffix}', Row()) for i in range(32)
                        for suffix in ('.self_attention.linear_proj', '.mlp.linear_fc2')]
                model = SimpleNamespace(named_modules=lambda: iter(rows), config=SimpleNamespace(num_layers=32))
                adapter = module.Adapter(model, arm, list(sources.SHAPES.values()))
                self.assertEqual(len(adapter.modules), 64)
                self.assertIsNone(adapter.option)

    def test_generated_worker_initialization_with_launcher_environment(self):
        import open_compress as controller
        torch = ModuleType('torch')
        torch.__version__ = 'cpu-fixture'
        torch.version = SimpleNamespace(cuda='fixture')
        torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()))
        torch.cuda = SimpleNamespace(set_device=lambda rank: None, device_count=lambda: 8,
            get_device_properties=lambda rank: SimpleNamespace(uuid=f'fixture-{rank}'),
            nccl=SimpleNamespace(version=lambda: (2, 27, 3)))
        dist = ModuleType('torch.distributed')
        dist.get_rank = lambda: 0
        torch.distributed = dist
        model = SimpleNamespace(train=lambda: None)
        optimizer = SimpleNamespace(scale_loss=lambda value: value)
        modules = {'torch': torch, 'torch.distributed': dist}
        def module(name, **attributes):
            value = ModuleType(name)
            value.__dict__.update(attributes)
            modules[name] = value
            return value
        module('communication', verify=lambda value: value)
        module('runtime', check_topology=lambda world: dict(world=world))
        module('megatron.training.initialize', initialize_megatron=lambda: None)
        training = module('megatron.training', get_timers=lambda: None)
        module('megatron.training.training', setup_model_and_optimizer=lambda *args: ([model], optimizer, None))
        module('megatron.core.enums', ModelType=SimpleNamespace(encoder_or_decoder='fixture'))
        module('megatron.core', parallel_state=SimpleNamespace())
        module('megatron.core.pipeline_parallel.schedules', get_forward_backward_func=lambda: None)
        module('megatron.core.distributed.finalize_model_grads', finalize_model_grads=lambda: None)
        module('megatron.core.utils', get_model_config=lambda value: SimpleNamespace())
        module('megatron.training.utils', unwrap_model=lambda value: value)
        module('megatron.core.tensor_parallel.random', get_cuda_rng_tracker=lambda: None)
        module('pretrain_gpt', model_provider=lambda: None, loss_func=lambda: None)
        module('adapter', Adapter=lambda *args: SimpleNamespace(arguments=args))
        module('megatron_taco')
        routing = module('routing')
        exec(sources.routing_source(), routing.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(python=Path(sys.executable), cuda_home=root/'cuda')
            communication = dict(environment={'NCCL_DEBUG': 'WARN', 'NCCL_NVLS_ENABLE': '0'},
                                 config_files={}, nvls_request='0')
            plan = dict(cuda_visible_devices='0,1,2,3,4,5,6,7', run_id='fixture', communication=communication)
            for arm in sources.ARMS:
                scripts = root/arm/'scripts'
                scripts.mkdir(parents=True)
                cfg = sources.configuration(arm, sources.selection_metadata(), communication)
                build.write(scripts/'config.json', cfg)
                native = scripts/'build'/arm
                native.mkdir(parents=True)
                for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                    (native/name).write_bytes(b'CPU fixture only')
                original_read = Path.read_text
                def read(path, *args, **kwargs):
                    if str(path) == '/proc/self/maps':
                        return '\n'.join('fake mapping '+str(native/name) for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'))
                    return original_read(path, *args, **kwargs)
                frozen = dict(cfg['common'], **cfg['cases'][0])
                kwargs = {key: frozen[value] for key, value in [('num_layers','layers'),('hidden_size','hidden'),
                    ('ffn_hidden_size','ffn'),('num_attention_heads','heads'),('seq_length','sequence'),
                    ('micro_batch_size','micro_batch'),('global_batch_size','global_batch')]}
                training.get_args = lambda: SimpleNamespace(**kwargs, tensor_model_parallel_size=8,
                    sequence_parallel=True, pipeline_model_parallel_size=1, context_parallel_size=1,
                    data_parallel_size=1, bf16=True, gradient_accumulation_fusion=False,
                    overlap_grad_reduce=False, overlap_param_gather=False, padded_vocab_size=128)
                job = dict(arm=arm,mode='smoke',trial=0,block=-1,index=0)
                env = controller.environment(args,root,scripts,job,root/'result',plan)
                env.update(LOCAL_RANK='0',WORLD_SIZE='8')
                source = sources.worker_source(arm)
                prefix = source[:source.index('\ndef fingerprint(')]
                namespace = {'__file__': str(scripts/'worker.py')}
                with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True), \
                     patch.object(Path, 'read_text', read):
                    exec(compile(prefix, str(scripts/'worker.py'), 'exec'), namespace)
                self.assertEqual(namespace['POLICY'], arm)
                self.assertEqual(namespace['CASE']['name'], env['E2E_CASE'])
                self.assertEqual(namespace['report']['rank'], 0)

    def test_training_ast_preserved_and_old_codec_checks_removed(self):
        for arm in sources.ARMS:
            worker = sources.worker_source(arm)
            sources.base.validate_worker(worker)
            for source in (worker, sources.adapter_source(arm), sources.taco_source(arm), sources.codec_source(arm)):
                compile(source, arm, 'exec')
            self.assertNotIn('from basic_quant_validation', worker)
            self.assertNotIn('op.codec.transport_state()', worker)
            self.assertIn("report['compression_states']", worker)
            self.assertIn("report['open_compress_implementation']", worker)

    def test_routing_covers_all_modules_without_native_fallback_or_selection(self):
        module = {}
        exec(sources.routing_source(), module)
        observed = {f'decoder.layers.{layer}{suffix}': dict(M=2048, N=4096, K_global=k, K_local=k//8)
            for layer in range(32) for suffix, k in (('.self_attention.linear_proj', 4096), ('.mlp.linear_fc2', 16384))}
        for arm in sources.ARMS:
            self.assertEqual(set(module['module_routes'](observed, arm).values()), {'flux_tensor'})
            self.assertFalse(module['is_native'](arm))
            self.assertFalse(module['is_fused_v2'](arm))
            self.assertTrue(module['is_tensor'](arm))
            self.assertFalse(sources.implementation(arm)['arrival'])
            self.assertFalse(sources.implementation(arm)['selective'])
            with self.assertRaises(ValueError): module['selection_entry']()
        with self.assertRaises(ValueError): module['module_routes']({}, sources.ARMS[0])
        with self.assertRaises(ValueError): module['definition']('flux-basic')

    def test_oracle_is_new_algorithm_and_matches_fixed_source_order(self):
        source = sources.taco_source('flux-int8')
        self.assertIn('from megatron_taco import reference_roundtrip', source)
        self.assertIn('for source in range(world):', source)
        self.assertIn('dist.reduce_scatter_tensor(bf16', source)
        self.assertNotIn('from reference import roundtrip', source)
        self.assertNotIn('masked_roundtrip', source)
        self.assertNotIn('gemm_rs_taco', source)
        tree = ast.parse(source)
        check = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_taco_output')
        self.assertTrue(any(isinstance(n, ast.Raise) for n in ast.walk(check)))

    def test_codec_identity_is_exact_for_both_implementations(self):
        for arm in sources.ARMS:
            identity = sources.codec_identity(arm)
            self.assertEqual(identity['source_sha256'], hashlib.sha256(sources.codec_source(arm).encode()).hexdigest())
            self.assertEqual(identity['reference_sha256'], hashlib.sha256(sources.reference_source(arm).encode()).hexdigest())
        self.assertNotIn('reference_roundtrip', sources.reference_source('flux-bf16'))

    def test_worker_case_name_matches_launcher(self):
        for arm in sources.ARMS:
            config = sources.configuration(arm, sources.selection_metadata(), {})
            self.assertEqual([case['name'] for case in config['cases']], ['rtx5090-open-compress'])


class Build(unittest.TestCase):
    def test_dirty_dependency_is_rejected_even_with_identical_head(self):
        with patch.object(build.subprocess, 'check_output', side_effect=['a'*40+'\n', '', 'b'*40+'\n', '']):
            self.assertEqual(build.dependency_identity(), {
                'cutlass': dict(commit='a'*40, tracked_clean=True),
                'nccl': dict(commit='b'*40, tracked_clean=True)})
        with patch.object(build.subprocess, 'check_output', side_effect=['a'*40+'\n', ' M include/cutlass/cutlass.h\n']):
            with self.assertRaisesRegex(ValueError, 'Tracked dependency changes'):
                build.dependency_identity()

    def test_only_comm_none_is_registered_and_sorted_changes_real_kernel(self):
        overrides = build.source_overrides(build.REPO)
        dirs = [line.strip() for line in overrides['src/CMakeLists.txt'].splitlines()
                if line.strip().startswith('add_subdirectory(')]
        self.assertEqual(dirs, ['add_subdirectory(comm_none)', 'add_subdirectory(cuda)', 'add_subdirectory(ths_op)'])
        self.assertIn('_Sm89{}', overrides['src/generator/gen_comm_none.cc'])
        self.assertIn('_RTX5090{}', overrides['src/generator/gen_comm_none.cc'])
        self.assertNotIn('gemm_rs', overrides['src/CMakeLists.txt'])
        sorted_sources = build.op1_a_swizzle.overrides(build.REPO)
        self.assertIn('Op1AInterleavedSwizzle', sorted_sources['src/comm_none/gemm_v2_comm_none.hpp'])
        self.assertNotEqual(build.native_root('/tmp/example', sources.ARMS[0]),
                            build.native_root('/tmp/example', sources.ARMS[2]))
        self.assertEqual(build.native_root('/tmp/example', sources.ARMS[0]),
                         build.native_root('/tmp/example', sources.ARMS[1]))

    def test_verifier_rejects_wrong_protocol_before_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            build.write(path/build.MANIFEST, dict(protocol='old-taco', actual_arch='sm120', arms=list(sources.ARMS)))
            with self.assertRaises(ValueError): build.verify(path)


if __name__ == '__main__':
    unittest.main()
