"""CPU oracle, native ABI/lifecycle and ordering checks; no GPU performance claim."""
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import ma17_adapter as ma17
import test_ma8_adapter as ma8_tests

LAYERS = Path(__file__).resolve().parents[4] / 'Megatron-LM/megatron/core/tensor_parallel/layers.py'


def original_source():
    text = LAYERS.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                and n.name == 'LinearWithGradAccumulationAndAsyncCommunication')
    return ma17.cache.normalized(ast.get_source_segment(text, node))


def candidate_source():
    return ma17.direct_allgather_class_source(original_source())


def valid_state(steps=3, mode='smoke', rank=4):
    return dict(protocol=ma17.PROTOCOL, installed=True,
        local_shape=list(ma17.LOCAL_SHAPE), world_size=8,
        source_sha256=ma17.CLASS_SHA256, backward_sha256=ma17.BACKWARD_SHA256,
        calls=steps * 256, failed=False, active=False,
        native=dict(library_path='/example/libnccl.so.2', library_sha256='a' * 64,
                    version=22105, torch_version=[2, 21, 5], group_ranks=list(range(8)),
                    rank=rank, device=rank, stream=19, calls=steps * 256,
                    initializations=1, async_checks=1, async_error=0,
                    failed=False, closed=False, datatype=9,
                    device_memory_before=[80000, 100000], device_memory_after=[70000, 100000],
                    device_memory_after_work=[60000, 100000],
                    linkage_verified=True, torch_cuda_library_path='/example/libtorch_cuda.so',
                    linkage_symbols=['ncclAllGather', 'ncclGetVersion'],
                    linkage_symbol_addresses={name: dict(direct=1234, torch_dependency=1234)
                                              for name in ('ncclAllGather', 'ncclGetVersion')}),
        modules={name: dict(role=role, rank=rank, calls=steps * 4,
                           audits=steps * 4 if mode == 'smoke' else 0)
                 for name, role in ma17.module_roles().items()})


class SourceTests(unittest.TestCase):
    def test_only_class_name_and_gather_submission_change(self):
        cached = ma17.cache.cached_class_source(original_source())
        candidate = candidate_source()
        restored = candidate.replace('_MA17DirectAllGatherLinear', '_MA8CachedRawLinear').replace(
            '_ma17_gather(', 'dist_all_gather_func(')
        self.assertEqual(cached, restored)
        self.assertEqual(hashlib.sha256(candidate.encode()).hexdigest(), ma17.CLASS_SHA256)
        self.assertEqual(ma17.backward_sha256(candidate), ma17.BACKWARD_SHA256)
        self.assertEqual(ma17.backward_sha256(candidate), ma17.backward_sha256(cached))

    def test_drift_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'pinned'):
            ma17.direct_allgather_class_source(original_source().replace('grad_input =', 'grad_input  ='))

    def test_real_autograd_all_ranks_both_bias_modes_and_retained_storage(self):
        torch.set_num_threads(1)
        for rank in range(8):
            for bias in (False, True):
                results, saved = [], []
                for candidate in (False, True):
                    events = []

                    def gather(output, input, group=None, async_op=False):
                        self.assertFalse(async_op)
                        events.append('gather')
                        output.copy_(torch.cat([input + peer - rank for peer in range(8)]))

                    def scatter(output, input, group=None, async_op=False):
                        self.assertTrue(async_op)
                        events.append('scatter')
                        rows = output.shape[0]
                        output.copy_(input[rank * rows:(rank + 1) * rows] * 8)
                        return types.SimpleNamespace(wait=lambda: events.append('wait'))

                    class Proxy:
                        cuda = types.SimpleNamespace(current_device=lambda: 'cpu')
                        def __getattr__(self, name):
                            return getattr(torch, name)

                    namespace = dict(torch=Proxy(), custom_fwd=lambda f: f, custom_bwd=lambda f: f,
                        get_tensor_model_parallel_world_size=lambda: 8,
                        get_tensor_model_parallel_group=lambda: None,
                        dist_all_gather_func=gather, _ma17_gather=gather,
                        dist_reduce_scatter_func=scatter,
                        prepare_input_tensors_for_wgrad_compute=lambda grad, inp:
                            (grad.contiguous().view(-1, grad.shape[-1]), inp.view(-1, inp.shape[-1])))
                    text = candidate_source() if candidate else ma17.cache.cached_class_source(original_source())
                    exec(compile(text, '<ma17-cpu-autograd>', 'exec'), namespace)
                    cls = namespace['_MA17DirectAllGatherLinear' if candidate else '_MA8CachedRawLinear']
                    value = (torch.arange(16).reshape(2, 1, 8).float() / 8 + rank).requires_grad_()
                    weight = (torch.arange(24).reshape(3, 8).float() / 8 - 2).requires_grad_()
                    offset = torch.tensor([.5, -1., 2.], requires_grad=True) if bias else None
                    output = cls.apply(value, weight, offset, False, False, True, None, None)
                    retained = output.grad_fn.saved_tensors[2]
                    saved.append(retained)
                    snapshot = retained.clone()
                    grad = torch.arange(output.numel()).reshape_as(output).float() / 8 - 1
                    output.backward(grad)
                    self.assertEqual(events, ['gather', 'scatter', 'wait'])
                    self.assertTrue(torch.equal(retained, snapshot))
                    results.append([output.detach(), value.grad, weight.grad] + ([offset.grad] if bias else []))
                self.assertNotEqual(saved[0].data_ptr(), saved[1].data_ptr())
                for left, right in zip(*results):
                    self.assertTrue(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


class FakeTensor:
    is_cuda = True
    device = torch.device('cuda', 4)
    dtype = torch.bfloat16
    def __init__(self, tensor):
        self.tensor, self.shape = tensor, tensor.shape
    def is_contiguous(self):
        return self.tensor.is_contiguous()
    def data_ptr(self):
        return self.tensor.data_ptr()
    def view(self, dtype):
        return self.tensor.view(dtype)


class FakeNative:
    device, stream = 4, 19
    def __init__(self, *args):
        self.calls, self.closed = 0, False
    def launch(self, output, input):
        output.tensor.copy_(torch.cat([input.tensor + peer for peer in range(8)]))
        self.calls += 1
    def state(self):
        return dict(valid_state()['native'], calls=self.calls)
    def close(self):
        self.closed = True


class TrackerTests(unittest.TestCase):
    def fixture(self, mode='timing', corrupt=False):
        markers, audits = [], []

        class Proxy:
            bfloat16, uint8 = torch.bfloat16, torch.uint8
            cuda = types.SimpleNamespace(current_device=lambda: 4,
                current_stream=lambda: types.SimpleNamespace(cuda_stream=19),
                is_current_stream_capturing=lambda: False)
            autograd = types.SimpleNamespace(profiler=types.SimpleNamespace(
                record_function=lambda name: Marker(name)))
            @staticmethod
            def empty_like(value):
                return FakeTensor(torch.empty_like(value.tensor))
            equal = staticmethod(torch.equal)

        class Marker:
            def __init__(self, name): self.name = name
            def __enter__(self): markers.append(self.name)
            def __exit__(self, *args): return False

        def oracle(output, input, group=None):
            audits.append(True)
            output.tensor.copy_(torch.cat([input.tensor + peer + int(corrupt) for peer in range(8)]))

        group = object()
        tracker = ma17.Tracker(Proxy(), group, 4, mode, FakeNative(), oracle, candidate_source())
        value = FakeTensor(torch.arange(16).reshape(2, 1, 8).to(torch.bfloat16))
        output = FakeTensor(torch.empty(16, 1, 8, dtype=torch.bfloat16))
        return tracker, value, output, markers, audits

    def test_complete_module_schedule_native_counts_and_profile_markers(self):
        tracker, value, output, markers, audits = self.fixture('profile')
        with patch.object(ma17, 'LOCAL_SHAPE', (2, 1, 8)):
            for _ in range(2):
                for name in tracker.names:
                    tracker.bound_gather(name)(output, value, tracker.group)
        self.assertEqual(tracker.calls, 128)
        self.assertEqual(tracker.native.calls, 128)
        self.assertEqual(audits, [])
        self.assertEqual(markers, ['ma17.column.gather.qkv', 'ma17.column.gather.fc1'] * 64)
        self.assertTrue(all(row['calls'] == 2 for row in tracker.modules.values()))

    def test_smoke_checks_all_bytes_and_timing_has_no_oracle_or_marker(self):
        for mode in ('smoke', 'timing'):
            tracker, value, output, markers, audits = self.fixture(mode)
            with patch.object(ma17, 'LOCAL_SHAPE', (2, 1, 8)):
                tracker.bound_gather(tracker.names[0])(output, value, tracker.group)
            self.assertEqual(len(audits), int(mode == 'smoke'))
            self.assertEqual(markers, [])

    def test_corruption_schedule_reentry_device_stream_layout_and_group_fail_closed(self):
        mutations = [lambda t, i, o: setattr(t, 'active', True),
                     lambda t, i, o: setattr(t, 'device', 3),
                     lambda t, i, o: setattr(t, 'stream', 20),
                     lambda t, i, o: setattr(t, 'calls', 1),
                     lambda t, i, o: setattr(i, 'shape', (1, 2, 8)),
                     lambda t, i, o: setattr(t, 'group', object())]
        for mutate in mutations:
            tracker, value, output, _, _ = self.fixture()
            group = tracker.group
            mutate(tracker, value, output)
            with patch.object(ma17, 'LOCAL_SHAPE', (2, 1, 8)), self.assertRaises(RuntimeError):
                tracker.bound_gather(tracker.names[0])(output, value, group)
            self.assertTrue(tracker.failed)
            self.assertEqual(tracker.native.calls, 0)
            with self.assertRaises(RuntimeError): tracker.state()
        tracker, value, output, _, _ = self.fixture('smoke', corrupt=True)
        with patch.object(ma17, 'LOCAL_SHAPE', (2, 1, 8)), self.assertRaisesRegex(RuntimeError, 'bytewise'):
            tracker.bound_gather(tracker.names[0])(output, value, tracker.group)
        self.assertTrue(tracker.failed)


class NativeTests(unittest.TestCase):
    def test_real_dynamic_linkage_accepts_shared_dependency_and_rejects_static_copy(self):
        # Exercise the actual ELF loader and RTLD_NOLOAD, with tiny host-only
        # libraries. No CUDA/NCCL runtime is required or impersonated globally.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_source = root / 'native.c'
            torch_source = root / 'torch.c'
            native_source.write_text('int ncclAllGather(void) { return 0; }\n'
                                     'int ncclGetVersion(void) { return 0; }\n')
            torch_source.write_text('extern int ncclAllGather(void);\n'
                                    'extern int ncclGetVersion(void);\n'
                                    'int use_nccl(void) { return ncclAllGather()+ncclGetVersion(); }\n')
            nccl_path = root / 'libfixture_nccl.so'
            linked_path = root / 'libfixture_torch.so'
            static_path = root / 'libfixture_static.so'
            subprocess.run(['cc', '-shared', '-fPIC', str(native_source), '-o', str(nccl_path)], check=True)
            subprocess.run(['cc', '-shared', '-fPIC', str(torch_source), '-L', str(root),
                            '-lfixture_nccl', '-Wl,-rpath,$ORIGIN', '-o', str(linked_path)], check=True)
            subprocess.run(['cc', '-shared', '-fPIC', str(native_source), '-o', str(static_path)], check=True)
            native = ctypes.CDLL(str(nccl_path), mode=ma17.os.RTLD_LOCAL)
            linked = ctypes.CDLL(str(linked_path), mode=ma17.os.RTLD_LOCAL)
            static = ctypes.CDLL(str(static_path), mode=ma17.os.RTLD_LOCAL)
            self.assertEqual(linked.use_nccl(), 0)
            self.assertEqual(static.ncclGetVersion(), 0)
            with patch.object(ma17, 'loaded_torch_cuda_path', return_value=linked_path):
                evidence = ma17.verify_torch_nccl_linkage(native)
            self.assertTrue(evidence['linkage_verified'])
            for row in evidence['linkage_symbol_addresses'].values():
                self.assertEqual(row['direct'], row['torch_dependency'])
            with patch.object(ma17, 'loaded_torch_cuda_path', return_value=static_path), \
                    self.assertRaisesRegex(RuntimeError, 'different NCCL'):
                ma17.verify_torch_nccl_linkage(native)

    def test_linkage_missing_symbols_and_ambiguous_torch_mapping_rejected(self):
        callback = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0)
        native = types.SimpleNamespace(ncclAllGather=callback, ncclGetVersion=callback)
        with patch.object(ma17, 'loaded_torch_cuda_path', return_value=Path('/already/loaded/libtorch_cuda.so')), \
                patch.object(ma17.ctypes, 'CDLL', return_value=types.SimpleNamespace()), \
                self.assertRaisesRegex(RuntimeError, 'cannot verify'):
            ma17.verify_torch_nccl_linkage(native)
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / 'libtorch_cuda.so', Path(tmp) / 'libtorch_cuda.so.1'
            one.touch(); two.touch()
            maps = '\n'.join('000-111 r-xp 0 00:00 1 ' + str(p) for p in (one, two))
            with self.assertRaises(RuntimeError): ma17.loaded_torch_cuda_path(maps)

    def test_loaded_library_selection_rejects_missing_multiple_and_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / 'libnccl.so.2'
            second = Path(tmp) / 'libnccl.so.3'
            first.touch(); second.touch()
            line = lambda path: '000-111 r-xp 0 00:00 1 ' + str(path)
            self.assertEqual(ma17.loaded_nccl_path(line(first) + '\n' + line(first)), first)
            for maps in ('', line(first) + '\n' + line(second), line(first) + ' (deleted)'):
                with self.assertRaises(RuntimeError): ma17.loaded_nccl_path(maps)

    def fixture(self, path, version=22105, failure=None):
        calls = []

        class Function:
            def __init__(self, name): self.name = name
            def __call__(self, *args):
                calls.append((self.name, args))
                if self.name == 'ncclGetVersion': args[0]._obj.value = version
                if self.name == 'ncclGetUniqueId':
                    ctypes.memmove(args[0], bytes(range(128)), 128)
                if self.name == 'ncclCommInitRank': args[0]._obj.value = 12345
                if self.name == 'ncclCommGetAsyncError': args[1]._obj.value = 3 if failure == 'async' else 0
                if self.name == 'ncclGetErrorString': return b'test error'
                return 3 if self.name == failure else 0

        names = ('ncclGetVersion', 'ncclGetUniqueId', 'ncclCommInitRank', 'ncclAllGather',
                 'ncclCommGetAsyncError', 'ncclCommDestroy', 'ncclCommAbort', 'ncclGetErrorString')
        library = types.SimpleNamespace(**{name: Function(name) for name in names})

        def identities(result, value, group=None): result[:] = [value] * 8
        def broadcast(payload, **kwargs): self.assertEqual(payload[0], bytes(range(128)))

        proxy = types.SimpleNamespace(device=torch.device,
            cuda=types.SimpleNamespace(current_device=lambda: 4,
                current_stream=lambda: types.SimpleNamespace(cuda_stream=19),
                nccl=types.SimpleNamespace(version=lambda: (2, 21, 5)),
                synchronize=lambda device: calls.append(('sync', (device,))),
                mem_get_info=lambda device: (80000, 100000)),
            distributed=types.SimpleNamespace(get_process_group_ranks=lambda group: list(range(8)),
                get_world_size=lambda group=None: 8, get_rank=lambda group=None: 0,
                all_gather_object=identities, broadcast_object_list=broadcast))
        linkage = {k: v for k, v in valid_state()['native'].items()
                   if k.startswith('linkage_') or k == 'torch_cuda_library_path'}
        with patch.object(ma17, 'loaded_nccl_path', return_value=path), \
                patch.object(ma17, 'verify_torch_nccl_linkage', return_value=linkage), \
                patch.object(ma17.ctypes, 'CDLL', return_value=library) as loader:
            native = ma17.NativeAllGather(proxy, object(), 0)
        self.assertTrue(loader.call_args.kwargs['mode'] & ma17.os.RTLD_NOLOAD)
        return native, library, calls

    def test_native_abi_current_stream_error_query_and_synchronized_destroy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'libnccl.so.2'; path.write_bytes(b'fake loaded ELF')
            native, library, calls = self.fixture(path)
            value = types.SimpleNamespace(data_ptr=lambda: 1000, numel=lambda: 1048576)
            output = types.SimpleNamespace(data_ptr=lambda: 2000)
            native.launch(output, value)
            launch = next(args for name, args in calls if name == 'ncclAllGather')
            self.assertEqual(launch[:4], (1000, 2000, 1048576, 9))
            self.assertEqual(launch[4].value, 12345)
            self.assertEqual(launch[5], 19)
            self.assertIs(library.ncclAllGather.argtypes[2], ctypes.c_size_t)
            state = native.state()
            self.assertEqual(state['library_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(state['calls'], 1)
            self.assertEqual(state['async_checks'], 1)
            self.assertEqual(state['device_memory_after_work'], [80000, 100000])
            self.assertTrue(state['linkage_verified'])
            native.close(); native.close()
            self.assertEqual([name for name, _ in calls][-2:], ['sync', 'ncclCommDestroy'])
            with self.assertRaises(RuntimeError): native.launch(output, value)
            with self.assertRaises(RuntimeError): native.state()

    def test_version_mismatch_launch_error_and_async_error_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'libnccl.so.2'; path.write_bytes(b'fake loaded ELF')
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                self.fixture(path, version=22200)
            for failure in ('ncclAllGather', 'async'):
                native, _, calls = self.fixture(path, failure=failure)
                with self.assertRaisesRegex(RuntimeError, 'NCCL'):
                    if failure == 'async': native.state()
                    else:
                        value = types.SimpleNamespace(data_ptr=lambda: 1000, numel=lambda: 1048576)
                        native.launch(value, value)
                self.assertTrue(native.failed and native.closed)
                self.assertIn('ncclCommAbort', [name for name, _ in calls])


class EvidenceTests(unittest.TestCase):
    def test_all_modes_ranks_and_json_roundtrip(self):
        for rank in range(8):
            for mode in ('smoke', 'timing', 'profile'):
                self.assertTrue(ma17.validate_state(json.loads(json.dumps(valid_state(mode=mode, rank=rank))),
                                                    3, mode, rank))

    def test_source_counts_rank_abi_errors_audits_and_provenance_rejected(self):
        changes = [lambda s: s.update(source_sha256='0' * 64),
            lambda s: s.update(backward_sha256='0' * 64),
            lambda s: s.update(calls=1), lambda s: s.update(active=True),
            lambda s: s['modules'].pop(next(iter(s['modules']))),
            lambda s: next(iter(s['modules'].values())).update(audits=0),
            lambda s: next(iter(s['modules'].values())).update(rank=3),
            lambda s: s['native'].update(async_error=3),
            lambda s: s['native'].update(initializations=2),
            lambda s: s['native'].update(datatype=7),
            lambda s: s['native'].update(closed=True),
            lambda s: s['native'].update(version=22200),
            lambda s: s['native'].update(library_path='relative.so'),
            lambda s: s['native'].update(library_sha256='g' * 64),
            lambda s: s['native'].update(device_memory_after=None),
            lambda s: s['native'].update(device_memory_after_work=None),
            lambda s: s['native'].update(linkage_verified=False),
            lambda s: s['native'].update(torch_cuda_library_path='relative.so'),
            lambda s: s['native']['linkage_symbol_addresses']['ncclAllGather'].update(torch_dependency=7)]
        for mutate in changes:
            state = valid_state(); mutate(state)
            with self.assertRaises(RuntimeError): ma17.validate_state(state, 3, 'smoke', 4)
        for mode in ('timing', 'profile'):
            with self.assertRaises(RuntimeError): ma17.validate_state(valid_state(), 3, mode, 4)


class InstallationTests(unittest.TestCase):
    def fixture(self):
        model, modules, packages, original, layers = ma8_tests.InstallationTests.fixture(self)
        model.config.context_parallel_size = 1
        layers.get_tensor_model_parallel_rank = lambda: 4
        with patch.dict(sys.modules, packages): ma17.cache.install_activation_cache(model)
        return model, modules, packages, original, layers

    def test_installation_preserves_original_wrapper_global_symbols_and_head(self):
        model, modules, packages, original, layers = self.fixture()
        with patch.dict(sys.modules, packages), patch.object(ma17, 'NativeAllGather', FakeNative):
            tracker = ma17.install_direct_allgather(model, 'timing')
            with self.assertRaisesRegex(RuntimeError, 'duplicate'):
                ma17.install_direct_allgather(model, 'timing')
        self.assertEqual(len(tracker.modules), 64)
        for _, module in modules[:-1]:
            wrapper = module.forward.__func__.__globals__['linear_with_grad_accumulation_and_async_allreduce']
            cls = wrapper.__globals__['LinearWithGradAccumulationAndAsyncCommunication']
            self.assertEqual(cls.__name__, '_MA17DirectAllGatherLinear')
            self.assertIs(module.forward.__func__.__code__, original.__code__)
        self.assertIs(modules[-1][1].forward.__func__, original)
        self.assertIs(layers.ColumnParallelLinear.forward, original)
        self.assertEqual(layers.LinearWithGradAccumulationAndAsyncCommunication.__name__,
                         'LinearWithGradAccumulationAndAsyncCommunication')
        tracker.close(); self.assertTrue(tracker.native.closed)

    def test_late_bad_module_fails_before_creating_communicator_or_mutating_model(self):
        model, modules, packages, _, _ = self.fixture()
        before = [module.forward for _, module in modules]
        modules[-2][1].weight.shape = (3, 4)
        with patch.dict(sys.modules, packages), patch.object(ma17, 'NativeAllGather') as native:
            with self.assertRaisesRegex(RuntimeError, 'incompatible cached'):
                ma17.install_direct_allgather(model, 'timing')
            native.assert_not_called()
        self.assertEqual(before, [module.forward for _, module in modules])


if __name__ == '__main__':
    unittest.main()
