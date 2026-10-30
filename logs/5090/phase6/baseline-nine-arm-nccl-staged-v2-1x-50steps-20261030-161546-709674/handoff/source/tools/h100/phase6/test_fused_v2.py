"""CPU-only checks of the pinned baseline and exception-safe H100 dispatch."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import build
import common


class FusedV2Tests(unittest.TestCase):
    def test_pinned_sources_and_overlay_isolation(self):
        root = common.HERE/'fused_v2'
        provenance = common.read(root/'provenance.json')
        common.verify_files(root, provenance['files'])
        # A source mismatch must fail before changing the prepared snapshot.
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            for relative in provenance['files']:
                p = dest/relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('snapshot placeholder')
            sentinel = dest/'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
            sentinel.parent.mkdir()
            sentinel.write_text('original order')
            identity = build.install_fused_v2(dest)
            self.assertEqual(identity, 'a800-fused-v2-cta-v1')
            self.assertEqual(sentinel.read_text(), 'original order')
            for relative, digest in provenance['files'].items():
                if relative.endswith('.py'):
                    compile((dest/relative).read_text(), relative, 'exec')
                else:
                    self.assertEqual(common.sha(dest/relative), digest)
            with mock.patch.object(build, 'verify_files', side_effect=ValueError('mismatch')):
                with self.assertRaises(ValueError):
                    build.install_fused_v2(dest)
            self.assertEqual(sentinel.read_text(), 'original order')

    def test_baseline_dispatch_is_per_instance_not_current_tls(self):
        source = (common.HERE/'fused_v2/src/gemm_rs/ths_op/gemm_reduce_scatter.cc').read_text()
        adapted = build.wrapper(source, legacy_fused=True)
        self.assertIn('const bool h100_taco = taco_config().enabled;', adapted)
        self.assertIn('if (h100_taco)', adapted)
        self.assertNotIn('owned_taco', adapted)
        self.assertIn('if (phase6_arch() == _Sm90{} and nnodes == 1)', adapted)
        self.assertIn('return get_arch();', adapted)
        self.assertNotIn('taco.deferred_reuse_barrier', adapted)

    def exercise_frontend(self, fail_at=None):
        source = (common.HERE/'fused_v2/python/flux/gemm_rs_taco.py').read_text()
        source = build.legacy_fused_python(source).replace('(8, 0)', '(9, 0)')
        torch = types.ModuleType('torch')
        dist = types.ModuleType('torch.distributed')
        torch.distributed = dist
        torch.bfloat16, torch.uint8 = object(), object()
        torch.cuda = mock.Mock()
        torch.cuda.get_device_capability.return_value = (9, 0)
        torch.cuda.current_device.return_value = 0
        torch.cuda.current_stream.return_value.cuda_stream = 17
        torch.cuda.device_count.return_value = 8
        torch.cuda.can_device_access_peer.return_value = True
        torch.cuda.is_current_stream_capturing.return_value = False
        dist.get_rank, dist.get_world_size = lambda group: 0, lambda group: 8
        cpp = types.ModuleType('fused_test.cpp_mod')
        cpp.init_flux_shm = mock.Mock()
        cpp.create_tensor_list = mock.Mock(return_value=[types.SimpleNamespace(data_ptr=lambda: 1)]*8)
        cpp.ReduceScatterOption = types.SimpleNamespace
        cpp.RingMode = types.SimpleNamespace(Ring1D=1)
        package = types.ModuleType('fused_test')
        package.cpp_mod = cpp
        state = {'active': False, 'events': []}
        lib = types.SimpleNamespace()

        def configure(*args):
            self.assertFalse(state['active'])
            state['active'] = True
            state['events'].append('configure')
            return 0

        def reset():
            state['active'] = False
            state['events'].append('reset')

        def forward(*args, **kwargs):
            self.assertTrue(state['active'])
            state['events'].append('forward')
            if fail_at == 'forward':
                raise RuntimeError('forward failed')
            return types.SimpleNamespace(reshape=lambda *shape: shape)

        def construct(*args, **kwargs):
            self.assertTrue(state['active'])  # C++ captures V2 before allocating buffers.
            state['events'].append('construct')
            if fail_at == 'construct':
                raise RuntimeError('construct failed')
            return types.SimpleNamespace(forward=forward)

        lib.taco_configure = mock.Mock(side_effect=configure)
        lib.taco_reset = mock.Mock(side_effect=reset)
        lib.taco_placement = mock.Mock(return_value=1)
        cpp.GemmRS = construct
        module = types.ModuleType('fused_test.gemm_rs_taco')
        module.__file__ = '/tmp/fused-test/gemm_rs_taco.py'
        module.__package__ = 'fused_test'
        modules = {'torch': torch, 'torch.distributed': dist, 'fused_test': package}
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, LOCAL_WORLD_SIZE='8', LOCAL_RANK='0'), \
             mock.patch.object(Path, 'read_text', return_value='/tmp/fused-test/lib/libflux_cuda.so'), \
             mock.patch.object(ctypes, 'CDLL', return_value=lib):
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            def run():
                codec = module.GemmRSTaco(None, 2048, 2048, 1024, placement='fused')
                self.assertFalse(state['active'])  # Attention constructed next selects V3.
                self.assertFalse(getattr(codec, 'owns_configuration', False))  # Keep adapter clone.
                self.assertEqual(codec.wire_bytes_per_rank, 3899392)
                tensors = []
                for shape in [(2048, 1024), (2048, 1024)]:
                    tensors.append(types.SimpleNamespace(is_cuda=True, device=types.SimpleNamespace(index=0),
                        dtype=torch.bfloat16, shape=shape, is_contiguous=lambda: True, requires_grad=False))
                for _ in range(2):
                    self.assertEqual(codec.forward(*tensors), (256, 2048))
                    self.assertFalse(state['active'])
            if fail_at:
                with self.assertRaisesRegex(RuntimeError, fail_at+' failed'):
                    run()
            else:
                run()
        self.assertFalse(state['active'])
        self.assertEqual(state['events'][:3], ['configure', 'construct', 'reset'])
        if fail_at != 'construct':
            self.assertEqual(state['events'][3:6], ['configure', 'forward', 'reset'])

    def test_configuration_does_not_leak_to_attention_or_next_forward(self):
        for fail_at in (None, 'construct', 'forward'):
            with self.subTest(fail_at=fail_at):
                self.exercise_frontend(fail_at)


if __name__ == '__main__':
    unittest.main()
