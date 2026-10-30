"""Exercise complete source reconstruction and fail-closed port provenance."""
import json
from pathlib import Path
import tempfile
import unittest

import rtx5090_loader

build = rtx5090_loader.load('build')


class NativePortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='rtx5090-build-test-')
        cls.root = Path(cls.temp.name)/'build'
        build.prepare(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_nine_arms_preserve_registered_algorithm_sources(self):
        record = build.verify(self.root)
        self.assertFalse(record['gpu_validated'])
        self.assertTrue(record['experimental'])
        self.assertFalse(record['peer_memory_required'])
        self.assertEqual(record['transport'], 'rtx5090-nccl-staged-v2')
        self.assertEqual(len(record['arms']), 9)
        self.assertEqual(build.native_root(self.root, 'ma8'), build.native_root(self.root, 'cr3'))
        self.assertEqual(build.native_root(self.root, 'ma9'), build.native_root(self.root, 'ma89'))
        self.assertEqual((self.root/'selective/selection-plan.json').read_bytes(),
                         (self.root/'reference/selection-plan.json').read_bytes())
        for arm in build.NATIVE_ARMS:
            source = build.native_root(self.root, arm)
            native = (source/'src/gemm_rs/local_transport.cu').read_text()
            self.assertIn('prop.major == 12', native)
            self.assertNotIn('cudaDeviceCanAccessPeer', native)
            self.assertNotIn('cudaIpc', native)
            self.assertIn('rtx5090_pack_tiles_kernel', native)
            self.assertIn('rtx5090_unpack_tiles_kernel', native)
            cmake = (source/'src/gemm_rs/CMakeLists.txt').read_text()
            self.assertIn('list(REMOVE_ITEM THS_FILES', cmake)
            self.assertNotIn('py::class_<GemmRSOpCls>', (source/'src/pybind/gemm_rs.cc').read_text())
            frontend = (source/'python/flux/gemm_rs_taco.py').read_text()
            self.assertIn('NATIVE_ARM = '+repr(arm), frontend)
            self.assertIn('all_to_all_single', frontend)
            self.assertNotIn('init_flux_shm(', frontend)
            self.assertNotIn('create_tensor_list(', frontend)
            self.assertNotIn('.peers', frontend)
            generator = (source/'src/generator/gen_gemm_rs.cc').read_text()
            self.assertIn('_Sm80{}', generator)
            self.assertNotIn('GemmV3', generator)
            mapping = (source/'check_mapping.cu').read_text()
            self.assertIn('prop.major!=12', mapping)
            self.assertIn('h100_v2_occupancy', mapping)
        basic = build.native_root(self.root, 'flux-basic')/'src/generator/gen_comm_none.cc'
        self.assertIn('_RTX5090{}', basic.read_text())
        self.assertIn('GemmV2CommNone', basic.read_text())
        self.assertNotIn('GemmV3', basic.read_text())

    def test_altered_port_is_rejected(self):
        source = build.native_root(self.root, 'a')/'src/gemm_rs/local_transport.cu'
        original = source.read_bytes()
        try:
            source.write_bytes(original+b'\n// unregistered edit\n')
            with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
                build.verify(self.root)
        finally:
            source.write_bytes(original)

    def test_prepared_sources_cannot_be_reported_as_built(self):
        with self.assertRaisesRegex(ValueError, 'Build incomplete'):
            build.verify(self.root, built=True)

    def test_legacy_direct_peer_build_is_rejected(self):
        path = self.root/'baseline-build.json'
        original = path.read_bytes()
        try:
            value = json.loads(original)
            value['protocol'] = 'rtx5090-frozen-nine-arm-native-sm120-v1'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'recipe changed'):
                build.verify(self.root)
        finally:
            path.write_bytes(original)

    def test_h100_reference_remains_sm90(self):
        reference = self.root/'reference/F4-cache-zcount-csoftmax'
        self.assertIn('requires physical SM90', (reference/build.WRAPPER).read_text())
        self.assertEqual(json.loads((self.root/'reference/manifest.json').read_text())['arch'], 'sm90')


if __name__ == '__main__':
    unittest.main()
