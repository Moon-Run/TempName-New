"""CPU source-tree repair tests; no native execution or GPU evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import repair_native_rank as r


class Repair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='rtx5090-rank-repair-cpu-')
        cls.root = Path(cls.temp.name)/'build'
        verifier_root = Path(cls.temp.name)/'legacy-verifier'
        verifier_root.mkdir()
        verifier = r.legacy_verifier(verifier_root, r.legacy_source())
        verifier.prepare(cls.root)
        paths = ['baseline-build.json', 'selective/manifest.json']
        paths += [f'selective/{arm}/{r.FRONTEND}' for arm in r.builder.NATIVE_ARMS]
        cls.originals = {name: (cls.root/name).read_bytes() for name in paths}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        for name, data in self.originals.items():
            (self.root/name).write_bytes(data)

    def test_repair_all_six_frontends_preserves_sources_and_backs_up_manifests(self):
        with self.assertRaisesRegex(ValueError, 'Local staging implementation changed'):
            r.builder.verify(self.root)
        originals = {str(path.relative_to(self.root)): r.builder.sha(path)
                     for path in self.root.rglob('*') if path.is_file()
                     and 'repairs' not in path.relative_to(self.root).parts
                     and str(path.relative_to(self.root)) not in self.originals}
        result = r.repair(self.root, require_built=False)
        self.assertEqual(result['status'], 'repaired')
        self.assertFalse(result['native_binaries_verified'])
        self.assertFalse(result['gpu_validated'])
        backup = Path(result['backup'])
        for name, data in self.originals.items():
            self.assertEqual((backup/name).read_bytes(), data)
            self.assertEqual(r.builder.sha(self.root/name), result['new_sha256'][name])
        for name, digest in originals.items():
            self.assertEqual(r.builder.sha(self.root/name), digest)
        for arm in r.builder.NATIVE_ARMS:
            source = (r.builder.native_root(self.root, arm)/r.FRONTEND).read_text()
            self.assertIn(r.RANK_INITIALIZATION, source)
        self.assertEqual(r.builder.verify(self.root)['status'], 'prepared')
        self.assertEqual(r.repair(self.root, require_built=False)['status'], 'already_repaired')

    def test_cli_default_rejects_an_uncompiled_build_without_changing_it(self):
        with self.assertRaisesRegex(ValueError, 'Build incomplete'):
            r.repair(self.root)
        for name, data in self.originals.items():
            self.assertEqual((self.root/name).read_bytes(), data)

    def test_corrupt_native_code_is_rejected_before_repair(self):
        source = r.builder.native_root(self.root, 'ma11')/'src/gemm_rs/local_transport.cu'
        original = source.read_bytes()
        try:
            source.write_bytes(original+b'\n// unregistered mutation\n')
            with self.assertRaisesRegex(ValueError, 'Missing or changed input'):
                r.repair(self.root, require_built=False)
            for name, data in self.originals.items():
                self.assertEqual((self.root/name).read_bytes(), data)
        finally:
            source.write_bytes(original)

    def test_unknown_frontend_revision_is_rejected(self):
        path = self.root/'baseline-build.json'
        record = r.builder.read(path)
        record['transport_inputs']['nccl_transport.py'] = 'f'*64
        r.builder.write(path, record)
        with self.assertRaisesRegex(ValueError, 'registered missing-rank frontend'):
            r.repair(self.root, require_built=False)

    def test_failed_post_verification_restores_exact_original_files(self):
        with patch.object(r.builder, 'verify', side_effect=ValueError('injected verification failure')):
            with self.assertRaisesRegex(ValueError, 'injected verification failure'):
                r.repair(self.root, require_built=False)
        for name, data in self.originals.items():
            self.assertEqual((self.root/name).read_bytes(), data)
        receipts = [r.builder.read(path) for path in (self.root/'repairs').glob('*/repair.json')]
        self.assertTrue(any(item['status'] == 'rolled_back' for item in receipts))


if __name__ == '__main__':
    unittest.main()
