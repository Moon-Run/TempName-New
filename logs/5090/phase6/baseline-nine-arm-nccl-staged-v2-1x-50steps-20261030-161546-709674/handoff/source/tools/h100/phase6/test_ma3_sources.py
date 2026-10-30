"""MA3 modifies receiver dispatch only; ownership and publication remain A's."""
import unittest

import ma3_sources as sources
import test_arrival_op3_build as fixture


class SourceTests(unittest.TestCase):
    setUp = fixture.SourceIntegrityTests.setUp
    tearDown = fixture.SourceIntegrityTests.tearDown

    def test_exact_dispatch_and_no_ownership_or_producer_changes(self):
        for arm, kernel, blocks in (('D', 'serial', 'taco_groups(c) / 4'),
                                    ('S', 'parallel', 'taco_groups(c)')):
            result = sources.sources(self.baseline, arm)
            self.assertEqual(set(result), {sources.RUNTIME, sources.DECODER})
            runtime = result[sources.RUNTIME]
            self.assertIn(f'taco_ma3_{kernel}_kernel<<<{blocks}, 128, 0, stream>>>', runtime)
            self.assertIn('!taco_ma3_exact_shape(c)', runtime)
            self.assertIn('taco_ma3_protocol_version() { return 1; }', runtime)
            self.assertIn(f'taco_ma3_source_parallel_enabled() {{ return {int(arm == "S")}; }}', runtime)
            self.assertIn('taco_ma3_prefetch_enabled() { return 0; }', runtime)
        self.assertEqual(sources.sources(self.baseline, 'D')[sources.DECODER],
                         sources.sources(self.baseline, 'S')[sources.DECODER])

    def test_rejects_unregistered_arm(self):
        with self.assertRaisesRegex(ValueError, 'only D/S'):
            sources.sources(self.baseline, 'P')

    def test_changed_dispatch_anchor_fails_closed(self):
        path = self.baseline/sources.RUNTIME
        path.write_text(path.read_text().replace('taco_decode_ring_kernel<<<', 'different_kernel<<<'))
        with self.assertRaises(ValueError):
            sources.sources(self.baseline, 'S')


if __name__ == '__main__':
    unittest.main()
