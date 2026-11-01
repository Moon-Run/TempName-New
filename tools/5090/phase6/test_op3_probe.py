"""CPU checks that corrupt OP3 wire, parity and fusion evidence are rejected."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import op3_probe as probe


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(arm):
    reference = dict(passed=True, budget_vs_bf16=.05, budget_vs_codec=.01,
        relative_l2=.01, protocol_relative_l2=.001, max_abs=.1)
    rows = []
    for rank in range(8):
        def wire(case):
            return dict(upstream_packet_bitwise=True, upstream_decode_fold_bitwise=True,
                sent={str(peer): dict(bytes=8192*136, sha256=digest(f'{case}/{rank}/{peer}'))
                      for peer in range(8) if peer != rank},
                received={str(peer): dict(bytes=8192*136, sha256=digest(f'{case}/{peer}/{rank}'))
                          for peer in range(8) if peer != rank})
        mapping = [[((i % 8+rank) % 8)*2+(i//8)//32, (i//8) % 32, i//32, i % 32]
                   for i in range(512)]
        row = dict(protocol=probe.PROTOCOL, completed=True, arm=arm, rank=rank,
            build_manifest_sha256='manifest', entrypoint_sha256=probe.sha(probe.__file__),
            reference_sha256=probe.sha(probe.REPO/'tools/a800/phase4/reference.py'),
            wire_oracle_sha256=probe.sha(probe.HERE/'op1_abc_probe.py'), probe_stream=7,
            default_stream=0, mapping={role: mapping for role in probe.ROLES}, roles={}, consumer_fusion=None)
        for role, k in probe.ROLES.items():
            cases = {case: dict(input_sha256=digest(case), weight_sha256=digest(role),
                output_sha256=digest(role+case), reference=reference, wire=wire(role+case)) for case in probe.CASES}
            row['roles'][role] = dict(k_local=k, cases=cases, retained_outputs=6, reuse_calls=7,
                retention_passed=True, independent_storage=True, delayed_reference=reference,
                state=dict(arm=arm, calls=10, rank=rank, fused_calls=0))
        if arm == 'op3-r':
            cases = {case: dict(two_rounding_bitwise=True, single_rounding_differs=True, inputs_unchanged=True,
                **{key: digest(case) for key in ('partial_sha256', 'bias_sha256', 'residual_sha256',
                                               'raw_sha256', 'expected_sha256', 'output_sha256')},
                wire=wire(case)) for case in probe.FUSION_CASES}
            row['consumer_fusion'] = dict(cases=cases, retained_outputs=6, reuse_calls=7,
                retention_passed=True, independent_storage=True, dynamic_bias_residual_storage=True,
                state=dict(arm=arm, calls=13, rank=rank, fused_calls=7))
        rows.append(row)
    return rows


def check_fixture_state(state, arm, calls, rank, fused_calls=0):
    if state != dict(arm=arm, calls=calls, rank=rank, fused_calls=fused_calls):
        raise ValueError('stale transport counts')


class ProbeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.state = patch.object(probe, 'check_state', side_effect=check_fixture_state)
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_accepts_full_raw_and_fused_contracts(self):
        for arm in probe.p.ARMS:
            self.assertTrue(probe.check_rows(fixture(arm), arm, 'manifest')['completed'])

    def test_rejects_received_packet_from_wrong_source(self):
        rows = fixture('op3-improved')
        rows[0]['roles']['attn']['cases']['random']['wire']['received']['1']['sha256'] = digest('wrong')
        with self.assertRaisesRegex(ValueError, 'directed packet'):
            probe.check_rows(rows, 'op3-improved', 'manifest')

    def test_rejects_fused_rounding_and_missing_sentinel(self):
        for field, value in [('output_sha256', digest('single-rounded')), ('single_rounding_differs', False)]:
            rows = fixture('op3-r')
            rows[0]['consumer_fusion']['cases']['half_ulp'][field] = value
            with self.assertRaises(ValueError):
                probe.check_rows(rows, 'op3-r', 'manifest')

    def test_rejects_stale_counts_and_reused_outputs(self):
        rows = fixture('op3-r')
        rows[0]['consumer_fusion']['state']['fused_calls'] = 6
        with self.assertRaisesRegex(ValueError, 'stale transport'):
            probe.check_rows(rows, 'op3-r', 'manifest')
        rows = fixture('op3-r')
        rows[0]['consumer_fusion']['retention_passed'] = False
        with self.assertRaisesRegex(ValueError, 'lifetime'):
            probe.check_rows(rows, 'op3-r', 'manifest')

    def test_rejects_changed_raw_output_between_arms(self):
        import op3_build as builder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in probe.p.ARMS:
                for rank, row in enumerate(fixture(arm)):
                    probe.write(root/arm/f'rank{rank}.json', row)
            plan = dict(build=dict(root=str(root/'build'), manifest_sha256='manifest'))
            with patch.object(builder, 'libraries', return_value=None), patch.object(builder, 'codec_library', return_value=None):
                self.assertTrue(probe.check_records(root, None, plan)['raw_projection_packet_output_bitwise'])
                path = root/'op3-improved/rank0.json'
                row = probe.read(path)
                row['roles']['mlp']['cases']['outlier']['output_sha256'] = digest('changed')
                probe.write(path, row)
                with self.assertRaisesRegex(ValueError, 'raw projection arithmetic'):
                    probe.check_records(root, None, plan)


if __name__ == '__main__':
    unittest.main()
