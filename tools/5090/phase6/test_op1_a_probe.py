"""Reject incomplete or inconsistent probe evidence; these fixtures are not GPU results."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import op1_a_native as native
import op1_a_probe as probe
import op1_a_protocol as p


def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def addresses(variant):
    keys = ['tx_wire', 'rx_bf16'] if variant == 'a' else [
        'tx_bf16', 'tx_packets', 'rx_bf16', 'tx_wire', 'rx_wire']
    rows = []
    for index in range(2):
        if variant == 'a':
            aliases = dict(tx_bf16=100, rx_bf16=100, tx_packets=200, tx_wire=200,
                           rx_wire=300+index*10, rx_packets=300+index*10)
        else:
            aliases = dict(tx_bf16=100, rx_bf16=200, tx_packets=300, tx_wire=400,
                           rx_wire=500, rx_packets=600+index*10)
        rows.append(dict(shared={key: aliases[key] for key in keys},
                         rx_packet=aliases['rx_packets'], native=1000+index, aliases=aliases))
    return dict(shared_keys=keys, instances=rows)


def mapping(rank):
    return [[((index % 8 + rank) % 8)*2+(index//8)//32, (index//8) % 32,
             index//32, index % 32] for index in range(512)]


class ProbeEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.hardware = dict(devices=[dict(uuid=f'gpu-{rank}') for rank in range(8)],
                             environment=dict(torch='fixture', cuda='12.9', nccl=[2, 27, 3]))
        self.manifest = digest('manifest')
        for variant in ('old-a', 'a', 'flux-basic-interleaved'):
            for rank in range(8):
                record = dict(protocol=probe.PROTOCOL, completed=True, rank=rank, variant=variant,
                              build_manifest_sha256=self.manifest, entrypoint_sha256=p.sha(probe.__file__),
                              gpu_uuid=f'gpu-{rank}', probe_stream=71, default_stream=0,
                              **self.hardware['environment'])
                if variant == 'flux-basic-interleaved':
                    record['mapping'] = {role: mapping(rank) for role in probe.ROLES}
                else:
                    record['roles'] = {}
                    for role, k in probe.ROLES.items():
                        cases = {}
                        for case in probe.CASES:
                            values = dict(input_sha256=digest((rank, role, case, 'input')),
                                          weight_sha256=digest((rank, role, case, 'weight')),
                                          output_sha256=digest((rank, role, case, 'output')),
                                          reference=dict(passed=True, relative_l2=.01, protocol_relative_l2=.001,
                                                         max_abs=.02, budget_vs_bf16=.05, budget_vs_codec=.01),
                                          tx={str(peer): digest((rank, peer, role, case)) for peer in range(8) if peer != rank},
                                          rx={str(peer): digest((peer, rank, role, case)) for peer in range(8) if peer != rank})
                            cases[case] = values
                        count = 10 if variant == 'old-a' else 0
                        state = dict(failed=False, pending_exchange=False, native_phase=0,
                                     calls=10, producer_calls=10, exchange_calls=10, wait_calls=10, decode_calls=10,
                                     pack_calls=count, unpack_calls=count, reconfigure_calls=count,
                                     device=rank, stream=71, native=dict(protocol_version=1 if variant == 'old-a' else 2))
                        if variant == 'a':
                            state.update(protocol=native.PROTOCOL, native_api=native.NATIVE_API,
                                         configuration_scope='instance', output_ownership='independent',
                                         gemm_submission_cache=False, initializations=10)
                        record['roles'][role] = dict(k_local=k, cases=cases, retained_outputs=6, reuse_calls=7,
                                                    retention_passed=True, shared_instances_passed=True,
                                                    shared_addresses=addresses(variant), transport=state)
                p.write(self.root/variant/f'rank{rank}.json', record)

    def tearDown(self):
        self.temporary.cleanup()

    def check(self):
        return probe.check_records(self.root, self.hardware, self.manifest)

    def mutate(self, callback, variant='a', rank=0):
        path = self.root/variant/f'rank{rank}.json'
        value = p.read(path)
        callback(value)
        p.write(path, value)

    def test_complete_fixture_only_validates_schema_and_cross_rank_consistency(self):
        result = self.check()
        self.assertTrue(result['completed'])
        self.assertEqual(len(result['evidence_files']), 24)

    def test_missing_rank_blocks_report(self):
        (self.root/'old-a/rank7.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_output_or_input_drift_cannot_pass_old_new_equality(self):
        for field in ('output_sha256', 'input_sha256', 'weight_sha256'):
            with self.subTest(field=field):
                path = self.root/'a/rank0.json'
                original = p.read(path)
                self.mutate(lambda row: row['roles']['attn']['cases']['random'].__setitem__(field, digest('changed')))
                with self.assertRaisesRegex(ValueError, 'packet/output bits'):
                    self.check()
                p.write(path, original)

    def test_directed_packet_mismatch_and_invalid_digest_are_rejected(self):
        self.mutate(lambda row: row['roles']['mlp']['cases']['zero']['tx'].__setitem__('1', digest('wrong')))
        with self.assertRaisesRegex(ValueError, 'directed packet'):
            self.check()

    def test_native_version_stream_or_call_count_drift_is_rejected(self):
        for field, value in (('stream', 0), ('decode_calls', 9), ('pack_calls', 10)):
            with self.subTest(field=field):
                path = self.root/'a/rank0.json'
                original = p.read(path)
                self.mutate(lambda row: row['roles']['attn']['transport'].__setitem__(field, value))
                with self.assertRaises(ValueError):
                    self.check()
                p.write(path, original)
        self.mutate(lambda row: row['roles']['attn']['transport']['native'].__setitem__('protocol_version', 1))
        with self.assertRaisesRegex(ValueError, 'ABI'):
            self.check()

    def test_default_stream_probe_cannot_claim_nondefault_validation(self):
        self.mutate(lambda row: row.__setitem__('probe_stream', 0))
        with self.assertRaisesRegex(ValueError, 'nondefault'):
            self.check()

    def test_missing_or_aliased_private_receive_packet_is_rejected(self):
        def break_private(row):
            evidence = row['roles']['attn']['shared_addresses']['instances']
            evidence[1]['rx_packet'] = evidence[0]['rx_packet']
        self.mutate(break_private)
        with self.assertRaisesRegex(ValueError, 'aliased'):
            self.check()

    def test_numerical_budget_failure_blocks_report(self):
        self.mutate(lambda row: row['roles']['attn']['cases']['outlier']['reference'].__setitem__('protocol_relative_l2', .011))
        with self.assertRaisesRegex(ValueError, 'numerical reference'):
            self.check()

    def test_mapping_requires_actual_permutation_and_integral_coordinates(self):
        good = mapping(0)
        probe.validate_mapping(good, 0)
        for variant in ('duplicate_original', 'wrong_sorted', 'float', 'short'):
            rows = copy.deepcopy(good)
            if variant == 'duplicate_original':
                rows[1][2:] = rows[0][2:]
            elif variant == 'wrong_sorted':
                rows[0][:2] = [15, 31]
            elif variant == 'float':
                rows[0][0] = 0.0
            else:
                rows.pop()
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                probe.validate_mapping(rows, 0)


if __name__ == '__main__':
    unittest.main()
