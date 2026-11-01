"""Regression for two all-remote C instances with two slots each (CPU fixtures only)."""
import copy
from pathlib import Path
import tempfile
import unittest

import flux_baseline_probe as current
from test_local_smoke import rows_for, s as historical


def all_remote_rows(arm):
    rows = rows_for(arm)
    for row in rows:
        row['protocol'] = current.probe.PROTOCOL
        for role in row['roles'].values():
            for instance in role['shared_scratch']['pointers']:
                for slot in instance:
                    slot['mask'] = None
    return rows


class AllRemoteScratch(unittest.TestCase):
    def test_complete_single_and_double_slot_records_accept_all_null_masks(self):
        for arm, count in (('a', 1), ('c', 2)):
            for rank, row in enumerate(all_remote_rows(arm)):
                for role, value in row['roles'].items():
                    with self.subTest(arm=arm, rank=rank, role=role):
                        record = value['shared_scratch']
                        masks = [slot['mask'] for instance in record['pointers'] for slot in instance]
                        self.assertEqual(masks, [None] * (2 * count))
                        current.probe.validate_shared_scratch(record, arm)

    def test_every_nonnull_mask_is_rejected_in_both_instances_and_all_slots(self):
        for arm, count in (('a', 1), ('c', 2)):
            record = all_remote_rows(arm)[0]['roles']['attn']['shared_scratch']
            for instance in range(2):
                for slot in range(count):
                    for mask in (0, 4096):
                        changed = copy.deepcopy(record)
                        changed['pointers'][instance][slot]['mask'] = mask
                        with self.subTest(arm=arm, instance=instance, slot=slot, mask=mask):
                            with self.assertRaisesRegex(RuntimeError, 'unexpected masks'):
                                current.probe.validate_shared_scratch(changed, arm)

    def test_null_masks_cannot_hide_missing_extra_or_shared_slots(self):
        original = all_remote_rows('c')[0]['roles']['attn']['shared_scratch']
        for mutate in (
            lambda r: r['pointers'][1].pop(),
            lambda r: r['pointers'][1].append(copy.deepcopy(r['pointers'][1][0])),
            lambda r: r['pointers'][1][1].update(packet=r['pointers'][0][1]['packet']),
            lambda r: r['pointers'][1][1].update(native=r['pointers'][0][1]['native']),
            lambda r: r['pointers'][1][1]['shared'].update(tx_wire=123),
            lambda r: r['before_close'][0]['slots'][1].update(wait_calls=0),
            lambda r: r.update(survivor_after_close_bitwise=False),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.assertRaises(RuntimeError):
                current.probe.validate_shared_scratch(changed, 'c')

    def test_complete_eight_rank_probe_report_reaches_all_later_checks(self):
        with tempfile.TemporaryDirectory(prefix='flux-probe-cpu-fixture-') as tmp:
            root = Path(tmp)
            for arm in ('a', 'c'):
                rows = all_remote_rows(arm)
                for rank, row in enumerate(rows):
                    current.probe.write(root/f'rank{rank}.json', row)
                result = current.validate_reports(root, arm, 'a'*64)
                self.assertTrue(result['completed'])
                self.assertEqual(len(result['rank_records']), 8)
                self.assertFalse(result['performance_measured'])
                # A valid mask vector must not bypass cross-rank wire checking.
                changed = copy.deepcopy(rows[0])
                changed['roles']['mlp']['wire']['received']['1']['sha256'] = 'f'*64
                current.probe.write(root/'rank0.json', changed)
                with self.assertRaisesRegex(RuntimeError, 'Cross-rank wire hash'):
                    current.validate_reports(root, arm, 'a'*64)

    def test_historical_selective_c_contract_is_unchanged(self):
        original = rows_for('c')[0]['roles']['attn']['shared_scratch']
        historical.validate_shared_scratch(original, 'c')
        with self.assertRaisesRegex(RuntimeError, 'unexpected masks'):
            current.probe.validate_shared_scratch(original, 'c')
        without_masks = all_remote_rows('c')[0]['roles']['attn']['shared_scratch']
        with self.assertRaises(RuntimeError):
            historical.validate_shared_scratch(without_masks, 'c')


if __name__ == '__main__':
    unittest.main()
