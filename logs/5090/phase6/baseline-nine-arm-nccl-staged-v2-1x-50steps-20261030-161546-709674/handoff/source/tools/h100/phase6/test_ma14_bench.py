"""MA14 operator harness measures the unchanged row projection only."""
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import ma14_bench as bench


class NativeFunction:
    def __init__(self, result):
        self.result = result

    def __call__(self):
        return self.result


def fake_slot(candidate=False):
    exports = dict(taco_ma4f_protocol_version=NativeFunction(1),
                   ma9_zero_count_protocol=NativeFunction(9),
                   ma9_zero_count_chunk_size=NativeFunction(32768),
                   ma11_softmax_protocol=NativeFunction(11),
                   ma11_softmax_forward_cuda=NativeFunction(0),
                   ma11_softmax_backward_cuda=NativeFunction(0))
    if candidate:
        exports.update(taco_ma14_epoch_protocol=NativeFunction(1402),
                       taco_ma14_epoch_barrier=NativeFunction(0))
    return types.SimpleNamespace(lib=types.SimpleNamespace(**exports),
                                 cr2_submission_state=lambda: dict(bound_k=512))


class BenchTests(unittest.TestCase):
    def test_balanced_complete_blocks(self):
        jobs = bench.schedule(8)
        self.assertEqual(len(jobs), 16)
        for half in (range(4), range(4, 8)):
            orders = Counter(tuple(j['arm'] for j in jobs if j['block'] == block) for block in half)
            self.assertEqual(orders, {bench.ARMS: 2, bench.ARMS[::-1]: 2})
        for blocks in (0, 4, 9, True):
            with self.assertRaises(ValueError):
                bench.schedule(blocks)

    def test_common_ma9_abi_and_candidate_only_epoch(self):
        plan = dict(tuning=bench.validate_tuning(None))
        for arm in bench.ARMS:
            op = types.SimpleNamespace(slots=[fake_slot(arm == bench.ARMS[1]) for _ in range(2)])
            bench.native_tuning(op, arm, plan)
            del op.slots[0].lib.ma9_zero_count_protocol
            with self.assertRaises(AttributeError):
                bench.native_tuning(op, arm, plan)
        op = types.SimpleNamespace(slots=[fake_slot(True), fake_slot(True)])
        with self.assertRaisesRegex(ValueError, 'MA11 control'):
            bench.native_tuning(op, bench.ARMS[0], plan)
        # Version 1 exports the same symbol names; its ABI identity must still
        # be rejected before it can enter a version 2 worker.
        op.slots[0].lib.taco_ma14_epoch_protocol.result = 14
        with self.assertRaisesRegex(ValueError, 'Wrong MA14'):
            bench.native_tuning(op, bench.ARMS[1], plan)

    def test_epoch_evidence_rejects_reset_alias_and_lost_generation(self):
        import copy
        states = [dict(protocol_version=1402, initializations=1, epoch=32,
                       local_ready_ptr=128+64*i, failed=False, slot=i) for i in range(2)]
        submissions = [dict(successful_calls=32, slot=i) for i in range(2)]
        bench.validate_epoch_state(states, submissions, bench.ARMS[1])
        for field, bad in [('protocol_version', 14), ('initializations', 2), ('epoch', 31), ('failed', True),
                           ('local_ready_ptr', 192)]:
            changed = copy.deepcopy(states)
            changed[0][field] = bad
            with self.assertRaises(ValueError):
                bench.validate_epoch_state(changed, submissions, bench.ARMS[1])
        with self.assertRaises(ValueError):
            bench.validate_epoch_state(states, submissions, bench.ARMS[0])

    def test_frozen_operator_import_is_self_contained(self):
        with tempfile.TemporaryDirectory(prefix='ma14-frozen-') as tmp:
            scripts = Path(tmp) / 'scripts'
            scripts.mkdir()
            bench.freeze_helpers(scripts)
            shutil.copy2(bench.__file__, scripts / 'bench.py')
            code = ('import sys;sys.path.insert(0,sys.argv[1]);import bench;'
                    'assert bench.ARMS==("F4-cache-zcount-csoftmax","F4-cache-zcount-csoftmax-epoch");'
                    'assert len(bench.schedule())==16')
            result = subprocess.run([sys.executable, '-I', '-c', code, str(scripts)],
                                    cwd=scripts, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
