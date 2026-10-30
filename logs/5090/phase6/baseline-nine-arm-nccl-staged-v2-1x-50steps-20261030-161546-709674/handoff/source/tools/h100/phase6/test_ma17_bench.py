"""MA17 operator harness measures the unchanged row projection only."""
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import ma17_bench as bench


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

    def test_both_arms_require_original_ma9_and_ma11_abi(self):
        plan = dict(tuning=bench.validate_tuning(None))
        for arm in bench.ARMS:
            op = types.SimpleNamespace(slots=[fake_slot(arm == bench.ARMS[1]) for _ in range(2)])
            bench.native_tuning(op, arm, plan)
            del op.slots[0].lib.ma9_zero_count_protocol
            with self.assertRaises(AttributeError):
                bench.native_tuning(op, arm, plan)
        for arm in bench.ARMS:
            op = types.SimpleNamespace(slots=[fake_slot(), fake_slot()])
            op.slots[0].lib.ma11_softmax_protocol.result = 12
            with self.assertRaisesRegex(ValueError, 'original MA11'):
                bench.native_tuning(op, arm, plan)

    def test_frozen_operator_import_is_self_contained(self):
        with tempfile.TemporaryDirectory(prefix='ma17-frozen-') as tmp:
            scripts = Path(tmp) / 'scripts'
            scripts.mkdir()
            bench.freeze_helpers(scripts)
            shutil.copy2(bench.__file__, scripts / 'bench.py')
            code = ('import sys;sys.path.insert(0,sys.argv[1]);import bench;'
                    'assert bench.ARMS==("F4-cache-zcount-csoftmax","F4-cache-zcount-csoftmax-directag");'
                    'assert len(bench.schedule())==16')
            result = subprocess.run([sys.executable, '-I', '-c', code, str(scripts)],
                                    cwd=scripts, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
