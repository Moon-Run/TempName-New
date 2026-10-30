"""Paired row sanity keeps MA11 arithmetic and verifies typed candidate identity."""
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import mav1_bench as bench


class NativeFunction:
    def __init__(self, result):
        self.result = result

    def __call__(self):
        return self.result


def slot():
    exports = dict(taco_ma4f_protocol_version=NativeFunction(1),
                   ma9_zero_count_protocol=NativeFunction(9),
                   ma9_zero_count_chunk_size=NativeFunction(32768),
                   ma11_softmax_protocol=NativeFunction(11),
                   ma11_softmax_forward_cuda=NativeFunction(0),
                   ma11_softmax_backward_cuda=NativeFunction(0))
    return types.SimpleNamespace(lib=types.SimpleNamespace(**exports),
                                 cr2_submission_state=lambda: dict(bound_k=512))


def extension(candidate):
    return (types.SimpleNamespace(Mav1ColumnPlan=object, Mav1RowPlan=object,
                                  mav1_native_protocol=lambda: 1)
            if candidate else types.SimpleNamespace())


class BenchTests(unittest.TestCase):
    def test_complete_balanced_schedule(self):
        jobs = bench.schedule(8)
        self.assertEqual(len(jobs), 16)
        for half in (range(4), range(4, 8)):
            orders = Counter(tuple(j['arm'] for j in jobs if j['block'] == b) for b in half)
            self.assertEqual(orders, {bench.ARMS: 2, bench.ARMS[::-1]: 2})

    def test_shared_ma11_and_typed_candidate_identity(self):
        for candidate, arm in enumerate(bench.ARMS):
            op = types.SimpleNamespace(slots=[slot(), slot()])
            with patch.dict(sys.modules, flux_ths_pybind=extension(candidate)):
                bench.native_tuning(op, arm, dict(tuning=bench.validate_tuning(None)))
                op.slots[0].lib.ma11_softmax_protocol.result = 12
                with self.assertRaisesRegex(ValueError, 'original MA11'):
                    bench.native_tuning(op, arm, dict(tuning=bench.validate_tuning(None)))
            with patch.dict(sys.modules, flux_ths_pybind=extension(not candidate)):
                op = types.SimpleNamespace(slots=[slot(), slot()])
                with self.assertRaisesRegex(ValueError, 'native|APIs'):
                    bench.native_tuning(op, arm, dict(tuning=bench.validate_tuning(None)))

    def test_frozen_operator_can_import_without_checkout(self):
        with tempfile.TemporaryDirectory(prefix='mav1-frozen-') as tmp:
            scripts = Path(tmp) / 'scripts'
            scripts.mkdir()
            bench.freeze_helpers(scripts)
            shutil.copy2(bench.__file__, scripts / 'bench.py')
            code = ('import sys;sys.path.insert(0,sys.argv[1]);import bench;'
                    'assert bench.ARMS==("F4-cache-zcount-csoftmax",'
                    '"F4-cache-zcount-csoftmax-native-linear");'
                    'assert len(bench.schedule())==16')
            result = subprocess.run([sys.executable, '-I', '-c', code, str(scripts)],
                                    cwd=scripts, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
