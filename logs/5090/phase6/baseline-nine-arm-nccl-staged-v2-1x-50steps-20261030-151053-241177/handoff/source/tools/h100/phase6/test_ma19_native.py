"""Exercise conversion-oracle coverage on CPU; actual GPU proof remains in smoke.

The fake publisher below deliberately implements copy_ and can be corrupted.
This tests the oracle's rejection power, not the native CUDA conversion itself.
Use check_ma19_native.py --gpu for the latter.
"""
import types
import unittest
from unittest.mock import patch

import torch
import ma19_adapter as ma19


class NativeOracleTests(unittest.TestCase):
    def test_full_midpoint_oracle_covers_all_bit_patterns_and_rejects_corruption(self):
        calls = []
        class Proxy:
            cuda = types.SimpleNamespace(synchronize=lambda: None)
            def __getattr__(self, name): return getattr(torch, name)
            def arange(self, *args, **kwargs):
                kwargs.pop('device', None)
                return torch.arange(*args, **kwargs)
        class CPUPlan:
            corrupt = False
            def __init__(self, torch_arg, library, pairs): self.pairs = pairs
            def publish(self):
                calls.append(1)
                for source, target in self.pairs:
                    target.copy_(source)
                if self.corrupt:
                    self.pairs[-1][1].view(torch.int16)[-1] = 0
            def verify_storage(self): pass
        with patch.object(ma19, 'NativePlan', CPUPlan):
            result = ma19.native_self_test(Proxy(), object())
            self.assertEqual(result['patterns'], 262144)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result['layouts'], ['aligned_full', 'unaligned_split'])
            CPUPlan.corrupt = True
            with self.assertRaisesRegex(RuntimeError, 'conversion differs'):
                ma19.native_self_test(Proxy(), object())


if __name__ == '__main__': unittest.main()
