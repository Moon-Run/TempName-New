"""MA11 identity, isolated frozen replay, exact acceptance and installation gates."""
import argparse
import ast
from collections import Counter
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import ma16_bench as bench
import ma16_fullstep as full
from test_ma2_cr2 import fixture_plan


class HarnessTests(unittest.TestCase):
    def test_original_training_and_timing_and_candidate_only_install(self):
        adapter, taco, worker = full.adapter_source(), full.taco_source(), full.worker_source()
        identity = full.validate_baseline_sources(adapter, taco, worker)
        self.assertTrue(identity['shared_zero_count_installation'])
        self.assertEqual(identity['control'], 'F4-cache-zcount-csoftmax')
        for old, new in [('successful, grad_norm, _ = optimizer.step()', 'successful, grad_norm, _ = optimizer.bad_step()'),
                         ('start=time.perf_counter()', 'start=time.monotonic()'),
                         ("install_activation_cache(raw, 'F4-cache')", "install_activation_cache(raw, 'F4')"),
                         ("ma11_adapter.install_softmax(raw, ma16_library, MODE)", "ma11_adapter.install_softmax(raw, ma16_library, 'timing')"),
                         ("ma16_adapter.install_local_first(raw, MODE)",
                          "ma16_adapter.install_local_first(raw, 'timing')")]:
            self.assertIn(old, worker)
            with self.assertRaises(ValueError):
                full.validate_baseline_sources(adapter, taco, worker.replace(old, new))

    def test_all_shared_compute_mechanisms_execute_in_both_arms(self):
        import ctypes
        tree = ast.parse(full.worker_source())
        start = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Import) and
                     any(a.name == 'ma8_adapter' for a in n.names))
        end = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "os.environ['H100_MA16_ARM'] == 'F4-cache-zcount-csoftmax-localfirst'")
        startup = compile(ast.Module(body=tree.body[start:end+1], type_ignores=[]), 'startup', 'exec')
        for arm in bench.ARMS:
            calls = []
            modules = {name:types.SimpleNamespace(**{method:lambda *a, label=name: calls.append(label) or object()})
                       for name, method in [('ma8_adapter','install_activation_cache'),
                                            ('ma9_adapter','install_zero_count'),('ma11_adapter','install_softmax'),('ma16_adapter','install_local_first')]}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, H100_MA16_ARM=arm,E2E_BUILD='/tmp/build'), \
                 patch.object(ctypes, 'CDLL', return_value=object()):
                exec(startup,dict(os=os,Path=Path,raw=object(),optimizer=object(),POLICY='selective',MODE='timing'))
            self.assertEqual(calls, ['ma8_adapter','ma9_adapter','ma11_adapter'] +
                             (['ma16_adapter'] if arm == bench.ARMS[1] else []))

    def test_balanced_schedule_and_two_percent_lower_bound(self):
        for trials in (1, 2):
            rows = full.expected_schedule(trials, 8)
            self.assertEqual(len(rows), trials*16)
            for trial in range(1,trials+1):
                self.assertEqual(Counter(r['arm'] for r in rows if r['trial']==trial),dict.fromkeys(bench.ARMS,8))
        key = bench.ARMS[1]+'/'+bench.ARMS[0]
        means = dict(zip(bench.ARMS,(460,450)))
        result = full.comparison_acceptance({key:dict(percent=2.1,ci95=[1.9,2.3])},means)
        self.assertTrue(result['target_point_observed'])
        self.assertFalse(result['target_validated'])
        result = full.comparison_acceptance({key:dict(percent=2.2,ci95=[2.0,2.4])},means)
        self.assertTrue(result['target_validated'])

    def test_shared_softmax_and_ma9_cannot_be_missing(self):
        with self.assertRaises(ValueError):
            full.validate_softmax_state({'installed':False},bench.ARMS[0],60,'timing')
        with self.assertRaises(ValueError):
            full.validate_softmax_state({'installed':True},bench.ARMS[0],60,'timing')
        with self.assertRaises(ValueError):
            full.validate_zero_count_state({'installed':False},bench.ARMS[0],60,'timing',0)

    def test_softmax_self_test_oracles_and_audits_are_separate_from_timing(self):
        for arm, helper in zip(bench.ARMS, (full.ma11_adapter, full.ma11_adapter)):
            state = dict(protocol=helper.PROTOCOL, installed=True, shape=list(helper.SHAPE),
                original_class_sha256=helper.CLASS_SHA256, original_mask_sha256=helper.MASK_SHA256,
                modules=[f'decoder.layers.{i}.self_attention.core_attention.scale_mask_softmax' for i in range(32)],
                forward_audits=640, backward_audits=640,
                self_test=dict(passed=True, cases=6, forward_bitwise=True,
                               backward_bitwise=True, rng_unchanged=True))
            full.validate_softmax_state(state, arm, 5, 'smoke')
            with self.assertRaises(ValueError):
                full.validate_softmax_state(state, arm, 60, 'timing')
            state.update(forward_audits=0, backward_audits=0, self_test=None)
            full.validate_softmax_state(state, arm, 60, 'timing')

    def test_frozen_controller_is_self_contained_and_detects_adapter_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp); build=parent/'build'; build.mkdir(); root=parent/'fullstep'
            bench.write(build/'selection-plan.json',fixture_plan()['selection'])
            args=argparse.Namespace(root=root,build=build,megatron=parent/'megatron',trials=2,blocks=8)
            libs={arm:dict(root='/synthetic/'+arm,files={}) for arm in bench.ARMS}
            with patch.object(bench,'build_identity',return_value={'tuning':bench.validate_tuning(None)}), \
                 patch.object(bench,'libraries',return_value=libs), \
                 patch.object(full,'snapshot_megatron',return_value={}), contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            full.verify(root,offline=True)
            script="""import ast,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'scripts'))
c=runpy.run_path(str(root/'scripts/fullstep-controller.py'))
p=c['verify'](root,offline=True)
assert p['arms']==['F4-cache-zcount-csoftmax','F4-cache-zcount-csoftmax-localfirst']
assert len(p['schedule'])==32
ast.parse(c['worker_source']())
assert 'ma16_softmax_state' in c['worker_source']()
"""
            result=subprocess.run([sys.executable,'-I','-c',script,str(root)],cwd=root/'scripts',capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            (root/'scripts/ma16_adapter.py').write_text('# drift\n')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):
                full.verify(root,offline=True)


if __name__ == '__main__':
    unittest.main()
