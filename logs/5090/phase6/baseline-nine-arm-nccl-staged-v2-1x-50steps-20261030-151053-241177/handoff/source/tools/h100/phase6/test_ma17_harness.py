"""MA11 identity, isolated frozen replay, exact acceptance and installation gates."""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import ma17_bench as bench
import ma17_fullstep as full
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
                         ("ma11_adapter.install_softmax(raw, ma17_library, MODE)", "ma11_adapter.install_softmax(raw, ma17_library, 'timing')"),
                         ("ma17_adapter.install_direct_allgather(raw, MODE)",
                          "ma17_adapter.install_direct_allgather(raw, 'timing')")]:
            self.assertIn(old, worker)
            with self.assertRaises(ValueError):
                full.validate_baseline_sources(adapter, taco, worker.replace(old, new))

    def test_native_communicator_teardown_is_required_after_evidence(self):
        adapter, taco, worker = full.adapter_source(), full.taco_source(), full.worker_source()
        state = worker.index("report['ma17_direct_allgather_state']")
        close = worker.index('    ma17_direct_allgather_tracker.close()')
        save = worker.index("(OUT/f'rank{rank}.json').write_text")
        self.assertLess(state, close)
        self.assertLess(close, save)
        self.assertLess(worker.index('elapsed=time.perf_counter()-start'), close)
        with self.assertRaisesRegex(ValueError, 'communicator teardown'):
            full.validate_baseline_sources(adapter, taco, worker.replace(
                '    ma17_direct_allgather_tracker.close()', '    pass'))

    def test_all_shared_compute_mechanisms_execute_in_both_arms(self):
        import ctypes
        tree = ast.parse(full.worker_source())
        start = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Import) and
                     any(a.name == 'ma8_adapter' for a in n.names))
        end = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "os.environ['H100_MA17_ARM'] == 'F4-cache-zcount-csoftmax-directag'")
        startup = compile(ast.Module(body=tree.body[start:end+1], type_ignores=[]), 'startup', 'exec')
        for arm in bench.ARMS:
            calls = []
            modules = {name:types.SimpleNamespace(**{method:lambda *a, label=name: calls.append(label) or object()})
                       for name, method in [('ma8_adapter','install_activation_cache'),
                                            ('ma9_adapter','install_zero_count'),('ma11_adapter','install_softmax'),('ma17_adapter','install_direct_allgather')]}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, H100_MA17_ARM=arm,E2E_BUILD='/tmp/build'), \
                 patch.object(ctypes, 'CDLL', return_value=object()):
                exec(startup,dict(os=os,Path=Path,raw=object(),optimizer=object(),POLICY='selective',MODE='timing'))
            self.assertEqual(calls, ['ma8_adapter','ma9_adapter','ma11_adapter'] +
                             (['ma17_adapter'] if arm == bench.ARMS[1] else []))

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

    def test_candidate_nccl_identity_is_stable_across_ranks_and_window_modes(self):
        stable = dict(library_sha256='a' * 64, version=22105, group_ranks=list(range(8)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / 'results' / ('smoke-' + bench.ARMS[1]) / 'rank0.json'
            smoke.parent.mkdir(parents=True)
            full.write(smoke, {'ma17_direct_allgather_state': {'native': stable}})
            before = smoke.read_bytes()
            rows = [{'ma17_direct_allgather_state': {'native': dict(stable, rank=i,
                     device=i, stream=100 + i, linkage_symbol_addresses={'ncclAllGather': 8000 + i})}}
                    for i in range(8)]
            for mode in ('smoke', 'timing', 'profile'):
                self.assertEqual(full.validate_window_nccl_provenance(root, rows,
                    dict(arm=bench.ARMS[1], mode=mode)), stable)
            self.assertEqual(smoke.read_bytes(), before)
            self.assertIsNone(full.validate_window_nccl_provenance(root, [],
                dict(arm=bench.ARMS[0], mode='timing')))

    def test_candidate_nccl_library_or_rank_mapping_drift_is_rejected(self):
        stable = dict(library_sha256='a' * 64, version=22105, group_ranks=list(range(8)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / 'results' / ('smoke-' + bench.ARMS[1]) / 'rank0.json'
            smoke.parent.mkdir(parents=True)
            full.write(smoke, {'ma17_direct_allgather_state': {'native': stable}})
            for key, value in [('library_sha256', 'b' * 64), ('version', 22106),
                               ('group_ranks', [1, 0, 2, 3, 4, 5, 6, 7])]:
                for mode in ('smoke', 'timing', 'profile'):
                    rows = [{'ma17_direct_allgather_state': {'native': copy.deepcopy(stable)}}
                            for _ in range(8)]
                    rows[7]['ma17_direct_allgather_state']['native'][key] = value
                    with self.subTest(key=key, mode=mode), self.assertRaisesRegex(ValueError, 'changed since smoke'):
                        full.validate_window_nccl_provenance(root, rows,
                            dict(arm=bench.ARMS[1], mode=mode))

    def test_missing_smoke_nccl_provenance_does_not_create_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Missing candidate smoke NCCL provenance'):
                full.validate_window_nccl_provenance(root, [], dict(arm=bench.ARMS[1], mode='timing'))
            self.assertEqual(list(root.iterdir()), [])

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
assert p['arms']==['F4-cache-zcount-csoftmax','F4-cache-zcount-csoftmax-directag']
assert len(p['schedule'])==32
ast.parse(c['worker_source']())
assert 'ma17_softmax_state' in c['worker_source']()
"""
            result=subprocess.run([sys.executable,'-I','-c',script,str(root)],cwd=root/'scripts',capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            (root/'scripts/ma17_adapter.py').write_text('# drift\n')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):
                full.verify(root,offline=True)


if __name__ == '__main__':
    unittest.main()
