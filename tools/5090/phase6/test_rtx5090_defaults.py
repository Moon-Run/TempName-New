"""Resource defaults and exact historical build admission, with no GPU work."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from rtx5090_loader import HERE, load
import rtx5090_defaults as defaults


def controller(filename):
    spec = importlib.util.spec_from_file_location('rtx5090_defaults_test_'+Path(filename).stem, HERE/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Parallelism(unittest.TestCase):
    def test_real_nproc_value_is_used(self):
        with patch.object(defaults.subprocess, 'check_output', return_value='48\n') as probe:
            self.assertEqual(defaults.default_build_jobs(), 48)
        self.assertEqual(probe.call_args.args[0], ['nproc'])

    def test_missing_or_invalid_nproc_uses_process_affinity(self):
        for result in (FileNotFoundError('nproc'), '0\n', '-2\n', 'not a number\n'):
            options = {'side_effect':result} if isinstance(result, Exception) else {'return_value':result}
            with self.subTest(result=result), patch.object(defaults.subprocess, 'check_output', **options), \
                    patch.object(defaults.os, 'sched_getaffinity', return_value={2,4,6,8}):
                self.assertEqual(defaults.default_build_jobs(),4)

    def test_final_fallback_never_returns_zero(self):
        for available,expected in ((12,12),(None,1),(0,1)):
            with patch.object(defaults.subprocess,'check_output',side_effect=OSError()), \
                    patch.object(defaults.os,'sched_getaffinity',side_effect=AttributeError()), \
                    patch.object(defaults.os,'cpu_count',return_value=available):
                self.assertEqual(defaults.default_build_jobs(),expected)

    def test_explicit_jobs_override_and_invalid_values(self):
        with patch.object(defaults,'default_build_jobs',return_value=48) as automatic:
            self.assertEqual(defaults.resolve_build_jobs(None),48)
            self.assertEqual(defaults.resolve_build_jobs(7),7)
            automatic.assert_called_once()
        for value in (0,-1,True,False,2.5,'4'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                defaults.resolve_build_jobs(value)

    def test_all_current_5090_launchers_use_nproc_and_respect_override(self):
        filenames = ('baseline.py','flux_baseline.py','op1_a.py','op1_abc.py',
                     'op2_abc.py','op3.py','op4.py','op5.py')
        for filename in filenames:
            value = controller(filename)
            with self.subTest(filename=filename), \
                    patch.object(defaults.subprocess,'check_output',return_value='48\n'):
                self.assertEqual(value.arguments([]).jobs,48)
                self.assertEqual(value.arguments(['--jobs','7']).jobs,7)

    def test_direct_builders_resolve_defaults_before_any_build_work(self):
        import flux_baseline_build, op1_a_build, op1_abc_build, op3_build
        modules = (load('build'),flux_baseline_build,op1_a_build,op1_abc_build,op3_build)
        for value in modules:
            parameters = inspect.signature(value.compile_build).parameters
            self.assertIsNone(parameters['jobs'].default)
            arguments = ['/CPU-TEST/build','/CPU-TEST/megatron','/CPU-TEST/cuda'] if 'megatron' in parameters \
                        else ['/CPU-TEST/build','/CPU-TEST/cuda']
            with self.subTest(module=value.__name__), \
                    patch.object(defaults,'default_build_jobs',return_value=48) as automatic, \
                    patch.object(value,'verify',side_effect=RuntimeError('stop before build')):
                with self.assertRaisesRegex(RuntimeError,'stop before build'):
                    value.compile_build(*arguments)
                automatic.assert_called_once()
                with self.assertRaises(ValueError):
                    value.compile_build(*arguments,jobs=True)


class HistoricalBuildAdmission(unittest.TestCase):
    def test_only_exact_default_change_recipe_pairs_are_admitted(self):
        for filename,(old,new) in defaults.DEFAULT_JOBS_RECIPE_COMPAT.items():
            with self.subTest(filename=filename):
                installed = defaults.RELOCATION_RECIPE_COMPAT.get(filename, ((), new))[1]
                self.assertEqual(hashlib.sha256((HERE/filename).read_bytes()).hexdigest(),installed)
                self.assertTrue(defaults.recipe_matches(HERE/filename,old,new))
                self.assertTrue(defaults.recipe_matches(HERE/filename,new,new))
                self.assertFalse(defaults.recipe_matches(HERE/filename,'0'*64,new))
                self.assertFalse(defaults.recipe_matches(HERE/filename,old,'0'*64))
                current=dict(recipe_sha256=new,native={'cuda':'unchanged'},swizzle={'permutation':'unchanged'})
                saved=dict(current,recipe_sha256=old)
                self.assertTrue(defaults.build_inputs_match(HERE/filename,saved,current))
                self.assertFalse(defaults.build_inputs_match(HERE/filename,dict(saved,native={}),current))
                self.assertFalse(defaults.build_inputs_match(HERE/filename,dict(saved,extra=True),current))
                if filename in defaults.RELOCATION_RECIPE_COMPAT:
                    self.assertTrue(defaults.recipe_matches(HERE/filename,old,installed))
                    self.assertTrue(defaults.recipe_matches(HERE/filename,new,installed))
                    self.assertFalse(defaults.recipe_matches(HERE/filename,'0'*64,installed))
        self.assertFalse(defaults.recipe_matches('unknown.py','a'*64,'b'*64))


if __name__ == '__main__':
    unittest.main()
