"""CPU-only regressions for inherited affinity in direct and frozen workers."""
import ast
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / 'tools'
MASKS = (frozenset(range(192)), frozenset((3, 8, 19, 28, 67, 101, 144, 179)),
         frozenset((7,)))
DIRECT_WORKERS = (
    'a800/phase2/run.py',
    'a800/phase3/run.py',
    'a800/phase3/extension_window.py',
    'a800/phase3/mechanism/worker.py',
    'a800/arrival/operator_worker.py',
    'a800/arrival8/operator_worker.py',
    'a800/e2e/worker.py',
    'a800/e2e/distributed/worker.py',
    'a800/e2e/scale/worker.py',
    'a800/e2e/mlp/model/worker.py',
    'a800/phase4/e2e/worker.py',
    'a800/phase5/e2e/worker.py',
    'a800/phase6/tp8/worker.py',
    'a800/phase6/dp2/worker.py',
    'a800/phase6/dp2/probe.py',
    'a800/phase6/scenario_operator.py',
    'a800/phase6/bench_decode_variants.py',
    'a800/phase6/bench_gemm_tuning.py',
    'a800/phase6/tp8_quant/bench_decoder.py',
    'a800/phase6/tp8_quant/bench_gemm.py',
)


def affinity_setters(tree):
    aliases = {'sched_setaffinity'}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'os':
            aliases.update(alias.asname or alias.name for alias in node.names
                           if alias.name == 'sched_setaffinity')
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
            ((isinstance(node.func, ast.Attribute) and node.func.attr == 'sched_setaffinity') or
             (isinstance(node.func, ast.Name) and node.func.id in aliases))]


def pre_torch_module(tree):
    """Execute the actual launch prefix, without importing Torch or CUDA code."""
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or '']
        else:
            continue
        if any(module.split('.')[0] == 'torch' for module in modules):
            return ast.Module(body=tree.body[:index], type_ignores=[])
    raise AssertionError('Expected an explicit Torch import boundary')


def recorded_expressions(tree, field):
    # Both dict(..., field=value) and report.update(field=value) are supported.
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == field]


class AffinityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.direct = {name: (TOOLS / name).read_text() for name in DIRECT_WORKERS}
        spec = importlib.util.spec_from_file_location(
            '_affinity_dp2_protocol', TOOLS / 'a800/phase6/dp2/protocol.py')
        protocol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol)
        cls.prefix_modules = {'protocol': protocol}
        cls.generated = {}
        for name in ('ma11_fullstep', 'ma14_fullstep', 'ma15_fullstep',
                     'basic_quant_sources', 'mav2_sources'):
            cls.generated[name] = importlib.import_module(name).worker_source()
        distributed = importlib.import_module('tp8_dp2_sources')
        for arm in ('ma11', 'flux-basic', 'megatron-cuda'):
            cls.generated['tp8_dp2_' + arm] = distributed.worker_source(arm)

    def test_all_active_tools_forbid_implicit_cpu_pinning(self):
        """Also catches reintroduced setters inside main/worker/helper functions."""
        offenders = []
        for path in sorted(TOOLS.rglob('*.py')):
            if path.name.startswith('test_') or '__pycache__' in path.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            offenders.extend(f'{path.relative_to(REPO)}:{node.lineno}'
                             for node in affinity_setters(tree))
        self.assertEqual(offenders, [], 'Workers must preserve launcher CPU affinity')

    def assert_prefix_preserves_mask(self, label, source, distributed=False):
        tree = ast.parse(source, filename=label)
        prefix = pre_torch_module(tree)
        code = compile(prefix, label, 'exec')
        guards = {node.value: '1' for node in ast.walk(prefix)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and node.value.endswith('_EXECUTE')}
        for mask in MASKS:
            for local_rank in range(8):
                state = {'mask': set(mask)}

                def set_affinity(pid, requested):
                    state['mask'] = set(requested)

                environment = dict(guards, LOCAL_RANK=str(local_rank),
                                   RANK=str(local_rank + (8 if distributed else 0)),
                                   WORLD_SIZE='16' if distributed else '8',
                                   LOCAL_WORLD_SIZE='8', SLURM_JOB_ID='cpu-affinity-test')
                with self.subTest(worker=label, cpu_count=len(mask), rank=local_rank), \
                     mock.patch.dict(os.environ, environment), \
                     mock.patch.dict(sys.modules, self.prefix_modules), \
                     mock.patch.object(os, 'sched_getaffinity',
                                       side_effect=lambda pid: set(state['mask']), create=True), \
                     mock.patch.object(os, 'sched_setaffinity',
                                       side_effect=set_affinity, create=True) as setter:
                    namespace = {'__file__': str(TOOLS / label),
                                 '__name__': '__affinity_test__'}
                    exec(code, namespace)
                    self.assertEqual(state['mask'], set(mask))
                    setter.assert_not_called()

    def test_direct_workers_preserve_large_restricted_and_single_cpu_masks(self):
        for label, source in self.direct.items():
            self.assert_prefix_preserves_mask(label, source)

    def test_generated_workers_preserve_masks_before_torch_import(self):
        for label, source in self.generated.items():
            with self.subTest(worker=label):
                tree = ast.parse(source, filename=label)
                self.assertEqual(affinity_setters(tree), [])
                self.assert_prefix_preserves_mask(label, source,
                                                  distributed=label.startswith('tp8_dp2_'))

    def test_reports_expose_effective_affinity_and_policy(self):
        sources = dict(self.direct, **self.generated)
        for name in ('ma4_f_bench.py', 'ma20_probe.py'):
            sources[name] = (Path(__file__).parent / name).read_text()
        for label, source in sources.items():
            tree = ast.parse(source, filename=label)
            policy_values = recorded_expressions(tree, 'cpu_affinity_policy')
            affinity_values = recorded_expressions(tree, 'cpu_affinity')
            with self.subTest(worker=label):
                self.assertTrue(policy_values, 'Missing inherited-affinity policy in result')
                self.assertTrue(affinity_values, 'Missing effective CPU set in result')
            for mask in MASKS:
                with self.subTest(worker=label, cpu_count=len(mask)), \
                     mock.patch.object(os, 'sched_getaffinity', return_value=set(mask), create=True):
                    namespace = {'os': os, 'cpu_affinity': sorted(mask)}
                    for value in policy_values:
                        self.assertEqual(eval(compile(ast.Expression(value), label, 'eval'),
                                              namespace), 'inherit')
                    for value in affinity_values:
                        self.assertEqual(eval(compile(ast.Expression(value), label, 'eval'),
                                              namespace), sorted(mask))


if __name__ == '__main__':
    unittest.main()
