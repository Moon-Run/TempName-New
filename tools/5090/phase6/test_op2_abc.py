"""CPU integration evidence for three optimized workers; never GPU results."""
import ast
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import copy
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rtx5090_loader import HERE, H100, load
import op2_abc as controller
import op2_abc_protocol as p
import op2_abc_sources as sources
import op2_abc_build as optimizers
import op2_abc_validation as validation
import op2_abc_profile as profile
import op2_abc_probe as probe
from test_op1_abc_validation import fixture as abc_fixture
from test_baseline_workers import fixture as native_fixture
from test_op1_abc_worker_reporting import WorkerLifecycle, reporting_program


def library_plan(root):
    result = {}
    for name in ('ma9', 'ma11'):
        path = root/(name+'.so')
        path.write_bytes(b'CPU-FIXTURE-'+name.encode())
        result[name+'_compat'] = dict(library=str(path), library_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def fixture(arm, mode='timing'):
    rows, job, plan, worker = abc_fixture(arm, mode)
    ma11_rows = native_fixture('ma11', mode)[0]
    plan.update(protocol=p.PROTOCOL, ma9_compat=dict(library='/CPU/ma9.so', library_sha256='a'*64),
                ma11_compat=dict(library='/CPU/ma11.so', library_sha256='b'*64))
    worker['selection'] = sources.selection_metadata()
    worker['definition'] = sources.definition(arm)
    worker['config'].update(protocol=p.PROTOCOL, scope_definition=worker['definition'],
        op2_implementation=sources.implementation(arm), op2_optimization_libraries=optimizers.libraries(plan))
    for row, native in zip(rows, ma11_rows):
        row.update(protocol=p.PROTOCOL, scope_definition=worker['definition'],
            op2_implementation=sources.implementation(arm), op2_optimization_libraries=optimizers.libraries(plan),
            **{field: copy.deepcopy(native[field]) for field in validation.FIELDS})
    return rows, job, plan, worker


class Protocol(unittest.TestCase):
    def test_three_arms_short_run_and_full_counterbalance(self):
        self.assertEqual(len(p.ARMS), 3)
        self.assertEqual(len(p.schedule()), 6)
        self.assertEqual({j['block'] for j in p.schedule()}, {0, 1})
        self.assertEqual([j['mode'] for j in p.jobs()[:6]], ['smoke']*3+['profile']*3)
        orders = [tuple(j['arm'] for j in p.schedule(blocks=6) if j['block'] == b) for b in range(6)]
        self.assertEqual(set(orders), set(itertools.permutations(p.ARMS)))
        for position in range(3):
            self.assertEqual(Counter(o[position] for o in orders), Counter(dict.fromkeys(p.ARMS, 2)))
        for count in (0, 1, 4, 8, True):
            with self.assertRaises(ValueError):
                p.schedule(blocks=count)

    def test_report_has_three_pairs_and_never_claims_incremental_gain(self):
        for blocks in (2, 6):
            samples = [dict(j, ms_per_step=900+20*p.ARMS.index(j['arm']), peak_allocated_gib=20,
                peak_reserved_gib=22, native_workspace_gib=0, peak_allocated_plus_taco_workspace_gib=20)
                for j in p.schedule(blocks=blocks)]
            report = p.report(samples, blocks=blocks)
            self.assertEqual(report['timing_windows'], blocks*3)
            self.assertEqual(report['formal_optimizer_steps'], blocks*150)
            self.assertFalse(report['increment_over_unoptimized_validated'])
            pairs = report['comparisons']['compare1']['pairs']
            self.assertEqual(len(pairs), 3)
            self.assertEqual(all(v['ci95'] is None for v in pairs.values()), blocks == 2)
            self.assertIn('3 组', p.markdown(report))
            self.assertNotIn('改进量化', p.markdown(report))
            with self.assertRaises(ValueError):
                p.report(samples[:-1], blocks=blocks)


class Sources(unittest.TestCase):
    def test_actual_worker_setup_and_final_state_lifecycle(self):
        for arm, mode in itertools.product(p.ARMS, ('smoke', 'profile', 'timing')):
            with self.subTest(arm=arm, mode=mode), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                worker = WorkerLifecycle(arm, mode, root)
                namespace = worker.namespace()
                plan = library_plan(root)
                namespace['PLAN'].update(op2_optimization_libraries=optimizers.libraries(plan),
                                         op2_implementation=sources.implementation(arm))
                namespace.update(hashlib=hashlib, optimizer=object())
                calls = []
                modules = {name: ModuleType(name) for name in ('ma8_adapter', 'ma9_adapter', 'ma11_adapter', 'ctypes')}
                modules['ctypes'].CDLL = lambda path: path
                def install_cache(raw, arm):
                    calls.append(('cache', worker.steps))
                    return {'installed': True}
                def install_counter(optimizer, library, mode):
                    calls.append(('zero', worker.steps))
                    return SimpleNamespace(state=lambda: dict(calls=worker.steps))
                def install_softmax(raw, library, mode):
                    calls.append(('softmax', worker.steps))
                    return SimpleNamespace(state=lambda: dict(forward_steps=worker.steps, backward_steps=worker.steps))
                modules['ma8_adapter'].install_activation_cache = install_cache
                modules['ma9_adapter'].install_zero_count = install_counter
                modules['ma11_adapter'].install_softmax = install_softmax
                stack.enter_context(patch.dict(sys.modules, modules))
                text = sources.worker_source(arm)
                tree = ast.parse(text)
                begin = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Import)
                             and any(a.name == 'ctypes' for a in n.names))
                end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'ma11_softmax_tracker' for t in n.targets))
                exec(compile(ast.Module(body=tree.body[begin:end+1], type_ignores=[]), '<op2-real-setup>', 'exec'), namespace)
                exec(reporting_program(text), namespace)
                report = json.loads((root/'rank0.json').read_text())
                self.assertEqual(calls, [('cache', 0), ('zero', 0), ('softmax', 0)])
                self.assertEqual(report['ma11_zero_count_state']['calls'], worker.steps)
                self.assertEqual(report['ma11_softmax_state']['forward_steps'], worker.steps)
                self.assertEqual(len(report['op1_abc_transport']), 64)
                self.assertEqual(report['op2_optimization_libraries'], optimizers.libraries(plan))
                sources.base.validate_worker(text)
                with self.assertRaisesRegex(ValueError, 'exactly once'):
                    sources.validate_installation(text+'\nma8_adapter.install_activation_cache(raw)\n')

    def test_packages_contain_exact_helpers_and_only_three_workers(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            native = root/'parent'
            native.mkdir()
            plan = dict(environment={'torch': optimizers.compat.TORCH}, communication={}, **library_plan(root))
            stack.enter_context(patch.object(sources.builder, 'codec_library', return_value={'path': '/CPU/codec', 'sha256': 'c'*64}))
            stack.enter_context(patch.object(sources.builder, 'native_root', return_value=native))
            stack.enter_context(patch.object(sources.builder, 'libraries', return_value={}))
            workers = sources.prepare_workers(root, root, plan)
            self.assertEqual(set(workers), set(p.ARMS))
            for arm, worker in workers.items():
                scripts = Path(worker['root'])
                self.assertEqual((scripts/'ma8_adapter.py').read_bytes(), (H100/'ma8_adapter.py').read_bytes())
                self.assertEqual((scripts/'ma9_adapter.py').read_bytes(), (H100/'ma9_adapter.py').read_bytes())
                namespace = {}
                exec((scripts/'routing.py').read_text(), namespace)
                self.assertEqual(namespace['POLICIES'], p.ARMS)
                self.assertEqual(namespace['PROTOCOL'], p.PROTOCOL)
                with self.assertRaises(ValueError):
                    namespace['definition']('flux-improved')
                cfg = json.loads((scripts/'config.json').read_text())
                self.assertEqual(cfg['op2_optimization_libraries'], optimizers.libraries(plan))
                self.assertIn(optimizers.compat.TORCH, (scripts/'ma11_adapter.py').read_text())
                self.assertEqual((scripts/'adapter.py').read_text(), sources.original.adapter_source(arm))
                for name, digest in worker['files'].items():
                    self.assertEqual(hashlib.sha256((scripts/name).read_bytes()).hexdigest(), digest)


class Admission(unittest.TestCase):
    def test_all_three_paths_all_modes_require_every_optimization(self):
        for arm, mode in itertools.product(p.ARMS, ('smoke', 'profile', 'timing')):
            with self.subTest(arm=arm, mode=mode):
                values = fixture(arm, mode)
                self.assertEqual(len(validation.validate_rows(*values)), 8)
                for field in validation.FIELDS:
                    rows, job, plan, worker = copy.deepcopy(values)
                    del rows[0][field]
                    with self.assertRaises(ValueError):
                        validation.validate_rows(rows, job, plan, worker)

    def test_wrong_library_stale_counter_missing_cache_and_changed_softmax_fail(self):
        for mutate in (
            lambda row: row['op2_optimization_libraries']['ma9'].update(sha256='f'*64),
            lambda row: row['ma11_zero_count_state'].update(calls=1),
            lambda row: row['ma11_activation_setup'].update(installed=False),
            lambda row: row['ma11_softmax_state'].update(forward_audits=0),
            lambda row: row.update(op1_abc_transport={}),
        ):
            rows, job, plan, worker = fixture(p.ARMS[0], 'smoke')
            mutate(rows[0])
            with self.assertRaises(ValueError):
                validation.validate_rows(rows, job, plan, worker)
        self.assertEqual(validation.original.p.PROTOCOL, sources.original.p.PROTOCOL)


class NativeBuild(unittest.TestCase):
    def test_standalone_ma9_is_compiled_from_identical_header_and_pinned(self):
        nvcc = shutil.which('nvcc') or '/data/apps/cuda/12.8/bin/nvcc'
        if not Path(nvcc).is_file():
            self.skipTest('No local nvcc')
        cuda = Path(nvcc).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = optimizers.prepare(root, cuda, optimizers.compat.TORCH)
            self.assertFalse(record['gpu_validated'])
            self.assertEqual(optimizers.verify(record, root, optimizers.compat.TORCH), record)
            path = Path(record['library'])
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):
                optimizers.verify(record, root, optimizers.compat.TORCH)


class Profiles(unittest.TestCase):
    @staticmethod
    def row_fixture():
        from test_baseline_profile_nccl import current_column
        events = current_column(row=True)
        removed = {e['args']['correlation'] for e in events if e['cat'] == 'kernel' and e['name'] == 'reduce_kernel'}
        events = [e for e in events if e.get('args', {}).get('correlation') not in removed]
        for event in events:
            event['name'] = event['name'].replace('ConsumerGemmRSFunction', 'GemmRSFunction').replace('.candidate', '.flux_tensor')
            if event['cat'] in ('kernel', 'gpu_memset', 'gpu_memcpy'):
                event['args']['device'] = 0
        return events

    def test_abc_backward_checks_its_own_autograd_and_cuda_dependencies(self):
        events = self.row_fixture()
        record = profile.row_backward(events, 0)
        self.assertEqual({role: data['calls'] for role, data in record.items()}, {'attn': 128, 'mlp': 128})
        for mutation in ('gather', 'autograd', 'stream', 'device'):
            changed = copy.deepcopy(events)
            if mutation == 'autograd':
                next(e for e in changed if e['name'] == 'GemmRSFunctionBackward')['name'] = 'ConsumerGemmRSFunctionBackward'
            else:
                event = next(e for e in changed if e['cat'] == 'kernel' and 'AllGather' in e['name'])
                if mutation == 'gather':
                    event['ts'] += 100
                else:
                    event['args'][mutation] = 99
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.row_backward(changed, 0)

    def test_composed_profile_requires_all_five_paths(self):
        with ExitStack() as stack:
            validators = [(profile.projection, 'chunk_records'), (profile, 'row_backward'),
                (profile.base.ma8_profile, 'trace_activation_cache_records'),
                (profile.base.ma9_profile, 'trace_zero_count_records'),
                (profile.base.ma11_profile, 'trace_attention_softmax_records')]
            mocks = [stack.enter_context(patch.object(module, name, return_value={'checked': name}))
                     for module, name in validators]
            result = profile.analyze([], p.ARMS[0], 0)
            self.assertTrue(result['execution_validated'])
            for mock in mocks:
                mock.assert_called_once()
            for mock in mocks:
                mock.side_effect = ValueError('Missing actual optimization')
                with self.assertRaisesRegex(ValueError, 'actual optimization'):
                    profile.analyze([], p.ARMS[0], 0)
                mock.side_effect = None


class Launcher(unittest.TestCase):
    def test_preview_default_is_only_three_paths_and_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            proc = subprocess.run(['bash', str(HERE/'run_op2_abc.sh'), '--out', str(out),
                '--abc-build', str(Path(tmp)/'absent-build')], text=True, capture_output=True,
                env=dict(os.environ, RTX5090_PYTHON=sys.executable))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            record = json.loads(proc.stdout[proc.stdout.index('{'):])
            self.assertEqual(record['arms'], list(p.ARMS))
            self.assertEqual(record['timing_windows'], 6)
            self.assertEqual(record['formal_optimizer_steps'], 300)
            self.assertFalse(out.exists())

    def test_native_probe_launches_only_abc(self):
        import tp8_dp2_train
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            controller.write(root/'environment-preflight.json', {})
            args = SimpleNamespace(out=root, python=Path(sys.executable), worker_timeout=3600)
            launched = []
            for name in ('verify_controllers', 'gpu_idle'):
                stack.enter_context(patch.object(controller.base, name))
            stack.enter_context(patch.object(controller, 'worker_environment', return_value={}))
            stack.enter_context(patch.object(tp8_dp2_train, 'run_process', side_effect=lambda cmd, *a: launched.append(cmd)))
            stack.enter_context(patch.object(probe, 'check_records', return_value={'completed': True}))
            stack.enter_context(patch.object(probe, 'native_evidence'))
            with redirect_stdout(io.StringIO()):
                probe.native_checks(args, root/'parent-build', {})
            self.assertEqual([cmd[cmd.index('--arm')+1] for cmd in launched], list(p.ARMS))
            self.assertEqual({path.name for path in (root/'native-probe').iterdir() if path.is_dir()}, set(p.ARMS))

    def test_supervisor_runs_small_library_gates_then_three_smokes_profiles_and_six_windows(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            build = root/'build'
            build.mkdir()
            environment = {'torch': optimizers.compat.TORCH}
            controller.write(build/'op1-abc-build.json', {'environment': environment})
            controller.write(root/'batch-state.json', {})
            args = controller.arguments(['--out', str(root), '--abc-build', str(build), '--execute'])
            modules = {name: ModuleType(name) for name in ('op1_abc_build', 'op2_abc_sources', 'op2_abc_profile', 'run', 'tp8_dp2_train')}
            modules['op1_abc_build'].verify = lambda *a: None
            modules['op1_abc_build'].base_root = lambda *a: root
            for name in ('worker_source', 'adapter_source', 'taco_source'):
                setattr(modules['op2_abc_sources'], name, lambda arm: 'pass')
            modules['op2_abc_sources'].prepare_workers = lambda *a: dict.fromkeys(p.ARMS, {})
            modules['op2_abc_sources'].definition = sources.definition
            modules['op2_abc_profile'].analyze = lambda events, arm, rank: dict(execution_validated=True, arm=arm, rank=rank)
            modules['run'].arguments = lambda *a: []
            calls, setup = [], []
            def run(command, cwd, env, log, timeout, peer):
                dest = Path(log).parent
                job = controller.read(dest/'command.json')['job']
                calls.append(job)
                if job['mode'] == 'profile':
                    for rank in range(8):
                        with gzip.open(dest/f'rank{rank}-profile.json.gz', 'wt') as handle:
                            json.dump({'traceEvents': []}, handle)
            modules['tp8_dp2_train'].run_process = run
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(controller, 'REPO', root))
            for name in ('verify_controllers', 'gpu_idle', 'verify_ma11_probe'):
                stack.enter_context(patch.object(controller.base, name))
            stack.enter_context(patch.object(controller.base, 'check_worker', side_effect=lambda root, arm, worker: root/'workers'/arm/'scripts'))
            def hardware(args):
                record = dict(environment=environment, communication={}, cuda_visible_devices='0,1,2,3,4,5,6,7')
                controller.write(root/'environment-preflight.json', record)
                return record
            stack.enter_context(patch.object(controller.base, 'hardware_preflight', side_effect=hardware))
            stack.enter_context(patch.object(controller.load('build'), 'environment_identity', return_value=environment))
            records = library_plan(root)
            stack.enter_context(patch.object(optimizers, 'prepare', side_effect=lambda *a: setup.append('ma9') or records['ma9_compat']))
            stack.enter_context(patch.object(optimizers.compat, 'prepare', side_effect=lambda *a: setup.append('ma11') or records['ma11_compat']))
            stack.enter_context(patch.object(controller.base, 'ma11_gpu_preflight', side_effect=lambda *a: setup.append('probe') or {'fixture': True}))
            stack.enter_context(patch.object(optimizers, 'verify_all'))
            stack.enter_context(patch.object(controller, 'native_checks'))
            stack.enter_context(patch.object(controller, 'worker_environment', return_value={}))
            stack.enter_context(patch.object(controller, 'window', return_value=[]))
            stack.enter_context(patch.object(controller, 'check_identity', return_value='CPU-FIXTURE'))
            def sample(rows, job):
                return dict(job, ms_per_step=900, peak_allocated_gib=20, peak_reserved_gib=22,
                            native_workspace_gib=0, peak_allocated_plus_taco_workspace_gib=20)
            stack.enter_context(patch.object(p, 'summarize', side_effect=sample))
            result = p.report([sample([], j) for j in p.schedule()])
            stack.enter_context(patch.object(controller, 'replay', return_value=result))
            with redirect_stdout(io.StringIO()):
                controller.supervise(args)
            self.assertEqual(setup, ['ma9', 'ma11', 'probe'])
            self.assertEqual([j['mode'] for j in calls], ['smoke']*3+['profile']*3+['timing']*6)
            self.assertEqual({j['arm'] for j in calls}, set(p.ARMS))
            self.assertEqual(controller.read(root/'batch-state.json')['status'], 'completed')
            self.assertEqual(controller.read(root/'report.json')['formal_optimizer_steps'], 300)
            self.assertEqual(len(controller.read(root/'profile-summary.json')['profiles']), 24)


if __name__ == '__main__':
    unittest.main()
