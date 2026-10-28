"""Exercise prepare.py in an isolated tree; never overwrite frozen experiments."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class GenerationTest(unittest.TestCase):
    def test_both_generated_workers_keep_audited_timing_and_reverse_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = ['tools/a800/phase3/extension_window.py',
                       'tools/a800/phase3/mechanism/worker.py']
            sources += ['tools/a800/e2e/scale/' + name for name in
                        ['adapter.py', 'worker.py', 'summarize.py', 'launch.py',
                         'config.json', 'check_mapping.cu']]
            sources += [f'tools/a800/{folder}/prepare.py' for folder in ('arrival', 'arrival8')]
            for name in sources:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            for folder in ('arrival', 'arrival8'):
                subprocess.run([sys.executable, str(root/f'tools/a800/{folder}/prepare.py'),
                                '--out', str(root/f'fresh/{folder}')],
                               check=True, capture_output=True)
                generated = root/f'fresh/{folder}/e2e-scripts'
                for script in generated.glob('*.py'):
                    compile(script.read_text(), str(script), 'exec')
                worker = (root/f'tools/a800/{folder}/operator_worker.py').read_text()
                self.assertEqual(worker, (ROOT/f'tools/a800/{folder}/operator_worker.py').read_text())
                model = (generated/'worker.py').read_text()
                self.assertIn("report['forward_checks']=adapter.forward_checks", model)
                self.assertIn('full-step-one-clear-v2', model)
                self.assertIn("Path(os.environ.get('ARRIVAL_OUTPUT_ROOT'", (generated/'launch.py').read_text())
                self.assertIn("MAPPING=BUILD/'mapping'", (generated/'launch.py').read_text())
                launch = (generated/'launch.py').read_text()
                section = launch[launch.index("if os.environ.get('ARRIVAL_REVERSE')"):launch.index('CASE=next(')]
                class Env:
                    environ = dict(ARRIVAL_REVERSE='1')
                plan = dict(orders=[['a', 'b'], ['b', 'a']])
                exec(section, dict(os=Env, PLAN=plan))
                self.assertEqual(plan['orders'], [['a', 'b'], ['b', 'a']])
                plan = dict(orders=[['a', 'b', 'c'], ['b', 'c', 'a']])
                exec(section, dict(os=Env, PLAN=plan))
                self.assertEqual(plan['orders'], [['a', 'c', 'b'], ['c', 'b', 'a']])

    def test_each_step_updates_before_clearing_gradients_once(self):
        for folder in ('scale', 'distributed'):
            tree = ast.parse((ROOT/f'tools/a800/e2e/{folder}/worker.py').read_text())
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'step']
            events = []
            context = dict(forward_backward_step=lambda: events.append('backward') or 'loss',
                           update=lambda: (events.append('update') or True, 1.),
                           clear_gradients=lambda: events.append('clear'))
            exec(compile(ast.Module(body=functions, type_ignores=[]), '<step>', 'exec'), context)
            for _ in range(3):
                self.assertEqual(context['step'](), ('loss', True, 1.))
            self.assertEqual(events, ['backward', 'update', 'clear'] * 3)


if __name__ == '__main__':
    unittest.main()
