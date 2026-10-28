"""CPU-only execution of generated Ablation7 routing, ownership and worker gates."""
import ast
import copy
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

import ablation7_protocol as routing
import ablation7_sources as sources
from ablation_plan import mask_sha
from common import write
import test_scope as scope_tests


class SourceTests(unittest.TestCase):
    def test_all_sources_compile_and_worker_gate_is_independent(self):
        for name, source in [('adapter', sources.adapter_source()), ('taco', sources.taco_source()),
                             ('worker', sources.worker_source()), ('source-only', sources.worker_source(True))]:
            compile(source, '<ablation7-'+name+'>', 'exec')
        with tempfile.TemporaryDirectory(prefix='ablation7-source-gates-') as tmp:
            path = Path(tmp)/'worker.py'
            env = dict(os.environ, H100_ABLATION4_EXECUTE='1')
            env.pop('H100_ABLATION7_EXECUTE', None)
            path.write_text(sources.worker_source())
            attempt = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, text=True)
            self.assertNotEqual(attempt.returncode, 0)
            self.assertIn('GPU worker requires ablation7 --execute', attempt.stderr)
            path.write_text(sources.worker_source(True))
            attempt = subprocess.run([sys.executable, str(path)],
                                     env=dict(env, H100_ABLATION7_EXECUTE='1'), capture_output=True, text=True)
            self.assertNotEqual(attempt.returncode, 0)
            self.assertIn('Source-only ablation7 worker is permanently non-executable', attempt.stderr)

    def test_worker_consumes_and_checks_frozen_10_plus_50_before_cuda(self):
        source = sources.worker_source()
        start, end = source.index('from routing import PROTOCOL, execution_timing'), source.index('CASE = next(')
        self.assertLess(end, source.index('torch.cuda.set_device(local_rank)'))
        check = compile(source[start:end], '<worker-config-gate>', 'exec')
        config = dict(protocol=routing.PROTOCOL, execution_timing=routing.execution_timing(),
                      **routing.execution_timing(), smoke_warmup_steps=1, smoke_steps=2)
        with mock.patch.dict(sys.modules, {'routing': routing}):
            namespace = {'PLAN': config}
            exec(check, namespace)
            self.assertEqual(namespace['execution'], {'warmup_steps': 10, 'timed_steps': 50})
            for key, value in [('protocol', 'h100-gpt67-tp8-ablation4-v1'), ('timed_steps', 60),
                               ('warmup_steps', 20), ('timed_steps', 50.0),
                               ('execution_timing', {'warmup_steps': 10, 'timed_steps': 60})]:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, '50-timed-step'):
                    exec(check, {'PLAN': dict(config, **{key: value})})
        assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id in ('warmup', 'count') for t in n.targets)]
        namespace = dict(SPEC=config, MODE='timing')
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<worker-step-counts>', 'exec'), namespace)
        self.assertEqual((namespace['warmup'], namespace['count']), (10, 50))
        self.assertIn("report['execution_timing'] = execution", source)

    def test_both_roles_use_fixed_fused_v2_for_a_and_b(self):
        class Codec:
            def __init__(self, group, m, n, k, **kwargs):
                self.args = (m, n, k, kwargs)
                self.rank = 0
                self.peers = [types.SimpleNamespace(data_ptr=lambda: id(self))]
        with mock.patch.dict(sys.modules, codec_modules(GemmRSTaco=Codec)):
            namespace = {}
            # No __file__ or selection plan exists: A/B must not read the C mask.
            exec(sources.taco_source(), namespace)
            ops = []
            for policy in routing.POLICIES:
                if not routing.is_fused_v2(policy):
                    continue
                for role, (m, n, k) in routing.SHAPES.items():
                    op = namespace['TacoForward'](None, m, n, k//8, policy, role)
                    ops.append(op)
                    self.assertEqual(op.codec.args, (m, n, k//8, {'placement': 'fused'}))
                    self.assertFalse(op.owns_output)
                    self.assertIsNone(op.plan_sha256)
                    self.assertEqual(op.workspace_info, [dict(shape=[m, n, k], packet_ptr=id(op.codec))])
            self.assertEqual(len(ops), 4)
            self.assertEqual(len({id(op.codec) for op in ops}), 4)

    def test_c_receives_separate_role_masks_and_two_owned_workspaces(self):
        class Codec:
            owns_configuration = True
            def __init__(self, group, m, n, k, **kwargs):
                self.kwargs = kwargs
                self.slots = [types.SimpleNamespace(m=m, n=n, k_local=k, rank=0,
                    selection_sha256=mask_sha(kwargs['selected']),
                    peers=[types.SimpleNamespace(data_ptr=lambda i=i: id(self)*10+i)],
                    selected=types.SimpleNamespace(data_ptr=lambda i=i: id(self)*10+i+2)) for i in range(2)]
        policy = next(p for p in routing.POLICIES if routing.definition(p)['family'] == 'C')
        records = {}
        for offset, (role, shape) in enumerate(routing.SHAPES.items()):
            mask = [[int(t//64 != src and offset <= t % 64 < offset+4) for t in range(512)]
                    for src in range(8)]
            self.assertEqual(list(map(sum, mask)), [28]*8)
            entry = dict(shape=shape, K_local=shape[2]//8, mask=mask)
            records[role] = dict(role=role, entry=entry, entry_sha256=routing.digest(entry))
        plan = dict(protocol=routing.PROTOCOL,
                    variants={policy: dict(definition=routing.definition(policy), roles=records)})
        with tempfile.TemporaryDirectory(prefix='ablation7-codec-') as tmp, \
                mock.patch.dict(sys.modules, codec_modules(GemmRSTacoDoubleBuffered=Codec)):
            path = Path(tmp)
            write(path/'selection-plan.json', plan)
            namespace = {'__file__': str(path/'taco_support.py')}
            exec(sources.taco_source(), namespace)
            ops = []
            for role, (m, n, k) in routing.SHAPES.items():
                op = namespace['TacoForward'](None, m, n, k//8, policy, role)
                ops.append(op)
                self.assertTrue(op.owns_output)
                self.assertEqual(op.codec.kwargs, dict(placement='fused',
                    selected=records[role]['entry']['mask'], quantization_mode='targeted'))
                self.assertEqual(len(op.workspace_info), 2)
                self.assertTrue(all(slot['shape'] == [m, n, k] for slot in op.workspace_info))
                self.assertTrue(all(slot['mask_sha256'] == mask_sha(records[role]['entry']['mask'])
                                    for slot in op.workspace_info))
            for field in ('packet_ptr', 'mask_ptr'):
                pointers = [slot[field] for op in ops for slot in op.workspace_info]
                self.assertEqual(len(set(pointers)), 4)
            saved = copy.deepcopy(records['attn']['entry']['mask'])
            for invalid in ('too_many', 'local', 'shape', 'not_binary'):
                mask = copy.deepcopy(saved)
                if invalid == 'too_many': mask[0][68] = 1
                elif invalid == 'local': mask[0][0], mask[0][64] = 1, 0
                elif invalid == 'shape': mask[0].pop()
                else: mask[0][64] = 2
                records['attn']['entry']['mask'] = mask
                records['attn']['entry_sha256'] = routing.digest(records['attn']['entry'])
                write(path/'selection-plan.json', plan)
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, '28/448'):
                    namespace['TacoForward'](None, 2048, 4096, 512, policy, 'attn')

    def test_a_b_clone_outputs_c_preserves_owned_output(self):
        node = next(n for n in ast.parse(sources.adapter_source()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'GemmRSFunction')
        namespace = dict(torch=types.SimpleNamespace(autograd=types.SimpleNamespace(Function=object)),
            parallel_state=types.SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<autograd-adapter>', 'exec'), namespace)
        class Value:
            shape = (2048, 1, 512)
            def __init__(self): self.clones = 0
            def reshape(self, *shape): return self
            def contiguous(self): return self
            def clone(self): self.clones += 1; return self
        for policy in routing.POLICIES:
            value = Value()
            op = types.SimpleNamespace(owns_output=not routing.is_fused_v2(policy), forward=lambda *a, **kw: value)
            namespace['GemmRSFunction'].forward(types.SimpleNamespace(save_for_backward=lambda *a: None),
                value, types.SimpleNamespace(shape=(4096, 512)), op, None)
            self.assertEqual(value.clones, int(routing.is_fused_v2(policy)))

    def test_64_projection_routes_survive_audit_for_50_steps(self):
        # Reuse the established mock Megatron model while executing the real new
        # adapter; each full step has four microbatches, hence 200 timed calls.
        fixture = textwrap.dedent(inspect.getsource(scope_tests.AdapterTests.exercise))
        fixture = fixture.replace('owns_output = True', 'owns_output = not sp.is_fused_v2(policy)')
        anchor = "            self.plan_sha256 = role+'-plan'"
        fixture = fixture.replace(anchor, anchor+"\n            if sp.is_fused_v2(policy): self.plan_sha256 = None")
        anchor = '        def forward(self, x, w):'
        fixture = fixture.replace(anchor,
            "            if sp.is_fused_v2(policy): self.workspace_info = [dict(shape=sp.SHAPES[role], packet_ptr=id(self))]\n"+anchor)
        fixture = fixture.replace('range(2):', 'range(50*4):').replace('expected*2)', 'expected*(50*4))')
        fixture = fixture.replace('set(adapter.timed_calls.values()), {2}', 'set(adapter.timed_calls.values()), {200}')
        namespace = dict(scope_tests.__dict__, sp=routing, sources=sources)
        exec(fixture, namespace)
        for policy in routing.POLICIES:
            for mode in ('timing', 'profile'):
                with self.subTest(policy=policy, mode=mode):
                    namespace['exercise'](self, policy, mode)


def codec_modules(**codecs):
    torch = types.ModuleType('torch')
    torch.no_grad = lambda: lambda f: f
    dist = types.ModuleType('torch.distributed')
    dist.get_world_size = lambda g: 8
    torch.distributed = dist
    return {'torch': torch, 'torch.distributed': dist, 'routing': routing,
            'reference': types.SimpleNamespace(roundtrip=lambda x: x), 'flux': types.ModuleType('flux'),
            'flux.gemm_rs_taco': types.SimpleNamespace(**codecs)}


if __name__ == '__main__':
    unittest.main()
