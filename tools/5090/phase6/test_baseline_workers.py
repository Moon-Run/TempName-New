"""CPU-only synthetic records: no fixture value is a GPU measurement."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rtx5090_loader import load

p = load('protocol')
s = load('sources')
v = load('validation')
profiles = load('profile')
from nccl_transport import wire_layout, SCRATCH_POLICY, SHARED_BUFFERS, PACKET_BYTES_PER_SOURCE
from test_ma2_cr2 import fixture_plan
from test_baseline_new import fixture_rows as old_fixture_rows


def fixture(arm, mode='timing'):
    selection = fixture_plan()['selection']
    codec = {'path': '/CPU-FIXTURE/basic.so', 'sha256': 'a'*64} if arm == 'flux-basic' else None
    config = s.configuration(arm, selection, {'environment': dict(p.TRANSPORT_ENV), 'CPU_FIXTURE': True}, codec)
    worker = dict(config=config, definition=s.definition(arm), selection=selection,
                  files={'config.json': 'b'*64}, libraries={})
    plan = dict(run_id='CPU', transport=p.TRANSPORT, communication=config['communication'], environment=dict(
        torch='2.7.0+cu128', cuda='12.8', nccl=[2, 26, 2], python='/CPU-FIXTURE/python'))
    job = dict(arm=arm, mode=mode, block=0, index=0, trial=1)
    old_arm = arm if arm in ('ma11', 'ma89', 'flux-basic') else 'ma89' if arm in ('ma8', 'ma9') else 'cr3'
    rows = old_fixture_rows(dict(job, arm=old_arm), plan, worker)
    names = set(rows[0]['target_modules'])
    for rank, row in enumerate(rows):
        row.update(protocol=p.PROTOCOL, transport=p.TRANSPORT, arm=arm, policy=p.POLICY[arm], model=p.MODEL, **plan['environment'])
        row['hardware'] = dict(host='CPU-FIXTURE', local_rank=rank, devices=[dict(
            name='NVIDIA GeForce RTX 5090', uuid=f'CPU-FIXTURE-{i}', capability=[12, 0], sm_count=170)
            for i in range(8)], transport=p.TRANSPORT, peer_access=[[i == j for j in range(8)] for i in range(8)])
        row['rtx5090_transport'] = {}
        row['native_workspace_bytes'] = 0
        warmup, count = (2, 3) if mode == 'smoke' else (10, 1 if mode == 'profile' else 50)
        calls = (warmup+count)*4
        if arm != 'flux-basic':
            for name in names:
                role = 'attn' if name.endswith('.self_attention.linear_proj') else 'mlp'
                mask = selection['variants'][p.POLICY[arm]]['roles'][role]['entry']['mask'] if arm in p.QUANT_SELECTIVE else None
                layout = wire_layout(mask, rank)
                double = arm in p.QUANT_SELECTIVE
                per_slot = calls//2 if double else calls
                initializations = 1 if arm in p.CR3_FAMILY else per_slot
                consumers = per_slot if arm in p.CR3_FAMILY else 0
                reconfigure = per_slot if arm == 'a' else 0
                slot = dict(protocol=p.TRANSPORT, native_api='rtx5090_local_v1', calls=per_slot,
                            producer_calls=per_slot, pack_calls=per_slot, exchange_calls=per_slot,
                            wait_calls=per_slot, unpack_calls=per_slot, decode_calls=per_slot,
                            consumer_calls=consumers, initializations=initializations,
                            reconfigure_calls=reconfigure, configuration_scope='per_call' if arm == 'a' else 'instance',
                            failed=False, pending_exchange=False, closed=False, native_phase=0, device=rank,
                            stream=1, peer_memory=False, native_workspace_bytes=0,
                            send_split_bytes=layout['tx_splits'],
                            receive_split_bytes=layout['rx_splits'], send_bytes=layout['tx_bytes'],
                            receive_bytes=layout['rx_bytes'],
                            synchronization='fixed_stream_producer_NCCL_work_wait_decode_reuse',
                            native=dict(protocol_version=1, initializations=initializations,
                                        producer_calls=per_slot, decode_calls=per_slot,
                                        bias_residual_calls=consumers, phase=0, failed=False,
                                        bound_k=512 if role == 'attn' else 2048, reconfigure_calls=reconfigure,
                                        workspace_bytes=0))
                row['rtx5090_transport'][name] = (dict(protocol=p.TRANSPORT, failed=False, next_slot=0,
                    slots=[dict(copy.deepcopy(slot), slot=i) for i in range(2)]) if double else slot)
                record = row['rtx5090_transport'][name]
                for i, value in enumerate(record.get('slots', [record])):
                    base = 10000 + (1000 if role == 'mlp' else 0) + 100*i
                    value.update(scratch_policy=SCRATCH_POLICY,
                                 shared_scratch_ptrs={kind: base+j for j, kind in enumerate(SHARED_BUFFERS)},
                                 shared_scratch_bytes=2*8*256*4096*2 + 8*PACKET_BYTES_PER_SOURCE
                                 + layout['tx_bytes'] + layout['rx_bytes'])
        if arm in ('ma8', 'ma9'):
            for suffix in ('consumer_state', 'submission_state'):
                row[arm+'_'+suffix] = row.pop('ma11_'+suffix)
            cache, zero = row.pop('ma11_activation_setup'), row.pop('ma11_zero_count_state')
            row.pop('ma11_softmax_state')
            row['ma8_activation_setup' if arm == 'ma8' else 'ma9_zero_count_state'] = cache if arm == 'ma8' else zero
        if arm in ('r', 'c', 'a'):
            row.pop('ma4_f_consumer_state')
            row.pop('ma4_f_submission_state')
            row['ma2c_arm'] = p.NATIVE_ARM[arm]
            slots = 2 if arm == 'c' else 1
            row['ma2c_codec_state'] = {name: dict(owns_output=arm != 'a', owns_configuration=arm != 'a',
                                                 workspaces=slots) for name in names}
            row['workspace_checks'] = {}
            for i, name in enumerate(sorted(names)):
                role = 'attn' if name.endswith('.self_attention.linear_proj') else 'mlp'
                values = [dict(shape=[2048, 4096, 4096 if role == 'attn' else 16384], packet_ptr=100+i*2+j)
                          for j in range(slots)]
                if arm == 'c':
                    mask = selection['variants'][p.POLICY[arm]]['roles'][role]['mask_sha256']
                    for value in values:
                        value.update(mask_ptr=value['packet_ptr']+1000, mask_sha256=mask)
                row['workspace_checks'][name] = values
    return rows, job, plan, worker


class Workers(unittest.TestCase):
    def _run_worker_hardware_startup(self, arm, rank, mutate=None, environment=None):
        devices = [SimpleNamespace(name='NVIDIA GeForce RTX 5090', major=12, minor=0,
                                   multi_processor_count=170, uuid=f'CPU-FIXTURE-{i}') for i in range(8)]
        cuda = SimpleNamespace(device_count=lambda: len(devices),
                               get_device_properties=lambda index: devices[index],
                               can_device_access_peer=lambda source, destination: False,
                               set_device=lambda index: None)
        if mutate is not None:
            mutate(cuda, devices)
        torch = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())))
        runtime = ModuleType('runtime')
        runtime.TRANSPORT, runtime.TRANSPORT_ENV = p.TRANSPORT, p.TRANSPORT_ENV
        exec(s.runtime_source(), runtime.__dict__)
        source = s.worker_source(arm)
        # Execute the shared hardware gate; the basic codec's separate GPU
        # self-test precedes this region and is outside this CPU fixture.
        startup = source[source.index('assert torch.cuda.device_count() == WORLD'):
                         source.index('from megatron.training.initialize')]
        namespace = dict(torch=torch, WORLD=8, local_rank=rank)
        env = dict(p.TRANSPORT_ENV, RTX5090_TRANSPORT=p.TRANSPORT, LOCAL_RANK=str(rank))
        env.update(environment or {})
        with patch.dict(os.environ, env), patch.dict(sys.modules, {'torch': torch, 'runtime': runtime}):
            exec(compile(startup, f'{arm}/worker-startup.py', 'exec'), namespace)
        return namespace['hardware']

    def test_all_nine_worker_startups_accept_zero_cross_gpu_peer_links(self):
        for arm in p.ARMS:
            for rank in range(8):
                with self.subTest(arm=arm, rank=rank):
                    hardware = self._run_worker_hardware_startup(arm, rank)
                    self.assertEqual(hardware['local_rank'], rank)
                    self.assertEqual(hardware['peer_access'], [[i == j for j in range(8)] for i in range(8)])
                    self.assertEqual(hardware['transport'], p.TRANSPORT)
            # The original H100 template still has its direct-peer requirement.
            self.assertIn('assert all(torch.cuda.can_device_access_peer', s.parent(arm).worker_source())
            with self.assertRaisesRegex(ValueError, 'peer access must remain diagnostic'):
                s.validate_worker(s.worker_source(arm)+'\nassert torch.cuda.can_device_access_peer(0, 1)\n')

    def test_worker_startup_still_rejects_wrong_hardware_and_transport(self):
        mutations = [lambda cuda, devices: setattr(devices[0], 'major', 9),
                     lambda cuda, devices: setattr(devices[0], 'name', 'NVIDIA H100'),
                     lambda cuda, devices: setattr(devices[0], 'multi_processor_count', 132),
                     lambda cuda, devices: setattr(devices[1], 'uuid', devices[0].uuid),
                     lambda cuda, devices: devices.pop()]
        for arm in p.ARMS:
            for mutate in mutations:
                with self.subTest(arm=arm, mutate=mutate), self.assertRaises((RuntimeError, AssertionError)):
                    self._run_worker_hardware_startup(arm, 0, mutate=mutate)
            for environment in ({'NCCL_P2P_DISABLE': '0'}, {'RTX5090_TRANSPORT': 'wrong-protocol'}):
                with self.subTest(arm=arm, environment=environment), self.assertRaises(RuntimeError):
                    self._run_worker_hardware_startup(arm, 0, environment=environment)

    def test_python312_ast_schema_preserves_frozen_training_hashes(self):
        parse = ast.parse

        def parse312(*args, **kwargs):
            tree = parse(*args, **kwargs)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if 'type_params' not in node._fields:
                        node._fields = (*node._fields, 'type_params')
                    node.type_params = []
            return tree

        # Exercise the target schema even when CI runs Python 3.10. The old
        # direct ast.dump comparison must fail; all nine generated workers
        # must pass after normalization, without accepting real code changes.
        class_fields = ast.FunctionDef._fields
        with patch.object(ast, 'parse', side_effect=parse312):
            for arm in p.ARMS:
                with self.subTest(arm=arm):
                    source = s.worker_source(arm)
                    tree = ast.parse(source)
                    functions = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest()
                                 for node in tree.body if isinstance(node, ast.FunctionDef)}
                    self.assertTrue(all(functions[name] != value
                                        for name, value in s.ma11_fullstep.BASELINE_FUNCTIONS.items()))
                    s.validate_worker(source)
                    changed = source.replace('def clear_gradients():',
                                             'def clear_gradients():\n    raise RuntimeError("changed")')
                    self.assertNotEqual(changed, source)
                    with self.assertRaisesRegex(ValueError, 'training functions changed: clear_gradients'):
                        s.validate_worker(changed)
                    with self.assertRaisesRegex(ValueError, 'timing loop changed'):
                        s.validate_worker(source.replace('time.perf_counter()', 'time.monotonic()'))
        self.assertEqual(ast.FunctionDef._fields, class_fields)

    def test_nonempty_type_parameters_are_not_ignored(self):
        source = s.worker_source('ma11')
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        if 'type_params' not in function._fields:
            function._fields = (*function._fields, 'type_params')
        function.type_params = [ast.Name(id='T', ctx=ast.Load())]
        with patch.object(ast, 'parse', return_value=tree):
            with self.assertRaisesRegex(ValueError, 'training functions changed: ' + function.name):
                s.validate_worker(source)

    def test_real_prepared_sources_form_complete_worker_packages(self):
        builder = load('build')
        with tempfile.TemporaryDirectory(prefix='rtx5090-workers-cpu-') as tmp:
            root = Path(tmp)
            build = builder.prepare(root/'build')
            # These deliberately invalid ELF fixtures only exercise packaging.
            # verify(built=True) must reject the entire uncompiled build.
            for arm in p.ARMS:
                for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                    (builder.native_root(build, arm)/name).write_text('CPU_FIXTURE_NOT_A_SHARED_LIBRARY\n')
            (build/'codec-basic/libbasic_quant.so').write_text('CPU_FIXTURE_NOT_A_SHARED_LIBRARY\n')
            plan = dict(protocol=p.PROTOCOL, run_id='CPU-INTEGRATION', arms=list(p.ARMS),
                        communication={'CPU_FIXTURE': True},
                        environment={'torch': '2.8.0a0+5228986c39.nv25.06'})
            workers = s.prepare_workers(build, root/'run', plan)
            self.assertEqual(tuple(workers), p.ARMS)
            for arm, worker in workers.items():
                scripts = Path(worker['root'])
                self.assertEqual((scripts/'build'/p.POLICY[arm]).resolve(), builder.native_root(build, arm).resolve())
                for name in worker['files']:
                    if name.endswith('.py'):
                        compile((scripts/name).read_text(), str(scripts/name), 'exec')
                namespace = {}
                exec((scripts/'routing.py').read_text(), namespace)
                original = namespace['definition'](p.POLICY[arm])
                for key in ('family', 'quantization', 'policy', 'roles'):
                    self.assertEqual(original[key], worker['definition'][key])
                self.assertEqual(worker['config']['scope_definition'], worker['definition'])
                if arm in ('a', 'r'):
                    self.assertTrue(namespace['is_fused_v2'](p.POLICY[arm]))
                self.assertNotIn('init_flux_shm(', (scripts/'adapter.py').read_text())
                self.assertNotIn('.peers', (scripts/'taco_support.py').read_text())
                self.assertEqual(worker['config']['cases'][0]['name'], 'rtx5090-phase6')
                self.assertEqual(worker['config']['policies'], [p.POLICY[arm]])
                for name in ('runtime.py', 'communication.py', 'adapter.py', 'taco_support.py', 'megatron_taco.py'):
                    self.assertIn(name, worker['files'])
                self.assertIn('capability=[prop.major, prop.minor]', (scripts/'runtime.py').read_text())
                if arm in ('ma11', 'ma89'):
                    helper = (scripts/'ma11_adapter.py').read_text()
                    self.assertIn(repr(plan['environment']['torch']), helper)
                    self.assertNotIn("torch.__version__.split('+')[0] == '2.6.0'", helper)
                    for audit in ('native_self_test(torch, native)', 'MA11 full-model softmax forward differs',
                                  'MA11 full-model softmax backward differs'):
                        self.assertIn(audit, helper)
            with self.assertRaisesRegex(ValueError, 'Build incomplete'):
                builder.verify(build, True)

    def test_all_workers_preserve_training_and_timing_ast(self):
        for arm in p.ARMS:
            source = s.worker_source(arm)
            s.validate_worker(source)
            compile(source, arm, 'exec')
            self.assertNotIn('operator-controller.py', source)
            self.assertIn('RTX5090_BASELINE_EXECUTE', source)
            with self.assertRaises(ValueError):
                s.validate_worker(source.replace('time.perf_counter()', 'time.monotonic()'))
        for arm in ('cr3', 'r', 'c', 'a', 'flux-basic'):
            source = s.worker_source(arm)
            self.assertNotIn('install_activation_cache', source)
            self.assertNotIn('install_zero_count', source)
            self.assertNotIn('install_softmax', source)
        self.assertNotIn('install_zero_count', s.worker_source('ma8'))
        self.assertNotIn('install_activation_cache', s.worker_source('ma9'))

    def test_worker_execution_guard_precedes_cuda_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arm in p.ARMS:
                worker = Path(tmp)/'worker.py'
                worker.write_text(s.worker_source(arm))
                env = dict(os.environ)
                env.pop('RTX5090_BASELINE_EXECUTE', None)
                result = subprocess.run([sys.executable, '-I', '-B', str(worker)], env=env,
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('GPU worker requires', result.stderr)
                self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_all_nine_arms_and_modes_validate_complete_evidence(self):
        for arm in p.ARMS:
            for mode in ('smoke', 'timing', 'profile'):
                with self.subTest(arm=arm, mode=mode):
                    args = fixture(arm, mode)
                    self.assertEqual(len(v.validate_rows(*args)), 8)

    def test_ma11_compatibility_library_identity_is_checked_for_each_rank(self):
        rows, job, plan, worker = fixture('ma11', 'smoke')
        plan['ma11_compat'] = dict(library='/CPU-FIXTURE/ma11.so', library_sha256='a'*64)
        for row in rows:
            row['rtx5090_ma11_softmax'] = dict(path=plan['ma11_compat']['library'], sha256='a'*64)
        self.assertEqual(len(v.validate_rows(rows, job, plan, worker)), 8)
        rows[3]['rtx5090_ma11_softmax']['sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'compatibility library identity'):
            v.validate_rows(rows, job, plan, worker)

    def test_wrong_hardware_software_sample_and_mechanism_rejected(self):
        for arm in p.ARMS:
            original, job, plan, worker = fixture(arm)
            mutations = [lambda row: row.update(elapsed_seconds=float('nan')),
                         lambda row: row.update(torch='2.6.0+cu124'),
                         lambda row: row.update(skips=1), lambda row: row.update(dp_rank=1),
                         lambda row: row.update(timed_steps=20),
                         lambda row: row.update(forward_checks={'unexpected': {'passed': True}}),
                         lambda row: row.update(parameter_trajectory=[{}]),
                         lambda row: row['hardware']['devices'][0].update(capability=[9, 0])]
            if arm not in ('ma11', 'ma89'):
                mutations.append(lambda row: row.update(ma11_softmax_state={'installed': True}))
            for mutation in mutations:
                rows = copy.deepcopy(original)
                mutation(rows[0])
                with self.subTest(arm=arm, mutation=mutation), self.assertRaises((ValueError, KeyError)):
                    v.validate_rows(rows, job, plan, worker)
            with self.assertRaises(ValueError):
                v.validate_rows(original[:-1], job, plan, worker)

    def test_scratch_pool_identity_memory_and_separation_are_enforced(self):
        original, job, plan, worker = fixture('ma11')
        names = sorted(name for name in original[0]['rtx5090_transport']
                       if name.endswith('.self_attention.linear_proj'))
        def first(row, index=0):
            return row['rtx5090_transport'][names[0]]['slots'][index]
        mutations = [lambda row: first(row).update(scratch_policy='private'),
                     lambda row: first(row).update(shared_scratch_bytes=0),
                     lambda row: first(row)['shared_scratch_ptrs'].update(tx_wire=0),
                     lambda row: first(row)['shared_scratch_ptrs'].update(tx_wire=99999),
                     lambda row: first(row, 1).update(shared_scratch_ptrs=first(row)['shared_scratch_ptrs'])]
        for mutate in mutations:
            rows = copy.deepcopy(original)
            mutate(rows[0])
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, 'scratch'):
                v.validate_rows(rows, job, plan, worker)

    def test_c_mask_and_rca_workspace_ownership_checked(self):
        rows, job, plan, worker = fixture('c')
        name = next(iter(rows[0]['workspace_checks']))
        rows[0]['workspace_checks'][name][0]['mask_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'mask'):
            v.validate_rows(rows, job, plan, worker)
        for arm in ('r', 'a'):
            rows, job, plan, worker = fixture(arm)
            name = next(iter(rows[0]['ma2c_codec_state']))
            rows[0]['ma2c_codec_state'][name]['owns_output'] = arm == 'a'
            with self.assertRaisesRegex(ValueError, 'ownership'):
                v.validate_rows(rows, job, plan, worker)

    def test_equivalence_includes_parameters_and_gradients(self):
        rows, *_ = fixture('ma11', 'smoke')
        v.equivalent(rows, copy.deepcopy(rows), True)
        for key in ('losses', 'grad_norms', 'final_parameters_sha256', 'parameter_trajectory'):
            altered = copy.deepcopy(rows)
            altered[0][key] = None
            with self.assertRaises(ValueError):
                v.equivalent(rows, altered, True)

    def test_profiles_require_device_path_evidence(self):
        for arm in p.ARMS:
            with self.assertRaises(ValueError):
                profiles.analyze([], arm, 0)
        self.assertTrue(profiles.is_basic_gemm('cutlass::Kernel2<flux_bf16_bf16_void_bf16_fp32_fp32_sm89_rtx5090_commnone_rcr_gemmv2_false_nil>'))
        for name in ('cublasGemm', 'cutlass_sm90_gmma', 'cutlass::GemmUniversal<Sm90>',
                     'flux_bf16_reducescatter', 'ncclKernel_AllToAll',
                     'cutlass::Kernel<cutlass::gemm::kernel::GemmUniversal<X>>'):
            self.assertFalse(profiles.is_basic_gemm(name))


if __name__ == '__main__':
    unittest.main()
