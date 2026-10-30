"""CPU oracle, native ABI/lifecycle and ordering checks; no GPU performance claim."""
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import ma18_adapter as ma18
import test_ma8_adapter as ma8_tests

LAYERS = Path(__file__).resolve().parents[4] / 'Megatron-LM/megatron/core/tensor_parallel/layers.py'


def original_source():
    text = LAYERS.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                and n.name == 'LinearWithGradAccumulationAndAsyncCommunication')
    return ma18.cache.normalized(ast.get_source_segment(text, node))


def candidate_source():
    return ma18.ipc_allgather_class_source(original_source())


def valid_state(steps=3, mode='smoke', rank=4):
    return dict(protocol=ma18.PROTOCOL, installed=True,
        local_shape=list(ma18.LOCAL_SHAPE), world_size=8,
        source_sha256=ma18.CLASS_SHA256, backward_sha256=ma18.BACKWARD_SHA256,
        calls=steps * 256, failed=False, active=False,
        native=dict(group_ranks=list(range(8)), rank=rank, device=rank, stream=19,
                    calls=steps * 256, initializations=1, slots=2, staging_bytes=2097152,
                    local_staging_bytes=4194304, copies_per_call=9, publication_barriers_per_call=1,
                    native_protocol=18, copy_order='cyclic_sources_rank_ordered_destination',
                    failed=False, closed=False,
                    self_test=(dict(passed=True, calls=20, rank_delays=True, retained_outputs=True,
                                    all_bytes=True, queued_before_audit=True) if mode=='smoke' else None),
                    device_memory_before=[80000, 100000], device_memory_after=[70000, 100000],
                    device_memory_after_work=[60000, 100000]),
        modules={name: dict(role=role, rank=rank, calls=steps * 4,
                           audits=steps * 4 if mode == 'smoke' else 0)
                 for name, role in ma18.module_roles().items()})


class SourceTests(unittest.TestCase):
    def test_only_class_name_and_gather_submission_change(self):
        cached = ma18.cache.cached_class_source(original_source())
        candidate = candidate_source()
        restored = candidate.replace('_MA18IpcAllGatherLinear', '_MA8CachedRawLinear').replace(
            '_ma18_gather(', 'dist_all_gather_func(')
        self.assertEqual(cached, restored)
        self.assertEqual(hashlib.sha256(candidate.encode()).hexdigest(), ma18.CLASS_SHA256)
        self.assertEqual(ma18.backward_sha256(candidate), ma18.BACKWARD_SHA256)
        self.assertEqual(ma18.backward_sha256(candidate), ma18.backward_sha256(cached))

    def test_drift_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'pinned'):
            ma18.ipc_allgather_class_source(original_source().replace('grad_input =', 'grad_input  ='))

    def test_real_autograd_all_ranks_both_bias_modes_and_retained_storage(self):
        torch.set_num_threads(1)
        for rank in range(8):
            for bias in (False, True):
                results, saved = [], []
                for candidate in (False, True):
                    events = []

                    def gather(output, input, group=None, async_op=False):
                        self.assertFalse(async_op)
                        events.append('gather')
                        output.copy_(torch.cat([input + peer - rank for peer in range(8)]))

                    def scatter(output, input, group=None, async_op=False):
                        self.assertTrue(async_op)
                        events.append('scatter')
                        rows = output.shape[0]
                        output.copy_(input[rank * rows:(rank + 1) * rows] * 8)
                        return types.SimpleNamespace(wait=lambda: events.append('wait'))

                    class Proxy:
                        cuda = types.SimpleNamespace(current_device=lambda: 'cpu')
                        def __getattr__(self, name):
                            return getattr(torch, name)

                    namespace = dict(torch=Proxy(), custom_fwd=lambda f: f, custom_bwd=lambda f: f,
                        get_tensor_model_parallel_world_size=lambda: 8,
                        get_tensor_model_parallel_group=lambda: None,
                        dist_all_gather_func=gather, _ma18_gather=gather,
                        dist_reduce_scatter_func=scatter,
                        prepare_input_tensors_for_wgrad_compute=lambda grad, inp:
                            (grad.contiguous().view(-1, grad.shape[-1]), inp.view(-1, inp.shape[-1])))
                    text = candidate_source() if candidate else ma18.cache.cached_class_source(original_source())
                    exec(compile(text, '<ma18-cpu-autograd>', 'exec'), namespace)
                    cls = namespace['_MA18IpcAllGatherLinear' if candidate else '_MA8CachedRawLinear']
                    value = (torch.arange(16).reshape(2, 1, 8).float() / 8 + rank).requires_grad_()
                    weight = (torch.arange(24).reshape(3, 8).float() / 8 - 2).requires_grad_()
                    offset = torch.tensor([.5, -1., 2.], requires_grad=True) if bias else None
                    output = cls.apply(value, weight, offset, False, False, True, None, None)
                    retained = output.grad_fn.saved_tensors[2]
                    saved.append(retained)
                    snapshot = retained.clone()
                    grad = torch.arange(output.numel()).reshape_as(output).float() / 8 - 1
                    output.backward(grad)
                    self.assertEqual(events, ['gather', 'scatter', 'wait'])
                    self.assertTrue(torch.equal(retained, snapshot))
                    results.append([output.detach(), value.grad, weight.grad] + ([offset.grad] if bias else []))
                self.assertNotEqual(saved[0].data_ptr(), saved[1].data_ptr())
                for left, right in zip(*results):
                    self.assertTrue(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


class FakeTensor:
    is_cuda = True
    device = torch.device('cuda', 4)
    dtype = torch.bfloat16
    def __init__(self, tensor):
        self.tensor, self.shape = tensor, tensor.shape
    def is_contiguous(self):
        return self.tensor.is_contiguous()
    def data_ptr(self):
        return self.tensor.data_ptr()
    def view(self, dtype):
        return self.tensor.view(dtype)


class FakeNative:
    device, stream = 4, 19
    def __init__(self, *args):
        self.calls, self.closed = 0, False
    def launch(self, output, input):
        output.tensor.copy_(torch.cat([input.tensor + peer for peer in range(8)]))
        self.calls += 1
    def state(self):
        return dict(valid_state()['native'], calls=self.calls)
    def close(self):
        self.closed = True


class TrackerTests(unittest.TestCase):
    def fixture(self, mode='timing', corrupt=False):
        markers, audits = [], []

        class Proxy:
            bfloat16, uint8 = torch.bfloat16, torch.uint8
            cuda = types.SimpleNamespace(current_device=lambda: 4,
                current_stream=lambda: types.SimpleNamespace(cuda_stream=19),
                is_current_stream_capturing=lambda: False)
            autograd = types.SimpleNamespace(profiler=types.SimpleNamespace(
                record_function=lambda name: Marker(name)))
            @staticmethod
            def empty_like(value):
                return FakeTensor(torch.empty_like(value.tensor))
            equal = staticmethod(torch.equal)

        class Marker:
            def __init__(self, name): self.name = name
            def __enter__(self): markers.append(self.name)
            def __exit__(self, *args): return False

        def oracle(output, input, group=None):
            audits.append(True)
            output.tensor.copy_(torch.cat([input.tensor + peer + int(corrupt) for peer in range(8)]))

        group = object()
        tracker = ma18.Tracker(Proxy(), group, 4, mode, FakeNative(), oracle, candidate_source())
        value = FakeTensor(torch.arange(16).reshape(2, 1, 8).to(torch.bfloat16))
        output = FakeTensor(torch.empty(16, 1, 8, dtype=torch.bfloat16))
        return tracker, value, output, markers, audits

    def test_complete_module_schedule_native_counts_and_profile_markers(self):
        tracker, value, output, markers, audits = self.fixture('profile')
        with patch.object(ma18, 'LOCAL_SHAPE', (2, 1, 8)):
            for _ in range(2):
                for name in tracker.names:
                    tracker.bound_gather(name)(output, value, tracker.group)
        self.assertEqual(tracker.calls, 128)
        self.assertEqual(tracker.native.calls, 128)
        self.assertEqual(audits, [])
        self.assertEqual(markers, ['ma18.column.gather.qkv', 'ma18.column.gather.fc1'] * 64)
        self.assertTrue(all(row['calls'] == 2 for row in tracker.modules.values()))

    def test_smoke_checks_all_bytes_and_timing_has_no_oracle_or_marker(self):
        for mode in ('smoke', 'timing'):
            tracker, value, output, markers, audits = self.fixture(mode)
            with patch.object(ma18, 'LOCAL_SHAPE', (2, 1, 8)):
                tracker.bound_gather(tracker.names[0])(output, value, tracker.group)
            self.assertEqual(len(audits), int(mode == 'smoke'))
            self.assertEqual(markers, [])

    def test_corruption_schedule_reentry_device_stream_layout_and_group_fail_closed(self):
        mutations = [lambda t, i, o: setattr(t, 'active', True),
                     lambda t, i, o: setattr(t, 'device', 3),
                     lambda t, i, o: setattr(t, 'stream', 20),
                     lambda t, i, o: setattr(t, 'calls', 1),
                     lambda t, i, o: setattr(i, 'shape', (1, 2, 8)),
                     lambda t, i, o: setattr(t, 'group', object())]
        for mutate in mutations:
            tracker, value, output, _, _ = self.fixture()
            group = tracker.group
            mutate(tracker, value, output)
            with patch.object(ma18, 'LOCAL_SHAPE', (2, 1, 8)), self.assertRaises(RuntimeError):
                tracker.bound_gather(tracker.names[0])(output, value, group)
            self.assertTrue(tracker.failed)
            self.assertEqual(tracker.native.calls, 0)
            with self.assertRaises(RuntimeError): tracker.state()
        tracker, value, output, _, _ = self.fixture('smoke', corrupt=True)
        with patch.object(ma18, 'LOCAL_SHAPE', (2, 1, 8)), self.assertRaisesRegex(RuntimeError, 'bytewise'):
            tracker.bound_gather(tracker.names[0])(output, value, tracker.group)
        self.assertTrue(tracker.failed)


class NativeSubmissionTests(unittest.TestCase):
    def test_constructor_rejects_cross_slot_alias_and_duplicate_device_assignment(self):
        def fixture(alias=False, duplicate_device=False):
            allocated=[]
            def create(shape,dtype,group,**kwargs):
                base=100 if not allocated or alias else 200
                peers=[types.SimpleNamespace(data_ptr=lambda p=base+rank:p) for rank in range(8)]
                allocated.append(peers)
                return peers
            def identities(output,value,group=None):
                output[:]=[(*value[:3],0 if duplicate_device else rank) for rank in range(8)]
            proxy=types.SimpleNamespace(bfloat16=torch.bfloat16,
                cuda=types.SimpleNamespace(current_device=lambda:4,
                    current_stream=lambda:types.SimpleNamespace(cuda_stream=19),
                    mem_get_info=lambda device:(70,100), device_count=lambda:8,
                    can_device_access_peer=lambda device,peer:True, synchronize=lambda device:None),
                distributed=types.SimpleNamespace(get_process_group_ranks=lambda group:list(range(8)),
                    get_world_size=lambda group=None:8, get_rank=lambda group=None:4,
                    all_gather_object=identities, barrier=lambda group:None))
            library=types.SimpleNamespace(ma18_ipc_protocol=lambda:18,
                ma18_ipc_stage_cuda=lambda *args:0,ma18_ipc_pull_cuda=lambda *args:0)
            flux=types.SimpleNamespace(create_tensor_list=create,
                GroupBarrier=lambda group,ring_mode:types.SimpleNamespace())
            return proxy,library,flux
        for alias,duplicate in ((False,False),(True,False),(False,True)):
            proxy,library,flux=fixture(alias,duplicate)
            with patch.dict(sys.modules,flux=flux):
                if alias or duplicate:
                    with self.assertRaisesRegex(RuntimeError,'aliased sender|distinct rank'):
                        ma18.NativeAllGather(proxy,library,object(),4,'timing')
                else:
                    native=ma18.NativeAllGather(proxy,library,object(),4,'timing')
                    self.assertEqual(len(native.pointer_tables),2)
                    self.assertEqual(native.calls,0)

    def make_native(self, fail=None):
        events=[]
        def stage(destination, source, size, stream):
            events.append(('stage', destination, source, size, stream))
            return int(fail == 'stage')
        def pull(output, peers, rank, size, stream):
            events.append(('pull', output, list(peers), rank, size, stream))
            return int(fail == 'pull')
        native=ma18.NativeAllGather.__new__(ma18.NativeAllGather)
        native.failed=native.closed=False
        native.calls=0; native.rank=4; native.stream=19
        native.library=types.SimpleNamespace(ma18_ipc_stage_cuda=stage, ma18_ipc_pull_cuda=pull)
        def tensor(pointer): return types.SimpleNamespace(data_ptr=lambda: pointer)
        native.peers=[[tensor(slot*100+rank+1) for rank in range(8)] for slot in range(2)]
        native.pointer_tables=[list(range(1,9)),list(range(101,109))]
        native.barrier=types.SimpleNamespace(barrier_all=lambda stream: events.append(('barrier',stream)))
        return native,events,tensor(1000),tensor(2000)

    def test_double_slot_submit_order_and_cyclic_native_inputs(self):
        native,events,output,input=self.make_native()
        for _ in range(5): native.launch(output,input)
        self.assertEqual(native.calls,5)
        self.assertEqual([event[0] for event in events],['stage','barrier','pull']*5)
        self.assertEqual([event[1] for event in events if event[0]=='stage'],[5,105,5,105,5])
        self.assertEqual([event[2] for event in events if event[0]=='pull'],
                         [list(range(1,9)),list(range(101,109)),list(range(1,9)),
                          list(range(101,109)),list(range(1,9))])

    def test_partial_submission_poisoned_and_never_retried(self):
        for fail,count in [('stage',1),('pull',3)]:
            native,events,output,input=self.make_native(fail)
            with self.assertRaisesRegex(RuntimeError,'submission failed'): native.launch(output,input)
            self.assertTrue(native.failed)
            self.assertEqual(native.calls,0)
            self.assertEqual(len(events),count)
            with self.assertRaisesRegex(RuntimeError,'failed/closed'): native.launch(output,input)
            self.assertEqual(len(events),count)

    def test_barrier_exception_poisoned(self):
        native,events,output,input=self.make_native()
        native.barrier.barrier_all=lambda stream: (_ for _ in ()).throw(RuntimeError('barrier failure'))
        with self.assertRaisesRegex(RuntimeError,'barrier failure'): native.launch(output,input)
        self.assertTrue(native.failed)
        self.assertEqual(len(events),1)


class StateTests(unittest.TestCase):
    def test_all_modes_and_ranks(self):
        for mode in ('smoke','timing','profile'):
            for rank in range(8):
                self.assertTrue(ma18.validate_state(valid_state(mode=mode,rank=rank),3,mode,rank))

    def test_reuse_memory_audit_and_coverage_fail_closed(self):
        for key,value in [('slots',1),('staging_bytes',1024),('calls',767),
                          ('copies_per_call',8),('self_test',None),
                          ('publication_barriers_per_call',2),('native_protocol',17)]:
            state=valid_state(); state['native'][key]=value
            with self.assertRaises(RuntimeError): ma18.validate_state(state,3,'smoke',4)
        state=valid_state(mode='timing'); state['native']['self_test']=valid_state()['native']['self_test']
        with self.assertRaises(RuntimeError): ma18.validate_state(state,3,'timing',4)
        state=valid_state(); state['modules'].pop(next(iter(state['modules'])))
        with self.assertRaises(RuntimeError): ma18.validate_state(state,3,'smoke',4)


class ReuseModelTests(unittest.TestCase):
    @staticmethod
    def explore(ranks,calls,slots,barrier=True):
        # pc per rank: stage -> publish -> wait -> read; a read checks every
        # peer slot before the consumer can advance to the next call.
        from collections import deque
        initial=((0,)*ranks,(-1,)*ranks,(-1,)*(ranks*slots))
        seen={initial}; pending=deque([initial])
        while pending:
            pcs,published,payload=pending.popleft()
            for rank in range(ranks):
                pc=pcs[rank]; call,phase=divmod(pc,4)
                if call>=calls: continue
                new_pc=list(pcs); new_pub=list(published); new_data=list(payload)
                if phase==0: new_data[rank*slots+call%slots]=call
                elif phase==1: new_pub[rank]=call
                elif phase==2 and barrier and min(published)<call: continue
                elif phase==3:
                    if any(payload[peer*slots+call%slots]!=call for peer in range(ranks)):
                        return False,len(seen)
                new_pc[rank]+=1
                state=(tuple(new_pc),tuple(new_pub),tuple(new_data))
                if state not in seen: seen.add(state); pending.append(state)
        return True,len(seen)

    def test_double_slot_all_interleavings_and_required_negative_controls(self):
        for ranks,calls in ((2,6),(3,3)):
            passed,states=self.explore(ranks,calls,2)
            self.assertTrue(passed); self.assertGreater(states,100)
        self.assertFalse(self.explore(2,6,1)[0])
        self.assertFalse(self.explore(2,6,2,False)[0])


if __name__=='__main__': unittest.main()
