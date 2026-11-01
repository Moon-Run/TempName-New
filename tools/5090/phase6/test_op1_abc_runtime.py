"""CPU scheduling/layout tests. These do not claim CUDA or NCCL validation."""
import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
import threading

import op1_abc_native as native
import op1_abc_sources as sources


def runtime_namespace():
    tree = ast.parse(Path(native.HERE/'op1_abc_codec.py').read_text())
    keep = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ('chunk_plan', 'TensorTacoRS')]
    namespace = dict(ARMS=sources.BASIC_ARMS, _PIPELINES={}, _POISONED_PIPELINES=set(), _POISONED_REFERENCES=[],
                     _SUBMISSION_LOCKS={}, threading=threading, nullcontext=nullcontext)
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<runtime-cpu-test>', 'exec'), namespace)
    return namespace


class Tensor:
    serial = 0

    def __init__(self, shape, dtype='bf16', device=None):
        self.shape = tuple(shape) if isinstance(shape, tuple) else (shape,)
        self.dtype, self.device = dtype, device
        Tensor.serial += 1
        self.pointer = Tensor.serial

    def numel(self):
        import math
        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def reshape(self, *shape):
        return Tensor(shape, self.dtype, self.device)

    def __getitem__(self, index):
        return Tensor(self.shape[1:], self.dtype, self.device)

    def data_ptr(self):
        return self.pointer


def fake_runtime(arm):
    ns, log = runtime_namespace(), []
    device = SimpleNamespace(index=0)
    current = []

    class Stream:
        serial = 0

        def __init__(self, device=None):
            Stream.serial += 1
            self.cuda_stream = Stream.serial

        def wait_event(self, event):
            if event.recorded is None:
                raise AssertionError('Wait before record')
            log.append(('wait_event', self.cuda_stream, event.recorded))

    current.append(Stream())

    class Event:
        def __init__(self):
            self.recorded = None

        def record(self, stream):
            self.recorded = (stream.cuda_stream, len(log))
            log.append(('event', stream.cuda_stream))

    @contextmanager
    def stream_context(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous

    def exchange(rx, tx, **kw):
        log.append(('exchange', current[0].cuda_stream, kw))
        return SimpleNamespace(wait=lambda: log.append(('work_wait', current[0].cuda_stream)))

    ns.update(torch=SimpleNamespace(bfloat16='bf16', uint8='uint8',
        device=lambda *args: device,
        empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
        empty_like=lambda tensor: Tensor(tensor.shape, tensor.dtype, tensor.device),
        cuda=SimpleNamespace(current_device=lambda: 0, Stream=Stream, Event=Event,
            current_stream=lambda device: current[0], stream=stream_context)),
        dist=SimpleNamespace(get_rank=lambda group: 0, get_world_size=lambda group: 8,
            all_to_all_single=exchange),
        os=SimpleNamespace(environ={'E2E_POLICY': arm}), load_library=lambda: None,
        encode_chunk=lambda partial, tx, rank, begin, count, raw:
            log.append(('encode', current[0].cuda_stream, begin)),
        decode_chunk=lambda rx, local, output, rank, begin, count, raw:
            log.append(('decode', current[0].cuda_stream, begin)))
    codec = ns['TensorTacoRS'](object(), 2048, 4096)
    return ns, codec, Tensor((2048, 4096), device=device), log, current


class RuntimeTests(unittest.TestCase):
    def test_layout_covers_each_group_once_and_sparse_budget(self):
        plan = runtime_namespace()['chunk_plan']
        for total in (64, 8192, 16384):
            for arm in sources.BASIC_ARMS:
                chunks = plan(total, arm)
                groups = [g for c in chunks for g in range(c['begin'], c['begin']+c['count'])]
                self.assertEqual(groups, list(range(total)))
                quantized = sum(c['count']-c['raw_groups'] for c in chunks)
                self.assertEqual(quantized, total//64 if arm == sources.BASIC_ARMS[2] else total)
                for chunk in chunks:
                    sizes = [256]*chunk['raw_groups']+[136]*(chunk['count']-chunk['raw_groups'])
                    self.assertEqual(sum(sizes), chunk['peer_bytes'])
                    self.assertTrue(all(size % 8 == 0 for size in sizes))
        for total in (0, 63, True):
            with self.assertRaises(ValueError):
                plan(total, sources.BASIC_ARMS[0])

    def test_completion_event_and_final_join_order(self):
        for arm in sources.BASIC_ARMS:
            ns, codec, partial, log, current = fake_runtime(arm)
            first = codec.reduce(partial)
            initial_log = list(log)
            second = codec.reduce(partial)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            state = codec.state()
            self.assertEqual(state['wait_calls'], 2*len(codec.chunks))
            self.assertFalse(state['pending_exchange'])
            for pos, event in enumerate(initial_log):
                if event[0] == 'work_wait' and codec.arrival:
                    self.assertEqual(initial_log[pos+1][0], 'event')
                    self.assertEqual(initial_log[pos+2][0], 'wait_event')
                    self.assertEqual(initial_log[pos+3][0], 'decode')
            if codec.arrival:
                self.assertEqual(initial_log[-1][0], 'wait_event')
                self.assertEqual(initial_log[-1][1], current[0].cuda_stream)
                self.assertEqual(initial_log[-1][2][0], codec.decode_stream.cuda_stream)
                decoders = [event for event in initial_log if event[0] == 'decode']
                self.assertEqual({x[1] for x in decoders}, {codec.decode_stream.cuda_stream})
            else:
                self.assertEqual([x[0] for x in initial_log], ['encode', 'exchange', 'work_wait', 'decode'])

    def test_failure_poison_retains_tensors_and_rejects_reuse(self):
        ns, codec, partial, log, _ = fake_runtime(sources.BASIC_ARMS[1])
        def fail(*args):
            raise RuntimeError('synthetic decoder failure')
        ns['decode_chunk'] = fail
        with self.assertRaisesRegex(RuntimeError, 'synthetic decoder'):
            codec.reduce(partial)
        self.assertIs(codec._failed_refs[0], partial)
        self.assertIn(codec._pipeline_key, ns['_POISONED_PIPELINES'])
        with self.assertRaisesRegex(ValueError, 'poisoned'):
            codec.reduce(partial)

    def test_concurrent_submission_is_rejected_before_enqueue(self):
        ns, codec, partial, log, _ = fake_runtime(sources.BASIC_ARMS[1])
        codec._submit_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'Concurrent ABC'):
                codec.reduce(partial)
            self.assertEqual(log, [])
        finally:
            codec._submit_lock.release()

    def test_generated_workers_keep_frozen_training_ast(self):
        for arm in sources.p.ARMS:
            source = sources.worker_source(arm)
            sources.base.validate_worker(source)
            ast.parse(sources.taco_source(arm))
            self.assertIn("report['op1_abc_transport']", source)
        self.assertEqual(sources.routing.PROTOCOL, sources.p.PROTOCOL)


if __name__ == '__main__':
    unittest.main()
