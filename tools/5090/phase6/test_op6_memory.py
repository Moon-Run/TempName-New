"""CPU tests of ownership, all-device rollback, EOF cleanup and GPU handoff."""
from contextlib import contextmanager
import ctypes
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import op6
import op6_child
import op6_memory as memory


class DriverLifecycle(unittest.TestCase):
    def allocation(self, fail_device=None):
        allocation = memory.CudaAllocation.__new__(memory.CudaAllocation)
        allocation.size, allocation.allocations = 30000*memory.MIB, []
        allocation.release_name = 'cuDevicePrimaryCtxRelease_v2'
        calls, current = [], [None]

        def call(name, *args):
            calls.append((name, current[0]))
            if name == 'cuDeviceGetCount':
                args[0]._obj.value = 8
            elif name == 'cuDeviceGet':
                args[0]._obj.value = args[1]
            elif name == 'cuDeviceGetName':
                args[0].value = b'NVIDIA GeForce RTX 5090'
            elif name == 'cuDeviceGetPCIBusId':
                args[0].value = f'0000:{args[2].value:02d}:00.0'.encode()
            elif name == 'cuDevicePrimaryCtxRetain':
                args[0]._obj.value = args[1].value+1
            elif name == 'cuCtxSetCurrent':
                current[0] = args[0].value-1
            elif name == 'cuMemGetInfo_v2':
                args[0]._obj.value = (10 if current[0] == fail_device else 32000)*memory.MIB
                args[1]._obj.value = 32768*memory.MIB
            elif name == 'cuMemAlloc_v2':
                self.assertEqual(args[1], 30000*memory.MIB)
                args[0]._obj.value = 100+current[0]
        allocation.call = call
        return allocation, calls

    def test_exact_allocation_on_all_eight_and_eof_releases_every_device(self):
        allocation, calls = self.allocation()
        output = io.StringIO()
        memory.holder(30000, io.BytesIO(b''), output, lambda _mib: allocation)
        ready = json.loads(output.getvalue())
        self.assertEqual([r['device'] for r in ready['devices']], list(range(8)))
        self.assertEqual({r['allocated_bytes'] for r in ready['devices']}, {30000*memory.MIB})
        self.assertEqual([device for name, device in calls if name == 'cuMemFree_v2'], list(range(7, -1, -1)))
        self.assertEqual(allocation.allocations, [])

    def test_partial_failure_frees_previous_allocations_and_current_context(self):
        allocation, calls = self.allocation(fail_device=2)
        output = io.StringIO()
        with self.assertRaisesRegex(RuntimeError, 'GPU 2'):
            memory.holder(30000, io.BytesIO(b''), output, lambda _mib: allocation)
        self.assertEqual(output.getvalue(), '')
        self.assertEqual([device for name, device in calls if name == 'cuMemFree_v2'], [1, 0])
        self.assertEqual(sum(name == allocation.release_name for name, _ in calls), 3)
        self.assertEqual(allocation.allocations, [])

    def test_interrupt_during_hold_frees_all_devices(self):
        allocation, calls = self.allocation()
        stream = Mock()
        stream.read.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            memory.holder(30000, stream, io.StringIO(), lambda _mib: allocation)
        self.assertEqual(sum(name == 'cuMemFree_v2' for name, _ in calls), 8)

    def test_one_free_failure_does_not_skip_other_contexts(self):
        allocation, calls = self.allocation()
        allocation.acquire()
        original = allocation.call
        failed = []
        def fail_once(name, *args):
            if name == 'cuMemFree_v2' and not failed:
                failed.append(True)
                raise RuntimeError('free failed')
            return original(name, *args)
        allocation.call = fail_once
        with self.assertRaisesRegex(RuntimeError, 'free failed'):
            allocation.close()
        self.assertEqual(sum(name == allocation.release_name for name, _ in calls), 8)
        self.assertEqual(allocation.allocations, [])


class HolderProcess(unittest.TestCase):
    @contextmanager
    def fake_gpu_process(self, root, *, ready=True, early_exit=False):
        """Real pipes and process exit, with no CUDA loaded in the child."""
        marker = root/'freed'
        code = f'''
import json, os, sys
from pathlib import Path
try:
    if {early_exit!r}: raise RuntimeError('allocation failed')
    if {ready!r}:
        print(json.dumps(dict(status='ready', pid=os.getpid(), devices=[
            dict(device=i, allocated_bytes=30000*1024**2, pci_bus_id=str(i)) for i in range(8)])), flush=True)
    sys.stdin.buffer.read(1)
finally:
    Path({str(marker)!r}).write_text('released')
'''
        original = subprocess.Popen
        def launch(_command, **kwargs):
            return original([sys.executable, '-u', '-c', code], **kwargs)
        with patch.object(memory.subprocess, 'Popen', side_effect=launch):
            yield marker

    def test_real_pipe_eof_cleanup_and_no_cuda_in_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.fake_gpu_process(root) as marker:
                idle = Mock()
                reserve = memory.Reservation(sys.executable, root, 30000, idle)
                reserve.acquire()
                process = reserve.process
                self.assertIsNone(process.poll())
                reserve.release()
                self.assertEqual(process.returncode, 0)
                self.assertEqual(marker.read_text(), 'released')
                self.assertIsNone(reserve.process)
                idle.assert_called_once()
                records = [json.loads(x) for x in (root/'handoff/memory-reservation.jsonl').read_text().splitlines()]
                self.assertEqual([x['event'] for x in records], ['acquired', 'released'])

    def test_startup_failure_and_readiness_timeout_leave_no_process(self):
        for options in ({'early_exit': True}, {'ready': False}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.fake_gpu_process(root, **options) as marker:
                    reserve = memory.Reservation(sys.executable, root, 30000, lambda: None, ready_timeout=.2)
                    with self.assertRaises(RuntimeError):
                        reserve.acquire()
                    self.assertIsNone(reserve.process)
                    self.assertEqual(marker.read_text(), 'released')

    def test_unexpected_holder_death_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.fake_gpu_process(root):
                reserve = memory.Reservation(sys.executable, root, 30000, lambda: None)
                reserve.acquire()
                reserve.process.kill()
                reserve.process.wait()
                with self.assertRaisesRegex(RuntimeError, 'release normally'):
                    reserve.release()
                self.assertIsNone(reserve.process)

    def test_disabled_mode_never_queries_or_allocates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'not-created'
            idle = Mock(side_effect=AssertionError('GPU query'))
            reserve = memory.Reservation(sys.executable, root, 0, idle)
            reserve.acquire()
            self.assertEqual(reserve.run(lambda: 7), 7)
            reserve.release()
            self.assertFalse(root.exists())

    def test_supervisor_releases_on_completion_child_failure_and_audit_failure(self):
        from test_op6_runner import Supervision
        for failure in (None, 'child', 'audit'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.fake_gpu_process(root) as marker, patch.object(op6.op3.base, 'gpu_idle'):
                    calls, error = Supervision().run_fixture(root, failure, reserve_mib=30000)
                self.assertEqual(error is None, failure is None)
                self.assertEqual(marker.read_text(), 'released')
                events = [json.loads(line) for line in (root/'handoff/memory-reservation.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['event'], 'released')
                self.assertEqual([x['holder_pid'] for x in events if x['event'] == 'acquired'],
                                 [x['holder_pid'] for x in events if x['event'] == 'released'])
                self.assertTrue(all(not x.get('forced', False) for x in events))


class StageHandoff(unittest.TestCase):
    def test_release_precedes_idle_check_and_every_gpu_process(self):
        import tp8_dp2_train
        held, events = [True], []
        reserve = memory.Reservation(sys.executable, '/unused', 1, lambda: None)
        def release(*_args, **_kwargs):
            held[0] = False
            events.append('release')
        def acquire():
            held[0] = True
            events.append('acquire')
        reserve.release, reserve.acquire = release, acquire
        def idle():
            self.assertFalse(held[0])
            events.append('idle')
        def work(label):
            self.assertFalse(held[0])
            events.append(label)
            if label == 'failed':
                raise RuntimeError('worker failed')
            return 42
        base = SimpleNamespace(gpu_idle=idle)
        with patch.object(tp8_dp2_train, 'run_process', side_effect=work) as original:
            with memory.child_handoff(reserve, base):
                base.gpu_idle()
                self.assertEqual(tp8_dp2_train.run_process('probe'), 42)
                self.assertTrue(held[0])
                # The next collective probe has no separate gpu_idle call.
                self.assertEqual(tp8_dp2_train.run_process('collectives'), 42)
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    tp8_dp2_train.run_process('failed')
                self.assertFalse(held[0])
            self.assertIs(tp8_dp2_train.run_process, original)
        self.assertIs(base.gpu_idle, idle)
        self.assertEqual(events, ['release', 'idle', 'release', 'probe', 'acquire',
                                  'release', 'collectives', 'acquire', 'release', 'failed'])

    def test_term_raises_for_worker_cleanup_and_restores_handlers(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(SystemExit) as error:
            with memory.termination_signals():
                os.kill(os.getpid(), signal.SIGTERM)
        self.assertEqual(error.exception.code, 128+signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_parent_cancel_terminates_only_its_child_group(self):
        process = Mock(pid=123456)
        process.wait.side_effect = [KeyboardInterrupt, 0]
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory.subprocess, 'Popen', return_value=process), \
                patch.object(memory.os, 'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):
                memory.run_child(['fake'], {}, Path(tmp)/'log')
            kill.assert_called_once_with(123456, signal.SIGTERM)

    def test_reservation_option_is_propagated_and_can_be_disabled(self):
        for mib in (0, 30000):
            args = op6.arguments(['--reserve-mib', str(mib)])
            rebuilt = op6.arguments(op6.supervisor_command(args)[2:])
            self.assertEqual(rebuilt.reserve_mib, mib)
            command = op6.child_command(args, op6.schedule()[0], Path('/build'))
            self.assertEqual(command[command.index('--reserve-mib')+1], str(mib))
        for mib in (-1, 32768):
            with self.assertRaises(ValueError):
                op6.arguments(['--reserve-mib', str(mib)])

    def test_child_does_not_reserve_before_main_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'new-child'
            def main(_args):
                self.assertFalse(root.exists())
            with patch.object(op6_child.op3, 'main', side_effect=main), \
                    patch.object(memory, 'session', side_effect=AssertionError('reserved too early')):
                op6_child.main(['--variant', 'op6', '--reserve-mib', '30000', '--execute',
                                '--foreground', '--out', str(root), '--op3-build', '/build'])


if __name__ == '__main__':
    unittest.main()
