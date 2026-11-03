"""Idle-only GPU memory reservation with explicit handoff to OP6 GPU stages.

CUDA is initialized only in the disposable holder process. Controllers never
keep a CUDA context, and no reservation overlaps a training/probe subprocess.
Closing the owner's stdin pipe releases the holder even if the owner dies.
"""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

DEFAULT_MIB = 30000
MIB = 1024**2
GPU_COUNT = 8


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def policy(mib):
    return dict(mode='idle-handoff' if mib else 'disabled', mib_per_gpu=mib,
                gpu_count=GPU_COUNT, allocation_bytes_per_gpu=mib*MIB,
                held_during_gpu_work=False, atomic_gpu_exclusivity=False)


class CudaAllocation:
    """One fixed allocation on each visible 5090; no kernels or tensor cache."""
    def __init__(self, mib):
        self.size = mib*MIB
        self.allocations = []
        self.lib = ctypes.CDLL('libcuda.so.1')
        signatures = {
            'cuInit': [ctypes.c_uint],
            'cuDeviceGetCount': [ctypes.POINTER(ctypes.c_int)],
            'cuDeviceGet': [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            'cuDeviceGetName': [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
            'cuDeviceGetPCIBusId': [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
            'cuDevicePrimaryCtxRetain': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
            'cuCtxSetCurrent': [ctypes.c_void_p],
            'cuMemGetInfo_v2': [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
            'cuMemAlloc_v2': [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
            'cuMemFree_v2': [ctypes.c_uint64],
            'cuGetErrorString': [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        }
        self.release_name = ('cuDevicePrimaryCtxRelease_v2'
                             if hasattr(self.lib, 'cuDevicePrimaryCtxRelease_v2')
                             else 'cuDevicePrimaryCtxRelease')
        signatures[self.release_name] = [ctypes.c_int]
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, ctypes.c_int

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            message = ctypes.c_char_p()
            self.lib.cuGetErrorString(code, ctypes.byref(message))
            detail = message.value.decode(errors='replace') if message.value else 'unknown CUDA error'
            raise RuntimeError(f'{name} failed ({code}): {detail}')

    def acquire(self):
        self.call('cuInit', 0)
        count = ctypes.c_int()
        self.call('cuDeviceGetCount', ctypes.byref(count))
        require(count.value == GPU_COUNT, 'Memory reservation requires exactly eight visible GPUs')
        evidence = []
        for index in range(GPU_COUNT):
            device, context, pointer = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
            name, bus = ctypes.create_string_buffer(128), ctypes.create_string_buffer(32)
            self.call('cuDeviceGet', ctypes.byref(device), index)
            self.call('cuDeviceGetName', name, len(name), device)
            self.call('cuDeviceGetPCIBusId', bus, len(bus), device)
            require('RTX 5090' in name.value.decode(), 'Reservation requires RTX 5090 GPUs')
            self.call('cuDevicePrimaryCtxRetain', ctypes.byref(context), device)
            self.allocations.append((device, context, pointer))
            self.call('cuCtxSetCurrent', context)
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            self.call('cuMemGetInfo_v2', ctypes.byref(free), ctypes.byref(total))
            require(free.value >= self.size,
                    f'GPU {index}: need {self.size//MIB} MiB, only {free.value//MIB} MiB free')
            self.call('cuMemAlloc_v2', ctypes.byref(pointer), self.size)
            evidence.append(dict(device=index, name=name.value.decode(), pci_bus_id=bus.value.decode(),
                allocated_bytes=self.size, free_before_bytes=free.value, total_bytes=total.value))
        require(len({row['pci_bus_id'] for row in evidence}) == GPU_COUNT, 'GPU devices are duplicated')
        return evidence

    def close(self):
        errors = []
        while self.allocations:
            device, context, pointer = self.allocations.pop()
            try:
                self.call('cuCtxSetCurrent', context)
                if pointer.value:
                    self.call('cuMemFree_v2', pointer)
            except Exception as exc:
                errors.append(str(exc))
            finally:
                try:
                    self.call(self.release_name, device)
                except Exception as exc:
                    errors.append(str(exc))
        require(not errors, '; '.join(errors))


@contextmanager
def termination_signals():
    """Turn TERM/HUP/INT into exceptions so existing worker-group cleanup runs."""
    saved = {}
    interrupted = False

    def stop(signum, _frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise SystemExit(128+signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            saved[sig] = signal.signal(sig, stop)
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def holder(mib, input_stream=None, output_stream=None, allocation_factory=CudaAllocation):
    require(mib > 0, 'Holder requires a positive MiB allocation')
    input_stream = sys.stdin.buffer if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    allocation = allocation_factory(mib)
    try:
        devices = allocation.acquire()
        print(json.dumps(dict(status='ready', pid=os.getpid(), devices=devices)), file=output_stream, flush=True)
        # EOF is deliberate release or owner death. Do not busy-loop on a GPU.
        input_stream.read(1)
    finally:
        allocation.close()


class Reservation:
    def __init__(self, python, root, mib, idle_check, *, ready_timeout=120):
        require(type(mib) is int and 0 <= mib < 32768, 'Use --reserve-mib 0..32767')
        self.python, self.root, self.mib = str(python), Path(root), mib
        self.idle_check, self.ready_timeout = idle_check, ready_timeout
        self.process = None

    def record(self, event, **details):
        path = self.root/'handoff/memory-reservation.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                event=event, mib_per_gpu=self.mib, **details))+'\n')

    def acquire(self):
        if not self.mib:
            return
        if self.process is not None:
            require(self.process.poll() is None, 'GPU memory holder exited unexpectedly')
            return
        self.idle_check()  # No existing GPU process is evicted or ignored.
        path = self.root/'handoff/memory-holder.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
        command = [self.python, str(Path(__file__).resolve()), '--hold', '--mib', str(self.mib)]
        try:
            with path.open('a') as errors:
                self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=errors, env=env, start_new_session=True, close_fds=True)
            deadline = time.monotonic()+self.ready_timeout
            pending = b''
            while b'\n' not in pending:
                require(time.monotonic() < deadline, 'Timed out reserving GPU memory; see '+str(path))
                readable, _, _ = select.select([self.process.stdout], [], [], .2)
                if readable:
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    require(chunk, 'GPU memory reservation failed; see '+str(path))
                    pending += chunk
                    require(len(pending) < 65536, 'Invalid GPU holder acknowledgement')
            ready = json.loads(pending.split(b'\n', 1)[0])
            devices = ready.get('devices', [])
            require(ready.get('status') == 'ready' and ready.get('pid') == self.process.pid
                    and [row['device'] for row in devices] == list(range(GPU_COUNT))
                    and all(row['allocated_bytes'] == self.mib*MIB for row in devices)
                    and len({row['pci_bus_id'] for row in devices}) == GPU_COUNT,
                    'Incomplete GPU reservation acknowledgement')
            require(self.process.poll() is None, 'GPU holder exited before handoff')
            self.record('acquired', holder_pid=self.process.pid, devices=devices)
        except BaseException:
            self.release(check=False)
            raise

    def release(self, *, check=True):
        process, self.process = self.process, None
        if process is None:
            return
        unexpected = process.poll() is not None
        forced = False
        try:
            process.stdin.close()  # EOF is also effective across process namespaces.
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forced = True
                process.kill()  # Only the holder created by this controller.
                code = process.wait(timeout=15)
        finally:
            process.stdout.close()
        self.record('released', holder_pid=process.pid, returncode=code, forced=forced,
                    unexpected_exit=unexpected)
        if check:
            require(not unexpected and code == 0 and not forced, 'GPU holder did not release normally')

    def run(self, callback, *args, **kwargs):
        self.release()
        # If work fails, leave GPUs free. Reacquire only after successful exit.
        result = callback(*args, **kwargs)
        self.acquire()
        return result


@contextmanager
def session(python, root, mib, idle_check):
    reservation = Reservation(python, root, mib, idle_check)
    with termination_signals():
        try:
            reservation.acquire()
            yield reservation
        except BaseException:
            reservation.release(check=False)
            raise
        else:
            reservation.release()


@contextmanager
def child_handoff(reservation, base):
    """Cover environment/native/operator/model probes without changing workers."""
    import tp8_dp2_train
    previous_idle, previous_run = base.gpu_idle, tp8_dp2_train.run_process

    def idle():
        reservation.release()
        return previous_idle()

    def run(*args, **kwargs):
        return reservation.run(previous_run, *args, **kwargs)

    base.gpu_idle, tp8_dp2_train.run_process = idle, run
    try:
        yield
    finally:
        base.gpu_idle, tp8_dp2_train.run_process = previous_idle, previous_run


def run_child(command, env, log):
    """Cancel only our child group; child handlers clean its GPU worker groups."""
    with Path(log).open('x') as output:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            cwd=Path(__file__).resolve().parents[3])
        try:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--hold', action='store_true', required=True)
    parser.add_argument('--mib', type=int, required=True)
    args = parser.parse_args()
    require(0 < args.mib < 32768, 'Holder allocation must be 1..32767 MiB')
    with termination_signals():
        holder(args.mib)


if __name__ == '__main__':
    main()
