"""CPU-only stage deadlines and an exec'd monitor, independent of the GPU/GIL."""

import importlib.util
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_active = ContextVar("inference_watchdog", default=None)
_DEFAULTS = {
    "init": 1800,
    "task": 1800,
    "broadcast": 60,
    "prepare": 300,
    "inference": 1800,
    "input_encode": 300,
    "text_encode": 120,
    "denoise": 900,
    "decode": 600,
    "barrier": 60,
    "output": 300,
    "cleanup": 60,
}


def inference_stage(name):
    """Bound a synchronous runner stage only when called by a watched server."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            watchdog = _active.get()
            if watchdog is None:
                return function(*args, **kwargs)
            with watchdog.stage(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


class InferenceWatchdog:
    def __init__(self, rank, timeouts=None):
        self.rank = rank
        self.timeouts = {}
        for name, default in _DEFAULTS.items():
            value = float((timeouts or {}).get(name, os.getenv(f"LIGHTX2V_TIMEOUT_{name.upper()}_S", default)))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name} timeout: expected finite positive seconds")
            self.timeouts[name] = value
        self._stack = []
        self.task_id = None
        self.closed = False
        self.failure = None
        parent, child = socket.socketpair()
        parent.settimeout(0.2)
        try:
            self.process = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--monitor", str(child.fileno()), str(os.getpid()), str(rank)],
                pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self.socket = parent
        try:
            parent.settimeout(5)
            ready = b""
            while len(ready) < 6:
                chunk = parent.recv(6 - len(ready))
                if not chunk:
                    raise RuntimeError("Inference monitor exited before readiness")
                ready += chunk
            if ready != b"READY\n":
                raise RuntimeError("Invalid inference monitor handshake")
        except BaseException:
            parent.close()
            self.process.kill()
            self.process.wait(timeout=2)
            raise
        parent.settimeout(0.2)

    def _publish(self):
        if self.closed or self.process.poll() is not None:
            self.failure = "watchdog unavailable"
            self._terminate(self.failure)
        state = self.snapshot()
        try:
            self.socket.sendall((json.dumps(state) + "\n").encode())
        except OSError:
            self.failure = "watchdog control channel failed"
            self._terminate(self.failure)

    @staticmethod
    def _terminate(reason):
        # Reuse #1685's GPU-free, bounded diagnostic/exit path in the rank.
        try:
            spec = importlib.util.spec_from_file_location("_watchdog_inference_errors", os.path.join(os.path.dirname(__file__), "inference_errors.py"))
            errors = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(errors)
            errors.terminate_inference_process(reason)
        finally:
            os._exit(1)

    def snapshot(self):
        now = time.monotonic()
        stack = tuple(self._stack)
        expired_stage, deadline = min(stack, key=lambda entry: entry[1]) if stack else (None, None)
        expired = deadline is not None and now >= deadline
        alive = not self.closed and self.process.poll() is None
        return {
            "status": "ok" if alive and not expired and self.failure is None else "unhealthy",
            "rank": self.rank,
            "task_id": self.task_id,
            "stage": stack[-1][0] if stack else "idle",
            "deadline": deadline,
            "timeout_stage": expired_stage,
            "remaining_s": max(0, deadline - now) if deadline is not None else None,
            "reason": self.failure or ("stage deadline exceeded" if expired else None) or (None if alive else "watchdog unavailable"),
        }

    @contextmanager
    def stage(self, name):
        # Preserve expired deadlines: a late return cannot clear a timed-out stage.
        if self.snapshot()["status"] != "ok":
            self._terminate("inference watchdog unhealthy")
        self._stack.append((name, time.monotonic() + self.timeouts[name]))
        self._publish()
        try:
            yield
        finally:
            if self.snapshot()["status"] != "ok":
                self._terminate("inference stage deadline exceeded")
            self._stack.pop()
            self._publish()

    @contextmanager
    def task(self, task_id):
        self.task_id = str(task_id)[:256]
        token = _active.set(self)
        try:
            with self.stage("task"):
                yield
        finally:
            _active.reset(token)
            self.task_id = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.socket.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


def _monitor(fd, parent_pid, rank):
    """No GPU imports, Python threads, logging locks, or device cleanup here."""
    channel = socket.socket(fileno=fd)
    channel.sendall(b"READY\n")
    pending = b""
    state = {"deadline": None}
    # Linux pidfd prevents signalling a reused PID after the rank exits.
    try:
        pidfd = os.pidfd_open(parent_pid) if hasattr(os, "pidfd_open") else None
    except OSError:
        pidfd = None  # Older kernels/security policies: verify the parent relation.
    try:
        while os.getppid() == parent_pid:
            deadline = state["deadline"]
            wait = min(0.1, max(0, deadline - time.monotonic())) if deadline is not None else 0.1
            # Consume already-queued completions before deciding to kill. The
            # rank rejects late completions itself; scheduling delay of this
            # monitor must not kill a stage that finished within its deadline.
            if select.select([channel], [], [], wait)[0]:
                chunk = channel.recv(16384)
                if not chunk:
                    return
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    state = json.loads(line)
                if select.select([channel], [], [], 0)[0]:
                    continue
            deadline = state["deadline"]
            if deadline is not None and time.monotonic() >= deadline:
                # Kill before diagnostics: a full stderr must never delay recovery.
                if pidfd is not None:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                elif os.getppid() == parent_pid:
                    os.kill(parent_pid, signal.SIGKILL)
                from inference_errors import terminate_inference_process

                terminate_inference_process(f"watchdog timeout rank={rank} task={state.get('task_id')} stage={state.get('stage')} deadline_stage={state.get('timeout_stage')}")

    finally:
        channel.close()
        if pidfd is not None:
            os.close(pidfd)


if __name__ == "__main__":
    _monitor(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
