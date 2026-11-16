"""CPU subprocess tests: real deadlines kill hung ranks, including a held GIL."""

import ast
import importlib.util
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "lightx2v/utils/inference_watchdog.py"
spec = importlib.util.spec_from_file_location("inference_watchdog", MODULE)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
PRELUDE = f"""
import importlib.util, os, sys, time, signal
spec = importlib.util.spec_from_file_location('inference_watchdog', {str(MODULE)!r})
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
"""


@unittest.skipUnless(sys.platform == "linux", "Linux worker process supervision")
class WatchdogProcessTests(unittest.TestCase):
    def run_child(self, code):
        return subprocess.run([sys.executable, "-c", PRELUDE + code], capture_output=True, timeout=8, cwd=ROOT)

    def assert_timeout(self, result, stage):
        # SIGKILL of the actual rank, not an unrelated monitor crash/SystemExit.
        self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr.decode())
        self.assertIn(b"watchdog timeout rank=1 task=hang", result.stderr)
        self.assertIn(f"stage={stage}".encode(), result.stderr)
        self.assertNotIn(b"UNREACHABLE", result.stdout)

    def test_hang_with_gil_held_is_killed(self):
        self.assert_timeout(
            self.run_child("""
import ctypes
monitor = w.InferenceWatchdog(1, {'text_encode': .6})
with monitor.task('hang'), monitor.stage('text_encode'):
    # PyDLL retains the GIL throughout this native call.
    ctypes.PyDLL(None).sleep(30)
print('UNREACHABLE')
"""),
            "text_encode",
        )

    def test_broadcast_barrier_inference_and_cleanup_hangs(self):
        for stage in ("broadcast", "barrier", "inference", "cleanup"):
            with self.subTest(stage=stage):
                self.assert_timeout(
                    self.run_child(f"""
monitor = w.InferenceWatchdog(1, {{{stage!r}: .5}})
with monitor.task('hang'), monitor.stage({stage!r}):
    time.sleep(30)
print('UNREACHABLE')
"""),
                    stage,
                )

    def test_total_deadline_survives_progress_and_nested_stages(self):
        result = self.run_child("""
monitor = w.InferenceWatchdog(1, {'task': .7, 'text_encode': .5})
with monitor.task('hang'):
    while True:
        with monitor.stage('text_encode'):
            time.sleep(.15)
print('UNREACHABLE')
""")
        self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr.decode())
        self.assertIn(b"deadline_stage=task", result.stderr)

    def test_normal_stages_ordinary_exception_and_idle_do_not_kill(self):
        result = self.run_child("""
monitor = w.InferenceWatchdog(1, {'task': 2, 'text_encode': .3})
for task in ('first', 'second'):
    try:
        with monitor.task(task):
            for _ in range(3):
                with monitor.stage('text_encode'):
                    time.sleep(.1)
            raise ValueError('ordinary validation failure')
    except ValueError:
        pass
assert monitor.snapshot()['stage'] == 'idle'
time.sleep(.4)  # no idle/keepalive deadline
assert monitor.snapshot()['status'] == 'ok'
monitor.close()
assert monitor.process.poll() == 0
""")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_delayed_monitor_accepts_already_completed_stage(self):
        result = self.run_child("""
monitor = w.InferenceWatchdog(1, {'text_encode': .3})
os.kill(monitor.process.pid, signal.SIGSTOP)
try:
    with monitor.task('completed'), monitor.stage('text_encode'):
        time.sleep(.05)
    time.sleep(.4)
finally:
    os.kill(monitor.process.pid, signal.SIGCONT)
time.sleep(.2)
assert monitor.snapshot()['status'] == 'ok'
monitor.close()
""")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_actual_initialization_hang_is_bounded(self):
        result = self.run_child(f"""
from types import SimpleNamespace
sys.path.insert(0, {str(ROOT / "tests")!r})
from test_inference_errors import WorkerIntegrationTests
cls = WorkerIntegrationTests().worker_class(lambda reason: os._exit(1))
ns = cls.init.__globals__
ns['InferenceWatchdog'] = w.InferenceWatchdog
ns['torch'] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
ns['build_startup_config'] = lambda *a: {{'parallel': False}}
ns['logger'] = SimpleNamespace(info=lambda *a: None, exception=lambda *a: None)
ns['build_runner'] = lambda config: time.sleep(30)
os.environ['LIGHTX2V_TIMEOUT_INIT_S'] = '.5'
worker = cls.__new__(cls)
worker.rank, worker.world_size = 1, 1
worker.dist_manager = SimpleNamespace()
args = SimpleNamespace(config_json=None, model_cls='test', model_variant=None, model_path='unused', task='t2i')
worker.init(args)
print('UNREACHABLE')
""")
        self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr.decode())
        self.assertIn(b"stage=init", result.stderr)
        self.assertNotIn(b"UNREACHABLE", result.stdout)

    def test_monitor_failure_is_unhealthy_and_next_task_exits(self):
        result = self.run_child("""
monitor = w.InferenceWatchdog(1)
monitor.process.kill()
monitor.process.wait()
assert monitor.snapshot()['status'] == 'unhealthy'
with monitor.task('after-monitor-death'):
    print('UNREACHABLE')
""")
        self.assertEqual(result.returncode, 1, result.stderr.decode())
        self.assertIn(b"inference watchdog unhealthy", result.stderr)
        self.assertNotIn(b"UNREACHABLE", result.stdout)

    def test_full_stderr_does_not_block_rank_termination(self):
        result = self.run_child("""
import fcntl
r, fd = os.pipe()
fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
try:
    while True:
        os.write(fd, b'x' * 4096)
except BlockingIOError:
    pass
os.dup2(fd, 2)
monitor = w.InferenceWatchdog(1, {'text_encode': .5})
with monitor.task('hang'), monitor.stage('text_encode'):
    time.sleep(30)
""")
        self.assertEqual(result.returncode, -signal.SIGKILL)

    def test_monitor_exits_when_rank_exits_without_cleanup(self):
        result = self.run_child("""
monitor = w.InferenceWatchdog(1)
print(monitor.process.pid, flush=True)
os._exit(0)
""")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        pid = int(result.stdout)
        # A reparented child can briefly be a zombie until init reaps it.
        stat = Path(f"/proc/{pid}/stat")
        try:
            state = stat.read_text().split()[2]
        except FileNotFoundError:
            return
        self.assertEqual(state, "Z")

    def test_actual_worker_wiring_kills_hung_rank_and_preserves_ordinary_error(self):
        for fault in ("text_encode", "barrier", "broadcast", "ordinary"):
            with self.subTest(fault=fault):
                result = self.run_child(f"""
import asyncio
from types import SimpleNamespace
sys.path.insert(0, str({str(ROOT / "tests")!r}))
from test_inference_errors import WorkerIntegrationTests
cls = WorkerIntegrationTests().worker_class(lambda reason: os._exit(1))
worker = cls.__new__(cls)
worker.rank, worker.world_size, worker.lora_dir = 1, 2, None
worker.watchdog = w.InferenceWatchdog(1, {{'text_encode': .5, 'barrier': .5, 'broadcast': .5}})
fault = {fault!r}
def broadcast(data):
    if fault == 'broadcast': time.sleep(30)
    return data
def barrier():
    if fault == 'barrier': time.sleep(30)
@w.inference_stage('text_encode')
def run(data):
    if fault == 'text_encode': time.sleep(30)
    if fault == 'ordinary': raise ValueError('invalid input')
worker.runner = SimpleNamespace(prepare_request=lambda x: x, set_reuse=lambda *a: None, run_request=run)
worker.dist_manager = SimpleNamespace(broadcast_task_data=broadcast, barrier=barrier)
if fault == 'broadcast': worker.rank = 0
out = asyncio.run(worker.process_request({{'task_id': 'hang'}}))
worker.watchdog.close()
if fault == 'ordinary':
    assert out is None  # nonzero rank retains its normal return contract
    print('ordinary completed')
else: print('UNREACHABLE')
""")
                if fault == "ordinary":
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                    self.assertIn(b"ordinary completed", result.stdout)
                else:
                    self.assert_timeout(result, fault)


class WatchdogStateTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux monitor startup")
    def test_monitor_startup_failure_is_not_silently_accepted(self):
        popen = subprocess.Popen
        children = []

        def exit_before_handshake(args, **kwargs):
            child = popen([sys.executable, "-c", "pass"], **kwargs)
            children.append(child)
            return child

        with patch.object(watch.subprocess, "Popen", exit_before_handshake):
            with self.assertRaisesRegex(RuntimeError, "before readiness"):
                watch.InferenceWatchdog(0)
        self.assertIsNotNone(children[0].poll())

    def test_reject_invalid_timeouts_before_spawning(self):
        for value in (0, -1, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                watch.InferenceWatchdog(0, {"task": value})

    def test_expired_state_and_late_return_cannot_refresh_deadline(self):
        monitor = watch.InferenceWatchdog.__new__(watch.InferenceWatchdog)
        monitor.rank, monitor.task_id = 0, "late"
        monitor.closed, monitor.failure = False, None
        monitor.process = SimpleNamespace(poll=lambda: None)
        monitor._stack = [("task", time.monotonic() - 1)]
        self.assertEqual(monitor.snapshot()["status"], "unhealthy")
        monitor._terminate = lambda _: (_ for _ in ()).throw(SystemExit(1))
        with self.assertRaises(SystemExit), monitor.stage("text_encode"):
            self.fail("expired request must not start another stage")

    def test_health_route_uses_state_without_gpu_calls(self):
        path = ROOT / "lightx2v/server/api/server.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ApiServer")
        setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_setup_routes")
        routes = {}

        class App:
            def get(self, path):
                def register(fn):
                    routes[path] = fn
                    return fn

                return register

            def middleware(self, kind):
                return lambda fn: fn

            def include_router(self, router):
                pass

        state = {"status": "ok", "stage": "idle"}
        service = SimpleNamespace(health_status=lambda: state)
        namespace = dict(get_services=lambda: SimpleNamespace(inference_service=service), JSONResponse=lambda content, status_code: (status_code, content), create_api_router=lambda: None)
        exec(compile(ast.Module(body=[setup], type_ignores=[]), str(path), "exec"), namespace)
        server = SimpleNamespace(app=App(), stop_processing=SimpleNamespace(is_set=lambda: False), processing_thread=SimpleNamespace(is_alive=lambda: True))
        namespace["_setup_routes"](server)
        self.assertEqual(routes["/health"]()[0], 200)
        for reason in ("stage deadline exceeded", "watchdog unavailable", "worker not ready"):
            state = {"status": "unhealthy", "reason": reason}
            self.assertEqual(routes["/health"](), (503, state))
        state = {"status": "ok"}
        server.processing_thread = SimpleNamespace(is_alive=lambda: False)
        self.assertEqual(routes["/health"]()[0], 503)
        service = None
        self.assertEqual(routes["/health"]()[0], 503)

    def test_h3_stage_guards_enclose_profiler_exit(self):
        tree = ast.parse((ROOT / "lightx2v/models/runners/minimax_h3/minimax_h3_runner.py").read_text())
        expected = {"run_text_encoder": "text_encode", "_run_input_encoder_local_h3": "input_encode", "run_segment": "denoise", "run_vae_decoder": "decode"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in expected:
                decorator = node.decorator_list[0]
                self.assertEqual(decorator.func.id, "inference_stage")
                self.assertEqual(decorator.args[0].value, expected.pop(node.name))
        self.assertFalse(expected)


if __name__ == "__main__":
    unittest.main()
