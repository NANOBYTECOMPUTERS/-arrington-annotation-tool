import json
import queue
import subprocess
import threading
from collections.abc import Sequence
from typing import Any


class WorkerProtocolError(RuntimeError):
    pass


class WorkerTimeoutError(WorkerProtocolError):
    pass


class WorkerExitedError(WorkerProtocolError):
    pass


class WorkerError(WorkerProtocolError):
    pass


_STDOUT_CLOSED = object()


class WorkerClient:
    def __init__(self, command: Sequence[str], request_timeout: float = 30.0, shutdown_timeout: float = 5.0):
        self.command = list(command)
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._stdout_queue: queue.Queue[str | object] = queue.Queue()
        self._stderr_chunks: list[str] = []
        self._stderr_lock = threading.Lock()

    def __enter__(self) -> "WorkerClient":
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                if self.process.stdin:
                    self.process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    self.process.stdin.flush()
            except Exception:
                pass
            try:
                self.process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    pass
        else:
            self.process.wait(timeout=self.shutdown_timeout)

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Worker process is not running")
        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            try:
                self.process.stdin.write(line + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise WorkerExitedError(f"Worker exited before accepting request. stderr={self._stderr_text()}") from error
            response_line = self._read_response_line(timeout if timeout is not None else self.request_timeout)
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as error:
            raise WorkerProtocolError(f"Invalid JSON response from worker: {response_line.rstrip()}") from error
        if response.get("ok") is False:
            raise WorkerError(str(response.get("error", "worker request failed")))
        return response

    def _read_response_line(self, timeout: float) -> str:
        try:
            response_line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as error:
            self._kill_worker()
            raise WorkerTimeoutError(f"Worker request timed out after {timeout:.3g}s") from error
        if response_line is _STDOUT_CLOSED:
            raise WorkerExitedError(f"Worker exited without a response. stderr={self._stderr_text()}")
        return str(response_line)

    def _read_stdout(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            self._stdout_queue.put(line)
        self._stdout_queue.put(_STDOUT_CLOSED)

    def _read_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        for chunk in self.process.stderr:
            with self._stderr_lock:
                self._stderr_chunks.append(chunk)
                joined = "".join(self._stderr_chunks)
                if len(joined) > 4000:
                    self._stderr_chunks = [joined[-4000:]]

    def _stderr_text(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_chunks).strip()

    def _kill_worker(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.kill()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
