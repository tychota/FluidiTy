"""Subprocess ownership and bounded main-thread polling for M1-B jobs."""

import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

BIN_DIR = Path(__file__).resolve().parent / "bin"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_JOBS = {}


class ToolError(RuntimeError):
    def __init__(self, detail, record=None):
        super().__init__(detail)
        self.record = record or {}


def binary(name):
    path = BIN_DIR / f"fluidity_{name}.exe"
    if not path.is_file():
        raise ToolError("Install the Windows x64 FluidiTy extension package from disk.")
    return path


def checked(name, *args, timeout=20):
    result = subprocess.run(
        [binary(name), *map(str, args)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
        creationflags=NO_WINDOW,
    )
    if name == "simulate":
        try:
            (record,) = [json.loads(line) for line in result.stdout.splitlines()]
        except ValueError as error:
            raise ToolError("The solver returned an invalid protocol response.") from error
        if not isinstance(record, dict) or record.get("protocol_version") != 1:
            raise ToolError("The solver protocol version is unsupported.")
        if result.returncode == 0 and record.get("event") == "result":
            return record
        raise ToolError(" ".join(str(record.get(key, "")) for key in ("code", "field", "cause")).strip(), record)
    if result.returncode:
        raise ToolError(result.stderr.strip() or "The cache bridge failed.")
    return result.stdout


def prepare(packet, backend, directory):
    directory = Path(directory)
    inputs, scene, state = (directory / name for name in ("scene.txt", "scene.flrs", "state.flst"))
    inputs.write_text(packet, encoding="ascii")
    checked("blender_bridge", "resolve", binary("simulate"), backend, inputs, scene, state)
    return scene, state


class Job:
    def __init__(self, scene, cache, packet, backend, last_frame, resume):
        self.scene = scene
        self.messages = Queue(maxsize=64)
        self.finished_reading = threading.Event()
        self.stop_reader = threading.Event()
        self.terminal = None
        self.cancel_deadline = None
        self.started_at = time.monotonic()
        self.last_heartbeat = None
        self.files = tempfile.TemporaryDirectory(prefix="fluidity-")
        try:
            path, seed = prepare(packet, backend, self.files.name)
            if resume:
                checked("simulate", "check", "--scene", path, "--cache", cache)
            args = [
                binary("simulate"),
                "run",
                "--stage",
                "simulate",
                "--scene",
                path,
                "--initial-state",
                seed,
                "--cache",
                cache,
                "--backend",
                backend,
                "--frame-end",
                str(last_frame),
            ]
            if resume:
                args.append("--resume")
            with open(Path(self.files.name) / "stderr.txt", "wb") as stderr:
                self.process = subprocess.Popen(
                    args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, creationflags=NO_WINDOW
                )
        except (OSError, ValueError, subprocess.SubprocessError, ToolError):
            self.files.cleanup()
            raise
        self.reader = threading.Thread(target=self._read_stdout, name="FluidiTy protocol reader", daemon=True)
        self.reader.start()

    def _read_stdout(self):
        # This thread only reads bytes and never calls Blender.
        try:
            while not self.stop_reader.is_set():
                line = self.process.stdout.readline(65_537)
                if not line:
                    break
                try:
                    if len(line) > 65_536 or not line.endswith(b"\n"):
                        raise ValueError("record length")
                    message = json.loads(line)
                    if not isinstance(message, dict) or message.get("protocol_version") != 1:
                        raise ValueError("protocol version")
                except ValueError:
                    message = {"event": "protocol_error"}
                # Backpressure the solver rather than lose a terminal or commit event.
                while not self.stop_reader.is_set():
                    try:
                        self.messages.put(message, timeout=0.05)
                        break
                    except Full:
                        continue
        finally:
            self.finished_reading.set()

    def cancel(self):
        if self.cancel_deadline is not None:
            return
        self.cancel_deadline = time.monotonic() + 10
        try:
            self.process.stdin.write(b"cancel\n")
            self.process.stdin.flush()
        except OSError:
            pass

    def poll(self):
        now = time.monotonic()
        settings = self.scene.fluidity
        for _ in range(16):
            try:
                record = self.messages.get_nowait()
            except Empty:
                break
            event = record.get("event")
            if event == "frame" and record.get("published") is True:
                settings.committed_frame = max(settings.committed_frame, record["frame"])
            elif event == "heartbeat":
                self.last_heartbeat = now
                frame = record.get("last_published_frame")
                if frame is not None:
                    settings.committed_frame = max(settings.committed_frame, frame)
            elif event in {"result", "error", "protocol_error"}:
                self.terminal = record
                if event == "protocol_error":
                    self.cancel()
        settings.elapsed = now - self.started_at
        settings.heartbeat_age = -1 if self.last_heartbeat is None else now - self.last_heartbeat
        if self.cancel_deadline is not None and now >= self.cancel_deadline and self.process.poll() is None:
            self.process.kill()
        return self.process.poll() is not None and self.finished_reading.is_set() and self.messages.empty()

    def close(self):
        self.stop_reader.set()
        if self.process.poll() is None:
            self.cancel()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join()
        self.process.stdin.close()
        self.process.stdout.close()
        self.files.cleanup()


def is_running(scene):
    return scene.as_pointer() in _JOBS


def start(scene, cache, packet, backend, last_frame, resume):
    if is_running(scene):
        raise ToolError("This scene already has a running job.")
    job = Job(scene, cache, packet, backend, last_frame, resume)
    _JOBS[scene.as_pointer()] = job
    return job


def cancel(scene):
    job = _JOBS.get(scene.as_pointer())
    if job is None:
        raise ToolError("No job is running.")
    job.cancel()


def tick():
    for key, job in list(_JOBS.items()):
        if not job.poll():
            continue
        record = job.terminal or {}
        settings = job.scene.fluidity
        if record.get("event") == "result" and job.process.returncode == 0:
            settings.committed_frame = max(settings.committed_frame, record["frame"])
            settings.message = "Bake complete. Attach the committed points to render."
        elif record.get("name") == "CANCELLED":
            settings.message = "Cancelled. Committed frames are available for Resume."
        elif record.get("event") == "error":
            settings.message = "Bake failed. Committed frames are preserved."
            settings.diagnostic = " ".join(str(record.get(key, "")) for key in ("code", "field", "cause")).strip()
        else:
            settings.message = "Job ended unexpectedly. Committed frames are preserved."
        job.close()
        del _JOBS[key]
    return 0.1 if _JOBS else None


def close_all():
    for job in _JOBS.values():
        job.close()
    _JOBS.clear()
