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


def simulate_record(output, returncode, event):
    try:
        (record,) = [json.loads(line) for line in output.splitlines()]
    except ValueError as error:
        raise ToolError("The solver returned an invalid protocol response.") from error
    if not isinstance(record, dict) or record.get("protocol_version") != 1:
        raise ToolError("The solver protocol version is unsupported.")
    if returncode == 0 and record.get("event") == event:
        return record
    raise ToolError(" ".join(str(record.get(key, "")) for key in ("code", "field", "cause")).strip(), record)


def checked(name, *args, timeout=20, event="result"):
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
        return simulate_record(result.stdout, result.returncode, event)
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
    def __init__(self, scene, stage, command):
        self.scene = scene
        self.stage = stage
        self.messages = Queue(maxsize=64)
        self.finished_reading = threading.Event()
        self.stop_reader = threading.Event()
        self.terminal = None
        self.cancel_deadline = None
        self.started_at = time.monotonic()
        self.last_heartbeat = None
        self.files = tempfile.TemporaryDirectory(prefix="fluidity-")
        try:
            args = command(Path(self.files.name))
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


def cutter_checkpoint(cache, frame):
    return cache / f"cutter-{frame:06d}.chk"


def check_cutter(packet):
    with tempfile.TemporaryDirectory(prefix="fluidity-cutter-check-") as directory:
        inputs = Path(directory) / "cutter.txt"
        inputs.write_text(packet, encoding="ascii", newline="\n")
        checked("simulate", "cutter-frame", inputs, "-", "-", event="cutter_frame")


class FrameJob:
    def __init__(self, scene, cache, last_frame):
        self.scene = scene
        self.cache = cache
        self.last_frame = last_frame
        self.process = None
        self.output = None
        self.terminal = None
        self.cancelled = False
        self.started_at = time.monotonic()
        self.committed_frame = 0
        binary("simulate")

    def cancel(self):
        self.cancelled = True

    def poll(self):
        settings = self.scene.fluidity
        settings.elapsed = time.monotonic() - self.started_at
        settings.heartbeat_age = -1
        settings.committed_frame = self.committed_frame if self.committed_frame else -1
        try:
            if self.process is not None:
                code = self.process.poll()
                if code is None:
                    return False
                self.output.seek(0)
                record = simulate_record(self.output.read(131073).decode("utf-8"), code, self.frame_event)
                self.output.close()
                self.output = None
                self.process = None
                next_frame = self.committed_frame + 1
                self.check_record(record, next_frame)
                self.committed_frame = next_frame
                settings.committed_frame = next_frame
                return False
            if self.cancelled:
                self.terminal = {"name": "CANCELLED"}
                return True
            if self.committed_frame == self.last_frame:
                self.terminal = {"event": "result", "frame": self.committed_frame}
                return True
            self.output = tempfile.TemporaryFile()
            self.process = subprocess.Popen(
                self.frame_command(),
                stdout=self.output,
                stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW,
            )
            return False
        except (OSError, ValueError, subprocess.SubprocessError, ToolError) as error:
            self.terminal = {"event": "error", "field": self.field, "cause": str(error)}
            return True

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        if self.output is not None:
            self.output.close()


class CutterJob(FrameJob):
    field = "cutter"
    frame_event = "cutter_frame"

    def __init__(self, scene, cache, packet, last_frame, resume):
        super().__init__(scene, cache, last_frame)
        self.inputs = cache / "cutter.txt"
        self.substeps = int(packet.splitlines()[1].split()[-1])
        if resume:
            if not self.inputs.is_file() or self.inputs.read_bytes() != packet.encode("ascii"):
                raise ToolError("The Cutter settings changed. Choose a fresh Bake cache.")
            while self.committed_frame < last_frame and cutter_checkpoint(cache, self.committed_frame + 1).is_file():
                self.committed_frame += 1
            if self.committed_frame:
                checkpoint = cutter_checkpoint(cache, self.committed_frame)
                record = checked("simulate", "cutter-inspect", checkpoint, self.inputs, event="cutter_inspect")
                if record["accepted_step"] != self.committed_frame * self.substeps:
                    raise ToolError("The Cutter checkpoint has an unexpected accepted step.")
        else:
            cache.mkdir(parents=True)
            with self.inputs.open("x", encoding="ascii", newline="\n") as stream:
                stream.write(packet)


    def check_record(self, record, frame):
        if record["accepted_step"] != frame * self.substeps:
            raise ToolError("The Cutter solver returned an unexpected accepted step.")

    def frame_command(self):
        previous = cutter_checkpoint(self.cache, self.committed_frame) if self.committed_frame else "-"
        return [binary("simulate"), "cutter-frame", self.inputs, previous,
                cutter_checkpoint(self.cache, self.committed_frame + 1)]


def rigid_checkpoint(cache, frame):
    return cache / f"rigid-{frame:06d}.chk"


def prepare_rigid(packets, directory):
    liquid, rigid = packets
    inputs, scene, seed, body = (directory / name for name in ("scene.txt", "scene.flrs", "state.flst", "rigid.txt"))
    inputs.write_text(liquid, encoding="ascii", newline="\n")
    body.write_text(rigid, encoding="ascii", newline="\n")
    checked("simulate", "blender-resolve", binary("simulate"), inputs, scene, seed, event="blender_resolved")
    return scene, seed, body


def check_rigid(packets):
    with tempfile.TemporaryDirectory(prefix="fluidity-rigid-check-") as directory:
        paths = prepare_rigid(packets, Path(directory))
        checked("simulate", "rigid-frame", *paths, "-", "-", event="rigid_frame")


def inspect_rigid(cache, frame):
    paths = (cache / name for name in ("scene.flrs", "state.flst", "rigid.txt"))
    return checked("simulate", "rigid-inspect", *paths, rigid_checkpoint(cache, frame), event="rigid_inspect")


class RigidJob(FrameJob):
    field = "rigid_box"
    frame_event = "rigid_frame"

    def __init__(self, scene, cache, packets, last_frame, resume):
        super().__init__(scene, cache, last_frame)
        self.paths = tuple(cache / name for name in ("scene.flrs", "state.flst", "rigid.txt"))
        if resume:
            for path, packet in zip((cache / "scene.txt", self.paths[2]), packets, strict=True):
                if not path.is_file() or path.read_bytes() != packet.encode("ascii"):
                    raise ToolError("The rigid bake settings changed. Choose a fresh Bake cache.")
            for frame in range(1, last_frame + 1):
                if rigid_checkpoint(cache, frame).is_file():
                    if frame != self.committed_frame + 1:
                        raise ToolError("The rigid bake cache has a missing frame.")
                    self.check_record(inspect_rigid(cache, frame), frame)
                    self.committed_frame = frame
        else:
            # Resolve and validate before creating a persistent bake directory.
            with tempfile.TemporaryDirectory(prefix="fluidity-rigid-prepare-") as directory:
                staging = Path(directory)
                paths = prepare_rigid(packets, staging)
                checked("simulate", "rigid-frame", *paths, "-", "-", event="rigid_frame")
                cache.mkdir(parents=True)
                for path in (*paths, staging / "scene.txt"):
                    with (cache / path.name).open("xb") as output:
                        output.write(path.read_bytes())

    def check_record(self, record, frame):
        if record["frame"] != frame or record["substep"] != 0:
            raise ToolError("The rigid solver returned an unexpected frame.")

    def frame_command(self):
        previous = rigid_checkpoint(self.cache, self.committed_frame) if self.committed_frame else "-"
        return [binary("simulate"), "rigid-frame", *self.paths, previous,
                rigid_checkpoint(self.cache, self.committed_frame + 1)]


def start_rigid(scene, cache, packets, last_frame, resume):
    if is_running(scene):
        raise ToolError("This scene already has a running job.")
    _JOBS[scene.as_pointer()] = RigidJob(scene, cache, packets, last_frame, resume)

def start_cutter(scene, cache, packet, last_frame, resume):
    if is_running(scene):
        raise ToolError("This scene already has a running job.")
    _JOBS[scene.as_pointer()] = CutterJob(scene, cache, packet, last_frame, resume)

def is_running(scene):
    return scene.as_pointer() in _JOBS


def start(scene, cache, packet, backend, last_frame, resume):
    def command(directory):
        path, seed = prepare(packet, backend, directory)
        if resume:
            checked("simulate", "check", "--scene", path, "--cache", cache)
        args = [
            binary("simulate"), "run", "--stage", "simulate", "--scene", path,
            "--initial-state", seed, "--cache", cache, "--backend", backend,
            "--frame-end", str(last_frame),
        ]
        if resume:
            args.append("--resume")
        return args

    return start_job(scene, "simulate", command)


def start_mesh(scene, source, output, last_frame, spacing, sigma, resume):
    def command(_directory):
        args = [
            binary("simulate"), "run", "--stage", "mesh", "--cache", source,
            "--output-cache", output, "--frame-end", str(last_frame),
            "--surface-spacing", str(spacing), "--surface-sigma", str(sigma),
        ]
        if resume:
            args.append("--resume")
        return args

    return start_job(scene, "mesh", command)


def start_job(scene, stage, command):
    if is_running(scene):
        raise ToolError("This scene already has a running job.")
    job = Job(scene, stage, command)
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
        frame_job = isinstance(job, FrameJob)
        if record.get("event") == "result" and (frame_job or job.process.returncode == 0):
            settings.committed_frame = max(settings.committed_frame, record["frame"])
            if isinstance(job, CutterJob):
                settings.message = "Cutter bake complete. Inspect a committed frame for particle sides."
            elif isinstance(job, RigidJob):
                settings.message = "Rigid bake complete."
            elif job.stage == "gas":
                settings.message = "Gas bake complete. Attach the committed Volume to render."
            elif job.stage == "mesh":
                settings.message = "Surface build complete. Attach the committed surface to render."
            else:
                settings.message = "Bake complete. Attach the committed points to render."
        elif record.get("name") == "CANCELLED":
            settings.message = "Cancelled. Committed frames are available for Resume."
        elif record.get("event") == "error":
            settings.message = (
                "Surface build failed. Committed frames are preserved."
                if not frame_job and job.stage == "mesh"
                else "Bake failed. Committed frames are preserved."
            )
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
