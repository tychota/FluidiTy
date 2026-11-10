"""Exercise frozen capture qualification without launching a native binary."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_liquid_acceptance as runner
from liquid_acceptance_metrics import PISTON_CASE, PISTON_COMPARISON_VOLUMES, PISTON_REFINED_CASE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(stdout, event, *, fixture=False, **fields):
    version_field = "fixture_version" if fixture else "protocol_version"
    stdout.write((json.dumps({version_field: 1, "event": event, **fields}) + "\n").encode())


@pytest.fixture
def launch(tmp_path, monkeypatch):
    solver, fixture = tmp_path / "solver.exe", tmp_path / "fixture.exe"
    solver.write_bytes(b"pinned solver")
    fixture.write_bytes(b"pinned fixture")
    pins = {"solver": sha(solver), "fixture": sha(fixture)}
    controls = {}
    invocations = []

    def process(command, *, stdout, **kwargs):
        executable, action, *arguments = command
        invocations.append(action)
        if action == "schema":
            assert executable == str(solver) and not arguments
            emit(stdout, "result", command="schema", execution_available=True,
                 build_digest=controls.get("schema_digest", "a" * 64))
        elif action == "device-info":
            assert executable == str(solver)
            backend = arguments[1]
            fields = {}
            if backend == "cuda":
                fields = dict(device_index=0, device_name="RTX process double", compute_major=12, compute_minor=0,
                              compiled_architecture=1200, sample_count=17, driver_version=13000, runtime_version=13000,
                              launch_result=0, copy_result=0, synchronize_result=0)
            emit(stdout, "result", command=action, backend=backend, cpu_available=True, cuda_compiled=True,
                 cuda_available=True if backend == "cuda" else None, **fields)
        elif action == "write-scene":
            assert executable == str(fixture) and arguments[2] == str(solver)
            case, backend, _, name = arguments
            scene = Path(name)
            scene.write_bytes(b"scene bytes")
            scene.with_suffix(".flrs.flst").write_bytes(b"seed")
            controls["author"] = dict(case=case, backend=backend, solver_digest="a" * 64, seed_digest="b" * 64,
                                      resolved_fingerprint=("c" if case == PISTON_CASE else "d") * 64,
                                      execution_id=("e" if case == PISTON_CASE else "f") * 64, seed_byte_length=4)
            emit(stdout, "scene", fixture=True, **controls["author"])
        elif action == "run":
            assert executable == str(solver)
            frame = int(arguments[arguments.index("--frame-end") + 1])
            backend = arguments[arguments.index("--backend") + 1]
            if frame != 1 and controls.get("interrupted"):
                return SimpleNamespace(returncode=7)
            cache = Path(arguments[arguments.index("--cache") + 1])
            cache.mkdir()
            (cache / "resolved.flrs").write_bytes(b"scene bytes")
            (cache / "data").write_bytes(b"immutable cache data")
            if frame != 1 and controls.get("mutate_author"):
                receipt = cache.parent / "scene-author.stdout.jsonl"
                contents = json.loads(receipt.read_bytes()) | {"extra": "written after authoring"}
                receipt.write_text(json.dumps(contents) + "\n", encoding="utf-8")
            if backend == "cuda":
                fingerprint = controls["author"]["resolved_fingerprint"]
                (cache / ".simulate-cuda-compatibility").write_bytes(b"FLCUDA1\n" + fingerprint.encode() + b"\n")
            actual = {}
            if backend == "cuda":
                actual = dict(compiled_architecture=1200, snapshot_copy_milliseconds=1,
                              cuda_execution_device_digest=controls.get("device_digest", "1" * 64),
                              cuda_execution_device_index=controls.get("device_index", 0))
                if frame != 1 and controls.get("missing_device"):
                    actual.pop("cuda_execution_device_digest")
            elif frame != 1 and controls.get("cpu_with_cuda"):
                actual["cuda_execution_device_digest"] = "1" * 64
            emit(stdout, "job_measurement", backend=backend, command="run", status="OK")
            emit(stdout, "result", **controls["author"], **actual, command="run", read_only=False, frame=frame,
                 block_count=1, cell_count=1, face_count=1, particle_count=1024, next_particle_id=1025,
                 source_next_child_ordinals=[], source_fractional_remainders=[])
        elif action == "read-tank-scene":
            assert executable == str(fixture)
            cache, fingerprint = Path(arguments[0]), arguments[1]
            assert (cache / "resolved.flrs").read_bytes() == b"scene bytes"
            assert fingerprint == controls["author"]["resolved_fingerprint"]
            if controls.get("mutate"):
                (cache / "data").write_bytes(b"changed by fixture")
            if controls.get("mutate_solver"):
                solver.write_bytes(b"changed solver")
            emit(stdout, "resolved_tank", fixture=True,
                 **(controls["author"] | {"solver_digest": controls.get("decoded_digest", "a" * 64)}))
        else:
            pytest.fail(f"unexpected process {command}")
        return SimpleNamespace(returncode=0)

    def evaluate(output, fixture_path, backend, case, device, authored, records, *, capture, resolved, verify_inputs, cache_files):
        verify_inputs()
        return {
            "case": case, "backend": backend, "scene_author": authored, "resolved_tank": resolved,
            "final_receipt": records[-1], "passed": controls.get("passed", True),
            "maximum_dt_seconds": 0.01 if case == PISTON_CASE else 0.005,
            "tight_spatial": {field: 0.25 for field in (*PISTON_COMPARISON_VOLUMES, "denominator_m3")},
        }


    monkeypatch.setattr(runner.subprocess, "run", process)
    monkeypatch.setattr(runner, "verify_frozen_tank", evaluate)
    monkeypatch.setattr(runner, "verify_resolved_tank", lambda *args: None)

    def live(name="live", backend="cuda", case=PISTON_CASE, expected_pins=pins):
        output = tmp_path / name
        result = runner.run_acceptance(solver, fixture, backend, case, output, pins=expected_pins)
        return output, result

    def offline(capture, expected_sha=None, *, backend="cuda", case=PISTON_CASE):
        output = tmp_path / "verification"
        manifest = capture / "capture-manifest.json"
        return runner.verify_captured_tank(
            capture, output, solver, fixture, backend, case, pins=pins,
            manifest_sha256=sha(manifest) if expected_sha is None else expected_sha,
        )

    return SimpleNamespace(live=live, offline=offline, controls=controls, pins=pins, calls=invocations)


@pytest.mark.parametrize("kind", ["solver", "fixture"])
def test_wrong_binary_pin_refuses_to_run_a_process(launch, kind):
    with pytest.raises(ValueError, match=rf"{kind} SHA-256"):
        launch.live(expected_pins=launch.pins | {kind: "0" * 64})
    assert launch.calls == []


@pytest.mark.parametrize("field", ["schema_digest", "decoded_digest"])
def test_schema_identity_must_match_authored_and_decoded_scene(launch, field):
    launch.controls[field] = "0" * 64
    with pytest.raises(ValueError, match="solver_digest"):
        launch.live()


@pytest.mark.parametrize(("control", "value", "error"), [
    ("missing_device", True, "cuda_execution_device_digest"),
    ("device_digest", "A" * 64, "cuda_execution_device_digest"),
    ("device_index", True, "cuda_execution_device_index"),
    ("device_index", 1, "device index"),
])
def test_cuda_uses_actual_run_identity(launch, control, value, error):
    launch.controls[control] = value
    with pytest.raises(ValueError, match=error):
        launch.live()


def test_interrupted_run_does_not_publish_a_manifest(tmp_path, launch):
    launch.controls["interrupted"] = True
    with pytest.raises(ValueError, match="exited 7"):
        launch.live()
    assert not (tmp_path / "live/capture/capture-manifest.json").exists()


def test_offline_verification_uses_pinned_manifest_and_writes_separately(launch):
    output, live = launch.live()
    capture = output / "capture"
    before = {path: (path.stat().st_mtime_ns, sha(path)) for path in capture.rglob("*") if path.is_file()}
    verified = launch.offline(capture)
    assert verified == live
    assert before == {path: (path.stat().st_mtime_ns, sha(path)) for path in capture.rglob("*") if path.is_file()}
    assert live["capture_provenance"]["manifest_sha256"] == sha(capture / "capture-manifest.json")
    assert launch.calls.count("run") == 2


def test_legacy_capture_with_no_manifest_is_diagnostic(tmp_path, launch):
    capture = tmp_path / "legacy"
    capture.mkdir()
    (capture / "status.txt").write_text("exit=0\n")
    with pytest.raises(ValueError, match="capture-manifest.json"):
        launch.offline(capture, "0" * 64)
    assert sorted(path.name for path in capture.iterdir()) == ["status.txt"]
    assert launch.calls == []


@pytest.mark.parametrize("content", [b"[]", b'{"manifest_version":1,"manifest_version":1}', b"{}"])
def test_even_a_pinned_malformed_manifest_is_refused(launch, content):
    output, _ = launch.live()
    capture = output / "capture"
    (capture / "capture-manifest.json").write_bytes(content)
    with pytest.raises(ValueError, match="manifest|duplicate JSON key"):
        launch.offline(capture)


def test_manifest_pin_is_independent_of_captured_manifest(launch):
    output, _ = launch.live()
    with pytest.raises(ValueError, match="manifest SHA-256"):
        launch.offline(output / "capture", "0" * 64)


@pytest.mark.parametrize("name", ["scene.flrs", "scene.flrs.flst", "simulate-run.stdout.jsonl",
                                 "cache/.simulate-cuda-compatibility", "cache/data"])
def test_any_changed_captured_file_refuses_qualification(launch, name):
    output, _ = launch.live()
    capture = output / "capture"
    path = capture / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="capture file inventory"):
        launch.offline(capture)


def test_missing_or_symlinked_file_refuses_qualification(tmp_path, launch):
    output, _ = launch.live()
    capture = output / "capture"
    path = capture / "scene.flrs"
    target = tmp_path / "scene-copy"
    path.rename(target)
    with pytest.raises(ValueError, match="capture file inventory"):
        launch.offline(capture)
    try:
        path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="symlink"):
        launch.offline(capture)


def test_refuses_output_inside_capture(launch):
    output, _ = launch.live()
    capture = output / "capture"
    with pytest.raises(ValueError, match="outside the capture"):
        runner.verify_captured_tank(
            capture, capture / "evaluation", None, None, "cuda", PISTON_CASE, pins=launch.pins,
            manifest_sha256=sha(capture / "capture-manifest.json"),
        )
    assert not (capture / "evaluation").exists()


@pytest.mark.parametrize(("control", "error"), [("mutate", "changed capture"), ("mutate_solver", "solver SHA-256")])
def test_changes_by_captured_reader_disqualify_the_result(launch, control, error):
    output, _ = launch.live()
    launch.controls[control] = True
    with pytest.raises(ValueError, match=error):
        launch.offline(output / "capture")


def test_cpu_does_not_qualify_cuda_execution_fields(launch):
    launch.controls["cpu_with_cuda"] = True
    with pytest.raises(ValueError, match="CPU"):
        launch.live(backend="cpu")


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_temporal_pair_qualifies_distinct_scenes_on_same_backend(tmp_path, launch, backend):
    output, primary = launch.live(backend=backend)
    _, refined = launch.live("refined", backend, PISTON_REFINED_CASE)
    reference = output / f"{PISTON_CASE}-measurements.json"
    assert primary["resolved_tank"]["resolved_fingerprint"] != refined["resolved_tank"]["resolved_fingerprint"]
    result = runner.verify_temporal_reference(reference, sha(reference), refined)
    assert result["passed"] is True and result["temporal_passed"] is True
    assert result["denominator_m3"] == 0.25
    if backend == "cuda":
        primary_record = output / "capture/cache/.simulate-cuda-compatibility"
        refined_record = tmp_path / "refined/capture/cache/.simulate-cuda-compatibility"
        assert primary_record.read_bytes() != refined_record.read_bytes()


@pytest.mark.parametrize(("field", "message"), [
    ("solver_sha256", "solver_sha256"), ("fixture_sha256", "fixture_sha256"), ("seed_digest", "seed_digest"),
    ("resolved_fingerprint", "resolved_fingerprint"), ("execution_id", "execution_id"),
    ("cuda_execution_device_digest", "cuda_execution_device_digest"),
])
def test_temporal_pair_rejects_mismatched_or_reused_identity(tmp_path, launch, field, message):
    _, primary = launch.live()
    _, refined = launch.live("refined", "cuda", PISTON_REFINED_CASE)
    if field in ("solver_sha256", "fixture_sha256"):
        primary["capture_provenance"][field] = "0" * 64
    elif field == "cuda_execution_device_digest":
        primary["capture_provenance"][field] = primary["final_receipt"][field] = "0" * 64
    else:
        value = "0" * 64 if field == "seed_digest" else refined["scene_author"][field]
        for record in ("capture_provenance", "scene_author", "resolved_tank"):
            primary[record][field] = value
        if field != "seed_digest":
            primary["final_receipt"][field] = value
    reference = tmp_path / "external.json"
    reference.write_text(json.dumps(primary), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        runner.verify_temporal_reference(reference, sha(reference), refined)


@pytest.mark.parametrize("bad_reference", [[], {"scene_author": []}, {"capture_provenance": []}, {"final_receipt": []}])
def test_pinned_primary_still_requires_a_qualified_object(tmp_path, launch, bad_reference):
    _, primary = launch.live()
    _, refined = launch.live("refined", "cuda", PISTON_REFINED_CASE)
    reference = tmp_path / "external.json"
    contents = primary | bad_reference if isinstance(bad_reference, dict) else bad_reference
    reference.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(ValueError, match="temporal reference"):
        runner.verify_temporal_reference(reference, sha(reference), refined)


def test_untrusted_primary_digest_refuses_comparison(launch):
    output, _ = launch.live()
    _, refined = launch.live("refined", "cuda", PISTON_REFINED_CASE)
    with pytest.raises(ValueError, match="temporal reference SHA-256"):
        runner.verify_temporal_reference(output / f"{PISTON_CASE}-measurements.json", "0" * 64, refined)


def test_failed_physics_remains_a_written_typed_measurement(launch):
    launch.controls["passed"] = False
    output, result = launch.live()
    measurements = json.loads((output / f"{PISTON_CASE}-measurements.json").read_text())
    assert result == measurements
    assert measurements["passed"] is False
    assert (output / "capture/capture-manifest.json").is_file()



def test_missing_temporal_metric_is_an_ordinary_validation_error(tmp_path, launch):
    _, primary = launch.live()
    _, refined = launch.live("refined", "cuda", PISTON_REFINED_CASE)
    del primary["tight_spatial"]
    reference = tmp_path / "external.json"
    reference.write_text(json.dumps(primary), encoding="utf-8")
    with pytest.raises(ValueError, match="temporal reference.*tight_spatial"):
        runner.verify_temporal_reference(reference, sha(reference), refined)


@pytest.mark.parametrize("missing", ["--capture-manifest-sha256", "--verification-output"])
def test_offline_cli_requires_trust_anchor_and_separate_output(tmp_path, monkeypatch, missing, capsys):
    args = {
        "--executable": "solver.exe", "--fixture": "fixture.exe", "--backend": "cpu", "--case": PISTON_CASE,
        "--verify-capture": str(tmp_path / "legacy"), "--verification-output": str(tmp_path / "evaluation"),
        "--solver-sha256": "0" * 64, "--fixture-sha256": "1" * 64, "--capture-manifest-sha256": "2" * 64,
    }
    del args[missing]
    monkeypatch.setattr("sys.argv", ["run_liquid_acceptance.py", *(token for pair in args.items() for token in pair)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert missing in capsys.readouterr().err



def test_live_run_cannot_replace_the_author_receipt_before_manifest_publication(tmp_path, launch):
    launch.controls["mutate_author"] = True
    with pytest.raises(ValueError, match="run changed launch inputs"):
        launch.live()
    assert not (tmp_path / "live/capture/capture-manifest.json").exists()
