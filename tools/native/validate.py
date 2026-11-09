"""Validate a locked native preset and publish a path-free success receipt."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from compile_fixtures import parse_schema1_json_object
from qualify_toolchain import PINNED_TOOLS

PLATFORMS = {
    "Windows": ("windows", "Windows", "msvc", "MSVC"),
    "Linux": ("linux", "Linux", "gcc", "GNU"),
    "Darwin": ("macos", "Macos", "apple-clang", "AppleClang"),
}
ARCHITECTURES = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "armv8", "aarch64": "armv8"}
LOCKS = ("conan.lock", "reference/uv.lock")
PUBLIC_REPORTS = ("ctest-cli.xml", "ctest-cuda-cli.xml", "summary.json", "ctest-cpu.xml", "ctest-gpu.xml", "ctest-cuda-mock.xml", "ctest-schema.xml")
CLI_REQUIRED = {f"SimulateCli.{name}" for name in ("arguments", "schema", "identity", "inspection", "cpu_device", "cpu_execution", "cpu_replay", "execution_validation")}
CUDA_CLI_REQUIRED = {"SimulateCli.cuda_device", "SimulateCli.cuda_execution", "SimulateCli.cuda_replay"}
SCHEMA_REQUIRED = {
    f"LiquidCacheSchema.{name}"
    for name in (
        "common", "liquid_state", "point_frame", "frame_manifest", "resolved_scene", "checkpoint_reference",
        "RejectIncompatibleField", "RejectIncompatibleCheckpointField"
    )
}
FIXTURE_TESTS = {
    "affine_apic.json": "Transfer.LocalAffineFixture",
    "compliant_pressure.json": "Pressure.CompliantFixtureRetainsMeanPressure",
    "stiff_viscosity.json": "Viscosity.StiffFixture",
    "two_cell_pressure.json": "Pressure.TwoCellFixtureUsesPositiveInternalJump",
    "uniform_mac_liquid_boundary_3d.json": "FixtureBridge.CanonicalFixturesPresent",
    "uniform_mac_pressure_3d.json": "FixtureBridge.CanonicalFixturesPresent",
    "uniform_mac_transfer_3d.json": "MacTransfer3D.AffineCornerMatchesOracle",
}
CPU_REQUIRED = {
    *FIXTURE_TESTS.values(),
    "Build.Cxx20StatusRoundTrip",
    "FixtureBridge.CanonicalFixturesPresent",
    "FixtureBridge.EscapedStringsAndTypedLiteralsRoundTrip",
}
GPU_PROBE_RECEIPT_CASE = "CudaProbe.SharedVec2Math"
GPU_PROBE_CASES = {GPU_PROBE_RECEIPT_CASE, "CudaProbe.LaunchIgnoresPriorRuntimeError"}
GPU_OPERATOR_ERROR_CASE = "CudaPressure.LaunchIgnoresPriorRuntimeError"
GPU_DEPOSIT_3D_RECEIPT_CASE = "CudaMacTransfer3D.DepositCornerMatchesOracleAndCpu"
GPU_GATHER_3D_RECEIPT_CASE = "CudaMacTransfer3D.GatherCornerMatchesOracleAndCpu"
GPU_PRESSURE_3D_MIXED_CASE = "CudaMacPressure3D.MixedLinksMatchOracle"
GPU_PRESSURE_3D_MULTILEVEL_CASE = "CudaMacPressure3D.MultilevelProjectionMatchesCpu"
GPU_BOUNDARY_GEOMETRY_3D_CASE = "CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle"
GPU_WALL_CONTACT_3D_CASE = "CudaLiquidBoundary3D.SweptContactMatchesOracle"
GPU_BOUNDARY_TRANSFER_3D_CASE = "CudaLiquidBoundary3D.CompatibleTransferAndGhostHistoryMatchCpu"
GPU_BOUNDARY_COUPLED_3D_CASE = "CudaLiquidBoundary3D.CoupledBoundaryOperatorsPreserveFailure"
GPU_SUBSTEP_3D_CASE = "CudaLiquidSubstep3D.PersistentSubstepMatchesCpu"
GPU_SUBSTEP_GROWTH_3D_CASE = "CudaLiquidSubstep3D.ReplaysAfterCapacityGrowth"
GPU_SUBSTEP_CASES = {GPU_SUBSTEP_3D_CASE, GPU_SUBSTEP_GROWTH_3D_CASE}
GPU_SNAPSHOT_3D_CASE = "CudaLiquidSnapshot3D.DelayedCopyFencesSourceEpoch"
GPU_SNAPSHOT_FAILURE_3D_CASE = "CudaLiquidSnapshot3D.DeviceFailureUsesLastValidHostSnapshot"
GPU_SNAPSHOT_CASES = {GPU_SNAPSHOT_3D_CASE, GPU_SNAPSHOT_FAILURE_3D_CASE}
GPU_OPERATOR_MEMCHECK_CASES = {
    *GPU_SNAPSHOT_CASES,
    *GPU_SUBSTEP_CASES,
    GPU_BOUNDARY_TRANSFER_3D_CASE,
    GPU_BOUNDARY_COUPLED_3D_CASE,
    GPU_BOUNDARY_GEOMETRY_3D_CASE,
    GPU_WALL_CONTACT_3D_CASE,
    GPU_PRESSURE_3D_MIXED_CASE,
    GPU_PRESSURE_3D_MULTILEVEL_CASE,
    "CudaMacPressure3D.RejectsIncompatibleAndUnconverged",
    GPU_DEPOSIT_3D_RECEIPT_CASE,
    GPU_GATHER_3D_RECEIPT_CASE,
    "CudaMacTransfer3D.RejectionPreservesOutputs",
    "CudaTransfer.DepositAnalyticAndCpuParity",
    "CudaTransfer.DepositRoundsContributionsBeforeOrderIndependentSum",
    "CudaTransfer.DepositRejectsInvalidSupportRangeAndCapacity",
    "CudaTransfer.DepositEmptyBatchClearsOutputs",
    "CudaTransfer.GatherAffineMatchesFixtureAndCpu",
    "CudaTransfer.GatherNonlinearKeepsTrueGradientSeparate",
    "CudaTransfer.GatherFailurePreservesOutputs",
    "CudaPressure.NeumannJumpMatchesFixture",
    "CudaPressure.ComplianceRetainsMeanPressure",
    "CudaPressure.DirichletAndWallFluxMatchAnalyticAndCpu",
    "CudaPressure.CapacityAndIncompatibilityPreserveOutputs",
}
GPU_GRAPH_RECEIPT_CASE = "CudaGraphReplay.PressureMatchesDirectAndMeasuresReplay"
GPU_GRAPH_CASES = {
    "CudaGraphReplay.DepositMatchesDirectAndClearsInactiveInputs",
    "CudaGraphReplay.RejectedDepositPreservesAcceptedOutputs",
    "CudaGraphReplay.FailedCaptureCanBeRetried",
    GPU_GRAPH_RECEIPT_CASE,
}
GPU_REQUIRED = GPU_PROBE_CASES | GPU_OPERATOR_MEMCHECK_CASES | {GPU_OPERATOR_ERROR_CASE} | GPU_GRAPH_CASES
GPU_OPERATOR_RECEIPT_CASES = {
    GPU_SNAPSHOT_3D_CASE: "liquid_snapshot_3d",
    GPU_SNAPSHOT_FAILURE_3D_CASE: "liquid_snapshot_3d_failure",
    GPU_SUBSTEP_3D_CASE: "liquid_substep_3d",
    GPU_SUBSTEP_GROWTH_3D_CASE: "liquid_substep_3d_growth",
    GPU_BOUNDARY_TRANSFER_3D_CASE: "liquid_boundary_transfer_3d",
    GPU_BOUNDARY_COUPLED_3D_CASE: "liquid_boundary_coupled_3d",
    GPU_BOUNDARY_GEOMETRY_3D_CASE: "liquid_boundary_geometry_3d",
    GPU_WALL_CONTACT_3D_CASE: "liquid_wall_contact_3d",
    GPU_PRESSURE_3D_MIXED_CASE: "pressure_3d_mixed",
    GPU_PRESSURE_3D_MULTILEVEL_CASE: "pressure_3d_multilevel",
    GPU_DEPOSIT_3D_RECEIPT_CASE: "p2g_3d",
    GPU_GATHER_3D_RECEIPT_CASE: "g2p_3d",
    "CudaTransfer.DepositAnalyticAndCpuParity": "p2g",
    "CudaTransfer.GatherAffineMatchesFixtureAndCpu": "g2p",
    "CudaPressure.NeumannJumpMatchesFixture": "pressure_neumann",
    "CudaPressure.ComplianceRetainsMeanPressure": "pressure_compliance",
}
# The floor prevents partially discovered suites from passing. New tests may increase the count.
MINIMUM_TESTS = {"cli": len(CLI_REQUIRED), "cuda-cli": len(CUDA_CLI_REQUIRED), "cpu": 123, "gpu": len(GPU_REQUIRED), "cuda-mock": 11, "schema": len(SCHEMA_REQUIRED)}
GPU_PROBE_RECEIPT_RE = re.compile(
    r"^device=(?P<device>[\w ()+.-]+) capability=(?P<capability>\d+\.\d+) "
    r"compiled_architecture=(?P<compiled_architecture>\d+) driver_api=(?P<driver_api>\d+) "
    r"runtime_api=(?P<runtime_api>\d+) launch=0 copy=0 synchronize=0 elements=17 sentinel=\(0,0\)$",
    re.MULTILINE,
)
OPERATOR_RECEIPT_RE = re.compile(
    r"operator=(?P<operator>[a-z_0-9]+) compiled_architecture=(?P<compiled_architecture>[0-9]+) "
    r"kernel_ms=(?P<kernel_ms>[0-9]+\.[0-9]+)"
)
SUBSTEP_RECEIPT_RE = re.compile(
    r"operator=(?P<operator>[a-z_0-9]+) compiled_architecture=(?P<compiled_architecture>[0-9]+) "
    r"substep_device_timeline_ms=(?P<substep_device_timeline_ms>[0-9]+\.[0-9]+) "
    r"timing_scope=topology_through_publication"
)
SNAPSHOT_RECEIPT_RE = re.compile(
    r"operator=(?P<operator>[a-z_0-9]+) compiled_architecture=(?P<compiled_architecture>[0-9]+) "
    r"snapshot_copy_ms=(?P<snapshot_copy_ms>[0-9]+\.[0-9]+) "
    r"timing_scope=d2h_stream_events"
)
GRAPH_REPLAY_ITERATIONS = 64
GRAPH_RECEIPT_RE = re.compile(
    r"graph_replay compiled_architecture=(?P<compiled_architecture>[0-9]+) iterations=(?P<iterations>[0-9]+) "
    r"capture_us=(?P<capture_us>[0-9]+\.[0-9]+) direct_submit_us=(?P<direct_submit_us>[0-9]+\.[0-9]+) "
    r"replay_submit_us=(?P<replay_submit_us>[0-9]+\.[0-9]+) direct_stream_us=(?P<direct_stream_us>[0-9]+\.[0-9]+) "
    r"replay_stream_us=(?P<replay_stream_us>[0-9]+\.[0-9]+)"
)


def sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def json_bytes(document):
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def publish(path, contents):
    # Only a complete report may replace the prior file, including on Windows.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            temporary.write(contents)
            temporary.close()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


class ValidationRun:
    def __init__(self, root, preset, report_dir, host_profile, build_profile):
        self.root = root
        self.report_dir = report_dir
        self.preset = preset
        self.config = preset.split("-")[1].title()
        self.mode = preset.split("-")[2]
        run_log_parent_dir = root / "out/native-validation" / preset
        run_log_parent_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_dir = Path(tempfile.mkdtemp(prefix="run-", dir=run_log_parent_dir))
        self.command_log_path = self.run_log_dir / "commands.jsonl"
        self.commands = []
        self.redactions = {sys.executable: "python", host_profile: "<host-profile>", build_profile: "<build-profile>"}

    def execute(self, stage, *arguments, check=True, timeout=None, env=None):
        argv = [str(argument) for argument in arguments]
        redacted_argv = []
        for argument in argv:
            if argument.startswith(str(self.run_log_dir) + os.sep):
                redacted_argv.append("<run-log>/" + Path(argument).name)
            else:
                redacted_argv.append(self.redactions.get(argument, argument))
        print(f"[{stage}] {redacted_argv}", flush=True)
        # A file avoids waiting on a pipe inherited by a sanitizer child after timeout.
        stage_log_path = self.run_log_dir / f"{stage}.log"
        with stage_log_path.open("w", encoding="utf-8") as stream:
            try:
                result = subprocess.run(
                    argv,
                    check=False,
                    cwd=self.root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired as error:
                self.record(stage, argv, redacted_argv, "timeout", stage_log_path)
                raise RuntimeError(f"{stage} timed out after {timeout} seconds") from error
            except OSError:
                self.record(stage, argv, redacted_argv, "unavailable", stage_log_path)
                raise
            if result.stdout is not None:
                stream.write(result.stdout)
        self.record(stage, argv, redacted_argv, result.returncode, stage_log_path)
        result.stdout = stage_log_path.read_text(encoding="utf-8", errors="replace")
        if check:
            require_success(stage, result)
        return result

    def record(self, stage, argv, redacted_argv, exit_code, stage_log_path):
        with self.command_log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "stage": stage,
                        "argv": argv,
                        "cwd": str(self.root),
                        "exit_code": exit_code,
                        "log": stage_log_path.name,
                    }
                )
                + "\n"
            )
        self.commands.append({"stage": stage, "argv": redacted_argv, "exit_code": exit_code})


def require_success(stage, result):
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-30:])
        raise RuntimeError(f"{stage} failed (exit {result.returncode})\n{tail}")


def check_locks(run):
    evidence = {}
    for index, lock in enumerate(LOCKS):
        committed_blob_id = run.execute(f"lock-{index}-head", "git", "rev-parse", f"HEAD:{lock}").stdout.strip()
        working_blob_id = run.execute(
            f"lock-{index}-working", "git", "hash-object", f"--path={lock}", lock
        ).stdout.strip()
        if working_blob_id != committed_blob_id:
            raise ValueError(f"{lock} differs from the committed lock in HEAD")
        evidence[lock] = {"git_blob": committed_blob_id, "sha256": sha256((run.root / lock).read_bytes())}
    return evidence


def hash_sources(run):
    tracked = run.execute(
        "source-list",
        "git",
        "ls-files",
        "-z",
        "--",
        "native",
        "schemas/simulate",
        "cmake",
        "reference",
        "tools/native",
        "tools/tests",
        "CMakeLists.txt",
        "CMakePresets.json",
        "conanfile.py",
        "conan.lock",
        ".github/workflows/native-cpu.yml",
    ).stdout
    tracked_source_paths = set(filter(None, tracked.split("\0")))
    # Include this script when validating it before the first commit.
    tracked_source_paths.update(("tools/native/validate.py", "tools/native/tests/test_validate.py"))
    return sha256(
        json_bytes(
            {
                source: sha256((run.root / source).read_bytes())
                for source in sorted(tracked_source_paths)
                if (run.root / source).is_file()
            }
        )
    )


def check_tools(run):
    if sys.version_info[:2] != (3, 13):
        raise ValueError("run the validator through the locked Python 3.13 native environment")
    versions = {"python": platform.python_version()}
    expected = {**PINNED_TOOLS, "ctest": PINNED_TOOLS["cmake"], "uv": None}
    prefixes = {"conan": "Conan version ", "cmake": "cmake version ", "ctest": "ctest version ", "uv": "uv "}
    for name, pinned in expected.items():
        output = run.execute(f"version-{name}", name, "--version").stdout.strip()
        prefix = prefixes.get(name, "")
        suffix = r"(?:\.git\.[\w.-]+)?" if name == "ninja" else ""
        match = re.match(rf"{prefix}(\d+\.\d+\.\d+){suffix}(?:\s|$)", output)
        if match is None or (pinned is not None and match[1] != pinned):
            raise ValueError(f"expected {name} {pinned}, observed {output!r}")
        if name == "uv" and tuple(map(int, match[1].split("."))) < (0, 13, 7):
            raise ValueError("uv >= 0.13.7 is required")
        versions[name] = match[0].strip().removeprefix(prefix)
    return versions


def check_windows_qualification(run, path):
    original = parse_schema1_json_object(path.read_bytes())
    if original.get("mode") != run.mode:
        raise ValueError(f"qualification receipt must use --mode {run.mode}")
    current_path = run.run_log_dir / "qualification.json"
    run.execute(
        "qualification",
        sys.executable,
        "tools/native/qualify_toolchain.py",
        "--mode",
        run.mode,
        "--output",
        current_path,
        timeout=120,
    )
    current = parse_schema1_json_object(current_path.read_bytes())
    for field in ("mode", "host", "source_sha256", "cuda"):
        if original.get(field) != current.get(field):
            raise ValueError(f"stale qualification ({field}). Requalify the Windows toolchain")
    for name, tool in current["tools"].items():
        recorded = original["tools"].get(name, {})
        if any(recorded.get(key) != tool[key] for key in ("version", "executable")):
            raise ValueError(f"stale qualification ({name}). Requalify the Windows toolchain")
    if not original["commands"] or any(command["exit_code"] for command in original["commands"]):
        raise ValueError("qualification receipt has missing or failed probe commands")
    if run.mode == "cuda" and current["cuda"]["architecture"] != 120:
        raise ValueError("the Windows CUDA presets require compute capability 12.0")
    return current


def windows_versions(qualification):
    host = qualification["host"]
    versions = {}
    for field in ("visual_studio_version", "toolset_version", "windows_sdk_version"):
        value = host.get(field)
        if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)+", value):
            versions[field] = value
    if qualification["mode"] == "cuda":
        tools = qualification["tools"]
        nvcc = re.search(r"\bV(\d+\.\d+\.\d+)\b", tools["nvcc"]["version"])
        smi_output = tools["nvidia-smi"]["version"]
        version_field = r"[ \t]*:[ \t]*(\d+(?:\.\d+)+)[ \t]*$"
        flags = re.IGNORECASE | re.MULTILINE
        driver = re.search(r"^KMD version" + version_field, smi_output, flags)
        if driver is None:
            driver = re.search(r"^DRIVER version" + version_field, smi_output, flags)
        utility = re.search(r"^NVIDIA-SMI version" + version_field, smi_output, flags)
        if nvcc is None or driver is None or utility is None:
            raise ValueError("CUDA qualification did not report usable nvcc and driver versions")
        versions.update(
            nvcc=nvcc[1], nvidia_driver=driver[1], nvidia_smi=utility[1], cuda_runtime=qualification["cuda"]["runtime"]
        )
    return versions


def conan_profiles(run, host_profile, build_profile, system, compiler, architecture):
    if host_profile == build_profile:
        raise ValueError("supply separate host and build profiles")
    profile_arguments = ["-pr:h", host_profile, "-pr:b", build_profile]
    expected_settings = {}
    for context, flag, config in (("host", "-s:h", run.config), ("build", "-s:b", "Release")):
        explicit = {"build_type": config, "compiler.cppstd": "20"}
        if compiler == "msvc":
            explicit.update({"compiler.runtime": "dynamic", "compiler.runtime_type": config})
        elif compiler == "gcc":
            explicit["compiler.libcxx"] = "libstdc++11"
        else:
            explicit["compiler.libcxx"] = "libc++"
        for name, value in explicit.items():
            profile_arguments.extend((flag, f"{name}={value}"))
        expected_settings[context] = {"os": system, "arch": architecture, "compiler": compiler, **explicit}

    effective = json.loads(
        run.execute("profiles", "conan", "profile", "show", *profile_arguments, "--format=json").stdout
    )
    profiles = {}
    paths = {}
    for context, name in (("host", host_profile), ("build", build_profile)):
        profile = effective[context]
        settings = profile["settings"]
        expected = expected_settings[context]
        for key, value in expected.items():
            if settings.get(key) != value:
                raise ValueError(f"{context} profile must use {key}={value}")
        if profile.get("package_settings") or profile.get("tool_requires"):
            raise ValueError(f"{context} profile contains package settings or tool requirement overrides")
        version = settings.get("compiler.version", "")
        if not re.fullmatch(r"\d+(?:\.\d+)?", version):
            raise ValueError(f"{context} profile has no usable compiler.version")
        path = Path(run.execute(f"profile-{context}-path", "conan", "profile", "path", name).stdout.strip())
        paths[context] = (run.root / path).resolve()
        profiles[context] = {
            "file_sha256": sha256(paths[context].read_bytes()),
            "effective_sha256": sha256(json_bytes(profile)),
            "settings": {**expected, "compiler.version": version},
        }
    if paths["host"] == paths["build"]:
        raise ValueError("host and build profile paths resolve to the same file")
    return profile_arguments, profiles, paths


def check_configuration(run, build, fixtures, conan, receipt_path, compiler_id, profiles):
    cache = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([\w_]+):\w+=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    expected = {"CMAKE_BUILD_TYPE": run.config, "CMAKE_GENERATOR": "Ninja"}
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(f"configured preset must use {key}={value}")
    for key, enabled in (("BUILD_TESTING", True), ("FLUIDITY_ENABLE_CUDA", run.mode == "cuda")):
        allowed = ("ON", "TRUE", "1") if enabled else ("OFF", "FALSE", "0")
        if cache.get(key, "").upper() not in allowed:
            raise ValueError(f"unexpected {key} in the configured preset")
    expected_paths = {"CMAKE_TOOLCHAIN_FILE": conan / "conan_toolchain.cmake", "FLUIDITY_FIXTURE_DIRECTORY": fixtures}
    if receipt_path:
        expected_paths["FLUIDITY_TOOLCHAIN_RECEIPT"] = receipt_path
    for key, path in expected_paths.items():
        if Path(cache.get(key, "")).resolve() != path.resolve():
            raise ValueError(f"configured preset uses an unexpected {key}")

    cmake_version = PINNED_TOOLS["cmake"]
    compiler_file = build / "CMakeFiles" / cmake_version / "CMakeCXXCompiler.cmake"
    compiler = compiler_file.read_text(encoding="utf-8")
    if f'set(CMAKE_CXX_COMPILER_ID "{compiler_id}")' not in compiler:
        raise ValueError(f"configured C++ compiler must be {compiler_id}")
    version_match = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "(\d+(?:\.\d+)+)"\)', compiler)
    if version_match is None:
        raise ValueError("CMake did not report a C++ compiler version")
    version = version_match[1]
    major, minor, *_ = version.split(".")
    conan_version = major + minor[0] if compiler_id == "MSVC" else major
    for context, profile in profiles.items():
        if profile["settings"]["compiler.version"] != conan_version:
            raise ValueError(f"{context} profile does not match configured {compiler_id} {version}")
    return {"id": compiler_id, "version": version}


def hash_fixtures(directory, build):
    manifest = directory / "manifest.json"
    contents = manifest.read_bytes()
    entries = parse_schema1_json_object(contents)["fixtures"]
    actual = {path.name for path in directory.iterdir() if path.suffix.lower() == ".json"}
    if entries.keys() != FIXTURE_TESTS.keys() or actual != {*FIXTURE_TESTS, "manifest.json"}:
        raise ValueError("expected the canonical exported fixture JSON files and their manifest")
    for name, expected in entries.items():
        if sha256((directory / name).read_bytes()) != expected:
            raise ValueError(f"fixture hash mismatch: {name}")
    return {
        "manifest_sha256": sha256(contents),
        "fixtures": entries,
        "header_sha256": sha256((build / "generated/fixture_data.h").read_bytes()),
    }


def junit_cases(path):
    report = ET.parse(path).getroot()
    # The pinned CTest and pytest write different wrappers and outcome fields.
    pytest_report = path.name == "pytest.xml"
    if pytest_report and report.tag == "testsuites" and len(report) == 1:
        report = report[0]
    elif pytest_report or report.tag != "testsuite":
        raise ValueError(f"unexpected JUnit root in {path.name}")
    if report.tag != "testsuite" or not report.get("name", "").strip():
        raise ValueError(f"invalid JUnit suite in {path.name}")
    required_totals = ("tests", "failures", "skipped", "errors" if pytest_report else "disabled")
    totals = {}
    for key in required_totals:
        value = report.get(key, "")
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"invalid JUnit {key} total in {path.name}")
        totals[key] = int(value)
    cases = report.findall("testcase")
    if len(cases) != totals["tests"]:
        raise ValueError(f"JUnit count mismatch in {path.name}")
    junit_children(report, {"testcase", "properties"}, path)
    counts = dict.fromkeys(required_totals[1:], 0)
    outcomes = {"failure", "skipped", "error"} if pytest_report else {"failure", "skipped"}
    ctest_markers = {"run": set(), "fail": {"failure"}, "notrun": {"skipped"}, "disabled": set()}
    for case in cases:
        if not all(case.get(key, "").strip() for key in ("name", "classname")):
            raise ValueError(f"unnamed JUnit testcase in {path.name}")
        children = junit_children(case, outcomes | {"properties", "system-out", "system-err"}, path)
        markers = outcomes & children
        if len(markers) > 1:
            raise ValueError(f"conflicting JUnit testcase outcomes in {path.name}")
        status = case.get("status")
        if pytest_report:
            if status is not None:
                raise ValueError(f"unexpected pytest JUnit testcase status in {path.name}")
            marker = next(iter(markers), None)
            if marker is not None:
                counts[marker if marker == "skipped" else marker + "s"] += 1
        else:
            if status not in ctest_markers or markers != ctest_markers[status]:
                raise ValueError(f"invalid CTest JUnit testcase status in {path.name}")
            if status != "run":
                counts[{"fail": "failures", "notrun": "skipped", "disabled": "disabled"}[status]] += 1
    if any(totals[key] != count for key, count in counts.items()):
        raise ValueError(f"JUnit outcome totals do not match testcase results in {path.name}")
    return cases, any(counts.values())


def junit_children(element, allowed, path):
    tags = set()
    for child in element:
        tag = child.tag
        if tag not in allowed or (tag in tags and tag != "testcase"):
            raise ValueError(f"unexpected JUnit {tag} element in {path.name}")
        tags.add(tag)
        if tag == "properties":
            for entry in child:
                if entry.tag != "property" or len(entry) or not entry.get("name") or "value" not in entry.attrib:
                    raise ValueError(f"invalid JUnit property in {path.name}")
        elif tag != "testcase" and len(child):
            raise ValueError(f"unexpected children of JUnit {tag} element in {path.name}")
    return tags


def case_outcome(case):
    if case.find("failure") is not None or case.find("error") is not None:
        return "failure"
    if case.find("skipped") is not None or case.get("status", "run") != "run":
        return "skipped"
    return "passed"


def publish_ctest(run, label, cases):
    # CTest stdout, hostnames and failure messages can contain local paths.
    report = ET.Element("testsuite", name=label, tests=str(len(cases)))
    counts = {"failure": 0, "skipped": 0}
    for index, case in enumerate(cases):
        name = case.attrib["name"]
        if re.fullmatch(r"[\w.]+(?:/[\w.]+)*", name) is None:
            name = f"test-{index + 1}"
        testcase = ET.SubElement(report, "testcase", name=name, classname=label)
        outcome = case_outcome(case)
        if outcome != "passed":
            counts[outcome] += 1
            ET.SubElement(testcase, outcome)
    report.set("failures", str(counts["failure"]))
    report.set("skipped", str(counts["skipped"]))
    publish(run.report_dir / f"ctest-{label}.xml", ET.tostring(report, encoding="utf-8", xml_declaration=True))


def without_test_controls(prefix):
    return {key: value for key, value in os.environ.items() if not key.upper().startswith(prefix)}


def google_test_case_output(case, label):
    name = case.attrib["name"]
    lines = case.findtext("system-out", "").splitlines()
    execution_markers = [
        (index, line) for index, line in enumerate(lines) if line.startswith(("[ RUN      ] ", "[       OK ] "))
    ]
    completion = rf"\[       OK \] {re.escape(name)} \(\d+ ms\)"
    if (
        len(execution_markers) != 2
        or execution_markers[0][1] != f"[ RUN      ] {name}"
        or not re.fullmatch(completion, execution_markers[1][1])
        or lines.count("[  PASSED  ] 1 test.") != 1
    ):
        raise ValueError(f"CTest did not show that GoogleTest {name} executed and passed")
    (run_index, _), (ok_index, _) = execution_markers
    if label == "gpu":
        outside_case = "\n".join(lines[:run_index] + lines[ok_index + 1 :])
        if name == GPU_PROBE_RECEIPT_CASE and GPU_PROBE_RECEIPT_RE.search(outside_case):
            raise ValueError("CTest gpu: hardware probe receipt outside its testcase")
        for receipt_type, receipt_lines in (("operator", operator_receipt_lines), ("graph replay", graph_receipt_lines)):
            if receipt_lines(outside_case):
                raise ValueError(f"CTest gpu: {name} reported a {receipt_type} receipt outside its testcase")
    return "\n".join(lines[run_index + 1 : ok_index])


def run_ctest(run, label, required):
    junit = run.run_log_dir / f"ctest-{label}.xml"
    result = run.execute(
        f"ctest-{label}",
        "ctest",
        "--preset",
        run.preset,
        "-L",
        f"^{label}$",
        "--output-on-failure",
        "--no-tests=error",
        "--output-junit",
        junit,
        check=False,
        env=without_test_controls("GTEST_"),
    )
    if result.returncode and not junit.is_file():
        require_success(f"ctest-{label}", result)
    cases, incomplete = junit_cases(junit)
    case_outputs = {
        case.attrib["name"]: google_test_case_output(case, label)
        for case in cases
        if case.attrib["status"] == "run" and label not in ("schema", "cli", "cuda-cli")
    }
    if label in ("cli", "cuda-cli"):
        case_outputs = {case.attrib["name"]: case.findtext("system-out", "").strip() for case in cases}
    if cases:
        publish_ctest(run, label, cases)
    require_success(f"ctest-{label}", result)
    names = {case.attrib["name"] for case in cases}
    if len(cases) != len(names) or len(names) < MINIMUM_TESTS[label] or not required <= names:
        raise ValueError(f"ctest {label} is missing expected tests or contains duplicate results")
    if label == "gpu" and names != required:
        raise ValueError("ctest gpu must contain exactly the required hardware probe, operator and graph replay tests")
    if label == "schema" and names != required:
        raise ValueError("ctest schema must contain exactly the required versioned schema conformance tests")
    if label in ("cli", "cuda-cli"):
        if names != required:
            raise ValueError(f"ctest {label} must contain exactly the required process checks")
        for name, output in case_outputs.items():
            lines = output.splitlines()
            expected_lines = 1 if label == "cli" else 2
            if len(lines) != expected_lines or lines[-1] != f"PASS {name.removeprefix('SimulateCli.')}":
                raise ValueError(f"ctest {label} is missing successful process evidence")
    if incomplete:
        raise ValueError(f"ctest {label} contains skipped or failed tests")
    return {"status": "passed", "tests": len(cases)}, case_outputs



def verify_cuda_cli_receipt(output, qualification, config):
    try:
        record = json.loads(output.splitlines()[0])
    except (ValueError, IndexError) as error:
        raise ValueError("CUDA CLI device receipt is malformed") from error
    device = qualification["cuda"]["configurations"][config]
    architecture = qualification["cuda"]["architecture"]
    expected = {
        "protocol_version": 1, "event": "result", "command": "device-info", "backend": "cuda",
        "cpu_available": True, "cuda_compiled": True, "cuda_available": True, "device_index": 0,
        "device_name": device["device"], "compute_major": architecture // 10, "compute_minor": architecture % 10,
        "compiled_architecture": architecture * 10, "driver_version": device["driver_api_version"],
        "runtime_version": device["runtime_version"], "launch_result": 0, "copy_result": 0,
        "synchronize_result": 0, "sample_count": 17,
    }
    if record != expected or any(type(record.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("CUDA CLI did not report the qualified device and executed image")
    return {"device": device["device"], "compiled_architecture": architecture * 10,
            "driver_api": device["driver_api_version"], "runtime_api": device["runtime_version"]}


def verify_cuda_execution_receipt(output, qualification, case):
    try:
        record = json.loads(output.splitlines()[0])
    except (ValueError, IndexError) as error:
        raise ValueError("CUDA CLI execution receipt is malformed") from error
    replay = case == "cuda_replay"
    expected = {
        "protocol_version": 1, "event": "result", "command": "replay" if replay else "run",
        "backend": "cuda", "frame": 10 if replay else 2, "substep": 1 if replay else 0,
        "epoch": 21 if replay else 4, "accepted_step": 21 if replay else 4,
        "particle_count": 1, "next_particle_id": 2, "frames_per_second": 24,
        "compiled_architecture": qualification["cuda"]["architecture"] * 10, "read_only": replay,
    }
    if not isinstance(record, dict) or any(record.get(key) != value or type(record.get(key)) is not type(value)
                                          for key, value in expected.items()):
        raise ValueError("CUDA CLI execution did not report the qualified image and accepted state")
    for key in ("graph_replay_count", "device_reserved_bytes"):
        if type(record.get(key)) is not int or record[key] <= 0:
            raise ValueError("CUDA CLI execution has no actual graph/device reservation receipt")
    elapsed = record.get("snapshot_copy_milliseconds")
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("CUDA CLI execution has no finite positive D2H event interval")
    for key in ("state_digest", "resolved_fingerprint", "execution_id"):
        if not isinstance(record.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", record[key]) is None:
            raise ValueError("CUDA CLI execution is missing exact state/cache identity")
    return {"compiled_architecture": expected["compiled_architecture"],
            "graph_replay_count": record["graph_replay_count"],
            "snapshot_copy_milliseconds": elapsed, "state_digest": record["state_digest"]}


def verify_gpu_probe_receipt(probe_output, qualification, config):
    observed_probe_records = GPU_PROBE_RECEIPT_RE.findall(probe_output)
    qualified_cuda = qualification["cuda"]
    qualified_device = qualified_cuda["configurations"][config]
    qualified_sm_code = qualified_cuda["architecture"]
    compute_capability = f"{qualified_sm_code // 10}.{qualified_sm_code % 10}"
    expected_compiled_architecture = qualified_sm_code * 10
    expected_probe_record = (
        qualified_device["device"],
        compute_capability,
        str(expected_compiled_architecture),
        str(qualified_device["driver_api_version"]),
        str(qualified_device["runtime_version"]),
    )
    if observed_probe_records != [expected_probe_record]:
        raise ValueError("hardware probe did not report the qualified CUDA device and successful launch")
    return {
        "device": qualified_device["device"],
        "compute_capability": compute_capability,
        "compiled_architecture": expected_compiled_architecture,
        "driver_api": qualified_device["driver_api_version"],
        "runtime_api": qualified_device["runtime_version"],
    }


def operator_receipt_lines(test_output):
    return [line for line in test_output.splitlines() if line.lstrip().startswith("operator=")]


def verify_operator_receipts(case_outputs, expected_compiled_architecture, evidence_label):
    measurements = {}
    for case_name, case_output in case_outputs.items():
        expected_operator = GPU_OPERATOR_RECEIPT_CASES.get(case_name)
        observed_receipt_lines = operator_receipt_lines(case_output)
        if expected_operator is None:
            if observed_receipt_lines:
                raise ValueError(f"{evidence_label}: unexpected operator receipt in {case_name}")
            continue
        if len(observed_receipt_lines) != 1:
            raise ValueError(f"{evidence_label}: {case_name} must report exactly one operator receipt")
        substep = case_name in GPU_SUBSTEP_CASES
        snapshot = case_name in GPU_SNAPSHOT_CASES
        if snapshot:
            receipt_pattern = SNAPSHOT_RECEIPT_RE
            duration_field = "snapshot_copy_ms"
        elif substep:
            receipt_pattern = SUBSTEP_RECEIPT_RE
            duration_field = "substep_device_timeline_ms"
        else:
            receipt_pattern = OPERATOR_RECEIPT_RE
            duration_field = "kernel_ms"
        receipt_match = receipt_pattern.fullmatch(observed_receipt_lines[0])
        if receipt_match is None:
            raise ValueError(f"{evidence_label}: malformed operator receipt in {case_name}")
        observed_operator = receipt_match["operator"]
        compiled_architecture_text = receipt_match["compiled_architecture"]
        observed_duration_ms = float(receipt_match[duration_field])
        if observed_operator != expected_operator:
            raise ValueError(f"{evidence_label}: {case_name} must report operator {expected_operator}")
        if compiled_architecture_text != str(expected_compiled_architecture):
            raise ValueError(f"{evidence_label}: {case_name} reported a mismatched compiled architecture")
        if not math.isfinite(observed_duration_ms):
            raise ValueError(f"{evidence_label}: {case_name} reported nonfinite kernel time")
        measurements[observed_operator] = {
            "compiled_architecture": expected_compiled_architecture,
            duration_field: observed_duration_ms,
        }
        if snapshot:
            if observed_duration_ms <= 0:
                raise ValueError(f"{evidence_label}: {case_name} reported nonpositive snapshot timeline")
            measurements[observed_operator]["timing_scope"] = "d2h_stream_events"
        if substep:
            if observed_duration_ms <= 0:
                raise ValueError(f"{evidence_label}: {case_name} reported nonpositive substep timeline")
            measurements[observed_operator]["timing_scope"] = "topology_through_publication"
    return measurements


def graph_receipt_lines(test_output):
    return [line for line in test_output.splitlines() if re.match(r"graph_replay\b", line.lstrip())]


def verify_graph_receipt(case_outputs, expected_compiled_architecture, evidence_label):
    for case_name, case_output in case_outputs.items():
        if case_name != GPU_GRAPH_RECEIPT_CASE and graph_receipt_lines(case_output):
            raise ValueError(f"{evidence_label}: unexpected graph replay receipt in {case_name}")

    receipt_label = f"{evidence_label}: {GPU_GRAPH_RECEIPT_CASE}"
    observed_lines = graph_receipt_lines(case_outputs[GPU_GRAPH_RECEIPT_CASE])
    if len(observed_lines) != 1:
        raise ValueError(f"{receipt_label} must report exactly one graph replay receipt")
    receipt_match = GRAPH_RECEIPT_RE.fullmatch(observed_lines[0])
    if receipt_match is None:
        raise ValueError(f"{receipt_label} reported a malformed graph replay receipt")

    fields = receipt_match.groupdict()
    if fields.pop("compiled_architecture") != str(expected_compiled_architecture):
        raise ValueError(f"{receipt_label} reported a mismatched compiled architecture")
    if fields.pop("iterations") != str(GRAPH_REPLAY_ITERATIONS):
        raise ValueError(f"{receipt_label} must report {GRAPH_REPLAY_ITERATIONS} iterations")

    timings = {field: float(value) for field, value in fields.items()}
    if not all(math.isfinite(value) for value in timings.values()):
        raise ValueError(f"{receipt_label} reported nonfinite graph replay time")
    return {"compiled_architecture": expected_compiled_architecture, "iterations": GRAPH_REPLAY_ITERATIONS, **timings}


def run_memcheck(run, sanitizer, stage, executable, expected_case_names, *test_arguments):
    memcheck_output = run.execute(
        stage,
        sanitizer,
        "--tool",
        "memcheck",
        "--require-cuda-init",
        "yes",
        "--error-exitcode",
        "97",
        "--launch-timeout",
        "30",
        executable,
        *test_arguments,
        timeout=90,
        env={**without_test_controls("GTEST_"), "NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE": "named-pipes"},
    ).stdout
    output_lines = memcheck_output.splitlines()
    error_summaries = [line.rstrip() for line in output_lines if line.startswith("========= ERROR SUMMARY:")]
    if error_summaries != ["========= ERROR SUMMARY: 0 errors"]:
        raise ValueError(f"Compute Sanitizer {executable.name} did not report one zero-error summary")
    expected_test_count = len(expected_case_names)
    expected_test_total = "1 test" if expected_test_count == 1 else f"{expected_test_count} tests"
    passed_summaries = [line for line in output_lines if line.startswith("[  PASSED  ]")]
    if passed_summaries != [f"[  PASSED  ] {expected_test_total}."] or any(
        line.startswith(("[  SKIPPED ]", "[  FAILED  ]")) for line in output_lines
    ):
        raise ValueError(f"Compute Sanitizer {executable.name} did not pass all required hardware tests")

    # Match each RUN with its own OK and attribute the output only to that testcase.
    case_markers = list(re.finditer(r"^\[( RUN      |       OK )\] (.+)$", memcheck_output, re.MULTILINE))
    if len(case_markers) != 2 * expected_test_count:
        raise ValueError(f"Compute Sanitizer {executable.name} did not run and complete every required hardware test")
    case_outputs = {}
    for started_case, completed_case in zip(case_markers[::2], case_markers[1::2]):
        observed_case_name = started_case[2]
        expected_completion = rf"{re.escape(observed_case_name)} \([0-9]+ ms\)"
        if (
            started_case[1] != " RUN      "
            or completed_case[1] != "       OK "
            or re.fullmatch(expected_completion, completed_case[2]) is None
            or observed_case_name in case_outputs
        ):
            raise ValueError(f"Compute Sanitizer {executable.name} has mismatched or duplicate GoogleTest results")
        case_outputs[observed_case_name] = memcheck_output[started_case.end() : completed_case.start()]
    if case_outputs.keys() != expected_case_names:
        raise ValueError(f"Compute Sanitizer {executable.name} did not run the required hardware cases")
    # Count the entire process output, including any lines outside the testcase intervals.
    expected_probe_receipt_count = int(GPU_PROBE_RECEIPT_CASE in expected_case_names)
    observed_probe_receipt_count = sum(line.lstrip().startswith("device=") for line in output_lines)
    if observed_probe_receipt_count != expected_probe_receipt_count:
        raise ValueError(
            f"Compute Sanitizer {executable.name} expected {expected_probe_receipt_count} hardware probe receipts, "
            f"found {observed_probe_receipt_count}"
        )
    for receipt_type, receipt_lines in (("operator", operator_receipt_lines), ("graph replay", graph_receipt_lines)):
        attributed_receipt_count = sum(len(receipt_lines(case_output)) for case_output in case_outputs.values())
        if attributed_receipt_count != len(receipt_lines(memcheck_output)):
            raise ValueError(f"Compute Sanitizer {executable.name} reported a {receipt_type} receipt outside its testcase")
    return case_outputs


def run_sanitizer(run, qualification, build, expected_compiled_architecture):
    if run.config != "Release":
        return {"status": "not_run", "reason": "release_only"}
    toolkit = Path(qualification["tools"]["nvcc"]["executable"]).parent.parent
    sanitizer = toolkit / "compute-sanitizer/compute-sanitizer.exe"
    if not sanitizer.is_file():
        raise ValueError("Compute Sanitizer is required for Windows CUDA Release. Install it in the qualified toolkit")
    run.redactions[str(sanitizer)] = "compute-sanitizer"
    version_output = run.execute("sanitizer-version", sanitizer, "--version", timeout=15).stdout
    version_match = re.search(r"^Version (\d+\.\d+\.\d+\.\d+) \(build (\d+)\)", version_output, re.MULTILINE)
    if version_match is None:
        raise ValueError("Compute Sanitizer did not report a usable version")
    relative_build = build.relative_to(run.root)
    probe_case_outputs = run_memcheck(
        run,
        sanitizer,
        "sanitizer-memcheck",
        relative_build / "fluidity_cuda_probe_tests.exe",
        {GPU_PROBE_RECEIPT_CASE},
        f"--gtest_filter={GPU_PROBE_RECEIPT_CASE}",
    )
    verify_gpu_probe_receipt(probe_case_outputs[GPU_PROBE_RECEIPT_CASE], qualification, run.config)
    operator_case_outputs = run_memcheck(
        run,
        sanitizer,
        "sanitizer-operators-memcheck",
        relative_build / "fluidity_cuda_operator_tests.exe",
        GPU_OPERATOR_MEMCHECK_CASES,
        f"--gtest_filter=*-{GPU_OPERATOR_ERROR_CASE}",
    )
    graph_case_outputs = run_memcheck(
        run,
        sanitizer,
        "sanitizer-graph-memcheck",
        relative_build / "fluidity_cuda_graph_tests.exe",
        GPU_GRAPH_CASES,
    )
    case_outputs = {**probe_case_outputs, **operator_case_outputs, **graph_case_outputs}
    verify_operator_receipts(case_outputs, expected_compiled_architecture, "Compute Sanitizer")
    verify_graph_receipt(case_outputs, expected_compiled_architecture, "Compute Sanitizer")
    return {
        "status": "passed",
        "version": version_match[1],
        "build": version_match[2],
        "tool": "memcheck",
        "local_connection": "named-pipes",
        "errors": 0,
        "tests": len(case_outputs),
    }


def parse_pytest_collection(output):
    # Pytest's xunit2 classname is the dotted test file and any enclosing class names.
    cases = set()
    for line in output.splitlines():
        address, bracket, parameters = line.partition("[")
        module, *names = address.split("::")
        if not names or not module.endswith(".py"):
            continue
        module = module.removesuffix(".py").replace("/", ".").replace("\\", ".")
        cases.add((".".join([module, *names[:-1]]), names[-1] + bracket + parameters))
    count = re.search(r"^([1-9][0-9]*) tests? collected in .+$", output, re.MULTILINE)
    if not count or len(cases) != int(count[1]):
        raise ValueError("Python collection was empty, partial, deselected or contained duplicates")
    return cases


def run_python(run):
    junit = run.run_log_dir / "pytest.xml"
    env = without_test_controls("PYTEST_")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    command = (
        sys.executable,
        "-m",
        "pytest",
        "reference/tests",
        "tools/tests",
        "tools/native/tests",
        "-q",
        "--color=no",
        "-o",
        "addopts=",
    )
    collection_stdout = run.execute("python-collection", *command, "--collect-only", env=env).stdout
    collected_case_ids = parse_pytest_collection(collection_stdout)
    result = run.execute(
        "python-tests",
        *command,
        "-o",
        "junit_family=xunit2",
        "--junitxml",
        junit,
        "--basetemp",
        run.run_log_dir / "pytest-temp",
        env=env,
    )
    cases, incomplete = junit_cases(junit)
    reported_case_ids = {(case.attrib["classname"], case.attrib["name"]) for case in cases}
    passed = re.search(r"^([1-9][0-9]*) passed(?:, [0-9]+ warnings?)? in .+$", result.stdout, re.MULTILINE)
    if (
        incomplete
        or not passed
        or int(passed[1]) != len(cases)
        or len(reported_case_ids) != len(cases)
        or reported_case_ids != collected_case_ids
    ):
        raise ValueError("Python results differ from the complete collected suite")
    return {"status": "passed", "tests": len(cases)}


def validate(run, args, system, compiler, compiler_id, architecture):
    lockfile_evidence = check_locks(run)
    source_tree_sha256 = hash_sources(run)
    head = run.execute("source-revision", "git", "rev-parse", "HEAD").stdout.strip()
    versions = check_tools(run)
    receipt_path = None
    qualification = None
    if system == "Windows":
        receipt_path = run.root / "out/qualification" / ("windows.json" if run.mode == "cuda" else "windows-cpu.json")
        receipt_sha256 = sha256(receipt_path.read_bytes())
        qualification = check_windows_qualification(run, receipt_path)
        versions["windows"] = windows_versions(qualification)
    arguments, profiles, profile_paths = conan_profiles(
        run, args.host_profile, args.build_profile, system, compiler, architecture
    )
    conan = Path("out/conan") / run.preset
    build = run.root / "out/build" / run.preset
    fixtures = run.root / "out/fixtures" / run.preset
    run.execute(
        "conan-install",
        "conan",
        "install",
        ".",
        "--output-folder",
        conan,
        *arguments,
        "--lockfile=conan.lock",
        "--build=missing",
    )
    gtest_env = without_test_controls("GTEST_")
    run.execute("configure", "cmake", "--preset", run.preset, env=gtest_env)
    versions["compiler"] = check_configuration(
        run, build, fixtures, run.root / conan, receipt_path, compiler_id, profiles
    )
    run.execute(
        "fixtures", "cmake", "--build", "--preset", run.preset, "--target", "fluidity_fixture_headers", env=gtest_env
    )
    fixture_evidence = hash_fixtures(fixtures, build)
    run.execute("build", "cmake", "--build", "--preset", run.preset, env=gtest_env)
    cpu, _ = run_ctest(run, "cpu", CPU_REQUIRED)
    evidence = {gate: {"status": "not_run"} for gate in ("gpu", "cuda_mock", "cuda_cli", "sanitizer", "proof")}
    evidence["native_cpu"] = cpu
    evidence["schema"], _ = run_ctest(run, "schema", SCHEMA_REQUIRED)
    evidence["cli"], _ = run_ctest(run, "cli", CLI_REQUIRED)
    if run.mode == "cuda":
        evidence["cuda_cli"], cli_outputs = run_ctest(run, "cuda-cli", CUDA_CLI_REQUIRED)
        evidence["cuda_cli"].update(verify_cuda_cli_receipt(cli_outputs["SimulateCli.cuda_device"], qualification, run.config))
        evidence["cuda_cli"]["execution"] = {
            case: verify_cuda_execution_receipt(cli_outputs["SimulateCli." + case], qualification, case)
            for case in ("cuda_execution", "cuda_replay")
        }
        evidence["cuda_mock"], _ = run_ctest(run, "cuda-mock", set())
        evidence["gpu"], gpu_case_outputs = run_ctest(run, "gpu", GPU_REQUIRED)
        probe_evidence = verify_gpu_probe_receipt(gpu_case_outputs[GPU_PROBE_RECEIPT_CASE], qualification, run.config)
        evidence["gpu"].update(probe_evidence)
        expected_compiled_architecture = probe_evidence["compiled_architecture"]
        evidence["gpu"]["operators"] = verify_operator_receipts(
            gpu_case_outputs, expected_compiled_architecture, "CTest gpu"
        )
        evidence["gpu"]["graph_replay"] = verify_graph_receipt(
            gpu_case_outputs, expected_compiled_architecture, "CTest gpu"
        )
        evidence["sanitizer"] = run_sanitizer(run, qualification, build, expected_compiled_architecture)
    evidence["python"] = run_python(run)

    if (
        lockfile_evidence != check_locks(run)
        or source_tree_sha256 != hash_sources(run)
        or head != run.execute("source-revision-final", "git", "rev-parse", "HEAD").stdout.strip()
        or fixture_evidence != hash_fixtures(fixtures, build)
        or any(profiles[key]["file_sha256"] != sha256(path.read_bytes()) for key, path in profile_paths.items())
    ):
        raise ValueError("validation inputs changed during the run")
    summary = {
        "schema_version": 1,
        "preset": run.preset,
        "platform": system,
        "architecture": architecture,
        "configuration": run.config,
        "head": head,
        "source_tree_sha256": source_tree_sha256,
        "locks": lockfile_evidence,
        "profiles": profiles,
        "toolchain": versions,
        "fixtures": fixture_evidence,
        "commands": run.commands,
        "evidence": evidence,
    }
    if qualification:
        if receipt_sha256 != sha256(receipt_path.read_bytes()):
            raise ValueError("Windows qualification receipt changed during the run")
        summary["qualification"] = {
            "mode": qualification["mode"],
            "source_sha256": qualification["source_sha256"],
            "receipt_sha256": receipt_sha256,
        }
    return summary


def output_path(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("--output requires a nonempty directory")
    return Path(value)


def clear_reports(output):
    for name in PUBLIC_REPORTS:
        (output / name).unlink(missing_ok=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--host-profile", required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--output", required=True, type=output_path, action="append")
    try:
        args = parser.parse_args(argv)
        if len(args.output) != 1:
            parser.error("specify --output exactly once")
    except SystemExit as error:
        # Recover the destination independently of the other arguments. Help and ambiguous
        # or malformed destinations must leave existing reports alone.
        options = argv[: argv.index("--")] if "--" in argv else argv
        if error.code and "-h" not in options and "--help" not in options:
            output_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
            output_parser.add_argument("--output", type=output_path, action="append")
            try:
                partial, _ = output_parser.parse_known_args(argv)
                if partial.output and len(partial.output) == 1:
                    clear_reports(partial.output[0].resolve())
            except argparse.ArgumentError:
                pass
            except (OSError, RuntimeError, ValueError) as cleanup_error:
                print(f"Could not remove stale validation reports: {cleanup_error}", file=sys.stderr)
        raise
    args.output = args.output[0]
    return args


def main(argv=None):
    args = parse_arguments(list(sys.argv[1:] if argv is None else argv))
    run = None
    try:
        output = args.output.resolve()
        clear_reports(output)
        prefix, system, compiler, compiler_id = PLATFORMS[platform.system()]
        modes = ("cpu", "cuda") if system == "Windows" else ("cpu",)
        presets = {f"{prefix}-{config}-{mode}" for config in ("debug", "release") for mode in modes}
        if args.preset not in presets:
            raise ValueError(f"unsupported preset {args.preset!r}, choose from {sorted(presets)}")
        architecture = ARCHITECTURES[platform.machine().lower()]
        if system == "Windows" and architecture != "x86_64":
            raise ValueError("Windows presets require an x64 host")
        output.mkdir(parents=True, exist_ok=True)
        run = ValidationRun(Path.cwd(), args.preset, output, args.host_profile, args.build_profile)
        summary = validate(run, args, system, compiler, compiler_id, architecture)
        publish(output / "summary.json", json_bytes(summary))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, ET.ParseError) as error:
        print(f"Native validation failed: {error}", file=sys.stderr)
        if run:
            print(f"Local logs: {run.run_log_dir}", file=sys.stderr)
        return 1
    print(f"Native validation succeeded: {output / 'summary.json'}")
    print(f"Local logs: {run.run_log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
