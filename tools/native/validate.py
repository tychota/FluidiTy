"""Validate a locked native preset and publish a path-free success receipt."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from compile_fixtures import schema_document
from qualify_toolchain import PINNED_TOOLS

PLATFORMS = {
    "Windows": ("windows", "Windows", "msvc", "MSVC"),
    "Linux": ("linux", "Linux", "gcc", "GNU"),
    "Darwin": ("macos", "Macos", "apple-clang", "AppleClang"),
}
ARCHITECTURES = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "armv8", "aarch64": "armv8"}
LOCKS = ("conan.lock", "reference/uv.lock")
FIXTURE_TESTS = {
    "affine_apic.json": "Transfer.LocalAffineFixture",
    "compliant_pressure.json": "Pressure.CompliantFixtureRetainsMeanPressure",
    "stiff_viscosity.json": "Viscosity.StiffFixture",
    "two_cell_pressure.json": "Pressure.TwoCellFixtureUsesPositiveInternalJump",
}
CPU_REQUIRED = {
    *FIXTURE_TESTS.values(),
    "Build.Cxx20StatusRoundTrip",
    "FixtureBridge.FourFixturesPresent",
    "FixtureBridge.EscapedStringsAndTypedLiteralsRoundTrip",
}
GPU_REQUIRED = {"CudaProbe.SharedVec2Math", "CudaProbe.LaunchIgnoresPriorRuntimeError"}
# The floor prevents partially discovered suites from passing. New tests may increase the count.
MINIMUM_TESTS = {"cpu": 123, "gpu": 2, "cuda-mock": 11}
GPU_RECORD = re.compile(
    r"^device=(?P<device>[\w ()+.-]+) capability=(?P<capability>\d+\.\d+) "
    r"compiled_architecture=(?P<compiled_architecture>\d+) driver_api=(?P<driver_api>\d+) "
    r"runtime_api=(?P<runtime_api>\d+) launch=0 copy=0 synchronize=0 elements=17 sentinel=\(0,0\)$",
    re.MULTILINE,
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
    def __init__(self, root, preset, output, host_profile, build_profile):
        self.root = root
        self.output = output
        self.preset = preset
        self.config = preset.split("-")[1].title()
        self.mode = preset.split("-")[2]
        raw_parent = root / "out/native-validation" / preset
        raw_parent.mkdir(parents=True, exist_ok=True)
        self.raw = Path(tempfile.mkdtemp(prefix="run-", dir=raw_parent))
        self.log = self.raw / "commands.jsonl"
        self.commands = []
        self.redactions = {sys.executable: "python", host_profile: "<host-profile>", build_profile: "<build-profile>"}

    def execute(self, stage, *arguments, check=True, timeout=None, env=None):
        argv = [str(argument) for argument in arguments]
        display = []
        for argument in argv:
            if argument.startswith(str(self.raw) + os.sep):
                display.append("<run-log>/" + Path(argument).name)
            else:
                display.append(self.redactions.get(argument, argument))
        print(f"[{stage}] {display}", flush=True)
        # A file avoids waiting on a pipe inherited by a sanitizer child after timeout.
        stdout = self.raw / f"{stage}.log"
        with stdout.open("w", encoding="utf-8") as stream:
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
                self.record(stage, argv, display, "timeout", stdout)
                raise RuntimeError(f"{stage} timed out after {timeout} seconds") from error
            except OSError:
                self.record(stage, argv, display, "unavailable", stdout)
                raise
            if result.stdout is not None:
                stream.write(result.stdout)
        self.record(stage, argv, display, result.returncode, stdout)
        result.stdout = stdout.read_text(encoding="utf-8", errors="replace")
        if check:
            require_success(stage, result)
        return result

    def record(self, stage, argv, display, exit_code, stdout):
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"stage": stage, "argv": argv, "cwd": str(self.root), "exit_code": exit_code, "log": stdout.name}
                )
                + "\n"
            )
        self.commands.append({"stage": stage, "argv": display, "exit_code": exit_code})


def require_success(stage, result):
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-30:])
        raise RuntimeError(f"{stage} failed (exit {result.returncode})\n{tail}")


def check_locks(run):
    evidence = {}
    for index, lock in enumerate(LOCKS):
        recorded = run.execute(f"lock-{index}-head", "git", "rev-parse", f"HEAD:{lock}").stdout.strip()
        actual = run.execute(f"lock-{index}-working", "git", "hash-object", f"--path={lock}", lock).stdout.strip()
        if actual != recorded:
            raise ValueError(f"{lock} differs from the committed lock in HEAD")
        evidence[lock] = {"git_blob": recorded, "sha256": sha256((run.root / lock).read_bytes())}
    return evidence


def hash_sources(run):
    tracked = run.execute(
        "source-list",
        "git",
        "ls-files",
        "-z",
        "--",
        "native",
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
    sources = set(filter(None, tracked.split("\0")))
    # Include this script when validating it before the first commit.
    sources.update(("tools/native/validate.py", "tools/native/tests/test_validate.py"))
    return sha256(
        json_bytes(
            {
                source: sha256((run.root / source).read_bytes())
                for source in sorted(sources)
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
    original = schema_document(path.read_bytes())
    if original.get("mode") != run.mode:
        raise ValueError(f"qualification receipt must use --mode {run.mode}")
    current_path = run.raw / "qualification.json"
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
    current = schema_document(current_path.read_bytes())
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
        driver = re.search(r"^NVIDIA-SMI version\s*:\s*(\d+(?:\.\d+)+)", tools["nvidia-smi"]["version"], re.MULTILINE)
        if nvcc is None or driver is None:
            raise ValueError("CUDA qualification did not report usable nvcc and driver versions")
        versions.update(nvcc=nvcc[1], nvidia_driver=driver[1], cuda_runtime=qualification["cuda"]["runtime"])
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
    entries = schema_document(contents)["fixtures"]
    actual = {path.name for path in directory.iterdir() if path.suffix.lower() == ".json"}
    if entries.keys() != FIXTURE_TESTS.keys() or actual != {*FIXTURE_TESTS, "manifest.json"}:
        raise ValueError("expected the four exported fixture JSON files and their manifest")
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
    suites = [report] if report.tag == "testsuite" else list(report.findall("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    declared = sum(int(suite.attrib["tests"]) for suite in suites)
    incomplete = any(
        int(suite.get(key, "0")) for suite in suites for key in ("failures", "errors", "disabled", "skipped")
    )
    if len(cases) != declared:
        raise ValueError(f"JUnit count mismatch in {path.name}")
    return cases, incomplete


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
    publish(run.output / f"ctest-{label}.xml", ET.tostring(report, encoding="utf-8", xml_declaration=True))


def run_ctest(run, label, required):
    junit = run.raw / f"ctest-{label}.xml"
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
    )
    if result.returncode and not junit.is_file():
        require_success(f"ctest-{label}", result)
    cases, incomplete = junit_cases(junit)
    publish_ctest(run, label, cases)
    require_success(f"ctest-{label}", result)
    names = {case.attrib["name"] for case in cases}
    if len(cases) != len(names) or len(names) < MINIMUM_TESTS[label] or not required <= names:
        raise ValueError(f"ctest {label} is missing expected tests or contains duplicate results")
    if label == "gpu" and names != required:
        raise ValueError("ctest gpu must contain exactly the two hardware probe tests")
    if incomplete or any(case_outcome(case) != "passed" for case in cases):
        raise ValueError(f"ctest {label} contains skipped or failed tests")
    return {"status": "passed", "tests": len(cases)}, cases


def gpu_receipt(output, qualified, config):
    records = GPU_RECORD.findall(output)
    device = qualified["cuda"]["configurations"][config]
    architecture = qualified["cuda"]["architecture"]
    capability = f"{architecture // 10}.{architecture % 10}"
    expected = (
        device["device"],
        capability,
        str(architecture * 10),
        str(device["driver_api_version"]),
        str(device["runtime_version"]),
    )
    if records != [expected]:
        raise ValueError("hardware probe did not report the qualified CUDA device and successful launch")
    return {
        "device": records[0][0],
        "compute_capability": capability,
        "compiled_architecture": architecture * 10,
        "driver_api": device["driver_api_version"],
        "runtime_api": device["runtime_version"],
    }


def run_sanitizer(run, qualified, build):
    if run.config != "Release":
        return {"status": "not_run", "reason": "release_only"}
    toolkit = Path(qualified["tools"]["nvcc"]["executable"]).parent.parent
    sanitizer = toolkit / "compute-sanitizer/compute-sanitizer.exe"
    if not sanitizer.is_file():
        return {"status": "not_run", "reason": "unavailable"}
    run.redactions[str(sanitizer)] = "compute-sanitizer"
    version = run.execute("sanitizer-version", sanitizer, "--version", timeout=15).stdout
    match = re.search(r"^Version (\d+\.\d+\.\d+\.\d+) \(build (\d+)\)", version, re.MULTILINE)
    if match is None:
        raise ValueError("Compute Sanitizer did not report a usable version")
    output = run.execute(
        "sanitizer-memcheck",
        sanitizer,
        "--tool",
        "memcheck",
        "--require-cuda-init",
        "yes",
        "--error-exitcode",
        "97",
        "--launch-timeout",
        "30",
        build.relative_to(run.root) / "fluidity_cuda_probe_tests.exe",
        "--gtest_filter=CudaProbe.SharedVec2Math",
        timeout=90,
        env={**os.environ, "NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE": "named-pipes"},
    ).stdout
    if not re.search(r"^========= ERROR SUMMARY: 0 errors\s*$", output, re.MULTILINE):
        raise ValueError("Compute Sanitizer did not report zero errors")
    if not re.search(r"^\[  PASSED  \] 1 test\.\s*$", output, re.MULTILINE):
        raise ValueError("Compute Sanitizer did not run the required hardware test")
    gpu_receipt(output, qualified, run.config)
    return {
        "status": "passed",
        "version": match[1],
        "build": match[2],
        "tool": "memcheck",
        "local_connection": "named-pipes",
        "errors": 0,
        "tests": 1,
    }


def run_python(run):
    junit = run.raw / "pytest.xml"
    run.execute(
        "python-tests",
        sys.executable,
        "-m",
        "pytest",
        "reference/tests",
        "tools/tests",
        "tools/native/tests",
        "-q",
        "--junitxml",
        junit,
        "--basetemp",
        run.raw / "pytest-temp",
    )
    cases, incomplete = junit_cases(junit)
    if not cases or incomplete or any(case_outcome(case) != "passed" for case in cases):
        raise ValueError("Python tests are missing, skipped or failed")
    return {"status": "passed", "tests": len(cases)}


def validate(run, args, system, compiler, compiler_id, architecture):
    locks = check_locks(run)
    sources = hash_sources(run)
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
    run.execute("configure", "cmake", "--preset", run.preset)
    versions["compiler"] = check_configuration(
        run, build, fixtures, run.root / conan, receipt_path, compiler_id, profiles
    )
    run.execute("fixtures", "cmake", "--build", "--preset", run.preset, "--target", "fluidity_fixture_headers")
    fixture_evidence = hash_fixtures(fixtures, build)
    run.execute("build", "cmake", "--build", "--preset", run.preset)
    cpu, _ = run_ctest(run, "cpu", CPU_REQUIRED)
    evidence = {gate: {"status": "not_run"} for gate in ("gpu", "cuda_mock", "sanitizer", "proof")}
    evidence["native_cpu"] = cpu
    if run.mode == "cuda":
        evidence["cuda_mock"], _ = run_ctest(run, "cuda-mock", set())
        evidence["gpu"], cases = run_ctest(run, "gpu", GPU_REQUIRED)
        probe = next(case for case in cases if case.get("name") == "CudaProbe.SharedVec2Math")
        evidence["gpu"].update(gpu_receipt(probe.findtext("system-out", ""), qualification, run.config))
        evidence["sanitizer"] = run_sanitizer(run, qualification, build)
    evidence["python"] = run_python(run)

    if (
        locks != check_locks(run)
        or sources != hash_sources(run)
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
        "source_tree_sha256": sources,
        "locks": locks,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--host-profile", required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    run = None
    try:
        for name in ("summary.json", "ctest-cpu.xml", "ctest-gpu.xml", "ctest-cuda-mock.xml"):
            (output / name).unlink(missing_ok=True)
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
            print(f"Local logs: {run.raw}", file=sys.stderr)
        return 1
    print(f"Native validation succeeded: {output / 'summary.json'}")
    print(f"Local logs: {run.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
