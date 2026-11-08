"""Exercise the validator with controlled process output, separate from toolchain evidence."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CPU_CASES = (
    "Build.Cxx20StatusRoundTrip",
    "FixtureBridge.FourFixturesPresent",
    "FixtureBridge.EscapedStringsAndTypedLiteralsRoundTrip",
    "Transfer.LocalAffineFixture",
    "Pressure.TwoCellFixtureUsesPositiveInternalJump",
    "Pressure.CompliantFixtureRetainsMeanPressure",
    "Viscosity.StiffFixture",
    *(f"Cpu.Case{index}" for index in range(116)),
)
PROBE_CASES = ("CudaProbe.SharedVec2Math", "CudaProbe.LaunchIgnoresPriorRuntimeError")
OPERATOR_ERROR_CASE = "CudaPressure.LaunchIgnoresPriorRuntimeError"
MEMCHECK_OPERATOR_CASES = (
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
)
PROBE_RECEIPT = (
    "device=NVIDIA GeForce RTX 5060 Ti capability=12.0 compiled_architecture=1200 driver_api=13030 "
    "runtime_api=13030 launch=0 copy=0 synchronize=0 elements=17 sentinel=(0,0)"
)
OPERATOR_RECEIPTS = {
    "CudaTransfer.DepositAnalyticAndCpuParity": "operator=p2g compiled_architecture=1200 kernel_ms=0.125",
    "CudaTransfer.GatherAffineMatchesFixtureAndCpu": "operator=g2p compiled_architecture=1200 kernel_ms=0.25",
    "CudaPressure.NeumannJumpMatchesFixture": "operator=pressure_neumann compiled_architecture=1200 kernel_ms=0.5",
    "CudaPressure.ComplianceRetainsMeanPressure": "operator=pressure_compliance compiled_architecture=1200 kernel_ms=1.0",
}
FIXTURES = ("affine_apic.json", "compliant_pressure.json", "stiff_viscosity.json", "two_cell_pressure.json")
PYTHON_NODE_IDS = (
    "tests/test_reference.py::test_analytic",
    "test_tools.py::test_export[0]",
    "test_native.py::TestNative::test_fixture[path::value]",
)
PYTHON_CASES = (
    ("tests.test_reference", "test_analytic"),
    ("test_tools", "test_export[0]"),
    ("test_native.TestNative", "test_fixture[path::value]"),
)
PUBLIC_REPORTS = ("summary.json", "ctest-cpu.xml", "ctest-gpu.xml", "ctest-cuda-mock.xml")


@pytest.fixture
def validator(monkeypatch):
    spec = importlib.util.spec_from_file_location("validate", ROOT / "tools/native/validate.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.syspath_prepend(str(ROOT / "tools/native"))
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module.platform, "machine", lambda: "AMD64")
    return module


@pytest.fixture
def invocation(tmp_path, monkeypatch, capsys, validator):
    root = tmp_path / "checkout with spaces"
    output = tmp_path / "evidence with spaces"
    root.mkdir()
    output.mkdir()
    for name in PUBLIC_REPORTS:
        (output / name).write_text("stale", encoding="utf-8")
    monkeypatch.chdir(root)

    def run(preset="windows-debug-cpu", *, argv=None):
        if argv is None:
            argv = [
                "--preset",
                preset,
                "--host-profile",
                "host profile",
                "--build-profile",
                "build profile",
                "--output",
                str(output),
            ]
        return validator.main(argv)

    def rejected(expected, preset="windows-debug-cpu"):
        assert run(preset) != 0
        assert expected in capsys.readouterr().err
        assert not (output / "summary.json").exists()

    return root, output, run, rejected


@pytest.fixture
def scenario(invocation, monkeypatch):
    root, output, run, rejected = invocation
    for name in (
        "conan.lock",
        "reference/uv.lock",
        "out/qualification/windows-cpu.json",
        "out/qualification/windows.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for context in ("host", "build"):
        (root / f"{context} profile").write_text(f"[settings]\n# {context}", encoding="utf-8")
    for preset in ("windows-debug-cpu", "windows-debug-cuda", "windows-release-cuda"):
        _, config, mode = preset.split("-")
        fixture_dir = root / "out/fixtures" / preset
        fixture_dir.mkdir(parents=True)
        for name in FIXTURES:
            (fixture_dir / name).write_text("{}", encoding="utf-8")
        manifest = {name: hashlib.sha256(b"{}").hexdigest() for name in FIXTURES}
        (fixture_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "fixtures": manifest}), encoding="utf-8"
        )
        build = root / "out/build" / preset
        (build / "generated").mkdir(parents=True)
        (build / "generated/fixture_data.h").write_text("// generated", encoding="utf-8")
        receipt = "windows.json" if mode == "cuda" else "windows-cpu.json"
        (build / "CMakeCache.txt").write_text(
            f"CMAKE_BUILD_TYPE:STRING={config.title()}\nCMAKE_GENERATOR:INTERNAL=Ninja\nBUILD_TESTING:BOOL=TRUE\n"
            f"FLUIDITY_ENABLE_CUDA:BOOL={'TRUE' if mode == 'cuda' else 'FALSE'}\n"
            f"FLUIDITY_FIXTURE_DIRECTORY:PATH={fixture_dir}\n"
            f"CMAKE_TOOLCHAIN_FILE:FILEPATH={root / 'out/conan' / preset / 'conan_toolchain.cmake'}\n"
            f"FLUIDITY_TOOLCHAIN_RECEIPT:FILEPATH={root / 'out/qualification' / receipt}\n",
            encoding="utf-8",
        )
        (build / "CMakeFiles/4.4.4").mkdir(parents=True)
        (build / "CMakeFiles/4.4.4/CMakeCXXCompiler.cmake").write_text(
            'set(CMAKE_CXX_COMPILER_ID "MSVC")\nset(CMAKE_CXX_COMPILER_VERSION "19.44.35229")\n', encoding="utf-8"
        )

    settings = {
        "os": "Windows",
        "arch": "x86_64",
        "compiler": "msvc",
        "compiler.version": "194",
        "compiler.cppstd": "20",
        "compiler.runtime": "dynamic",
    }
    versions = {
        "uv": "uv 0.13.7",
        "conan": "Conan version 2.34.0",
        "cmake": "cmake version 4.4.4",
        "ctest": "ctest version 4.4.4",
        "ninja": "1.13.2",
    }

    toolkit = root / "Private Toolchain"
    sanitizer_path = toolkit / "compute-sanitizer/compute-sanitizer.exe"
    sanitizer_path.parent.mkdir(parents=True)
    sanitizer_path.touch()

    def qualified_receipt(mode):
        receipt = {
            "schema_version": 1,
            "mode": mode,
            "source_sha256": {"cpu probe.cc": "0" * 64},
            "tools": {"cl": {"executable": "C:/Private Toolchain/cl.exe", "version": "19.44.35229"}},
            "host": {
                "architecture": "x64",
                "pointer_bits": 64,
                "configurations": {
                    config: {"compiler_version": "19.44.35229"} for config in ("Debug", "Release")
                },
            },
            "commands": [{"argv": ["C:/Private Toolchain/cl.exe"], "exit_code": 0}],
        }

        if mode == "cuda":
            receipt["tools"].update(
                nvcc={"executable": str(toolkit / "bin/nvcc.exe"), "version": "Cuda release 13.4, V13.4.59"},
                **{"nvidia-smi": {"executable": "nvidia-smi", "version": "NVIDIA-SMI version : 612.00\nKMD version : 610.88"}},
            )
            receipt["cuda"] = {
                "architecture": 120,
                "runtime": "Shared",
                "configurations": {
                    config: {
                        "device": "NVIDIA GeForce RTX 5060 Ti",
                        "device_architecture": 120,
                        "kernel_architecture": 120,
                        "result": 7,
                        "runtime_version": 13030,
                        "driver_api_version": 13030,
                    }
                    for config in ("Debug", "Release")
                },
            }
        return receipt

    for mode, name in (("cpu", "windows-cpu"), ("cuda", "windows")):
        (root / f"out/qualification/{name}.json").write_text(json.dumps(qualified_receipt(mode)), encoding="utf-8")

    class ExternalProcesses:
        failure = None
        stale_qualification = False
        stale_lock = False
        cpu_cases = CPU_CASES
        first_cpu_status = "run"
        gtest_result = None
        gpu_cases = (*PROBE_CASES, *MEMCHECK_OPERATOR_CASES, OPERATOR_ERROR_CASE)
        skipped_gpu_case = None
        sanitizer_cases = MEMCHECK_OPERATOR_CASES
        sanitizer_result = None
        operator_sanitizer_error_summary = "========= ERROR SUMMARY: 0 errors"
        python_cases = PYTHON_CASES
        python_deselected = False
        collection_deselected = False

        def __init__(self):
            self.commands = []
            self.environments = {}
            self.sanitizer_path = sanitizer_path
            self.gpu_receipts = {PROBE_CASES[0]: PROBE_RECEIPT, **OPERATOR_RECEIPTS}
            self.sanitizer_receipts = OPERATOR_RECEIPTS.copy()

        @staticmethod
        def gtest_case_output(case_name, case_receipt, result="       OK"):
            return f"[ RUN      ] {case_name}\n{case_receipt}\n[{result} ] {case_name} (0 ms)\n"

        def run(self, argv, *, cwd, **options):
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            assert cwd == root
            self.commands.append(argv)
            command = Path(argv[0]).stem
            arguments = argv[1:]
            self.environments[(command, tuple(arguments[:2]))] = options.get("env") or {}
            stdout = ""
            if command == "git":
                if arguments[0] == "hash-object" and self.stale_lock:
                    stdout = "b" * 40
                elif arguments[0] != "ls-files":
                    stdout = "a" * 40
            elif arguments == ["--version"] and command in versions:
                stdout = versions[command]
            elif arguments[:2] == ["profile", "path"]:
                stdout = str(root / arguments[2])
            elif arguments[:2] == ["profile", "show"]:
                host_config = next(value.removeprefix("build_type=") for value in arguments if value.startswith("build_type="))
                stdout = json.dumps(
                    {
                        context: {"settings": dict(settings, build_type=config, **{"compiler.runtime_type": config})}
                        for context, config in (("host", host_config), ("build", "Release"))
                    }
                )
            elif arguments and arguments[0].endswith("qualify_toolchain.py"):
                mode = arguments[arguments.index("--mode") + 1]
                if self.failure == "device" and mode == "cuda":
                    return subprocess.CompletedProcess(argv, 1, "CUDA error: no CUDA-capable device")
                receipt = qualified_receipt(mode)
                if self.stale_qualification:
                    receipt["source_sha256"]["cpu probe.cc"] = "1" * 64
                path = Path(arguments[arguments.index("--output") + 1])
                path.write_text(json.dumps(receipt), encoding="utf-8")
            elif arguments and arguments[0] == "install":
                assert command == "conan"
            elif command == "compute-sanitizer":
                if arguments == ["--version"]:
                    stdout = "Version 2026.3.0.0 (build 12345)"
                else:
                    assert arguments[:8] == [
                        "--tool", "memcheck", "--require-cuda-init", "yes",
                        "--error-exitcode", "97", "--launch-timeout", "30",
                    ]
                    assert options["timeout"] == 90
                    assert options["env"]["NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE"] == "named-pipes"
                    assert not any(key.startswith("GTEST_") for key in options["env"])
                    executable = Path(arguments[8]).name
                    if executable == "fluidity_cuda_probe_tests.exe":
                        assert arguments[9:] == ["--gtest_filter=CudaProbe.SharedVec2Math"]
                        stdout = self.gtest_case_output(PROBE_CASES[0], PROBE_RECEIPT)
                        stdout += "[  PASSED  ] 1 test.\n========= ERROR SUMMARY: 0 errors"
                    else:
                        assert executable == "fluidity_cuda_operator_tests.exe"
                        assert arguments[9:] == ["--gtest_filter=*-CudaPressure.LaunchIgnoresPriorRuntimeError"]
                        stdout = "".join(
                            self.gtest_case_output(
                                case_name,
                                self.sanitizer_receipts.get(case_name, ""),
                                self.sanitizer_result if index == 1 and self.sanitizer_result else "       OK",
                            )
                            for index, case_name in enumerate(self.sanitizer_cases)
                        )
                        stdout += "[  PASSED  ] 11 tests.\n"
                        stdout += self.operator_sanitizer_error_summary
            elif command == "cmake":
                if self.failure == "build" and "--build" in arguments and "--target" not in arguments:
                    return subprocess.CompletedProcess(argv, 2, "compile failed")
            elif command == "ctest":
                if self.failure == "ctest":
                    return subprocess.CompletedProcess(argv, 8, "test failed")
                label = arguments[arguments.index("-L") + 1]
                cases = self.cpu_cases
                skipped = int(self.first_cpu_status == "notrun")
                disabled = int(self.first_cpu_status == "disabled")
                if label == "^gpu$":
                    cases = self.gpu_cases
                    skipped, disabled = int(self.skipped_gpu_case is not None), 0
                elif label == "^cuda-mock$":
                    cases = tuple(f"CudaMock.Case{index}" for index in range(11))
                    skipped, disabled = 0, 0
                suite = ET.Element(
                    "testsuite",
                    name="(empty)",
                    tests=str(len(cases)),
                    failures="0",
                    skipped=str(skipped),
                    disabled=str(disabled),
                )
                for index, name in enumerate(cases):
                    status = self.first_cpu_status if index == 0 and label == "^cpu$" else "run"
                    if label == "^gpu$" and name == self.skipped_gpu_case:
                        status = "notrun"
                    testcase = ET.SubElement(suite, "testcase", name=name, classname=name, status=status, time="0.01")
                    reported = self.gtest_result if index == 0 and self.gtest_result is not None else name
                    console = (
                        self.gtest_case_output(reported, self.gpu_receipts.get(name, "")) + "[  PASSED  ] 1 test."
                        if reported
                        else "Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests."
                    )
                    ET.SubElement(testcase, "system-out").text = f"{root}\n{console}\n"
                    if status == "notrun":
                        ET.SubElement(testcase, "skipped")
                ET.ElementTree(suite).write(arguments[arguments.index("--output-junit") + 1])
            elif arguments[:2] == ["-m", "pytest"]:
                if "--collect-only" in arguments:
                    summary = "3 tests collected in 0.01s"
                    if self.collection_deselected:
                        summary = "3/4 tests collected (1 deselected) in 0.01s"
                    stdout = "\n".join((*PYTHON_NODE_IDS, "", summary, ""))
                else:
                    report = ET.Element("testsuites")
                    suite = ET.SubElement(
                        report,
                        "testsuite",
                        name="pytest",
                        tests=str(len(self.python_cases)),
                        failures="0",
                        errors="0",
                        skipped="0",
                    )
                    for classname, name in self.python_cases:
                        ET.SubElement(suite, "testcase", classname=classname, name=name, time="0.01")
                    ET.ElementTree(report).write(arguments[arguments.index("--junitxml") + 1])
                    deselection = ", 1 deselected" if self.python_deselected else ""
                    stdout = f"{len(self.python_cases)} passed{deselection} in 0.01s\n"
            else:
                pytest.fail(f"unexpected command: {argv}")
            return subprocess.CompletedProcess(argv, 0, stdout)

    processes = ExternalProcesses()
    monkeypatch.setattr(subprocess, "run", processes.run)

    return processes, root, output, run, rejected


def test_wrong_preset_invalidates_stale_success(invocation):
    invocation[3]("preset", "windows-debug-gpu")


@pytest.mark.parametrize("joined", [False, True])
def test_invalid_cli_clears_the_one_named_output(invocation, joined):
    _, output, run, _ = invocation
    args = [f"--output={output}"] if joined else ["--output", str(output)]
    with pytest.raises(SystemExit) as error:
        run(argv=args)
    assert error.value.code == 2
    assert not any((output / name).exists() for name in PUBLIC_REPORTS)


def test_help_retains_output(invocation):
    _, output, run, _ = invocation
    with pytest.raises(SystemExit) as error:
        run(argv=["--output", str(output), "--help"])
    assert error.value.code == 0
    assert all((output / name).exists() for name in PUBLIC_REPORTS)


def test_ambiguous_output_retains_both_destinations(invocation):
    root, output, run, _ = invocation
    alternate = root / "another output"
    alternate.mkdir()
    (alternate / "summary.json").write_text("stale", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        run(
            argv=[
                "--preset",
                "windows-debug-cpu",
                "--host-profile",
                "host",
                "--build-profile",
                "build",
                "--output",
                str(output),
                f"--output={alternate}",
            ]
        )
    assert error.value.code == 2
    assert all((output / name).exists() for name in PUBLIC_REPORTS)
    assert (alternate / "summary.json").exists()


def test_worktree_lock_must_match_committed_lock(scenario):
    processes, _, _, _, rejected = scenario
    processes.stale_lock = True
    rejected("conan.lock")


def test_stale_qualification_is_rejected(scenario):
    processes, _, _, _, rejected = scenario
    processes.stale_qualification = True
    rejected("qualification")


@pytest.mark.parametrize(("failure", "diagnostic"), [("build", "compile failed"), ("ctest", "test failed")])
def test_failed_stage_invalidates_stale_success(scenario, failure, diagnostic):
    processes, _, _, _, rejected = scenario
    processes.failure = failure
    rejected(diagnostic)


def test_missing_fixture_invalidates_stale_success(scenario):
    _, root, _, _, rejected = scenario
    (root / "out/fixtures/windows-debug-cpu/affine_apic.json").unlink()
    rejected("fixture")


def test_cuda_requires_a_device_even_with_a_qualification_receipt(scenario):
    processes, _, _, _, rejected = scenario
    processes.failure = "device"
    rejected("no CUDA-capable device", "windows-debug-cuda")


@pytest.mark.parametrize("missing", ["all", "one"])
def test_cpu_suite_must_run_in_full(scenario, missing):
    processes, _, output, _, rejected = scenario
    if missing == "all":
        processes.cpu_cases = ()
    else:
        processes.cpu_cases = CPU_CASES[:-1]
    rejected("cpu")
    if missing == "all":
        assert not (output / "ctest-cpu.xml").exists()


@pytest.mark.parametrize("status", ["notrun", "disabled"])
def test_ctest_preserves_incomplete_report_after_zero_exit(scenario, status):
    processes, root, output, _, rejected = scenario
    processes.first_cpu_status = status
    rejected("contains skipped or failed tests")
    report = output / "ctest-cpu.xml"
    assert report.is_file()
    content = report.read_text(encoding="utf-8")
    suite = ET.fromstring(content)
    assert suite.attrib["skipped"] == "1"
    skipped = suite.findall("testcase/skipped/..")
    assert [case.attrib["name"] for case in skipped] == ["Build.Cxx20StatusRoundTrip"]
    assert str(root) not in content


@pytest.mark.parametrize("reported", ["", "Different.Test"])
def test_ctest_must_prove_each_expected_case_executed(scenario, reported):
    processes, _, output, _, rejected = scenario
    processes.gtest_result = reported
    rejected("GoogleTest")
    assert not (output / "ctest-cpu.xml").exists()


@pytest.mark.parametrize("problem", ["partial", "duplicate", "unexpected", "deselected", "collection_deselected"])
def test_python_results_must_match_full_collection(scenario, problem):
    processes, _, _, _, rejected = scenario
    if problem == "partial":
        processes.python_cases = PYTHON_CASES[:1]
    elif problem == "duplicate":
        processes.python_cases = (*PYTHON_CASES[:2], PYTHON_CASES[0])
    elif problem == "unexpected":
        processes.python_cases = (*PYTHON_CASES[:2], ("another_test", "test_unexpected"))
    elif problem == "deselected":
        processes.python_deselected = True
    else:
        processes.collection_deselected = True
    rejected("Python")


@pytest.mark.parametrize(
    ("producer", "xml"),
    [
        ("ctest", '<unexpected><testsuite tests="1"><testcase /></testsuite></unexpected>'),
        ("ctest", '<testsuite name="ctest" tests="0" failures="0" skipped="0" disabled="-1"/>'),
        (
            "ctest",
            '<testsuite name="ctest" tests="1" failures="0" skipped="0" disabled="0"><testcase name="A" classname="A" /></testsuite>',
        ),
        ("pytest", '<testsuite name="pytest" tests="0" failures="0" errors="0" skipped="0" />'),
        ("pytest", '<testsuites><testsuite name="pytest" tests="0" failures="0" skipped="0" /></testsuites>'),
        (
            "pytest",
            '<testsuites><testsuite name="pytest" tests="1" failures="0" errors="0" skipped="0"><testcase classname="Tests" /></testsuite></testsuites>',
        ),
        (
            "pytest",
            '<testsuites><testsuite name="pytest" tests="1" failures="0" errors="0" skipped="0"><testcase classname="Tests" name="test_case"><unexpected /></testcase></testsuite></testsuites>',
        ),
    ],
)
def test_junit_rejects_unknown_shape_or_invalid_required_fields(validator, tmp_path, producer, xml):
    report = tmp_path / f"{producer}.xml"
    report.write_text(xml, encoding="utf-8")
    with pytest.raises(ValueError, match="JUnit"):
        validator.junit_cases(report)


@pytest.mark.parametrize("driver_field", ["KMD version", "Driver Version"])
def test_windows_versions_measure_driver_separately_from_nvidia_smi(validator, driver_field):
    versions = validator.windows_versions(
        {
            "host": {},
            "mode": "cuda",
            "tools": {
                "nvcc": {"version": "Cuda compilation tools, release 13.4, V13.4.59"},
                "nvidia-smi": {"version": f"NVIDIA-SMI version  : 612.00\n{driver_field} : 610.88\n"},
            },
            "cuda": {"runtime": "Shared"},
        }
    )
    assert versions["nvidia_driver"] == "610.88"
    assert versions["nvidia_smi"] == "612.00"


def test_cpu_receipt_distinguishes_evidence_and_preserves_spaced_arguments(scenario, monkeypatch):
    processes, root, output, run, _ = scenario
    monkeypatch.setenv("GTEST_TOTAL_SHARDS", "2")
    monkeypatch.setenv("GTEST_SHARD_INDEX", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k test_analytic")
    monkeypatch.setenv("PYTEST_PLUGINS", "unexpected_plugin")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert run() == 0
    summary = (output / "summary.json").read_text(encoding="utf-8")
    evidence = json.loads(summary)["evidence"]
    assert evidence["python"] == {"status": "passed", "tests": 3}
    assert evidence["native_cpu"] == {"status": "passed", "tests": 123}
    for gate in ("gpu", "cuda_mock", "sanitizer", "proof"):
        assert evidence[gate]["status"] == "not_run"
    install = next(argv for argv in processes.commands if argv[1] == "install")
    assert install[install.index("-pr:h") + 1] == "host profile"
    assert install[install.index("-pr:b") + 1] == "build profile"
    assert "compiler.runtime_type=Debug" in install
    assert "compiler.runtime_type=Release" in install
    junit = next(argv for argv in processes.commands if "--output-junit" in argv)
    assert Path(junit[junit.index("--output-junit") + 1]).parent.is_dir()
    assert " " in junit[junit.index("--output-junit") + 1]
    ctest_env = processes.environments[("ctest", ("--preset", "windows-debug-cpu"))]
    assert not any(key.startswith("GTEST_") for key in ctest_env)
    assert ctest_env["CUDA_VISIBLE_DEVICES"] == "-1"
    pytest_env = processes.environments[(Path(sys.executable).stem, ("-m", "pytest"))]
    assert "PYTEST_ADDOPTS" not in pytest_env and "PYTEST_PLUGINS" not in pytest_env
    assert pytest_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert str(root) not in summary and "Private Toolchain" not in summary
    assert str(root) not in (output / "ctest-cpu.xml").read_text(encoding="utf-8")


@pytest.mark.parametrize("config", ["debug", "release"])
def test_cuda_success_requires_all_hardware_cases_and_reports_four_measurements(scenario, config):
    _, _, output, run, _ = scenario
    assert run(f"windows-{config}-cuda") == 0
    gpu_names = {case.attrib["name"] for case in ET.parse(output / "ctest-gpu.xml").findall("testcase")}
    assert gpu_names == {*PROBE_CASES, *MEMCHECK_OPERATOR_CASES, OPERATOR_ERROR_CASE}
    evidence = json.loads((output / "summary.json").read_bytes())["evidence"]
    assert evidence["gpu"] == {
        "status": "passed",
        "tests": 14,
        "device": "NVIDIA GeForce RTX 5060 Ti",
        "compute_capability": "12.0",
        "compiled_architecture": 1200,
        "driver_api": 13030,
        "runtime_api": 13030,
        "operators": {
            "p2g": {"compiled_architecture": 1200, "kernel_ms": 0.125},
            "g2p": {"compiled_architecture": 1200, "kernel_ms": 0.25},
            "pressure_neumann": {"compiled_architecture": 1200, "kernel_ms": 0.5},
            "pressure_compliance": {"compiled_architecture": 1200, "kernel_ms": 1.0},
        },
    }
    if config == "release":
        assert evidence["sanitizer"] == {
            "status": "passed",
            "version": "2026.3.0.0",
            "build": "12345",
            "tool": "memcheck",
            "local_connection": "named-pipes",
            "errors": 0,
            "tests": 12,
        }
    else:
        assert evidence["sanitizer"] == {"status": "not_run", "reason": "release_only"}


@pytest.mark.parametrize(
    "omitted_cases",
    [MEMCHECK_OPERATOR_CASES, (MEMCHECK_OPERATOR_CASES[-1],), (OPERATOR_ERROR_CASE,)],
    ids=["all-memcheck-operators", "one-memcheck-operator", "error-injection"],
)
def test_gpu_requires_each_operator_case(scenario, omitted_cases):
    processes, _, _, _, rejected = scenario
    processes.gpu_cases = tuple(case_name for case_name in processes.gpu_cases if case_name not in omitted_cases)
    rejected("missing expected tests", "windows-debug-cuda")


def test_gpu_rejects_a_skipped_operator(scenario):
    processes, _, _, _, rejected = scenario
    processes.skipped_gpu_case = "CudaTransfer.GatherFailurePreservesOutputs"
    rejected("contains skipped or failed tests", "windows-debug-cuda")


@pytest.mark.parametrize(
    "receipt",
    [
        "",
        OPERATOR_RECEIPTS["CudaTransfer.DepositAnalyticAndCpuParity"] + "\n"
        + OPERATOR_RECEIPTS["CudaTransfer.DepositAnalyticAndCpuParity"],
        "operator=p2g compiled_architecture=890 kernel_ms=0.125",
        "operator=g2p compiled_architecture=1200 kernel_ms=0.125",
        "operator=p2g compiled_architecture=1200 kernel_ms=-0.125",
        "operator=p2g compiled_architecture=1200 kernel_ms=nan",
        "operator=p2g compiled_architecture=1200 kernel_ms=inf",
        "operator=p2g compiled_architecture=1200 kernel_ms=" + "9" * 310 + ".0",
    ],
    ids=["missing", "duplicate", "wrong-architecture", "wrong-operator", "negative", "nan", "inf", "overflow"],
)
def test_gpu_rejects_invalid_receipts_in_a_passed_operator_case(scenario, receipt):
    processes, _, _, _, rejected = scenario
    processes.gpu_receipts["CudaTransfer.DepositAnalyticAndCpuParity"] = receipt
    rejected("CudaTransfer.DepositAnalyticAndCpuParity", "windows-debug-cuda")


def test_gpu_rejects_an_operator_receipt_from_an_unmeasured_case(scenario):
    processes, _, _, _, rejected = scenario
    case_name = "CudaTransfer.DepositEmptyBatchClearsOutputs"
    processes.gpu_receipts[case_name] = OPERATOR_RECEIPTS["CudaTransfer.DepositAnalyticAndCpuParity"]
    rejected(case_name, "windows-debug-cuda")


@pytest.mark.parametrize("problem", ["missing", "wrong-name", "skipped", "missing-receipt", "wrong-receipt", "errors"])
def test_release_rejects_incomplete_operator_memcheck(scenario, problem):
    processes, _, _, _, rejected = scenario
    if problem == "missing":
        processes.sanitizer_cases = MEMCHECK_OPERATOR_CASES[:-1]
    elif problem == "wrong-name":
        processes.sanitizer_cases = (*MEMCHECK_OPERATOR_CASES[:-1], "CudaPressure.DifferentTest")
    elif problem == "skipped":
        processes.sanitizer_result = "  SKIPPED"
    elif problem == "missing-receipt":
        del processes.sanitizer_receipts["CudaPressure.ComplianceRetainsMeanPressure"]
    elif problem == "wrong-receipt":
        processes.sanitizer_receipts["CudaPressure.ComplianceRetainsMeanPressure"] = (
            OPERATOR_RECEIPTS["CudaPressure.NeumannJumpMatchesFixture"]
        )
    else:
        processes.operator_sanitizer_error_summary += "\n========= ERROR SUMMARY: 1 error"
    rejected("Compute Sanitizer", "windows-release-cuda")


def test_release_does_not_publish_success_when_required_memcheck_is_unavailable(scenario):
    processes, _, _, _, rejected = scenario
    processes.sanitizer_path.unlink()
    rejected("Compute Sanitizer is required", "windows-release-cuda")
