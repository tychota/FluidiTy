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
FIXTURES = ("affine_apic.json", "compliant_pressure.json", "stiff_viscosity.json", "two_cell_pressure.json")


@pytest.fixture
def invocation(tmp_path, monkeypatch, capsys):
    root = tmp_path / "checkout with spaces"
    output = tmp_path / "evidence with spaces"
    root.mkdir()
    output.mkdir()
    (output / "summary.json").write_text('{"stale":true}', encoding="utf-8")
    monkeypatch.chdir(root)

    def run(preset="windows-debug-cpu"):
        spec = importlib.util.spec_from_file_location("validate", ROOT / "tools/native/validate.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        monkeypatch.syspath_prepend(str(ROOT / "tools/native"))
        try:
            spec.loader.exec_module(module)
        except FileNotFoundError:
            pytest.fail("native validation CLI is missing")
        monkeypatch.setattr(module.platform, "system", lambda: "Windows")
        monkeypatch.setattr(module.platform, "machine", lambda: "AMD64")
        return module.main(
            [
                "--preset",
                preset,
                "--host-profile",
                "host profile",
                "--build-profile",
                "build profile",
                "--output",
                str(output),
            ]
        )

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
    for mode in ("cpu", "cuda"):
        fixture_dir = root / f"out/fixtures/windows-debug-{mode}"
        fixture_dir.mkdir(parents=True)
        for name in FIXTURES:
            (fixture_dir / name).write_text("{}", encoding="utf-8")
        manifest = {name: hashlib.sha256(b"{}").hexdigest() for name in FIXTURES}
        (fixture_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "fixtures": manifest}), encoding="utf-8"
        )
        build = root / f"out/build/windows-debug-{mode}"
        (build / "generated").mkdir(parents=True)
        (build / "generated/fixture_data.h").write_text("// generated", encoding="utf-8")
        receipt = "windows.json" if mode == "cuda" else "windows-cpu.json"
        (build / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Debug\nCMAKE_GENERATOR:INTERNAL=Ninja\nBUILD_TESTING:BOOL=TRUE\n"
            f"FLUIDITY_ENABLE_CUDA:BOOL={'TRUE' if mode == 'cuda' else 'FALSE'}\n"
            f"FLUIDITY_FIXTURE_DIRECTORY:PATH={fixture_dir}\n"
            f"CMAKE_TOOLCHAIN_FILE:FILEPATH={root / f'out/conan/windows-debug-{mode}/conan_toolchain.cmake'}\n"
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

    def qualified_receipt(mode):
        return {
            "schema_version": 1,
            "mode": mode,
            "source_sha256": {"cpu probe.cc": "0" * 64},
            "tools": {"cl": {"executable": "C:/Private Toolchain/cl.exe", "version": "19.44.35229"}},
            "host": {
                "architecture": "x64",
                "pointer_bits": 64,
                "configurations": {"Debug": {"compiler_version": "19.44.35229"}},
            },
            "commands": [{"argv": ["C:/Private Toolchain/cl.exe"], "exit_code": 0}],
        }

    for mode, name in (("cpu", "windows-cpu"), ("cuda", "windows")):
        (root / f"out/qualification/{name}.json").write_text(json.dumps(qualified_receipt(mode)), encoding="utf-8")

    class ExternalProcesses:
        failure = None
        stale_qualification = False
        stale_lock = False
        cpu_cases = CPU_CASES
        skipped_cpu = False

        def __init__(self):
            self.commands = []

        def run(self, argv, *, cwd, **options):
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            assert cwd == root
            self.commands.append(argv)
            command = Path(argv[0]).stem
            arguments = argv[1:]
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
                stdout = json.dumps(
                    {
                        context: {"settings": dict(settings, build_type=config, **{"compiler.runtime_type": config})}
                        for context, config in (("host", "Debug"), ("build", "Release"))
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
            elif command == "cmake":
                if self.failure == "build" and "--build" in arguments and "--target" not in arguments:
                    return subprocess.CompletedProcess(argv, 2, "compile failed")
            elif command == "ctest":
                if self.failure == "ctest":
                    return subprocess.CompletedProcess(argv, 8, "test failed")
                cases = self.cpu_cases
                suite = ET.Element(
                    "testsuite", tests=str(len(cases)), failures="0", skipped=str(int(self.skipped_cpu)), disabled="0"
                )
                for index, name in enumerate(cases):
                    testcase = ET.SubElement(suite, "testcase", name=name, status="run")
                    ET.SubElement(testcase, "system-out").text = str(root)
                    if self.skipped_cpu and index == 0:
                        ET.SubElement(testcase, "skipped")
                ET.ElementTree(suite).write(arguments[arguments.index("--output-junit") + 1])
            elif arguments[:2] == ["-m", "pytest"]:
                suite = ET.Element("testsuite", tests="1", failures="0", errors="0", skipped="0")
                ET.SubElement(suite, "testcase", name="passed")
                ET.ElementTree(suite).write(arguments[arguments.index("--junitxml") + 1])
            else:
                pytest.fail(f"unexpected command: {argv}")
            return subprocess.CompletedProcess(argv, 0, stdout)

    processes = ExternalProcesses()
    monkeypatch.setattr(subprocess, "run", processes.run)
    return processes, root, output, run, rejected


def test_wrong_preset_invalidates_stale_success(invocation):
    invocation[3]("preset", "windows-debug-gpu")


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


@pytest.mark.parametrize("missing", ["all", "one", "skipped"])
def test_cpu_suite_must_run_in_full(scenario, missing):
    processes, _, _, _, rejected = scenario
    if missing == "all":
        processes.cpu_cases = ()
    elif missing == "one":
        processes.cpu_cases = CPU_CASES[:-1]
    else:
        processes.skipped_cpu = True
    rejected("cpu")


def test_cpu_receipt_distinguishes_evidence_and_preserves_spaced_arguments(scenario):
    processes, root, output, run, _ = scenario
    assert run() == 0
    summary = (output / "summary.json").read_text(encoding="utf-8")
    evidence = json.loads(summary)["evidence"]
    assert evidence["python"] == {"status": "passed", "tests": 1}
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
    assert str(root) not in summary and "Private Toolchain" not in summary
    assert str(root) not in (output / "ctest-cpu.xml").read_text(encoding="utf-8")
