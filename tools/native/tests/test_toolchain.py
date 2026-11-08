"""Process doubles exercise receipt handling. They provide no qualification evidence."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.native import qualify_toolchain


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    bindir = tmp_path / "native tools"
    bindir.mkdir()
    versions = {
        "cl": "",
        "uv": "uv 0.13.7",
        "conan": "Conan version 2.34.0",
        "cmake": "cmake version 4.4.4",
        "ninja": "1.13.2.git.kitware.jobserver-pipe-1",
    }
    for name in versions:
        (bindir / f"{name}.exe").touch()

    class Toolchain:
        compile_exit = 0
        debug_crt = "MDd"
        def __init__(self):
            self.sources = {}

        def run(self, argv, *, cwd, stdout, stderr, text, errors, check):
            assert check is False
            executable = Path(argv[0])
            name = executable.stem
            assert executable.is_absolute()
            assert isinstance(argv, list)
            if name == "cl":
                assert argv[1:5] == ["/nologo", "/Bv", "/std:c++20", "/Zc:__cplusplus"]
                source = Path(argv[-1])
                assert source.parent == cwd
                assert " " in str(source)
                self.sources[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
                return subprocess.CompletedProcess(
                    argv, self.compile_exit,
                    "Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35227 for x64",
                )
            if name in ("Debug", "Release"):
                crt = self.debug_crt if name == "Debug" else "MD"
                return subprocess.CompletedProcess(
                    argv, 0,
                    f"architecture=x64 pointer_bits=64 crt={crt} "
                    "msvc_ver=1944 msvc_full_ver=194435227 cxx_standard=202002\n",
                )
            assert argv[1:] == ["--version"]
            return subprocess.CompletedProcess(argv, 0, versions[name])

    double = Toolchain()
    monkeypatch.setattr(qualify_toolchain, "resolve_tool", lambda name: (
        path if (path := bindir / f"{name}.exe").is_file() else None
    ))
    monkeypatch.setattr(qualify_toolchain.subprocess, "run", double.run)
    return double, bindir, tmp_path / "qualification evidence" / "receipt.json"


def run_qualification(receipt):
    return qualify_toolchain.main(["--mode", "cpu", "--output", str(receipt)])


def test_missing_tool_fails(toolchain, capsys):
    _, bindir, receipt = toolchain
    (bindir / "cl.exe").unlink()

    assert run_qualification(receipt) != 0
    assert "cl" in capsys.readouterr().err
    assert not receipt.exists()


def test_wrong_crt_fails(toolchain, capsys):
    double, _, receipt = toolchain
    double.debug_crt = "MD"

    assert run_qualification(receipt) != 0
    assert "CRT" in capsys.readouterr().err
    assert not receipt.exists()


def test_paths_with_spaces(toolchain):
    double, bindir, receipt = toolchain

    assert run_qualification(receipt) == 0
    evidence = json.loads(receipt.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 1
    assert evidence["mode"] == "cpu"
    assert Path(evidence["tools"]["cl"]["executable"]) == bindir / "cl.exe"
    assert evidence["host"]["architecture"] == "x64"
    assert evidence["host"]["pointer_bits"] == 64
    assert evidence["host"]["configurations"]["Debug"]["crt"] == "MDd"
    assert evidence["host"]["configurations"]["Release"]["crt"] == "MD"
    assert evidence["source_sha256"] == {"cpu probe.cc": double.sources["cpu probe.cc"]}
    assert all(command["exit_code"] == 0 for command in evidence["commands"])


def test_failed_compile_removes_stale_success(toolchain, capsys):
    double, _, receipt = toolchain
    receipt.parent.mkdir()
    receipt.write_text('{"schema_version": 1}', encoding="utf-8")
    double.compile_exit = 2

    assert run_qualification(receipt) != 0
    assert "2" in capsys.readouterr().err
    assert not receipt.exists()
