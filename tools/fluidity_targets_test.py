"""Tests for cmake/fluidity_targets.cmake: each configures a throwaway CMake project that includes the helper file.

Configuring needs cmake and ninja (both in uv.lock) and a C++ compiler; nothing is built.
"""

import subprocess
from pathlib import Path

import pytest

TARGETS_FILE = Path(__file__).resolve().parent.parent / "cmake" / "fluidity_targets.cmake"


def configure(project: Path, body: str, subfolder_body: str = "") -> subprocess.CompletedProcess:
    """Writes a project whose CMakeLists.txt includes the helper file, then `body`, then runs the check.

    CMake wraps a long message over several lines, so a test compares the output with its whitespace collapsed."""
    project.mkdir(parents=True, exist_ok=True)
    (project / "source.cc").write_text("int Answer() { return 42; }\n", encoding="utf-8")
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.31)\n"
        "project(probe LANGUAGES CXX)\n"
        f'include("{TARGETS_FILE.as_posix()}")\n'
        f"{body}\n"
        "fluidity_check_compile_flags()\n",
        encoding="utf-8",
    )
    if subfolder_body:
        (project / "subfolder").mkdir(exist_ok=True)
        (project / "subfolder" / "CMakeLists.txt").write_text(subfolder_body + "\n", encoding="utf-8")
    return subprocess.run(
        ["cmake", "-S", str(project), "-B", str(project / "build"), "-G", "Ninja"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )




def test_target_created_through_the_helpers_passes(tmp_path):
    configured = configure(
        tmp_path / "project",
        "add_library(marked STATIC source.cc)\nfluidity_apply_compile_flags(marked EXCEPTIONS OFF)\n",
    )
    assert configured.returncode == 0, configured.stdout


def test_target_without_the_flags_fails_by_name(tmp_path):
    configured = configure(
        tmp_path / "project",
        "add_library(marked STATIC source.cc)\n"
        "fluidity_apply_compile_flags(marked EXCEPTIONS OFF)\n"
        "add_executable(plain_program source.cc)\n",
    )
    assert configured.returncode != 0
    assert "lack the numerical flags: plain_program." in " ".join(configured.stdout.split())


def test_target_without_the_flags_in_a_subfolder_fails_by_name(tmp_path):
    configured = configure(
        tmp_path / "project",
        "add_subdirectory(subfolder)\n",
        subfolder_body="add_library(plain_library STATIC ../source.cc)\n",
    )
    assert configured.returncode != 0
    assert "lack the numerical flags: plain_library." in " ".join(configured.stdout.split())


def test_interface_and_imported_targets_are_not_checked(tmp_path):
    configured = configure(
        tmp_path / "project",
        "add_library(header_only INTERFACE)\nadd_library(from_elsewhere STATIC IMPORTED)\n",
    )
    assert configured.returncode == 0, configured.stdout
