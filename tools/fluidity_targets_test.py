"""Tests for cmake/fluidity_targets.cmake: each configures a throwaway CMake project that includes the helper file.

Configuring needs cmake and ninja (both in uv.lock) and a C++ compiler; nothing is built.
"""

import os
import subprocess
from pathlib import Path

import pytest

TARGETS_FILE = Path(__file__).resolve().parent.parent / "cmake" / "fluidity_targets.cmake"


REPO_ROOT = TARGETS_FILE.parent.parent


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


def gpu_only_sources_of(project: Path, body: str) -> list[str]:
    """Configures a probe with FLUIDITY_ENABLE_CUDA set but no CUDA language (nothing is compiled), then `body` and
    fluidity_write_gpu_only_sources, and returns the file names it wrote to gpu_only_sources.txt."""
    configured = configure(
        project,
        "set(FLUIDITY_ENABLE_CUDA ON)\n"
        "set(BUILD_TESTING ON)\n"
        "add_library(GTest::gtest INTERFACE IMPORTED)\n"
        "add_library(fluidity STATIC source.cc)\n"
        "fluidity_apply_compile_flags(fluidity EXCEPTIONS OFF)\n"
        f"{body}\n"
        "fluidity_write_gpu_only_sources()\n",
    )
    assert configured.returncode == 0, configured.stdout
    listed = (project / "build" / "gpu_only_sources.txt").read_text(encoding="utf-8").splitlines()
    return [Path(line).name for line in listed]


def test_gpu_sources_record_their_cc_files_as_gpu_only(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "gpu_helper.cc").write_text("int Helper() { return 1; }\n", encoding="utf-8")
    assert gpu_only_sources_of(project, "fluidity_add_gpu_sources(gpu_helper.cc)") == ["gpu_helper.cc"]


def test_test_library_gpu_sources_record_their_cc_files_as_gpu_only(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "gpu_support.cc").write_text("int Support() { return 1; }\n", encoding="utf-8")
    body = "fluidity_add_test_library(support SOURCES source.cc GPU_SOURCES gpu_support.cc)"
    assert gpu_only_sources_of(project, body) == ["gpu_support.cc"]


def test_cuda_configuration_refuses_nvcc_flags_from_the_environment(tmp_path):
    """The refusal comes before enable_language(CUDA), so this needs no CUDA compiler."""
    for variable in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("NVCC_")}
        environment[variable] = "--fmad=true"
        configured = subprocess.run(
            [
                "cmake",
                "-S",
                str(REPO_ROOT),
                "-B",
                str(tmp_path / variable),
                "-G",
                "Ninja",
                "-DFLUIDITY_ENABLE_CUDA=ON",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
        assert configured.returncode != 0
        assert "Unset NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS before configuring CUDA" in " ".join(
            configured.stdout.split()
        )
