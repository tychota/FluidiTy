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


def test_cuda_configuration_fails_on_nvcc_flags_from_the_environment(tmp_path):
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


def test_a_test_gets_its_backend_label_its_module_and_the_unit_tier(tmp_path):
    """The label list is written into the discovery script that gtest_discover_tests generates, where ctest reads it."""
    project = tmp_path / "project"
    (project / "grid").mkdir(parents=True)
    (project / "grid" / "grid.test.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
    (project / "grid" / "CMakeLists.txt").write_text(
        "fluidity_add_test(grid_test SOURCES grid.test.cc LABEL cpu)\n", encoding="utf-8"
    )
    configured = configure(
        project,
        "include(GoogleTest)\n"
        "set(BUILD_TESTING ON)\n"
        "add_library(GTest::gtest_main INTERFACE IMPORTED)\n"
        "add_library(fluidity STATIC source.cc)\n"
        "fluidity_apply_compile_flags(fluidity EXCEPTIONS OFF)\n"
        "add_subdirectory(grid)\n",
    )
    assert configured.returncode == 0, configured.stdout
    generated = "".join(
        path.read_text(encoding="utf-8") for path in (project / "build" / "grid").glob("grid_test*_discovery.cmake")
    )
    assert "[==[LABELS]==] [==[cpu;grid;unit]==]" in generated


def test_a_test_in_the_testing_folder_of_a_module_gets_the_name_of_its_module(tmp_path):
    project = tmp_path / "project"
    (project / "grid" / "testing").mkdir(parents=True)
    (project / "grid" / "testing" / "grid_support.test.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
    (project / "grid" / "CMakeLists.txt").write_text(
        "fluidity_add_test(grid_support_test SOURCES testing/grid_support.test.cc LABEL cpu)\n", encoding="utf-8"
    )
    configured = configure(
        project,
        "include(GoogleTest)\n"
        "set(BUILD_TESTING ON)\n"
        "add_library(GTest::gtest_main INTERFACE IMPORTED)\n"
        "add_library(fluidity STATIC source.cc)\n"
        "fluidity_apply_compile_flags(fluidity EXCEPTIONS OFF)\n"
        "add_subdirectory(grid)\n",
    )
    assert configured.returncode == 0, configured.stdout
    generated = "".join(
        path.read_text(encoding="utf-8")
        for path in (project / "build" / "grid").glob("grid_support_test*_discovery.cmake")
    )
    assert "[==[LABELS]==] [==[cpu;grid;unit]==]" in generated


def configure_one_test(tmp_path, folder, call):
    """Configures a project with one test program in `folder` (below the project) made by `call`."""
    project = tmp_path / "project"
    (project / folder).mkdir(parents=True)
    (project / folder / "grid.test.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
    (project / folder / "CMakeLists.txt").write_text(call + "\n", encoding="utf-8")
    configured = configure(
        project,
        "include(GoogleTest)\n"
        "enable_testing()\n"
        "set(BUILD_TESTING ON)\n"
        "add_library(GTest::gtest_main INTERFACE IMPORTED)\n"
        "add_library(fluidity STATIC source.cc)\n"
        "fluidity_apply_compile_flags(fluidity EXCEPTIONS OFF)\n"
        f"add_subdirectory({folder})\n",
    )
    generated = "".join(
        path.read_text(encoding="utf-8") for path in (project / "build" / folder).glob("*_test*_discovery.cmake")
    )
    ctest_file = project / "build" / folder / "CTestTestfile.cmake"
    if ctest_file.exists():
        generated += ctest_file.read_text(encoding="utf-8")
    return configured, generated


def test_a_test_with_the_functional_tier_gets_that_label(tmp_path):
    call = "fluidity_add_test(grid_test SOURCES grid.test.cc LABEL cpu TIER functional)"
    configured, generated = configure_one_test(tmp_path, "grid", call)
    assert configured.returncode == 0, configured.stdout
    assert 'LABELS "cpu;grid;functional"' in generated


def test_a_functional_program_is_one_ctest_entry_that_runs_the_whole_program(tmp_path):
    call = "fluidity_add_test(grid_test SOURCES grid.test.cc LABEL cpu TIER functional)"
    configured, generated = configure_one_test(tmp_path, "grid", call)
    assert configured.returncode == 0, configured.stdout
    assert 'add_test("grid_test" ' in generated
    assert '"--gtest_brief=1" "--gtest_output=xml:' in generated
    assert 'TIMEOUT "1800"' in generated
    assert "discover" not in generated.lower()


def test_a_test_under_tests_gets_the_name_of_its_module_folder(tmp_path):
    call = "fluidity_add_test(grid_test SOURCES grid.test.cc LABEL cpu TIER functional)"
    configured, generated = configure_one_test(tmp_path, "tests/grid", call)
    assert configured.returncode == 0, configured.stdout
    assert 'LABELS "cpu;grid;functional"' in generated


def test_a_tier_other_than_unit_or_functional_fails_the_configuration(tmp_path):
    call = "fluidity_add_test(grid_test SOURCES grid.test.cc LABEL cpu TIER slow)"
    configured, _ = configure_one_test(tmp_path, "grid", call)
    assert configured.returncode != 0
    assert "needs TIER unit or functional" in " ".join(configured.stdout.split())


def test_test_programs_are_listed_with_their_module_and_tier(tmp_path):
    project = tmp_path / "project"
    for folder, call in (
        ("grid", "fluidity_add_test(grid_test SOURCES a.test.cc LABEL cpu)\n"
         "fluidity_add_test(grid_big_functional_test SOURCES a.test.cc LABEL cpu TIER functional)"),
        ("contact", "fluidity_add_test(contact_test SOURCES a.test.cc LABEL cpu)"),
    ):
        (project / folder).mkdir(parents=True)
        (project / folder / "a.test.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
        (project / folder / "CMakeLists.txt").write_text(call + "\n", encoding="utf-8")
    configured = configure(
        project,
        "include(GoogleTest)\n"
        "set(BUILD_TESTING ON)\n"
        "add_library(GTest::gtest_main INTERFACE IMPORTED)\n"
        "add_library(fluidity STATIC source.cc)\n"
        "fluidity_apply_compile_flags(fluidity EXCEPTIONS OFF)\n"
        "add_subdirectory(grid)\n"
        "add_subdirectory(contact)\n"
        "fluidity_write_test_programs()\n",
    )
    assert configured.returncode == 0, configured.stdout
    listed = (project / "build" / "test_programs.txt").read_text(encoding="utf-8").splitlines()
    assert listed == ["grid unit grid_test", "grid functional grid_big_functional_test", "contact unit contact_test"]


def test_no_test_program_is_listed_when_testing_is_off(tmp_path):
    project = tmp_path / "project"
    configured = configure(
        project,
        "set(BUILD_TESTING OFF)\nfluidity_add_test(never SOURCES a.cc LABEL cpu)\nfluidity_write_test_programs()\n",
    )
    assert configured.returncode == 0, configured.stdout
    assert (project / "build" / "test_programs.txt").read_text(encoding="utf-8") == ""
