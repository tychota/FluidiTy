"""Tests for tools/check_commit.py. Every command goes to a fake `subprocess.run`, so nothing is built here."""

import json
import subprocess
from pathlib import Path

import pytest

import check_commit

DEBUG_PRESET, RELEASE_PRESET = check_commit.host_cpu_presets()
HEADER_CHECK_FILE = "header_check/fluidity/a.h.cc"

PINNED_VERSIONS = {"clang-format": "23.1.5", "clang-tidy": "23.1.4"}
PYPROJECT_TEXT = '[dependency-groups]\ndev = [\n  "clang-format==23.1.5",\n  "clang-tidy==23.1.4",\n]\n'
# What each tool prints for --version, with the version in braces.
VERSION_TEXT = {
    "clang-format": "clang-format version {}\n",
    "clang-tidy": "LLVM (http://llvm.org/):\n  LLVM version {}\n  Optimized build.\n",
}

PAPER_TEXT = "= Conventions <sec-conventions>\n\nSee @sec-conventions.\n\n```cpp\n#include <expected>\n```\n"

HEADER_WITH_KNOWN_LABELS = (
    "// A thing.\n"
    "//\n"
    "// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>, <sec-conventions>\n"
    "\n"
    "#pragma once\n"
)


def header_check_path(header: str, extension: str = ".cc") -> str:
    """The container path of the header-alone file the CUDA build compiles for `header` (src/<include path>)."""
    return f"/out/build/linux-release-cuda/header_check/{header.removeprefix('src/')}{extension}"


class FakeRun:
    """Stands in for subprocess.run. Records each command and answers with a chosen exit code.

    `tracked` is what `git ls-files` lists, `status` what `git status --porcelain` prints, and `timing_out` names
    commands that raise subprocess.TimeoutExpired."""

    def __init__(
        self,
        failing=None,
        missing_programs=(),
        tracked=None,
        status="",
        timing_out=None,
        versions=None,
        compiled=None,
        rules_output="",
        changed=(),
        untracked=(),
        test_times=(("Fast.Unit", "base;cpu;unit", 0.01),),
    ):
        self.commands = []
        self._failing = failing or (lambda command: False)
        self._missing_programs = set(missing_programs)
        self._tracked = tracked if tracked is not None else ["src/fluidity/a.h", "src/fluidity/a.cc"]
        self._status = status
        self._timing_out = timing_out or (lambda command: False)
        self._versions = versions if versions is not None else dict(PINNED_VERSIONS)
        # What the CUDA build compiles, as container paths; by default every tracked .cc and .cu file, and the
        # header-alone file of every tracked header.
        if compiled is None:
            compiled = [f"/src/{path}" for path in self._tracked if path.endswith((".cc", ".cu"))]
            compiled += [header_check_path(path) for path in self._tracked if path.endswith((".h", ".cuh"))]
        self._compiled = compiled
        self._rules_output = rules_output
        self._changed = list(changed)
        self._untracked = list(untracked)
        # What ctest writes to the JUnit file: (test name, `;`-joined labels, seconds) for each test it ran.
        self._test_times = list(test_times)

    def junit_text(self):
        cases = "".join(
            f'<testcase name="{name}" classname="{name}" time="{seconds}" status="run"><properties>'
            f'<property name="cmake_labels" value="{labels}"/></properties></testcase>'
            for name, labels, seconds in self._test_times
        )
        return f'<?xml version="1.0"?><testsuite name="(empty)" tests="{len(self._test_times)}">{cases}</testsuite>'

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        if Path(command[0]).name in self._missing_programs:
            raise FileNotFoundError(command[0])
        if self._timing_out(command):
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))
        output = ""
        if command[:2] == ["git", "ls-files"]:
            output = "".join(f"{path}\n" for path in self._tracked)
        if command[:3] == ["git", "ls-files", "--others"]:
            output = "".join(f"{path}\0" for path in self._untracked)
        if command[:3] == ["git", "diff", "--name-only"]:
            output = "".join(f"{path}\0" for path in self._changed)
        if command[:2] == ["git", "status"]:
            output = self._status
        if command[1:] == ["--version"] and Path(command[0]).name in self._versions:
            output = VERSION_TEXT[Path(command[0]).name].format(self._versions[Path(command[0]).name])
        if command[:2] == ["docker", "run"]:
            output = "".join(f"compiled: {path}\n" for path in self._compiled)
        if command[0] == "ctest" and "--output-junit" in command:
            Path(command[command.index("--output-junit") + 1]).write_text(self.junit_text(), encoding="utf-8")
        if any(Path(part).name == "check_rules.py" for part in command):
            output = self._rules_output
        return_code = 1 if self._failing(command) else 0
        return subprocess.CompletedProcess(command, return_code, stdout=output, stderr=None)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "fluidity").mkdir(parents=True)
    (tmp_path / "src" / "fluidity" / "a.h").write_text(HEADER_WITH_KNOWN_LABELS, encoding="utf-8")
    (tmp_path / "docs" / "method-paper").mkdir(parents=True)
    (tmp_path / "docs" / "method-paper" / "fluidity-method.typ").write_text(PAPER_TEXT, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT_TEXT, encoding="utf-8")
    build_folder = tmp_path / "out" / "build" / DEBUG_PRESET
    build_folder.mkdir(parents=True)
    entries = [
        {"directory": str(build_folder), "file": str(tmp_path / "src" / "fluidity" / "a.cc"), "command": "c++"},
        {"directory": str(build_folder), "file": str(build_folder / HEADER_CHECK_FILE), "command": "c++"},
    ]
    (build_folder / "compile_commands.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


def run_gate(monkeypatch, capsys, repo, fake, *arguments):
    monkeypatch.setattr(check_commit.subprocess, "run", fake)
    exit_code = check_commit.main(list(arguments), repo_root=repo)
    return exit_code, capsys.readouterr().out.splitlines()


def test_all_steps_pass_exits_zero(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun())
    assert exit_code == 0
    assert lines == [
        "PASS format",
        "PASS rules",
        "PASS message",
        f"PASS cpu {DEBUG_PRESET}",
        f"PASS cpu {RELEASE_PRESET}",
        "PASS tidy",
        "PASS paper",
        "PASS cuda",
    ]


def test_rules_step_lints_the_workflow_files(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "rules")
    assert ["actionlint"] in fake.commands


def test_failing_actionlint_fails_the_rules_step(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: command == ["actionlint"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "rules")
    assert exit_code == 1
    assert lines[0].startswith("FAIL rules: `actionlint")


def test_read_line_with_unknown_label_fails_paper_step(monkeypatch, capsys, repo):
    header = repo / "src" / "fluidity" / "b.h"
    header.write_text("// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-missing>\n", encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: src/fluidity/b.h:3 cites <sec-missing>, which the paper does not define"]


def test_read_lines_with_known_labels_pass(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 0
    assert lines == ["PASS paper"]


def test_label_inside_paper_code_is_not_a_definition(monkeypatch, capsys, repo):
    paper = repo / "docs" / "method-paper" / "fluidity-method.typ"
    paper.write_text(PAPER_TEXT + "```text\n<sec-in-code>\n```\n", encoding="utf-8")
    header = repo / "src" / "fluidity" / "b.h"
    header.write_text("// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-in-code>\n", encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: src/fluidity/b.h:3 cites <sec-in-code>, which the paper does not define"]


@pytest.mark.parametrize("text", ["<expected>", "<first author> <year>", "<a.h>"])
def test_angle_brackets_that_are_not_a_label_are_not_a_citation(monkeypatch, capsys, repo, text):
    header = repo / "src" / "fluidity" / "b.h"
    header.write_text(f"// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ {text}\n", encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 0
    assert lines == ["PASS paper"]


def test_first_failure_is_reported_and_exit_is_nonzero(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: any(Path(part).name == "check_rules.py" for part in command))
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake)
    assert exit_code == 1
    assert lines[0] == "PASS format"
    assert lines[1].startswith("FAIL rules: ")
    assert lines[2:] == [
        "NOT RUN message",
        f"NOT RUN cpu {DEBUG_PRESET}",
        f"NOT RUN cpu {RELEASE_PRESET}",
        "NOT RUN tidy",
        "NOT RUN paper",
        "NOT RUN cuda",
    ]


def test_failed_step_writes_its_log(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: command[0] == "clang-format")
    run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert (repo / "out" / "check-commit" / "format.log").exists()


def test_docker_not_running_fails_cuda_step(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: command[:2] == ["docker", "info"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake)
    assert exit_code == 1
    assert lines[-1] == "FAIL cuda: Docker is not running (start it with: colima start)"


def test_only_flag_runs_one_step(monkeypatch, capsys, repo):
    fake = FakeRun()
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 0
    assert lines == ["PASS format"]
    assert all(command[0] in ("git", "clang-format") for command in fake.commands)


def test_cuda_is_never_reported_skipped(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(missing_programs={"docker"}))
    output = "\n".join(lines)
    assert exit_code == 1
    assert "FAIL cuda" in output
    assert "SKIP" not in output


def test_cuda_step_mounts_the_repository_read_only(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    docker_run = next(command for command in fake.commands if command[:2] == ["docker", "run"])
    assert f"{repo}:/src:ro" in docker_run
    assert f"{check_commit.cuda_out_volume(repo)}:/out" in docker_run


def test_cuda_volume_name_depends_on_the_repository_path(tmp_path):
    first = check_commit.cuda_out_volume(tmp_path / "one")
    second = check_commit.cuda_out_volume(tmp_path / "two")
    assert first != second
    assert first == check_commit.cuda_out_volume(tmp_path / "one")
    assert first.startswith("fluidity-cuda-out-")
    assert all(character.isalnum() or character in "-_." for character in first)


def test_source_missing_from_the_cuda_build_fails_cuda(monkeypatch, capsys, repo):
    tracked = ["src/fluidity/a.cc", "src/fluidity/b_gpu.cu", "src/fluidity/c.test.cc", "src/fluidity/a.h"]
    fake = FakeRun(tracked=tracked, compiled=["/src/src/fluidity/a.cc"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 1
    assert lines == [
        "FAIL cuda: not compiled by the CUDA build: src/fluidity/b_gpu.cu, src/fluidity/c.test.cc, "
        "src/fluidity/a.h"
    ]


def test_header_without_a_header_alone_file_fails_cuda_by_name(monkeypatch, capsys, repo):
    tracked = ["src/fluidity/a.cc", "src/fluidity/a.h", "src/fluidity/orphan.h"]
    compiled = ["/src/src/fluidity/a.cc", header_check_path("src/fluidity/a.h")]
    fake = FakeRun(tracked=tracked, compiled=compiled)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 1
    assert lines == ["FAIL cuda: not compiled by the CUDA build: src/fluidity/orphan.h"]


def test_cuda_header_compiled_alone_as_a_cu_file_passes_cuda(monkeypatch, capsys, repo):
    tracked = ["src/fluidity/a.cu", "src/fluidity/launch.cuh"]
    compiled = ["/src/src/fluidity/a.cu", header_check_path("src/fluidity/launch.cuh", ".cu")]
    fake = FakeRun(tracked=tracked, compiled=compiled)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 0
    assert lines == ["PASS cuda"]


def test_headers_in_a_testing_folder_need_no_header_alone_file(monkeypatch, capsys, repo):
    tracked = [
        "src/fluidity/base/testing/base_support.h",
        "src/fluidity/grid/testing/cell_size_rule_runner_gpu.h",
        "src/fluidity/base/latest.h",
    ]
    fake = FakeRun(tracked=tracked, compiled=[header_check_path("src/fluidity/base/latest.h")])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 0
    assert lines == ["PASS cuda"]


def test_header_with_test_in_its_name_outside_a_testing_folder_needs_a_header_alone_file(monkeypatch, capsys, repo):
    tracked = ["src/fluidity/base/test_helpers.h"]
    fake = FakeRun(tracked=tracked, compiled=[])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 1
    assert lines == ["FAIL cuda: not compiled by the CUDA build: src/fluidity/base/test_helpers.h"]


def test_header_in_a_testing_folder_at_another_depth_needs_a_header_alone_file(monkeypatch, capsys, repo):
    tracked = ["src/testing/x.h", "src/fluidity/grid/tables/testing/y.h"]
    fake = FakeRun(tracked=tracked, compiled=[])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 1
    assert lines == ["FAIL cuda: not compiled by the CUDA build: src/testing/x.h, src/fluidity/grid/tables/testing/y.h"]


def test_container_tidies_the_sources_only_the_cuda_build_compiles(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    script = next(command for command in fake.commands if command[:2] == ["docker", "run"])[-1]
    assert "clang-tidy -p" in script
    assert "gpu_only_sources.txt" in script


def test_clang_format_version_other_than_pinned_fails_format(monkeypatch, capsys, repo):
    fake = FakeRun(versions={"clang-format": "19.1.5", "clang-tidy": "23.1.4"})
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 1
    assert lines == ["FAIL format: clang-format 19.1.5 runs, but pyproject.toml pins 23.1.5 (run: uv sync --frozen)"]
    assert not any(command[:2] == ["clang-format", "--dry-run"] for command in fake.commands)


def test_clang_tidy_version_other_than_pinned_fails_tidy(monkeypatch, capsys, repo):
    fake = FakeRun(versions={"clang-format": "23.1.5", "clang-tidy": "23.1.0"})
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy")
    assert exit_code == 1
    assert lines == ["FAIL tidy: clang-tidy 23.1.0 runs, but pyproject.toml pins 23.1.4 (run: uv sync --frozen)"]


def test_rules_step_prints_the_notes_of_staged_lines(monkeypatch, capsys, repo):
    note = "src/fluidity/a.cc:3: NOTE one-sense-word 'row' is banned in one sense only: alone"
    fake = FakeRun(rules_output=note + "\n")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "rules")
    rules_command = next(command for command in fake.commands if any(Path(part).name == "check_rules.py" for part in command))
    assert "--notes-on-staged-lines" in rules_command
    assert exit_code == 0
    assert lines == [f"rules note: {note}", "PASS rules"]


def test_container_runs_cpu_and_no_gpu_tests_by_exact_label(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    docker_run = next(command for command in fake.commands if command[:2] == ["docker", "run"])
    assert any("-L '^(cpu|no_gpu)$' --no-tests=error" in part for part in docker_run)


def test_container_always_fetches_the_conan_packages(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    script = next(command for command in fake.commands if command[:2] == ["docker", "run"])[-1]
    conan_line = next(line for line in script.splitlines() if "conan install" in line)
    assert not conan_line.startswith(" ")
    assert "if [" not in script


def test_docker_info_timeout_fails_as_not_running(monkeypatch, capsys, repo):
    fake = FakeRun(timing_out=lambda command: command[:2] == ["docker", "info"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 1
    assert lines == ["FAIL cuda: Docker is not running (start it with: colima start)"]


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("arm64", "linux/arm64"), ("aarch64", "linux/arm64"), ("x86_64", "linux/amd64"), ("AMD64", "linux/amd64")],
)
def test_container_platform_follows_the_machine(monkeypatch, machine, expected):
    monkeypatch.setattr(check_commit.platform, "machine", lambda: machine)
    assert check_commit.container_platform() == expected


def test_cpu_step_always_fetches_the_conan_packages(monkeypatch, capsys, repo):
    toolchain = repo / "out" / "conan" / DEBUG_PRESET / "conan_toolchain.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text("", encoding="utf-8")
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    debug_conan_runs = [c for c in fake.commands if c[:2] == ["conan", "install"] and f"out/conan/{DEBUG_PRESET}" in c]
    assert len(debug_conan_runs) == 1


def test_missing_compile_commands_fails_tidy(monkeypatch, capsys, repo):
    (repo / "out" / "build" / DEBUG_PRESET / "compile_commands.json").unlink()
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "tidy")
    assert exit_code == 1
    assert lines == [
        f"FAIL tidy: out/build/{DEBUG_PRESET}/compile_commands.json is missing; run the cpu step first"
    ]


def tidy_commands(fake):
    """The commands that run tools/run_tidy.py, which runs clang-tidy over the files it is given."""
    return [command for command in fake.commands if any(Path(part).name == "run_tidy.py" for part in command)]


def test_tidy_checks_each_header_alone(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy")
    tidied = [part for command in tidy_commands(fake) for part in command]
    assert "src/fluidity/a.cc" in tidied
    assert any(part.endswith(HEADER_CHECK_FILE) for part in tidied)


def test_tidy_leaves_a_file_outside_the_cpu_build_to_the_cuda_step(monkeypatch, capsys, repo):
    fake = FakeRun(tracked=["src/fluidity/a.cc", "src/fluidity/b.gpu.test.cc"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy")
    tidied = [part for command in tidy_commands(fake) for part in command]
    assert exit_code == 0
    assert lines == ["PASS tidy"]
    assert "src/fluidity/b.gpu.test.cc" not in tidied


def test_dirty_tree_stops_every_step(monkeypatch, capsys, repo):
    fake = FakeRun(status=" M src/fluidity/a.h\0?? src/fluidity/new.h\0M  src/fluidity/a.cc\0")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 1
    assert lines == [
        "FAIL tree: unstaged or untracked: src/fluidity/a.h, src/fluidity/new.h",
        "NOT RUN format",
    ]
    assert not any(command[0] == "clang-format" for command in fake.commands)


def test_dirty_path_with_a_space_is_printed_unquoted(monkeypatch, capsys, repo):
    fake = FakeRun(status="?? src/fluidity/new file.h\0 M src/fluidity/caf\u00e9.h\0")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 1
    assert lines[0] == "FAIL tree: unstaged or untracked: src/fluidity/new file.h, src/fluidity/caf\u00e9.h"


def test_malformed_compile_commands_fails_tidy_with_a_reason(monkeypatch, capsys, repo):
    compile_commands = repo / "out" / "build" / DEBUG_PRESET / "compile_commands.json"
    compile_commands.write_text(json.dumps([{"file": "a.cc"}]), encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "tidy")
    assert exit_code == 1
    assert lines[0].startswith(f"FAIL tidy: out/build/{DEBUG_PRESET}/compile_commands.json is malformed")


def test_staged_changes_are_a_clean_tree(monkeypatch, capsys, repo):
    renamed = "R  src/fluidity/b.h\0src/fluidity/old.h\0"
    fake = FakeRun(status="M  src/fluidity/a.cc\0A  src/fluidity/new.h\0" + renamed)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 0
    assert lines == ["PASS format"]


def test_allow_dirty_notes_and_continues(monkeypatch, capsys, repo):
    fake = FakeRun(status="?? src/fluidity/new.h\0")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format", "--allow-dirty")
    assert exit_code == 0
    assert lines == ["NOTE tree: unstaged or untracked files are being tested", "PASS format"]


def test_read_label_wrapped_onto_the_next_comment_line_is_checked(monkeypatch, capsys, repo):
    header = repo / "src" / "fluidity" / "b.h"
    header.write_text(
        "// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>,\n//          <sec-wrapped>\n",
        encoding="utf-8",
    )
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: src/fluidity/b.h:4 cites <sec-wrapped>, which the paper does not define"]


@pytest.mark.parametrize("comment", ["// a note <sec-commented>\n", "/* a note\n<sec-commented> */\n"])
def test_label_inside_paper_comment_is_not_a_definition(monkeypatch, capsys, repo, comment):
    paper = repo / "docs" / "method-paper" / "fluidity-method.typ"
    paper.write_text(PAPER_TEXT + comment, encoding="utf-8")
    header = repo / "src" / "fluidity" / "b.h"
    header.write_text(
        "// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-commented>\n", encoding="utf-8"
    )
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: src/fluidity/b.h:3 cites <sec-commented>, which the paper does not define"]


def without_junit(command):
    """`command` without the `--output-junit <file>` pair that the debug run adds."""
    if "--output-junit" not in command:
        return list(command)
    index = command.index("--output-junit")
    return list(command[:index]) + list(command[index + 2 :])


def test_cpu_step_runs_ctest_with_one_job_per_core(monkeypatch, capsys, repo):
    monkeypatch.setattr(check_commit.os, "cpu_count", lambda: 7)
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    ctest_runs = [without_junit(command) for command in fake.commands if command[0] == "ctest"]
    assert ctest_runs == [
        ["ctest", "--preset", DEBUG_PRESET, "--parallel", "7"],
        ["ctest", "--preset", RELEASE_PRESET, "--parallel", "7"],
    ]


def test_cpu_step_runs_ctest_with_one_job_when_the_core_count_is_unknown(monkeypatch, capsys, repo):
    monkeypatch.setattr(check_commit.os, "cpu_count", lambda: None)
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    assert ["ctest", "--preset", DEBUG_PRESET, "--parallel", "1"] in [without_junit(c) for c in fake.commands]


def test_container_runs_its_tests_on_every_core(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    script = next(command for command in fake.commands if command[:2] == ["docker", "run"])[-1]
    ctest_line = next(line for line in script.splitlines() if line.startswith("ctest"))
    assert '--parallel "$(nproc)"' in ctest_line


def test_tidy_runs_one_command_that_may_use_the_cache(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy")
    (command,) = tidy_commands(fake)
    assert "--no-cache" not in command
    assert command[command.index("--build-folder") + 1] == f"out/build/{DEBUG_PRESET}"


def test_no_cache_flag_reaches_the_tidy_command(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy", "--no-cache")
    (command,) = tidy_commands(fake)
    assert "--no-cache" in command


def test_failing_tidy_fails_the_step_with_its_log(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: any(Path(part).name == "run_tidy.py" for part in command))
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy")
    assert exit_code == 1
    assert lines[0].startswith("FAIL tidy:")
    assert "out/check-commit/tidy.log" in lines[0]


ALL_STEP_LINES = [
    "PASS format",
    "PASS rules",
    "PASS message",
    f"PASS cpu {DEBUG_PRESET}",
    f"PASS cpu {RELEASE_PRESET}",
    "PASS tidy",
    "PASS paper",
    "PASS cuda",
]


def ctest_commands(fake, preset=DEBUG_PRESET):
    return [command for command in fake.commands if command[:3] == ["ctest", "--preset", preset]]


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (["src/fluidity/grid/a.cc"], frozenset({"grid"})),
        (["src/fluidity/grid/a.cc", "src/fluidity/base/b.h"], frozenset({"grid", "base"})),
        (["src/fluidity/grid/testing/grid_support.h"], frozenset({"grid"})),
        (["src/fluidity/contact/testing/contact_cases.h", "src/fluidity/base/a.test.cc"], frozenset({"contact", "base"})),
        (["docs/design/x.md", "tools/check_commit.py", "AGENTS.md"], frozenset()),
        ([], frozenset()),
        (["src/fluidity/grid/a.cc", "src/fluidity/CMakeLists.txt"], None),
        (["src/CMakeLists.txt"], None),
        (["cmake/fluidity_targets.cmake"], None),
        (["CMakeLists.txt"], None),
        (["CMakePresets.json"], None),
        (["conanfile.py"], None),
        (["conan.lock"], None),
        (["pyproject.toml"], None),
        (["uv.lock"], None),
        (["docs/CMakeLists.txt"], frozenset()),
        (["tests/grid/build_grid.test.cc"], frozenset({"grid"})),
        (["tests/grid/CMakeLists.txt", "src/fluidity/base/a.h"], frozenset({"grid", "base"})),
        (["tests/CMakeLists.txt"], None),
        (["tests/README.md"], None),
    ],
)
def test_modules_of_changes(changed, expected):
    assert check_commit.modules_of_changes(changed) == expected


def test_commit_is_the_default_speed_and_runs_every_step(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun())
    assert exit_code == 0
    assert lines == ALL_STEP_LINES


def test_commit_speed_runs_all_tests_and_uses_the_tidy_cache(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["src/fluidity/grid/a.cc"])
    run_gate(monkeypatch, capsys, repo, fake, "--speed", "commit")
    assert all("-L" not in command for command in ctest_commands(fake))
    assert "--no-cache" not in tidy_commands(fake)[0]


def test_full_speed_runs_every_step_and_ignores_the_tidy_cache(monkeypatch, capsys, repo):
    fake = FakeRun()
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "full")
    assert exit_code == 0
    assert lines == ALL_STEP_LINES
    assert "--no-cache" in tidy_commands(fake)[0]


def test_quick_runs_the_debug_tests_of_the_changed_modules_and_names_what_it_left_out(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["src/fluidity/grid/a.cc"], untracked=["src/fluidity/contact/new.h"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert exit_code == 3
    assert lines == [
        "PASS format",
        "PASS rules",
        "NOT RUN message (quick)",
        "NOTE modules: contact, grid",
        f"PASS cpu {DEBUG_PRESET}",
        f"NOT RUN cpu {RELEASE_PRESET} (quick)",
        "PASS tidy",
        "NOT RUN paper (quick)",
        "NOT RUN cuda (quick)",
        "QUICK ONLY: not the commit gate",
    ]
    (debug_ctest,) = ctest_commands(fake)
    assert debug_ctest[debug_ctest.index("-L") + 1] == "^(contact|grid)$"
    assert "^unit$" in debug_ctest and "^functional$" not in debug_ctest
    assert ctest_commands(fake, RELEASE_PRESET) == []
    assert not any(command[:2] == ["docker", "build"] or command[0] == "typst" for command in fake.commands)
    assert ["cmake", "--build", "--preset", DEBUG_PRESET] in fake.commands
    assert ["cmake", "--build", "--preset", RELEASE_PRESET] not in fake.commands


def test_quick_runs_every_module_for_a_change_outside_a_module_folder(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["cmake/fluidity_targets.cmake"])
    _, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert "NOTE modules: all" in lines
    (debug_ctest,) = ctest_commands(fake)
    assert debug_ctest[debug_ctest.index("-L") + 1] == "^unit$"
    assert "^functional$" not in debug_ctest


def test_quick_runs_no_test_when_only_docs_and_tools_changed(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["docs/design/x.md", "tools/check_commit.py"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert exit_code == 3
    assert "NOTE modules: none" in lines
    assert ctest_commands(fake) == []
    assert ["cmake", "--build", "--preset", DEBUG_PRESET] in fake.commands


def test_quick_runs_every_module_when_git_cannot_list_the_changes(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: command[:3] == ["git", "diff", "--name-only"])
    _, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert "NOTE modules: all" in lines


def test_a_failure_in_quick_still_says_what_it_left_out(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: command[0] == "clang-format")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert exit_code == 1
    assert lines[0].startswith("FAIL format")
    assert lines[1:] == [
        "NOT RUN rules",
        "NOT RUN message (quick)",
        f"NOT RUN cpu {DEBUG_PRESET}",
        f"NOT RUN cpu {RELEASE_PRESET} (quick)",
        "NOT RUN tidy",
        "NOT RUN paper (quick)",
        "NOT RUN cuda (quick)",
    ]


@pytest.mark.parametrize("speed", ["quick", "commit", "full"])
def test_only_runs_one_step_at_every_speed(monkeypatch, capsys, repo, speed):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--speed", speed, "--only", "paper")
    assert exit_code == (3 if speed == "quick" else 0)
    assert lines == ["PASS paper", *(["QUICK ONLY: not the commit gate"] if speed == "quick" else [])]


@pytest.mark.parametrize("speed", ["commit", "full"])
def test_commit_and_full_speed_exit_zero_when_every_check_passes(monkeypatch, capsys, repo, speed):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--speed", speed)
    assert exit_code == 0
    assert "QUICK ONLY: not the commit gate" not in lines


def test_a_quick_pass_never_exits_zero_and_ends_with_the_quick_only_line(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--speed", "quick")
    assert exit_code == check_commit.EXIT_QUICK_ONLY == 3
    assert lines[-1] == "QUICK ONLY: not the commit gate"


def test_a_failed_quick_run_exits_one_and_not_three(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: command[0] == "clang-format")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert exit_code == 1
    assert "QUICK ONLY: not the commit gate" not in lines


def test_the_help_text_names_the_exit_codes(capsys):
    with pytest.raises(SystemExit):
        check_commit.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "0 = commit or full speed" in help_text
    assert "3 = quick speed" in help_text
    assert "QUICK ONLY: not the commit gate" in help_text


def test_quick_with_only_runs_the_named_step_in_full(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["src/fluidity/grid/a.cc"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick", "--only", "cpu")
    assert exit_code == 3
    assert lines[:2] == [f"PASS cpu {DEBUG_PRESET}", f"PASS cpu {RELEASE_PRESET}"]
    assert all("-L" not in command for command in ctest_commands(fake))


def test_quick_lists_a_moved_file_under_its_old_and_new_path(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["src/fluidity/grid/a.cc"])
    run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    (diff_command,) = [command for command in fake.commands if command[:3] == ["git", "diff", "--name-only"]]
    assert "--no-renames" in diff_command


def test_a_change_under_a_testing_folder_selects_the_functional_tests_of_its_module():
    changed = ["src/fluidity/grid/testing/grid_support.cc", "src/fluidity/base/a.h", "src/fluidity/contact/testing/x.h"]
    assert check_commit.modules_with_changed_functional_tests(changed) == frozenset({"grid", "contact"})
    assert check_commit.modules_with_changed_functional_tests(["src/fluidity/grid/a.test.cc"]) == frozenset()


def test_modules_with_changed_functional_tests_are_those_with_a_change_under_tests():
    changed = ["tests/grid/a.test.cc", "src/fluidity/base/a.h", "tests/contact/CMakeLists.txt", "tests/CMakeLists.txt"]
    assert check_commit.modules_with_changed_functional_tests(changed) == frozenset({"grid", "contact"})
    assert check_commit.modules_with_changed_functional_tests(["src/fluidity/base/a.h"]) == frozenset()


def test_quick_runs_the_functional_tests_of_a_module_whose_tests_changed(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["tests/grid/a.test.cc", "src/fluidity/base/a.h"])
    _, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert "NOTE modules: base, grid" in lines
    assert "NOTE functional tests: grid" in lines
    unit_run, functional_run = ctest_commands(fake)
    assert unit_run[unit_run.index("-L") + 1] == "^(base|grid)$" and "^unit$" in unit_run
    assert functional_run[functional_run.index("-L") + 1] == "^(grid)$" and "^functional$" in functional_run
    assert "^unit$" not in functional_run


def test_quick_runs_no_functional_test_when_only_src_changed(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["src/fluidity/grid/a.cc"])
    _, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert not any(line.startswith("NOTE functional tests") for line in lines)
    assert all("^functional$" not in command for command in ctest_commands(fake))


def test_commit_speed_times_the_debug_run_only(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--speed", "commit")
    assert "--output-junit" in ctest_commands(fake)[0]
    assert "--output-junit" not in ctest_commands(fake, RELEASE_PRESET)[0]


def test_unit_test_over_200_ms_fails_the_debug_run_and_is_named(monkeypatch, capsys, repo):
    fake = FakeRun(test_times=[("Fast.Unit", "base;cpu;unit", 0.01), ("Slow.Case", "grid;cpu;unit", 0.25)])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    assert exit_code == 1
    assert lines[0] == f"FAIL cpu {DEBUG_PRESET}: unit tests over 200 ms: Slow.Case (250 ms)"


def test_unit_test_between_50_and_200_ms_gets_a_note_and_passes(monkeypatch, capsys, repo):
    fake = FakeRun(test_times=[("Middling.Case", "grid;cpu;unit", 0.08), ("Exactly.Limit", "grid;cpu;unit", 0.2)])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    assert exit_code == 0
    assert "NOTE unit test over 50 ms: Middling.Case (80 ms)" in lines
    assert "NOTE unit test over 50 ms: Exactly.Limit (200 ms)" in lines
    assert f"PASS cpu {DEBUG_PRESET}" in lines


def test_functional_test_may_take_any_time(monkeypatch, capsys, repo):
    fake = FakeRun(test_times=[("Long.Run", "grid;cpu;functional", 30.0)])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    assert exit_code == 0
    assert not any(line.startswith("NOTE unit test") for line in lines)


def test_unreadable_junit_file_fails_the_debug_run(monkeypatch, capsys, repo):
    class NoJunit(FakeRun):
        def junit_text(self):
            return "not xml"

    exit_code, lines = run_gate(monkeypatch, capsys, repo, NoJunit(), "--only", "cpu")
    assert exit_code == 1
    assert lines[0].startswith(f"FAIL cpu {DEBUG_PRESET}: the JUnit file of ctest cannot be read")


def test_slow_unit_tests_reads_labels_and_times_from_the_junit_text():
    fake = FakeRun(test_times=[("A.One", "base;cpu;unit", 0.3), ("B.Two", "base;cpu;functional", 9.0), ("C.Three", "base;cpu;unit", 0.06)])
    over_limit, over_note = check_commit.slow_unit_tests(fake.junit_text())
    assert over_limit == [("A.One", 0.3)]
    assert over_note == [("C.Three", 0.06)]


def test_test_without_a_tier_label_fails_the_debug_run(monkeypatch, capsys, repo):
    fake = FakeRun(test_times=[("Fine.Case", "grid;cpu;unit", 0.01), ("Plain.Case", "grid;cpu", 0.01)])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    assert exit_code == 1
    assert lines[0] == f"FAIL cpu {DEBUG_PRESET}: tests without a tier label (unit or functional): Plain.Case"


def test_missing_junit_file_fails_the_debug_run(monkeypatch, capsys, repo):
    class DeletesJunit(FakeRun):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if command[0] == "ctest" and "--output-junit" in command:
                Path(command[command.index("--output-junit") + 1]).unlink()
            return result

    exit_code, lines = run_gate(monkeypatch, capsys, repo, DeletesJunit(), "--only", "cpu")
    assert exit_code == 1
    assert lines[0].startswith(f"FAIL cpu {DEBUG_PRESET}: the JUnit file of ctest cannot be read")


def test_rules_step_covers_src_and_tests(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "rules")
    rules_command = next(c for c in fake.commands if any(Path(part).name == "check_rules.py" for part in c))
    assert str(repo / "src") in rules_command and str(repo / "tests") in rules_command


def test_tidy_and_the_cuda_check_cover_tests_as_they_cover_src(monkeypatch, capsys, repo):
    fake = FakeRun(tracked=["src/fluidity/a.h", "src/fluidity/a.cc", "tests/grid/b.test.cc"])
    run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy")
    listings = [c for c in fake.commands if c[:2] == ["git", "ls-files"]]
    assert any("tests/*.cc" in listing for listing in listings)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(tracked=["src/fluidity/a.h", "src/fluidity/a.cc", "tests/grid/b.test.cc"], compiled=[header_check_path("src/fluidity/a.h"), "/src/src/fluidity/a.cc"]), "--only", "cuda")
    assert exit_code == 1
    assert lines == ["FAIL cuda: not compiled by the CUDA build: tests/grid/b.test.cc"]


def test_paper_check_reads_the_read_lines_of_tests_too(monkeypatch, capsys, repo):
    (repo / "tests" / "grid").mkdir(parents=True)
    (repo / "tests" / "grid" / "a.test.cc").write_text("// Read:    docs/method-paper/fluidity-method.typ <sec-nowhere>\n", encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: tests/grid/a.test.cc:1 cites <sec-nowhere>, which the paper does not define"]


def message_commands(fake):
    return [command for command in fake.commands if any(Path(part).name == "check_message.py" for part in command)]


def test_message_step_checks_the_message_of_head_when_the_tree_has_no_change(monkeypatch, capsys, repo):
    fake = FakeRun()
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "message")
    assert exit_code == 0
    assert lines == ["PASS message"]
    assert message_commands(fake)[0][-1] == "HEAD"


@pytest.mark.parametrize("fake", [FakeRun(changed=["src/fluidity/a.cc"]), FakeRun(untracked=["src/fluidity/b.h"])])
def test_message_step_is_not_run_with_uncommitted_changes(monkeypatch, capsys, repo, fake):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "message", "--allow-dirty")
    assert exit_code == 0
    assert lines[-1] == "NOT RUN message (uncommitted changes)"
    assert message_commands(fake) == []


def test_failing_message_fails_the_gate(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: any(Path(part).name == "check_message.py" for part in command))
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "message")
    assert exit_code == 1
    assert lines[0].startswith("FAIL message: ")


def test_quick_leaves_the_message_step_out(monkeypatch, capsys, repo):
    fake = FakeRun()
    _, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert "NOT RUN message (quick)" in lines
    assert message_commands(fake) == []


class NotesWriter(FakeRun):
    """A FakeRun whose check_rules.py writes `count` notes to the --json path it is given, as the real one does."""

    def __init__(self, count, **kwargs):
        super().__init__(**kwargs)
        self._count = count

    def __call__(self, command, **kwargs):
        if "--json" in command:
            Path(command[command.index("--json") + 1]).write_text(json.dumps([{}] * self._count), encoding="utf-8")
        return super().__call__(command, **kwargs)


def test_rules_step_prints_the_count_of_doubtful_lines_and_the_command_that_lists_them(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, NotesWriter(3), "--only", "rules")
    assert exit_code == 0
    assert lines == [
        "NOTE 3 doubtful lines: uv run python tools/check_rules.py --notes-since HEAD~1 --json out/check-commit/notes.json",
        "PASS rules",
    ]


def test_rules_step_prints_no_doubtful_line_line_for_zero_notes(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, NotesWriter(0), "--only", "rules")
    assert exit_code == 0
    assert lines == ["PASS rules"]


def test_doubtful_lines_never_fail_the_gate_and_a_failed_listing_prints_nothing(monkeypatch, capsys, repo):
    fake = NotesWriter(2, failing=lambda command: "--json" in command)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "rules")
    assert exit_code == 0
    assert lines == ["PASS rules"]


def test_quick_does_not_list_doubtful_lines(monkeypatch, capsys, repo):
    fake = NotesWriter(2)
    _, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert not any("doubtful" in line for line in lines)
