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
    """The container path of the header-alone file the CUDA build compiles for `header` (native/<include path>)."""
    return f"/out/build/linux-release-cuda/header_check/{header.removeprefix('native/')}{extension}"


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
    ):
        self.commands = []
        self._failing = failing or (lambda command: False)
        self._missing_programs = set(missing_programs)
        self._tracked = tracked if tracked is not None else ["native/fluidity/a.h", "native/fluidity/a.cc"]
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
        if any(Path(part).name == "check_rules.py" for part in command):
            output = self._rules_output
        return_code = 1 if self._failing(command) else 0
        return subprocess.CompletedProcess(command, return_code, stdout=output, stderr=None)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "native" / "fluidity").mkdir(parents=True)
    (tmp_path / "native" / "fluidity" / "a.h").write_text(HEADER_WITH_KNOWN_LABELS, encoding="utf-8")
    (tmp_path / "docs" / "method-paper").mkdir(parents=True)
    (tmp_path / "docs" / "method-paper" / "fluidity-method.typ").write_text(PAPER_TEXT, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT_TEXT, encoding="utf-8")
    build_folder = tmp_path / "out" / "build" / DEBUG_PRESET
    build_folder.mkdir(parents=True)
    entries = [
        {"directory": str(build_folder), "file": str(tmp_path / "native" / "fluidity" / "a.cc"), "command": "c++"},
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
        f"PASS cpu {DEBUG_PRESET}",
        f"PASS cpu {RELEASE_PRESET}",
        "PASS tidy",
        "PASS paper",
        "PASS cuda",
    ]


def test_rules_step_lints_the_workflow_files(monkeypatch, capsys, repo):
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "rules")
    assert fake.commands[-1] == ["actionlint"]


def test_failing_actionlint_fails_the_rules_step(monkeypatch, capsys, repo):
    fake = FakeRun(failing=lambda command: command == ["actionlint"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "rules")
    assert exit_code == 1
    assert lines[0].startswith("FAIL rules: `actionlint")


def test_read_line_with_unknown_label_fails_paper_step(monkeypatch, capsys, repo):
    header = repo / "native" / "fluidity" / "b.h"
    header.write_text("// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-missing>\n", encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: native/fluidity/b.h:3 cites <sec-missing>, which the paper does not define"]


def test_read_lines_with_known_labels_pass(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 0
    assert lines == ["PASS paper"]


def test_label_inside_paper_code_is_not_a_definition(monkeypatch, capsys, repo):
    paper = repo / "docs" / "method-paper" / "fluidity-method.typ"
    paper.write_text(PAPER_TEXT + "```text\n<sec-in-code>\n```\n", encoding="utf-8")
    header = repo / "native" / "fluidity" / "b.h"
    header.write_text("// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-in-code>\n", encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: native/fluidity/b.h:3 cites <sec-in-code>, which the paper does not define"]


@pytest.mark.parametrize("text", ["<expected>", "<first author> <year>", "<a.h>"])
def test_angle_brackets_that_are_not_a_label_are_not_a_citation(monkeypatch, capsys, repo, text):
    header = repo / "native" / "fluidity" / "b.h"
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
    tracked = ["native/fluidity/a.cc", "native/fluidity/b_gpu.cu", "native/fluidity/c_test.cc", "native/fluidity/a.h"]
    fake = FakeRun(tracked=tracked, compiled=["/src/native/fluidity/a.cc"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 1
    assert lines == [
        "FAIL cuda: not compiled by the CUDA build: native/fluidity/b_gpu.cu, native/fluidity/c_test.cc, "
        "native/fluidity/a.h"
    ]


def test_header_without_a_header_alone_file_fails_cuda_by_name(monkeypatch, capsys, repo):
    tracked = ["native/fluidity/a.cc", "native/fluidity/a.h", "native/fluidity/orphan.h"]
    compiled = ["/src/native/fluidity/a.cc", header_check_path("native/fluidity/a.h")]
    fake = FakeRun(tracked=tracked, compiled=compiled)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 1
    assert lines == ["FAIL cuda: not compiled by the CUDA build: native/fluidity/orphan.h"]


def test_cuda_header_compiled_alone_as_a_cu_file_passes_cuda(monkeypatch, capsys, repo):
    tracked = ["native/fluidity/a.cu", "native/fluidity/launch.cuh"]
    compiled = ["/src/native/fluidity/a.cu", header_check_path("native/fluidity/launch.cuh", ".cu")]
    fake = FakeRun(tracked=tracked, compiled=compiled)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 0
    assert lines == ["PASS cuda"]


def test_test_headers_need_no_header_alone_file(monkeypatch, capsys, repo):
    tracked = ["native/fluidity/test_support.h", "native/fluidity/math_test_gpu.h", "native/fluidity/latest.h"]
    fake = FakeRun(tracked=tracked, compiled=[header_check_path("native/fluidity/latest.h")])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "cuda")
    assert exit_code == 0
    assert lines == ["PASS cuda"]


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
    note = "native/fluidity/a.cc:3: NOTE one-sense-word 'row' is banned in one sense only: alone"
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
    assert "native/fluidity/a.cc" in tidied
    assert any(part.endswith(HEADER_CHECK_FILE) for part in tidied)


def test_tidy_leaves_a_file_outside_the_cpu_build_to_the_cuda_step(monkeypatch, capsys, repo):
    fake = FakeRun(tracked=["native/fluidity/a.cc", "native/fluidity/b_gpu_test.cc"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "tidy")
    tidied = [part for command in tidy_commands(fake) for part in command]
    assert exit_code == 0
    assert lines == ["PASS tidy"]
    assert "native/fluidity/b_gpu_test.cc" not in tidied


def test_dirty_tree_stops_every_step(monkeypatch, capsys, repo):
    fake = FakeRun(status=" M native/fluidity/a.h\0?? native/fluidity/new.h\0M  native/fluidity/a.cc\0")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 1
    assert lines == [
        "FAIL tree: unstaged or untracked: native/fluidity/a.h, native/fluidity/new.h",
        "NOT RUN format",
    ]
    assert not any(command[0] == "clang-format" for command in fake.commands)


def test_dirty_path_with_a_space_is_printed_unquoted(monkeypatch, capsys, repo):
    fake = FakeRun(status="?? native/fluidity/new file.h\0 M native/fluidity/caf\u00e9.h\0")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 1
    assert lines[0] == "FAIL tree: unstaged or untracked: native/fluidity/new file.h, native/fluidity/caf\u00e9.h"


def test_malformed_compile_commands_fails_tidy_with_a_reason(monkeypatch, capsys, repo):
    compile_commands = repo / "out" / "build" / DEBUG_PRESET / "compile_commands.json"
    compile_commands.write_text(json.dumps([{"file": "a.cc"}]), encoding="utf-8")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "tidy")
    assert exit_code == 1
    assert lines[0].startswith(f"FAIL tidy: out/build/{DEBUG_PRESET}/compile_commands.json is malformed")


def test_staged_changes_are_a_clean_tree(monkeypatch, capsys, repo):
    renamed = "R  native/fluidity/b.h\0native/fluidity/old.h\0"
    fake = FakeRun(status="M  native/fluidity/a.cc\0A  native/fluidity/new.h\0" + renamed)
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format")
    assert exit_code == 0
    assert lines == ["PASS format"]


def test_allow_dirty_notes_and_continues(monkeypatch, capsys, repo):
    fake = FakeRun(status="?? native/fluidity/new.h\0")
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--only", "format", "--allow-dirty")
    assert exit_code == 0
    assert lines == ["NOTE tree: unstaged or untracked files are being tested", "PASS format"]


def test_read_label_wrapped_onto_the_next_comment_line_is_checked(monkeypatch, capsys, repo):
    header = repo / "native" / "fluidity" / "b.h"
    header.write_text(
        "// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>,\n//          <sec-wrapped>\n",
        encoding="utf-8",
    )
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: native/fluidity/b.h:4 cites <sec-wrapped>, which the paper does not define"]


@pytest.mark.parametrize("comment", ["// a note <sec-commented>\n", "/* a note\n<sec-commented> */\n"])
def test_label_inside_paper_comment_is_not_a_definition(monkeypatch, capsys, repo, comment):
    paper = repo / "docs" / "method-paper" / "fluidity-method.typ"
    paper.write_text(PAPER_TEXT + comment, encoding="utf-8")
    header = repo / "native" / "fluidity" / "b.h"
    header.write_text(
        "// B.\n//\n// Read:    docs/method-paper/fluidity-method.typ <sec-commented>\n", encoding="utf-8"
    )
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--only", "paper")
    assert exit_code == 1
    assert lines == ["FAIL paper: native/fluidity/b.h:3 cites <sec-commented>, which the paper does not define"]


def test_cpu_step_runs_ctest_with_one_job_per_core(monkeypatch, capsys, repo):
    monkeypatch.setattr(check_commit.os, "cpu_count", lambda: 7)
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    ctest_runs = [command for command in fake.commands if command[0] == "ctest"]
    assert ctest_runs == [
        ["ctest", "--preset", DEBUG_PRESET, "--parallel", "7"],
        ["ctest", "--preset", RELEASE_PRESET, "--parallel", "7"],
    ]


def test_cpu_step_runs_ctest_with_one_job_when_the_core_count_is_unknown(monkeypatch, capsys, repo):
    monkeypatch.setattr(check_commit.os, "cpu_count", lambda: None)
    fake = FakeRun()
    run_gate(monkeypatch, capsys, repo, fake, "--only", "cpu")
    assert ["ctest", "--preset", DEBUG_PRESET, "--parallel", "1"] in fake.commands


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
        (["native/fluidity/grid/a.cc"], frozenset({"grid"})),
        (["native/fluidity/grid/a.cc", "native/fluidity/base/b.h"], frozenset({"grid", "base"})),
        (["docs/design/x.md", "tools/check_commit.py", "AGENTS.md"], frozenset()),
        ([], frozenset()),
        (["native/fluidity/grid/a.cc", "native/fluidity/CMakeLists.txt"], None),
        (["native/CMakeLists.txt"], None),
        (["cmake/fluidity_targets.cmake"], None),
        (["CMakeLists.txt"], None),
        (["CMakePresets.json"], None),
        (["conanfile.py"], None),
        (["conan.lock"], None),
        (["docs/CMakeLists.txt"], frozenset()),
    ],
)
def test_modules_of_changes(changed, expected):
    assert check_commit.modules_of_changes(changed) == expected


def test_commit_is_the_default_speed_and_runs_every_step(monkeypatch, capsys, repo):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(changed=["native/fluidity/grid/a.cc"]))
    assert exit_code == 0
    assert lines == ALL_STEP_LINES


def test_commit_speed_runs_all_tests_and_uses_the_tidy_cache(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["native/fluidity/grid/a.cc"])
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
    fake = FakeRun(changed=["native/fluidity/grid/a.cc"], untracked=["native/fluidity/contact/new.h"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert exit_code == 0
    assert lines == [
        "PASS format",
        "PASS rules",
        "NOTE modules: contact, grid",
        f"PASS cpu {DEBUG_PRESET}",
        f"NOT RUN cpu {RELEASE_PRESET} (quick)",
        "PASS tidy",
        "NOT RUN paper (quick)",
        "NOT RUN cuda (quick)",
    ]
    (debug_ctest,) = ctest_commands(fake)
    assert debug_ctest[debug_ctest.index("-L") + 1] == "^(contact|grid)$"
    assert ctest_commands(fake, RELEASE_PRESET) == []
    assert not any(command[:2] == ["docker", "build"] or command[0] == "typst" for command in fake.commands)
    assert ["cmake", "--build", "--preset", DEBUG_PRESET] in fake.commands
    assert ["cmake", "--build", "--preset", RELEASE_PRESET] not in fake.commands


def test_quick_runs_every_module_for_a_change_outside_a_module_folder(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["cmake/fluidity_targets.cmake"])
    _, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert "NOTE modules: all" in lines
    (debug_ctest,) = ctest_commands(fake)
    assert "-L" not in debug_ctest


def test_quick_runs_no_test_when_only_docs_and_tools_changed(monkeypatch, capsys, repo):
    fake = FakeRun(changed=["docs/design/x.md", "tools/check_commit.py"])
    exit_code, lines = run_gate(monkeypatch, capsys, repo, fake, "--speed", "quick")
    assert exit_code == 0
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
        f"NOT RUN cpu {DEBUG_PRESET}",
        f"NOT RUN cpu {RELEASE_PRESET} (quick)",
        "NOT RUN tidy",
        "NOT RUN paper (quick)",
        "NOT RUN cuda (quick)",
    ]


@pytest.mark.parametrize("speed", ["quick", "commit", "full"])
def test_only_runs_one_step_at_every_speed(monkeypatch, capsys, repo, speed):
    exit_code, lines = run_gate(monkeypatch, capsys, repo, FakeRun(), "--speed", speed, "--only", "paper")
    assert exit_code == 0
    assert lines == ["PASS paper"]
