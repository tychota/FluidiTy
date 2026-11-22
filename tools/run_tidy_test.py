"""Tests for tools/run_tidy.py. The compiler and clang-tidy are a fake `subprocess.run`, so nothing is compiled."""

import json
import subprocess
from pathlib import Path

import pytest

import run_tidy

COMPILER = "fake-cc"
VERSION_TEXT = "LLVM version 23.1.4\n"


class FakeRun:
    """Stands in for subprocess.run. A `fake-cc ... -M` command prints the dependencies chosen for its file; a
    clang-tidy command is recorded and fails for the files in `failing`."""

    def __init__(self, repo: Path, dependencies: dict[str, list[str]], failing=()):
        self.repo = repo
        self.dependencies = dependencies
        self.failing = set(failing)
        self.checked: list[str] = []

    def __call__(self, command, **kwargs):
        if command[:2] == ["clang-tidy", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout=VERSION_TEXT)
        if command[0] == COMPILER:
            source = command[command.index("--source") + 1]
            names = " ".join([source, *self.dependencies.get(Path(source).name, [])])
            return subprocess.CompletedProcess(command, 0, stdout=f"x.o: {names}\n")
        assert command[0] == "clang-tidy", command
        name = Path(command[-1]).name
        self.checked.append(name)
        code = 1 if name in self.failing else 0
        return subprocess.CompletedProcess(command, code, stdout=f"warning in {name}\n" if code else "")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "native" / "fluidity").mkdir(parents=True)
    (tmp_path / "out" / "build" / "debug").mkdir(parents=True)
    (tmp_path / ".clang-tidy").write_text("Checks: '-*,bugprone-*'\n", encoding="utf-8")
    for name in ("a", "b", "c"):
        (tmp_path / "native" / "fluidity" / f"{name}.cc").write_text(f'#include "{name}.h"\n', encoding="utf-8")
        (tmp_path / "native" / "fluidity" / f"{name}.h").write_text("#pragma once\n", encoding="utf-8")
    entries = [
        {
            "directory": str(tmp_path / "out" / "build" / "debug"),
            "command": f"{COMPILER} -I{tmp_path} --source {tmp_path / 'native' / 'fluidity' / (name + '.cc')} -o x.o -c",
            "file": str(tmp_path / "native" / "fluidity" / f"{name}.cc"),
        }
        for name in ("a", "b", "c")
    ]
    (tmp_path / "out" / "build" / "debug" / "compile_commands.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


def sources(repo: Path) -> list[Path]:
    return [repo / "native" / "fluidity" / f"{name}.cc" for name in ("c", "a", "b")]


def run(monkeypatch, repo: Path, fake: FakeRun, *, no_cache: bool = False) -> list[run_tidy.TidyResult]:
    monkeypatch.setattr(run_tidy.subprocess, "run", fake)
    cache = None if no_cache else repo / "out" / "tidy-cache"
    return run_tidy.run_tidy(sources(repo), repo / "out" / "build" / "debug", repo, cache, jobs=4)


def dependencies_of(repo: Path) -> dict[str, list[str]]:
    folder = repo / "native" / "fluidity"
    return {f"{name}.cc": [str(folder / f"{name}.h"), str(folder / "shared.h")] for name in ("a", "b", "c")} | {
        "c.cc": [str(folder / "c.h")]
    }


def test_files_that_passed_are_not_checked_again(monkeypatch, repo):
    (repo / "native" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    first = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, first)
    assert sorted(first.checked) == ["a.cc", "b.cc", "c.cc"]
    second = FakeRun(repo, dependencies_of(repo))
    results = run(monkeypatch, repo, second)
    assert second.checked == []
    assert all(result.passed for result in results)


def test_a_changed_header_makes_its_includers_miss_the_cache(monkeypatch, repo):
    shared = repo / "native" / "fluidity" / "shared.h"
    shared.write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    shared.write_text("// shared, changed\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert sorted(second.checked) == ["a.cc", "b.cc"]


def test_a_changed_unrelated_file_does_not_miss_the_cache(monkeypatch, repo):
    (repo / "native" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    (repo / "native" / "fluidity" / "unrelated.h").write_text("// new\n", encoding="utf-8")
    (repo / "native" / "fluidity" / "c.h").write_text("#pragma once\n// changed\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert second.checked == ["c.cc"]


def test_a_changed_source_file_is_checked_again(monkeypatch, repo):
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo) | {"a.cc": []}))
    (repo / "native" / "fluidity" / "a.cc").write_text("// edited\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo) | {"a.cc": []})
    run(monkeypatch, repo, second)
    assert second.checked == ["a.cc"]


def test_a_changed_config_file_or_compiler_version_misses_the_cache(monkeypatch, repo):
    (repo / "native" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    (repo / ".clang-tidy").write_text("Checks: '-*,bugprone-*,misc-*'\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert sorted(second.checked) == ["a.cc", "b.cc", "c.cc"]
    monkeypatch.setattr(run_tidy, "clang_tidy_version_text", lambda: "LLVM version 99\n")
    third = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, third)
    assert sorted(third.checked) == ["a.cc", "b.cc", "c.cc"]


def test_a_failure_is_never_stored(monkeypatch, repo):
    (repo / "native" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    first = FakeRun(repo, dependencies_of(repo), failing=["b.cc"])
    results = run(monkeypatch, repo, first)
    assert [result.passed for result in results] == [True, False, True]
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert second.checked == ["b.cc"]


def test_no_cache_checks_everything_and_stores_nothing(monkeypatch, repo):
    (repo / "native" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    first = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, first, no_cache=True)
    assert not (repo / "out" / "tidy-cache").exists()
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    again = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, again, no_cache=True)
    assert sorted(again.checked) == ["a.cc", "b.cc", "c.cc"]


def test_results_and_printed_failures_are_sorted_by_path(monkeypatch, repo, capsys):
    monkeypatch.setattr(run_tidy.subprocess, "run", FakeRun(repo, {}, failing=["a.cc", "b.cc", "c.cc"]))
    files = [str(path.relative_to(repo)) for path in sources(repo)]
    exit_code = run_tidy.main(
        ["--build-folder", str(repo / "out" / "build" / "debug"), "--repository", str(repo), "--no-cache", *files]
    )
    printed = capsys.readouterr().out
    assert exit_code == 1
    positions = [printed.index(f"clang-tidy failed: {repo / 'native' / 'fluidity' / f'{name}.cc'}") for name in "abc"]
    assert positions == sorted(positions)
    assert "warning in a.cc" in printed


def test_a_header_alone_file_turns_the_include_check_off(tmp_path):
    header_alone = tmp_path / "header_check" / "fluidity" / "a.h.cc"
    assert run_tidy.tidy_arguments(header_alone) == ["--checks=-misc-include-cleaner"]
    assert run_tidy.tidy_arguments(tmp_path / "native" / "a.cc") == []


def test_the_dependency_command_drops_outputs_and_asks_for_every_header():
    command = run_tidy.dependency_command("cc -Ia -o x.o -MD -MT x.o -MF x.d -c main.cc")
    assert command == ["cc", "-Ia", "main.cc", "-M"]


def test_dependencies_are_read_from_a_makefile_rule():
    rule = "x.o: /a/main.cc \\\n  /a/with\\ space.h /b/other.h\n"
    assert run_tidy.parse_dependencies(rule) == ["/a/main.cc", "/a/with space.h", "/b/other.h"]


def test_a_header_outside_the_repository_counts_by_size_and_time_not_contents(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "system.h"
    outside.write_text("1234", encoding="utf-8")
    inside = repository / "own.h"
    inside.write_text("abcd", encoding="utf-8")
    assert ":4:" in run_tidy.header_fingerprint(outside, repository)
    assert ":4:" not in run_tidy.header_fingerprint(inside, repository)
    inside.write_text("abce", encoding="utf-8")
    first = run_tidy.header_fingerprint(inside, repository)
    inside.write_text("abcd", encoding="utf-8")
    assert run_tidy.header_fingerprint(inside, repository) != first


def test_config_files_of_the_folder_and_every_folder_above_apply(tmp_path):
    (tmp_path / "native" / "grid").mkdir(parents=True)
    (tmp_path / ".clang-tidy").write_text("a\n", encoding="utf-8")
    (tmp_path / "native" / "grid" / ".clang-tidy").write_text("b\n", encoding="utf-8")
    found = run_tidy.applicable_config_files(tmp_path / "native" / "grid" / "x.cc", tmp_path)
    assert found == [tmp_path.resolve() / "native" / "grid" / ".clang-tidy", tmp_path.resolve() / ".clang-tidy"]
