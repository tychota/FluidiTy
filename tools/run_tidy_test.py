"""Tests for tools/run_tidy.py. The compiler and clang-tidy are a fake `subprocess.run`, so nothing is compiled."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import run_tidy

COMPILER = "fake-cc"
VERSION_TEXT = "LLVM version 23.1.4\n"


class FakeRun:
    """Stands in for subprocess.run. A `fake-cc ... -M` command prints the dependencies chosen for its file; a
    clang-tidy command is recorded and fails for the files in `failing`."""

    def __init__(
        self,
        repo: Path,
        dependencies: dict[str, list[str]],
        failing=(),
        on_tidy=None,
        variant_dependencies: dict[str, list[str]] | None = None,
        raw_rules: dict[str, str] | None = None,
        listing_exit_code: int = 0,
        implicit_folders: tuple[str, ...] = (),
        compiler_version_text: str = "fake-cc version 1\n",
    ):
        self.repo = repo
        self.dependencies = dependencies
        self.failing = set(failing)
        # Called with the file name while clang-tidy "reads" it, to change files in the middle of a check.
        self.on_tidy = on_tidy
        # Extra dependencies of a command that carries -DVARIANT, the compile command of a second target.
        self.variant_dependencies = variant_dependencies or {}
        # The exact text the compiler prints for a file name, in place of a well-formed rule.
        self.raw_rules = raw_rules or {}
        self.listing_exit_code = listing_exit_code
        # The folders the compiler says it searches by itself, in the text of `-E -x c++ -v`.
        self.implicit_folders = implicit_folders
        self.asked_for_folders = 0
        # What the compiler prints for `--version`.
        self.compiler_version_text = compiler_version_text
        self.checked: list[str] = []
        # The files whose headers the compiler was asked to list.
        self.listed: list[str] = []

    def __call__(self, command, **kwargs):
        if command[:2] == ["clang-tidy", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout=VERSION_TEXT)
        if command == [COMPILER, "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout=self.compiler_version_text)
        if command[0] == COMPILER and "-v" in command:
            self.asked_for_folders += 1
            lines = ["#include <...> search starts here:", *(f" {folder}" for folder in self.implicit_folders)]
            text = "\n".join([*lines, "End of search list."]) + "\n"
            return subprocess.CompletedProcess(command, 0, stdout="", stderr=text)
        if command[0] == COMPILER:
            source = command[command.index("--source") + 1]
            name = Path(source).name
            self.listed.append(name)
            if self.listing_exit_code != 0:
                return subprocess.CompletedProcess(command, self.listing_exit_code, stdout="")
            if name in self.raw_rules:
                return subprocess.CompletedProcess(command, 0, stdout=self.raw_rules[name])
            extra = self.variant_dependencies.get(name, []) if "-DVARIANT" in command else []
            names = " ".join([source, *self.dependencies.get(name, []), *extra])
            return subprocess.CompletedProcess(command, 0, stdout=f"x.o: {names}\n")
        assert command[0] == "clang-tidy", command
        name = Path(command[-1]).name
        self.checked.append(name)
        if self.on_tidy is not None:
            self.on_tidy(name)
        code = 1 if name in self.failing else 0
        return subprocess.CompletedProcess(command, code, stdout=f"warning in {name}\n" if code else "")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "fluidity").mkdir(parents=True)
    (tmp_path / "out" / "build" / "debug").mkdir(parents=True)
    (tmp_path / ".clang-tidy").write_text("Checks: '-*,bugprone-*'\n", encoding="utf-8")
    for name in ("a", "b", "c"):
        (tmp_path / "src" / "fluidity" / f"{name}.cc").write_text(f'#include "{name}.h"\n', encoding="utf-8")
        (tmp_path / "src" / "fluidity" / f"{name}.h").write_text("#pragma once\n", encoding="utf-8")
    entries = [
        {
            "directory": str(tmp_path / "out" / "build" / "debug"),
            "command": f"{COMPILER} -I{tmp_path} --source {tmp_path / 'src' / 'fluidity' / (name + '.cc')} -o x.o -c",
            "file": str(tmp_path / "src" / "fluidity" / f"{name}.cc"),
        }
        for name in ("a", "b", "c")
    ]
    (tmp_path / "out" / "build" / "debug" / "compile_commands.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


def sources(repo: Path) -> list[Path]:
    return [repo / "src" / "fluidity" / f"{name}.cc" for name in ("c", "a", "b")]


def run(monkeypatch, repo: Path, fake: FakeRun, *, no_cache: bool = False) -> list[run_tidy.TidyResult]:
    monkeypatch.setattr(run_tidy.subprocess, "run", fake)
    cache = None if no_cache else repo / "out" / "tidy-cache"
    return run_tidy.run_tidy(sources(repo), repo / "out" / "build" / "debug", repo, cache, jobs=4)


def dependencies_of(repo: Path) -> dict[str, list[str]]:
    folder = repo / "src" / "fluidity"
    return {f"{name}.cc": [str(folder / f"{name}.h"), str(folder / "shared.h")] for name in ("a", "b", "c")} | {
        "c.cc": [str(folder / "c.h")]
    }


def test_files_that_passed_are_not_checked_again(monkeypatch, repo):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    first = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, first)
    assert sorted(first.checked) == ["a.cc", "b.cc", "c.cc"]
    second = FakeRun(repo, dependencies_of(repo))
    results = run(monkeypatch, repo, second)
    assert second.checked == []
    assert all(result.passed for result in results)


def test_a_changed_header_makes_its_includers_miss_the_cache(monkeypatch, repo):
    shared = repo / "src" / "fluidity" / "shared.h"
    shared.write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    shared.write_text("// shared, changed\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert sorted(second.checked) == ["a.cc", "b.cc"]


def test_a_changed_unrelated_file_does_not_miss_the_cache(monkeypatch, repo):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    (repo / "src" / "fluidity" / "unrelated.h").write_text("// new\n", encoding="utf-8")
    (repo / "src" / "fluidity" / "c.h").write_text("#pragma once\n// changed\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert second.checked == ["c.cc"]


def test_a_changed_source_file_is_checked_again(monkeypatch, repo):
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo) | {"a.cc": []}))
    (repo / "src" / "fluidity" / "a.cc").write_text("// edited\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo) | {"a.cc": []})
    run(monkeypatch, repo, second)
    assert second.checked == ["a.cc"]


def test_a_changed_config_file_or_compiler_version_misses_the_cache(monkeypatch, repo):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
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
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    first = FakeRun(repo, dependencies_of(repo), failing=["b.cc"])
    results = run(monkeypatch, repo, first)
    assert [result.passed for result in results] == [True, False, True]
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert second.checked == ["b.cc"]


def test_no_cache_checks_everything_and_stores_nothing(monkeypatch, repo):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
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
    positions = [printed.index(f"clang-tidy failed: {repo / 'src' / 'fluidity' / f'{name}.cc'}") for name in "abc"]
    assert positions == sorted(positions)
    assert "warning in a.cc" in printed


def test_a_header_alone_file_turns_the_include_check_off(tmp_path):
    header_alone = tmp_path / "header_check" / "fluidity" / "a.h.cc"
    assert run_tidy.tidy_arguments(header_alone) == ["--checks=-misc-include-cleaner"]
    assert run_tidy.tidy_arguments(tmp_path / "src" / "a.cc") == []


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
    (tmp_path / "src" / "grid").mkdir(parents=True)
    (tmp_path / ".clang-tidy").write_text("a\n", encoding="utf-8")
    (tmp_path / "src" / "grid" / ".clang-tidy").write_text("b\n", encoding="utf-8")
    found = run_tidy.applicable_config_files(tmp_path / "src" / "grid" / "x.cc", tmp_path)
    assert found == [tmp_path.resolve() / "src" / "grid" / ".clang-tidy", tmp_path.resolve() / ".clang-tidy"]


def test_a_file_edited_while_clang_tidy_reads_it_is_not_stored_under_the_old_key(monkeypatch, repo):
    shared = repo / "src" / "fluidity" / "shared.h"
    shared.write_text("// shared, broken\n", encoding="utf-8")
    first = FakeRun(
        repo,
        dependencies_of(repo),
        on_tidy=lambda name: shared.write_text("// shared, fixed\n", encoding="utf-8"),
    )
    run(monkeypatch, repo, first)
    # The fix is undone: the broken text of the first key is back, and no pass was stored for it.
    shared.write_text("// shared, broken\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert sorted(second.checked) == ["a.cc", "b.cc"]


def test_a_source_with_two_compile_commands_depends_on_the_headers_of_both(monkeypatch, repo):
    folder = repo / "src" / "fluidity"
    (folder / "shared.h").write_text("// shared\n", encoding="utf-8")
    (folder / "extra.h").write_text("// only with the define\n", encoding="utf-8")
    database = repo / "out" / "build" / "debug" / "compile_commands.json"
    entries = json.loads(database.read_text(encoding="utf-8"))
    variant = dict(entries[0], command=entries[0]["command"] + " -DVARIANT")
    # The variant command comes first and the plain one last: the key must not follow the last entry only.
    database.write_text(json.dumps([variant, *entries]), encoding="utf-8")
    variant_dependencies = {"a.cc": [str(folder / "extra.h")]}
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo), variant_dependencies=variant_dependencies))
    (folder / "extra.h").write_text("// only with the define, changed\n", encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo), variant_dependencies=variant_dependencies)
    run(monkeypatch, repo, second)
    assert second.checked == ["a.cc"]


def test_a_changed_compile_command_misses_the_cache(monkeypatch, repo):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    database = repo / "out" / "build" / "debug" / "compile_commands.json"
    entries = json.loads(database.read_text(encoding="utf-8"))
    entries[0]["command"] += " -DOTHER"
    database.write_text(json.dumps(entries), encoding="utf-8")
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert second.checked == ["a.cc"]


def test_the_tidy_options_of_a_header_alone_file_are_part_of_its_key(tmp_path):
    source = tmp_path / "a.h.cc"
    source.write_text("\n", encoding="utf-8")
    plain = run_tidy.cache_key("v", [], [], ["cc"], source, [], tmp_path)
    header_alone = run_tidy.cache_key("v", [], [run_tidy.HEADER_ALONE_CHECKS], ["cc"], source, [], tmp_path)
    assert plain != header_alone


@pytest.mark.parametrize(
    "rule_for_a",
    ["", "x.o:\n", "x.o: /elsewhere/other.cc\n", "x.o: {source} {repo}/src/fluidity/not_there.h\n"],
    ids=["empty", "no-files", "not-the-source", "missing-file"],
)
def test_an_unusable_dependency_list_fails_the_file_and_stores_nothing(monkeypatch, repo, rule_for_a):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    source = repo / "src" / "fluidity" / "a.cc"
    rule = rule_for_a.format(source=source, repo=repo)
    fake = FakeRun(repo, dependencies_of(repo), raw_rules={"a.cc": rule})
    results = run(monkeypatch, repo, fake)
    assert [result.passed for result in results] == [False, True, True]
    assert "cannot list the dependencies" in results[0].output
    assert "a.cc" not in fake.checked
    again = FakeRun(repo, dependencies_of(repo), raw_rules={"a.cc": rule})
    run(monkeypatch, repo, again)
    assert "a.cc" not in again.checked


def test_a_failing_dependency_listing_fails_the_file_and_stores_nothing(monkeypatch, repo):
    fake = FakeRun(repo, dependencies_of(repo), listing_exit_code=1)
    results = run(monkeypatch, repo, fake)
    assert [result.passed for result in results] == [False, False, False]
    assert fake.checked == []
    assert not (repo / "out" / "tidy-cache").exists()


def test_without_the_cache_the_dependencies_are_not_listed(monkeypatch, repo):
    fake = FakeRun(repo, dependencies_of(repo), listing_exit_code=1)
    results = run(monkeypatch, repo, fake, no_cache=True)
    assert all(result.passed for result in results)


def test_an_escaped_hash_in_a_dependency_is_a_hash():
    assert run_tidy.parse_dependencies("x.o: /a/main.cc /a/dir\\#1/b.h\n") == ["/a/dir#1/b.h", "/a/main.cc"]


def test_a_marker_unused_for_a_month_is_deleted_and_a_recent_one_stays(tmp_path):
    old = tmp_path / "old"
    recent = tmp_path / "recent"
    old.touch()
    recent.touch()
    now = 100 * 24 * 60 * 60.0
    os.utime(old, (now - 31 * 24 * 60 * 60, now - 31 * 24 * 60 * 60))
    os.utime(recent, (now - 1 * 24 * 60 * 60, now - 1 * 24 * 60 * 60))
    run_tidy.prune_cache(tmp_path, now)
    assert not old.exists()
    assert recent.exists()


# The stat record: a file that passed is taken as passed without listing its headers -------------------------

OLD_NS = 1_000_000_000 * 10**9


def age_files(repo: Path) -> None:
    """Gives every file of the repository, but the cache, a modification time long before now: the stat record is not
    stored for a file modified in the last seconds."""
    for path in repo.rglob("*"):
        if path.is_file() and "tidy-cache" not in path.parts:
            os.utime(path, ns=(OLD_NS, OLD_NS))


def touch_to(path: Path, seconds_after_old: int) -> None:
    """Sets a modification time that differs from the aged one and is still old."""
    os.utime(path, ns=(OLD_NS + seconds_after_old * 10**9, OLD_NS + seconds_after_old * 10**9))


def records(repo: Path) -> list[Path]:
    return sorted((repo / "out" / "tidy-cache").glob(f"{run_tidy.STAT_RECORD_PREFIX}*"))


@pytest.fixture
def stored(monkeypatch, repo):
    """A repository where every file passed twice: the first run stored the passes, the second one, with aged files,
    stored the stat records. The shared header is listed by a and b."""
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    age_files(repo)
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    assert len(records(repo)) == 3
    return repo


def test_a_file_that_passed_is_a_hit_without_listing_its_headers(monkeypatch, stored):
    third = FakeRun(stored, dependencies_of(stored))
    results = run(monkeypatch, stored, third)
    assert third.listed == []
    assert third.checked == []
    assert all(result.passed for result in results)


def test_a_record_is_not_stored_for_files_modified_a_moment_ago(monkeypatch, repo):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo)))
    assert records(repo) == []
    third = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, third)
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]


def test_a_changed_header_misses_the_record_and_its_includers_are_checked(monkeypatch, stored):
    (stored / "src" / "fluidity" / "shared.h").write_text("// shared, changed\n", encoding="utf-8")
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert sorted(third.checked) == ["a.cc", "b.cc"]
    assert "c.cc" not in third.listed or third.checked != ["c.cc"]


def test_a_changed_header_of_the_same_size_and_a_later_time_is_a_miss(monkeypatch, stored):
    shared = stored / "src" / "fluidity" / "shared.h"
    shared.write_text("// sharee\n", encoding="utf-8")
    assert shared.stat().st_size == len("// shared\n")
    touch_to(shared, 5)
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert sorted(third.checked) == ["a.cc", "b.cc"]


def test_a_touched_but_identical_header_is_listed_again_and_still_a_pass_and_is_recorded_again(monkeypatch, stored):
    touch_to(stored / "src" / "fluidity" / "shared.h", 7)
    third = FakeRun(stored, dependencies_of(stored))
    results = run(monkeypatch, stored, third)
    assert sorted(third.listed) == ["a.cc", "b.cc"]
    assert third.checked == []
    assert all(result.passed for result in results)
    age_files(stored)
    run(monkeypatch, stored, FakeRun(stored, dependencies_of(stored)))
    fifth = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, fifth)
    assert fifth.listed == []


def test_a_changed_source_file_misses_the_record(monkeypatch, stored):
    (stored / "src" / "fluidity" / "c.cc").write_text("// edited\n", encoding="utf-8")
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert third.checked == ["c.cc"]
    assert third.listed == ["c.cc"]


def test_a_changed_compile_flag_misses_the_record(monkeypatch, stored):
    build = stored / "out" / "build" / "debug"
    entries = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    entries[0]["command"] += " -DOTHER_FLAG"
    (build / "compile_commands.json").write_text(json.dumps(entries), encoding="utf-8")
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert third.checked == ["a.cc"]
    assert third.listed == ["a.cc"]


def test_a_changed_config_file_of_the_same_size_misses_the_record(monkeypatch, stored):
    config = stored / ".clang-tidy"
    config.write_text("Checks: '-*,bugprone-?'\n", encoding="utf-8")
    assert config.stat().st_size == len("Checks: '-*,bugprone-*'\n")
    touch_to(config, 9)
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert sorted(third.checked) == ["a.cc", "b.cc", "c.cc"]


def test_a_new_config_file_in_a_folder_of_the_source_misses_the_record(monkeypatch, stored):
    (stored / "src" / "fluidity" / ".clang-tidy").write_text("Checks: '-*'\n", encoding="utf-8")
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert sorted(third.checked) == ["a.cc", "b.cc", "c.cc"]


def test_a_header_added_where_the_compiler_looks_misses_the_record(monkeypatch, stored):
    (stored / "src" / "fluidity" / "new_header.h").write_text("// new\n", encoding="utf-8")
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]
    assert third.checked == []


def test_a_header_removed_where_the_compiler_looks_misses_the_record(monkeypatch, stored):
    (stored / "src" / "fluidity" / "unused.h").write_text("// unused\n", encoding="utf-8")
    age_files(stored)
    run(monkeypatch, stored, FakeRun(stored, dependencies_of(stored)))
    run(monkeypatch, stored, FakeRun(stored, dependencies_of(stored)))
    (stored / "src" / "fluidity" / "unused.h").unlink()
    after = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, after)
    assert sorted(after.listed) == ["a.cc", "b.cc", "c.cc"]


def test_a_new_clang_tidy_version_misses_the_record(monkeypatch, stored):
    monkeypatch.setattr(run_tidy, "clang_tidy_version_text", lambda: "LLVM version 99\n")
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert sorted(third.checked) == ["a.cc", "b.cc", "c.cc"]


def test_a_new_compiler_version_text_misses_the_record(monkeypatch, stored):
    hit = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, hit)
    assert hit.listed == []
    third = FakeRun(stored, dependencies_of(stored), compiler_version_text="fake-cc version 2\n")
    run(monkeypatch, stored, third)
    # The record no longer holds, so the headers are listed again; the key is the same, so nothing is checked again.
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]
    assert third.checked == []


def test_a_replaced_clang_tidy_binary_with_the_same_version_text_misses_the_record(monkeypatch, tmp_path, repo):
    binary = tmp_path / "bin" / "clang-tidy"
    binary.parent.mkdir()
    binary.write_text("one\n", encoding="utf-8")
    monkeypatch.setattr(run_tidy.shutil, "which", lambda name: str(binary) if name == "clang-tidy" else None)
    make_old_records(monkeypatch, repo)
    second = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, second)
    assert second.checked == [] and second.listed == []
    binary.write_text("two!\n", encoding="utf-8")
    third = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, third)
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]


def test_a_stamp_holds_the_size_the_times_and_the_inode(tmp_path):
    header = tmp_path / "a.h"
    header.write_text("// one\n", encoding="utf-8")
    status = header.stat()
    assert run_tidy.file_stamp(header) == [status.st_size, status.st_mtime_ns, status.st_ino, status.st_ctime_ns]


def test_a_copy_that_keeps_the_size_and_the_modification_time_changes_the_stamp(tmp_path):
    header = tmp_path / "a.h"
    header.write_text("// one\n", encoding="utf-8")
    before = run_tidy.file_stamp(header)
    copy = tmp_path / "copy.h"
    shutil.copy2(header, copy)
    copy.replace(header)
    after = run_tidy.file_stamp(header)
    assert after[:2] == before[:2]
    assert after != before


def test_a_record_does_not_pass_a_file_whose_marker_is_gone(monkeypatch, stored):
    for marker in (stored / "out" / "tidy-cache").iterdir():
        if not marker.name.startswith(run_tidy.STAT_RECORD_PREFIX):
            marker.unlink()
    third = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, third)
    assert sorted(third.checked) == ["a.cc", "b.cc", "c.cc"]


def test_an_unreadable_record_is_ignored(monkeypatch, stored):
    for record in records(stored):
        record.write_text("not json", encoding="utf-8")
    third = FakeRun(stored, dependencies_of(stored))
    results = run(monkeypatch, stored, third)
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]
    assert third.checked == []
    assert all(result.passed for result in results)


def test_a_failure_gets_no_record_and_is_checked_on_every_run(monkeypatch, repo):
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    age_files(repo)
    for _ in range(2):
        fake = FakeRun(repo, dependencies_of(repo), failing=["b.cc"])
        results = run(monkeypatch, repo, fake)
        assert [result.passed for result in results] == [True, False, True]
    assert len(records(repo)) == 2
    third = FakeRun(repo, dependencies_of(repo), failing=["b.cc"])
    run(monkeypatch, repo, third)
    assert third.checked == ["b.cc"]


def test_a_file_that_changes_while_clang_tidy_reads_it_gets_no_record(monkeypatch, repo):
    shared = repo / "src" / "fluidity" / "shared.h"
    shared.write_text("// shared\n", encoding="utf-8")
    age_files(repo)

    def change_time(name):
        if name == "a.cc":
            touch_to(shared, 3)

    run(monkeypatch, repo, FakeRun(repo, dependencies_of(repo), on_tidy=change_time))
    source = repo / "src" / "fluidity" / "a.cc"
    assert not run_tidy.stat_record_path(repo / "out" / "tidy-cache", source, []).exists()
    assert run_tidy.stat_record_path(repo / "out" / "tidy-cache", repo / "src" / "fluidity" / "c.cc", []).exists()
    third = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, third)
    assert "a.cc" in third.listed


def test_no_cache_reads_and_stores_no_record(monkeypatch, stored):
    before = {record: record.read_text(encoding="utf-8") for record in records(stored)}
    fake = FakeRun(stored, dependencies_of(stored))
    run(monkeypatch, stored, fake, no_cache=True)
    assert sorted(fake.checked) == ["a.cc", "b.cc", "c.cc"]
    assert fake.listed == []
    assert {record: record.read_text(encoding="utf-8") for record in records(stored)} == before


def test_a_record_that_nothing_used_for_a_month_is_deleted(tmp_path):
    cache = tmp_path / "tidy-cache"
    cache.mkdir()
    old = cache / f"{run_tidy.STAT_RECORD_PREFIX}abc"
    old.write_text("{}", encoding="utf-8")
    os.utime(old, (1000, 1000))
    run_tidy.prune_cache(cache, now=1000 + run_tidy.CACHE_MARKER_MAX_AGE_SECONDS + 1)
    assert not old.exists()


def make_old_records(monkeypatch, repo, extra_dependencies=(), implicit_folders=()):
    """Runs every file twice with aged files so that records are stored, and returns what the next run needs."""
    (repo / "src" / "fluidity" / "shared.h").write_text("// shared\n", encoding="utf-8")
    dependencies = dependencies_of(repo)
    dependencies["a.cc"] = [*dependencies["a.cc"], *map(str, extra_dependencies)]
    run(monkeypatch, repo, FakeRun(repo, dependencies, implicit_folders=implicit_folders))
    age_files(repo)
    run(monkeypatch, repo, FakeRun(repo, dependencies, implicit_folders=implicit_folders))
    return dependencies


@pytest.mark.parametrize("variable", ["CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "SDKROOT"])
def test_a_changed_include_environment_variable_misses_the_record(monkeypatch, repo, variable):
    dependencies = make_old_records(monkeypatch, repo)
    assert len(records(repo)) == 3
    monkeypatch.setenv(variable, str(repo / "elsewhere"))
    third = FakeRun(repo, dependencies)
    run(monkeypatch, repo, third)
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]


def test_a_header_added_in_a_folder_of_cpath_misses_the_record(monkeypatch, repo):
    folder = repo / "extra"
    folder.mkdir()
    monkeypatch.setenv("CPATH", str(folder))
    dependencies = make_old_records(monkeypatch, repo)
    (folder / "new.h").write_text("// new\n", encoding="utf-8")
    third = FakeRun(repo, dependencies)
    run(monkeypatch, repo, third)
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]


def test_a_header_added_in_a_folder_the_compiler_searches_by_itself_misses_the_record(monkeypatch, repo):
    folder = repo / "sysinclude"
    folder.mkdir()
    dependencies = make_old_records(monkeypatch, repo, implicit_folders=(str(folder),))
    (folder / "new.h").write_text("// new\n", encoding="utf-8")
    third = FakeRun(repo, dependencies, implicit_folders=(str(folder),))
    run(monkeypatch, repo, third)
    assert sorted(third.listed) == ["a.cc", "b.cc", "c.cc"]


def test_the_folders_a_compiler_searches_by_itself_are_asked_once_per_run(monkeypatch, repo):
    fake = FakeRun(repo, dependencies_of(repo))
    run(monkeypatch, repo, fake)
    assert fake.asked_for_folders == 1


def test_the_folders_of_a_compiler_are_read_from_its_search_list(monkeypatch):
    text = "ignored\n#include <...> search starts here:\n /usr/include\n /sdk/Frameworks (framework directory)\nEnd of search list.\n"
    monkeypatch.setattr(
        run_tidy.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stderr=text)
    )
    assert run_tidy.compiler_search_folders("cc", ()) == ("/usr/include", "/sdk/Frameworks")


def test_the_options_that_move_the_search_folders_are_kept():
    words = ["-I/x", "-isysroot", "/sdk", "-stdlib=libc++", "-O2", "-std=c++23", "-o", "x.o", "-target", "arm64-apple"]
    assert run_tidy.search_options(words) == ("-isysroot", "/sdk", "-stdlib=libc++", "-std=c++23", "-target", "arm64-apple")


def test_a_record_is_not_stored_for_a_file_that_lists_a_generated_header(monkeypatch, repo):
    generated = repo / "out" / "build" / "debug" / "gen" / "fluidity" / "config.h"
    generated.parent.mkdir(parents=True)
    generated.write_text("// generated\n", encoding="utf-8")
    make_old_records(monkeypatch, repo, extra_dependencies=[generated])
    assert len(records(repo)) == 2


def test_a_record_is_stored_for_a_file_that_lists_a_conan_package_header(monkeypatch, repo):
    package_header = repo / "out" / "conan-home" / "p" / "b" / "gtest1" / "p" / "include" / "gtest.h"
    package_header.parent.mkdir(parents=True)
    package_header.write_text("// package\n", encoding="utf-8")
    make_old_records(monkeypatch, repo, extra_dependencies=[package_header])
    assert len(records(repo)) == 3


def test_a_stamp_is_read_again_each_time_it_is_asked(tmp_path):
    memo = run_tidy.FileSystemMemo(tmp_path)
    header = tmp_path / "a.h"
    header.write_text("// one\n", encoding="utf-8")
    first = memo.stamp(str(header))
    header.write_text("// a longer text\n", encoding="utf-8")
    assert memo.stamp(str(header)) != first
