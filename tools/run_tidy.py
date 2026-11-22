"""Runs clang-tidy over source files in parallel and skips the files that already passed.

Usage: python tools/run_tidy.py --build-folder <folder> [--repository <folder>] [--cache-folder <folder>]
                                [--jobs <n>] [--no-cache] <file>...

Each file is one clang-tidy process, with as many at once as there are cores (--jobs). The output of each failing
file is printed after all files finished, sorted by path, so two runs print the same text. The exit code is 0 when
every file passed and 1 otherwise.

A header-alone file (a path with a `header_check` folder that ends in `.h.cc`) includes its header and uses nothing
from it, which is its purpose, so the include check is off for it. The header itself is still checked through
HeaderFilterRegex.

The cache holds the files that passed. A file is not checked again while nothing it depends on has changed: the cache
key is the SHA-256 of
  - the clang-tidy version text,
  - the contents of every .clang-tidy file that applies (the file's folder and each folder above it, up to the
    repository),
  - the arguments given to clang-tidy for the file,
  - every compile command of the file in <build folder>/compile_commands.json (clang-tidy checks the file once for
    each of them, and a shared source can be compiled by several targets with different options),
  - the contents of the file, and
  - the contents of every header that any of those commands includes, as the compiler lists them. A header outside
    the repository, or inside the repository's out/ folder (system and Conan headers, generated files), counts by
    path, size and modification time instead of contents.
Only a pass is stored: a file that fails is checked again on the next run. The key is computed again after
clang-tidy returned 0, from the same dependency list, and the pass is stored only when both keys are equal, so a
file edited while clang-tidy read it is never stored under the text that clang-tidy did not see. --no-cache ignores
the cache and stores nothing. A marker that was not used for 30 days is deleted at the end of a run.

A file whose dependency list cannot be made fails its check (it is never a pass and never checked unseen): the
compiler fails, prints no dependency, does not list the file itself, or lists a file that does not exist.

Assumption: the dependency list comes from the compiler of the compile command (the build compiler), not from the
preprocessor of the pinned clang-tidy. The tool environment has no clang of the version of clang-tidy (the pip
package holds only clang-tidy), so the two can differ in predefined macros (`__clang__`, `__GNUC__`, `_MSC_VER`,
`__apple_build_version__`, `__clang_analyzer__`) and in `__has_include` answers. The list is right while no
`#include` depends on such a macro; tools/check_rules.py fails an `#include` under a condition on one of the
four compiler macros (rule `compiler-include`), so the assumption is checked on every commit. On Windows the
compile command is an MSVC one, `shlex` mangles it and `cl` has no `-M`, so the listing fails and every file fails
unless --no-cache is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_FOLDER = Path("out") / "tidy-cache"
# A cache marker that no run used for this long is deleted.
CACHE_MARKER_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
HEADER_CHECK_FOLDER = "header_check"
HEADER_ALONE_CHECKS = "--checks=-misc-include-cleaner"
CLANG_TIDY_CONFIG_NAME = ".clang-tidy"
# Options of a compile command that name an output or a dependency file, and take one argument.
_OPTIONS_WITH_ARGUMENT = frozenset({"-o", "-MF", "-MT", "-MQ"})
# Options of a compile command that choose what the compiler does, which the dependency listing replaces.
_OPTIONS_TO_DROP = frozenset({"-c", "-MD", "-MMD", "-M", "-MM", "-MG", "-MP", "-E", "-S"})


@dataclass(frozen=True)
class TidyResult:
    """What one file's check gave: `passed`, and the text clang-tidy printed."""

    path: str
    passed: bool
    output: str


def tidy_arguments(path: Path) -> list[str]:
    """The arguments clang-tidy gets for `path`, besides the build folder and the file itself."""
    if HEADER_CHECK_FOLDER in path.parts and path.name.endswith(".h.cc"):
        return [HEADER_ALONE_CHECKS]
    return []


def dependency_command(compile_command: str) -> list[str]:
    """The compile command turned into one that prints the files the file depends on, as a Makefile rule: the output
    and dependency-file options are dropped and `-M` (system headers included) is added."""
    kept: list[str] = []
    words = iter(shlex.split(compile_command))
    for word in words:
        if word in _OPTIONS_WITH_ARGUMENT:
            next(words, None)
        elif word in _OPTIONS_TO_DROP:
            continue
        else:
            kept.append(word)
    return [*kept, "-M"]


def parse_dependencies(makefile_rule: str) -> list[str]:
    """The files of a Makefile rule `target: dependency ...` as the compiler prints it: a backslash and a newline
    continue the line, and a backslash and a space is a space inside a name, and a backslash and `#` is a `#`. The target is not returned."""
    joined = makefile_rule.replace("\\\r\n", " ").replace("\\\n", " ")
    _, _, after_target = joined.partition(": ")
    names = re.split(r"(?<!\\)\s+", after_target.strip())
    return sorted({name.replace("\\ ", " ").replace("\\#", "#") for name in names if name})


def applicable_config_files(path: Path, repository: Path) -> list[Path]:
    """The .clang-tidy files of the folder of `path` and each folder above it, up to `repository`, nearest first.
    A path outside the repository (a generated header-alone file) is looked up from the repository root."""
    absolute = path.resolve()
    root = repository.resolve()
    folder = absolute.parent
    if root not in (folder, *folder.parents):
        folder = root
    found = []
    while True:
        candidate = folder / CLANG_TIDY_CONFIG_NAME
        if candidate.is_file():
            found.append(candidate)
        if folder == root:
            return found
        folder = folder.parent


def header_fingerprint(path: Path, repository: Path) -> str:
    """What a dependency contributes to the key: its contents when it is a source of the repository, and its path,
    size and modification time when it is a system, package or generated file."""
    resolved = path.resolve()
    root = repository.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        relative = None
    try:
        if relative is not None and relative.parts[:1] != ("out",):
            return f"{resolved}:{hashlib.sha256(resolved.read_bytes()).hexdigest()}"
        status = resolved.stat()
    except OSError:
        return f"{resolved}:missing"
    return f"{resolved}:{status.st_size}:{status.st_mtime_ns}"


def cache_key(
    version_text: str,
    config_files: Sequence[Path],
    tidy_options: Sequence[str],
    compile_commands: Sequence[str],
    source: Path,
    dependencies: Sequence[Path],
    repository: Path,
) -> str:
    """The SHA-256 key of one file's check, over everything the module docstring lists."""
    digest = hashlib.sha256()
    parts = [
        version_text,
        *(f"config:{config.resolve()}:{hashlib.sha256(config.read_bytes()).hexdigest()}" for config in config_files),
        "options:" + "\0".join(tidy_options),
        *(f"command:{command}" for command in compile_commands),
        "source:" + header_fingerprint(source, repository),
        *(f"dependency:{header_fingerprint(dependency, repository)}" for dependency in dependencies),
    ]
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_compile_commands(build_folder: Path) -> dict[Path, list[tuple[str, Path]]]:
    """Every compile command and working folder of each file of <build folder>/compile_commands.json, by resolved
    path, in the order of the file. A source that several targets compile has one command for each."""
    entries = json.loads((build_folder / "compile_commands.json").read_text(encoding="utf-8"))
    found: dict[Path, list[tuple[str, Path]]] = {}
    for entry in entries:
        directory = Path(entry["directory"])
        command = entry.get("command") or shlex.join(entry["arguments"])
        commands = found.setdefault(Path(directory, entry["file"]).resolve(), [])
        if (command, directory) not in commands:
            commands.append((command, directory))
    return found


class DependencyListError(Exception):
    """The files that a compile command depends on could not be listed."""


def read_dependencies(compile_command: str, directory: Path, source: Path) -> list[Path]:
    """The files the compiler lists for the command. Raises DependencyListError when the compiler fails, when its
    output does not name `source` itself (an empty or foreign list), or when a listed file does not exist."""
    completed = subprocess.run(
        dependency_command(compile_command),
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DependencyListError(f"the compiler exited with {completed.returncode}")
    dependencies = [Path(directory, name) for name in parse_dependencies(completed.stdout)]
    if source.resolve() not in {dependency.resolve() for dependency in dependencies}:
        raise DependencyListError("the compiler's list does not name the file itself")
    missing = [str(dependency) for dependency in dependencies if not dependency.exists()]
    if missing:
        raise DependencyListError("a listed file does not exist: " + ", ".join(missing))
    return dependencies


def clang_tidy_version_text() -> str:
    """The text `clang-tidy --version` prints."""
    completed = subprocess.run(
        ["clang-tidy", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    return completed.stdout or ""


def check_file(
    path: Path,
    build_folder: Path,
    repository: Path,
    cache_folder: Path | None,
    version_text: str,
    commands: dict[Path, list[tuple[str, Path]]],
) -> TidyResult:
    """Checks one file with clang-tidy, unless the cache holds a pass for its key, and stores a pass. With the cache
    on, a file whose dependencies cannot be listed fails without running clang-tidy."""
    options = tidy_arguments(path)
    marker: Path | None = None
    key = ""
    entries = commands.get(path.resolve())
    compile_commands = [command for command, _ in entries or []]
    config_files = applicable_config_files(path, repository)
    dependencies: list[Path] = []
    if cache_folder is not None and entries:
        union: set[Path] = set()
        try:
            for command, directory in entries:
                union.update(read_dependencies(command, directory, path))
        except DependencyListError as error:
            return TidyResult(str(path), False, f"cannot list the dependencies of {path}: {error}\n")
        dependencies = sorted(union)
        key = cache_key(version_text, config_files, options, compile_commands, path, dependencies, repository)
        marker = cache_folder / key
        if marker.exists():
            marker.touch()
            return TidyResult(str(path), True, "")
    completed = subprocess.run(
        ["clang-tidy", "-p", str(build_folder), "--quiet", *options, str(path)],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    passed = completed.returncode == 0
    if passed and marker is not None:
        # A file may change while clang-tidy reads it. Store the pass only for text that is still the text of the key.
        key_after = cache_key(version_text, config_files, options, compile_commands, path, dependencies, repository)
        if key_after == key:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
    return TidyResult(str(path), passed, completed.stdout or "")


def prune_cache(cache_folder: Path, now: float) -> None:
    """Deletes the markers of `cache_folder` that were last used more than CACHE_MARKER_MAX_AGE_SECONDS before `now`."""
    if not cache_folder.is_dir():
        return
    for marker in cache_folder.iterdir():
        try:
            if now - marker.stat().st_mtime > CACHE_MARKER_MAX_AGE_SECONDS:
                marker.unlink()
        except OSError:
            continue


def run_tidy(
    files: Sequence[Path],
    build_folder: Path,
    repository: Path,
    cache_folder: Path | None,
    jobs: int,
) -> list[TidyResult]:
    """Checks every file, `jobs` at a time, and returns the results sorted by path. A cache folder of None ignores the
    cache and stores nothing."""
    version_text = clang_tidy_version_text()
    commands = read_compile_commands(build_folder)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(
            pool.map(
                lambda path: check_file(path, build_folder, repository, cache_folder, version_text, commands),
                files,
            )
        )
    if cache_folder is not None:
        prune_cache(cache_folder, time.time())
    return sorted(results, key=lambda result: result.path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run clang-tidy in parallel and skip files that already passed.")
    parser.add_argument("--build-folder", type=Path, required=True, help="folder of compile_commands.json")
    parser.add_argument("--repository", type=Path, default=Path.cwd(), help="repository root (default: the cwd)")
    parser.add_argument("--cache-folder", type=Path, default=None, help=f"default: <repository>/{DEFAULT_CACHE_FOLDER}")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processes at once (default: cores)")
    parser.add_argument("--no-cache", action="store_true", help="check every file and store nothing")
    parser.add_argument("files", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    cache_folder = None if arguments.no_cache else arguments.cache_folder or arguments.repository / DEFAULT_CACHE_FOLDER
    files = [arguments.repository / path for path in arguments.files]
    results = run_tidy(files, arguments.build_folder, arguments.repository, cache_folder, arguments.jobs)
    failed = [result for result in results if not result.passed]
    for result in failed:
        print(f"clang-tidy failed: {result.path}")
        print(result.output, end="" if result.output.endswith("\n") else "\n")
    print(f"clang-tidy: {len(results) - len(failed)} of {len(results)} files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
