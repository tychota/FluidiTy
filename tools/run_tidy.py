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

The stat record. Listing the headers means running the compiler's preprocessor on the file, which is most of the time
of a run in which nothing changed. After a pass the run therefore also stores, for the file, the list of its
dependencies with what the file system says about everything the key reads: the (size, modification time in
nanoseconds, the inode number and the change time) of the file, of each listed header, of each `.clang-tidy` file that
applies, of the compiler and of the clang-tidy binary, and the `--version` text of the compiler; the
compile commands and the tidy options and version text; the values of the environment variables that add header
folders or a sysroot (TIDY_ENVIRONMENT_VARIABLES); and a fingerprint of the folder layout, which is the sorted names
in each folder where the compiler looks (every folder under an include folder of the repository, outside its out/
folder; the folder alone for any other include folder, for each folder of CPATH, C_INCLUDE_PATH and
CPLUS_INCLUDE_PATH, and for each folder that the compiler searches without being told, which it prints for
`-E -x c++ -v`, asked once per run for each compiler and its sysroot options) and in the folder of each listed
header, so a header added or removed where it would be found first changes the record. The stamps are read from the
file system each time, never kept for the run, so an edit made while the run goes on is seen by the files checked
after it; only the folder layouts and the compiler's folders are asked once. The record is stored only after a pass, only
when the key before and after clang-tidy are equal and the record taken before and after are equal, and only
when no listed file was modified in the last STAT_RACY_WINDOW_NS before the record was taken, and only when no header
listed besides the file itself is a generated file of the build, that is, under the repository's out/ folder but not
in the Conan packages (out/conan-home): a folder of generated headers is not searched recursively, so a new header
there would go unseen. A later run takes the
file as passed without running the compiler when all of these are equal to the stored ones and the marker of the
stored key still exists. Any difference, an unreadable record, or a missing marker makes the run do what it did
without records: list the headers, compute the key and look for its marker; a touched file whose contents did not
change is therefore a hit there, and its record is stored again. A record never holds a failure, and it
cannot make a file pass that no earlier run saw pass with the same key. The assumption is the one of make: a file
whose contents change keeps neither its size nor its modification time. --no-cache reads and stores no record.

A file whose dependency list cannot be made fails its check (it is never a pass and never checked unseen): the
compiler fails, prints no dependency, does not list the file itself, or lists a file that does not exist.

Assumption: the dependency list comes from the compiler of the compile command (the build compiler), not from the
preprocessor of the pinned clang-tidy. The tool environment has no clang of the version of clang-tidy (the pip
package holds only clang-tidy), so the two can differ in predefined macros (`__clang__`, `__GNUC__`, `_MSC_VER`,
`__apple_build_version__`, `__clang_analyzer__`) and in `__has_include` answers. The list is right while no
`#include` depends on such a macro; tools/check_rules.py fails an `#include` under a condition on one of them
(rule `compiler-include`), so the assumption is checked on every commit. On Windows the
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
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

DEFAULT_CACHE_FOLDER = Path("out") / "tidy-cache"
# A cache marker that no run used for this long is deleted.
CACHE_MARKER_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
HEADER_CHECK_FOLDER = "header_check"
HEADER_ALONE_CHECKS = "--checks=-misc-include-cleaner"
CLANG_TIDY_CONFIG_NAME = ".clang-tidy"
# A stored list of what a file depends on (see "The stat record" in the module docstring) is kept in the cache folder
# under this prefix, next to the markers.
STAT_RECORD_PREFIX = "stat-"
# A file whose modification time is this close to the moment its record was taken may still change within the same
# tick of a coarse clock, so such a record is not stored: the next run takes it, when the file is older.
STAT_RACY_WINDOW_NS = 2_000_000_000
# The environment variables that add folders where the compiler looks for headers or move its sysroot; clang-tidy's
# driver and the build compiler both read them, so the stat record keeps their values and watches their folders.
TIDY_ENVIRONMENT_VARIABLES = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "SDKROOT")
_ENVIRONMENT_FOLDER_VARIABLES = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH")
# Generated files of the build live under this folder of the repository, except the Conan packages, whose folders
# never change after they are made (the name of a package folder holds its hash).
GENERATED_FOLDER = "out"
PACKAGE_FOLDER = ("out", "conan-home")
# Options of a compile command that change the folders the compiler searches without being told: a word that starts
# with one of the prefixes, and the word after one of the options that take an argument.
_SEARCH_PREFIXES = ("--sysroot=", "-stdlib=", "-nostdinc", "--target=", "-std=", "-isysroot")
_SEARCH_OPTIONS_WITH_ARGUMENT = frozenset({"-isysroot", "-target", "-arch"})
# Options of a compile command that name a folder where the compiler looks for headers, and take one argument.
_INCLUDE_FOLDER_OPTIONS = frozenset({"-I", "-isystem", "-iquote", "-idirafter"})
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


def file_stamp(path: Path) -> list[int] | None:
    """The size, the modification time in nanoseconds, the inode number and the change time in nanoseconds of `path`, or
    None when it cannot be read. A copy that keeps the modification time (`cp -p`, `rsync -a`, `touch -r`) of a file of
    the same size gets another inode or change time, which no user tool can set."""
    try:
        status = path.stat()
    except OSError:
        return None
    return [status.st_size, status.st_mtime_ns, status.st_ino, status.st_ctime_ns]


def include_folders(compile_command: str, directory: Path) -> list[Path]:
    """The folders that the compile command names for header lookup (-I, -isystem, -iquote, -idirafter, written with
    or without a space), as absolute paths, in the order of the command."""
    folders: list[Path] = []
    words = iter(shlex.split(compile_command))
    for word in words:
        value = None
        if word in _INCLUDE_FOLDER_OPTIONS:
            value = next(words, None)
        else:
            for option in _INCLUDE_FOLDER_OPTIONS:
                if word.startswith(option) and len(word) > len(option):
                    value = word[len(option) :]
                    break
        if value:
            folders.append(Path(directory, value))
    return folders


def search_options(words: Sequence[str]) -> tuple[str, ...]:
    """The words of a compile command that change which folders the compiler searches by itself (sysroot, standard
    library, target, language standard), in order."""
    kept: list[str] = []
    following = iter(words)
    for word in following:
        if word in _SEARCH_OPTIONS_WITH_ARGUMENT:
            kept.append(word)
            kept.extend(list(islice(following, 1)))
        elif word.startswith(_SEARCH_PREFIXES):
            kept.append(word)
    return tuple(kept)


def compiler_search_folders(compiler: str, options: Sequence[str]) -> tuple[str, ...]:
    """The folders `compiler` searches for headers besides the ones its command names, read from the `#include <...>
    search starts here:` list that `-E -x c++ -v` prints on an empty input. A macOS framework folder counts as the
    folder it names. Empty when the compiler cannot be run."""
    try:
        completed = subprocess.run(
            [compiler, *options, "-E", "-x", "c++", "-v", os.devnull],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return ()
    folders: list[str] = []
    inside_list = False
    for line in (completed.stderr or "").splitlines():
        if line.startswith("#include <...> search starts here"):
            inside_list = True
        elif line.startswith("End of search list"):
            break
        elif inside_list and line.startswith(" "):
            folders.append(line.strip().removesuffix("(framework directory)").strip())
    return tuple(folders)


class FileSystemMemo:
    """What the file system said during one run, kept so that the many files which share headers do not ask again: the
    fingerprint of a folder, which is the sorted names in it, alone or with every folder under it, and the folders a
    compiler searches by itself. A file stamp is never kept. The threads of a run share one. The out/ and .git folders of the repository are never entered: the run
    itself writes to out/ (the cache, the build), and git writes to .git."""

    def __init__(self, repository: Path, share_compiler_folders_with: FileSystemMemo | None = None) -> None:
        root = repository.resolve()
        self.skipped = {root / "out", root / ".git"}
        self.remembered: dict[tuple[Path, bool], str] = {}
        # What each compiler prints for `--version`, asked once per run: a Command Line Tools update changes it while the
        # `/usr/bin/c++` shim that stamps the compiler stays the same file.
        self.compiler_versions: dict[str, str] = (
            share_compiler_folders_with.compiler_versions if share_compiler_folders_with else {}
        )
        # The folders each compiler searches by itself, asked once per run; a memo made later can share them.
        self.compiler_folders: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = (
            share_compiler_folders_with.compiler_folders if share_compiler_folders_with else {}
        )

    def stamp(self, path: str) -> list[int] | None:
        """The (size, modification time, inode, change time) of `path`, read now: a stamp kept from earlier in the run
        could hide a file that an editor saved since."""
        return file_stamp(Path(path))

    def implicit_folders(self, compiler: str, options: tuple[str, ...]) -> tuple[str, ...]:
        """The folders `compiler` searches for headers without being told, with the sysroot and language options of
        its command, as its `-E -x c++ -v` run on an empty input prints them (see compiler_search_folders). Asked once
        per run; an empty answer when the compiler cannot be run."""
        key = (compiler, options)
        if key not in self.compiler_folders:
            self.compiler_folders[key] = compiler_search_folders(compiler, options)
        return self.compiler_folders[key]

    def compiler_version(self, compiler: str) -> str:
        """The text `compiler --version` prints, asked once per run; empty when the compiler cannot be run."""
        if compiler not in self.compiler_versions:
            try:
                completed = subprocess.run(
                    [compiler, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
                )
                self.compiler_versions[compiler] = completed.stdout or ""
            except OSError:
                self.compiler_versions[compiler] = ""
        return self.compiler_versions[compiler]

    def fingerprint(self, folder: Path, with_subfolders: bool) -> str:
        key = (folder, with_subfolders)
        found = self.remembered.get(key)
        if found is None:
            digest = hashlib.sha256()
            stack = [folder]
            while stack:
                current = stack.pop()
                try:
                    entries = sorted(os.scandir(current), key=lambda entry: entry.name)
                except OSError:
                    digest.update(f"{current}:missing\n".encode("utf-8"))
                    continue
                for entry in entries:
                    is_folder = entry.is_dir()
                    digest.update(f"{current}/{entry.name}:{'d' if is_folder else 'f'}\n".encode("utf-8"))
                    if is_folder and with_subfolders and Path(entry.path) not in self.skipped:
                        stack.append(Path(entry.path))
            found = digest.hexdigest()
            self.remembered[key] = found
        return found


def stat_snapshot(
    source: Path,
    options: Sequence[str],
    version_text: str,
    entries: Sequence[tuple[str, Path]],
    config_files: Sequence[Path],
    dependencies: Sequence[Path],
    repository: Path,
    memo: FileSystemMemo,
) -> dict[str, object]:
    """What the file system says about everything that the key of `source` reads, without reading any contents (see
    "The stat record" in the module docstring)."""
    root = repository.resolve()
    tidy_binary = shutil.which("clang-tidy")
    watched_with_subfolders: set[Path] = set()
    watched_alone: set[str] = {os.path.dirname(os.path.abspath(source))}
    compilers: dict[str, list[int] | None] = {}
    compiler_versions: dict[str, str] = {}
    for command, directory in entries:
        words = shlex.split(command)
        found = shutil.which(words[0]) if words else None
        compilers[words[0] if words else ""] = memo.stamp(os.path.realpath(found)) if found else None
        if words:
            compiler_versions[words[0]] = memo.compiler_version(words[0])
            watched_alone.update(memo.implicit_folders(words[0], search_options(words[1:])))
        for folder in include_folders(command, directory):
            resolved = folder.resolve()
            inside_repository = root in (resolved, *resolved.parents)
            if inside_repository and "out" not in resolved.relative_to(root).parts[:1]:
                watched_with_subfolders.add(resolved)
            else:
                watched_alone.add(str(resolved))
    for variable in _ENVIRONMENT_FOLDER_VARIABLES:
        watched_alone.update(name for name in os.environ.get(variable, "").split(os.pathsep) if name)
    dependency_names = [str(dependency) for dependency in dependencies]
    watched_alone.update(os.path.dirname(os.path.abspath(name)) for name in dependency_names)
    tops = [str(top) + os.sep for top in watched_with_subfolders]
    layout = {str(folder): memo.fingerprint(folder, True) for folder in sorted(watched_with_subfolders)}
    for folder in sorted(watched_alone):
        if not any((folder + os.sep).startswith(top) for top in tops):
            layout[folder] = memo.fingerprint(Path(folder), False)
    return {
        "version": version_text,
        "options": list(options),
        "commands": [[command, str(directory)] for command, directory in entries],
        "source": memo.stamp(str(source)),
        "dependencies": {name: memo.stamp(name) for name in dependency_names},
        "configs": {str(config): memo.stamp(str(config)) for config in config_files},
        "compilers": compilers,
        "compiler_versions": compiler_versions,
        "tidy_binary": memo.stamp(os.path.realpath(tidy_binary)) if tidy_binary else None,
        "environment": {name: os.environ.get(name) for name in TIDY_ENVIRONMENT_VARIABLES},
        "layout": layout,
    }


def has_generated_dependency(source: Path, dependencies: Sequence[Path], repository: Path) -> bool:
    """True when a header listed for `source` (the file itself excepted) is a generated file of the build: under the
    repository's out/ folder, but not in the Conan packages (GENERATED_FOLDER, PACKAGE_FOLDER). Such a file sits in a
    folder that the record does not search recursively, so no record is stored for the file."""
    root = repository.resolve()
    for dependency in dependencies:
        resolved = dependency.resolve()
        if resolved == source.resolve():
            continue
        try:
            parts = resolved.relative_to(root).parts
        except ValueError:
            continue
        if parts[:1] == (GENERATED_FOLDER,) and parts[: len(PACKAGE_FOLDER)] != PACKAGE_FOLDER:
            return True
    return False


def snapshot_is_racy(snapshot: dict[str, object], taken_ns: int) -> bool:
    """True when the source or a listed file was modified within STAT_RACY_WINDOW_NS of `taken_ns`, or cannot be
    read."""
    dependency_stamps = snapshot["dependencies"].values()  # type: ignore[union-attr]
    config_stamps = snapshot["configs"].values()  # type: ignore[union-attr]
    stamps = [snapshot["source"], *dependency_stamps, *config_stamps]
    return any(stamp is None or stamp[1] > taken_ns - STAT_RACY_WINDOW_NS for stamp in stamps)


def stat_record_path(cache_folder: Path, source: Path, options: Sequence[str]) -> Path:
    identity = "\0".join([str(source.resolve()), *options])
    return cache_folder / f"{STAT_RECORD_PREFIX}{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def read_stat_record(record_path: Path) -> dict[str, object] | None:
    """The stored record, or None when there is none or it is not a well-formed one."""
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if isinstance(record["key"], str) and isinstance(record["snapshot"], dict):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def write_stat_record(record_path: Path, key: str, snapshot: dict[str, object]) -> None:
    """Stores the record by writing a temporary file and renaming it, so a reader never sees half of one."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = record_path.with_name(f"{record_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps({"key": key, "snapshot": snapshot}), encoding="utf-8")
        os.replace(temporary, record_path)
    except OSError:
        temporary.unlink(missing_ok=True)


def stat_record_hit(
    cache_folder: Path,
    record_path: Path,
    source: Path,
    options: Sequence[str],
    version_text: str,
    entries: Sequence[tuple[str, Path]],
    config_files: Sequence[Path],
    repository: Path,
    memo: FileSystemMemo,
) -> str | None:
    """The stored key when the record of `source` says its last pass still holds, that is, when the marker of the key
    exists and the file system says the same as when the record was taken; otherwise None."""
    record = read_stat_record(record_path)
    if record is None or not (cache_folder / str(record["key"])).exists():
        return None
    stored = record["snapshot"]
    listed = stored.get("dependencies")  # type: ignore[union-attr]
    if not isinstance(listed, dict):
        return None
    current = stat_snapshot(
        source, options, version_text, entries, config_files, [Path(name) for name in listed], repository, memo
    )
    return str(record["key"]) if current == stored else None


def check_file(
    path: Path,
    build_folder: Path,
    repository: Path,
    cache_folder: Path | None,
    version_text: str,
    commands: dict[Path, list[tuple[str, Path]]],
    memo: FileSystemMemo | None = None,
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
    memo = memo if memo is not None else FileSystemMemo(repository)
    record_path: Path | None = None
    snapshot: dict[str, object] = {}
    taken_ns = 0

    def remember_pass() -> None:
        """Stores the record of this pass, unless a file changed while the check ran or one is too new to trust."""
        if record_path is None:
            return
        if has_generated_dependency(path, dependencies, repository):
            return
        fresh_memo = FileSystemMemo(repository, share_compiler_folders_with=memo)
        later = stat_snapshot(
            path, options, version_text, entries or [], config_files, dependencies, repository, fresh_memo
        )
        if later == snapshot and not snapshot_is_racy(snapshot, taken_ns):
            write_stat_record(record_path, key, snapshot)

    if cache_folder is not None and entries:
        record_path = stat_record_path(cache_folder, path, options)
        stored_key = stat_record_hit(
            cache_folder, record_path, path, options, version_text, entries, config_files, repository, memo
        )
        if stored_key is not None:
            (cache_folder / stored_key).touch()
            record_path.touch()
            return TidyResult(str(path), True, "")
        union: set[Path] = set()
        try:
            for command, directory in entries:
                union.update(read_dependencies(command, directory, path))
        except DependencyListError as error:
            return TidyResult(str(path), False, f"cannot list the dependencies of {path}: {error}\n")
        dependencies = sorted(union)
        taken_ns = time.time_ns()
        snapshot = stat_snapshot(path, options, version_text, entries, config_files, dependencies, repository, memo)
        key = cache_key(version_text, config_files, options, compile_commands, path, dependencies, repository)
        marker = cache_folder / key
        if marker.exists():
            marker.touch()
            remember_pass()
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
            remember_pass()
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
    memo = FileSystemMemo(repository)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(
            pool.map(
                lambda path: check_file(path, build_folder, repository, cache_folder, version_text, commands, memo),
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
