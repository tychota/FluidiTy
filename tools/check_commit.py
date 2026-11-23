"""Runs every check a commit must pass, in order, and prints one line per check.

Usage: uv run python tools/check_commit.py [--speed quick|commit|full] [--only format|rules|message|cpu|tidy|paper|cuda]
       [--allow-dirty] [--no-cache]

The gate checks what will be committed. Before any step it reads `git status`: an untracked file or an unstaged
change prints `FAIL tree: unstaged or untracked: <paths>` and no step runs. Staged changes are fine. With
--allow-dirty it prints `NOTE tree: unstaged or untracked files are being tested` and goes on, for work in progress.

The tree is read again after the last step, and right before and after the cuda container run (which mounts the
live working tree): when the index tree (`git write-tree`) differs from the one at the start, or a file became
unstaged or untracked, the gate prints `FAIL tree: changed while the gate ran (<what changed>)` and exits 1,
whatever the steps printed. This holds at every speed and with --only.

The speed (--speed, default commit):
  quick   while working: format, rules, the test programs of the modules whose files changed built in the debug
          preset, their debug unit tests, and tidy (the cache makes it check only what changed). The modules are named by a line
          `NOTE modules: ...`. A changed file is one that differs from HEAD, staged or not, or is untracked. A change
          under src/fluidity/<module>/ selects that module's unit tests; a change under tests/<module>/ selects the
          module too and also runs its functional tests, in the release build, because they take over 20 s in the
          debug build (a line `NOTE functional tests: ...` names them; the release preset builds and runs them alone,
          and reads `NOT RUN <release preset> (quick)` when no such module is selected); a change
          elsewhere under src/ or tests/ or under cmake/, or to a top-level build file (CMakeLists.txt,
          CMakePresets.json, conanfile.py, conan.lock, pyproject.toml, uv.lock), selects the unit tests of every
          module; a change elsewhere (docs, tools) selects none. Only the test programs of the selected modules are
          built (the unit programs in the debug preset, and the functional ones of a module whose tests changed in the release
          preset; the list is
          out/build/<preset>/test_programs.txt, which CMake writes), so a quick run does not see a break in a module
          that is not selected; `NOTE built: <n> test programs` says so. With every module selected, with no list, or
          with a list that lacks a program that a CMakeLists.txt of a selected module makes, the whole preset is
          built; with none selected nothing is built. The list is read after Conan and the configure, which rewrite
          it. Conan and the configure do not run while the build folder holds a hash (gate-configure-hash) equal to the
          hash of the contents of conanfile.py, conan.lock, CMakePresets.json, CMakeUserPresets.json (when there is
          one), pyproject.toml, uv.lock, every CMakeLists.txt and every file under cmake/, and, on a second line, the
          hash of the CMakeCache.txt that the configure left (so a configure by an IDE or a hand-run `cmake -D...`
          runs both again); a changed file under tests/<name>/ with no src/fluidity/<name>/ folder fails the debug
          preset, because no test would be selected; the line
          `NOTE conan and configure: not run (their files are unchanged)` says so. A run of Conan and the configure
          that passed stores the hash of the files as they were before it began, at every speed; one that failed
          removes it. A compiler or SDK
          upgrade is not in the hash: run `--speed commit` to configure again. The release build, the paper and cuda are left out, each printed as
          `NOT RUN <step> (quick)`, so a quick pass is never read as a full one. An untracked file is a changed file,
          but the gate stops at `FAIL tree` for one unless --allow-dirty is given, so it selects modules only then.
          With --only, quick runs the named step in full (every module's unit and functional tests, both presets).
  commit  before each commit: every step below, with the parallel tests and the tidy cache.
The `message` step checks the messages of the commits a push would send (tools/check_message.py): every commit of
`@{upstream}..HEAD`, or HEAD alone when the branch has no upstream or nothing is unpushed. It runs when the working
tree has no change, that is, when the gate checks commits already made; otherwise it prints
`NOT RUN message (uncommitted changes)`, and with `--only message` that is a failure (exit 1), because nothing was
checked. Quick leaves it out. The message of the commit being made is checked before the commit exists by the
`commit-msg` hook of tools/hooks (see AGENTS.md "Commands").
  full    before a push, and in CI: commit with --no-cache, so tidy checks every file.
--only <step> runs that step at any speed; a step that quick leaves out runs in full when named.

The exit code (EXIT_* below):
  0  commit or full speed, and no check failed. Only this code satisfies the commit rule. A step can still read
     `NOT RUN message (uncommitted changes)` under 0: that message step is the hook's job, and `--only message` fails.
  1  a check failed (FAIL), a step named by --only could not run, or git could not be read.
  3  quick speed, and every check that ran read PASS. This is not the commit gate: the last line printed is
     `QUICK ONLY: not the commit gate`, and the code is never 0, so a script or a hook that reads only the exit code
     cannot take a quick run for a commit-gate pass.

The steps, in order:
  format  check that the clang-format on PATH is the version pyproject.toml pins, then clang-format, as a dry run,
          on every tracked C++ and CUDA file.
  rules   tools/check_rules.py on src/ and tests/, then the pytest suite of tools/, then actionlint on .github/workflows/.
          The rules check prints a NOTE only for a line the staged changes add; the gate prints each such note on a
          line `rules note: <note>` before the PASS or FAIL line (every NOTE rule, not only a word banned in one sense).
          At commit and full speed the same scan also writes the doubtful lines of the commit being checked
          (--also-notes-since): the staged lines when the tree differs from HEAD, else the lines of HEAD (HEAD~1..HEAD),
          so the files are read once. The pytest suite
          runs one pytest process for each test file of tools/, all at once. At quick speed it is left out, with a
          line `NOT RUN tools tests (quick, tools unchanged)`, while no changed file is under a folder of
          TOOLS_TEST_FOLDERS (tools/, cmake/, .github/) or is one of the files of TOOLS_TEST_INPUTS (CMakeLists.txt,
          pyproject.toml, uv.lock, AGENTS.md, docs/glossary.md); a change that git cannot list runs it. The rules check and actionlint always run.
  cpu     (always, at commit and full speed) fetch the Conan packages (Conan keeps its home in out/conan-home), then configure, build and test the
          debug and the release CPU presets of the platform the gate runs on. ctest runs one test per core at once.
          Which tests run: at commit speed the debug preset runs the unit tests only (label `unit`) and prints
          `NOT RUN functional tests in debug (commit speed; run by --speed full)`, and the release preset runs the unit
          and the functional tests; at full speed both presets run every test; --only cpu follows the speed.
          (A test preset cannot say "all cores" in a form that every CMake version accepts, so the gate passes
          `--parallel <cores>`.) On the debug preset ctest also writes a JUnit file (out/check-commit/), and the step
          fails, naming the test, when a test labelled `unit` took more than 200 ms; it prints a line
          `NOTE unit test over 50 ms: <name> (<ms> ms)` for each one above 50 ms. The rule is 50 ms for a unit test run
          alone; 200 ms allows for one test per core running at once. A test over 200 ms in the parallel run is run
          again alone (`ctest -R` with its exact name, one job), up to twice, and the step fails only when its fastest
          time alone is still over 200 ms, naming both times; when a rerun clears it the step prints
          `NOTE unit test over 200 ms under load, <ms> ms alone: <name>`, so load alone cannot fail the gate. A functional test program is ONE ctest entry
          (all its TESTs run in one process); a functional entry has no 50/200 ms limit, and the step prints
          `NOTE functional tests run: <n> gtest tests in <m> programs (one ctest entry each)`, counted from the XML
          report each program writes (out/build/<preset>/gtest_reports/), for each preset that ran functional tests.
  tidy    check that the clang-tidy on PATH is the pinned version, then clang-tidy, with the compile commands of the
          debug preset, over every tracked .cc file under src/ and tests/ that the CPU build compiles, and over each
          header-alone file (header_check/), so every header is checked by itself. A .cc file that only the CUDA
          build compiles, such as a GPU-only test, is tidied by the cuda step. tools/run_tidy.py runs one clang-tidy
          per file, one per core at once, and does not check a file again while nothing it depends on has changed
          (the cache is out/tidy-cache/); --no-cache checks every file. The cache also keeps, after a pass, the list of
          the file's headers with the size and modification time of each (the stat record of tools/run_tidy.py), so a
          file whose list is unchanged is a hit without running the compiler to list its headers.
  paper   compile the method paper with typst, and check that every label named on a `Read:` line under src/ and tests/
          (<sec-...>, <fig-...>, <proc-...>, <eq-...> or <tab-...>, with `-` or `:`) is defined by the paper.
  cuda    build the CUDA image, then, inside a container, build everything of the preset linux-release-cuda, run
          clang-tidy over the .cc files that only the CUDA build compiles (the list CMake writes to
          gpu_only_sources.txt), and run the tests labelled cpu and no_gpu. The container has no GPU, so GPU code is
          compiled, not run. Then every tracked .cc and .cu file under src/ and tests/ must appear in the CUDA build's compile
          commands, and every tracked .h and .cuh file under src/ and tests/ must be compiled alone there, as a header-alone
          file (header_check/<include path>.cc or .cu, which fluidity_add_headers writes): a file that is neither
          fails the step by name. The exception is a test header (a header directly in
          src/fluidity/<module>/testing/, such as fluidity/base/testing/base_support.h): the header-alone files belong to the library, which does not link
          GoogleTest, so a test header is compiled by the test programs that include it.

The container keeps its tools, Conan packages and build tree in a Docker volume named after a short hash of the
repository's absolute path (cuda_out_volume), so two checkouts never share a build tree. List the volumes with
`docker volume ls --filter name=fluidity-cuda-out` and remove one that belongs to a deleted checkout with
`docker volume rm <name>`.

Output: `PASS <step>` or `FAIL <step>: <reason>`, one line each. The cpu step prints one line per preset. After a
failure the later lines read `NOT RUN <step>`. The output of each step's commands goes to out/check-commit/<step>.log,
which keeps the last run.
The exit code is 0 when no step failed and 1 otherwise (see the exit codes above). No step is ever reported as passed
without running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STEP_NAMES = ("format", "rules", "message", "cpu", "tidy", "paper", "cuda")

# The CUDA image, pinned by digest: tag 13.4.1-devel-ubuntu24.04, nvcc 13.4.59. It exists for linux/arm64 and
# linux/amd64; container_platform() picks the one that runs without emulation here.
CUDA_IMAGE = "nvidia/cuda@sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311"
CUDA_BUILD_IMAGE = "fluidity-cuda-build"
CUDA_OUT_VOLUME_PREFIX = "fluidity-cuda-out"
CUDA_PRESET = "linux-release-cuda"
# The repository as the container sees it.
CONTAINER_SOURCE_ROOT = "/src"
# Printed by the container for each file of the CUDA build's compile commands.
COMPILED_MARKER = "compiled: "
# The tools whose version the gate checks against pyproject.toml before it runs them.
PINNED_TOOLS = ("clang-format", "clang-tidy")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_QUICK_ONLY = 3
QUICK_ONLY_LINE = "QUICK ONLY: not the commit gate"
EXIT_CODES_HELP = (
    "exit codes: 0 = commit or full speed and no check failed (the only code that satisfies the commit rule; the "
    "message step may read NOT RUN (uncommitted changes), which the commit-msg hook covers); "
    "1 = a check failed; 3 = quick speed and every check that ran passed, which is not the commit gate "
    f"(the last line printed is '{QUICK_ONLY_LINE}')"
)
SPEED_NAMES = ("quick", "commit", "full")
DEFAULT_SPEED = "commit"
# A change to one of these files, or under src/ outside a module folder, or under cmake/, can change any module.
TOP_LEVEL_BUILD_FILES = frozenset(
    {"CMakeLists.txt", "CMakePresets.json", "conanfile.py", "conan.lock", "pyproject.toml", "uv.lock"}
)
# What the tests of tools/ open, stated once; a quick run leaves them out while no changed file is one of these. The
# folders: the tools and their hook and test files, the CMake helper files (fluidity_targets_test.py configures a
# project with them), and the workflow that check_message_test.py reads. The files: the top-level CMakeLists.txt,
# pyproject.toml and uv.lock (the same test), and the glossary and AGENTS.md that check_rules_test.py reads as they
# are (the banned words and the allowed verbs).
TOOLS_TEST_FOLDERS = ("tools", "cmake", ".github")
TOOLS_TEST_INPUTS = frozenset({"CMakeLists.txt", "pyproject.toml", "uv.lock", "AGENTS.md", "docs/glossary.md"})
# The build folder keeps, after a successful Conan install and configure, a hash of everything those two read; a quick
# run that finds the same hash does neither. CMake writes the list of test programs next to it (see
# fluidity_write_test_programs in cmake/fluidity_targets.cmake).
CONFIGURE_STATE_FILE = "gate-configure-hash"
TEST_PROGRAMS_FILE = "test_programs.txt"
# The files a configure reads, besides every CMakeLists.txt and everything under cmake/.
CONFIGURE_INPUT_FILES = (
    "conanfile.py",
    "conan.lock",
    "CMakePresets.json",
    "CMakeUserPresets.json",
    "pyproject.toml",
    "uv.lock",
)
CONFIGURE_INPUT_FOLDER = "cmake"
NOT_RUN_TOOLS_TESTS = "NOT RUN tools tests (quick, tools unchanged)"
MODULES_ROOT = ("src", "fluidity")
# The functional tests, one folder per module: tests/<module>/.
FUNCTIONAL_TESTS_ROOT = "tests"
# The source folders that every file-wise check covers: the unit tests sit beside the headers, the functional tests
# under tests/.
SOURCE_FOLDERS = ("src", "tests")
# Where CMake's fluidity_add_test makes gtest write the XML report of each functional program, below the build folder.
FUNCTIONAL_REPORTS_FOLDER = "gtest_reports"
# A unit test takes at most 50 ms alone in the debug build. The gate runs one test per core at once, which slows each
# one, so it fails a unit test only above 200 ms and prints a note above 50 ms.
UNIT_TEST_NOTE_SECONDS = 0.05
UNIT_TEST_LIMIT_SECONDS = 0.2
# A unit test over the limit in the parallel run is run again alone, up to this many times; it fails only when its
# fastest time alone is still over the limit, so a busy machine cannot fail the gate and a slow test cannot hide.
UNIT_TEST_RERUN_COUNT = 2
# Printed by the debug preset at commit speed, which runs the unit tests only: the functional programs take about
# 2 minutes there and a few times less in the release build, which runs them at that speed.
FUNCTIONAL_TESTS_NOT_RUN_IN_DEBUG = "NOT RUN functional tests in debug (commit speed; run by --speed full)"

PAPER_DIRECTORY = Path("docs") / "method-paper"
PAPER_TYPST_FILE = PAPER_DIRECTORY / "fluidity-method.typ"
PAPER_PDF = Path("out") / "paper" / "fluidity-method.pdf"
LOG_DIRECTORY = Path("out") / "check-commit"
# The commits a push would send, for the message step.
UNPUSHED_RANGE = "@{upstream}..HEAD"
# The doubtful lines of the commit being checked: the file the scan writes them to, and the command that lists them
# again. The commit is the staged changes (base HEAD) or, on a clean tree, HEAD itself (base HEAD~1).
NOTES_JSON = LOG_DIRECTORY / "notes.json"
STAGED_NOTES_BASE = "HEAD"
LAST_COMMIT_NOTES_BASE = "HEAD~1"


def notes_since_arguments(base: str) -> str:
    return f"--notes-since {base} --json {NOTES_JSON.as_posix()}"


CPP_PATTERNS = ("*.cc", "*.h", "*.cu", "*.cuh")
# The sources a build compiles; every tracked one must be in the CUDA build.
COMPILED_PATTERNS = tuple(f"{folder}/*{suffix}" for folder in SOURCE_FOLDERS for suffix in (".cc", ".cu", ".h", ".cuh"))
HEADER_EXTENSIONS = (".h", ".cuh")
DOCKER_NOT_RUNNING = "Docker is not running (start it with: colima start)"
DOCKER_NOT_FOUND = "Docker is not installed (no docker command on PATH)"
DOCKER_INFO_TIMEOUT_SECONDS = 10
HEADER_CHECK_FOLDER = "header_check"
# Where the CUDA build writes the header-alone files: <this>/<include path>.cc or .cu, for src/<include path>.
CONTAINER_HEADER_CHECK_ROOT = f"/out/build/{CUDA_PRESET}/{HEADER_CHECK_FOLDER}/"
# A test header is a header in a folder named `testing`: base/testing/base_support.h.
TEST_CODE_FOLDER = "testing"

# Runs inside the container. /src is the repository, read-only; /out is the named volume, which keeps the Python
# tools, the Conan packages and the build tree between runs. The presets write to ${sourceDir}/out, which is
# read-only here, so the build folder (-B) and the toolchain file (--toolchain) are given on the command line;
# both take precedence over the preset's values. After the build it prints one `compiled: <path>` line per file of
# the compile commands, which the gate compares with the tracked sources, and tidies the .cc files that only this
# build compiles (CMake lists them in gpu_only_sources.txt), with the clang-tidy of uv.lock.
CONTAINER_SCRIPT = f"""
set -euo pipefail
export CONAN_HOME=/out/conan-home
export UV_PROJECT_ENVIRONMENT=/out/venv
export UV_CACHE_DIR=/out/uv-cache
export UV_PYTHON_INSTALL_DIR=/out/uv-python
cd /src
uv sync --frozen --quiet
export PATH=/out/venv/bin:$PATH
conan profile detect --exist-ok > /dev/null
toolchain_folder=/out/conan/{CUDA_PRESET}
# Always, so a change to conanfile.py or conan.lock reaches the toolchain; packages already built are reused.
conan install /src --lockfile /src/conan.lock -of "$toolchain_folder" -s build_type=Release \\
    -s compiler.cppstd=23 -s:b compiler.cppstd=23 --build=missing
cmake --preset {CUDA_PRESET} -B /out/build/{CUDA_PRESET} --toolchain "$toolchain_folder/conan_toolchain.cmake"
cmake --build /out/build/{CUDA_PRESET}
python -c 'import json, sys; [print("{COMPILED_MARKER}" + entry["file"]) for entry in json.load(open(sys.argv[1]))]' \\
    /out/build/{CUDA_PRESET}/compile_commands.json
# One path per line; with no path, nothing runs.
xargs --no-run-if-empty clang-tidy -p /out/build/{CUDA_PRESET} --quiet < /out/build/{CUDA_PRESET}/gpu_only_sources.txt
# ctest labels are regular expressions: a bare `gpu` would also match `no_gpu`.
ctest --test-dir /out/build/{CUDA_PRESET} --parallel "$(nproc)" -L '^(cpu|no_gpu)$' --no-tests=error --output-on-failure
"""

# A label in Typst text: <sec-run-on>, <eq:pressure>.
_LABEL = re.compile(r"<([A-Za-z][A-Za-z0-9_:.-]*)>")
# A label cited on a `Read:` line has a kind prefix, so `<expected>` or `<first author>` there is not a citation.
_CITED_LABEL = re.compile(r"<((?:sec|fig|proc|eq|tab)[-:][A-Za-z0-9_:.-]+)>")
# Raw text in Typst files, where `<name>` is code, not a label: fenced blocks first, then inline spans.
_TYPST_RAW = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_TYPST_LINE_COMMENT = re.compile(r"(?<!:)//[^\n]*")
_TYPST_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
# A comment line that continues the previous labelled section: `//` and at least two spaces, as in
# `//          <fig-axes>`. A line that starts a new section (`// Fails: ...`) has only one space.
_READ_CONTINUATION = re.compile(r"^\s*//\s{2,}\S")


@dataclass(frozen=True)
class Outcome:
    """The result of one line of the gate. `reason` is empty when the line passed."""

    passed: bool
    reason: str = ""
    notes: tuple[str, ...] = ()  # Lines printed before PASS or FAIL.
    output: str = ""  # What the command printed, standard output and standard error together.


@dataclass(frozen=True)
class Check:
    """One line of the gate: the step it belongs to, the text printed after PASS or FAIL, and how to run it.
    `in_quick` is False for a line that the quick speed leaves out."""

    step: str
    title: str
    run: Callable[[], Outcome]
    in_quick: bool = True
    # Answers why the line cannot run now, as `NOT RUN <title> (<reason>)`, or None when it can.
    unavailable_reason: Callable[[], str | None] = lambda: None


def container_platform() -> str:
    """The Docker platform that matches this processor, so the CUDA image runs without emulation."""
    is_arm = platform.machine().lower() in {"arm64", "aarch64"}
    return "linux/arm64" if is_arm else "linux/amd64"


def cuda_out_volume(repo_root: Path) -> str:
    """The Docker volume of this checkout's CUDA build: the prefix and 12 hex digits of the SHA-256 of its absolute
    path. Two checkouts get two volumes, so Ninja never compares one checkout's sources with another's build."""
    digest = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()
    return f"{CUDA_OUT_VOLUME_PREFIX}-{digest[:12]}"


def pinned_versions(pyproject_text: str) -> dict[str, str]:
    """The `name==version` pins of pyproject.toml, by name."""
    return dict(re.findall(r'"([A-Za-z0-9_.-]+)==([^"]+)"', pyproject_text))


def read_tool_version(version_output: str) -> str | None:
    """The first `version X.Y.Z` of a `--version` output, or None."""
    match = re.search(r"\bversion (\d+(?:\.\d+)+)", version_output)
    return match.group(1) if match else None


def is_test_header(path: str) -> bool:
    """Whether `path` is a header of test code, which the header-alone check does not cover: a header directly in
    `src/fluidity/<module>/testing/`, and nowhere else."""
    parts = Path(path).parts
    is_in_module_testing_folder = len(parts) == 5 and parts[:2] == ("src", "fluidity") and parts[3] == TEST_CODE_FOLDER
    return path.endswith(HEADER_EXTENSIONS) and is_in_module_testing_folder


def sources_not_compiled(tracked: Sequence[str], compiled_output: str) -> list[str]:
    """The tracked files that no `compiled: <path>` line of the container covers, in tracked order.

    The container sees the repository at /src, so `/src/src/a.cc` covers `src/a.cc`. A header is covered by its
    header-alone file: `/out/build/<preset>/header_check/fluidity/a.h.cc` (or `.h.cu`) covers `src/fluidity/a.h`.
    A test header (is_test_header) is never reported."""
    prefix = CONTAINER_SOURCE_ROOT + "/"
    compiled = set()
    for line in compiled_output.splitlines():
        if not line.startswith(COMPILED_MARKER):
            continue
        path = line[len(COMPILED_MARKER) :].strip()
        if path.startswith(prefix):
            compiled.add(path[len(prefix) :])
        elif path.startswith(CONTAINER_HEADER_CHECK_ROOT):
            include_path = str(Path(path[len(CONTAINER_HEADER_CHECK_ROOT) :]).with_suffix(""))
            compiled.add("src/" + include_path)
    return [path for path in tracked if path not in compiled and not is_test_header(path)]


def affects_tools_tests(path: str) -> bool:
    """True when a change to `path` can change the result of the tests of tools/: a file under a folder of
    TOOLS_TEST_FOLDERS, or one of the files of TOOLS_TEST_INPUTS (the files the tests read, by their path)."""
    parts = Path(path).parts
    if parts[:1] in {(folder,) for folder in TOOLS_TEST_FOLDERS}:
        return True
    return Path(path).as_posix() in TOOLS_TEST_INPUTS


def modules_of_changes(changed_paths: Sequence[str]) -> frozenset[str] | None:
    """The modules whose folders hold a changed file: None when every module (the change is under src/ outside a
    module folder, under cmake/, under tests/ outside a module folder, or to a top-level build file), otherwise the
    module folders named under src/fluidity/ or tests/, possibly none. A module that only includes a changed header
    of another module is not selected."""
    modules: set[str] = set()
    for path in changed_paths:
        parts = Path(path).parts
        if len(parts) == 1 and parts[0] in TOP_LEVEL_BUILD_FILES:
            return None
        if parts[:1] == ("cmake",):
            return None
        if parts[:1] == (FUNCTIONAL_TESTS_ROOT,):
            if len(parts) >= 3:
                modules.add(parts[1])
                continue
            return None
        if parts[:1] != (MODULES_ROOT[0],):
            continue
        if parts[:2] == MODULES_ROOT and len(parts) >= 4:
            modules.add(parts[2])
        else:
            return None
    return frozenset(modules)


def unknown_test_modules(changed_paths: Sequence[str], repo_root: Path) -> list[str]:
    """The folders tests/<name>/ that hold a changed file although src/fluidity/<name>/ does not exist, sorted. The
    label of such a name selects no test, and `ctest --no-tests=ignore` would pass without running any, so a typo in the
    folder of a new test cannot be a quick pass."""
    names = {
        parts[1]
        for parts in (Path(path).parts for path in changed_paths)
        if parts[:1] == (FUNCTIONAL_TESTS_ROOT,) and len(parts) >= 3
    }
    return sorted(name for name in names if not (repo_root / Path(*MODULES_ROOT) / name).is_dir())


def unknown_test_modules_reason(names: Sequence[str]) -> str:
    listed = ", ".join(f"tests/{name}/" for name in names)
    return f"changed files under {listed}, which is not a module folder of src/fluidity/; no test would be selected"


def modules_with_changed_functional_tests(changed_paths: Sequence[str]) -> frozenset[str]:
    """The modules with a changed file under tests/<module>/ or under src/fluidity/<module>/testing/: their
    functional tests run in a quick run too, because the functional tests use the shared test code of that folder."""
    modules = set()
    for path in changed_paths:
        parts = Path(path).parts
        if parts[:1] == (FUNCTIONAL_TESTS_ROOT,) and len(parts) >= 3:
            modules.add(parts[1])
        if parts[:2] == MODULES_ROOT and len(parts) >= 5 and parts[3] == "testing":
            modules.add(parts[2])
    return frozenset(modules)


def tests_without_tier(junit_text: str) -> list[str]:
    """The tests of the JUnit file of ctest whose labels (`cmake_labels`) hold neither `unit` nor `functional`, in
    file order. Every test has a tier, so one without it means the rule that times the unit tests cannot see it."""
    untiered: list[str] = []
    for case in ElementTree.fromstring(junit_text).iter("testcase"):
        labels = ""
        for label_property in case.iter("property"):
            if label_property.get("name") == "cmake_labels":
                labels = label_property.get("value", "")
        if not {"unit", "functional"} & set(labels.split(";")):
            untiered.append(case.get("name", ""))
    return untiered


def count_functional_tests(report_texts: Sequence[str]) -> int:
    """The number of gtest tests in the XML reports that the functional test programs wrote (one report per program,
    from `--gtest_output=xml:`): the `tests` count of each root element, summed. A functional program is one ctest
    entry, so ctest's own count is the number of programs, not of tests."""
    return sum(int(ElementTree.fromstring(text).get("tests", "0")) for text in report_texts)


def slow_unit_tests(junit_text: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """The unit tests that took too long, from the JUnit file of ctest: (above the limit, above the note level but not
    above the limit), each slowest first. A test is a unit test when its labels (`cmake_labels`, joined by `;`) hold
    `unit`."""
    over_limit: list[tuple[str, float]] = []
    over_note: list[tuple[str, float]] = []
    for case in ElementTree.fromstring(junit_text).iter("testcase"):
        labels = ""
        for label_property in case.iter("property"):
            if label_property.get("name") == "cmake_labels":
                labels = label_property.get("value", "")
        if "unit" not in labels.split(";"):
            continue
        seconds = float(case.get("time", "0"))
        name = case.get("name", "")
        if seconds > UNIT_TEST_LIMIT_SECONDS:
            over_limit.append((name, seconds))
        elif seconds > UNIT_TEST_NOTE_SECONDS:
            over_note.append((name, seconds))
    over_limit.sort(key=lambda entry: entry[1], reverse=True)
    over_note.sort(key=lambda entry: entry[1], reverse=True)
    return over_limit, over_note


def ctest_exact_name_pattern(test_name: str) -> str:
    """The ctest `-R` regular expression that matches exactly `test_name`: CMake's regular expressions read `\\` before
    each of their special characters as that character."""
    return "^" + re.sub(r"([\\^$.\[\]*+?()|])", r"\\\1", test_name) + "$"


def test_seconds_in_junit(junit_text: str, test_name: str) -> float | None:
    """The time of the test of that name in the JUnit file of ctest, or None when the file has no such test."""
    for case in ElementTree.fromstring(junit_text).iter("testcase"):
        if case.get("name") == test_name:
            return float(case.get("time", "0"))
    return None


def configure_inputs_hash(repo_root: Path, preset: str) -> str:
    """The SHA-256 over the contents of everything that Conan install and CMake configure read for `preset`: the files
    of CONFIGURE_INPUT_FILES, every CMakeLists.txt, every file under cmake/, and the preset and platform names. A file
    that does not exist counts by its absence. Folders that start with a dot, and out/, are not entered."""
    digest = hashlib.sha256()
    digest.update(f"{preset}\0{platform.system()}\0".encode("utf-8"))
    paths = [repo_root / name for name in CONFIGURE_INPUT_FILES]
    for folder, subfolders, names in os.walk(repo_root):
        subfolders[:] = sorted(
            name for name in subfolders if not name.startswith(".") and name not in ("out", "__pycache__")
        )
        relative_folder = Path(folder).relative_to(repo_root)
        in_cmake_folder = relative_folder.parts[:1] == (CONFIGURE_INPUT_FOLDER,)
        paths.extend(Path(folder, name) for name in sorted(names) if name == "CMakeLists.txt" or in_cmake_folder)
    for path in sorted(set(paths)):
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            content_hash = "absent"
        digest.update(f"{path.relative_to(repo_root).as_posix()}\0{content_hash}\0".encode("utf-8"))
    return digest.hexdigest()


def test_programs_to_build(
    programs_text: str, modules: frozenset[str], functional_modules: frozenset[str]
) -> list[str] | None:
    """The test programs a quick run needs: the unit programs of `modules` and the functional programs of
    `functional_modules`, from the lines `<module> <tier> <program>` of test_programs.txt, sorted. None when no line is
    well formed, so that the caller builds everything."""
    wanted: set[str] = set()
    seen_line = False
    for line in programs_text.splitlines():
        fields = line.split()
        if len(fields) != 3:
            return None
        seen_line = True
        module, tier, program = fields
        if (tier == "unit" and module in modules) or (tier == "functional" and module in functional_modules):
            wanted.add(program)
    return sorted(wanted) if seen_line else None


# A test program is made by `fluidity_add_test(<program> ...)`.
_ADD_TEST_CALL = re.compile(r"\bfluidity_add_test\s*\(\s*([A-Za-z0-9_]+)")


def has_unlisted_test_program(repo_root: Path, programs_text: str, modules: frozenset[str]) -> bool:
    """True when a `CMakeLists.txt` of a module in `modules` (src/fluidity/<module>/ or tests/<module>/) makes a test
    program that the lines `<module> <tier> <program>` of `programs_text` do not list for that module: the list is
    older than the file, so it cannot say what to build."""
    listed: dict[str, set[str]] = {}
    for line in programs_text.splitlines():
        fields = line.split()
        if len(fields) == 3:
            listed.setdefault(fields[0], set()).add(fields[2])
    for module in modules:
        for folder in (Path(*MODULES_ROOT, module), Path(FUNCTIONAL_TESTS_ROOT, module)):
            try:
                text = (repo_root / folder / "CMakeLists.txt").read_text(encoding="utf-8")
            except OSError:
                continue
            if any(program not in listed.get(module, set()) for program in _ADD_TEST_CALL.findall(text)):
                return True
    return False


def host_cpu_presets() -> tuple[str, str]:
    """The debug and release CPU presets of the platform the gate runs on."""
    prefix = {"Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")
    return f"{prefix}-debug-cpu", f"{prefix}-release-cpu"


class TreeChanged(Exception):
    """The tree the gate is checking changed while the gate ran. The message says what changed."""


class Gate:
    """Runs the checks for one repository. Every command goes through `subprocess.run`."""

    def __init__(self, repo_root: Path, use_tidy_cache: bool = True) -> None:
        self.repo_root = repo_root
        self.use_tidy_cache = use_tidy_cache
        self.environment = dict(os.environ, CONAN_HOME=str(repo_root / "out" / "conan-home"))
        self.started_logs: set[str] = set()
        # The changed files, read once: no step of the gate edits a tracked file.
        self.changed_cache: list[str] | None | bool = False
        # The tree being checked, recorded by `record_tree` at the start: the index tree hash and the dirty paths.
        self.recorded_tree: tuple[str, tuple[str, ...]] | None = None

    def command(self, log_name: str, arguments: Sequence[str]) -> Outcome:
        """Runs one command from the repository root and adds its output to the step's log, which keeps the last run."""
        return self.log_outcome(log_name, arguments, self.run_command(arguments))

    def run_command(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
        """Runs one command from the repository root, or returns None when the program is not on PATH. It writes no
        log, so several can run at once."""
        try:
            return subprocess.run(
                list(arguments),
                cwd=self.repo_root,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None

    def log_outcome(
        self, log_name: str, arguments: Sequence[str], completed: subprocess.CompletedProcess[str] | None
    ) -> Outcome:
        """Adds a finished command to the step's log and reads its outcome."""
        if completed is None:
            return Outcome(False, f"{arguments[0]} was not found on PATH")
        log_path = self.repo_root / LOG_DIRECTORY / f"{log_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if log_name in self.started_logs else "w"
        self.started_logs.add(log_name)
        output = completed.stdout or ""
        with log_path.open(mode, encoding="utf-8") as log:
            log.write(f"$ {' '.join(arguments)}\n{output}\n")
        if completed.returncode == 0:
            return Outcome(True, output=output)
        shown = " ".join(arguments[:3])
        relative_log = log_path.relative_to(self.repo_root).as_posix()
        return Outcome(False, f"`{shown} ...` exited with {completed.returncode} (log: {relative_log})", output=output)

    def commands_at_once(self, log_name: str, command_list: Sequence[Sequence[str]]) -> Outcome:
        """Runs the commands at the same time, one thread each, then logs them in the order given. Returns the outcome
        of the first one that failed, or a pass when none did."""
        with ThreadPoolExecutor(max_workers=max(1, len(command_list))) as pool:
            finished = list(pool.map(self.run_command, command_list))
        outcomes = [self.log_outcome(log_name, arguments, done) for arguments, done in zip(command_list, finished)]
        return next((outcome for outcome in outcomes if not outcome.passed), Outcome(True))

    def check_tool_version(self, log_name: str, tool: str) -> Outcome:
        """Fails unless `tool --version` prints the version pyproject.toml pins for it."""
        try:
            pins = pinned_versions((self.repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        except OSError as error:
            return Outcome(False, f"pyproject.toml cannot be read: {error}")
        pinned = pins.get(tool)
        if pinned is None:
            return Outcome(False, f"pyproject.toml pins no version of {tool}")
        asked = self.command(log_name, [tool, "--version"])
        if not asked.passed:
            return asked
        running = read_tool_version(asked.output)
        if running != pinned:
            return Outcome(False, f"{tool} {running} runs, but pyproject.toml pins {pinned} (run: uv sync --frozen)")
        return Outcome(True)

    def commands(self, log_name: str, command_list: Sequence[Sequence[str]]) -> Outcome:
        """Runs commands in order and stops at the first failure."""
        for arguments in command_list:
            outcome = self.command(log_name, arguments)
            if not outcome.passed:
                return outcome
        return Outcome(True)

    def tracked_files(self, patterns: Sequence[str]) -> list[str] | None:
        """Tracked files that match the git pathspecs, or None when git fails."""
        try:
            completed = subprocess.run(
                ["git", "ls-files", "--", *patterns],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        if completed.returncode != 0:
            return None
        return [line for line in (completed.stdout or "").splitlines() if line.strip()]

    # Steps ------------------------------------------------------------------------------------------------

    def check_format(self) -> Outcome:
        files = self.tracked_files(CPP_PATTERNS)
        if files is None:
            return Outcome(False, "git ls-files failed")
        if not files:
            return Outcome(False, "no tracked C++ or CUDA file was found")
        version = self.check_tool_version("format", "clang-format")
        if not version.passed:
            return version
        return self.command("format", ["clang-format", "--dry-run", "--Werror", *files])

    def parent_commit_exists(self) -> bool:
        """True when `HEAD~1` names a commit, so the lines the last commit adds can be listed."""
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", "HEAD~1"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return False
        return completed.returncode == 0

    def doubtful_lines_base(self) -> str | None:
        """The commit that the doubtful lines are counted from: HEAD when the tree differs from HEAD (the staged lines
        are the change being made), HEAD~1 when it does not (HEAD is the change), None when git cannot say or there
        is no parent to count from."""
        changed = self.changed_paths()
        if changed is None:
            return None
        if changed:
            return STAGED_NOTES_BASE
        return LAST_COMMIT_NOTES_BASE if self.parent_commit_exists() else None

    def count_doubtful_lines(self) -> int:
        """The notes of the lines that the commit being checked adds, as the tree check wrote them to NOTES_JSON in the same
        scan, or 0 when the file cannot be read."""
        try:
            return len(json.loads((self.repo_root / NOTES_JSON).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return 0

    def tools_tests_unchanged(self) -> bool:
        """True when no changed file can change the result of the tools' tests (see TOOLS_TEST_INPUTS). False when git
        cannot list the changes."""
        changed = self.changed_paths()
        if changed is None:
            return False
        return not any(affects_tools_tests(path) for path in changed)

    def tools_test_commands(self, tools: Path) -> list[list[str]]:
        """One pytest command for each test file of tools/, so they can run at once. With no test file found it is one
        command for the whole folder, as before."""
        files = sorted({*tools.glob("*_test.py"), *tools.glob("test_*.py")})
        if not files:
            return [[sys.executable, "-m", "pytest", "-q", str(tools)]]
        return [[sys.executable, "-m", "pytest", "-q", str(file)] for file in files]

    def check_rules(self, lists_doubtful_lines: bool = False, skips_unchanged_tools_tests: bool = False) -> Outcome:
        tools = self.repo_root / "tools"
        notes_path = self.repo_root / NOTES_JSON
        # One scan gives the tree check and, when asked, the doubtful lines of the commit being checked: the staged
        # lines when something is staged, else the lines of HEAD.
        notes_base = self.doubtful_lines_base() if lists_doubtful_lines else None
        lists_notes = notes_base is not None
        if lists_notes:
            notes_path.unlink(missing_ok=True)
        rules = self.command(
            "rules",
            [
                sys.executable,
                str(tools / "check_rules.py"),
                *[str(self.repo_root / folder) for folder in (*SOURCE_FOLDERS, PAPER_DIRECTORY)],
                "--notes-on-staged-lines",
                *(["--also-notes-since", notes_base, "--also-notes-json", str(notes_path)] if lists_notes else []),
                "--repository",
                str(self.repo_root),
            ],
        )
        notes = tuple(f"rules note: {line}" for line in rules.output.splitlines() if ": NOTE " in line)
        if not rules.passed:
            return Outcome(False, rules.reason, notes)
        if skips_unchanged_tools_tests and self.tools_tests_unchanged():
            notes += (NOT_RUN_TOOLS_TESTS,)
            tests = Outcome(True)
        else:
            tests = self.commands_at_once("rules", self.tools_test_commands(tools))
        # With no argument actionlint finds the workflow files of the repository it runs in.
        outcome = tests if not tests.passed else self.command("rules", ["actionlint"])
        if lists_notes and outcome.passed:
            count = self.count_doubtful_lines()
            if count:
                command = f"uv run python tools/check_rules.py {notes_since_arguments(notes_base)}"
                notes += (f"NOTE {count} doubtful lines: {command}",)
        return Outcome(outcome.passed, outcome.reason, notes)

    def unpushed_range(self) -> str | None:
        """`@{upstream}..HEAD` when the branch has an upstream and holds commits that it lacks, else None."""
        try:
            completed = subprocess.run(
                ["git", "rev-list", UNPUSHED_RANGE],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        if completed.returncode != 0 or not (completed.stdout or "").split():
            return None
        return UNPUSHED_RANGE

    def check_message(self) -> Outcome:
        """Checks every commit that the upstream lacks with tools/check_message.py, or only HEAD when the branch has no
        upstream or nothing is unpushed."""
        script = str(self.repo_root / "tools" / "check_message.py")
        unpushed = self.unpushed_range()
        return self.command("message", [sys.executable, script, *(["--range", unpushed] if unpushed else ["HEAD"])])

    def message_unavailable_reason(self) -> str | None:
        """The message of HEAD is checked when the working tree has no change: then the gate is checking a commit."""
        changed = self.changed_paths()
        if changed is None:
            return "git could not read the changes"
        return "uncommitted changes" if changed else None

    def changed_paths(self) -> list[str] | None:
        """The files that differ from HEAD, staged or not, and the untracked files; None when git fails. The answer
        is kept for the rest of the run.

        `--no-renames` lists a moved file under both its old and its new path, so the module it left is selected."""
        if self.changed_cache is False:
            self.changed_cache = self.read_changed_paths()
        return self.changed_cache  # type: ignore[return-value]

    def read_changed_paths(self) -> list[str] | None:
        listed: list[str] = []
        for arguments in (
            ["git", "diff", "--name-only", "-z", "--no-renames", "HEAD"],
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        ):
            try:
                completed = subprocess.run(
                    arguments,
                    cwd=self.repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                return None
            if completed.returncode != 0:
                return None
            listed.extend(path for path in (completed.stdout or "").split("\0") if path)
        return listed

    def configure_state_path(self, preset: str) -> Path:
        return self.repo_root / "out" / "build" / preset / CONFIGURE_STATE_FILE

    def configure_cache_hash(self, preset: str) -> str:
        """The SHA-256 of CMakeCache.txt of the build folder of `preset`, or "absent". A configure made by another tool
        (an IDE, a hand-run `cmake -D...`) rewrites the cache without changing any file the inputs hash reads."""
        try:
            cache = (self.repo_root / "out" / "build" / preset / "CMakeCache.txt").read_bytes()
        except OSError:
            return "absent"
        return hashlib.sha256(cache).hexdigest()

    def store_configure_state(self, preset: str, inputs_hash: str) -> None:
        """Keeps `inputs_hash`, the hash of the files as they were before the Conan install and the configure that just
        passed began, so a file edited while they ran is not recorded as read, and on a second line the hash of the
        CMakeCache.txt that the configure left. When it cannot be written the next quick run runs both again."""
        state_path = self.configure_state_path(preset)
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(f"{inputs_hash}\n{self.configure_cache_hash(preset)}\n", encoding="utf-8")
        except OSError:
            pass

    def configure_is_current(self, preset: str) -> bool:
        """True when the build folder of `preset` was configured with the files as they are now: Conan's toolchain file
        and CMake's cache exist, the hash that the last successful install and configure stored is the hash of the
        files now (see configure_inputs_hash), and CMakeCache.txt is the one that configure left."""
        build_folder = self.repo_root / "out" / "build" / preset
        toolchain = self.repo_root / "out" / "conan" / preset / "conan_toolchain.cmake"
        if not (build_folder / "CMakeCache.txt").is_file() or not toolchain.is_file():
            return False
        try:
            stored = self.configure_state_path(preset).read_text(encoding="utf-8").split()
        except OSError:
            return False
        if len(stored) != 2:
            return False
        return stored == [configure_inputs_hash(self.repo_root, preset), self.configure_cache_hash(preset)]

    def quick_build_targets(
        self, preset: str, modules: frozenset[str], functional_modules: frozenset[str]
    ) -> list[str] | None:
        """The test programs to build for the modules a quick run tests, or None when the whole preset is built: the
        list CMake wrote is missing or is not well formed, or a CMakeLists.txt of a selected module makes a program
        that the list does not have (see has_unlisted_test_program)."""
        try:
            programs_text = (self.repo_root / "out" / "build" / preset / TEST_PROGRAMS_FILE).read_text(encoding="utf-8")
        except OSError:
            return None
        if has_unlisted_test_program(self.repo_root, programs_text, modules | functional_modules):
            return None
        return test_programs_to_build(programs_text, modules, functional_modules)

    def check_cpu_preset(self, preset: str, speed: str = "full", changed_modules_only: bool = False) -> Outcome:
        """Fetches the packages, configures, builds and tests `preset`. At commit speed the debug preset runs the unit
        tests only (a note says the functional tests were not run), and the release preset runs every test; at full
        speed both run every test. With `changed_modules_only` (a quick run) the debug preset runs only the unit tests
        of the modules that the changed files select (see modules_of_changes), and the release preset only the
        functional tests of the modules whose tests changed, which take too long in the debug build; each builds only
        those test programs, says which on notes, and does not run Conan or configure again while the files they read
        are the ones of the last successful run (see configure_is_current)."""
        toolchain_folder = Path("out") / "conan" / preset
        build_type = "Debug" if "-debug-" in preset else "Release"
        setup_commands: list[list[str]] = [
            ["conan", "profile", "detect", "--exist-ok"],
            [
                "conan",
                "install",
                ".",
                "--lockfile",
                "conan.lock",
                "-of",
                toolchain_folder.as_posix(),
                "-s",
                f"build_type={build_type}",
                "-s",
                "compiler.cppstd=23",
                "-s:b",
                "compiler.cppstd=23",
                "--build=missing",
            ],
            ["cmake", "--preset", preset],
        ]
        notes: tuple[str, ...] = ()
        cores = str(os.cpu_count() or 1)
        # The debug preset also times its unit tests: ctest writes each test's time and labels to a JUnit file.
        junit_path = self.repo_root / LOG_DIRECTORY / f"junit-{preset}.xml"
        is_debug = "-debug-" in preset
        ctest = ["ctest", "--preset", preset, "--parallel", cores]
        if is_debug:
            ctest += ["--output-junit", str(junit_path)]
        ctest_runs: list[list[str]] = [ctest]
        if is_debug and speed == "commit" and not changed_modules_only:
            ctest_runs = [[*ctest, "-L", "^unit$"]]
            notes = (FUNCTIONAL_TESTS_NOT_RUN_IN_DEBUG,)
        build_command = ["cmake", "--build", "--preset", preset]
        changed: list[str] | None = None
        modules: frozenset[str] | None = None
        functional_modules: frozenset[str] = frozenset()
        if changed_modules_only and is_debug:
            changed = self.changed_paths()
            unknown = unknown_test_modules(changed or (), self.repo_root)
            if unknown:
                return Outcome(False, unknown_test_modules_reason(unknown), notes, "")
            modules = None if changed is None else modules_of_changes(changed)
            # Labels are regular expressions; the anchors keep `grid` from matching another module's name and
            # `unit` from matching nothing else.
            if modules is None:
                notes = ("NOTE modules: all",)
                ctest_runs = [[*ctest, "-L", "^unit$"]]
            elif modules:
                names = sorted(modules)
                notes = (f"NOTE modules: {', '.join(names)}",)
                module_label = "^(" + "|".join(re.escape(name) for name in names) + ")$"
                ctest_runs = [[*ctest, "-L", module_label, "-L", "^unit$", "--no-tests=ignore"]]
            else:
                notes = ("NOTE modules: none",)
                ctest_runs = []
        elif changed_modules_only:
            # The release preset at quick speed: the functional programs of the modules whose tests changed, and
            # nothing else (the debug preset runs the unit tests).
            changed = self.changed_paths()
            functional_modules = modules_with_changed_functional_tests(changed or ())
            modules = frozenset()
            functional = sorted(functional_modules)
            ctest_runs = []
            if functional:
                notes = (f"NOTE functional tests: {', '.join(functional)} (release build; debug takes over 20 s)",)
                functional_label = "^(" + "|".join(re.escape(name) for name in functional) + ")$"
                ctest_runs = [[*ctest, "-L", functional_label, "-L", "^functional$", "--no-tests=ignore"]]
        if is_debug:
            junit_path.parent.mkdir(parents=True, exist_ok=True)
            junit_path.unlink(missing_ok=True)
        log_name = f"cpu-{preset}"
        if changed_modules_only and self.configure_is_current(preset):
            notes += ("NOTE conan and configure: not run (their files are unchanged)",)
        else:
            # Conan runs every time at the commit and full speeds, so a change to conanfile.py or conan.lock always
            # reaches the toolchain file. The hash is of the files as they are before the commands run.
            inputs_hash = configure_inputs_hash(self.repo_root, preset)
            self.configure_state_path(preset).unlink(missing_ok=True)
            fetched = self.commands(log_name, setup_commands)
            if not fetched.passed:
                return Outcome(False, fetched.reason, notes, fetched.output)
            self.store_configure_state(preset, inputs_hash)
        if changed_modules_only and modules is not None:
            # After the configure, which rewrites the list of test programs that this reads.
            targets = self.quick_build_targets(preset, modules, functional_modules)
            if targets is not None:
                notes += (f"NOTE built: {len(targets)} test programs",)
                build_command = [*build_command, "--target", *targets] if targets else []
        reports_folder = self.repo_root / "out" / "build" / preset / FUNCTIONAL_REPORTS_FOLDER
        for stale_report in reports_folder.glob("*.xml"):
            stale_report.unlink()
        outcome = self.commands(log_name, [*([build_command] if build_command else []), *ctest_runs])
        if outcome.passed:
            notes += self.functional_test_count_notes(reports_folder)
        if not outcome.passed or not is_debug or not ctest_runs:
            return Outcome(outcome.passed, outcome.reason, notes, outcome.output)
        return self.check_unit_test_times(junit_path, notes, outcome.output, preset, log_name)

    def functional_test_count_notes(self, reports_folder: Path) -> tuple[str, ...]:
        """One note with the number of gtest tests that ran inside the functional programs of this run, from the reports
        they wrote, so that the few ctest entries (one per program) are not read as few tests."""
        report_paths = sorted(reports_folder.glob("*.xml"))
        if not report_paths:
            return ()
        try:
            count = count_functional_tests([path.read_text(encoding="utf-8") for path in report_paths])
        except (OSError, ElementTree.ParseError, ValueError):
            return ("NOTE functional tests run: count unreadable",)
        return (f"NOTE functional tests run: {count} gtest tests in {len(report_paths)} programs (one ctest entry each)",)

    def check_unit_test_times(
        self, junit_path: Path, notes: tuple[str, ...], output: str, preset: str, log_name: str
    ) -> Outcome:
        """Fails when a unit test of the debug run is still slower than UNIT_TEST_LIMIT_SECONDS when run alone, naming it
        with both times, and adds a note for each one above UNIT_TEST_NOTE_SECONDS. A test over the limit in the
        parallel run is run again alone (see rerun_slow_unit_test); one that clears the limit gets a note."""
        try:
            junit_text = junit_path.read_text(encoding="utf-8")
            over_limit, over_note = slow_unit_tests(junit_text)
            untiered = tests_without_tier(junit_text)
        except (OSError, ElementTree.ParseError, ValueError) as error:
            return Outcome(False, f"the JUnit file of ctest cannot be read ({error})", notes, output)
        if untiered:
            shown = ", ".join(untiered[:3])
            more = f" and {len(untiered) - 3} more" if len(untiered) > 3 else ""
            return Outcome(False, f"tests without a tier label (unit or functional): {shown}{more}", notes, output)
        time_notes = tuple(
            f"NOTE unit test over {UNIT_TEST_NOTE_SECONDS * 1000:.0f} ms: {name} ({seconds * 1000:.0f} ms)"
            for name, seconds in over_note
        )
        still_over: list[tuple[str, float, float]] = []
        for name, parallel_seconds in over_limit:
            alone_seconds, error = self.rerun_slow_unit_test(preset, name, junit_path, log_name)
            if alone_seconds is None:
                return Outcome(False, f"unit test {name} could not be run alone ({error})", notes + time_notes, output)
            if alone_seconds > UNIT_TEST_LIMIT_SECONDS:
                still_over.append((name, parallel_seconds, alone_seconds))
            else:
                time_notes += (
                    f"NOTE unit test over {UNIT_TEST_LIMIT_SECONDS * 1000:.0f} ms under load, "
                    f"{alone_seconds * 1000:.0f} ms alone: {name}",
                )
        if still_over:
            shown = ", ".join(
                f"{name} ({parallel * 1000:.0f} ms in the run, {alone * 1000:.0f} ms alone)"
                for name, parallel, alone in still_over[:3]
            )
            more = f" and {len(still_over) - 3} more" if len(still_over) > 3 else ""
            reason = f"unit tests over {UNIT_TEST_LIMIT_SECONDS * 1000:.0f} ms: {shown}{more}"
            return Outcome(False, reason, notes + time_notes, output)
        return Outcome(True, "", notes + time_notes, output)

    def rerun_slow_unit_test(
        self, preset: str, test_name: str, junit_path: Path, log_name: str
    ) -> tuple[float | None, str]:
        """Runs one unit test alone (`ctest -R` with its exact name, one job), up to UNIT_TEST_RERUN_COUNT times, and
        stops at the first run within the limit. Returns its fastest time and "", or None and the reason when ctest
        fails or its JUnit file has no time for the test."""
        rerun_path = junit_path.with_name(f"{junit_path.stem}-alone{junit_path.suffix}")
        command = [
            "ctest",
            "--preset",
            preset,
            "-R",
            ctest_exact_name_pattern(test_name),
            "--parallel",
            "1",
            "--output-junit",
            str(rerun_path),
        ]
        fastest: float | None = None
        for _ in range(UNIT_TEST_RERUN_COUNT):
            rerun_path.unlink(missing_ok=True)
            outcome = self.command(log_name, command)
            if not outcome.passed:
                return None, outcome.reason
            try:
                seconds = test_seconds_in_junit(rerun_path.read_text(encoding="utf-8"), test_name)
            except (OSError, ElementTree.ParseError, ValueError) as error:
                return None, f"the JUnit file of the rerun cannot be read: {error}"
            if seconds is None:
                return None, "the rerun did not run the test"
            fastest = seconds if fastest is None else min(fastest, seconds)
            if fastest <= UNIT_TEST_LIMIT_SECONDS:
                break
        return fastest, ""

    def check_tidy(self) -> Outcome:
        version = self.check_tool_version("tidy", "clang-tidy")
        if not version.passed:
            return version
        debug_preset = host_cpu_presets()[0]
        build_folder = Path("out") / "build" / debug_preset
        compile_commands_path = self.repo_root / build_folder / "compile_commands.json"
        if not compile_commands_path.exists():
            return Outcome(False, f"{build_folder.as_posix()}/compile_commands.json is missing; run the cpu step first")
        try:
            entries = json.loads(compile_commands_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return Outcome(False, f"{build_folder.as_posix()}/compile_commands.json cannot be read: {error}")
        try:
            compiled = {Path(entry["directory"], entry["file"]).resolve() for entry in entries}
        except (TypeError, KeyError) as error:
            # Not a list of objects with "directory" and "file": the build folder is broken, not the sources.
            reason = f"is malformed ({type(error).__name__}: {error}); delete {build_folder.as_posix()} and run cpu"
            return Outcome(False, f"{build_folder.as_posix()}/compile_commands.json {reason}")
        files = self.tracked_files([f"{folder}/*.cc" for folder in SOURCE_FOLDERS])
        if files is None:
            return Outcome(False, "git ls-files failed")
        files = [path for path in files if path.endswith(".cc")]
        # A file outside the CPU build, such as a GPU-only test, is tidied by the cuda step, and the cuda step fails
        # for a tracked source that no build compiles.
        in_build = [path for path in files if (self.repo_root / path).resolve() in compiled]
        if not in_build:
            return Outcome(False, "no tracked .cc file under src/ or tests/ is in the CPU build")
        header_checks = sorted(
            str(path) for path in compiled if HEADER_CHECK_FOLDER in path.parts and path.name.endswith(".h.cc")
        )
        # tools/run_tidy.py runs one clang-tidy per file, as many at once as there are cores, prints the failures
        # sorted by path, and skips a file that passed while nothing it depends on has changed. It turns the include
        # check off for a header-alone file, which includes its header and uses nothing from it. The header itself is
        # still checked through HeaderFilterRegex.
        command = [
            sys.executable,
            str(self.repo_root / "tools" / "run_tidy.py"),
            "--build-folder",
            build_folder.as_posix(),
            "--repository",
            str(self.repo_root),
        ]
        if not self.use_tidy_cache:
            command.append("--no-cache")
        return self.command("tidy", [*command, *in_build, *header_checks])

    def check_paper(self) -> Outcome:
        # typst does not make the folder of its output file, and a fresh checkout has no out/paper/.
        (self.repo_root / PAPER_PDF).parent.mkdir(parents=True, exist_ok=True)
        compiled = self.command("paper", ["typst", "compile", PAPER_TYPST_FILE.as_posix(), PAPER_PDF.as_posix()])
        if not compiled.passed:
            return compiled
        defined = self.paper_labels()
        missing = []
        paths = sorted(path for folder in SOURCE_FOLDERS for path in (self.repo_root / folder).rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in {".h", ".cuh", ".cc", ".cu"}:
                continue
            relative = path.relative_to(self.repo_root).as_posix()
            for number, cited_text in read_line_texts(path.read_text(encoding="utf-8")):
                for label in _CITED_LABEL.findall(cited_text):
                    if label not in defined:
                        missing.append(f"{relative}:{number} cites <{label}>, which the paper does not define")
        if not missing:
            return Outcome(True)
        more = f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""
        return Outcome(False, missing[0] + more)

    def paper_labels(self) -> set[str]:
        """Labels defined in the method paper's Typst files. Code and comments are left out."""
        labels: set[str] = set()
        for path in sorted((self.repo_root / PAPER_DIRECTORY).rglob("*.typ")):
            text = _TYPST_RAW.sub(" ", path.read_text(encoding="utf-8"))
            text = _TYPST_BLOCK_COMMENT.sub(" ", text)
            text = _TYPST_LINE_COMMENT.sub(" ", text)
            labels.update(_LABEL.findall(text))
        return labels

    def check_cuda(self) -> Outcome:
        try:
            docker_info = subprocess.run(
                ["docker", "info"],
                cwd=self.repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=DOCKER_INFO_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            return Outcome(False, DOCKER_NOT_FOUND)
        except subprocess.TimeoutExpired:
            # A stopped colima virtual machine can leave the client waiting instead of failing.
            return Outcome(False, DOCKER_NOT_RUNNING)
        if docker_info.returncode != 0:
            return Outcome(False, DOCKER_NOT_RUNNING)
        tracked = self.tracked_files(COMPILED_PATTERNS)
        if tracked is None:
            return Outcome(False, "git ls-files failed")
        tracked = [path for path in tracked if path.endswith((".cc", ".cu", *HEADER_EXTENSIONS))]
        container_folder = self.repo_root / "tools" / "cuda-container"
        docker_platform = container_platform()
        image = self.command(
            "cuda", ["docker", "build", "--platform", docker_platform, "-t", CUDA_BUILD_IMAGE, str(container_folder)]
        )
        if not image.passed:
            return image
        # The container reads the live working tree, so a change during its run could be compiled half written.
        self.fail_if_tree_changed()
        built = self.command(
            "cuda",
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                docker_platform,
                "-v",
                f"{self.repo_root}:{CONTAINER_SOURCE_ROOT}:ro",
                "-v",
                f"{cuda_out_volume(self.repo_root)}:/out",
                CUDA_BUILD_IMAGE,
                "bash",
                "-c",
                CONTAINER_SCRIPT,
            ],
        )
        self.fail_if_tree_changed()
        if not built.passed:
            return built
        missing = sources_not_compiled(tracked, built.output)
        if missing:
            return Outcome(False, "not compiled by the CUDA build: " + ", ".join(missing))
        return Outcome(True)

    def fail_if_tree_changed(self) -> None:
        """Raises TreeChanged when the tree differs from the one recorded at the start."""
        change = self.tree_change()
        if change:
            raise TreeChanged(change)

    def dirty_paths(self) -> list[str] | None:
        """Untracked files and files with unstaged changes, or None when git fails. Ignored files are left out.

        `-z` ends each entry with a NUL byte and never quotes a path, so a path with a space or a non-ASCII
        character is printed as it is."""
        try:
            completed = subprocess.run(
                ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        if completed.returncode != 0:
            return None
        dirty = []
        entries = iter((completed.stdout or "").split("\0"))
        for entry in entries:
            # Two status letters, a space, the path. The first letter is the index, the second the working tree.
            if len(entry) < 4:
                continue
            index_status, tree_status, path = entry[0], entry[1], entry[3:]
            if index_status in "RC":
                # A rename or a copy is followed by a second entry, the original path, which is not dirty itself.
                next(entries, None)
            if index_status == "?" or tree_status != " ":
                dirty.append(path)
        return dirty

    def index_tree(self) -> str | None:
        """The hash of the tree of the index (`git write-tree`), or None when git fails."""
        try:
            completed = subprocess.run(
                ["git", "write-tree"],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        if completed.returncode != 0:
            return None
        return (completed.stdout or "").strip()

    def record_tree(self, dirty: Sequence[str]) -> bool:
        """Records the tree that the gate checks: the index tree and the paths that differ from it. False when git
        cannot say."""
        tree = self.index_tree()
        if tree is None:
            return False
        self.recorded_tree = (tree, tuple(dirty))
        return True

    def tree_change(self) -> str | None:
        """What differs from the tree recorded at the start, or None when it is the same tree. The working tree is
        compared with the index as at the start: a path that became unstaged or untracked is a change."""
        if self.recorded_tree is None:
            return None
        recorded_hash, recorded_dirty = self.recorded_tree
        tree = self.index_tree()
        dirty = self.dirty_paths()
        if tree is None or dirty is None:
            return "git could not be read"
        reasons = []
        if tree != recorded_hash:
            reasons.append(f"the index tree is {tree}, it was {recorded_hash}")
        new_dirty = [path for path in dirty if path not in recorded_dirty]
        if new_dirty:
            reasons.append(f"unstaged or untracked: {', '.join(new_dirty)}")
        return "; ".join(reasons) or None

    def checks(self, speed: str = DEFAULT_SPEED, only: str | None = None) -> list[Check]:
        debug_preset, release_preset = host_cpu_presets()
        changed_only = speed == "quick" and only is None

        def check_debug_preset() -> Outcome:
            return self.check_cpu_preset(debug_preset, speed, changed_modules_only=changed_only)

        def check_release_preset() -> Outcome:
            return self.check_cpu_preset(release_preset, speed, changed_modules_only=changed_only)

        # A quick run builds and runs the functional programs of a module whose tests changed in the release build only.
        runs_functional_in_release = changed_only and bool(modules_with_changed_functional_tests(self.changed_paths() or ()))

        return [
            Check("format", "format", self.check_format),
            Check(
                "rules",
                "rules",
                lambda: self.check_rules(lists_doubtful_lines=not changed_only, skips_unchanged_tools_tests=changed_only),
            ),
            Check(
                "message",
                "message",
                self.check_message,
                in_quick=False,
                unavailable_reason=self.message_unavailable_reason,
            ),
            Check("cpu", f"cpu {debug_preset}", check_debug_preset),
            Check("cpu", f"cpu {release_preset}", check_release_preset, in_quick=runs_functional_in_release),
            Check("tidy", "tidy", self.check_tidy),
            Check("paper", "paper", self.check_paper, in_quick=False),
            Check("cuda", "cuda", self.check_cuda, in_quick=False),
        ]


def read_line_texts(text: str) -> list[tuple[int, str]]:
    """The text after each `Read:` and of the comment lines that continue it, with 1-based line numbers."""
    found = []
    in_read = False
    for number, line in enumerate(text.splitlines(), start=1):
        if "Read:" in line:
            found.append((number, line.split("Read:", 1)[1]))
            in_read = True
        elif in_read and _READ_CONTINUATION.match(line):
            found.append((number, line))
        else:
            in_read = False
    return found


def main(argv: Sequence[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(
        description="Run every check a commit must pass.",
        epilog=EXIT_CODES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--speed",
        choices=SPEED_NAMES,
        default=DEFAULT_SPEED,
        help="quick while working, commit before each commit (default), full before a push and in CI",
    )
    parser.add_argument("--only", choices=STEP_NAMES, help="run one step only")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="test the working tree even with unstaged or untracked files"
    )
    parser.add_argument("--no-cache", action="store_true", help="run clang-tidy on every file, whatever passed before")
    arguments = parser.parse_args(argv)
    gate = Gate(repo_root, use_tidy_cache=not (arguments.no_cache or arguments.speed == "full"))
    checks = [check for check in gate.checks(arguments.speed, arguments.only) if arguments.only in (None, check.step)]
    failed = False
    dirty = gate.dirty_paths()
    if dirty is None:
        print("FAIL tree: git status failed", flush=True)
        failed = True
    elif dirty and arguments.allow_dirty:
        print("NOTE tree: unstaged or untracked files are being tested", flush=True)
    elif dirty:
        print(f"FAIL tree: unstaged or untracked: {', '.join(dirty)}", flush=True)
        failed = True
    if dirty is not None and not gate.record_tree(dirty):
        print("FAIL tree: git write-tree failed", flush=True)
        failed = True
    tree_change_reported = False
    for check in checks:
        if arguments.speed == "quick" and arguments.only is None and not check.in_quick:
            print(f"NOT RUN {check.title} (quick)", flush=True)
            continue
        if failed:
            print(f"NOT RUN {check.title}", flush=True)
            continue
        reason = check.unavailable_reason()
        if reason:
            print(f"NOT RUN {check.title} ({reason})", flush=True)
            # A step that was asked for by name and could not run checked nothing, so it is not a pass.
            failed = failed or arguments.only == check.step
            continue
        try:
            outcome = check.run()
        except TreeChanged as changed:
            print(f"FAIL tree: changed while the gate ran ({changed})", flush=True)
            print(f"NOT RUN {check.title} (the tree changed)", flush=True)
            failed = True
            tree_change_reported = True
            continue
        for note in outcome.notes:
            print(note, flush=True)
        if outcome.passed:
            print(f"PASS {check.title}", flush=True)
        else:
            print(f"FAIL {check.title}: {outcome.reason}", flush=True)
            failed = True
    change = None if tree_change_reported else gate.tree_change()
    if change:
        print(f"FAIL tree: changed while the gate ran ({change})", flush=True)
        failed = True
    if failed:
        return EXIT_FAIL
    if arguments.speed == "quick":
        print(QUICK_ONLY_LINE, flush=True)
        return EXIT_QUICK_ONLY
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
