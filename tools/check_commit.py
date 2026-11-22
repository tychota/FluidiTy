"""Runs every check a commit must pass, in order, and prints one line per check.

Usage: uv run python tools/check_commit.py [--speed quick|commit|full] [--only format|rules|message|cpu|tidy|paper|cuda]
       [--allow-dirty] [--no-cache]

The gate checks what will be committed. Before any step it reads `git status`: an untracked file or an unstaged
change prints `FAIL tree: unstaged or untracked: <paths>` and no step runs. Staged changes are fine. With
--allow-dirty it prints `NOTE tree: unstaged or untracked files are being tested` and goes on, for work in progress.

The speed (--speed, default commit):
  quick   while working: format, rules, the debug preset built, the debug unit tests of the modules whose files
          changed, and tidy (the cache makes it check only what changed). The modules are named by a line
          `NOTE modules: ...`. A changed file is one that differs from HEAD, staged or not, or is untracked. A change
          under src/fluidity/<module>/ selects that module's unit tests; a change under tests/<module>/ selects the
          module too and also runs its functional tests (a line `NOTE functional tests: ...` names them); a change
          elsewhere under src/ or tests/ or under cmake/, or to a top-level build file (CMakeLists.txt,
          CMakePresets.json, conanfile.py, conan.lock, pyproject.toml, uv.lock), selects the unit tests of every
          module; a change elsewhere (docs, tools) selects none. The release build, the paper and cuda are left out, each printed as
          `NOT RUN <step> (quick)`, so a quick pass is never read as a full one. An untracked file is a changed file,
          but the gate stops at `FAIL tree` for one unless --allow-dirty is given, so it selects modules only then.
          With --only, quick runs the named step in full (every module's unit and functional tests, both presets).
  commit  before each commit: every step below, with the parallel tests and the tidy cache.
The `message` step checks the message of HEAD (tools/check_message.py). It runs when the working tree has no change,
that is, when the gate checks a commit already made; otherwise it prints `NOT RUN message (uncommitted changes)`.
Quick leaves it out.
  full    before a push, and in CI: commit with --no-cache, so tidy checks every file.
--only <step> runs that step at any speed; a step that quick leaves out runs in full when named.

The exit code (EXIT_* below):
  0  commit or full speed, and every check read PASS. Only this code satisfies the commit rule.
  1  a check failed (FAIL), or git could not be read.
  3  quick speed, and every check that ran read PASS. This is not the commit gate: the last line printed is
     `QUICK ONLY: not the commit gate`, and the code is never 0, so a script or a hook that reads only the exit code
     cannot take a quick run for a commit-gate pass.

The steps, in order:
  format  check that the clang-format on PATH is the version pyproject.toml pins, then clang-format, as a dry run,
          on every tracked C++ and CUDA file.
  rules   tools/check_rules.py on src/ and tests/, then the pytest suite of tools/, then actionlint on .github/workflows/.
          The rules check prints a NOTE only for a line the staged changes add; the gate prints each such note on a
          line `rules note: <note>` before the PASS or FAIL line. At commit and full speed the same scan also writes
          the doubtful lines of the last commit (--also-notes-since), so the files are read once. The pytest suite
          runs one pytest process for each test file of tools/, all at once. At quick speed it is left out, with a
          line `NOT RUN tools tests (quick, tools unchanged)`, while no changed file is under tools/, cmake/ or
          .github/ or is CMakeLists.txt, pyproject.toml or uv.lock at the top level (TOOLS_TEST_INPUTS); a change
          that git cannot list runs it. The rules check and actionlint always run.
  cpu     fetch the Conan packages (Conan keeps its home in out/conan-home), then configure, build and test the
          debug and the release CPU presets of the platform the gate runs on. ctest runs one test per core at once.
          (A test preset cannot say "all cores" in a form that every CMake version accepts, so the gate passes
          `--parallel <cores>`.) On the debug preset ctest also writes a JUnit file (out/check-commit/), and the step
          fails, naming the test, when a test labelled `unit` took more than 200 ms; it prints a line
          `NOTE unit test over 50 ms: <name> (<ms> ms)` for each one above 50 ms. The rule is 50 ms for a unit test run
          alone; 200 ms allows for one test per core running at once.
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
The exit code is 0 when every step passed and 1 otherwise. No step is ever reported as passed without running.
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
    "exit codes: 0 = commit or full speed and every check passed (the only code that satisfies the commit rule); "
    "1 = a check failed; 3 = quick speed and every check that ran passed, which is not the commit gate "
    f"(the last line printed is '{QUICK_ONLY_LINE}')"
)
SPEED_NAMES = ("quick", "commit", "full")
DEFAULT_SPEED = "commit"
# A change to one of these files, or under src/ outside a module folder, or under cmake/, can change any module.
TOP_LEVEL_BUILD_FILES = frozenset(
    {"CMakeLists.txt", "CMakePresets.json", "conanfile.py", "conan.lock", "pyproject.toml", "uv.lock"}
)
# The tests of tools/ read the tools, the CMake helper file (fluidity_targets_test.py configures a project with it and
# the top-level CMakeLists.txt) and the workflows; a quick run leaves them out while no changed file is one of these.
TOOLS_TEST_FOLDERS = ("tools", "cmake", ".github")
TOOLS_TEST_INPUTS = frozenset({"CMakeLists.txt", "pyproject.toml", "uv.lock"})
NOT_RUN_TOOLS_TESTS = "NOT RUN tools tests (quick, tools unchanged)"
MODULES_ROOT = ("src", "fluidity")
# The functional tests, one folder per module: tests/<module>/.
FUNCTIONAL_TESTS_ROOT = "tests"
# The source folders that every file-wise check covers: the unit tests sit beside the headers, the functional tests
# under tests/.
SOURCE_FOLDERS = ("src", "tests")
# A unit test takes at most 50 ms alone in the debug build. The gate runs one test per core at once, which slows each
# one, so it fails a unit test only above 200 ms and prints a note above 50 ms.
UNIT_TEST_NOTE_SECONDS = 0.05
UNIT_TEST_LIMIT_SECONDS = 0.2

PAPER_DIRECTORY = Path("docs") / "method-paper"
PAPER_TYPST_FILE = PAPER_DIRECTORY / "fluidity-method.typ"
PAPER_PDF = Path("out") / "paper" / "fluidity-method.pdf"
LOG_DIRECTORY = Path("out") / "check-commit"
# The doubtful lines of the last commit: the command that lists them and the file it writes.
NOTES_JSON = LOG_DIRECTORY / "notes.json"
NOTES_SINCE_ARGUMENTS = f"--notes-since HEAD~1 --json {NOTES_JSON.as_posix()}"

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
    """True when a change to `path` can change the result of the tests of tools/: a file under tools/, cmake/ or
    .github/, or one of the top-level files that the tests configure or read (see TOOLS_TEST_INPUTS)."""
    parts = Path(path).parts
    if parts[:1] in {(folder,) for folder in TOOLS_TEST_FOLDERS}:
        return True
    return len(parts) == 1 and parts[0] in TOOLS_TEST_INPUTS


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


def host_cpu_presets() -> tuple[str, str]:
    """The debug and release CPU presets of the platform the gate runs on."""
    prefix = {"Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")
    return f"{prefix}-debug-cpu", f"{prefix}-release-cpu"


class Gate:
    """Runs the checks for one repository. Every command goes through `subprocess.run`."""

    def __init__(self, repo_root: Path, use_tidy_cache: bool = True) -> None:
        self.repo_root = repo_root
        self.use_tidy_cache = use_tidy_cache
        self.environment = dict(os.environ, CONAN_HOME=str(repo_root / "out" / "conan-home"))
        self.started_logs: set[str] = set()
        # The changed files, read once: no step of the gate edits a tracked file.
        self.changed_cache: list[str] | None | bool = False

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

    def count_doubtful_lines(self) -> int:
        """The notes of the lines that the last commit adds, as the tree check wrote them to NOTES_JSON in the same
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
        # One scan gives the tree check and, when asked, the doubtful lines of the last commit.
        lists_notes = lists_doubtful_lines and self.parent_commit_exists()
        if lists_notes:
            notes_path.unlink(missing_ok=True)
        rules = self.command(
            "rules",
            [
                sys.executable,
                str(tools / "check_rules.py"),
                *[str(self.repo_root / folder) for folder in (*SOURCE_FOLDERS, PAPER_DIRECTORY)],
                "--notes-on-staged-lines",
                *(["--also-notes-since", "HEAD~1", "--also-notes-json", str(notes_path)] if lists_notes else []),
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
                notes += (f"NOTE {count} doubtful lines: uv run python tools/check_rules.py {NOTES_SINCE_ARGUMENTS}",)
        return Outcome(outcome.passed, outcome.reason, notes)

    def check_message(self) -> Outcome:
        """Checks the message of HEAD with tools/check_message.py."""
        return self.command("message", [sys.executable, str(self.repo_root / "tools" / "check_message.py"), "HEAD"])

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

    def check_cpu_preset(self, preset: str, changed_modules_only: bool = False) -> Outcome:
        """Fetches the packages, configures, builds and tests `preset`. With `changed_modules_only` it runs only the
        tests of the modules that the changed files select (see modules_of_changes), and says which on a note."""
        # Conan runs every time, so a change to conanfile.py or conan.lock reaches the toolchain file.
        toolchain_folder = Path("out") / "conan" / preset
        build_type = "Debug" if "-debug-" in preset else "Release"
        command_list: list[list[str]] = [
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
        ]
        command_list.append(["cmake", "--preset", preset])
        command_list.append(["cmake", "--build", "--preset", preset])
        notes: tuple[str, ...] = ()
        cores = str(os.cpu_count() or 1)
        # The debug preset also times its unit tests: ctest writes each test's time and labels to a JUnit file.
        junit_path = self.repo_root / LOG_DIRECTORY / f"junit-{preset}.xml"
        is_debug = "-debug-" in preset
        ctest = ["ctest", "--preset", preset, "--parallel", cores]
        if is_debug:
            ctest += ["--output-junit", str(junit_path)]
        ctest_runs: list[list[str]] = [ctest]
        if changed_modules_only:
            changed = self.changed_paths()
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
                functional = sorted(modules_with_changed_functional_tests(changed or ()))
                if functional:
                    notes += (f"NOTE functional tests: {', '.join(functional)}",)
                    functional_label = "^(" + "|".join(re.escape(name) for name in functional) + ")$"
                    ctest_runs.append(
                        ["ctest", "--preset", preset, "--parallel", cores]
                        + ["-L", functional_label, "-L", "^functional$", "--no-tests=ignore"]
                    )
            else:
                notes = ("NOTE modules: none",)
                ctest_runs = []
        if is_debug:
            junit_path.parent.mkdir(parents=True, exist_ok=True)
            junit_path.unlink(missing_ok=True)
        command_list.extend(ctest_runs)
        outcome = self.commands(f"cpu-{preset}", command_list)
        if not outcome.passed or not is_debug or not ctest_runs:
            return Outcome(outcome.passed, outcome.reason, notes, outcome.output)
        return self.check_unit_test_times(junit_path, notes, outcome.output)

    def check_unit_test_times(self, junit_path: Path, notes: tuple[str, ...], output: str) -> Outcome:
        """Fails when a unit test of the debug run took more than UNIT_TEST_LIMIT_SECONDS, naming it, and adds a
        note for each one above UNIT_TEST_NOTE_SECONDS."""
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
        if over_limit:
            shown = ", ".join(f"{name} ({seconds * 1000:.0f} ms)" for name, seconds in over_limit[:3])
            more = f" and {len(over_limit) - 3} more" if len(over_limit) > 3 else ""
            reason = f"unit tests over {UNIT_TEST_LIMIT_SECONDS * 1000:.0f} ms: {shown}{more}"
            return Outcome(False, reason, notes + time_notes, output)
        return Outcome(True, "", notes + time_notes, output)

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
        if not built.passed:
            return built
        missing = sources_not_compiled(tracked, built.output)
        if missing:
            return Outcome(False, "not compiled by the CUDA build: " + ", ".join(missing))
        return Outcome(True)

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

    def checks(self, speed: str = DEFAULT_SPEED, only: str | None = None) -> list[Check]:
        debug_preset, release_preset = host_cpu_presets()
        changed_only = speed == "quick" and only is None

        def check_debug_preset() -> Outcome:
            return self.check_cpu_preset(debug_preset, changed_modules_only=changed_only)

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
            Check("cpu", f"cpu {release_preset}", lambda: self.check_cpu_preset(release_preset), in_quick=False),
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
            continue
        outcome = check.run()
        for note in outcome.notes:
            print(note, flush=True)
        if outcome.passed:
            print(f"PASS {check.title}", flush=True)
        else:
            print(f"FAIL {check.title}: {outcome.reason}", flush=True)
            failed = True
    if failed:
        return EXIT_FAIL
    if arguments.speed == "quick":
        print(QUICK_ONLY_LINE, flush=True)
        return EXIT_QUICK_ONLY
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
