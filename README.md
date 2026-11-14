# FluidiTy

FluidiTy will be a simulator for liquids, snow, gas (fire and smoke) and moving solids, driven from Blender. When it
is finished, it will move particles through a grid of cells of mixed sizes, run on the CPU or on an NVIDIA GPU, and
write its results in open formats (Alembic, USD, OpenVDB) that Blender reads natively. Most of it does not exist
yet (see "Status").

## Status

This branch, `rebuild`, is a rebuild from the ground up, in progress. It shares no history with the earlier proof of
concept. Modules arrive one at a time, each in its final form, so most of the solver does not exist here yet.
GPU code is compiled but not run until the owner's hardware run is recorded in `docs/gpu-runs.md`.

## Where things are

- `AGENTS.md`: the rules every file and every commit follow. Read it before writing code.
- `docs/glossary.md`: the words the code uses, what they mean, and the words it never uses.
- `docs/design/programme.md`: the design of the whole rebuild, its decisions and the module order.

## Build

The tools come from `uv` (see `pyproject.toml`). Conan keeps its home inside `out/`, so your own Conan profile is
never read or changed. From the repository root, on macOS (on Linux, use the `linux-` presets):

```sh
export CONAN_HOME="$PWD/out/conan-home"
uv run conan profile detect   # once, the first time
uv run conan install . --lockfile conan.lock -of out/conan/macos-debug-cpu -s build_type=Debug \
    -s compiler.cppstd=23 -s:b compiler.cppstd=23 --build=missing
uv run cmake --preset macos-debug-cpu
uv run cmake --build --preset macos-debug-cpu
uv run ctest --preset macos-debug-cpu
```

For a release build, replace `debug` with `release` and `Debug` with `Release`. After a change to `conanfile.py`,
refresh the lockfile with `uv run conan lock create . -s compiler.cppstd=23 -s:b compiler.cppstd=23`.

On Windows, use the `windows-` presets from a Visual Studio Developer shell (or run Conan's `conanvcvars` script
first), so that CMake finds the MSVC compiler. In PowerShell, the first line is
`$env:CONAN_HOME = "$PWD/out/conan-home"`; the other commands are the same.

Only the macOS presets have been run so far, and `linux-release-cuda` has been built in the container below. The
other Linux and Windows presets are written but not yet run.

### CI

`.github/workflows/checks.yml` describes the checks CI runs: format and rules, the CPU build and tests on Linux, macOS and Windows,
and the CUDA compile. It is linted with actionlint and has run on GitHub (scratch branch `rebuild-ci`) with every job green, so the
Windows CPU code has been compiled and its tests run. No CI job runs GPU code, and the Windows CUDA build has never
been compiled. The local gate, `tools/check_commit.py`, remains the commit rule.

### GPU code

The GPU code is compiled in a Linux container that has the CUDA compiler. Docker must be running (on macOS,
`colima start`). From the repository root:

```sh
uv run python tools/check_commit.py --only cuda
```

This builds the image of `tools/cuda-container/Dockerfile` and runs it with the repository mounted read-only. The
tools, the Conan packages and the build tree live in a Docker volume named after this checkout's path,
`fluidity-cuda-out-<hash>`, so only the first run fetches them and two checkouts never share a build tree. Inside,
it builds everything of the preset `linux-release-cuda`, tidies the `.cc` files only that build compiles, and runs
the tests labelled `cpu` and `no_gpu`. The container has no GPU, so the GPU code is compiled, not run. The output
of the last run is in `out/check-commit/cuda.log`. To start again from nothing, or to remove the volume of a deleted
checkout, list the volumes with `docker volume ls --filter name=fluidity-cuda-out` and remove one with
`docker volume rm <name>`.

The CUDA presets compile for RTX 30, 40 and 50 GPUs. The owner's RTX 5060 Ti is the card for the hardware run,
which `docs/gpu-runs.md` will record; until then no card is tested. RTX 30 and 40 are compiled for, unverified.
That file also says how to run it on Windows:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/run_gpu_tests.ps1
```

On a machine with a GPU, run the GPU tests with `ctest --preset linux-release-cuda -L '^gpu$'` (labels are regular
expressions, and a bare `gpu` also matches `no_gpu`).

Stage your changes first: the full gate that every commit passes, `uv run python tools/check_commit.py`, refuses
unstaged or untracked files; `AGENTS.md` ("Commands") says what each line of its output means.
