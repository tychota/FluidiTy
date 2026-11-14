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

Only the macOS presets have been run so far. The Linux and Windows presets are written but not yet run.
