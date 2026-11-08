# Blender point-cache reader qualification

The M0 point fixture qualified the native Alembic Points reader in **Blender 5.2.3 LTS**
on Windows x64, Release build `e13b728c6880`. Blender imported an independently written
Ogawa archive as a `POINTCLOUD` with a native `MESH_SEQUENCE_CACHE` modifier. The saved
scene reopened in a fresh process with the archive beside it and evaluated all four
samples, including the empty sample. The earlier Blender-exported vertex-only mesh
probe returned counts of 3, 5, 4 and 4 for a fixture that expected 3, 5, 0 and 4.
[Design record](../superpowers/specs/2026-11-07-fluidity-design.md#delivery-choice-and-prior-art)

## Reproduce the Windows run

Use the pinned Blender build and PowerShell 7.4 or later initialized for x64 Visual Studio 2022.
Run from the repository root. The measured setup used Windows 25H2 build 26200.9711,
VS Build Tools 17.14.44, MSVC 19.44.35229.0 with SDK 10.0.26100.0, CPython 3.13.18
and uv 0.13.7. The locked build tools were Conan 2.34.0, CMake and CTest 4.4.4, and
Ninja 1.13.2 (`1.13.2.git.kitware.jobserver-pipe-1`). The writer is a standalone
C++20 Release target using the dynamic MSVC CRT.

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$writer = 'tools/blender/point_cache_writer'
$runOut = 'out/blender-point-cache-qualification'
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$env:CONAN_HOME = "$PWD/out/blender-conan-home"
$uvArgs = @('run', '--project', 'reference', '--locked', '--group', 'native', '--offline')

if (Test-Path -LiteralPath $runOut) { throw 'Choose a fresh output directory under out.' }
& $blender --version
chcp 65001
```

For first-time setup with network access, populate the locked Python environment
and cache the exact Conan Center binaries below. Profile detection in the measured
shell selected Windows x64, MSVC 194 and the Release dynamic CRT. With these
dependencies already cached, continue to the build.

```powershell
uv sync --project reference --locked --group native
uv @uvArgs conan profile detect
uv @uvArgs conan download 'imath/3.2.2#d5203a84ee631ae6ab2da7c3a6b16787:4d8ab52ebb49f51e63d5193ed580b5a7672e23d5#2602df6cb51d3206c03156ca1e66cefa' --remote conancenter
uv @uvArgs conan download 'alembic/1.8.8#1eb7f96673ab004a34f1085147fac5f2:2e8456ff78d7c190194ff9cd0f6de6a26bffd7b1#7b92fa15f02d065078c9c47cbd40ed38' --remote conancenter
```

The writer's [Conan lock](../../tools/blender/point_cache_writer/conan.lock) pins
the recipes for Alembic 1.8.8 and Imath 3.2.2. The download commands identify
the measured package IDs and revisions. Conan selected the static C++14
binaries through its compatibility mapping for the C++20 writer. The Alembic
package uses Ogawa with HDF5 disabled.

```powershell
uv @uvArgs conan install $writer --lockfile "$writer/conan.lock" `
  --output-folder "$runOut/conan" --build=never --no-remote -s:h compiler.cppstd=20 `
  -c:h tools.cmake.cmaketoolchain:generator=Ninja -c:h 'tools.cmake.cmaketoolchain:user_presets='
uv @uvArgs cmake -S $writer -B "$runOut/build" -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_TOOLCHAIN_FILE=$PWD/$runOut/conan/conan_toolchain.cmake"
uv @uvArgs cmake --build "$runOut/build"
uv @uvArgs ctest --test-dir "$runOut/build" --verbose --no-tests=error

$archive = "$runOut/fixture/FluidiTyPoints.abc"
New-Item -ItemType Directory (Split-Path -Parent $archive) | Out-Null
& "$runOut/build/fluidity_write_point_cache.exe" $archive
Get-FileHash -Algorithm SHA256 -LiteralPath $archive

$launchArgs = @('run', '--offline', '--locked', '--script', 'tools/blender/qualify_point_cache.py')
uv @launchArgs --blender $blender --archive $archive --out-dir "$runOut/playback"
```

The writer reads back its own archive and prints `PASS: FluidiTyPoints Ogawa archive readback`.
The measured archive was 2,219 bytes with SHA-256
`18562f138f695e0bcf11b90b53d694cba53cab66874d6a279d9dfe5a19d85a77`.
The launcher checks the pinned Blender build, saves the scene and copies the
scene and archive together into a directory with spaces and non-ASCII characters.
A third process opens another copy of the saved scene with the archive absent.
All three use factory startup, disable Python auto-execution, isolate Blender
user resources and have a 90-second timeout.

## Measured playback on 2026-11-08

The scene uses 24 fps and one meter per Blender unit. The native modifier binds
`/FluidiTyPoints` through `//FluidiTyPoints.abc`, which resolves beside the relocated
scene. Both positive processes reported zero frame-change handlers.

One committed point frame is one producer sample at a physical time. Blender
timeline frames 1 through 4 select the Alembic samples at 1/24 through 4/24 seconds.
The writer verified four samples each for positions, widths, velocities and
unsigned 64-bit IDs. All four arrays had length zero at frame 3.

| Visit | Timeline frame | Point count | First world position in meters | Display radius range in meters |
| --- | --- | --- | --- | --- |
| 1 | 1 | 3 | (1, 0, −1) | 0.0625–0.1875 |
| 2 | 2 | 5 | (2, 0, −2) | 0.0625–0.3125 |
| 3 | 3 | 0 | — | — |
| 4 | 4 | 4 | (4, 0, −4) | 0.0625–0.25 |
| 5 | 3 | 0 | — | — |
| 6 | 1 | 3 | (1, 0, −1) | 0.0625–0.1875 |

For zero-based index `i` at timeline frame `f`, the expected Blender Z-up world
position was `(f, i, -(f+i))` meters and the display radius was `(i+1)/16` meters.
The writer converts vectors to Alembic Y-up as `(x, z, -y)` and stores width as
full diameter. The verifier checked all positions and radii at an absolute
tolerance of 1e-6. The maximum observed component and radius errors were both
zero across the 15 evaluated points, counting repeat visits.

Velocity is diagnostic. With cache velocity name `.velocities` and unit `SECOND`,
Blender exposed a `POINT`-domain `FLOAT_VECTOR` velocity attribute with one
`(1, -2, 3)` m/s vector per point. Its maximum component error was zero, and
the attribute had zero values on both empty visits. The writer verified IDs
`100+i` in the archive. The Blender evaluation uses indices within each sample
and makes no particle-identity claim across samples.

| Process | Exit | Wall time |
| --- | --- | --- |
| Import and save scene | 0 | 1.318 s |
| Reopen relocated scene with archive | 0 | 1.354 s |
| Reopen copied scene with archive absent | 23 | 1.234 s |

These single-run process times include Blender startup and come from source
revision `d9c45b95861227d6781f7c379db35d9d9767a2f0`. In the
missing-archive control, Blender emitted reader warnings and showed the saved
frame-one points on all six visits, with counts of 3, 3, 3, 3, 3 and 3. The
verifier recorded the missing file and the stale geometry, including the first
point at `(1, 0, -1)` meters on frame 2. The per-process receipt failed. The
aggregate receipt passed the control after recording the nonzero exit and
a geometry difference from the relocated playback.

The clean Release repeat passed standalone CTest (1/1), the focused locked Python
tests (10/10), and the full locked Python suite across `reference/tests`,
`tools/tests` and `tools/native/tests` (279/279). Ruff 0.17.4 and clang-format
19.1.5 passed the Python and C++/CUDA checks respectively. Run the checks from
the repository root with those formatter versions on PATH.

```powershell
uv @uvArgs pytest tools/tests/test_qualify_point_cache.py -q --basetemp "$runOut/pytest-focused"
uv @uvArgs pytest reference/tests tools/tests tools/native/tests -q --basetemp "$runOut/pytest-full"

$pythonFiles = @(
  'tools/blender/point_cache_playback.py',
  'tools/blender/qualify_point_cache.py',
  'tools/tests/test_qualify_point_cache.py'
)
ruff format --check --line-length 120 @pythonFiles
ruff check @pythonFiles

$nativeSources = @(git ls-files -- 'native/*.cc' 'native/*.h' 'native/*.cu' 'native/*.cuh')
$nativeSources += 'tools/blender/point_cache_writer/write_point_cache.cc'
clang-format --version
clang-format --dry-run --Werror @nativeSources
```

The ignored `$runOut/playback/receipt.json` stores the archive digest, process arguments,
exit codes, timings, binding and per-visit measurements. Individual logs and
JSON reports are under `original`, `Relocated cache – échantillons` and
`Missing cache – échantillons` in that directory.

## Scope and remaining gates

Qualification covers this four-sample fixture and the exact Windows Blender
build. A later M1-B export adapter can map published point frames to this native
carrier with explicit units, time mapping and a relative project binding.
Positions, counts and radii are the display contract. Keep stable IDs and
diagnostic velocity in the authoritative published data for inspection.

The solver publisher, restart schema, extension and Points Geometry Nodes asset
belong to later integration. The real bake, independent relink and fresh render
with the extension disabled remain M1-B gates. Other platforms, Blender builds,
subframe playback, motion blur and production-scale performance need their own
measurements.

The [2D adaptive-basis qualification](../adaptivity/README.md) records the measured M0 fixture.
Production adaptive 3D and GPU gates remain separate. The bounded CUDA
transfer, pressure and graph results are recorded in the native toolchain
matrix. Lean pilots and later formal contracts use the separate proof workflow.
A release license decision remains open.
[Design milestones](../superpowers/specs/2026-11-07-fluidity-design.md#152-milestone-capabilities-not-an-implementation-plan) ·
[Native toolchain matrix](../native/toolchains.md) ·
[Lean workflow](../../proofs/README.md)
