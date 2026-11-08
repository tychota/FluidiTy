# Native build and validation

The native foundation contains a C++20 CPU reference implementation and a small CUDA hardware
probe. CMake presets use Ninja and separate Conan, build and fixture directories for every
configuration. The [toolchain matrix](toolchains.md) records the measured Windows configuration
and its recipe revisions.

## Windows prerequisites

- An x64 Visual Studio 2022 Native Tools PowerShell with the MSVC C++ workload and Windows SDK.
  The measured setup used MSVC 19.44.35229 and SDK 10.0.26100.0. Use the compiler installed on
  your machine and qualify it from that shell.
- Git, uv 0.13.7 or later and Python 3.13. The measured versions were uv 0.13.7 and Python
  3.13.18. The native uv group installs the locked Conan 2.34.0, CMake 4.4.4 and Ninja 1.13.2
  into the reference environment.
- Access to Conan Center or a local cache with the pinned recipes and their dependencies.
  Keep separate host and build profiles in Conan home.
- For the CUDA runs, the installed CUDA Toolkit, a compatible NVIDIA driver and a compute
  capability 12.0 device. The measured setup used CUDA Toolkit 13.4.1, nvcc 13.4.59, driver
  610.88 and a GeForce RTX 5060 Ti. Install Compute Sanitizer with the toolkit for the
  Release memcheck gate.

## CPU-only build on Windows

Run these commands from the repository root. The validator installs from both committed locks,
generates the fixture headers, configures and builds the preset, and runs CTest and Python.

```powershell
chcp.com 65001
$env:PYTHONUTF8 = '1'
uv sync --project reference --locked --group native
if ($LASTEXITCODE) { throw 'uv sync failed.' }
$uvArgs = @('run', '--project', 'reference', '--locked', '--group', 'native')
uv @uvArgs python tools/native/qualify_toolchain.py --mode cpu --output out/qualification/windows-cpu.json
if ($LASTEXITCODE) { throw 'CPU qualification failed.' }

foreach ($profile in @('fluidity-host', 'fluidity-build')) {
    uv @uvArgs conan profile detect --name $profile --force
    if ($LASTEXITCODE) { throw "Profile detection failed for $profile." }
}
uv @uvArgs conan profile show -pr:h fluidity-host -pr:b fluidity-build
if ($LASTEXITCODE) { throw 'Profile inspection failed.' }

foreach ($preset in @('windows-debug-cpu', 'windows-release-cpu')) {
    uv @uvArgs python tools/native/validate.py --preset $preset --host-profile fluidity-host `
        --build-profile fluidity-build --output "out/validation/$preset"
    if ($LASTEXITCODE) { throw "Validation failed for $preset." }
}
```

Inspect the profile output for Windows, x86_64 and the detected MSVC toolset. Profile detection
overwrites these two profile names. The validator supplies C++20, the dynamic CRT and the
selected configuration in the host context. The build context stays on Release and its
dynamic Release CRT. Repeating the command uses the same build tree for an incremental run.
Keep concurrent invocations in separate preset trees with distinct evidence directories.

For manual Conan install and CMake commands, including the explicit ABI flags, see the
[toolchain matrix reproduction commands](toolchains.md#reproduce-on-windows). The lock pins
recipe revisions and `--build=missing` permits building a binary from a pinned recipe.

## CUDA hardware and sanitizer gates

Use the same shell and profiles after the CPU setup. The CUDA qualification receipt is
separate. Both presets run the two real `gpu` tests and the `cuda-mock` API-double tests.

```powershell
uv @uvArgs python tools/native/qualify_toolchain.py --mode cuda --output out/qualification/windows.json
if ($LASTEXITCODE) { throw 'CUDA qualification failed.' }

foreach ($preset in @('windows-debug-cuda', 'windows-release-cuda')) {
    uv @uvArgs python tools/native/validate.py --preset $preset --host-profile fluidity-host `
        --build-profile fluidity-build --output "out/validation/$preset"
    if ($LASTEXITCODE) { throw "Validation failed for $preset." }
}
```

Release invokes the qualified toolkit's Compute Sanitizer on `CudaProbe.SharedVec2Math`
when installed. The validator sets the named-pipe connection override in the sanitizer
process environment and bounds launch and execution time. A passing result requires the
real launch receipt, one completed test and a zero-error memcheck. An unavailable
sanitizer leaves that gate at `not_run` with reason `unavailable`.

For the equivalent manual memcheck, keep the connection setting scoped to the command.

```powershell
$sanitizer = Join-Path $env:CUDA_PATH 'compute-sanitizer/compute-sanitizer.exe'
$previousConnection = $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE
try {
    $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
    & $sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 `
        out/build/windows-release-cuda/fluidity_cuda_probe_tests.exe --gtest_filter=CudaProbe.SharedVec2Math
    if ($LASTEXITCODE) { throw 'Release memcheck failed.' }
} finally {
    $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = $previousConnection
}
```

## Fixtures and evidence

CMake exports `affine_apic.json`, `two_cell_pressure.json`, `compliant_pressure.json` and
`stiff_viscosity.json` into `out/fixtures/<preset>/` with a SHA256 manifest. It validates the JSON
on each build and regenerates the test header at `out/build/<preset>/generated/fixture_data.h`
when the exported values, tolerances or provenance change. The transfer, pressure and viscosity
tests use the exported expected values and compare against the fixture's absolute plus
relative tolerance. To restore a missing or modified export, run this command for the preset.

```powershell
uv @uvArgs python -m fluidity_reference.cli export "out/fixtures/$preset" --overwrite
if ($LASTEXITCODE) { throw 'Fixture export failed.' }
```

Run the validator again after restoring the files. A successful run writes `summary.json`
and path-free CTest reports under `--output`. The receipt records the source revision and
working source hash, both lock hashes, profile and effective ABI hashes, tool versions,
fixture and header hashes, commands and separate results for Python, native CPU, GPU,
CUDA doubles, sanitizer and proofs. The proof gate remains `not_run` in this validator.
The validator clears inherited GoogleTest and pytest selectors. It checks each named native
case's execution output and compares the Python results with the complete pytest collection.

Full command arguments, tool output, Python and CTest XML, and the live requalification
receipt stay under `out/native-validation/<preset>/run-*/`. These local files can contain
machine paths. An invocation with one valid `--output` destination clears its previous reports,
including when another command-line argument is invalid.
On failure, read the named stage's log and CTest report if the suite ran. Requalify after
compiler, SDK, toolkit, driver or device changes. If CMake cached a different compiler,
start with a fresh preset build tree. A localized MSVC include scan requires code page
65001, and Debug requires a working MSVC program database service.

## Hosted CPU checks and remaining M0 work

The [Native CPU workflow](../../.github/workflows/native-cpu.yml) targets Windows 2022 x64,
Ubuntu 24.04 with GCC 13, and macOS 15 ARM64 with Xcode 16.4. It runs Debug and Release
Ninja presets with the same committed locks. The matrix has separate compiler- and
configuration-keyed Conan caches and uploads the path-free receipt and CTest report.
The measured Windows hardware runs are documented in the [toolchain matrix](toolchains.md).
Hosted CPU checks leave the CUDA, sanitizer and proof results at `not_run`.

M0 still requires the Blender reader, uniform MAC CUDA transfer and pressure spikes,
CUDA Graph execution and the 2D adaptive-basis spike, each with measured results.
Proof pilots and later formal contracts use the separate [Lean workflow](../../proofs/README.md).
A release license decision remains open. See the [design milestones and evidence
requirements](../superpowers/specs/2026-11-07-fluidity-design.md#15-validation-milestones-and-review-boundary).
