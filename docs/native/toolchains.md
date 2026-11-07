# Native toolchains

## Windows qualification measured on 2026-11-07

| Component | Observed version or setting |
| --- | --- |
| Host target | Windows x64, 64-bit pointers |
| Visual Studio | Build Tools 2022 17.14.44, VC toolset 14.44.35207 |
| MSVC and Windows SDK | `cl` 19.44.35229, `_MSC_VER=1944`, `_MSC_FULL_VER=194435229`, SDK 10.0.26100.0 |
| Language and CRT | C++20 (`__cplusplus=202002`), Debug `/MDd`, Release `/MD` |
| Python and uv | CPython 3.13.18 x64, uv 0.13.7 (`316c827e0`) |
| Native build tools | Conan 2.34.0, CMake 4.4.4, Ninja 1.13.2 (`1.13.2.git.kitware.jobserver-pipe-1`) |
| CUDA SDK and compiler | Toolkit 13.4.1, nvcc 13.4.59 (`cuda_13.4.r13.4/compiler.38657139_0`) |
| Driver and device | NVIDIA KMD/NVML 610.88, GeForce RTX 5060 Ti, compute capability 12.0 |
| CUDA runtime | CMake `Shared`; driver and runtime APIs both returned 13030 (CUDA 13.3) |

The CPU qualification compiled and ran both CRT probes, with `cl /Bv` captured during each compilation.
The temporary CUDA qualification compiled C++20 SASS and PTX for architecture 120 and launched both
Debug and Release probes on the RTX 5060 Ti. Each returned 7 and reported device and kernel
architecture 120. Both CPU-only and CUDA-enabled presets built and passed the CPU
`Build.Cxx20StatusRoundTrip` test in Debug and Release. The temporary launch is separate from the
repository GPU test and Compute Sanitizer gates.

NVIDIA's Windows installation guide lists VS 2022 17.x with the `193x` shorthand. The installed
CUDA header accepts `1920 <= _MSC_VER < 1960`. The measured 1944 compiler compiled and ran the
device probe with the guard active. Qualification applies to this measured tuple.
[NVIDIA CUDA Windows installation guide](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-installation-guide-microsoft-windows/index.html)

nvcc 13.4 warns that its `--cudart=shared` spelling is deprecated in favor of `hybrid` on Windows.
The measured runtime came from the display driver. CUDA 13.x minor version compatibility supports
this launch on R610. CUDA 13.4 features that require R615 need a separate qualification.
[NVIDIA runtime version API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html),
[NVIDIA CUDA 13.4 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

The schema-version 1 receipts under `out/qualification/` record local executables, full version
output, compiled source hashes and command results. Keep the receipts with the local build.

## Windows CUDA repository probe measured on 2026-11-08

The repository probe used the Windows x64 toolchain above. The NVIDIA GeForce RTX 5060 Ti reported
compute capability 12.0 with driver 610.88. Both the driver and runtime APIs returned 13030. The
executed kernel reported `__CUDA_ARCH__=1200`. Both executables contained an `sm_120` cubin and PTX
generated for `compute_120`.

| Check | Debug CUDA | Release CUDA |
| --- | --- | --- |
| MSVC CRT | `/MDd` | `/MD` |
| CMake architectures | `120-real;120-virtual` | `120-real;120-virtual` |
| Hardware tests (`gpu`) | 2 passed | 2 passed |
| CPU-labeled tests | 123 passed | 123 passed |
| CUDA API doubles (`cuda-mock`) | 11 passed | 11 passed |
| Compute Sanitizer memcheck 2026.3.0.0 (build 38637409) | Release gate | 1 passed, 0 errors, exit 0 |

The `CudaProbe.SharedVec2Math` test verified all 17 `Vec2` results and the sentinel after a 32-thread launch.
Launch, stream synchronization and device-to-host copy each returned zero. A second hardware test
verified that the probe succeeds and preserves the pending error after an unrelated CUDA call fails.
The device build used C++20 with `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. CMake used the
`Shared` runtime setting and linked `CUDA::cudart` from the installed toolkit. The earlier
direct nvcc probe recorded the Windows `shared` deprecation described above.

The `gpu` label runs the device test. The `cuda-mock` and `cpu` labels record separate evidence.
The probe qualifies CUDA build and execution infrastructure and shared vector arithmetic.

## Locked dependencies and ABI

Conan Center at `https://center2.conan.io` resolved these recipe revisions for both Windows
configurations.

| Conan reference | Recipe revision |
| --- | --- |
| `abseil/20250814.2` | `10d8fe8d3fe5ef417d119e4becd655ec` |
| `gtest/1.17.0` | `43201cc96e5d35332662ce8890cceb27` |
| `cmake/4.4.4` (transitive build requirement) | `9009f1989569bdbc583e48ea6e132723` |

Both profiles detected `os=Windows`, `arch=x86_64`, `compiler=msvc` and `compiler.version=194`.
Use separate build and host profiles. The build context uses Release, C++20 and the dynamic Release
CRT. The host context uses C++20 and the dynamic CRT for its selected build type. Conan's lock
pins recipes. Package binaries, their local compiler and the installed driver have separate evidence.

## Reproduce on Windows

Run from the repository root in an x64 Native Tools PowerShell. Use console code page 65001 for both
configure and build so Ninja can parse localized MSVC include reports. CPU and CUDA qualification use distinct
receipts by default. To use a different receipt path, set the CMake cache variable
`FLUIDITY_TOOLCHAIN_RECEIPT` when configuring.

```powershell
chcp.com 65001
uv sync --project reference --locked --group native
$uvArgs = @('run', '--project', 'reference', '--locked', '--group', 'native')
uv @uvArgs python tools/native/qualify_toolchain.py --mode cpu --output out/qualification/windows-cpu.json
# For the CUDA presets, run this additional qualification.
uv @uvArgs python tools/native/qualify_toolchain.py --mode cuda --output out/qualification/windows.json
uv @uvArgs conan profile detect --name fluidity-build --force
uv @uvArgs conan profile detect --name fluidity-host --force
```

Review the detected profiles against the qualified compiler. Select Debug or Release and CPU or CUDA.
The Conan output folder must match the CMake preset name.

```powershell
$config = 'Debug'
$mode = 'cpu'
$preset = "windows-$($config.ToLowerInvariant())-$mode"
$profiles = @(
    '-pr:h', 'fluidity-host', '-pr:b', 'fluidity-build',
    '-s:h', "build_type=$config", '-s:h', 'compiler.cppstd=20',
    '-s:h', 'compiler.runtime=dynamic', '-s:h', "compiler.runtime_type=$config",
    '-s:b', 'build_type=Release', '-s:b', 'compiler.cppstd=20',
    '-s:b', 'compiler.runtime=dynamic', '-s:b', 'compiler.runtime_type=Release'
)
uv @uvArgs conan install . --output-folder "out/conan/$preset" @profiles --lockfile=conan.lock --build=missing
uv @uvArgs cmake --preset $preset
uv @uvArgs cmake --build --preset $preset
uv @uvArgs ctest --preset $preset -L cpu --output-on-failure
```

With the CUDA presets, run the hardware gate in both configurations and memcheck in Release.
Use a process-local override to select named pipes for the Windows sanitizer connection.

```powershell
uv @uvArgs ctest --preset windows-debug-cuda -L gpu --output-on-failure
uv @uvArgs ctest --preset windows-release-cuda -L gpu --output-on-failure
$sanitizer = Join-Path $env:CUDA_PATH 'compute-sanitizer/compute-sanitizer.exe'
$env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
& $sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 `
  out/build/windows-release-cuda/fluidity_cuda_probe_tests.exe --gtest_filter=CudaProbe.SharedVec2Math
```

For a different GPU, generate a separate receipt using `--cuda-architecture` and a distinct
`--output`. Configure with explicit `CMAKE_CUDA_ARCHITECTURES` and `FLUIDITY_TOOLCHAIN_RECEIPT`
overrides for that receipt. The hardware test compares the device name with the configured
receipt and checks that the device image matches the reported compute capability.

Requalify after changes to the compiler, toolkit, runtime or target device. Use a fresh CMake
configure after changing compilers. The Linux and macOS CPU presets use the same lock and
per-preset output folders with platform-appropriate host and build profiles. Their toolchain
versions are outside this Windows qualification.

## Reference fixtures in native tests

Configure through the locked uv environment so CMake finds its Python 3.13 interpreter. The
native build runs the existing exporter into `out/fixtures/<preset>/` on the first build and
after exporter or uv lock changes. Each preset has its own export directory and a test-only
header at `out/build/<preset>/generated/fixture_data.h`.

Each build checks the complete JSON set and its manifest hashes before compiling the tests.
An invalid or incomplete export fails the build and invalidates its generated header.
To restore the reference export, run the following command, then rebuild and rerun CTest.

```powershell
uv @uvArgs python -m fluidity_reference.cli export "out/fixtures/$preset" --overwrite
```
