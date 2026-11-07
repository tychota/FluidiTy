# Native Foundation and Windows CUDA Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a locked C++20 build, an independently checked CPU operator harness, and a real Windows CUDA compile/device gate before implementing GPU simulation kernels.

**Architecture:** Conan 2 supplies host dependencies; CMake presets and Ninja build a small native library and GoogleTest executables. Python exports and validates the existing JSON fixtures, then generates typed test data without computing new expected values. Serial FP64 native operators consume those cases; a separate CUDA smoke target validates compiler, device, shared headers and error propagation.

**Tech Stack:** C++20, CMake, Conan 2, Ninja, Abseil Status/StatusOr, GoogleTest, Python 3.13 with uv; optional CUDA C++20 with the vendor-supported Windows host compiler and an actual NVIDIA GPU.

**Spec:** `docs/superpowers/specs/2026-11-07-fluidity-design.md`, especially §§4–7, 13.1, 14.3 and 15; numerical oracle `reference/`; formal boundaries `docs/references/formal-contracts.md`.

**Handoff:** Written on macOS at baseline `500511d`. This commit contains a plan only. The owner will push and execute on Windows with CUDA. No Windows compiler, native dependency graph, CUDA launch or GPU performance has been validated by writing this plan. Begin Windows execution on a new `feat/native-foundation` branch and isolated worktree from the pushed plan commit; recreate the uv environment, never copy the macOS `.venv` or temporary paths.

## Global Constraints

- Production uses C++20, CUDA on Windows/Linux and a CPU backend on macOS.
- One staggered MAC grid for all velocity components and continua.
- All public technical prose and identifiers are English.
- No host exceptions: Status/StatusOr; device status bits are decoded at a safe boundary. Abseil is host-only.
- No modules, coroutines, production RTTI, intricate metaprogramming or `using namespace`; no production BLAS dependency.
- Google C++ style with mandatory braces, `#pragma once`,100 columns. `.h` is portable host/device, `.cuh` requires nvcc, `.cu` kernels, `.cc` host. One `FLUIDITY_HD` definition.
- No test is reported passing because it was skipped or only a CPU placeholder ran.
- Pin concrete compatible compiler/toolkit/library versions in the plan and CI from actual builds; do not guess a universal version guarantee here.
- A missing or skipped test is not a success. Preserve the four distinct evidence classes: proofs, independent reference, native differential/runtime checks, and target-device quality/performance.
- No third-party source, proprietary binaries, personal paths, downloaded papers or model weights committed. Review logs remain ignored under `docs/reviews/`.

## Scope and selected approach

This is the next bounded part of **M0**, not M1 and not completion of M0. Deliver serial, allocation-bounded, tiny FP64 operators and a CUDA readiness gate. Production sparse storage, i64 deposition/quantization, GPU transfer/pressure kernels, CUDA Graphs, 3D scenes, CPIC/ghosts, AMR, resampling, material models, Blender readers and mesh reconstruction are later plans. Existing Python resampling and transaction references remain Python references in this delivery. No new Lean theorem is claimed here.

The CPU dense solve is an oracle for at most **64 unknowns**, deliberately unsuitable for the production pressure grid. It provides independent tiny-system checks without Eigen, BLAS or a production multigrid decision. A successful CUDA vector/math probe establishes readiness only; it is not a simulator benchmark or CPU/CUDA solver parity result.

Keep the dependency graph small: host Abseil and test-only GoogleTest. Defer OpenVDB, Alembic, FlatBuffers and Benchmark until a deliverable consumes them. Generate typed fixture headers with Python's standard library instead of introducing a native JSON/hash dependency. Generated expectations come directly from the validated JSON; the generator must never call a numerical solver.

## Review Focus

1. Windows paths with spaces and mixed Debug/Release CRTs: use argument arrays, isolated presets and matching Conan runtime settings; detect mismatches before linking (Tasks 1, 9).
2. Missing CUDA, an unsupported compiler/architecture, or no usable GPU: CPU presets remain usable; CUDA-required configuration/test fails explicitly, never skips to green (Tasks 1, 8).
3. Missing/stale/tampered fixtures and wrongly shaped numeric data: validate manifest hashes, schema, extents, finite values, geometry and tolerances before emitting or consuming C++ data (Task 2).
4. Tiny/singular/nonsymmetric moments, overflow and incomplete particle support: reject before unsafe indexing/solving; outputs remain unchanged on failure (Tasks 3–7).
5. Sealed/disconnected pressure regions and stiff/moving-wall viscosity: preserve physical compatibility, gauges, residual scaling and the shared mass-metric stability condition (Tasks 6–7).

## Tool and dependency decisions

These are **selected published pins to qualify**, not an already tested Windows tuple:

| Component | Selection and qualification rule |
|---|---|
| Python / uv | Existing Python 3.13, uv >=0.13.7, committed `reference/uv.lock` |
| Native build tools | Add `conan==2.34.0`, `cmake==4.4.4`, `ninja==1.13.2` to a `native` dependency group in `reference/pyproject.toml`; resolve/commit the uv lock on Windows |
| Host libraries | `abseil/20250814.2`; test requirement `gtest/1.17.0`; capture actual recipe revisions and transitive build requirements in `conan.lock` |
| Windows compiler | x64 MSVC from a supported Visual Studio installation; record full `cl /Bv`, toolset and Conan compiler settings. Use the CUDA version's support table plus an actual compile; do not assume every newer MSVC is supported |
| CUDA | Installed toolkit that compiles C++20 and `sm_120`; CUDA 13.4.2 is a documented candidate, not an installer mandate. Record exact toolkit, host compiler, driver and device after the real probe |
| GPU code generation | Reference RTX 5060 Ti is compute capability 12.0: explicit `120-real;120-virtual` in its CUDA preset; no default to Ada `89` or datacenter Blackwell `100` |
| Windows CRT | Conan `compiler.runtime=dynamic`; Debug uses `runtime_type=Debug`, Release uses `Release`. Match CMake's DLL CRT selection; CUDA runtime is a separate explicitly selected Shared runtime |

A failed qualification blocks that mode with an actionable failure. CPU qualification allows CPU tasks to proceed while an unavailable CUDA gate stays open; the complete handoff still requires the real Task8 result. A replacement pin requires a recorded reason, updated locks and a successful probe; do not use `--allow-unsupported-compiler`, invent a successful receipt, or silently substitute a different toolchain. Record full local paths only in ignored evidence, not committed profiles. Conan lockfiles pin recipes; they do not alone prove binary reproducibility or pin the CUDA driver.

## File map

| Files | Responsibility |
|---|---|
| `CMakeLists.txt`, `CMakePresets.json`, `cmake/NativeOptions.cmake` | Targets, committed single-config presets, owned warning/FP/CRT policies |
| `conanfile.py`, `conan.lock` | Small locked dependency graph; CMakeDeps/CMakeToolchain generation |
| `tools/native/qualify_toolchain.py`, `tools/native/tests/test_toolchain.py` | Tool discovery, real qualification receipts and orchestration tests |
| `tools/native/compile_fixtures.py`, `tools/native/tests/test_compile_fixtures.py` | Strict existing-fixture validation and deterministic test-only header generation |
| `native/include/fluidity/base/{host_device,types}.h` | One annotation macro and trivial Vec2/Mat2 types |
| `native/include/fluidity/host/{views,grid,basis,small_solve,transfers,pressure,viscosity}.h` | Host-only checked APIs; no Abseil dependency in device math headers |
| `native/src/{grid,basis,small_solve,transfers,pressure,viscosity}.cc` | Serial CPU implementations |
| `native/tests/{test_fixture_bridge,test_grid_basis,test_small_solve,test_transfers,test_pressure,test_viscosity}.cc` | Analytic, generated-fixture and adversarial checks |
| `native/include/fluidity/host/cuda_probe.h`, `native/cuda/{probe_bridge.h,probe.cuh,probe.cu,probe_host.cc}`, `native/tests/test_cuda_probe.cc` | Real GPU readiness and checked runtime errors |
| `tools/native/validate.py`, `.github/workflows/native-cpu.yml` | Local validation/evidence and hosted CPU CI |
| `docs/native/{README,toolchains}.md`, root `README.md` | Windows handoff, qualification matrix and measured capability status |
| `reference/{pyproject.toml,uv.lock}` | Reproducible Python/native build tooling; numerical dependencies unchanged |

Generated JSON/headers, CMake/Conan files, receipts and test reports live under ignored `out/` or `artifacts/`. Ignore `CMakeUserPresets.json` if a developer creates one; committed presets do not depend on it. Do not ignore `conan.lock` or `uv.lock`.

## Shared interface contract

Namespace `fluidity`; host APIs use `absl::Status` or `absl::StatusOr<T>`. Invalid inputs/shapes/overlap return `kInvalidArgument`; unsupported tiny-oracle sizes return `kUnimplemented`; incompatible physics or singular solve returns `kFailedPrecondition`; nonfinite computed arithmetic returns `kOutOfRange`. Never publish partially computed numerical outputs on error. Scratch may change on failure and is explicitly nonpersistent.

`Vec2` is two doubles `x,y`; `Mat2` is row-major four doubles. `Grid2D` holds `uint32_t nx,ny`, `double h`, `Vec2 origin`; every checked entry point validates it and checked integer counts. Host views use `std::span`, device math uses trivial values/raw pointers; Abseil and host containers never enter a kernel. Numerical inputs are const views. Output and scratch spans must have declared extents and not overlap inputs or each other; validate byte ranges without undefined cross-allocation pointer comparisons. Zero-length views are allowed where the operation explicitly permits an empty particle batch.

The native numerical functions use caller-provided outputs/workspaces and no dynamic allocation in their numeric loops. Each public workspace/scratch query computes checked extents before access. Numeric output validation and staging are independent of test-harness allocations. Fixture/test executables may own standard containers. This is not a global heap-exhaustion/transactional allocator implementation. Production targets disable exceptions and RTTI; tests/third-party dependencies keep the settings they require. Do not blanket-apply project warning/exception flags to Conan dependencies or feed MSVC flags directly to nvcc.

## Commands used by every implementation task

After Task1 setup, all commands run from the repository root through uv's `native` group. Before changing behavior, add the task's named failing tests and run the focused command; verify a genuine failure. After implementation, build and run the same tests until green, then run the relevant existing tests once before committing.

```powershell
uv run --project reference --locked --group native pytest tools/native/tests/test_toolchain.py -q
uv run --project reference --locked --group native cmake --build --preset windows-debug-cpu
uv run --project reference --locked --group native ctest --preset windows-debug-cpu -R "^(Grid|Basis)\." --output-on-failure
uv run --project reference --locked --group native pytest reference/tests tools/tests tools/native/tests -q
```

Substitute the task's exact Python test file or GTest prefix (`Build`, `FixtureBridge`, `SmallSolve`, `Transfer`, `Pressure`, `Viscosity`) for focused runs. Repeat required native checks with Release. The first RED in Task1 uses the existing uv environment before the new group/module exists. Expected GREEN is exit0 with all requested tests executed, not a missing-target/no-tests result. Capture commands, exit codes and outputs in ignored evidence; commit each task separately.

## Task 1: Qualify and lock the native build

**Files:** Create `CMakeLists.txt`, `CMakePresets.json`, `cmake/NativeOptions.cmake`, `conanfile.py`, `conan.lock`, `tools/native/{__init__,qualify_toolchain}.py`, `tools/native/tests/test_toolchain.py`, `native/tests/test_build.cc`, `docs/native/toolchains.md`; modify reference tooling and `.gitignore`.

**Interfaces:** `qualify_toolchain.main()` accepts `--mode cpu|cuda --output PATH [--cuda-architecture 120]`; writes schema-version 1 evidence only after actual required checks pass. Receipt records executable/version/compiler/CRT/architecture, command exit codes, source hashes and probe output. Mocked process tests are explicitly not qualification evidence.

- [ ] **Step 1:** Write tests `test_missing_tool_fails`, `test_wrong_crt_fails`, `test_paths_with_spaces`, `test_failed_compile_removes_stale_success`; add native `Build.Cxx20StatusRoundTrip` asserting a successful `absl::StatusOr<int>(7)` and a distinct InvalidArgument failure. RED: run Python tests before implementation; then attempt the absent CMake preset and record failure.
- [ ] **Step 2:** Implement the qualification CLI and add the exact `native` uv group above. Run `uv sync --project reference --group native`, commit its resolved lock. From x64 Native Tools PowerShell, record MSVC/tool versions and compile temporary Debug/Release C++20 programs with pointer-width and CRT checks. Capture `cl /Bv` as part of a compile with an actual source file, rather than treating a no-input compiler invocation as a successful compile. CUDA mode additionally compiles/runs a temporary C++20 device probe with the selected architecture, without unsupported-compiler overrides. Commit a path-free qualification summary with exact measured versions; retain raw receipts under `out/qualification/`.
- [ ] **Step 3:** Add a Conan consumer recipe with `settings=(os,compiler,build_type,arch)`, the exact Abseil requirement and GoogleTest test requirement, CMakeDeps and CMakeToolchain. Construct `CMakeToolchain(self, generator="Ninja")`, set `tc.user_presets_path=False`; do not use `cmake_layout` in this simple consumer so `--output-folder out/conan/<preset>` places `conan_toolchain.cmake` directly there. Generate/commit a Release lock and extend it with Debug; ordinary installs require that lock. Use separate host/build profiles, never rely on an existing unrecorded default profile.
- [ ] **Step 4:** Add presets `windows-debug-cpu`, `windows-release-cpu`, `windows-debug-cuda`, `windows-release-cuda`, plus Debug/Release CPU presets for Linux and macOS. Use CMake minimum 3.31 and preset schema 6. All use Ninja, `out/build/<preset>`, matching toolchains under `out/conan/<preset>`, explicit build type and testing enabled. Resolved CXX/CUDA compiler paths and runtime settings must match the qualification receipt; otherwise requalify before building. Start `project(... LANGUAGES CXX)`; only CUDA presets enable CUDA. Build/test presets have the same names. CPU tests use label `cpu`; CUDA presets register a separate label `gpu`. Configure `noTestsAction=error`. Create an INTERFACE `fluidity_native_options` target for the later core policy, a `fluidity_native_tests` executable for the CPU GTests, and GoogleTest discovery with `cpu` labels. The compiled `fluidity_native` library begins in Task3; no empty placeholder library is needed. Add core-only C++20/warnings-as-errors/no RTTI/no exceptions/precise FP policy, retain Conan CRT flags, and avoid duplicate conflicting MSVC defaults. Real Debug and Release Windows builds must pass `Build.Cxx20StatusRoundTrip`; run focused qualification tests and `git diff --check`.
- [ ] **Step 5:** Commit `build(native): qualify Windows toolchain and lock Conan dependencies`. A missing compiler/device leaves its gate incomplete; no empty lockfile, guessed recipe revision or claimed CUDA success is acceptable.

## Task 2: Validate and compile existing reference fixtures

**Files:** Create `tools/native/compile_fixtures.py`, its tests and `native/tests/test_fixture_bridge.cc`; modify CMake and native instructions.

**Interfaces:** `compile_fixtures.main()` takes `--input DIRECTORY --output HEADER`. Produce test-only `fluidity::fixtures` records: `ArrayRecord(name,dtype,shape,f64_data,i64_data,units)`, `FixtureRecord(name,arrays,rtol,atol,provenance,default_policy)`, and `AllFixtures()->std::span<const FixtureRecord>`. DType is `kFloat64|kInt64`; metadata strings are emitted as escaped data, never identifiers/code. `FindArray(const FixtureRecord&,std::string_view name)->absl::StatusOr<const ArrayRecord*>` returns a checked pointer into static generated data, never a StatusOr reference.

- [ ] **Step 1:** Tests `test_manifest_tamper_rejected`, `test_missing_field_rejected`, `test_wrong_dtype_or_shape_rejected`, `test_nonfinite_or_bad_tolerance_rejected`, `test_deterministic_generation`, `test_quote_backslash_and_space_paths`. Reject unknown schema versions, duplicate JSON keys, mismatched names/file basenames, unsafe manifest paths, absent/extra fixture files and malformed SHA256. RED: focused Python tests and missing native bridge target.
- [ ] **Step 2:** Validate exact source bytes against schema1 manifest hashes before parsing. Require shape/product/nested-data consistency, finite numeric data, declared dtype, units, positive spacing, finite origin, finite nonnegative rtol/atol, and nonempty derivation/default policy. Support the four existing fixtures without changing their numbers or their dimensionless unit convention. Header generation copies literals and metadata only; use exact binary64 literals and checked int64 literals, with deterministic order and safe string escaping. Do not call NumPy/SciPy/reference solve functions from the generator.
- [ ] **Step 3:** CMake finds Python3 3.13 (reject >=3.14) from the uv environment. Export JSON into `out/fixtures/` with the existing CLI, compile the header into `out/build/<preset>/generated/`, and list exporter/schema/generator/lock sources as build dependencies. Do not download tools or run `uv sync` during CMake configure. Regeneration removes/replaces only its own outputs. Add a build dependency so native fixture tests cannot compile against missing/stale data.
- [ ] **Step 4:** `FixtureBridge.FourFixturesPresent` checks the exact four names, valid typed extents and metadata. A changed JSON byte with an unchanged manifest must fail generation; corrupt data must never leave a usable stale header. Verify repeated generation is byte-identical and changing a valid input invalidates the build dependency. Native tests compile generated strings containing quotes/backslashes. Run Python generator tests and native bridge CTest; no skipped fixture.
- [ ] **Step 5:** Commit `test(native): bridge hashed Python fixtures into typed C++ cases`.

## Task 3: Define portable math, MAC geometry and quadratic stencils

**Files:** Create `native/include/fluidity/base/{host_device,types}.h`, `native/include/fluidity/host/{views,grid,basis}.h`, `native/src/{grid,basis}.cc`, `native/tests/test_grid_basis.cc`; modify CMake.

**Interfaces:** `ValidateGrid(const Grid2D&)->Status`; checked `CellCount/FaceCount(const Grid2D&)->StatusOr<size_t>`; `FaceGeometry(grid,index)->StatusOr<FaceInfo>` with position, physical unit-depth area, distance, tail/head cell indices (a missing side has sentinel -1). `QuadraticStencil2D(Vec2 relative_position,double h,Vec2 offset)->StatusOr<Stencil2D>` containing exactly nine integer nodes, weights and physical gradients. Node ordinal is `3*ix+iy`, matching Python's product order. Declare checked view extents/overlap helpers in `views.h`. Validate generated positive finite cell volume, area and distance too; arithmetic underflow to a zero geometric measure or size overflow returns OutOfRange.

- [ ] **Step 1:** Tests: `Grid.TwoCellOrientation` checks 2x1,h1 gives two cells/seven faces, internal incidence(+1,-1), low boundary−1/high+1. `Grid.NonunitOriginAndDistance` checks h=.25,origin(-2,3), boundary distance h/2, internal h, areas h and volumes h² at unit depth. `Basis.Moments` checks sumw1, first moment0, derivative sum0, derivative linear momentI and D=h²I/4. Add negative coordinates, exact half-cell ties, invalid counts/h, integer-count overflow and computed nonfinite geometry; no huge allocation. RED: named GTests fail before code.
- [ ] **Step 2:** Implement x faces first in row-major `(ny,nx+1)`, then y `(ny+1,nx)`, cells `j*nx+i`. Validate products before casts/indexing. Basis uses `base=floor(position/h−offset−.5)` and tensor quadratic B-splines; never truncate negative coordinates. Require node indices representable before conversion. FaceInfo derives the same outward incidence everywhere, without constructing a second velocity layout.
- [ ] **Step 3:** Create the compiled `fluidity_native` library, link host Abseil and its owned compile policy, and add it to the existing CPU test executable. Define the sole `FLUIDITY_HD` macro in `host_device.h`; keep only trivial math there. Host headers explicitly separate checked StatusOr/view operations from device-callable math. Ensure failed extent/overlap checks leave caller numerical output buffers unchanged.
- [ ] **Step 4:** Run `Grid.*`, `Basis.*`, view validation tests and existing fixture bridge tests in both Windows Debug/Release. Acceptance tolerances: moment errors <=1e-12 after h nondimensionalization, exact integer ordering/signs. Existing 100 Python tests remain green.
- [ ] **Step 5:** Commit `feat(native): add checked MAC geometry and quadratic basis`.

## Task 4: Add the bounded independent SPD solver

**Files:** Create `native/include/fluidity/host/small_solve.h`, `native/src/small_solve.cc`, `native/tests/test_small_solve.cc`.

**Interfaces:** `kMaxDenseUnknowns=64`; `SmallSolveWorkspace` owns fixed-size factor/rhs working arrays. `SolveSmallSpd(span<const double> row_major_matrix,span<const double> rhs,span<double> solution,SmallSolveWorkspace&)->Status`, n=rhs.size, exact extents n*n/n; n=0 is identity, n>64 Unimplemented. No BLAS/Eigen or hidden allocation.

- [ ] **Step 1:** `SmallSolve.HandTwoByTwo` solves [[2,−1],[−1,2]]*[x,y]=[1,0] as [2/3,1/3]. Add diagonal 1e-13 scaling, nonsymmetric [[1e-13,9e-13],[0,1e-13]], indefinite/singular/nonfinite, n65, output alias and unchanged-on-error tests. RED: run these tests before solver code.
- [ ] **Step 2:** Implement a deterministic dense Cholesky solve with scale-relative symmetry/pivot checks and finite arithmetic guards. If rounding-level asymmetry is allowed, symmetrize once and use that same matrix throughout. Reject unsafe/ill-conditioned pivots with FailedPrecondition rather than fabricating a solution. Stage results in workspace and publish only on success; workspace must not overlap input/output. Report no guarantee outside the documented finite/conditioning domain.
- [ ] **Step 3:** Add `SmallSolve.Residual` asserting `max|b−Ax| <=1e-12*max(1,max|b|)` on the regular hand systems; deliberately singular systems must fail, not return zero. No unit-scale absolute symmetry tolerance on tiny matrices.
- [ ] **Step 4:** Run focused solver tests and prior native tests in Debug/Release. Confirm compiled core has exceptions/RTTI disabled and no external linear-algebra link dependency.
- [ ] **Step 5:** Commit `feat(native): add bounded checked SPD oracle`.

## Task 5: Port complete-support FP64 APIC operators

**Files:** Create `native/include/fluidity/host/transfers.h`, `native/src/transfers.cc`, `native/tests/test_transfers.cc`.

**Interfaces:** `ParticleView2D` contains const spans positions, masses, velocities, affine with one common particle count. `P2G(grid,particles,face_mass,face_momentum,scratch)->Status`; `G2P(grid,positions,old_faces,new_faces,pic,increments,affine,gradient,scratch)->Status`; `FitAffine2D(position,samples,weights,values)->StatusOr<AffineFit2D(value,row)>`; `VelocityGradient2D(values,gradients)->StatusOr<Vec2>`; `BlendHistory2D(pic,residual,energy,phase,separation)->StatusOr<VelocityPair(stored,advecting)>`. `P2GScratchDoubles(const Grid2D&)->StatusOr<size_t>` returns twice the checked face count; `G2PScratchDoubles(size_t particle_count)->StatusOr<size_t>` returns checked 12*n doubles for staged PIC/increment/C/L data. Callers supply disjoint spans. Phase enum covers Liquid/Gas/Elastic/Granular/Snow only for this blend policy.

- [ ] **Step 1:** `Transfer.LocalAffineFixture` consumes the existing four-sample JSON case, asserting value3,row(2,3) with exported tolerances. `Transfer.StaggeredAffine` covers actual nine-sample MAC supports at nonunit h/nonzero origin; `Transfer.MassAndAngularMoments` checks each component mass separately. Add C≠L for nonlinear data, nonzero old/new increment, retained residual with identical grids, empty batches, incomplete support, singular fit, nonfinite inputs, insufficient scratch/alias and unchanged-on-error tests. RED before implementations.
- [ ] **Step 2:** Implement deterministic serial FP64 loops, centered weights and full moment fits matching current Python. For each component keep deposited mass and pure momentum distinct. Do not normalize absent faces or invent ghosts. Reject the whole operation before publishing if any particle lacks complete support. Zero particle masses follow the current oracle's nonnegative-mass policy; negative masses fail. Physical active-particle acceptance rules are a later integration contract.
- [ ] **Step 3:** Stored velocity=`PIC+alpha*residual`; advecting=`PIC+beta*alpha*residual`, alpha=.99e,beta=e*separation, e/separation∈[0,1]; Gas alpha=beta0; Elastic beta0. The caller's residual is old particle velocity minus old-grid gather; grid increment is not an extra blend term. `Gradient` uses true basis derivatives, not fitted C. Mark all these operators FP64 reference ports, not production quantized deposition.
- [ ] **Step 4:** Run all transfer tests and Python tests. Use exported tolerances for fixtures, scaled 1e-12 for regular analytic patches; do not loosen tolerances to conceal a sign/order bug. Record any conservative conditioning-domain rejection explicitly.
- [ ] **Step 5:** Commit `feat(native): add complete-support APIC reference operators`.

## Task 6: Port tiny pressure assembly, compatibility and correction

**Files:** Create `native/include/fluidity/host/pressure.h`, `native/src/pressure.cc`, `native/tests/test_pressure.cc`; reuse grid and small solver.

**Interfaces:** `PressureInputsView` has grid,dt and exact const face spans velocity,aperture,mobility,distance,wall_velocity,dirichlet_mask (uint8 values0or1),dirichlet_pressure,jump; exact cell spans source,fluid_volume,bulk_modulus,previous_pressure. The C++ API is explicit: fixture adapters materialize Python's declared defaults. `PressureOutputsView` contains pressure,velocity,original_matrix (row-major n*n),rhs,residual,constraint_defect,gauges and a gauge_count set on success. `PressureWorkspace` is fixed-capacity for64cells and193faces (maximum `2*n+nx+ny` at n<=64); no core heap allocation. `SolvePressure2D(inputs,outputs,workspace)->Status`.

- [ ] **Step 1:** Consume both JSON pressure fixtures using exported spacing/origin and default policies: two-cell p=[0,5],corrected u=0; compliant p=[6],no gauge. Add both Dirichlet orientations,J=0 at capillary boundary, positive internal jump sign, disconnected gauges(0,1), incompatible Neumann flux/source, blocked moving-wall flux, tiny volume residual identity, invalid internal Dirichlet/jump, insufficient extents/overlap and unchanged outputs on failure. RED: focused pressure tests fail before code.
- [ ] **Step 2:** Independently assemble dense A,b from face adjacency: a=area*aperture; active=open internal or explicit boundary Dirichlet; k=dt*a*mobility/d; C=V/(K*dt). A=R diag(k)R.T+diag(C); b=V*s−R[a*u+(area−a)*wall]−R(k*jump)+Dirichlet lift+C*pstar. Boundary lift is signed by outward incidence. Other external velocities are prescribed flux. Allow positive infinity only for bulk modulus; all other coefficients finite and in their physical domains.
- [ ] **Step 3:** Connected components use positive internal k. Check each unanchored component's compatibility before gauging, using the Python reference's scaled 1e-12 policy; compliance/Dirichlet anchors prevent gauge. Pin lowest cell, row AND column, without altering original A,b. Solve with Task4; correct active faces by dt*mobility/d*(R.T*p+jump−signed external pressure). Compute true r=b−A*p and physical defect `(R[a*u_new+(area−a)*wall])/V−s+(p−pstar)/(K*dt)`; compare with −r/V. Never count both jump and capillary Dirichlet lift on one boundary.
- [ ] **Step 4:** Run regular/invalid pressure tests in both configs; compare original A,b as well as p/u/residual. Require the volume-scaled identity with declared case tolerances and no singular-solve warning/fabricated result. n>64 is explicitly Unimplemented. This does not implement production pressure preconditioning, AMR or a pressure-kernel/solvability Lean theorem.
- [ ] **Step 5:** Commit `feat(native): add tiny conservative pressure oracle`.

## Task 7: Port tensorial strain and tiny viscosity steps

**Files:** Create `native/include/fluidity/host/viscosity.h`, `native/src/viscosity.cc`, `native/tests/test_viscosity.cc`; reuse small solver.

**Interfaces:** `MacStrain2D(grid,mu,E,W)->Status` writes row-major E and W with normals then interior-vertex shear ordering. `ViscosityInputsView` has velocity,mass,row-major E,rows,W,boundary_strain,dt,mode,safety; n velocity DOFs<=64. `ViscousStep(inputs,outputs,workspace)->Status` returns velocity, method, spectral_bound, energy_before/after, wall_work,numerical_dissipation through a checked output record. Caller-owned arrays; no allocation; `MacStrainShape2D(grid)->StatusOr<StrainShape(rows,columns,coefficients)>` returns checked E/W extents and rejects more than64velocityunknowns. E may have more than64rows; all rectangular products are checked. `ViscosityWorkspace` owns bounded matrix/vector working storage.

- [ ] **Step 1:** `Viscosity.StiffFixture` consumes the existing JSON system and expects implicit velocity(+1/21,−1/21). Test translation/rotation null modes, Couette shear, K symmetry, normals/shear factors; mass(.01,1),K[[1,−1],[−1,1]] has row bound110,dt.1 forces implicit and forced-explicit fails; dt=.99*2/110 permits explicit. Add both moving-wall energy identities, zero mu/dt identity, nonfinite/negative coefficients, n>64, aliases and unchanged failure outputs. RED before code.
- [ ] **Step 2:** E normal rows du/dx,dv/dy at cells; Dxy=.5(du/dy+dv/dx) at interior vertices. W normal2*mu*h²,shear4*mu*h². K=E.T W E, load=E.T W*d_wall; M=diag(m). Pressure integration must supply its matching positive face inertia; this generic oracle does not substitute P2G mass automatically.
- [ ] **Step 3:** Bound S=M^-1/2 K M^-1/2 by max absolute row sum. Explicit iff dt*bound<=2*(1−safety), otherwise auto uses (M+dt*K)u_new=M*u+dt*load. No per-row mixed integration. BE: Δenergy+.5||du||M²+dt||E*u_new||W²=dt*u_new.T*load. Explicit uses old velocity for wall work and the opposite sign for increment dissipation in its identity. Validate finite assembled coefficients and outputs before publication.
- [ ] **Step 4:** Run fixture/analytic/adversarial viscosity tests and prior CPU suite in Debug/Release. Label the bound float64 reference evidence, not a directed-rounding production certificate; boundary strain remains algebraic, not a physical free-surface traction discretization.
- [ ] **Step 5:** Commit `feat(native): add tensorial viscosity reference and stability selection`.

## Task 8: Prove Windows CUDA readiness on the actual device

**Files:** Create `native/include/fluidity/host/cuda_probe.h`, `native/cuda/{probe_bridge.h,probe.cuh,probe.cu,probe_host.cc}`, `native/tests/test_cuda_probe.cc`; modify CUDA targets/presets and qualification docs.

**Interfaces:** `.cu` includes only device-safe base headers, the bridge and CUDA APIs. Public `host/cuda_probe.h` declares the report/wrapper without CUDA SDK types; `probe_bridge.h` declares `int LaunchProbeKernel(Vec2* device_output,size_t count,void* stream)` for the host wrapper and nvcc. `.cc` includes the CUDA host API and bridge, never a `.cuh`. `RunCudaProbe(device_index,ProbeReport*)->Status` is a host wrapper; `ProbeReport` records actual device name/CC, driver/runtime versions, compiled architecture (kernel-reported `__CUDA_ARCH__`), checked launch/copy/sync result and element count. CUDA calls return typed failure; the kernel never consumes Abseil or host spans.

- [ ] **Step 1:** `CudaProbe.SharedVec2Math` launches a kernel on17elements, deliberately not a whole block, exercises FLUIDITY_HD Vec2 arithmetic with exactly representable inputs and verifies every returned value plus a sentinel for out-of-bounds detection. Add invalid device and wrapper error-path tests with process/API doubles labeled nonhardware evidence. Before real implementation, CUDA-required target/test must fail, not report a skip.
- [ ] **Step 2:** CUDA presets require a toolkit and explicit architectures; no silent CPU fallback. A different local GPU requires an explicit architecture override and separate receipt; it does not qualify the reference RTX5060Ti. After conditional `enable_language(CUDA)`, call `find_package(CUDAToolkit REQUIRED)`. Build a separate `fluidity_cuda_probe` library and `fluidity_cuda_probe_tests` executable; link the host wrapper to `CUDA::cudart` and host Status dependencies, label its tests `gpu`. The CPU test executable never links CUDA. Set CUDA_STANDARD20, CUDA_STANDARD_REQUIRED, precise arithmetic/no fast math and Shared CUDA runtime. Match qualified MSVC/CRT; forward only intended host flags via nvcc. Check every CUDA return, immediate launch status and stream synchronization/copy status. All device discovery/initialization belongs inside probe execution, never static initialization or GTest discovery. Use RAII cleanup that cannot throw; preserve output report on failure except a separate diagnostic record.
- [ ] **Step 3:** Run on the Windows NVIDIA device: `ctest --preset windows-release-cuda -L gpu --output-on-failure`. Zero devices, insufficient driver, wrong architecture or failed kernel is a nonzero result. Run Compute Sanitizer memcheck on the probe executable; preserve exit codes/device receipt. A sanitizer unavailable on that platform leaves that check open, not passed. Validate both GPU result and the CPU-labelled operator suite in the CUDA build.
- [ ] **Step 4:** Record exact real compiler/toolkit/driver/device, architecture flags and successful result in the qualification matrix. CPU-only hosted CI/macOS cannot substitute for this gate. This kernel is an infrastructure probe, not a GPU APIC/pressure implementation or throughput claim.
- [ ] **Step 5:** Commit `test(cuda): qualify Windows device and shared math headers`.

## Task 9: Automate CPU validation and publish the Windows handoff

**Files:** Create `tools/native/validate.py`, `tools/native/tests/test_validate.py`, `.github/workflows/native-cpu.yml`, `docs/native/README.md`; update root README, toolchain matrix and ignore rules.

**Interfaces:** `validate.py --preset NAME --host-profile NAME --build-profile NAME --output DIRECTORY` runs lock/tool qualification checks, Conan install from the committed lock with the preset's explicit ABI settings, fixture generation, configure/build/CTest and writes evidence only on success. Evidence separates Python tests, native CPU tests, real GPU probe, sanitizer and proof results; absent evidence is `not_run`, never `passed`. It records source/lock/profile hashes and actual toolchain/commands without publishing personal paths.

- [ ] **Step 1:** Tests wrong preset, stale qualification/lock, failed build/test, missing fixture and CUDA-required/no-device all fail with no stale success receipt; verify subprocess argument lists preserve spaces. Process doubles do not certify a compiler/device. RED before implementation.
- [ ] **Step 2:** Add hosted Windows/Linux/macOS CPU CI with Debug/Release (Ninja) and exact uv-native tool pins, locked Conan dependencies and separate profiles/output trees. Pin third-party action commit SHAs from their official repositories when implementing the workflow. Cache keys include platform/compiler/config/lock hashes. Persist CTest/evidence artifacts; do not hide zero tests or nonzero statuses. Do not add public-PR execution on a self-hosted GPU runner in this plan; the GPU gate is a deliberate local Windows run.
- [ ] **Step 3:** Run an actual clean Windows CPU configure/build/test and a second incremental run; change one generated-fixture input and verify regeneration. Run both CPU configs, full Python suite, real CUDA probe and available sanitizer. Confirm the four JSON examples are consumed by native CTest with their exported tolerances, not copied literal answers in native test logic. CPU CI success cannot mark CUDA/proof gates green.
- [ ] **Step 4:** Write `docs/native/README.md` with exact working PowerShell commands below, installed-tool prerequisites, measured qualification matrix, evidence locations, fixture generation, CPU-only mode, failure diagnostics and remaining M0 gates. No global tool installation, license choice or full-product completion implied. Run `git diff --check`, verify no generated files/secrets/personal paths are staged, and request a fresh whole-branch review after task-scoped reviews.
- [ ] **Step 5:** Commit `ci(native): validate locked CPU builds and document CUDA handoff`. Preserve the branch for owner review; do not push/merge automatically.

## Windows execution order and command contract

These commands are the intended **post-Task1** interface; they cannot work in the current plan-only checkout. Run from repository root in an x64 Native Tools PowerShell whose MSVC matches the selected CUDA toolkit. Do not copy a Visual Studio compiler directory from a different machine.

```powershell
uv sync --project reference --locked --group native
uv run --project reference --locked --group native python tools/native/qualify_toolchain.py --mode cuda --output out/qualification/windows.json
uv run --project reference --locked --group native conan profile detect --name fluidity-local --force
```

Inspect the detected profile and save its path-free settings/explicit corrections in the qualification summary. The local profile stays in Conan home. Native build and host contexts are both x64 for this initial build. Create/extend locks once in Task1, then commit them:

```powershell
uv run --project reference --locked --group native conan lock create . -pr:h fluidity-local -pr:b fluidity-local -s:h build_type=Release -s:h compiler.cppstd=20 -s:h compiler.runtime=dynamic -s:h compiler.runtime_type=Release -s:b build_type=Release -s:b compiler.cppstd=20 --lockfile-out=conan.lock
uv run --project reference --locked --group native conan lock create . -pr:h fluidity-local -pr:b fluidity-local -s:h build_type=Debug -s:h compiler.cppstd=20 -s:h compiler.runtime=dynamic -s:h compiler.runtime_type=Debug -s:b build_type=Release -s:b compiler.cppstd=20 --lockfile=conan.lock --lockfile-out=conan.lock
```

For a Debug CUDA consumer, explicitly set runtime settings if the detected profile does not already match. Install and build from the lock:

```powershell
uv run --project reference --locked --group native conan install . --output-folder out/conan/windows-debug-cuda -pr:h fluidity-local -pr:b fluidity-local -s:h build_type=Debug -s:h compiler.cppstd=20 -s:h compiler.runtime=dynamic -s:h compiler.runtime_type=Debug -s:b build_type=Release -s:b compiler.cppstd=20 --lockfile=conan.lock --build=missing
uv run --project reference --locked --group native cmake --preset windows-debug-cuda
uv run --project reference --locked --group native cmake --build --preset windows-debug-cuda
uv run --project reference --locked --group native ctest --preset windows-debug-cuda -L cpu --output-on-failure
uv run --project reference --locked --group native ctest --preset windows-debug-cuda -L gpu --output-on-failure
```

Repeat with the Release preset/output folder, `build_type=Release`, `runtime_type=Release`; CPU-only presets follow the same structure with CUDA disabled. Build-context tools remain Release. Lock creation and install commands above use the same host cppstd/runtime choices. Keep those choices consistent when materializing the qualified profile; do not lock a different ABI. Preserve remote configuration and dependency-license metadata; `--build=missing` permits rebuilding a locked recipe, not resolving an unlocked version.

## Acceptance and remaining plans

| Claim | Required evidence |
|---|---|
| Native foundation works on Windows | Qualified exact tools, locked Debug/Release builds, all CPU CTests, no skipped/missing fixtures, existing Python suite green |
| Tiny CPU operators match reference | Four generated JSON cases plus analytic/adversarial native tests, declared tolerance/conditioning domains and original-system residual checks |
| CUDA environment is ready | Actual sm120 compile/device math launch, checked statuses/copies/sync, device receipt and available sanitizer result |
| Production GPU solver exists | **Not delivered by this plan** |
| M0 is complete | **Not delivered**: GPU transfer/pressure/graph measurements, Blender reader and 2D adaptive-basis spike remain open |
| Full proofs or physical quality/performance are established | **Not delivered**; current Lean pilot scope unchanged |

The following plan should implement uniform CUDA transfers and tiny pressure differential tests using this harness, with explicit quantization/range/ordering contracts, capacity checks and actual GPU measurements. Blender-reader and adaptive-basis spikes should have separate bounded plans. Do not let a successful infrastructure probe promote any of those claims.

## Plan self-review

Spec coverage: this plan implements only the native build/CPU differential/CUDA-readiness subset of M0. It deliberately leaves all other milestones and the listed M0 spikes open. File ownership, Status codes, grid ordering, full moments, pressure signs/gauges/residual, viscosity metric and fixture defaults were checked against current source. Every Review Focus item has owning tests. Solver64unknown and pressure193face caps are explicit. No planned JSON/native linear-algebra dependency violates the approved family list. Task interfaces are serial dependencies; each task has focused RED/GREEN verification and its own commit. Tool pins are selected from published records and become measured supported tuples only through Task1/Task8. The owner requested planning and will implement on Windows; no implementation or qualification is performed by this document.

## Primary documentation checked while planning

- Published native-tool pins: [Conan2.34.0](https://pypi.org/project/conan/2.34.0/), [CMake4.4.4](https://pypi.org/project/cmake/4.4.4/), [Ninja1.13.2](https://pypi.org/project/ninja/1.13.2/). Existence is not Windows/CUDA qualification.

- [Conan CMakeToolchain and presets](https://docs.conan.io/2/reference/tools/cmake/cmaketoolchain.html): explicit output and user-presets control.
- [Conan multi-configuration locks](https://docs.conan.io/2/ci_tutorial/packages_pipeline/multi_configuration_lockfile.html): capture Debug/Release graphs, not guessed revisions.
- [Conan Center Abseil versions](https://github.com/conan-io/conan-center-index/blob/master/recipes/abseil/config.yml), [GoogleTest versions](https://github.com/conan-io/conan-center-index/blob/master/recipes/gtest/config.yml): selected recipe versions exist; no binary qualification inferred.
- [CMake enable_language](https://cmake.org/cmake/help/latest/command/enable_language.html), [CUDA architectures](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html), [CUDA runtime](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_RUNTIME_LIBRARY.html): enable CUDA deliberately; do not use `OPTIONAL` as a silent fallback.
- [CUDA 13.4 Windows installation/compiler support](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-installation-guide-microsoft-windows/index.html): use the toolkit-specific table and a real compile; drivers are installed separately in this toolkit generation.
- [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus): RTX5060Ti is12.0, hence sm120 for this reference device.
- [Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html): compatible cubin/PTX matters; distinct Blackwell targets are not interchangeable.
