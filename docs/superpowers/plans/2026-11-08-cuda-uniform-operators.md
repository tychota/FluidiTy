# Uniform CUDA Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure complete-support uniform 2D MAC CUDA transfers and tiny CUDA pressure parity against independent fixtures.

**Architecture:** Reuse the host geometry and quadratic basis to prepare bounded, checked POD inputs. CUDA executes the numerical transfers and a dense tiny pressure solve; the host stages all results through CUDA completion and cleanup before publishing. Keep the hardware-only executable and validation separate from the CPU oracle.

**Tech Stack:** C++20, CUDA, Abseil Status, GoogleTest, uv, Conan 2, CMake/Ninja.

**Spec:** [FluidiTy design](../specs/2026-11-07-fluidity-design.md) §§4–6, 11, 13, 15 and [F5/F8](../../references/formal-contracts.md).

## Global Constraints

- Work on `codex/cuda-uniform-operators` from verified `origin/main`. Preserve both existing worktrees. Preserve raw logs in ignored directories. Never commit paths, credentials, binaries or dependencies.
- Follow AGENTS.md and checked-in formatter, including the current 120-column limit. C++20, Google style, explicit braces, checked host Status/StatusOr, no Abseil in device code, no exceptions/RTTI/modules/coroutines in native implementation.
- Complete uniform quadratic stencils, x-normal faces then y-normal faces in row-major order, node y index varies fastest within each x slot. All nine indices per component must exist even for zero weights. SI units at unit depth.
- Windows CUDA uses `120-real;120-virtual` on the qualified compute capability 12.0 device. No unsupported compiler bypass, CPU fallback, skipped/absent GPU test as pass, or firewall changes.
- No production simulation, CUDA Graphs, Blender, AMR, pressure scalability or M0 completion claim. No integration action.
- Each task keeps RED/GREEN raw output, commits reviewed task and fixes separately. Focused CUDA verification runs on hardware, with full validation at the end.

## Design and numerical contracts

Host-prepared geometry reuses the carefully tested host tie and representability rules. This deliberately measures transfer arithmetic, synchronization and differential correctness while deferring GPU stencil generation and production sparse pipelines. Pure device geometry would widen this spike and risk a second near-tie policy. CPU-oracle-driven pressure correction alone would leave assembly, compatibility and solve untested, so the tiny device reference assembles, solves and corrects independently on device. The one-thread pressure path prioritizes inspectability and deterministic arithmetic.

**Transfer capacity:** at most 1024 particles and 4096 faces per call, inclusive; empty batches allowed. Reject excess as ResourceExhausted before allocation or launch. Host API selects CUDA ordinal 0, and every nonempty transfer and pressure call uses that device. Host input/output spans must have exact extents; read-only spans may alias; writable spans and output scalars must be disjoint from inputs and each other. Each call owns private buffers. A failure from validation, CUDA, device arithmetic, or cleanup preserves every output, including timing. Runtime errors map to Internal except no usable device (FailedPrecondition), and allocation failure (ResourceExhausted). Include an architecture marker from the operator kernel and event elapsed time; successful launches require an architecture match with device properties and finite nonnegative milliseconds. Empty G2P produces no outputs and no launch; empty P2G publishes zero faces after a real device reset/launch.

**P2G:** accept `ParticleView2D` plus nonnegative finite reference volumes and finite row-major Kirchhoff stresses (volume × stress × weight gradient gives force). Complete-support weights are the original basis; center the affine offsets using the weighted mean. Compute each particle–face contribution on CUDA in FP64, in order mass `N*m`, pure momentum `(N*m)*(v + C·r')`, force `-V0*(tau_row·gradN)`. Independently scale each channel by a positive normal binary power of two (binary exponents -1022 through 1023); require the decoded worst-case `ldexp(scale,62)` finite only if exposing decoded results, which this API does not. Return signed i64 totals; decode externally. Reject nonfinite actual or scaled contributions and any computed `abs(c/scale) > 2^52`, before nearest-even signed conversion. Since a particle visits a face at most once and 1024=2^10, sum of magnitudes is at most 2^62. Device atomic order is unspecified and exact integer sums are invariant under permutation. Accumulate via unsigned 64-bit atomicAdd of signed conversion modulo 2^64 and copy the final bit pattern back into signed words (checked two's-complement assumption). No saturation. Per-face, per-channel ideal rounding bound is count×scale/2, separately report FP arithmetic allowance. Range is evaluated on the computed FP64 scaled value; no exact-real claim at rounding boundaries. Guard the whole face contribution before writing its channels.

**G2P:** fixed slot order independently sum old and new PIC values, then subtract for increment, so identical inputs give exact zero. Fit the centered affine row with the measured symmetric moment and a scaled determinant test (smaller eigenvalue > 2 epsilon times larger). Keep true gradient as a separate derivative sum. FP64, no contraction except explicit FMA if needed, no fast math. Reject nonfinite intermediates as OutOfRange and unsafe moments as FailedPrecondition. No history blending, ghosts, extrapolation or partial support. Domain is well-scaled finite FP64 spike; reject nonrepresentable intermediates without publishing.

**Pressure:** at most 8 cells (25 faces), all quantities FP64. Use face geometry for oriented tail +1/head -1 and face measure, with supplied apertures, distances, inverse density, wall flux, oriented jumps, boundary pressures, source, fluid volume, bulk modulus and previous pressure. Derive compliance on device as V/(K*dt), with positive infinity selecting zero. Require validated physical inputs with ordinary representable arithmetic, positive normal active conductance/compliance, and no jump on Dirichlet. Iterate ascending face/cell order. Detect components on positive-conductance links; pin the lowest cell of each unanchored component after compatibility (absolute RHS sum tolerance `64*epsilon*sum(abs(rhs))`; zero-scale component requires exact zero). Do not pin compliant or Dirichlet components. Keep original A/b, solve the pinned system with dense FP64 Cholesky using positive pivots, recompute original residual and compliance-adjusted defect, correct only active faces with the same coefficients. Reject nonfinite intermediates as OutOfRange, incompatible components/unsafe solve as FailedPrecondition. Publish gauges in ascending order and leave unused gauge slots untouched. Differential scope is tiny well-scaled fixtures; capacity rejection is ResourceExhausted.

## Review Focus

- Exact ties still require zero-weight nodes and non-origin lattices retain correct placement (Task 1/2).
- Affine or stress force can breach range with zero centroid speed. Bound equality, signed ties, cancellation and reversed particles matter (Task 1).
- A failed later particle cannot expose earlier scratch, and aliased writable views are rejected (Tasks 1–3).
- Original residual and moving-wall/Dirichlet signs must survive assembly and gauge removal (Task 3).
- Absent/skipped hardware cases or incomplete sanitizer runs cannot generate a passing receipt (Task 4).

### Task 1: Bounded CUDA P2G and shared CUDA execution

**Files:** create `native/include/fluidity/host/cuda_measurement.h`, `native/include/fluidity/host/cuda_transfers.h`, `native/cuda/operator_runtime.h`, `native/cuda/transfer_bridge.h`, `native/cuda/transfer_host.cc`, `native/cuda/transfers.cu`, `native/tests/test_cuda_transfers.cc`; modify `CMakeLists.txt`.

**Interfaces:** In namespace fluidity, `CudaOperatorMeasurement` in `native/include/fluidity/host/cuda_measurement.h` has `int compiled_architecture; float kernel_ms;`. Add `struct CudaDepositScales { double mass; double pure_momentum; double stress_force; };` and `struct CudaDepositViews { std::span<std::int64_t> mass_quanta, pure_momentum_quanta, stress_force_quanta; };`. Entry point `absl::Status CudaP2G(const Grid2D& mac_grid, ParticleView2D particle_inputs, std::span<const double> particle_reference_volumes, std::span<const Mat2> particle_kirchhoff_stresses, CudaDepositScales quantization_scales, CudaDepositViews deposit_outputs, CudaOperatorMeasurement& kernel_measurement);`. Shared host runtime owns per-call CUDA buffers, stream and events and merges cleanup errors. Private transfer bridge defines POD per-component 9 slots with flat index, weight, gradient, and relative offset, plus a reusable host preparation helper for Task 2. Make a CUDA-only static library `fluidity_cuda_operators` and executable `fluidity_cuda_operator_tests` linked with native, CUDA runtime and GTest; hardware cases discovered with label `gpu`. Use existing probe flags including explicit architecture and precision, no CUDA dependency in CPU-only builds. Generated fixtures added to this target in Task 2.

- [ ] Write hardware tests first (`CudaTransfer.DepositAnalyticAndCpuParity`, `DepositRoundsContributionsBeforeOrderIndependentSum`, `DepositRejectsInvalidSupportRangeAndCapacity`, `DepositEmptyBatchClearsOutputs`). Use grid 4×4, h=1, origin (0,0), particle (1.5,1.5); at x-normal (1,1) weight .5×.75=.375 and gradient (-.75,0); unit mass, velocity (2,-3), zero affine, V0=2 and stress xx=4 give mass .375, momentum .75 and force +6 at that face. Compare all decoded mass/momentum with CPU oracle for nonzero affine and analytic force signs. Use force scale and momentum scale 1 for ties with exact chosen contributions, opposite signs, reversal, and bound 2^52 at an individual face. Ensure a later range or incomplete-support failure leaves sentinel outputs and timing intact. Cover nonbinary scale, overlapping spans, and oversize particles/faces. Print the Task 4 operator receipt for p2g.
- [ ] Observe hardware RED using an uncommitted stub that returns Unimplemented if needed to compile the public interface. Confirm the actual success assertions fail. Capture raw logs.
- [ ] Implement prepared stencils and device deposit. Check bounds before indexing, stage output buffers and measurement before commit, validate launch/sync/copy/cleanup. Reuse `QuadraticStencil2D`, grid functions, and overlap checks. CUDA threads use one particle each; both components visited in stable slot order.
- [ ] Build `windows-debug-cuda` through locked uv and run `ctest --preset windows-debug-cuda -R '^CudaTransfer\.Deposit' --output-on-failure --no-tests=error`. Require all four tests passing and run the baseline CPU/Python checks with the existing harness (full updated validator in Task 4).
- [ ] Commit `feat(cuda): add bounded quantized uniform deposition`.

### Task 2: Complete-support CUDA gather

**Files:** modify Task 1 transfer header, bridge, host, device, tests and `CMakeLists.txt`.

**Interfaces:** `struct CudaGatherViews { std::span<Vec2> pic_velocities, grid_velocity_increments; std::span<Mat2> velocity_fit_slopes, velocity_gradients; };`. Add `absl::Status CudaG2P(const Grid2D& mac_grid, std::span<const Vec2> particle_world_positions, std::span<const double> old_face_velocities, std::span<const double> new_face_velocities, CudaGatherViews gather_outputs, CudaOperatorMeasurement& kernel_measurement);`. Consume Task 1 checked per-component slots and private runtime. Link the operator test to `fluidity_fixture_headers` and include the generated directory.

- [ ] Add hardware tests `CudaTransfer.GatherAffineMatchesFixtureAndCpu`, `GatherNonlinearKeepsTrueGradientSeparate`, `GatherFailurePreservesOutputs`. Use typed `affine_apic` expected value/row for x face samples, embed its four axial samples at offsets about interior x-face center (2,1.5) and fill the remaining samples from the fixture affine field; apply actual quadratic MAC weights, and compare PIC, increment, affine and gradient with CPU for both components. Identical old/new field must give exact zero increment; asymmetric quadratic field must distinguish C from L, and compare both with CPU within 2e-12 plus 2e-12 relative. Test invalid nonfinite fields, incomplete support at a zero-weight endpoint, overlap, valid empty G2P, and an unrepresentable device intermediate. Print the Task 4 operator receipt for g2p.
- [ ] Observe the new tests fail on hardware through a local Unimplemented stub. Capture raw output.
- [ ] Implement one CUDA thread per particle. Reuse validation/staging from Task 1 and keep computed full centered moment and true gradient separate. Use status bits for nonfinite or singular arithmetic. Do not alter the CPU oracle.
- [ ] Build and run `ctest --preset windows-debug-cuda -R '^CudaTransfer\.' --output-on-failure --no-tests=error`; all seven hardware tests pass. Run CPU/Python baseline checks.
- [ ] Commit `feat(cuda): gather complete uniform MAC stencils`.

### Task 3: Tiny CUDA pressure differential

**Files:** create `native/include/fluidity/host/cuda_pressure.h`, `native/cuda/pressure_bridge.h`, `native/cuda/pressure_host.cc`, `native/cuda/pressure.cu`, `native/tests/test_cuda_pressure.cc`; modify `CMakeLists.txt`.

**Interfaces:** `absl::Status CudaSolvePressure2D(const PressureInputsView& pressure_inputs, PressureOutputsView pressure_outputs, CudaOperatorMeasurement& kernel_measurement);`. Consume Task 1 runtime; use Task 2 fixture dependency. Inputs/outputs retain the native pressure units, layout, and extents; no caller scratch. Host packages bounded POD, validates geometry, extents, aliases and physical domains. CUDA runs a deterministic one-thread dense reference with private fixed-capacity workspace.

- [ ] Add hardware tests `CudaPressure.NeumannJumpMatchesFixture`, `ComplianceRetainsMeanPressure`, `DirichletAndWallFluxMatchAnalyticAndCpu`, `CapacityAndIncompatibilityPreserveOutputs`. Use generated `two_cell_pressure` and `compliant_pressure` typed expected values and independent CPU comparison of original A, b, velocity, gauges, residual and defect; fixture tolerance or 2e-12 absolute + 2e-12 relative for CPU comparisons. Add 1×1 h=1, dt=1 case with two x faces, low Dirichlet p=3 with aperture .5 and distance .5, high wall velocity +1 with aperture 0, sealed y, no source/compliance; A=[1], b=[2], p=2, low face corrected velocity +2, no gauges, residual and defect zero. Test disconnected compatible gauges, disconnected incompatible sources with zero global sum, rejected 9-cell input, invalid boundary jump and unchanged outputs including gauge_count. Print the Task 4 operator receipts for pressure_neumann and pressure_compliance.
- [ ] Run the focused pressure tests with Unimplemented stub and retain expected RED on real CUDA preset.
- [ ] Implement full tiny pressure assembly/compatibility/Cholesky/correction on device, with staged host publication. Reuse only host face geometry and IO checks, never invoke the CPU solver inside CUDA API. Verify positive normal coefficients before division and finite intermediates.
- [ ] Build and run `ctest --preset windows-debug-cuda -R '^CudaPressure\.' --output-on-failure --no-tests=error` with all five tests passing. Run CPU/Python baseline checks.
- [ ] Commit `feat(cuda): add tiny pressure differential operator`.

Keep the pressure pending-error regression in its own `CudaPressure.LaunchIgnoresPriorRuntimeError` case.
CTest runs it on hardware. This case and the corresponding probe case intentionally trigger CUDA API errors,
so the Release memcheck invocations select the successful probe and the eleven numerical and validation operator cases.

### Task 4: Fail-closed validation and evidence

**Files:** modify `tools/native/validate.py`, `tools/native/tests/test_validate.py`, `docs/native/README.md`; record machine measurements and commands in ignored evidence; optionally record portable measured totals in `docs/native/toolchains.md`.

**Interfaces:** Consume the two existing probe and twelve operator hardware names verbatim and the same `gpu` label; preserve existing public JSON fields and add an operator measurement section. New tests print `operator=<p2g|g2p|pressure_neumann|pressure_compliance> compiled_architecture=<int> kernel_ms=<decimal>` exactly once in the matching case; add the receipt requirement by that name. Record elapsed event time as a tiny-kernel smoke measurement only.

- [ ] Write Python validation tests that reject an omitted CUDA operator, a skipped case, missing/duplicate/mismatched architecture receipt, negative/nonfinite timing and incomplete memcheck. Assert a successful report with all fourteen names and exactly four measurements; preserve existing probe receipt and CPU-only semantics. Observe RED via `uv run --project reference --locked --group native pytest tools/native/tests/test_validate.py -q`.
- [ ] Require all fourteen hardware names and all four operator receipts. For Windows Release extend named-pipe Compute Sanitizer to run the successful probe case and the eleven operator cases other than `CudaPressure.LaunchIgnoresPriorRuntimeError`. Require the exact case names, zero errors and the same four operator receipts. Keep 30-second launch and 90-second command timeout and fail closed. Explicitly report unavailability as a remaining gate if observed.
- [ ] Re-run focused Python tests and all four validator presets with locked profiles and fresh evidence directories. Confirm CUDA probes and operators on real hardware, preserve Compute Sanitizer logs, run cuobjdump for sm_120 cubin and PTX and format-check touched native sources. Document reproduction commands and result boundaries.
- [ ] Commit `test(cuda): require uniform operator hardware evidence`.

## Whole-branch acceptance

Independent review from verified main to candidate tip. Fix findings with their own commits and re-review. Re-run the full four-preset validator on final HEAD, retaining receipts, timings, exact compiler/driver/device and lock revisions, and verify no untracked source or staged artifacts. Keep the branch, worktree, and ignored evidence available for review.
