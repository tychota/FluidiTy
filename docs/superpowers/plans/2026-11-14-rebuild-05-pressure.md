# Rebuild Pressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pressure module of the rebuild: where the liquid is (`ComputeDistanceToSurface`,
`FindLiquidSurface`, `GeometrySnapshot`), the sharp boundary between two liquids that do not mix (first
behind its exclusion, lifted in Task 42), air pockets, the open sea, the cell equations over face pieces, the
parallel solve with its ladder, the pieces the joint viscous solve and gas reuse, and the one stop rule, on CPU
and (compiled, not run) GPU. Owner answer A18 (Tasks 43 to 50): a blocking material's inertia and a sealed
tracked-gas blob's springiness are solved inside the liquid pressure solve, not handed over one step late.
Owner answer A15 (Tasks 54 to 61): when the solve's working buffers do not fit the GPU, the same solve runs batch by
batch with its working buffers in CPU memory (`SolveBatches`, shared with viscosity and gas), agreeing with the one-piece
solve within the solve's tolerance.

**Architecture:** Folder `native/fluidity/pressure/`, namespace `fluidity::pressure`. Each operation has one
public CPU-only header and private `_cpu.cc` / `_gpu.cu` files; every per-element formula lives in a shared
header that both the CPU loop and the GPU function call. All arrays are `grid::CellValues`, `FaceValues`,
`CornerValues` on the processor chosen by `RunOn`; working buffers live in one `PressureBuffers` that allocates once.
Every total is one partial per cell block in the grid's block order, then `math::TreeSum`.

**Tech Stack:** C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest), uv (Python
reference with NumPy/SciPy as the transfer reference already uses), Lean 4 + Mathlib (proof), Typst + CeTZ
(method paper), clang-format and clang-tidy 23.1 (the versions `pyproject.toml` pins).

**Spec:** `docs/superpowers/specs/modules/05-pressure-design.md` (the design: its section 3 is the contract),
and the programme design `docs/superpowers/specs/2026-11-13-production-rebuild-design.md` (sections 6, 7, 8,
8a). Both in the `main` checkout, read-only. Also read there: `modules/owner-answers.md`,
`modules/shared-interface.md`, section 3 of `02-grid-design.md`, 3.1 and 3.3 of `03-particles-design.md`, the
material-set rule of `04-transfer-design.md`, 3.2 of `01-contact-design.md`.

## Global Constraints

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes
  the `AGENTS.md` checklist; new public names listed in the commit message with a one-line reason; tests arrive
  with the code. A commit must pass every part of the gate that exists at that commit.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages carry no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no session link); the owner is the
  author. After committing, write the commit's note with `uv run python tools/add_session_note.py --session
  <session link> --models <implementer's model id>,claude-fable-5-2`
  (a git note in `refs/notes/claude-session`; owner answer A55, 2026-11-15).
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. The owner runs on
  hardware once, after all modules are written. Never claim otherwise.
- Host C++23, exceptions and RTTI off in the library; device and shared headers C++20 subset; a shared header
  never includes `<expected>`, `<string>` or anything that allocates. No Abseil.
- Device numerical flags: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Host: `-ffp-contract=off`,
  `/fp:precise`. No fast-math.
- Z-up, right-handed. Gravity points along `{0, 0, -1}`.
- Naming: no `3D` suffix, no `Cuda`/`Gpu`/`Device`/`host` in public names, no bare `bool` parameters, no
  abbreviations (MAC, PIC, APIC, P2G, G2P, EOS), no milestone codes. Function verbs: `Compute`, `Build`, `Check`,
  `Find`, `Apply`, `Solve`, `Read`, `Write`; anything else is listed for the owner's review.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are named `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure patterns apply to every task (`AGENTS.md`
  "Structure"; `specs/modules/plan-decisions.md` "Foundation after the final review"). *Two halves:* the public
  function lives in `<name>_cpu.cc` (checks, dispatch by `RunOn`, the CPU loop, and the GPU half called under
  `#if defined(FLUIDITY_HAS_GPU)`); the GPU half is `<name>_gpu.cu`, declared in the private `<name>_gpu.h`, returns
  `RunOnCallStatus`, which the CPU half maps with `ToBufferError` and then into the module's failure through one
  mapping function; it launches, waits and holds per-call GPU memory only through `base/gpu_launch.cuh`
  (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`). *Kernel wrappers:* every GPU function is a
  thin loop over `FLUIDITY_CPU_AND_GPU` per-item or per-thread functions in a shared header (the module's existing
  formula headers, or a `<name>_steps.h`); a `cpu` test (`<name>_steps_test.cc`) runs them for every thread block
  and thread of the launch, in launch order and in reverse (phase by phase where the GPU function waits), against
  the CPU loop: the main evidence until the hardware run. Integer atomics are allowed; floating-point atomics never.
  *Working buffers:* callers of `ComputeTreeSum`, `ComputeTreeDot`, `ComputeLargestMagnitude`,
  `BuildStableOrder`, `ComputeExclusivePrefixSum` and `BuildGroupsByDestination` pass the
  `SumBuffers` / `OrderBuffers` that the module's buffers own (A31: these functions have one form, which takes its
  buffers; a test builds its own with `BuildSumBuffers` / `BuildOrderBuffers`) (members `sum_buffers`, `order_buffers`, built once,
  sized from the buffers' limits), and `Build<Module>MemoryEstimate` counts their bytes with
  `ComputeSumBuffersSize` / `ComputeOrderBuffersSize`. *Errors:* a failed check writes nothing; after
  `kRunOnCallFailed` the outputs are unspecified and the run stops; `kOutOfMemory` comes only from building forms.
  *Same processor (A30):* buffers, working buffers and arrays passed together to one function live on one
  processor; this is a documented precondition (`Requires:`), checked by an assertion in debug builds, never an
  error value. **Amended 2026-11-14 (A30|A31):** one form per function taking its buffers; mixed processors a
  precondition.
  *Exactness:* discrete outputs are equal on CPU and GPU; floating-point results agree within this plan's
  tolerances, unless the function's header promises the same bits unless NaN (compared with
  `CheckSameBitsOrBothNaN`). *Tests:* every test links `fluidity_base_test_support` (`ToBits`,
  `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`, `CheckSameBitsOrBothNaN`, the fixed generators) and the
  module's test library made with `fluidity_add_test_library` (named in `LINKS`), and defines no copy of them;
  every `gpu` test passes its main outputs to `WriteResultDigest` so the hardware run's repeat check covers them.
  Labels: `cpu` for every test that runs without a GPU (the container's cpu step runs them too), `no_gpu` only for
  behaviour on a machine without a GPU, `gpu` only for tests that need the hardware. Every target is created
  through the helpers of `cmake/fluidity_targets.cmake` and every source file is listed in a CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate (`uv run python tools/check_commit.py`) runs at every
  commit: the CPU build and tests in the debug and the release preset, and the CUDA build in the container (its
  `cuda` step fails for a tracked `.cc` or `.cu` file that no build compiles). clang-format and clang-tidy are the
  versions `pyproject.toml` pins (23.1).

Pressure's own constraints (values from the design; never change one without the owner):

- **Section 3 is the contract.** Implement its declarations exactly. The few places this plan adds to a
  declaration are marked **Plan decision** in the task that makes them and repeated in the hand-over.
- **Errors:** every fallible function returns `std::expected<T, PressureFailure>`; shared formulas return
  `PressureError` (`kNone` on success). The failure reported is the one with the smallest step number, then the
  smallest `cell_index`, then the smallest `face_piece_index`. On failure no output of the call is valid.
  **Amended 2026-11-14 (Foundation as built):** a failed check writes nothing; after a failed GPU call
  (`kRunOnCallFailed`) the outputs are unspecified and the run stops.
  A `BufferError` or `GridError` from below becomes `kRunOnCallFailed` (or `kBadInput` for a size mismatch).
  **Amended 2026-11-15 (A58, reconcile item 31):** in a building function (`BuildPressureBuffers`,
  `BuildSolveBatches`, the only two) a `BufferError::kRunMemoryTooSmall` or a failed allocation becomes
  `PressureError::kOutOfMemory` (appended last, 16 values at the module's end), `magnitude` = bytes asked; no other
  function returns it, and out of memory is no longer a `kRunOnCallFailed`.
  `kWrongGridVersion` whenever an input's `grid_version()` or `block_count()` differs from the grid's.
- **One reserved block (A58; Foundation Tasks 17 to 19; amended 2026-11-15).** `BuildPressureBuffers` and
  `BuildSolveBatches` build on both processors (batched layout), so they take `RunMemory& run_memory, RunMemory&
  cpu_run_memory` as their last two parameters (`plan-decisions.md` item 3 and its same-day correction; in a CPU run
  the caller passes the same object twice). Every `where = kRunOn` buffer comes from `run_memory`
  (`BuildBuffer(run_on, size, run_memory)`, the `RunMemory&` forms of `BuildSumBuffers`, `BuildOrderBuffers`,
  `BuildGridBuffers`, `BuildGridBuilderBuffers`, `BuildBorderSearchBuffers`); every `where = kCpu` buffer
  (`kWholeRun` or `kOneStepTry`) from `cpu_run_memory`. `Requires: run_on == run_memory.run_on()`,
  `cpu_run_memory.run_on() == RunOn::kCpu` (debug assertions, A30). Every memory estimate row stands for `copies`
  buffers of `count x bytes_each` bytes each, so `ComputeRunMemoryBytes(estimate, counts, where)` is exactly what the
  build hands out from the block of that `where`. Tests that only need buffers use the test-only `PressureBuffersHeld
  BuildHeldPressureBuffers(const BufferSizes& sizes, RunOn run_on)` (Task 13, `fluidity_pressure_test_scenes`: a
  `RunMemory` of exactly that size and the buffers built into it, members `run_memory`, `buffers`, declared in that
  order so the buffers are destroyed first).
- **Constants:** `kMinLiquidShare = 0.05` (also the floor of `low_share`); curvature gradient floor `1e-8`;
  18 curvature samples at the cell's own width; distance cap `+2` cell widths, ~~grid-only cells `-2` cell widths~~
  (A64: not in the first version; later group at the end of this plan);
  `max_volume_error_rate = 2e-5 1/s`; `jump_error_fraction = 1e-3`; `max_iterations = 256`;
  `iterations_per_progress_read = 8`; weighted Jacobi weight `2/3`, one pass down and one up; direct solve of a
  blob once it has at most 32 coarse cells; `heat_ratio = 1.4`; `atmosphere_pressure = 101325 Pa`;
  `min_radius_in_cells = 2`; air-pocket step limit `0.5 * radius * sqrt(rho / (3 heat_ratio pressure))`;
  volume correction by 64 bisections; open-sea wave number by 64 bisections.
- **Order of sums (design step 20):** a dot product, norm or report total is one partial per cell block (values
  of that block's places in ascending index, added left to right), then `math::TreeSum` over the block partials
  in the grid's block order. Every per-cell sum (one cell equation, one coarse cell) is done by one thread: own
  term first, then members or connections in ascending index. No floating-point atomics. **Per-group totals
  (amended 2026-11-14, A15):** a blob's, sealed gas blob's, free solid's or air pocket's total is, per cell block,
  the sum of the group's cells (or connections) of that block in ascending index, then `math::TreeSum` over the
  group's block partials in ascending block index. This replaces "one thread per blob over its cells or
  connections" wherever Tasks 24, 32, 44 (b), 46, 47 and 48 say it, in one piece as in batches.
- **Batched solves (A15, Tasks 54 to 61; `plan-decisions.md` "Batched solves (A15)").** Buffers built with
  `BufferSizes::max_loaded_blocks > 0` keep their per-place rows in CPU memory (`CpuMemory::kLockedForCopies`) and
  every function taking `PressureBuffers&` then runs batch by batch through `SolveBatches`; no signature changes.
  A one-piece run is the batched algorithm with one batch: the arithmetic of Task 58 is the arithmetic of both
  modes, and Tasks 31, 32, 34 and 47 build it that way from the start. Batched against one piece (same processor):
  counts, labels, classes equal; velocity and pressure within the CPU-against-GPU bounds below. Batched against
  batched with another cut, and repeats: identical bytes.
- **Every liquid is one material set** (shared-interface section d). `ComputeDistanceToSurface` is the one
  function that accepts an all-zero `material_is_counted`. Gas markers and blocking materials are never read
  from particles; solids reach this module only through `SolidCoverage`, `FreeSolidPush` and (A18)
  `BlockingMaterialFace`; gas reaches the liquid solve only through `gas_pressure_above_atmosphere`,
  `holds_tracked_gas` and (A18) `SealedGasBlob` with its two cell arrays.
- **Coupling inside the solve (A18, Tasks 43 to 50).** The method is fixed in Task 44's plan decisions and
  copied to `plan-decisions.md` ("Coupling inside the pressure solve (A18)"). Sign convention everywhere:
  `flow_imbalance` is inflow minus outflow, `s = +1` for the low end of a face piece. **Amended 2026-11-14
  (A24):** the pieces viscosity's joint solve calls accept both terms (`plan-decisions.md` "Viscous liquids with
  blocking materials and sealed gas (A24), and what counts as viscous (A25)", subsection "A24: pressure pieces").
  No piece returns `kBadInput` because the equations hold a blocking-material face or a sealed gas blob; only the
  forms without a blob buffer refuse equations with `sealed_gas_blob_count > 0` (the far-end pressure is an input
  they lack), and Tasks 45 and 47 add the forms that carry the blob pressures. `viscosity::SolveViscosityAndPressure`
  carries material faces (springy solids and snow beside a viscous liquid, owner option C) and sealed gas blobs, so
  no viscous scene reads sealed gas one step late. ~~Free solids beside a viscous liquid stay excluded (viscosity's
  rule).~~ **Amended 2026-11-14 (A26/A27):** the joint solve keeps material face velocities and free-solid motions as
  velocity unknowns (not eliminated) and calls the `CoupledVelocities` forms of Task 47 (`plan-decisions.md` "A viscous
  liquid sticks, and free solids in the viscous solve (A26, A27)"); the A24 forms stay for `SolvePressure`, unchanged.
- **One per-element function, two loops.** Each GPU function calls the same `FLUIDITY_CPU_AND_GPU` per-cell
  or per-face function the CPU loop calls (private shared headers named in each task), so CPU tests cover the
  arithmetic of the device code. GPU launches go through `ComputeLaunchSize` (Task 12). **Amended 2026-11-14
  (Foundation as built):** through base's `ComputeLaunchSize` (`base/run_on_call.h`) and `ApplyGpuFunction`
  (`base/gpu_launch.cuh`); the module has no launch-size header of its own. Every device task adds a
  `<name>_steps_test.cc` (`cpu`) that runs its per-item functions over every thread block and thread of the launch,
  in launch order and in reverse; every total over cell blocks goes through `block_sums.h` (Task 28) and the
  buffers' `sum_buffers`; every stable order, prefix sum and grouping uses the buffers' `order_buffers`.
- **Tolerances, CPU against GPU (design section 6):** counts, classes, labels, `blob_id` equal; geometry,
  equations and one application of the equations `|a - b| <= 1e-12 * scale + 1e-300` (`scale` = largest
  magnitude of that array); velocity after a solve `<= 2 * max_volume_error_rate * cell width + 1e-9 * max
  |velocity|`; pressure that bound times `face_density * pressure_distance / dt`. Iteration counts not compared.
- **Test labels:** ~~shared-formula tests and tests of a CPU implementation that the GPU mirrors
  (label propagation, ladder building) use `LABEL no_gpu` (they run in the host build and in the container);~~
  **Amended 2026-11-14 (Foundation as built):** shared-formula tests, tests of a CPU implementation that the GPU
  mirrors (label propagation, ladder building) and operation tests use `LABEL cpu` (the container's cpu step runs
  them too); `LABEL no_gpu` only for "on the GPU without a GPU" tests; device tests `LABEL gpu`. A known-open test
  (form amended 2026-11-14, Foundation as built: `fluidity_add_test` takes one label and discovers every test of
  its program under it, so a failing test there would fail the `cpu` gate) is written with GoogleTest's `DISABLED_`
  prefix, so discovery registers it as disabled and the gate skips it, and is registered a second time, inside
  `if(BUILD_TESTING)`, by `add_test(NAME <Suite>.<Name> COMMAND <program> --gtest_filter=<Suite>.DISABLED_<Name>
  --gtest_also_run_disabled_tests)` with `set_tests_properties(<Suite>.<Name> PROPERTIES WILL_FAIL TRUE LABELS
  known_open)` (`add_test` makes no C++ target, so the helper rule holds; the container's `cpu|no_gpu` run skips
  it), and a line in `docs/known-open-physics.md` (create the file with a one-line header if absent).
- **Inputs from later modules are hand-built in tests:** `SolidCoverage`, `grid_edge_is_open_to_sky`,
  `FreeSolidPush` and inverse inertia (amended 2026-11-15, A59: `FreeSolidInertia` with its
  `solids::AllowedMotions`, from Task 47a on), `BlockingMaterialFace` entries (solids); `gas_pressure_above_atmosphere`,
  `holds_tracked_gas`, `SealedGasBlob` entries, `sealed_gas_blob_of_cell`, `gas_pressure_above_background`, gas
  snapshots with `kGas` (gas); ~~`is_grid_only_region` (cell resize);~~ (A64: later group) `mixture_density`, `bulk_stiffness`,
  `carried_pressure` (step, materials). No later module is stubbed. Each such input is tested null and filled.
- **Code comments** number long bodies `// Step N:` with the numbers of design section 5 (1, 1b, 2, 3, 3b, 4 ...
  21), which are also the step numbers of the paper's two procedures.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed:

- Foundation, with the `base/` additions of `shared-interface.md` section (a): `Vec3`, `Mat3`, `Axis`, `kUp`,
  `RunOn`, `CheckCanRunOn`, `FLUIDITY_CPU_AND_GPU`, `cpu_only.h`, `scalar_math.h`, `tree_sum.h`, `buffer.h`,
  `ordered_sums.h`, `stable_order.h`, `memory_count.h`, the method paper with `figure_style.typ` and
  `pseudocode.typ`, `docs/glossary.md`, `docs/gpu-runs.md`, the gate.
- Contact (`contact::DomainSide`, `contact::DomainBox`), grid (all of grid 3.1 to 3.7, including
  `FindCornerPlaces`, `ComputeCornerPosition`, `ComputeBigCell`, `FindFacePieces`, `FindTouchingCells`,
  `BuildCellValues`, `ApplyGridVersion`, `ReadValues`, `WriteValues`), particles (`ParticlesTry`,
  `BuildFieldsToRead`), transfer (its material-subset rule and `transfer::CellsFromParticles`).
- `proofs/` as a Lake project (added by the particles module's `Merge.lean` task) and `reference/` as a uv
  project with NumPy (added by the transfer module's `least_squares.py` task). If either is missing, stop and
  ask the owner: this plan does not create them.

- **Amended 2026-11-14 (Foundation as built):** Foundation as built also provides `base/run_on_call.h`
  (`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, `SumBuffers` and `OrderBuffers` taken by
  the sums, the stable order, the prefix sum and the grouping (one form each, A31), `base/test_support.h` (library
  `fluidity_base_test_support`) and the CMake helper `fluidity_add_test_library`; read `AGENTS.md` "Structure"
  before the first task. The gate prints seven `PASS` lines: format, rules, cpu debug, cpu release, tidy, paper,
  cuda.

- **Amended 2026-11-15 (A58):** Foundation's "Additions after Foundation closed" (Tasks 17 to 19: `RunMemory`,
  `BuildRunMemory`, `BuildBuffer(RunOn, size, RunMemory&)`, `ComputeRunMemoryBytes`, `ComputeRunMemoryPieceBytes`,
  `BufferError::kRunMemoryTooSmall`, the `RunMemory&` forms of `BuildSumBuffers` / `BuildOrderBuffers`) and the
  grid plan's `RunMemory&` forms of its builders are on the branch.
- **Amended 2026-11-15 (A59):** Task 47a alone also needs solids' shared header `native/fluidity/solids/guide_math.h`
  (`solids::AllowedMotions`, `solids::ComputeAllowedMotions`; solids plan, its A59 task). The coordinator runs Task
  47a right after that solids task, as viscosity's A27 tasks run after solids' free-solid tasks; every other task of
  this plan runs in order without it.

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line, and
`grep -n "ComputeBigCell\|FindCornerPlaces" native/fluidity/grid/*.h` finds both declarations, and
`grep -n "class RunMemory" native/fluidity/base/run_memory.h` finds the class.

## Review Focus

1. **A sealed blob whose carried pressure is large** (1e5 Pa) and whose right side is compatible only to
   round-off: the conjugate gradient must not drift along the blob constant, and the offset must restore the
   mean. Pinned: `SolvePressure, SealedCubeWithLargeCarriedPressure_KeepsMean` in Task 34.
2. **Negative cell coordinates in the ladder merge** (`ComputeBigCell` halves with a floor; the old pairing
   fallback is gone). Pinned: `BuildPressureLadder, NegativeCoordinates_MergeWithFloor` in Task 31.
3. **The surface a hair above a cell centre:** `liquid_share` tends to 0 and must be raised to 0.05 and counted.
   Pinned: `FindLiquidSurface, SurfaceJustAboveCentre_ShareRaisedAndCounted` in Task 16.
4. **Which pressure an air end sees** (a sealed gas blob solved with the liquid, then tracked gas read one step
   late, then an enclosed resolved pocket, then atmosphere; a grid edge is atmosphere). Pinned:
   `BuildPressureEquations, AirEndPrecedence_GasThenPocketThenAtmosphere` in Task 23 (no sealed blob) and
   `BuildPressureEquations, AirEndPrecedence_SealedBlobFirst` in Task 46.
5. **A sealed blob cut by a batch boundary** (amended 2026-11-14, A15): its error total, its mean and its rank-one
   sums must come from block partials over every batch, or the batched solve drifts along the blob constant.
   Pinned: `SolvePressureInBatches, SealedCubeCutIntoThree_WithinToleranceAndSameBytesForEveryCut` in Task 61.
   (Geometry built per range still gives the same bytes: `FindLiquidSurface, TwoRangesEqualOneRange_SameBytes`,
   Task 16.)

---

### Task 1: Method paper, where the liquid is and the cell equation

**Kind:** mechanical

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/pressure.typ` (included from `docs/method-paper/fluidity-method.typ`
  after the transfer section; if earlier modules put their sections inline instead, follow them),
  `docs/method-paper/figures/partly_filled_cell.typ`, `figures/face_piece_flow.typ`, `figures/air_pockets.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include; notation rows)

**Interfaces:**
- Produces labels `<sec-liquid-surface>`, `<sec-pressure>`, `<eq:pressure>`, `<eq:correct>`, `<sec-tension>`,
  `<sec-pressure-law>`, `<sec-air-pockets>`, `<fig-partly-filled-cell>`, `<fig-face-piece-flow>`,
  `<fig-air-pockets>`, `<proc-find-liquid-surface>`.

- [ ] **Step 1: Text**, design section 8 outline items (1) to (6), in the decided vocabulary (design section 9):
  distance to the surface from particles and its cap; the six tetrahedra on the 0-7 diagonal with their corner
  lists; `open_area` as the single source and the face piece split along its local corner 0-3 diagonal (the
  diagonal the tetrahedra induce on every face); the cell equation `<eq:pressure>` exactly as design step 15
  with its term order, `flow_imbalance` as inflow minus outflow, `s = +1` for the low cell; the correction
  `<eq:correct>` as step 21; blobs, their three classes and the free constant (steps 16, decision 1);
  surface tension counted once, in `reference_pressure`; `squeeze_allowance` and the two pressure laws with
  `K0 = rest_density * rest_sound_speed^2`; air pockets (steps 9 to 12). Cite [21], [22], [23], [24], [26] from
  `references.typ`. Notation rows: one per paper symbol used (`p`, `V`, `A`, `rho_f`, `d`, `dt`, `sigma`,
  `kappa`, `theta` for `liquid_share`, `K`, `J`).
- [ ] **Step 2: Figures** with `figure_style.typ`: `<fig-partly-filled-cell>` (one cell, a slanted solid, the
  liquid plane, the six tetrahedra in light guides, one face piece with its open part shaded and its 0-3
  diagonal); `<fig-face-piece-flow>` (a big cell beside four small cells, the flow arrow and `pressure_distance`
  on each piece, one air end with `liquid_share` drawn as a fraction of the centre line); `<fig-air-pockets>`
  (a sparse grid section with an enclosed, a vented and an unknown-frontier pocket, a missing block touching a
  solid). Each caption states the one thing to see.
- [ ] **Step 3: Pseudocode** `<proc-find-liquid-surface>` with `procedure(...)`: steps 1, 1b, 2, 3, 3b, 4, 5, 6,
  7, 8 of design section 5 A, numbered exactly so; inputs, outputs, "fails when" (`kSurfaceDirectionUnclear`,
  `kCellSplitBySolid`, `kStepTooLongForSurfaceTension`).
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.
  Expected: exit 0, no warning. Look at every new page; fix overlaps and clipped labels. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Describe where the liquid is and the pressure equation of one cell`.

### Task 2: Method paper, the solve, the check, the open sea, gas and two liquids

**Amended 2026-11-14 (A18):** the gas paragraph is written as the state before Task 43, which rewrites it.

**Amended 2026-11-14 (A24):** the joint viscous solve no longer reads sealed gas one step late; Task 43's rewrite
says the joint solve carries the same two terms (viscosity paper section).

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/pressure.typ`
- Create: `docs/method-paper/figures/pressure_ladder.typ`

**Interfaces:**
- Produces labels `<sec-pressure-solve>`, `<sec-open-sea>`, `<fig-pressure-ladder>`, `<proc-solve-pressure>`.

- [ ] **Step 1: Text**, outline items (7) to (11): conjugate gradient with one ladder pass per iteration (cite
  [25], [6]); the merge rule by size level then `cell_coord / 2` with a floor, never across a blob; Galerkin
  sums; weighted Jacobi 2/3; direct solve at 32 cells with one grounded cell; the fixed-order sums of step 20
  (refer to `<sec-fixed-order-sums>`); the check from the corrected velocity only, one stop rule shared with the
  joint solve (cite [9]) and gas; the volume budget (`net_volume_change`, `total_volume_change`); the open-sea
  zones, `ramp = x^2 (3 - 2x)`, weight `1 - exp(-ramp * rate * dt)`, the second-order wave in finite depth with
  return current, dispersion `omega^2 = g k tanh(k depth)` (cite [20]); what gas reuses and the gas pressure the
  liquid feels one step late (stated as the current state; Task 43 rewrites it for sealed blobs, which go
  inside the solve and inside the joint viscous solve, and adds blocking materials); one distance per material,
  the boundary distance `(d_a - d_b) / 2`, and the exclusion `kMaterialBoundaryNotSupported` stated as the current state (Task 42
  rewrites this paragraph when it lifts the exclusion).
- [ ] **Step 2: Figure** `<fig-pressure-ladder>`: a mixed-size blob, rung 0 to 2, small siblings merged first,
  then pairs of big cells; arrows show the gather (each coarse cell reads its fixed member list).
- [ ] **Step 3: Pseudocode** `<proc-solve-pressure>`: steps 13 to 21 of design section 5 C and D, numbered so.
- [ ] **Step 4: Run** typst, look at the pages, run the gate. **Commit.** Subject:
  `Describe the pressure solve, its check and the open sea`.

### Task 3: Error enum, names and the module's build lines

**Amended 2026-11-14 (Foundation as built):** the shared header's test is `LABEL cpu` (was `no_gpu`).

**Amended 2026-11-15 (A58, reconcile item 31):** `kOutOfMemory` is appended as the last value (after
`kRunOnCallFailed`): 17 values here, 16 after Task 42 removes `kMaterialBoundaryNotSupported`.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/pressure/pressure_error.h`, `pressure_error_message.h`, `pressure_error_message.cc`,
  `pressure_error_test.cc`
- Modify: the CMake lines that list library sources and tests (as `native/fluidity/transfer/` does);
  `docs/glossary.md`

**Interfaces:**
- Produces: `PressureError`, `PressureFailure`, `ToMessage(PressureError)`, `ToName(PressureError)`, exactly
  as design section 3 (`pressure_error.h` shared; the message header CPU only).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the shared header part and the messages):

  ```cpp
  TEST(ToName, EveryError_IsLowerSnakeCaseWithoutK)  // design 7 "ToName, EveryError": all 17 values (A58),
                                                    // kSealedBlobVolumeMismatch -> "sealed_blob_volume_mismatch",
                                                    // kRunOnCallFailed -> "run_on_call_failed", kOutOfMemory ->
                                                    // "out_of_memory"; all distinct
  TEST(PressureError, OutOfMemory_IsLastValue)      // A58: static_cast<int>(kOutOfMemory) == value count - 1
  TEST(ToMessage, EveryError_IsDistinctAndNotEmpty)
  TEST(PressureFailure, Default_HasNoIndexes)        // cell_index and face_piece_index == grid::kNoIndex, error kNone
  ```
- [ ] **Step 2: Run.** Expected: compile error. **Step 3: Implement.** Messages say what the user can do
  ("A thin solid leaves liquid on both of its sides inside one cell. Use smaller cells."), using the comments of
  section 3. `PressureFailure` never allocates.
- [ ] **Step 4: Glossary rows** (`| Say | Plain meaning | Never say |`): `blob` (a connected body of liquid in one
  solve; never "component" for this), `air pocket`, `unknown frontier`, `volume_error`, `equation_error` (never
  `residual`), `flow_imbalance` (inflow minus outflow; never "divergence" for this).
- [ ] **Step 5: Run** the gate. **Commit.** Subject: `Add the pressure module's errors`. New names:
  `PressureError` and its 17 values (A58: `kOutOfMemory` last), `PressureFailure`, `ToMessage`, `ToName`.

### Task 4: Snapshot types and face clipping

**Amended 2026-11-14 (Foundation as built):** the shared-formula tests are `LABEL cpu` (was `no_gpu`; the container's
cpu step runs them).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/geometry_snapshot.h`, `viscosity_geometry.h`, `geometry_clip_math.h`,
  `geometry_clip_math_test.cc`
- Modify: `docs/glossary.md`

**Interfaces:**
- Produces: design 3 `CellContent`, `CellGeometry`, `FacePieceEnds`, `FacePieceGeometry`, `GeometrySnapshot`,
  `ComputeOpenFraction`, `CellEighthGeometry`, `FacePieceCentre`, `ComputeBlockedCornerAreas`, exactly.
- Produces (**Plan decision**, `geometry_clip_math.h`, all `FLUIDITY_CPU_AND_GPU`, distances in m, areas m2):

  ```cpp
  inline constexpr double kMinLiquidShare = 0.05;
  struct FaceAreas { double open_area = 0; double air_open_area = 0; };
  // The piece is two triangles, local corners {0,1,3} and {0,2,3} (the diagonal the cell tetrahedra induce on
  // every face). open_area: part with distance_to_solid > 0; air_open_area: part of it with distance_to_surface > 0.
  FaceAreas ComputeFaceAreas(const double distance_to_surface[4], const double distance_to_solid[4], double area);
  // liquid_share of design step 5: d_liquid / (d_liquid - d_other), clamped to [kMinLiquidShare, 1].
  struct LiquidShare { double share = 1; bool was_raised = false; };  // was_raised: the floor applied.
  LiquidShare ComputeLiquidShare(double distance_liquid, double distance_other);
  ```
  `ComputeBlockedCornerAreas`: in each triangle, `corner_area[c]` += the integral over the part with
  `distance_to_solid <= 0` of corner c's barycentric weight; a corner not in the triangle gets nothing from it.
  "Both triangle splits" in the design's test means both triangles of the one split.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeFaceAreas, NoSolidLiquidPlaneAcross_SplitsAreaExactly)  // piece 0.1 m square, surface distance
      // linear z - 0.037 on a vertical piece: air_open_area == 0.1 * 0.063, open_area == 0.01; 1e-12 relative
  TEST(ComputeFaceAreas, AllSolidDistancesZero_IsClosed)              // open_area == 0 exactly
  TEST(ComputeFaceAreas, SlantedSolid_MatchesPolygonClip)             // solid distance x + y - 0.13 (scaled to a
      // unit normal): open_area equals an independent Sutherland-Hodgman clip of the square in the test; 1e-12
  TEST(ComputeBlockedCornerAreas, SumsToBlockedArea)                  // design 7 row: 1000 random corner
      // distances in [-1,1] from a fixed integer formula, all sign patterns included: sum == area - open_area of
      // ComputeFaceAreas within 1e-14 * area; each >= 0
  TEST(ComputeLiquidShare, NearZero_IsRaisedAndFlagged)               // (-1e-9, 0.1) -> 0.05, was_raised
  TEST(ComputeLiquidShare, OtherSideAlsoNegative_IsCappedAtOne)
  TEST(ComputeOpenFraction, HalfOpen_IsHalf)
  TEST(CellEighthGeometry, Size_Is96Bytes)                            // static_assert too; FacePieceCentre 24
  ```
- [ ] **Step 2: Run.** Expected: compile error. **Step 3: Implement.** Clip each triangle against a linear field
  as a convex polygon of at most 5 vertices in fixed-size arrays; a triangle whose distances are all exactly 0
  has no part on either side. Port the idea of
  `git show main:native/include/fluidity/base/liquid_boundary_geometry_math_3d.h` lines 767-847
  (face mode), without the 64-epsilon checks (design section 10).
- [ ] **Step 4: Glossary rows:** `filled_volume`, `air_volume`, `open_volume`, `open_area`, `open_fraction`,
  `air_open_area`, `solid_volume_rate`, `pressure_distance`, `liquid_share`, `GeometrySnapshot`, `liquid_covered`.
- [ ] **Step 5: Run** the gate. **Commit.** Subject: `Add the liquid geometry types and face clipping`.

### Task 5: Cell clipping: volumes, the split check, cell eighths and covered centres

**Amended 2026-11-14 (Foundation as built):** the shared-formula tests are `LABEL cpu` (was `no_gpu`; the container's
cpu step runs them).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/geometry_clip_math.h`, `geometry_clip_math_test.cc`

**Interfaces:**
- Consumes: Task 4's header.
- Produces (**Plan decision**, shared; cell corners in grid order, bit 0 = high x, 1 = high y, 2 = high z;
  positions are offsets in m from the cell's low corner):

  ```cpp
  inline constexpr int kCellTetrahedra[6][4] = {{0,1,3,7},{0,1,5,7},{0,2,3,7},{0,2,6,7},{0,4,5,7},{0,4,6,7}};
  struct CellVolumes { double filled_volume = 0; double air_volume = 0; double open_volume = 0; };
  CellVolumes ComputeCellVolumes(const double distance_to_surface[8], const double distance_to_solid[8],
                                 double cell_width);
  // kCellSplitBySolid when the corners with distance_to_solid > 0 and distance_to_surface < 0 fall into two or
  // more groups joined only along cell edges whose both ends have distance_to_solid > 0. kNone otherwise.
  PressureError CheckCellSplitBySolid(const double distance_to_surface[8], const double distance_to_solid[8]);
  void ComputeCellEighthVolumes(const double distance_to_surface[8], const double distance_to_solid[8],
                                double cell_width, double filled_volume_in_cell_eighth[8]);
  struct CoveredPart { Vec3 centre_offset{}; double area = 0; };
  CoveredPart ComputeLiquidCoveredSolid(const double distance_to_surface[8], const double distance_to_solid[8],
                                        double cell_width);
  // Offset from the piece's low corner, in its plane. Returns the piece centre's offset when the part is empty.
  Vec3 ComputeLiquidCoveredOpenCentre(const double distance_to_surface[4], const double distance_to_solid[4],
                                      double area, Axis axis);
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`), cell width 0.1 m:

  ```cpp
  TEST(ComputeCellVolumes, FlatSurfaceNoSolid_IsExactSlab)           // z - 0.037: filled 0.1*0.1*0.037, air rest,
                                                                     // open 1e-3; 1e-12 relative
  TEST(ComputeCellVolumes, FlatSurfaceAndSlantedSolid_MatchPrism)    // solid (x + z - 0.12)/sqrt(2): the y-invariant
      // cross-section clipped by an independent 2D polygon clip in the test, times 0.1; 1e-12 relative
  TEST(ComputeCellVolumes, AllSurfaceDistancesZero_HoldsNoLiquid)    // filled_volume == 0 exactly
  TEST(CheckCellSplitBySolid, TwoWetCornersSeparatedBySolid_IsSplit) // corners 0 and 7 wet and open, the other six
                                                                     // solid: kCellSplitBySolid
  TEST(CheckCellSplitBySolid, OneConnectedOpenPart_IsNone)
  TEST(ComputeCellEighthVolumes, SumEqualsFilledVolume)              // 500 random fields: sum within 1e-15 * w^3
  TEST(ComputeCellEighthVolumes, FlatSurfaceAtThreeQuarters_IsExact) // plane z = 0.075: low four 1.25e-4 m3
      // each, high four 0.05 * 0.05 * 0.025 = 6.25e-5 m3 each; 1e-12 relative
  TEST(ComputeLiquidCoveredSolid, FlatFloorUnderLiquid_CentreAndAreaExact)  // solid z - 0.02, surface z - 0.07:
                                                                     // area 0.01, centre (0.05, 0.05, 0.02)
  TEST(ComputeLiquidCoveredOpenCentre, EmptyPart_IsPieceCentre)
  ```
- [ ] **Step 2: Run** (fail), **Step 3: Implement.** Clip each tetrahedron to the convex part liquid `< 0`,
  solid `> 0` (fixed arrays, at most 8 vertices), volume by a fan of tetrahedra; for eighths clip that part
  again by the three mid-planes. Ported from `liquid_boundary_geometry_math_3d.h` lines 652-655 and 767-847
  (cell mode); generalised to any sampled solid (design step 3). The split rule is this plan's reading of
  "ported limit" (design step 8): only corner samples are available.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add cell clipping into liquid, air and solid`.

### Task 6: Surface tension formulas

**Amended 2026-11-14 (Foundation as built):** the shared-formula tests are `LABEL cpu` (was `no_gpu`; the container's
cpu step runs them).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/surface_tension.h`, `surface_tension_test.cc`

**Interfaces:**
- Produces: `ComputeSurfaceCurvature`, `ComputeSurfaceTensionStepLimit` exactly as design 3. **Plan decision:**
  `inline constexpr double kMinSurfaceGradient = 1e-8;` and the sample order of `around[18]`: offsets
  `(dz, dy, dx)` in `{-1,0,1}^3` in lexicographic order (z slowest, x fastest), leaving out `(0,0,0)` and the
  eight with all three non-zero.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeSurfaceTensionStepLimit, WaterAt35Millimetres_Is0p2165)  // design 7 "Step limit": densities 0 and
      // 1000, sigma 0.0728, w 0.035 -> 0.2165 s within 1e-4 relative (exact value 0.216487)
  TEST(ComputeSurfaceTensionStepLimit, WaterAtOneMillimetre_Is1p0455e3) // w 1e-3 -> 1.0455e-3 s, 1e-4 relative
  TEST(ComputeSurfaceCurvature, SphereSamples_AreTwoOverRadius)        // samples of |x| - R at R = 8 w around
      // a surface cell centre: within 2 %; at R = 16 w within 0.5 %; positive
  TEST(ComputeSurfaceCurvature, PlaneSamples_AreZero)                  // |curvature| < 1e-12 / w
  TEST(ComputeSurfaceCurvature, FlatField_IsSurfaceDirectionUnclear)   // all 19 equal -> kSurfaceDirectionUnclear
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 6 and 7: centred differences, `n = g/|g|`,
  `sum_a (H_aa - n_a (H n)_a) / |g|`, mixed second differences from the four edge samples. Port from
  `liquid_boundary_geometry_math_3d.h` lines 667-689 (step limit) and 690-729 (curvature; on `main` the 27
  samples are x fastest, Y up: rotate when comparing).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add surface curvature and its step limit`.

### Task 7: Pressure law

**Amended 2026-11-14 (Foundation as built):** the shared-formula tests are `LABEL cpu` (was `no_gpu`; the container's
cpu step runs them).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/pressure_law.h`, `pressure_law_test.cc`
- Modify: `docs/glossary.md`

**Interfaces:**
- Produces: `PressureLawKind`, `PressureLaw`, `PressureLawState`, `ComputePressureLawState`, exactly.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), law `rest_density 1000, rest_sound_speed 1500`
  (`K0 = 2.25e9 Pa`):

  ```cpp
  TEST(ComputePressureLawState, SteadyStiffness_RoundTrip)       // design 7 "Pressure law": for p in {-1e5, 0,
      // 1e4, 1e7}: volume_ratio == exp(-p / K0), density == 1000 / volume_ratio, bulk_stiffness == K0; 1e-13 rel
  TEST(ComputePressureLawState, StiffensUnderPressure_RoundTrip) // volume_ratio == (1 + 7 p / K0)^(-1/7),
      // bulk_stiffness == K0 + 7 p; 1e-13 relative
  TEST(ComputePressureLawState, InfiniteSoundSpeed_IsCannotBeSqueezed)  // volume_ratio 1, bulk_stiffness +inf
  TEST(ComputePressureLawState, TaitBelowLowestPressure_IsOutOfRange)   // p = -K0 / 7 -> kPressureLawOutOfRange
  TEST(ComputePressureLawState, CannotBeSqueezed_HasInfiniteStiffness)
  ```
  The "bytes equal to the incompressible build" half of the design's row is in Task 23.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/liquid_eos.cc` lines 27-85
  (Hencky and Tait; tangent bulk modulus). Non-finite input: `kBadInput`.
- [ ] **Step 4: Glossary rows:** `pressure law`, `volume_ratio` if absent, `bulk_stiffness`, `squeeze_allowance`.
  Run the gate. **Commit.** Subject: `Add the pressure law of a liquid that can be squeezed`.

### Task 8: Air pocket formulas

**Amended 2026-11-14 (Foundation as built):** the shared-formula tests are `LABEL cpu` (was `no_gpu`; the container's
cpu step runs them).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/air_pocket_math.h`, `air_pocket_math_test.cc`

**Interfaces:**
- Produces: `ComputeAirPocketPressure` exactly. **Plan decision** (shared, same header):
  `double ComputeGasAmount(double rest_volume, double rest_pressure, double heat_ratio)` (m3 Pa^(1/heat_ratio)),
  `double ComputeEquivalentRadius(double air_volume)` (`cbrt(3 V / (4 pi))`),
  `double ComputeAirPocketStepLimit(double radius, double density, double pressure, double heat_ratio)`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeAirPocketPressure, HalfVolume_IsTwoToTheHeatRatio)  // design 7 "Air pocket law": (1e-3 m3, 101325
      // Pa) at 5e-4 m3 -> 101325 * 2^1.4; 1e-13 relative
  TEST(ComputeGasAmount, MergeOfTwoPockets_Adds)                  // (V1, p1) = (1e-3, 101325), (V2, p2) = (2e-3,
      // 120000): the merged pocket at rest_pressure 101325 has rest_volume giving amount1 + amount2; 1e-13 rel
  TEST(ComputeEquivalentRadius, SphereVolume_GivesRadius)         // V = 4/3 pi 0.02^3 -> 0.02
  TEST(ComputeAirPocketStepLimit, MatchesFormula)                 // 0.5 r sqrt(rho / (3 * 1.4 * p)), r 0.02,
                                                                  // rho 1000, p 101325
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/liquid_cavity.cc` lines 14,
  24-41, 104-151 (without the `frexp` scaling: the inputs are checked finite and positive instead).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the gas law of an enclosed air pocket`.

### Task 9: Connection and cell formulas

**Amended 2026-11-14 (Foundation as built):** the shared-formula tests are `LABEL cpu` (was `no_gpu`; the container's
cpu step runs them).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/cell_equation_math.h`, `cell_equation_math_test.cc`

**Interfaces:**
- Produces (**Plan decision**, shared, all `FLUIDITY_CPU_AND_GPU`):

  ```cpp
  // design step 13. share_* in m: the part of pressure_distance in each cell.
  double ComputeFaceDensity(double density_low, double share_low, double density_high, double share_high);
  struct ConnectionCoefficients { double velocity_per_pressure = 0; double flow_per_pressure = 0; };
  ConnectionCoefficients ComputeConnectionCoefficients(double open_area, double face_density,
                                                       double pressure_distance, double step_duration);
  double ComputeSqueezeAllowance(double filled_volume, double bulk_stiffness, double step_duration);  // 0 at +inf
  // design step 21: velocity_before_pressure - velocity_per_pressure * (pressure_high - pressure_low - jump).
  double ComputeVelocityAfterPressure(double velocity_before_pressure, double velocity_per_pressure,
                                      double pressure_low, double pressure_high, double pressure_jump);
  // design step 21: flow_imbalance + added_volume_rate - squeeze_allowance * (pressure - carried_pressure).
  double ComputeVolumeErrorOfCell(double flow_imbalance, double added_volume_rate, double squeeze_allowance,
                                  double pressure, double carried_pressure);
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeFaceDensity, SameDensity_IsThatDensity)
  TEST(ComputeFaceDensity, SeriesOfTwo_IsDistanceWeighted)   // (1000, 0.05, 800, 0.025) -> 2800 / 3, 1e-15 rel
  TEST(ComputeConnectionCoefficients, MatchDefinition)       // open 0.25, rho 1, d 0.75, dt 1 -> 4/3, 1/3
  TEST(ComputeSqueezeAllowance, InfiniteStiffness_IsZero)
  TEST(ComputeVelocityAfterPressure, PushesFromHighToLow)    // 0 - (4/3)(1.375 - 1.75 - 0) == 0.5
  TEST(ComputeVolumeErrorOfCell, CarriedPressureCancels)     // squeeze 2, p == carried -> flow + added
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement**, ported from
  `git show main:native/src/mac_pressure_3d_arithmetic.h` lines 110-146, 174-245, 440-503, without the
  representability checks (design section 10).
- [ ] **Step 4: Glossary rows:** `face_density`, `velocity_per_pressure`, `flow_per_pressure`,
  `reference_pressure`, `added_volume_rate`, `carried_pressure`. Run the gate. **Commit.** Subject:
  `Add the coefficients of one pressure connection`.

### Task 10: Open-sea wave

**Amended 2026-11-14 (Foundation as built):** the shared-formula tests are `LABEL cpu` (was `no_gpu`).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/open_sea_math.h`, `open_sea.h`, `open_sea_cpu.cc`, `open_sea_test.cc`

**Interfaces:**
- Produces: `OpenSeaWave`, `OpenSeaSettings`, `OpenSeaWaveSample`, `ReadyOpenSea`, `BuildOpenSea`,
  `ComputeOpenSeaWave` from design 3.
- **Plan decision:** `OpenSeaSettings` gains `contact::DomainBox domain_box;` (zone positions need the domain
  sides; section 3 gives `ApplyOpenSeaRelaxation` no other way to know them). `ReadyOpenSea` is a copyable
  class holding the checked settings, `wave_number`, `angular_frequency` and the harmonic coefficients.
  Shared, in `open_sea_math.h`: `double ComputeWaveNumber(double angular_frequency, double gravity, double
  water_depth)` (64 bisections of `x tanh x = omega^2 depth / g`, bracket `[0, q + 2 sqrt(q)]`),
  `double ComputeRelaxationWeight(double zone_position, double pull_rate, double step_duration)`
  (`-expm1(-ramp * rate * dt)`, `ramp = x^2 (3 - 2x)`), `OpenSeaWaveSample ComputeStokesWave(...)` with the
  coefficients as plain doubles.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the math and `BuildOpenSea`):

  ```cpp
  TEST(ComputeWaveNumber, SatisfiesDispersion)            // depth 0.5, period 1.2, g 9.81: omega^2 == g k
                                                          // tanh(k depth) within 1e-10 relative
  TEST(ComputeRelaxationWeight, ZoneEdges)                // x = 0 -> 0; x = 1, rate 2, dt 0.01 -> 1 - exp(-0.02)
  TEST(ComputeOpenSeaWave, ZeroHeight_IsStillWaterAtRest)
  TEST(ComputeOpenSeaWave, MatchesMainRotatedToZUp)       // 4 positions x 3 times against values printed by
      // main's ResolvedOceanSwell3D::Sample (ocean_boundary_3d.cc:100-140; world Y there is Z here, main's
      // Z is -Y here), stored in the test with the main commit hash; 1e-12 relative
  TEST(BuildOpenSea, SecondHarmonicTooLarge_IsBadInput)   // harmonic ratio > 0.25 (main line 64)
  TEST(BuildOpenSea, DirectionNotHorizontal_IsBadInput)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/ocean_boundary_3d.cc`
  lines 14-140 (Z-up: depth below the surface uses world Z; `travel_direction` has zero Z and is normalised).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the open-sea wave`. New names include the plan field
  `OpenSeaSettings::domain_box` with its reason.

### Task 11: Open-sea relaxation on CPU and GPU

**Amended 2026-11-14 (Foundation as built):** two halves (`open_sea_relaxation_gpu.h` private), the per-face function
run over every thread block and thread by a `cpu` test, results through `WriteResultDigest`.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/open_sea_relaxation_cpu.cc`, `open_sea_relaxation_gpu.h` (private),
  `open_sea_relaxation_gpu.cu`, `open_sea_relaxation_test.cc`, `open_sea_relaxation_steps_test.cc`,
  `open_sea_relaxation_gpu_test.cc`, their CMake lines
- Modify: `open_sea.h`

**Interfaces:**
- Consumes: Task 10; `grid::ComputeFacePosition`, `grid::ToAxis`.
- Produces: `ApplyOpenSeaRelaxation` exactly as design 3. Private shared
  `ApplyOpenSeaRelaxationToFace(...)` in `open_sea_math.h` used by both loops.

- [ ] **Step 1: Failing tests:**

  ```cpp
  // open_sea_relaxation_test.cc, LABEL cpu. Grid 8 x 1 x 1 cells of 0.1 m from x = 0; open_sea_side kLeft,
  // zone width 0.2; outlet kRight, zone width 0.2.
  TEST(ApplyOpenSeaRelaxation, FaceAtZonePositionOne_IsLerpToWave)  // design 7 "Open sea": face on x = 0, rate
      // r = 3, dt = 0.01: lerp(u, target_x, 1 - exp(-r dt)) within 1e-13
  TEST(ApplyOpenSeaRelaxation, OutletFace_PullsTowardRest)
  TEST(ApplyOpenSeaRelaxation, FaceOutsideZones_IsUnchanged)        // bytes equal
  TEST(ApplyOpenSeaRelaxation, ZeroRate_TurnsZoneOff)
  TEST(ApplyOpenSeaRelaxation, VelocityOfOtherGridVersion_IsWrongGridVersion)
  // open_sea_relaxation_steps_test.cc, LABEL cpu (Foundation as built: the main evidence until the hardware run)
  TEST(ApplyOpenSeaRelaxationToFace, EveryThreadBlockAndThread_EqualsCpuLoop)  // the GPU function's loop body run
      // for every thread block and thread of ComputeLaunchSize(face count), launch order and reverse: bytes
  // open_sea_relaxation_gpu_test.cc, LABEL gpu
  TEST(ApplyOpenSeaRelaxation, OnGpu_EqualsCpu)                     // 1e-12 * scale (CheckWithinTolerance:
                                                                    // math::Expm1 may differ in the last bits)
  TEST(ApplyOpenSeaRelaxation, OnGpuTwice_GivesIdenticalBytes)      // velocities through WriteResultDigest
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** One loop per face of `blocks`; position `x` measured from
  the zone's inner edge (0) to the domain side (1), clamped. `ApplyOpenSeaRelaxation` lives in
  `open_sea_relaxation_cpu.cc` and calls `ApplyOpenSeaRelaxationOnGpu` (`open_sea_relaxation_gpu.h`, returns
  `RunOnCallStatus`) under `#if defined(FLUIDITY_HAS_GPU)`. Ported from `ocean_boundary_3d.cc` lines 142-156,
  184-221, along the side's axis instead of X only. Who calls it: the step runs `solids::DriveWaves` in stage
  `DriveWaves`, and `solids::DriveWaves` calls `pressure::ApplyOpenSeaRelaxation` (coordinator decision of
  2026-11-14; on `main` the call sits at `liquid_substep_3d.cc:1248-1273`). This module only provides and tests
  the function; the function's docstring names `solids::DriveWaves` in its `Used by` line.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Relax face velocities in the open-sea zones`. Body:
  `CUDA: compiled, not run`.

### Task 12: Pressure buffers, memory estimate entries and launch sizes

**Amended 2026-11-14 (A31):** the module's working buffers are `SumBuffers sum_buffers` and `OrderBuffers order_buffers`; tests renamed `BuffersRowsMatchBuffersSizes`, `OnCpu_BuffersBytesMatchRows`; no form of a base function builds its own buffers.

**Amended 2026-11-14 (Foundation as built):** `launch_size.h` and its test removed (base's `ComputeLaunchSize`);
`PressureBuffers` owns `sum_buffers` and `order_buffers`; `ComputeBufferBytes` and `BuildPressureMemoryEstimate` take
`RunOn` (cross-plan).

**Amended 2026-11-15 (A58):** `BuildPressureBuffers(sizes, run_on, RunMemory& run_memory, RunMemory& cpu_run_memory)`; one estimate row per
buffer (sum and order buffers one row each); test `BuildPressureBuffers, IntoRunMemory_HandsOutExactlyTheEstimate`;
`kOutOfMemory` when a piece does not fit; the "without a GPU" test replaced.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/pressure/pressure_buffers.h`, `pressure_buffers.cc`, `pressure_memory_estimate.h`,
  `pressure_memory_estimate.cc`, ~~`launch_size.h`,~~ `pressure_buffers_test.cc`, `pressure_buffers_no_gpu_test.cc`,
  `native/fluidity/pressure/CMakeLists.txt` lines for each

**Interfaces:**
- Produces: `BufferSizes`, `PressureBuffers`, `ComputeBufferBytes`, `BuildPressureBuffers`,
  `BuildPressureMemoryEstimate` exactly. **Plan decisions:** `max_cells` is a multiple of 64; `max_face_pieces ==
  max_cells / 64 * 240` and `max_corners == max_cells / 64 * 125` (else `kBadInput`), so every value array is
  built with `max_cell_blocks = max_cells / 64`. ~~Private shared `launch_size.h`:
  `struct LaunchSize { std::uint32_t block_count; std::uint32_t threads_per_block; };`
  `constexpr LaunchSize ComputeLaunchSize(std::uint64_t element_count)` (256 threads per block).~~ (Foundation as
  built: base's `LaunchSize` and `ComputeLaunchSize` in `base/run_on_call.h` replace it.)
- **Plan decisions (Foundation as built):**
  - The two functions that count the table take the processor, because the working buffers' bytes depend on it (this
    changes the design's declarations; consumers: step plan, gas plan):
    ```cpp
    std::uint64_t ComputeBufferBytes(const BufferSizes& sizes, RunOn run_on);
    std::vector<MemoryEstimateEntry> BuildPressureMemoryEstimate(const BufferSizes& sizes, RunOn run_on);
    ```
  - `PressureBuffers` owns `SumBuffers sum_buffers` (built with `BuildSumBuffers(run_on, max_cells / 64)`: every
    total is a tree sum over one partial per cell block, Task 28) and `OrderBuffers order_buffers` (built with
    `BuildOrderBuffers(run_on, n)`, `n` the largest item count any stable order, prefix sum or grouping of the
    module makes: `max_face_pieces` in one piece, raised to `max_blocking_material_faces` when that is larger
    (Task 44); the batched layout of Task 56 sets its own `n`). Every call of these base
    functions passes these buffers (A31: the base functions have no other form).
  - ~~Rows `"pressure: sum buffers"` (given count `max_cells / 64`, `bytes_each =
    ComputeSumBuffersSize(run_on).bytes_per_item`), `"pressure: sum buffers fixed bytes"` (given count
    `ComputeSumBuffersSize(run_on).fixed_bytes`, `bytes_each = 1`) and the two `"pressure: order buffers"` rows the
    same way with `ComputeOrderBuffersSize(run_on)` and `n`;~~ **(A58)** Row `"pressure: sum buffers"` (given count
    1, `bytes_each = ComputeSumBuffersSize(run_on).bytes_per_item * (max_cells / 64) + fixed_bytes`) and row
    `"pressure: order buffers"` (given count 1, the same with `ComputeOrderBuffersSize(run_on)` and `n`): each
    working buffer is one piece of the `RunMemory` (Foundation Task 18), so one row each keeps the rounding to 256 B
    exact; `where = kRunOn`, `KeptFor::kWholeRun`.
    `BuildPressureBuffers` builds the two working buffers with `BuildSumBuffers(run_on, max_cells / 64, run_memory)` /
    `BuildOrderBuffers(run_on, n, run_memory)` (the walk over the other rows skips them); ~~a `BufferError` from
    either is `kRunOnCallFailed` (Global Constraints)~~ (A58) `kRunMemoryTooSmall` from either is `kOutOfMemory`,
    `magnitude` = bytes asked.
  - **Plan decision (A58):** the declaration is `std::expected<std::unique_ptr<PressureBuffers>, PressureFailure>
    BuildPressureBuffers(const BufferSizes& sizes, RunOn run_on, RunMemory& run_memory, RunMemory&
    cpu_run_memory);` (design 3; `kCpu` rows of the batched layout, Task 56, from `cpu_run_memory`). Each row of
    the table below stands for **one** buffer, `copies` identical buffers when it says so: the rows that list
    several arrays are split into one row per array, keeping their names (`"pressure: corner distance liquid"`,
    `"... corner distance solid"`, `"... cell geometry"`, `"... centre distance"`, `"... curvature samples"`,
    `"... cell equations"`, `"... equation of cell"`, `"... first equation of block"`, `"... connections"`, `"...
    connection offsets"`, `"... connection pairs"`, `"... ladder own weights"` / `"... ladder connection weights"`
    (copies 2 each), `"... solve vectors"` (copies 7), `"... labels"`, `"... blobs"`, `"... block partials"`, `"...
    air pocket keys"`, `"... air pocket order"`; one-material distances: copies `max_boundary_materials` for each
    of the corner and centre rows). The allocation walk builds each row's buffers in table order with
    `BuildBuffer(run_on, count, run_memory)` (`kCpu` rows: `BuildBuffer(RunOn::kCpu, count, cpu_run_memory)`), so
    the handed-out bytes equal `ComputeRunMemoryBytes(rows, counts, where)` per block.
- `PressureBuffers` members are private; later tasks add accessors in a private `pressure_buffers_access.h`.
  Rows (all `KeptFor::kOneStepTry`, `copies = 1`), names prefixed `"pressure: "`:

  | Row | Counted per | Bytes each |
  |---|---|---|
  | corner distances, liquid and solid | corner place | 16 |
  | one-material distances | corner place + cell place, times `max_boundary_materials` | 8 each |
  | `CellGeometry`, centre distance, `CurvatureSamples` | cell place | `sizeof` each, summed |
  | `FacePieceGeometry` | face place | `sizeof(FacePieceGeometry)` |
  | `CellEighthGeometry` / `FacePieceCentre` (only `has_viscous_ingredient`) | cell / face place | 96 / 24 |
  | `MaterialBoundary` | face place, only `max_boundary_materials > 0` | `sizeof(MaterialBoundary)` |
  | `CellEquation`, equation of cell, first equation of block | cell place | `sizeof(CellEquation)` + 4 + 4 |
  | `CellConnection`, offsets, (connection, other equation) pairs | face place | `sizeof(CellConnection)` + 4 + 8 |
  | ladder, two copies of rung 0 as an upper bound (8/7 is under 2) | cell place; face place | 2 x 24; 2 x 8 |
  | direct-solve factors | cell place | 32 x 8 |
  | solve vectors `p`, error, `z`, `d`, `q`, products and `volume_error` | cell place | 7 x 8 |
  | labels, `PressureBlob`, block partials | cell place; cell place; cell block | 8; `sizeof(PressureBlob)`; 8 |
  | air pocket working buffers: overlap keys and order | cell place | 16 |
  | free solids: impulse sums | given count `max_free_solids` | 48 |
  | progress record | given count 1 | 64 |

- [ ] **Step 1: Failing tests:**

  ```cpp
  // LABEL cpu
  TEST(ComputeBufferBytes, EqualsSumOfMemoryEstimate)       // sizes for 10 blocks, both flags on and off, kCpu
                                                            // and kGpu (rows only, no GPU needed)
  TEST(BuildPressureMemoryEstimate, BuffersRowsMatchBuffersSizes)  // Foundation as built: 10 blocks, kCpu and kGpu:
      // the four sum and order buffers rows give ComputeSumBuffersSize(run_on) at 10 values and ComputeOrderBuffersSize(run_on)
      // at 2400 items (bytes_per_item * count + fixed_bytes)
  TEST(BuildPressureBuffers, OnCpu_BuffersBytesMatchRows)   // sum_buffers.byte_count() 0, order_buffers.byte_count()
                                                            // == 4 * 2400 + 8192
  TEST(ComputeBufferBytes, ViscousFlag_AddsEighthsAndCentres)   // 10 blocks: difference == 640 * 96 + 2400 * 24
  TEST(BuildPressureBuffers, CellsNotAMultipleOf64_IsBadInput)
  TEST(BuildPressureBuffers, OnCpu_Succeeds)
  ~~TEST(ComputeLaunchSize, TwoMillionElements_Uses8192Blocks)~~  // removed: base's run_on_call_test covers it
  // A58 (design 7):
  TEST(BuildPressureBuffers, IntoRunMemory_HandsOutExactlyTheEstimate)  // 10 blocks, max_free_solids 2, both flags
      // on, kCpu: BuildRunMemory(kCpu, sum over both where of ComputeRunMemoryBytes(BuildPressureMemoryEstimate(
      // sizes, kCpu), counts, where)), passed as both blocks (a CPU run); build succeeds; run_memory.handed_out_bytes() == run_memory.reserved_bytes(), exact
  TEST(BuildPressureBuffers, RunMemoryOneByteShort_IsOutOfMemory)  // the same reserved minus 1: kOutOfMemory,
      // magnitude == bytes the last row's piece asked (ComputeRunMemoryPieceBytes of its bytes), exact
  TEST(BuildPressureMemoryEstimate, OnGpuRows_CountedWithoutAGpu)  // kGpu rows and ComputeRunMemoryBytes of them
      // computed without a GPU; every row a multiple-of-one-buffer row (copies x count x bytes_each), exact
  // ~~pressure_buffers_no_gpu_test.cc, LABEL no_gpu~~
  ~~TEST(BuildPressureBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed)~~   // A58: removed; a GPU RunMemory cannot be
      // built without a GPU (Foundation's BuildRunMemory, OnGpuWithoutAGpu_IsRunOnCallFailed covers the path)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** `BuildPressureBuffers` allocates by walking the rows
  `BuildPressureMemoryEstimate` returns, so the estimate cannot drift; nothing allocates after it.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the pressure buffers and their memory estimate entries`.

### Task 13: Test scenes for the pressure tests

**Amended 2026-11-14 (reconcile):** test libraries end in `_test_support` (reconcile item 29: `fluidity_grid_test_support`).

**Amended 2026-11-14 (reconcile):** the test helper is `EquationSceneArrays` / `BuildEquationSceneArrays` (was
`EquationArrays`, the name of Task 23's opaque type behind `PressureEquations::arrays`).

**Amended 2026-11-14 (Foundation as built):** `fluidity_pressure_test_scenes` is created with
`fluidity_add_test_library` and linked through `LINKS`.

**Amended 2026-11-15 (A58):** adds `PressureBuffersHeld` and `BuildHeldPressureBuffers` (buffers built into a
`RunMemory` of exactly their estimate) for every later test that needs buffers.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/pressure/pressure_test_scenes.h`, `pressure_test_scenes.cc`,
  `pressure_test_scenes_test.cc` (a test-only library `fluidity_pressure_test_scenes`, linked by every later
  pressure test, never by `fluidity`). **Foundation as built:** `fluidity_add_test_library(fluidity_pressure_test_scenes
  SOURCES pressure_test_scenes.cc LINKS fluidity_base_test_support fluidity_grid_test_support)`; every later pressure
  test is `fluidity_add_test(<name> SOURCES ... LABEL <label> LINKS fluidity_pressure_test_scenes)` and uses
  `base/test_support.h` for bits, tolerances, buffer builders and digests instead of its own copies

**Interfaces:**
- Produces (**Plan decision**, namespace `fluidity::pressure::testing`, CPU only):

  ```cpp
  grid::StaggeredGrid BuildBoxGrid(Vec3 origin, double cell_width, grid::CellCoord cell_count);  // level 0
  // Level 0 cells of width 2 w over cell_count, with every big cell whose centre lies in split_box replaced by
  // its 8 small cells (width w). Negative origins allowed.
  grid::StaggeredGrid BuildTwoLevelGrid(Vec3 origin, double small_cell_width, grid::CellCoord big_cell_count,
                                        contact::DomainBox split_box);
  struct SurfaceArrays;    // owns corner, centre CellValues and the CurvatureSamples vector; .view() gives
                           // SurfaceDistances
  // surface_tension_max 0 leaves the curvature samples out, as in SurfaceDistanceInputs.
  SurfaceArrays BuildSurfaceFromFunction(const grid::StaggeredGrid& grid, double (*distance)(Vec3, const void*),
                                         const void* context, double surface_tension_max);
  struct SolidArrays;      // owns distance_to_solid and solid_normal_velocity; .view() gives SolidCoverage
  SolidArrays BuildSolidFromFunction(const grid::StaggeredGrid& grid, double (*distance)(Vec3, const void*),
                                     double (*normal_velocity)(Vec3, Axis, const void*), const void* context);
  double DistanceToOpenTopBox(Vec3 position, const void* box);   // contact::DomainBox*; walls and floor
  struct SnapshotArrays;   // owns writable CellValues<CellGeometry>, FaceValues<FacePieceGeometry>; .view()
  SnapshotArrays BuildEmptySnapshot(const grid::StaggeredGrid& grid, double step_duration);  // all kSolid
  struct EquationSceneArrays;  // owns velocity_before_pressure, mixture_density, bulk_stiffness (+inf),
                               // carried_pressure (0); .inputs(grid, snapshot) gives EquationInputs
  EquationSceneArrays BuildEquationSceneArrays(const grid::StaggeredGrid& grid, double density);
  // A58. Member order is destruction order in reverse: the buffers die before the block they are pieces of.
  struct PressureBuffersHeld {
    RunMemory run_memory;
    std::optional<RunMemory> cpu_run_memory;   // empty in a CPU run (run_memory is passed twice)
    std::unique_ptr<PressureBuffers> buffers;
  };
  // A CPU run: one RunMemory of both where sums, passed twice. A GPU run (gpu tests): run_memory on the GPU of the
  // kRunOn sum plus a CPU cpu_run_memory of the kCpu sum (member cpu_run_memory, declared before buffers); then
  // BuildPressureBuffers into them; a failure ends the test (ASSERT in the caller via the returned expected).
  std::expected<PressureBuffersHeld, PressureFailure> BuildHeldPressureBuffers(const BufferSizes& sizes, RunOn run_on);
  ```
  The flat plane and the sphere are distance functions here (design step 1: they are test inputs now).

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `BuildTwoLevelGrid, SplitBox_GivesBothLevels` (counts per
  level), `BuildSurfaceFromFunction, PlaneAtCorners_IsExact`, `BuildSolidFromFunction, OpenTopBox_WallsAreZero`.
- [ ] **Step 2: Run** (fail), **Step 3: Implement**, **Step 4: Run** the gate. **Commit.** Subject:
  `Add test scenes for the pressure module`.

### Task 14: `ComputeDistanceToSurface` on the CPU

**Amended 2026-11-15 (A64):** no grid-only input: the plan decision on grid-only corners and the test
`GridOnlyCell_IsMinusTwoWidths` move to Task 63 of the later group "Deep liquid without particles (later; needs review)".

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/liquid_surface.h`, `distance_to_surface_cpu.cc`, `distance_to_surface_math.h`
  (private shared), `distance_to_surface_test.cc`

**Interfaces:**
- Consumes: `particles::BuildFieldsToRead(particles_try, material_index, cell_blocks)`, `GridTables`
  (`same_level_blocks`, `bigger_block`, `smaller_blocks`), Task 12 buffers.
- Produces: `CurvatureSamples`, `SurfaceDistances`, `kAllLiquidMaterials`, `SurfaceDistanceInputs`,
  `ComputeDistanceToSurface` exactly; the other declarations of `liquid_surface.h` come in Task 16.
  **Plan decisions:** the cap at a corner uses the width of the size level the corner is stored at (the
  smaller); ~~a corner of any grid-only cell gets `-2` widths;~~ (A64: Task 63) "touching air" for `around_surface_cells` means
  centre distance `< 0` and some face piece's other cell (or a grid edge) has centre distance `>= 0`; the 18
  samples are evaluated from particles, not interpolated.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), particles of material 0 on a lattice of spacing 0.025 m filling
  `z < 0.2`, radius 0.0125, grid 8 x 8 x 8 of 0.05 m:

  ```cpp
  TEST(ComputeDistanceToSurface, OneParticle_IsDistanceMinusRadius)   // exact, at every corner within 2 w
  TEST(ComputeDistanceToSurface, FarFromParticles_IsCappedAtTwoWidths)
  ~~TEST(ComputeDistanceToSurface, GridOnlyCell_IsMinusTwoWidths)~~  // A64: moved to Task 63 (later)
  TEST(ComputeDistanceToSurface, UncountedMaterial_IsNotRead)        // material 1 particles above the surface
  TEST(ComputeDistanceToSurface, OneMaterialField_LeavesAllLiquidsFieldUntouched)
  TEST(ComputeDistanceToSurface, MoreMaterialFieldsThanBuffers_IsBadInput)
  TEST(ComputeDistanceToSurface, SurfaceTensionZero_HasNoCurvatureSamples)
  TEST(ComputeDistanceToSurface, CurvatureSamplesSorted_OnlySurfaceCells)
  TEST(ComputeDistanceToSurface, ParticleOrderPermuted_SameBytes)    // the minimum is exact in any order
  ```
  The all-zero mask is tested end to end in Task 16 (`ComputeDistanceToSurface, AllZeroMask_IsNoLiquid`).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design step 1: for each sample point, the 27 same-level
  blocks around its block plus bigger and smaller blocks; minimum over particles of counted materials.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Compute the distance to the liquid surface from particles`.

### Task 15: `ComputeDistanceToSurface` on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (Foundation as built):** two halves (`distance_to_surface_gpu.h`), a `cpu` steps test over every
thread block and thread, the curvature compaction through the buffers' `order_buffers`; the bit-for-bit test is kept by
a deliberate same-bits promise in the header.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/distance_to_surface_gpu.h` (private), `distance_to_surface_gpu.cu`,
  `distance_to_surface_steps_test.cc`, `distance_to_surface_gpu_test.cc`, their CMake lines
- Modify: `distance_to_surface_cpu.cc` (calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`), `liquid_surface.h`

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the main evidence until the hardware run):
  `ComputeDistanceToSurfaceAtSample, EveryThreadBlockAndThread_EqualsCpuLoop` (the shared per-sample function of
  `distance_to_surface_math.h` run for every thread block and thread of `ComputeLaunchSize(sample count)`, launch
  order and reverse, on Task 14's scene: bytes equal to `ComputeDistanceToSurface` on the CPU).
  (`LABEL gpu`): `ComputeDistanceToSurface, OnGpu_EqualsCpuBitForBit` (the
  minimum of exact differences: equal bytes except where `math::Sqrt` is used, which is correctly rounded, so
  bytes equal; compared with `CheckSameBitsOrBothNaN` on every value); `ComputeDistanceToSurface,
  OnGpuTwice_GivesIdenticalBytes` (both fields and the curvature samples through `WriteResultDigest`).
  **Plan decision (Foundation as built):** `ComputeDistanceToSurface`'s docstring promises the same bits on both
  processors unless NaN, with its reason: every value is a minimum (exact) of `math::Length(position - particle) -
  radius` and of the caps, and `math::Length` is a same-bits function of `base/scalar_math.h`.
- [ ] **Step 2: Implement.** One thread per sample point calling the shared function of Task 14; curvature
  samples compacted with `BuildGroupsByDestination` (the buffers' `order_buffers`) into
  ascending cell order. `distance_to_surface_gpu.cu` returns `RunOnCallStatus`; the CPU half maps it with
  `ToBufferError`, then into `PressureFailure` (`kRunOnCallFailed`).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Compute the distance to the surface on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 16: `FindLiquidSurface` on the CPU, cells and face pieces

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/liquid_surface_cpu.cc`, `liquid_surface_math.h` (private shared, per cell
  and per piece), `liquid_surface_test.cc`
- Modify: `liquid_surface.h`

**Interfaces:**
- Consumes: Tasks 4, 5, 13, 14.
- Produces: `SolidCoverage`, `LiquidSurfaceInputs`, `FindLiquidSurface` exactly. **Plan decisions:** content
  is `kLiquid` (centre distance `< 0`, `filled_volume > 0`), else `kSolid` when `open_volume == 0`, else
  `kAir`. `ends`: `kLiquidToLiquid`, `kLiquidToAir` (other cell `kAir`, or a grid edge with `open_area > 0`),
  `kLiquidToSolid` (other cell `kSolid` or `open_area == 0`), else `kNoLiquid`. At a grid edge with
  `open_area > 0`: `liquid_share = 1`, `pressure_distance` = the grid's half width. `FindLiquidSurface` fails
  `kStepTooLongForSurfaceTension` (`magnitude = step_duration`, `limit` = the limit) when the step is longer
  than `surface_tension_step_limit` (section 3 lists no caller that raises it otherwise).

- [ ] **Step 1: Failing tests** (`LABEL cpu`), each on `BuildBoxGrid` (0.1 m) and on `BuildTwoLevelGrid`:

  ```cpp
  TEST(FindLiquidSurface, FlatSurface_VolumesAndAreasExact)        // design 7 "Clipping, flat surface": plane at
      // H = 0.137, no solid: filled_volume, air_volume, open_area, air_open_area exact; 1e-12 relative
  TEST(FindLiquidSurface, FlatSurfaceSlantedSolid_MatchesPolyhedra) // solid (x + z - 0.12)/sqrt(2); exact values
      // from Task 5's independent prism clip; 1e-12 relative
  TEST(FindLiquidSurface, BigFace_EqualsSumOfItsFourPieces)        // two-level grid: big cell's face computed
      // from its own corners (ComputeFaceAreas over the whole face) == sum of the 4 pieces; 1e-12 relative
  TEST(FindLiquidSurface, SolidVolumeRate_IsBlockedAreaTimesNormalVelocity)
  TEST(FindLiquidSurface, SurfaceJustAboveCentre_ShareRaisedAndCounted)   // Review Focus 3: H = centre + 1e-9:
      // liquid_share 0.05, lengthened_distance_count == number of such pieces
  TEST(FindLiquidSurface, TwoRangesEqualOneRange_SameBytes)        // Review Focus 5: two block ranges with
      // border, then one range of all blocks: identical CellGeometry and FacePieceGeometry bytes
  TEST(FindLiquidSurface, ThinSolidWallInCell_IsCellSplitBySolid)  // corners 0, 7 wet; lowest cell reported
  TEST(FindLiquidSurface, SolidOfOtherGridVersion_IsWrongGridVersion)
  TEST(ComputeDistanceToSurface, AllZeroMask_IsNoLiquid)           // design 7 row: gas-material particles only,
      // all-zero mask, kAllLiquidMaterials: every distance > 0 (+2 widths), FindLiquidSurface gives every open
      // cell kAir and no kLiquid cell (so no equation); the same mask with material_index 0 -> kBadInput
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 2, 3, 4, 5, 8 (`// Step N:`), from
  `git show main:native/src/liquid_boundary_geometry_3d.cc` lines 765-1072 and 1077-1286 (classification at
  873-893), without the obstacle variants, box freezes and piston override (design section 10). Snapshot arrays
  are written only inside `blocks`; border cells supply corner samples.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Find the liquid cells and the open area of every face piece`.

### Task 17: `FindLiquidSurface` on the CPU, curvature and step limit

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/liquid_surface_cpu.cc`, `liquid_surface_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`), sphere of radius `R = 8 w`, `w = 1 mm`, centre on a cell corner
  plus `(0.3 w, 0.1 w, 0.2 w)`, `sigma = 0.0728`, densities 1000 and 0:

  ```cpp
  TEST(FindLiquidSurface, SphereSurfaceCells_CurvatureNearTwoOverRadius)  // every kLiquid cell touching air
      // within 2 % of 2/R; all other cells 0
  TEST(FindLiquidSurface, StepLimit_IsSmallestOverSurfaceCells)          // == ComputeSurfaceTensionStepLimit(0,
      // 1000, w, 0.0728); +infinity with sigma 0
  TEST(FindLiquidSurface, StepLongerThanLimit_IsStepTooLongForSurfaceTension)
  TEST(FindLiquidSurface, FlatDistanceAtSurfaceCell_IsSurfaceDirectionUnclear)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 6 and 7 from `CurvatureSamples`.
  **Plan decision:** `rho_low + rho_high` of step 7 is `0 + lightest_density` (air counts as 0), the only
  density `LiquidSurfaceInputs` carries.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add surface curvature to the liquid snapshot`.

### Task 18: `FindLiquidSurface` on the CPU, cell eighths and centres

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/liquid_surface_cpu.cc`, `liquid_surface_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`), design 7 "Cell eighths and centres", both grids:

  ```cpp
  TEST(FindLiquidSurface, CellEighths_SumToFilledVolume)          // planes at H in {0.013, 0.05, 0.087} in a cell
      // of 0.1 m, slanted flat solid: sum within 1e-12 relative; each equals Task 5's independent clip
  TEST(FindLiquidSurface, CoveredCentresAndArea_Exact)            // flat floor: liquid_covered_area == wetted floor
      // area, centre on the floor; liquid_covered_open_centre of a wall piece exact; 1e-12 relative
  TEST(FindLiquidSurface, WithoutViscosity_EighthsAndCentresNull) // both pointers null
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design step 3b with Task 5's functions; world centre = cell
  low corner (`ComputeCornerPosition` of corner 0) plus the offset.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the liquid in each cell eighth for viscous scenes`.

### Task 19: `FindLiquidSurface` on the GPU

**Amended 2026-11-14 (Foundation as built):** two halves (`liquid_surface_gpu.h`), per-block reductions as per-thread
functions in `liquid_surface_steps.h` with a `cpu` steps test, digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/liquid_surface_gpu.h` (private), `liquid_surface_gpu.cu`,
  `liquid_surface_steps.h` (private shared), `liquid_surface_steps_test.cc`, `liquid_surface_gpu_test.cc`, their
  CMake lines
- Modify: `liquid_surface_cpu.cc` (calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`)

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the main evidence until the hardware run):
  `FindLiquidSurfaceSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (each phase's per-cell and per-piece functions
  of `liquid_surface_math.h` and the per-cell-block reductions of `liquid_surface_steps.h`, run for every thread
  block and thread of each launch, launch order and reverse, phase by phase, on the Tasks 16-18 scenes: bytes equal
  to the CPU loop; count, step limit and failure equal).
  (`LABEL gpu`): `FindLiquidSurface, OnGpu_EqualsCpu` (Tasks 16-18 scenes,
  geometry tolerance), `FindLiquidSurface, OnGpuTwice_GivesIdenticalBytes` (snapshot arrays through
  `WriteResultDigest`).
- [ ] **Step 2: Implement.** One GPU function per phase (clip cells, clip face pieces, eighths and centres,
  curvature), each calling `liquid_surface_math.h`. `lengthened_distance_count` and the step limit are reduced
  per cell block, then over cell blocks (integer sum; minimum is exact); the first failure is the integer minimum of
  `(step, cell_index, face_piece_index)` packed in 64 bits. The per-cell-block reductions and the packing are
  `FLUIDITY_CPU_AND_GPU` functions in `liquid_surface_steps.h`; the GPU half returns `RunOnCallStatus`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Find the liquid surface on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 20: `FindMaterialBoundaries`, behind its exclusion

**Amended 2026-11-14 (Foundation as built):** two halves (`material_boundary_gpu.h`), the per-piece crossing check
shared in `material_boundary_math.h` (created here) and run over every thread block and thread by a `cpu` test.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/material_boundary.h`, `material_boundary_cpu.cc`, `material_boundary_gpu.h`
  (private), `material_boundary_gpu.cu`, `material_boundary_math.h` (private shared: the per-piece crossing check),
  `material_boundary_test.cc`, `material_boundary_steps_test.cc`, `material_boundary_gpu_test.cc`, their CMake lines
- Modify: `docs/glossary.md` (`MaterialBoundary`, `low_share`)

**Interfaces:**
- Produces: `MaterialBoundary` (declared in `pressure_equations.h` per design; create that header here with
  only this struct and `kNoEquation`, Task 23 adds the rest), `MaterialBoundaryInputs`,
  `FindMaterialBoundaries` exactly.

- [ ] **Step 1: Failing tests:**

  ```cpp
  // LABEL cpu. Oil (material 1) over water (material 0) in a closed box, boundary at z = 0.23, both grids.
  TEST(FindMaterialBoundaries, CrossedPiece_IsMaterialBoundaryNotSupported)  // face_piece_index = lowest crossed
  TEST(FindMaterialBoundaries, LiquidsApart_IsEmpty)                         // a gap of air between them
  TEST(FindMaterialBoundaries, FieldsOfOtherGridVersion_IsWrongGridVersion)
  // LABEL cpu (material_boundary_steps_test.cc): the main evidence until the hardware run
  TEST(FindMaterialBoundariesSteps, EveryThreadBlockAndThread_EqualsCpuLoop)  // the crossing check over every
      // thread block and thread, launch order and reverse: the lowest crossed piece equal
  // LABEL gpu
  TEST(FindMaterialBoundaries, OnGpu_EqualsCpu)              // failure and face_piece_index equal
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design step 1b: boundary distance `(d_low - d_high) / 2` at
  centres; a piece between two `kLiquid` cells whose centre values differ in sign is crossed; return the
  exclusion on the first (lowest index). This task only detects crossings; `low_share`, curvature and the
  entries come in Tasks 40 (CPU) and 41 (GPU), and Task 42 removes the exclusion. The full "Two liquids at rest"
  test is Task 38 (known open until Task 42).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Find where two liquids that do not mix touch`. Body:
  `CUDA: compiled, not run`; states the capability exclusion.

### Task 21: `FindAirPockets` on the CPU

**Amended 2026-11-14 (Foundation as built):** `label_propagation_test.cc` is `LABEL cpu` (was `no_gpu`).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/air_pockets.h`, `air_pockets_cpu.cc`, `label_propagation.h` (private shared),
  `air_pockets_test.cc`, `label_propagation_test.cc`
- Modify: `docs/glossary.md` (`blob_id`, `BlobHistory`, `gas amount`, `vented`, `enclosed`)

**Interfaces:**
- Consumes: Tasks 8, 16 (`GeometrySnapshot`).
- Produces: `AirPocketClass`, `AirPocket`, `AirPocketSettings`, `BlobHistory`, `kNoAirPocket`, `AirPockets`,
  `AirPocketArrays`, `AirPocketInputs`, `FindAirPockets` exactly. `label_propagation.h`: per-cell
  `ApplySmallestLabel` and `ApplyPointerJump` steps used by the CPU loop and the GPU (design section 6).
- **Plan decisions:** (a) the domain's top side is marked by the solids module in `grid_edge_is_open_to_sky`
  like any other sky opening; `FindAirPockets` reads only that array. (b) Overlap uses the new snapshot's
  `air_volume` of each cell whose committed and new pocket match (the committed list keeps no per-cell
  volume). (c) A receiving pocket's tracked gas is stored at `rest_pressure = atmosphere_pressure`,
  `rest_volume = amount / atmosphere_pressure^(1/heat_ratio)`. (d) A new enclosed resolved pocket with no gas
  takes `blob_id = history.next_blob_id++`. (e) `FindAirPockets` fails `kStepTooLongForAirPocket` when
  `geometry.step_duration` exceeds the new `step_limit`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `label_propagation_test.cc` too), grid 8 x 8 x 8 of
  0.05 m, liquid everywhere except the bubbles named:

  ```cpp
  TEST(FindAirPockets, ClosedBubble_IsEnclosed)          // design 7 "Air pocket classes", 3 x 3 x 3 cells
  TEST(FindAirPockets, BubbleUnderOpenTop_IsVented)      // touches a top grid-edge piece marked open to sky
  TEST(FindAirPockets, BubbleBesideMissingBlock_IsUnknownFrontier)  // block absent, edge not marked
  TEST(FindAirPockets, SplitBubble_LargerPartKeepsBlobId) // second call with the first result as committed:
      // gas shares add to the old amount; ties broken by larger overlap, then lower old blob_id
  TEST(FindAirPockets, TrackedGasReachesFrontier_IsAirPocketFateUnknown)
  TEST(FindAirPockets, TrackedGasCells_BelongToNoPocket)           // holds_tracked_gas -> kNoAirPocket
  TEST(FindAirPockets, SmallPocket_IsNotResolvedAndAtAtmosphere)  // radius < 2 w; history.too_small records gas
  TEST(FindAirPockets, MorePocketsThanStorage_IsBadInput)
  TEST(ApplySmallestLabel, AnySchedule_GivesSmallestIndexOfEachSet)  // three schedules, same labels
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 9 to 12, from
  `git show main:native/src/liquid_air_components.cc` lines 36-122, 124-201 and `liquid_cavity.cc` lines 45-80,
  153-304. Labels by propagation (not breadth-first), so the CPU and GPU give the same labels.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Find air pockets and what became of their gas`.

### Task 22: `FindAirPockets` on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (Foundation as built):** two halves (`air_pockets_gpu.h`), a `cpu` steps test of the propagation
and pair-sum functions over every thread block and thread, `BuildStableOrder` with the buffers' `order_buffers`,
digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/air_pockets_gpu.h` (private), `air_pockets_gpu.cu`,
  `air_pockets_steps_test.cc`, `air_pockets_gpu_test.cc`, their CMake lines
- Modify: `air_pockets_cpu.cc` (calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`)

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the main evidence until the hardware run):
  `FindAirPocketsSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (`ApplySmallestLabel`, `ApplyPointerJump` and the
  per-pair run sum, run pass by pass for every thread block and thread, launch order and reverse, on Task 21's
  scenes: labels and pair sums equal to the CPU loop's bytes).
  (`LABEL gpu`): `FindAirPockets, OnGpu_EqualsCpu` (classes, labels, `blob_id`
  equal; volumes and pressures within the geometry tolerance), `FindAirPockets, OnGpuTwice_GivesIdenticalBytes`
  (`pocket_of_cell` and the pocket list through `WriteResultDigest`).
- [ ] **Step 2: Implement.** Label propagation passes until a pass changes nothing (a device flag read every
  pass); overlaps by `BuildStableOrder` (the buffers' `order_buffers`) over keys
  `(old_pocket << 32) | new_pocket`, then one thread per pair
  run sums in order; the pocket list (at most `max_air_pockets`) is copied to the CPU `pocket_storage`, where the
  history matching runs (it is a small list; the per-cell work stays on the device).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Find air pockets on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 23: Equation indexes, connections and cell equations on the CPU

**Amended 2026-11-15 (A59):** built here with the design's former `free_solid_inverse_inertia` (two `Mat3` per
solid, the unguided case); Task 47a replaces it by `EquationInputs::free_solids` and the guided term.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_equations.h`
- Create: `pressure_equations_cpu.cc`, `pressure_equations_math.h` (private shared), `pressure_equations_test.cc`
- Modify: `docs/glossary.md` (`cell_equation`, `cell_connection`, `own_weight`, `push`)

**Interfaces:**
- Consumes: Tasks 9, 16, 21; `math::TreeSum`.
- Produces: `kNoEquation`, `kNoBlob`, `CellEquation`, `CellConnection`, `BlobClass`, `PressureBlob`,
  `FreeSolidPush`, `SealedBlobCheck`, `EquationInputs`, `PressureEquations`, `BuildPressureEquations`,
  `ComputeEquationIndexes` exactly, with these **Plan decisions**:
  - `EquationInputs` gains `double lightest_density = 0;` (step 13's fallback density; section 3 gives the build
    no other source) and `double max_volume_error_rate = 2e-5;` (step 16's limit; the build has no
    `SolveSettings`).
  - `PressureEquations` gains `double max_volume_error_rate`, and an opaque `const EquationArrays* arrays` with
    `std::uint64_t build_number`. Every function taking `PressureEquations` fails `kWrongGridVersion` when the
    buffers have been rebuilt since (`build_number` differs) or after a failed build.
  - A `kLiquid` cell whose `bulk_stiffness` is not `> 0` gives `kPressureLawOutOfRange`.
  - Blobs are joined through connections and through cells pushing the same free solid.
  - The blob label and class are built here (step 16, first two sentences); Task 24 adds the rest of step 16.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildPressureEquations, SymmetricAndPositive)   // design 7: two-level grid, slanted solid, a free solid
      // with 3 hand-built FreeSolidPush entries, 20 random pressure pairs: x.A y == y.A x and x.A x >= 0 within
      // 1e-12 relative; x.A x == 0 only for blob constants of sealed blobs
  TEST(ComputeEquationIndexes, MatchesBuild)           // design 7: same scene and a kGas copy of its snapshot
  TEST(BuildPressureEquations, NonFiniteDensity_IsBadInput)     // cell_index of the cell
  TEST(BuildPressureEquations, InfiniteStiffness_SameBytesAsCannotBeSqueezed)  // design 7 "Pressure law" second
      // half: bulk_stiffness from ComputePressureLawState with infinite sound speed: squeeze_allowance 0, every
      // CellEquation and CellConnection byte equal to the kCannotBeSqueezed build
  TEST(BuildPressureEquations, AirEndPrecedence_GasThenPocketThenAtmosphere)   // Review Focus 4: one air end
      // each: tracked gas at 300 Pa, an enclosed resolved pocket at 200 Pa, a vented pocket, a grid edge:
      // reference_pressure 300, 200, 0, 0, each plus sigma * curvature
  TEST(BuildPressureEquations, MaterialBoundaryEntry_SetsJumpAndSeriesDensity) // hand-built MaterialBoundary
      // (low_share 0.3, sigma 0.03, curvature 2): pressure_jump == -0.06, face_density in series
  TEST(BuildPressureEquations, CellWithoutParticles_TakesNeighbourDensity)     // lowest-index touching cell
      // with mass; none -> lightest_density
  TEST(BuildPressureEquations, ConnectionsOrderedByFacePiece)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 13, 14, 15 and the start of 16, from
  `git show main:native/src/mac_pressure_3d_arithmetic.h` lines 263-313 (right side), 326-399 (components,
  without the pin) and `mac_pressure_3d_system.cc` lines 279-473 (face-piece assembly). Free-solid terms
  `dt * push^T * inverse_inertia * push` are applied as `dt * (f_i . invM F + t_i . invI T)` with `F`, `T`
  summed over the solid's pushes in their sorted order.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Build the pressure equation of every liquid cell`. New
  names include the four plan additions with reasons.

### Task 24: Blobs, the sealed-blob check and added volume rates

**Amended 2026-11-14 (reconcile):** blob totals by the per-group block rule (block partials, then a tree sum), not
one thread per blob.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_equations.h`, `pressure_equations_cpu.cc`, `pressure_equations_test.cc`

**Interfaces:**
- Produces: `ApplyAddedVolumeRate`, `ComputeBlobOfCell` exactly; `incompatible_volume_rate` and the
  `SealedBlobCheck` behaviour of step 16.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), full box 4 x 4 x 4 of 0.1 m, `distance_to_surface = -1`:

  ```cpp
  TEST(BuildPressureEquations, SealedBlobOneSidePushedIn_IsVolumeMismatch)  // design 7: low X wall driven at
      // U = 0.1: kSealedBlobVolumeMismatch, magnitude 0.016 (U * area) within 1e-12 relative, limit 2e-5 * 0.064
  TEST(SealedBlobCheck, WhenRatesApplied)              // design 7: sealed gas box (kGas snapshot) squeezed:
      // kWhenBuilt fails; kWhenRatesApplied builds with incompatible_volume_rate 0.016; ApplyAddedVolumeRate
      // with rates summing to -0.016 passes; with zero rates fails with the same magnitude; 1e-12 relative
  TEST(ComputeBlobOfCell, TwoPoolsAndAChamber)         // design 7: each cell's blob has first_cell_index equal to
      // the smallest connected cell; kNoBlob elsewhere
  TEST(BuildPressureEquations, BlobClasses)            // open pool kHasReferencePressure; squeezable chamber
                                                       // kCanBeSqueezed; full box kSealed
  TEST(BuildPressureEquations, TwoBadCells_LowerCellReported)   // design 7 "Errors": NaN carried_pressure in cells
      // 70 and 5: kBadInput, cell_index 5; a PressureEquations from an earlier successful build on the same
      // buffers now gives kWrongGridVersion in ComputeBlobOfCell (no output is readable)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the rest of step 16: blob totals by the per-group rule of
  Global Constraints (per cell block, the blob's member cells of that block in ascending cell index; then, per blob,
  `math::TreeSum` over its block partials in ascending block index).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Report the volume a sealed blob cannot absorb`.

### Task 25: Equations on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (Foundation as built):** two halves (`pressure_equations_gpu.h`), a `cpu` steps test, the scan and
the grouping through the buffers' `order_buffers`, digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/pressure_equations_gpu.h` (private), `pressure_equations_gpu.cu`,
  `pressure_equations_steps_test.cc`, `pressure_equations_gpu_test.cc`, their CMake lines
- Modify: `pressure_equations_cpu.cc` (calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`)

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the main evidence until the hardware run):
  `BuildPressureEquationsSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (the per-cell and per-connection functions
  of `pressure_equations_math.h`, the per-cell-block blob partials and the per-cell gathers, over every thread block
  and thread, launch order and reverse, on the Tasks 23 and 24 scenes: bytes equal to the CPU build).
  (`LABEL gpu`): `BuildPressureEquations, OnGpu_EqualsCpu` (scenes of Tasks 23 and
  24; counts and labels equal, coefficients within 1e-12 * scale), `BuildPressureEquations,
  OnGpuTwice_GivesIdenticalBytes` (rows and connections through `WriteResultDigest`), `ApplyAddedVolumeRate,
  OnGpu_EqualsCpu`, `ComputeBlobOfCell, OnGpu_EqualsCpu`.
- [ ] **Step 2: Implement.** Equation indexes by an integer scan of "has an equation" per cell block
  (`ComputeExclusivePrefixSum`); connection lists with `BuildGroupsByDestination` (each
  list in ascending face piece index), both with the buffers' `order_buffers`; blob labels by Task 21's
  `label_propagation.h`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Build the pressure equations on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 26: Volume correction rate, CPU and GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (Foundation as built):** two halves (`volume_correction_gpu.h`), a `cpu` steps test over every
thread block and thread, the sealed-blob sums by `block_sums.h` with the buffers' `sum_buffers` once Task 28 exists
(this task writes the per-group partials directly).

**Amended 2026-11-15 (A64):** `ComputeVolumeCorrectionRate` has no `is_grid_only_region` parameter (design 3); the
test `GridOnlyCell_IsZero` moves to Task 63 of the later group.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/volume_correction.h`, `volume_correction_cpu.cc`, `volume_correction_gpu.h`
  (private), `volume_correction_gpu.cu`, `volume_correction_math.h` (private shared: the per-cell rate and the
  per-cell-block partial of one bisection), `volume_correction_test.cc`, `volume_correction_steps_test.cc`,
  `volume_correction_gpu_test.cc`, their CMake lines

**Interfaces:**
- Produces: `VolumeCorrectionSettings`, `ComputeVolumeCorrectionRate` exactly. **Plan decision:** a sealed blob
  is found from the snapshot alone (the same labelling as Task 23 run on the geometry), because the function
  runs before `BuildPressureEquations`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; GPU ones `gpu`):

  ```cpp
  TEST(ComputeVolumeCorrectionRate, OneCell_IsLogRatioOverDuration)  // target 1, sensed 1.01, duration 0.5:
      // filled_volume * (-ln 1.01 / 0.5) added to the existing value; clamped at max_rate
  ~~TEST(ComputeVolumeCorrectionRate, GridOnlyCell_IsZero)~~       // A64: moved to Task 63 (later)
  TEST(ComputeVolumeCorrectionRate, SealedBlob_SumsToZero)          // |sum| <= 1e-12 * sum |rate|
  TEST(ComputeVolumeCorrectionRateSteps, EveryThreadBlockAndThread_EqualsCpuLoop)  // LABEL cpu: the per-cell and
      // per-cell-block functions over every thread block and thread, launch order and reverse: bytes
  TEST(ComputeVolumeCorrectionRate, OnGpu_EqualsCpu)              // 1e-12 * scale; rates through WriteResultDigest
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design step 17 from
  `git show main:native/src/liquid_eos_sensor_3d.cc` lines 437-456 and 458-569 (64 bisections for `lambda`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the volume correction rate`. Body: `CUDA: compiled, not run`.

### Task 27: Carried pressure, CPU and GPU

**Amended 2026-11-14 (Foundation as built):** two halves (`carried_pressure_gpu.h`), the per-cell function shared and
run over every thread block and thread by a `cpu` test, digests.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/carried_pressure.h`, `carried_pressure_cpu.cc`, `carried_pressure_gpu.h`
  (private), `carried_pressure_gpu.cu`, `carried_pressure_math.h` (private shared: one cell's carried value),
  `carried_pressure_test.cc`, `carried_pressure_steps_test.cc`, `carried_pressure_gpu_test.cc`, their CMake lines

**Interfaces:**
- Produces: `ComputeCarriedPressure` exactly. Face velocity at the cell centre: mean of the two face values per
  axis (pieces of a big face: area-weighted). Back point = centre - dt * velocity; value linear between the
  centres of the cell containing the back point and its same-size neighbours (trilinear on the 8 centres around).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; GPU `gpu`):

  ```cpp
  TEST(ComputeCarriedPressure, ZeroVelocity_IsCommitted)                // bytes equal
  TEST(ComputeCarriedPressure, UniformVelocityLinearPressure_IsExact)  // p = 3 x + 2 z, u = (0.1, 0, 0),
      // dt 0.5: carried == p(x - 0.05) in interior cells; 1e-12 relative
  TEST(ComputeCarriedPressure, NoCommittedValue_KeepsOwnOrFallback)
  TEST(ComputeCarriedPressureSteps, EveryThreadBlockAndThread_EqualsCpuLoop)  // LABEL cpu: the per-cell function
      // over every thread block and thread, launch order and reverse: bytes
  TEST(ComputeCarriedPressure, OnGpu_EqualsCpu)                   // 1e-12 * scale; values through WriteResultDigest
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/liquid_substep_3d_staging.cc`
  around line 77 (`AdvectLiquidPressureHistory3D`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Carry the committed pressure along the flow`. Body:
  `CUDA: compiled, not run`.

### Task 28: Pieces of the solve on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A24):** the pieces built here are the forms without a blob buffer. From Task 45 on they
accept equations with blocking-material faces (Task 45 adds the face terms); given equations with
`sealed_gas_blob_count > 0` they return `kBadInput` because the blob pressure is an input they lack, never because
of material faces. The forms that carry the sealed gas blob pressures, and `ComputeBlockingMaterialVelocities`, come
in Task 47 (`plan-decisions.md` "A24: pressure pieces").

**Amended 2026-11-14 (Foundation as built):** `block_sums.h` is CPU-only and its total takes the buffers' `SumBuffers`
(passed to `ComputeTreeSum`); the per-block partial is a shared function in `block_sum_steps.h`.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/flow_imbalance.h`, `coupled_solve_pieces.h`, `solve_pieces_cpu.cc`,
  `block_sums.h` (private, CPU only), `block_sum_steps.h` (private shared), `solve_pieces_test.cc`

**Interfaces:**
- Consumes: Tasks 9, 23, 24.
- Produces: `ComputeFaceInertia`, both `ComputePressurePush` forms, `ApplyPressure`, `ComputeFlowImbalance`,
  `GivenTerms`, `ComputeVolumeError`, `ComputeOwnWeight`, `ComputeStartPressure`, `ComputeWasSolved` exactly.
  `block_sums.h`: `ComputeBlockPartials` and `ComputeTotalOfBlocks` implementing the order of sums (Global
  Constraints) for cell and face values; used by every total from here on. **Plan decision (Foundation as
  built):** the partial of one cell block (its places in ascending index, left to right) is the
  `FLUIDITY_CPU_AND_GPU` function of `block_sum_steps.h`, which the CPU loop and the GPU function both call;
  `ComputeTotalOfBlocks(const Buffer<double>& block_partials, std::uint32_t block_count, SumBuffers& sum_buffers)` calls
  `ComputeTreeSum` with them on both processors (the buffers' `sum_buffers`; on the CPU it holds no bytes
  and gives the bits of `math::TreeSum`), so no total allocates and a CPU total has the GPU's bits.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeFlowImbalance, OneFaceInflow_IsPositive)    // design 7: one cell, 1 m3/s entering through its low
                                                          // X face: +1 m3/s within 1e-15
  TEST(ComputeFaceInertia, IsAreaDistanceDensity)         // 0 on a piece with no connection
  TEST(ComputePressurePush, LeftOut_IgnoresReferenceAndJump)
  TEST(ApplyPressure, EqualsPushOverInertia)
  TEST(ComputeVolumeError, LeftOut_IsOpenFlowMinusSqueeze)
  TEST(ComputeOwnWeight, NullScale_EqualsCellEquationOwnWeight)  // bytes equal; connections summed in ascending
                                                                 // face_piece_index after squeeze_allowance
  TEST(ComputeStartPressure, CarriedInEquationCellsZeroElsewhere)
  TEST(ComputeWasSolved, MarksConnectedPiecesAndEquationCells)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `cell_equation_math.h`; the GivenTerms forms follow the
  comments of design 3 exactly.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Expose the pressure pieces the joint solve reuses`.

### Task 29: Check, zero mean and offset on the CPU

**Amended 2026-11-14 (A24):** `ApplyZeroMeanToSealedBlobs` and `ApplyPressureOffset` accept equations with both
terms, arithmetic unchanged; the `CheckFlowImbalance` form with given sealed gas blob pressures (the joint solve's)
comes in Task 47.

**Kind:** numerics

**Files:**
- Modify: `flow_imbalance.h`, `coupled_solve_pieces.h`, `solve_pieces_cpu.cc`, `solve_pieces_test.cc`

**Interfaces:**
- Produces: `SolveSettings`, `FlowImbalanceReport` (declared in `pressure_solve.h`, created here with these two
  types only), `CheckFlowImbalance`, `ApplyZeroMeanToSealedBlobs`, `ApplyPressureOffset` exactly.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(CheckFlowImbalance, WrongPressure_ReportsHandComputedError)  // design 7 "Check is independent": the
      // two-cell fixture of Task 34 with p = (1.7, 1.4): max_volume_error_rate, worst_cell_index,
      // net/total_volume_change, equation_error_norm equal values computed in the test; 1e-12 relative
  TEST(CheckFlowImbalance, JumpPresent_AddsNormLimit)              // jump_error_fraction applies only then
  TEST(ApplyZeroMeanToSealedBlobs, VolumeWeightedMeanIsZero)
  TEST(ApplyPressureOffset, SealedMeanEqualsCarriedMean)           // other blobs unchanged
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 16 (offset) and 21 (check) from the velocity
  only; the check never reads the solver's vectors.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the one stop rule of every pressure solve`.

### Task 30: Pieces on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A24):** the A24 forms of the pieces (Task 47) get their GPU forms in Task 48
(`CoupledSolvePieces, OnGpu_EqualsCpu`): blob buffers whole on the GPU, blob sums by the per-group rule.

**Amended 2026-11-14 (Foundation as built):** two halves (`solve_pieces_gpu.h`), a `cpu` steps test over every thread
block and thread, totals through `ComputeTotalOfBlocks` with the buffers' `sum_buffers`, digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/solve_pieces_gpu.h` (private), `solve_pieces_gpu.cu`,
  `solve_pieces_steps_test.cc`, `solve_pieces_gpu_test.cc`, their CMake lines
- Modify: `solve_pieces_cpu.cc` (calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`)

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the main evidence until the hardware run):
  `SolvePiecesSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (the per-face and per-cell functions and the
  `block_sum_steps.h` partial, over every thread block and thread, launch order and reverse, on the Task 23 scene:
  bytes equal to the CPU pieces).
  (`LABEL gpu`): `SolvePieces, OnGpu_EqualsCpu` (every function of Tasks 28-29 on
  the Task 23 scene, 1e-12 * scale), `CheckFlowImbalance, OnGpuTwice_GivesIdenticalBytes` (the report through
  `WriteResultDigest`).
- [ ] **Step 2: Implement.** Per-face and per-cell functions over `cell_equation_math.h`; totals through
  `block_sums.h` (one thread per cell block, then `ComputeTotalOfBlocks`, `ComputeTreeSum` with
  the buffers' `sum_buffers`, over the block partials).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Run the pressure pieces on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 31: Ladder rungs on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers. Test library `fluidity_grid_test_support` (reconcile item 29).

**Amended 2026-11-14 (A15):** rung 1 is built block by block so that a batched solve (Tasks 58, 59) streams only
rung 0 and builds rung 1 per batch; rung 0 is never terminal; the direct solve has a room. One piece uses the same
layout, so both modes have the same ladder.

**Amended 2026-11-14 (Foundation as built):** tests `LABEL cpu` (was `no_gpu`); every `BuildStableOrder` and
`ComputeExclusivePrefixSum` takes the buffers' `order_buffers`.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/pressure/pressure_ladder.h`, `pressure_ladder_cpu.cc`, `pressure_ladder_math.h`
  (private shared), `pressure_ladder_test.cc`
- Modify: `docs/glossary.md` (`ladder`, `rung`, `connection_scale`), `pressure_buffers.h` and
  `pressure_memory_estimate.cc` (A15: `BufferSizes::max_direct_solve_values`, the factor row counted from it)

**Interfaces:**
- Produces: `BuildPressureLadder` exactly. **Plan decisions:** rung 0 is the equations, each connection weighted
  `flow_per_pressure * connection_scale`, own weight `squeeze_allowance` plus those weights (free-solid terms are
  left out of the ladder; it is a preconditioner). Merge key of a coarse cell: first pass `BuildStableOrder` on
  the merged `CellId.value`, second pass on `blob_index` (so coarse cells are ordered by blob, then id, members
  in ascending fine index). Merge rule (design step 18): while a blob holds cells of more than one size level,
  siblings of the smallest level present merge with `grid::ComputeBigCell`, other cells stay alone; at level 0,
  `cell_coord` halves with a floor. A blob stops at 32 coarse cells or fewer (terminal). Coarse weights are plain
  sums of crossing fine weights; coarse own weight = sum of member squeeze allowances plus crossing weights.
- **Plan decisions (A15), both modes:** rung 1 is built per cell block: its merge (the rule above) puts each coarse
  cell's members inside one cell block (siblings share a block; a level-0 2 x 2 x 2 group lies in one block because
  blocks are 4 cells wide and aligned), so rung 1 is numbered by (block, `blob_index`, merged `CellId.value`): per
  block one thread orders its at most 64 keys, per-block counts go through `ComputeExclusivePrefixSum`, and no
  grid-wide sort is made for rung 1. Rungs 2 and deeper are built from rung 1 by the rule above (two passes of
  `BuildStableOrder`). (Foundation as built: both calls, and the `ComputeExclusivePrefixSum` above, take
  the buffers' `order_buffers`.) Rung 0 is never terminal: a blob of at most 32 equations passes one to
  one into rung 1, its
  cells are skipped by rung 0's weighted Jacobi, and its direct solve runs at rung 1 (the same mathematics as a
  direct solve of the equations). `BufferSizes` gains `std::uint32_t max_direct_solve_values` (doubles; 0 means
  `32 * max_cells`, the old bound); terminal blobs take factor room in ascending first-cell order and a blob beyond
  the room gets one weighted-Jacobi pass down and one up at its terminal rung instead (Task 32), counted for
  `SolveResult::blobs_without_direct_solve` (Task 34).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildPressureLadder, CoarseEqualsMergeOfFineOfSpread)   // design 7 "Ladder": every rung of a two-level
      // case with two blobs, random v: coarse(v) == merge(fine(spread(v))) within 1e-12 relative
  TEST(BuildPressureLadder, NegativeCoordinates_MergeWithFloor) // Review Focus 2: grid origin at (-0.8,-0.8,-0.8),
      // cells at coords -3..2: members of each coarse cell match floor(c / 2) by hand
  TEST(BuildPressureLadder, MergeNeverCrossesABlob)
  TEST(BuildPressureLadder, SmallestLevelMergesFirst)
  TEST(BuildPressureLadder, RebuiltBuffers_IsWrongGridVersion)
  TEST(BuildPressureLadder, FirstRungMembersShareOneBlock)     // A15: 100 random three-level grids
      // (grid_test_support), two blobs each: every rung-1 coarse cell's members lie in one block; rung-1 order is
      // (block, blob, merged id), checked against a test-side sort
  TEST(BuildPressureLadder, BlobOfTwentyEquations_PassesOneToOneIntoRungOne)  // A15: a 20-cell drop beside a
      // big pool: 20 rung-1 cells with one member each; the drop is terminal at rung 1
  TEST(BuildPressureLadder, DirectSolveRoomFull_LaterBlobsWithoutDirectSolve)  // A15: three 10-cell drops, room
      // 250 values: the first two (ascending first cell) get factors, the third is counted without
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `git show main:native/src/mac_pressure_3d_multigrid.cc`
  lines 102-165 (coarsening; replace the pairing fallback and `floor(coord / 2)` inside connected blocks with
  the size-level rule), coarse sums by one thread per coarse cell over its fixed member list.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Build the pressure ladder by size level`.

### Task 32: Ladder pass on the CPU

**Amended 2026-11-14 (A15):** per-group totals for sealed blobs, the pass without direct solve beyond the factor
room, and no rung-0 smoothing of blobs that pass one to one into rung 1 (Task 31).

**Amended 2026-11-14 (Foundation as built):** tests `LABEL cpu` (was `no_gpu`).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_ladder.h`, `pressure_ladder_cpu.cc`, `pressure_ladder_test.cc`

**Interfaces:**
- Produces: `ApplyPressureLadder` exactly. **Plan decision** for symmetry: each `kSealed` blob's input total is
  first removed, shared out in proportion to `filled_volume` (the transpose of the mean removal); then one pass
  (zero guess, weighted Jacobi 2/3 down, gather, Cholesky of each terminal blob with its first coarse cell
  grounded, spread, weighted Jacobi 2/3 up); then each sealed blob's volume-weighted mean is removed. **Amended
  2026-11-14 (A15):** blob totals and means by the per-group rule (Global Constraints); a terminal blob without
  factor room (Task 31) gets one weighted-Jacobi pass down and one up instead of its direct solve; rung-0 cells of a
  blob that passes one to one into rung 1 are not smoothed at rung 0.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ApplyPressureLadder, PreconditionerIsSymmetricAndPositive)  // design 7 "Ladder as preconditioner":
      // connection_scale random in [0.1, 1] from a fixed formula: x.L y == y.L x, x.L x > 0 per blob for x
      // not volume-proportional on a sealed blob; 1e-12 relative
  TEST(ApplyPressureLadder, ScaleOne_EqualsLadderSolvePressureBuilds)   // null scale and all-1 scale: same bytes
  TEST(ApplyPressureLadder, ThirtyTwoCellBlob_IsSolvedExactly)          // a blob of 32 cells: L == A^-1 on
                                                                         // compatible input, 1e-12 relative
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `mac_pressure_3d_multigrid.cc` lines 167-214 (direct
  blocks) and `mac_pressure_workspace_3d.cc` lines 299-398 (one pass), all `double` (design decision 4).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add one pass down and up the pressure ladder`.

### Task 33: Ladder on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (Foundation as built):** two halves (`pressure_ladder_gpu.h`), a `cpu` steps test over every thread
block and thread, the sort through the buffers' `order_buffers`, digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/pressure_ladder_gpu.h` (private), `pressure_ladder_gpu.cu`,
  `pressure_ladder_steps_test.cc`, `pressure_ladder_gpu_test.cc`, their CMake lines
- Modify: `pressure_ladder_cpu.cc` (calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`),
  `pressure_ladder_math.h` (the per-blob factorisation and solve become `FLUIDITY_CPU_AND_GPU` functions the CPU
  loop also calls)

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the main evidence until the hardware run):
  `PressureLadderSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (the per-coarse-cell sums, the gather, spread and
  weighted-Jacobi functions and the per-blob direct solve of `pressure_ladder_math.h`, over every thread block and
  thread, launch order and reverse, on Task 31's two-level case: bytes equal to the CPU ladder).
  (`LABEL gpu`): `BuildPressureLadder, OnGpu_EqualsCpu` (rung count, member lists
  equal; weights within 1e-12 * scale), `ApplyPressureLadder, OnGpu_EqualsCpu`, `ApplyPressureLadder,
  OnGpuTwice_GivesIdenticalBytes` (output through `WriteResultDigest`).
- [ ] **Step 2: Implement.** Each rung built on the device (sort by merge key with
  `BuildStableOrder` and the buffers' `order_buffers`, one thread per coarse cell);
  gather down, spread up, one thread per terminal blob for the direct solve. No rung is built on the CPU.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Build and apply the pressure ladder on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 34: `SolvePressure` on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** the iteration is written in the arithmetic both modes share, so that Task 59 only
adds sweep boundaries; `SolveResult` gains `blobs_without_direct_solve`.

**Amended 2026-11-14 (Foundation as built):** `rz`, `dq` and every other total of the iteration go through
`ComputeTotalOfBlocks` with the buffers' `sum_buffers`; nothing allocates per iteration.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_solve.h`, `pressure_test_scenes.h`, `pressure_test_scenes.cc`
- Create: `pressure_solve_cpu.cc`, `pressure_solve_test.cc`

**Interfaces:**
- Produces: `SolvedArrays`, `SolveResult`, `SolvePressure` exactly. **Plan decisions:** a progress read happens
  before the first iteration and then every `iterations_per_progress_read` iterations (so a cancel set before the
  call is always seen); the conjugate gradient's right side for a sealed blob has its total removed shared by
  `filled_volume` (the check still uses the true right side); when the running error passes the stop rule the
  solve corrects the velocity and runs `CheckFlowImbalance`; if that fails it restarts from the true error.
- **Plan decisions (A15), the arithmetic of one iteration in both modes** (`plan-decisions.md` "Batched solves
  (A15)"): `q_local = A_local d` (own term, connections, later material faces); per free solid `P_s = push_s . d`
  by the per-group rule; `k_s = dt * inverse_inertia_s * P_s` (Task 47a, A59: `k_s = dt * G_s P_s`); `dq = d . q_local + sum_s P_s . k_s` (Task 47 adds
  the sealed gas blob part); wherever `q` is used, `q_c = q_local_c + sum over c's push entries of push . k_s`. In
  the ladder's sealed-blob steps `E_b` is the blob's error total and `m_b` the blob's volume-weighted mean of the
  ladder's raw output `z_raw`; `z = z_raw - m_b`; `rz = error . z_raw - sum_b m_b E_b`. Every total by Global
  Constraints' two rules. `SolveResult` gains `std::uint32_t blobs_without_direct_solve = 0;` (Task 31's room).
  (Foundation as built: every grid-wide total of the iteration, `rz`, `dq`, the norms of the stop test, is
  `ComputeTotalOfBlocks` with the buffers' `sum_buffers` (passed to `ComputeTreeSum`), on both processors;
  the solve allocates nothing after `BuildPressureBuffers`.)

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(SolvePressure, TwoCellsExteriorPressureAndSolidFlow)  // design 7 "Two cells": BuildTwoLevelGrid with 2 x 1
      // x 1 big cells of width 1, the second split: big cell L (V 1) and small cell S at small coords (2, 0, 0)
      // (V 0.125); the other 7 small cells kSolid; hand-built snapshot. L-S piece: open 0.25, density 1,
      // pressure_distance 0.75. S's low-Y grid-edge piece: area 0.25, open 0.125, kLiquidToAir, distance 0.25,
      // solid_volume_rate +0.0625 (into S); reference 1 Pa via surface_tension_with_air 1 and S curvature 1.
      // L's low-X grid-edge piece: open 0, solid_volume_rate +0.125. dt 1. Expected p = (1.75, 1.375) Pa,
      // L-S velocity +0.5 m/s, S edge velocity -1.5 m/s; 1e-9 relative
  TEST(SolvePressure, TwoCellsAddedRateAndBlockedInteriorPiece)  // second fixture, same grid: L-S piece open
      // 0.125 of 0.25 with solid_volume_rate +0.0625; L added_volume_rate 0.09375; S low-Y edge open 0.25,
      // distance 0.25, no solid, reference 1 Pa: p = (1.28125, 1.09375), velocities +0.25 and -0.375; 1e-9 rel
  TEST(SolvePressure, LinearPressureInSealedCube)            // design 7: cube [-1,1]^3, w 0.25 uniform and the
      // two-level layout (0.5 m for x < 0.5), dt 0.5, density 2, X pieces start at 0.1875 m/s: velocity after
      // within 2e-5 1/s per cell, pressure minus its mean equals 0.75 x minus its mean within 1e-4 Pa
  TEST(SolvePressure, SealedCubeWithLargeCarriedPressure_KeepsMean)  // Review Focus 1: same with carried 1e5 Pa
  TEST(SolvePressure, CancelSet_ReturnsCancelled)            // design 7: hydrostatic column (BuildHydrostaticColumn,
      // added here to pressure_test_scenes and reused by Task 36), cancel set before the call: kCancelled; with
      // null, the result equals a run without the pointer, bytes
  TEST(SolvePressure, OutputsOfOtherGridVersion_IsWrongGridVersion)
  TEST(SolvePressure, ZeroIterationsAllowed_IsNotConverged)
  TEST(SolvePressure, FreeSolidRankOne_DqEqualsDirectProduct)   // A15: Task 23's scene with its free solid, 10
      // random d: dq from the shared arithmetic equals d . (A d) summed directly in the test, 1e-12 relative
  ```
  Source of the two-cell numbers: `git show main:docs/adaptivity/native-pressure-boundaries.md`,
  `adaptive-pressure-graph-inputs.md`, and `native/tests/test_mac_pressure_3d_system.cc` lines 52-120 and 255-360
  on `main`, re-expressed with `s = +1` for the low cell.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 19 to 21 with `// Step N:` comments, from
  `git show main:native/src/mac_pressure_workspace_3d.cc` lines 462-659 and `mac_pressure_3d_multigrid.cc`
  lines 341-357, without the pin, FP32 vectors and dense path (design section 10).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Solve for the pressure of scenes without viscosity`.

### Task 35: `SolvePressure` on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (Foundation as built):** two halves (`pressure_solve_gpu.h`), per-cell iteration updates in
`pressure_solve_steps.h` with a `cpu` steps test, totals with the buffers' `sum_buffers`, digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/pressure_solve_gpu.h` (private), `pressure_solve_gpu.cu`,
  `pressure_solve_steps.h` (private shared: the per-cell updates of one iteration), `pressure_solve_steps_test.cc`,
  `pressure_solve_gpu_test.cc`, their CMake lines
- Modify: `pressure_solve_cpu.cc` (its loop calls `pressure_solve_steps.h`; the GPU half under
  `#if defined(FLUIDITY_HAS_GPU)`)

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the main evidence until the hardware run):
  `SolvePressureSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (the per-cell updates of one iteration and the
  block partials of `rz` and `dq`, over every thread block and thread, launch order and reverse, on Task 34's
  sealed cube: bytes equal to one CPU iteration).
  (`LABEL gpu`): `SolvePressure, OnGpu_EqualsCpu` (Task 34 cases; velocity and
  pressure bounds of the Global Constraints), `SolvePressure, OnGpuTwice_GivesIdenticalBytes` (pressure and
  velocity through `WriteResultDigest`), `SolvePressure, OnGpuCancel_ReturnsCancelled`.
- [ ] **Step 2: Implement.** `rz`, `dq` and the step sizes stay on the device (totals by `ComputeTotalOfBlocks`
  with the buffers' `sum_buffers`); the CPU reads the progress record
  (iteration, finished flag, `PressureFailure`, `rz`, `dq`) at each progress read; every function returns at once
  when the finished flag is set.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Solve for the pressure on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 36: Scene tests, hydrostatic column, driven solid, static drop

**Kind:** integration

**Files:**
- Create: `native/fluidity/pressure/pressure_scene_test.cc`
- Modify: `docs/known-open-physics.md`

Every test runs the chain `FindLiquidSurface` (analytic surface and solid from Task 13) ->
`FindAirPockets` -> `BuildPressureEquations` -> `SolvePressure`, on `BuildBoxGrid` and `BuildTwoLevelGrid`.
`rho = 1000`, `g = 9.81`, `dt = 0.01`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(PressureScene, HydrostaticColumn)          // design 7, BuildHydrostaticColumn of Task 34: open-top box
      // 0.4 x 0.4, w 0.1 (two-level: big 0.2 below z 0.4, small 0.1 above), H 0.43 (two-level 0.53),
      // Z pieces start at -g dt: p == rho g (H - z) at centres within the pressure bound; |u| after <= 2e-5 * w
  TEST(SolvePressure, WritesStepArrays)           // design 7: caller-owned
      // SolvedArrays and AirPocketArrays hold the results; a piece with no connection keeps its
      // velocity_before_pressure, bytes
  TEST(PressureScene, DrivenSolidSweptVolume)     // design 7: tank 0.8 x 0.4, w 0.1, H 0.27, solid plane
      // x = 0.05 moving +x at U 0.1: outflow through air ends == sum of solid_volume_rate of equation cells
      // within 2e-5 * liquid volume
  TEST(PressureScene, PistonReceivingVolume_KnownOpen)  // WILL_FAIL, LABEL known_open: same scene; outflow ==
      // U * wetted wall area (0.27 * 0.4); fails because the surface cell receives its whole blocked area
  TEST(PressureScene, StaticDrop)                 // design 7: R = 8 w and 16 w, w 1 mm, sigma 0.0728, no gravity,
      // dt 5e-4: surface curvature 2/R, mean inside pressure 2 sigma / R (2 % at 8 w, 0.5 % at 16 w);
      // max |u| after at 16 w below that at 8 w
  ```
- [ ] **Step 2: Run.** Expected: all pass except the known-open one, which fails as stated (CTest reports it
  passed because of `WILL_FAIL`). If another fails, fix the owning commit (rebase), not the test.
- [ ] **Step 3: Add** the piston line to `docs/known-open-physics.md` (test name, what it measures, why it fails).
  Run the gate. **Commit.** Subject: `Test the pressure module on whole liquid scenes`.

### Task 37: Scene tests, blobs and a container moved as one

**Kind:** integration

**Files:**
- Modify: `native/fluidity/pressure/pressure_scene_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(PressureScene, SealedContainerMovedAsOne)  // design 7: full box 4^3 of 0.1 m, all six sides driven at
      // (0.1, 0, 0): incompatible_volume_rate 0 within 1e-12 of sum |solid_volume_rate|; velocity after equals
      // the solid's (X pieces 0.1, others 0) within the solve limit
  TEST(PressureScene, TwoBlobsOneSealed)          // design 7: an open pool and a full closed chamber with
      // carried 500 Pa, separated by a solid wall: the pool is referenced to air; the chamber's volume-weighted
      // mean pressure is 500; the same scene mirrored in x (other cell numbering) gives the same pressure at
      // each position within 1e-9 relative
  ```
- [ ] **Step 2: Run**, then the gate. **Commit.** Subject: `Test sealed and open blobs together`.

### Task 38: Scene tests, gas beside liquid and two liquids

**Amended 2026-11-14 (A18):** `RisingGasPressureFollowedEachSolve` now names the path it tests (gas read one
step late: open gas, slivers, held blobs, cells that touch gas); the sealed blob solved inside is Task 50.

**Amended 2026-11-14 (A24):** "viscous scenes" left the list above: the joint viscous solve carries sealed gas
blobs too. The test is unchanged.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/pressure/pressure_scene_test.cc`, `docs/known-open-physics.md`

- [ ] **Step 1: Failing tests** (`LABEL cpu`), hydrostatic column of Task 36 unless stated:

  ```cpp
  TEST(PressureScene, GasCellsSameAsLiquid)            // design 7 "CellContent::kGas, SameAsLiquid": snapshot with
      // every kLiquid set to kGas: identical equations, solve and report bytes
  TEST(PressureScene, GasPressureAtLiquidSurface)      // design 7: cells above the water hold tracked gas at
      // P = 200 Pa: p == P + rho g (H - z); blob kHasReferencePressure; gas cells pocket_of_cell kNoAirPocket
  TEST(PressureScene, RisingGasPressureFollowedEachSolve)  // the path read one step late (no SealedGasBlob
      // given): three solves with gas pressure 100, 150, 225 Pa passed as the committed array: the surface cell's
      // pressure in solve k follows the k-th array (the one-step lag is the caller's), within the solve limit
  TEST(PressureScene, TwoLiquidsAtRest_NotSupportedYet)  // design 7 "Two liquids at rest", first state: oil
      // (800) over water (1000) in a closed box, both grids: FindMaterialBoundaries gives
      // kMaterialBoundaryNotSupported
  TEST(PressureScene, TwoLiquidsAtRest_SharpBoundary)    // WILL_FAIL, LABEL known_open: the same scene solved
      // with the boundaries: pressure jumps by rho_oil g h across layers, low_share exact for a flat boundary
      // (1e-12), velocity after 0 (solve limit)
  ```
- [ ] **Step 2: Run**; add the sharp-boundary line to `docs/known-open-physics.md` ("capability exclusion
  `kMaterialBoundaryNotSupported`, lifted by Task 42 when `FindMaterialBoundaries` returns its entries"). Run the
  gate.
  **Commit.** Subject: `Test gas beside liquid and two liquids that do not mix`.

### Task 39: Repeat run and the GPU scene runs

**Amended 2026-11-14 (Foundation as built):** the `gpu` scene tests pass pressure, velocity and the pocket list to
`WriteResultDigest`.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/pressure_scene_gpu_test.cc`
- Modify: `native/fluidity/pressure/pressure_scene_test.cc`

- [ ] **Step 1: Tests:**

  ```cpp
  // LABEL cpu
  TEST(PressureScene, RepeatRun_IdenticalBytes)          // design 7 "Repeat run": Tasks 36-38 scenes twice:
      // snapshot, equations, pressure, velocity, report, pocket list identical bytes
  // LABEL gpu
  TEST(PressureScene, OnGpu_EqualsCpu)                   // every Task 36-38 scene, section 6 tolerances
  TEST(PressureScene, OnGpuRepeatRun_IdenticalBytes)     // pressure, velocity, pocket list through WriteResultDigest
  TEST(PressureScene, TwoMillionEquations_Converges)     // 128^3 full sealed box (2,097,152 equations), carried 0,
      // one driven side matched by an added rate: converges; prints iterations and milliseconds per iteration
  ```
- [ ] **Step 2: Run** the CPU test and the gate. **Commit.** Subject: `Check repeat runs and list the GPU scene
  runs`. Body: `CUDA: compiled, not run`.

### Task 40: Sharp boundary between two liquids on the CPU

**Amended 2026-11-14 (Foundation as built):** `material_boundary_math.h` is modified (Task 20 created it with the
crossing check).

**Kind:** numerics

The design determines the method (steps 1b and 13): boundary distance `(d_a - d_b) / 2` from the two
one-material fields, negative on the side of the lower material index `a`; a piece between two `kLiquid` cells
is crossed when the two centre values differ in sign; `low_share = d_low / (d_low - d_high)` raised to at least
0.05; `face_density` in series with `d_low = low_share * pressure_distance`, `d_high = (1 - low_share) *
pressure_distance` and the two cells' `mixture_density`; `pressure_jump = - sigma * surface_curvature`, with the
curvature of step 6 taken from boundary-distance samples at the low cell, negated when the low cell's centre is
on `b`'s side, so it is positive when the low cell's liquid bulges out. Task 23 already turns a `MaterialBoundary`
into a jump and a series density; this task produces the entries. The public `FindMaterialBoundaries` keeps its
exclusion until Task 42.

**Files:**
- Create: `material_boundary_internal.h` (private CPU), `material_boundary_scene_test.cc`
- Modify: `native/fluidity/pressure/material_boundary_math.h` (private shared, one crossed piece; created by
  Task 20), `material_boundary_cpu.cc`, `distance_to_surface_cpu.cc`, `distance_to_surface_test.cc`,
  `pressure_test_scenes.h`, `pressure_test_scenes.cc`

**Interfaces:**
- Consumes: Tasks 6, 14, 16, 20, 23, 34.
- Produces (private): `std::expected<std::span<const MaterialBoundary>, PressureFailure>
  FindCrossedPieces(const MaterialBoundaryInputs& inputs, PressureBuffers& buffers)` (the entries, sorted by
  `face_piece_index`; what `FindMaterialBoundaries` will return); shared `MaterialBoundary
  ComputeMaterialBoundaryPiece(...)` in `material_boundary_math.h`. Test scenes gain
  `BuildTwoLiquidSurfaces(grid, boundary_height_function, top_surface_height)` giving the all-liquids field and
  the two one-material fields.
- **Plan decision** (the design leaves the sample set of a one-material field open): with `material_index` set
  and `surface_tension_max > 0`, `around_surface_cells` holds every cell whose centre distance has magnitude
  below 2 cell widths, on both sides of that field's zero, so both fields have the 18 samples at every cell beside
  the boundary. The all-liquids field keeps Task 14's rule.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), oil (`rho 800`, material 1) over water (`rho 1000`, material 0),
  `g = 9.81`, closed box 0.4 x 0.4 x 0.8, `w = 0.05` and the two-level grid, `mixture_density` hand-built as the
  density of the material at each cell centre (what the step's particle sums give away from the boundary):

  ```cpp
  TEST(ComputeDistanceToSurface, OneMaterialField_SamplesBothSidesOfItsZero)
  TEST(FindCrossedPieces, FlatBoundary_LowShareExact)          // boundary z = 0.2375: every crossed Z piece has
      // low_share == 0.25 (distance from the low centre over the centre distance) within 1e-12; curvature 0
  TEST(FindCrossedPieces, OilDrop_CurvatureTwoOverRadius)      // oil sphere R = 8 w in water: curvature within 2 %
      // of 2/R, positive where the low cell is oil, negative where it is water; sigma copied from the input
  TEST(TwoLiquids, FlatLayersAtRest_HydrostaticWithJump)       // the design's "Two liquids at rest" through the
      // private path: crossed entries -> BuildPressureEquations -> SolvePressure. Across each crossed Z piece
      // p_low - p_high == g (1000 * 0.0125 + 800 * 0.0375); away from it p follows rho g of its layer, so the oil
      // layer of height h adds 800 g h; pressure bound of the Global Constraints; |u| after <= 2e-5 * w
  TEST(TwoLiquids, OilDropNoGravity_JumpIsTwoSigmaOverRadius)  // sigma 0.03: mean oil pressure minus mean water
      // pressure == 2 sigma / R within 2 % at R = 8 w
  TEST(TwoLiquids, LighterLayerStaysFlatForHundredSteps)       // moving case: flat boundary, 100 steps of
      // dt = 0.005: each step velocity_before_pressure = previous velocity_after_pressure - g dt on Z pieces, the
      // boundary height moved in the test by dt times the mean vertical velocity of its crossed pieces, fields
      // rebuilt: |height - 0.2375| <= 1e-3 * w after every step; max |u| <= 2e-5 * w
  TEST(TwoLiquids, BumpUnderLighterLayer_DoesNotGrow)          // same loop, boundary 0.2375 + 0.1 w cos(2 pi x /
      // 0.4): bump amplitude after 100 steps <= 1.05 * 0.1 w (stable layering)
  TEST(TwoLiquids, BumpUnderHeavierLayer_Grows)                // densities swapped: amplitude after 100 steps >
      // 0.1 w (sign check of the jump and the series density)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** `ComputeMaterialBoundaryPiece` and `FindCrossedPieces` with
  `// Step 1b:` comments; `FindMaterialBoundaries` now calls `FindCrossedPieces` and returns the exclusion when
  the span is not empty. From `git show main:native/src/liquid_eos_sensor_3d.cc` lines 289-435 and
  `liquid_boundary_geometry_3d.cc` lines 1175-1235 (crossings, series density, signed jump; `main` has one
  configured pair on uniform cells, here any two materials on any cell sizes).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Compute the sharp boundary between two liquids`.

### Task 41: Sharp boundary between two liquids on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (Foundation as built):** a `cpu` steps test of `ComputeMaterialBoundaryPiece` over every thread
block and thread; the compaction through the buffers' `order_buffers`; digests.

**Kind:** device

**Files:**
- Modify: `native/fluidity/pressure/material_boundary_gpu.cu`, `material_boundary_gpu.h`,
  `material_boundary_steps_test.cc`, `material_boundary_gpu_test.cc`, `distance_to_surface_gpu.cu`

- [ ] **Step 1: Failing tests** (`LABEL gpu`): `FindCrossedPieces, OnGpu_EqualsCpu` (flat boundary and oil drop
  of Task 40: entry count and `face_piece_index` equal, `low_share`, curvature within 1e-12 * scale),
  `FindCrossedPieces, OnGpuTwice_GivesIdenticalBytes` (entries through `WriteResultDigest`), `TwoLiquids,
  FlatLayersAtRestOnGpu_EqualsCpu` (solve tolerances of the Global Constraints). (`LABEL cpu`, the main evidence
  until the hardware run): `FindCrossedPiecesSteps, EveryThreadBlockAndThread_EqualsCpuLoop`
  (`ComputeMaterialBoundaryPiece` over every thread block and thread, launch order and reverse, on Task 40's flat
  boundary and oil drop: bytes equal to the CPU entries).
- [ ] **Step 2: Implement.** One thread per face piece calling `ComputeMaterialBoundaryPiece`; crossed pieces
  compacted in ascending index with `BuildGroupsByDestination` (the buffers' `order_buffers`);
  the one-material sample rule of Task 40 in the GPU distance function.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Compute the sharp boundary between two liquids on the GPU`.
  Body: `CUDA: compiled, not run`.

### Task 42: Lift the exclusion for liquids that do not mix

**Amended 2026-11-15 (A58):** after the removal `PressureError` has 16 values, `kOutOfMemory` last.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/pressure/pressure_error.h`, `pressure_error_message.cc`, `pressure_error_test.cc`,
  `material_boundary.h`, `material_boundary_cpu.cc`, `material_boundary_gpu.cu`, `material_boundary_test.cc`,
  `material_boundary_scene_test.cc`, `pressure_scene_test.cc`, the CTest lines of `pressure_scene_test.cc`,
  `docs/known-open-physics.md`, `docs/method-paper/sections/pressure.typ`, `docs/glossary.md`

- [ ] **Step 1: Change the tests first.** In Task 38's file, delete `TwoLiquidsAtRest_NotSupportedYet`; remove
  `WILL_FAIL` and `known_open` from `TwoLiquidsAtRest_SharpBoundary` (now `LABEL cpu`: drop its `DISABLED_` prefix
  and its `add_test` line, Foundation as built), which runs the public
  path `ComputeDistanceToSurface` (all liquids, then each material) -> `FindLiquidSurface` ->
  `FindMaterialBoundaries` -> `BuildPressureEquations` -> `SolvePressure` and asserts the design's values. Switch
  Task 40's `TwoLiquids` tests from `FindCrossedPieces` to `FindMaterialBoundaries`. Replace
  `FindMaterialBoundaries, CrossedPiece_IsMaterialBoundaryNotSupported` with
  `FindMaterialBoundaries, CrossedPieces_ReturnedInFaceOrder`. `ToName, EveryError_IsLowerSnakeCaseWithoutK` now
  expects 16 values (amended 2026-11-15, A58: `kOutOfMemory` stays last; `PressureError, OutOfMemory_IsLastValue`
  still passes). Run: the new expectations fail.
- [ ] **Step 2: Implement.** `FindMaterialBoundaries` returns `FindCrossedPieces`' span (fold
  `FindCrossedPieces` into it and delete the private header); delete `PressureError::kMaterialBoundaryNotSupported`
  and its message; the function's docstring and the paper paragraph from Task 2 now describe the sharp boundary
  as delivered; remove the line from `docs/known-open-physics.md`.
- [ ] **Step 3: Check** `grep -rn "MaterialBoundaryNotSupported" native docs` finds nothing. Run the gate.
- [ ] **Step 4: Commit.** Subject: `Solve two liquids that do not mix with a sharp boundary`. Body lists the
  removed public name `PressureError::kMaterialBoundaryNotSupported` and why, and states that the step module's
  capability reason for touching liquids is no longer needed by pressure.

### Task 43: Method paper, coupling inside the pressure solve

**Amended 2026-11-14 (A18):** new task; the paper states the two couplings that moved into the solve.

**Amended 2026-11-14 (A24):** viscous scenes leave the "read one step late" list; one sentence states that the
joint viscous solve carries the same two terms.

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/pressure.typ`, `docs/method-paper/references.typ` (only if an entry below
  is missing)
- Create: `docs/method-paper/figures/coupled_faces.typ`

**Interfaces:**
- Produces labels `<sec-coupled-solve>`, `<eq:material-face>`, `<eq:sealed-gas-row>`, `<eq:sealed-gas-elimination>`,
  `<fig-coupled-faces>`.

- [ ] **Step 1: Text** from Task 44's plan decisions (a) to (f), in the decided vocabulary: why a lagged push is
  unstable when the added mass exceeds the body's mass (cite Banks, Henshaw and Schwendeman 2014, Robinson-Mosher
  et al. 2011; URLs in `docs/superpowers/specs/modules/literature-coupling.md`); the blocking-material face term
  `<eq:material-face>` (`dt A^2 / m`, the split of Fang et al. 2020: the material's internal force first, then one
  solve that holds its inertia; tangential motion stays each set's own); the sealed gas row `<eq:sealed-gas-row>`
  with its squeeze allowance `V / (heat_ratio p0 dt)` (the papers' `V / (gamma p dt^2)` with the row divided by
  `dt`; cite Aanjaneya 2013, Patkar et al. 2013) and its exact elimination `<eq:sealed-gas-elimination>`; what
  stays read one step late (open gas, gas above a free surface that is not sealed, slivers, held blobs, cells that
  touch gas) and why that is harmless there. One sentence: the joint viscous solve carries the same two terms, the
  material face eliminated and the sealed gas blob kept as one more unknown (refer to the viscosity paper section,
  "Coupled terms"). Rewrite Task 2's gas paragraph accordingly. State what is not in the
  solve: the material's own volumetric stiffness (open point, Task 44 (f)).
- [ ] **Step 2: Figure** `<fig-coupled-faces>`: a liquid cell column under a snow slab and beside a sealed gas
  pocket; the blocked piece with its face mass and the shared normal velocity; the gas pocket drawn as one extra
  node joined to the surface cells. The caption states that both move in the same solve.
- [ ] **Step 3: Run** typst, look at the pages, run the gate. **Commit.** Subject:
  `Describe the blocking materials and sealed gas solved with the liquid`.

### Task 44: Types, inputs and shared formulas of the coupling

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A18):** new task; the declarations and the method every later coupling task follows.

**Amended 2026-11-14 (reconcile):** `SealedGasBlob` comments copied from `plan-decisions.md` verbatim; the
rank-one term's sum over connections follows the per-group block rule, not one thread per blob.

**Amended 2026-11-14 (Foundation as built):** shared-formula tests `LABEL cpu` (no `fluidity_add_shared_test`); the
order buffers covers `max_blocking_material_faces`.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_equations.h`, `pressure_solve.h`, `pressure_buffers.h`,
  `pressure_buffers.cc`, `pressure_memory_estimate.cc`, `cell_equation_math.h`, `cell_equation_math_test.cc`,
  `pressure_buffers_test.cc`, `docs/glossary.md` (`blocking-material face`, `sealed gas blob`, `face mass`)

**Interfaces:**
- Produces (**Plan decision**, A18; shared types in `pressure_equations.h`):

  ```cpp
  struct BlockingMaterialFace {          // One face piece where liquid meets a blocking material set.
    grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
    std::uint32_t material_set = 0;      // Index into the step's list of blocking sets (solids writes back).
    double blocked_area = 0;             // m2, > 0: this set's corner areas (ComputeBlockedCornerAreas) summed.
    double face_mass = 0;                // kg, > 0: the set's GridFromParticles::mass on the piece.
    double velocity_before_pressure = 0; // m/s along +axis: the set's face velocity after ApplyForces, CollideSolids.
    double coverage_volume_rate = 0;     // m3/s along +axis: the part of the snapshot's solid_volume_rate this
  };                                     //   set's corners gave (sum corner_area * corner velocity . axis).
  struct SealedGasBlob {                 // One sealed tracked-gas blob that touches liquid, not held at atmosphere.
    grid::CellPlaceIndex first_cell_index{grid::kNoIndex};
    double gas_volume = 0;               // m3, > 0, step start.
    double background_pressure = 0;      // Pa absolute, > 0, step start, equalised.
    double heat_ratio = 1.4;
    double pressure_above_atmosphere = 0;  // Pa: the blob's carried value.
    double heat_volume_rate = 0;         // m3/s = ((heat_ratio - 1) / heat_ratio) * sum V q / p0.
    double boundary_volume_rate = 0;     // m3/s of solid moving into the blob through pieces that are not liquid
  };                                     //   connections (inflow positive).
  ```
  `EquationInputs` gains `std::span<const BlockingMaterialFace> blocking_material_faces` (sorted by
  `face_piece_index`, then `material_set`), `std::span<const SealedGasBlob> sealed_gas_blobs` (by
  `first_cell_index`), `const grid::CellValues<std::uint32_t>* sealed_gas_blob_of_cell = nullptr` (index into
  `sealed_gas_blobs`, `kNoBlob` elsewhere) and `const grid::CellValues<double>* gas_pressure_above_background =
  nullptr` (Pa, the gas solve's pressure of the last commit). `PressureEquations` gains
  `blocking_material_face_count` and `sealed_gas_blob_count`. `SolvedArrays` gains `std::span<double>
  blocking_material_velocity_after_pressure` (m/s, one per face, input order) and `std::span<double>
  sealed_gas_pressure` (Pa above atmosphere at the step end, one per blob). `FlowImbalanceReport` gains
  `double max_sealed_gas_volume_error_rate = 0` (1/s). `BufferSizes` gains `std::uint32_t
  max_blocking_material_faces = 0, max_sealed_gas_blobs = 0` (more entries: `kBadInput`).
- Shared formulas (`cell_equation_math.h`, all `FLUIDITY_CPU_AND_GPU`):

  ```cpp
  // flow_per_pressure = dt * A^2 / m (m3/(Pa s)); velocity_per_pressure = dt * A / m ((m/s)/Pa).
  ConnectionCoefficients ComputeMaterialFaceCoefficients(double blocked_area, double face_mass, double step_duration);
  // m3/(Pa s) = gas_volume / (heat_ratio * background_pressure * step_duration).
  double ComputeGasSqueezeAllowance(double gas_volume, double background_pressure, double heat_ratio,
                                    double step_duration);
  // m/s = velocity_before_pressure + velocity_per_pressure * (pressure_low - pressure_high).
  double ComputeMaterialVelocityAfterPressure(double velocity_before_pressure, double velocity_per_pressure,
                                              double pressure_low, double pressure_high);
  ```
- **The method (plan decisions, copied to `plan-decisions.md`).** `s = +1` for the low end of a piece.
  - (a) *Blocking-material face* `k` on piece `f`, ends `a` (low) and `b` (high): `g_k = dt A_k^2 / m_k`. An end
    is solved when its cell has an equation; an unsolved end's pressure is 0, whatever the cell holds (the rule of
    solids' motion map: gas and air-pocket pressure push no solid). Each solved end `e` gets own weight `+= g_k`
    and `flow_imbalance_before_pressure += -s_e * (A_k * velocity_before_pressure - coverage_volume_rate)`; two
    solved ends get the off-diagonal `-g_k`. After the solve `u_k = ComputeMaterialVelocityAfterPressure(...)` with
    `p` of an unsolved end 0; its flow `A_k u_k` counts in the volume error of each solved end. No solved end:
    `kBadInput` (solids builds only pieces touching a `kLiquid` cell); `face_mass <= 0` or `blocked_area <= 0`:
    `kBadInput`. A material face between two equation cells joins their blobs; one with an unsolved end is a
    reference end for `BlobClass`.
  - (b) *Sealed gas blob* `j`: `S_j = ComputeGasSqueezeAllowance(...)`, `c_j = pressure_above_atmosphere`. Its
    connections are the `CellConnection`s whose air cell has `sealed_gas_blob_of_cell == j`; such a connection's
    other-end pressure is `p_j + reference_pressure`, `reference_pressure = sigma * curvature +
    gas_pressure_above_background(air cell)`. Row of the blob: `S_j p_j + sum_c g_c (p_j - p_c) = r_j`, `r_j =
    inflow_j + heat_volume_rate + boundary_volume_rate + S_j c_j - sum_c g_c ref_c`, `inflow_j = sum_c -s_jc
    (open_area_c u_c + solid_volume_rate_c)` from `velocity_before_pressure` (`s_jc = +1` when the air cell is the
    piece's low end), `g_c = flow_per_pressure`. The unknown is eliminated exactly: `D_j = S_j + sum_c g_c > 0`,
    `p_j = (r_j + sum_c g_c p_c) / D_j`. Each cell row with blob connections keeps `g_c` in its own weight (as
    for any air end, counted once), its right side gets `g_c ref_c + g_c r_j / D_j`, and the operator subtracts `(g_c / D_j) * sum_c' g_c' p_c'` (one rank-one term
    per blob, applied like the free-solid term: the sum over the blob's connections by the per-group rule of
    Global Constraints, per cell block in ascending `face_piece_index` then `math::TreeSum` over the blob's block
    partials in ascending block index; each cell gathers). Cells joined to one gas blob are one liquid blob, class
    `kCanBeSqueezed` unless it has a reference end (`S_j > 0` makes it definite).
  - (c) *Air-end precedence* (step 13): sealed gas blob, else `holds_tracked_gas` (read one step late), else
    enclosed resolved pocket, else 0. A `sealed_gas_blob_of_cell` index past `sealed_gas_blobs`: `kBadInput`.
  - (d) *Symmetry and positiveness.* Every material term is `g (e_a - e_b)(e_a - e_b)^T` or `g e e^T` with
    `g > 0`; the blob elimination is the Schur complement of a symmetric positive semidefinite system with a
    positive pivot. Order of one cell row: own term, connections by ascending `face_piece_index`, material faces
    by ascending (`face_piece_index`, `material_set`), then blob terms by ascending `face_piece_index`.
  - (e) *Ladder:* a material face with two solved ends adds `g_k` to the pair weight of its piece (a pair is
    created if the piece has no connection); one solved end adds `g_k` to own weight; blob connections stay
    reference ends (own weight `g_c`), the rank-one term is left out, as for free solids. *Check:* each blob row
    is checked like a cell, `|inflow_after + heat + boundary - S_j (p_j - c_j)| <= max_volume_error_rate *
    gas_volume`, reported as `max_sealed_gas_volume_error_rate`, not counted in `net_volume_change`.
  - (f) *Open, not built:* the material's own volumetric stiffness inside the solve (half of the volumetric
    response in Fang et al. 2020). `materials::ApplyInternalForce` already applies the full stress, so adding a
    squeeze allowance for the material would count it twice; doing it needs materials to drop its volumetric part
    and the material cells to get rows. Recorded in the hand-over as an owner question.
- Memory estimate entries (prefix `"pressure: "`): material faces (given `max_blocking_material_faces`,
  `sizeof(BlockingMaterialFace)` + 16 for `g` and velocity per pressure + 2 x 12 for the per-equation lists + 16
  for a ladder pair); sealed gas blobs (given `max_sealed_gas_blobs`, `sizeof(SealedGasBlob)` + 24 for `D`, `r`,
  the blob sum); blob of connection (face place, 4). (Foundation as built: the order buffers's item count `n` of
  Task 12 becomes the larger of `max_face_pieces` and `max_blocking_material_faces`, since the per-equation
  material lists are grouped with it.)

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the shared formulas and the buffers):

  ```cpp
  TEST(ComputeMaterialFaceCoefficients, MatchDefinition)   // A 0.0025, m 0.0375, dt 0.01: 1.6666667e-6 and
                                                           // 6.6666667e-4; 1e-15 relative
  TEST(ComputeGasSqueezeAllowance, MatchesDefinition)      // V 3.75e-4, p0 101325, 1.4, dt 1e-3: 2.6436e-6
                                                           // (closed form in the test); 1e-15 relative
  TEST(ComputeMaterialVelocityAfterPressure, PushesFromHighToLow)  // 0 + 2e-4 * (3333.33 - 0) == 0.666667
  TEST(ComputeBufferBytes, CouplingRows)                   // 10 blocks, 100 faces, 4 blobs: difference to 0 and 0
                                                           // equals the two rows
  TEST(BuildPressureBuffers, CouplingSizesZero_SameBytesAsBefore)  // ComputeBufferBytes unchanged
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the declarations and formulas; every function that takes
  `EquationInputs` still ignores the new fields (Tasks 45 and 46 read them).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the inputs of blocking materials and sealed gas to the
  pressure equations`. New names listed with the reason "A18".

### Task 45: Blocking-material faces in the equations and the ladder, on the CPU

**Amended 2026-11-14 (reconcile):** ~~OPEN note at `ComputeVolumeError, CoupledEquations_IsBadInput`, where option
(C) of the springy-solid-beside-viscous-liquid decision would land; nothing implemented, the test unchanged.~~
**Replaced** by the A24 amendment below (owner chose option C).

**Amended 2026-11-14 (A18):** new task; Task 44 (a), (d), (e) for material faces, and the pieces' guard.

**Amended 2026-11-14 (A24):** the pieces viscosity's joint solve calls accept material faces instead of refusing
them (`plan-decisions.md` "A24: pressure pieces"). The forms without a blob buffer refuse only equations with a
sealed gas blob. Test `ComputeVolumeError, CoupledEquations_IsBadInput` is replaced by `CoupledSolvePieces,
MaterialFacesAccepted_MatchHandValues` (here) and `CoupledSolvePieces, BlobFormsWithoutBlobBuffer_IsBadInput`
(Task 46, the first task whose equations hold a sealed gas blob).

**Amended 2026-11-14 (Foundation as built):** ladder tests `LABEL cpu` (was `no_gpu`).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_equations_cpu.cc`, `pressure_equations_math.h`,
  `pressure_ladder_cpu.cc`, `pressure_ladder_math.h`, `solve_pieces_cpu.cc`, `coupled_solve_pieces.h`,
  `flow_imbalance.h` (A24: the comments state which equations each piece accepts), `pressure_equations_test.cc`,
  `pressure_ladder_test.cc`, `solve_pieces_test.cc`

**Interfaces:**
- Consumes: Tasks 23, 24, 28, 31, 32, 44. Proof-of-concept source of the term: `git show
  main:native/src/mixed_contact_3d.cc` lines 432-534 (`SolvePlanarSnowPressureTrial3D`: one face, snow mass,
  impulse `dt * area * p`, effective mass `m + dt A dp/du`) and `main:native/src/plastic_solid_boundary_3d.cc`
  lines 406-546 (`PlasticSolidPressureMobility3D::Create` and `Apply`: `B M^-1 B^T` from the same blocked-face
  corner coefficients, here lumped to the set's face mass); its use, the second solver loop this replaces, at
  `main:native/src/liquid_substep_3d.cc` lines 1766-1801 and 2213-2228.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the ladder tests too):

  ```cpp
  TEST(BuildPressureEquations, MaterialFaceBetweenTwoCells_WeighsLikeAConnection)  // two liquid cells of 0.1 m
      // with their shared piece fully blocked (open 0), entry A 0.01, m 0.5, dt 0.01: equations(x) equals a hand
      // product with g = 2e-6 between them; flow_imbalance_before of each from -s (A u* - coverage); 1e-15 rel
  TEST(BuildPressureEquations, MaterialFaceOneSolvedEnd_IsAReferenceEnd)    // closed box of liquid whose lid is
      // 16 material faces: class kHasReferencePressure; own weight includes g; no right-side reference term
  TEST(BuildPressureEquations, MaterialFaceAtCoverageVelocity_SameFlowAsWithout)   // u* equal to the coverage
      // velocity and A equal to the blocked area: flow_imbalance_before within 1e-14 relative of the build
      // without entries
  TEST(BuildPressureEquations, SymmetricAndPositiveWithMaterialFaces)       // Task 23's scene plus two faces
      // between equation cells and four with one solved end; 20 random pairs: x.A y == y.A x, x.A x >= 0; 1e-12
  TEST(BuildPressureEquations, MaterialFaceErrors_AreBadInput)              // no solved end; face_mass 0; unsorted;
                                                                            // more than max_blocking_material_faces
  TEST(BuildPressureLadder, MaterialFacesInRungZero)   // CoarseEqualsMergeOfFineOfSpread with the faces of the
      // symmetric test, including one fully blocked piece that has no connection; 1e-12 relative
  TEST(ApplyPressureLadder, SymmetricAndPositiveWithMaterialFaces)          // as Task 32's test
  TEST(CoupledSolvePieces, MaterialFacesAccepted_MatchHandValues)  // A24; the first test's fixture (two liquid
      // cells of 0.1 m, shared piece fully blocked, A 0.01, m 0.5, dt 0.01, g = 2e-6). ComputeOwnWeight adds
      // exactly 2e-6 to each cell's own weight compared with the same equations without the face, with a null
      // connection_scale and with a scale of 0.5 on every piece (g_k is never scaled). ComputeVolumeError,
      // kLeftOut, p = (1, 0), velocity 0 on every piece: the material term gives -2e-6 m3/s in the low cell and
      // +2e-6 m3/s in the high cell (-s_e g (p_a - p_b)). kIncluded: each cell's error includes -s_e (A u_k -
      // coverage_volume_rate) with u_k = ComputeMaterialVelocityAfterPressure(u*, dt A / m, p_a, p_b).
      // ComputeFaceInertia, ComputePressurePush, ApplyPressure, ComputeFlowImbalance, ApplyZeroMeanToSealedBlobs,
      // ApplyPressureOffset, ComputeWasSolved: no failure, bytes equal to the same equations without the face
      // where the arithmetic does not read it; 1e-15 relative
  ```
  The second A24 test, `CoupledSolvePieces, BlobFormsWithoutBlobBuffer_IsBadInput`, needs equations that hold a
  sealed gas blob, which `BuildPressureEquations` first builds in Task 46; it arrives there.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** Task 44 (a), (d), (e) with `// Step 15:` comments; the
  per-equation material lists are built like the connection lists (ascending `face_piece_index`, then set). In
  `solve_pieces_cpu.cc` (A24): `ComputeVolumeError` adds each material face's flow (`kIncluded`: `-s_e (A_k u_k -
  coverage_volume_rate)`; `kLeftOut`: `-s_e g_k (p_a - p_b)`, an unsolved end at 0); `ComputeOwnWeight` adds `g_k`
  per solved end, not multiplied by `connection_scale`; `BuildPressureLadder` with a non-null scale adds `g_k`
  unscaled; the other pieces accept material faces with unchanged arithmetic.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Put the inertia of blocking materials in the pressure
  equations`.

### Task 46: Sealed gas blobs in the equations and the ladder, on the CPU

**Amended 2026-11-14 (A18):** new task; Task 44 (b), (c), (d), (e) for sealed gas blobs.

**Amended 2026-11-14 (A24):** gains `CoupledSolvePieces, BlobFormsWithoutBlobBuffer_IsBadInput` (moved here from
Task 45 because the equations first hold a sealed gas blob in this task): the forms of the pieces without a blob
buffer refuse equations with `sealed_gas_blob_count > 0`. `BuildPressureLadder` with a non-null scale treats blob
connections as reference ends with own weight `g_c * scale` (no caller today; kept consistent).

**Amended 2026-11-14 (Foundation as built):** ladder tests `LABEL cpu` (was `no_gpu`).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_equations_cpu.cc`, `pressure_equations_math.h`,
  `pressure_ladder_cpu.cc`, `solve_pieces_cpu.cc`, `pressure_equations_test.cc`, `pressure_ladder_test.cc`,
  `solve_pieces_test.cc`

**Interfaces:**
- Consumes: Tasks 23, 24, 44, 45. Air-pocket form for comparison: the enclosed pocket's pressure is read as a
  reference (Task 23); here it becomes an unknown with the squeeze allowance of Task 44.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the ladder tests too):

  ```cpp
  TEST(BuildPressureEquations, AirEndPrecedence_SealedBlobFirst)   // Review Focus 4: an air cell in a sealed blob
      // that also holds_tracked_gas at 300 Pa: its connection goes to the blob, reference sigma * curvature + 7 Pa
      // (gas_pressure_above_background); the other three air ends as Task 23
  TEST(BuildPressureEquations, SealedBlobElimination_EqualsSchurProduct)   // column 2 x 2 x 6 of 0.05 m, four gas
      // cells in one blob: equations(x) == A_cells x - (g g^T x) / D for 20 random x, A_cells and g from the
      // connection list in the test; right side == r_cells + g r / D; 1e-13 relative
  TEST(BuildPressureEquations, SymmetricAndPositiveWithSealedBlob)  // and with Task 45's faces together
  TEST(BuildPressureEquations, LiquidUnderSealedBlob_IsCanBeSqueezed) // closed box, water under gas: with the blob
      // kCanBeSqueezed; without it (tracked gas read late) kHasReferencePressure
  TEST(BuildPressureEquations, TwoPoolsUnderOneBlob_AreOneLiquidBlob)
  TEST(BuildPressureEquations, SealedBlobErrors_AreBadInput)        // index past the list; gas_volume 0; pressure 0;
                                                                    // more than max_sealed_gas_blobs
  TEST(ApplyPressureLadder, SealedBlobConnectionsAreReferenceEnds)  // symmetric and positive; 1e-12 relative
  TEST(CoupledSolvePieces, BlobFormsWithoutBlobBuffer_IsBadInput)   // A24; the Schur test's column (one sealed
      // blob) given to both ComputePressurePush forms, ComputeVolumeError, ComputeOwnWeight and
      // ComputeStartPressure without a blob buffer: kBadInput, no output written. The same column with Task 45's
      // material faces and no blob: success. ComputeFaceInertia, ApplyPressure, ComputeFlowImbalance,
      // ApplyZeroMeanToSealedBlobs, ApplyPressureOffset, ComputeWasSolved on the column with the blob: success
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** Task 44 (b) to (e) with `// Step 13:`, `// Step 15:` and
  `// Step 16:` comments; in `solve_pieces_cpu.cc` the guard of the forms without a blob buffer (A24).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Solve sealed gas blobs with the liquid pressure`.

### Task 47: Solve, correction and check with both couplings, on the CPU

**Amended 2026-11-14 (A30):** a buffer on another processor than the equations is a precondition (`Requires:`, debug assertion), not `kBadInput`; the test is `ShortBuffers_AreBadInput` without the other-processor case.

**Amended 2026-11-14 (A18):** new task; `SolvePressure` and `CheckFlowImbalance` carry both couplings.

**Amended 2026-11-14 (A15):** the sealed gas blob part of the shared arithmetic: per blob `G_j = sum_c g_c d_c`
by the per-group rule, `h_j = G_j / D_j`, `dq -= sum_j G_j h_j`, `q_c -= sum over c's blob terms of g_c h_j`;
`p_j` after the solve from the per-group sum of `g_c p_c`. Material faces are part of `A_local`.

**Amended 2026-11-14 (A24):** the pieces gain the forms that carry the sealed gas blob pressures for
`viscosity::SolveViscosityAndPressure`, which keeps each blob as one more unknown (`plan-decisions.md` "A24:
pressure pieces"), and `ComputeBlockingMaterialVelocities`, which `SolvePressure` uses for its material output.

**Amended 2026-11-14 (A26/A27):** the joint viscous solve keeps material face velocities `w` and free-solid motions
`xi` as velocity unknowns (`plan-decisions.md` "A viscous liquid sticks, and free solids in the viscous solve (A26,
A27)"), so it no longer calls the A24 forms that eliminate material faces. This task adds, for the joint solve only,
`struct CoupledVelocities`, the `ComputeVolumeError` and `CheckFlowImbalance` forms with `CoupledVelocities`,
`ComputeCoupledPush` and the `ComputeOwnWeight` form with `material_face_scale`, and one input field (plan decision
below). The A24 forms stay for `SolvePressure`'s arithmetic; `SolvePressure` is unchanged (material faces eliminated,
free solids as the rank-one term). Six tests added (A26/A27 block of Step 1).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_solve_cpu.cc`, `solve_pieces_cpu.cc`, `coupled_solve_pieces.h`,
  `flow_imbalance.h`, `pressure_solve_test.cc`, `solve_pieces_test.cc`; A26/A27: `pressure_equations.h`,
  `pressure_equations_cpu.cc` (`free_solid_motion_before_pressure`), `pressure_equations_test.cc`

**Interfaces:**
- Consumes: Tasks 29, 34, 44 to 46. Produces the new `SolvedArrays` spans and report field. **Plan decision:**
  `CheckFlowImbalance` recomputes each material velocity and each `p_j` from the `pressure` passed (never from the
  solver's vectors); `SolvePressure` writes the material velocities and `p_j` only after the check passes; a span
  whose size is not the count: `kBadInput`.
- Produces (A24, declared exactly as `plan-decisions.md` "A24: pressure pieces"; one value per sealed gas blob,
  entries `[0, sealed_gas_blob_count)`, on the equations' processor):
  - `ComputePressurePush(equations, pressure, sealed_gas_pressure, given_terms, push)`: a blob connection's far-end
    pressure is `sealed_gas_pressure[j] + ref_c` (`kIncluded`) or `sealed_gas_pressure[j]` (`kLeftOut`).
  - `ComputeVolumeError(equations, pressure, sealed_gas_pressure, velocity, given_terms, volume_error,
    sealed_gas_volume_error)`: cells as the form without a blob buffer; each blob `E_j = inflow_j(velocity) +
    heat_volume_rate + boundary_volume_rate - S_j (q_j - c_j)` (`kIncluded`) or `inflow_j(velocity) - S_j q_j`
    (`kLeftOut`), `inflow_j = sum_c -s_jc (open_area_c u_c + solid_volume_rate_c)` (`kLeftOut`: open areas only),
    m3/s.
  - `ComputeOwnWeight(equations, connection_scale, own_weight, sealed_gas_own_weight)`: cells as the form without a
    blob buffer; each blob `S_j + sum_c g_c * connection_scale_c` (null scale: 1), m3/(Pa s).
  - `ComputeStartPressure(equations, pressure, sealed_gas_pressure)`: cells as the form without; each blob
    `c_j = pressure_above_atmosphere`.
  - `ComputeBlockingMaterialVelocities(equations, pressure, velocity_after_pressure)`: each face's
    `ComputeMaterialVelocityAfterPressure(velocity_before_pressure, dt A_k / m_k, p_a, p_b)`, an unsolved end at 0,
    input order.
  - `CheckFlowImbalance(equations, pressure, sealed_gas_pressure, velocity, settings, buffers)`: the stop rule with
    the blob pressures given instead of recovered by the elimination; each blob row as above (`kIncluded`) against
    `max_volume_error_rate * gas_volume`, reported as `max_sealed_gas_volume_error_rate`.
  - A blob buffer shorter than the count, or a material span whose size is not the count: `kBadInput`, no output
    written. Requires (A30): every blob buffer on the equations' processor (debug assertion, not an error value). Blob sums by the per-group rule of Global Constraints.
- Produces (A26/A27, for the joint solve only; declared in `coupled_solve_pieces.h` exactly as `plan-decisions.md`
  "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)", subsection "Declarations"):

  ```cpp
  struct CoupledVelocities {                              // Given velocity unknowns other than the liquid's.
    const Buffer<double>* blocking_material_velocity = nullptr;   // one per BlockingMaterialFace
    const Buffer<double>* free_solid_motion = nullptr;            // 6 per free solid: V then Omega
  };
  std::expected<void, PressureFailure> ComputeVolumeError(const PressureEquations&, const grid::CellValues<double>&
      pressure, const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity, CoupledVelocities,
      GivenTerms, grid::CellValues<double>& volume_error, Buffer<double>& sealed_gas_volume_error);
  // Material face flow A_k w_k - coverage_volume_rate (kIncluded) or A_k w_k (kLeftOut); a free solid's flow
  // sum_e push_e . (xi - xi_before) (kIncluded) or . xi (kLeftOut), sign as SolvePressure's free-solid term.
  std::expected<void, PressureFailure> ComputeCoupledPush(const PressureEquations&, const grid::CellValues<double>&
      pressure, Buffer<double>& blocking_material_push, Buffer<double>& free_solid_push);
  // dt A_k (p_low - p_high) per BlockingMaterialFace; dt sum_e (force, torque)_per_pressure p_e per free solid.
  std::expected<void, PressureFailure> ComputeOwnWeight(const PressureEquations&, const grid::FaceValues<double>*
      connection_scale, const Buffer<double>* material_face_scale, grid::CellValues<double>& own_weight,
      Buffer<double>& sealed_gas_own_weight);
  std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(const PressureEquations&,
      const grid::CellValues<double>& pressure, const Buffer<double>& sealed_gas_pressure,
      const grid::FaceValues<double>& velocity, CoupledVelocities, const SolveSettings&, PressureBuffers&);
  ```
  Notation: `F` = the equations' free solid count (half the size of `EquationInputs::free_solid_inverse_inertia`, two
  `Mat3` per solid); a free solid's six values are `xi = (V, Omega)`, m/s then rad/s, world axes, entries `[6 s, 6 s
  + 6)` for `solid_index` `s`; `push_e = (force_per_pressure_e, torque_per_pressure_e)` of `FreeSolidPush` entry `e`;
  `s_e = +1` for the low end of a material face; an unsolved end's pressure is 0 (Task 44 (a)).
  - `ComputeVolumeError` (with `CoupledVelocities`): cells as the A24 blob form, except that a material face is
    **not** eliminated: each solved end `e` of face `k` gets `-s_e (A_k w_k - coverage_volume_rate)` (`kIncluded`)
    or `-s_e A_k w_k` (`kLeftOut`), `w_k = (*blocking_material_velocity)[k]`, whatever `pressure` is; and each cell
    gets, for each of its `FreeSolidPush` entries `e` (ascending `solid_index`, after the material faces, before the
    blob terms), `-push_e . (xi_s - xi_before_s)` (`kIncluded`) or `-push_e . xi_s` (`kLeftOut`), m2 x m/s = m3/s
    (the solid moving along `force_per_pressure` leaves the cell, as a material moving out of its low end; the
    sign that makes the free-solid term of Task 23 `+dt push^T M^-1 push` in the operator). Blob rows as the A24 blob
    form (a blob connection is a liquid face, so no material or solid term enters a blob row).
  - **Plan decision (A27, for the owner):** the `kIncluded` free-solid flow needs `xi_before` (the snapshot's
    `solid_volume_rate` already holds the solid's motion before pressure, as `coverage_volume_rate` does for a
    material face), and no input of pressure holds it. `EquationInputs` gains `std::span<const double>
    free_solid_motion_before_pressure` (6 per free solid, `V` then `Omega`, m/s and rad/s; the step fills it from
    solids' `FreeSolidForViscousSolve::velocity_before_pressure` and `turn_rate_before_pressure`); size 0 or `6 F`,
    else `BuildPressureEquations` gives `kBadInput`. `SolvePressure` does not read it. Size 0 with `F > 0` and
    `kIncluded`: the `CoupledVelocities` form of `ComputeVolumeError` and `CheckFlowImbalance` give `kBadInput`.
  - `ComputeCoupledPush`: `blocking_material_push[k] = dt A_k (p_low - p_high)`, N s along +axis, input order;
    `free_solid_push[6 s .. 6 s + 5] = dt sum_e (force_per_pressure_e, torque_per_pressure_e) p_e` over solid `s`'s
    entries, N s then N m s, components in order force x, y, z, torque x, y, z; each solid's sums by the per-group
    rule of Global Constraints (per cell block the solid's entries of that block in ascending `equation_index`, then
    `math::TreeSum` over the solid's block partials in ascending block index). Pressure only: no reference pressure,
    jump or given velocity enters (viscosity adds the push to its momentum rows).
  - `ComputeOwnWeight` (with `material_face_scale`): cells as the A24 blob form, except that each solved end of
    material face `k` adds `g_k * (*material_face_scale)[k]` (null scale: 1, the A24 value; the joint solve passes
    `face_mass_k / velocity_diagonal_k`); free solids are left out of the diagonal (as the ladder leaves them out of
    `SolvePressure`). Blob own weight as the A24 blob form.
  - `CheckFlowImbalance` (with `CoupledVelocities`): the stop rule of step 21 with the blob pressures, material
    velocities and free-solid motions given (never recovered from `pressure` by an elimination); each cell row as
    the `kIncluded` volume error above, each blob row as the A24 given-blob form; same report fields and limits.
  - `kBadInput`, no output written: a `CoupledVelocities` pointer null while its count is `> 0`
    (`blocking_material_face_count` for `blocking_material_velocity`, `F` for `free_solid_motion`); a given or output
    buffer (`blocking_material_velocity`, `free_solid_motion`, `material_face_scale`, `blocking_material_push`,
    `free_solid_push`) shorter than its count (`blocking_material_face_count`, or `6 F`); the blob buffers as the A24
    forms. Requires (A30): every buffer on the equations' processor (debug assertion, not an error value). A null pointer with count 0 is accepted.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(SolvePressure, LiquidCellUnderMaterialFace_ShareMomentum)  // hand-built snapshot (Task 34 style): one liquid
      // cell of 0.1 m, rho 1000, low Z piece kLiquidToAir open 0.01 m2, pressure_distance 0.1, u 1 m/s up; top
      // piece fully blocked, entry A 0.01, m 0.5 kg, u* 0; sides closed; dt 0.01. Liquid face mass M 1 kg.
      // p == 3333.33 Pa; liquid and material velocity both M U / (M + m) == 2/3 m/s; 1e-9 relative
  TEST(SolvePressure, LiquidCellUnderSealedGas_SpringStep)        // one cell of width 1, rho 1, dt 1, both Z pieces
      // area 1, pressure_distance 0.5, u 1 up; below air at 0, above a blob V 1.4, p0 1, ratio 1.4 (S = 1),
      // carried 0: p == 0.25 Pa, sealed_gas_pressure == 0.5 Pa, both velocities 0.5 m/s; 1e-9 relative
  TEST(CheckFlowImbalance, MaterialFlowFromPressure)              // first fixture with p = 3000: worst cell error
                                                                  // equals the hand value; 1e-12 relative
  TEST(CheckFlowImbalance, SealedBlobRowChecked)                  // second fixture with p = 0.3:
      // max_sealed_gas_volume_error_rate equals the hand value; kNotConverged when above the limit
  TEST(SolvePressure, CouplingAbsent_SameBytesAsBefore)           // Task 34's cases with empty spans: identical bytes
  TEST(SolvePressure, CouplingSpansWrongSize_IsBadInput)
  // A24, the blob forms on the second fixture. Its numbers: g = flow_per_pressure = open_area * dt / (rho *
  // pressure_distance) = 2 m3/(Pa s) on both Z pieces, velocity_per_pressure 2 (m/s)/Pa, face inertia 0.5 kg,
  // S = 1.4 / (1.4 * 1 * 1) = 1, c = 0, heat and boundary volume rates 0, no surface tension (ref_c = 0).
  TEST(CoupledSolvePieces, SealedGasRowFromGivenPressure)  // ComputeVolumeError, kIncluded, p = 0.25, velocity 0.5
      // on both Z pieces (SolvePressure's answer), q = 0.5: blob error 0 and cell error 0 within 1e-12 m3/s;
      // q = 0.6: blob error 0.5 - 1 * (0.6 - 0) = -0.1 m3/s; kLeftOut with q = 0.6: 0.5 - 0.6 = -0.1 m3/s; 1e-15
      // relative
  TEST(ComputePressurePush, SealedGasFarEndFromGivenPressure)     // p = 0.25, q = 0.5, both GivenTerms: top piece
      // push 1 * 1 * (0.25 - 0.5) = -0.25 N s, bottom piece 1 * 1 * (0 - 0.25) = -0.25 N s; ApplyPressure from
      // velocity 1 gives 1 - 0.25 / 0.5 = 0.5 m/s on both; 1e-15 relative
  TEST(ComputeOwnWeight, SealedGasOwnWeight)                      // scale 0.5 on every piece: blob own weight
      // 1 + 2 * 0.5 = 2 m3/(Pa s); null scale: 1 + 2 = 3 (= D_j of Task 44 (b)); 1e-15 relative
  TEST(ComputeStartPressure, SealedGasStartIsCarried)             // blob carried pressure 0.3 Pa: sealed_gas_pressure
      // == 0.3 exactly; cells as the form without a blob buffer
  TEST(ComputeBlockingMaterialVelocities, MatchesFormula)         // first fixture, p = 3000: 0 + 2e-4 * (3000 - 0)
      // == 0.6 m/s; 1e-15 relative; a span of the wrong size: kBadInput, nothing written
  TEST(CheckFlowImbalance, GivenBlobPressureChecked)              // second fixture, p = 0.25, velocity 0.5:
      // q = 0.5 + 1.4e-5: E_j = -1.4e-5 m3/s, max_sealed_gas_volume_error_rate == 1.4e-5 / 1.4 = 1e-5 1/s (under
      // 2e-5), 1e-9 relative; q = 0.6 (|E_j| / V = 0.1 / 1.4): kNotConverged; a blob buffer of size 0: kBadInput
  // A26/A27, the CoupledVelocities forms. Fixtures: Task 45's two-cell material fixture (two liquid cells of 0.1 m,
  // shared piece fully blocked, A 0.01, m 0.5, dt 0.01, g = 2e-6; low cell end a, high cell end b; its
  // coverage_volume_rate as written there), Task 23's scene with its free solid (3 hand-built FreeSolidPush
  // entries), and Task 47's first fixture (one liquid cell under a material face, A 0.01, m 0.5, dt 0.01). Every
  // "difference" is against the same call on the same equations with the term absent (w = 0, or xi = xi_before).
  TEST(ComputeVolumeError, GivenMaterialVelocity)    // two-cell fixture, velocity 0 on every piece, sealed blob
      // buffers of size 0, w = 0.3 m/s: kLeftOut, p = (0, 0): the material term is -s_e A w, so the difference is
      // -0.003 m3/s in the low cell and +0.003 m3/s in the high cell (A w = 0.01 * 0.3 = 0.003); the same with
      // p = (1, 0) (w is given: no -s_e g (p_a - p_b) term, unlike the A24 form's -2e-6 / +2e-6). kIncluded,
      // p = (0, 0): each cell's error equals the A24 kIncluded form's at the same p minus s_e (A w - A u_k), with
      // u_k = ComputeMaterialVelocityAfterPressure(u*, dt A / m, 0, 0) = u*, so the material flow is
      // A w - coverage_volume_rate, computed in the test; 1e-15 relative
  TEST(ComputeVolumeError, GivenFreeSolidMotion)     // Task 23's scene, velocity 0, p 0, free_solid_motion_before_
      // pressure xi_before and given xi (fixed seed, components in [-1, 1]): kIncluded, difference from xi =
      // xi_before in each pushed cell == -sum over its entries e of push_e . (xi - xi_before), summed in the test
      // over the entries in ascending solid_index; kLeftOut: -sum_e push_e . xi; cells without an entry: difference
      // exactly 0; 1e-14 relative. kIncluded with free_solid_motion_before_pressure empty: kBadInput
  TEST(ComputeCoupledPush, MaterialAndFreeSolid)     // two-cell fixture, p = (1, 0): blocking_material_push ==
      // dt A (p_low - p_high) = 0.01 * 0.01 * (1 - 0) = 1e-4 N s; p = (0, 1): -1e-4 N s. Task 47's first fixture
      // (high end unsolved), p 3000: 0.01 * 0.01 * 3000 = 0.3 N s. Task 23's free solid with a given p (fixed
      // seed): free_solid_push[0..5] == dt sum_e (force_per_pressure_e, torque_per_pressure_e) p_{equation_index_e}
      // over its 3 entries, summed in the test in ascending equation_index; 1e-15 relative (material), 1e-14
      // relative (solid)
  TEST(ComputeOwnWeight, MaterialFaceScale)          // two-cell fixture: with material_face_scale 0.5 for the face,
      // each cell's own weight gains exactly g * 0.5 = 1e-6 m3/(Pa s) over the same equations without the face;
      // null material_face_scale: 2e-6 (the A24 value); connection_scale 0.5 on every piece changes neither (g_k is
      // scaled only by material_face_scale). Task 23's scene: own weight equal with and without the free solid
      // (left out of the diagonal), bytes; 1e-15 relative
  TEST(CheckFlowImbalance, CoupledVelocitiesChecked) // Task 47's first fixture, p 3000, liquid velocity 2/3 m/s
      // on the low Z piece, w = 2/3 m/s: worst cell error equals the A24 form's (which recovers w = 0.6 from p)
      // minus A (2/3 - 0.6) = 6.6667e-4 m3/s; at p = 10000/3 (SolvePressure's answer) with w = 2/3 both forms
      // give the same report within 1e-12 relative; w = 2/3 + 1e-3: worst cell error -1e-5 m3/s, rate 1e-5 over
      // the cell's filled_volume, kNotConverged. Task 23's scene after SolvePressure: its pressure and velocity
      // with xi = xi_before + dt M^-1 sum_e push_e p_e (computed in the test): the report equals the A24 form's
      // within 1e-12 relative
  TEST(CoupledVelocityPieces, ShortBuffers_AreBadInput)   // amended (A30); each CoupledVelocities form and
      // ComputeCoupledPush: blocking_material_velocity null on the two-cell fixture (count 1); free_solid_motion null
      // on Task 23's scene (F 1); each given or output buffer one entry short (0 of 1, 5 of 6); nothing written (outputs keep a written pattern). Null
      // pointers on equations with count 0: success. BuildPressureEquations on Task 23's scene with
      // free_solid_motion_before_pressure of size 5: kBadInput; size 0 and 6: success
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step 19:` to `// Step 21:` comments; the blob forms
  share each per-cell and per-face function with the forms without a blob buffer; `SolvePressure` writes
  `blocking_material_velocity_after_pressure` through `ComputeBlockingMaterialVelocities`. A26/A27: the
  `CoupledVelocities` forms share the per-cell function of the A24 blob forms with the material term chosen by a
  `FLUIDITY_CPU_AND_GPU` helper (given `w_k`, or the elimination of Task 44 (a)), so both forms keep one cell order;
  `BuildPressureEquations` keeps `free_solid_motion_before_pressure` as a view for the `kIncluded` free-solid term.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Correct material faces and sealed gas in the pressure solve`.

### Task 47a: Guided free solids in the inviscid solve, CPU and GPU

**Amended 2026-11-15 (A59):** new task; the free-solid term becomes `dt (K J)(Jᵀ M J)⁻¹(K J)ᵀ` with `xi_0` in the
right-hand side, and the solid update `q = (Jᵀ M J)⁻¹ Jᵀ (M xi_before + impulse)`, `xi = J q + xi_0`, consuming
`solids::AllowedMotions` (`plan-decisions.md` "Owner answers A40-A64 applied", item 5).

**Kind:** numerics

Run after the solids task that creates `native/fluidity/solids/guide_math.h` (Before you start).

**Files:**
- Create: `native/fluidity/pressure/free_solid_guide_math.h` (shared), `free_solid_guide_math_test.cc` (`LABEL cpu`),
  `guided_free_solid_test.cc` (`LABEL cpu`), `guided_free_solid_gpu_test.cc` (`LABEL gpu`), their CMake lines
- Modify: `pressure_equations.h` (`FreeSolidInertia`, `EquationInputs::free_solids` replacing
  `free_solid_inverse_inertia`), `pressure_equations_cpu.cc`, `pressure_equations_gpu.cu`, `pressure_equations_math.h`,
  `pressure_solve_cpu.cc`, `pressure_solve_gpu.cu`, `solve_pieces_steps.h` (the per-solid `k_s`), the
  `CoupledVelocities` forms' free-solid count, `pressure_memory_estimate.cc` (row `"pressure: free solids: guided
  inverse mass"`), `pressure_test_scenes.h` / `.cc` (`BuildBoxFreeSolidPushes`), every test that built
  `free_solid_inverse_inertia` (Tasks 23, 34, 47, 48, 59), `docs/method-paper/sections/pressure.typ` (one paragraph
  and the term in `<eq:pressure>`), `docs/glossary.md` (`allowed motions`, `guided free solid`)

**Interfaces:**
- Consumes: `solids::AllowedMotions`, `solids::ComputeAllowedMotions(const FreeMotionLimits&, Vec3 origin_at_start)`
  (solids' shared `guide_math.h`, header-only: pressure includes it and links nothing of solids); Tasks 23, 34, 47.
- Produces exactly as design 3: `FreeSolidInertia`, `EquationInputs::free_solids`, `GuidedInverseMass`,
  `ComputeGuidedInverseMass`, `ComputeGuidedSolidMotion`. Solids' `ApplyPressureToSolids` calls
  `ComputeGuidedSolidMotion` (handed over to the solids plan). Test-only (`fluidity_pressure_test_scenes`):
  `std::vector<FreeSolidPush> BuildBoxFreeSolidPushes(const grid::StaggeredGrid& grid, const GeometrySnapshot&
  geometry, std::span<const std::uint32_t> equation_of_cell, std::uint32_t solid_index, Vec3 centre, double
  half_width, Vec3 origin_at_start);` (the brute-force rule of `solids::ToFreeSolidPushes` for an axis-aligned box:
  per liquid cell and blocked face piece, `force_per_pressure` = blocked area times the unit normal from the cell
  into the box, `torque_per_pressure` = (piece centre - origin) x that; sorted by equation index).
- **Plan decisions:**
  - `ComputeGuidedInverseMass`: `M = diag(mass I3, inertia_at_start)`; `B = Jᵀ M J` (`B[a][b] = sum_i sum_k J[i][a]
    M[i][k] J[k][b]`, `i`, `k` ascending); Cholesky `B = L Lᵀ` column by column, pivot not `> 0`: `kBadInput`;
    `G = Y Yᵀ` with `Y = J L⁻ᵀ` (rows by forward substitution), each entry summed over columns ascending, so `G` is
    symmetric to the last bit (`G[i][j]` and `G[j][i]` are the same sum). `column_count == 0`: `G = 0`.
  - `ComputeGuidedSolidMotion`: `r = Jᵀ (M xi_before + impulse)`, `q = B⁻¹ r` by the same factor, `xi_after = J q +
    prescribed_motion`, entries ascending, no contraction.
  - `BuildPressureEquations` computes `G_s` once per solid (stored in the module's buffers, 288 B per solid,
    `max_free_solids` of them) and `kBadInput` for a solid that fails `ComputeGuidedInverseMass`, a
    `FreeSolidPush::solid_index >= free_solids.size()`, or `free_solid_motion_before_pressure` not of size 0 or `6 x
    free_solids.size()`. Cells pushing one solid are joined into one blob only when its `column_count > 0`.
  - Iteration (Task 34's shared arithmetic): `k_s = dt * G_s P_s` (6 x 6 times 6, rows ascending); everything else
    unchanged. The ladder still leaves free-solid terms out.
  - The prescribed motion `xi_0` is never read by the solve: it reaches the right-hand side through the snapshot's
    `solid_volume_rate` and, in the joint solve, `free_solid_motion_before_pressure`, both built by solids from the
    guided motion before pressure `J q + xi_0`. Today `xi_0 = 0` (no moving rail is exposed).

- [ ] **Step 1: Failing tests:**

  ```cpp
  // free_solid_guide_math_test.cc, LABEL cpu. Box 0.1 m, mass 0.5 kg, inertia R diag(8.3333e-4, 1.0e-3, 1.2e-3) Rᵀ
  // kg m2, R a 30° turn about z (not diagonal); origin o = (0, 0, 0).
  TEST(ComputeGuidedInverseMass, SixAxisColumns_IsInverseMass)  // ComputeAllowedMotions(all six allowed, o): G ==
      // diag(2 I3, inverse of the tensor by the adjugate in the test); 1e-14 relative of each block's largest entry
  TEST(ComputeGuidedInverseMass, HingeColumn_IsOneOverMomentAboutPivot)  // hinge a = (0, 1, 0) through p = (0, 0,
      // 0.5): a.G_OmegaOmega.a == 1 / (I_yy + 0.5 * 0.25); every G[i][j] == J_i J_j / (Jᵀ M J), 1e-14 relative;
      // G[i][j] and G[j][i] bit-equal
  TEST(ComputeGuidedInverseMass, NoAllowedMotion_IsZero)       // column_count 0: all 36 entries exactly 0
  TEST(ComputeGuidedInverseMass, DependentColumns_IsBadInput)  // hand-built 2 columns both e_x; and mass 0
  TEST(ComputeGuidedSolidMotion, RailUnderGravity_MovesAlongRail)  // rail d = (cos 30°, 0, sin 30°), xi_before 0,
      // impulse (0, 0, -0.5 * 9.81 * 1e-3, 0, 0, 0): V == -4.905e-3 d m/s within 1e-15 relative; Omega exactly 0
  TEST(ComputeGuidedSolidMotion, SixAxisColumns_IsVelocityPlusImpulseOverMass)  // random xi_before, impulse
      // (fixed seed): xi_before + M⁻¹ impulse (adjugate in the test), 1e-14 relative
  TEST(ComputeGuidedSolidMotion, PrescribedMotion_IsAdded)     // hand-built: one column e_y, prescribed (0.1, 0, 0, 0,
      // 0, 0), impulse (0, 0.5, 0, 0, 0, 0) N s, xi_before 0: V == (0.1, 1.0, 0) m/s exactly; Omega exactly 0
  // guided_free_solid_test.cc, LABEL cpu
  TEST(BuildPressureEquations, GuidedSolid_SymmetricAndPositive)  // Task 23's scene, its free solid hinged (a =
      // (0, 1, 0) through a point 0.05 m above origin): 20 random pairs, x.A y == y.A x and x.A x >= 0, 1e-12 rel
  TEST(SolvePressure, GuidedFreeSolid_DqEqualsDirectProduct)  // design 7: same scene with the hinge, then a rail d =
      // (cos 30°, 0, sin 30°); 10 random d: dq == d.(A d) with the free block dt (K J)(Jᵀ M J)⁻¹(K J)ᵀ formed in the
      // test from the push list and J; 1e-12 relative
  TEST(SolvePressure, TiltedRailInLiquid_MovesOnlyAlongRail)  // design 7: BuildBoxGrid 16^3 of 0.025 m, open top,
      // water (1000) to z 0.3; box 0.1 m of 500 kg/m3 centred (0.2, 0.2, 0.15), its 4^3 cells kSolid, pushes by
      // BuildBoxFreeSolidPushes; rail d as above; dt 1e-3, gravity; the box's solid_normal_velocity from V_pred =
      // ComputeGuidedSolidMotion(solid, 0, (0, 0, -m g dt, 0, 0, 0)); SolvePressure, then xi = ComputeGuidedSolidMotion(
      // solid, 0, push + (0, 0, -m g dt, 0, 0, 0)), push from ComputeCoupledPush: |V - (V.d) d| <= 1e-12 |V|,
      // Omega exactly 0, V.d > 0, the check passes
  TEST(SolvePressure, HingedBodyInLiquid_KeepsThePivot)       // design 7: same tank; box 0.1 m of 2000 kg/m3 turned
      // 10° about y, hinge a = (0, 1, 0) through the point 0.1 m above its centre, xi_before 0, gravity: after the
      // solve and the update |V + Omega x (p - o)| <= 1e-12 (|V| + 0.1 |Omega|); Omega_x, Omega_z exactly 0
  TEST(SolvePressure, AxisAlignedGuides_Unchanged)            // design 7: Task 23's and Task 47's free-solid scenes
      // with six axis columns: pressure and velocity after equal a run whose k_s uses diag(1/m I3, I⁻¹) built in the
      // test, 1e-12 relative
  TEST(BuildPressureEquations, LockedFreeSolid_DoesNotJoinBlobs)  // two sealed chambers both pushing one free solid:
      // column_count 0 gives two kSealed blobs; six columns give one
  TEST(BuildPressureEquations, FreeSolidsMismatch_IsBadInput) // a push with solid_index 1 and one free solid;
      // mass 0; free_solid_motion_before_pressure of size 5
  // guided_free_solid_gpu_test.cc, LABEL gpu
  TEST(SolvePressure, GuidedFreeSolidOnGpu_EqualsCpu)         // the rail and hinge scenes: CPU-against-GPU bounds of
      // Global Constraints; G_s bytes equal (shared function); outputs through WriteResultDigest
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 15, 16 and 19 as amended (A59), `// Step 15:`
  comments at the free-solid term; replace `free_solid_inverse_inertia` everywhere (`grep -rn
  free_solid_inverse_inertia native` finds nothing after); method paper: the guided term `dt (K J)(Jᵀ M J)⁻¹(K
  J)ᵀ`, the update, and the citation of Batty 2007 and Aanjaneya 2018 (`references.typ`, as solids' A59 task added
  them; add them here if absent).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Solve guided free solids in the pressure solve`. Body: new
  names `FreeSolidInertia`, `EquationInputs::free_solids` (replaces `free_solid_inverse_inertia`, A59),
  `GuidedInverseMass`, `ComputeGuidedInverseMass`, `ComputeGuidedSolidMotion`; `CUDA: compiled, not run`.

### Task 48: The coupling on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A18):** new task; GPU forms of Tasks 45 to 47.

**Amended 2026-11-14 (reconcile):** each blob's sum by the per-group block rule (block partials, then a tree sum),
not one thread per blob.

**Amended 2026-11-14 (A24):** the A24 forms of Tasks 45 to 47 (material faces in the pieces, the blob forms,
`ComputeBlockingMaterialVelocities`, the `CheckFlowImbalance` form with given blob pressures) get their GPU forms:
blob buffers whole on the GPU, blob sums by the per-group rule. Test `CoupledSolvePieces, OnGpu_EqualsCpu`.

**Amended 2026-11-14 (A26/A27):** the `CoupledVelocities` forms of Task 47 (`ComputeVolumeError`,
`CheckFlowImbalance`), `ComputeCoupledPush` and the `ComputeOwnWeight` form with `material_face_scale` get their GPU
forms, calling the per-cell helpers of Task 47: the material-face and free-solid buffers whole on the GPU (indexed by
`BlockingMaterialFace` entry and by `6 * solid_index`); each free solid's six push sums by the per-group rule (one
thread per cell block writes the solid's partial of that block, entries in ascending `equation_index`; then one
`math::TreeSum` per component over the solid's block partials in ascending block index), no atomics. Test
`CoupledVelocityPieces, OnGpu_EqualsCpu`.

**Amended 2026-11-14 (Foundation as built):** the groupings through the buffers' `order_buffers`, a `cpu` steps test of
the coupled per-cell, per-face and per-cell-block functions, digests.

**Kind:** device

**Files:**
- Modify: `native/fluidity/pressure/pressure_equations_gpu.cu`, `pressure_ladder_gpu.cu`, `pressure_solve_gpu.cu`,
  `solve_pieces_gpu.cu`, their `_gpu.h` headers, `pressure_equations_gpu_test.cc`, `pressure_solve_gpu_test.cc`,
  `solve_pieces_gpu_test.cc`, `pressure_equations_steps_test.cc`, `solve_pieces_steps_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL gpu`): `BuildPressureEquations, CoupledOnGpu_EqualsCpu` (Tasks 45 and 46
  scenes; coefficients within 1e-12 * scale), `ApplyPressureLadder, CoupledOnGpu_EqualsCpu`,
  `SolvePressure, CoupledOnGpu_EqualsCpu` (Task 47 fixtures; material velocity with the velocity bound,
  `sealed_gas_pressure` with the pressure bound), `SolvePressure, CoupledOnGpuTwice_GivesIdenticalBytes`,
  `CoupledSolvePieces, OnGpu_EqualsCpu` (A24: every piece of Tasks 45 to 47 on Task 46's column with Task 45's
  faces, blob forms included; cell, face and blob values within 1e-12 * scale; the `CheckFlowImbalance` report
  equal within the same bound), `CoupledVelocityPieces, OnGpu_EqualsCpu` (A26/A27: every `CoupledVelocities` form,
  `ComputeCoupledPush` and the `material_face_scale` form on Task 46's column with Task 45's faces and on Task 23's
  scene with its free solid, the given `w`, `xi`, `xi_before` and scale of Task 47's tests; cell, blob, material-push
  and free-solid-push values within `1e-12 * scale`; the `CheckFlowImbalance` report equal within the same bound).
  (`LABEL no_gpu`): `SolvePressure, CoupledOnGpuWithoutAGpu_IsRunOnCallFailed`. (`LABEL cpu`, the main evidence
  until the hardware run): `CoupledPiecesSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (the material-face, blob
  and free-solid per-cell, per-face and per-cell-block functions of Tasks 45 to 47, over every thread block and
  thread, launch order and reverse, on Task 46's column with Task 45's faces and on Task 23's scene: bytes equal to
  the CPU forms). Every `gpu` test of this task passes its main outputs to `WriteResultDigest`.
- [ ] **Step 2: Implement.** Material faces and blobs copied to the device with the equations; per-equation
  material lists and each blob's connection list with `BuildGroupsByDestination` (the buffers'
  `order_buffers`); each blob's sum by the per-group
  rule of Global Constraints (one thread per cell block writes the blob's partial of that block, connections in
  ascending `face_piece_index`; then one `math::TreeSum` per blob over its block partials in ascending block
  index), one thread per cell gathers (no atomics).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Solve blocking materials and sealed gas on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 49: Scene tests, liquid on and beside a light blocking material

**Amended 2026-11-14 (A18):** new task; analytic scenes of a liquid resting on and floating a light material.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/pressure/pressure_scene_test.cc`, `pressure_test_scenes.h`, `pressure_test_scenes.cc`
  (`BuildBlockingMaterialFacesForTest(grid, snapshot, face_mass_of_piece, velocity_of_piece)`: one entry per
  fully blocked piece beside a `kLiquid` cell, as solids will build them)

The material's own elasticity is a test stand-in applied as `ApplyForces` would; geometry is fixed in the first
two tests (displacements stay below 0.2 w) and rebuilt each step in the last two. Each lagged reference is
test-only (about ten lines): the material is a driven solid at its velocity (no `BlockingMaterialFace`), and the
next step's material velocity takes `-dt A p / m` from this step's pressure. Banks et al.: that scheme is unstable
at every step when the liquid's added mass `M` exceeds `m`, the face velocity difference growing by `-M / m` per
step.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  // Snow slab: BuildBoxGrid w 0.05, 4 x 4 x 8 (0.2 x 0.2 x 0.4 m); walls and floor (DistanceToOpenTopBox); slab
  // solid distance min(walls, z - 0.1): 16 fully blocked Z pieces at z 0.1, each an entry (A 0.0025, m_s = 300 *
  // w^3 = 0.0375 kg, coverage velocity u_s); water 1000 to H 0.3 (h_w 0.2, share 0.5); g 9.81; dt 0.01. Snow
  // stand-in per face: u_s_before = u_s_after_prev + dt (-k x - m_s g) / m_s, k 475 N/m, x += dt u_s_after;
  // liquid Z pieces before = after_prev - g dt. M = 1000 A (h_w - w/2) = 0.4375 kg; p_1 = 1000 g (h_w - w/2) =
  // 1716.75 Pa at the lowest centre; x* = -(A p_1 + m_s g) / k = -9.8098e-3 m; omega = sqrt(k / (m_s + M)) = 31.6.
  TEST(PressureScene, WaterOnSnowSlab_InterfaceHoldsAndCarriesTheWeight)  // start at x*, hydrostatic, at rest; 500
      // steps: |x_n - x*| <= 1.3e-7 m each step (twice the velocity bound over omega), no trend (means of steps
      // 1-100 and 401-500 within 1.3e-7 m); interface pressure 1.5 p(z 0.125) - 0.5 p(z 0.175) == 1000 g h_w =
      // 1962 Pa within twice the pressure bound; push per area m_s (u_after - u_before) / (dt A) == -p_1, 1e-9 rel
  TEST(PressureScene, LightSnowSlab_StableWhereLagGrows)   // m_s / M = 0.0857; start at x* + 0.1 w; 500 steps:
      // max |x - x*| over steps 401-500 <= 1.02 * 0.1 w. Lagged reference at dt / 10: slab-water velocity
      // difference times -M / m_s = -11.67 per step (steps 2 to 5, 1 %); at dt: grows at least 5 times per step
  // Floating slab: BuildBoxGrid w 0.025, 16 x 2 x 12 (0.4 x 0.05 x 0.3 m), walls and floor; a slab of density
  // 200 (0.2 of the water), t 0.1 m, over x in [0, 0.2], bottom z_b, side at x 0.2 a solid at rest; free surface
  // h_r over x in [0.2, 0.4]; liquid distance z - (z_b + w) for x < 0.2, z - h_r otherwise. Each step: rebuild the
  // snapshot, one entry per Z piece at the top of a cut cell (16: A w^2, m 200 w^2 t = 0.0125 kg, the slab's
  // velocity), solve, z_b += dt * mean slab face velocity, h_r += dt * mean surface velocity. Equilibrium: the cut
  // cell's centre z_c lies 0.2 t below h_r; with z_b + h_r = 0.295: z_b* = z_c = 0.1375, h_r* = 0.1575.
  TEST(PressureScene, LightSlab_FloatsAtDisplacedDepth)    // start at (z_b*, h_r*), at rest, dt 0.01, 300 steps:
      // |z_b - z_b*| <= 1e-6 w each step; h_r - z_c == 0.2 t within 1e-6 w; z_b + h_r kept within the solve limit
  TEST(PressureScene, LightSlab_OscillationDoesNotGrowWhereLagGrows)  // start z_b* - 0.2 w, h_r* + 0.2 w, dt 0.01,
      // 500 steps (same cut cell throughout): max |z_b - z_b*| over the last 100 <= 1.02 * 0.2 w; mean z_b over
      // the last 300 within 0.03 w of z_b*. Lagged reference at dt / 10 = 0.001 s: the slab-water velocity
      // difference grows at least 5 times per step, steps 2 to 5 (M / m >= 1000 z_c / (200 t) = 6.9)
  ```
- [ ] **Step 2: Run.** If one fails, fix the owning commit (Tasks 44 to 47), not the test.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Test liquid resting on and floating a light material`.

### Task 50: Scene tests, a sealed gas pocket solved with the liquid

**Amended 2026-11-14 (A18):** new task; analytic gas-spring scenes, and the heated pocket in the same solve.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/pressure/pressure_scene_test.cc`

The test loop does between steps what gas will do: `V -= dt * (blob inflow after)`, `p0 = atmosphere +
sealed_gas_pressure`, carried value `= sealed_gas_pressure`; liquid `velocity_before = velocity_after_prev`. The
lagged reference (test-only, about ten lines) passes no `SealedGasBlob` and gives the carried value as
`gas_pressure_above_atmosphere`, updating it with `p0 (V_n / V_n+1)^heat_ratio`. The coupled scheme is backward
Euler on the gas spring (per-step factor `1 / sqrt(1 + (omega dt)^2)`, phase `atan(omega dt)`); the lagged one is
symplectic Euler, unstable for `omega dt > 2` with factor `((omega dt)^2 - 2 + sqrt(((omega dt)^2 - 2)^2 - 4)) / 2`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  // Column: BuildBoxGrid w 0.025, 2 x 2 x 16 (A 0.0025 m2, 0.4 m); side walls and a closed top (solid distance
  // min(x, 0.05 - x, y, 0.05 - y, 0.4 - z)), bottom grid edge open; no gravity; water 1000 in z [0.05, 0.25]
  // (both surfaces on faces: m = 1000 A h_l = 0.5 kg exactly); the 24 cells above one SealedGasBlob (V 3.75e-4,
  // p0 101325, 1.4, no heat or boundary rate), holds_tracked_gas 1, gas_pressure_above_background 0. k = 1.4 p0
  // A^2 / V = 2364.3 N/m, omega = 68.765 1/s. Start: water Z pieces at u0, carried 0.
  TEST(PressureScene, SealedGasAboveColumn_PeriodIsSpringPeriod)  // u0 1e-4 m/s, dt 0.1 / omega = 1.4542e-3 s,
      // 50 periods (3142 steps): period from upward zero crossings (mean of periods 1 to 5) == 2 pi dt /
      // atan(omega dt) within 0.05 % and within 0.5 % of 2 pi / omega = 0.091371 s; each period's largest |u| <=
      // the previous one's; per-step factor 1 / sqrt(1.01) within 1e-4 relative
  TEST(PressureScene, SealedGasAboveColumn_StableWhereLagGrows)   // u0 1e-9 m/s, dt 4 / omega = 0.058169 s (about
      // 54 times the lagged scheme's air-pocket limit 1.084e-3 s), 79 steps (50 periods): |u| never grows, factor
      // 1 / sqrt(17) = 0.24254 within 1e-4; lagged reference: |u| times 13.928 per step, steps 2 to 4, within 1 %
  TEST(PressureScene, SealedGasHeatedAboveStillWater_RisesInTheSameSolve)  // closed box 0.2 x 0.2 x 0.4, w 0.05,
      // water to 0.2 at rest, g 9.81, the gas above one SealedGasBlob (V 0.008) with heat_volume_rate (0.4 / 1.4)
      // * 35000 * 1.25e-4 / 101325 (35 kW/m3 in one cell); one solve at dt 0.03 s (ten times the lagged limit
      // 3.0e-3 s): sealed_gas_pressure == 0.4 * 546.875 * 0.03 = 6.5625 Pa within 1e-9 relative (the S of the
      // blob uses the same p0); liquid pressure == 6.5625 + 1000 g (0.2 - z) at centres within the pressure bound;
      // water velocity 0 within the velocity bound
  ```
- [ ] **Step 2: Run.** If one fails, fix the owning commit, not the test.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Test a sealed gas pocket solved with the liquid`.

### Task 51: Python reference

**Amended 2026-11-14 (A18):** renumbered from Task 43; a fourth case holds a material face and a sealed gas blob.

**Kind:** numerics

**Files:**
- Create: `reference/pressure/dense_cell_equations.py`, `reference/pressure/dense_cell_equations_test.py`,
  `reference/pressure/cases/size_step.json`, `cases/jump_and_squeeze.json`, `cases/driven_sealed.json`,
  `cases/coupled.json`, `native/fluidity/pressure/dense_reference_test.cc`,
  `native/fluidity/pressure/number_arrays_for_tests.h`

**Interfaces:**
- **Plan decision**, case format (JSON of number arrays only, so the C++ test reads it with a 40-line scanner):
  `"frame": [small_cell_width, size_level_count]`, `"cells": [[level, x, y, z, filled_volume, bulk_stiffness,
  carried_pressure, added_volume_rate, density], ...]`, `"pieces": [[axis, level, x, y, z, open_area,
  pressure_distance, liquid_share, ends, solid_volume_rate, velocity_before_pressure, pressure_jump,
  reference_pressure], ...]`, `"dt": [s]`, and, written by the script, `"pressure": [...]`, `"velocity": [...]`.
  A18 additions, optional keys: `"material_faces": [[axis, level, x, y, z, blocked_area, face_mass,
  velocity_before_pressure, coverage_volume_rate], ...]`, `"gas_blobs": [[gas_volume, background_pressure,
  heat_ratio, pressure_above_atmosphere, heat_volume_rate, boundary_volume_rate], ...]`, `"gas_cells": [[level,
  x, y, z, blob, pressure_above_background], ...]`, and written by the script `"material_velocity": [...]`,
  `"gas_pressure": [...]`.

- [ ] **Step 1: Write** `dense_cell_equations.py` (about 190 lines, SciPy sparse matrix and direct solve,
  sealed blobs by a least-norm solve then the mean rule of step 16), deliberately not structured like the C++.
  Three cases, each with a size step, partly filled cells, a jump, squeeze allowance, a driven solid and a sealed
  blob (one dominant feature per file name). A fourth, `coupled.json`: a two-level case with two material faces
  between equation cells, three with one solved end, and one sealed gas blob touching four cells, which the
  script solves as a true extra unknown (one more row and column, no elimination), so it checks Task 46's
  elimination independently. `uv run python reference/pressure/dense_cell_equations.py
  --write-expected reference/pressure/cases/*.json`.
- [ ] **Step 2: Tests:** Python `test_cases_reproduce_stored_answers`; C++ (`LABEL cpu`)
  `DenseReference, FourCases_MatchStoredAnswers`: builds each case on `BuildTwoLevelGrid` with a hand-filled
  snapshot, solves, compares pressure, velocity, material velocity and gas pressure at `1e-8` relative.
- [ ] **Step 3: Run** `uv run pytest reference/pressure -q` and the gate. **Commit.** Subject:
  `Add a dense reference solve of the pressure equations`.

### Task 52: Lean proof

**Amended 2026-11-14 (A18):** renumbered from Task 44; the connection list of the proof includes material faces.

**Kind:** numerics

**Files:**
- Create: `proofs/Fluidity/PressureEquations.lean`
- Modify: `proofs/Fluidity.lean`

- [ ] **Step 1: Write** from `git show main:proofs/Fluidity/Pressure.lean` (`pressure_psd`) and
  `PressureTopology.lean` (`pressure_kernel_eq_component_constants`, `pressure_solvable_iff_component_compatible`),
  in the new vocabulary: `pressure_equations_positive_semidefinite`,
  `pressure_equations_null_space_is_blob_constants` (blobs with no reference pressure and no squeeze allowance),
  `pressure_equations_solvable_iff_blob_sums_vanish`. Leave out `pressure_gauge_positive_definite` and
  `PressureCorrection.lean` (design 7). The proof's connections are weighted pairs and weighted single ends, so
  blocking-material faces (Task 44 (a)) are in its model; a material face with one solved end counts as a
  reference end in the null-space theorem. The sealed gas elimination is not proved here (tested by Tasks 46 and
  51); the hand-over records it.
- [ ] **Step 2: Run** `lake build Fluidity.PressureEquations` in `proofs/`. Expected: no `sorry`, exit 0. Gate.
- [ ] **Step 3: Commit.** Subject: `Prove the pressure equations have only blob constants as free values`.

### Task 53: Comparison with the proof of concept

**Amended 2026-11-14 (A31):** the throwaway program is built in a temporary directory.

**Amended 2026-11-14 (A18):** renumbered from Task 45; adds `main`'s one-face snow pressure trial.

**Kind:** integration

**Files:**
- Create: `native/fluidity/pressure/testdata/proof_of_concept.json`, `native/fluidity/pressure/proof_of_concept_test.cc`

- [ ] **Step 1: Produce the old numbers** (plan-decisions: export `main` with `git archive` to a temporary directory and
  build its CPU preset there). In a temporary directory write a program linked to `main`'s library that runs, on the
  Task 51 cases rotated back to Y-up and on the hydrostatic column: `PrepareExplicitMacPressureSystem3D`
  (coefficients, right sides), `ProjectMacPressure3D` (pressure, velocity), `ComputeGeometry` in curvature mode
  (`liquid_boundary_geometry_math_3d.h:690-729`), `EvaluateLiquidCavity3D` (`liquid_cavity.cc:104`),
  `EvaluateLiquidEosAtPressure3D` (`liquid_eos.cc:27`); and `SolvePlanarSnowPressureTrial3D`
  (`mixed_contact_3d.cc:432-534`) on the column of 4 x 8 x 4 cells of 0.05 m (Y-up) whose floor face under one
  column is a sealed wall, `snow_mass` 0.0375 kg, incoming normal velocity -0.1 m/s: its
  `solved_wall_normal_velocity` and `pressure_impulse_on_snow`. Rotate results into Z-up with rows
  `{1,0,0},{0,0,-1},{0,1,0}` and store them in the JSON with the `main` commit hash. The program itself is not
  committed; its outline goes into the commit body.
- [ ] **Step 2: Tests** (`LABEL cpu`) `ProofOfConcept, MatchesMainWhereTheMethodIsUnchanged`: coefficients and
  right sides at `1e-12` relative; pressure after removing each sealed blob's mean and velocity at twice the old
  stopping limit; curvature, pocket pressure, pressure law at `1e-12` relative.
  `ProofOfConcept, PlanarSnowTrialEqualsOneMaterialFace`: the same scene with that face as one
  `BlockingMaterialFace` (A w^2, m 0.0375, u* -0.1): material velocity after pressure and `face_mass * (u - u*)`
  at twice the old stopping limit. Supporting evidence only: every quantity here is also covered by an analytic
  test above.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Compare the pressure module with the proof of concept`.

### Task 54: Method paper, solving in batches

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** new task; the paper states how a solve too large for the GPU runs batch by batch.

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/pressure.typ`, `docs/glossary.md`, `docs/method-paper/references.typ` (only
  if an entry below is missing)
- Create: `docs/method-paper/figures/solve_batches.typ`

**Interfaces:**
- Produces labels `<sec-solve-in-batches>`, `<fig-solve-batches>`, `<proc-solve-pressure-in-batches>`.

- [ ] **Step 1: Text** from `plan-decisions.md` "Batched solves (A15)": what stays on the GPU (state, geometry, the
  coarse rungs) and what goes to CPU memory (equations, vectors, working buffers); one arithmetic for both modes, so a
  batched solve has the one-piece iteration count and normally its bytes; the two sum rules (grid-wide, per group);
  labels joined across batches; the ladder's rung 1 built block by block; the sweeps of one iteration (3, or 4 with
  a sealed blob) and their copy volume; the alternative of overlapping pieces with a coarse correction and why it is
  not used (cite Smith, Bjorstad and Gropp 1996 once, by author and year: textbook "additive Schwarz"). Glossary
  rows: `solve batch`, `sweep`, `batched layout`, `per-group total`.
- [ ] **Step 2: Figure** `<fig-solve-batches>`: a time line of one iteration: sweeps A to D over three batches
  (each a range with its touching border), the coarse rungs between B and C on the GPU, the arrows of copies up
  and down with their sizes per block.
- [ ] **Step 3: Pseudocode** `<proc-solve-pressure-in-batches>`: steps 1 to 6 as Task 59 implements (cut, build,
  then per iteration A/B, rungs, C, D, then correction and check). Run typst, look at the pages, run the gate.
  **Commit.** Subject: `Describe how a pressure solve runs in batches`.

### Task 55: `SolveBatches`: the cut, the loaded piece and the copies, on the CPU

**Amended 2026-11-14 (A15):** new task; the shared mechanism of every batched solve (pressure, viscosity, gas).

**Amended 2026-11-14 (Foundation as built):** `solve_batches.cc` is `solve_batches_cpu.cc` (two halves; GPU half in Task
60); `SolveBatches` owns an `OrderBuffers` for its prefix sums, counted in `BuildSolveBatchesMemoryEstimate`.

**Amended 2026-11-14 (A31):** `SolveBatches` owns the `GridBuilderBuffers` lent to its loaded `GridBuffers`
(`BuildGridBuffers` has one form).

**Amended 2026-11-15 (A58):** `BuildSolveBatches(..., stacked_widths, RunMemory& run_memory, RunMemory&
cpu_run_memory)` (design 3): `kRunOn` rows from `run_memory`, the border lists and staging buffer (`kCpu`) from
`cpu_run_memory`; the order buffers one row; `kOutOfMemory` when a piece does not fit; the "without a GPU" test
replaced by `BuildSolveBatches, IntoRunMemory_HandsOutExactlyTheEstimate`. The A15 mechanism is unchanged.

**Kind:** integration

**Files:**
- Create: `native/fluidity/pressure/solve_batches.h`, `solve_batches_cpu.cc`, `solve_batches_test.cc`,
  `solve_batches_no_gpu_test.cc`, their CMake lines
- Modify: `pressure_test_scenes.h`, `pressure_test_scenes.cc` (`CountBlocksInLargestLayer`, `BuildTallBoxGrid`)

**Interfaces:**
- Consumes: grid `BuildRanges`, `BorderSearchBuffers`, `BuildBorderSearchBuffers`, the stacked buffer form of
  `FindBorderBlocks`, the buffer form of `BuildGridPiece`, `GridPiece::block_in_whole_grid_buffer`,
  `BuildGridBuilderBuffers`, `BuildGridBuffers` (one form, taking lent `GridBuilderBuffers`; A31),
  `BuildGridMemoryEstimate`; Foundation `BuildBuffer(..., CpuMemory::kLockedForCopies)`, the slice
  forms of `WriteBuffer`, `ReadBuffer`, `CopyBuffer`.
- Produces exactly the declarations of `plan-decisions.md` "Batched solves (A15)": `SolveBatches`,
  `BuildSolveBatches`, `BuildSolveBatchesMemoryEstimate`, `ApplySolveBatches`, `LoadSolveBatch`, `WriteBatchValues`,
  `ReadBatchValues`, `WriteBatchGroups`, `ReadBatchGroups`, `SolveCopyReport`, `ReadSolveCopyReport`,
  `ResetSolveCopyReport` (explicit instantiations for `double`, `std::uint8_t`, `std::uint32_t` and every
  struct a later task streams; each with `static_assert(sizeof(Value) <= 64)`, the staging bound).
- Test-only (plan decision, `fluidity_pressure_test_scenes`, reused by the viscosity, gas and batches plans):
  `std::uint32_t CountBlocksInLargestLayer(const grid::StaggeredGrid& grid);` (largest number of blocks sharing one
  block coordinate `z`) and `grid::StaggeredGrid BuildTallBoxGrid(double cell_width, grid::CellCoord cell_count);`
  (`BuildBoxGrid` at the origin; used with at least 8 block layers along `z`).
- **Plan decisions:** a whole array is valid when its size is at least `grid.counts().cell_blocks *
  values_per_block` (else `kBadInput`); the loaded array has `max_loaded_blocks * values_per_block` values. The
  loaded order is the piece order (ascending whole block index), so the gathered staging buffer goes up in one
  slice `WriteBuffer`, and the range comes back in one slice `ReadBuffer` (one run in both grids). Copies are
  counted in `SolveCopyReport` by direction; `sweep_count` grows when batch 0 is loaded. Rows (`"pressure: solve
  batches "` prefix, `KeptFor::kWholeRun`): loaded grid buffers (`BuildGridMemoryEstimate(run_on, 1, max_loaded_blocks)`), border search
  (`BuildBorderSearchMemoryEstimate(run_on, max_loaded_blocks)`), two border buffers and the map copy (4 B
  each per loaded block), the per-batch border lists (`where = kCpu`, 4 B per cell block times 2: an upper bound of
  stacked borders over all batches), the staging buffer (`kCpu`, `max_loaded_blocks * 240 * 64` bytes).
  **Plan decision (Foundation as built):** `SolveBatches` owns an `OrderBuffers` built by `BuildSolveBatches` with
  `BuildOrderBuffers(run_on, max_loaded_blocks)`; the loaded begin offsets of `WriteBatchGroups` (an exclusive prefix
  sum over the loaded blocks' item counts) pass it to `ComputeExclusivePrefixSum`, on the
  processor (a CPU `SolveBatches` holds CPU order buffers). **Plan decision (A31):** `SolveBatches` also owns one
  `GridBuilderBuffers` built by `BuildSolveBatches` with `BuildGridBuilderBuffers(run_on, max_loaded_blocks)` and
  lent to its loaded `GridBuffers` (`BuildGridBuffers(settings, builder_buffers)`, the only form); the loaded grid
  buffers row `BuildGridMemoryEstimate(run_on, 1, max_loaded_blocks)` (default `BuilderBuffersSharing::kOnePerGrid`) already counts them. ~~Rows `"pressure: solve batches order buffers"` (given count
  `max_loaded_blocks`, `bytes_each = ComputeOrderBuffersSize(run_on).bytes_per_item`) and `"... order buffers fixed
  bytes"` (given count `fixed_bytes`, `bytes_each = 1`), `where = kRunOn`.~~ **(A58)** Row `"pressure: solve batches
  order buffers"`, given count 1, `bytes_each = ComputeOrderBuffersSize(run_on)` at `max_loaded_blocks` (one piece),
  `where = kRunOn`; the grid, border search and order buffers are built by their `RunMemory&` forms into
  `run_memory`, the border lists and staging buffer into `cpu_run_memory`. Nothing allocates after
  `BuildSolveBatches`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, everything on the CPU):

  ```cpp
  TEST(ApplySolveBatches, RangesCoverEveryBlockOnceAndFit)   // 100 random grids (grid_test_support), max loaded
      // 3 * CountBlocksInLargestLayer, widths {kTouchingCells} and {kTouchingCells, kTouchingCells}: ranges in
      // order, each block once; range plus stacked border <= max_loaded_blocks
  TEST(ApplySolveBatches, LargestRangeIsChosen)              // one block more per range would not fit some range
  TEST(ApplySolveBatches, OneLayerDoesNotFit_IsRunOnCallFailed)  // max loaded = layer: magnitude = blocks needed,
      // limit = max_loaded_blocks
  TEST(WriteBatchValues, LoadedBlocksEqualWholeBlocksThroughTheMap)  // cell, face, corner and block arrays of
      // distinct values on BuildTallBoxGrid(0.1, {8, 8, 64}): every loaded block's values equal the whole block's
  TEST(ReadBatchValues, WritesRangeBlocksOnly)               // border blocks of the whole array keep their bytes
  TEST(WriteBatchGroups, ItemsOfLoadedBlocksInPieceOrder)    // 0 to 5 items per block; loaded begin offsets
  TEST(WriteBatchValues, WholeTooShort_IsBadInput)
  TEST(LoadSolveBatch, CancelSet_IsCancelledAndLoadsNothing) // loaded_batch() unchanged
  TEST(ReadSolveCopyReport, CountsLoadsSweepsAndBytes)       // two sweeps of 4 batches: 8 loads, 2 sweeps, bytes
      // equal the sum the test computes from the loaded block counts
  TEST(BuildSolveBatchesMemoryEstimate, MatchesAllocatedBytes)  // the order buffers rows included
  TEST(BuildSolveBatches, IntoRunMemory_HandsOutExactlyTheEstimate)  // A58: 64 blocks, loaded 16, kCpu; one CPU
      // RunMemory of the sum over both where of ComputeRunMemoryBytes(BuildSolveBatchesMemoryEstimate(...), counts,
      // where), passed twice: handed_out_bytes() == reserved_bytes(); one byte fewer: kOutOfMemory; exact
  // ~~solve_batches_no_gpu_test.cc, LABEL no_gpu~~
  ~~TEST(BuildSolveBatches, OnGpuWithoutAGpu_IsRunOnCallFailed)~~  // A58: a GPU RunMemory cannot exist without a GPU
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (allocates nothing after `BuildSolveBatches`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Cut a grid into batches for a solve that does not fit`.
  New names listed with the reason "A15".

### Task 56: Batched layout of the buffers, the snapshot from pass 1, labels across batches

**Amended 2026-11-15 (A58):** the `kCpu` rows of the batched layout are pieces of `cpu_run_memory`; the batched
estimate test goes through `IntoRunMemory_HandsOutExactlyTheEstimate`; the "without a GPU" test is removed.

**Amended 2026-11-14 (A30|A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers. `WriteSnapshotRange` requires both buffers on one processor (A30).

**Amended 2026-11-14 (A15):** new task.

**Amended 2026-11-14 (reconcile):** adds `ResetSolveCopyReport(PressureBuffers&)` with two tests (the step resets
the copy report through the buffers; `FindSolveBatches` stays const).

**Amended 2026-11-14 (Foundation as built):** in the batched layout the sum and order buffers stay `kRunOn`, the order buffers
sized for the loaded piece or the whole cell count; `solve_batches_cpu.cc` is the file name.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_buffers.h`, `pressure_buffers.cc`, `pressure_memory_estimate.cc`,
  `pressure_buffers_access.h`, `pressure_buffers_test.cc`, `solve_batches.h`, `solve_batches_cpu.cc`,
  `label_propagation.h`
- Create: `snapshot_range.cc`, `snapshot_range_test.cc`, `label_join.cc`, `label_join_test.cc`

**Interfaces:**
- Consumes: Tasks 12, 16 to 18, 21, 31, 55.
- Produces: `BufferSizes::max_loaded_blocks`, `FindSolveBatches(const PressureBuffers&)`,
  `ReadSolveCopyReport(const PressureBuffers&)`, `ResetSolveCopyReport(PressureBuffers&)`, `WriteSnapshotRange`,
  `LabelJoin`, `JoinLabelsAcrossBatches` (declarations: `plan-decisions.md` "Batched solves (A15)" and
  "Reconcile pass (2026-11-14)").
- **Plan decision (reconcile, item 3):** `FindSolveBatches` stays const; the step resets the copy report through
  the buffers, in `pressure_buffers.h`:

  ```cpp
  // Sets the copy report of the buffers' SolveBatches to zero (ResetSolveCopyReport on it). Does nothing for
  // one-piece buffers (no SolveBatches). Allocates nothing, copies nothing.
  void ResetSolveCopyReport(PressureBuffers& buffers);
  ```
- **Plan decisions:** with `max_loaded_blocks > 0` (`kBadInput` above `max_cells / 64`): the rows written by
  `ComputeDistanceToSurface` and `FindLiquidSurface`, the ladder rows (rungs 1 and deeper), the direct-solve factors
  (default room `4 * max_cells` values), and every row counted per cell block or given count stay `where = kRunOn`
  at the limits; every other row counted per cell, face or corner place is `where = kCpu` (built with
  `CpuMemory::kLockedForCopies`) at the limits and again `kRunOn` at `max_loaded_blocks`; `BuildSolveBatches(run_on,
  max_cells / 64, max_loaded_blocks, {kTouchingCells}, run_memory, cpu_run_memory)` is held inside (A58: the `kCpu`
  rows are pieces of `cpu_run_memory`, which the step builds `CpuMemory::kLockedForCopies` in a batched GPU run). `WriteSnapshotRange` copies, for the range
  blocks of `piece`, every pass-1 row, and takes the largest of the snapshot's two scalars; both buffers one-piece
  or the whole one batched (`kBadInput` otherwise); requires both on one processor (A30: debug assertion). Labels: inside a loaded piece Task 21's
  propagation gives each cell the smallest whole cell place index reachable within the piece; for every border
  cell a `LabelJoin{label given here, label its own batch gave}`; joined on the CPU by union with the smaller label
  as root. **(Foundation as built)** The `sum_buffers` and `order_buffers` rows stay `where = kRunOn` in the batched
  layout (they are given-count rows); the sum buffers keeps `max_cells / 64` values (block partials are kept for every
  block across batches), and the order buffers holds the larger of `max_loaded_blocks * 240` (groupings of one loaded
  piece), `max_cells` (rung 1 and deeper, built whole on the processor) and `max_blocking_material_faces`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeBufferBytes, BatchedLayout_RowsSplitByWhere)   // 64 blocks, loaded 16: the kCpu rows equal the
      // one-piece bytes of the same rows; the kRunOn rows are the kept rows plus the loaded rows plus the batches'
  TEST(BuildPressureBuffers, BatchedLayout_AllocatesWhatTheRowsSay)  // A58: through
      // BuildPressureBuffers, IntoRunMemory_HandsOutExactlyTheEstimate's batched case (loaded 4): handed out ==
      // reserved for the CPU block passed twice, exact
  TEST(BuildPressureBuffers, LoadedBlocksAboveMaxBlocks_IsBadInput)
  TEST(WriteSnapshotRange, PiecesGiveTheWholeGridBytes)      // BuildTwoLevelGrid 8 block layers, analytic surface
      // and slanted solid; BuildRanges(grid, 2 layers); per range a piece with its touching border, FindLiquidSurface
      // into piece buffers, WriteSnapshotRange: every pass-1 row equals FindLiquidSurface on the whole grid, bytes
  TEST(WriteSnapshotRange, PieceOfAnotherGrid_IsWrongGridVersion)
  TEST(JoinLabelsAcrossBatches, ChainThroughFiveBatches_SmallestLabelWins)  // hand-made joins
  TEST(JoinLabelsAcrossBatches, NoJoins_EveryLabelItsOwn)
  TEST(ResetSolveCopyReport, BatchedBuffers_ReportZeroAfterReset)  // reconcile: one ApplySolveBatches and a
      // LoadSolveBatch through the buffers' SolveBatches, then ResetSolveCopyReport(buffers): every field of
      // *ReadSolveCopyReport(buffers) is 0
  TEST(ResetSolveCopyReport, OnePieceBuffers_DoesNothing)          // FindSolveBatches null before and after
  ```
  ~~`LABEL no_gpu` (`pressure_buffers_no_gpu_test.cc`):
  `BuildPressureBuffers, BatchedLayoutOnGpuWithoutAGpu_IsRunOnCallFailed`.~~ (A58: removed; a GPU `RunMemory`
  cannot be built without a GPU.)
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject:
  `Keep the pressure working buffers in CPU memory when the solve runs in batches`.

### Task 57: Air pockets, material boundaries and the equations in batches, on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** new task.

**Amended 2026-11-14 (Foundation as built):** the prefix sum through the buffers' `order_buffers`, the per-group sums
through `ComputeTotalOfBlocks` and the buffers' `sum_buffers`.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/air_pockets_cpu.cc`, `material_boundary_cpu.cc`, `pressure_equations_cpu.cc`,
  `volume_correction_cpu.cc`
- Create: `batched_equations_test.cc`

**Interfaces:**
- Consumes: Tasks 20, 21, 23, 24, 26, 42, 44 to 46, 55, 56.
- Produces: the batched path (buffers in the batched layout) of `FindAirPockets`, `FindMaterialBoundaries`,
  `BuildPressureEquations` (with `ComputeEquationIndexes`, `ComputeBlobOfCell`, the blob classes and the sealed-blob
  check), `ApplyAddedVolumeRate` and `ComputeVolumeCorrectionRate`. No signature changes.
- **Plan decisions:** each function cuts with `ApplySolveBatches` on its grid when the bound version differs, then
  sweeps: equation indexes from per-block counts and one `ComputeExclusivePrefixSum` (the buffers'
  `order_buffers`); rows, connections and the
  material and blob term lists per batch, written back for the range blocks; blob and pocket labels by Task 56's
  join and one relabel sweep; blob classes, pocket volumes, `incompatible_volume_rate` and the gas blob sums by the
  per-group rule; material boundary entries per batch, concatenated in batch order (that is face-piece order); the
  volume correction's 64 bisections are 64 sweeps of `filled_volume` and the raw rate, per sealed blob by the
  per-group rule. Lists that stay whole on the processor (pockets, blobs, free-solid entries, gas blobs) are
  written once after the sweeps.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; each scene built on a grid of at least 8 block layers, run with
  `max_loaded_blocks` = 3 and 5 times `CountBlocksInLargestLayer`, and in one piece; the build is local
  arithmetic plus the shared sum rules, so bytes are compared):

  ```cpp
  TEST(FindAirPockets, InBatches_SameLabelsAndListAsOnePiece)       // tank 0.8 x 0.8 x 1.6 m, w 0.05, two
      // bubbles, one across a batch boundary, and a vented column: pocket_of_cell and the list, bytes
  TEST(BuildPressureEquations, InBatches_SameEquationsAsOnePiece)   // Task 23's SymmetricAndPositive scene on
      // BuildTwoLevelGrid with 8 layers (its free solid's 3 entries in two batches), Task 45's material faces and a
      // sealed gas blob over three batches: indexes, rows, connections, lists, labels, classes, sums: bytes
  TEST(FindMaterialBoundaries, InBatches_SameEntries)               // Task 40's oil drop moved across a boundary
  TEST(ComputeVolumeCorrectionRate, InBatches_SealedBlobSumsToZero) // as Task 26, the blob over four batches:
      // |sum| <= 1e-12 * sum |rate|, and bytes equal to one piece
  TEST(BuildPressureEquations, CancelBetweenBatches_IsCancelled)    // flag set by a LoadSolveBatch counter at the
      // third load (test hook through pressure_buffers_access.h): kCancelled
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with the `// Step N:` numbers of design section 5.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Find air pockets and build the pressure equations in batches`.

### Task 58: The ladder in batches, on the CPU

**Amended 2026-11-14 (A15):** new task.

**Amended 2026-11-14 (Foundation as built):** tests `LABEL cpu` (was `no_gpu`).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_ladder_cpu.cc`, `pressure_ladder_math.h`, `pressure_ladder_test.cc`

**Interfaces:**
- Consumes: Tasks 31, 32, 45, 46, 55, 56.
- Produces: the batched path of `BuildPressureLadder` and `ApplyPressureLadder`.
- **Plan decisions:** `BuildPressureLadder` builds rung 1 per batch from the streamed equations (Task 31's
  per-block numbering, so the result does not depend on the cut) into the whole rung storage on the processor,
  then rungs 2 and deeper there. `ApplyPressureLadder` in the batched layout is three steps: sweep "down" (sealed
  blob totals removed, rung-0 weighted Jacobi, residual, gather into rung 1), the whole rungs, sweep "up" (spread,
  rung-0 weighted Jacobi up, sealed blob means removed in a further sweep). Task 59 fuses these sweeps with the
  iteration's own; the public function used alone (tests, viscosity never) takes 4 sweeps.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildPressureLadder, InBatches_SameRungsAsOnePiece)    // BuildTwoLevelGrid 8 layers, two blobs, loaded 3
      // and 5 layers: every rung's cells, members and weights equal one piece, bytes
  TEST(ApplyPressureLadder, InBatches_SameOutputAsOnePiece)   // random input, a sealed blob over three batches:
      // bytes (same arithmetic); with the factor room of the batched layout and a drop beyond it: within 1e-12
      // relative of a one-piece pass given the same room
  TEST(ApplyPressureLadder, InBatches_SymmetricAndPositive)   // Task 32's test in the batched layout
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject:
  `Build and apply the pressure ladder in batches`.

### Task 59: `SolvePressure` and the pieces in batches, on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** new task.

**Amended 2026-11-14 (A24):** the A24 forms of Tasks 45 to 47 run in batches too: the blob buffers stay whole on
the processor (like `D_j`, `r_j`), the material-face lists stream with the rows, blob sums by the per-group rule.
Test `CoupledSolvePieces, InBatches_SameAsOnePiece`.

**Amended 2026-11-14 (A26/A27):** the `CoupledVelocities` forms of Task 47, `ComputeCoupledPush` and the
`ComputeOwnWeight` form with `material_face_scale` run in batches too, each one sweep. **Plan decision:** the
free-solid buffers (`free_solid_motion`, `free_solid_push`, `free_solid_motion_before_pressure`) and the buffers
indexed by `BlockingMaterialFace` entry (`blocking_material_velocity`, `material_face_scale`,
`blocking_material_push`) stay whole on the processor (like the blob buffers); each batch reads the entries of its
loaded pieces and writes only the entries of its range's face pieces (each entry by exactly one batch); the
per-equation material and free-solid entry lists stream with the rows; each free solid's push sums by the per-group
rule (per block partials kept across batches, one `math::TreeSum` per component after the sweep). Test
`CoupledVelocityPieces, InBatches_SameAsOnePiece`.

**Amended 2026-11-14 (Foundation as built):** grid-wide totals between sweeps through `ComputeTotalOfBlocks` with the
buffers' `sum_buffers`.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/pressure_solve_cpu.cc`, `solve_pieces_cpu.cc`, `pressure_solve_test.cc`,
  `solve_pieces_test.cc`

**Interfaces:**
- Consumes: Tasks 28, 29, 34, 47, 55 to 58.
- Produces: the batched path of `SolvePressure`, `CheckFlowImbalance` and every function of
  `coupled_solve_pieces.h` and `flow_imbalance.h` (each one sweep when called alone; viscosity's Task 27 calls them
  with whole arrays in CPU memory or on the processor).
- **Plan decisions** (numbered as `<proc-solve-pressure-in-batches>`): (1) `ApplySolveBatches` if the version
  changed; (2) start: `ComputeStartPressure` and the true error, one sweep, plus the rank-one sums; (3) per
  iteration the sweeps A and B (one sweep when no blob is `kSealed`), the whole rungs, C, D of `plan-decisions.md`,
  computing exactly Task 34's and Task 47's arithmetic; the step sizes and the stop test on the processor between
  sweeps (every grid-wide total by `ComputeTotalOfBlocks` with the buffers' `sum_buffers`, over block partials
  kept across the batches of a sweep); (4) when the running error passes the stop rule: correction and
  `CheckFlowImbalance` (two sweeps), restart
  from the true error (two sweeps) when the check fails; (5) after acceptance, one write-back sweep writes
  `SolvedArrays` (pressure, velocity, material velocities, `sealed_gas_pressure`) for the range blocks; nothing is
  written to them before; (6) a progress read (and the cancel flag) before every batch load, so cancel is seen
  between batches. A sweep runs all its batches and keeps the smallest failure key. Memory: nothing allocates.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, both processors the CPU):

  ```cpp
  TEST(SolvePressureInBatches, TwoCellFixturesInATallGrid_SameAsOnePiece)  // Task 34's two fixtures placed in a
      // BuildTallBoxGrid(1, {4, 4, 32}) of kSolid cells: pressures (1.75, 1.375) and (1.28125, 1.09375), 1e-9 rel
  TEST(SolvePressureInBatches, CancelBetweenBatches_ReturnsCancelledOutputsUntouched)  // hook at the 7th load:
      // kCancelled; SolvedArrays bytes equal the pattern written before the call
  TEST(SolvePressureInBatches, FailureInOneBatch_PublishesNothing)  // a NaN written (pressure_buffers_access.h)
      // into the right side of one equation of the last batch: kNotFinite with that cell_index, the same failure
      // as one piece given the same NaN; SolvedArrays keep their bytes
  TEST(SolvePressureInBatches, SweepsPerIteration)          // copy report: 3 sweeps per iteration with no sealed
      // blob, 4 with one; batch_loads == sweeps * batch_count
  TEST(CheckFlowImbalance, InBatches_SameReportAsOnePiece)  // Task 29's wrong-pressure case in a tall grid: bytes
  TEST(CoupledSolvePieces, InBatches_SameAsOnePiece)        // A24: Task 47's two fixtures placed in a
      // BuildTallBoxGrid(1, {4, 4, 32}) of kSolid cells: every A24 form (blob forms, material faces,
      // ComputeBlockingMaterialVelocities, CheckFlowImbalance with given blob pressures) gives the bytes of the one
      // piece; blob buffers whole on the processor
  TEST(CoupledVelocityPieces, InBatches_SameAsOnePiece)     // A26/A27: Task 45's two-cell fixture and Task 47's first
      // fixture placed in a BuildTallBoxGrid(1, {4, 4, 32}) of kSolid cells, and Task 23's free solid with its 3
      // entries in 3 batches (as Task 61): every CoupledVelocities form, ComputeCoupledPush and the
      // material_face_scale form, with the given w, xi, xi_before and scale of Task 47's tests, give the bytes of
      // the one piece; free-solid and material-entry buffers whole on the processor, per-group sums
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step N:` comments of the procedure.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Solve for the pressure in batches`.

### Task 60: Solving in batches on the GPU

**Amended 2026-11-14 (A15):** new task; GPU forms of Tasks 55 to 59.

**Amended 2026-11-14 (Foundation as built):** `solve_batches_gpu_calls.h` is `solve_batches_gpu.h` returning
`RunOnCallStatus`; the copy and partial kernels wrap per-thread functions of `solve_batches_steps.h` with a `cpu` steps
test; digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/pressure/solve_batches_gpu.cu`, `solve_batches_gpu.h` (private; was
  `solve_batches_gpu_calls.h`), `solve_batches_steps.h` (private shared), `solve_batches_steps_test.cc`,
  `solve_batches_gpu_test.cc`, `pressure_solve_batched_gpu_test.cc`, their CMake lines
- Modify: `solve_batches_cpu.cc` (dispatch on `run_on`; the GPU half under `#if defined(FLUIDITY_HAS_GPU)`),
  `pressure_equations_gpu.cu`, `pressure_ladder_gpu.cu`,
  `pressure_solve_gpu.cu`, `solve_pieces_gpu.cu`, `air_pockets_gpu.cu`, `volume_correction_gpu.cu`

**Interfaces:**
- Produces (private): `GatherBlocksIntoPiece` (one thread block per loaded cell block copies `bytes_per_block`
  from the whole array on the processor through the piece map: the "within processor" copy), the per-cell-block
  group-partial GPU functions, and the GPU path of every function of Tasks 55 to 59. Copies to and from CPU memory
  are slice `WriteBuffer` / `ReadBuffer` of the staging buffer and of the range; nothing else crosses. **(Foundation
  as built)** Each of them returns `RunOnCallStatus` and launches through `base/gpu_launch.cuh`; the copy of one
  byte range through the map and the group partial of one cell block are `FLUIDITY_CPU_AND_GPU` functions of
  `solve_batches_steps.h`, which the CPU path of Tasks 55 to 59 calls too.

- [ ] **Step 1: Tests** (`LABEL gpu`): `ApplySolveBatches, OnGpu_SameCutAsCpu`, `WriteBatchValues,
  OnGpu_LoadedEqualsWholeThroughTheMap`, `BuildPressureEquations, InBatchesOnGpu_SameAsOnePieceOnGpu` (Task 57
  scenes, bytes), `SolvePressureInBatches, OnGpu_WithinToleranceOfOnePieceOnGpu` (Task 61 scenes, the Global
  Constraints' batched bounds), `SolvePressureInBatches, OnGpuTwice_GivesIdenticalBytes`,
  `SolvePressureInBatches, OnGpuTwoCuts_GiveIdenticalBytes`, `PressureScene, TwoMillionEquationsInBatches_Converges`
  (Task 39's 128^3 box with `max_loaded_blocks` = a quarter of its blocks: converges in the one-piece iteration
  count; prints iterations, seconds per iteration and bytes copied per iteration). Every `gpu` test of this task
  passes its main outputs to `WriteResultDigest`. `LABEL no_gpu`:
  `SolvePressure, BatchedLayoutOnGpuWithoutAGpu_IsRunOnCallFailed`. `LABEL cpu` (the main evidence until the
  hardware run): `SolveBatchesSteps, EveryThreadBlockAndThread_EqualsCpuLoop` (the map copy and the group partial
  over every thread block and thread, launch order and reverse, on `BuildTallBoxGrid(0.1, {8, 8, 64})`: bytes equal
  to `WriteBatchValues` and to the CPU partials).
- [ ] **Step 2: Implement.** **Step 3: Run** the gate. **Commit.** Subject: `Solve for the pressure in batches on
  the GPU`. Body: `CUDA: compiled, not run`.

### Task 61: Scene tests, a batched solve against one piece

**Amended 2026-11-14 (A15):** new task; the analytic scenes of Tasks 34 to 38, 49 and 50 rerun in batches.

**Kind:** integration

**Files:**
- Create: `native/fluidity/pressure/pressure_batched_scene_test.cc`

Each test builds the named scene on a grid of at least 8 block layers (the scene's own builder with the domain
lengthened along `z` or the cell width halved, stated per test), runs the chain of Task 36 once in one piece and
in batches with `max_loaded_blocks` = 3, 5 and 7 times `CountBlocksInLargestLayer` (a small forced GPU limit), and
asserts: counts, labels, classes and pocket lists equal; velocity within `2 * max_volume_error_rate * w + 1e-9 *
max |u|` and pressure within that bound times `face_density * pressure_distance / dt` of the one-piece solve; the
three cuts identical bytes; the analytic expectation of the original test where it has one.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(SolvePressureInBatches, HydrostaticColumn_WithinToleranceOfOnePiece)  // open-top box 0.4 x 0.4 x 1.6 m,
      // w 0.025, H 1.43 (two-level: big below 0.8); also p == rho g (H - z) within the pressure bound
  TEST(SolvePressureInBatches, SealedCubeCutIntoThree_WithinToleranceAndSameBytesForEveryCut)  // Review Focus 5:
      // Task 34's sealed cube at w 0.0625 (32^3, 8 layers) with carried 1e5 Pa: mean kept; linear pressure
  TEST(SolvePressureInBatches, TwoBlobsOneSealed_WithinTolerance)    // Task 37's scene lengthened along z
  TEST(SolvePressureInBatches, FreeSolidAcrossBatches_WithinTolerance)  // Task 23's free solid, entries in 3 batches
  TEST(SolvePressureInBatches, WaterOnSnowSlab_WithinTolerance)      // Task 49's slab at w 0.0125 (16 x 16 x 32), one
      // step; material velocities within the velocity bound
  TEST(SolvePressureInBatches, SealedGasAcrossBatches_WithinTolerance)  // Task 50's heated pocket at w 0.0125: the
      // blob spans four layers; sealed_gas_pressure within the pressure bound, 6.5625 Pa within 1e-9 relative
  TEST(SolvePressureInBatches, FitsOnlyInBatches)                    // hydrostatic scene: budget B = half the
      // one-piece kRunOn bytes (ComputeMemoryBytes of BuildPressureMemoryEstimate, where kRunOn); the batched layout at
      // 3 layers has its kRunOn bytes below B; built and solved in that layout: within tolerance of one piece
  TEST(SolvePressureInBatches, RepeatRun_IdenticalBytes)
  ```
- [ ] **Step 2: Run.** A difference beyond the bound names the first cell: find the sweep with Task 59's
  `CheckFlowImbalance, InBatches_SameReportAsOnePiece` and fix the owning commit, never the bound.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Test batched pressure solves against one piece`.

### Task 62: Comment pass and hand-over

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A18):** renumbered from Task 46; its lists include Tasks 43 to 50.

**Amended 2026-11-14 (A15):** renumbered from Task 54; its lists include Tasks 54 to 61.

**Amended 2026-11-14 (A24):** the hand-over lists the A24 changes to the pieces.

**Amended 2026-11-14 (A26/A27):** the hand-over lists the A26/A27 additions for the joint solve (Step 3).

**Amended 2026-11-14 (Foundation as built):** the hand-over also lists the Foundation-as-built adoption (sum and order buffers
members and rows, the `RunOn` parameter of `ComputeBufferBytes` and `BuildPressureMemoryEstimate`, the removed `launch_size.h`,
the same-bits promise of `ComputeDistanceToSurface`, `solve_batches_cpu.cc`).

**Amended 2026-11-15 (A58, A59, A60, A64, reconcile item 31):** the hand-over lists the `RunMemory` builders and
rows, `kOutOfMemory` last, the guided free solids of Task 47a, the A60 note (no new pressure function), and the
grid-only pieces moved to the later group (Task 63, not part of the done-criteria).

**Kind:** mechanical

**Files:**
- Modify: any file of the module; `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every file of `native/fluidity/pressure/` from top to bottom as a C++
  programmer who has never read a fluid paper: each comment true of the final code; says why, not what; a worked
  example with numbers for each non-trivial public function (the two-cell fixture is the natural one for the
  equations); every word in the glossary; `Uses` and `Used by` lists correct; `// Step N:` numbers match the
  paper's procedures.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`),
  run format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently: the
  plan decisions of Tasks 4, 5, 10, 12, 13, 14, 16, 17, 21, 23, 26, 31, 32, 34, 40, 42, 44, 47, 51, 55 to 59; the
  removal of `kMaterialBoundaryNotSupported` (Task 42); the A18 coupling (Tasks 43 to 50) with its open point (Task
  44 (f), the material's own volumetric stiffness) and the unproved sealed gas elimination (Task 52); the A15
  batched solve (Tasks 54 to 61: `SolveBatches`, the batched layout, the per-group sum rule and the ladder layout
  in both modes, the direct-solve room); the A24 pieces for the joint viscous solve (Tasks 45 to 48, 59: no piece
  refuses material faces; the forms without a blob buffer refuse only equations with a sealed gas blob; the overloads
  `ComputePressurePush` with `sealed_gas_pressure`, `ComputeVolumeError` with `sealed_gas_volume_error`,
  `ComputeOwnWeight` with `sealed_gas_own_weight`, `ComputeStartPressure` with `sealed_gas_pressure`,
  `ComputeBlockingMaterialVelocities`, `CheckFlowImbalance` with given blob pressures; `g_k` unscaled in own weights
  and in `BuildPressureLadder` with a scale); the A26/A27 pieces for the joint viscous solve (Tasks 47, 48, 59:
  `struct CoupledVelocities`, the `ComputeVolumeError` and `CheckFlowImbalance` forms with `CoupledVelocities`,
  `ComputeCoupledPush`, the `ComputeOwnWeight` form with `material_face_scale`; material faces and free solids are
  velocity unknowns there, not eliminated; the A24 forms without `CoupledVelocities` no longer called by viscosity
  and kept for `SolvePressure`, which is unchanged; the plan decision
  `EquationInputs::free_solid_motion_before_pressure` as an owner question; free solids left out of the outer diagonal); the remaining known-open test (piston
  receiving volume); the Foundation-as-built adoption (the `sum_buffers` and `order_buffers` of `PressureBuffers`
  and of `SolveBatches` with their rows, the `RunOn` parameter of `ComputeBufferBytes` and
  `BuildPressureMemoryEstimate`, the removed `launch_size.h`, the same-bits promise of `ComputeDistanceToSurface`,
  `solve_batches_cpu.cc`, the `_steps_test.cc` files); (2026-11-15) the A58 builders `BuildPressureBuffers` and
  `BuildSolveBatches` with `RunMemory& run_memory, RunMemory& cpu_run_memory`, one estimate row per buffer,
  `PressureError::kOutOfMemory` appended last (Tasks 3, 12, 42, 55, 56); the A59 guided term, `FreeSolidInertia`,
  `EquationInputs::free_solids` replacing `free_solid_inverse_inertia`, `ComputeGuidedSolidMotion` handed to solids
  (Task 47a); A60: the force on driven and fixed solids is solids' reduction, pressure adds nothing; A64: no
  grid-only input in the first version (Tasks 14, 26; Task 63 later).
- [ ] **Step 4: `docs/gpu-runs.md`:** a `pressure` row, result `compiled, not run`, listing every `gpu` test of
  Tasks 11, 15, 19, 20, 22, 25, 26, 27, 30, 33, 35, 39, 41, 47a, 48, 60 for the end-of-programme run.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate's output, new public
  names with reasons, the plan decisions, and the statement that no device code has run.

---

## Deep liquid without particles (later; needs review)

**Amended 2026-11-15 (A64):** new group. These tasks are excluded from the module's done-criteria: the pressure
module is done at Task 62 without them. Owner: the cell resize and batches plan's group of the same name
(`2026-11-14-rebuild-11-cell-resize-and-batches.md`, "Deep liquid without particles (later; needs review)"); an
independent review of that design comes first, and this group runs only after that group's two gate tests
(`GridOnlyRegion, StandingWave_KeepsAmplitude`, `GridOnlyRegion, ParticleBandEdge_StaysStable`) are written. In the
first version every liquid cell holds particles, so pressure reads no grid-only mark.

### Task 63: Grid-only cells in the distance and the volume correction (later)

**Amended 2026-11-15 (A64):** moved here from Tasks 14 and 26; design text marked "not in the first version (A64)".

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/pressure/liquid_surface.h`, `distance_to_surface_cpu.cc`, `distance_to_surface_gpu.cu`,
  `distance_to_surface_math.h`, `distance_to_surface_test.cc`, `volume_correction.h`, `volume_correction_cpu.cc`,
  `volume_correction_gpu.cu`, `volume_correction_math.h`, `volume_correction_test.cc`, `pressure.typ` (one sentence)

**Interfaces:**
- Consumes: the cell resize group's `is_grid_only_region` (`grid::CellValues<std::uint8_t>`, 1 for a grid-only cell).
- Produces: `SurfaceDistanceInputs::is_grid_only_region` (null: every cell holds particles) and the parameter
  `const grid::CellValues<std::uint8_t>* is_grid_only_region` of `ComputeVolumeCorrectionRate`, after
  `target_volume_ratio`, as the design's "not in the first version" text states. **Plan decision** (moved from Task
  14): a corner of any grid-only cell gets `-2` widths.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; Task 14's lattice and grid, Task 26's cell):

  ```cpp
  TEST(ComputeDistanceToSurface, GridOnlyCell_IsMinusTwoWidths)   // the cells of z < 0.1 marked: every corner of
      // a marked cell == -2 w exactly; unmarked corners unchanged (bytes)
  TEST(ComputeVolumeCorrectionRate, GridOnlyCell_IsZero)          // the Task 26 cell marked: rate added == 0
      // exactly; null mark: Task 26's value
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design steps 1 and 17 with the marks. **Step 4: Run** the
  gate. **Commit.** Subject: `Treat grid-only cells as full in the liquid surface`.
