# Rebuild Viscosity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the viscosity module of the rebuild: the deformation-rate samples of a thick liquid on the
mixed-size staggered grid, the cell viscosity of ingredient mixtures, sticking to solids, and
`SolveViscosityAndPressure`, the one joint solve of face velocity and cell pressure for every scene with a
viscous ingredient, with its capability check, on CPU and (compiled, not run) GPU. All three geometric
capability exclusions are lifted inside this plan; no combination of materials is excluded because of viscosity
(A24: snow and springy solids; A27: free solids). Owner answer A15 (Tasks 26 to 28): when the joint solve's buffers do not
fit the GPU, it runs batch by batch with its working buffers in CPU memory through pressure's `SolveBatches`, agreeing with
the one-piece solve within its tolerance. Owner answer A24 (Tasks 15a, 15b, 17a, 19a, 27a): the joint solve carries
the blocking-material face terms and the sealed gas term, as `pressure::SolvePressure` does (A18). Owner answer A25
(Task 4): a liquid is viscous exactly when its viscosity is above 0; `ComputeViscousEffectPerStep` gives the planner
the number it prints. Owner answers A26 and A27 (Tasks 15c, 15d): a viscous liquid sticks to snow and springy
solids, whose face velocities beside the liquid are unknowns of the joint solve, and free solids have their six
velocities in it; an inviscid liquid keeps sliding. Owner answers of 2026-11-15 (`plan-decisions.md` "Owner answers
A40-A64 applied"): A57 (Tasks 4, 4a) the advice to set viscosity to 0 compares with the transfer's own smoothing;
A59 (Task 15c) guided free solids are solved in their generalized velocities; A60 (Tasks 15e, 15g) the viscous push
on every solid is read from the solve; A53 (Task 15f) fixed and driven solids have a slip length; A58 (Task 7) every
buffer is a piece of the run's reserved memory; reconcile item 31 (Task 3) `kOutOfMemory` is the last error value.

**Architecture:** Folder `native/fluidity/viscosity/`, namespace `fluidity::viscosity`. One public CPU-only
header `solve_viscosity_and_pressure.h`, one shared formula header `deformation_rate.h`, one shared error
header. Samples are built by one shared per-cell function (`BuildSamplesOfCell`) that a CPU loop and a GPU
function both call; the velocity operator is two gathers (sample pass, face pass) with no floating-point
atomics. The outer pressure loop calls pressure's exposed pieces; pressure includes no viscosity header. All
storage lives in one `ViscosityBuffers` that allocates once.

**Tech Stack:** C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest), Typst + CeTZ
(method paper), clang-format and clang-tidy 23.1 (the versions `pyproject.toml` pins), uv.

**Spec:** `docs/superpowers/specs/modules/06-viscosity-design.md` (the design: its section 3 is the contract),
and the programme design `docs/superpowers/specs/2026-11-13-production-rebuild-design.md` (sections 6, 7, 8,
8a). Both in the `main` checkout, read-only. Also read there: `modules/owner-answers.md`,
`modules/shared-interface.md`, `modules/plan-decisions.md` (wins over the design), section 3 of
`05-pressure-design.md` (the pieces this module calls), 3.7 of `02-grid-design.md`, 3.2 of
`01-contact-design.md`, and the transfer design's `CellsFromParticles`.

## Global Constraints

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes
  the `AGENTS.md` checklist; new public names listed in the commit message with a one-line reason; tests arrive
  with the code. A commit must pass every part of the gate that exists at that commit.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the strict CPU build and every test
  in the debug and the release preset, and the CUDA build compiled in the container (its `cuda` step fails, naming
  the file, for a tracked `.cc` or `.cu` file that no build compiles). clang-format and clang-tidy are the pinned
  23.1.
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure patterns apply to every task (`AGENTS.md`
  "Structure"; `specs/modules/plan-decisions.md` "Foundation after the final review"). Two halves: the public
  function lives in `<name>_cpu.cc` (checks, `RunOn` dispatch, the CPU loop) and calls the GPU half under
  `#if defined(FLUIDITY_HAS_GPU)`; the GPU half `<name>_gpu.cu` is declared in the private `<name>_gpu.h`, returns
  `RunOnCallStatus`, which the CPU half maps with `ToBufferError` and then into `ViscosityFailure` through one
  mapping function, and launches, waits and holds temporary memory only through `base/gpu_launch.cuh`
  (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`). Every GPU function is a thin loop over
  `FLUIDITY_CPU_AND_GPU` per-item or per-thread functions in a shared header; a `cpu` test runs them for every
  thread block and thread of the launch, in launch order and in reverse, against the CPU loop (the main evidence
  until the hardware run). Working buffers: `SumBuffers` and `OrderBuffers` are members `sum_buffers` and
  `order_buffers` of `ViscosityBuffers`, built once by `BuildViscosityBuffers` from its settings and passed to
  `ComputeTreeSum`, `ComputeLargestMagnitude`, `BuildStableOrder`, `ComputeExclusivePrefixSum` and
  `BuildGroupsByDestination` in every step; `BuildViscosityMemoryEstimate` counts their bytes
  (`ComputeSumBuffersSize`, `ComputeOrderBuffersSize`). Errors: a failed check writes nothing; after
  `kRunOnCallFailed` the outputs are unspecified and the run stops; `kOutOfMemory` comes only from building forms (`ViscosityError::kOutOfMemory`, returned only by
  `BuildViscosityBuffers`; reconcile item 25). Buffers and arrays passed together live on one processor: a
  precondition (`Requires:`, debug assertion), not an error value (A30). Each base sum, order, prefix sum and
  grouping has one form, which takes its buffers (A31). **Amended 2026-11-14 (A30|A31|reconcile).**
  **Amended 2026-11-15 (A58):** `BuildViscosityBuffers` takes `RunMemory& run_memory, RunMemory& cpu_run_memory` last
  (the batched layout builds on both processors; in a CPU run both are the one block); `kOutOfMemory` also maps
  `BufferError::kRunMemoryTooSmall` (magnitude = bytes asked) and is the last value of `ViscosityError` (reconcile
  item 31). Tests build their `RunMemory` from the estimate (`ComputeRunMemoryBytes`).
  Exactness: discrete outputs equal on CPU and GPU, floating-point within this module's tolerances unless a header
  promises the same bits unless NaN. Tests link `fluidity_base_test_support` (`ToBits`, `BuildBufferHolding`,
  `ReadAll`, `CheckWithinTolerance`, `CheckSameBitsOrBothNaN`) and the module's helper libraries made with
  `fluidity_add_test_library`, named in `LINKS`; `gpu` tests call `WriteResultDigest` on their main outputs. Labels:
  `cpu` everywhere, `no_gpu` only for behaviour on a machine without a GPU, `gpu` only on hardware. Every target is
  created through the helpers of `cmake/fluidity_targets.cmake` and every source file is listed in a CMake file.
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

Viscosity's own constraints (values from the design; never change one without the owner):

- **Section 3 is the contract.** Implement its declarations exactly. Additions are marked **Plan decision**
  in the task that makes them and repeated in the hand-over.
- **Words.** `deformation_rate`, `stretch`, `shear`, `sample_weight`, `rate_from_solids`, `viscosity` (Pa s;
  papers: dynamic viscosity), "surface force per area" (never "traction"), `filled_volume`, `open_volume`,
  `face_inertia`, `flow_imbalance`, `squeeze_allowance`. Never `strain`, `residual`, `prescribed`, `row`
  (alone), `owner` (of an edge), `publish`, `quadrature`, `stencil`. A literature word appears once in a
  comment, in parentheses.
- **Constants:** `kMaxFacesPerSample = 8`, `kFitReach = 1.5` cell widths, `kFitIndependence = 1.0e-4` cell
  widths, at most 15 samples per cell (3 stretch, 12 shear); `max_velocity_iterations = 256`,
  `velocity_tolerance = 1.0e-10`, `max_pressure_iterations = 256`, `kept_response_count = 32`; acceptance by
  `pressure::SolveSettings` (`max_volume_error_rate = 2e-5 1/s`, `jump_error_fraction = 1e-3`).
  Stretch weight `2 x viscosity x filled_volume`; shear weight `4 x` (volume-weighted harmonic mean over the
  four liquid quarters) `x` their volume.
- **Order of sums** (amended 2026-11-14, A15, both modes). Samples ordered by the cell's block index, then
  ascending `CellId::value`, then `SampleKind`, then axis pair (XY, XZ, YZ), then quarter (0..3); coefficients inside a sample by ascending `FacePlaceIndex`; the by-face view lists
  `(sample index, coefficient)` in ascending sample index. Sample pass: one thread per sample, its
  coefficients in order. Face pass: one thread per face, `face_inertia x u` first, then its uses in order.
  Dot products: one partial per cell block (the block's places in ascending index), then `math::TreeSum` over
  the blocks in block order (A15: replaces `ComputeTreeDot` over the whole buffer wherever Tasks 12, 14, 15 and 17
  say it); largest values: `ComputeLargestMagnitude` (exact in any order). **Amended 2026-11-14 (Foundation as
  built):** where the block partials sit in a buffer, the tree sum over them is `ComputeTreeSum(partials,
  block_count, buffers.sum_buffers)`, whose header promises `math::TreeSum`'s bits on both processors; largest
  values are `ComputeLargestMagnitude(values, count, buffers.sum_buffers)`. `power_lost_to_viscosity`: one partial per cell
  block (its samples in sample order), in the grid's block order, then `math::TreeSum`. No floating-point
  atomics; CPU threads only split by destination.
- **Every liquid is in one material set** (shared-interface section d): `cells_from_particles` has exactly one
  entry; gas marker, snow and springy-solid sets are never passed. Two liquids that do not mix are supported
  by pressure (plan-decisions): their jumps reach this module only through `pressure::ComputePressurePush` with
  `GivenTerms::kIncluded`; no viscous jump term exists.
- **Viscous liquid always sticks to solids** (decision 3), on every path, with no setting. **Amended 2026-11-15
  (A53):** sticking is the default; a fixed or driven solid or the domain box may have a slip length `lambda`
  (`ViscosityInputs::solid_slip_length`), entered in rule (a) as `d + lambda` (Task 15f); free solids and blocking
  materials always stick.
- **Blocking materials and sealed gas inside the joint solve (A24; Tasks 15a, 15b, 17a, 19a, 27a;
  `plan-decisions.md` "Viscous liquids with blocking materials and sealed gas (A24)").** Amended 2026-11-14 (A24),
  replacing the reconcile pass's open note (the owner chose option C). Each `BlockingMaterialFace` is eliminated
  exactly into the volume balance of its solved ends, as `SolvePressure` does (`g_k = dt A_k^2 / m_k`, not scaled by
  `connection_scale`); each `SealedGasBlob` is one more unknown `q_j` of the outer loop (not eliminated: its
  connections answer through the viscous velocity solve, so no two-point weight exists). The inner velocity solve
  and the samples do not change: at a liquid-material face the liquid slides (a blocking material is not a
  `contact::Solid`, rule (b)) and the stretch samples see the material at its velocity before pressure. Outputs go
  to `SolvedCouplingArrays`. Snow, springy solids and sealed gas beside a viscous liquid are never excluded; free
  solids stay excluded (Task 6). No path of this module reads sealed gas one step late.
- **A viscous liquid sticks; free solids in the solve (A26, A27; Tasks 15c, 15d; `plan-decisions.md` "A viscous
  liquid sticks, and free solids in the viscous solve (A26, A27)").** Amended 2026-11-14; replaces, in the bullet
  above, the elimination of material faces, the slide at a material face and "free solids stay excluded". The
  velocity unknowns are `x = (u, w, xi)`: liquid faces, each `ViscousMaterialFace` (the band of a blocking set within
  `kFitReach` widths of a liquid cell centre) and six per free solid; `K = M + dt E^T W E` over `x`; nothing is
  eliminated but sealed gas stays an outer unknown. Stick: rule (a) with the material's face unknown at its true
  distance, or the free solid's `V + Omega x r`. No combination is excluded. Snow has no special case; melting by
  liquid water is not modelled, so liquid-on-snow is checked only for not breaking (Task 19a) and every analytic
  test uses springy solids or rigid bodies. **Amended 2026-11-15 (A59, A60):** a free solid's unknowns are the m
  generalized velocities `q` of its `solids::AllowedMotions` (`xi = J q + xi_0`; block `J^T (M + dt R^T E_s^T W E_s R)
  J`, coefficients `R J`); the solve never moves a forbidden direction (Task 15c). Every sample reading a solid of any
  motion kind keeps its solid index and six `R` coefficients, and the solve returns the viscous push on every solid
  (Task 15e).
- **What counts as viscous (A25).** A liquid ingredient is viscous exactly when its viscosity is above 0; the step
  decides stage 11 once per run. This module gives the planner `ComputeViscousEffectPerStep` and
  `kSmallestVisibleViscousEffectPerStep = 2.5e-4` (Task 4); nothing in the solve depends on the number.
  **Amended 2026-11-15 (A57):** the constant is removed; the planner also reads `viscosity_advice.h` (Task 4a:
  `kNegligibleViscosityRatio`, `kSmallestVisibleWavelengthInCells`, `kNegligibleViscousChange`,
  `ComputeViscosityToSmoothingRatio`, `ComputeFourCellFeatureChange`, `ComputeReynoldsNumber`).
- **Batched solve (A15, Tasks 26 to 28; `plan-decisions.md` "Batched solves (A15)").** `ViscosityBufferSettings`
  gains `max_loaded_blocks`; buffers built with it keep their per-place, per-sample and per-coefficient rows in CPU
  memory, and `BuildViscositySamples`, the passes and `SolveViscosityAndPressure` then run batch by batch through
  `pressure::SolveBatches` (stacked widths `{kTouchingCells, kTouchingCells}`); pressure's pieces run their own
  sweeps. No signature changes. Batched against one piece: the CPU-against-GPU tolerances below; against batched
  with another cut, and repeats: identical bytes.
- **Nothing is written to `velocity_after_pressure` or `cell_pressure` before step 12**; on any failure,
  including `kCancelled`, both are byte-identical to their state before the call. **Amended 2026-11-14 (Foundation
  as built):** except after `kRunOnCallFailed` from a failed GPU call, when the outputs are unspecified and the run
  stops (`AGENTS.md` "One error enum per module"); a failed check, a solver failure and `kCancelled` still leave
  both outputs byte-identical.
- **Tolerances, CPU against GPU (design section 6):** sample coefficients and weights, one operator use:
  `1e-15 + 1e-13 x |value|`; solved velocity `1e-9 m/s + 1e-7 x largest speed`; pressure
  `1e-6 Pa + 1e-6 x largest pressure`.
- **Test labels.** **Amended 2026-11-14 (Foundation as built):** shared-formula tests and every CPU-path test are
  `fluidity_add_test(<name> SOURCES ... LABEL cpu LINKS fluidity_base_test_support ...)`; the container's `cuda`
  step runs the `cpu` tests too, so the whole CPU path runs there as design section 7 asks (the retired
  `fluidity_add_shared_test` is not used). Device tests: `fluidity_add_test(... LABEL gpu)`. A known-open test
  (form amended 2026-11-14, Foundation as built: `fluidity_add_test` takes one label and discovers every test of
  its program under it, so a failing test there would fail the `cpu` gate) is written with GoogleTest's `DISABLED_`
  prefix, so discovery registers it as disabled and the gate skips it, and is registered a second time, inside
  `if(BUILD_TESTING)`, by `add_test(NAME <Suite>.<Name> COMMAND <program> --gtest_filter=<Suite>.DISABLED_<Name>
  --gtest_also_run_disabled_tests)` with `set_tests_properties(<Suite>.<Name> PROPERTIES WILL_FAIL TRUE LABELS
  known_open)` (`add_test` makes no C++ target, so the helper rule holds; the container's `cpu|no_gpu` run skips
  it), and a line in `docs/known-open-physics.md`.
- **Inputs from later modules are hand-built in tests:** ingredient viscosities (materials), solid motion and
  `SolidCoverage` (solids), `ViscositySceneSummary` (step). No later module is stubbed.
- **Code comments** number long bodies `// Step N:` with the numbers 1 to 12 of design section 5, which are
  the step numbers of the paper's two procedures.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed:

- Foundation with the `base/` additions (`run_on.h` shared with `RunOn`; `check_can_run_on.h`; `scalar_math.h`,
  `tree_sum.h`, `buffer.h`, `ordered_sums.h`, `stable_order.h`, `memory_count.h`), the method paper with
  `figure_style.typ` and `pseudocode.typ`, `docs/glossary.md`, `docs/gpu-runs.md`, `docs/known-open-physics.md`,
  the gate.
- **Amended 2026-11-14 (Foundation as built):** Foundation as built also provides `base/run_on_call.h`
  (`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, `SumBuffers` taken by the sums
  (`ordered_sums.h`), `OrderBuffers` taken by the order, prefix sum and grouping (`stable_order.h`; one form each),
  `base/test_support.h` (library `fluidity_base_test_support`) and `fluidity_add_test_library`; read `AGENTS.md`
  "Structure" (two halves, private GPU headers, working buffers) before the first task. The gate's expected output is seven
  `PASS` lines: format, rules, cpu debug, cpu release, tidy, paper, cuda.
- Contact (`contact::Solid`, `DomainBox`, `SolidVelocitySample`, `ComputeSolidVelocitySample`, `ToSolid`), grid (3.1 to 3.7, including `FindCellsAroundEdge`, `FindNearbyFaces`,
  `FindFacePieces`, `ToCellBlockIndex`), particles, transfer (`CellsFromParticles`, `BuildCellsFromParticles`).
- Pressure, all 62 tasks of its plan (A15: including `solve_batches.h` and the batched layout, Tasks 54 to 61; A24:
  the pieces accept coupled equations and have the sealed-gas overloads and `ComputeBlockingMaterialVelocities`): `GeometrySnapshot` with `CellEighthGeometry` and `FacePieceCentre`,
  `BuildPressureEquations`, `flow_imbalance.h` and `coupled_solve_pieces.h` (`ComputeFaceInertia`,
  `ComputePressurePush` both forms, `GivenTerms`, `ComputeFlowImbalance`, `ComputeVolumeError`,
  `ComputeOwnWeight`, `ComputeStartPressure`, `ApplyZeroMeanToSealedBlobs`, `ApplyPressureOffset`,
  `CheckFlowImbalance`), `SolvePressure`, and the test-only library `fluidity_pressure_test_scenes`
  (`BuildBoxGrid`, `BuildTwoLevelGrid`, `BuildSurfaceFromFunction`, `BuildSolidFromFunction`,
  `BuildEquationSceneArrays`, `BuildHydrostaticColumn`).

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line, and
`grep -n "ComputeOwnWeight\|FindCellsAroundEdge\|ComputeSolidVelocitySample" native/fluidity/*/*.h` finds all
three declarations.

## Review Focus

1. **An edge whose lowest-`CellId` cell holds no liquid.** The sample must still be built exactly once, by
   that cell. Pinned: `BuildViscositySamples, EdgeLowestCellIsAir_BuiltOnce` in Task 10.
2. **A face inside a solid when the grid extends below the floor.** The derivative must use the solid surface
   at its true distance (half a cell), not the face one cell away. Pinned:
   `BuildViscositySamples, GridBelowFloor_UsesTrueDistanceToSolid` in Task 10.
3. **Given faces (`face_inertia == 0`) inside a sample.** Their velocity times coefficient moves into
   `rate_from_solids` and the stored coefficient becomes zero, or the operator loses symmetry. Pinned:
   `BuildViscositySamples, GivenFace_FoldedIntoRateFromSolids` in Task 11.
4. **A given term leaking into a search direction** (reference pressure, jump, solid volume rate, carried
   pressure, base momentum; A24: a material face's `A u_before - coverage_volume_rate`, a blob's heat, boundary
   rate and carried value). The measured response would then not be linear. Pinned:
   `SolveViscosityAndPressure, ResponseIsLinearInDirection` in Task 15 and
   `SolveViscosityAndPressure, ResponseIsLinearWithCouplingTerms` in Task 15a.
5. **Outputs touched by a failure in the middle of the loop.** Pinned:
   `SolveViscosityAndPressure, CancelInThirdOuterIteration_OutputsUntouched` in Task 16.

---

### Task 1: Method paper, samples, solids and the liquid surface

**Kind:** mechanical

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/viscosity.typ` (included from `docs/method-paper/fluidity-method.typ`
  after the pressure section), `docs/method-paper/figures/viscosity_samples.typ`,
  `docs/method-paper/figures/viscosity_size_change.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include; notation rows)

**Interfaces:**
- Produces labels `<sec-viscosity>`, `<eq:deformation-rate>`, `<eq:viscosity-equations>`,
  `<sec-viscosity-samples>`, `<sec-viscosity-solids>`, `<sec-viscosity-surface>`, `<fig-viscosity-samples>`,
  `<fig-viscosity-size-change>`, `<proc-build-viscosity-samples>`.

- [ ] **Step 1: Text**, design section 8 outline items 1 to 5, in the vocabulary of design section 9: why drag
  needs the whole deformation rate (a rigid rotation must lose nothing; three separate smoothings would damp
  it); `<eq:deformation-rate>`; `<eq:viscosity-equations>` exactly as design section 5 (both lines), with units
  of each term; `W = 2 mu V` and `4 mu V` and why shear counts twice; power lost `sum W (E u + g)^2 >= 0`;
  where samples sit (stretch at the centre, edge shear at same-size edges, fitted shear at a centre); the
  quarter tiling (every cell's liquid quarters are counted once per axis pair); the stretch divisor
  `open_volume` and weight on `filled_volume` (decision 5); the edge rules (a) solid between, at its true
  distance, (b) liquid surface beyond (Task 10 states the exact rule; copy it); the fit through four
  independent positions; the free surface needs no extra equation; harmonic mean between viscosities. Cite
  [8], [9], [21] from `references.typ`. Notation rows: `u`, `M`, `E`, `W`, `g`, `D`, `C`, `p`, `mu`, `V`, `dt`.
- [ ] **Step 2: Figures** with `figure_style.typ`: `<fig-viscosity-samples>` (one cell in the XZ plane, its
  stretch samples at the centre, edge samples at the four corners, the four faces each reads, the liquid
  quarter each weighs, a surface line cutting the cell); `<fig-viscosity-size-change>` (a big cell beside two
  small cells: which quarters keep edge samples, which join a fitted sample, the four positions the fit picks
  and the one it rejects as not independent). Each caption states the one thing to see.
- [ ] **Step 3: Pseudocode** `<proc-build-viscosity-samples>`: steps 1 to 5 of design section 5, numbered so;
  inputs, outputs, "fails when" (`kTooFewNearbyFaces`, `kLiquidAtEdgeNotConnected`, `kNoFacePairForShear`,
  `kNotFinite`, `kBuffersTooSmall`).
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.
  Expected: exit 0, no warning. Look at every new page; fix overlaps and clipped labels. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Describe how thick liquid is measured on the grid`.

### Task 2: Method paper, the two loops and the exclusions

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** one paragraph on the batched form (Step 3b).

**Amended 2026-11-14 (A24):** the exclusion table has four rows (no snow row); Step 3c states the coupled terms.

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/viscosity.typ`
- Create: `docs/method-paper/figures/viscosity_two_passes.typ`

**Interfaces:**
- Produces labels `<sec-viscosity-solve>`, `<sec-viscosity-exclusions>`, `<fig-viscosity-two-passes>`,
  `<proc-solve-viscosity-and-pressure>`.

- [ ] **Step 1: Text**, outline items 6 and 7: the inner solve (conjugate gradients, diagonal scaling, the
  stored coefficient `sqrt(dt W) x coefficient` that makes the operator symmetric by construction, the true
  error recheck); the outer loop (pressure diagonal with `connection_scale = face_inertia / velocity_diagonal`,
  measured responses kept and made orthogonal, why the response is measured and not assumed, why given terms
  are left out of a direction); the one acceptance rule shared with `SolvePressure` (cite [9]); the four
  exclusions as a table with the test that lifts each, stating the state at this commit (all four present;
  Tasks 21 to 23 update the table when they lift three; the free solid stays).
- [ ] **Step 2: Figure** `<fig-viscosity-two-passes>`: the sample pass and the face pass as two gathers, the
  fixed order marked by numbers on the arrows.
- [ ] **Step 3: Pseudocode** `<proc-solve-viscosity-and-pressure>`: steps 6 to 12, numbered so; "fails when"
  lists `kVelocitySolveDidNotConverge`, `kPressureSolveDidNotConverge`, `kNotFinite`, `kCancelled`.
- [ ] **Step 3b (A15):** one paragraph after the pseudocode: the same solve in batches (working buffers in CPU memory,
  two sweeps per inner iteration, the outer step's sweeps, why it is possible and slow for a syrup scene); refer to
  pressure's `<sec-solve-in-batches>`.
- [ ] **Step 3c (A24):** label `<sec-viscosity-coupled>`, one page: the material face eliminated into the volume
  balance (`g_k`, unscaled diagonal), the sealed gas blob kept as one more unknown and why (no two-point weight
  through the viscous velocity solve), the extended operator `dt D K^-1 D^T + C + L` and why it is symmetric
  positive semidefinite, the slide at a liquid-material face and the stretch samples reading the material at its
  velocity before pressure (cite Zarifi and Batty 2017 and Fang et al. 2020 as the shared-velocity form not taken;
  URLs in `docs/superpowers/specs/modules/literature-coupling.md`); refer to pressure's `<sec-coupled-solve>`.
- [ ] **Step 4: Run** typst, look at the pages, run the gate. **Commit.** Subject:
  `Describe the joint solve of drag and pressure`.

### Task 3: Errors, stable names and the module's build lines

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the folder's `CMakeLists.txt` uses only the helpers; tests link
`fluidity_base_test_support`.

**Files:**
- Create: `native/fluidity/viscosity/viscosity_error.h`, `solve_viscosity_and_pressure.h` (with only `ToMessage`
  and `ToName` for now), `viscosity_error_message.cc`, `viscosity_error_test.cc`,
  `native/fluidity/viscosity/CMakeLists.txt` (headers by `fluidity_add_headers`, `.cc` by
  `target_sources(fluidity PRIVATE ...)`, `.cu` by `fluidity_add_gpu_sources`, tests by
  `fluidity_add_test(... LABEL cpu LINKS fluidity_base_test_support)`; every later task adds its files here, so no
  source is left out of the build)
- Modify: the top-level `add_subdirectory` list (as `native/fluidity/pressure/` is added); `docs/glossary.md`

**Interfaces:**
- **Amended 2026-11-14 (A24):** 15 values; `kSnowNotSupported` is never added. **Amended 2026-11-14 (A27):** 14
  values; `kFreeSolidNotSupported` is never added either. **Amended 2026-11-14 (reconcile item 25):** 15 values
  with `kOutOfMemory`.
- **Amended 2026-11-15 (reconcile item 31):** `kOutOfMemory` is the last value (after `kSizeChangeNotSupported`);
  `kCancelled` and the three exclusions move up by one (accepted); the `ToName` and message tables follow enum order.
- Produces: design 3 `ViscosityError` (all 15 values, in order), `ViscosityFailure`, `ToMessage(ViscosityError)`,
  `ToName(ViscosityError)`, exactly. `solve_viscosity_and_pressure.h` includes `base/cpu_only.h` first.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ToName, EveryError_IsLowerSnakeCaseWithoutK)  // design 7 "Stable codes": all 15 values;
      // kMovingSolidNotSupported -> "moving_solid_not_supported", kRunOnCallFailed -> "run_on_call_failed",
      // kVelocitySolveDidNotConverge -> "velocity_solve_did_not_converge"; all distinct
  TEST(ViscosityError, OutOfMemory_IsTheLastValue)   // amended 2026-11-15 (reconcile item 31):
      // static_cast<int>(kOutOfMemory) == 14 and kSizeChangeNotSupported == 13; the ToName table is in enum order
  TEST(ToMessage, EveryError_IsDistinctAndNotEmpty)
  TEST(ViscosityFailure, Default_HasNoFaceAndNoCell)  // error kNone, cell.value 0, face.value grid::kNoIndex
  ```
- [ ] **Step 2: Run.** Expected: compile error. **Step 3: Implement.** Messages say what the user can do
  ("The velocity solve used all its iterations. Shorten the step."), from the comments of section 3; each
  exclusion message names the suggested change (stop the driven solid cutting cells, align it with the axes, use
  one cell size there).
- [ ] **Step 4: Glossary rows** (`| Say | Plain meaning | Never say |`): `deformation_rate` (how fast the liquid
  is stretched and sheared, 1/s; never "strain rate"), `stretch`, `shear`, `DeformationRateSample` (never
  "strain row", "row"), `sample_weight` (never "quadrature weight"), `rate_from_solids` (never "prescribed
  strain"), `viscosity` (Pa s), `surface force per area` (never "traction"), `liquid quarter` (the two cell
  eighths of a cell beside one edge), `given face` (a face piece with `face_inertia == 0`: its velocity is
  known, not solved), `power_lost_to_viscosity`.
- [ ] **Step 5: Run** the gate. **Commit.** Subject: `Add the viscosity module's errors`. New names:
  `ViscosityError` and its 15 values, `ViscosityFailure`, `ToMessage`, `ToName`.

### Task 4: Sample types and weight formulas

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test is `fluidity_add_test(... LABEL cpu)`, not the retired
`fluidity_add_shared_test`.

**Amended 2026-11-15 (A57):** `kSmallestVisibleViscousEffectPerStep` and its test are removed (and the comment line
about it); `ComputeViscousEffectPerStep` stays. The advice moves to Task 4a.

**Files:**
- Create: `native/fluidity/viscosity/deformation_rate.h`, `deformation_rate_test.cc`

**Interfaces:**
- Consumes: `grid::FacePlaceIndex`, `grid::CellId`, `Axis`, `Vec3`, `FLUIDITY_CPU_AND_GPU`, `math::IsFinite`.
- Produces: design 3 `kMaxFacesPerSample`, `kFitReach`, `kFitIndependence`, `SampleKind`, `FaceCoefficient`,
  `DeformationRateSample`, `ComputeStretchWeight`, `ComputeShearWeight`, `ComputeMixtureViscosity`,
  `NearbyFacePosition`, `FittedChange` exactly (`ComputeFittedChange` in Task 5).
  **Amended 2026-11-14 (A25):** also
  `ComputeViscousEffectPerStep(double viscosity, double step_duration, double density, double cell_width)` =
  `viscosity * step_duration / (density * cell_width * cell_width)` (no unit: Pa s / (kg/m3) = m2/s, times s, over
  m2), the number the step's planner prints per liquid (`plan-decisions.md` A25). Comment: one implicit step
  multiplies the velocity difference between two neighbouring cells by `1 / (1 + 4 N)`. (Removed 2026-11-15, A57:
  the constant `kSmallestVisibleViscousEffectPerStep = 2.5e-4` and "below the constant that change is under 0.1 %
  per step".)
  **Plan decisions:** `ComputeShearWeight` returns `4 x V^2 / sum over parts with V_q > 0 of (V_q / mu_q)`,
  `V = sum V_q` (`viscosity.cc:115-148`); 0 when `V == 0` or any part with `V_q > 0` has `mu_q == 0`; parts
  summed in index order 0..3. `ComputeMixtureViscosity` returns `sum(volume_i x viscosity_i) / sum(volume_i)`,
  both sums in ingredient order, 0 when the volume sum is 0. `static_assert(sizeof(DeformationRateSample) ==
  40)` and `sizeof(FaceCoefficient) == 16` (if a compiler pads differently the memory estimate entries use `sizeof`).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeStretchWeight, IsTwiceViscosityTimesVolume)      // (10, 1e-6) -> 2e-5 exactly
  TEST(ComputeShearWeight, EqualQuarters_IsFourMuV)            // V_q 0.25e-6 each, mu 3: 4 * 3 * 1e-6, 1e-15 rel
  TEST(ComputeShearWeight, TwoViscosities_IsHarmonicMean)      // V_q all 1, mu {1,1,1000,1000}:
                                                               // 4 * 16 / (2 + 0.002) = 31.968031968..., 1e-15 rel
  TEST(ComputeShearWeight, AirQuarters_AreLeftOut)             // V_q {1, 1, 0, 0}, mu {2, 2, 0, 0}: 4*4/1 = 16
  TEST(ComputeShearWeight, WetPartWithoutViscosity_IsZero)     // V_q {1,1,1,1}, mu {1,1,1,0}: 0
  TEST(ComputeMixtureViscosity, SyrupAndWater_IsVolumeWeighted)  // volumes {3e-6, 1e-6}, viscosities {10, 1e-3}:
                                                               // 7.50025, 1e-15 rel
  TEST(ComputeMixtureViscosity, NoVolume_IsZero)
  TEST(DeformationRateSample, Size_Is40Bytes)
  TEST(ComputeViscousEffectPerStep, MatchesDefinition)         // A25: (1e-3, 0.01, 1000, 0.01) -> 1e-4;
                                                               // (10, 0.01, 1400, 0.01) -> 0.714285714...; 1e-15 rel
  // (Removed 2026-11-15, A57: TEST(kSmallestVisibleViscousEffectPerStep, IsTenthOfAPercentPerStep).)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (port `viscosity.cc:92-148` without `FactorProduct`,
  `ScaledProduct`, `StrainWeight3D`; design section 10). File header `Read:` cites `<sec-viscosity-samples>`.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the deformation-rate sample and its weights`.

### Task 4a: The viscosity advice numbers

**Amended 2026-11-15 (A57):** new task; replaces A25's threshold (`plan-decisions.md` "Owner answers A40-A64 applied"
item 4; design 3 `viscosity_advice.h`, 5.8, decision 24).

**Amended 2026-11-15 (A65):** the paper's assumption reads "uniform particles of equal mass within a cell (eight per cell, sized to it); `h` the smallest cell width", not one particle size everywhere.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/viscosity/viscosity_advice.h`, `viscosity_advice.cc`, `viscosity_advice_test.cc`
- Modify: `native/fluidity/viscosity/CMakeLists.txt`, `docs/method-paper/sections/viscosity.typ`, `docs/glossary.md`

**Interfaces:**
- Consumes: `ComputeViscousEffectPerStep` (Task 4). Produces design 3 `viscosity_advice.h` exactly:
  `kNegligibleViscosityRatio = 0.1`, `kSmallestVisibleWavelengthInCells = 4`, `kNegligibleViscousChange = 0.01`,
  `std::optional<double> ComputeViscosityToSmoothingRatio(double viscous_effect_per_step, double own_motion_kept)`
  (`4 N / (1 - alpha)`, empty when `alpha == 1`), `double ComputeFourCellFeatureChange(double viscosity, double
  density, double run_duration, double cell_width)` (`-std::expm1(-kappa^2 nu T)`, `kappa = 2 pi /
  (kSmallestVisibleWavelengthInCells h)`), `double ComputeReynoldsNumber(double viscosity, double density, double
  smallest_domain_extent, double gravity_size)` (`sqrt(2 g L) L / nu`). CPU only (it includes `<optional>`).
- **Plan decisions:** preconditions (`Requires:`, debug assertion; A30 style): `N >= 0`, `0 <= alpha <= 1`,
  `density > 0`, `cell_width > 0`, `run_duration >= 0`, `smallest_domain_extent > 0`, `gravity_size >= 0`, viscosity
  `> 0` for the Reynolds number (it is called only for a viscous liquid). The step's planner (step's `run_plan.h`,
  `LiquidViscousEffect`) owns the advice rule `ratio < kNegligibleViscosityRatio || change < kNegligibleViscousChange`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ViscosityAdvice, Constants_AsDecided)                       // 0.1, 4, 0.01; exact
  TEST(ComputeViscosityToSmoothingRatio, WaterAtOneCentimetre_Is0p008)  // N 1e-4 (water 1e-3 Pa s, 1000 kg/m3,
      // dt 0.01 s, h 0.01 m), alpha 0.95: 0.008 within 1e-12 relative
  TEST(ComputeViscosityToSmoothingRatio, KeepsAllOwnMotion_IsEmpty)     // N 1e-4, alpha 1: has_value() false
  TEST(ComputeFourCellFeatureChange, Water_Is0p2187)               // (1e-3, 1000, 10, 0.01): D 0.1; expected
      // -std::expm1(-pi^2 * 0.1 / 4) = 0.2186563... written in the test; 1e-12 relative
  TEST(ComputeFourCellFeatureChange, ThresholdAtDOf4p073e-3)       // D = 4 * std::log(1 / 0.99) / pi^2
      // (4.0732477e-3) computed in the test; viscosity = 1000 * 0.01^2 * D / 1, density 1000, T 1 s, h 0.01 m:
      // 0.01 within 1e-12 relative
  TEST(ComputeReynoldsNumber, WaterInOneMetreBox_Is4p43e6)         // (1e-3, 1000, 1, 9.81): sqrt(19.62) / 1e-6 =
      // 4.4294469e6; 1e-12 relative
  TEST(ComputeViscosityToSmoothingRatio, SyrupAtOneCentimetre_Is57p1)   // N 0.714285714... (10 Pa s, 1400 kg/m3,
      // dt 0.01, h 0.01), alpha 0.95: 4 N / 0.05 = 57.142857...; 1e-12 relative (I/O's syrup line)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement**; each function's header comment carries its example.
- [ ] **Step 4: Paper.** In `viscosity.typ`, label `<sec-viscosity-advice>` (design 8 item 8): the derivation of design
  5.8 written out (`sinc^3` per quadratic B-spline, `sinc^6(k h) ~ 1 - (kappa h)^2 / 4`, one diffusion step `1 - nu
  kappa^2 dt`, hence `nu_num = (1 - alpha) h^2 / (4 dt)` and `r = 4 N / (1 - alpha)`), with the sentence "The
  smoothing figure `nu_num = (1 - alpha) h^2 / (4 dt)` is this project's derivation from the filter analysis of
  Hammerquist and Nairn (2017), not a published value"; its assumptions (uniform particles of equal mass within a
  cell, eight per cell sized to it (A65), `h` the smallest cell width); why the
  largest step makes `r` a lower bound; the four-cell decay over the run and the threshold `D = 4.0732477e-3`; the
  Reynolds number; that viscosity 0 also switches the liquid from sticking to slipping. Cite Hammerquist and Nairn
  2017 (URL in `docs/superpowers/specs/modules/literature-practice.md` topic 1) and Bridson's course notes. Run
  typst; look at the page.
- [ ] **Step 5: Glossary rows:** "viscosity to transfer smoothing ratio", "four-cell feature change", "Reynolds
  number". **Run** the gate. **Commit.** Subject: `Add the numbers that say when viscosity is not visible`. New names:
  the six of `viscosity_advice.h` (reason "A57").

### Task 5: The fit through four nearby positions

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Modify: `native/fluidity/viscosity/deformation_rate.h`, `deformation_rate_test.cc`

**Interfaces:**
- Produces: `ComputeFittedChange` exactly. **Plan decisions** on the contract the design left open:
  - Candidates: positions whose every offset component has magnitude `<= kFitReach`. Order: squared offset
    length, ties by lower `face.value`, a position with `face == grid::kNoIndex` before faces at equal length.
  - Independence, greedy in that order: the first is taken; each next is taken when its distance (cell widths)
    from the affine span of those taken (point, line, plane) exceeds `kFitIndependence`; stop at four.
    Fewer than four: `kTooFewNearbyFaces` (`liquid_adaptive_step_3d.cc:125-150`).
  - The fit: `value = a + gradient . offset` through the four, solved as a 4x4 system by Gaussian elimination
    with partial pivoting in fixed index order; `per_axis[k][j]` is the coefficient (1/m, offsets times
    `cell_width`) of position `j` in the derivative along axis `k`, and `per_axis[k][j].face` is that
    position's face (or `kNoIndex`). A pivot below `1e-12` of the largest: `kTooFewNearbyFaces`.
  - Any non-finite input or result: `kNotFinite`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeFittedChange, LinearField_ReproducedExactly)      // 12 positions from a fixed integer formula,
      // values 0.3 + 1.7 x - 0.4 y + 2.2 z (offsets in widths, cell_width 0.01): sum_j coefficient * value
      // == 170, -40, 220 per axis within 1e-12 relative
  TEST(ComputeFittedChange, CollinearPositions_AreSkipped)       // positions 0, 1, 2 on one line, then 3 off it:
      // chosen faces are {0, 1, 3, next independent}
  TEST(ComputeFittedChange, EqualDistances_LowerFaceFirst)
  TEST(ComputeFittedChange, KnownPointBeforeFaceAtEqualDistance)
  TEST(ComputeFittedChange, BeyondReach_IsIgnored)               // a position at offset (1.6, 0, 0) never chosen
  TEST(ComputeFittedChange, ThreeIndependentOnly_IsTooFewNearbyFaces)  // 9 positions in one plane
  TEST(ComputeFittedChange, NaNOffset_IsNotFinite)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** in fixed-size arrays; port the idea of
  `git show main:native/src/liquid_adaptive_step_3d.cc` lines 98-173 without the whole-grid scan and the
  allocation (design section 10).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Fit the local change of a face velocity`.

### Task 6: The capability check

**Amended 2026-11-14 (A24):** four exclusions; no snow exclusion (snow's inertia enters the joint solve through its
material faces, Task 15a); no springy-solid exclusion (the owner chose option C); the open note is removed.

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure.h`
- Create: `viscosity_capability.cc`, `viscosity_capability_test.cc`

**Interfaces:**
- Produces: `ViscositySceneSummary`, `CheckViscosityCanRun` exactly. Order of checks: the order of
  `ViscosityError` (moving solid, inclined solid, size change). A summary with
  `has_viscous_ingredient == false` always passes.
- `ViscositySceneSummary` has no `has_snow` (A24) and no `has_free_solid_touching_liquid` (**Amended 2026-11-14
  (A27)**: the free solid's six velocities are unknowns of the joint solve, Task 15c). Three exclusions, all
  geometric, each with its lifting task (21 to 23).

- [ ] **Step 1: Failing tests** (design 7 "Each exclusion"):

  ```cpp
  TEST(CheckViscosityCanRun, NoViscousIngredient_CanRunWhateverElse)   // every other flag true: has_value()
  TEST(CheckViscosityCanRun, ViscousOnly_CanRun)
  TEST(CheckViscosityCanRun, EachExclusion_ReturnsItsError)            // three summaries, one flag each: the three
                                                                       // errors in order; cell 0, face kNoIndex
  TEST(CheckViscosityCanRun, TwoExclusions_ReturnsTheFirst)            // moving and inclined: kMovingSolidNotSupported
  ```
- [ ] **Step 2: Run** (fail), **Step 3: Implement** one line per exclusion, each with a comment naming the test
  that lifts it (design 5.7). **Step 4: Run** the gate. **Commit.** Subject:
  `Reject the viscous scenes not supported yet`. New names: `ViscositySceneSummary`, `CheckViscosityCanRun`.

### Task 7: Buffers and memory estimate entries

**Amended 2026-11-14 (A30|A31|reconcile):** working buffers are `sum_buffers` / `order_buffers` (test `BuffersRows_MatchBuffersByteCounts`); out of memory is the new `ViscosityError::kOutOfMemory`, returned only by `BuildViscosityBuffers` (reconcile item 25); `ToViscosityFailure` maps no processor mismatch (A30).

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** `ViscosityBuffers` owns a `SumBuffers` and an `OrderBuffers`, with
four memory estimate entries, and holds the one mapping from `BufferError` to `ViscosityFailure`.

**Amended 2026-11-15 (A58):** `BuildViscosityBuffers(const ViscosityBufferSettings& settings, RunMemory& run_memory,
RunMemory& cpu_run_memory)`: every buffer is a piece of a `RunMemory` (`BuildBuffer(run_on, size, run_memory)`;
`BuildSumBuffers` / `BuildOrderBuffers` in their `RunMemory&` forms); one estimate entry per array; `kOutOfMemory`
also from `kRunMemoryTooSmall`; estimate test `IntoRunMemory_HandsOutExactlyTheEstimate`. **Amended 2026-11-15
(A60):** the A27 row `"viscosity: sample solid coefficients"` is renamed `"viscosity: solid sample coefficients"`
(52 B per solid sample; filled for a solid of any motion kind in Task 15e).

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure.h`
- Create: `viscosity_buffers.cc`, `viscosity_buffers_access.h` (private), `viscosity_memory_estimate.cc`,
  `viscosity_buffers_test.cc`, `viscosity_buffers_no_gpu_test.cc`

**Interfaces:**
- Produces: `ViscosityBufferSettings`, `ComputeViscosityBufferBytes`, `ViscosityBuffers`,
  `BuildViscosityBuffers`, `BuildViscosityMemoryEstimate`, `ViscositySamplesSnapshot`, `ReadViscositySamples`
  (returns empty vectors until Task 9 fills samples).
- **Amended 2026-11-14 (A24):** `ViscosityBufferSettings::max_sealed_gas_blobs` (`std::uint32_t`, 0 without gas)
  and the row `"viscosity: sealed gas blob vectors"` (given count `max_sealed_gas_blobs`, `70 x 8` bytes: pressure,
  imbalance, diagonal, direction, response, working values and the `2 x 32` kept entries; always `kRunOn`, also in the
  batched layout); test `ComputeViscosityBufferBytes, SealedGasBlobRow` (4 blobs add exactly 2,240 bytes).
  **Amended 2026-11-14 (A26, A27):** `max_material_faces` and `max_free_solids` (both `std::uint32_t`, 0 by
  default) and the rows `"viscosity: material face vectors"` (given `max_material_faces`, `12 x 8 + 16` bytes, streamed
  like face rows in the batched layout), `"viscosity: free solid vectors and blocks"` (given `max_free_solids`,
  `(12 x 6 + 2 x 36) x 8` bytes, always `kRunOn`), `"viscosity: solid sample coefficients"` (renamed 2026-11-15,
  A60; given `max_sample_count`, `4 + 6 x 8` = 52 bytes); test `ComputeViscosityBufferBytes, MaterialAndSolidRows`
  (100 faces and 2 solids add exactly 11,200 + 2,304 bytes, plus the sample row). **Amended 2026-11-15 (A58):** with
  one entry per array these rows become `"viscosity: material face vectors"` (copies 12, 8 B) and `"viscosity:
  material face entries"` (16 B), `"viscosity: free solid vectors"` (copies 12, 6 x 8 B) and `"viscosity: free solid
  blocks"` (copies 2, 36 x 8 B), the sealed gas row copies 70, 8 B; the byte totals of the test are unchanged before
  rounding.
- **Plan decisions:**
  - `max_cell_count` is a positive multiple of 64 and `max_face_count == max_cell_count / 64 x 240`, else
    `kBadInput`; every value array is a `grid::CellValues` / `FaceValues` built with
    `max_cell_blocks = max_cell_count / 64` and bound with `grid::ApplyGridVersion` at each build.
    `max_sample_count <= 15 x max_cell_count` and `max_coefficient_count <= 8 x max_sample_count`, else
    `kBadInput`. Out of memory: `kOutOfMemory`, magnitude = bytes asked, returned only by `BuildViscosityBuffers`
    (reconcile item 25; was `kRunOnCallFailed`).
  - Rows (names prefixed `"viscosity: "`, `KeptFor::kOneStepTry`, `copies = 1`, `MemoryWhere::kRunOn`, bytes
    from `sizeof` with a `static_assert` each):

    | Row | Counted per | Bytes each |
    |---|---|---|
    | cell viscosity, cell order key and order, samples per cell, first sample of cell | cell place | 8 + 8 + 4 + 4 + 4 |
    | samples | given count `max_sample_count` | `sizeof(DeformationRateSample)` |
    | sample rate, `sqrt(dt x sample_weight)`, block of sample, samples of block order | `max_sample_count` | 8 + 8 + 4 + 4 |
    | samples of block starts | cell block + 1 | 4 |
    | coefficients by sample; face of coefficient, sample of coefficient, use order | `max_coefficient_count` | 16; 4 + 4 + 4 |
    | first use of face | face place + 1 | 4 |
    | face arrays: `face_inertia`, `velocity_diagonal`, `connection_scale`, `base_momentum`, push, momentum, candidate velocity, search velocity, error, scaled error, direction, operator output | face place | 12 x 8 |
    | cell arrays: pressure, imbalance, pressure diagonal, pressure direction, response, power partial working values, plus `2 x 32` kept directions and responses | cell place | 70 x 8 |
    | block partials of power lost | cell block | 8 |
    | progress record (GPU: two imbalance norms, inner iteration count, error) | given count 1 | 64 |

    **Amended 2026-11-15 (A58):** each row of the table that names several arrays is written as one entry per array
    (or one entry with `copies` = the number of arrays of one element size: face arrays `copies 12`, 8 B; cell arrays
    `copies 70`, 8 B; sample rate and `sqrt(dt x sample_weight)` `copies 2`, 8 B; block of sample and samples of block
    order `copies 2`, 4 B; and so on), because `BuildBuffer(run_on, size, run_memory)` rounds every array to
    `kRunMemoryPieceAlignmentBytes` (256) and `ComputeRunMemoryBytes` rounds per entry and copy. Names keep the
    `"viscosity: "` prefix and the array's name. No `kRunOn` entry is `kOneCall`.

  - `ViscosityBuffers` keeps `RunOn`, the bound `GridVersion`, the built sample and coefficient counts and the
    `step_duration` of the last build; internals reach later tasks through `viscosity_buffers_access.h`.
  - **Amended 2026-11-14 (Foundation as built), working buffers.** `ViscosityBuffers` holds `SumBuffers sum_buffers` and
    `OrderBuffers order_buffers` (private members, reached through `viscosity_buffers_access.h`), built once by
    `BuildViscosityBuffers` on `settings.run_on` with `BuildSumBuffers` and `BuildOrderBuffers`; no later call
    allocates. Their `max_count` (one piece; Task 26 gives the batched values): sum buffers `max_face_count` (the
    largest count passed to `ComputeLargestMagnitude`; the block partials given to `ComputeTreeSum` are fewer);
    order buffers `max(max_cell_count + 1, max_coefficient_count)` (cells ordered by `BuildStableOrder`, cell counts
    summed by `ComputeExclusivePrefixSum`, coefficients grouped by `BuildGroupsByDestination`). Four more rows, all
    `kGivenCount`, `kWholeRun`, `kRunOn`: `"viscosity: sum buffers"` (given count the sum buffers `max_count`,
    `ComputeSumBuffersSize(run_on).bytes_per_item` bytes), `"viscosity: sum buffers fixed bytes"` (given count
    `fixed_bytes`, 1 byte), and `"viscosity: order buffers"`, `"viscosity: order buffers fixed bytes"` likewise from
    `ComputeOrderBuffersSize(run_on)`. The settings already carry `run_on` and every count, so no signature changes.
  - **Amended 2026-11-14 (Foundation as built), errors.** One private mapping in `viscosity_buffers_access.h`:
    `ViscosityFailure ToViscosityFailure(BufferError error, std::uint64_t bytes_asked);` (`kOutOfMemory` ->
    `kOutOfMemory` with `magnitude` the bytes asked, as above, reachable only from `BuildViscosityBuffers`;
    `kSizeMismatch` -> `kBuffersTooSmall`; `kRunOnCallFailed` -> `kRunOnCallFailed`). **Amended 2026-11-14
    (A30|reconcile):** no processor-mismatch value is mapped (buffers on one processor are a precondition, A30),
    and out of memory is `kOutOfMemory` (reconcile item 25). Every GPU half's `RunOnCallStatus`
    goes through `ToBufferError`, then this function. **Amended 2026-11-15 (A58):** `kRunMemoryTooSmall` ->
    `kOutOfMemory` with `magnitude` the bytes of the piece asked (a planner bug: debug builds also assert in
    `BuildBuffer`).
  - **Amended 2026-11-15 (A58), which block.** `kRunOn` entries are pieces of `run_memory`, `kCpu` entries (the
    batched layout, Task 26) pieces of `cpu_run_memory`; in a CPU run the step passes the same object twice.
    `Requires:` `settings.run_on == run_memory.run_on()` and `cpu_run_memory.run_on() == RunOn::kCpu` (debug
    assertion, A30). Plain `BuildBuffer` is not used by the module (tests that need a raw buffer use
    `BuildBufferHolding`).

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ComputeViscosityBufferBytes, EqualsSumOfMemoryEstimate)
  TEST(BuildViscosityMemoryEstimate, BuffersRows_MatchBuffersByteCounts)  // amended (Foundation as built): on the
      // CPU, the four sum and order buffers rows equal sum_buffers.byte_count() and order_buffers.byte_count() read through
      // viscosity_buffers_access.h; exact
  TEST(ComputeViscosityBufferBytes, SyrupSceneSizing_IsUnderOneGigabyte)   // design 6: 524,288 cells, 1,966,080
      // faces, 3,000,000 samples, 13,000,000 coefficients: prints the figure; between 0.5e9 and 1.0e9 bytes
  TEST(BuildViscosityBuffers, CellsNotAMultipleOf64_IsBadInput)
  TEST(BuildViscosityBuffers, OnCpu_Succeeds)
  TEST(ReadViscositySamples, FreshBuffers_AreEmpty)
  // Amended 2026-11-15 (A58):
  TEST(BuildViscosityBuffers, IntoRunMemory_HandsOutExactlyTheEstimate)  // syrup sizing at a tenth (52,416 cells,
      // 196,560 faces, 300,000 samples, 1,300,000 coefficients) on the CPU: BuildRunMemory(kCpu,
      // ComputeRunMemoryBytes(BuildViscosityMemoryEstimate(settings), counts, kRunOn)); passed as both blocks:
      // handed_out_bytes() == reserved_bytes(); exact
  TEST(BuildViscosityBuffers, RunMemoryOneByteShort_IsOutOfMemory)  // the same block 256 bytes smaller: in the
      // release preset kOutOfMemory, magnitude the bytes of the piece that did not fit; in the debug preset
      // BuildBuffer's planner-bug assertion fires (EXPECT_DEBUG_DEATH covers both); exact
  // viscosity_buffers_no_gpu_test.cc, LABEL no_gpu
  TEST(BuildViscosityBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** `BuildViscosityBuffers` allocates by walking the rows, so
  the estimate cannot drift; nothing allocates after it. **Step 4: Run** the gate. **Commit.** Subject:
  `Add the viscosity buffers and their memory estimate entries`. New names as listed.

### Task 8: Test scenes for the viscosity tests

**Amended 2026-11-14 (reconcile):** pressure's test helper is called `BuildEquationSceneArrays` (renamed in
pressure Task 13; also in "Before you start").

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the library is made with `fluidity_add_test_library` and linked by
tests through `LINKS`.

**Files:**
- Create: `native/fluidity/viscosity/viscosity_test_scenes.h`, `viscosity_test_scenes.cc`,
  `viscosity_test_scenes_test.cc` (test-only library `fluidity_viscosity_test_scenes`:
  `fluidity_add_test_library(fluidity_viscosity_test_scenes SOURCES viscosity_test_scenes.cc LINKS
  fluidity_pressure_test_scenes fluidity_base_test_support)`; every viscosity test that uses a scene names it in
  `LINKS`; never linked by `fluidity`)
- Modify: `native/fluidity/viscosity/CMakeLists.txt`

**Interfaces:**
- Produces (**Plan decision**, namespace `fluidity::viscosity::testing`, CPU only):

  ```cpp
  enum class GridLayout : std::uint8_t { kSameSize, kSizeChange };  // kSizeChange: twice-wide cells where stated
  // Owns everything a ViscosityInputs points to: grid, pressure buffers, surface and solid arrays (built with
  // pressure::testing helpers and FindLiquidSurface with has_viscous_ingredient = true), FindAirPockets output,
  // equations (BuildEquationSceneArrays, carried pressure 0), one CellsFromParticles, ingredient viscosities, solids,
  // velocity_before_pressure (each face takes the component of velocity(face position) along its axis).
  class ViscousScene {
   public:
    ViscosityInputs inputs() const;          // block_range = every block; pressure_buffers set (Task 15).
    const grid::StaggeredGrid& grid() const;
    ViscosityBufferSettings buffer_settings() const;  // sized for this grid, 15 samples and 120 coefficients per cell
  };
  struct ViscousSceneSettings {
    GridLayout layout = GridLayout::kSameSize;
    Vec3 origin{}; double cell_width = 0; grid::CellCoord cell_count{};  // in cells of cell_width
    contact::DomainBox size_change_box{};      // kSizeChange: cells whose centre is inside are cell_width wide,
                                               // the others 2 cell_width (pressure::testing::BuildTwoLevelGrid)
    double (*distance_to_surface)(Vec3, const void*) = nullptr;   // m, negative in liquid
    double (*distance_to_solid)(Vec3, const void*) = nullptr;     // m, positive outside; null: no solid
    double (*solid_normal_velocity)(Vec3, Axis, const void*) = nullptr;
    Vec3 (*velocity)(Vec3, const void*) = nullptr;
    double (*ingredient_share)(Vec3, int ingredient, const void*) = nullptr;  // share of filled_volume
    const void* context = nullptr;
    std::vector<double> ingredient_viscosity;  // Pa s
    std::vector<contact::Solid> solids;        // ascending solid_id
    double density = 1;                        // kg/m3
    double step_duration = 0;                  // s
    double surface_tension = 0;                // N/m
    std::vector<double> tracked_gas_pressure;  // empty, or Pa above atmosphere per cell place (holds_tracked_gas
                                               // set where finite): gives an air end its own outside pressure
  };
  std::unique_ptr<ViscousScene> BuildViscousScene(const ViscousSceneSettings& settings);
  ```
  Named fixtures, each a function returning `ViscousSceneSettings` (all Z-up, SI):
  - `ShearChannelSettings(layout, n, viscosity_ratio, top_speed, step_duration, profile)`: the Z-up turn of
    `main`'s Couette fixture (`native/tests/mac_viscous_traction_cases.h:205-330`). Liquid `x in [0, 2/n]`,
    `y in [0, 4/n]`, `z in [0, 1]`, cell width `1/n`; one air cell layer beyond each y side (free surfaces);
    grid z from 0 to `1 + 1/n`. Floor: the domain box bottom; x sides: solid outside `[0, 2/n]` whose
    `solid_normal_velocity` on X faces is `profile(z)` (inlet and outlet: closed, given faces); top: a
    `contact::FlatSolid` at `z = 1`, normal `-z`, moving `(top_speed, 0, 0)`. Ingredient 0 (viscosity 1) for
    `z < 0.5`, ingredient 1 (viscosity `viscosity_ratio`) above. `profile` default: `u_x = s (min(z, 0.5) +
    max(z - 0.5, 0) / ratio)`, `s = 2 / (1 + 1/ratio)` (so `top_speed = 1`). `kSizeChange`: cells above
    `z = 0.75` are `2/n` wide (`n` even).
  - `PlanarExtensionSettings(layout, n, viscosity_ratio, step_duration)`: liquid `x in [0, 1]`, `z in [0, 1]`,
    `y in [0, 4/n]`, one air layer on every side, no solid, no gravity, velocity `(x - 0.5, 0, 0.5 - z)`;
    viscosity 1 for `x < 0.5`, `ratio` above. Outside pressures by tracked gas in the air cells (**Plan
    decision**, because liquid sticks to every solid and `main`'s free-slip walls no longer exist): air beside an
    x side `0`; beside a z side `4 mu(x)`; beside a y side `2 mu(x)` (Pa; `mu(x)` of the touching liquid cell).
    With these the field is an exact steady state with `p = 2 mu(x)`. `kSizeChange`: cells with `z > 0.75` are
    `2/n` wide.
  - `FloatingCubeSettings(layout, velocity)`: liquid cube `[0.2, 0.6]^3`, width 0.05 (two-level: 0.1 outside
    `z < 0.4`), air around, no solid, no gravity, viscosity 2.
  - `SyrupColumnSettings(layout)`: `pressure::testing::BuildHydrostaticColumn`'s box (open top 0.4 x 0.4,
    w 0.1, H 0.43; two-level as pressure Task 36), syrup 10 Pa s, 1400 kg/m3, `dt = 0.01`, Z faces start at
    `-9.81 x dt`.

- [ ] **Step 1: Failing tests:** `BuildViscousScene, ShearChannel_HasGivenInletFaces` (every X face on `x = 0`
  has `face_inertia == 0`), `BuildViscousScene, FloatingCube_EighthsSumToFilledVolume` (1e-15 x w^3),
  `BuildViscousScene, PlanarExtension_AirEndsCarryTrackedGasPressure`.
- [ ] **Step 2: Run** (fail), **Step 3: Implement**, **Step 4: Run** the gate. **Commit.** Subject:
  `Add test scenes for the viscosity module`.

### Task 9: Build on the CPU: cell viscosity, stretch samples, order and check

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** samples are ordered by the cell's block first, so that batches number them as one
piece does.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the order, prefix sum and grouping take
`buffers.order_buffers`; the count and write passes are per-cell functions the GPU will reuse.

**Files:**
- Create: `native/fluidity/viscosity/samples_of_cell.h` (private, shared, C++20 subset),
  `viscosity_samples_cpu.cc`, `viscosity_samples_test.cc`
- Modify: `solve_viscosity_and_pressure.h` (`ViscosityInputs`, `BuildViscositySamples`), `viscosity_buffers.cc`,
  `CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 4, 7, 8; `pressure::ComputeFaceInertia`; `grid::FindFacePieces`, `FindTouchingCells`,
  `ToCellId`; `BuildStableOrder(keys, count, nullptr, order, buffers.order_buffers)`,
  `ComputeExclusivePrefixSum(values, count, sums, buffers.order_buffers)` and
  `BuildGroupsByDestination(destination, item_count, destination_count, first_of_group, item_order,
  buffers.order_buffers)`.
- Produces: `ViscosityInputs` (design 3) **plus one field (Plan decision)**:
  `pressure::PressureBuffers* pressure_buffers = nullptr;` — the buffers that built `pressure_equations`; the
  check, the zero mean and the offset of the joint solve need them (their signatures take `PressureBuffers&`)
  and `SolveViscosityAndPressure`'s signature has no other place for them. Null: `kBadInput` in the solve;
  not read by `BuildViscositySamples`. `BuildViscositySamples` exactly.
  Private `samples_of_cell.h`: `struct CellSampleInputs` (plain pointers on one processor: `grid::GridTables`,
  the snapshot's cell, face, eighth and centre arrays, `face_inertia`, `velocity_before_pressure`,
  `cell_viscosity`, `const contact::Solid* solids`, `solid_count`, `step_duration`) and
  `FLUIDITY_CPU_AND_GPU ViscosityError BuildSamplesOfCell(const CellSampleInputs&, grid::CellPlaceIndex cell,
  DeformationRateSample* samples, FaceCoefficient* coefficients, int* sample_count, int* coefficient_count,
  grid::FacePlaceIndex* failing_face)` (null `samples`: count only).
- **Plan decisions:** `face_inertia` is computed into the buffers by `pressure::ComputeFaceInertia` at the start
  of the build. Step 1 runs over every cell of the range with `filled_volume > 0`; a cell with liquid but zero
  ingredient volume takes the value of the touching cell (`FindTouchingCells`) with ingredient volume and the
  lowest `CellId`, else 0 (one ring). Step 2: one stretch sample per cell with an equation and per axis; its
  coefficients are its face pieces of that axis (`FindFacePieces`, filtered by `ToAxis` and side) with
  `sign x open_area / open_volume`; pieces with `open_area == 0` get no coefficient; `rate_from_solids +=
  sign x solid_volume_rate / open_volume`. Build loop: cells of the range sorted by (block index, `CellId::value`)
  (`BuildStableOrder` on that key), count pass (`BuildSamplesOfCell` with null `samples`, one call per cell),
  `ComputeExclusivePrefixSum` of the counts, write pass (one call per cell at its prefix sum): the same per-cell
  calls the GPU threads make in Task 13. Step 5's finite check scans weights,
  coefficients and `rate_from_solids` once; the first non-finite sample in order gives `kNotFinite` with its
  cell and `magnitude` the value. `ViscosityFailure::cell` is always the stable `CellId`.
  `kBadInput`: a null pointer named in section 3, `cells_from_particles.size() != 1`,
  `ingredient_viscosity.size() != ingredient_count`, a negative or non-finite viscosity, `step_duration` not
  positive and finite, any input or buffer bound to another grid version or block count, a range outside the
  grid. More samples or coefficients than built for: `kBuffersTooSmall`, magnitude = count needed.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(BuildViscositySamples, MaterialSets_CellViscosityIsVolumeWeightedMixture)  // design 7 "Material sets":
      // floating cube with syrup (10 Pa s) and water (1e-3) shares 0.25/0.75 in its lower half: cell_viscosity
      // == 2.50075 there, 10 above; 1e-15 relative
  TEST(BuildViscositySamples, GasMarkerSetBeside_ChangesNoByte)   // same, with a second CellsFromParticles for a
      // gas marker set built and filled but not passed: ReadViscositySamples bytes identical
  TEST(BuildViscositySamples, TwoMaterialSetsPassed_IsBadInput)
  TEST(BuildViscositySamples, LiquidCellWithoutIngredient_TakesLowestIdNeighbour)
  TEST(BuildViscositySamples, StretchSamplesMatchFlowImbalance)   // design 7 "Stretch samples match pressure":
      // floating cube with its top at z = 0.585 (partly filled) and a floor solid at z = 0.215 cutting the
      // bottom layer; random velocity (fixed integer formula): open_volume * (sum of the three stretch rates,
      // rate_from_solids included) == -flow_imbalance of pressure::ComputeFlowImbalance (flow_imbalance is
      // inflow minus outflow), every cell with an equation; 1e-12 relative to the cell's largest term
  TEST(BuildViscositySamples, FullCell_StretchCoefficientIsOneOverWidth)  // +-1/h exactly on a full cell
  TEST(BuildViscositySamples, SamplesOrderedByBlockThenCellIdThenKindThenAxes)
  TEST(BuildViscositySamples, NullCellEighths_IsBadInput)          // and null face_piece_centres
  TEST(BuildViscositySamples, HugeViscosity_IsNotFinite)           // 1e308 Pa s in a 1 m cell: first cell's id
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** steps 1, 2, 4 (stretch part) and 5 with `// Step N:` comments.
  Normal coefficients replace `liquid_adaptive_step_3d.cc:331-370` and the tank edit of
  `liquid_substep_3d_cpu.cc:690-698` with the one `open_volume` rule (design step 2, FIXED/NEW). Mixture from
  `liquid_substep_3d.cc:1653-1659` with the one-ring rule.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Build the stretch samples of thick liquid on the CPU`.
  New names: `ViscosityInputs` (with `pressure_buffers`), `BuildViscositySamples`.

### Task 10: Build on the CPU: edge shear samples and sticking to solids

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/viscosity/samples_of_cell.h`, `viscosity_samples_cpu.cc`, `viscosity_samples_test.cc`

**Interfaces:**
- Consumes: `grid::FindCellsAroundEdge`, `contact::ComputeSolidVelocitySample`.
- **Plan decisions** (the design's rules made exact):
  - **Which cell builds an edge sample:** every active cell of the range visits its 12 (axis pair, quarter)
    edges; it builds the edge sample when it has the lowest `CellId` among the active cells around that edge,
    whether or not it holds liquid. Zero weight: no sample.
  - **Inclined solid at an edge:** a cell holds a solid surface that is not axis-aligned when its
    `liquid_covered_area > 0` and `ComputeSolidVelocitySample(solids, count, 0, liquid_covered_solid_centre,
    cell_width)` has a solid whose `normal_out_of_solid` has largest component magnitude below `1 - 1e-12`.
    Elapsed time 0: the snapshot's geometry is at the step start.
  - **Usable face** of a nominal pair: a face piece with `face_inertia > 0` (solved), or with `open_area == 0`
    whose centre is on a solid surface (`|distance| <= 1e-9 x cell_width` from `ComputeSolidVelocitySample`
    with search distance `cell_width`): a given face whose value is `velocity_before_pressure`. Anything else is missing.
  - **Rule (a), solid between:** for a missing face `F` with usable partner `P` along the derivative axis,
    sample the solids at `P`'s centre with search distance `cell_width`; if a solid lies between (`0 < distance <=
    cell_width` and its normal points from the solid toward `P`), the derivative is `(u_P - v_solid) / distance`
    with the sign of `P`'s side: coefficient on `P` is `+-0.5 / distance`, `rate_from_solids -= +-0.5 x
    v_solid / distance` (`v_solid`: the solid velocity's component along the face axis). A domain side gives
    `distance = cell_width / 2`. Ported from `viscosity.cc:1290-1327`, now the production rule.
  - **Rule (b), liquid surface beyond:** otherwise the nearest face pair (same axis, one width apart along the
    derivative axis, both usable) that shares an open connection with the edge, smallest squared offset first,
    ties by lower `FacePlaceIndex` (port `viscosity.cc:999-1045`). None: `kNoFacePairForShear`.
  - Connection check first (port `viscosity.cc:973-995`): the wet quarters must touch through open faces, else
    `kLiquidAtEdgeNotConnected`.

- [ ] **Step 1: Failing tests** (same-size grids):

  ```cpp
  TEST(BuildViscositySamples, LinearShear_EveryXzSampleIsHalfTheShear)  // design 7 "Linear shear", samples:
      // ShearChannelSettings(kSameSize, 16, ratio 1, top 1, dt 0.005, u_x = z): every XZ sample == 0.5,
      // including the floor and plate samples; 1e-10 relative
  TEST(BuildViscositySamples, Floor_UsesHalfCellToSolid)                // floor edge: one coefficient 1/h on the
      // face at h/2, rate_from_solids 0
  TEST(BuildViscositySamples, GridBelowFloor_UsesTrueDistanceToSolid)   // Review Focus 2: same channel with two
      // solid cell layers below z = 0: identical samples, bytes
  TEST(BuildViscositySamples, DrivenPlate_RateFromSolidsCarriesItsSpeed)  // plate edge: rate_from_solids ==
      // -0.5 * 1 / (h/2) with the face on the low side
  TEST(BuildViscositySamples, EdgeLowestCellIsAir_BuiltOnce)            // Review Focus 1: floating cube; each top
      // edge appears exactly once and is built by the air cell with the lowest CellId
  TEST(BuildViscositySamples, EachLiquidQuarterCountedOncePerAxisPair)  // floating cube: per axis pair, sum of
      // shear weights == 4 * viscosity * total filled_volume, 1e-13 relative
  TEST(BuildViscositySamples, DiagonalLiquidAcrossSolid_IsLiquidAtEdgeNotConnected)  // 2x1x2 cells, liquid in
      // (0,0,0) and (1,0,1), solid boxes fill the other two: the error and the lower id's cell
  TEST(BuildViscositySamples, OneCellThickLayer_IsNoFacePairForShear)  // liquid 4 x 1 x 4 between air in y
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** step 3 (edge part) in `BuildSamplesOfCell` with
  `// Step 3:` comments, from `viscosity.cc:764-935` and `965-1194` (weights over liquid quarters).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Build edge shear samples that stick to solids`.

### Task 11: Build on the CPU: fitted samples, the by-face view and given faces

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the by-face view uses
`BuildGroupsByDestination` with `buffers.order_buffers`.

**Files:**
- Modify: `native/fluidity/viscosity/samples_of_cell.h`, `viscosity_samples_cpu.cc`, `viscosity_samples_test.cc`

**Interfaces:**
- Consumes: Task 5; `grid::FindNearbyFaces`, `ComputeCellCenter`, `ToCellBlockIndex`.
- **Plan decisions:** a fitted sample per cell and axis pair `(a, b)` gathers that cell's quarters whose edges
  are not edge samples; weight `4 x cell_viscosity x` their filled volume. Positions for the fit of `u_a`: the
  usable faces (Task 10's rule) of axis `a` from `FindNearbyFaces(tables, centre, a, block, &error)`, at their
  `liquid_covered_open_centre`, plus the cell's `liquid_covered_solid_centre` when `liquid_covered_area > 0`
  (known velocity from `ComputeSolidVelocitySample` at elapsed 0); likewise for `u_b`. The sample is
  `0.5 x (d u_a / d b + d u_b / d a)`: up to 8 face coefficients, merged by face in ascending
  `FacePlaceIndex`; known positions add to `rate_from_solids`. A `GridError` from `FindNearbyFaces` is
  `kTooFewNearbyFaces`. Step 4's by-face view: `BuildGroupsByDestination(..., buffers.order_buffers)` (it
  allocates nothing) with destination = face of each coefficient. Step 5: each coefficient whose face has `face_inertia == 0` adds `coefficient x
  velocity_before_pressure` to `rate_from_solids` and becomes 0. Then every stored coefficient becomes
  `sqrt(step_duration x sample_weight) x coefficient`; `ReadViscositySamples` divides that factor back out
  (snapshot coefficients are in 1/m).

- [ ] **Step 1: Failing tests** (`kSizeChange` layouts):

  ```cpp
  TEST(BuildViscositySamples, SizeChange_LinearShearSamplesAreHalfTheShear)  // ShearChannel kSizeChange, u_x = z:
      // every XZ sample (edge or fitted) == 0.5, 1e-10 relative
  TEST(BuildViscositySamples, SizeChange_QuartersCountedOncePerAxisPair)     // as in Task 10, two-level cube
  TEST(BuildViscositySamples, SameSizeGrid_HasNoFittedSample)
  TEST(BuildViscositySamples, OneCellThickGridWithInclinedSolid_IsTooFewNearbyFaces)  // BuildBoxGrid {4,1,4},
      // solid plane x + z = 0.13 (unit normal): positions are coplanar in y; the first fitted cell's id
  TEST(BuildViscositySamples, GivenFace_FoldedIntoRateFromSolids)   // Review Focus 3: ShearChannel inlet faces:
      // stored coefficient 0, rate_from_solids includes coefficient * profile; snapshot rate unchanged
  TEST(BuildViscositySamples, ByFaceViewListsTheSamePairs)          // every (sample, coefficient) once, in
                                                                    // ascending sample index per face
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step 3:`, `// Step 4:`, `// Step 5:` comments; the fit
  replaces `liquid_adaptive_step_3d.cc:279-302` (design step 3, NEW split by edge).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Build fitted shear samples where cell sizes change`.

### Task 12: The two passes and the power lost, on the CPU

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the per-sample, per-face and per-block bodies live in a shared
`velocity_solve_steps.h` that the CPU loops call now and the GPU functions call in Task 13.

**Files:**
- Create: `native/fluidity/viscosity/velocity_solve.h` (private, declared for tests), `velocity_solve_cpu.cc`,
  `velocity_solve_steps.h` (private, shared, C++20 subset), `power_lost_test.cc`
- Modify: `solve_viscosity_and_pressure.h` (`ComputePowerLostToViscosity`), `CMakeLists.txt`

**Interfaces:**
- Produces: `ComputePowerLostToViscosity` exactly. **Plan decision**, private, all on the buffers' processor:

  ```cpp
  // Sample pass: sample_rate[s] = sum over its stored coefficients of coefficient x velocity[face], in order.
  std::expected<void, ViscosityFailure> ComputeSampleRates(ViscosityBuffers& buffers,
                                                           const grid::FaceValues<double>& velocity);
  // Face pass: momentum[f] = face_inertia[f] x velocity[f] + sum over its uses of coefficient x sample_rate[s];
  // 0 on given faces. One use of (face mass + step_duration x drag).
  std::expected<void, ViscosityFailure> ComputeMomentumFromVelocity(ViscosityBuffers& buffers,
      const grid::FaceValues<double>& velocity, grid::FaceValues<double>& momentum);
  ```
  Power lost = `sum_s (stored rate_s + sqrt(dt W_s) x rate_from_solids_s)^2 / dt`, per block (block of the
  sample's cell) then `math::TreeSum`.
  **Amended 2026-11-14 (Foundation as built), plan decision:** `velocity_solve_steps.h` holds, on plain pointers
  of one processor (`struct PassArrays`), `FLUIDITY_CPU_AND_GPU double ComputeSampleRateOfSample(PassArrays,
  std::uint32_t sample)`, `FLUIDITY_CPU_AND_GPU double ComputeMomentumOfFace(PassArrays, grid::FacePlaceIndex face)`,
  `FLUIDITY_CPU_AND_GPU double ComputePowerPartialOfBlock(PassArrays, std::uint32_t block)` and
  `FLUIDITY_CPU_AND_GPU double ComputeDotPartialOfBlock(const double* first, const double* second, std::uint32_t
  block)` (the block's places in ascending index, as "Order of sums" states). The CPU loops of this task call them
  item by item; the GPU functions of Task 13 and 17 are grid-stride loops around the same calls.

- [ ] **Step 1: Failing tests** (both layouts unless stated):

  ```cpp
  TEST(BuildViscositySamples, RigidTranslation_EverySampleRateIsZero)  // design 7: floating cube, (0.3,-0.2,0.7):
      // |rate| <= 1e-13 * speed / cell_width; power lost <= that bound squared * weights
  TEST(BuildViscositySamples, RigidRotation_EverySampleRateIsZero)     // design 7: 1 rad/s about Z through the cube
      // centre and about (1,1,1)/sqrt(3): |rate| <= 1e-12; power lost 0 within 1e-24 * sum of weights
  TEST(ComputePowerLostToViscosity, LinearShear_IsViscosityShearSquaredVolume)  // design 7: channel ratio 1,
      // u_x = 2 z: power == 1 * 4 * liquid volume, 1e-10 relative
  TEST(ComputeMomentumFromVelocity, SymmetricAndNeverNegative)         // design 7: channel kSizeChange and cube,
      // random x, y on solved faces: x.(K y) == y.(K x), x.(K x) >= 0, K the face pass, 1e-12 relative
  TEST(ComputePowerLostToViscosity, SameBytesOnRepeat)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the gathers (`viscosity.cc:1424-1433` turned from scatter to
  gather); CPU threads may split each pass by destination. **Step 4: Run** the gate. **Commit.** Subject:
  `Apply drag in two gathers and measure the power it takes`. New name: `ComputePowerLostToViscosity`.

### Task 13: Build and passes on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two GPU halves with private `_gpu.h` headers returning
`RunOnCallStatus`, launches through `base/gpu_launch.cuh`, GPU functions as thin loops over `samples_of_cell.h`
and `velocity_solve_steps.h`, Foundation's prefix sum and grouping (taking `buffers.order_buffers`) instead of a scan written here,
`cpu` tests over every thread block and thread, digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/viscosity/viscosity_samples_gpu.h` (private), `viscosity_samples_gpu.cu`,
  `velocity_solve_gpu.h` (private), `velocity_solve_gpu.cu`, `viscosity_steps_test.cc` (label `cpu`),
  `viscosity_samples_gpu_test.cc`, `viscosity_samples_no_gpu_test.cc`
- Modify: `viscosity_samples_cpu.cc` and `velocity_solve_cpu.cc` (dispatch by `RunOn`; the GPU half called under
  `#if defined(FLUIDITY_HAS_GPU)`, its status mapped by `ToBufferError` then `ToViscosityFailure`; without GPU code
  a `kGpu` call is `kRunOnCallFailed`), `CMakeLists.txt` (the two `.cu` by `fluidity_add_gpu_sources`, the three
  tests)

**Interfaces:**
- Produces (private, may say `Gpu`): in `viscosity_samples_gpu.h`, `RunOnCallStatus BuildSamplesOneThreadPerCell(...)`;
  in `velocity_solve_gpu.h`, `RunOnCallStatus ComputeSampleRatesOneThreadPerSample(...)`,
  `ComputeMomentumOneThreadPerFace(...)`, `ComputePowerPartialsOneThreadPerBlock(...)` (plain pointers and counts;
  no CUDA header). Each starts its GPU functions with `ApplyGpuFunction(function, ComputeLaunchSize(count), ...)`
  and ends with `CheckGpuWork()`; every array is in `ViscosityBuffers`, so nothing allocates (no `GpuArray` here).
  Build: cell order by `BuildStableOrder(..., buffers.order_buffers)`, count GPU function (one
  `BuildSamplesOfCell` call with null `samples` per cell), `ComputeExclusivePrefixSum(counts, cell_count, starts,
  buffers.order_buffers)` (Foundation's; integers, so exact), write GPU function (one `BuildSamplesOfCell` per cell
  at its start), then `BuildGroupsByDestination(..., buffers.order_buffers)`. The failing cell is the lowest
  `CellId` whose flag is set (an integer minimum over the per-cell error flags, a shared per-cell function
  `ComputeFailingCellKey` in `samples_of_cell.h`).

- [ ] **Step 1: Tests:**

  ```cpp
  // viscosity_steps_test.cc, LABEL cpu: the GPU arithmetic without a GPU (the main evidence until the hardware run)
  TEST(BuildSamplesOfCell, EveryThreadOfLaunch_EqualsCpuBuild)  // Tasks 9-11 scenes: for every thread block and
      // thread of ComputeLaunchSize(cell_count), in launch order and in reverse, the count phase then (after
      // ComputeExclusivePrefixSum) the write phase, each thread walking its grid-stride items: samples,
      // coefficients and counts equal BuildViscositySamples on the CPU; bytes (same functions, same processor)
  TEST(ComputeFailingCellKey, EveryThreadOfLaunch_GivesLowestFailingCellId)  // two failing cells in different
      // thread blocks, the lower CellId at the higher index: both orders give the lower id
  TEST(ComputeSampleRateOfSample, EveryThreadOfLaunch_EqualsCpuPass)  // and ComputeMomentumOfFace,
      // ComputePowerPartialOfBlock: random velocity, launch order and reverse; bytes
  // LABEL gpu
  TEST(BuildViscositySamples, OnGpu_EqualsCpu)              // Tasks 9-11 scenes: counts and faces equal; weights,
                                                            // coefficients, rate_from_solids 1e-15 + 1e-13 |v|
  TEST(BuildViscositySamples, OnGpu_ByFaceViewListsTheSamePairs)   // design 7 GPU row
  TEST(ComputeMomentumFromVelocity, OnGpu_EqualsCpu)        // one operator use, same tolerance
  TEST(BuildViscositySamples, OnGpuTwice_GivesIdenticalBytes)
  // LABEL no_gpu
  TEST(BuildViscositySamples, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
  CPU against GPU: discrete outputs (counts, faces, order, by-face view) equal; doubles within design 6 with
  `CheckWithinTolerance` (no header promises same bits for these formulas). Every `gpu` test calls
  `WriteResultDigest` on its samples, coefficients and momentum.
- [ ] **Step 2: Implement**; nothing allocates after `BuildViscosityBuffers`. **Step 3: Run** the gate.
  **Commit.** Subject: `Build viscosity samples and apply drag on the GPU`. Body: `CUDA: compiled, not run`.

### Task 14: The velocity solve on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** every dot of the iteration is block partials then
`ComputeTreeSum(partials, block_count, buffers.sum_buffers)` and every largest `|error|` or `|momentum|` is
`ComputeLargestMagnitude(values, count, buffers.sum_buffers)`: nothing allocates inside the loop.

**Files:**
- Modify: `native/fluidity/viscosity/velocity_solve.h`, `velocity_solve_cpu.cc`
- Create: `velocity_solve_test.cc`

**Interfaces:**
- Produces (private, **Plan decision**):

  ```cpp
  struct VelocitySolveResult { int iteration_count = 0; double largest_momentum_error = 0; };  // kg m/s
  // Step 9: solves (face mass + dt x drag) velocity = momentum on solved faces, from the start in velocity,
  // diagonal scaling by velocity_diagonal (Step 6: face_inertia + sum of squared stored coefficients).
  std::expected<VelocitySolveResult, ViscosityFailure> SolveVelocity(ViscosityBuffers& buffers,
      const grid::FaceValues<double>& momentum, const ViscositySettings& settings,
      grid::FaceValues<double>& velocity);
  ```
  Stop when `largest |error| <= velocity_tolerance x largest |momentum|`; then recompute the true error with
  the face pass and accept only if it also passes, else restart from the true error with a fresh direction
  (`viscosity.cc:1485-1502`). A direction with curvature `<= 0` gives `kVelocitySolveDidNotConverge` with
  magnitude the curvature. Iterations used up:
  `kVelocitySolveDidNotConverge`, magnitude = last error. Non-finite: `kNotFinite`.
  Also produces `ViscositySettings` in the public header exactly.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(SolveVelocity, ShearWaveDecay_MatchesDiscreteRate)   // design 7 "Shear wave decay": ShearChannel kSameSize
      // n = 32, ratio 1, top plate at rest, density 1000, viscosity 1, dt 0.005, u_x = sin(k z), k = 2 pi:
      // momentum = face_inertia u: after one solve amplitude ratio == 1 / (1 + dt nu k_h^2),
      // k_h = (2/h) sin(k h / 2), nu = 1e-3; 1e-9 relative on every X face
  TEST(SolveVelocity, ZeroMomentum_GivesZero)
  TEST(SolveVelocity, OneIterationAllowed_IsVelocitySolveDidNotConverge)
  TEST(SolveVelocity, AcceptedOnlyAfterTrueErrorPasses)     // largest_momentum_error recomputed <= tolerance
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `viscosity.cc:1331-1511` (gather form, `double`), with
  `// Step 9:` comments. **Step 4: Run** the gate. **Commit.** Subject: `Solve for the velocity under drag on
  the CPU`. New name: `ViscositySettings`.

### Task 15: The joint solve on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A24):** the open note is removed; the coupled terms are Task 15a's (option C, the owner's
choice). Until Task 15a is committed, equations holding a material face or a sealed gas blob give `kBadInput` here.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** dots by block partials and `ComputeTreeSum` with
`buffers.sum_buffers`, largest values by `ComputeLargestMagnitude` with the same buffers; no `ComputeTreeDot`.

**Files:**
- Create: `native/fluidity/viscosity/solve_viscosity_and_pressure_cpu.cc`, `solve_viscosity_and_pressure_private.h`,
  `solve_viscosity_and_pressure_test.cc`
- Modify: `solve_viscosity_and_pressure.h` (`ViscosityResult`, `SolveViscosityAndPressure`)

**Interfaces:**
- Consumes: Tasks 9 to 14; pressure's `ComputePressurePush(..., GivenTerms, ...)`, `ComputeVolumeError`,
  `ComputeOwnWeight`, `ComputeStartPressure`, `ApplyZeroMeanToSealedBlobs`, `ApplyPressureOffset`,
  `CheckFlowImbalance`; `ComputeDotPartialOfBlock` (Task 12) then `ComputeTreeSum(partials, block_count,
  buffers.sum_buffers)`, and `ComputeLargestMagnitude(values, count, buffers.sum_buffers)` (nothing
  allocates inside the iteration loop).
- Produces: `ViscosityResult`, `SolveViscosityAndPressure` exactly. Private
  (**Plan decision**, for the cancel test):

  ```cpp
  struct BeforeInnerSolve { void (*function)(int outer_iteration, void* context) = nullptr; void* context = nullptr; };
  std::expected<ViscosityResult, ViscosityFailure> SolveViscosityAndPressureWithHook(
      const ViscosityInputs&, const ViscositySettings&, const std::atomic<bool>* cancel_requested,
      ViscosityBuffers&, grid::FaceValues<double>& velocity_after_pressure, grid::CellValues<double>& cell_pressure,
      BeforeInnerSolve before_inner_solve);   // the public function passes {}
  // Step 11's response for a direction, exposed for the linearity test.
  std::expected<void, ViscosityFailure> ComputeResponse(const ViscosityInputs&, const ViscositySettings&,
      ViscosityBuffers&, const grid::CellValues<double>& direction, grid::CellValues<double>& response);
  ```
- **Plan decisions:**
  - Acceptance: a pressure is accepted when `pressure::CheckFlowImbalance(..., settings.pressure_acceptance,
    *inputs.pressure_buffers)` returns a report; `PressureError::kNotConverged` means "not yet" (as
    `SolvePressure` uses it, pressure Task 34). Any other `PressureFailure` maps: `kBadInput`,
    `kWrongGridVersion` to `kBadInput`; `kNotFinite`; `kRunOnCallFailed`. If on the branch `CheckFlowImbalance`
    returns a report for a pressure outside the limits, stop and ask the owner before adding a comparison here.
  - Given faces of the output take `velocity_before_pressure`, as `SolvePressure` leaves pieces with no
    connection (after `CollideSolids`, a solid face already holds the solid's velocity).
  - Step 11 direction: `imbalance / pressure diagonal` (the design's diagonal; pressure's ladder is not used),
    then `ApplyZeroMeanToSealedBlobs`; response and direction are both reduced against the kept pairs in the
    order kept, then both scaled by `1 / |response|`; `pressure -= (imbalance . response) x direction`.
    `|response| == 0`: `kPressureSolveDidNotConverge`. `kept_response_count` outside 1..32 or an iteration
    limit below 1: `kBadInput`.
  - `block_range` not equal to every block: `kBadInput`; an output that is the same object as an input array:
    `kBadInput`; null `pressure_buffers`: `kBadInput`. Outputs are written only in step 12, by copies from the
    buffers; `cell_pressure` is 0 in cells without an equation.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(SolveViscosityAndPressure, RigidRotationGiven_LeavesItUnchanged)  // design 7 "Rigid rotation", solve:
      // floating cube both layouts, rotation about Z and about (1,1,1)/sqrt(3), 1 rad/s, dt 0.01, every face
      // starts at the rotation: velocity after == before within 1e-9 relative to the largest speed
  TEST(SolveViscosityAndPressure, LinearShear_VelocityUnchanged)        // design 7 "Linear shear", solve: channel
      // both layouts, ratio 1, u_x = z, top speed 1: velocity after == before, 1e-10 relative
  TEST(SolveViscosityAndPressure, NearlyInviscid_EqualsSolvePressure)   // design 7: SyrupColumn with viscosity
      // 1e-12 and a driven-solid variant (pressure Task 36 scene): velocity and pressure equal SolvePressure's
      // within the pressure stop rule (velocity 2 * 2e-5 * w)
  TEST(SolveViscosityAndPressure, ResponseIsLinearInDirection)          // Review Focus 4: PlanarExtension with
      // surface tension 0.08, carried pressure 50 Pa, a driven solid side: response(2 d) == 2 response(d) and
      // response(d1 + d2) == response(d1) + response(d2), 1e-10 relative (inner tolerance)
  TEST(SolveViscosityAndPressure, ResultCounts)                         // velocity_solve_count ==
      // pressure_iteration_count + 1 + number of corrections; power_lost equals ComputePowerLostToViscosity bytes
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** steps 6 to 12 with `// Step N:` comments, from
  `git show main:native/src/mac_viscous_pressure_3d.cc` lines 151-461 (without the zero-pressure first solve,
  the second limit, `CompensatedSum` and the representable checks; design section 10). Cancel is read before
  each inner solve (Task 16 tests it).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Solve drag and pressure together on the CPU`. New names:
  `ViscosityResult`, `SolveViscosityAndPressure`.

### Task 15a: Blocking materials and sealed gas in the joint solve, on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A24):** new task; the method of `plan-decisions.md` "Viscous liquids with blocking materials
and sealed gas (A24)".

**Amended 2026-11-14 (Foundation as built):** tests are `LABEL cpu`; the cell part of each dot keeps Task 15's
block partials and `ComputeTreeSum(..., buffers.sum_buffers)`.

**Amended 2026-11-14 (A26):** the material-face elimination of this task (Step 6 `g_k` on the diagonal, the
`ComputeVolumeError` form without `CoupledVelocities`) is replaced in Task 15c, where material faces become
velocity unknowns; the sealed gas part stays as written; the tests below keep their expectations (their material
cases are limits that Task 15c's form also meets) and `NearlyInviscidCoupled_EqualsSolvePressure` uses Task 49's
slab declared a springy solid (owner note: liquid-on-snow is not a validation target).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure.h` (`SolvedCouplingArrays`, the last parameter of
  `SolveViscosityAndPressure` and of the hook form), `solve_viscosity_and_pressure_cpu.cc`, `viscosity_buffers.cc`,
  `viscosity_test_scenes.h`, `viscosity_test_scenes.cc`, `docs/method-paper/sections/viscosity.typ` (pseudocode
  lines of Step 3c)
- Create: `native/fluidity/viscosity/coupled_terms_test.cc`

**Interfaces:**
- Consumes: Tasks 7, 15; pressure's A24 forms (pressure Tasks 45 to 47): `ComputePressurePush(equations, pressure,
  sealed_gas_pressure, given_terms, push)`, `ComputeVolumeError(..., sealed_gas_pressure, velocity, given_terms,
  volume_error, sealed_gas_volume_error)`, `ComputeOwnWeight(..., own_weight, sealed_gas_own_weight)`,
  `ComputeStartPressure(..., pressure, sealed_gas_pressure)`, `ComputeBlockingMaterialVelocities`, the
  `CheckFlowImbalance` form with given blob pressures; `pressure::testing::BuildBlockingMaterialFacesForTest`.
- Produces: design 3 `SolvedCouplingArrays` and the A24 signature of `SolveViscosityAndPressure` exactly.
  Test-only: `ViscousSceneSettings` gains `std::vector<pressure::BlockingMaterialFace> (*blocking_material_faces)(
  const grid::StaggeredGrid&, const pressure::GeometrySnapshot&, const void*) = nullptr;`,
  `std::vector<pressure::SealedGasBlob> sealed_gas_blobs;` and `double (*sealed_gas_blob_of_point)(Vec3, const
  void*) = nullptr;` (blob index, or -1, at a cell centre), passed into the scene's equations; and
  `ViscousScene::coupling_arrays()` (spans of the right sizes owned by the scene).
- **Plan decisions** (by step of design 5; `q` = blob pressures, a `Buffer<double>` of `max_sealed_gas_blobs`):
  - Step 6: cell and blob diagonals from the four-argument `ComputeOwnWeight` with `connection_scale = face_inertia
    / velocity_diagonal` (material weights `g_k` unscaled, blob `S_j + sum_c g_c scale_c`).
  - Step 7: base momentum with the blob form of `ComputePressurePush` at `p = 0`, `q = 0`, `kIncluded` (so a blob
    connection's given part is its `ref_c`). Step 8: `ComputeStartPressure` blob form (`q = c_j`).
  - Step 9: the push of a trial `(p, q)` is the blob form with `kLeftOut`. The inner solve is unchanged.
  - Step 10: volume errors of cells and blobs by the blob form of `ComputeVolumeError` (`kIncluded`); acceptance by
    the `CheckFlowImbalance` form with the given `q`.
  - Step 11: the direction has blob entries `E_j / diagonal_j`; the response is the blob form of `ComputeVolumeError`
    with `kLeftOut` (material terms and blob rows included); every dot = cell part (block partials, `math::TreeSum`)
    plus blob part (`math::TreeSum` over blobs in index order), added once in that order; kept pairs hold blob
    entries; `ApplyZeroMeanToSealedBlobs` unchanged (a liquid blob joined to a gas blob is never `kSealed`).
  - Step 12: material velocities by `ComputeBlockingMaterialVelocities(equations, accepted pressure,
    coupling_arrays.blocking_material_velocity_after_pressure)`; `q` copied into `coupling_arrays.sealed_gas_pressure`;
    both only after acceptance, as the two other outputs.
  - `kBadInput`: a span size not equal to its count; `sealed_gas_blob_count > max_sealed_gas_blobs`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(SolveViscosityAndPressure, CouplingAbsent_SameBytesAsBefore)       // Task 15's cases, empty spans: bytes
  TEST(SolveViscosityAndPressure, NearlyInviscidCoupled_EqualsSolvePressure)  // viscosity 1e-12 Pa s; (a) pressure
      // Task 49's slab (declared a springy solid) started at x* + 0.1 w, one step; (b) pressure Task 50's gas column, u0 1e-4, one step at
      // dt 0.1 / omega; (c) Task 50's heated pocket; (d, A27) (a) with a free box resting on the liquid: velocity and cell pressure within the pressure stop rule
      // (velocity 2 * 2e-5 * w + 1e-9 max |u|, pressure that bound times face_density * pressure_distance / dt);
      // material velocities within the velocity bound; sealed_gas_pressure within the pressure bound
  TEST(SolveViscosityAndPressure, ResponseIsLinearWithCouplingTerms)      // Review Focus 4: the slab scene with
      // viscosity 1 and the gas column with viscosity 0.1, carried pressure 50 Pa, blob carried 3 Pa, heat rate on:
      // response(2 d) == 2 response(d), response(d1 + d2) == response(d1) + response(d2), blob entries included,
      // 1e-10 relative
  TEST(SolveViscosityAndPressure, MaterialVelocityAndBlobRowFromAcceptedPressure)  // spans equal
      // ComputeBlockingMaterialVelocities(accepted pressure) bytes; CheckFlowImbalance with the written
      // sealed_gas_pressure accepts
  TEST(SolveViscosityAndPressure, CouplingSpansWrongSize_IsBadInput)      // and blobs above max_sealed_gas_blobs
  TEST(SolveViscosityAndPressure, CancelWithCoupling_AllOutputsUntouched) // both spans pattern-filled: unchanged
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step N:` comments; remove Task 15's temporary
  `kBadInput` on coupled equations. **Step 4: Run** the gate. **Commit.** Subject: `Carry blocking materials and
  sealed gas in the joint solve`. New names: `SolvedCouplingArrays`, `ViscosityBufferSettings::max_sealed_gas_blobs`
  (A24).

### Task 15b: Scene tests, blocking materials and sealed gas beside a viscous liquid

**Amended 2026-11-14 (A24):** new task; pressure Tasks 49 and 50 rerun with a viscous liquid where an exact answer
still exists.

**Amended 2026-11-14 (A26):** pressure Task 49's slab is declared a springy solid here (the stand-in elastic force
is a spring); the liquid now sticks to it (Task 15c), which changes no expectation below (at rest no gradient; the
decay comparison only gains dissipation). Runs after Task 15c.

**Kind:** integration

**Files:**
- Create: `native/fluidity/viscosity/coupled_scene_test.cc`

Each scene is the pressure plan's, built with `ViscousSceneSettings` (Task 15a fields) and stepped by the same test
loop as there (geometry fixed or rebuilt as the pressure test says; the snow stand-in and the gas update between
steps as there). The inviscid comparison runs the same loop with `pressure::SolvePressure`. Tolerances are the
pressure tests' (the joint solve has the same acceptance rule).

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ViscosityScene, ViscousLiquidOnSpringySlab_InterfaceHoldsAndCarriesTheWeight)  // pressure Task 49's slab with
      // the liquid at viscosity 10 Pa s, density 1000 (x* -9.8098e-3 m, p_1 1716.75 Pa unchanged: at rest every
      // velocity gradient is 0, so the hydrostatic state solves the joint system): 500 steps, |x_n - x*| <=
      // 1.3e-7 m each step, no trend; interface pressure 1962 Pa within twice the pressure bound; push per area
      // == -p_1, 1e-9 relative
  TEST(ViscosityScene, LightSlabInViscousLiquid_FloatsAtDisplacedDepth)  // pressure Task 49's floating slab, liquid
      // 1 Pa s, 1000 kg/m3 (viscous effect per step 0.016): 300 steps from (z_b* 0.1375, h_r* 0.1575) at rest:
      // |z_b - z_b*| <= 1e-6 w each step; h_r - z_c == 0.2 t within 1e-6 w
  TEST(ViscosityScene, LightSlabInViscousLiquid_DecaysAtLeastAsFastAsInviscid)  // start z_b* - 0.2 w, h_r* + 0.2 w,
      // dt 0.01, 500 steps, viscous (1 Pa s) and inviscid runs: in each window of 100 steps max |z_b - z_b*| of
      // the viscous run <= that of the inviscid run + 1e-6 w; over the last 100 <= 1.02 * 0.2 w; mean z_b over
      // the last 300 within 0.03 w of z_b*
  // Gas column of pressure Task 50 (2 x 2 x 16 cells, w 0.025, m 0.5 kg, k 2364.25 N/m, omega 68.764 1/s) with
  // the water at viscosity mu = 0.1 Pa s. By symmetry the four cell columns each touch two side walls, so the
  // flow stays a plug: every wall edge sample is U / w (the wall at half a cell, rule (a)); weights 2 mu V on
  // the 7 inner z levels and mu V on the 2 surface levels, 8 wall edges per level: power lost = 128 mu V U^2 /
  // w^2, drag c = 128 mu w = 0.32 N s/m. Backward Euler of m v' = -k x - c v: per-step factor 1 / sqrt(1 + a + b),
  // a = (c / m) dt, b = (omega dt)^2; phase theta = acos((2 + a) / (2 sqrt(1 + a + b))).
  TEST(ViscosityScene, SealedGasAboveViscousColumn_PeriodAndDamping)  // dt 0.1 / omega = 1.45425e-3 s, u0 1e-4
      // m/s, a 9.3072e-4, b 0.01: per-step factor 0.994579 within 1e-4 relative (inviscid 0.995037); period from
      // upward zero crossings (mean of periods 1 to 5) == 2 pi dt / theta = 0.091720 s (theta 0.0996215) within
      // 0.05 %, and within 0.1 % of the inviscid run's measured period (0.091677 s: the 0.047 % difference is
      // backward Euler's first-order part; the continuous frequency changes by 1.1e-5 relative); each period's
      // largest |u| <= the previous one's and <= the inviscid run's in the same period
  TEST(ViscosityScene, SealedGasAboveViscousColumn_StableWhereLagGrows)  // dt 4 / omega = 0.058170 s, u0 1e-9 m/s,
      // 79 steps: |u| never grows; per-step factor 1 / sqrt(17 + a), a 0.037229, = 0.242270 within 1e-4
  TEST(ViscosityScene, SealedGasHeatedAboveSyrup_RisesInTheSameSolve)  // pressure Task 50's heated pocket with syrup
      // 10 Pa s, 1400 kg/m3, one solve at dt 0.03 s: sealed_gas_pressure == 6.5625 Pa within 1e-9 relative; liquid
      // pressure == 6.5625 + 1400 g (0.2 - z) at centres within the pressure bound; velocity 0 within the velocity
      // bound
  ```
- [ ] **Step 2: Run.** A failure is fixed in the owning commit (Task 15a, or pressure's A24 pieces), never by a
  looser bound; if the column is not a plug within the velocity bound, stop and report (the drag derivation then
  does not hold). **Step 3: Run** the gate. **Commit.** Subject: `Test materials and sealed gas beside a viscous
  liquid`.

### Task 15c: Material faces and free solids as unknowns of the joint solve, on the CPU

**Amended 2026-11-14 (A26, A27):** new task; the method of `plan-decisions.md` "A viscous liquid sticks, and free
solids in the viscous solve (A26, A27)".

**Amended 2026-11-14 (Foundation as built):** tests are `LABEL cpu`; the per-solid sums and the 6 x 6 factor are
shared per-item functions in `velocity_solve_steps.h`, so Task 17a's GPU functions reuse them.

**Amended 2026-11-15 (A59):** a free solid's unknowns are the m generalized velocities `q` of its
`FreeSolidForViscousSolve::allowed_motions` (`xi = J q + xi_0`); the block is `J^T (M + dt R^T E_s^T W E_s R) J` (m x m)
and the per-sample coefficients `R J`; the "Guided solids" decision below and its test are replaced (design 5
"Velocity unknowns", decision 25). This touches the A27 block derivation that reconcile-todo item 23 asks to have
reviewed; item 23 stays open and now covers the m x m block.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure.h` (`ViscosityInputs::material_faces`,
  `free_solids`; `SolvedCouplingArrays::material_face_velocity_after_solve`, `free_solid_motion`;
  `ViscosityBufferSettings::max_material_faces`, `max_free_solids`), `samples_of_cell.h`, `viscosity_samples_cpu.cc`,
  `velocity_solve.h`, `velocity_solve_cpu.cc`, `velocity_solve_steps.h` (plan decision: `FLUIDITY_CPU_AND_GPU`
  `ComputeSolidSumOfSolid` (one solid's six sums by the per-group rule), `ComputeSolidBlockFactor` and
  `ApplySolidBlockFactor` (the 6 x 6 Cholesky factor and its solve)), `solve_viscosity_and_pressure_cpu.cc`,
  `viscosity_buffers.cc`,
  `viscosity_memory_estimate.cc`, `viscosity_test_scenes.h`, `viscosity_test_scenes.cc`
- Create: `native/fluidity/viscosity/coupled_unknowns_test.cc`

**Interfaces:**
- Consumes: Tasks 9 to 15a; solids' `ViscousMaterialFace`, `FreeSolidForViscousSolve`, `FreeSolidMotion` (solids
  Tasks 23a and 23c; tests hand-build them); pressure's A26/A27 forms (pressure Task 47): `CoupledVelocities`, the
  `ComputeVolumeError` and `CheckFlowImbalance` forms with it, `ComputeCoupledPush`, the `ComputeOwnWeight` form with
  `material_face_scale`; `pressure::FreeSolidPush` entries of the equations.
- Produces: the A26/A27 fields of design 3 exactly. Test-only: `ViscousSceneSettings` gains a springy-slab and a
  free-box description (`std::vector<solids::ViscousMaterialFace> (*material_faces)(...)` built by brute force from a
  material distance function, face mass `density * w^3`; `std::vector<FreeBox> free_boxes` with centre, half size,
  density, velocity and turn rate).
- **Plan decisions** (design 5 "Coupled terms", by step):
  - Build (steps 3, 5): rule (a) generalised: when the surface between a partner face and a missing face belongs to
    a blocking material (the coverage `distance_to_solid` and its corner owner), the coefficient `-/+0.5 / d` goes on
    the unknown of the set's `ViscousMaterialFace` at the missing face (larger face mass when two sets hold it;
    ties lower set); when it belongs to a free solid, six coefficients on `xi` (`e_a` for `V`, `(y - origin) x e_a`
    for `Omega`, times `-/+0.5 / d`), `y` the surface point; fitted samples likewise for a liquid-covered solid
    centre (nearest band face of that axis for a material). Driven and fixed solids stay in `rate_from_solids`.
    A sample's solid coefficients are stored beside it (solid index and six numbers; at most one free solid per
    sample; a second free solid in one sample is folded into `rate_from_solids` at its velocity before pressure and
    counted in `ViscosityResult`'s hand-over note: plan decision, listed for the owner).
  - Unknown layout: liquid face places, then material entries in input order, then six per solid. The sample pass
    reads all three; the face pass gathers liquid and material destinations; each solid's six sums by the per-group
    rule. Step 5's given-face rule applies only to liquid faces.
  - Step 6: `velocity_diagonal` of a material entry `face_mass + sum of squared stored coefficients`; each solid's
    6 x 6 block `[m I3, 0; 0, I_start] + dt R^T E_s^T W E_s R`, factored once (Cholesky; non-positive pivot:
    `kVelocitySolveDidNotConverge` with that solid's index in `magnitude`). Outer diagonal: `ComputeOwnWeight` with
    `material_face_scale_k = face_mass_k / velocity_diagonal_k`; solids left out.
  - Step 7: base momentum of a material entry `face_mass * velocity_before_pressure` (its internal force is inside
    it: `materials::ApplyInternalForce` ran before `CollideSolids`), of a solid `[m V_before; I Omega_before]`, minus
    the stored-coefficient part of `rate_from_solids`, as for liquid faces.
  - Step 9: push of `(p, q)` on liquid faces (A24 form) plus `ComputeCoupledPush` on materials and solids.
  - Steps 10, 11: volume errors and acceptance with `CoupledVelocities` of the current `w`, `xi`.
  - Step 12: `material_face_velocity_after_solve` and `free_solid_motion` written; the `BlockingMaterialFace`
    subset also to `blocking_material_velocity_after_pressure`; all only on success.
  - ~~Guided solids: a component forbidden by `FreeSolidForViscousSolve::limits` is held at its value before pressure
    (identity row and column in the 6 x 6 block, its coefficients folded into `rate_from_solids`): `K` stays
    symmetric positive definite; test `SolveViscosityAndPressure, GuidedBoxKeepsForbiddenMotion` (a vertical guide:
    the solved x, y velocity and every turn rate equal the values before pressure, bytes).~~ **Replaced 2026-11-15
    (A59).** Guided solids in generalized velocities: with `J` the `column_count` (m) columns of
    `allowed_motions.columns` and `xi_0 = allowed_motions.prescribed_motion`, the solid's unknowns are `q` (six slots,
    the first m used, the rest held at 0). Build: the six stored `R` numbers stay as before (Task 15e stores them for
    every solid); the sample pass reads `R J q` (m dot products of length 6, `J` read from the inputs) and `E_s R
    xi_0` goes to `rate_from_solids`. Step 6: `ComputeSolidBlockFactor` forms `J^T ([m I3, 0; 0, I_start] + dt R^T
    E_s^T W E_s R) J` (m x m; the 6 x 6 sums of `ComputeSolidSumOfSolid` first, then the two products with `J` in
    fixed order) and its Cholesky factor (non-positive pivot: `kVelocitySolveDidNotConverge`, solid index in
    `magnitude`). Step 7: base momentum `J^T (M xi_before - (M + dt R^T E_s^T W E_s R) xi_0)`. Step 9: the six numbers of
    `ComputeCoupledPush` multiplied by `J^T`. Steps 10 to 12: `xi = J q + xi_0` for `CoupledVelocities` and
    `free_solid_motion`. `m == 0`: the solid has no unknown and `xi = xi_0`. `K` is symmetric positive definite
    because `J` has full column rank. The solve never moves a forbidden direction; solids no longer drops forbidden
    motions after it (solids Task 23c, amended).
  - `free_solids` lists every free solid of the try in `solid_index` order (solids plan decision); one with no sample
    use moves by its own row.
  - `kBadInput`: span sizes; more entries than `max_material_faces` or `max_free_solids`; a material entry whose
    face place is not a face of the grid version.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildViscositySamples, RigidTranslationWithSlabAndBox_EveryRateZero)  // floating cube of liquid with a
      // springy slab under it and a free box inside, all moving at (0.3, -0.2, 0.7): every rate <= 1e-13 * speed / w
  TEST(BuildViscositySamples, RigidRotationWithBox_EveryRateZero)            // 1 rad/s about Z through the box
      // origin, liquid and box together: every rate <= 1e-12
  TEST(BuildViscositySamples, SlabInterface_UsesTrueDistance)                // springy slab top at z = w, liquid
      // above: the interface edge sample has +0.5 / (w/2) on the liquid face and -0.5 / (w/2) on the slab's face
  TEST(ComputeMomentumFromVelocity, SymmetricWithMaterialAndSolids)          // x.(K y) == y.(K x), x.(K x) >= 0
      // over x = (u, w, xi), random, 1e-12 relative
  TEST(SolveViscosityAndPressure, ResponseIsLinearWithMaterialAndSolids)     // Review Focus 4, 1e-10 relative
  TEST(SolveViscosityAndPressure, NearlyInviscidWithSlabAndBox_EqualsSolvePressure)  // 1e-12 Pa s: pressure Task 49's
      // slab (springy) and pressure Task 23's free box: velocity, pressure, material velocities, box motion within
      // the pressure stop rule of SolvePressure's (which eliminates the slab faces and holds the box as the rank-one
      // term)
  TEST(SolveViscosityAndPressure, SlabAndBoxOutputsWrittenOnlyOnSuccess)     // cancel at outer 1: all six outputs
      // keep their pattern
  TEST(SolveViscosityAndPressure, MaterialOrSolidSpansWrongSize_IsBadInput)
  // Amended 2026-11-15 (A59); replaces GuidedBoxKeepsForbiddenMotion:
  TEST(SolveViscosityAndPressure, AxisGuidedBox_ForbiddenComponentsExactlyZero)  // the free box of the slab-and-box
      // scene with allowed_motions = ComputeAllowedMotions({kAxisLimits, z only, no turning}, origin) (one selection
      // column) and xi_before {0, 0, -0.01962}: solved V.x, V.y and every Omega component == 0.0 (bytes); V.z within
      // the velocity bound of the same solve with the 6 x 6 block's z row alone
  TEST(ComputeSolidBlockFactor, HingedBlock_IsJTransposeFullBlockJ)  // the slab-and-box scene's free box samples; J of a hinge
      // through origin + {0, 0, 0.05} about y (one column): the 1 x 1 block equals J^T K6 J of the six-slot block
      // formed with J = identity; 1e-13 relative
  TEST(SolveViscosityAndPressure, HingedPlank_EqualsProjectedFullBlock)  // closed box 0.2 m of liquid (1 Pa s,
      // 1000 kg/m3), w 0.0125, no gravity; plank half size {0.05, 0.0125, 0.025} m, density 2000 (0.25 kg), centred at
      // {0.1, 0.1, 0.1} (faces on cell faces), hinged at its centre about y (J = [0; e_y]); xi_before V {0.01, 0.02,
      // 0.03} m/s, Omega {0.1, 0.2, 0.3} rad/s (projected by the test with J (J^T M J)^-1 J^T M for the hinged run),
      // dt 0.005, one step; again with no guide (J = identity), its motion then projected the same way: turn rates
      // about y equal within 1e-6 relative; the hinged V and Omega x, z == 0.0 (bytes). Exact up to the solve
      // tolerance because the scene's three mirror symmetries decouple the turn about y from the other five motions
      // (plan decision: an off-centre hinge differs at first order in the added mass and is not compared)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step N:` comments. **Step 4: Run** the gate.
  **Commit.** Subject: `Solve materials and free solids with a viscous liquid that sticks`. New names: the A26/A27
  fields (reason "A26, A27").
- [ ] **Step 5 (amended 2026-11-15, A59): Paper.** In `<sec-viscosity-coupled>`: guided free solids in generalized
  velocities (`xi = J q + xi_0`, the m x m block, the coefficients `R J`), why the block stays symmetric positive
  definite, and that the solve never moves a forbidden direction; cite Batty 2007 ([21]) and Aanjaneya 2018 (URL in
  `literature-practice.md` topic 3). Run typst; look at the page.

### Task 15d: Scene tests, a springy slab dragged, a block on a tilted slab, rigid bodies in syrup

**Amended 2026-11-14 (A26, A27):** new task; derived analytic scenes on springy solids and rigid bodies (owner note:
no analytic liquid-on-snow test).

**Kind:** integration

**Files:**
- Create: `native/fluidity/viscosity/sticking_scene_test.cc`

The test loops step as pressure Task 49 does (geometry fixed where displacements stay below 0.2 w, rebuilt per
step for the rigid bodies); a free body's pose advances by its solved motion; the springy slab's elastic force is
the stand-in of pressure Task 49 (a uniform slab moving as one has none).

- [ ] **Step 1: Failing tests:**

  ```cpp
  // Couette over a movable springy slab: w 0.0025; a springy slab one cell thick (density 20000, so mass per area
  // sigma = 50 kg/m2) on a frictionless floor (friction 0), 64 x 16 cells in x, y, one liquid cell between each slab
  // end and the x walls; liquid H = 2 w = 0.005 m above it, mu 1 Pa s, rho 1000; a FlatSolid plate at the top moving
  // +x at U = 1e-3 m/s; dt 0.01, 50 steps (slab moves under 0.2 w: geometry fixed). Quasi-steady (the liquid's
  // diffusion time rho H^2 / mu = 0.025 s is a tenth of T): sigma_eff dV/dt = mu (U - V) / H with sigma_eff = sigma +
  // sum over the two liquid X faces of rho w (1 - z_i / H)^2 = 50 + 0.625 rho w = 51.5625 kg/m2, so T = sigma_eff H /
  // mu = 0.25781 s (continuous: sigma + rho H / 3 gives 0.25833 s); implicit per-step factor of U - V:
  // 1 / (1 + dt / T) = 0.96266.
  TEST(ViscosityScene, SpringySlabDraggedByShear_ReachesLiquidSpeed)  // slab velocity (mean over its X faces in the
      // middle third in x and y) per step within 2 % of a 1D implicit reference (slab node, two liquid nodes,
      // the plate; written in the test, about 20 lines) and V(T) == 0.632 U within 3 %; after t = 0.075 s the ratio
      // of U - V between steps == 0.96266 within 0.5 %
  // Viscous block on a tilted springy slab: gravity 9.81 at 30 degrees from -z toward +x; a springy slab (density
  // 1000, 0.02 m thick) on the floor with floor friction 0.7 (> tan 30 = 0.577: the slab stays); a liquid block
  // 0.04 x 0.04 x 0.02 m on it, w 0.0025, dt 0.005, 40 steps.
  TEST(ViscosityScene, ViscousBlockOnTiltedSlab_DoesNotSlide)   // mu 1000 Pa s, rho 1400: every liquid speed
      // <= 1.1 * rho g sin(30) h^2 / (2 mu) = 1.511e-3 m/s (the surface speed of a steady film of the block's height
      // h = 0.02 m with 10 % margin: a block that sticks at its base creeps no faster than that film); liquid faces
      // touching the slab within the velocity bound of the slab's faces
  TEST(ViscosityScene, InviscidBlockOnTiltedSlab_Slides)        // the same block with viscosity 0 (stage 12,
      // SolvePressure): centre-of-mass velocity along the slope == g sin(30) t = 0.4905 m/s at t = 0.1 s within 1 %
  // Rigid cube settling in syrup: cube side L 0.02 m (8 cells at w 0.0025), density 2800, in syrup 1400 kg/m3,
  // mu 10, box 0.12 x 0.12 x 0.2 m, from rest 0.06 m below the surface, dt 0.005, 40 steps; terminal speed = mean
  // over steps 20 to 40 (the start-up time (m + added mass) / drag ~ 7 ms). Stokes estimate: volume-equivalent
  // radius 0.012407 m times the cube's shape factor 1.08 gives drag 2.526 U N; buoyant weight 0.10987 N; unbounded
  // 0.04350 m/s; Faxen's cylinder correction at a/R = 0.2233 (the square box as a cylinder of radius 0.06) 0.5529:
  // 0.02405 m/s; Reynolds number 0.067.
  TEST(ViscosityScene, CubeSettlingInSyrup_StokesLinearity)     // runs at mu 10 and 20: speed ratio 2.00 within
      // 2 %; at density 2800 and 4200: ratio 2.00 within 3 % (Stokes flow is linear; inertia stays under 2 %)
  TEST(ViscosityScene, CubeSettlingInSyrup_NearWallCorrectedStokes)  // terminal speed within [0.7, 1.2] x 0.02405
      // m/s: a bound, not a match (the shape factor, the box treated as a cylinder and 8 cells across the cube each
      // carry several percent); prints the value and the ratio
  // Neutrally buoyant cube in shear: Couette between plates at z = +-0.05 m moving +-U, U 0.01 m/s (shear rate
  // 0.2 1/s), syrup 1400 kg/m3 and mu 10, cube side 0.02 m of density 1400 at the centre, w 0.0025, domain 0.16 x
  // 0.08 x 0.1 m, dt 0.01, 100 steps.
  TEST(ViscosityScene, NeutralCubeInShear_TurnsAtHalfTheShearRate)  // turn rate about y == 0.1 rad/s (half the shear
      // rate: exact for a body of cubic symmetry in Stokes flow, Jeffery's sphere result) within 5 % (walls two
      // sides away, 8 cells across), mean over steps 20 to 100; centre velocity <= 1e-3 U
  TEST(ViscosityScene, FloatingBoxInSyrup_SameRestAsInviscid)  // free box density 500, side 0.04, in a liquid of
      // density 1000 at mu 10, started at rest at draft 0.02 m: 200 steps of 0.01 s; centre within 1e-6 w of the
      // same scene at viscosity 0 (SolvePressure) each step; speed within the velocity bound 2 * 2e-5 * w
  TEST(ViscosityScene, BoxInLiquidBlock_MomentumChangedOnlyAtAirEnds)  // no gravity, no surface tension: a liquid
      // block 0.1 m with a free box 0.02 m inside moving at 0.01 m/s, and again with a springy cube inside: per step,
      // the change of sum(face_inertia u) + sum(face_mass w) + m V (and of the angular momentum about the block centre,
      // with I Omega) equals the impulse at the air ends (sum over air-end face pieces of the push from the accepted
      // pressure) within 1e-10 relative: interior pressure and viscous terms cancel exactly
  ```
- [ ] **Step 2: Run.** A failure is fixed in Task 15c (or the owning module), never by a looser bound; if the cube
  settles outside the bound, report the value to the owner before changing anything. **Step 3: Run** the gate.
  **Commit.** Subject: `Test sticking to springy solids and rigid bodies in syrup`.

### Task 15e: The viscous push on every solid, on the CPU

**Amended 2026-11-15 (A60):** new task (`plan-decisions.md` "Owner answers A40-A64 applied" item 6; design 3
`ViscosityResult::viscous_push_on_solids`, `max_solids`, 5 "Force on every solid", decision 26).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure.h` (`ViscosityBufferSettings::max_solids`,
  `ViscosityResult::viscous_push_on_solids`), `samples_of_cell.h`, `viscosity_samples_cpu.cc`, `velocity_solve_steps.h`,
  `solve_viscosity_and_pressure_cpu.cc`, `viscosity_buffers.cc`, `viscosity_memory_estimate.cc`,
  `viscosity_test_scenes.h`, `viscosity_test_scenes.cc`, `docs/method-paper/sections/viscosity.typ`
- Create: `native/fluidity/viscosity/liquid_force_test.cc`

**Interfaces:**
- Consumes: Tasks 10, 11, 15c; solids' `ViscousPushOnSolid { Vec3 push; Vec3 turning_push; }` (solids design 3.6;
  hand-built in tests, as every solids input here).
- Produces: design 3 `max_solids` and `viscous_push_on_solids` exactly.
- **Plan decisions:**
  - Build (steps 3, 5): rule (a) and the fitted-sample solid centre store, beside the sample, the solid index and the
    six `R` coefficients for a solid of **any** motion kind (fixed, driven, free, the domain box), scaled like the
    sample's face coefficients (`sqrt(dt W) x (-/+0.5 / d) x` the `R` row); a driven or fixed solid's known velocity
    still goes to `rate_from_solids`. A sample that reads no solid stores `contact::kNoSolid`. A sample reading two
    different solids stores the lower solid index; the other's term stays in `rate_from_solids` (design Open).
  - Step 12: per solid and cell block, `-sum over the block's samples with that solid index, in sample order, of
    (stored R row) x (stored rate + sqrt(dt W) x rate_from_solids)` into `"viscosity: solid viscous push partials"`;
    then per solid and component `math::TreeSum` over the block partials in block order into `"viscosity: viscous
    push on solids"`, copied once to its CPU array; `viscous_push_on_solids` views it (`inputs.solids.size()`
    entries). Shared per-item function `ComputeViscousPushPartialOfBlock` in `velocity_solve_steps.h` (Task 17a's GPU
    loop calls it). Written only on success.
  - `kBadInput`: `inputs.solids.size() > max_solids`.
  - Rows (one entry per array, A58): `"viscosity: solid viscous push partials"` (per cell block, `48 x max_solids` B,
    `kRunOn`, `kOneStepTry`), `"viscosity: viscous push on solids"` (given `max_solids`, 48 B, once `kRunOn` and once
    `kCpu`).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; Task 19's tight settings: `max_volume_error_rate 1e-9`,
  `velocity_tolerance 1e-13`):

  ```cpp
  TEST(BuildViscositySamples, DrivenPlate_StoresSolidIndexAndR)  // Task 10's channel: every sample of the plate edge
      // stores the plate's index and R = sqrt(dt W) * (+-0.5 / (h/2)) * {e_x, (y - origin) x e_x}; others kNoSolid; exact
  TEST(SolveViscosityAndPressure, FreeBox_PressurePlusViscousPushIsMomentumChange)  // Task 15c's slab-and-box
      // scene, the box free: pressure part (summed in the test: dt * sum over the box's FreeSolidPush entries of
      // (force_per_pressure, torque_per_pressure) * accepted pressure) + viscous_push == m (V_after - V_start) - dt
      // (m g + steady_force), and turning: == I_start (Omega_after - Omega_before_pressure); within
      // velocity_tolerance * largest momentum entry (the box's row of the inner solve)
  TEST(SolveViscosityAndPressure, ViscousPushOnSolids_OnePerSolidInOrder)  // Task 15d's neutral cube in shear: one
      // entry per inputs.solids entry, in order; the entry of a solid no sample reads is {0, 0} bytes
  // Sheared plate (literature-practice.md topic 4; ShearedPlateSettings, a new fixture of Task 8's library): liquid
  // x, y in [0, 0.2], z in [0, 0.1] m, w 0.0125 (16 x 16 x 8 cells); inlet and outlet given X faces at x = 0 and 0.2
  // carrying u = U z / H (as ShearChannel); one air layer beyond each y side; floor the domain box bottom; top a
  // driven FlatSolid at z = 0.1, normal -z, moving +x at U = 0.01 m/s; mu 10 Pa s, rho 1000, no gravity; velocity
  // starting at the linear profile (5 s from rest reaches it within exp(-pi^2 nu t / H^2) = exp(-49)), dt 0.01,
  // 10 steps.
  TEST(LiquidForce, ShearedPlate_IsMuAUOverH)   // viscous_push_on_solids of the plate / dt: x == -mu A U / H =
      // -0.04 N (A = 0.04 m2) within 1e-6 relative (the linear profile is exact on the grid); the floor's +0.04 N
      // within 1e-6 relative; y components 0 within 1e-12 N
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Paper.** In `<sec-viscosity-coupled>`: the viscous push
  on a solid read from the solve's own terms (momentum exchange; Uhlmann 2005 and Batty 2007, URLs in
  `literature-practice.md` topic 4), torque about the step-start origin, the guide's reaction included. **Step 5:
  Run** the gate. **Commit.** Subject: `Report the viscous push of the liquid on every solid`. New names:
  `max_solids`, `viscous_push_on_solids` (reason "A60").

### Task 15f: Slip length on fixed and driven solids, on the CPU

**Amended 2026-11-15 (A53):** new task (`plan-decisions.md` "Owner answers A40-A64 applied" item 7; design 3
`ViscosityInputs::solid_slip_length`, 5 "Slip length", decision 27).

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure.h` (`ViscosityInputs::solid_slip_length`),
  `samples_of_cell.h`, `viscosity_samples_cpu.cc`, `solve_viscosity_and_pressure_cpu.cc`, `viscosity_test_scenes.h`,
  `viscosity_test_scenes.cc` (`ViscousSceneSettings::solid_slip_length`, `std::vector<double>`, empty: all 0),
  `docs/method-paper/sections/viscosity.typ`
- Create: `native/fluidity/viscosity/slip_length_test.cc`

**Interfaces:**
- Consumes: Tasks 10, 11, 15e; solids' `SolidsTry::slip_length()` and `SolidSettings::slip_length` (hand-built here).
- Produces: design 3 `solid_slip_length` exactly.
- **Plan decisions:** rule (a) with `lambda = solid_slip_length[solid_index]` (0 when the span is empty): coefficient
  `-/+0.5 / (d + lambda)`, `rate_from_solids -= +-0.5 x v_solid / (d + lambda)`, the stored `R` row (Task 15e) with
  the same divisor; `lambda = +infinity` gives exactly 0 by IEEE division (no branch). Fitted samples: the known
  velocity at the liquid-covered solid centre enters with that centre moved `lambda` along the solid's normal out of
  the liquid (an infinite `lambda` drops the position from the fit). Free solids and material faces use `d` (their
  span entry is 0 by solids' contract; this module does not read it for them). `kBadInput`: span size neither 0
  nor `solids.size()`, an entry negative or NaN (outputs untouched).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; `ShearChannelSettings(kSameSize, 16, ratio 1, top_speed, dt 0.005,
  profile)`, H = 1 m; 10 steps; Task 19's tight settings):

  ```cpp
  TEST(SolveViscosityAndPressure, CouetteFullStick_LinearProfile)  // slip lengths 0, top_speed U = 1, profile
      // u = U z / H: every X face u == U z / H within 1e-6 relative
  TEST(SolveViscosityAndPressure, CouetteFreeSlip_PlugFlow)  // floor (domain box) and plate slip length +infinity,
      // plate at rest (top_speed 0), profile and start 0.01 m/s everywhere: every X face 0.01 m/s within 1e-9
      // relative; the plate's viscous_push_on_solids / dt == 0 within 1e-12 N
  TEST(SolveViscosityAndPressure, CouettePartialSlip_NavierProfile)  // plate slip length lambda = H = 1 m, floor 0,
      // U = 1, profile u = U z / (H + lambda): every X face u == U z / (H + lambda) within 1e-6 relative; u(H) == U / 2
  TEST(SolveViscosityAndPressure, SlipLengthSpanWrongSizeOrNegative_IsBadInput)  // one short; -1; NaN: kBadInput,
      // outputs untouched
  TEST(BuildViscositySamples, InfiniteSlipLength_PlateCoefficientIsZero)  // the plate edge sample: coefficient on the
      // plate side and its rate_from_solids part both 0.0; exact
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Paper.** In `<sec-viscosity-solids>`: the slip length
  as a wall displaced by `lambda` (Navier `tau = (mu / lambda) u_slip`), a positive boundary energy term, so the
  system stays symmetric positive definite; free solids always stick. **Step 5: Run** the gate. **Commit.** Subject:
  `Let thick liquid slip along a solid by a set length`. New names: `solid_slip_length` (reason "A53").

### Task 15g: Scene tests of the force on solids

**Amended 2026-11-15 (A74):** contact provides `kSphere` and `kCylinder`; the rotating-cylinder test (renamed
`RotatingCylinder_TorquePerLength`, square outer wall) and the settling sphere (renamed back from `SettlingCube_...`)
use round shapes; `docs/known-open-physics.md` gains no row for them (remove it if an earlier commit added one).

**Amended 2026-11-15 (A60):** new task; the two remaining analytic force tests of `plan-decisions.md` item 6 that need
the joint solve (the buoyancy test is solids Task 36a, the sheared plate Task 15e).

**Kind:** integration

**Files:**
- Modify: `native/fluidity/viscosity/liquid_force_test.cc`, `docs/known-open-physics.md`

- [ ] **Step 1: Tests** (`LABEL cpu`; Task 15d's stepping loop):

  ```cpp
  // Amended 2026-11-15 (A74): contact has a sphere and a cylinder; both tests use round shapes (plan-decisions
  // "Sphere and cylinder in contact (A74)"). Neither is known-open.
  // Rotating cylinder: driven cylinder axis z, R1 0.05 m, half height 0.06 (through both ends), Omega 1 rad/s
  // (a spin about its axis: contact P17); outer wall the domain box [-0.1,0.1]^2 x [0,0.1] m (a square of side
  // 2 R2, R2 0.1: contact's cylinder is solid, so no hollow outer cylinder), sides stick, bottom and top slip length
  // +infinity; mu 1 Pa s, rho 1000, w 0.005 (10 cells across R2 - R1), no gravity; start from the Taylor-Couette
  // profile u_theta = A r + B / r for R2 inside r <= R2 and at rest in the corners; dt 0.01, 10 steps.
  TEST(LiquidForce, RotatingCylinder_TorquePerLength)  // the cylinder's viscous turning push / dt, z:
      // -4 pi mu Omega R1^2 R2^2 / (R2^2 - R1^2) * 0.1 = -4.18879e-3 N m within 10 % (a bound: Uhlmann reports 5 to
      // 10 % at this resolution; in 0.1 s the profile changes only within Sqrt(nu t) = 0.01 m of the corners'
      // disturbance, two cells, so the cylinder still sees the Taylor-Couette field; the steady square's torque lies
      // between 3.590e-3 and 4.189e-3 N m)
  // Settling sphere: free sphere r 0.01 m, density 1100 (mass 4.60767e-3 kg, inertia 0.4 m r^2 = 1.84307e-7 kg m2 on
  // each axis), in liquid 1000 kg/m3, mu 10 Pa s, closed box 0.2 m, w 0.004 (5 cells across the sphere), from rest at
  // the box centre, dt 0.002, 40 steps; means over steps 20 to 40.
  TEST(LiquidForce, SettlingSphere_BalancesNetWeight)  // pressure part (summed as in Task 15e) + viscous push, z, / dt
      // == net weight (1100 - 1000) * 9.81 * 4.18879e-6 = 4.1092e-3 N within 2 % (a balance); settling speed within
      // [U_Stokes / 1.4, U_Stokes], U_Stokes = 4.1092e-3 / (6 pi * 10 * 0.01) = 2.180e-3 m/s (a bound: the walls slow
      // it; Faxen's factor at r / L = 0.1 is about 0.79)
  ```
- [ ] **Step 2: Run.** A failure of the settling sphere or the rotating cylinder is reported to the owner with the measured value before
  anything changes. **Step 3: Run** the gate. **Commit.** Subject: `Test the force of thick liquid on solids`.

### Task 16: Every error and cancel, on the CPU

**Amended 2026-11-14 (A24):** the coupling spans in the `kBadInput` case; four exclusions.

**Amended 2026-11-15 (A53, A60):** the `kBadInput` case also covers a `solid_slip_length` span of the wrong size and
`inputs.solids.size() > max_solids`; `kOutOfMemory` (now the last value) is `BuildViscosityBuffers`' and is tested in
Task 7.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** every case here is a failed check, a solver failure or a cancel, so
"outputs untouched" holds; the `no_gpu` `kRunOnCallFailed` case fails at the check that the GPU can run, before
any write. A failed GPU call on hardware leaves the outputs unspecified (Global Constraints) and is not tested
here.

**Files:**
- Create: `native/fluidity/viscosity/viscosity_errors_test.cc`

**Interfaces:**
- Consumes: Tasks 8 to 15.

- [ ] **Step 1: Tests** (design 7 "Each error" and "Cancel"). Each case asserts the error, the failing
  `CellId` (or 0) and that both outputs are byte-identical to their state before the call (filled with a fixed
  pattern first):

  ```cpp
  TEST(SolveViscosityAndPressure, EveryError_ReportedWithCellAndOutputsUntouched)  // one SCOPED_TRACE per value:
      // kBadInput: null cell_eighths; block_range short of the grid by one block; null pressure_buffers; (A24)
      // a coupling span of the wrong size; (A26, A27) a material or solid span of the wrong size; (A53, A60) a
      // solid_slip_length span one short; more solids than max_solids.
      // kTooFewNearbyFaces: Task 11's one-cell-thick grid. kLiquidAtEdgeNotConnected, kNoFacePairForShear:
      // Task 10's scenes. kNotFinite: 1e308 Pa s. kVelocitySolveDidNotConverge: max_velocity_iterations 1 on
      // the shear wave. kPressureSolveDidNotConverge: max_pressure_iterations 1 on PlanarExtension ratio 1000.
      // kBuffersTooSmall: max_sample_count 64. kRunOnCallFailed: in viscosity_errors_no_gpu_test.cc (no_gpu).
      // kCancelled: below. The three exclusions are CheckViscosityCanRun's (Task 6) and are not returned here.
  TEST(SolveViscosityAndPressure, CancelBeforeFirstInnerSolve_OutputsUntouched)
  TEST(SolveViscosityAndPressure, CancelInThirdOuterIteration_OutputsUntouched)   // Review Focus 5: PlanarExtension
      // n 32, ratio 1000, carried 0 (first asserted to take >= 3 outer iterations); a BeforeInnerSolve hook sets
      // the flag at outer_iteration 3: kCancelled, outputs untouched
  TEST(SolveViscosityAndPressure, NullCancel_SameBytesAsUnsetFlag)
  ```
- [ ] **Step 2: Run.** Expected: pass (behaviour implemented in Tasks 9 to 15); a failure is fixed in the owning
  commit. **Step 3: Run** the gate. **Commit.** Subject: `Test every viscosity error and cancel`.

### Task 17: The joint solve on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `solve_viscosity_and_pressure_gpu.h` returning
`RunOnCallStatus`; vector updates and block partials are shared per-item functions with a `cpu` test over every
thread block and thread; dots by block partials and `ComputeTreeSum`, largest values by `ComputeLargestMagnitude`,
both with `buffers.sum_buffers`; digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/viscosity/solve_viscosity_and_pressure_gpu.h` (private),
  `solve_viscosity_and_pressure_gpu.cu`, `solve_viscosity_and_pressure_steps_test.cc` (label `cpu`),
  `solve_viscosity_and_pressure_gpu_test.cc`, `solve_viscosity_and_pressure_no_gpu_test.cc`
- Modify: `velocity_solve_steps.h` (the vector updates), `velocity_solve_gpu.h`, `velocity_solve_gpu.cu` (the inner
  solve's launches), `solve_viscosity_and_pressure_cpu.cc` and `velocity_solve_cpu.cc` (dispatch under
  `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

**Interfaces:**
- Steps 6 to 12 stay on the GPU; vector updates one thread per element, each a grid-stride loop around a shared
  per-item function of `velocity_solve_steps.h` (plan decision: `ApplySearchStepOfFace` (velocity and error
  updates of one face), `ApplyDirectionUpdateOfFace`, `ApplyPressureUpdateOfCell`, `ComputeDiagonalOfFace`), the
  same functions the CPU loops of Tasks 14 and 15 are changed to call; dots by `ComputeDotPartialOfBlock` per cell
  block, then `ComputeTreeSum(partials, block_count, buffers.sum_buffers)` (same bits as the CPU's `math::TreeSum`
  over the same partials, by `ordered_sums.h`'s promise); largest values by `ComputeLargestMagnitude(values,
  count, buffers.sum_buffers)`; the pressure pieces run on the GPU through pressure's own dispatch. Per outer
  iteration the CPU reads one record (two imbalance norms, inner iteration count, error code) and reads
  `*cancel_requested` before the next inner solve. The GPU half's functions in
  `solve_viscosity_and_pressure_gpu.h` return `RunOnCallStatus`; launches by `ApplyGpuFunction` and
  `ComputeLaunchSize`, waits by `CheckGpuWork`; every array is in `ViscosityBuffers` (no `GpuArray`).

- [ ] **Step 1: Tests:**

  ```cpp
  // solve_viscosity_and_pressure_steps_test.cc, LABEL cpu (the main evidence until the hardware run)
  TEST(ApplySearchStepOfFace, EveryThreadOfLaunch_EqualsCpuLoop)  // and ApplyDirectionUpdateOfFace,
      // ApplyPressureUpdateOfCell, ComputeDiagonalOfFace, ComputeDotPartialOfBlock: Task 15's shear-wave scene,
      // every thread block and thread of ComputeLaunchSize(count), launch order and reverse: arrays equal the
      // CPU loop's; bytes
  // LABEL gpu
  TEST(SolveViscosityAndPressure, OnGpu_EqualsCpu)       // Task 15 cases: velocity 1e-9 m/s + 1e-7 * largest speed,
                                                         // pressure 1e-6 Pa + 1e-6 * largest pressure
  TEST(SolveViscosityAndPressure, OnGpuCancel_ReturnsCancelledOutputsUntouched)  // cancel is read on the CPU
                                                         // between outer iterations; no GPU call failed
  TEST(ComputePowerLostToViscosity, OnGpu_EqualsCpu)
  // LABEL no_gpu
  TEST(SolveViscosityAndPressure, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
  CPU against GPU: iteration counts may differ (an operator use is not promised same bits, so the stop test can
  fall one iteration apart); the solved arrays are compared within section 6, not by bits. Every `gpu` test calls
  `WriteResultDigest` on velocity, pressure and power lost.
- [ ] **Step 2: Implement**; nothing allocates. **Step 3: Run** the gate. **Commit.** Subject:
  `Solve drag and pressure together on the GPU`. Body: `CUDA: compiled, not run`. Lists the replaced one-thread
  solve of `native/cuda/viscosity_velocity_block_3d.cu` as not ported.

### Task 17a: The coupled terms on the GPU

**Amended 2026-11-14 (A24):** new task; GPU form of Task 15a.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the GPU functions wrap Task 15c's shared per-solid functions; a
`cpu` test runs them over every thread block and thread; digests in the `gpu` tests.

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure_gpu.cu`, `solve_viscosity_and_pressure_gpu.h`,
  `solve_viscosity_and_pressure_steps_test.cc`, `solve_viscosity_and_pressure_gpu_test.cc`

**Interfaces:** blob vectors on the GPU in the buffers; the blob part of each dot one `math::TreeSum` over blobs in
one thread (a GPU function of one thread around the same `math::TreeSumOf` call the CPU makes); the pressure A24 pieces through pressure's own dispatch; the per-outer-iteration record unchanged;
the two spans written as `SolvePressure` writes them (pressure Task 48).

**Amended 2026-11-14 (A26, A27):** also Task 15c's unknowns (material entries as face-like destinations, each
solid's sums by the per-group rule, its 6 x 6 factor on the GPU) and Tasks 15c, 15d scenes.

**Amended 2026-11-15 (A53, A59, A60):** also the m x m block and the `R J` products (Task 15c as amended), the stored
solid coefficients of every motion kind and `ComputeViscousPushPartialOfBlock` with its per-solid tree sum and one
counted read of `48 x max_solids` bytes (Task 15e), and the slip divisor `d + lambda` (Task 15f, inside the shared
`BuildSamplesOfCell`). `cpu` test `ComputeViscousPushPartialOfBlock, EveryThreadOfLaunch_EqualsCpuLoop` (Task 15e's
sheared plate and Task 15d's neutral cube; launch order and reverse; bytes); the `gpu` tests add Tasks 15e, 15f,
15g scenes and `WriteResultDigest` on `viscous_push_on_solids`.

- [ ] **Step 1: Tests** (`LABEL cpu`, in `solve_viscosity_and_pressure_steps_test.cc`, the main evidence until the
  hardware run): `ComputeSolidSumOfSolid, EveryThreadOfLaunch_EqualsCpuLoop` and
  `ComputeSolidBlockFactor, EveryThreadOfLaunch_EqualsCpuLoop` (Task 15c's settling-cube and free-box scenes, every
  thread block and thread, launch order and reverse; bytes). (`LABEL gpu`):
  `SolveViscosityAndPressure, CoupledOnGpu_EqualsCpu` (Task 15a, 15b, 15c, 15d scenes,
  one step; section 6 bounds for velocity and pressure; material velocities with the velocity bound,
  `sealed_gas_pressure` with the pressure bound), `SolveViscosityAndPressure, CoupledOnGpuTwice_GivesIdenticalBytes`;
  both call `WriteResultDigest` on the four coupling spans.
  (`LABEL no_gpu`): `SolveViscosityAndPressure, CoupledOnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement.** **Step 3: Run** the gate. **Commit.** Subject: `Carry blocking materials and sealed gas
  in the joint solve on the GPU`. Body: `CUDA: compiled, not run`.

### Task 18: Scene tests, two viscosities and sticking

**Kind:** integration

**Files:**
- Create: `native/fluidity/viscosity/viscosity_scene_test.cc`

Force per area of a shear sample (**Plan decision**, as `main` normalises it): `2 x mu_s x rate`,
`mu_s = sample_weight / (4 x liquid volume of its quarters)`; of a stretch sample: `2 x viscosity x rate -
pressure`.

- [ ] **Step 1: Tests:**

  ```cpp
  TEST(ViscosityScene, LinearShearTwoViscosities)     // design 7: ShearChannel, both layouts, ratios 1, 10, 1000,
      // n 16 and 32, dt 0.005 and 0.0025 (24 cases): every XZ sample's surface force per area within 5 % of
      // the scale 2 of 2 / (1 + 1/ratio); every stretch sample's normal force per area within 5 % of 2 of 0
  TEST(ViscosityScene, PlanarExtensionTwoViscosities) // design 7, PlanarExtensionSettings (Task 8 plan decision),
      // same 24 cases: normal force per area F = 2 mu rate_x - pressure in the two cell columns beside x = 0.5
      // is 0 on both sides (continuous; the exact state has p = 2 mu a): error |F| / 2, rms per side <= 0.05,
      // largest <= 0.10; XZ samples one column either side: |force| / 2 <= 0.01
  TEST(ViscosityScene, SticksToSolids)               // design 7: ShearChannel ratio 1, u_x = z, plate moving at 1:
      // the X faces nearest the plate move at 1 - h/2 and those nearest the floor at h/2 (1e-10 relative); the
      // kSizeChange channel gives the same velocity at matching face positions, 1e-10 relative
  ```
- [ ] **Step 2: Run.** Expected: pass. A failure is fixed in the owning commit, never by loosening a tolerance;
  if `PlanarExtensionTwoViscosities` fails because of the boundary arrangement, stop and ask the owner.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Test two viscosities in shear and extension`.

### Task 19: Scene tests, energy, the free surface, rest, and samples built per range

**Amended 2026-11-14 (A15):** `OnePieceAndBatched_SameBytes` becomes `SamplesPerRange_SameBytesAsWholeGrid` (a
local build compared by range); the batched solve against one piece is Task 27, within tolerance.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/viscosity/viscosity_scene_test.cc`

- [ ] **Step 1: Tests:**

  ```cpp
  TEST(ViscosityScene, EnergyBalance)                 // design 7: ShearChannel ratio 1 at rest, plate at 1 m/s,
      // one step: dKE == dt * solid work - dt * power lost + pressure work - 0.5 * (u - u0).M(u - u0), with
      // M = face_inertia on solved faces, solid work = sum_s W_s g_s (E u + g)_s, pressure work = u . push of
      // the accepted pressure (kIncluded; Plan decision: the design's line leaves it out, it is 0 only to the
      // pressure tolerance); 1e-10 relative to the largest term
  TEST(ViscosityScene, FreeSurfaceStretchedSlab)     // design 7: w 0.05, liquid x in [-0.2, 0.2], y 4 cells,
      // z from -0.2 to the surface z = 0.0125 (the top liquid cells [0, 0.05] are a quarter full), u = (a x, 0,
      // -a z), a 0.5, viscosity 2, no gravity; atmosphere above; tracked gas on the other sides (x sides
      // -4 mu a, y sides -2 mu a, bottom 0, Pa above atmosphere): pressure in the top cells == -2 mu a = -2 Pa
      // within 5 %
  TEST(ViscosityScene, SyrupAtRest)                  // design 7: SyrupColumn both layouts, pressure_acceptance
      // max_volume_error_rate 1e-9 and velocity_tolerance 1e-13 (Plan decision: the default 2e-5 1/s allows
      // about 2e-6 m/s): every speed after <= 1e-9 m/s
  TEST(ViscosityScene, SamplesPerRange_SameBytesAsWholeGrid)  // design 7: LinearShearTwoViscosities n 32 ratio
      // 10: BuildViscositySamples per range of 4 blocks (grid::BuildRanges), each into its own buffers; their
      // samples concatenated in range order equal the whole-grid samples, bytes
  ```
- [ ] **Step 2: Run.** If `FreeSurfaceStretchedSlab` reads `-2 mu a x filled_volume / open_volume` in the top
  cells (the weight on `filled_volume` with the divisor `open_volume` predicts this), stop and ask the owner:
  the design's expectation and its weight rule then disagree. **Step 3: Run** the gate. **Commit.** Subject:
  `Test energy, the free surface and rest for thick liquid`.

### Task 19a: Scene test, syrup on a snow bed

**Amended 2026-11-14 (A24):** new task; the former known-open `ViscosityKnownOpen, SyrupOnSnowBed_StaysAtRest`
(Task 20) passes now that snow's inertia enters the joint solve, so snow is never excluded.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/viscosity/coupled_scene_test.cc`

- [ ] **Step 1: Test:**

  ```cpp
  TEST(ViscosityScene, SyrupOnSnowBed_StaysAtRest)   // SyrupColumn (both layouts, Task 19 settings: max_volume_
      // error_rate 1e-9, velocity_tolerance 1e-13) whose lower half is a snow bed: a blocking set of density 300,
      // its top Z pieces material faces (BuildBlockingMaterialFacesForTest, face mass 300 * w^3, velocity 0) and
      // the bed's coverage at rest; one step: every liquid speed <= 1e-9 m/s; every material velocity <= 1e-9 m/s;
      // the bed carries the syrup's weight: push per area of the bed's faces == -(1400 g h) at the bed top within
      // the pressure bound
  ```
- [ ] **Step 2: Run.** A failure is fixed in Task 15a. **Step 3: Run** the gate. **Commit.** Subject: `Test syrup
  resting on a snow bed`.

### Task 20: (removed)

**Amended 2026-11-14 (A27):** removed. Its free-solid test is no longer known-open: the free solid is in the joint
solve (Task 15c) and the momentum exchange is tested in Task 15d (`BoxInLiquidBlock_MomentumChangedOnlyAtAirEnds`).
No known-open test remains in this module unless a lifting task of 21 to 23 falls back to one.

### Task 21: Lift the exclusion for an inclined solid

**Kind:** integration

**Files:**
- Modify: `native/fluidity/viscosity/viscosity_error.h`, `viscosity_error_message.cc`, `viscosity_error_test.cc`,
  `solve_viscosity_and_pressure.h`, `viscosity_capability.cc`, `viscosity_capability_test.cc`,
  `viscosity_scene_test.cc`, `docs/method-paper/sections/viscosity.typ`

The design gives the method (fitted samples with the liquid-covered solid centre, Task 11); this task proves it
and removes the line.

- [ ] **Step 1: Test first:**

  ```cpp
  TEST(ViscosityScene, ShearBetweenPlatesTilted30Degrees)  // lifts kInclinedSolidNotSupported: two FlatSolids
      // 0.25 m apart, normals tilted 30 degrees about Y, the upper moving at 0.5 m/s along its plane, w 1/32,
      // viscosity 1, density 1000, dt 0.005, velocity starting at the linear profile between them: surface force
      // per area of the fitted samples within the middle third of the channel == mu U / gap = 2 Pa within 5 %
  ```
  Run it. If it does not pass, do not loosen it: commit it as known-open (Task 20's form) and stop; report the
  measured error to the owner.
- **Amended 2026-11-15 (reconcile item 31):** after each lift `kOutOfMemory` stays the last value (it moves up by
  one); `ViscosityError, OutOfMemory_IsTheLastValue` checks the new index (13 after this task, 12 after Task 22, 11
  after Task 23).
- [ ] **Step 2: Lift.** Delete `ViscosityError::kInclinedSolidNotSupported`, its message, its line in
  `CheckViscosityCanRun` and `ViscositySceneSummary::has_inclined_solid` (**Plan decision**: a field nothing
  reads is deleted); update the Task 3 and Task 6 tests (14 values with `kOutOfMemory`; two exclusions; **A27**: one lower again); the paper's exclusion
  table. `grep -rn "InclinedSolidNotSupported\|has_inclined_solid" native docs` finds nothing. (**Amended
  2026-11-14 (A24):** counts one lower than first written, no snow value.)
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Let thick liquid touch inclined solids`. Body lists the
  removed public names and that the step's `CapabilityReason::kViscousLiquidWithInclinedSolid` is no longer needed.

### Task 22: Lift the exclusion for a driven solid cutting cells

**Kind:** integration

**Files:**
- Modify: as Task 21 (enum, message, capability, their tests, scene test, paper)

- [ ] **Step 1: Test first:**

  ```cpp
  TEST(ViscosityScene, DrivenPlateCuttingCells_ShearMatches)  // lifts kMovingSolidNotSupported: ShearChannel
      // n 32 with the plate at z = 0.53 (cuts cells) moving +x at 1 m/s, floor fixed, velocity starting at
      // u_x = z / 0.53, ratio 1: surface force per area of floor and plate samples in the middle third (x faces
      // away from the inlets) == 1 / 0.53 Pa within 5 %
  ```
  Same rule as Task 21 if it fails.
- [ ] **Step 2: Lift** `kMovingSolidNotSupported` and `has_driven_solid_through_cells` as in Task 21 (13 values with `kOutOfMemory`;
  **amended 2026-11-14 (A24, A27)**: counts two lower, no snow and no free-solid value).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Let driven solids cut cells of thick liquid`.

### Task 23: Lift the exclusion at a size change

**Kind:** integration

**Files:**
- Modify: as Task 21

- [ ] **Step 1: Tests first:**

  ```cpp
  TEST(ViscosityScene, LinearShearViscosityChangeOnSizeChange)       // lifts kSizeChangeNotSupported:
      // LinearShearTwoViscosities with the size change at z = 0.5 (cells above 2/n wide), ratios 1, 10, 1000,
      // n 16 and 32, dt 0.005: forces within 5 % of 2
  TEST(ViscosityScene, PlanarExtensionViscosityChangeOnSizeChange)  // PlanarExtension with cells x > 0.5 twice
      // as wide: 5 % rms, 10 % largest, sideways 1 %
  ```
  Same rule as Task 21 if either fails.
- [ ] **Step 2: Lift** `kSizeChangeNotSupported` and `has_size_change_at_surface_solid_or_viscosity_change`
  (12 values with `kOutOfMemory`; no exclusion remains, `CheckViscosityCanRun` then always succeeds and stays for the step's call;
  **amended 2026-11-14 (A24, A27)**).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Let thick liquid meet a change of cell size`.

### Task 24: GPU scene runs and the repeat run

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the CPU repeat test is `LABEL cpu`; the `gpu` tests write result
digests so the hardware run's repeat check covers them.

**Files:**
- Create: `native/fluidity/viscosity/viscosity_scene_gpu_test.cc`
- Modify: `native/fluidity/viscosity/viscosity_scene_test.cc`

- [ ] **Step 1: Tests:**

  ```cpp
  // LABEL cpu
  TEST(ViscosityScene, RepeatRun_IdenticalBytes)          // Tasks 15b, 18-19, 19a, 21-23 scenes twice: samples,
                                                          // velocity, pressure, power_lost, (A24) both spans, (A26, A27) Task 15d scenes and all four spans bytes
  // LABEL gpu
  TEST(ViscosityScene, OnGpu_EqualsCpu)                   // every scene of Tasks 15, 15b, 15d, 18, 19, 19a, 21-23
                                                          // (A24, A26, A27 amended), section 6 tolerances
  TEST(ViscosityScene, OnGpuRepeatRun_IdenticalBytes)     // design 7 GPU row: two-viscosity shear, two runs
  TEST(ViscosityScene, SyrupSizing_Converges)             // decision 6 grid at 2.5 mm (0.4 x 0.4 x 0.3 m box, a
      // settled 1 litre layer): converges; prints iterations and milliseconds per outer iteration
  ```
  Each `gpu` test calls `WriteResultDigest` on velocity, pressure, samples and the coupling spans of each scene.
- [ ] **Step 2: Run** the CPU test and the gate. **Commit.** Subject: `Check repeat runs and list the GPU
  viscosity runs`. Body: `CUDA: compiled, not run`.

### Task 25: Comparison with the proof of concept

**Amended 2026-11-14 (A31):** the old numbers are produced in a temporary directory.

**Kind:** integration

**Files:**
- Create: `native/fluidity/viscosity/testdata/proof_of_concept.json`, `proof_of_concept_test.cc`

- [ ] **Step 1: Produce the old numbers** (plan-decisions: `git archive main` into a temporary directory, build there,
  never commit the driver). The driver builds `main`'s two fixtures of `native/tests/mac_viscous_traction_cases.h`
  for `n = 16`, `dt = 0.005`, ratio 10, and records (a) `AssembleFullCellMacStrain3D` rows of interior cells
  (cells whose four edge cells exist): coefficients and weights; (b) for the Couette fixture only, with the
  driver adding `MakeMacTangentialWallStrain3D` rows (wall speed 0, coefficient 1/h) on the old-Z walls and the
  same half-cell rows in place of the inlet rows, so that `main` sticks wherever this module sticks: face
  velocities and cell pressures of `SolveCoupled`. Rotate into Z-up with rows `{1,0,0},{0,0,-1},{0,1,0}` and
  store with the `main` commit hash. The driver's outline goes into the commit body.
- [ ] **Step 2: Test** `ProofOfConcept, MatchesMainWhereTheMethodIsUnchanged`: (a) for both fixtures, interior
  stretch and edge samples match by face position at 1e-12 relative (stored coefficients divided back);
  (b) the matching channel (2 x 1 x 16 cells, given inlet faces, plate and floor sticking, y walls sticking)
  velocity and pressure within 1e-6 relative. Supporting evidence only; every quantity is also covered above.
  The planar-extension solve is not compared: its boundary changed (Task 8 plan decision).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Compare the viscosity module with the proof of concept`.

### Task 26: Batched layout of the viscosity buffers, samples built batch by batch, on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** new task.

**Amended 2026-11-14 (reconcile):** adds `ResetSolveCopyReport(ViscosityBuffers&)` (the step resets the copy report
through the buffers; `FindSolveBatches` stays const).

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the sum and order buffers are sized for the loaded piece in the batched
layout; the per-load grouping uses `buffers.order_buffers`; tests are `LABEL cpu`.

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure.h`, `viscosity_buffers.cc`,
  `viscosity_memory_estimate.cc`, `viscosity_buffers_access.h`, `viscosity_samples_cpu.cc`, `velocity_solve_cpu.cc`
- Create: `viscosity_batched_test.cc`, `viscosity_batched_no_gpu_test.cc`

**Interfaces:**
- Consumes: Tasks 7, 9 to 12; pressure Tasks 55, 56 (`SolveBatches`, `ApplySolveBatches`, `LoadSolveBatch`,
  `WriteBatchValues`, `ReadBatchValues`, `WriteBatchGroups`, `ReadBatchGroups`, `ReadSolveCopyReport`) and
  `pressure::testing::CountBlocksInLargestLayer`.
- Produces: `ViscosityBufferSettings::max_loaded_blocks` (`std::uint32_t`, 0: one piece),
  `const pressure::SolveBatches* FindSolveBatches(const ViscosityBuffers&)`, `const pressure::SolveCopyReport*
  ReadSolveCopyReport(const ViscosityBuffers&)`, `void ResetSolveCopyReport(ViscosityBuffers&)` (reconcile: zeroes
  the held `SolveBatches`' report, does nothing in one piece; the step calls it per try), and the batched path of
  `BuildViscositySamples`,
  `ComputeSampleRates`, `ComputeMomentumFromVelocity`, `ComputePowerLostToViscosity`.
- **Amended 2026-11-14 (A24):** the sealed gas blob vectors stay whole on the processor (`kRunOn`) in the batched
  layout, like pressure's `D_j` and `r_j`.
- **Amended 2026-11-15 (A58):** the `kCpu` rows of the batched layout are pieces of `cpu_run_memory` (its block is
  `CpuMemory::kLockedForCopies` in a batched GPU run), the `kRunOn` rows of `run_memory`; the held
  `pressure::SolveBatches` is built with pressure's `RunMemory&` form of `BuildSolveBatches` from the same blocks.
  `BuildViscosityBuffers, IntoRunMemory_HandsOutExactlyTheEstimate` gains a batched case (64 blocks, loaded 16): both
  blocks sized by `ComputeRunMemoryBytes(..., kRunOn)` and `(..., kCpu)`, each handed out exactly.
- **Plan decisions:** with `max_loaded_blocks > 0`: every Task 7 row counted per cell place, face place, sample or
  coefficient (the kept directions and responses included) is `where = kCpu` (`CpuMemory::kLockedForCopies`) at
  the limits and again `kRunOn` at `max_loaded_blocks` (samples `15 * 64` and coefficients `8 * 15 * 64` per
  loaded block); block-counted rows and the progress record stay `kRunOn`; `pressure::BuildSolveBatches(run_on,
  max_cell_count / 64, max_loaded_blocks, {kTouchingCells, kTouchingCells})` is held inside. Samples are built per
  batch for the range cells (count per block, `ComputeExclusivePrefixSum(..., buffers.order_buffers)` over blocks for the whole sample
  index, write)
  and kept grouped by block (the `samples of block starts` row is the begin offsets). The by-face view of a loaded
  piece is rebuilt at each load from its loaded coefficients (`BuildGroupsByDestination(...,
  buffers.order_buffers)`, piece sample indexes, which follow the whole order because pieces keep the block order).
  Power lost: block partials of the range, then `math::TreeSum` over every block.
  **Amended 2026-11-14 (Foundation as built):** in the batched layout the sum and order buffers stay `kRunOn` and are sized
  for the largest call: sum buffers `max(240 * max_loaded_blocks, max_cell_count / 64)` (largest values of a loaded
  piece; tree sums over the whole grid's block partials), order buffers `max(64 * max_loaded_blocks + 1,
  8 * 15 * 64 * max_loaded_blocks, max_cell_count / 64 + 1)` (cells of a batch, coefficients of a loaded piece,
  the block prefix sum of the whole sample index); their four rows take these counts.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; everything on the CPU; the loaded piece is 3 and
  5 times `CountBlocksInLargestLayer`):

  ```cpp
  TEST(ComputeViscosityBufferBytes, BatchedLayout_RowsSplitByWhere)     // 64 blocks, loaded 16
  TEST(BuildViscositySamples, InBatches_SameSamplesAsOnePiece)          // LinearShearTwoViscosities n 32 ratio 10
      // (8 block layers) and the floating cube on a grid lengthened to 8 layers: samples, coefficients, order,
      // bytes (a local build)
  TEST(ComputeMomentumFromVelocity, InBatches_SameAsOnePiece)           // random velocity, same scenes: bytes
  TEST(ComputePowerLostToViscosity, InBatches_SameAsOnePiece)           // bytes
  TEST(BuildViscositySamples, InBatchesTooManySamples_IsBuffersTooSmall)  // max_sample_count 64
  TEST(ResetSolveCopyReport, BatchedViscosityBuffers_ReportZeroAfterReset)  // reconcile: after the samples build,
      // ResetSolveCopyReport(buffers): every field of *ReadSolveCopyReport(buffers) is 0
  // viscosity_batched_no_gpu_test.cc, LABEL no_gpu
  TEST(BuildViscosityBuffers, BatchedLayoutOnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject:
  `Keep the viscosity working buffers in CPU memory and build samples in batches`.

### Task 27: The joint solve in batches, on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** new task.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** tests are `LABEL cpu`; reductions use the buffers' `sum_buffers`.

**Files:**
- Modify: `native/fluidity/viscosity/velocity_solve_cpu.cc`, `solve_viscosity_and_pressure_cpu.cc`
- Create: `viscosity_batched_scene_test.cc`

**Interfaces:**
- Consumes: Tasks 14 to 16, 26; pressure Task 59 (the pieces of `coupled_solve_pieces.h` and `flow_imbalance.h`
  in their sweep form, given whole arrays in CPU memory).
- Produces: the batched path of `SolveVelocity` and `SolveViscosityAndPressure` (and its hook form).
- **Plan decisions:** inner velocity iteration = 2 sweeps (sample pass and face pass on the loaded piece with the
  block partials of `d . Kd`; then the velocity, error and scaled-error updates with the partials of the next dot
  and the largest error per block); the true-error recheck is one sweep. Outer step: the push of the direction and
  the volume error through pressure's pieces (one sweep each), the inner solve, one sweep for the dots of the
  response against the kept responses (block partials) and one for the updates and the scaling. Acceptance by
  `pressure::CheckFlowImbalance` (its sweep). Outputs are written only by step 12's write-back sweep; cancel is read
  before every batch load and before each inner solve. A sweep runs all batches and keeps the smallest failure.
  The tree sum over the whole grid's block partials and the largest value of a loaded piece use
  `buffers.sum_buffers` (nothing allocates in a sweep).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; both processors the CPU; loaded 3, 5 and 7 times
  `CountBlocksInLargestLayer`; tolerances: velocity `1e-9 m/s + 1e-7 x largest speed`, pressure `1e-6 Pa + 1e-6 x
  largest pressure`, the CPU-against-GPU bounds):

  ```cpp
  TEST(SolveViscosityAndPressureInBatches, LinearShearTwoViscosities_WithinToleranceOfOnePiece)  // n 32, ratio 10
  TEST(SolveViscosityAndPressureInBatches, SyrupAtRest_WithinTolerance)   // Task 19's SyrupColumn on 8 layers:
      // also every speed <= 1e-9 m/s with Task 19's settings
  TEST(SolveViscosityAndPressureInBatches, FreeSurfaceStretchedSlab_WithinTolerance)  // Task 19's slab, 8 layers
  TEST(SolveViscosityAndPressureInBatches, ThreeCuts_IdenticalBytes)
  TEST(SolveViscosityAndPressureInBatches, CancelBetweenBatches_OutputsUntouched)  // flag set at the 5th load
  TEST(SolveViscosityAndPressureInBatches, FailureInOneBatch_OutputsUntouched)  // 1e308 Pa s in one cell of the
      // last batch: kNotFinite with that CellId, as one piece
  TEST(SolveViscosityAndPressureInBatches, SweepsPerInnerIteration)  // copy report: 2 sweeps per inner iteration
      // plus the recheck; prints sweeps and bytes per outer iteration
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject:
  `Solve drag and pressure together in batches`.

### Task 27a: The coupled terms in batches, on the CPU

**Amended 2026-11-14 (A24):** new task; Task 15a in the batched layout.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** tests are `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Modify: `native/fluidity/viscosity/solve_viscosity_and_pressure_cpu.cc`, `viscosity_batched_scene_test.cc`

**Interfaces:**
- Consumes: Tasks 15a, 27; pressure Task 59 (the A24 overloads in their sweep form). Blob entries whole on the
  processor; each blob's inflow and connection sums by the per-group rule (block partials, then `math::TreeSum`
  over the blob's blocks); material-face lists stream with the rows (pressure); the material velocities are written
  by the batch whose range stores the entry's face place, in the write-back sweep; `q` written once after it.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; loaded 3, 5 and 7 times
  `CountBlocksInLargestLayer`; Task 27's tolerances; material velocities with the velocity bound,
  `sealed_gas_pressure` with the pressure bound):

  ```cpp
  TEST(SolveViscosityAndPressureInBatches, ViscousLiquidOnSpringySlab_WithinToleranceOfOnePiece)  // Task 15b's slab at
      // w 0.0125 (16 x 16 x 32 cells, 8 layers), viscosity 10, one step from x* + 0.1 w
  TEST(SolveViscosityAndPressureInBatches, SealedGasAcrossBatches_WithinToleranceOfOnePiece)  // Task 15b's heated
      // pocket above syrup at w 0.0125: the blob spans four layers; 6.5625 Pa within 1e-9 relative
  TEST(SolveViscosityAndPressureInBatches, CoupledThreeCuts_IdenticalBytes)
  // Amended 2026-11-14 (A26, A27): material entries stream with the face rows; solids' motions, blocks and factors
  // whole on the processor; each sample's solid coefficients stream with the samples.
  // Amended 2026-11-15 (A59, A60): the m x m blocks and AllowedMotions whole on the processor; the viscous push
  // partials per block are written by the batch whose range holds the block and summed once over every block.
  TEST(SolveViscosityAndPressureInBatches, ShearedPlateForce_WithinToleranceOfOnePiece)  // Task 15e's sheared plate
      // on 8 block layers in x: viscous_push_on_solids within 1e-9 relative of one piece; three cuts identical bytes
  TEST(SolveViscosityAndPressureInBatches, SpringySlabDragged_WithinToleranceOfOnePiece)  // Task 15d's Couette slab
      // lengthened to 8 layers in x; slab and liquid velocities within the section 6 bounds
  TEST(SolveViscosityAndPressureInBatches, CubeAcrossBatches_WithinToleranceOfOnePiece)  // Task 15d's settling cube
      // on a grid where the cube spans three batches; its motion within the velocity bound
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject: `Carry
  materials and sealed gas in the batched joint solve`.

### Task 28: The joint solve in batches on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** new task.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the private headers are the three `_gpu.h` of Tasks 13 and 17; the
batched per-block functions are shared and run by a `cpu` test; digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/viscosity/viscosity_batched_gpu_test.cc`
- Modify: `viscosity_samples_gpu.h`, `viscosity_samples_gpu.cu`, `velocity_solve_gpu.h`, `velocity_solve_gpu.cu`,
  `solve_viscosity_and_pressure_gpu.h`, `solve_viscosity_and_pressure_gpu.cu`, `viscosity_steps_test.cc`,
  `CMakeLists.txt`

- [ ] **Step 1: Tests** (`LABEL gpu`): `BuildViscositySamples, InBatchesOnGpu_SameAsOnePieceOnGpu` (Task 26
  scenes, bytes), `SolveViscosityAndPressureInBatches, OnGpu_WithinToleranceOfOnePieceOnGpu` (Task 27 and, amended
  2026-11-14 (A24), Task 27a scenes),
  `SolveViscosityAndPressureInBatches, OnGpuTwice_GivesIdenticalBytes`, `ViscosityScene, SyrupSizingInBatches_Converges`
  (Task 24's syrup sizing with `max_loaded_blocks` a quarter of its blocks: converges; prints outer and inner
  iterations, sweeps, seconds and bytes per outer iteration). The `gpu` tests call `WriteResultDigest` on samples,
  velocity and pressure. (`LABEL cpu`, in `viscosity_steps_test.cc`, the main evidence until the hardware run):
  `BuildSamplesOfCell, EveryThreadOfLoadedPiece_EqualsBatchedCpuBuild` (Task 26 scenes, every thread block and
  thread of each load, launch order and reverse; bytes).
- [ ] **Step 2: Implement:** the per-load by-face view (`BuildGroupsByDestination(..., buffers.order_buffers)`) and
  the block partials on the GPU, each GPU function a loop around the shared per-item functions; copies only through
  `pressure::SolveBatches`. **Step 3: Run** the gate. **Commit.** Subject: `Solve drag and pressure in batches on
  the GPU`. Body: `CUDA: compiled, not run`.

### Task 29: Comment pass and hand-over

**Amended 2026-11-14 (A31):** working memory is named buffers (`SumBuffers`, `OrderBuffers`, members `sum_buffers`, `order_buffers`); each base sum, order, prefix sum and grouping has one form, which takes those buffers.

**Amended 2026-11-14 (A15):** renumbered from Task 26; its lists include Tasks 26 to 28.

**Amended 2026-11-14 (A24):** its lists include Tasks 15a, 15b, 17a, 19a, 27a.

**Amended 2026-11-14 (A26, A27):** and Tasks 15c, 15d; Task 20 removed; the plan decisions of Task 15c and the owner
questions of the A26/A27 section (the band, the larger face mass at a shared face, solids out of the outer
diagonal, a second free solid in one sample folded at its velocity before pressure).

**Amended 2026-11-15 (A53, A57, A58, A59, A60, reconcile item 31):** its lists include Tasks 4a, 15e, 15f, 15g; the
hand-over records the removed `kSmallestVisibleViscousEffectPerStep`, the two `RunMemory` parameters of
`BuildViscosityBuffers`, the generalized-velocity block (and that reconcile-todo item 23's review now covers it),
the stored solid coefficients of every motion kind, the slip divisor, `kOutOfMemory` last. **Amended 2026-11-15
(A74):** the settling sphere and the rotating cylinder are round shapes (no cube stand-in, nothing known-open for
want of a shape; the rotating cylinder's outer wall is the square domain box).

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the pass also checks `Uses:` lists (never a standard header) and
`Words:` (only where a file introduces domain words); the hand-over records the Foundation adoptions.

**Files:**
- Modify: any file of the module; `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every file of `native/fluidity/viscosity/` top to bottom as a C++
  programmer who has never read a fluid paper: each comment true of the final code; says why, not what; a
  worked example with numbers for each non-trivial public function (the linear shear channel: sample 0.5,
  weight `4 mu V`, power `mu shear^2 V`); every word in the glossary; `Uses` (project and library headers only,
  never a standard header) and `Used by` correct; `Words` present only where a file introduces domain words; "thread
  block" (never "block" alone) for the GPU sense, the cell block keeping its glossary name;
  `// Step N:` numbers match the paper's two procedures.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently: the
  `pressure_buffers` field; the plan decisions of Tasks 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 15a, 18, 19, 26, 27, 27a;
  the three lifted exclusions and the one known-open test (A24: no snow exclusion); the A24 coupled terms and their
  owner questions (slide at a material face, the material's normal velocity change left out of the stretch samples);
  the A25 number and its constant; the sign of the stretch invariant; the A15 sample order
  and dot order (both modes) and the batched solve; (Foundation as built) the sum and order buffers in `ViscosityBuffers`
  and their rows, `ToViscosityFailure`, the shared step headers (`samples_of_cell.h`, `velocity_solve_steps.h`)
  and their `cpu` tests, the three private `_gpu.h` headers.
- [ ] **Step 4: `docs/gpu-runs.md`:** a `viscosity` row, result `compiled, not run`, listing every `gpu` test of
  Tasks 13, 17, 17a, 24 and 28 for the end-of-programme run.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate's output, new and removed
  public names with reasons, the plan decisions, and the statement that no device code has run.
