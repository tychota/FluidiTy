# Rebuild Cell Resize and Batches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the last simulation module of the rebuild. Part A, cell resize: after each committed step, choose
which cells split or merge with the shared cell size rule (grid's `cell_size_rule.h`: depth, solids, shear, artist
regions and map; A41, A61 to A63), rebuild the grid in one transaction, remap every grid-carried value without
losing any, and split or merge the particles with their cells, so every liquid cell holds particles of its own size
level at every depth (A64). Deep liquid without particles is a later group at the end of this plan, not part of
the first version. Part B, batches: run a scene whose
particles do not fit the GPU in batches of cell blocks, with the solves either on the whole grid or, when their
working buffers do not fit either (owner answer A15), batch by batch through pressure's `SolveBatches`; the planner
chooses one piece, batches, or batches with solves in pieces and shows the planned memory and copies. A batched
run agrees with the one-piece run within the solver's tolerance, and gives the same bytes for every cut and on
every repeat. CPU first; device code compiled, not run.

**Amended 2026-11-15 (A65, A66, A67, A68):** every liquid cell holds about eight particles of its own size: a
particle's `size_level` is its cell's level, its rest volume is one eighth of its cell (`(w/2)³` for a cell of width
`w`), so a cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest particle's volume (A65). A run starts on
`step::BuildStartCells`' grid, already following the size rule (A66); the test scenes that start all-smallest do so
on purpose and say so. `ResizeCells` runs the extended `particles::AdjustParticleCount` at every pass, so particles
that moved across a size change are split or merged (A67). The later groups come in the order of A68; the first
version keeps slab batches and the pinned surface, and the batch plan exposes `SlabLimit` for the budget line.

**Architecture:** Two folders. `native/fluidity/cell_resize/` (namespace `fluidity::cell_resize`): shared formula
headers (`remap_math.h`, `velocity_gradient_math.h`, `cell_size_change_math.h`, `cell_resize_error.h`), CPU-only
list work (`split_and_merge_cells.h`, the resize plan), one public header per operation group
(`depth_below_surface.h`, `cell_size_changes.h`, `liquid_totals.h`, `remap_grid_values.h`, `resize_cells.h`) with
one private `_cpu.cc` / `_gpu.cu` pair per operation, and the step callback. The wanted width itself is grid's
(`native/fluidity/grid/cell_size_rule.h`, grid plan, A41); this module fills its facts, applies the hysteresis and
edits the cell list. **Amended 2026-11-15 (A41, A64):** `cell_size_rule.h` moved to grid; `carried_velocity_math.h`,
`grid_only_move_math.h`, `region_wish_math.h` and the grid-only parts of `regions.h` moved to the later group;
`regions.h` is split into `depth_below_surface.h` and `liquid_totals.h` (plan decision). `native/fluidity/batches/` (namespace
`fluidity::batches`): the batch plan, the batched run and its two passes. Batches has no device code of its own:
it moves arrays with `base/` copies, finds borders and builds pieces with the grid's buffer forms (A17), and runs
the step's stage functions on a loaded piece. The solves in pieces belong to the solvers (pressure's
`solve_batches.h`); batches sizes them and reports their copies.

**Tech Stack:** as Foundation (C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 with GoogleTest,
uv, clang-format and clang-tidy 23.1 as `pyproject.toml` pins them, Docker via colima, Typst with CeTZ). Nothing new. No Python reference and
no Lean proof (design section 7).

**Spec:** `docs/superpowers/specs/modules/11-cell-resize-and-batches-design.md` ("the design"; in the `main`
checkout, never committed; its section 3 is the contract) and the programme design `docs/design/programme.md` on
`rebuild` (sections 6, 7, 8, 8a). Also in the `main` checkout: `modules/owner-answers.md`,
`modules/shared-interface.md`, `modules/plan-decisions.md` (it wins over every design), section 3 of the grid,
particles, transfer, pressure, viscosity, materials, solids, gas and step designs, and the plans
`plans/2026-11-14-rebuild-0[2-9]-*.md` and `plans/2026-11-14-rebuild-10-step.md` for the file and helper names
they created.

## Global Constraints

Foundation's list, unchanged:

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
- **Amended 2026-11-14 (Foundation as built):** every task follows `AGENTS.md` "Structure" and `plan-decisions.md`
  "Foundation after the final review". Two halves: the public function lives in `<name>_cpu.cc` (checks, `RunOn`
  dispatch, the CPU loop, the GPU half called under `#if defined(FLUIDITY_HAS_GPU)`); the GPU half `<name>_gpu.cu` is
  declared in the private `<name>_gpu.h`, returns `RunOnCallStatus`, which the CPU half maps with `ToBufferError` and
  then into `CellResizeFailure` / `BatchesFailure` through one mapping function; it launches, waits and holds
  per-call GPU memory only through `base/gpu_launch.cuh` (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`,
  `GpuArray`). Every GPU function is a thin loop over `FLUIDITY_CPU_AND_GPU` per-item functions in a shared
  `<name>_steps.h` (or an existing shared formula header); a `cpu` test `<name>_steps_test.cc` runs them for every
  thread block and thread of `ComputeLaunchSize(count)`, in launch order and in reverse, against the CPU loop: the
  main evidence until the hardware run. Working buffers: per-step sums and prefix sums pass to
  `ComputeTreeSum`, `ComputeLargestMagnitude` and `ComputeExclusivePrefixSum` (one form each, A31) the `sum_buffers` and
  `order_buffers` that `CellResize` and `BatchedRun` own (built once, sized from their limits, counted in their memory
  estimates with `ComputeSumBuffersSize` / `ComputeOrderBuffersSize`); a function whose buffers or working buffers
  live on different processors breaks a documented precondition (`Requires:` same processor, a debug assertion,
  A30), never an error value. Errors: a failed check writes nothing; after
  `kRunOnCallFailed` the outputs are unspecified and the run stops; `kOutOfMemory` comes only from building forms.
  Exactness: discrete outputs equal on CPU and GPU; floating-point within design 6 unless a header promises the same
  bits unless NaN (`CheckWithinTolerance`, `CheckSameBitsOrBothNaN`). Tests link `fluidity_base_test_support` and the
  helper libraries they use (`fluidity_grid_test_support`, `fluidity_step_test_support`,
  `fluidity_pressure_test_scenes`, this module's `fluidity_cell_resize_test_support`, made with
  `fluidity_add_test_library`) through `LINKS`; `gpu` tests pass their main outputs to `WriteResultDigest`. Labels:
  `cpu` everywhere, `no_gpu` only for behaviour on a machine without a GPU, `gpu` only on hardware. Every target goes
  through the helpers of `cmake/fluidity_targets.cmake`, and every source file is listed in a CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the
  release preset, and CUDA compiled in the container, whose `cuda` step fails for a tracked `.cc` or `.cu` file that
  no build compiles. clang-format and clang-tidy are the 23.1 that `pyproject.toml` pins.

This module's own:

- **Design section 3 is the contract.** Spell every name as there. Additions are marked **Plan decision** in the
  task that makes them and repeated in the hand-over. `ResizeCells`, `SplitCell`, `MergeCells` are decided names;
  if the rules check reports them under `verb`, add them to `tools/check_rules_allowed_names.txt` with the reason
  `decided cell resize name (cell resize design 9)`.
- **Constants (design 3.2, 3.4, 3.7, never changed without the owner):** `kAmountTolerance = 1e-12`;
  `grid::CellSizeSettings` defaults (grid's, A41, A62, A63) `surface_reach_in_cell_widths = 8`,
  `solid_reach_in_cell_widths = 2`, `shear_width_factor = 0.25` (starting value, to calibrate),
  `merge_width_share = 0.8`, `merge_wait_duration = 0.25` s, `min_merge_wait_steps = 8`,
  `max_cell_splits_per_resize = 65536`; `BatchSettings` defaults `max_whole_grid_share = 0.5`,
  `min_blocks_per_batch = 64`. **Amended 2026-11-15 (A64):** `kMaxFilledShare` and `RegionSettings` belong to the
  later group.
- **Particles everywhere (A64).** Every liquid cell holds particles, at every depth and every cell size: no
  grid-only region, record, velocity blend, carried velocity or region transfer in the first version. A cell of size
  level `l` holds particles of `size_level == l`; `particles::AdjustParticleCount` merges eight particles of level
  `l + 1` into one of level `l` when a resize merges their cells and splits one into eight when it splits its cell
  ~~(C2: only particles at least 1.5 widths inside the change on both grids; drifted ones keep their level and are
  counted in `skipped_incomplete_count`)~~ **replaced** (A67): at every pass, over all blocks, every particle whose
  `size_level` differs from its cell's level is split (repeated to the smallest level in a smallest-level cell) or
  merged (chunks of eight at once, two to seven after `kMaxMergeWaitPasses = 16` passes, a lone one stays, counted);
  the both-grid check of C2 stays only for changed cells and counts an incomplete split instead of refusing it. One
  level per cell per resize, so a deep cell coarsens over several resizes and its particles follow level by level.
  **A65 wording, used everywhere:** every liquid cell holds about eight particles of its own size: a particle's
  `size_level` is its cell's level, its rest volume is one eighth of its cell (`(w/2)³`), so a cell k levels coarser than the smallest holds 8
  particles each of `8^k` times the smallest particle's volume. **A66:** runs start on `step::BuildStartCells`' grid;
  a test scene that starts all-smallest does so on purpose (it tests coarsening) and says so. The tasks of the later
  groups at the end ("Can the fine surface band shrink from 8 cells to 4? (later, test only)", "Coarser cells on calm
  surfaces (later)", "Batches cut by columns or tiles (later)", "Deep liquid without particles (later; needs review)")
  are not part of this module's done-criteria.
- **One reserved block per processor (A58; reconcile item 31; `plan-decisions.md` "Owner answers A40–A64 applied"
  items 3 (with its "Correction (same day, reconcile)") and 11).** `BuildCellResize` and `BuildBatchedRun` build on
  both processors, so they take `RunMemory& run_memory, RunMemory& cpu_run_memory` as their last two parameters
  (every other parameter stays): `where = kRunOn` buffers are pieces of `run_memory` (`BuildBuffer(run_on, size,
  run_memory)` and the `RunMemory` forms of `BuildSumBuffers`, `grid::BuildBorderSearchBuffers`), `where = kCpu`
  buffers kept `kWholeRun` (cell resize's CPU grid copies and mirrors, the batches' piece map copy) are pieces of
  `cpu_run_memory`; in a CPU run both are the same object. `Requires:` `run_memory.run_on()` is the buffers'
  processor and `cpu_run_memory.run_on() == kCpu` (debug assertions, A30). No `kRunOn` entry is `kOneCall` or
  `kOneStepTry`: such rows are `kWholeRun`. Each entry stands for whole buffers (`copies` = the number of buffers,
  each `count x bytes_each` bytes), so `ComputeRunMemoryBytes(estimate, counts, where)` per `where` equals the bytes
  the build hands out of each block (tests `..., IntoRunMemory_HandsOutExactlyTheEstimate`). Plain `BuildBuffer` is
  left for tests and `kOneCall` CPU arrays. `BufferError::kRunMemoryTooSmall` and `kOutOfMemory` map to the module's
  `kOutOfMemory` with `magnitude` = the bytes asked. `CellResizeError::kOutOfMemory` and `BatchesError::kOutOfMemory`
  are the last values of their enums, returned only by `BuildCellResize` and `BuildBatchedRun`.
- **Span forms (A30; reconcile item 32).** A CPU (span or table) form called with tables of a GPU grid, or a CPU list
  with GPU buffers (`FindCellsThatMustSplit`, `SplitCell`, `MergeCells`, `BuildResizePlan`, `BuildResizeMap`, the
  span form of `grid::FindBorderBlocks` in tests), is a `Requires:` precondition with a debug assertion, never
  `kBadInput`.
- **One child order:** `cell_eighth = x + 2 y + 4 z` (grid 3.1) everywhere. `main`'s permutation
  `[0,4,2,6,1,5,3,7]` appears only inside the comparison tests that convert `main`'s numbers.
- **Size levels:** level 0 is the biggest cell; `level + 1` is half the width (grid plan Task 8; amended
  2026-11-14, A17: the task was renumbered).
- **Sums:** every total over cells, particles or blocks is one partial per cell block (cell places, then
  particles in grouped order inside the block), then `math::TreeSum` over the block partials in block order
  (design 5.10). A per-destination value is gathered by one iteration in a fixed order. No floating-point
  atomics. The reported failure is the first failing cell in the grid's fixed order, then the lowest particle id.
- **One material per record (A8, A10):** later group only (A64). A grid-only record holds one liquid material; gas
  markers, snow and springy-solid particles are never taken into a record; `ApplyGridOnlyRegionToGrid` writes only
  into the one liquid material set; a grid-only cell never touches a cell that holds gas.
- **Same bytes for the size decisions (A41, A63).** `ComputeCellSizeChanges` reads local stencils and exact
  minimums and maximums only; widths are compared with `grid::CellWidthsByLevel` (never `log2` or `pow`); no FMA
  contraction. So the `size_change` bytes are equal on CPU and GPU and for any cut into block ranges, and the GPU
  run reads back only those bytes (1 B per cell place, one counted copy) for the CPU list edit.
- **The resize is a transaction (design 5.1):** any failure after `BuildResizeTry` calls `RejectResize` (and
  `grid::TakeBuffers` on a resized grid already built); the committed state stays the same to the byte.
- **Batched requirement (design 5.10, amended 2026-11-14, A15):** a batched run gives the same bytes for every cut
  and on every repeat, in each mode (`kBatched`, `kBatchedWithSolvesInPieces`); against the one-piece run of the
  same processor it agrees within the solver's tolerance (one step: pressure's velocity and pressure bounds; whole
  runs: the step design 6.4 tolerances), not byte for byte. The local outputs of pass 1 still equal the whole-grid
  computation's bytes (the check that stacked borders are wide enough). `plan-decisions.md` "Batched solves (A15)".
- **A run is all CPU or all GPU (A17).** In a GPU run nothing here reads CPU tables of a GPU grid: borders with the
  buffer form of `grid::FindBorderBlocks`, pieces with the buffer form of `grid::BuildGridPiece`, the piece map read
  back once per load (4 B per loaded block); where CPU work needs the tables (the resize plan, the closure search,
  `ResizeMap`), one `grid::ReadGrid` copy per call, counted in a `where = kCpu` memory estimate entry and in the copies.
  Amended 2026-11-14 (A17, batched-run buffers): the step plan's Appendix A lists every buffer of a batched run by
  processor; batches follows it (Tasks 35 to 39, 43) and adds no buffer or copy it does not list.
- **Tolerances (design 6):** CPU against GPU: plans, `size_change` bytes, wait counters, ids and counts equal;
  remapped values `|a - b| <= 1e-13 * max(|a|, |b|) + 1e-13 * (largest value of that array in the case)`;
  `depth_below_surface` within `1e-13` of the cell width; `held_velocity_gradient_size` equal bytes (fixed order,
  no contraction). Repeat runs: identical bytes. (Amended 2026-11-15, A64: regions, layers, the ledger, moved
  amounts and the carried velocity are the later group's.)
- **Test labels (amended 2026-11-14, Foundation as built):** shared-formula, operation and scene tests `LABEL cpu`
  (the container's `cuda` step runs them too); device tests `LABEL gpu`. Every operation with a
  `_gpu.cu` has a `no_gpu` test "on the GPU without a GPU gives `kRunOnCallFailed`".
- **Test scenes** come from the step's `fluidity_step_test_support` (`step_test_scenes.h`, extended in Task 1); a
  step scene used with cell resize gets `has_cell_resize = true` through `scene_to_change()` (Task 1a adds the scenes
  of the size rule, A41 to A64). This module's
  test-only library `fluidity_cell_resize_test_support` holds the run builders (Tasks 10, 24, 29 and 39); amended
  2026-11-14 (Foundation as built): it is made with `fluidity_add_test_library(fluidity_cell_resize_test_support
  SOURCES ... LINKS fluidity_step_test_support fluidity_grid_test_support fluidity_base_test_support)` and each test
  names it in `LINKS`; no test defines its own bit comparison, buffer reader or tolerance check.
- **Proof-of-concept numbers** are produced as `plan-decisions.md` says (export `main` outside the worktree,
  build there, a throwaway driver never committed), rotated from Y-up with rows `{1,0,0},{0,0,-1},{0,1,0}`: `main`'s
  `y` neighbours become this code's `z`, `main`'s `+z` neighbour becomes `-y`, and children are matched by their
  centre position after the rotation, not by index. `main`'s sources are under `native/src/`.

## Task order

Part A (Tasks 1 to 33) is complete, with its scene tests through `step::TryStep` and `AdvanceTo`, before Part B
(Tasks 34 to 45) starts. Task 1 changes the step's test scenes and is placed first (rule: a change to an earlier
module's file is its own task, before any code that needs it). The two step changes only Part B needs (Tasks 35,
36) open Part B, right after its method-paper task, so that Part A stays whole; nothing in Part A uses them.
**Amended 2026-11-15 (A64, A41):** Tasks 3, 18 to 22, 25 and 26 moved with their numbers to the group "Deep liquid
without particles (later; needs review)" after Task 45, beside the later parts of mixed tasks (10b, 11b, 15b, 16b,
17b, 23b, 24b, 28b, 29b, 30c, 31b, 32b) and its two gating tests (Task 32c); Task 7 moved to the grid plan. New
main-sequence tasks: 1a, 7a, 13a, 27a, 30a, 30b. The group "Coarser cells on calm surfaces (later)" follows. Part A
is Tasks 1 to 33 without the moved ones.
**Amended 2026-11-15 (A66, A67, A68):** new main-sequence task 30d (particles that cross a size change, the start on
the start cells); Tasks 1a, 24, 27, 38, 45 amended. The later groups after Task 45 come in this order (A68): (1) "Can
the fine surface band shrink from 8 cells to 4? (later, test only)" (Task 45a); (2) "Coarser cells on calm surfaces
(later)" (Task 46) together with "Batches cut by columns or tiles (later)" (a sketch list, no task); (3) "Deep liquid
without particles (later; needs review)", last. The first version keeps slab batches and the pinned surface.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: Foundation,
contact, grid, particles, transfer, pressure, viscosity, materials, solids, gas and step, each with its comment
pass done; the step with `resize_access.h` (`ResizeTry`, `BuildResizeTry`, `ApplyResizedGrid`, `CommitResize`,
`RejectResize`), `merge_ready_duration`, `held_velocity_gradient_size` and `merge_ready_step_count` in `GridFields`
(committed and try copies), `CommittedStepLimits::largest_liquid_speed`, the callbacks `resize_cells` and
`run_stages`, stage 2 calling `grid::ComputeNewCellSizeLevel`, and `StepArray` / `ReadStepArrays` (step plan Tasks
32, 33); grid's `cell_size_rule.h` (grid plan, A41); Foundation's `RunMemory` (Foundation Tasks 17 to 19); and the
A15 tasks of pressure (54 to 61: `solve_batches.h`, the batched layout), viscosity (26 to 28) and gas (29, 30), and
the step's amended Tasks 8, 33 (`RunMode::kBatchedWithSolvesInPieces`, `RunPlan::max_solve_loaded_blocks`, the
loaded try's piece buffers). **Amended 2026-11-15 (A41, A58, A64):** the grid-only arrays and callbacks
(`apply_grid_only_region`, `move_grid_only_liquid`, `read_largest_grid_only_speed`) and `choose_cell_size` are not
there in the first version.

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line,
`grep -c "resize_cells\|run_stages\|ReadStepArrays\|free_grid_storage\|largest_liquid_speed" native/fluidity/step/advance_to.h native/fluidity/step/step_stages.h native/fluidity/step/resize_access.h`
counts at least one in each file, and `grep -c "ComputeWantedCellWidth\|ComputeArtistSizeLevel" native/fluidity/grid/cell_size_rule.h`
and `grep -c "BuildRunMemory" native/fluidity/base/run_memory.h` count at least one. If any fails, stop and ask the
owner.

**Amended 2026-11-14 (Foundation as built):** Foundation as built also provides `base/run_on_call.h`
(`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, the sums with their `SumBuffers` and the stable
order and prefix sum with their `OrderBuffers` (one form each, A31), `base/test_support.h` (library `fluidity_base_test_support`) and
`fluidity_add_test_library`; read `AGENTS.md` "Structure" before the first task. The gate prints seven `PASS` lines:
`format`, `rules`, `cpu` debug, `cpu` release, `tidy`, `paper`, `cuda`.

## Review Focus

1. **A resize that fails after the resized grid was built.** The free storage must come back, the committed
   grid, fields, particles and gas keep their bytes, and the next `TryStep` commits. Pinned:
   `ResizeCells, FailureAtEachStep_ChangesNothing` in Task 27a (amended 2026-11-15, A64: was Task 26).
2. **Merge wait against "nothing to do".** A second resize at the same committed time must add no wait and
   change nothing, while a change of `merge_ready_duration` alone is committed. Pinned:
   `ResizeCells, CalledTwiceAtSameTime_SecondDoesNothing` in Task 24.
3. **The last small cell of a split leaving the range by rounding.** `B` minus the sum of the first seven must
   not go below the range; then every small cell gets `B / 8`. Pinned:
   `ComputeSplitAmounts, LastCellOutOfRange_AllGetAnEighth` in Task 5.
4. **A deep cell coarsening over several resizes (A64).** Each merge must take the eight particles of the next
   level into one of the cell's level, one level per resize, so after the cascade every liquid cell holds particles
   of its own level and nothing was lost on the way. Pinned: `ResizeCells,
   CalmDeepTank_CoarsensWithParticlesInEveryCell` in Task 30a (amended 2026-11-15, A64: the former focus, a foreign
   particle in a grid-only cell, moved to the later group with `ApplyRegionTransfer,
   OwnAndForeignParticleInOneCell_CellReturnsToParticles`, Task 25).
5. **A batch cut that does not follow the layers** (a range ending in the middle of a layer of blocks). Pinned:
   `ApplyStagesInBatches, CutInsideALayer_SameBytesAsLayerCut` in Task 40 (A15: compared with the cut at whole
   layers of the same mode, and within tolerance of one piece).
6. **The held gradient across a broken wait (A63).** A shear burst that ends must keep the cells small for the whole
   merge wait, and a break must restart both the time and the step count. Pinned: `ComputeCellSizeChanges,
   ShearBurstEnds_HeldForTheWait` in Task 13.
7. **A large particle rising into the surface band (A67).** A particle carried up out of a coarse cell must arrive
   in the smallest cells already split to the smallest size, in the same pass it enters them. Pinned: `ResizeCells,
   ParticleRisingFromDeepCell_ArrivesSplitToSmallest` in Task 30d.

---

# Part A: changing cell sizes

### Task 1: Step test scenes for cell resize

**Amended 2026-11-15 (A64):** unchanged; `BuildRisingFloorScene` and `BuildStandingWaveTankScene` are used only by
the later group (the deep tank's particles stay in every cell in the first version).

**Amended 2026-11-15 (A66):** these scenes keep their all-smallest start (smallest cells, 8 particles of the smallest
size per cell) on purpose: the tests that use them exercise splits, merges and the coarsening from that start. A run
built by I/O starts on `step::BuildStartCells`' grid instead; each scene passes the step's test support the
all-smallest start cells (the cover of its particles at `smallest_size_level`), as the step plan's A66 test support
provides. One line in each builder's comment says so.

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/step/step_test_scenes.h`, `step_test_scenes.cc`, `step_test_scenes_test.cc`

**Interfaces:**
- Produces (test-only, plan decision; all Z-up, gravity `{0, 0, -9.81}`, seed 20261114, `has_cell_resize = true`,
  grid frame origin at the domain's low corner, biggest cell 0.1 m, 3 size levels (0.1, 0.05, 0.025 m), start
  particles at the smallest level, 8 per cell at the cell-eighth centres, ids in x-fastest order):

  ```cpp
  TestScene BuildDeepTankScene();            // closed box 0.4 x 0.4 x 1.0 m, water to 0.8 m, at rest
  TestScene BuildMixedSizeColumnScene();     // closed box 0.4 x 0.4 x 0.6 m, water to 0.43 m, at rest
  TestScene BuildRisingFloorScene();         // the deep tank, open top, a driven slab 0.4 x 0.4 x 0.05 m on the
                                             //   floor moving +Z at 0.01 m/s
  TestScene BuildStandingWaveTankScene();    // closed box 0.8 x 0.1 x 0.8 m, water surface 0.6 + 0.02 cos(pi x/0.8)
  TestScene BuildTwoLiquidsDeepScene();      // closed box 0.8 x 0.4 x 1.0 m, water x < 0.4 and oil (800 kg/m3) x >= 0.4
                                             //   to 0.8 m, both in the liquid set; a snow cube of 0.1 m on the floor at
                                             //   x 0.7..0.8, y 0.3..0.4 in its own set
  TestScene BuildSmokeOverDeepWaterScene();  // open box 0.4 x 0.4 x 1.0 m, water to 0.7 m, ambient gas above with 8
                                             //   markers per open cell, a smoke source at (0.2, 0.2, 0.75), a driven
                                             //   flat paddle at x = 0.05, stroke 0.02 m, period 1 s
  TestScene BuildFloatingCubeScene();        // closed box 0.4 x 0.4 x 0.4 m, a water cube of 5 x 5 x 5 cells of
                                             //   0.05 m at rest, centred, in air, particles at level 1
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `BuildDeepTankScene, ParticleCountAndMass` (8192 cells x 8 = 65536
  particles, mass `1000 * 0.025^3 / 8` each); `BuildTwoLiquidsDeepScene, LiquidSetHoldsBothLiquids`;
  `BuildFloatingCubeScene, OneThousandParticlesAtLevelOne`; `BuildSmokeOverDeepWaterScene, GasSetAlone`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with the existing builders' helpers (`BuildWaveScene`'s paddle,
  `BuildGasBesideLiquidScene`'s markers). **Step 4:** gate. **Commit.** Subject:
  `Add test scenes for changing cell sizes`.

### Task 1a: Step test scenes for the shared size rule

**Amended 2026-11-15 (A41, A62, A63, A64):** new task; the scenes of the size-rule tests (`plan-decisions.md`
"Owner answers A40–A64 applied" items 1 and 2).

**Amended 2026-11-15 (A66):** the five scenes below keep the all-smallest start on purpose (the calm-tank and
calm-pool tests measure the coarsening and its time from it; their comments say so); one new scene starts the calm
deep tank on `step::BuildStartCells`' grid, as a run does.

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/step/step_test_scenes.h`, `step_test_scenes.cc`, `step_test_scenes_test.cc`

**Interfaces:**
- Produces (test-only, plan decision; Z-up, gravity `{0, 0, -9.81}` unless named, seed 20261115, `has_cell_resize =
  true`, frame origin at the domain's low corner, 5 size levels, start particles at the smallest level, 8 per cell
  at the cell-eighth centres, ids in x-fastest order; the box is a whole number of biggest cells):

  ```cpp
  TestScene BuildCalmDeepTankScene();   // w_min 0.05 m (widths 0.8 .. 0.05 m); closed box 0.8 x 0.8 x 2.4 m, water to
                                        //   2.0 m, at rest
  TestScene BuildCalmPoolScene();       // w_min 0.02 m (0.32 .. 0.02 m); closed box 0.32 x 0.32 x 1.28 m, water to
                                        //   1.0 m, at rest
  TestScene BuildThinSheetScene();      // w_min 0.02 m; closed box 0.64 x 0.32 x 0.64 m: a slab 0.64 x 0.32 x 0.04 m at
                                        //   z 0.40..0.44 m falling at rest onto a pool 0.10 m deep
  TestScene BuildLiquidBridgeScene();   // w_min 0.02 m; no gravity; closed box 0.64 x 0.32 x 0.32 m: a cylinder of
                                        //   radius 0.06 m along x through the box centre, velocity x sign(x - 0.32) m/s
  TestScene BuildJetScene();            // w_min 0.02 m; closed box 1.28 x 0.64 x 1.28 m, water to 1.0 m at rest; a
                                        //   liquid source of radius 0.05 m at (0.10, 0.32, 0.40) m along +x at 2 m/s
  // Amended 2026-11-15 (A66). The calm deep tank on the start cells: step::BuildStartCells on the water box (one
  // StartRegionShape kBox, centre (0.4, 0.4, 1.0) m, half size (0.4, 0.4, 1.0) m, no solid shape, no artist
  // sizes), then 8 particles per start cell of width w at its cell-eighth centres (offsets +-w/4, the order of
  // particles::ComputeRefillParticle), rest_volume (w/2)^3, mass 1000 x rest_volume, size_level the cell's level,
  // at rest: the rule of io::BuildStartParticles, repeated in test support because I/O comes after this module.
  TestScene BuildCalmDeepTankSceneOnStartCells();
  ```
- The cell resize tests run these scenes with `CellResizeSettings::cell_size.smallest_size_level = 4` and the other
  `grid::CellSizeSettings` defaults unless a test names a value.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `BuildCalmDeepTankScene, ParticleCountAndMass` (16 x 16 x 40 cells x
  8 = 81,920 particles, mass `1000 * 0.05^3 / 8` each, 1e-15 relative); `BuildCalmPoolScene, ParticleCount` (16 x
  16 x 50 x 8 = 102,400); `BuildJetScene, SourceSpeedAndRadius`; `BuildLiquidBridgeScene, NoGravityOppositeVelocities`;
  (A66) `BuildCalmDeepTankSceneOnStartCells, EightPerCellAtItsLevel` (every start cell holds exactly 8 particles of
  its level; `grid::CheckSizeStepRule` passes; total mass `1000 x 0.8 x 0.8 x 2.0 = 1280 kg` within 1e-12 relative,
  the box being aligned with the cells; particle count below `BuildCalmDeepTankScene`'s 81,920 and every cell
  within 8 `w_min` = 0.4 m of the surface at `w_min`).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with Task 1's helpers. **Step 4:** gate. **Commit.** Subject:
  `Add test scenes for the shared cell size rule`.

### Task 2: Method paper, the resize transaction, the size rule and the remap

**Amended 2026-11-15 (A41, A61 to A64):** the size-rule text describes grid's shared rule (five terms, the reference
speed) as this module uses it, the pass `ComputeCellSizeChanges` and its hysteresis, the artist inputs, and
particles in every liquid cell; new pseudocode `<proc-compute-cell-size-changes>`; new glossary rows.

**Amended 2026-11-15 (A65, A66, A67):** Step 1's text also gives the A65 wording (every liquid cell holds about eight
particles of its own size), the start on `step::BuildStartCells`' grid (A66), and particles that cross a size change
split or merged at every pass with the merge's energy reported (A67, design 5.4); `<proc-resize-cells>` step 3 has
the particle-only pass; glossary rows `mismatched particle`, `start cells`.

**Kind:** mechanical

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/cell_resize.typ` (included after the step section), `figures/cell_size_rule.typ`,
  `figures/split_remap.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include, notation rows), `docs/glossary.md`

**Interfaces:**
- Produces labels `<sec-cell-resize>`, `<sec-cell-size-rule>`, `<sec-remap>`, `<fig-cell-size-rule>`,
  `<fig-split-remap>`, `<proc-resize-cells>`, `<proc-find-cells-that-must-split>`, `<proc-compute-cell-size-changes>`
  (amended 2026-11-15).

- [ ] **Step 1: Text,** design 8 outline items 1 to 5: why resizing is its own transaction and what a failed one
  leaves (5.1); the wanted width of design 5.3 (amended, A41 to A63): `w_band`, `w_solid`, `w_shear = c_u U_ref /
  g` with `U_ref = max(largest liquid speed, sqrt(|gravity| w_max))`, `w_artist`, the clamp, the smallest cells at
  the surface (A61), the velocity gradient's Frobenius norm and why speed alone is not used, the held maximum and
  the merge wait of time and steps, the 8 `w_min` no-merge band, one level per resize, the artist regions and the
  map; the cells that must also split; the remap of amounts (limited slope, cell 7 takes the rest), pressures,
  labels and face flows (5.4, the table); splitting particles needs both grids (5.4, C2, the plain consequence), and
  the particles of a deep cell following its size level over several resizes (A64), with the cost bound of design
  5.3 ("Particles in deep cells"). Cite [6] Wang, Sun, Zhu 2026, [1] Jiang 2015, Losasso, Gibou, Fedkiw 2004, Ando,
  Thürey, Wojtan 2013, Ando and Batty 2020, Adams et al. 2007 and Popinet (Gerris) by author and year (the last four
  as `literature-cell-size.md` reads them). Glossary rows: `wanted cell width`, `merge_ready_duration`,
  `merge_ready_step_count`, `held_velocity_gradient_size`, `velocity gradient size`, `reference speed`, `cell size
  region`, `cell size map`, `depth_below_surface`, `ResizeMap`, `cell source`.
- [ ] **Step 2: Figures.** `<fig-cell-size-rule>`: a vertical cut through a wave over a solid step with a jet under
  the surface, cells coloured by size, both reaches, the 8 `w_min` band and the shear layer drawn.
  `<fig-split-remap>`: one big cell and its 8 small cells, the outer pieces, the 7 tree faces numbered as Task 6 and
  the 5 zero faces.
- [ ] **Step 3: Pseudocode.** `<proc-resize-cells>`: the nine steps of design 5.1, numbered 1 to 9 exactly, "fails
  when" listing the `CellResizeError` values per step. `<proc-find-cells-that-must-split>`: design 5.3 step 4.
  `<proc-compute-cell-size-changes>`: the two parts of design 5.3 ("The pass"), numbered as Task 13 implements.
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`, look at
  every new page, fix overlaps; gate. **Commit.** Subject: `Describe how cells change size between steps`.

### Task 3: Method paper, the two regions and the region transfer (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 4: Errors, names and the module's build lines

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** tests `LABEL cpu` (no `fluidity_add_shared_test`); the folder's own `CMakeLists.txt` lists headers,
sources and tests through the helpers.

**Files:**
- Create: `native/fluidity/cell_resize/cell_resize_error.h` (shared), `cell_resize_message.h`,
  `cell_resize_message.cc`, `cell_resize_error_test.cc`
- Create: `native/fluidity/cell_resize/CMakeLists.txt` (`fluidity_add_headers`, `target_sources(fluidity PRIVATE
  ...)`, `fluidity_add_gpu_sources`, `fluidity_add_test`; every later task adds its files here)
- Modify: the CMake lines that list module folders (register `native/fluidity/cell_resize/` as `step/` is)

**Amended 2026-11-15 (A64; reconcile item 31):** `CellResizeError` has 10 values: `kWrongMaterialSet`,
`kGridOnlyCellDrained` and `kGridOnlyCellTooFull` exist only for the grid-only region and are appended by the
later group (Task 4b) after `kOutOfMemory`; `kOutOfMemory` moves to the end (after `kRunOnCallFailed`).

**Interfaces:**
- Produces design 3.1 exactly: `CellResizeError` (10 values, `kOutOfMemory` last), `CellResizeFailure`,
  `ToMessage`, `ToName`.

- [ ] **Step 1: Failing tests:** `ToName, EveryCellResizeError_IsLowerSnakeCaseWithoutK`
  (`kTooManyCellBlocks` -> `"too_many_cell_blocks"`, all 10 by `static_assert` on the count);
  `ToMessage, EveryCellResizeError_DistinctAndNotEmpty`; `CellResizeFailure, DefaultIsNone`;
  `CellResizeError, OutOfMemoryIsLast` (`static_cast<int>(kOutOfMemory) == 9`, `kRunOnCallFailed == 8`)
  (shared header, `LABEL cpu`).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Messages are one sentence saying what the user can do
  ("The resized grid needs more cell blocks than the limit. Raise the cell block limit or the smallest cell
  width."). **Step 4:** gate. **Commit.** Subject:
  `Name why a cell resize can fail`. New names: the 3.1 list.

### Task 5: Split and merge of an amount

**Amended 2026-11-14 (A31):** test comment says a throwaway search.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header tests are `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/cell_resize/remap_math.h` (shared, amounts part), `remap_math_test.cc`,
  `testdata/proof_of_concept_split_amounts.txt`

**Interfaces:**
- Produces design 3.6 `SiblingAmounts`, `ComputeMergedAmount`, `ComputeSplitAmounts`, `ComputeSplitLimiter`.
- **Plan decisions** (design 5.4 "Amounts on a split"): per axis `slope = minmod(B - low, high - B)` in the units
  of `B`, 0 where `has_neighbour` is false; small cell `k` with signs `s = (2 bit - 1)` per axis gets
  `B/8 + (limiter/8) * (s_x slope_x + s_y slope_y + s_z slope_z) / 4` for `k = 0..6`, and `B - sum(0..6)` for `k = 7`
  (summed in index order). `ComputeSplitLimiter` returns the largest value in `[0, 1]` that keeps `8 * small` of
  every `k = 0..6` inside `[min, max]` of `B` and its present neighbours; if cell 7 then leaves that range,
  `ComputeSplitAmounts` gives every cell `B / 8` (the caller passes the limiter; the fallback is inside).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeSplitAmounts, ThenComputeMergedAmount_ReturnsStart)  // design 7 "Amounts, split then merge": random
      // positive amounts in [0.5, 2] in a 3 x 3 x 3 patch (seed 20261114), split the centre: the 8 sum to B within
      // 2 rounding steps of B; ComputeMergedAmount of them returns B within 2 rounding steps
  TEST(ComputeSplitAmounts, LinearField_FollowsTheSlope)   // design 7 "Amounts, limiter": B 8, neighbours B -+ g with
      // g = {0.8, 0.4, 0.2}: limiter 1; 8 * small(k) == 8 + sum s g / 4 within 1e-14 relative
  TEST(ComputeSplitAmounts, Jump_StaysWithinNeighbours)    // x: low 0, B 1, high 10; y, z flat: every 8 * small inside
      // [0, 10] within 1e-14 relative
  TEST(ComputeSplitAmounts, MissingNeighbour_FlatOnThatAxis)
  TEST(ComputeSplitAmounts, LastCellOutOfRange_AllGetAnEighth)  // Review Focus 3: inputs found once by a throwaway
      // search over random amounts (written into the test with a comment saying so) where B - sum(0..6) falls
      // below the range: every small cell == B / 8 exactly
  TEST(ComputeSplitAmounts, ZeroBigCell_AllZero)           // B 0, neighbours positive: all eight exactly 0
  TEST(ComputeMergedAmount, SumsInIndexOrder)              // {1e16, 1, -1e16, 1, 0, 0, 0, 0} == left-to-right sum
  ```
- [ ] **Step 2: Produce the old numbers** (supporting only): `ProlongLiquidInventory3D`
  (`native/src/liquid_remap_3d.cc:198-299`, minmod at `:112-122`) on one cell with six neighbours, mass lane only,
  five cases; write the children's masses, rotated and re-indexed as Global Constraints say, into the testdata
  file with `main`'s commit hash. Test `ComputeSplitAmounts, MatchesProofOfConcept`: `1e-12` relative.
- [ ] **Step 3: Run** (fail). **Step 4: Implement.** **Step 5:** gate. **Commit.** Subject:
  `Split and merge an amount between a cell and its eight small cells`.

### Task 6: The inner face flows of a split

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header tests are `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Modify: `native/fluidity/cell_resize/remap_math.h` (face part), `remap_math_test.cc`
- Create: `testdata/proof_of_concept_inner_flows.txt`

**Interfaces:**
- Produces design 3.6 `InnerFaceFlows`, `ComputeInnerFaceFlows`.
- **Plan decisions:** inner face index `= axis * 4 + b_low + 2 * b_high`, where `b_low`, `b_high` are the small
  cells' bits on the two other axes in ascending axis order; a flow is m3/s along `+axis`, from the cell with that
  axis bit 0 to the cell with it 1. Tree (design 5.4): for `k = 7` down to `1`, `parent = k & (k - 1)`; the face
  between them is on the lowest set bit of `k`; `k` sends to `parent` what makes its net outflow (outer faces plus
  inner faces already set) equal `big_cell_outflow / 8`. Tree faces `{0, 1, 2, 3, 4, 6, 8}`; zero faces
  `{5, 7, 9, 10, 11}`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeInnerFaceFlows, EveryCellGetsAnEighthOfTheOutflow)  // outer {1..8} * 1e-3, big 0.036: each small net
      // outflow 0.0045 within 1e-13 * 0.008
  TEST(ComputeInnerFaceFlows, ZeroImbalance_EveryNetIsZero)
  TEST(ComputeInnerFaceFlows, FacesOutsideTheTree_AreZero)        // indexes 5, 7, 9, 10, 11 exactly 0
  ```
- [ ] **Step 2: Old numbers:** `RedistributeSiblingFluxes` (`native/src/adaptive_liquid_remap_3d.cc:439-514`, tree
  `:482-501`) for three outflow sets; rotate and re-index; `ComputeInnerFaceFlows, MatchesProofOfConcept`, `1e-12`.
- [ ] **Step 3: Run** (fail). **Step 4: Implement.** **Step 5:** gate. **Commit.** Subject:
  `Give the inner faces of a split the flows that keep each small cell's imbalance`.

### Task 7: The change each cell asks for

**Amended 2026-11-15 (A41, A63):** the wanted width moved to grid (`grid/cell_size_rule.h`, grid plan: its tests
`ComputeWantedCellWidth, DepthsGrowWithDistance`, `..., SolidDistanceCappedByBiggest`, `..., AirPocketOrGas_IsSmallest`
go there). This task keeps the per-cell change: `CellSizeChange` (two values added) and `ComputeCellSizeChange`,
now reading the grid rule's facts and wanted width, with the merge vetoes of A63.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header tests are `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/cell_resize/cell_size_change_math.h` (shared), `cell_size_change_math_test.cc`,
  `testdata/proof_of_concept_width_decisions.txt`

**Interfaces:**
- Consumes: grid `CellSizeSettings`, `CellWidthsByLevel`, `CellSizeFacts` (grid plan, `cell_size_rule.h`).
- Produces design 3.2 exactly: `CellSizeChange { kKeep, kSplit, kMayMerge, kMergeReady, kKeepsItsSize }`,
  `ComputeCellSizeChange(settings, widths, facts, wanted_width)`. Rules (design 5.3 rules 2, 3 and 5's per-cell
  part): `kKeepsItsSize` when `holds_particles_that_keep_their_count`; else `kSplit` when `cell_width >
  wanted_width`; else `kMayMerge` when `2 * cell_width <= merge_width_share * wanted_width` and none of
  `touches_air_gas_or_solid`, `touches_air_pocket`, `wants_smallest_cells`, `depth_below_surface <
  surface_reach_in_cell_widths * widths.width[smallest_size_level]`; else `kKeep`. Never returns `kMergeReady`
  (Task 13 writes it after the wait). No FMA contraction.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; smallest width `w = 0.025`, 5-level `CellWidthsByLevel` 16 w .. w):

  ```cpp
  TEST(ComputeCellSizeChange, WiderThanWanted_IsSplit)     // width 2 w, wanted 1.5 w: kSplit
  TEST(ComputeCellSizeChange, TwiceWidthWithinShare_IsMayMerge)  // width w, wanted 2.5 w, depth 20 w: 2 w <= 0.8 * 2.5 w
  TEST(ComputeCellSizeChange, JustAboveShare_IsKeep)       // width w, wanted 2.49 w, depth 20 w: kKeep
  TEST(ComputeCellSizeChange, KeepsItsCount_IsKeepsItsSize)  // holds_particles_that_keep_their_count, width 4 w,
      // wanted w: kKeepsItsSize (never kSplit)
  TEST(ComputeCellSizeChange, WithinEightSmallestOfSurface_NeverMayMerge)  // width w, wanted 4 w, depth 7.99 w: kKeep;
      // depth 8 w: kMayMerge
  TEST(ComputeCellSizeChange, TouchingAirGasSolidOrPocket_NeverMayMerge)  // one case per flag: kKeep
  ```
- [ ] **Step 2: Old numbers:** `DecideAdaptiveWidth3D` (`native/src/adaptive_width_policy_3d.cc:53-108`) with every
  demand set to the wanted width and zero eligible duration, for 20 (width, wanted) pairs at depth 20 w; record the
  decision (`kRefine` = `kSplit`, coarsen-eligible = `kMayMerge`). `ComputeCellSizeChange, MatchesProofOfConcept`:
  equal.
- [ ] **Step 3: Run** (fail). **Step 4: Implement.** **Step 5:** gate. **Commit.** Subject:
  `Decide the change each cell asks for`.

### Task 7a: The size of the velocity gradient

**Amended 2026-11-15 (A62):** new task (`plan-decisions.md` "Owner answers A40–A64 applied" item 2, "Velocity
gradient").

**Kind:** numerics

**Files:**
- Create: `native/fluidity/cell_resize/velocity_gradient_math.h` (shared), `velocity_gradient_math_test.cc`
- Modify: `native/fluidity/cell_resize/CMakeLists.txt`

**Interfaces:**
- Consumes: grid `GridTables`, `CellPlaceIndex`, `FindFacePieces`, `FindTouchingCells`, `ComputeFaceArea`,
  `ComputeCellCenterDistance`, `ComputeCellWidth`; pressure `CellContent`.
- Produces design 3.2 `ComputeVelocityGradientSize(const grid::GridTables& tables, const double* velocity,
  grid::CellPlaceIndex cell)` (1/s). **Plan decisions (order):** the nine entries `G[i][j] = ∂u_i/∂x_j` in row order
  `i = x, y, z`, `j = x, y, z`. Diagonal `G[a][a]` = (area-weighted mean of the `+a` face's pieces - the same of the
  `-a` face) / cell width, each mean summing pieces in ascending `FacePieceId`, area times velocity then divided by
  the face's whole area. Off-diagonal `G[i][j]`, `i != j`: the cell-centre value of `u_i` (mean of its two `i`
  faces, as above, halved) in the face-touching cell on the `+j` side minus that on the `-j` side, over the distance
  between their centres (`grid::ComputeCellCenterDistance` along `j`); a bigger neighbour gives its centre value;
  four smaller ones give their mean in ascending `CellId`; a missing side (grid edge, solid or air cell) uses the
  cell itself with half the distance (one-sided). Norm `sqrt(sum G[i][j]^2)` summed in row order. 0 when the cell's
  `CellContent` is not liquid. The caller passes `velocity = velocity_after_pressure.buffer().data()` of the
  committed liquid set.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `LINKS fluidity_grid_test_support fluidity_base_test_support`; grids
  from `grid_test_support`, every cell liquid unless named):

  ```cpp
  TEST(ComputeVelocityGradientSize, UniformField_IsZero)    // u = {0.3, -0.2, 0.1} m/s on a mixed-size grid: exactly 0
  TEST(ComputeVelocityGradientSize, LinearField_ExactGradient)  // u = A x + b, A = {{0.5, 2, 0}, {-1, 0.25, 0.3},
      // {0, 0.7, -0.75}}: sqrt(sum A_ij^2) = 2.5406692 in every cell with same-size neighbours and in a cell with
      // four smaller neighbours on one side, 1e-13 relative
  TEST(ComputeVelocityGradientSize, BiggerNeighbour_WithinItsOffset)  // the same field, a cell beside a bigger one:
      // a bigger neighbour's centre sits w / 2 off the axis in each other direction, so G[i][j] is off by at most
      // (|A_ik| + |A_il|) * (w / 2) / (1.5 w); the test asserts that bound (plan decision: first order there)
  TEST(ComputeVelocityGradientSize, RigidRotation_IsSqrtTwoOmega)  // u = Ω x r, Ω = 10 rad/s about z:
      // sqrt(2) * 10 = 14.142136, 1e-13 relative
  TEST(ComputeVelocityGradientSize, AirCell_IsZero)
  TEST(ComputeVelocityGradientSize, SameBitsInEveryCallOrder)  // all cells forward and in reverse: same bits
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (`FLUIDITY_CPU_AND_GPU`, no allocation, no contraction).
  **Step 4:** gate. **Commit.** Subject: `Measure the size of the velocity gradient in a cell`. New names:
  `ComputeVelocityGradientSize`.

### Task 8: `SplitCell` and `MergeCells` on the cell list

**Kind:** numerics

**Files:**
- Create: `native/fluidity/cell_resize/split_and_merge_cells.h` (CPU only), `split_and_merge_cells_cpu.cc`,
  `split_and_merge_cells_test.cc`

**Interfaces:**
- Produces design 3.3 `SplitCell`, `MergeCells`, `ResizePlan` (the list stays sorted by `grid::ComesBefore`; one
  `std::vector` insert or erase per call).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, frame width 1 m, 3 levels, the 64 level-0 cells of one block):
  `SplitCell, ThenMergeCells_GivesStartingList` (design 7 "Split, merge on the list": split `{1,1,1}`, merge it back:
  the starting list; `grid::CheckSizeStepRule` passes after each call); `SplitCell, SmallestLevel_IsBadInput`;
  `SplitCell, NotInList_IsBadInput`; `MergeCells, SiblingMissing_IsBadInput` (failure `cell` = the big cell).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `grid::ComputeSmallCell`, `ComputeBigCell`, `ComesBefore`.
  **Step 4:** gate. **Commit.** Subject: `Split and merge cells in the sorted cell list`.

### Task 9: Cells that must also split

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test links `fluidity_grid_test_support` through `LINKS`.

**Files:**
- Modify: `split_and_merge_cells.h`, `split_and_merge_cells_cpu.cc`, `split_and_merge_cells_test.cc`

**Interfaces:**
- Produces design 3.3 `FindCellsThatMustSplit`. **Plan decision:** it fails `kTooManyCellBlocks` when
  `active cells + 7 * (cells that must split) > max_cell_count`; `ResizeCells` passes `64 * limits.max_cell_blocks`.
  Ported search: `native/src/adaptive_mac_topology_3d.cc:345-424` (26 probes per split cell, repeated), without its
  4,096-cell limit; probes use `grid::FindCellAt` on the committed tables.
- **Amended 2026-11-15 (A30; reconcile item 32):** `Requires:` `committed` are CPU tables (non-null arrays), checked
  by a debug assertion; a GPU grid's version-0 tables are a caller error, not `kBadInput`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, frame width 1 m, 3 levels):

  ```cpp
  TEST(FindCellsThatMustSplit, CornerTwoLevelsBigger_HandMadeList)  // design 7 "Closure", the old paper's corner
      // case: level-0 A {0,0,0}; the other seven level-0 cells of the 2 x 2 x 2 region split into level 1 (56
      // cells); request level-1 {2,2,2} (touches A's corner at (1,1,1)): result exactly {A, {2,2,2}} in fixed order
  TEST(FindCellsThatMustSplit, ChainOfForcedSplits_Repeats)  // 4 levels, one request forces two more levels
  TEST(FindCellsThatMustSplit, AboveMaxCellCount_IsTooManyCellBlocks)
  TEST(FindCellsThatMustSplit, ResultPassesSizeStepRule)    // 100 random grids (fluidity_grid_test_support), 3 requests
      // each: applying SplitCell to every result cell gives a list CheckSizeStepRule accepts
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step N:` from `<proc-find-cells-that-must-split>`.
  **Step 4:** gate. **Commit.** Subject: `Find the cells a split forces to split too`.

### Task 10: `CellResize`, its settings, buffers and memory estimate entries

**Amended 2026-11-14 (A17):** a CPU copy of the committed grid for a simulation whose grid is on the GPU.

**Amended 2026-11-14 (reconcile):** the grid copies depend on the layout (`GridCopies`): in a batched GPU run the
two CPU grids are the step's, so `CellResize` neither allocates nor counts them; `BuildGridMemoryEstimate(kCpu, 1, limits.max_cell_blocks)`
with its storage count.

**Amended 2026-11-14 (Foundation as built):** `CellResize` also owns a `SumBuffers`, an `OrderBuffers` and a one-word changed flag, built once and
counted in its memory estimate; the test library is made with `fluidity_add_test_library`.

**Amended 2026-11-14 (A31, A30):** "scratch" wording replaced by "working buffers"; the estimate test is
`WorkingBufferEntriesMatchTheBuffers`; `ToCellResizeFailure` no longer maps a processor mismatch (none exists).

**Amended 2026-11-15 (A64, A41, A63, A58; reconcile item 31):** the region settings, `kMaxFilledShare`,
`RegionTransferLedger`, `ReadLargestGridOnlySpeed`, the `OrderBuffers` and the grid-only rows moved to the later
group (Task 10b); `CellResizeSettings::cell_size` is a `grid::CellSizeSettings`; a new `const grid::ArtistCellSizes&
artist` parameter (CPU views, I/O's `RunScene::artist_cell_sizes()`) uploaded once into two rows; new rows `"cell resize: size changes"` and the wait counters' "after" rows; every `kRunOn` row is
`kWholeRun`; `BuildCellResize` takes `RunMemory& run_memory, RunMemory& cpu_run_memory` last (correction, reconcile); the estimate tests become one
`IntoRunMemory_HandsOutExactlyTheEstimate`.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/cell_resize/resize_cells.h` (types and `BuildCellResize` part), `cell_resize.cc`,
  `cell_resize_access.h` (private, CPU only), `cell_resize_test.cc`, `cell_resize_no_gpu_test.cc`; test-only
  library `fluidity_cell_resize_test_support`: `cell_resize_test_support.h`, `cell_resize_test_support.cc`
- Modify: `native/fluidity/cell_resize/CMakeLists.txt` (`fluidity_add_test_library(fluidity_cell_resize_test_support
  SOURCES cell_resize_test_support.cc LINKS fluidity_step_test_support fluidity_grid_test_support
  fluidity_base_test_support)`; `cell_resize_test` `LABEL cpu`, `cell_resize_no_gpu_test` `LABEL no_gpu`, both with
  `LINKS fluidity_cell_resize_test_support`)

**Interfaces:**
- Produces design 3.4 `kAmountTolerance`, `CellResizeSettings`, `ResizeCellsReport`, `CellResize`,
  `BuildCellResize(settings, limits, ingredient_count, const grid::ArtistCellSizes& artist, RunMemory& run_memory,
  RunMemory& cpu_run_memory)`, `ReadLastReport`,
  `BuildCellResizeMemoryEstimate(ingredient_count, grid_run_on, grid_copies, cell_size_region_count,
  cell_size_map_pixel_count)`.
- **Plan decisions:** settings checks (`kBadInput`): every value finite; reaches positive; `shear_width_factor >= 0`;
  `merge_width_share` in `(0, 1]`; `merge_wait_duration >= 0`; `min_merge_wait_steps` 0 to 255;
  `max_cell_splits_per_resize >= 1`; every artist region with `low_corner < high_corner` on each axis and
  `size_level <= smallest_size_level`; `map_levels.size() == map_width * map_height`, every level `<=
  smallest_size_level`, `domain_width_x > 0` and `domain_width_y > 0` when a map is given; `CheckCanRunOn(run_on)`.
  `Requires:` `run_on == run_memory.run_on()`, `cpu_run_memory.run_on() == kCpu` (debug assertions, A30). `CellResize` holds, at
  `limits.max_cell_blocks` on `run_on`, the rows below (names `"cell resize: <row>"`, `sizeof` of the stored type
  with a `static_assert` each, one entry per buffer, `where = kRunOn`, all `kWholeRun`):

  | Row | Element | Per |
  |---|---|---|
  | depth below surface, wanted width, merge durations after, held velocity gradient size after, merge weight | `double` x 5 | cell place |
  | size changes (A41), merge wait steps after | `std::uint8_t` x 2 | cell place |
  | resize map cell source, cell old place | `std::uint8_t`, `std::uint32_t` | cell place |
  | resize map face source, face old place | `std::uint8_t`, `std::uint32_t` | face place |
  | block partial sums | `double` x (3 + ingredients) | cell block |
  | cell size regions (A63) | `grid::CellSizeRegion` (56 B) | given count `cell_size_region_count` |
  | cell size map (A63) | `std::uint8_t` | given count `cell_size_map_pixel_count` |
  | sum buffers (amended, Foundation as built) | `ComputeSumBuffersSize(run_on).bytes_per_item` bytes | cell block |
  | sum buffers fixed bytes | 1 byte x `ComputeSumBuffersSize(run_on).fixed_bytes` | given count |
  | changed flag (amended) | `std::uint32_t` | given count 1 |

  `cell_resize_access.h` gives the operations of this module writable views of these buffers, and `grid::ArtistCellSizes`
  views on the two artist rows. `BuildCellResize` copies `artist`'s regions and map levels into the two artist rows with one
  `WriteBuffer` each (once per run, not counted in `ResizeCellsReport`); empty inputs build zero-size rows.
- **Plan decision (amended 2026-11-14, Foundation as built; 2026-11-15, A58):** `CellResize` owns `SumBuffers
  sum_buffers` (built by `BuildSumBuffers(run_on, limits.max_cell_blocks, run_memory)`: the block-partial sums of
  Tasks 23 and 24), reached through `cell_resize_access.h`; per-step calls pass these buffers and allocate nothing.
  The estimate's working-buffers entries use the `CellResize`'s own processor, which its arguments already fix:
  `kCpu` when `grid_copies == kHeldByStep` (the batched GPU run resizes on the CPU, Task 43), else `grid_run_on`. A
  failed build (`BufferError::kOutOfMemory` or `kRunMemoryTooSmall`) is `kOutOfMemory` with `magnitude` = the bytes
  asked; `kRunOnCallFailed` stays `kRunOnCallFailed`; the mapping function `ToCellResizeFailure(BufferError)`
  (private, `cell_resize_access.h`) is the one place a `BufferError`, and through `ToBufferError` a
  `RunOnCallStatus`, becomes a `CellResizeFailure`.
- **Plan decision (A17):** `CellResizeSettings` gains `RunOn grid_run_on = RunOn::kCpu;` (the processor of the
  simulation's grid; the caller's choice, A12). With `grid_run_on == kGpu` the rows add `"cell resize: committed
  grid copy"` and `"cell resize: resized grid on the CPU"` (`where = kCpu`, pieces of `cpu_run_memory`) and `"cell resize:
  cell list upload"` (8 B per cell place, `where = kRunOn`, a piece of `run_memory`), used by Task 24.
- **Plan decision (reconcile, coordinator item 6):** `enum class GridCopies : std::uint8_t { kNone, kHeldByCellResize,
  kHeldByStep };` `CellResizeSettings` gains `GridCopies grid_copies = GridCopies::kNone;`. `kNone` with a CPU grid;
  `kHeldByCellResize` in a one-piece GPU run (the two CPU grid rows above, built as two `GridBuffers` sharing one CPU
  `grid::GridBuilderBuffers`, entries `grid::BuildGridMemoryEstimate(kCpu, 2, limits.max_cell_blocks,
  grid::BuilderBuffersSharing::kOneForAllGrids)` with `where = kCpu`); `kHeldByStep` in a batched GPU run (no CPU
  grid row: the committed grid is `step::FindCpuGridCopy(simulation)` and the resized CPU grid is built into
  `ResizeTry::free_cpu_grid_copy_buffers()`, step plan Task 32). `kBadInput`: `kHeldByCellResize` or `kHeldByStep`
  with `grid_run_on == kCpu`, `kNone` with `kGpu`. I/O sets it from the run plan's mode.
- **Plan decision (A58):** `run_memory` and `cpu_run_memory` are the run's blocks (the step's, which
  `step::BuildSimulation` sized from the plan's estimate, this module's entries included); they must outlive the
  `CellResize`. After `step::ApplyLimits`
  (new block) the caller builds a new `CellResize` into the new block before the next `ResizeCells` (design 11,
  Open 2).
- Test support (plan decision; amended 2026-11-15, A58): `std::expected<step::Simulation, step::StepFailure>
  BuildResizeTestSimulation(const step::TestScene& scene, std::span<const MemoryEstimateEntry> cell_resize_estimate,
  const step::StepCallbacks& callbacks)` (plan from `step::BuildRunPlan` with ample memory and the entries added,
  then `step::BuildSimulation`, whose block then has room for a `CellResize`).

- [ ] **Step 1: Failing tests:** `BuildCellResize, Defaults_Succeed`; `BuildCellResize,
  ArtistRegionLevelBelowSmallest_IsBadInput` (region `size_level` 5 with `smallest_size_level` 4);
  `BuildCellResize, MapOfWrongSize_IsBadInput` (4 x 4 map, 15 levels); `BuildCellResize,
  IntoRunMemory_HandsOutExactlyTheEstimate` (A58: CPU run, one block `BuildRunMemory(kCpu, ComputeRunMemoryBytes(
  BuildCellResizeMemoryEstimate(3, kCpu, kNone, 2, 16), counts at limits.max_cell_blocks = 40, kRunOn) +
  ComputeRunMemoryBytes(..., kCpu))` passed as both blocks, then `BuildCellResize` with 2 regions and a 4 x 4 map:
  `handed_out_bytes() == reserved_bytes()`, exact; the same with `kHeldByCellResize` on CPU buffers and two blocks,
  each exact; replaces
  `RowsMatchWhatIsBuilt` and `WorkingBufferEntriesMatchTheBuffers`); `BuildCellResize,
  RunMemoryOnePieceShort_IsOutOfMemory` (the same block less 256 B; `GTEST_SKIP` in debug builds, which assert:
  `kOutOfMemory`, `magnitude` == the bytes the failing piece asked); `BuildCellResize, ArtistInputsUploadedOnce`
  (the region row reads back equal to the input, bytes); `ReadLastReport, BeforeFirstResize_NotResized`;
  `BuildCellResizeMemoryEstimate, GridOnGpu_AddsTheGridCopies` (A17); `BuildCellResizeMemoryEstimate,
  HeldByStep_NoGridCopyRows` (reconcile: no `"cell resize: committed grid copy"` or `"resized grid on the CPU"`
  entry, and `BuildCellResize` builds none); `BuildCellResize, GridCopiesNotMatchingGridRunOn_IsBadInput`
  (reconcile); `BuildCellResizeMemoryEstimate, EveryRunOnRowIsWholeRun` (A58); `LABEL no_gpu`:
  `BuildCellResize, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Hold the settings and buffers of the cell resize`.

### Task 11: `ComputeDepthBelowSurface` on the CPU

**Amended 2026-11-15 (A64):** `RegionKind`, `GridOnlyAmounts` and `GridOnlyAmountsToChange` moved to the later
group (Task 11b); the header is `depth_below_surface.h` (plan decision, was `regions.h`); `ResizeCells` passes
`max_depth = surface_reach_in_cell_widths * widths.width[0]` (design 5.2: beyond it the depth term is capped at the
biggest width).

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** `depth_below_surface_cpu.cc` holds the public function (checks, dispatch, CPU loop); until Task 12
the `kGpu` branch returns `kRunOnCallFailed`; the per-cell pass update is a shared function Task 12's GPU loop reuses.

**Files:**
- Create: `native/fluidity/cell_resize/depth_below_surface.h`, `depth_below_surface_cpu.cc`,
  `depth_below_surface_steps.h` (shared: `ComputeCellDepthInPass`, one cell place of one pass, reading the previous
  pass's depths and writing the next, so the CPU and GPU passes are the same Jacobi passes), `depth_below_surface_test.cc`

**Interfaces:**
- Produces design 3.5 `ComputeDepthBelowSurface`.
- **Plan decisions (design 5.2):** depth 0 in a liquid cell (`content == kLiquid`) with `air_volume > 0` or with a
  face piece whose `ends == kLiquidToAir`; a cell whose content is `kAir` or `kGas` never counts as liquid. Each
  other liquid cell takes the minimum over touching liquid cells (`grid::FindTouchingCells`, 26 directions) of
  their depth plus `grid::ComputeCellCenterDistance`, repeated in passes over cell places until a pass changes
  nothing, capped at `max_depth`. Paths leaving `blocks` and its touching border are not seen; `ResizeCells` passes
  every block. `kBadInput` for `max_depth <= 0`; `kWrongGridVersion` for an array of another version or block count.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `LINKS fluidity_grid_test_support fluidity_base_test_support`;
  hand-built `GeometrySnapshot` on grids from `grid_test_support`; the expected values from a test-side Dijkstra over
  touching cells):

  ```cpp
  TEST(ComputeDepthBelowSurface, FlatSurfaceOverThreeSizes_DistanceFromSurfaceCells)  // design 7 "Depth": 1e-13 w
  TEST(ComputeDepthBelowSurface, ClosedBubble_CellsBelowMeasureFromIt)
  TEST(ComputeDepthBelowSurface, TrackedGasAbove_MeasuresFromTheGas)    // gas cells' content kGas
  TEST(ComputeDepthBelowSurface, DeepCell_CappedAtMaxDepth)
  TEST(ComputeDepthBelowSurface, OutsideLiquid_IsInfinity)
  TEST(ComputeDepthBelowSurface, ArrayOfOtherVersion_IsWrongGridVersion)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Measure how deep each liquid cell lies below the surface`.

### Task 12: `ComputeDepthBelowSurface` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** GPU half in `depth_below_surface_gpu.h`/`.cu` returning `RunOnCallStatus`, the GPU function a thin
loop over `ComputeCellDepthInPass`, a `cpu` steps test over every thread block and thread, digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/cell_resize/depth_below_surface_gpu.cu`, `depth_below_surface_gpu.h` (private:
  `RunOnCallStatus ComputeDepthBelowSurfacePassOnGpu(...)`, one pass), `depth_below_surface_steps_test.cc`,
  `depth_below_surface_gpu_test.cc`, `cell_resize_no_gpu_test.cc` (extend)
- Modify: `depth_below_surface_cpu.cc` (the `kGpu` branch under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`
  (`fluidity_add_gpu_sources`; the steps test `LABEL cpu`, the GPU test `LABEL gpu`, both `LINKS
  fluidity_cell_resize_test_support fluidity_base_test_support`)

- [ ] **Step 1: Tests.** `LABEL cpu` (`depth_below_surface_steps_test.cc`, the main evidence until the hardware
  run): `ComputeCellDepthInPass, EveryThreadOrder_EqualsCpuPasses` (Task 11 cases: each pass run for every thread
  block and thread of `ComputeLaunchSize(cell place count)`, in launch order and in reverse, until no cell changes:
  the depths' bits equal the CPU function's, same pass count). `LABEL gpu`: `ComputeDepthBelowSurface,
  OnGpu_EqualsCpu` (Task 11 cases; equal bytes expected, demanded within `1e-13` of the cell width; depths passed to
  `WriteResultDigest`); `ComputeDepthBelowSurface, OnGpuTwice_GivesIdenticalBytes`. `LABEL no_gpu`:
  `ComputeDepthBelowSurface, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement:** one thread per cell place per pass through `ApplyGpuFunction`; the changed flag is
  `CellResize`'s one-word buffer (Task 10), cleared before and read back with `ReadBuffer` after each pass by the
  CPU half (design 6, "until a pass changes nothing"); a status other than `kDone` maps through
  `ToCellResizeFailure(ToBufferError(status))`.
- [ ] **Step 3:** gate. **Commit.** Subject: `Measure the depth below the surface on the GPU`. Body: `CUDA:
  compiled, not run`.

### Task 13: `ComputeCellSizeChanges` on the CPU

**Amended 2026-11-15 (A41, A62, A63):** replaces the private facts pass (`CellSizeFactsInputs`, `MergeVeto`,
`ComputeCellSizeFacts`) by the public pass of `plan-decisions.md` item 2 ("The pass", "Hysteresis", "Held maximum
and wait counters"): it fills `grid::CellSizeFacts`, calls grid's `ComputeWantedCellWidth`, applies the merge test
and the wait of time and steps, and writes one `size_change` byte per cell place. The merge wait moved here from
Task 14. The GPU half is Task 13a.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** two halves; per-cell work in a shared steps header with a `cpu` test
over every thread block and thread.

**Files:**
- Create: `native/fluidity/cell_resize/cell_size_changes.h`, `cell_size_changes_cpu.cc` (checks, dispatch, CPU
  loop; until Task 13a the `kGpu` branch returns `kRunOnCallFailed`), `cell_size_changes_steps.h` (shared:
  `ComputeCellSizeFactsOfCell` and `ComputeWantedSizeOfCell` (part 1, one cell place), `ComputeMergeTestOfCell`
  (part 2, one cell place), from raw arrays), `cell_size_changes_test.cc`
- Modify: `CMakeLists.txt` (`LINKS fluidity_cell_resize_test_support fluidity_base_test_support`)

**Interfaces:**
- Consumes: grid `CellSizeSettings`, `CellWidthsByLevel`, `CellSizeFacts`, `ArtistCellSizes`,
  `ComputeWantedCellWidth`, `ComputeArtistSizeLevel` (grid plan); Task 7 `ComputeCellSizeChange`; Task 7a
  `ComputeVelocityGradientSize`; Task 11 `ComputeDepthBelowSurface`'s output.
- Produces design 3.2 `CellSizeChangesPart`, `CellSizeChangesInputs`, `CellSizeChangesOutputs`,
  `ComputeCellSizeChanges(inputs, blocks, part, outputs, cell_resize)`. **Plan decisions (`// Step N:` from
  `<proc-compute-cell-size-changes>`):**
  1. *Part `kWantedWidthsAndChanges`, per active cell place of `blocks`:* facts: `cell_width =
     widths.width[level]`; `depth_below_surface` from the depth array; `distance_to_solid` = smallest corner value of
     the cell in `SolidCoverage`; `held = max(committed held_velocity_gradient_size, ComputeVelocityGradientSize)`
     as `velocity_gradient_size`; `artist_size_level = ComputeArtistSizeLevel(artist, cell centre)`;
     `holds_particles_that_keep_their_count` = a particle of a `kKeepsItsCount` material in the cell (grouped order)
     or a cut solid in it; `touches_air_pocket` = the cell or a touching cell has `pocket_of_cell != kNoAirPocket`;
     `wants_smallest_cells` = the cell or a touching cell has `gas_volume > 0` or `content == kGas`;
     `touches_air_gas_or_solid` = a touching cell's content is `kAir`, `kGas` or `kSolid`, or the cell has a solid
     part. Then `wanted_width = ComputeWantedCellWidth(settings, widths, reference_speed, facts)` and
     `size_change = ComputeCellSizeChange(settings, widths, facts, wanted_width)`; `held` is kept in the "held
     velocity gradient size after" row for part 2.
  2. *Part `kMergeTest`, per active cell place:* the cell's 8 siblings (`grid::ComputeBigCell`, then
     `ComputeSmallCell` k = 0..7, `FindCellPlace`) are all active and all `kMayMerge`, and every active cell touching
     the big cell satisfies `2 w <= merge_width_share * wanted` at the big cell's width (its `wanted_width` from part
     1). Holds: `merge_ready_duration_after = committed + elapsed_since_last_merge_test`,
     `merge_ready_step_count_after = min(255, committed + steps_since_last_merge_test)`, held after = part 1's
     `held`. Breaks: duration 0, step count 0, held after = the current `ComputeVelocityGradientSize` (A63 reset).
     Then a `kMayMerge` cell becomes `kMergeReady` when the smallest duration of its 8 siblings `>=
     merge_wait_duration` **and** their smallest step count `>= min_merge_wait_steps` (siblings read the same
     inputs, so all 8 decide alike). Order inside a cell is fixed; no write reads another cell's part-2 output.
  3. One level per cell per resize: the pass never looks past the cell's own level; Task 14 splits each cell once
     and merges each set of siblings once.
  Errors: `kBadInput` (a span of the wrong length, `steps_since_last_merge_test` above 255 is saturated, not an
  error), `kWrongGridVersion` (an array of another version or block count), `kNotFinite` (a non-finite velocity in a
  liquid cell; failure `cell` the first in fixed order), `kRunOnCallFailed`. `Requires:` every array and
  `cell_resize`'s buffers on the grid's processor (A30).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; hand-built grids and arrays from `grid_test_support`, `w_min` 0.01 m, 4
  levels 0.08 .. 0.01 m, every cell liquid unless named, `reference_speed` given):

  ```cpp
  TEST(ComputeCellSizeChanges, RigidRotation_Splits)     // literature-cell-size.md 5(f)4: u = Ω x r, Ω = 10 rad/s about z,
      // U_ref = 1 m/s, deep (depth 1 m), cells at 0.04 m: gradient size sqrt(2) Ω = 14.142136, w_shear = 0.25 /
      // 14.142136 = 0.017678 m: every cell kSplit, wanted_width 0.017678 within 1e-15 relative; cells at 0.01 m kKeep
  TEST(ComputeCellSizeChanges, UniformTranslation_NoShearSplit)  // u = {0.3, -0.2, 0.1}, deep, cells 0.04 m, depth term
      // 0.125 m: no kSplit
  TEST(ComputeCellSizeChanges, MergeAfterTimeAndSteps)   // deep calm cells at 0.01 m, wanted 0.08 m: elapsed 0.05 s and
      // 1 step per call: kMayMerge for calls 1..4 (0.20 s), kMayMerge at call 5 (0.25 s, 5 steps < 8), kMergeReady
      // at call 8 (0.40 s, 8 steps: steps decide); with elapsed 0.02 s per call: first at call 13 (0.26 s; time decides)
  TEST(ComputeCellSizeChanges, BrokenOnce_WaitStartsAgain)  // one call with a touching air cell: counters 0, then 8
      // more calls before kMergeReady
  TEST(ComputeCellSizeChanges, ZeroElapsed_CountersUnchanged)  // elapsed 0, 0 steps: after rows == committed, bytes
  TEST(ComputeCellSizeChanges, ShearBurstEnds_HeldForTheWait)  // Review Focus 6: rotation for 1 call then rest: held
      // after == 14.142136 while the test holds, so the shear term keeps wanted 0.017678 and the cells kSplit/kKeep,
      // never kMayMerge; a break resets held to the current 0
  TEST(ComputeCellSizeChanges, NeighbourWantsSmaller_NoMerge)
  TEST(ComputeCellSizeChanges, SurfaceCell_IsSmallest)    // A61: depth 0: wanted_width == w_min whatever the shear and
      // artist terms; a 0.02 m cell there kSplit
  TEST(ComputeCellSizeChanges, ArtistRegion_AsksFiner)   // a region asking level 3 over deep calm cells at 0.04 m: kSplit
      // inside the box, kMayMerge outside
  TEST(ComputeCellSizeChanges, SnowCell_KeepsItsSize)    // BuildSnowAndJellyScene after one step: snow cells
      // kKeepsItsSize
  TEST(ComputeCellSizeChanges, GasCells_WantSmallest)    // BuildSmokeOverDeepWaterScene after one step
  TEST(ComputeCellSizeChanges, SolidDistanceIsSmallestCorner)  // BuildFreeBoxScene(500)
  TEST(ComputeCellSizeChanges, AnyBatchCut_SameBytes)    // A41: the shear scene of RigidRotation_Splits on a 4-block
      // grid: part 1 over each range then part 2 over each range, for one range, 1-block and 3-block ranges: every
      // output's bytes equal the whole-grid call's
  ```
  `LABEL cpu` (`cell_size_changes_steps_test.cc`, main evidence until the hardware run, written here, extended in
  Task 13a): `CellSizeChangesSteps, EveryThreadOrder_EqualsCpu` (the rotation and snow cases: each part's steps
  function for every thread block and thread of `ComputeLaunchSize(cell place count)`, launch order and reverse:
  outputs' bytes equal the CPU function's).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (one iteration per cell place; particles read in grouped order).
  **Step 4:** gate. **Commit.** Subject: `Decide each cell's size change with the shared rule and its wait`. New
  names: `ComputeCellSizeChanges`, `CellSizeChangesPart`, `CellSizeChangesInputs`, `CellSizeChangesOutputs`.

### Task 13a: `ComputeCellSizeChanges` on the GPU

**Amended 2026-11-15 (A41):** new task (was the GPU part of Task 13).

**Kind:** device

**Files:**
- Create: `native/fluidity/cell_resize/cell_size_changes_gpu.h` (private: `RunOnCallStatus
  ComputeCellSizeChangesOnGpu(...)`, one part per call), `cell_size_changes_gpu.cu`, `cell_size_changes_steps_test.cc`
  (extend), `cell_size_changes_gpu_test.cc`; extend `cell_resize_no_gpu_test.cc`
- Modify: `cell_size_changes_cpu.cc` (the `kGpu` branch under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`
  (`fluidity_add_gpu_sources`; the GPU test `LABEL gpu`)

- [ ] **Step 1: Tests.** `LABEL gpu`: `ComputeCellSizeChanges, OnGpu_SameBytesAsCpu` (A41: the shear scene of Task 13
  and the snow case: `size_change`, `wanted_width`, the three after rows equal bytes, all to `WriteResultDigest`);
  `ComputeCellSizeChanges, OnGpuTwice_GivesIdenticalBytes`. `LABEL no_gpu`: `ComputeCellSizeChanges,
  OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement:** one thread per cell place per part through `ApplyGpuFunction`, each a thin loop over the
  Task 13 steps functions; the artist regions and map are `CellResize`'s rows (Task 10), `CellWidthsByLevel` passed by
  value; the first failing cell by an integer `atomicMin` over cell places; the status maps through
  `ToCellResizeFailure(ToBufferError(status))`. **Step 3:** gate. **Commit.** Subject: `Decide cell size changes on
  the GPU`. Body: `CUDA: compiled, not run`.

### Task 14: The resize plan

**Amended 2026-11-15 (A41, A63, A64):** the plan reads the `size_change` bytes of Task 13 (on a GPU grid read back in
one counted copy, 1 B per cell place) instead of the wanted widths, changes, vetoes, region bytes and durations; the
merge wait moved to Task 13; one level per cell per resize.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test links `fluidity_grid_test_support` and `fluidity_base_test_support` through `LINKS`.

**Files:**
- Create: `native/fluidity/cell_resize/resize_plan.h` (private, CPU only), `resize_plan_cpu.cc`, `resize_plan_test.cc`

**Interfaces:**
- Produces (plan decision, private; design 5.3 rules 3 to 5; amended 2026-11-15):

  ```cpp
  struct ResizePlanInputs {
    const grid::StaggeredGrid* committed_grid;              // CPU grid (Requires, A30).
    const grid::CellSizeSettings* settings;
    std::span<const std::uint8_t> size_change;              // CellSizeChange per committed cell place (CPU memory).
    std::uint64_t max_cell_count = 0;                       // 64 * limits.max_cell_blocks.
  };
  std::expected<ResizePlan, CellResizeFailure> BuildResizePlan(const ResizePlanInputs& inputs);
  ```
  The first `max_cell_splits_per_resize` `kSplit` cells in fixed order, then `FindCellsThatMustSplit`; a split whose
  closure reaches a `kKeepsItsSize` cell is removed and counted in `refused_split_count`, then the closure is found
  again without it. Each set of 8 `kMergeReady` siblings merges unless one of them, or a cell touching the big cell,
  is in the split list or its closure, or the merge would break the size-step rule against them; a refused set is
  counted in `refused_merge_count`. A cell is split at most once and a set merged at most once per call (one level).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `LINKS fluidity_grid_test_support fluidity_base_test_support`;
  hand-built `size_change` on grids from `grid_test_support`): `BuildResizePlan, MoreSplitsThanTheLimit_FirstInFixedOrder`
  (5 wanted, limit 3: `wanted_split_count` 5, the first 3 and their closure); `BuildResizePlan,
  ClosureReachesKeepSizeCell_SplitRefusedAndRerun`; `BuildResizePlan, EightMergeReady_Merge`; `BuildResizePlan,
  SevenMergeReady_NoMerge`; `BuildResizePlan, MergeAgainstASplit_Refused`; `BuildResizePlan,
  OneLevelPerResize` (a cell whose `size_change` is `kSplit` two levels too wide: split once, its small cells not
  split again in this call).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Plan which cells split and merge in one resize`.

### Task 15: `ResizeMap` and the remap of cell values on the CPU

**Amended 2026-11-15 (A64; A30):** `ApplyResizeToCellAmounts` (the grid-only amounts) and its test moved to the later
group (Task 15b); `BuildResizeMap` `Requires:` two CPU grids (debug assertion, A30).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/cell_resize/remap_grid_values.h`, `remap_grid_values_cpu.cc`, `remap_grid_values_test.cc`

**Interfaces:**
- Produces design 3.6 `CellSource`, `ResizeMap`, `BuildResizeMap`, `CellValueKind`, `ApplyResizeToCellValues`
  (explicit instantiations for `double`, `std::uint8_t`, `std::uint32_t`).
- **Plan decisions:** `ResizeMap` keeps pointers to both grids (valid during the transaction) and, per new cell
  place, the source and one old place (the big cell of a split; small cell 0 of a merge: siblings share one old
  block, so the other seven follow by local offset). For `kAmount` the neighbours of a split are the old cells of the
  big cell's level across its 6 faces; with `old_weight` given, a neighbour counts only when its weight is above 0.
  `kLabel`'s missing value is `std::numeric_limits<Value>::max()` (equal to `pressure::kNoAirPocket` for
  `std::uint32_t`). `kPerUnitOfWeight` with all weights 0 takes the equal mean.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; old grid with one split cell and one merge elsewhere):
  `BuildResizeMap, SplitAndMerge_SourceOfEveryPlace` (no `kNewCell`); `ApplyResizeToCellValues,
  PerUnitOfWeight_SplitCopiesMergeWeightedMean` (weights 1..8, 1 rounding step; null weight: equal mean);
  `ApplyResizeToCellValues, Label_MergeTakesSmallestPresent` (`{max, 5, 3, max, ...}` -> 3; all `max` -> `max`);
  `ApplyResizeToCellValues, Amount_SplitSumsAndMergeSums` (2 rounding steps, the gas volume's case);
  `ApplyResizeToCellValues, OldValuesOfAnotherGrid_IsWrongGridVersion`; `ApplyResizeToCellValues,
  NaNInOldValues_IsNotFinite` (failure `cell` = first new cell in fixed order).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (one iteration per new cell place gathers its sources). **Step
  4:** gate. **Commit.** Subject: `Map a grid onto its resized version and carry cell values over`.

### Task 16: Face velocities on the CPU

**Amended 2026-11-15 (A64):** `ApplyResizeToIngredientVolumes` (the grid-only record's ingredient volumes) and its
test moved to the later group (Task 16b).

**Kind:** numerics

**Files:**
- Modify: `remap_grid_values.h`, `remap_grid_values_cpu.cc`, `remap_grid_values_test.cc`

**Interfaces:**
- Produces design 3.6 `ApplyResizeToFaceVelocity` (design 5.4 "Face velocity").
  **Plan decision:** face places of the new grid have a private source (`kSamePiece`, `kQuarterOfBigPiece`,
  `kSumOfFourPieces`, `kInnerFaceOfSplit`); flows use the whole piece area (`grid::ComputeFaceArea`), never an
  open area; the 4 pieces of a merged face are summed in ascending `FacePieceId`; the 12 inner faces of a split are
  written by the iteration of its small cell 0 with `ComputeInnerFaceFlows`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ApplyResizeToFaceVelocity, Split_EachSmallCellHasAnEighthOfTheImbalance)  // design 7 "Face flow, split":
      // zero imbalance, then imbalance q: each small net outflow q / 8; each outer big face's flow kept; 1e-13 of
      // the largest flow
  TEST(ApplyResizeToFaceVelocity, Merge_BigFaceFlowIsSumOfPieces)  // design 7: velocities 0.1, 0.2, 0.3, 0.4 on 4
      // pieces: big velocity x area == their flow sum within 1 rounding step
  TEST(ApplyResizeToFaceVelocity, MergeBesideSmallCells_PiecesLiveOn)
  TEST(ApplyResizeToFaceVelocity, UnchangedPieces_CopiedByFacePieceId)  // exact
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Carry face velocities onto resized cells`.

### Task 17: The remap on the GPU

**Amended 2026-11-15 (A64):** `ComputeRemappedCellAmounts`, `ComputeRemappedIngredientVolumes` and their GPU tests
moved to the later group (Task 17b).

**Amended 2026-11-14 (A17):** the map's two CPU grids are named (Task 24).

**Amended 2026-11-14 (Foundation as built):** private `remap_grid_values_gpu.h` returning `RunOnCallStatus`; the per-place gathers in
`remap_grid_values_steps.h`, which the CPU loops of Tasks 15, 16 may call and a `cpu` test runs over every thread
block and thread; digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/cell_resize/remap_grid_values_gpu.cu`, `remap_grid_values_gpu.h` (private),
  `remap_grid_values_steps.h` (shared: `ComputeRemappedCellValue`, `ComputeRemappedFaceVelocity`,
  `ComputeInnerFaceVelocitiesOfSplit`,
  one new place each, over raw map and value arrays), `remap_grid_values_steps_test.cc`,
  `remap_grid_values_gpu_test.cc`; extend `cell_resize_no_gpu_test.cc`
- Modify: `remap_grid_values_cpu.cc` (the `kGpu` branches under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

- [ ] **Step 1: Tests.** `LABEL cpu` (`remap_grid_values_steps_test.cc`, the main evidence until the hardware run):
  `RemapGridValuesSteps, EveryThreadOrder_EqualsCpu` (Tasks 15 and 16 cases: each per-place function run for every
  thread block and thread of `ComputeLaunchSize` over new cell places and new face places, launch order and reverse:
  outputs' bits equal the CPU functions'). `LABEL gpu`: `BuildResizeMap, OnGpu_EqualsCpu` (bytes: the map is
  discrete); `ApplyResizeToCellValues, OnGpu_EqualsCpu` (labels equal; others design 6), `ApplyResizeToFaceVelocity,
  OnGpu_EqualsCpu`
  (Tasks 15 and 16 cases, design 6 tolerance; every output to `WriteResultDigest`); `...OnGpuTwice_
  GivesIdenticalBytes` for the face velocity. `LABEL no_gpu`: `BuildResizeMap, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** design 6 "Work": one thread per new cell place, one per new face place, the inner faces by
  the thread of the big cell; each GPU function a grid-stride loop over its steps function, launched with
  `ApplyGpuFunction`. The map is built on the CPU from the two grids' tables and copied up (A17: the
  committed grid's `ReadGrid` copy and the CPU-built resized grid of Task 24).
- [ ] **Step 3:** gate. **Commit.** Subject: `Remap grid values on the GPU`. Body: `CUDA: compiled, not run`.

### Task 18: The carried velocity and the region formulas (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 19: `ApplyGridOnlyRegionToGrid` on the CPU (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 20: `ApplyGridOnlyRegionToGrid` on the GPU (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 21: `MoveGridOnlyLiquid` on the CPU (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 22: `MoveGridOnlyLiquid` on the GPU (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 23: `ComputeLiquidTotals`, CPU and GPU

**Amended 2026-11-15 (A64):** particles only: the `GridOnlyAmounts` parameter is removed (exact signature below,
`plan-decisions.md` item 1: totals over particles); header `liquid_totals.h` (plan decision, was `regions.h`); the
later group (Task 23b) adds the records back.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** two halves with `liquid_totals_gpu.h`; per-block partials in a shared steps function with a `cpu`
test; the sum over blocks with `ComputeTreeSum` and its `SumBuffers`; the GPU comparison within design 6.

**Amended 2026-11-14 (A31, A30):** `ComputeLiquidTotals` has one form, which takes `SumBuffers& sum_buffers` (the
design's form without buffers and the overload pair are gone); buffers on the other processor are a precondition,
not `kBadInput`; `ScratchForm_SameBytesAsFormWithout` removed, `ScratchTooSmall_IsBadInput` renamed.

**Files:**
- Create: `liquid_totals_cpu.cc`, `liquid_totals_gpu.h` (private), `liquid_totals_gpu.cu`, `liquid_totals_steps.h`
  (shared: `ComputeLiquidBlockPartials`, one cell block: its particles in grouped order), `liquid_totals.h`,
  `liquid_totals_test.cc`, `liquid_totals_steps_test.cc`, `liquid_totals_gpu_test.cc`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces design 3.5 `LiquidTotals`, `ComputeLiquidTotals`. **Plan decision:** a private
  `ComputeIngredientTotals` (same rule, one total per scene ingredient) in `liquid_totals_cpu.cc` for the check of
  `ResizeCells` step 8.
- **Plan decision (amended 2026-11-14, Foundation as built; one form since A31, replacing the design's form):**

  ```cpp
  // Allocates nothing: the block partials are summed by ComputeTreeSum with `sum_buffers`. kBadInput when the
  // block count is above sum_buffers.max_count(). Requires: sum_buffers.run_on() is the grid's processor (checked
  // by an assertion in debug builds, A30).
  std::expected<LiquidTotals, CellResizeFailure> ComputeLiquidTotals(
      const particles::ParticlesTry& particles_try, std::span<const std::uint8_t> material_is_counted,
      const grid::StaggeredGrid& grid, grid::CellBlockRange blocks, SumBuffers& sum_buffers);
  ```
  `ResizeCells` passes `cell_resize.sum_buffers`; render output passes its recorder's
  (render output plan Task 31); tests build their own with `BuildSumBuffers(run_on, blocks.count)`.

- [ ] **Step 1: Tests.** `LABEL cpu`: `ComputeLiquidTotals, Particles_TreeSumOfBlocks` (bytes against a test-side
  block-partial sum; was `ParticlesPlusRecords_TreeSumOfBlocks`, A64); `ComputeLiquidTotals, GasMarkersLeftOut` (`BuildSmokeOverDeepWaterScene`, liquid mask);
  `ComputeLiquidTotals, MaskOfWrongLength_IsBadInput`; `ComputeLiquidBlockPartials, EveryThreadOrder_EqualsCpu`
  (steps test, the main evidence until the hardware run: every thread block and thread, launch order and reverse).
  `LABEL gpu`: `ComputeLiquidTotals, OnGpu_EqualsCpu` (within design 6, amounts; totals to `WriteResultDigest`).
  `ComputeLiquidTotals, BuffersTooSmall_IsBadInput` (`SumBuffers` for 1 block, 2 blocks given).
  `LABEL no_gpu`: on the GPU without a GPU. The totals over blocks use `ComputeTreeSum` with the given `SumBuffers` on
  both processors.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Sum the liquid held by particles`. Body: `CUDA: compiled, not run`.

### Task 24: `ResizeCells`, the transaction for cells and grid values

**Amended 2026-11-15 (A41, A63, A64):** step 2 runs `ComputeDepthBelowSurface` and the two parts of
`ComputeCellSizeChanges` with the reference speed from `CommittedStepLimits::largest_liquid_speed`, then reads the
`size_change` bytes back on a GPU grid (one counted copy); the grid-only remaps, region bytes and the
`BuildGridOnlySnapshot` support moved to the later group (Task 24b); the three wait arrays are remapped;
`ResizeCellsReport` loses the ledger and gains the particle totals; `RecordsRemapped_TotalsKept` is rewritten on
the all-particle tank.

**Amended 2026-11-15 (A67):** step 3 "nothing to do" counts mismatched particles: with an empty plan and unchanged
wait arrays it runs `particles::AdjustParticleCount` on the committed grid as both grids and is "nothing to do" only
when `mismatched_particle_count == 0` (the call and its tests land in Task 27; this task leaves the seam: an empty
plan with unchanged wait arrays calls a private `ApplyParticleOnlyPass` that Task 27 fills, returning "no mismatched
particle" until then).

**Amended 2026-11-14 (A29, A31):** `grid::TakeBuffers` (was `ReleaseBuffers`); `ComputeLiquidTotals` in its one form with `cell_resize.sum_buffers`.

**Amended 2026-11-14 (A17):** on a GPU grid the CPU parts read a `ReadGrid` copy; the resized grid is built on the
CPU for the map and on the GPU by the step from the same cell list.

**Amended 2026-11-14 (A17, batched-run buffers):** the report also counts its copies, so the run's planned and
measured copies per step include the resize (`plan-decisions.md` "Copies counted").

**Amended 2026-11-14 (reconcile):** the step receives the uploaded list through `step::BuildResizedGrid`; in a
batched GPU run the CPU resized grid goes to the step with `step::ApplyResizedCpuGridCopy`.

**Amended 2026-11-14 (Foundation as built):** the "changes nothing" promise covers every failed check; a failed GPU call (`kRunOnCallFailed`)
also rejects the resize, but the try's outputs are unspecified and the run stops. Step 8's totals pass
`cell_resize.sum_buffers` to `ComputeLiquidTotals` (its one form, A31).

**Kind:** integration

**Files:**
- Create: `native/fluidity/cell_resize/resize_cells_cpu.cc`, `resize_cells_test.cc`
- Modify: `resize_cells.h`, `cell_resize.cc`, `cell_resize_test_support.h/.cc`

**Interfaces:**
- Produces design 3.4 `ResizeCells` with design 5.1 steps 1 to 6, 8 and 9 (step 7 in Task 27). Body numbered
  `// Step N:` from `<proc-resize-cells>`.
- **Plan decisions:**
  - Step 2 (amended 2026-11-15): `widths` from the committed frame (`grid::ComputeCellWidth` per level, on the CPU,
    each call); `ComputeDepthBelowSurface(..., max_depth = surface_reach_in_cell_widths * widths.width[0], ...)`;
    `reference_speed = grid::ComputeReferenceSpeed(limits.largest_liquid_speed, |gravity|, widths.width[0])` with
    `limits` the committed `step::CommittedStepLimits`; `ComputeCellSizeChanges` part 1 then part 2 over every block
    with `elapsed_since_last_merge_test` (committed time since the last committed resize; the first call: the
    committed step's duration) and `steps_since_last_merge_test` (committed step count since then; first call 1);
    on a GPU grid one `ReadBuffer` of the `size_change` row into CPU memory (1 B per cell place, counted in
    `bytes_copied` and `copy_count`); then `BuildResizePlan`.
  - Step 3 "nothing to do": the plan is empty, ~~no particle wants a split or merge~~ (A67) the particle-only pass
    finds `mismatched_particle_count == 0`, and no wait array changes (the
    three "after" rows equal the committed arrays, bytes); a change of the wait arrays alone, or (A67) of mismatched
    particles alone, is committed as a resize with the same cells and `version + 1` (design 5.1 step 3 and 5.3
    together). The time and step count of
    the last committed merge test are kept in `CellResize` and move only at a commit.
  - Step 5 remaps: `pressure` (`kPerUnitOfWeight`, weight = `has_pressure` as 0 or 1 in the merge-weight row),
    `has_pressure` (merge: 1 if any; private rule beside the `kLabel` code), `pocket_of_cell` (`kLabel`),
    `velocity_after_pressure` and `gas_velocity_after_pressure` (`ApplyResizeToFaceVelocity`), from the "after" rows:
    `merge_ready_duration` and `merge_ready_step_count` (0 for split and merged cells, else the value after the
    pass), `held_velocity_gradient_size` (`kPerUnitOfWeight` with null weight on a split, so copied; 0 for a merged
    cell; `plan-decisions.md` item 2 "Remap"); the gas arrays through `gas::ToArraysToRemap(*resize_try.gas_try())` per design 5.4 (densities and
    flame weighted by old `gas_volume`, `gas_volume` by `ApplyResizeToCellValues(kAmount)` with null weight, both
    pressures copied on split and `gas_volume`-weighted on merge). Each air pocket's `first_cell_index` becomes the
    smallest new cell place carrying its label.
  - (Amended, Foundation as built) Any failure after `BuildResizeTry` calls `RejectResize`; for a failed check the
    committed state keeps its bytes (the tests below), and after `kRunOnCallFailed` the try's arrays are unspecified
    and the step reports the failure so the run stops (`AGENTS.md` "Structure"); no test demands bytes after a GPU
    call failure. Step 8 sums with `ComputeLiquidTotals(..., cell_resize.sum_buffers)`.
  - Step 8 tolerance: `|after - before| <= kAmountTolerance * max(|before|, |after|)`, for mass, rest volume,
    current volume, each ingredient volume over liquid particles (A64: no records) and the gas amount `sum
    gas_volume * gas_density`. `ResizeCellsReport::totals_before` and `totals_after` (plan decision, replacing the
    ledger) hold the liquid totals.
  - `ResizeResult{resized, module_error = static_cast<std::uint8_t>(failure.error)}`.
  - **A17, grid on the GPU** (`grid_run_on == kGpu`, one piece or batched): step 1 makes one `grid::ReadGrid` copy of
    the committed grid into the CPU `GridBuffers` of Task 10; the closure search (Task 9), the plan (Task 14) and
    `BuildResizeMap` read it. Step 4 builds the resized grid on the CPU from the new cell list (for the map) and
    uploads the same list into the `Buffer<grid::CellId>` of Task 10 with one slice `WriteBuffer`, then calls
    `step::BuildResizedGrid(resize_try, ResizedCells{{}, &upload, cell_count, &cpu_resized_grid})` (step plan Task
    32, amended by the reconcile pass), which builds the try's grid on the GPU into `free_grid_storage()` (A17: equal
    tables) and checks version and block count against the CPU grid; `BuildResizeMap` checks them too
    (`kWrongGridVersion`). On a CPU grid step 4 builds into `free_grid_storage()` and calls
    `step::ApplyResizedGrid` as before. With `GridCopies::kHeldByStep` (batched GPU run, Task 43) step 1 reads
    `step::FindCpuGridCopy` instead of `ReadGrid` and step 4 builds the CPU resized grid into
    `resize_try.free_cpu_grid_copy_buffers()`, then hands it over with `step::ApplyResizedCpuGridCopy` after the
    map is built and before `CommitResize`.
    `ResizeCellsReport` gains (plan decision) `std::uint64_t bytes_copied = 0;` (both directions, this call) and
    (A17, batched-run buffers) `std::uint32_t copy_count = 0;` (one per `ReadGrid`, `ReadBuffer` or `WriteBuffer`
    call between CPU memory and the grid's processor, this call; amended 2026-11-15, A41: `ChooseCellSize` is gone,
    and the `size_change` read back is one of them).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ResizeCells, DeepTank_SplitsNearSurfaceMergesBelow)   // BuildDeepTankScene, one step, then ResizeCells: cells
      // within 0.2 m of the surface at 0.025 m, deeper bigger; CheckSizeStepRule; state_version + 1, time unchanged
  TEST(ResizeCells, DeepTank_ParticleTotalsKept)              // A64 (was RecordsRemapped_TotalsKept): BuildDeepTankScene,
      // 3 steps each with a resize: totals of the step-8 list over particles within kAmountTolerance per resize
  TEST(ResizeCells, ReferenceSpeedFromCommittedLimits)        // A62: largest_liquid_speed 0.5 m/s, biggest width
      // 0.1 m: the speed passed is max(0.5, sqrt(9.81 * 0.1)) = 0.990454 m/s (exact max); with 2 m/s: 2 m/s
  TEST(ResizeCells, WaitArraysRemapped)                       // a split cell's children: duration and steps 0, held
      // copied; a merged cell: all three 0
  TEST(ResizeCells, CalledTwiceAtSameTime_SecondDoesNothing)  // design 7 "Nothing to do"; Review Focus 2
  TEST(ResizeCells, MergeWait_NeedsUnbrokenTimeAndFailedResizeEarnsNone)  // design 7 "Merge wait": merge test held
      // 0.2 s, broken one step, held again: no merge before 0.25 s unbroken; one resize made to fail (limits lowered
      // so a split elsewhere is kTooManyCellBlocks, raised again with ApplyLimits and the CellResize rebuilt into
      // the new block with its last-merge-test time and step count carried) adds no wait
  TEST(ResizeCells, AirPocketFirstCellTranslated)
  TEST(ResizeCells, GasArraysRemapped_GasAmountKept)          // BuildSmallFireScene with has_cell_resize
  TEST(ResizeCells, TooManyCellBlocks_ChangesNothing)          // ExpectSameBytes on snapshots before and after
  TEST(ResizeCells, GridCopyPath_SameAsCpuGrid)                // A17: DeepTank with grid_run_on kGpu forced onto CPU
      // buffers (the copy path runs, CPU to CPU): same report and snapshot bytes as grid_run_on kCpu;
      // bytes_copied == 456 * committed blocks + 64 * committed blocks (size_change, A41) + 8 * new cells;
      // copy_count == 3
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Resize cells after a committed step as one transaction`.

### Task 25: Region transfer, rules 1 to 3 (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 26: Region transfer, rules 4 and 5, the ledger, every failure (moved to the later group, A64)

**Amended 2026-11-15 (A64):** moved to "Deep liquid without particles (later; needs review)" at the end of
this plan, with its number; not part of the first version (every liquid cell holds particles).

### Task 27: Particle count changes inside the resize, and conflict C2

**Amended 2026-11-15 (A64):** no grid-only marks: both `NeighbourhoodInputs` get `is_grid_only_region = nullptr`
(no grid-only cell). States how the particles of deep cells follow the resizes (below); the deep-tank cascade test
is in Task 30a. The step-7 sub-case of `FailureAtEachStep_ChangesNothing` is in Task 27a.

**Amended 2026-11-15 (A67):** `ResizeCells` calls the extended `AdjustParticleCount` at every pass (step 7, and in
step 3's particle-only pass with the committed grid as both grids); the "drifted particles keep their level"
statement and its test are replaced; the report fields are wired into `ResizeCellsReport` and `step::ResizeResult`.

**Kind:** integration

**Files:**
- Modify: `resize_cells_cpu.cc`, `resize_cells_test.cc`

**Interfaces:**
- Produces design 5.1 step 7: `particles::AdjustParticleCount(particles_try, committed grid, its
  NeighbourhoodInputs, resized grid, its NeighbourhoodInputs, every block)`; committed inputs from
  `last_solids_at_end()` (`outside_solid`, `side`); resized inputs from `solids_at_faces_to_fill()` after
  `solids::ComputeSolidsAtFaces(resized grid, all blocks, solids, ...)`; `is_grid_only_region` null in both (A64);
  raw pointers from each array's `buffer().data()`. `GroupParticlesByBlock` afterwards when anything was split or
  merged. `particle_count` of the report is the result.
- **Deep cells (A64; particles design 5.6, plan decision on the order):** a cell of level `l` holds particles of
  `size_level == l`. When the plan merges eight cells of level `l + 1` into one of level `l`, `AdjustParticleCount`
  merges each group of eight level-`l + 1` particles that its rule pairs inside the new cell into one level-`l`
  particle; when it splits a cell, each level-`l` particle inside becomes eight of level `l + 1`. Because the pass
  allows one level per cell per resize and the merge wait restarts at 0 for a merged cell, a cell `k` levels too
  fine reaches its wanted level after `k` merges, at least `merge_wait_duration` and `min_merge_wait_steps` apart,
  and its particles follow level by level: a big cell holds about the seeded count (8) of particles with `8^k`
  times the volume. ~~Particles that drifted across a size change keep their level (C2), are counted in
  `skipped_incomplete_count`, and may make a cell hold more than 8. No new particle code.~~ **Replaced** (A67):
  particles that moved across a size change are split or merged by the extended `AdjustParticleCount` at the next
  pass (design 5.4 "Particles whose size no longer matches their cell"); the particle code is the particles plan's
  A67 task, this task only calls it at every pass. One wording (A65): every liquid cell holds about eight particles of
  its own size: a particle's `size_level` is its cell's level, its rest volume is one eighth of its cell (`(w/2)³`),
  so a cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest particle's volume.
- **Every pass (A67, plan decisions):** step 7 runs over all blocks at every pass; step 3's `ApplyParticleOnlyPass`
  (Task 24's seam) calls `AdjustParticleCount(particles_try, committed grid, committed inputs, committed grid,
  committed inputs, every block)` (the same grid twice: read-only aliasing the particles design must allow) and
  returns its result; `mismatched_particle_count == 0` is "nothing to do"; otherwise the same-cell resize goes on
  and step 7 keeps that result (not run twice). `ResizeCellsReport::particle_count` is the pass's result with its new
  fields (`mismatched_particle_count`, `waiting_particle_count`, `lone_particle_count`, `incomplete_split_count`,
  `kinetic_energy_change`); `ResizeCells` returns `step::ResizeResult{resized, module_error,
  size_change_kinetic_energy = particle_count.kinetic_energy_change, mismatched_particle_count =
  particle_count.mismatched_particle_count}` (both 0 on a failure or nothing to do; the step plan's A67 task adds
  the two `ResizeResult` members and copies them into `StepReport`).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ResizeCells, FloatingCube_SplitsDeepInsideParticles)   // design 7 "C2" (amended A67): BuildFloatingCubeScene,
      // first resize splits all 125 cells; every particle is split with the conservation of particles 5.4 (their
      // tolerances); ~~the others keep their size and are counted in skipped_incomplete_count~~ those less than 1.5
      // old widths inside on either grid are counted in incomplete_split_count (count from positions, test side)
  TEST(ResizeCells, DriftAcrossSizeChange_MatchedOrCounted)    // A67, replaces DriftAcrossSizeChange_NeverSplit
      // (design 7 "Drift", replaced): BuildMixedSizeColumnScene, one smallest-level particle given -0.5 m/s (down)
      // across the size change into a level-1 cell, 20 steps with a resize each: at the first pass after the crossing
      // mismatched_particle_count >= 1; by 16 passes after it the particle is merged or counted in
      // lone_particle_count; particle mass within 1e-15 relative and momentum exact to fixed-order rounding per resize
  TEST(ResizeCells, OnlyMismatchedParticles_PassStillRuns)     // A67 (design 7 "Particles-only pass"):
      // BuildMixedSizeColumnScene settled (plan empty, wait arrays unchanged at two calls in a row), then one
      // level-1 particle moved by snapshot edit into a smallest-level cell: the next ResizeCells commits a same-cell
      // resize, state_version + 1, resized true, that particle replaced by 8 of the smallest level; with no
      // mismatched particle: resized false, same state_version
  TEST(ResizeCells, ReportFieldsReachTheStepReport)            // A67 (design 7 "Energy of the pass reported"): the
      // run above through step::AdvanceTo: ResizeCellsReport::particle_count.kinetic_energy_change and
      // mismatched_particle_count equal ResizeResult's and StepReport's size_change_kinetic_energy and
      // mismatched_particle_count, bytes; a split-only pass: |kinetic_energy_change| <= 1e-12 x the particle's energy
  TEST(ResizeCells, RefusedSplitNextToSnowOrJelly_GridUnchangedThere)  // design 7 "Refused split":
      // BuildSnowAndJellyScene with has_cell_resize: a split whose closure reaches a snow cell, then a jelly cell:
      // refused_split_count 1 each, those cells unchanged
  TEST(ResizeCells, MergedCellHoldsParticlesOfItsLevel)        // A64: BuildDeepTankScene with merge_wait 0 and
      // min_merge_wait_steps 0: after the first resize that merges, every merged cell's particles have the cell's
      // size_level except those counted in waiting_particle_count and lone_particle_count (amended A67; was
      // skipped_incomplete_count); particle mass and momentum before and after within kAmountTolerance relative
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Split and merge particles inside the resize with both grids` (amended A67: `Bring every particle to its cell's
  size at each resize`).

### Task 27a: Every failure of the resize changes nothing

**Amended 2026-11-15 (A64):** new task; `ResizeCells, FailureAtEachStep_ChangesNothing` moved here from Task 26
(now in the later group), its sub-cases rewritten without records.

**Kind:** integration

**Files:**
- Modify: `resize_cells_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ResizeCells, FailureAtEachStep_ChangesNothing)    // design 7 "Failed resize"; Review Focus 1. Sub-cases:
      // step 4, block limit below the resized grid (kTooManyCellBlocks); step 5, a NaN in one committed pressure
      // value of a snapshot rebuilt with BuildSimulationFromSnapshot (kNotFinite from the remap); step 7, particle
      // limit hit by AdjustParticleCount's split (kTooManyParticles); step 8, two particles of mass 1e308 whose
      // totals overflow (kNotFinite). Each: snapshot bytes equal those taken after the step's commit;
      // state_version unchanged; free storage given back (the next TryStep builds version + 1 into it) and commits
  ```
- [ ] **Step 2: Run** (fail, then fix the owning commit if a sub-case leaks). **Step 3:** gate. **Commit.** Subject:
  `Show a failed resize changes nothing`.

### Task 28: The whole resize on the GPU

**Amended 2026-11-15 (A64):** the region transfer's GPU half moved to the later group (Task 28b); this task keeps the
GPU tests of the whole resize.

**Kind:** device

**Files:**
- Create: `native/fluidity/cell_resize/resize_cells_gpu_test.cc`
- Modify: `CMakeLists.txt` (`LABEL gpu`, `LINKS fluidity_cell_resize_test_support fluidity_base_test_support`)

- [ ] **Step 1: Tests.** `LABEL gpu`: `ResizeCells, OnGpu_EqualsCpu` (Tasks 24, 27 cases: plans, `size_change`, ids,
  counts equal; values within design 6); `ResizeCells, OnGpuTwice_GivesIdenticalBytes`; every test passes its main
  outputs to `WriteResultDigest`. `LABEL no_gpu`: `ResizeCells, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** nothing new: the GPU halves of Tasks 12, 13a, 17 and 23 run; returned to the CPU: the
  `size_change` bytes, one failure record. **Step 3:** gate. **Commit.** Subject: `Run the whole resize on the GPU`.
  Body: `CUDA: compiled, not run`.

### Task 29: The step callback

**Amended 2026-11-15 (A41, A64; item 18 superseded by A41):** `ChooseCellSize` is removed (stage 2 calls
`grid::ComputeNewCellSizeLevel`, grid plan; its tests `ChooseCellSize, OneLevelSmallerThanNearest`,
`CappedAtSmallestLevel`, `NullCommittedTables_UsesTheHeldCopy`, `BatchedGpuRun_ProbesTheStepsCpuCopy` are withdrawn;
the grid plan's `ComputeNewCellSizeLevel, NextToBigCell_OneLevelFiner` replaces the first two). The reconcile
small fix a (null tables in a one-piece GPU run, the held copy) and the A17 batched-run-buffers note on this task are
superseded: no callback reads the committed tables any more. `ApplyGridOnlyRegionInStep` and
`MoveGridOnlyLiquidInStep` moved to the later group (Task 29b). `GridCopies` stays (Task 10), for the resize map.

**Amended 2026-11-14 (Foundation as built):** the test support lives in `fluidity_cell_resize_test_support` (made with `fluidity_add_test_library`)
and the tests name it in `LINKS`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/cell_resize/step_callbacks.cc`, `step_callbacks_test.cc`
- Modify: `resize_cells.h`, `cell_resize.cc`, `cell_resize_test_support.h/.cc`, `CMakeLists.txt` (`step_callbacks_test`
  `LABEL cpu LINKS fluidity_cell_resize_test_support fluidity_base_test_support`)

**Interfaces:**
- Consumes: Task 24 `ResizeCells` (the one callback, `step::StepCallbacks::resize_cells`).
- Test support (plan decision; amended 2026-11-15): `step::StepCallbacks BuildCellResizeCallbacks(CellResize&
  cell_resize)` (the `resize_cells` pair, as I/O will set it), and `struct CellResizeTestRun {
  std::unique_ptr<CellResize> cell_resize; step::Simulation simulation; };` `std::expected<CellResizeTestRun,
  step::StepFailure> BuildCellResizeTestRun(const step::TestScene& scene, const CellResizeSettings& settings)` (the
  `CellResize` built into the simulation's `RunMemory` blocks).

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `AdvanceTo, CellResizeCallbacks_StepThenResize` (`kResizeCells`
  recorded); `AdvanceTo, NewCellsGetTheSharedRule` (A41: `BuildMixedSizeColumnScene` with a source adding particles
  above the surface: every cell created by stage 2 has `grid::ComputeNewCellSizeLevel`'s level, the CPU test side
  computing it on the committed tables; no copy counted for it).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Connect the cell resize to the step's callback`.

### Task 30: Scene tests, rest across a resize

**Amended 2026-11-15 (A64):** `CellResizeScene, RegionCycle_TotalsKeptEveryResize` and `BorderRefill_EveryBorderEighthHoldsAParticle`
moved to the later group (Task 30c).

**Kind:** integration

**Files:**
- Create: `native/fluidity/cell_resize/cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`, runs from `BuildCellResizeTestRun`, steps through `step::AdvanceTo`):

  ```cpp
  TEST(CellResizeScene, PressureAtRestAcrossAResize)   // design 7: BuildMixedSizeColumnScene, merge_wait 0,
      // min_merge_wait_steps 0; resize with surface reach 4, then a CellResize with reach 5 (one more layer splits),
      // then reach 4 (it merges), one step after each: largest |velocity_after_pressure| <= max_volume_error_rate *
      // 0.025 (pressure's bound)
  ```
- [ ] **Step 2: Run.** If it fails, fix the owning commit (rebase), not the test. **Step 3:** gate. **Commit.**
  Subject: `Test pressure at rest across a resize`.

### Task 30a: Scene tests, deep liquid coarsens with particles and stays put

**Amended 2026-11-15 (A64, A41, A63):** new task; `plan-decisions.md` item 1 "New tests" and item 2 "Tests" (1),
(2) and the owner's calm-tank test.

**Amended 2026-11-15 (A65, A66, A67):** the A65 wording in the deep-tank test; the three tests below start
all-smallest on purpose (they test the coarsening cascade and the calm pool's coarsening time from that start; a run
starts on `step::BuildStartCells`' grid, tested in Task 30d), and each test's comment says so; the drifted-particle
exception becomes the waiting and lone counts (A67).

**Kind:** integration

**Files:**
- Modify: `cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`; scenes of Task 1a with `smallest_size_level = 4`, defaults otherwise; a resize
  after every committed step; "the depth term's level" of a cell = `grid::ComputeWantedSizeLevel(widths, w_band)` at
  its depth, computed test side):

  ```cpp
  TEST(ResizeCells, CalmDeepTank_CoarsensWithParticlesInEveryCell)  // A64; Review Focus 4: BuildCalmDeepTankScene
      // (2 m deep, w_min 0.05 m, 5 levels; all-smallest start on purpose, A66), at rest, 3 s: every cell deeper than
      // 8 w_min = 0.4 m holds the depth term's level (bottom cells 0.2 m: w_band = 2.0 / 8 = 0.25 m); every liquid
      // cell holds about eight particles of its own size (A65): the count per cell is 1..16 and after every resize
      // every particle's size_level equals its cell's level except those counted in waiting_particle_count and
      // lone_particle_count (A67; was "except drifted particles counted in skipped_incomplete_count");
      // particle mass and momentum before and after each resize within kAmountTolerance (momentum relative to
      // sum m |v| + 1e-12 kg m/s)
  TEST(ResizeCells, CalmPool_ReachesBiggestCellsThenStays)  // literature-cell-size.md 5(f)2 (all-smallest start on
      // purpose: the coarsening time is measured from it, A66): BuildCalmPoolScene (1 m,
      // w_min 0.02 m, 5 levels): every cell deeper than 8 w_min = 0.16 m reaches the depth term's level within
      // 0.25 s x 4 levels + 32 steps of committed time; then for 10 s no split and no merge (wanted_split_count 0,
      // plan empty at every resize)
  TEST(ResizeCells, CalmTank_InsideCoarsensSurfaceStaysFinest)  // owner (A61): BuildCalmDeepTankScene, 3 s: at every
      // resize every cell within 8 w_min of the surface is at w_min, and after 3 s some cell below 0.4 m is coarser
  TEST(ResizeCells, SplitThenMerge_ConservesMassAndMomentum)  // 5(f)1: BuildMixedSizeColumnScene, reach 5 then reach
      // 4 (one layer split then merged): particle mass, rest volume, current volume and momentum before the split
      // and after the merge within kAmountTolerance relative
  ```
- [ ] **Step 2: Run.** A fail is a defect in an owning commit, except a calibration finding (the 10 s flicker check)
  which is reported to the owner, not fixed by changing a default. **Step 3:** gate. **Commit.** Subject:
  `Test deep liquid coarsening with particles in every cell`.

### Task 30b: Scene tests, thin sheets, splashes and shear layers

**Amended 2026-11-15 (A41, A62, A63):** new task; `plan-decisions.md` item 2 "Tests" (3), (7) and the owner's jet
test.

**Kind:** integration

**Files:**
- Modify: `cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`; Task 1a scenes, `smallest_size_level = 4`, a resize after every step):

  ```cpp
  TEST(ResizeCells, ThinSheetAndPinchOff_StaySmallest)  // 5(f)3: BuildThinSheetScene 0.5 s and BuildLiquidBridgeScene
      // until the bridge breaks (at most 1 s): at every resize no cell within 8 w_min = 0.16 m of either face of
      // the sheet or the neck is wider than w_min, and no merge happens there
  TEST(ResizeCells, SplashIntoCoarseRegion_NoStuckParticles)  // 5(f)7: BuildCalmDeepTankScene run 3 s (deep cells
      // coarse), snapshot, a water ball of radius 0.1 m at the smallest level added 0.3 m above the surface falling
      // at 2 m/s (test-side snapshot edit, BuildSimulationFromSnapshot), 1 s: no particle whose speed stays below
      // 1e-3 x 2 m/s for 20 steps while above the surface; particle momentum before and after each resize within
      // 1e-12 relative
  TEST(ResizeCells, Jet_KeepsSmallCellsAlongShearLayer)  // owner (A62): BuildJetScene, 1.5 s: cells within 2 jet
      // radii (0.1 m) of the jet edge (radius 0.05 m about the source axis, x from 0.1 to 0.6 m) stay at w_min at
      // every resize after 0.5 s
  ```
- [ ] **Step 2: Run.** A fail of the jet test with `shear_width_factor = 0.25` is a calibration finding: report the
  smallest factor that passes, do not change the default (owner question 3 of `plan-decisions.md`). **Step 3:**
  gate. **Commit.** Subject: `Test sheets, splashes and jets against the size rule`.

### Task 30d: Scene tests, particles that cross a size change, and the start on the start cells

**Amended 2026-11-15 (A66, A67):** new task; `plan-decisions.md` "Owner answers A65–A68 applied" A66 and A67 tests;
design 7 rows "Start on the start cells", "Rising particle split on the way", "Sinking particles merged".
**Amended 2026-11-15 (A66):** also `BuildStartCells, DeepTank_SameParticlesAsCoarsenedRun`, handed over by the step
plan because it runs `ResizeCells`; the all-smallest scenes use the step's `ApplyStartCellsForTest(scene,
TestStartCells::kAllSmallest)`. Needs the
particles plan's A67 task (the extended `AdjustParticleCount`) and the step plan's A66 task (`BuildStartCells`).

**Kind:** integration

**Files:**
- Modify: `cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`; `smallest_size_level = 4`, defaults otherwise; a resize after every committed
  step; moves "by snapshot edit" use `BuildSimulationFromSnapshot` on a test-side edited snapshot, as Task 30b's
  splash test does):

  ```cpp
  TEST(ResizeCells, CalmDeepTankOnStartCells_NeverSplitsForDepth)  // A66: BuildCalmDeepTankSceneOnStartCells, at
      // rest, 3 s: wanted_split_count 0 at every resize (the start is never coarser than the rule: Euclidean depth
      // minus half a diagonal <= the through-liquid depth); mismatched_particle_count 0 at every pass; particle count
      // never above the start count; after 3 s every cell deeper than 8 w_min = 0.4 m at the depth term's level
  // Amended 2026-11-15 (A66): handed over by the step plan (the step cannot run ResizeCells).
  TEST(BuildStartCells, DeepTank_SameParticlesAsCoarsenedRun)  // BuildCalmDeepTankSceneOnStartCells (2 m deep, 8 per
      // start cell at its level) against BuildCalmDeepTankScene started all-smallest (the step's test-only
      // ApplyStartCellsForTest(scene, TestStartCells::kAllSmallest)), at rest, a resize after every step, run until a
      // resize changes nothing (plan empty, wait arrays unchanged, mismatched_particle_count 0; at most 10 s):
      // coarsened count <= start count <= 1.25 x coarsened count (the start uses the Euclidean depth minus half a
      // cell diagonal, so it is never coarser and at most about one level finer in places)
  TEST(ResizeCells, ParticleRisingFromDeepCell_ArrivesSplitToSmallest)  // A67; Review Focus 7: BuildCalmDeepTankScene
      // run 3 s (all-smallest start on purpose, so the deep cells are coarse), then one particle of a level-2 cell
      // (0.2 m, depth >= 1.6 m) moved up by 0.5 w_min = 0.025 m before each step (snapshot edit, velocity kept)
      // until it is within w_min of the surface: after every resize every liquid particle in a smallest-level cell
      // has size_level 4 (so in every committed step where the particle or a child of it is in a smallest-level cell
      // it has the smallest size: the split repeats within one pass in such a cell); total particle mass within
      // 1e-15 relative and momentum within 1e-15 relative of sum m |v| per resize
  TEST(ResizeCells, SmallParticlesInCoarseCell_MergedByThePass)  // A67: the same coarsened tank, eight level-4
      // particles moved by snapshot edit into one level-3 cell (0.1 m, depth >= 0.8 m): the next pass merges them
      // into one level-3 particle (mismatched_particle_count 8, kinetic_energy_change <= 0); mass exact to the
      // fixed-order sum, momentum within 1e-15 relative
  TEST(ResizeCells, ThreeSmallParticles_MergedAfterSixteenPasses)  // A67: as above with three particles: unmerged
      // and waiting_particle_count 3 after passes 1..15, merged into one level-3 particle at pass 16
  ```
- [ ] **Step 2: Run.** A fail is a defect in an owning commit (this module's call, or the particles or step task it
  names). **Step 3:** gate. **Commit.** Subject: `Test particles that cross a size change and the adaptive start`.

### Task 31: Scene tests, material sets and gas

**Amended 2026-11-15 (A64):** `CellResizeScene, Absorb_ParticleTakenIntoRecord` and
`ForeignParticle_CellReturnsToParticles` moved to the later group (Task 31b); the other two rewritten on the
all-particle scene.

**Kind:** integration

**Files:**
- Modify: `cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`):

  ```cpp
  TEST(CellResizeScene, TwoLiquidsMaterialSets)           // design 7 "Two liquids, material sets" (A64 form):
      // BuildTwoLiquidsDeepScene, 50 steps with resizes: the snow cells keep their size (kKeepsItsSize); every
      // liquid cell holds particles of its level; each material's particle totals within kAmountTolerance per
      // resize; MaterialSetKey checked each step
  TEST(CellResizeScene, SmokeOverMovingWater)             // design 7 "Gas beside liquid" (A64 form):
      // BuildSmokeOverDeepWaterScene, resize every step, 40 steps: gas amount within kAmountTolerance per resize;
      // every cell holding or touching gas at w_min; no marker taken out by a resize; ComputeLiquidTotals with the
      // liquid set leaves markers out
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Test resizes beside other materials and gas`.

### Task 32: Scene test, the cut beside resized water

**Amended 2026-11-15 (A64):** `CellResizeScene, StandingWave_GridOnlyRegionMatchesParticles` moved to the later
group (Task 32b, the base of the gating test of Task 32c).

**Kind:** integration

**Files:**
- Modify: `cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`):

  ```cpp
  TEST(Cut, PlateThroughBlock_SidesNeverMixWithResize)  // handed by the step plan: the split-and-merge part of
      // Cut, PlateThroughBlock_SidesNeverMix: BuildSnowAndJellyScene's plate cut, cell resize on with the water beside
      // the jelly; the run restarted from a snapshot after the first hit, and with a rejected try before it: bytes
      // equal to the uninterrupted run; no jelly particle split, merged or changes side; driven_work == elastic
      // energy gained + cut_work + kinetic energy within 5 %
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Cut beside resized water`.

### Task 33: Part A on the GPU, scene runs

**Amended 2026-11-15 (A64):** the rows are those of Tasks 30, 30a, 30b, 31 and 32.

**Amended 2026-11-15 (A66, A67):** and those of Task 30d.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the scene runs pass their snapshots to `WriteResultDigest` so the hardware repeat run covers them;
CPU-against-GPU values within design 6.

**Files:**
- Create: `native/fluidity/cell_resize/cell_resize_scene_gpu_test.cc`

- [ ] **Step 1: Tests.** `LABEL gpu`: every row of Tasks 30, 30a, 30b, 30d, 31 and 32 with `RunOn::kGpu` (design 7 "Every
  row on `RunOn::kGpu`"), against the CPU within design 6 for one resize (the `size_change` bytes equal) and for the
  stated scene tolerances over the run; `CellResizeScene, OnGpuTwice_GivesIdenticalBytes` (deep tank, 20 steps); each
  test writes the digest of its final snapshot with `WriteResultDigest`. `LABEL no_gpu`: `BuildCellResizeTestRun`
  with `kGpu` gives `kRunOnCallFailed`. CMake: `LINKS fluidity_cell_resize_test_support fluidity_base_test_support`.
- [ ] **Step 2:** gate. **Commit.** Subject: `List the cell resize's GPU scene runs`. Body: `CUDA: compiled, not run`.

---

# Part B: the batched run

### Task 34: Method paper, batches

**Amended 2026-11-14 (A15):** the solves are no longer "not batched": the text gives the three ways to run and the
planner's choice; pressure's `<sec-solve-in-batches>` holds the solve itself.

**Amended 2026-11-15 (A68):** Step 1's text also states that this version cuts slab batches, the plan-area limit
`a_max = R x 16 w_min² / (n b)` with the worked numbers of design 5.9 item 4 (about 3.2 m² at 1 cm, 0.8 m² at 0.5 cm),
and that batches cut by columns or tiles come later (design 5.11); glossary rows `slab batch`, `plan-area limit`.

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/cell_resize.typ`, `docs/glossary.md`; Create: `figures/batch_passes.typ`

**Interfaces:**
- Produces labels `<sec-batches>`, `<fig-batch-passes>`, `<proc-apply-stages-in-batches>`.

- [ ] **Step 1: Text,** outline item 9 (design 5.7 to 5.10, with `plan-decisions.md` "Batched solves (A15)"): what
  stays on the GPU in each mode, the two passes, the stacked border, the three ways to run (one piece; batches with
  whole-grid solves; batches with solves in pieces) and how the planner chooses and reports the memory budgets and
  copies, why the CPU fallback is gone (A17), the order of sums and agreement within the solver's tolerance, and the
  plain list of which scenes each mode is for (syrup in pieces: possible, slow). Cite [5] Wang 2020 and [15] Fei,
  Huang, Gao 2021. Glossary rows: `whole-grid arrays`, `loaded
  batch`, `stacked border`, `pass`, `block partial sum`.
- [ ] **Step 2: Figure** `<fig-batch-passes>`: a time line of one try: CPU stages, pass 1 over three batches with
  their borders, the solve (whole, or its sweeps over solve batches, drawn as a second lane), pass 2, the CPU
  stages after it.
- [ ] **Step 3: Pseudocode** `<proc-apply-stages-in-batches>` numbered 1 to 4 as Task 40 implements.
- [ ] **Step 4:** typst, look, gate. **Commit.** Subject: `Describe the batched run`.

### Task 35: Step: the whole-grid part of a stage

**Amended 2026-11-15 (A64):** `kMoveParticles` has no whole-grid part in the first version (the grid-only move is
the later group's): the parts are those of stages 3, 9 and 19; the test is `StageTable,
WholeGridPartsOnlyForStages3_9_19`.

**Amended 2026-11-14 (A17, batched-run buffers):** stage 3 also gets a whole-grid part, `FindGridEdgesOpenToSky`,
whose flood needs every block (step plan Appendix A row M11).

**Amended 2026-11-14 (reconcile):** ~~the two five-material step tests it re-runs are marked blocked by the open
decision on a springy solid beside a viscous liquid.~~ Option C chosen (A24): test unblocked. The two
five-material step tests run as written (the joint viscous solve carries the material faces, `plan-decisions.md`,
"Viscous liquids with blocking materials and sealed gas (A24)").

**Amended 2026-11-14 (Foundation as built):** `step_test_support.h/.cc` belong to the step's `fluidity_step_test_support` library (made with
`fluidity_add_test_library`); the new tests link it and `fluidity_base_test_support` through `LINKS`.

**Kind:** integration

This changes files of the step module (design section 4, step item 5) and extends its tests.

**Files:**
- Modify: `native/fluidity/step/step_stages.h`, `stage_table.cc`, `try_step.cc`, `solids_stages.cc` (stage 3, A17),
  `pressure_stages.cc` (stage 9),
  `move_particles_cpu.cc` (stage 16), `check_step_cpu.cc` (stage 19), `step_test_support.h/.cc`
  (`BuildRejectingStageTable` covers both parts), `batched_seam_test.cc`

**Interfaces:**
- **Plan decision (new step field):** design 5.8 runs stage 9 partly per batch and partly on the whole grid, and the
  ends of stages 16 and 19 on the whole grid, while a `StageTable` entry is one function. So:

  ```cpp
  struct StageTable {
    std::array<StageFunction, kStageCount> run{};
    // The part of a stage that needs every block of the try grid at once, called with blocks = every block.
    // kFindSolidSurfaces (A17): FindGridEdgesOpenToSky (its flood crosses every block).
    // kFindLiquidSurface: FindMaterialBoundaries, FindAirPockets, CheckViscosityCanRun. kCheckStep: CheckSolidExchange, the gas changes, the next
    // CommittedStepLimits. Null for every other stage. A one-piece run calls it right after `run`.
    std::array<StageFunction, kStageCount> run_whole_grid{};
  };
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `StageTable, WholeGridPartsOnlyForStages3_9_19` (A17, A64: was
  `..._9_16_19`); `FindSolidSurfaces, OpenToSkyOnWholeGridEqualsOnePiece` (A17: a lid over two ranges cut through
  the pocket under it: the whole part's marks equal the one-piece stage's bytes, and a range-only call would differ);
  `TryStepWithStages, WholeGridPartRunsAfterRangePart` (`CallbackLog` order); `TryStepWithStages,
  RejectAfterWholeGridPart_ChangesNothing` (stages 9, 19). The step's existing `TryStepWithStages,
  RangeCut_SameBytes`, `ReferenceScene, RepeatRun_IdenticalBytes` and `BuildSimulationFromSnapshot,
  ContinuedFiveMaterialRun_SameBytes` must pass unchanged.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** by moving the named calls, not by changing them. **Step 4:**
  gate. **Commit.** Subject: `Split the stages that need the whole grid into two parts`.

### Task 36: Step: access to the loaded try and copies by block

**Amended 2026-11-14 (A17, batched-run buffers):** `ApplyGridPiece` takes the CPU map of the piece, which stage 18
hands to foam and spray (a GPU piece has no CPU map of its own; step plan Appendix A row M13).

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/step_stages.h`, `step_arrays.cc`, `simulation.cc`, `batched_seam_test.cc`

**Interfaces:**
- **Plan decision (new step names):** the step builds the loaded `StepTry` and its `GridBuffers` (step design 5.9),
  so batches reaches them through the step:

  ```cpp
  StepTry* FindLoadedTry(Simulation& simulation);                  // Null in a one-piece run.
  grid::GridBuffers* FindLoadedGridBuffers(Simulation& simulation);  // Null in a one-piece run.
  // Points loaded_try.grid() at piece.grid() and binds every loaded array to it with grid::ApplyGridVersion.
  // kTooManyCellBlocks when the piece has more blocks than the loaded arrays hold. A17: block_in_whole_grid is the
  // piece map in CPU memory (piece.block_in_whole_grid() for a CPU piece, the batched run's copy for a GPU piece),
  // one entry per piece block, else kBadInput; stage 18 passes it as FoamAndSprayInputs::block_in_whole_grid.
  std::expected<void, StepFailure> ApplyGridPiece(StepTry& loaded_try, const grid::GridPiece& piece,
                                                  std::span<const grid::CellBlockIndex> block_in_whole_grid);
  struct BlockCopy { grid::CellBlockIndex from_block; grid::CellBlockIndex to_block; std::uint32_t block_count; };
  // Copies array instance `array_index` (an index into ReadStepArrays) block by block between two tries, on
  // whichever processors they live (base/ slice copies). An array counted per particle follows the particle groups
  // of the blocks in each try. kBadInput when a block lies outside either try's grid.
  std::expected<void, StepFailure> WriteStepArrayBlocks(const StepTry& from, std::size_t array_index,
                                                        std::span<const BlockCopy> copies, StepTry& to);
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `FindLoadedTry, OnePieceRun_IsNull`; `FindLoadedTry,
  BatchedLayout_HasRoomForMaxLoadedBlocks`; `ApplyGridPiece, BindsEveryLoadedArray` (every array's `block_count()`
  equals the piece's); `ApplyGridPiece, MapOfWrongLength_IsBadInput` (A17); `MakeFoamAndSpray,
  LoadedTryPassesThePieceMap` (A17: a hand-written callback sees the given span); `WriteStepArrayBlocks,
  ThereAndBackByBlocks_SameBytes`; `WriteStepArrayBlocks, PerParticleArray_FollowsGroups`; `WriteStepArrayBlocks, BlockOutsideTheGrid_IsBadInput`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Let a batched run reach the loaded try and copy arrays by block`.

### Task 37: Batches errors, stage places and the sum over blocks

**Amended 2026-11-15 (reconcile item 31; A64):** `BatchesError::kOutOfMemory` is the last value (after
`kRunOnCallFailed`; 8 values); no stage-16 whole part.

**Amended 2026-11-14 (A15):** `SolvePlace` is `{ kWholeGridOnProcessor, kInPieces }`.

**Amended 2026-11-14 (A17, batched-run buffers):** stage 3's whole part has its place, and `kStagePlaces` must give
the step's array places (step plan Task 33a).

**Amended 2026-11-14 (reconcile):** ~~two layout tests on the five-material scene marked blocked under option (A) of
the open decision on a springy solid beside a viscous liquid.~~ Option C chosen (A24): test unblocked. Both layout
tests run as written.

**Amended 2026-11-14 (Foundation as built):** `ComputeSumOverBlocks` takes a `SumBuffers` (the buffers pattern); the folder gets its own
`CMakeLists.txt`.

**Amended 2026-11-14 (A31, A30):** `ComputeSumOverBlocks` has one form, which takes `SumBuffers& sum_buffers` (the
design's form without buffers is gone); buffers on another processor are a precondition, not a NaN;
`ScratchTooSmall_IsQuietNaN` renamed `BuffersTooSmall_IsQuietNaN`, `CutsGiveEqualBytes` no longer compares two forms.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/batches/batches_error.h` (shared), `batches_message.h`, `batches_message.cc`,
  `batch_plan.h` (types and `kStagePlaces`), `batch_plan.cc` (the table), `batched_run.h`
  (`ComputeSumOverBlocks`), `sum_over_blocks.cc`, `batches_error_test.cc`, `stage_places_test.cc`,
  `sum_over_blocks_test.cc`, `native/fluidity/batches/CMakeLists.txt` (through the helpers; every later task adds its
  files here)
- Modify: the CMake lines that list module folders

**Interfaces:**
- Produces design 3.7 `BatchesError`, `BatchesFailure`, `ToMessage`, `ToName` (plan decision: both in
  `batches_message.h`, so the shared header holds no string type), `StagePlace`, `kStagePlaces`, `SolvePlace`
  (plan decision A15: `enum class SolvePlace : std::uint8_t { kWholeGridOnProcessor, kInPieces };` the design's
  `kWholeGridOnCpu` is not built),
  `BatchSettings`, `BatchPlan`, `ComputeSumOverBlocks` (`ComputeTreeSum` over `block_count()` values; a failed GPU
  read returns a quiet NaN, documented). **Plan decision (amended 2026-11-14, Foundation as built; one form since
  A31, replacing the design's form):**

  ```cpp
  // Allocates nothing (ComputeTreeSum with `sum_buffers`). A block count above sum_buffers.max_count() or a failed
  // GPU call returns a quiet NaN, documented. Requires: sum_buffers.run_on() is the values' processor (checked by
  // an assertion in debug builds, A30).
  double ComputeSumOverBlocks(const grid::CellBlockValues<double>& block_partial_sums, SumBuffers& sum_buffers);
  ```
  Per-step callers pass `batched_run.sum_buffers` (Task 39); tests build theirs with `BuildSumBuffers`.
- **Plan decision, `kStagePlaces`** (design 5.8, with stage 10 whole because every call of step design 5.2 item 10
  either takes `all` or reads arrays pass 1 has written back): `kNone` `kNotRun`; 1, 2, 17, 20, 21
  `kWholeStateOnCpu`; 3 to 9 `kInBatchesBeforeSolve`; 10 to 13 `kWholeGridOnProcessor`; 14, 15, 16, 18, 19
  `kInBatchesAfterSolve`. Every non-null `run_whole_grid` part runs at `kWholeGridOnProcessor`: 3's (A17) right after
  pass 1, then 9's, 19's after stage 17 (A64: no 16's).

- [ ] **Step 1: Failing tests:** `ToName, EveryBatchesError_IsLowerSnakeCaseWithoutK` (`kBatchDoesNotFit` ->
  `"batch_does_not_fit"`, all 8 by `static_assert`); `BatchesError, OutOfMemoryIsLast` (`kOutOfMemory == 7`); `kStagePlaces, MatchesThePlanTable` (all 22); `kStagePlaces,
  PassTwoReadsNothingWrittenLater` (on `BuildFiveMaterialScene` in the batched layout, from `ReadStepArrays`: no array
  read by a stage of pass 2 is written by stage 17 or a `run_whole_grid` part that runs after pass 2; if this fails,
  stop and ask the owner); `kStagePlaces, GiveTheStepsArrayPlaces` (A17: for every array of `ReadStepArrays` on
  `BuildFiveMaterialScene` in the batched layout, the placement rule computed from `kStagePlaces` equals
  `StepArray::whole_place`; also: no range part of pass 1 reads `grid_edge_is_open_to_sky`, else stop and ask the
  owner); `ComputeSumOverBlocks, CutsGiveEqualBytes` (design 7 "Sum over blocks": 10,000 random
  partials, the partials of 1, 7 and 64 batches placed in one array: equal bytes);
  `ComputeSumOverBlocks, BuffersTooSmall_IsQuietNaN` (`SumBuffers` of `max_count` 10 given 11 blocks: NaN). Tests `LABEL cpu`, `LINKS fluidity_base_test_support`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Name the places of every stage in a batched run`.

### Task 38: `BuildBatchPlan` and the batches' memory estimate entries

**Amended 2026-11-14 (A31):** `"batches: sum buffers"` entry name; the loaded try borrows the step's `grid::GridBuilderBuffers`; test `SumBuffersEntriesMatchTheBuffers`.

**Amended 2026-11-14 (A15):** the plan chooses between whole-grid solves and solves in pieces, sizes the solve
piece, checks the CPU, and carries the budgets and copies shown to the user.

**Amended 2026-11-14 (A17, batched-run buffers):** the loaded and CPU rows come from the step's batched-layout rows
(step plan Tasks 33a, 33b: loaded module instances, the stage-17 solids buffers in CPU memory, the mirrors and the
CPU grid copy), so the budgets I/O shows (I/O Task 23a) equal what `BuildSimulation` allocates.

**Amended 2026-11-14 (reconcile):** the loaded grid buffers are counted without a builder buffers (they borrow the
step's).

**Amended 2026-11-14 (Foundation as built):** `BuildBatchesMemoryEstimate(plan)` adds the two entries of `BatchedRun`'s `SumBuffers` (Task 39).

**Amended 2026-11-15 (A58; correction, reconcile):** `BuildBatchesMemoryEstimate(plan)` lists exactly what
`BuildBatchedRun` builds (the loaded rows it listed for reporting are the step's batched rows of item 7 and are
not repeated), so the run's blocks can hold it; `bytes_on_processor` is `ComputeRunMemoryBytes(rows, counts, kRunOn)`
(pieces rounded to 256 B) and includes those entries; `ReadMemoryReport` names `handed_out_bytes` and
`handed_out_cpu_bytes`.

**Amended 2026-11-15 (A65, A66, A68):** the 512 particles per full block are stated in the A65 wording (every level);
start blocks are `counts.start_cell_blocks` (A66); the plan fills design 3.7's `SlabLimit` on success and in the
pass-slab `kBatchDoesNotFit` failure, the numbers I/O's budget line names (A68).

**Kind:** numerics

**Files:**
- Modify: `batch_plan.h`, `batch_plan.cc`, (A68) `batches_error.h` (`SlabLimit`, `BatchesFailure::slab_limit`);
  Create: `batch_plan_test.cc`

**Interfaces:**
- Produces design 3.7 `BuildBatchPlan`, `BuildBatchesMemoryEstimate`.
- **Plan decisions** (design 5.9; the start grid is not built, so the plan allocates nothing):
  1. Whole-grid rows: `estimate.by_module` of `kGrid`, `kPressure`, `kViscosity`, `kGas`, `kSolids` (rows not counted
     per particle) and the step's `GridFields` rows, at the limits. At most `max_whole_grid_share * allowed_bytes`:
     `kWholeGridOnProcessor`. Above (A15): `kInPieces`; the whole rows are recomputed from the solvers' row builders
     in the batched layout (their `kRunOn` rows at the limits only); `room = allowed - whole`;
     `bytes_per_solve_loaded_block` = the solvers' loaded rows at one block (pressure liquid and gas, viscosity, gas:
     they coexist); `max_solve_loaded_blocks = min(floor(solve_piece_share * room / bytes_per_solve_loaded_block),
     limits.max_cell_blocks)`; fewer than `3 * layer` (layer as item 4): `kBatchDoesNotFit` with the bytes; the
     loaded batch then gets `(1 - solve_piece_share) * room` in item 3 instead of `room`. `BatchSettings` gains
     `double solve_piece_share = 0.25;` (A15; owner to confirm).
  2. `bytes_per_loaded_block` = `ComputeMemoryBytes` of the loaded rows (transfer for every set, the step's per-try
     cell and face rows, `grid::BuildGridMemoryEstimate(run_on, 1, limits.max_cell_blocks, grid::BuilderBuffersSharing::kOneForAllGrids)` without
     its `builder buffers:` entries, since the loaded try's buffers borrow the step's `grid::GridBuilderBuffers`; reconcile item 11)
     at the counts of one cell block and no particle;
     `bytes_per_loaded_particle` the same at one particle and no block; `particles_per_block = ceil(512 *
     start_limit_headroom)` per material (8 per cell, full block). (A65) 512 holds at every level: every liquid cell
     holds about eight particles of its own size (a particle's `size_level` is its cell's level, rest volume
     `(w/2)³`), never `8 · 8^k` in a coarse cell.
  3. `max_loaded_blocks = min((allowed - whole) / (per block + particles_per_block * per particle),
     limits.max_cell_blocks)`; `max_loaded_particles = max_loaded_blocks * particles_per_block`. (Amended
     2026-11-15, A58: `bytes_per_loaded_block` includes `BatchedRun`'s per-loaded-block entries, the border search and
     two border buffers; then `max_loaded_blocks` is lowered one at a time while `ComputeRunMemoryBytes` of all
     `kRunOn` rows exceeds `allowed`, since each piece rounds up by less than 256 B.)
  4. Blocks are ordered `z` first, so a range is a slab: `layer = (domain cells along x / 4) * (along y / 4)` at the
     smallest width; `T = floor(max_loaded_blocks / layer) - 8`; `max_blocks_per_batch = T * layer`;
     `kBatchDoesNotFit` when `T < 1` or `max_blocks_per_batch < min_blocks_per_batch`. The exact cut is found per try
     in Task 40. `range_share_of_loaded = T / (T + 8)`; `batch_count = ceil(start blocks / max_blocks_per_batch)`;
     `bytes_copied_per_try = (2 / range_share_of_loaded + 2) * start blocks * (per block + particles_per_block *
     per particle)`. (A66) `start blocks = counts.start_cell_blocks` (`step::SceneCounts`, the blocks of
     `step::BuildStartCells`' grid).
  4a. (A68) `SlabLimit` (design 3.7, in `batches_error.h` beside `BatchesFailure`): `blocks_per_layer = layer`;
     `bytes_per_full_loaded_block = b = per block + particles_per_block * per particle`; `loaded_batch_room_bytes = R`
     (item 3's room: `allowed - whole`, or `(1 - solve_piece_share) * room` with `kInPieces`); `T_min = max(1,
     ceil(min_blocks_per_batch / layer))`, `loaded_layer_count = n = T_min + 8`; `needed_bytes = n * layer * b`
     (64-bit, exact); `plan_area = X * Y` (domain extents, m²); `max_plan_area = R * 16 * w_min * w_min / (n * b)`
     (doubles, in this order). Filled in `BatchPlan::slab_limit` on success, and on item 4's `kBatchDoesNotFit` in
     `BatchesFailure::slab_limit` with `needed_bytes = slab_limit.needed_bytes`, `allowed_bytes = R`; every other
     failure leaves `slab_limit` zero (`loaded_layer_count == 0`). I/O's `BuildBudgetLines` reads these fields for
     the line of `plan-decisions.md` A68 (`<n>` = `loaded_layer_count`, the first `<b>` = `needed_bytes`, the second
     `<b>` = `loaded_batch_room_bytes`, `<a>` = `max_plan_area` and `plan_area`).
  5. (A15) CPU: every row with `where = kCpu` (the particles, the per-try arrays only particle stages read, and in
     `kInPieces` the solvers' working buffers) at most `plan.allowed_cpu_bytes`, else `kStateDoesNotFitCpu`.
  6. (A15) Report, `BatchPlan` gains: `max_solve_loaded_blocks`, `solve_batch_count` (start grid, bisection as
     `pressure::ApplySolveBatches` would cut it), `bytes_on_processor` and `bytes_on_cpu` (the memory budgets: sums
     of the rows by `where`; amended 2026-11-15, A58: `bytes_on_processor = ComputeRunMemoryBytes(rows, counts,
     kRunOn)` and `bytes_on_cpu = ComputeRunMemoryBytes(rows, counts, kCpu)` plus the plain `kOneCall` CPU rows),
     `pass_loads_per_try = 2 * batch_count`, `solve_loads_per_iteration` (3, or 4 when the
     scene can hold a sealed blob: a closed domain, times `solve_batch_count`), `solve_bytes_per_iteration` (that
     many sweeps of the streamed rows of every loaded block). Zero with `kWholeGridOnProcessor`.
  `BuildBatchesMemoryEstimate(plan)` (amended 2026-11-15, A58): design 3.7, exactly the entries `BuildBatchedRun`
  builds, all `kWholeRun`: `"batches: border search"` (`grid::BuildBorderSearchMemoryEstimate(run_on, limits.max_cell_blocks)` at
  `max_loaded_blocks`), `"batches: border layer a"`, `"batches: border layer b"` (4 B per loaded block each),
  `"batches: sum buffers"` (`kCellBlock`, `ComputeSumBuffersSize(plan's run_on).bytes_per_item`) and `"batches: sum
  buffers fixed bytes"` (`kGivenCount`, `fixed_bytes` items of 1 byte), `where = kRunOn`, and `"batches: piece map
  copy"` (4 B per loaded block, `where = kCpu`). The step adds them to its estimate (`by_module` `kBatches`) so
  `run_memory` and `cpu_run_memory` hold them.
  7. (A17, batched-run buffers) Every row sum above takes the rows of `step::BuildStepMemoryEstimate(scene, counts,
     &probe_plan)`: per loaded block with a probe plan of `max_loaded_blocks = 1`, `max_loaded_particles = 0`; per
     loaded particle with `0` and `1`; the whole and CPU rows (the mirrors, the CPU grid copy in a GPU run, the
     stage-17 `SolidsBuffers` in CPU memory, the `MarkerChanges` rows) at the limits by their `where`. So
     `bytes_per_loaded_block` and `bytes_per_loaded_particle` include the loaded `TransferBuffers`,
     `MaterialBuffers`, `ContactBuffers` and `SolidsBuffers`, and `bytes_on_processor` / `bytes_on_cpu` equal the
     allocation of `BuildSimulation` for the copied plan by `where`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; `RunPlan` from `step::BuildRunPlan` with chosen `MemoryAvailable`):
  `BuildBatchPlan, WholeGridBelowShare_SolveOnProcessor`; `BuildBatchPlan, WholeGridAboveShare_SolveInPieces` (A15);
  `BuildBatchPlan, FitsOnlyWithSolvesInPieces` (A15: allowed bytes chosen so that the whole-grid solve rows exceed
  the share but the batched layout fits: `kInPieces`, and every row sum below `allowed_bytes`);
  `BuildBatchPlan, SolvePieceBelowThreeLayers_IsBatchDoesNotFit` (A15); `BuildBatchPlan,
  CpuRowsAboveAllowedCpu_IsStateDoesNotFitCpu` (A15); `BuildBatchPlan, ReportsBudgetsAndCopies` (A15:
  `bytes_on_processor` and `bytes_on_cpu` equal `ComputeMemoryBytes` of the rows by `where`;
  `solve_loads_per_iteration == 3 * solve_batch_count` for an open-top wave, `4 *` for a closed box);
  `BuildBatchPlan, BudgetsEqualTheBatchedAllocation` (A17: the plan copied into the step plan,
  `step::BuildSimulation` on the CPU with a forced batched layout, then `BuildBatchedRun` into its blocks:
  `ReadMemoryReport` `handed_out_bytes` and `handed_out_cpu_bytes` equal `bytes_on_processor` and `bytes_on_cpu -
  estimate.snapshots`; amended 2026-11-15, A58);
  `BuildBatchPlan, LoadedRowsIncludeModuleInstances` (A17: `bytes_per_loaded_particle` exceeds the particle rows by
  the per-particle rows of the four loaded module instances, exactly);
  `BuildBatchPlan, LayerWiderThanABatch_IsBatchDoesNotFit`; `BuildBatchPlan, MoreMaterialSets_FewerLoadedBlocks` (1
  and 3 sets); `BuildBatchPlan, RunOnGpuWithoutAGpu_StillPlans` (all design 7 "Batch plan");
  `BuildBatchesMemoryEstimate, OnlyWhatBuildBatchedRunBuilds` (A58: six entries, names as above; replaces
  `LoadedRowsAtMaxLoadedBlocks`); the run-memory test of the estimate is Task 39's
  `BuildBatchedRun, IntoRunMemory_HandsOutExactlyTheEstimate` (replaces `SumBuffersEntriesMatchTheBuffers`);
  (A68, design 7 "Slab limit named") `BuildBatchPlan, LayerDoesNotFit_ReportsSlabLimit` (a closed pool 0.8 x 0.8 x
  0.4 m at `w_min` 0.0125 m, so `blocks_per_layer` = 16 x 16 = 256; `MemoryAvailable` chosen so that `R = 8 * 256 * b
  + 100` B: `kBatchDoesNotFit`, `slab_limit.blocks_per_layer` 256, `loaded_layer_count` 9, `needed_bytes = 9 * 256 *
  b`, `failure.needed_bytes == needed_bytes`, `failure.allowed_bytes == R`, `plan_area` 0.64 m², `max_plan_area ==
  R * 16 * 0.0125² / (9 b)` to the byte and in [0.5688, 0.5690] m²) and `BuildBatchPlan, Fits_ReportsSlabLimit` (the
  same pool with `R` = 10 layers: `T` = 2, `max_blocks_per_batch` 512, `plan.slab_limit` by the same formulas);
  `BuildBatchPlan, SolvePieceBelowThreeLayers_IsBatchDoesNotFit` also checks `slab_limit.loaded_layer_count == 0`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Plan the batches of a scene that does not fit the GPU`.

### Task 39: `BatchedRun`, `WriteBatch` and `ReadBatch`

**Amended 2026-11-15 (A58; correction, reconcile; item 11):** `BuildBatchedRun(const BatchPlan& plan, const
step::ReadyScene& scene, RunMemory& run_memory, RunMemory& cpu_run_memory)` builds the entries of Task 38's
`BuildBatchesMemoryEstimate` as pieces (the piece map copy from `cpu_run_memory`); a failed piece is `kOutOfMemory`
(the last `BatchesError`) with `needed_bytes` = the bytes asked; new test `BuildBatchedRun,
IntoRunMemory_HandsOutExactlyTheEstimate`.

**Amended 2026-11-14 (A17, A15):** borders and pieces through the grid's buffer forms with the piece map read back
once per load; pass 1's geometry reaches the whole buffers through `pressure::WriteSnapshotRange`.

**Amended 2026-11-14 (A17, batched-run buffers):** after pass 1 `ReadBatch` also moves the `CollideSolids` block
partials into the stage-17 solids buffers in CPU memory, and foam and spray get the CPU copy of the piece map
(step plan Appendix A rows M10, M13).

**Amended 2026-11-14 (Foundation as built):** `BatchedRun` owns `sum_buffers`; the test support goes into `fluidity_cell_resize_test_support`
(made with `fluidity_add_test_library`), which links `fluidity_pressure_test_scenes` for `CountBlocksInLargestLayer`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/batches/batched_run_cpu.cc`, `stacked_border.h` (private, CPU only), `stacked_border.cc`,
  `batched_run_test.cc`, `stacked_border_test.cc`; test-only `batches_test_support.h/.cc` (in
  `fluidity_cell_resize_test_support`)
- Modify: `batched_run.h`, `native/fluidity/batches/CMakeLists.txt`, `native/fluidity/cell_resize/CMakeLists.txt`
  (amended: `batches_test_support.cc` joins the SOURCES of `fluidity_cell_resize_test_support`, whose LINKS gain
  `fluidity_pressure_test_scenes`)

**Interfaces:**
- Produces design 3.7 `BatchReport`, `BatchedRun`, `BuildBatchedRun` (with the two `RunMemory&` last), `ReadLastReport`,
  `WriteBatch`, `ReadBatch`.
- **Plan decisions:** (amended, Foundation as built) `BatchedRun` also holds `SumBuffers sum_buffers`
  (`BuildSumBuffers(run_on, limits.max_cell_blocks, run_memory)`, the two entries of Task 38) for `ComputeSumOverBlocks`.
  `BatchedRun` holds the plan, a `grid::BorderSearchBuffers` and two `Buffer<CellBlockIndex>` of
  `max_loaded_blocks` on the run's processor (A17), a CPU copy of the piece map (4 B per loaded block, row `"batches:
  piece map copy"`, `where = kCpu`), the current `std::optional<grid::GridPiece>` and the report; the piece is built
  into `step::FindLoadedGridBuffers`. Private (A17, `plan-decisions.md`): `struct StackedBorder { const
  Buffer<grid::CellBlockIndex>* blocks; std::uint32_t count; };` `std::expected<StackedBorder, BatchesFailure>
  FindStackedBorder(const grid::StaggeredGrid& grid, grid::CellBlockRange range, std::span<const grid::BorderWidth>
  widths_from_last_stage, grid::BorderSearchBuffers& search_buffers, Buffer<grid::CellBlockIndex>& layer_a,
  Buffer<grid::CellBlockIndex>& layer_b)`: one buffer-form `grid::FindBorderBlocks` per layer, the two buffers in
  turn, only counts come back; pass 1 widths `kTouchingCells, kTouchingCells, kFourCellWidths, kTouchingCells`; pass
  2 `kFourCellWidths, kTouchingCells`. `WriteBatch`: release the previous piece, the buffer form of
  `grid::BuildGridPiece`, one `ReadBuffer` of `block_in_whole_grid_buffer()` into the map copy (a CPU run uses
  `block_in_whole_grid()`; counted in `bytes_from_processor`), `step::ApplyGridPiece`,
  `particles::WriteParticleGroups`, then `step::WriteStepArrayBlocks` (whole -> loaded, one `BlockCopy` per run of
  loaded blocks, runs computed from the map copy) for every array read by a stage of the pass. `ReadBatch`: for
  the range blocks only, every array written by a stage of the pass that a later place reads or the commit keeps
  (loaded -> whole; for whole-grid arrays this is the GPU-to-GPU slice copy of design 5.8); `particles::ReadParticleGroups` with pass 1 `{kPlateSide}` and
  pass 2 every field `BuildFieldsToChange` accepts; after pass 1 also `pressure::WriteSnapshotRange(loaded try's
  piece buffers, piece, whole buffers)` (A15; step plan Task 33) and (A17, batched-run buffers)
  `solids::WriteBlockPartialsRange(loaded try's SolidsBuffers, piece, *step::FindContactReactionBuffers(sim))`,
  its bytes added to `bytes_from_processor` (part of the load: a load counts as one copy, plan-decisions). Every load passes the map copy to
  `step::ApplyGridPiece(loaded_try, piece, map_copy)` (Task 36), which stage 18 hands to `make_foam_and_spray` as
  `FoamAndSprayInputs::block_in_whole_grid` (a GPU piece's own span is empty, A17). Test support: `BatchPlan
  BuildForcedBatchPlan(const
  step::ReadyScene&, const step::RunPlan&, const step::StepSettings&, std::uint32_t max_blocks_per_batch,
  std::uint32_t max_solve_loaded_blocks = 0)` (ample allowed bytes, then the cut forced; a non-zero
  `max_solve_loaded_blocks` forces `kInPieces`); layer counts with `pressure::testing::CountBlocksInLargestLayer`
  (A15: the pressure test library's, not a second copy).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, both processors the CPU; `LINKS fluidity_cell_resize_test_support
  fluidity_base_test_support`):

  ```cpp
  TEST(FindStackedBorder, PassOne_EqualsRepeatedFindBorderBlocks)  // 100 random grids from grid_test_support; the
      // buffer form on CPU buffers against the span form layer by layer (A17)
  TEST(WriteBatch, PieceMapReadBackOncePerLoad)       // A17: the report's bytes_from_processor grows by 4 B per loaded
      // block per load
  TEST(ReadBatch, PassOneGeometryEqualsWholeGrid)     // A15: after pass 1 the whole buffers' snapshot rows equal
      // FindLiquidSurface on the whole try, bytes
  TEST(WriteBatch, TwoRangesInTurn_EachLoadRebinds)   // design 7 "Load binding": ranges of different block counts:
      // block_count() follows each piece
  TEST(WriteBatch, ArrayBoundAtEarlierLoad_IsWrongGridVersion)  // a loaded array kept from the first load, passed to
      // transfer::ParticlesToGrid after the second: kWrongGridVersion from transfer
  TEST(ReadBatch, WritesBackRangeBlocksOnly)          // border blocks of the whole try keep their bytes
  TEST(WriteBatch, LoadsTheParticlesOfLoadedBlocks)
  TEST(BuildBatchedRun, ZeroLoadedBlocks_IsBadInput)
  TEST(BuildBatchedRun, IntoRunMemory_HandsOutExactlyTheEstimate)  // A58: CPU run, one block of exactly
      // ComputeRunMemoryBytes(BuildBatchesMemoryEstimate(plan), counts, kRunOn) + the same for kCpu passed as both
      // blocks: handed_out_bytes() == reserved_bytes(), exact
  TEST(BuildBatchedRun, RunMemoryOnePieceShort_IsOutOfMemory)  // the same block less 256 B (GTEST_SKIP in debug):
      // kOutOfMemory, needed_bytes == the bytes the failing piece asked
  TEST(ReadLastReport, CountsBytesBothWays)
  TEST(ReadBatch, CollideSolidsPartialsReachTheStage17Buffers)  // A17 M10: BuildFreeBoxScene(500), 3 batches: the
      // CPU instance's block partials equal a one-piece CollideSolids' bytes; the report counts their bytes once
  TEST(WriteBatch, FoamInputsGetThePieceMap)       // A17 M13: a hand-written make_foam_and_spray sees
      // block_in_whole_grid equal to the map copy, never empty, in every batch
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Load a range with its stacked border and read its results back`.

### Task 40: `ApplyStagesInBatches`

**Amended 2026-11-14 (A15):** the byte comparison with one piece is kept only for pass 1's local outputs; cuts are
compared with each other; the solvers' copy reports go into `BatchReport`.

**Amended 2026-11-14 (reconcile):** ~~the tests on the five-material scene are marked blocked by the open decision on
a springy solid beside a viscous liquid (expectations unchanged).~~ Option C chosen (A24): test unblocked. The
five-material tests run as written (syrup beside the jelly set: the joint viscous solve carries the material faces).

**Kind:** integration

**Files:**
- Create: `native/fluidity/batches/apply_stages_in_batches.cc`, `apply_stages_in_batches_test.cc`
- Modify: `batched_run.h`

**Interfaces:**
- Produces design 3.7 `ApplyStagesInBatches`, numbered `// Step N:` from `<proc-apply-stages-in-batches>`:
  1. cut: the largest `n <= plan.max_blocks_per_batch` for which every range of `grid::BuildRanges(try grid, n)`
     with its pass-1 stacked border has at most `max_loaded_blocks` blocks (bisection with `FindStackedBorder`); none:
     `kStageFailed`, `module = kBatches`, `module_error = kBatchDoesNotFit`;
  2. pass 1: per range, `WriteBatch`, the `run` of stages 3 to 9 on the loaded try (each stage on the contiguous
     piece blocks from the first to the last of the range and the stacked borders of the stages after it), `ReadBatch`;
  3. stage 9's `run_whole_grid`, then 10 to 13 on the whole try with every block;
  4. pass 2 (stages 14, 15, 16, 18, 19 `run`), then 16's `run_whole_grid`, stage 17, 19's `run_whole_grid`.
  The cancel flag is read before every stage and every batch (`kCancelRequested`; in `kInPieces` the solvers also
  read it before every solve batch and their `kCancelled` maps to `kCancelRequested`). A stage outcome other than
  `kNone` ends the call; reports are summed by the step from block partials. `BatchReport` per try (A15: plus
  `solve_batch_loads`, `solve_bytes_to_processor`, `solve_bytes_from_processor`, `solve_copy_elapsed_seconds`,
  summed from each solver's `pressure::SolveCopyReport` after stage 13).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, both processors the CPU; `BuildFiveMaterialScene` without snow, cut 1
  layer, unless named):

  ```cpp
  TEST(ApplyStagesInBatches, PassOneStagePrefix_SameBytesAsWholeGrid)  // design 7 "Stacked border" (A15: pass 1
      // only): for each stage k of pass 1, a table with the run parts after k null, on a mixed-size grid
      // (BuildDeepTankScene after 5 resizes): the arrays pass 1 wrote, batched vs the same stages on the whole
      // grid: ExpectSameBytes (local work: a difference means a border too narrow)
  TEST(ApplyStagesInBatches, CutInsideALayer_SameBytesAsLayerCut)  // Review Focus 5: max_blocks_per_batch 7 and 13
      // against one layer, both batched modes: bytes; against one piece: within the step design 6.4 tolerances
  TEST(ApplyStagesInBatches, CancelBetweenBatches_IsCancelRequested)  // nothing committed; the next try commits
  TEST(ApplyStagesInBatches, GridGrewPastLoadedBlocks_IsBatchDoesNotFit)
  TEST(ApplyStagesInBatches, ReportCountsBatches)
  TEST(ApplyStagesInBatches, ReportCountsSolveCopies)       // A15: kInPieces: solve_batch_loads == the solvers'
      // reports summed; 0 with kWholeGridOnProcessor
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Run a step's stages in batches around a whole-grid solve`.

### Task 41: The central test, a batched run is repeatable and agrees with one piece

**Amended 2026-11-14 (A15):** was "a batched run gives the bytes of a one-piece run". Now: identical bytes for
every cut of one mode, and agreement within the solver's tolerance with one piece, in both batched modes.

**Kind:** integration

The module is not done until every test of this task exists and passes.

**Files:**
- Create: `native/fluidity/batches/batches_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`, both processors the CPU: `StepSettings::run_on = kCpu`, the plan forced with
  `BuildForcedBatchPlan` to `kBatched` and, separately, to `kBatchedWithSolvesInPieces` with
  `max_solve_loaded_blocks` = 3 layers of the start grid, `run_stages = {ApplyStagesInBatches, &batched_run}`). For
  each scene, 20 committed steps through `step::AdvanceTo` (`on_step_done` stops after 20), then `ReadSnapshot`
  with every field; runs with `max_blocks_per_batch` = 1 layer (`pressure::testing::CountBlocksInLargestLayer` of
  the start grid), 3 layers and every block (and, in the second mode, solve pieces of 3 and 5 layers), and the
  one-piece run of the same settings. Within one mode every cut `ExpectSameBytes`; each batched run against one
  piece within the step design 6.4 tolerances (the CPU-against-GPU rule; design 7 "GPU and batches", container
  form); the test also prints whether the bytes happen to be equal:

  ```cpp
  TEST(ApplyStagesInBatches, SmallWave_SameBytesForEveryCut)              // BuildWaveScene(1)
  TEST(ApplyStagesInBatches, SmallWave_WithinStepToleranceOfOnePiece)
  TEST(ApplyStagesInBatches, SmallSyrup_SameBytesForEveryCut)             // BuildSmallSyrupScene
  TEST(ApplyStagesInBatches, SmallSyrup_WithinStepToleranceOfOnePiece)
  TEST(ApplyStagesInBatches, SmallFire_SameBytesForEveryCut)              // BuildFireReferenceScene(0.1)
  TEST(ApplyStagesInBatches, SmallFire_WithinStepToleranceOfOnePiece)
  TEST(ApplyStagesInBatches, SmallSnow_SameBytesForEveryCut)              // BuildSmallSnowExample
  TEST(ApplyStagesInBatches, SmallSnow_WithinStepToleranceOfOnePiece)
  ```
- [ ] **Step 2: Run.** A cut that changes bytes names the field and the first index: find the stage with
  `PassOneStagePrefix_SameBytesAsWholeGrid` (pass 1) or the solvers' `...InBatches_...` tests (solves), and fix the
  owning commit (a sum not made of block partials, a border too narrow), never the test. **Step 3:** gate.
  **Commit.** Subject: `Show a batched run is repeatable and agrees with one piece`.

### Task 42: Batched solids, fire beside water and cell resize

**Amended 2026-11-15 (A64):** the deep-tank resize test runs with particles in every cell (no grid-only stage parts).

**Amended 2026-11-14 (A15):** the tests handed over by solids and gas, and the resize test, compare with one piece
within tolerance, and cuts of one mode by bytes, in both batched modes.

**Kind:** integration

**Files:**
- Modify: `batches_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`, as Task 41, both batched modes):

  ```cpp
  TEST(Batches, FloatingBox_WithinStepToleranceOfOnePiece)   // handed by the solids plan (A15 name): FreeBox,
      // HalfDensity_FloatsHalfSubmerged (BuildFreeBoxScene(500), AdvanceTo 1 s) in batches: every solid state and
      // ExchangeTotals within the step design 6.4 tolerances of one piece; identical bytes across cuts
  TEST(Batches, FreeBoxCuts_SamePushesAndReactions)  // design 7 "Per-solid sums": cut into 1, 3 and 9 batches (and
      // solve pieces of 3 and 5 layers): pushes and contact reactions byte for byte (cuts of one mode)
  TEST(TryStep, BatchedFireBesideWater_WithinStepTolerance)  // handed by the gas plan (A15 name): BuildSmallFireScene
      // with a 0.2 m pool under the plume, 20 steps: gas amounts and both sets' velocities within the step design
      // 6.4 tolerances of one piece; identical bytes across cuts and on repeat
  TEST(Batches, DeepTankWithCellResize_WithinStepToleranceOfOnePiece)  // plan addition: BuildDeepTankScene with
      // BuildCellResizeCallbacks plus run_stages, 20 steps, a resize after each
  ```
- [ ] **Step 2: Run**; fix owning commits if needed. **Step 3:** gate. **Commit.** Subject:
  `Test solids, gas and resized cells in batches`.

### Task 43: Batched runs on the GPU, and the resize beside them

**Amended 2026-11-15 (A64, A41, A58):** the mirrors hold the three wait arrays instead of the seven region arrays;
the `size_change` bytes are computed on the CPU mirrors in this layout (no read back); the mirrors and the CPU grid
copies are pieces of `cpu_run_memory`.

**Amended 2026-11-14 (A31):** the resize's working buffers (was "scratches") are on the CPU in this layout.

**Amended 2026-11-14 (A15, A17):** GPU runs of both batched modes; the resize's CPU mirrors with the grid copy of
Task 24 instead of CPU tables of a GPU grid.

**Amended 2026-11-14 (A17, batched-run buffers):** in a batched GPU run the resize reads the step's CPU copy of the
committed grid instead of making its own `ReadGrid` copy (step plan Appendix A row M14).

**Amended 2026-11-14 (reconcile):** `GridCopies::kHeldByStep`: no idle CPU grid copy in this layout; the resized CPU
grid becomes the step's CPU copy.

**Amended 2026-11-14 (Foundation as built):** `gpu` tests pass their snapshots to `WriteResultDigest`; GPU against CPU within tolerance; the
resize's working buffers are on the CPU in this layout (the estimate uses the CPU sizes, Task 10).

**Kind:** device

**Files:**
- Create: `native/fluidity/batches/batches_gpu_test.cc`, `batches_no_gpu_test.cc`
- Modify: `native/fluidity/cell_resize/cell_resize.cc`, `resize_cells_cpu.cc`, `resize_cells_gpu_test.cc`,
  `cell_resize_test.cc` (memory estimate entries)

**Interfaces:**
- **Plan decision** (design 5.8: `ResizeCells` runs on the CPU in a batched run and reads `GridFields` back once): a
  `CellResize` with `run_on = kCpu` holds CPU mirrors (`where = kCpu` rows: `GridFields` both velocities, pressure,
  `has_pressure`, `pocket_of_cell`, `merge_ready_duration`, `merge_ready_step_count`, `held_velocity_gradient_size`
  (A64: was the seven region arrays); `CellGeometry` and `FacePieceGeometry` of the last
  geometry; the seven gas arrays). When `simulation.run_on()` differs, `ResizeCells` reads the committed arrays into
  the mirrors, works on the CPU, and writes the try arrays back with slice `WriteBuffer`. **A17:** such a
  `CellResize` is built with `grid_run_on = kGpu` (Task 10), so the grid it reads is Task 24's `ReadGrid` copy and the
  try's grid is built on the GPU by the step from the uploaded list; the mirrors and the copy are counted in
  `ResizeCellsReport::bytes_copied`. In `kBatchedWithSolvesInPieces` the mirrored arrays are the same (state and
  geometry stay on the GPU in both batched modes).
- **Plan decision (A17, batched-run buffers; amended by the reconcile pass):** in a batched GPU run `CellResize` is
  built with `GridCopies::kHeldByStep` (Task 10): `ResizeCells` step 1 reads `step::FindCpuGridCopy(simulation)` in
  place of a `ReadGrid` copy (version and block count checked against the committed grid, `kWrongGridVersion`
  otherwise), and step 4 builds the CPU resized grid into `ResizeTry::free_cpu_grid_copy_buffers()` and hands it to
  the step with `step::ApplyResizedCpuGridCopy`; `bytes_copied` and `copy_count` leave out the 456 B per block of
  the copy. Neither CPU grid row of `CellResize` is built or counted in this layout (the earlier "still built, idle"
  is withdrawn).

- [ ] **Step 1: Tests.** `LABEL gpu`: `ApplyStagesInBatches, OnGpuEveryScene_WithinStepTolerance` (Task 41's scenes
  with `run_on = kGpu`, both batched modes: the cuts equal bytes, and within step design 6.4 of the one-piece GPU
  run); `ApplyStagesInBatches, OnGpuTwice_GivesIdenticalBytes` (both modes); `ResizeCells,
  InBatchedGpuRun_EqualsOnePieceCpuResize` (within design 6); `ResizeCells, InBatchedGpuRun_ReadsTheStepsGridCopy`
  (A17: `bytes_copied` has no `456 * blocks` term; the result equals the run with its own copy; reconcile: after the
  commit `step::FindCpuGridCopy` is the CPU resized grid and `step::ReadCopyCounts` did not grow); `ResizeCells,
  OnePieceGpuRun_UsesTheGridCopy` (A17: `bytes_copied` as Task 24's formula; same report as the CPU run within design 6);
  every `gpu` test writes the digest of its final snapshot or report with `WriteResultDigest`. `LABEL no_gpu`:
  `BuildBatchedRun, OnGpuWithoutAGpu_IsRunOnCallFailed`; `BuildBatchPlan,
  OnGpuInCudaBuild_StillPlans`. `LABEL cpu`: `BuildCellResizeMemoryEstimate, CpuMirrorsCountedOnCpu`;
  `BuildCellResizeMemoryEstimate, BatchedGpuLayout_NoGridCopyAllocated` (reconcile: a `kHeldByStep` `CellResize`
  allocates exactly its estimate, which has no CPU grid entry, and its working-buffers entries are the CPU sizes).
- [ ] **Step 2: Implement.** **Step 3:** gate. **Commit.** Subject: `List the batched GPU runs and resize beside
  them`. Body: `CUDA: compiled, not run`.

### Task 44: A scene that fits only with its solves in pieces

**Amended 2026-11-14 (A15):** new task; the planner's third choice through a whole run, with the planned and the
measured copies, cancel and failure inside the solve batches.

**Amended 2026-11-14 (Foundation as built):** `BuildPlanForMemory` lives in `fluidity_cell_resize_test_support` (`fluidity_add_test_library`);
the scene test names it in `LINKS`.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/batches/batches_scene_test.cc`, `batches_test_support.h/.cc`

**Interfaces:**
- Consumes: Tasks 38 to 43; step `BuildRunPlan`, `BuildSimulation`, `AdvanceTo`, `ReadSnapshot`, `ExpectSameBytes`.
- Test support (plan decision): `step::RunPlan BuildPlanForMemory(const step::TestScene& scene, std::uint64_t
  processor_bytes, std::uint64_t cpu_bytes)` (`step::BuildRunPlan` with those `MemoryAvailable` values, then
  `BuildBatchPlan` copied in as the I/O `Run` does).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, both processors the CPU; `BuildDeepTankScene` without cell resize
  and with its domain lengthened to 3.2 m, so the start grid has at least 16 block layers):

  ```cpp
  TEST(BatchedRun, FitsOnlyWithSolvesInPieces_Commits)      // processor bytes chosen between the batched layout with
      // solves in pieces and the one with whole-grid solves: plan kBatchedWithSolvesInPieces (and kBatchDoesNotFit
      // when SolvePlace is forced to kWholeGridOnProcessor); BuildSimulation hands out the planned bytes (kRunOn and kCpu
      // sums each); 10 steps commit; liquid volume change within pressure's reported change
  TEST(BatchReport, FirstTrySolveCopiesEqualThePlan)        // first try: solve_batch_loads per iteration ==
      // BatchPlan::solve_loads_per_iteration and solve bytes per iteration == solve_bytes_per_iteration (same grid,
      // same cut, same rows)
  TEST(ApplyStagesInBatches, CancelBetweenSolveBatches_IsCancelRequested)  // the flag set by a pressure test hook at
      // the 5th solve batch load: kCancelRequested; snapshot bytes equal those before the try; the next try commits
  TEST(ApplyStagesInBatches, SolveFailsInOneBatch_CommittedBytesKept)  // pressure max_iterations 0 for one try
      // (kNotConverged in the solve batches): the try is rejected, ExpectSameBytes on the committed snapshot, the
      // retry with the default settings commits
  ```
- [ ] **Step 2: Run** (fail until Tasks 38 to 43 are right; fix the owning commit). **Step 3:** gate. **Commit.**
  Subject: `Run a scene that fits only with its solves in pieces`.

### Task 45: Comment pass and hand-over

**Amended 2026-11-14 (A15):** renumbered from Task 44; its lists include Task 44 and the A15 and A17 changes.

**Amended 2026-11-14 (Foundation as built):** the hand-over also records the working-buffers members, the `_steps.h` headers, and that
the `cpu` steps tests are the GPU evidence until the hardware run.

**Amended 2026-11-14 (A29, A31):** the hand-over lists the one-form `ComputeLiquidTotals` and `ComputeSumOverBlocks`
and `grid::TakeBuffers`.

**Amended 2026-11-15 (A41, A58, A61 to A64):** the lists follow the first version (no grid-only work); the
hand-over states that the later groups are not done and records the A40 to A64 deliveries.

**Amended 2026-11-15 (A65 to A68):** Step 3 also records: the A65 wording in comments and the paper; the scenes
that start all-smallest on purpose and `BuildCalmDeepTankSceneOnStartCells` (A66); `AdjustParticleCount` at every
pass, the particle-only pass of step 3, `step::ResizeResult::size_change_kinetic_energy` and
`mismatched_particle_count` (A67); `SlabLimit` and the statement that this version keeps slab batches and the
pinned surface, with the later groups in A68's order, none done (A68).

**Kind:** mechanical

**Files:**
- Modify: any file of `native/fluidity/cell_resize/` and `native/fluidity/batches/`, the step files of Tasks 1, 35,
  36, `docs/method-paper/sections/cell_resize.typ`, `docs/glossary.md`, `docs/gpu-runs.md`,
  `tools/check_rules_allowed_names.txt`

- [ ] **Step 1: Comment pass.** Reread every file of both folders as a C++ programmer who has never read a fluid
  paper: comments true of the final code; why, not what; a worked example with numbers for `ComputeSplitAmounts`
  (the linear case of Task 5), `ComputeInnerFaceFlows` (Task 6's outflows), `ResizeCells` (the deep tank's first
  resize: how many cells split and merge, and the particles that follow) and `BuildBatchPlan` (A15: the three-way choice with the
  budgets and copies of a worked scene); every word in the glossary; `Uses` /
  `Used by` lists; `// Step N:` numbers match `<proc-resize-cells>`, `<proc-find-cells-that-must-split>`,
  `<proc-compute-cell-size-changes>`, `<proc-apply-stages-in-batches>`.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently:
  a wait-arrays-only change commits a same-cell resize and the wait grows by elapsed committed time and steps;
  `FindCellsThatMustSplit`'s `max_cell_count` rule; the private resize plan; the GPU plan reading per-cell arrays back instead of compacted
  ids; `BuildBatchPlan` cutting by layers (no start grid), the per-try bisection; stage 10 whole in a batched run;
  the step's `run_whole_grid`, `FindLoadedTry`, `FindLoadedGridBuffers`, `ApplyGridPiece`, `BlockCopy`,
  `WriteStepArrayBlocks`; the CPU mirrors of a batched resize; the test scenes of Task 1; A15: solves in pieces
  (`SolvePlace::kInPieces`, no CPU fallback), `solve_piece_share`, the plan's budgets and copies, the replaced byte
  tests; A17: the buffer-form `FindStackedBorder`, the piece map read back per load, the `ReadGrid` copy and
  `grid_run_on` of the resize; A17 batched-run buffers (amended 2026-11-14): `ResizeCellsReport::copy_count`,
  stage 3's whole part, `kStagePlaces` against the step's array places, the
  planner on the step's batched rows, `WriteBlockPartialsRange` in `ReadBatch`, the piece map handed to
  `ApplyGridPiece`, the resize reading `FindCpuGridCopy`; Foundation as built: `CellResize::sum_buffers` and the changed flag, `BatchedRun::sum_buffers`, the one-form (A31) `ComputeLiquidTotals`
  and `ComputeSumOverBlocks` taking `SumBuffers`, `ToCellResizeFailure`, the shared `*_steps.h` headers and their `cpu` tests, the Jacobi
  passes of the depth below the surface; A40 to A64 (2026-11-15): particles in every liquid cell and the later group,
  grid's `cell_size_rule.h` used through `ComputeCellSizeChanges` (two parts, `kKeepsItsSize` added to
  `CellSizeChange`), `ComputeVelocityGradientSize` (one-sided beside a missing neighbour, first order beside a bigger
  one), the held gradient and step count, the artist inputs, `depth_below_surface.h` and `liquid_totals.h`,
  `ResizeCellsReport::totals_before` / `totals_after`, `BuildCellResize` and `BuildBatchedRun` into the run's two
  `RunMemory` blocks, `BuildBatchesMemoryEstimate` limited to what `BuildBatchedRun` builds, `kOutOfMemory` last.
- [ ] **Step 4: `docs/gpu-runs.md`:** rows `cell resize` and `batches`, `compiled, not run`, listing every `gpu` test
  of Tasks 12, 13a, 17, 23, 28, 33 and 43, and the `cpu` steps tests that stand for them until the run.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate's output, new public names
  with reasons, the plan decisions, and the statement that no device code has run.

---

# After the first version: the later groups, in order (A68)

**Amended 2026-11-15 (A68):** owner answer A68 orders the later work: **(1)** "Can the fine surface band shrink from
8 cells to 4? (later, test only)", **(2)** "Coarser cells on calm surfaces (later)" together with "Batches cut by
columns or tiles (later)", **(3)** "Deep liquid without particles (later; needs review)", last. **The first version
keeps slab batches and the surface pinned to the smallest cells (A61).** When the slab rule is what stops a scene,
the planner's budget line names the plan-area limit: I/O's `BuildBudgetLines` writes the line (I/O plan) from the
`SlabLimit` fields Task 38 exposes (`blocks_per_layer`, `loaded_layer_count`, `bytes_per_full_loaded_block`,
`loaded_batch_room_bytes`, `needed_bytes`, `plan_area`, `max_plan_area`). None of the groups below is in a
done-criterion of this module.

---

# Can the fine surface band shrink from 8 cells to 4? (later, test only)

**Amended 2026-11-15 (A68):** new group, first after the first version. **Excluded from the module's done-criteria.**
It adds no production code: it measures the setting `surface_reach_in_cell_widths` (default 8, ported, never
measured; the fine band is twice the reach in smallest cells, so 16, 12 or 8 cells deep) and proposes a default to
the owner. Design 5.11 (1).

### Task 45a: Measure the surface reach 8, 6 and 4

**Amended 2026-11-15 (A68):** new task, numbered after Task 45.

**Kind:** integration (measurement; test only)

**Files:**
- Create: `native/fluidity/cell_resize/surface_reach_measure_test.cc` (`LABEL cpu`, plus `LABEL slow`: not in the
  gate's default test set), `docs/measurements/surface-reach.md` (the table and the decision)
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes the existing analytic scene tests that depend on the surface band, each run three times with
  `grid::CellSizeSettings::surface_reach_in_cell_widths` 8, 6 and 4, everything else unchanged: the standing wave
  (step's `BuildStandingWaveTankScene`, 10 periods: amplitude and period against linear theory), the dam-break front
  position (step's dam-break scene against its reference curve), the hydrostatic pressure of a column at rest
  (pressure's bound), the sloshing tank (step's sloshing scene: kinetic energy and free-surface height at the wall);
  one visual scene, the bathtub splash (I/O's bathtub scene, 2 x 1 x 0.5 m at 1 cm, rendered by the render-output
  path at frames 0, 1 s and 2 s).
- Records per reach: each analytic error and its test's tolerance, the planned memory from `step::BuildRunPlan`
  (bathtub at 1 cm: about 5.0, 4.3, 3.6 GB expected, `tradeoff-particles.md` 1.7, 6 interpolated), the mean step
  time over the run, and the cell and particle counts.

- [ ] **Step 1: Tests** (`LABEL cpu slow`): `SurfaceReach, AnalyticScenesAtReach8_6_4` (prints the table; fails only
  when a run fails to complete, never on an error value: a report, not a gate); `SurfaceReach, BathtubSplashFrames`
  (writes the frames for the owner).
- [ ] **Step 2: Decision rule** (write it in `docs/measurements/surface-reach.md` with the numbers): the proposed
  default is the smallest reach whose every analytic error stays within its test's tolerance **and** within 1.2 x the
  reach-8 error (1.2 a starting value), and whose bathtub frames the owner accepts beside the reach-8 frames;
  otherwise 8 stays. The default is changed only by the owner, in grid's `CellSizeSettings`, in its own commit.
- [ ] **Step 3:** gate (the slow tests excluded). **Commit.** Subject: `Measure the surface reach at 8, 6 and 4`.

---

# Coarser cells on calm surfaces (later)

**Amended 2026-11-15 (A61):** new group (`plan-decisions.md` item 2, "Later group"). **Amended 2026-11-15 (A68):**
second in the order, together with "Batches cut by columns or tiles (later)" below. **These tasks are excluded from
the module's done-criteria.** The first version keeps the surface, pockets, gas and solids at the smallest cells,
which design 5.3 rule 7 relies on (size changes only in bulk liquid).

### Task 46: Curvature term and coarser surface cells

**Amended 2026-11-15 (A61):** new task, numbered after the main sequence.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/grid/cell_size_rule.h` (grid's, a grid-plan amendment first), `cell_size_changes_steps.h`,
  `cell_size_changes_test.cc`; pressure's surface condition (a pressure-plan task first)

**Interfaces (decided in `plan-decisions.md`; details for the owner):**
- `w_curv = c_k / max(|∇²φ|, tiny)`, `c_k = 0.25` (`1/γ_φ` of Ando and Batty 2020: a sphere of radius `R` has
  `|∇²φ| = 2/R`, so `R/8` cells, about 16 across); `φ` a smoothed distance (not `distance_to_surface`, a union of
  particle balls); 7-point Laplacian on cells within 2 cells of the surface, clamped to `1 / w_min`; spread by a max
  over the 26 neighbours, 2 passes (order-independent); surface cells up to 1 or 2 levels coarser where `w_curv` and
  `w_shear` allow.
- Needs first: a T-junction-aware free-surface pressure condition (Ando and Batty 2020, Eq. 25) in pressure, and
  bump-free surface resampling of particles (Ando, Thürey, Wojtan 2013) in particles.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ComputeWantedCellWidth, SphereOfRadiusR_GetsREighth` (grid);
  `ResizeCells, CalmFlatSurface_CoarsensOneLevel`; `ResizeCells, SurfaceTJunction_NoPressureArtefact` (a flat pool at
  rest with coarse surface cells: largest speed below pressure's bound). **Step 2-4:** as the main tasks. **Commit.**
  Subject: `Let calm surfaces use coarser cells`.

---

# Batches cut by columns or tiles (later)

**Amended 2026-11-15 (A68):** new group, beside "Coarser cells on calm surfaces (later)". **Excluded from the
module's done-criteria.** Design 5.11 (2) holds the sketch. The first version keeps slab batches: a slab must hold
`T + 8 >= 9` layers of smallest-level blocks, each layer `(X / 4 w_min) x (Y / 4 w_min)` blocks at `b` (about 540 kB)
per loaded block, inside the room `R` (about 9.6 GB of a 13.05 GB budget), so the plan area is capped at
`a_max = R x 16 w_min² / (9 b)`: about 3.2 m² at 1 cm and 0.8 m² at 0.5 cm (Task 38 item 4a). Tasks, as a sketch list
only (each becomes a full task, with files, interfaces and tests, when the group is planned):

- Grid: a block-list form of `grid::BuildGridPiece` and of the stacked border search (`FindStackedBorder`), with
  lateral borders only (four sides of `kBorderLayers = 4` block columns, none above or below).
- Batches: the tile cut in `BuildBatchPlan` (tile side `s` blocks with `(s + 2 kBorderLayers)² x (blocks per
  column) x b <= R`), kept only when the slab cut does not fit; `SlabLimit` unchanged for slabs and a matching report
  for tiles; the per-try cut in `ApplyStagesInBatches` by tiles in ascending first-block order.
- Pressure, viscosity, gas: solve pieces as tiles with the same lateral borders (their plans).
- Tests: `ApplyStagesInBatches, TileCut_SameBytesForEveryCut` and `..._WithinStepToleranceOfOnePiece` (the 5.10 rule
  unchanged: block partials, tree sum in the whole grid's block order); `BuildBatchPlan, WidePool_TilesFit` (pool
  10 x 5 x 1.5 m at 3.5 cm planned batched); the copy cost of a tall narrow scene against slabs reported.
- I/O: the budget line drops the plan-area sentence when tiles are on (I/O plan).

---

# Deep liquid without particles (later; needs review)

**Amended 2026-11-15 (A68):** last of the later groups (moved after "Batches cut by columns or tiles (later)").

**Amended 2026-11-15 (A64):** new group. The first version holds particles in every liquid cell; this group holds
the grid-only region as a later optimisation. **These tasks are excluded from the module's done-criteria.** An
independent review of this design comes first: before any task here starts, a reviewer who did not write the design
reads cell resize design 1 Part A, 3.4 to 3.6, 5.5 and 5.6 (the passages marked "not in the first version (A64)")
and `literature-cell-size.md` 3 and 5(e) (Ferstl 2016: a particle/grid velocity blend "leads to inherently unstable
simulations"; Ando and Batty 2020: coarse semi-Lagrangian cells damp energy), and approves it or names the changes;
the owner decides. The group is then gated by the two tests of Task 32c: nothing here joins the first version's
history until both pass. Other plans' grid-only pieces (step callbacks and `GridFields` arrays,
`CommittedStepLimits::grid_only_speed`, the batched mirror of grid-only marks, transfer's `kFromGridOnlyRegion`,
pressure's "a grid-only cell counts as full", render output's grid-only surface inputs, the I/O and add-on fields
`particle_region_depth_in_cell_widths`, `grid_only_depth_in_cell_widths`) sit in a group of the same name in their
own plans, each pointing here as its owner. Order inside the group: 4b, 3, 10b, 11b, 15b, 16b, 17b, 18 to 22, 23b,
24b, 25, 26, 28b, 29b, 30c, 31b, 32b, 32c.

### Task 4b: The grid-only errors

**Amended 2026-11-15 (A64; reconcile item 31):** new task. Appends to `CellResizeError`, after `kOutOfMemory`
(earlier values keep their numbers): `kWrongMaterialSet` (10), `kGridOnlyCellDrained` (11), `kGridOnlyCellTooFull`
(12), with the 2026-11-14 comments of design 3.1.

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/cell_resize/cell_resize_error.h`, `cell_resize_message.cc`, `cell_resize_error_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ToName, EveryCellResizeError_IsLowerSnakeCaseWithoutK` extended to 13
  (`kGridOnlyCellDrained` -> `"grid_only_cell_drained"`); `CellResizeError, GridOnlyErrorsAfterOutOfMemory` (values
  10, 11, 12).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** ("A grid-only cell would empty in this step. A shorter step is
  tried."). **Step 4:** gate. **Commit.** Subject: `Name the failures of the grid-only region`.

### Task 3: Method paper, the two regions and the region transfer

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/cell_resize.typ`, `docs/glossary.md`; Create: `figures/regions.typ`

**Interfaces:**
- Produces labels `<sec-regions>`, `<sec-region-transfer>`, `<fig-regions>`, `<proc-apply-region-transfer>`.

- [ ] **Step 1: Text,** outline items 6 to 8: every amount is held once; two regions and one ledger, one material per
  record (5.5, 5.6); the grid-only region inside a step (the four face cases, the carried velocity followed back,
  the move of amounts by shares, the filled share); region transfer by particles (rules 1 to 5, why it replaces
  the flow exchange of `main`). Cite [17] Ferstl 2016. Glossary rows: `particle_region`, `grid_only_region`,
  `carried_velocity`, `particle_share`, `record`.
- [ ] **Step 2: Figure** `<fig-regions>`: particle region, overlap layers, grid-only region, the two depths, the
  four transfers as arrows.
- [ ] **Step 3: Pseudocode** `<proc-apply-region-transfer>`: wish per cell, then rules 1 to 5 numbered as design 5.6.
- [ ] **Step 4:** typst, look, gate. **Commit.** Subject: `Describe the particle and grid-only regions`.

### Task 10b: Region settings and grid-only rows in `CellResize`

**Amended 2026-11-15 (A64):** new task; the parts of Task 10 that exist only for the grid-only region.

**Kind:** mechanical

**Files:**
- Modify: `native/fluidity/cell_resize/resize_cells.h`, `cell_resize.cc`, `cell_resize_access.h`, `cell_resize_test.cc`

**Interfaces:**
- Produces design 3.4 (marked "not in the first version (A64)"): `RegionSettings` (defaults `6`, `7`,
  `overlap_layer_count = 2`, `particles_per_refilled_cell = 8`), `kMaxFilledShare = 2.0`, `RegionTransferLedger`,
  `CellResizeSettings::regions`, `ResizeCellsReport::ledger`, `ReadLargestGridOnlySpeed`. Settings checks
  (`kBadInput`): `grid_only_depth_in_cell_widths > particle_region_depth_in_cell_widths > 0`;
  `overlap_layer_count` 0 to 254; `particles_per_refilled_cell` 1 to 9. Rows (all `kWholeRun`, pieces of
  `run_memory`): region wish (`std::uint8_t`, cell place); removed amounts (`double` x (3 + ingredients), cell
  place); refill list (`particles::NewParticles` arrays, added particle); amounts at step start (`double` x (3 +
  ingredients), cell place); block partial sums widened to `double` x (8 + ingredients); `OrderBuffers
  order_buffers` (`BuildOrderBuffers(run_on, 64 * limits.max_cell_blocks, run_memory)`, the refill ids of Tasks 25
  and 28b) with its two entries.

- [ ] **Step 1: Failing tests:** `BuildCellResize, TenParticlesPerRefilledCell_IsBadInput`; `BuildCellResize,
  GridOnlyNotDeeperThanParticleRegion_IsBadInput`; `ReadLargestGridOnlySpeed, BeforeFirstStep_IsZero`;
  `BuildCellResize, IntoRunMemory_HandsOutExactlyTheEstimate` extended with the rows above.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Hold the settings and buffers of the grid-only region`.

### Task 11b: The region types

**Amended 2026-11-15 (A64):** new task; the types Task 11 dropped.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/cell_resize/grid_only_regions.h`, `grid_only_regions_test.cc`

**Interfaces:**
- Produces design 3.5 `RegionKind`, `GridOnlyAmounts` (marked "not in the first version (A64)"). **Plan decision:**
  `struct GridOnlyAmountsToChange` has the same seven members as `GridOnlyAmounts`, non-const pointers, and
  `ingredient_count`. The seven arrays are the step's later-group `GridFields` arrays.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `GridOnlyAmounts, DefaultIsOneIngredient`; `RegionKind,
  TwoValues`. **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Name the two regions and the grid-only record`.

### Task 15b: The grid-only amounts on a resize

**Amended 2026-11-15 (A64):** new task; moved from Task 15.

**Kind:** numerics

**Files:**
- Modify: `remap_grid_values.h`, `remap_grid_values_cpu.cc`, `remap_grid_values_test.cc`

**Interfaces:**
- Produces (plan decision, design 3.6):

  ```cpp
  // m3, kg: the grid-only mass, rest_volume and current_volume remapped together with one limiter per split cell,
  // the smallest of the three ComputeSplitLimiter values, so a split keeps density and volume ratio.
  std::expected<void, CellResizeFailure> ApplyResizeToCellAmounts(
      const ResizeMap& map, std::span<const grid::CellValues<double>* const> old_amounts,
      const grid::CellValues<double>* old_weight, std::span<grid::CellValues<double>* const> new_amounts);
  ```
  The transaction passes the old `grid_only_current_volume` as `old_weight`, so only same-size grid-only neighbours
  count.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ApplyResizeToCellAmounts, ProportionalAmounts_KeepDensity` (mass =
  1000 x current volume: each small cell's ratio 1000 within 1e-14). **Step 2: Run** (fail). **Step 3: Implement.**
  **Step 4:** gate. **Commit.** Subject: `Carry the grid-only amounts onto resized cells`.

### Task 16b: Ingredient volumes of the record on a resize

**Amended 2026-11-15 (A64):** new task; moved from Task 16.

**Kind:** numerics

**Files:**
- Modify: `remap_grid_values.h`, `remap_grid_values_cpu.cc`, `remap_grid_values_test.cc`

**Interfaces:**
- Produces design 3.6 `ApplyResizeToIngredientVolumes` (marked "not in the first version (A64)").

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ApplyResizeToIngredientVolumes, SplitThenMerge_FractionsKept` (design 7:
  3 ingredients; small cells' fractions equal the big cell's; merge returns the start volumes within 2 rounding
  steps). **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Carry the record's ingredient volumes onto resized cells`.

### Task 17b: The grid-only remap on the GPU

**Amended 2026-11-15 (A64):** new task; moved from Task 17.

**Kind:** device

**Files:**
- Modify: `remap_grid_values_steps.h` (`ComputeRemappedCellAmounts`, one limiter for the three amounts;
  `ComputeRemappedIngredientVolumes`), `remap_grid_values_gpu.cu`, `remap_grid_values_cpu.cc`,
  `remap_grid_values_steps_test.cc`, `remap_grid_values_gpu_test.cc`

- [ ] **Step 1: Tests.** `LABEL cpu`: `RemapGridValuesSteps, EveryThreadOrder_EqualsCpu` extended with Tasks 15b, 16b.
  `LABEL gpu`: `ApplyResizeToCellAmounts, OnGpu_EqualsCpu`, `ApplyResizeToIngredientVolumes, OnGpu_EqualsCpu`
  (design 6 tolerance; outputs to `WriteResultDigest`). **Step 2: Implement.** **Step 3:** gate. **Commit.**
  Subject: `Remap the grid-only record on the GPU`. Body: `CUDA: compiled, not run`.

### Task 18: The carried velocity and the region formulas

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** shared-header tests `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/cell_resize/carried_velocity_math.h`, `region_wish_math.h` (shared),
  `carried_velocity_math_test.cc`, `region_wish_math_test.cc`

**Interfaces:**
- Produces design 3.5 `ComputeCarriedVelocity` (design 5.5: `start = face centre - step_duration * velocity at
  the face centre`, each axis read with `grid::ComputeNearbyFaceWeights`; the face's own axis read at `start` with
  the same weights, all faces kept).
- Produces (plan decision, `region_wish_math.h`):

  ```cpp
  struct RegionWishFacts {               // One cell of the resized grid (design 5.6).
    RegionKind current = RegionKind::kParticleRegion;
    double cell_width = 0, depth_below_surface = 0;   // m.
    bool is_full_open_liquid = false;    // filled_volume == open_volume == cell volume.
    bool particles_are_one_liquid = false;   // Every particle of one kLiquid material (true when none).
    bool matches_record_material = true;     // That material is the record's (true without a record).
    bool has_plate_side_particle = false;
    bool touches_air_pocket_gas_or_solid = false;
    bool touches_other_material = false;     // Particles or a record of another material in a touching cell.
  };
  FLUIDITY_CPU_AND_GPU RegionKind ComputeRegionWish(const RegionWishFacts& facts,
                                                    double particle_region_depth_in_cell_widths,
                                                    double grid_only_depth_in_cell_widths);
  // clamp((mean_layer - 0.5) / overlap_layer_count, 0, 1); 1 when overlap_layer_count is 0.
  FLUIDITY_CPU_AND_GPU double ComputeParticleShare(double mean_layer, int overlap_layer_count);
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ComputeCarriedVelocity, RigidTranslation_SameField`
  (design 7: uniform `{0.3, -0.2, 0.1}` m/s on a mixed-size grid, 1e-13); `ComputeCarriedVelocity, Rest_ExactlyCommitted`
  (zero field: exactly 0); `ComputeCarriedVelocity, ShearFollowedBack` (`u_x = 2 z`, `u_z = 0.5`: the value at the
  start point, 1e-13); `ComputeRegionWish, DepthBands` (6.99 w particle stays particle, grid-only stays grid-only; 7 w
  full one liquid -> grid-only; 5.99 w -> particle); `ComputeRegionWish, AnyConditionFails_IsParticleRegion` (one
  case per bool); `ComputeParticleShare, Layers` (mean 1, count 2 -> 0.25; 2.5 -> 1; 0.5 -> 0).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Follow the grid-only velocity back one step and decide each cell's region`.

### Task 19: `ApplyGridOnlyRegionToGrid` on the CPU

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** `grid_only_to_grid_cpu.cc` holds the public function (checks, dispatch, CPU loop); the per-face and per-cell work goes in `grid_only_to_grid_steps.h` for Task 20; tests link the helper libraries.

**Files:**
- Create: `native/fluidity/cell_resize/grid_only_to_grid_cpu.cc`, `grid_only_to_grid_steps.h` (shared:
  `ComputeGridOnlyFaceValues` per face place, `ComputeGridOnlyCellValues` per cell place), `grid_only_to_grid_test.cc`
- Modify: `grid_only_regions.h`, `CMakeLists.txt`

**Interfaces:**
- Produces design 3.5 `GridOnlyToGridInputs`, `ApplyGridOnlyRegionToGrid` (design 5.5 cases 1 to 4).
- **Plan decisions:** a face written by the region gets `velocity = carried velocity`, `momentum = mass *
  velocity`, `weight_sum` and `mass` as case 1, `value_kind = kFromGridOnlyRegion`; in case 2 the particle mass and
  momentum already summed by transfer are kept and the region's mass and `mass * carried velocity` added, the
  velocity being the carried one; in case 3 only `velocity` changes. `committed_velocity` is
  `StepTry::fields().velocity_after_pressure` read in stage 4, which holds the committed values carried onto the try
  grid by `FacePieceId` (step design 5.3 step 3).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `LINKS fluidity_transfer_test_scenes fluidity_grid_test_support
  fluidity_base_test_support`; transfer's builders on grids from `grid_test_support`):

  ```cpp
  TEST(ApplyGridOnlyRegionToGrid, BothCellsGridOnly_CarriedVelocityAndMass)  // 4 x 4 x 4 grid-only full cells, record
      // mass 1000 V, uniform committed velocity: each face velocity == committed, mass == 1000 * area * w,
      // weight_sum == 8 * area * w / V, kind kFromGridOnlyRegion
  TEST(ApplyGridOnlyRegionToGrid, GridOnlyBesideParticles_MassesAdded)
  TEST(ApplyGridOnlyRegionToGrid, OverlapLayers_BlendByParticleShare)
  TEST(ApplyGridOnlyRegionToGrid, TooFewParticlesInOverlap_TakesCarriedVelocity)
  TEST(ApplyGridOnlyRegionToGrid, CellsFromParticles_RecordAddedToParticles)  // a drifted-in particle counted once
  TEST(ApplyGridOnlyRegionToGrid, SetWithoutLiquid_LeftUntouched)  // snow mask: bytes of both outputs unchanged
  TEST(ApplyGridOnlyRegionToGrid, SwappedSetArray_IsWrongMaterialSet)
  TEST(ApplyGridOnlyRegionToGrid, StaleArray_IsWrongGridVersion)       // block_count differs, same version
  TEST(ApplyGridOnlyRegionToGrid, NaNRecord_IsNotFinite)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (one iteration per face place, then per cell place). **Step 4:**
  gate. **Commit.** Subject: `Add the grid-only region to what particles gave the grid`.

### Task 20: `ApplyGridOnlyRegionToGrid` on the GPU

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `grid_only_to_grid_gpu.h` returning `RunOnCallStatus`; GPU functions as thin loops over
the Task 19 steps functions; a `cpu` steps test; discrete equal, values within design 6; digests.

**Files:**
- Create: `grid_only_to_grid_gpu.cu`, `grid_only_to_grid_gpu.h` (private), `grid_only_to_grid_steps_test.cc`,
  `grid_only_to_grid_gpu_test.cc`
- Modify: `grid_only_to_grid_cpu.cc` (the `kGpu` branch under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

- [ ] **Step 1: Tests.** `LABEL cpu` (`grid_only_to_grid_steps_test.cc`, the main evidence until the hardware run):
  `GridOnlyToGridSteps, EveryThreadOrder_EqualsCpu` (Task 19 cases: both steps functions for every thread block and
  thread, launch order and reverse: bytes equal the CPU function's). `LABEL gpu`: `ApplyGridOnlyRegionToGrid,
  OnGpu_EqualsCpu` (Task 19 cases: `value_kind` equal, values within design 6; outputs to `WriteResultDigest`),
  `...OnGpuTwice_GivesIdenticalBytes`. `LABEL no_gpu`: `ApplyGridOnlyRegionToGrid, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** gathers per face and per cell, launched with `ApplyGpuFunction`; the status maps
  through `ToCellResizeFailure(ToBufferError(status))`. **Step 3:** gate. **Commit.** Subject:
  `Add the grid-only region to the grid on the GPU`. Body: `CUDA: compiled, not run`.

### Task 21: `MoveGridOnlyLiquid` on the CPU

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the public function in `move_grid_only_liquid_cpu.cc` (checks, dispatch, CPU loop); the shared test is `LABEL cpu`; the per-cell move and block partials are shared functions Task 22 reuses.

**Files:**
- Create: `native/fluidity/cell_resize/grid_only_move_math.h` (shared), `move_grid_only_liquid_cpu.cc`,
  `move_grid_only_liquid_test.cc`, `grid_only_move_math_test.cc`
- Modify: `grid_only_regions.h`

**Interfaces:**
- Produces design 3.5 `MoveGridOnlyLiquidInputs`, `MoveGridOnlyLiquidResult`, `MoveGridOnlyLiquid`. Ported from
  `native/src/adaptive_liquid_core_transport_3d.cc:130-310`.
- Produces (plan decision, shared):

  ```cpp
  // One face piece seen from the receiving cell: share of the giver's start amounts that crosses it.
  struct PieceExchange { double share = 0; double giver_amount_at_start = 0; std::int8_t direction = 0; };  // -1 out, +1 in
  // m3/s * s / m3: |velocity| * open_area * step_duration / giver's current_volume at step start.
  FLUIDITY_CPU_AND_GPU double ComputeOutgoingShare(double velocity, double open_area, double step_duration,
                                                   double giver_current_volume_at_start);
  // amount_at_start, then in ascending face place index: += direction * share * giver_amount_at_start.
  FLUIDITY_CPU_AND_GPU double ComputeAmountAfterMove(double amount_at_start, const PieceExchange* pieces,
                                                     int piece_count);
  ```
  Only pieces between two grid-only cells move anything. Per giving cell, its outgoing shares summed in ascending
  face place index above 1: `kGridOnlyCellDrained` (`magnitude` = that sum, `limit` = 1). After the move,
  `current_volume / open_volume > kMaxFilledShare`: `kGridOnlyCellTooFull` (`magnitude` = the share, `limit` 2).

- [ ] **Step 1: Failing tests** (all `LABEL cpu`): shared `ComputeAmountAfterMove, OutThenInInFaceOrder`; and:

  ```cpp
  TEST(MoveGridOnlyLiquid, UniformFlowThroughRow_TotalsKeptVolumesUnchanged)  // design 7 "Grid-only move": 8 grid-only
      // cells along x, u on the 7 inner faces, 0 at both walls: totals within kAmountTolerance; cells 1..6 current
      // volume unchanged within kAmountTolerance; cell 0 loses and cell 7 gains u * A * dt
  TEST(MoveGridOnlyLiquid, FlowThatEmptiesACell_IsGridOnlyCellDrained)       // the same row, dt so that u A dt > V
  TEST(MoveGridOnlyLiquid, AboveMaxFilledShare_IsGridOnlyCellTooFull)
  TEST(MoveGridOnlyLiquid, RegionFace_MovesNothing)
  TEST(MoveGridOnlyLiquid, Mixture_MovesAsOneLiquid)        // 3 ingredients: ratios kept exactly in every cell
  TEST(MoveGridOnlyLiquid, MovedVolume_TreeSumOfBlockPartials)
  TEST(MoveGridOnlyLiquid, LargestSpeedInCellWidths)        // u / w
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (one iteration per receiving cell). **Step 4:** gate. **Commit.**
  Subject: `Move liquid between grid-only cells with the solved velocity`.

### Task 22: `MoveGridOnlyLiquid` on the GPU

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `move_grid_only_liquid_gpu.h` returning `RunOnCallStatus`; per-cell and per-block steps
functions with a `cpu` test; `moved_volume` and the largest speed from the sums with `CellResize`'s
`sum_buffers`; digests.

**Amended 2026-11-14 (A31):** the sums have one form, which takes `cell_resize.sum_buffers`; "scratch" wording removed.

**Files:**
- Create: `move_grid_only_liquid_gpu.cu`, `move_grid_only_liquid_gpu.h` (private), `move_grid_only_liquid_steps.h`
  (shared: `ComputeMovedCellAmounts` per receiving cell, with `ComputeOutgoingShare` and `ComputeAmountAfterMove` of
  `grid_only_move_math.h`; `ComputeMoveBlockPartials` per cell block, with `math::TreeSumOf` over its cell places),
  `move_grid_only_liquid_steps_test.cc`, `move_grid_only_liquid_gpu_test.cc`
- Modify: `move_grid_only_liquid_cpu.cc` (the `kGpu` branch under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

- [ ] **Step 1: Tests.** `LABEL cpu` (`move_grid_only_liquid_steps_test.cc`, the main evidence until the hardware
  run): `MoveGridOnlyLiquidSteps, EveryThreadOrder_EqualsCpu` (Task 21 cases: both functions for every thread block
  and thread, launch order and reverse; the integer minimum of failing cells taken in that order: bytes and failure
  cell equal the CPU function's). `LABEL gpu`: `MoveGridOnlyLiquid, OnGpu_EqualsCpu` (Task 21 cases; the failure
  cell equal; amounts and `moved_volume` within design 6; outputs to `WriteResultDigest`),
  `...OnGpuTwice_GivesIdenticalBytes`. `LABEL no_gpu`: on the GPU without a GPU.
- [ ] **Step 2: Implement:** one thread per receiving cell; the first failing cell by an integer `atomicMin` over
  cell places; `moved_volume` from block partials and `ComputeTreeSum(partials, block_count,
  cell_resize.sum_buffers)`, the largest speed with `ComputeLargestMagnitude(..., cell_resize.sum_buffers)`; no
  allocation per call. **Step 3:** gate. **Commit.** Subject:
  `Move grid-only liquid on the GPU`. Body: `CUDA: compiled, not run`.

### Task 23b: Liquid totals with the records

**Amended 2026-11-15 (A64):** new task. The first version's `ComputeLiquidTotals` sums particles only (Task 23);
here the records join it.

**Kind:** numerics

**Files:**
- Modify: `liquid_totals.h`, `liquid_totals_cpu.cc`, `liquid_totals_steps.h`, `liquid_totals_gpu.cu`, `liquid_totals_test.cc`

**Interfaces:**
- Changes `ComputeLiquidTotals` to take `const GridOnlyAmounts& amounts` after `particles_try` (design 3.5's later
  form: one form, A31; the callers named in Task 23 and render output's later group pass the amounts); the block
  partial adds the cell records before the particles, in cell place order.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ComputeLiquidTotals, ParticlesPlusRecords_TreeSumOfBlocks` (bytes against
  a test-side block-partial sum). **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.**
  Subject: `Sum the liquid held by particles and grid-only cells`.

### Task 24b: The resize with grid-only cells

**Amended 2026-11-15 (A64):** new task; the grid-only parts of Task 24.

**Kind:** integration

**Files:**
- Modify: `resize_cells_cpu.cc`, `resize_cells_test.cc`, `cell_resize_test_support.h/.cc`

**Interfaces:**
- Step 3 "nothing to do" also requires no region change, no particle in a grid-only cell and no cell eighth needing
  a refill. Step 5 also remaps the three grid-only amounts (`ApplyResizeToCellAmounts`, weight = old
  `grid_only_current_volume`), `grid_only_ingredient_volume` (`ApplyResizeToIngredientVolumes`), region bytes and
  layers (`kLabel`). Step 8's totals include the records.
- Test support (plan decision): `step::SimulationSnapshot BuildGridOnlySnapshot(const step::SimulationSnapshot&
  committed, double below_height)`: every liquid cell whose centre is below `below_height` becomes grid-only with a
  record equal to its particles' sums, those particles removed.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ResizeCells, RecordsRemapped_TotalsKept` (`BuildGridOnlySnapshot(below
  0.5 m)` rebuilt with `BuildSimulationFromSnapshot`: totals of the step-8 list within `kAmountTolerance`).
  **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Carry grid-only cells through a resize`.

### Task 25: Region transfer, rules 1 to 3

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Amended 2026-11-14 (A31):** `ComputeExclusivePrefixSum` named without "scratch form" (one form, A31).

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the refill ids' integer prefix sum is `ComputeExclusivePrefixSum(counts, cell_count, first_ids, cell_resize.order_buffers)`; the public function lives in `region_transfer_cpu.cc` (checks, dispatch, CPU loop).

**Files:**
- Create: `native/fluidity/cell_resize/region_transfer_cpu.cc`, `region_transfer_test.cc`
- Modify: `grid_only_regions.h`, `resize_cells_cpu.cc` (step 6 calls it)

**Interfaces:**
- Produces design 3.5 `ApplyRegionTransfer` (rules 1 to 3 of design 5.6; ledger counts and both
  `ParticleAmounts`).
- **Plan decisions:** it reads the depth and the `ResizeMap` the transaction left in `cell_resize`: a new cell's depth,
  fullness and content come from its source (same cell: copied; small cell of a split: the big cell's; merged cell:
  the smallest depth of the 8, full only if all 8 are). Cells in cell place order of the resized grid, particles in
  id order; removed particles' amounts added to the record in that order. A refilled cell's velocity is the
  area-weighted mean of its face-piece velocities per axis (`fields_to_fill().velocity_after_pressure`), its
  `fitted_velocity_change` diagonal = (high mean - low mean) / width; refill ids come from an integer prefix sum of
  per-cell counts (amended: `ComputeExclusivePrefixSum` with `cell_resize.order_buffers`, its
  `BufferError` mapped by `ToCellResizeFailure`). `kTooManyParticles` when the refill passes a limit.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, through `ResizeCells` on simulations of Task 1's scenes):

  ```cpp
  TEST(ApplyRegionTransfer, DeepCellsBecomeGridOnly_ParticlesIntoRecords)  // BuildDeepTankScene, first resize: every
      // full cell at depth >= 7 widths is grid-only; record == sum of its particles in id order (bytes); ledger
      // into_grid_only_region == the removed sums; totals within kAmountTolerance
  TEST(ApplyRegionTransfer, OwnParticleInGridOnlyCell_TakenIn)
  TEST(ApplyRegionTransfer, OwnAndForeignParticleInOneCell_CellReturnsToParticles)  // the former Review Focus 4
  TEST(ApplyRegionTransfer, ShallowGridOnlyCell_RefilledWithEightParticles)  // density, volume ratio, fractions of the
      // record; record exactly 0; AddReason::kRefillRegion; ledger into_particle_region == the old record
  TEST(ApplyRegionTransfer, CellTouchingSolidOrGas_StaysParticleRegion)
  TEST(ApplyRegionTransfer, GasMarkersNeverTakenIn)           // BuildSmokeOverDeepWaterScene
  TEST(ApplyRegionTransfer, RefillPastParticleLimit_IsTooManyParticles)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step N:` from `<proc-apply-region-transfer>`. **Step 4:**
  gate. **Commit.** Subject: `Move liquid between particles and grid-only cells at a resize`.

### Task 26: Region transfer, rules 4 and 5, the ledger, every failure

**Amended 2026-11-15 (A64):** moved here unchanged from the main sequence; read `grid_only_regions.h` as `grid_only_regions.h`
(Task 11b) and every `GridOnlyAmounts` use as that task gives it. Not part of the module's done-criteria.


**Amended 2026-11-14 (A31):** the ledger sum is `ComputeTreeSum` with `cell_resize.sum_buffers` (one form).

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** on a GPU run the ledger's block partials are summed with `ComputeTreeSum` and `cell_resize.sum_buffers` (its one form, A31); `FailureAtEachStep_ChangesNothing` covers check failures only.

**Files:**
- Modify: `region_transfer_cpu.cc`, `region_transfer_test.cc`, `resize_cells_cpu.cc`, `resize_cells_test.cc`

**Interfaces:**
- Produces design 5.6 rules 4 (refill at the region border) and 5 (`layers_from_grid_only_region` by the repeated
  minimum over face-touching cells, capped at `overlap_layer_count + 1`), `momentum_not_carried` (plan decision:
  removed particles' momentum minus, per cell turned grid-only, the mass taken in times the cell's area-weighted
  face-mean velocity), the ledger totals as block partials then `math::TreeSum`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ApplyRegionTransfer, ExcessRefillsEmptyEighthsAtBorder)   // excess 2.5 w^3/8, 4 empty eighths of P touching
      // the piece: 2 particles at eighth centres in ascending face place, then eighth order; each exactly w^3/8
  TEST(ApplyRegionTransfer, LayersFromGridOnlyRegion_RepeatedMinimum)  // 0 inside, 1, 2, then 3 for all farther
  TEST(ApplyRegionTransfer, MomentumNotCarried)          // face velocity equal to the particles' -> 1e-12 of |sum m v|;
                                                         // faces at rest -> sum m v
  TEST(ApplyRegionTransfer, LedgerTotals_TreeSumOfBlocks)
  ```
  And add to `ResizeCells, FailureAtEachStep_ChangesNothing` (Task 27a) the region sub-cases: step 5, a NaN in one
  record of a snapshot rebuilt with `BuildSimulationFromSnapshot` (`kNotFinite`); step 6, particle limit hit by a
  refill (`kTooManyParticles`).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Refill the region border and keep the ledger of a resize`.

### Task 28b: Region transfer on the GPU

**Amended 2026-11-15 (A64):** new task; the region-transfer half of Task 28, as written there on 2026-11-14.

**Kind:** device

**Files:**
- Create: `region_transfer_gpu.cu`, `region_transfer_gpu.h` (private), `region_transfer_steps.h` (shared:
  `ComputeCellRegionWish` per cell, `ComputeTakenInAmounts` per cell (its particles in id order), `ComputeRefillCount`
  per cell, `WriteRefillParticles` per cell from its first id, `ComputeLedgerBlockPartials` per cell block),
  `region_transfer_steps_test.cc`, `region_transfer_gpu_test.cc`; extend `cell_resize_no_gpu_test.cc`
- Modify: `region_transfer_cpu.cc` (the `kGpu` branch under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

- [ ] **Step 1: Tests.** `LABEL cpu`: `RegionTransferSteps, EveryThreadOrder_EqualsCpu` (Tasks 25, 26 cases, with the
  CPU `ComputeExclusivePrefixSum` between the count and write phases: regions, counts, ids and amounts' bytes equal
  the CPU function's). `LABEL gpu`: `ApplyRegionTransfer, OnGpu_EqualsCpu` (regions, counts, ids equal; amounts within
  design 6); `ResizeCells, OnGpu_EqualsCpu` extended with Task 24b's case. `LABEL no_gpu`: `ApplyRegionTransfer,
  OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** design 6: the refill list from `ComputeExclusivePrefixSum(counts, cell_count, first_ids,
  cell_resize.order_buffers)`; the ledger's totals by `ComputeTreeSum(partials, block_count,
  cell_resize.sum_buffers)`; no allocation per call; the status maps through `ToCellResizeFailure(ToBufferError(status))`.
  **Step 3:** gate. **Commit.** Subject: `Run the region transfer on the GPU`. Body: `CUDA: compiled, not run`.

### Task 29b: The grid-only step callbacks

**Amended 2026-11-15 (A64):** new task; the grid-only half of Task 29 as written on 2026-11-14 (the step's later
group adds the callbacks `apply_grid_only_region`, `move_grid_only_liquid`, `read_largest_grid_only_speed` and
`CommittedStepLimits::grid_only_speed`).

**Kind:** integration

**Files:**
- Modify: `native/fluidity/cell_resize/step_callbacks.cc`, `step_callbacks_test.cc`, `resize_cells.h`,
  `cell_resize_test_support.h/.cc`

**Interfaces:**
- Produces design 3.4 `ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep` (marked "not in the first version
  (A64)"). `MoveGridOnlyLiquidInStep` first copies the try's three amounts and ingredient volumes into the `amounts
  at step start` row, then calls `MoveGridOnlyLiquid` writing the try copy; it stores the try's
  `largest_speed_in_cell_widths`, which `ReadLargestGridOnlySpeed` returns (the step reads it only in `CommitStep`).
  Mapping: `kNotFinite` -> `kValueNotFinite`; `kGridOnlyCellDrained`, `kGridOnlyCellTooFull` ->
  `kParticleTravelledTooFar` with magnitude and limit as filled shares; others -> `kStageFailed`; context `module =
  Module::kCellResize`. `BuildCellResizeCallbacks` sets the three more pairs. Stage 16's `run_whole_grid` part
  (Task 35) gets the move again; the batches' `kStagePlaces` runs it right after pass 2.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ApplyGridOnlyRegionInStep, OnlyTheLiquidSetReceives`
  (`BuildTwoLiquidsDeepScene`: snow set's arrays bytes unchanged); `ApplyGridOnlyRegionInStep, NaN_IsValueNotFinite`;
  `MoveGridOnlyLiquidInStep, DrainedCell_IsShortenStep` (`kParticleTravelledTooFar`, kind `kShortenStep`; the half
  step commits); `MoveGridOnlyLiquidInStep, SpeedStoredAtCommit` (`CommittedStepLimits::grid_only_speed ==
  ReadLargestGridOnlySpeed`); `Batches, DeepTankWithGridOnlyRegion_WithinStepToleranceOfOnePiece` (Task 42's test
  with the region, `ApplyGridOnlyRegionInStep` per batch, the move on the whole grid).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Connect the grid-only region to the step's callbacks`.

### Task 30c: Scene tests, region cycle and border refill

**Amended 2026-11-15 (A64):** moved from Task 30 unchanged.

**Kind:** integration

- [ ] **Step 1: Tests** (`LABEL cpu`, in `cell_resize_scene_test.cc`):

  ```cpp
  TEST(CellResizeScene, RegionCycle_TotalsKeptEveryResize)  // design 7 "Region cycle": BuildDeepTankScene; 100 times:
      // one TryStep, then ResizeCells directly with a CellResize of depths 6/7, then one of 14/15 (the surface moving
      // through the two depths, relative to the cells): mass, rest volume, current volume, each ingredient volume of
      // particles plus records within kAmountTolerance per resize; no particle in a grid-only cell after a resize
  TEST(CellResizeScene, BorderRefill_EveryBorderEighthHoldsAParticle)  // design 7: BuildRisingFloorScene, 200 steps:
      // after each resize every cell eighth of a particle-region cell touching a grid-only cell holds a particle;
      // excess below w^3/8 per face quarter; totals within kAmountTolerance
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Test the two regions through whole steps`.

### Task 31b: Scene tests, absorb and foreign particles

**Amended 2026-11-15 (A64):** moved from Task 31; `TwoLiquidsMaterialSets` and `SmokeOverMovingWater` gain their
record assertions of 2026-11-14 (records reach only the liquid set's arrays, no record of two materials, cells along
the contact stay particle region; no grid-only cell touches a gas cell, no marker removed `kIntoGridOnlyRegion`).

**Kind:** integration

- [ ] **Step 1: Tests** (`LABEL cpu`, in `cell_resize_scene_test.cc`):

  ```cpp
  TEST(CellResizeScene, Absorb_ParticleTakenIntoRecord)   // design 7: BuildDeepTankScene after its first resize, a
      // water particle added by a source aimed into a grid-only cell: removed at the next resize; the record grew by
      // its amounts within 1 rounding step
  TEST(CellResizeScene, ForeignParticle_CellReturnsToParticles)  // design 7: BuildTwoLiquidsDeepScene: a snow particle,
      // then an oil particle sent into a water grid-only cell: the cell is particle region after the next resize; no
      // record holds two materials; each material's totals within kAmountTolerance
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Test records against other materials and gas`.

### Task 32b: Scene test, the full-particle comparison

**Amended 2026-11-15 (A64):** moved from Task 32 unchanged; Task 32c's standing-wave test is its stricter form.

**Kind:** integration

- [ ] **Step 1: Tests** (`LABEL cpu`, in `cell_resize_scene_test.cc`):

  ```cpp
  TEST(CellResizeScene, StandingWave_GridOnlyRegionMatchesParticles)  // design 7 "Full-particle comparison":
      // BuildStandingWaveTankScene with the region (6/7) and without (depths 1e9/1e9 + 1), same cell settings, 2 s:
      // surface height at x = 0.0125 within 0.025 m at every 0.1 s; kinetic energy (particles, plus per grid-only cell
      // 0.5 mass |area-weighted face-mean velocity|^2) within 5 % of the largest value of the run without
      // the region
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Compare the grid-only region with particles`.

### Task 32c: The two gating tests of the grid-only region

**Amended 2026-11-15 (A64):** new task; `plan-decisions.md` item 1 "Where it goes". The group joins the first version
only when both pass.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/step_test_scenes.h/.cc` (one scene), `cell_resize_scene_test.cc`

**Interfaces:**
- Produces (test-only, plan decision): `TestScene BuildHalfGridOnlyWaveTankScene();` closed box 2.0 x 0.1 x 0.8 m,
  `w_min` 0.0125 m, 4 levels (0.1 .. 0.0125 m), water 0.5 m deep with surface `0.5 + 0.01 cos(2 π x / 1.0)` (a
  standing wave of wavelength 1 m, amplitude 0.01 m), at rest; with the region settings that make the bottom 0.25 m
  grid-only (`particle_region_depth_in_cell_widths` and `grid_only_depth_in_cell_widths` chosen at 0.25 m / 0.1 m =
  2.5 and 3.5 biggest widths, plan decision). Period from linear theory `T = 2π / sqrt(g k tanh(k h))` with `k = 2π`,
  `h = 0.5`: `sqrt(9.81 x 6.2832 x tanh(3.1416))` = 7.8363 1/s, so `T` = 0.80180 s; 10 periods = 8.0180 s.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(GridOnlyRegion, StandingWave_KeepsAmplitude)  // the scene above with the region and an all-particle run (depths
      // 1e9 / 1e9 + 1), 10 periods: the amplitude at x = 0 (mean surface height of the first column, peak over the
      // last period) with the region within 5 % of the all-particle run's (5 % is a starting value, owner question 3)
  TEST(GridOnlyRegion, ParticleBandEdge_StaysStable)  // BuildDeepTankScene with the region (6/7) and a paddle
      // sloshing it (BuildWaveScene's paddle, stroke 0.02 m, period 1 s) for 20 s: no step rejected for travel at the
      // band edge (no kParticleTravelledTooFar from a cell with layers_from_grid_only_region 1..overlap_layer_count),
      // kinetic energy never above its start value plus the work of gravity and the paddle, no particle in a
      // grid-only cell after any resize
  ```
- [ ] **Step 2: Run.** A failure is reported to the owner with both numbers; the design is changed, not the
  tolerance. **Step 3:** gate. **Commit.** Subject: `Gate the grid-only region on a standing wave and a sloshing
  tank`.
