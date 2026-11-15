# Rebuild Cell Resize and Batches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the last simulation module of the rebuild. Part A, cell resize: after each committed step, choose
which cells split or merge, rebuild the grid in one transaction, remap every grid-carried value without losing
any, and move liquid between the particle region and the grid-only region. Part B, batches: run a scene whose
particles do not fit the GPU in batches of cell blocks, with the solves either on the whole grid or, when their
working buffers do not fit either (owner answer A15), batch by batch through pressure's `SolveBatches`; the planner
chooses one piece, batches, or batches with solves in pieces and shows the planned memory and copies. A batched
run agrees with the one-piece run within the solver's tolerance, and gives the same bytes for every cut and on
every repeat. CPU first; device code compiled, not run.

**Architecture:** Two folders. `native/fluidity/cell_resize/` (namespace `fluidity::cell_resize`): shared formula
headers (`cell_size_rule.h`, `remap_math.h`, `carried_velocity_math.h`, `grid_only_move_math.h`,
`region_wish_math.h`, `cell_resize_error.h`), CPU-only list work (`split_and_merge_cells.h`, the resize plan), one
public header per operation group (`regions.h`, `remap_grid_values.h`, `resize_cells.h`) with one private
`_cpu.cc` / `_gpu.cu` pair per operation, and the step callbacks. `native/fluidity/batches/` (namespace
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
- Commit messages end with one `Assisted-by: Claude Code:<model id>` line per model that worked on the commit
  (the implementer's model, for example `claude-opus-5-5`, plus `claude-fable-5-2` for the session that directs and
  reviews), then `Claude-Session: <session link>`. No `Co-Authored-By`
  line; the owner is the sole author (owner answer A39, 2026-11-15).
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
- **Constants (design 3.2, 3.4, 3.7, never changed without the owner):** `kMaxFilledShare = 2.0`,
  `kAmountTolerance = 1e-12`; `CellSizeSettings` defaults `surface_reach_in_cell_widths = 8`,
  `solid_reach_in_cell_widths = 2`, `merge_width_share = 0.8`, `merge_wait_duration = 0.25` s,
  `max_cell_splits_per_resize = 65536`; `RegionSettings` defaults `6`, `7`, `overlap_layer_count = 2`,
  `particles_per_refilled_cell = 8` (1 to 9); `BatchSettings` defaults `max_whole_grid_share = 0.5`,
  `min_blocks_per_batch = 64`.
- **One child order:** `cell_eighth = x + 2 y + 4 z` (grid 3.1) everywhere. `main`'s permutation
  `[0,4,2,6,1,5,3,7]` appears only inside the comparison tests that convert `main`'s numbers.
- **Size levels:** level 0 is the biggest cell; `level + 1` is half the width (grid plan Task 8; amended
  2026-11-14, A17: the task was renumbered).
- **Sums:** every total over cells, particles or blocks is one partial per cell block (cell places, then
  particles in grouped order inside the block), then `math::TreeSum` over the block partials in block order
  (design 5.10). A per-destination value is gathered by one iteration in a fixed order. No floating-point
  atomics. The reported failure is the first failing cell in the grid's fixed order, then the lowest particle id.
- **One material per record (A8, A10):** a grid-only record holds one liquid material; gas markers, snow and
  springy-solid particles are never taken into a record; `ApplyGridOnlyRegionToGrid` writes only into the one
  liquid material set; a grid-only cell never touches a cell that holds gas.
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
- **Tolerances (design 6):** CPU against GPU: plans, regions, layers, ids, counts and the ledger's counts equal;
  remapped values, moved amounts, carried velocity `|a - b| <= 1e-13 * max(|a|, |b|) + 1e-13 * (largest value of
  that array in the case)`; `depth_below_surface` within `1e-13` of the cell width. Repeat runs: identical bytes.
- **Test labels (amended 2026-11-14, Foundation as built):** shared-formula, operation and scene tests `LABEL cpu`
  (the container's `cuda` step runs them too); device tests `LABEL gpu`. Every operation with a
  `_gpu.cu` has a `no_gpu` test "on the GPU without a GPU gives `kRunOnCallFailed`".
- **Test scenes** come from the step's `fluidity_step_test_support` (`step_test_scenes.h`, extended in Task 1); a
  step scene used with cell resize gets `has_cell_resize = true` through `scene_to_change()`. This module's
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

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: Foundation,
contact, grid, particles, transfer, pressure, viscosity, materials, solids, gas and step, each with its comment
pass done; the step with `resize_access.h` (`ResizeTry`, `BuildResizeTry`, `ApplyResizedGrid`, `CommitResize`,
`RejectResize`), the seven region arrays in `GridFields`, the callbacks `apply_grid_only_region`,
`move_grid_only_liquid`, `read_largest_grid_only_speed`, `resize_cells`, `choose_cell_size`, `run_stages`, and
`StepArray` / `ReadStepArrays` (step plan Tasks 32, 33); and the A15 tasks of pressure (54 to 61:
`solve_batches.h`, the batched layout), viscosity (26 to 28) and gas (29, 30), and the step's amended Tasks 8, 33
(`RunMode::kBatchedWithSolvesInPieces`, `RunPlan::max_solve_loaded_blocks`, the loaded try's piece buffers).

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line, and
`grep -c "apply_grid_only_region\|move_grid_only_liquid\|run_stages\|ReadStepArrays\|free_grid_storage" native/fluidity/step/advance_to.h native/fluidity/step/step_stages.h native/fluidity/step/resize_access.h`
counts at least one in each file. If either fails, stop and ask the owner.

**Amended 2026-11-14 (Foundation as built):** Foundation as built also provides `base/run_on_call.h`
(`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, the sums with their `SumBuffers` and the stable
order and prefix sum with their `OrderBuffers` (one form each, A31), `base/test_support.h` (library `fluidity_base_test_support`) and
`fluidity_add_test_library`; read `AGENTS.md` "Structure" before the first task. The gate prints seven `PASS` lines:
`format`, `rules`, `cpu` debug, `cpu` release, `tidy`, `paper`, `cuda`.

## Review Focus

1. **A resize that fails after the resized grid was built.** The free storage must come back, the committed
   grid, fields, particles and gas keep their bytes, and the next `TryStep` commits. Pinned:
   `ResizeCells, FailureAtEachStep_ChangesNothing` in Task 26.
2. **Merge wait against "nothing to do".** A second resize at the same committed time must add no wait and
   change nothing, while a change of `merge_ready_duration` alone is committed. Pinned:
   `ResizeCells, CalledTwiceAtSameTime_SecondDoesNothing` in Task 24.
3. **The last small cell of a split leaving the range by rounding.** `B` minus the sum of the first seven must
   not go below the range; then every small cell gets `B / 8`. Pinned:
   `ComputeSplitAmounts, LastCellOutOfRange_AllGetAnEighth` in Task 5.
4. **A grid-only cell holding, after a step, one particle of its own material and one of another.** The cell
   returns to the particle region with its record refilled, the foreign particle untouched, each material's
   totals kept. Pinned: `ApplyRegionTransfer, OwnAndForeignParticleInOneCell_CellReturnsToParticles` in Task 25.
5. **A batch cut that does not follow the layers** (a range ending in the middle of a layer of blocks). Pinned:
   `ApplyStagesInBatches, CutInsideALayer_SameBytesAsLayerCut` in Task 40 (A15: compared with the cut at whole
   layers of the same mode, and within tolerance of one piece).

---

# Part A: changing cell sizes

### Task 1: Step test scenes for cell resize

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

### Task 2: Method paper, the resize transaction, the size rule and the remap

**Kind:** mechanical

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/cell_resize.typ` (included after the step section), `figures/cell_size_rule.typ`,
  `figures/split_remap.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include, notation rows), `docs/glossary.md`

**Interfaces:**
- Produces labels `<sec-cell-resize>`, `<sec-cell-size-rule>`, `<sec-remap>`, `<fig-cell-size-rule>`,
  `<fig-split-remap>`, `<proc-resize-cells>`, `<proc-find-cells-that-must-split>`.

- [ ] **Step 1: Text,** design 8 outline items 1 to 5: why resizing is its own transaction and what a failed one
  leaves (5.1); the wanted width, its two reaches and the merge wait (5.3); the cells that must also split; the
  remap of amounts (limited slope, cell 7 takes the rest), pressures, labels and face flows (5.4, the table);
  splitting particles needs both grids (5.4, C2, the plain consequence). Cite [6] Wang, Sun, Zhu 2026, [1] Jiang
  2015, Losasso, Gibou, Fedkiw 2004 by author and year. Glossary rows: `wanted cell width`, `merge_ready_duration`,
  `depth_below_surface`, `ResizeMap`, `cell source`, `filled share`, `excess`.
- [ ] **Step 2: Figures.** `<fig-cell-size-rule>`: a vertical cut through a wave over a solid step, cells coloured
  by size, both reaches drawn. `<fig-split-remap>`: one big cell and its 8 small cells, the outer pieces, the 7
  tree faces numbered as Task 6 and the 5 zero faces.
- [ ] **Step 3: Pseudocode.** `<proc-resize-cells>`: the nine steps of design 5.1, numbered 1 to 9 exactly, "fails
  when" listing the `CellResizeError` values per step. `<proc-find-cells-that-must-split>`: design 5.3 step 4.
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`, look at
  every new page, fix overlaps; gate. **Commit.** Subject: `Describe how cells change size between steps`.

### Task 3: Method paper, the two regions and the region transfer

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

**Interfaces:**
- Produces design 3.1 exactly: `CellResizeError` (13 values), `CellResizeFailure`, `ToMessage`, `ToName`.

- [ ] **Step 1: Failing tests:** `ToName, EveryCellResizeError_IsLowerSnakeCaseWithoutK`
  (`kGridOnlyCellDrained` -> `"grid_only_cell_drained"`, all 13 by `static_assert` on the count);
  `ToMessage, EveryCellResizeError_DistinctAndNotEmpty`; `CellResizeFailure, DefaultIsNone`
  (shared header, `LABEL cpu`).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Messages are one sentence saying what the user can do
  ("A grid-only cell would empty in this step. A shorter step is tried."). **Step 4:** gate. **Commit.** Subject:
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

### Task 7: The cell size rule

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the shared-header tests are `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/cell_resize/cell_size_rule.h` (shared), `cell_size_rule_test.cc`,
  `testdata/proof_of_concept_width_decisions.txt`

**Interfaces:**
- Produces design 3.2 exactly: `CellSizeSettings`, `CellSizeFacts`, `ComputeWantedCellWidth`, `CellSizeChange`,
  `ComputeCellSizeChange` (design 5.3 rules 1 and 2).

- [ ] **Step 1: Failing tests** (`LABEL cpu`; smallest width `w = 0.025`):

  ```cpp
  TEST(ComputeWantedCellWidth, DepthsGrowWithDistance)   // design 7 "Wanted width", 5-level frame (biggest 16 w):
      // depth 0, 4, 8, 16, 64 w, no solid: w, w, w, 2 w, 8 w exactly
  TEST(ComputeWantedCellWidth, SolidDistanceCappedByBiggest)  // 2-level frame (biggest 2 w), depth infinite: solid at
      // 1 w -> w; at 5 w -> 2.5 w capped to 2 w; on the 5-level frame 5 w -> 2.5 w
  TEST(ComputeWantedCellWidth, AirPocketOrGas_IsSmallest)  // touches_air_pocket, then wants_smallest_cells: w
  TEST(ComputeCellSizeChange, WiderThanWanted_IsSplit)
  TEST(ComputeCellSizeChange, TwiceWidthWithinShare_IsMayMerge)  // width w, wanted 2.5 w: 2 w <= 0.8 * 2.5 w
  TEST(ComputeCellSizeChange, KeepsItsCount_AlwaysKeep)   // holds_particles_that_keep_their_count, width 4 w, wanted w
  ```
- [ ] **Step 2: Old numbers:** `DecideAdaptiveWidth3D` (`native/src/adaptive_width_policy_3d.cc:53-108`) with every
  demand set to the wanted width and zero eligible duration, for 20 (width, wanted) pairs; record the decision
  (`kRefine` = `kSplit`, coarsen-eligible = `kMayMerge`). `ComputeCellSizeChange, MatchesProofOfConcept`: equal.
- [ ] **Step 3: Run** (fail). **Step 4: Implement** rule 1 as design 5.3: `wanted = smallest * max(1, depth /
  (surface_reach * smallest))`, the same with the solid distance and its reach, the smaller, capped by the
  biggest width. **Step 5:** gate. **Commit.** Subject: `Decide the width each cell may have`.

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
two CPU grids are the step's, so `CellResize` neither allocates nor counts them; `BuildGridMemoryEstimate(kCpu, 1)`
with its storage count.

**Amended 2026-11-14 (Foundation as built):** `CellResize` also owns a `SumBuffers`, an `OrderBuffers` and a one-word changed flag, built once and
counted in its memory estimate; the test library is made with `fluidity_add_test_library`.

**Amended 2026-11-14 (A31, A30):** "scratch" wording replaced by "working buffers"; the estimate test is
`WorkingBufferEntriesMatchTheBuffers`; `ToCellResizeFailure` no longer maps a processor mismatch (none exists).

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
- Produces design 3.4 `RegionSettings`, `kMaxFilledShare`, `kAmountTolerance`, `CellResizeSettings`,
  `RegionTransferLedger`, `ResizeCellsReport`, `CellResize`, `BuildCellResize`, `ReadLastReport`,
  `ReadLargestGridOnlySpeed`, `BuildCellResizeMemoryEstimate`.
- **Plan decisions:** settings checks (`kBadInput`): every value finite; reaches positive; `merge_width_share` in
  `(0, 1]`; `merge_wait_duration >= 0`; `max_cell_splits_per_resize >= 1`; `grid_only_depth_in_cell_widths >
  particle_region_depth_in_cell_widths > 0`; `overlap_layer_count` 0 to 254; `particles_per_refilled_cell` 1 to 9;
  `CheckCanRunOn(run_on)`. `CellResize` holds, at `limits.max_cell_blocks` on `run_on`, the rows below (names
  `"cell resize: <row>"`, `sizeof` of the stored type with a `static_assert` each):

  | Row | Element | Per | Kept for |
  |---|---|---|---|
  | depth below surface, wanted width, merge durations after, merge weight | `double` x 4 | cell place | one call |
  | wanted change, merge veto, region wish | `std::uint8_t` x 3 | cell place | one call |
  | resize map cell source, cell old place | `std::uint8_t`, `std::uint32_t` | cell place | one call |
  | resize map face source, face old place | `std::uint8_t`, `std::uint32_t` | face place | one call |
  | removed amounts | `double` x (3 + ingredients) | cell place | one call |
  | refill list (`particles::NewParticles` arrays) | `Vec3` x 2, `Mat3`, `double` x (3 + ingredients), `std::uint8_t` | added particle | one call |
  | amounts at step start | `double` x (3 + ingredients) | cell place | one step try |
  | block partial sums | `double` x (8 + ingredients) | cell block | one step try |
  | sum buffers (amended, Foundation as built) | `ComputeSumBuffersSize(run_on).bytes_per_item` bytes | cell block | whole run |
  | sum buffers fixed bytes | 1 byte x `ComputeSumBuffersSize(run_on).fixed_bytes` | given count | whole run |
  | order buffers (amended) | `ComputeOrderBuffersSize(run_on).bytes_per_item` bytes | cell place | whole run |
  | order buffers fixed bytes | 1 byte x `ComputeOrderBuffersSize(run_on).fixed_bytes` | given count | whole run |
  | changed flag (amended) | `std::uint32_t` | given count 1 | whole run |

  `cell_resize_access.h` gives the operations of this module writable views of these buffers.
- **Plan decision (amended 2026-11-14, Foundation as built):** `CellResize` owns `SumBuffers sum_buffers` (built by
  `BuildSumBuffers(run_on, limits.max_cell_blocks)`: the block-partial sums of Tasks 22, 23, 26 and 28) and
  `OrderBuffers order_buffers` (`BuildOrderBuffers(run_on, 64 * limits.max_cell_blocks)`: the integer prefix sums of
  the refill ids, Tasks 25 and 28), reached through `cell_resize_access.h`; per-step calls pass these buffers and
  allocate nothing. The estimate's working-buffers entries use the `CellResize`'s own processor, which its arguments already
  fix: `kCpu` when `grid_copies == kHeldByStep` (the batched GPU run resizes on the CPU, Task 43), else
  `grid_run_on`; so `BuildCellResizeMemoryEstimate` keeps its signature. A failed build of these buffers is `kOutOfMemory`
  (or `kRunOnCallFailed`), as for the other buffers of this table; the mapping function `ToCellResizeFailure(BufferError)`
  (private, `cell_resize_access.h`) is the one place a `BufferError`, and through `ToBufferError` a
  `RunOnCallStatus`, becomes a `CellResizeFailure`.
- **Plan decision (A17):** `CellResizeSettings` gains `RunOn grid_run_on = RunOn::kCpu;` (the processor of the
  simulation's grid; the caller's choice, A12) and `BuildCellResizeMemoryEstimate(std::uint32_t ingredient_count, RunOn
  grid_run_on = RunOn::kCpu)`. With `grid_run_on == kGpu` the rows add `"cell resize: committed grid copy"` and
  `"cell resize: resized grid on the CPU"` (each `grid::BuildGridMemoryEstimate(kCpu, 1)` tables at
  `limits.max_cell_blocks`, `where = kCpu`) and `"cell resize: cell list upload"` (8 B per cell place, `where =
  kRunOn`), used by Task 24.
- **Plan decision (reconcile, coordinator item 6):** `enum class GridCopies : std::uint8_t { kNone, kHeldByCellResize,
  kHeldByStep };` `CellResizeSettings` gains `GridCopies grid_copies = GridCopies::kNone;` and the estimate becomes
  `BuildCellResizeMemoryEstimate(std::uint32_t ingredient_count, RunOn grid_run_on = RunOn::kCpu, GridCopies
  grid_copies = GridCopies::kNone)`. `kNone` with a CPU grid; `kHeldByCellResize` in a one-piece GPU run (the two
  CPU grid rows above, built as two `GridBuffers` sharing one CPU `grid::GridBuilderBuffers`, entries
  `grid::BuildGridMemoryEstimate(kCpu, 2, grid::BuilderBuffersSharing::kOneForAllGrids)` with `where = kCpu`);
  `kHeldByStep` in a batched GPU run (no CPU grid row: the committed grid is `step::FindCpuGridCopy(simulation)` and
  the resized CPU grid is built into `ResizeTry::free_cpu_grid_copy_buffers()`, step plan Task 32). `kBadInput`:
  `kHeldByCellResize` or `kHeldByStep` with `grid_run_on == kCpu`, `kNone` with `kGpu`. I/O sets it from the run
  plan's mode.
- Test support (plan decision): `std::expected<step::Simulation, step::StepFailure> BuildResizeTestSimulation(
  const step::TestScene& scene, std::span<const MemoryEstimateEntry> cell_resize_estimate, const step::StepCallbacks& callbacks)`
  (plan from `step::BuildRunPlan` with ample memory, then `step::BuildSimulation`).

- [ ] **Step 1: Failing tests:** `BuildCellResize, Defaults_Succeed`; `BuildCellResize, TenParticlesPerRefilledCell_IsBadInput`;
  `BuildCellResize, GridOnlyNotDeeperThanParticleRegion_IsBadInput`; `BuildCellResizeMemoryEstimate, RowsMatchWhatIsBuilt`
  (bytes allocated by `BuildCellResize` == `ComputeMemoryBytes` of the rows at the limits); `ReadLargestGridOnlySpeed,
  BeforeFirstStep_IsZero`; `ReadLastReport, BeforeFirstResize_NotResized`; `BuildCellResizeMemoryEstimate,
  GridOnGpu_AddsTheGridCopies` (A17); `BuildCellResizeMemoryEstimate, HeldByStep_NoGridCopyRows` (reconcile: no
  `"cell resize: committed grid copy"` or `"resized grid on the CPU"` entry, and `BuildCellResize` allocates none);
  `BuildCellResize, GridCopiesNotMatchingGridRunOn_IsBadInput` (reconcile); `BuildCellResizeMemoryEstimate,
  WorkingBufferEntriesMatchTheBuffers` (amended, Foundation as built: the four working-buffers entries sum to
  `sum_buffers.byte_count() + order_buffers.byte_count()` for `kCpu`, and for `kHeldByStep` they use the CPU sizes);
  `LABEL no_gpu`:
  `BuildCellResize, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Hold the settings and buffers of the cell resize`.

### Task 11: The region types and `ComputeDepthBelowSurface` on the CPU

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** `depth_below_surface_cpu.cc` holds the public function (checks, dispatch, CPU loop); until Task 12
the `kGpu` branch returns `kRunOnCallFailed`; the per-cell pass update is a shared function Task 12's GPU loop reuses.

**Files:**
- Create: `native/fluidity/cell_resize/regions.h` (types and the depth function), `depth_below_surface_cpu.cc`,
  `depth_below_surface_steps.h` (shared: `ComputeCellDepthInPass`, one cell place of one pass, reading the previous
  pass's depths and writing the next, so the CPU and GPU passes are the same Jacobi passes), `depth_below_surface_test.cc`

**Interfaces:**
- Produces design 3.5 `RegionKind`, `GridOnlyAmounts`, `ComputeDepthBelowSurface`. **Plan decision:**
  `struct GridOnlyAmountsToChange` has the same seven members as `GridOnlyAmounts`, non-const pointers, and
  `ingredient_count`.
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

### Task 13: Cell size facts, one pass per cell, CPU and GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two halves with `cell_size_facts_gpu.h`, per-cell facts in a shared steps header with a `cpu` test
over every thread block and thread; the GPU comparison demands equal discrete outputs and `wanted_width` within
tolerance instead of equal bytes.

**Files:**
- Create: `native/fluidity/cell_resize/cell_size_facts.h` (private, CPU only), `cell_size_facts_cpu.cc` (the function:
  checks, dispatch, CPU loop), `cell_size_facts_gpu.h` (private), `cell_size_facts_gpu.cu`, `cell_size_facts_steps.h`
  (shared: `ComputeCellSizeFactsOfCell`, one cell place: its facts, wanted width, change and veto, from raw arrays),
  `cell_size_facts_test.cc`, `cell_size_facts_steps_test.cc`, `cell_size_facts_gpu_test.cc`
- Modify: `CMakeLists.txt` (sources, GPU source, the three tests with `LINKS fluidity_cell_resize_test_support
  fluidity_base_test_support`)

**Interfaces:**
- Produces (plan decision, private):

  ```cpp
  struct CellSizeFactsInputs {
    const grid::StaggeredGrid* grid; grid::CellBlockRange blocks;
    const grid::CellValues<double>* depth_below_surface;       // m, ComputeDepthBelowSurface.
    pressure::SolidCoverage solids;                             // ResizeTry::last_solid_coverage().
    const pressure::GeometrySnapshot* geometry;                 // ResizeTry::last_geometry().
    const grid::CellValues<std::uint32_t>* pocket_of_cell;      // Committed GridFields.
    const grid::CellValues<double>* gas_volume;                 // Committed gas amounts; null without gas.
    const particles::ParticlesTry* particles_try;               // Grouped for grid.
    std::span<const std::uint8_t> material_keeps_its_count;     // One per material (CountRule::kKeepsItsCount).
    const CellSizeSettings* settings;
  };
  enum MergeVeto : std::uint8_t { kKeepsItsSize = 1, kTouchesAirPocketGasOrSolid = 2 };
  struct CellSizeFactsOutputs { grid::CellValues<double>* wanted_width; grid::CellValues<std::uint8_t>* change;
                                grid::CellValues<std::uint8_t>* merge_veto; };
  std::expected<void, CellResizeFailure> ComputeCellSizeFacts(const CellSizeFactsInputs&, CellSizeFactsOutputs);
  ```
  `distance_to_solid` = smallest corner value of the cell; `touches_air_pocket` = the cell or a touching cell has
  `pocket_of_cell != kNoAirPocket`; `wants_smallest_cells` = the cell or a touching cell has `gas_volume > 0` or
  `content == kGas`; `kTouchesAirPocketGasOrSolid` also when a touching cell's content is `kAir` or `kSolid`.

- [ ] **Step 1: Tests.** `LABEL cpu`, on simulations from `BuildResizeTestSimulation` after one step and a
  `step::BuildResizeTry`: `ComputeCellSizeFacts, SnowCells_KeepTheirSize` (`BuildSnowAndJellyScene`);
  `ComputeCellSizeFacts, GasCells_WantSmallest` (`BuildSmokeOverDeepWaterScene`); `ComputeCellSizeFacts,
  DeepTank_WantedWidthFromDepth`; `ComputeCellSizeFacts, SolidDistanceIsSmallestCorner` (`BuildFreeBoxScene(500)`).
  `LABEL cpu` (`cell_size_facts_steps_test.cc`, the main evidence until the hardware run):
  `ComputeCellSizeFactsOfCell, EveryThreadOrder_EqualsCpu` (the deep tank and snow cases: the function run for every
  thread block and thread of `ComputeLaunchSize(cell place count)`, launch order and reverse: all three outputs
  equal the CPU function's bytes). `LABEL gpu`: `ComputeCellSizeFacts, OnGpu_EqualsCpu` (`change` and `merge_veto`
  equal; `wanted_width` within `1e-13` of the cell width, the depth tolerance of design 6, since no header promises
  its bits; all three to `WriteResultDigest`). `LABEL no_gpu`: on the GPU without a GPU.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (one iteration per cell place reads its own particles in grouped
  order; the GPU function is a grid-stride loop over `ComputeCellSizeFactsOfCell`, launched with `ApplyGpuFunction`;
  the status maps through `ToCellResizeFailure(ToBufferError(status))`). **Step 4:** gate. **Commit.** Subject: `Gather what the size rule reads about each cell`. Body: `CUDA:
  compiled, not run`.

### Task 14: The resize plan

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test links `fluidity_grid_test_support` and `fluidity_base_test_support` through `LINKS`.

**Files:**
- Create: `native/fluidity/cell_resize/resize_plan.h` (private, CPU only), `resize_plan_cpu.cc`, `resize_plan_test.cc`

**Interfaces:**
- Produces (plan decision, private; design 5.3 rules 3 to 5):

  ```cpp
  struct ResizePlanInputs {
    const grid::StaggeredGrid* committed_grid; const CellSizeSettings* settings;
    const grid::CellValues<double>* wanted_width;  const grid::CellValues<std::uint8_t>* change;
    const grid::CellValues<std::uint8_t>* merge_veto;  const grid::CellValues<std::uint8_t>* is_grid_only_region;
    const grid::CellValues<double>* merge_ready_duration;   // Committed, s.
    double elapsed_since_last_merge_test = 0;               // s. See below.
    std::uint64_t max_cell_count = 0;                       // 64 * limits.max_cell_blocks.
  };
  std::expected<ResizePlan, CellResizeFailure> BuildResizePlan(const ResizePlanInputs& inputs,
                                                               grid::CellValues<double>& merge_ready_duration_after);
  ```
  CPU arrays (a GPU run reads the per-cell arrays back with `grid::ReadValues`). **Plan decision (merge wait):**
  `merge_ready_duration` grows by `elapsed_since_last_merge_test`, the committed time elapsed since the last
  committed resize evaluated the merge test (the first call: the committed step's duration), so a second call at
  the same committed time adds nothing. A split whose closure reaches a cell with `kKeepsItsSize` is removed and
  counted in `refused_split_count`, then the closure is found again without it. A merge also needs every cell
  touching the big cell to satisfy `2 w <= merge_width_share * wanted` at the big cell's width, all 8 in one region,
  no veto, and no conflict with the split cells; a refused merge whose 8 had waited long enough is counted in
  `refused_merge_count`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `LINKS fluidity_grid_test_support fluidity_base_test_support`;
  hand-built arrays on grids from `grid_test_support`): `BuildResizePlan, MoreSplitsThanTheLimit_FirstInFixedOrder` (5 wanted, limit 3: `wanted_split_count` 5, the first 3
  and their closure); `BuildResizePlan, ClosureReachesKeepSizeCell_SplitRefusedAndRerun`;
  `BuildResizePlan, MergeAfterUnbrokenWait` (0.1 s per call: no merge at 0.2 s, merge at 0.3 s);
  `BuildResizePlan, BrokenOnce_WaitStartsAgain`; `BuildResizePlan, ZeroElapsed_DurationsUnchanged`;
  `BuildResizePlan, MergeVetoedByTouchingAir_Refused`; `BuildResizePlan, MergeAgainstASplit_Refused`;
  `BuildResizePlan, NeighbourWantsSmaller_NoMerge`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Plan which cells split and merge in one resize`.

### Task 15: `ResizeMap` and the remap of cell values on the CPU

**Kind:** numerics

**Files:**
- Create: `native/fluidity/cell_resize/remap_grid_values.h`, `remap_grid_values_cpu.cc`, `remap_grid_values_test.cc`

**Interfaces:**
- Produces design 3.6 `CellSource`, `ResizeMap`, `BuildResizeMap`, `CellValueKind`, `ApplyResizeToCellValues`
  (explicit instantiations for `double`, `std::uint8_t`, `std::uint32_t`).
- **Plan decisions:** `ResizeMap` keeps pointers to both grids (valid during the transaction) and, per new cell
  place, the source and one old place (the big cell of a split; small cell 0 of a merge: siblings share one old
  block, so the other seven follow by local offset). For `kAmount` the neighbours of a split are the old cells of the
  big cell's level across its 6 faces; with `old_weight` given, a neighbour counts only when its weight is above 0
  (the transaction passes the old `grid_only_current_volume`, so only same-size grid-only cells count). `kLabel`'s
  missing value is `std::numeric_limits<Value>::max()` (equal to `pressure::kNoAirPocket` for `std::uint32_t`).
  `kPerUnitOfWeight` with all weights 0 takes the equal mean. **New public name (plan decision):**

  ```cpp
  // m3, kg: the grid-only mass, rest_volume and current_volume remapped together with one limiter per split cell,
  // the smallest of the three ComputeSplitLimiter values, so a split keeps density and volume ratio.
  std::expected<void, CellResizeFailure> ApplyResizeToCellAmounts(
      const ResizeMap& map, std::span<const grid::CellValues<double>* const> old_amounts,
      const grid::CellValues<double>* old_weight, std::span<grid::CellValues<double>* const> new_amounts);
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`; old grid with one split cell and one merge elsewhere):
  `BuildResizeMap, SplitAndMerge_SourceOfEveryPlace` (no `kNewCell`); `ApplyResizeToCellValues,
  PerUnitOfWeight_SplitCopiesMergeWeightedMean` (weights 1..8, 1 rounding step; null weight: equal mean);
  `ApplyResizeToCellValues, Label_MergeTakesSmallestPresent` (`{max, 5, 3, max, ...}` -> 3; all `max` -> `max`);
  `ApplyResizeToCellValues, Amount_SplitSumsAndMergeSums` (2 rounding steps); `ApplyResizeToCellAmounts,
  ProportionalAmounts_KeepDensity` (mass = 1000 x current volume: each small cell's ratio 1000 within 1e-14);
  `ApplyResizeToCellValues, OldValuesOfAnotherGrid_IsWrongGridVersion`; `ApplyResizeToCellValues,
  NaNInOldValues_IsNotFinite` (failure `cell` = first new cell in fixed order).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** (one iteration per new cell place gathers its sources). **Step
  4:** gate. **Commit.** Subject: `Map a grid onto its resized version and carry cell values over`.

### Task 16: Ingredient volumes and face velocities on the CPU

**Kind:** numerics

**Files:**
- Modify: `remap_grid_values.h`, `remap_grid_values_cpu.cc`, `remap_grid_values_test.cc`

**Interfaces:**
- Produces design 3.6 `ApplyResizeToIngredientVolumes`, `ApplyResizeToFaceVelocity` (design 5.4 "Face velocity").
  **Plan decision:** face places of the new grid have a private source (`kSamePiece`, `kQuarterOfBigPiece`,
  `kSumOfFourPieces`, `kInnerFaceOfSplit`); flows use the whole piece area (`grid::ComputeFaceArea`), never an
  open area; the 4 pieces of a merged face are summed in ascending `FacePieceId`; the 12 inner faces of a split are
  written by the iteration of its small cell 0 with `ComputeInnerFaceFlows`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ApplyResizeToIngredientVolumes, SplitThenMerge_FractionsKept)  // design 7: 3 ingredients; small cells'
      // fractions equal the big cell's; merge returns the start volumes within 2 rounding steps
  TEST(ApplyResizeToFaceVelocity, Split_EachSmallCellHasAnEighthOfTheImbalance)  // design 7 "Face flow, split":
      // zero imbalance, then imbalance q: each small net outflow q / 8; each outer big face's flow kept; 1e-13 of
      // the largest flow
  TEST(ApplyResizeToFaceVelocity, Merge_BigFaceFlowIsSumOfPieces)  // design 7: velocities 0.1, 0.2, 0.3, 0.4 on 4
      // pieces: big velocity x area == their flow sum within 1 rounding step
  TEST(ApplyResizeToFaceVelocity, MergeBesideSmallCells_PiecesLiveOn)
  TEST(ApplyResizeToFaceVelocity, UnchangedPieces_CopiedByFacePieceId)  // exact
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Carry ingredient volumes and face velocities onto resized cells`.

### Task 17: The remap on the GPU

**Amended 2026-11-14 (A17):** the map's two CPU grids are named (Task 24).

**Amended 2026-11-14 (Foundation as built):** private `remap_grid_values_gpu.h` returning `RunOnCallStatus`; the per-place gathers in
`remap_grid_values_steps.h`, which the CPU loops of Tasks 15, 16 may call and a `cpu` test runs over every thread
block and thread; digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/cell_resize/remap_grid_values_gpu.cu`, `remap_grid_values_gpu.h` (private),
  `remap_grid_values_steps.h` (shared: `ComputeRemappedCellValue`, `ComputeRemappedCellAmounts` (one limiter for the
  three amounts), `ComputeRemappedIngredientVolumes`, `ComputeRemappedFaceVelocity`, `ComputeInnerFaceVelocitiesOfSplit`,
  one new place each, over raw map and value arrays), `remap_grid_values_steps_test.cc`,
  `remap_grid_values_gpu_test.cc`; extend `cell_resize_no_gpu_test.cc`
- Modify: `remap_grid_values_cpu.cc` (the `kGpu` branches under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

- [ ] **Step 1: Tests.** `LABEL cpu` (`remap_grid_values_steps_test.cc`, the main evidence until the hardware run):
  `RemapGridValuesSteps, EveryThreadOrder_EqualsCpu` (Tasks 15 and 16 cases: each per-place function run for every
  thread block and thread of `ComputeLaunchSize` over new cell places and new face places, launch order and reverse:
  outputs' bits equal the CPU functions'). `LABEL gpu`: `BuildResizeMap, OnGpu_EqualsCpu` (bytes: the map is
  discrete); `ApplyResizeToCellValues, OnGpu_EqualsCpu` (labels equal; others design 6), `ApplyResizeToCellAmounts,
  OnGpu_EqualsCpu`, `ApplyResizeToIngredientVolumes, OnGpu_EqualsCpu`, `ApplyResizeToFaceVelocity, OnGpu_EqualsCpu`
  (Tasks 15 and 16 cases, design 6 tolerance; every output to `WriteResultDigest`); `...OnGpuTwice_
  GivesIdenticalBytes` for the face velocity. `LABEL no_gpu`: `BuildResizeMap, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** design 6 "Work": one thread per new cell place, one per new face place, the inner faces by
  the thread of the big cell; each GPU function a grid-stride loop over its steps function, launched with
  `ApplyGpuFunction`. The map is built on the CPU from the two grids' tables and copied up (A17: the
  committed grid's `ReadGrid` copy and the CPU-built resized grid of Task 24).
- [ ] **Step 3:** gate. **Commit.** Subject: `Remap grid values on the GPU`. Body: `CUDA: compiled, not run`.

### Task 18: The carried velocity and the region formulas

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

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** `grid_only_to_grid_cpu.cc` holds the public function (checks, dispatch, CPU loop); the per-face and per-cell work goes in `grid_only_to_grid_steps.h` for Task 20; tests link the helper libraries.

**Files:**
- Create: `native/fluidity/cell_resize/grid_only_to_grid_cpu.cc`, `grid_only_to_grid_steps.h` (shared:
  `ComputeGridOnlyFaceValues` per face place, `ComputeGridOnlyCellValues` per cell place), `grid_only_to_grid_test.cc`
- Modify: `regions.h`, `CMakeLists.txt`

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

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the public function in `move_grid_only_liquid_cpu.cc` (checks, dispatch, CPU loop); the shared test is `LABEL cpu`; the per-cell move and block partials are shared functions Task 22 reuses.

**Files:**
- Create: `native/fluidity/cell_resize/grid_only_move_math.h` (shared), `move_grid_only_liquid_cpu.cc`,
  `move_grid_only_liquid_test.cc`, `grid_only_move_math_test.cc`
- Modify: `regions.h`

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

### Task 23: `ComputeLiquidTotals`, CPU and GPU

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** two halves with `liquid_totals_gpu.h`; per-block partials in a shared steps function with a `cpu`
test; the sum over blocks with `ComputeTreeSum` and its `SumBuffers`; the GPU comparison within design 6.

**Amended 2026-11-14 (A31, A30):** `ComputeLiquidTotals` has one form, which takes `SumBuffers& sum_buffers` (the
design's form without buffers and the overload pair are gone); buffers on the other processor are a precondition,
not `kBadInput`; `ScratchForm_SameBytesAsFormWithout` removed, `ScratchTooSmall_IsBadInput` renamed.

**Files:**
- Create: `liquid_totals_cpu.cc`, `liquid_totals_gpu.h` (private), `liquid_totals_gpu.cu`, `liquid_totals_steps.h`
  (shared: `ComputeLiquidBlockPartials`, one cell block: cell places, then particles in grouped order),
  `liquid_totals_test.cc`, `liquid_totals_steps_test.cc`, `liquid_totals_gpu_test.cc`
- Modify: `regions.h`, `CMakeLists.txt`

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
      const particles::ParticlesTry& particles_try, const GridOnlyAmounts& amounts,
      std::span<const std::uint8_t> material_is_counted, const grid::StaggeredGrid& grid,
      grid::CellBlockRange blocks, SumBuffers& sum_buffers);
  ```
  `ResizeCells` and the step callbacks pass `cell_resize.sum_buffers`; render output passes its recorder's
  (render output plan Task 31); tests build their own with `BuildSumBuffers(run_on, blocks.count)`.

- [ ] **Step 1: Tests.** `LABEL cpu`: `ComputeLiquidTotals, ParticlesPlusRecords_TreeSumOfBlocks` (bytes against a
  test-side block-partial sum); `ComputeLiquidTotals, GasMarkersLeftOut` (`BuildSmokeOverDeepWaterScene`, liquid mask);
  `ComputeLiquidTotals, MaskOfWrongLength_IsBadInput`; `ComputeLiquidBlockPartials, EveryThreadOrder_EqualsCpu`
  (steps test, the main evidence until the hardware run: every thread block and thread, launch order and reverse).
  `LABEL gpu`: `ComputeLiquidTotals, OnGpu_EqualsCpu` (within design 6, amounts; totals to `WriteResultDigest`).
  `ComputeLiquidTotals, BuffersTooSmall_IsBadInput` (`SumBuffers` for 1 block, 2 blocks given).
  `LABEL no_gpu`: on the GPU without a GPU. The totals over blocks use `ComputeTreeSum` with the given `SumBuffers` on
  both processors.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Sum the liquid held by particles and grid-only cells`. Body: `CUDA: compiled, not run`.

### Task 24: `ResizeCells`, the transaction for cells and grid values

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
- Produces design 3.4 `ResizeCells` with design 5.1 steps 1 to 5, 6 without `ApplyRegionTransfer` (Task 25 adds it),
  8 and 9; regions are copied unchanged in this task. Body numbered `// Step N:` from `<proc-resize-cells>`.
- **Plan decisions:**
  - Step 3 "nothing to do" also requires that no `merge_ready_duration` changes; a change of durations alone is
    committed as a resize with the same cells and `version + 1` (design 5.1 step 3 and 5.3 step 5 together). The
    time of the last committed merge test is kept in `CellResize` and moves only at a commit.
  - Step 5 remaps: `pressure` (`kPerUnitOfWeight`, weight = `has_pressure` as 0 or 1 in the merge-weight row),
    `has_pressure` (merge: 1 if any; private rule beside the `kLabel` code), `pocket_of_cell` (`kLabel`),
    `velocity_after_pressure` and `gas_velocity_after_pressure` (`ApplyResizeToFaceVelocity`), the three grid-only
    amounts (`ApplyResizeToCellAmounts`, weight = old `grid_only_current_volume`), `grid_only_ingredient_volume`,
    region bytes and layers (`kLabel`), `merge_ready_duration` (0 for split and merged cells, else the value after
    the plan); the gas arrays through `gas::ToArraysToRemap(*resize_try.gas_try())` per design 5.4 (densities and
    flame weighted by old `gas_volume`, `gas_volume` by `ApplyResizeToCellValues(kAmount)` with null weight, both
    pressures copied on split and `gas_volume`-weighted on merge). Each air pocket's `first_cell_index` becomes the
    smallest new cell place carrying its label.
  - (Amended, Foundation as built) Any failure after `BuildResizeTry` calls `RejectResize`; for a failed check the
    committed state keeps its bytes (the tests below), and after `kRunOnCallFailed` the try's arrays are unspecified
    and the step reports the failure so the run stops (`AGENTS.md` "Structure"); no test demands bytes after a GPU
    call failure. Step 8 sums with `ComputeLiquidTotals(..., cell_resize.sum_buffers)`.
  - Step 8 tolerance: `|after - before| <= kAmountTolerance * max(|before|, |after|)`, for mass, rest volume,
    current volume, each ingredient volume (liquid materials) and the gas amount `sum gas_volume * gas_density`.
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
    call between CPU memory and the grid's processor, this call; with `ChooseCellSize`'s first copy of Task 29).
- Test support (plan decision): `step::SimulationSnapshot BuildGridOnlySnapshot(const step::SimulationSnapshot&
  committed, double below_height)`: every liquid cell whose centre is below `below_height` becomes grid-only with a
  record equal to its particles' sums, those particles removed.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ResizeCells, DeepTank_SplitsNearSurfaceMergesBelow)   // BuildDeepTankScene, one step, then ResizeCells: cells
      // within 0.2 m of the surface at 0.025 m, deeper bigger; CheckSizeStepRule; state_version + 1, time unchanged
  TEST(ResizeCells, RecordsRemapped_TotalsKept)               // BuildGridOnlySnapshot(below 0.5 m) rebuilt with
      // BuildSimulationFromSnapshot: totals of the step-8 list within kAmountTolerance
  TEST(ResizeCells, CalledTwiceAtSameTime_SecondDoesNothing)  // design 7 "Nothing to do"; Review Focus 2
  TEST(ResizeCells, MergeWait_NeedsUnbrokenTimeAndFailedResizeEarnsNone)  // design 7 "Merge wait": merge test held
      // 0.2 s, broken one step, held again: no merge before 0.25 s unbroken; one resize made to fail (limits lowered
      // so a split elsewhere is kTooManyCellBlocks, raised again with ApplyLimits) adds no wait
  TEST(ResizeCells, AirPocketFirstCellTranslated)
  TEST(ResizeCells, GasArraysRemapped_GasAmountKept)          // BuildSmallFireScene with has_cell_resize
  TEST(ResizeCells, TooManyCellBlocks_ChangesNothing)          // ExpectSameBytes on snapshots before and after
  TEST(ResizeCells, GridCopyPath_SameAsCpuGrid)                // A17: DeepTank with grid_run_on kGpu forced onto CPU
      // buffers (the copy path runs, CPU to CPU): same report and snapshot bytes as grid_run_on kCpu;
      // bytes_copied == 456 * committed blocks + 8 * new cells; copy_count == 2 (A17 batched-run buffers)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Resize cells after a committed step as one transaction`.

### Task 25: Region transfer, rules 1 to 3

**Amended 2026-11-14 (A31):** `ComputeExclusivePrefixSum` named without "scratch form" (one form, A31).

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the refill ids' integer prefix sum is `ComputeExclusivePrefixSum(counts, cell_count, first_ids, cell_resize.order_buffers)`; the public function lives in `region_transfer_cpu.cc` (checks, dispatch, CPU loop).

**Files:**
- Create: `native/fluidity/cell_resize/region_transfer_cpu.cc`, `region_transfer_test.cc`
- Modify: `regions.h`, `resize_cells_cpu.cc` (step 6 calls it)

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
  TEST(ApplyRegionTransfer, OwnAndForeignParticleInOneCell_CellReturnsToParticles)  // Review Focus 4
  TEST(ApplyRegionTransfer, ShallowGridOnlyCell_RefilledWithEightParticles)  // density, volume ratio, fractions of the
      // record; record exactly 0; AddReason::kRefillRegion; ledger into_particle_region == the old record
  TEST(ApplyRegionTransfer, CellTouchingSolidOrGas_StaysParticleRegion)
  TEST(ApplyRegionTransfer, GasMarkersNeverTakenIn)           // BuildSmokeOverDeepWaterScene
  TEST(ApplyRegionTransfer, RefillPastParticleLimit_IsTooManyParticles)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step N:` from `<proc-apply-region-transfer>`. **Step 4:**
  gate. **Commit.** Subject: `Move liquid between particles and grid-only cells at a resize`.

### Task 26: Region transfer, rules 4 and 5, the ledger, every failure

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
  TEST(ResizeCells, FailureAtEachStep_ChangesNothing)    // design 7 "Failed resize"; Review Focus 1. Sub-cases:
      // step 4, block limit below the resized grid (kTooManyCellBlocks); step 5, a NaN in one record of a snapshot
      // rebuilt with BuildSimulationFromSnapshot (kNotFinite); step 6, particle limit hit by a refill
      // (kTooManyParticles); step 8, two particles of mass 1e308 whose totals overflow (kNotFinite). Each: snapshot
      // bytes equal those taken after the step's commit; state_version unchanged; free storage given back (the next
      // TryStep builds version + 1 into it) and commits
  ```
  The step-7 sub-case (particle limit hit by `AdjustParticleCount`) is added to this test in Task 27, where the
  split runs inside the resize.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Refill the region border and keep the ledger of a resize`.

### Task 27: Particle count changes inside the resize, and conflict C2

**Kind:** integration

**Files:**
- Modify: `resize_cells_cpu.cc`, `resize_cells_test.cc`

**Interfaces:**
- Produces design 5.1 step 7: `particles::AdjustParticleCount(particles_try, committed grid, its
  NeighbourhoodInputs, resized grid, its NeighbourhoodInputs, every block)`; committed inputs from
  `last_solids_at_end()` (`outside_solid`, `side`) and committed `is_grid_only_region`; resized inputs from
  `solids_at_faces_to_fill()` after `solids::ComputeSolidsAtFaces(resized grid, all blocks, solids, ...)` and the try's
  `is_grid_only_region`; raw pointers from each array's `buffer().data()`. `GroupParticlesByBlock` afterwards when
  anything was split or merged. `particle_count` of the report is the result.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ResizeCells, FloatingCube_SplitsDeepInsideParticles)   // design 7 "C2": BuildFloatingCubeScene, first resize
      // splits all 125 cells; particles at least 1.5 old widths inside on both grids are split with the conservation
      // of particles 5.4 (their tolerances); the others keep their size and are counted in skipped_incomplete_count
      // (count from positions, test side)
  TEST(ResizeCells, DriftAcrossSizeChange_NeverSplit)          // design 7 "Drift": BuildMixedSizeColumnScene, one
      // particle given 0.5 m/s across the size change, 5 steps with a resize each: size_level unchanged, counted
  TEST(ResizeCells, RefusedSplitNextToSnowOrJelly_GridUnchangedThere)  // design 7 "Refused split":
      // BuildSnowAndJellyScene with has_cell_resize: a split whose closure reaches a snow cell, then a jelly cell:
      // refused_split_count 1 each, those cells unchanged
  ```
  And add sub-case 7 to `ResizeCells, FailureAtEachStep_ChangesNothing`.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Split and merge particles inside the resize with both grids`.

### Task 28: Region transfer and the whole resize on the GPU

**Amended 2026-11-14 (A31):** "scratch" wording replaced; the prefix sum and sums take `CellResize`'s buffers (one form).

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `region_transfer_gpu.h` returning `RunOnCallStatus`; per-cell work in
`region_transfer_steps.h` with a `cpu` test over every thread block and thread; the prefix sum and the ledger sums
through Foundation's one-form functions with `CellResize`'s `order_buffers` and `sum_buffers` (A31); discrete equal, amounts within design 6; digests.

**Files:**
- Create: `region_transfer_gpu.cu`, `region_transfer_gpu.h` (private), `region_transfer_steps.h` (shared:
  `ComputeCellRegionWish` per cell, `ComputeTakenInAmounts` per cell (its particles in id order), `ComputeRefillCount`
  per cell, `WriteRefillParticles` per cell from its first id, `ComputeLedgerBlockPartials` per cell block),
  `region_transfer_steps_test.cc`, `region_transfer_gpu_test.cc`, `resize_cells_gpu_test.cc`; extend
  `cell_resize_no_gpu_test.cc`
- Modify: `region_transfer_cpu.cc` (the `kGpu` branch under `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

- [ ] **Step 1: Tests.** `LABEL cpu` (`region_transfer_steps_test.cc`, the main evidence until the hardware run):
  `RegionTransferSteps, EveryThreadOrder_EqualsCpu` (Tasks 25, 26 cases: each function for every thread block and
  thread of `ComputeLaunchSize`, launch order and reverse, with the CPU `ComputeExclusivePrefixSum` between the count
  and write phases: regions, counts, ids and amounts' bytes equal the CPU function's). `LABEL gpu`:
  `ApplyRegionTransfer, OnGpu_EqualsCpu` (Tasks 25, 26 cases: regions, counts, ids equal; amounts within design 6);
  `ResizeCells, OnGpu_EqualsCpu` (Tasks 24, 27 cases: plans, regions, ids, counts equal; values within design 6);
  `ResizeCells, OnGpuTwice_GivesIdenticalBytes`; every test passes its main outputs to `WriteResultDigest`.
  `LABEL no_gpu`: `ApplyRegionTransfer, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** design 6: one thread per cell sums its particles in id order; the refill list from
  `ComputeExclusivePrefixSum(counts, cell_count, first_ids, cell_resize.order_buffers)` of per-cell counts; the
  ledger's totals by `ComputeTreeSum(partials, block_count, cell_resize.sum_buffers)`; returned to the CPU: the counts
  of step 3, the ids to split and merge, one failure record; launches through `ApplyGpuFunction`, no allocation per
  call; the status maps through `ToCellResizeFailure(ToBufferError(status))`. **Step 3:** gate. **Commit.** Subject:
  `Run the region transfer and the resize on the GPU`. Body: `CUDA: compiled, not run`.

### Task 29: The step callbacks and `ChooseCellSize`

**Amended 2026-11-14 (A17, batched-run buffers):** on a GPU grid `ChooseCellSize` reads a CPU copy of the committed
grid it already holds (closes "Batched solves (A15)" open point 5 in a batched run; the one-piece copies stay open
for the owner; step plan Appendix A row M8).

**Amended 2026-11-14 (reconcile, small fix a):** per reconcile item 13 the step passes
`committed_grid.tables(RunOn::kCpu)` as `committed`: real tables in a CPU run; in a batched GPU run
`step::FindCpuGridCopy(simulation)->tables(RunOn::kCpu)`, the real tables of the step's CPU copy (step Tasks 14,
33b), so `ChooseCellSize` probes them and makes no copy; only in a one-piece GPU run are they version 0 with null
arrays, and then `ChooseCellSize` reads its own CPU copy. `BuildCellResizeCallbacks` no longer takes or stores
`FindCpuGridCopy`. Replaces the earlier text, which gave null tables in every GPU run and handed
`step::FindCpuGridCopy(simulation)` to `BuildCellResizeCallbacks`.

**Amended 2026-11-14 (Foundation as built):** the test support lives in `fluidity_cell_resize_test_support` (made with `fluidity_add_test_library`)
and the tests name it in `LINKS`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/cell_resize/step_callbacks.cc`, `step_callbacks_test.cc`
- Modify: `resize_cells.h`, `cell_resize.cc`, `cell_resize_test_support.h/.cc`, `CMakeLists.txt` (`step_callbacks_test`
  `LABEL cpu LINKS fluidity_cell_resize_test_support fluidity_base_test_support`)

**Interfaces:**
- Produces design 3.4 `ChooseCellSize`, `ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep`.
- **Plan decisions:** `ChooseCellSize` probes `grid::FindCellAt` at `position + {-1, 0, 1}^3 * biggest width` and takes
  the found cell with the nearest centre (ties: fixed cell order); result `min(level + 1, smallest_size_level)`, or
  `smallest_size_level` when nothing is found. **(A17, batched-run buffers; amended, reconcile small fix a)** The
  step passes `committed_grid.tables(RunOn::kCpu)`: in a CPU run the real tables, probed directly; in a batched GPU
  run `step::FindCpuGridCopy(simulation)->tables(RunOn::kCpu)`, the real tables of the step's CPU copy, probed
  directly (no copy made or counted; cell resize holds no CPU grid there, `GridCopies::kHeldByStep`); in a one-piece
  GPU run version 0 with null arrays: `ChooseCellSize` then probes the CPU copy of the committed grid this
  `CellResize` holds (Task 24's `ReadGrid` copy, or its CPU resized grid after a committed resize), checking its
  version against the step's committed version; with no current copy it makes one with `grid::ReadGrid` and counts
  it in the next `ResizeCellsReport` (`bytes_copied`, `copy_count`). `MoveGridOnlyLiquidInStep` first copies the try's three amounts and
  ingredient volumes into the `amounts at step start` row, then calls `MoveGridOnlyLiquid` writing the try copy; it
  stores the try's `largest_speed_in_cell_widths`, which `ReadLargestGridOnlySpeed` returns (the step reads it only
  in `CommitStep`, where the latest try is the committed one; the docstring says so). Mapping: `kNotFinite` ->
  `kValueNotFinite`; `kGridOnlyCellDrained`, `kGridOnlyCellTooFull` -> `kParticleTravelledTooFar` with magnitude and
  limit as filled shares; others -> `kStageFailed`; context `module = Module::kCellResize`.
- Test support (plan decision): `step::StepCallbacks BuildCellResizeCallbacks(CellResize& cell_resize)` (the five
  pairs, `read_largest_grid_only_speed` as a wrapper around `ReadLargestGridOnlySpeed`, as I/O will set them), and
  `struct CellResizeTestRun { std::unique_ptr<CellResize> cell_resize; step::Simulation simulation; };`
  `std::expected<CellResizeTestRun, step::StepFailure> BuildCellResizeTestRun(const step::TestScene& scene,
  const CellResizeSettings& settings)`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `ApplyGridOnlyRegionInStep, OnlyTheLiquidSetReceives`
  (`BuildTwoLiquidsDeepScene`: snow set's arrays bytes unchanged); `ApplyGridOnlyRegionInStep, NaN_IsValueNotFinite`;
  `MoveGridOnlyLiquidInStep, DrainedCell_IsShortenStep` (design 7 "Grid-only move", step part: `kParticleTravelledTooFar`,
  kind `kShortenStep`; the half step commits); `MoveGridOnlyLiquidInStep, SpeedStoredAtCommit`
  (`CommittedStepLimits::grid_only_speed == ReadLargestGridOnlySpeed`); `ChooseCellSize, OneLevelSmallerThanNearest`;
  `ChooseCellSize, CappedAtSmallestLevel`; `AdvanceTo, CellResizeCallbacks_StepThenResize` (`kResizeCells` recorded);
  `ChooseCellSize, NullCommittedTables_UsesTheHeldCopy` (A17; reconcile small fix a: one-piece GPU run only, tables
  version 0 with null arrays: same level as with CPU tables; a stale copy is replaced once, counted: `copy_count` 1,
  `bytes_copied` 456 * blocks); `ChooseCellSize, BatchedGpuRun_ProbesTheStepsCpuCopy` (reconcile small fix a: given
  `FindCpuGridCopy(simulation)->tables(RunOn::kCpu)`: same level as with CPU tables, `copy_count` 0,
  `bytes_copied` 0, no `ReadGrid` call).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Connect the cell resize to the step's callbacks`.

### Task 30: Scene tests, rest, region cycle and border refill

**Kind:** integration

**Files:**
- Create: `native/fluidity/cell_resize/cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`, runs from `BuildCellResizeTestRun`, steps through `step::AdvanceTo`):

  ```cpp
  TEST(CellResizeScene, PressureAtRestAcrossAResize)   // design 7: BuildMixedSizeColumnScene, merge_wait 0; resize
      // with surface reach 4, then a CellResize with reach 5 (one more layer splits), then reach 4 (it merges), one
      // step after each: largest |velocity_after_pressure| <= max_volume_error_rate * 0.025 (pressure's bound)
  TEST(CellResizeScene, RegionCycle_TotalsKeptEveryResize)  // design 7 "Region cycle": BuildDeepTankScene; 100 times:
      // one TryStep, then ResizeCells directly with a CellResize of depths 6/7, then one of 14/15 (the surface moving
      // through the two depths, relative to the cells): mass, rest volume, current volume, each ingredient volume of
      // particles plus records within kAmountTolerance per resize; no particle in a grid-only cell after a resize
  TEST(CellResizeScene, BorderRefill_EveryBorderEighthHoldsAParticle)  // design 7: BuildRisingFloorScene, 200 steps:
      // after each resize every cell eighth of a particle-region cell touching a grid-only cell holds a particle;
      // excess below w^3/8 per face quarter; totals within kAmountTolerance
  ```
- [ ] **Step 2: Run.** If one fails, fix the owning commit (rebase), not the test. **Step 3:** gate. **Commit.**
  Subject: `Test resizes and the two regions through whole steps`.

### Task 31: Scene tests, absorb, foreign particles, material sets and gas

**Kind:** integration

**Files:**
- Modify: `cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`):

  ```cpp
  TEST(CellResizeScene, Absorb_ParticleTakenIntoRecord)   // design 7: BuildDeepTankScene after its first resize, a
      // water particle added by a source aimed into a grid-only cell: removed at the next resize; the record grew by
      // its amounts within 1 rounding step
  TEST(CellResizeScene, ForeignParticle_CellReturnsToParticles)  // design 7: BuildTwoLiquidsDeepScene: a snow particle,
      // then an oil particle sent into a water grid-only cell: the cell is particle region after the next resize; no
      // record holds two materials; each material's totals within kAmountTolerance
  TEST(CellResizeScene, TwoLiquidsMaterialSets)           // design 7 "Two liquids, material sets": 50 steps; records
      // reach only the liquid set's arrays; the snow set's untouched; no record of two materials; cells along the
      // contact stay particle region; MaterialSetKey checked each step
  TEST(CellResizeScene, SmokeOverMovingWater)             // design 7 "Gas beside liquid": BuildSmokeOverDeepWaterScene,
      // resize every step, 40 steps: gas amount within kAmountTolerance per resize; no grid-only cell touches a gas
      // cell; no marker removed kIntoGridOnlyRegion; ComputeLiquidTotals with the liquid set leaves markers out
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Test records against other materials and gas`.

### Task 32: Scene tests, the full-particle comparison and the cut test with resizes

**Kind:** integration

**Files:**
- Modify: `cell_resize_scene_test.cc`

- [ ] **Step 1: Tests** (`LABEL cpu`):

  ```cpp
  TEST(CellResizeScene, StandingWave_GridOnlyRegionMatchesParticles)  // design 7 "Full-particle comparison":
      // BuildStandingWaveTankScene with the region (6/7) and without (depths 1e9/1e9 + 1), same cell settings, 2 s:
      // surface height at x = 0.0125 within 0.025 m at every 0.1 s; kinetic energy (particles, plus per grid-only cell
      // 0.5 mass |area-weighted face-mean velocity|^2) within 5 % of the largest value of the run without
      // the region
  TEST(Cut, PlateThroughBlock_SidesNeverMixWithResize)  // handed by the step plan: the split-and-merge part of
      // Cut, PlateThroughBlock_SidesNeverMix: BuildSnowAndJellyScene's plate cut, cell resize on with the water beside
      // the jelly; the run restarted from a snapshot after the first hit, and with a rejected try before it: bytes
      // equal to the uninterrupted run; no jelly particle split, merged or changes side; driven_work == elastic
      // energy gained + cut_work + kinetic energy within 5 %
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject:
  `Compare the grid-only region with particles and cut beside resized water`.

### Task 33: Part A on the GPU, scene runs

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the scene runs pass their snapshots to `WriteResultDigest` so the hardware repeat run covers them;
CPU-against-GPU values within design 6.

**Files:**
- Create: `native/fluidity/cell_resize/cell_resize_scene_gpu_test.cc`

- [ ] **Step 1: Tests.** `LABEL gpu`: every row of Tasks 30 to 32 with `RunOn::kGpu` (design 7 "Every row on
  `RunOn::kGpu`"), against the CPU within design 6 for one resize and for the stated scene tolerances over the run;
  `CellResizeScene, OnGpuTwice_GivesIdenticalBytes` (deep tank, 20 steps); each test writes the digest of its final
  snapshot with `WriteResultDigest`. `LABEL no_gpu`: `BuildCellResizeTestRun` with `kGpu` gives `kRunOnCallFailed`.
  CMake: `LINKS fluidity_cell_resize_test_support fluidity_base_test_support`.
- [ ] **Step 2:** gate. **Commit.** Subject: `List the cell resize's GPU scene runs`. Body: `CUDA: compiled, not run`.

---

# Part B: the batched run

### Task 34: Method paper, batches

**Amended 2026-11-14 (A15):** the solves are no longer "not batched": the text gives the three ways to run and the
planner's choice; pressure's `<sec-solve-in-batches>` holds the solve itself.

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
    // kFindLiquidSurface: FindMaterialBoundaries, FindAirPockets, CheckViscosityCanRun. kMoveParticles: the
    // move_grid_only_liquid callback. kCheckStep: CheckSolidExchange, the gas changes, the next
    // CommittedStepLimits. Null for every other stage. A one-piece run calls it right after `run`.
    std::array<StageFunction, kStageCount> run_whole_grid{};
  };
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `StageTable, WholeGridPartsOnlyForStages3_9_16_19` (A17: was
  `..._9_16_19`); `FindSolidSurfaces, OpenToSkyOnWholeGridEqualsOnePiece` (A17: a lid over two ranges cut through
  the pocket under it: the whole part's marks equal the one-piece stage's bytes, and a range-only call would differ);
  `TryStepWithStages, WholeGridPartRunsAfterRangePart` (`CallbackLog` order); `TryStepWithStages,
  RejectAfterWholeGridPart_ChangesNothing` (stages 9, 16, 19). The step's existing `TryStepWithStages,
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
  pass 1, then 9's, 16's right after pass 2, 19's after stage 17.

- [ ] **Step 1: Failing tests:** `ToName, EveryBatchesError_IsLowerSnakeCaseWithoutK` (`kBatchDoesNotFit` ->
  `"batch_does_not_fit"`); `kStagePlaces, MatchesThePlanTable` (all 22); `kStagePlaces,
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

**Kind:** numerics

**Files:**
- Modify: `batch_plan.h`, `batch_plan.cc`; Create: `batch_plan_test.cc`

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
     cell and face rows, `grid::BuildGridMemoryEstimate(run_on, 1, grid::BuilderBuffersSharing::kOneForAllGrids)` without
     its `builder buffers:` entries, since the loaded try's buffers borrow the step's `grid::GridBuilderBuffers`; reconcile item 11)
     at the counts of one cell block and no particle;
     `bytes_per_loaded_particle` the same at one particle and no block; `particles_per_block = ceil(512 *
     start_limit_headroom)` per material (8 per cell, full block).
  3. `max_loaded_blocks = min((allowed - whole) / (per block + particles_per_block * per particle),
     limits.max_cell_blocks)`; `max_loaded_particles = max_loaded_blocks * particles_per_block`.
  4. Blocks are ordered `z` first, so a range is a slab: `layer = (domain cells along x / 4) * (along y / 4)` at the
     smallest width; `T = floor(max_loaded_blocks / layer) - 8`; `max_blocks_per_batch = T * layer`;
     `kBatchDoesNotFit` when `T < 1` or `max_blocks_per_batch < min_blocks_per_batch`. The exact cut is found per try
     in Task 40. `range_share_of_loaded = T / (T + 8)`; `batch_count = ceil(start blocks / max_blocks_per_batch)`;
     `bytes_copied_per_try = (2 / range_share_of_loaded + 2) * start blocks * (per block + particles_per_block *
     per particle)`.
  5. (A15) CPU: every row with `where = kCpu` (the particles, the per-try arrays only particle stages read, and in
     `kInPieces` the solvers' working buffers) at most `plan.allowed_cpu_bytes`, else `kStateDoesNotFitCpu`.
  6. (A15) Report, `BatchPlan` gains: `max_solve_loaded_blocks`, `solve_batch_count` (start grid, bisection as
     `pressure::ApplySolveBatches` would cut it), `bytes_on_processor` and `bytes_on_cpu` (the memory budgets: sums
     of the rows by `where`), `pass_loads_per_try = 2 * batch_count`, `solve_loads_per_iteration` (3, or 4 when the
     scene can hold a sealed blob: a closed domain, times `solve_batch_count`), `solve_bytes_per_iteration` (that
     many sweeps of the streamed rows of every loaded block). Zero with `kWholeGridOnProcessor`.
  `BuildBatchesMemoryEstimate(plan)`: design 3.7, rows `where = kRunOn`, for reporting; the step's batched layout
  (built from the step's own rows) is not counted twice. Amended (Foundation as built): it also lists `"batches: sum
  buffers"` (`kCellBlock`, `ComputeSumBuffersSize(plan's run_on).bytes_per_item`) and `"batches: sum buffers fixed
  bytes"` (`kGivenCount`, `fixed_bytes` items of 1 byte), `kWholeRun`, `where = kRunOn`, which `BuildBatchedRun`
  allocates.
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
  `step::BuildSimulation` on the CPU with a forced batched layout: `ReadMemoryReport` `allocated_bytes` and
  `allocated_cpu_bytes` equal `bytes_on_processor` and `bytes_on_cpu - estimate.snapshots`);
  `BuildBatchPlan, LoadedRowsIncludeModuleInstances` (A17: `bytes_per_loaded_particle` exceeds the particle rows by
  the per-particle rows of the four loaded module instances, exactly);
  `BuildBatchPlan, LayerWiderThanABatch_IsBatchDoesNotFit`; `BuildBatchPlan, MoreMaterialSets_FewerLoadedBlocks` (1
  and 3 sets); `BuildBatchPlan, RunOnGpuWithoutAGpu_StillPlans` (all design 7 "Batch plan");
  `BuildBatchesMemoryEstimate, LoadedRowsAtMaxLoadedBlocks`; `BuildBatchesMemoryEstimate, SumBuffersEntriesMatchTheBuffers`
  (amended: the two entries equal `batched_run.sum_buffers.byte_count()` at the limits).
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Plan the batches of a scene that does not fit the GPU`.

### Task 39: `BatchedRun`, `WriteBatch` and `ReadBatch`

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
- Produces design 3.7 `BatchReport`, `BatchedRun`, `BuildBatchedRun`, `ReadLastReport`, `WriteBatch`, `ReadBatch`.
- **Plan decisions:** (amended, Foundation as built) `BatchedRun` also holds `SumBuffers sum_buffers`
  (`BuildSumBuffers(run_on, limits.max_cell_blocks)`, the two entries of Task 38) for `ComputeSumOverBlocks`.
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
      // BuildCellResizeCallbacks plus run_stages, 20 steps (ApplyGridOnlyRegionInStep per batch, the move whole grid)
  ```
- [ ] **Step 2: Run**; fix owning commits if needed. **Step 3:** gate. **Commit.** Subject:
  `Test solids, gas and resized cells in batches`.

### Task 43: Batched runs on the GPU, and the resize beside them

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
  `has_pressure`, `pocket_of_cell`, the seven region arrays; `CellGeometry` and `FacePieceGeometry` of the last
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
      // when SolvePlace is forced to kWholeGridOnProcessor); BuildSimulation allocates the estimate (kRunOn and kCpu
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

**Kind:** mechanical

**Files:**
- Modify: any file of `native/fluidity/cell_resize/` and `native/fluidity/batches/`, the step files of Tasks 1, 35,
  36, `docs/method-paper/sections/cell_resize.typ`, `docs/glossary.md`, `docs/gpu-runs.md`,
  `tools/check_rules_allowed_names.txt`

- [ ] **Step 1: Comment pass.** Reread every file of both folders as a C++ programmer who has never read a fluid
  paper: comments true of the final code; why, not what; a worked example with numbers for `ComputeSplitAmounts`
  (the linear case of Task 5), `ComputeInnerFaceFlows` (Task 6's outflows), `ResizeCells` (the deep tank's first
  resize: how many cells split, merge, become grid-only) and `BuildBatchPlan` (A15: the three-way choice with the
  budgets and copies of a worked scene); every word in the glossary; `Uses` /
  `Used by` lists; `// Step N:` numbers match `<proc-resize-cells>`, `<proc-find-cells-that-must-split>`,
  `<proc-apply-region-transfer>`, `<proc-apply-stages-in-batches>`.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently:
  `ApplyResizeToCellAmounts` (one limiter for three amounts); a durations-only change commits a same-cell resize and
  the wait grows by elapsed committed time; `FindCellsThatMustSplit`'s `max_cell_count` rule; the private resize plan,
  facts pass, `RegionWishFacts`, `PieceExchange`; the GPU plan reading per-cell arrays back instead of compacted
  ids; `BuildBatchPlan` cutting by layers (no start grid), the per-try bisection; stage 10 whole in a batched run;
  the step's `run_whole_grid`, `FindLoadedTry`, `FindLoadedGridBuffers`, `ApplyGridPiece`, `BlockCopy`,
  `WriteStepArrayBlocks`; the CPU mirrors of a batched resize; the test scenes of Task 1; A15: solves in pieces
  (`SolvePlace::kInPieces`, no CPU fallback), `solve_piece_share`, the plan's budgets and copies, the replaced byte
  tests; A17: the buffer-form `FindStackedBorder`, the piece map read back per load, the `ReadGrid` copy and
  `grid_run_on` of the resize; A17 batched-run buffers (amended 2026-11-14): `ResizeCellsReport::copy_count`,
  `ChooseCellSize` on a held CPU copy, stage 3's whole part, `kStagePlaces` against the step's array places, the
  planner on the step's batched rows, `WriteBlockPartialsRange` in `ReadBatch`, the piece map handed to
  `ApplyGridPiece`, the resize reading `FindCpuGridCopy`; Foundation as built: `CellResize::sum_buffers`,
  `order_buffers` and the changed flag, `BatchedRun::sum_buffers`, the one-form (A31) `ComputeLiquidTotals`
  and `ComputeSumOverBlocks` taking `SumBuffers`, `ToCellResizeFailure`, the shared `*_steps.h` headers and their `cpu` tests, the Jacobi
  passes of the depth below the surface.
- [ ] **Step 4: `docs/gpu-runs.md`:** rows `cell resize` and `batches`, `compiled, not run`, listing every `gpu` test
  of Tasks 12, 13, 17, 20, 22, 23, 28, 33 and 43, and the `cpu` steps tests that stand for them until the run.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate's output, new public names
  with reasons, the plan decisions, and the statement that no device code has run.
