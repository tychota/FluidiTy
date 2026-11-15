# Rebuild Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `native/fluidity/grid/`, the mixed-size staggered grid every later module computes on: cell and
face identity, cell blocks and their tables built into storage that is allocated once, the weight curve and the
nearby-face weights, ranges with their border, grid pieces for the batched run, and the per-place value buffers.

**Architecture:** Space is cut into cubic cells of up to 16 size levels (each half the previous), stored in
blocks of 4 x 4 x 4 cell places with 240 face places and 125 corner places each. `BuildGrid` fills a
`GridBuffers` (allocated once, at a block limit) on the processor the run uses: a run is all CPU or all GPU
(owner answer A17). In a GPU run the cell list already lives on the GPU, every table is built there (sort, then
mark, prefix-sum and scatter), and only a small status record comes back; no table is copied in either
direction. The CPU builder serves CPU runs and is the oracle: GPU tables are compared with it bit for bit. Both
builders call the same per-cell and per-face formulas in a private shared header. Everything a stage needs on
either processor (addressing, geometry, nearby faces and their weights) lives in four shared headers that read
the raw `GridTables`. A test-only geometric checker that knows nothing of blocks backs every "random grids" test.

**Tech Stack:** As Foundation: C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest),
uv, clang-format and clang-tidy 23.1 (the versions `pyproject.toml` pins), Docker via colima for the CUDA compile, Typst + CeTZ for the method paper.

**Spec:** `docs/superpowers/specs/modules/02-grid-design.md` (the grid design; in the `main` checkout, untracked).
Its section 3 is the contract. Programme design: `docs/design/programme.md` on `rebuild` (sections 6, 7, 8, 8a).
Cross-module names: `docs/superpowers/specs/modules/shared-interface.md` and `owner-answers.md` (`main` checkout).

## Global Constraints

Copied from the Foundation plan:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes the
  `AGENTS.md` checklist; new public names listed in the commit message with a one-line reason; tests arrive with
  the code. A commit must pass every part of the gate that exists at that commit.
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure patterns apply to every task (`AGENTS.md`
  "Structure"; `specs/modules/plan-decisions.md` "Foundation after the final review"). Two halves: the public
  function lives in its `.cc` file (checks, `RunOn` dispatch, the CPU loop, and the GPU sequence under
  `#if defined(FLUIDITY_HAS_GPU)`); each GPU half `<name>_gpu.cu` is declared in the private `<name>_gpu.h` (plain
  pointers and counts, no CUDA header, no CPU-only header), returns `RunOnCallStatus`, which the CPU half maps with
  `ToBufferError` and then into `GridFailure` through the module's one `BufferError` mapping (below), and launches,
  waits and holds temporary GPU memory only through `base/gpu_launch.cuh` (`ComputeLaunchSize`, `ApplyGpuFunction`,
  `CheckGpuWork`, `GpuArray`). Every GPU function is a thin grid-stride loop over `FLUIDITY_CPU_AND_GPU` per-item
  functions of a shared header (`grid_build_steps.h`, `border_block_steps.h`), plus at most an integer atomic of the
  value they return, and `cpu` tests run those functions for every thread block and thread of the launch, in
  launch order and in reverse, against the CPU builder: the main evidence until the hardware run. Scratch: the
  build scratch is `GridScratch` (with `BuildGridScratch` and `ComputeGridScratchSize`, Task 6), which holds the
  `OrderScratch` that every `BuildStableOrder` and `ComputeExclusivePrefixSum` call of a build takes (scratch forms
  only); border search keeps its own in `BorderSearchBuffers`. Errors: a failed check writes nothing; after
  `kRunOnCallFailed` the outputs are unspecified and the run stops; `kOutOfMemory` comes only from building forms.
  Exactness: discrete outputs (every table, count and failure) equal on CPU and GPU; floating-point results within
  the CPU-versus-GPU tolerance below unless a header promises the same bits unless NaN. Tests link
  `fluidity_base_test_support` and `fluidity_grid_test_helpers` (made with `fluidity_add_test_library`, Task 5) in
  `LINKS`; every `gpu` test passes its main outputs to `WriteResultDigest`. Labels: `cpu` everywhere, `no_gpu` only
  for behaviour on a machine without a GPU, `gpu` only for tests that need hardware. Every target is created
  through the helpers of `cmake/fluidity_targets.cmake` and every source file is listed in a CMake file. The GPU
  sense of "block" is always written "thread block"; "cell block" is the grid's own word.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the
  release presets, and the CUDA build compiled in the container (its `cuda` step fails, naming the file, for a
  tracked `.cc` or `.cu` file that no build compiles). clang-format and clang-tidy are the pinned 23.1.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages end with two lines: `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>` and
  `Claude-Session: <session link>`.
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. Never claim otherwise.
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

Grid's own:

- Namespace `fluidity::grid`, folder `native/fluidity/grid/`. Shared (S) headers: `grid_ids.h`, `grid_error.h`
  (plan decision, Task 2), `grid_tables.h`, `weight_curve.h`, `nearby_face_weights.h`, and the private shared
  `grid_build_status.h` (Task 6), `grid_build_steps.h` (Task 7), `border_block_steps.h` (Task 22). Every
  function in them is `constexpr` or `inline`, `FLUIDITY_CPU_AND_GPU`, `noexcept`, allocates nothing, uses
  `math::` from `base/scalar_math.h` (never `<cmath>`), and sums in list order with plain doubles.
- Constants, exactly: `kMaxSizeLevels = 16`, `kBlockWidthInCells = 4`, `kCellPlacesPerBlock = 64`,
  `kFacePlacesPerBlock = 240`, `kCornerPlacesPerBlock = 125`, `kMaxCoord = (1 << 18) - 1`,
  `kNoIndex = 0xFFFFFFFF`, `kWeightCurveReach = 1.5`, `kMaxNearbyFaces = 76`, `kMinWeightSum = 0.1`,
  `kMinPivotShare = 1e-12`, `kMaxCellsNearFace = 112`, `kMaxFaceSearchRadius = 4.0`,
  `GridSettings::max_cell_blocks = 0` (**plan decision, reconcile item 11**, replaces the design's `1u << 22`:
  no compile-time room is ever an allocation size; `BuildGridBuffers` and `BuildGridScratch` with 0 are
  `kBadInput`, and the allocating `BuildGrid(inputs, settings)` with 0 sizes its buffers to exactly the blocks of
  the list, `ComputeGridCounts(inputs).cell_blocks`, at least 1). **Plan decision (A17):** `max_cell_blocks` above
  `kMaxCellBlocksLimit = (1u << 26) - 1` is `kBadInput` in `BuildGridBuffers`, so every cell-place count
  (`64 * max_cell_blocks`) fits the `std::uint32_t` orders and prefix sums of `base/`.
- **Build scratch shared between buffers (plan decision, reconcile item 11; `plan-decisions.md` "Reconcile pass
  (2026-11-14)", "Grid buffers sized by the planner").** The `grid: build scratch: ...` entries live in a separate
  move-only `GridScratch` that the caller owns and lends to every `GridBuffers` built with
  `BuildGridBuffers(settings, scratch)`; buffers built with `BuildGridBuffers(settings)` own one as before. No
  operation on a built grid reads its buffers' build scratch, so two buffers that hold two live grids may share one
  scratch, used by one build or check at a time (the step's committed and try grids: 2 x tables + 1 x scratch).
- Tolerances, exactly (design section 7): discrete results (ids, counts, kinds, indexes, areas that are powers of
  two) are compared exactly. Weight curve: `|sum - 1| <= 4e-16`, slope sum `<= 4e-16`. Complete case: `W = 1`,
  `mean_offset = 0`, `weighted_spread = h^2/4 I` within `1e-14 h^2`. Any kept set: `|sum weight - 1| <= 1e-14`,
  `|sum weight_change| <= 1e-13 / h`. Linear field: value within `1e-13 (|a| + |b| h)`, fitted row within
  `1e-11 |b|`. Finite difference: step `1e-6 h`, within `1e-7 / h`. Comparison with `main`: `1e-13` relative.
  CPU versus GPU: `|difference| <= 1e-15 + 1e-13 |value|` (offsets: times `h`). Repeat runs: identical bytes.
- No allocation, on either processor, in `BuildGrid(inputs, buffers)`, the buffers forms of `ComputeGridCounts`
  and `CheckSizeStepRule`, both `BuildGridPiece` forms, `ReleaseBuffers`, `ApplyGridVersion`, the span form of
  `FindBorderBlocks` and its buffer form. The only allocations are `BuildGridBuffers`, `BuildBorderSearchBuffers`,
  the `Build...Values` forms, and the forms the design marks "allocates; for tests, tools and planning".
- **One processor per run (A17).** Every input list, output list and scratch of a grid operation lives on the
  grid's (or the buffers') `run_on()`; a list on the other processor is `kBadInput`, never copied. A grid
  operation on the GPU returns to the CPU only its `GridBuildStatus` record (Task 7) and, for border search, a
  count. A GPU grid keeps no CPU copy of its tables; `tables(RunOn::kCpu)` on it returns version 0 and null
  arrays, and `ReadGrid` (Task 11) is the one explicit copy, for tests and tools.
- **The CPU is the oracle.** Grid tables, counts and failures built on the GPU equal the CPU's byte for byte;
  there is no tolerance for them. The GPU's only atomics are integer `atomicMin` (first failure) and
  `atomicAdd` (counts), whose results do not depend on thread timing.
- The grid never looks at physics. Which faces a particle may use is the caller's kept mask (material sets, A8;
  plate sides; `outside_solid`). The grid owns no physical field and has no identifier containing `owner`.
- Fixed order everywhere: blocks (and, plan decision, cells) by low corner in level-15 cell units by `z`, then `y`,
  then `x`, then bigger level first. A reported failure is the first in that order: precisely, the smallest
  `(step, position)` failure key of Task 7, on both processors.
- Every `std::expected` of this module has `GridFailure` as its error. `BufferError::kOutOfMemory` becomes
  `GridError::kOutOfMemory`, `kRunOnCallFailed` becomes `kRunOnCallFailed`, and `kSizeMismatch` (a span of the
  wrong size) becomes `kBadInput`.
- Random tests use `std::mt19937_64` with the seed written in the test and map its raw 64-bit output to numbers
  with the test-support helpers of Task 5 (never `std::uniform_*_distribution`, whose output differs between
  standard libraries).
- Comparisons with the proof of concept rotate Y-up to Z-up with `new = {old.x, -old.z, old.y}`: old cell `z = k`
  becomes `y = -k - 1`; old Z-faces become Y-faces with `cell_low`/`cell_high` exchanged.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`: the whole Foundation plan
(including `base/` Tasks 12 to 16: `scalar_math.h`, `buffer.h`, `tree_sum.h`, `memory_count.h`, `run_on.h`,
`cpu_and_gpu.h`, `cpu_only.h`, `vec3.h`, `mat3.h`, `axis.h`, `ordered_sums.h`, `stable_order.h`), the method
paper with `figure_style.typ` and `pseudocode.typ`, and the contact module. The grid includes only `base/`.

**Needed from Foundation before Task 6 (A17):** `base/stable_order.h` must also provide `OrderScratch`,
`BuildOrderScratch`, `ComputeOrderScratchSize`, the `BuildStableOrder` form that takes an `OrderScratch&` and
allocates nothing, and `ComputeExclusivePrefixSum` (declarations in `plan-decisions.md`, "Grid built on the GPU
(A17)"). **Amended 2026-11-14 (Foundation as built):** Foundation as built provides them, and also `base/run_on_call.h`
(`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, the scratch forms of the sums (`SumScratch`) and
of `BuildGroupsByDestination`, `base/test_support.h` (library `fluidity_base_test_support`) and
`fluidity_add_test_library`; read `AGENTS.md` "Structure" before the first task. Prove it:

```sh
uv run python tools/check_commit.py
```

Expected: seven `PASS` lines (format, rules, cpu debug, cpu release, tidy, paper, cuda). Read how contact added its sources and tests to the build (its commit touching
`CMakeLists.txt`) and add the grid's the same way in every task below ("register in the build").

## Review Focus

1. **A position one rounding step from a cell side at a size change.** `FindCellAt` must give the same cell from
   the binary search and from `near`, and the containing cell must contain the position by the one floor formula.
   Pinned: `FindCellAt, NearAndSearch_AgreeOnRandomPositions` and `..., OnHighSide_BelongsToNext` (Task 13).
2. **A face piece stored one block beyond a touching cell** (plane 4 moved to plane 0 of the high block). A border
   that only lists blocks of touching cells misses it. Pinned:
   `FindBorderBlocks, PlaneFourPieceBeyondTouchingCell_IsInBorder` (Task 22).
3. **A failed build into used buffers**, on either processor. The buffers must stay usable and the next build
   must not read stale entries past `block_count` (on the GPU: no kernel of a later step may run on a failed
   earlier step's data). Pinned: `BuildGrid, AboveStorageRoom_IsTooManyCellBlocks` and
   `BuildGrid, IntoStorage_AllocatesNothing` (Task 9), `BuildGrid, OnGpuFailedBuildThenRebuild_EqualsFreshBuffers`
   (Task 11).
4. **A piece whose border block's high neighbour exists in the whole grid but was not loaded.** Plane 4 must stay
   `kNotAPiece` and the other side become `kOutsidePiece`, or a piece appears that the whole grid lacks. Pinned:
   `BuildGridPiece, KeepsWholeGridOrder` (Task 24) and `BuildGridPiece, OnGpu_EqualsCpuPiece` (Task 25).
5. **The face-seen weight drifting one bit from the particle-seen weight**, which would break transfer's
   gather/scatter symmetry. Pinned: `ComputeFaceWeight, SeenFromFace_IsBitIdentical` (Task 19).
6. **(A17) The GPU reporting a different first failure than the CPU** when a list breaks several rules at once
   (a repeated cell after a missing sibling in fixed order, say). The rule is "earliest step, then smallest
   position" on both. Pinned: `CheckSizeStepRule, OnGpuSeveralFailures_SameFirstFailureAsCpu` (Task 10).

---

### Task 1: Method paper section for the grid

**Amended 2026-11-14 (A17):** adds the subsection, figure and pseudocode on building the tables in parallel on
the GPU, with the CPU builder as the oracle.

**Kind:** mechanical

Load the `typesetting-with-typst`, `drawing-with-cetz` and `typesetting-documents` skills first.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ`, `docs/method-paper/figure_style.typ`,
  `docs/method-paper/references.typ` (nothing removed; Harlow and Welch is not added, decision 11)
- Create: `docs/method-paper/figures/staggered_grid.typ`, `figures/face_pieces.typ`, `figures/nearby_faces.typ`,
  `figures/parallel_build.typ`

**Interfaces:**
- Produces labels: `<sec-grid>`, `<sec-cell-sizes>`, `<sec-face-pieces>`, `<sec-cell-blocks>`,
  `<sec-nearby-faces>`, `<eq:weight-curve>`, `<eq:scaled-weights>`, `<eq:weighted-spread>`, `<fig-staggered-grid>`,
  `<fig-face-pieces>`, `<fig-nearby-faces>`, `<proc-build-grid>`, `<proc-nearby-face-weights>`,
  `<sec-parallel-build>`, `<fig-parallel-build>`, `<proc-build-grid-in-parallel>`.
- Produces `cell-grid(origin, cells, ...)` in `figure_style.typ`: draws a 2D slice of square cells of mixed sizes
  from a list of `(x, y, size_level)`, with optional face arrows and centre dots. Every later grid figure uses it.

- [ ] **Step 1: Text.** `= The grid <sec-grid>` with the outline of design section 8: 1 where values live (cell
  centres, faces, `+axis` orientation, `cell_low`/`cell_high`); 2 `<sec-cell-sizes>` size levels (0 is biggest),
  sibling cells exist together, the size-step rule and why one level; 3 `<sec-face-pieces>` one identity, one area,
  stored with the smaller cell, plane 4 versus plane 0; 4 `<sec-cell-blocks>` blocks, the fixed order, id versus
  index, grid versions, storage built once, ranges and border, grid pieces; 5 the weight curve with
  `<eq:weight-curve>` (design 5.3); 6 `<sec-nearby-faces>` candidates, kept mask, `<eq:scaled-weights>` (design 5.4
  steps 4 to 6), mean, `<eq:weighted-spread>`, the three fit outcomes, the complete case as a special case, and the
  stated properties (constants always reproduced, linear fields when `kFull`, weights positive, weights jump when
  a particle changes cell across a size change); 7 memory per block (the table of design 3.6, with the GPU build
  rows of Task 6); 8 `<sec-parallel-build>` building the tables in parallel (A17): a run is all CPU or all GPU,
  so in a GPU run the cell list never leaves the GPU and only a status record comes back; each sequential step
  of `<proc-build-grid>` becomes "compute a key, stable sort, mark, prefix-sum, scatter" (one paragraph with
  the block-index example: blocks `[A, A, B, C, C, C]` in sorted order give marks `[1, 0, 1, 1, 0, 0]`, exclusive
  sums `[0, 1, 1, 2, 3, 3]`, so the first entry of each run writes its block at index 0, 1, 2); the fixed order
  is kept because the sort keys are the fixed order itself; the first failure is the smallest
  `(step, position)` pair, found with an integer minimum, so it does not depend on thread timing; why the tables
  are then identical to the CPU's (each entry is a function of the sorted list, not of the order work was done).
- [ ] **Step 2: Citations.** [2] Ding 2020, [1] Jiang 2015, [3] Hu 2018, [18] Stomakhin 2014, [5] Wang 2020, [15]
  Fei 2021, [6] Wang 2026, by their numbers in `references.typ`; Harlow and Welch 1965 by author and year only.
- [ ] **Step 3: Figures** as design section 8 describes, each drawn with `figure_style.typ` and `cell-grid`, each
  with a caption stating the one thing to see. `<fig-nearby-faces>`: kept faces as filled dots sized by weight,
  left-out faces hollow, the mean offset arrow, the spread ellipse, beside the complete 3 x 3 case.
  `<fig-parallel-build>`: a 2D slice of six cells of two sizes (drawn with `cell-grid`) above four rows of boxes
  — the cells' block keys in input order, after the stable sort, the marks, the exclusive sums — with an arrow
  from each marked entry to its block index; caption: "the first entry of each run writes its block; the sums
  give every block its index in the fixed order".
- [ ] **Step 4: Pseudocode** with `procedure(...)`: `<proc-build-grid>` with the 9 steps of design 5.1 item 8;
  `<proc-build-grid-in-parallel>` with the 12 parallel steps of Tasks 10 and 11 (P1 input checks on the CPU, P2
  per-cell id checks, P3 sort into fixed cell order, P4 overlaps, P5 siblings, P6 size-step rule, P7 add split
  cells, P8 sort cells by block and number the blocks, P9 neighbour tables, P10 cell states, P11 face piece
  kinds, P12 counts and status), each line naming its one-thread-per-what; and `<proc-nearby-face-weights>`
  with the 9 steps of design 5.4. Code comments `// Step N:` in Tasks 8, 9, 17 and 18 and `// Step PN:` in
  Tasks 10 and 11 follow these numbers.
- [ ] **Step 5: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`. Read
  every new page; fix overlaps and clipped labels. Run the gate.
- [ ] **Step 6: Commit.** Subject: `Describe the grid in the method paper`.

### Task 2: Grid errors

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/grid/grid_error.h` (S), `native/fluidity/grid/staggered_grid.h`,
  `native/fluidity/grid/staggered_grid_cpu.cc`, `native/fluidity/grid/grid_error_test.cc`
- Modify: build registration

**Interfaces:**
- Produces: `enum class GridError` with the twelve enumerators of design 3.3 in that order (**plan decision:** in
  the shared `grid_error.h`, because `FindNearbyFaces` in a shared header reports through `GridError*`);
  `GridFailure` (Task 3 adds its `CellId` fields once `grid_ids.h` exists: for this task, declare it with
  `error` only), `ToMessage(GridError)`, `ToName(GridError)` in `staggered_grid.h`.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ToName, EveryError_IsStableSnakeCase)`: for each of the 12 values, the name is lower snake case of the
    enumerator without `k`; all 12 distinct; `ToName(GridError::kTooManyCellBlocks) == "too_many_cell_blocks"`,
    `ToName(GridError::kNone) == "none"`.
  - `TEST(ToMessage, EveryError_HasADistinctNonEmptyMessage)`.
- [ ] **Step 2: Run** and see the compile fail. **Step 3: Implement.** Messages say what the user can do
  (`kTooManyCellBlocks`: "The grid needs more cell blocks than the limit. Raise max_cell_blocks or use fewer
  small cells."). **Step 4: Run** tests and the gate.
- [ ] **Step 5: Commit.** Subject: `Add the grid's errors`. New names: `GridError` (one enum for the module),
  `GridFailure`, `ToMessage`, `ToName`.

### Task 3: Cell, block and face identity

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/grid_ids.h` (S), `native/fluidity/grid/grid_ids_test.cc`
- Modify: `staggered_grid.h` (`GridFailure` gains `cell`, `other_cell`), `docs/glossary.md`

**Interfaces:**
- Consumes: `Axis` (`base/axis.h`).
- Produces: every declaration of design 3.1 and the `grid_ids.h` part of 3.7 (`kCornerPlacesPerBlock`,
  `CornerPlaceIndex`, both `ToCellBlockIndex`), exactly as written there.
- Produces also `struct CellBlockRange { std::uint32_t first_block; std::uint32_t block_count; };` (design 3.4),
  **declared here in the shared `grid_ids.h`** (coordinator decision, 2026-11-14): other modules' shared
  headers (particles) take it on either processor. `cell_block_range.h` (Task 22) keeps only the CPU-only
  functions and `RangeWithBorder`.
- **Plan decisions** (the design says only "packs"):
  - Valid cell coordinates are `-kMaxCoord..kMaxCoord` on each axis. A coordinate is stored as `c + kMaxCoord`
    in 19 bits.
  - `CellId`: bit 63 set in every valid id; bits 57..60 size level; bits 38..56 z, 19..37 y, 0..18 x.
    `CellId{0}` is never valid, so `GridFailure`'s "zero when unused" is unambiguous. `CellBlockId`: same layout
    with the block coordinate `floor(c / 4)`.
  - `FacePieceId`: bits 61..62 hold `axis + 1` (so 0 is never valid), 57..60 size level, coordinates as above.
    `cell_high_coord` may be `kMaxCoord + 1` along the normal axis (high side of the last cell); it still fits.
  - `ComesBefore`: compare `(z, y, x)` of the low corner scaled to level-15 units with `std::int64_t`
    (block coordinate `* 4 << (15 - level)`), then smaller level value first.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ToCellId, RoundTrip_AtCoordLimits)`: for levels 0 and 15 and every combination of `-kMaxCoord`, `0`,
    `kMaxCoord` per axis, `ToSizeLevel` and `ToCellCoord` give back the inputs exactly.
  - `TEST(ToFacePieceId, HighSideOfCellAtCoordLimit_RoundTrips)`: axis X, level 3, `cell_high_coord`
    `{kMaxCoord + 1, -kMaxCoord, 0}`; the id is non-zero and differs from the same with axis Y.
  - `TEST(ComputeBigCell, NegativeCoords_RoundTowardMinusInfinity)`: level 2 `{-1, -2, 3}` -> level 1 `{-1, -1, 1}`.
  - `TEST(ComputeSmallCell, EveryEighth_GivesBackItsBigCellAndEighth)`: for `cell_eighth` 0..7,
    `ComputeBigCell(ComputeSmallCell(c, e)) == c` and `ComputeCellEighth(...) == e`.
  - `TEST(ToCellBlockId, CellsOfOneBlock_ShareIt)`: `{-4..-1}` on each axis at level 1 give one block; `{0,..}` another.
  - `TEST(ComesBefore, BiggerBlockAtSameCorner_ComesFirst)` and `TEST(ComesBefore, LowerZ_ComesFirst)`.
  - `TEST(ToCellBlockIndex, PlaceIndexes_DivideByPlacesPerBlock)`: cell 130 -> 2; face 479 -> 1.
  - `TEST(CellBlockRange, IsTriviallyCopyable)`: `static_assert(std::is_trivially_copyable_v<CellBlockRange>)` and
    `sizeof(CellBlockRange) == 8`.
  Each `static_assert`s one case so `constexpr` is proven.
  Label `no_gpu`, file `native/fluidity/grid/grid_ids_no_gpu_test.cu`, built only with `FLUIDITY_ENABLE_CUDA`:
  a `__global__` kernel that takes a `CellBlockRange` by value plus a `CellPlaceIndex` and writes
  `ToCellBlockIndex(cell).value - range.first_block` to a pointer; it is never launched.
  `TEST(CellBlockRange, PassesToAGpuKernel_Compiles)` takes the kernel's address and checks it is not null, so the
  container's CUDA build proves the struct and `grid_ids.h` compile in device code.
- [ ] **Step 2: Run**, see failures. **Step 3: Implement.** File header: `Read:` `<sec-cell-sizes>`,
  `<sec-cell-blocks>`. Glossary rows: `place` ("a fixed position in a cell block that may or may not hold an active
  cell, a face piece or a corner"; never say `slot`), `cell_eighth`, `grid version`. **Step 4: Run** the gate.
- [ ] **Step 5: Commit.** Subject: `Add stable ids and per-version indexes for cells, blocks and faces`. List every
  new name of 3.1, the 3.7 additions and `CellBlockRange` (shared so stages on either processor take it).

### Task 4: The weight curve

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/weight_curve.h` (S), `native/fluidity/grid/weight_curve_test.cc`

**Interfaces:**
- Produces: `WeightCurveValue`, `ComputeWeightCurve`, `kWeightCurveReach`, `kMaxNearbyFaces`, `kMinWeightSum`,
  `kMinPivotShare` (design 3.5) and `ComputeCompleteWeightedSpread` (3.7), exactly.
- Port: `git show main:native/src/sparse_mac_grid_3d.cc` lines 285-296 (the three factors and derivatives) and
  `main:reference/src/fluidity_reference/basis.py` lines 22-23. Only the formula of design 5.3; none of the
  rounding machinery of lines 83-198 and 312-336.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeWeightCurve, AnyPosition_ThreeNeighboursSumToOne)`: 1,000 offsets `s` in `[-0.5, 0.5)` from seed
    `20261114`; `Q(s - 1) + Q(s) + Q(s + 1)` within `4e-16` of 1; slopes' sum within `4e-16` of 0.
  - `TEST(ComputeWeightCurve, AtAndBeyondReach_IsZero)`: `t = 1.5, -1.5, 2.0` give weight 0 and slope 0.
  - `TEST(ComputeWeightCurve, AtHalf_BothBranchesAgree)`: `t = 0.5`: weight `0.5`, slope `-1.0`, exact.
  - `TEST(ComputeCompleteWeightedSpread, WidthTwo_IsOne)`: exact; `static_assert`.
- [ ] **Step 2: Run**, see failures. **Step 3: Implement**; header comment gives `Q`, its slope and a worked
  example (`t = 0.25`: weight 0.6875, slope -0.5). `Read:` `<eq:weight-curve>`. **Step 4: Run** the gate.
- [ ] **Step 5: Commit.** Subject: `Add the quadratic weight curve`.

### Task 5: Test-only geometric checker and random grids

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the helper library is made with `fluidity_add_test_library` and linked through `LINKS`.

**Files:**
- Create: `native/fluidity/grid/grid_test_helpers.h`, `grid_test_helpers.cc`, `grid_test_support_test.cc`
- Modify: build registration: `fluidity_add_test_library(fluidity_grid_test_helpers SOURCES grid_test_helpers.cc
  LINKS fluidity_base_test_support)`; every grid test from here on names it in `LINKS` of `fluidity_add_test`;
  never linked by `fluidity`

**Interfaces:**
- Consumes: Task 3.
- Produces (test-only, `namespace fluidity::grid::testing`):
  - `std::uint64_t ToUniformIndex(std::uint64_t raw, std::uint64_t count)` and `double ToUnitDouble(std::uint64_t
    raw)` (53 high bits / 2^53): the only mappings random tests use.
  - `std::vector<CellId> BuildRandomGridCells(std::uint64_t seed, int size_level_count, int biggest_cells_per_axis)`:
    a valid list (siblings together, size-step rule) made by splitting random cells of a box of level-0 cells,
    then splitting bigger neighbours until the rule holds.
  - `std::vector<CellId> ApplySizeStepClosure(std::vector<CellId> cells, int size_level_count)`: that last part on
    its own (split any active cell two levels bigger than a touching cell, repeat until none), for hand-built
    scenes.
  - `class GeometricGrid`: built from `GridFrame` and an active-cell list, with no blocks and no tables. Works in
    integer smallest-cell units. Answers: `FacePieces()` (every piece as axis, level, `cell_high_coord`, low cell
    id or none, high cell id or none, area), `TouchingCells(CellId)`, `Corners()` (each physical corner once, at
    the smallest level touching it), `ContainingCell(Vec3)`, `NearbyFaces(Vec3, Axis)` (the candidate rule of
    design 5.4 step 2 by brute force), `FacesWithin(FacePieceId, double radius_in_cell_widths)`.

- [ ] **Step 1: Failing tests** (label `cpu`): `TEST(GeometricGrid, OneBigCell_HasSixPieces)`;
  `TEST(GeometricGrid, SplitCellBesideBigCell_HasFourQuarterPieces)` (areas 0.25 m2 each);
  `TEST(BuildRandomGridCells, SameSeed_GivesSameList)`;
  `TEST(BuildRandomGridCells, AnySeed_KeepsSiblingsAndSizeStep)` (100 seeds, checked by brute force).
- [ ] **Step 2: Run**, fail; **Step 3: Implement** by brute force (clarity over speed: quadratic loops are fine
  up to a few thousand cells). **Step 4: Run** the gate.
- [ ] **Step 5: Commit.** Subject: `Add a test-only geometric grid checker`. Body: test-only, never linked into
  the library.

### Task 6: Memory estimate entries, `GridBuffers` and the build status record

**Amended 2026-11-14 (A17):** a GPU grid's buffers hold the tables, the build scratch, the piece map and the
status record on the GPU and nothing on the CPU; the CPU copy of the tables is gone.

**Amended 2026-11-14 (reconcile):** the build scratch can be one `GridScratch` lent to several buffers
(`BuildGridBuffers(settings, scratch)`, `BuildGridScratch`, `ScratchSharing` in `BuildGridMemoryEstimate`), and
`GridSettings::max_cell_blocks` has no default room (0 is `kBadInput`). Test file named
`grid_memory_estimate_test.cc` (A23).

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** `GridScratch` follows the scratch pattern: `ComputeGridScratchSize` added and tested against the estimate's scratch rows.

**Files:**
- Create: `native/fluidity/grid/grid_memory_estimate.h`, `grid_memory_estimate.cc`, `grid_memory_estimate_test.cc`,
  `grid_storage_internal.h` (private), `grid_build_status.h` (S, private), `grid_storage_no_gpu_test.cc`
- Modify: `staggered_grid.h`, `staggered_grid_cpu.cc`, build registration

**Interfaces:**
- Consumes: `Buffer`, `BuildBuffer`, `MemoryEstimateEntry` (with its `where` field), `MemoryWhere`, `MemoryCounts`,
  `CountedPer`, `KeptFor`, both `ComputeMemoryBytes` forms, `CheckCanRunOn`; `OrderScratch`,
  `BuildOrderScratch`, `ComputeOrderScratchSize` (Foundation, see "Before you start"); `GridError`, `CellId`.
- Produces: `GridSettings`, `GridBuffers`, `BuildGridBuffers` (design 3.3); `ToMemoryCounts`,
  `BuildGridMemoryEstimate` (3.6); private `GridBuildStatus`. **Plan decisions:**
  - Rows come in this fixed order, names prefixed `grid: `: `block ids`, `same-level neighbour blocks`,
    `bigger and smaller blocks`, `cell states`, `face piece kinds`, each `where = MemoryWhere::kRunOn`.
    Then, for `run_on == kCpu`: `build scratch: cell ids` (kCellPlace, 8 B) and `piece map` (kCellBlock, 4 B),
    both `kCpu` (unchanged). For `run_on == kGpu`, all `kRunOn`: `build scratch: cell ids` (kCellPlace, 8 B; the
    sorted active cells, then active and split cells), `build scratch: sort keys` (kCellPlace, 8 B),
    `build scratch: order` (kCellPlace, 4 B), `build scratch: second order` (kCellPlace, 4 B; also the marks and
    their prefix sums), `build scratch: ordering` (kCellPlace, `ComputeOrderScratchSize(kGpu).bytes_per_item`),
    `build scratch: ordering, fixed part` (kGivenCount 1, `fixed_bytes`), `build scratch: prefix sum ends`
    (kGivenCount 2, 4 B: the order buffers have `64 * max_cell_blocks + 1` entries, because a prefix sum writes
    its total after the last entry), `piece map` (kCellBlock, 4 B) and `build status` (kGivenCount 1,
    `sizeof(GridBuildStatus)`). No row of a GPU grid is `kCpu`. All
    `KeptFor::kWholeRun`, `copies = storage_count`. `BuildGridBuffers` walks these rows and allocates each on
    `run_on` or the CPU as its `where` says, so the estimate cannot drift; there is no other table of processors.
  - `max_cell_blocks` above `kMaxCellBlocksLimit` is `kBadInput`; 0 is `kBadInput` (no default room).
  - **Shared build scratch (reconcile item 11).** Public, in `staggered_grid.h`:

    ```cpp
    // The temporary arrays of grid builds and checks on one processor for grids of at most max_cell_blocks
    // blocks: the "grid: build scratch: ..." entries of BuildGridMemoryEstimate (cell ids, sort keys, two
    // orders, the prefix-sum ends and an OrderScratch for 64 * max_cell_blocks + 1 items). Holds nothing between
    // calls. Owned by the caller and lent to every GridBuffers built with it; those buffers must not outlive it,
    // it must not be moved while they exist, and one build or check uses it at a time. Move-only.
    class GridScratch {
     public:
      std::uint32_t max_cell_blocks() const;
      RunOn run_on() const;
    };
    // kBadInput: max_cell_blocks 0 or above kMaxCellBlocksLimit; kOutOfMemory; kRunOnCallFailed.
    std::expected<GridScratch, GridFailure> BuildGridScratch(RunOn run_on, std::uint32_t max_cell_blocks);
    // As BuildGridBuffers(settings), but the buffers allocate no build scratch and use `scratch` for every
    // BuildGrid, BuildGridPiece and buffers form of ComputeGridCounts and CheckSizeStepRule. kBadInput when
    // scratch.run_on() != settings.run_on or scratch.max_cell_blocks() < settings.max_cell_blocks.
    std::expected<GridBuffers, GridFailure> BuildGridBuffers(const GridSettings& settings, GridScratch& scratch);
    // Who holds the build scratch entries of BuildGridMemoryEstimate.
    enum class ScratchSharing : std::uint8_t {
      kOnePerBuffers,     // Each of the storage_count buffers owns one (BuildGridBuffers(settings)).
      kOneForAllBuffers,  // One GridScratch lent to all of them (BuildGridBuffers(settings, scratch)).
    };
    std::vector<MemoryEstimateEntry> BuildGridMemoryEstimate(
        RunOn run_on, std::uint32_t storage_count, ScratchSharing scratch_sharing = ScratchSharing::kOnePerBuffers);
    ```
    **Amended 2026-11-14 (Foundation as built):** `GridScratch` follows Foundation's scratch pattern, so it also has
    (in `staggered_grid.h`, plan decision):

    ```cpp
    // Exact bytes BuildGridScratch allocates: bytes_per_cell_block * max_cell_blocks + fixed_bytes.
    struct GridScratchSize {
      std::uint32_t bytes_per_cell_block = 0;  // Bytes for each block of max_cell_blocks.
      std::uint64_t fixed_bytes = 0;           // Bytes that do not depend on max_cell_blocks.
    };
    // The bytes BuildGridScratch allocates on `run_on`: the sum of the "grid: build scratch: ..." entries of
    // BuildGridMemoryEstimate(run_on, 1), one definition for both.
    GridScratchSize ComputeGridScratchSize(RunOn run_on);
    ```
    `BuildGridScratch` allocates exactly these bytes and `GridScratch` has `[[nodiscard]] std::size_t byte_count()
    const` (0 after a move); the forms that take the scratch allocate nothing, and `BuildGridBuffers(settings)` and
    `BuildGrid(inputs, settings)` build one for their buffers.
    **Amended 2026-11-14 (A26/A27 pass, small fix c):** `GridScratch`, `BuildGridScratch`, the lent form of
    `BuildGridBuffers` and `enum class ScratchSharing` are declared in `grid/staggered_grid.h`;
    `BuildGridMemoryEstimate` (with its new third parameter) stays in `grid/grid_memory_estimate.h`, which includes
    `staggered_grid.h` for `ScratchSharing` (as the grid design 3.3 and 3.6 place them).
    With `kOneForAllBuffers` the `build scratch:` entries have `copies = 1` and every other entry `copies =
    storage_count`; the default keeps every existing caller's numbers. Per block on the GPU: tables 456 B and piece
    map 4 B per buffers; scratch `1536 + 64 x 6 = 1920` B once (`ComputeOrderScratchSize(kGpu) = {6, 4096}`), plus
    the fixed bytes. On the CPU: tables 456 B (`kRunOn`) and piece map 4 B per buffers; scratch 512 B
    once (the cell ids, `kCpu`; the CPU builder sorts with `std::sort` and needs no `OrderScratch`). The design's 1,996 B per block is the old one-buffers figure without the
    ordering scratch; the shared form costs `2 x 460 + 1920 = 2840` B per block for two GPU grids (was
    `2 x 2380 = 4760`).
  - `grid_storage_internal.h` declares `struct GridBuffersArrays` (one `Buffer` per row, the `OrderScratch` for a
    GPU grid) and `const GridBuffersArrays& ReadStorageArrays(const GridBuffers&)`, for this module's code and
    tests only.
  - `ToMemoryCounts` sets `cell_places = 64`, `face_places = 240`, `corner_places = 125` times `cell_blocks`.
  - `grid_build_status.h` (shared, private; written by GPU kernels, read back with one `ReadBuffer`):

    ```cpp
    inline constexpr std::uint64_t kNoFailureKey = ~std::uint64_t{0};
    struct GridBuildStatus {        // The only thing a grid operation on the GPU returns to the CPU.
      std::uint64_t failure_key;    // kNoFailureKey, or (step << 56) | (position << 8) | detail (Task 7).
      GridError error;              // kNone on success; else the failure's error.
      CellId cell;                  // GridFailure::cell; zero when unused.
      CellId other_cell;            // GridFailure::other_cell; zero when unused.
      std::uint32_t cell_blocks;    // Blocks built (grid or piece); later kernels read it as their count.
      std::uint32_t split_cells;    // Split cells added at step P7; later kernels read it. The CPU ignores it.
      std::uint32_t result_count;   // FindBorderBlocks: entries found. Otherwise 0.
      std::uint64_t active_cells;
      std::uint64_t active_cells_per_level[kMaxSizeLevels];
      std::uint64_t face_pieces;    // Places whose kind is not kNotAPiece.
    };
    ```
    `static_assert(std::is_trivially_copyable_v<GridBuildStatus>)`; its `sizeof` is the row's bytes.

- [ ] **Step 1: Failing tests.** Label `cpu`:
  - `TEST(ComputeMemoryBytes, MatchesAllocatedBytes)`: `BuildGridBuffers({.max_cell_blocks = 1000})`; sum of
    `size() * sizeof(element)` over `ReadStorageArrays` equals `ComputeMemoryBytes(BuildGridMemoryEstimate(kCpu, 1),
    counts at 1000 blocks)`, which equals `1000 * 972` (8 + 108 + 36 + 64 + 240 + 512 + 4); with `where`,
    `kRunOn` gives `1000 * 456` and `kCpu` gives `1000 * 516`. Its GPU half is in Task 11.
  - `TEST(BuildGridMemoryEstimate, GpuRun_KeepsNothingOnTheCpu)`: at 1000 blocks, `BuildGridMemoryEstimate(kGpu, 1)` gives
    `kCpu` 0 and `kRunOn` `1000 * (456 + 1536 + 4 + 64 * bytes_per_item) + fixed_bytes + 8 +
    sizeof(GridBuildStatus)` (1536 = 64 x (8 + 8 + 4 + 4)), with `ComputeOrderScratchSize(kGpu)` read in the test.
  - `TEST(BuildGridMemoryEstimate, NoRowIsACpuCopy)`: no row name of either processor ends in `(CPU copy)`.
  - `TEST(BuildGridMemoryEstimate, TwoStorages_DoublesEveryRow)`: every row has `copies == 2`; totals per `where` double.
  - `TEST(BuildGridMemoryEstimate, SharedScratch_CountedOnce)` (reconcile): both processors, storage count 2,
    `kOneForAllBuffers`: every `build scratch:` row has `copies == 1`, every other row `copies == 2`; on the GPU
    at 1,000 blocks the `kRunOn` total is `1000 * (2 * 460 + 1920) + 4096 + 8 + 2 * (sizeof(GridBuildStatus))`
    (fixed rows as listed above), read with `ComputeOrderScratchSize(kGpu)`.
  - `TEST(BuildGridBuffers, TwoBuffersShareOneScratch_AllocatedEqualsEstimate)` (reconcile): `BuildGridScratch(kCpu,
    1000)` and two `BuildGridBuffers({.max_cell_blocks = 1000}, scratch)`: bytes summed over the scratch and both
    `ReadStorageArrays` equal `ComputeMemoryBytes(BuildGridMemoryEstimate(kCpu, 2, kOneForAllBuffers), counts at
    1000)`; neither buffers holds a scratch array of its own.
  - `TEST(BuildGridBuffers, LentScratchTooSmallOrOtherProcessor_IsBadInput)` (reconcile): scratch for 500, buffers
    for 1000: `kBadInput`, nothing allocated.
  - `TEST(BuildGridScratch, ZeroBlocks_IsBadInput)` and `TEST(GridSettings, DefaultRoomIsZero_BuildGridBuffersIsBadInput)`
    (reconcile).
  - `TEST(ComputeGridScratchSize, CpuScratch_EqualsByteCountAndScratchRows)` (Foundation as built): for 1,000
    blocks, `BuildGridScratch(kCpu, 1000)->byte_count()` equals `1000 * bytes_per_cell_block + fixed_bytes` and the
    sum of the `build scratch:` rows of `BuildGridMemoryEstimate(kCpu, 1)`; the same equality of the formula and the
    rows for `kGpu` (no GPU memory needed for that half).
  - `TEST(BuildGridBuffers, ZeroBlocks_IsBadInput)` and `TEST(BuildGridBuffers, AboveLimit_IsBadInput)`
    (`kMaxCellBlocksLimit + 1`).
  - `TEST(GridBuffers, MovedFrom_HasNoRoom)`: `max_cell_blocks() == 0` after a move.
  - `TEST(GridBuildStatus, IsTriviallyCopyable)`.
  Label `no_gpu`: `TEST(BuildGridBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed)`: no storage returned.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Add the grid storage that is allocated once`. New names: `GridSettings`,
  `GridBuffers`, `BuildGridBuffers`, `ToMemoryCounts`, `BuildGridMemoryEstimate`, `kMaxCellBlocksLimit`,
  `GridScratch`, `BuildGridScratch`, `ScratchSharing` (reason: reconcile item 11, one build scratch per run),
  `GridScratchSize`, `ComputeGridScratchSize` (Foundation's scratch pattern).

### Task 7: Shared build formulas

**Amended 2026-11-14 (A17):** new task; every per-cell, per-block and per-face rule of the build in one shared
header, called by the CPU builder (Tasks 8, 9) and the GPU kernels (Tasks 10, 11) alike.

**Amended 2026-11-14 (Foundation as built):** `grid_build_steps.h` is the shared per-item header of the kernel-wrapper rule; its test is `LABEL cpu` (no `fluidity_add_shared_test`).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/grid_tables.h` (S; the `GridFrame`, `CellState`, `FacePieceKind`, `GridTables`
  types only), `native/fluidity/grid/grid_build_steps.h` (S, private; included by no public header),
  `native/fluidity/grid/grid_build_steps_test.cc` (`fluidity_add_test(... LABEL cpu LINKS fluidity_grid_test_helpers)`;
  the container's `cuda` step runs it too)

**Interfaces:**
- Consumes: Tasks 2, 3, `GridBuildStatus` (Task 6). Writes the `GridFrame`, `CellState`, `FacePieceKind` and
  `GridTables` types of design 3.2 into `grid_tables.h` now (types only; Task 9 used to add them, it now
  consumes them), because the formulas below take them.
- Produces, all `FLUIDITY_CPU_AND_GPU constexpr` or `inline`, `noexcept`, on raw pointers and counts, allocating
  nothing (**plan decisions**, the design gives the steps but not their parallel form):
  - `std::uint64_t ToCellSortKey(CellId cell, int pass)`: the three passes that put cells in fixed cell order
    with `BuildStableOrder` run least significant first. With `scaled(c) = c * 2^(15 - level) + kMaxCoord * 2^15`
    (34 bits, never negative): pass 0 `(scaled(x) << 4) | level`, pass 1 `scaled(y)`, pass 2 `scaled(z)`. So the
    result is `z`, then `y`, then `x`, then bigger level first: exactly `ComesBefore`.
  - `std::uint64_t ToBlockSortKey(CellId cell, int pass)`: the same three passes for the low corner of the cell's
    block (block coordinate times 4 at the cell's level), so cells sorted by it are grouped by block in the fixed
    block order.
  - `GridError CheckCellId(const GridFrame& frame, CellId cell)`: step 1 for one id: `kBadInput` (bit 63 clear,
    level `>= size_level_count`), `kCoordOutOfRange`, else `kNone`.
  - `std::uint32_t FindSortedCell(const CellId* sorted_cells, std::uint32_t count, CellId cell)` and
    `CellBlockIndex FindSortedBlock(const CellBlockId* sorted_blocks, std::uint32_t count, CellBlockId block)`:
    binary search with `ComesBefore`; `kNoIndex` if absent. `FindCellBlock` (Task 12) calls the second.
  - `struct CellCheck { GridError error; CellId cell; CellId other_cell; std::uint8_t detail; };` and the
    checks of one sorted position `i`: `CheckOverlapAt` (step 2: `sorted[i] == sorted[i - 1]`, else the nearest
    active ancestor of `sorted[i]`, one level up first), `CheckSiblingsAt` (step 3: the first missing eighth of
    `ComputeBigCell(sorted[i])`, detail = that eighth), `CheckSizeStepAt` (step 4: the first of the 26
    directions in `(dz, dy, dx)` order whose same-level neighbour place is covered by an active ancestor two or
    more levels up; detail = direction 0..25). Each check looks only at the smaller cell of a pair, so every
    broken pair is reported at exactly one position.
  - Failure key: `ToFailureKey(int step, std::uint64_t position, std::uint8_t detail)` =
    `(step << 56) | (position << 8) | detail`, and `ToFailureStep`, `ToFailurePosition`. **The first failure is
    the smallest key**: earliest step, then smallest position (input index for step 1, place in fixed cell order
    for steps 2 to 4). Step 5 (active plus split cells exceed `64 * max_cell_blocks`) and step 6 (blocks exceed
    `max_cell_blocks`) use position 0 and give `kTooManyCellBlocks` with `cell` zero. The CPU, which runs each
    step to its end before the next and scans positions in order, gives the same failure as the GPU's
    `atomicMin`.
  - `int CountSplitCellsToAdd(CellId cell)` and `CellId ComputeAncestor(CellId cell, int levels_up)`: an active
    cell adds its ancestor one level up when it is eighth 0, the ancestor two levels up when that ancestor's
    child on its path is also eighth 0, and so on. Because siblings exist together, every split cell is added
    exactly once (by the eighth-0 path below it), with no search and no level loop.
  - `void FindBlockNeighbours(const CellBlockId* block_ids, std::uint32_t block_count, CellBlockIndex block,
    std::uint32_t* same_level_27, std::uint32_t* bigger_1, std::uint32_t* smaller_8)`: one block's 36 table
    entries by `FindSortedBlock`, in these layouts (Task 9 repeats them):
    `same_level_blocks[block * 27 + ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)]` (entry 13 is the block itself),
    `smaller_blocks[block * 8 + cell_eighth]`; face local place for axis X `(z * 4 + y) * 5 + x`, Y
    `(z * 5 + y) * 4 + x`, Z `(z * 4 + y) * 4 + x`, the coordinate along the normal being the plane 0..4.
  - `CellState ComputeCoveredCellState(const GridTables& tables, CellPlaceIndex cell)`: for a place that is
    neither active nor split, `kInsideBiggerCell` when the covering place of `bigger_block` is `kActive`, else
    `kAbsent`. It reads only `kActive` marks, which a concurrent writer never changes.
  - `FacePieceKind ComputeFacePieceKind(const GridTables& tables, FacePlaceIndex face)`: design 5.1 item 5 for one
    face place (both sides, plane-4 rule), from the cell states and neighbour tables.
  - `struct FailureCells { CellId cell; CellId other_cell; };` `FailureCells ComputeFailureCells(...)`: re-derives
    the two cells from a failure key and the sorted list, so a one-thread kernel can fill `GridBuildStatus` after
    the minimum is known.

- [ ] **Step 1: Failing tests** (label `cpu`), frame origin 0, width 1 m, 3 levels unless stated:
  - `TEST(ToCellSortKey, ThreePasses_GiveComesBeforeOrder)`: 10,000 random ids at levels 0..3 from seed
    `20261114` (coordinates via `ToUniformIndex` in `-kMaxCoord..kMaxCoord`); sorting index lists by the three
    passes with `std::stable_sort` (test only) gives the order of `std::sort` with `ComesBefore`.
  - `TEST(ToCellSortKey, CoordLimits_FitInThirtyEightBits)`: `kMaxCoord` and `-kMaxCoord` at levels 0 and 15.
  - `TEST(ToBlockSortKey, CellsOfOneBlock_HaveEqualKeys)`.
  - `TEST(CheckOverlapAt, RepeatAndAncestor_NameBothCells)`; `TEST(CheckSiblingsAt, SevenOfEight_NamesTheMissing)`;
    `TEST(CheckSizeStepAt, CornerTouchTwoLevelsApart_NamesDirectionZero)` (level 2 `{4,4,4}` with level 0
    `{0,0,0}` below-left: direction `(dz, dy, dx) = (-1, -1, -1)` is 0).
  - `TEST(ToFailureKey, StepOutranksPosition)`: `ToFailureKey(2, 0, 0) > ToFailureKey(1, 2^40, 255)`.
  - `TEST(CountSplitCellsToAdd, RandomGrids_AddEachSplitCellOnce)`: 200 grids from `BuildRandomGridCells`: the
    added ancestors are distinct and equal the brute-force set of split cells.
  - `TEST(ComputeFacePieceKind, TwoSizeFixture_GivesFourQuarterPieces)`: tables built by hand for level 0
    `{0,0,0}` and the 8 level-1 cells of `{1,0,0}`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement.** Header comment: "the CPU and the GPU builders call these
  functions and nothing else for a table entry; that is why their tables are identical". `Read:`
  `<sec-parallel-build>`. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Put every per-cell and per-face build rule in one shared header`. Private
  names only; `GridFrame`, `CellState`, `FacePieceKind`, `GridTables` listed as new public names.

### Task 8: Checking a cell list: `CheckSizeStepRule` and `ComputeGridCounts`

**Amended 2026-11-14 (A17):** `GridInputs` takes the list either as a CPU span or as a buffer on the grid's
processor, the buffers forms of both checks are added, and the first-failure rule is the one of Task 7.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/grid_cell_checks.h` (private), `grid_cell_checks.cc`, `grid_cell_checks_test.cc`
- Modify: `staggered_grid.h`, `staggered_grid_cpu.cc`

**Interfaces:**
- Consumes: Task 7.
- Produces: `GridInputs`, `GridCounts`, both `ComputeGridCounts` forms, both `CheckSizeStepRule` forms (design
  3.3 plus the buffers forms below). Private: helpers that take a frame and a **sorted** span of ids (fixed cell
  order) and run `<proc-build-grid>` steps 1 to 4 by calling Task 7's checks position by position. `BuildGrid`
  (Task 9) runs them on the storage's scratch.
- **Plan decision (A17): `GridInputs` carries the list in one of two forms.**

  ```cpp
  struct GridInputs {
    GridFrame frame;
    std::span<const CellId> active_cells;                // A list in CPU memory, any order.
    const Buffer<CellId>* active_cell_buffer = nullptr;  // Or a list in a buffer, any order.
    std::uint32_t active_cell_buffer_count = 0;          // Entries of *active_cell_buffer used.
    GridVersion version;                                 // 1 or more.
  };
  ```
  When `active_cell_buffer` is set it is the list and `active_cells` must be empty; otherwise `active_cells` is
  the list. A list on another processor than the buffers' (or, for the one-argument forms, than the CPU) is
  `kBadInput`, except an empty list, which has no processor. `active_cell_buffer_count > size()` is `kBadInput`.
  So a GPU run passes a `Buffer<CellId>` on the GPU, a CPU run either form, tests and tools a span.
- **Plan decision (A17): the buffers forms**, which allocate nothing and run on `buffers.run_on()` (CPU here;
  GPU in Tasks 10 and 11); `buffers` holds no grid before or after and its contents become unspecified:

  ```cpp
  std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs, GridBuffers& buffers);
  std::expected<void, GridFailure> CheckSizeStepRule(const GridInputs& inputs, GridBuffers& buffers);
  ```
  `ComputeGridCounts(inputs, buffers)` runs steps 1 to 8 (face pieces are counted from built kinds) and so,
  unlike the one-argument form, fails `kTooManyCellBlocks` (`cell` zero) when the grid does not fit the buffers;
  a planner that must size beyond the room uses the one-argument form on a CPU list. `CheckSizeStepRule` (both
  forms) runs steps 1, 2 and 4: it does not require siblings, because cell resize checks lists while it splits.
  The one-argument forms (`ComputeGridCounts(inputs)`, `CheckSizeStepRule(frame, span)`) accept only a CPU list
  (a GPU buffer is `kBadInput`) and sort a CPU copy of the ids (8 B per active cell); recorded in the hand-over.
- **Plan decisions:** failure context as Task 7's checks give it: `kCellsOverlap` names the smaller (or repeated)
  cell in `cell` and the bigger (or earlier) in `other_cell`; `kSiblingCellMissing` names the first missing small
  cell (eighth order) in `cell` and its big cell in `other_cell`, for the first cell in fixed order whose
  siblings are incomplete; `kSizeStepRuleBroken` names the smaller cell in `cell` and the bigger in `other_cell`;
  `kCoordOutOfRange` and `kBadInput` from an id name the first such id in input order. Frame: `origin` finite,
  `biggest_cell_width` finite and positive, `size_level_count` in 1..16, else `kBadInput`; version 0 is
  `kBadInput`. These are checked on the CPU before any work on either processor.

- [ ] **Step 1: Failing tests** (label `cpu`), frame origin 0, width 1 m, 3 levels:
  - `TEST(CheckSizeStepRule, CornerTouchTwoLevelsApart_IsRejected)`: level 0 `{0,0,0}` and level 2 `{4,4,4}`:
    `kSizeStepRuleBroken`, `cell` = the level-2 id, `other_cell` = the level-0 id. Same test body repeats with
    level 2 `{4,4,0}` (edge) and `{4,0,0}` (face).
  - `TEST(CheckSizeStepRule, OneLevelApart_Passes)`: level 0 `{0,0,0}` and level 1 `{2,0,0}`.
  - `TEST(ComputeGridCounts, CellAndItsBigCell_IsCellsOverlap)` and `..., SameCellTwice_IsCellsOverlap`.
  - `TEST(ComputeGridCounts, CoordBeyondLimit_IsCoordOutOfRange)`: coordinate `kMaxCoord + 1`.
  - `TEST(ComputeGridCounts, OneSplitCell_CountsBlocksOfBothLevels)`: level-0 cells `{0,0,0}` and the 8 level-1
    cells of `{1,0,0}`: `cell_blocks == 2`, `active_cells_per_level = {1, 8}`, `active_cells == 9`.
  - `TEST(ComputeGridCounts, BadIdAfterRepeat_IsBadInputOfFirstBadId)`: input `{a, a, bad1, bad2}` (two ids
    without bit 63): `kBadInput` naming `bad1` (step 1 outranks step 2; input order within step 1).
  - `TEST(GridInputs, SpanAndBufferBoth_IsBadInput)`; `TEST(GridInputs, BufferCountAboveSize_IsBadInput)`;
    `TEST(GridInputs, CpuBufferList_GivesSameCountsAsSpan)` (a CPU `Buffer<CellId>` holding the list of
    `OneSplitCell`); `TEST(ComputeGridCounts, EmptyList_GivesZeroCounts)` (both forms).
  - `TEST(ComputeGridCounts, IntoBuffers_EqualsAllocatingForm)`: 200 random grids; the buffers form on CPU
    buffers for 4,096 blocks equals `ComputeGridCounts(inputs)`; a grid of 4,097 blocks gives
    `kTooManyCellBlocks` with `cell` zero, while the one-argument form gives `cell_blocks == 4097`. (Its
    no-allocation half is in Task 9, which brings the allocation counter.)
  - `TEST(CheckSizeStepRule, IntoBuffers_EqualsSpanForm)`: the three `CornerTouch...` lists and 200 random grids.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** with `// Step N:` comments from `<proc-build-grid>`. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Check a list of active cells before building a grid`. New names: `GridInputs`
  (with `active_cell_buffer`, `active_cell_buffer_count`), `GridCounts`, `ComputeGridCounts` (two forms),
  `CheckSizeStepRule` (two forms).

### Task 9: `BuildGrid` into storage on the CPU, `StaggeredGrid` and `ReleaseBuffers`

**Amended 2026-11-14 (A17):** the CPU builder fills table entries only through Task 7's formulas, has no upload
step, and hands GPU buffers to the GPU path of Task 11.

**Amended 2026-11-14 (reconcile):** a test pins that a build into buffers with a lent `GridScratch` leaves another
live grid that shares the scratch unchanged.

**Amended 2026-11-14 (Foundation as built):** the allocation counter is a test source listed in each counting test program; tests link the helper libraries.

**Kind:** numerics

**Files:**
- Modify: `staggered_grid.h`, `staggered_grid_cpu.cc`, `grid_cell_checks.cc`
- Create: `staggered_grid_test.cc`, `allocation_counter_test_helpers.h`, `allocation_counter_test_helpers.cc`,
  `staggered_grid_no_gpu_test.cc`
- Modify: build registration: `allocation_counter_test_helpers.cc` is listed in the `SOURCES` of every test program
  that counts allocations (Tasks 9, 21, 22), never in a library (a replaced `operator new` must be part of the
  program itself, and a program that does not count must not get it); tests `LINKS fluidity_grid_test_helpers`

**Interfaces:**
- Consumes: Task 7 (`grid_tables.h` types and the build formulas), Task 8.
- Produces: `StaggeredGrid`, both `BuildGrid` forms, `ReleaseBuffers(StaggeredGrid&&)` (3.3), with every accessor
  including `all_blocks()` (returns the `CellBlockRange` of Task 3, `{0, counts().cell_blocks}`).
- **Plan decisions:** cell scratch sort key is (size level, block coordinate `z, y, x`, local place), so the cells
  of one block are adjacent; blocks are created level by level from the smallest up in `block_ids`, then sorted
  in place with `std::sort` and `ComesBefore`. No `std::stable_sort` (it may allocate). Table layouts:
  `same_level_blocks[block * 27 + ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)]` (entry 13 is the block itself);
  face local place for axis X `(z * 4 + y) * 5 + x`, Y `(z * 5 + y) * 4 + x`, Z `(z * 4 + y) * 4 + x`, where the
  coordinate along the normal is the plane 0..4 and plane `p` is the low side of local cell `p`. Neighbour
  entries come from `FindBlockNeighbours`, states from the active and split marks then `ComputeCoveredCellState`,
  kinds from `ComputeFacePieceKind` (Task 7): the CPU builder writes no table entry any other way.
- **Plan decision (A17), the block limit on both processors:** when `active_cells` plus the split cells they
  imply exceed `64 * max_cell_blocks`, or the blocks exceed `max_cell_blocks`, the failure is
  `kTooManyCellBlocks` with `cell` zero (the design's "first cell of the first block beyond the limit" is not
  reported: the GPU cannot name it without scratch beyond the room, and the step's answer is to raise the limit).
- `BuildGrid(inputs, buffers)` with `buffers.run_on() == kGpu` calls the GPU builder of Task 11; until Task 11
  exists it returns `kRunOnCallFailed`. `tables(RunOn::kGpu)` on a CPU grid and `tables(RunOn::kCpu)` on a GPU
  grid return version 0 and null arrays.
- `allocation_counter_test_helpers.cc` replaces every global `operator new`/`delete` form (plain, nothrow,
  `std::align_val_t`) and counts calls. First read `base/buffer_cpu.cc`: if `Buffer` does not allocate through
  `operator new`, stop and ask the owner how CPU buffer allocations are to be counted.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(BuildGrid, LoneSmallCell_IsSiblingCellMissing)`: 7 of the 8 level-1 cells of level-0 `{0,0,0}`:
    `kSiblingCellMissing`, `cell` = the missing one.
  - `TEST(BuildGrid, VersionZero_IsBadInput)`.
  - `TEST(BuildGrid, AnyInputOrder_GivesSameTables)`: 200 grids from `BuildRandomGridCells` (seeds 1..200, 4
    levels); each list shuffled with seed `seed + 1000`; the used part of every table array is byte-identical.
  - `TEST(ComputeGridCounts, RandomGrids_EqualBuiltCounts)`: same 200 grids.
  - `TEST(BuildGrid, KindsAndStates_MatchGeometricChecker)`: same grids; every geometric piece is exactly one
    place whose kind is `kBetweenCells`, `kGridEdgeLow` or `kGridEdgeHigh` as the checker says; no other place is
    a piece.
  - `TEST(BuildGrid, AboveStorageRoom_IsTooManyCellBlocks)`: storage for 8 blocks; a grid of 9 level-0 blocks
    (one cell in each of 9 blocks along X) fails, no grid; the same storage then builds an 8-block grid.
  - `TEST(BuildGrid, IntoStorage_AllocatesNothing)`: storage for 4,096 blocks; 100 random grids built and
    released in turn; after `BuildGridBuffers`, the counter stays 0 and the addresses in `ReadStorageArrays` do
    not change; each grid's used tables equal those of `BuildGrid(inputs, settings)` byte for byte.
  - `TEST(ComputeGridCounts, IntoBuffers_AllocatesNothing)`: the 100 grids of the test above through
    `ComputeGridCounts(inputs, buffers)` and `CheckSizeStepRule(inputs, buffers)`: the counter stays 0.
  - `TEST(BuildGrid, EmptyList_GivesEmptyGrid)`: version 1, zero blocks, `all_blocks() == {0, 0}`.
  - `TEST(BuildGrid, SharedScratch_OtherLiveGridUnchanged)` (reconcile): one `GridScratch` for 4,096 blocks lent to
    two buffers; grid A built into the first, then 100 random grids built and released in turn into the second
    (and `ComputeGridCounts`, `CheckSizeStepRule` on it): A's five tables, counts and `FindCellAt` answers on 1,000
    random positions are byte-identical before and after; each B equals B built into buffers with their own
    scratch.
  - `TEST(BuildGrid, DefaultSettings_SizedToTheList)` (reconcile): `BuildGrid(inputs, GridSettings{})` of a 9-block
    list gives buffers with `max_cell_blocks() == 9`.
  - `TEST(BuildGrid, SplitCellsFromEighthZeroPaths_MatchTheBuiltStates)`: 200 random grids; the cells
    `CountSplitCellsToAdd`/`ComputeAncestor` add are exactly the places the CPU build marks `kSplit` (ties the
    GPU's P7 rule to the oracle).
  Label `no_gpu`: `TEST(BuildGrid, OnGpuWithoutAGpu_IsRunOnCallFailed)`: settings `run_on = kGpu`; nothing returned.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** steps 5 to 8 of `<proc-build-grid>`. Step 9 (upload) no longer
  exists: a CPU grid stays on the CPU and a GPU grid is built on the GPU (Task 11). The paper's `<proc-build-grid>`
  keeps step 9 as "on the GPU, see `<proc-build-grid-in-parallel>`". **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Build a grid version into storage without allocating`. List the new names.

### Task 10: Checking a cell list on the GPU

**Amended 2026-11-14 (A17):** new task; steps P1 to P6 of `<proc-build-grid-in-parallel>`: the GPU sorts a cell
list that lives on the GPU into fixed cell order and checks it, returning only the status record.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two-halves pattern: the P1 to P6 sequence runs from the CPU half (`grid_cell_checks.cc`), each kernel is a GPU-half function of `staggered_grid_gpu.h` returning `RunOnCallStatus`, launched with Foundation's `ComputeLaunchSize` (the plan's `LaunchShape`, `ComputeLaunchShape` and `kBuildLaunchBlocks` are dropped with their `no_gpu` test); new `cpu` launch-order test; digests.

**Files:**
- Create: `native/fluidity/grid/staggered_grid_gpu.h` (private: the GPU half's functions on plain pointers, counts
  and a `GridBuildStatus*`), `native/fluidity/grid/staggered_grid_gpu.cu` (private), `grid_build_launch_test.cc`
  (label `cpu`), `grid_cell_checks_gpu_test.cc` (label `gpu`). (`grid_cell_checks_no_gpu_test.cc` is dropped: its
  only test was the launch-shape test.)
- Modify: `grid_cell_checks.cc` (dispatch on `buffers.run_on()`; the GPU sequence under
  `#if defined(FLUIDITY_HAS_GPU)`), `grid_build_steps.h` (per-thread functions below), build registration
  (`fluidity_add_gpu_sources(staggered_grid_gpu.cu)`; tests with `LINKS fluidity_base_test_support
  fluidity_grid_test_helpers`)

**Interfaces:**
- Consumes: Task 7 (`ToCellSortKey`, `CheckCellId`, the three checks, the failure key, `ComputeFailureCells`),
  Task 8 (`GridInputs` rules, CPU helpers as oracle), `ReadStorageArrays`, `GridBuildStatus`; Foundation
  `BuildStableOrder(keys, count, previous_order, order, OrderScratch&)` (the scratch of the buffers' `GridScratch`),
  `ReadBuffer`, `ToBufferError`, `RunOnCallStatus`, `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`,
  `ComputeGridStride`; test support `ReadAll`, `WriteResultDigest`.
- Produces: the GPU path of `CheckSizeStepRule(inputs, buffers)`; private, in `grid_cell_checks.h`:
  `enum class SiblingCheck : std::uint8_t { kSkip, kRequire };`
  `std::expected<void, GridFailure> CheckCellListOnBuffers(const GridInputs& inputs, GridBuffers& buffers,
  SiblingCheck sibling_check);` (the CPU half: P1 to P6 and the status read; Task 11 continues from its sorted
  list). It calls the GPU half of `staggered_grid_gpu.h`, one function per kernel (`ResetGpuBuildStatus`,
  `CheckGpuCellIds`, `WriteGpuCellSortKeys`, `CopyGpuSortedCells`, `CheckGpuSortedCells`, `FillGpuFailureCells`;
  names may say `Gpu`), each returning `RunOnCallStatus`, and Foundation's `BuildStableOrder` between them. A
  status other than `kDone`, and a `BufferError` from `BuildStableOrder` or `ReadBuffer`, go through
  `ToBufferError` and the module's one mapping into `GridFailure` (Global Constraints); after `kRunOnCallFailed`
  the buffers' contents are unspecified, they hold no grid, and the run stops.
- Shared per-thread functions (added to `grid_build_steps.h`, `FLUIDITY_CPU_AND_GPU`, plan decision): each
  kernel's body is one of these, plus an integer `atomicMin` of the key it returns:
  `std::uint64_t ComputeCellIdFailureKey(const GridFrame& frame, const CellId* input, std::uint64_t index)` (P2;
  `kNoFailureKey` when the id is good) and `std::uint64_t ComputeSortedCellFailureKey(const GridFrame& frame, const
  CellId* sorted, std::uint32_t count, std::uint64_t position, int step)` (P4 to P6: the Task 7 check of that step at
  that position). Every launch is `ComputeLaunchSize(count)` and every kernel a grid-stride loop
  (`ComputeGridStride`).
- **Plan decisions (A17), the parallel form of steps 1 to 4:**
  - P1 (CPU, nothing launched yet): Task 8's frame, version and input-shape rules; `n = active_cell_buffer_count`
    (or the span's size) above `64 * max_cell_blocks` is `kTooManyCellBlocks`, `cell` zero. Then a one-thread
    kernel resets the status (`failure_key = kNoFailureKey`, counts zero): nothing is written from the CPU.
  - P2, one thread per input index `i`: `CheckCellId`; on failure `atomicMin(failure_key, ToFailureKey(1, i, 0))`.
  - P3, sort into fixed cell order: for pass 0, 1, 2 a kernel writes `ToCellSortKey(input[i], pass)` into
    `sort keys`, then `BuildStableOrder` (pass 0 with no previous order, then the previous pass's order),
    alternating `order` and `second order`; a kernel then gathers `cell ids[j] = input[order[j]]`. An id that
    failed P2 gets key 0: the sort still runs on well-defined numbers and its result is never used.
  - P4, P5 (only with `kRequire`), P6, one thread per sorted position `j`: each kernel first reads
    `failure_key` and returns when an earlier step failed; else it runs its Task 7 check and `atomicMin`s
    `ToFailureKey(step, j, detail)`. So no step runs on data an earlier step rejected.
  - Status: a one-thread kernel fills `error`, `cell`, `other_cell` from the minimum key with
    `ComputeFailureCells` (step 1: the input id at that index). The CPU reads `GridBuildStatus` with one
    `ReadBuffer`: the only transfer of the operation.

- [ ] **Step 1: Tests.** Label `cpu` (`grid_build_launch_test.cc`, the main evidence until the hardware run):
  - `TEST(CheckCellListLaunch, EveryThreadBlockAndThread_SameFirstFailureAsCpu)`: P2 to P6 replayed on CPU buffers:
    each kernel's per-thread function run for every thread block and thread of `ComputeLaunchSize(count)`, in launch
    order and in reverse, the atomic replaced by a plain minimum, `BuildStableOrder` on the CPU between the passes;
    on the lists of `OnGpu_EqualsCpu` and the 500 mutated lists of `OnGpuSeveralFailures_SameFirstFailureAsCpu`
    below, both `SiblingCheck` values: the sorted list and the `GridFailure` equal the CPU helpers' exactly.
  Label `gpu` (lists are put on the GPU with `WriteBuffer` in the test's setup only; each test passes the status
  records it reads to `WriteResultDigest`):
  - `TEST(CheckSizeStepRule, OnGpu_EqualsCpu)`: the three `CornerTouchTwoLevelsApart` lists, `OneLevelApart`, the
    empty list and 200 random grids (seeds 1..200, 4 levels, shuffled with `seed + 1000`): result identical to
    `CheckSizeStepRule(frame, span)`.
  - `TEST(CheckSizeStepRule, OnGpuSeveralFailures_SameFirstFailureAsCpu)`: 500 random grids, each with 1 to 4
    mutations drawn from seed `seed + 7000` (repeat a cell; drop a cell; add a cell's big cell; add a cell two
    levels smaller beside a big cell, with its siblings; clear bit 63 of one id; set a coordinate to
    `kMaxCoord + 1`): for both `SiblingCheck` values, `GridFailure` (error, cell, other_cell) equals the CPU
    helpers' exactly.
  - `TEST(CheckSizeStepRule, OnGpuTwice_GivesIdenticalFailure)`: the 500 mutated lists, twice.
  - `TEST(CheckSizeStepRule, OnGpuBuffersWithCpuList_IsBadInput)` and
    `TEST(ComputeGridCounts, GpuBufferListInOneArgumentForm_IsBadInput)`.
  - `TEST(CheckSizeStepRule, OnGpu_AllocatesNothing)`: 100 calls; `ReadMemoryAvailable(kGpu)` free bytes
    unchanged after the first.
  Label `no_gpu`: none beyond Task 6's: GPU buffers cannot exist without a GPU, so the "without a GPU" case is
  `BuildGridBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed`; the launch size is Foundation's, tested in
  `base/run_on_call_test.cc`.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda` (compiled and linked; the container runs the `cpu` tests,
  the launch-order test among them, and the `no_gpu` tests).
- [ ] **Step 3: Commit.** Subject: `Check a cell list on the GPU in the CPU's order`. Body: `GPU: compiled, not
  run`; new private names (`SiblingCheck`, `CheckCellListOnBuffers`, the GPU-half functions,
  `ComputeCellIdFailureKey`, `ComputeSortedCellFailureKey`); the six `gpu` tests for the final hardware run.

### Task 11: Building the tables on the GPU, and `ReadGrid`

**Amended 2026-11-14 (A17):** new task; steps P7 to P12: blocks, neighbour tables, cell states and face piece
kinds built on the GPU, compared with the CPU oracle bit for bit; replaces the upload of the old design.

**Amended 2026-11-14 (reconcile):** GPU form of the shared-scratch test.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** P7 to P12 run from the CPU half (`staggered_grid_cpu.cc`) as GPU-half functions of `staggered_grid_gpu.h` returning `RunOnCallStatus`, with Foundation's scratch-form order and prefix sum between them; P11 loops thread blocks over cell blocks; per-thread bodies in `grid_build_steps.h` with a `cpu` launch-order test; digests.

**Files:**
- Modify: `staggered_grid_gpu.cu`, `staggered_grid_gpu.h`, `grid_build_steps.h`, `grid_build_launch_test.cc`,
  `staggered_grid.h`, `staggered_grid_cpu.cc` (the GPU sequence under `#if defined(FLUIDITY_HAS_GPU)`), build
  registration
- Create: `native/fluidity/grid/staggered_grid_gpu_test.cc` (label `gpu`), `read_grid_test.cc` (label `cpu`)

**Interfaces:**
- Consumes: Task 10 (`CheckCellListOnBuffers`, the GPU-half pattern), Task 7 (`CountSplitCellsToAdd`,
  `ComputeAncestor`, `ToBlockSortKey`, `FindBlockNeighbours`, `ComputeCoveredCellState`, `ComputeFacePieceKind`);
  Foundation `BuildStableOrder` and `ComputeExclusivePrefixSum` (scratch forms, with the `OrderScratch` of the
  buffers' `GridScratch`), slice `ReadBuffer`, `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`.
- Shared per-thread functions (added to `grid_build_steps.h`, plan decision): `ApplySplitCellWritesAt` (P7: one
  active cell's ancestors at `n + sum + k`), `ComputeBlockStartMark` (P8: 1 at the first entry of a block run,
  never at `CellId{0}`), `ApplyBlockIdWriteAt` (P8 scatter), `ApplySortedCellStateAt` (P10: `kActive` or `kSplit`
  at the entry's block and place, and the level whose count it adds), and the Task 7 formulas for P9, P10's
  covered states and P11. Each kernel is a grid-stride loop over one of them; counts per level and face pieces are
  integer `atomicAdd`s of the values they return.
- Produces: the GPU paths of `BuildGrid(inputs, buffers)`, `BuildGrid(inputs, settings)` and
  `ComputeGridCounts(inputs, buffers)`; **plan decision (A17), public:**

  ```cpp
  // Copies the used part of grid's five tables into cpu_buffers (456 B per block from the GPU) and returns a
  // grid on the CPU with the same version and counts. The one explicit copy of a GPU grid: for tests, tools
  // and a caller that accepts the transfer. Allocates nothing. Fails kBadInput (cpu_buffers not on kCpu),
  // kTooManyCellBlocks (room below grid.counts().cell_blocks), kRunOnCallFailed. On failure cpu_buffers keeps
  // its arrays.
  std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);
  ```
- **Plan decisions (A17), the parallel form of steps 5 to 8** (every kernel loops grid-stride over a count it
  reads from `GridBuildStatus`, so no count comes back mid-build; every kernel returns at once when an earlier
  step failed):
  - P7 add split cells: one thread per sorted active cell writes `CountSplitCellsToAdd` into `second order`;
    `ComputeExclusivePrefixSum` in place; `split_cells` = the total; when `n + split_cells > 64 *
    max_cell_blocks`, step 5 fails; else each thread writes its ancestors to `cell ids[n + sum + k]`.
  - P8 sort cells by block and number the blocks: `sort_count = min(n + n / 7, 64 * max_cell_blocks)`, a CPU
    number bounding active plus split cells (each split cell has 8 children, so `split_cells <= n / 7`);
    places from `n + split_cells` to `sort_count` hold `CellId{0}`, whose keys are all ones, so they sort last.
    Three `ToBlockSortKey` passes with `BuildStableOrder`; then one thread per sorted place marks the first
    entry of each block (`CellId{0}` never marks), `ComputeExclusivePrefixSum` in place, `cell_blocks` = the
    total; above `max_cell_blocks`, step 6 fails; else each marked entry writes its `CellBlockId` to
    `block_ids[sum]`. The sort keys are the fixed block order, so `block_ids` comes out in fixed order with no
    second sort.
  - P9 neighbour tables: one thread per block, `FindBlockNeighbours`.
  - P10 cell states: one thread per cell place writes `kAbsent`; one thread per sorted entry writes `kActive`
    (an entry from the active list) or `kSplit` (an added one) at its block (`sum` of its run) and local place,
    and adds 1 to `active_cells_per_level[level]` for an active entry; then one thread per place still
    `kAbsent` writes `ComputeCoveredCellState`.
  - P11 face piece kinds: thread blocks of 240 threads, one face place each; the launch has
    `min(max_cell_blocks, kMaxBlockCount)` thread blocks and thread block `t` handles cell blocks `t`, `t +
    (thread blocks)`, ... below `cell_blocks` (read from the status), `ComputeFacePieceKind`; pieces counted per
    thread block, then one integer `atomicAdd` into `face_pieces`.
  - P12 status: one thread sets `active_cells` (sum of the 16 levels) or decodes the failure; one `ReadBuffer`
    of `GridBuildStatus`. On success the buffers move into the returned grid with the record's counts; on
    failure no grid is returned and the buffers stay usable.
  - Corner places need no table (design 3.7): `FindCornerPlaces` reads the tables built here.

- [ ] **Step 1: Tests.** Label `cpu`, added to `grid_build_launch_test.cc` (the main evidence until the hardware
  run): `TEST(BuildGridLaunch, EveryThreadBlockAndThread_EqualsCpuTables)`: P2 to P12 replayed on CPU buffers, each
  kernel's per-thread function for every thread block and thread of its launch (P11: every thread block and its
  240 threads), in launch order and in reverse, atomics replaced by plain minimum and addition,
  `BuildStableOrder` and `ComputeExclusivePrefixSum` on the CPU between the kernels; on the lists of
  `OnGpu_EqualsCpuTablesBitForBit` and `OnGpuFailures_EqualCpuFailures` below: version, counts, the used part of the
  five tables and every `GridFailure` equal the CPU builder's byte for byte.
  Label `gpu` (lists put on the GPU with `WriteBuffer` in setup only; tables read with `ReadGrid`; each test passes
  the tables, counts and failures it reads to `WriteResultDigest`):
  - `TEST(BuildGrid, OnGpu_EqualsCpuTablesBitForBit)`: the empty list, one cell, the two-size fixture (level 0
    `{0,0,0}` and the 8 level-1 cells of `{1,0,0}`), and 200 random grids (seeds 1..200, 4 levels, shuffled
    with `seed + 1000`), into GPU buffers for 4,096 blocks: version, counts and the used part of all five
    tables equal the CPU build byte for byte.
  - `TEST(BuildGrid, OnGpuFailures_EqualCpuFailures)`: Task 10's 500 mutated lists, plus a list of 9 level-0
    blocks into buffers for 8 (step 6) and a list of `64 * 8 + 1` cells into the same (step 5): `GridFailure`
    identical to the CPU's.
  - `TEST(BuildGrid, OnGpuFailedBuildThenRebuild_EqualsFreshBuffers)`: a failing mutated list, then grid B into
    the same buffers; B equals B built into fresh buffers.
  - `TEST(BuildGrid, RebuildIntoUsedGpuStorage_EqualsFreshStorage)`: build A, release, build B into the same
    buffers; B's tables equal B built into fresh buffers.
  - `TEST(BuildGrid, IntoGpuStorage_AllocatesNothing)`: 100 grids built and released; `ReadMemoryAvailable(kGpu)`
    free bytes unchanged after the first; the global `operator new` counter (Task 9) stays 0.
  - `TEST(ComputeMemoryBytes, MatchesAllocatedGpuBytes)`: GPU half of Task 6's test at 1,000 blocks: the drop in
    `ReadMemoryAvailable(kGpu)` free bytes equals `ComputeMemoryBytes(BuildGridMemoryEstimate(kGpu, 1), counts,
    MemoryWhere::kRunOn)` (up to the driver's allocation granularity, which the test reads and states), and no
    CPU array is allocated.
  - `TEST(ComputeGridCounts, OnGpu_EqualsCpu)`: the 200 grids.
  - `TEST(BuildGrid, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(BuildGrid, OnGpuSharedScratch_OtherLiveGridUnchanged)` (reconcile): Task 9's shared-scratch test on GPU
    buffers, compared through `ReadGrid`; and the drop in free GPU bytes for one scratch and two buffers equals
    `ComputeMemoryBytes(BuildGridMemoryEstimate(kGpu, 2, kOneForAllBuffers), counts, kRunOn)` (granularity as
    above).
  Label `cpu`: `TEST(ReadGrid, CpuGrid_CopiesIntoOtherCpuBuffers)` (same tables, version, counts),
  `TEST(ReadGrid, BuffersTooSmall_IsTooManyCellBlocks)`, `TEST(StaggeredGrid, CpuGrid_HasNoGpuTables)`.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda`.
- [ ] **Step 3: Commit.** Subject: `Build the grid tables on the GPU and compare them with the CPU's`. Body: `GPU:
  compiled, not run`; new public name `ReadGrid`; the eight `gpu` tests.

### Task 12: Addressing and geometry in `grid_tables.h`

**Amended 2026-11-14 (A17):** `FindCellBlock` is `FindSortedBlock` (Task 7) over `tables.block_ids`, so the
build and every stage search blocks with one function.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/grid/grid_tables.h`
- Create: `native/fluidity/grid/grid_tables_test.cc`

**Interfaces:**
- Produces (design 3.2 and 3.7): `ComputeCellWidth`, `ComputeCellCenter`, `ComputeFacePosition`,
  `ComputeFaceArea`, `ToAxis`, `ToCellId(tables, cell)`, `ToFacePieceId(tables, face)`, `FindCellLow`,
  `FindCellHigh`, `ComputeCellCenterDistance`, `FindCellBlock`, `FindCellPlace`, `FindFacePlace`,
  `ToSizeLevel(tables, block)`.
- **Plan decision:** `ComputeCellCenterDistance` is the distance along the piece's axis (as `main`'s
  `cell_center_distance`). `FindCellLow`/`FindCellHigh` on a piece beside a bigger cell return the bigger active
  cell's place (one level up, via `bigger_block`).

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(BuildGrid, OneBigCell_HasSixEdgePieces)`: 1 level, cell `{0,0,0}`, width 1 m: 6 pieces, each
    `ComputeFaceArea == 1.0`, 3 `kGridEdgeLow`, 3 `kGridEdgeHigh`; `ComputeCellCenterDistance == 0.5` for each.
  - `TEST(BuildGrid, SplitCellBesideBigCell_HasFourQuarterPieces)`: 2 levels, width 1 m, level 0 `{0,0,0}` and the
    8 level-1 cells of `{1,0,0}`: exactly 4 `kBetweenCells` X-pieces between the two sizes, each 0.25 m2, the same
    `FindCellLow`, areas summing to 1.0, `ComputeCellCenterDistance == 0.75`.
  - `TEST(FindCellPlace, EveryActiveCell_RoundTripsThroughItsId)` and `TEST(FindFacePlace, EveryPiece_RoundTrips)`:
    200 random grids; `FindFacePlace` of a `kNotAPiece` place's id is `kNoIndex`.
  - `TEST(ComputeFacePosition, EveryPiece_MatchesGeometricChecker)`: exact (positions are dyadic).
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; header comment carries the worked example of the two-size
  fixture above. `Read:` `<sec-face-pieces>`. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Add addressing and face geometry readable on either processor`.

### Task 13: `FindCellAt`

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`. Create: `native/fluidity/grid/find_cell_at_test.cc`

**Interfaces:**
- Produces: both `FindCellAt` forms (design 3.2), by design 5.2: `coordinate = floor((position - origin) /
  smallest_cell_width)` with `math::Floor`, then walk down following `kSplit`. A non-finite position returns
  `kNoIndex`.

- [ ] **Step 1: Failing tests** (label `cpu`), 2 levels, width 1 m, level 0 `{0,0,0}` and the 8 level-1 cells of
  `{1,0,0}`:
  - `TEST(FindCellAt, PositionOnLowSide_BelongsToCell)`: `{1.0, 0.25, 0.25}` gives level-1 `{2,0,0}`.
  - `TEST(FindCellAt, OnHighSide_BelongsToNext)`: `{std::nextafter(1.0, 0.0), 0.5, 0.5}` gives level-0
    `{0,0,0}`; `{1.0, 0.5, 0.5}` gives level-1 `{2,1,1}`; `{1.5, 0.25, 0.25}` gives level-1 `{3,0,0}`.
  - `TEST(FindCellAt, OutsideEveryCell_IsNoIndex)`: `{-0.5, 0.5, 0.5}` and `{2.0, 0.5, 0.5}`.
  - `TEST(FindCellAt, NearAndSearch_AgreeOnRandomPositions)`: 200 random grids, 1,000 positions each; both forms
    (near = the found cell's block and a block two hops away) agree, and equal `GeometricGrid::ContainingCell`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; the comment says why there is no exact-edge recovery (design
  5.2). **Step 4:** gate. **Step 5: Commit.** Subject: `Find the active cell that contains a position`.

### Task 14: Touching cells and a cell's face pieces

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`. Create: `native/fluidity/grid/touching_cells_test.cc`

**Interfaces:**
- Produces: `TouchingCells`, `FindTouchingCells`, `CellFacePieces`, `FindFacePieces` (design 3.2).

- [ ] **Step 1: Failing tests** (label `cpu`), the 200 random grids of Task 9 (up to 4 levels):
  - `TEST(BuildGrid, EveryCellSide_IsCoveredOnce)`: for every active cell and side, the areas of its pieces on that
    side (from `FindFacePieces`) sum exactly to the cell's face area.
  - `TEST(BuildGrid, ClosedSurface_SignedAreasCancel)`: per cell, sum over pieces of `+area * axis` (cell is
    `cell_low`) or `-area * axis` (cell is `cell_high`) is exactly `{0, 0, 0}`.
  - `TEST(FindTouchingCells, RandomGrids_MatchGeometricChecker)`: same set, ascending, `count <= 56`.
  - `TEST(FindTouchingCells, AllNeighboursSplit_Has56)`: a level-0 cell whose 26 neighbours are split.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Find touching cells and the face pieces of a cell`.

### Task 15: Corners and the cells around an edge

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`. Create: `native/fluidity/grid/corners_test.cc`

**Interfaces:**
- Produces (design 3.7): `CellCorners`, `FaceCorners`, both `FindCornerPlaces`, `ComputeCornerPosition`,
  `CellsAroundEdge`, `FindCellsAroundEdge`. Storage rule of design 3.7 "Where a corner is stored".

- [ ] **Step 1: Failing tests** (label `cpu`). No design-7 test covers these; the plan sets:
  - `TEST(FindCornerPlaces, EveryPhysicalCorner_HasOnePlace)`: 200 random grids; across all active cells, two
    corners have the same `CornerPlaceIndex` exactly when `ComputeCornerPosition` gives the same position; the set
    of positions equals `GeometricGrid::Corners()`.
  - `TEST(FindCornerPlaces, CornerSharedWithSmallCells_IsStoredAtSmallLevel)`: the two-size fixture of Task 12: the
    big cell's corner at `{1,0,0}` has a place in a level-1 block.
  - `TEST(FindCornerPlaces, FaceCorners_AreTheCellCornersOnThatSide)`: for every piece of a uniform 8^3 grid.
  - `TEST(FindCellsAroundEdge, UniformGrid_FindsFourSameSize)`; `..., BesideSplitCell_IsNotSameSize`;
    `..., AtGridEdge_IsNoIndexAndSameSize` (absent places are not "another size").
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Find cell corners and the cells around an edge`. Names for pressure, solids
  and viscosity.

### Task 16: Cells near a face and the face search

**Kind:** numerics

**Files:**
- Modify: `grid_tables.h`. Create: `native/fluidity/grid/face_search_test.cc`

**Interfaces:**
- Produces (design 3.7): `kMaxCellsNearFace`, `CellsNearFace`, `FindCellsNearFace`, `kMaxFaceSearchRadius`,
  `FaceSearch`, `BuildFaceSearch`, `FindNextFace`. **Plan decision:** a radius above `kMaxFaceSearchRadius` is
  treated as `kMaxFaceSearchRadius` (documented, tested); "within" is `|x_f - x_face| < radius * width` on each
  axis, the same strict box as design 5.4.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindCellsNearFace, RandomGrids_MatchEveryCellThatListsTheFace)`: 200 random grids; for every piece, the
    set equals the cells whose positions can list it (low, high and cells touching either), ascending.
  - `TEST(FindCellsNearFace, RandomGrids_NeverExceedTheBound)`: 10,000 random grids; `count <= 112`; the largest
    count found is reported with `RecordProperty("largest_cells_near_face", ...)`.
  - `TEST(FindNextFace, RandomGrids_MatchGeometricChecker)`: radii 1.5, 2.5, 4.0; same faces in ascending order.
  - `TEST(BuildFaceSearch, RadiusAboveLimit_IsClamped)`: radius 6 gives the radius-4 result.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Find the cells near a face and walk faces within a radius`.

### Task 17: `FindNearbyFaces`

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/nearby_face_weights.h` (S), `native/fluidity/grid/nearby_faces_test.cc`

**Interfaces:**
- Consumes: `FindCellAt(tables, position, near)`, `FindTouchingCells`, `FindFacePieces`, `GridError`.
- Produces: `NearbyFaceList`, `NearbyFaceMask`, `NearbyFaceFit`, `FindNearbyFaces`, `ToMaskOfAll` (design 3.5).
  Candidate rule: design 5.4 steps 1 and 2. On `kPositionOutsideGrid` or `kTooManyNearbyFaces` the list has
  `count` 0 and `*error` is set; otherwise `*error = kNone`.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindNearbyFaces, AllNeighboursSplit_Has76)`: level-0 cell `{1,1,1}` of a 3^3 level-0 box whose 26 other
    cells are split; a position at the centre; each axis gives exactly 76.
  - `TEST(FindNearbyFaces, RandomGrids_NeverExceed76)`: 10,000 random grids, 10 positions each, every axis.
  - `TEST(FindNearbyFaces, RandomGrids_MatchGeometricChecker)`: 200 grids, same list in ascending order.
  - `TEST(FindNearbyFaces, UniformGrid_Has27)`; `TEST(FindNearbyFaces, Outside_IsPositionOutsideGrid)`.
  - `TEST(ToMaskOfAll, SeventySixFaces_SetsExactlyThoseBits)`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; the header explains "why 76" (design 5.4). `Read:`
  `<sec-nearby-faces>`. **Step 4:** gate. **Step 5: Commit.** Subject: `List the faces near a position`.

### Task 18: `ComputeNearbyFaceWeights` and `ComputeFaceWeight`

**Kind:** numerics

**Files:**
- Modify: `nearby_face_weights.h`. Create: `native/fluidity/grid/nearby_face_weights_test.cc`
- Modify: `docs/glossary.md` (`kept`, `offset`, `offset_from_mean`, `weight_change`, `weighted_spread`,
  `mean_offset`)

**Interfaces:**
- Consumes: `ComputeWeightCurve`, `ComputeFacePosition`, `ComputeFaceArea`, `kMinWeightSum`, `kMinPivotShare`.
- Produces: `NearbyFaceWeights`, `FaceWeight`, `ComputeNearbyFaceWeights`, `ComputeFaceWeight(tables, weights,
  position_in_list)` (design 3.5). Steps 4 to 9 of design 5.4. Port the two-pass sum and the Cholesky with
  pivot floor from `git show main:native/src/graded_mac_transfer_3d.cc` lines 185-309 (`BuildSupport`,
  `SolveAffineSlope`; pivot constant at line 19). Not ported: its radial weight (lines 171-176).
- **Plan decision:** a private inline `ComputeRawFaceWeight(tables, face, position, cell_width)` returns `w_f` and
  `g_f`; every weight in this module (here and in Task 19) goes through it and scales with the same expressions
  (`weight = w / W`, `weight_change = (g - weight * G) / W`), so the two forms stay bit-identical. Sums run over
  kept faces in list order. `inverse_weighted_spread` is computed from the Cholesky factor of
  `weighted_spread / h^2`, then scaled by `1 / h^2`; it is zero unless `fit == kFull`.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeNearbyFaceWeights, CompleteSameSize_MatchesClosedForm)`: uniform 8^3, `h = 0.25` m, 1,000
    positions inside the central 4^3 cells, all kept: 27 faces; `W = 1` within `1e-14`, `mean_offset = 0` within
    `1e-14 h`, `weighted_spread = h^2/4 I` within `1e-14 h^2`; `is_complete`; `fit == kFull`.
  - `TEST(ComputeNearbyFaceWeights, AnyKeptSet_WeightsSumToOne)`: 200 random grids, 50 positions, random masks
    with at least one kept face of positive weight: `|sum weight - 1| <= 1e-14`, `|sum weight_change| <= 1e-13 / h`
    (each axis), every kept weight `> 0`.
  - `TEST(ComputeNearbyFaceWeights, LinearField_IsReproduced)`: value `a + b . x_f` (`a = 0.7`,
    `b = {0.3, -1.1, 2.0}`) on mixed grids and masks, only where `fit == kFull`: `sum weight * value - row .
    mean_offset` within `1e-13 (|a| + |b| h)` of `a + b . x_p`; `row = (sum weight * value * offset_from_mean) *
    inverse_weighted_spread` within `1e-11 |b|` of `b`.
  - `TEST(ComputeNearbyFaceWeights, WeightChange_MatchesFiniteDifference)`: centred difference with step
    `1e-6 h`, positions at least `0.01 h` from every cell side and from `t = 0.5, 1.5` for every face, same list
    and mask: within `1e-7 / h`.
  - `TEST(ComputeNearbyFaceWeights, FacesOnOnePlane_IsMeanOnly)`: uniform grid, mask keeps the 9 faces of one
    plane: `kMeanOnly`; empty mask: `kNone`.
  - `TEST(ComputeNearbyFaceWeights, CrossingSizeChange_WeightsJump)`: two positions `1e-9` m apart on either side
    of the two-size boundary of Task 12: the weight of a shared face differs by more than `1e-3`. Documents the
    stated property (known open physics "traction across cells of different sizes"); a fix must change this test.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** with `// Step N:` comments from `<proc-nearby-face-weights>`;
  the header states the four properties of design 5.4 and gives the complete case's numbers as a worked example.
  **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Compute nearby-face weights from the faces actually kept`.

### Task 19: The weight seen from a face

**Kind:** numerics

**Files:**
- Modify: `nearby_face_weights.h`. Create: `native/fluidity/grid/face_seen_weight_test.cc`

**Interfaces:**
- Produces (design 3.7): `NearbyFaceSums`, `ToNearbyFaceSums`, `ComputeFaceWeight(tables, face, position, cell,
  sums)`. Returns weight 0 (and zero vectors) when `face` fails the candidate rule for `position`.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(ComputeFaceWeight, SeenFromFace_IsBitIdentical)`: 200 random grids, 50 positions, random masks; for every
    kept entry, the face form with `ToNearbyFaceSums(weights)` returns the same bytes (`std::memcmp`) for
    `weight`, `offset`, `offset_from_mean`, `weight_change` as the list form.
  - `TEST(ComputeFaceWeight, FaceNotInList_IsZero)`: a face two cells away.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** through `ComputeRawFaceWeight`. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Compute one face's weight from a particle's sums`. Names for transfer.

### Task 20: The shared grid functions on the GPU

**Amended 2026-11-14 (A17):** the tables the kernels read are built on the GPU (Task 11), the CPU reference
grid is the CPU build of the same list, and the table and memory tests moved to Task 11.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the test follows Foundation's `shared_math_gpu_test` form (a `.cc` test plus a test-only `_test_gpu.h`/`.cu` runner launched through `base/gpu_launch.cuh`); comparisons through `CheckWithinTolerance`; digests.

**Files:**
- Create: `native/fluidity/grid/grid_gpu_test.cc` (tests, label `gpu`), `grid_functions_test_gpu.h` and
  `grid_functions_test_gpu.cu` (test-only GPU runner, as `base/shared_math_test_gpu.h`: one function that launches
  a grid-stride kernel per shared function with `ApplyGpuFunction` and returns `RunOnCallStatus`)
- Modify: build registration (`fluidity_add_test(grid_gpu_test SOURCES grid_gpu_test.cc grid_functions_test_gpu.cu
  LABEL gpu LINKS fluidity_base_test_support fluidity_grid_test_helpers)`)

**Interfaces:**
- Consumes: `GridTables` from `tables(RunOn::kGpu)` of a grid built on the GPU, `FindCellAt`, `FindNearbyFaces`,
  `ComputeNearbyFaceWeights`, `ComputeFaceWeight` (both forms), `FindCornerPlaces`, `ReadMemoryAvailable`.
- Each test builds every grid twice from one list: on the CPU (the reference) and on the GPU from a GPU copy of
  the list made in the test's setup.

- [ ] **Step 1: Tests** (label `gpu`, one thread per position, outputs copied back with `ReadBuffer`):
  - `TEST(ComputeNearbyFaceWeights, OnGpu_MatchesCpu)`: 10,000 positions on 10 random mixed grids, all axes, random
    masks: lists, masks, `fit`, `is_complete` exact; weights and sums within `1e-15 + 1e-13 |value|`
    (`CheckWithinTolerance`, the CPU value the reference), offsets and spreads with the floor times `h` (`h^2`).
    Every test passes the GPU outputs to `WriteResultDigest`.
  - `TEST(ComputeNearbyFaceWeights, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(FindCornerPlaces, OnGpu_EqualsCpu)`: every active cell of the 10 grids; the 8 corner places identical
    (corners have no table: this proves the GPU-built tables answer them as the CPU's do).
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda` (compiled and linked; nothing of this task runs in the
  container: the shared functions' `cpu` tests are the evidence until the hardware run).
- [ ] **Step 3: Commit.** Subject: `Compile the shared grid functions into GPU tests`. Body: `GPU: compiled, not
  run`; lists the three tests for the final hardware run.

### Task 21: Value buffers bound to a grid version

**Amended 2026-11-14 (A17):** states that `ApplyGridVersion` on a GPU grid runs on the CPU and touches no GPU
memory, and tests it on a grid built on the GPU.

**Amended 2026-11-14 (Foundation as built):** the allocation counter is listed in the test's sources; `gpu` tests write result digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/grid/grid_values.h`, `grid_values_cpu.cc`, `grid_values_test.cc`,
  `grid_values_no_gpu_test.cc`, `grid_values_gpu_test.cc`
- Modify: build registration

**Interfaces:**
- Produces: `CellValues`, `FaceValues`, `CornerValues`, `CellBlockValues`, all `Build...Values` forms,
  `ApplyGridVersion` (four), `ReadValues`/`WriteValues` (four pairs), `CheckGridVersion` (design 3.6). Templates
  live in the header; `grid_values_cpu.cc` holds `CheckGridVersion` and uses the module's one `BufferError`
  mapping. No GPU half: values cross only through `base/` `Buffer` calls. `grid_values_test` lists
  `allocation_counter_test_helpers.cc` in its `SOURCES`; every test `LINKS fluidity_base_test_support
  fluidity_grid_test_helpers`; the `gpu` tests pass the values read back to `WriteResultDigest`. The
  value concept is `std::is_trivially_copyable_v<Value>`. Read/Write use the slice forms over the bound part only.
- **Plan decision (A17):** `ApplyGridVersion` reads only `grid.version()` and `grid.counts().cell_blocks`, which
  every grid keeps on the CPU (from the status record for a GPU grid), and records them; it launches nothing and
  copies nothing, on either processor. Values on another processor than the grid's are `kBadInput` (the rule of
  `plan-decisions.md`, Materials).

- [ ] **Step 1: Failing tests.** Label `cpu`:
  - `TEST(ApplyGridVersion, RebindsWithoutAllocating)`: each of the four types with `Value = double`, room for
    4,096 blocks, bound to 100 random grids in turn: zero counted allocations (allocation counter of Task 9);
    `grid_version()` and `block_count()` follow each grid; a grid of 4,097 blocks gives `kTooManyCellBlocks` and
    leaves version and count unchanged.
  - `TEST(ReadValues, NotBoundOrWrongSpan_Fails)`: unbound values give `kWrongGridVersion`; bound values with a
    span one entry short give `kBadInput`.
  - `TEST(WriteValues, ThenReadValues_ReturnsTheSameValues)` (CPU) for each type.
  - `TEST(CheckGridVersion, OtherVersion_IsWrongGridVersion)`; `TEST(BuildCellValues, ZeroBlocks_IsBadInput)`.
  Label `no_gpu`: `TEST(BuildFaceValues, OnGpuWithoutAGpu_IsRunOnCallFailed)`.
  Label `gpu`: `TEST(WriteValues, OnGpuThenReadValues_ReturnsTheSameValues)` for each type, bound to a grid built
  on the GPU; `TEST(ApplyGridVersion, OnGpuGrid_LaunchesNothing)` (100 GPU grids bound in turn: free bytes
  unchanged, values' contents read back unchanged, since a bind does not touch them);
  `TEST(ApplyGridVersion, ValuesOnOtherProcessor_IsBadInput)` (CPU values, GPU grid).
- [ ] **Step 2: Run**, fail. **Step 3: Implement**; the header states the rule of design 3.6 (a consumer checks
  `grid_version()` and `block_count()`; values of a piece are bound again at every load). **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Add value buffers that are built once and bound per grid version`. Body: `GPU
  copies and binds: compiled, not run`.

### Task 22: Ranges and their border

**Amended 2026-11-14 (A17):** the per-block border test becomes a shared formula, and a stacked buffer form of
`FindBorderBlocks` (with `BorderSearchBuffers`) is added whose CPU version is the oracle of Task 23.

**Amended 2026-11-14 (Foundation as built):** shared tests `LABEL cpu`; `BorderSearchBuffers` owns the `OrderScratch` the prefix sum takes (scratch form only); allocation counter in the test's sources.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/grid/cell_block_range.h` (CPU only: includes `base/cpu_only.h` first, then
  `grid_ids.h` for `CellBlockRange`), `cell_block_range_cpu.cc`, `border_block_steps.h` (S, private),
  `cell_block_range_test.cc`, `border_block_steps_test.cc` (label `cpu`)
- Modify: `grid_memory_estimate.h`, `grid_memory_estimate.cc`, build registration

**Interfaces:**
- Consumes: `CellBlockRange` from `grid_ids.h` (Task 3); it is not redeclared here. `GridBuildStatus` (Task 6).
- Produces: `RangeWithBorder` (holds a `std::vector`, so CPU only), `BorderWidth`, both `BuildRangeWithBorder`,
  `BuildRanges`, both `FindBorderBlocks` (design 3.4, 3.7), by the single ascending walk of design 5.5 (each block tested against the
  sorted set through its own neighbour tables).
- **Plan decision (A17), shared:** `border_block_steps.h` declares
  `struct BlockSet { CellBlockRange range; const CellBlockIndex* sorted_blocks; std::uint32_t sorted_count; };`,
  `bool IsInBlockSet(const BlockSet& set, CellBlockIndex block)` (range test, then binary search) and
  `bool IsBorderBlock(const GridTables& tables, const BlockSet& set, CellBlockIndex block, BorderWidth width)`:
  the per-block test of design 5.5 through `block`'s own 27 same-level neighbours, its bigger block and its 8
  smaller blocks (the owner's spatial index: no other search structure is built). The CPU walk is "for each
  block in index order, if not in the set and `IsBorderBlock`, append"; the span forms use
  `BlockSet{{0, 0}, blocks.data(), blocks.size()}`.
- **Plan decision (A17), the stacked buffer form** (in `cell_block_range.h`; runs on `grid.run_on()`, CPU here,
  GPU in Task 23):

  ```cpp
  // Room to search the border of any block set of a grid with at most max_cell_blocks blocks. Move-only.
  class BorderSearchBuffers {
   public:
    std::uint32_t max_cell_blocks() const;
    RunOn run_on() const;
  };
  // The one allocation of border search. Fails kBadInput (0 or above kMaxCellBlocksLimit), kRunOnCallFailed,
  // kOutOfMemory.
  std::expected<BorderSearchBuffers, GridFailure> BuildBorderSearchBuffers(RunOn run_on,
                                                                           std::uint32_t max_cell_blocks);
  std::vector<MemoryEstimateEntry> BuildBorderSearchMemoryEstimate(RunOn run_on);
  // Every block outside `range` that is in border_so_far or in the border, at `width`, of range plus
  // border_so_far: one more layer of a stacked border, merged with the layers before. Ascending, each once.
  // Writes the first border_blocks.size() entries and returns how many there are in all. Every buffer is on
  // grid.run_on(). Allocates nothing. Fails kBadInput (a buffer on another processor; border_so_far not
  // ascending, holding a block of `range` or an index >= the grid's block count; border_blocks is
  // border_so_far; border_so_far_count above its size), kTooManyCellBlocks (search_buffers too small for the
  // grid), kRunOnCallFailed.
  std::expected<std::uint32_t, GridFailure> FindBorderBlocks(const StaggeredGrid& grid, CellBlockRange range,
                                                             const Buffer<CellBlockIndex>& border_so_far,
                                                             std::uint32_t border_so_far_count, BorderWidth width,
                                                             BorderSearchBuffers& search_buffers,
                                                             Buffer<CellBlockIndex>& border_blocks);
  ```
  A stacked border is this call once per layer with two output buffers used in turn; no merge is needed between
  layers. Rows (`grid: ` prefix, `KeptFor::kWholeRun`, `copies = 1`): `border marks` (kCellBlock, 4 B) and
  `border marks end` (kGivenCount 1, 4 B), `where = kRunOn`; for `kGpu` also `border ordering`
  (kCellBlock, `ComputeOrderScratchSize(kGpu).bytes_per_item`), `border ordering, fixed part` (kGivenCount 1)
  and `border status` (kGivenCount 1, `sizeof(GridBuildStatus)`). On the GPU the `border ordering` rows are the bytes of an
  `OrderScratch` built once in `BuildBorderSearchBuffers` (`BuildOrderScratch(kGpu, max_cell_blocks + 1)`) and
  passed to every `ComputeExclusivePrefixSum` of the search (scratch form; nothing allocated per call).
  `cell_block_range_test` lists `allocation_counter_test_helpers.cc` in its `SOURCES`.
- **Plan decision (A17):** the design's span forms and `BuildRangeWithBorder` read CPU tables: on a GPU grid the
  span forms fail `kBadInput`, and `BuildRangeWithBorder` (which has no failure value) has the precondition
  `grid.run_on() == RunOn::kCpu`, stated in its comment and checked by a debug assertion. `BuildRanges` reads only
  `counts()` and works on either.
- Test-only helper in `grid_test_helpers`: `std::vector<CellBlockIndex> FindBorderGeometrically(...)` built from
  `GeometricGrid` (touching cells, their pieces, and for `kFourCellWidths` the blocks `FaceSearch` reads).

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(FindBorderBlocks, MatchesGeometricReach)`: 200 random grids, 5 random ascending block sets each, both
    widths: equal to the geometric list; ascending, no repeats, no block of the set; the span form with a span of
    exact size gives the same list, and with a span 1 short writes the first `n - 1` and returns `n`.
  - `TEST(FindBorderBlocks, PlaneFourPieceBeyondTouchingCell_IsInBorder)`: one level, width 1 m, active cells
    `{3,3,0}` (block A at block coordinate `{0,0,0}`), `{4,3,0}` (block B, `{1,0,0}`) and `{6,6,0}` (block C,
    `{1,1,0}`); block indexes A = 0, B = 1, C = 2. Cell `{4,3,0}` touches the range cell `{3,3,0}`; its high Y
    face (an edge piece) is plane 4 of B, so it is stored at plane 0 of C; no cell of C touches `{3,3,0}`. The
    `kTouchingCells` border of `{0}` is exactly `{1, 2}`; a border built from touching cells alone would be `{1}`.
  - `TEST(FindBorderBlocks, StackedBorders_HoldEveryRead)`: 100 random grids cut by `BuildRanges(grid, 4)`; for
    each range, stack layers for the pass-1 widths (touching, touching, four widths, touching), computing from the
    last stage backwards; for each stage, every cell and face it reads from its run set (geometric checker) lies in
    a loaded block.
  - `TEST(FindNearbyFaces, RangeAndBorder_ContainEveryFace)`: random grids cut by `BuildRanges(grid, 3)`; 100
    positions in range cells: every listed face is stored in the range or its `kTouchingCells` border.
  - `TEST(BuildRanges, Cuts_CoverEveryBlockOnceInOrder)`; `TEST(FindBorderBlocks, UnsortedSet_IsBadInput)`.
  - `TEST(FindBorderBlocks, StackedBufferForm_EqualsSpanFormLayerByLayer)`: 100 random grids, ranges from
    `BuildRanges(grid, 4)`, layers (touching, touching, four widths, touching); at each layer the buffer form's
    list equals `std::merge` of `border_so_far` and the span form on range plus `border_so_far`.
  - `TEST(FindBorderBlocks, StackedBufferFormShortOutput_WritesFirstAndReturnsTotal)`;
    `TEST(FindBorderBlocks, StackedBufferFormAliasedOutput_IsBadInput)`;
    `TEST(FindBorderBlocks, BorderSoFarInsideRange_IsBadInput)`;
    `TEST(FindBorderBlocks, StackedBufferForm_AllocatesNothing)` (allocation counter of Task 9).
  - `TEST(BuildBorderSearchMemoryEstimate, MatchesAllocatedBytes)` (CPU rows: `4 * max_cell_blocks + 4`).
  - `TEST(IsBorderBlock, PlaneFourFixture_MarksBlockC)` (label `cpu`; the fixture of
    `PlaneFourPieceBeyondTouchingCell_IsInBorder` built by hand).
  Label `no_gpu`: `TEST(BuildBorderSearchBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed)`.
- [ ] **Step 2: Run**, fail. **Step 3: Implement**. **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Cut the grid into ranges and find their border blocks`. New names include
  `BorderSearchBuffers`, `BuildBorderSearchBuffers`, `BuildBorderSearchMemoryEstimate` and the stacked form.

### Task 23: Border blocks on the GPU

**Amended 2026-11-14 (A17):** new task; the stacked `FindBorderBlocks` runs on the GPU over the grid's
neighbour tables (mark, prefix-sum, scatter) and is compared with the CPU walk of Task 22.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** GPU half `cell_block_range_gpu.h`/`.cu` returning `RunOnCallStatus`, sequenced from `cell_block_range_cpu.cc`, launches with `ComputeLaunchSize`; per-thread bodies in `border_block_steps.h` with a `cpu` launch-order test; digests.

**Files:**
- Create: `native/fluidity/grid/cell_block_range_gpu.h` (private, replaces `border_blocks_gpu_calls.h`),
  `cell_block_range_gpu.cu` (private), `border_block_launch_test.cc` (label `cpu`), `cell_block_range_gpu_test.cc`
  (label `gpu`)
- Modify: `cell_block_range_cpu.cc` (dispatch on `grid.run_on()`, the GPU sequence under
  `#if defined(FLUIDITY_HAS_GPU)`), `border_block_steps.h`, build registration

**Interfaces:**
- Consumes: `BlockSet`, `IsInBlockSet`, `IsBorderBlock` (Task 22), `GridBuildStatus`, the failure key (Task 7);
  Foundation `ComputeExclusivePrefixSum` (scratch form, the `OrderScratch` of `BorderSearchBuffers`), `ReadBuffer`,
  `ToBufferError`, `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`.
- Shared per-thread functions (added to `border_block_steps.h`, plan decision):
  `std::uint64_t ComputeBorderSoFarFailureKey(...)` (step 2, one entry), `std::uint32_t ComputeBorderMark(...)`
  (step 3, one block), `void ApplyBorderBlockWriteAt(...)` (step 5, one marked block). Each kernel is a grid-stride
  loop over one of them (plus an integer `atomicMin` of the key); the GPU half's functions return
  `RunOnCallStatus`, mapped through `ToBufferError` into `GridFailure`.
- Produces: the GPU path of the stacked `FindBorderBlocks` and of `BuildBorderSearchBuffers(kGpu, ...)`.
- **Plan decisions (A17), parallel form:** (1) CPU: processors, aliasing, counts and room checked; a one-thread
  kernel resets the status. (2) One thread per `border_so_far` entry `i`: ascending, outside `range`, below the
  block count, else `atomicMin(ToFailureKey(1, i, 0))` with `kBadInput`. (3) One thread per block `b` of the
  grid (count known on the CPU from `grid.counts()`): mark = `b` outside `range` and (`b` in `border_so_far` or
  `IsBorderBlock(tables, set, b, width)`), with `set = {range, border_so_far, count}`. (4)
  `ComputeExclusivePrefixSum` in place, count `cell_blocks` (the total lands at index `cell_blocks`). (5) Each marked `b` writes itself at
  `border_blocks[sum]` when `sum < border_blocks.size()`; `result_count` = the total. (6) One `ReadBuffer` of the
  status: the count (or the failure) is all that comes back. Index order of `b` is kept by the prefix sum, so the
  list is the CPU's.

- [ ] **Step 1: Tests.** Label `cpu` (`border_block_launch_test.cc`, the main evidence until the hardware run):
  `TEST(FindBorderBlocksLaunch, EveryThreadBlockAndThread_EqualsCpuStackedLayers)`: steps 2 to 5 replayed on CPU
  buffers, each per-thread function for every thread block and thread of `ComputeLaunchSize(count)`, in launch order
  and in reverse, `ComputeExclusivePrefixSum` on the CPU between them; on Task 22's 100 grids, ranges and layers and
  the failing inputs of `OnGpuFailures_EqualCpuFailures`: lists, counts and failures equal the CPU walk's.
  Label `gpu` (every grid built on both processors from one list; lists and counts read back go to
  `WriteResultDigest`):
  - `TEST(FindBorderBlocks, OnGpu_EqualsCpuStackedLayers)`: Task 22's 100 grids, ranges and layers; at each layer
    the GPU list (read back in the test) and count equal the CPU's.
  - `TEST(FindBorderBlocks, OnGpuShortOutput_SameCountAsCpu)`: output buffer of 3 entries.
  - `TEST(FindBorderBlocks, OnGpuFailures_EqualCpuFailures)`: unsorted, inside-range, out-of-range and aliased
    inputs.
  - `TEST(FindBorderBlocks, OnGpu_AllocatesNothing)` and `TEST(FindBorderBlocks, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(BuildBorderSearchMemoryEstimate, MatchesAllocatedGpuBytes)`.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda`.
- [ ] **Step 3: Commit.** Subject: `Find border blocks on the GPU through the neighbour tables`. Body: `GPU:
  compiled, not run`; the six `gpu` tests.

### Task 24: Grid pieces for the batched run

**Amended 2026-11-14 (A17):** adds the buffer form of `BuildGridPiece` and the piece map as a buffer, and puts
the per-block and per-face piece rules in the shared build header for Task 25.

**Kind:** integration

**Files:**
- Create: `native/fluidity/grid/grid_piece.h` (plan decision: the design puts `GridPiece` in
  `cell_block_range.h`; it gets its own header because it owns storage), `grid_piece_cpu.cc`, `grid_piece_test.cc`
- Modify: `cell_block_range.h` (includes `grid_piece.h` so the design's spelling holds), `grid_build_steps.h`,
  `docs/glossary.md` (`grid piece`: one range of a whole grid with its border, loaded for a batch; not a face
  piece)

**Interfaces:**
- Consumes: `BuildGrid` internals (tables layout), `ReadStorageArrays`, `FindBorderBlocks`, `FindSortedBlock`.
- Produces: `GridPiece`, `BuildGridPiece`, `ReleaseBuffers(GridPiece&&)` (design 3.4), by design 5.8 "Pieces keep
  the fixed order".
- **Plan decision (A17), the buffer form and the map as a buffer** (both forms run on `whole.run_on()`; the
  span form needs a CPU whole grid, else `kBadInput`):

  ```cpp
  // Same as the span form, with the border in a buffer on whole.run_on() (a stacked FindBorderBlocks result).
  std::expected<GridPiece, GridFailure> BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range,
                                                       const Buffer<CellBlockIndex>& border_blocks,
                                                       std::uint32_t border_block_count, GridBuffers& buffers);
  // GridPiece member: the piece map on grid().run_on(), [grid().counts().cell_blocks] entries used, ascending.
  const Buffer<CellBlockIndex>& block_in_whole_grid_buffer() const;
  ```
  `block_in_whole_grid()` (the span) is the map for a CPU piece and empty for a GPU piece.
- **Plan decision (A17), shared** (added to `grid_build_steps.h`): `CellBlockIndex ToPiecePlace(...)` (the piece
  index of a loaded block: border entries below the range first, then the range, then the rest, so each
  entry's place is computed alone), `std::uint32_t ToPieceNeighbour(const CellBlockIndex* map, std::uint32_t
  count, std::uint32_t whole_neighbour)` (binary search; `kNoIndex` when not loaded) and
  `FacePieceKind ComputePieceFaceKind(const GridTables& whole, const CellBlockIndex* map, std::uint32_t count,
  CellBlockIndex piece_block, int face_place_in_block)` (the whole grid's kind, `kOutsidePiece` and plane-4
  rules of design 5.8). The CPU form writes every entry through them.

- [ ] **Step 1: Failing tests** (label `cpu`):
  - `TEST(BuildGridPiece, KeepsWholeGridOrder)`: 100 random grids, ranges from `BuildRanges(grid, 4)`, borders
    stacked twice (touching, four widths): `block_in_whole_grid` ascending; `range()` is a run and maps to
    `range_in_whole_grid()`; per block, cell states and face kinds equal the whole grid's except `kOutsidePiece`
    exactly where the other cell's block was not loaded; a plane-4 place whose high neighbour exists in the whole
    grid but was not loaded is `kNotAPiece`.
  - `TEST(BuildGridPiece, SameWeightsAsWholeGrid)`: 100 positions in range cells, all axes, all kept: lists map to
    the whole grid's through `block_in_whole_grid` (place within block unchanged) and every weight field is
    byte-identical.
  - `TEST(BuildGridPiece, BlockSumsInPieceOrder_EqualWholeGrid)`: per-block partial = ordered sum over the block's
    active cells of `ComputeCellCenter(...).z * ComputeCellWidth(...)`; whole-grid total = `math::TreeSum` over the
    partials; for 3 cuts (`max_blocks_per_range` 1, 3, 7) partials computed in pieces are placed at
    `range_in_whole_grid` and tree-summed: identical bytes.
  - `TEST(BuildGridPiece, AboveStorageRoom_IsTooManyCellBlocks)`; `TEST(FindCellAt, InPieceOutsideLoadedBlocks_IsNoIndex)`.
  - `TEST(BuildGridPiece, BufferForm_EqualsSpanForm)`: the 100 grids of `KeepsWholeGridOrder`, border from the
    stacked buffer form of `FindBorderBlocks` on CPU buffers: same tables, map and counts as the span form.
- [ ] **Step 2: Run**, fail. **Step 3: Implement** (allocates nothing; translate neighbour indexes; `kNoIndex` for a
  block not loaded). **Step 4:** gate.
- [ ] **Step 5: Commit.** Subject: `Build a grid piece that keeps the whole grid's block order`. New names include
  the buffer form and `block_in_whole_grid_buffer`.

### Task 25: Grid pieces on the GPU

**Amended 2026-11-14 (A17):** new task; `BuildGridPiece` on a GPU whole grid builds the piece's tables on the
GPU from the whole grid's, compared with the CPU piece bit for bit.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** GPU half `grid_piece_gpu.h`/`.cu` returning `RunOnCallStatus`, sequenced from `grid_piece_cpu.cc`, launches with `ComputeLaunchSize`; step 6 loops thread blocks over piece blocks; `cpu` launch-order test; digests.

**Files:**
- Create: `native/fluidity/grid/grid_piece_gpu.h` (private), `grid_piece_gpu.cu` (private),
  `grid_piece_launch_test.cc` (label `cpu`), `grid_piece_gpu_test.cc` (label `gpu`)
- Modify: `grid_piece_cpu.cc` (dispatch on `whole.run_on()`, the GPU sequence under `#if defined(FLUIDITY_HAS_GPU)`),
  `grid_build_steps.h`, build registration

**Interfaces:**
- Consumes: `ToPiecePlace`, `ToPieceNeighbour`, `ComputePieceFaceKind` (Task 24), `GridBuildStatus`, the failure
  key (Task 7), `ReadGrid` (Task 11, tests); `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`,
  `ToBufferError`. Each kernel is a grid-stride loop over one shared function of `grid_build_steps.h` (the three
  above, plus `ApplyPieceBlockCopyAt` and `ApplyPieceCellStateCopyAt` for steps 4 and 5, added here); counts are
  integer `atomicAdd`s.
- Produces: the GPU path of the buffer form of `BuildGridPiece`.
- **Plan decisions (A17), parallel form** (no sort and no prefix sum: the border is already ascending):
  (1) CPU: processors, `range` inside the whole grid, `range.block_count + border_block_count <=
  buffers.max_cell_blocks()` (else `kTooManyCellBlocks`, `cell` zero); reset the status. (2) One thread per
  border entry: ascending, outside `range`, below the whole block count, else step-1 failure `kBadInput`. (3)
  One thread per loaded block (count known on the CPU): writes the map at `ToPiecePlace`. (4) One thread per
  piece block: copies the block id, translates its 36 neighbour entries with `ToPieceNeighbour`. (5) One thread
  per cell place: copies the state and counts active cells per level. (6) Thread blocks of 240 threads, thread
  block `t` handling piece blocks `t`, `t + (thread blocks)`, ... (`min(piece blocks, kMaxBlockCount)` thread
  blocks): `ComputePieceFaceKind`, pieces counted per thread block then one integer `atomicAdd`. (7) One `ReadBuffer` of the
  status. Steps 3 to 6 return at once after a failure of step 2.

- [ ] **Step 1: Tests.** Label `cpu` (`grid_piece_launch_test.cc`, the main evidence until the hardware run):
  `TEST(BuildGridPieceLaunch, EveryThreadBlockAndThread_EqualsCpuPiece)`: steps 2 to 6 replayed on CPU buffers,
  each per-thread function for every thread block and thread of its launch, in launch order and in reverse, atomics
  replaced by plain minimum and addition; on Task 24's 100 grids, ranges and stacked borders and the failing inputs
  below: tables, map, counts and failures equal the CPU piece's byte for byte.
  Label `gpu` (whole grids built on both processors from one list; tables, maps, counts and failures read back go
  to `WriteResultDigest`):
  - `TEST(BuildGridPiece, OnGpu_EqualsCpuPiece)`: Task 24's 100 grids, ranges and stacked borders (each processor
    finding its own border with the buffer form of `FindBorderBlocks`): `ReadGrid(piece.grid(), ...)` tables,
    the map (read back), version and counts equal the CPU piece's byte for byte.
  - `TEST(BuildGridPiece, OnGpuFailures_EqualCpuFailures)`: unsorted border, a border block inside the range, a
    border index past the whole grid, too little room.
  - `TEST(BuildGridPiece, OnGpu_AllocatesNothing)`; `TEST(BuildGridPiece, OnGpuTwice_GivesIdenticalBytes)`.
  - `TEST(BuildGridPiece, SpanFormOnGpuWhole_IsBadInput)`.
- [ ] **Step 2: Run** the gate. Expected: `PASS cuda`.
- [ ] **Step 3: Commit.** Subject: `Build grid pieces on the GPU in the whole grid's order`. Body: `GPU:
  compiled, not run`; the five `gpu` tests.

### Task 26: Scene-level tests and comparison with `main`

**Amended 2026-11-14 (A17):** adds the wave-tank band built, cut and pieced on the GPU and compared with the CPU.

**Amended 2026-11-14 (Foundation as built):** the GPU weights come from Task 20's test-only runner; the `gpu` test writes result digests.

**Kind:** integration

**Files:**
- Create: `native/fluidity/grid/grid_scene_test.cc`, `native/fluidity/grid/grid_main_comparison_test.cc`,
  `native/fluidity/grid/grid_scene_gpu_test.cc` (label `gpu`),
  `native/fluidity/grid/testdata/main_uniform_weights.inc` (generated, below)

- [ ] **Step 1: Scene test** (label `cpu`): `TEST(StaggeredGrid, WaveTankBand_PassesEveryCheck)`: frame width 1 m,
  3 levels, level-0 box 16 x 4 x 8; every level-2 cell whose centre is within 1 m of the plane `z = 4 + 0.25 x`
  is active (with its siblings), then `ApplySizeStepClosure` fills the rest. Build; check counts against `ComputeGridCounts`; cut into
  ranges of 8 blocks; for every range build a piece with a stacked touching border; at 1,000 positions in the band,
  piece weights equal whole-grid weights byte for byte and sum to 1 within `1e-14`.
  Label `gpu`: `TEST(StaggeredGrid, WaveTankBandOnGpu_EqualsCpu)`: the same cell list put on the GPU; the GPU grid
  equals the CPU grid (`ReadGrid`, byte for byte); for every range of 8 blocks, the GPU's stacked touching border
  and GPU piece equal the CPU's; `ComputeNearbyFaceWeights` run through Task 20's runner
  (`grid_functions_test_gpu.cu`, also in this test's `SOURCES`) on the GPU piece at the 1,000 positions matches the
  CPU piece within the CPU-versus-GPU tolerance (`CheckWithinTolerance`); grids, borders, pieces and weights read
  back go to `WriteResultDigest`.
- [ ] **Step 2: Produce `main`'s numbers.** In the scratchpad: `git worktree add --detach <scratch>/main-build main`,
  build its `macos-release` preset, and write a scratch driver (not committed) that calls
  `BuildMacFaceGeometryStencil3D` (`main:native/include/fluidity/host/sparse_mac_grid_3d.h:59-64`) on a uniform
  geometry, origin 0, width 0.25 m, for 100 positions `p_i = {1.0 + 0.0071 i, 1.0 + 0.0053 i, 1.0 + 0.0037 i}` (Y-up)
  and each axis, printing with `%.17g` the face position, weight, `face_offset_from_particle` and
  `weight_gradient` of the 27 samples. Rotate positions, offsets and gradients with `new = {x, -z, y}` and map the
  old axis (X, Y, Z) to the new (X, Z, Y). Write the rows into `main_uniform_weights.inc` with a header comment
  naming the commit of `main` used and the driver's formula. Remove the worktree.
- [ ] **Step 3: Comparison tests** (label `cpu`, supporting evidence only):
  - `TEST(ComputeNearbyFaceWeights, CompleteCase_MatchesMainWithinRelativeTolerance)`: same rotated positions on a
    uniform 16^3 grid at width 0.25 m with origin `{0, -4, 0}`; faces matched by position; weight, offset and
    `weight_change` within `1e-13` relative.
  - `TEST(BuildGrid, MainJunctionFixture_HasSamePiecesAsMain)`: `main:native/tests/test_adaptive_mac_topology_3d.cc`
    `MakeJunction(3, true)` (lines 22-37, finest width 0.5 m, all 8 small cells present), rotated: 24
    `kBetweenCells` pieces, all 0.25 m2; 12 with centre distance 0.75, 12 with 0.5 (the old test's own assertions,
    lines 166-177; no run of `main` needed).
- [ ] **Step 4: Run** the gate. **Step 5: Commit.** Subject: `Check the grid on a wave-tank band and against the
  proof of concept`. Body names the `main` commit the numbers came from.

### Task 27: Comment pass and hand-over

**Amended 2026-11-14 (A17):** the deviation list and the GPU test list follow the GPU build.

**Amended 2026-11-14 (Foundation as built):** the deviation list names the GPU halves' private headers, `ComputeGridScratchSize` and the dropped launch-shape helper.

**Kind:** mechanical

**Files:**
- Modify: any file under `native/fluidity/grid/`, `docs/glossary.md`, `docs/method-paper/`, `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every grid file top to bottom as a C++ programmer new to fluids: every
  comment true of the final code, says why, has a worked example where a contract is hard to picture (the
  two-size fixture, the complete case, the plane-4 rule); every word in the glossary; `Uses`/`Used by`/`Read:`
  correct.
- [ ] **Step 2: Fold** each change into its owning commit (`git commit --fixup`, `git rebase --autosquash`); run
  `--only format` and `--only rules` on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3.1 to 3.7, 5, 6, 7, 10) and record in the hand-over what
  was delivered differently: `GridError` in `grid_error.h`; `GridPiece` in `grid_piece.h`; the one-argument
  `ComputeGridCounts` and `CheckSizeStepRule` sort a CPU copy of the ids; `CellBlockRange` declared in the shared
  `grid_ids.h` rather than `cell_block_range.h`; and, from A17 (replacing design decision 8 and sections 5.1
  step 9, 5.8 "upload", 6 "CPU" and "Per step for the GPU run"): a GPU grid is built on the GPU from a GPU cell
  list (`GridInputs::active_cell_buffer`), keeps no CPU copy of its tables (`tables(kCpu)` is empty; `ReadGrid`
  copies on request), returns only `GridBuildStatus`; `kTooManyCellBlocks` names no cell; `CheckSizeStepRule`
  does not check siblings; the buffers forms of `ComputeGridCounts` and `CheckSizeStepRule`; the stacked
  buffer form of `FindBorderBlocks` with `BorderSearchBuffers`; the buffer form of `BuildGridPiece` and
  `block_in_whole_grid_buffer()`; `ApplyGridVersion` rejects values on another processor; `max_cell_blocks` is
  capped at `kMaxCellBlocksLimit`; GPU memory per storage is 1,996 B per block plus Foundation's ordering
  scratch, CPU memory 0. `staggered_grid_gpu.cu`, `cell_block_range_gpu.cu` and `grid_piece_gpu.cu` exist, each
  declared in its private `_gpu.h`, returning `RunOnCallStatus` and launching through `base/gpu_launch.cuh` with
  Foundation's `ComputeLaunchSize` (the plan's `ComputeLaunchShape` was dropped); `grid_values_gpu.cu` does not
  (values cross only through `base/` `Buffer` calls). `GridScratch` has `ComputeGridScratchSize` and
  `byte_count()` (Foundation's scratch pattern).
- [ ] **Step 4: `docs/gpu-runs.md`:** add the grid row, result `compiled, not run`, notes listing the `gpu` tests
  of Tasks 10, 11, 20, 21, 23, 25 and 26 by full name (`Suite.Name`), among them
  `BuildGrid.OnGpu_EqualsCpuTablesBitForBit`, `CheckSizeStepRule.OnGpuSeveralFailures_SameFirstFailureAsCpu`,
  `BuildGrid.OnGpuFailedBuildThenRebuild_EqualsFreshBuffers`, `ComputeMemoryBytes.MatchesAllocatedGpuBytes`,
  `ComputeNearbyFaceWeights.OnGpu_MatchesCpu`, `WriteValues.OnGpuThenReadValues_ReturnsTheSameValues`,
  `FindBorderBlocks.OnGpu_EqualsCpuStackedLayers`, `BuildGridPiece.OnGpu_EqualsCpuPiece`,
  `StaggeredGrid.WaveTankBandOnGpu_EqualsCpu`. Commit with subject `Record the grid's GPU status`.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the grid commits, the gate output, the new
  public names, the deviations of Step 3, and the statement that no grid device code has run.
