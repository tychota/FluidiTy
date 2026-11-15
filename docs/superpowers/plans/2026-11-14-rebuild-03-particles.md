# Rebuild Particles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `native/fluidity/particles/`: every material's per-particle arrays with stable ids, the try and its
commit without a copy, grouping by cell block, add, remove, split, merge and the count rule run inside a resize,
snapshots, and the copies of particle groups for a batched run, on the CPU and (compiled, not run) on the GPU.

**Architecture:** One owner, `Particles`, holds two buffers per array per material; a `ParticlesTry` hands out
read and write views and `CommitTry` flips which buffer is committed. Grouping sorts by (cell place, id) with two
`BuildStableOrder` passes. Split, merge and refill formulas live in one shared header used by the CPU loops and
the GPU kernels. Every operation is one public header with private `_cpu.cc` and, where it has device work,
`_gpu.cu` declared in a private `_gpu.h` (Foundation's two halves; amended 2026-11-14).

**Tech Stack:** C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest), uv, clang-format and
clang-tidy 23.1 (the versions `pyproject.toml` pins), Typst + CeTZ for the method paper, Lean 4 + Mathlib (one proof file).

**Spec:** `docs/superpowers/specs/modules/03-particles-design.md` ("the design"; in the `main` checkout,
untracked) and `docs/superpowers/specs/2026-11-13-production-rebuild-design.md` (committed on `rebuild` as
`docs/design/programme.md`). Cross-module names: `docs/superpowers/specs/modules/shared-interface.md`.

## Global Constraints

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes the `AGENTS.md` checklist; new public
  names listed in the commit message with a one-line reason; tests arrive with the code. A commit must pass every
  part of the gate that exists at that commit.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages end with one `Assisted-by: Claude Code:<model id>` line per model that worked on the commit
  (the implementer's model, for example `claude-opus-5-5`, plus `claude-fable-5-2` for the session that directs and
  reviews), then `Claude-Session: <session link>`. No `Co-Authored-By`
  line; the owner is the sole author (owner answer A39, 2026-11-15).
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. The owner runs on
  hardware once, after all modules are written, so Foundation ends without a hardware run. Never claim otherwise.
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
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure applies to every task (`AGENTS.md`
  "Structure"; `specs/modules/plan-decisions.md` "Foundation after the final review"). Two halves: the public
  function lives in `<name>_cpu.cc` (checks, `RunOn` dispatch, the CPU loop, and the GPU half under
  `#if defined(FLUIDITY_HAS_GPU)`); the GPU half `<name>_gpu.cu` is declared in the private `<name>_gpu.h`, returns
  `RunOnCallStatus`, which the CPU half maps with `ToBufferError` and then into `ParticlesFailure` through one
  mapping function, and launches, waits and holds temporary memory only through `base/gpu_launch.cuh`
  (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`). Every GPU function is a thin loop over
  `FLUIDITY_CPU_AND_GPU` per-item or per-thread functions in a shared header (`<name>_steps.h` unless an existing
  shared formula header holds them); `cpu` tests run those functions over every item, or every thread block and
  thread of `ComputeLaunchSize(count)` in launch order and in reverse, against the CPU loop. Working buffers
  (`<Name>Buffers`, `Build<Name>Buffers`, `Compute<Name>BuffersSize`): every function that needs working memory
  has one form, which takes its buffers; callers pass buffers built once and owned by the module's storage
  (`sum_buffers`, `order_buffers`), and the memory estimate counts them with `ComputeSumBuffersSize` /
  `ComputeOrderBuffersSize`. Buffers and their working buffers on the same processor are a precondition
  (`Requires:`, asserted in debug builds), not an error value (A30). Errors: a failed check writes nothing; after
  `kRunOnCallFailed` the outputs are unspecified and the run stops; `kOutOfMemory` comes only from building
  functions and from `ComputeAmounts`' one-call GPU memory (Task 17). Exactness: discrete outputs equal on CPU and GPU; floating-point within the module's tolerance
  unless the function's header promises the same bits unless NaN. Tests link `fluidity_base_test_support`
  (`ToBits`, `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`, `CheckSameBitsOrBothNaN`) and the module's
  helper library made with `fluidity_add_test_library`, named in `LINKS`; `gpu` tests call `WriteResultDigest` on
  their main outputs. Labels: `cpu` everywhere, `no_gpu` only for behaviour on a machine without a GPU, `gpu` only
  on hardware. Every target is created through the helpers of `cmake/fluidity_targets.cmake` and every source file
  is listed in a CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the
  release preset, and CUDA compiled in the container (its `cuda` step fails for a tracked `.cc` or `.cu` file that no
  build compiles), so a task that adds a source adds it to a CMake file in the same commit.

Particles-specific (every task respects these):

- **Folder and files.** `native/fluidity/particles/`, namespace `fluidity::particles`. Public headers are the
  design's section 3 names. **Amended 2026-11-14 (Foundation as built):** device work is reached from the
  `_cpu.cc` file through a private `<operation>_gpu.h` (no CUDA header, no `std::expected`; returns
  `RunOnCallStatus`) defined in `<operation>_gpu.cu`, called only under `FLUIDITY_HAS_GPU`, the pattern of
  `base/buffer_gpu.h` and `base/ordered_sums_gpu.h`. The lowest failing id of a GPU check is written by an integer
  `atomicMin` into a one-entry `Buffer<std::uint64_t>` that `particles_storage.h` holds (memory estimate entry
  `"particles: lowest failing id"`, `kGivenCount` 1, 8 B, `kWholeRun`), and the CPU half reads it back; a
  one-thread GPU function writes its failure into a one-entry `Buffer<ParticlesFailure>` held beside it (entry
  `"particles: GPU failure"`, `kGivenCount` 1, `sizeof(ParticlesFailure)`). A `.cu`
  file never includes a CPU-only header. Private shared header `material_arrays_view.h` holds the raw pointers a
  kernel needs; private CPU-only header `particles_storage.h` holds the buffers.
- **Index convention (plan decision; the design left it open).** Every array pointer in `ParticleFields` and
  `ParticleFieldsToChange` points at particle 0 of the material. `group_begin` points at entry
  `cell_blocks.first_block` of the material's table, so its values are whole-material indexes; `particle_count`
  is `group_begin[block_count] - group_begin[0]`. A range's particles are `[group_begin[0], group_begin[block_count])`.
  A `particle_index` passed to `RemoveParticles`, `SplitParticle`, `MergeParticles` is such an index. A range
  outside the current grouping, or a material with no grouping yet, gives a view with `particle_count = 0` and
  null pointers.
- **Views name their processor (coordinator decision, changes design 3.1).** `ParticleFields` and
  `ParticleFieldsToChange` each gain the member `RunOn run_on = RunOn::kCpu;` (after `cell_blocks`): `kCpu` when
  the pointers are CPU pointers, `kGpu` when they are GPU pointers. Every `BuildFieldsToRead` and
  `BuildFieldsToChange` sets it from the particles. No code asks the CUDA runtime where a pointer lives.
- **Amounts return failures (coordinator decision, changes design 3.2).** Retyped declarations:

  ```cpp
  // Chooses the processor from fields.run_on. kOutOfMemory (the GPU path's one-call GpuArray of one
  // ParticleAmounts per block), kRunOnCallFailed. Never NaN as a failure signal.
  std::expected<ParticleAmounts, ParticlesFailure> ComputeAmounts(const ParticleFields& fields);
  // kBadInput when per_block.size() != fields.cell_blocks.block_count; kOutOfMemory, kRunOnCallFailed.
  // A failed check leaves per_block unchanged; after kRunOnCallFailed per_block is unspecified (amended
  // 2026-11-14, Foundation as built).
  std::expected<void, ParticlesFailure> ComputeAmountsPerBlock(const ParticleFields& fields,
                                                               std::span<ParticleAmounts> per_block);
  ```
- **Raising the block limit (coordinator decision, changes design 3.2).** Retyped declaration:

  ```cpp
  // Not inside a try (kTryInProgress). max_particle_counts: one per material, each at least the current count;
  // max_cell_blocks: at least settings' current value (so at least the committed grouping's block count).
  // Reallocates the particle arrays and the per-block tables (group_begin, cell_begin, written marks, loaded
  // block origin) at the new limits and copies the committed values. kBadInput, kOutOfMemory, kRunOnCallFailed;
  // on failure the particles are unchanged and usable.
  std::expected<void, ParticlesFailure> ApplyParticleLimits(Particles& particles,
                                                            std::span<const std::size_t> max_particle_counts,
                                                            std::uint32_t max_cell_blocks);
  ```
- **Constants (exact, from the design):** `kMaxIngredientCount = 8`; flag bits `kCarriesFittedChange = 1`,
  `kHasOwnMotion = 2`; `kFractionSumTolerance = 1e-12`; `kMergeGapFloor = 1e-9`; split offset of a unit cell
  `0.4330127018922193`; ids start at 1, are never zero, never reused.
- **Two buffers per array (design 5.3), per block (plan decision).** Each material keeps, per cell block, a
  `FieldList` of arrays written in this try ("written marks", 2 B per block). `BuildFieldsToChange` for a range:
  all blocks of the range unwritten → `read` = committed buffer, `write` = the other; all written → both the try
  buffer; mixed → first copy the committed values of the unwritten blocks of the range into the try buffer
  (`CopyBuffer` slice per run of blocks), then both the try buffer. It then marks the range written. A write
  pointer never points at a committed buffer. `velocity.for_this_step` has one buffer; its read pointer is null
  in a view unless every block of the range was written in this try.
- **Grouping (design 5.8, plan decisions).** Keys: pass 1 `BuildStableOrder` by id, pass 2 by `CellPlaceIndex`
  of `grid::FindCellAt(tables, position)` (binary-search form). Removed particles get key `UINT64_MAX` and are
  dropped. Input = grouped particles then waiting particles. An array whose newest values are committed is
  gathered into the other buffer; an array already changed in this try (second grouping in one try, as in a
  resize) is regrouped **in place** through an 8-byte-per-particle lane copy: for each 8-byte lane of the element
  (1 byte for `std::uint8_t` arrays), gather the lane into the copy, then write it back. Committed buffers are
  never written.
- **Removal marks (plan decision).** 1 byte per grouped particle, try only: 0 kept, `1 + RemoveReason` removed,
  255 replaced by a split or a merge (not a removal, not reported).
- **Waiting list (plan decision).** One per material, `max_added_per_try` entries of each array the material
  carries except `position_when_grouped` and `velocity.for_this_step`. `waiting_count()` is the sum over
  materials. Ids come from one `next_particle_id` shared by all materials.
- **`ParticlesSettings::max_cell_blocks` (plan decision; the design has no count to size `group_begin`,
  `cell_begin` and the written marks).** `std::uint32_t max_cell_blocks = 0;` added after `max_added_per_try`:
  the grid's `GridSettings::max_cell_blocks` (for a loaded try, the batch plan's `max_loaded_blocks`). Zero is
  `kBadInput`. `ApplyParticleLimits` raises it together with the grid limit (above). A grid with more blocks
  than this is `kTooManyParticles` at grouping, with `magnitude` the block count.
- **Sum orders.** `ComputeAmounts`: per block, sequential in grouped order (`mass`, `rest_volume`,
  `volume_ratio * rest_volume`, `mass * velocity.for_next_step` per component), then `math::TreeSum` over block
  partials, per component. Add and remove amounts: sequential in input / index order. Merge sums: sequential in
  ascending id. Split count changes and refusals in `AdjustParticleCount`: block partials, then `math::TreeSum`.
- **Order of new ids in `AdjustParticleCount` (plan decision).** Blocks in order; inside a block, first the
  split children of each split particle in grouped order (cell place, then id), eight consecutive ids per
  parent in child order; then the merged particles in cell-place order, chunks in order.
- **Tolerances (design 6.6).** Ids, counts, order, flags, `size_level`, `group_begin`, `cell_begin`: exact.
  Floats, CPU against GPU: `|a - b| <= floor + 1e-13 * max(|a|, |b|)`, floor `1e-13 * cell_width` (positions),
  `1e-13 * largest speed in the case` (velocities), `1e-13 * largest value in the case` (amounts, fitted
  change). Repeat runs on one processor: identical bytes. **Amended 2026-11-14 (Foundation as built):** one
  exception, by header promise: `ComputeAmounts` and `ComputeAmountsPerBlock` give the same bits on CPU and GPU
  unless the result is NaN (Task 17), because both processors run the same shared per-block function (additions and
  multiplications only, in a fixed order, no fused multiply-add under the flags) and the tree sum over block partials
  runs on the CPU for both.
- **Test fixtures** (`particles_test_support.h`, Task 9, in the test library `fluidity_particles_test_support`
  made with `fluidity_add_test_library` and linked by every particles test through `LINKS`; tests use only these
  and `fluidity_base_test_support` unless a task says otherwise).
  Frame: origin `{0, 0, 0}`, `biggest_cell_width = 1`, `size_level_count = 3` (widths 1, 0.5, 0.25). Domain:
  level-0 cells x 0..15, y 0..7, z 0..7. `GridSettings{.max_cell_blocks = 64}`.
  `BuildBigCellsGrid` (all level 0); `BuildCentreSplitGrid` (big cells with x, y, z in 2..5 split into level 1);
  `BuildCornerSplitGrid` (x, y, z in 4..7 split); `BuildFarSplitGrid` (x in 10..13, y, z in 2..5 split);
  `BuildCentreAndFarSplitGrid`; `BuildThreeBlockGrid` (level-0 cells x 0..7, y, z 0..3, big cell (4,0,0) split:
  blocks in order L0 (0,0,0), L0 (1,0,0), L1 (2,0,0)). Each takes `(GridVersion, RunOn)`. Points: centre cell
  `P = {3.5, 3.5, 3.5}` (big cell (3,3,3)), far cell `Q = {11.5, 3.5, 3.5}`. Materials: `LiquidSettings`,
  `MixtureSettings(3)`, `SnowSettings`, `SpringySolidWithSideSettings`, `GasMarkerSettings`,
  `LiquidWithPlateSideSettings` (the six rows of design 5.1). Values: `SplitMix64` with seed `20261114`, doubles
  as `(x >> 11) * 0x1p-53` (never `<random>` distributions: they differ between standard libraries).
  Named particle sets:
  - *Split parent:* level 0, m = 8, `rest_volume` 2, `volume_ratio` 0.5, v = {1,2,3}, C = rows {1,2,3},{4,5,6},
    {7,8,9}, `kCarriesFittedChange` on; at `{0.5,0.5,0.5}` (`H = 1`) in formula tests, at `P` or `Q` in grid tests.
  - *Eight close:* level 1, m = 1, `rest_volume` 0.125, at `P + (+-0.125, +-0.125, +-0.125)`, v = 0, C = 0,
    `kCarriesFittedChange` on.
  - *Six on three blocks* (`BuildThreeBlockGrid`, liquid), added in this order so ids are 1 to 6:
    `{5.5,0.5,0.5}`, `{0.5,0.5,0.5}`, `{4.25,0.25,0.25}`, `{0.6,0.4,0.5}`, `{4.75,0.25,0.25}`, `{1.5,0.5,0.5}`.
    Grouped: ids 2, 4, 6 | 1 | 3, 5; `group_begin {0, 3, 4, 6}`; `cell_begin` block 0: place 0 -> 0, place 1 -> 2,
    later places -> 3; block 1: places 0 and 1 -> 0, later -> 1; block 2: place 0 -> 0, place 1 -> 1, later -> 2.
  - *Cloud:* 200 liquid particles, positions `SplitMix64`-uniform in `[1, 7)^3`, v in `[-1, 1]^3`, C = 0,
    m = `rest_volume` = 0.001, on `BuildCentreSplitGrid`.
  - *Solid face:* `FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{1}, CellCoord{9, 7, 7}))` of
    `BuildCentreSplitGrid` (x = 4.5, near split child 7 of a parent at `P`, far from child 0).
  - `BuildNeighbourhoodBuffers(grid)`: every face outside solids, every side `{0, kNoPlate}`, no grid-only cell.
  - `RunTestStep(Particles&, grid, GroupingSettings, double step_duration)` (Task 13): `BuildTry`; `CheckGrouping`,
    on failure `GroupParticlesByBlock`; per material over `all_blocks()` `BuildFieldsToChange({kPosition,
    kVelocityForNextStep})`; `v += {0, 0, -9.81 dt}`, then `x += v dt`; `CommitTry(time + dt)`.
  - Test-only setters in `particles_storage.h` (Task 10): `ApplyNextParticleIdForTest(Particles&, std::uint64_t)`,
    `ApplyCommittedIdForTest(Particles&, material_index, particle_index, std::uint64_t id)`.
- **Memory estimate entries** (`BuildParticlesMemoryEstimate`, plan decision; the design's 32 B of working memory is split into what is
  really allocated). Per material, all `CountedPer::kGivenCount`. Committed + try (`kWholeRun`, `copies = 2`,
  `given_count = max_particle_count`): `id` 8, `position` 24, `position_when_grouped` 24,
  `velocity.for_next_step` 24, `fitted_velocity_change` 72, `mass` 8, `rest_volume` 8, `volume_ratio` 8, `flags`
  1, `size_level` 1; when carried `ingredient_fractions` 8 N, `springy_deformation` 72, `permanent_volume_change`
  8, `plate_side` 8; `group_begin` 8 (`max_cell_blocks + 1`); `cell_begin` 4 (`max_cell_blocks * 64`). Try only
  (`kOneStepTry`, `copies = 1`): `velocity.for_this_step` 24 and removal marks 1 (`max_particle_count`); written
  marks 2 and loaded block origin 4 (`max_cell_blocks`). Working buffers (`kOneCall`): grouping keys 8 and two orders
  4 + 4 (`max_particle_count + max_added_per_try`); lane copy 8 (`max_particle_count`); waiting list
  (`max_added_per_try`, bytes = the carried arrays without `position_when_grouped`). One liquid: 2 x 178 + 24 +
  1 + 24 = 405 B per particle (design 6.4 says 412). **Amended 2026-11-14 (Foundation as built):** once per
  `Particles` (not per material), `kWholeRun`, `where = kRunOn`: `"particles: order buffers"` (`kGivenCount`,
  `given_count` = the largest `max_particle_count + max_added_per_try` over materials, `bytes_each =
  ComputeOrderBuffersSize(run_on).bytes_per_item`), `"particles: order buffers fixed bytes"` (`kGivenCount`,
  `given_count = ComputeOrderBuffersSize(run_on).fixed_bytes`, `bytes_each = 1`), `"particles: sum buffers"`
  (the largest `max_particle_count`, `ComputeSumBuffersSize(run_on).bytes_per_item`), `"particles: sum buffers
  fixed bytes"`, `"particles: lowest failing id"` (1 x 8) and `"particles: GPU failure"` (1 x
  `sizeof(ParticlesFailure)`). These hold the `OrderBuffers` of the two
  `BuildStableOrder` passes and of `ComputeExclusivePrefixSum`, and the `SumBuffers` of `ComputeLargestMagnitude`,
  members `order_buffers` and `sum_buffers` of `particles_storage.h`, built by `BuildParticles` and rebuilt by
  `ApplyParticleLimits` from `settings.run_on` and the limits it already has (no signature changes). Per particle
  that adds 4 B on the CPU and 6 + 14 = 20 B on the GPU to the 405 B above.
- **Proof of concept is Y-up.** Rotate its vectors and matrices with `R = rows {1,0,0},{0,0,-1},{0,1,0}`
  (`v' = R v`, `C' = R C R^T`), match split children by position, never by child order.
- Split, merge and `AdjustParticleCount` act only on materials with `CountRule::kMayBeSplitAndMerged`.
- No floating-point atomics. Integer atomics only for "lowest failing id". A failure names the lowest id.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`: Foundation complete
(including `base/` Tasks 12 to 16: `scalar_math.h`, `buffer.h` with slice forms and `CpuMemory`, `tree_sum.h`,
`ordered_sums.h`, `stable_order.h`, `per_particle_velocity.h`, `memory_count.h`; **amended 2026-11-14
(Foundation as built):** also `run_on_call.h`, `gpu_launch.cuh`, `ToBufferError`, `SumBuffers` and `OrderBuffers`
(the sums and the stable order, each in its one form taking buffers), `test_support.h` in the library
`fluidity_base_test_support`, and `fluidity_add_test_library`; read `AGENTS.md` "Structure" before Task 1), the contact module
(`contact::PlateSide` in `contact/solid_shapes.h`) and the grid module (design 3.1 to 3.7, including
`GridPiece`, `BuildGridPiece`, `FindBorderBlocks`, `BuildRanges`, `ComputeCompleteWeightedSpread`). Prove it:

```bash
cd /Users/tychota/Documents/Code/openSource/FluidiTy-rebuild && uv run python tools/check_commit.py
```

Expected: seven `PASS` lines (format, rules, cpu debug, cpu release, tidy, paper, cuda). Read `AGENTS.md`, `docs/glossary.md` and the design in full before Task 1.

## Review Focus

1. **Second grouping in one try** (a resize groups, splits and merges, then groups again): the arrays are
   already in the try buffers, so the regroup is in place; the committed values must survive for `RejectTry`.
   Pinned: Task 12, `GroupParticlesByBlock, TwiceInOneTry_KeepsCommittedValues`; GPU in Task 15.
2. **A second range of an array already changed for another range** in the same try: the read pointer must
   show committed values for the new range, not stale try-buffer bytes. Pinned: Task 13,
   `BuildFieldsToChange, SecondRangeOfSameArray_ReadsCommittedValues`.
3. **A removed particle that left the grid** (the step removes particles that left the domain, then groups):
   it must be dropped, not reported `kOutsideGrid`. Pinned: Task 16,
   `GroupParticlesByBlock, RemovedParticleOutsideGrid_IsDropped`.
4. **Merge far from the origin**: sums relative to the first particle; a group at `x = 1e6` must give the same
   relative result as at the origin. Pinned: Task 6, `ComputeMergedParticle, FarFromOrigin_SameAsNearOrigin`.
5. **Batch copy-in when the whole try wrote an array for some blocks only**: `WriteParticleGroups` must take
   the newest value per block. Pinned: Task 23, `WriteParticleGroups, WholeWrittenForSomeBlocks_CopiesNewestValues`.

---

### Task 1: Check that the shared views can be declared

**Kind:** mechanical

No file change, no commit. `particle_fields.h` is shared and holds a `grid::CellBlockRange` and a `RunOn`.

- [ ] **Step 1: Check.** Run `grep -n "struct CellBlockRange" native/fluidity/grid/grid_ids.h` and
  `grep -rn "enum class RunOn" native/fluidity/base/`. Expected: `CellBlockRange` is declared in
  `grid/grid_ids.h`, and `RunOn` is declared in a `base/` header that does not include `base/cpu_only.h`.
  If either is not so, stop and tell the coordinator; do not change the grid or `base/` files.

### Task 2: Method paper, particles part 1: what a particle carries, order, try, add and remove, copies

**Kind:** mechanical

Load `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz`. Use `figure_style.typ` and
`pseudocode.typ` exactly as Foundation's figures do.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ` (new section after the grid section, before References)
- Create: `docs/method-paper/figures/particle_fields.typ`, `figures/particle_grouping.typ`, `figures/particle_try.typ`

**Interfaces:**
- Produces labels: `<sec-particles>`, `<sec-particle-fields>`, `<tab:particle-fields>`, `<tab:particle-memory>`,
  `<sec-grouping>`, `<sec-particle-try>`, `<sec-add-remove>`, `<sec-particle-copies>`, `<fig-particle-fields>`,
  `<fig-grouping>`, `<fig-particle-try>`, `<proc-group-particles>`, `<proc-commit-try>`.

- [ ] **Step 1: Text.** `= Particles <sec-particles>` with subsections from design 5.1 (fields, per-material
  layout, fractions stored not amounts), 5.2 and 5.8 (identity and order: ids, (cell place, id), why indexes do
  not outlive a grouping, `position_when_grouped`), 5.3 (try and commit, per-block written marks of this plan),
  5.7 (add, refill placement, remove), 5.9 and 5.10 (snapshots, loaded groups). `<tab:particle-fields>`: one row
  per array with unit and who has it. `<tab:particle-memory>`: the memory estimate entries of Global Constraints.
  Cite [1], [5], [17], [19], [3] where design 8 says. Decided vocabulary; each literature term once in parentheses.
- [ ] **Step 2: Figures.** `<fig-particle-fields>`: one row per material (one liquid, mixture of 3, snow,
  springy solid with side, gas markers), a box per array, extra arrays shaded. `<fig-grouping>`: the "six on
  three blocks" fixture before and after grouping, indexes change, ids do not, with its `group_begin` and
  `cell_begin`. `<fig-particle-try>`: two buffers per array,
  a try writing two arrays, the flip at commit. Compile, look at each page, fix overlaps and clipped labels.
- [ ] **Step 3: Pseudocode.** `<proc-group-particles>` steps: 1 find each particle's active cell (removed
  ones skipped); 2 report the lowest outside id; 3 build keys; 4 order by id, then by cell place; 5 move each
  array (gather, or in place by lanes); 6 set `position_when_grouped`; 7 fill `group_begin`, `cell_begin`;
  8 record the grid version. `<proc-commit-try>` steps: 1 check written marks; 2 flip changed arrays; 3 take the
  try's counts, tables, next id and grid version; 4 set time, add one to `state_version`.
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`, then
  the gate. **Commit.** Subject: `Describe what a particle carries and how its arrays are kept`.

### Task 3: Method paper, particles part 2: split, merge, keeping the count right

**Kind:** numerics

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ`
- Create: `docs/method-paper/figures/particle_split.typ`, `figures/particle_merge.typ`

**Interfaces:**
- Produces labels: `<sec-split>`, `<eq:split-offset>`, `<sec-merge>`, `<eq:merge-fit>`, `<eq:merge-gap>`,
  `<sec-adjust-count>`, `<fig-split>`, `<fig-merge>`, `<proc-split-particle>`, `<proc-merge-particles>`,
  `<proc-adjust-particle-count>`.

- [ ] **Step 1: Text.** From design 5.4 to 5.6: derive `offset = sqrt(spread(H) - spread(H/2)) = (sqrt(3)/4) H`
  as `<eq:split-offset>` and show in four lines why kinetic energy including the fitted part is kept; merge
  formulas of 5.5 with `<eq:merge-fit>` (the `C*` line) and `<eq:merge-gap>` (the gap line), the energy lost,
  why a positive gap means no energy is gained (cite the Lean theorem named below); the two grids of a
  resize (C2) and the drifted particle that keeps its size level; the id order of `AdjustParticleCount` from
  Global Constraints. Cite [1] and [2]. The Lean theorem is `merge_preserves_moments_and_bounds_energy`
  (`proofs/Fluidity/Merge.lean`); name it as the statement, and never write that it has been checked by Lean:
  whether it was built is recorded only in the proof's own commit.
- [ ] **Step 2: Figures** as design 8 says: `<fig-split>` three panels (parent with eight children at
  `sqrt(3)/4 H`, the grey `H/4` corners and the three spreads as circles; a solid face near one child, refused;
  committed and resized grids side by side with a drifted particle refused on both). `<fig-merge>` two panels
  (spread ellipse inside the allowed circle, merged; outside, refused). Look at every page.
- [ ] **Step 3: Pseudocode.** `<proc-split-particle>`: 1 check grouping; 2 check index; 3 check count rule;
  4 check fitted change; 5 check capacity and ids; 6 parent's neighbourhood on `grid_before_resize`; 7 each
  child's on `grid`; 8 write eight children to the waiting list; 9 mark the parent replaced.
  `<proc-merge-particles>`: 1 grouping; 2 indexes; 3 count rule; 4 agreement; 5 fitted change; 6 each
  particle's neighbourhood on `grid_before_resize`; 7 compute in id order; 8 gap; 9 merged position's
  neighbourhood on `grid`; 10 capacity and ids; 11 write; 12 mark replaced.
  `<proc-adjust-particle-count>`: 1 check grouping; 2 per block, splits in grouped order; 3 per active cell,
  merge chunks; 4 place results by prefix sums; 5 tree-sum the block partials.
- [ ] **Step 4: Run** typst and the gate. **Commit.** Subject: `Describe splitting and merging particles`.

### Task 4: Views, field lists, errors and settings

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/particles/particle_fields.h`, `particle_fields_test.cc`, `particles.h` (declarations
  of design 3.2 up to and including `ParticlesSettings` only), `particles_errors.cc`, `particles_errors_test.cc`
- Modify: the CMake file that lists grid sources (register `native/fluidity/particles/` exactly as
  `native/fluidity/grid/` is registered); `docs/glossary.md`

**Interfaces:**
- Consumes: `Vec3`, `Mat3`, `PerParticleVelocity`, `RunOn` (from the shared `base/` header Task 1 found),
  `CpuMemory`, `grid::CellBlockRange` (from `grid/grid_ids.h`), `contact::PlateSide`.
- Produces: design 3.1 (`kMaxIngredientCount`, `kCarriesFittedChange`, `kHasOwnMotion`,
  `ParticleFields`, `ArrayToChange`, `Field`, `FieldList`, `ToFieldList`, `HasField`, `ParticleFieldsToChange`,
  `ParticleAmounts`), with the `run_on` member of Global Constraints in `ParticleFields` and
  `ParticleFieldsToChange`; from 3.2 `ParticlesError`, `ToMessage`, `ToName`, `ParticlesFailure`, `CountRule`,
  `ExtraFields`, `MaterialParticlesSettings`, `ParticlesSettings` with the added `max_cell_blocks` (Global
  Constraints).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ToFieldList, ThreeFields_OrsTheirBits)        // {kPosition, kVolumeRatio, kMass} -> bits 1 | 16 | 512
  TEST(HasField, ListWithoutField_IsFalse)           // static_assert as well
  TEST(ParticleFields, Default_HasOneIngredientNullArraysAndCpu)  // run_on == RunOn::kCpu; same for
                                                                  // ParticleFieldsToChange
  TEST(ToName, EveryError_LowerSnakeCase)            // all 17 values; kMergeTooSpread -> "merge_too_spread";
                                                     // kNone -> "none"; all distinct
  TEST(ToMessage, EveryError_HasADistinctNonEmptyMessage)
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** `particle_fields.h` is shared (`FLUIDITY_CPU_AND_GPU` where functions exist,
  `ToFieldList` and `HasField` `constexpr`). `particles.h` includes `base/cpu_only.h` first. Messages say what
  the user can do. File headers cite `Read: docs/method-paper/fluidity-method.typ <sec-particle-fields>`.
  Docstrings state the index convention of Global Constraints.
- [ ] **Step 4: Glossary.** For each of `ingredient_fractions`, `size_level` (of a particle), `count rule`,
  `amounts`, `flags`: add a row `| Say | Plain meaning | Never say |` if absent, with the "Never say" entries
  from design section 9 (`constituent`, `composition`, `phase_id`, `inventory (of particles)`, `tag`).
- [ ] **Step 5: Run** the gate; if `HasField` fails the `verb` rule, add it to
  `tools/check_rules_allowed_names.txt` with `# predicate on a field list, design 3.1`. **Commit.** Subject:
  `Add the particle views, field lists and errors`. List every name above; the body records the plan decision
  `ParticleFields::run_on`, `ParticleFieldsToChange::run_on` (changes design 3.1: the views name their processor).

### Task 5: Split formulas

**Kind:** numerics

**Files:**
- Create: `native/fluidity/particles/split_and_merge_math.h` (split part), `split_and_merge_math_test.cc`

**Interfaces:**
- Consumes: `grid::ComputeCompleteWeightedSpread`, `math::Sqrt`.
- Produces: `OneParticle`, `kMergeGapFloor`, `ComputeSplitOffset`, `ComputeSplitChild` (design 3.6), all
  `FLUIDITY_CPU_AND_GPU`, `noexcept`. Test helpers (in the test file): `ComputeKineticEnergy(OneParticle, width)`
  `= (m/2)(|v|^2 + spread(width) * sum of squares of C)` and `ComputeAngularMomentum(OneParticle, width)` `=
  m x × v + m spread(width) (C.zy - C.yz, C.xz - C.zx, C.yx - C.xy)`.

- [ ] **Step 1: Failing tests** (the "split parent" of Global Constraints at `{0.5, 0.5, 0.5}`, `H = 1`; sums
  over the eight children in child order; "rel" relative to the largest term).

  | Test | Expected | Tolerance |
  |---|---|---|
  | `ComputeSplitOffset, UnitCell_IsRootThreeOverFour` | `0.4330127018922193` | 1 ulp |
  | `ComputeSplitChild, EightChildren_KeepAmounts` | each m = 1, `rest_volume` 0.25, `volume_ratio` 0.5; sums equal parent | exact |
  | `ComputeSplitChild, EightChildren_KeepMomentum` | `sum m_k v_k = m v` | 1e-14 rel |
  | `ComputeSplitChild, EightChildren_KeepAngularMomentumAndEnergy` | C = rows {0,-2,0},{2,0,0},{0,0,0} (rotation about Z), then the general C: angular momentum and energy (children at `H/2`, parent at `H`) equal | 1e-13 rel |
  | `ComputeSplitChild, QuarterWidthOffset_WouldLoseEnergy` | general C, offset `H/4`: energy strictly below the parent's | sign |
  | `ComputeSplitChild, ChildIndexBits_PickSides` | child 5 (bits x, z) is at `+offset, -offset, +offset` | exact |

- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** Ported from `git show main:native/src/liquid_resampling_3d.cc` lines 397 (offset)
  and 414-445 (child loop): bit 0 of `child_index` is x (set = high side), bit 1 y, bit 2 z;
  `position_k = position + offset_vector`; `velocity_k = velocity + C * (position_k - position)` (the stored
  offset, as `main`); `mass / 8`, `rest_volume / 8`; C and `volume_ratio` copied. `ComputeSplitOffset(width) =
  math::Sqrt(spread(width) - spread(width / 2))`. Drop the exactness checks (`:379-396`). Header cites
  `<sec-split>` and `<eq:split-offset>`; `// Step N:` comments follow `<proc-split-particle>` only in Task 18.
- [ ] **Step 4: Glossary** row `split_offset` if absent. **Run** the gate. **Commit.** Subject:
  `Add the split formulas`. Names: `OneParticle`, `kMergeGapFloor`, `ComputeSplitOffset`, `ComputeSplitChild`.

### Task 6: Merge formulas

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/particles/split_and_merge_math.h`, `split_and_merge_math_test.cc`

**Interfaces:**
- Produces: `MergedParticle`, `ComputeMergedParticle` (design 3.6) and, **plan decision** (design 3.6 has no
  fraction input, yet `MergeParticles` must keep each ingredient volume):

  ```cpp
  // fractions: [particle * ingredient_count + ingredient] of the same particles, same order. Writes
  // ingredient_count values: sum(f J V0) / sum(J V0) per ingredient, summed in particle order.
  FLUIDITY_CPU_AND_GPU void ComputeMergedFractions(const OneParticle* particles, const double* fractions,
                                                   int particle_count, std::uint32_t ingredient_count,
                                                   double* merged_fractions) noexcept;
  ```

- [ ] **Step 1: Failing tests.**

  | Test | Setup | Expected | Tolerance |
  |---|---|---|---|
  | `ComputeMergedParticle, TwoEqual_MeanAndSum` | `h = 1`; 1 kg at x = -0.25 and +0.25, x velocity -1 and +1, C = 0 | mass 2, centre 0, velocity 0, `C*.xx = 0.25`, others 0, gap 0.6875, energy lost 0.9375 J | 1e-14 |
  | `ComputeMergedParticle, SplitThenMerge_GivesParentBack` | the "split parent" split at `H = 1`, children merged at `h = 0.5` | parent position, velocity, C, amounts; energy lost 0; gap 0 | 1e-13 rel; gap within 1e-14 |
  | `ComputeMergedParticle, KeepsEachIngredientVolume` | 3 particles, 3 ingredients, J = 0.5, 1, 1.25, V0 = 1, 2, 3, fractions rows {0.5,0.25,0.25},{0,1,0},{0.125,0.375,0.5} | `sum f J V0` per ingredient unchanged (merged fractions times merged J V0) | 1e-14 rel |
  | `ComputeMergedParticle, KeepsAngularMomentum` | 5 particles from `SplitMix64`: positions in `[-0.3, 0.3]^3`, v in `[-2, 2]^3`, C entries in `[-1, 1]`, m in `[0.5, 2]`, `h = 1` | orbital + fitted angular momentum (children at `h`, merged at `2h`) unchanged | 1e-13 rel |
  | `ComputeMergedParticle, EnergyNeverGrows` | 1000 cases as above, n = 2 + case % 7, redraw (same stream) until gap > floor | `kinetic_energy_lost >= -1e-13 * sum energy` | as stated |
  | `ComputeMergedParticle, FarFromOrigin_SameAsNearOrigin` | the `KeepsAngularMomentum` group, then shifted by `{1e6, -1e6, 1e6}` | velocity, C, gap, energy lost equal; position equal after removing the shift | 1e-9 abs for position, 1e-13 rel otherwise |

- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** in this order (plan decision; all sums sequential over the input order, which the
  caller gives in ascending id):
  1. `M = sum m_c`; `w_c = m_c / M`.
  2. `dx_c = x_c - x_0`, `dv_c = v_c - v_0`; `center_offset = sum w_c dx_c`; `mean_v_offset = sum w_c dv_c`;
     `center = x_0 + center_offset`; `mean_v = v_0 + mean_v_offset`.
  3. `r_c = dx_c - center_offset`; `S = sum w_c r_c r_c^T / h^2`;
     `gap = (spread(2h) - spread(h)) / h^2 - max_a (S_aa + sum_{b != a} |S_ab|)`.
  4. `cross = sum m_c (dv_c - mean_v_offset) r_c^T`;
     `C* = cross / (M * spread(2h)) + (spread(h) / spread(2h)) * sum w_c C_c`.
  5. `rest_volume* = sum V0_c`; `volume_ratio* = sum (J_c V0_c) / rest_volume*`.
  6. `kinetic_energy_lost = sum energy(c, h) - energy(merged, 2h)`.
  Ported from `main` `liquid_resampling_3d.cc:244-259` (gap row-sum bound, without interval arithmetic),
  `:265-297` (centre relative to the first particle), `:303-354` (fit; the pair-difference form there equals
  step 4). Header cites `<sec-merge>`, `<eq:merge-fit>`, `<eq:merge-gap>`.
- [ ] **Step 4: Glossary** row `gap` (merge gap; never say `covariance_gap_lower_bound`). **Run** the gate.
  **Commit.** Subject: `Add the merge formulas`. Names: `MergedParticle`, `ComputeMergedParticle`,
  `ComputeMergedFractions` (plan decision: fractions are not in `OneParticle`).

### Task 7: Merge proof in Lean

**Kind:** numerics

**Files:**
- Create: `proofs/lakefile.toml`, `proofs/lean-toolchain`, `proofs/lake-manifest.json` (copied unchanged from
  `git show main:proofs/<name>`), `proofs/Fluidity.lean`, `proofs/Fluidity/Merge.lean`
- Modify: `.gitignore` (add `proofs/.lake/`), `AGENTS.md` Commands (one line: how to check the proofs, and
  that they are not part of the gate); the proof's status goes in the commit body and the hand-over

`lake` is not installed on the development Mac. Never write that the proof checks unless `lake build` was run
and exited 0 in this task.

**Interfaces:**
- Produces theorems `merge_preserves_moments_and_bounds_energy` and `merge_preserves_ingredient_volumes`
  (plan decision: the second is `main`'s `merge_preserves_volume_inventories` renamed into the decided words).

- [ ] **Step 1: Trim.** Start from `git show main:proofs/Fluidity/Merge.lean` (354 lines). Keep only what the
  two theorems need (`main` lines 240-354 and the private lemmas they use, `:14-216`). Restate for the single
  quadratic spread (`spread(w) = w^2/4`; drop every cubic-normal branch). Rename to the glossary words:
  constituent → ingredient, inventory → amount, slope/affine → fitted change; theorem docstrings state, in
  plain words, what each guarantees (positive gap implies no energy gain for every admissible input).
  `Fluidity.lean` imports only `Fluidity.Merge`.
- [ ] **Step 2: Build if possible.** `grep -n sorry proofs/Fluidity/Merge.lean` must print nothing. Then try a
  Lean container (Docker via colima, image with `elan`; toolchain from `proofs/lean-toolchain`), mounting
  `proofs/` and running `lake exe cache get && lake build`. Exactly one of two outcomes is recorded:
  - the container build exits 0: status **"built in a Lean container"**, with the image reference and the
    `lake build` output's last lines in the commit body;
  - no container could be set up or the build did not finish: status **"written, not built here"**, with the
    reason in one line. A failing build is not committed: fix it, or stop and tell the coordinator.
- [ ] **Step 3: Run** the gate (the rules check scans `.lean`). **Commit.** Subject:
  `Write the proof that a merge with a positive gap never gains energy`. The body states the status of Step 2
  word for word and that the proofs are not part of the gate.

### Task 8: Refill formula and the comparison with the proof of concept

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/particles/split_and_merge_math.h`, `split_and_merge_math_test.cc`
- Create: `native/fluidity/particles/testdata/proof_of_concept_split_merge_refill.txt`
- Temporary only: `<temp-dir>/poc-main/` (a detached worktree of `main`), `<temp-dir>/poc-main/generate_values.cc`

**Interfaces:**
- Produces: `RefillCell`, `ComputeRefillParticle` (design 3.6).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ComputeRefillParticle, NineParticles_KeepCellAmounts)   // centre {0,0,0}, h = 1, V = 1, density 1000,
      // J = 0.5, v = {1,2,3}, C rows {1,0,0},{0,2,0},{0,0,3}: masses sum to 1000, rest volumes to 2, momentum to
      // 1000 v (offsets cancel). 1e-14 rel.
  TEST(ComputeRefillParticle, OddCount_FirstAtCentre)          // n = 3: particle 0 at the centre, 1 and 2 in
      // eighths 0 and 7 (offsets -h/4 and +h/4 on each axis). exact
  TEST(ComputeRefillParticle, EvenCount_PairsOppositeEighths)  // n = 4: eighths 0, 7, 1, 6. exact
  TEST(SplitAndMergeMath, AgreesWithProofOfConcept)            // reads the testdata file; 1e-12 rel
  ```
- [ ] **Step 2: Implement `ComputeRefillParticle`** (port of `main` `liquid_band_handoff_3d.cc:240-254`
  `SeedOffset` and `:577-600`; plan decision on rounding order, same as `main`): `particle_current_volume =
  current_volume / n`; `mass = density * particle_current_volume`; `rest_volume = particle_current_volume /
  volume_ratio`; for odd n particle 0 has offset 0 and later indexes shift down by one; pair p gives eighth p
  then `7 - p`; eighth bit 0 = high x, bit 1 = high y, bit 2 = high z, offset `+-h/4`; `velocity = v + C *
  offset`; `fitted_velocity_change = C`; `volume_ratio` from the cell. Cite `<sec-add-remove>`.
- [ ] **Step 3: Produce the old numbers.** `git -C /Users/tychota/Documents/Code/openSource/FluidiTy worktree
  add --detach <temp-dir>/poc-main main`. In it, write `generate_values.cc` (temporary, never committed)
  linked to `main`'s library with preset `macos-release-cpu`. It builds 20 split parents, 20 merge groups
  (2 to 8 particles, gap well above zero, masses powers of two so `main`'s exactness checks pass) and 9 refill
  cells from `SplitMix64` seed 20261114, in Z-up; rotates each input into Y-up with `R^T`, calls
  `SplitUniformLiquidParticle3D`, `MergeUniformLiquidParticles3D` (`kQuadratic`, an admissibility callback that
  returns OK) and `ReseedLiquidCoreCell3D`, rotates outputs back with `R`, and prints one line per value
  (`%.17g`). The testdata file starts with comment lines naming the `main` commit hash and these steps. Remove
  the temporary worktree afterwards (`git worktree remove`).
- [ ] **Step 4: The comparison test** reads the file (path from a compile definition set in the particles
  CMake), recomputes with Tasks 5, 6 and this task's functions, matches split children by position, and
  checks `1e-12` relative.
- [ ] **Step 5: Glossary** row `refill` if absent. **Run** the gate. **Commit.** Subject:
  `Add the refill formula and compare split, merge and refill with the proof of concept`.

### Task 9: Complete neighbourhood and the test fixtures

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the fixtures are the test library `fluidity_particles_test_support` made with
`fluidity_add_test_library`; tests name it and `fluidity_base_test_support` in `LINKS`.

**Files:**
- Modify: `native/fluidity/particles/split_and_merge_math.h`, `native/fluidity/particles/CMakeLists.txt`
  (`fluidity_add_test_library(fluidity_particles_test_support SOURCES particles_test_support.cc LINKS
  fluidity_base_test_support)`; every particles test from here on is `fluidity_add_test(... LINKS
  fluidity_particles_test_support)`, and `split_and_merge_math_test` gains that `LINKS`)
- Create: `native/fluidity/particles/particles_test_support.h`, `particles_test_support.cc` (test-only, never
  linked by `fluidity`), `find_neighbourhood_test.cc`

**Interfaces:**
- Consumes: `grid::GridTables`, `FindCellAt`, `ToCellBlockIndex`, `ToSizeLevel(tables, block)`,
  `FindTouchingCells`, `FindNearbyFaces`, `ComputeNearbyFaceWeights`, `NearbyFaceMask`, `grid::BuildGrid(inputs,
  settings)`, `ToCellId`, `ToFacePieceId`, `FindFacePlace`.
- Produces: `NeighbourhoodInputs`, `Neighbourhood`, `FindNeighbourhood` (design 3.5; **plan decision**: all
  three are declared in the shared `split_and_merge_math.h`, since a shared function takes them). Every test
  fixture of Global Constraints except `RunTestStep` and the test-only setters (later tasks add those), with
  `BuildNeighbourhoodBuffers` returning vectors sized `block_count * 240` and `block_count * 64`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(FindNeighbourhood, CentreOfBigCells_IsComplete)          // P, level 0, BigCells grid, null inputs
  TEST(FindNeighbourhood, WrongSizeLevel_IsIncomplete)          // P, level 1, BigCells grid
  TEST(FindNeighbourhood, TouchingSmallerCells_IsIncomplete)    // P, level 0, CornerSplit grid
  TEST(FindNeighbourhood, FaceInsideSolid_IsIncomplete)         // ComputeSplitChild(P parent, offset of H = 1, 7)
      // on CentreSplit at level 1 with the "solid face" outside_solid 0: incomplete; child 0 stays complete
  TEST(FindNeighbourhood, FaceOnOtherPlateSide_IsIncomplete)    // same face, face_side {7, kAlongNormal},
      // particle side {7, kAgainstNormal}
  TEST(FindNeighbourhood, TouchingGridOnlyCell_IsIncomplete)    // P level 0, BigCells; cell (4,3,3) grid-only
  TEST(FindNeighbourhood, LevelBeyondSmallest_IsIncomplete)     // size_level 3 on a 3-level frame
  ```
- [ ] **Step 2: Implement** (shared, allocates nothing): 1 `cell = FindCellAt(tables, position)`; `kNoIndex`
  or its level `!= size_level` → incomplete; 2 the cell and every `FindTouchingCells` cell must have
  `is_grid_only_region == 0` (plan decision: the containing cell counts too); 3 for each axis
  `FindNearbyFaces(tables, position, axis, ToCellBlockIndex(cell), &error)`, error → incomplete; mask bit i kept
  iff (`outside_solid` null or nonzero at the face) and (`face_side` null or equal to `side`);
  `ComputeNearbyFaceWeights(...).is_complete` false → incomplete. Cite `<sec-split>`.
- [ ] **Step 3: Glossary** row `complete neighbourhood` (never say `admissibility`, `certify`). **Run** the
  gate. **Commit.** Subject: `Decide when a particle's neighbourhood is complete`.

### Task 10: The particle arrays, their limits and their memory estimate entries

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** `particles_storage.h` also holds `order_buffers`, `sum_buffers` and the lowest-failing-id buffer,
built and rebuilt with the arrays and counted in the memory estimate (Global Constraints); the GPU test compares
with the `kGpu` estimate.

**Files:**
- Modify: `native/fluidity/particles/particles.h` (rest of design 3.2 except `CheckParticles`,
  `kFractionSumTolerance`, `ComputeAmounts*`)
- Create: `particles_storage.h` (private, CPU only), `material_arrays_view.h` (private, shared),
  `particles_cpu.cc`, `particle_memory_estimate.h`, `particle_memory_estimate.cc`, `particles_test.cc`,
  `particles_no_gpu_test.cc`, `particles_gpu_test.cc`

**Interfaces:**
- Consumes: `BuildBuffer`, `CopyBuffer` (slice), `CheckCanRunOn`, `MemoryEstimateEntry`, `CountedPer`, `KeptFor`.
- Produces: `class Particles` (all accessors of design 3.2), `BuildParticles`, `ApplyParticleLimits` (the
  retyped three-argument declaration of Global Constraints), `BuildFieldsToRead(const Particles&,
  material_index, cell_blocks)` (sets `run_on` from the particles), `BuildParticlesMemoryEstimate` (the memory
  rows of Global Constraints, in that order); private, tests only, in `particles_storage.h`:
  `std::uint64_t ComputeAllocatedBytes(const Particles&)` and the two test-only setters of Global Constraints.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(BuildParticles, LiquidOnly_AllocatesNoExtraArrays)  // extra pointers null in BuildFieldsToRead;
      // ComputeAllocatedBytes == ComputeMemoryBytes(BuildParticlesMemoryEstimate(settings), {})
  TEST(BuildParticlesMemoryEstimate, EveryMaterialKind_MatchesAllocation)  // the six fixture materials together,
      // max_particle_count 1000 each, max_added_per_try 100, max_cell_blocks 64: rows sum == allocated bytes
  TEST(BuildParticles, ZeroMaterialIdOrRepeatedId_IsBadInput)
  TEST(BuildParticles, TooManyIngredients_IsBadInput)       // 9
  TEST(BuildParticles, ZeroMaxCellBlocks_IsBadInput)
  TEST(BuildParticles, Fresh_HasNoParticlesAndVersionsZero) // counts 0, next_particle_id 1, time 0
  TEST(ApplyParticleLimits, RaisedLimit_KeepsEveryValue)    // here: counts, next id, versions kept, new limits
                                                            // reported (Task 13 extends it to every array)
  TEST(ApplyParticleLimits, RaisedBlockLimit_ReallocatesBlockTables)  // max_cell_blocks 64 -> 128 with the
      // particle limits unchanged: material(i) limits kept; ComputeAllocatedBytes equals the rows at 128 blocks;
      // committed counts, versions and next id kept (Task 13 extends it: group_begin, cell_begin equal byte for
      // byte, and a grid of 88 blocks can then be grouped)
  TEST(ApplyParticleLimits, BelowCurrentCount_IsBadInput)   // particles unchanged
  TEST(ApplyParticleLimits, LowerBlockLimit_IsBadInput)     // 64 -> 32: kBadInput; particles unchanged
  TEST(ApplyParticleLimits, InsideATry_IsTryInProgress)
  // particles_no_gpu_test.cc, label no_gpu
  TEST(BuildParticles, OnGpuWithoutAGpu_IsRunOnCallFailed)
  // particles_gpu_test.cc, label gpu
  TEST(BuildParticles, OnGpu_AllocatesTheSameRowsAsCpu)  // amended: the same entries as the CPU estimate except
      // the buffers entries, which follow ComputeOrderBuffersSize(kGpu) and ComputeSumBuffersSize(kGpu);
      // ComputeAllocatedBytes == ComputeMemoryBytes(BuildParticlesMemoryEstimate(gpu settings), {}); its digest
      // (WriteResultDigest) is that byte count
  ```
  `ComputeAllocatedBytes` adds `order_buffers.byte_count()` and `sum_buffers.byte_count()`.
- [ ] **Step 2: Implement.** `BuildParticles` calls `CheckCanRunOn(settings.run_on)` first, then allocates
  every buffer from `BuildParticlesMemoryEstimate`' table walk (one function walks the material's arrays for both,
  so they cannot drift); on any failure frees everything and returns `kOutOfMemory` or `kRunOnCallFailed`.
  `cpu_memory` is passed to every CPU buffer. `ApplyParticleLimits` builds every new buffer first (particle
  arrays at the new particle limits, `group_begin`, `cell_begin`, written marks and loaded block origin at the
  new `max_cell_blocks`), copies the committed part with slice `CopyBuffer` (`group_begin`: committed block
  count + 1 entries; `cell_begin`: committed block count * 64), then swaps; any failure leaves the particles
  unchanged. `Particles` is move-only. The working buffers are built with `BuildOrderBuffers(settings.run_on, ...)` and
  `BuildSumBuffers(settings.run_on, ...)` at the counts of Global Constraints; `ApplyParticleLimits` builds the new
  ones before the swap.
- [ ] **Step 3: Run** the tests (they fail before Step 2), then the gate. **Commit.** Subject:
  `Add the particle arrays and their memory estimate entries`. Body:
  `GPU: compiled, not run`; names listed; the 405 B note; the coordinator decision that `ApplyParticleLimits`
  also takes `max_cell_blocks` (changes design 3.2).

### Task 11: The try and adding particles

**Amended 2026-11-14 (A17, batched-run buffers):** `AddParticles` reports the copies it makes to particles on the
GPU, so the step counts them (step plan Appendix A row M1).

**Amended 2026-11-14 (Foundation as built):** the `gpu` tests write result digests.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/particles/particles_try.h`, `particles_try_cpu.cc`, `particles_try_test.cc`,
  `add_particles.h`, `add_particles_cpu.cc`, `add_particles_test.cc`, `add_particles_gpu_test.cc`

**Interfaces:**
- Consumes: `particles_storage.h`, `material_arrays_view.h`; `WriteBuffer` (slice).
- Produces: `ParticlesTry` (`particle_count`, `waiting_count`), `BuildTry`, `BuildFieldsToRead(const
  ParticlesTry&, ...)`, `CommitTry`, `RejectTry` (first form: no array changes yet; Task 13 adds the written
  marks), `AddReason`, `NewParticles`, `AddParticlesResult`, `AddParticles`. **Plan decision:** `AddParticles`
  has no `_gpu.cu`; for both processors it validates the CPU spans, sums amounts on the CPU and writes the waiting
  list with slice `WriteBuffer`.
- **Plan decision (A17, batched-run buffers; `plan-decisions.md` "Copies counted"):** `AddParticlesResult` gains
  `std::uint32_t copy_count = 0;` (one per `WriteBuffer` slice call into a buffer outside CPU memory) and
  `std::uint64_t bytes_copied = 0;` (their bytes). Both are 0 when the particles are in CPU memory (a CPU run, or a
  batched run whose particles are in locked CPU memory).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(BuildTry, WhileAnotherTryLives_IsTryInProgress)
  TEST(RejectTry, ThenBuildTry_Succeeds)
  TEST(CommitTry, NothingChanged_BumpsStateVersionAndSetsTime)   // time 0.25, state_version 1
  TEST(ParticlesTry, DestroyedWithoutCommit_ActsAsReject)        // a new BuildTry succeeds; nothing changed
  TEST(CommitTry, TryOfOtherParticles_IsBadInput)
  TEST(AddParticles, NonFiniteOrNonPositiveValue_IsBadInput)     // NaN position; mass 0; rest_volume -1
  TEST(AddParticles, FractionsNotSummingToOne_IsBadInput)        // mixture of 3, row {0.5, 0.5, 0.1}
  TEST(AddParticles, CarriedArrayMissingOrExtraGiven_IsWrongFields)
  TEST(AddParticles, MoreThanWaitingRoom_IsTooManyParticles)     // max_added_per_try 4, add 5; try unchanged
  TEST(AddParticles, IdsWouldOverflow_IsIdsExhausted)            // ApplyNextParticleIdForTest(UINT64_MAX - 2),
      // add 3: kIdsExhausted; try and next_particle_id unchanged
  TEST(AddParticles, OnCpu_NoCopyCounted)                        // A17: copy_count 0, bytes_copied 0
  // add_particles_gpu_test.cc, label gpu
  TEST(AddParticles, OnGpu_SameIdsAndAmountsAsCpu)
  TEST(AddParticles, OnGpu_CopiesCountedPerArray)                // A17: 5 liquid particles: copy_count == the
      // number of waiting-list arrays written (the docstring lists them, defaults included), bytes_copied == 5 *
      // the sum of their sizeof
  ```
  The two `gpu` tests write the ids and the waiting-list bytes with `WriteResultDigest`.
- [ ] **Step 2: Implement.** Checks in this order: material index (`kBadInput`), spans (`kWrongFields` when a
  carried array is missing or an uncarried one given; `kBadInput` when lengths differ), values (finite; `mass`,
  `rest_volume`, `volume_ratio` positive; fraction rows sum to 1 within `kFractionSumTolerance` summed in order),
  room (`kTooManyParticles`, `magnitude` = excess), ids (`kIdsExhausted`). New particles: flags
  `kCarriesFittedChange` when `fitted_velocity_change` is given, `kHasOwnMotion` off; `volume_ratio` 1 when
  empty. `AddReason` only labels the result. Cite `<sec-add-remove>`, `<sec-particle-try>`.
- [ ] **Step 3: Glossary** row `waiting` (added, not yet grouped in). Add `CommitTry`, `RejectTry` to
  `tools/check_rules_allowed_names.txt` if the `verb` rule fails (`# DECIDED, design 11.6`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the particle try and adding particles`.

### Task 12: Grouping particles by cell block, on the CPU

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the two `BuildStableOrder` passes take the particles' `order_buffers`, so
grouping allocates no temporary order arrays.

**Amended 2026-11-14 (A31):** `BuildStableOrder` has one form, which takes its `OrderBuffers`.

**Files:**
- Create: `native/fluidity/particles/group_particles_by_block.h`, `group_particles_by_block_cpu.cc`,
  `group_particles_by_block_test.cc`
- Read: `docs/method-paper/fluidity-method.typ` `<proc-group-particles>` (step numbers for the code)

**Interfaces:**
- Consumes: `BuildStableOrder` (`keys, count, previous_order, order, OrderBuffers&`, with
  `order_buffers` of `particles_storage.h`), `grid::FindCellAt(tables, position)`,
  `grid::StaggeredGrid::tables(kCpu)`, `version()`, `counts()`.
- Produces: `GroupingSettings`, `GroupParticlesByBlock`, `CheckGrouping` (design 3.4), CPU path.

- [ ] **Step 1: Failing tests** (fixture "six on three blocks").

  ```cpp
  TEST(GroupParticlesByBlock, InsideCell_IdsAscend)   // order, group_begin and cell_begin as in Global
      // Constraints; position_when_grouped == position
  TEST(AddParticles, GivesConsecutiveIds_AndNoOwnMotion)  // 5 particles: first_id == next id before, ids
      // n..n+4 after grouping, kHasOwnMotion off, added amounts equal sums of the inputs in order (exact)
  TEST(GroupParticlesByBlock, ParticleOutsideGrid_ReportsLowestId)  // the six with ids 4 and 6 added at x = 9
      // and x = -1 instead: kOutsideGrid, particle_id 4; try's grouping unchanged
  TEST(GroupParticlesByBlock, TwiceInOneTry_KeepsCommittedValues)   // commit six; new try: add 2, group, add 1,
      // group again; arrays equal a one-grouping run of the same nine; RejectTry; committed arrays byte-equal
  TEST(GroupParticlesByBlock, MoreBlocksThanLimit_IsTooManyParticles)  // max_cell_blocks 2
  TEST(CheckGrouping, AfterAdd_IsStale)               // and after a grid of another version: kGroupingStale
  ```
- [ ] **Step 2: Implement** following `<proc-group-particles>` with `// Step N:` comments and the Global
  Constraints rules (keys, removal marks, in-place lanes, waiting list emptied, try counts and tables set,
  grouping marks every array written for every block). `CheckGrouping`: `kGroupingStale` when anything waits,
  a removal mark is set, or `grid.version()` differs; then `kMovedTooFarSinceGrouped` when any axis of
  `position - position_when_grouped` is strictly above `max_move_since_grouped` (lowest id; `magnitude` = the
  largest excess). Ported idea from `main` `liquid_step_workspace_3d.cc:210-267`; not ported: shifted bins.
- [ ] **Step 3: Glossary** rows `group`, `group_begin`, `cell_begin` (never say `bin`, `sort bin`). Add
  `GroupParticlesByBlock` to the allowed names if the `verb` rule fails (`# DECIDED name, design 3.4`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Group particles by cell block and cell place`.

### Task 13: Changing arrays and committing them

**Kind:** integration

**Files:**
- Modify: `native/fluidity/particles/particles_try.h`, `particles_try_cpu.cc`, `particles_try_test.cc`,
  `particles_test.cc` (extend `ApplyParticleLimits, RaisedLimit_KeepsEveryValue` to compare every array of the
  "cloud", and `ApplyParticleLimits, RaisedBlockLimit_ReallocatesBlockTables` to compare `group_begin` and
  `cell_begin` byte for byte and then group the "cloud" on an 88-block grid: the fixture frame with level-0
  cells x 0..79, y, z 0..7 (80 blocks) plus the centre split (8 blocks), `GridSettings{.max_cell_blocks = 128}`;
  before the raise the same grouping is `kTooManyParticles`)
- Modify: `particles_test_support.h/.cc` (add `RunTestStep`)

- Read: `docs/method-paper/fluidity-method.typ` `<proc-commit-try>`

**Interfaces:**
- Produces: `BuildFieldsToChange` (design 3.3; sets `run_on` from the particles); `CommitTry` with `kArrayPartlyWritten` and the flip; the test
  helper `RunTestStep` of Global Constraints.

- [ ] **Step 1: Failing tests** (fixture "cloud" unless stated).

  ```cpp
  TEST(CommitTry, ChangedArraysOnly_NoCopy)   // change velocity_for_next_step only: after commit its read pointer
      // is the former write pointer, every other array's pointer unchanged; state_version + 1
  TEST(CommitTry, ArrayWrittenForSomeBlocks_IsRefused)  // BuildFieldsToChange for blocks [0, 3) only:
      // kArrayPartlyWritten; committed values and state_version unchanged
  TEST(BuildFieldsToChange, SecondRangeOfSameArray_ReadsCommittedValues)  // write blocks [0, 3) to 7.0, then ask
      // [2, 6): read shows 7.0 for blocks 2 and committed values for 3..5; after writing, commit succeeds only
      // once [6, end) is written too
  TEST(BuildFieldsToChange, MassOrUncarriedArray_IsWrongFields)  // kMass; kSpringyDeformation on a liquid
  TEST(BuildFieldsToChange, ThisStepVelocity_ReadIsNullUntilWritten)
  TEST(CheckGrouping, MovedTooFar_IsRefused)  // max_move 0.1, one particle moved 0.2 along y:
      // kMovedTooFarSinceGrouped, its id, magnitude 0.1 within 1e-15
  TEST(GroupParticlesByBlock, AnyInputOrder_SameResult)  // three committed states, each assigning the same six
      // start positions to ids 1..6 in a different permutation (identity, reversed, {3,1,6,2,5,4}); in each a try
      // moves particle k to the same final position F_k (BuildThreeBlockGrid; F = the "six on three blocks"
      // positions; start positions = F shifted by +0.125 on y; every other value depends on the id only:
      // v = {id, 0, 0}, m = rest_volume = 0.001 * id), regroups and commits: every array identical in bytes
  ```
- [ ] **Step 2: Implement** the per-block written marks of Global Constraints; `CommitTry` follows
  `<proc-commit-try>` with `// Step N:` comments; `kArrayPartlyWritten` names the material and, in
  `magnitude`, the first unwritten block. `RejectTry` clears marks, waiting list, removal marks.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Change particle arrays in a try and commit by swapping`.

### Task 14: Snapshots

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** `ExpectSnapshotsClose` uses `CheckWithinTolerance` from `fluidity_base_test_support`.

**Files:**
- Create: `native/fluidity/particles/particles_snapshot.h`, `particles_snapshot_cpu.cc`,
  `particles_snapshot_test.cc`
- Modify: `particles_test_support.h/.cc` (add `ExpectSameBytes(const ParticlesSnapshot&, const ParticlesSnapshot&)`
  comparing every member with `std::memcmp`, and `ExpectSnapshotsClose(cpu, gpu)` with design 6.6 tolerances
  through `CheckWithinTolerance` of `base/test_support.h`, discrete members with `==`)

**Interfaces:**
- Consumes: `ReadBuffer` (slice), `grid::FindCellAt`.
- Produces: `ParticlesSnapshot`, `SnapshotFields`, `kOutputSnapshotFields`, `ReadSnapshot`,
  `BuildParticlesFromSnapshot` (design 3.7).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(BuildParticlesFromSnapshot, FullSnapshotOfEveryKind_RoundTripsBytes)  // six fixture materials, 50 each,
      // two RunTestStep calls; ReadSnapshot(all) -> build -> ReadSnapshot(all): ExpectSameBytes
  TEST(BuildParticlesFromSnapshot, UngroupedMove_RestoresOrder)  // BigCells grid, 10 liquid particles,
      // max_move 0.3; a try moves id 3 from {3.9,3.5,3.5} to {4.1,3.5,3.5} without regrouping, commit; full
      // snapshot; rebuild; same order, group_begin, cell_begin; RunTestStep once on both: ExpectSameBytes
  TEST(BuildParticlesFromSnapshot, OutputSnapshot_IsRefused)     // kOutputSnapshotFields: kBadInput
  TEST(BuildParticlesFromSnapshot, OrderNotAGrouping_IsGroupingStale)  // two particles swapped in the snapshot
  TEST(BuildParticlesFromSnapshot, MaterialsDiffer_IsBadInputOrWrongFields)
  TEST(ReadSnapshot, OutputFields_CarryConstantArrays)  // mixture of 3 and snow: id, rest_volume, size_level,
      // fractions, position, velocity_for_next_step, volume_ratio filled; every other vector empty
  TEST(ReadSnapshot, ThisStepVelocity_IsRefused)        // kWrongFields
  TEST(ReadSnapshot, InsideATry_IsTryInProgress)
  ```
- [ ] **Step 2: Implement** per design 5.9. `BuildParticlesFromSnapshot` checks in order: `is_full`
  (`kBadInput`), materials match settings (`kBadInput` on id, count or ingredient count; `kWrongFields` on a
  vector size), every `position_when_grouped` in an active cell (`kOutsideGrid`), stored order is (cell place,
  id) (`kGroupingStale`); then builds and writes every array, rebuilds `group_begin` and `cell_begin`, sets
  `time`, `state_version`, `next_particle_id` from the snapshot, `grid_version` from the grid and
  `grouping_version = 1` (plan decision: not stored). No array is reordered. Cite `<sec-particle-copies>`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Read and restore particle snapshots exactly`.

### Task 15: Grouping on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two halves (`_gpu.h` returning `RunOnCallStatus`), launches through `base/gpu_launch.cuh`,
kernel bodies in `group_particles_by_block_steps.h` with a `cpu` test over every thread block and thread, the
order and sums taking the particles' `order_buffers` and `sum_buffers`, result digests.

**Amended 2026-11-14 (A30, A31):** one form per Foundation call (with buffers); same processor for buffers and working buffers is a `Requires:` precondition.

**Files:**
- Create: `native/fluidity/particles/group_particles_by_block_gpu.cu`, `group_particles_by_block_gpu.h`
  (private), `group_particles_by_block_steps.h` (shared), `group_particles_by_block_steps_test.cc`,
  `group_particles_by_block_gpu_test.cc`
- Modify: `group_particles_by_block_cpu.cc` (dispatch on `run_on`; the CPU loop calls the same per-particle
  functions), `CMakeLists.txt` (`fluidity_add_gpu_sources(group_particles_by_block_gpu.cu)`, the header in
  `fluidity_add_headers`, the `cpu` steps test and the `gpu` test with `LINKS fluidity_particles_test_support`)

**Interfaces:**
- Consumes: `material_arrays_view.h`, `grid::GridTables` from `grid.tables(RunOn::kGpu)`, `BuildStableOrder`
  on GPU buffers with `order_buffers`, `ComputeLargestMagnitude(values, count, sum_buffers)`;
  `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork` (`base/gpu_launch.cuh`), `ToBufferError`.
- Produces: no new public name. Private, in `group_particles_by_block_steps.h`, `FLUIDITY_CPU_AND_GPU`: the
  per-particle key and outside flag (`ComputeGroupingKey`), the per-item gather and lane copy
  (`CopyGatheredValue`, `CopyLaneValue`), the per-block and per-place starts by binary search of the sorted keys
  (`ComputeGroupBegin`, `ComputeCellBegin`), the per-particle excess of `CheckGrouping` (`ComputeMoveExcess`).
  Private, in `group_particles_by_block_gpu.h`: one function per GPU phase, each returning `RunOnCallStatus`.

- Read: `group_particles_by_block_test.cc` (the CPU cases this task repeats on the GPU)

- [ ] **Step 1: Failing tests.** `group_particles_by_block_steps_test.cc`, label `cpu` (the main evidence until
  the hardware run): runs every function of `group_particles_by_block_steps.h` for every thread block and thread
  of `ComputeLaunchSize(count)`, in launch order and in reverse, phase by phase as the GPU functions run them, on
  the "six on three blocks" and "cloud" fixtures, and compares keys, outside flags, gathered arrays, `group_begin`,
  `cell_begin` and excesses with the CPU loop of Task 12 exactly (the same functions on one processor). Label
  `gpu`: `GroupParticlesByBlock, OnGpu_SameAsCpu` ("six on three blocks" and "cloud", `ExpectSnapshotsClose` with
  exact ids and tables), `GroupParticlesByBlock, OnGpuTwice_GivesIdenticalBytes`,
  `GroupParticlesByBlock, OnGpuTwiceInOneTry_KeepsCommittedValues`,
  `GroupParticlesByBlock, OnGpuParticleOutsideGrid_ReportsLowestId`, `CheckGrouping, OnGpuMovedTooFar_SameAsCpu`.
  Each `gpu` test passes its snapshot's ids, positions, `group_begin` and `cell_begin` to `WriteResultDigest`.
- [ ] **Step 2: Implement** (design 6.3): one thread per particle computes its key (and the outside flag,
  lowest id by integer `atomicMin` into the storage's lowest-failing-id buffer); two `BuildStableOrder` passes
  with `order_buffers`; one gather GPU function per array (or the lane pair for an in-place regroup);
  `group_begin` and `cell_begin` by binary search of the sorted keys, one thread per block and per place;
  `CheckGrouping` excess per particle, lowest failing id by integer minimum, largest excess by
  `ComputeLargestMagnitude` with `sum_buffers`. Each GPU function is a grid-stride loop (`ComputeGridStride`)
  around one steps function, started with `ApplyGpuFunction`. The CPU half maps a `RunOnCallStatus` with
  `ToBufferError`, then into `ParticlesFailure` through the module's one mapping function, and reads the lowest
  failing id with `ReadBuffer`. Nothing in this task allocates: every array is in `particles_storage.h`.
- [ ] **Step 3: Run** the gate (CUDA compiles; `gpu` tests compiled, not run). **Commit.** Subject:
  `Group particles on the GPU`. Body: `GPU: compiled, not run`.

### Task 16: Removing particles

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** `remove_particles_gpu.h` returning `RunOnCallStatus`, kernel bodies in
`remove_particles_steps.h` with a `cpu` test over every thread block and thread, result digests.

**Files:**
- Create: `native/fluidity/particles/remove_particles.h`, `remove_particles_cpu.cc`, `remove_particles_gpu.cu`,
  `remove_particles_gpu.h` (private), `remove_particles_steps.h` (shared), `remove_particles_steps_test.cc`,
  `remove_particles_test.cc`, `remove_particles_gpu_test.cc`
- Modify: `CMakeLists.txt` (headers, `remove_particles_cpu.cc`, `fluidity_add_gpu_sources(remove_particles_gpu.cu)`,
  the tests with `LINKS fluidity_particles_test_support`)

**Interfaces:**
- Produces: `RemoveReason`, `RemoveParticlesResult`, `RemoveParticles` (design 3.5). Private, shared:
  `CheckRemovalMark` (per listed index: already marked or not) and `AddRemovedAmounts` (one particle's amounts
  added to a running total, in index order), used by the CPU loop and by the GPU functions.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(RemoveParticles, MarkerNotNeeded_ReturnsAmounts)  // gas markers, 10 grouped; remove indexes {1, 4, 7}
      // with kMarkerNotNeeded: amounts == sequential sums of those three (exact); still readable; after the next
      // GroupParticlesByBlock the count is 7 and the three ids are gone
  TEST(RemoveParticles, NotAscendingOrOutOfRangeOrRemovedTwice_IsBadInput)  // try unchanged in each case
  TEST(GroupParticlesByBlock, RemovedParticleOutsideGrid_IsDropped)  // "six on three blocks": a try moves id 1
      // to x = 9 (BuildFieldsToChange), removes it with kLeftDomain, groups: success, five particles
  TEST(CheckGrouping, AfterRemove_IsStale)
  TEST(RejectTry, AfterRemove_KeepsEveryParticle)
  // remove_particles_steps_test.cc, label cpu: CheckRemovalMark over every thread block and thread of
  // ComputeLaunchSize(listed count), launch order and reverse: the first already-marked index equals the CPU
  // loop's; AddRemovedAmounts over the listed particles equals the CPU loop's sums bit for bit (same function,
  // same processor). The main evidence until the hardware run.
  // label gpu
  TEST(RemoveParticles, OnGpu_SameAmountsAndMarksAsCpu)  // marks exact; amounts within design 6.6; marks and
      // amounts through WriteResultDigest
  ```
- [ ] **Step 2: Implement.** Indexes are checked on the CPU against the try's count, and marks on the CPU
  copy (CPU run) or by a GPU function that reports the first index already marked with an integer `atomicMin`
  into the lowest-failing-id buffer (GPU run). Amounts: on the GPU one thread sums the listed particles in index
  order with `AddRemovedAmounts`. Launches through `ApplyGpuFunction`; the status maps through `ToBufferError`.
  The reason is stored in the mark (Global Constraints). Cite `<sec-add-remove>`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Remove particles with a reason`. Body:
  `GPU: compiled, not run`.

### Task 17: Checking values and summing amounts

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** `particles_gpu.h` returning `RunOnCallStatus`; per-block and per-particle bodies in
`particles_steps.h` with a `cpu` test over every thread block and thread; the one-call partials through `GpuArray`;
`ComputeAmounts` promises the same bits on CPU and GPU unless NaN (reason in Global Constraints "Tolerances"), so
its CPU-against-GPU test compares with `CheckSameBitsOrBothNaN`; result digests.

**Files:**
- Modify: `native/fluidity/particles/particles.h`, `particles_cpu.cc`, `particles_try.h`, `particles_try_cpu.cc`,
  `CMakeLists.txt` (`fluidity_add_gpu_sources(particles_gpu.cu)`, the new header and tests)
- Create: `particles_gpu.cu`, `particles_gpu.h` (private), `particles_steps.h` (shared), `particles_steps_test.cc`,
  `particle_amounts_test.cc`, `particle_amounts_gpu_test.cc`

**Interfaces:**
- Consumes: `math::TreeSum`, `math::IsFinite`; `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`,
  `GpuArray` (`base/gpu_launch.cuh`), `ToBufferError`.
- Produces (private, `particles_steps.h`, `FLUIDITY_CPU_AND_GPU`): `ComputeBlockAmounts` (one cell block's partial,
  sequential in grouped order) and `CheckOneParticle` (one particle's value check, returning its error), which the
  CPU loops call too.
- Produces: `CheckParticles` (both forms), `kFractionSumTolerance`, `ComputeAmounts`, `ComputeAmountsPerBlock`.
  `ComputeAmounts` and `ComputeAmountsPerBlock` have the retyped `std::expected` declarations of Global
  Constraints (coordinator decision) and choose the processor from `fields.run_on`; they never ask the CUDA
  runtime where a pointer lives. `fields.run_on == kGpu` in a build without GPU code is `kRunOnCallFailed`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(CheckParticles, OnTry_SeesNewestValues)       // "cloud": NaN written into volume_ratio of one particle
      // through BuildFieldsToChange: kBadInput with its id on the try; CheckParticles(committed) passes
  TEST(CheckParticles, IdsNotAscendingInsideAPlace_IsBadInput)   // "six on three blocks", ApplyCommittedIdForTest
      // swaps ids 2 and 4 (same place)
  TEST(CheckParticles, FractionsOffByMoreThanTolerance_IsBadInput)  // mixture of 3, a row 2e-12 off, written
      // through the committed buffer by a snapshot round trip with the edited row
  TEST(ComputeAmounts, AnyRangeCut_SameBytes)        // "cloud": ComputeAmountsPerBlock over
      // 1-block ranges, then math::TreeSum per component, equals *ComputeAmounts over all_blocks() in bytes;
      // both has_value()
  TEST(ComputeAmountsPerBlock, WrongSpanSize_IsBadInput)  // error kBadInput; span unchanged
  // particle_amounts_test.cc, compiled only when FLUIDITY_HAS_GPU is undefined
  TEST(ComputeAmounts, GpuViewInBuildWithoutGpuCode_IsRunOnCallFailed)  // a "cloud" view with run_on set to kGpu
  // label gpu
  TEST(ComputeAmounts, OnGpu_SameBytesAsCpu)          // same tree order; the GPU view has run_on == kGpu;
      // CheckSameBitsOrBothNaN per component (the header's promise); the amounts through WriteResultDigest
  TEST(CheckParticles, OnGpuTwoBadParticles_ReportsLowestId)
  // particles_steps_test.cc, label cpu (the main evidence until the hardware run): ComputeBlockAmounts and
  // CheckOneParticle for every thread block and thread of ComputeLaunchSize(block count) and
  // ComputeLaunchSize(particle count), launch order and reverse, on the "cloud": partials bit for bit equal to the
  // CPU loop's, lowest failing id equal
  ```
- [ ] **Step 2: Implement** per Global Constraints sum orders; GPU (`fields.run_on == kGpu`): one thread per
  block runs `ComputeBlockAmounts` into a one-call `GpuArray<ParticleAmounts>` held by the GPU half (the view has
  no storage to own it; amended 2026-11-14, A31: `ComputeAmounts` keeps its one form, which holds this
  one-call GPU memory through `GpuArray` and keeps `kOutOfMemory`, because its callers pass views of particles
  of any owner, with no buffers to lend; a caller that must not allocate uses `ComputeAmountsPerBlock`), copied to the CPU
  and tree-summed there, which is `ComputeTreeSum`'s order; the GPU half returns `RunOnCallStatus`, mapped with
  `ToBufferError` and then to `kOutOfMemory` or `kRunOnCallFailed`. The header of `ComputeAmounts` states the
  same-bits promise and its reason. Check failure on the GPU: integer `atomicMin` of failing ids into the
  lowest-failing-id buffer of `particles_storage.h`. Callers in other tests (`ComputeAmounts` before and
  after a split or merge) test `has_value()` and use `*`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Check particle values and sum amounts in a fixed order`.
  Body: `GPU: compiled, not run`; records the coordinator decision that `ComputeAmounts` and
  `ComputeAmountsPerBlock` return `std::expected<..., ParticlesFailure>` (changes design 3.2).

### Task 18: Splitting one particle, on the CPU

**Kind:** integration

**Files:**
- Create: `native/fluidity/particles/split_particle.h`, `split_particle_cpu.cc`, `count_change_arrays.h`
  (private, shared), `split_particle_test.cc`
- Read: `docs/method-paper/fluidity-method.typ` `<proc-split-particle>`: checks in order grouping stale,
  index (`kBadInput`), count rule, fitted change, room and ids, parent's neighbourhood on `grid_before_resize`,
  each child's on `grid`; then write the children and mark the parent replaced

**Interfaces:**
- Consumes: `ComputeSplitChild`, `ComputeSplitOffset`, `FindNeighbourhood`, `grid::ComputeCellWidth`.
- Produces: `SplitParticleResult`, `SplitParticle` (design 3.5); private shared
  `WriteSplitChildren(const MaterialArraysView&, ...)`, `WriteMergedParticle(...)` (used by Tasks 19 to 22).

- [ ] **Step 1: Failing tests** (the "split parent" at `P`, liquid; committed on `BuildBigCellsGrid(1)`; try
  grouped for `BuildCentreSplitGrid(2)`; `BuildNeighbourhoodBuffers` for each grid).

  | Test | Change | Expected |
  |---|---|---|
  | `SplitParticle, ParentOnOldGridChildrenOnNewGrid_IsSplit` | none | `first_child_id` = next id before; after grouping, eight level-1 particles, each m = 1, `rest_volume` 0.25, `volume_ratio` 0.5 (exact), momentum 1e-14 rel; the same call with the resized grid as both grids: `kNeighbourhoodIncomplete` |
  | `SplitParticle, ChildNextToSolid_IsNotSplit` | the "solid face" set inside a solid in the resized grid's inputs | `kNeighbourhoodIncomplete`; try unchanged; `next_particle_id` unchanged |
  | `SplitParticle, ParentAtSizeBoundary_IsNotSplit` | `grid_before_resize` = `BuildCornerSplitGrid(1)` | same |
  | `SplitParticle, TryGroupedForOtherGrid_IsStale` | try grouped for `BuildBigCellsGrid`, `grid` = resized | `kGroupingStale`; try unchanged |
  | `SplitParticle, Snow_IsRefused` | snow material | `kMaterialKeepsItsCount` |
  | `SplitParticle, WithoutFittedChange_IsRefused` | flag off | `kNoFittedChange` |
  | `SplitParticle, WaitingRoomFull_IsTooManyParticles` | `max_added_per_try` 7 | `kTooManyParticles` |

- [ ] **Step 2: Implement** following `<proc-split-particle>` with `// Step N:` comments, checks in the order
  of the Read line. Children: `ComputeSplitChild`, `size_level + 1`, `flags`, fractions and `plate_side` copied, other
  extra arrays not carried (count rule). The parent's mark becomes 255. Cite `<sec-split>`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Split one particle inside a resize`.

### Task 19: Merging particles, on the CPU

**Kind:** integration

**Files:**
- Create: `native/fluidity/particles/merge_particles.h`, `merge_particles_cpu.cc`, `merge_particles_test.cc`
- Modify: `count_change_arrays.h`
- Read: `<proc-merge-particles>`: checks in order grouping stale, indexes (2 to 8, ascending, in range:
  `kBadInput`), count rule, agreement on material, `plate_side`, flags, `size_level` (`kParticlesDiffer`), fitted
  change, each particle's neighbourhood on `grid_before_resize`, gap (`kMergeTooSpread`), merged position's
  neighbourhood at `size_level - 1` on `grid`, room and ids

**Interfaces:**
- Consumes: `ComputeMergedParticle`, `ComputeMergedFractions`, `FindNeighbourhood`.
- Produces: `MergeParticlesResult`, `MergeParticles` (design 3.5).

- [ ] **Step 1: Failing tests** (committed on `BuildCentreSplitGrid(1)`, try grouped for
  `BuildBigCellsGrid(2)`; level-1 liquid particles, flags `kCarriesFittedChange`, C = 0 unless stated).

  | Test | Particles | Expected |
  |---|---|---|
  | `MergeParticles, EightClose_AreMerged` | "eight close" | `merged_id` = next id; one level-0 particle at `P` after grouping; amounts kept 1e-13 rel; `kinetic_energy_lost` equals `ComputeMergedParticle`'s |
  | `MergeParticles, TooSpread_IsRefused` | two at `{3.25,3.5,3.5}` and `{4.25,3.5,3.5}` (`2h` apart) | `kMergeTooSpread`; try unchanged |
  | `MergeParticles, DifferentSide_IsRefused` | `LiquidWithPlateSideSettings`, two close particles on sides `{7, kAlongNormal}` and `{7, kAgainstNormal}` | `kParticlesDiffer` |
  | `MergeParticles, CompleteOnlyOnNewGrid_IsRefused` | "eight close", but committed and `grid_before_resize` = `BuildBigCellsGrid` | `kNeighbourhoodIncomplete` |
  | `MergeParticles, OneOrNineIndexes_IsBadInput` | | `kBadInput` |
  | `MergeParticles, LevelZeroParticles_IsIncomplete` | two level-0 particles | `kNeighbourhoodIncomplete` |

- [ ] **Step 2: Implement** following `<proc-merge-particles>` with `// Step N:` comments and checks in the
  order of the Read line. Particles are sorted by id before any sum (indexes may cross cell places). Merged particle:
  `size_level - 1`, the common flags, `plate_side`, fractions from `ComputeMergedFractions`. The merged
  particles' marks become 255. Cite `<sec-merge>`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Merge particles inside a resize`.

### Task 20: Splitting and merging on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `_gpu.h` headers returning `RunOnCallStatus`, launches through `base/gpu_launch.cuh`; the
shared-formula `gpu` test compares floating-point results within design 6.6 instead of bit for bit (no header
promises the same bits for these formulas); result digests.

**Files:**
- Create: `native/fluidity/particles/split_particle_gpu.cu`, `split_particle_gpu.h` (private),
  `merge_particles_gpu.cu`, `merge_particles_gpu.h` (private), `split_and_merge_gpu_test.cc`,
  `split_and_merge_math_test_gpu.cu` (test-only GPU function that runs the shared formulas)
- Modify: `split_particle_cpu.cc`, `merge_particles_cpu.cc` (dispatch), `count_change_arrays.h` (the whole
  per-call body of a split and of a merge, checks after grouping included, as shared functions `ApplySplitParticle`
  and `ApplyMergeParticles` that the CPU path calls), `CMakeLists.txt` (the two `.cu` files through
  `fluidity_add_gpu_sources`; `fluidity_add_test(split_and_merge_gpu_test SOURCES split_and_merge_gpu_test.cc
  split_and_merge_math_test_gpu.cu LABEL gpu LINKS fluidity_particles_test_support)`)
- Read: `split_particle_test.cc`, `merge_particles_test.cc`, `split_and_merge_math_test.cc`,
  `find_neighbourhood_test.cc` (the cases repeated here)

**Interfaces:**
- Produces: no new public name.

- [ ] **Step 1: Failing tests** (label `gpu`): every test of `split_particle_test.cc` and
  `merge_particles_test.cc` run with `RunOn::kGpu` (grids and neighbourhood buffers on the GPU), results within
  design 6.6 of the CPU run, errors and ids identical; `SplitParticle, OnGpuTwice_GivesIdenticalBytes`; a `gpu`
  test that calls `ComputeSplitChild`, `ComputeMergedParticle`, `ComputeMergedFractions`,
  `ComputeRefillParticle`, `FindNeighbourhood` on the inputs of `split_and_merge_math_test.cc` and
  `find_neighbourhood_test.cc` and compares with the CPU: `is_complete` and every discrete value exact,
  floating-point values with `CheckWithinTolerance` at design 6.6, printing the largest difference per function.
  Each `gpu` test passes its main outputs (children, merged particle, ids) to `WriteResultDigest`. CPU evidence:
  each GPU function is a one-thread wrapper over `ApplySplitParticle` or `ApplyMergeParticles`, which the CPU path
  calls in Tasks 18 and 19, so those `cpu` tests are the main evidence until the hardware run (one thread, so no
  thread order to reverse).
- [ ] **Step 2: Implement.** One-thread GPU functions started with `ApplyGpuFunction(..., LaunchSize{1, 1}, ...)`
  calling the same shared functions and `count_change_arrays.h` writers as the CPU path; the GPU half returns
  `RunOnCallStatus`; the error and the failing id come back through the one-entry `Buffer<ParticlesFailure>` of
  `particles_storage.h` (Global Constraints), read by the CPU half with `ReadBuffer`.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Split and merge particles on the GPU`. Body:
  `GPU: compiled, not run`.

### Task 21: Keeping the count right inside a resize, on the CPU

**Kind:** integration

**Files:**
- Create: `native/fluidity/particles/adjust_particle_count.h`, `adjust_particle_count_cpu.cc`,
  `adjust_particle_count_test.cc`
- Read: `<proc-adjust-particle-count>`; `split_particle_cpu.cc`, `merge_particles_cpu.cc`,
  `count_change_arrays.h` (reuse their checks and writers)

**Interfaces:**
- Produces: `AdjustParticleCountResult`, `AdjustParticleCount` (design 3.5).

- [ ] **Step 1: Failing tests.**

  | Test | Setup | Expected |
  |---|---|---|
  | `AdjustParticleCount, ParticleInSmallerCells_IsSplitOnce` | "split parent" at `P`; committed `BigCells(1)`, resized `CentreSplit(2)` | `split_count` 1; after grouping, a second call: `split_count` 0; `ComputeAmounts` before and after 1e-13 rel |
  | `AdjustParticleCount, EightInBiggerCell_AreMerged` | "eight close"; committed `CentreSplit(1)`, resized `BigCells(2)` | `merged_group_count` 1; amounts 1e-13 rel |
  | `AdjustParticleCount, DriftedAcrossSizeChange_KeepsSizeLevel` | committed `CentreSplit(1)`, a level-0 particle at `P`; call 1 with `CentreSplit` as both grids; call 2 with resized `CentreAndFarSplit(2)` | not split in either; `skipped_incomplete_count` 1 each time; its arrays unchanged |
  | `AdjustParticleCount, AnyRangeCut_SameBytes` | committed `CentreSplit(1)` with "eight close" and the "split parent" at `Q`; resized `FarSplit(2)` | ranges from `grid::BuildRanges(grid, n)`, n = all, 1, 3, one call per range: identical result numbers and, after grouping and commit, identical snapshot bytes |
  | `AdjustParticleCount, ChildrenBeyondWaitingLimit_ChangesNothing` | as the first row with `max_added_per_try` 7 | `kTooManyParticles`; try unchanged |
  | `AdjustParticleCount, SnowAndUnfittedParticles_AreNotCandidates` | snow at `P`; liquid at `Q` with flag off | all counts 0 |

- [ ] **Step 2: Implement** following `<proc-adjust-particle-count>` and design 5.6 with these plan decisions:
  particles with `kCarriesFittedChange` off are not candidates and are not counted; in a cell, merge candidates
  form classes by (`plate_side`, `flags`) in order of their first member's id, each class in id order, chunks of
  eight, a last chunk of two to seven tried, a single one left; a refused split or merge chunk adds one to the
  matching skip count; counts and `kinetic_energy_lost` are block partials then `math::TreeSum`; first pass
  computes per-block counts, a prefix sum gives places in the waiting list (Global Constraints id order), so
  nothing is written when room or ids run out.
- [ ] **Step 3: Run** the gate; add `AdjustParticleCount` to allowed names if needed (`# DECIDED, design 11.2`).
  **Commit.** Subject: `Split and merge where a resize changed the cells`.

### Task 22: Keeping the count right on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** `adjust_particle_count_gpu.h` returning `RunOnCallStatus`; kernel bodies in
`adjust_particle_count_steps.h`, which the CPU loop of Task 21 then calls too, with a `cpu` test over every thread
block and thread; the prefix sum through `ComputeExclusivePrefixSum` with the particles' `order_buffers`; result
digests.

**Files:**
- Create: `native/fluidity/particles/adjust_particle_count_gpu.cu`, `adjust_particle_count_gpu.h` (private),
  `adjust_particle_count_steps.h` (shared), `adjust_particle_count_steps_test.cc`,
  `adjust_particle_count_gpu_test.cc`
- Modify: `adjust_particle_count_cpu.cc` (dispatch; its loops call the steps functions), `CMakeLists.txt`
  (`fluidity_add_gpu_sources(adjust_particle_count_gpu.cu)`, the header, the `cpu` steps test and the `gpu` test
  with `LINKS fluidity_particles_test_support`)
- Read: `adjust_particle_count_test.cc` (the cases repeated here)

**Interfaces:**
- Consumes: `ComputeExclusivePrefixSum(values, count, sums, OrderBuffers&)` with `order_buffers`;
  `ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`; `ToBufferError`.
- Produces: no new public name. Private, `adjust_particle_count_steps.h`, `FLUIDITY_CPU_AND_GPU`: the per-particle
  split test (`ComputeSplitDecision`), the per-cell merge chunks (`ComputeMergeChunks`) and the per-item write
  (`ApplyCountChangeWrite`).

- [ ] **Step 1: Failing tests.** `adjust_particle_count_steps_test.cc`, label `cpu` (the main evidence until the
  hardware run): runs the three steps functions for every thread block and thread of `ComputeLaunchSize(count)`,
  launch order and reverse, phase by phase (decisions, prefix sum, writes), on the cases of
  `adjust_particle_count_test.cc`, and compares decisions, places, ids, written arrays and block partials with the
  CPU loop exactly. Label `gpu`: every test of `adjust_particle_count_test.cc` on the GPU, CPU-against-GPU within
  6.6, ids and counts exact; `AdjustParticleCount, OnGpuTwice_GivesIdenticalBytes`. Each `gpu` test passes the
  result counts, the new ids and the snapshot bytes after grouping to `WriteResultDigest`.
- [ ] **Step 2: Implement** (design 6.3): kernel A, one thread per particle: split test, children count,
  refusal; kernel B, one thread per active cell: merge chunks; integer prefix sums over the per-block counts
  (`ComputeExclusivePrefixSum` with `order_buffers`) give each thread its places and ids (`next_particle_id +
  place`); kernel C writes; block partials of counts and energy copied to the CPU and tree-summed. Each kernel is a
  grid-stride loop around one steps function.
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Keep the particle count right on the GPU`. Body:
  `GPU: compiled, not run`.

### Task 23: Copying particle groups for a batched run

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the `gpu` test writes the loaded arrays' bytes with `WriteResultDigest`; a failed GPU copy leaves
the loaded slices unspecified (the run stops), a failed check writes nothing.

**Files:**
- Create: `native/fluidity/particles/particle_groups.h`, `particle_groups_cpu.cc`, `particle_groups_test.cc`,
  `particle_groups_gpu_test.cc`

**Interfaces:**
- Consumes: `grid::GridPiece` (`grid()`, `range()`, `range_in_whole_grid()`, `block_in_whole_grid()`),
  `grid::BuildGridPiece`, `grid::FindBorderBlocks`, `BorderWidth::kTouchingCells`; slice `WriteBuffer`,
  `ReadBuffer`, `CopyBuffer`.
- Produces: `WriteParticleGroups`, `ReadParticleGroups` (design 3.8).

- [ ] **Step 1: Failing tests** (whole try: `BuildCentreSplitGrid`, the six fixture materials, 30 particles
  each; piece: a range of 3 blocks and its 4-block border, chosen by the test with `FindBorderBlocks`; loaded
  `Particles` on the CPU with `max_cell_blocks` 7; the `gpu` file repeats the first test with loaded on GPU).

  ```cpp
  TEST(ReadParticleGroups, RoundTrip_SameBytes)  // loaded group_begin, cell_begin equal the hand computation
      // from the whole try; change positions in the loaded try, read {kPosition} back: range blocks equal the
      // loaded values, border blocks untouched (bytes)
  TEST(ReadParticleGroups, EveryRangeRead_CommitSucceeds)  // 3 ranges from BuildRanges, velocity_for_next_step
      // changed in each loaded try and read back: CommitTry succeeds
  TEST(ReadParticleGroups, OneRangeMissing_IsPartlyWritten)
  TEST(ReadParticleGroups, RemovedInLoaded_RemovedInWhole)  // 2 removed in loaded, one in a border block: after
      // the whole try's next grouping the range one is gone, the border one stays
  TEST(WriteParticleGroups, PieceTooFull_IsRefused)         // loaded limit 10: kTooManyParticles; whole unchanged
  TEST(WriteParticleGroups, WholeWrittenForSomeBlocks_CopiesNewestValues)  // whole changed velocity for the
      // range only before the copy: loaded sees new values in range blocks, committed ones in border blocks
  TEST(ReadParticleGroups, OtherRangeOrRegrouped_Fails)    // kBadInput; kGroupingStale after AddParticles
  TEST(ReadParticleGroups, MassField_IsWrongFields)
  ```
  `particle_groups_gpu_test.cc` (label `gpu`) passes the loaded and read-back arrays to `WriteResultDigest`.
- [ ] **Step 2: Implement** per design 5.10 and these plan decisions: `WriteParticleGroups` writes into the
  loaded particles' committed buffers, per array one slice for the range run and one per run of consecutive
  border blocks, from the whole try's newest buffer **per block**; clears the loaded try's marks; copies
  `velocity.for_this_step` and marks it written when the whole try wrote it for every copied block; records the
  piece's `block_in_whole_grid` (4 B per loaded block) and a grouping generation. `ReadParticleGroups` uses
  `BuildFieldsToChange`'s buffer rule on the whole try for the range blocks, copies the newest loaded values,
  marks them written, carries removal reasons. Cite `<sec-particle-copies>`.
- [ ] **Step 3: Glossary** row `loaded try`. **Run** the gate. **Commit.** Subject:
  `Copy particle groups between the whole try and a loaded piece`. Body: `GPU: compiled, not run`.

### Task 24: Scene-level tests

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** `particles_test_support_gpu.cu` is a `GPU_SOURCES` file of `fluidity_particles_test_support`;
`LoadedOnGpu_SameBytesAsWhole` runs the whole try on the GPU too (one processor, so bytes compare); `gpu` tests
write result digests.

**Files:**
- Create: `native/fluidity/particles/particles_scene_test.cc`, `particles_scene_gpu_test.cc`,
  `particles_test_support_gpu.cu` (test-only: `RunTestStep`'s position and velocity change as a GPU function, a
  grid-stride loop over a `FLUIDITY_CPU_AND_GPU` per-particle change in `particles_test_support.h` that the CPU
  `RunTestStep` calls too)
- Modify: `CMakeLists.txt` (`fluidity_add_test_library(fluidity_particles_test_support SOURCES
  particles_test_support.cc GPU_SOURCES particles_test_support_gpu.cu LINKS fluidity_base_test_support)`; the
  scene tests `LABEL cpu` and `LABEL gpu` with `LINKS fluidity_particles_test_support`)

- [ ] **Step 1: Tests.**

  ```cpp
  TEST(ParticleGroups, BatchedAndWhole_SameBytes)  // CentreSplit grid, six materials x 100 particles; three
      // RunTestStep-like steps whose change runs per range on a loaded try (CPU) for 1-, 2- and 5-block
      // batches (BuildRanges), read back {kPosition, kVelocityForNextStep}: committed snapshots equal the
      // one-piece run's in bytes
  TEST(RejectTry, AfterEveryKindOfChange_ChangesNothing)  // committed on CentreSplit(1): "eight close" and the
      // "split parent" at Q; try: add 3, remove 2, group for FarSplit(2), split Q's parent, merge the eight,
      // group, change every changeable field, then RejectTry: ReadSnapshot(all) bytes, next_particle_id,
      // state_version, grouping_version, grid_version equal; BuildTry succeeds
  // label gpu
  TEST(ParticleGroups, LoadedOnGpu_SameBytesAsWhole)  // whole try and loaded tries both on the GPU
  TEST(GroupParticlesByBlock, OnGpuMillionParticlesTwice_GivesIdenticalBytes)  // 1,000,000 liquid particles,
      // CentreAndFarSplit grid, max_cell_blocks 64
  TEST(Particles, CpuAndGpuStepSequence_WithinTolerance)  // three RunTestStep calls, add, remove, adjust on both
  // every gpu test passes its final snapshot bytes to WriteResultDigest
  ```
- [ ] **Step 2: Run** the gate. Fix any failure in the commit that owns the code (fixup), not here.
  **Commit.** Subject: `Check particles over whole step sequences and batch cuts`.

### Task 25: Comment pass and hand-over

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** Step 3 also records the storage's working buffers and failure buffers, the `ComputeAmounts` same-bits
promise, and `fluidity_particles_test_support` with its `GPU_SOURCES`.

**Files:**
- Modify: any file of `native/fluidity/particles/`, `docs/glossary.md`, `docs/method-paper/`, `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every particles file top to bottom as a C++ programmer who has never read
  a fluid paper. Is every comment true of the final code; does each say why; does each non-trivial public
  function have a worked example with numbers (the six-particle grouping, the 1 kg merge, the unit split);
  are the index convention and the buffer rule stated where a caller meets them; every word in the glossary;
  `Uses`/`Used by` lists right.
- [ ] **Step 2: Fold** each change into its owning commit (`git commit --fixup`, `git rebase --autosquash`),
  run format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design** section by section (3.1 to 3.8, 5, 6.4, 7) and record what was delivered
  differently: `max_cell_blocks` in `ParticlesSettings`; `ComputeMergedFractions`; 405 B per liquid particle;
  `AddParticles` without `.cu`; `run_on` in both views and `std::expected` from `ComputeAmounts` and
  `ComputeAmountsPerBlock` (coordinator); `ApplyParticleLimits` taking `max_cell_blocks` (coordinator); split
  visit order inside a block; unfitted particles not candidates; `order_buffers`, `sum_buffers` and the two
  failure buffers in `particles_storage.h` and their memory estimate entries; the same-bits promise of
  `ComputeAmounts`; Lean proof status from Task 7 ("built in a
  Lean container" or "written, not built here"), outside the gate; theorem rename.
- [ ] **Step 4: `docs/gpu-runs.md`:** (amended 2026-11-14, Foundation as built: the `gpu` tests write result
  digests, so the owner's repeat run covers them) row `particles | <tip hash> | <date> | - | compiled, not run |` with the
  `gpu` tests to run at the end of the programme: every test in `*_gpu_test.cc` of this folder (list them).
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module's commits, the gate output, all new
  public names with one-line reasons, the differences of Step 3, and the statement that no particles device
  code has run.
