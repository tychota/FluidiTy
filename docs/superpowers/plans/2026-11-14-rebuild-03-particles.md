# Rebuild Particles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan
> task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/fluidity/particles/`: every material's per-particle arrays with stable ids, the try and its
commit without a copy, grouping by cell block and cell place, add and remove, the count rule that splits and merges
particles to the size of their cell, snapshots, and the copies of particle groups for a batched run, on the CPU and
(compiled, not run) on the GPU.

**Architecture:** One owner, `Particles`, holds two buffers per array per material; a `ParticlesTry` hands out read
and write pointers and `CommitTry` flips which buffer is committed. Grouping sorts by (cell place, id) with two
`BuildStableOrder` passes. The split, merge and refill formulas live in one shared header that the CPU loops and the
GPU functions both call. Each operation is one public header with a private `_cpu.cc` and, where it has GPU work, a
`_gpu.cu` declared in a private `_gpu.h`.

**Tech Stack:** as `AGENTS.md` "Language" on the branch; Typst and CeTZ for the method paper; Lean 4 with Mathlib
for one proof file.

**Spec:** `docs/superpowers/specs/modules/03-particles-design.md` ("the design") and
`docs/superpowers/specs/2026-11-13-production-rebuild-design.md` (on `rebuild`: `docs/design/programme.md`).
Owner answers: `specs/modules/owner-answers.md` (A128 to A131 are this module's). Revisions of this plan are listed
in `2026-11-14-rebuild-03-particles-revisions.md` beside it, never in this file.

## Global Constraints

- `AGENTS.md` and `docs/glossary.md` on the branch are the authority for names, comments, structure, tests, the
  gate and the commit rule. The dispatch text (`docs/superpowers/sdd/dispatch-template.md`) carries the git rules
  and the words to keep out. Nothing here repeats them.
- **The design's section 3 is the contract.** A task implements the declarations of the sections it names exactly.
  Design 5.11 "Rules the code is built on" binds every task: indexes, written marks, removal marks, the waiting
  list, grouping keys and the in-place regroup, GPU failures, sum orders, the memory estimate entries, random
  values.
- **Files.** `src/fluidity/particles/`, namespace `fluidity::particles`. Unit tests beside their header
  (`<header>.test.cc`, at most 50 ms in debug, input written by hand). Functional tests in `tests/particles/`
  (`TIER functional`), at the sizes this plan gives. Shared test code in `src/fluidity/particles/testing/`.
  Private headers: `particles_storage.h` (CPU only: the buffers), `material_arrays.h` (shared: the raw pointers a
  GPU function needs), `<operation>_steps.h` (shared per-item functions), `<operation>_gpu.h`.
- **Errors.** `ParticlesError` has the 13 values of design 3.2. One mapping function turns a `BufferError` into a
  `ParticlesFailure`: `kOutOfMemory` and `kRunMemoryTooSmall` give `kOutOfMemory` (magnitude the bytes asked),
  `kRunOnCallFailed` gives `kRunOnCallFailed`, `kSizeMismatch` gives `kInternalFault`. A state no input produces is
  `kInternalFault` in every build, listed under `Fails:`, with a test that reaches it through a private form.
- **Ranges (A116).** Every function that takes a `grid::CellBlockRange` returns `kBadInput` for a range not inside
  the grouping's blocks, and for an empty range unless the grouping has no block.
- **Constants:** `kMaxIngredientCount = 8`; `kCarriesFittedChange = 1`, `kHasOwnMotion = 2`;
  `kFractionSumTolerance = 1e-12`; `kMergeGapFloor = 1e-9`; `kMaxMergeWaitPasses = 16`; the split offset of a unit
  cell is `0.4330127018922193`; ids start at 1.
- **Tolerances (design 6.6).** Ids, counts, order, flags, size levels, `group_begin`, `cell_begin`: exact. Floats,
  CPU against GPU: `|a - b| <= floor + 1e-13 * max(|a|, |b|)`, floor `1e-13 x` cell width (positions), largest
  speed of the case (velocities), largest value of the case (amounts, fitted change). `ComputeAmounts` and
  `ComputeAmountsPerBlock`: the same bits on both processors unless NaN. Two runs on one processor: the same bytes.
- **GPU.** Every task with a `_gpu.cu` writes its `gpu` tests in the same commit, labels `no_gpu` wherever the
  behaviour without a GPU can be reached in the container, and ends its commit message with the line
  `GPU: compiled, not run`. A `gpu` test writes a result digest (`WriteResultDigest`) of what it checks. Each GPU
  function is a thin loop over the shared per-item functions, and a `cpu` test runs those functions over every
  thread of `ComputeLaunchSize(count)` in launch order and in reverse (`grid/testing/thread_replay.h`).
- **`main` is Y-up and is never built.** Read it with `git show main:<path>`. Rotate its vectors and matrices with
  `R = rows {1,0,0},{0,0,-1},{0,1,0}` (`v' = R v`, `C' = R C R^T`); match split children by position.
- **One wording** for particle size in every comment and paper text: design 5.6 "Particle size follows the cell".
- **Review findings** follow plan rule 11 of `00-how-to-write-a-module-plan.md`. Every review checks each number and
  claim of the comments the diff adds.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`: Foundation (`base/`), contact
and grid are built (grid Tasks 1 to 28). Prove it:

```bash
cd /Users/tychota/Documents/Code/openSource/FluidiTy-rebuild && uv run python tools/check_commit.py
```

Expected: exit code 0. Read `AGENTS.md`, `docs/glossary.md` and the sections of the design your brief attaches.

## Review Focus

1. **A second grouping in one try** (a resize groups, adjusts the count, then groups again): the arrays are already
   in the try buffers, so the regroup is in place; the committed values must survive for `RejectTry`. Pinned:
   Task 11, `GroupParticlesByBlock, TwiceInOneTry_KeepsCommittedValues`; GPU in Task 16.
2. **A second range of an array already changed for another range** in one try: the read pointer must show the
   committed values of the new range, not old bytes of the try buffer. Pinned: Task 12,
   `BuildFieldsToChange, SecondRangeOfSameArray_ReadsCommittedValues`.
3. **A removed particle that left the grid** must be dropped, not reported `kOutsideGrid`. Pinned: Task 13,
   `GroupParticlesByBlock, RemovedParticleOutsideGrid_IsDropped`.
4. **A position on a cell side and one rounding step below it** belongs to the cell `FindCellAt` gives, on both
   sides of a block and of a size change. Pinned: Task 10, the positions of the reference; Task 11 runs them.
5. **Merge far from the origin:** sums are relative to the first particle. Pinned: Task 5,
   `ComputeMergedParticle, FarFromOrigin_SameAsNearOrigin`.
6. **Merge waiting off by one:** `merge_wait_passes` grows before the test, so a group that appears at pass 1 merges
   at pass 16; it must persist through commit and a saved state. Pinned: Task 19,
   `AdjustParticleCount, ThreeSmallInLargeCell_MergeAfterSixteenPasses`; Task 17's snapshot test.
7. **Batch copy-in when the whole try wrote an array for some blocks only:** `WriteParticleGroups` takes the newest
   value per block. Pinned: Task 22, `WriteParticleGroups, WholeWrittenForSomeBlocks_CopiesNewestValues`.

---

### Task 1: Contact accepts an output that is its own input

**Kind:** numerics. **Review:** Opus: (1) trace one particle through `ComputeParticleMoveAt` and
`ApplyParticleOutputsAt` on each processor and show no value is read after its own index is written; (2) which overlap
that is not "exactly its own input" still passes? (3) which plausible wrong implementation do the tests accept?
**Needs:** none. **Touches:** `src/fluidity/contact/move_particles.h`, `move_particles_cpu.cc`,
`move_particles.test.cc`, `tests/contact/move_particles_on_gpu.test.cc`, `docs/method-paper/fluidity-method.typ`
(`<sec-contact>`).

**Brief:**
- Design sections: 5.3 step 2; owner answer A128.
- Consumes, checked: `CheckCallArrays`, `ToByteRange`, `AreApart` (`move_particles_cpu.cc`);
  `ComputeParticleMoveAt`, `ApplyParticleOutputsAt` (`move_particles_steps.h`).
- Decided before dispatch: `MovedParticles::position` may equal `MoveParticlesInputs::start_position`, and
  `MovedParticles::velocity_for_next_step` may equal `velocity.for_next_step`: the same pointer, so the same
  bytes. Every other shared byte stays `kBadInput` (an output over `velocity.for_this_step`, over an input that
  starts elsewhere, over another output, over the solids). No GPU file changes: the check runs before either half.
  The header says: after a failed call, an input written in place is unspecified as the outputs are.
- Tests, unit, label `cpu`: `TEST(MoveParticles, PositionInPlace_SameBytesAsApart)` and
  `TEST(MoveParticles, NextStepVelocityInPlace_SameBytesAsApart)` on the ten floor particles of the file (two of
  them hit), both in place at once in a third test; `OutputOverlappingAnArray_IsBadInput` keeps its other cases
  and gains "position over `velocity.for_this_step`" and "position at `start_position + 1`". Label `gpu`: the two
  in-place cases, same bytes as the CPU within contact's tolerance.
- If the design is wrong: stop BLOCKED with the particle index whose bytes differ.

- [ ] Failing tests; implement; method paper sentence in `<sec-contact>`; gate; commit.
  Subject: `Let MoveParticles write a particle array in place`.

### Task 2: Method paper: what a particle carries, grouping, try, add and remove, copies

**Kind:** numerics. **Review:** Opus: (1) does each numbered step of `<proc-group-particles>` and
`<proc-commit-try>` match design 5.3, 5.8 and 5.11 item by item? (2) is every number of the memory table the sum of
5.11 item 8? (3) which wrong reading of the pseudocode would a programmer new to fluids make?
**Needs:** none. **Touches:** `docs/method-paper/fluidity-method.typ`, `docs/method-paper/figures/`,
`docs/glossary.md`.

**Brief:**
- Design sections: 1, 5.1, 5.2, 5.3, 5.7, 5.8, 5.9, 5.10, 5.11, 6.4, 8 (the labels and figures of this half), 9.
- Decided: section `<sec-particles>` after `<sec-grid>`, with `<sec-particle-fields>` (tables
  `<tab-particle-fields>`, `<tab-particle-memory>`), `<sec-grouping>`, `<sec-particle-try>`, `<sec-add-remove>`,
  `<sec-particle-copies>`; figures `<fig-particle-fields>`, `<fig-grouping>`, `<fig-particle-try>`; procedures
  `<proc-group-particles>`, `<proc-commit-try>`. Labels use `-`, not `:`. The grouping figure shows the six
  particles of Task 8's "six on three blocks" with their ids before and after. The try figure shows a second change
  in place (A128). Glossary rows: `try`, `commit`, `reject`, `waiting`, `group`, `group_begin`, `cell_begin`,
  `written mark`, `removal mark`, `loaded try`, `ingredient`, `rest_volume`, `volume_ratio`.
- Tests: the gate's `paper` step (labels resolve, no page overflow).

- [ ] Write the text, figures and procedures; gate; commit. Subject: `Describe particles, grouping and the try`.

### Task 3: Method paper: split, merge and keeping the count right

**Kind:** numerics. **Review:** Opus: (1) derive `<eq-split-offset>` and the gap of `<eq-merge-gap>` from the
spread and check the worked numbers (offset 0.4330127018922193; two particles at ±0.25: gap 0.6875, energy lost
0.9375 J); (2) does `<proc-adjust-particle-count>` state the pass-16 rule and the surface rule as design 5.6?
(3) which wrong reading would the pseudocode allow?
**Needs:** Task 2. **Touches:** the same three paths as Task 2.

**Brief:**
- Design sections: 5.4, 5.5, 5.6, 8 (this half), 10; A65, A67, A129, A130.
- Decided: `<sec-split>` with `<eq-split-offset>`, `<sec-merge>` with `<eq-merge-fit>`, `<eq-merge-gap>`,
  `<sec-adjust-count>`; figures `<fig-split>` (two panels as design 8), `<fig-merge>`; procedures
  `<proc-split-particle>`, `<proc-merge-particles>`, `<proc-adjust-particle-count>`. The text says in one sentence
  that no completeness check runs and why (A130). Glossary rows: `mismatched particle`, `too large`, `too small`,
  `lone particle`, `chunk` (of a merge), `gap` (of a merge), `merge_wait_passes`, `size_level` (of a particle).
- Tests: the gate's `paper` step.

- [ ] Write; gate; commit. Subject: `Describe how particles are split and merged to the size of their cell`.

### Task 4: Errors, fields and field lists

**Kind:** mechanical. **Review:** Sonnet, scoped to: the 13 values and their order, the bit values of `Field`,
struct sizes asserted, the mapping of each `BufferError`.
**Needs:** Task 2. **Touches:** `particles/particle_fields.h`, `particles_error.h`, `particles_error_message.h`,
`particles_error_message.cc`, their tests, `particles/CMakeLists.txt`, `src/fluidity/CMakeLists.txt`.

**Brief:**
- Design sections: 3.1, the error part of 3.2 (`ParticlesError`, `ToMessage`, `ToName`, `ParticlesFailure`,
  `CountRule`, `ExtraFields`), 11.15, 11.23.
- Consumes, checked: `PerParticleVelocity` (`base/per_particle_velocity.h`), `contact::PlateSide`
  (`contact/solid_shapes.h`), `grid::CellBlockRange`, `grid::SizeLevel` (`grid/grid_ids.h`), `RunOn`,
  `BufferError` (`base/buffer.h`); the pattern of `grid/grid_error.h` and `grid_error_message.h`.
- Decided: `ParticlesError` lives in the shared `particles_error.h` (no `<expected>`), with
  `ParticlesFailure` (`static_assert(sizeof == 24)`: error at 0, `material_index` at 4, `particle_id` at 8,
  `magnitude` at 16; declare the fields in that order) and `ParticlesFailure ToParticlesFailure(BufferError,
  std::uint64_t bytes_asked)` in the CPU-only message header. `kInternalFault`'s message says the fault is in
  FluidiTy and asks for a report with the scene. `ParticleFields::size_level` is `const grid::SizeLevel*`.
- Tests, unit: `ParticlesError, OutOfMemory_IsTheLastValue` (13th); `ToName, EveryError_LowerSnakeCase`;
  `ToMessage, EveryError_HasItsOwnText`; `ToParticlesFailure, EachBufferError_MapsAsStated`;
  `ToFieldList, ThreeFields_OrsTheirBits`; `HasField`; a `gpu_compile` test that includes both shared headers.

- [ ] Failing tests; implement; gate; commit. Subject: `Add the particle fields and the errors of particles`.

### Task 5: Split, merge and refill formulas

**Kind:** numerics. **Review:** Opus: (1) recompute by hand the expected values of `TwoEqual_MeanAndSum` and of
`MergeEnergy` (C*xx 0.5, 4 J before, 0.25 J after); (2) is `QuarterWidthOffset_WouldLoseEnergy` the only test that
rejects the offset `H/4`, and does one test reject a merged `C*` without the `spread(h)/spread(2h)` factor?
(3) are sums relative to the first particle? (4) which plausible wrong implementation do the tests accept?
**Needs:** Task 3, Task 4. **Touches:** `particles/split_and_merge_math.h`, its unit tests,
`tests/particles/merge_energy_on_many_cases.test.cc`, both `CMakeLists.txt`.

**Brief:**
- Design sections: 3.6, 5.4, 5.5, 5.7 (`ComputeRefillParticle`), and from 7 the rows `ComputeSplitOffset`,
  `ComputeSplitChild` (4), `ComputeMergedParticle` (5), `ComputeRefillParticle` (2).
- Consumes, checked: `grid::ComputeCompleteWeightedSpread` (`grid/weight_curve.h`), `math::Sqrt`, `math::Abs`
  (`base/scalar_math.h`), `Vec3`, `Mat3`.
- Decided: `void ComputeMergedFractions(const double* fractions, const double* current_volumes, int
  particle_count, std::uint32_t ingredient_count, double* merged)` beside `ComputeMergedParticle`
  (current-volume-weighted, in id order). `particle_count` outside 2..8 or 1..9 (refill) is a `Requires:` with
  `assert`. Ported from `main:native/src/liquid_resampling_3d.cc:358-458` (split), `:460-562` and `:32-86, 215-354`
  (merge), `main:native/src/liquid_band_handoff_3d.cc:523-610` (refill).
- Tests, unit: the rows above with the design's numbers, plus `ComputeMergedParticle,
  FarFromOrigin_SameAsNearOrigin` (the five-particle case moved by `{1e6, 0, 0}`: velocity, `C*` and gap within
  1e-9 relative) and `ComputeMergedParticle, MergedChangeWithoutSpreadRatio_IsRejected` (a case whose `C*` differs
  by that factor, expected value by hand). Functional (`merge_energy_on_many_cases`): `EnergyNeverGrows` on 1000
  seeded cases of 2 to 8 particles with gap above the floor.

- [ ] Failing tests; implement; gate; commit. Subject: `Add the split, merge and refill formulas`.

### Task 6: The merge proof in Lean and the gate's proof step

**Kind:** numerics. **Review:** Opus: (1) do the two theorem statements say what design 5.5 claims, with the
gap as defined there, and no stronger hypothesis than a positive gap? (2) does the gate step fail when `lake` is
missing or a `sorry` is present? (3) what does the proof not cover (rounding)?
**Needs:** Task 5. **Touches:** `proofs/` (new), `tools/check_commit.py`, `tools/check_commit_test.py`,
`AGENTS.md` ("Commands"), `.github/workflows/checks.yml`.

**Brief:**
- Design sections: 5.5, 7 ("Proof"); A21, A75.
- Decided: a Lake project `proofs/` (toolchain `leanprover/lean4:v4.36.0`, the one installed; Mathlib pinned in
  `lake-manifest.json`, fetched with `lake exe cache get`), one file `proofs/Fluidity/Merge.lean` with
  `merge_preserves_moments_and_bounds_energy` and `merge_preserves_volume_inventories`, restated from
  `main:proofs/Fluidity/Merge.lean` for the single quadratic spread in the new words. New gate step `proof`, after
  `paper`, not in `quick`: `lake build` in `proofs/`, then `leanchecker` (the independent checker, on `PATH` from
  elan) on the module; `FAIL proof: lake is not installed` when it is missing; a `sorry` or an `axiom` beyond
  Lean's three fails. CI installs elan in the `rules` job and runs the step there.
- Tests: `tools/check_commit_test.py` gains "the step is listed after paper", "missing lake fails", "a sorry
  fails" (on a temporary project of three lines).
- If the proof cannot be closed in the task's time: commit nothing, stop BLOCKED with the goal that remains.

- [ ] Project and proof; gate step and its tests; `AGENTS.md` line; gate; commit.
  Subject: `Prove that a merge with a positive gap cannot add energy`.

### Task 7: Comparison of the formulas with the proof of concept

**Kind:** mechanical (comparison only: no separate review). **Needs:** Task 5.
**Touches:** `particles/testing/particles_main_comparison_cases.h`,
`tests/particles/formulas_main_comparison.test.cc`, `tests/particles/CMakeLists.txt`.

**Brief:**
- Design section 7 ("Comparison with `main`"), section 2 (the rotation).
- Decided: the cases are data read from `main:native/tests/` (the tests of `SplitUniformLiquidParticle3D`,
  `MergeUniformLiquidParticles3D` with `kQuadratic`, `ReseedLiquidCoreCell3D`): 20 inputs in all with the values
  `main`'s tests expect, as `contact/testing/contact_main_comparison_cases.h` does it. The header says how each
  expected value was found and that `main` was not built. Agreement 1e-12 relative.

- [ ] Cases and test; gate; commit. Subject: `Compare the split, merge and refill formulas with the old code`.

### Task 8: Test grids, materials and named particle sets

**Kind:** mechanical. **Review:** Sonnet, scoped to: each grid's block count and order, the hand-computed
`group_begin` and `cell_begin` of the six, that every named position is in the cell the text says by `FindCellAt`.
**Needs:** Task 4. **Touches:** `particles/testing/particles_support.h`, `.cc`, `particles_support.test.cc`,
`particles/CMakeLists.txt`.

**Brief:**
- Consumes, checked: `TestGrid`, `BuildTestGrid` (`grid/testing/grid_support.h`), `FindCellAt`,
  `MaterialParticlesSettings` (declared here from design 3.2 if Task 9 has not landed: Task 9 then moves nothing).
- Decided (test library `fluidity_particles_test_support`): `SplitMix64` is created here (no generator of that
  name exists on the branch): `struct SplitMix64 { std::uint64_t state; }`, `std::uint64_t ComputeNext(SplitMix64&)`,
  `double ComputeNextUnitDouble(SplitMix64&)`, seed 20261114, with a test of its first three values. Frame origin
  `{0,0,0}`, biggest cell width 1,
  3 size levels (widths 1, 0.5, 0.25); domain of level-0 cells x 0..15, y 0..7, z 0..7; `max_cell_blocks` 64.
  Grids, each taking `(GridVersion, RunOn)`: `BuildBigCellsGrid` (all level 0); `BuildCentreSplitGrid` (big cells
  with x, y, z in 2..5 at level 1); `BuildFarSplitGrid` (x 10..13, y, z 2..5 at level 1); `BuildCascadeGrid(int
  step)` (A: big cells 1..6 at level 1 and 2..5 at level 2; B: 1..6 at level 1; C: all level 0);
  `BuildThreeBlockGrid` (level-0 cells x 0..7, y, z 0..3, big cell (4,0,0) split: blocks L0 (0,0,0), L0 (1,0,0),
  L1 (2,0,0)). Points `P = {3.5,3.5,3.5}`, `Q = {11.5,3.5,3.5}`. Six materials: liquid, mixture of 3, snow, springy
  solid with side, gas markers, liquid with plate side. Named sets: *split parent* (level 0, m 8, rest volume 2,
  volume ratio 0.5, v {1,2,3}, C rows {1,2,3},{4,5,6},{7,8,9}, fitted change on); *eight close* (level 1, m 1,
  rest volume 0.125, at `P + (±0.125, ±0.125, ±0.125)`, v 0, C 0); *six on three blocks*, added in this order:
  `{5.5,0.5,0.5}`, `{0.5,0.5,0.5}`, `{4.25,0.25,0.25}`, `{0.6,0.4,0.5}`, `{4.75,0.25,0.25}`, `{1.5,0.5,0.5}`
  (grouped ids 2, 4, 6 | 1 | 3, 5; `group_begin {0,3,4,6}`); *cloud* (200 liquid particles, `SplitMix64`-uniform in
  `[1,7)^3`, v in `[-1,1]^3`, m = rest volume = 0.001, on the centre-split grid); `BuildSideAndStepPositions(grid)`:
  for each of 8 chosen cells (two per grid kind: beside a block side and beside a size change) its low corner, the
  centre of each low side, and each of those one `std::nextafter` step below on that axis.
- Tests, unit: one per grid (block count, order), one per named set (the cell of each position).

- [ ] Failing tests; implement; gate; commit. Subject: `Add the test grids and particle sets of particles`.

### Task 9: The particle arrays, their limits and their memory estimate

**Kind:** integration. **Review:** Opus: (1) trace one liquid and one snow material through the one function that
walks the arrays, and show the bytes handed out equal the estimate entry by entry; (2) after a failed
`ApplyParticleLimits`, what state is the object in? (3) which plausible wrong implementation do the tests accept?
**Needs:** Task 4, Task 8. **Touches:** `particles/particles.h`, `particles_cpu.cc`, `particles_storage.h`,
`material_arrays.h`, `particle_memory_estimate.h`, `.cc`, `particles.test.cc`, `particles.no_gpu.test.cc`,
`tests/particles/particles_on_gpu.test.cc`, both `CMakeLists.txt`.

**Brief:**
- Design sections: 3.2 except `CheckParticles` and `ComputeAmounts*` (`kFractionSumTolerance` is declared here,
  for Task 11); 3.7 (`BuildParticlesMemoryEstimate`); 5.1,
  5.2, 5.11 items 1, 4, 6, 8; 6.4; 11.14.
- Consumes, checked: `BuildBuffer(RunOn, std::size_t, RunMemory&)` (`base/run_memory.h`), `CopyBuffer` slice form
  (`base/buffer.h`), `BuildOrderBuffers`, `BuildSumBuffers` with `RunMemory&`, `ComputeOrderBuffersSize`,
  `ComputeSumBuffersSize`, `ComputeRunMemoryBytes`, `CheckCanRunOn`.
- Decided: `BuildFieldsToRead(const Particles&, ...)` returns `std::expected` (range rule); a material with no
  grouping yet and a grid with no block give a view of 0 particles. A fresh `Particles` has counts 0, versions 0,
  `next_particle_id` 1, time 0. Private, for tests, in `particles_storage.h`: `ComputeAllocatedBytes`,
  `ApplyNextParticleIdForTest`, `ApplyCommittedIdForTest`, `ApplyMergeWaitPassesForTest`. `kInternalFault`: the
  walk hands out a piece count that differs from the estimate's entry count (reached by a private form that takes
  the entries).
- Tests, unit (limits 1000 particles, 100 waiting, 64 blocks): `BuildParticles`: `LiquidOnly_AllocatesNoExtraArrays`,
  `MergeWaitPassesOnlyForSplitAndMergedMaterials`, `ZeroMaterialIdOrRepeatedId_IsBadInput`,
  `TooManyIngredients_IsBadInput` (9), `ZeroMaxCellBlocks_IsBadInput`, `Fresh_HasNoParticles`,
  `IntoRunMemory_HandsOutExactlyTheEstimate`, `RunMemoryOnePieceShort_IsOutOfMemory`,
  `EntriesDisagreeWithWalk_IsInternalFault`; `BuildParticlesMemoryEstimate, EveryMaterialKind_MatchesAllocation`;
  `ApplyParticleLimits`: `RaisedLimit_KeepsCountsAndVersions`, `RaisedBlockLimit_ReallocatesBlockTables` (64 to
  128), `BelowCurrentCount_IsBadInput`, `LowerBlockLimit_IsBadInput`, `IntoNewRunMemory_HandsOutExactlyTheNewEstimate`;
  `BuildFieldsToRead, RangeOutsideGrid_IsBadInput`, `EmptyRange_IsBadInput`. `no_gpu`:
  `BuildParticles, OnGpuWithoutAGpu_IsRunOnCallFailed`. `gpu`: `OnGpu_HandsOutExactlyTheEstimate`.

- [ ] Failing tests; implement; gate; commit. Subject: `Add the particle arrays and their memory estimate`.

### Task 10: The grouping of a set of particles and the newest value of an array, stated plainly

**Kind:** definition. **Review:** Opus: is the reference independent of the rule (no key, no stable order, no
`near` form), and is the rule complete (every particle once, removed ones dropped, sides and steps below them)?
Does the newest-value model know nothing of buffers and marks? Which smaller or shifted set does each checker
accept?
**Needs:** Task 8. **Touches:** `particles/testing/grouping_reference.h`, `.cc`, `grouping_reference.test.cc`,
`particles/testing/newest_values_model.h`, `newest_values_model.test.cc`, `particles/CMakeLists.txt`.

**Brief:**
- Design sections: 5.8 with its Property, 5.11 items 1, 3, 5; design 4 "A position belongs to the cell `FindCellAt`
  returns".
- Decided: `struct GroupedParticles { std::vector<std::uint64_t> id; std::vector<std::uint64_t> group_begin;
  std::vector<std::uint32_t> cell_begin; }`; `GroupedParticles BuildGroupingReference(const grid::GridTables&,
  std::span<const Vec3> position, std::span<const std::uint64_t> id, std::span<const std::uint8_t>
  removal_mark)`: per particle `FindCellAt(tables, position)` (the form without `near`), then
  `std::stable_sort` of (cell place, id) pairs, tables by counting. `std::string FindFirstGroupingDifference(const
  GroupedParticles& expected, const ParticleFields& actual)`: empty when equal, else the first difference in words.
- Decided, for 5.11 item 2 (which values a try shows): `template <typename T> struct NewestValuesModel {
  std::vector<T> committed; std::vector<T> in_try; std::vector<std::uint8_t> is_written; }` with
  `ApplyWrite(model, first_particle, std::span<const T>)` and `T ComputeNewest(model, particle)`: one flag per
  particle, no block, no buffer. Property: a `read` pointer of a range shows `ComputeNewest` for every particle of
  the range; after `CommitTry` the committed values are `ComputeNewest`; `CommitTry` fails exactly when some
  particle of a changed array has no flag. Tasks 12 and 22 check their arrays against it.
- Tests, unit: the model on 6 values written for particles 0..2 then 2..4; the six on three blocks give the
  hand-computed tables of Task 8; the checker is empty on the
  reference's own output and names the fault for each of five wrong groupings built by hand: a particle on a cell
  side put in the lower cell, one particle missing, two ids swapped inside a cell place, a removed particle kept, a
  `cell_begin` entry off by one.
- If the property is wrong or incomplete: stop BLOCKED with the grid, the position and the two sets.

- [ ] References, checker and tests; gate; commit. Subject: `State which group and cell run a particle belongs to`.

### Task 11: The try, adding particles and grouping on the CPU

**Kind:** integration. **Review:** Opus: (1) trace the six on three blocks through the two passes and the gather;
(2) trace a second grouping in one try and show no committed byte is written; (3) which input reaches
`kTooManyParticles` and which `kOutsideGrid`, and is the lowest id reported? (4) which plausible wrong
implementation do the tests accept?
**Needs:** Task 9, Task 10. **Touches:** `particles/particles_try.h`, `particles_try_cpu.cc`, `add_particles.h`,
`add_particles_cpu.cc`, `group_particles_by_block.h`, `group_particles_by_block_cpu.cc`,
`group_particles_by_block_steps.h`, their tests, `tests/particles/grouping_against_reference.test.cc`,
`tests/particles/add_particles_on_gpu.test.cc`, both `CMakeLists.txt`.

**Brief:**
- Design sections: 3.3 (`ParticlesTry`, `BuildTry`, `BuildFieldsToRead`, `RejectTry`, and `CommitTry` for a try
  that changed no array), 3.4, 3.5 (`AddReason`, `NewParticles`, `AddParticlesResult`, `AddParticles`), 5.3, 5.7,
  5.8, 5.11 items 1, 3, 4, 5; from 7 the rows `AddParticles` (3), `GroupParticlesByBlock` (3 and the property
  row), `CheckGrouping, AfterAdd_IsStale`.
- Consumes, checked: `BuildStableOrder(keys, count, previous_order, order, OrderBuffers&)` (`base/stable_order.h`),
  `WriteBuffer` slice form, `FindCellAt` both forms (`grid/find_cell_at.h`), `StaggeredGrid::tables`, `version`,
  `all_blocks`; `BuildGroupingReference`, `FindFirstGroupingDifference` (Task 10).
- Decided: `AddParticles` has no `_gpu.cu` (design 3.5). Its checks, in order: material index (`kBadInput`), spans
  (`kWrongFields` for a carried array missing or an uncarried one given; `kBadInput` for lengths), values, room
  (`kTooManyParticles`, magnitude the excess), ids (`kIdsExhausted`). The shared `ComputeGroupingKey(tables,
  position, removal_mark)` of `group_particles_by_block_steps.h` is what Task 16 runs per thread. `CheckGrouping`:
  `kGroupingStale` when anything waits, a removal mark is set or the grid version differs; then
  `kMovedTooFarSinceGrouped` when an axis of the move is strictly above the limit (lowest id; magnitude the largest
  excess). A try destroyed without commit acts as `RejectTry`.
- Tests, unit: `BuildTry, WhileAnotherTryLives_IsTryInProgress`; `RejectTry, ThenBuildTry_Succeeds`;
  `CommitTry, NothingChanged_BumpsStateVersion` (time 0.25); `AddParticles`: `GivesConsecutiveIds_AndNoOwnMotion`,
  `NonFiniteOrNonPositiveValue_IsBadInput`, `FractionsNotSummingToOne_IsBadInput`,
  `CarriedArrayMissingOrExtraGiven_IsWrongFields`, `MoreThanWaitingRoom_IsTooManyParticles` (room 4, add 5),
  `IdsWouldOverflow_IsIdsExhausted`, `OnCpu_NoCopyCounted`, `NewParticles_WaitZeroPasses`;
  `GroupParticlesByBlock`: `InsideCell_IdsAscend` (the six), `ParticleOutsideGrid_ReportsLowestId`,
  `TwiceInOneTry_KeepsCommittedValues`, `MoreBlocksThanLimit_IsTooManyParticles` (limit 2);
  `CheckGrouping, AfterAdd_IsStale`. Functional (`grouping_against_reference`):
  `Groups_MatchPerParticleReference` on the 5 fixture grids with the cloud (200) plus
  `BuildSideAndStepPositions`; `AnyInputOrder_SameResult` (3 orders of the six, bytes). `gpu`:
  `AddParticles, OnGpu_CopiesCountedPerArray`, `OnGpu_SameIdsAndAmountsAsCpu`.

- [ ] Failing tests; implement following `<proc-group-particles>` with `// Step N:`; gate; commit.
  Subject: `Add the particle try, adding particles and grouping by cell block`.

### Task 12: Changing arrays in a try and committing them

**Kind:** integration. **Review:** Opus: (1) trace the written marks through "blocks [0,3) then [2,6)" and show
which bytes `read` shows for each block, against `NewestValuesModel`; (2) show no write pointer can point at a
committed buffer; (3) what does
`CommitTry` name when an array is partly written? (4) which plausible wrong implementation do the tests accept?
**Needs:** Task 11. **Touches:** `particles/particles_try.h`, `particles_try_cpu.cc`, `particles_try.test.cc`,
`particles.test.cc`, `particles/testing/particles_support.h`, `.cc`,
`tests/particles/try_values_against_model.test.cc`.

**Brief:**
- Design sections: 3.3 (`BuildFieldsToChange`, `CommitTry`), 5.3, 5.11 item 2; from 7 the rows `CommitTry` (2),
  `CheckGrouping, MovedTooFar`, `ApplyParticleLimits, RaisedBlockLimit`.
- Decided: `kArrayPartlyWritten` names the material and, in `magnitude`, the first unwritten block. A private
  writer hands `AdjustParticleCount` the `merge_wait_passes` array and sets its mark; `BuildFieldsToChange` gives
  `kWrongFields` for `kMass`, `kMergeWaitPasses` and an array the material does not carry. Test helper
  `RunTestStep(Particles&, grid, GroupingSettings, double step_duration)`: `BuildTry`; `CheckGrouping`, on failure
  `GroupParticlesByBlock`; per material over all blocks change position and velocity (`v += {0,0,-9.81 dt}`,
  `x += v dt`); `CommitTry(time + dt)`.
- Tests, unit (the cloud): `CommitTry, ChangedArraysOnly_NoCopy`; `CommitTry,
  ArrayWrittenForSomeBlocks_IsPartlyWritten`;
  `BuildFieldsToChange`: `SecondRangeOfSameArray_ReadsCommittedValues`, `SecondChange_ReadEqualsWrite` (A128),
  `MassOrUncarriedArray_IsWrongFields`, `ThisStepVelocity_ReadIsNullUntilWritten`, `EmptyRange_IsBadInput`;
  `CheckGrouping, MovedTooFar_NamesLowestId` (limit 0.1, moved 0.2 on y: magnitude 0.1 within 1e-15);
  `ApplyParticleLimits, RaisedLimit_KeepsEveryValue` and `RaisedBlockLimit_ReallocatesBlockTables` extended to
  every array, then a grid of 88 blocks groups (before the raise: `kTooManyParticles`). Functional
  (`try_values_against_model`): the cloud, 40 seeded changes of position over seeded ranges of 1 to 5 blocks: after
  each, every `read` value equals `ComputeNewest`; `CommitTry` succeeds exactly when the model has every flag.

- [ ] Failing tests; implement following `<proc-commit-try>`; gate; commit.
  Subject: `Change particle arrays in a try and commit by swapping buffers`.

### Task 13: Removing particles

**Kind:** mechanical. **Review:** Sonnet, scoped to: each bad index case, the amounts summed in index order, the
dropped particle outside the grid.
**Needs:** Task 12. **Touches:** `particles/remove_particles.h`, `remove_particles_cpu.cc`,
`remove_particles_gpu.cu`, `remove_particles_gpu.h`, `remove_particles_steps.h`, their tests,
`tests/particles/remove_particles_on_gpu.test.cc`, both `CMakeLists.txt`.

**Brief:**
- Design sections: 3.5 (`RemoveReason`, `RemoveParticlesResult`, `RemoveParticles`), 5.7, 5.11 items 1, 3, 7; the
  row `RemoveParticles, MarkerNotNeeded_ReturnsAmounts`.
- Decided: the indexes are a CPU span; the CPU half checks them (out of range, not ascending, already removed:
  `kBadInput`, try unchanged), and on the GPU writes them with one slice into the try's working array, sets the
  marks with one launch and sums the amounts per thread block then in block order on the CPU. A removed particle
  stays readable until the next grouping.
- Tests, unit: `MarkerNotNeeded_ReturnsAmounts`; `IndexOutOfRange_IsBadInput`; `NotAscending_IsBadInput`;
  `AlreadyRemoved_IsBadInput`; `GroupParticlesByBlock, RemovedParticleOutsideGrid_IsDropped`;
  `CheckGrouping, AfterRemove_IsStale`; `RejectTry, AfterRemove_KeepsTheParticle`. `gpu`: same marks and amounts as
  the CPU.

- [ ] Failing tests; implement; gate; commit. Subject: `Remove particles from a try`.

### Task 14: Checking values and summing amounts

**Kind:** device. **Review:** Opus: (1) show the per-block function is the one both processors run and that the
tree sum runs on the CPU for both; (2) which value does `CheckParticles` name when two particles fail? (3) which
plausible wrong implementation do the tests accept (a sum over particles instead of per block, for one)?
**Needs:** Task 12. **Touches:** `particles/particles.h`, `particles_try.h`, `check_particles_cpu.cc`,
`check_particles_gpu.cu`, `.h`, `compute_amounts_cpu.cc`, `compute_amounts_gpu.cu`, `.h`, `particle_checks_steps.h`,
their tests, `tests/particles/amounts_in_every_range_cut.test.cc`, `tests/particles/amounts_on_gpu.test.cc`.

**Brief:**
- Design sections: 3.2 (`CheckParticles`, `ComputeAmounts`, `ComputeAmountsPerBlock`), 3.3
  (`CheckParticles` on a try), 5.11 items 6, 7; 6.3 "Sums"; 11.11; the rows `ComputeAmounts, AnyRangeCut_SameBytes`,
  `CheckParticles, OnTry_SeesNewestValues`.
- Consumes, checked: `math::TreeSum` (`base/tree_sum.h`), `GpuArray`, `ApplyGpuFunction`, `CheckGpuWork`
  (`base/gpu_launch.cuh`), `math::IsFinite`.
- Decided: `ComputeAmounts` on the GPU holds one `GpuArray` of one `ParticleAmounts` per block for the call, reads
  it back and tree-sums on the CPU. `CheckParticles` reports the lowest id; magnitude is the offending value.
- Tests, unit: `ComputeAmounts, ThreeParticlesTwoBlocks_SumsByHand` (expected by hand, exact);
  `ComputeAmounts, PerBlockThenTree_NotOneRunningSum` (values `1e16, 1, -1e16, 1` over four blocks: 0, where a
  running sum gives 1); `ComputeAmountsPerBlock, WrongSpanSize_IsBadInput`; `CheckParticles`:
  `NaNPosition_IsBadInput`, `FractionsOff_IsBadInput`, `IdsNotAscendingInACell_IsBadInput`,
  `OnTry_SeesNewestValues`, `TwoBad_NamesLowestId`. Functional: `AnyRangeCut_SameBytes` (the cloud, ranges of
  1 and 3 blocks and all). `gpu`: same bits as the CPU; digest.

- [ ] Failing tests; implement; gate; commit. Subject: `Check particle values and sum their amounts per block`.

### Task 15: Reading and restoring snapshots

**Kind:** integration. **Review:** Opus: (1) why is the restored order the grouping of `position_when_grouped`
and not of `position`: trace `UngroupedMove_RestoresOrder`; (2) which fields of a restored `Particles` could
differ from the original and are they compared? (3) which plausible wrong implementation do the tests accept?
**Needs:** Task 12, Task 14. **Touches:** `particles/particles_snapshot.h`, `particles_snapshot_cpu.cc`, its tests,
`tests/particles/snapshot_round_trip.test.cc`, `tests/particles/snapshot_on_gpu.test.cc`.

**Brief:**
- Design sections: 3.7 (all but the estimate), 5.9; the five snapshot rows of 7.
- Decided: `BuildParticlesFromSnapshot` sets `grouping_version` 1 and checks the stored order with
  `BuildGroupingReference`'s rule written in the library (not the test code). `ReadSnapshot` inside a try is
  `kTryInProgress`. No `_gpu.cu`: slices of `ReadBuffer` and `WriteBuffer`.
- Tests, unit: `ReadSnapshot, OutputFields_CarryConstantArrays`, `ThisStepVelocity_IsWrongFields`,
  `InsideATry_IsTryInProgress`; `BuildParticlesFromSnapshot, OutputSnapshot_IsBadInput`,
  `OrderNotTheGrouping_IsGroupingStale`, `MaterialsDiffer_IsWrongFields`. Functional: `RoundTrip_SameBytes` (the
  six materials, 50 particles each, `merge_wait_passes` set to 7 on three of them);
  `UngroupedMove_RestoresOrder` (the cloud, `max_move_since_grouped` 0.5, 3 steps of `RunTestStep` with 0.01 s,
  snapshot, rebuild, one more step in both: committed bytes equal). `gpu`: the round trip on the GPU.

- [ ] Failing tests; implement; gate; commit. Subject: `Read a snapshot of the particles and rebuild from one`.

### Task 16: Grouping on the GPU

**Kind:** device. **Review:** Opus: (1) show each thread writes only its own places and that the lowest failing id
is an integer minimum; (2) trace the in-place regroup of one `Mat3` array through its nine 8-byte parts; (3) how
are `group_begin` and `cell_begin` found without a serial loop, and do they equal the CPU's for an empty cell
place at the end of a block? (4) which plausible wrong implementation do the tests accept?
**Needs:** Task 11, Task 12. **Touches:** `particles/group_particles_by_block_gpu.cu`, `.h`,
`group_particles_by_block_steps.h`, `group_particles_by_block_cpu.cc`,
`tests/particles/grouping_steps_in_every_thread_order.test.cc`, `tests/particles/grouping_on_gpu.test.cc`,
`particles/testing/grouping_runner_gpu.cu`, `.h`.

**Brief:**
- Design sections: 5.8, 5.11 items 5, 6; 6.3 "Grouping"; the GPU paragraph of 7.
- Consumes, checked: the shared steps of Task 11; `BuildStableOrder` on GPU buffers; `ApplyEveryThreadOfLaunch`
  (`grid/testing/thread_replay.h`).
- Decided: launches: keys (one thread per particle), two stable orders, one gather per array, one launch that finds
  the first particle of each cell place by a lower-bound search over the sorted keys. Per-item functions for the
  key, the gather of one element and the search live in `group_particles_by_block_steps.h`.
- Tests, functional, label `cpu`: the per-item functions over every thread in launch order and in reverse give the
  CPU grouping's bytes (the cloud; the six). Label `gpu`: `SameBytesAsCpu` on the 5 fixture grids with the cloud,
  `TwiceOnSameInput_SameBytes`, `MillionParticles_TwiceSameBytes` (1,000,000 liquid particles uniform in the
  centre-split grid), `TwiceInOneTry_KeepsCommittedValues`; digests.

- [ ] Failing tests; implement; gate; commit. Subject: `Group particles by cell block on the GPU`.

### Task 17: Who is split and which chunks merge, stated per particle

**Kind:** definition. **Review:** Opus: is each reference independent of the rule (no `cell_begin`, no grouped
order), and is the rule complete (surface cells, classes, the remainder, materials that keep their count)? Which
smaller set does each checker accept?
**Needs:** Task 8. **Touches:** `particles/testing/count_rule_reference.h`, `.cc`, its test,
`particles/CMakeLists.txt`.

**Brief:**
- Design sections: 5.5 Property, 5.6 with its Property (steps 0 to 2 and 5), 5.11.
- Decided: `struct CountRuleParticle { std::uint64_t id; Vec3 position; grid::SizeLevel size_level;
  contact::PlateSide plate_side; std::uint8_t flags; std::uint8_t merge_wait_passes; }` (no snapshot type, so no
  later task is needed); `enum class ChunkOutcome { kMergesNow, kWaits, kLone }`; `struct CountRuleExpectation {
  std::vector<std::uint64_t> split_ids; std::vector<int> generations; std::vector<std::vector<std::uint64_t>>
  chunks; std::vector<ChunkOutcome> outcomes; }` (chunks in the order the rule visits them, each in ascending id).
  `BuildCountRuleReference(const grid::GridTables&, std::span<const CountRuleParticle>, CountRule)`: per particle
  `FindCellAt` and `ToSizeLevel`; chunks from one flat `std::sort` by (cell, class, id), classes ordered by their
  first member's id; outcome: eight members merge now; two to seven merge now exactly when the first member's
  `merge_wait_passes + 1 >= kMaxMergeWaitPasses`, else wait; one is lone. The gap test is not part of it (a formula,
  tested by value). `FindFirstCountRuleDifference(expected, actual)` in words.
- Tests, unit, on hand-built snapshots: the split parent at `P` on the centre-split grid (1 split, 1 generation);
  on cascade grid A (1 split, 2 generations); the eight close on the big-cells grid (1 chunk of 8); eleven
  too-small particles of one class (chunks of 8 and 3: merges now, waits); three with 14 and with 15 passes waited
  (waits, merges now); one alone (lone); two classes of four (no chunk of 8); snow is in neither
  list; each checker names the fault for four wrong answers built by hand (a chunk across two cells, a missing
  split, a chunk of nine, a chunk of three that merges at 14 passes).

- [ ] Reference, checkers and tests; gate; commit. Subject: `State which particles are split and which merge`.

### Task 18: Keeping the count right on the CPU: the split half

**Kind:** numerics. **Review:** Opus: (1) trace the level-0 parent on cascade grid A to its 64 descendants and
their ids; (2) where does a pass that would overflow the waiting room stop, and is the try unchanged? (3) is
`kinetic_energy_change` of a split zero by the formula or by luck of the test values? (4) which plausible wrong
implementation do the tests accept?
**Needs:** Task 5, Task 12, Task 17. **Touches:** `particles/adjust_particle_count.h`,
`adjust_particle_count_cpu.cc`, `adjust_particle_count_steps.h`, their tests,
`tests/particles/count_rule_against_reference.test.cc`, both `CMakeLists.txt`.

**Brief:**
- Design sections: 3.5 (`kMaxMergeWaitPasses`, `AdjustParticleCountResult`, `AdjustParticleCount`), 5.4, 5.6 steps
  0, 1, 3, 4 and "Ids", 5.11; from 7 the rows `ParticleInSmallerCells_IsSplitOnce`,
  `DriftedIntoSmallerCells_IsSplitWithoutResize`, `TooLargeInSurfaceCell_SplitToSmallestInOnePass`,
  `TryGroupedForOtherGrid_IsStale`, `Snow_IsLeftAlone`, `WhoIsSplit_MatchesPerParticleReference`.
- Decided: this task leaves too-small particles untouched except that every particle that matches its cell gets
  `merge_wait_passes` 0; the merge counts of the result stay 0. Per-item functions in
  `adjust_particle_count_steps.h`: `ComputeSplitGenerations(particle_level, cell_level, smallest_level)` and
  `ComputeDescendant(parent, parent_cell_width, generations, descendant_index)` (first generation's child index
  most significant). Places in the waiting list come from an exclusive prefix sum of per-particle counts
  (`ComputeExclusivePrefixSum`); a total above the room or the particle limit writes nothing
  (`kTooManyParticles`). A split parent gets removal mark 255.
- Tests, unit: the rows above with the design's numbers; `ComputeDescendant, TwoGenerations_EqualsSplitOfSplit`;
  `AdjustParticleCount, WaitingRoomTooSmall_IsTooManyParticles` (room 7, one split); `EmptyRange_IsBadInput`.
  Functional: `WhoIsSplit_MatchesPerParticleReference` on 3 seeded grids of 2 and 3 size levels (the cascade grids
  A and B and the centre-split grid) with 200 liquid and 50 snow particles each at seeded levels.

- [ ] Failing tests; implement following `<proc-adjust-particle-count>`; gate; commit.
  Subject: `Split particles that are too large for their cell`.

### Task 19: Keeping the count right on the CPU: the merge half

**Kind:** numerics. **Review:** Opus: (1) trace three too-small particles through passes 1, 15 and 16; (2) trace
eleven particles of two classes in one cell to their chunks and ids; (3) recompute `MergeEnergy_MatchesAnalytic`
by hand; (4) is the result the same when the blocks are cut into ranges, and why? (5) which plausible wrong
implementation do the tests accept?
**Needs:** Task 18. **Touches:** the files of Task 18, `tests/particles/count_rule_in_every_range_cut.test.cc`.

**Brief:**
- Design sections: 5.5, 5.6 steps 2 to 5 and "Particle size follows the cell", 5.11; from 7 the rows
  `EightInBiggerCell_AreMerged`, `EightSmallInLargeCell_MergeInOnePass`,
  `ThreeSmallInLargeCell_MergeAfterSixteenPasses`, `LoneSmallParticle_StaysAndIsCounted`,
  `MergeEnergy_MatchesAnalytic`, `SplitThenMerge_MassAndMomentumExact`, `TwoFarApart_AreNotMerged`,
  `DifferentSides_FormTwoClasses`, `Chunks_MatchFlatSortReference`, `AnyRangeCut_SameBytes`, `AnyBatchCut_SameBytes`,
  `ThreeLevelCascade_KeepsAmounts`; `RejectTry, AfterEveryKindOfChange_ChangesNothing`.
- Decided: one visit per active cell forms the classes and chunks in the cell's grouped order; the per-item
  function `ComputeCellChunks` of the steps header is what Task 20 runs per thread. A cell holds at most
  `group size` candidates, so the visit needs no memory beyond the cell's run. Merged particles get removal mark
  255 on their sources. "One committed pass" in a test is: `BuildTry`, group if stale, `AdjustParticleCount` over
  the ranges, group, `CommitTry`.
- Tests, unit: the rows above with the design's numbers. Functional: `Chunks_MatchFlatSortReference` (the grids
  and counts of Task 18, too-small particles of 2 plate sides and 2 flag values; one range and 1-block ranges);
  `AnyRangeCut_SameBytes` and `AnyBatchCut_SameBytes` (cascade grid A with the surface-split parent, the three
  waiting and the eight close; 17 passes; ranges of `BuildRanges(grid, n)` for n = the block count, 1, 3).

- [ ] Failing tests; implement; gate; commit. Subject: `Merge particles that are too small for their cell`.

### Task 20: Keeping the count right on the GPU

**Kind:** device. **Review:** Opus: (1) show every thread writes only places given by the prefix sum, and that
ids do not depend on thread timing; (2) how much stack does the per-cell visit need and where is that number
written? (3) show the CPU and the GPU form the same chunks for a cell whose particles span two thread blocks;
(4) which plausible wrong implementation do the tests accept?
**Needs:** Task 16, Task 19. **Touches:** `particles/adjust_particle_count_gpu.cu`, `.h`,
`adjust_particle_count_cpu.cc`, `adjust_particle_count_steps.h`, `particle_memory_estimate.cc`,
`tests/particles/count_rule_steps_in_every_thread_order.test.cc`, `tests/particles/count_rule_on_gpu.test.cc`,
`particles/testing/count_rule_runner_gpu.cu`, `.h`.

**Brief:**
- Design sections: 5.6, 6.3 "`AdjustParticleCount`"; the row `OnGpu_SameBytesAsCpu`.
- Decided: launches: counts per particle (splits) and per active cell (merges); one prefix sum each; the writes.
  Working arrays are the storage's (counts and places: 4 B per particle and per cell place, entries
  `"particles: count rule counts"`, added to the estimate and its test in this task). Stack: measure the per-cell
  visit's frame, write it in the header comment and, if it is above the driver's default, add a `kDriverStack`
  entry as `grid/cell_block_range.h` does.
- Tests, functional, label `cpu`: the per-item functions over every thread in launch order and in reverse give the
  CPU's bytes for the 17-pass case of Task 19. Label `gpu`: every case of Tasks 18 and 19 gives the CPU's result
  numbers and snapshot bytes; `TwiceOnSameInput_SameBytes`; digests.

- [ ] Failing tests; implement; gate; commit. Subject: `Split and merge particles to their cell's size on the GPU`.

### Task 21: Which particles a loaded try holds

**Kind:** definition. **Review:** Opus: is the reference independent of the copy's mechanics (no `group_begin`
arithmetic), and is the rule complete (border blocks, removed particles, a block with no particle)? Which smaller
set does the checker accept?
**Needs:** Task 15. **Touches:** `particles/testing/loaded_groups_reference.h`, `.cc`, its test.

**Brief:**
- Design sections: 5.10 with its Property; 3.8; design 4 "In a grid piece".
- Consumes, checked: `GridPiece`, `BuildGridPiece`, `BuildRangeWithBorder` (`grid/grid_piece.h`,
  `grid/cell_block_range.h`).
- Decided: `std::vector<std::vector<std::uint64_t>> BuildLoadedIdsReference(const grid::GridTables& whole,
  std::span<const grid::CellBlockIndex> piece_map, const ParticlesSnapshot& whole_snapshot, std::size_t
  material_index)`: per piece block, the ids of the particles whose `FindCellAt` block is that whole-grid block, in
  snapshot order. `FindFirstLoadedDifference(expected, const ParticleFields& loaded)`. For the copy back:
  `std::vector<std::uint64_t> BuildWrittenBackIdsReference(whole, grid::CellBlockRange range_in_whole_grid,
  whole_snapshot, material_index)`: the ids whose `FindCellAt` block lies in the range, and no id of a border
  block.
- Tests, unit: the six on three blocks with the range `{1, 1}` and its border (written back: id 1 only); the
  checker names the fault for a border block left out, a particle twice, and two blocks swapped.

- [ ] Reference, checker and tests; gate; commit. Subject: `State which particles a loaded piece holds`.

### Task 22: Copying particle groups for a batched run

**Kind:** integration. **Review:** Opus: (1) trace particle `i` of loaded block `b` to its whole-try index and
back; (2) show border blocks are never copied back and why that is right; (3) what does `WriteParticleGroups` read
for a block the whole try wrote in this try and for one it did not? (4) which plausible wrong implementation do
the tests accept?
**Needs:** Task 13, Task 14, Task 21. **Touches:** `particles/particle_groups.h`, `particle_groups_cpu.cc`, its
tests, `tests/particles/particle_groups_in_batches.test.cc`, `tests/particles/particle_groups_on_gpu.test.cc`.

**Brief:**
- Design sections: 3.8, 5.10, 5.11 items 1, 2, 3; design 4 (the piece map of a GPU piece, A131); 6.5; the five
  rows of 7 from `WriteParticleGroups then ReadParticleGroups` to `ParticleGroups, BatchedAndWhole_SameBytes`, and
  `LoadedSet_MatchesFilterReference`.
- Consumes, checked: `GridPiece::grid`, `range`, `range_in_whole_grid`, `block_in_whole_grid`,
  `block_in_whole_grid_buffer`; `ReadBuffer`, `WriteBuffer`, `CopyBuffer` slice forms.
- Decided: for a CPU piece the map is read from the span and `copy_count` is 0; for a GPU piece one `ReadBuffer`
  into the loaded block origin table (`copy_count` 1, `bytes_copied` 4 x blocks). No `_gpu.cu`.
- Tests, unit (the six, range `{1,1}`): `RoundTrip_SameBytes`; `PieceTooFull_IsTooManyParticles`;
  `WholeHasWaitingParticles_IsGroupingStale`; `WholeWrittenForSomeBlocks_CopiesNewestValues`;
  `ReadParticleGroups, RangeNotTheLoadedOne_IsBadInput`, `MassField_IsWrongFields`,
  `RemovedInLoaded_RemovedInWhole`, `EveryRangeRead_CommitSucceeds`, `OneRangeMissing_IsPartlyWritten`. Functional:
  `LoadedSet_MatchesFilterReference` (the cloud on the centre-split grid, every range of `BuildRanges(grid, 3)`
  with its border; after changing every loaded position and reading back, the changed ids of the whole try equal
  `BuildWrittenBackIdsReference`); `BatchedAndWhole_SameBytes` (the cloud, 5 steps of `RunTestStep`'s changes applied
  on the whole
  try and in batches of 1, 2 and 5 blocks). `gpu`: the round trip with a loaded try on the GPU; `copy_count` 1.

- [ ] Failing tests; implement; gate; commit. Subject: `Copy particle groups between a whole try and a loaded try`.

### Task 23: Scenes

**Kind:** integration. **Review:** Sonnet (scene check): each scene's sizes, expected numbers and tolerances as
written here.
**Needs:** Task 15, Task 19, Task 22. **Touches:** `tests/particles/particles_scenes.test.cc`,
`tests/particles/particles_scenes_on_gpu.test.cc`, `tests/particles/CMakeLists.txt`.

**Brief:**
- Decided, each on the CPU and (label `gpu`, compiled) on the GPU within design 6.6:
  1. *Falling cloud:* the cloud, 100 steps of `RunTestStep` with 0.002 s, `max_move_since_grouped` 0.25:
     `ComputeAmounts` mass and rest volume unchanged to the bit; momentum z equals `mass x (-9.81 x 0.2)` plus the
     start within 1e-12 relative; `CheckParticles` passes every step; the run regroups at least once and fewer
     than 100 times.
  2. *Cascade and back:* `ThreeLevelCascade` run as committed passes with a snapshot round trip between each;
     amounts equal before and after every pass within 1e-13 relative.
  3. *Six materials:* 50 particles of each of the six materials on the centre-split grid, 20 steps, one
     `AdjustParticleCount` pass per step: snow and springy solid counts never change; a rejected try in step 10
     leaves the committed snapshot's bytes unchanged.
  4. *Batched equals whole:* scene 1 for 10 steps on the whole try and in batches of 3 blocks: committed bytes
     equal.

- [ ] Tests (they pass on arrival; a failure is a finding for the owning task); gate; commit.
  Subject: `Test particles over whole scenes`.

### Task 24: Comment pass and hand-over

**Kind:** mechanical. **Review:** Sonnet, scoped to: the list of files changed by three or more tasks, the
`docs/gpu-runs.md` row, the as-built list.
**Needs:** every task. **Touches:** files of `src/fluidity/particles/` changed by three or more tasks,
`docs/gpu-runs.md`, `AGENTS.md` (the particle verbs: `SplitParticle` and `MergeParticles` leave the list),
`docs/design/` (the module's design copy).

**Brief:**
- Reread each file changed by three or more tasks (`particles.h`, `particles_try.h`, `particles_try_cpu.cc`,
  `particles_storage.h`, `adjust_particle_count*`, `group_particles_by_block*`) as a C++ programmer new to fluids;
  fix comments only; fold each change into the commit that owns the line. No pass over `testing/` and tests.
- Check the design section by section against the code and write what was delivered differently to the report
  (the coordinator brings the design to as-built from it and from the ledger's "design section to change" lines).
- `docs/gpu-runs.md`: the particles row, "compiled, not run", with the list of `gpu` test programs.
- Run the gate at `--speed full` on the tip.

- [ ] Comment pass; hand-over files; full gate; commit(s). Subject of the hand-over commit:
  `Record particles as compiled for the GPU and not run`.

## Later: deep liquid without particles (needs its own design pass)

Not in the first version (A64). When the cell resize plan's group of that name is built, particles gains the caller
of `RemoveReason::kIntoGridOnlyRegion` and the liquid caller of `AddReason::kRefillRegion`. No task is written yet.
