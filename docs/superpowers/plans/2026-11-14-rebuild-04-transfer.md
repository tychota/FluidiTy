# Rebuild Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the transfer module on the `rebuild` branch: velocity moved from particles to the grid's faces and
back, once per material set, with the same-side weight rule, the two grid snapshots and their filled faces, the
fitted and the exact velocity change kept distinct, and the mixing formula, on the CPU and (compiled, not run)
on the GPU.

**Architecture:** Folder `native/fluidity/transfer/`, namespace `fluidity::transfer`. Five operations
(`ParticlesToGrid`, `SaveGridBeforeForces`, `SaveGridAfterPressure`, `GridToParticles`,
`ComputeCellsFromParticles`), each with one public header and private `_cpu.cc` / `_gpu.cu` files that call the
same shared formula headers. Every per-face sum has one writer that visits its particles in the fixed order of
design 5.2 (face-owned sums, design section 6). Inputs from modules that do not exist yet (solids'
`SolidsAtFaces`, the solves' `velocity_after_pressure`, `was_solved`, `cell_was_solved`, materials' internal force
and shares) are built by hand in this module's tests.

**Tech Stack:** as Foundation (C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2, uv, clang-format and
clang-tidy 23.1 (the versions `pyproject.toml` pins), Docker via colima, Typst). New here: NumPy in the `dev` dependency group, for the one Python
reference piece.

**Spec:** `docs/superpowers/specs/modules/04-transfer-design.md` ("the design"; in the `main` checkout, never
committed) and `docs/design/programme.md` on `rebuild` (sections 6, 7, 8, 8a). Names shared with other modules:
`docs/superpowers/specs/modules/shared-interface.md`. Owner decisions: `owner-answers.md` (A1, A2, A5, A8, A10,
A11, A13 apply here).

## Global Constraints

Foundation's list, unchanged:

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 23.1 clean; clang-tidy 23.1 clean on host sources; every touched file passes the `AGENTS.md` checklist; new public
  names listed in the commit message with a one-line reason; tests arrive with the code. A commit must pass every
  part of the gate that exists at that commit.
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure patterns apply to every task (`AGENTS.md`
  "Structure"; `specs/modules/plan-decisions.md` "Foundation after the final review"). Two halves: each public
  stage lives in its `<name>_cpu.cc` (checks, `RunOn` dispatch, the CPU loop, the GPU half called under
  `#if defined(FLUIDITY_HAS_GPU)`); the GPU half `<name>_gpu.cu` is declared in the private `<name>_gpu.h` (plain
  pointers, the on-processor views and counts; no CUDA header, no CPU-only header), returns `RunOnCallStatus`, which
  the CPU half maps with `ToBufferError` and then into `TransferFailure` through one private mapping
  (`kRunOnCallFailed` for every failed GPU call, as for the builders), and launches, waits and holds temporary GPU
  memory only through `base/gpu_launch.cuh` (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`).
  Every GPU function is a thin grid-stride loop over `FLUIDITY_CPU_AND_GPU` per-item functions of a shared
  `<name>_steps.h` (which call the shared formula headers), plus at most an integer atomic of the value they
  return, and `cpu` tests run those functions for every thread block and thread, in launch order and in reverse,
  against the CPU loop: the main evidence until the hardware run. Working buffers: per-step memory is the
  `TransferBuffers` and the arrays the builders return (below); every stage has one form, which takes the
  `TransferBuffers` (A31), and buffers on another processor than the stage's arrays are a `Requires:`
  precondition asserted in debug builds, not an error value (A30); temporary GPU memory of one call only through
  `GpuArray`. Errors: a failed check writes nothing; after `kRunOnCallFailed` the outputs are unspecified (the
  output's `step_try` is 0) and the run stops. Exactness: kinds, reasons, counts, keys and failures equal on CPU and
  GPU; floating-point results within the tolerances below unless a header promises the same bits unless NaN
  (`cells_from_particles.h`, Task 16). Tests link `fluidity_base_test_support` and `fluidity_transfer_test_scenes`
  (made with `fluidity_add_test_library`, Task 5) in `LINKS`; every `gpu` test passes its main outputs to
  `WriteResultDigest`. Labels: `cpu` everywhere (the container's `cuda` step runs them too), `no_gpu` only for
  behaviour on a machine without a GPU, `gpu` only for tests that need hardware. Every target is created through
  the helpers of `cmake/fluidity_targets.cmake` and every source file is listed in a CMake file.
- **Amended 2026-11-15 (A58; reconcile item 31; `plan-decisions.md` "Owner answers A40–A64 applied" items 3, 11
  and the item-3 correction):** one reserved block per processor per run (Foundation Tasks 17 to 19). Every builder
  (`BuildTransferBuffers`, `BuildSolidsAtFaces`, `BuildGridFromParticles`, `BuildGridSnapshot`,
  `BuildCellsFromParticles`) takes `RunMemory& run_memory` as its last parameter (every other parameter stays),
  the block of `run_on`; `run_on == run_memory.run_on()` is a `Requires:` checked by a debug assertion (A30); every
  buffer is a piece (`BuildBuffer(run_on, size, run_memory)`, the grid's `Build...Values(..., run_memory)`); a
  failed builder leaves its pieces in the block. `TransferError::kOutOfMemory` is appended as the **last** value
  (twelve values), returned only by the builders, for `BufferError::kOutOfMemory` and `kRunMemoryTooSmall` and for
  the grid's `kOutOfMemory`, with `magnitude` = bytes asked (replaces the plan decision that mapped them to
  `kRunOnCallFailed`). Every estimate row stands for whole buffers (`copies` = their number, each `count x
  bytes_each`), so the composite rows of Task 4 are split into one row per array; `ComputeRunMemoryBytes` of the
  rows then equals the bytes the builders hand out for any room (each row rounds as its piece does). Tests build
  a `RunMemory` of the estimate first.
- **Amended 2026-11-15 (A64; item 1):** particles everywhere in the liquid. `FaceValueKind::kFromGridOnlyRegion`
  and every use of it (the source rule of Tasks 11 and 12, the test
  `SaveGridAfterPressure, GridOnlyRegionFace_IsASourceAndKeepsItsKind`) are not in the first version: they move to
  the group "Deep liquid without particles (later; needs review)" at the end of this plan, owned by the cell
  resize plan's group of that name. `FaceValueKind` has four values in the first version. `GridOnlyReason` is
  unrelated (why a particle takes the grid velocity only) and stays.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and the
  release presets, and the CUDA build compiled in the container (its `cuda` step fails, naming the file, for a
  tracked `.cc` or `.cu` file that no build compiles). clang-format and clang-tidy are the pinned 23.1.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages carry no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no session link); the owner is the
  author. After committing, write the commit's note with `uv run python tools/add_session_note.py --session
  <session link> --models <implementer's model id>,claude-fable-5-2`
  (a git note in `refs/notes/claude-session`; owner answer A55, 2026-11-15).
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

Transfer's own:

- **Design section 3 is the contract.** Spell every name exactly as it is there. The stage names
  `ParticlesToGrid`, `GridToParticles`, `SaveGridBeforeForces`, `SaveGridAfterPressure`, `MixParticleVelocity`
  are DECIDED in the naming review; if the rules check reports `verb` on one, add it to
  `tools/check_rules_allowed_names.txt` with the reason `decided stage name (naming review; transfer design 9)`.
- **Material sets (A8, A10; design 5.8).** Every stage and `ComputeCellsFromParticles` takes
  `material_is_counted`: host memory, exactly one entry per `material_index` of the `ParticlesTry`, each 0 or 1,
  at least one 1; anything else is `kBadInput` with `magnitude` = the first bad `material_index` (a wrong length
  reports the length). A call reads particles of counted materials only: no weight, side, mass or check of an
  uncounted particle is ever computed. Every output records the set's `MaterialSetKey`; an input of another set
  is `kWrongMaterialSet`. Results do not depend on the other sets or on the order sets are run.
- **Order of every face sum (design 5.2, decision 1).** For one face: for each source cell block in increasing
  `CellBlockIndex` (blocks of `grid::FindCellsNearFace`'s cells): `partial = 0`; for each counted material in
  ascending `material_index`: for each of that block's listed cells in ascending `CellPlaceIndex`: for each
  particle of that material in that cell in grouped (ascending id) order: `partial += contribution`. Then
  `sum += partial`, `sum` starting at `0.0`. Cell sums (`ComputeCellsFromParticles`): one running sum per cell
  from `0.0`, counted materials ascending, then the cell's particles in grouped order.
- **Constants (design 3):** `kMaxSidesAtFace = 2`, `kFillRadiiInCells{2.0, 3.0, 4.0}` tried in order,
  `kFillPivotTolerance = 1e-10` (ratio to the largest pivot of the column-pivoted QR), spread pivot floor
  `grid::kMinPivotShare` (1e-12 of the trace; never declared again here), `TransferSettings::min_weight_sum_at_face`
  default 0, no unit. No `0.99`, no "energy" anywhere in the module.
- **Tolerances (design 7).** `eps = std::numeric_limits<double>::epsilon()`. `M` = sum of `|contribution|` of the
  quantity summed, `U` = largest speed in the test, `h` = cell width. CPU tests: as each task states. GPU against
  CPU (all `gpu` tests): face sums `1e-13 M`; face and particle velocities `1e-12 U + 1e-12 |value|`; fitted and
  exact change `1e-10 U / h + 1e-10 |value|`; filled velocities `1e-9 U`; kinds, reasons, counts, keys and the
  reported failure exactly. Repeat-run tests: identical bytes.
- **Failure contract (design 3, last paragraph).** Inputs are read-only. A failed stage sets its output's
  `step_try` to 0, so any later stage given it returns `kWrongStageOrder`. The failure reported is the first in
  (pass, then stable particle id or `FacePlaceIndex`) order; on the GPU it is chosen with an integer minimum on
  that key, never by thread timing.
- **Ranges.** A stage writes only values stored in `inputs.blocks`. It reads particles through
  `particles::BuildFieldsToRead(particles_try, m, inputs.grid.all_blocks())` (plan decision: all loaded blocks,
  which in a batched piece are the range and its border in the whole grid's order), so the border of design 6
  is read without a second list. Output arrays are bound with `grid::ApplyGridVersion` only when their
  `grid_version()` differs from `inputs.grid.version()` (rebinding would make earlier ranges' values unspecified).
- **Grouping.** Transfer calls `particles::CheckGrouping(particles_try, grid, particles::GroupingSettings{})`
  (`max_move_since_grouped = 0`) and maps any failure to `kParticlesNotGrouped`. Plan decision; see the hand-over.
- **Plan decisions beyond design section 3** (the design leaves who owns working buffers and how arrays are built
  open). Declared in Task 4, used everywhere:

  ```cpp
  // transfer_buffers.h (CPU only). Working buffers on run_on, reused by every set and stage in turn. Move-only.
  class TransferBuffers {
   public:
    RunOn run_on() const;
    std::size_t max_particle_count() const;
    std::size_t material_count() const;
  };
  // max_particle_count: all materials together; material_count: the scene's materials (one offset each; reconcile
  // item 27). Holds 3 x grid::NearbyFaceSums per particle, the per-material offsets, and the status record
  // (failure key, six reason counts, two face counts). Zero for either count is kBadInput.
  // Amended 2026-11-15 (A58): every builder takes run_memory last; its buffers are pieces of it.
  std::expected<TransferBuffers, TransferFailure> BuildTransferBuffers(RunOn run_on, std::size_t max_particle_count,
                                                                       std::size_t material_count,
                                                                       RunMemory& run_memory);
  // Builders (the step calls them once per set at max_cell_blocks):
  std::expected<SolidsAtFaces, TransferFailure> BuildSolidsAtFaces(RunOn run_on, std::uint32_t max_cell_blocks,
                                                                   RunMemory& run_memory);
  std::expected<GridFromParticles, TransferFailure> BuildGridFromParticles(RunOn run_on, std::uint32_t max_cell_blocks,
                                                                           RunMemory& run_memory);
  enum class FilledFromSolved : std::uint8_t { kNotKept, kKept };  // kNotKept: a set no solve writes.
  std::expected<GridSnapshot, TransferFailure> BuildGridSnapshot(RunOn run_on, std::uint32_t max_cell_blocks,
                                                                GridTime time, FilledFromSolved filled_from_solved,
                                                                RunMemory& run_memory);
  std::expected<CellsFromParticles, TransferFailure> BuildCellsFromParticles(RunOn run_on,
      std::uint32_t max_cell_blocks, std::uint32_t ingredient_count, RunMemory& run_memory);
  // Views for shared formulas (on_processor_views.h, formula) and their conversions (CPU only):
  SolidsAtFacesOnProcessor ToSolidsAtFacesOnProcessor(const SolidsAtFaces& solids);
  GridSnapshotOnProcessor ToGridSnapshotOnProcessor(const GridSnapshot& snapshot);
  ```

  **Amended 2026-11-14 (A30, A31, reconcile):** each stage has one form, with `TransferBuffers& transfer_buffers` as
  its last parameter; it allocates nothing (design section 3). Tests and tools build a `TransferBuffers` locally
  (`test::BuildTransferBuffersFor`, Task 5). A particle count above `max_particle_count()` or a material count
  above `material_count()` is `kBadInput` (magnitude = the count); buffers on another processor than the arrays
  are a `Requires:` precondition (debug assertion), not an error. Its exact bytes are the estimate's entries
  (`nearby face sums buffers`, `material offsets`, `status record`, Task 4), the one definition, so no separate
  `ComputeTransferBuffersSize` is added. An empty
  `filled_from_solved` is a default-constructed `grid::FaceValues` (`max_cell_blocks() == 0`).
- **Memory figures (plan decision, differs from design 5.8).** Rows use `sizeof` of the stored element, so the
  estimate cannot drift from the buffers. With `sizeof(contact::PlateSide) == 8`, `sizeof(SidesAtFace) == 20`,
  `sizeof(FilledVelocity) == 24` (`static_assert`ed in Task 4): per face `GridFromParticles` 61 B, snapshot
  before forces 67 B, after pressure 115 B (67 B without `filled_from_solved`); per cell `24 + 8 I` B. Per extra
  set and cell block: 59,856 + 512 I bytes (48,336 + 512 I for a set no solve writes). The design's 48,576 and
  40,416 count packed bytes; the hand-over reports the difference.

## Before you start

On `rebuild` already: Foundation (all of `base/` listed in `shared-interface.md` section (a): `Vec3`, `Mat3`,
`math::*` including `TreeSum`, `Buffer`, `PerParticleVelocity`, `memory_count.h`, `RunOn`, `CheckCanRunOn`), the
gate, the method paper with `figure_style.typ` and `pseudocode.typ`; then the contact, grid and particles modules
with their section 3 names (`contact::PlateSide`, `SolidSide`, `SolidVelocitySample`,
`ComputeSolidVelocitySample`, `FindSolidSide`, `Solid`, `ToSolid`; `grid::StaggeredGrid`, `GridTables`,
`FaceValues`, `CellValues`, `ApplyGridVersion`, `ReadValues`, `WriteValues`, `FindNearbyFaces`,
`ComputeNearbyFaceWeights`, both `ComputeFaceWeight`, `NearbyFaceSums`, `ToNearbyFaceSums`, `FindCellsNearFace`,
`BuildFaceSearch`, `FindNextFace`, `kMinPivotShare`, `BorderWidth`; `particles::ParticlesTry`,
`BuildFieldsToRead`, `CheckGrouping`, `GroupParticlesByBlock`, `AddParticles`, `kCarriesFittedChange`,
`kHasOwnMotion`). Read how contact dispatches one operation to its CPU and GPU files
(`native/fluidity/contact/move_particles.h`, `move_particles_cpu.cc`, `move_particles_gpu.h`,
`move_particles_gpu.cu`, `move_particles_steps.h`) and copy that shape for every operation here: the private GPU
header, how a CPU-only build answers `RunOn::kGpu`. **Amended 2026-11-14 (Foundation as built):** Foundation as built
provides `base/run_on_call.h` (`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, the sums
(`SumBuffers`) and the stable order and grouping (`OrderBuffers`), each in one form taking its buffers, `base/test_support.h` (library
`fluidity_base_test_support`) and `fluidity_add_test_library`; read `AGENTS.md` "Structure" (the two halves, the
private GPU headers, working buffers) before the first task.

Proof: from the worktree `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, run
`uv run python tools/check_commit.py`. Expected: seven `PASS` lines (format, rules, cpu debug, cpu release, tidy,
paper, cuda). If any fails, stop: transfer does not start
on a red branch.

## Review Focus

1. **A particle exactly on a cell's low side or on a face plane.** The face-side weight
   (`grid::ComputeFaceWeight(tables, face, position, cell, sums)`) must equal the particle-side weight bit for bit,
   or mass appears or vanishes. Pinned in Task 9: `ParticlesToGrid, ParticleOnCellBoundary_ConservesMass`.
2. **An uncounted material holding bad values** (a third plate side, values of 1e30) must not even be read,
   neither by the checks of pass 1 nor by the face visit. Pinned in Task 17:
   `MaterialSets, ParticleOfSetANeverAddsToSetB`.
3. **Rank loss at every radius with ties.** Coplanar sources fall to `kNearestSource`; two sources at the same
   distance go to the lower `FacePlaceIndex`, whatever the search order. Pinned in Task 11:
   `SaveGridBeforeForces, CoplanarSources_CopiesNearest` (its two nearest sources are equidistant).
4. **A fill whose sources lie in another range.** The snapshot stages read source faces up to 4 cell widths
   away; a range cut through the sources must give the same bytes. Pinned in Task 11:
   `SaveGridBeforeForces, SplitIntoTwoRanges_SameBytes`.
5. **A stage handed the output of a failed or older stage.** `step_try == 0` or a different `step_try` must give
   `kWrongStageOrder`, never stale values. Pinned in Task 14: `GridToParticles, AfterFailedSnapshot_IsWrongStageOrder`.

---

### Task 1: Method paper section, text, equations and pseudocode

**Kind:** mechanical

Load the `typesetting-with-typst` and `typesetting-documents` skills first.

**Files:**
- Modify: `docs/method-paper/fluidity-method.typ`, `docs/glossary.md`

**Interfaces:**
- Produces labels (design 8): `<sec-transfer>`, `<sec-same-side-weights>`, `<eq:same-side-weights>`,
  `<sec-particles-to-grid>`, `<eq:face-sums>`, `<proc-particles-to-grid>`, `<sec-grid-snapshots>`,
  `<eq:own-motion>`, `<sec-filled-faces>`, `<proc-fill-face>`, `<sec-grid-to-particles>`, `<eq:fit>`,
  `<eq:exact-change>`, `<proc-grid-to-particles>`, `<sec-mix>`, `<eq:mix>`, `<sec-fixed-order>`,
  `<sec-material-sets>`. Every `Read:` line of later tasks cites one of these.

- [ ] **Step 1: Write `= Moving velocity between particles and grid <sec-transfer>`** after the particles
  section, one subsection per label above, from design 5.1 to 5.8, in the decided vocabulary (no `PIC`, `APIC`,
  `FLIP`, `ghost`, `donor`, `stencil`). Content the reader must find:
  - `<sec-same-side-weights>`: the three sums `S`, `G`, `mu` and the three per-face values of 5.1; why they hold
    for incomplete and mixed-size neighbourhoods; `S <= 0` is an error.
  - `<sec-particles-to-grid>`: the four sums of 5.2 step 3 (`<eq:face-sums>`) and the conservation identities
    (`sum n = 1`, `sum n rc = 0`, `sum dn = 0`) with the three-line algebra.
  - `<sec-grid-snapshots>`: why the first snapshot is saved before forces; `<eq:own-motion>`; the free-fall
    argument of 5.3 with `(1 - own_motion_kept) g step_duration`.
  - `<sec-grid-to-particles>`: `<eq:fit>` (spread and fitted row, the `mu` correction) and `<eq:exact-change>`;
    state that the exact change equals a linear field's gradient only where `mu` is constant.
  - `<sec-mix>`: `<eq:mix>`, the two shares as independent effective values; how `main`'s `0.99 e` and
    `0.99 e^2` map onto them.
  - `<sec-fixed-order>`: the order contract of the Global Constraints, with a worked example of two block
    partials.
  - `<sec-material-sets>`: one grid velocity per set; per-set order; gas markers out of the liquid's sums.
  - Notation table rows: `C` fitted_velocity_change, `L` exact_velocity_change, `D` weighted spread, `alpha`
    own_motion_kept, `beta alpha` own_motion_in_travel. Cite [1] Jiang 2015, [2] Ding 2020, [3] Hu 2018,
    [4] Fei 2021, [18] Stomakhin 2014 from `references.typ` (numbers as there).
- [ ] **Step 2: Pseudocode** with `procedure(...)` from `pseudocode.typ`. Step numbers are the contract for the
  `// Step N:` comments of later tasks:
  - `<proc-particles-to-grid>`: 1 check inputs; 2 per particle and axis, the three sums; 3 per face, the
    ordered visit and the four sums; 4 value kind and velocity (design 5.2).
  - `<proc-fill-face>`: 1 source faces; 2 fit at radii 2, 3, 4; 3 nearest source, solid velocity, or failure;
    4 solid correction (design 5.4).
  - `<proc-grid-to-particles>`: 1 check inputs; 2 nearby faces and weights over all faces; 3 read each face's
    value directly or filled, in both snapshots (5.4, 5.5); 4 weighted means, spread, exact change; 5 solve
    the spread (Cholesky, cell-width units); 6 `mu` correction of both velocities; 7 grid-only reason and own
    motion (design 5.6).
- [ ] **Step 3: Glossary.** Add rows for the words this section introduces that are not yet in
  `docs/glossary.md`: material set, source face, filled face (fill), own motion, weight sum, sides reaching,
  grid-only reason, before forces, after pressure. "Never say" column from design 9 (ghost, donor,
  continuation, deposit, gather, stencil, energy blend).
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.
  Expected: exit 0, no warning. Read the new pages. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Describe how velocity moves between particles and grid`.

### Task 2: Method paper figures for transfer

**Kind:** mechanical

Load `drawing-with-cetz` and use `figure_style.typ` (and the grid module's `cell-grid` helper).

**Files:**
- Create: `docs/method-paper/figures/same_side_weights.typ`, `grid_snapshots.typ`, `filled_face.typ`,
  `fixed_order.typ`
- Modify: `docs/method-paper/fluidity-method.typ`

**Interfaces:**
- Produces `<fig-same-side-weights>`, `<fig-grid-snapshots>`, `<fig-filled-face>`, `<fig-fixed-order>`.

- [ ] **Step 1: Draw** (design 8): one particle beside a plate, its nearby faces on the three face sets, faces
  across the plate greyed, `mu` as an arrow; a time line of one step with the two snapshots and a falling
  particle, gravity counted once; one face read across a plate, its sources inside radius 2, the solid normal
  correction; one face, its source blocks, cells and particles numbered in visit order, two block partials
  joined. Each figure is referenced from its subsection and its caption states the one thing to see.
- [ ] **Step 2: Compile and look.** Fix overlaps, clipped labels, inconsistent sizes. Run the gate.
- [ ] **Step 3: Commit.** Subject: `Draw the transfer figures`.

### Task 3: Errors, settings and material sets

**Kind:** mechanical

**Amended 2026-11-15 (reconcile item 31):** `TransferError` gains `kOutOfMemory` as its twelfth and last value.

**Files:**
- Create in `native/fluidity/transfer/`: `transfer_error.h`, `transfer_message.h`, `transfer_message.cc`,
  `transfer_settings.h`, `material_set.h`, `material_set.cc`, `transfer_message_test.cc`, `material_set_test.cc`,
  `CMakeLists.txt` (or the per-module list the earlier modules use)
- Modify: the CMake file that lists module folders

**Interfaces:**
- Produces design 3 exactly: `TransferError` (all twelve values, `kOutOfMemory` last), `TransferFailure`, `ToMessage`, `ToName`,
  `kMaxSidesAtFace`, `kFillRadiiInCells`, `kFillPivotTolerance`, `TransferSettings`, `MaterialSetKey`,
  `ToMaterialSetKey`, `CheckMaterialSets`.
- Plan decision: `TransferFailure` members have defaults (`error = kNone`, `particle_id = 0`,
  `face = {grid::kNoIndex}`, `axis = Axis::kX`, `magnitude = 0`).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(TransferError, EveryValue_HasNameAndMessage)   // design 7: ToName lower snake case without k, all twelve
                                                     // unique ("wrong_material_set"); ToMessage never empty;
                                                     // static_assert(kOutOfMemory == 11), "out_of_memory"
  TEST(ToMaterialSetKey, LiquidAndGas_KeyIsLowestCountedAndCount)  // {0,1,1,0}, 4 -> {1, 2}
  TEST(ToMaterialSetKey, LengthNotMaterialCount_IsBadInput)        // 3 entries, 4 materials; magnitude 3
  TEST(ToMaterialSetKey, EntryTwo_IsBadInputForThatMaterial)       // {1,2,0}; magnitude 1
  TEST(ToMaterialSetKey, AllZero_IsBadInput)
  TEST(CheckMaterialSets, DisjointAndComplete_Succeeds)            // {1,0,0},{0,1,1}
  TEST(CheckMaterialSets, MaterialInTwoSets_IsBadInputForLowest)   // {1,1,0},{0,1,1} -> magnitude 1
  TEST(CheckMaterialSets, MaterialInNoSet_IsBadInputForLowest)     // {1,0,0,0},{0,0,1,0} -> magnitude 1
  ```
- [ ] **Step 2: Run** the build. Expected: compile error, headers missing.
- [ ] **Step 3: Implement.** `transfer_error.h` and `transfer_settings.h` are shared (`FLUIDITY_CPU_AND_GPU`,
  C++20 subset); `transfer_message.h` and `material_set.h` include `base/cpu_only.h` first. Each message says
  what the user can do. `Read:` lines cite `<sec-material-sets>`.
- [ ] **Step 4: Run** build, tests, format, rules; then the gate.
- [ ] **Step 5: Commit.** Subject: `Add transfer errors and material sets`. New names: the list above.

### Task 4: The arrays transfer reads and writes, and their memory estimate entries

**Amended 2026-11-14 (reconcile):** test file named `transfer_memory_estimate_test.cc` (A23).

**Amended 2026-11-14 (A31, reconcile):** `TransferBuffers` (was `TransferScratch`) is built with `material_count` (reconcile item 27); test names say "buffers".

**Amended 2026-11-15 (A58; reconcile item 31):** builders take `RunMemory& run_memory` last; composite estimate rows are split into one row per array (byte totals per block unchanged); out of memory is `kOutOfMemory`; new tests `BuildTransferBuffers, IntoRunMemory_HandsOutExactlyTheEstimate` and `BuildGridSnapshot, RunMemoryOnePieceShort_IsOutOfMemory`.

**Kind:** mechanical

**Files:**
- Create: `solids_at_faces.h`, `grid_from_particles.h`, `grid_snapshot.h` (types only; functions in Task 11),
  `cells_from_particles.h` (types only; function in Task 16), `on_processor_views.h`, `transfer_buffers.h`,
  `transfer_arrays.cc`, `transfer_memory_estimate.h`, `transfer_memory_estimate.cc`, `transfer_arrays_test.cc`,
  `transfer_memory_estimate_test.cc`

**Interfaces:**
- Consumes: `grid::BuildFaceValues`, `BuildCellValues`, `BuildBuffer`, `contact::PlateSide`,
  `contact::SolidVelocitySample`, `MemoryEstimateEntry`.
- Produces design 3: `SolidsAtFaces`, `FaceValueKind`, `SidesAtFace`, `GridFromParticles`, `GridTime`, `FillKind`,
  `FilledVelocity`, `GridSnapshot`, `CellsFromParticles`, `SolidsAtFacesOnProcessor`, `GridSnapshotOnProcessor`,
  `BuildTransferMemoryEstimate`. Plus the plan decisions of the Global Constraints: `TransferBuffers`,
  `BuildTransferBuffers`, `BuildSolidsAtFaces`, `BuildGridFromParticles`, `FilledFromSolved`, `BuildGridSnapshot`,
  `BuildCellsFromParticles`, `ToSolidsAtFacesOnProcessor`, `ToGridSnapshotOnProcessor`.
- Plan decision, members of the two views (pointers on the arrays' `run_on()`, valid for one grid version):
  `SolidsAtFacesOnProcessor { const std::uint8_t* outside_solid; const contact::PlateSide* side;
  const contact::SolidVelocitySample* solid_velocity; double time; }`;
  `GridSnapshotOnProcessor { GridTime time; const double* velocity; const FaceValueKind* value_kind;
  const std::uint8_t* is_source; const contact::PlateSide* side; const std::uint8_t* outside_solid;
  const std::array<FilledVelocity, kMaxSidesAtFace>* filled; const std::array<FilledVelocity, kMaxSidesAtFace>*
  filled_from_solved; }` (last one null when empty).

- [ ] **Step 0: Check `grid::FaceValues` is default-constructible with `max_cell_blocks() == 0`.** If it is not,
  stop and ask the owner: "empty" in `GridSnapshot::filled_from_solved` needs it, and grid's files are not
  changed by this plan.
- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(TransferArrays, ElementSizes_AreThePlannedOnes)  // static_assert and EXPECT: PlateSide 8, SidesAtFace 20,
                                                        // FilledVelocity 24 (GridAtParticle 200: Task 14)
  TEST(BuildGridFromParticles, OnCpu_BoundToNoGridWithRoom)        // 8 blocks: max_cell_blocks 8, version 0,
                                                                   // step_try 0
  TEST(BuildGridSnapshot, NotKept_LeavesFilledFromSolvedEmpty)     // max_cell_blocks() == 0
  TEST(BuildGridSnapshot, Kept_AllocatesFilledFromSolved)
  TEST(BuildCellsFromParticles, ThreeIngredients_BufferIsThreePerCellPlace)  // 8 blocks -> 1536 doubles
  TEST(BuildTransferBuffers, OnCpu_HasTheAskedCounts)              // max_particle_count and material_count
  TEST(BuildTransferBuffers, ZeroMaterials_IsBadInput)
  TEST(BuildGridFromParticles, ZeroBlocks_IsBadInput)
  TEST(BuildTransferMemoryEstimate, OneSetWithSolve_PerBlockIs59856Plus512PerIngredient)
      // ComputeMemoryBytes over the rows with MemoryCounts{cell_blocks 1, cell_places 64, face_places 240,
      // particles 0}, ingredient_count 3, material_count 2, sets 1, sets_without_solve 0:
      // 59856 + 512 * 3 + the given-count rows
  TEST(BuildTransferMemoryEstimate, SetWithoutSolve_LeavesOutFilledFromSolved)  // 2 sets, 1 without solve:
      // face-place bytes per block == 2 * (61 + 67 + 67) * 240 + 48 * 240
  TEST(BuildTransferMemoryEstimate, NearbyFaceSumsBuffersEntry_Is168BytesPerParticle)
  TEST(BuildTransferBuffers, IntoRunMemory_HandsOutExactlyTheEstimate)  // amended 2026-11-15 (A58): 2 sets (one
      // without solve), ingredient_count 3, material_count 2, 1,000 particles, 1,000 blocks: a CPU RunMemory of
      // ComputeRunMemoryBytes(BuildTransferMemoryEstimate(...), counts, kRunOn); BuildTransferBuffers and, per set,
      // BuildSolidsAtFaces once, BuildGridFromParticles, both BuildGridSnapshot, BuildCellsFromParticles into it:
      // handed_out_bytes() == reserved_bytes(), exact
  TEST(BuildGridSnapshot, RunMemoryOnePieceShort_IsOutOfMemory)  // (A58; GTEST_SKIP in debug builds) a 256 B
      // block, 8 blocks asked: kOutOfMemory, magnitude = the first piece's bytes
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** Rows (all names start `transfer: `; bytes from `sizeof`; `copies` as stated;
  amended 2026-11-15, A58: each row below that sums several arrays is written as one row per array, named
  `transfer: <row>: <member>` in member order, e.g. `face sums from particles: mass` (face place, 8, sets); the
  per-block totals and the tests above are unchanged):
  face sums from particles (face place, 61, sets, `kOneStepTry`); snapshot before forces (face place, 67, sets);
  snapshot after pressure (face place, 67, sets); filled from solved sources (face place, 48,
  `sets - sets_without_solve`); cell sums from particles (cell place, `24 + 8 * ingredient_count`, sets);
  nearby face sums buffers (particle, 168, 1, `kWholeRun`); material offsets (given count `material_count`, 8,
  1); status record (given count 1, `sizeof` of the record, 1). `GridAtParticle` has no row here: the step's
  own rows count it (step design 6.3). Amended 2026-11-15 (reconcile item 31): a builder that gets
  `kOutOfMemory` (or `kRunMemoryTooSmall`) from the grid or `base/` returns `kOutOfMemory` with `magnitude` = the
  bytes asked; `kRunOnCallFailed` stays `kRunOnCallFailed` (the earlier plan decision mapping both to
  `kRunOnCallFailed` is replaced). File headers cite `<sec-transfer>` and give the bytes
  table.
- [ ] **Step 4: Run** build, tests, format, rules, gate.
- [ ] **Step 5: Commit.** Subject: `Add the arrays transfer reads and writes`. New names: every name of the
  Interfaces list, each with its one-line reason.

### Task 5: Test scenes

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the scenes library is made with `fluidity_add_test_library` and links `fluidity_base_test_support`.

**Amended 2026-11-14 (A31):** adds `BuildTransferBuffersFor`, so every test calls the stages' one form with locally built buffers.

**Files:**
- Create: `native/fluidity/transfer/transfer_test_scenes.h`, `transfer_test_scenes.cc`,
  `transfer_test_scenes_test.cc`
- Modify: the transfer `CMakeLists.txt`: `fluidity_add_test_library(fluidity_transfer_test_scenes SOURCES
  transfer_test_scenes.cc LINKS fluidity_base_test_support)`, named in `LINKS` by every transfer test target from
  here on; never linked by `fluidity`

**Interfaces:**
- Consumes: grid, particles and contact section 3 names.
- Produces, in `namespace fluidity::transfer::test` (every later task uses these exact fixtures):
  - `kCellWidth = 0.1` (m). `BuildUniformGrid(RunOn)`: one level, origin `{0,0,0}`, biggest width 0.1, active
    cells with coordinates 0..11 on each axis (12 cells, 3 x 3 x 3 blocks). Interior with complete
    neighbourhoods: positions in `[0.3, 0.9)` on each axis.
  - `BuildSizeStepGrid(RunOn)`: biggest width 0.2, two levels; level-0 cells for `x` in `[0, 0.8)`, level-1
    cells (0.1 m) for `x` in `[0.8, 1.6)`; `y`, `z` in `[0, 0.8)`.
  - `BuildSlabGrid(RunOn)`: one level, width 0.1, `x`, `y` in `0..11`, `z` coordinate 5 only (one layer).
  - Solids: `kPlateId = 7`; `PlateAtX(double x)` = `contact::Plate{.center = {x, 0.6, 0.6}, .normal = {1,0,0},
    .half_thickness = 0.01}`; `PlateAtY(double y)` the same along `{0,1,0}`;
    `ObliqueFloor()` = `contact::FlatSolid{.point_on_surface = {0.6, 0.6, 0.25}, .normal_out_of_solid =
    {0.6, 0, 0.8}}`.
  - `BuildSolidsAtFaces(const grid::StaggeredGrid&, std::span<const contact::Solid>, double elapsed, RunOn)`:
    for every face, `ComputeSolidVelocitySample(..., elapsed, face position, reach = 2 * cell width of the
    face)`; `outside_solid = 1` unless some solid's `ComputeDistanceToSolid` is `<= 0`; `side` from
    `FindSolidSide` of the plate (`{kPlateId, side}`, or `{0, kNoPlate}`). Bound to the grid; `time = elapsed`.
  - `ParticleSpec { std::uint64_t id; Vec3 position; Vec3 velocity; Mat3 fitted_velocity_change; double mass;
    double rest_volume; contact::PlateSide plate_side; std::uint8_t flags; std::uint32_t material_index; }` and
    `BuildParticlesTry(RunOn, const grid::StaggeredGrid&, std::span<const ParticleSpec>, std::uint32_t
    material_count)`: builds `Particles`, a try, adds, groups. Defaults: `rest_volume = mass / 1000`,
    `flags = kCarriesFittedChange | kHasOwnMotion`, side from the scene's plate if any.
  - `TransferBuffers BuildTransferBuffersFor(const particles::ParticlesTry&)`: `BuildTransferBuffers` on the
    try's processor at its total particle limit and `material_count()` (the test fails on an error); every
    stage call in a test passes one.
  - `BuildCloud(std::uint64_t seed, int count, Vec3 low, Vec3 high)`: positions from `std::mt19937_64(seed)`,
    each draw `x` turned into `(x >> 11) * 0x1.0p-53` (portable; no `std::uniform_real_distribution`), ids
    `1..count`, mass 0.125e-3 kg.
  - `BuildGridFromParticlesByFunction(grid, solids, function)`: writes a hand-made `GridFromParticles` (velocity,
    value kind, `sides_reaching`) from a function of (face position, axis), for snapshot tests.
  - `ReadAllBytes(const GridFromParticles&)`, the same for `GridSnapshot`, `CellsFromParticles`,
    `Buffer<GridAtParticle>`: CPU copies through `grid::ReadValues` / `ReadBuffer` (test support `ReadAll` for a
    plain buffer), as one `std::vector<std::byte>` per array, for exact comparisons and for `WriteResultDigest`.

- [ ] **Step 1: Failing tests** (`transfer_test_scenes_test.cc`, label `cpu`): `UniformGrid, Has27Blocks`;
  `SizeStepGrid, HasTwoLevels`; `BuildSolidsAtFaces, PlateAtX_XFacesOnThePlaneAreInsideTheSolid` (X faces at
  `x = 0.6`: `outside_solid == 0`; at `0.5` side `kAgainstNormal`, at `0.7` `kAlongNormal`);
  `BuildCloud, SameSeed_SamePositions`; `BuildParticlesTry, GroupsByCellThenId`.
- [ ] **Step 2: Run, implement, run.** Expected: five pass.
- [ ] **Step 3: Gate; commit.** Subject: `Add the scenes the transfer tests share`.

### Task 6: Same-side weights and face sums (shared formulas)

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** tests are `cpu` only (no `no_gpu` target); the container's `cuda` step runs them.

**Files:**
- Create: `native/fluidity/transfer/same_side_weights.h`, `face_sums.h`, `same_side_weights_test.cc`,
  `face_sums_test.cc`

**Interfaces:**
- Consumes: `grid::FindNearbyFaces`, `ComputeNearbyFaceWeights`, `ToNearbyFaceSums`, both `ComputeFaceWeight`.
- Produces (private, `FLUIDITY_CPU_AND_GPU`, plan decisions):
  - `grid::NearbyFaceMask FindSameSideFaces(const grid::NearbyFaceList& list, const SolidsAtFacesOnProcessor& solids,
    contact::PlateSide particle_side)`: bit kept when `outside_solid && side == particle_side` (design 5.1).
  - `struct ParticleSums { grid::NearbyFaceSums axis[3]; }` and `ParticleSums ComputeParticleSums(const
    grid::GridTables&, Vec3 position, contact::PlateSide side, const SolidsAtFacesOnProcessor&, TransferError*,
    Axis* failing_axis)`: pass 2; `weight_sum <= 0` on an axis gives `kNoFaceOnSameSide` (axes in X, Y, Z order).
  - `struct FaceContribution { double mass; double momentum; double force; double weight; }` and
    `FaceContribution ComputeFaceContribution(const grid::FaceWeight&, Axis, double mass, Vec3 velocity,
    const Mat3* fitted_velocity_change, const Mat3* internal_force_per_area, double rest_volume)`: the four terms
    of design 5.2 step 3 (the fitted term only when the pointer is non-null; the force term only with a matrix).
  - `struct FaceSumsInputs` (the per-material views, counted material list, the per-particle sums in `TransferBuffers`, solids view) and
    `FaceSums ComputeFaceSums(const grid::GridTables&, grid::FacePlaceIndex, const FaceSumsInputs&, TransferFailure*)`:
    pass 3 and 4 for one face, in the order of the Global Constraints; `FaceSums` carries mass, momentum, force,
    weight_sum, velocity, `FaceValueKind`, `SidesAtFace`.

- [ ] **Step 1: Failing tests** (label `cpu`).

  ```cpp
  TEST(ComputeParticleSums, InteriorParticle_WeightSumIsOneAndMeanZero)   // uniform, {0.53,0.47,0.61}: within 4 eps
  TEST(ComputeParticleSums, BesidePlate_NormalisedWeightsSumToOne)        // x = 0.57, PlateAtX(0.6): sum n within 4 eps
  TEST(ComputeParticleSums, BesidePlate_WeightChangesSumToZero)           // sum dn within 16 eps / h
  TEST(ComputeParticleSums, BesidePlate_CentredOffsetsHaveZeroMean)       // sum n rc within 16 eps h
  TEST(ComputeParticleSums, WrongSideOfPlate_IsNoFaceOnSameSideOnX)       // side claims kAlongNormal at x = 0.55
                                                                          // with every face of the list masked
  TEST(ComputeFaceContribution, NoFittedChange_MomentumIsMassTimesVelocity)
  TEST(ComputeFaceContribution, InternalForce_IsMinusRestVolumeTimesRowDotWeightChange)  // hand numbers:
      // rest_volume 2e-3, row x {1,2,3} Pa, dn {4,5,6} 1/m -> force -0.064
  TEST(ComputeFaceSums, FaceWithOneParticle_EqualsItsContribution)
  TEST(ComputeFaceSums, ThirdSide_IsTooManySidesAtFace)
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** Transfer does not repeat the grid's arithmetic: it builds the mask and calls the
  grid. `Read:` `<sec-same-side-weights>`, `<sec-particles-to-grid>`; `// Step N:` follow
  `<proc-particles-to-grid>`.
- [ ] **Step 4: Run** build, tests, format, rules, gate.
- [ ] **Step 5: Commit.** Subject: `Add the same-side weights and the ordered face sum`.

### Task 7: The two fits (shared formulas)

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** tests are `cpu` only (no `no_gpu` label).

**Files:**
- Create: `native/fluidity/transfer/linear_fit.h`, `velocity_fit.h`, `linear_fit_test.cc`, `velocity_fit_test.cc`

**Interfaces:**
- Produces (private, shared, plan decisions on names):
  - `inline constexpr int kMaxFitRows = 64;`, `struct LinearFit { double value; Vec3 change; bool has_full_rank; }`
    and `LinearFit SolveLinearFit(const Vec3* offsets_in_cell_widths, const double* values, int count)`:
    `velocity ~ c0 + c . offset` by Householder QR with column pivoting; a pivot below `kFillPivotTolerance`
    times the largest is rank loss. `count > kMaxFitRows` uses the first `kMaxFitRows` rows in the given order
    (design 2 states up to 36 sources at radius 2; 64 covers radius 4 in uniform cells; the hand-over records it).
  - `struct SpreadSolve { Vec3 solution; bool is_invertible; }` and `SpreadSolve SolveSpread(const Mat3& spread_in_cell_widths, Vec3 right_side)`:
    Cholesky; a pivot `<= grid::kMinPivotShare * trace` is not invertible (design 5.6).

- [ ] **Step 1: Failing tests** (label `cpu`).

  ```cpp
  TEST(SolveLinearFit, ExactLinearData_IsReproduced)              // 12 offsets, value 0.5 + {2,-1,0.5}.offset; 1e-13
  TEST(SolveLinearFit, CoplanarOffsets_LosesRank)                 // all offsets with z = 0
  TEST(SolveLinearFit, FourPointsInGeneralPosition_HasFullRank)
  TEST(SolveLinearFit, RowOrderPermuted_SameValueWithin1e-15)
  TEST(SolveSpread, CompleteSpread_IsQuarterIdentitySolve)        // spread I/4 -> solution 4 * right_side, exact
  TEST(SolveSpread, FlatSpread_IsNotInvertible)                    // diag {1/4, 1/4, 1e-14}
  TEST(SolveSpread, AgreesWithGridFit_OnSlabAndInterior)          // is_invertible == (NearbyFaceWeights::fit == kFull)
                                                                   // for an interior and a slab particle
  ```
- [ ] **Step 2: Run** (compile error), **implement**, **run**. `Read:` `<sec-filled-faces>`, `<eq:fit>`.
- [ ] **Step 3: Gate; commit.** Subject: `Add the least-squares fill fit and the spread solve`.

### Task 8: Python reference for the two fits

**Kind:** numerics

**Files:**
- Create: `reference/transfer/least_squares.py`, `reference/transfer/least_squares_test.py`,
  `native/fluidity/transfer/testdata/least_squares_cases.txt`, `native/fluidity/transfer/least_squares_reference_test.cc`
- Modify: `pyproject.toml`, `uv.lock` (NumPy in the `dev` group, newest 2.x; pin what resolves),
  the transfer `CMakeLists.txt` (`FLUIDITY_TRANSFER_TESTDATA_DIR` compile definition for this test; a CTest
  `reference_transfer_least_squares` running `uv run pytest reference/transfer -q`, label `cpu`)

**Interfaces:**
- Produces `reference/transfer/least_squares.py` (about 60 lines, design 7): `fill_fit(offsets, values)` and
  `fitted_change(weights, centred_offsets, values)` through `numpy.linalg.lstsq` (the second on rows scaled by
  `sqrt(weight)`), in Z-up and the decided vocabulary; `write_cases(path)` writes 40 cases with a fixed seed
  (`numpy.random.default_rng(20261114)`): 20 non-linear (`value = sin(3 x) + y^2 - z`), 20 nearly rank-deficient
  (coplanar offsets plus a perturbation of `1e-6` along the normal). One case per line:
  `kind count offsets... weights... values... expected...`, every number as `float.hex`.

- [ ] **Step 1: Failing tests.** Python: `test_cases_file_is_current` (regenerate into `tmp_path`, compare text
  with the committed file), `test_fill_fit_reproduces_linear_data`. C++ (label `cpu`):
  `TEST(SolveLinearFit, MatchesLeastSquaresReference)` and `TEST(SolveSpread, MatchesLeastSquaresReference)`:
  every case within `1e-9 U` (`U` = largest `|value|` of the case); cases the reference marks rank-deficient by
  NumPy's rank must lose rank in C++ too.
- [ ] **Step 2: Run** (fail), **write the reference**, `uv run python reference/transfer/least_squares.py --write
  native/fluidity/transfer/testdata/least_squares_cases.txt`, **run** both. The rules check scans `.py` files:
  keep it clean.
- [ ] **Step 3: Gate; commit.** Subject: `Check the two fits against a NumPy least-squares reference`. Body: the
  uncertainty it removes (design 7).

### Task 9: `ParticlesToGrid` on the CPU

**Kind:** numerics

**Amended 2026-11-14 (A31):** the stage has one form, taking `TransferBuffers& transfer_buffers` last; tests pass `test::BuildTransferBuffersFor(particles_try)`.

**Files:**
- Create: `native/fluidity/transfer/particles_to_grid.h`, `particles_to_grid_cpu.cc`, `particles_to_grid_test.cc`

**Interfaces:**
- Consumes: Tasks 3 to 6.
- Produces: design 3 `ParticlesToGridInputs`, `ParticlesToGridResult`, `ParticlesToGrid` (one form, taking
  `TransferBuffers&`; amended 2026-11-14, A31). Fills every member of `GridFromParticles` for the faces of `inputs.blocks`
  (`velocity` 0 unless `kFromParticles`), then `material_set`, `grid_version`, `step_try`.

- [ ] **Step 1: Failing tests** (label `cpu`; `settings` default unless stated).

  | Test | Setup | Expected | Tolerance |
  |---|---|---|---|
  | `AnyCloud_ConservesMassAndMomentum` | size-step grid, `PlateAtY(0.4)` (centre x 0.8), 200 particles `BuildCloud(1, 200, {0.1,0.1,0.1}, {1.5,0.7,0.7})`, velocities and fitted-change entries in `[-1, 1]` from seed 2 | per axis: sum of face mass = sum of particle mass; sum of face momentum = sum of `m v_a` | `64 eps M` |
  | `ConstantVelocity_EveryFaceHasIt` | uniform, 400 particles in `[0.2, 1.0)`, all `{1, -2, 3}`, zero fitted change | every `kFromParticles` face velocity equals its axis component | `8 eps U`, `U = sqrt(14)` |
  | `LinearFieldCompleteNeighbourhood_IsReproduced` | uniform, 8 particles per cell (2 x 2 x 2 at quarter points) in `[0.1, 1.1)`, `v = a + B x`, `a = {1,-2,3}`, `B` rows `{0.5,0.1,-0.2}`, `{0.3,-0.4,0.25}`, `{-0.1,0.2,0.6}`, fitted change `B` | faces with centre in `[0.4, 0.8]`: `(a + B x_f)_axis` | `1e-13 U` |
  | `InternalForce_AddsUpToZeroPerParticle` | one particle at `{0.53,0.47,0.61}`, symmetric matrix entries in `[-1e3, 1e3]` (seed 3), rest volume `1e-3`; then at `{0.57,0.5,0.5}` with `PlateAtX(0.6)` | per axis, face forces sum to 0 | `64 eps M` |
  | `CompleteNeighbourhood_ConservesAngularMomentum` | uniform, 50 interior particles, random `v`, `C` (seed 4) | `sum_f x_f x (q_f e_a) = sum_p m_p (x_p x v_p + (h^2/4) sum_a row_a(C_p) x e_a)` | `1e-12 sum_p m_p (|x_p||v_p| + h^2/4 sum |C_p|)` |
  | `WeightSumThresholds_SetValueKind` | one particle `{0.55,0.55,0.55}`; first run reads the weight sum `w` of X face `(6,5,5)`; second run with `min_weight_sum_at_face = w` | a face with weight sum 0 `kNoValue`; that face `kTooFewParticles`, velocity 0; a face above `w` `kFromParticles` | exact |
  | `StorageOrderChanged_SameBytes` | the cloud of the first row added in reversed order | `ReadAllBytes` equal | exact |
  | `SplitIntoTwoRanges_SameBytes` | same cloud; blocks `[0, n/2)` then `[n/2, n)` into one output | equal to one call on `all_blocks()` | exact |
  | `RunTwice_SameBytes` | same cloud, twice | equal | exact |
  | `NegativeMass_IsBadInputForLowestId` | ids 12 and 5 with mass `-1` | `kBadInput`, `particle_id 5`, `magnitude -1` | exact |
  | `NoSameSideFace_IsReportedForLowestId` | ids 9 and 4 at `{0.55,0.6,0.6}` given side `kAlongNormal` of `PlateAtX(0.6)` | `kNoFaceOnSameSide`, id 4, `Axis::kX` | exact |
  | `ThreeSides_IsTooManySidesForLowestFace` | `PlateAtX(0.6)` particles of both sides plus id 30 with side `{9, kAlongNormal}` beside them | `kTooManySidesAtFace`, the lowest `FacePlaceIndex` reached by three sides, id 30 | exact |
  | `ParticleOnCellBoundary_ConservesMass` (Review Focus 1) | particles exactly at `x = 0.5`, at `{0.5,0.5,0.5}` and at a face centre `{0.5,0.55,0.55}` | as the first row | `64 eps M` |
  | `FailedCall_SetsStepTryToZero` | the negative-mass input after a good call | `step_try == 0` | exact |
  | `MaskOfWrongLength_IsBadInput` | 2 materials, mask `{1}` | `kBadInput`, magnitude 1 | exact |
  | `OtherGridVersion_IsWrongGridVersion` | `solids_at_start` bound to another version | `kWrongGridVersion` | exact |

  Name each `TEST(ParticlesToGrid, <name>)`.
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement** the four passes of `<proc-particles-to-grid>` with `// Step N:` comments. Pass 1 checks
  in this order: mask, grid versions of inputs, grouping, then per counted particle (ascending id across the
  set) finite position, velocity, fitted change, `mass >= 0`. Pass 2 writes `ParticleSums` to the `TransferBuffers`. Passes
  3 and 4 call `ComputeFaceSums` once per face of the range in ascending `FacePlaceIndex`, skipping
  `kNotAPiece` places. Result counts are integer sums. Plain single-thread loops (threads are allowed later, one
  writer per face).
- [ ] **Step 4: Run** build, tests (both macOS presets), format, rules, gate.
- [ ] **Step 5: Commit.** Subject: `Move particle velocity to grid faces on the CPU`. New names: `ParticlesToGrid`,
  `ParticlesToGridInputs`, `ParticlesToGridResult`.

### Task 10: `ParticlesToGrid` on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two-halves pattern: private `particles_to_grid_gpu.h` returning `RunOnCallStatus`, kernels as thin loops over `particles_to_grid_steps.h`, a `cpu` launch-order test, digests.

**Files:**
- Create: `particles_to_grid_gpu.h` (private), `particles_to_grid_gpu.cu`, `particles_to_grid_steps.h` (shared,
  private), `particles_to_grid_steps_test.cc` (label `cpu`), `particles_to_grid_gpu_test.cc` (label `gpu`),
  `particles_to_grid_no_gpu_test.cc` (label `no_gpu`)
- Modify: `particles_to_grid_cpu.cc` (dispatch under `#if defined(FLUIDITY_HAS_GPU)`, contact's shape), the
  transfer `CMakeLists.txt` (`fluidity_add_headers(particles_to_grid_steps.h)`,
  `fluidity_add_gpu_sources(particles_to_grid_gpu.cu)`, the three tests with `LINKS fluidity_base_test_support
  fluidity_transfer_test_scenes`)

**Interfaces:**
- Consumes: `ComputeParticleSums`, `ComputeFaceSums`, `TransferBuffers`; `ComputeLaunchSize`, `ApplyGpuFunction`,
  `CheckGpuWork`, `ToBufferError`, `ReadBuffer`.
- Produces: the `RunOnCallStatus ParticlesToGridOnGpu(...)` GPU half (private, plain pointers and the
  on-processor views) and, in `particles_to_grid_steps.h` (plan decision, `FLUIDITY_CPU_AND_GPU`):
  `std::uint64_t ToTransferFailureKey(int pass, std::uint64_t id_or_face)` (`(pass << 56) | id_or_face`),
  `std::uint64_t ApplyParticleSumsAt(...)` (pass 2 for one counted particle: writes its `ParticleSums` to the
  `TransferBuffers`, returns its failure key or `kNoFailureKey`) and `std::uint64_t ApplyFaceSumsAt(...)` (passes 3 and 4 for
  one face place of the range: writes its outputs, returns its failure key). The `RunOn::kGpu` path of
  `ParticlesToGrid`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  // particles_to_grid_steps_test.cc, label cpu: the main evidence until the hardware run
  TEST(ParticlesToGridSteps, EveryThreadBlockAndThread_SameBytesAsCpu)
      // every Task 9 scene: ApplyParticleSumsAt, then ApplyFaceSumsAt, run for every thread block and thread of
      // ComputeLaunchSize(count), in launch order and in reverse, on CPU arrays, the failure key reduced with a
      // plain minimum: ReadAllBytes equal to the CPU stage's, and the same failure on the failing scenes
  // particles_to_grid_no_gpu_test.cc, label no_gpu
  TEST(ParticlesToGrid, GpuWithoutAGpu_IsRunOnCallFailed)
  // particles_to_grid_gpu_test.cc, label gpu (compiled here, run by the owner)
  TEST(ParticlesToGrid, OnGpu_MatchesCpuOnEveryTaskNineScene)   // tolerances of the Global Constraints
  TEST(ParticlesToGrid, OnGpuRunTwice_SameBytes)
  TEST(ParticlesToGrid, OnGpuStorageOrderChanged_SameBytes)
  TEST(ParticlesToGrid, OnGpuSplitIntoTwoRanges_SameBytes)
  TEST(ParticlesToGrid, OnGpuNegativeMass_ReportsTheCpuFailure)   // same error, id 5
  ```
  Every `gpu` test passes the GPU `GridFromParticles` bytes (`ReadAllBytes`) and the result to
  `WriteResultDigest`. The CPU-versus-GPU comparison uses `CheckWithinTolerance` (the CPU value the reference)
  with the Global Constraints' tolerances; the `_SameBytes` tests compare GPU runs with each other.
- [ ] **Step 2: Implement.** The counted material indexes are read from the host mask before launch and passed
  as a short ascending list. One kernel over counted particles for pass 2 (`ApplyParticleSumsAt`); one kernel over
  the range's face places for passes 3 and 4 (`ApplyFaceSumsAt`); each a grid-stride loop launched with
  `ApplyGpuFunction(..., ComputeLaunchSize(count), ...)`. The first failure: integer `atomicMin` of the returned
  key, then `CheckGpuWork` and one read of the status record. No floating-point atomics. Nothing is copied to the
  CPU but the status record. A status other than `kDone` goes through `ToBufferError` and the stage's mapping.
- [ ] **Step 3: Run** the gate. Expected: `PASS cuda` (compiled; the `cpu` steps test and the `no_gpu` test pass in
  the container).
- [ ] **Step 4: Commit.** Subject: `Move particle velocity to grid faces on the GPU`. Body: `CUDA: compiled, not run`;
  private names `ParticlesToGridOnGpu`, `ToTransferFailureKey`, `ApplyParticleSumsAt`, `ApplyFaceSumsAt`.

### Task 11: Filled faces and `SaveGridBeforeForces` on the CPU

**Amended 2026-11-15 (A64):** `kFromGridOnlyRegion` is not a source kind in the first version (Task L1 adds it).

**Kind:** numerics

**Amended 2026-11-14 (A31):** the stage has one form, taking `TransferBuffers& transfer_buffers` last; tests pass `test::BuildTransferBuffersFor(particles_try)`.

**Files:**
- Create: `fill_face.h`, `filled_velocity.h`, `grid_snapshot_cpu.cc`, `grid_snapshot_test.cc`
- Modify: `grid_snapshot.h` (functions)

**Interfaces:**
- Consumes: `SolveLinearFit`, `grid::BuildFaceSearch`, `FindNextFace`, `GridSnapshotOnProcessor`.
- Produces: design 3 `SaveGridInputs`, `SaveGridBeforeForces` (one form, taking `TransferBuffers&`; amended
  2026-11-14, A31), `FilledVelocityInputs`, `ComputeFilledVelocity`. Private (`fill_face.h`, shared): `FilledVelocity ComputeFill(const grid::GridTables&,
  const GridSnapshotOnProcessor&, const SolidsAtFacesOnProcessor&, grid::FacePlaceIndex, contact::PlateSide,
  const std::uint8_t* was_solved, TransferError*)` (`was_solved` null: every source counts), which
  `ComputeFilledVelocity` calls with null.
- Plan decisions: radii are in widths of the filled face's cell (`ComputeCellWidth` of its block's level);
  "nearest" uses the distance between face positions, ties to the lower `FacePlaceIndex`; step 4's other axes
  are fitted from the faces of those axes around the face position, by the same search and fit.

- [ ] **Step 1: Failing tests** (`TEST(SaveGridBeforeForces, ...)`; inputs from
  `BuildGridFromParticlesByFunction` on the uniform grid with `PlateAtX(0.6)` unless stated).

  | Test | Setup | Expected | Tolerance |
  |---|---|---|---|
  | `LinearFieldAcrossPlate_FillIsExact` | side A (`x < 0.6`) X faces `0.5 + {2,-1,0.5}.x`, side B faces `1e30`; faces near the plate reached by both sides | `filled[A]` of every B-side face reached by A is `0.5 + {2,-1,0.5}.x_f`, kind `kFitted` | `1e-11 U` |
  | `CoplanarSources_CopiesNearest` (Review Focus 3) | only faces with `z = 0.55` are sources; the two nearest sources of the tested face are at equal distance | `kNearestSource`, value of the lower `FacePlaceIndex` of the two | exact |
  | `NoSourceButSolid_UsesSolidVelocity` | no source within 4 widths; `ObliqueFloor()` at the face with solid velocity `{0.1, 0, 0}` | `kSolidVelocity`, value 0.1 on an X face | exact |
  | `Nothing_IsFaceCannotBeFilled` | no source, no solid | `kFaceCannotBeFilled`, that `face` | exact |
  | `SourcesAtThreeWidths_FitsAtSecondRadius` | linear field on sources only at distances in `(2, 3]` widths | `kFitted`, value of the field | kind exact, `1e-11 U` |
  | `TooFewParticlesFace_IsNotASource` | one source-distance face `kTooFewParticles` with velocity `1e30` | `is_source == 0` there; filled values unchanged from a run without it | exact |
  | `SplitIntoTwoRanges_SameBytes` (Review Focus 4) | cut between blocks so a fill's sources lie in the other range | equal to one call | exact |
  | `OutputOfAnotherSet_IsWrongMaterialSet` | `grid_from_particles.material_set` of mask `{0,1}`, mask `{1,0}` | `kWrongMaterialSet` | exact |

  And `TEST(ComputeFilledVelocity, MatchesTheSavedFill)`: for every saved `filled[s]` of the first row, equal bytes.
- [ ] **Step 2: Run** (compile error), **implement** `<proc-fill-face>` steps 1 to 4 with `// Step N:`. Copy
  `velocity`, `value_kind`; `is_source` 1 for `kFromParticles`, `kFromSolve` (amended 2026-11-15, A64:
  `kFromGridOnlyRegion` is added by the later group L1); `side` and
  `outside_solid` from `inputs.solids`; for each face and each side in `sides_reaching`, `kNotNeeded` when the
  side reads the face directly (design 5.4 first paragraph), otherwise the fill. Port from `git show
  main:native/src/liquid_boundary_transfer_3d.cc` lines 186-307 (`ContinueComponent`, `FillVelocity`) and
  `main:native/include/fluidity/base/liquid_boundary_transfer_math_3d.h` lines 34-144 (QR), rotated to Z-up.
- [ ] **Step 3: Run** build, tests, format, rules, gate.
- [ ] **Step 4: Commit.** Subject: `Save the grid before forces with its filled faces`.

### Task 12: `SaveGridAfterPressure` on the CPU

**Kind:** numerics

**Amended 2026-11-15 (A64):** the `kFromGridOnlyRegion` rule and its test `GridOnlyRegionFace_IsASourceAndKeepsItsKind` move to the later group (Task L1).

**Amended 2026-11-14 (A31):** the stage has one form, taking `TransferBuffers& transfer_buffers` last; tests pass `test::BuildTransferBuffersFor(particles_try)`.

**Files:**
- Modify: `grid_snapshot.h`, `grid_snapshot_cpu.cc`, `grid_snapshot_test.cc`

**Interfaces:**
- Produces: design 3 `SaveGridAfterPressure` (one form, taking `TransferBuffers&`; amended 2026-11-14, A31).
- Rules (design 5.5): `velocity` from `velocity_after_pressure`; a `kNoValue` face with `was_solved` becomes
  `kFromSolve`; other kinds kept; `is_source` as before forces. `filled` uses every
  source, for particles outside solved cells (spray). Plan decision, because this function receives no
  `cell_was_solved`: when `was_solved` is not null, `filled_from_solved[s]` is computed for every face and side in
  `sides_reaching`, as `kNotNeeded` where the face has `was_solved`, otherwise by `ComputeFill(..., was_solved)`,
  which accepts only sources with `was_solved`. The design's "kept only for faces touching a solved cell" is then a
  memory optimisation not taken (the array is allocated for every face anyway). Null `was_solved` leaves
  `filled_from_solved` empty. `GridSnapshot::time = kAfterPressure`; sides and solid samples from `inputs.solids`
  at the end pose.

- [ ] **Step 1: Failing tests.**

  | Test | Setup | Expected | Tolerance |
  |---|---|---|---|
  | `SaveGridAfterPressure, ObliqueSolid_NoMotionThroughIt` (it needs both snapshots) | uniform grid, `ObliqueFloor()`, liquid faces of the constant field `w = {1, 0.5, -2}`; solid velocity `{0.1,0,0}` at time 0 and `{0,0,0.3}` at `step_duration = 0.01`; both snapshots | every filled X and Z face at a solid of its side: `w_a - ((w - s) . n) n_a` with each snapshot's own `s` | `1e-12 U` |
  | `SaveGridAfterPressure, NoSolveForSet_EveryUsableFaceRead` | `was_solved` and `cell_was_solved` null; a `1e30` face that a solve would not have written | `filled_from_solved` empty, no `kFromSolve`, kinds and fills equal the before-forces snapshot of the same input | exact |
  | `SaveGridAfterPressure, SolvedNoValueFace_IsFromSolve` | one `kNoValue` face with `was_solved` | `kFromSolve`, `is_source 1` | exact |
  | `SaveGridAfterPressure, BeforeForcesSnapshotPassed_IsWrongStageOrder` | output built with `GridTime::kBeforeForces` | `kWrongStageOrder` | exact |
- [ ] **Step 2: Run, implement, run.** `Read:` `<sec-grid-snapshots>`, `<sec-filled-faces>`.
- [ ] **Step 3: Gate; commit.** Subject: `Save the grid after pressure, filled from solved faces`.

### Task 13: Both snapshots on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two-halves pattern: private `grid_snapshot_gpu.h` returning `RunOnCallStatus`, the per-face-and-side body in `grid_snapshot_steps.h`, a `cpu` launch-order test, digests.

**Files:**
- Create: `grid_snapshot_gpu.h` (private), `grid_snapshot_gpu.cu`, `grid_snapshot_steps.h` (shared, private:
  `std::uint64_t ApplyFillAt(...)`, one face place and side: copies the face's values, writes `ComputeFill`'s result,
  returns the failure key from `ToTransferFailureKey`), `grid_snapshot_steps_test.cc` (label `cpu`),
  `grid_snapshot_gpu_test.cc`, `grid_snapshot_no_gpu_test.cc`
- Modify: `grid_snapshot_cpu.cc` (dispatch under `#if defined(FLUIDITY_HAS_GPU)`), the transfer `CMakeLists.txt`

- [ ] **Step 1: Failing tests.** `cpu` (the main evidence until the hardware run):
  `TEST(GridSnapshotSteps, EveryThreadBlockAndThread_SameBytesAsCpu)`: every Task 11 and Task 12 scene,
  `ApplyFillAt` for every thread block and thread of `ComputeLaunchSize(face places x kMaxSidesAtFace)`, in launch
  order and in reverse, on CPU arrays: `ReadAllBytes` and the failure equal the CPU stage's.
  `no_gpu`: `TEST(SaveGridBeforeForces, GpuWithoutAGpu_IsRunOnCallFailed)`,
  `TEST(SaveGridAfterPressure, GpuWithoutAGpu_IsRunOnCallFailed)`. `gpu`:
  `TEST(SaveGridBeforeForces, OnGpu_MatchesCpuOnEveryTaskElevenScene)`,
  `TEST(SaveGridAfterPressure, OnGpu_MatchesCpuOnEveryTaskTwelveScene)`, `TEST(SaveGridBeforeForces,
  OnGpuRunTwice_SameBytes)`, `TEST(SaveGridBeforeForces, OnGpuSplitIntoTwoRanges_SameBytes)`. Every `gpu` test
  passes the GPU snapshot bytes to `WriteResultDigest`; CPU-versus-GPU values through `CheckWithinTolerance`.
- [ ] **Step 2: Implement.** One grid-stride kernel over face places and sides calling `ApplyFillAt`, launched with
  `ApplyGpuFunction`; the search walk runs inside the thread (`grid::FaceSearch` is a fixed-size struct). Failure
  key `(pass << 56) | face index`, reduced by integer `atomicMin`.
- [ ] **Step 3: Gate; commit.** Subject: `Save both grid snapshots on the GPU`. Body: `CUDA: compiled, not run`.

### Task 14: `GridToParticles` on the CPU

**Kind:** numerics

**Amended 2026-11-14 (A31):** the stage has one form, taking `TransferBuffers& transfer_buffers` last; tests pass `test::BuildTransferBuffersFor(particles_try)`.

**Files:**
- Create: `grid_at_particle.h`, `grid_to_particles.h`, `grid_to_particles_cpu.cc`, `grid_to_particles_test.cc`

**Interfaces:**
- Consumes: `SolveSpread`, both snapshot views, grid's weights with `grid::ToMaskOfAll`.
- Produces: design 3 `GridOnlyReason`, `kGridOnlyReasonCount`, `GridAtParticle`, `GridToParticlesInputs`,
  `GridToParticlesResult`, `GridToParticles` (one form, taking `TransferBuffers&`; amended 2026-11-14, A31). Private
  shared
  `GridAtParticle ComputeGridAtParticle(const grid::GridTables&, const GridSnapshotOnProcessor& before, const
  GridSnapshotOnProcessor& after, const std::uint8_t* cell_was_solved, Vec3 position, contact::PlateSide side,
  Vec3 velocity, std::uint8_t flags, TransferError*)`.
- Rules (design 5.6): per axis, all faces of the nearby list (`ToMaskOfAll`); a face's value is the snapshot's
  `velocity` when read directly, else `filled[i]` of the particle's side (after pressure: `filled_from_solved`
  for a particle whose cell has `cell_was_solved` and a face without `was_solved`). Cholesky in units of the
  particle's cell width. Reasons in the order of the enum. Without a reason `own_motion = v -` (before-forces
  velocity at the particle).

- [ ] **Step 1: Failing tests** (`TEST(GridToParticles, ...)`; snapshots saved from hand-made face sums).

  | Test | Setup | Expected | Tolerance |
  |---|---|---|---|
  | `LinearField_VelocityAndFitExact` | four cases: uniform interior; size-step grid at `x = 0.85`; uniform at `x = 0.04` (domain side); `PlateAtX(0.6)` at `x = 0.57` with fill; field `a + B x` of Task 9 on every face and side | `grid_velocity_at_particle = a + B x_p`; `fitted_velocity_change = B` | `1e-12 U`; `1e-10 U / h` |
  | `ExactChange_IsDerivativeOfInterpolant` | field `u_x = x^2`, other axes 0, the four cases | `exact_velocity_change` row x equals the central difference (step `1e-6 h`) of `sum n_f u_f` computed in the test from grid's weights; in the size-step case `fitted` differs from `exact` by more than `1e-3 U / h` | `1e-6 U / h` |
  | `SameSnapshots_ParticleVelocityComesBack` | after = before, 200 interior particles | `grid_velocity_at_particle + own_motion = v_p` | `8 eps U` |
  | `FreeFall_GainsGravityOnceForAnyKeptShare` | 300 interior particles at `{0.3,-0.2,0.1}`, zero fitted change; `ParticlesToGrid`, before-forces snapshot, add `0.01 * -9.81` to every `kFromParticles` Z face, after-pressure snapshot (`was_solved` null); then `MixParticleVelocity` with kept 0, 0.5, 1 and travel 0 | change of `for_next_step` is `{0, 0, -0.0981}`; a control with the first snapshot saved after the force gains `(1 - kept) * -0.0981` (design 5.3) | `1e-12 * 0.0981` |
  | `EachReason_IsReportedAndCounted` | one particle each: `kHasOwnMotion` off; beside a `kTooFewParticles` face; beside a `kNearestSource` fill before forces; the same after pressure only; in `BuildSlabGrid` | reasons `kNewParticle` .. `kSpreadNotInvertible`, `own_motion` zero, `carries_fitted_change` false, `particle_count_by_reason` 1 each plus the rest `kNone` | exact |
  | `ParticleInSolvedCell_ReadsSolvedFacesOnly` | an unsolved face with `1e30` next to a solved cell, particle in the solved cell | result equal to the run with that face set to the field | `1e-12 U` |
  | `SetMask_WritesOnlyCountedMaterials` | three materials, mask `{1,0,1}`, buffer 1 filled with byte `0xA5` | buffer 1 unchanged; counts only of materials 0 and 2 | exact |
  | `AfterFailedSnapshot_IsWrongStageOrder` (Review Focus 5) | `after_pressure.step_try = 0`; then snapshots of tries 1 and 2 | `kWrongStageOrder` both | exact |
  | `GridAtParticle, Size_Is200Bytes` (plan-added) | `static_assert(sizeof(GridAtParticle) == 200)` | the step design's 6.3 row | exact |

- [ ] **Step 2: Run** (compile error), **implement** `<proc-grid-to-particles>` steps 1 to 7. Port from
  `main:native/src/mac_gather_3d_internal.h` lines 35-67 (sums), `main:native/src/liquid_boundary_transfer_3d.cc`
  lines 518-651 (`GatherCore`) and the `mu` correction of `main:native/src/graded_mac_transfer_3d.cc` lines
  524-560; never the `4 / cell_width` factor.
- [ ] **Step 3: Run** build, tests, format, rules, gate.
- [ ] **Step 4: Commit.** Subject: `Move grid velocity back to particles on the CPU`.

### Task 15: `GridToParticles` on the GPU, and `MixParticleVelocity`

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** two-halves pattern: private `grid_to_particles_gpu.h` returning `RunOnCallStatus`, the per-particle body in `grid_to_particles_steps.h`, a `cpu` launch-order test, digests; the mix test is `cpu` only.

**Files:**
- Create: `grid_to_particles_gpu.h` (private), `grid_to_particles_gpu.cu`, `grid_to_particles_steps.h` (shared,
  private: `std::uint64_t ApplyGridAtParticleAt(...)`, one counted particle: writes `ComputeGridAtParticle`'s
  result, returns the failure key; `GridOnlyReason` of the result is what the kernel counts),
  `grid_to_particles_steps_test.cc` (label `cpu`), `grid_to_particles_gpu_test.cc`,
  `grid_to_particles_no_gpu_test.cc`, `mix_particle_velocity.h`, `mix_particle_velocity_test.cc`
- Modify: `grid_to_particles_cpu.cc` (dispatch under `#if defined(FLUIDITY_HAS_GPU)`), the transfer
  `CMakeLists.txt`

**Interfaces:**
- Produces: the GPU path; design 3 `MixInputs`, `MixResult`, `MixParticleVelocity` (formula, design 5.7).

- [ ] **Step 1: Failing tests.**

  ```cpp
  // grid_to_particles_steps_test.cc, label cpu: the main evidence until the hardware run
  TEST(GridToParticlesSteps, EveryThreadBlockAndThread_SameBytesAsCpu)   // every Task 14 scene; launch order and
                                                             // reverse; GridAtParticle bytes, reason counts and
                                                             // failure equal the CPU stage's
  // mix_particle_velocity_test.cc, label cpu
  TEST(MixParticleVelocity, Example_MatchesHandNumbers)      // grid {1,0,0}, own {0.2,0,0}, kept 0.5, travel 0.25:
                                                             // for_next_step {1.1,0,0}, for_this_step {1.05,0,0},
                                                             // exact; also static_assert
  TEST(MixParticleVelocity, Reason_GivesGridVelocityAndFlag) // reason kNewParticle: both grid, used_grid_only
  TEST(MixParticleVelocity, TravelShareIndependentOfKept)    // kept 0, travel 1: for_this_step = grid + own
  // grid_to_particles_no_gpu_test.cc
  TEST(GridToParticles, GpuWithoutAGpu_IsRunOnCallFailed)
  // grid_to_particles_gpu_test.cc, label gpu
  TEST(GridToParticles, OnGpu_MatchesCpuOnEveryTaskFourteenScene)
  TEST(GridToParticles, OnGpuRunTwice_SameBytes)
  ```
  Both `gpu` tests pass the GPU `GridAtParticle` bytes and the result to `WriteResultDigest`; CPU-versus-GPU values
  through `CheckWithinTolerance`.
- [ ] **Step 2: Implement.** One grid-stride kernel over the particles of counted materials calling
  `ApplyGridAtParticleAt`, launched with `ApplyGpuFunction`; reason counts as integer `atomicAdd`s in the status
  record. `Read:` `<sec-mix>`.
- [ ] **Step 3: Gate; commit.** Subject: `Move grid velocity back to particles on the GPU and mix it`. Body:
  `CUDA: compiled, not run`; new names `MixInputs`, `MixResult`, `MixParticleVelocity`.

### Task 16: `ComputeCellsFromParticles` on both processors

**Kind:** numerics

**Amended 2026-11-14 (A31):** the stage has one form, taking `TransferBuffers& transfer_buffers` last; tests pass `test::BuildTransferBuffersFor(particles_try)`.

**Amended 2026-11-14 (Foundation as built):** two-halves pattern (private `cells_from_particles_gpu.h`, per-cell body in `cells_from_particles_steps.h`, `cpu` launch-order test); the bit-for-bit GPU test is kept by a deliberate same-bits promise in `cells_from_particles.h`; digests.

**Files:**
- Create: `cells_from_particles_cpu.cc`, `cells_from_particles_gpu.h` (private), `cells_from_particles_gpu.cu`,
  `cells_from_particles_steps.h` (shared, private: `std::uint64_t ApplyCellSumsAt(...)`, one cell place: its sums
  in the order of the Global Constraints, returns its failure key), `cells_from_particles_test.cc`,
  `cells_from_particles_steps_test.cc` (label `cpu`), `cells_from_particles_gpu_test.cc`,
  `cells_from_particles_no_gpu_test.cc`
- Modify: `cells_from_particles.h` (functions), the transfer `CMakeLists.txt`

**Interfaces:**
- Produces: design 3 `CellsFromParticlesInputs`, `ComputeCellsFromParticles` (one form, taking `TransferBuffers&`; amended
  2026-11-14, A31),
  `ComputeMixtureDensity`, `ComputeSensedVolumeRatio`.
- Plan decisions: `ingredient_volume` is the **current** volume of each ingredient: `fraction * volume_ratio *
  rest_volume` (fraction 1 for the material's only ingredient), so a cell's row sums to `current_volume`;
  `ComputeMixtureDensity` is `mass / current_volume`, `ComputeSensedVolumeRatio` is `current_volume /
  rest_volume`, each 0 when its divisor is 0. A cell with no particle of the set is all zeros.
- Same-bits promise (deliberate, plan decision): `cells_from_particles.h` states that `ComputeCellsFromParticles`
  gives the same bits on CPU and GPU unless NaN, because each cell's sums are written by one thread (one CPU loop
  iteration) with plain `*` and `+` in one fixed order, and both sides are compiled without contraction
  (`--fmad=false`, `-ffp-contract=off`); the CPU loop and the kernel call the same `ApplyCellSumsAt`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ComputeMixtureDensity, EmptyCell_IsZero)                 // also static_assert
  TEST(ComputeSensedVolumeRatio, HalfCompressed_IsHalf)         // (0.5e-3, 1e-3) -> 0.5
  TEST(ComputeCellsFromParticles, TwoIngredients_RowSumsToCurrentVolume)   // within 4 eps of current_volume
  TEST(ComputeCellsFromParticles, GasMarkersLeftOut_LiquidSumsUnchanged)
      // design 7: uniform grid, liquid (material 0, 2 ingredients) in z < 0.6, gas markers (material 1) in the
      // surface cells; mask {1,0}: mass, rest_volume, current_volume, ingredient_volume, both formulas, and
      // ParticlesToGrid's GridFromParticles bytes equal the run on a try without the markers. Exact.
  TEST(ComputeCellsFromParticles, SplitIntoTwoRanges_SameBytes)
  // cells_from_particles_steps_test.cc, label cpu: the main evidence until the hardware run
  TEST(CellsFromParticlesSteps, EveryThreadBlockAndThread_SameBytesAsCpu)   // the scenes above; launch order and
                                                                // reverse
  // cells_from_particles_no_gpu_test.cc, label no_gpu
  TEST(ComputeCellsFromParticles, GpuWithoutAGpu_IsRunOnCallFailed)
  // label gpu
  TEST(ComputeCellsFromParticles, OnGpu_EqualsCpuBitForBit)     // same order, plain adds: CheckSameBitsOrBothNaN
                                                                // per value (the promise above); bytes to
                                                                // WriteResultDigest
  ```
- [ ] **Step 2: Implement.** Order of the Global Constraints. GPU: one grid-stride kernel over cell places calling
  `ApplyCellSumsAt`, launched with `ApplyGpuFunction`; a status other than `kDone` goes through `ToBufferError`. `Read:`
  `<sec-material-sets>`.
- [ ] **Step 3: Gate; commit.** Subject: `Sum particle amounts per cell for each material set`. Body:
  `CUDA: compiled, not run`.

### Task 17: Material sets across the four stages, and the proof-of-concept comparison

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the two `gpu` tests write result digests of every stage's output.

**Files:**
- Create: `native/fluidity/transfer/material_sets_test.cc`, `material_sets_gpu_test.cc`,
  `proof_of_concept_test.cc`, `testdata/proof_of_concept_uniform.txt`, `testdata/proof_of_concept_plate.txt`

**Interfaces:**
- Consumes: every stage. Produces no library name.

- [ ] **Step 1: Failing tests** (label `cpu`).

  | Test | Setup | Expected |
  |---|---|---|
  | `MaterialSets, TwoSetsApart_SameBytesAsSeparateRuns` | uniform grid; set A liquid (material 0) in `[0.1, 0.4)^3`; set B a springy block (material 1) with internal force (symmetric matrix, seed 5) and `PlateAtX(0.9)` side, in `[0.8, 1.1) x [0.1, 0.4)^2`; masks `{1,0}`, `{0,1}`; the four stages and `ComputeCellsFromParticles` per set on the try with both, then on a try holding only A and only B (renumbered from 0); sets run A then B and B then A | identical bytes of `GridFromParticles`, both snapshots, `CellsFromParticles`, `GridAtParticle` and results; keys compared by material, not index. Exact |
  | `MaterialSets, ParticleOfSetANeverAddsToSetB` (Review Focus 2) | A and B interleaved in the same cells, both sides of `PlateAtX(0.6)`; then A's mass, velocity, fitted change and internal force times `1e30` and a third side `{9, kAlongNormal}` for one A particle | B's `GridFromParticles`, `sides_reaching`, snapshots, `CellsFromParticles` unchanged bytes; B's calls report no error. Exact |
  | `MaterialSets, BadMaskOrWrongSet_IsReported` | mask too short, an entry 2, all zero; set A's snapshot with set B's mask; `CheckMaterialSets` with material 1 in two sets and material 3 in none | `kBadInput` with the material index; `kWrongMaterialSet`; `kBadInput` magnitude 1. Exact |
  | `ProofOfConcept, UniformNoPlate_MatchesMain` | 64 particles, inputs read from `proof_of_concept_uniform.txt` (Y-up), rotated to Z-up | per-axis face totals and every particle's outputs within `1e-12 * |value| + 1e-12 * scale` |
  | `ProofOfConcept, OnePlate_MatchesMain` | `proof_of_concept_plate.txt` | filled values and gather outputs within `1e-9 U` |

  `gpu`: `MaterialSets, OnGpuTwoSetsApart_SameBytesAsSeparateRuns`, `MaterialSets,
  OnGpuParticleOfSetANeverAddsToSetB` (GPU runs compared with GPU runs; every stage's output bytes to
  `WriteResultDigest`).
- [ ] **Step 2: Produce the old numbers** (supporting evidence only; never the only test of this task). Export
  `main` without checking it out: `git archive main | tar -x -C <temp-dir>/main-export`. In that copy add a
  throwaway GoogleTest file (never committed anywhere) that builds 64 particles on `main`'s uniform grid of 12^3
  cells of width 0.1 from the origin, in `main`'s Y-up frame. Particle `i` in `0..63`: position
  `{0.45 + 0.025 (i % 4) + 0.003 (i % 3), 0.45 + 0.025 ((i / 4) % 4) + 0.002 (i % 5), 0.45 + 0.025 (i / 16) +
  0.001 (i % 7)}` m; velocity `{0.1 (i % 5) - 0.2, 0.05 (i % 7) - 0.15, 0.03 (i % 11) - 0.15}` m/s; fitted change
  entry (row r, column c) `0.01 ((i + 3 r + c) % 9 - 4)` 1/s; mass `1e-3` kg (the same formula goes into the data
  file's header and into the test). It runs `DepositMac3D` then `GatherMac3D` (`native/include/fluidity/host/transfers_3d.h`; see
  `native/tests/test_transfers_3d.cc` lines 137-150 for the calls) with old = new face velocities from the
  deposit, and prints inputs and outputs in `%a`. For the plate case, `GatherLiquidBoundaryParticles3D`
  (`native/tests/test_liquid_boundary_transfer_3d.cc` shows its fixtures). Build with `main`'s own presets
  there. Save the printed text to the two `testdata` files, each starting with the command lines used and the
  `main` commit hash. The test rotates with `R = rows {1,0,0},{0,0,-1},{0,1,0}` (vectors `R v`, matrices
  `R M R^T`): old Y totals compare with new Z totals, old Z totals with minus new Y totals. Rotated positions
  have `y` in `(-1.2, 0]`, so the test builds its Z-up grid with origin `{0, -1.2, 0}` (not `BuildUniformGrid`).
- [ ] **Step 3: Run, fix anything the first three tests show** (in the commit that owns the fault, `--fixup`),
  run the gate.
- [ ] **Step 4: Commit.** Subject: `Check material sets across every transfer stage`. Body records the `main`
  hash used and that the comparison is supporting evidence.

### Task 18: Comment pass and hand-over

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the hand-over lists the `_gpu.h` / `_steps.h` files, the same-bits promise of `cells_from_particles.h`, and the steps tests as the evidence until the hardware run.

**Files:**
- Modify: any file of `native/fluidity/transfer/`, `reference/transfer/`, `docs/method-paper/`,
  `docs/glossary.md`, `docs/gpu-runs.md`, `tools/check_rules_allowed_names.txt` that the pass changes

- [ ] **Step 1: Comment pass.** Reread every transfer file top to bottom as a C++ programmer new to fluids: every
  comment true of the final code; why, not what; a worked example with numbers in each non-trivial public
  docstring (the mix example, a two-block face sum, one fill); every word in the glossary; `Uses` / `Used by`
  lists correct; `// Step N:` match the paper's procedures.
- [ ] **Step 2: Fold** each change into the commit that owns the file (`git commit --fixup <hash>`, then
  `git rebase --autosquash` from the first transfer commit's parent). Run
  `git rebase <first-transfer-parent> --exec "uv run python tools/check_commit.py --only format" --exec
  "uv run python tools/check_commit.py --only rules"`, then the full gate on the tip. Expected: all pass.
- [ ] **Step 3: Check the design section by section** (1 to 11) against the branch and record, for the
  hand-over, anything delivered differently: at least the plan decisions of the Global Constraints
  (`TransferBuffers` with `material_count`, one form per stage, and the builders, the `sizeof`-based memory figures, `ingredient_volume` as current volume,
  the grouping rule, `kRunOnCallFailed` for allocation failures, `kMaxFitRows`, the "touches a solved cell"
  reading of Task 12), and from Foundation as built: every GPU half in a private `<name>_gpu.h` returning
  `RunOnCallStatus`, the kernel bodies in `<name>_steps.h` tested on the CPU over every thread block and thread, the
  same-bits promise of `ComputeCellsFromParticles`.
- [ ] **Step 4: `docs/gpu-runs.md`.** Add the row: module transfer, the tip commit, no date, no GPU,
  result `compiled, not run`, notes listing the `gpu` tests to run at the end of the programme:
  `ParticlesToGrid` (5), `SaveGridBeforeForces`/`SaveGridAfterPressure` (4), `GridToParticles` (2),
  `ComputeCellsFromParticles` (1), `MaterialSets` (2), and the `GridToParticles` and `SaveGrid...` repeat runs
  named in Tasks 13 and 15. Run the gate; commit as `Record transfer as compiled, not run`.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the transfer commits; the gate's output; every
  new public name with its reason; the list from Step 3; the statement that no transfer device code has run.

## Deep liquid without particles (later; needs review)

**Amended 2026-11-15 (A64):** this group is excluded from the module's done-criteria. Its owner is the group of the
same name in the cell resize and batches plan (`2026-11-14-rebuild-11-cell-resize-and-batches.md`), opened by "an
independent review of this design comes first" and gated by `GridOnlyRegion, StandingWave_KeepsAmplitude` and
`GridOnlyRegion, ParticleBandEdge_StaysStable`.

### Task L1: Faces written by the grid-only region

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/transfer/grid_from_particles.h`, `grid_snapshot_cpu.cc`, `grid_snapshot_steps.h`,
  `grid_snapshot_test.cc`

**Interfaces:**
- Produces: `FaceValueKind::kFromGridOnlyRegion`, appended as the fifth value (design 3, "not in the first version
  (A64)"): the grid-only region's velocity, written by `cell_resize::ApplyGridOnlyRegionToGrid` in the set holding
  the liquid; a source face read directly like `kFromParticles`.
- Consumes: the cell resize later group's `ApplyGridOnlyRegionToGrid`.

- [ ] **Step 1: Failing test** (label `cpu`; the GPU snapshot test of Task 13 gains the same faces, label `gpu`):
  `TEST(SaveGridAfterPressure, GridOnlyRegionFace_IsASourceAndKeepsItsKind)`: `kFromGridOnlyRegion` faces beside
  an unfilled face, one of them `was_solved`: read directly (`kNotNeeded`), used as a source, kind unchanged after
  pressure; exact kinds.
- [ ] **Step 2: Implement:** `is_source` 1 for `kFromGridOnlyRegion` before forces and after pressure; the kind is
  kept after pressure even when solved.
- [ ] **Step 3: Gate. Commit.** Subject: `Read faces written by the grid-only region as sources`.
