# Rebuild Solids Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the solids module of the rebuild: fixed, driven and free solids under try and commit, the solids
seen by the grid (`transfer::SolidsAtFaces`, `pressure::SolidCoverage`, `grid_edge_is_open_to_sky`), one motion
map and its transpose for pressure, the blocking-material faces the liquid pressure solve holds (A18), the
material band and free-solid inputs of the viscous solve and their write-back (A26, A27), contact
reactions summed per solid, `CollideSolids` between material sets
and at solids, `DriveWaves`, plate sides and cut work, and the exchange check, on the CPU and (compiled, not
run) on the GPU.

**Architecture:** Folder `native/fluidity/solids/`, namespace `fluidity::solids`. `Solids` and `SolidsTry` live
in CPU memory on both processors; each per-place operation has one public CPU-only header and private
`_cpu.cc` / `_gpu.cu` files that call the same shared formula headers (`solid_state.h`, `solid_dynamics.h`,
`motion_map_math.h`, `material_contact.h`). Every per-solid or scene total is one partial per cell block in the
grid's fixed block order, then `math::TreeSum`; contact reactions are ordered by `particle_id`. Working buffers
live in one `SolidsBuffers` built once.

**Tech Stack:** C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest), uv, clang-format and
clang-tidy 23.1 (the versions `pyproject.toml` pins), Docker via colima, Typst + CeTZ for the method paper. Nothing new.

**Spec:** `docs/superpowers/specs/modules/08-solids-design.md` ("the design"; its section 3 is the contract) and
the programme design `docs/superpowers/specs/2026-11-13-production-rebuild-design.md` (sections 6, 7, 8, 8a),
both in the `main` checkout, read-only. Also read there: `modules/owner-answers.md`, `modules/shared-interface.md`
(sections c to e), `modules/plan-decisions.md` (it wins over the design), section 3 of the contact, grid,
particles, transfer, pressure, viscosity and materials designs.

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
  `RunOnCallStatus`, which the CPU half maps with `ToBufferError` and then into `SolidsFailure` through one mapping
  function (`ToSolidsFailure`, Task 13), and launches, waits and holds temporary memory only through
  `base/gpu_launch.cuh` (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`). Every GPU function is
  a thin loop over `FLUIDITY_CPU_AND_GPU` per-item or per-thread functions in a shared header; a `cpu` test runs
  them for every thread block and thread of the launch, in launch order and in reverse, against the CPU loop (the
  main evidence until the hardware run). Working buffers: `SolidsBuffers` owns an `OrderBuffers order_buffers`, built
  once by `BuildSolidsBuffers` and passed to `BuildStableOrder`, `ComputeExclusivePrefixSum` and
  `BuildGroupsByDestination` in every step; `BuildSolidsMemoryEstimate` counts its bytes with
  `ComputeOrderBuffersSize`. (Solids makes no `ComputeTreeSum` call: every total is block partials then
  `math::TreeSum`, on the GPU one thread per solid and component around `math::TreeSumOf`, so it owns no
  `SumBuffers`.) Errors: a failed check writes nothing; after `kRunOnCallFailed` the outputs are unspecified and the
  run stops; `kOutOfMemory` comes only from building forms
  (`SolidsError::kOutOfMemory`, returned only by `BuildSolidsBuffers`; reconcile item 25). Buffers and arrays passed
  together live on one processor: a precondition (`Requires:`, debug assertion), not an error value (A30); each base
  order, prefix sum and grouping has one form, which takes its buffers (A31). **Amended 2026-11-14
  (A30|A31|reconcile).** Exactness: discrete outputs equal on CPU and GPU,
  floating-point within design 6 unless a header promises the same bits unless NaN. Tests link
  `fluidity_base_test_support` (`ToBits`, `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`,
  `CheckSameBitsOrBothNaN`) and the module's helper libraries made with `fluidity_add_test_library`, named in
  `LINKS`; `gpu` tests call `WriteResultDigest` on their main outputs. Labels: `cpu` everywhere, `no_gpu` only for
  behaviour on a machine without a GPU, `gpu` only on hardware. Every target is created through the helpers of
  `cmake/fluidity_targets.cmake` and every source file is listed in the folder's `CMakeLists.txt`.
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

Solids' own constraints (values from the design; never change one without the owner):

- **Section 3 is the contract.** Implement its declarations exactly. Additions are marked **Plan decision** in
  the task that makes them and repeated in the hand-over.
- **Errors.** Every fallible host function returns `std::expected<T, SolidsFailure>`; shared formulas return
  `SolidsError` (`kNone` on success). The failure reported is the smallest `solid_id`, then `particle_id`, then
  `face`; on the GPU chosen by an integer minimum on that key. On failure no output of the call is valid. A
  `BufferError`, `GridError`, `TransferError`, `PressureError`, `ContactError` or `ParticlesFailure` from below
  becomes `kRunOnCallFailed` or `kBadInput` (a size mismatch); out of memory in `BuildSolidsBuffers` is
  `kOutOfMemory` (`magnitude` = bytes asked; reconcile item 25, was `kRunOnCallFailed`).
  **Amended 2026-11-14 (Foundation as built):** a failed check writes nothing (the tests that say "bytes
  unchanged" after a check stay); after `kRunOnCallFailed` from a failed GPU call the outputs are unspecified and
  the run stops. A GPU half's `RunOnCallStatus` reaches `SolidsFailure` through `ToBufferError`, then
  `ToSolidsFailure` (Task 13).
  `kWrongGridVersion` whenever an array's `grid_version()` or `block_count()` differs from the grid's.
  `kWrongTime` whenever an array's `time` differs from the try's `step_start_time()` (or, for the end pose, the
  step end) or its `try_id` from the try's.
- **Constants:** `kSolidReachInCells = 2`, `kCoverageCapInCells = 2`, `kFaceStopReachInCells = 0.5`,
  `kMaxPlateCount = 1`, `kMaxFreeSolidCount = 16`, `max_travel_in_cells = 0.5`, `max_motion_map_entries = 2^20`,
  a key within `1e-12 * max(1, time)` of a step end counts as that end, unclear contact normal below `1e-6 w`,
  `particle_radius <= (4 - kCoverageCapInCells) * smallest_cell_width`, plate minimum
  `(|n.x| + |n.y| + |n.z| + max(|n.x|, |n.y|, |n.z|)) * largest_cell_width`, turn per step
  `<= contact::kMaxTurnPerStep`, `ExchangeTolerances::relative = 1e-10`, `keys_per_period = 32`.
- **Order of sums (design 3 preamble, 5.5 step 5, 5.6, 5.7 step 1, 5.8).** Per solid (or per scene) and cell
  block: the block's items in their stated order (motion map entries in (cell, face, corner) order; faces in
  ascending face place), summed with `math::TreeSum` per component; then `math::TreeSum` over the block
  partials in the grid's fixed block order. Contact reactions: stable order by `particle_id` over the whole
  try, grouped by solid, `math::TreeSum` per solid. No floating-point atomics.
- **Sign of work (Plan decision, design 3.6 to 3.9 disagree with themselves).** Every work in this module is
  the energy a solid gives the material: positive when a paddle pushes liquid. With `push` the push **on** the
  solid (N s) and `turning_push` about its step-start origin: pressure work given `= -(push . v +
  turning_push . w)`; `PressureOnSolids::flow_work = dt * sum_i p_i * (solid volume rate into cell i) = -sum
  over solids of (push . v + turning_push . w)`; `ContactOnSolids::work` the same expression from the contact
  pushes; `ExchangeTotals::driven_work` sums both for driven solids. The identity tests of 5.5 step 2 are kept
  as written (`sum p_i flow_out_i dt = push . v + turning_push . w`, `flow_out` as design 5.5 step 1).
- **Material sets (A8).** Every liquid is one set; gas markers are never passed to `CollideSolids` or
  `ApplyContactReactions`; a blocking material's particles are read only through `blocking_materials`.
- **Test labels and helpers:** **Amended 2026-11-14 (Foundation as built):** shared-header tests and CPU operation
  tests are `fluidity_add_test(... LABEL cpu LINKS fluidity_base_test_support ...)` (the container's `cuda` step
  runs `cpu` tests too; the retired `fluidity_add_shared_test` is not used); device tests `LABEL gpu`; every
  operation with a `_gpu.cu` has `<Operation>, OnGpuWithoutAGpu_IsRunOnCallFailed` (`LABEL no_gpu`). A
  known-open test
  (form amended 2026-11-14, Foundation as built: `fluidity_add_test` takes one label and discovers every test of
  its program under it, so a failing test there would fail the `cpu` gate) is written with GoogleTest's `DISABLED_`
  prefix, so discovery registers it as disabled and the gate skips it, and is registered a second time, inside
  `if(BUILD_TESTING)`, by `add_test(NAME <Suite>.<Name> COMMAND <program> --gtest_filter=<Suite>.DISABLED_<Name>
  --gtest_also_run_disabled_tests)` with `set_tests_properties(<Suite>.<Name> PROPERTIES WILL_FAIL TRUE LABELS
  known_open)` (`add_test` makes no C++ target, so the helper rule holds; the container's `cpu|no_gpu` run skips
  it), and a line in `docs/known-open-physics.md`.
- **Inputs from later modules are hand-built in tests** (gas, step, cell resize, I/O). No later module is stubbed.
- **Earlier plans' decisions this plan relies on (`plan-decisions.md`):** `RunOn` is declared alone in the shared
  header `base/run_on.h`, so `solid_settings.h` (shared) includes that header for `SolidsSettings::run_on`;
  `CheckCanRunOn` lives in the CPU-only `base/check_can_run_on.h`, included only by `_cpu.cc` / `.cc` files.
  Transfer arrays are built with `transfer::BuildSolidsAtFaces`, `BuildGridFromParticles`, `BuildGridSnapshot`,
  `BuildCellsFromParticles`; every transfer stage a test or loop here calls uses the form that takes a
  `transfer::TransferBuffers` (from `BuildTransferBuffers`) as its last argument. `pressure::OpenSeaSettings`
  carries `contact::DomainBox domain_box`. `particles::ParticleFields` carries `run_on`, and its pointers point
  at particle 0 of the material with whole-material `group_begin` indexes.
- **CPU against GPU (design 6):** owners, sides, `outside_solid`, counts, open-to-sky marks exact; a double `a`
  against `b` passes when `|a - b| <= floor + 0x1p-40 * max(|a|, |b|)`, `floor` = contact tolerance for
  distances, `0x1p-40 *` largest speed for velocities, `0x1p-40 *` (largest pressure `* w^2 * dt`) for pushes.
- **Code comments** number long bodies `// Step N:` with the step numbers of the paper's procedures
  (`<proc-free-solid-step>` 1 to 4 = design 5.3; `<proc-build-motion-map>` 1 to 6 = 5.5; `<proc-collide-solids>`
  1 to 6 = 5.7; `<proc-apply-plate-sides>` 1 to 3 as Task 2 numbers them).
- **Plan decisions used by several tasks** (each declared in the task named):
  1. `pressure::FreeSolidPush::solid_index` is the index of the free solid among the free solids of
     `SolidsTry::solids()`, in that order; `ToFreeSolidInverseInertia` gives two `Mat3` per free solid in the
     same order (Task 20).
  2. `BuildSolidsBuffers(const SolidsSettings& settings, std::size_t max_particle_count, std::uint32_t
     max_cell_blocks)`: the design's two-argument form cannot size per-block arrays (Task 13).
  3. `std::expected<SolidCoverageValues, SolidsFailure> BuildSolidCoverageValues(RunOn run_on, std::uint32_t
     max_cell_blocks)` builds the step-owned coverage arrays (Task 13).
  4. `Solids` copies its settings, path keys and blocking materials; `settings()` points into its copies
     (Task 9). `try_id = (state_version << 32) | tries_since_commit`, the counter a `mutable` member of `Solids`
     that `BuildTry` raises and `CommitTry` resets; it is not part of the state or the snapshot (Task 10).
  5. A `SolidsTry` remembers the `state_version` it was built from; `CommitTry` with another fails `kWrongTime`
     (Task 11). `ApplyPressureToSolids` fails `kSolidMovesTooFar` when a free solid's velocity after pressure
     breaks the travel or turn limit (Task 21).
  6. `MotionMap` is a move-only view into `SolidsBuffers` with `try_id`, `time`, `grid_version` and the buffers'
     build number; a map from an earlier build on the same buffers gives `kWrongTime` (Task 20).
  7. `CollideSolids` working buffers are one `double` per face place per material set, for at most
     `blocking_materials.size() + 1` sets; more sets give `kBuffersTooSmall` (Task 13, Task 25).
  8. `DriveWaves` has a private `drive_waves_gpu.cu`, declared in `drive_waves_gpu.h`, that forms the block
     partials where the arrays live (Task 30; amended 2026-11-14, Foundation as built).
  9. **A18, blocking materials inside the liquid solve** (Task 23; replaces design decision 3 "one step late").
     Solids builds `pressure::BlockingMaterialFace` entries with `BuildBlockingMaterialFaces` (each blocking set's
     own face mass and face velocity on every face piece where it blocks liquid) before the liquid's
     `BuildPressureEquations`; after the solve `ApplyPressureToMaterials` writes the solved face velocities into
     each set's `GridFromParticles::velocity`, before `SaveGridAfterPressure`. `MaterialPushFields` and the
     particle push are removed; `ApplyPressureToSolids` stores no push owed to particles.
  10. **A26 and A27, the viscous solve sticks and carries free solids** (Tasks 23a, 23b, 23c; `plan-decisions.md`
     "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)"). In a viscous scene (stage 11)
     the material faces near the liquid are velocity unknowns of viscosity's joint solve, and so are the free
     solids' six motions. Solids builds the band with `BuildViscousMaterialFaces` (stage 10, after
     `BuildBlockingMaterialFaces`) and gives the free solids with `ToFreeSolidsForViscousSolve`. After the solve,
     `ApplySolvedMotionToSolids` replaces `ApplyPressureToSolids` and `ApplyViscousSolveToMaterials` replaces
     `ApplyPressureToMaterials` in stage 11. Stage 12 (`SolvePressure`, an inviscid liquid) is unchanged: the
     liquid slides, material faces are eliminated (A18), free solids enter as the rank-one term. No combination of
     materials is refused because of viscosity.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: Foundation
with every `base/` piece of `shared-interface.md` section (a) and `changes-to-apply.md` (`Buffer`, `CopyBuffer`,
`TreeSum`, `ComputeTreeSum`, `BuildStableOrder`, `BuildGroupsByDestination`, `memory_count.h`,
`PerParticleVelocity`); the method paper with `figure_style.typ`, `pseudocode.typ` and the earlier modules'
sections; `docs/glossary.md`, `docs/gpu-runs.md`, `docs/known-open-physics.md`; then contact, grid, particles,
transfer (with `fluidity_transfer_test_scenes`), pressure (with `fluidity_pressure_test_scenes`,
`ComputeBlockedCornerAreas`, `ComputeEquationIndexes`, `ApplyOpenSeaRelaxation`), viscosity
(`SolveViscosityAndPressure`) and materials (`AtSolidFaces`, `MaterialKind::kSnow`, `kSpringySolid`,
`OwnMotionSettings`). **Amended 2026-11-14 (Foundation as built):** Foundation as built also provides
`base/run_on_call.h` (`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, `SumBuffers` taken by
the sums, `OrderBuffers` taken by the order, prefix sum and grouping (one form each, A31), `base/test_support.h`
(library `fluidity_base_test_support`) and `fluidity_add_test_library`. Read `AGENTS.md` "Structure" (two halves,
private GPU headers, working buffers) and copy that shape, as contact's `MoveParticles` does. The gate's expected output is
seven `PASS` lines: format, rules, cpu debug, cpu release, tidy, paper, cuda.

Prove it: `uv run python tools/check_commit.py` prints seven `PASS` lines, and
`grep -n "ComputeBlockedCornerAreas\|ComputeEquationIndexes" native/fluidity/pressure/*.h` and
`grep -n "struct SolidsAtFaces" native/fluidity/transfer/*.h` each find the declaration. If not, stop.

## Review Focus

1. **A path key a hair inside a step end.** A key at `step_end - 1e-13` must count as the end (no
   `kPathJumpInsideStep`), one at `step_end - 1e-11` must not. Pinned: `BuildTry,
   KeyAtStepEndWithinRoundOff_IsNotAJump` in Task 10.
2. **A corner at equal distance from an analytic solid and a particle ball.** The analytic solid keeps the
   corner (a material wins only when strictly nearer). Pinned: `ComputeSolidCoverage,
   ParticleTiesAnalyticSolid_AnalyticKeeps` in Task 17.
3. **A free solid that is slow before pressure and fast after it.** The travel limit must be checked again
   after the solve. Pinned: `ApplyPressureToSolids, FastAfterPressure_IsSolidMovesTooFar` in Task 21.
4. **The same hits handed in different ranges and orders.** Sums must follow `particle_id`, not the call cut.
   Pinned: `ApplyContactReactions, ReactionsSplitIntoRanges_SameBytes` in Task 28.
5. **The sign of work.** A paddle pushing into liquid must report positive `flow_work` and `driven_work`.
   Pinned: `ApplyPressureToSolids, PaddlePushingLiquid_WorkIsPositive` in Task 21.

---

### Task 1: Method paper, solids that move, coverage, motion map, free solids

**Amended 2026-11-14 (A18):** the paper describes blocking materials solved with the liquid, not a one-step lag.

**Amended 2026-11-14 (A26, A27):** Step 1's text adds one paragraph: in a viscous scene the material faces within
1.5 cell widths of a liquid cell centre and the free solids' six motions are velocity unknowns of the viscous solve
(refer to viscosity's coupled terms section), so a viscous liquid sticks to snow, springy solids and free solids;
an inviscid liquid keeps the slide and the elimination; "tangential motion stays the set's own" holds for the
inviscid case only.

**Kind:** mechanical

Load `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` first.

**Files:**
- Create: `docs/method-paper/sections/solids.typ` (included after the materials section; follow the earlier
  modules if they put sections inline), `docs/method-paper/figures/motion_map.typ`, `free_solid_step.typ`,
  `driven_path.typ`, `corner_owners.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include, notation rows)

**Interfaces:**
- Produces labels `<sec-solids>`, `<sec-solid-motion>`, `<sec-solid-coverage>`, `<sec-motion-map>`,
  `<sec-free-solids>`, `<eq:driven-motion>`, `<eq:free-solid>`, `<eq:motion-map>`, `<eq:motion-map-transpose>`,
  `<fig-motion-map>`, `<fig-free-solid-step>`, `<fig-driven-path>`, `<fig-corner-owners>`,
  `<proc-free-solid-step>`, `<proc-build-motion-map>`.

- [ ] **Step 1: Text** from design 5.1 to 5.5 and 8, decided vocabulary (design 9; never "rigid body",
  "impulse", "wetted", "body"): one solid array per try and its time stamp; driven paths, keys as jumps, the two
  step limits (`<eq:driven-motion>`); free solids, the four steps and the contact lag (`<eq:free-solid>`, mark
  the equations of motion "citation needed"); faces, corners, the particle-ball search and why 27 blocks
  suffice; the motion map, the transpose and the work identity term by term (`<eq:motion-map>`,
  `<eq:motion-map-transpose>`, cite [21] Batty 2007 and [22] Ng 2009); blocking materials solved with the liquid
  (A18): each blocking set's face mass enters the pressure solve on the pieces it blocks (`dt A^2 / m`, refer to
  pressure's `<sec-coupled-solve>`), the solved face velocity is written back to the set in the same step, and
  tangential motion stays the set's own (A8); limits carried (piston receiving volume, air-pocket and gas
  pressure push no solid, one owner per corner). State the sign of work
  of the Global Constraints in one sentence.
- [ ] **Step 2: Figures** (design 8) with `figure_style.typ`: `<fig-motion-map>`, `<fig-free-solid-step>`,
  `<fig-driven-path>`, `<fig-corner-owners>`. Each caption states the one thing to see.
- [ ] **Step 3: Pseudocode** `<proc-free-solid-step>` (steps 1 to 4 of 5.3) and `<proc-build-motion-map>` (steps
  1 to 6 of 5.5), inputs, outputs, "fails when".
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.
  Expected: exit 0, no warning. Look at every new page; fix overlaps and clipped labels. Run the gate.
- [ ] **Step 5: Commit.** Subject: `Describe solids that move and the one motion map`.

### Task 2: Method paper, contact between materials, waves, cutting, checks

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/solids.typ`
- Create: `docs/method-paper/figures/plate_sides.typ`, `material_contact.typ`

**Interfaces:**
- Produces `<sec-collide-solids>`, `<sec-drive-waves>`, `<sec-cutting>`, `<sec-solid-checks>`,
  `<eq:material-contact>`, `<eq:plate-thickness>`, `<eq:cut-work>`, `<fig-plate-sides>`,
  `<fig-material-contact>`, `<proc-collide-solids>`, `<proc-apply-plate-sides>`.

- [ ] **Step 1: Text** from 5.6 to 5.10: reactions ordered by id; the rule at solids and the pair rule
  (`<eq:material-contact>`, cite [19] Stomakhin 2013, [18] Stomakhin 2014; "citation needed" for contact
  between two materials on a grid); waves as a forced system (cite [20] Jacobsen 2012); the unbounded plate, side
  history, reconnection, the thickness rule with its closed form 2 to 2.74 widths (`<eq:plate-thickness>`, cite
  [3] Hu 2018); cut work (`<eq:cut-work>`); the four checks.
- [ ] **Step 2: Figures** `<fig-plate-sides>`, `<fig-material-contact>` (design 8).
- [ ] **Step 3: Pseudocode** `<proc-collide-solids>` with six steps numbered as design 5.7 (the design's
  "five steps" in section 8 is a miscount; plan decision), and `<proc-apply-plate-sides>`: 1 skip particles that
  already have a side; 2 side of the start position at the step-start pose; 3 inside the thickness fails.
- [ ] **Step 4: Run** typst, look at the pages, run the gate. **Commit.** Subject:
  `Describe contact between materials, waves, cutting and the solid checks`.

### Task 3: Errors, settings, build lines and glossary

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the folder's `CMakeLists.txt` uses only the helpers; tests are
`LABEL cpu` and link `fluidity_base_test_support`.

**Files:**
- Create in `native/fluidity/solids/`: `solids_error.h`, `solids_error_message.h`, `solids_error_message.cc`,
  `solid_settings.h`, `solids_error_test.cc`, `CMakeLists.txt` (headers by `fluidity_add_headers`, `.cc` by
  `target_sources(fluidity PRIVATE ...)`, `.cu` by `fluidity_add_gpu_sources`, tests by `fluidity_add_test(...
  LINKS fluidity_base_test_support)`; every later task adds its files here, so no source is left out of the build)
- Modify: the CMake file that lists module folders; `docs/glossary.md`

**Interfaces:**
- Produces design 3.1 and 3.2 exactly: `SolidsError` (18 values with `kOutOfMemory`, reconcile item 25), `SolidsFailure`, `ToMessage`, `ToName`,
  `SolidMotionKind`, `PathKey`, `AfterPath`, `DrivenPath`, `FreeMotionLimits`, `FreeSolidSettings`,
  `SolidSettings`, `MaterialTheLiquidCannotEnter`, `SolidsSettings`, `kSolidReachInCells`,
  `kCoverageCapInCells`, `kFaceStopReachInCells`, `kMaxPlateCount`, `kMaxFreeSolidCount`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the shared part and the messages):

  ```cpp
  TEST(ToName, EveryError_IsLowerSnakeCaseWithoutK)  // all 17; kPlateTooThinForLiquid -> "plate_too_thin_for_liquid",
                                                     // kRunOnCallFailed -> "run_on_call_failed"; all distinct
  TEST(ToMessage, EveryError_IsDistinctAndNotEmpty)
  TEST(SolidsFailure, Default_HasNoIds)              // error kNone, solid_id 0, particle_id 0, face == grid::kNoIndex
  TEST(SolidsSettings, Defaults_AsDeclared)          // gravity {0,0,-9.81}, max_travel_in_cells 0.5, 1u << 20
  ```
- [ ] **Step 2: Run.** Expected: compile error. **Step 3: Implement.** Messages say what the user can do
  ("Two plates are in the scene; only one is supported. Remove one."). `Read:` lines cite `<sec-solids>`.
- [ ] **Step 4: Glossary rows** (`| Say | Plain meaning | Never say |`): `fixed`, `driven`, `free` (never
  "rigid body", "prescribed", "animated obstacle"), `own axes`, `push` (N s; never "impulse"), `turning_push`
  (never "angular impulse"), `motion map`, `corner owner`, `blocking material`, `wave paddle` (never "piston"),
  `driven_work`, `flow_work`, `swept_volume`, `cut_work`, `plate`.
- [ ] **Step 5: Run** the gate. **Commit.** Subject: `Add the solids module's errors and settings`.

### Task 4: Orientation and the rotation vector

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/solids/solid_state.h`, `solid_state_test.cc`

**Interfaces:**
- Produces design 3.3 `Orientation`, `ToAxes`, `ToOrientation`, `ComputeTurnedOrientation`,
  `ComputeRotationVector`, `SolidState`, all `FLUIDITY_CPU_AND_GPU`.
- **Plan decision:** `ComputeTurnedOrientation(start, v)` is `q(v) * start` (the turn is in world axes, as
  `main`'s `SampleRigidBodyPose3D`, `rigid_body_3d.cc:833-859`), then renormalised. `ToOrientation` uses the
  largest-diagonal branch (Shepperd) and returns `w >= 0`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`). Random draws in every task use
  `std::mt19937_64(seed)` with `(x >> 11) * 0x1.0p-53` (as transfer's test scenes):

  ```cpp
  TEST(ToAxes, ThenToOrientation_GivesSameRotation)        // 1,000 random unit quaternions, seed 8: axes equal
                                                           // within 1e-15; q and -q give the same axes
  TEST(ComputeTurnedOrientation, QuarterTurnAboutZ_MatchesMathRotation)  // ToAxes(result) == math::Rotation(v)
                                                           // * ToAxes(start), random start; 1e-15
  TEST(ComputeRotationVector, TurnThenMeasure_GivesTheTurn)  // design 7: 1,000 random o, v with |v| < pi, seed 9:
                                                           // equals v within 1e-13 rad
  TEST(ComputeRotationVector, SameRotationBothSigns_IsZero)  // (o, o) and (o, -o) give {0,0,0} exactly
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement**; `ComputeRotationVector` takes the shortest turn
  (`q_rel = to * conj(from)`, flipped to `w >= 0`, angle `2 atan2(|vector|, w)`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the orientation of a solid and its rotation vector`.

### Task 5: Contact between two materials at one sample

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/solids/material_contact.h`, `material_contact_test.cc`

**Interfaces:**
- Produces design 3.8 `MaterialContactInputs`, `MaterialContactResult`, `ComputeContactBetweenMaterials`
  (shared). Ported from `git show main:native/include/fluidity/base/distinct_body_contact_3d.h` lines 47-120
  (5.7 step 5), without the id checks.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), masses 2 and 6 kg, first velocity `{-2, 0, 3}`,
  second `{2, 0, 0}`, `normal_toward_first = {1, 0, 0}` (main's `m5_contact_3d_cases.h:9-29` turned into Z-up);
  1e-14 relative:

  ```cpp
  TEST(ComputeContactBetweenMaterials, HeadOn_SharesNormalSpeed)   // friction 0: approach_speed -4,
                                                                   // push_on_first {6,0,0}; both x speeds 1
  TEST(ComputeContactBetweenMaterials, Separating_NoPush)          // normal {-1,0,0}: approach 4, push zero
  TEST(ComputeContactBetweenMaterials, Stick_SameSidewaysVelocity) // friction 1: both z velocities 0.75
  TEST(ComputeContactBetweenMaterials, Slide_MatchesMain)          // friction 0.25: push {6, 0, -1.5};
                                                                   // first after {1,0,2.25}, second {1,0,0.25}
  TEST(ComputeContactBetweenMaterials, AnyCase_KeepsMomentumAndNoEnergyGain)  // 1,000 random inputs, seed 5
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `shared_mass` (design 9). **Step 4: Run** the gate.
- [ ] **Step 5: Commit.** Subject: `Add the contact rule between two materials`.

### Task 6: Free-solid formulas

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/solids/solid_dynamics.h`, `solid_dynamics_test.cc`

**Interfaces:**
- Consumes: Task 4. Produces design 3.3 `InverseInertia`, `ComputeInverseInertia`,
  `ComputeVelocityBeforePressure`, `ComputeMotionAfterPush` (shared). `ComputeDrivenMotion` comes in Task 7.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeInverseInertia, GuidedAlongZ_OnlyZMoves)     // design 7: limits z only: every row of per_mass
                                                           // but zz, and all of per_turning_inertia, zero; exact
  TEST(ComputeInverseInertia, TurnedSolid_IsRotatedDiagonal)  // R diag(1/I) R^T, R from a random orientation; 1e-15
  TEST(ComputeMotionAfterPush, PushOfMassTimesSpeed_AddsSpeed)
  TEST(ComputeVelocityBeforePressure, SpinAboutPrincipalAxis_KeepsAngularVelocity)  // exact
  TEST(ComputeVelocityBeforePressure, NotFiniteVelocity_IsNotFinite)
  TEST(ComputeVelocityBeforePressure, MainRigidBodyCase_SameVelocities)
  ```
  The last is the comparison with `main` (supporting): `test_rigid_body_3d.cc:442-470` turned into Z-up: mass 2,
  inertia `{2, 3, 4}`, orientation `{w = sqrt(2)/2, vector = {0, -sqrt(2)/2, 0}}`, angular velocity
  `{1, -3, 2}`, `steady_force = {4, 0, 0}`, gravity zero, `dt = 0.125`; then `ComputeMotionAfterPush` with
  `push = 0`, `turning_push = dt * {0, 8, 0}` (the force at arm `{0, 0, 2}`). Expected linear `{0.25, 0, 0}`,
  angular `{0.5, -2.8125, 2.1875}`, 1e-13 relative.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.3 step 1 from `git show main:native/src/
  rigid_body_3d.cc` lines 781-831 (no torque stored; forbidden motions set to zero).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the velocity of a free solid before and after a push`.

### Task 7: Driven paths and the wave paddle's path

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/solids/solid_dynamics.h`, `solid_dynamics_test.cc`
- Create: `native/fluidity/solids/drive_waves.h` (only `WavePaddleSettings`, `BuildWavePaddlePath` now),
  `wave_paddle_path.cc`, `wave_paddle_path_test.cc`

**Interfaces:**
- Produces `ComputeDrivenMotion`, `WavePaddleSettings`, `BuildWavePaddlePath` exactly.
- **Plan decisions:** the orientation at key `k + 1` is `ComputeTurnedOrientation(orientation at k,
  angular_velocity_until_next_k * (t_{k+1} - t_k))`, from `rest_pose` at key 0; before the first key the solid
  rests at key 0 with zero motion; after the last key zero motion. A step that contains a key returns
  `kPathJumpInsideStep` (key tolerance of the Global Constraints). Paddle stroke `origin(t) = rest_origin +
  travel_direction * (stroke / 2) * r(t) * sin(2 pi t / wave_period)`, `r(t) = min(1, t / ramp_duration)` (1 when
  the ramp is 0); keys at `t_k = k * wave_period / keys_per_period` up to the first key at or after `end_time`.

- [ ] **Step 1: Failing tests:**

  ```cpp
  TEST(ComputeDrivenMotion, TwoKeys_ConstantVelocity)   // design 7: keys 0 s and 2 s, origins 1 m apart along x:
                                                        // velocity 0.5 m/s; pose at 1 s halfway; 1e-15 relative
  TEST(ComputeDrivenMotion, BeforeFirstKey_RestsAtFirstKey)
  TEST(ComputeDrivenMotion, ThreeKeysTurning_OrientationBySuccessiveTurns)  // 1 rad/s about z then about x; 1e-14
  TEST(ComputeDrivenMotion, KeyInsideStep_IsPathJumpInsideStep)
  TEST(BuildWavePaddlePath, Sine_KeysOnTheCurve)        // design 7: stroke 0.2 m, period 2 s, ramp 1 s, end 6 s:
                                                        // every key on the formula above within 1e-15
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.2; the two-key path idea from
  `git show main:blender/fluidity/scene_inputs.py` lines 62-90.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add driven paths and the wave paddle's path`.

### Task 8: The plate thickness rule

**Kind:** numerics

**Files:**
- Create: `native/fluidity/solids/plate_sides.h` (only `ComputeMinPlateThickness` now), `plate_sides_cpu.cc`,
  `plate_thickness_test.cc`

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeMinPlateThickness, AxisAlignedAndDiagonal_ClosedForms)  // {0,0,1}: 2 w; {1,1,1}/sqrt 3: 4/sqrt(3) w
  TEST(ComputeMinPlateThickness, AnyNormal_NoCellSeesBothSides)       // design 7: 1,000 normals (seed 11), unit
      // cells, plate of exactly that thickness at a random offset in [0,1)^3: some cell of the 8^3 lattice
      // around the plate has 8 corners with |signed distance| <= half thickness, and no cell has a corner with
      // distance > half thickness and another < -half thickness; exact
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.9. **Step 4: Run** the gate. **Commit.** Subject:
  `Add the plate thickness that keeps the two sides apart`.

### Task 9: `Solids`, `BuildSolids`, `CheckSolidsCanRun`, `ComputeSolidsSummary`

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/solids/solids.h`, `solids.cc`, `solids_test.cc`

**Interfaces:**
- Produces `Solids` (accessors only here), `BuildSolids`, `SolidsSummary`, `ComputeSolidsSummary`,
  `CheckSolidsCanRun` exactly; Plan decision 4 (copies).
- **Plan decisions:** `BuildSolids` fails `kBadInput` on the first rule it can check without cell widths (ids,
  finiteness, mass, inertia, keys) in `solid_id` order. In `CheckSolidsCanRun`, `kTooManyPlates` is one entry for
  each plate after the first in `solid_id` order; `kTooManyFreeSolids` and the radius rule have `solid_id` 0.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(CheckSolidsCanRun, ThreeBrokenRules_AllReported)  // design 7: solid 1 free box mass 0, solid 2 plate of
      // thickness 2.1 w, solid 3 plate of thickness 1.9 w, has_liquid_or_gas: entries (1, kBadInput),
      // (3, kTooManyPlates), (3, kPlateTooThinForLiquid, magnitude 1.9 w, limit 2 w); without them: none; exact
  TEST(CheckSolidsCanRun, SeventeenFreeSolids_IsTooManyFreeSolids)  // one entry, magnitude 17, limit 16
  TEST(CheckSolidsCanRun, ParticleRadiusAboveBlockLimit_IsBadInput) // radius 2.0001 w: limit 2 w
  TEST(BuildSolids, IdsNotAscending_IsBadInput)
  TEST(BuildSolids, CallerSpansFreed_SettingsStillReadable)        // the keys' vector destroyed after the call
  TEST(ComputeSolidsSummary, TurnedBoxAndPlate_FlagsSet)           // has_inclined_solid, has_plate, plate_id
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject:
  `Add the scene's solids and their scene checks`.

### Task 10: `BuildTry` and the step duration limit

**Kind:** numerics

**Files:**
- Create: `native/fluidity/solids/solids_try.h`, `solids_try.cc`, `solids_try_test.cc`

**Interfaces:**
- Consumes: Tasks 4, 6, 7, 9; `contact::Solid`, `ToSolid`, `ComputeSolidPose`, `ComputeDistanceToSolid`,
  `kMaxTurnPerStep`. Produces `SolidsTry`, `BuildTry`, `ComputeStepDurationLimit`; Plan decision 4 (`try_id`).
- **Plan decisions:** travel radius is the half diagonal `|half_size|` of a box and 0 otherwise; a free solid's
  limit in `ComputeStepDurationLimit` uses its committed velocities; `BuildTry` checks velocity before pressure;
  `kSolidsTooClose` when a free solid's distance to another solid (sampled at the box's 8 corners and centre
  against the other's `ComputeDistanceToSolid`, and the reverse) is below the largest cell width.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildTry, KeyInsideStep_IsPathJumpInsideStep)     // design 7: key 0.3 s, step 0..0.5: limit 0.3 within 1e-15
  TEST(BuildTry, KeyAtStepEndWithinRoundOff_IsNotAJump)  // Review Focus 1: key at 0.5 - 1e-13 passes;
                                                         // 0.5 - 1e-11 fails
  TEST(BuildTry, TurningKey_PoseMatchesContact)          // design 7: 1 rad/s about z for 0.5 s: ToAxes of the pose a
                                                         // commit would give equals contact::ComputeSolidPose(.., 0.5);
                                                         // 1e-14 (checked again after commit in Task 11)
  TEST(BuildTry, TooFast_IsSolidMovesTooFar)             // driven 1 m/s, smallest w 0.1, step 0.06: limit 0.05
  TEST(BuildTry, FreeBoxHalfACellFromWall_IsSolidsTooClose)
  TEST(BuildTry, FixedDrivenFree_ReportsReactionAndFrictionScaleSet)  // reports_reaction false only for fixed
  TEST(BuildTry, PathEndedLeavingScene_SolidNotInTry)
  TEST(BuildTry, TwoTriesOfOneState_DifferentTryIds)
  TEST(ComputeStepDurationLimit, NextKeyAndTravel_SmallerOfTheTwo)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.1 to 5.3 step 1; travel rule ported from
  `git show main:native/src/liquid_substep_3d.cc` lines 2738-2741.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Build one try of the solids for a step`.

### Task 11: `CommitTry` and `RejectTry`

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/solids/solids_try.h`, `solids_try.cc`, `solids_try_test.cc`

**Interfaces:**
- Produces `CommitTry`, `RejectTry`; Plan decision 5 (state version). Totals stay empty until Tasks 21 and 28
  fill the try.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeVelocityBeforePressure, FreeFall_GainsGTimesDt)  // design 7: free box, no liquid, dt 0.01: one try
      // v.z == -9.81 dt; 100 BuildTry + CommitTry: z == z0 - g t^2/2 - g t dt/2; 1e-13 relative
  TEST(CommitTry, NoTorque_KeepsTurningMomentum)  // design 7: inertia {1,2,3}, spin {0.3, 1, -0.7} rad/s, 10^4
      // steps of 1e-4 s: world turning momentum R I R^T w constant and energy drift < 1e-6; 1e-6 relative
  TEST(CommitTry, TurningDriven_AxesEqualContactPose)          // the commit side of Task 10's turning test; 1e-14
  TEST(CommitTry, TryOfOlderState_IsWrongTime)
  TEST(RejectTry, AfterBuild_StateVersionAndStatesUnchanged)   // bytes of ReadSnapshot (Task 12) compared in Task 33
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.3 step 4 with `// Step N:` comments.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Commit or reject a try of the solids`.

### Task 12: Snapshot, rebuild and contact solids for render output

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/solids/exchange_check.h` (types `ExchangeTotals`, `ExchangeTolerances`,
  `ExchangeReport` only; the function in Task 33), `solids_snapshot.cc`, `solids_snapshot_test.cc`
- Modify: `solids.h`

**Interfaces:**
- Produces `SolidsSnapshot`, `ReadSnapshot`, `BuildSolidsFromSnapshot`, `BuildContactSolids`, `ExchangeTotals`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildContactSolids, Snapshot_PosesAndVelocities)   // design 7: free box and a driven plate that left: one
                                                          // contact::Solid, pose and motion of the box; exact
  TEST(BuildContactSolids, IdsDiffer_IsEmpty)
  TEST(BuildSolidsFromSnapshot, ReadThenBuild_SameSnapshotBytes)   // after 3 commits of a free box
  TEST(BuildSolidsFromSnapshot, IdsNotThoseOfSettings_IsBadInput)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject:
  `Save and restore the solids' state`.

### Task 13: Buffers, coverage arrays and memory estimate entries

**Amended 2026-11-14 (A30|A31|reconcile):** working buffers are `order_buffers`; out of memory is the new `SolidsError::kOutOfMemory`, returned only by `BuildSolidsBuffers` (reconcile item 25); `ToSolidsFailure` maps no processor mismatch (A30).

**Amended 2026-11-14 (A18):** the row of pushes owed to particles becomes the rows of blocking-material faces.

**Amended 2026-11-14 (A17, batched-run buffers):** a batched run builds up to three `SolidsBuffers` (whole on the
GPU, loaded on the GPU, and one in CPU memory for stage 17); the rows say so and nothing assumes one instance (step
plan Appendix A rows M9, M10).

**Amended 2026-11-14 (A26, A27):** two rows are added later, with the code that needs them: viscous material faces
(Task 23a) and free solids for the viscous solve (Task 23c). This task's rows and tests do not change.

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** `SolidsBuffers` owns an `OrderBuffers` and a copy of the try's
contact solids for the GPU halves, with their rows, and holds the one mapping `ToSolidsFailure`.

**Files:**
- Create: `native/fluidity/solids/solids_buffers.h`, `solids_buffers.cc`, `solids_buffers_access.h` (private),
  `solids_buffers_test.cc`, `solids_buffers_no_gpu_test.cc`
- Create: `native/fluidity/solids/solids_on_grid.h` (types `CornerOwner`, `SolidCoverageValues`,
  `SolidsOnGridInputs`, `ToSolidCoverage` only)

**Interfaces:**
- Produces `SolidsBuffers`, `BuildSolidsBuffers` (Plan decision 2), `BuildSolidsMemoryEstimate`,
  `BuildSolidCoverageValues` (Plan decision 3), `CornerOwner`, `SolidCoverageValues`, `ToSolidCoverage`.
- Rows (names prefixed `"solids: "`, bytes `sizeof` with a `static_assert` each, `kOneStepTry` unless stated):
  solids at faces, start and end (face place, `1 + sizeof(PlateSide) + sizeof(SolidVelocitySample)`, copies 2,
  step-owned); coverage (corner place, `8 + sizeof(CornerOwner) + 24`; face place 8; step-owned);
  `grid_edge_is_open_to_sky` (face place, 1, step-owned); motion map entries (given `max_motion_map_entries`,
  `sizeof(MotionMapEntry)`); equation of cell (cell place, 4); `FreeSolidPush` list (given
  `max_motion_map_entries`, 56); blocking-material faces (given `max_motion_map_entries`,
  `sizeof(pressure::BlockingMaterialFace)` + 4 for the count per block); per-set pushes on blocking sets (given
  `blocking_materials.size()`, 24, `MemoryWhere::kCpu`); reaction keys
  and order (particle, 16); before-relaxation copy (face place, 8); `CollideSolids` working buffers (face place,
  `8 * (blocking_materials.size() + 1)`); block partials (cell block, `80 * moving solids + 32`); per-solid
  results (given solid count, 128, `MemoryWhere::kCpu`).
- **Plan decision (A17, batched-run buffers):** `BuildSolidsBuffers` keeps its signature; with
  `settings.run_on = kCpu` every row has `where = kCpu`. Instances built from one `Solids` with different counts or
  processors may coexist (the step's whole instance with zero particles, its loaded instance, its CPU stage-17
  instance); no buffer holds a pointer into another instance, and `SolidsTry` never keeps a pointer into a
  `SolidsBuffers` across calls (Task 28).
- **Plan decision (amended 2026-11-14, Foundation as built), working buffers and solids copy.** `SolidsBuffers` holds
  `OrderBuffers order_buffers` (private, reached through `solids_buffers_access.h`), built once by
  `BuildSolidsBuffers` on `settings.run_on` with `BuildOrderBuffers`, for `max_particle_count + 64 *
  max_cell_blocks + 2 * settings.max_motion_map_entries` items: the hits ordered by `BuildStableOrder` (Task 29,
  at most one per particle), the per-cell and per-block counts summed by `ComputeExclusivePrefixSum` (Tasks 20, 23,
  23a, 24) and the entries grouped by `BuildGroupsByDestination` (Task 24, at most `2 * max_motion_map_entries`).
  The sum is a bound that never sits below the largest call and lets the rows be counted per particle and per cell
  place. It also holds `Buffer<contact::Solid> solids_copy` (`settings.solids.size()` entries) that every GPU half
  fills with one `WriteBuffer` per call, so no GPU half allocates. Rows, all `kWholeRun`, `kRunOn` (`kCpu` when
  `settings.run_on = kCpu`): `"solids: order buffers per particle"` (particle,
  `ComputeOrderBuffersSize(run_on).bytes_per_item`), `"solids: order buffers per cell place"` (cell place, the same
  bytes), `"solids: order buffers per motion map entry"` (given `2 * max_motion_map_entries`, the same bytes),
  `"solids: order buffers fixed bytes"` (given `fixed_bytes`, 1 byte), `"solids: solids copy"` (given
  `settings.solids.size()`, `sizeof(contact::Solid)`). `BuildSolidsMemoryEstimate(settings)` keeps its signature
  (it already has `run_on` and the motion map limit; the per-particle and per-cell rows take their counts from
  `MemoryCounts`). Solids owns no `SumBuffers` (no `ComputeTreeSum` call; Global Constraints).
- **Plan decision (amended 2026-11-14, Foundation as built), one mapping.** In `solids_buffers_access.h`:
  `SolidsFailure ToSolidsFailure(BufferError error, std::uint64_t bytes_asked);` (`kOutOfMemory` ->
  `kOutOfMemory` with `magnitude` the bytes asked, reachable only from `BuildSolidsBuffers`; `kSizeMismatch` ->
  `kBadInput`; `kRunOnCallFailed` -> `kRunOnCallFailed`). **Amended 2026-11-14 (A30|reconcile):** no processor
  mismatch is mapped (buffers on one processor are a precondition, A30); out of memory is `kOutOfMemory`
  (reconcile item 25). Every GPU half's `RunOnCallStatus` goes through `ToBufferError`, then
  this function.

- [ ] **Step 1: Failing tests:**

  ```cpp
  // LABEL cpu
  TEST(SolidsTypes, Sizes_AreThePlannedOnes)     // CornerOwner 16, MotionMapEntry 24, SolidVelocitySample 72
  TEST(BuildSolidsMemoryEstimate, SumEqualsAllocatedBytes)   // 10 blocks, 1000 particles, 2 blocking materials
      // (amended, Foundation as built: the order buffers rows equal order_buffers.byte_count())
  TEST(BuildSolidsBuffers, ZeroBlocks_IsBadInput)
  TEST(BuildSolidsMemoryEstimate, CpuInstance_EveryRowOnCpu)   // A17: settings.run_on kCpu: every row where kCpu
  TEST(BuildSolidCoverageValues, OnCpu_HasRoomForBlocks)
  // solids_buffers_no_gpu_test.cc, LABEL no_gpu
  TEST(BuildSolidsBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement**; `BuildSolidsBuffers` allocates by walking its own rows.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the solids buffers and their memory estimate entries`.

### Task 14: Test scenes for the solids tests

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the library is made with `fluidity_add_test_library` and linked by
tests through `LINKS`; `ReadAllBytes` reads through `ReadAll` of the test support.

**Files:**
- Create: `native/fluidity/solids/solids_test_scenes.h`, `solids_test_scenes.cc`, `solids_test_scenes_test.cc`
  (test-only library: `fluidity_add_test_library(fluidity_solids_test_scenes SOURCES solids_test_scenes.cc LINKS
  fluidity_pressure_test_scenes fluidity_transfer_test_scenes fluidity_base_test_support)`; every solids test that
  uses a scene names it in `LINKS`; never linked by `fluidity`)
- Modify: `native/fluidity/solids/CMakeLists.txt`

**Interfaces:**
- Produces (**Plan decision**, `namespace fluidity::solids::testing`, CPU only):

  ```cpp
  class SceneSolids {            // Owns settings and keys; settings() gives a SolidsSettings pointing into them.
   public:
    SceneSolids& AddFixedBox(std::uint32_t id, Vec3 center, Vec3 half_size, Mat3 axes = kIdentity);
    SceneSolids& AddTank(std::uint32_t id, contact::DomainBox box, SolidMotionKind kind);  // kFixed or kDriven
                                 // (kDriven: one key, never moves, so its reactions are reported)
    SceneSolids& AddDrivenFlatSolid(std::uint32_t id, Vec3 point, Vec3 normal, Vec3 velocity, double end_time);
    SceneSolids& AddDrivenPlate(std::uint32_t id, Vec3 center, Vec3 normal, double half_thickness, Vec3 velocity,
                                double end_time, AfterPath after);
    SceneSolids& AddFreeBox(std::uint32_t id, Vec3 center, Vec3 half_size, double mass, Vec3 inertia,
                            FreeMotionLimits limits = {});
    SceneSolids& AddBlockingMaterial(std::uint32_t material_index, double particle_radius);
    SolidsSettings settings(RunOn run_on) const;
  };
  // transfer::BuildSolidsAtFaces(grid's run_on, grid block count), then grid::ApplyGridVersion to `grid`.
  transfer::SolidsAtFaces BuildBoundSolidsAtFaces(const grid::StaggeredGrid& grid);
  SolidCoverageValues BuildBoundCoverage(const grid::StaggeredGrid& grid);
  // FindLiquidSurface on `grid` with the analytic surface z = surface_height and the given coverage.
  pressure::GeometrySnapshot BuildLiquidSnapshot(const grid::StaggeredGrid& grid, const SolidCoverageValues& coverage,
                                                 double surface_height, double step_duration,
                                                 pressure::PressureBuffers& buffers);
  std::vector<std::byte> ReadAllBytes(const SolidCoverageValues& coverage);  // and for SolidsAtFaces; each array
                                 // read with fluidity::ReadAll (base/test_support.h), then concatenated
  ```
  Grids come from `pressure::testing::BuildBoxGrid` / `BuildTwoLevelGrid`; particles from
  `transfer::test::BuildParticlesTry` with `ParticleSpec`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `SceneSolids, Tank_TopOpen`; `BuildLiquidSnapshot,
  HalfFullBox_HalfTheCellsAreLiquid`.
- [ ] **Step 2: Run, implement, run.** **Step 3: Gate; commit.** Subject: `Add the scenes the solids tests share`.

### Task 15: `ComputeSolidsAtFaces` on the CPU

**Kind:** numerics

**Files:**
- Create: `native/fluidity/solids/solids_on_grid_cpu.cc`, `solids_at_faces_test.cc`
- Modify: `solids_on_grid.h`

**Interfaces:**
- Produces both `ComputeSolidsAtFaces` forms exactly. Reach `kSolidReachInCells * w`, `w` the width of the face's
  block level (`grid::ComputeCellWidth`); position `grid::ComputeFacePosition`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeSolidsAtFaces, PlateAcrossFaces_SidesAndInside)  // design 7: plate normal {0.6,0,0.8}, half
      // thickness 0.01, on BuildTwoLevelGrid: side of each face centre by sign(n.(x - c)) beyond the thickness,
      // {0,kNoPlate} inside, outside_solid 0 exactly where |n.(x - c)| <= 0.01; exact
  TEST(ComputeSolidsAtFaces, CommittedPose_EqualsNextTryAtStart)  // design 7: fixed box and driven flat solid
      // between keys, after one commit: committed-pose form == next try at elapsed 0; bytes
  TEST(ComputeSolidsAtFaces, EndPoseOfDrivenSolid_MovedByVelocityTimesDt)
  TEST(ComputeSolidsAtFaces, TwoRangesEqualOneRange_SameBytes)
  TEST(ComputeSolidsAtFaces, ArraysOfOtherGrid_IsWrongGridVersion)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.4 step 1, meaning ported from
  `git show main:native/src/liquid_wall_contact_3d.cc` lines 592-683 and `elastic_blade_deformation_3d.cc` lines
  265-406, for several solids. A `RunOn::kGpu` call in a CPU build returns `kRunOnCallFailed`.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Tell every face where the solids are`.

### Task 16: `ComputeSolidCoverage` for analytic solids on the CPU

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the math test is `LABEL cpu` (`fluidity_add_shared_test` retired).

**Files:**
- Create: `native/fluidity/solids/motion_map_math.h`, `motion_map_math_test.cc`, `solid_coverage_test.cc`
- Modify: `solids_on_grid.h`, `solids_on_grid_cpu.cc`

**Interfaces:**
- Produces `ComputeSolidCoverage` (analytic part; `particles_try` null), `ToSolidCoverage`. **Plan decision**,
  shared `motion_map_math.h`:

  ```cpp
  // solid_normal_velocity of one face piece (design 5.4 step 4): sum corner_area * (velocity . axis) / sum
  // corner_area; 0 when the sum is 0.
  FLUIDITY_CPU_AND_GPU double ComputeSolidNormalVelocity(const double corner_area[4], const Vec3 corner_velocity[4],
                                                         Axis axis);
  ```

- [ ] **Step 1: Failing tests** (`LABEL cpu`, the math test included):

  ```cpp
  TEST(ComputeSolidNormalVelocity, NothingBlocked_IsZero)
  TEST(ComputeSolidCoverage, NoLiquid_CoversEveryActiveCell)  // design 7: box in a tank, arrays pre-filled with NaN:
      // no NaN left in any corner or face place of the blocks; equal bytes with and without a liquid's particles
  TEST(ComputeSolidCoverage, FlatSolid_BlockedAreaIsExact)    // design 7: flat solid normal {0.36,0.48,0.8}
      // through BuildTwoLevelGrid: per piece, sum of ComputeBlockedCornerAreas == area of the square clipped by
      // the plane (Sutherland-Hodgman in the test); 1e-12 w^2
  TEST(ComputeSolidCoverage, Box_BlockedAreaConverges)        // design 7: box corner inside cells, w 0.1, 0.05,
      // 0.025: error of the total blocked area falls by 4 within 20 %
  TEST(ComputeSolidCoverage, EqualDistances_LowerSolidIndexOwns)
  TEST(ComputeSolidCoverage, ClosedSideCounts_OpenTopDoesNot)
  TEST(ComputeSolidCoverage, CoverageOfOtherTry_TimeAndTryIdStamped)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.4 steps 2 and 4 (cap at `kCoverageCapInCells * w`;
  owner only below the cap; free solids at velocity before pressure).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Sample analytic solids at every corner of the grid`.

### Task 17: `ComputeSolidCoverage` for blocking materials

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/solids/solids_on_grid_cpu.cc`, `solid_coverage_test.cc`

**Interfaces:**
- Consumes: `particles::BuildFieldsToRead`, `GridTables::same_level_blocks`. Completes `ComputeSolidCoverage`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ComputeSolidCoverage, BallsOfParticles_MatchesExhaustiveScan)  // design 7: 2,000 random particles (seed 3)
      // of one blocking material, radius 0.6 w, three missing blocks, ten particles at cap + radius + 1e-9 w,
      // two particles at one place (ids 40 and 41): distance and owner equal a scan over all particles in the
      // test; exact
  TEST(ComputeSolidCoverage, ParticleTiesAnalyticSolid_AnalyticKeeps)  // Review Focus 2
  TEST(ComputeSolidCoverage, CornerVelocity_UsesFittedChange)    // v + C (corner - x); 1e-15
  TEST(ComputeSolidCoverage, BlockingMaterialWithoutFittedChange_IsMaterialNotCarried)
  TEST(ComputeSolidCoverage, BlockingMaterialAndNoParticles_IsBadInput)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.4 step 3, ported from
  `git show main:native/src/plastic_solid_boundary_3d.cc` lines 36-151 and
  `native/include/fluidity/base/sampled_solid_liquid_geometry_3d.h` lines 243-352, with the 27-block search
  and its bigger and smaller blocks instead of the scan.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Let snow and springy particles block the liquid`.

### Task 18: Solids on the grid on the GPU

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `solids_on_grid_gpu.h` returning `RunOnCallStatus`;
launches through `base/gpu_launch.cuh`; the per-place functions of `solids_on_grid_math.h` run by a `cpu` test over
every thread block and thread; solids copied into the buffers' `solids_copy`; digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/solids/solids_on_grid_gpu.cu`, `solids_on_grid_gpu.h` (private),
  `solids_on_grid_math.h` (private, shared), `solids_on_grid_steps_test.cc` (label `cpu`),
  `solids_on_grid_gpu_test.cc`, `solids_on_grid_no_gpu_test.cc`
- Modify: `solids_on_grid_cpu.cc` (the CPU loops call the per-place functions; `kGpu` calls the GPU half under
  `#if defined(FLUIDITY_HAS_GPU)`, its status mapped by `ToBufferError` then `ToSolidsFailure`), `CMakeLists.txt`

- [ ] **Step 1: Tests:**

  ```cpp
  // solids_on_grid_steps_test.cc, LABEL cpu (the main evidence until the hardware run)
  TEST(SolidsOnGridMath, EveryThreadOfLaunch_EqualsCpuLoop)  // Tasks 15-17 scenes: the face-place and
      // corner-place functions for every thread block and thread of ComputeLaunchSize(place count), launch order
      // and reverse: arrays equal the CPU loop's; bytes
  TEST(SolidsOnGridMath, FailureKeyOfEveryThread_GivesSmallestKey)  // two failing places in different thread
      // blocks: the integer minimum over the per-place failure keys, both orders, names the smaller key
  // LABEL gpu
  TEST(ComputeSolidsAtFaces, OnGpu_EqualsCpu)              // Task 15 scenes; design 6 tolerances
  TEST(ComputeSolidCoverage, OnGpu_EqualsCpu)              // Tasks 16 and 17 scenes; owners exact
  TEST(ComputeSolidCoverage, SameInputTwice_SameBytes)
  // LABEL no_gpu
  TEST(ComputeSolidsAtFaces, OnGpuWithoutAGpu_IsRunOnCallFailed)
  TEST(ComputeSolidCoverage, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
  CPU against GPU: owners, sides, counts exact; doubles within design 6 (`CheckWithinTolerance`). The `gpu` tests
  call `WriteResultDigest` on the solids-at-faces and coverage arrays.
- [ ] **Step 2: Implement** one thread per face or corner place (a grid-stride loop started by
  `ApplyGpuFunction(..., ComputeLaunchSize(place_count), ...)`) calling the same per-place functions as the CPU
  loop (moved to a private shared `solids_on_grid_math.h`); solids written into `solids_copy` with each call
  (`WriteBuffer`, no allocation); failures by integer minimum over a shared per-place failure key; `CheckGpuWork`
  before the failure is read back. **Step 3: Run** the gate. **Commit.** Subject: `Find the solids on the grid on the GPU`.
  Body: `CUDA: compiled, not run`.

### Task 19: `FindGridEdgesOpenToSky`

**Amended 2026-11-14 (A17):** on a GPU grid no block id comes down: one GPU pass marks the boxes that hold a cell
block, the CPU floods as before, one GPU pass writes the face marks (`plan-decisions.md` "Batched solves (A15)",
"A17 follow-ups").

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the public function is in `open_to_sky_cpu.cc`; the GPU half is
declared in `open_to_sky_gpu.h` and returns `RunOnCallStatus`; its two per-place functions run in a `cpu` test over
every thread block and thread; digests in the `gpu` test.

**Files:**
- Create: `native/fluidity/solids/open_to_sky_cpu.cc`, `open_to_sky_test.cc`, `open_to_sky_gpu.cu` (private, A17),
  `open_to_sky_gpu.h` (private), `open_to_sky_steps_test.cc` (label `cpu`), `open_to_sky_gpu_test.cc`,
  `open_to_sky_no_gpu_test.cc`
- Modify: `solids_on_grid.h`, `solids_on_grid_math.h`, `CMakeLists.txt`

**Interfaces:**
- Produces `FindGridEdgesOpenToSky` exactly. **Plan decisions:** boxes are the biggest-level block lattice of
  the grid frame over the domain box; the top layer is the boxes whose top reaches the domain top; a point
  above an open top counts as reached, a point outside any other side as not reached; nothing is marked when
  `closed.top`. Written with `grid::WriteValues` (the array may be on the GPU).
- **Plan decision (A17):** the work splits in three, each on its natural processor. (1) Box occupancy, one byte per
  box of the lattice: on a CPU grid from the CPU tables; on a GPU grid one thread per cell block writes 1 at its
  box (an idempotent byte store, no sum), then one `ReadBuffer` of the box bytes. (2) On the CPU, unchanged: the
  solid test at each box centre and the flood from the top layer in ascending box order. (3) The face marks: on a
  CPU grid as before; on a GPU grid one `WriteBuffer` of the reached-box bytes and one thread per face place reads
  the box holding the point half a cell width beyond its grid-edge piece. Copies: 2 B per box, nothing per block.
  Shared per-place formulas (box of a block, box beyond a piece) in `solids_on_grid_math.h`. Rows: two box arrays on
  `where = kCpu` and, for a GPU grid, the same two on `kRunOn` (boxes counted at the domain's lattice).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(FindGridEdgesOpenToSky, PocketUnderLid_IsNotOpen)   // design 7: open top, a fixed box roof over two
                                                           // missing blocks: marks equal a hand-made list; exact
  TEST(FindGridEdgesOpenToSky, OpenTopNoSolid_EveryUpperEdgeMarked)
  TEST(FindGridEdgesOpenToSky, ClosedTop_NothingMarked)
  // open_to_sky_steps_test.cc, LABEL cpu (amended, Foundation as built; the main evidence until the hardware run)
  TEST(FindGridEdgesOpenToSky, PerPlaceFunctionsOverEveryThread_EqualCpuMarks)  // the box-of-a-block and
      // box-beyond-a-piece functions for every thread block and thread of the two launches, launch order and
      // reverse: box bytes and face marks equal the CPU path's; exact
  // open_to_sky_gpu_test.cc, LABEL gpu (A17)
  TEST(FindGridEdgesOpenToSky, OnGpuGrid_EqualsCpuGrid)    // the three cases, grid built on both processors from one
      // list: marks equal, exact; the only copies are the two box arrays; WriteResultDigest on the marks
  // open_to_sky_no_gpu_test.cc, LABEL no_gpu (A17)
  TEST(FindGridEdgesOpenToSky, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.4 step 5. **Step 4: Run** the gate. **Commit.**
  Subject: `Find which grid edges are open to the sky`. Body (A17): `CUDA: compiled, not run`.

### Task 20: `BuildMotionMap` on the CPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the count and fill of one cell and the block partial of one solid
are shared per-item functions (`motion_map_steps.h`) that Task 24's GPU functions reuse; the prefix sum is
`ComputeExclusivePrefixSum` with `buffers.order_buffers`.

**Files:**
- Create: `native/fluidity/solids/motion_map.h`, `motion_map_cpu.cc`, `motion_map_steps.h` (private, shared),
  `motion_map_test.cc`
- Modify: `motion_map_math.h`, `motion_map_math_test.cc`, `CMakeLists.txt`

**Interfaces:**
- Consumes: `pressure::ComputeBlockedCornerAreas`, `pressure::ComputeEquationIndexes`, `grid::FindFacePieces`,
  `FindCornerPlaces`, `ComputeCornerPosition`. Produces `MotionMapEntry`, `MotionMap` (Plan decision 6),
  `MotionMapInputs`, `BuildMotionMap`, `ToFreeSolidPushes` (Plan decision 1), `ToFreeSolidInverseInertia`.
  Shared: `ComputeEntryPush(signed_corner_area, axis)` (m2) and `ComputeEntryTurning(signed_corner_area, arm,
  axis)` (m3).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildMotionMap, AnyMotion_FlowEqualsPushDotVelocity)   // design 7: free turned box, inertia {1,1,1}, gravity
      // 0, random start v and w (seed 13), liquid all around: per liquid cell, the sum of s * solid_volume_rate of
      // its pieces in the snapshot equals push_i . v + turning_i . w; 1e-13 relative
  TEST(BuildMotionMap, EntriesOrderedByCellFaceCorner)
  TEST(ToFreeSolidPushes, SortedBySolidThenEquation)
  TEST(ToFreeSolidInverseInertia, TwoPerFreeSolid)
  TEST(BuildMotionMap, MoreEntriesThanBuffers_IsBuffersTooSmall)
  TEST(BuildMotionMap, CoverageOfOtherTry_IsWrongTime)
  TEST(BuildMotionMap, BlocksNotWholeGrid_IsBadInput)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.5 steps 1 to 4 (two passes: count per cell, fill at
  the prefix sum). Plan decision (Foundation as built): `motion_map_steps.h` declares
  `FLUIDITY_CPU_AND_GPU std::uint32_t ComputeEntryCountOfCell(...)`, `FLUIDITY_CPU_AND_GPU void
  CopyEntriesOfCell(...)` and (Task 21) `FLUIDITY_CPU_AND_GPU SolidBlockPartial ComputeBlockPartialOfSolid(...)`
  (one block's push, turning push, work and swept volume of one solid, the block's entries in order); the CPU loop
  calls them cell by cell, and the counts are summed by `ComputeExclusivePrefixSum(counts, cell_count, starts,
  buffers.order_buffers)`. **Step 4: Run** the gate. **Commit.** Subject: `Build the motion map of the solids`.

### Task 21: `ApplyPressureToSolids` on the CPU

**Amended 2026-11-14 (A18):** it no longer stores pushes owed to blocking particles (plan decision 9).

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the block partials come from `ComputeBlockPartialOfSolid` in
`motion_map_steps.h`, so the GPU of Task 24 forms the same partials.

**Files:**
- Modify: `native/fluidity/solids/motion_map.h`, `motion_map_cpu.cc`, `motion_map_steps.h`, `motion_map_test.cc`

**Interfaces:**
- Produces `PressureOnSolids`, `ApplyPressureToSolids` exactly; the try keeps the push, turning push, work given
  and swept volume (`dt * solid volume rate into liquid cells`) per solid for `CommitTry`. Plan decision 5.
  **Plan decision (A18):** the design's "stores the pushes owed to blocking-material particles" is dropped;
  `push_to_materials` stays a report, the sum of magnitudes of the map entries whose corner owner is a blocking
  material (the same areas and pressures as Task 23's write-back, pressure of a non-liquid cell 0).

- [ ] **Step 1: Failing tests** (`LABEL cpu`); pressure arrays hand-filled:

  ```cpp
  TEST(ApplyPressureToSolids, AnyPressure_WorkBalances)       // design 7: Task 20 scene, random p (seed 17):
      // sum p_i flow_out_i dt == push . v + turning_push . w, flow_work == -(push . v + turning_push . w); 1e-13
  TEST(ApplyPressureToSolids, UniformPressure_NoNetPushOnClosedBox)   // design 7: p 1000 Pa: |push|,
                                                                      // |turning_push| <= 1e-12 p w^2 dt
  TEST(ApplyPressureToSolids, Hydrostatic_PushIsWeightOfDisplacedLiquid)  // design 7: box 0.4 m submerged, w = 0.4/16,
      // p = 1000 * 9.81 * depth at centres: push.z / dt == rho g V within 2 %
  TEST(ApplyPressureToSolids, OffCentrePressure_TurningPushIsArmCrossPush)  // design 7: p on one piece only:
      // turning_push == (blocked centre - origin) x push; 1e-13 relative
  TEST(ApplyPressureToSolids, FreeSolid_MotionIsVelocityAfterPressure)
  TEST(ApplyPressureToSolids, FastAfterPressure_IsSolidMovesTooFar)   // Review Focus 3
  TEST(ApplyPressureToSolids, PaddlePushingLiquid_WorkIsPositive)     // Review Focus 5: driven flat solid moving
                                                                      // into liquid at p > 0: flow_work > 0
  TEST(ApplyPressureToSolids, PressureOfOtherGrid_IsWrongGridVersion)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.5 step 5 (block partials, then tree sum). Ported
  idea: `git show main:native/src/rigid_body_3d.cc` lines 712-779 without the gradient torque (`:770-776`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Return the pressure push to the solids`.

### Task 22: One solve shares velocity between a free solid and the liquid

**Kind:** integration

**Files:**
- Create: `native/fluidity/solids/free_solid_solve_test.cc`

- [ ] **Step 1: Failing test** (`LABEL cpu`):

  ```cpp
  TEST(FreeSolid, LightBoxOnOneCellOfLiquid_OneSolveGivesSharedVelocity)  // design 7. BuildBoxGrid 1 x 1 x 2 of
      // w 0.1; the upper cell's place is a free box (guided along z, mass m 0.5 kg, at rest) whose bottom face
      // lies on the top face of the lower cell. Hand-built snapshot: lower cell kLiquid, its low-Z face
      // kLiquidToAir, open area w^2, pressure_distance w, velocity_before_pressure U = 1 m/s; side faces closed;
      // top face open 0, solid_volume_rate 0; density 1000. ComputeSolidCoverage, BuildMotionMap,
      // BuildPressureEquations with ToFreeSolidPushes, SolvePressure, ApplyPressureToSolids. With
      // M = 1000 w^3: box velocity and the low face's velocity after pressure both M U / (M + m),
      // within 2e-5 * w + 1e-12
  ```
- [ ] **Step 2: Run.** Expected: pass (Tasks 20, 21 and pressure exist); if not, fix the owning commit.
- [ ] **Step 3: Gate; commit.** Subject: `Test that one solve moves a free solid with the liquid`.

### Task 23: Blocking-material faces for the liquid solve, and their velocity after it

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Amended 2026-11-14 (A18):** replaces the push returned to particles one step late; solids now builds the faces
the liquid solve holds and writes their solved velocity back to each set in the same step.

**Amended 2026-11-14 (A24):** viscosity's joint solve carries the same material faces (`plan-decisions.md`,
"Viscous liquids with blocking materials and sealed gas (A24)"), so `BuildBlockingMaterialFaces` and
`ApplyPressureToMaterials` serve stage 11 and stage 12 alike: the step builds the faces at stage 10 whatever stage
follows, and after stage 11 passes `viscosity::SolvedCouplingArrays::blocking_material_velocity_after_pressure` as
`velocity_after_pressure`. Snow or a springy solid beside a viscous liquid is no longer excluded; free solids beside
a viscous liquid stay excluded (viscosity's `kFreeSolidNotSupported`). No signature and no test of this task
changes; the stage 11 scenes are the step's and viscosity's tests. **Replaced in part by the A26/A27 amendment
below:** "after stage 11 passes ... as `velocity_after_pressure`" and "free solids beside a viscous liquid stay
excluded".

**Amended 2026-11-14 (A26, A27):** in a viscous scene the joint solve no longer eliminates the material faces: the
faces of each blocking set within 1.5 cell widths of the liquid are velocity unknowns and the liquid sticks to them
(A26), and free solids beside a viscous liquid are no longer excluded: their six motions are unknowns too (A27).
Stage 10 still calls `BuildBlockingMaterialFaces` in every scene with liquid and a blocking set (its entries give
each band face's pressure term, and `SolvePressure` uses them in an inviscid scene), then, in a viscous scene,
`BuildViscousMaterialFaces` (Task 23a) and `ToFreeSolidsForViscousSolve` (Task 23c). Stage 11 calls
`ApplySolvedMotionToSolids` (Task 23c) in place of `ApplyPressureToSolids` and `ApplyViscousSolveToMaterials`
(Task 23b) in place of `ApplyPressureToMaterials`. Stage 12 is unchanged. No signature and no test of this task
changes.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the count and fill of one block and the per-set partial are shared
per-item functions (`blocking_material_faces_steps.h`) for Task 24's GPU half; the prefix sum is
`ComputeExclusivePrefixSum` with `buffers.order_buffers`.

**Files:**
- Create: `native/fluidity/solids/blocking_material_faces.h`, `blocking_material_faces_cpu.cc`,
  `blocking_material_faces_steps.h` (private, shared), `blocking_material_faces_test.cc`,
  `blocking_material_solve_test.cc`
- Modify: `native/fluidity/solids/motion_map.h` (the design's `MaterialPushFields` and particle form of
  `ApplyPressureToMaterials` are not declared), `docs/glossary.md` (`blocking set`)

**Interfaces:**
- Consumes: `pressure::BlockingMaterialFace`, `ComputeMaterialVelocityAfterPressure` (pressure plan Task 44),
  `pressure::ComputeBlockedCornerAreas`, `grid::FindFacePieces`, `FindCornerPlaces`, `transfer::GridFromParticles`,
  `SolidCoverageValues::corner_owner`, `corner_velocity`.
- Produces (**Plan decision**, A18; replaces design 3.6's `MaterialPushFields` and the particle form):

  ```cpp
  struct BlockingSet {                                 // One material set of blocking materials (A8).
    std::span<const std::uint8_t> material_is_counted; // The set's mask, one entry per material_index.
    transfer::GridFromParticles* grid_from_particles;  // After ApplyForces and CollideSolids; its velocity is
  };                                                   //   written by ApplyPressureToMaterials.
  struct BlockingMaterialFacesInputs {
    const grid::StaggeredGrid& grid;
    grid::CellBlockRange blocks;                 // Every block of the grid, else kBadInput (equation order).
    const pressure::GeometrySnapshot& geometry;  // Same try: which cells are kLiquid; open_area per piece.
    const SolidCoverageValues& coverage;         // Same time and try, else kWrongTime.
    std::span<const BlockingSet> blocking_sets;  // Every blocking material in exactly one set, else kBadInput.
  };
  class BlockingMaterialFaces;  // Move-only view into SolidsBuffers (like MotionMap): entries, try_id, time,
                                // grid_version, build number, std::uint32_t faces_without_mass() const.
  std::expected<BlockingMaterialFaces, SolidsFailure> BuildBlockingMaterialFaces(
      const BlockingMaterialFacesInputs& inputs, SolidsBuffers& buffers);
  // What pressure::EquationInputs::blocking_material_faces takes.
  std::span<const pressure::BlockingMaterialFace> ToBlockingMaterialFaces(const BlockingMaterialFaces& faces);
  struct PressureOnMaterials {   // Per blocking set: sum of face_mass * (after - before) * axis, N s, in entry
    std::span<const Vec3> push;  //   order (block partials, then math::TreeSum); and the sum of magnitudes.
    double push_magnitude = 0;
  };
  // Stage SolvePressure (stage 12), after the liquid solve and before SaveGridAfterPressure. Writes
  // velocity_after_pressure[k] (pressure::SolvedArrays::blocking_material_velocity_after_pressure) into the face
  // velocity of entry k's set. kBadInput for a span of another size or another set list; kWrongTime for faces of
  // an earlier build. In a viscous scene stage 11 calls ApplyViscousSolveToMaterials instead (A26, Task 23b).
  std::expected<PressureOnMaterials, SolidsFailure> ApplyPressureToMaterials(
      const BlockingMaterialFaces& faces, std::span<const double> velocity_after_pressure,
      std::span<const BlockingSet> blocking_sets, SolidsBuffers& buffers);
  ```
- **Rule (plan decision).** One entry per (face piece, set) where the piece touches at least one `kLiquid` cell
  and some of its corners are owned (`corner_owner.material_index`) by a material of that set: `blocked_area` =
  sum of those corners' `ComputeBlockedCornerAreas`; `face_mass` = the set's `mass` at the piece;
  `velocity_before_pressure` = the set's `velocity` there; `coverage_volume_rate` = sum over those corners of
  `corner_area * (corner_velocity . axis)` (the share of pressure's `solid_volume_rate` these corners gave).
  Sorted by `face_piece_index`, then set; two passes (count per block, fill at the prefix sum). A piece whose set
  has no face mass there (value kind not `kFromParticles`) gives no entry, is counted in `faces_without_mass()`,
  and its blocked part stays in `solid_volume_rate` at the sampled velocity (one way, as before A18). A piece that
  touches no `kLiquid` cell gives no entry: the liquid does not move that face. More entries than
  `max_motion_map_entries`: `kBuffersTooSmall`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildBlockingMaterialFaces, SnowLayerUnderLiquid_OneEntryPerBlockedPiece)  // snow layer (one blocking
      // material, 300 particles, seed 19) under liquid: entries are exactly the pieces touching kLiquid with snow
      // corners; per piece blocked_area == area - open_area and coverage_volume_rate == the snapshot's
      // solid_volume_rate (only snow blocks there); 1e-14 relative
  TEST(BuildBlockingMaterialFaces, TwoSetsOnOnePiece_TwoEntriesInSetOrder)   // snow and jelly corners on one piece
  TEST(BuildBlockingMaterialFaces, PieceWithoutSetMass_CountedNotEntered)    // faces_without_mass == 1
  TEST(BuildBlockingMaterialFaces, MaterialInNoSet_IsBadInput)               // and a material in two sets
  TEST(BuildBlockingMaterialFaces, CoverageOfOtherTry_IsWrongTime)
  TEST(BuildBlockingMaterialFaces, BlocksNotWholeGrid_IsBadInput)
  TEST(ApplyPressureToMaterials, MaterialMomentumEqualsPushOnLiquid)  // same layer, random p (seed 19), each entry's
      // velocity after from pressure::ComputeMaterialVelocityAfterPressure (non-liquid end 0): PressureOnMaterials
      // push == minus the push on the liquid (sum over the entries of -dt (p_low - p_high) blocked_area axis);
      // 1e-13 relative; the set's velocity changed at entry pieces only (other bytes equal)
  TEST(ApplyPressureToMaterials, SpanOfOtherSize_IsBadInput)
  // blocking_material_solve_test.cc
  TEST(BlockingMaterial, LightSnowOnOneCellOfLiquid_OneSolveGivesSharedVelocity)  // Task 22's lid, but the upper
      // cell is snow: one blocking set, hand-built coverage (the shared face's corners owned by the snow material,
      // velocity 0) and GridFromParticles (mass m 0.5 kg, velocity 0 at the shared face). BuildBlockingMaterialFaces,
      // BuildPressureEquations with the entries, SolvePressure, ApplyPressureToMaterials: the set's face velocity
      // and the liquid's low face velocity both M U / (M + m) = 2/3 m/s (M = 1000 w^3 = 1 kg, U 1 m/s), within
      // 2e-5 * w + 1e-12
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** The proof of concept's form of the same term is the
  particle mobility `git show main:native/src/plastic_solid_boundary_3d.cc` lines 406-546 (corner coefficients
  from `StagePlasticSolidPressureFace3D`, lines 152-271); here it is lumped to the set's face mass, as design
  decision 3 option c describes (`dt * blocked_area^2 / face mass`).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Give the liquid solve the blocking materials' faces`. New
  names: `BlockingSet`, `BlockingMaterialFacesInputs`, `BlockingMaterialFaces`, `BuildBlockingMaterialFaces`,
  `ToBlockingMaterialFaces`, `PressureOnMaterials`, the new `ApplyPressureToMaterials`; removed: `MaterialPushFields`.

### Task 23a: The material faces the viscous solve moves

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Amended 2026-11-14 (A26):** new task. In a viscous scene the liquid sticks to a blocking material, so the
material's faces near the liquid become velocity unknowns of viscosity's joint solve (`plan-decisions.md` "A viscous
liquid sticks, and free solids in the viscous solve (A26, A27)"). The CPU form is here; the GPU form is in Task 24.

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the count and fill of one block are shared per-item functions in
`blocking_material_faces_steps.h` (reused by Task 24's GPU half); the prefix sum is `ComputeExclusivePrefixSum` with
`buffers.order_buffers`.

**Files:**
- Create: `native/fluidity/solids/viscous_material_faces.h`, `viscous_material_faces_cpu.cc`,
  `viscous_material_faces_test.cc`; modify `blocking_material_faces_steps.h`, `CMakeLists.txt`
- Modify: `solids_buffers.h`, `solids_buffers.cc`, `solids_buffers_test.cc` (one row), `docs/glossary.md`
  (`material band`: the faces of a blocking set that the viscous solve moves)

**Interfaces:**
- Consumes: Task 23 (`BlockingSet`, `BlockingMaterialFacesInputs`, `BlockingMaterialFaces`,
  `ToBlockingMaterialFaces`); `transfer::GridFromParticles` (`mass`, `value_kind`, `velocity`),
  `transfer::FaceValueKind::kFromParticles`; `pressure::GeometrySnapshot` (`CellGeometry::filled_volume`);
  `grid::FindFacePieces`, `ComputeCellCenter`, `ComputeCellWidth`, `grid::kWeightCurveReach`.
- Produces exactly as `plan-decisions.md` declares them: `ViscousMaterialFace`, `ViscousMaterialFaces`,
  `BuildViscousMaterialFaces`, `ToViscousMaterialFaces`. Viscosity's `ViscosityInputs::material_faces` takes the
  span.
- **Rule (plan decision, A26).** For each blocking set (index into `blocking_sets`) and each face piece of the
  blocks where the set's `value_kind` is `kFromParticles`: an entry when the piece centre lies within
  `grid::kWeightCurveReach * w` (1.5 widths; it equals viscosity's `kFitReach`, which solids cannot include) of
  the centre of some cell with `filled_volume > 0`, distance `<=`, `w` the width of the smaller of the piece's
  two cells. `face_mass` and `velocity_before_pressure` are the set's `mass` and `velocity` at the piece, read
  as `BuildBlockingMaterialFaces` reads them. `blocking_face_index` is the index of the `pressure_faces` entry
  with the same (piece, set), else `0xFFFFFFFFu`. Every `pressure_faces` entry must be matched (it touches a
  `kLiquid` cell, so it is in the band); an unmatched one is `kBadInput`. Where two sets have face mass on one
  piece, both are listed; viscosity chooses the sample's material side. Sorted by `face_piece_index`, then set;
  two passes (count per block, fill at the prefix sum).
- **Failures (plan decision).** `geometry` not the object `inputs.geometry` refers to: `kBadInput`;
  `pressure_faces` of an earlier build, or coverage of another try: `kWrongTime`; blocks not the whole grid:
  `kBadInput`; more entries than `2 * max_motion_map_entries`: `kBuffersTooSmall`.
- **Row** (`"solids: viscous material faces"`): given `2 * max_motion_map_entries`,
  `sizeof(ViscousMaterialFace)` (32, `static_assert`) plus 4 B per block for the count.

Two test scenes, built in the test file and reused by Task 23b. Springy solids are the validation material;
snow is only checked for not breaking, in the step's five-material scene (owner note, 2026-11-14):

- **Springy layer.** `pressure::testing::BuildBoxGrid` 8 x 8 x 8 cells of `w = 0.025` m. One `kSpringySolid`
  material in its own set, `particle_radius 0.6 w`, 300 particles (seed 19) uniform in x and y in [0, 0.2] m and
  z in [0, 0.05] m; `GridFromParticles` from `transfer::ParticlesToGrid` of that set; coverage from
  `ComputeSolidCoverage`; liquid snapshot from `BuildLiquidSnapshot` with `surface_height 0.125` m.
- **Springy lid.** Task 23's `BlockingMaterial, LightSnowOnOneCellOfLiquid_...` scene with a `kSpringySolid`
  material in the upper cell: `BuildBoxGrid` 1 x 1 x 2 of `w = 0.1`; lower cell `kLiquid`, `filled_volume` 1e-3
  m3; hand-built coverage (the shared face's corners owned by the springy material, velocity 0); hand-built
  `GridFromParticles`: mass 0.5 kg, `kFromParticles`, velocity 0 at the shared face only.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildViscousMaterialFaces, SpringyLayerUnderLiquid_EntriesAreTheBand)  // springy layer: the test scans every
      // face piece with set face mass against every cell centre with filled_volume > 0 (brute force, <= 1.5 w):
      // entries are exactly that set, sorted by (piece, set); face_mass and velocity_before_pressure equal the
      // GridFromParticles values; at least one piece with face mass (the layer's bottom, 2.5 w from the nearest
      // liquid centre) is not an entry; exact
  TEST(BuildViscousMaterialFaces, EveryBlockingFace_IndexPointsBack)  // springy layer: for each BlockingMaterialFace
      // j exactly one entry has blocking_face_index j, with the same face_piece_index and set; every other entry
      // has 0xFFFFFFFF; exact
  TEST(BuildViscousMaterialFaces, SpringyLid_OneEntry)  // one entry: the shared face, set 0, face_mass 0.5,
      // velocity_before_pressure 0, blocking_face_index 0; exact
  TEST(BuildViscousMaterialFaces, SetWithoutFaceMass_GivesNone)  // springy layer plus a second springy set whose
      // GridFromParticles has no kFromParticles face: no entry has material_set 1
  TEST(BuildViscousMaterialFaces, TwoSetsOnOnePiece_BothListed)  // springy lid with a second springy set, mass 0.3 kg
      // at the shared face and no corner: two entries (set 0 then set 1), face_mass 0.5 and 0.3,
      // blocking_face_index 0 and 0xFFFFFFFF
  TEST(BuildViscousMaterialFaces, MoreEntriesThanBuffers_IsBuffersTooSmall)  // springy layer, buffers built with
      // max_motion_map_entries 2
  TEST(BuildViscousMaterialFaces, OtherGeometry_IsBadInput)       // a copy of the snapshot, not inputs.geometry
  TEST(BuildViscousMaterialFaces, BlockingFacesOfEarlierBuild_IsWrongTime)
  TEST(BuildSolidsMemoryEstimate, SumEqualsAllocatedBytes)       // extended: the new row is counted
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the rule above; the cell search looks at the cells within two
  widths of the piece through the grid's neighbour tables (the test's scan is the oracle).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Find the material faces the viscous solve moves`. New names:
  `ViscousMaterialFace`, `ViscousMaterialFaces`, `BuildViscousMaterialFaces`, `ToViscousMaterialFaces`.

### Task 23b: Material velocities after the viscous solve

**Amended 2026-11-14 (A26):** new task. The CPU form is here; the GPU form is in Task 24.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/solids/viscous_material_faces.h`, `viscous_material_faces_cpu.cc`,
  `viscous_material_faces_test.cc`

**Interfaces:**
- Consumes: Task 23a; Task 23's `PressureOnMaterials`, `ApplyPressureToMaterials` (for one comparison test) and
  the per-set push row.
- Produces `ApplyViscousSolveToMaterials` exactly as `plan-decisions.md` declares it. Stage 11 of a viscous
  scene, after `ApplySolvedMotionToSolids` and before `ComputeSolidsAtFaces` at `dt` and
  `SaveGridAfterPressure`; `velocity_after_solve` is
  `viscosity::SolvedCouplingArrays::material_face_velocity_after_solve` (the step-owned span
  `"step: material face velocity after solve"`).
- **Rule (plan decision).** Writes `velocity_after_solve[k]` into the face velocity of entry k's set at its piece
  (as `ApplyPressureToMaterials` writes a piece); every other face of every set keeps its bytes.
  `PressureOnMaterials::push` per set is the sum of `face_mass * (after - before) * axis` (block partials in entry
  order, then `math::TreeSum` over blocks); `push_magnitude` as `ApplyPressureToMaterials` defines it.
  `kBadInput` for a span whose size is not the entry count or a set list of another size (nothing written);
  `kWrongTime` for faces of an earlier build.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ApplyViscousSolveToMaterials, SpringyLid_VelocityAndPush)  // after {2.0 / 3.0}: the set's velocity at the
      // shared face == 2.0 / 3.0 and push[0] == {0, 0, 0.5 * (2.0 / 3.0)}; exact
  TEST(ApplyViscousSolveToMaterials, SpringyLayer_WrittenAndPushIsMassTimesChange)  // velocity_after_solve[k] =
      // velocity_before_pressure[k] + 0.01 * ((k % 7) - 3) m/s: each entry's face velocity equals the value
      // given, exact; every other byte of the set's velocity array unchanged; push equals the test's plain sum of
      // face_mass * (after - before) * axis; 1e-13 relative
  TEST(ApplyViscousSolveToMaterials, BlockingSubsetOnly_EqualsApplyPressureToMaterials)  // springy layer: entries
      // with a blocking_face_index j take after_j = velocity_before + 0.02 * ((j % 5) - 2), the others keep
      // velocity_before: push equals ApplyPressureToMaterials's push given the after_j; 1e-13 relative
  TEST(ApplyViscousSolveToMaterials, SpanOfOtherSize_IsBadInput)  // entry count - 1 and + 1: error, bytes unchanged
  TEST(ApplyViscousSolveToMaterials, FacesOfEarlierBuild_IsWrongTime)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4: Run** the gate. **Commit.** Subject:
  `Write the viscous solve's velocities into the blocking sets`. New name: `ApplyViscousSolveToMaterials`.

### Task 23c: Free solids in the viscous solve, and their motion after it

**Amended 2026-11-14 (A27):** new task. Free solids beside a viscous liquid are no longer excluded: their linear
and turning velocities are unknowns of the viscous solve. The CPU form is here; the GPU form is in Task 24.

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/solids/motion_map.h`, `motion_map_cpu.cc`, `motion_map_test.cc`, `solids_try.h`,
  `solids_try.cc`, `solids_buffers.h`, `solids_buffers.cc`, `solids_buffers_test.cc` (one row)

**Interfaces:**
- Consumes: Tasks 10, 20, 21 (`ApplyPressureToSolids`'s per-solid pressure sums, reused unchanged), Task 6
  (`ComputeInverseInertia`), Task 4 (`ToAxes`).
- Produces exactly as `plan-decisions.md` declares them: `FreeSolidForViscousSolve`,
  `ToFreeSolidsForViscousSolve`, `FreeSolidMotion`, `ApplySolvedMotionToSolids`. Viscosity's
  `ViscosityInputs::free_solids` takes the span; `SolvedCouplingArrays::free_solid_motion` is the span given back.
- **Plan decisions (A27):**
  1. **One entry per free solid of the try**, in `FreeSolidPush::solid_index` order (`solid_index` k at entry k),
     including a free solid that touches no liquid this step: it has no pressure or sample term, so the solve gives
     its motion back unchanged. (`plan-decisions.md`'s comment says "touching liquid"; listing all keeps one index
     for the motion span and costs at most `kMaxFreeSolidCount` entries.) `BuildMotionMap` fills the list:
     `mass`; `inertia_at_start = R I R^T`, `R = ToAxes` of `pose_at_step_start`, `I` the diagonal of
     `inertia_in_own_axes`; `origin_at_start`; `velocity_before_pressure` and `turn_rate_before_pressure` equal to
     the try's motion from `BuildTry`. Row `"solids: free solids for the viscous solve"`: given
     `kMaxFreeSolidCount`, `sizeof(FreeSolidForViscousSolve)` (160, `static_assert`), on the run's processor.
  2. `ApplySolvedMotionToSolids`, in order. Step 1: the pressure part, computed from the map and `pressure`
     exactly as `ApplyPressureToSolids` computes its push, turning push, `flow_work`, swept volume and
     `push_to_materials` (same code, block partials then tree sum), kept in the try as `ApplyPressureToSolids`
     keeps them (so `CommitTry`'s `ExchangeTotals` and `CheckSolidExchange` are unchanged). Step 2: each free
     solid's motion becomes `motion_after_solve[k]`, except that a motion a guide forbids (`FreeMotionLimits`)
     keeps its value before pressure. Step 3: the travel and turn limits of Task 10 are checked on the new motion
     (`kSolidMovesTooFar`, `limit` the allowed step). Step 4: the solved push, `mass * (velocity after - before)`
     and `inertia_at_start * (turn rate after - before)`, kept in the try. Failures: a span whose size is not
     the entry count (`kBadInput`), a map of an earlier build or another try (`kWrongTime`), pressure of another
     grid (`kWrongGridVersion`), a value that is not finite (`kNotFinite`); on failure the try is unchanged.
  3. `SolidsTry` gains two CPU accessors (copies inside the try; no pointer into `SolidsBuffers`):

     ```cpp
     struct SolvedMotionReport {          // One per free solid, FreeSolidPush::solid_index order.
       Vec3 push{};                       // N s: mass * (velocity after - velocity before), the whole change.
       Vec3 turning_push{};               // N m s about origin_at_start: inertia_at_start * (turn rate change).
       Vec3 pressure_push{};              // N s: the pressure part (step 1); equals ApplyPressureToSolids's push.
       Vec3 pressure_turning_push{};      // N m s.
     };
     // Filled by ApplySolvedMotionToSolids; empty after BuildTry and after ApplyPressureToSolids.
     std::span<const SolvedMotionReport> solved_motion() const;
     // The pressure part of the last ApplyPressureToSolids or ApplySolvedMotionToSolids. The step passes it as
     // ExchangeInputs::pressure_on_solids after stage 11.
     PressureOnSolids pressure_on_solids() const;
     ```
     The viscous part (`push - pressure_push`) is not an `ExchangeTotals` sum; its balance is viscosity's momentum
     test.
  4. **Guides are not carried by `FreeSolidForViscousSolve`** (the declaration has no `FreeMotionLimits`), so the
     solve treats a guided solid as free and step 2 drops the forbidden motions. Listed for the owner in the
     hand-over (Task 38): viscosity should hold the forbidden motions at their value before pressure.

Test scene (built in `motion_map_test.cc`): `SceneSolids` with a fixed tank 0.4 x 0.4 x 0.4 m (id 1) and
`AddFreeBox(2, {0.2, 0.2, 0.2}, {0.05, 0.05, 0.05}, 0.5, {1e-3, 2e-3, 3e-3})`; `BuildBoxGrid` 16 x 16 x 16 of
`w = 0.025`; `BuildLiquidSnapshot` with `surface_height 0.3`; gravity `{0, 0, -9.81}`, `dt = 0.002` s, from rest.
So the box's velocity before pressure is `{0, 0, -0.01962}` m/s, its turn rate 0, its travel radius `|half_size| =
0.0866` m, and the travel limit `0.5 * 0.025 = 0.0125` m per step.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ToFreeSolidsForViscousSolve, TwoFreeBoxes_EqualTryBeforePressure)  // the scene plus box 3 written as a
      // SolidSettings directly: centre {0.08, 0.08, 0.2}, half 0.03, mass 0.2, inertia_in_own_axes
      // {2e-4, 3e-4, 4e-4}, rest_pose axes a quarter turn about z, start_angular_velocity {0, 0, 0.5}.
      // Two entries, solid_index 0 and 1 (the tank is absent). Box 2: mass 0.5, origin {0.2, 0.2, 0.2},
      // inertia_at_start diag {1e-3, 2e-3, 3e-3}, exact. Box 3: inertia_at_start diag {3e-4, 2e-4, 4e-4} within
      // 1e-19 kg m2 per entry. Both: velocity and turn rate bytes equal SolidsTry::solids()[i].motion
      // (box 2: {0, 0, -0.01962}, 0; box 3: {0, 0, -0.01962}, {0, 0, 0.5})
  TEST(ToFreeSolidsForViscousSolve, BoxAboveLiquid_StillListed)  // surface_height 0.1: one entry, no FreeSolidPush
  TEST(ToFreeSolidsForViscousSolve, NoFreeSolid_IsEmpty)
  TEST(ApplySolvedMotionToSolids, HandMotion_WrittenAndPushIsMassTimesChange)  // pressure 0 everywhere; motion
      // {velocity {0.01, 0, 0.03}, turn_rate {0, 0, 0.2}}: the try's motion is exactly that; solved_motion()[0]
      // push == 0.5 * (after - before) = {0.005, 0, 0.02481} and turning_push == {0, 0, 3e-3 * 0.2} = {0, 0, 6e-4}
      // (the test's own expressions, exact); pressure_push 0
  TEST(ApplySolvedMotionToSolids, PressurePart_EqualsApplyPressureToSolids)  // p = 1000 * 9.81 * (0.3 - z) at
      // kLiquid cell centres: pressure_push, pressure_turning_push and pressure_on_solids() (push, turning_push,
      // flow_work, push_to_materials) equal ApplyPressureToSolids's on a second BuildTry of the same Solids with
      // the same pressure; bytes
  TEST(ApplySolvedMotionToSolids, GuidedBox_ForbiddenMotionsKeepTheirValue)  // limits z only, no turning; motion
      // {{0.01, 0.02, 0.03}, {0.1, 0, 0}}: written {{0, 0, 0.03}, {0, 0, 0}}; push {0, 0, 0.5 * (0.03 + 0.01962)}
      // = {0, 0, 0.02481}; turning_push 0
  TEST(ApplySolvedMotionToSolids, FastAfterSolve_IsSolidMovesTooFar)  // velocity {0, 0, -7}: 7 * 0.002 = 0.014 >
      // 0.0125; limit 0.0125 / 7 s within 1e-15 relative; the try's motion unchanged
  TEST(ApplySolvedMotionToSolids, MotionOfOtherSize_IsBadInput)  // 0 and 2 entries for one free solid
  TEST(ApplySolvedMotionToSolids, NotFinite_IsNotFinite)        // velocity.x NaN
  TEST(ApplySolvedMotionToSolids, MapOfEarlierBuild_IsWrongTime)
  TEST(BuildSolidsMemoryEstimate, SumEqualsAllocatedBytes)       // extended: the new row is counted
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `// Step N:` comments for steps 1 to 4 above.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Give free solids to the viscous solve and take their motion
  back`. New names: `FreeSolidForViscousSolve`, `ToFreeSolidsForViscousSolve`, `FreeSolidMotion`,
  `ApplySolvedMotionToSolids`, `SolvedMotionReport`, `SolidsTry::solved_motion`, `SolidsTry::pressure_on_solids`.

### Task 24: The motion map on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Amended 2026-11-14 (A18):** adds `BuildBlockingMaterialFaces` and the new `ApplyPressureToMaterials` on the GPU.

**Amended 2026-11-14 (A26, A27):** adds the GPU forms of Tasks 23a, 23b and 23c: `BuildViscousMaterialFaces`,
`ApplyViscousSolveToMaterials`, the free-solid list `BuildMotionMap` fills, and `ApplySolvedMotionToSolids`.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** three private `_gpu.h` headers returning `RunOnCallStatus`; GPU
functions as thin loops over `motion_map_steps.h` and `blocking_material_faces_steps.h`; prefix sums and grouping
through Foundation's functions taking `buffers.order_buffers`; a `cpu` test over every thread block and thread;
digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/solids/motion_map_gpu.cu`, `motion_map_gpu.h` (private), `motion_map_steps_test.cc`
  (label `cpu`), `motion_map_gpu_test.cc`, `motion_map_no_gpu_test.cc`, `blocking_material_faces_gpu.cu`,
  `blocking_material_faces_gpu.h` (private), `viscous_material_faces_gpu.cu` (A26), `viscous_material_faces_gpu.h`
  (private)
- Modify: `motion_map_cpu.cc`, `blocking_material_faces_cpu.cc`, `viscous_material_faces_cpu.cc` (`kGpu` calls the
  GPU half under `#if defined(FLUIDITY_HAS_GPU)`, its status mapped by `ToBufferError` then `ToSolidsFailure`),
  `motion_map_steps.h` and `blocking_material_faces_steps.h` (the per-group sum of one solid or set,
  `ComputeSumOfGroup`, a `math::TreeSumOf` over the group's block partials in block order), `CMakeLists.txt`

- [ ] **Step 1: Tests:** `cpu` (`motion_map_steps_test.cc`, the main evidence until the hardware run):
  `MotionMapSteps, EveryThreadOfLaunch_EqualsCpuMap` (Task 20 scene: `ComputeEntryCountOfCell`, then
  `ComputeExclusivePrefixSum`, then `CopyEntriesOfCell` for every thread block and thread of
  `ComputeLaunchSize(cell_count)`, launch order and reverse: entries equal the CPU map; bytes),
  `MotionMapSteps, PartialsAndGroupSumsOverEveryThread_EqualCpuSums` (Task 21 scene: `ComputeBlockPartialOfSolid`
  and `ComputeSumOfGroup` per thread, both orders: push, turning push, work bytes equal the CPU's),
  `BlockingMaterialFacesSteps, EveryThreadOfLaunch_EqualsCpuEntries` (Tasks 23 and 23a scenes, both kinds of entry).
  `gpu`: `BuildMotionMap, OnGpu_EqualsCpu`, `BuildMotionMap, SameInputTwice_SameBytes`,
  `ApplyPressureToSolids, OnGpu_EqualsCpu`, `ApplyPressureToSolids, SameInputTwice_SameBytes`,
  `BuildBlockingMaterialFaces, OnGpu_EqualsCpu`, `ApplyPressureToMaterials, OnGpu_EqualsCpu` (Tasks 20, 21, 23
  scenes; entries exact, sums design 6); A26/A27: `BuildViscousMaterialFaces, OnGpu_EqualsCpu`,
  `ApplyViscousSolveToMaterials, OnGpu_EqualsCpu` (springy layer of Task 23a; entries and written velocities
  exact, pushes design 6), `ToFreeSolidsForViscousSolve, OnGpu_EqualsCpu` (the list read back; exact),
  `ApplySolvedMotionToSolids, OnGpu_EqualsCpu` (Task 23c scene; motion exact, pushes design 6); `no_gpu`:
  `<each of the four operations>, OnGpuWithoutAGpu_IsRunOnCallFailed`, and the same for
  `BuildViscousMaterialFaces`, `ApplyViscousSolveToMaterials`, `ApplySolvedMotionToSolids`. Every `gpu` test calls
  `WriteResultDigest` on its entries, written velocities and per-solid sums. CPU against GPU: entries, counts and
  written velocities exact (copies, no arithmetic); sums within design 6 (`CheckWithinTolerance`).
- [ ] **Step 2: Implement** the two-pass build (count GPU function, `ComputeExclusivePrefixSum(...,
  buffers.order_buffers)`, fill GPU function, each a grid-stride loop started by `ApplyGpuFunction(...,
  ComputeLaunchSize(count), ...)`); per-solid sums grouped by `BuildGroupsByDestination(...,
  buffers.order_buffers)` and reduced with the tree of `math::TreeSum` (one thread per solid and component around
  `ComputeSumOfGroup`); `CheckGpuWork` before each read-back; only two `Vec3` per solid and the counts come back to
  the CPU. Every array is in `SolidsBuffers` (no `GpuArray`). A26/A27:
  `BuildViscousMaterialFaces` is two passes like the blocking faces; `ApplyViscousSolveToMaterials` is one thread
  per entry, its per-set push grouped like the per-solid sums; the free-solid list is computed on the CPU from the
  try and sent with one `WriteBuffer` (at most 16 x 160 B); `ApplySolvedMotionToSolids` reuses the pressure sums
  of `ApplyPressureToSolids` and does steps 2 to 4 on the CPU (the motion span is read back, 48 B per free solid).
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Build and apply the motion map on the GPU`. Body:
  `CUDA: compiled, not run`.

### Task 25: `CollideSolids`, the rule at solids

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/solids/collide_solids.h`, `collide_solids_cpu.cc`, `collide_solids_math.h` (private
  shared), `collide_solids_test.cc`

**Interfaces:**
- Consumes: Task 5; `transfer::GridFromParticles`, `CellsFromParticles`, `MaterialSetKey`; `materials::AtSolidFaces`.
- Produces `MaterialSetFaces`, `CollideSolidsInputs`, `CollideSolidsResult`, `CollideSolids` (step 1 only here;
  sets with `kParticlesOnly` untouched); Plan decision 7. Face pushes kept per solid and block in the try.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), hand-built `GridFromParticles` on `BuildBoxGrid`:

  ```cpp
  TEST(CollideSolids, SnowFaceTowardPlaten_TakesPlatenNormalSpeed)  // design 7: kNoMotionIntoSolid set; driven
      // platen normal {0,0,-1} at -1 m/s; a Z face 0.3 w below it at +0.2 m/s, mass 0.8 kg: velocity -1; push on
      // the platen +0.96 N s along z; a leaving face and a kParticlesOnly set unchanged; 1e-14 relative
  TEST(CollideSolids, ObliqueSolid_SidewaysSpeedKept)   // design 7: normal {0.6,0,0.8}, face velocity along the
                                                        // surface: unchanged; 1e-14
  TEST(CollideSolids, SetArraysDisagree_IsWrongMaterialSet)  // design 7: CellsFromParticles of another set; one
                                                             // set given twice: error, every velocity byte unchanged
  TEST(CollideSolids, SetWithMixedRules_IsBadInput)
  TEST(CollideSolids, SolidsAtFacesOfOtherTime_IsWrongTime)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.7 step 1 (all reads before writes, through the
  working buffers), ported from `git show main:native/src/dry_solid_step_3d.cc` lines 367-378, generalised to any normal.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Stop snow and springy faces at solids`.

### Task 26: `CollideSolids`, two sets on one face

**Kind:** numerics

**Files:**
- Modify: `native/fluidity/solids/collide_solids_cpu.cc`, `collide_solids_math.h`, `collide_solids_test.cc`

**Interfaces:**
- Completes `CollideSolids` (design 5.7 steps 2 to 6). **Plan decision:** at one face, pairs run in (first set,
  second set) order of `material_sets`, each on the face's values left by the earlier pair; every other face is
  read from the values after step 1.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(CollideSolids, FlatFrontAlongAxis_SharesNormalSpeed)   // design 7: two blocks of two sets meeting on an
      // x plane: X faces take the mass-weighted mean; Y, Z faces unchanged at friction 0; 1e-13 relative
  TEST(CollideSolids, AnyCase_KeepsMomentumPerAxis)           // design 7: 200 random particles in two sets
      // (seed 23) through transfer::ParticlesToGrid: sum of mass * velocity per axis unchanged; 64 eps relative
  TEST(CollideSolids, CoincidentMassCentres_CountedUnclear)  // unclear_normal_count 1, face unchanged
  TEST(CollideSolids, AnyBatchCut_SameBytes)                  // design 7: platen on snow beside liquid, as 1, 2
      // and 7 ranges: face velocities and per-solid pushes identical
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement**; sampling of the two other parts and the normal as design
  5.7 steps 3, 4 (`unclear` below `1e-6 w`). **Step 4: Run** the gate. **Commit.** Subject:
  `Apply the contact rule where two material sets meet`.

### Task 27: `CollideSolids` on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `collide_solids_gpu.h` returning `RunOnCallStatus`; the GPU
function wraps a per-face function of `collide_solids_math.h` that a `cpu` test runs over every thread block and
thread; digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/solids/collide_solids_gpu.cu`, `collide_solids_gpu.h` (private),
  `collide_solids_steps_test.cc` (label `cpu`), `collide_solids_gpu_test.cc`, `collide_solids_no_gpu_test.cc`
- Modify: `collide_solids_cpu.cc` (dispatch under `#if defined(FLUIDITY_HAS_GPU)`), `collide_solids_math.h`
  (`ApplyCollideSolidsToFace`, the whole per-face work of steps 1 to 6 reading the working buffers and writing its own
  face, called by the CPU loop too), `CMakeLists.txt`

- [ ] **Step 1: Tests:** `cpu` (`collide_solids_steps_test.cc`, the main evidence until the hardware run):
  `ApplyCollideSolidsToFace, EveryThreadOfLaunch_EqualsCpuLoop` (Tasks 25, 26 scenes, every thread block and
  thread of `ComputeLaunchSize(face_place_count)`, launch order and reverse: face velocities and block partials
  equal the CPU's; bytes). `gpu`: `CollideSolids, OnGpu_EqualsCpu` (Tasks 25, 26 scenes; design 6),
  `CollideSolids, SameInputTwice_SameBytes`, both with `WriteResultDigest` on face velocities and pushes; `no_gpu`:
  `CollideSolids, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** one thread per face place reading the before-contact values and writing its own
  face (a grid-stride loop around `ApplyCollideSolidsToFace`, launched by `ApplyGpuFunction`; `CheckGpuWork` before
  the counts are read back). **Step 3: Gate; commit.** Subject: `Apply contact between sets on the GPU`. Body: `CUDA: compiled, not run`.

### Task 28: `ApplyContactReactions` on the CPU

**Amended 2026-11-14 (A17, batched-run buffers):** in a batched run the contact reactions are summed on the CPU in a
CPU instance of the buffers while `CollideSolids` wrote its block partials in the loaded GPU instance; a copy of
those partials by range, and per-solid results kept in `SolidsTry`, make that work (step plan Appendix A row M10).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/solids/contact_reactions.h`, `contact_reactions_cpu.cc`, `contact_reactions_test.cc`

**Interfaces:**
- Produces `ReactionFields`, `ContactOnSolids`, `ApplyContactReactions` exactly; adds the `CollideSolids` face
  partials (design 5.6). **Plan decision:** a second call in one try replaces the first.
- **Plan decisions (A17, batched-run buffers; `plan-decisions.md` "Buffers in batched runs"):**
  `std::expected<void, SolidsFailure> WriteBlockPartialsRange(const SolidsBuffers& piece_buffers, const
  grid::GridPiece& piece, SolidsBuffers& whole_buffers);` copies the block-partials row of the range blocks of
  `piece` (piece positions `range()`, whole positions `range_in_whole_grid()`, one contiguous run each) from buffers
  whose partials were written on `piece.grid()` into buffers sized for the whole grid, on either processor pair
  (GPU to CPU with one slice `ReadBuffer`, CPU to CPU with a copy); `kBadInput` when the whole buffers hold fewer
  blocks or a partial row has another moving-solid count. `ApplyContactReactions` copies its per-solid results
  (push, turning push, work) into `SolidsTry` before returning, as `ApplyPressureToSolids` does, so `CommitTry`
  reads stage 12's results from the whole instance and stage 17's from the CPU instance through `SolidsTry`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), reactions hand-built:

  ```cpp
  TEST(ApplyContactReactions, TwoHits_PushAndTurningPush)    // design 7: two 0.1 kg particles hit a driven box
      // at +-0.05 m off centre, velocity change {0,0,2}: push {0,0,-0.4}, turning_push closed form; two equal
      // particles placed symmetrically give zero turning push; 1e-14 relative
  TEST(ApplyContactReactions, ReactionsSplitIntoRanges_SameBytes)   // Review Focus 4: 500 hits in one call, in 3
                                                                    // ranges, reversed: identical bytes
  TEST(ApplyContactReactions, FacePushesOfCollideSolids_Added)
  TEST(ApplyContactReactions, MoreHitsThanBuffers_IsBuffersTooSmall)
  TEST(WriteBlockPartialsRange, ThreeRangesGiveTheWholePartials)  // A17: CollideSolids on three pieces of a 9-block
      // grid, each piece's partials written into one whole CPU instance: equal bytes to CollideSolids on the whole
      // grid; ApplyContactReactions then gives the same ContactOnSolids bytes
  TEST(WriteBlockPartialsRange, WholeBuffersTooSmall_IsBadInput)  // A17
  TEST(CommitTry, ResultsFromTwoInstances_SameAsOne)         // A17: ApplyPressureToSolids with one instance,
      // ApplyContactReactions with a second: committed states equal the one-instance run's bytes
  TEST(CommitTry, ContactPush_AddedAtEndPose)                // free box: velocity after pressure + inverse inertia
                                                             // at the end pose * contact push; 1e-15
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.6, ported from
  `git show main:native/src/liquid_substep_3d_cpu.cc` lines 316-376 and `liquid_substep_3d.cc` lines 2700-2815.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Sum the contact pushes on each solid`.

### Task 29: `ApplyContactReactions` on the GPU

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** private `contact_reactions_gpu.h` returning `RunOnCallStatus`;
compaction, order and grouping through Foundation's functions taking `buffers.order_buffers`; per-hit and
per-solid functions in `contact_reactions_steps.h` run by a `cpu` test; digests in the `gpu` tests.

**Files:**
- Create: `native/fluidity/solids/contact_reactions_gpu.cu`, `contact_reactions_gpu.h` (private),
  `contact_reactions_steps.h` (private, shared: `ComputeHitFlagOfParticle`, `CopyHitOfParticle`,
  `ComputeReactionSumOfSolid` (a `math::TreeSumOf` over the solid's hits in `particle_id` order)),
  `contact_reactions_steps_test.cc` (label `cpu`), `contact_reactions_gpu_test.cc`, `contact_reactions_no_gpu_test.cc`
- Modify: `contact_reactions_cpu.cc` (the CPU path calls the same per-hit and per-solid functions; dispatch under
  `#if defined(FLUIDITY_HAS_GPU)`), `CMakeLists.txt`

- [ ] **Step 1: Tests:** `cpu` (`contact_reactions_steps_test.cc`, the main evidence until the hardware run):
  `ContactReactionsSteps, EveryThreadOfLaunch_EqualsCpuSums` (Task 28's 500 hits: flag, `ComputeExclusivePrefixSum`,
  copy, then the per-solid sums, for every thread block and thread, launch order and reverse; bytes equal the CPU
  path's). `gpu`: `ApplyContactReactions, OnGpu_EqualsCpu`, `ApplyContactReactions,
  SameInputTwice_SameBytes`, `ApplyContactReactions, ParticlesRegrouped_SameBytes` (design 7: the same hits after
  a regrouping), each with `WriteResultDigest` on the per-solid push, turning push and work; `no_gpu`:
  `ApplyContactReactions, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement:** compact the hits (flag GPU function, `ComputeExclusivePrefixSum(...,
  buffers.order_buffers)`, copy GPU function), two passes of `BuildStableOrder(..., buffers.order_buffers)` (id,
  then solid, the second given the first as previous order), `BuildGroupsByDestination(...,
  buffers.order_buffers)` by solid, reduce per solid (one thread per solid and component). Nothing allocates.
- [ ] **Step 3: Gate; commit.** Subject: `Sum contact pushes on the GPU`. Body: `CUDA: compiled, not run`.

### Task 30: `DriveWaves`

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the GPU half is declared in `drive_waves_gpu.h` and returns
`RunOnCallStatus`; the block partial of one block is a shared function run by a `cpu` test over every thread block
and thread; digest in the `gpu` test.

**Files:**
- Modify: `native/fluidity/solids/drive_waves.h`, `CMakeLists.txt`
- Create: `drive_waves_cpu.cc`, `drive_waves_gpu.cu`, `drive_waves_gpu.h` (private), `drive_waves_steps.h`
  (private, shared: `ComputeDriveWavesPartialOfBlock`, the momentum and energy added in one block, its faces in
  ascending face place), `drive_waves_test.cc`, `drive_waves_gpu_test.cc`, `drive_waves_no_gpu_test.cc`

**Interfaces:**
- Consumes: `pressure::ReadyOpenSea`, `BuildOpenSea` (its `OpenSeaSettings::domain_box` set to the test grid's
  domain), `ApplyOpenSeaRelaxation`, `ComputeRelaxationWeight`.
- Produces `DriveWavesInputs`, `DriveWavesResult`, `DriveWaves` exactly; Plan decision 8.

- [ ] **Step 1: Tests**, grid 8 x 4 x 4 of 0.1 m, outlet `kRight` zone 0.2 m, rate 3, dt 0.01, face mass 1 kg:

  ```cpp
  // LABEL cpu
  TEST(DriveWaves, UniformPull_MomentumIsMassTimesChange)  // design 7 (plan decision: an outlet zone at 1 m/s
      // pulled toward rest, because OpenSeaWave has no uniform current): momentum_added.x == -sum ComputeRelaxation
      // Weight(x_f), energy_added == sum ((1 - weight)^2 - 1) / 2; 1e-13 relative
  TEST(DriveWaves, AnyBatchCut_SameBytes)                  // design 7: one range and 5 ranges, block partials
                                                           // tree-summed by the test: identical totals
  TEST(DriveWaves, NoOpenSea_ChangesNothing)
  TEST(ComputeDriveWavesPartialOfBlock, EveryThreadOfLaunch_EqualsCpuPartials)  // amended (Foundation as built):
      // every thread block and thread of ComputeLaunchSize(block_count), launch order and reverse; bytes; the main
      // evidence until the hardware run
  // LABEL gpu
  TEST(DriveWaves, OnGpu_EqualsCpu)                        // design 6; WriteResultDigest on face velocities and
                                                           // the two totals
  // LABEL no_gpu
  TEST(DriveWaves, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.8 (`CopyBuffer` of the zone velocities, the call,
  block partials in ascending face place through `ComputeDriveWavesPartialOfBlock`, on the GPU one thread per
  block launched by `ApplyGpuFunction`; the copy lives in `SolidsBuffers`' before-relaxation row, so nothing
  allocates). **Step 4: Gate; commit.** Subject: `Drive waves and measure what
  they add`. Body: `CUDA: compiled, not run`.

### Task 31: Plate sides and plate removal

**Kind:** device

**Amended 2026-11-14 (Foundation as built):** the GPU half is declared in `plate_sides_gpu.h` and returns
`RunOnCallStatus`; the per-particle work is a shared function run by a `cpu` test over every thread block and
thread; digest in the `gpu` test.

**Files:**
- Modify: `native/fluidity/solids/plate_sides.h`, `plate_sides_cpu.cc`, `CMakeLists.txt`
- Create: `plate_sides_gpu.cu`, `plate_sides_gpu.h` (private), `plate_sides_steps.h` (private, shared:
  `FindPlateSideOfParticle` and the failure key of one particle, `ApplyPlateRemovalToParticle`),
  `plate_sides_steps_test.cc` (label `cpu`), `plate_sides_test.cc`, `plate_sides_gpu_test.cc`,
  `plate_sides_no_gpu_test.cc`

**Interfaces:**
- Produces `PlateSideFields`, `ApplyPlateSides`, `ApplyPlateRemoval` exactly.

- [ ] **Step 1: Tests** (`LABEL cpu` unless stated):

  ```cpp
  TEST(ApplyPlateSides, EachSide_GetsItsSide)                  // design 7: count returned; exact
  TEST(ApplyPlateSides, InsideThickness_IsParticleInsidePlate) // design 7: lowest particle_id of two inside
  TEST(ApplyPlateSides, AlreadyGiven_IsKept)                   // design 7
  TEST(ApplyPlateRemoval, AllSidesOfThePlate_Cleared)          // sides of another plate id untouched
  TEST(ApplyPlateSides, MaterialWithoutPlateSide_IsMaterialNotCarried)
  // plate_sides_steps_test.cc, LABEL cpu (the main evidence until the hardware run)
  TEST(FindPlateSideOfParticle, EveryThreadOfLaunch_EqualsCpuLoop)  // the five cases above, every thread block
      // and thread, launch order and reverse: sides, count and lowest failing particle_id equal the CPU's
  // LABEL gpu / no_gpu
  TEST(ApplyPlateSides, OnGpu_EqualsCpu)                       // exact; WriteResultDigest on the sides
  TEST(ApplyPlateSides, OnGpuWithoutAGpu_IsRunOnCallFailed)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** `<proc-apply-plate-sides>`, ported from
  `git show main:native/src/liquid_wall_contact_3d.cc` lines 592-683 (classification only).
- [ ] **Step 4: Gate; commit.** Subject: `Give particles the side of the plate`. Body: `CUDA: compiled, not run`.

### Task 32: `ComputeCutWork`

**Kind:** numerics

**Amended 2026-11-14 (Foundation as built):** the per-entry cut work is a shared function in `plate_sides_steps.h`
that the GPU function wraps and a `cpu` test runs over every thread block and thread; digest in the `gpu` test.

**Files:**
- Modify: `native/fluidity/solids/plate_sides.h`, `plate_sides_cpu.cc`, `plate_sides_gpu.h`, `plate_sides_gpu.cu`,
  `plate_sides_steps.h` (`ComputeCutWorkOfBlock`), `plate_sides_steps_test.cc`
- Create: `cut_work_test.cc`

**Interfaces:**
- Produces `CutWorkFields`, `CutWork`, `ComputeCutWork` exactly.

- [ ] **Step 1: Failing tests:**

  ```cpp
  // LABEL cpu
  TEST(ComputeCutWork, UncutBlock_IsZero)   // design 7: 27 particles in a 3^3 block of BuildBoxGrid, affine
      // velocity on faces (L = {{0.1,0.2,0},{0,-0.1,0.3},{0.05,0,0}}), constant stress; face_force_work_rate of
      // each particle from transfer::ParticlesToGrid run for that particle alone (its face forces times the
      // face velocities, per axis): |cut_work| <= 1e-6 |deformation_work|
  TEST(ComputeCutWork, EntriesOfDifferentBlocks_IsBadInput)
  TEST(ComputeCutWorkOfBlock, EveryThreadOfLaunch_EqualsCpuLoop)  // amended (Foundation as built), in
      // plate_sides_steps_test.cc: launch order and reverse; bytes
  // LABEL gpu
  TEST(ComputeCutWork, OnGpu_EqualsCpu)     // design 6; WriteResultDigest on the cut work
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.9 cut work, ported from
  `git show main:native/src/elastic_blade_deformation_3d.cc` lines 176-233.
- [ ] **Step 4: Gate; commit.** Subject: `Report the work the cut absorbs`. Body: `CUDA: compiled, not run`.

### Task 33: `CheckSolidExchange`, and a rejected try after every stage

**Kind:** integration

**Files:**
- Create: `native/fluidity/solids/exchange_check.cc`, `exchange_check_test.cc`
- Modify: `exchange_check.h`, `solids_try.cc` (totals added on commit), `solids_try_test.cc`

**Interfaces:**
- Produces `ExchangeInputs`, `CheckSolidExchange` exactly. **Plan decisions:** the liquid side of check 1 and
  check 3 is computed from `geometry.face_pieces` (pieces touching a `kLiquid` cell; pressure of any other cell
  counts 0); check 4 fails `kExchangeNotBalanced` with `magnitude = contact_tolerance / 2 -
  smallest_end_distance`; `particles_inside_solid` is 1 when the smallest end distance is below the limit (the
  step has the count). `CommitTry` adds `ExchangeTotals` (turning pushes moved to the world origin).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(CheckSolidExchange, BrokenMap_IsExchangeNotBalanced)  // design 7: Task 21 scene passes; one entry's
                                                             // signed_corner_area changed by 1 %: error
  TEST(CheckSolidExchange, EndDistanceInsideSolid_IsExchangeNotBalanced)
  TEST(RejectTry, AfterEveryStage_ChangesNothing)  // design 7: free box and a driven plate with snow and liquid:
      // every stage call of this module in step order (shared-interface e), then RejectTry: ReadSnapshot bytes
      // equal those before. Amended 2026-11-14 (A26, A27): run twice, once with the stage 12 calls and once
      // with the stage 11 calls (BuildViscousMaterialFaces, ToFreeSolidsForViscousSolve, ApplySolvedMotionToSolids
      // and ApplyViscousSolveToMaterials with hand-filled solved values)
  TEST(CommitTry, Totals_AddedOnCommitOnly)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.10. **Step 4: Run** the gate. **Commit.** Subject:
  `Check what solids and material exchange in a step`.

### Task 34: A hand-written liquid step loop for scene tests

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Kind:** integration

**Amended 2026-11-14 (Foundation as built):** the library is made with `fluidity_add_test_library` and linked by
tests through `LINKS`.

**Files:**
- Create: `native/fluidity/solids/solids_liquid_loop.h`, `solids_liquid_loop.cc` (test-only library:
  `fluidity_add_test_library(fluidity_solids_liquid_loop SOURCES solids_liquid_loop.cc LINKS
  fluidity_solids_test_scenes fluidity_base_test_support)`; Tasks 35 and 36 name it in `LINKS`),
  `solids_liquid_loop_test.cc`
- Modify: `native/fluidity/solids/CMakeLists.txt`

**Interfaces:**
- Produces (**Plan decision**, `namespace fluidity::solids::testing`): `LiquidLoopScene { grid; domain_box;
  SolidsSettings solids; std::vector<transfer::test::ParticleSpec> particles; double density = 1000;
  double step_duration; const pressure::ReadyOpenSea* open_sea = nullptr; }`, `class LiquidLoop` built by
  `BuildLiquidLoop(const LiquidLoopScene&)` (CPU only, one liquid material in one set, the grid fixed for the
  run), `std::expected<LiquidLoopStep, SolidsFailure> RunStep()`, and `LiquidLoopStep { SolidsSnapshot solids;
  PressureOnSolids pressure; ContactOnSolids contact; ExchangeReport exchange; Vec3 liquid_momentum;
  Vec3 liquid_turning_momentum; double filled_volume_above(double height) const; }` (copies of the values).
- One step, in this order: `BuildTry`; `ComputeSolidsAtFaces` (0), `ComputeSolidCoverage`,
  `FindGridEdgesOpenToSky`; `ParticlesToGrid`, `SaveGridBeforeForces` (each taking one `TransferBuffers`
  built with the loop); gravity on `kFromParticles` Z faces;
  `DriveWaves` when `open_sea`; `ComputeDistanceToSurface`, `FindLiquidSurface`, `FindAirPockets` (no committed
  pockets), `BuildMotionMap`, `BuildPressureEquations` (density uniform, `bulk_stiffness` +infinity, carried 0,
  free-solid pushes), `SolvePressure`, `ComputeWasSolved`; `ApplyPressureToSolids`, `ComputeSolidsAtFaces`
  (`dt`); `SaveGridAfterPressure`, `GridToParticles`, `MixParticleVelocity` with the liquid defaults of
  `materials::OwnMotionSettings`; `contact::MoveParticles` with `SolidsTry::solids()`, friction 0;
  `ApplyContactReactions`; `CheckSolidExchange` (push floor `1e-12 * total mass * largest speed`); solids
  `CommitTry`, particles `CommitTry`, regroup.

- [ ] **Step 1: Failing tests** (`LABEL cpu`): `LiquidLoop, StillWaterInFixedTank_StaysStill` (8 x 4 x 8 cells
  of 0.05 m, half full, 20 steps of 0.005 s: largest particle speed below `1e-3` m/s, exchange passes each step);
  `LiquidLoop, RepeatRun_IdenticalBytes` (two runs, every `LiquidLoopStep` equal).
- [ ] **Step 2: Implement; run.** **Step 3: Gate; commit.** Subject: `Add a liquid step loop for solids scene
  tests`. Body: the stage list above, and "the step module replaces this loop with `TryStep`".

### Task 35: Scene tests, a floating box

**Kind:** integration

**Files:**
- Create: `native/fluidity/solids/solids_scene_test.cc`

The tank is `AddTank(id 1, ..., kDriven)` (never moves; its reactions are reported, so balances close).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(FreeBox, HalfDensity_FloatsHalfSubmerged)   // design 7: cube of 16 cells per side (w 0.0125, side 0.2 m),
      // 500 kg/m3, tank 0.4 x 0.4 x 0.3 m, water 0.15 m deep, box placed at half draught, 50 steps of 0.002 s:
      // draught within 2 % of 0.1 m at every step; each step, change of liquid + box momentum (linear and turning
      // about the world origin) equals gravity impulse plus the tank's pressure and contact pushes within 1e-9
      // relative
  TEST(FreeBox, OffCentreSplash_ReactionEqualsLiquidChange)  // design 7: 8 cells per side, a jet of 200 particles
      // at 1 m/s down onto the box 0.05 m off centre: over 30 steps, box pressure + contact push equals minus
      // (liquid momentum change - gravity impulse on the liquid - tank pushes) within 1e-9 relative
  ```
- [ ] **Step 2: Run.** Record the debug-preset run time in the commit body; above 120 s, stop and ask the owner
  before lowering a size. If a test fails, fix the owning commit, not the test.
- [ ] **Step 3: Gate; commit.** Subject: `Test a free box floating in liquid`.

### Task 36: Scene tests, swept volume and a plate that leaves

**Kind:** integration

**Files:**
- Modify: `native/fluidity/solids/solids_scene_test.cc`, `docs/known-open-physics.md`

- [ ] **Step 1: Tests** (`LABEL cpu` unless stated):

  ```cpp
  TEST(DrivenFlatSolid, SweptVolume_IsReceivedByLiquid)  // design 7, LABEL known_open, WILL_FAIL: tank 0.8 x 0.1 x
      // 0.4 m, w 0.025, water 0.2 m deep, open top, paddle normal {1,0,0} at x 0.05 moving +x at 0.5 m/s, 40
      // steps of 0.0025 s: ExchangeTotals::swept_volume against filled_volume_above(0.2) within 1 %
  TEST(Cut, PlateLeaves_SidesRejoin)      // design 7: one material on both sides of a driven plate (kLeavesScene,
      // last key at 0.01 s); after the commit that ends the path: ApplyPlateRemoval, ComputeSolidsAtFaces,
      // transfer::ParticlesToGrid: every face within 2 w of the old plane has sides_reaching {{0,kNoPlate}}, count 1,
      // and mass from particles of both former sides
  ```
- [ ] **Step 2: Run.** Expected: the known-open test fails as `main` did (7.6 %), CTest reports it passed. Add
  its line to `docs/known-open-physics.md` (name, what it measures, decision 4 of the design).
- [ ] **Step 3: Gate; commit.** Subject: `Test the swept volume and a plate that leaves`.

### Task 37: Comparison with the proof of concept

**Amended 2026-11-14 (A31):** the old numbers are produced in a temporary directory.

**Kind:** integration

**Files:**
- Create: `native/fluidity/solids/testdata/coupled_turned_box.json`, `proof_of_concept_test.cc`

- [ ] **Step 1: Produce the old numbers** (plan-decisions: export `main` with `git archive` to a temporary directory,
  build its CPU preset there): run `main`'s coupled turned box (`liquid_substep_3d.cc` lines 1806-1993) for one
  step on its same-size grid with a flat surface; print the velocity after pressure; turn it into Z-up
  (`(x, y, z) -> (x, -z, y)`); store it with the `main` commit hash and the scene. The driver is not committed;
  its outline goes in the commit body.
- [ ] **Step 2: Test** (`LABEL cpu`) `ProofOfConcept, CoupledTurnedBox_OneSolveMatches`: the same scene through
  `BuildMotionMap`, one `SolvePressure` with `FreeSolidPush`, `ApplyPressureToSolids`: velocity after pressure
  within the solve limit. Supporting only: Tasks 21 and 22 are the analytic tests of this path.
- [ ] **Step 3: Gate; commit.** Subject: `Compare the free-solid solve with the proof of concept`.

### Task 38: Comment pass and hand-over

**Amended 2026-11-14 (A31):** working memory is named buffers (`OrderBuffers order_buffers`, `TransferBuffers`); each base order, prefix sum and grouping has one form, which takes its buffers.

**Amended 2026-11-14 (A18):** the design check records plan decision 9 and Task 21's change.

**Amended 2026-11-14 (A26, A27):** the design check also records plan decision 10, the decisions of Tasks 23a,
23b and 23c (the band rule, every free solid listed, `SolvedMotionReport` and the two `SolidsTry` accessors, guides
dropped after the solve), and lists the guide question for the owner (Task 23c decision 4).

**Kind:** mechanical

**Amended 2026-11-14 (Foundation as built):** the pass also checks `Uses:` (never a standard header), `Words:`
(only where a file introduces domain words) and "thread block" for the GPU sense; the hand-over records the
Foundation adoptions.

**Files:**
- Modify: any file of the module; `docs/gpu-runs.md`

- [ ] **Step 1: Comment pass.** Reread every file of `native/fluidity/solids/` top to bottom as a C++ programmer
  who has never read a fluid paper: each comment true of the final code; says why, not what; a worked example
  with numbers for each non-trivial public function (Task 22's lid for the motion map, Task 25's platen for
  `CollideSolids`); every word in the glossary; `Uses` (project and library headers only, never a standard header)
  and `Used by` lists correct; `Words` only where a file introduces domain words; "thread block" (never "block"
  alone) for the GPU sense, the cell block keeping its glossary name; `// Step N:` numbers match
  the paper's procedures; the sign of work stated in each header that reports work.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit, the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently: the
  Global Constraints' plan decisions 1 to 10 (9: design decision 3 overturned by owner answer A18; 10: A26, A27),
  the sign of work, the decisions of Tasks 4, 7, 9, 10, 14, 16, 19, 21, 23, 23a, 23b, 23c, 26, 28, 30, 33, 34,
  the known-open test, the A17
  batched-run buffer changes of Tasks 13 and 28 (amended 2026-11-14: several instances, `WriteBlockPartialsRange`,
  results kept in `SolidsTry`), and the scene tests left to the step plan (below); (Foundation as built) the
  `OrderBuffers` and solids copy in `SolidsBuffers` with their rows, `ToSolidsFailure`, the private `_gpu.h`
  headers and the shared step headers with their `cpu` tests, `open_to_sky.cc` renamed `open_to_sky_cpu.cc`.
- [ ] **Step 4: `docs/gpu-runs.md`:** a `solids` row, result `compiled, not run`, listing every `gpu` test of
  Tasks 18, 19 (A17), 24, 27, 29, 30, 31, 32.
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate's output, new public
  names with reasons, the plan decisions, and the statement that no device code has run.

---

## Tests the step plan must add

These need `TryStep` (materials' response inside a step, gas, batches, saved state, step-duration control or a
long run). The design's numbers are the contract.

1. `WavePaddle, RegularWave_SpeedAndHeight`: wave height 2 % of depth, 3 wavelengths, paddle from
   `BuildWavePaddlePath`, outlet zone: phase speed within 5 % of `omega^2 = g k tanh(k depth)`; height within 10 %;
   reflection from the outlet below 10 %; volume drift below 0.1 %; `flow_work` (given by the paddle, this plan's
   sign) equals the gain of wave energy plus `DriveWavesResult::energy_added` within 15 %.
2. `Cut, PlateThroughBlock_SidesNeverMix`: springy block, driven plate; the same run restarted from a saved state
   in the step after the first hit, with a rejected try before it, and with a split-and-merge pass on a liquid
   beside it: identical bytes to the uninterrupted run; no particle changes side; the plate's `driven_work`
   equals elastic energy gained plus `cut_work` plus kinetic energy within 5 %.
3. `Platen, SnowBesideWater_SetsKeepTheirVelocities` (amended 2026-11-14, A18): a platen presses a snow bed with
   water beside it; two sets; no water face takes the snow's velocity except through `CollideSolids` steps 2 to
   6; the snow faces the water presses on move in the same step's pressure solve (their velocity after equals
   `pressure::SolvedArrays::blocking_material_velocity_after_pressure`); momentum of water plus snow plus platen
   changes only by gravity, the domain and the platen's push.
4. `FreeBox, FallsThroughGas_NotPushedByGas`: a free box falls through a gas-filled box: free fall within
   `1e-13` relative while the gas flows around it (gas sees `solid_volume_rate`; marker hits are not summed).
5. `Batches, FloatingBox_WithinStepToleranceOfOnePiece` (renamed 2026-11-14, A15; was
   `..._SameBytesAsOnePiece`): `FreeBox, HalfDensity_FloatsHalfSubmerged` run in both batched modes agrees with the
   one-piece run within the step design 6.4 tolerances for every solid state and `ExchangeTotals`, and gives
   identical bytes for every cut of one mode.
6. `FreeBox, HalfDensity_FloatsHalfSubmerged` and `FreeBox, OffCentreSplash_ReactionEqualsLiquidChange` again
   through `TryStep` (dropped from rest, settling with step-duration control), and every scene of Tasks 35 and 36
   on `RunOn::kGpu` (label `gpu`) against the CPU within design section 6.
7. `SpringyBlock, FifthDensity_FloatsAtDisplacedDepth` (added 2026-11-14, A18; the test design decision 3 named
   as the one that decides): a springy cube of side 0.1 m and density 200 kg/m3 (0.2 of the water) set on still
   water, steps of 0.01 s (the step design's `max_step_duration`): its mean bottom height settles to 0.02 m below
   the still surface within half the smallest cell width; its vertical velocity amplitude over the last 0.2 s is
   no larger than over 0.2 to 0.4 s. Pressure's Task 49 shows the lagged form unstable at a tenth of this step.
8. `TryStep, FiveMaterialsWithSnowAndFreeBox_Commits` (added 2026-11-14, A26, A27): the step's five-material
   scene with snow, a springy solid, a viscous liquid (stage 11) and a free box beside the syrup commits every
   step; in stage 11 the step calls `BuildViscousMaterialFaces` and `ToFreeSolidsForViscousSolve` (stage 10),
   `ApplySolvedMotionToSolids` and `ApplyViscousSolveToMaterials` (no `ApplyPressureToSolids`, no
   `ApplyPressureToMaterials`), and `CheckSolidExchange` passes with `SolidsTry::pressure_on_solids()`. Snow is
   checked here for not breaking only (no melting is modelled; owner note, 2026-11-14).
9. `TryStep, FloatingBoxInSyrup_RestDepthAsInWater` (added 2026-11-14, A27): `FreeBox, HalfDensity_FloatsHalfSubmerged`
   with the liquid given a viscosity above 0 (stage 11): the box settles at the same draught as in water (0.1 m
   within 2 %); the viscosity changes how it gets there, not where it rests.
