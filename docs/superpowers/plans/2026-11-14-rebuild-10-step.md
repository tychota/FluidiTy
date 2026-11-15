# Rebuild Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the step module on the `rebuild` branch: the `Simulation` that owns every module's committed state,
one `TryStep` for every scene that runs the 21 stages on a try and commits by swapping, the capability check,
the sizing plan, the step duration, `AdvanceTo` with retries and limit raises, snapshots and a continued run,
and the hand-over seams for cell resize, batches, I/O and render output (callbacks, tested here with
hand-written callback functions), on the CPU and (compiled, not run) on the GPU. Amended 2026-11-14 (A17,
batched-run buffers): every buffer of the three run modes has one place (Appendix A), every copy between CPU memory
and the run's processor is planned and counted (Appendix B), and the plan reports where its bytes live, so I/O can
show the planned cost.

**Architecture:** Folder `native/fluidity/step/`, namespace `fluidity::step`. Public headers are the ones of
design section 3 (`step_error.h`, `step_message.h`, `ready_scene.h`, `step_settings.h`, `capability.h`,
`run_plan.h`, `simulation.h`, `try_step.h`, `step_duration.h`, `advance_to.h`, `resize_access.h`,
`step_stages.h`). The six step-owned operations (`add_new_particles`, `choose_active_cells`, `apply_forces`,
`move_particles`, `check_step`, `commit_step`) each have one header and private `_cpu.cc` / `_gpu.cu` files;
every other stage is glue that calls another module's section 3 function. `TryStep` is exactly
`TryStepWithStages(simulation, step_inputs, kStages)`; the module is built up one stage at a time, so after each
task `TryStep` runs a longer part of the stage list on a real scene, and each task proves with a stage table
that a try rejected after its new stage changes no committed byte.

**Tech Stack:** as Foundation (C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 with GoogleTest,
uv, clang-format and clang-tidy 23.1 as `pyproject.toml` pins them, Docker via colima, Typst with CeTZ). Nothing
new. (Amended 2026-11-14, Foundation as built: replaced "19.1".)

**Spec:** `docs/superpowers/specs/modules/10-step-design.md` ("the design"; in the `main` checkout, never
committed; its section 3 is the contract) and the programme design `docs/design/programme.md` on `rebuild`
(sections 6, 7, 8, 8a). Also in the `main` checkout: `modules/owner-answers.md`, `modules/shared-interface.md`,
`modules/plan-decisions.md` (it wins over every design), and section 3 of the contact, grid, particles,
transfer, pressure, viscosity, materials, solids and gas designs.

## Global Constraints

**Amended 2026-11-14 (A28, A30, A31):** `StepInputs`; one form per Foundation function (taking its buffers); arguments on one processor are a precondition, not an error.

**Amended 2026-11-15 (A40, A41, A43, A45, A47, A57, A58, A60, A61-A64, reconcile 31-33):** one reserved block per processor, a budget of free memory, the shared cell size rule, particles everywhere, viscosity advice, liquid force on solids, the forced run mode; the source is `plan-decisions.md` "Owner answers A40-A64 applied (2026-11-15)" with its same-day correction, and design 11.39 to 11.47.

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
- **Amended 2026-11-14 (Foundation as built):** Foundation's structure patterns apply to every task (`AGENTS.md`
  "Structure"; `plan-decisions.md` "Foundation after the final review"). Two halves: the public function lives in
  `<name>_cpu.cc` (checks, `RunOn` dispatch, the CPU loop, the GPU half called under `#if
  defined(FLUIDITY_HAS_GPU)`); the GPU half is `<name>_gpu.cu`, declared in the private `<name>_gpu.h`, returns
  `RunOnCallStatus`, which the CPU half maps with `ToBufferError` and then into `StageOutcome` through Task 3's one
  `ToStageOutcome(BufferError)`; launches, waits and temporary GPU memory only through `base/gpu_launch.cuh`
  (`ComputeLaunchSize`, `ApplyGpuFunction`, `CheckGpuWork`, `GpuArray`). Every GPU function is a thin loop over
  `FLUIDITY_CPU_AND_GPU` per-item functions in a shared header (an existing formula header, else
  `<name>_steps.h`); a `cpu` test runs them for every thread block and thread of `ComputeLaunchSize(count)`, in
  launch order and in reverse, against the CPU loop: the main evidence until the hardware run. Integer atomics are
  allowed; floating-point atomics never. Working buffers: per-try sums and orders pass the step's own
  `SumBuffers` and `OrderBuffers` (each Foundation function has one form, which takes its buffers; A31) (members `sum_buffers`, `order_buffers` of `TryArrays`, Task 9), built once at the
  limits and counted in the memory estimate with `ComputeSumBuffersSize` / `ComputeOrderBuffersSize`; every other
  module's working buffers live in its own buffers and its own `Build<Module>MemoryEstimate`. Errors: a failed check writes
  nothing; after `kRunOnCallFailed` the outputs are unspecified and the run stops (`kRunOnCallFailed` is
  `kCannotContinue`); `kOutOfMemory` comes only from building forms. Same processor (A30): the buffers, views and working buffers of one call live on one
  processor, a `Requires:` precondition checked by an assertion in debug builds, not an error value; only the
  user's processor choice (`StepSettings::run_on`) is checked and reported. Exactness: discrete outputs equal on CPU and
  GPU; floating-point within the design 6.4 tolerances (`CheckWithinTolerance`) unless a header promises the same
  bits unless NaN (`CheckSameBitsOrBothNaN`). Tests link `fluidity_base_test_support` (`ToBits`,
  `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`, `CheckSameBitsOrBothNaN`, the fixed generators, digests)
  and the step's `fluidity_step_test_support` (made with `fluidity_add_test_library`, named in `LINKS`); every
  `gpu` test calls `WriteResultDigest` on its main outputs so the hardware run's repeat check covers it. Labels:
  `cpu` everywhere, `no_gpu` only for behaviour on a machine without a GPU, `gpu` only on hardware. Every target
  is created through the helpers of `cmake/fluidity_targets.cmake`; every source file is listed in a CMake file.
- **Amended 2026-11-14 (Foundation as built):** the gate runs at every commit: the CPU build in the debug and
  release presets, and the CUDA build compiled in the container, whose `cuda` step fails for a tracked `.cc` or
  `.cu` file that no build compiles. clang-format and clang-tidy are the 23.1 that `pyproject.toml` pins.

The step's own:

- **Design section 3 is the contract.** Spell every name as there, with the corrections of `plan-decisions.md`:
  there is **no** `CapabilityReason::kLiquidsThatDoNotMixTouch` (pressure supports two liquids that do not mix;
  `PressureError::kMaterialBoundaryNotSupported` does not exist) and **no** `CapabilityReason::kViscousLiquidWithSnow`
  (A24: snow beside a viscous liquid is carried by the joint solve) and **no**
  `CapabilityReason::kViscousLiquidWithFreeSolid` (A27: a free solid's motion is an unknown of the viscous solve),
  so `kCapabilityReasonCount` is 16 at most, lowered by one for each viscosity exclusion lifted on the branch
  ("Before you start", step 2 of the check: 13 to 16; replaced: "17 at most ... 14 to 17", A24 count);
  **no** `SuggestedChange::kSetViscosityToZero` (A27: no reason in the main sequence uses it; the sand plan adds
  it back with `kViscousLiquidWithSand`);
  `kRejectReasonCount = 23`, `kSuggestedChangeCount = 12` (replaced: 13), `kStepLimitCount = 11`, `kStageCount = 22`,
  `kModuleCount = 14`. Names the rules check reports under `verb` (`TryStep`, `TryStepWithStages`, `AdvanceTo`,
  `CommitResize`, `RejectResize`) go into `tools/check_rules_allowed_names.txt` with the reason
  `decided step name (step design 3, 9)`.
- **Constants:** `kMaxTravelInCellWidths = 0.5`, `kCoveredReachInCellWidths = 2.5`; `StepSettings` defaults of
  design 3.3 (`max_step_duration` 0.01 s, `max_retries_per_step` 2, `max_limit_raises_per_step` 4,
  `limit_growth` 1.5, `start_limit_headroom` 1.25, `memory_share_of_free` = `kDefaultMemoryShareOfFree` 0.9,
  `memory_left_free_bytes` = `kDefaultMemoryLeftFreeBytes` 536,870,912; A58, replaced: "`memory_share_of_total`
  0.7, `memory_share_of_free` 0.8").
- **One reserved block per processor (A58; `plan-decisions.md` item 3 and its same-day correction).** Every
  `kRunOn` buffer is a piece of the simulation's `run_memory`; in a GPU run every `kCpu` buffer kept `kWholeRun` or
  `kOneStepTry` is a piece of `cpu_run_memory` (`CpuMemory::kLockedForCopies` in a batched GPU run); in a CPU run one
  block holds both and `cpu_run_memory` is it. Plain `BuildBuffer` only for `kOneCall` / `kOneSnapshot` CPU arrays
  and tests. Every module builder the step calls gets the block of the processor its buffers live on as its last
  argument (a builder for both processors: `run_memory, cpu_run_memory` last); every step-owned private builder
  (`BuildCommittedState`, `BuildTryArrays`, `BuildLoadedTry` in `simulation_state.h`) takes `RunMemory& run_memory`
  last, or `RunMemory& run_memory, RunMemory& cpu_run_memory` when it builds on both. `BufferError::kRunMemoryTooSmall`
  and `kOutOfMemory` of a builder become `StepError::kOutOfMemory` (Task 3's `ToBuildFailure`).
- **Particles everywhere (A64).** No grid-only region: every task's grid-only piece is in the group "Deep liquid
  without particles (later; needs review)" at the end of this plan (Task 39a), excluded from the done-criteria.
- **Shared cell size rule (A41, A61).** Stage 2 calls `grid::ComputeNewCellSizeLevel` (grid plan, `cell_size_rule.h`)
  for new cells on the committed tables' processor; there is no `choose_cell_size` callback and no copy for it.
- **Stage order and calls:** design 5.2, items 1 to 21, with the material-set order of shared-interface (d):
  every liquid in one set, the gas material alone in its set, sets visited in ascending lowest
  `material_index`. Each stage function body is numbered `// Step N:` with the 5.2 item's letters
  ((a), (b), ...) where it has them. Code comments name the design item, the paper label and the `main` lines
  the rule is ported from, where the design gives them.
- **Coupling inside the liquid solve (owner answer A18; tasks marked "Amended 2026-11-14 (A18)").** Stage 10
  calls, before `pressure::BuildPressureEquations`: `solids::BuildBlockingMaterialFaces` (scene with liquid and a
  blocking set) and `gas::ApplyGasStepStart` (scene with gas and liquid); their outputs go into `EquationInputs`
  (`blocking_material_faces`, `sealed_gas_blobs`, `sealed_gas_blob_of_cell`, `gas_pressure_above_background`).
  Stage 12 passes the step-owned spans `SolvedArrays::blocking_material_velocity_after_pressure` and
  `sealed_gas_pressure`, then calls `solids::ApplyPressureToMaterials` right after `ApplyPressureToSolids` (no
  longer in stage 17). Stage 13 passes `sealed_gas_pressure` to `SolveGas`. In a gas scene without stage 10,
  `ApplyGasStepStart` runs at
  the start of stage 13. Step-owned rows: `"step: material face velocity after pressure"` (given solids'
  `max_motion_map_entries`, 8) and `"step: sealed gas pressure"` (given `max_gas_blobs`, 8). The liquid's
  `pressure::BufferSizes` sets `max_blocking_material_faces` to solids' `max_motion_map_entries` (0 without a
  blocking set) and `max_sealed_gas_blobs` to gas's `max_gas_blobs` (0 without gas).
- **Viscous liquids carry blocking materials and sealed gas (owner answer A24, option C of reconcile item 4;
  tasks marked "Amended 2026-11-14 (A24)").** Stage 11 mirrors stage 12:
  `viscosity::SolveViscosityAndPressure(..., fields.velocity_after_pressure, fields.pressure,
  viscosity::SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure})`, then
  `ComputeWasSolved`, `ApplyPressureToSolids`, then (scene with a blocking set) `solids::ApplyPressureToMaterials`,
  exactly as stage 12. Stage 10 builds the material faces and the sealed blob list whatever stage follows. Stage 13
  passes `sealed_gas_pressure` after either stage. `viscosity::ViscosityBufferSettings::max_sealed_gas_blobs` = gas's
  `max_gas_blobs` (0 without gas). No `gas::SealedGasCoupling` exists: `ApplyGasStepStart` and `SolveGas` take no
  coupling argument, and the gas step limit is the flow-share limit only (no `ComputeSealedGasStepLimit`). Open
  gas, slivers, cells that touch gas and held blobs are still read one step late through `gas_seen`. Snow beside a
  viscous liquid is no longer excluded. Replaced (A27): "a free solid beside a viscous liquid stays excluded
  (`kViscousLiquidWithFreeSolid`: the free solid's velocity would be an unknown of the viscous velocity solve,
  which A24 does not cover)". No exclusion for a springy solid beside a viscous liquid exists, and every test once
  marked blocked by that decision runs as written. The stage 11 calls of this paragraph are amended by A26/A27
  below (`ApplyPressureToSolids` and `ApplyPressureToMaterials` are no longer called by stage 11).
- **A viscous liquid sticks to materials, and free solids are in the viscous solve (owner answers A26, A27;
  `plan-decisions.md` "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)", section "Step
  (stage 10, 11)"; tasks marked "Amended 2026-11-14 (A26)" or "(A27)").** In a viscous scene, stage 10 also calls,
  after `BuildBlockingMaterialFaces` (scene with a blocking set), `solids::BuildViscousMaterialFaces(
  BlockingMaterialFacesInputs, faces, geometry, solids_buffers)` giving `viscous_faces`, and, before
  `BuildPressureEquations`, `solids::ToFreeSolidsForViscousSolve(motion_map)` (one entry per free solid touching
  liquid), from which it fills the step-owned `free_solid_motion_before_pressure` (6 doubles per free solid, `V`
  then `Omega`, in `FreeSolidPush::solid_index` order, from `velocity_before_pressure` and
  `turn_rate_before_pressure`) given to `pressure::EquationInputs::free_solid_motion_before_pressure` (pressure
  plan, A27; `BuildPressureEquations` gives `kBadInput` unless its size is 0 or 6 x the free solid count; stage 12
  passes it empty). Stage 11 calls
  `viscosity::SolveViscosityAndPressure(ViscosityInputs{..., material_faces =
  solids::ToViscousMaterialFaces(viscous_faces), free_solids}, viscosity, cancel_requested, viscosity_buffers,
  fields.velocity_after_pressure, fields.pressure, viscosity::SolvedCouplingArrays{
  blocking_material_velocity_after_pressure, sealed_gas_pressure, material_face_velocity_after_solve,
  free_solid_motion})`; then `pressure::ComputeWasSolved`; then `solids::ApplySolvedMotionToSolids(motion_map,
  free_solid_motion, fields.pressure, solids_try, solids_buffers)` (in place of `ApplyPressureToSolids`: each free
  solid touching liquid takes its solved velocity and turn rate, travel and turn limits checked again,
  `kSolidMovesTooFar` -> `kSolidTravelledTooFar`); then (scene with a blocking set)
  `solids::ApplyViscousSolveToMaterials(viscous_faces, material_face_velocity_after_solve, {set S, set E},
  solids_buffers)` (in place of `ApplyPressureToMaterials`: each band face of each blocking set gets its solved
  velocity in its set's `GridFromParticles::velocity`; every other face of the set keeps its velocity after
  `ApplyForces` and `CollideSolids`; its `PressureOnMaterials::push` per set into the report).
  `blocking_material_velocity_after_pressure` is still filled by stage 11 (the `BlockingMaterialFace` subset) but
  no longer written back by it. Stage 12 is unchanged (inviscid: slides, material faces eliminated, free solids as
  the rank-one term). Step-owned rows: `"step: material face velocity after solve"` (given solids'
  `max_motion_map_entries` x 2, 8 B), `"step: free solid motion"` (given the free solid count, 48 B) and `"step:
  free solid motion before pressure"` (given the free solid count, 48 B; the pressure plan's input).
  `viscosity::ViscosityBufferSettings::max_material_faces` = 2 x solids' `max_motion_map_entries` (0 without a
  blocking set), `max_free_solids` = the scene's free solid count. No combination of materials is refused because
  of viscosity in the main sequence.
- **What counts as viscous (owner answer A25; tasks marked "Amended 2026-11-14 (A25)").** A liquid ingredient is
  viscous exactly when its `viscosity` setting is above 0 (`> 0.0`, no threshold); the scene is viscous when any
  liquid ingredient is, and then stage 11 runs, else stage 12. The choice is made once from the ready scene, in
  `CheckSceneCanRun` (Task 7) and `BuildSimulation` (Task 9), and never changes: retries, shortened steps, limit
  raises, cell resizes, batches and the viscous effect number do not change it. The water preset has viscosity 0.
  `BuildRunPlan` (Task 8) reports the viscous effect per step of each liquid and the path (`RunPlan::viscous_path`).
- **Calls into other modules use their final forms:** transfer's stages, each taking
  its `transfer::TransferBuffers` (their one form; built once in `BuildSimulation`, counted in the memory estimate entries); the step calls
  `solids::DriveWaves`, which calls `pressure::ApplyOpenSeaRelaxation` itself; `particles::ApplyParticleLimits(
  Particles&, std::span<const std::size_t> max_particle_counts, std::uint32_t max_cell_blocks)` raises the
  particle and block limits in one call; `ParticleFields` carries `run_on`; `particles::ComputeAmounts` returns
  `std::expected`. `RunOn` is in the shared `base/run_on.h`; `CheckCanRunOn` in `base/check_can_run_on.h`.
  Include each module's public headers by the file names its plan created (`plans/2026-11-14-rebuild-*.md`;
  for viscosity, materials, solids and gas, whose plans are written alongside this one, read the plan if it
  exists, else the header names of design section 3).
- **Builders and forms fixed by the solids, materials and gas plans** (`plan-decisions.md`):
  `solids::BuildSolidsBuffers(const SolidsSettings&, std::size_t max_particle_count, std::uint32_t
  max_cell_blocks)`; `solids::BuildSolidCoverageValues(RunOn, std::uint32_t max_cell_blocks)` (the step's
  coverage arrays); `materials::BuildMaterialBuffers(RunOn, const MaterialTable&, std::span<const std::size_t>
  max_particle_counts, std::uint32_t max_cell_blocks)`; `materials::ApplyInternalForce` and
  `ComputeCellSqueezeValues`, each one form taking the `MaterialBuffers` last and allocating nothing (A31); the gas ingredient's rest
  density given to materials is `gas::ComputeAmbientDensity(law)`; `gas::ToGasSnapshot` takes three arguments
  (not called by the step). `solids::ApplyPressureToSolids` (stage 12) and, A27, `solids::ApplySolvedMotionToSolids`
  (stage 11) can fail with `kSolidMovesTooFar`, mapped like `BuildTry`'s to `kSolidTravelledTooFar` (shorten and
  retry). `gas::FindCellsToKeepActive` marks cells of the
  existing grid only: growing the active region around them and around gas source boxes is the step's, in
  `ChooseActiveCells` (Task 29). Every work in solids is the energy a solid gives the material, positive when a
  paddle pushes liquid.
- **Builders take the run's block (A58).** The signatures of this list and of every task gain the `RunMemory&`
  argument(s) last, as the owning plans give them; no other argument changes.
- **No stubs.** Callbacks of `StepCallbacks` for cell resize, batches, I/O and render output stay null in
  production code here; tests pass hand-written functions with their own context objects (Task 5's
  `step_test_support.h`). No other module is stubbed.
- **Try discipline (design 5.6):** every stage writes only try copies or working buffers; no stage receives a writable
  pointer reachable from committed values. A rejected try leaves, to the byte, everything design 5.6 lists.
  The reported failure of a stage is the one with the highest `RejectKind` (`kCannotContinue`, then
  `kRaiseLimits`, then `kShortenStep`), then the lowest stable id (`particle_id`, else `CellId`, else
  `solid_id`), after every range, set and material of the stage ran.
- **Module errors to reasons:** design 5.6 "Module errors to reasons", minus the `kMaterialBoundaryNotSupported`
  row. One private function per module, `ToStageOutcome(<Module>Failure)` in `stage_outcome_from_modules.h`
  (Task 3), so the mapping lives in one place.
- **Sums:** every total over the grid is one partial per cell block in the grid's block order, then
  `math::TreeSum` over the block partials (design 3 preamble). Largest values use `ComputeLargestMagnitude`.
  Amended 2026-11-14 (Foundation as built): on buffers, `ComputeTreeSum`, `ComputeTreeDot` and
  `ComputeLargestMagnitude` are called with `TryArrays::sum_buffers` (their one form; no allocation per try).
- **Tolerances (design 6.4):** CPU against GPU after one committed step from the same snapshot: ids, counts,
  active cell ids, flags, `RejectReason`, `rejected_in`, `active_limit` equal; positions
  `1e-10 * smallest cell width + 1e-12 * largest coordinate`; velocities `1e-9 * (largest speed + |gravity| *
  step_duration)`; pressure and gas amounts the pressure and gas modules' own. Many steps: liquid volume 0.1 %,
  kinetic energy 2 %, front position one smallest cell width. Repeat runs and CPU runs: identical bytes.
- **Test labels** (amended 2026-11-14, Foundation as built: `fluidity_add_shared_test` is retired; the container's
  `cpu` step runs `cpu` tests): shared-formula, operation and scene tests `fluidity_add_test(... LABEL cpu LINKS
  fluidity_base_test_support fluidity_step_test_support)` as each needs; device tests `LABEL gpu`. Every operation
  with a `_gpu.cu` has a `no_gpu` test "on the GPU without a GPU gives `kRunOnCallFailed`". A known-open test
  (form amended 2026-11-14, Foundation as built: `fluidity_add_test` takes one label and discovers every test of
  its program under it, so a failing test there would fail the `cpu` gate) is written with GoogleTest's `DISABLED_`
  prefix, so discovery registers it as disabled and the gate skips it, and is registered a second time, inside
  `if(BUILD_TESTING)`, by `add_test(NAME <Suite>.<Name> COMMAND <program> --gtest_filter=<Suite>.DISABLED_<Name>
  --gtest_also_run_disabled_tests)` with `set_tests_properties(<Suite>.<Name> PROPERTIES WILL_FAIL TRUE LABELS
  known_open)` (`add_test` makes no C++ target, so the helper rule holds; the container's `cpu|no_gpu` run skips
  it), and a line in `docs/known-open-physics.md`.
- **Test scenes** come only from `step_test_scenes.h` (Task 5), which owns every vector a `ReadyScene` borrows.

### Buffers by processor, and copies counted (A17, 2026-11-14)

Owner answer A17: a run is all CPU or all GPU; in a batched GPU run only what does not fit (the particles, and in
`kBatchedWithSolvesInPieces` the solver buffers) stays in CPU memory, moved in planned, counted copies.
`plan-decisions.md` "Planned cost shown to the user, and buffers in batched runs" holds the declarations.

- **Appendix A** (end of this plan) lists, for every stage and each run mode, every buffer the stage touches, where
  it lives and where the stage's code runs, with the resolution of each mismatch (M1 to M15). A task that adds a
  stage call checks its row; a call that touches a buffer on the other processor without an Appendix A copy is a
  defect of that task.
- **Appendix B** gives the step's own planned copies with their byte formulas. Every one of them is counted in
  `StepReport::copies` (per try) and `ReadCopyCounts(simulation)` (cumulative), type `CopyCounts`; `RunPlan::
  planned_step_copies` is their upper bound per try at the limits. Copies of batches, of the solvers'
  `SolveBatches` and of cell resize are counted by those modules; the records of the "not counted" list of
  `plan-decisions.md` are the only uncounted transfers.
- **Placement rule** for whole try arrays in both batched modes (Task 33a): on the run's processor when a
  `kWholeGridOnProcessor` part reads or writes them; else in CPU memory when a `kWholeStateOnCpu` stage reads or
  writes them, the commit keeps them, or a later pass reads them; no whole instance when only one pass uses them.
- **A batched CPU run** builds the same mirrors and makes the same copies (CPU to CPU, counted the same), so the
  container tests the counts; only the grid copy and the cell-list upload are skipped (the grid is already on the
  CPU). A one-piece CPU run counts no copy.

## Before you start

On branch `rebuild` in `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild`, already committed: Foundation
with every `base/` piece of `shared-interface.md` section (a) and the `plan-decisions.md` additions (`Buffer`,
`CopyBuffer`, `ReadMemoryAvailable`, `MemoryEstimateEntry` with `where`, `TreeSum`, `ComputeLargestMagnitude`,
`BuildStableOrder`, `PerParticleVelocity`, `processor_description.h`); then contact, grid, particles, transfer,
pressure, viscosity, materials, solids and gas, each with its section 3 names, its test-support library and its
comment pass done. The method paper has the sections of all nine modules.

Prove it: `uv run python tools/check_commit.py` prints `PASS` on every line, and
`grep -ln "SolveGas\|SolveViscosityAndPressure\|CollideSolids\|ApplyInternalForce" native/fluidity/*/*.h`
lists one header in each of `gas/`, `viscosity/`, `solids/`, `materials/`. If either fails, stop and ask the owner.
**Amended 2026-11-15 (A58, A41, A57):** also `grep -l "BuildRunMemory" native/fluidity/base/run_memory.h`
(Foundation Tasks 17 to 19), `grep -l "ComputeNewCellSizeLevel" native/fluidity/grid/cell_size_rule.h` and `grep -l
"ComputeFourCellFeatureChange" native/fluidity/viscosity/viscosity_advice.h` each print the file.

**Amended 2026-11-14 (Foundation as built):** Foundation as built also provides `base/run_on_call.h`
(`RunOnCallStatus`, `ComputeLaunchSize`), `base/gpu_launch.cuh`, the sums (taking `SumBuffers`) and
the stable order and grouping (taking `OrderBuffers`), `base/test_support.h` (library `fluidity_base_test_support`) and
`fluidity_add_test_library` with `LINKS`; read `AGENTS.md` "Structure" (two halves, private GPU headers, working buffers)
before the first task. The expected gate output is seven `PASS` lines: format, rules, cpu debug, cpu release, tidy,
paper, cuda.

Then fix the viscosity exclusions (`plan-decisions.md`, "Viscosity"): the viscosity plan tried to lift three of
them, each falling back to known-open. Run

```bash
grep -n "inclined_solid_not_supported\|moving_solid_not_supported\|size_change_not_supported" native/fluidity/viscosity/*.cc
```

(the `ToName(ViscosityError)` table) and keep the result for Task 39's hand-over. Each `ViscosityError` value found keeps its `CapabilityReason`; each
value **not** found was lifted, and its reason does not exist in this module:

| `ViscosityError` (ToName) | `CapabilityReason` | Its suggested change row (Task 6) | Its test (Task 7) |
|---|---|---|---|
| `kInclinedSolidNotSupported` (`inclined_solid_not_supported`) | `kViscousLiquidWithInclinedSolid` | `kAlignSolidWithAxes` | `ViscousWithInclinedSolid_IsExcluded` |
| `kMovingSolidNotSupported` (`moving_solid_not_supported`) | `kViscousLiquidWithMovingSolidThroughCells` | `kMakeSolidFixed` (its own row since A27; was shared with `kViscousLiquidWithFreeSolid`) | `ViscousWithMovingSolid_IsExcluded` |
| `kSizeChangeNotSupported` (`size_change_not_supported`) | `kViscousLiquidAtSizeChange` | `kUseOneCellSizeThere` (shared row) | `ViscousAtSizeChange_IsExcluded` |

`kCapabilityReasonCount` = 16 minus the number of lifted rows: 16 if all three values are found, 13 if none is
(amended 2026-11-14, A24: `kViscousLiquidWithSnow` no longer exists; amended 2026-11-14, A27:
`kViscousLiquidWithFreeSolid` no longer exists; so every count is two lower than before; replaced: "17 minus ...
17 if all three values are found, 14 if none is").
For a lifted row, the enumerator, its `ToMessage` text, its row in Task 6's table (the reason only; a shared row
keeps its other reason) and its Task 7 test are left out, and Task 3's mapping has no line for it. The remaining
enumerators keep the design's order with no gap (the stable names are I/O's and none is stored yet).
`kAlignSolidWithAxes` and `kMakeSolidFixed` stay in `SuggestedChange` even if no reason uses them
(`kSuggestedChangeCount = 12`).
`kSetViscosityToZero` is removed (A27: no reason of the main sequence uses it; the I/O budget's note on a viscous
effect too small to see advises the same change in words; the sand plan adds it back with
`kViscousLiquidWithSand`). Replaced (A27): "(`kSuggestedChangeCount = 13`); so does `kSetViscosityToZero`, whose
only reason (`kViscousLiquidWithSnow`) A24 removed".

## Review Focus

1. **A rejected try that built a new grid version.** The free `GridBuffers` must come back with
   `grid::TakeBuffers`, the committed grid keep its version, and the next try build version + 1 again into the
   same storage with identical bytes. Pinned: `ChooseActiveCells, RejectedNewGrid_StorageReturnedAndVersionReused`
   in Task 13.
2. **The last step of `AdvanceTo` lands one rounding step short of the target.** `time()` must equal
   `target_time` exactly, not `target_time - ulp`. Pinned: `AdvanceTo, LastStepOneRoundingShort_EndsOnTarget` in
   Task 23.
3. **A source whose `start_time` falls inside a step called directly through `TryStep`** (no `kSourceWindow`
   limit applied by a loop): it is active only from a step that starts at or after `start_time`, and adds
   nothing in that step. Pinned: `AddNewParticles, SourceStartsInsideStep_AddsNothingThisStep` in Task 12.
4. **Two particles that travel too far in two different ranges.** The reported particle must be the lowest id
   whatever the cut of the grid into ranges. Pinned: `MoveParticles, TwoTooFarInTwoRanges_LowestIdReported` in
   Task 19.
5. **A run whose limits are raised several times** must give the bytes of a run that started with large limits
   (`math::TreeSum` pads by count, grid storages rebuilt with the same inputs, block tables of particles raised
   with the grid's). Pinned: `ApplyLimits, RaisedRun_SameBytesAsLargeLimits` in Task 24.

---

### Task 1: Method paper, one step for every scene

**Amended 2026-11-14 (A28):** the pseudocode's first checks are the step-input checks (`StepInputs`, was `StepRequest`).

**Kind:** mechanical

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills first.

**Files:**
- Create: `docs/method-paper/sections/step.typ` (included after the gas section; if earlier modules put their
  sections inline, follow them), `docs/method-paper/figures/step_stages.typ`
- Modify: `docs/method-paper/fluidity-method.typ` (include, notation rows), `docs/glossary.md`

**Interfaces:**
- Produces labels `<sec-step>`, `<tab-step-stages>`, `<fig-step-stages>`, `<proc-try-step>`.

- [ ] **Step 1: Text,** design section 8 outline items 1 to 4: one step for every scene and why the grid is
  fixed inside it; material sets (all liquids one set, gas alone, snow and springy solids each their own) and
  the stage table `<tab-step-stages>` (the 21 rows of design 5.2's table: stage, writes, border, skipped when);
  the two time marks (before forces, after pressure) and why wave relaxation follows the first (cite ref 20,
  Jacobsen 2012); the liquid solve, then the gas solve, and what each reads of the other (sealed gas blobs solved inside
  the liquid solve, stage 11 or 12 alike (A18, A24); open gas read one step late; the gas reads the liquid's velocity after this step's solve). Cite ref 4 (Fei 2021) for the velocity for
  this step and the half-cell limit, ref 9 (Larionov 2017) for stage 11. Words: `try`, `commit`, `reject`,
  `material set`, `stage`, `step_duration`, `state_version` (glossary rows for any word not there yet:
  `owed_particles`, `position_spread`, `allowed_bytes`, `RunPlan`, `KeepGoing`, `RejectKind`).
- [ ] **Step 2: Figure** `<fig-step-stages>` with `figure_style.typ`: the 21 stages as a column grouped particles
  / grid / particles, one lane per material set (L, S, E, G of design 5.2), arrows for the values that cross
  stages, the two snapshots marked. Caption states the one thing to see.
- [ ] **Step 3: Pseudocode** `<proc-try-step>` with `procedure(...)`: the step-input checks, then stages 1 to 20
  numbered exactly as design 5.2, inputs, outputs, "fails when" (the `RejectKind`s).
- [ ] **Step 4: Run** `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`; look at
  every new page; fix overlaps. Run the gate. **Commit.** Subject: `Describe one step for every scene`.

### Task 2: Method paper, try and commit, the step duration, the check and sizing

**Amended 2026-11-14 (A31):** the figure shows shared working buffers (the word "scratch" is not used).

**Kind:** mechanical

**Files:**
- Modify: `docs/method-paper/sections/step.typ`, `docs/glossary.md`
- Create: `docs/method-paper/figures/try_and_commit.typ`, `figures/step_tries.typ`

**Interfaces:**
- Produces labels `<sec-try-and-commit>`, `<sec-step-duration>`, `<sec-capability>`, `<sec-sizing>`,
  `<fig-try-and-commit>`, `<fig-step-tries>`, `<proc-advance-to>`, `<proc-compute-step-duration>`,
  `<proc-apply-limits>`.

- [ ] **Step 1: Text,** outline items 5 to 9: what a try owns, what is swapped, what a reject leaves alone (the
  list of design 5.6); three ways a try ends without a commit (shorter step, larger limits, cancel) and why they
  differ, plus "cannot continue"; the step duration, its eleven `StepLimit`s with the table of design 5.8 and why
  half a cell; the capability check as a predicate over the scene; sizing (counts, rows, fit, one piece or
  batches; design 5.9). Cite refs 5 and 15 (state kept on the GPU between stages). The random words cite Salmon,
  Moraes, Dror and Shaw 2011 by author and year only (design decision 9).
- [ ] **Step 2: Figures** `<fig-try-and-commit>` (committed and try copies of each owner side by side, shared
  working buffers below, the swap at commit and the drop at reject) and `<fig-step-tries>` (a time line: one committed
  step, one reject then two half steps, one limit raise then the same step again).
- [ ] **Step 3: Pseudocode** `<proc-advance-to>` (choose, try, branch on `RejectKind`; numbered 1 to 7 as Task 23
  implements), `<proc-compute-step-duration>` (1 limits, 2 smallest, 3 `n = ceil(...)`, 4 equal steps, 5 which
  limit), `<proc-apply-limits>` (the five steps of design 5.7).
- [ ] **Step 4: Run** typst, look at the pages, run the gate. **Commit.** Subject:
  `Describe try and commit, the step duration and the sizing plan`.

### Task 3: Errors, reasons and their names

**Amended 2026-11-14 (A24):** the mapping example uses the free-solid exclusion; the snow exclusion is lifted.

**Amended 2026-11-14 (A27):** the free-solid exclusion is lifted too (no `kFreeSolidNotSupported`, no
`kViscousLiquidWithFreeSolid`): the mapping example's viscosity row is now `kMovingSolidNotSupported` if that value
still exists, else `kVelocitySolveDidNotConverge`. Replaced: "the mapping example uses the free-solid exclusion".

**Amended 2026-11-14 (Foundation as built):** tests registered with `fluidity_add_test(... LABEL cpu)` (no
`fluidity_add_shared_test`); the GPU halves' status reaches `ToStageOutcome` only through `ToBufferError`.

**Amended 2026-11-15 (reconcile item 31, A58):** `StepError::kOutOfMemory` is the last value (later values shift by
one, accepted); a private `ToBuildFailure` maps a builder's `BufferError` (with `kRunMemoryTooSmall`) to it.

**Kind:** mechanical

**Files:**
- Create: `native/fluidity/step/step_error.h` (shared), `step_message.h`, `step_message.cc`,
  `stage_outcome_from_modules.h` (private, CPU only), `stage_outcome_from_modules.cc`, `step_error_test.cc`,
  `stage_outcome_from_modules_test.cc`
- Modify: the CMake lines that list module folders (register `native/fluidity/step/` as
  `native/fluidity/gas/` is), `docs/glossary.md`

**Interfaces:**
- Produces design 3.1 exactly: `Stage`, `kStageCount`, `RejectReason`, `RejectKind`, `ToRejectKind`, `Module`,
  `kModuleCount`, the opaque `enum class CapabilityReason : std::uint8_t;`, `RejectContext`, `StepError`,
  `StepFailure`, `kRejectReasonCount`, `ToMessage(StepError)`, `ToMessage(RejectReason)`, `ToMessage(Stage)`,
  `ToName(StepError)`. `StageOutcome` (design 3.11) is declared here too (plan decision: the mapping below
  returns it, and `step_stages.h` includes this header).
- Produces (private): `StageOutcome ToStageOutcome(const <Module>Failure& failure)` for contact, grid,
  particles, transfer, pressure, viscosity, materials, solids, gas, and `ToStageOutcome(BufferError)`.
- Produces (private, A58): `StepFailure ToBuildFailure(BufferError error, Module module, std::uint64_t bytes_asked)`:
  `kOutOfMemory` and `kRunMemoryTooSmall` give `StepError::kOutOfMemory` with `context.module = module`,
  `module_error` the `BufferError` value and `magnitude = bytes_asked`; `kRunOnCallFailed` gives `kRunOnCallFailed`;
  any other value `kBadInput`. Used only by `BuildSimulation`, `BuildSimulationFromSnapshot` and `ApplyLimits`
  (reconcile item 31: no stage returns `kOutOfMemory`).

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ToRejectKind, EveryReason_HasItsGroup)  // kNone->kNone; kParticleTravelledTooFar..kValueNotFinite (12)
      // ->kShortenStep; kTooManyParticles, kTooManyCellBlocks, kBuffersTooSmall->kRaiseLimits; kCancelRequested
      // ->kCancelled; kBadStepDuration, kWrongStateVersion->kBadStepInputs; the last four->kCannotContinue;
      // static_assert on all 23
  TEST(ToMessage, EveryRejectReason_DistinctAndNotEmpty)
  TEST(ToMessage, EveryStage_DistinctAndNotEmpty)
  TEST(ToName, EveryStepError_IsLowerSnakeCaseWithoutK)  // kBatchedRunNotBuilt -> "batched_run_not_built"
  TEST(Module, NumbersNeverChange)                       // static_assert kStep == 0, kBase == 10, kRenderOutput == 13
  TEST(ToStageOutcome, EveryModuleErrorRow_GivesTheDesignReason)  // one row per line of design 5.6 "Module errors
      // to reasons": e.g. pressure kStepTooLongForAirPocket with limit 0.004 -> kStepTooLongForAirPocket,
      // context.limit 0.004, context.module kPressure, module_error its value; A27: viscosity
      // kMovingSolidNotSupported -> kNotSupportedNow with capability kViscousLiquidWithMovingSolidThroughCells
      // (only if that value still exists, "Before you start"; else viscosity kVelocitySolveDidNotConverge ->
      // kViscosityNotConverged); (replaced, A27: viscosity kFreeSolidNotSupported -> kNotSupportedNow with
      // capability kViscousLiquidWithFreeSolid); solids kSolidsTooClose -> kNotSupportedNow,
      // kFromSolids; pressure kCellSplitBySolid -> kThinSolidInsideOneCell; gas kPressureSolveFailed holding
      // kNotConverged -> kPressureNotConverged; any kRunOnCallFailed -> kRunOnCallFailed; any kCancelled ->
      // kCancelRequested; an unlisted error (pressure kEquationsNotPositive) -> kStageFailed
  TEST(ToStageOutcome, ContextCarriesIds)  // particle id, CellId, solid_id and magnitude copied from the failure
  TEST(StepError, OutOfMemoryIsTheLastValue)  // A58, reconcile 31: static_assert kOutOfMemory == kRunOnCallFailed + 1
      // and no value above it; ToName(kOutOfMemory) == "out_of_memory"
  TEST(ToBuildFailure, RunMemoryTooSmall_IsOutOfMemoryWithBytes)  // A58: (kRunMemoryTooSmall, Module::kGrid, 4096)
      // -> error kOutOfMemory, context.module kGrid, magnitude 4096; (kOutOfMemory, kPressure, 1e6) the same shape;
      // (kRunOnCallFailed, ...) -> kRunOnCallFailed
  ```
  The first five are `fluidity_add_test(step_error_test SOURCES step_error_test.cc LABEL cpu)`; the mapping test
  is `fluidity_add_test(stage_outcome_from_modules_test ... LABEL cpu)`. Every GPU half of the step reaches
  `StageOutcome` as `ToStageOutcome(ToBufferError(status))`: no other mapping of `RunOnCallStatus` exists.
- [ ] **Step 2: Run** (compile error). **Step 3: Implement.** `CapabilityReason` is completed in Task 6; its
  enumerator values are fixed here in a comment so `RejectContext` stays one shape. Messages are one English
  sentence that says what the user can do.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Name why a step try did not commit`. New names: the 3.1 list.

### Task 4: The step's shared formulas

**Amended 2026-11-14 (Foundation as built):** `step_math_test` is `LABEL cpu` (no `fluidity_add_shared_test`); the
four shared headers and the test are listed in `native/fluidity/step/CMakeLists.txt`.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/step/step_duration_math.h`, `apply_forces_math.h`, `travel_math.h`, `random_words.h`
  (all shared, in `fluidity_add_headers`), `step_math_test.cc`
- Modify: `native/fluidity/step/CMakeLists.txt`

**Interfaces:**
- Produces: `ComputeTravelStepLimit(double speed, double acceleration, double travel)` (design 3.8);
  `ComputeVelocityAfterGravity(double velocity, double gravity_along_axis, double step_duration)`;
  `ComputeTravelInCellWidths(Vec3 start, Vec3 end, double cell_width)` (largest axis part of `end - start` over
  the width); `FindDomainSideCrossed(const contact::DomainBox&, Vec3 end)` returning
  `contact::DomainSide` and a found flag (plan decision: `struct DomainSideCrossed { contact::DomainSide side;
  std::uint8_t crossed; }`, the first side in `DomainSide` order whose plane `end` is beyond).
  **Plan decision** for the generator (Philox 4x32 with 10 rounds, ported from
  `git show main:native/src/liquid_substep_3d_staging.cc` lines 13-27):

  ```cpp
  struct RandomWords { std::uint32_t word[4]; };
  // counter = {low and high half of child_number, low and high half of source_id}; key = the two halves of key.
  FLUIDITY_CPU_AND_GPU RandomWords ComputeRandomWords(std::uint64_t child_number, std::uint64_t source_id,
                                                      std::uint64_t key);
  // (word + 0.5) * 2^-32 mapped to [-1, 1): 2 u - 1, as main's staging.cc lines 664-668.
  FLUIDITY_CPU_AND_GPU double ComputeSignedUnitValue(std::uint32_t word);
  ```

- [ ] **Step 1: Failing tests** (`fluidity_add_test(step_math_test SOURCES step_math_test.cc LABEL cpu LINKS
  fluidity_base_test_support)`; bit comparisons with `ToBits`):

  ```cpp
  TEST(ComputeRandomWords, PublishedZeroVector)          // (0, 0, 0) == {0x6627e8d5, 0xe169c58d, 0xbc57ac4c,
                                                         //  0x9b00dbd8} (main's test_liquid_substep_3d.cc:5262)
  TEST(ComputeRandomWords, CounterHalvesInMainsOrder)    // child 0x100000002, source 3: same words as main's
      // function on counter {2, 1, 3, 0}, key {0, 0} (copy the four expected words from a throwaway run of main)
  TEST(ComputeSignedUnitValue, Extremes_StayInsideOpenInterval)  // word 0 -> -1 + 2^-32; max -> 1 - 2^-32
  TEST(ComputeTravelStepLimit, NoAcceleration_IsTravelOverSpeed) // (2, 0, 0.5) == 0.25 exactly
  TEST(ComputeTravelStepLimit, AtRest_IsSqrtOfTravelOverAcceleration)  // (0, 9.81, 0.05): sqrt(0.05/9.81), 1e-15
  TEST(ComputeTravelStepLimit, General_SolvesTheQuadratic)  // t = 2d / (v + sqrt(v^2 + 4 a d)); v t + a t^2 == d,
                                                            // 1e-15 relative, for v 1.5, a 9.81, d 0.0125
  TEST(ComputeTravelStepLimit, NothingMoves_IsInfinity)
  TEST(ComputeVelocityAfterGravity, AddsTheProductOnce)  // (0.25, -9.81, 0.01) == 0.25 + (0.01 * -9.81) bit for bit
  TEST(ComputeTravelInCellWidths, LargestAxisPart)       // start 0, end {0.1, -0.3, 0.2}, width 0.5 -> 0.6
  TEST(FindDomainSideCrossed, BeyondTopAndRight_ReportsFirstInOrder)  // kRight before kTop
  TEST(FindDomainSideCrossed, Inside_IsNotCrossed)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** The travel limit uses the stable form of design 5.8 and
  `math::Sqrt`; the gravity product is formed once (`step_duration * gravity_along_axis`) and added, so gas can
  take it back exactly. Amended 2026-11-14 (Foundation as built): the docstring of `ComputeVelocityAfterGravity`
  promises the same bits on the CPU and the GPU unless the result is NaN (one product and one sum, each rounded
  once by IEEE 754, no contraction under `-ffp-contract=off` and `--fmad=false`); Task 16's GPU test relies on it.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add the step's travel, gravity and random-word formulas`.

### Task 5: The ready scene, the settings, and the test scenes

**Amended 2026-11-14 (Foundation as built):** the test-only library is created with
`fluidity_add_test_library(fluidity_step_test_support SOURCES step_test_scenes.cc step_test_support.cc LINKS
fluidity_base_test_support <earlier modules' test-helper libraries it uses>)` and every step test names it in
`LINKS`; it reuses `base/test_support.h` instead of its own bit or buffer helpers.

**Amended 2026-11-15 (A58, A47, A41, A64):** `StepSettings` has `memory_share_of_free` (0.9) and
`memory_left_free_bytes` (536,870,912) with the two `kDefault...` constants, no `memory_share_of_total`;
`ReadyScene::gas_wind` is `wind`, `ReadyScene` gains `grid::CellSizeSettings cell_size` and the new comment of
`new_cell_size_level` (design 3.2, 3.3). Test `StepSettings, DefaultsAreTheBudgetConstants` (share 0.9, left free
536,870,912, exact).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/ready_scene.h`, `step_settings.h`; test-only library `fluidity_step_test_support`
  (`fluidity_add_test_library`; the name `fluidity_step_test_support` was renamed by `plan-decisions.md` "Names that
  broke the glossary"): `step_test_scenes.h`, `step_test_scenes.cc`, `step_test_support.h`, `step_test_support.cc`,
  `step_test_scenes_test.cc`
- Modify: `native/fluidity/step/CMakeLists.txt` (the library, the test, the two public headers)

**Interfaces:**
- Produces design 3.2 and 3.3 exactly (`ParticleSource`, `StartParticles`, `ReadyScene`,
  `kMaxTravelInCellWidths`, `kCoveredReachInCellWidths`, `SimulationLimits`, `StepSettings`).
- Produces (test-only, plan decision):

  ```cpp
  // step_test_scenes.h. TestScene owns every vector, span target and key array; scene() borrows from it.
  // Move-only (the spans point into its own storage). Every builder: Z-up, gravity {0, 0, -9.81}, seed 20261114.
  class TestScene { public: const ReadyScene& scene() const; ReadyScene& scene_to_change(); };
  TestScene BuildEmptyScene();             // closed box [0, 0.4]^3, biggest cell 0.1 m, 2 levels, one water
                                           //   material, its set, no particle, no source
  TestScene BuildFallingBlockScene(double own_motion_kept);  // open top; water block [0.1, 0.3]^2 x [0.2, 0.3],
                                           //   8 particles per 0.05 m cell, at rest; no solid
  TestScene BuildColumnAtRestScene();      // closed box 0.4 x 0.4 x 0.6, water to 0.43, cells 0.05
  TestScene BuildTwoLiquidsScene();        // closed box, oil (800 kg/m3) over water (1000) in one liquid set,
                                           //   a MaterialPair between them
  // Later tasks add BuildSourceScene (Task 12), BuildSyrupColumnScene (25), BuildFreeBoxScene,
  // BuildDrivenBoxScene (26), BuildWaveScene (27), BuildSnowAndJellyScene (28), BuildSmallFireScene (29),
  // BuildGasBesideLiquidScene (30), BuildFiveMaterialScene (31: the scene of design 5.2), BuildSmallSyrupScene,
  // BuildSmallSnowExample (36), and the scenes of Tasks 34 and 35.
  // step_test_support.h
  struct CallbackLog { std::vector<std::string> calls; void* context_seen = nullptr; };  // Hand-written callbacks
                                                                                       //   append "name".
  ```
  Two more helpers join `step_test_support.h` in the tasks that first need them (each declared and defined in
  its task, never before): `BuildRejectingStageTable` (Task 10) and `ExpectSameBytes` (Task 11). No step helper
  repeats one of `base/test_support.h` (`ToBits`, `BuildBufferHolding`, `ReadAll`, `CheckWithinTolerance`,
  `CheckSameBitsOrBothNaN`, `WriteResultDigest`); step helpers call them.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `LINKS fluidity_step_test_support`): `TestScene, MovedScene_SpansStillPointIntoItsStorage`;
  `BuildFallingBlockScene, ParticleCountAndMass` (4 x 4 x 2 cells x 8 = 256 particles, mass `1000 * 0.05^3 / 8`
  each); `BuildTwoLiquidsScene, OneLiquidSetWithBothMaterials` (mask `{1, 1}`); `StepSettings,
  DefaultsAreTheBudgetConstants` (A58).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the types and the four builders. Particle placement: 8 per cell
  at the cell's eighth centres, ids in x-fastest order.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Declare the ready scene and the step settings`. New names:
  `ParticleSource`, `StartParticles`, `ReadyScene`, `SimulationLimits`, `StepSettings`, the four constants
  (`kDefaultMemoryShareOfFree`, `kDefaultMemoryLeftFreeBytes`: A58).

### Task 6: `CheckSceneCanRun`, materials, sets, grid and duration

**Amended 2026-11-14 (reconcile):** the table's level column uses `CapabilityLevel` (`kUnavailable`,
`kExperimental`), not the design's `Support` words.

**Amended 2026-11-14 (A24):** no `kViscousLiquidWithSnow` (its table row is removed; the counts are 17 and 14).

**Amended 2026-11-14 (A27):** no `kViscousLiquidWithFreeSolid` (removed from the first table row) and no
`SuggestedChange::kSetViscosityToZero`; the counts are 16 and 13 (replaced: 17 and 14), `kSuggestedChangeCount = 12`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/capability.h`, `capability.cc`, `capability_test.cc`
- Modify: `native/fluidity/step/step_error.h` (`CapabilityReason` completed), `step_message.cc`

**Interfaces:**
- Produces design 3.4 exactly, without `kLiquidsThatDoNotMixTouch`, without `kViscousLiquidWithSnow` (A24),
  without `kViscousLiquidWithFreeSolid` (A27) and without each viscosity reason whose error was lifted ("Before you
  start"): 16 values if no exclusion was lifted, 13 if all three were (replaced: 17 and 14);
  `static_assert(kCapabilityReasonCount == <that number>)`; `SuggestedChange` without `kSetViscosityToZero` (A27),
  `static_assert(kSuggestedChangeCount == 12)`.
  **Plan decision** (design 3.4 gives no table): `CapabilityLevel` and `SuggestedChange` per reason (reconcile: the glossary's
  `CapabilityLevel { kCanRun, kExperimental, kUnavailable }` replaces the design's `Support`):

  | Reason | `CapabilityLevel` | Suggested change |
  |---|---|---|
  | `kViscousLiquidWithMovingSolidThroughCells` (only if not lifted; A27: `kViscousLiquidWithFreeSolid` removed from this row) | `kUnavailable` | `kMakeSolidFixed` |
  | `kViscousLiquidWithInclinedSolid` (only if not lifted) | `kUnavailable` | `kAlignSolidWithAxes` |
  | `kViscousLiquidAtSizeChange` / `kInternalForceAcrossSizeChange` | `kUnavailable` / `kExperimental` | `kUseOneCellSizeThere` |
  | `kThinSolidInsideOneCell` | `kUnavailable` | `kUseSmallerCells` |
  | `kMoreThanOnePlateNearAFace` | `kUnavailable` | `kSeparateTheObjects` |
  | `kSolidThinnerThanTolerance` | `kUnavailable` | `kMakeSolidThicker` |
  | `kDomainTooLargeForSmallestCell` | `kUnavailable` | `kMakeDomainSmaller` |
  | `kTooManyIngredients` | `kUnavailable` | `kUseFewerIngredients` |
  | `kLiquidSurfaceTensionsDiffer` | `kUnavailable` | `kUseOneSurfaceTension` |
  | `kMaterialSetsNotValid` | `kUnavailable` | `kPutAllLiquidsInOneSet` |
  | `kWavePaddleReceivingVolume` | `kExperimental` | `kNone` |
  | `kStepDurationFloor` | `kUnavailable` | `kNone` |
  | `kFromSolids` | `kUnavailable` | `kUseFewerSolids` for solids' plate and free-solid count errors, else `kNone` |
  | `kFromGas` | `kUnavailable` | `kNone` |

- [ ] **Step 1: Failing tests** (`LABEL cpu`), each on `BuildFallingBlockScene` changed by one field:

  ```cpp
  TEST(CheckSceneCanRun, ValidScene_CanRunWithNoFinding)
  TEST(CheckSceneCanRun, TwoSurfaceTensions_IsLiquidSurfaceTensionsDiffer)  // module kMaterials, module_code
                                                                            // "surface_tensions_differ"
  TEST(CheckSceneCanRun, LiquidsInTwoSets_IsMaterialSetsNotValid)           // BuildTwoLiquidsScene split in two
  TEST(CheckSceneCanRun, SetsNotInLowestIndexOrder_IsMaterialSetsNotValid)
  TEST(CheckSceneCanRun, GasSharingASet_IsMaterialSetsNotValid)
  TEST(CheckSceneCanRun, MaskWithWrongLength_IsMaterialSetsNotValid)        // from transfer::CheckMaterialSets
  TEST(CheckSceneCanRun, MoreThanSixteenLevels_IsDomainTooLargeForSmallestCell)
  TEST(CheckSceneCanRun, TooManyIngredients_IsTooManyIngredients)
  TEST(CheckSceneCanRun, MaxStepBelowRounding_IsStepDurationFloor)          // start_time 1e9, max_step 1e-9
  TEST(CheckSceneCanRun, BadMaterialNumber_IsBadInputFinding)               // kStep finding carries kMaterials code
  TEST(CheckSceneCanRun, OnGpuWithoutAGpu_StillAnswers)                     // StepSettings::run_on kGpu
  TEST(CheckSceneCanRun, TwoLiquidsThatDoNotMix_CanRun)                     // BuildTwoLiquidsScene: no finding
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the checks of design 5.14 that need only materials,
  transfer, grid and particles: `materials::BuildMaterialTable`; `transfer::CheckMaterialSets` and the set
  rules of design 3.2; grid frame and levels (`grid_frame.origin == domain_box.low_corner`, else a `kBadInput`
  finding); ingredient count; the duration floor. Findings sorted as 3.4 says; `findings` allocated once.
  Viscosity, solids, contact and gas checks come in Task 7.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Check that a scene can run before allocating`.

### Task 7: `CheckSceneCanRun`, viscosity, solids, contact and gas

**Amended 2026-11-14 (reconcile):** level names in two test comments use `CapabilityLevel`. (Its OPEN note on a
springy solid beside a viscous liquid is removed by A24.)

**Amended 2026-11-14 (A24):** snow beside a viscous liquid is no longer excluded: `ViscousWithSnow_IsExcluded`
becomes `ViscousWithSnowAndSpringySolid_CanRun`; option (A) of reconcile item 4 is withdrawn (no
`kViscousLiquidWithSpringySolid`).

**Amended 2026-11-14 (A25):** the check fills `ViscositySceneSummary::has_viscous_ingredient` by the A25 rule and
fixes the scene's viscous path.

**Amended 2026-11-14 (A27):** a free solid beside a viscous liquid is no longer excluded (its motion is an unknown
of the viscous solve): `ViscousWithFreeSolid_IsExcluded` becomes `ViscousWithFreeSolid_CanRun`; the summary has no
`has_free_solid_touching_liquid`.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/capability.cc`, `capability_test.cc`, `step_test_scenes.h/.cc` (a scene with a
  plate, a free box and a driven box: `BuildSolidsCheckScene()`)

- [ ] **Step 1: Failing tests** (`LABEL cpu`), the rest of design 7 "Capability" (one scene per reason):

  ```cpp
  TEST(CheckSceneCanRun, ViscousWithFreeSolid_CanRun)           // A27: a viscous liquid with a free box touching
      // it: no finding (the box's motion is an unknown of stage 11). Replaced: ViscousWithFreeSolid_IsExcluded
  TEST(CheckSceneCanRun, ViscousWithSnowAndSpringySolid_CanRun) // A24: a viscous liquid beside a snow set and a
      // springy-solid set: no finding (their faces are carried by the joint solve of stage 11)
  TEST(CheckSceneCanRun, ViscousWithMovingSolid_IsExcluded)     // only if kMovingSolidNotSupported still exists
  TEST(CheckSceneCanRun, ViscousWithInclinedSolid_IsExcluded)   // only if kInclinedSolidNotSupported still exists
  TEST(CheckSceneCanRun, ViscousAtSizeChange_IsExcluded)        // only if kSizeChangeNotSupported still exists
  TEST(CheckSceneCanRun, LiftedViscousCase_CanRun)              // one per lifted exclusion (none if none lifted):
      // the scene of the missing test above gives no finding
  TEST(CheckSceneCanRun, PlateThinnerThanCell_IsThinSolidInsideOneCell)
  TEST(CheckSceneCanRun, TwoPlatesNearAFace_IsMoreThanOnePlateNearAFace)
  TEST(CheckSceneCanRun, SolidThinnerThanTolerance_IsExcluded)  // contact::CheckSolid, kSolidTooThin
  TEST(CheckSceneCanRun, SnowAcrossSizeChange_IsKnownLimit)     // level kExperimental; can_run stays true
  TEST(CheckSceneCanRun, WavePaddle_IsKnownLimit)
  TEST(CheckSceneCanRun, SolidsRule_IsFromSolidsWithSolidId)    // a free solid of mass 0
  TEST(CheckSceneCanRun, TwoGasMaterials_IsFromGas)             // gas kMoreThanOneGas
  TEST(CheckSceneCanRun, UnorderedGasSourceKeys_IsFromGas)      // gas::CheckGasSources
  TEST(CheckSceneCanRun, LiquidAndGas_CanRun)                   // A10: no finding
  TEST(CheckSceneCanRun, ThreeBrokenRules_AllFindingsInOrder)   // design 7 "All findings": two solids rules on
      // two solids and a material-set rule: three findings, kUnavailable first, then by reason, solid_id
  ```
- **The viscous path (A25).** The check fills `ViscositySceneSummary::has_viscous_ingredient` = some liquid
  ingredient has `materials::MaterialTable::ingredient_viscosity()` above 0 (`> 0.0`, no threshold). This is the
  one rule that chooses stage 11 (viscous) or stage 12 (every liquid has viscosity 0) for the run; `BuildRunPlan`
  (Task 8) and `BuildSimulation` (Task 9) apply the same rule to the same ready scene, and nothing later changes
  it. Its tests are in Tasks 8 and 25 (they need the plan and the stage table).
- **Free solid beside a viscous liquid can run (A27).** Replaced: "stays excluded (`kViscousLiquidWithFreeSolid`,
  A24): the free solid's velocity would be an unknown of the viscous velocity solve (viscous drag through
  `rate_from_solids`, its impulse), which A24 does not cover; A24 covers material faces and sealed gas only." Since
  A27 the free solid's linear and angular velocity are unknowns of stage 11's solve, so no combination of
  materials is refused because of viscosity.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the rest of design 5.14: the viscosity summary from
  `solids::ComputeSolidsSummary` and the materials (with `has_viscous_ingredient` by the A25 rule; no `has_snow`,
  A24; no `has_free_solid_touching_liquid`, A27), then `viscosity::CheckViscosityCanRun`; one `kFromSolids`
  finding per entry of `solids::CheckSolidsCanRun(solids, smallest_cell_width, largest_cell_width)`;
  `contact::CheckSolid` and `CheckContactTolerance` per solid; with gas, `gas::CheckGasCanRun` and
  `gas::CheckGasSources`; the two known limits. `kThinSolidInsideOneCell` and `kMoreThanOnePlateNearAFace`
  are found here from the plate thickness against the smallest cell and the plate count near a face as
  solids' public limits state them (they also arrive per try through Task 3's mapping).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Report every reason a scene cannot run, with its fix`.

### Task 8: Memory estimate entries and the run plan

**Amended 2026-11-14 (A15):** a third run mode, the solve piece size and the CPU allowance in the plan.

**Amended 2026-11-14 (A17, batched-run buffers):** the plan says where its bytes live (`on_cpu`), returns a plan
that does not fit instead of failing, and carries the step's planned copies, so I/O can show the budget.

**Amended 2026-11-14 (reconcile):** the grid's room comes from this planner only (never from a default), the two
grid buffers share one builder buffers in the estimate, and a planned room above `grid::kMaxCellBlocksLimit` is a
plan that does not fit; rows B5 is withdrawn from the batched test.

**Amended 2026-11-14 (A25):** the plan carries the viscous effect per step of each liquid and the run's viscous
path, for the I/O budget lines.

**Amended 2026-11-14 (A26, A27):** in a viscous scene the estimate takes `BuildViscosityMemoryEstimate` with
`max_material_faces` = 2 x solids' `max_motion_map_entries` (0 without a blocking set) and `max_free_solids` = the
scene's free solid count (`SceneCounts::free_solid_count`), giving viscosity's entries `"viscosity: material face
vectors"`, `"viscosity: free solid vectors and blocks"` and `"viscosity: sample solid coefficients"`; the step's
own rows `"step: material face velocity after solve"`, `"step: free solid motion"` and `"step: free solid motion
before pressure"` are added to
`step_memory_estimate.cc` by Tasks 28 and 26 (each with the code that writes it).

**Amended 2026-11-14 (Foundation as built):** the step's own `SumBuffers` and `OrderBuffers` get four entries; each
module's estimate already carries its own working-buffer entries, so the step adds none for them.

**Amended 2026-11-14 (A31, reconcile):** "scratch" wording and test names become "working buffers"
(`StepWorkingBufferEntries_FromBuffersSizes`); the CPU grid-copy pair is counted with
`grid::BuildGridMemoryEstimate(RunOn::kCpu, 2, limits.max_cell_blocks, kOneForAllGrids)` (reconcile 19); pressure and gas estimates take
`run_on` (reconcile 24); the loaded try's grid buffers borrow the same builder buffers.

**Amended 2026-11-15 (A58):** the allowed bytes come from `ComputeAllowedMemoryBytes` (free memory only);
`on_processor` is `ComputeRunMemoryBytes(entries, counts, MemoryWhere::kRunOn)` and the new
`MemoryEstimate::cpu_run_memory_bytes` is the same over the `kCpu` entries kept `kWholeRun` or `kOneStepTry`; no
`kRunOn` entry is `kOneCall` (the needed block id row becomes `kWholeRun`).

**Amended 2026-11-15 (A64):** `SceneCounts` counts every liquid cell at the depth term's levels
(`particles_at_depth_levels`, `start_particles_in_deep_cells`); the particle limit uses the larger of that and the
start count; the cell resize rows are the three cell-size arrays.

**Amended 2026-11-15 (A41):** row O4 and the row `"step: new cells for the size choice"` are withdrawn.

**Amended 2026-11-15 (A57):** `LiquidViscousEffect` carries the advice numbers; `RunPlan` gains
`viscous_effect_run_duration` and `reynolds_length`.

**Amended 2026-11-15 (A40):** `BuildRunPlan` takes `std::optional<RunMode> forced_mode` last.

**Kind:** numerics

**Files:**
- Create: `native/fluidity/step/run_plan.h`, `run_plan.cc`, `step_memory_estimate.cc`, `run_plan_test.cc`

**Interfaces:**
- Produces design 3.5 exactly: `SceneCounts`, `ModuleBytes`, `MemoryEstimate`, `RunMode`, `RunPlan`,
  `BuildRunPlan`, `MemoryReport`, `BuildStepMemoryEstimate`.
- **Plan decisions:** the step's rows (design 6.3) use `sizeof` of the stored types with a `static_assert` each
  (`transfer::GridAtParticle`, `contact::ParticleContact` (56), `contact::SolidReaction`,
  `materials::InternalForceTotals`, `SourceCounter`, `pressure::AirPocket`, the `DriveWaves` block partial);
  rows set `where` (`kCpu` for source counters and open-sea site bits on a GPU run). If transfer's
  `BuildTransferMemoryEstimate` has no row for `TransferBuffers` (read transfer plan Task 4), the step adds
  `"step: transfer buffers"` per particle with the bytes `TransferBuffers` documents, and the hand-over reports
  it. The second `PressureBuffers` (gas) is `BuildPressureMemoryEstimate(gas_sizes, run_on)` with the gas sizes (no
  air pocket, free solid or boundary); the pressure and gas estimates take the run's `RunOn` (amended 2026-11-14,
  Foundation as built: their working-buffer rows depend on the processor), as `BuildPressureMemoryEstimate(sizes, run_on)`
  and `BuildGasMemoryEstimate(gas_sizes, run_on)`. `one_call` is the largest per-module `kOneCall` sum,
  viscosity and pressure added.
- **Plan decisions (amended 2026-11-14, Foundation as built): the step's working buffers.** `TryArrays` (Task 9) holds
  `SumBuffers sum_buffers` for `max_sum_count = max(limits.max_particle_count summed over materials, 240 *
  limits.max_cell_blocks)` values (the largest buffer `CheckStep` and the stage totals reduce: one value per
  particle or per face place) and `OrderBuffers order_buffers` for the summed particle limit (stage 2's grouping of
  particles by block on the GPU, Task 14). `BuildStepMemoryEstimate` adds, `where = kRunOn`, `kWholeRun`:
  `"step: sum buffers"` (kGivenCount, `given_count = max_sum_count`, `bytes_each =
  ComputeSumBuffersSize(run_on).bytes_per_item`), `"step: sum buffers fixed bytes"` (kGivenCount, `given_count =
  ComputeSumBuffersSize(run_on).fixed_bytes`, `bytes_each = 1`), and `"step: order buffers"` / `"step: order buffers
  fixed bytes"` the same way from `ComputeOrderBuffersSize(run_on)` and the summed particle limit. The working buffers
  of every other module (transfer's `TransferBuffers`, the solvers' `SumBuffers` and `OrderBuffers` in their buffers)
  comes in through that module's `Build<Module>MemoryEstimate`; the step adds no row for it.
- **Plan decisions (A15, `plan-decisions.md` "Batched solves (A15)"):** `enum class RunMode : std::uint8_t {
  kOnePiece, kBatched, kBatchedWithSolvesInPieces };` (`BuildRunPlan` itself returns only the first two; the caller
  copies `batches::BuildBatchPlan`'s `SolvePlace::kInPieces` into the third); `RunPlan` gains
  `std::uint32_t max_solve_loaded_blocks = 0;` (copied from `BatchPlan`, like `max_loaded_blocks`) and
  `std::uint64_t allowed_cpu_bytes = 0;` (design 5.9 step 3 applied to `on_cpu`). `BuildStepMemoryEstimate` and the
  estimate take the solvers' rows from their builders with `max_loaded_blocks = plan.max_solve_loaded_blocks` in
  the third mode (rows then carry their own `where`).
- **Amended 2026-11-14 (A17, batched-run buffers):** plan decisions for the budget shown to the user (I/O Task
  23a): `MemoryEstimate` gains `std::uint64_t on_cpu = 0;` (rows `where = kCpu` plus `snapshots`; `on_processor`
  sums only `kRunOn` rows); `ModuleBytes` gains `std::uint64_t on_cpu = 0;`; `enum class PlanFit : std::uint8_t {
  kFits, kDoesNotFit };` and `RunPlan` gains `PlanFit fit = PlanFit::kFits;` and `CopyCounts planned_step_copies;`
  (Appendix B formulas at the limits for the plan's mode). `struct CopyCounts { std::uint32_t copy_count = 0;
  std::uint64_t bytes_to_processor = 0; std::uint64_t bytes_from_processor = 0; };` is declared here in
  `run_plan.h` (Task 10's `StepReport` uses it). A scene that fits nowhere is no longer a failure of `BuildRunPlan`: it returns the plan with
  `fit = kDoesNotFit`, `mode = kBatched` and every estimate filled; `BuildSimulation` (Task 9) fails `kDoesNotFit`
  on it. `BuildStepMemoryEstimate(const ReadyScene& scene, const SceneCounts& counts, const RunPlan* batched_plan =
  nullptr)`: null gives the design 6.3 rows; a batched plan gives the rows of Appendix A (loaded-only rows at
  `max_loaded_blocks` / `max_loaded_particles` with `where = kRunOn`, whole rows placed by the placement rule, and
  the batched-run rows of Task 33a).
- **Plan decisions (reconcile item 11, grid room; `plan-decisions.md` "Reconcile pass (2026-11-14)").**
  `limits.max_cell_blocks` comes from this planner only: the cell blocks design 5.9 counts from the scene (the
  2.5-width cover of the start particles, as `BuildRunPlan, CountsOfFallingBlock` pins), times
  `start_limit_headroom` (1.25, the stated growth margin), rounded up, capped by the blocks of the domain box at
  the smallest cell size; at least 1. The step never builds a `grid::GridBuffers` or
  `grid::GridBuilderBuffers` from `grid::GridSettings`'s default (which is 0, `kBadInput`). The grid's entries are
  `grid::BuildGridMemoryEstimate(run_on, 2, limits.max_cell_blocks, grid::BuilderBuffersSharing::kOneForAllGrids)` (committed and free buffers,
  one builder buffers) plus, in a batched run, `grid::BuildGridMemoryEstimate(run_on, 1, limits.max_cell_blocks, kOneForAllGrids)` without
  its builder-buffers entries for the loaded try's buffers at `max_loaded_blocks` (they borrow the same builder
  buffers). A
  planned room above `grid::kMaxCellBlocksLimit` gives `fit = kDoesNotFit` (the budget prints the does-not-fit lines
  with the smallest-cell suggestion). During the run, a grid that outgrows the room is the existing
  `kTooManyCellBlocks` (`kRaiseLimits`) path: `ApplyLimits` (Task 24) raises by `limit_growth`. On a 16 GiB GPU
  with all of it free (`allowed_bytes` = min(floor(0.9 x 17,179,869,184), 17,179,869,184 - 536,870,912) =
  15,461,882,265 B, A58; replaced: "0.7 x 17,179,869,184 = 12.03e9 B"), two GPU grids with one set of builder
  buffers cost 2,840 B per block, so a quarter of the budget (3.87e9 B) holds about 1.36 M blocks (replaced: "3.01e9
  B ... about 1.06 M blocks"); the design's old default of `2^22` blocks would take 11.9e9 B, 77 % of the budget, for
  the grid alone.
- **Amended 2026-11-14 (A25), `plan-decisions.md` "A25: what counts as viscous".** Consumes
  `viscosity::ComputeViscousEffectPerStep(double viscosity, double step_duration, double density, double
  cell_width)` (`deformation_rate.h`, shared; `viscosity * step_duration / (density * cell_width * cell_width)`, no
  unit). Produces, in `run_plan.h`:

  ```cpp
  struct LiquidViscousEffect {                 // step run_plan.h, NEW (A25)
    std::uint32_t ingredient_index = 0;        // scene ingredient
    double viscosity = 0;                      // Pa s, the setting
    double density = 0;                        // kg/m3, rest density of its material
    double viscous_effect_per_step = 0;        // no unit; 0 when viscosity is 0
  };
  enum class ViscousPath : std::uint8_t { kPressureSolve, kViscousSolve };   // stage 12 or 11, fixed for the run
  // RunPlan gains:
  std::vector<LiquidViscousEffect> liquid_viscous_effects;
  double viscous_effect_cell_width = 0;        // m
  double viscous_effect_step_duration = 0;     // s
  ViscousPath viscous_path = ViscousPath::kPressureSolve;
  ```
  `BuildRunPlan` fills one `LiquidViscousEffect` per liquid ingredient (scene ingredient order) from the
  ingredient's viscosity and its material's rest density, with `viscous_effect_step_duration =
  StepSettings::max_step_duration` and `viscous_effect_cell_width` = the grid frame's smallest cell width;
  `viscous_path` is `kViscousSolve` exactly when some liquid ingredient's viscosity is above 0 (Task 7's rule).
  The number only advises (I/O prints it); it never chooses the stage.
- **Amended 2026-11-15 (A57), `plan-decisions.md` item 4.** `LiquidViscousEffect` gains (design 3.5 exactly)
  `std::optional<double> viscosity_to_smoothing_ratio` (`viscosity::ComputeViscosityToSmoothingRatio(N,
  own_motion_kept of the liquid's material)`), `double four_cell_feature_change`
  (`ComputeFourCellFeatureChange(viscosity, density, T, h)`), `double spreading_in_cells` (`sqrt(nu T) / h`), `double
  reynolds_number` (`ComputeReynoldsNumber(viscosity, density, L, |gravity|)`), `bool advises_zero_viscosity`
  (viscosity > 0 and (ratio < `kNegligibleViscosityRatio` or change < `kNegligibleViscousChange`)); `RunPlan` gains
  `double viscous_effect_run_duration` (T = `end_time - start_time`, s) and `double reynolds_length` (L, the domain
  box's smallest extent, m). A viscosity-0 entry: numbers 0, ratio as the function gives for N = 0, Reynolds
  +infinity, not advised.
- **Amended 2026-11-15 (A58).** In `run_plan.h`: `std::uint64_t ComputeAllowedMemoryBytes(MemoryAvailable available,
  double memory_share_of_free, std::uint64_t memory_left_free_bytes);` = `min(floor(share x free_bytes), free_bytes
  - memory_left_free_bytes)`, 0 when the second is negative (`total_bytes` unread). `allowed_bytes` and
  `allowed_cpu_bytes` use it. `MemoryEstimate::on_processor = ComputeRunMemoryBytes(all entries, counts, kRunOn)`
  (each copy rounded to 256 B), NEW `cpu_run_memory_bytes = ComputeRunMemoryBytes(kCpu entries kept kWholeRun or
  kOneStepTry, counts, kCpu)`, `on_cpu = cpu_run_memory_bytes +` the `kCpu` `kOneCall` entries `+ snapshots`;
  `one_call` sums `kCpu` entries only. Each step row's `copies` equals the number of buffers it stands for.
- **Amended 2026-11-15 (A64), `plan-decisions.md` item 1.** `SceneCounts` gains `std::vector<std::uint64_t>
  particles_at_depth_levels` (per material_index; empty without `has_cell_resize`) and `std::uint64_t
  start_particles_in_deep_cells`, computed on the CPU by a private `ComputeParticlesAtDepthLevels` in
  `run_plan.cc` exactly as design 5.9 states (start particles binned into `w_min` cells; column surface = top of
  its highest liquid cell; depth = surface - cell top; level from `grid::ComputeWantedSizeLevel(widths,
  grid::ComputeWantedCellWidth(scene.cell_size, widths, grid::ComputeReferenceSpeed(0, |g|, w_max), facts))` with
  only `cell_width` and `depth_below_surface` set and `distance_to_solid` +infinity; distinct ancestors x 8). The
  particle limit per material is `ceil(start_limit_headroom x (max(start, at depth levels) + from sources))`.
  `BuildStepMemoryEstimate` with `has_cell_resize` gives three rows (design 6.3): `"step: merge ready duration"` 8 B,
  `"step: held velocity gradient size"` 8 B, `"step: merge wait steps"` 1 B, each per cell place, 2 copies, `kRunOn`,
  `kWholeRun` (replaced: the seven region arrays, 34 + 8 x ingredients B). Test scene `BuildDeepWaterScene()` in
  `step_test_scenes.h` (closed box 0.4 x 0.4 x 1.0 m, water to 0.8 m at rest, biggest cell 0.1 m, 3 levels, start
  particles at level 2, 8 per cell at the eighth centres, `has_cell_resize`, `cell_size.smallest_size_level = 2`,
  `new_cell_size_level = 2`, surface reach 8).
- **Amended 2026-11-15 (A41).** `planned_step_copies` has no O4 (one piece: O1, O2, O3; batched: B1, B2, B4; B3 is
  in the later group, A64). The row `"step: new cells for the size choice"` is not built.
- **Amended 2026-11-15 (A40).** `BuildRunPlan(..., std::span<const MemoryEstimateEntry> cell_resize_estimate = {},
  std::optional<RunMode> forced_mode = std::nullopt)`: set, the plan has `mode = *forced_mode` and `fit` by step 5's
  test for that mode (design 3.5), never another mode.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(BuildRunPlan, CountsOfFallingBlock)        // 256 start particles; cell blocks of the 2.5-width cover;
      // limits = counts * 1.25 rounded up; blocks capped by the domain's blocks
  TEST(BuildRunPlan, SourceAddsRateTimesDuration) // 10 /s from 0 to 1 s -> particles_from_sources 10
  TEST(BuildRunPlan, JustBelowAllowed_IsOnePiece) // design 7 "Plan": on_processor chosen so on_processor bytes
  TEST(BuildRunPlan, JustAboveAllowed_IsBatched)  //   are allowed_bytes - 1, then + 1 with CPU room, then neither
  TEST(BuildRunPlan, FitsNowhere_PlanSaysDoesNotFit)  // amended (A17 budget): fit kDoesNotFit, mode kBatched,
      // estimate.on_processor and on_cpu filled (the I/O budget prints them)
  TEST(BuildRunPlan, OnCpuIsTheCpuRowsPlusSnapshots) // A17 budget: BuildFallingBlockScene with one water source
      // added through scene_to_change(), RunOn::kGpu one piece: on_cpu == ComputeMemoryBytes(rows, counts,
      // MemoryWhere::kCpu) + snapshots; by_module[kStep].on_cpu holds the source counter row; on_processor has no
      // kCpu row
  TEST(BuildRunPlan, PlannedStepCopies_AppendixBAtTheLimits)  // A17: the same scene with has_cell_resize, RunOn::kGpu:
      // one piece == Appendix B row O1 at the limits; forced batched == row B4 (amended 2026-11-15, A41, A64:
      // replaced "O1 + O4" and "B3 + B4"; B5 withdrawn; B1, B2 need gas; Task 29 adds the gas case to this test);
      // RunOn::kCpu one piece: all zero
  TEST(ComputeAllowedMemoryBytes, SmallerOfShareAndHeadroom)  // A58: share 0.9, left free 536,870,912 B: free 16e9 ->
      // 14,400,000,000; free 4e9 -> 3,463,129,088; free 0.4e9 -> 0; total_bytes changes nothing (exact).
      // Replaces TEST(BuildRunPlan, UnknownTotal_UsesFreeShareOnly) (total_bytes 0)
  TEST(BuildRunPlan, AllowedBytesFromFreeOnly)               // A58: on_processor {total 32e9, free 16e9}:
      // allowed_bytes 14,400,000,000; on_cpu {total 64e9, free 4e9}: allowed_cpu_bytes 3,463,129,088
  TEST(BuildRunPlan, OnProcessorIsRunMemoryBytes)            // A58: RunOn::kGpu plan of BuildFallingBlockScene with one
      // source: on_processor == ComputeRunMemoryBytes(all entries, counts, kRunOn), cpu_run_memory_bytes ==
      // ComputeRunMemoryBytes(kCpu kWholeRun/kOneStepTry entries, counts, kCpu); no kRunOn entry has kOneCall
  TEST(BuildRunPlan, DeepWaterCountsEveryLiquidCellAtDepthLevels)  // A64: BuildDeepWaterScene: layers 0..15 below the
      // surface stay at 0.025 m (4,096 cells), layers 16..31 at 0.05 m (512 cells): particles_at_depth_levels[water]
      // 36,864, start_particles_in_deep_cells 49,152 (start particles of layers 8..31); start 65,536; the water's
      // particle limit ceil(1.25 x 65,536) = 81,920; the same scene without has_cell_resize: the vector empty, the
      // limit unchanged (exact)
  TEST(BuildRunPlan, ForcedOnePieceThatDoesNotFit_KeepsModeSaysDoesNotFit)  // A40: free bytes so that on_processor
      // is allowed_bytes + 1, forced_mode kOnePiece: mode kOnePiece, fit kDoesNotFit; the same with forced_mode empty:
      // mode kBatched, fit kFits
  TEST(BuildRunPlan, ForcedBatchedThatFitsOnePiece_StaysBatched)  // A40: a scene that fits one piece, forced_mode
      // kBatched: mode kBatched, fit kFits, max_loaded_blocks 0
  TEST(BuildRunPlan, OutputRows_CountedUnderRenderOutput)     // one hand-written MemoryEstimateEntry, by_module[kRenderOutput]
  TEST(BuildRunPlan, CellResizeRows_CountedUnderCellResize)
  TEST(BuildRunPlan, SceneThatCannotRun_IsSceneNotSupported)
  TEST(BuildRunPlan, OnGpuWithoutAGpu_StillPlans)             // inputs are given; nothing allocated
  TEST(BuildRunPlan, AllowedCpuBytes_FromCpuShare)            // A15: on_cpu given: allowed_cpu_bytes by step 3
  TEST(BuildRunPlan, SolvesInPieces_SolverRowsCountedOnCpu)   // A15: a plan copied to kBatchedWithSolvesInPieces with
      // max_solve_loaded_blocks 64: the pressure rows counted per place move to MemoryWhere::kCpu, the loaded rows
      // at 64 blocks stay kRunOn
  TEST(BuildStepMemoryEstimate, StepWorkingBufferEntries_FromBuffersSizes)  // amended (Foundation as built, A31):
      // RunOn::kGpu plan of BuildFallingBlockScene: the four step working-buffer entries equal ComputeSumBuffersSize(kGpu) and
      // ComputeOrderBuffersSize(kGpu) applied to the counts above; a RunOn::kCpu plan: sum entries 0 bytes,
      // order entries {4, 8192}
  TEST(BuildStepMemoryEstimate, GasAddsGasVelocityAndKeepActive)
  TEST(BuildStepMemoryEstimate, CellResizeAddsThreeCellSizeArrays)  // A64, A63 (replaces CellResizeAddsSevenRegion
      // Arrays, 34 + 8 x ingredients bytes): the three named rows, 8 + 8 + 1 = 17 B per cell place, two copies each
  TEST(BuildRunPlan, BlockRoomFromSceneNotFromDefault)  // reconcile: BuildFallingBlockScene: limits.max_cell_blocks ==
      // ceil(1.25 * cover blocks), well below 1u << 22; a scene whose domain holds fewer blocks: capped there
  TEST(BuildStepMemoryEstimate, TwoGridBuffersShareBuilderBuffers)  // reconcile: RunOn::kGpu plan: the "grid: builder buffers:"
      // entries have copies 1, the other grid entries copies 2; grid bytes == 2840 * max_cell_blocks + fixed
  TEST(BuildRunPlan, RoomAboveGridLimit_DoesNotFit)  // reconcile: a domain of kMaxCellBlocksLimit + 1 cover blocks
      // (counts given, nothing allocated): fit kDoesNotFit
  TEST(BuildRunPlan, ViscousEffectPerLiquid)  // A25: water 1e-3 Pa s 1000 kg/m3 and syrup 10 Pa s 1400 kg/m3,
      // smallest cell 0.01 m, max_step_duration 0.01 s: two entries in ingredient order with 1e-4 and
      // 0.714285714..., 1e-15 relative; viscous_effect_cell_width 0.01, viscous_effect_step_duration 0.01;
      // viscous_path kViscousSolve. Amended 2026-11-15 (A57): own_motion_kept 0.95 for both, start 0, end 10 s,
      // domain smallest extent 1 m, gravity 9.81: viscous_effect_run_duration 10, reynolds_length 1; water ratio
      // 0.008, change 0.21865627, spreading 0.31622777, Reynolds 4.4294469e6, advised; syrup ratio 57.142857,
      // change 1, spreading 26.726124, Reynolds 620.12257, not advised (1e-12 relative)
  TEST(BuildRunPlan, PureFlipLiquid_HasNoSmoothingRatio)  // A57: own_motion_kept 1: viscosity_to_smoothing_ratio
      // empty; advises_zero_viscosity from the change alone (water above: true)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.9 steps 1 to 5. Counting the cover of the start
  particles reuses the marking rule of design 5.3 step 1 on the CPU (a private `ComputeNeededBlocks` in
  `choose_active_cells_rule.h`, CPU-only, also used by Task 13).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Estimate a scene's memory and choose how it runs`. New
  names (amended 2026-11-15): `ComputeAllowedMemoryBytes`, `MemoryEstimate::cpu_run_memory_bytes`,
  `SceneCounts::particles_at_depth_levels`, `start_particles_in_deep_cells`, the A57 fields, `forced_mode`.

### Task 9: `Simulation` and `BuildSimulation`

**Amended 2026-11-14 (A17, batched-run buffers):** the memory report splits allocated bytes by processor, and a plan
that does not fit is refused here.

**Amended 2026-11-14 (reconcile):** the simulation owns one `grid::GridBuilderBuffers` lent to both grid buffers.

**Amended 2026-11-14 (A24):** the viscosity buffers hold the sealed gas blobs.

**Amended 2026-11-14 (A25):** the viscous path is fixed here, once, from the ready scene.

**Amended 2026-11-14 (A26, A27):** the viscosity buffers also hold the material-face and free-solid unknowns.

**Amended 2026-11-14 (Foundation as built):** `TryArrays` builds the step's `sum_buffers` and `order_buffers` once
(Task 8's counts); a failed build is `kOutOfMemory`, mapped as any `BufferError`.

**Amended 2026-11-15 (A58, with the same-day correction):** `BuildSimulation` reserves the run's blocks before
anything else and builds every buffer as a piece; `MemoryReport` reports planned, reserved and handed-out bytes;
`allocated_bytes` / `allocated_cpu_bytes` become `handed_out_bytes` / `handed_out_cpu_bytes`; `FindRunMemory` and
`FindCpuRunMemory` let I/O build render output and cell resize into the blocks.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/simulation.h`, `simulation.cc`, `simulation_state.h` (private, CPU only),
  `simulation_test.cc`, `simulation_no_gpu_test.cc`
- Modify: `native/fluidity/step/advance_to.h` (created here with `Callback`, `StepCallbacks`, `KeepGoing`,
  `ResizeResult`, `FoamAndSprayInputs` of design 3.9 only; `AdvanceTo` comes in Task 23)

**Interfaces:**
- Produces design 3.6: `Simulation` (all accessors), `BuildSimulation`, `ReadMemoryReport`, `SourceCounter`,
  `CommittedStepLimits`; `SimulationSnapshot`, `ReadSnapshot` and `BuildSimulationFromSnapshot` are declared in
  Task 11, `ApplyLimits` in Task 24.
- **Plan decisions:** `simulation_state.h` holds `struct CommittedState` (grid, `GridFields`, air pockets,
  counters, `CommittedStepLimits`, time, versions) and `struct TryArrays` (every try copy and working buffer of design
  5.6, built here at the limits, including `SumBuffers sum_buffers` and `OrderBuffers order_buffers` from
  `BuildSumBuffers(run_on, max_sum_count)` and `BuildOrderBuffers(run_on, summed particle limit)`, Task 8; amended
  2026-11-14, Foundation as built); `Simulation` holds two `grid::GridBuffers`s, the committed one and the free
  one, built with `grid::BuildGridBuffers(settings, grid_builder_buffers)` from one `grid::GridBuilderBuffers` the simulation
  state owns (`grid::BuildGridBuilderBuffers(run_on, limits.max_cell_blocks)`, built first, freed last, never moved
  while the buffers exist; reconcile item 11). The loaded try's buffers of a batched run (Task 33) borrow the same
  builder buffers. **Amended 2026-11-14 (A31):** `BuildGridBuffers` has one form, taking the lent
  `GridBuilderBuffers`; test `StepScratchBuiltAtTheLimits` renamed `StepWorkingBuffersBuiltAtTheLimits`. `allocated_bytes` is summed from the rows actually walked while building. With `RunMode::kBatched` this
  task returns `kBatchedRunNotBuilt` whatever the callbacks (Task 33 builds the batched layout).
- **Amended 2026-11-14 (A17, batched-run buffers):** `MemoryReport` gains `std::uint64_t allocated_cpu_bytes = 0;`
  (rows walked with `where = kCpu`); `allocated_bytes` sums the `kRunOn` rows only, so `allocated_bytes ==
  estimate.on_processor` and `allocated_cpu_bytes == estimate.on_cpu - estimate.snapshots`. A plan with `fit ==
  PlanFit::kDoesNotFit` gives `kDoesNotFit` before anything is allocated.
- **Amended 2026-11-14 (A25):** `BuildSimulation` decides the viscous path from the ready scene with Task 7's rule
  (some liquid ingredient's viscosity above 0: stage 11; else stage 12) and keeps it in the state; Task 10's
  `stage_runs` reads it. Nothing changes it afterwards: not a retry, a shortened step, `ApplyLimits`, a cell resize
  (`CommitResize`), a batched layout, `BuildSimulationFromSnapshot` (same rule on the same scene), nor the viscous
  effect number of Task 8.
- **Amended 2026-11-14 (A24):** `viscosity::BuildViscosityBuffers` gets `ViscosityBufferSettings::
  max_sealed_gas_blobs` = gas's `max_gas_blobs` (0 without gas), so stage 11 can carry every sealed blob stage 10
  lists.
- **Amended 2026-11-14 (A26, A27):** `ViscosityBufferSettings::max_material_faces` = 2 x solids'
  `max_motion_map_entries` (0 without a blocking set) and `max_free_solids` = the scene's free solid count, so stage
  11 can carry every `ViscousMaterialFace` and every free solid touching liquid that stage 10 lists.
- **Amended 2026-11-15 (A58; design 3.5, 3.6, 5.9).** Order in `BuildSimulation`: `CheckSceneCanRun`;
  `CheckCanRunOn`; `fit` check; `ReadMemoryAvailable(run_on)` (free before); `BuildRunMemory(run_on,
  plan.estimate.on_processor)` and, in a GPU run, `BuildRunMemory(RunOn::kCpu, plan.estimate.cpu_run_memory_bytes,
  batched ? kLockedForCopies : kOrdinary)` (in a CPU run one block of both sums, and `cpu_run_memory` refers to it);
  `ReadMemoryAvailable` again (`memory_taken_bytes`, GPU only); then every module builder and the private
  `BuildCommittedState(..., run_memory, cpu_run_memory)`, `BuildTryArrays(..., run_memory)` (with `BuildSumBuffers(run_on,
  max_sum_count, run_memory)` and `BuildOrderBuffers(run_on, summed particle limit, run_memory)`), each block passed
  last. `simulation_state.h` holds the blocks as the first members of the state (freed last). `MemoryReport` gains
  design 3.5's `planned_bytes`, `reserved_bytes`, `handed_out_bytes`, `reserved_block_count`, `memory_taken_bytes`,
  `requested_beyond_plan_bytes`, `handed_out_cpu_bytes`, `reserved_cpu_bytes` (replaced: `allocated_bytes`,
  `allocated_cpu_bytes`). A builder's `kOutOfMemory` or `kRunMemoryTooSmall` is `ToBuildFailure` (Task 3); on any
  failure the pieces are dropped, then the blocks. `simulation.h` gains `RunMemory& FindRunMemory(Simulation&)` and
  `RunMemory& FindCpuRunMemory(Simulation&)` (design 3.6; the same object in a CPU run), which I/O uses to build
  cell resize, the batched run and render output right after `BuildSimulation`. `BuildSimulation` calls no
  `StepCallbacks` function. `memory_taken_bytes` is measured at the latest reservation.

- [ ] **Step 1: Failing tests.**

  ```cpp
  // simulation_test.cc, LABEL cpu
  TEST(BuildSimulation, FallingBlock_TimeVersionAndCounts)   // time == start_time, state_version 1, step_count 0,
                                                             // 256 particles, grid version 1
  TEST(BuildSimulation, HandedOutEqualsPlanned)              // amended 2026-11-15 (A58; was AllocatedEqualsEstimate):
      // ReadMemoryReport(sim): handed_out_bytes == reserved_bytes == planned_bytes == plan.estimate.on_processor,
      // reserved_block_count 1, memory_taken_bytes 0 (CPU)
  TEST(BuildSimulation, ReservesOneBlockFirst)               // A58: a replaced global aligned operator new counting
      // calls: the first call during BuildSimulation asks plan.estimate.on_processor bytes (a CPU run of
      // BuildFallingBlockScene(1) has no kCpu row), and no later call asks bytes for a kRunOn row
  TEST(BuildSimulation, StartParticlesBadState_IsBadInput)   // materials::CheckMaterialState fails: nothing kept
  TEST(BuildSimulation, SceneThatCannotRun_IsSceneNotSupported)
  TEST(BuildSimulation, BatchedPlan_IsBatchedRunNotBuilt)
  TEST(BuildSimulation, PlanThatDoesNotFit_IsDoesNotFitAndNothingAllocated)  // A17 budget: fit kDoesNotFit; the
      // replaced operator new counts nothing (A58: no block reserved)
  TEST(BuildSimulation, CpuBlockEqualsCpuRows)               // A17 budget, amended 2026-11-15 (A58; was
      // AllocatedCpuBytesEqualCpuRows): BuildFallingBlockScene with one source, a forced batched CPU plan (mirrors
      // are kCpu rows): handed_out_cpu_bytes == reserved_cpu_bytes == estimate.cpu_run_memory_bytes
  TEST(BuildSimulation, EmptyScene_HasNoActiveCell)          // BuildEmptyScene: grid with zero blocks, version 1
  TEST(BuildSimulation, GridBuffersShareBuilderBuffers)          // reconcile: the bytes walked for the grid equal
      // ComputeMemoryBytes(BuildGridMemoryEstimate(kCpu, 2, limits.max_cell_blocks, kOneForAllGrids), counts); a try that builds a new
      // grid version leaves the committed grid's tables byte-identical
  TEST(BuildSimulation, StepWorkingBuffersBuiltAtTheLimits)   // amended (Foundation as built): sum_buffers.max_count()
      // == max_sum_count, order_buffers.max_count() == the summed particle limit, both run_on() == settings.run_on;
      // their bytes are pieces of run_memory (A58; was "part of allocated_bytes")
  TEST(FindRunMemory, OutputRowsHaveRoomAfterBuild)          // A58: a plan with one hand-written output_estimate row
      // of 1000 B kRunOn: after BuildSimulation handed_out_bytes == reserved_bytes - 1024; BuildBuffer<std::byte>(
      // run_on, 1000, FindRunMemory(sim)) succeeds and then handed_out_bytes == reserved_bytes
  // simulation_no_gpu_test.cc, LABEL no_gpu
  TEST(BuildSimulation, OnGpuWithoutAGpu_IsRunOnCallFailedAndNothingAllocated)  // counter of the replaced new: 0
  ```
  If `grid::BuildGrid` refuses an empty cell list, stop and ask the owner: the step does not special-case it.
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.9 last paragraph and 3.6: run `CheckSceneCanRun`
  again; `CheckCanRunOn(settings.run_on)`; build every module's state and buffers at the plan's limits
  (`grid::BuildGridBuilderBuffers` once, `grid::BuildGridBuffers(settings, grid_builder_buffers)` twice, `particles::BuildParticles` with `ParticlesSettings::max_cell_blocks` = the
  grid limit, `transfer::BuildTransferBuffers(run_on, summed particle limit, material_count)` (reconcile 27), per set `BuildGridFromParticles`, `BuildCellsFromParticles`,
  two `BuildGridSnapshot`s (`FilledFromSolved::kNotKept` for a set no solve writes), two `BuildSolidsAtFaces`,
  `pressure::BuildPressureBuffers` twice when gas is present, `viscosity::BuildViscosityBuffers` with a viscous
  ingredient, `materials::BuildMaterialBuffers(run_on, table, max_particle_counts, max_cell_blocks)`,
  `solids::BuildSolids`, `BuildSolidsBuffers(solids_settings, max_particle_count, max_cell_blocks)` and
  `BuildSolidCoverageValues(run_on, max_cell_blocks)`,
  `gas::BuildGasAmounts` and `BuildGasBuffers` with gas, `contact::BuildContactBuffers`); add the start
  particles; `materials::CheckMaterialState`; choose the first active cells with Task 8's rule and group.
  On any failure every buffer already built is released before returning.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Build a simulation from a ready scene and its plan`. New
  names (amended 2026-11-15, A58): the six `MemoryReport` fields, `handed_out_cpu_bytes`, `reserved_cpu_bytes`,
  `FindRunMemory`, `FindCpuRunMemory`.

### Task 10: The try loop on an empty scene

**Amended 2026-11-14 (A17, batched-run buffers):** the try report and the simulation count the step's own copies
between CPU memory and the run's processor.

**Amended 2026-11-14 (Foundation as built):** `BuildRejectingStageTable` lives in `fluidity_step_test_support`
(tests `LINKS` it); a stage that returns `kRunOnCallFailed` leaves its try outputs unspecified and the simulation
unusable, never a partial commit.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/step_stages.h`, `try_step.h`, `try_step.cc`, `stage_table.cc`,
  `check_step.h`, `check_step_cpu.cc`, `commit_step.h`, `commit_step_cpu.cc`, `try_step_test.cc`
- Modify: `simulation.cc`, `simulation_state.h` (`stage_runs`), `step_test_support.h/.cc`
  (`BuildRejectingStageTable`)

**Amended 2026-11-14 (A28, A29):** `StepRequest` is `StepInputs`, the parameter `step_inputs`, `kBadRequest`
`kBadStepInputs`; a reject takes the storage back with `grid::TakeBuffers`.

**Amended 2026-11-15 (A60, A62):** `StepReport` declares `std::vector<solids::LiquidForceOnSolid>
liquid_force_on_solids` (filled by Task 26; empty here) and `CommittedStepLimits` has `largest_liquid_speed` (filled
by Task 21), no `grid_only_speed` (A64).

**Interfaces:**
- Produces design 3.7 (`StepInputs`, `StepTimings`, `StepReport`, `StepResult`, `TryStep`) and 3.11 except
  `StepArray` / `ReadStepArrays` (Task 33): `StepTry`, `StageFunction`, `StageTable`, `kStages`,
  `TryStepWithStages`.
- **Plan decisions:** whether a stage runs is fixed per simulation from design 5.2's "Skipped when" column,
  computed once in `BuildSimulation` (`std::array<std::uint8_t, kStageCount> stage_runs` in the state); a null
  table entry is "not a stage here" and is not recorded. `kStages` holds, at the end of this task, stages 19
  (`CheckStep`) and 20 (`CommitStep`) only; each later task fills its entries, and the comment above `kStages`
  lists the stages present. `TryStepWithStages` runs stages 1 and 2 on the whole state, builds the ranges (one
  range, every block) from the try grid, runs 3 to 19 through the table (or `run_stages` from Task 33), then 20.
  `kResizeCells` is recorded by `AdvanceTo`, never by `TryStepWithStages`.
- **Plan decisions (A17, batched-run buffers):** `StepReport` gains `CopyCounts copies{};` (the step-own copies of
  this try, Appendix B; valid for a reject up to the failed stage) and `simulation.h` gains `CopyCounts
  ReadCopyCounts(const Simulation& simulation);` (cumulative since `BuildSimulation`: every try, committed or not,
  and the grid copy refresh of Task 33b). Each stage adds what its calls copy: module results that report copies
  (`particles::AddParticlesResult`, `gas::MarkerChanges`) and the step's own `ReadBuffer` / `WriteBuffer` slices,
  one copy per call, bytes as passed.
- Produces (test-only): `StageTable BuildRejectingStageTable(Stage stage, RejectReason reason)`: a copy of
  `kStages` whose entry `stage` runs the real stage (when the entry is not null) and then returns `reason`
  (design 5.12). Plain function pointers carry no context, so it is built from one template instantiation per
  (stage, reason) pair over the representative reasons `kParticleTravelledTooFar` (shorten),
  `kTooManyParticles` (raise limits), `kCancelRequested` (cancelled) and `kStageFailed` (cannot continue).

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `BuildEmptyScene`):

  ```cpp
  TEST(TryStep, EmptyScene_Commits)               // committed, time 0.01, state_version 2, step_count 1,
                                                  // stages_run == {kCheckStep, kCommitStep}
  TEST(TryStep, StaleVersion_IsWrongStateVersion) // design 7 "Stale version": kBadStepInputs, rejected_in kNone,
                                                  // stage_count 0
  TEST(TryStep, ZeroOrNaNOrTooSmallDuration_IsBadStepDuration)  // 0, -1, NaN, and time + d == time
  TEST(TryStep, CancelBeforeTheTry_IsCancelRequested)  // nothing ran; next TryStep commits
  TEST(TryStepWithStages, RejectAfterCheckStep_ChangesNothing)  // for each of the four representative reasons:
      // the committed time, versions, grid version and particle count before == after; rejected_in kCheckStep;
      // shorter_step_duration present exactly for kShortenStep (= d/2); limits_needed exactly for kRaiseLimits
  TEST(TryStep, UnusableSimulation_IsSimulationUnusable)  // after a stage table returning kRunOnCallFailed
      // (amended, Foundation as built: the try's outputs are unspecified after it; the test checks only that the
      // committed snapshot taken before is unchanged and that every later call gives kSimulationUnusable)
  TEST(TryStep, OnePieceCpu_CountsNoCopy)         // A17: copies == {} after 3 tries; ReadCopyCounts == {}
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** the step-input checks of design 5.2 (preamble), `cancel_requested`
  read before the try and before every stage, `particles::BuildTry`, the stage loop with `StepTimings`, the
  reject path (`RejectTry` of every try, storage back with `grid::TakeBuffers`), `CheckStep` (design 5.6 list:
  `time + step_duration > time`; ids and counters did not go back, ported from `main`
  `liquid_state_3d.cc:789-816`), `CommitStep` (`particles::CommitTry`, the swaps, `time += step_duration`,
  `state_version += 1`, `step_count += 1`; a failed module commit marks the simulation unusable).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Try and commit a step of an empty scene`. New names: the 3.7
  and 3.11 list above.

### Task 11: Snapshots, and the byte comparison every reject test uses

**Amended 2026-11-14 (reconcile):** the grid is rebuilt from the snapshot with designated `GridInputs` fields and,
in a GPU run, from the list uploaded into the active cell list buffer.

**Amended 2026-11-14 (Foundation as built):** `ExpectSameBytes` joins `fluidity_step_test_support` and compares
snapshots taken on one processor only (reject, replay, reload); a CPU-against-GPU comparison uses the design 6.4
tolerances, never `ExpectSameBytes`.

**Amended 2026-11-15 (A63, A64, A58):** `SimulationSnapshot` carries the three cell-size arrays
(`merge_ready_duration`, `held_velocity_gradient_size`, `merge_ready_step_count`; empty without cell resize) instead
of the seven region arrays; `BuildSimulationFromSnapshot` reserves its blocks first, as `BuildSimulation`. Test
`BuildSimulationFromSnapshot, CellSizeArraysRoundTrip` (a `has_cell_resize` falling block with hand-set values 0.125
s, 3.5 1/s and 7 in one cell: `ExpectSameBytes` after the round trip).

**Kind:** integration

Placed this early so that every later "reject after this stage changes nothing" test compares whole
snapshots with `ExpectSameBytes`. The continued-run tests that need the liquid stages are in Tasks 21, 29 and 31.

**Files:**
- Create: `native/fluidity/step/simulation_snapshot.cc`, `simulation_snapshot_test.cc`
- Modify: `native/fluidity/step/simulation.h` (`SimulationSnapshot`, `ReadSnapshot`,
  `BuildSimulationFromSnapshot`), `step_test_support.h/.cc` (`ExpectSameBytes`, comparing every field of
  `SimulationSnapshot` as bytes, `NaN` payloads included, doubles through `ToBits` of `base/test_support.h`; on a
  mismatch it names the field and the first index; for two snapshots of the same processor only),
  `try_step_test.cc` (Task 10's reject test now uses it)

**Interfaces:**
- Produces design 3.6 snapshot names (`SimulationSnapshot`, `ReadSnapshot`, `BuildSimulationFromSnapshot`) and
  design 5.13: particles through `particles::ReadSnapshot` / `BuildParticlesFromSnapshot` (stored order kept,
  never regrouped), the grid rebuilt with `grid::BuildGrid(GridInputs{.frame = grid_frame, .active_cells =
  active_cells, .version = grid_version}, committed_buffers)` in a CPU run (reconcile: designated fields; the A17
  `GridInputs` has two more members), and in a GPU run from the snapshot's list written into the `"step: active cell
  list"` buffer (one slice `WriteBuffer`, counted in `ReadCopyCounts`) with `.active_cell_buffer` and
  `.active_cell_buffer_count` (a CPU list with GPU buffers is `kBadInput`, A17),
  `GridFields` with `grid::ReadValues` / `WriteValues`, solids through `solids::ReadSnapshot` /
  `BuildSolidsFromSnapshot`, gas through `gas::ReadGasAmountsSnapshot` / `BuildGasAmountsFromSnapshot`, counters,
  air pockets, `BlobHistory`, `CommittedStepLimits` and limits as stored.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ReadSnapshot, FallingBlock_EveryField)            // active_cells ascending, one entry per active cell;
      // pressure, has_pressure, pocket_of_cell by active cell; source_counters; limits
  TEST(BuildSimulationFromSnapshot, RoundTrip_SameSnapshot)  // BuildFallingBlockScene and BuildEmptyScene after one
      // committed try: ReadSnapshot -> BuildSimulationFromSnapshot -> ReadSnapshot: ExpectSameBytes
  TEST(BuildSimulationFromSnapshot, WrongMaterialCount_IsSnapshotDoesNotMatch)
  TEST(BuildSimulationFromSnapshot, NotFull_IsSnapshotDoesNotMatch)       // particle_fields not all
  TEST(BuildSimulationFromSnapshot, GridVersionKept)                      // version 1 rebuilt as version 1
  TEST(ReadSnapshot, UnusableSimulation_IsSimulationUnusable)
  TEST(BuildSimulationFromSnapshot, CellSizeArraysRoundTrip)  // A63: see the amendment above
  ```
  `ReadSnapshot, DuringAResizeTry_IsTryInProgress` is written in Task 32, where `ResizeTry` exists.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Read a simulation's state and rebuild it from a snapshot`. New names: `SimulationSnapshot`, `ReadSnapshot`,
  `BuildSimulationFromSnapshot`.

### Task 12: `AddNewParticles` from sources, on the CPU

**Amended 2026-11-14 (A17, batched-run buffers):** the stage adds `AddParticlesResult::copy_count` and
`bytes_copied` to `StepReport::copies` (Appendix A row M1, Appendix B row O1).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/add_new_particles.h`, `add_new_particles_cpu.cc`, `add_new_particles_test.cc`
- Modify: `stage_table.cc` (entry 1), `step_test_scenes.h/.cc` (`BuildSourceScene(double rate)`: empty box, one
  water source at the centre, spread 0.02 m, velocity `{0, 0, -1}`)

**Interfaces:**
- Consumes: `ComputeRandomWords`, `ComputeSignedUnitValue` (Task 4); `particles::AddParticles` with
  `AddReason::kFromSource`.
- Produces: stage 1 part (a) of design 5.2 and the counters of design 5.4. Open sea (b) comes in Task 27,
  gas markers (c) in Task 29.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(AddNewParticles, Sources_SameParticlesWhateverTheSteps)  // design 7 "Sources": rate 10, 1 s in steps of
      // 0.03 s (the last shorter) and of 0.01 s: 10 particles, consecutive ids, identical birth positions
  TEST(AddNewParticles, ChildPosition_FromRandomWords)  // child 0 of source 7: position + spread * (2u - 1) per axis
  TEST(AddNewParticles, OwedFraction_CarriedToNextStep)  // rate 10, step 0.03: 0 then 0 then 0 then 1 particles,
                                                         // owed 0.3, 0.6, 0.9, 0.2
  TEST(AddNewParticles, SourceStartsInsideStep_AddsNothingThisStep)  // Review Focus 3
  TEST(AddNewParticles, TwoSourcesOneMaterial_OneCallAscendingIds)   // ids of source 2 follow those of source 1
  TEST(TryStepWithStages, RejectAfterAddNewParticles_UsesNoIdAndNoChildNumber)  // next_particle_id, counters,
      // committed particles unchanged; the next commit gives the ids the rejected try would have used
  TEST(AddNewParticles, PastTheLimit_IsTooManyParticles)  // max_added_per_try 1, two due: kRaiseLimits with
                                                          // limits_needed max(2, 1 * 1.5)
  TEST(AddNewParticles, CopiesOfAddParticlesCounted)      // A17 M1: the try's copies equal the sum of the
      // AddParticlesResult copy fields of its calls (0 on the CPU; the GPU case is in Task 14)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.4 (ported from `main`
  `liquid_substep_3d_staging.cc:632-685`, FIXED: no step number in the key). New particles have
  `kHasOwnMotion` off. The counters are a CPU `std::vector<SourceCounter>` try copy swapped at commit.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add particles from sources in the step`.

### Task 13: `ChooseActiveCells` on the CPU

**Amended 2026-11-15 (A41, A61, A64):** new cells take `grid::ComputeNewCellSizeLevel` (no `choose_cell_size`
callback); no grid-only mark; a new cell's three cell-size values are 0.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/choose_active_cells.h`, `choose_active_cells_cpu.cc`, `choose_active_cells_test.cc`
- Modify: `choose_active_cells_rule.h` (Task 8), `stage_table.cc` (entry 2)

**Interfaces:**
- Produces: stage 2 of design 5.2 without gas (parts (a) to (d) and (f)), design 5.3 steps 1 to 4. Binds every
  try array that exists so far with `grid::ApplyGridVersion` and carries `GridFields` by `CellId` and
  `FacePieceId` (new cell: `has_pressure` 0, pressure 0, `pocket_of_cell = kNoAirPocket`, the three cell-size
  arrays 0). Each new cell's level is `grid::ComputeNewCellSizeLevel(committed.tables(RunOn::kCpu), cell centre,
  scene.new_cell_size_level)` (A41, A61; replaced: "The `choose_cell_size` callback is honoured.").

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `BuildFallingBlockScene(1)`):

  ```cpp
  TEST(ChooseActiveCells, CoverIsTwoAndAHalfWidths)       // every cell within 2.5 widths of a particle is in a
                                                          // needed block; a block farther away is not
  TEST(ChooseActiveCells, SameCells_KeepsCommittedVersion)
  TEST(ChooseActiveCells, ParticleMovedAcrossABlock_BuildsVersionPlusOne)  // particle moved by a test stage table
  TEST(ChooseActiveCells, RejectedNewGrid_StorageReturnedAndVersionReused) // Review Focus 1
  TEST(ChooseActiveCells, NewCellBesideBigCell_OneLevelFiner)  // A41, A61 (replaces CallbackGivesLevel): start
      // particles at level 0 (0.1 m cells), new_cell_size_level 1; a stage table moves one particle into a new block
      // beside a level-0 cell: the new cells have level 1
  TEST(ChooseActiveCells, NewCellFarFromCells_TakesNewCellSizeLevel)  // A41 (replaces NullCallback_TakesNearest
      // CellsLevel): the particle moved beyond every probe of ComputeNewCellSizeLevel: level 1
  TEST(ChooseActiveCells, CarriesPressureByCellId)        // committed pressure 5 Pa in cell C: the try has 5 Pa in C
  TEST(ChooseActiveCells, PastTheBlockLimit_IsTooManyCellBlocks)  // kRaiseLimits
  TEST(TryStepWithStages, RejectAfterChooseActiveCells_ChangesNothing)  // four reasons; grid version, particle
                                                                        // order and group_begin unchanged
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.3. `kSizeStepRuleBroken` is `kStageFailed`;
  `kOutsideGrid` after grouping is `kParticleOutsideGrid`.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Choose the active cells of a step try`.

### Task 14: Stages 1 and 2 on the GPU

**Amended 2026-11-14 (A31):** `BuildStableOrder` has one form, taking `TryArrays::order_buffers`.

**Amended 2026-11-14 (A17, batched-run buffers):** the needed cell list stays on the GPU and `grid::BuildGrid` runs
there (A17; the design's "ids come back and the CPU builds the grid" is withdrawn, Appendix A row M6); the copies of
stage 1 are counted (M1); a `choose_cell_size` callback is served by two counted copies (M8; superseded 2026-11-15, A41: no callback, no
copy).

**Amended 2026-11-14 (reconcile):** `ChooseActiveCells, OnGpu_EqualsCpu` and `OnGpuTwice_IdenticalBytes` compare
through a test-side `grid::ReadGrid` and the stage status record's entry count, not a needed-cell list read back;
the `committed` tables the step passes to `choose_cell_size` are stated (superseded 2026-11-15, A41).

**Amended 2026-11-15 (A41):** new cells' levels come from `grid::ComputeNewCellSizeLevel` in the GPU marking pass;
the size-choice copies (row O4), the row `"step: new cells for the size choice"` and the test
`OnGpuSizeChoiceCopiesCounted` are withdrawn.

**Amended 2026-11-14 (Foundation as built):** private `_gpu.h` headers returning `RunOnCallStatus` (no
`_gpu_calls.h`), launches through `base/gpu_launch.cuh`, kernel bodies in `add_new_particles_steps.h` /
`choose_active_cells_steps.h` with a `cpu` test over every thread block and thread, the particle grouping through
`BuildStableOrder` with `TryArrays::order_buffers` (its one form; A31), digests in the `gpu` tests.

**Kind:** device

**Files:**
- Create: `native/fluidity/step/add_new_particles_gpu.cu`, `add_new_particles_gpu.h` (private),
  `add_new_particles_steps.h` (shared: one child's position and velocity from `ComputeRandomWords`),
  `choose_active_cells_gpu.cu`, `choose_active_cells_gpu.h` (private), `choose_active_cells_steps.h` (shared: the
  block id of one particle, the unique-block flag of one sorted entry, the cells of one needed block written into
  the list, one level applied, the carry of one cell's or face's `GridFields` values by `CellId` / `FacePieceId`),
  `step_stages_steps_test.cc` (`cpu`), `step_stages_gpu_test.cc`,
  `step_stages_no_gpu_test.cc`
- Modify: `native/fluidity/step/CMakeLists.txt` (the two `.cu` through `fluidity_add_gpu_sources`, the steps headers
  in `fluidity_add_headers`, the three tests),
  `add_new_particles_cpu.cc`, `choose_active_cells_cpu.cc` (dispatch on `run_on`: the GPU half under
  `#if defined(FLUIDITY_HAS_GPU)`, its `RunOnCallStatus` mapped by `ToStageOutcome(ToBufferError(status))`),
  `step_memory_estimate.cc` (row `"step: active cell list"`: 8 B per cell place, `kOneStepTry`, `where = kRunOn`, in a
  GPU run; A41 withdrew the row `"step: new cells for the size choice"`)

- [ ] **Step 0: Per-thread test** (amended 2026-11-14, Foundation as built; `step_stages_steps_test.cc`, `LABEL cpu`,
  `LINKS fluidity_base_test_support fluidity_step_test_support`): `AddNewParticlesSteps,
  EveryThreadBothOrders_EqualCpuLoop` (Task 12's sources case: the functions of `add_new_particles_steps.h` run for
  every thread block and thread of `ComputeLaunchSize(child count)`, in launch order and in reverse, give the CPU
  loop's positions and velocities bit for bit: same function, same processor) and `ChooseActiveCellsSteps,
  EveryThreadBothOrders_EqualCpuLoop` (`BuildFallingBlockScene(1)`: block ids, unique flags, the cell list and the
  applied levels equal the CPU stage's needed cells, phase by phase). These are the main evidence until the hardware
  run.
- [ ] **Step 1: Tests.** `LABEL gpu`: `AddNewParticles, OnGpu_EqualsCpu` (Task 12's sources case: ids exact,
  positions within 6.4 with `CheckWithinTolerance`), `AddNewParticles, OnGpuTwice_IdenticalBytes`,
  `ChooseActiveCells, OnGpu_EqualsCpu`
  (no needed block or cell id comes back under A17, so the test compares through two test-side reads: the try grid
  copied with `grid::ReadGrid(try_grid, cpu_buffers)` into test-owned CPU `GridBuffers` equals the CPU run's try
  grid table by table, byte for byte; and the entry count in the stage's status record equals the CPU run's needed
  cell count; the test's `ReadGrid` is not a stage copy and is not in `StepReport::copies`),
  `ChooseActiveCells, OnGpuTwice_IdenticalBytes` (the same two reads, identical between the runs),
  `AddNewParticles, OnGpuSourceCopiesCounted` (A17 M1: `StepReport::copies.bytes_to_processor` == the
  `AddParticlesResult::bytes_copied` of the call, > 0), `ChooseActiveCells, OnGpu_NoCellIdComesBack` (A17 M6:
  the stage adds no copy; the grid's version and block count come from `grid.counts()`),
  `ChooseActiveCells, OnGpuNewCellLevels_EqualCpuNoCopy` (A41; replaces `OnGpuSizeChoiceCopiesCounted`: Task 13's
  `NewCellBesideBigCell_OneLevelFiner` on the GPU: the try grid read by a test-side `grid::ReadGrid` equals the CPU
  run's byte for byte, stage 2 adds `{0, 0, 0}` to `StepReport::copies`). Every `gpu` test calls
  `WriteResultDigest` on its main outputs (new particles' ids, positions, velocities; the try grid's tables read with
  `grid::ReadGrid`). `LABEL no_gpu`:
  `AddNewParticles, OnGpuWithoutAGpu_IsRunOnCallFailed`, `ChooseActiveCells, OnGpuWithoutAGpu_IsRunOnCallFailed`
  (each calls the stage with `RunOn::kGpu`; its `_cpu.cc` half runs `CheckCanRunOn(RunOn::kGpu)` before the GPU
  half). `LABEL cpu`:
  `BuildStepMemoryEstimate, GpuRunHasActiveCellListOnGpu` (a `RunOn::kGpu` plan, no GPU needed: the row exists with
  `where = kRunOn`; absent in a CPU plan).
- [ ] **Step 2: Implement.** Sources: one thread per child, positions from `ComputeRandomWords` (the shared
  header); the new particles go to `particles::AddParticles` as on the CPU, whose copy fields the stage adds to
  `StepReport::copies`. Each GPU function is a thin grid-stride loop (`ComputeGridStride`) over a function of the
  matching `_steps.h`, started with `ApplyGpuFunction(function, ComputeLaunchSize(count), ...)`; per-call
  temporary arrays only through `GpuArray`; the half ends with `CheckGpuWork` before the count is read. Marking
  (design 5.3 last paragraph, A17): one pass writes a block id per particle;
  `BuildStableOrder(keys, count, nullptr, order, try_arrays.order_buffers)` (no allocation) and a
  unique pass give the needed blocks; one pass writes the needed cells into the `"step:
  active cell list"` buffer; only the entry count comes back, in the stage's status record; `grid::BuildGrid(
  GridInputs{frame, {}, &list, count, version + 1}, free_storage)` builds on the GPU (plan-decisions "Grid built on
  the GPU"). A41: the pass that writes the needed cells calls `grid::ComputeNewCellSizeLevel(committed GPU tables,
  cell centre, new_cell_size_level)` for each cell not in the committed grid (the shared header, per thread), so
  no level crosses. Replaced (A41): "With `choose_cell_size` set (cell resize): the needed cells not in the
  committed grid are compacted and read back (one copy, 8 B each), the callback runs per cell on the CPU, the levels
  go back (one copy, 1 B each) and a GPU pass applies them before the build; both copies are counted (Appendix B row
  O4)", and the paragraph "What the step passes (reconcile item 13)".
- [ ] **Step 3: Run** the gate. **Commit.** Subject: `Add particles and choose cells on the GPU`. Body:
  `CUDA: compiled, not run`; the per-thread `cpu` tests are the evidence.

### Task 15: Stages 3 (skipped form), 4, 5 and 6 for liquid

**Amended 2026-11-15 (A64):** no `apply_grid_only_region` call; its test moves to Task 39a (later).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/stage_functions.h` (private), `transfer_stages.cc`, `apply_forces.h`,
  `apply_forces_cpu.cc`, `liquid_stages_test.cc`
- Modify: `stage_table.cc` (entries 4, 5, 6), `choose_active_cells_cpu.cc`

**Interfaces:**
- Produces: design 5.2 items 4 (without materials' internal forces and friction for sets that have none; the
  calls are made for every set, they are no-ops for liquid), 5, and 6 (gravity on every set; internal force in
  Task 28). (Replaced, A64: "`apply_grid_only_region` is called at the end of stage 4 when set.")
- **Plan decision:** when stage 3 is skipped (design 5.2 table: no solid but the domain box and no blocking
  material), stage 2 fills `solids_at_start` and `solids_at_end` with `solids::ComputeSolidsAtFaces` once per
  new grid version (the domain box alone), so later stages always read filled arrays.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `BuildFallingBlockScene(1)` unless named):

  ```cpp
  TEST(TryStep, LiquidThroughApplyForces_Commits)     // stages_run {1, 2, 4, 5, 6, 19, 20}
  TEST(ApplyForces, GravityAddedOnceExactly)          // each kFromParticles face: after == before_forces +
                                                      // (0.01 * -9.81) on Z faces, bit for bit; X and Y unchanged
  TEST(ApplyForces, FacesOfOtherKinds_Untouched)
  TEST(FindSolidSurfaces, SkippedScene_SolidsAtFacesEqualComputed)  // the stage-2 fill equals a direct call
  TEST(TryStepWithStages, RejectAfterParticlesToGrid_ChangesNothing)          // four reasons each
  TEST(TryStepWithStages, RejectAfterSaveGridBeforeForces_ChangesNothing)
  TEST(TryStepWithStages, RejectAfterApplyForces_ChangesNothing)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** with `ComputeVelocityAfterGravity`. Transfer calls take the
  simulation's `TransferBuffers` as their last argument.
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Write liquid onto the grid and apply gravity in the step`.

### Task 16: `ApplyForces` on the GPU

**Amended 2026-11-14 (Foundation as built):** private `apply_forces_gpu.h` returning `RunOnCallStatus`; the GPU
function a thin loop over `ComputeVelocityAfterGravity` (Task 4, already shared); a `cpu` test over every thread
block and thread; the bit-equal GPU test rests on that header's same-bits promise; digests.

**Kind:** device

**Files:**
- Create: `native/fluidity/step/apply_forces_gpu.cu`, `apply_forces_gpu.h` (private)
- Modify: `native/fluidity/step/CMakeLists.txt` (`fluidity_add_gpu_sources`), `apply_forces_cpu.cc` (dispatch: the
  GPU half under `#if defined(FLUIDITY_HAS_GPU)`, status through `ToStageOutcome(ToBufferError(status))`),
  `step_stages_steps_test.cc`, `step_stages_gpu_test.cc`, `step_stages_no_gpu_test.cc`

- [ ] **Step 1: Tests.** `cpu` (`step_stages_steps_test.cc`): `ApplyForcesSteps, EveryThreadBothOrders_EqualCpuLoop`
  (`BuildFallingBlockScene(1)` after stage 5: the per-face function the GPU function calls, run for every thread
  block and thread of `ComputeLaunchSize(face places)` in launch order and in reverse, gives the CPU stage's faces
  bit for bit; the main evidence until the hardware run). `gpu`: `ApplyForces, OnGpu_EqualsCpuBitForBit` (the
  product is formed once on both; `ComputeVelocityAfterGravity` promises the same bits unless NaN, so the faces are
  compared with `CheckSameBitsOrBothNaN`), `ApplyForces, OnGpuTwice_IdenticalBytes`, both writing
  `WriteResultDigest` of the faces; `no_gpu`: `ApplyForces, OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement** one `ApplyGpuFunction` launch per set and range over face places (`ComputeLaunchSize`,
  grid-stride loop), then `CheckGpuWork`. **Step 3:** gate. **Commit.**
  Subject: `Apply gravity on the GPU`. Body: `CUDA: compiled, not run`.

### Task 17: Stages 9 and 10 for liquid

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/pressure_stages.cc`
- Modify: `stage_functions.h`, `stage_table.cc` (entries 9, 10), `liquid_stages_test.cc`

**Interfaces:**
- Produces: design 5.2 items 9 (all-liquid distance; one per liquid material that is in a `MaterialPair` with
  another liquid, then `FindMaterialBoundaries`; `FindLiquidSurface`; `FindAirPockets` into
  `fields.pocket_of_cell` and the try pocket storage; `viscosity::CheckViscosityCanRun` on this try's summary)
  and 10 (carried pressure when a law can be squeezed; mixture density and sensed volume ratio; cell squeeze
  values; volume correction; `solids::BuildMotionMap` with `ToFreeSolidPushes` and
  `ToFreeSolidInverseInertia`; `BuildPressureEquations` with `SealedBlobCheck::kWhenBuilt`). Gas inputs are
  the "no gas" arrays (all zero) until Task 29.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, LiquidThroughEquations_Commits)        // stages_run adds 9 and 10
  TEST(FindLiquidSurface, StageUsesStartPositions)     // geometry equals a direct call on the committed positions
  TEST(BuildPressureEquations, MixtureDensityFromLiquidSet)  // BuildTwoLiquidsScene: a shared cell's density is the
                                                             // volume-weighted mix, 1e-15 relative
  TEST(TryStepWithStages, RejectAfterFindLiquidSurface_ChangesNothing)   // air pockets and next_blob_id included
  TEST(TryStepWithStages, RejectAfterBuildPressureEquations_ChangesNothing)
  TEST(FindLiquidSurface, StepTooLongForSurfaceTension_ShortensToItsLimit)  // drop of 4 cells, sigma 0.0728,
      // d 0.01: kStepTooLongForSurfaceTension in kFindLiquidSurface, shorter_step_duration == context.limit
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** `kCellSplitBySolid` arrives as `kNotSupportedNow` /
  `kThinSolidInsideOneCell` through Task 3's mapping. **Step 4:** gate. **Commit.** Subject:
  `Find the liquid surface and build the pressure equations in the step`.

### Task 18: Stages 12, 14 and 15 for liquid

**Amended 2026-11-14 (A28):** `SolvePressure` receives the step inputs' `cancel_requested`.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/pressure_stages.cc`, `transfer_stages.cc`, `stage_table.cc` (entries 12, 14, 15),
  `liquid_stages_test.cc`

**Interfaces:**
- Produces: design 5.2 items 12 (`SolvePressure` into `fields.pressure`, `fields.velocity_after_pressure`, with
  the step inputs' `cancel_requested`; `ComputeWasSolved` into `L.was_solved` and `fields.has_pressure`;
  `solids::ApplyPressureToSolids`, whose `kSolidMovesTooFar` is `kSolidTravelledTooFar`), 14 (`ComputeSolidsAtFaces` at the end pose, then `SaveGridAfterPressure`
  per set) and 15 (`GridToParticles` per set into the step-owned `GridAtParticle` buffers).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, LiquidThroughGridToParticles_Commits)  // stages_run adds 12, 14, 15; committed pressure has a value
  TEST(SolvePressure, ColumnOneStep_HydrostaticPressure)  // BuildColumnAtRestScene: p == 1000 * 9.81 * (0.43 - z)
                                                          // at full cells within pressure's solve limit
  TEST(SolvePressure, CancelSetDuringSolve_IsCancelRequested)  // design 7 "Cancel" during 12: a thread sets the flag
      // when the solve's first progress read returns (hand-written wait on an atomic counter in the test);
      // committed snapshot unchanged; the next TryStep commits
  TEST(TryStepWithStages, RejectAfterSolvePressure_ChangesNothing)  // GridFields bytes unchanged
  TEST(TryStepWithStages, RejectAfterSaveGridAfterPressure_ChangesNothing)
  TEST(TryStepWithStages, RejectAfterGridToParticles_ChangesNothing)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Solve the liquid pressure and read the grid back in the step`.

### Task 19: `MoveParticles` on the CPU

**Amended 2026-11-15 (A64):** no `move_grid_only_liquid` call; its test moves to Task 39a (later).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/move_particles.h`, `move_particles_cpu.cc`, `move_particles_test.cc`
- Modify: `stage_table.cc` (entry 16)

**Interfaces:**
- Produces: design 5.5 steps 1 to 6 (replaced, A64: "and the `move_grid_only_liquid` call at the end of the
  stage"). Mixing is
  `transfer::MixParticleVelocity` into `velocity.for_next_step` and `velocity.for_this_step` (A1);
  `contact::MoveParticles` with the simulation's `ContactBuffers`; travel check with
  `ComputeTravelInCellWidths`; `materials::ApplyMaterialResponse`; removal through open sides with
  `FindDomainSideCrossed` and `RemoveReason::kLeftDomain`, amounts into `StepReport::removed_through_open_sides`.
  Contact `kTooCloseToCall` is `kContactTooCloseToCall`; `kStartsInsideSolid`, `kOnWrongSideOfPlate` are
  `kStageFailed`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, FreeFall_VelocityIsMinusGTimesT)  // design 7 "Free fall", three tests (own_motion_kept 0, 0.5, 1):
      // BuildFallingBlockScene, 20 steps of 0.005 s: every velocity.for_next_step.z == -9.81 t, each particle's z
      // dropped by the sum of v * step_duration; 1e-12 relative. (Parameterised over the three values.)
  TEST(MoveParticles, TravelTooFar_RejectsWithIdAndHalfStep)  // design 7 "Travel reject": one particle at 0.6
      // cell per step: kParticleTravelledTooFar, its id, shorter_step_duration d/2; at d/2 the step commits
  TEST(MoveParticles, TwoTooFarInTwoRanges_LowestIdReported)  // Review Focus 4: a stage table whose entry 16
      // calls the real stage once per one-block range; ids 9 and 4 in two blocks: 4 reported, as with one range
  TEST(MoveParticles, LeavesThroughOpenTop_RemovedWithItsAmounts)  // design 7 "Open side": mass, volume, momentum
      // in removed_through_open_sides equal the particle's exactly
  TEST(MoveParticles, ClosedTop_ParticleStays)
  TEST(TryStepWithStages, RejectAfterMoveParticles_ChangesNothing)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `main` `liquid_substep_3d_cpu.cc:239-398` (the mix,
  contact and travel parts; the `0.99` blend and the whole-cell travel field are not ported, design 10).
- [ ] **Step 4:** gate. **Commit.** Subject: `Move particles at the end of a step try`.

### Task 20: `MoveParticles`, `CheckStep` and `CommitStep` on the GPU

**Amended 2026-11-14 (A31):** `CheckStep`'s reductions use the one form of each sum, with `TryArrays::sum_buffers`.

**Amended 2026-11-14 (Foundation as built):** private `_gpu.h` headers returning `RunOnCallStatus`, the fused pass
and the failure key in `move_particles_steps.h` with a `cpu` test over every thread block and thread, `CheckStep`'s
reductions with `TryArrays::sum_buffers` (one form; A31), digests in the `gpu` tests.

**Kind:** device

**Files:**
- Create: `native/fluidity/step/move_particles_gpu.cu`, `move_particles_gpu.h` (private), `move_particles_steps.h`
  (shared: the fused mix, travel check and open-side mark of one particle, and its 64-bit failure key
  `(kind, particle id)`), `check_step_gpu.cu`, `check_step_gpu.h` (private), `commit_step_gpu.cu`,
  `commit_step_gpu.h` (private)
- Modify: `native/fluidity/step/CMakeLists.txt`, the three `_cpu.cc` files (dispatch: the GPU half under
  `#if defined(FLUIDITY_HAS_GPU)`, status through `ToStageOutcome(ToBufferError(status))`),
  `step_stages_steps_test.cc`, `step_stages_gpu_test.cc`, `step_stages_no_gpu_test.cc`

- [ ] **Step 1: Tests.** `cpu` (`step_stages_steps_test.cc`): `MoveParticlesSteps, EveryThreadBothOrders_EqualCpuLoop`
  (the free-fall scene and Task 19's two-too-far table: the per-particle functions run for every thread block and
  thread of `ComputeLaunchSize(particle count)` in launch order and in reverse give the CPU stage's velocities,
  positions, marks bit for bit, and the minimum failure key names particle 4; the main evidence until the hardware
  run). `gpu`: `MoveParticles, OnGpu_EqualsCpu`, `CheckStep, OnGpu_EqualsCpu`,
  `CommitStep, OnGpu_EqualsCpu` (the free-fall scene, one step: ids, flags and the reported failure equal;
  positions and velocities within 6.4 tolerances with `CheckWithinTolerance`; `CheckStep`'s sums with
  `CheckSameBitsOrBothNaN` only where the summed buffer itself is equal on both, else 6.4), each
  `..., OnGpuTwice_IdenticalBytes`; each writes `WriteResultDigest` of positions, velocities and the committed
  `GridFields`. `no_gpu`: the three `OnGpuWithoutAGpu_IsRunOnCallFailed`.
- [ ] **Step 2: Implement.** Mix, travel check and open-side marking are one fused pass per material and range
  (design 5.5), a thin `ApplyGpuFunction` loop over `move_particles_steps.h`; the travel failure is chosen by an
  integer minimum (64-bit integer `atomicMin`) over `(kind, particle id)`; `CheckStep` reduces with
  `ComputeLargestMagnitude(values, count, try_arrays.sum_buffers)` and `ComputeTreeSum(values, count,
  try_arrays.sum_buffers)`; `CommitStep` waits once (`CheckGpuWork`) for all work of the try before the swaps
  (design 5.6).
- [ ] **Step 3:** gate. **Commit.** Subject: `Move, check and commit on the GPU`. Body: `CUDA: compiled, not run`;
  the per-thread `cpu` tests are the evidence.

### Task 21: `CheckStep` in full, two liquids, and the liquid scene

**Amended 2026-11-15 (A62):** the next `CommittedStepLimits` carry `largest_liquid_speed`, the largest
`|velocity.for_next_step|` of liquid particles (`ComputeLargestMagnitude` with `TryArrays::sum_buffers`, in the
travel pass; exact in any order), kept by the commit. Test `CheckStep, LargestLiquidSpeed_ExactMaxOverLiquidParticles`.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/check_step_cpu.cc`, `try_step.cc`, `liquid_stages_test.cc`

**Interfaces:**
- Produces: design 5.6 `CheckStep` in full (`particles::CheckParticles` on every range, finite `GridFields`,
  the solids' end poses through `contact::CheckSolid`, `solids::CheckSolidExchange`, the next
  `CommittedStepLimits` with the smallest `ComputeTravelStepLimit` per particle and its own cell width) and the
  "which failure is reported" rule in `TryStepWithStages`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(CheckStep, NaNVelocity_IsValueNotFinite)            // a stage table writes NaN in one face of fields
  TEST(CheckStep, NextLimits_SmallestTravelLimitOfParticles)  // closed form over the falling block, 1e-15
  TEST(CheckStep, LargestLiquidSpeed_ExactMaxOverLiquidParticles)  // A62: BuildTwoLiquidsScene, one step of 0.01 s; a
      // stage table sets one oil particle's velocity.for_next_step to {0, 0.7, 0} at the end of stage 16: committed
      // largest_liquid_speed == 0.7 bit for bit, for one range and for one range per block; a scene without liquid
      // particles (BuildEmptyScene): 0
  TEST(TryStepWithStages, TwoFailuresOneStage_HighestKindThenLowestId)  // a table stage reports a shorten and a
      // raise-limits failure: kRaiseLimits reported
  TEST(TryStep, TwoLiquidsThatDoNotMix_StayLayered)        // BuildTwoLiquidsScene (pressure supports the sharp
      // boundary, plan-decisions): oil 0.1 m over water 0.2 m at rest, 50 steps: every oil particle stays above
      // the water's highest particle minus one cell width; largest speed below pressure's solve limit times the
      // cell width; CheckSceneCanRun gives no finding
  TEST(BuildSimulationFromSnapshot, ContinuedLiquidRun_SameBytes)  // BuildColumnAtRestScene with a squeezable law:
      // full snapshot at step 20, rebuild, 20 more steps, against 40 uninterrupted: ExpectSameBytes
  TEST(TryStep, LiquidStageTrace)                          // {1,2,4,5,6,9,10,12,14,15,16,19,20} (the full "Stage
                                                           // trace" row set is Task 31's)
  TEST(TryStepWithStages, RejectAfterCheckStepWithLiquid_ChangesNothing)  // design 5.6 list, every committed array
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Check a liquid step try before it commits`.

### Task 22: `ComputeStepDuration`

**Amended 2026-11-15 (A64):** `kParticleTravel` is `CommittedStepLimits::particle_travel` alone; the test
`GridOnlySpeed_IsActive` moves to Task 39a (later).

**Kind:** numerics

**Files:**
- Create: `native/fluidity/step/step_duration.h`, `step_duration.cc`, `step_duration_test.cc`

**Interfaces:**
- Produces design 3.8 exactly (`StepLimit`, `kStepLimitCount`, `ChosenStepDuration`, `ComputeStepDuration`).
  `kSolidTurn` = `contact::kMaxTurnPerStep` over the largest committed angular speed of a solid.

- [ ] **Step 1: Failing tests** (`LABEL cpu`), design 7 "Even steps" and the "Each limit" rows that need no
  later stage (`kMaterialWaveSpeed` is in Task 28, `kGas` in Task 29, `kAfterReject` in Task 23):

  ```cpp
  TEST(ComputeStepDuration, EvenSteps_FourOf0_00875)   // max 0.01, target 0.035: 4 steps, 0.00875, kEvenStepsToTarget
  TEST(ComputeStepDuration, MaxStep_IsActive)          // target 1.0: 0.01, kMaxStepDuration
  TEST(ComputeStepDuration, ParticleTravel_IsActive)   // committed particle_travel 0.004: 0.004 (1e-15)
  TEST(ComputeStepDuration, SolidMotion_IsActive)      // a driven box whose next key is 0.003 s away
  TEST(ComputeStepDuration, SolidTurn_IsActive)        // angular speed 100 rad/s: kMaxTurnPerStep / 100
  TEST(ComputeStepDuration, SurfaceTension_IsActive)   // committed surface_tension 0.002
  TEST(ComputeStepDuration, AirPocket_IsActive)        // committed air_pocket 0.003
  TEST(ComputeStepDuration, SourceWindow_IsActive)     // a source starting 0.002 s after time
  TEST(ComputeStepDuration, InfiniteLimits_DoNotApply) // limit_by_kind holds +infinity where none applies
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.8 (ported from `main` `liquid_substep_3d.cc:2823-2863`,
  with `// Step N:` of `<proc-compute-step-duration>`). **Step 4:** gate. **Commit.** Subject:
  `Choose the step duration and say which limit set it`.

### Task 23: `AdvanceTo` and the callbacks

**Amended 2026-11-14 (A28):** `RejectKind::kBadRequest` is `kBadStepInputs`.

**Amended 2026-11-15 (A64, A62):** no `read_largest_grid_only_speed`; `ReadLargestGridOnlySpeed_StoredAtCommit`
moves to Task 39a (later) and is replaced here by `AdvanceTo, LargestLiquidSpeedKeptAcrossSteps`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/advance_to.cc`, `advance_to_test.cc`
- Modify: `native/fluidity/step/advance_to.h` (`StopReason`, `AdvanceResult`, `AdvanceTo`), `commit_step_cpu.cc`

**Interfaces:**
- Produces design 3.9 in full. Loop (`<proc-advance-to>`): 1 cancel check; 2 `ComputeStepDuration`;
  3 `TryStep`; 4 `on_try_done`; 5 by `RejectKind`: shorten (at most `max_retries_per_step`, then
  `kRetriesUsedUp`), raise limits (`ApplyLimits` from Task 24, at most `max_limit_raises_per_step`),
  cancelled, bad step inputs, cannot continue; 6 after a commit: `resize_cells` (stage 21 recorded), then
  `on_step_done`; 7 stop at `target_time`. The last step marked by the loop snaps `time` to `target_time` when
  the sum is one rounding step away (ported, `main` `liquid_state_3d.cc:757-762`).
  (Replaced, A64: "`read_largest_grid_only_speed` is read in `CommitStep` into
  `CommittedStepLimits::grid_only_speed`.") Until Task 24, a `kRaiseLimits` reject stops with `kLimitsCannotBeRaised`.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(AdvanceTo, ColumnAtRest_StaysAtRestAndEndsOnTarget)  // design 7 "Column at rest": 100 steps (target 1.0,
      // max 0.01): largest speed below pressure's solve limit times the cell width; time() == 1.0 exactly
  TEST(AdvanceTo, LastStepOneRoundingShort_EndsOnTarget)   // Review Focus 2: target 0.3 reached by 0.1 steps
  TEST(AdvanceTo, ShorterRetry_SameBytesAsTwoHalfSteps)    // design 7 "Shorter retry": reject at d (table), then
      // two commits at d/2, against two commits at d/2 from the same state: ExpectSameBytes
  TEST(ComputeStepDuration, AfterReject_IsActive)          // the retry's chosen duration: kAfterReject
  TEST(AdvanceTo, ThreeRejects_RetriesUsedUp)
  TEST(AdvanceTo, Callbacks_EachGetsItsOwnContext)         // design 7 "Callbacks": resize_cells, make_foam_and_spray,
      // on_step_done with three CallbackLog contexts: each function sees its own; order on_try_done, resize_cells,
      // on_step_done; ResizeResult{resized false} keeps the step
  TEST(AdvanceTo, OnStepDoneStop_StopsAtThatCommit)        // kStoppedByCallback
  TEST(AdvanceTo, CancelFlag_StopsCancelled)
  TEST(AdvanceTo, LargestLiquidSpeedKeptAcrossSteps)       // A62: BuildFallingBlockScene(1), AdvanceTo 0.05 s:
      // after each commit (on_step_done) CommittedStepLimits::largest_liquid_speed == the largest |for_next_step| of
      // the committed particles read by a snapshot, bit for bit; a rejected try leaves it unchanged
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Advance a simulation to a target time with retries`. New names: `AdvanceTo`, `StopReason`, `AdvanceResult`.

### Task 24: `ApplyLimits`

**Amended 2026-11-14 (A31):** working buffers instead of "scratch"; the old builder buffers are freed with the old storages; test `StepScratchRaised` renamed `StepWorkingBuffersRaised`.

**Amended 2026-11-14 (reconcile):** the shared grid builder buffers is rebuilt with the two storages.

**Amended 2026-11-14 (Foundation as built):** the step's `sum_buffers` and `order_buffers` are rebuilt at the new
limits with the other try arrays (each module rebuilds its own working buffers inside its buffers); a sum over more values
than the old `sum_buffers` hold would be `kSizeMismatch`, which this prevents.

**Amended 2026-11-15 (A58, with the same-day correction):** a raise reserves new blocks for the whole new plan,
builds every buffer into them, copies the committed values, swaps, then frees the old blocks whole; the check is old
+ new reserved <= allowed (tighter than before; design 5.7).

**Amended 2026-11-15 (reconcile):** `StepCallbacks::limits_raised` (added to `advance_to.h` here) lets I/O rebuild the
buffers it built into the blocks before the old blocks are freed; test `AdvanceTo,
LimitRaise_CallsLimitsRaisedBeforeFreeingOldBlock`.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/apply_limits.cc`, `apply_limits_test.cc`
- Modify: `native/fluidity/step/simulation.h` (declaration), `advance_to.cc`

**Interfaces:**
- Produces `ApplyLimits` (design 3.6, 5.7 steps 1 to 5): check the new estimate against `allowed_bytes`; a new
  `grid::GridBuilderBuffers` and two new grid storages lent it, at the new `max_cell_blocks` (reconcile item 11; the old
  builder buffers are freed with the old storages); rebuild the committed grid with the same `GridInputs` (same
  version); `grid::Build...Values` and `ApplyGridVersion`; slice `CopyBuffer` of the committed values;
  `particles::ApplyParticleLimits(particles, max_particle_counts, max_cell_blocks)`; every module's buffers at
  the new sizes (their working buffers with them) and the step's `sum_buffers` and `order_buffers` at Task 8's new counts;
  swap; free the old ones. A limit below the current one, or all equal, is `kLimitNotRaised`.
- **Amended 2026-11-15 (A58).** Step 1 checks `ComputeRunMemoryBytes(new entries, counts, kRunOn) +
  old reserved_bytes <= allowed_bytes` and, in a GPU run, the same for the CPU blocks against `allowed_cpu_bytes`
  (replaced: "check the new estimate against `allowed_bytes`"); step 2 reads free memory, `BuildRunMemory` for the
  new block(s), reads free memory again (`memory_taken_bytes` of the new block), builds every buffer into them with
  the private builders of Task 9 (each block last); after the swap the old buffers are dropped (they free nothing)
  and the old blocks freed whole. `reserved_block_count += 1`; `requested_beyond_plan_bytes = max(it, old + new
  reserved - planned_bytes)` (and the CPU blocks likewise). A failure at 1 or 2 frees any new block and leaves the
  simulation unchanged. The buffers I/O built into the old blocks with `FindRunMemory` (render output, cell resize)
  follow the raise through the new `StepCallbacks::limits_raised` (reconcile, 2026-11-15; design 3.9): `ApplyLimits`
  calls it after the new blocks are reserved and the step's own buffers rebuilt, before the old blocks are freed;
  `false` frees the new blocks and fails `kDoesNotFit` (simulation as it was), and `AdvanceTo` stops with
  `kLimitsCannotBeRaised`; null means nothing outside the step holds run buffers. `advance_to.h` declares it.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(ApplyLimits, RaisedRun_SameBytesAsLargeLimits)  // Review Focus 5 and design 7 "Limits": source scene with
      // start limits of half the need (particles and blocks), AdvanceTo 0.5 s, against a run with large limits:
      // ExpectSameBytes; limit_raise_count > 0; rejected tries did not advance time
  TEST(ApplyLimits, LowerLimit_IsLimitNotRaised)       // simulation unchanged and usable
  TEST(ApplyLimits, DoesNotFit_IsDoesNotFitAndUnchanged)  // allowed bytes from a plan with tiny MemoryAvailable
  TEST(ApplyLimits, ReportAfterRaise_EqualsNewEstimate)   // amended 2026-11-15 (A58): handed_out_bytes ==
      // reserved_bytes == new estimate.on_processor; planned_bytes unchanged (was "allocated_bytes == new
      // estimate.on_processor")
  TEST(ApplyLimits, OldPlusNewAboveAllowed_IsDoesNotFit)  // A58: allowed_bytes set (through the plan's
      // MemoryAvailable) between the new block alone and old + new: kDoesNotFit, simulation unchanged and usable,
      // reserved_block_count still 1
  TEST(AdvanceTo, LimitRaise_CallsLimitsRaisedBeforeFreeingOldBlock)  // reconcile 2026-11-15: the raise scene with one
      // 1000 B kRunOn output_estimate row and a hand-written limits_raised that builds a 1000 B piece into
      // FindRunMemory and logs: called once with the new limits, after the new block is reserved and before the old
      // one is freed (operator new/delete log), handed_out_bytes == reserved_bytes after; returning false: AdvanceTo
      // stops kLimitsCannotBeRaised, reserved_block_count 1, snapshot ExpectSameBytes as before
  TEST(AdvanceTo, LimitRaise_ReservesSecondBlockAndFreesFirst)  // A58: source scene with half the particle limit,
      // a replaced operator new/delete counting calls: after the raise reserved_block_count 2, reserved_bytes == the
      // new plan's on_processor, requested_beyond_plan_bytes == old + new - planned (= the new block's bytes), and
      // the old block is freed once, after the new block was reserved
  TEST(ApplyLimits, StepWorkingBuffersRaised)             // amended (Foundation as built): sum_buffers.max_count()
                                                          // and order_buffers.max_count() equal Task 8's new counts
  TEST(AdvanceTo, LimitsNeeded_RaisedThenSameStepRepeated) // same duration, same state_version
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Raise the limits of a simulation between steps`.

### Task 25: Viscous liquid, stage 11

**Amended 2026-11-14 (reconcile):** an OPEN note marked where option (C) of the springy-solid-beside-viscous-liquid
decision would land. Replaced by the A24 amendment below.

**Amended 2026-11-14 (A24):** the owner chose option (C): stage 11 carries the blocking-material faces and the
sealed gas blobs, and calls `solids::ApplyPressureToMaterials`, exactly as stage 12.

**Amended 2026-11-14 (A25):** stage 11 runs exactly when a liquid ingredient's viscosity is above 0, fixed for the
run; tests `CheckSceneCanRun, ViscosityZero_IsInviscid` and `TryStep, ViscousPathFixedWhenStepShortens`.

**Amended 2026-11-14 (A26):** the liquid sticks to blocking materials: stage 11 passes the band material faces
(`ViscousMaterialFace`) as unknowns and calls `solids::ApplyViscousSolveToMaterials` in place of
`ApplyPressureToMaterials`.

**Amended 2026-11-14 (A27):** free solids touching the liquid are unknowns of stage 11: it passes
`ToFreeSolidsForViscousSolve(motion_map)` and calls `solids::ApplySolvedMotionToSolids` in place of
`ApplyPressureToSolids`.

**Amended 2026-11-15 (A57):** `kSmallestVisibleViscousEffectPerStep` no longer exists; `TryStep,
ViscousPathFixedWhenStepShortens` keeps its 1e-6 Pa s liquid without the threshold wording.

**Amended 2026-11-15 (A60):** stage 11 hands `ViscosityResult`'s per-solid viscous impulse to the solids try
(`solids::ApplyViscousImpulses`, name as the solids plan gives it) right after `SolveViscosityAndPressure`.
Exact call (solids plan Task 23d): `solids::ApplyViscousImpulses(viscosity_result.viscous_push_on_solids,
solids_try)` with `std::span<const solids::ViscousPushOnSolid>`, after `ApplySolvedMotionToSolids`.

**Amended 2026-11-15 (A53, A58):** stage 11 passes `solids_try.slip_length()` as `ViscosityInputs::solid_slip_length`;
`BuildSimulation` builds the viscosity buffers with `BuildViscosityBuffers(settings, run_memory, cpu_run_memory)`
and sets `ViscosityBufferSettings::max_solids` to the scene's solid count.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/viscosity_stage.cc`, `viscous_stages_test.cc`
- Modify: `stage_functions.h`, `stage_table.cc` (entry 11), `step_test_scenes.h/.cc` (`BuildSyrupColumnScene`:
  closed box 0.1 x 0.1 x 0.1, syrup 10 Pa s, 1400 kg/m3, to 0.06 m, cells 0.01)

**Interfaces:**
- Produces design 5.2 item 11 (`SolveViscosityAndPressure` with `cells_from_particles` = `{&L.cells_from_particles}`
  and `ViscosityInputs::pressure_buffers` = the liquid `PressureBuffers` that stage 10 built the equations with
  (`plan-decisions.md`, "Viscosity"; null is viscosity's `kBadInput`), then `ComputeWasSolved`, then
  `ApplySolvedMotionToSolids` (A27; replaced: `ApplyPressureToSolids`)); stage 12 is skipped when 11 runs. **Plan decision:** the stage builds its inputs in
  the private `ViscosityInputs BuildViscosityStageInputs(Simulation&, StepTry&)` (declared in
  `stage_functions.h`), so the test below reads exactly what the stage passes.
- **Stage 11 carries blocking materials and sealed gas (A24, `plan-decisions.md` "A24: step, gas, solids").**
  Stage 11 calls `viscosity::SolveViscosityAndPressure(inputs, viscosity, cancel_requested, viscosity_buffers,
  fields.velocity_after_pressure, fields.pressure, viscosity::SolvedCouplingArrays{
  blocking_material_velocity_after_pressure, sealed_gas_pressure})` with the two step-owned spans of Global
  Constraints (A18), sized to the equations' `blocking_material_face_count` and `sealed_gas_blob_count` (the
  equations stage 10 built hold the material faces and the sealed blob list whatever stage follows); then
  `pressure::ComputeWasSolved`, `solids::ApplyPressureToSolids`, then, in a scene with a blocking set,
  `solids::ApplyPressureToMaterials(faces, blocking_material_velocity_after_pressure, {set S, set E},
  solids_buffers)` exactly as stage 12 (Task 28; its `PressureOnMaterials::push` per set into the report). Stage
  13 then passes `sealed_gas_pressure` to `SolveGas` (Task 30). The viscosity buffers hold `max_sealed_gas_blobs`
  (Task 9). Order of the build: here both spans are empty (the syrup column has no blocking set and no gas, so
  both counts are 0); Task 28 adds the material face row and the `ApplyPressureToMaterials` call to stages 11 and
  12 together, and Task 30 the sealed gas row to both. The blocking and gas cases run in Tasks 30, 31 and 35.
  **Replaced in part (A26, A27):** stage 11 no longer calls `ApplyPressureToSolids` nor `ApplyPressureToMaterials`;
  see the next bullet.
- **Stage 11 with stick and free solids (A26, A27; `plan-decisions.md` "A viscous liquid sticks, and free solids in
  the viscous solve", section "Step (stage 10, 11)").** In a viscous scene stage 10 also calls, after
  `BuildBlockingMaterialFaces` (scene with a blocking set), `solids::BuildViscousMaterialFaces(
  BlockingMaterialFacesInputs{...}, faces, geometry, solids_buffers)` giving `viscous_faces`, and
  `solids::ToFreeSolidsForViscousSolve(motion_map)` giving `free_solids`, before `BuildPressureEquations`, which
  receives `EquationInputs::free_solid_motion_before_pressure` filled from it (Global Constraints, A26/A27; empty
  for stage 12). Stage 11 calls
  `viscosity::SolveViscosityAndPressure(ViscosityInputs{..., material_faces =
  solids::ToViscousMaterialFaces(viscous_faces), free_solids}, viscosity, cancel_requested, viscosity_buffers,
  fields.velocity_after_pressure, fields.pressure, viscosity::SolvedCouplingArrays{
  blocking_material_velocity_after_pressure, sealed_gas_pressure, material_face_velocity_after_solve,
  free_solid_motion})`, the four step-owned spans sized to the equations' `blocking_material_face_count`, their
  `sealed_gas_blob_count`, `ToViscousMaterialFaces(viscous_faces).size()` and `free_solids.size()`; then
  `pressure::ComputeWasSolved(equations, L.was_solved, fields.has_pressure)`; then
  `solids::ApplySolvedMotionToSolids(motion_map, free_solid_motion, fields.pressure, solids_try, solids_buffers)`
  (in place of `ApplyPressureToSolids`; each free solid touching liquid takes its solved velocity and turn rate,
  the travel and turn limits are checked again, `kSolidMovesTooFar` is `kSolidTravelledTooFar`; push report
  `mass * (after - before)` with the pressure part kept separately); then, in a scene with a blocking set,
  `solids::ApplyViscousSolveToMaterials(viscous_faces, material_face_velocity_after_solve, {set S, set E},
  solids_buffers)` (in place of `ApplyPressureToMaterials`; each band face of each blocking set gets its solved
  velocity in its set's `GridFromParticles::velocity`, every other face of the set keeps its velocity after
  `ApplyForces` and `CollideSolids`; `PressureOnMaterials::push` per set into the report). The
  `blocking_material_velocity_after_pressure` span is still filled (the `BlockingMaterialFace` subset) but not
  written back by stage 11. Stage 12 is unchanged. Order of the build: on the syrup column both new spans are
  empty (no blocking set, no free solid) and `ApplySolvedMotionToSolids` runs on an empty list; Task 26 adds the
  `"step: free solid motion"` and `"step: free solid motion before pressure"` rows and the free-solid case, Task 28 `BuildViscousMaterialFaces`, the `"step:
  material face velocity after solve"` row and `ApplyViscousSolveToMaterials`.
- **Which stage runs (A25).** Stage 11 is in the stage table exactly when the scene is viscous by Task 7's rule
  (a liquid ingredient's viscosity above 0, no threshold); else stage 12 is. The table is fixed in
  `BuildSimulation` and never changes during the run, whatever the viscous effect per step (Task 8) of the liquid.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, SyrupStageTrace)                       // design 7 "Stage trace", syrup row: 11 in place of 12
  TEST(TryStep, SyrupAtRest_StaysAtRest)               // 20 steps: speeds below 1e-9 m/s (viscosity's "Syrup at rest")
  TEST(TryStep, ZeroViscosity_RunsSolvePressure)       // the same scene with viscosity 0: 12, not 11
  TEST(SolveViscosityAndPressure, CancelSetDuringSolve_IsCancelRequested)  // as Task 18, during 11
  TEST(TryStepWithStages, RejectAfterSolveViscosityAndPressure_ChangesNothing)
  TEST(SolveViscosityAndPressure, StagePassesTheBuffersThatBuiltTheEquations)  // after stages 1 to 10 (a stage
      // table with entries 11 to 20 null), BuildViscosityStageInputs(simulation, step_try).pressure_buffers equals
      // the address of the liquid PressureBuffers that stage 10 passed to BuildPressureEquations: not null, not
      // the gas instance (run on the syrup column with an ambient gas set added)
  TEST(CheckSceneCanRun, ViscosityZero_IsInviscid)  // A25: BuildColumnAtRestScene, water preset (viscosity 0): no
      // finding; BuildRunPlan's viscous_path kPressureSolve; the simulation's stage table has stage 12 and no
      // stage 11 (stages_run of one committed TryStep)
  TEST(TryStep, ViscousPathFixedWhenStepShortens)   // A25: BuildSyrupColumnScene with viscosity 1e-6 Pa s (a
      // viscosity too small to see; amended 2026-11-15, A57: the threshold wording is dropped): a forced reject
      // (BuildRejectingStageTable(kCheckStep, kParticleTravelledTooFar)) then two half steps: stage 11 in the
      // stages_run of every try, stage 12 in none; the same on the water preset (BuildColumnAtRestScene):
      // stage 12 in every try, stage 11 in none
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Run the joint viscosity and pressure solve in the step`.

### Task 26: Solids, stages 3, 8 and 17

**Amended 2026-11-14 (A18):** `ApplyPressureToMaterials` is no longer part of stage 17 (it runs in stage 12, Task 28).

**Amended 2026-11-14 (A27):** a free solid touching a viscous liquid is an unknown of stage 11: this task adds the
free-solid case of stage 10 (`ToFreeSolidsForViscousSolve`) and stage 11 (`free_solid_motion`,
`ApplySolvedMotionToSolids`), the step-owned row `"step: free solid motion"`, and the test
`TryStep, FreeBoxInViscousLiquid_SolvedMotionWrittenBack`.

**Amended 2026-11-15 (A60):** stage 20 copies `solids_try.liquid_forces()` into `StepReport::liquid_force_on_solids`
just before `solids::CommitTry` (every solid, ascending `solid_id`; empty on a reject). Tests `TryStep,
LiquidForceOnSolids_FromCommittedTry`, `TryStepWithStages, RejectAfterCheckStep_LiquidForcesEmpty`, `TryStep,
LiquidForceOnSolidsInSyrup_HasViscousPart`. The force values themselves are tested by the solids and viscosity
plans (sheared plate, buoyancy, rotating cylinders, settling sphere).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/solids_stages.cc`, `solids_stages_test.cc`
- Modify: `stage_functions.h`, `stage_table.cc` (entries 3, 8, 17), `commit_step_cpu.cc`, `check_step_cpu.cc`,
  `pressure_stages.cc` and `viscosity_stage.cc` (A27: stage 10's `free_solids`, stage 11's free-solid span and
  `ApplySolvedMotionToSolids`), `step_memory_estimate.cc` (A27: the free solid motion row),
  `choose_active_cells_cpu.cc` (the stage-2 fill of Task 15 now only when stage 3 is skipped),
  `step_test_scenes.h/.cc` (`BuildFreeBoxScene(double box_density)`: open box 0.8 x 0.8 x 0.8, water to 0.4, a
  free cube of side 0.2; `BuildDrivenBoxScene`: a driven box moving +X at 0.1 m/s through still water)

**Interfaces:**
- Produces design 5.2 items 3 (without `ApplyPlateSides`, Task 28), 8 (liquid set and the solids;
  `MaterialSetFaces` of every set but gas), 17 (`ApplyContactReactions` of materials but gas; A18: item 17 no
  longer calls `ApplyPressureToMaterials`, which Task 28 puts in stage 12), `solids::CommitTry` in stage 20 (its
  failure after another commit:
  `kStageFailed` in `kCommitStep`, simulation unusable), `solids::RejectTry` on reject.
- **Free solids in the viscous solve (A27; Task 25's stage 11 bullet).** In a viscous scene stage 10 gives
  `solids::ToFreeSolidsForViscousSolve(motion_map)` to `ViscosityInputs::free_solids` and, before
  `BuildPressureEquations`, fills the step-owned `free_solid_motion_before_pressure` from it (6 doubles per free
  solid, `V` then `Omega`, `FreeSolidPush::solid_index` order) for `EquationInputs::
  free_solid_motion_before_pressure` (empty in an inviscid scene); stage 11 passes the
  step-owned `free_solid_motion` span (one `solids::FreeSolidMotion` per entry) in `SolvedCouplingArrays` and then
  calls `solids::ApplySolvedMotionToSolids(motion_map, free_solid_motion, fields.pressure, solids_try,
  solids_buffers)` in place of `ApplyPressureToSolids`: each free solid touching liquid takes its solved velocity
  and turn rate; the travel and turn limits are checked again (`kSolidMovesTooFar` -> `kSolidTravelledTooFar`,
  shorten and retry); its push report is `mass * (after - before)` with the pressure part kept separately. A free
  solid not touching liquid keeps its motion before pressure, as in stage 12. Step-owned rows `"step: free solid
  motion"` and `"step: free solid motion before pressure"` (each given the scene's free solid count, 48 B, 1 copy,
  one step try).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, FreeBoxStageTrace)                         // 3, 8, 17 present
  TEST(TryStep, FreeBoxInViscousLiquid_SolvedMotionWrittenBack)  // A27: BuildFreeBoxScene(500) with the water's
      // viscosity set to 10 Pa s through scene_to_change(), the box placed touching the water: stage 11 runs (no 12);
      // free_solids has one entry; after
      // stages 1 to 11 (a stage table with entries 12 to 20 null) the box's velocity and turn rate in solids_try
      // equal free_solid_motion[0] (bytes); with the full table the step commits
  TEST(TryStep, FreeBoxInLiquid_MomentumChangeIsGravityImpulse)  // design 7 "Solid and liquid momentum": density 500
      // box dropped from 0.05 m above still water, 50 steps: liquid + solid momentum change == (total mass) * g * t
      // within 1e-8 relative (domain-side impulse: closed box variant)
  TEST(TryStep, DrivenBox_PoseFollowsPath)                 // committed pose equals the driven path at time()
  TEST(BuildTry, KeyInsideStep_IsSolidTravelledTooFar)     // solids kPathJumpInsideStep -> limit from the key
  TEST(ApplyPressureToSolids, MovesTooFar_ShortensAndRetries)  // a light free box under a strong push: kSolidTravelled
      // TooFar in kSolvePressure; the half step commits
  TEST(TryStepWithStages, RejectAfterFindSolidSurfaces_SolidPosesUnchanged)
  TEST(TryStepWithStages, RejectAfterCollideSolids_ChangesNothing)
  TEST(TryStepWithStages, RejectAfterMoveSolids_ChangesNothing)  // solid poses, velocities, solids snapshot bytes
  TEST(TryStep, LiquidForceOnSolids_FromCommittedTry)      // A60: BuildFreeBoxScene(500), one committed step: one
      // entry per solid in ascending solid_id (the domain box, the free box); bytes equal to solids_try.liquid_forces()
      // read by a stage table at the end of stage 19; viscous parts exactly 0 (inviscid water)
  TEST(TryStepWithStages, RejectAfterCheckStep_LiquidForcesEmpty)  // A60: the same scene, a reject after stage 19:
      // liquid_force_on_solids is empty
  TEST(TryStep, LiquidForceOnSolidsInSyrup_HasViscousPart)  // A60: TryStep, FreeBoxInViscousLiquid_SolvedMotion
      // WrittenBack's scene after the box moved for 5 steps: the box's viscous_force is not zero and equals the
      // impulse handed to the solids try divided by the step duration (bytes)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** Snapshots now include `solids::ReadSnapshot`; extend
  `ExpectSameBytes`. **Step 4:** gate. **Commit.** Subject: `Move fixed, driven and free solids in the step`.

### Task 27: The open sea and the wave paddle, stage 7 and stage 1 (b)

**Amended 2026-11-14 (A17, batched-run buffers):** the sites are marked on the processor where the particles live
(the run's processor in one piece, the CPU in a batched run, Appendix A row M2); in a one-piece GPU run the site
bits and the removal list come back in two counted copies (Appendix B row O2).

**Amended 2026-11-14 (Foundation as built):** the GPU site marking is a thin loop over a per-site function added to
`add_new_particles_steps.h`, with a `cpu` test over every thread block and thread; the `gpu` tests write digests.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/solids_stages.cc`, `add_new_particles_cpu.cc`, `add_new_particles_gpu.cu`,
  `add_new_particles_steps.h` (the mark of one lattice site), `choose_active_cells_rule.h`, `stage_table.cc`
  (entry 7), `solids_stages_test.cc`, `step_stages_steps_test.cc`, `step_stages_gpu_test.cc`,
  `step_test_scenes.h/.cc` (`BuildWaveScene(double scale)`, below)

**Interfaces:**
- Produces design 5.2 item 7 (`solids::DriveWaves`, which calls `pressure::ApplyOpenSeaRelaxation` itself;
  block partials kept, totals with `math::TreeSum` into `StepReport::wave_momentum_added`,
  `wave_energy_added`), stage 1 (b) and design 5.4 "Open-sea zone" (one bit per lattice site marked on the
  simulation's processor; no 4096-site limit), and the open-sea blocks in the stage 2 cover.
- **Plan decision, the small wave scene** (`BuildWaveScene(1)`): domain x in [0, 1.6], y in [0, 0.2], z in
  [0, 0.4] m, top open; still water depth 0.2 m; cells 2.5 cm, one size level; a driven flat paddle at
  x = 0.05 with stroke 0.02 m and period 1 s (solids' `BuildWavePaddlePath`); the open-sea zone x in [1.2, 1.6]
  with no incoming wave; `open_sea_particle_spacing` 1.25 cm.

- [ ] **Step 1: Failing tests:** (`LABEL cpu`) `TryStep, WaveStageTrace` (7 present);
  `DriveWaves, ReportTotals_EqualTreeSumOfBlocks` (bytes); `AddNewParticles, OpenSeaAboveSurface_RemovedThroughOutlet`;
  `AddNewParticles, OpenSeaEmptySite_RefilledWithWaveVelocity`; `TryStepWithStages,
  RejectAfterDriveWaves_ChangesNothing`; `TryStepWithStages, RejectAfterAddNewParticlesWithOpenSea_ChangesNothing`;
  `OpenSeaSteps, EveryThreadBothOrders_EqualCpuMarks` (`BuildWaveScene(1)`: the per-site function run for every
  thread block and thread of `ComputeLaunchSize(sites)`, launch order and reverse, gives the CPU site bits; the
  main evidence until the hardware run);
  (`LABEL gpu`, each writing `WriteResultDigest` of the site bits and the refilled particles)
  `AddNewParticles, OpenSeaOnGpu_EqualsCpu` (site bits and removal list equal; refill positions within 6.4); `AddNewParticles, OpenSeaOnGpu_CopiesCounted` (A17 M2:
  `bytes_from_processor` of the stage == `ceil(sites / 8) + 8 * removed`, two copies, plus the refill's
  `AddParticlesResult` copies).
- [ ] **Step 2: Run** (fail). **Step 3: Implement** from `main` `liquid_substep_3d_staging.cc:547-630` (the rule;
  not the CPU scan or its limit). **Step 4:** gate. **Commit.** Subject:
  `Drive waves and keep the open sea filled in the step`. Body: `CUDA: compiled, not run` (site marking).

### Task 28: Snow and springy solid sets, and the plate

**Amended 2026-11-14 (A18):** stage 10 builds the blocking-material faces for the liquid solve and stage 12 writes
their solved velocity back to each set, instead of a push to particles in stage 17.

**Amended 2026-11-14 (reconcile):** an OPEN note marked where option (B) of the springy-solid-beside-viscous-liquid
decision would land. Removed by A24.

**Amended 2026-11-14 (A24):** item 10 builds the faces whatever stage follows; stage 11 writes them back as stage 12
does (Task 25). Option (B), a one-step-late push for viscous liquids, is withdrawn.

**Amended 2026-11-14 (A26):** a viscous liquid sticks to the blocking sets: in a viscous scene item 10 also builds
the band faces (`solids::BuildViscousMaterialFaces`), and stage 11 writes them back with
`solids::ApplyViscousSolveToMaterials` in place of `ApplyPressureToMaterials`; the step-owned row `"step: material
face velocity after solve"` is added. Replaced (A24 sentence above): "stage 11 writes them back as stage 12 does".

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/transfer_stages.cc`, `apply_forces_cpu.cc`, `apply_forces_gpu.cu`,
  `solids_stages.cc`, `pressure_stages.cc`, `viscosity_stage.cc` (A24: stage 11's span; A26: the material face
  span and `ApplyViscousSolveToMaterials`, replacing `ApplyPressureToMaterials` there), `step_memory_estimate.cc`
  (the material face velocity after pressure row; A26: the material face velocity after solve row),
  `commit_step_cpu.cc`, `step_test_scenes.h/.cc` (`BuildSnowAndJellyScene`: water, a snow
  block and a jelly block in three sets beside each other on the floor of a closed box 0.6 x 0.3 x 0.4, a fixed
  plate cutting the jelly), `solids_stages_test.cc`, `step_duration_test.cc`

**Interfaces:**
- Produces design 5.2: item 4's `ComputeInternalForces` and `ComputeParticleFriction`; item 6's
  `ApplyInternalForce` (its one form, `MaterialBuffers` last, no allocation) per set with internal force into
  `totals_of_block` (the range's sub-span), totals with
  `ComputeInternalForceTotal` per set in set order into `StepReport::internal_force`; item 3's
  `ApplyPlateSides`; item 8 with every non-gas set; item 10's `solids::BuildBlockingMaterialFaces(
  BlockingMaterialFacesInputs{grid, all, geometry, coverage, {set S, set E}}, solids_buffers)` after
  `BuildMotionMap`, its `ToBlockingMaterialFaces` given to `EquationInputs::blocking_material_faces`; item 12's
  `SolvedArrays` with the step-owned `blocking_material_velocity_after_pressure` span and, after
  `ApplyPressureToSolids`, `solids::ApplyPressureToMaterials(faces, that span, {set S, set E}, solids_buffers)`
  (its `PressureOnMaterials::push` per set into the report next to the solids' pushes); item 14 and 15 for a set
  no solve writes (its `GridFromParticles::velocity` after `CollideSolids` and, on the pieces the liquid presses,
  after `ApplyPressureToMaterials`, or, in a viscous scene, on its band faces after `ApplyViscousSolveToMaterials`
  (A26); null `was_solved`, null `cell_was_solved`); item 17's `ComputeCutWork`; item
  20's `ApplyPlateRemoval` when the plate's path ends inside the step.
- **Both liquid solves (A24).** Item 10 builds the faces in every scene with liquid and a blocking set, whether
  stage 11 or stage 12 follows; viscosity's joint solve carries them as `SolvePressure` does, and stage 11 calls
  `ApplyPressureToMaterials` with the same arguments (Task 25). This task's scene (`BuildSnowAndJellyScene`) holds
  water with viscosity 0, so stage 12 runs here; the viscous case runs on the five-material scene (Task 31).
  **Replaced in part (A26):** "viscosity's joint solve carries them as `SolvePressure` does, and stage 11 calls
  `ApplyPressureToMaterials` with the same arguments"; see the next bullet.
- **The band faces of a viscous scene (A26; Task 25's stage 11 bullet).** In a viscous scene with a blocking set,
  item 10 calls, right after `BuildBlockingMaterialFaces`, `solids::BuildViscousMaterialFaces(
  BlockingMaterialFacesInputs{grid, all, geometry, coverage, {set S, set E}}, faces, geometry, solids_buffers)`
  giving `viscous_faces` (per blocking set, the faces with face mass within `kFitReach` cell widths of a liquid cell
  centre; every `BlockingMaterialFace` piece is among them); stage 11 gives `solids::ToViscousMaterialFaces(
  viscous_faces)` to `ViscosityInputs::material_faces`, passes the step-owned `material_face_velocity_after_solve`
  span (one entry per `ViscousMaterialFace`) in `SolvedCouplingArrays`, and then calls
  `solids::ApplyViscousSolveToMaterials(viscous_faces, material_face_velocity_after_solve, {set S, set E},
  solids_buffers)`: each band face of each blocking set gets its solved velocity in its set's
  `GridFromParticles::velocity`; every other face of the set keeps its velocity after `ApplyForces` and
  `CollideSolids`; its `PressureOnMaterials::push` per set goes into the report as in stage 12. Items 14 and 15 for
  `S` and `E` then read that velocity. Step-owned row `"step: material face velocity after solve"` (given solids'
  `max_motion_map_entries` x 2, 8 B, 1 copy, one step try; only in a viscous scene with a blocking set). An
  inviscid scene (this task's tests) builds no band faces and stage 12 is unchanged. The viscous case runs on the
  five-material scene (Task 31: `FiveMaterialsWithSnow_Commits`, `FiveMaterialsWithSnowAndFreeBox_Commits`).

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, SnowStageTrace)                       // design 7 "Stage trace", snow row
  TEST(TryStep, ThreeSets_EachSetsArraysCarryItsKey)  // MaterialSetKey of each set's GridFromParticles and snapshots
  TEST(ApplyForces, InternalForceTotals_SameForOneOrManyRanges)  // stage table entry 6 called per one-block range
  TEST(ComputeStepDuration, MaterialWaveSpeed_IsActive)          // committed material_wave_speed from the snow set
  TEST(ComputeInternalForces, StepTooLong_ShortensToMaterialLimit) // kStepTooLongForMaterial, limit = magnitude
  TEST(ApplyPlateSides, JellyParticlesGetTheirSide)
  TEST(SolvePressure, SnowAndJellyFacesMoveInTheSameStep)  // A18: after stage 12 each set's face velocity on every
      // entry piece equals the solved blocking_material_velocity_after_pressure (bytes); other faces unchanged
  TEST(TryStepWithStages, RejectAfterParticlesToGridWithSnow_ChangesNothing)  // and after 6, 8, 10, 12, 14, 15,
                                                                              // 16, 17 on this scene, one test each
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** The GPU gravity pass gains nothing; the internal force is
  materials' own operation. **Step 4:** gate. **Commit.** Subject: `Carry snow and springy solids as their own sets`.

### Task 29: Gas, stages 1 (c), 2 (e) and 13

**Amended 2026-11-14 (A18):** stage 13 starts with `gas::ApplyGasStepStart` in a scene where stage 10 does not run.

**Amended 2026-11-14 (A24):** `GasStepStartInputs` has no `sealed_gas_coupling` (deleted with `SealedGasCoupling`).

**Amended 2026-11-14 (A17, batched-run buffers):** the marker lists live in CPU memory in every GPU layout and their
read-back is counted (Appendix A row M3, Appendix B row O3); `FindCellsToKeepActive` always runs on the amounts'
processor (M4). The batched-layout paths of stages 1 (c) and 2 (a) are Task 33b's.

**Amended 2026-11-14 (Foundation as built):** the `gpu` test writes the marker lists' digest.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/gas_stage.cc`, `gas_stages_test.cc`
- Modify: `add_new_particles_cpu.cc`, `choose_active_cells_cpu.cc`, `choose_active_cells_rule.h`,
  `stage_functions.h`, `stage_table.cc` (entry 13), `commit_step_cpu.cc`, `check_step_cpu.cc`,
  `simulation_snapshot.cc`, `step_test_scenes.h/.cc` (`BuildSmallFireScene()`, below), `step_duration_test.cc`,
  `step_stages_gpu_test.cc`, `run_plan_test.cc` (A17)

**Interfaces:**
- Produces design 5.2 items 1 (c) (`FindMarkerChanges`, `RemoveParticles` with `kMarkerNotNeeded`,
  `AddParticles` with `kRefillRegion`, counts into `markers_added`, `markers_removed`), 2 (a) and (e)
  (`FindCellsToKeepActive` on the committed grid; then the step adds to the needed cell list every cell that
  touches a marked cell, and the cells of each active gas source box, before `grid::BuildGrid`; then
  `gas::BuildTry`, `ComputeGasSeenByLiquid`), 6 for the gas set, 9 with an all-zero
  mask in a gas-only scene, 13 (in a gas-only scene first `gas::ApplyGasStepStart(GasStepStartInputs{&grid, all,
  geometry, &grid_edge_is_open_to_sky, domain_box, gas_sources, step_start_time, step_duration}, *gas,
  gas_amounts_try, gas_buffers, gas_pressure_buffers)` (A24: no coupling argument), whose blob list is empty
  without liquid; then `SolveGas` with null liquid pointers and an empty `sealed_gas_pressure`), 14 and 15 for `G`;
  `gas::CommitTry` at commit, `gas::RejectTry` on reject; `CommittedStepLimits::gas`; the snapshot's
  `gas_velocity_after_pressure` and `gas` (`ReadGasAmountsSnapshot`, `BuildGasAmountsFromSnapshot`).
- **Plan decision, the small fire scene** (`BuildSmallFireScene()`): the gas plan's small fire (box 0.5 x 0.5 x
  1.0 m, 0.0625 m cells, its source, burn and wind values), as a `ReadyScene`, 8 markers per cell, no liquid.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, GasOnlyStageTrace)                   // design 7 "Stage trace", gas only: 9, 13; no 10, 11, 12
  TEST(SolveGas, TooStrongSource_ShortensToItsLimit) // design 7 "Gas flow reject": heat 1e12 W/m3:
      // kStepTooLongForGasFlow in kSolveGas, shorter_step_duration == context.limit; the shorter try commits
  TEST(ComputeStepDuration, Gas_IsActive)            // committed gas limit from the last commit
  TEST(SolveGas, CancelSetDuringSolve_IsCancelRequested)
  TEST(ChooseActiveCells, GasMarkedCells_GrowTheActiveRegion)  // a hot cell at the edge of the active region:
      // the next try's grid holds every cell touching it
  TEST(AddNewParticles, MarkersAddedAndRemovedReported)
  TEST(TryStepWithStages, RejectAfterSolveGas_GasAmountsUnchanged)  // and after 1 and 2 on this scene
  TEST(BuildSimulationFromSnapshot, ContinuedFireRun_SameBytes)     // gas design "Snapshot round trip" through the
                                                                    // step: 10 + 10 steps against 20
  TEST(AddNewParticles, MarkerListCopiesCounted)     // A17 M3: the try's copies include MarkerChanges' copy fields
                                                      // (0 on the CPU); gpu case below
  TEST(BuildRunPlan, PlannedStepCopies_GasRows)      // A17: Task 8's test on BuildSmallFireScene: one piece O1 + O3,
                                                      // forced batched B1 + B2 + B4
  // LABEL gpu, step_stages_gpu_test.cc
  TEST(AddNewParticles, OnGpuMarkerListsReadBackCounted)  // A17 M3: bytes_from_processor of stage 1 ==
      // MarkerChanges::bytes_copied, > 0 in the small fire; to_add and to_remove spans point into CPU memory;
      // amended (Foundation as built): WriteResultDigest of to_add and to_remove
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Run tracked gas in the step`.

### Task 30: Gas beside liquid

**Amended 2026-11-14 (A18):** a sealed blob beside inviscid liquid is solved in stage 12 with the liquid; open gas
and gas beside a viscous liquid are still read one step late.

**Amended 2026-11-14 (A24):** a sealed blob beside a viscous liquid is solved in stage 11 with the liquid, as in
stage 12; no coupling choice remains (`SealedGasCoupling` is deleted). Only open gas (also above a viscous free
surface), slivers, cells that touch gas and held blobs are read one step late. `TryStep,
SealedGasBesideViscousLiquid_ReadOneStepLate` becomes `TryStep, SealedGasBesideViscousLiquid_SolvedInStageEleven`.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/pressure_stages.cc` (gas inputs of 9 and 10 from `gas_seen` and from
  `ApplyGasStepStart`), `viscosity_stage.cc` (A24: stage 11's `sealed_gas_pressure` span), `gas_stage.cc`,
  `gas_stages_test.cc`, `step_memory_estimate.cc` (the sealed gas pressure row),
  `step_test_scenes.h/.cc` (`BuildGasBesideLiquidScene(bool open_top)` as a builder
  argument enum `Top { kOpen, kClosed }`: box 0.4 x 0.4 x 0.6, water to 0.2 m, ambient gas above, 8 markers per
  open cell)

**Interfaces:**
- Produces the order rule of design 5.2 with A18 and A24: stage 10 calls `gas::ApplyGasStepStart(
  GasStepStartInputs{&grid, all, geometry, &grid_edge_is_open_to_sky, domain_box, gas_sources, step_start_time,
  step_duration}, *gas, gas_amounts_try, gas_buffers, gas_pressure_buffers)` before
  `pressure::BuildPressureEquations`, whose `EquationInputs` take `gas_seen` and the `SealedGasForLiquid` views;
  stage 12's `SolvedArrays`, or stage 11's `viscosity::SolvedCouplingArrays` (Task 25), take the step-owned
  `sealed_gas_pressure` span; `SolveGas` then reads `fields.velocity_after_pressure`, `L.was_solved`, the blob list
  and that span, whichever liquid stage ran. The blob list is the same whichever stage follows stage 10.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, LiquidAndGasStageTrace)               // design 7: 12 before 13
  TEST(TryStep, MarkersInWater_RemovedAtNextStep)    // kMarkerNotNeeded, exact count (the gas plan checks the
                                                      // other gas-beside-liquid rows with its own loop)
  TEST(TryStep, LiquidReadsOpenGasPressureOneStepLate)  // Top::kOpen: the air-end pressure at step n equals the gas
                                                        // pressure committed at step n - 1, exact
  TEST(TryStep, SealedGasSolvedWithLiquid)            // Top::kClosed: the committed background pressure after step n
      // equals atmosphere + stage 12's sealed_gas_pressure of step n (exact); the liquid's air-end pressure in step
      // n is that value + the gas pressure of the last commit
  TEST(TryStep, SealedGasBesideViscousLiquid_SolvedInStageEleven)  // A24: Top::kClosed with syrup: stage 11 runs
      // (no 12), the blob is listed by stage 10, sealed_gas_pressure is written by stage 11 and read by stage 13;
      // each committed background pressure == atmosphere + that step's sealed_gas_pressure (exact)
  TEST(TryStepWithStages, RejectAfterSolveGasBesideLiquid_ChangesNothing)  // every committed array of both sets
  TEST(TryStepWithStages, RejectAfterBuildPressureEquationsWithGas_ChangesNothing)  // ApplyGasStepStart wrote only
                                                                                    // the try
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Run gas beside liquid in one step`.

### Task 31: The whole stage list, every reject and cancel

**Amended 2026-11-14 (reconcile):** the tests on the five-material scene were marked blocked by the open decision on
a springy solid beside a viscous liquid (expectations unchanged). Unblocked by A24.

**Amended 2026-11-14 (A24):** option (C) is built (stage 11 carries the material faces, Task 25), so every test of
this task runs as written; the five-material scene with snow now commits (`TryStep,
FiveMaterialsWithSnow_IsViscousLiquidWithSnow` becomes `TryStep, FiveMaterialsWithSnow_Commits`).

**Amended 2026-11-14 (A26, A27):** stage 11 writes the band faces back with `ApplyViscousSolveToMaterials` and the
free solids with `ApplySolvedMotionToSolids`; the free box no longer has to stay away from the syrup.
`FiveMaterialsWithSnow_Commits` is kept (free box away from the syrup, its expectation changed to the A26
write-back), and `TryStep, FiveMaterialsWithSnowAndFreeBox_Commits` is added (the same scene with the free box
beside the syrup): it extends, and does not replace, the former test.

**Amended 2026-11-15 (A64):** `ContinuedFiveMaterialRun_SameBytes` carries the three cell-size arrays (the scene keeps
`has_cell_resize`); no grid-only array exists.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/foam_and_spray_stage.cc`, `whole_step_test.cc`
- Modify: `stage_table.cc` (entry 18), `step_test_scenes.h/.cc` (`BuildFiveMaterialScene(with_snow)`: the
  scene of design 5.2 at small size, cells 2.5 cm, closed box 0.8 x 0.4 x 0.4 with the top open, water and
  syrup in set L, snow S, jelly E, smoke G; the fixed plate, a driven paddle, a free box away from the syrup,
  an open-sea zone, a water-syrup `MaterialPair`, a squeezable law on water, `has_cell_resize` true; A27, plan
  decision: a second builder argument `enum class FreeBoxPlace { kAwayFromSyrup, kBesideSyrup }`, default
  `kAwayFromSyrup` for every test that does not name it; `kBesideSyrup` floats the box on the syrup, touching it)

**Interfaces:**
- Produces design 5.2 item 18: `make_foam_and_spray` with `FoamAndSprayInputs` filled as written there
  (`after_pressure = {&L.after_pressure}`, `step_count + 1`, `blocks`, empty `block_in_whole_grid`, `&fields`);
  `false` is `kCallbackFailed`. `kStages` now holds every stage 1 to 20.

- [ ] **Step 1: Failing tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStepWithStages, RejectAfterEveryStage_ChangesNothing)  // design 7 "Reject changes nothing": for stage 1
      // to 19 and each of the four representative reasons, on BuildFiveMaterialScene(without snow):
      // ReadSnapshot before and after are ExpectSameBytes; rejected_in is that stage
  TEST(TryStep, CancelBeforeEachStage_ChangesNothingAndNextCommits)  // design 7 "Cancel", before stage k, each k:
      // a stage table sets the flag at the end of stage k - 1
  TEST(TryStep, StageTraceOfEveryScene)        // design 7 "Stage trace": liquid, syrup, snow, gas only, liquid and
      // gas, the five-material scene without snow: the lists of design 5.2 with skipped stages absent
  TEST(TryStep, MaterialSets_KeysAndSums)      // design 7 "Material sets": every set's arrays carry its key; the
      // liquid set's sums equal a run without markers (bytes); the liquids given in two sets: kMaterialSetsNotValid
  TEST(TryStep, FiveMaterialsWithSnow_Commits)  // A24: BuildFiveMaterialScene(with snow): CheckSceneCanRun gives
      // no kUnavailable finding; a step commits; stage 11 runs (no 12); A26: ApplyViscousSolveToMaterials writes
      // both blocking sets (S and E face velocities on every band face equal material_face_velocity_after_solve,
      // bytes; their other faces keep the velocity CollideSolids left). Replaced (A26): "ApplyPressureToMaterials
      // writes both blocking sets (S and E face velocities on their entry pieces equal
      // blocking_material_velocity_after_pressure)"
  TEST(TryStep, FiveMaterialsWithSnowAndFreeBox_Commits)  // A26, A27: BuildFiveMaterialScene(with snow,
      // FreeBoxPlace::kBesideSyrup): CheckSceneCanRun gives no finding of level kUnavailable; stage 11 runs (no 12)
      // with free_solids holding the box and material_faces holding band faces of both S and E; after stage 11
      // (a stage table with entries 12 to 20 null) the S and E velocities on every band face equal
      // material_face_velocity_after_solve and their other faces the velocity CollideSolids left (bytes), and the
      // box's velocity and turn rate in solids_try equal free_solid_motion (bytes); with the full table the step
      // commits
  TEST(BuildSimulationFromSnapshot, ContinuedFiveMaterialRun_SameBytes)  // design 7 "Continued run": snapshot at
      // step 20, rebuild, 20 more, against 40 uninterrupted (cell-size arrays present, no resize callback; amended 2026-11-15, A64: was "region arrays")
  TEST(MakeFoamAndSpray, CallbackSeesTheTryAndItsContext)  // hand-written: one entry in after_pressure, step_count
                                                           // = committed + 1; returning false: kCallbackFailed
  ```
  `BuildFiveMaterialScene` holds syrup in set L beside the jelly set E (and, with snow, the snow set S), so stage 10
  builds material faces and stage 11 carries them (A24; viscosity Tasks 15 and 15a, pressure Tasks 28 and 45; A26:
  as velocity unknowns, viscosity Task 15c). Replaced (A27): "The free box stays away from the syrup: a free solid
  beside a viscous liquid is still `kViscousLiquidWithFreeSolid`." The free box may touch the syrup
  (`FreeBoxPlace::kBesideSyrup`); the other tests keep it away so their expectations are unchanged.
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Run every stage of the step on one scene`.

### Task 32: The resize hand-over

**Amended 2026-11-14 (A30):** `BuildResizedGrid` requires the list on the simulation's processor (debug assertion); test `ListOnOtherProcessorOrNoCpuGridOnGpu_IsBadInput` becomes `NoCpuGridOnGpu_IsBadInput`.

**Amended 2026-11-15 (A58, reconcile item 31, A63):** `BuildResizeTry` builds no buffer (its try copies are pieces BuildSimulation
built), so it returns only `kTryInProgress` (no `kOutOfMemory`); `GridFields` holds the three cell-size arrays
(design 3.10), which the hand-written resize copies like the other fields.

**Amended 2026-11-14 (reconcile):** adds `ResizedCells`, `BuildResizedGrid` (the step builds the resized grid from a
cell list, on the GPU from the list cell resize uploads), `ResizeTry::free_cpu_grid_copy_buffers` and
`ApplyResizedCpuGridCopy` (a batched GPU run keeps cell resize's CPU-built grid as its CPU copy, no read-back).

**Amended 2026-11-14 (Foundation as built):** the `gpu` test writes the digest of the resized grid's tables (grid
tables are discrete, so the byte comparison with the CPU grid stays).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/resize_access.h`, `resize_access.cc`, `resize_access_test.cc`

**Interfaces:**
- Produces design 3.10 exactly: `GridFields` (moved here from `simulation_state.h`, which now includes it),
  `ResizeTry` (every accessor), `BuildResizeTry`, `ApplyResizedGrid`, `CommitResize`, `RejectResize`. Plan
  decision: `ResizeTry` keeps a pointer to the simulation and marks it busy; `TryStep`, `ReadSnapshot` and
  `ApplyLimits` return `kTryInProgress` while one is alive.
- **Plan decision (reconcile items 12, coordinator 1 and 6; `plan-decisions.md` "Reconcile pass (2026-11-14)"),
  `resize_access.h`:**

  ```cpp
  // The cell list of a resized grid, on the simulation's run_on, any order (as grid::GridInputs).
  struct ResizedCells {
    std::span<const grid::CellId> cells;                 // A CPU run: the list in CPU memory.
    const Buffer<grid::CellId>* cell_buffer = nullptr;   // A GPU run: the list uploaded by cell resize.
    std::uint32_t cell_buffer_count = 0;                 // Entries of *cell_buffer used.
    const grid::StaggeredGrid* cpu_resized_grid = nullptr;  // A GPU run: cell resize's CPU build of the same list.
  };
  // Builds the resized grid (committed frame, version committed + 1) into free_grid_storage() on the
  // simulation's run_on with grid::BuildGrid, then binds the try as ApplyResizedGrid does, and returns the grid
  // for CommitResize. In a GPU run the version and block count must equal *cpu_resized_grid's (else kStageFailed,
  // module kGrid). A GPU run without cpu_resized_grid: kBadInput. Requires: the list on the simulation's
  // processor (debug assertion; A30). Grid
  // failures map as in stage 2 (kTooManyCellBlocks is kRaiseLimits). On failure free_grid_storage() is usable.
  std::expected<grid::StaggeredGrid, StepFailure> BuildResizedGrid(ResizeTry& resize_try,
                                                                   const ResizedCells& resized_cells);
  // ResizeTry member: in a batched GPU run, the free half of "step: grid copy on the CPU", into which cell resize
  // builds its CPU resized grid; null in every other run.
  grid::GridBuffers* free_cpu_grid_copy_buffers();
  // A batched GPU run: keeps the CPU resized grid built into free_cpu_grid_copy_buffers() as the CPU copy that
  // CommitResize swaps in (RejectResize releases it). kBadInput in other runs, for a grid in other buffers, or a
  // version or block count different from the resized grid's.
  std::expected<void, StepFailure> ApplyResizedCpuGridCopy(ResizeTry& resize_try,
                                                           grid::StaggeredGrid&& cpu_resized_grid);
  ```
  `ApplyResizedGrid(ResizeTry&, const grid::StaggeredGrid&)` stays (design 3.10) for a caller that built the grid
  into `free_grid_storage()` itself (a CPU run, tests). `CommitResize` in a batched GPU run requires
  `ApplyResizedCpuGridCopy` first (else `kBadInput`, nothing committed) and swaps the CPU copy pair with no copy.

- [ ] **Step 1: Failing tests** (`LABEL cpu`, `BuildColumnAtRestScene` after 5 steps; the "resize" is a
  hand-written callback that rebuilds the same cell list with version + 1, regroups, copies every field):

  ```cpp
  TEST(CommitResize, SameCells_VersionGrowsTimeDoesNot)   // state_version + 1; time and step_count unchanged
  TEST(CommitResize, AfterResize_StepGivesSameBytesAsWithout)  // except the grid version
  TEST(RejectResize, ChangesNothing)                      // ExpectSameBytes
  TEST(BuildResizeTry, Twice_IsTryInProgress)
  TEST(ReadSnapshot, DuringAResizeTry_IsTryInProgress)
  TEST(ApplyResizedGrid, WithGas_BuildsTheGasTry)         // BuildSmallFireScene: gas_try() not null after the call
  TEST(AdvanceTo, ResizeCallback_UsesTheHandOver)         // resize_cells set to the hand-written function; the next
                                                          // TryStep carries the version after the resize
  TEST(BuildResizedGrid, CpuList_EqualsApplyResizedGridOfTheSameGrid)  // reconcile: same tables and bound try
  TEST(BuildResizedGrid, NoCpuGridOnGpu_IsBadInput)        // reconcile; A30: list on the other processor dropped
  TEST(ApplyResizedCpuGridCopy, NotABatchedGpuRun_IsBadInput)  // reconcile (CPU run: free_cpu_grid_copy_buffers null)
  // LABEL gpu (step_stages_gpu_test.cc)
  TEST(BuildResizedGrid, OnGpuFromUploadedList_EqualsCpuResizedGrid)  // reconcile: ReadGrid of the result equals
      // cpu_resized_grid byte for byte (discrete tables); WriteResultDigest of the tables
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement** design 5.11. **Step 4:** gate. **Commit.** Subject:
  `Hand the committed state to a cell resize between steps`. New names: the 3.10 list.

### Task 33: The batched seam: `run_stages`, the batched layout, `StepArray`

**Amended 2026-11-14 (A15):** the third mode builds the solvers' buffers in their batched layout, the loaded try
owns its own one-piece `PressureBuffers` for pass 1, and the solvers' copy reports are reset per try.

**Amended 2026-11-14 (A24):** `RangeCut_SameBytes` runs as written (option (C) built; its reconcile "blocked" mark is
removed).

**Amended 2026-11-14 (reconcile):** the per-try reset calls `ResetSolveCopyReport` on the solvers' buffers (new
buffers forms in pressure Task 56, viscosity Task 26, gas Task 29), not on a `SolveBatches` found through the const
`FindSolveBatches`.

**Amended 2026-11-15 (A58):** the batched layout's buffers are pieces of the run's blocks: the particle arrays and
every whole array placed in CPU memory are pieces of `cpu_run_memory` (`CpuMemory::kLockedForCopies`), the loaded
`StepTry`'s buffers pieces of `run_memory` (private `BuildLoadedTry(..., run_memory)`); the solver builders of
`kBatchedWithSolvesInPieces` get `run_memory, cpu_run_memory` last. "AllocatedEqualsEstimate" in the test names below
means `handed_out_bytes == reserved_bytes` and `handed_out_cpu_bytes == reserved_cpu_bytes` (names kept).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/step_arrays.cc`, `batched_seam_test.cc`
- Modify: `native/fluidity/step/step_stages.h` (`StepArray`, `ReadStepArrays`), `simulation.cc` (batched layout),
  `try_step.cc`

**Interfaces:**
- Produces design 3.11's `StepArray` and `ReadStepArrays` (one entry per array instance, generated from the
  memory estimate entries), the `run_stages` path of `TryStepWithStages` (stages 3 to 19), and the batched layout of design
  5.9 in `BuildSimulation` (particle arrays with `CpuMemory::kLockedForCopies` and per-try arrays in CPU
  memory, the loaded `StepTry` over a `GridBuffers` of `max_loaded_blocks` blocks, built with
  `grid::BuildGridBuffers(settings, grid_builder_buffers)` from the simulation's one `grid::GridBuilderBuffers`; reconcile item 11). `kBatchedRunNotBuilt` when
  `run_stages` is null or `max_loaded_blocks` is 0.
- **Plan decisions (A15):** in both batched modes the loaded `StepTry` owns a one-piece `PressureBuffers` built
  for `max_loaded_blocks` blocks on the run's processor (the rows of `ComputeDistanceToSurface` and
  `FindLiquidSurface`), which stage 9's range part uses on a loaded try; `batches::ReadBatch` moves its range into
  the whole buffers with `pressure::WriteSnapshotRange`. In `kBatchedWithSolvesInPieces` the liquid and gas
  `PressureBuffers`, `ViscosityBuffers` and `GasBuffers` are built with `max_loaded_blocks =
  plan.max_solve_loaded_blocks`; the stage calls 9 to 13 do not change. At the start of each try the step calls
  the buffers form of the reset on each solver's buffers (reconcile: `FindSolveBatches` returns a const pointer, so
  the step never resets through it): `pressure::ResetSolveCopyReport(PressureBuffers&)` on the liquid and the gas
  `PressureBuffers`, `viscosity::ResetSolveCopyReport(ViscosityBuffers&)`, `gas::ResetSolveCopyReport(GasBuffers&)`
  (each does nothing in one piece); `kBatchedRunNotBuilt` also when the mode is `kBatchedWithSolvesInPieces` and
  `max_solve_loaded_blocks` is 0.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; the hand-written `run_stages` runs each stage of the given table
  once per block range of the whole grid, in order, on the CPU):

  ```cpp
  TEST(TryStepWithStages, RangeCut_SameBytes)            // design 7 "Range cut": BuildFiveMaterialScene without
                                                         // snow, one range against one range per block
  TEST(BuildSimulation, BatchedWithoutRunStages_IsBatchedRunNotBuilt)
  TEST(BuildSimulation, BatchedLayout_AllocatedEqualsEstimate)
  TEST(BuildSimulation, BatchedWithSolvesInPieces_AllocatedEqualsEstimate)  // A15: kCpu and kRunOn sums each equal
  TEST(BuildSimulation, SolvesInPiecesWithoutSolvePiece_IsBatchedRunNotBuilt)  // A15
  TEST(TryStep, SolveCopyReportsResetEachTry)                                 // A15
  TEST(ReadStepArrays, MatchesTheRows)                   // one entry per row instance, per-set rows once per set;
                                                         // read_by_stages and written_by_stages as design 5.2's table
  TEST(TryStep, RunStagesCallbackGetsItsContext)
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Open the step to a batched run of its stages`.

### Task 33a: Where each buffer of a batched run lives

**Amended 2026-11-14 (A17, batched-run buffers):** new task. The placement rule, the loaded instances of the module
buffers, and the CPU solids buffers of stage 17 (Appendix A rows M9, M10, M12).

**Amended 2026-11-14 (A24):** option (A) is withdrawn, so the two layout tests on the five-material scene that the
reconcile pass marked blocked under it run as written.

**Amended 2026-11-15 (A64, A58):** stage 16 has no whole-grid part (the grid-only move is in Task 39a), so the
stage-place table gives 19's whole part only in pass 2; the stage-17 CPU `SolidsBuffers` is built into
`cpu_run_memory` (not plain `BuildBuffer`).

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/step_stages.h` (`ArrayPlace`, `StepArray::whole_place`), `step_arrays.cc`,
  `step_memory_estimate.cc`, `simulation.cc`, `simulation_state.h`, `solids_stages.cc` (stage 17 on the CPU instance),
  `batched_seam_test.cc`
- Create: `native/fluidity/step/batched_buffers_test.cc`

**Interfaces:**
- Consumes: Task 33's batched layout; `solids::BuildSolidsBuffers`, `materials::BuildMaterialBuffers`,
  `transfer::BuildTransferBuffers` (same `material_count`), `contact::BuildContactBuffers` with smaller particle counts (the batched layout changes no signature);
  `solids::WriteBlockPartialsRange` (solids plan Task 28, amended).
- **Plan decisions:**
  - `enum class ArrayPlace : std::uint8_t { kLoadedOnly, kWholeOnProcessor, kWholeInCpuMemory };` and `StepArray`
    gains `ArrayPlace whole_place;`, set by the placement rule of Global Constraints from `read_by_stages`,
    `written_by_stages`, "kept by the commit" and a private table in `step_arrays.cc` holding the place of each
    stage as batches design 5.8 gives it with the A17 amendment (stages 1, 2, 17, 20, 21 whole state on the CPU; 3
    to 9 pass 1, with 3's `FindGridEdgesOpenToSky` and 9's whole part on the processor; 10 to 13 on the processor;
    14, 15, 16, 18, 19 pass 2, with 19's whole part on the processor; A64: stage 16 has no whole part in the first version, replaced "16's and 19's whole parts"). The step includes no batches
    header; batches Task 37 tests that its `kStagePlaces` gives the same `whole_place` for every array. The rule's
    results the tests pin: each set's `GridFromParticles` and `CellsFromParticles`, the six cell arrays of stage 10, coverage,
    `grid_edge_is_open_to_sky`, the per-block partials of pass 2 read by 19's whole part: `kWholeOnProcessor`;
    `before_forces`, `after_pressure`, `SolidsAtFaces` (start: written by stage 2's fill when stage 3 is skipped;
    end: read by the resize), `SolidReaction`: `kWholeInCpuMemory`; `GridAtParticle`, `ParticleContact`:
    `kLoadedOnly`. In a one-piece run every `whole_place` is `kWholeOnProcessor` (the field is unused there).
  - **Loaded instances (M9).** The loaded `StepTry` owns, on the run's processor, a `transfer::TransferBuffers`, a
    `materials::MaterialBuffers`, a `contact::ContactBuffers` and a `solids::SolidsBuffers`, each built at
    `plan.max_loaded_particles` per material and `plan.max_loaded_blocks` (rows `kRunOn`, counted once in the
    estimate as loaded rows); stage functions run on a loaded try use them. The simulation's whole instances are
    built with every particle count 0 (their grid rows stay at `max_cell_blocks` for stages 10 to 13 and the
    whole parts).
  - **Stage 17 in CPU memory (M10).** A third `SolidsBuffers` with `run_on = kCpu`, the whole particle limit and
    `max_cell_blocks` (every row `where = kCpu`); stage 17 calls `solids::ApplyContactReactions` and
    `ComputeCutWork` with it on the CPU (the `SolidReaction` and cut-work inputs are in CPU memory). The
    `CollideSolids` block partials reach it through `solids::WriteBlockPartialsRange`, called by `batches::ReadBatch`
    after pass 1 (cell resize and batches Task 39); a test here calls it the same way with a hand-written
    `run_stages`. Accessor for batches: `solids::SolidsBuffers* FindContactReactionBuffers(Simulation&)` (null in
    one piece).
  - **Stage 3 once per try.** In a batched run stage 3's `run` is called once per batch; it builds the solids try
    (`solids::BuildTry`) and runs the contact checks only in its first call of a try (the try number is kept in the
    try state) and reuses them after (`plan-decisions.md` "Planned cost ...", open point 3).
  - `BuildStepMemoryEstimate(scene, counts, &plan)` returns these rows with their `where`, so `BuildBatchPlan`'s
    `bytes_on_processor`, `bytes_on_cpu` and `BuildSimulation`'s allocation stay equal.

- [ ] **Step 1: Failing tests** (`LABEL cpu`; plans built with `step::BuildRunPlan` for `RunOn::kGpu` need no GPU;
  layouts built on the CPU use `RunOn::kCpu` with a forced batched plan as Task 33):

  ```cpp
  TEST(ReadStepArrays, BatchedPlacementFollowsTheRule)   // A17 M12: BuildFiveMaterialScene without snow, batched
      // layout: every array's whole_place equals the rule recomputed by the test from read_by_stages,
      // written_by_stages and kStagePlaces; the pinned arrays above have the pinned places
  TEST(BuildStepMemoryEstimate, BatchedGpu_PerParticleModuleRowsOnlyLoaded)  // A17 M9: a RunOn::kGpu batched plan of
      // BuildFiveMaterialScene: every row counted per particle with where kRunOn is at max_loaded_particles; no
      // kRunOn row is counted per particle at the limits; GridAtParticle and ParticleContact have no whole row
  TEST(BuildSimulation, BatchedLayout_LoadedTryOwnsModuleBuffers)  // A17 M9: the loaded try's four instances have
      // room for max_loaded_particles and max_loaded_blocks; the whole MaterialBuffers holds 0 particles
  TEST(BuildStepMemoryEstimate, BatchedGpu_ContactReactionBuffersOnCpu)  // A17 M10: the "solids: " rows of the third
      // instance all have where kCpu; with RunOn::kGpu none of the stage-17 rows is kRunOn
  TEST(MoveSolids, BatchedLayout_ContactPushesEqualOnePiece)  // A17 M10: BuildFreeBoxScene(500), 5 steps, batched
      // CPU layout with run_stages calling WriteBlockPartialsRange after pass 1: ContactOnSolids push, turning push
      // and work equal the one-piece run's bytes (the CPU instance changes where, not what)
  TEST(BuildSimulation, BatchedLayout_AllocatedEqualsEstimateByPlace)  // kRunOn and kCpu sums each equal the plan
  TEST(FindSolidSurfaces, BatchedRangesBuildTheSolidsTryOnce)  // a hand-written run_stages calling stage 3 per
      // one-block range: one solids try per step try (its try_id unchanged across ranges); the step commits
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Place every buffer of a batched run by one rule`. New names: `ArrayPlace`, `StepArray::whole_place`,
  `FindContactReactionBuffers`.

### Task 33b: The CPU stages of a batched run, and their counted copies

**Amended 2026-11-14 (A29):** a reject takes the free CPU copy's buffers back with `grid::TakeBuffers`.

**Amended 2026-11-14 (A17, batched-run buffers):** new task. Stages 1 and 2 run on the CPU in a batched run while
the grid, the gas amounts and the committed grid values live on the run's processor (Appendix A rows M3, M4, M5,
M7, M14; Appendix B rows B1 to B4; B5 withdrawn by the reconcile pass).

**Amended 2026-11-14 (reconcile):** after a committed resize the CPU copy is the grid cell resize built on the CPU
(`ApplyResizedCpuGridCopy`, Task 32), not a `ReadGrid` read-back; row B5 is withdrawn. The CPU copy pair shares one
CPU `grid::GridBuilderBuffers`. `choose_cell_size` gets the CPU copy's tables (replaced, A41: stage 2 calls `grid::ComputeNewCellSizeLevel` on the CPU copy's tables).

**Amended 2026-11-14 (Foundation as built):** the two `gpu` tests write the digests of the grid tables they compare
(discrete tables: the byte comparison with the CPU copy stays).

**Amended 2026-11-15 (A64, A41, A58):** the grid-only marks mirror (M5, row B3) and its test move to Task 39a
(later); the CPU copy no longer serves a callback; the mirrors and the CPU grid copy are pieces of `cpu_run_memory`
(locked for copies in a batched GPU run; the one block in a CPU run).

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/add_new_particles_cpu.cc`, `choose_active_cells_cpu.cc`, `commit_step_cpu.cc`,
  `resize_access.cc` (`CommitResize`), `simulation.h` (`FindCpuGridCopy`), `simulation.cc`, `simulation_state.h`,
  `step_memory_estimate.cc`, `batched_buffers_test.cc`, `step_stages_gpu_test.cc`

**Interfaces:**
- Consumes: Task 33a; `gas::FindMarkerChanges(const MarkerGridInputs&, ...)` and `MarkerChanges` copy fields (gas
  plan Task 20, amended); `grid::BuildGrid` with `GridInputs::active_cell_buffer`; `grid::ReadGrid`.
- Produces: `const grid::StaggeredGrid* FindCpuGridCopy(const Simulation& simulation);` (the CPU copy in a batched
  GPU run, the committed grid itself in a CPU run, null in a one-piece GPU run).
- **Plan decisions** (`plan-decisions.md` "Buffers in batched runs"), in both batched modes:
  - **Grid (M7).** Stage 2 builds the needed cell list on the CPU (particles are there), uploads it into the
    `"step: active cell list"` buffer (one counted copy, 8 B per active cell, B4) and builds the try grid on the GPU
    from it, and a CPU copy of the same grid with the CPU builder from the same list into the free half of `"step:
    grid copy on the CPU"`; version or block count differing is `kStageFailed` (`module = kGrid`). Grouping,
    `grid::ComputeNewCellSizeLevel` (A41; was `choose_cell_size`) and the marker upkeep read the copy. `CommitStep` swaps the copy pair with the grid pair; a
    reject takes the free copy's buffers back with `grid::TakeBuffers`. A committed resize swaps in the CPU resized grid
    cell resize built into `free_cpu_grid_copy_buffers()` and handed over with `ApplyResizedCpuGridCopy` (Task 32):
    no copy (reconcile; B5 withdrawn). The two CPU copy buffers are built with `grid::BuildGridBuffers(settings,
    cpu_grid_builder_buffers)` from one CPU `grid::GridBuilderBuffers` (reconcile item 11). New cells' levels come from `grid::ComputeNewCellSizeLevel(FindCpuGridCopy(simulation)->tables(RunOn::kCpu), ...)` (A41; replaced: "`choose_cell_size` receives ... as `committed` (reconcile item 13)").
  - **Markers (M3).** Before stage 1 (c), when the committed `state_version` differs from the mirrors', one slice
    `ReadBuffer` each of the committed `gas_volume` and `gas_velocity_after_pressure` into `"step: committed gas
    volume on the CPU"` (8 B per cell place) and `"step: committed gas velocity on the CPU"` (8 B per face place):
    two counted copies (B1); then `gas::FindMarkerChanges(MarkerGridInputs{FindCpuGridCopy(sim), &volume_mirror,
    &velocity_mirror}, all, markers, law, settings, gas_buffers)`.
  - **Cells to keep active (M4).** `gas::FindCellsToKeepActive` runs on the amounts' processor into `keep_active`,
    then one counted copy into `"step: keep active on the CPU"` (1 B per cell place, B2) feeds the CPU marking.
  - (Grid-only marks, M5, B3: not in the first version, A64; Task 39a.)
  - **Resize (M14).** `FindCpuGridCopy` is what `CellResize` reads in a batched GPU run instead of its own
    `ReadGrid` copy (cell resize and batches Task 43).
  - Rows (all `where = kCpu` except the list): the four above (A64: was five) with `KeptFor::kWholeRun` (mirrors) or
    `kOneStepTry` (keep active), and the grid copy as `grid::BuildGridMemoryEstimate(RunOn::kCpu, 2, limits.max_cell_blocks,
    grid::BuilderBuffersSharing::kOneForAllGrids)` with `where` set to `kCpu`, only in a batched GPU run (a batched CPU run has every mirror but no grid copy and no list).

- [ ] **Step 1: Failing tests** (`LABEL cpu` unless named; batched CPU layout as Task 33a, copies CPU to CPU):

  ```cpp
  TEST(AddNewParticles, BatchedMarkerInputs_CopiedOncePerCommittedVersion)  // A17 M3: BuildSmallFireScene: the
      // first try's copies hold 2 copies of 8 * cell places + 8 * face places bytes; a rejected try then a retry
      // (stage table rejecting stage 19 once) adds none; the next committed version adds 2 again
  TEST(AddNewParticles, BatchedMarkerChanges_EqualOnePiece)   // A17 M3: to_add and to_remove bytes equal the one-
      // piece run's at the same committed state
  TEST(ChooseActiveCells, BatchedKeepActiveReadOnceCounted)   // A17 M4: one copy of 1 * cell places per try
  TEST(BuildStepMemoryEstimate, BatchedGpu_MirrorsOnCpuAndListOnGpu)   // A17 M3 to M7: RunOn::kGpu batched plan of
      // BuildSmallFireScene: the mirrors and "step: grid copy on the CPU" rows have where kCpu, "step: active cell
      // list" kRunOn; a RunOn::kCpu batched plan has the mirrors and neither grid row
  TEST(FindCpuGridCopy, CpuRunIsTheGridOnePieceGpuIsNull)     // RunOn::kGpu built only as a plan: documented null
  // LABEL gpu, step_stages_gpu_test.cc
  TEST(ChooseActiveCells, BatchedOnGpu_ListUploadCountedAndCopyEqualsGrid)  // A17 M7: copies of stage 2 hold one
      // copy of 8 * active cells to the processor; ReadGrid of the GPU grid equals FindCpuGridCopy byte for byte
      // (discrete tables); WriteResultDigest of the tables (both gpu tests)
  TEST(CommitResize, BatchedOnGpu_GridCopyTakenFromResizeNotCopied)  // reconcile (was GridCopyRefreshedAndCounted):
      // ReadCopyCounts does not grow at the commit; FindCpuGridCopy equals ReadGrid of the resized GPU grid
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Run the CPU stages of a batched run with counted copies`. Body: `CUDA: compiled, not run` (two `gpu` tests). New
  names: `FindCpuGridCopy`.

### Task 34: Solids scene tests the solids plan hands over

**Amended 2026-11-14 (A18):** the snow-beside-water test checks that the water moves the snow faces in the same
step, and the floating springy block of the solids list (item 7) is added.

**Amended 2026-11-14 (A27):** `TryStep, FloatingBoxInSyrup_RestDepthAsInWater` is added (handed by
`plan-decisions.md` "A viscous liquid sticks, and free solids in the viscous solve", section "Step (stage 10,
11)"): a free box floating at rest in a viscous liquid stays where it floats in the same liquid without viscosity.

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/solids_scene_test.cc`
- Modify: `step_test_scenes.h/.cc` (the scenes below; A27, plan decision: `BuildFloatingBoxInLiquidScene(double
  viscosity)`, the scene of viscosity's `ViscosityScene, FloatingBoxInSyrup_SameRestAsInviscid` (its tank, cell
  width `w` and box placement, read from the viscosity plan) as a `ReadyScene`: the water preset (1000 kg/m3,
  0.072 N/m) with its viscosity replaced by the argument, a free box of density 500 and side 0.04 m, axes aligned,
  centred horizontally, at its equilibrium draft 0.02 m, liquid and box at rest)

The list "Tests the step plan must add" at the end of `plans/2026-11-14-rebuild-08-solids.md` (also in
`plan-decisions.md`, "Solids"). Its numbers are the solids design's section 7. `Batches,
FloatingBox_WithinStepToleranceOfOnePiece` (A15 name) and the split-and-merge pass of the cut test belong to the
cell resize and batches plan (they need its batched run and its resize). The swept-volume and plate-leaves tests are run by the
solids plan itself (its Tasks 35 and 36). Work signs as in Global Constraints.

- [ ] **Step 1: Tests** (`LABEL cpu`):

  ```cpp
  TEST(WavePaddle, RegularWave_SpeedAndHeight)          // BuildWaveScene stretched to 3 wavelengths, wave height 2 %
      // of depth, paddle from solids::BuildWavePaddlePath, outlet zone: phase speed within 5 % of
      // omega^2 = g k tanh(k depth); height within 10 %; reflection from the outlet below 10 %; volume drift below
      // 0.1 %; flow_work (positive when the paddle pushes liquid) == wave energy gain + DriveWavesResult
      // energy_added within 15 %
  TEST(Cut, PlateThroughBlock_SidesNeverMix)            // springy block, driven plate; the same run restarted from a
      // snapshot in the step after the first hit, and with a rejected try (stage table) before it: identical bytes
      // to the uninterrupted run; no particle changes side; driven_work == elastic energy gained + cut_work +
      // kinetic energy within 5 %
  TEST(Platen, SnowBesideWater_SetsKeepTheirVelocities) // a platen presses a snow bed with water beside it; two
      // sets; no water face takes the snow's velocity except through CollideSolids steps 2 to 6; the snow faces
      // the water presses on take stage 12's blocking_material_velocity_after_pressure in the same step (bytes);
      // momentum of water + snow + platen changes only by gravity, the domain and the platen's push
  TEST(SpringyBlock, FifthDensity_FloatsAtDisplacedDepth)  // solids list item 7: a springy cube of side 0.1 m and
      // density 200 set on still water in a closed tank 0.4 x 0.4 x 0.3 m, water 0.15 m deep, cells 0.0125 m,
      // steps of 0.01 s through AdvanceTo for 2 s: mean bottom height over the last 0.2 s is 0.02 m below the still
      // surface within 0.00625 m (half a cell); vertical velocity amplitude over the last 0.2 s <= that over 0.2 to
      // 0.4 s (pressure plan Task 49 shows the lagged form unstable at a tenth of this step)
  TEST(FreeBox, FallsThroughGas_NotPushedByGas)         // free box through a gas-filled box: free fall within 1e-13
      // relative while the gas flows around it (marker hits are not summed)
  TEST(FreeBox, HalfDensity_FloatsHalfSubmerged)        // through TryStep and AdvanceTo: 500 kg/m3 cube of 16 cells
      // per side dropped from rest, settling with step-duration control: draught within 2 % of half; linear and
      // turning momentum of liquid plus solid change only by gravity and the sides, 1e-9 relative per step
  TEST(FreeBox, OffCentreSplash_ReactionEqualsLiquidChange)  // through TryStep: a jet (a source) hits the floating
      // box off centre: pressure push plus contact push on the box equals minus the change of liquid momentum
  TEST(TryStep, FloatingBoxInSyrup_RestDepthAsInWater)  // A27: BuildFloatingBoxInLiquidScene(10) (syrup: 1000 kg/m3,
      // 10 Pa s; stage 11, the box in free_solids) and BuildFloatingBoxInLiquidScene(0) (the water preset; stage 12):
      // 200 TryStep calls of 0.01 s each, every one committed; after each step the box centre of the syrup run is
      // within 1e-6 w of the water run's, and the box speed of the syrup run is within the velocity bound
      // 2 * 2e-5 * w (the bound of viscosity's FloatingBoxInSyrup_SameRestAsInviscid)
  ```
- [ ] **Step 2: Run.** If one fails, fix the owning commit (rebase), not the test. **Step 3:** gate. **Commit.**
  Subject: `Test solids against liquid, snow and gas through whole steps`.

### Task 35: Gas scene tests the gas plan hands over

**Amended 2026-11-14 (A18):** the step-limit test names where the sealed limit still applies, and a sealed pocket
above inviscid water is added.

**Amended 2026-11-14 (A24):** no sealed step limit exists any more (`ComputeSealedGasStepLimit` is deleted):
`GasStepLimitTakenIntoNextDuration` keeps only the small fire, and `SealedPocketAboveSyrup_NoSealedStepLimit` is
added (a sealed pocket above a viscous liquid is solved in stage 11).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/gas_scene_test.cc`
- Modify: `step_test_scenes.h/.cc`

The list "Tests the step plan must add" at the end of `plans/2026-11-14-rebuild-09-gas.md` (also in
`plan-decisions.md`, "Gas"). `TryStep, BatchedFireBesideWater_WithinStepTolerance` (A15 name) belongs to the cell
resize and batches plan. The gas-beside-liquid rows of the gas design are run by the gas plan with its own loop.

- [ ] **Step 1: Tests** (`LABEL cpu`):

  ```cpp
  TEST(TryStep, GasHeatPulse_RejectedAndCommittedFramesKept)  // BuildSmallFireScene with the branch's 1e12 W/m3 pulse
      // (gas plan Task 28 data) from frame 2: AdvanceTo to each frame time with on_step_done recording a snapshot
      // per frame; the pulse is rejected after max_retries_per_step shorter tries (kRetriesUsedUp); the snapshots
      // of the frames before it are ExpectSameBytes with a run without the pulse
  TEST(TryStep, GasStepLimitTakenIntoNextDuration)            // A24: the small fire only: every chosen duration
      // after a commit is no longer than that commit's GasResult::step_limit (the flow-share limit); kGas active at
      // least once
  TEST(TryStep, SealedPocketAboveWater_NoSealedStepLimit)      // A18: the heated sealed pocket above inviscid water:
      // GasResult::step_limit has no sealed term (+infinity with no outflow); steps of max_step_duration commit; each
      // committed background pressure == atmosphere + that step's sealed_gas_pressure (exact)
  TEST(TryStep, SealedPocketAboveSyrup_NoSealedStepLimit)      // A24: the heated sealed pocket above syrup (10 Pa s,
      // 1400 kg/m3): GasResult::step_limit has no sealed term; steps of max_step_duration commit; each committed
      // background pressure == atmosphere + that step's sealed_gas_pressure (exact); stage 11 runs (no 12)
  TEST(TryStep, SparseFireKeepsItsActiveCells)                // the small fire in a domain twice as wide, 40 steps:
      // no cell whose temperature differs from ambient by more than 1 K lies within 8 cells of an inactive cell
  ```
- [ ] **Step 2: Run**; fix owning commits if needed. **Step 3:** gate. **Commit.** Subject:
  `Test gas through retries, step limits and a growing active region`.

### Task 36: Reference scenes at small size

**Amended 2026-11-14 (A15):** the estimate-against-build test covers the third layout.

**Amended 2026-11-15 (A58):** `ReferenceScenes_AllocatedEqualsEstimate` checks `handed_out_bytes == reserved_bytes ==
estimate.on_processor` and `handed_out_cpu_bytes == reserved_cpu_bytes` in each layout (name kept).

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/reference_scenes_test.cc`
- Modify: `step_test_scenes.h/.cc` (`BuildSmallSyrupScene`, `BuildFireReferenceScene`, `BuildSmallSnowExample`)

**Interfaces:**
- **Plan decisions,** small sizes (programme design 8a; no design fixes the wave's): wave = `BuildWaveScene(1)`
  of Task 27; syrup = the viscosity design's decision 6 scene (domain 0.4 x 0.4 x 0.3 m, 10 Pa s, 1400 kg/m3,
  0.08 N/m, a 2 cm stream at 0.3 m/s from 0.25 m, floor only) at one cell width of 5 mm, the source rate
  `pi * 0.01^2 * 0.3 / (0.005^3 / 8)` particles per second; fire = `BuildFireReferenceScene(0.1)`, the gas design's
  fire reference scene (gas design 7, last paragraph) at one cell width of 0.1 m (16 x 16 x 32 cells); snow example = a snow
  cube of 0.2 m dropped from 0.1 m onto the floor, cells 2.5 cm (the Blender add-on's fourth preset).

- [ ] **Step 1: Tests** (`LABEL cpu`):

  ```cpp
  TEST(ReferenceScene, SmallWave_CommitsWithVolumeKept)    // design 7 "Four reference scenes": AdvanceTo 0.2 s;
  TEST(ReferenceScene, SmallSyrup_CommitsWithVolumeKept)   //   every step commits after retries; liquid volume
  TEST(ReferenceScene, SmallFire_Commits)                  //   change within pressure's reported volume change
  TEST(ReferenceScene, SmallSnow_Commits)
  TEST(BuildSimulation, ReferenceScenes_AllocatedEqualsEstimate)  // design 7 "Estimate against build": the four,
      // one-piece layout, batched layout (plan forced to kBatched with a hand-written run_stages) and, A15, the
      // batched layout with solves in pieces (forced, max_solve_loaded_blocks = 3 layers)
  TEST(ReferenceScene, RepeatRun_IdenticalBytes)           // each scene twice, 10 steps
  ```
- [ ] **Step 2: Run.** **Step 3:** gate. **Commit.** Subject: `Run the reference scenes at small size`.

### Task 37: Comparison with the proof of concept

**Kind:** integration

**Files:**
- Create: `native/fluidity/step/testdata/proof_of_concept.json`, `native/fluidity/step/proof_of_concept_test.cc`

- [ ] **Step 1: Produce the old numbers** (supporting only). Export `main` outside the worktree as
  `plan-decisions.md` says; build its CPU preset; a throwaway program linked to `main`'s library runs
  `LiquidSubstepScheduler3D` on: free fall (the case of `test_liquid_substep_3d.cc` `LiquidGravity3D`), the
  column at rest, and a dam break (water block 0.2 x 0.1 x 0.2 m in a 0.6 x 0.1 x 0.3 m closed box, same-size
  cells of 1 cm) to 0.5 s. Record front position, liquid volume and kinetic energy at 0.1 s intervals, rotated
  from Y-up with rows `{1,0,0},{0,0,-1},{0,1,0}`, with `main`'s commit hash. The program is not committed; its
  outline goes into the commit body.
- [ ] **Step 2: Test** (`LABEL cpu`) `ProofOfConcept, DamBreakFrontVolumeEnergy`: front within one cell, volume
  0.1 %, kinetic energy 2 % (design 7); `ProofOfConcept, FreeFallAndColumn`. Every quantity here is also covered
  by an analytic test above.
- [ ] **Step 3:** gate. **Commit.** Subject: `Compare whole steps with the proof of concept`.

### Task 38: GPU scene runs

**Amended 2026-11-14 (Foundation as built):** CPU against GPU is within tolerance (never bytes); every `gpu` test
writes result digests; the two test files are registered in the step's `CMakeLists.txt`.

**Kind:** device

**Files:**
- Create: `native/fluidity/step/step_scene_gpu_test.cc`, `step_scene_no_gpu_test.cc`
- Modify: `native/fluidity/step/CMakeLists.txt` (`LABEL gpu` / `LABEL no_gpu`, `LINKS fluidity_base_test_support
  fluidity_step_test_support`)

- [ ] **Step 1: Tests.** `LABEL gpu`, each writing `WriteResultDigest` of the committed positions, velocities and
  `GridFields` after its last step: `TryStep, OnGpu_EqualsCpuAfterOneStep` for each scene of Tasks 21, 25, 26,
  28, 29, 30, 31, 34 and the solids plan's Tasks 35 and 36 scenes (discrete outputs equal; floating-point within
  design 6.4 tolerances with `CheckWithinTolerance`; solids design 6 for the solid states); `TryStepWithStages, OnGpuRejectAfterEveryStage_ChangesNothing`;
  `ReferenceScene, OnGpuFiftySteps_RepeatRunSameBytes` (the four small scenes); `AdvanceTo, OnGpuManySteps_SumsAgree`
  (liquid volume 0.1 %, kinetic energy 2 %, front one cell); `TryStep, OnGpuRunOnCallFailed_SimulationUnusable`
  (a stage table returning `kRunOnCallFailed`). `LABEL no_gpu`: `CheckSceneCanRun` and `BuildRunPlan` with
  `RunOn::kGpu` give the CPU answers; `BuildSimulation` with `kGpu` gives `kRunOnCallFailed` with nothing
  allocated (repeat of Task 9 inside the CUDA build).
- [ ] **Step 2:** gate. **Commit.** Subject: `List the step's GPU scene runs`. Body: `CUDA: compiled, not run`.

### Task 39: Comment pass and hand-over

**Amended 2026-11-14 (A31):** the hand-over records the step's working buffers in `TryArrays`.

**Amended 2026-11-14 (Foundation as built):** the `docs/gpu-runs.md` row also lists Task 32's `gpu` test; the
hand-over records the step's working buffers in `TryArrays` and the same-bits promise of `ComputeVelocityAfterGravity`.

**Kind:** mechanical

**Files:**
- Modify: any file of `native/fluidity/step/`, `docs/method-paper/sections/step.typ`, `docs/glossary.md`,
  `docs/gpu-runs.md`, `tools/check_rules_allowed_names.txt`

- [ ] **Step 1: Comment pass.** Reread every file of the module as a C++ programmer who has never read a fluid
  paper: comments true of the final code; why, not what; a worked example with numbers for `TryStep` (the
  falling block: one step of 0.01 s, what each stage writes), `ComputeStepDuration` (the 0.035 s example) and
  `BuildRunPlan`; every word in the glossary; `Uses` / `Used by` lists; `// Step N:` numbers match
  `<proc-try-step>`, `<proc-advance-to>`, `<proc-compute-step-duration>`, `<proc-apply-limits>`; the comment
  above `kStages` lists all twenty entries.
- [ ] **Step 2: Fold each change into its owning commit** (`git commit --fixup`, `git rebase --autosquash`); run
  format and rules on each rewritten commit and the full gate on the tip.
- [ ] **Step 3: Check the design section by section** (3 to 10) and record what was delivered differently: no
  `kLiquidsThatDoNotMixTouch` and no `kViscousLiquidWithSnow` (A24; count 16 at most since A27, replaced: 17); stage
  11 with `SolvedCouplingArrays` (A24; `ApplyPressureToMaterials` there replaced by A26); no `SealedGasCoupling`
  argument (A24); the viscous
  path fixed once by the A25 rule and `RunPlan::liquid_viscous_effects`, `viscous_path` (A25); `StageOutcome` declared in `step_error.h`; the private
  `ToStageOutcome`; the support and suggested-change table of Task 6; the `RandomWords` form; the stage-2 fill
  of `SolidsAtFaces` when stage 3 is skipped; `stage_runs` fixed per simulation; the transfer buffers row if
  added; the small reference scene sizes; which viscosity exclusions were lifted and the resulting
  `kCapabilityReasonCount`; `BuildViscosityStageInputs`; `BuildSmallFireScene` matching the gas plan's; tests assigned to the
  cell resize and batches plan (`Batches, FloatingBox_WithinStepToleranceOfOnePiece`, `TryStep,
  BatchedFireBesideWater_WithinStepTolerance`, the split-and-merge part of the cut test); the A26/A27 changes (stage 10's
  `BuildViscousMaterialFaces` and `ToFreeSolidsForViscousSolve`, stage 11's four `SolvedCouplingArrays` spans,
  `ApplySolvedMotionToSolids` and `ApplyViscousSolveToMaterials` in place of `ApplyPressureToSolids` and
  `ApplyPressureToMaterials`, the three step-owned rows (with `EquationInputs::free_solid_motion_before_pressure`), no `kViscousLiquidWithFreeSolid` (count 16 at most) and no
  `kSetViscosityToZero` (`kSuggestedChangeCount` 12), `FreeBoxPlace`, `BuildFloatingBoxInLiquidScene`); the A15
  changes of Tasks 8, 33 and 36 (`RunMode::kBatchedWithSolvesInPieces`, `max_solve_loaded_blocks`,
  `allowed_cpu_bytes`, the
  solvers' batched layout, the loaded try's piece `PressureBuffers`); the A17 batched-run buffer changes
  (amended 2026-11-14): `PlanFit` and a plan returned when it does not fit, `on_cpu` and `allocated_cpu_bytes`,
  `CopyCounts` with `StepReport::copies`, `ReadCopyCounts` and `planned_step_copies`, the GPU cell list of Task 14,
  Appendices A and B, `ArrayPlace`, the loaded module instances, the CPU solids buffers of stage 17,
  `FindContactReactionBuffers`, `FindCpuGridCopy`, the mirrors; Foundation as built (amended 2026-11-14):
  `TryArrays::sum_buffers` and `order_buffers` with their four memory entries, the `_steps.h` headers and their
  per-thread `cpu` tests, the same-bits promise of `ComputeVelocityAfterGravity`, `fluidity_add_shared_test`
  retired.
- [ ] **Step 4: `docs/gpu-runs.md`:** a `step` row, `compiled, not run`, listing every `gpu` test of Tasks 14,
  16, 20, 27, 29, 32, 33b, 38 (amended 2026-11-14, Foundation as built: Task 32's test added; each writes result
  digests, so the hardware run's repeat check covers it).
- [ ] **Step 5: Hand over** in one message: `git log --oneline` of the module, the gate's output, new public names
  with reasons, the plan decisions, and the statement that no device code has run. **Amended 2026-11-15:** the
  hand-over also lists the A40-A64 changes (the run's blocks and the `MemoryReport` fields, `FindRunMemory` /
  `FindCpuRunMemory`, `ComputeAllowedMemoryBytes`, stage 2's `grid::ComputeNewCellSizeLevel`, `largest_liquid_speed`,
  the three cell-size arrays, `SceneCounts`' depth estimate, the A57 advice fields, `liquid_force_on_solids`,
  `forced_mode`, `kOutOfMemory` last), `StepCallbacks::limits_raised` (reconcile, 2026-11-15), and that Task 39a (later) is not done.

---

## Deep liquid without particles (later; needs review)

Excluded from this module's done-criteria (owner answer A64; `plan-decisions.md` "Owner answers A40-A64 applied",
item 1). The owner of this group is the cell resize and batches plan's group "Deep liquid without particles (later;
needs review)": an independent review of the grid-only design comes first, and that group's two gate tests
(`GridOnlyRegion, StandingWave_KeepsAmplitude`, `GridOnlyRegion, ParticleBandEdge_StaysStable`) must pass before this
task lands. Design text kept and marked "not in the first version (A64)" in design 3.6, 3.9, 3.10, 5.2, 5.3, 5.5,
5.8, 5.11, 6.5.

### Task 39a (later): The step's share of the grid-only region

**Amended 2026-11-15 (A64):** new task holding what A64 took out of Tasks 13, 15, 19, 22, 23, 33b and Appendices A
and B.

**Kind:** integration

**Files:**
- Modify: `native/fluidity/step/advance_to.h` (three callbacks), `simulation.h` (`CommittedStepLimits::
  grid_only_speed`, the six snapshot vectors), `resize_access.h` (six `GridFields` arrays), `choose_active_cells_cpu.cc`,
  `transfer_stages.cc`, `move_particles_cpu.cc`, `commit_step_cpu.cc`, `step_duration.cc`, `step_memory_estimate.cc`,
  `stage_table.cc` (`run_whole_grid[kMoveParticles]`), their tests

**Interfaces:**
- Produces, as the design's marked text gives them: `StepCallbacks::apply_grid_only_region` (end of stage 4),
  `move_grid_only_liquid` (end of stage 16, the whole-grid part of 16), `read_largest_grid_only_speed` (stage 20,
  into `CommittedStepLimits::grid_only_speed`); the `kParticleTravel` term `kMaxTravelInCellWidths / grid_only_speed`;
  `GridFields::is_grid_only_region`, `layers_from_grid_only_region`, `grid_only_mass`, `grid_only_rest_volume`,
  `grid_only_current_volume`, `grid_only_ingredient_volume` (34 + 8 x ingredients B per cell place, 2 copies);
  stage 2's mark of committed grid-only cells; the batched mirror `"step: committed grid-only marks on the CPU"`
  (Appendix A M5, Appendix B B3).

- [ ] **Step 1: Failing tests** (`LABEL cpu`), the ones A64 moved here:

  ```cpp
  TEST(ParticlesToGrid, ApplyGridOnlyRegionCallback_GetsItsContextAndRange)  // hand-written callback, one range
  TEST(MoveParticles, MoveGridOnlyLiquidCallback_GetsItsContext)  // hand-written; its kParticleTravelledTooFar
                                                                   // outcome is reported as it is
  TEST(ComputeStepDuration, GridOnlySpeed_IsActive)    // grid_only_speed 200 1/s: 0.5 / 200
  TEST(AdvanceTo, ReadLargestGridOnlySpeed_StoredAtCommit) // hand-written callback returns 7.5: committed 7.5
  TEST(ChooseActiveCells, BatchedGridOnlyMarksReadOncePerVersion)  // A17 M5: BuildDeepWaterScene with cell resize:
      // one copy of 1 * cell places per committed version, none on a retry
  TEST(BuildStepMemoryEstimate, CellResizeAddsGridOnlyArrays)  // 34 + 8 x ingredients bytes per cell place, two
      // copies, beside the three cell-size arrays
  ```
- [ ] **Step 2: Run** (fail). **Step 3: Implement.** **Step 4:** gate. **Commit.** Subject:
  `Hand the grid-only region its seams in the step`.

---

## Appendix A: buffers by processor in the three run modes (A17, 2026-11-14)

Cited by Global Constraints and by the tasks marked "Amended 2026-11-14 (A17, batched-run buffers)". For a **GPU
run**. In a CPU run every buffer and every stage is on the CPU and nothing here is a mismatch (a batched CPU run
still makes the counted copies of Appendix B rows B1 and B2, CPU to CPU; B3 is in Task 39a, A64). Columns: **1** one piece (`kOnePiece`),
**B** `kBatched`, **P** `kBatchedWithSolvesInPieces`. Places: `G` the GPU, `C` CPU memory, `L` the loaded instance
on the GPU (one batch at a time), `G+C` both (a planned copy). "Code" is where the stage's work runs; a CPU stage
may launch GPU work on GPU buffers (marked "launches"). The resolution column names the mismatch (M1 to M15) and
how it is closed: **copy** (planned, counted, Appendix B or the named module report), **move buffer**, **move
work**. Rows without an M are consistent as listed.

| # | Stage (code: 1 / B, P) | Buffer (owner) | 1 | B | P | Mismatch and resolution |
|---|---|---|---|---|---|---|
| 1 | 1 `AddNewParticles` (G, launched from the CPU / C) | source counters (step) | C | C | C | — |
| 2 | 1 (a) | `NewParticles` spans (step) into the particles try (particles) | C into G | C into C | C into C | **M1** one piece: `AddParticles` writes the waiting list with `WriteBuffer`: **copy**, counted from `AddParticlesResult::copy_count`, `bytes_copied` (particles Task 11; step Tasks 12, 14; B row O1) |
| 3 | 1, 2, passes | particles, committed and try (particles) | G | C (locked) | C | the reason for batches; passes load them (`BatchReport`) |
| 4 | 1 (b) | open-sea site bits, removal list (step) | G | C | C | **M2** one piece: marked on the GPU, read back: **copy** (O2); batched: marked on the CPU where the particles are, no copy (Task 27) |
| 5 | 1 (c) | committed gas amounts (gas) | G | G | G | **M3** batched: the upkeep needs `gas_volume` on the CPU: **copy** into a mirror once per committed `state_version` (B1, Task 33b) |
| 6 | 1 (c) | committed `gas_velocity_after_pressure` (`GridFields`) | G | G+C | G+C | **M3** as row 5 (mirror, B1) |
| 7 | 1 (c), 2 | committed and try grid tables (grid) | G | G+C | G+C | **M7** batched: CPU copy built by the CPU builder from the same cell list (no copy; A17 equal tables), swapped and released with the grid; after a resize, cell resize's CPU-built resized grid becomes the copy (no copy; reconcile, B5 withdrawn) |
| 8 | 1 (c) | `MarkerChanges` lists (gas) | C | C | C | **M3** one piece: computed on the GPU, read back into `kCpu` rows: **copy** (`MarkerChanges::copy_count`, `bytes_copied`; O3); batched: CPU form `FindMarkerChanges(MarkerGridInputs, ...)` on the mirrors and the CPU grid copy (gas Task 20, step Task 33b) |
| 9 | 2 `ChooseActiveCells` (G / C, launches) | needed cell list (step, `"step: active cell list"`) | G | C, sent to G | C, sent to G | **M6** one piece: the list stays on the GPU, only its count comes back (A17; Task 14 corrected); **M7** batched: built on the CPU, sent: **copy** (B4) |
| 10 | 2 (a) | `keep_active` (step row, written by gas) | G | G+C | G+C | **M4** batched: `FindCellsToKeepActive` runs on the GPU (**move work**), result read back: **copy** (B2) |
| 11 | 2 (a) | committed `is_grid_only_region` (`GridFields`) | — | — | — | not in the first version (A64; Task 39a): **M5** batched, cell resize: mirror once per committed version: **copy** (B3) |
| 12 | 2 (b) | new cells' levels: `grid::ComputeNewCellSizeLevel` on the committed tables | G | C (grid copy) | C (grid copy) | no mismatch (A41): computed in the marking pass on the tables' processor, no copy. Replaced: "`choose_cell_size` inputs ... **M8** one piece: new cells read back, levels sent: **copy** (O4) ... Open for the owner (a shared rule instead of a callback)" |
| 13 | 2 (c) | try `GridFields`, `carried_pressure`, the six cell arrays (step) | G | G (launches the carry) | G | — |
| 14 | 2 (d) | grouping tables `group_begin`, `cell_begin` (particles) | G | C | C | **M7** batched: grouped on the CPU against the CPU grid copy |
| 15 | 2 (e) | gas try amounts, `gas_seen` (gas) | G | G (launches) | G | — |
| 16 | 3 to 9 pass 1 (G / L) | `TransferBuffers`, `MaterialBuffers`, `ContactBuffers`, `SolidsBuffers` used by pass stages (transfer, materials, contact, solids) | G | L at loaded counts | L | **M9** **move buffer**: loaded instances in the loaded `StepTry`; whole instances keep zero particles (Task 33a) |
| 17 | 3, 14 | `SolidsAtFaces` start and end (solids rows, step-owned) | G | C whole + L | C whole + L | placement rule (stage 2 fill, resize reads the end copy) |
| 18 | 3 | `grid_edge_is_open_to_sky` (solids row, step-owned) | G | G whole | G whole | **M11** its flood needs the whole grid but stage 3 runs per batch: **move work** to stage 3's `run_whole_grid` part, on the GPU right after pass 1 (batches Tasks 35, 37) |
| 19 | 4 to 10 | each set's `GridFromParticles`, `CellsFromParticles`; coverage; geometry; the six cell arrays (transfer, solids, pressure, step) | G | G whole + L | G whole + L | **M12** placement rule: read by stages 10 to 13 on the GPU, so whole on the GPU (Task 33a) |
| 20 | 5, 14, 15 | `before_forces`, `after_pressure` per set (transfer) | G | C whole + L | C whole + L | placement rule: carriers between passes |
| 21 | 8, 17 | `CollideSolids` block partials (solids) | G | L, then C (stage-17 instance) | same | **M10** written in pass 1 on the GPU, read by stage 17 on the CPU: **copy** by `solids::WriteBlockPartialsRange` in `ReadBatch` (counted in `BatchReport`) |
| 22 | 9 range part | piece `PressureBuffers` (pressure) | — | L | L | A15 (step Task 33) |
| 23 | 9 whole, 10 to 13 | `PressureBuffers` liquid and gas, `ViscosityBuffers`, `GasBuffers` (pressure, viscosity, gas) | G | G | G rows + solver buffers C | A15: working buffers streamed by `SolveBatches`, counted in `SolveCopyReport` / `BatchReport` |
| 24 | 10 to 12 | `SolidsBuffers` motion map, blocking faces, free-solid entries, and (A26, A27, viscous scene) the `ViscousMaterialFaces` and `FreeSolidForViscousSolve` entries (solids whole instance); the step-owned spans of stages 11 and 12, with (A26, A27) `"step: material face velocity after solve"`, `"step: free solid motion"` and `"step: free solid motion before pressure"` | G | G | G | — |
| 25 | 15, 16 | `GridAtParticle`, `ParticleContact` (step) | G | L only | L only | placement rule: one pass uses them |
| 26 | 16, 17 | `SolidReaction` per particle (step) | G | L, then C whole | same | placement rule; read back with the pass (`BatchReport`) |
| 27 | 17 `MoveSolids` (G / C) | stage-17 `SolidsBuffers`: reaction keys and order, block partials, per-solid results (solids) | G (results C) | C (third instance) | C | **M10** **move buffer**: a CPU `SolidsBuffers` for `ApplyContactReactions` and `ComputeCutWork` (step Task 33a; solids Tasks 13, 28) |
| 28 | 18 pass 2 | `FoamAndSprayInputs::block_in_whole_grid`; the recorder's device working buffers and its kept records (render output) | G (span empty) | L, map copy C | same | **M13** a GPU piece has no CPU map: batches passes its CPU map copy through `ApplyGridPiece` (batches Tasks 36, 39); the recorder's read-back is a **copy** counted by `ReadRecorderCopyCounts` (render output Task 32) |
| 29 | 19 whole part | block partials of pass 2 for `CheckSolidExchange` and the next limits (step) | G | G whole | G whole | placement rule |
| 30 | 20 `CommitStep` (G / C, launches) | swaps; `gas::CommitTry`; `ApplyPlateRemoval` on particles | G | grid pair G + copy pair C swapped; gas G; particles C | same | **M7** the copy pair is swapped with the grid pair (Task 33b) |
| 31 | 21 `ResizeCells` | `CellResize` grid copy and mirrors (cell resize) | C (its `ReadGrid` copy, counted) | C (the step's copy) | same | **M14** batched: reads `FindCpuGridCopy` instead of its own `ReadGrid` and builds the resized CPU grid into `free_cpu_grid_copy_buffers()` (batches Task 43; reconcile: its own two CPU grid entries are not allocated in this layout); its copies stay in `ResizeCellsReport` |
| 32 | between steps | `ReadSnapshot` for frames and saved states (step) | G to C | C, `GridFields` G to C | same | **M15** **copy** per frame, counted by I/O (`RunCost::snapshot_bytes_per_frame_mean`, budget `snapshot_bytes_per_frame`) |
| 33 | every stage | status records, solver progress, `AirPocket`, `BlobHistory`, `GasBlob`, per-solid results, box marks | G to C | same | same | the one uncounted class (bounded; `plan-decisions.md` "Copies counted and the one exception") |

## Appendix B: the step's own planned copies (A17, 2026-11-14)

Counted in `StepReport::copies` and `ReadCopyCounts`; `BuildRunPlan` fills `RunPlan::planned_step_copies` with the
sum of the rows of the plan's mode at the limits (upper bounds per try). GPU runs; rows B1 and B2 (B3: Task 39a, A64) also in a batched
CPU run (CPU to CPU). `cell places` and `face places` are at `limits.max_cell_blocks`; `A_m` is the material's bytes
per new particle (the sum over its carried `NewParticles` arrays of the element size); `k_m` is the number of those
arrays.

| Row | Mode | When | Direction | Bytes (at most) | Copies |
|---|---|---|---|---|---|
| O1 | one piece | each `AddParticles` call (sources, open-sea refill, markers) | to | `max_added_per_try_m * A_m` per material | `k_m` per material with additions |
| O2 | one piece, open sea | stage 1 (b) | from | `ceil(sites / 8) + 8 * max_added_per_try` (open-sea material) | 2 |
| O3 | one piece, gas | stage 1 (c) | from | `max_added_per_try_gas * A_gas + 8 * max_particle_count_gas` | 2 |
| ~~O4~~ | withdrawn (A41, 2026-11-15): no size-choice copy; was one piece, `choose_cell_size`, `8 * cell places` from, `1 * cell places` to | | | 0 | 0 |
| B1 | batched, gas | stage 1 (c), new committed version | from | `8 * cell places + 8 * face places` | 2 |
| B2 | batched, gas | stage 2 (a), each try | from | `1 * cell places` | 1 |
| ~~B3~~ | not in the first version (A64, 2026-11-15; Task 39a): batched, cell resize, `1 * cell places`, 1 copy | | | 0 | 0 |
| B4 | batched GPU | stage 2 (b), new grid version | to | `8 * cell places` | 1 |
| ~~B5~~ | withdrawn (reconcile): the CPU copy after a resize is cell resize's CPU-built grid, no read-back | | | 0 | 0 |
