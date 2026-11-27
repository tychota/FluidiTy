# Changes to apply to the fourteen module designs

Written 2026-11-14 after all fourteen designs existed. It collects (1) the owner's answers, (2) the decisions
on points where two designs disagreed, and (3) every change one design asked of another. Each design's owner
applies the entries under its own heading. Where this file and a design disagree, this file wins; where this
file and `owner-answers.md` disagree, `owner-answers.md` wins.

Rule for every question not listed in `owner-answers.md`: the recommended option is taken (owner's instruction).
So each design's section 11 becomes "Decisions": every question keeps its text, then one line `Decision:` with
the option taken and who decided (owner answer An, or "recommended option, owner's standing instruction").
A design must no longer say "until the owner answers".

## Applies to every design

- **A1.** `PerParticleVelocity<T> { T for_next_step; T for_this_step; }`. Replace `velocity.state` by
  `velocity.for_next_step` and `velocity.travel` by `velocity.for_this_step` everywhere, including prose
  ("state velocity" becomes "the velocity for the next step"). Decided names that contain `travel`
  (`own_motion_in_travel`) are unchanged.
- **A2.** Error with context is `<Module>Failure`; functions return `std::expected<T, <Module>Failure>`.
  Every error enum starts with `kNone`; "the chosen processor cannot be used" is `kRunOnCallFailed` (Q11).
- **A3, A4.** `place` words and the plain material words are final; remove alternatives from the text.
- **A6.** The two solves are `pressure::SolvePressure` (scenes with no viscous ingredient) and
  `viscosity::SolveViscosityAndPressure` (replaces `ApplyViscosity`, same parameters). Any design that calls or
  mentions `ApplyViscosity` uses the new name. The step stage formerly "ApplyViscosity" is
  `SolveViscosityAndPressure`.
- **A9.** Sand is not a material of the main sequence. No design except `15-sand-design.md` declares, tests or
  depends on sand. Where a design listed "snow and sand", it now says "snow" and, if sand needs something
  specific from it later, lists that under a short heading "For the sand module" in its section 4.
  `MaterialKind` is `{ kLiquid, kGas, kSnow, kSpringySolid }`; the sand module adds `kSand`.
- **A14.** Playback and render products are written in Alembic, USD or OpenVDB (details under 12 and 13).
- **Stable error codes.** Every module provides `std::string_view ToName(<Module>Error error)` beside
  `ToMessage`: the enumerator's name without the `k` in lower snake case (`solid_too_thin`). The CLI and the
  add-on use it as the stable code.
- **Sums over ranges.** Any total a design sums "over ranges" is computed as one partial per cell block (in the
  grid's fixed block order) followed by `math::TreeSum` over the block partials, so the result is identical
  whether the run is in one piece or in batches.
- **Memory estimate entries.** Every module exposes `Build<Module>MemoryEstimate`.
- **Cancel.** `cancel_requested` is never a field of a `...Settings` type (settings are hashed). Long-running
  solves take it as a separate input `const std::atomic<bool>* cancel_requested` (may be null) and return
  their module error `kCancelled` when it is set.

## Additions to Foundation `base/` (beyond `shared-interface.md` section a)

- `math::Expm1(double)` in `scalar_math.h` (last-bits group, not bit-identical between CPU and GPU).
- `CopyBuffer(const Buffer<Value>& from, Buffer<Value>& to)` on the processor both live on; slice forms of
  `CopyBuffer`, `WriteBuffer`, `ReadBuffer` taking `(first, count)`.
- `enum class CpuMemory { kOrdinary, kLockedForCopies }` as an optional argument of `BuildBuffer` for CPU
  buffers that are copied to the GPU every step (batched run).
- `std::expected<std::uint64_t, BufferError> ReadMemoryAvailable(RunOn run_on)`.
- `ProcessorDescription ReadProcessorDescription(RunOn run_on)` (model, driver, runtime versions as strings),
  host only, used by the run fingerprint and the run report.

## 01 Contact

- A1, A2, `ToName`. No sand. `contact::BuildContactSolids` is not contact's: see 08.
- Provide what solids, step and render output read: nothing new beyond its section 3.
- **Contact as built (2026-11-19).** Module 1 is built (branch `rebuild`, tip `f2375124`); design sections 3.5 and
  6 now describe it, with its known limits. What it changes for later designs is under 08 Solids, 10 Step, 11 Cell
  resize and batches and 01b Mesh solids below, each headed "From contact as built (2026-11-19)".

## 02 Grid

- From step: values that are built for a limit and bound to a grid version without allocating:
  `BuildCellValues(run_on, max_cell_blocks)` (and the face, corner, block forms), `ApplyGridVersion(values,
  grid)`, and `GridStorage` so `BuildGrid` writes into storage sized for `max_cell_blocks` and allocates
  nothing per version. Same for `GridPiece` (batches).
- From module 11 (batches): `GridPiece` keeps the grid's fixed block order (never "range blocks first");
  add `FindBorderBlocks(grid, range, border_width)` for stacked borders.
- Glossary row for `place`.

## 03 Particles

- A1 in every array name and view.
- A10 / gas: `RemoveReason::kMarkerNotNeeded`.
- Conflict C2 is settled as module 11 designed it: `AdjustParticleCount`, `SplitParticle` and `MergeParticles`
  take two grids (before and after the resize) and two `NeighbourhoodInputs`; they are called inside the resize
  transaction. A particle that drifts across an existing size change is never split or merged; say so.
- From batches: `WriteParticleGroups` and `ReadParticleGroups` (bulk copy of the particles of a block range
  between CPU and GPU buffers).
- From I/O: spell out `ParticlesSnapshot::Material` member by member.
- From render output: state that an output snapshot carries `position`, `velocity.for_next_step`,
  `volume_ratio`, and the constant arrays (`id`, `rest_volume`, `size_level`, ingredient fractions).
- A9: remove sand from extra-field examples; snow keeps `springy_deformation` and `permanent_volume_change`.

## 04 Transfer

- A1.
- A8: materials that do not mix are separate **material sets**. `ParticlesToGrid`, `GridToParticles`,
  `SaveGridBeforeForces`, `SaveGridAfterPressure` and `ComputeCellsFromParticles` take a material-subset input
  (`std::span<const std::uint8_t> material_is_counted`, one entry per material) and are called once per set;
  each set has its own `GridFromParticles` and snapshots. State the memory cost per extra set.
- A10: the same input keeps gas markers out of the liquid's per-cell sums.
- From batches: `FaceValueKind::kFromGridOnlyRegion`.
- A5: nothing depends on a pressure hook; `kFromSolve` is recorded for faces either solve wrote.

## 05 Pressure

- A5: remove `VelocityResponse` and `connection_scale` from `SolvePressure`. Keep exposed the pieces
  `viscosity::SolveViscosityAndPressure` calls (`ComputeFaceInertia`, `ComputePressurePush`, `ApplyPressure`,
  `ComputeFlowImbalance`, `CheckFlowImbalance`, the stop rule, the ladder as a preconditioner).
  `SolvePressure`'s docstring says: for scenes with no viscous ingredient.
- Conflict C3 (recommended): add `CellEighthGeometry` (liquid volume per cell eighth) and `FacePieceCentre`
  (the two liquid-covered centres), filled only when the scene has a viscous ingredient.
- Conflict C4 (recommended): pressure computes one distance field per liquid material; the sharp boundary
  between two liquids that do not mix is declared now and its first commits may ship behind a capability
  exclusion.
- From solids: `ComputeBlockedCornerAreas` (the shared clipping formula) and `ComputeEquationIndexes`.
- From gas (A10): `ComputeBlobOfCell` with `kNoBlob`; `EquationInputs::sealed_blob_check` of type
  `enum class SealedBlobCheck { kWhenBuilt, kWhenRatesApplied }`; `CellContent::kGas`.
- From batches: delete `EquationInputs::border_cell_pressure` and `kLiquidToBorderCell`: the solves always run
  on the whole grid.
- Cancel input as in "every design"; error value `kCancelled`.
- From step: `SolvePressure` writes pressure and air pockets into arrays the step passes (try copies), so
  "kept only on commit" needs no extra copy. If the design already returns its own buffers, change it.

## 06 Viscosity

- A5, A6: this module owns the joint solve; the function is `SolveViscosityAndPressure`. Remove any text
  written for the other options.
- Q28 (recommended): viscous liquid always sticks to solids.
- Uses pressure's `CellEighthGeometry` and `FacePieceCentre` (C3).
- Cancel input; `kCancelled`.
- A9: the exclusion "snow or sand" reads "snow".

## 07 Materials

- A9: move everything about sand (response, return, limits, friction angle and cohesion as sand settings, the
  eight-step diagnosis, the sand tests, `ComputeSandResponse`, `AtSolidFaces` behaviour for sand) into a new
  file `15-sand-design.md` written with the same eleven sections, as module 13 of the build order ("Sand",
  after the Blender add-on). `07` keeps liquid, gas, snow and springy solid, the support rule, the work-rate
  oracle, and names the extension points sand will use. `snow_and_sand_material.h` becomes `snow_material.h`;
  the sand module adds `sand_material.h`.
- From gas: the gas material (one ingredient; viscosity 0, friction 0, surface tension 0; rest density equal
  to ambient; cannot be squeezed; own-motion shares 0 and 0), `ingredient_rest_density` by scene ingredient
  index, and which ingredient is the gas.
- From render output: one export name per ingredient in the ingredient table (used for `share_<name>`).
- From batches: `InternalForceBudget` is summed per block.
- Materials question 6 (recommended): one surface tension against air per scene, else `kSurfaceTensionsDiffer`.

## 08 Solids

- A8 is now decided: write the design for material sets only (remove the "degrades to one set" branches).
- A9: blocking materials are snow and springy solids.
- Recommended options stand: one unbounded plate per scene; pressure against a blocking material returned one
  step late; the piston receiving volume as a known-open test.
- From Blender: `CheckSolidsCanRun` returns every broken scene rule (a list with `solid_id` each), not the
  first; the plate and free-solid limits are public constants.
- One signature for `CheckSolidsCanRun`: `CheckSolidsCanRun(const SolidsSettings& settings, double
  smallest_cell_width, double largest_cell_width)`; step uses this one.
- From render output: `std::vector<contact::Solid> BuildContactSolids(std::span<const SolidSettings>,
  const SolidsSnapshot&)`.
- From I/O: `Vec3 ComputeRotationVector(Orientation from, Orientation to)`.
- From gas: `pressure::SolidCoverage` over every active cell (also where there is no liquid);
  `grid_edge_is_open_to_sky`. A free solid moves gas but is not pushed by it; say so.
- From batches: per-solid sums are independent of where batches are cut (block partials, then tree sum); the
  "merge push lists across batches" duty is removed.
- `DriveWaves`: the step calls it (solids question 8, recommended).
- **From contact as built (2026-11-19).** Each fact below is in the contact headers (`native/fluidity/contact/`).
  - *Reacting solids.* 5.1 (`reports_reaction = true` unless fixed) and the stage text that shortens the step on
    `kSecondReactingSolid` meet a limit: one particle record holds one reacting solid. When one response at the start
    of a step pushes two reacting solids (a particle in the crease of two driven shapes that touch), the failure has
    `can_shorter_step_cure` false and recurs at every step length. Shorten the step on `kSecondReactingSolid` only
    when `can_shorter_step_cure` is true. Whether driven solids report a reaction at all is an open owner question
    (final review of module 1, question 2); until it is answered, say what a driven solid's reaction is for.
  - *What `MoveParticles` writes.* After a failed call the records and reactions are unspecified;
    `ApplyContactReactions` reads them only from a try that succeeded. `SolidReaction` is 56 bytes: `solid_index` at
    0, `hit_position` at 8 (where the first change against this solid happened), `velocity_for_next_step_change` at
    32 (the sum of every change this solid caused in the step).
  - *Limits a solid must meet.* Each solid's origin over the step plus its reach must lie within 2^40 contact
    tolerances of the origin on every axis, and its linear and angular velocity, each summed over its axes, at most
    2^480; otherwise `MoveParticles` fails `kBadInput` naming the solid. A tolerance from `ComputeContactTolerance`
    over a scale that holds the domain box and every solid meets the first.
  - *Solid count.* Contact can leave a far solid out of a particle's reads only for the first 64 solids of the array
    (one 64-bit mask per particle, `kMaxLeftOutSolids`); the domain box and solids at index 64 and above are read by
    every particle. Results do not change, the cost does. Nothing else bounds `max_solid_count`.
  - *Sampling faces and corners (5.4).* `ComputeSolidVelocitySample` poses every solid again at each call (two sines
    for a turning one), so a call per face costs faces times solids poses. A box's distance outside an edge or a
    corner is the largest of its face-plane distances, shorter than the straight line by up to `Sqrt(3)`. `side` is
    the plate side the point is on, `{0, kNoPlate}` inside the plate's thickness; the domain box is a candidate like
    any solid.

## 09 Gas

- A10 is decided "together from the start": remove `kGasWithLiquidNotSupported`. Design, in this module, gas
  beside liquid: the gas region's boundary at the liquid surface (gas sees the liquid as a moving boundary
  with the liquid's face velocity; the liquid sees the gas pressure of a sealed or open gas blob in place of
  atmosphere where they touch), markers kept out of the liquid's sums (transfer's material sets), air pockets
  versus tracked gas (one cell is never both), and the tests: smoke over a moving water surface, a sealed gas
  pocket above liquid that is heated, liquid poured into a gas-filled box. Anything that still cannot work is
  a named capability exclusion with its lifting test.
- From batches: `kBatchedRunNotSupported` is removed (the gas solve runs on the whole grid); writable cell
  arrays on a try bound to the resized grid; `BuildTry` must not count resized cells as removed.
- From I/O: a full snapshot of the gas amounts for `saved_state` with a builder from it (the output
  `GasSnapshot` stays for volumes); a keyed type for sources and wind that change in time (`GasSource` and
  `GasWind` with keys, sampled at the middle time of each step).
- Step calls as in the step design after its own update.

## 10 Step

- A1, A5, A6, A8, A9, A10 in the stage list and everywhere:
  - stage 11 `SolveViscosityAndPressure` (viscous scenes) or stage 12 `SolvePressure` (others);
  - transfer stages run once per material set;
  - gas and liquid in one scene: remove the exclusion; order of the liquid and gas solves as the gas design
    states.
- From materials: the call order already aligned; no sand rungs.
- From solids: `ReadyScene::solids` is `solids::SolidsSettings`; one `CheckSolidsCanRun` signature (see 08);
  the step owns the `SolidsAtFaces` storage and calls `DriveWaves`.
- From gas: `ReadyScene` carries keyed `gas::GasSource` and `gas::GasWind`; a second `PressureBuffers`
  instance counted in the estimate; `step_limit` from gas; retry on `kStepTooLongForGasFlow`.
- From Blender: `CapabilityResult` reports every finding (no cap of 16; a growing list on the CPU, built once
  per check) so "all problems at once" holds.
- From I/O: `SimulationSnapshot` gains everything the next step reads (`has_pressure`, `pocket_of_cell`, the
  inputs of `ComputeStepDuration` kept at commit); its gas part is the full gas amounts;
  `ReadGridSnapshot(const Simulation&)`; `ReadyScene::keeps_velocity_after_pressure`;
  `kCapabilityReasonCount`. `ReadyScene` stays declared in `step/`.
- From render output: `FoamAndSprayInputs` gains `after_pressure`, `surface_distances`, `solids`,
  `step_count`; `SimulationSnapshot::is_grid_only_region`; `Module::kRenderOutput`; `BuildRunPlan` takes
  `std::span<const MemoryEstimateEntry> output_estimate`.
- From cell resize and batches: `GridFields` gains the seven arrays; callbacks `apply_grid_only_region` and
  `move_grid_only_liquid` with read access on `StepTry`; `ResizeTry` exposes the last try's geometry and solid
  coverage, the committed step duration, the material table, writable air pockets and `SolidsAtFaces` storage
  for the resized grid; `ComputeStepDuration` includes grid-only face speeds; ranges are built from the try
  grid after stage 2; a `run_stages` callback; the batched `BuildSimulation` layout; a `StepArray` table
  generated from the memory estimate entries; `max_blocks_per_batch` from `BuildBatchPlan`.
- Cancel: between stages always; inside a solve through the solve's cancel input.
- Foundation's `CopyBuffer` and `ReadMemoryAvailable` exist.
- From contact (plan P19, 2026-11-15; FOR THE OWNER): `kTooCloseToCall/kSqueezedBetweenSolids` is not always
  cured by a shorter step. Two solids closing on a particle with nowhere to go, or closing at a grazing angle that
  would throw it out faster than 64 times the largest speed involved (one face closing on a fixed one at under
  0.895 degrees, two at under 1.79), give the same squeeze at every step length. The step design must say what a
  persistent squeeze does instead of stopping the run after two retries: for example remove the particle and count
  it, keep it in place with the solids' velocity, or report the solids by id as a scene problem. Measured in the
  contact model: none in wedges of 5 degrees or more; frequent where a solid is driven into another (a ball pushed
  into a tilted plane it penetrates: 30 % of such draws).
- **From contact as built (2026-11-19).** Each fact below is in the contact headers (`contact_error.h`,
  `move_particles.h`, `move_particles_steps.h`, `solid_contact_types.h`); design 01 sections 3.5 and 6 give them in
  full.
  - *Retry rule.* `ContactFailure::can_shorter_step_cure` (byte 3) says whether a shorter step can cure a
    `kTooCloseToCall`. It is false for every other error and for close calls met at the start of the step whose test
    reads no time left: a particle in the crease of a turning solid and another solid, two reacting solids pushed by
    one response, solids closing on a particle, an escape faster than 64 times the speeds involved, more than 8
    directions or 16 surfaces. Map `kTooCloseToCall` to `kContactTooCloseToCall` (shorten) only when the flag is true;
    when false, stop or act as the owner decides, naming the solid (`solid_id`) and contact's `ToName` code. The
    comment of `kContactTooCloseToCall` ("`kSecondReactingSolid` included") becomes "when `can_shorter_step_cure`".
    True says a cure exists, not that one halving finds it: one catalogue scene needs the step cut to 2^-26 of its
    length, 26 halvings. Two retries (`max_retries_per_step`) are far from that, so the step needs a limit of its own
    for contact's curable close calls, and the owner's answer on what a lasting failure does (remove the particle,
    stop with a message, or check such scenes when they are loaded).
  - *Errors 5.6 does not map yet.* `kStartsInNarrowGap` (a scene error, as `kStartsInsideSolid`); `kBadInput` from the
    coordinate check: a particle's start plus its straight path, or a solid's origin over the step plus its reach,
    beyond `kLargestCoordinateInTolerances` = 2^40 contact tolerances on some axis, so the step computes the tolerance
    from a scale that holds the domain box and removes particles beyond open sides before they travel that far;
    `kBadInput` from a speed above `kLargestInputSpeedSum` = 2^480 m/s summed over the axes; the new last close call
    `kMovesTooFastForStep` (cured by a shorter step); `kOutOfMemory` from `BuildContactBuffers` only.
  - *Which failure.* `MoveParticles` reports the lowest (particle id, index), whatever failed; `has_particle_id` (byte
    2) is true for a failure of one particle and false for a failure of the call's checks, where `particle_id` 0 means
    nothing. So inside one call a scene error can hide behind a close call of a particle with a lower id; the step's
    "highest `RejectKind`" rule sees only what each call reports.
  - *Outputs.* After a failure that is not a call check, the outputs of the call are unspecified: particles before
    and after the failing one may have been written. The step discards the whole try. No output array may share a
    byte with an input array or another output (`kBadInput`), so `MovedParticles::position` and
    `velocity_for_next_step` are try arrays apart from the committed `start_position` and velocities.
  - *Signature and record.* `MoveParticles(const MoveParticlesInputs&, const MovedParticles&, ContactBuffers&)`.
    `ParticleContact` lives in `solid_contact_types.h`, 56 bytes. `hit_count` is 0 to 4 (`kMaxHitsPerStep`) and counts
    moments, not surfaces; `normal_out_of_solid` and `approach_speed` are those of the first hit only; a `kHit` may
    record an approach speed of zero or more (a particle leaving a turning solid).
  - *Memory.* The contact names of section 4 list `BuildContactMemoryEstimate` only. Beside it,
    `BuildContactStackEstimate(RunOn)` gives one entry, `MemoryWhere::kDriverStack`: the GPU stack of the particle
    visits, about 12.8 kB per resident thread, 707 MB on a 36-multiprocessor card. `BuildContactBuffers` on the GPU
    reserves it at once and fails `kOutOfMemory` with its bytes when the card cannot hold it; it stays with the GPU's
    context for the rest of the process. The run plan compares the `kRunOn` sum plus the largest `kDriverStack` entry (A115) with the
    GPU's memory. The failure array is 32 bytes per particle.
  - *GPU pass.* Steps 2, 4 and 6 of 5.5 are one fused pass on the GPU. If that pass moves particles itself, it calls
    `ComputeParticleMoveAt` (the speed and coordinate checks, then `MoveParticle`), not `MoveParticle`. Each
    `MoveParticles` call copies the solids to the GPU and reads two values back, so each call waits for the GPU.

## 11 Cell resize and batches

- Recommended options stand: liquid crosses regions as particles; grid-only velocities are followed back; the
  solves are not batched (whole grid on the GPU, CPU fallback).
- A9: remove sand from "practical scenes"; A10: gas arrays are remapped; the fire scene is workable in batches.
- Folder names: `cell_resize/` and `batches/` (recommended).
- Align with the updated step, grid and particles entries above.
- **From contact as built (2026-11-19).** Each fact below is in `contact/move_particles.h` and
  `contact/move_particles_steps.h`.
  - *GPU stack, once per process.* `BuildContactBuffers` on the GPU raises the GPU's per-thread stack for the
    particle visits (about 707 MB on a 36-multiprocessor card, `BuildContactStackEstimate`,
    `MemoryWhere::kDriverStack`).
    It belongs to the GPU's context, not to one `ContactBuffers`: the loaded instance and the whole instance of 5.7
    share it, so the batch plan counts that entry once, outside `bytes_per_loaded_block` and
    `bytes_per_loaded_particle`. Per loaded particle contact takes 32 bytes (one `ContactFailure`), beside the step's
    56-byte record and 56-byte reaction.
  - *One tolerance for every batch.* Every particle and solid must lie within 2^40 contact tolerances of the origin
    (`kBadInput` otherwise), and the bits of a move depend on the tolerance, so every batch passes the run's one
    `contact_tolerance`, computed from the whole domain box, never from the loaded piece.
  - *Failures across batches.* Inside one call the failure reported is the lowest (particle id, index). Ids are
    unique, so comparing `particle_id` (when `has_particle_id`) across batches gives the same answer wherever the
    batches are cut; never compare the index, which depends on the cut. After a failed call the outputs are
    unspecified: a failed batched try discards the outputs of every batch, not only of the failing one.

## 12 I/O and CLI

- A14: the playback cache is not FluidiTy's array file. The array file remains for `saved_state` and internal
  run records (including the internal motion samples the surface builder reads). Remove `ParticlesFrame` as a
  playback product; particle playback is an open-format product written by module 13's writers.
  I/O owns: the run folder, `WriterLock`, `file_list`, `WriteListedFile` (the one publication transaction),
  an `OutputFile` sink (not visible to readers until listed, with `Write(bytes)` and `path_while_writing()`
  for libraries that must write to a path), the bounded output queue, the run report, the fingerprint.
- Words (recommended): a file a reader can see is `listed`; `WriterLock`; `RunReport`, `RunScene`,
  `RunFolder`, `Digest`.
- CLI (I/O owns the names; recommended set `capabilities | check | run | resume | report`), extended with
  what the add-on needs: `check --cache <run_folder>` (per output `same | changed | absent` with
  `changed_fields`), `report <run_folder>` as the read-only query of a run folder (fingerprint, last listed
  frame, saved-state frame, last tried work, lock holder, output list), and `read-particle <run_folder>
  --frame n --particle-id id`. `capabilities` returns within 200 ms without touching the GPU and lists static
  limits. One process, no child process. Stdin: `cancel`, `stop_after_frame`; end of stdin means stop after
  the frame. Exit codes are I/O's.
- Events: I/O's names; add fields the add-on needs (`sent_at` in UTC set when the event is made, heartbeat at
  least once per second with the time of the last advance, findings with `code` from `ToName`, ids, scene file
  path, magnitude, limit, suggested changes).
- Derivations done here: particle rate, mass and rest volume from a source's `volume_rate`; seeding start
  regions (8 particles per smallest cell); box inertia from mass and shape; blocking materials and plates.
- Scene file: JSON checked against `schemas/simulate/v2/scene_file.fbs` (recommended); it carries
  `CellResizeSettings`, `BatchSettings`, `render_output::RenderSettings`, keyed gas sources and wind.
- `saved_state` stores every module's snapshot including the seven region arrays, the full gas amounts, the
  render-output snapshot (foam state), written at frame times only.
- `run_mode` stays in the fingerprint unless module 11's design states batched and one-piece runs give
  identical bytes (it is designed to; keep it in until a test proves it).
- Dependencies added here: FlatBuffers, BLAKE3, CLI11 (recommended: keep CLI11).

## 13 Surface, foam and spray (folders `render_output/`, `surface/`, `foam_and_spray/`)

- A14: open formats for every product, chosen by what Blender's native readers consume fully:
  - volumes: OpenVDB, one file per frame;
  - simulation particles, foam, spray, bubbles, droplets (points): Alembic, unless the Blender witness test
    shows USD points carry velocity and attributes through the native reader, in which case USD;
  - surface: USD if the witness test shows a per-frame USD surface with velocity (motion blur) and per-vertex
    attributes is read natively; otherwise Alembic. The design keeps both writers behind one
    `WriteSurfaceFile(format, ...)` only if both pass; otherwise one. Record the witness results in the design
    as "to be proven in module 12" and make the format a build-time table, not scattered conditionals;
  - solid motion: USD transform samples (or Alembic xform) if the witness test passes, else the JSON file the
    add-on turns into transform keys.
  This module owns all open-format writers, including simulation-particle playback (taken over from I/O), and
  the build target `fluidity_interchange` (Alembic, OpenVDB, USD libraries), linked by the CLI only.
- A9: no sand surface; snow points only as on `main` (recommended).
- A10: gas volume beside liquid surface in one frame.
- Uses I/O's `OutputFile`, `WriteListedFile`, queue; counts grid-only cells as liquid and takes volume targets
  from `cell_resize::ComputeLiquidTotals`; `make_foam_and_spray` works once per batch on a loaded piece with no
  whole-grid sum inside.
- `VolumeTransport.lean` is not ported (recommended).

## 14 Blender add-on

- Adapts to I/O's CLI commands, event names, exit codes and words (`listed`, `WriterLock`) and to module 13's
  products and channel table; nothing is typed in that the `file_list` provides.
- A14: native readers for Alembic, USD and OpenVDB; the witness tests that decide between USD and Alembic for
  points, surface and solid motion are part of this module's test list and must run in Blender 5.3 and 5.2.
- A9: no sand preset and no sand reference scene until the sand module; three reference scenes (wave, syrup,
  fire) plus a snow example.
- A10: one scene may contain liquid and gas.
- Recommended options stand (a run continues when Blender closes: recommended option of its question 2;
  settings live on the domain object; English labels; Inspect in the first version as designed).

## 15 Sand (new, last module)

- Written from the sand parts removed from `07` (see there), plus: what sand needs from contact, transfer
  (fill radii for sand), solids (`AtSolidFaces` rule for sand faces), render output (sand points), I/O (sand
  settings in the scene file) and Blender (sand preset, the sand reference scene). The settling creep is
  investigated here with the eight-step diagnosis; the module is done when all eight pass or the owner decides
  otherwise.

## 01b Mesh solids (A80; simple route A94, 2026-11-15)

**From contact as built (2026-11-19).** The 01b plan (`plans/2026-11-14-rebuild-01b-mesh-solids.md`) and the 01b
design's contact sections (5.4 to 5.7) are written against the contact of plan Task 10, which no longer exists: a list
of touched surfaces with the normal recorded at the hit, neighbour passes, a resting sweep at `tol / 2` whose hit is
moved to the rest gap, the rim move, a seventeenth hit that rests, and the per-particle names Task 16 of the contact
plan first proposed (`ApplyMoveParticleAt` and its call description). As built there is no surface list (one joint
response over every active surface, plan P20), no hit is moved (P21), a hit on a turning solid ends the sweeps with a
carry in the solid's own axes, and the call description is `MoveParticlesCall` with `ComputeParticleMoveAt`. The 01b
plan must be planned again on the code as built before any of its tasks runs; the entries below that came from it are to
be checked again in that pass. Facts the new plan must start from, each in the contact headers:
- `ContactFailure` (32 bytes): `error` 0, `close_call` 1, `has_particle_id` 2, `can_shorter_step_cure` 3,
  `particle_id` 8, `solid_id` 16, `magnitude` 24. `mesh_problem` cannot sit at offset 2 or 3: both are taken. Bytes 4
  to 7 are padding (the header says so), so byte 4 is free for a one-byte `MeshProblem`, size unchanged. Every mesh
  failure must set `can_shorter_step_cure` by the rule of `contact_error.h` (`kMeshNotUsable`: false).
- `Solid` is 200 bytes, `reports_reaction` at 192; a 4-byte `mesh_index` fits at 196.
- `hit_count` is at most `kMaxHitsPerStep` = 4, and a hit is a moment with one joint response, not a surface.
- `MoveParticles` checks each solid's origin over the step plus its reach, `Length(half_size)`, against 2^40
  contact tolerances; a mesh's `half_size` must hold every vertex for that check to cover the mesh.
- Only the first 64 solids can be left out of a particle's reads (`kMaxLeftOutSolids`).
- Adding `meshes` to the call changes `MoveParticlesCall`, whose size (136 bytes) is asserted.
- On the GPU the visit already takes 255 registers and up to 12,784 bytes of stack per thread; a tree search needs
  its own stack on top, and `BuildContactStackEstimate` grows with it.
- What module 1's final review asks of contact before 1b is planned (section 4 of `final-review.md` in the contact
  ledger): the surfaces of one solid within a window as one per-shape function with a varying count (a concave edge
  of one mesh is two surfaces of one solid; as built `FindActiveSurfaces` lists one surface per solid, so a particle
  pressed into a concave edge would end in `kTooManyHits` at every step length); per-shape answers in exhaustive
  switches; the carry only for shapes whose distance is convex.

Rewritten 2026-11-15 from `01b-mesh-solids-design.md` ("01b") after A94. Every entry is **to apply**; each edit
is marked `(A80)` where it lands. Order (A82): the contact entries are applied by the 01b plan
(`plans/2026-11-14-rebuild-01b-mesh-solids.md`, right after module 1's Task 18); every other module's entries
when that module's design and plan are next touched. The same list as tables, with "no change" rows, is
`01b-mesh-solids-cross-module.md`. Entries that served thin walls only (sides on mesh parts, `PlateSide::part`,
`IsOnOppositeSides`, `FindPlateSide`, `FindThinPartCrossing`, closed faces, `MeshSurface`, growth for the grid,
`mesh_thin_part_grown`, `mesh_passage_closed_for_liquid`, `mesh_gap_narrower_than_a_cell`) are deleted: A85 and
A89 to A93 bind nothing in the first version. Particles 03, materials 07, gas 09 and sand 15: no edit.

**01 Contact** (design `01-contact-design.md`; code and tests by the 01b plan):
- Plan Tasks 9 to 16: the "Room for module 1b" notes are already in the contact plan. Task 16's "48 indexes"
  reads 24 (`kMaxMeshTreeDepth`).
- Design 3.1 (01b plan Task 2): `ContactError::kMeshNotUsable`, `MeshProblem`, `ContactFailure::mesh_problem`.
- Design 3.2 (01b plan Tasks 2, 7): `SolidShape::kMesh`, `SolidFace::kMeshTriangle`, `kNoMesh`,
  `Solid::mesh_index`, `CheckSolid`'s `kMesh` case, "Requires: no solid is a `kMesh`" on the point functions
  without `MeshShapes`.
- Design 3.4, 3.5 (01b plan Tasks 7, 15): `MeshShapes meshes` in `SolidContactInputs` and `MoveParticlesInputs`;
  `hit_count`'s comment (at most `kMaxHitsPerStep + kMaxMeshHitsPerStep + 1`).
- Design 1, 5.5, 5.7 (01b plan Task 18): "Owns" names the mesh shape; 5.7's "each convex solid is hit at most
  once" gains "a mesh may be hit several times (01b 5.6)"; 5.5 step 4 gains "a hit on a mesh is moved to the
  rest gap before the response, which uses the normal recorded at the hit (01b 5.5)".
- Contact plan P18 and Task 9b (added 2026-11-15, to apply to the 01b design's "a mesh that turns" section and the
  01b plan's Task 10, which must choose with measurements): the stepping search toward a turning solid now takes the
  larger of the first-order advance and a second-order advance that is safe only for a distance convex in the solid's
  own axes (the reported normal a subgradient). A mesh's distance, the smallest distance over its triangles, is not
  convex, so a turning mesh cannot use the second-order advance on the mesh's distance as a whole: on a test-only
  turning room (also a smallest-of-several distance) it misses walls on 27.5 % of paths, and on an L of two boxes on
  1.5 %. `HasConvexDistance(kMesh)` returns false unless 1b chooses otherwise. Options for 1b: (a) first order alone,
  "Task 9 unchanged" as 01b 5.7 says, with Task 9's close-call rate: 10.75 % of the module-1 generated paths and
  about 60 % of particles near a turning paddle, which with two retries per step would stop runs near a turning mesh
  paddle; (b) second order per grown triangle, each convex: advance by the smallest, over the triangles the tree query
  returns within the sweep's reach, of each triangle's own safe advance (first or second order from that triangle's
  distance and normal), which is safe because the mesh's gap is the smallest triangle gap and each stays above its own
  model; it costs one model per candidate triangle per advance and needs the tree query to return every triangle the
  advance can reach; (c) (b) only for meshes the scene marks convex. The 01b design states the choice with measured
  close-call shares and advance counts on a mesh paddle and a mesh cube (the cube must match module 1's turning box
  within the counts of Task 9b's tests), and its Task 10 tests it against the lower-bound reference.

**02 Grid:** design 5.9 gains one sentence: the size rule relies on a 1-Lipschitz distance, which the mesh
distance is (01b 5.3). No plan edit.

**04 Transfer:** plan Task 5: the fixture `BuildSolidsAtFaces` calls the mesh-aware
`ComputeSolidVelocitySample`; test `BuildSolidsAtFaces, MeshCube_EqualsBox`. No design edit.

**05 Pressure:** no interface or method edit. `plan-decisions.md` "Pressure", the thin-solid sentence, gains: "A
mesh has no such rule in the first version (A94): a mesh part thinner than 2.5 smallest cells is reported before
the run as a known limit (`mesh_part_too_thin_for_the_grid`, 01b 5.9) and may meet `kCellSplitBySolid` or pass
unseen." Design 5 step 8 and decision 7: "a plate" becomes "a plate or a thin mesh part".

**06 Viscosity:** design 3 `ViscosityInputs` gains `contact::MeshShapes meshes` beside `solids`; design 4 and
plan Task 10 call `ComputeSolidVelocitySample(..., meshes)`; design 5.7 and plan Task 6: a mesh solid counts as
inclined (excluded until Task 21); plan Task 21: the tilted plate again as a mesh slab, same 5 %.

**08 Solids:** design 1 and 10 drop "solid meshes (none)"; 3.2 and plan Task 3: `SolidSettings::mesh_index`,
`SolidsSettings::meshes`; 5.1 and plan Tasks 10, 12: `BuildTry` and `BuildContactSolids` handle `kMesh`, and
`kSolidsTooClose` samples a free mesh at the corners and centre of the box of its `half_size` (A83); 5.4 and plan
Tasks 15, 16, 18: `ComputeSolidsAtFaces` and `ComputeSolidCoverage` (both processors) take `MeshShapes` and pass
the coverage cap as `largest_distance`; 3.4 and plan Task 9: `has_inclined_solid` is true for any mesh. 5.9
(plate sides, `ComputeMinPlateThickness`, cutting): unchanged, one sentence added: "a thin mesh part has no side
and no thickness rule (01b 5.9)". Tests (plan Tasks 16, 35, 36a): `ComputeSolidCoverage, MeshCube_EqualsBox`,
`..., Icosphere_BlockedAreaConverges`, `Hydrostatic_PushIsWeightOfDisplacedLiquid` with a mesh, a driven mesh
paddle thicker than 2.5 cells.

**10 Step:** design 3.2 and plan Task 5: the step owns one `contact::MeshShapeBuffers`; scenes
`BuildDrivenMeshScene`, `BuildFreeMeshScene` (A83). Stages 11 and 16, `FoamAndSprayInputs` (plan Tasks 19, 20,
25, 26) fill `meshes`. Design 3.4 and plan Task 3: `CapabilityReason::kMeshNotUsable` (NEW, `kUnavailable`) and
`kMeshPartTooThinForTheGrid` (NEW, `kExperimental`, `kMakeSolidThicker`); `kCapabilityReasonCount` grows by 2;
`inline constexpr double kMinWallThicknessInCells = 2.5`; `kThinSolidInsideOneCell`'s comment says "a plate or a
mesh part". Design 5.14 and plan Task 7: `CheckSceneCanRun` calls the mesh-aware `CheckSolid` and
`ComputeContactScale`, `CheckMeshShape` once per mesh, and adds the thin-part finding per mesh solid whose
`thinnest_wall` is below `kMinWallThicknessInCells * smallest_cell_width` (`solid_id`, `magnitude`, `limit` in
m; whatever the materials); tests `CheckSceneCanRun, MeshFinThinnerThanTheGridNeeds_IsAKnownLimitNamingTheSolid`,
`..., MeshSlabThreeCellsThick_HasNoFinding`. Design 5.9 and plan Task 8: `BuildRunPlan` adds
`BuildMeshShapeMemoryEstimate`. Design 5.15 and plan Task 8a: `BuildStartCells` takes `MeshShapes` (CPU) and
tests the face centre as a fifth point in "against a solid".

**11 Cell resize and batches:** design 5.9 and plan Task 38: mesh rows are whole-run rows on the processor in
every batch. No interface edit.

**12 I/O and CLI:** design scene file and plan Task 26: `solids[].shape` gains `mesh`, new field `mesh_file`,
scene schema version raised. New plan Task 26a and design section for `io/mesh_solid.h`: read the PLY,
`contact::BuildMeshShape`, `MeshProblem` to findings, volume, centre of mass, inertia, re-centring for a free
solid (A83), `inputs/mesh_solids` listed and read on resume, mesh files in the fingerprint. Plan Tasks 4, 36:
the message of `mesh_part_too_thin_for_the_grid` (01b 5.9). Plan Task 15: `BuildStartParticles` calls the
mesh-aware distance. Plan Task 23a: memory entry "io: mesh solids"; budget lines show mesh bytes. Plan Task 28:
resume without the PLY.

**13 Surface, foam and spray:** design `FrameInputs`, `MotionSample` and plan Tasks 23, 45 carry
`contact::MeshShapes` (CPU); design 5.3, 5.5, 5.6 and plan Tasks 18, 22, 35, 37 call the mesh-aware distance
with `largest_distance` 4 spacings; tests `BuildSurfaceField, BesideAMeshSolid_SameHeight`,
`MoveFoamParticles, StopsAtAMeshSolid`.

**14 Blender add-on:** design 3.2 and plan Task 5: `Shape.MESH`, the default for a `SOLID` object that is not
exactly a box, sphere or cylinder (A88). Design 5.1 item 2 and plan Tasks 8, 8a, 27: export of the render-level
evaluated mesh (A86) as PLY (`inputs/solids/solid_<id>.ply`, fan rule, scale applied); the digest covers the
files; design 10 drops "evaluated mesh shapes". Design 3.4 and plan Tasks 11, 13: findings
`solid_mesh_not_closed` (`BLOCKS_RUN`, A87; the repair button is later work), `solid_mesh_deforms` (`BLOCKS_RUN`,
A84), `solid_mesh_triangles_small` (`NOTE`, A86), `solid_mesh_too_many_triangles` (`BLOCKS_RUN`), and from the
native check `mesh_part_too_thin_for_the_grid` (`HAS_KNOWN_LIMIT`) and the `MeshProblem` codes (`BLOCKS_RUN`).

**`shared-interface.md`:** (b) and (c) Contact gain the names of 01b 3.1 to 3.4. **`tradeoff-particles.md`:** 1.3
gains one line: mesh solids, 64 to 74 B per triangle on the processor.

## Names changed while the first designs were updated

Later designs use the new name wherever they cite the old one.

| Module | Old | New |
|---|---|---|
| contact | `SolidReaction::state_velocity_change` | `SolidReaction::velocity_for_next_step_change` |
| contact | `MovedParticles::state_velocity` | `MovedParticles::velocity_for_next_step` |
| contact | `FindFirstHit(..., Vec3 travel_velocity, ...)` | `FindFirstHit(..., Vec3 velocity_for_this_step, ...)` |
| contact | (new) | `ToName(ContactError)`, `ToName(ContactCloseCall)` |
| viscosity | `ApplyViscosity(inputs, settings, buffers, velocity_after_pressure, cell_pressure)` | `SolveViscosityAndPressure(inputs, settings, cancel_requested, buffers, velocity_after_pressure, cell_pressure)` |
| viscosity | `ViscositySceneSummary::has_snow_or_sand`, `ViscosityError::kSnowOrSandNotSupported` | `has_snow`, `kSnowNotSupported` |
| viscosity | `const transfer::CellsFromParticles* cells_from_particles` | `std::span<const transfer::CellsFromParticles* const> cells_from_particles` (one per liquid material set, in the step's order; gas marker sets are not passed) |
| viscosity | (new) | `ViscosityError::kCancelled` (the step maps it to a cancelled try); `block_range` must be every block of the grid |
| pressure (kept for viscosity) | | `ComputeOwnWeight(equations, connection_scale, own_weight)` keeps its `connection_scale` argument; `GivenTerms`, `ComputePressurePush`, `ComputeVolumeError`, `ComputeStartPressure`, `ApplyZeroMeanToSealedBlobs`, `ApplyPressureOffset`, `ComputeWasSolved`, `ComputeFaceInertia`, `CheckFlowImbalance`, `ComputeFlowImbalance` stay public |
| transfer | `can_hold_liquid` (face flag in `SolidsAtFaces`, `GridSnapshot`, and wherever other designs cite it) | `outside_solid` |
| transfer | `SaveGridAfterPressure(..., const grid::FaceValues<std::uint8_t>& was_solved, ...)`; `GridToParticlesInputs::cell_was_solved` as a reference | both are pointers, null for a material set that no pressure solve writes (snow, springy solids); for such a set the step passes the set's velocity after `CollideSolids` as the after-pressure velocity |
| transfer | `BuildTransferMemoryEstimate(ingredient_count, material_count)` | `BuildTransferMemoryEstimate(ingredient_count, material_count, material_set_count, sets_without_solve)` |
| transfer | "empty mask means all materials" (written in the gas and solids designs) | `material_is_counted` always has one entry per material, each 0 or 1, at least one 1; a scene with one set passes all ones; an empty mask is `kBadInput` |
| transfer | (new) | `MaterialSetKey`, `ToMaterialSetKey`, `CheckMaterialSets` (the step calls it in its capability check), `TransferError::kWrongMaterialSet`, `FaceValueKind::kFromGridOnlyRegion`; memory per extra set per cell block: 48,576 + 512 x ingredients bytes (40,416 + 512 x ingredients for a set with no solve) |
| grid | `BuildGrid(inputs, settings, storage)` (as the step wrote it) | `BuildGrid(const GridInputs& inputs, GridStorage& storage)`; storage from `BuildGridStorage(const GridSettings&)`, given back with `ReleaseStorage(StaggeredGrid&&)`; `StaggeredGrid` is move-only; a built grid has `GridVersion` 1 or more (0 means "bound to no grid") |
| grid | `ApplyGridVersion(const StaggeredGrid& grid, CellValues<Value>& values)` | `ApplyGridVersion(CellValues<Value>& values, const StaggeredGrid& grid)` (also for face, corner and block values); values are built with `Build...Values(RunOn run_on, std::uint32_t max_cell_blocks)` |
| grid | `kGridMemoryEstimate` | `BuildGridMemoryEstimate(RunOn run_on, std::uint32_t storage_count)`; the step passes 2 (committed and try grid); 972 B per block per storage on the CPU, plus 456 B on the GPU for a GPU run |
| grid | `BuildGridPiece(whole, const RangeWithBorder&, const GridSettings&)`, `GridPiece { grid; range; block_in_whole_grid; }` | `BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range, std::span<const CellBlockIndex> border_blocks, GridStorage& storage)`; `GridPiece` is a class with `grid()`, `range()`, `range_in_whole_grid()`, `block_in_whole_grid()`; its blocks keep the whole grid's fixed order; module 11 counts its storage with `BuildGridMemoryEstimate(run_on, 1)` |
| grid | `std::vector<CellBlockIndex> FindBorderBlocks(grid, blocks, width)` | returns `std::expected<std::vector<CellBlockIndex>, GridFailure>`; a no-allocation form takes a `std::span<CellBlockIndex>` output and returns the count |
| grid | border "blocks holding a cell that touches the range" | the border also includes blocks that store a face piece of a touching cell (a face piece can be stored one block beyond); transfer and batches rely on this definition |
| grid | version checks on values | values bound to a piece are bound again at every load; a check compares both `grid_version()` and `block_count()` |
| particles | `particles::BuildMemoryEstimate` | `particles::BuildParticlesMemoryEstimate` |
| particles | `BuildParticlesFromSnapshot(settings, snapshot)` | `BuildParticlesFromSnapshot(const ParticlesSettings&, const ParticlesSnapshot&, const grid::StaggeredGrid& grid)`; a full snapshot also stores `position_when_grouped`, and the stored order is kept (never regrouped), so a continued run gives the bytes of an uninterrupted one |
| particles | `struct SnapshotFields { FieldList fields; bool all = true; }` | `struct SnapshotFields { bool all = true; FieldList fields{}; }`; new `kOutputSnapshotFields`, `Field::kMass`, `ToFieldList`, `HasField` |
| particles | `Field::kStateVelocity`, `Field::kTravelVelocity`; snapshot array `velocity` | `Field::kVelocityForNextStep`, `Field::kVelocityForThisStep`; snapshot array `velocity_for_next_step` |
| particles | `SplitParticle`, `MergeParticles`, `AdjustParticleCount` with one grid | take `grid_before_resize` with its `NeighbourhoodInputs`, then `grid` with its inputs; called only inside the resize transaction |
| particles | (new) | `WriteParticleGroups(ParticlesTry& loaded, const grid::GridPiece&, const ParticlesTry& whole)`, `ReadParticleGroups(const ParticlesTry& loaded, grid::CellBlockRange, FieldList, ParticlesTry& whole)` (the range is in whole-grid block indexes and equals the range of the piece last loaded); `CheckParticles(const ParticlesTry&, grid::CellBlockRange)`; `ComputeAmountsPerBlock`; `ParticlesSettings::cpu_memory`; `RemoveReason::kMarkerNotNeeded`; `ToName(ParticlesError)` |
| pressure | `SolvePressure(equations, velocity_response, settings, buffers)` returning array pointers | `SolvePressure(const PressureEquations&, const SolveSettings&, const std::atomic<bool>* cancel_requested, SolveOutputs outputs, PressureBuffers&)`; `SolveOutputs { pressure, velocity_after_pressure }` are the step's try arrays; `SolveResult { report; blobs; iteration_count; ladder_rung_count; }` |
| pressure | `FindAirPockets(inputs, buffers)` | `FindAirPockets(const AirPocketInputs&, AirPocketOutputs outputs, PressureBuffers&)`; `AirPocketOutputs { pocket_of_cell, pocket_storage }` are the step's try arrays |
| pressure | `MaterialBoundary::viscous_jump` | removed (nobody provides it; the joint solve carries the viscous force itself) |
| pressure | (new, A10) | `EquationInputs::gas_pressure_above_atmosphere` (`CellValues<double>`, Pa) and `holds_tracked_gas` (`CellValues<std::uint8_t>`), also `AirPocketInputs::holds_tracked_gas`; the gas module provides both arrays bound to the try grid; the liquid reads the gas pressure committed at the end of the previous step (one step late) |
| pressure | (new) | `ComputeDistanceToSurface` is called once with `kAllLiquidMaterials` and once per liquid material that touches another, then `FindMaterialBoundaries`; inputs take `material_is_counted` and `has_viscous_ingredient`; `BufferSizes::max_boundary_materials`, `BufferSizes::has_viscous_ingredient`; `BuildPressureLadder`, `ApplyPressureLadder`; `ComputeBlockedCornerAreas`, `ComputeEquationIndexes`; `kNoBlob`, `ComputeBlobOfCell`, `SealedBlobCheck`, `CellContent::kGas`; `kNoEquation`, `kNoAirPocket`; `PressureError::kCancelled`; capability exclusion `kMaterialBoundaryNotSupported`; `ToName(PressureError)` |
| pressure | a liquid cell with no particle had no density rule | it takes the mixture density of the touching liquid cell with mass and the lowest index, else `lightest_density` |
| materials | `MaterialKind { kLiquid, kGas, kSand, kSnow, kSpringySolid }` | `{ kLiquid, kGas, kSnow, kSpringySolid }`; the sand module appends `kSand = 4`; solids cites only `kSnow` and `kSpringySolid` |
| materials | `MaterialResponseInputs` fields `start_position`, `travel_velocity`, `solids`; `MotionInStep::exact_velocity_change_after_hit`, `duration_after_hit`; `MaterialResponseResult::hit_corrected_count` | removed from 07 (the sand module adds them back, with `velocity_for_this_step`); the step does not pass them |
| materials | `std::expected<InternalForceBudget, MaterialsFailure> ApplyInternalForce(grid, blocks, step_duration, settings, grid_from_particles)` | `std::expected<void, MaterialsFailure> ApplyInternalForce(..., transfer::GridFromParticles& grid_from_particles, std::span<InternalForceBudget> budget_of_block)` plus `ComputeInternalForceBudgetTotal(std::span<const InternalForceBudget>)`; the step keeps one budget per block |
| materials | `LimitCase { kInsideLimits, kShapeReduced, kPulledApart, kClamped }`, `kLimitCaseCount = 4` | `{ kInsideLimits, kClamped }`, count 2 (sand appends two) |
| materials | `snow_and_sand_material.h` | `snow_material.h`; shared `stretch_response.h` |
| materials | (new) | `MaterialTable::gas_ingredient_index()` (`std::optional<std::uint32_t>`), `MaterialTable::ingredient_export_names()` (`std::span<const std::string>`, already normalised), `MaterialTableInputs::ingredient_export_names`, `kBadExportName`, `ToName(MaterialsError)` |
| build order | module numbers cited in section 4 of several designs | the programme design's table: 0 Foundation, 1 contact, 2 grid, 2b particles, 3 transfer, 4 pressure, 5 viscosity, 6 materials, 7 solids, 7b gas, 8 step, 9 cell resize and batches, 10 I/O and CLI, 11 surface with foam and spray, 12 Blender add-on, 13 sand. Refer to another design by its name, not its number, to avoid confusion with file numbers |
| cell resize | `ComputeLiquidTotals(particles_try, amounts, grid, blocks)` | `ComputeLiquidTotals(const particles::ParticlesTry&, const GridOnlyAmounts&, std::span<const std::uint8_t> material_is_counted, const grid::StaggeredGrid&, grid::CellBlockRange)` (gas markers are left out) |
| cell resize | (new) | `cell_resize::ApplyGridOnlyRegionInStep` and `cell_resize::MoveGridOnlyLiquidInStep`, both `step::StageOutcome(void* context, step::Simulation&, step::StepTry&, grid::CellBlockRange)`; `cell_resize::ReadLargestGridOnlySpeed(const CellResize&)` read by `step::ComputeStepDuration`; `BatchPlan::max_loaded_particles`; the grid-only move runs at the end of stage `kMoveParticles` (no stage of its own) and its failures map to `kParticleTravelledTooFar` |
| step (decision) | `StepCallbacks` with one `void* context` for all callbacks | every callback is a pair `{ function pointer, void* context }`, so cell resize, batches and render output each pass their own object; `cell_resize::CallbackContext` is then not needed by other modules |
| step (asked by cell resize) | | `ResizeTry` also gives the free `grid::GridStorage` for the resized grid, the end-pose `SolidsAtFaces` of the last try, and the material sets; `StepArray` has one entry per array instance (one per material set for a per-set array) |
| solids | `CheckSolidsCanRun` returning `std::expected<void, ...>` or taking a scene summary | `std::vector<SolidsFailure> CheckSolidsCanRun(const SolidsSettings& settings, double smallest_cell_width, double largest_cell_width)`; an empty list means the scene can run |
| solids | names the step draft used: `ComputeSolidsForStep`, `ComputeStepLimits`, `BuildFreeSolidPushes`, `MoveSolids` (as a solids function), `SolidsSceneSummary`, `BuildSolids(span, DomainBox)` | `ComputeStepDurationLimit`; `BuildTry(solids, step_duration, smallest_cell_width)`; `BuildMotionMap` with `ToFreeSolidPushes` and `ToFreeSolidInverseInertia`; in the step's stage `MoveSolids`: `ApplyContactReactions` and `ApplyPressureToMaterials`; `SolidsSummary`; `BuildSolids(const SolidsSettings&)` |
| solids | the step calling `pressure::ApplyOpenSeaRelaxation` in `DriveWaves` | the step calls `solids::DriveWaves`, which returns `momentum_added_of_block` and `energy_added_of_block` |
| solids | `void CommitTry(...)` | `std::expected<void, SolidsFailure> CommitTry(...)` |
| solids | `SolidsSettings::max_free_solid_count` | removed; public constants `kMaxPlateCount = 1`, `kMaxFreeSolidCount = 16`; new `SolidsSettings::has_liquid_or_gas` (I/O fills it) |
| solids | (new) | `BuildContactSolids(std::span<const SolidSettings>, const SolidsSnapshot&)`; `ComputeRotationVector(Orientation from, Orientation to)`; `ComputeSolidsAtFaces(const grid::StaggeredGrid&, grid::CellBlockRange, const Solids&, transfer::SolidsAtFaces&)` using committed poses (for cell resize); `MaterialSetFaces::cells_from_particles`; `SolidsError::kWrongMaterialSet`; `CutWorkFields::cell_blocks`, `group_begin`; `ToName(SolidsError)` |
| solids | rules for the step | every material set except gas is passed to `CollideSolids`; gas markers' contact reactions are never passed to `ApplyContactReactions` |
| gas | the step draft's gas calls `AddGasFromSources`, `ComputeCellsNeeded`, `ApplyGasToGrid`, `ApplyGasForces`, `ComputeGasGeometry`, `ComputeAddedVolumeRate`, `MoveGas`, `gas::ComputeStepLimit`, `gas::GasTry` | `FindMarkerChanges`, `FindCellsToKeepActive`, `BuildTry` and `ComputeGasSeenByLiquid` (both in `ChooseActiveCells`, after `grid::BuildGrid`), one stage `SolveGas` after the liquid solve and `ComputeWasSolved` (no `MoveGas` stage), `GasResult::step_limit`, `gas::GasAmountsTry` |
| gas | | `SolveGas(inputs, settings, cancel_requested, SolveGasOutputs{velocity_after_pressure, was_solved, cell_was_solved}, amounts_try, gas_buffers, pressure_buffers)`; `GasError::kCancelled`; `CollideSolids` is not run on the gas set |
| gas | `SimulationSnapshot` gas part as `GasSnapshot` | `gas::GasAmountsSnapshot` with `ReadGasAmountsSnapshot`, `BuildGasAmountsFromSnapshot` (`GasSnapshot` stays for volume output) |
| gas | `GasSource` (box with rates), wind as one acceleration | keyed `gas::GasSource { source_id; low_corner; high_corner; std::span<const GasSourceKey> keys; }`, `gas::GasWind { std::span<const GasWindKey> keys; }`, `SampleGasSource`, `SampleGasWind`, `CheckGasSources`; I/O owns the key storage |
| gas | `SolveGasInputs::gas_ingredient_index`, `ingredient_rest_density`, `cells_from_particles`, `gas_share` | removed from `SolveGasInputs` (spray no longer adds to the gas density); `GasSceneSummary::gas_ingredient_index` and `gas_ingredient_rest_density`; gas reads the gas set's `GridFromParticles::value_kind` and `velocity`, and nothing of the liquid set's `CellsFromParticles` |
| gas | (new) | `ComputeGasSeenByLiquid` fills `gas_pressure_above_atmosphere` and `holds_tracked_gas`; `ComputeSealedGasStepLimit`; `GasSettings::min_sealed_radius_in_cells`; `GasBlob::touches_liquid`, `held_at_atmosphere`; `BuildResizeTry`, `GasArraysToRemap`, `ToArraysToRemap` (cell resize uses them, then `CommitTry(amounts, try, amounts.time())`); `ToName(GasError)` |
| pressure (decision) | whether `ComputeDistanceToSurface` accepts a mask with no counted material | it does: an all-zero `material_is_counted` means "no liquid" and gives every open cell as air (a gas-only scene calls it this way). This is the one function where the mask may be all zero; transfer's stages still require at least one counted material |
| render output | `FoamAndSprayInputs` gaining one `after_pressure` snapshot | `std::span<const transfer::GridSnapshot* const> after_pressure` (one per liquid material set, in the step's order), plus `surface_distances`, `solids`, `step_count`, `grid::CellBlockRange blocks`, `std::span<const grid::CellBlockIndex> block_in_whole_grid`, `const GridFields* fields` |
| render output | `RecordStep(recorder, inputs, ranges)` | `RecordStep(StepRecorder&, const step::FoamAndSprayInputs& inputs)`; the range is `inputs.blocks`; it runs once per batch on a loaded piece, and every cross-batch sum happens in `CommitRecordedStep` |
| render output | `SimulationSnapshot` gaining `is_grid_only_region` | also `std::vector<double> grid_only_current_volume` and `grid_only_ingredient_volume` (by active cell) |
| render output | `CheckRenderSettings(settings, smallest_cell_width)` | `CheckRenderSettings(const RenderSettings&, double smallest_cell_width, const materials::MaterialTable& materials)` |
| render output | I/O writing particle and motion playback files | this module owns every open-format writer (`WritePointsFile`, `WriteSurfaceFile`, `WriteVolumeFile`, `WriteSolidMotionFile`) in the target `fluidity_interchange`; I/O opens each `OutputFile` with the extension from `ReadFormatChoice(kind)`; `FrameIdentity` gains `frames_per_second`; `kOutputContractVersion = 2` |
| render output | the add-on's product table (surface as USD, one `solid_motion.json`, `ingredient_share_<id>`, four gas grids) | the format table of the design's section 5.7: today points Alembic, surface Alembic, volumes OpenVDB, solid motion one JSON file per frame, each with a USD candidate "to be proven by the Blender add-on" (witness tests in Blender 5.2 and 5.3, results returned as `WitnessRecord` lines; a USD candidate that passes every witness replaces the current choice, one that fails is removed with its writer); gas grids `density`, `smoke`, `temperature`, `fuel`, `flame`, `known`; shares are `share_<ingredient_export_name>`; `id` is written as int32 where the reader only imports int32 |
| render output | (new) | `ToName(RenderOutputError)`, `kFormatNotInBuild`, `kFileWriteFailed`; library pins matching Blender's bundled versions (OpenUSD 26.03, Alembic 1.8.3, OpenVDB 13.0.0, oneTBB 2022.3.0, Imath 3.2.2) |
| I/O | step additions "`ReadGridSnapshot(const Simulation&)`" and "`ReadyScene::keeps_velocity_after_pressure`" | not needed (I/O no longer writes a motion file); `kCapabilityReasonCount` is still needed, and so are counts (or a last enumerator) for `SuggestedChange`, `RejectReason` and `StepLimit`; the `SimulationSnapshot` inputs of `ComputeStepDuration` include the grid-only speed from `cell_resize::ReadLargestGridOnlySpeed` |
| I/O | `ParticlesFrame`, `MotionFrame`, `StatePart`, `ExtraArray` | removed; the array file holds `ready_scene`, `saved_state` and the per-frame `StoredFrameInputs` (output snapshot, `RecordedInterval`, sample at the frame; kept for rebuilding outputs and for `read-particle`) |
| I/O | CLI as the add-on drafted it (`read-cache`, `run --cache --frames`, event names `run_started`, `frame_published`, ..., exit codes 0/2/3/4/5, `published_through`, lease) | `capabilities`, `check [--cache <run_folder>]`, `run <scene_file> --into <run_folder> [--last-frame n] [--outputs ids]`, `resume <run_folder>`, `report <run_folder>`, `read-particle <run_folder> --frame n --particle-id id`; events `started`, `preparing`, `heartbeat` (with `activity`, `last_advance_sent_at`), `finding`, `frame_checked`, `frame_done` (with memory numbers), `saved_state_listed`, `particle`, `stopping`, `done` / `stopped` / `failed`, each with `protocol`, `sequence`, `sent_at`; also written to `<run_folder>/events.jsonl`; stdin `stop_after_frame` / `cancel`; exit codes 0 done, 1 failed, 3 scene cannot run, 4 stopped on request, 5 run folder refused (lock held), 6 processor failed; `last_listed_frame`; `lock_holder.is_held`; findings carry `module` + `code`; severities `blocks_run`, `has_known_limit`, `note` |
| I/O | scene file as the add-on drafted it | driven path keys carry an orientation (w, x, y, z) and I/O derives the angular velocity; gas source keys `{time, fuel_rate, heat_rate}`; wind keys `{time, acceleration}`; start regions with `region_id` |
| I/O | (new) | `OutputDeclaration`, `BuildOutputDeclarations`, `BuildOutputFile`, `DiscardOutputFile`, `WriteListedFile(RunFolder&, OutputFile&&, CheckAfterWrite)` with `kSizeOnly` for open-format files, `io::WriteNotBuilt`, `ListedFile::product_report`, `FrameDone::snapshot_digest`, `RebuildOutputs`, `io::FailingModule`; open-format files sit under `outputs/` in the run folder |
| gas and render output (decision) | render output reads `gas::GasSnapshot`, while the step's snapshot holds `gas::GasAmountsSnapshot` | gas provides `GasSnapshot ToGasSnapshot(const GasAmountsSnapshot& amounts, const GasLaw& law)` (computes temperature and the output channels from the amounts); render output calls it |
| step (decision) | "materials that do not mix are separate sets" read as one set per liquid | **all liquids are in one material set** (one velocity on the grid for every liquid; two liquids that do not mix are kept apart by pressure's material boundary). Other sets: snow; springy solids; gas markers. So spans "one per liquid material set" have exactly one entry. `ReadyScene::material_sets` holds the masks; the step's set order is ascending lowest material index; any other layout is `CapabilityReason::kMaterialSetsNotValid` with `SuggestedChange::kPutAllLiquidsInOneSet` |
| step | stage numbers | 13 `kSolveGas`, 14 `kSaveGridAfterPressure`, 15 `kGridToParticles`, 16 `kMoveParticles`, 17 `kMoveSolids`; there is no `MoveGas` stage; `kStageCount = 22` |
| step | `TryStepWithStages(simulation, request, stages, ranges)` | `TryStepWithStages(Simulation&, const StepRequest&, const StageTable&)` (ranges are built from the try grid after stage 2) |
| step | (new) | `ApplyResizedGrid(ResizeTry&, const grid::StaggeredGrid&)` (call before `ResizeTry::gas_try()`); `StepCallbacks` members are `Callback<Function>{function, context}` pairs: `make_foam_and_spray` (`Callback<bool(void*, const FoamAndSprayInputs&)>`), `choose_cell_size`, `on_try_done`, `resize_cells`, `on_step_done`, `apply_grid_only_region`, `move_grid_only_liquid`, `run_stages`, `read_largest_grid_only_speed` (`Callback<double(void* context)>`, I/O wraps `cell_resize::ReadLargestGridOnlySpeed`) |
| step | `BuildRunPlan(scene, settings, on_processor, on_cpu, output_estimate)` | `BuildRunPlan(..., std::span<const MemoryEstimateEntry> output_estimate, std::span<const MemoryEstimateEntry> cell_resize_estimate = {})`; I/O copies `max_blocks_per_batch`, `max_loaded_blocks`, `max_loaded_particles` from `BatchPlan` into `RunPlan` |
| step | `ReadyScene` fields I/O fills | also `material_sets`, `open_sea_material_index`, `gas_sources`, `gas_wind`, `gas_markers`, `gas_active_cells`, `has_cell_resize`; `SimulationSnapshot::gas` is `std::optional<gas::GasAmountsSnapshot>` |
| step | reason `liquids_that_do_not_mix_are_blended` | `liquids_that_do_not_mix_touch` (`CapabilityReason::kLiquidsThatDoNotMixTouch`); counts `kCapabilityReasonCount = 19`, `kRejectReasonCount = 23`, `kSuggestedChangeCount = 13`, `kStepLimitCount = 11`; the step declares only `ToName(StepError)`; `io/stable_names.h` declares `ToName` for `Stage`, `RejectReason`, `CapabilityReason`, `SuggestedChange`, `StepLimit` |
| Foundation (decision) | `std::expected<std::uint64_t, BufferError> ReadMemoryAvailable(RunOn)` | `struct MemoryAvailable { std::uint64_t total_bytes; std::uint64_t free_bytes; };` and `std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on)`; a total of 0 means unknown |

## Nearby faces: 216 per axis and a mask of four words (grid, owner decision 2026-11-19)

The grid's bound of 76 nearby faces per axis was wrong (a valid grid reaches 124). `grid::kMaxNearbyFaces` is now
216, `grid::NearbyFaceMask` is `std::uint64_t bits[4]` (32 B, was 16 B), `sizeof(grid::NearbyFaceList)` is 888 B
(was 328) and `sizeof(grid::NearbyFaceWeights)` is 1,152 B (was about 600). Candidates at exactly 1.5 widths are
kept with weight 0, so a kept weight may be 0. Apply to each document below; none was edited.

- **04 Transfer (design and plan).** Any array or loop sized by 76, and any mask handled as two words
  (`FindSameSideFaces` returns a `grid::NearbyFaceMask`): use `kMaxNearbyFaces` and four words, or loop on
  `list.count`. The per-set memory figures of `shared-interface.md` (d) that count per-particle or per-thread
  bytes must be recomputed if they include a list, a mask or weights. A GPU function that calls
  `ComputeNearbyFaceWeights` keeps 1,152 B per axis (3,456 B for three axes) per thread, above the driver's
  default of 1,024 B: transfer declares a `MemoryWhere::kDriverStack` entry, as `BuildContactStackEstimate` does,
  from the compiled function's local bytes. A test that expects every kept weight above 0 must allow 0.
- **03 Particles (design and plan).** The same for its uses of `FindNearbyFaces`, `NearbyFaceMask` and the
  weights: sizes, the stack entry if a GPU function of particles calls them, and "weight above 0".
- **06 Viscosity (design).** Its `FindNearbyFaces` call per cell centre: an output sized by 76 becomes
  `kMaxNearbyFaces`; at a cell centre the list has 27 faces on a uniform grid, 9 of them of weight 0.
- **08 Solids and plates.** Code that clears mask bits (other side of a plate, not `outside_solid`): bit `i` is
  in word `i / 64`, up to index 215.
- **13 Surface, foam and spray (design).** Its uses of `ToMaskOfAll`, `NearbyFaceMask`, the weights: sizes and the
  stack entry if on the GPU.
- **10 Step (run plan and memory).** The sum of `kDriverStack` entries now has one per module whose GPU function
  calls the weights, not only contact's. The driver keeps one stack size for the whole context, so the bytes held
  are the largest of the entries times the resident threads, not their sum: the run plan should take the largest.
- **11 Cell resize and batches.** No size changes; `grid::CellSizeFacts::velocity_gradient_size` is now
  `velocity_change_size` (the glossary bans the old words).
- **Every plan that names them:** `storage_count` is `grid_count`; `IsBorderBlock` is `ComputeIsBorderBlock`;
  `ApplySizeStepClosure` is `BuildSizeStepClosure`; the four value classes are aliases of `grid::GridValues`.

## An impossible internal state has its own error value (owner answer A106, 2026-11-20)

Rule, for every module: the error enum has `kInternalFault` (stable name `internal_fault`), placed just before
`kOutOfMemory` where the module has that value last, else last. It is returned, in debug and release alike, for a
state no input can produce. It is never `kBadInput` (or the module's word for bad input) and never a debug-only
assertion. `ToMessage` says it is a fault in FluidiTy, not in the scene, and asks the user to report it with the
scene. The module's `BufferError` mapping sends `kSizeMismatch` to it, after the module has checked every caller's
sizes itself. No design below was edited.

- **01 Contact (as built; one small commit).** `contact/contact_error.h`: add `ContactError::kInternalFault`.
  `kStartsInNarrowGap` is documented "Last value" and `kOutOfMemory` is not last there, so say where it goes
  (recommended: last, after `kStartsInNarrowGap`, so no built value moves) and update that comment.
  `contact/contact_error_message.cc` and `.h`: its message and name. `contact/contact_error_test.cc`: the counts
  and the new name. `contact/move_particles_cpu.cc` line 35 to 40: the mapping sends `kSizeMismatch` to
  `kBuffersTooSmall`; decide per call whether a caller's count can cause it (then `kBuffersTooSmall` stays, checked
  before the `base/` call) or not (then `kInternalFault`). The first-failure reduction's "flag without index" case
  (`ComputeFirstFailure, FlagWithoutIndex_IsRunOnCallFailed` in `move_particles_steps_test.cc`) is an impossible
  state reported as `kRunOnCallFailed`: it becomes `kInternalFault`. The `assert`s of `contact_tolerance.h` and
  `solid_contact_types.h` on finite inputs are checks of a caller's values, not internal states; review each one.
  Method paper `<sec-contact>` if it lists the errors. Glossary: no new word.
- **02 Grid.** Done in the grid plan (eighth and ninth revisions) and design 3.3. `grid::GridError::kTooManyNearbyFaces`
  no longer exists (A107): it is `kInternalFault`, value 10; `kOutOfMemory` is 11. Every module that maps or names
  the grid's errors (particles, transfer, viscosity, surface and foam, the step's reasons, the CLI's and the
  add-on's lists of stable names) drops `too_many_nearby_faces` and maps the one value `kInternalFault`.
- **03 Particles, 04 Transfer, 05 Pressure, 06 Viscosity, 07 Materials, 08 Solids, 09 Gas, 11 Cell resize and
  batches, 13 Surface, foam and spray, 15 Sand (design and plan of each).** Add the value to the module's error
  enum in section 3, with its message and stable name; change the `BufferError` mapping; and go through section
  3's `Fails:` lists and section 5 for every "cannot happen", "should never", "checked by a debug assertion" or
  bad-input result that no caller input can cause (a count read back from the GPU above its room, a group or
  order that base/ returns inconsistent, a grid version that changed inside one call, a solver state that
  contradicts its own bookkeeping) and return `kInternalFault` there. A module that maps the grid's errors maps
  `grid::GridError::kInternalFault` to its own `kInternalFault`.
- **10 Step.** Its error and reject kinds gain the value; a stage's `kInternalFault` stops the run (it is not a
  reason to retry with a shorter step or raised limits) and the report shows the message with the module's name.
  `kRejectReasonCount` and any count of error values change.
- **12 I/O and CLI, 14 Blender add-on.** The CLI's exit code and the add-on's panel show the message as a fault
  in FluidiTy with the request to report it; the stable name `internal_fault` is added to the lists they match
  on.
- **01b Mesh solids.** As contact, when it is planned again.

## Grid names and limits as built (2026-11-20)

- **Every plan and design that names them:** `grid::CellWidthsByLevel` is `CellWidthsBySizeLevel`,
  `ComputeCellWidthsByLevel` is `ComputeCellWidthsBySizeLevel` (cell resize, step), and
  `GridTables::same_level_blocks` is `same_size_level_blocks` (any module that walks the neighbour tables). The rule
  behind it: an identifier or a test name says "SizeLevel", never a bare "Level".
- **10 Step and 12 I/O (where a scene's frame is chosen).** A cell can be split only when each of its coordinates is
  within `-(2^17 - 1)..2^17 - 1` at its own size level. With `L` size levels, only the size-level-0 cells from
  `-(2^(19 - L) - 1)` to `2^(19 - L) - 1` can be split all the way: -7 to 7 with 16 size levels. So the frame's
  origin is placed at (or just below) the domain's low corner or at its centre, never far from the scene, and the
  scene check fails before the run, with a message about the domain's size, when the domain spans more than
  `2^(20 - L) - 2` size-level-0 cells on an axis or lies outside that range. Otherwise the first sign is a
  `kSiblingCellMissing` that names no cell.
- **Every module that reads `grid::GridFailure`.** `other_cell` is also the big cell of a sibling failure, and
  `cell` is `CellId{0}` when the missing small cell lies outside the coordinate range; a message that prints `cell`
  must handle that case.

## The grid's block limit is 17,895,697 (2026-11-21)

`grid::kMaxCellBlocksLimit` is `floor((2^32 - 1) / 240) = 17,895,697`, not `(1u << 26) - 1 = 67,108,863`: a face
place index has 32 bits and a block has 240 face places. At most 1,145,324,608 cell places. No document below was
edited.

- **10 Step (design and plan; `plan-decisions.md`, `shared-interface.md`, `reconcile-todo.md` where they quote the
  number).** The run plan's cap on `limits.max_cell_blocks`, its "does not fit" test and any text or test that
  uses `1u << 26`, `2^26 - 1` or 67,108,863: use the constant, never the number. A scene that needs more blocks
  than the limit does not fit, whatever the memory.
- **12 I/O and CLI (design and plan).** The scene check that compares the domain's blocks at the smallest cell
  with the limit, and its message, quote the new number through the constant.
- **13 Surface, foam and spray (design and plan).** The same wherever a buffer or a test is sized at the limit.
- **11 Cell resize and batches.** `max_loaded_blocks` and any limit raised by `ApplyLimits` stay at or below the
  constant; a test at "limit + 1" uses the constant.

## From the particles pre-flight (A128 to A131, 2026-11-27)

Applied to `03-particles-design.md` the same day. The other documents below are not edited yet; each is changed
when its module's plan is written.

- **01 Contact (design 3.5, 5; done by the first task of the particles plan).** `MoveParticles` accepts an output
  that is exactly its own input array: `MovedParticles::position == MoveParticlesInputs::start_position` and
  `velocity_for_next_step == velocity.for_next_step` (the same pointer and count). Every other overlap stays
  `kBadInput`. After a failed call the inputs written in place are unspecified too (A128).
- **10 Step (design 5.5).** The call of stage 16 passes the velocity in place in every step and the position in
  place after a regrouping in the same try; a failed stage leaves a try that can only be rejected (A128).
- **11 Cell resize and batches (design 3, 5.1 step 6 and 7, 5.4; plan).** `particles::AdjustParticleCount(
  particles_try, grid, cell_blocks)` takes the resized grid only; `NeighbourhoodInputs`, `FindNeighbourhood`,
  `SplitParticle`, `MergeParticles` and `AdjustParticleCountResult::incomplete_split_count` do not exist (A129,
  A130). Check whether step 6's `solids::ComputeSolidsAtFaces` on the resized grid still has a reader; if not,
  delete it. `WriteBatch` adds `WriteParticleGroupsResult::copy_count` and `bytes_copied` to the copies it reports
  (A131).
- **08 Solids (design, "Cell resize and batches" paragraph).** `ComputeSolidsAtFaces` after a resize is no longer
  called for particles.
- **Every design that reads particle views.** `particles::BuildFieldsToRead` returns
  `std::expected<ParticleFields, ParticlesFailure>` (the range rule of A116); a particle's size level is a
  `particles::ParticleSizeLevel` (A132): whoever makes particles (I/O's start seeding, sources, gas markers) writes
  `ToParticleSizeLevel(cell_size_level)`, and whoever compares it with a cell uses `ToSizeLevel(size_level)`.
- **`shared-interface.md`** (rows "A complete neighbourhood", "Split, merge, count changes", the 3.5 list, the
  `NeighbourhoodInputs` note, the `AdjustParticleCountResult` copy) and **`01b-mesh-solids-cross-module.md`** (row
  3.1, 3.5): the same removals.
