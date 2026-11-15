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

## 11 Cell resize and batches

- Recommended options stand: liquid crosses regions as particles; grid-only velocities are followed back; the
  solves are not batched (whole grid on the GPU, CPU fallback).
- A9: remove sand from "practical scenes"; A10: gas arrays are remapped; the fire scene is workable in batches.
- Folder names: `cell_resize/` and `batches/` (recommended).
- Align with the updated step, grid and particles entries above.

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

## 01b Mesh solids (A80)

Written 2026-11-15 from `01b-mesh-solids-design.md` (draft). **Pending: apply none of these until the owner has
answered that design's section 11**; entries marked (Q2), (Q3), (Q5) depend on that question. Entries for module 1's
remaining plan tasks are the exception: they cost nothing, hold whatever the owner answers, and apply now. Each
edit is marked `(A80)` where it lands. Names are those of the 01b design, sections 3.1 to 3.3.

**01 Contact (plan `2026-11-14-rebuild-01-contact.md`; apply now):**
- Task 8, 8a: private helpers of `solid_contact_steps.h` that evaluate a solid take `const SolidContactInputs&
  inputs` and a `solid_index` (`ComputeDistanceForParticle(inputs, solid_index, pose, point)`); `FindFirstHit` calls
  one private `FindFirstHitOnSolid(inputs, solid_index, start, velocity, start_elapsed, duration, FirstHit*)` whose
  `switch` on the shape has no computing `default`; the touching rule stays inside each shape's branch.
- Task 9: the stepping search reads the gap only through `ComputeDistanceForParticle`.
- Task 10: a surface list entry holds the solid index and the normal recorded at the hit; a solid may appear more
  than once; no line skips "the solid just hit".
- Task 11 (and Task 10 step 6): the move to `d = tol` is one private `MoveToRestGap(inputs, solid_index, pose, point)`.
- Task 13: the carry in a turning solid's axes is one private function taking `inputs` and `solid_index`.
- Task 14: the "ends outside every solid, control enters" check is a helper in `contact_test_cases.h` over any solids.
- Task 15: `static_assert(sizeof(Solid) == 200)` beside the memory estimate.
- Task 16: the per-particle GPU function takes its scalar inputs as one struct.
- Design, after the owner's answers: 3.1 `ContactError::kMeshNotUsable`, `MeshProblem`, `ContactFailure::mesh_problem`;
  3.2 `SolidShape::kMesh`, `SolidFace::kMeshTriangle`, `kNoMesh`, `Solid::mesh_index`, the `ToSolid` form; 3.4 and 3.5
  `MeshShapes meshes` in `SolidContactInputs` and `MoveParticlesInputs`; section 1 "Owns" names the mesh shape; 5.7's
  "each convex solid is hit at most once" gains "a mesh may be hit several times (01b 5.6)".

**02 Grid, 03 Particles, 07 Materials, 09 Gas:** no edit. (Grid 5.9 and particles 5.6 each gain one sentence
pointing at 01b 5.3 and 5.5 for the 1-Lipschitz distance and the start-inside rule.)

**04 Transfer:** plan Task 5 fixture `BuildSolidsAtFaces` calls the mesh-aware `ComputeSolidVelocitySample`; one
test `BuildSolidsAtFaces, MeshCube_EqualsBox`. Decision 5 gains a note: a mesh part thinner than a cell has no
sides (Q5).

**05 Pressure:** `plan-decisions.md` "Pressure" (the thin-solid sentence) and design decision 7 gain "for a mesh,
solids' thickness finding (01b 5.2 check 8) keeps that case out or reports it" (Q5). No interface edit.

**06 Viscosity:** design 3 `ViscosityInputs` gains `contact::MeshShapes meshes` beside `solids`; 4 and plan Task 10
call `ComputeSolidVelocitySample(..., meshes)`; 5.7 states that a mesh solid counts as inclined (excluded until
Task 21); test `ShearChannel` twin with a tilted mesh slab, 5 %. (Q3: with a volume, Task 10's rule (a) and the
`1e-9 w` on-surface test need a tolerance instead.)

**08 Solids:** sections 1 and 10 drop "solid meshes (none)"; 3.2 `SolidSettings::mesh_index`,
`SolidsSettings::meshes`; 5.1 `BuildTry` and `BuildContactSolids` handle `kMesh`; 5.4 `ComputeSolidsAtFaces` and
`ComputeSolidCoverage` (and plan Task 18's GPU halves) take `MeshShapes` and pass the coverage cap as
`largest_distance`; `ComputeSolidsSummary::has_inclined_solid` is true for any mesh; `CheckSolidsCanRun` gains the
findings "mesh part thinner than the smallest cell" and "triangles much smaller than a cell"; plan Task 10
`kSolidsTooClose` samples a free mesh at its bounding box corners and centre (Q2); tests
`ComputeSolidCoverage, MeshCube_EqualsBox`, `..., Icosphere_BlockedAreaConverges`,
`Hydrostatic_PushIsWeightOfDisplacedLiquid` with a mesh, a driven mesh paddle.

**10 Step:** 3.2 and 3.6: the step owns one `contact::MeshShapeBuffers`; stages 3, 11, 16, 19 and
`FoamAndSprayInputs` fill `meshes`; 5.14 `CheckSceneCanRun` calls the mesh-aware `CheckSolid` and
`ComputeContactScale`, and `CheckMeshShape` once (`CapabilityReason::kMeshNotUsable`, NEW); 5.9 `BuildRunPlan` adds
`BuildMeshShapeMemoryEstimate`; 5.15 `BuildStartCells` takes `MeshShapes` (CPU) and tests the face centre as a
fifth point in "against a solid"; plan Tasks 7, 8, 8a, 26 gain a mesh scene (`BuildDrivenMeshScene`).

**11 Cell resize and batches:** 5.9 and plan Task 38: mesh rows are whole-run rows on the processor in every
batch. No interface edit.

**12 I/O and CLI:** scene file `solids[].shape` gains `mesh`, new field `mesh_file`; scene schema version raised;
new `io/mesh_solid.h` (read PLY, `contact::BuildMeshShape`, findings, volume, centre of mass, inertia;
re-centring for a free solid (Q2)); run folder `inputs/mesh_solids` listed and read on resume; mesh files in the
fingerprint; `BuildStartParticles` calls the mesh-aware distance; memory entry "io: mesh solids"; 5.10 budget
lines show mesh bytes; plan gains a task after Task 29d.

**13 Surface, foam and spray:** `FrameInputs` and `MotionSample` carry `contact::MeshShapes` (CPU); 5.3, 5.5,
5.6 call the mesh-aware distance with `largest_distance` 4 spacings; tests `BuildSurfaceField, BesideAMeshSolid_SameHeight`,
`MoveFoamParticles, StopsAtAMeshSolid`.

**14 Blender add-on:** 3.2 `Shape.MESH`; 5.1 item 2 exports the evaluated mesh as PLY (`inputs/solids/solid_<id>.ply`,
fan rule, scale applied); 10 drops "evaluated mesh shapes"; findings `solid_mesh_not_closed`, `solid_mesh_deforms`,
`solid_mesh_thinner_than_cell`, `solid_mesh_triangles_small`, `solid_mesh_too_many_triangles`; the digest covers
solid mesh files; default shape (Q8); subdivision level (Q6).

**15 Sand:** no edit.

**`shared-interface.md`:** (b) and (c) Contact gain the names of 01b 3.1 to 3.3. **`tradeoff-particles.md`:** 1.3
gains one line: mesh solids, 58 B per triangle on the processor (Q3).

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
