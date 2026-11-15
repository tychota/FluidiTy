# Decisions made while writing the plans

Written 2026-11-14. Each entry is a decision a module's plan made beyond its design that other modules' plans
must follow. Plans written later read this file after `shared-interface.md`. Where it disagrees with a design,
this file wins (the designs are not edited again).

## Applies to every plan

- **Running the proof of concept for comparison numbers.** Never check `main` out into the worktree. Export it
  with `git -C /Users/tychota/Documents/Code/openSource/FluidiTy archive main | tar -x -C <temp dir>` (or
  `git worktree add --detach <temp dir> main` outside the rebuild worktree), build and run there, and commit
  only the resulting numbers as a `testdata` file with a note saying how they were produced. The throwaway driver
  is never committed.
- **A `no_gpu` test "on the GPU without a GPU gives `kRunOnCallFailed`"** exists for every operation that has a
  `_gpu.cu`, not only where a design lists it.
- **Out of memory in a module with no `kOutOfMemory` value** is reported as that module's `kRunOnCallFailed`
  with the bytes asked in the failure's magnitude field.
- **Sizes in memory estimate entries are `sizeof` of the real types**, with a `static_assert` on each; where a design gave
  packed byte counts, the `sizeof` value replaces it.

## Foundation `base/`

- `MemoryEstimateEntry` has a field `MemoryWhere where { kRunOn, kCpu }`, and `ComputeMemoryBytes(rows, counts, where)`
  sums only the rows kept there. Every module's rows set it.
- The processor description header is `base/processor_description.h`.

## Grid

- `GridError`, `GridFailure`, `ToMessage`, `ToName` live in the shared header `grid/grid_error.h`.
- `GridPiece` is declared in `grid/grid_piece.h` (included by `cell_block_range.h`); the glossary has separate
  rows for "face piece" and "grid piece".
- `CellId{0}` and `FacePieceId{0}` are never valid ids. Ids are opaque: other modules never build or split one
  by hand; they use `ToCellId`, `ToCellCoord`, `ToSizeLevel`, `ComesBefore`.
- `ComputeGridCounts` and `CheckSizeStepRule` sort a CPU copy of the cell list (8 bytes per cell); they are not
  allocation-free.
- No-allocation tests count allocations through a replaced global `operator new` in the test executable.

## Transfer

- **Working buffers** (A31, 2026-11-14: one form per stage, below). `transfer::TransferBuffers` with `BuildTransferBuffers(run_on, max_particle_count, material_count)` holds the per-particle sums and the
  small status record. Every stage (`ParticlesToGrid`, `SaveGridBeforeForces`, `SaveGridAfterPressure`,
  `GridToParticles`, `ComputeCellsFromParticles`) takes the buffers as its last argument and
  allocates nothing. (The forms without them, which allocated for one call, are removed by A31: tests and tools
  build a `TransferBuffers`.)
- **Builders.** `BuildSolidsAtFaces`, `BuildGridFromParticles`, `BuildGridSnapshot(..., FilledFromSolved)`
  with `enum class FilledFromSolved { kNotKept, kKept }`, `BuildCellsFromParticles`,
  `ToSolidsAtFacesOnProcessor`, `ToGridSnapshotOnProcessor`.
- **Memory per material set per cell block:** 59,856 + 512 x ingredients bytes (48,336 + 512 x ingredients for
  a set with no solve). These replace the packed figures in the designs.
- `CellsFromParticles::ingredient_volume` is the current volume (fraction x volume_ratio x rest_volume); each
  cell's row adds up to `current_volume`.
- Transfer calls `particles::CheckGrouping` with a maximum move of 0: particles must not have moved between
  grouping and a transfer stage.
- `kMaxFitRows = 64` caps the rows of one fill fit.

## Foundation `base/` (continued)

- `kUp` is `inline constexpr Vec3 kUp{0, 0, 1}` (not an `Axis`). `ProcessorDescription { model; driver; runtime; }`.

## Contact

- Analytic contact tests use a fixed tolerance of `0x1p-30`; expected values include the rest gap (a particle
  in contact rests one tolerance outside the surface).
- A particle that **starts** closer to a solid than the tolerance (down to minus `start_overlap_tolerance`) is
  moved out to one tolerance before the sweep, whatever its velocity. So particles placed by a source next to
  a solid do not fail the end check. `kEndsTooCloseToSolid` is tested with a different case (a squeeze between
  two solids closing faster than the particle can leave).
- `CheckSolid` and `FindFirstHit` each have an overload with a `ContactCloseCall*` output.
- `ContactRecord` is 56 bytes (the design said 64); the memory estimate entries use `sizeof`.
- `fluidity_add_shared_test(...)` (CMake helper, added by the contact plan in `cmake/fluidity_targets.cmake`)
  registers tests of shared CPU-and-GPU headers as label `cpu` in a CPU build and `no_gpu` in a CUDA build.
  Later modules use it for their shared-formula tests.
- Private GPU entry points are named for what they do without `Gpu` or `Device` in the identifier (contact:
  `MoveParticlesOneThreadPerParticle` in `move_particles_gpu_calls.h`); `_gpu` appears only in file names.
- `ComputeContactScale` takes the largest coordinate from the solids (origins at both ends of the step plus
  their half sizes) and the domain box, not from particles.

## Pressure

- **Open sea.** The step calls `solids::DriveWaves`; `solids::DriveWaves` calls
  `pressure::ApplyOpenSeaRelaxation`. `pressure::OpenSeaSettings` has a field `contact::DomainBox domain_box`.
- `EquationInputs` has `lightest_density` and `max_volume_error_rate`; `PressureEquations` has
  `max_volume_error_rate`, an opaque `const EquationArrays* arrays` and `build_number`.
- `BufferSizes::max_cells` is a multiple of 64; every array is built from `max_cell_blocks = max_cells / 64`.
- `kStepTooLongForSurfaceTension` is returned by `FindLiquidSurface`; `kStepTooLongForAirPocket` by
  `FindAirPockets`. `kPressureLawOutOfRange` is returned by the equation build when `bulk_stiffness <= 0`.
- A grid-edge face piece with open area has air at atmosphere on its other end.
- `kCellSplitBySolid` is decided from corner samples only: a solid thinner than a cell that passes between
  corners is not seen. Solids' plate-thickness rule is what keeps that case out.
- The domain's top side is marked in `grid_edge_is_open_to_sky` (filled by solids).
- The ladder leaves free-solid terms out; `ApplyPressureLadder` removes each sealed blob's input total before
  the pass.
- A progress read (and so a cancel check) happens before the first solve iteration.
- The pressure plan needs `proofs/` and `reference/` to exist already (created by the particles and transfer
  plans); it adds `proofs/Fluidity/PressureEquations.lean` and a dense Python reference with three cases.

## Commits

- Commit messages carry no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no session link); the owner is the
  author. Each commit gets a git note in `refs/notes/claude-session` (`session:` and `models:` lines), written with
  `tools/add_session_note.py` (owner answer A55, 2026-11-15; replaces A39).

## Grid (continued)

- `grid::CellBlockRange` (the plain struct of two block indexes) is declared in the shared header
  `grid/grid_ids.h`, so shared CPU-and-GPU headers of other modules may use it. `cell_block_range.h` keeps the
  CPU-only functions that build ranges.

## Particles

- `ParticlesSettings::max_cell_blocks` sizes the per-block tables (`group_begin`, `cell_begin`, written marks);
  zero is `kBadInput`. `ApplyParticleLimits` can raise it together with the particle limits, so
  `step::ApplyLimits` raises the grid limit and the particles' block limit in one call.
- `ParticleFields` and `ParticleFieldsToChange` carry `RunOn run_on`. `ComputeAmounts` and
  `ComputeAmountsPerBlock` return `std::expected<..., ParticlesFailure>` (no NaN signalling, no asking the
  runtime where a pointer lives).
- View pointers point at particle 0 of the material; `group_begin` holds whole-material indexes;
  `particle_count` is the count in the range. A user of `ParticleFieldsToChange` that needs `group_begin`
  takes the matching `ParticleFields`.
- Removal marks: 1 byte per particle (0 kept, `1 + RemoveReason` removed, 255 replaced by a split or merge).
- Memory: 405 bytes per liquid particle (the design said 412); rows use `sizeof`.
- `ComputeMergedFractions` (NEW) merges ingredient fractions; `NeighbourhoodInputs` and `Neighbourhood` are in
  the shared `split_and_merge_math.h`.
- `AdjustParticleCount` visits particles in grouped order (cell place, then id) inside a block.
- `BuildParticlesFromSnapshot` sets `grouping_version = 1`.
- `proofs/` (Lean project with `Merge.lean`, theorem `merge_preserves_ingredient_volumes`) is created by the
  particles plan. Proofs are built by hand with `lake build`, not in the gate; `lake` is not installed on the
  development Mac, so a proof task records "written, not built here" unless a Lean container is used.

## Pressure (continued)

- **Two liquids that do not mix are supported.** The pressure plan implements the sharp boundary and removes
  the exclusion: `PressureError::kMaterialBoundaryNotSupported` does not exist at the end of the pressure
  module (`ToName(PressureError)` covers 15 values). Consequences for the step: there is no
  `CapabilityReason::kLiquidsThatDoNotMixTouch` and no stable name `liquids_that_do_not_mix_touch`;
  `kCapabilityReasonCount` is 18.
- In a one-material distance field with surface tension, every cell whose centre distance is under two cell
  widths in magnitude, on both sides of the boundary, gets curvature samples.

## Foundation `base/` (continued)

- `enum class RunOn : std::uint8_t { kCpu, kGpu }` is declared alone in the **shared** header `base/run_on.h`
  (usable in GPU code and in other modules' shared headers). `RunOnError`, `CheckCanRunOn`, `ToMessage` and
  `ToName` are in the CPU-only header `base/check_can_run_on.h`.

## Particles (continued)

- `ApplyParticleLimits(Particles&, std::span<const std::size_t> max_particle_counts, std::uint32_t max_cell_blocks)`.
- `ComputeAmounts` returns `std::expected<ParticleAmounts, ParticlesFailure>`; `ComputeAmountsPerBlock` returns
  `std::expected<void, ParticlesFailure>`.

## Solids

- `pressure::FreeSolidPush::solid_index` is the index of the free solid among the free solids of
  `SolidsTry::solids()`, in that order; `ToFreeSolidInverseInertia` gives two `Mat3` per free solid in the same
  order.
- `std::expected<SolidsBuffers, SolidsFailure> BuildSolidsBuffers(const SolidsSettings& settings,
  std::size_t max_particle_count, std::uint32_t max_cell_blocks);`
- `std::expected<SolidCoverageValues, SolidsFailure> BuildSolidCoverageValues(RunOn run_on,
  std::uint32_t max_cell_blocks);` (the step builds its coverage arrays with it).
- **Sign of work.** Every work in solids is the energy a solid gives the material, positive when a paddle
  pushes liquid: `flow_work = step_duration * sum over cells of pressure * (solid volume rate into the cell)`.
  `ContactOnSolids::work` and `ExchangeTotals::driven_work` use the same sign.
- `try_id = (state_version << 32) | tries_since_commit`; `CommitTry` with a try built before another commit
  returns `kWrongTime`.
- `ApplyPressureToSolids` can fail with `kSolidMovesTooFar`; the step shortens the step and retries on it, as
  for `BuildTry`.
- `MotionMap` is a move-only view into `SolidsBuffers`; a map from an earlier build on the same buffers gives
  `kWrongTime`.
- `CollideSolids` takes at most `blocking_materials.size() + 1` material sets per call (`kBuffersTooSmall`
  otherwise); at one face, pairs run in `material_sets` order.
- `DriveWaves` has a private `drive_waves_gpu.cu`.
- `CheckSolidExchange`: the liquid side is taken from `GeometrySnapshot` face pieces; a non-liquid cell's
  pressure counts as 0; the no-penetration check fails with `kExchangeNotBalanced`;
  `particles_inside_solid` is 0 or 1 (the step keeps the real count).
- `Solids` owns copies of its settings, path keys and blocking materials.
- Test libraries `fluidity_solids_test_scenes` (`solids::testing::SceneSolids`) and
  `fluidity_solids_liquid_loop` (`solids::testing::LiquidLoop`, a hand-written liquid step used until the step
  module exists).
- **Tests the step plan must add** (from the solids plan): `WavePaddle, RegularWave_SpeedAndHeight`;
  `Cut, PlateThroughBlock_SidesNeverMix`; `Platen, SnowBesideWater_SetsKeepTheirVelocities`;
  `FreeBox, FallsThroughGas_NotPushedByGas`; ~~`Batches, FloatingBox_SameBytesAsOnePiece`~~ **replaced** by
  `Batches, FloatingBox_WithinStepToleranceOfOnePiece` ("Batched solves (A15)" table; in the batches plan);
  the two floating-box tests run again through `TryStep`.

## Materials

- `BuildMaterialBuffers(RunOn run_on, const MaterialTable& table, std::span<const std::size_t> max_particle_counts,
  std::uint32_t max_cell_blocks)`; `MaterialBuffers::run_on()`.
- The step uses the forms that allocate nothing: `ApplyInternalForce(..., std::span<InternalForceBudget>
  budget_of_block, MaterialBuffers& buffers)` and `ComputeCellSqueezeValues(inputs, bulk_stiffness,
  target_volume_ratio, const MaterialBuffers& buffers)`.
- `kStepTooLong` reports `material_index` of the material with the smallest limit and `magnitude = step_limit`;
  a step exactly at the limit is accepted.
- With no liquid in the scene, `surface_tension_max`, `surface_tension_with_air` and `lightest_density` are 0.
- No `gas/` header is included by materials: the gas ingredient's rest density is an input value (the caller
  passes `gas::ComputeAmbientDensity(law)`).
- No `default:` in any switch on `MaterialKind` or `LimitCase` (so the sand module's additions are found by the
  compiler). Enum values are fixed with `static_assert`.
- The work-rate oracle's expected value in the complete-neighbourhood test is -87.5 W for both rates (the
  design's +87.5 had the wrong sign).
- Test library `fluidity_materials_test_scenes` (namespace `materials::test`).
- A view or array whose `run_on` differs from the buffers' is `kBadInput` (rule for every module's batch
  operations).

## Gas

- Gas law forms: `ComputeGasDensity = p0 / (gas_constant * T)`, `ComputeAmbientDensity(law) =
  ComputeGasDensity(law, atmosphere_pressure, ambient_temperature)`, `ComputeTemperature = p0 /
  (gas_constant * rho)`.
- Lift is computed from densities, not through temperature: `ComputeLiftFromDensities(law,
  background_pressure, gas_density, smoke_density)` (the temperature round trip is not exact and would push
  still air).
- `ComputeJoinedPressure`, `ComputeSourceCoverage`; key sampling by linear interpolation in the key interval.
- `GasBufferSizes::max_cells` is a multiple of 64. `GasBuffers` holds a `carried_pressure` copy so
  `SolvePressure`'s output never aliases its input.
- The gas `EquationInputs` take `lightest_density = ComputeAmbientDensity(law)` and
  `max_volume_error_rate = settings.pressure_solve.max_volume_error_rate`; `SolveGas` calls
  `pressure::ApplyAddedVolumeRate` right after `ComputeGasAddedVolumeRate`.
- Sources are applied before the flow (the branch applied them after): one-step numbers differ from the branch
  for that reason, and the comparison test says so.
- The gas mass balance leaves `gas_mass_from_volume_error` out (it returns as `gas_mass_from_geometry` the next
  step); it is reported but not counted twice.
- New markers sit at cell-eighth centres (`ComputeCellEighthCenter`); a marker view on a different processor
  from the amounts is `kBadInput`.
- **`FindCellsToKeepActive` marks only cells of the existing grid.** Growing the active region (creating the
  cell blocks around marked cells and source boxes) is the step's job in `ChooseActiveCells`: the step adds
  the cells next to every marked cell to the cell list before `grid::BuildGrid`.
- `ToGasSnapshot(const GasAmountsSnapshot&, const GasLaw&, const grid::GridFrame&)` (three arguments).
- Test library `fluidity_gas_test_scenes` (namespace `gas::testing`), with `BuildSmallFireScene`
  (0.5 x 0.5 x 1.0 m, 0.0625 m cells).
- **Tests the step plan must add** (from the gas plan): `TryStep, GasHeatPulse_RejectedAndCommittedFramesKept`;
  `TryStep, GasStepLimitTakenIntoNextDuration`; `TryStep, SparseFireKeepsItsActiveCells`;
  ~~`TryStep, BatchedFireBesideWater_SameBytes`~~ **replaced** by `TryStep,
  BatchedFireBesideWater_WithinStepTolerance` ("Batched solves (A15)" table; in the batches plan).

## Step

- `StageOutcome` is declared in `step/step_error.h`. A private `ToStageOutcome(const <Module>Failure&)` in
  `stage_outcome_from_modules.h` maps each module's errors to step reasons.
- Shared random words (step module, shared header): `struct RandomWords { std::uint32_t word[4]; }`,
  `ComputeRandomWords(std::uint64_t child_number, std::uint64_t source_id, std::uint64_t key)` (Philox
  4x32-10), `ComputeSignedUnitValue(std::uint32_t word)`. Foam and spray use these too.
- `struct DomainSideCrossed { contact::DomainSide side; std::uint8_t crossed; }` from `FindDomainSideCrossed`.
- Which stages run is fixed once per simulation; a null `StageTable` entry means "not a stage here";
  `kStages` holds stages 1 to 20; `kResizeCells` is recorded only by `AdvanceTo`.
- When there is no solid but the domain box, stage 2 fills `solids_at_start` and `solids_at_end` with
  `solids::ComputeSolidsAtFaces` once per new grid version.
- `ResizeTry` marks the simulation busy: `TryStep`, `ReadSnapshot` and `ApplyLimits` return `kTryInProgress`
  while one is alive.
- `grid::BuildGrid` must accept an empty cell list (an empty scene is valid).
- Test library `fluidity_step_test_support`: `TestScene` (owns everything a `ReadyScene` borrows) and its scene
  builders, `ExpectSameBytes`, `BuildRejectingStageTable(Stage, RejectReason)`, `CallbackLog`.
- Small reference scenes used in tests: wave 1.6 x 0.2 x 0.4 m, 0.2 m deep, 2.5 cm cells, paddle stroke 0.02 m
  with a 1 s period, open-sea zone x 1.2 to 1.6; syrup as the viscosity design at 5 mm cells; fire as the gas
  design at 0.1 m cells; snow example a 0.2 m cube at 2.5 cm cells; `BuildSmallFireScene` as the gas plan.
- Names exempt from the verb rule (in `tools/check_rules_allowed_names.txt`): `TryStep`, `TryStepWithStages`,
  `AdvanceTo`, `CommitResize`, `RejectResize`.
- **Tests handed to the cell resize and batches plan:** `Batches, FloatingBox_WithinStepToleranceOfOnePiece` and
  `TryStep, BatchedFireBesideWater_WithinStepTolerance` (the A15 names; were `..._SameBytesAsOnePiece` and
  `..._SameBytes`, **replaced** by the "Batched solves (A15)" table); the split-and-merge part of
  `Cut, PlateThroughBlock_SidesNeverMix`.

## Viscosity

- `viscosity::ViscosityInputs` has a field `pressure::PressureBuffers* pressure_buffers = nullptr;` (the buffers
  that built `pressure_equations`); null is `kBadInput`. The step's `SolveViscosityAndPressure` call sets it.
- Viscosity treats a pressure as accepted when `pressure::CheckFlowImbalance` returns a report;
  `PressureError::kNotConverged` means "not yet".
- A face with `face_inertia == 0` is "given": its value is `velocity_before_pressure`, as in `SolvePressure`.
- `max_cell_count` is a multiple of 64 and `max_face_count == max_cell_count / 64 * 240`.
- ~~Samples are ordered by ascending `CellId` with `BuildStableOrder`.~~ **Replaced** by "Batched solves (A15)",
  section "Viscosity and gas": in both modes, samples are ordered by the cell's block, then ascending `CellId`,
  kind, axis pair, quarter (reconcile pass 2026-11-14).
- A grid-version mismatch is reported as `kBadInput` (the enum has no `kWrongGridVersion`).
- **Exclusions.** The viscosity plan lifts three exclusions in its own tasks (inclined solid, driven solid
  cutting through cells, size change at a surface, solid or viscosity change), each with the fallback "if the
  lifting test fails, commit it as known-open and keep the exclusion". Free solid and snow with viscous liquid
  stay excluded, as known-open tests. Whoever executes the step plan reads `ToName(ViscosityError)` on the
  branch at that time: for each exclusion that was lifted, the matching `CapabilityReason`
  (`kViscousLiquidWithInclinedSolid`, `kViscousLiquidWithMovingSolidThroughCells`,
  `kViscousLiquidAtSizeChange`) does not exist and `kCapabilityReasonCount` is lower by one (15 if all three
  were lifted, 18 if none). **Replaced in part** (A24): snow is no longer excluded (`kSnowNotSupported` and
  `kViscousLiquidWithSnow` removed, the snow known-open test passes); counts 17 at most, 14 if all three lifted.
- If the free-surface slab test gives `-2 mu a * filled_volume / open_volume` instead of `-2 mu a`, the
  implementer stops and reports (possible conflict between the test's expected value and the weight rule).

## Names that broke the glossary (ruling of 2026-11-14)

Writing the glossary from the owner's naming review showed that several names in the module designs use words
the owner decided never to say. The glossary wins. These renames apply everywhere (designs, plans, code); a
single search-and-replace pass over the plans is made when all plans are written, and anyone who meets an old
name uses the new one.

| Old (in designs and early plans) | New | Banned word |
|---|---|---|
| `GridStorage`, `BuildGridStorage`, `ReleaseStorage` | `GridBuffers`, `BuildGridBuffers`, `TakeBuffers` (A29) | storage (say buffers) |
| `InternalForceBudget`, `budget_of_block`, `ComputeInternalForceBudgetTotal` | `InternalForceTotals`, `totals_of_block`, `ComputeInternalForceTotal` | budget |
| `ContactRecord` | `ParticleContact` | Record (no role suffix) |
| `SolveOutputs`, `AirPocketOutputs`, `SolveGasOutputs` | `SolvedArrays`, `AirPocketArrays`, `SolvedGasArrays` | role suffix other than Inputs, Settings, Result |

The rules check (Foundation Task 3) will find more of these when code is written. The rule for the implementer
is the same each time: the glossary wins; rename, list the new name in the commit message, and tell the
coordinator so later plans follow. A comment that gives the textbook or literature word once ("papers call
this ...", "textbook: ...", "also called ...") is exempt from the banned-word check on that line.

## Build targets

- The library's CMake target is `fluidity` (Foundation). The command-line program's CMake target is
  `fluidity_cli`, with `OUTPUT_NAME fluidity` so the executable file is still called `fluidity`.

## Surface, foam and spray (render output)

- Headers: shared `render_output/lattice.h` (`LatticeFrame`, `LatticeCoord`, `kLatticeBlockWidth`, `ToBlockKey`,
  `ComputePointPosition`, `FindLowPoint`); CPU-only `render_output/lattice_values.h` (`LatticeBlocks`,
  `LatticeValues`, `FindBlock`, `LatticeReading`, `ComputeLatticeReading`); shared
  `render_output/render_limits.h` (`RenderLimits`) and `render_output/frame_identity.h` (`FrameIdentity`);
  shared `foam_and_spray/foam_source.h` (`Ramp`, `FoamSourceSettings`, `FoamSourceEvent`).
- Exceptions and RTTI are on only in `render_output/interchange/*.cc` and `cli/read_command_line.cc`. Each
  public writer catches everything: `std::bad_alloc` becomes `kOutOfMemory`, anything else `kFileWriteFailed`.
- `InterchangeLibraries::missing_resource_folder` (empty when OpenUSD found its plugin resources); the CLI's
  `capabilities` reports it. CMake copies USD's plugin resources to `usd_resources/` beside the CLI.
- Every output file carries the keys `fluidity_frame`, `fluidity_state_version`, `fluidity_time`,
  `fluidity_frames_per_second`, `fluidity_contract_version`, `fluidity_channels`.
- Step recorder: a try ends at a frame time when `|end - (first + n * frame_duration)| <= 1e-9 *
  frame_duration`; `RecordStep` reads `particles_try` and `fields` only on such a try; `after_pressure` must
  have exactly one entry.
- Foam random words: `step::ComputeRandomWords(number in event, block key, (step_count << 9) | point in block)`.
- Static library target `fluidity_interchange`; CMake helper `fluidity_add_interchange_test` (labels `cpu` or
  `no_gpu`, plus `interchange`). Conan packages in the container are kept on the persistent volume.
- The render-output plan creates every `render_output/` header; its Task 47 completes the I/O seams that name
  render-output types (render settings in the scene, the step callbacks, the frame worker, `capabilities`,
  linking the CLI). The I/O plan therefore declares those seams against an interface it owns (`FrameWriter`)
  and does not include `render_output/` headers.
- Library tasks stop and report (ruling by the coordinator) if a pinned package is missing from Conan Center,
  if OpenVDB pulls Boost or Blosc, if Alembic cannot turn HDF5 off, or if static OpenUSD cannot register its
  schema types.

## Names that broke the glossary (continued)

| Old | New | Banned word |
|---|---|---|
| `step::Support` (enum of the capability check), `kSupported` | `step::CapabilityLevel`, `kCanRun` (`kExperimental`, `kUnavailable` unchanged) | support |
| test libraries `fluidity_step_test_helpers`, `fluidity_interchange_test_helpers`, `fluidity_render_output_test_helpers` (replaced by `_test_support`, reconcile item 29) | `fluidity_step_test_support`, `fluidity_interchange_test_support`, `fluidity_render_output_test_support` | support |
| `StoredFrameInputs`, `FileList::ignored_tail_bytes` | `KeptFrameInputs`, `ignored_end_bytes` | stored, tail |

## I/O and CLI

- Targets: `io/` sources are part of the library target `fluidity` (links FlatBuffers, BLAKE3, Threads
  privately); `cli/` sources form the OBJECT library `fluidity_cli_parts` and the program `fluidity_cli`;
  `cli/read_command_line.cc` is alone in the OBJECT library `fluidity_command_line` (exceptions and RTTI on).
  Test library `fluidity_io_test_support` (namespace `fluidity::io::testing`).
- I/O includes no `render_output/` header. Its seams: `FrameWriter` (with `start(void* context)`),
  `FrameOutput` and `KeptFrameInputs` carrying `std::int64_t frame` and `double time`, a `file_pattern` without
  extension (the writer appends its own), an empty scene table `outputs.render: RenderSection`, and the
  program's default writer `cli::BuildFrameWriterWithoutProducts()` which lists each declared output as "not
  built" with reason `no_product_writer`. The render-output plan's Task 47 fills these in.
- `RunParts { FrameWriter frame_writer; OutputQueueSettings output; RunCallbacks callbacks; }`; `Run` builds and
  owns `CellResize` and `BatchedRun` after `step::BuildRunPlan`.
- `RunOverrides::assume_memory_bytes` (`check --assume-memory`; tests use it to force a batched plan).
- `ReadRunFolderFacts(const Path&, FileCheck)` with `enum class FileCheck { kListOnly, kEveryListedFile }`.
- Base failures in findings: `FailingModule::kBase` with `module_error` = the `BufferError` value, or 64 + the
  `RunOnError` value.
- Every file kind and the events protocol start at version 2 (`io/schema_versions.h`).
- Start particles: 8 per smallest cell on a lattice, kept when the unjittered point is inside the region and
  at least half the spacing from every solid; jitter from `step::ComputeRandomWords(lattice_index, region_id,
  seed)`.
- `io/material_presets.h`: `MaterialPreset { kCustom, kWater, kSyrup, kSnow, kSpringySolid, kAir }`,
  `kMaterialPresets`, `FindMaterialPreset`.
- Process tests are pytest in `tests/cli/`, run as the CTest `cli_end_to_end`; the independent decoder is
  `reference/io/read_array_file.py`.

## Cell resize and batches

- **Step, test scenes added by this plan's Task 1** (in the step's test helpers): `BuildDeepTankScene`,
  `BuildMixedSizeColumnScene`, `BuildRisingFloorScene`, `BuildStandingWaveTankScene`,
  `BuildTwoLiquidsDeepScene`, `BuildSmokeOverDeepWaterScene`, `BuildFloatingCubeScene` (0.1 m biggest cell,
  3 levels, `has_cell_resize = true`).
- **Step, `StageTable` split.** `StageTable` has `std::array<StageFunction, kStageCount> run_whole_grid{}`,
  non-null only for `kFindLiquidSurface` (`FindMaterialBoundaries`, `FindAirPockets`,
  `CheckViscosityCanRun`), `kMoveParticles` (the grid-only move) and `kCheckStep` (`CheckSolidExchange`, the
  gas changes, the next limits). A one-piece run calls it right after `run`. Stage 10 runs on the whole grid.
- **Step, loaded try.** `FindLoadedTry(Simulation&)`, `FindLoadedGridBuffers(Simulation&)`,
  `ApplyGridPiece(StepTry& loaded_try, const grid::GridPiece& piece)`, `struct BlockCopy { from_block;
  to_block; block_count; }`, `WriteStepArrayBlocks(const StepTry& from, std::size_t array_index,
  std::span<const BlockCopy> copies, StepTry& to)`. The loaded batch is built by `step::BuildSimulation`
  (not by `BatchedRun`); `BuildBatchesMemoryEstimate` is for reporting only, so nothing is counted twice.
- `cell_resize::ApplyResizeToCellAmounts(const ResizeMap&, old_amounts, old_weight, new_amounts)` remaps the
  three grid-only amounts with one limiter per split cell.
- `merge_ready_duration` grows by the committed time since the last committed merge test; a change of
  durations alone commits a same-cell resize with version + 1 (so a "nothing to do" resize still keeps them).
- `ApplyGridOnlyRegionToGrid` writes `velocity` = the carried velocity and `momentum = mass * velocity`.
- `ReadLargestGridOnlySpeed` returns the latest try's value; the step reads it only in `CommitStep`.
- `BuildBatchPlan` cuts by layers of blocks without building the start grid; the exact cut is found per try
  inside `ApplyStagesInBatches` by bisection.
- In a batched GPU run, a `CellResize` with `run_on = kCpu` keeps CPU mirrors of the arrays it reads.
- If the guard test on stage order (batched pass order against the one-piece order of stages 16 to 18) fails,
  or a stage rejects on incomplete inputs in the outer border blocks of a piece, the implementer stops and
  reports: the fix belongs in the commits of the module that owns the stage.
- Test library `fluidity_cell_resize_test_support`.

## Names that broke the glossary (continued)

| Old | New | Banned word |
|---|---|---|
| render output `ChannelStorage`, field `storage` | `ChannelValueType`, `value_type` | storage |
| `WitnessOutcome` | `WitnessCaseResult` | role suffix |

## Blender add-on

- Blender on the development Mac is 5.2.3 LTS (build `e13b728c6880`, Python 3.13.13); Blender 5.3 is not
  installed, so every 5.3 result is recorded "written, not run in Blender" until it is.
- Checked in Blender 5.2.3: `bpy.data.cache_files.load` and `bpy.data.volumes.load` do not exist. A cache file
  is made with `bpy.ops.cachefile.open(filepath=...)` (works in background mode); a volume with
  `bpy.data.volumes.new` then setting `filepath`. `--offline-mode` is a global Blender option, not an option
  of `extension build`.
- Renames forced by the glossary: package `logic/` (was `core/`); `attach` for joining outputs to objects
  (`attach_plan.py`, `output_attach.py`, `ATTACH_ROUTES`, `fluidity.attach_outputs`, `fluidity.reattach_run`);
  `path_in_blend`; `copy_problem_report`; `RunState.NEEDS_NEW_RUN`; `starter_assets.py`; "artist's journey"
  (`tools/blender/run_artist_journey.py`); "add the package" for installing.
- Test levels: (a) plain pytest in `tests/addon/logic/` (CTest `addon_logic`) and `tests/addon/with_fluidity/`
  (CTest `addon_with_fluidity`); (b) background Blender in `tests/addon/in_blender/`, run by
  `tools/blender/run_blender_tests.py`, not in the gate; (c) a Blender window, recorded in
  `docs/blender-runs.md`.
- Scene file field names come from the branch's `scene_file.fbs` (I/O), not from the add-on design's outline;
  a test fails on any field path the schema does not define.
- Witness records when only one Blender version ran: the version that ran gets its result; the other keeps
  `kNotRun`. A USD candidate is chosen only when it passed in both 5.2 and 5.3; a failure in either removes it.
  So no USD format can be chosen until Blender 5.3 is installed and the witness tests are run there.
- The witness file writer is `fluidity_write_witness_files` in `native/fluidity/cli/write_witness_files.cc`.

## Sand

- A known limit is `step::CapabilityLevel::kExperimental`; an exclusion is `kUnavailable`.
  `CapabilityLevel { kCanRun, kExperimental, kUnavailable }` is the full list.
- `kSandSettlingCreep` (level `kExperimental`, one finding per sand material) and `kViscousLiquidWithSand`
  (level `kUnavailable`, suggestion `kSetViscosityToZero`); stable names `sand_settling_creep`,
  `viscous_liquid_with_sand`. `ViscosityError::kSandNotSupported` and `ViscositySceneSummary::has_sand` are
  appended last. `MaterialPreset::kSand` is appended after `kAir`. `common.fbs` `MaterialKind` gains `sand`.
- Every switch on `MaterialKind` outside materials gets its sand case in the commit that adds `kSand`.
- Step additions used by the diagnosis: `StepReport::particle_count_by_case` and
  `StepTry::solid_reactions(std::size_t material_index)`.
- Decision records go in `docs/decisions/sand-NN-<words>.md` (Problem, Hypothesis, Experiment, Result,
  Decision). In the investigation a confirmed fix may be committed only inside a method sand owns; a fix in
  another module's method, or any change to the angle measure or the pile setup, is written up as a proposal
  and stops for the owner. No task tunes a number or adds damping.

## The single rename pass (done 2026-11-14)

The renames in the "Names that broke the glossary" tables were applied by search-and-replace to every plan,
every module design and `shared-interface.md`, except the capability enum: plans may still write
`step::Support` / `kSupported`; read them as `step::CapabilityLevel` / `kCanRun`.

## Foundation `base/` (as built, 2026-11-14)

- `base/run_on.h` (shared) declares both `enum class RunOn : std::uint8_t { kCpu, kGpu }` and
  `enum class RunOnError : std::uint8_t { kNone, kGpuNotBuilt, kNoGpuFound, kGpuCallFailed, kGpuMathDiffers }`.
  `base/check_can_run_on.h` (CPU only) declares `CheckCanRunOn`, `ToMessage(RunOnError)`, `ToName(RunOnError)`.
- Each module folder has its own `CMakeLists.txt` (added from the top-level file with `add_subdirectory`); it
  lists public headers with `fluidity_add_headers(...)` (each header is then compiled alone with the library's
  flags), adds sources with `target_sources(fluidity PRIVATE ...)`, and tests with
  `fluidity_add_test(<name> SOURCES <files> LABEL <cpu|gpu|no_gpu>)`. Tests are discovered at test time.
- `scalar_math.h` states two groups: same bits on CPU and GPU (`Floor`, `Abs`, `Sqrt`, `Min`, `Max`, `IsFinite`,
  `Length`, `Normalized`) and may differ in the last bits (`Cbrt`, `Exp`, `Expm1`, `Log`, `Pow`, `Sin`, `Cos`,
  `Rotation`). A formula that uses the second group is compared between CPU and GPU with a tolerance.
- The CUDA compiler of the pinned toolkit accepts GPU code that reads `kUp` and `kIdentity` at run time.
- Build commands use a Conan home inside the worktree (`CONAN_HOME=$PWD/out/conan-home`) and
  `-s compiler.cppstd=23`; see `README.md` "Build".

## Amendments of 2026-11-14 to the Blender and render-output plans

Owner answers A16 (Blender 5.3 only), A19 (newest library versions), A20 (wind from Blender's force fields) and
A22 (ordinary words allowed outside their fluid-jargon sense) changed the Blender add-on plan
(`2026-11-14-rebuild-14-blender-addon.md`) and the render-output plan
(`2026-11-14-rebuild-13-surface-foam-and-spray.md`). These entries replace the "Blender add-on" and "Surface, foam
and spray" entries above wherever they differ.

- **Blender version.** Blender 5.3 is the one supported version. The development Mac has Blender 5.3.0 (build
  `02f0899d78e3`, Python 3.13.13) at `/Applications/Blender.app/Contents/MacOS/Blender`; 5.2.3 is gone, and
  `/opt/homebrew/bin/blender` is a wrapper that starts the same 5.3.0. The extension manifest says
  `blender_version_min = "5.3.0"`, no maximum; a later Blender loads the add-on with one note. Background-Blender
  tests are runnable on this Mac. A commit names the Blender build that ran its tests and never claims a pass
  without a run. The 5.2.3 API facts are re-checked in 5.3 by `tools/blender/probe_blender_api.py` (add-on Task
  27) in the first task that uses each one.
- **Witness record shape** (render output `format_table.h`, Task 43):
  `struct WitnessRecord { WitnessProperty property; WitnessResult blender_5_3 = WitnessResult::kNotRun;
  std::string_view evidence; };`. There is no `blender_5_2`. A table line is
  `{WitnessProperty::<P>, WitnessResult::<r>, "<test>; Blender <version> <build hash>; files <files_digest>"},`.
  `ChooseFormat` takes the first candidate whose every record is `kPassed`, else the last. The rule: a USD
  candidate that passes every witness in Blender 5.3 becomes the chosen format; one that fails is removed with
  its writer (and its link route). `capabilities` prints each record as `{"property", "blender_5_3",
  "evidence"}`; the add-on's `WitnessRecordFact` has `property`, `blender_5_3`, `evidence`. The add-on's
  `write_witness_records.py` takes one results file (`results_<5.3.x>.jsonl`) and `--witness-folder`, and
  refuses a results file from any version other than 5.3.x.
- **Library versions** (render output Tasks 4 to 9): zlib, Imath, oneTBB, Alembic, OpenVDB, OpenUSD are each
  pinned exactly to the newest release that `uv run conan list "<name>/*" -r conancenter` shows when the task
  runs, with `conancenter` pointing at `https://center2.conan.io` (the old `center.conan.io` is frozen). On
  2026-11-14 that list gave zlib 1.4, Imath 3.2.3, oneTBB 2023.1.0, Alembic 1.8.8, OpenVDB 12.1.1 (older than
  Blender 5.3's 13.0) and OpenUSD 26.08. Version tests compare with the recorded pin; no task or text relies on
  Blender's bundled versions. Stop rules kept: package missing; Alembic that cannot turn HDF5 off; static
  OpenUSD that cannot register its schema types; an OpenVDB dependency other than zlib, Imath, oneTBB, Boost or
  Blosc. Boost or Blosc are turned off by a recipe option where one exists, otherwise accepted and recorded with
  what they add.
- **Operator ids and module names of the add-on** (A22 brings back the design's words): `fluidity.link_outputs`
  (Show Result), `fluidity.relink_run` (Find Moved Cache...), `fluidity.copy_support_report` ("Copy Support
  Report"); preference `link_outputs_when_ready`; modules `blender/fluidity/output_links.py` (`LinkRoute`,
  `LINK_ROUTES`, `apply_link_plan`, `read_output_row`), `blender/fluidity/logic/output_plan.py` (`LinkAction`,
  `LinkStep` with `path_in_blend`, `LinkPlan`, `build_link_plan`, `OutputRow`, `LINK_ROUTE_KEYS`),
  `blender/fluidity/node_assets.py`; `AddonError.FORMAT_NOT_LINKABLE`, `LINK_FAILED`;
  `RunSupervisor.request_link_plan`, `SupervisorSnapshot.link_plans`; method-paper label
  `<proc-apply-link-plan>`; test files `tests/addon/in_blender/test_link_routes.py`, `test_link_transaction.py`,
  `test_node_assets.py`. Renames that stay: package `logic/` (was `core/`), `path_in_blend`,
  `(code, level, message)`, `RunState.NEEDS_NEW_RUN`, `ignored_end_bytes`, `value_type`,
  `to_platform_name(os_name, machine)`, `WitnessCaseResult`, "artist's journey". The `attach` names of the
  first version of the plan (`attach_plan.py`, `output_attach.py`, `ATTACH_ROUTES`, `fluidity.attach_outputs`,
  `fluidity.reattach_run`, `copy_problem_report`, `starter_assets.py`, "add the package") are withdrawn.
- **Wind** (A20; add-on D13, Tasks 13a, 27, 27a). The domain has no `wind_acceleration` property; it has
  `force_field_collection` (a Blender collection; empty means every force field object in the scene, Blender's
  own effector rule). The add-on derives the wind: for each Wind force field, acceleration =
  `WIND_ACCELERATION_PER_STRENGTH * strength * direction`, with `WIND_ACCELERATION_PER_STRENGTH = 1.0` m/s2 per
  unit of Blender's Strength (the gas branch's factor, `gas_authoring.py:63-67` on
  `origin/codex/m7-low-mach-gas`) and `direction` the field object's local +Z axis in world axes (from Blender's
  `mathutils`); several Wind fields add. The scene file carries the result as the I/O schema's top-level
  `gas_wind: [{time, acceleration}]` (time in s from the first frame, acceleration in m/s2, world, Z-up), one
  key per frame when the strength or the turn is keyed, one key at time 0 otherwise, and no keys in a domain
  without gas. Neither I/O nor the gas module converts anything: they read the keys as written. Settings that
  are not honoured give findings (module `"addon"`, found by `"scene_reader"`): `wind_falloff_not_honoured`,
  `wind_noise_not_honoured`, `wind_shape_not_honoured` (all `HAS_KNOWN_LIMIT`), `wind_flow_not_honoured` and
  `wind_without_gas` (`NOTE`); the other field types give `force_field_not_honoured` (`HAS_KNOWN_LIMIT`) or
  `force_field_ignored` (`NOTE`) according to `FORCE_FIELD_TYPE_TABLE` in `logic/force_fields.py`.

## Grid built on the GPU (A17)

Owner answer A17 (2026-11-14): a run is all CPU or all GPU. **This replaces the earlier choice "`BuildGrid` runs
on the CPU, also for GPU runs"** (grid design decision 8, owner question Q41, and the grid design's section 6
bullets "CPU", "Per step for the GPU run" and 5.1 step 9 / 5.8 "upload"): no block ids come down and no tables go
up. The grid plan (`plans/2026-11-14-rebuild-02-grid.md`, Tasks 6 to 11 and 20 to 27) carries the details.

- **Input list.** `grid::GridInputs` gains two fields; the list lives on the buffers' processor:

  ```cpp
  struct GridInputs {
    GridFrame frame;
    std::span<const CellId> active_cells;                // A list in CPU memory, any order.
    const Buffer<CellId>* active_cell_buffer = nullptr;  // Or a list in a buffer, any order.
    std::uint32_t active_cell_buffer_count = 0;          // Entries of *active_cell_buffer used.
    GridVersion version;                                 // 1 or more.
  };
  ```
  When `active_cell_buffer` is set it is the list and `active_cells` must be empty. A list on another processor
  than the `GridBuffers` (for the span forms: than the CPU) is a `Requires:` precondition checked in debug builds
  (A30; was `kBadInput`), except an empty list.
- **Building.** `BuildGrid(const GridInputs&, GridBuffers&)` keeps its signature. With GPU buffers it validates,
  sorts and builds every table on the GPU and reads back only a private status record (`GridBuildStatus`: first
  failure key, error, two cells, block count, split-cell count, result count, active cells, active cells per
  level, face pieces). The tables, counts and failures equal the CPU builder's byte for byte; the CPU builder is
  the oracle. **The step calls `BuildGrid` on its run's processor each time the cell set changes**; in a GPU run
  `ChooseActiveCells` hands it a `Buffer<CellId>` on the GPU and the entry count as a CPU number (from the
  step's own stage status), and takes the new grid's counts from `grid.counts()`.
- **Checks before building, on either processor, allocating nothing:**
  `std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs, GridBuffers& buffers);`
  `std::expected<void, GridFailure> CheckSizeStepRule(const GridInputs& inputs, GridBuffers& buffers);`
  The buffers hold no grid before or after. The buffers form of `ComputeGridCounts` fails `kTooManyCellBlocks`
  when the grid does not fit; the one-argument `ComputeGridCounts(inputs)` (CPU list only) still counts beyond
  any room. `CheckSizeStepRule` (both forms) runs id checks, overlaps and the size-step rule, not siblings.
- **First failure, both processors:** the smallest key `(step << 56) | (position << 8) | detail`: step 1 id
  checks (position = input index), 2 overlaps, 3 siblings, 4 size-step rule (position = place in fixed cell
  order), 5 and 6 room. `kTooManyCellBlocks` always has `cell` zero.
- **No CPU copy of a GPU grid.** `tables(RunOn::kCpu)` on a GPU grid returns version 0 and null arrays (the design's
  "kCpu always works" no longer holds). The one explicit copy:
  `std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);`
  (456 B per block; for tests, tools and a caller that accepts the transfer).
- **Memory** (`BuildGridMemoryEstimate`): a CPU grid unchanged (456 B per block on `kRunOn`, 516 B on `kCpu`). A GPU
  grid: everything `kRunOn`, nothing `kCpu`: 456 B tables + 1,536 B builder buffers (cell ids, sort keys, two
  orders) + 4 B piece map = 1,996 B per block, plus `64 x ComputeOrderBuffersSize(kGpu).bytes_per_item` per block,
  `fixed_bytes`, 8 B and `sizeof(GridBuildStatus)` per storage. `GridSettings::max_cell_blocks` at most
  `grid::kMaxCellBlocksLimit = (1u << 26) - 1`.
- **Values.** `ApplyGridVersion` runs on the CPU for every grid (it records version and block count, launches and
  copies nothing); values on another processor than the grid's are `kBadInput`.
- **Border search on either processor** (`cell_block_range.h`):

  ```cpp
  class BorderSearchBuffers { public: std::uint32_t max_cell_blocks() const; RunOn run_on() const; };  // Move-only.
  std::expected<BorderSearchBuffers, GridFailure> BuildBorderSearchBuffers(RunOn run_on,
                                                                           std::uint32_t max_cell_blocks);
  std::vector<MemoryEstimateEntry> BuildBorderSearchMemoryEstimate(RunOn run_on);
  // Blocks outside `range` in border_so_far or in the border of range + border_so_far at `width`: one more
  // stacked layer, merged with the layers before; ascending; returns the total, writes at most
  // border_blocks.size(). All buffers on grid.run_on(); border_blocks must not be border_so_far.
  std::expected<std::uint32_t, GridFailure> FindBorderBlocks(const StaggeredGrid& grid, CellBlockRange range,
                                                             const Buffer<CellBlockIndex>& border_so_far,
                                                             std::uint32_t border_so_far_count, BorderWidth width,
                                                             BorderSearchBuffers& search_buffers,
                                                             Buffer<CellBlockIndex>& border_blocks);
  ```
  On the GPU it marks each block through its own 27 same-level, bigger and 8 smaller neighbours (the grid's
  spatial index), prefix-sums and scatters; only the count comes back. A stacked border is one call per layer
  with two output buffers used in turn. The design's span forms of `FindBorderBlocks` fail `kBadInput` on a GPU
  grid; `BuildRangeWithBorder` requires a CPU grid (precondition); `BuildRanges` works on either.
- **Pieces on either processor** (`grid_piece.h`):

  ```cpp
  std::expected<GridPiece, GridFailure> BuildGridPiece(const StaggeredGrid& whole, CellBlockRange range,
                                                       const Buffer<CellBlockIndex>& border_blocks,
                                                       std::uint32_t border_block_count, GridBuffers& buffers);
  // GridPiece member: the piece map on grid().run_on().
  const Buffer<CellBlockIndex>& block_in_whole_grid_buffer() const;
  ```
  The span form of `BuildGridPiece` needs a CPU whole grid; `GridPiece::block_in_whole_grid()` (span) is empty
  for a GPU piece. **Cell resize and batches build pieces with the buffer form and find stacked borders with the
  buffer form of `FindBorderBlocks`** in a GPU run.
- **Needed from Foundation** (`base/stable_order.h`, before the grid's Task 6):

  ```cpp
  // Temporary arrays of BuildStableOrder and ComputeExclusivePrefixSum on one processor, for up to max_count
  // items, so that neither allocates per call. Move-only.
  class OrderBuffers { public: std::size_t max_count() const; RunOn run_on() const; };
  std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count);
  // Exact bytes BuildOrderBuffers allocates: bytes_per_item * max_count + fixed_bytes.
  struct OrderBuffersSize { std::uint32_t bytes_per_item; std::uint64_t fixed_bytes; };
  OrderBuffersSize ComputeOrderBuffersSize(RunOn run_on);
  // As BuildStableOrder, allocating nothing. kSizeMismatch when count > order_buffers.max_count().
  std::expected<void, BufferError> BuildStableOrder(const Buffer<std::uint64_t>& keys, std::size_t count,
                                                    const Buffer<std::uint32_t>* previous_order,
                                                    Buffer<std::uint32_t>& order, OrderBuffers& order_buffers);
  // sums[i] = values[0] + ... + values[i - 1] for i in [0, count]; sums[count] is the total. Exact unsigned
  // arithmetic; a total above 2^32 - 1 is kSizeMismatch. sums.size() >= count + 1. `sums` may be the same buffer
  // as `values` (in place). Allocates nothing. Same bytes on CPU and GPU.
  std::expected<void, BufferError> ComputeExclusivePrefixSum(const Buffer<std::uint32_t>& values, std::size_t count,
                                                             Buffer<std::uint32_t>& sums, OrderBuffers& order_buffers);
  ```
- **Left to the plans that consume these (not changed by the grid plan):** the batches plan's
  `FindStackedBorder`, `WriteStepArrayBlocks` / `BlockCopy` runs (computed on the CPU from the piece map) and
  `ResizeCells` with CPU mirrors in a batched GPU run all assume CPU tables or a CPU map of a GPU grid; under A17
  they either use the buffer forms above, read the map back (4 B per loaded block) or call `ReadGrid`. The solids
  design's `FindGridEdgesOpenToSky` "runs on the CPU from the block ids": on a GPU grid it has no CPU block ids.
  The cell resize plan cites "grid plan Task 7" for size levels: that task is now Task 8. (All four settled in
  "Batched solves (A15)", section "A17 follow-ups".)

## Foundation as built: `Buffer` (controller rulings, 2026-11-14)

- (Replaced by A30, 2026-11-14: `kRunOnMismatch` is removed; buffers on one processor are a `Requires:`
  precondition checked in debug builds, and modules map nothing.) `BufferError` gained a fifth value, after
  `kSizeMismatch`, for mixed processors in `CopyBuffer`, `BuildStableOrder`, `ComputeExclusivePrefixSum` and the
  ordered sums.
- `MemoryAvailable` and `ReadMemoryAvailable` live in `base/processor_description.h` from Foundation Task 13
  (private `processor_description_cpu.cc`, `processor_description_gpu.cu`); Task 16 adds `ProcessorDescription`
  to the same header.

## Coupling inside the pressure solve (A18)

Written 2026-11-14 from owner answer A18 and `literature-coupling.md`. It overturns solids design decision 3
("one step late") and the gas design's "liquid reads the gas pressure one step late" for sealed blobs, and wins
over `shared-interface.md` where its rows say "one step late" or put `ApplyPressureToMaterials` in stage 17.
Plans: pressure Tasks 43 to 50 (method in Task 44), solids Task 23, gas Tasks 16 and 18, step Tasks 28 to 30.

**Pressure (`pressure_equations.h`, shared types):**

```cpp
struct BlockingMaterialFace {          // One face piece where liquid meets a blocking material set.
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
  std::uint32_t material_set = 0;      // Index into the step's list of blocking sets.
  double blocked_area = 0;             // m2, > 0: this set's corner areas (ComputeBlockedCornerAreas) summed.
  double face_mass = 0;                // kg, > 0: the set's GridFromParticles::mass on the piece.
  double velocity_before_pressure = 0; // m/s along +axis, after ApplyForces and CollideSolids.
  double coverage_volume_rate = 0;     // m3/s along +axis: this set's share of the snapshot's solid_volume_rate.
};
struct SealedGasBlob {                 // One sealed tracked-gas blob that touches liquid, not held at atmosphere.
  grid::CellPlaceIndex first_cell_index{grid::kNoIndex};
  double gas_volume = 0;               // m3, > 0, step start.
  double background_pressure = 0;      // Pa absolute, > 0, step start, equalised.
  double heat_ratio = 1.4;
  double pressure_above_atmosphere = 0;  // Pa: the blob's carried value.
  double heat_volume_rate = 0;         // m3/s = ((heat_ratio - 1) / heat_ratio) * sum V q / p0.
  double boundary_volume_rate = 0;     // m3/s of solid moving into the blob through pieces that are not liquid
};                                     //   connections (inflow positive).
```

- `EquationInputs` gains `std::span<const BlockingMaterialFace> blocking_material_faces` (sorted by
  `face_piece_index`, then `material_set`), `std::span<const SealedGasBlob> sealed_gas_blobs` (by
  `first_cell_index`), `const grid::CellValues<std::uint32_t>* sealed_gas_blob_of_cell = nullptr` (`kNoBlob`
  elsewhere), `const grid::CellValues<double>* gas_pressure_above_background = nullptr` (Pa, gas solve pressure
  of the last commit). `PressureEquations` gains `blocking_material_face_count`, `sealed_gas_blob_count`.
- `SolvedArrays` gains `std::span<double> blocking_material_velocity_after_pressure` (one per face, input
  order) and `std::span<double> sealed_gas_pressure` (Pa above atmosphere at the step end, one per blob).
  `FlowImbalanceReport` gains `double max_sealed_gas_volume_error_rate` (1/s). `BufferSizes` gains
  `max_blocking_material_faces`, `max_sealed_gas_blobs`.
- Shared formulas in `cell_equation_math.h`: `ComputeMaterialFaceCoefficients(blocked_area, face_mass,
  step_duration)` -> `ConnectionCoefficients{dt A / m, dt A^2 / m}`; `ComputeGasSqueezeAllowance(gas_volume,
  background_pressure, heat_ratio, step_duration)` = `V / (heat_ratio p0 dt)` (the papers' `V / (gamma p dt^2)`
  with the row divided by `dt`); `ComputeMaterialVelocityAfterPressure(u, velocity_per_pressure, p_low, p_high)`
  = `u + velocity_per_pressure * (p_low - p_high)`.
- Method: a material face is a two-point term of weight `dt A^2 / m` (an end with no equation is at pressure 0,
  as in the motion map: gas and air-pocket pressure push no solid); its flow `A u - coverage_volume_rate`
  replaces the coverage part in `flow_imbalance` (inflow minus outflow, `s = +1` low end). A sealed gas blob is
  one unknown `p_j` with row `S_j p_j + sum g_c (p_j - p_c) = r_j`; the plan eliminates it exactly (rank-one
  term per blob, like the free-solid term), recovers `p_j` after the solve, and checks its row against
  `max_volume_error_rate * gas_volume`. Air-end precedence: sealed blob, then `holds_tracked_gas` (one step
  late), then enclosed pocket, then 0. The ladder adds material weights to pairs or own weights and treats blob
  connections as reference ends (rank-one term left out). ~~Equations with either term are solved only by
  `SolvePressure`; the pieces viscosity's joint solve calls return `kBadInput` for them.~~ **Replaced** (A24): the
  joint solve carries both terms; see "Viscous liquids with blocking materials and sealed gas (A24)".
- The dense Python reference has four cases (the fourth, `coupled.json`, solves the gas blob as a true extra
  unknown).

**Solids (`blocking_material_faces.h`):** `struct BlockingSet { std::span<const std::uint8_t>
material_is_counted; transfer::GridFromParticles* grid_from_particles; };`, `struct BlockingMaterialFacesInputs
{ grid; blocks (whole grid); geometry; coverage; std::span<const BlockingSet> blocking_sets; }`, `class
BlockingMaterialFaces` (move-only view into `SolidsBuffers`, `faces_without_mass()`),
`std::expected<BlockingMaterialFaces, SolidsFailure> BuildBlockingMaterialFaces(const
BlockingMaterialFacesInputs&, SolidsBuffers&)`, `std::span<const pressure::BlockingMaterialFace>
ToBlockingMaterialFaces(const BlockingMaterialFaces&)`, `struct PressureOnMaterials { std::span<const Vec3> push;
double push_magnitude; }`, `std::expected<PressureOnMaterials, SolidsFailure> ApplyPressureToMaterials(const
BlockingMaterialFaces&, std::span<const double> velocity_after_pressure, std::span<const BlockingSet>,
SolidsBuffers&)` (writes each solved face velocity into its set's `GridFromParticles::velocity`). Removed:
`MaterialPushFields` and the particle form of `ApplyPressureToMaterials`; `ApplyPressureToSolids` stores no push
owed to particles (`push_to_materials` stays a report). A piece whose set has no face mass is not an entry (its
blocked part stays one-way in `solid_volume_rate`); a piece touching no `kLiquid` cell is not an entry.
Solids' "Tests the step plan must add" gains `SpringyBlock, FifthDensity_FloatsAtDisplacedDepth`.

**Gas (`solve_gas.h`):** `enum class SealedGasCoupling : std::uint8_t { kInsideLiquidSolve, kReadOneStepLate }`;
`struct GasStepStartInputs { grid; blocks; liquid_geometry; grid_edge_is_open_to_sky; domain; sources;
step_start_time; step_duration; sealed_gas_coupling; }`; `struct SealedGasForLiquid { std::span<const
pressure::SealedGasBlob> blobs; const grid::CellValues<std::uint32_t>* sealed_blob_of_cell; const
grid::CellValues<double>* pressure_above_background; }`; `std::expected<SealedGasForLiquid, GasFailure>
ApplyGasStepStart(const GasStepStartInputs&, const GasSettings&, GasAmountsTry&, GasBuffers&,
pressure::PressureBuffers&)` runs design 5.3 steps 1 to 4 and lists the sealed blobs touching liquid;
`SolveGas` then runs step 1 (pieces) and 5 to 12 and fails `kBadInput` without it. `SolveGasInputs` gains
`sealed_gas_coupling`, `sealed_gas_blobs`, `sealed_gas_pressure`; a coupled blob's end background pressure is
`atmosphere_pressure + sealed_gas_pressure[j]`. `ComputeGasSeenByLiquid` is unchanged (used where no sealed
blob claims a cell). `ComputeSealedGasStepLimit` is kept and enters `GasResult::step_limit` only with
`kReadOneStepLate`. Gas's "Tests the step plan must add": `GasStepLimitTakenIntoNextDuration` amended,
`SealedPocketAboveWater_NoSealedStepLimit` added. **Replaced in part** (A24): `SealedGasCoupling`, its fields and
`ComputeSealedGasStepLimit` are deleted (section "Viscous liquids with blocking materials and sealed gas (A24)").

**Step:** stage 10 calls `solids::BuildBlockingMaterialFaces` and `gas::ApplyGasStepStart` before
`pressure::BuildPressureEquations` (a gas scene without stage 10 calls `ApplyGasStepStart` at the start of stage
13); stage 12 passes the two step-owned spans and calls `solids::ApplyPressureToMaterials` after
`ApplyPressureToSolids`; stage 17 no longer calls it; stage 13 passes `sealed_gas_pressure`. ~~Coupling is
`kInsideLiquidSolve` when stage 12 runs, `kReadOneStepLate` when stage 11 runs.~~ **Replaced** (A24): stage 11 does
the same as stage 12; no coupling choice remains.

**Open (owner question):** the material's own volumetric stiffness inside the solve (half of the volumetric
response in Fang et al. 2020) is not built: `materials::ApplyInternalForce` already applies the full stress, so
it needs materials to drop its volumetric part and material cells to get rows. ~~A viscous liquid keeps reading
sealed gas one step late because viscosity's joint solve does not carry the blob term.~~ **Replaced** (A24).

## Batched solves (A15)

Written 2026-11-14 from owner answer A15, with A17's principle (a run is all CPU or all GPU; a batched GPU run
keeps on the CPU only what does not fit and moves it in planned, counted copies). Plans: pressure Tasks 31, 32,
34, 47 (amended) and 54 to 61 (new; comment pass now Task 62); viscosity Tasks 2, 9, 19 (amended) and 26 to 28
(new; comment pass now Task 29); gas Task 2 (amended) and 29, 30 (new; comment pass now Task 31); step Tasks 8,
33, 36 (amended); cell resize and batches Tasks 10, 17, 24, 34, 37 to 43 (amended) and 44 (new; comment pass now
Task 45); solids Task 19 (A17 follow-up).

**Replaces.** "The pressure solve, the joint viscosity solve and the gas solve are not batched" (batches design
5.9, pressure design 6 "Block ranges" and 10, solids design 6); `batches::SolvePlace::kWholeGridOnCpu` (a CPU
solve inside a GPU run, against A17); the requirement "a batched run gives the bytes of a one-piece run" and every
test that asserts it (renamed list at the end); "per-blob sums over cells in ascending index by one thread"
(pressure, gas) and viscosity's `ComputeTreeDot` over a whole value buffer, both replaced in **both** modes by the
block rules below.

### Three ways to run, decided per run

- `enum class RunMode : std::uint8_t { kOnePiece, kBatched, kBatchedWithSolvesInPieces };` (step `run_plan.h`).
  `kBatched`: particle stages in batches, every solve once on the whole grid on the run's processor (as before).
  `kBatchedWithSolvesInPieces`: as `kBatched`, and the solvers' working buffers live in CPU memory and goes through the
  GPU one loaded piece at a time. `step::BuildRunPlan` still chooses only `kOnePiece` or `kBatched`; the caller
  (I/O `Run`, tests) copies `batches::BuildBatchPlan`'s choice into the plan.
- `step::RunPlan` gains `std::uint32_t max_solve_loaded_blocks = 0;` (from `BatchPlan`) and
  `std::uint64_t allowed_cpu_bytes = 0;` (step 3 of `BuildRunPlan` applied to `on_cpu`).
- `enum class SolvePlace : std::uint8_t { kWholeGridOnProcessor, kInPieces };` (batches `batch_plan.h`).
- A CPU run can be given `max_loaded_blocks > 0` too (tests in the container): the loaded piece is then on the CPU
  and every copy is CPU to CPU, counted the same way.

### What lives where in `kBatchedWithSolvesInPieces`

Rule: **state and geometry stay whole on the GPU; solver buffers go to the CPU.**

- *Whole grid, on the GPU, for the run:* both grids' tables; `GridFields` (both copies); gas amounts (both
  copies); every array written in pass 1 by `ComputeDistanceToSurface` and `FindLiquidSurface` (corner and centre
  distances, `CellGeometry`, `FacePieceGeometry`, curvature samples, one-material distances, `CellEighthGeometry`,
  `FacePieceCentre`) and the gas geometry snapshot; the step's solve inputs (each solved set's
  `velocity_before_pressure`, the six cell arrays of stage 10, `SolidCoverageValues`, the motion map, the
  blocking-material faces, the free-solid entries); `blob_of_cell` and `pocket_of_cell`; the pressure ladder's
  rungs 1 and deeper with their direct-solve factors; every per-block partial, per-group total and progress record.
- *Whole grid, in CPU memory (`CpuMemory::kLockedForCopies`), loaded batch by batch:* every other row of the
  liquid `PressureBuffers`, the gas `PressureBuffers`, `ViscosityBuffers` and `GasBuffers` that is counted per
  cell place, face place, corner place, sample or coefficient: `CellEquation` and equation of cell, connections and
  their lists, the material-face and gas-blob term lists, rung-0 weights, solve vectors, `volume_error`, air-pocket
  and label working buffers, viscosity's samples, coefficients, face and cell vectors and kept directions, gas working buffers.
- Each module's memory estimate entries say it: built with `max_loaded_blocks > 0`, `BuildPressureMemoryEstimate`,
  `BuildViscosityMemoryEstimate` and `BuildGasMemoryEstimate` return those rows with `where = MemoryWhere::kCpu` at the
  limits, the same rows again with `where = kRunOn` at `max_loaded_blocks` (the loaded piece), and
  `BuildSolveBatchesMemoryEstimate`. One allocation table, as before.
- With this rule, about 15 kB per full block stays on the GPU for a liquid scene (27 kB with viscosity, about 22
  kB with gas), against 55 kB (190 kB) when the solves are whole. On a 16 GB GPU with 64 GB of CPU memory the run
  is then limited by CPU memory (about 400 kB per block of particles plus the streamed working buffers), as A15 asks.

### The shared mechanism: `pressure/solve_batches.h` (CPU only)

It lives in pressure because pressure is the lowest module the three solvers share (viscosity and gas already
call pressure), its own tests need it, and the batches module comes after all three. Batches only sizes it
(`BuildBatchPlan`) and the step builds the solvers' buffers with it. Pressure includes no later module.

```cpp
// The cut of one grid into batches for a solve whose working buffers are in CPU memory, the one loaded piece, the
// staging buffer and the copy counts. Move-only. Held inside PressureBuffers, ViscosityBuffers and GasBuffers
// built with max_loaded_blocks > 0; one-piece buffers hold none.
class SolveBatches {
 public:
  RunOn run_on() const;                          // Processor of the loaded piece (the buffers' run_on).
  std::uint32_t max_loaded_blocks() const;       // Range plus stacked border of one batch.
  std::uint32_t batch_count() const;             // For the bound grid; 0 before ApplySolveBatches.
  grid::CellBlockRange range(std::uint32_t batch) const;   // Whole-grid blocks this batch writes.
  std::uint32_t loaded_batch() const;            // grid::kNoIndex when none is loaded.
  const grid::GridPiece& loaded_piece() const;   // Precondition: loaded_batch() != kNoIndex.
};
// One allocation: loaded GridBuffers, BorderSearchBuffers, two border buffers and the map copy at
// max_loaded_blocks, the per-batch border lists (CPU), a locked staging buffer of max_loaded_blocks * 240 * 64
// bytes. kBadInput: max_loaded_blocks 0 or above max_cell_blocks, no width; kRunOnCallFailed: as plan-decisions.
std::expected<SolveBatches, PressureFailure> BuildSolveBatches(RunOn run_on, std::uint32_t max_cell_blocks,
                                                               std::uint32_t max_loaded_blocks,
                                                               std::span<const grid::BorderWidth> stacked_widths);
std::vector<MemoryEstimateEntry> BuildSolveBatchesMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks,
                                                   std::uint32_t max_loaded_blocks);
// Cuts `grid` (on run_on) with the largest n such that every range of grid::BuildRanges(grid, n) plus its stacked
// border holds at most max_loaded_blocks blocks (bisection; the buffer form of grid::FindBorderBlocks, one call
// per layer; each border list is read back once, 4 B per border block, counted). No n: kRunOnCallFailed with
// magnitude = blocks needed by the smallest range, limit = max_loaded_blocks. Binds the batches to grid's version.
std::expected<void, PressureFailure> ApplySolveBatches(const grid::StaggeredGrid& grid, SolveBatches& batches);
// Builds the piece of `batch` into the loaded GridBuffers (buffer form of grid::BuildGridPiece). Reads the
// cancel flag first: set gives kCancelled and loads nothing.
std::expected<void, PressureFailure> LoadSolveBatch(const std::atomic<bool>* cancel_requested, std::uint32_t batch,
                                                    SolveBatches& batches);
// Whole array -> the loaded array of the batch last loaded: every loaded block, in piece order. A whole array in
// CPU memory is gathered into the staging buffer block by block and sent with one slice WriteBuffer; a whole
// array on the run's processor is gathered by one private kernel through the piece map (counted "within").
// values_per_block: 64 (cell places), 240 (face places), 125 (corner places), 1 (cell blocks).
template <typename Value>
std::expected<void, PressureFailure> WriteBatchValues(const Buffer<Value>& whole, std::uint32_t values_per_block,
                                                      SolveBatches& batches, Buffer<Value>& loaded);
// Loaded -> whole for the range blocks only: one contiguous slice (the range is one run in both grids).
template <typename Value>
std::expected<void, PressureFailure> ReadBatchValues(const Buffer<Value>& loaded, std::uint32_t values_per_block,
                                                     SolveBatches& batches, Buffer<Value>& whole);
// Arrays counted per item with a begin offset per cell block (samples, coefficients, term lists): the items of
// every loaded block in piece order, and the loaded begin offsets (block count + 1). Returns the item count.
template <typename Value>
std::expected<std::uint32_t, PressureFailure> WriteBatchGroups(const Buffer<Value>& whole,
                                                               const Buffer<std::uint32_t>& whole_begin_of_block,
                                                               SolveBatches& batches, Buffer<Value>& loaded,
                                                               Buffer<std::uint32_t>& loaded_begin_of_block);
template <typename Value>
std::expected<void, PressureFailure> ReadBatchGroups(const Buffer<Value>& loaded,
                                                     const Buffer<std::uint32_t>& loaded_begin_of_block,
                                                     SolveBatches& batches, Buffer<Value>& whole,
                                                     const Buffer<std::uint32_t>& whole_begin_of_block);
// Connected labels over the whole grid from per-batch labels (smallest whole cell place index reachable inside
// the loaded piece). `joins` lists, for every border cell of every batch, (the label this batch gave it, the label
// its own batch gave it). Joined on the CPU, smallest label wins; final_label[l] is the final label of label l
// (CPU, max cell places). Exact: the labels equal the one-piece labels.
struct LabelJoin { std::uint32_t first = 0; std::uint32_t second = 0; };
std::expected<void, PressureFailure> JoinLabelsAcrossBatches(std::span<const LabelJoin> joins,
                                                             Buffer<std::uint32_t>& final_label);
struct SolveCopyReport {                // Since the last ResetSolveCopyReport (the step resets it per try).
  std::uint32_t batch_loads = 0;
  std::uint32_t sweep_count = 0;        // One sweep = every batch loaded once, in order.
  std::uint64_t bytes_to_processor = 0, bytes_from_processor = 0, bytes_within_processor = 0;
  double copy_elapsed_seconds = 0;
};
const SolveCopyReport& ReadSolveCopyReport(const SolveBatches& batches);
void ResetSolveCopyReport(SolveBatches& batches);
```

- Buffer settings gain one field each, `std::uint32_t max_loaded_blocks = 0;` (0: one piece): pressure
  `BufferSizes`, viscosity `ViscosityBufferSettings`, gas `GasBufferSizes`. Accessors:
  `const SolveBatches* FindSolveBatches(const PressureBuffers&)` (also `ViscosityBuffers`, `GasBuffers`; null in
  one piece), `const SolveCopyReport* ReadSolveCopyReport(const PressureBuffers&)` (same three; null in one piece).
- Stacked widths: pressure and gas `{kTouchingCells}`; viscosity `{kTouchingCells, kTouchingCells}` (the face
  pass reads sample rates of samples whose cells touch the range; those read faces one more layer out).
- **No signature of a solver changes.** Every function that takes `PressureBuffers&`, `ViscosityBuffers&` or
  `GasBuffers&` runs batch by batch when the buffers are in the batched layout; its other arrays stay where they
  are (whole, on the GPU, per the rule above; or the caller's CPU arrays, as viscosity passes to pressure's
  pieces). So the step's stage calls 9 to 13 do not change.
- **Pass 1 into whole geometry, both batched modes:** `std::expected<void, PressureFailure>
  WriteSnapshotRange(const PressureBuffers& piece_buffers, const grid::GridPiece& piece, PressureBuffers&
  whole_buffers);` copies the range blocks of every pass-1 row (list above) and the snapshot's largest values
  (maxima, exact) from one-piece buffers bound to `piece.grid()` into whole buffers bound to the whole grid
  (GPU to GPU, or CPU to CPU). The loaded `StepTry` owns those piece buffers (one-piece layout at
  `max_loaded_blocks`); `batches::ReadBatch` calls it after pass 1. This closes a gap of the first batches plan
  (a piece grid with whole-grid buffers is `kWrongGridVersion`).

### Method: the same iteration, with sweeps (chosen: out-of-core form, option (i))

- **One arithmetic for both modes.** A one-piece run is the batched algorithm with one batch: the same values
  computed from the same terms in the same order; a batched solve only inserts sweep boundaries. So a batched solve
  normally gives the one-piece bytes, and its result does not depend on the cut (so a batched run is repeatable
  even when the free memory of the day changes the cut). The owner's rule is still the test: agreement within the
  solver's tolerance, because the direct-solve room of the batched layout is smaller (below).
- **Grid-wide sums** (dots, norms, report totals): one partial per cell block, then `math::TreeSum` over every
  block in block order (unchanged). Per batch, the partials of its range blocks are written into a whole per-block
  array on the run's processor; the tree sum runs once after the sweep.
- **Per-group totals** (each blob's or sealed gas blob's sums, each free solid's push sums, each air pocket's
  volume, gas blob sums): per cell block, the group's cells of that block in ascending cell place; then, per group,
  `math::TreeSum` over its block partials in ascending block index. This replaces "one thread per blob over its
  cells (or connections) in ascending index" in both modes.
- **Labels** (blobs, air pockets, gas blobs): label propagation inside each loaded piece, then
  `JoinLabelsAcrossBatches`, then one relabel sweep. Equal to the one-piece labels. Equation indexes: per-block
  counts, an exclusive prefix sum over blocks (`ComputeExclusivePrefixSum`), then local: equal.
- **Ladder layout (both modes).** Rung 0 is the equations (with `connection_scale` when given), streamed. Rung 1 is
  built per batch: each coarse cell's members lie in one cell block (siblings, or a level-0 2 x 2 x 2 group, both
  inside one block because blocks are 4 cells wide and aligned), coarse cells numbered by (block, blob, merged
  `CellId`) through per-block counts and a prefix sum (no grid-wide sort). Rungs 2 and deeper are built from rung 1
  on the run's processor, whole, by the existing rule (blob, then merged id). Rung 0 is never terminal: a blob of at
  most 32 equations passes one to one into rung 1, is not smoothed at rung 0, and gets its direct solve there (the
  same mathematics as a direct solve at rung 0). Direct-solve room: `BufferSizes::max_direct_solve_values` (doubles;
  one piece `32 * max_cells`, as before; batched layout `4 * max_cells`); terminal blobs take room in ascending
  first-cell order, and a blob beyond the room gets one weighted-Jacobi pass down and one up at its terminal rung
  instead, counted in the new `SolveResult::blobs_without_direct_solve` (std::uint32_t).
- **Iteration arithmetic (both modes)**, with `A_local` = own term, connections, material faces (the local part):
  `q_local = A_local d`; per free solid `P_s = push_s . d` and per sealed gas blob `G_j = sum_c g_c d_c` (per-group
  rule); `k_s = dt * inverse_inertia_s * P_s`, `h_j = G_j / D_j`; `dq = d . q_local + sum_s P_s . k_s - sum_j G_j
  h_j`; and wherever `q` is used, `q_c = q_local_c + sum over c's push entries of push . k_s - sum over c's blob
  terms of g_c h_j`. The ladder's sealed-blob steps use `E_b` (the blob's error total) and `m_b` (its
  volume-weighted mean of the ladder output), and `rz = error . z_raw - sum_b m_b E_b` with `z = z_raw - m_b`.
- **Sweeps per iteration:** A (update `p` and `error` with the full `q`; partials of `E_b` and the largest error per
  block), B (remove `E_b`; weighted Jacobi down at rung 0; residual; gather into rung 1), the whole rungs on the
  processor, C (spread from rung 1; Jacobi up at rung 0 gives `z_raw`; partials of `error . z_raw` and of each
  sealed blob's volume-weighted `z_raw`), D (`z`, `d`, `q_local`; partials of `d . q_local`, `P_s`, `G_j`). With
  no `kSealed` blob, A and B are one sweep: **3 sweeps per iteration, 4 with a sealed blob.** Before the
  iterations: equation build (about 5 sweeps: counts, rows, labels, relabel and classes, rung 1), and 64 sweeps of
  two cell arrays for the sealed blobs' volume correction when a sealed blob exists; after: correction and check
  (2 sweeps), and a restart costs 2.
- **Iteration count: the one-piece count** (same preconditioner except the direct-solve room). **Copies per
  iteration:** 3 or 4 times the streamed bytes of every loaded block: about 250 B per cell (equation, connections
  with their lists, vectors), about 16 kB per full block per sweep, so 50 to 65 kB per block per iteration with
  the border. A 100,000-block scene copies about 6 GB per iteration: about 0.25 s at 25 GB/s, so a 30-iteration
  solve adds about 8 s per try. Serial copies (no overlap with compute).
- **Option (ii), rejected:** overlapping pieces solved on the GPU with a coarse correction, inside the same outer
  conjugate gradient (textbook: additive Schwarz). Two sweeps per outer iteration, but two to four times the
  iterations of a whole-grid multigrid preconditioner for slab pieces with one or two blocks of overlap, growing
  with the batch count; a different preconditioner from the one-piece run; an inner solve per piece to tune. Per
  one-piece iteration it copies 4 to 8 sweeps' worth against 3 or 4, and loses "same answer, same iteration count".
- **Revisit when:** a measured breaking-wave solve needs more than 60 iterations per try (then a stronger local solve
  per loaded piece pays for its copies); copy time exceeds 80 % of solve time (then overlap copies with compute,
  which needs asynchronous copies in `base/`); or the whole rungs plus factors exceed a quarter of the GPU budget.
- **Free solids:** entries and inverse inertias stay whole on the GPU (sized by the solid surface); a piece reads
  the entries of its loaded cells through a per-equation begin list streamed with the equations; `P_s` by the
  per-group rule; the rank-one part enters `q` as above. Left out of the ladder, as before.
- **Sealed gas blobs (A18):** `D_j`, `r_j` and the blob list whole on the GPU; each cell's blob terms streamed with
  its row; `G_j` by the per-group rule; `p_j` recovered after the solve from the per-group sum of `g_c p_c`.
- **Blocking-material faces (A18):** a two-point term on one face piece, stored with the rows of its two cells like
  a connection; its far end lies in the same touching layer as a connection's, so no special handling. The solved
  material velocity of an entry is written by the batch whose range stores the entry's face place (the input is
  sorted by `face_piece_index`, so each batch writes one contiguous run).
- **Failures and cancel.** A sweep runs every batch and keeps the smallest failure key (step, cell, face piece:
  an integer minimum, exact), so the failure is the one-piece failure. The cancel flag is read before every batch
  load (`kCancelled`). In the batched layout every output (`SolvedArrays`, `AirPocketArrays`, viscosity's two
  outputs, gas try amounts) is written only in the last write-back sweep after acceptance, so on any failure the
  outputs keep their bytes.
- **Tolerance, batched against one piece** (same processor): counts, labels, classes, blob ids, pocket lists
  equal; one solve: velocity `<= 2 * max_volume_error_rate * cell width + 1e-9 * max |velocity|`, pressure that
  bound times `face_density * pressure_distance / dt` (pressure's CPU-against-GPU bound); viscosity its own
  CPU-against-GPU bounds; whole runs over several steps: the step design 6.4 tolerances. Batched against batched
  with another cut, and a repeat: identical bytes.

### Viscosity and gas

- **Viscosity** (`SolveViscosityAndPressure`): pressure's pieces (`coupled_solve_pieces.h`,
  `flow_imbalance.h`) in the batched layout accept whole arrays in CPU memory or on the processor and run one sweep
  per call; viscosity's own passes run in sweeps with its own `SolveBatches`: one inner velocity iteration is 2
  sweeps (sample pass and face pass with `d . Kd`; then the updates with their dots), the outer step about 8 sweeps
  plus the inner solve; the 2 x 32 kept directions and responses stay in CPU memory. Samples are built batch by
  batch and kept grouped by block. **Both modes:** samples ordered by the cell's block, then ascending `CellId`,
  kind, axis pair, quarter; dots and largest values by block partials then `math::TreeSum` (not `ComputeTreeDot`
  over the buffer). Expect a syrup step in this mode to take hundreds to thousands of sweeps: possible, not
  practical, as the batches design says.
- **Gas:** `ApplyGasStepStart` and `SolveGas` run each per-cell and per-face step of design 5.3 as a sweep (local
  steps fused), per-blob totals by the per-group rule, and the gas pressure solve is `pressure::SolvePressure` on
  the gas `PressureBuffers` in the batched layout. Gas amounts and the gas geometry stay whole on the GPU.

### Planner and report (`batches::BuildBatchPlan`)

- `BatchSettings` gains `double solve_piece_share = 0.25;` (share of the GPU room left after the whole rows that
  goes to the loaded solve piece; the rest to the loaded batch). **Owner to confirm** (a new default constant).
- Order: (1) `kWholeGridOnProcessor` when the whole rows with every solver row on the processor fit within
  `max_whole_grid_share * allowed_bytes` (the existing rule); (2) else `kInPieces`: whole rows without the rows that
  move to the CPU; `max_solve_loaded_blocks = min(solve_piece_share * room / bytes_per_solve_loaded_block,
  limits.max_cell_blocks)`, `bytes_per_solve_loaded_block` = the loaded rows of every solver buffer of the scene at
  one block (they coexist); the solve piece must hold three of the largest possible layers (`3 * layer`, layer as
  `BuildBatchPlan` step 4) else `kBatchDoesNotFit`; the loaded batch gets the rest by the existing formula; (3)
  `kStateDoesNotFitCpu` when the CPU rows exceed `RunPlan::allowed_cpu_bytes`.
- `BatchPlan` gains, for the report shown before the run (A22 "budget"): `std::uint32_t max_solve_loaded_blocks`,
  `std::uint32_t solve_batch_count` (start grid), `std::uint64_t bytes_on_processor`, `std::uint64_t bytes_on_cpu`
  (the memory budgets), `std::uint32_t pass_loads_per_try` (= `2 * batch_count`), `std::uint32_t
  solve_loads_per_iteration` (sweeps per iteration of the liquid solve times `solve_batch_count`), `std::uint64_t
  solve_bytes_per_iteration`. `BatchReport` gains the measured `solve_batch_loads`, `solve_bytes_to_processor`,
  `solve_bytes_from_processor`, `solve_copy_elapsed_seconds` (sums of the solvers' `SolveCopyReport` of the try).
- I/O's `check` and run report should print these fields; the I/O plan is not amended here (open).

### A17 follow-ups (cell resize and batches, solids)

- `FindStackedBorder` (private, batches): `struct StackedBorder { const Buffer<grid::CellBlockIndex>* blocks;
  std::uint32_t count; };` `std::expected<StackedBorder, BatchesFailure> FindStackedBorder(const
  grid::StaggeredGrid& grid, grid::CellBlockRange range, std::span<const grid::BorderWidth> widths_from_last_stage,
  grid::BorderSearchBuffers& search_buffers, Buffer<grid::CellBlockIndex>& layer_a, Buffer<grid::CellBlockIndex>&
  layer_b);` one buffer-form `grid::FindBorderBlocks` per layer, the two buffers used in turn; only counts come back.
  `BatchedRun` holds the `BorderSearchBuffers` and the two buffers on the run's processor.
- `WriteBatch` builds the piece with the buffer form of `grid::BuildGridPiece` and reads the piece map back once per
  load (`block_in_whole_grid_buffer`, 4 B per loaded block, row `"batches: piece map copy"`, `where = kCpu`,
  counted in `bytes_from_processor`) to compute the `BlockCopy` runs on the CPU; a CPU run uses the span.
- `ResizeCells` in every GPU run (one piece or batched): its CPU parts (closure search, resize plan, `ResizeMap`,
  region transfer lists) read a CPU copy of the committed grid made once per call with `grid::ReadGrid` into CPU
  `GridBuffers` held by `CellResize` (row `"cell resize: committed grid copy"`, `where = kCpu`, 456 B per block);
  the resized grid is built on the CPU for the map and, from the same cell list uploaded into a `Buffer<CellId>` (8 B
  per cell, counted), by the step on the GPU (`ApplyResizedGrid`; reconcile pass: `step::BuildResizedGrid`); A17
  makes their tables equal; `BuildResizeMap` checks version and block count of both. CPU mirrors of `GridFields` stay as before (the state is on the GPU in
  both batched modes).
- The cell resize plan's "grid plan Task 7" for size levels reads "grid plan Task 8".
- `solids::FindGridEdgesOpenToSky` on a GPU grid: one GPU pass marks, per box of the biggest-level block lattice,
  whether a cell block lies in it (1 B per box, read back); the CPU tests solids and floods as before; one GPU pass
  per face place writes the marks from the reached boxes (1 B per box, sent). No block id comes down.

### Tests replaced (byte identity between modes)

| Old | New |
|---|---|
| `ApplyStagesInBatches, EveryStagePrefix_SameBytesAsOnePiece` | `ApplyStagesInBatches, PassOneStagePrefix_SameBytesAsWholeGrid` (local pass-1 outputs only: the border check) |
| `ApplyStagesInBatches, CutInsideALayer_SameBytes` | `ApplyStagesInBatches, CutInsideALayer_SameBytesAsLayerCut` |
| `ApplyStagesInBatches, Small*_SameBytesAsOnePieceForEveryCut` (4) | `..._SameBytesForEveryCut` and `..._WithinStepToleranceOfOnePiece` (4 each, both batched modes) |
| `Batches, FloatingBox_SameBytesAsOnePiece` (solids) | `Batches, FloatingBox_WithinStepToleranceOfOnePiece` |
| `TryStep, BatchedFireBesideWater_SameBytes` (gas) | `TryStep, BatchedFireBesideWater_WithinStepTolerance` |
| `Batches, DeepTankWithCellResize_SameBytesAsOnePiece` | `Batches, DeepTankWithCellResize_WithinStepToleranceOfOnePiece` |
| `ViscosityScene, OnePieceAndBatched_SameBytes` | `ViscosityScene, SamplesPerRange_SameBytesAsWholeGrid` (local build) and viscosity Task 27's batched tests |

`FindLiquidSurface, TwoRangesEqualOneRange_SameBytes` (pressure) and `Batches, FreeBoxCuts_SamePushesAndReactions`
(cuts of one mode) stay: they compare a local computation by range, not two modes.

### Open points

1. `solve_piece_share = 0.25` and the batched direct-solve room `4 * max_cells` are new constants (owner).
2. The ladder's rung-1 numbering and the rung-0 terminal rule change the one-piece ladder's numbering (not its
   mathematics); viscosity's sample order and dot order change in both modes. Recorded here for the owner.
3. Geometry stays whole on the GPU so that solids (`BuildMotionMap`, `BuildBlockingMaterialFaces`,
   `CheckSolidExchange`), gas and cell resize need no change. If a scene ever becomes GPU-limited in this mode, the
   geometry is the next row to stream, which needs those three solids functions to run batch by batch.
4. A solve batch that cannot hold one range with its border is `kRunOnCallFailed` (no new error value); the planner
   reserves three layers so a grid within its limits cannot reach it.
5. `ChooseCellSize(void*, const grid::GridTables& committed, Vec3)` needs CPU tables; on a GPU grid (A17) the step
   has none. Not settled here (proposal: the step passes the `ReadGrid` copy that `CellResize` keeps).
6. The I/O plan must print the planned budgets and copies (`BatchPlan` fields) and the measured ones.

## Planned cost shown to the user, and buffers in batched runs (2026-11-14)

Written 2026-11-14. Closes "Batched solves (A15)" open points 5 (in batched runs; the one-piece GPU case stays
open, below) and 6, and owner question round 2 items 11 and 12. Principle (A17): a run is all CPU or all GPU; in a
batched GPU run only what does not fit (the particles, and in `kBatchedWithSolvesInPieces` the solver buffers)
stays in CPU memory, moved in planned copies that the planner counts and the reports measure. The owner: "warning
the user about the expected cost (memory and time) is nice"; `budget` is a planned cost shown to the user (A22).
Plans (amended tasks carry "Amended 2026-11-14 (A17, batched-run buffers)"): step Tasks 8, 9, 10, 12, 14, 27, 29,
39, new Tasks 33a and 33b, Appendices A and B; particles Task 11; gas Tasks 10, 20, 29, 31; solids Tasks 13, 28,
38; cell resize and batches Tasks 24, 29, 35, 36, 37, 38, 39, 43, 45; render output Tasks 32, 47; I/O Tasks 29, 34,
35, 36, 37, 39, 42, new Tasks 23a and 29a, seam 11; Blender add-on Tasks 12, 32. No task was renumbered.

### Buffers in batched runs

The full inventory (every stage, every buffer, each of the three modes) is the step plan's **Appendix A**; the
step's own planned copies are its **Appendix B**. Rules every plan follows:

- **Placement rule for whole try arrays** (both batched modes, either processor): a whole array lives on the run's
  processor when a part at `kWholeGridOnProcessor` reads or writes it; else in CPU memory when a
  `kWholeStateOnCpu` stage reads or writes it, the commit keeps it, or a later pass reads it (a carrier between
  passes); an array read and written only inside one pass has **no** whole instance, only the loaded one.
  `ReadStepArrays` reports each array's place; a test checks every array against the rule.
- **Loaded instances.** The loaded `StepTry` owns, on the run's processor, one instance of every module buffer a
  pass stage uses: `transfer::TransferBuffers`, `materials::MaterialBuffers`, `contact::ContactBuffers`,
  `solids::SolidsBuffers` (each at `max_loaded_particles` per material and `max_loaded_blocks`), beside the piece
  `pressure::PressureBuffers` of A15. The simulation's whole instances keep their per-particle rows at zero
  particles, except the stage-17 solids instance below. No module signature changes: the builders take the
  smaller counts.
- **Stage 17 in CPU memory.** `solids::ApplyContactReactions` and `ComputeCutWork` run on the CPU in a batched run
  (their per-particle inputs are there), with a separate `SolidsBuffers` built with `run_on = kCpu`, the whole
  particle limit and `max_cell_blocks` (rows prefixed `"solids: "`, all `where = kCpu`). The `CollideSolids` block
  partials each range wrote in pass 1 reach it through the new
  `std::expected<void, SolidsFailure> WriteBlockPartialsRange(const SolidsBuffers& piece_buffers, const
  grid::GridPiece& piece, SolidsBuffers& whole_buffers);` (range blocks only, either processor pair), called by
  `batches::ReadBatch` after pass 1 and counted in `BatchReport`. Every solids call copies its per-solid results
  into `SolidsTry` (CPU) before it returns, so `CommitTry` reads results of stages 12 and 17 from two instances.
  The step reaches the instance through `solids::SolidsBuffers* step::FindContactReactionBuffers(Simulation&)`
  (null in one piece); `step::StepArray` gains `ArrayPlace whole_place` with `enum class ArrayPlace : std::uint8_t {
  kLoadedOnly, kWholeOnProcessor, kWholeInCpuMemory };`.
- **CPU copy of the grid** (batched GPU runs): stage 2 builds the needed cell list on the CPU, uploads it into
  `Buffer<grid::CellId>` (row `"step: active cell list"`, 8 B per cell place, `where = kRunOn`, counted) for
  `grid::BuildGrid` on the GPU, and builds a CPU copy of the same grid with the CPU builder from the same list
  (rows `"step: grid copy on the CPU"` = ~~`grid::BuildGridMemoryEstimate(RunOn::kCpu, 2)`~~ **replaced** (small fix b,
  A26/A27 pass): `grid::BuildGridMemoryEstimate(RunOn::kCpu, 2, grid::BuilderBuffersSharing::kOneForAllGrids)` (the pair
  shares one CPU `GridBuilderBuffers`, reconcile item 11) with every row's `where` set
  to `kCpu`; no copy, A17 makes the tables equal; version and block count compared, a difference is
  `kStageFailed`). The copy is swapped with the grid at commit, released with it on reject, and ~~refreshed after a
  committed resize with `grid::ReadGrid` (456 B per block, counted)~~ **replaced** (reconcile pass): after a
  committed resize it is the CPU resized grid cell resize built into the step's free copy buffers, no copy. Accessor
  `const grid::StaggeredGrid* step::FindCpuGridCopy(const Simulation&)`: the copy in a batched GPU run, the grid
  itself in a CPU run, null in a one-piece GPU run. Grouping, `choose_cell_size`, marker upkeep and
  `CellResize` (instead of its own `ReadGrid`) read it.
- **Mirrors of committed values** (both batched modes; built and copied in a batched CPU run too, CPU to CPU,
  counted the same, so the container tests the counts): `"step: committed gas volume on the CPU"` (8 B per cell
  place), `"step: committed gas velocity on the CPU"` (8 B per face place), `"step: committed grid-only marks on the
  CPU"` (1 B per cell place, with cell resize), refreshed once per committed `state_version` (a retry copies
  nothing); `"step: keep active on the CPU"` (1 B per cell place, per try, after `gas::FindCellsToKeepActive` ran
  on the amounts' processor). All `where = kCpu`.
- **Gas markers.** `MarkerChanges` rows are `where = kCpu` in every GPU layout (one piece too). New CPU form
  `struct MarkerGridInputs { const grid::StaggeredGrid* grid; const grid::CellValues<double>* gas_volume; const
  grid::FaceValues<double>* committed_velocity; };` and `std::expected<MarkerChanges, GasFailure>
  FindMarkerChanges(const MarkerGridInputs& inputs, grid::CellBlockRange blocks, const particles::ParticleFields&
  gas_markers, const GasLaw& law, const MarkerSettings& settings, GasBuffers& buffers);` (every input in CPU
  memory, else `kBadInput`; same lists as the amounts form). `MarkerChanges` gains `std::uint32_t copy_count = 0;
  std::uint64_t bytes_copied = 0;` (the GPU form's read-back of its two lists; 0 on the CPU).
- **Stage 3's whole part.** `solids::FindGridEdgesOpenToSky` floods over the whole grid, so it moves to a new
  `run_whole_grid` entry of `kFindSolidSurfaces`, run at `kWholeGridOnProcessor` right after pass 1 and before
  stage 9's whole part (a one-piece run calls it right after stage 3's `run`, as every whole part).
- **Foam and spray in pass 2.** `FoamAndSprayInputs::block_in_whole_grid` is the batched run's CPU copy of the
  piece map (already read back per load), never the empty span of a GPU piece: `step::ApplyGridPiece(StepTry&
  loaded_try, const grid::GridPiece& piece, std::span<const grid::CellBlockIndex> block_in_whole_grid)` (replaces
  the two-argument form of "Cell resize and batches") keeps it for stage 18.
- **Choosing a new cell's size on a GPU grid (one piece).** Stage 2 reads the needed cells that are not in the
  committed grid back (8 B each, counted), calls `choose_cell_size` on the CPU, sends the levels (1 B each,
  counted). `cell_resize::ChooseCellSize` reads `CellResize`'s CPU copy of the committed grid (its `ReadGrid` copy,
  or its CPU resized grid after a committed resize; one `ReadGrid` at the first call when it holds none, counted in
  `ResizeCellsReport`). **Open** (owner): a shared CPU-and-GPU size rule instead of a callback would remove these
  copies.

### Copies counted and the one exception

- `struct CopyCounts { std::uint32_t copy_count = 0; std::uint64_t bytes_to_processor = 0; std::uint64_t
  bytes_from_processor = 0; };` (step `run_plan.h`, so `RunPlan` and `StepReport` share it). A copy is one
  `ReadBuffer` or `WriteBuffer` slice call between CPU memory and the run's processor (and, in a batched CPU run, the same calls CPU to CPU). A batch load and a
  solve batch load count one copy each (their arrays inside are bytes, not copies).
- `StepReport::copies` (this try's step-own copies, Appendix B); `CopyCounts step::ReadCopyCounts(const
  Simulation&)` (cumulative since `BuildSimulation`: tries, and the CPU grid copy refresh after a resize).
  `particles::AddParticlesResult` gains `std::uint32_t copy_count = 0; std::uint64_t bytes_copied = 0;` (written to
  the processor; 0 when the particles are in CPU memory). `cell_resize::ResizeCellsReport` gains `std::uint32_t
  copy_count = 0;` beside `bytes_copied`. The render-output recorder: `CopyCounts
  render_output::ReadRecorderCopyCounts(const StepRecorder&)` (cumulative), summed by I/O through its seam.
- **Not counted** (each at most the bound given, listed once in the budget lines as "small records"): a stage's
  status record (at most 64 B), solver progress records, `AirPocket` records and `BlobHistory` (at most
  `max_air_pockets` records), `GasBlob` records (at most `max_gas_blobs`), per-solid and per-set result rows
  (`where = kCpu`), `FindGridEdgesOpenToSky`'s box marks (1 B per box of the biggest-level block lattice each way),
  `StepReport` sums and `CommittedStepLimits`.
- **Planned step copies.** `step::RunPlan` gains `CopyCounts planned_step_copies;` (upper bound per try from the
  Appendix B formulas at the limits, for the plan's mode). `MemoryEstimate` gains `std::uint64_t on_cpu = 0;` (rows
  `where = kCpu` plus snapshots; `on_processor` counts only `kRunOn` rows), `ModuleBytes` gains `std::uint64_t
  on_cpu = 0;`, `MemoryReport` gains `std::uint64_t allocated_cpu_bytes = 0;`. `enum class PlanFit : std::uint8_t
  { kFits, kDoesNotFit };` and `RunPlan::fit`: `BuildRunPlan` returns a plan that does not fit (mode `kBatched`,
  every estimate filled) instead of failing; `BuildSimulation` (and I/O `BuildRun`) fail `kDoesNotFit` on it.
  `BuildStepMemoryEstimate(const ReadyScene&, const SceneCounts&, const RunPlan* batched_plan = nullptr)`: with a
  batched plan the rows follow Appendix A.

### The budget (I/O `io/run_budget.h`, CPU only)

```cpp
enum class ModeReason : std::uint8_t { kFitsProcessor, kParticlesDoNotFitProcessor, kSolverBuffersDoNotFitProcessor,
                                       kStateDoesNotFitCpu, kLayerDoesNotFitProcessor, kOnePieceDoesNotFitCpu };
enum class BudgetVerdict : std::uint8_t { kFits, kDoesNotFitCpu, kDoesNotFitProcessor };
struct MemoryBudget { std::uint64_t budget_bytes = 0, allowed_bytes = 0, total_bytes = 0, free_bytes = 0; };
struct ModuleBudget { step::Module module = step::Module::kStep; std::uint64_t processor_bytes = 0, cpu_bytes = 0; };
struct CopyBudget {                          // Per committed step unless named. Upper bounds where Appendix B says so.
  std::uint64_t step_bytes = 0;  std::uint32_t step_copies = 0;            // RunPlan::planned_step_copies.
  std::uint64_t pass_bytes = 0;  std::uint32_t pass_loads = 0;             // BatchPlan bytes_copied_per_try, pass_loads_per_try.
  std::uint64_t resize_bytes = 0; std::uint32_t resize_copies = 0;         // GPU grid, cell resize: (456 + 512) B x blocks
                                                                           //   in one piece, 512 B x blocks batched.
  std::uint64_t solve_bytes_per_iteration = 0; std::uint32_t solve_loads_per_iteration = 0;
  std::uint32_t solve_iteration_limit = 0;                                 // pressure_solve.max_iterations.
  std::uint64_t snapshot_bytes_per_frame = 0;                              // one_snapshot rows at the limits.
  std::uint64_t bytes_per_step = 0;  std::uint32_t copies_per_step = 0;    // step + pass + resize.
};
struct FitSuggestion { std::uint64_t particle_count = 0, max_particle_count = 0; double smallest_cell_width = 0,
                       min_smallest_cell_width = 0; std::uint64_t more_memory_bytes = 0; RunOn more_memory_on{}; };
struct RunBudget {
  std::string scene_name; RunOn run_on{}; std::string processor_model;
  step::RunMode run_mode{}; step::PlanFit fit{}; ModeReason mode_reason{}; BudgetVerdict verdict{};
  std::uint32_t batch_count = 0, solve_batch_count = 0;
  MemoryBudget processor_memory, cpu_memory;
  std::vector<ModuleBudget> modules;         // Every module with bytes, in Module order.
  CopyBudget copies;
  std::optional<FitSuggestion> to_fit;       // With a verdict other than kFits.
  std::vector<std::string> lines;            // BuildBudgetLines, verbatim.
};
std::expected<RunBudget, IoFailure> BuildRunBudget(const step::ReadyScene&, const step::StepSettings&,
    const step::RunPlan&, const std::optional<batches::BatchPlan>&, const std::optional<batches::BatchesFailure>&,
    MemoryAvailable on_processor, MemoryAvailable on_cpu, const ProcessorDescription&, std::string_view scene_name);
std::vector<std::string> BuildBudgetLines(const RunBudget&);
// Reads the memory available (or RunOverrides::assume_memory_bytes / assume_cpu_memory_bytes), plans the run (step,
// and batches for a batched plan) and builds the budget. Allocates no buffer. Used by check, run and BuildRun.
std::expected<RunBudget, IoFailure> BuildSceneBudget(const RunScene& scene, const RunOverrides& overrides,
                                                     std::span<const MemoryEstimateEntry> output_estimate);
```

- `RunOverrides` gains `std::optional<std::uint64_t> assume_cpu_memory_bytes` and the CLI `check` gains
  `--assume-cpu-memory <bytes>` (`Command::assume_cpu_memory_bytes`), beside `--assume-memory`; neither is in a
  fingerprint.

- Suggestions: `s = allowed / needed` of the failing memory; `max_particle_count = floor(s * particle_count)`;
  `min_smallest_cell_width = ceil_1e-4(w / cbrt(s))` for a memory verdict, `ceil_1e-4(w / sqrt(s))` for a layer
  verdict (no particle suggestion); `more_memory_bytes = ceil(needed / memory_share_of_total) - total` (total known)
  else `needed - allowed`. Printed as "estimates".
- **Lines** (exact templates; sizes in decimal units: below 1000 `"<n> B"`, else two decimals of the largest of kB,
  MB, GB, TB not above the value; integers grouped by commas; widths `%.4f m`; `<p>` is `cpu` or `gpu`):
  `plan for <scene_name> on <p> (<processor_model>)`; `mode: <one piece | batches | batches with solves in pieces |
  none> (<reason>)`; `<p> memory: budget <a> of <b> allowed (<t> total, <f> free)` per processor (a CPU run prints
  one `cpu memory:` line with both budgets summed; total unknown: `(total not known, <f> free)`; a failed plan:
  `one piece needs` / `batches need` in place of `budget`); `largest on <p>: <module> <bytes>, ...` (up to 5 by
  processor bytes); in batched modes `largest in cpu memory: ...`; `copies per step: at most <b> in <n> copies` (one
  piece GPU), `copies per step: <b> in <n> batch loads, at most <b> in <n> step copies` (batched; `, <b> in <n>
  resize copies` appended with cell resize on a GPU grid), `copies per step: none` (CPU one piece), `copies per
  step: not planned (the scene does not fit)`; `solve copies: <b> in <n> piece loads per iteration (<b> at the
  limit of <n> iterations)` (solves in pieces only); `copies per frame: snapshot up to <b>`; `small records: status
  records, air pockets, gas blobs and per-solid results are not counted`; `time: not estimated before the run; a
  projected finish is printed after 3 frames` (or `time: not estimated` when it does not fit); last line `fits:
  yes` or `fits: no. To fit (estimates): at most <n> particles (now <n>), or a smallest cell of at least <w> m (now
  <w> m), or <b> more <p> memory`. Reasons: `everything fits the <p>`; `the particles do not fit the <p>; particle
  stages run in <n> batches, solves on the whole grid`; `the particles and the solver buffers do not fit the <p>;
  particle stages run in <n> batches, solves in <m> pieces`; `in one piece it does not fit the <p>, and in batches
  the state does not fit cpu memory`; `one layer of cell blocks with its border does not fit the <p>`; `the scene
  does not fit cpu memory`.

### Measured cost and time (I/O `io/run_cost.h`, CPU only)

```cpp
struct StageSeconds { step::Stage stage{}; double total_seconds = 0; std::uint64_t try_count = 0; };
struct RunCost {
  std::uint64_t committed_steps = 0, tries = 0, frames = 0;
  std::uint64_t largest_processor_bytes = 0, largest_cpu_bytes = 0;      // MemoryReport, largest over the run.
  double copy_bytes_per_step_mean = 0; std::uint64_t copy_bytes_per_step_max = 0;
  double copies_per_step_mean = 0; std::uint32_t copies_per_step_max = 0;
  double snapshot_bytes_per_frame_mean = 0;
  double solve_iterations_per_step_mean = 0, solve_bytes_per_iteration_mean = 0;
  std::vector<StageSeconds> stage_seconds;   // Every stage that ran, Stage order, all tries.
  std::vector<std::string> warnings, lines;
};
// Per try (on_try_done): result.report.copies, result.report.timings, solver iteration counts, and the batched
// run's last BatchReport (null in one piece). The try's copies wait until the step commits.
void ApplyTryToRunCost(const step::StepResult& result, const batches::BatchReport* batch_report, RunCost& cost);
// Per committed step (on_step_done): the memory report and the resize's and recorder's copies since the last one.
void ApplyStepToRunCost(const step::MemoryReport& memory, const step::CopyCounts& resize_and_recorder_copies,
                        RunCost& cost);
void ApplyFrameToRunCost(std::uint64_t snapshot_bytes, RunCost&);
struct ActivitySeconds { double simulate = 0, snapshot = 0, build_outputs = 0, write = 0, wait = 0; };  // RunReport's.
void BuildCostLines(const RunBudget&, const ActivitySeconds&, RunCost&);   // Fills warnings and lines.
inline constexpr double kMemoryWarningShare = 1.10;   // Owner to confirm.
inline constexpr double kCopyWarningShare = 1.25;     // Owner to confirm.
struct FrameMark { double time = 0; double sent_at = 0; };   // s simulated; s UTC.
struct TimeProjection { double simulated_seconds_per_second = 0; double projected_remaining_seconds = -1;
                        std::uint32_t frames_measured = 0; };
TimeProjection ComputeTimeProjection(std::span<const FrameMark> frames_done_in_this_process, double end_time);
```

- Copies of a committed step: every try's `StepReport::copies`, `BatchReport` (pass and solve) and, after the
  commit, the resize's and the recorder's copies. Warnings: memory measured above `kMemoryWarningShare` times its
  budget (per processor); copies per step (mean) above `kCopyWarningShare` times the planned; solve bytes per
  iteration (mean) the same. Lines: `cost of this run, planned / measured:`; `<p> memory: <a> / <b>` (gpu, then
  cpu; a CPU run prints one `cpu memory:` line with both summed); `copies per
  step: at most <b> / mean <b>, max <b> over <n> steps`; `copies per frame: up to <b> / mean <b> over <n> frames`;
  `solve copies per iteration: <b> / mean <b>, <x.x> iterations per step` (solves in pieces only); `time by stage
  over <n> tries: <stage name> <x.x> s, ...`; `time by activity: simulate <x.x> s, snapshot <x.x> s, build outputs
  <x.x> s, write <x.x> s, wait <x.x> s`; `warning: <p> memory <b> is <k> % above the planned <b>`; `warning: copies
  per step <b> (mean) are <k> % above the planned <b>`; `warning: solve copies per iteration <b> (mean) are <k> %
  above the planned <b>`. Stage names are I/O's stable `ToName(step::Stage)`.
- **Time** (honest): nothing before the run. `simulated_seconds_per_second` over the last up to 6 frames listed in
  this process (at least 2); `projected_remaining_seconds = (end_time - last frame time) / rate` from 3 frames on,
  `-1` before. Sent in `heartbeat` and `frame_done`; `run` and `resume` print `time: about <duration> left (<rate>
  simulated s per s over frames <a> to <b>)` on standard error after the 3rd frame listed in this process and every
  10th after it. Durations: `<n> s` below 60 s, `<m> min <ss> s` below 1 h, `<h> h <mm> min`; rate 3 significant
  digits.

### Where the budget appears

- `check`: `scene_checked.budget` (table `RunBudget`), the lines on standard error, exit 3 when `verdict !=
  kFits` (with the existing blocking finding `does_not_fit` and `use_bigger_cells`). `run` and `resume`:
  `started.budget`, the lines on standard error and in `log.txt`; at the end `done` / `stopped` / `failed` carry
  `budget` and `cost` (table `RunCost`), the cost lines and warnings on standard error and in `log.txt`;
  `run_report.json` carries `budget` and `cost` (cost rewritten with each report). Schema
  `schemas/simulate/v2/run_budget.fbs` (tables `RunBudget`, `MemoryBudget`, `ModuleBudget`, `CopyBudget`,
  `FitSuggestion`, `RunCost`, `StageSeconds`; enums as stable lower-snake strings: `run_mode` `one_piece`,
  `batched`, `batched_with_solves_in_pieces`; `mode_reason` `fits_processor`, `particles_do_not_fit_processor`,
  `solver_buffers_do_not_fit_processor`, `state_does_not_fit_cpu`, `layer_does_not_fit_processor`,
  `one_piece_does_not_fit_cpu`; `verdict` `fits`, `does_not_fit_cpu`, `does_not_fit_processor`; `module` and
  `stage` by I/O's stable names), included by `run_report.fbs` and `events.fbs`. Field names carry their unit
  (`_bytes`, `_seconds`, `_m` for `smallest_cell_width_m` and `min_smallest_cell_width_m`).
- Blender add-on: `RunPlanFacts.budget: RunBudgetFacts | None` from `scene_checked` (Task 12), its `lines` drawn
  verbatim in `FLUIDITY_PT_performance` (Task 32). The remaining-time estimate of the status line stays the
  add-on's own (three listed frames, the same rule).

### Open points (owner)

1. `kMemoryWarningShare = 1.10` and `kCopyWarningShare = 1.25` are new constants.
2. One-piece GPU `choose_cell_size` (above): copies or a shared size rule.
3. Stage 3's range part calls `solids::BuildTry` and the contact checks once per try; in a batched run it runs per
   batch. The step builds the solids try only in the first call of a try (step Task 33a checks it); the batches
   plan's guard test on stage order covers it. If the owner prefers, those calls move to stage 2.
4. Planned step copies are upper bounds at the limits; the measured mean is expected well below them. The warning
   compares means, so it fires only when a bound is wrong.

## Owner answer A23 applied (2026-11-14)

One rename pass over every plan and design: `MemoryRow` is `MemoryEstimateEntry`; every
`Build<Module>MemoryRows` is `Build<Module>MemoryEstimate` (it returns the module's entries);
`*_memory_rows`, `output_rows` and `cell_resize_rows` are `*_memory_estimate`, `output_estimate` and
`cell_resize_estimate`; prose says "memory estimate entry". "row" alone means a matrix row.

## Foundation as built: the rest (reconcile pass, 2026-11-14)

Copied from the built headers in `FluidiTy-rebuild/native/fluidity/base/` (the source; `shared-interface.md` section
(a) copies them too). Where an earlier entry of this file differs, this one wins.

- **`buffer.h`: what a buffer may hold, and slices.**

  ```cpp
  inline constexpr std::size_t kBufferAlignmentBytes = 64;
  template <typename Value>
  concept BufferValue = std::is_trivially_copyable_v<Value> && alignof(Value) <= kBufferAlignmentBytes;
  template <BufferValue Value> class Buffer;   // size(), run_on(), data(); move-only; Buffer() is an empty CPU buffer.
  template <BufferValue Value>
  std::expected<Buffer<Value>, BufferError>
  BuildBuffer(RunOn run_on, std::size_t size, CpuMemory cpu_memory = CpuMemory::kOrdinary);
  constexpr std::expected<void, BufferError> CheckSliceIsInside(std::size_t first, std::size_t count, std::size_t size);
  template <BufferValue Value> std::expected<void, BufferError>
  WriteBuffer(std::type_identity_t<std::span<const Value>> values, Buffer<Value>& buffer, std::size_t first);
  template <BufferValue Value> std::expected<void, BufferError>
  WriteBuffer(std::type_identity_t<std::span<const Value>> values, Buffer<Value>& buffer);
  template <BufferValue Value> std::expected<void, BufferError>
  ReadBuffer(const Buffer<Value>& buffer, std::size_t first, std::type_identity_t<std::span<Value>> values);
  template <BufferValue Value> std::expected<void, BufferError>
  ReadBuffer(const Buffer<Value>& buffer, std::type_identity_t<std::span<Value>> values);
  template <BufferValue Value> std::expected<void, BufferError>
  CopyBuffer(const Buffer<Value>& from, std::size_t first, std::size_t count, Buffer<Value>& to, std::size_t to_first);
  template <BufferValue Value>
  std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, Buffer<Value>& to);
  ```
  Rules: a slice is `count` items from index `first`; every copy has a slice form. The value type is deduced from
  the buffer only (`std::type_identity_t`), so a `std::vector` or `std::array` passes as the span directly (a
  braced list does not). A span that overlaps the slice, overlapping slices of one buffer, and a buffer copied onto
  itself are `kSizeMismatch`; `CopyBuffer` across processors is a precondition (A30; was `kRunOnMismatch`). A failed check writes nothing;
  after `kRunOnCallFailed` the target slice is unspecified. Every module's array element type satisfies
  `BufferValue` (a `static_assert` in its memory estimate is enough).
- **`tree_sum.h` (shared, `namespace fluidity::math`):**

  ```cpp
  template <typename ValueAt>
    requires std::invocable<ValueAt&, std::size_t>
  FLUIDITY_CPU_AND_GPU double TreeSumOf(std::size_t count, ValueAt value_at) noexcept;
  FLUIDITY_CPU_AND_GPU inline double TreeSum(const double* values, std::size_t count) noexcept;
  FLUIDITY_CPU_AND_GPU constexpr double LargerMagnitude(double first, double second) noexcept;
  ```
  `TreeSumOf` calls `value_at` once per index in index order (each value rounded before it is added, so a product
  is never fused with the sum), 64 doubles on the stack, no recursion: the per-group and per-block tree sums of every
  plan use it without building an array. `TreeSum` takes a pointer and a count (not a span). `LargerMagnitude`
  returns NaN when either side is NaN (unlike `math::Max`), both zeros give `+0.0`: the running maximum of every
  "largest value" uses it.
- **Same bits unless the result is NaN.** Every "same bits on CPU and GPU" promise of `base/` (`Floor`, `Abs`,
  `Sqrt`, `Min`, `Max`, `IsFinite`, `Length`, `Normalized`, `TreeSum`, `TreeSumOf`) holds except when the result is
  NaN, whose bits IEEE 754 leaves open. A CPU-against-GPU byte comparison in any plan compares NaN results as "both
  NaN", not by bits.
- **`mat3.h`:** functions take a `Mat3` by value, as they take a `Vec3` (`operator*`, `+`, `-`, `Transpose`,
  `Determinant`, `IsFinite`, `Rotation`); `Mat3` has no default member values (trivial type; `Mat3{}` is zeros).
  GPU code may pass `kIdentity` by value, never bind a reference to it.
- **Private GPU headers (`AGENTS.md`).** A private header `<name>_gpu.h` declares only what `<name>_gpu.cu`
  defines; only files of the same folder include it; its names may say `Gpu` and use any verb; it includes no CUDA
  header. The rules check fails an `#include` of a `*_gpu.h` from another folder or from a header that is not
  itself `_gpu`. (This replaces the contact entry's `move_particles_gpu_calls.h` naming for new code only where a
  plan says so; existing `_gpu_calls.h` names in plans stay, they are not `_gpu.h` headers.)
- **Allowed verbs** (`AGENTS.md` "Allowed verbs"): `Compute`, `Build`, `Check`, `Find`, `Apply`, `Solve`, `Read`,
  `Write`, `Copy` (values between two arrays on the same processor), `Take` (gets preallocated buffers back out of an object
  no longer needed, `TakeBuffers`) and `Free` (frees memory) since A29 (`Release` is not allowed), `To<Type>`; plain
  math words in `namespace math`. So `CopyBuffer` and `TakeBuffers` need no
  allowed-names entry.
- **`stable_order.h`:** `OrderBuffers` also has `[[nodiscard]] std::size_t byte_count() const;` (bytes held:
  `ComputeOrderBuffersSize(run_on())` applied to `max_count()`, 0 after a move). `ComputeOrderBuffersSize` gives
  `{4, 8192}` on the CPU and `{6, 4096}` on the GPU.
- **`run_on.h` (shared):** `enum class RunOn : std::uint8_t { kCpu, kGpu };` and `enum class RunOnError :
  std::uint8_t { kNone, kGpuNotBuilt, kNoGpuFound, kGpuCallFailed, kGpuMathDiffers };` (as "Foundation `base/` (as
  built)"). `check_can_run_on.h` (CPU only): `std::expected<void, RunOnError> CheckCanRunOn(RunOn run_on);`,
  `std::string_view ToMessage(RunOnError error);`, `std::string_view ToName(RunOnError error);`.
- **`memory_count.h` (A23 names, as built):** `enum class CountedPer : std::uint8_t { kCellBlock, kCellPlace,
  kFacePlace, kCornerPlace, kParticle, kGivenCount };`, `enum class MemoryWhere : std::uint8_t { kRunOn, kCpu };`,
  `enum class KeptFor : std::uint8_t { kWholeRun, kOneStepTry, kOneCall, kOneSnapshot };`,

  ```cpp
  struct MemoryEstimateEntry {
    std::string_view name;  // Must point at a string literal.
    CountedPer counted_per = CountedPer::kGivenCount;
    std::uint64_t given_count = 0;  // Read only when counted_per is kGivenCount.
    std::uint32_t bytes_each = 0;
    std::uint32_t copies = 1;
    KeptFor kept_for = KeptFor::kWholeRun;
    MemoryWhere where = MemoryWhere::kRunOn;
  };
  struct MemoryCounts { std::uint64_t cell_blocks = 0, cell_places = 0, face_places = 0, corner_places = 0,
                        particles = 0; };
  std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts);
  std::uint64_t ComputeMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts,
                                   MemoryWhere where);
  ```
  (`MemoryCounts` is declared field by field in the header, one per line.)
- **`processor_description.h`:** `struct MemoryAvailable { std::uint64_t total_bytes = 0; std::uint64_t free_bytes
  = 0; };` (0 means unknown), `std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on);`,
  `struct ProcessorDescription { std::string model; std::string driver; std::string runtime; };`,
  `ProcessorDescription ReadProcessorDescription(RunOn run_on);`. CPU-only `processor_description_text.h`:
  `std::string FindCpuModelInText(std::string_view cpuinfo_text);`.
- **`per_particle_velocity.h` (shared):** `template <typename T> struct PerParticleVelocity { T for_next_step;
  T for_this_step; };` (A1).

## Reconcile pass (2026-11-14)

Applies `reconcile-todo.md` items 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 15 and the coordinator's additional list to the
plans (tasks marked `**Amended 2026-11-14 (reconcile):**`). Item 4 was an owner question (decided afterwards: option C, see the A24 section at the end).

1. **Gas Task 18:** the `ApplyGasStepStart` comment says stage 10 whenever stage 10 runs (viscous scenes too, with
   `kReadOneStepLate`); only a gas scene without stage 10 calls it at stage 13. Done.
2. **Step Task 14:** the A17 amendment had moved the list to the GPU but the test still compared "the needed cell
   list"; `ChooseActiveCells, OnGpu_EqualsCpu` and `OnGpuTwice_IdenticalBytes` now compare through a test-side
   `grid::ReadGrid` and the stage status record's entry count. Done.
3. **Copy report reset.** New declarations (the smallest change; `FindSolveBatches` stays const):
   `void pressure::ResetSolveCopyReport(PressureBuffers& buffers);`, `void viscosity::ResetSolveCopyReport(
   ViscosityBuffers& buffers);`, `void gas::ResetSolveCopyReport(GasBuffers& buffers);` (each zeroes the held
   `SolveBatches`' report; does nothing for one-piece buffers; allocates and copies nothing). Pressure Task 56,
   viscosity Task 26, gas Task 29 produce them with a test each; step Task 33 calls them per try. Done.
4. **Springy solid beside a viscous liquid.** ~~OPEN (owner decision pending).~~ **Decided** (owner, option C;
   section "Viscous liquids with blocking materials and sealed gas (A24)"): the open notes below were removed, the
   option C notes became task text, and every blocked test runs as written. Original text kept: Notes at the landing
   place of each option: (A) exclude before the run: step Task 7 (`CapabilityReason::kViscousLiquidWithSpringySolid`,
   `kUnavailable`, `kSetViscosityToZero`) and viscosity Task 6 (`ViscosityError::kSpringySolidNotSupported`,
   `ViscositySceneSummary::has_springy_solid_touching_liquid`); (B) one-step-late push for viscous liquids: step
   Task 28 (stage 10 skips the faces when stage 11 runs; a lagged solids form A18 removed); (C) material faces in the
   joint solve: step Task 25, viscosity Task 15, pressure Tasks 28 and 45 (`ComputeVolumeError,
   CoupledEquations_IsBadInput`). Pressure Global Constraints no longer claims springy solids are excluded. Tests
   marked **Blocked by open decision (springy solid beside viscous liquid)**, expectations unchanged (all on
   `BuildFiveMaterialScene` without snow, which holds syrup in set L beside the jelly set E): step Task 31
   `TryStepWithStages, RejectAfterEveryStage_ChangesNothing`, `TryStep, StageTraceOfEveryScene`,
   `BuildSimulationFromSnapshot, ContinuedFiveMaterialRun_SameBytes`, and `TryStep,
   CancelBeforeEachStage_ChangesNothingAndNextCommits`, `TryStep, MaterialSets_KeysAndSums` if run on that scene;
   step Task 33 `TryStepWithStages, RangeCut_SameBytes`; step Task 33a (option A only) `ReadStepArrays,
   BatchedPlacementFollowsTheRule`, `BuildStepMemoryEstimate, BatchedGpu_PerParticleModuleRowsOnlyLoaded`; batches
   Task 35 (re-runs `RangeCut_SameBytes`, `ContinuedFiveMaterialRun_SameBytes`); batches Task 37 (option A only)
   `kStagePlaces, PassTwoReadsNothingWrittenLater`, `kStagePlaces, GiveTheStepsArrayPlaces`; batches Task 40
   `ApplyStagesInBatches, CutInsideALayer_SameBytesAsLayerCut`, `CancelBetweenBatches_IsCancelRequested`,
   `GridGrewPastLoadedBlocks_IsBatchDoesNotFit`, `ReportCountsBatches`, `ReportCountsSolveCopies`; I/O Task 19
   `WriteSavedState, RegionArrays_RoundTrip`. Under (A) each needs a scene change agreed with the owner (the scene no
   longer plans); under (B) or (C) each runs as written once that option is built.
5. **Pressure Tasks 24, 44 (b), 48:** "one thread per blob" replaced by the per-group block rule (block partials,
   then `math::TreeSum` per group in ascending block index). Done.
6. **"Viscosity" sample order:** the old line is struck and points to the A15 block-first order. Done.
7. **Pressure Task 13:** the test helper is `EquationSceneArrays` / `BuildEquationSceneArrays` (viscosity plan
   updated); `EquationArrays` is only the opaque type behind `PressureEquations::arrays`. Done.
8. **Pressure Task 44 `SealedGasBlob`:** comments copied verbatim from "Coupling inside the pressure solve (A18)".
   Done.
11. **Grid buffers sized by the planner (plan decision).**
    - `grid::GridSettings::max_cell_blocks` defaults to **0** (the design's `1u << 22` is withdrawn):
      `BuildGridBuffers` and `BuildGridBuilderBuffers` with 0 are `kBadInput`; the test-only `BuildTestGrid(inputs, settings)`
      (grid test support; the allocating `BuildGrid(inputs, settings)` is removed by A31) with 0 sizes to the list's
      blocks. No compile-time room is an allocation size.
    - New in `grid/staggered_grid.h` (grid plan Task 6):

      ```cpp
      class GridBuilderBuffers { public: std::uint32_t max_cell_blocks() const; RunOn run_on() const; };  // Move-only.
      std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffers(RunOn run_on, std::uint32_t max_cell_blocks);
      std::expected<GridBuffers, GridFailure> BuildGridBuffers(const GridSettings& settings,
                                                               GridBuilderBuffers& builder_buffers);
      enum class BuilderBuffersSharing : std::uint8_t { kOnePerGrid, kOneForAllGrids };
      std::vector<MemoryEstimateEntry> BuildGridMemoryEstimate(RunOn run_on, std::uint32_t storage_count,
          BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid);
      ```
      `GridBuilderBuffers` holds the `grid: builder buffers: ...` arrays (and the `OrderBuffers`); the caller owns it
      and lends it to every `GridBuffers`, which must not outlive it; it is not moved while
      lent; one build or check uses it at a time. No operation on a built grid reads the builder buffers. With
      `kOneForAllGrids` the builder buffers entries have `copies = 1`. (A31, 2026-11-14: the form
      `BuildGridBuffers(settings)` that owned its own is removed; pressure's `SolveBatches`, cell resize, tests and
      tools own a `GridBuilderBuffers` too.)
    - Owner: the step's simulation state owns one `GridBuilderBuffers` (built first, freed last, rebuilt by `ApplyLimits`
      with the storages) lent to the committed and free `GridBuffers` and, in a batched run, the loaded try's
      buffers; the batched GPU run's CPU grid copy pair shares one CPU `GridBuilderBuffers`; cell resize's two CPU grids in
      a one-piece GPU run share one CPU `GridBuilderBuffers`.
    - Numbers (GPU): per buffers 460 B per block (456 tables + 4 piece map) plus the status record; builder buffers 1,920 B
      per block (`1536 + 64 x 6`) plus 4,096 + 6 + 8 B, once (the 6 B: the order buffers' extra item, reconcile item 28). Two grids: 2,840 B per block (was 4,760 with builder
      buffers each). 16 GiB GPU: `allowed_bytes = 0.7 x 17,179,869,184 = 12.03e9 B`; a quarter of it, 3.01e9 B, holds about
      **1.06 M blocks** (about `2^20`, 67 M cell places) for tables + builder buffers; the old default `2^22` would take
      11.9e9 B.
    - Room: `limits.max_cell_blocks` = the blocks `BuildRunPlan` counts from the scene x `start_limit_headroom`
      (1.25), capped by the domain's blocks at the smallest cell, at least 1. Above `grid::kMaxCellBlocksLimit`, or
      when the estimate does not fit, the plan has `fit = kDoesNotFit` and the budget prints the does-not-fit lines
      before the run. Outgrowing the room during a run is `kTooManyCellBlocks` (`kRaiseLimits`, `ApplyLimits`).
    - Tests: grid Task 6 `BuildGridMemoryEstimate, SharedBuilderBuffers_CountedOnce`, `BuildGridBuffers,
      TwoGridBuffersShareBuilderBuffers_AllocatedEqualsEstimate`, `..., LentBuilderBuffersTooSmall_IsBadInput`,
      `BuildGridBuilderBuffers, ZeroBlocks_IsBadInput`, `GridSettings, DefaultRoomIsZero_BuildGridBuffersIsBadInput`; grid
      Task 9 `BuildGrid, SharedBuilderBuffers_OtherLiveGridUnchanged`, `BuildTestGrid, DefaultSettings_SizedToTheList`; grid
      Task 11 `BuildGrid, OnGpuSharedBuilderBuffers_OtherLiveGridUnchanged`; step Task 8 `BuildRunPlan,
      BlockRoomFromSceneNotFromDefault`, `BuildStepMemoryEstimate, TwoGridBuffersShareBuilderBuffers`, `BuildRunPlan,
      RoomAboveGridLimit_DoesNotFit`; step Task 9 `BuildSimulation, GridBuffersShareBuilderBuffers`.
12. **How the step receives the uploaded resized cell list** (with coordinator item 1). New in step
    `resize_access.h` (step Task 32): `struct ResizedCells { std::span<const grid::CellId> cells; const
    Buffer<grid::CellId>* cell_buffer = nullptr; std::uint32_t cell_buffer_count = 0; const grid::StaggeredGrid*
    cpu_resized_grid = nullptr; };` and `std::expected<grid::StaggeredGrid, StepFailure> BuildResizedGrid(ResizeTry&
    resize_try, const ResizedCells& resized_cells);` (builds into `free_grid_storage()` on the run's processor, binds
    the try as `ApplyResizedGrid`, checks version and block count against `cpu_resized_grid` in a GPU run).
    `ApplyResizedGrid(ResizeTry&, const grid::StaggeredGrid&)` stays for a caller that built the grid itself. Cell
    resize Task 24 calls `BuildResizedGrid` on a GPU grid. Stage 2's batched upload was already settled ("Buffers in
    batched runs").
13. **`choose_cell_size`'s `committed` argument** (with coordinator item 2): the step passes
    `committed_grid.tables(RunOn::kCpu)`: real tables in a CPU run; version 0 and null arrays in a one-piece GPU run
    (then `cell_resize::ChooseCellSize` reads its own CPU copy, batches Task 29); `FindCpuGridCopy(...)->tables(kCpu)`
    in a batched GPU run (step Tasks 14, 33b). The one-piece copies stay an owner question (open point 2 above).
15. **"Foundation as built: the rest"** added above. Done.

**Coordinator's additional list:**

- (1) `ApplyResizedGrid` vs uploaded list: item 12 above; step Task 32 and batches Task 24 agree.
- (2) `choose_cell_size` in a one-piece GPU run: item 13 above.
- (3) I/O plan: Task 5 `ToName(step::RunMode)` gains `"batched_with_solves_in_pieces"` and declares
  `ToName(step::Module)` (lower snake case, 14 names); Task 29's `AdvanceRunTo, BatchedEqualsOnePiece` becomes
  `AdvanceRunTo, BatchedWithinStepToleranceOfOnePiece` plus `AdvanceRunTo, EachModeRepeatRun_SameFiles` (`run_mode`
  stays in the fingerprint); Task 41's `OnGpuBatchedEqualsOnePiece` becomes
  `OnGpuBatchedWithinStepToleranceOfOnePiece` plus `OnGpuEachModeRepeatRun_SameFiles`; Task 36's base text plans
  through `BuildSceneBudget`.
- (4) Old test names in "Solids", "Gas" and "Step" marked replaced by the A15 names above.
- (5) Batches Task 10: `grid::BuildGridMemoryEstimate(kCpu, 1)`.
- (6) After a committed resize in a batched GPU run the step's CPU grid copy is the CPU resized grid cell resize
  built into `ResizeTry::free_cpu_grid_copy_buffers()` and handed over with `std::expected<void, StepFailure>
  ApplyResizedCpuGridCopy(ResizeTry& resize_try, grid::StaggeredGrid&& cpu_resized_grid);` (step Task 32):
  no `ReadGrid`, Appendix B row B5 withdrawn. Cell resize gets `enum class GridCopies : std::uint8_t { kNone,
  kHeldByCellResize, kHeldByStep };`, `CellResizeSettings::grid_copies`, and
  `BuildCellResizeMemoryEstimate(std::uint32_t ingredient_count, RunOn grid_run_on = RunOn::kCpu, GridCopies
  grid_copies = GridCopies::kNone)`: in a batched GPU run (`kHeldByStep`) its two CPU grids are neither allocated
  nor counted (batches Tasks 10, 24, 43; I/O Task 29 sets it from the mode).
- (7) Blender plan Task 17: `RunStatus.run_mode` and the `started` event's `run_mode` take
  `"batched_with_solves_in_pieces"`.
- (8) `RejectReason::kNotSupportedNow`: no glossary table in this file renames it (the tables rename only
  `step::Support` / `kSupported`, the capability enum); A22 allows "support" in its ordinary sense. Kept.

## Viscous liquids with blocking materials and sealed gas (A24), and what counts as viscous (A25)

Written 2026-11-14 from owner answers A24 and A25, and the owner's choice of option C for reconcile item 4 (a
springy solid beside a viscous liquid: material faces carried by the joint solve). Plans (tasks marked
`**Amended 2026-11-14 (A24)**`, `(A25)` or `(reconcile)`): viscosity Tasks 2, 6, 7, 15, 16, 17, 20, 26, 27, 28 and
new Tasks 15a, 15b, 17a, 19a, 27a; pressure Tasks 2, 28, 29, 30, 43, 45, 47, 48, 59, 62; gas Tasks (paper, formulas,
step start, solve, scene tests, comment pass); step Tasks 3, 6, 7, 8, 25, 28, 30, 31, 33, 33a, 35; cell resize and
batches Tasks 35, 37, 40; I/O Tasks 5, 19 and the preset and budget tasks; Blender add-on preset, finding and panel
tasks. It replaces, wherever they say otherwise: the A18 lines "Equations with either term are solved only by
`SolvePressure`; the pieces viscosity's joint solve calls return `kBadInput` for them", "A viscous liquid keeps
reading sealed gas one step late", "Coupling is `kInsideLiquidSolve` when stage 12 runs, `kReadOneStepLate` when
stage 11 runs", the gas lines on `SealedGasCoupling` and `ComputeSealedGasStepLimit`, the "Viscosity" entry's
"Free solid and snow with viscous liquid stay excluded" (snow part), and reconcile item 4.

### A24: the method (viscosity plan Task 15a; viscosity design 5, "Coupled terms")

Unknowns of the joint solve: liquid face velocities `u` (solved faces), cell pressures `p` (equation cells) and,
NEW, one pressure `q_j` per sealed gas blob `j` (Pa above atmosphere). Blocking-material face velocities `v_k` are
eliminated exactly. Signs as A18: `s = +1` for the low end of a piece; `flow_imbalance` is inflow minus outflow.

- **Replaced by "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)" for the joint solve
  (material faces are velocity unknowns there; `SolvePressure` keeps this elimination).** **Material face `k`**
  (`BlockingMaterialFace`, ends `a` low, `b` high): `m_k v_k = m_k v_k_before + dt A_k (p_a -
  p_b)` has no viscous term (the material's velocity is not read by any deformation-rate sample), so `v_k =
  ComputeMaterialVelocityAfterPressure(v_k_before, dt A_k / m_k, p_a, p_b)` (unsolved end at 0) is substituted into
  the volume balance of each solved end exactly as `SolvePressure` does: the cell's volume error gains `-s_e (A_k
  v_k - coverage_volume_rate)` (`kIncluded`) or `-s_e g_k (p_a - p_b)`, `g_k = dt A_k^2 / m_k` (`kLeftOut`, a
  direction). The pressure diagonal of the outer loop gains `g_k` per solved end, **not** multiplied by
  `connection_scale` (the material has no viscous resistance: its own face mass is its whole inertia). The face
  term is not seen by the inner velocity solve.
- **Replaced (A26): the liquid sticks; see the A26/A27 section.** **Viscous stress at a liquid-material face** (plan decision, follows solids design 5.5a item 3, "tangential
  motion stays the set's own (A8)"; the shared-velocity papers stick instead). Tangential: no stick; a blocking
  material is not a `contact::Solid`, so an edge sample beside it takes rule (b) (liquid surface beyond), as at a
  free surface. Normal: the stretch samples read the material's blocked part through the snapshot's
  `solid_volume_rate` (the material at its velocity before pressure, as a driven solid), so `BuildViscositySamples`
  is unchanged. The viscous normal stress from the material's own velocity change in this step, `(dt A_k / m_k)
  (p_a - p_b)`, is left out and not given to the material: zero at rest, zero as viscosity goes to 0. Listed for
  the owner.
- **Sealed gas blob `j`** (`SealedGasBlob`): kept as one more unknown of the outer loop, **not** eliminated. In
  `SolvePressure` the rank-one elimination uses the connection weights `g_c`, which exist only because a liquid
  face's velocity there is `u_before + (dt A / m)(p_c - p_j)`; in the joint solve that face's velocity answers its
  end pressures through the viscous velocity solve, so no two-point weight exists and the blob is carried like an
  equation cell whose only faces are its connections. Its far-end pressure in every liquid push is `q_j + ref_c`
  (`ref_c` as A18 (b)); its row is the A18 row with the solved velocity: `E_j = inflow_j(u) + heat_volume_rate +
  boundary_volume_rate - S_j (q_j - c_j)` (`kIncluded`), `inflow_j(u) - S_j q_j` (`kLeftOut`), `inflow_j = sum_c
  -s_jc (open_area_c u_c + solid_volume_rate_c)` (`kLeftOut`: open areas only), `S_j =
  ComputeGasSqueezeAllowance(...)`. Its diagonal: `S_j + sum_c g_c * connection_scale_c`. Start value `c_j`.
- **Why the system stays symmetric and solvable.** The extended pressure operator is `S = dt D K^-1 D^T + C + L`,
  `K = M + dt E^T W E` (symmetric positive definite on solved faces), `D` the flow operator of cells and blobs (a
  blob's flow row is the transpose of its push column: both are the open areas of its connections), `C =
  diag(squeeze_allowance, S_j)` with `S_j > 0`, `L = sum_k g_k (e_a - e_b)(e_a - e_b)^T` (or `g_k e e^T`), so `S` is
  symmetric positive semidefinite with null space only the constants of `kSealed` liquid blobs (handled by
  `ApplyZeroMeanToSealedBlobs`, unchanged: a liquid blob joined to a gas blob is `kCanBeSqueezed`; a material face
  between two equation cells joins their blobs, one with an unsolved end is a reference end, as A18). The outer loop
  (measured responses kept and made orthogonal) needs only a response linear in the direction; the inner solve and
  its symmetry are untouched. No ladder is involved (viscosity's direction is `imbalance / diagonal`).
- **Vectors and sums.** The outer loop's pressure-like vectors (pressure, imbalance, diagonal, direction, response,
  the `2 x 32` kept pairs) gain one entry per sealed gas blob. A dot product is the cell part (block partials, then
  `math::TreeSum`) plus the blob part (`math::TreeSum` over blobs in index order), added once in that order. Each
  `inflow_j` and connection sum by the per-group rule (A15). In the batched layout the blob entries stay whole on
  the processor (like pressure's `D_j`, `r_j`), the material-face lists stream with the rows (as pressure).
- **Outputs.** Material velocities from the accepted cell pressure; blob pressures are the accepted `q_j`.
  Written only on success (step 12), in the batched layout in the write-back sweep.

### A24: pressure pieces (`coupled_solve_pieces.h`, `flow_imbalance.h`; pressure Tasks 28, 29, 45, 47)

No piece returns `kBadInput` because the equations hold a material face or a sealed gas blob. Changes (NEW, A24):

```cpp
// coupled_solve_pieces.h. One value per sealed gas blob of the equations, entries [0, sealed_gas_blob_count), on
// the equations' processor: Pa above atmosphere (pressure), m3/s (volume error), m3/(Pa s) (own weight).
std::expected<void, PressureFailure> ComputePressurePush(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, GivenTerms given_terms, grid::FaceValues<double>& push);
std::expected<void, PressureFailure> ComputeVolumeError(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity, GivenTerms given_terms,
    grid::CellValues<double>& volume_error, Buffer<double>& sealed_gas_volume_error);
std::expected<void, PressureFailure> ComputeOwnWeight(
    const PressureEquations& equations, const grid::FaceValues<double>* connection_scale,
    grid::CellValues<double>& own_weight, Buffer<double>& sealed_gas_own_weight);
std::expected<void, PressureFailure> ComputeStartPressure(const PressureEquations& equations,
    grid::CellValues<double>& pressure, Buffer<double>& sealed_gas_pressure);   // c_j per blob
// Each BlockingMaterialFace's velocity after `pressure` (ComputeMaterialVelocityAfterPressure, unsolved end 0),
// input order; written as SolvePressure writes SolvedArrays::blocking_material_velocity_after_pressure.
std::expected<void, PressureFailure> ComputeBlockingMaterialVelocities(const PressureEquations& equations,
    const grid::CellValues<double>& pressure, std::span<double> velocity_after_pressure);
// flow_imbalance.h. The stop rule with the blob pressures given (the joint solve), instead of recovered by the
// elimination (SolvePressure's form above).
std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(
    const PressureEquations& equations, const grid::CellValues<double>& pressure,
    const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity,
    const SolveSettings& settings, PressureBuffers& buffers);
```

- The forms without a blob buffer accept material faces; given equations with `sealed_gas_blob_count > 0` they
  return `kBadInput` (the far-end pressure is an input they lack), never because of material faces.
  `ComputeVolumeError` adds each material face's flow (rule above); `ComputeOwnWeight` adds `g_k` unscaled per
  solved end; `ComputeFaceInertia`, `ApplyPressure`, `ComputeFlowImbalance` (flow of the velocity only),
  `ApplyZeroMeanToSealedBlobs`, `ApplyPressureOffset`, `ComputeWasSolved` accept both terms with unchanged
  arithmetic. `BuildPressureLadder` with a non-null scale adds `g_k` unscaled and treats blob connections as
  reference ends with own weight `g_c * scale` (no caller today; kept consistent). `SolvePressure` uses
  `ComputeBlockingMaterialVelocities` for its material output.
- A blob buffer shorter than the count, on another processor, or a material span of the wrong size: `kBadInput`.
- Test `ComputeVolumeError, CoupledEquations_IsBadInput` is **replaced** by `CoupledSolvePieces,
  MaterialFacesAccepted_MatchHandValues` and `CoupledSolvePieces, BlobFormsWithoutBlobBuffer_IsBadInput`.

### A24: viscosity interface (viscosity design 3; plan Tasks 7, 15, 15a)

```cpp
struct ViscosityBufferSettings {   // gains (NEW, A24):
  std::uint32_t max_sealed_gas_blobs = 0;   // >= the equations' sealed_gas_blob_count, else kBadInput.
};
struct SolvedCouplingArrays {      // NEW (A24). Sizes equal the equations' counts (kBadInput otherwise).
  std::span<double> blocking_material_velocity_after_pressure;   // m/s along +axis, input order.
  std::span<double> sealed_gas_pressure;                         // Pa above atmosphere at the step end.
};
std::expected<ViscosityResult, ViscosityFailure> SolveViscosityAndPressure(
    const ViscosityInputs& inputs, const ViscositySettings& settings,
    const std::atomic<bool>* cancel_requested, ViscosityBuffers& buffers,
    grid::FaceValues<double>& velocity_after_pressure, grid::CellValues<double>& cell_pressure,
    SolvedCouplingArrays coupling_arrays = {});
```
Memory estimate entry `"viscosity: sealed gas blob vectors"` (given `max_sealed_gas_blobs`, `70 x 8` bytes, always
`kRunOn`). `ViscosityInputs` comments lose "must be 0". `ViscosityError::kBadInput` loses the A18 clause.

### A24: exclusions (viscosity Task 6; step Tasks 3, 6, 7)

- **Lifted:** `ViscosityError::kSnowNotSupported`, `ViscositySceneSummary::has_snow`, `CapabilityReason::
  kViscousLiquidWithSnow` and its stable names (`snow_not_supported`, `viscous_liquid_with_snow`): snow was
  excluded because it "never enters these equations"; since A18 + A24 its inertia enters the joint solve through
  its material faces. `ViscosityError` loses one value (16 -> 15 before the plan's three lifting tasks; 12 if all
  three are lifted); `kCapabilityReasonCount` lower by one: **17** at most, 14 if all three are lifted. Test
  `ViscosityKnownOpen, SyrupOnSnowBed_StaysAtRest` becomes the passing scene test `ViscosityScene,
  SyrupOnSnowBed_StaysAtRest` (viscosity Task 19a); step test `CheckSceneCanRun, ViscousWithSnow_IsExcluded` becomes
  `CheckSceneCanRun, ViscousWithSnowAndSpringySolid_CanRun`; `TryStep, FiveMaterialsWithSnow_IsViscousLiquidWithSnow`
  becomes `TryStep, FiveMaterialsWithSnow_Commits`.
- **Replaced (A27): lifted; see the A26/A27 section.** **Kept:** `kFreeSolidNotSupported` / `kViscousLiquidWithFreeSolid`: its reason is the free solid's velocity as an
  unknown of the viscous velocity solve (viscous drag on the solid through `rate_from_solids`, its impulse); A24
  covers material faces and sealed gas only. The three plan-lifted exclusions keep their own rule.
- **Never added:** `kSpringySolidNotSupported`, `has_springy_solid_touching_liquid`, `kViscousLiquidWithSpringySolid`
  (options A and B of reconcile item 4 are withdrawn). Every test marked "Blocked by open decision (springy solid
  beside viscous liquid)" runs as written (option C).

### A24: step, gas, solids

- Stage 11 mirrors stage 12: `viscosity::SolveViscosityAndPressure(..., fields.velocity_after_pressure,
  fields.pressure, SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure})`, then
  `ComputeWasSolved`, `ApplyPressureToSolids`, then (scene with a blocking set) `solids::ApplyPressureToMaterials`.
  Stage 10 builds the material faces and the sealed blob list whatever stage follows. Stage 13 passes
  `sealed_gas_pressure` after either stage. Viscosity buffers: `max_sealed_gas_blobs` = gas's `max_gas_blobs` (0
  without gas).
- **Gas:** `SealedGasCoupling`, `GasStepStartInputs::sealed_gas_coupling`, `SolveGasInputs::sealed_gas_coupling`,
  `ComputeSealedGasStepLimit` and `GasResult::step_limit`'s sealed term are **deleted**: no path reads a sealed blob
  touching liquid one step late. Still read one step late, through `gas_pressure_above_atmosphere` and
  `holds_tracked_gas` (air-end precedence unchanged): open gas (also above a viscous free surface), slivers, cells
  that touch gas, held blobs. Lagged references in gas tests are test-side (as pressure Task 50): the liquid is
  given the blob as tracked gas, and the test passes `sealed_gas_pressure` from its own adiabatic update. Removed
  tests: `ComputeSealedGasStepLimit, MatchesFormula`, `GasBesideLiquid, SealedStepLimit`; `TryStep,
  GasStepLimitTakenIntoNextDuration` keeps only the small fire; added `TryStep, SealedPocketAboveSyrup_NoSealedStepLimit`;
  `TryStep, SealedGasBesideViscousLiquid_ReadOneStepLate` becomes `TryStep, SealedGasBesideViscousLiquid_SolvedInStageEleven`.
- **Solids:** the "Viscosity" text of solids design 4 and Open item lose the refusal; `BuildBlockingMaterialFaces`
  and `ApplyPressureToMaterials` serve stages 11 and 12 alike.

### A25: what counts as viscous, and the viscous effect per step

- **Definition.** A liquid ingredient is viscous exactly when its `viscosity` setting is above 0 (`> 0.0`; no
  threshold). The scene is viscous when any liquid ingredient is; then stage 11 runs, else stage 12. The step
  fills `ViscositySceneSummary::has_viscous_ingredient` and chooses the stage table **once**, in `CheckSceneCanRun`
  and `BuildSimulation`, from the ready scene; retries, shortened steps, limit raises, cell resizes, batches and
  the number below never change it. The water preset has viscosity 0 (`MaterialPreset::kWater`: 1000 kg/m3,
  **0 Pa s**, 0.072 N/m; was 1e-3).
- **Number** (viscosity `deformation_rate.h`, shared): `inline constexpr double kSmallestVisibleViscousEffectPerStep
  = 2.5e-4;` and `FLUIDITY_CPU_AND_GPU double ComputeViscousEffectPerStep(double viscosity, double step_duration,
  double density, double cell_width);` = `viscosity * step_duration / (density * cell_width * cell_width)`: Pa s /
  (kg/m3) = m2/s, times s, over m2: no unit (papers: the diffusion number). Reason for the threshold: on the
  grid, one implicit step multiplies the velocity difference between two neighbouring cells (the mode `k h = pi`)
  by `1 / (1 + 4 N)`, so it changes by `4 N / (1 + 4 N)`; below `N = 2.5e-4` that is under 0.1 % per step, which
  no frame shows. Owner to confirm (a new constant). Example: 1e-3 Pa s, 1000 kg/m3, 0.01 s, 0.01 m gives 1e-4.
- **Where computed: the planner.** `step::BuildRunPlan` fills `RunPlan::liquid_viscous_effects` with one
  `step::LiquidViscousEffect` per liquid ingredient (scene ingredient order) from the ingredient's viscosity and
  its material's rest density, `StepSettings::max_step_duration` and the grid frame's smallest cell width:

  ```cpp
  struct LiquidViscousEffect {                 // step run_plan.h, NEW (A25)
    std::uint32_t ingredient_index = 0;        // scene ingredient
    double viscosity = 0;                      // Pa s, the setting
    double density = 0;                        // kg/m3, rest density of its material
    double viscous_effect_per_step = 0;        // no unit; 0 when viscosity is 0
  };
  // RunPlan gains: std::vector<LiquidViscousEffect> liquid_viscous_effects; double viscous_effect_cell_width = 0;
  // (m) double viscous_effect_step_duration = 0; (s) and ViscousPath viscous_path;
  enum class ViscousPath : std::uint8_t { kPressureSolve, kViscousSolve };   // stage 12 or 11, fixed for the run
  ```
- **Lines** (I/O `BuildBudgetLines`, before the last `fits:` line, only in a scene with liquid; `%g` for viscosity
  and density, `%.3g` for the number, `%.4f` for width and step; ingredient export names):
  - `liquids: pressure solve (every liquid has viscosity 0)` then per liquid `  <name>: viscosity 0 Pa s, density
    <d> kg/m3, inviscid`;
  - or `liquids: viscous solve (a liquid has viscosity above 0); effect per step at the smallest cell <w> m and
    the largest step <t> s` then per liquid `  <name>: viscosity <v> Pa s, density <d> kg/m3, viscous effect per
    step <n>` (a liquid with viscosity 0 in a viscous scene: `..., inviscid`), followed, when `0 < n <
    kSmallestVisibleViscousEffectPerStep`, by `  note: <name>'s viscosity changes velocity differences between
    neighbouring cells by less than 0.1 % per step, which is not visible; set its viscosity to 0`.
  - Exact example (I/O test `BuildBudgetLines, TwoLiquids_ViscousEffectLines`): water 1e-3 Pa s 1000 kg/m3 and
    syrup 10 Pa s 1400 kg/m3, smallest cell 0.01 m, largest step 0.01 s:
    ```text
    liquids: viscous solve (a liquid has viscosity above 0); effect per step at the smallest cell 0.0100 m and the largest step 0.0100 s
      water: viscosity 0.001 Pa s, density 1000 kg/m3, viscous effect per step 0.0001
      note: water's viscosity changes velocity differences between neighbouring cells by less than 0.1 % per step, which is not visible; set its viscosity to 0
      syrup: viscosity 10 Pa s, density 1400 kg/m3, viscous effect per step 0.714
    ```
- **JSON** (`run_budget.fbs`): `RunBudget` gains `viscous_path: string` (`"pressure_solve"`, `"viscous_solve"`),
  `viscous_effect_cell_width_m: double`, `viscous_effect_step_seconds: double`, `liquids: [ViscousEffectOfLiquid]`
  with table (C++ `io::ViscousEffectOfLiquid`; renamed in the A26/A27 pass from `LiquidViscousEffect` so it does
  not share a name with `step::LiquidViscousEffect`) `ViscousEffectOfLiquid { ingredient_name: string; viscosity_pa_s: double; density_kg_per_m3: double;
  viscous_effect_per_step: double; is_viscous: bool; effect_not_visible: bool; }`.
- **Blender add-on:** the water preset's viscosity is 0; the viscosity property's tooltip: "Pa s. 0:
  the liquid has no viscosity and the scene uses the faster pressure solve; above 0: the viscous solve, for the whole run";
  the `liquids:` lines appear verbatim in `FLUIDITY_PT_performance` (they are part of `RunBudget.lines`);
  `RunBudgetFacts` gains `liquids` and `viscous_path`.
- **Tests:** viscosity `ComputeViscousEffectPerStep, MatchesDefinition` (1e-3, 0.01, 1000, 0.01 -> 1e-4; 10, 0.01,
  1400, 0.01 -> 0.714285714...; 1e-15 relative) and `kSmallestVisibleViscousEffectPerStep, IsTenthOfAPercentPerStep`
  (`4 N / (1 + 4 N) < 1e-3` at `N` and `>= 1e-3` at `1.001 N`); step `BuildRunPlan, ViscousEffectPerLiquid`,
  `CheckSceneCanRun, ViscosityZero_IsInviscid` (water preset: stage 12, no stage 11), `TryStep,
  ViscousPathFixedWhenStepShortens` (a liquid of 1e-6 Pa s, below the threshold, with a forced reject and two
  half steps: stage 11 in every try; and the water preset: stage 12 in every try); I/O the exact text above,
  `BuildBudgetLines, InviscidScene_PressureSolveLine`, `RunBudget, LiquidsRoundTrip`; Blender
  `test_water_preset_viscosity_zero`, `test_performance_panel_draws_liquids_lines`.

### Owner questions raised here

1. `kSmallestVisibleViscousEffectPerStep = 2.5e-4` (0.1 % per step) is a new constant.
2. **Answered by A26 (stick).** At a liquid-material face the viscous solve slides (no tangential stick) and leaves out the normal viscous
   stress of the material's own velocity change in the step (method above). The shared-velocity papers (Zarifi and
   Batty 2017, Fang et al. 2020) stick; doing so needs the material velocity as a velocity unknown of the viscous
   solve.
3. **Answered by A27 (lifted).** Free solids beside a viscous liquid stay excluded (not covered by A24).

## A viscous liquid sticks, and free solids in the viscous solve (A26, A27)

Written 2026-11-14 from owner answers A26 and A27 and the owner's note that liquid water melting snow is not
modelled (snow stays a blocking material with no special case; analytic tests use springy solids and rigid
bodies; for snow only the hydrostatic rest test and the five-material scene). Plans (tasks marked
`**Amended 2026-11-14 (A26)**` or `(A27)`): viscosity Tasks 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15a, 15b, 17a, 19a,
20, 24, 26, 27a, 28, 29 and new Tasks 15c, 15d; pressure Tasks 47, 48, 59, 62; solids Task 23 and new solids Tasks
for `BuildViscousMaterialFaces`, `ApplyViscousSolveToMaterials`, `ApplySolvedMotionToSolids`; step Tasks 3, 6, 7,
8, 25, 26, 28, 31, 34; I/O and Blender capability counts. **Replaces** in "Viscous liquids with blocking materials
and sealed gas (A24)": the "Material face" bullet (elimination in the joint solve), the "Viscous stress at a
liquid-material face" bullet (slide), the "Kept: `kFreeSolidNotSupported`" bullet, owner questions 2 and 3, and
the material-face changes of the pressure pieces where they concern the joint solve (`SolvePressure` keeps them).
An inviscid liquid (stage 12, `SolvePressure`) is unchanged: it slides along materials and solids, material faces
are eliminated (A18), free solids enter as the rank-one term.

### Unknowns, rows and block form (viscosity plan Task 15c; viscosity design 5 "Coupled terms")

Velocity unknowns `x = (u, w, xi)`: liquid face velocities `u` (solved faces); **material face velocities `w`**,
one per entry of `ViscousMaterialFace` (below); **free-solid motions `xi`**, six per free solid touching liquid
(`V` linear, m/s, world axes; `Omega` angular, rad/s, about the solid's origin at the step start). Pressure-like
unknowns `(p, q)`: cell pressures and sealed gas blob pressures (A24, unchanged). Momentum rows:

```text
(M + dt E^T W E) x = M x_before - dt E^T W g + dt D^T (p, q) + given push
D x + C ((p, q) - carried) = given flows
M = diag(face_inertia (u), face_mass (w), solid mass * I3 and inertia at the start pose (xi))     # SPD
E = [E_u  E_w  E_s R]     # deformation-rate samples; R maps (V, Omega) to the solid's velocity at a point
D = [D_u  D_w  D_s]       # flow rows: open areas (u), blocked areas A_k (w), FreeSolidPush entries (xi)
```
`K = M + dt E^T W E` is symmetric positive definite (M SPD, `E^T W E` PSD); the joint system is the symmetric saddle
point of A24 with `x` in place of `u`; its Schur complement on `(p, q)`, `dt D K^-1 D^T + C`, is symmetric positive
semidefinite with null space the constants of `kSealed` liquid blobs (unchanged). The outer loop and its
acceptance are A24's; the inner velocity solve (conjugate gradients) runs over `x`. **Nothing is eliminated** in
the viscous solve except sealed gas (never eliminated, A24): `w` and `xi` have viscous terms now, so the A18
elimination (`g_k = dt A^2 / m`) and the free-solid rank-one term do not apply. **Cost:** one unknown per
`ViscousMaterialFace` entry and six per free solid touching viscous liquid, plus one 6 x 6 block per solid.

- **Which material faces are unknowns (plan decision).** Per blocking set, the faces with face mass
  (`value_kind == kFromParticles`) whose centre lies within `kFitReach` (1.5) cell widths, in the width of the cell
  holding the face centre, of the centre of a cell with `filled_volume > 0` of the liquid: the band every
  deformation-rate sample of the liquid can read (edge samples read one width, fitted samples `kFitReach`). Every
  `BlockingMaterialFace` piece is in the band. The rest of the material keeps its velocity after `ApplyForces` and
  `CollideSolids` in this step (as in an inviscid scene away from liquid); the drag reaches it next step through
  `materials::ApplyInternalForce`. Reason: the pressure coupling (A18) also involves only faces touching liquid;
  the band is the smallest set that makes every sample's material side an unknown.
- **Material row.** `face_mass * w_after = face_mass * w_before + dt A_k (p_low - p_high)` (only for band faces
  that are `BlockingMaterialFace` pieces, `A_k` the blocked area) `- dt (E_w^T W (E x + g))` (the viscous traction
  from the liquid). `w_before` is the set's velocity after `ApplyForces` (which applied the material's internal
  force) and `CollideSolids`: the internal force enters only through it, so it is not counted twice. The material
  carries no sample weight of its own (its stress is the materials module's).
- **Stick at a liquid-material face (discretisation, plan decision).** Rule (a) of the edge samples, generalised:
  for a missing liquid face `F` with usable partner `P`, the surface between is found from the coverage
  `distance_to_solid` (which includes blocking materials) at `P`'s centre; when its corner owner is a blocking
  material and `0 < d <= cell_width`, the derivative is `(u_P - w_F) / d` (coefficient `+-0.5 / d` on `P`, `-+0.5 /
  d` on the unknown `w_F` of the band face of that set at `F`, the material's velocity at its surface), exactly as
  a solid's known velocity at its true distance, but unknown. Using the material face at `F` one cell away with
  `1 / cell_width` instead would undercount the interface's shear by a factor 4 (worked out on the Couette slab:
  the edge sample weighs only the liquid half). Fitted samples: a `liquid_covered_solid_centre` whose owner is a
  blocking material takes, per axis, the unknown of that set's band face of that axis nearest the point (ties:
  lower face place index) — an approximation of the material's velocity at the point, first order in the cell
  width. Weights stay the liquid quarters' (`filled_volume`). On mixed sizes the material faces sit on the
  same face pieces as the liquid's, so the fitted rule applies unchanged. A rigid translation of liquid and
  material together gives every sample rate 0, so stick costs nothing when they move together. Where two sets
  hold one face piece, the set with the larger face mass is used (ties: lower set index).
- **Free solid rows.** `[m I3, 0; 0, I_start] xi_after = [m I3, 0; 0, I_start] xi_before + dt sum_e (force_per_pressure,
  torque_per_pressure)_e p_e - dt R^T E_s^T W (E x + g)`. `xi_before` is `solids`' motion before pressure (gravity
  and the contact reactions of the last step already in it, the known forces; contact reactions of this step are
  applied in stage 17 as before). The pressure part uses the existing `pressure::FreeSolidPush` entries (shared with
  `SolvePressure`); new is the viscous part. **Stick at the solid's surface:** a sample whose rule (a) finds a
  free solid uses, instead of a known `rate_from_solids`, coefficients on `xi`: the solid's velocity along axis
  `a` at point `y` is `e_a . V + ((y - origin) x e_a) . Omega`, so each such use adds six coefficients (`R`).
  Fitted samples' `liquid_covered_solid_centre` of a free solid likewise. Driven and fixed solids stay in
  `rate_from_solids`. Faces covered by the free solid are not unknowns of their own (their velocity is the body's).
- **Sums in a fixed order.** Each solid's six sums (`R^T E_s^T ...` and its pressure part) by the per-group rule:
  per cell block, the solid's uses in ascending sample index, then `math::TreeSum` over its block partials in
  ascending block index (as pressure's free-solid sums). Material faces are face places: their sums are the face
  pass (gather by destination). Dots over `x`: liquid and material faces by block partials and `math::TreeSum`, plus
  the solids' part (`math::TreeSum` over solids in index order, components in order `V, Omega`), added in that order.
- **Inner preconditioner.** Diagonal scaling for `u` and `w` (`face_inertia` or `face_mass` plus the sum of squared
  stored coefficients); per free solid the exact inverse of its 6 x 6 block `[m I3, 0; 0, I] + dt R^T E_s^T W E_s R`
  (built once per solve, Cholesky).
- **Outer diagonal.** `ComputeOwnWeight` with `connection_scale` for liquid pieces and, for material faces,
  `material_face_scale_k = face_mass_k / velocity_diagonal_k` on `g_k` (the same rule as a liquid connection); free
  solids left out of the diagonal (as the ladder leaves them out of `SolvePressure`).
- **Sealed gas:** unchanged from A24.
- **In pieces (A15):** material-face entries stream with the face rows (they are face pieces); the free solids'
  motions, 6 x 6 blocks and inverses stay whole on the processor; each sample's solid coefficients (solid index and
  six numbers) stream with the samples; per-solid sums by block partials as above.
- **Snow:** no special case; melting of snow by liquid water is not modelled, so liquid-on-snow scenes run
  mechanically but are not a validation target.

### Declarations

```cpp
// solids/viscous_material_faces.h (NEW, A26). One band face of one blocking set (rule above).
struct ViscousMaterialFace {
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
  std::uint32_t material_set = 0;        // Index into the step's blocking sets.
  double face_mass = 0;                  // kg, > 0.
  double velocity_before_pressure = 0;   // m/s along +axis, after ApplyForces and CollideSolids.
  std::uint32_t blocking_face_index = 0xFFFFFFFFu;  // Its pressure::BlockingMaterialFace entry, or none.
};
class ViscousMaterialFaces;             // Move-only view into SolidsBuffers, sorted by face_piece_index, then set.
std::expected<ViscousMaterialFaces, SolidsFailure> BuildViscousMaterialFaces(const BlockingMaterialFacesInputs&,
    const BlockingMaterialFaces& pressure_faces, const pressure::GeometrySnapshot& geometry, SolidsBuffers&);
std::span<const ViscousMaterialFace> ToViscousMaterialFaces(const ViscousMaterialFaces&);
// Writes each solved band velocity into its set's GridFromParticles::velocity; push report as ApplyPressureToMaterials.
std::expected<PressureOnMaterials, SolidsFailure> ApplyViscousSolveToMaterials(const ViscousMaterialFaces&,
    std::span<const double> velocity_after_solve, std::span<const BlockingSet>, SolidsBuffers&);
// solids/motion_map.h (NEW, A27). The free solids' inputs to the viscous solve: every free solid of the try, in
// FreeSolidPush::solid_index order (solids plan decision: one index for the motion span; a solid not touching liquid
// has no sample use and moves by its own row).
struct FreeSolidForViscousSolve {
  std::uint32_t solid_index = 0;          // As FreeSolidPush::solid_index.
  double mass = 0;                        // kg
  Mat3 inertia_at_start;                  // kg m2, world axes, about origin_at_start
  Vec3 origin_at_start{};                 // m
  Vec3 velocity_before_pressure{};        // m/s
  Vec3 turn_rate_before_pressure{};       // rad/s
  FreeMotionLimits limits{};              // NEW (guides): forbidden motions are held at their value before pressure.
};
std::span<const FreeSolidForViscousSolve> ToFreeSolidsForViscousSolve(const MotionMap&);
struct FreeSolidMotion { Vec3 velocity{}; Vec3 turn_rate{}; };   // m/s, rad/s
// Like ApplyPressureToSolids, with the motion solved by the viscous solve (travel and turn limits checked again,
// kSolidMovesTooFar); push report = mass (after - before), pressure part kept separately.
std::expected<void, SolidsFailure> ApplySolvedMotionToSolids(const MotionMap&,
    std::span<const FreeSolidMotion> motion_after_solve, const grid::CellValues<double>& pressure,
    SolidsTry&, SolidsBuffers&);

// viscosity (A26, A27): ViscosityInputs gains
//   std::span<const solids::ViscousMaterialFace> material_faces;
//   std::span<const solids::FreeSolidForViscousSolve> free_solids;
// (the viscosity design's section 4 uses the solids types; solids does not include viscosity.)
// SolvedCouplingArrays gains
//   std::span<double> material_face_velocity_after_solve;      // one per ViscousMaterialFace
//   std::span<solids::FreeSolidMotion> free_solid_motion;      // one per free solid
// (blocking_material_velocity_after_pressure stays filled, for the BlockingMaterialFace subset.)
// ViscosityBufferSettings gains std::uint32_t max_material_faces = 0, max_free_solids = 0.
// Memory estimate entries: "viscosity: material face vectors" (given max_material_faces, 12 x 8 + 16 bytes),
// "viscosity: free solid vectors and blocks" (given max_free_solids, (12 x 6 + 2 x 36) x 8 bytes),
// "viscosity: sample solid coefficients" (given max_sample_count, 4 + 6 x 8 bytes).

// pressure (A26, A27) coupled_solve_pieces.h, for the joint solve:
struct CoupledVelocities {                              // Given velocity unknowns other than the liquid's.
  const Buffer<double>* blocking_material_velocity = nullptr;   // one per BlockingMaterialFace
  const Buffer<double>* free_solid_motion = nullptr;            // 6 per free solid: V then Omega
};
std::expected<void, PressureFailure> ComputeVolumeError(const PressureEquations&, const grid::CellValues<double>&
    pressure, const Buffer<double>& sealed_gas_pressure, const grid::FaceValues<double>& velocity, CoupledVelocities,
    GivenTerms, grid::CellValues<double>& volume_error, Buffer<double>& sealed_gas_volume_error);
// Material face flow A_k w_k - coverage_volume_rate (kIncluded) or A_k w_k (kLeftOut); a free solid's flow
// sum_e push_e . (xi - xi_before) (kIncluded) or . xi (kLeftOut), sign as SolvePressure's free-solid term.
std::expected<void, PressureFailure> ComputeCoupledPush(const PressureEquations&, const grid::CellValues<double>&
    pressure, Buffer<double>& blocking_material_push, Buffer<double>& free_solid_push);
// dt A_k (p_low - p_high) per BlockingMaterialFace; dt sum_e (force, torque)_per_pressure p_e per free solid.
std::expected<void, PressureFailure> ComputeOwnWeight(const PressureEquations&, const grid::FaceValues<double>*
    connection_scale, const Buffer<double>* material_face_scale, grid::CellValues<double>& own_weight,
    Buffer<double>& sealed_gas_own_weight);
std::expected<FlowImbalanceReport, PressureFailure> CheckFlowImbalance(const PressureEquations&,
    const grid::CellValues<double>& pressure, const Buffer<double>& sealed_gas_pressure,
    const grid::FaceValues<double>& velocity, CoupledVelocities, const SolveSettings&, PressureBuffers&);
```
Added by the pressure plan (A27, plan decision): `EquationInputs::free_solid_motion_before_pressure`
(`std::span<const double>`, 6 per free solid in `FreeSolidPush::solid_index` order, V then Omega; size 0 or 6 x the
free solid count, else `kBadInput`), filled by the step in stage 10 from `solids::ToFreeSolidsForViscousSolve`; the
free solid count is half the size of `free_solid_inverse_inertia`; the free-solid flow term carries a minus sign in
the volume error (`-push_e . (xi - xi_before)`), the sign that makes `SolvePressure`'s operator term positive.
The A24 forms without `CoupledVelocities` stay (they eliminate material faces from the pressure; `SolvePressure`'s
arithmetic) and are no longer called by viscosity.

### Exclusions (step, viscosity, I/O, Blender)

- **Lifted:** `ViscosityError::kFreeSolidNotSupported`, `ViscositySceneSummary::has_free_solid_touching_liquid`,
  `CapabilityReason::kViscousLiquidWithFreeSolid` (stable names `free_solid_not_supported`,
  `viscous_liquid_with_free_solid`), the known-open test `ViscosityKnownOpen, PlateDraggedThroughSyrup_...`
  (becomes a passing scene test). `ViscosityError` 14 values before the plan's three lifting tasks (11 if all are
  lifted); `kCapabilityReasonCount` **16** at most, 13 if all three are lifted.
- **`SuggestedChange::kSetViscosityToZero`** has no user in the main sequence: removed; `kSuggestedChangeCount`
  **12**. The sand module (A9, after the add-on) adds it back with `kViscousLiquidWithSand` (sand plan; flagged for
  the owner: that is the one combination still refused because of viscosity, outside the main sequence).
- **Remaining refusals because of viscosity in the main sequence:** none by material combination. The three
  geometry exclusions the viscosity plan tries to lift (inclined solid, driven solid cutting cells, size change at
  a surface, solid or viscosity change) stay only if their lifting tests fail (then committed known-open).

### Step (stage 10, 11)

Viscous scene, stage 10: as A24, plus (scene with a blocking set) `solids::BuildViscousMaterialFaces` after
`BuildBlockingMaterialFaces`, and `solids::ToFreeSolidsForViscousSolve(motion_map)`. Stage 11:
`viscosity::SolveViscosityAndPressure(ViscosityInputs{..., material_faces, free_solids}, ...,
SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure,
material_face_velocity_after_solve, free_solid_motion})`; `pressure::ComputeWasSolved`;
`solids::ApplySolvedMotionToSolids(motion_map, free_solid_motion, fields.pressure, solids_try, solids_buffers)`
(in place of `ApplyPressureToSolids`); `solids::ApplyViscousSolveToMaterials(viscous_faces,
material_face_velocity_after_solve, blocking_sets, solids_buffers)` (in place of `ApplyPressureToMaterials`; each
band face of each set gets its solved velocity). Step-owned rows `"step: material face velocity after solve"`
(given solids' `max_motion_map_entries` x 2, 8 B) and `"step: free solid motion"` (given the free solid count,
48 B). Viscosity buffers: `max_material_faces` = 2 x `max_motion_map_entries`, `max_free_solids` = free solid
count. Scenes handed to the step plan: the five-material scene with snow and a free box beside the syrup
(`TryStep, FiveMaterialsWithSnowAndFreeBox_Commits`), `TryStep, FloatingBoxInSyrup_RestDepthAsInWater`.

### Tests (viscosity plan Tasks 15d, 19a, 20; numbers derived in the plan)

Springy slab dragged in plane shear (Couette, `T = (sigma + rho H / 3) H / mu`); viscous block on a tilted springy
slab (film bound) against the inviscid slide (`g sin(theta) t`); rigid cube settling in syrup (Stokes linearity
exact, absolute value a stated bound); neutrally buoyant cube in shear (`Omega = shear rate / 2`, exact for a body
of cubic symmetry in Stokes flow); floating box in syrup equal to the inviscid run; momentum of liquid plus body
changed only by the air-end impulse; limit viscosity -> 0 equal to `SolvePressure`; batched and GPU forms. Snow:
only `ViscosityScene, SyrupOnSnowBed_StaysAtRest` and the step's five-material scene.

### Owner questions raised here

1. The material band (faces within `kFitReach` widths of a liquid cell centre) and "the rest of the material keeps
   its velocity after `CollideSolids` this step".
2. Where two blocking sets hold one face piece beside viscous liquid, the set with the larger face mass is the
   sample's material side.
3. Free solids are left out of the outer loop's diagonal (a preconditioner choice).
4. `kSetViscosityToZero` removed from the main sequence; the sand module's `kViscousLiquidWithSand` remains.
6. Guided free solids (solids plan finding): `FreeSolidForViscousSolve::limits` carries the guide; the viscous solve
   holds each forbidden component at its value before pressure (its row and column of the 6 x 6 block replaced by
   the identity, its coefficients folded into `rate_from_solids`), which keeps `K` symmetric positive definite.
   `ApplySolvedMotionToSolids` reports push from the solved motion; solids exposes `SolidsTry::solved_motion()` and
   `pressure_on_solids()` (the step passes the latter to `CheckSolidExchange` after stage 11).
5. Stage 19's `CheckSolidExchange` on a free solid moved by stage 11 (plan decision, step design Open): it compares
   the pressure part that `ApplySolvedMotionToSolids` reports separately with the flow work, as in stage 12; the
   viscous part of the push is checked by viscosity's momentum test (`BoxInLiquidBlock_MomentumChangedOnlyAtAirEnds`),
   not by the exchange balance.

Step additions from the pressure plan's field: stage 10 calls `ToFreeSolidsForViscousSolve` before
`BuildPressureEquations` and fills `free_solid_motion_before_pressure` from it (stage 12 passes it empty); step-owned
row `"step: free solid motion before pressure"` (48 B per free solid).

## Foundation after the final review (2026-11-14, owner review 2026-11-15)

The final whole-branch review of Foundation, and the touch-up after its re-review, changed or added the declarations
below; they are copied from the headers of the `rebuild` branch after the owner-review fold (the tree of fixup tip
d918805a, A28 to A32). A plan that uses an earlier form follows these. Section (a) of `shared-interface.md` follows the same
headers.

### Rules a module now copies

- **Two halves.** The public function lives in `<name>_cpu.cc`: it checks, dispatches by `RunOn`, runs the CPU
  loop, and calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`. The GPU half is `<name>_gpu.cu`, declared in
  the private `<name>_gpu.h`, returns `RunOnCallStatus`, and launches, waits and holds GPU memory of its own
  through `base/gpu_launch.cuh`; a direct `cudaMemcpy` or `cudaMemset` is allowed and its status goes through
  `ToRunOnCallStatus`. `kOutOfMemory` comes only from an allocation (nothing written); every other failed GPU call
  is `kCallFailed`. The CPU half maps the status with one function (`ToBufferError` for `BufferError`).
- **Kernel-wrapper rule.** A GPU function (`__global__`) is a thin loop around `FLUIDITY_CPU_AND_GPU` per-item or
  per-thread functions in a shared header; `cpu` tests call those functions over every item, or every thread block and
  thread in launch order and in reverse (phase by phase where the GPU function waits), against a plain reference
  (`base/tree_sum_layers_test.cc`, `base/stable_order_steps_test.cc` are the examples).
- **Working buffers.** `<Name>Buffers` (move-only), `Build<Name>Buffers(run_on, max_count)`,
  `Compute<Name>BuffersSize(run_on)` returning `<Name>BuffersSize { bytes_per_item; fixed_bytes; }` (exact bytes,
  one definition). ONE form per function, which takes its buffers and allocates nothing (`kSizeMismatch` for
  `count > max_count()`); `kOutOfMemory` comes only from `Build<Name>Buffers` and `BuildBuffer`. The buffers hold
  their memory in one `Buffer`; their contents mean nothing between calls (A31).
- **Preconditions.** Two buffers (or a buffer and its working buffers) on one processor is a precondition, stated on
  a `Requires:` line and checked by `assert` from `<cassert>` in debug builds only; there is no error value for it
  (`BufferError::kRunOnMismatch` is removed, A30). Checks that remain run sizes, then values.
- **Verbs.** `Copy`, `Free` (frees memory: `FreeBufferBytes`, `GpuArray::Free`) and `Take` (gets preallocated
  buffers back out of an object no longer needed); `Release` is not a verb (A29). Naming rule 9 has no exception:
  the step's input is `StepInputs` (A28).
- **Vocabulary.** The GPU sense is always "thread block" (`kThreadBlockSize`, `LaunchSize::thread_block_count`);
  never "block" or "group" alone in that sense. "group" is a destination group, "cell block" the grid's.
- **Exactness.** Discrete outputs are equal on CPU and GPU; floating-point results agree within a tolerance,
  except functions whose header promises the same bits (unless NaN).
- **Errors.** `base/` returns the bare `RunOnError` and `BufferError`; a module maps them. A failed check writes
  nothing; after `kRunOnCallFailed` the outputs are unspecified and the run must stop.
- **Test support.** Shared helpers live in `fluidity_base_test_support` (`base/test_support.h`); a module's own go
  into its own `fluidity_add_test_library` and its tests name them in `LINKS`. GPU tests append result digests
  (`WriteResultDigest`) so the hardware repeat run compares bytes.
- **Renamed headers.** `run_on_gpu_check_math.h` -> `check_can_run_on_math.h`; `check_can_run_on.cc` ->
  `check_can_run_on_cpu.cc`.

### CMake helpers

```cmake
fluidity_add_test(<name> SOURCES <files...> LABEL <cpu|gpu|no_gpu> [LINKS <targets...>])
fluidity_add_test_library(<name> SOURCES <files...> [GPU_SOURCES <files...>] [LINKS <targets...>])
fluidity_apply_compile_flags(<target> EXCEPTIONS <ON|OFF>)   # also sets FLUIDITY_COMPILE_FLAGS_APPLIED
fluidity_check_compile_flags()        # last line of the top-level CMakeLists.txt; fails for an unmarked target
fluidity_write_gpu_only_sources()     # writes gpu_only_sources.txt, which the gate's cuda step tidies
fluidity_record_gpu_only_sources(<files...>)   # the .cc among them go to gpu_only_sources.txt; called by
                                               # fluidity_add_test (gpu, no_gpu), fluidity_add_gpu_sources and
                                               # fluidity_add_test_library's GPU_SOURCES
fluidity_add_headers(<files...>)      # a .cuh gets only a .cu header-alone check, with CUDA on
fluidity_add_gpu_sources(<files...>)
```

The gate's cuda step also fails, naming it, for a tracked native `.h` or `.cuh` that is not compiled alone
(`fluidity_add_headers`), except a test header (`test` as a word of its file name). With CUDA on, configuration
fails while `NVCC_PREPEND_FLAGS` or `NVCC_APPEND_FLAGS` is set.

### `base/run_on_call.h` (shared)

```cpp
// How a call to a GPU half ended. The CPU half turns it into its error: ToBufferError for BufferError.
enum class RunOnCallStatus : std::uint8_t {
  kDone,  // The outputs hold the answer.
  kOutOfMemory,  // An allocation of the call failed. Nothing was written. Only allocations return it.
  kSizeMismatch,  // A check made by the GPU half refused the inputs, such as a total above 2^32 - 1. Nothing written.
  kCallFailed,  // Another GPU call failed. The outputs are unspecified.
};

// Threads in one thread block of a simple GPU function: one thread per item, as many items as threads.
inline constexpr unsigned int kThreadBlockSize = 256;

// The most thread blocks a simple GPU function starts. The GPU accepts up to 2^31 - 1 thread blocks along x; fewer
// are started, and each thread loops over several items, so a count of any size works.
inline constexpr unsigned int kMaxThreadBlockCount = 65535;

// The shape of one launch of a GPU function.
struct LaunchSize {
  unsigned int thread_block_count = 0;  // Thread blocks; at least 1 for a launch.
  unsigned int thread_block_size = 0;  // Threads in each thread block.
};

FLUIDITY_CPU_AND_GPU constexpr LaunchSize ComputeLaunchSize(std::size_t item_count);
```

### `base/gpu_launch.cuh` (GPU halves only)

```cpp
inline RunOnCallStatus ToRunOnCallStatus(cudaError_t error);
template <typename... Parameters>
RunOnCallStatus ApplyGpuFunction(
    void (*function)(Parameters...),
    LaunchSize launch_size,
    std::type_identity_t<Parameters>... arguments
);
inline RunOnCallStatus CheckGpuWork();
struct GridStride {
  std::size_t first_item = 0;  // The global index of the thread.
  std::size_t item_step = 0;  // The number of threads in the launch.
};
__device__ inline GridStride ComputeGridStride();
template <typename T>
class GpuArray;   // RunOnCallStatus Build(std::size_t count); RunOnCallStatus Free(); T* values() const;
```

### `base/buffer.h` (changed and new)

```cpp
enum class BufferError : std::uint8_t {
  kNone,  // No error. No function returns it as an error.
  kOutOfMemory,  // The chosen processor could not hold the array. Only allocations return it; nothing was written.
  // CheckCanRunOn failed for the chosen processor, or a call to it failed. The reason is not kept here: the outer
  // boundary calls CheckCanRunOn(run_on) before any work, which returns the RunOnError and its message.
  kRunOnCallFailed,
  // Sizes do not fit: a span and a buffer, or two buffers, have different sizes; a slice is out of range; two ranges
  // overlap; or a count or a total is above what the function can index (each function's Fails says which). The
  // last case can come from a scene that is too large, not only from a bug in the caller.
  kSizeMismatch,
};

BufferError ToBufferError(RunOnCallStatus status);

// Requires: from.run_on() == to.run_on(), checked by assert in debug builds.
template <BufferValue Value>
std::expected<void, BufferError>
CopyBuffer(const Buffer<Value>& from, std::size_t first, std::size_t count, Buffer<Value>& to, std::size_t to_first);
```

Only the building functions (`BuildBuffer`, `BuildSumBuffers`, `BuildOrderBuffers`) return `kOutOfMemory`; a GPU
copy or query that fails for any reason, a memory error included, is `kRunOnCallFailed`.

### `base/ordered_sums.h` (working buffers; one form per sum)

```cpp
class SumBuffers;
std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count);
struct SumBuffersSize {
  std::uint32_t bytes_per_item = 0;  // Bytes for each value of max_count.
  std::uint64_t fixed_bytes = 0;  // Bytes that do not depend on max_count.
};
SumBuffersSize ComputeSumBuffersSize(RunOn run_on);
// Each Requires: values (and second) and buffers on one processor, checked by assert in debug builds.
std::expected<double, BufferError> ComputeTreeSum(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);
std::expected<double, BufferError>
ComputeTreeDot(const Buffer<double>& first, const Buffer<double>& second, std::size_t count, SumBuffers& buffers);
std::expected<double, BufferError>
ComputeLargestMagnitude(const Buffer<double>& values, std::size_t count, SumBuffers& buffers);
class SumBuffers {   // Move-only; max_count(), run_on(), byte_count(); moved-from buffers have max_count() 0.
};
```

There is no form without buffers (A31). `ComputeSumBuffersSize(RunOn::kGpu)` is `{14, 128}`,
`ComputeSumBuffersSize(RunOn::kCpu)` is `{0, 0}`; CPU sum buffers hold no bytes but still check the count.

### `base/stable_order.h` (working buffers; one form per function)

```cpp
class OrderBuffers;
std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count);
// Each Requires: every buffer and the order buffers on one processor, checked by assert in debug builds.
std::expected<void, BufferError> BuildStableOrder(
    const Buffer<std::uint64_t>& keys,
    std::size_t count,
    const Buffer<std::uint32_t>* previous_order,
    Buffer<std::uint32_t>& order,
    OrderBuffers& buffers
);
std::expected<void, BufferError> ComputeExclusivePrefixSum(
    const Buffer<std::uint32_t>& values,
    std::size_t count,
    Buffer<std::uint32_t>& sums,
    OrderBuffers& buffers
);
std::expected<void, BufferError> BuildGroupsByDestination(
    const Buffer<std::uint32_t>& destination,
    std::size_t item_count,
    std::uint32_t destination_count,
    Buffer<std::uint32_t>& first_of_group,
    Buffer<std::uint32_t>& item_order,
    OrderBuffers& buffers
);
struct OrderBuffersSize {
  std::uint32_t bytes_per_item = 0;  // Bytes for each item of max_count.
  std::uint64_t fixed_bytes = 0;  // Bytes that do not depend on max_count.
};
OrderBuffersSize ComputeOrderBuffersSize(RunOn run_on);
class OrderBuffers {   // Move-only; max_count(), run_on(), byte_count(); moved-from buffers have max_count() 0.
};
```

There is no form without buffers (A31). `ComputeOrderBuffersSize(RunOn::kCpu)` is `{4, 8192}`,
`ComputeOrderBuffersSize(RunOn::kGpu)` is `{6, 4096}`. The same `OrderBuffers` serves the order, the prefix sum and
the grouping.

### `base/tree_sum_layers.h` (shared, private to the sums)

```cpp
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputePairCount(std::size_t side_count);
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputeLayerValueCount(std::size_t count);
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputeLayerArrayOffset(std::size_t count, std::size_t layer);
struct LayerSides {
  const double* values = nullptr;  // `count` sides in memory of the processor that runs the layer.
  std::size_t count = 0;  // The number of sides.
};
FLUIDITY_CPU_AND_GPU inline double ComputeLayerPairSum(LayerSides sides, std::size_t pair);
FLUIDITY_CPU_AND_GPU inline double ComputeLayerLargerMagnitude(LayerSides sides, std::size_t pair);
```

### `base/stable_order_steps.h` (shared, private to the stable order)

```cpp
inline constexpr std::uint32_t kChunkThreadBlockSize = kDigitValueCount;
inline constexpr std::uint32_t kItemsPerThread = kItemsPerChunk / kChunkThreadBlockSize;
inline constexpr int kMaxPrefixSumLayerCount = 8;
template <typename Key>
struct PassSource {
  const Key* keys = nullptr;  // Indexed by item.
  const std::uint32_t* sequence = nullptr;  // `count` items, or null for 0, 1, ..., count - 1.
  std::size_t count = 0;  // Places in the sequence.
  std::uint32_t pass = 0;  // Below 8.
};
struct ChunkItems {
  std::uint8_t* digits = nullptr;
  std::uint32_t* items = nullptr;
  std::uint32_t item_count = 0;  // ComputeChunkItemCount of the chunk; set once the chunk is copied.
};
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t ComputeChunkItemCount(std::size_t count, std::size_t chunk);
template <typename Key>
FLUIDITY_CPU_AND_GPU void
CopyChunkItem(PassSource<Key> source, std::size_t chunk, std::uint32_t place, ChunkItems chunk_items);
FLUIDITY_CPU_AND_GPU inline std::uint32_t ComputeChunkDigitCount(ChunkItems chunk_items, std::uint32_t digit);
FLUIDITY_CPU_AND_GPU inline void CopyChunkItemsOfDigit(
    ChunkItems chunk_items,
    std::uint32_t digit,
    std::uint32_t* next_sequence,
    std::uint32_t first_place
);
template <typename Value>
struct PrefixSumValues {
  const Value* values = nullptr;
  std::size_t count = 0;
};
struct ThreadValues {
  // A C array: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
  std::uint64_t own_values[kItemsPerThread] = {};  // NOLINT(modernize-avoid-c-arrays): see the comment above.
  std::uint64_t sum = 0;
};
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputeThreadFirstIndex(std::size_t chunk, std::uint32_t thread);
template <typename Value>
FLUIDITY_CPU_AND_GPU ThreadValues ComputeThreadValues(PrefixSumValues<Value> values, std::size_t thread_first);
struct ThreadBlockRound {
  std::uint32_t thread = 0;  // Below kChunkThreadBlockSize.
  std::uint32_t distance = 0;  // The halving rounds use 128, 64, ..., 1; the doubling rounds 1, 2, ..., 128.
};
FLUIDITY_CPU_AND_GPU inline void ApplyHalvingRound(std::uint64_t* thread_sums, ThreadBlockRound round);
FLUIDITY_CPU_AND_GPU inline std::uint64_t
ComputeRunningSumAddend(const std::uint64_t* running_sums, ThreadBlockRound round);
template <typename Sum>
struct PrefixSums {
  Sum* sums = nullptr;
  std::size_t count = 0;
};
template <typename Sum>
FLUIDITY_CPU_AND_GPU void CopyThreadPrefixSums(
    ThreadValues thread_values,
    std::uint64_t first_running_sum,
    PrefixSums<Sum> output,
    std::size_t thread_first
);
struct SortedDestinations {
  const std::uint32_t* values = nullptr;
  std::size_t count = 0;
};
FLUIDITY_CPU_AND_GPU inline std::uint32_t ComputeGroupStart(SortedDestinations destinations, std::size_t group);
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t ComputeDestinationPassCount(std::uint32_t destination_count);
struct PrefixSumLayers {
  // C arrays: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
  std::size_t first_entry[kMaxPrefixSumLayerCount + 1] = {};  // NOLINT(modernize-avoid-c-arrays): see above.
  std::size_t entry_count[kMaxPrefixSumLayerCount + 1] = {};  // NOLINT(modernize-avoid-c-arrays): see above.
  int top_layer = 0;
};
FLUIDITY_CPU_AND_GPU constexpr PrefixSumLayers BuildPrefixSumLayers(std::size_t value_count);
```

### `base/test_support.h` (library `fluidity_base_test_support`, tests only)

```cpp
std::uint64_t ToBits(double value);
std::vector<std::uint64_t> ToBits(std::span<const double> values);
struct Tolerance {
  double absolute_floor = 0.0;  // Same unit as the values; the allowed difference near zero.
  double relative_term = 0.0;  // No unit; the allowed difference per unit of |reference|.
};
::testing::AssertionResult CheckWithinTolerance(double reference, double compared, Tolerance tolerance);
::testing::AssertionResult CheckSameBitsOrBothNaN(double expected, double actual);
template <std::ranges::contiguous_range Values>
Buffer<std::ranges::range_value_t<Values>> BuildBufferHolding(RunOn run_on, const Values& values);
template <BufferValue Value>
std::vector<Value> ReadAll(const Buffer<Value>& buffer);
std::vector<double> BuildMixedValues(std::size_t count);
std::vector<std::uint64_t> BuildKeysWithManyTies(std::size_t count);
std::vector<std::uint64_t> BuildKeysDifferingInTopByte(std::size_t count);
inline constexpr std::string_view kResultDigestFileVariable = "FLUIDITY_RESULT_DIGEST_FILE";
std::uint64_t ComputeResultDigest(std::span<const std::byte> bytes);
void WriteResultDigestToFile(
    const std::filesystem::path& path,
    std::string_view name,
    std::span<const std::byte> bytes
);
template <typename Value>
void WriteResultDigest(std::string_view name, std::span<const Value> values);
std::string FindResultDigestFile();
```

## Plans adopt Foundation as built (2026-11-14)

The fifteen module plans were amended to build on Foundation as built (final-fix-report.md section 7; reconcile
item 22). Each changed task carries `**Amended 2026-11-14 (Foundation as built):**`. The rules, in ten lines:

1. A: public function in `<name>_cpu.cc` (checks, `RunOn` dispatch, CPU loop, GPU half under `FLUIDITY_HAS_GPU`); GPU half `<name>_gpu.cu` in private `<name>_gpu.h`, returning `RunOnCallStatus`, mapped by `ToBufferError` then one module mapping.
2. A: launches, waits and per-call GPU memory only through `base/gpu_launch.cuh`; integer atomics allowed, floating-point atomics never.
3. A: each GPU function is a thin loop over `FLUIDITY_CPU_AND_GPU` per-item functions in a shared header (`<name>_steps.h` unless an existing formula header serves); a `cpu` test `<name>_steps_test.cc` runs them for every thread block and thread, forward and reverse, against the CPU loop: the main evidence until the hardware run.
4. B: per-step sums and groupings pass a `SumBuffers` / `OrderBuffers` held by the module's buffers (members `sum_buffers`, `order_buffers`), built once from the module's limits; the estimate adds `"<module>: sum|order buffers"` and `"... fixed bytes"` entries from `ComputeSumBuffersSize` / `ComputeOrderBuffersSize`. (Superseded by A31: every caller passes buffers; there is no form that builds its own.)
5. C: tests link `fluidity_base_test_support`; module helper libraries are made with `fluidity_add_test_library` and named in `LINKS`; every source is in a CMake file; every target goes through the helpers (a library that needs exceptions or RTTI calls `fluidity_apply_compile_flags` right after `add_library`).
6. C: labels: `cpu` everywhere (the container runs them), `no_gpu` only for "the GPU was chosen and none is usable", `gpu` only on hardware; `gpu` tests call `WriteResultDigest` on their main outputs.
7. D: a failed check writes nothing; after `kRunOnCallFailed` outputs are unspecified and the run stops; `kOutOfMemory` only from building forms.
8. D: CPU against GPU: discrete outputs equal; floating-point within the module tolerance (`CheckWithinTolerance`), unless the header promises the same bits unless NaN (`CheckSameBitsOrBothNaN`); same-processor repeat and replay tests stay byte tests.
9. E: `MemoryEstimateEntry`, `Build<Module>MemoryEstimate`, "thread block" for the GPU sense (the grid's "cell block" untouched), `check_can_run_on_math.h`, `Uses:` never names a standard header.
10. F: each plan's Global Constraints state the patterns once (pointing to `AGENTS.md` "Structure") and the gate at every commit (CPU debug and release, CUDA compiled in the container, a tracked source no build compiles fails); clang-format and clang-tidy 23.1, not 19.1.

### Helpers retired, and one added

- `fluidity_add_shared_test` (contact Task 2) is retired everywhere: a shared-header test is
  `fluidity_add_test(<name> SOURCES ... LABEL cpu [LINKS ...])`. `fluidity_add_interchange_test` (render output) is
  retired the same way.
- I/O plan Task 34 adds an optional argument; render output Task 4 (`fluidity_interchange`) uses it:

  ```cmake
  fluidity_apply_compile_flags(<target> EXCEPTIONS <ON|OFF> [RTTI <ON|OFF>])   # default OFF; ON replaces
  # -fno-rtti, /GR- and -Xcompiler=/GR- by -frtti / /GR (never added after them); test in tools/fluidity_targets_test.py
  ```
- Known-open tests (pressure, viscosity, solids, gas, step, sand): `fluidity_add_test` takes one label and discovers
  every test of its program under it, so a known-open test is written with GoogleTest's `DISABLED_` prefix (discovery
  registers it as disabled), and registered again inside `if(BUILD_TESTING)` by
  `add_test(NAME <Suite>.<Name> COMMAND <program> --gtest_filter=<Suite>.DISABLED_<Name> --gtest_also_run_disabled_tests)`
  with `set_tests_properties(<Suite>.<Name> PROPERTIES WILL_FAIL TRUE LABELS known_open)`. Closing it drops the
  prefix and the `add_test` line. Sand's rung tests (plain and known-open) are registered only this way, label
  `sand_diagnosis` added. For the owner to confirm.

### Interface changes that cross plans (exact declarations)

```cpp
// pressure (Task 12): the working buffers' bytes depend on the processor. Consumer: step Task 8 (updated).
std::uint64_t ComputeBufferBytes(const BufferSizes& sizes, RunOn run_on);                        // was (sizes)
std::vector<MemoryEstimateEntry> BuildPressureMemoryEstimate(const BufferSizes& sizes, RunOn run_on);  // was (sizes)

// gas (Task 10). Consumer: step Task 8 (updated).
std::vector<MemoryEstimateEntry> BuildGasMemoryEstimate(const GasBufferSizes& sizes, RunOn run_on);    // was (sizes)

// cell resize (Task 23); consumers: render output Task 31, the step. The only form since A31 (the design's form
// without SumBuffers is removed).
std::expected<LiquidTotals, CellResizeFailure> ComputeLiquidTotals(
    const particles::ParticlesTry& particles_try, const GridOnlyAmounts& amounts,
    std::span<const std::uint8_t> material_is_counted, const grid::StaggeredGrid& grid,
    grid::CellBlockRange blocks, SumBuffers& sum_buffers);
```

### Additive changes inside one plan (no consumer edit needed)

- Grid Task 6: `GridBuilderBuffers` follows the pattern: `struct GridBuilderBuffersSize { bytes_per_cell_block; fixed_bytes; }`,
  `GridBuilderBuffersSize ComputeGridBuilderBuffersSize(RunOn run_on);`, `GridBuilderBuffers::byte_count()`. Grid's own `LaunchShape`,
  `ComputeLaunchShape`, `kBuildLaunchBlocks` dropped for `ComputeLaunchSize`.
- Pressure Task 12: `launch_size.h` deleted (base's `ComputeLaunchSize`); Task 28: `block_sums.h` CPU-only with
  `ComputeTotalOfBlocks(const Buffer<double>& block_partials, std::uint32_t block_count, SumBuffers& sum_buffers)`, the
  per-block partial in shared `block_sum_steps.h`. Task 55: `solve_batches.cc` -> `solve_batches_cpu.cc`;
  `SolveBatches` owns an `OrderBuffers`.
- Batches Task 37: `double ComputeSumOverBlocks(const grid::CellBlockValues<double>& block_partial_sums, SumBuffers& sum_buffers);`
- Working buffers members: `PressureBuffers`, `GasBuffers`, `ViscosityBuffers`, `CellResize` (both), `particles` storage
  (both), `step::TryArrays` (both), `StepRecorder` (both), `BatchedRun::sum_buffers`, `GasAmounts` (sum),
  `BorderSearchBuffers`, `SolidsBuffers` (order; plus `Buffer<contact::Solid> solids_copy`, so no GPU half allocates).
  Materials, contact and transfer need none. No `Build<Module>Buffers` signature changed.
- One mapping per module: `ToContactFailure(BufferError)`, `ToViscosityFailure(BufferError, std::uint64_t)`,
  `ToSolidsFailure(BufferError, std::uint64_t)`, `ToCellResizeFailure(BufferError)`, the step's
  `ToStageOutcome(ToBufferError(status))`.
- Same-bits-unless-NaN promises added deliberately, each with its reason in the plan (for the owner, beside D7):
  contact shared formulas for solids that do not turn; transfer `ComputeCellsFromParticles`; particles
  `ComputeAmounts`; materials' per-cell-block budget of `ApplyInternalForce` and `ComputeParticleFriction`; pressure
  `ComputeDistanceToSurface`; gas `BuildTry`, `ComputeGasSeenByLiquid`, `FindMarkerChanges`; step
  `ComputeVelocityAfterGravity`.

## Owner's review of the Foundation rulings applied (A28–A33)

Applied 2026-11-14/15 to the fifteen module plans, the Foundation plan (names only), every module design,
`shared-interface.md` (not section (a)) and this file (not "Foundation after the final review", which is
regenerated from the code). Changed tasks carry `**Amended 2026-11-14 (A28|A29|A30|A31|reconcile):**`; each design's
section 11 has one entry per answer applied and its vocabulary section lists the replaced names once. A32 and A33
approve Foundation as built and change no plan text. A39 (commit attribution) is applied to every plan's Global
Constraints and to "Commits" above.

### Renames (old -> new)

| Old | New | Answer |
|---|---|---|
| `OrderScratch`, `BuildOrderScratch`, `OrderScratchSize`, `ComputeOrderScratchSize` | `OrderBuffers`, `BuildOrderBuffers`, `OrderBuffersSize`, `ComputeOrderBuffersSize` | A31 |
| `SumScratch`, `BuildSumScratch`, `SumScratchSize`, `ComputeSumScratchSize` | `SumBuffers`, `BuildSumBuffers`, `SumBuffersSize`, `ComputeSumBuffersSize` | A31 |
| members `order_scratch`, `sum_scratch`; parameters `scratch` | `order_buffers`, `sum_buffers`; the buffers' name (`order_buffers`, `sum_buffers`, `builder_buffers`, `transfer_buffers`, `point_object_buffers`, `packing_buffers`) | A31 |
| grid `GridScratch`, `BuildGridScratch`, `GridScratchSize`, `ComputeGridScratchSize` | `GridBuilderBuffers`, `BuildGridBuilderBuffers`, `GridBuilderBuffersSize`, `ComputeGridBuilderBuffersSize` (not `GridBuildBuffers`: it would give `BuildGridBuildBuffers`; `GridBuffers` already names one grid's tables) | A31 |
| grid `ScratchSharing { kOnePerBuffers, kOneForAllBuffers }`, `scratch_sharing` | `BuilderBuffersSharing { kOnePerGrid, kOneForAllGrids }` (declared in `staggered_grid.h`, reconcile item 20), `builder_buffers_sharing` | A31 |
| entries `grid: build scratch: ...`, `... ordering`, `border ordering` | `grid: builder buffers: ...`, `... order buffers`, `border order buffers` | A31 |
| variables `grid_scratch`, `cpu_grid_scratch` | `grid_builder_buffers`, `cpu_grid_builder_buffers` | A31 |
| transfer `TransferScratch`, `BuildTransferScratch(run_on, max_particle_count)`, `ComputeTransferScratchSize`, `transfer_scratch.h` | `TransferBuffers` (+ `material_count()`), `BuildTransferBuffers(run_on, max_particle_count, material_count)`, `ComputeTransferBuffersSize`, `transfer_buffers.h` | A31, reconcile 27 |
| render output `PointScratch` | `PointObjectBuffers` | A31 |
| I/O (private) `SnapshotArraysScratch` | `SnapshotPackingBuffers` | A31 |
| step `ModeReason::kSolverScratchDoesNotFitProcessor`, `solver_scratch_does_not_fit_processor` | `kSolverBuffersDoNotFitProcessor`, `solver_buffers_do_not_fit_processor` | A31 |
| `grid::ReleaseBuffers(StaggeredGrid&&)`, `ReleaseBuffers(GridPiece&&)` | `grid::TakeBuffers(...)` (the only `Release...` identifiers in plans and designs; `ReleaseBufferBytes` is Foundation's, section (a)) | A29 |
| `step::StepRequest`, parameter `request` | `StepInputs`, parameter `step_inputs` (not `inputs`: the stages already use `inputs` for the module input structs they build); no earlier `StepInputs` existed | A28 |
| `step::RejectKind::kBadRequest` | `kBadStepInputs` | A28 |
| test libraries `fluidity_<module>_test_helpers` | `fluidity_<module>_test_support` | reconcile 29 |
| tests: grid `SharedScratch_CountedOnce`, `TwoBuffersShareOneScratch_AllocatedEqualsEstimate`, `LentScratchTooSmallOrOtherProcessor_IsBadInput`, `SharedScratch_OtherLiveGridUnchanged`, `OnGpuSharedScratch_OtherLiveGridUnchanged`, `CpuScratch_EqualsByteCountAndScratchRows`, `ComputeGridCounts, IntoBuffers_EqualsAllocatingForm`, `BuildGrid, DefaultSettings_SizedToTheList`, `BuildGrid, OnGpuWithoutAGpu_IsRunOnCallFailed` (Task 9), `BuildGridBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed` | `SharedBuilderBuffers_CountedOnce`, `TwoGridBuffersShareBuilderBuffers_AllocatedEqualsEstimate`, `LentBuilderBuffersTooSmall_IsBadInput`, `SharedBuilderBuffers_OtherLiveGridUnchanged`, `OnGpuSharedBuilderBuffers_OtherLiveGridUnchanged`, `CpuBuilderBuffers_EqualsByteCountAndBuilderBuffersRows`, `IntoBuffers_EqualsSpanForm`, `BuildTestGrid, DefaultSettings_SizedToTheList`, `BuildTestGrid, OnGpuWithoutAGpu_IsRunOnCallFailed`, `BuildGridBuilderBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed` | A30, A31 |
| tests: Foundation `CountAboveScratchMaxCount_IsSizeMismatch`; transfer `ScratchRow_Is168BytesPerParticle`, `OnCpu_HasTheAskedCount`; pressure `ScratchRowsMatchScratchSizes`, `OnCpu_ScratchBytesMatchRows`, `ShortOrMisplacedBuffers_AreBadInput`; viscosity `ScratchRows_MatchScratchByteCounts`; gas `ScratchRowsMatchScratchSizes`; step `StepScratchEntries_FromScratchSizes`, `StepScratchBuiltAtTheLimits`, `StepScratchRaised`, `BuildStepMemoryEstimate, TwoGridBuffersOneScratch`, `BuildSimulation, GridBuffersShareOneScratch`, `ListOnOtherProcessorOrNoCpuGridOnGpu_IsBadInput` | `CountAboveBuffersMaxCount_IsSizeMismatch`; `NearbyFaceSumsBuffersEntry_Is168BytesPerParticle`, `OnCpu_HasTheAskedCounts`; `BuffersRowsMatchBuffersSizes`, `OnCpu_BuffersBytesMatchRows`, `ShortBuffers_AreBadInput`; `BuffersRows_MatchBuffersByteCounts`; `WorkingBufferRowsMatchBuffersSizes`; `StepWorkingBufferEntries_FromBuffersSizes`, `StepWorkingBuffersBuiltAtTheLimits`, `StepWorkingBuffersRaised`, `TwoGridBuffersShareBuilderBuffers`, `GridBuffersShareBuilderBuffers`, `NoCpuGridOnGpu_IsBadInput` | A30, A31 |
| tests: cell resize and batches `ScratchEntriesMatchScratchBytes`, `ComputeLiquidTotals, ScratchTooSmall_IsBadInput`, `ComputeSumOverBlocks, ScratchTooSmall_IsQuietNaN`, `SumScratchEntriesMatchTheScratch`; render output `ScratchReused_NoNewAllocationSecondTime`, `ScratchesSizedFromLimits`, `OnGpuRejectedTry_ScratchLeftUnused`, `RecorderScratchEntriesMatchTheRecorder` | `WorkingBufferEntriesMatchTheBuffers`, `BuffersTooSmall_IsBadInput`, `BuffersTooSmall_IsQuietNaN`, `SumBuffersEntriesMatchTheBuffers`; `BuffersReused_NoNewAllocationSecondTime`, `BuffersSizedFromLimits`, `OnGpuRejectedTry_BuffersLeftUnused`, `RecorderBufferEntriesMatchTheRecorder` | A31 |
| prose "scratch"; temporary-work senses (`<scratchpad>`, scratch program, driver, worktree, folder) | "working buffers" or the buffers' name; `<temp-dir>`, throwaway program or driver, temporary worktree or folder | A31 |

New names: test-only `grid::testing::TestGrid` / `BuildTestGrid(const GridInputs&, const GridSettings& = {})` in
`fluidity_grid_test_support` (replaces the allocating `BuildGrid(inputs, settings)` in tests);
`RenderLimits::max_field_parts = 1u << 26`, `RenderOutputError::kTooManyFieldParts` (value 15, 16 values),
`RenderOutput::frame_order_buffers`; `ViscosityError::kOutOfMemory` (15 values), `SolidsError::kOutOfMemory`
(18 values); `transfer::TransferBuffers::material_count()`.

### One form per function (A31)

Every function that needs working memory has one form, which takes its buffers; the caller owns them (in its
module buffers, or built locally in a test or tool). Foundation's `BuildStableOrder`, `ComputeExclusivePrefixSum`,
`BuildGroupsByDestination`, `ComputeTreeSum`, `ComputeTreeDot`, `ComputeLargestMagnitude` always take their
buffers, also for once-per-frame CPU work. Module functions that had two forms:

- `grid::BuildGridBuffers(settings)` and `(settings, lent)`: only `BuildGridBuffers(settings, builder_buffers)`.
  Pressure's `SolveBatches` owns a `GridBuilderBuffers` for its loaded piece; cell resize and the step lend theirs.
- `grid::BuildGrid(inputs, settings)` (allocating): removed; tests use `BuildTestGrid`, tools build both buffers.
- Kept with a stated reason: `grid::ComputeGridCounts` and `CheckSizeStepRule` keep a span form beside the buffers
  form (the planner counts beyond any room; cell resize checks CPU lists while it splits, before grid buffers exist).
- Transfer stages (`ParticlesToGrid`, `SaveGridBeforeForces`, `SaveGridAfterPressure`, `GridToParticles`,
  `ComputeCellsFromParticles`): one form, `TransferBuffers& transfer_buffers` last.
- Materials `ApplyInternalForce` and `ComputeCellSqueezeValues`: one form, `MaterialBuffers` last.
- `cell_resize::ComputeLiquidTotals` and `batches::ComputeSumOverBlocks`: one form, `SumBuffers& sum_buffers` last
  ("Interface changes that cross plans" above: no added overload).
- Render output's per-frame CPU work (`ComputeAddedVolume`, `BuildSurfaceField`, `BuildSurfaceFrame`,
  `BuildFoamVolume`): take `OrderBuffers&`, owned by `RenderOutput::frame_order_buffers`; `BuildRenderOutput` now
  returns `std::expected<RenderOutput, RenderOutputFailure>` (`kOutOfMemory`).
- Kept as one allocating form, reason in the task: `particles::ComputeAmounts` (callers pass views of particles of
  any owner, with no buffers to lend; a caller that must not allocate uses `ComputeAmountsPerBlock`),
  `materials::ComputeWorkRates` (test oracle) and `CheckMaterialState` (scene load): none runs per step.
- Kept: gas `FindMarkerChanges` amounts form and CPU form (they differ by processor, not working memory).
- Removed tests that only compared two forms: Foundation `BuildStableOrder, WithScratch_EqualsFormWithoutScratch`;
  materials `ApplyInternalForce, BuffersForm_SameBytesAsOneCallForm`; cell resize
  `ComputeLiquidTotals, ScratchForm_SameBytesAsFormWithout`; `ComputeSumOverBlocks, CutsGiveEqualBytes` compares
  cuts only.

### Same processor is a precondition (A30)

`BufferError::kRunOnMismatch` is gone. A function taking two buffers (or buffers and their working buffers, or a
view and the module's buffers) requires them on one processor: stated as `Requires:` in its description, checked
by an assertion in debug builds, never an error value. No module maps it. A caller-facing boundary that receives
the processor choice from the user (`StepSettings::run_on`, I/O, the CLI, the add-on's command) still reports a
wrong or unavailable processor as an error. Removed tests: Foundation `BuildStableOrder,
ScratchOnOtherProcessor_IsRunOnMismatch`; grid `ApplyGridVersion, ValuesOnOtherProcessor_IsBadInput`,
`CheckSizeStepRule, OnGpuBuffersWithCpuList_IsBadInput`, `ComputeGridCounts, GpuBufferListInOneArgumentForm_IsBadInput`,
`BuildGridPiece, SpanFormOnGpuWhole_IsBadInput`. Narrowed: grid `LentBuilderBuffersTooSmall_IsBadInput` (too small
only), pressure `CoupledVelocityPieces, ShortBuffers_AreBadInput`, step `BuildResizedGrid, NoCpuGridOnGpu_IsBadInput`.
Mappings removed: `ToViscosityFailure`, `ToSolidsFailure`, `ToCellResizeFailure`, the step's and the pressure,
grid, gas, materials, transfer and particles designs' section 4.

### Reconcile items done

- 19: the step's CPU grid-copy pair is `BuildGridMemoryEstimate(RunOn::kCpu, 2, kOneForAllGrids)` (step Task 8).
- 20: `BuilderBuffersSharing` is declared in `staggered_grid.h` (grid Task 6, design 3.3).
- 24: pressure and gas designs' section 3 take `RunOn run_on` in `ComputeBufferBytes`, `BuildPressureMemoryEstimate`,
  `BuildGasMemoryEstimate`; `fluidity_apply_compile_flags(<target> EXCEPTIONS <ON|OFF> [RTTI <ON|OFF>])` recorded
  (pressure decision 20, gas decision 19).
- 25: `ViscosityError::kOutOfMemory` and `SolidsError::kOutOfMemory` (magnitude = bytes asked), returned only by
  `BuildViscosityBuffers` / `BuildSolidsBuffers`; value counts updated.
- 26: render output's per-field-part stable order is bounded by `RenderLimits::max_field_parts`
  (`kTooManyFieldParts` beyond it) and counted by `"render output: frame order buffers"` (+ fixed bytes, `kWholeRun`,
  `kCpu`).
- 27: `BuildTransferBuffers` takes `material_count` (`transfer: material offsets`, 8 B per material).
- 28: the grid's GPU `OrderBuffers` holds `64 * max_cell_blocks + 1` items; the extra item's 6 B were in no entry. The
  fixed-part entry is now `fixed_bytes + bytes_per_item` (builder buffers fixed bytes 4,096 + 6 + 8); the border
  search's `OrderBuffers` gets the same fix.
- 29: one suffix, `_test_support`, everywhere.
- 30: the known-open form is kept as written (owner confirmation pending).

## Owner answers A40–A64 applied (2026-11-15)

Applies A40–A64 as refined: A43 and A45 by A58, A44 replaced by A57, A46 by A59 and A60, A52 by A61–A63, A39 and A51
by A55 (already in the plans). Changed plan tasks carry `**Amended 2026-11-15 (A<n>):**`; inserted tasks use
"12a"-style numbers; later-task groups are headed "(later)" and are not part of any module's done-criteria. Every
design's section 11 has one decision per answer; replaced decisions keep their text marked **replaced**. Reconcile
items 31–33 are applied here too (item 11). Everything below wins over a design and over earlier sections of this
file where they differ; the earlier text it replaces is named in each item.

### 1. Particles everywhere in the liquid (A64)

- **First version.** Every liquid cell holds particles, at every depth and every cell size. There is no grid-only
  region, no grid-only record, no particle/grid velocity blend, no carried (semi-Lagrangian) grid velocity and no
  region transfer. Replaces: cell resize design 1 Part A ("deep liquid ... needs no particles"), 3.4 `RegionSettings`,
  `kMaxFilledShare`, `ReadLargestGridOnlySpeed`, `ApplyGridOnlyRegionInStep`, `MoveGridOnlyLiquidInStep`; 3.5
  (except `ComputeDepthBelowSurface` and `ComputeLiquidTotals`, which stay, the latter over particles only); 5.5,
  5.6; the `ResizeCells` steps that call them; transfer's `FaceValueKind::kFromGridOnlyRegion` uses; the step's
  `apply_grid_only_region`, `move_grid_only_liquid`, `read_largest_grid_only_speed` callbacks,
  `CommittedStepLimits::grid_only_speed`, the grid-only `GridFields` arrays (`is_grid_only_region`,
  `layers_from_grid_only_region`, `grid_only_mass`, `grid_only_rest_volume`, `grid_only_current_volume`,
  `grid_only_ingredient_volume`), the batched-run mirror `"step: committed grid-only marks on the CPU"`, the
  stage-2 mark "is a grid-only cell of the committed grid", pressure's "a grid-only cell counts as full", render
  output's grid-only surface inputs, the I/O and add-on fields `particle_region_depth_in_cell_widths`,
  `grid_only_depth_in_cell_widths`, and the capability findings and errors that exist only for them
  (`kGridOnlyCellDrained`, `kGridOnlyCellTooFull`, `kAmountsDoNotBalance` keeps its meaning over particles only).
- **Where it goes.** Design text is kept, each passage marked "not in the first version (A64)". The tasks move to
  one group at the end of the cell resize and batches plan: **"Deep liquid without particles (later; needs
  review)"**, excluded from the done-criteria, opened by the note "an independent review of this design comes
  first", and gated by two tests named there: `GridOnlyRegion, StandingWave_KeepsAmplitude` (a standing wave of
  wavelength 1 m in a 0.5 m deep tank with the bottom half grid-only keeps its amplitude within 5 % of the
  all-particle run's after 10 periods; 5 % is a starting value) and `GridOnlyRegion, ParticleBandEdge_StaysStable`
  (a tank sloshed for 20 s: no step rejected for travel at the band edge, kinetic energy never above its start value
  plus the work of gravity, no particle stuck in a grid-only cell). Other plans' grid-only pieces (step, transfer,
  pressure, render output, I/O, Blender) move to a group of the same name in their own plan, each pointing at the
  cell resize group as its owner.
- **Particles in big deep cells.** A cell of size level `l` holds particles of `size_level == l` (particles design
  5.6): `AdjustParticleCount` merges eight particles of level `l + 1` into one of level `l` when a resize merges
  their cells, and splits one into eight when a resize splits its cell. So a big cell holds about the same number of
  particles as a small one (the seeded count, 8 per cell with `ComputeRefillParticle`'s pattern), each with 8^k
  times the volume. No new particle code is needed; the cell resize plan gains the deep-tank test below, which
  proves the cascade works over several levels (one level per resize). Particles that drifted across a size change
  keep their level (C2) and are counted, so a deep cell may hold more than 8.
- **Cost, stated honestly.** With the depth term `w = depth / 8` the number of cells below the band in a tank of
  surface area `A` is `∫ A dz / w(z)^3 = 512 A ∫_{8 w_min}^{D} z^-3 dz < 4 A / w_min^2`, against `8 A / w_min^2` in
  the band of 8 smallest cells: particles everywhere costs at most about 1.5 times the band-only count for any
  depth, when the merges are allowed (discrete levels and the 0.8 merge share raise this toward 2). The scenes that
  need many more particles are those where deep cells cannot coarsen: a small `size_level_count`, a strong shear
  term, artist regions, or a solid or gas pocket deep down; there the count grows with depth, `8 x depth / w x A /
  w^2`. `SceneCounts`' particle estimate counts every liquid cell of the start grid at the levels the depth term
  gives, times the seeded count, plus the sources' estimate as before.
- **Does-not-fit line.** When particles are the largest processor entry and the plan does not fit, `BuildBudgetLines`
  adds, before `fits: no`: `particles: <n> needed, <m> of them in cells deeper than <8 w_min> m below the surface;
  in this version every liquid cell holds particles (deep liquid without particles comes later)`. The suggestion
  line is unchanged. I/O test `BuildBudgetLines, DeepTankDoesNotFit_NamesDeepParticles`.
- **Tests that used a grid-only scene** are rewritten on the all-particle scene (same setup, every cell with
  particles) or moved to the later group when they test the region itself; each plan lists which.
- **New tests.** Cell resize `ResizeCells, CalmDeepTank_CoarsensWithParticlesInEveryCell` (2 m deep tank, `w_min`
  = 0.05 m, 5 levels, at rest: after 3 s every cell deeper than `8 w_min` reaches the level the depth term allows,
  every liquid cell holds particles of its own level, the count per cell lies in 1..16 except drifted ones counted
  in `skipped_incomplete_count`, mass and momentum over particles equal before and after each resize within
  `kAmountTolerance`).

### 2. The shared cell size rule (A41, A61, A62, A63; A52 refined)

- **Owner and file.** `native/fluidity/grid/cell_size_rule.h` (S, `FLUIDITY_CPU_AND_GPU`, C++20 subset, no
  allocation), namespace `fluidity::grid`, built and tested in a new grid plan task (numerics). Grid owns it because
  both the step (stage 2) and cell resize read it and both come after grid. `CellSizeSettings` and `CellSizeFacts`
  move from `cell_resize/cell_size_rule.h` into it (cell resize design 3.2 becomes "moved to grid");
  `cell_resize::CellResizeSettings::cell_size` is a `grid::CellSizeSettings`.
- **Declarations** (exact; comments in the grid task):

  ```cpp
  struct CellSizeSettings {                     // Chosen once per scene. Replaces cell_resize::CellSizeSettings.
    SizeLevel smallest_size_level{};            // Width w_min: the surface, pockets, gas and solids.
    double surface_reach_in_cell_widths = 8;    // No unit. Depth term: w_band = w_min * max(1, depth / (8 w_min)).
    double solid_reach_in_cell_widths = 2;      // No unit. Solid term: w_solid = w_min * max(1, d / (2 w_min)).
    double shear_width_factor = 0.25;           // No unit. c_u of literature-cell-size.md 5(a); starting value,
                                                //   to calibrate (A62).
    double merge_width_share = 0.8;             // No unit. Merge only while 2 w <= 0.8 x wanted width.
    double merge_wait_duration = 0.25;          // s of committed time the merge test holds without a break.
    std::uint32_t min_merge_wait_steps = 8;     // Committed steps the merge test holds without a break.
    std::uint32_t max_cell_splits_per_resize = 1u << 16;   // Limit. More wanted splits wait.
  };
  struct CellWidthsByLevel {                    // m. width[l] = ComputeCellWidth(frame, l), computed once on the CPU.
    double width[kMaxSizeLevels];               // Compared, never recomputed with log2 or pow on either processor.
    std::uint8_t size_level_count = 1;
  };
  struct CellSizeFacts {                        // What the rule reads about one active cell.
    double cell_width = 0;                      // m.
    double depth_below_surface = 0;             // m. 0 at the surface; +infinity beyond reach.
    double distance_to_solid = 0;               // m. +infinity: none within reach.
    double velocity_gradient_size = 0;          // 1/s. Held maximum (below); 0 outside liquid.
    SizeLevel artist_size_level{};              // Finest level an artist region or the map asks here; 0: none.
    bool holds_particles_that_keep_their_count = false;
    bool touches_air_pocket = false;
    bool wants_smallest_cells = false;          // Gas in or beside the cell (A10).
    bool touches_air_gas_or_solid = false;      // Merge veto (5.3 rule 5).
  };
  // m/s. U_ref = max(largest_liquid_speed, sqrt(gravity_size * biggest_cell_width)) (literature-cell-size.md 5(a)).
  FLUIDITY_CPU_AND_GPU double ComputeReferenceSpeed(double largest_liquid_speed, double gravity_size,
                                                    double biggest_cell_width);
  // m. wanted = clamp(min(w_band, w_solid, w_shear, w_artist), w_min, w_max); w_shear = c_u U_ref / g for g > 0, else
  // +infinity; w_artist = widths.width[artist_size_level] when above 0, else +infinity. w_min where
  // touches_air_pocket or wants_smallest_cells. No FMA contraction (strict rounding of the shared headers).
  FLUIDITY_CPU_AND_GPU double ComputeWantedCellWidth(const CellSizeSettings& settings,
                                                     const CellWidthsByLevel& widths, double reference_speed,
                                                     const CellSizeFacts& facts);
  // The biggest level (smallest index) whose width <= wanted_width, by comparison with widths.width.
  FLUIDITY_CPU_AND_GPU SizeLevel ComputeWantedSizeLevel(const CellWidthsByLevel& widths, double wanted_width);
  // A cell that is not in the committed grid (stage ChooseActiveCells): the level of the nearest committed active
  // cell (grid::FindCellAt probes at position + {-1,0,1}^3 * biggest width, nearest centre, ties in fixed cell
  // order) plus one, capped at smallest_size_level; smallest_size_level when none is found. This is the rule above
  // for a new cell: it borders air, so its depth is 0 and its wanted width is w_min (A61); the cap keeps the
  // size-step rule against the committed cell it grew from. Reads the tables on their own processor.
  FLUIDITY_CPU_AND_GPU SizeLevel ComputeNewCellSizeLevel(const GridTables& committed, Vec3 position,
                                                         SizeLevel smallest_size_level);
  struct CellSizeRegion {                       // An artist region: world axis-aligned box.
    Vec3 low_corner{}, high_corner{};           // m, world.
    SizeLevel size_level{};                     // The asked cell width rounded at load to the biggest level whose
                                                //   width <= the asked width.
  };
  struct ArtistCellSizes {                      // Views on the run's processor.
    const CellSizeRegion* regions = nullptr;  std::uint32_t region_count = 0;
    const std::uint8_t* map_levels = nullptr;   // [map_height * map_width], row 0 at the domain's low y. 0: none.
    std::uint32_t map_width = 0, map_height = 0;
    Vec3 domain_low_corner{};  double domain_width_x = 0, domain_width_y = 0;   // m.
  };
  // Finest level any region containing `center` or the map pixel under it asks for; 0: none. Pixel column =
  // min(floor((x - x0) / domain_width_x * map_width), map_width - 1), row likewise; outside the box: none.
  FLUIDITY_CPU_AND_GPU SizeLevel ComputeArtistSizeLevel(const ArtistCellSizes& artist, Vec3 center);
  ```

- **Velocity gradient.** `cell_resize/velocity_gradient_math.h` (S): `FLUIDITY_CPU_AND_GPU double
  ComputeVelocityGradientSize(const grid::GridTables& tables, const double* velocity, grid::CellPlaceIndex cell);`
  the Frobenius norm of the full 3x3 gradient at the cell centre from the committed `velocity_after_pressure`:
  diagonal terms from the two opposite faces of the cell (area-weighted mean of their pieces, difference over the
  width), off-diagonal terms from the cell-centre velocities of the two face-touching cells along each other axis
  (a bigger neighbour: its centre value; four smaller ones: their mean in ascending `CellId`), difference over the
  distance between centres. Fixed order, no contraction. Liquid cells only (cells whose `CellContent` is liquid);
  0 elsewhere. A uniform field gives 0; a linear field gives its exact gradient between same-size neighbours, and
  first-order accuracy beside a bigger neighbour, whose centre is off the axis by w/2 (reconcile, cell resize plan
  Task 7a, bound test) (literature-cell-size.md 5(c)).
- **The pass.** `cell_resize::ComputeCellSizeChanges` runs on the grid's processor over all blocks after a committed
  step, inside `ResizeCells` step 2: per active cell it fills `CellSizeFacts` (depth from
  `ComputeDepthBelowSurface`, distance from `pressure::SolidCoverage`, the held gradient, the artist level), calls
  `ComputeWantedCellWidth`, updates the try copies of the wait counters, applies the merge test of design 5.3 rule 5
  plus "never within `surface_reach_in_cell_widths * w_min` of the surface", and writes one byte per cell place,
  `grid::CellValues<std::uint8_t> size_change` (`CellSizeChange::kKeep`, `kSplit`, `kMayMerge`, `kMergeReady`).
  On a GPU grid that byte array is read back in one counted copy (1 B per cell place) for the CPU list edit (split
  and merge on the CPU cell list stay as designed); nothing else is read back for sizing.
- **Hysteresis (literature-cell-size.md 5(d)).** Split at once when the cell is wider than wanted. Merge eight
  siblings only after `2 w <= merge_width_share x wanted` has held for them and every touching cell without a break
  for `merge_wait_duration` of committed time **and** `min_merge_wait_steps` committed steps; never a cell touching
  air, a pocket, gas or a solid, or within `8 w_min` of the surface; at most one level per cell per resize.
- **Held maximum and wait counters (per cell place, committed and try copies in `step::GridFields`, beside
  `merge_ready_duration`):** `grid::CellValues<double> held_velocity_gradient_size` (1/s; 8 B) =
  max(held, current) at each resize, reset to the current value when the merge test breaks or the cell is merged (a split
  copies it, which only delays a later merge: reconcile); `grid::CellValues<std::uint8_t> merge_ready_step_count` (1 B, saturating at 255; reset with
  `merge_ready_duration`). The held value equals "the maximum over the current wait" (A63); it is redundant while
  `ResizeCells` runs after every committed step and is kept so a longer resize interval stays correct. Remap:
  split copies, merge gives 0 (as `merge_ready_duration`). Memory estimate entries (step): `"step: held velocity
  gradient size"` 8 B per cell place x 2, `"step: merge wait steps"` 1 B per cell place x 2, `kRunOn`; cell resize:
  `"cell resize: size changes"` 1 B per cell place, `"cell resize: cell size regions"` 56 B per region (`kGivenCount`),
  `"cell resize: cell size map"` 1 B per pixel (`kGivenCount`), all `kRunOn`, `kWholeRun`. Saved in the state.
- **Reference speed.** `step::CommittedStepLimits` gains `double largest_liquid_speed = 0; // m/s. Largest particle
  speed over liquid materials at the last commit; an exact max, so any order and any batch cut give the same bytes.`
  The step fills it in `CommitStep` from the travel pass it already runs. `ResizeCells` reads it and calls
  `ComputeReferenceSpeed` with `|gravity|` and the frame's biggest cell width.
- **Removed.** `step::StepCallbacks::choose_cell_size`, `cell_resize::ChooseCellSize`, the one-piece copies of step
  design 6.5 row O4 ("new cells back, levels sent") and step Appendix B's O4 row, `ResizeCellsReport`'s
  "ChooseCellSize's first copy", the "Open (owner): a shared size rule" items, batches Task 29's null-tables text.
  Stage 2 calls `grid::ComputeNewCellSizeLevel` in its marking pass on the grid's processor. **Kept:**
  `cell_resize::GridCopies` and the step's batched CPU grid copy, because `FindCellsThatMustSplit`, the list edit,
  `BuildResizeMap`, grouping and the marker upkeep still read CPU tables in a GPU run; only `choose_cell_size` stops
  being a reader.
- **Artist inputs (A63).** Scene file: `cell_size_regions: [{low_corner: [x,y,z], high_corner: [x,y,z],
  cell_width: m}]` and `cell_size_map: {image: <path relative to the scene file, PNG, 8- or 16-bit grey>}` (absent:
  none). The map covers the domain box's x-y extent seen from above (Z-up) and applies at every height; pixel value
  `v` in [0, 1] (white 1): `v == 0` no request, else level `round(v x (size_level_count - 1))`, converted once at
  load on the CPU into `map_levels` and uploaded once. The PNG is a top view with +y up: its top row is the
  domain's high y, so the loader flips it (`map_levels` row 0 is the low y). A region `cell_width` of 0 means the
  smallest cell width. The run folder's `inputs/cell_size_map.png` is left alone by `run --into`. The add-on's
  property is `memory_left_free_mib` (a Blender int is 32-bit); the scene file and `RunDetail` keep bytes. Reason for a 2D map: Blender has a plain way to paint an
  image, none to paint a 3D field; depth-specific requests use regions. Blender: a domain property
  `cell_size_region_collection` (objects in it are regions; each object gets `fluidity_cell_size_region.cell_width`,
  m, default the smallest cell width); the region box is the object's world axis-aligned bounding box (finding
  `cell_size_region_rotated`, NOTE, when the object is rotated: "the axis-aligned box around it is used"); a domain
  property `cell_size_map_image` (an Image); the exporter writes it as an 8-bit grey PNG `inputs/cell_size_map.png`
  in the run folder (luminance). Panel: Physical Detail, "Cell size" box. The fields are part of the scene
  fingerprint.
- **Same bytes.** Every term is a local stencil read plus exact min and max; widths are compared with
  `CellWidthsByLevel`; no `log2`, no `pow`, no FMA. So CPU and GPU give equal `size_change` bytes, and any batch cut
  gives equal bytes. Tests: grid `ComputeWantedCellWidth, OnGpu_SameBytesAsCpu` (10,000 fact sets from a fixed
  formula), cell resize `ComputeCellSizeChanges, OnGpu_SameBytesAsCpu` (shear scene), `ComputeCellSizeChanges,
  AnyBatchCut_SameBytes` (one range, 1-block and 3-block ranges).
- **Tests** (cell resize unless named): from literature-cell-size.md 5(f): (1) `ResizeCells,
  SplitThenMerge_ConservesMassAndMomentum` (bytes for the record-free totals, `kAmountTolerance` relative);
  (2) `ResizeCells, CalmPool_ReachesBiggestCellsThenStays` (flat pool 1 m deep, `w_min` 0.02 m, 5 levels: every
  cell deeper than `8 w_min` reaches the depth term's level within `0.25 s x 4` levels plus 32 steps, then no split
  or merge for 10 s); (3) `ResizeCells, ThinSheetAndPinchOff_StaySmallest` (no merge within `8 w_min` of either
  face); (4) grid `ComputeWantedCellWidth, UniformTranslation_NoShearSplit` and `ComputeCellSizeChanges,
  RigidRotation_Splits` (gradient size `sqrt(2) Ω`; with Ω = 10 rad/s, U_ref = 1 m/s: w_shear = 0.25 / 14.142 =
  0.017678 m); (5) the batch-cut test above; (6) the CPU-GPU test above; (7) `ResizeCells,
  SplashIntoCoarseRegion_NoStuckParticles` (no particle whose speed stays below 1e-3 of the splash speed for 20 steps
  while above the surface; liquid momentum continuous across the resize within 1e-12 relative). Owner's additions:
  `ResizeCells, CalmTank_InsideCoarsensSurfaceStaysFinest`, `ResizeCells, Jet_KeepsSmallCellsAlongShearLayer` (a
  0.1 m jet at 2 m/s into still liquid: cells within 2 jet radii of the jet edge stay at `w_min` for every resize
  after 0.5 s), grid `ComputeArtistSizeLevel, RegionAndMap_FinestWins`, `ComputeNewCellSizeLevel,
  NextToBigCell_OneLevelFiner`, and A61's `ComputeWantedCellWidth, SurfaceCell_IsSmallest` (depth 0 gives `w_min`
  whatever the other terms).
- **Later group "Coarser cells on calm surfaces (later)"** in the cell resize plan holds A61's curvature work:
  `w_curv = c_k / max(|∇²φ|, tiny)` with `c_k = 0.25`, curvature from a smoothed distance (not
  `distance_to_surface`), 7-point Laplacian within 2 cells of the surface clamped to `1 / w_min`, spread by a max
  over 26 neighbours twice, surface cells up to 1 or 2 levels coarser, a T-junction-aware surface pressure condition
  (Ando and Batty 2020 Eq. 25) and bump-free surface resampling (Ando 2013). Not in the done-criteria.

### 3. GPU memory: one reserved block, a true budget, warnings (A58; A43, A45 refined)

- **Budget** (both processors; replaces `memory_share_of_total` 0.7 / `memory_share_of_free` 0.8 of step design 3.3
  and step 6.3 item 4): `allowed = min(memory_share_of_free x free, free - memory_left_free_bytes)` (0 when negative),
  with `StepSettings::memory_share_of_free = kDefaultMemoryShareOfFree` (0.9; Redshift's documented default) and
  `StepSettings::memory_left_free_bytes = kDefaultMemoryLeftFreeBytes` (536,870,912 B = 512 MiB; reasoning: about
  twice Redshift's 256 MB minimum; starting value). `memory_share_of_total` is removed. One rule for the CPU too
  (it was min(0.7 total, 0.8 free)); owner to note. Declared in step `run_plan.h`:
  `std::uint64_t ComputeAllowedMemoryBytes(MemoryAvailable available, double memory_share_of_free, std::uint64_t
  memory_left_free_bytes);`. On the GPU `free` is read by `ReadMemoryAvailable(RunOn::kGpu)` after the context
  exists (Foundation Task 19). The suggestion `more_memory_bytes` becomes `max(ceil(needed /
  memory_share_of_free), needed + memory_left_free_bytes) - free`. The grid's 25 % headroom stays (A37). The add-on's
  `memory_share_of_total` property is removed, `memory_share_of_free` defaults to 0.9, `memory_left_free_bytes`
  (MiB in the UI) is added.
- **One reserved block** (Foundation Tasks 17–19, `base/run_memory.h`). Names: `RunMemory` (move-only; one block on
  one processor), `BuildRunMemory(RunOn run_on, std::size_t byte_count, CpuMemory cpu_memory =
  CpuMemory::kOrdinary)`, methods `run_on()`, `reserved_bytes()`, `handed_out_bytes()`; `BuildBuffer<Value>(RunOn
  run_on, std::size_t size, RunMemory& run_memory)`; `kRunMemoryPieceAlignmentBytes = 256` (CUDA's documented
  `cudaMalloc` alignment; the same on the CPU so a plan's bytes do not depend on the processor);
  `std::size_t ComputeRunMemoryPieceBytes(std::size_t byte_count)` (rounded up to 256; 0 stays 0);
  `std::uint64_t ComputeRunMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts)`
  (in `memory_count.h`: over entries with `where == kRunOn`, `copies x ComputeRunMemoryPieceBytes(item count x
  bytes_each)`, saturating); `BufferError::kRunMemoryTooSmall` appended at the end of `BufferError`; forms
  `BuildSumBuffers(RunOn, std::size_t max_count, RunMemory&)` and `BuildOrderBuffers(RunOn, std::size_t max_count,
  RunMemory&)`. Rules: a piece is never freed alone; a `Buffer` from a `RunMemory` frees nothing when destroyed;
  the `RunMemory` frees the block as a whole and must outlive its pieces (debug assertion on a live-piece count held
  in a heap cell the `RunMemory` owns, so moving it keeps the pieces' pointer valid). Swaps of try and committed
  buffers, `grid::TakeBuffers` and rebuilding a grid into its storage move `Buffer` objects and free nothing, so they
  fit unchanged. A piece that does not fit returns `kRunMemoryTooSmall`, takes nothing and writes nothing; it is a
  planner bug (debug builds also assert). Plain `BuildBuffer` (no `RunMemory`) stays for tests, tools, `where =
  kCpu` buffers and `KeptFor::kOneCall` / `kOneSnapshot` CPU arrays. No `kRunOn` entry may be `kOneCall`; a plan
  that has one makes it `kWholeRun`.
- **Every module builder** (`Build<Module>Buffers`, `BuildGridBuffers`, `BuildGridBuilderBuffers`,
  `BuildBorderSearchBuffers`, `Build...Values`, `BuildCellResize`, `BuildRenderOutput`'s run-processor part, and
  every other function that builds `kRunOn` buffers) takes **`RunMemory& run_memory` as its last parameter**; every
  existing parameter stays; `run_on` must equal `run_memory.run_on()` (`Requires:`, debug assertion, A30).
  **Correction (same day, reconcile):** a run has one `RunMemory` per processor it uses. `run_memory` lives on the
  run's processor and holds the `where = kRunOn` entries; in a GPU run `cpu_run_memory` (CPU; `CpuMemory::
  kLockedForCopies` in a batched GPU run, ordinary otherwise) holds the `where = kCpu` entries kept `kWholeRun` or
  `kOneStepTry` (stage 17's CPU `SolidsBuffers`, the step's CPU grid copy and mirrors, cell resize's CPU grids, gas
  marker lists). In a CPU run both kinds come from the one `run_memory`, and `cpu_run_memory` is that same object.
  A builder takes the `RunMemory` of the processor its buffers live on (a builder called with `run_on = kCpu` in a GPU
  run gets `cpu_run_memory` as its `run_memory`); a builder that builds on both processors takes `RunMemory&
  run_memory, RunMemory& cpu_run_memory` as its last two parameters. `ComputeRunMemoryBytes` gains a `MemoryWhere
  where` parameter (sums the entries with that `where`); in a CPU run the step sums both. Plain `BuildBuffer` is left
  for tests, tools and `kOneCall` / `kOneSnapshot` CPU arrays. `MemoryReport::allocated_cpu_bytes` becomes
  `handed_out_cpu_bytes` with `reserved_cpu_bytes` beside it; the CPU warning compares them with the plan the same
  way. Each module's estimate test becomes: build a `RunMemory` of exactly
  `ComputeRunMemoryBytes(Build<Module>MemoryEstimate(...), counts)`, build the module's buffers into it, assert
  `handed_out_bytes() == reserved_bytes()` (test name `Build<Module>Buffers, IntoRunMemory_HandsOutExactlyTheEstimate`).
  An entry's `copies` must equal the number of buffers it stands for, each `count x bytes_each` bytes. A buffer
  whose size is a per-item part plus a fixed part (working buffers from `Compute<Name>BuffersSize`, orders with one
  extra item) is **one** entry: `counted_per = kGivenCount`, `bytes_each = 1`, `given_count` = its exact byte count at
  the plan's limits, so the reservation is exact for any `max_cell_blocks` (reconcile, grid agent). So the grid's
  estimate builders take the room: `BuildGridMemoryEstimate(RunOn run_on, std::uint32_t storage_count, std::uint32_t
  max_cell_blocks, BuilderBuffersSharing builder_buffers_sharing = kOnePerGrid)` and
  `BuildBorderSearchMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks)`; callers pass
  `limits.max_cell_blocks` (for example `BuildGridMemoryEstimate(run_on, 2, limits.max_cell_blocks, kOneForAllGrids)`).
  `kRunMemoryTooSmall` maps to the module's `kOutOfMemory` (item 11) with `magnitude` = bytes asked.
- **Step.** `BuildSimulation` reads `ReadMemoryAvailable` (free before), builds `run_memory` from
  `ComputeRunMemoryBytes(estimate)` **before anything else**, reads free again, then builds every buffer into it.
  `ApplyLimits` builds a new `RunMemory` for the whole new plan, rebuilds every buffer into it, copies the committed
  values, swaps, and frees the old block whole; its check becomes `new reserved + old reserved <= allowed_bytes`
  (tighter than before, because the old try copies cannot be freed alone; stated in step design 5.7).
  `MemoryReport` gains `std::uint64_t planned_bytes` (= `ComputeRunMemoryBytes` of the plan), `reserved_bytes`
  (current block), `handed_out_bytes`, `std::uint32_t reserved_block_count` (blocks reserved since
  `BuildSimulation`, 1 + raises), `std::uint64_t memory_taken_bytes` (free before minus free after the reservation;
  GPU only, 0 on the CPU), `std::uint64_t requested_beyond_plan_bytes` (bytes reserved after a limit raise beyond the
  plan shown at start, plus measured `kCpu` bytes beyond their plan). `allocated_bytes` becomes `handed_out_bytes`.
  `estimate.on_processor` is `ComputeRunMemoryBytes` (rounded).
- **Warnings** (I/O `run_cost.h`; `kMemoryWarningShare` and `kCopyWarningShare` removed; A43): `warning: <p>
  memory: <b> requested beyond the planned <b>` for any `requested_beyond_plan_bytes > 0` or measured peak above its
  planned budget on either processor; `warning: copies per step <b> (max) are above the planned <b>` when any
  committed step's copy bytes exceed `planned_step_copies` (max, not mean: the plan is an upper bound); the same for
  solve copies per iteration; `warning: gpu memory taken <b> is less than the <b> reserved; Windows may have placed
  part of it in shared system memory, which is much slower. To make the run fail instead, set "CUDA - Sysmem
  Fallback Policy" to "Prefer No Sysmem Fallback" for fluidity.exe and blender.exe in the NVIDIA Control Panel.`
  when `memory_taken_bytes + kMemoryTakenToleranceBytes < reserved_bytes` (`kMemoryTakenToleranceBytes = 2,097,152`,
  2 MiB, reasoning, to be confirmed by the Windows run; never warn when taken is above reserved: other programs
  count there). Cost lines gain `gpu memory: planned <b>, reserved <b> in <n> block(s), handed out <b>, taken <b>`.
  JSON `RunCost` gains `planned_bytes`, `reserved_bytes`, `reserved_block_count`, `handed_out_bytes`,
  `memory_taken_bytes`, `requested_beyond_plan_bytes`; `RunBudget`'s `MemoryBudget` gains `memory_share_of_free`
  and `memory_left_free_bytes`. On Windows `check` and the add-on print the fallback note once in the budget lines:
  `note: on Windows a full GPU makes the run slow instead of failing; see "Prefer No Sysmem Fallback" in the NVIDIA
  Control Panel`. Blender: the warnings arrive in the cost lines drawn in `FLUIDITY_PT_performance` (no new
  widget). Tests: I/O `BuildCostLines, RequestedBeyondPlan_Warns`, `BuildCostLines, CopiesAboveBound_Warns`,
  `BuildCostLines, MemoryTakenBelowReserved_WarnsSpill`, `BuildCostLines, MemoryTakenAboveReserved_NoWarning`;
  step `ComputeAllowedMemoryBytes, SmallerOfShareAndHeadroom` (free 16e9: 14.4e9; free 4e9: 3.463e9 = 4e9 - 536,870,912;
  free 0.4e9: 0), `BuildSimulation, ReservesOneBlockFirst`, `BuildSimulation, HandedOutEqualsPlanned`,
  `AdvanceTo, LimitRaise_ReservesSecondBlockAndFreesFirst`.

- **Buffers held outside the step after a raise (reconcile).** `step::StepCallbacks` gains
  `Callback<bool(void* context, Simulation& simulation, const SimulationLimits& new_limits)> limits_raised;`, called
  by `ApplyLimits` after the new blocks are reserved and the step's buffers rebuilt, before the old blocks are freed;
  I/O rebuilds `CellResize`, the `BatchedRun` and the render `StepRecorder` into `step::FindRunMemory(Simulation&)` /
  `step::FindCpuRunMemory(Simulation&)` (same object in a CPU run); false gives `kLimitsCannotBeRaised` with nothing
  changed. `BuildSimulation` calls no callback. `BuildRunPlan` gains a last parameter `std::optional<RunMode>
  forced_mode = std::nullopt` (resume, item 10).

### 4. Viscosity advice (A57; replaces A44 and A25's threshold)

- Removed: `kSmallestVisibleViscousEffectPerStep` and its tests (`kSmallestVisibleViscousEffectPerStep,
  IsTenthOfAPercentPerStep`), the A25 "note" line. Kept: A25's definition (the setting decides the path) and
  `ComputeViscousEffectPerStep` (`N = ν Δt / h²`, ν = viscosity / density).
- New, viscosity `viscosity_advice.h` (CPU only): `inline constexpr double kNegligibleViscosityRatio = 0.1;`
  `inline constexpr double kSmallestVisibleWavelengthInCells = 4;` `inline constexpr double kNegligibleViscousChange
  = 0.01;` (all judgments, literature-practice.md topic 1); `std::optional<double>
  ComputeViscosityToSmoothingRatio(double viscous_effect_per_step, double own_motion_kept);` = `4 N / (1 - α)`,
  empty when α = 1 (no smoothing figure exists for pure FLIP); `double ComputeFourCellFeatureChange(double
  viscosity, double density, double run_duration, double cell_width);` = `-expm1(-π² D / 4)` with `D = ν T / h²`
  (a wavelength of 4 cells has angular wavenumber `π / (2h)`, so it decays as `exp(-ν π² T / (4 h²))`);
  `double ComputeReynoldsNumber(double viscosity, double density, double smallest_domain_extent, double
  gravity_size);` = `U L / ν`, `U = sqrt(2 g L)`. Advice when the ratio is below 0.1 or the change is below 0.01.
- **The smoothing formula.** `ν_num = (1 - α) h² / (4 Δt)` is this project's derivation from the XPIC filter
  analysis (Hammerquist and Nairn 2017: a quadratic B-spline round trip filters by `sinc^6(k h) ≈ 1 - (κh)²/4`,
  matched to one diffusion step), not a published figure. The method paper says so in those words; the ratio uses
  the largest step, so it is a lower bound (the cautious side).
- **Planner.** `step::LiquidViscousEffect` (run_plan.h) becomes `{ ingredient_index; viscosity; density;
  viscous_effect_per_step; std::optional<double> viscosity_to_smoothing_ratio; double four_cell_feature_change;
  double spreading_in_cells (sqrt(ν T) / h); double reynolds_number; bool advises_zero_viscosity; }` at the smallest
  cell h, `StepSettings::max_step_duration`, the liquid material's `own_motion_kept` and `T = ReadyScene::end_time`;
  `RunPlan` gains `double viscous_effect_run_duration` (s) and `double reynolds_length` (m, the domain box's smallest
  extent).
- **Lines** (I/O, replacing the A25 per-liquid lines; `%.3g` numbers, percentages rounded to integers):
  `liquids: viscous solve (a liquid has viscosity above 0); at the smallest cell <w> m, the largest step <t> s and
  the run's <T> s` then per viscous liquid `  <name>: viscosity <v> Pa s, density <d> kg/m3, grid number <N>;
  viscosity / transfer smoothing <r> (or "none: the liquid keeps all its own motion"); a feature 4 cells wide changes
  by <c> % over the run (<s> cells of spreading); Reynolds number <Re>`, then when advised `  advice: <name>'s
  viscosity will not be visible: set it to 0`, then for every viscous liquid `  note: at viscosity 0 <name> slips
  along solids instead of sticking`. Exact test (I/O `BuildBudgetLines, WaterAndSyrup_ViscosityAdviceLines`): water
  1e-3 Pa s 1000 kg/m3 and syrup 10 Pa s 1400 kg/m3, α = 0.95 for both, h = 0.01 m, Δt = 0.01 s, T = 10 s, L = 1 m,
  g = 9.81: water N 1e-4, ratio 0.008, change 22 % (0.2186563), spreading 0.316 cells, Re 4.43e+06, advised;
  syrup N 0.714, ratio 57.1, change 100 %, spreading 26.7 cells, Re 620, not advised. Viscosity tests
  `ComputeViscosityToSmoothingRatio, WaterAtOneCentimetre_Is0p008`, `..., KeepsAllOwnMotion_IsEmpty`,
  `ComputeFourCellFeatureChange, Water_Is0p2187` (1e-12 relative), `ComputeFourCellFeatureChange,
  ThresholdAtDOf4p073e-3` (`D = 4 ln(1/0.99) / π² = 4.0732477e-3` gives 0.01), `ComputeReynoldsNumber,
  WaterInOneMetreBox_Is4p43e6`. JSON `ViscousEffectOfLiquid` replaces `effect_not_visible` by
  `viscosity_to_smoothing_ratio` (absent when none), `four_cell_feature_change`, `spreading_in_cells`,
  `reynolds_number`, `advises_zero_viscosity`; `RunBudget` gains `viscous_effect_run_seconds`, `reynolds_length_m`.
  Blender: lines verbatim in `FLUIDITY_PT_performance`; `RunBudgetFacts.liquids` fields follow.

### 5. Guided rigid bodies in general form (A59; A46 first half refined)

- **Settings** (solids `solid_settings.h`; `FreeMotionLimits` extended, its axis flags kept):
  `enum class GuideKind : std::uint8_t { kAxisLimits, kRail, kHinge };` and in `FreeMotionLimits`: `GuideKind
  guide_kind = GuideKind::kAxisLimits;` `Vec3 rail_direction{0, 0, 1};` (no unit, world, normalised at load; kRail:
  slides along it, no turning) `Vec3 hinge_point{};` (m, world, at time 0) `Vec3 hinge_axis{0, 0, 1};` (no unit,
  world). **World frame**, decided: for one hinge with a fixed pivot the axis is unchanged by rotation about itself,
  so world and body frames agree for the axis, and the pivot is fixed in the world; a rail with no turning keeps its
  direction. Body-fixed axes and moving rails are not exposed (the internal prescribed motion stays 0).
- **Internal form** (solids, shared `guide_math.h`): `struct AllowedMotions { double columns[6][6]; std::uint8_t
  column_count = 6; double prescribed_motion[6] = {}; };` rows `(V, Ω)` in world axes about `origin_at_start`; built
  each try by `FLUIDITY_CPU_AND_GPU AllowedMotions ComputeAllowedMotions(const FreeMotionLimits& limits, Vec3
  origin_at_start);`: axis limits give selection columns (0/1, exactly); a rail `J = [d; 0]`; a hinge `J = [a x (o -
  p); a]` with `o = origin_at_start`. `FreeSolidForViscousSolve` and the pressure solve's free-solid data carry
  `AllowedMotions` instead of the flags. Motion `ξ = J q + ξ_0`.
- **Viscous solve** (A27 block): per solid the `m x m` block `Jᵀ (M + Δt Rᵀ E_sᵀ W E_s R) J` (Cholesky) and
  per-sample coefficients `R J` (Batty 2007, Aanjaneya 2018; literature-practice.md topic 3). Replaces "a motion a
  guide forbids keeps its value before pressure after the solve" (solids Task 23c): the solve itself never moves a
  forbidden direction. **Pressure solve** free-solid term: the rank-6 term `K M⁻¹ Kᵀ` becomes `(K J) (Jᵀ M J)⁻¹ (K
  J)ᵀ` (rank m), with `ξ_0` in the right-hand side; solid update `q = (Jᵀ M J)⁻¹ Jᵀ (M ξ_before + Δt F_pressure +
  Δt F_external)`, `ξ = J q + ξ_0`.
- **Hinge position correction.** At commit a hinged body's pose is rotated about `hinge_point` by `ω Δt` about
  `hinge_axis` (not translated by `V Δt`), so the pivot distance is kept to rounding; `kRail` bodies are moved
  along `d` only. Solids design 5 gains the step.
- **Scene and add-on.** Scene file `free_solid.guide: {kind: "axis_limits" | "rail" | "hinge", moves_along_x, ...,
  turns, rail_direction, hinge_point, hinge_axis}`. Blender: the free-solid panel gains a "Guide" enum (Axes, Rail,
  Hinge); Rail reads the direction from the local Z axis of a chosen empty (`guide_object`), Hinge reads the point
  and axis from the same empty's world location and local Z axis at frame start.
- **Tests** (solids): `ComputeAllowedMotions, AxisLimits_GiveSelectionColumns` (exact);
  `FreeSolid, TiltedRail_SlidesWithGSinTilt` (rail at 30° from horizontal, no liquid, Δt = 1e-3 s, 1000 steps:
  speed along the rail `n Δt g sin 30°` = 4.905 m/s within 1e-12 relative; distance off the rail below 1e-12 m);
  `FreeSolid, OffOriginHinge_PendulumPeriod` (box 0.1 m, 1 kg, hinge 0.5 m above its centre, 5° start, no liquid,
  Δt = 1e-3 s: period `2π sqrt((I_cm + m l²) / (m g l))` = 1.42322 s (small angle; the 5° correction is 0.05 %)
  within 0.5 %; pivot distance drift below 1e-9 m over 10 periods); `FreeSolid, AxisAlignedGuides_Unchanged` (the
  existing guided-box tests keep their tolerances); viscosity `SolveViscosityAndPressure, HingedPlank_EqualsProjectedFullBlock`
  (plank hinged at its centre in a mirror-symmetric scene, solve tolerance; reconcile: with an off-centre hinge the
  added mass couples the turn to the forbidden motions at first order, so the literal small-step claim is false).

### 6. Force of the liquid on every solid (A60; A46 second half refined)

- `solids::LiquidForceOnSolid { std::uint32_t solid_id; Vec3 pressure_force; Vec3 pressure_torque; Vec3
  viscous_force; Vec3 viscous_torque; }` (N, N m; mean over the step = impulse / step duration; world axes; torque
  about the body's origin at the start of the step; `T_p = T_o + (o - p) x F` moves it). Every solid of the scene,
  ascending `solid_id`, fixed and domain box included; viscous parts 0 for an inviscid liquid.
  `SolidsTry::liquid_forces()` returns them; `step::StepReport` gains `std::vector<solids::LiquidForceOnSolid>
  liquid_force_on_solids;` (filled from the committed try; StepReport is a CPU result, so the vector is allowed).
- **From the solve's own terms.** Pressure part `Δt Σ_e (force_per_pressure, torque_per_pressure)_e p_e` over the
  motion map entries of the solid (free solids already; driven and fixed solids by the same reduction over their
  entries after the solve). Viscous part `-Δt Rᵀ E_sᵀ W (E x + g)` over the solid's samples: the viscosity samples
  keep, for every solid sample of any motion kind, the solid index (4 B) and its six coefficients R (48 B), memory
  entry `"viscosity: solid sample coefficients"` 52 B per solid sample, `kRunOn`; `ViscosityResult` returns the
  per-solid viscous impulse. Both are one partial per cell block then `math::TreeSum` in block order (fixed order).
  Guided bodies report all six components (the guide's reaction included). Invariant test: for a free unguided
  solid `pressure + viscous impulse == M (ξ_after - ξ_before) - Δt (M g + steady force)` within the solve tolerance.
- **Four tests** (solids/viscosity scene tests; values derived from literature-practice.md topic 4):
  `LiquidForce, ShearedPlate_IsMuAUOverH` (μ = 10 Pa s, ρ = 1000, gap H = 0.1 m, driven plate 0.2 x 0.2 m at U =
  0.01 m/s, sides periodic or free-slip, 5 s (`exp(-π² ν t / H²)` = exp(-49)): viscous force 0.04 N, pressure part
  0 in x, within 1e-6 relative since the linear profile is exact); `LiquidForce, BuoyancyAtRest_IsRhoGV` (fixed box
  0.1 m cube in water at rest: pressure force z 9.81 N within 2 %, viscous 0 exactly at rest);
  `LiquidForce, RotatingCylinders_TorquePerLength` (known-open until contact has a cylinder shape; owner; R1 = 0.05 m driven at 1 rad/s, R2 = 0.1 m fixed, μ = 1 Pa s,
  height 0.1 m with free-slip ends: torque 4π μ Ω R1² R2² / (R2² - R1²) x 0.1 = 4.18879e-3 N m within 10 % at 10
  cells across the gap: a bound, per Uhlmann's 5–10 %); `LiquidForce, SettlingCube_BalancesNetWeight` (reconcile: contact has no sphere; a cube of the volume of a
  sphere of R = 0.01 m,
  1100 kg/m3 in 1000 kg/m3, μ = 10 Pa s, box 0.2 m: at steady speed pressure + viscous force = net weight 4.1092e-3
  N within 2 % (a balance); speed between `U_Stokes / 1.4` and `U_Stokes` = 2.18e-3 m/s (a bound: walls slow it)).
- I/O: `frame_done` and `run_report.json` gain `liquid_force_on_solids: [{solid_id, pressure_force_n,
  pressure_torque_n_m, viscous_force_n, viscous_torque_n_m}]` (the last committed step of the frame); the solid
  motion output file is unchanged. Blender: one line per selected solid in the solid panel, "Liquid force: <|F|> N
  (pressure <p>, viscous <v>)", from the last `frame_done` (cheap: no new data path).

### 7. Stick and slip on prescribed-motion solids (A53)

- `solids::SolidSettings` gains `double slip_length = 0;` (m; 0 full stick, the default; `+infinity` free slip;
  read for fixed and driven solids and the domain box beside a viscous liquid; free solids always stick, A27, and a
  value there gives the `check` NOTE `slip_ignored_on_free_solid`). Form chosen: a **Navier slip length**, entered as
  a wall displaced by λ: in the viscous solve's stick-at-the-true-distance term the distance `d` becomes `d + λ`, so
  the tangential coefficient `μ / d` becomes `μ / (d + λ)` (0 at λ = ∞). It is a positive boundary energy term, so
  the system stays symmetric positive definite; it is the standard Navier condition `τ = (μ / λ) u_slip`, which the
  variational viscosity formulations admit as a boundary term. An inviscid liquid keeps the contact rule (friction).
  Scene file per solid: `slip_length` (number, m) and `free_slip` (bool; true means λ = +infinity and overrides the
  number), two fields because a schema field has one type (reconcile, I/O); Blender: solid panel "Slip length" (m) with a "Free
  slip" checkbox, shown when the domain has a viscous liquid.
- Tests (viscosity): `SolveViscosityAndPressure, CouetteFullStick_LinearProfile` (bottom fixed, top driven at U,
  `u = U y / H`, 1e-6 relative); `CouetteFreeSlip_PlugFlow` (both plates free slip, liquid starting at 0.01 m/s: stays
  0.01 m/s everywhere, top plate's viscous force 0 within 1e-12 N); `CouettePartialSlip_NavierProfile` (top plate slip
  length λ = H, bottom stick: `u = U y / (H + λ)`, `u(H) = U / 2`, 1e-6 relative).

### 8. Wind (A47, A50)

- Wind acts on gas in the solver (unchanged) and on spray and foam in render output; never on the liquid body.
  The add-on writes wind keys whenever a Wind field exists (`build_wind_keys(readings)`, `has_gas` dropped); the
  scene field `gas_wind` is renamed `wind` (same keys `{time, acceleration}`, m/s2); `ReadyScene` keeps
  `gas::GasWind wind`; `render_output::RenderSettings` gains `gas::GasWind wind;` filled by I/O from the ready scene.
- Laws (simplest honest; no drag model): spray particles get the same acceleration as gas, `v += Δt (g +
  a_wind(t + Δt / 2))`, no new constant. Foam moves with the surface velocity plus a drift `kFoamWindDriftDuration x
  (a_wind - (a_wind · z) z)` with `kFoamWindDriftDuration = 0.1 s` (render output; starting value, a judgment: with
  Blender Strength 1 the foam drifts 0.1 m/s). Tests (render output): `MakeFoamAndSpray, SprayInWind_AcceleratesByWind`
  (exact per step), `MakeFoamAndSpray, FoamInWind_DriftsHorizontally` (vertical wind gives no drift).
- Findings: `wind_without_gas` (NOTE) text: "The Wind field moves spray and foam only; the liquid is not pushed by
  wind." (only in a scene without gas; with neither spray nor foam: "The Wind field has no effect in this scene.");
  `wind_flow_not_honoured` only when `field.flow != 1.0` (A50).

### 9. Blender versions and the CUDA runtime (A48, A49)

- Manifest `blender_version_min = "5.3.0"`, no maximum; when `bpy.app.version > (5, 3, x)` the add-on shows one NOTE
  in the domain panel: "FluidiTy was tested with Blender 5.3 only." (test `test_newer_blender_shows_one_note`).
- The executable is linked to the shared CUDA runtime (Foundation's `CMAKE_CUDA_RUNTIME_LIBRARY Shared`). The
  package ships, in `bin/<platform>/` beside `fluidity[.exe]`, the runtime file of the toolkit it was built with:
  Windows `cudart64_<major>.dll`, Linux `libcudart.so.<major>` (the real file under its SONAME, not a link), taken
  from `CUDA::cudart`'s imported location by the packager; macOS ships none. Loader: Windows searches the
  application's directory first; Linux executables are linked with `RUNPATH $ORIGIN`. The package also ships
  `licenses/cuda-runtime.txt`: "Contains the NVIDIA CUDA Runtime <version>, redistributed under the NVIDIA CUDA
  Toolkit End User License Agreement, Attachment A (redistributable software)" with the EULA link (owner to confirm
  the wording). CPU runs never depend on a GPU library: the runtime ships with the executable, and the only GPU
  library not shipped, the driver (`nvcuda.dll`, `libcuda.so.1`), is loaded by the runtime at the first CUDA call,
  which `CheckCanRunOn(RunOn::kGpu)` makes; without a driver it returns an error and CPU runs go on. Tests:
  `test_package_contains_cuda_runtime` (windows-x64, linux-x64: the file is present and its version equals
  `ReadProcessorDescription(kGpu).runtime` of the build), `test_linux_executable_runpath_is_origin` (`readelf -d`),
  `test_packaged_executable_runs_cpu_scene_without_driver` (in the container without a GPU driver).

### 10. Resume (A40)

- `resume` reuses the recorded `run_mode` (one piece, batched, batched with solves in pieces): it plans with the
  mode forced. If the forced mode does not fit the memory of the day it refuses with `IoError::kRecordedModeDoesNotFit`
  (appended at the end of `IoError`; code `recorded_mode_does_not_fit`), message: "This run was recorded in <mode>,
  which no longer fits in memory (<needed> needed, <allowed> allowed). Resume with --choose-mode-again to continue in
  another mode; the results then differ slightly from an uninterrupted run (within the solver's tolerance)." The flag
  `--choose-mode-again` (`Command::choose_mode_again`) plans the mode afresh as `run` does. The run record gains
  `run_mode_changes: [{frame, from, to}]`, written when a resume changes mode; `kWrongRunFingerprint` with `kRunMode`
  is no longer raised for a mode change made with the flag. Blender: on that error the run panel offers "Resume in
  another mode", which passes the flag. Tests (I/O): `Resume, RecordedModeStillFits_ReusesIt`, `Resume,
  RecordedModeDoesNotFit_RefusesNamingTheFlag`, `Resume, ChooseModeAgain_RecordsTheChange`; Blender
  `test_resume_offers_other_mode_on_recorded_mode_does_not_fit`.

### 11. Reconcile items 31–33

- **31.** Every module error enum has `kOutOfMemory`, **appended as the last value** (a module that gains it keeps
  every earlier value's number), returned only by that module's building functions; it is also what
  `BufferError::kRunMemoryTooSmall` maps to. Viscosity and solids move theirs to the end: the values after its old
  place shift down by one, accepted because no code exists yet; their enum tests and `ToName` tables follow.
  Pressure, gas and transfer gain it. Each design's value count
  and each plan's enum test follow.
- **32.** A span (CPU) form called with a GPU grid, or a CPU list with GPU buffers, is a `Requires:` precondition
  with a debug assertion, not `kBadInput` (`FindBorderBlocks`, `BuildGridPiece`, the grid checks).
- **33.** Accepted as written: test-only `BuildTestGrid` replaces the allocating `BuildGrid(inputs, settings)`; span
  forms of `ComputeGridCounts` / `CheckSizeStepRule` kept; `ComputeAmounts`, `ComputeWorkRates`,
  `CheckMaterialState` keep one allocating form each; `kBadStepInputs`, `step_inputs`; `max_field_parts = 1u << 26`
  with one frame `OrderBuffers`. Sand Task 7's per-call GPU copy is settled in the sand module.

### New public names (A40–A64 pass)

Foundation: `RunMemory`, `BuildRunMemory`, `BuildBuffer(RunOn, size, RunMemory&)`, `ComputeRunMemoryBytes`,
`ComputeRunMemoryPieceBytes`, `kRunMemoryPieceAlignmentBytes`, `BufferError::kRunMemoryTooSmall`, the `RunMemory&`
forms of `BuildSumBuffers` and `BuildOrderBuffers`. Grid: `CellSizeSettings` (moved), `CellSizeFacts` (moved),
`CellWidthsByLevel`, `CellSizeRegion`, `ArtistCellSizes`, `ComputeReferenceSpeed`, `ComputeWantedCellWidth` (moved),
`ComputeWantedSizeLevel`, `ComputeNewCellSizeLevel`, `ComputeArtistSizeLevel`. Cell resize:
`ComputeVelocityGradientSize`, `ComputeCellSizeChanges`, `CellSizeChange::kMergeReady`. Step:
`kDefaultMemoryShareOfFree`, `kDefaultMemoryLeftFreeBytes`, `StepSettings::memory_left_free_bytes`,
`ComputeAllowedMemoryBytes`, `CommittedStepLimits::largest_liquid_speed`, `GridFields::held_velocity_gradient_size`,
`GridFields::merge_ready_step_count`, `MemoryReport::{planned_bytes, reserved_bytes, handed_out_bytes,
reserved_block_count, memory_taken_bytes, requested_beyond_plan_bytes}`, `StepReport::liquid_force_on_solids`,
`RunPlan::{viscous_effect_run_duration, reynolds_length}`. Viscosity: `kNegligibleViscosityRatio`,
`kSmallestVisibleWavelengthInCells`, `kNegligibleViscousChange`, `ComputeViscosityToSmoothingRatio`,
`ComputeFourCellFeatureChange`, `ComputeReynoldsNumber`. Solids: `GuideKind`, `FreeMotionLimits::{guide_kind,
rail_direction, hinge_point, hinge_axis}`, `AllowedMotions`, `ComputeAllowedMotions`, `LiquidForceOnSolid`,
`SolidsTry::liquid_forces`, `SolidSettings::slip_length`. Render output: `RenderSettings::wind`,
`kFoamWindDriftDuration`. I/O: `kMemoryTakenToleranceBytes`, `IoError::kRecordedModeDoesNotFit`,
`Command::choose_mode_again`, CLI `--choose-mode-again`, scene fields `wind`, `cell_size_regions`, `cell_size_map`,
`slip_length`, `free_solid.guide`, report fields above. Blender: `cell_size_region_collection`,
`fluidity_cell_size_region.cell_width`, `cell_size_map_image`, `memory_left_free_bytes`, `guide_kind`,
`guide_object`, `slip_length`, `free_slip`.

### Owner questions raised here

1. The CPU budget now follows the GPU rule (0.9 of free, 512 MiB left free) instead of min(0.7 total, 0.8 free).
2. `ApplyLimits` now needs room for the old and the new block at once (pieces cannot be freed alone).
3. Starting values to calibrate: `shear_width_factor` 0.25, `kFoamWindDriftDuration` 0.1 s,
   `kMemoryTakenToleranceBytes` 2 MiB (needs the Windows run), the standing-wave loss 5 % of the later group.
4. The CUDA runtime licence note's wording.
5. The 2D refinement map (top view) instead of a 3D one.
6. A cylinder shape in contact (for the Taylor–Couette torque test), or another torque test.
7. A viscosity sample next to two solids reports its stress to the lower solid index only (viscosity design Open).
8. `BuildPressureEquations` in module 5 consumes `solids/guide_math.h` (module 8, header-only): pressure Task 47a
   runs right after solids Task 5a.
