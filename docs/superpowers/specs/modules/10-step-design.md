# Step module design

Status: updated 2026-11-14 with the owner's answers (A1 to A14), `changes-to-apply.md` and the final section 3
of the contact, grid, particles, transfer, pressure, viscosity, materials, solids, gas, cell resize and
batches, and surface, foam and spray designs. Module 8 of the build order. The I/O and CLI design and the
Blender add-on design are cited as `changes-to-apply.md` states their interface. Updated again 2026-11-14 with
the decisions recorded in `plan-decisions.md` and the step plan's amended tasks: A15 (batched solves, third run
mode), A17 (grid built on the run's processor), A18 (blocking materials and sealed gas inside the liquid solve),
and the step-related plan decisions (11.11 to 11.16). Updated again 2026-11-14 with owner answers A24 (viscous
liquids carry blocking materials and sealed gas; 11.26), A25 (what counts as viscous, and the viscous effect per
step; 11.27) and the reconcile pass of `plan-decisions.md` (items 3, 11, 12, 13 and the coordinator's list; 11.28
to 11.32), which the step plan already carried. Updated again 2026-11-14 with owner answers A26 (a viscous liquid
sticks to snow and springy solids: their band faces are unknowns of the viscous solve) and A27 (free solids in the
viscous solve; no combination of materials refused because of viscosity) (11.33).

## 1. Purpose

A simulation advances in steps. One step moves every material forward by a short duration: it adds particles
from sources, writes the particles onto the grid, applies forces, finds the pressure, reads the grid back and
moves the particles. The step module runs these stages in a fixed order for every scene, on a copy of the
values a step changes ("a try"), and replaces the kept state only when every stage passed ("a commit").

Owns: the `Simulation` object (the ready scene's persistent choices, the committed state of every module, all
buffers); `TryStep`; the stage order; the material sets of a scene and the order in which stages visit them;
the try-and-commit rule across modules; `RejectReason`; the capability check; the step duration and which limit
set it; particle sources and their counters; `ChooseActiveCells`, `ApplyForces`, the glue of `MoveParticles`
(mixing, the travel limit, removal through open sides); the sizing estimate and the choice between one resident
run and a batched run; limits and raising them; cancellation; callbacks; the snapshot of the whole state.

Does not own: any formula of contact, grid, particles, transfer, pressure, viscosity, materials, solids or gas
(it calls them); cell resizing and the batched execution (the cell resize and batches design, which receives
the hand-over of 5.11 and the stage table of 3.11); files, the scene file and messages for the user (the I/O
and CLI design); the surface, foam and spray (the surface, foam and spray design, reached through callbacks).

## 2. What exists today

| Where on `main` | What it does | What is wrong |
|---|---|---|
| `host/liquid_substep_3d.h:453-626` | 22 `AttemptWith...` names (23 declarations) plus `Attempt`, and four `AdvanceFrame...` loops (`:632-662`) | One public method per scene kind (naming review A1). Grouping in 5.1 |
| `h:225-232, 445` | `LiquidStageFaultHook3D`, `SetStageFaultHookForTesting` | A test hook in the public interface (A8) |
| `h:137-156` | `LiquidPhysicalStage3D`, the 17 stages | Kept as the base of the stage list. `kBodyContact` and `kWhitewater` are empty markers (`liquid_substep_3d.cc:1275, 2736`) |
| `src/liquid_substep_3d.cc:1121-2822` `Execute` | The stages for same-size cells, about 1,700 lines | No map; piston, box and snow passed as nullable pointers and two bare flags |
| `cc:1066-1120` `ExecuteAdaptive` | A second, shorter stage list for mixed sizes: no `Emit`, no `Forces`, no `Interface` | Two step paths; a scene gets one physics or the other by grid kind |
| `cc:3888-4056` `AttemptImpl` | Copies committed into the try, runs, maps a status code to a disposition (`:4017-4030`: `Aborted` is "shorten", `ResourceExhausted` is "capacity", the rest fatal) | 13 parameters; the outcome is encoded in Abseil codes with empty messages |
| `cc:2877-3103` `Create`; `cc:1771, 2884-2908, 3179-3719, 3906-3909` | Capability exclusions | 53 `Unimplemented`/`FailedPrecondition` returns, some after allocation, some inside a try, no typed reason, no suggested change |
| `cc:2823-2863` `ScheduledDuration` | Duration from the largest velocity part, gravity, one surface-tension limit, the time left, split evenly | One cell width and one density for the scene; does not say which limit was active; the surface-tension limit exists a second time in pressure |
| `cc:4102-4305` `AdvanceFrameImpl` | Loop to a frame time; retry; capacity restart (`:4235-4302`) | Restarts the whole frame from a third full copy, once; snow kept by the caller (`:4199-4203`) |
| `src/liquid_state_3d.cc:715-726` | `ResetTrialFromCommitted`: `CopyState(committed_, trial_)` | A full copy of the state per try, although the old paper says a reject costs no copy |
| `state.cc:736-827` `FinishAttempt` | Halves on retry, at most twice (`:773-783`); checks ids and counters did not go back (`:789-816`); snaps the last step to the frame time within one rounding step (`:757-762`); swaps (`:822`) | Good rules, ported. Stored as flags (`fatal_`, `capacity_pending_`) |
| `state.cc:828-848` | `RestartFrameWithCapacity` rebuilds every owner from `frame_start_` | Capacity is distinct from a numerical reject (kept); it costs a frame of work and a third copy (not kept) |
| `src/liquid_substep_3d_staging.cc:632-685` `Emit` | Sources: `owed = remainder + duration * rate`, counted particles, jitter from Philox keyed by seed, source and committed step number | The jitter of a particle depends on which step added it |
| `staging.cc:489-630` | Open-sea refill: scans all particles on the CPU, at most 4096 lattice sites | CPU scan of the whole state per try; fixed limit |
| `src/liquid_substep_3d_cpu.cc:239-398` `UpdateParticle` | Mix (`0.99 * blend`, `:248`), travel rule (`:308-310`), contact, travel reject above half a cell (`:384-386`), `volume_ratio *= exp(duration * divergence)` (`:387-390`) | Five modules' work in one function; four call shapes by scene kind |
| `cc:2738-2742, 2804-2811` | Solid travel above half a cell rejects, checked after the particles moved | Checked late; now also a duration limit chosen before the try |
| `native/cuda/liquid_substep_3d_host.cc`, `liquid_arena_3d_host.cc` | A table of function pointers per stage; CPU staging between stages; device memory allowed = min(70 % of total, 80 % of free) (`:794-797`) | Report, CUDA 1.1 and 3.1: arrays return to the CPU between dependent stages; a wait per stage (`:283-322`) |
| `proofs/Fluidity/LiquidTransaction.lean:143-212` | Five theorems on a model: reject keeps the state, commit advances the clock once | True by `rfl` of a model that is not the code |
| `docs/papers/fluidity-method.typ:58-82, 161-172, 464-470` | "The sub-step" (14 steps), "Time step", "Transactions" | Source for 5.2, 5.8, 5.6; Y-up and old words |
| gas branch `gas_substep_3d.cc:121-183`, `gas_frame_3d.cc` | Gas sub-step with its own loop, CPU staging, a `project` callback | Replaced by the stage `SolveGas` inside the one step (5.2) |

Y-up on `main`: gravity `{0, -9.81, 0}` (`h:100`), `floor_y`, the open side is +Y, the still-water height and the
box guide are along Y (`h:35, 584`; `staging.cc:494, 569`). Here gravity comes from the scene (default
`{0, 0, -9.81}`), the open domain side is `top` (+Z), and heights are along `kUp`. To compare a result of `main`,
rotate with Foundation's matrix (Y to Z).

## 3. Public interface

Folder `native/fluidity/step/`, namespace `fluidity::step`. All headers are CPU only unless marked (shared).
Names are NEW unless marked D (DECIDED). Sections 3.1 to 3.9 are what the CLI, the add-on and tests of other
modules use; 3.10 and 3.11 exist for the cell resize and batches design. Every fallible function returns
`std::expected<T, StepFailure>` (A2). Every total over the grid is one partial per cell block in the grid's
fixed block order, then `math::TreeSum` over the block partials.

### 3.1 `step_error.h` (shared) and `step_message.h`

```cpp
enum class Stage : std::uint8_t {  // D (the 17 names of main and their order); NEW: FindSolidSurfaces,
                                   //   BuildPressureEquations, SolveGas, MoveSolids. A6: stage 11 is named
                                   //   after the function it runs.
  kNone,                       // Before the first stage: the request itself was rejected.
  kAddNewParticles,            // 1
  kChooseActiveCells,          // 2
  kFindSolidSurfaces,          // 3 NEW
  kParticlesToGrid,            // 4
  kSaveGridBeforeForces,       // 5
  kApplyForces,                // 6
  kDriveWaves,                 // 7
  kCollideSolids,              // 8
  kFindLiquidSurface,          // 9
  kBuildPressureEquations,     // 10 NEW
  kSolveViscosityAndPressure,  // 11 (A6; was ApplyViscosity). Scenes with a viscous ingredient: a liquid
                               //   ingredient whose viscosity setting is above 0 (> 0.0, no threshold; A25).
                               //   Chosen once from the ready scene (CheckSceneCanRun, BuildSimulation) and
                               //   never changed during the run.
  kSolvePressure,              // 12 Scenes with liquid and no viscous ingredient (every liquid ingredient has
                               //   viscosity 0; A25). The same one-time choice.
  kSolveGas,                   // 13 NEW (gas). Scenes with gas. There is no MoveGas stage.
  kSaveGridAfterPressure,      // 14
  kGridToParticles,            // 15
  kMoveParticles,              // 16
  kMoveSolids,                 // 17 NEW
  kMakeFoamAndSpray,           // 18
  kCheckStep,                  // 19
  kCommitStep,                 // 20
  kResizeCells,                // 21
};
inline constexpr int kStageCount = 22;

enum class RejectReason : std::uint8_t {  // D name. Why a try did not commit. Grouped by ToRejectKind.
  kNone,
  // A shorter step can help.
  kParticleTravelledTooFar,       // A particle or a grid-only cell's liquid moved more than allowed in one step.
  kSolidTravelledTooFar,          // solids kPathJumpInsideStep, kSolidMovesTooFar; contact kSolidTurnsTooFar.
  kContactTooCloseToCall,         // contact kTooCloseToCall (kSecondReactingSolid included).
  kParticleOutsideGrid,           // A particle is in no active cell after grouping.
  kStepTooLongForSurfaceTension,  // pressure. magnitude = step_duration, limit = the allowed duration, s.
  kStepTooLongForAirPocket,       // pressure. Same fields.
  kStepTooLongForGasFlow,         // NEW. gas kStepTooLongForGasFlow. Same fields.
  kPressureNotConverged,          // pressure, viscosity or the gas solve used all iterations above the limit.
  kViscosityNotConverged,         // The velocity solve used all its iterations.
  kStepTooLongForMaterial,        // materials kStepTooLong. limit = the sound limit, s.
  kDeformationCollapsed,          // materials kDeformationCollapsed: a deformation lost its positive volume.
  kValueNotFinite,                // A stage produced infinity or NaN.
  // A limit was reached. Raise it with ApplyLimits, then try the same step again.
  kTooManyParticles, kTooManyCellBlocks, kBuffersTooSmall,
  // The caller asked to stop.
  kCancelRequested,
  // The request is wrong. Nothing ran.
  kBadStepDuration,               // Not finite, not positive, or time + step_duration == time.
  kWrongStateVersion,             // StepRequest::state_version is not the committed one.
  // No retry helps.
  kNotSupportedNow,               // The state left what the capability check allows. context.capability says how.
  kStageFailed,                   // A module failed for a reason a shorter step does not cure. See context.
  kCallbackFailed,                // A callback of StepCallbacks reported a failure.
  kRunOnCallFailed,               // The processor failed. The simulation is no longer usable.
};
enum class RejectKind : std::uint8_t {  // What the caller can do. A pure function of the reason.
  kNone, kShortenStep, kRaiseLimits, kCancelled, kBadRequest, kCannotContinue };
constexpr RejectKind ToRejectKind(RejectReason reason);

enum class Module : std::uint8_t {     // NEW values appended after kBase, so the stored numbers never change.
  kStep, kContact, kGrid, kParticles, kTransfer, kPressure, kViscosity, kMaterials, kSolids, kGas, kBase,
  kCellResize, kBatches, kRenderOutput,
};
inline constexpr int kModuleCount = 14;
enum class CapabilityReason : std::uint8_t;   // 3.4
struct RejectContext {                 // Compact, never allocates. Zero where not relevant.
  Module module = Module::kStep;       // Whose failure this was.
  std::uint8_t module_error = 0;       // That module's error enum value, for ToMessage and ToName at the boundary.
  std::uint64_t particle_id = 0;
  std::uint64_t material_id = 0;
  grid::CellId cell{};
  std::uint32_t solid_id = 0;
  double magnitude = 0;                // What was measured, in the unit the reason names.
  double limit = 0;                    // What was allowed, same unit.
  CapabilityReason capability{};       // Only with kNotSupportedNow (3.4).
};

enum class StepError : std::uint8_t {  // The module's error enum, for everything that is not a try.
  kNone,
  kBadInput,            // A scene or settings value is not finite or out of range. context names the module.
  kSceneNotSupported,   // CheckSceneCanRun found a finding that blocks the run.
  kDoesNotFit,          // BuildSimulation, ApplyLimits: the plan's fit is PlanFit::kDoesNotFit (the scene fits
                        //   neither the processor nor, in batches, the CPU memory), or a raise does not fit.
                        //   BuildRunPlan no longer returns it (A17 planned cost, 11.17).
  kBatchedRunNotBuilt,  // The plan chose a batched run and StepCallbacks::run_stages is null, or
                        //   RunPlan::max_loaded_blocks is 0, or (A15) the mode is kBatchedWithSolvesInPieces and
                        //   RunPlan::max_solve_loaded_blocks is 0.
  kOutOfMemory,         // An allocation failed although the estimate fitted. Nothing changed.
  kLimitNotRaised,      // ApplyLimits: a new limit is below the current one, or all are equal.
  kTryInProgress,       // A step try or a resize try of this simulation is alive.
  kSnapshotDoesNotMatch,// BuildSimulationFromSnapshot: material ids, counts, grid or versions do not fit.
  kSimulationUnusable,  // An earlier kRunOnCallFailed or failed commit. Start again from the last saved state.
  kRunOnCallFailed,     // CheckCanRunOn failed for StepSettings::run_on, or a call to it failed.
};
struct StepFailure { StepError error = StepError::kNone; RejectContext context{}; };  // A2
// Plan decision (11.16): StageOutcome is declared here (it was in step_stages.h, 3.11). A private
// ToStageOutcome(const <Module>Failure&) per module in stage_outcome_from_modules.h maps errors to reasons (5.6).
struct StageOutcome { RejectReason reason = RejectReason::kNone; RejectContext context{}; };
inline constexpr int kRejectReasonCount = 23;   // NEW (I/O). Every enumerator, kNone included.
// step_message.h. ToMessage: one English sentence. ToName (NEW): the stable code of the error enum, the
// enumerator's name without the k in lower snake case ("too_many_cell_blocks"). The stable names of Stage,
// RejectReason, CapabilityReason, SuggestedChange and StepLimit are I/O's (io/stable_names.h), sized by the
// counts declared here; the step declares no second ToName for them.
std::string_view ToMessage(StepError error);
std::string_view ToMessage(RejectReason reason);
std::string_view ToMessage(Stage stage);
std::string_view ToName(StepError error);       // NEW
```

`RejectReason` is not a second error enum: a rejected try is an ordinary result of `TryStep`, not a failure of
the call (decision 3).

### 3.2 `ready_scene.h`

Declared here (decision 9), filled by the I/O and CLI design from the scene file (the pattern of
`transfer::SolidsAtFaces`). Spans are borrowed for the call that reads them; `BuildSimulation` copies what it
keeps. Every field is a setting that changes results and is hashed with the scene; no run control lives here.

```cpp
struct ParticleSource {                // One point source. Its counters are step state (5.4).
  std::uint64_t source_id = 0;         // Stable, nonzero, unique. Sources are processed in ascending id.
  std::uint32_t material_index = 0;    // Not the gas material: gas markers are made by gas::FindMarkerChanges.
  double particles_per_second = 0;     // 1/s, zero or more.
  double start_time = 0, end_time = 0; // s. Active while start_time <= time < end_time.
  Vec3 position{};                     // m, world.
  Vec3 position_spread{};              // m, zero or more. Half size of the box new particles are spread in.
  Vec3 velocity{};                     // m/s, world.
  double mass = 0;                     // kg per particle.
  double rest_volume = 0;              // m3 per particle.
  std::uint8_t size_level = 0;
  std::span<const double> ingredient_fractions;  // Sums to 1. Empty for one ingredient.
};
struct StartParticles { std::uint32_t material_index = 0; particles::NewParticles particles; };

struct ReadyScene {                    // D word. Defaults filled in and checked. SI units, Z-up.
  contact::DomainBox domain_box;
  Vec3 gravity{0, 0, -9.81};           // m/s2, world. solids.gravity must equal it (kBadInput).
  grid::GridFrame grid_frame;          // origin must equal domain_box.low_corner.
  std::uint8_t new_cell_size_level = 0;  // Size of a new cell when no cell exists nearby and no callback says.
  double start_time = 0;               // s.
  double end_time = 0;                 // s. Used only to estimate how many particles sources will add.
  std::uint64_t random_seed = 0;
  materials::MaterialTableInputs materials;   // Materials, ingredients, export names, pairs. material_index is
                                              //   the position in materials.materials.
  materials::MaterialSettings material_settings;  // Support rule, sound step share, state tolerance.
  // NEW (A8). One transfer material_is_counted array per material set (one entry per material, each 0 or 1),
  // in ascending order of the lowest counted material_index: the step's set order. Rules (5.2): the sets are
  // disjoint and complete (transfer::CheckMaterialSets); every kLiquid material is in one set, the liquid set,
  // which holds no other kind; the kGas material is alone in its set; the materials of one set share
  // materials::Material::at_solid_faces. Anything else is CapabilityReason::kMaterialSetsNotValid.
  std::span<const std::span<const std::uint8_t>> material_sets;
  std::span<const StartParticles> start_particles;
  std::span<const ParticleSource> sources;
  solids::SolidsSettings solids;              // NEW type (was a span of SolidSettings). solids.run_on must equal
                                              //   StepSettings::run_on (kBadInput); nothing changes it.
  std::optional<pressure::OpenSeaSettings> open_sea;
  std::uint32_t open_sea_material_index = 0;  // NEW. The liquid material the open-sea zone removes and refills.
  double open_sea_particle_spacing = 0;       // m. Refill lattice in the open-sea zone (5.4).
  std::optional<gas::GasSettings> gas;        // Present exactly when a material is kGas.
  std::span<const gas::GasSource> gas_sources;  // NEW. Keyed, ascending source_id. I/O owns the key storage.
  gas::GasWind gas_wind;                      // NEW. Keyed. Empty: no wind.
  gas::MarkerSettings gas_markers;            // NEW.
  gas::ActiveGasSettings gas_active_cells;    // NEW.
  transfer::TransferSettings transfer;
  pressure::SolveSettings pressure_solve;
  pressure::AirPocketSettings air_pockets;
  pressure::VolumeCorrectionSettings volume_correction;
  viscosity::ViscositySettings viscosity;
  double particle_radius_in_cell_widths = 0.5;  // For pressure::SurfaceDistanceInputs::particle_radius.
  // NEW (cell resize). The scene has cell resize settings: the seven region arrays of GridFields are built.
  bool has_cell_resize = false;
  std::uint32_t snapshots_kept_at_once = 1;     // For the sizing estimate (kOneSnapshot entries).
};
```

### 3.3 `step_settings.h`

```cpp
inline constexpr double kMaxTravelInCellWidths = 0.5;  // Ported (main: half a cell). Why: 5.8.
inline constexpr double kCoveredReachInCellWidths = 2.5;  // Cells this near a particle are active (5.3).

struct SimulationLimits {              // D pattern max_... . Raised by ApplyLimits, never lowered.
  std::vector<std::size_t> max_particle_counts;  // Per material_index.
  std::size_t max_added_per_try = 0;
  std::uint32_t max_cell_blocks = 0;   // Reconcile (11.31): set only by BuildRunPlan from the scene, never from
                                       //   grid::GridSettings' default (0); raised by ApplyLimits.
  std::uint32_t max_air_pockets = 0;
  std::uint32_t max_gas_blobs = 0;     // NEW. gas::GasBufferSizes::max_gas_blobs. 0 without gas.
};
struct StepSettings {                  // D suffix. Chosen once, kept for the run. How to run, never what.
  RunOn run_on = RunOn::kCpu;          // D (A12). The caller's choice; nothing changes it silently.
  double max_step_duration = 0.01;     // s. Ported default.
  int max_retries_per_step = 2;        // Ported: at most two shorter tries, then AdvanceTo stops.
  int max_limit_raises_per_step = 4;   // main: one restart per frame.
  double limit_growth = 1.5;           // No unit, above 1. A raised limit is at least this times the old one.
  double start_limit_headroom = 1.25;  // No unit, 1 or more. Start limits over the counted scene.
  double memory_share_of_total = 0.7;  // Ported policy (liquid_arena_3d_host.cc:794).
  double memory_share_of_free = 0.8;   // Ported policy. Allowed bytes = the smaller of the two.
  particles::GroupingSettings grouping{};  // max_move_since_grouped = 0: regroup every try.
};
```

### 3.4 `capability.h`

```cpp
enum class CapabilityReason : std::uint8_t {
  kNone,
  // From viscosity::CheckViscosityCanRun (viscosity 5.7), one each. (No kViscousLiquidWithSnow: A24 carries snow's
  // material faces in the joint solve; 11.26. No kViscousLiquidWithFreeSolid: A27 makes a free solid's motion an
  // unknown of the viscous solve; 11.33. Replaced: "kViscousLiquidWithFreeSolid, // kFreeSolidNotSupported. Kept
  // (A24)".)
  kViscousLiquidWithMovingSolidThroughCells,  // kMovingSolidNotSupported.
  kViscousLiquidWithInclinedSolid,            // kInclinedSolidNotSupported.
  kViscousLiquidAtSizeChange,                 // kSizeChangeNotSupported.
  // From pressure, transfer, contact, grid, particles, materials.
  kThinSolidInsideOneCell,         // pressure kCellSplitBySolid: a plate thinner than a cell, liquid on both sides.
  kMoreThanOnePlateNearAFace,      // transfer kTooManySidesAtFace (kMaxSidesAtFace = 2).
  kSolidThinnerThanTolerance,      // contact kSolidTooThin.
  kDomainTooLargeForSmallestCell,  // grid kCoordOutOfRange, or more than kMaxSizeLevels levels.
  kTooManyIngredients,             // particles kMaxIngredientCount.
  // (No kLiquidsThatDoNotMixTouch: pressure supports two liquids that do not mix; plan decision, 11.16.)
  kLiquidSurfaceTensionsDiffer,    // materials kSurfaceTensionsDiffer (materials decision 6).
  kMaterialSetsNotValid,           // NEW. transfer::CheckMaterialSets, or a set rule of ReadyScene::material_sets.
  // Programme design section 9, carried forward.
  kInternalForceAcrossSizeChange,  // Snow or a springy solid in cells of different sizes (A9).
  kWavePaddleReceivingVolume,      // Known limit: volume error behind a paddle that moves away (solids).
  // This design.
  kStepDurationFloor,              // A static limit is below what time + duration can represent.
  kFromSolids, kFromGas,           // An exclusion or scene rule those modules declare. module_code says which.
};
// NEW (I/O). Every enumerator, kNone included. 16 at most (A24 removed kViscousLiquidWithSnow, A27
// kViscousLiquidWithFreeSolid); lower by one for each viscosity exclusion (kViscousLiquidWithInclinedSolid,
// ...WithMovingSolidThroughCells, ...AtSizeChange) that the viscosity module lifted: that reason then does not exist
// (13 if all three were lifted). Plan decisions "Pressure (continued)", "Viscosity", "A24: exclusions", "A viscous
// liquid sticks, and free solids in the viscous solve (A26, A27)". Replaced: 17 at most, 14.
inline constexpr int kCapabilityReasonCount = 16;
// Renamed (glossary ruling; plan decision "Sand"): was Support { kHasKnownLimit, kNotSupported }. A known limit is
// kExperimental; an exclusion is kUnavailable. Only kUnavailable blocks a run. The full list.
enum class CapabilityLevel : std::uint8_t { kCanRun, kExperimental, kUnavailable };
// No kSetViscosityToZero (A27, 11.33): no reason of the main sequence uses it; the sand module adds it back with
// kViscousLiquidWithSand. Replaced: "kSetViscosityToZero is kept although A24 removed its only reason."
enum class SuggestedChange : std::uint8_t {
  kNone, kUseOneSurfaceTension, kUseSmallerCells, kUseOneCellSizeThere, kMakeSolidThicker, kMakeSolidFixed,
  kAlignSolidWithAxes, kSeparateTheObjects, kUseFewerIngredients, kMakeDomainSmaller, kPutAllLiquidsInOneSet,
  kUseFewerSolids };
struct CapabilityFinding {
  CapabilityReason reason = CapabilityReason::kNone;
  CapabilityLevel level = CapabilityLevel::kUnavailable;  // Field name: Open (11).
  SuggestedChange suggested_change = SuggestedChange::kNone;
  std::uint64_t material_id = 0;       // 0 when not about one material.
  std::uint32_t solid_id = 0;          // 0 when not about one solid.
  double magnitude = 0, limit = 0;     // In the unit the reason names (m for the thin-solid reasons).
  Module module = Module::kStep;       // NEW. The module whose check found it.
  std::uint8_t module_error = 0;       // NEW. That module's error value.
  std::string_view module_message;     // ToMessage of that module's error. A string literal.
  std::string_view module_code;        // NEW. ToName of that module's error ("plate_too_thin_for_liquid").
};
struct CapabilityResult {              // NEW shape (Blender): every finding, no cap.
  std::vector<CapabilityFinding> findings;  // kUnavailable first, then by reason, solid_id, material_id.
  bool can_run = true;                 // No finding with kUnavailable.
};
// A pure predicate over the scene. Allocates only `findings`, once per call; touches no processor; works on a
// machine with no GPU. Reports every finding of every check of 5.14, not the first.
CapabilityResult CheckSceneCanRun(const ReadyScene& scene, const StepSettings& settings);
std::string_view ToMessage(CapabilityReason reason);
std::string_view ToMessage(SuggestedChange change);
inline constexpr int kSuggestedChangeCount = 12;     // NEW (I/O). Every enumerator, kNone included. A27: was 13.
```

### 3.5 `run_plan.h`: analyse, estimate, check the fit, choose

```cpp
struct SceneCounts {                   // Step 1 of programme design 8a.
  grid::GridCounts grid;               // Cells per size level, cell blocks, face pieces, at the start.
  MemoryCounts at_limits;              // What the limits allow: blocks, places, particles (base/memory_count.h).
  std::uint64_t start_particles = 0, particles_from_sources = 0;
  std::uint32_t material_count = 0, ingredient_count = 0, solid_count = 0, free_solid_count = 0;
  std::uint32_t material_set_count = 0;      // NEW (A8).
  std::uint32_t sets_without_solve = 0;      // NEW. Sets no solve writes (snow, springy solids).
};
struct ModuleBytes {
  Module module; std::uint64_t whole_run, one_step_try, one_call, one_snapshot;
  std::uint64_t on_cpu = 0;            // NEW (A17 planned cost). This module's entries with where = kCpu.
};
struct MemoryEstimate {                // Step 2. Sums of the modules' MemoryEstimateEntry lists at the limits.
  std::array<ModuleBytes, kModuleCount> by_module{};
  std::uint64_t whole_run = 0;         // Committed state and buffers reused every step.
  std::uint64_t one_step_try = 0;      // Try copies of what a step changes.
  std::uint64_t one_call = 0;          // Largest scratch alive at once (5.9).
  std::uint64_t snapshots = 0;         // one_snapshot entries times snapshots_kept_at_once. CPU memory.
  std::uint64_t on_processor = 0;      // whole_run + one_step_try + one_call, entries with where = kRunOn only.
  std::uint64_t on_cpu = 0;            // NEW (A17 planned cost). Entries with where = kCpu, plus snapshots.
};
// NEW (A17, batched-run buffers). Copies between CPU memory and the run's processor: one ReadBuffer or WriteBuffer
// slice call is one copy (in a batched CPU run the same calls, CPU to CPU, counted the same); a batch load and a
// solve batch load count one copy each. Shared by RunPlan and StepReport.
struct CopyCounts {
  std::uint32_t copy_count = 0;
  std::uint64_t bytes_to_processor = 0;
  std::uint64_t bytes_from_processor = 0;
};
// NEW (A17 planned cost). BuildRunPlan returns a plan that does not fit instead of failing.
enum class PlanFit : std::uint8_t { kFits, kDoesNotFit };
// MemoryAvailable { total_bytes; free_bytes; } is Foundation's (base/, from ReadMemoryAvailable(run_on)).
// total_bytes 0 means "not known": the share of the total is then not applied.
// A15. kBatched: particle stages in batches, every solve once on the whole grid on the run's processor.
// kBatchedWithSolvesInPieces (NEW): as kBatched, and the solvers' scratch lives in CPU memory and goes through the
// processor one loaded piece at a time (pressure/solve_batches.h). BuildRunPlan chooses only kOnePiece or
// kBatched; the caller (I/O Run, tests) copies batches::BuildBatchPlan's choice (SolvePlace::kInPieces) in.
enum class RunMode : std::uint8_t { kOnePiece, kBatched, kBatchedWithSolvesInPieces };
// NEW (A25). One per liquid ingredient (scene ingredient order), filled by BuildRunPlan from the ingredient's
// viscosity, its material's rest density, StepSettings::max_step_duration and the grid frame's smallest cell
// width: viscous_effect_per_step = viscosity::ComputeViscousEffectPerStep(viscosity, step_duration, density,
// cell_width). It advises (I/O prints it); it never chooses the stage.
struct LiquidViscousEffect {                 // step run_plan.h, NEW (A25)
  std::uint32_t ingredient_index = 0;        // scene ingredient
  double viscosity = 0;                      // Pa s, the setting
  double density = 0;                        // kg/m3, rest density of its material
  double viscous_effect_per_step = 0;        // no unit; 0 when viscosity is 0
};
enum class ViscousPath : std::uint8_t { kPressureSolve, kViscousSolve };   // stage 12 or 11, fixed for the run
struct RunPlan {
  SceneCounts counts;
  SimulationLimits limits;             // Start limits: counts times start_limit_headroom. Reconcile (11.31):
                                       //   max_cell_blocks = the scene's cover blocks x start_limit_headroom,
                                       //   rounded up, capped by the domain's blocks at the smallest cell, >= 1.
  MemoryEstimate estimate;
  std::uint64_t allowed_bytes = 0;     // Step 3: min of the two shares of StepSettings.
  RunMode mode = RunMode::kOnePiece;   // Step 4.
  // Both batched modes. BuildRunPlan leaves them 0; the caller copies them from batches::BuildBatchPlan(scene,
  // plan, settings, batch_settings) before BuildSimulation.
  std::uint32_t max_blocks_per_batch = 0;
  std::uint32_t max_loaded_blocks = 0;       // NEW. BatchPlan::max_loaded_blocks.
  std::uint64_t max_loaded_particles = 0;    // NEW. BatchPlan::max_loaded_particles.
  std::uint32_t max_solve_loaded_blocks = 0; // NEW (A15). BatchPlan::max_solve_loaded_blocks.
  std::uint64_t allowed_cpu_bytes = 0;       // NEW (A15). Step 3 of BuildRunPlan applied to on_cpu.
  PlanFit fit = PlanFit::kFits;              // NEW (A17 planned cost). kDoesNotFit: mode kBatched, every estimate
                                             //   filled; BuildSimulation (and I/O BuildRun) fail kDoesNotFit on it.
                                             //   Also when limits.max_cell_blocks > grid::kMaxCellBlocksLimit
                                             //   (reconcile, 11.31).
  CopyCounts planned_step_copies;            // NEW (A17). Upper bound per try of the step's own copies (6.5,
                                             //   Appendix B of the plan) at the limits, for the plan's mode.
  // NEW (A25). The viscous effect per step of each liquid, at viscous_effect_cell_width (m, the grid frame's
  // smallest cell width) and viscous_effect_step_duration (s, StepSettings::max_step_duration); viscous_path is
  // kViscousSolve exactly when some liquid ingredient's viscosity is above 0.
  std::vector<LiquidViscousEffect> liquid_viscous_effects;
  double viscous_effect_cell_width = 0;      // m
  double viscous_effect_step_duration = 0;   // s
  ViscousPath viscous_path = ViscousPath::kPressureSolve;
};
// Allocates nothing on any processor. `on_processor` is what StepSettings::run_on offers; `on_cpu` is the CPU
// memory a batched run would keep the state in. Both are inputs so the plan can be tested with no GPU. A scene that
// fits nowhere is a plan with fit = kDoesNotFit, not a failure (A17 planned cost).
// NEW: output_estimate are render_output::BuildRenderOutputMemoryEstimate, counted as Module::kRenderOutput;
// cell_resize_estimate are cell_resize::BuildCellResizeMemoryEstimate, counted as Module::kCellResize. Empty: none.
std::expected<RunPlan, StepFailure> BuildRunPlan(const ReadyScene& scene, const StepSettings& settings,
                                                 MemoryAvailable on_processor, MemoryAvailable on_cpu,
                                                 std::span<const MemoryEstimateEntry> output_estimate,
                                                 std::span<const MemoryEstimateEntry> cell_resize_estimate = {});
struct MemoryReport {                  // Estimate against use. Reported before the run and by AdvanceTo.
  MemoryEstimate estimate;
  std::uint64_t allocated_bytes = 0;   // Sum of every buffer built with where = kRunOn. Equals estimate.on_processor.
  std::uint64_t allocated_cpu_bytes = 0;  // NEW (A17 planned cost). Rows walked with where = kCpu. Equals
                                          //   estimate.on_cpu - estimate.snapshots.
  SimulationLimits limits;
  std::vector<std::size_t> largest_particle_counts;  // Per material, largest committed so far.
  std::uint32_t largest_cell_blocks = 0;
  std::uint32_t limit_raise_count = 0;
};
// 6.3. Amended (A17, batched-run buffers): null batched_plan gives the one-piece entries of 6.3; a batched plan
// gives the entries of 6.5 (loaded-only entries at max_loaded_blocks / max_loaded_particles with where = kRunOn,
// whole entries placed by the placement rule, the batched-run entries: mirrors, grid copy, cell list, stage-17
// solids instance).
std::vector<MemoryEstimateEntry> BuildStepMemoryEstimate(const ReadyScene& scene, const SceneCounts& counts,
                                                         const RunPlan* batched_plan = nullptr);
```

### 3.6 `simulation.h`

```cpp
struct StepCallbacks;                  // 3.9
struct GridFields;                     // 3.10
class Simulation {                     // Move-only. Owns everything listed in section 1. Built once.
 public:
  double time() const;                 // s. Committed.
  std::uint64_t state_version() const; // D. Grows by one at each commit of a step or of a resize.
  std::uint64_t step_count() const;    // Committed steps since the start of the run.
  RunOn run_on() const;
  bool is_usable() const;              // False after kRunOnCallFailed or a failed commit (5.6).
  const SimulationLimits& limits() const;
  const grid::StaggeredGrid& grid() const;          // Committed.
  const GridFields& committed_fields() const;       // NEW. Committed.
  const particles::Particles& particles() const;    // Committed.
  const pressure::AirPockets& air_pockets() const;  // Committed.
  const solids::Solids& solids() const;             // NEW. Committed.
  const gas::GasAmounts* gas_amounts() const;       // NEW. Committed. Null without gas.
  const materials::MaterialTable& material_table() const;           // NEW
  std::span<const std::span<const std::uint8_t>> material_sets() const;  // NEW. The step's set order.
};
// Runs CheckSceneCanRun again, allocates from the plan's memory estimate entries, adds the start particles, checks them with
// materials::CheckMaterialState, chooses the first active cells and groups (5.9). Fails with
// kSceneNotSupported, kBadInput, kDoesNotFit, kBatchedRunNotBuilt, kOutOfMemory, kRunOnCallFailed. On failure
// nothing stays allocated. Fixes the viscous path (stage 11 or 12) from the ready scene by the A25 rule; nothing
// afterwards changes it. Reconcile (11.31): the simulation owns one grid::GridScratch, built first and released
// last, lent to the committed and free GridBuffers (and the loaded try's buffers in a batched run).
std::expected<Simulation, StepFailure> BuildSimulation(const ReadyScene& scene, const StepSettings& settings,
                                                       const RunPlan& plan, const StepCallbacks& callbacks);
// The explicit growth of 5.7. Not inside a try. On failure the simulation is unchanged and usable.
std::expected<MemoryReport, StepFailure> ApplyLimits(Simulation& simulation, const SimulationLimits& limits);
MemoryReport ReadMemoryReport(const Simulation& simulation);
// NEW (A17, batched-run buffers). The step's own copies (6.5) since BuildSimulation: every try, committed or not.
// (Reconcile, 11.29: a committed resize adds none; the refresh by grid::ReadGrid is withdrawn.)
CopyCounts ReadCopyCounts(const Simulation& simulation);
// NEW (A17, batched-run buffers). The CPU copy of the committed grid in a batched GPU run (built by the CPU builder
// from the same cell list, swapped at commit; after a committed resize, the CPU resized grid handed over with
// ApplyResizedCpuGridCopy, no copy; reconcile, 11.29), the committed grid itself in a CPU run, null in a one-piece
// GPU run. Grouping, choose_cell_size, the marker upkeep and CellResize read it. The pair of CPU copy buffers
// shares one CPU grid::GridScratch (11.31).
const grid::StaggeredGrid* FindCpuGridCopy(const Simulation& simulation);

struct SourceCounter { std::uint64_t source_id = 0; double owed_particles = 0; std::uint64_t next_child_number = 0; };
// NEW (I/O). What ComputeStepDuration reads from the last commit besides the scene, kept at each commit (5.8).
struct CommittedStepLimits {
  double particle_travel = 0;          // s. Smallest ComputeTravelStepLimit over particles of the last commit.
  double largest_speed_in_cell_widths = 0;  // 1/s. Particles, solids, gas markers; for the report.
  double surface_tension = 0;          // s. GeometrySnapshot::surface_tension_step_limit of the last commit.
  double air_pocket = 0;               // s. AirPockets::step_limit of the last commit.
  double material_wave_speed = 0;      // s. InternalForcesResult::step_limit, smallest over ranges.
  double gas = 0;                      // s. GasResult::step_limit. +infinity without gas.
  double grid_only_speed = 0;          // 1/s. NEW (I/O). StepCallbacks::read_largest_grid_only_speed at commit
                                       //   (cell_resize::ReadLargestGridOnlySpeed). 0 without cell resize.
  double last_step_duration = 0;       // s. Of the last committed step; 0 before the first.
};                                     // +infinity where a limit does not apply.
struct SimulationSnapshot {            // D word. CPU-owned copy of the committed state at one state_version.
  double time = 0;  std::uint64_t state_version = 0, step_count = 0;
  grid::GridVersion grid_version{};                // NEW. BuildGrid rebuilds the committed grid with it.
  particles::ParticlesSnapshot particles;          // Full (with position_when_grouped) for a saved state.
  std::vector<grid::CellId> active_cells;          // Ascending; one entry per active cell of the grid.
  // By active cell, in active_cells order.
  std::vector<double> pressure;                    // Pa.
  std::vector<std::uint8_t> has_pressure;          // NEW (I/O).
  std::vector<std::uint32_t> pocket_of_cell;       // NEW (I/O). pressure::kNoAirPocket outside every pocket.
  // NEW (cell resize and render output). The seven region arrays; empty without cell resize.
  std::vector<std::uint8_t> is_grid_only_region, layers_from_grid_only_region;
  std::vector<double> merge_ready_duration;        // s.
  std::vector<double> grid_only_mass;              // kg.
  std::vector<double> grid_only_rest_volume, grid_only_current_volume;   // m3.
  std::vector<double> grid_only_ingredient_volume; // m3, [active cell * ingredient_count + ingredient].
  // By grid::FacePieceId order. Empty when not kept (6.3).
  std::vector<double> velocity_after_pressure;     // m/s. The liquid set.
  std::vector<double> gas_velocity_after_pressure; // m/s. NEW. The gas set; empty without gas.
  std::vector<pressure::AirPocket> air_pockets;  pressure::BlobHistory blob_history;
  std::vector<SourceCounter> source_counters;
  CommittedStepLimits step_limits;                 // NEW (I/O).
  solids::SolidsSnapshot solids;
  std::optional<gas::GasAmountsSnapshot> gas;      // NEW type (I/O): the full gas amounts. Empty without gas.
  SimulationLimits limits;
};
// Blocking. Only between steps. On the GPU this is the one large copy to the CPU.
std::expected<SimulationSnapshot, StepFailure> ReadSnapshot(const Simulation& simulation,
                                                            particles::SnapshotFields particle_fields);
// Rebuilds the state of a full snapshot so that the next steps are the bytes of an uninterrupted run (5.13).
std::expected<Simulation, StepFailure> BuildSimulationFromSnapshot(
    const ReadyScene& scene, const StepSettings& settings, const RunPlan& plan, const StepCallbacks& callbacks,
    const SimulationSnapshot& snapshot);
```

### 3.7 `try_step.h`

```cpp
struct StepRequest {                   // D name. Fields verified against the 22 methods (5.1).
  double step_duration = 0;            // s, positive. D.
  std::uint64_t state_version = 0;     // The committed version this step starts from.
  const std::atomic<bool>* cancel_requested = nullptr;  // Read between stages and inside the solves (5.10).
};
struct StepTimings { std::array<double, kStageCount> stage_elapsed_seconds{}; };  // D name. Clock time.
struct StepReport {                    // What one try measured. Valid for a reject up to the failed stage.
  std::array<Stage, kStageCount> stages_run{};  int stage_count = 0;   // Skipped stages are absent.
  particles::ParticleAmounts added{}, removed_through_open_sides{}, removed_in_open_sea{};
  particles::ParticleAmounts markers_added{}, markers_removed{};      // NEW. Gas markers.
  std::array<std::uint64_t, transfer::kGridOnlyReasonCount> particle_count_by_reason{};  // Summed over sets.
  std::uint64_t hit_count = 0, started_inside_count = 0;
  pressure::FlowImbalanceReport flow_imbalance{};     // The liquid solve.
  materials::InternalForceTotals internal_force{};    // ComputeInternalForceTotal per set, in set order.
  solids::CollideSolidsResult collide_solids{};       // NEW
  Vec3 wave_momentum_added{};  double wave_energy_added = 0;  // NEW. DriveWaves, tree sum over blocks.
  gas::GasAmountChanges gas{};                        // NEW. GasResult::changes.
  pressure::FlowImbalanceReport gas_flow_imbalance{}; // NEW
  int pressure_iteration_count = 0, velocity_iteration_count = 0, gas_iteration_count = 0;
  double largest_travel_in_cell_widths = 0;  // No unit.
  std::uint32_t cell_block_count = 0;
  StepTimings timings{};
  CopyCounts copies{};                 // NEW (A17). This try's step-own copies (6.5); for a reject up to the failed
                                       //   stage. Copies of batches, the solvers and cell resize are theirs.
};
struct StepResult {                    // D name.
  bool committed = false;              // D.
  RejectReason reject_reason = RejectReason::kNone;  // D.
  Stage rejected_in = Stage::kNone;    // The stage that failed. kNone with kBadRequest.
  std::optional<double> shorter_step_duration;       // s. D. Present exactly when the kind is kShortenStep.
  std::optional<SimulationLimits> limits_needed;     // Present exactly when the kind is kRaiseLimits.
  RejectContext context{};
  double time = 0;                     // s. Committed time after this call. Unchanged by a reject.
  std::uint64_t state_version = 0;     // Committed version after this call.
  StepReport report{};
};
// One try of one step for any scene. Commits by swapping, or leaves the simulation exactly as it was.
// Fails (std::unexpected) only with kSimulationUnusable or kTryInProgress; every other outcome is a StepResult.
std::expected<StepResult, StepFailure> TryStep(Simulation& simulation, const StepRequest& request);  // D
```

### 3.8 `step_duration.h`

```cpp
enum class StepLimit : std::uint8_t {  // Which limit gave the duration.
  kMaxStepDuration, kEvenStepsToTarget, kParticleTravel, kSolidMotion, kSolidTurn, kSurfaceTension,
  kAirPocket, kMaterialWaveSpeed, kGas, kSourceWindow, kAfterReject };
inline constexpr int kStepLimitCount = 11;
struct ChosenStepDuration {
  double step_duration = 0;            // s.
  StepLimit active_limit = StepLimit::kMaxStepDuration;
  std::array<double, kStepLimitCount> limit_by_kind{};  // s, +infinity where a limit does not apply.
  int steps_to_target = 0;             // Equal steps planned to reach target_time.
};
// Reads only committed values (CommittedStepLimits, the solids' committed state) and the scene. No large copy.
std::expected<ChosenStepDuration, StepFailure> ComputeStepDuration(const Simulation& simulation,
                                                                   double target_time);
// (shared, step_duration_math.h) Largest duration for which speed * t + acceleration * t^2 <= travel, s.
FLUIDITY_CPU_AND_GPU double ComputeTravelStepLimit(double speed, double acceleration, double travel);
```

### 3.9 `advance_to.h`: the loop and the callbacks

```cpp
class StepTry; struct StageOutcome; struct StageTable; struct GridFields;   // 3.10, 3.11
enum class KeepGoing : std::uint8_t { kYes, kStop };
struct FoamAndSprayInputs {            // Read-only views of the try, valid during the call.
  const grid::StaggeredGrid* grid;  const pressure::GeometrySnapshot* geometry;
  const grid::FaceValues<double>* velocity_after_pressure;  // The liquid set's.
  const particles::ParticlesTry* particles_try;
  double step_start_time = 0, step_duration = 0;  std::uint64_t try_number = 0;
  // NEW (render output). One entry per liquid material set; a scene has exactly one liquid set (5.2), so the
  // span has one entry, or none without liquid. Snow, springy-solid and gas sets are not passed.
  std::span<const transfer::GridSnapshot* const> after_pressure;
  const pressure::SurfaceDistances* surface_distances;  // The kAllLiquidMaterials field.
  std::span<const contact::Solid> solids;               // solids::SolidsTry::solids().
  std::uint64_t step_count = 0;                         // The number this step gets if it commits.
  grid::CellBlockRange blocks;                          // Blocks of `grid` to record (a batch: the piece's range).
  std::span<const grid::CellBlockIndex> block_in_whole_grid;  // Of `grid`'s blocks. Empty in a one-piece run. In a
                                        //   batched run the CPU copy of the piece map kept by ApplyGridPiece (A17).
  const GridFields* fields;                             // The try GridFields, region arrays included.
};
struct ResizeResult { bool resized = false; std::uint8_t module_error = 0; };  // A failed resize keeps the step.
// NEW (decision of 2026-11-14). A callback is a function and the context it receives, so cell resize, batches,
// render output and I/O each pass their own object. `function` null: not called.
template <typename Function>
struct Callback { Function* function = nullptr; void* context = nullptr; };
struct StepCallbacks {                 // D rule: named after the event or the question.
  // Stage MakeFoamAndSpray (once per batch in a batched run). Records what it needs and keeps it only if
  // on_try_done says committed. false: kCallbackFailed.
  Callback<bool(void* context, const FoamAndSprayInputs& inputs)> make_foam_and_spray;
  // Stage ChooseActiveCells: the size level of a new cell (grid Q40). Null: the nearest active cell's level.
  // Called on the CPU. The step passes committed_grid.tables(RunOn::kCpu) as `committed` (reconcile, 11.30): the
  // real tables in a CPU run; version 0 and null arrays in a one-piece GPU run (A17), where the step reads the new
  // cells back (8 B each) and sends the levels (1 B each), both counted (6.5 row O4), and
  // cell_resize::ChooseCellSize reads its own CPU copy of the committed grid; FindCpuGridCopy(...)->tables(kCpu)
  // in a batched GPU run. The step never reads a GPU grid back for it. A shared size rule instead: Open (11).
  Callback<grid::SizeLevel(void* context, const grid::GridTables& committed, Vec3 position)> choose_cell_size;
  // NEW. End of stage ParticlesToGrid, after every set was transferred (cell_resize::ApplyGridOnlyRegionInStep).
  Callback<StageOutcome(void* context, Simulation& simulation, StepTry& step_try, grid::CellBlockRange blocks)>
      apply_grid_only_region;
  // NEW. End of stage MoveParticles, whole grid (cell_resize::MoveGridOnlyLiquidInStep).
  Callback<StageOutcome(void* context, Simulation& simulation, StepTry& step_try, grid::CellBlockRange blocks)>
      move_grid_only_liquid;
  // NEW. 1/s, largest grid-only face speed over its cell width (I/O wraps cell_resize::ReadLargestGridOnlySpeed).
  // Read once in stage CommitStep and stored in CommittedStepLimits::grid_only_speed. Null: 0.
  Callback<double(void* context)> read_largest_grid_only_speed;
  // NEW. A batched run: runs stages first..last instead of the step's own loop
  // (batches::ApplyStagesInBatches). Null with either batched RunMode: kBatchedRunNotBuilt.
  Callback<StageOutcome(void* context, Simulation& simulation, StepTry& step_try, const StageTable& stages,
                        Stage first, Stage last)> run_stages;
  // After every try, committed or not. Outside the try: the simulation may be read, not stepped.
  Callback<void(void* context, const Simulation& simulation, const StepResult& result)> on_try_done;
  // Stage ResizeCells: right after a commit, before on_step_done (cell_resize::ResizeCells).
  Callback<ResizeResult(void* context, Simulation& simulation)> resize_cells;
  // After a commit and its resize. D name OnStepDone. kStop ends AdvanceTo at this committed state.
  Callback<KeepGoing(void* context, const Simulation& simulation, const StepResult& result,
                     const ChosenStepDuration& chosen)> on_step_done;
};
enum class StopReason : std::uint8_t { kReachedTarget, kCancelled, kStoppedByCallback, kRetriesUsedUp,
                                       kLimitsCannotBeRaised, kCannotContinue, kBadRequest };
struct AdvanceResult {
  StopReason stop_reason = StopReason::kReachedTarget;
  StepResult last_result{};            // The try that ended the loop.
  std::uint64_t committed_step_count = 0, rejected_try_count = 0, limit_raise_count = 0;
  std::array<std::uint64_t, kStepLimitCount> step_count_by_limit{};  // How often each StepLimit was active.
  MemoryReport memory{};
};
// Chooses durations, tries, retries shorter, raises limits, until time() == target_time exactly. NEW verb.
std::expected<AdvanceResult, StepFailure> AdvanceTo(Simulation& simulation, double target_time,
                                                    const std::atomic<bool>* cancel_requested);
```

### 3.10 `resize_access.h`: the hand-over to cell resize and batches

```cpp
struct GridFields {                    // The grid values that survive a commit. One committed, one try copy.
  grid::CellValues<double> pressure;               // Pa. SolvedArrays::pressure of the liquid solve.
  grid::CellValues<std::uint8_t> has_pressure;     // The liquid solve's cell_was_solved (ComputeWasSolved).
  grid::FaceValues<double> velocity_after_pressure;      // m/s. The liquid set. Kept or not: 6.3.
  grid::FaceValues<double> gas_velocity_after_pressure;  // NEW. m/s. The gas set. No room without gas.
  grid::CellValues<std::uint32_t> pocket_of_cell;  // AirPocketArrays::pocket_of_cell; pressure::kNoAirPocket.
  // NEW (cell resize). No room unless ReadyScene::has_cell_resize. Carried by CellId in ChooseActiveCells.
  grid::CellValues<std::uint8_t> is_grid_only_region;
  grid::CellValues<std::uint8_t> layers_from_grid_only_region;
  grid::CellValues<double> merge_ready_duration;   // s.
  grid::CellValues<double> grid_only_mass;         // kg.
  grid::CellValues<double> grid_only_rest_volume;  // m3.
  grid::CellValues<double> grid_only_current_volume;   // m3.
  Buffer<double> grid_only_ingredient_volume;      // m3. [cell place * ingredient_count + scene ingredient].
};
class ResizeTry {                      // Exclusive access to the committed state between two steps. Move-only.
 public:
  const grid::StaggeredGrid& committed_grid() const;
  const GridFields& committed_fields() const;
  particles::ParticlesTry& particles_try();        // Built from the simulation's particles.
  GridFields& fields_to_fill();                    // Try copy; bound to the resized grid by ApplyResizedGrid.
  grid::GridBuffers& free_grid_storage();          // NEW. The storage BuildGrid writes the resized grid into.
  // NEW. Of the last committed try, still valid for committed_grid().
  const pressure::GeometrySnapshot& last_geometry() const;
  pressure::SolidCoverage last_solid_coverage() const;
  const transfer::SolidsAtFaces& last_solids_at_end() const;
  double committed_step_duration() const;          // s.
  const materials::MaterialTable& material_table() const;
  std::span<const std::span<const std::uint8_t>> material_sets() const;
  const solids::Solids& solids() const;            // Committed. Solids remap nothing.
  const pressure::AirPockets& committed_air_pockets() const;
  pressure::AirPocketArrays air_pockets_to_fill();   // NEW. Try copy; first_cell_index of the resized grid.
  transfer::SolidsAtFaces& solids_at_faces_to_fill(); // NEW. Bound to the resized grid by ApplyResizedGrid.
  gas::GasAmountsTry* gas_try();                   // NEW. After ApplyResizedGrid; null without gas.
  // NEW (reconcile, 11.29). In a batched GPU run, the free half of "step: grid copy on the CPU", into which cell
  // resize builds its CPU resized grid; null in every other run.
  grid::GridBuffers* free_cpu_grid_copy_buffers();
};
std::expected<ResizeTry, StepFailure> BuildResizeTry(Simulation& simulation);  // kTryInProgress, kOutOfMemory
// NEW. Binds fields_to_fill and solids_at_faces_to_fill to resized_grid with grid::ApplyGridVersion and, with
// gas, builds the gas try with gas::BuildResizeTry(amounts, resized_grid). Fails kTooManyCellBlocks (as
// RejectContext), kRunOnCallFailed. Kept (reconcile, 11.28) for a caller that built the grid into
// free_grid_storage() itself (a CPU run, tests); a GPU resized grid is built from the uploaded list by
// BuildResizedGrid.
std::expected<void, StepFailure> ApplyResizedGrid(ResizeTry& resize_try, const grid::StaggeredGrid& resized_grid);
// NEW (reconcile, 11.28). The cell list of a resized grid, on the simulation's run_on, any order (as
// grid::GridInputs).
struct ResizedCells {
  std::span<const grid::CellId> cells;                 // A CPU run: the list in CPU memory.
  const Buffer<grid::CellId>* cell_buffer = nullptr;   // A GPU run: the list uploaded by cell resize.
  std::uint32_t cell_buffer_count = 0;                 // Entries of *cell_buffer used.
  const grid::StaggeredGrid* cpu_resized_grid = nullptr;  // A GPU run: cell resize's CPU build of the same list.
};
// NEW (reconcile, 11.28). Builds the resized grid (committed frame, version committed + 1) into
// free_grid_storage() on the simulation's run_on with grid::BuildGrid, then binds the try as ApplyResizedGrid does,
// and returns the grid for CommitResize. In a GPU run the version and block count must equal *cpu_resized_grid's
// (else kStageFailed, module kGrid). A list on the wrong processor, or a GPU run without cpu_resized_grid:
// kBadInput. Grid failures map as in stage 2 (kTooManyCellBlocks is kRaiseLimits). On failure
// free_grid_storage() is usable.
std::expected<grid::StaggeredGrid, StepFailure> BuildResizedGrid(ResizeTry& resize_try,
                                                                 const ResizedCells& resized_cells);
// NEW (reconcile, 11.29). A batched GPU run: keeps the CPU resized grid built into free_cpu_grid_copy_buffers() as
// the CPU copy that CommitResize swaps in (RejectResize releases it). kBadInput in other runs, for a grid in other
// buffers, or a version or block count different from the resized grid's.
std::expected<void, StepFailure> ApplyResizedCpuGridCopy(ResizeTry& resize_try,
                                                         grid::StaggeredGrid&& cpu_resized_grid);
// The resized grid (version = committed + 1) and the try become committed (gas::CommitTry(amounts, try,
// amounts.time()), particles::CommitTry at time()); the old grid's storage becomes the free one.
// state_version grows by one; time and step_count do not change. The caller has already regrouped the
// particles for resized_grid. In a batched GPU run it requires ApplyResizedCpuGridCopy first (else kBadInput,
// nothing committed) and swaps the CPU copy pair with no copy (reconcile, 11.29; replaces the grid::ReadGrid
// refresh, row B5 withdrawn).
std::expected<void, StepFailure> CommitResize(Simulation& simulation, ResizeTry&& resize_try,
                                              grid::StaggeredGrid&& resized_grid);
void RejectResize(Simulation& simulation, ResizeTry&& resize_try);  // Nothing changed.
```

### 3.11 `step_stages.h`: the stage table (for cell resize and batches, and for the step's own tests)

```cpp
// StageOutcome: step_error.h (3.1).
class StepTry {                        // Every try array of one step. Built and dropped by TryStepWithStages.
 public:                               // In a batched run, the loaded try over one grid::GridPiece.
  const grid::StaggeredGrid& grid() const;          // The try grid (after stage 2).
  double step_start_time() const;  double step_duration() const;  // s.
  std::uint64_t try_number() const;
  const std::atomic<bool>* cancel_requested() const;
  particles::ParticlesTry& particles_try();
  std::size_t material_set_count() const;           // In the step's set order.
  std::span<const std::uint8_t> material_is_counted(std::size_t set_index) const;
  std::optional<std::size_t> liquid_set_index() const;  // The set holding the kLiquid materials.
  transfer::GridFromParticles& grid_from_particles(std::size_t set_index);
  transfer::CellsFromParticles& cells_from_particles(std::size_t set_index);
  const transfer::SolidsAtFaces& solids_at_start() const;
  const pressure::GeometrySnapshot& geometry() const;   // Valid after stage 9.
  const GridFields& committed_fields() const;
  GridFields& fields();                             // The try copy, bound to grid().
  const materials::MaterialTable& material_table() const;
};
using StageFunction = StageOutcome (*)(Simulation& simulation, StepTry& step_try, grid::CellBlockRange blocks);
struct StageTable {                    // Indexed by Stage. Null: not a stage here.
  std::array<StageFunction, kStageCount> run{};
  // NEW (plan decision "Cell resize and batches"). The part of a stage that needs every block of the try grid at
  // once, called with blocks = every block. kFindSolidSurfaces (NEW, A17 batched-run buffers):
  // solids::FindGridEdgesOpenToSky, which floods over the whole grid; a batched run calls it at
  // kWholeGridOnProcessor right after pass 1, before stage 9's whole part. kFindLiquidSurface:
  // FindMaterialBoundaries, FindAirPockets, CheckViscosityCanRun. kMoveParticles: the move_grid_only_liquid
  // callback. kCheckStep: CheckSolidExchange, the gas changes, the next CommittedStepLimits. Null for every other
  // stage. A one-piece run calls it right after `run`. Stage 10 runs on the whole grid.
  std::array<StageFunction, kStageCount> run_whole_grid{};
};
extern const StageTable kStages;       // The production stages 1 to 20 of 5.2 (kResizeCells is recorded only
                                       //   by AdvanceTo).
// TryStep is exactly TryStepWithStages(simulation, request, kStages). Stages 1 and 2 run on the whole state;
// the ranges are then built from the try grid (one range, every block, in a one-piece run); with
// StepCallbacks::run_stages set, stages 3 to 19 run through it; stages 20 and 21 run here. Which stages run is
// fixed once per simulation (5.2 "Skipped when").
std::expected<StepResult, StepFailure> TryStepWithStages(Simulation& simulation, const StepRequest& request,
                                                         const StageTable& stages);
// NEW (batches). One entry per array instance of a StepTry (a per-set array has one entry per material set),
// generated from the memory estimate entries of 6.3 and of the modules, so WriteBatch and ReadBatch copy exactly what a
// pass reads and writes. read_by_stages, written_by_stages: bit (1 << stage).
// NEW (A17, batched-run buffers). Where the whole instance of a try array lives in a batched run (placement rule,
// 6.5). In a one-piece run every whole_place is kWholeOnProcessor (unused there).
enum class ArrayPlace : std::uint8_t { kLoadedOnly, kWholeOnProcessor, kWholeInCpuMemory };
struct StepArray {
  std::string_view row_name; CountedPer counted_per; std::uint32_t bytes_each, material_index;
  std::uint32_t read_by_stages, written_by_stages;
  ArrayPlace whole_place;              // NEW (A17, batched-run buffers).
};
std::span<const StepArray> ReadStepArrays(const Simulation& simulation);
// NEW (plan decision "Cell resize and batches"). The step builds the loaded StepTry and its GridBuffers (5.9), so
// batches reaches them through the step.
StepTry* FindLoadedTry(Simulation& simulation);                  // Null in a one-piece run.
grid::GridBuffers* FindLoadedGridBuffers(Simulation& simulation);  // Null in a one-piece run.
// Points loaded_try.grid() at piece.grid() and binds every loaded array to it with grid::ApplyGridVersion.
// kTooManyCellBlocks when the piece has more blocks than the loaded arrays hold. Amended (A17, batched-run
// buffers; replaces the two-argument form): block_in_whole_grid is the batched run's CPU copy of the piece map
// (read back once per load by batches), kept for FoamAndSprayInputs::block_in_whole_grid of stage 18; never the
// empty span of a GPU piece.
std::expected<void, StepFailure> ApplyGridPiece(StepTry& loaded_try, const grid::GridPiece& piece,
                                                std::span<const grid::CellBlockIndex> block_in_whole_grid);
struct BlockCopy { grid::CellBlockIndex from_block; grid::CellBlockIndex to_block; std::uint32_t block_count; };
// Copies array instance `array_index` (an index into ReadStepArrays) block by block between two tries, on
// whichever processors they live (base/ slice copies). An array counted per particle follows the particle groups
// of the blocks in each try. kBadInput when a block lies outside either try's grid.
std::expected<void, StepFailure> WriteStepArrayBlocks(const StepTry& from, std::size_t array_index,
                                                      std::span<const BlockCopy> copies, StepTry& to);
// NEW (A17, batched-run buffers). The separate SolidsBuffers of stage 17 in a batched run (run_on = kCpu, the whole
// particle limit, max_cell_blocks; entries "solids: ", all where = kCpu), which batches::ReadBatch fills with
// solids::WriteBlockPartialsRange after pass 1. Null in a one-piece run.
solids::SolidsBuffers* FindContactReactionBuffers(Simulation& simulation);
```

Step-owned operations with `_cpu.cc` and `_gpu.cu` beside one header each, reached only through the table:
`add_new_particles.h`, `choose_active_cells.h`, `apply_forces.h`, `move_particles.h`, `check_step.h`,
`commit_step.h`. Their shared formulas are in `step_duration_math.h`, `apply_forces_math.h`
(`ComputeVelocityAfterGravity`), `travel_math.h` (`ComputeTravelInCellWidths`, `FindDomainSideCrossed`) and
`random_words.h` (`ComputeRandomWords`, the counter-based generator of `main`). Plan decisions ("Step"):
`struct RandomWords { std::uint32_t word[4]; }`, `ComputeRandomWords(std::uint64_t child_number, std::uint64_t
source_id, std::uint64_t key)` (Philox 4x32-10), `ComputeSignedUnitValue(std::uint32_t word)`, shared, also used by
foam and spray; `struct DomainSideCrossed { contact::DomainSide side; std::uint8_t crossed; }` from
`FindDomainSideCrossed`. Step-owned try entries (A18, 6.3): `"step: material face velocity after pressure"` (one
per blocking-material face) and `"step: sealed gas pressure"` (one per gas blob), the spans stage 12 passes as
`SolvedArrays::blocking_material_velocity_after_pressure` and `sealed_gas_pressure`, and stage 11 passes as
`viscosity::SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure}` (A24). NEW (A26,
A27; 6.3): `"step: material face velocity after solve"` (one per `solids::ViscousMaterialFace`) and `"step: free
solid motion"` (one `solids::FreeSolidMotion` per free solid touching liquid), the third and fourth spans of stage
11's `SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure,
material_face_velocity_after_solve, free_solid_motion}`; and `"step: free solid motion before pressure"` (6 doubles
per free solid, stage 10's `EquationInputs::free_solid_motion_before_pressure`; empty for stage 12).

## 4. Consumes from other modules

EXISTS: in Foundation, including the `base/` additions. PROVIDED: by the named design, with the name its
section 3 has (for I/O and CLI and the Blender add-on, as `changes-to-apply.md` states it).

**Foundation (EXISTS):** `Vec3`, `Mat3`, `Axis`, `kUp`, `math::Dot`, `RunOn`, `CheckCanRunOn`,
`FLUIDITY_CPU_AND_GPU`, `cpu_only.h`, `std::expected`. **Additions (EXISTS, `shared-interface.md` (a) and
`changes-to-apply.md`):** `Buffer`, `BuildBuffer` with `CpuMemory { kOrdinary, kLockedForCopies }`,
`WriteBuffer`, `ReadBuffer` and their `(first, count)` slice forms, `CopyBuffer(from, to)` and its slice form
`CopyBuffer(from, first, count, to, to_first)` (for `ApplyLimits`), `MemoryAvailable { total_bytes;
free_bytes; }` and `std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn)` (a total of 0
means unknown), `ReadProcessorDescription`,
`PerParticleVelocity<T> { T for_next_step; T for_this_step; }`, `math::Sqrt`, `Abs`, `Max`, `Min`, `IsFinite`,
`math::TreeSum`, `ComputeLargestMagnitude`, `BuildStableOrder`, `MemoryEstimateEntry`, `CountedPer`, `KeptFor`,
`MemoryCounts`, `ComputeMemoryBytes`. **As built (plan decisions "Foundation `base/`" and "Foundation as built"):**
`RunOn` and `RunOnError` in the shared `base/run_on.h`, `CheckCanRunOn`, `ToMessage(RunOnError)`,
`ToName(RunOnError)` in the CPU-only `base/check_can_run_on.h`; `MemoryEstimateEntry::where` (`MemoryWhere { kRunOn, kCpu
}`) and `ComputeMemoryBytes(rows, counts, where)`, which the estimate uses to sum the processor and CPU entries
apart; `MemoryAvailable`, `ReadMemoryAvailable` and `ProcessorDescription` in `base/processor_description.h`;
`BufferError::kRunOnMismatch` (mapped to the step's `kBadInput`); `kUp` as `inline constexpr Vec3 kUp{0, 0, 1}`.

**Contact (PROVIDED, 3.1 to 3.5):** `DomainBox`, `ClosedSides`, `DomainSide`, `Solid`, `PlateSide`,
`CheckSolid`, `ComputeContactScale`, `ComputeContactTolerance`, `CheckContactTolerance`, `kMaxTurnPerStep`,
`ParticleContact`, `SolidReaction` (`velocity_for_next_step_change`), `ContactBuffers`, `BuildContactBuffers`,
`MoveParticlesInputs`, `MovedParticles` (`velocity_for_next_step`), `MoveParticlesResult`, `MoveParticles`,
`ContactError` (`kTooCloseToCall`, `kStartsInsideSolid`, `kOnWrongSideOfPlate`, `kSolidTooThin`),
`ContactCloseCall`, `ContactFailure`, `ToName`, `BuildContactMemoryEstimate`.

**Grid (PROVIDED, 3.1 to 3.7):** `StaggeredGrid` (move-only; `version()`, `counts()`, `all_blocks()`,
`tables()`, `max_cell_blocks()`), `GridBuffers`, `BuildGridBuffers(const GridSettings&)`,
`BuildGrid(const GridInputs&, GridBuffers&)`, `ReleaseBuffers(StaggeredGrid&&)`, `GridInputs`, `GridSettings`,
`ComputeGridCounts`, `GridCounts`, `CheckSizeStepRule`, `GridFrame`, `GridTables`, `GridVersion` (1 or more for
a built grid), `CellId`, `CellBlockId`, `FacePieceId`, `CellBlockIndex`, `SizeLevel`, `ToCellId`,
`ToCellBlockId`, `FindCellAt`, `FindCellPlace`, `FindFacePlace`, `FindTouchingCells`, `ComputeCellWidth`,
`ToSizeLevel`, `CellBlockRange`, `BuildRanges`, `CellValues`, `FaceValues`, `CornerValues`, `CellBlockValues`,
`BuildCellValues(RunOn, std::uint32_t max_cell_blocks)` and the face, corner and block forms,
`ApplyGridVersion(values, grid)` (all four forms), `ReadValues`, `WriteValues`, `ToMemoryCounts`,
`BuildGridMemoryEstimate(RunOn, std::uint32_t storage_count)` (the step passes 2), `GridError`
(`kSizeStepRuleBroken`, `kTooManyCellBlocks`), `GridFailure`, `ToName`. **A17 (plan decision "Grid built on the
GPU"):** `GridInputs::active_cell_buffer` (`const Buffer<CellId>*`) and `active_cell_buffer_count`, the list on
the buffers' processor; `BuildGrid` builds on the processor of its `GridBuffers` and reads back only its status
record; the buffers forms `ComputeGridCounts(const GridInputs&, GridBuffers&)` and `CheckSizeStepRule(const
GridInputs&, GridBuffers&)`; `ReadGrid(const StaggeredGrid&, GridBuffers& cpu_buffers)` (the one explicit copy;
`tables(RunOn::kCpu)` of a GPU grid is empty); `GridPiece`, `CellBlockRange` in `grid/grid_ids.h`. **A17,
batched-run buffers:** the CPU builder (`BuildGrid` with CPU `GridBuffers`) for the CPU grid copy of a batched GPU
run, from the same list; `BuildGridMemoryEstimate(RunOn::kCpu, 2, ScratchSharing::kOneForAllBuffers)` for its
entries (`where` set to `kCpu`). (Reconcile: `ReadGrid` no longer refreshes it after a committed resize, 11.29.)
**Reconcile item 11 (11.31):** `GridScratch` (move-only; `max_cell_blocks()`, `run_on()`), `BuildGridScratch(RunOn,
std::uint32_t max_cell_blocks)`, `BuildGridBuffers(const GridSettings&, GridScratch&)` (the scratch lent, not
moved while lent, outliving the buffers), `enum class ScratchSharing { kOnePerBuffers, kOneForAllBuffers }`,
`BuildGridMemoryEstimate(RunOn, std::uint32_t storage_count, ScratchSharing = kOnePerBuffers)` (with
`kOneForAllBuffers` the `grid: build scratch:` entries have `copies = 1`), `kMaxCellBlocksLimit`;
`GridSettings::max_cell_blocks` defaults to 0, and `BuildGridBuffers` or `BuildGridScratch` with 0 is `kBadInput`.

**Particles (PROVIDED, 3.1 to 3.7):** `Particles`, `ParticlesSettings`, `MaterialParticlesSettings`,
`BuildParticles`, `BuildParticlesFromSnapshot(const ParticlesSettings&, const ParticlesSnapshot&, const
grid::StaggeredGrid&)`, `ApplyParticleLimits`, `ParticlesTry`, `BuildTry`, `CommitTry`, `RejectTry`,
`BuildFieldsToRead`, `BuildFieldsToChange`, `ParticleFields`, `ParticleFieldsToChange`, `ArrayToChange`,
`Field` (`kVelocityForNextStep`, `kVelocityForThisStep`), `FieldList`, `ToFieldList`, `NewParticles`,
`AddParticles`, `RemoveParticles`, `AddReason`, `RemoveReason` (`kLeftDomain`, `kThroughOutlet`,
`kMarkerNotNeeded`), `GroupingSettings`, `CheckGrouping`, `GroupParticlesByBlock`, `CheckParticles(const
ParticlesTry&, grid::CellBlockRange)`, `ParticleAmounts`, `ComputeAmountsPerBlock`, `ParticlesSnapshot`,
`SnapshotFields`, `kOutputSnapshotFields`, `ReadSnapshot`, `kCarriesFittedChange`, `kHasOwnMotion`,
`ParticlesError`, `ParticlesFailure`, `ToName`, `BuildParticlesMemoryEstimate`. **A17, batched-run buffers:**
`AddParticlesResult::copy_count`, `bytes_copied` (written to the processor; 0 when the particles are in CPU
memory), added to `StepReport::copies`.

**Transfer (PROVIDED, 3):** material sets (A8): `MaterialSetKey`, `ToMaterialSetKey`, `CheckMaterialSets`
(called in `CheckSceneCanRun`); `ParticlesToGrid`, `ParticlesToGridInputs`, `GridFromParticles`,
`FaceValueKind` (`kFromParticles`, `kFromGridOnlyRegion`), `SaveGridBeforeForces`, `SaveGridAfterPressure` (with
`was_solved` a pointer, null for a set no solve writes), `SaveGridInputs`, `GridSnapshot`, `GridToParticles`,
`GridToParticlesInputs` (with `cell_was_solved` a pointer), `GridAtParticle`, `kGridOnlyReasonCount`,
`MixParticleVelocity`, `MixInputs`, `MixResult`, `SolidsAtFaces` (`outside_solid`, `side`, `solid_velocity`),
`CellsFromParticles`, `CellsFromParticlesInputs`, `ComputeCellsFromParticles`, `ComputeMixtureDensity`,
`ComputeSensedVolumeRatio`, `TransferSettings`, `TransferError` (`kNotFinite`, `kTooManySidesAtFace`,
`kWrongMaterialSet`), `ToName`, `BuildTransferMemoryEstimate(ingredient_count, material_count,
material_set_count, sets_without_solve)`.

**Pressure (PROVIDED, 3):** `BuildPressureBuffers`, `BufferSizes` (`max_boundary_materials`,
`has_viscous_ingredient`), `PressureBuffers` (two instances: liquid and gas), `ComputeDistanceToSurface`,
`SurfaceDistanceInputs`, `kAllLiquidMaterials`, `SurfaceDistances`, `FindLiquidSurface`, `LiquidSurfaceInputs`,
`GeometrySnapshot` (`surface_tension_step_limit`), `SolidCoverage`, `FindMaterialBoundaries`,
`MaterialBoundaryInputs`, `FindAirPockets(const AirPocketInputs&, AirPocketArrays, PressureBuffers&)`,
`AirPocketInputs` (`holds_tracked_gas`), `AirPocketArrays`, `AirPockets`, `AirPocket`, `BlobHistory`,
`kNoAirPocket`, `AirPocketSettings`, `OpenSeaSettings`, `ReadyOpenSea`, `BuildOpenSea`, `ComputeOpenSeaWave`,
`ComputeCarriedPressure`, `PressureLawKind`, `ComputeVolumeCorrectionRate`, `VolumeCorrectionSettings`,
`EquationInputs` (`gas_pressure_above_atmosphere`, `holds_tracked_gas`, `free_solid_pushes`,
`free_solid_inverse_inertia`, `sealed_blob_check`; A27: `free_solid_motion_before_pressure`, 6 doubles per free
solid, `V` then `Omega` in `FreeSolidPush::solid_index` order, size 0 or 6 x the free solid count, else
`kBadInput`), `SealedBlobCheck`, `BuildPressureEquations`,
`PressureEquations`, `SolvePressure(const PressureEquations&, const SolveSettings&, const std::atomic<bool>*
cancel_requested, SolvedArrays, PressureBuffers&)`, `SolvedArrays`, `SolveResult`, `SolveSettings`,
`FlowImbalanceReport`, `ComputeWasSolved`, `PressureError` (`kCancelled`, `kCellSplitBySolid`, ...; no
`kMaterialBoundaryNotSupported`), `PressureFailure`, `ToName`, `BuildPressureMemoryEstimate`. **A18 (plan decision
"Coupling inside the pressure solve"):** `BlockingMaterialFace`, `SealedGasBlob`; `EquationInputs` fields
`blocking_material_faces`, `sealed_gas_blobs`, `sealed_gas_blob_of_cell`, `gas_pressure_above_background`;
`SolvedArrays` fields `blocking_material_velocity_after_pressure`, `sealed_gas_pressure`; `BufferSizes` fields
`max_blocking_material_faces`, `max_sealed_gas_blobs`. **A15 (plan decision "Batched solves"):**
`BufferSizes::max_loaded_blocks`; `pressure/solve_batches.h`: `SolveBatches`, `FindSolveBatches(const
PressureBuffers&)` (also the viscosity and gas forms; const, so the step never resets through it),
`ReadSolveCopyReport`, `SolveCopyReport`; `WriteSnapshotRange` (called by batches, not by the step). **Reconcile
item 3 (11.32):** `void pressure::ResetSolveCopyReport(PressureBuffers& buffers)` (zeroes the held `SolveBatches`'
report; nothing for one-piece buffers; allocates and copies nothing), called by the step per try. **A24:** no
piece of `coupled_solve_pieces.h` returns `kBadInput` because the equations hold a material face or a sealed gas
blob (viscosity calls them).

**Viscosity (PROVIDED, 3):** `ViscositySceneSummary` (`has_viscous_ingredient`, filled by the step by the A25
rule; A24: no `has_snow`), `CheckViscosityCanRun`,
`ViscosityBufferSettings`, `BuildViscosityBuffers`, `ViscosityBuffers`, `ViscositySettings`, `ViscosityInputs`
(`cells_from_particles` as a span with one entry, the one liquid set), `SolveViscosityAndPressure(inputs, settings,
cancel_requested, buffers, velocity_after_pressure, cell_pressure, SolvedCouplingArrays coupling_arrays = {})`,
`ViscosityResult`, `ViscosityError` (`kCancelled`, ...; A24: no `kSnowNotSupported`; A27: no
`kFreeSolidNotSupported`, and `ViscositySceneSummary` has no `has_free_solid_touching_liquid`),
`ViscosityFailure`, `ToName`, `BuildViscosityMemoryEstimate`;
`ViscosityInputs::pressure_buffers` (set by the step); `ViscosityBufferSettings::max_loaded_blocks` (A15).
**A24:** `struct SolvedCouplingArrays { std::span<double> blocking_material_velocity_after_pressure; std::span<double>
sealed_gas_pressure; }` (sizes equal the equations' counts, else `kBadInput`);
`ViscosityBufferSettings::max_sealed_gas_blobs` (>= the equations' `sealed_gas_blob_count`, else `kBadInput`; the
step sets gas's `max_gas_blobs`, 0 without gas). **A26, A27:** `ViscosityInputs` gains `std::span<const
solids::ViscousMaterialFace> material_faces` and `std::span<const solids::FreeSolidForViscousSolve> free_solids`;
`SolvedCouplingArrays` gains `std::span<double> material_face_velocity_after_solve` (one per `ViscousMaterialFace`)
and `std::span<solids::FreeSolidMotion> free_solid_motion` (one per free solid),
`blocking_material_velocity_after_pressure` staying filled for the `BlockingMaterialFace` subset; `ViscosityBufferSettings` gains `max_material_faces`
and `max_free_solids` (the step sets 2 x solids' `max_motion_map_entries`, 0 without a blocking set, and the free
solid count); memory entries `"viscosity: material face vectors"`, `"viscosity: free solid vectors and blocks"`,
`"viscosity: sample solid coefficients"`. **A25** (`deformation_rate.h`, shared):
`kSmallestVisibleViscousEffectPerStep = 2.5e-4`, `ComputeViscousEffectPerStep(double viscosity, double
step_duration, double density, double cell_width)`. **Reconcile item 3 (11.32):** `void
viscosity::ResetSolveCopyReport(ViscosityBuffers& buffers)`.

**Materials (PROVIDED, 3):** `MaterialTableInputs`, `MaterialTable` (`materials()`,
`first_ingredient_of_material()`, `ingredient_viscosity()`, `ingredient_friction()`,
`ingredient_pressure_law()`, `gas_ingredient_index()`, `ingredient_rest_density()`, `surface_tension_max()`,
`surface_tension_with_air()`, `lightest_density()`), `BuildMaterialTable`, `FindSurfaceTensionBetween`,
`MaterialKind { kLiquid, kGas, kSnow, kSpringySolid }`, `AtSolidFaces`, `HasInternalForce`, `PlatesInScene`,
`BuildParticlesSettings`, `MaterialSettings`, `MaterialBuffers`, `BuildMaterialBuffers`,
`ComputeInternalForces` with `InternalForcesResult::step_limit`, `ComputeParticleFriction`,
`ApplyInternalForce(..., std::span<InternalForceTotals> totals_of_block)`, `InternalForceTotals`,
`ComputeInternalForceTotal`, `ComputeCellSqueezeValues`, `ComputeOwnMotionShares`, `OwnMotionInputs`,
`ApplyMaterialResponse`, `MaterialResponseInputs`, `MaterialResponseResult`, `CheckMaterialState`,
`MaterialsError`, `MaterialsFailure`, `ToName`, `BuildMaterialsMemoryEstimate`. The step calls them in the order
materials 4 gives (5.2). No `CheckMaterialsCanRun` and no `ComputeStepLimit` are needed: `kSurfaceTensionsDiffer`
comes from `BuildMaterialTable` in `CheckSceneCanRun`, and the step keeps the `step_limit` of the last committed
try (5.8).

**Solids (PROVIDED, 3):** `SolidsSettings`, `SolidSettings`, `MaterialTheLiquidCannotEnter`, `Solids`,
`BuildSolids(const SolidsSettings&)`, `SolidsSummary`, `ComputeSolidsSummary`, `CheckSolidsCanRun(const
SolidsSettings&, double smallest_cell_width, double largest_cell_width)` returning every broken rule,
`kMaxPlateCount`, `kMaxFreeSolidCount`, `ComputeStepDurationLimit`, `SolidsTry`, `BuildTry(solids,
step_duration, smallest_cell_width)`, `CommitTry` (fallible), `RejectTry`, `SolidsSnapshot`, `ReadSnapshot`,
`BuildSolidsFromSnapshot`, `SolidsOnGridInputs`, `ComputeSolidsAtFaces` (both forms), `SolidCoverageValues`,
`ToSolidCoverage`, `ComputeSolidCoverage`, `FindGridEdgesOpenToSky`, `MotionMap`, `MotionMapInputs`,
`BuildMotionMap`, `ToFreeSolidPushes`, `ToFreeSolidInverseInertia`, `PressureOnSolids`,
`ApplyPressureToSolids`, (A18) `BlockingSet`, `BlockingMaterialFacesInputs`, `BlockingMaterialFaces`,
`BuildBlockingMaterialFaces`, `ToBlockingMaterialFaces`, `PressureOnMaterials`, `ApplyPressureToMaterials(const
BlockingMaterialFaces&, std::span<const double> velocity_after_pressure, std::span<const BlockingSet>,
SolidsBuffers&)` (`MaterialPushFields` and the particle form are removed), (A26, `viscous_material_faces.h`)
`ViscousMaterialFace`, `ViscousMaterialFaces`, `BuildViscousMaterialFaces(const BlockingMaterialFacesInputs&, const
BlockingMaterialFaces& pressure_faces, const pressure::GeometrySnapshot& geometry, SolidsBuffers&)`,
`ToViscousMaterialFaces`, `ApplyViscousSolveToMaterials(const ViscousMaterialFaces&, std::span<const double>
velocity_after_solve, std::span<const BlockingSet>, SolidsBuffers&)` returning `PressureOnMaterials`, (A27,
`motion_map.h`) `FreeSolidForViscousSolve`, `ToFreeSolidsForViscousSolve(const MotionMap&)`, `FreeSolidMotion`,
`ApplySolvedMotionToSolids(const MotionMap&, std::span<const FreeSolidMotion> motion_after_solve, const
grid::CellValues<double>& pressure, SolidsTry&, SolidsBuffers&)` (travel and turn limits checked again,
`kSolidMovesTooFar`), `ReactionFields`, `ContactOnSolids`,
`ApplyContactReactions`, `MaterialSetFaces`, `CollideSolidsInputs`, `CollideSolidsResult`, `CollideSolids`,
`DriveWavesInputs`, `DriveWavesResult` (`momentum_added_of_block`, `energy_added_of_block`), `DriveWaves`,
`PlateSideFields`, `ApplyPlateSides`, `ApplyPlateRemoval`, `CutWorkFields`, `ComputeCutWork`, `ExchangeInputs`,
`ExchangeTolerances`, `CheckSolidExchange`, `SolidsBuffers`, `BuildSolidsBuffers`, `SolidsError`,
`SolidsFailure`, `ToName`, `BuildSolidsMemoryEstimate` (which include the step-owned `SolidsAtFaces`,
`SolidCoverageValues` and `grid_edge_is_open_to_sky`). The step owns those arrays and the `SolidsBuffers`.
**A17, batched-run buffers:** `BuildSolidsBuffers(settings, max_particle_count, max_cell_blocks)` called with the
loaded counts (the loaded instance), with 0 particles (the whole instance) and with `run_on = kCpu` (the stage-17
instance); `WriteBlockPartialsRange(const SolidsBuffers& piece_buffers, const grid::GridPiece& piece, SolidsBuffers&
whole_buffers)` (called by batches, not by the step; the step's test calls it through a hand-written
`run_stages`); every solids call copies its per-solid results into `SolidsTry` (CPU) before it returns;
`FindGridEdgesOpenToSky` on a GPU grid (box marks, 1 B per box each way, not counted).

**Gas (PROVIDED, 3):** `GasLaw`, `GasSettings`, `MarkerSettings`, `ActiveGasSettings`, `GasSource`,
`GasWind`, `CheckGasSources`, `GasAmounts`, `GasAmountsTry`, `BuildGasAmounts`, `BuildTry(amounts, grid)`,
`CommitTry`, `RejectTry`, `BuildResizeTry`, `GasSeenByLiquid`, `ComputeGasSeenByLiquid`, `FindMarkerChanges`,
`MarkerChanges`, `FindCellsToKeepActive`, `GasBufferSizes` (A15: `max_loaded_blocks`), `GasBuffers`,
`BuildGasBuffers`, (A18) `GasStepStartInputs`, `SealedGasForLiquid`, `ApplyGasStepStart`,
`SolveGasInputs` (A18: `sealed_gas_blobs`, `sealed_gas_pressure`; A24: no `sealed_gas_coupling`, and
`SealedGasCoupling` and `ComputeSealedGasStepLimit` are deleted),
`SolvedGasArrays`, `SolveGas(inputs, settings, cancel_requested, outputs, amounts_try, gas_buffers,
pressure_buffers)`, `GasResult` (`step_limit`: the flow-share limit only, A24; `changes`, `flow_imbalance`,
`iteration_count`), `gas::ResetSolveCopyReport(GasBuffers&)` (reconcile item 3, 11.32),
`GasAmountChanges`, `GasAmountsSnapshot`, `ReadGasAmountsSnapshot`, `BuildGasAmountsFromSnapshot`,
`GasSceneSummary`, `CheckGasCanRun`, `GasError` (`kStepTooLongForGasFlow`, `kCancelled`,
`kPressureSolveFailed`, ...), `GasFailure`, `ToName`, `BuildGasMemoryEstimate`. **A17, batched-run buffers:**
`MarkerChanges` entries are `where = kCpu` in every GPU layout and `MarkerChanges` gains `copy_count`,
`bytes_copied` (the GPU form's read-back of its two lists; 0 on the CPU); the CPU form `struct MarkerGridInputs {
const grid::StaggeredGrid* grid; const grid::CellValues<double>* gas_volume; const grid::FaceValues<double>*
committed_velocity; };` with `FindMarkerChanges(const MarkerGridInputs& inputs, grid::CellBlockRange blocks, const
particles::ParticleFields& gas_markers, const GasLaw& law, const MarkerSettings& settings, GasBuffers& buffers)`
(every input in CPU memory, else `kBadInput`), used in both batched modes; `FindCellsToKeepActive` runs on the
amounts' processor.

**Cell resize and batches (PROVIDED, 3):** reached only through `StepCallbacks`, which the I/O and CLI design
sets: `cell_resize::ResizeCells` (`resize_cells`), `ChooseCellSize` (`choose_cell_size`),
`ApplyGridOnlyRegionInStep` (`apply_grid_only_region`), `MoveGridOnlyLiquidInStep` (`move_grid_only_liquid`),
`ReadLargestGridOnlySpeed` (wrapped as `read_largest_grid_only_speed`), `batches::ApplyStagesInBatches`
(`run_stages`); `batches::BuildBatchPlan` fills `RunPlan::max_blocks_per_batch`, `max_loaded_blocks`,
`max_loaded_particles` and (A15) `max_solve_loaded_blocks`, and its `SolvePlace::kInPieces` becomes
`RunMode::kBatchedWithSolvesInPieces`; `BuildCellResizeMemoryEstimate` is passed to `BuildRunPlan`. **A17,
batched-run buffers:** `batches::ReadBatch` calls `solids::WriteBlockPartialsRange` into
`FindContactReactionBuffers`'s instance after pass 1, calls stage 3's `run_whole_grid` after pass 1, and passes its
CPU copy of each piece map to `ApplyGridPiece`; `cell_resize::ResizeCells` reads `FindCpuGridCopy` in a batched GPU
run instead of its own `ReadGrid`; batches' `kStagePlaces` gives the same `whole_place` for every array (batches
test). **Reconcile (11.28 to 11.30):** cell resize calls `BuildResizedGrid` on a GPU grid (batches Task 24); in a
batched GPU run it builds its CPU resized grid into `ResizeTry::free_cpu_grid_copy_buffers()` and hands it over
with `ApplyResizedCpuGridCopy` (`CellResizeSettings::grid_copies = GridCopies::kHeldByStep`: its own two CPU grids
are neither allocated nor counted); `ChooseCellSize` reads its own CPU copy when `committed` has version 0 (a
one-piece GPU run). The step includes no header of this design.

**Surface, foam and spray (PROVIDED, 3):** reached through `make_foam_and_spray` and `on_try_done`
(`render_output::RecordStep`, `CommitRecordedStep`, `RejectRecordedStep`, composed by I/O);
`BuildRenderOutputMemoryEstimate` is passed to `BuildRunPlan` as `output_estimate`.

**I/O and CLI (PROVIDED, `changes-to-apply.md` 12):** fills `ReadyScene` (material sets, keyed gas sources and
wind, blocking materials, `has_cell_resize`; `io::RunScene` owns every vector and key array the ready scene
borrows), sets each `StepCallbacks` pair (`RunParts`), names the step's enums in `io/stable_names.h` with
`kStageCount`, `kRejectReasonCount`, `kCapabilityReasonCount`, `kSuggestedChangeCount`, `kStepLimitCount`, calls
Foundation's `ReadMemoryAvailable` for the two `MemoryAvailable` inputs of `BuildRunPlan`, stores `SimulationSnapshot` in `saved_state`.
**A17 planned cost:** I/O's `BuildRunBudget` reads `RunPlan::fit`, `estimate.on_processor`, `on_cpu`,
`ModuleBytes::on_cpu` and `planned_step_copies`; its `ApplyTryToRunCost` reads `StepReport::copies` and timings;
`BuildRun` fails `kDoesNotFit` on a plan that does not fit.

**Per material, the step carries:** liquids and mixtures: the liquid set, nothing beyond the calls; viscous
liquid (viscosity above 0, A25): stage 11 in place of 12 for the whole run; snow and springy solids: one set each,
the internal force before `ParticlesToGrid`, its push in `ApplyForces`, the sound limit, `CollideSolids`, blocking
coverage, and (A18) its faces in the liquid solve (stage 10) with the solved face velocity written back in stage 12,
or in stage 11 beside a viscous liquid (A24); moving solids: `SolidsTry` in the same commit; gas: its own set, `GasAmountsTry` in the
same commit, markers in stage 1, cells in stage 2, `ApplyGasStepStart` in stage 10 (or at the start of 13
without liquid), stage 13.

## 5. Method

### 5.1 The 22 methods, grouped by what they add

| Group | Methods on `main` | What they add | Where it goes |
|---|---|---|---|
| A. Cell sizes change before the step (3) | `AdaptiveFamily`, `AdaptiveRefinementClosure`, `AdaptiveWidth` | Which cells split or merge, with caller-supplied evidence | `ResizeCells` after a commit (cell resize); the rule is a persistent choice |
| B. Named particles change (3) | `AdaptiveQuadraticTransfer`, `AdaptiveQuadraticSplit`, `AdaptiveQuadraticMerge` | Another transfer for one particle; a split; a merge | One transfer for all (transfer); `AdjustParticleCount` in the resize try |
| C. Liquid moves between particle and grid-only regions (5) | `AdaptiveOwnerExchange`, `AdaptiveBandOverlap`, `AdaptiveBandPlacement`, `AdaptiveBandRetirement`, `AdaptiveCoreReseed` | Which region carries a cell | Region transfer (cell resize) |
| D. Geometry given per try (1) | `AdaptivePhysicalGeometry` | Corner distances, outside pressures and added volume per cell from the caller | Solids fill `SolidCoverage`; the scene gives added volume |
| E. A solid that exchanges force (9) | `PlanarPistonPressureLoad`, `DynamicPlanarPiston`, `CoupledPlanarPiston`, `GuidedFloatingBox`, `TranslatingFloatingBox`, `CoupledGuidedFloatingBox`, `OrientedFloatingBoxPressure`, `OrientedFloatingBox`, `CoupledOrientedFloatingBox` | A solid's state passed by value and returned; which motions are free; coupled or not | `solids::SolidsSettings` in the scene; state in `SolidsTry`; one coupling |
| F. A second material (1) | `Snow` | Snow particles kept by the caller | A material of the scene in its own material set, in the same try |

Result: no group needs a field in `StepRequest`. Each is either a persistent choice of the scene (D, E, F) or
a change of representation made from the committed state after the step (A, B, C). The request keeps
`step_duration`, gains `state_version` (every one of the 12 "adaptive" methods checked a committed epoch) and
`cancel_requested`. The three optional parts of the naming review's sketch (`cell_resize`, `particle_change`,
`solid`) are not kept (decision 1). The four `AdvanceFrame...` loops become `AdvanceTo`.

### 5.2 The stages

**Material sets (A8).** A set is the materials that share one grid velocity; materials that do not mix are in
different sets. The scene gives them (`ReadyScene::material_sets`); the step's set order is ascending lowest
counted `material_index`. Every transfer stage and `ComputeCellsFromParticles` runs once per set, in that order,
with the set's `material_is_counted`; each set has its own `GridFromParticles`, `CellsFromParticles` and two
`GridSnapshot`. All liquid materials are in one set, because pressure's `EquationInputs` take one
`velocity_before_pressure`; two liquids that do not mix in it are separated by pressure's material boundary
(C4). The gas material is a set of its own, so markers never enter the liquid's sums (A10). Each snow or
springy-solid object is a set of its own (or a group the artist chooses).

**The scene of this list.** Materials (`material_index`): 0 water (`kLiquid`), 1 syrup (`kLiquid`, a viscous
ingredient), 2 snow (`kSnow`), 3 jelly (`kSpringySolid`), 4 smoke (`kGas`). Sets in the step's order: `L` =
{0, 1}, mask `{1,1,0,0,0}`; `S` = {2}; `E` = {3}; `G` = {4}. Solids: the domain box (top open), a fixed plate
cutting the jelly, a driven wave paddle, a free box; snow and jelly are blocking materials; an open-sea zone on
one side; water and syrup form a `MaterialPair`. "For X in sets" means `L, S, E, G` in that order; a set's mask is
`mask(X)`, its arrays `X.grid_from_particles`, `X.cells_from_particles`, `X.before_forces`, `X.after_pressure`.
`all` is `grid.all_blocks()`; `blocks` is the stage's range (every block in a one-piece run). `fields` is the
try `GridFields`, `committed` the committed one. This exact scene runs as written, with or without the snow (A24:
stage 11 carries the faces of snow and jelly and the sealed gas blobs; A26: the band faces of snow and jelly are
velocity unknowns of the viscous solve, so the syrup sticks to them), and with the free box anywhere, beside the
syrup included (A27: the free box's linear and angular velocity are unknowns of the viscous solve). Replaced
(A27): "it needs the free box away from the syrup (`kViscousLiquidWithFreeSolid`: the free solid's velocity would
be an unknown of the viscous velocity solve, which A24 does not cover)". The list is written for it so that every
call has its place. With
syrup's viscosity at zero (A25: a liquid ingredient is viscous exactly when its viscosity is above 0), stage 12
runs in place of 11 for the whole run.

Before stage 1, `TryStep` checks the request (`kBadStepDuration`, `kWrongStateVersion`, `kTryInProgress`,
`kSimulationUnusable`), reads `cancel_requested`, and calls `particles::BuildTry(particles)`. Each later stage
first reads `cancel_requested` (5.10). Each stage runs on every range given, in range order, and writes only
values stored in the range's blocks; "whole grid" stages take `all`. Everything written is a try copy or
scratch; no stage receives a writable pointer reachable from committed values.

1. **`AddNewParticles`** (whole state). (a) Sources, ascending `source_id` (5.4): per material, one
   `particles::AddParticles(particles_try, m, new_particles, AddReason::kFromSource)`. (b) Open sea:
   `pressure::ComputeOpenSeaWave(ready_open_sea, site, time)` per lattice site; then
   `particles::RemoveParticles(particles_try, open_sea_material_index, above_surface, RemoveReason::kThroughOutlet)`
   and `particles::AddParticles(particles_try, open_sea_material_index, refill, AddReason::kFromSource)`.
   (c) Gas: `gas::FindMarkerChanges(gas_amounts, committed_grid, all, particles::BuildFieldsToRead(particles_try,
   4, all), committed.gas_velocity_after_pressure, gas->law, gas_markers, gas_buffers)`, then
   `particles::RemoveParticles(particles_try, 4, changes.to_remove, RemoveReason::kMarkerNotNeeded)` and
   `particles::AddParticles(particles_try, 4, changes.to_add, AddReason::kRefillRegion)`. In both batched modes
   (A17, batched-run buffers) the CPU form `gas::FindMarkerChanges(MarkerGridInputs{FindCpuGridCopy(simulation),
   &volume_mirror, &velocity_mirror}, all, markers, law, settings, gas_buffers)` on the committed-value mirrors
   (6.5). Every `AddParticles` and `FindMarkerChanges` copy count goes into `StepReport::copies`.
2. **`ChooseActiveCells`** (whole state; 5.3). (a) Mark the needed blocks: particles, waiting particles, the
   open-sea zone, committed grid-only cells, and `gas::FindCellsToKeepActive(gas_amounts, committed_grid, all,
   gas_sources, gas->law, gas_active_cells, keep_active)`; the step adds every cell touching a marked cell and
   the cells of each active gas source box (plan decision "Gas"). (b) If the list differs:
   `choose_cell_size` for each new cell, then `grid::BuildGrid(GridInputs{grid_frame, cells, committed version
   + 1}, free_storage)` on the run's processor (A17: in a GPU run the list is a `Buffer<CellId>` on the GPU,
   `GridInputs{grid_frame, {}, &cell_buffer, cell_count, version}`, with `cell_count` a CPU number from the
   stage's own status record; the new counts come from `grid.counts()`; in a batched GPU run the list is built on
   the CPU, uploaded into that buffer and a CPU copy of the same grid is built from it, 5.3); else the try uses the
   committed grid. (c) `grid::ApplyGridVersion(values, grid)` on
   every try array (the try `GridFields`, every set's transfer arrays, the two `SolidsAtFaces`,
   `SolidCoverageValues`, `grid_edge_is_open_to_sky`, the six cell arrays of stage 10, the `was_solved` arrays,
   `keep_active`), then carry `committed` into `fields` and `carried_pressure` by `CellId`. (d)
   `particles::CheckGrouping(particles_try, grid, grouping)`; on failure
   `particles::GroupParticlesByBlock(particles_try, grid)`. (e) `gas::BuildTry(gas_amounts, grid)`, then
   `gas::ComputeGasSeenByLiquid(grid, all, gas->law, gas_amounts_try)` giving `gas_seen`. (f) Build the ranges
   from the try grid; with `run_stages` set, stages 3 to 19 go to it.
3. **`FindSolidSurfaces`**. `solids::BuildTry(solids, step_duration, smallest_cell_width)`;
   `contact::CheckSolid(solid, step_duration)` for each of `solids_try.solids()`;
   `contact::ComputeContactScale(...)`, `ComputeContactTolerance`, `CheckContactTolerance`;
   `solids::ApplyPlateSides(solids_try, plate_side_fields)` (materials 0 to 4, scene with a plate);
   `solids::ComputeSolidsAtFaces(SolidsOnGridInputs{grid, blocks, solids_try}, 0, solids_at_start)`;
   `solids::ComputeSolidCoverage(inputs, &particles_try, coverage)` (snow and jelly block liquid);
   then, in stage 3's `run_whole_grid` part (A17, batched-run buffers: it floods over the whole grid),
   `solids::FindGridEdgesOpenToSky(grid, domain_box, solids_try, grid_edge_is_open_to_sky)`. In a batched run
   `run` is called once per batch; it builds the solids try and runs the contact checks only in its first call of
   a try (the try number is kept in the try state) and reuses them after.
4. **`ParticlesToGrid`**. `materials::ComputeInternalForces(InternalForcesInputs{table, particles_try, blocks,
   step_duration, smallest_cell_width}, material_settings, material_buffers)`;
   `materials::ComputeParticleFriction(table, particles_try, blocks, material_buffers)`; for X in sets:
   `transfer::ParticlesToGrid(ParticlesToGridInputs{grid, blocks, particles_try, mask(X), solids_at_start,
   material_buffers.internal_force_per_area(), try_number}, transfer, X.grid_from_particles)`, then
   `transfer::ComputeCellsFromParticles(CellsFromParticlesInputs{grid, blocks, particles_try, mask(X),
   table.first_ingredient_of_material(), try_number}, X.cells_from_particles)`; then `apply_grid_only_region`.
5. **`SaveGridBeforeForces`**. For X in sets: `transfer::SaveGridBeforeForces(SaveGridInputs{grid, blocks,
   mask(X), X.grid_from_particles, solids_at_start}, X.before_forces)`.
6. **`ApplyForces`**. For X in `S`, `E` (sets with internal force): `materials::ApplyInternalForce(grid,
   blocks, step_duration, material_settings, X.grid_from_particles, X.totals_of_block)` (the sub-span of the
   range). Then for X in sets, gas included, step: `ComputeVelocityAfterGravity` on every face whose kind is
   `kFromParticles` or `kFromGridOnlyRegion`: `velocity += (step_duration * gravity)[axis]`, the product
   formed once per axis, so gas can take it back exactly.
7. **`DriveWaves`** (scene with an open sea). `solids::DriveWaves(DriveWavesInputs{grid, blocks,
   &ready_open_sea, &L.grid_from_particles.mass, step_end_time, step_duration}, L.grid_from_particles.velocity,
   solids_buffers)`; the block partials are kept for the report.
8. **`CollideSolids`**. `solids::CollideSolids(CollideSolidsInputs{grid, blocks, {faces(L), faces(S),
   faces(E)}, solids_at_start}, solids_try, solids_buffers)`, where `faces(X)` is `MaterialSetFaces{
   &X.grid_from_particles, &X.grid_from_particles.velocity, &X.cells_from_particles, smallest ingredient
   friction of X, at_solid_faces of X}`. `G` is never passed.
9. **`FindLiquidSurface`** (in every scene with liquid or gas). `pressure::ComputeDistanceToSurface(
   SurfaceDistanceInputs{grid, blocks, particles_try, mask(L), kAllLiquidMaterials, particle_radius,
   surface_tension_max, &fields.is_grid_only_region}, pressure_buffers)` (a gas-only scene passes an all-zero
   mask); then, because water and syrup form a pair, the same call with `material_index` 0, then 1;
   `pressure::FindLiquidSurface(LiquidSurfaceInputs{grid, blocks, distances, solids::ToSolidCoverage(coverage),
   step_start_time, step_duration, surface_tension_max, lightest_density, has_viscous_ingredient},
   pressure_buffers)`; `pressure::FindMaterialBoundaries(MaterialBoundaryInputs{grid, all, geometry, field 0,
   field 1, materials::FindSurfaceTensionBetween(table, 0, 1)}, pressure_buffers)`;
   `pressure::FindAirPockets(AirPocketInputs{grid, all, geometry, &committed_air_pockets,
   &grid_edge_is_open_to_sky, gas_seen.holds_tracked_gas, air_pockets, lightest_density},
   AirPocketArrays{&fields.pocket_of_cell, try_pocket_storage}, pressure_buffers)`;
   `viscosity::CheckViscosityCanRun` on this try's summary.
10. **`BuildPressureEquations`** (scene with liquid). `pressure::ComputeCarriedPressure(grid, all,
    committed.pressure, committed.has_pressure, committed.velocity_after_pressure, step_duration,
    fallback_pressure, carried_pressure, pressure_buffers)` only when an ingredient's law is not
    `kCannotBeSqueezed` (else `carried_pressure` keeps what stage 2 carried); step: `mixture_density` and
    `sensed_volume_ratio` per cell from `L.cells_from_particles` with `transfer::ComputeMixtureDensity` and
    `ComputeSensedVolumeRatio`; `materials::ComputeCellSqueezeValues(CellSqueezeInputs{table, grid, blocks,
    L.cells_from_particles, &carried_pressure}, bulk_stiffness, target_volume_ratio)`;
    `pressure::ComputeVolumeCorrectionRate(geometry, sensed_volume_ratio, target_volume_ratio,
    &fields.is_grid_only_region, volume_correction, added_volume_rate, pressure_buffers)`;
    `solids::BuildMotionMap(MotionMapInputs{grid, all, geometry, coverage, solids_try}, solids_buffers)`,
    `solids::ToFreeSolidPushes(motion_map)`, `solids::ToFreeSolidInverseInertia(motion_map)`;
    (A18, scene with a blocking set; whether stage 11 or 12 follows, A24) `solids::BuildBlockingMaterialFaces(
    BlockingMaterialFacesInputs{grid, all, geometry, coverage, {BlockingSet{mask(S), &S.grid_from_particles},
    BlockingSet{mask(E), &E.grid_from_particles}}}, solids_buffers)` giving `faces`; (A18, scene with gas) `sealed =
    gas::ApplyGasStepStart(GasStepStartInputs{&grid, all, geometry, &grid_edge_is_open_to_sky, domain_box,
    gas_sources, step_start_time, step_duration}, *gas, gas_amounts_try, gas_buffers, gas_pressure_buffers)` (gas
    design 5.3 steps 1 to 4 and the list of sealed blobs touching liquid; A24: the same list whichever liquid stage
    follows, no coupling argument); `pressure::BuildPressureEquations(EquationInputs{grid, all, geometry,
    &L.grid_from_particles.velocity, &mixture_density, &bulk_stiffness, &carried_pressure, &added_volume_rate,
    &air_pockets, gas_seen.gas_pressure_above_atmosphere, gas_seen.holds_tracked_gas, surface_tension_with_air,
    material_boundaries, free_solid_pushes, free_solid_inverse_inertia, SealedBlobCheck::kWhenBuilt,
    blocking_material_faces = solids::ToBlockingMaterialFaces(faces), sealed_gas_blobs = sealed.blobs,
    sealed_gas_blob_of_cell = sealed.sealed_blob_of_cell, gas_pressure_above_background =
    sealed.pressure_above_background}, pressure_buffers)`. In a viscous scene only (stage 11 follows; A26, A27):
    before `BuildPressureEquations`, `free_solids = solids::ToFreeSolidsForViscousSolve(motion_map)` (one entry per
    free solid touching liquid), from which the step fills the step-owned `free_solid_motion_before_pressure` (6
    doubles per free solid, `V` then `Omega`, in `FreeSolidPush::solid_index` order, from each entry's
    `velocity_before_pressure` and `turn_rate_before_pressure`) given to `EquationInputs::
    free_solid_motion_before_pressure` (pressure plan; `BuildPressureEquations` gives `kBadInput` unless its size is
    0 or 6 x the free solid count; stage 12 passes it empty); after `BuildBlockingMaterialFaces` (scene with a
    blocking set), `viscous_faces = solids::BuildViscousMaterialFaces(BlockingMaterialFacesInputs{grid, all,
    geometry, coverage, {BlockingSet S, BlockingSet E}}, faces, geometry, solids_buffers)` (per blocking set, the
    faces with face mass within `kFitReach` cell widths of a liquid cell centre; every `BlockingMaterialFace` piece
    is among them). An inviscid scene calls neither.
11. **`SolveViscosityAndPressure`** (a liquid ingredient has viscosity above 0, A25; A5, A6). The stage table
    holds 11 or 12, chosen once from the ready scene in `CheckSceneCanRun` and `BuildSimulation` (5.14); retries,
    shortened steps, limit raises, cell resizes, batches and the viscous effect number never change it.
    `viscosity::SolveViscosityAndPressure(ViscosityInputs{grid, all, &geometry, &equations,
    {&L.cells_from_particles}, table.ingredient_viscosity(), solids_try.solids(),
    &L.grid_from_particles.velocity, step_duration, material_faces = solids::ToViscousMaterialFaces(viscous_faces),
    free_solids}, viscosity, cancel_requested, viscosity_buffers, fields.velocity_after_pressure, fields.pressure,
    viscosity::SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure,
    material_face_velocity_after_solve, free_solid_motion})` (the four step-owned spans, A24, A26, A27); then
    `pressure::ComputeWasSolved(equations, L.was_solved, fields.has_pressure)`; then
    `solids::ApplySolvedMotionToSolids(motion_map, free_solid_motion, fields.pressure, solids_try, solids_buffers)`
    (A27, in place of `ApplyPressureToSolids`: each free solid touching liquid takes its solved velocity and turn
    rate, travel and turn limits checked again, `kSolidMovesTooFar`; push report `mass * (after - before)`, the
    pressure part kept separately); then (scene with a blocking set) `solids::ApplyViscousSolveToMaterials(
    viscous_faces, material_face_velocity_after_solve, {BlockingSet S, BlockingSet E}, solids_buffers)` (A26, in
    place of `ApplyPressureToMaterials`: each band face of each blocking set gets its solved velocity in its set's
    `GridFromParticles::velocity`; every other face of the set keeps its velocity after `ApplyForces` and
    `CollideSolids`; `PressureOnMaterials::push` per set into the report). `blocking_material_velocity_after_pressure`
    is still filled (the `BlockingMaterialFace` subset), not written back. The joint solve carries the band
    material faces and the free solids as velocity unknowns (the liquid sticks to them) and the sealed gas blobs of
    stage 10 (A24, A26, A27; viscosity 5 "Coupled terms"); the viscosity buffers hold `max_sealed_gas_blobs` =
    gas's `max_gas_blobs` (0 without gas), `max_material_faces` = 2 x solids' `max_motion_map_entries` (0 without a
    blocking set), `max_free_solids` = the free solid count. Replaced (A26, A27): "then
    `solids::ApplyPressureToSolids(motion_map, fields.pressure, solids_try, solids_buffers)`; then (scene with a
    blocking set) `solids::ApplyPressureToMaterials` exactly as in 12". Replaced (A24): "Sealed gas is read one step
    late here (`kReadOneStepLate`): viscosity's joint solve does not carry the blob term (A18 open point)."
12. **`SolvePressure`** (liquid, and every liquid ingredient has viscosity 0, A25; the same one-time choice).
    `pressure::SolvePressure(equations,
    pressure_solve, cancel_requested, SolvedArrays{&fields.pressure, &fields.velocity_after_pressure,
    blocking_material_velocity_after_pressure, sealed_gas_pressure}, pressure_buffers)` (the last two are the
    step-owned spans, A18); then `pressure::ComputeWasSolved(equations, L.was_solved, fields.has_pressure)` and
    `solids::ApplyPressureToSolids(motion_map, fields.pressure, solids_try, solids_buffers)` (unchanged by A26, A27:
    an inviscid liquid slides, material faces are eliminated, free solids are the rank-one term); then
    (A18, scene with a blocking set) `solids::ApplyPressureToMaterials(faces,
    blocking_material_velocity_after_pressure, {BlockingSet S, BlockingSet E}, solids_buffers)`, which writes each
    solved face velocity into its set's `GridFromParticles::velocity`; its `PressureOnMaterials::push` per set goes
    into the report beside the solids' pushes.
13. **`SolveGas`** (scene with gas; after the liquid solve, 11 or 12). In a gas scene without stage 10 (no
    liquid), first `gas::ApplyGasStepStart` as in 10 (empty blob list). `gas::SolveGas(SolveGasInputs{&grid,
    all, geometry, &G.grid_from_particles, &fields.velocity_after_pressure, &L.was_solved, lightest_density,
    &grid_edge_is_open_to_sky, domain_box, gas_sources, gas_wind, step_start_time, step_duration,
    sealed_gas_blobs = sealed.blobs, sealed_gas_pressure}, *gas,
    cancel_requested, SolvedGasArrays{&fields.gas_velocity_after_pressure, &G.was_solved,
    &G.cell_was_solved}, gas_amounts_try, gas_buffers, gas_pressure_buffers)`; a coupled blob's end background
    pressure is `atmosphere_pressure + sealed_gas_pressure[j]`, written by stage 11 or 12 (A24). The two liquid
    pointers are null and `sealed_gas_pressure` is empty in a scene without liquid. `SolveGas` without `ApplyGasStepStart` in the same
    try is `kBadInput`.
14. **`SaveGridAfterPressure`**. `solids::ComputeSolidsAtFaces(SolidsOnGridInputs{grid, blocks, solids_try},
    step_duration, solids_at_end)`; then for X in sets, `transfer::SaveGridAfterPressure(SaveGridInputs{grid,
    blocks, mask(X), X.grid_from_particles, solids_at_end}, velocity, was_solved, X.after_pressure)` with:
    `L`: `fields.velocity_after_pressure`, `&L.was_solved`; `S`, `E`: `X.grid_from_particles.velocity` (after
    `CollideSolids` and, on the pieces the liquid presses, after `ApplyPressureToMaterials`, or in a viscous scene on
    the band faces after `ApplyViscousSolveToMaterials`, A26), null; `G`: `fields.gas_velocity_after_pressure`, `&G.was_solved`.
15. **`GridToParticles`**. For X in sets: `transfer::GridToParticles(GridToParticlesInputs{grid, blocks,
    particles_try, mask(X), X.before_forces, X.after_pressure, cell_was_solved}, grid_at_particles)` with
    `cell_was_solved` = `&fields.has_pressure` for `L`, `&G.cell_was_solved` for `G`, null for `S` and `E`.
16. **`MoveParticles`** (5.5). `materials::ComputeOwnMotionShares(OwnMotionInputs{table, grid, blocks,
    particles_try, grid_at_particles, distances.at_corners, coverage.distance_to_solid}, material_buffers)`;
    then per material 0 to 4: mix with `transfer::MixParticleVelocity`, `contact::MoveParticles`, the travel
    check, `materials::ApplyMaterialResponse`, `particles::RemoveParticles(..., RemoveReason::kLeftDomain)`;
    then `move_grid_only_liquid` on the whole grid.
17. **`MoveSolids`** (whole state). `solids::ApplyContactReactions(reaction_fields of materials 0 to 3,
    solids_try, solids_buffers)` (never gas); with a plate, the step fills `face_force_work_rate` and calls
    `solids::ComputeCutWork(cut_work_fields of 3, step_duration, solids_buffers)`. (A18: no longer
    `ApplyPressureToMaterials`, and no push owed to blocking particles: their faces moved in stage 11 or 12.) In a
    batched run (A17, batched-run buffers) both calls run on the CPU with the stage-17 `SolidsBuffers`
    (`FindContactReactionBuffers`), whose `CollideSolids` block partials batches wrote after pass 1; `CommitTry`
    reads the per-solid results of stages 12 and 17 from two instances.
18. **`MakeFoamAndSpray`**. `make_foam_and_spray` with `FoamAndSprayInputs{..., after_pressure =
    {&L.after_pressure}, &distances, solids_try.solids(), step_count + 1, blocks, {}, &fields}` (in a batched run,
    `block_in_whole_grid` is the piece map copy given to `ApplyGridPiece`).
19. **`CheckStep`** (5.6). `particles::CheckParticles(particles_try, blocks)`; finite checks of `fields`;
    `solids::CheckSolidExchange(ExchangeInputs{motion_map, fields.pressure, geometry, pressure_on_solids,
    contact_on_solids, moved, smallest_end_distance, contact_tolerance}, solids_try, tolerances)`; the gas
    `GasAmountChanges`; the next `CommittedStepLimits`.
20. **`CommitStep`** (5.6). `solids::ApplyPlateRemoval(plate_id, plate_side_fields)` when the plate's path ends
    with `kLeavesScene` inside this step; `particles::CommitTry(particles, particles_try, time +
    step_duration)`; `gas::CommitTry(gas_amounts, gas_amounts_try, time + step_duration)`;
    `solids::CommitTry(solids, solids_try)`; the swaps (in a batched GPU run the CPU grid copy pair with the grid
    pair); then `read_largest_grid_only_speed` into the committed
    `CommittedStepLimits::grid_only_speed` (the grid-only move of stage 16 is committed by now).
21. **`ResizeCells`**. `resize_cells`, after `TryStep`'s own work is complete (5.11).

| # | Stage | Writes (try or scratch) | Border | Skipped when |
|---|---|---|---|---|
| 1 | `AddNewParticles` | counters, waiting particles, removal marks | none | never |
| 2 | `ChooseActiveCells` | try grid, grouping, bound arrays, `fields` carried, `gas_amounts_try`, `gas_seen` | none | never |
| 3 | `FindSolidSurfaces` | `solids_try`, `solids_at_start`, coverage, open-to-sky marks, `plate_side` | touching | no solid but the domain box and no blocking material |
| 4 | `ParticlesToGrid` | `internal_force_per_area`, `friction`, each set's `GridFromParticles` and `CellsFromParticles` | touching | never |
| 5 | `SaveGridBeforeForces` | each set's `before_forces` | four cell widths | never |
| 6 | `ApplyForces` | each set's `velocity`; `totals_of_block` per set with internal force | none | never |
| 7 | `DriveWaves` | `L` velocity; block partials | none | no open sea |
| 8 | `CollideSolids` | velocities of `L`, `S`, `E` | touching | no solid and one non-gas set |
| 9 | `FindLiquidSurface` | `SurfaceDistances`, `GeometrySnapshot`, boundaries, try air pockets, `fields.pocket_of_cell` | touching | no liquid and no gas |
| 10 | `BuildPressureEquations` | six cell arrays, motion map, blocking-material faces, gas step-start work in `gas_amounts_try` and the sealed blob list, `PressureEquations`; viscous scene (A26, A27): the band faces `viscous_faces`, `free_solid_motion_before_pressure` | touching | no liquid |
| 11 | `SolveViscosityAndPressure` | `fields.velocity_after_pressure`, `fields.pressure`, `has_pressure`, `L.was_solved`; the four step-owned spans (A24, A26, A27); each free solid touching liquid's solved motion (A27); `S`, `E` velocity on their band faces (A26). Replaced: "the same as 12 (A24): ... free solid motion; the two step-owned spans; `S`, `E` velocity on the faces the liquid presses" | none | no viscous ingredient: every liquid ingredient has viscosity 0 (A25; chosen once from the ready scene, fixed for the run) |
| 12 | `SolvePressure` | `fields.velocity_after_pressure`, `fields.pressure`, `has_pressure`, `L.was_solved`, free solid motion; the two step-owned spans; `S`, `E` velocity on the faces the liquid presses | none | no liquid, or a liquid ingredient has viscosity above 0 (11 runs instead; A25, fixed for the run) |
| 13 | `SolveGas` | `fields.gas_velocity_after_pressure`, `G` was_solved arrays, `gas_amounts_try` | none | no gas |
| 14 | `SaveGridAfterPressure` | `solids_at_end`; each set's `after_pressure` | four cell widths | never |
| 15 | `GridToParticles` | `GridAtParticle` per material | touching | never |
| 16 | `MoveParticles` | particle arrays, `ParticleContact`, `SolidReaction`, removal marks; grid-only amounts | none | never |
| 17 | `MoveSolids` | `solids_try` (A18: no blocking particle's velocity) | none | no driven or free solid and no blocking material |
| 18 | `MakeFoamAndSpray` | nothing of the step | none | callback null |
| 19 | `CheckStep` | `StepReport`, next `CommittedStepLimits` | none | never |
| 20 | `CommitStep` | committed state | | |
| 21 | `ResizeCells` | committed, in its own try | | callback null |

Order rules kept from `main` and its paper (`fluidity-method.typ:79-81`): the snapshot of stage 5 comes before
every force, wave and contact change, so `own_motion` never contains a force twice; wave relaxation (7) comes
after it (`liquid_substep_3d.cc:1247-1274`); the liquid surface (9) is found at the start positions; the liquid
solve is the last write to the liquid face velocity before stage 14; the solid surfaces use the start pose for
geometry and equations and the end pose for the second snapshot. New rules (A18, A24): the gas start-of-step
work (`ApplyGasStepStart`) runs in stage 10 before the liquid's equations are built; a sealed gas blob that
touches liquid is one more unknown of the liquid solve, in stage 12 or, beside a viscous liquid, in stage 11
(A24), and stage 13 takes its solved `sealed_gas_pressure` after either; open gas (also above a viscous free
surface), slivers, cells that touch gas and held blobs are read from the last commit through `gas_seen` (one step
late); `SolveGas` (13) then reads the liquid's velocity after this step's solve and `L.was_solved` (gas design
5.3). A free solid's end pose is known after `ApplyPressureToSolids` (12) or `ApplySolvedMotionToSolids` (11,
A27); gas moves with free solids through `solid_volume_rate` and does not push them. Snow and jelly are not solved
for, but the liquid solve holds their face inertia: on each face piece where an inviscid liquid meets a blocking
set with face mass, stage 12 gives that set's velocity after pressure in the same step (`ApplyPressureToMaterials`
right after `ApplyPressureToSolids`); elsewhere their after-pressure velocity is the one `CollideSolids` left.
Beside a viscous liquid (A26, A27) the liquid sticks: stage 11 solves the band faces of each blocking set and the
motion of each free solid touching liquid together with the liquid, and writes them back right after
`ComputeWasSolved` (`ApplySolvedMotionToSolids`, then `ApplyViscousSolveToMaterials`); every face of a set outside
the band keeps the velocity `CollideSolids` left (the drag reaches it next step through the material's internal
force). Stage 11 or 12 is chosen once per run (A25). Replaced (A26, A27): "stage 11 or 12 gives that set's velocity
after pressure in the same step (`ApplyPressureToMaterials` right after `ApplyPressureToSolids`)" for stage 11. Replaced: "the liquid solve reads the gas pressure one step late" for sealed blobs, and "the pressure on
blocking materials comes back one step late (`ApplyPressureToMaterials` in 17)" (11.11); "every gas beside a
viscous liquid (stage 11, `kReadOneStepLate`) is read one step late" (11.26).

### 5.3 `ChooseActiveCells`

1. Mark. A cell block is needed when one of its cells lies within `kCoveredReachInCellWidths` of a particle
   (waiting particles and gas markers included), measured in the width of the particle's `size_level`; or holds
   a cell marked by `gas::FindCellsToKeepActive` (committed amounts); or lies in an open-sea zone below the
   highest wave; or is a grid-only cell of the committed grid. 2.5 widths cover the nearby faces (1.5), the
   liquid surface samples one cell beyond a surface cell, and the first fill radius (2); farther fills walk into
   blocks that exist or stop.
2. Compare the needed block ids with the committed grid's. Equal: the try grid is the committed grid, no new
   version. Different: new cells take the level of `choose_cell_size`, else of the nearest committed active
   cell, else `new_cell_size_level`; sibling cells come together (grid Q39); `grid::BuildGrid(inputs,
   free_storage)` with `version + 1`, on the run's processor, allocating nothing. `kSizeStepRuleBroken` is `kStageFailed`;
   `kTooManyCellBlocks` is `kTooManyCellBlocks`. A rejected try gives the storage back with
   `grid::ReleaseBuffers`.
3. Bind every try array to the try grid with `grid::ApplyGridVersion` and carry `GridFields` by `CellId` and
   `FacePieceId` from the committed copy (a new cell: `has_pressure = 0`, pressure 0, `pocket_of_cell =
   kNoAirPocket`, particle region). Cells never change size here, so no remap is needed.
4. `CheckGrouping`; on failure `GroupParticlesByBlock`. `kOutsideGrid` cannot occur for a covered particle;
   if it does it is `kParticleOutsideGrid`.
5. Gas: `gas::BuildTry` copies the committed amounts by `CellId` (cells the try grid lost are counted in its
   report), then `gas::ComputeGasSeenByLiquid` fills `gas_pressure_above_atmosphere` and `holds_tracked_gas`.

The needed set depends only on the try's positions and the committed gas amounts, never on earlier steps, so
results do not depend on history. The grid is then fixed until the commit. `GridFrame::origin` is
`domain_box.low_corner` for the run. On the GPU the marking is one pass that writes a block id per particle,
and the needed cell list stays on the GPU in a `Buffer<CellId>`; only its entry count comes back, in the stage's
status record, and `grid::BuildGrid` builds the tables on the GPU (A17; the CPU builder is the oracle, equal byte
for byte). Replaced (A17): "unique ids come back to the CPU (8 bytes per block, grid Q41), which builds the
grid". With `choose_cell_size` set in a one-piece GPU run, the needed cells not in the committed grid are compacted
and read back (one copy, 8 B each), the callback runs per cell on the CPU, the levels go back (one copy, 1 B each)
and a GPU pass applies them before the build (6.5 row O4).

**Batched runs (A17, batched-run buffers).** The particles are in CPU memory, so stages 1 and 2 run on the CPU
(launching GPU work on GPU buffers). Stage 2 marks and builds the needed cell list on the CPU, uploads it into
`"step: active cell list"` (one counted copy, 8 B per active cell, B4), builds the try grid on the GPU from it and a
CPU copy of the same grid with the CPU builder from the same list into the free half of `"step: grid copy on the
CPU"`; a version or block count that differs is `kStageFailed` (`module = kGrid`). No table is copied: A17 makes
the tables equal. Grouping, `choose_cell_size` (which receives `FindCpuGridCopy(simulation)->tables(RunOn::kCpu)` as
`committed`, 11.30) and the marker upkeep read the copy. `CommitStep` swaps the copy pair with the grid pair; a
reject releases the free copy with `grid::ReleaseBuffers`; a committed resize swaps in the CPU resized grid cell
resize built into `ResizeTry::free_cpu_grid_copy_buffers()` and handed over with `ApplyResizedCpuGridCopy`, with no
copy (reconcile, 11.29; replaces the `grid::ReadGrid` refresh, row B5 withdrawn). The two CPU copy buffers are built
with `grid::BuildGridBuffers(settings, cpu_grid_scratch)` from one CPU `grid::GridScratch` (11.31).
`gas::FindCellsToKeepActive` runs on the amounts' processor into `keep_active`, then one counted copy into `"step:
keep active on the CPU"` (B2) feeds the CPU marking; with cell resize, `is_grid_only_region` of the committed fields
is mirrored into `"step: committed grid-only marks on the CPU"` once per committed `state_version` (B3). A batched
CPU run makes the same mirror copies (CPU to CPU, counted the same) but has no grid copy and no list upload.

### 5.4 `AddNewParticles` and the counters

For each source in ascending `source_id`, active at the step's start time:
`owed = owed_particles + step_duration * particles_per_second`; `count = floor(owed)`; children
`next_child_number .. + count - 1` in order; then `owed_particles = owed - count`. Child `k` is placed at
`position + spread * (2 u - 1)` per axis, with `u` from `ComputeRandomWords(counter = {k, source_id},
key = random_seed)`. FIXED: `main` also keys on the step number, so a particle's place depended on the step
that added it; here the same particles appear whatever the step durations. All children of all sources of one
material go to one `particles::AddParticles` call (ids are consecutive in that order); they have
`kHasOwnMotion` off. The counters live in a try copy (`std::vector<SourceCounter>`, CPU) swapped at commit.

Open-sea zone (ported rule, `staging.cc:547-630`): lattice sites of `open_sea_particle_spacing` in the zone;
a particle of `open_sea_material_index` in the zone above `ComputeOpenSeaWave(...).surface_height` is removed
(`RemoveReason::kThroughOutlet`); an empty site below the surface and outside solids gets a particle with the
wave's velocity (`AddReason::kFromSource`). FIXED: sites are marked by a pass over the particles grouped in
the zone's blocks on the processor where the particles live (the run's processor in one piece; the CPU in a
batched run, no copy); in a one-piece GPU run the CPU reads one bit per site and the removal list back in two
counted copies (6.5 row O2). No 4096-site limit.

Gas markers: `gas::FindMarkerChanges` decides from the committed amounts and the committed gas-set velocity;
the step applies its two lists and reports them in `markers_added` and `markers_removed`. The lists are in CPU
memory in every GPU layout: one piece reads them back (`MarkerChanges::copy_count`, `bytes_copied`, row O3); a
batched run calls the CPU form on mirrors of the committed `gas_volume` and `gas_velocity_after_pressure`, copied
once per committed `state_version` (two counted copies, B1; a retry copies nothing).

`kTooManyParticles` from `AddParticles` is a limit, not a numerical reject (5.7).

### 5.5 `MoveParticles`

For the range: 1 `materials::ComputeOwnMotionShares` (the old liquid travel rule,
`liquid_substep_3d_cpu.cc:308-310`, lives there). Then per material, with `particles::BuildFieldsToChange(
particles_try, m, blocks, {kPosition, kVelocityForNextStep, kVelocityForThisStep, kFittedVelocityChange,
kFlags, kVolumeRatio} plus the state arrays the material carries)`: 2 `transfer::MixParticleVelocity` per
particle, written to `velocity.for_next_step` and `velocity.for_this_step` (A1); `fitted_velocity_change` from
`GridAtParticle` where `carries_fitted_change`, else zero with `kCarriesFittedChange` off; `kHasOwnMotion` on;
3 `contact::MoveParticles(MoveParticlesInputs{count, id, position, velocity, material_buffers.friction(m),
plate_side, solids_try.solids(), step_duration, start_overlap_tolerance, contact_tolerance},
MovedParticles{position.write, velocity.for_next_step.write, records[m], reactions[m]}, contact_buffers)`
(travel along the straight path with `velocity.for_this_step`, contact, friction; gas markers with friction 0);
4 the travel check: `ComputeTravelInCellWidths` = largest axis part of `end - start` over the width of the
particle's cell; above `kMaxTravelInCellWidths` the try is rejected with `kParticleTravelledTooFar`, naming the
lowest `particle_id` (ported: `cpu.cc:384-386`); 5 `materials::ApplyMaterialResponse(MaterialResponseInputs{
table, m, fields, grid_at_particles[m], records[m], step_duration}, to_change, material_buffers)` (it writes
`volume_ratio` for every kind, which `main` did at `cpu.cc:387-390`; the start position, the velocity for this
step and the solids are not inputs: materials needs only the exact velocity change and the contact record);
6 a particle whose end position is outside `domain_box` through a side with `ClosedSides` false is listed and
removed with `RemoveReason::kLeftDomain`; its mass, volume and momentum go to
`StepReport::removed_through_open_sides`. Steps 2, 4 and 6 are one fused pass on the GPU.
`contact::ContactError::kTooCloseToCall` becomes `kContactTooCloseToCall`; `kStartsInsideSolid` and
`kOnWrongSideOfPlate` become `kStageFailed`. After every material and range, `move_grid_only_liquid` moves the
grid-only amounts on the whole grid; its failures arrive as `kParticleTravelledTooFar`.

Not ported: the whole-cell six-face travel field (`base/liquid_regular_cell_transport_3d.h`), valid only in a
fixed axis-aligned domain; every particle uses the velocity for this step of step 2 (transfer left this to the
step). The material response is inside this stage, as the materials design places it; it is not a stage of
its own.

### 5.6 Try and commit

**What a try owns.** `particles::ParticlesTry`; the try grid (in the free `GridBuffers`); the try `GridFields`;
the try air-pocket list and the returned `AirPockets` (with `BlobHistory`); source counters;
`solids::SolidsTry`; `gas::GasAmountsTry`; the next `CommittedStepLimits`; `StepReport`. Scratch (each set's
transfer arrays, the six cell arrays, `was_solved` arrays, equations, samples, the motion map, the
blocking-material faces, the two step-owned spans of stage 11 or 12, `GridAtParticle`, contact records, both
`PressureBuffers`) is reused by every try and holds nothing a later step reads.

**Commit** (stage 20) only after stage 19 passed and, on the GPU, one wait showed that all work of the try
finished: `ApplyPlateRemoval` when due, then `particles::CommitTry(particles, try, time + step_duration)`,
`gas::CommitTry`, `solids::CommitTry`, then swap the grid storages, `GridFields`, air pockets, counters and
`CommittedStepLimits` with their committed twins; `time += step_duration` (or exactly `target_time` when
`AdvanceTo` marked the last step and the sum is one rounding step away, ported from `liquid_state_3d.cc:757-762`);
`state_version += 1`; `step_count += 1`. No array is copied. FIXED: `main` copies the whole committed state into
the try at the start of every try (`state.cc:723`); here a try starts from views. The three `CommitTry` calls
only check what stage 19 already checked; if one fails after another succeeded, the try reports `kStageFailed`
in `kCommitStep` and the simulation is marked unusable (`kSimulationUnusable` afterwards), as for
`kRunOnCallFailed`.

**A reject** drops the tries (`RejectTry` of particles, gas and solids; the try grid's storage back with
`ReleaseBuffers`). Untouched, to the byte: time, `state_version`, `step_count`, every particle array and
`next_particle_id`, grouping, the committed grid and its version, source counters, solid poses and velocities,
air pockets and `next_blob_id`, every committed `GridFields` array, gas amounts, `CommittedStepLimits`, limits.
What may change: `StepReport`, timings, buffer contents.

**`CheckStep`** (stage 19): `particles::CheckParticles` on every range; every `GridFields` value finite;
`time + step_duration > time`; ids and counters did not go back (ported, `state.cc:789-816`); the solids'
end poses passed `contact::CheckSolid`; `solids::CheckSolidExchange`; the next `CommittedStepLimits`: the
smallest `ComputeTravelStepLimit(|velocity.for_next_step|, |gravity|, kMaxTravelInCellWidths * cell_width)` over
particles with their own cell width, the largest `speed / cell_width` of particles, solids and gas markers
(reduced with `ComputeLargestMagnitude`, exact in any order), and the try's surface-tension, air-pocket,
material and gas limits.

**Which failure is reported.** A stage runs all its ranges, sets and materials even after one failed, then
reports the failure with the highest `RejectKind` (`kCannotContinue`, then `kRaiseLimits`, then `kShortenStep`)
and, among equals, the lowest stable id (`particle_id`, else `CellId`, else `solid_id`). The answer therefore
does not depend on how the grid is cut into ranges. The first failing stage ends the try.

**Module errors to reasons.** `kRunOnCallFailed` of any module: `kRunOnCallFailed`. `kCancelled` of pressure,
viscosity or gas: `kCancelRequested`. `kTooManyParticles`, `kTooManyCellBlocks`, any `kBuffersTooSmall`: the
limit reasons. `kNotFinite` of any module: `kValueNotFinite`. Pressure `kStepTooLongFor...`: the same-named
reasons, `limit` = the allowed duration. Gas `kStepTooLongForGasFlow`: the same-named reason, `limit` = the
largest duration with no cell emptied. Materials `kStepTooLong`: `kStepTooLongForMaterial`, `limit` = its
`magnitude`; `kDeformationCollapsed`: the same-named reason; `kMaterialUnstable`, `kOutsideAllowedRegion`,
`kPressureLawOutOfRange`: `kStageFailed`. Pressure `kNotConverged`, viscosity `kPressureSolveDidNotConverge`,
gas `kPressureSolveFailed` holding `kNotConverged`: `kPressureNotConverged`; viscosity
`kVelocitySolveDidNotConverge`: `kViscosityNotConverged`. Solids `kPathJumpInsideStep`, `kSolidMovesTooFar`
and contact `kSolidTurnsTooFar`: `kSolidTravelledTooFar` with its `limit` (also `kSolidMovesTooFar` from
`ApplyPressureToSolids` and, A27, from `ApplySolvedMotionToSolids`). Viscosity's exclusions that remain, pressure `kCellSplitBySolid`, solids
`kSolidsTooClose`: `kNotSupportedNow` with the
matching `CapabilityReason` (`kFromSolids` for the last). Everything else (`kSealedBlobVolumeMismatch`,
`kEquationsNotPositive`, `kAirPocketFateUnknown`, `kFaceCannotBeFilled`, gas `kTemperatureOutOfRange`,
`kBackgroundPressureOutOfRange`, solids `kParticleInsidePlate`, `kExchangeNotBalanced`, ...): `kStageFailed`
with `Module` and the module's error number. A callback's `StageOutcome` is taken as it is; `false` from
`make_foam_and_spray` is `kCallbackFailed`. As the old paper says, a volume balance that cannot be met is an
error, not a retry.

**After a reject.** `shorter_step_duration = step_duration / 2`, or `context.limit` when the reason carries a
smaller allowed duration. `TryStep` keeps no count; `AdvanceTo` allows `max_retries_per_step` shorter tries
for one step, then stops with `kRetriesUsedUp` (ported: two).

### 5.7 Limits and the capacity restart

A limit reason means buffers are too small; a shorter step would not help and time must not be spent.
`TryStep` returns `limits_needed`: each exceeded limit set to `max(needed, limit * limit_growth)`.
`AdvanceTo` calls `ApplyLimits`, then repeats the same request (same duration, same `state_version`).
`ApplyLimits` is the explicit growth of the report (CUDA 3.2): 1 build the estimate for the new limits and
check `estimate.on_processor + committed bytes of the old buffers <= allowed_bytes`, after dropping try copies
and scratch, which hold nothing; 2 allocate the new buffers (a new `grid::GridScratch` and
`grid::BuildGridBuffers(settings, scratch)` twice at the new `max_cell_blocks`, the old scratch freed with the
old storages, reconcile 11.31; `grid::Build...Values(run_on, max_cell_blocks)`, the modules' buffers); 3 rebuild the
committed grid into one new storage with the same `GridInputs` (same tables, same version), bind the new values
with `ApplyGridVersion`, and copy the committed values on the processor (`particles::ApplyParticleLimits` for
particles, slice `CopyBuffer` for the rest); 4 swap; 5 free the old ones. A failure at 1 or 2 leaves the
simulation unchanged (`kDoesNotFit`, `kOutOfMemory`) and `AdvanceTo` stops with `kLimitsCannotBeRaised`; the
caller may save the state and build a batched run. FIXED: `main` restarts the whole frame from a third copy of
the state, once per frame; here only the step is repeated and no frame-start copy exists. Invariant: no result
depends on a limit (`math::TreeSum` pads by count, not by buffer size), so a run with raises gives the bytes of
a run that started with large limits.

### 5.8 The step duration

`ComputeStepDuration` takes the smallest of the limits below, then `n = ceil((target_time - time) / smallest)`
and `step_duration = (target_time - time) / n`, so no tiny last step is left (ported, `cc:2857-2858`). The
limit that gave `smallest` is `active_limit`; when the rounding to `n` equal steps lowered it,
`kEvenStepsToTarget`.

| `StepLimit` | Value, s | From |
|---|---|---|
| `kMaxStepDuration` | `StepSettings::max_step_duration` | ported |
| `kParticleTravel` | the smaller of `CommittedStepLimits::particle_travel` and `kMaxTravelInCellWidths / CommittedStepLimits::grid_only_speed`; `t = 2 d / (v + sqrt(v^2 + 4 a d))` | ported (`cc:2836-2846`); FIXED: per particle with its own cell width; grid-only face speeds (cell resize) |
| `kSolidMotion` | `solids::ComputeStepDurationLimit(solids, time, smallest_cell_width)`: no path key crossed, no surface beyond `max_travel_in_cells` | `cc:2738-2742` moved before the try |
| `kSolidTurn` | `kMaxTurnPerStep` over the largest committed angular speed of a solid | contact 3.4 |
| `kSurfaceTension` | `CommittedStepLimits::surface_tension` (pressure's `surface_tension_step_limit`) | one formula, pressure's |
| `kAirPocket` | `CommittedStepLimits::air_pocket` (`AirPockets::step_limit`) | pressure |
| `kMaterialWaveSpeed` | `CommittedStepLimits::material_wave_speed`; stage 4 checks the try itself (`kStepTooLong`) | materials 5.5 |
| `kGas` | `CommittedStepLimits::gas` (`GasResult::step_limit`, the flow-share limit only); stage 13 checks the try itself (`kStepTooLongForGasFlow`). No sealed-blob limit: every sealed blob touching liquid is solved with the liquid in stage 11 or 12 (A18, A24; `ComputeSealedGasStepLimit` is deleted) | gas |
| `kSourceWindow` | time to the next `start_time` or `end_time` of a source | NEW, so a source never starts mid-step |
| `kAfterReject` | `shorter_step_duration` of the last reject of this step | ported |

Why half a cell: stage 2 covers 2.5 widths around the start position and stages 4 to 15 use the start
positions; half a width keeps the end position inside the cells that touched the start cell, so the next
grouping moves a particle by at most one cell and the liquid surface found at the start still bounds it.
The limits are estimates from committed values; stages 3, 4, 9, 13, 16 and 19 check them on the try (the
same-named reasons), which is what makes a too-long guess safe.

### 5.9 Sizing

`BuildRunPlan`: 1 `CheckSceneCanRun`; 2 counts: `grid::ComputeGridCounts` on the cells the start particles
need (5.3 on the CPU, no buffer), particles per material at the start plus `rate * (end_time - start_time)`
per source, material sets and sets without a solve; limits = counts times `start_limit_headroom`, cell blocks
capped by the blocks the domain box can hold (reconcile item 11, 11.31: `limits.max_cell_blocks` = the blocks of
the 2.5-width cover of the start particles x `start_limit_headroom` 1.25, rounded up, capped by the domain's blocks
at the smallest cell, at least 1; the only source of the room: no `grid::GridBuffers` or `GridScratch` is built
from `GridSettings`' default, which is 0; a room above `grid::kMaxCellBlocksLimit` gives `fit = kDoesNotFit`, and
outgrowing the room during the run is `kTooManyCellBlocks`, raised by `ApplyLimits`); also (A25) one
`LiquidViscousEffect` per liquid ingredient with `viscosity::ComputeViscousEffectPerStep` at
`max_step_duration` and the smallest cell width, and `viscous_path` by the A25 rule; 3 the memory estimate entries
of every module at those limits: `grid::BuildGridMemoryEstimate(run_on, 2, grid::ScratchSharing::kOneForAllBuffers)`
(committed and free buffers, one build scratch; in a batched run plus `BuildGridMemoryEstimate(run_on, 1,
kOneForAllBuffers)` without its scratch entries for the loaded try's buffers at `max_loaded_blocks`),
`particles::BuildParticlesMemoryEstimate`, `BuildContactMemoryEstimate`,
`BuildTransferMemoryEstimate(ingredient_count, material_count, material_set_count, sets_without_solve)`,
`BuildPressureMemoryEstimate` twice (liquid sizes; gas sizes with no air pocket, free solid or boundary),
`BuildViscosityMemoryEstimate`, `BuildMaterialsMemoryEstimate`, `BuildSolidsMemoryEstimate(solids)`,
`BuildGasMemoryEstimate(gas sizes)`, `BuildStepMemoryEstimate`, `output_estimate` (`kRenderOutput`) and `cell_resize_estimate`
(`kCellResize`), summed by `ComputeMemoryBytes` per `KeptFor`; `one_call` is the largest per-module `kOneCall`
sum, with viscosity and pressure added together because the joint solve holds both; 4 `allowed_bytes =
min(memory_share_of_total * total, memory_share_of_free * free)` (the free share only when the total is not
known), and the same rule applied to `on_cpu` gives `allowed_cpu_bytes` (A15); 5 `estimate.on_processor <=
allowed_bytes`: `kOnePiece`. Otherwise `kBatched` if `whole_run + one_step_try` fits the CPU's allowed bytes;
otherwise `kDoesNotFit`. For `kBatched` the caller calls `batches::BuildBatchPlan` and copies
`max_blocks_per_batch`, `max_loaded_blocks` and `max_loaded_particles` into the plan; when the batch plan says
`SolvePlace::kInPieces`, the caller sets `mode = kBatchedWithSolvesInPieces` and copies
`max_solve_loaded_blocks` (A15). In that mode the estimate takes the solvers' entries from
`BuildPressureMemoryEstimate`, `BuildViscosityMemoryEstimate` and `BuildGasMemoryEstimate` built with `max_loaded_blocks =
max_solve_loaded_blocks`: the entries counted per place with `where = kCpu` at the limits, the same entries with
`where = kRunOn` at the loaded piece, and `BuildSolveBatchesMemoryEstimate`; each entry keeps its own `where`. The
liquid's `BufferSizes` set `max_blocking_material_faces` to solids' `max_motion_map_entries` (0 without a
blocking set) and `max_sealed_gas_blobs` to gas's `max_gas_blobs` (0 without gas) (A18); the
`ViscosityBufferSettings` set `max_sealed_gas_blobs` to gas's `max_gas_blobs` (0 without gas) (A24; entry
`"viscosity: sealed gas blob vectors"`), `max_material_faces` to 2 x solids' `max_motion_map_entries` (0 without a
blocking set) and `max_free_solids` to the free solid count (A26, A27; entries `"viscosity: material face
vectors"`, `"viscosity: free solid vectors and blocks"`, `"viscosity: sample solid coefficients"`).

Amended (A17 planned cost): `on_processor` sums only `kRunOn` entries and `on_cpu` the `kCpu` entries plus
`snapshots`, per module too (`ModuleBytes::on_cpu`). Step 5's "otherwise `kDoesNotFit`" no longer fails: the plan
is returned with `fit = kDoesNotFit`, `mode = kBatched` and every estimate filled, so I/O can print what it needs;
`BuildSimulation` fails `kDoesNotFit` on it before allocating. `planned_step_copies` is the sum of the 6.5 rows of
the plan's mode at the limits (upper bounds per try). With a batched plan, `BuildStepMemoryEstimate(scene, counts,
&plan)` returns the entries of 6.5 with their `where`, so `BatchPlan::bytes_on_processor`, `bytes_on_cpu` and
`BuildSimulation`'s allocation stay equal.

`BuildSimulation` builds one `grid::GridScratch` first (`grid::BuildGridScratch(run_on, limits.max_cell_blocks)`),
lends it to the committed and free `GridBuffers` (`grid::BuildGridBuffers(settings, grid_scratch)` twice) and, in a
batched run, to the loaded try's buffers, never moves it while they exist and releases it last (11.31); a batched
GPU run's CPU grid copy pair shares one CPU `GridScratch`. It fixes the viscous path from the ready scene (A25).
`BuildSimulation` allocates by walking the same entries, so `allocated_bytes == estimate.on_processor` and
`allocated_cpu_bytes == estimate.on_cpu - estimate.snapshots` are equalities a test checks (in every batched
layout, the `kCpu` and `kRunOn` sums each). With `kBatched` it builds the batched layout: the particle arrays
(with `CpuMemory::kLockedForCopies`) in CPU memory, every whole try array where the placement rule of 6.5 puts it
(replaces "the per-try arrays in CPU memory"), and one loaded `StepTry` over a `GridBuffers` of
`max_loaded_blocks` blocks on the GPU, whose stage functions are the ones of `kStages`, unchanged. The loaded
`StepTry` owns, on the run's processor, a `transfer::TransferScratch`, a `materials::MaterialBuffers`, a
`contact::ContactBuffers` and a `solids::SolidsBuffers`, each built at `max_loaded_particles` per material and
`max_loaded_blocks`; the simulation's whole instances are built with every particle count 0 (their grid entries stay
at `max_cell_blocks`), except the stage-17 `SolidsBuffers` (`run_on = kCpu`, whole particle limit). No module
signature changes: the builders take the smaller counts. `ReadStepArrays` lists the arrays of that layout with
their `whole_place`.
In both batched modes the loaded `StepTry` also owns a one-piece `PressureBuffers` for `max_loaded_blocks` blocks
on the run's processor (the entries of `ComputeDistanceToSurface` and `FindLiquidSurface`), used by stage 9's range
part; `batches::ReadBatch` moves its range into the whole buffers with `pressure::WriteSnapshotRange` (A15). In
`kBatchedWithSolvesInPieces` the liquid and gas `PressureBuffers`, the `ViscosityBuffers` and the `GasBuffers` are
built with `max_loaded_blocks = max_solve_loaded_blocks` (their batched layout: state and geometry whole on the
processor, solver scratch whole in CPU memory, loaded one piece at a time); no solver signature changes, so the
stage calls 9 to 13 are the same in all three modes. At the start of each try the step calls the buffers form of
the reset on each solver's buffers (reconcile item 3, 11.32): `pressure::ResetSolveCopyReport(PressureBuffers&)`
on the liquid and the gas `PressureBuffers`, `viscosity::ResetSolveCopyReport(ViscosityBuffers&)` and
`gas::ResetSolveCopyReport(GasBuffers&)` (each does nothing in one piece; `FindSolveBatches` is const, so the step
never resets through it); batches
sums the solvers' `SolveCopyReport` of the try into its `BatchReport`. A batched solve agrees with a one-piece
solve within the solver's tolerance, not byte for byte (A15); each mode repeats its own bytes.
`MemoryReport` then compares the limits with the largest counts reached.

### 5.10 Cancellation and a failed processor

`cancel_requested` is read before the try and before every stage, and is passed as the cancel input of
`pressure::SolvePressure`, `viscosity::SolveViscosityAndPressure` and `gas::SolveGas`, which read it at each
progress read and return their `kCancelled`; a solve in pieces also reads it before every batch load (A15). It
is never a field of a settings type. A set flag ends the try as
a reject with `kCancelRequested`: nothing is committed, the simulation is intact and can step again. It is never
reported as an error. `kRunOnCallFailed` is different: the GPU context may be unusable (report, architecture
"Cancellation is not a numerical error"). The simulation is marked unusable, every later call returns
`kSimulationUnusable`, and no snapshot is read from it; the last saved state is what remains.

### 5.11 Where the step hands over to cell resize and batches

`TryStep` returns after stage 20. At that point the swaps are done, `state_version` is `v + 1`, the tries are
dropped, and on the GPU no work of the step is in flight. `AdvanceTo` then calls `resize_cells` if set,
records stage 21, and only then `on_step_done`. The resize works through `BuildResizeTry`, `ApplyResizedGrid`,
`CommitResize`, `RejectResize` (3.10): it builds the resized grid into `free_grid_storage()`, binds the try
fields and the `SolidsAtFaces` storage to it, fills them (the gas arrays through `gas_try()` and
`gas::ToArraysToRemap`), and commits. A failed or rejected resize keeps the committed step: time stays advanced
and the step is not replayed (report, architecture "The physical step keeps a fixed topology"). The next
`TryStep` must carry the `state_version` after the resize. Reconcile (11.28, 11.29): on a GPU grid cell resize
builds the resized grid on the CPU for its map, uploads the same cell list (8 B per cell, counted in
`ResizeCellsReport`) and calls `BuildResizedGrid(resize_try, ResizedCells{{}, &cell_buffer, count,
&cpu_resized_grid})`, which builds the GPU grid into `free_grid_storage()` and binds the try (version and block
count checked against the CPU grid); in a CPU run it may pass the list (`ResizedCells::cells`) or build the grid
itself and call `ApplyResizedGrid`. In a batched GPU run `CellResize` reads `FindCpuGridCopy` instead of making its
own `ReadGrid` copy, builds its CPU resized grid into `free_cpu_grid_copy_buffers()` and hands it over with
`ApplyResizedCpuGridCopy`; `CommitResize` swaps the CPU copy pair with no copy (5.3). Inside a step, the
grid-only region acts through `apply_grid_only_region` (end of stage 4) and `move_grid_only_liquid` (end of stage
16); a batched run through `run_stages`.

### 5.12 Fault injection without a hook

`TryStep` calls `TryStepWithStages(simulation, request, kStages)`. The table is a real seam with a production
user (the batched run), declared in a library header that the CLI and the add-on do not include. A test builds
a copy of `kStages` in which entry `k` calls the real stage and then returns a chosen `RejectReason`, and passes
it to `TryStepWithStages`. `Simulation` has no setter, no test flag and no branch that production code does not
take. Where a real cause exists it is also used (a NaN velocity, a limit of one particle, a particle sent at ten
cells per step, a gas source too strong for the step).

### 5.13 Snapshot and continued run (NEW, from I/O)

`ReadSnapshot` copies, between steps: `particles::ReadSnapshot(particles, particle_fields)`, the committed
grid's active cells and version, every committed `GridFields` array by active cell or face piece (with
`grid::ReadValues`), the air-pocket list and history, source counters, `CommittedStepLimits`,
`solids::ReadSnapshot(solids)`, `gas::ReadGasAmountsSnapshot(gas_amounts, grid)`, limits.
`BuildSimulationFromSnapshot` needs a full snapshot (`particles.is_full`): it rebuilds the committed grid with
`grid::BuildGrid(GridInputs{grid_frame, active_cells, grid_version}, storage)`, then
`particles::BuildParticlesFromSnapshot(settings, snapshot.particles, grid)` (stored order kept, never
regrouped), `gas::BuildGasAmountsFromSnapshot(*snapshot.gas, grid, law, run_on)`,
`solids::BuildSolidsFromSnapshot(solids, snapshot.solids)`, writes the `GridFields` arrays with
`grid::WriteValues`, and restores the counters, air pockets and `CommittedStepLimits` as stored. The next steps
are then the bytes of an uninterrupted run (test in 7). A mismatch of material ids, counts, grid or arrays is
`kSnapshotDoesNotMatch`. The step converts nothing for output: render output turns `snapshot.gas` into volume
channels with `gas::ToGasSnapshot(amounts, law)`.

### 5.14 `CheckSceneCanRun`

It runs every check and keeps every finding: `materials::BuildMaterialTable` (`kSurfaceTensionsDiffer`, bad
materials as `kBadInput` findings); `transfer::CheckMaterialSets` and the set rules of 3.2
(`kMaterialSetsNotValid`, `kPutAllLiquidsInOneSet`); the viscosity summary from `solids::ComputeSolidsSummary`
and the materials, then `viscosity::CheckViscosityCanRun` (A24: snow and springy solids beside a viscous liquid
give no finding; A27: a free solid beside one gives no finding either, so no combination of materials is refused
because of viscosity; replaced: "a free solid beside one stays `kViscousLiquidWithFreeSolid`");
`solids::CheckSolidsCanRun(solids,
smallest_cell_width, largest_cell_width)`, one `kFromSolids` finding per entry with its `solid_id`;
`contact::CheckSolid` and `CheckContactTolerance` per solid (`kSolidThinnerThanTolerance`); with gas,
`gas::CheckGasCanRun(GasSceneSummary{...}, *gas)` and `gas::CheckGasSources(gas_sources, gas_wind)`
(`kFromGas`); grid frame and levels (`kDomainTooLargeForSmallestCell`); ingredient counts; known limits
(`kInternalForceAcrossSizeChange`, `kWavePaddleReceivingVolume`) as `kExperimental`. (Two liquids that do not
mix give no finding: 11.16.) Liquid and gas in one scene, and a batched run, are not
exclusions (A10).

**The stage choice (A25).** The check fills `ViscositySceneSummary::has_viscous_ingredient`: a liquid ingredient
is viscous exactly when its `viscosity` setting is above 0 (`> 0.0`, no threshold); the scene is viscous when any
liquid ingredient is. Then stage 11 runs, else stage 12. `CheckSceneCanRun` and `BuildSimulation` (and
`BuildRunPlan`'s `viscous_path`) apply this rule to the ready scene, once; retries, shortened steps, limit raises,
cell resizes, batches and the viscous effect per step (3.5, advice only) never change it. The water preset has
viscosity 0, so water scenes take stage 12.

## 6. CPU and GPU

**6.1 Shared formulas.** `ComputeVelocityAfterGravity(velocity, gravity_along_axis, step_duration)`,
`ComputeTravelStepLimit`, `ComputeTravelInCellWidths`, `FindDomainSideCrossed`, `ComputeRandomWords`,
`ToRejectKind`. Everything else the stages run is another module's shared formula.

**6.2 Loops and layout.** CPU: each stage loops over ranges, then sets in the step's order, then blocks in index
order, then cell places or particles in grouped order; a thread owns whole blocks, so each destination value is
written by one thread. GPU: one compute stream; each stage's module launches its own functions over blocks; the
step's own functions are one launch per material and range. Between stages no array goes to the CPU: a stage
returns one status record (its module's failure, at most 64 bytes) that the CPU reads before starting the next
stage. Other values returned to the CPU per try: the needed cell count and the grid build's status record (stage
2; A17, no block ids and no tables cross), in a one-piece GPU run the open-sea site bits and removal list and the
marker lists (stage 1) and, with `choose_cell_size`, the new cells (stage 2), all counted copies (6.5);
`AirPocket` records and `BlobHistory` (stage 9), the solves' progress records and `GasBlob` records (11 to 13),
`StepReport` sums, `CommittedStepLimits` (stage 19), which are the one uncounted class (each bounded; "small
records" in the I/O budget). A run is all CPU or all GPU (A17); in a batched GPU run only what does not fit (the
particles, and in `kBatchedWithSolvesInPieces` the solver scratch) stays in CPU memory, moved in planned, counted
copies (6.5; `BatchReport`, `SolveCopyReport`). The committed
state is two copies of what a step changes (committed and try) and one copy of the rest. There is no ring of
committed versions: `ReadSnapshot` is a blocking copy between steps, so no reader holds a committed buffer across
a swap (decision 8).

**6.3 The step's own memory estimate entries** (`BuildStepMemoryEstimate`):

| Entry | Counted per | Bytes each | Copies | Kept for |
|---|---|---|---|---|
| `GridFields::pressure`, `has_pressure`, `pocket_of_cell` | cell place | 8 + 1 + 4 | 2 | whole run |
| `GridFields::velocity_after_pressure` (2 copies with a pressure law, gas or cell resize, which read it next step; else 1 try copy) | face place | 8 | 2 | whole run |
| `GridFields::gas_velocity_after_pressure` (gas) | face place | 8 | 2 | whole run |
| the seven region arrays (`has_cell_resize`) | cell place | 34 + 8 x ingredients | 2 | whole run |
| `carried_pressure`, `mixture_density`, `sensed_volume_ratio`, `added_volume_rate`, `bulk_stiffness`, `target_volume_ratio` | cell place | 8 each | 1 | one step try |
| `was_solved` of the liquid set; `was_solved`, `cell_was_solved` of the gas set | face, face, cell place | 1 | 1 | one step try |
| `keep_active` (gas) | cell place | 1 | 1 | one step try |
| `"step: material face velocity after pressure"` (A18, scene with a blocking set; written by stage 11 or 12, A24) | given (solids' `max_motion_map_entries`) | 8 | 1 | one step try |
| `"step: sealed gas pressure"` (A18, scene with gas and liquid) | given (`max_gas_blobs`) | 8 | 1 | one step try |
| `"step: material face velocity after solve"` (A26, viscous scene with a blocking set; written by stage 11) | given (solids' `max_motion_map_entries` x 2) | 8 | 1 | one step try |
| `"step: free solid motion"` (A27, viscous scene with a free solid; written by stage 11) | given (the free solid count) | 48 | 1 | one step try |
| `"step: free solid motion before pressure"` (A27, viscous scene with a free solid; written by stage 10, read by `BuildPressureEquations`) | given (the free solid count) | 48 | 1 | one step try |
| `InternalForceTotals` per block, per set with internal force; `DriveWaves` block partials | cell block | 24; 32 | 1 | one step try |
| `GridAtParticle` | particle | 200 | 1 | one step try |
| `ParticleContact`; `SolidReaction` (only with a reacting solid) | particle | 56 (`sizeof`, plan decision "Contact"); 56 | 1 | one step try |
| needed block id and its order | particle | 8 + 4 | 1 | one call |
| source counters, open-sea site bits, air pockets | given count | 24; 1/8; 80 | 2 | whole run |

The two grid storages are the grid's entries (`BuildGridMemoryEstimate(run_on, 2, ScratchSharing::kOneForAllBuffers)`:
one build scratch for both, its entries with `copies = 1`; reconcile 11.31; on a GPU run two grids with one scratch
cost 2,840 B per block); each set's transfer arrays, both `PressureBuffers`, the solids arrays the step owns and the
gas buffers are those modules' entries. Every entry's size is `sizeof` of its stored type with a `static_assert`, and
every entry sets `where` (`kCpu` for source counters and open-sea site bits on a GPU run) (plan decisions).
GPU run (A17): `"step: active cell list"` (8 B per cell place, one step try, `kRunOn`) and, with `has_cell_resize`, `"step: new cells for the size
choice"` (9 B per cell place, one call, `kCpu`). Batched entries: 6.5.

**6.5 Buffers in batched runs and planned copies (A17, batched-run buffers; plan Appendices A and B).**
*Placement rule* for whole try arrays (both batched modes, either processor): a whole array lives on the run's
processor when a part at `kWholeGridOnProcessor` reads or writes it; else in CPU memory when a `kWholeStateOnCpu`
stage reads or writes it, the commit keeps it, or a later pass reads it (a carrier between passes); an array read
and written only inside one pass has no whole instance, only the loaded one. Stage places (batches design 5.8 with
A17): 1, 2, 17, 20, 21 whole state on the CPU; 3 to 9 pass 1 (3's `run_whole_grid` and 9's whole part on the
processor); 10 to 13 on the processor; 14, 15, 16, 18, 19 pass 2 (16's and 19's whole parts on the processor).
Pinned results: each set's `GridFromParticles` and `CellsFromParticles`, the six cell arrays of stage 10,
coverage, `grid_edge_is_open_to_sky`, pass-2 block partials read by 19's whole part: `kWholeOnProcessor`;
`before_forces`, `after_pressure`, both `SolidsAtFaces`, `SolidReaction`: `kWholeInCpuMemory`; `GridAtParticle`,
`ParticleContact`: `kLoadedOnly`.

*Inventory, GPU run* (columns: one piece / `kBatched` / `kBatchedWithSolvesInPieces`; G GPU, C CPU memory, L loaded
instance on the GPU; Mn = mismatch and its resolution; a CPU run has no mismatch):

| Stage | Buffers | 1 / B / P | Resolution |
|---|---|---|---|
| 1 | source counters | C / C / C | — |
| 1 (a) | `NewParticles` into the particles try | C into G / C / C | M1 one piece: copy (`AddParticlesResult`, O1) |
| 1, 2, passes | particles | G / C locked / C | loaded per batch (`BatchReport`) |
| 1 (b) | open-sea bits, removal list | G / C / C | M2 one piece: read back (O2); batched: marked on the CPU |
| 1 (c) | committed gas amounts; committed `gas_velocity_after_pressure` | G / G+C / G+C | M3 mirrors once per committed version (B1) |
| 1 (c), 2 | committed and try grid tables | G / G+C / G+C | M7 CPU copy built from the same list, swapped with the grid; after a resize, cell resize's CPU resized grid (`ApplyResizedCpuGridCopy`, no copy; B5 withdrawn, reconcile) |
| 1 (c) | `MarkerChanges` lists | C / C / C | M3 one piece: read back (O3); batched: CPU form on mirrors |
| 2 | needed cell list | G / C sent to G / same | M6 one piece: stays on the GPU, count only; M7 batched: uploaded (B4) |
| 2 (a) | `keep_active`; committed `is_grid_only_region` | G / G+C / G+C | M4 computed on the GPU, read back (B2); M5 mirror (B3) |
| 2 (b) | `choose_cell_size` inputs | C needed / C copy / same | M8 one piece: new cells back, levels sent (O4); batched: the CPU copy |
| 2 (c), (e) | try `GridFields`, six cell arrays, gas try, `gas_seen` | G / G / G | — |
| 2 (d) | grouping tables | G / C / C | M7 grouped against the CPU copy |
| 3 to 9 | `TransferScratch`, `MaterialBuffers`, `ContactBuffers`, `SolidsBuffers` of pass stages | G / L / L | M9 loaded instances; whole instances at 0 particles |
| 3, 14 | `SolidsAtFaces` start, end | G / C whole + L / same | placement rule |
| 3 | `grid_edge_is_open_to_sky` | G / G whole / same | M11 flood moved to 3's `run_whole_grid` after pass 1 |
| 4 to 10 | sets' transfer arrays, coverage, geometry, six cell arrays | G / G whole + L / same | M12 placement rule |
| 5, 14, 15 | `before_forces`, `after_pressure` | G / C whole + L / same | carriers between passes |
| 8, 17 | `CollideSolids` block partials | G / L then C / same | M10 `solids::WriteBlockPartialsRange` in `ReadBatch` (`BatchReport`) |
| 9 range | piece `PressureBuffers` | — / L / L | A15 |
| 9 whole, 10 to 13 | liquid and gas `PressureBuffers`, `ViscosityBuffers`, `GasBuffers` | G / G / G + scratch C | A15 `SolveBatches` (`SolveCopyReport`) |
| 10 to 12 | motion map, blocking faces, free-solid entries | G / G / G | — |
| 15, 16 | `GridAtParticle`, `ParticleContact` | G / L only / same | one pass uses them |
| 16, 17 | `SolidReaction` | G / L then C whole / same | read back with the pass |
| 17 | stage-17 `SolidsBuffers` | G (results C) / C instance / same | M10 CPU `SolidsBuffers` (`FindContactReactionBuffers`) |
| 18 | `block_in_whole_grid`; recorder scratch | G (empty span) / L, map copy C / same | M13 map copy through `ApplyGridPiece`; recorder copies `ReadRecorderCopyCounts` |
| 19 whole | pass-2 block partials | G / G whole / same | placement rule |
| 20 | swaps, `gas::CommitTry`, `ApplyPlateRemoval` | G / grid pair G + copy pair C, particles C / same | M7 copy pair swapped |
| 21 | `CellResize` grid copy, mirrors | C own `ReadGrid` / C step's copy / same | M14 reads `FindCpuGridCopy`; builds the CPU resized grid into `free_cpu_grid_copy_buffers()` (its own two CPU grids neither allocated nor counted, reconcile) |
| between steps | `ReadSnapshot` | G to C / `GridFields` G to C / same | M15 per frame, counted by I/O |
| every stage | status records, progress, `AirPocket`, `BlobHistory`, `GasBlob`, per-solid results, box marks | G to C | the one uncounted class |

*Batched-run entries* (all `where = kCpu` except the list): `"step: committed gas volume on the CPU"` (8 B per cell
place), `"step: committed gas velocity on the CPU"` (8 B per face place), `"step: committed grid-only marks on the
CPU"` (1 B per cell place, with cell resize), all `KeptFor::kWholeRun`, refreshed once per committed
`state_version`; `"step: keep active on the CPU"` (1 B per cell place, `kOneStepTry`); `"step: grid copy on the
CPU"` = `grid::BuildGridMemoryEstimate(RunOn::kCpu, 2, grid::ScratchSharing::kOneForAllBuffers)` with `where` set
to `kCpu` (the pair shares one CPU `GridScratch`, 11.31); `"step: active cell list"`
(`kRunOn`); the stage-17 `"solids: "` entries. The grid copy and the list exist only in a batched GPU run; a batched
CPU run has the mirrors. Loaded-only entries at `max_loaded_blocks` / `max_loaded_particles`, `kRunOn`; no
`kRunOn` entry is counted per particle at the limits.

*The step's own planned copies* (counted in `StepReport::copies` and `ReadCopyCounts`; summed at the limits into
`RunPlan::planned_step_copies`; cell and face places at `limits.max_cell_blocks`; `A_m` bytes per new particle of
material m over its `k_m` carried `NewParticles` arrays):

| Row | Mode | When | Direction | Bytes (at most) | Copies |
|---|---|---|---|---|---|
| O1 | one piece | each `AddParticles` call | to | `max_added_per_try_m * A_m` per material | `k_m` per material with additions |
| O2 | one piece, open sea | stage 1 (b) | from | `ceil(sites / 8) + 8 * max_added_per_try` | 2 |
| O3 | one piece, gas | stage 1 (c) | from | `max_added_per_try_gas * A_gas + 8 * max_particle_count_gas` | 2 |
| O4 | one piece, `choose_cell_size` | stage 2, new grid version | from, to | `8 * cell places` from, `1 * cell places` to | 2 |
| B1 | batched, gas | stage 1 (c), new committed version | from | `8 * cell places + 8 * face places` | 2 |
| B2 | batched, gas | stage 2 (a), each try | from | `1 * cell places` | 1 |
| B3 | batched, cell resize | stage 2 (a), new committed version | from | `1 * cell places` | 1 |
| B4 | batched GPU | stage 2 (b), new grid version | to | `8 * cell places` | 1 |
| ~~B5~~ | withdrawn (reconcile, 11.29): the CPU copy after a resize is cell resize's CPU-built grid, no read-back | | | 0 | 0 |

B1 to B3 also in a batched CPU run (CPU to CPU). A one-piece CPU run counts no copy. Copies of batches, of the
solvers' `SolveBatches` and of cell resize are counted in those modules' reports.

**6.4 Tolerance, CPU against GPU.** After one committed step from the same snapshot: ids, counts, active
cell ids, flags, `RejectReason`, `rejected_in`, `active_limit`: equal. Positions:
`1e-10 * smallest cell width + 1e-12 * largest coordinate`. Velocities and face velocities:
`1e-9 * (largest speed + |gravity| * step_duration)`. Pressure and gas amounts: the pressure and gas modules' own
tolerances. After many steps only sums are compared (ported from the old paper's parity rule): liquid volume
0.1 %, kinetic energy 2 %, front position one smallest cell width. The owner's hardware run calibrates the
one-step numbers.

## 7. Tests

**Analytic and conservation.**

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| Free fall | one block of liquid, no solid, gravity `{0,0,-9.81}`, 20 steps, `own_motion_kept` 0, 0.5, 1 | velocity `-9.81 t`; each particle's z drops by the sum of `v * step_duration` | 1e-12 relative (old paper: the snapshot comes before forces) |
| Column at rest | closed box, liquid at rest, 100 steps | largest speed stays below the solve limit times the cell width; `time == target_time` exactly | pressure's limit; exact |
| Reject changes nothing | for each stage 1 to 19 and each `RejectKind`: snapshot, reject after that stage through the table, snapshot | the two snapshots are equal bytes; `rejected_in` is that stage | exact |
| Shorter retry | reject at `d`, then two commits at `d/2`, against two commits at `d/2` from the same state | equal bytes | exact |
| Limits | a run with start limits of half the need and raises (grid storages included), against a run with large limits | equal snapshots; `limit_raise_count > 0`; time not advanced by the rejected tries | exact |
| Sources | rate 10 per second, steps of 0.03 s and of 0.01 s, 1 s | 10 particles, consecutive ids, the same positions at birth in both runs; a reject uses no id and no child number | exact |
| Even steps | `max_step_duration` 0.01, target 0.035 | 4 steps of 0.00875; last time equals the target | exact |
| Each limit | one tiny scene per `StepLimit` | that `active_limit`; duration equals the formula | 1e-15 relative |
| Travel reject | one particle at 0.6 cell per step | `kParticleTravelledTooFar`, its id, `shorter_step_duration = d/2`; commits at `d/2` | exact |
| Gas flow reject (NEW) | one gas source too strong for the step | `kStepTooLongForGasFlow` in `kSolveGas`, `shorter_step_duration` = its `limit`; the shorter try commits | exact |
| Open side | particle leaving through the open top; the same with the top closed | removed, amounts in the report equal the particle's; with the top closed it stays | exact |
| Capability | one scene per `CapabilityReason` | that reason, its `CapabilityLevel` and `SuggestedChange`, `module_code`; `CheckSceneCanRun` and `BuildRunPlan` run with `RunOn::kGpu` on a machine with no GPU | exact |
| All findings (NEW) | a scene breaking three rules, two of them solids rules on two solids | every finding, ordered as 3.4, each solids entry with its `solid_id` | exact |
| Material sets (NEW) | the scene of 5.2 without snow; then the liquids given in two sets | every set's arrays carry its `MaterialSetKey`; the liquid set's sums equal a run without markers; two liquid sets: `kMaterialSetsNotValid` | exact |
| Plan (amended, A17 planned cost: `FitsNowhere_PlanSaysDoesNotFit`) | counts chosen so the estimate is just below and just above `allowed_bytes` | `kOnePiece`, then `kBatched`, then a plan with `fit = kDoesNotFit`, `mode = kBatched`, `on_processor` and `on_cpu` filled (was the failure `kDoesNotFit`); `output_estimate` appear under `kRenderOutput` | exact |
| Planned cost (A17, plan Tasks 8, 9, 29) | `BuildRunPlan, OnCpuIsTheCpuRowsPlusSnapshots` (falling block with one source, `kGpu` one piece); `BuildRunPlan, PlannedStepCopies_AppendixBAtTheLimits` (with cell resize, `kGpu`); `BuildRunPlan, PlannedStepCopies_GasRows` (small fire); `BuildSimulation, PlanThatDoesNotFit_IsDoesNotFitAndNothingAllocated`; `BuildSimulation, AllocatedCpuBytesEqualCpuRows` | `on_cpu` = `kCpu` entries + snapshots, `by_module[kStep].on_cpu` holds the source counters, no `kCpu` entry in `on_processor`; planned copies: one piece O1 + O4, batched B3 + B4 (B5 withdrawn, reconcile; gas: O1 + O3; B1 + B2 + B4), CPU one piece zero; `kDoesNotFit` with nothing allocated; `allocated_cpu_bytes == on_cpu - snapshots` | exact |
| Copies counted (A17, plan Tasks 10, 12, 14, 27, 29) | `TryStep, OnePieceCpu_CountsNoCopy`; `AddNewParticles, CopiesOfAddParticlesCounted`; `AddNewParticles, MarkerListCopiesCounted`; gpu: `AddNewParticles, OnGpuSourceCopiesCounted`, `ChooseActiveCells, OnGpu_NoCellIdComesBack`, `ChooseActiveCells, OnGpuSizeChoiceCopiesCounted`, `AddNewParticles, OpenSeaOnGpu_CopiesCounted`, `AddNewParticles, OnGpuMarkerListsReadBackCounted`; cpu: `BuildStepMemoryEstimate, GpuRunHasActiveCellListOnGpu` | CPU one piece: `copies == {}` after 3 tries; try copies = sum of the module copy fields; size choice with 5 new cells: `{2, 5, 40}`; open sea: `ceil(sites / 8) + 8 * removed` in two copies plus the refill's; no cell id comes back without the callback; the list entry `kRunOn` in a GPU plan only | exact |
| GPU stages compared (amended, reconcile: `ChooseActiveCells, OnGpu_EqualsCpu`, `OnGpuTwice_IdenticalBytes`) | stage 2 on both processors | the try grid copied by a test-side `grid::ReadGrid` equals the CPU try grid table by table; the stage status record's entry count equals the CPU needed cell count; the test's `ReadGrid` is not in `StepReport::copies` | exact |
| Placement (A17, plan Task 33a) | `ReadStepArrays, BatchedPlacementFollowsTheRule`; `BuildStepMemoryEstimate, BatchedGpu_PerParticleModuleRowsOnlyLoaded`; `BuildSimulation, BatchedLayout_LoadedTryOwnsModuleBuffers`; `BuildStepMemoryEstimate, BatchedGpu_ContactReactionBuffersOnCpu`; `MoveSolids, BatchedLayout_ContactPushesEqualOnePiece`; `BuildSimulation, BatchedLayout_AllocatedEqualsEstimateByPlace`; `FindSolidSurfaces, BatchedRangesBuildTheSolidsTryOnce` | every `whole_place` equals the rule recomputed from `read_by_stages`, `written_by_stages` and the stage places, pinned arrays as 6.5; per-particle `kRunOn` entries only at `max_loaded_particles`; the four loaded instances sized at the loaded counts, whole `MaterialBuffers` at 0 particles; stage-17 entries all `kCpu`; contact pushes, turning push and work equal one piece (bytes); `kRunOn` and `kCpu` sums each equal the plan; one solids try per step try | exact |
| CPU stages of a batched run (A17, plan Task 33b) | `AddNewParticles, BatchedMarkerInputs_CopiedOncePerCommittedVersion`; `AddNewParticles, BatchedMarkerChanges_EqualOnePiece`; `ChooseActiveCells, BatchedKeepActiveReadOnceCounted`; `ChooseActiveCells, BatchedGridOnlyMarksReadOncePerVersion`; `BuildStepMemoryEstimate, BatchedGpu_MirrorsOnCpuAndListOnGpu`; `FindCpuGridCopy, CpuRunIsTheGridOnePieceGpuIsNull`; gpu: `ChooseActiveCells, BatchedOnGpu_ListUploadCountedAndCopyEqualsGrid`, `CommitResize, BatchedOnGpu_GridCopyTakenFromResizeNotCopied` (reconcile; was `..._GridCopyRefreshedAndCounted`) | 2 copies of `8 * cell places + 8 * face places` per committed version, none on a retry; marker lists equal one piece; one copy of `1 * cell places` per try; mirrors and grid copy `kCpu`, list `kRunOn`, a CPU batched plan has neither grid entry; one upload of `8 * active cells`, `ReadGrid` of the GPU grid equals the copy byte for byte; `ReadCopyCounts` does not grow at a resize commit, and `FindCpuGridCopy` equals `ReadGrid` of the resized GPU grid | exact |
| Plan, A15 (plan Task 8: `AllowedCpuBytes_FromCpuShare`, `SolvesInPieces_SolverRowsCountedOnCpu`) | `on_cpu` given; a plan copied to `kBatchedWithSolvesInPieces` with `max_solve_loaded_blocks` 64 | `allowed_cpu_bytes` by step 3; the pressure entries counted per place move to `kCpu`, the loaded entries at 64 blocks stay `kRunOn` | exact |
| Estimate against build (amended, A15; plan Tasks 33, 36) | `BuildSimulation` for the four reference scenes at small size: one-piece layout, batched layout (forced `kBatched`, hand-written `run_stages`), and batched with solves in pieces (forced, `max_solve_loaded_blocks` = 3 layers) | `allocated_bytes == estimate.on_processor` (the `kCpu` and `kRunOn` sums each in the third); `ReadStepArrays` matches the entries | exact |
| Solves in pieces (A15, plan Task 33) | `SolvesInPiecesWithoutSolvePiece_IsBatchedRunNotBuilt`; `SolveCopyReportsResetEachTry` | `kBatchedRunNotBuilt` with `max_solve_loaded_blocks` 0; each solver's `SolveCopyReport` is zero at the start of every try (reset through the buffers forms of `ResetSolveCopyReport`, reconcile item 3) | exact |
| Blocking faces in the same step (A18, plan Task 28: `SolvePressure, SnowAndJellyFacesMoveInTheSameStep`) | snow and jelly beside water | after stage 12 each set's face velocity on every entry piece equals `blocking_material_velocity_after_pressure` (bytes); other faces unchanged; reject after 10 and 12 changes nothing | exact |
| Sealed gas with the liquid (A18, A24, plan Task 30) | `BuildGasBesideLiquidScene`: `LiquidReadsOpenGasPressureOneStepLate` (top open); `SealedGasSolvedWithLiquid` (top closed); `SealedGasBesideViscousLiquid_SolvedInStageEleven` (closed, syrup; A24, was `..._ReadOneStepLate`); `RejectAfterBuildPressureEquationsWithGas_ChangesNothing` | open: air-end pressure at step n = gas pressure committed at n - 1; closed: committed background pressure = atmosphere + stage 12's `sealed_gas_pressure`, air-end pressure that value + the gas pressure of the last commit; syrup: stage 11 runs, the blob is listed, `sealed_gas_pressure` written by stage 11 and read by stage 13, each committed background pressure = atmosphere + `sealed_gas_pressure`; `ApplyGasStepStart` wrote only the try | exact |
| Viscous liquid beside blocking materials (A24, plan Tasks 7, 31) | `CheckSceneCanRun, ViscousWithSnowAndSpringySolid_CanRun` (was `ViscousWithSnow_IsExcluded`); `TryStep, FiveMaterialsWithSnow_Commits` (was `..._IsViscousLiquidWithSnow`); the five-material tests once marked blocked (reject after every stage, stage trace, continued run, range cut, placement) | no finding; the scene of 5.2 with snow commits a step, stage 11 runs, `ApplyViscousSolveToMaterials` writes both blocking sets (A26; was `ApplyPressureToMaterials`); the others as written | exact |
| Stick and free solids in the viscous solve (A26, A27, plan Tasks 7, 26, 31, 34) | `CheckSceneCanRun, ViscousWithFreeSolid_CanRun` (was `ViscousWithFreeSolid_IsExcluded`); `TryStep, FreeBoxInViscousLiquid_SolvedMotionWrittenBack`; `TryStep, FiveMaterialsWithSnowAndFreeBox_Commits` (the scene of 5.2 with snow and the free box beside the syrup; adds to `FiveMaterialsWithSnow_Commits`); `TryStep, FloatingBoxInSyrup_RestDepthAsInWater` (free box 500 kg/m3, side 0.04 m, in a liquid of 1000 kg/m3 at 10 Pa s, started at rest at its 0.02 m draft, 200 steps of 0.01 s, against the same scene with the water preset, viscosity 0, stage 12) | no finding; the box's motion after stage 11 equals `free_solid_motion`; stage 11 runs, each band face of S and E equals `material_face_velocity_after_solve`, other faces keep the `CollideSolids` velocity, the box equals `free_solid_motion`, the step commits; box centre within `1e-6 w` of the water run each step, speed within `2 * 2e-5 * w` | exact (bytes); `1e-6 w`, velocity bound |
| What counts as viscous (A25, plan Tasks 8, 25) | `BuildRunPlan, ViscousEffectPerLiquid` (water 1e-3 Pa s 1000 kg/m3, syrup 10 Pa s 1400 kg/m3, smallest cell 0.01 m, `max_step_duration` 0.01 s); `CheckSceneCanRun, ViscosityZero_IsInviscid` (water preset, viscosity 0); `TryStep, ViscousPathFixedWhenStepShortens` (a liquid of 1e-6 Pa s, below `kSmallestVisibleViscousEffectPerStep`, with a forced reject and two half steps; and the water preset) | 1e-4 and 0.714285714..., `viscous_path` `kViscousSolve`; `kPressureSolve`, stage 12 and no stage 11; stage 11 in every try for 1e-6 Pa s, stage 12 in every try for the water preset | 1e-15 relative; exact |
| Grid room and one scratch (reconcile item 11, plan Tasks 8, 9) | `BuildRunPlan, BlockRoomFromSceneNotFromDefault`; `BuildStepMemoryEstimate, TwoGridBuffersOneScratch`; `BuildRunPlan, RoomAboveGridLimit_DoesNotFit`; `BuildSimulation, GridBuffersShareOneScratch` | `limits.max_cell_blocks == ceil(1.25 * cover blocks)`, capped by the domain; `grid: build scratch:` entries `copies` 1, other grid entries 2, grid bytes `2840 * max_cell_blocks` + fixed on a GPU plan; `kMaxCellBlocksLimit + 1` blocks: `fit = kDoesNotFit`; bytes walked for the grid equal `BuildGridMemoryEstimate(kCpu, 2, kOneForAllBuffers)`, a new grid version leaves the committed tables unchanged | exact |
| Resize hand-over (reconcile items 12, coordinator 6, plan Task 32) | `BuildResizedGrid, CpuList_EqualsApplyResizedGridOfTheSameGrid`; `BuildResizedGrid, ListOnOtherProcessorOrNoCpuGridOnGpu_IsBadInput`; `ApplyResizedCpuGridCopy, NotABatchedGpuRun_IsBadInput`; gpu: `BuildResizedGrid, OnGpuFromUploadedList_EqualsCpuResizedGrid` | same tables and bound try; `kBadInput`; `kBadInput` (`free_cpu_grid_copy_buffers` null in a CPU run); `ReadGrid` of the result equals `cpu_resized_grid` byte for byte | exact |
| Gas-only step start (A18, plan Task 29) | `BuildSmallFireScene` | `stages_run` 9, 13, no 10 to 12; `ApplyGasStepStart` at the start of 13 with an empty blob list | exact |
| Cancel | flag set before stage `k`, each `k`; and set from another thread during each solve (11, 12, 13) | `kCancelRequested`; snapshot unchanged; the next `TryStep` commits | exact |
| Stage trace | liquid; syrup; snow; gas only; liquid and gas; the scene of 5.2 without snow | `stages_run` is the list of 5.2 with the skipped stages absent; 11 or 12 before 13 | exact |
| Range cut | the same step with one range and with one range per block | equal bytes | exact |
| Stale version | `state_version` one behind | `kWrongStateVersion`, nothing ran | exact |
| Continued run (NEW) | the scene of 5.2 without snow, with a pressure law and cell resize: full snapshot at step 20, `BuildSimulationFromSnapshot`, 20 more steps, against 40 steps uninterrupted | equal `SimulationSnapshot` bytes | exact |
| Callbacks (NEW) | `resize_cells`, `make_foam_and_spray` and `on_step_done` with three different contexts | each function receives its own context | exact |
| Solid and liquid momentum | free box dropped into still liquid, 50 steps | change of liquid momentum plus change of solid momentum equals gravity's impulse | 1e-8 relative |
| Four reference scenes, small | 0.2 s each | commits, liquid volume within pressure's reported volume change | pressure's limit |

**Scene tests handed over by the solids and gas plans** (plan Tasks 34, 35; `plan-decisions.md` "Solids", "Gas",
amended for A18). Solids: `WavePaddle, RegularWave_SpeedAndHeight`; `Cut, PlateThroughBlock_SidesNeverMix`
(without its split-and-merge part); `Platen, SnowBesideWater_SetsKeepTheirVelocities` (amended: the snow faces
the water presses on take stage 12's `blocking_material_velocity_after_pressure` in the same step, bytes); NEW
`SpringyBlock, FifthDensity_FloatsAtDisplacedDepth` (cube 0.1 m, density 200, on water 0.15 m deep in a closed
0.4 x 0.4 x 0.3 m tank, cells 0.0125 m, steps 0.01 s through `AdvanceTo` for 2 s: mean bottom height over the
last 0.2 s is 0.02 m below the still surface within 0.00625 m; vertical velocity amplitude over the last 0.2 s at
most that over 0.2 to 0.4 s); `FreeBox, FallsThroughGas_NotPushedByGas`; `FreeBox, HalfDensity_FloatsHalfSubmerged`
and `FreeBox, OffCentreSplash_ReactionEqualsLiquidChange` through `TryStep`. Gas:
`TryStep, GasHeatPulse_RejectedAndCommittedFramesKept`; `TryStep, GasStepLimitTakenIntoNextDuration` (amended,
A24: the small fire only; every chosen duration after a commit is at most that commit's `GasResult::step_limit`,
the flow-share limit; `kGas` active at least once); NEW `TryStep, SealedPocketAboveWater_NoSealedStepLimit` (the
heated sealed pocket above inviscid water: `GasResult::step_limit` has no sealed term, +infinity with no outflow;
steps of `max_step_duration` commit; each committed background pressure equals atmosphere + that step's
`sealed_gas_pressure`, exact); NEW (A24) `TryStep, SealedPocketAboveSyrup_NoSealedStepLimit` (the heated sealed
pocket above syrup, 10 Pa s, 1400 kg/m3: no sealed term; steps of `max_step_duration` commit; each committed
background pressure equals atmosphere + that step's `sealed_gas_pressure`; stage 11 runs);
`TryStep, SparseFireKeepsItsActiveCells`.

**Handed to the cell resize and batches plan** (A15 names, replacing the byte-identity names):
`Batches, FloatingBox_WithinStepToleranceOfOnePiece` (was `..._SameBytesAsOnePiece`); `TryStep,
BatchedFireBesideWater_WithinStepTolerance` (was `..._SameBytes`); the split-and-merge part of `Cut,
PlateThroughBlock_SidesNeverMix`. Batched against one piece is compared within the solver's tolerance (the step
design 6.4 tolerances for whole runs); batched against batched with another cut, and a repeat, identical bytes.

**Reference and proofs: none.** The step is sequencing and bookkeeping; each formula it runs has its module's
reference. `LiquidTransaction.lean` proves that a function which returns its input on reject returns its
input; the reject test above checks the real state, byte for byte, after every stage on both processors, and
leaves no uncertainty a model proof would remove.

**Comparison with the proof of concept** (supporting only): free fall and the column at rest on same-size
cells, and a dam break to 0.5 s, `main`'s output rotated to Z-up: front within one cell, volume 0.1 %, kinetic
energy 2 %. Bytes cannot match: weights, fills and sums changed.

**GPU.** Every row above on `RunOn::kGpu` within 6.4, plus: two runs of 50 steps of each small reference
scene give equal snapshot bytes; `kRunOnCallFailed` marks the simulation unusable; `ChooseActiveCells` on the
GPU builds a grid whose tables equal the CPU builder's byte for byte (A17); every operation with a `_gpu.cu` has a
`no_gpu` test "on the GPU without a GPU gives `kRunOnCallFailed`". In a container with no
GPU: all CPU rows, the capability and plan rows for `kGpu`, and `BuildSimulation` with `kGpu` returning
`kRunOnCallFailed` with nothing allocated.

## 8. Method paper section

Labels: `<sec-step>`, `<tab-step-stages>`, `<sec-try-and-commit>`, `<sec-step-duration>`,
`<sec-capability>`, `<sec-sizing>`; figures `<fig-step-stages>`, `<fig-try-and-commit>`, `<fig-step-tries>`.

Outline: 1 one step for every scene, and why the grid is fixed inside it; 2 material sets, and the stage table
with what each reads and writes; 3 the two time marks (before forces, after pressure) and why wave relaxation
follows the first; 4 the liquid solve, then the gas solve, and what each reads of the other; 5 try and commit:
what is copied, what is swapped, what a reject leaves alone; 6 three ways a try ends without a commit (shorter
step, larger limits, cancel) and why they differ; 7 the step duration and its limits; 8 the capability check as
a predicate over the scene; 9 sizing: counts, memory estimate, fit, one piece, batches, or batches with solves in pieces.

Figures: `<fig-step-stages>`, the 21 stages as a column, grouped particles / grid / particles, with one lane per
material set, arrows for the values that cross stages and the two snapshots marked; `<fig-try-and-commit>`,
committed and try copies of each owner side by side, shared scratch below, the swap at commit and the drop at
reject; `<fig-step-tries>`, a time line with one committed step, one reject followed by two half steps, and one
limit raise followed by the same step.

Pseudocode: `TryStep` (the numbered stages), `AdvanceTo` (choose, try, branch on `RejectKind`),
`ComputeStepDuration`, `ApplyLimits`. Cite from the old bibliography: ref 4 (Fei 2021, the velocity for this
step and the half-cell limit), ref 20 (Jacobsen 2012, relaxation after the snapshot), ref 9 (Larionov 2017, the
joint solve of stage 11), refs 5 and 15 (Wang 2020; Fei, Huang, Gao 2021: state kept on the GPU between stages).
The random words cite Salmon, Moraes, Dror and Shaw 2011 by author and year (decision 9).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `LiquidSubstepScheduler3D` | `Simulation` | NEW |
| `Attempt`, `AttemptWith...` (22) | `TryStep` | DECIDED |
| `AdvanceFrame`, `AdvanceFrameWith...` | `AdvanceTo` | NEW verb |
| `AttemptImpl` with its 13 parameters | `TryStepWithStages`, `StageTable` | NEW |
| `LiquidPhysicalStage3D` and its values | `Stage`, the names of naming review 4.7 | DECIDED; four NEW |
| the draft stage `ApplyViscosity` | `kSolveViscosityAndPressure` | A6 |
| the draft stage `MoveGas`; gas branch `AdvanceGasSubstep3D` | `kSolveGas` | gas design |
| `velocity.state`, `velocity.travel` | `velocity.for_next_step`, `velocity.for_this_step` | A1 |
| `LiquidAttemptOutcome3D`, `LiquidAttemptDisposition3D { kAccepted, kRetryDt, kCapacity, kFatal }` | `StepResult`, `RejectKind { kNone, kShortenStep, kRaiseLimits, kCannotContinue }` plus `kCancelled`, `kBadRequest` | DECIDED / NEW |
| `absl::StatusCode::kAborted`, `kResourceExhausted` as outcomes | `RejectReason` | DECIDED |
| `retry_duration` | `shorter_step_duration` | DECIDED |
| `LiquidSceneInputs3D`, `LiquidStepOptions3D` | `ReadyScene`, `StepSettings` | DECIDED words |
| `LiquidStateCapacities3D`, `capacity_replay_reservations` | `SimulationLimits`, `ApplyLimits` | DECIDED pattern `max_...` |
| `RestartFrameWithCapacity`, `capacity_replays` | `ApplyLimits`, `limit_raise_count` | NEW |
| `LiquidFrameTransaction3D`, `committed_`, `trial_`, `frame_start_` | the committed and try copies inside `Simulation`; no frame-start copy | VERIFIED by reading `liquid_state_3d.cc:690-857` |
| `clock.epoch`, `accepted_step_ordinal`, `substep_ordinal`, `frame_ordinal` | `state_version`, `step_count`; frames are the caller's | DECIDED (`epoch`) |
| `LiquidStepDiagnostics3D`, `LiquidAcceptedStepMeasurement3D`, `LiquidAttemptMeasurement3D` | `StepReport`, `StepTimings` | NEW / DECIDED |
| `LiquidProgressObserver3D::observe`, `observe_attempt` | `on_step_done`, `on_try_done` | DECIDED rule |
| one `void* context` for all callbacks | `Callback { function; context; }` per callback | decision of 2026-11-14 |
| `LiquidWorkPhaseObserver3D`, `LiquidWorkPhase3D` | `StepTimings::stage_elapsed_seconds` | DECIDED word `elapsed` |
| `LiquidStageFaultHook3D`, `SetStageFaultHookForTesting` | removed; `StageTable` | NEW |
| `LiquidSubstepBackend3D` (function pointers per stage) | removed; `RunOn` and private `_cpu.cc`, `_gpu.cu` | DECIDED |
| `LiquidEmissionSource3D`, `source_ordinal`, `world_position_jitter` | `ParticleSource`, `source_id`, `position_spread` | NEW |
| `fractional_remainder`, `next_child_ordinal` | `owed_particles`, `next_child_number` | NEW |
| `Emit`, `OceanInletParticles3D::Apply` | `AddNewParticles` | DECIDED |
| `PrepareField`, `backend.topology`, `ApplyTrialTopology`, `activation_age` | `ChooseActiveCells`; no age | DECIDED |
| `ScheduledDuration` | `ComputeStepDuration`, `StepLimit` | NEW |
| `ValidateScene`, the `Unimplemented` returns | `CheckSceneCanRun`, `CapabilityReason` | NEW |
| `QueryLiquidWorkspaceBytes3D`, `ReservationBytes`, `reserved_bytes()` | `BuildRunPlan`, `MemoryEstimate`, `ReadMemoryReport` | NEW |
| `device_budget_bytes` | `allowed_bytes` | DECIDED (a hard limit; A22 allows `budget` only for a planned cost shown to the user, I/O's `RunBudget`) |
| `ValidateLiquidState3D` | `CheckStep` | DECIDED |
| `particle_energy_blend`, `0.99` | gone: `own_motion_kept` from materials | DECIDED |
| `world_acceleration` | `gravity` | NEW |
| `LiquidSnapshot`, `CudaLiquidEpochLease3D` | `SimulationSnapshot`, `ReadSnapshot` | DECIDED word |

Verify items checked. Request fields (naming review 7.2): grouped in 5.1; none of the three optional parts
survives. `RejectReason { kBeforeTry, kAfterTry }`: "when" is `rejected_in` (`Stage::kNone` is before the
try), "why" is the reason. `RunOn` in `StepSettings` (rule 14, O10, A12): the CPU and GPU stages take the same
inputs because every array is a `Buffer` on the simulation's processor.

## 10. Not ported

- The 22 `AttemptWith...` methods, the three `AdvanceFrameWith...` loops, `bool refine`, `bool dynamic_piston`,
  `bool coupled_piston`, the selector callbacks (`liquid_substep_3d.h:380-399, 453-662`).
- `SetStageFaultHookForTesting`, `LiquidStageFaultHook3D`; "Qualification limits" (`h:404`).
- `ExecuteAdaptive` as a second stage list (`liquid_substep_3d.cc:1066-1120`); `AdaptiveCarrierMode`.
- The frame inside the step: `BeginFrame`, `FinishFrame`, `frame_start_`, `RestartFrameWithCapacity`
  (`liquid_state_3d.cc:696-714, 828-857`). A frame is a target time given to `AdvanceTo`.
- `CopyState(committed_, trial_)` per try (`state.cc:723`).
- `LiquidSubstepBackend3D` and `Complete(backend)`; CPU staging between GPU stages
  (`cuda/liquid_substep_3d_host.cc`, `liquid_transfer_stages_3d_host.cc`); pinned CPU mirrors of device
  ranges; graph capture (`liquid_arena_3d_host.cc:428-480`, performance tuning).
- Scene-specific surfaces and their checks: `LiquidAnalyticSphere3D`, `LiquidHorizontalPlaneSurface3D`,
  `LiquidIsolatedCovarianceDrop3D` with its 552 particles (`h:23-48`, `staging.cc:407-436`),
  `CorrectIsolatedDropProjectedChange`, `SetIsolatedDropPressureLifts`.
- `LiquidStepCavityProvider3D`, `LiquidSceneCavity3D` callbacks (`h:64-95`): air pockets are pressure's.
- `mass_per_quantum`, `momentum_per_quantum` (`h:125-126`): fixed-point deposits are gone.
- The whole-cell travel field (`RegularTransportSample`, `cc:2656-2662`); `RouteUnsupportedPressureImpulse`,
  `ApplyBoxFaceSamples`, `ApplyOrientedBoxFaceSamples`, `OrientedFloatingBoxLoads3D` (`cc:72-365`,
  `h:285-301`): one motion map in solids.
- `motion_sample_intervals`, `whitewater_capture` in the step's settings: outputs belong to the I/O and CLI
  design and the surface, foam and spray design.
- The step number in the random key (`staging.cc:647-655`); the 4096-site limit (`staging.cc:505`).
- The gas branch's own sub-step loop and its CPU staging between the pressure callback and transport.
- `proofs/Fluidity/LiquidTransaction.lean`; no Python reference for the step.
- A ring of committed versions, a second stream for output, a device-side retry loop: later, behind
  `ReadSnapshot` and `AdvanceTo` as they are.

## 11. Decisions

Already settled and used here: A1 (`for_next_step`, `for_this_step`), A2 (`StepFailure`, `std::expected`), A5
and A6 (viscosity runs the joint solve; stage 11 `SolveViscosityAndPressure`, stage 12 `SolvePressure`), A8
(material sets), A9 (no sand), A10 (gas and liquid together), A12 (`RunOn` is the caller's), Q10 (`CommitTry`,
`RejectTry`; hence `CommitResize`, `RejectResize`), Q22 (regroup when `CheckGrouping` fails), Q39, Q40, Q41,
Q47 (open sea in `pressure/`), Q52. Q41 ("`BuildGrid` runs on the CPU, also for GPU runs"; block ids come down) is
replaced by A17 (11.12).

1. **`StepRequest` has no optional parts.** The grouping of 5.1 leaves nothing to put in them.
   ```cpp
   struct StepRequest { double step_duration; std::uint64_t state_version; const std::atomic<bool>* cancel_requested; };  // a
   struct StepRequest { double step_duration; std::optional<CellResize> cell_resize; std::optional<ParticleChange> particle_change;
                        std::optional<SolidCoupling> solid; };                                            // b) the sketch
   ```
   b lets a caller change cells or solids per try, which is what made 22 methods. Recommendation: a.
   Decision: a. Recommended option (owner's standing instruction, 2026-11-14).
2. **Four new stage names.** `FindSolidSurfaces`, `BuildPressureEquations`, `MoveSolids`, `MoveGas` join
   the 17. Alternative: keep 17 and fold them (3 into `ChooseActiveCells`, 10 into
   `FindLiquidSurface`, 16 and 17 into `MoveParticles`); then a reject test cannot stop between them and
   `MoveParticles` writes solids and gas. Recommendation: the four stages.
   Decision: the four stages, with `MoveGas` replaced by `SolveGas` after the liquid solve (the gas design has
   no `MoveGas`: the flow of amounts is part of `SolveGas`) and `ApplyViscosity` renamed
   `SolveViscosityAndPressure` (A6). Recommended option (owner's standing instruction, 2026-11-14), with A6
   and `changes-to-apply.md` "10 Step".
3. **Two enums in the step.** `StepError` for calls that are not a try, `RejectReason` with `RejectKind` for
   the result of a try, and `bool committed` beside them as in your sketch.
   ```cpp
   auto result = TryStep(simulation, request);  // std::expected<StepResult, StepFailure>
   if (result && !result->committed && ToRejectKind(result->reject_reason) == RejectKind::kShortenStep) ...  // a
   StepResult TryStep(...);  // b) one enum; "simulation unusable" is one more RejectReason
   ```
   Recommendation: a. Also confirm the words `kShortenStep`, `kRaiseLimits`, `kCannotContinue`.
   Decision: a, with those three words. Recommended option (owner's standing instruction, 2026-11-14).
4. **No allocation inside a try needs two things the grid design lacks.** `BuildGrid` returns a new grid and
   `BuildCellValues(grid)` sizes values to it, so a changed cell list allocates. Options: a) the grid gains
   storage for a limit and values that are bound to a version; b) allow allocation in stage 2 only,
   when the list changed, and count two grids at the limit in the estimate. Recommendation: a; it also is what
   `PressureBuffers` already assumes.
   Decision: a: `GridBuffers`, `BuildGrid(inputs, storage)`, `ReleaseBuffers`, `Build...Values(run_on,
   max_cell_blocks)` and `ApplyGridVersion(values, grid)`, now in the grid design. Recommended option (owner's
   standing instruction, 2026-11-14).
5. **Two more Foundation additions:** `CopyBuffer` (copy on the processor, for `ApplyLimits` and for keeping
   the solve's pressure) and `ReadMemoryAvailable(RunOn)`. Alternative for the first: pressure writes its
   result into arrays the step passes, as `ApplyViscosity` does. Recommendation: add both.
   Decision: both added to Foundation (`changes-to-apply.md`, Foundation additions); and pressure writes into
   the step's try arrays (`SolvedArrays`, `AirPocketArrays`), so `CopyBuffer` serves `ApplyLimits` and the
   batched run only. Recommended option (owner's standing instruction, 2026-11-14).
6. **Cancel inside a solve.** Pressure and viscosity have no input for it; without one a cancel waits for the
   end of a solve of up to 256 iterations. Option a: `const std::atomic<bool>* cancel_requested` in
   `pressure::SolveSettings`, read at each progress read, and `kCancelled` in both error enums. Option b:
   cancel only between stages. Recommendation: a.
   Decision: a, with the pointer as a separate input of `SolvePressure`, `SolveViscosityAndPressure` and
   `SolveGas`, never in a settings type (settings are hashed). Recommended option (owner's standing
   instruction, 2026-11-14), as `changes-to-apply.md` "Cancel" states it.
7. **The loop's name and place.** `AdvanceTo(simulation, target_time, cancel_requested)` is a new verb.
   Alternatives: `RunUntil`, `StepUntil`; or no loop in the step and the I/O and CLI design writes it.
   Recommendation: `AdvanceTo`, in the step, because retry, limits and the duration are step rules.
   Decision: `AdvanceTo`, in the step. Recommended option (owner's standing instruction, 2026-11-14).
8. **Two copies and a blocking snapshot, not a ring.** The report prefers a small ring of committed versions
   so output can run beside the simulation. That costs one more copy of what a step changes. Recommendation:
   two copies now; decide the ring in the I/O design, where the memory for it is counted.
   Decision: two copies and a blocking `ReadSnapshot`. Recommended option (owner's standing instruction,
   2026-11-14).
9. **`ReadyScene` is declared in `step/`** and filled by the I/O and CLI design, and module settings that
   change results sit in it while `StepSettings` holds how to run. Alternative: a `scene/` folder.
   Recommendation: `step/`. Also: the random generator (`ComputeRandomWords`, Philox 4x32 with 10 rounds on
   `main`) has no entry in the old bibliography; please supply it or agree to author and year only.
   Decision: `step/` (`changes-to-apply.md` "10 Step"); the generator is cited by author and year only.
   Recommended option (owner's standing instruction, 2026-11-14).
10. **Smaller points to confirm.** Limits raise the step, not the frame, up to four times per step. The
    whole-cell travel field is dropped (transfer left it to this design). `kSourceWindow` is a new duration
    limit. Liquid and gas in one scene is excluded until the gas design. `kMaxTravelInCellWidths` (0.5) and
    `kCoveredReachInCellWidths` (2.5) are constants, not settings. New words: `owed_particles`,
    `position_spread`, `allowed_bytes`, `RunPlan`, `KeepGoing`.
    Decision: all confirmed as written, except liquid and gas in one scene, which is supported from the start
    (A10; `kLiquidAndGasInOneScene` is removed). Recommended option (owner's standing instruction, 2026-11-14)
    and A10.
11. **Blocking materials and sealed gas inside the liquid solve.** Stage 10 calls
    `solids::BuildBlockingMaterialFaces` (scene with liquid and a blocking set) and `gas::ApplyGasStepStart`
    (scene with gas and liquid) before `pressure::BuildPressureEquations` and passes their outputs in
    `EquationInputs`; stage 12 passes the step-owned spans `blocking_material_velocity_after_pressure` and
    `sealed_gas_pressure` in `SolvedArrays` and calls `solids::ApplyPressureToMaterials` right after
    `ApplyPressureToSolids`; stage 17 no longer calls it and no push is owed to particles; stage 13 passes
    `sealed_gas_pressure`; `SealedGasCoupling` is `kInsideLiquidSolve` when stage 12 runs, `kReadOneStepLate` when
    stage 11 runs; a gas scene without stage 10 calls `ApplyGasStepStart` at the start of stage 13; the sealed
    step limit counts only with `kReadOneStepLate`. Replaces, in 5.2: "the liquid solve reads the gas pressure
    committed at the end of the previous step through `gas_seen` (one step late)" (for sealed blobs) and "the
    pressure on blocking materials comes back one step late (`ApplyPressureToMaterials` in 17)".
    Decision: A18 with `literature-coupling.md`; `plan-decisions.md` "Coupling inside the pressure solve (A18)";
    step plan Tasks 26, 28 to 30, 34, 35 (amended). **Replaced in part (A24, 11.26):** `SealedGasCoupling`
    (`kInsideLiquidSolve`, `kReadOneStepLate`) and the sealed step limit no longer exist; stage 11 does what stage
    12 does with both terms.
12. **Grid built on the run's processor.** `grid::BuildGrid` runs on the run's processor each time the cell set
    changes; in a GPU run `ChooseActiveCells` hands it a `Buffer<CellId>` and the entry count as a CPU number and
    takes the counts from `grid.counts()`; no block ids come down and no tables go up. Replaces Q41 and the 5.3
    text "unique ids come back to the CPU (8 bytes per block, grid Q41), which builds the grid".
    Decision: A17; `plan-decisions.md` "Grid built on the GPU (A17)".
13. **A third run mode and the solvers' batched layout.** `RunMode::kBatchedWithSolvesInPieces`;
    `RunPlan::max_solve_loaded_blocks`, `allowed_cpu_bytes`; solver buffers built with `max_loaded_blocks`; the
    loaded try's one-piece `PressureBuffers` for pass 1; copy reports reset per try; batched against one piece
    within the solver's tolerance. Replaces "the solves are not batched" and every byte-identity test between
    modes (renamed tests in 7). Decision: A15; `plan-decisions.md` "Batched solves (A15)"; step plan Tasks 8, 33,
    36 (amended).
14. **`StageTable::run_whole_grid` and the loaded-try access** (`FindLoadedTry`, `FindLoadedGridBuffers`,
    `ApplyGridPiece`, `BlockCopy`, `WriteStepArrayBlocks`). Decision: `plan-decisions.md` "Cell resize and
    batches" (batches plan Tasks 35, 36). Partly replaced: `run_whole_grid` gains `kFindSolidSurfaces` (11.23) and
    `ApplyGridPiece` takes a third argument (11.24).
15. **Foundation as built.** `RunOn`/`RunOnError` in `base/run_on.h`; `MemoryEstimateEntry::where`; memory sizes are
    `sizeof`; `BufferError::kRunOnMismatch` maps to `kBadInput`; `processor_description.h`. Decision:
    `plan-decisions.md` "Foundation `base/`" and "Foundation as built".
16. **Other step plan decisions.** `StageOutcome` in `step_error.h` with a private `ToStageOutcome` per module;
    the `RandomWords` form; `DomainSideCrossed`; `kStages` holds stages 1 to 20; stage 2 fills `solids_at_start`
    and `solids_at_end` once per grid version when stage 3 is skipped; no `kLiquidsThatDoNotMixTouch` (pressure
    supports two liquids that do not mix; `kCapabilityReasonCount` 18, lower by each lifted viscosity exclusion;
    17 since A24, 11.26; 16 since A27, 11.33);
    `step::Support` renamed `CapabilityLevel { kCanRun, kExperimental, kUnavailable }`. Replaces the 3.4 text
    `kLiquidsThatDoNotMixTouch, // pressure kMaterialBoundaryNotSupported (C4)`, `kCapabilityReasonCount = 19` and
    `enum class Support : std::uint8_t { kHasKnownLimit, kNotSupported };`. Decision: `plan-decisions.md` "Step",
    "Pressure (continued)", "Viscosity", "Sand", "Names that broke the glossary (continued)".
17. **A plan that does not fit is returned, and says where its bytes live.** `PlanFit { kFits, kDoesNotFit }` and
    `RunPlan::fit`; `BuildRunPlan` returns a plan with `fit = kDoesNotFit`, `mode = kBatched` and every estimate
    filled instead of failing; `BuildSimulation` fails `kDoesNotFit` on it before allocating; `MemoryEstimate::on_cpu`,
    `ModuleBytes::on_cpu`, `MemoryReport::allocated_cpu_bytes`; `on_processor` counts only `kRunOn` entries.
    Replaces the 5.9 text "otherwise `kDoesNotFit`" as a failure of `BuildRunPlan` and the test row "then
    `kDoesNotFit`". Decision: plan decision, 2026-11-14 (planned cost); owner principle A17 and A22 ("budget" is a
    planned cost shown to the user); step plan Tasks 8, 9 (amended).
18. **The step's own copies are counted and planned.** `CopyCounts` in `run_plan.h`; `StepReport::copies`;
    `ReadCopyCounts`; `RunPlan::planned_step_copies` from the rows O1 to O4 and B1 to B4 of 6.5 at the limits (B5
    withdrawn by the reconcile pass, 11.29); one
    `ReadBuffer` or `WriteBuffer` slice call is one copy; status records, solver progress, `AirPocket`,
    `BlobHistory`, `GasBlob`, per-solid results and box marks are the one uncounted class. Decision: plan decision,
    2026-11-14 (planned cost) and (buffers in batched runs); owner principle A17; step plan Tasks 8, 10, 12, 14,
    27, 29, Appendix B.
19. **Placement rule and loaded module instances.** `ArrayPlace` and `StepArray::whole_place`; whole try arrays
    placed by the rule of 6.5 (replaces 5.9's "the per-try arrays in CPU memory"); the loaded `StepTry` owns
    `TransferScratch`, `MaterialBuffers`, `ContactBuffers` and `SolidsBuffers` at the loaded counts, the whole
    instances keep zero particles; `BuildStepMemoryEstimate(scene, counts, batched_plan)`. Decision: plan
    decision, 2026-11-14 (buffers in batched runs); owner principle A17; step plan Tasks 8, 33a, Appendix A.
20. **Stage 17 on the CPU in a batched run.** A separate `SolidsBuffers` with `run_on = kCpu`, the whole particle
    limit and `max_cell_blocks`; `FindContactReactionBuffers`; batches moves the `CollideSolids` block partials into
    it with `solids::WriteBlockPartialsRange`. Decision: plan decision, 2026-11-14 (buffers in batched runs);
    owner principle A17; step plan Task 33a.
21. **CPU copy of the grid in a batched GPU run.** Stage 2 builds the cell list on the CPU, uploads it for
    `grid::BuildGrid` on the GPU and builds a CPU copy with the CPU builder from the same list; swapped at commit,
    released on reject, refreshed by `ReadGrid` after a committed resize; `FindCpuGridCopy`. Grouping,
    `choose_cell_size`, marker upkeep and `CellResize` read it. Decision: plan decision, 2026-11-14 (buffers in
    batched runs); owner principle A17; step plan Task 33b. **Replaced in part (reconcile, 11.29):** no `ReadGrid`
    refresh after a resize; the CPU resized grid becomes the copy (`ApplyResizedCpuGridCopy`).
22. **Mirrors of committed values and the CPU form of the marker upkeep.** Committed gas volume, gas velocity and
    grid-only marks mirrored once per committed `state_version`; `keep_active` read back per try after
    `gas::FindCellsToKeepActive` ran on the amounts' processor; `gas::FindMarkerChanges(MarkerGridInputs, ...)` in
    both batched modes; `MarkerChanges` lists in CPU memory in every GPU layout. Decision: plan decision, 2026-11-14
    (buffers in batched runs); owner principle A17; step plan Tasks 29, 33b.
23. **Stage 3's whole part.** `FindGridEdgesOpenToSky` moves to `run_whole_grid[kFindSolidSurfaces]`, run at
    `kWholeGridOnProcessor` right after pass 1; stage 3's `run` builds the solids try once per try in a batched run.
    Replaces "`run_whole_grid` non-null only for `kFindLiquidSurface`, `kMoveParticles` and `kCheckStep`" (11.14).
    Decision: plan decision, 2026-11-14 (buffers in batched runs); step plan Task 33a.
24. **Three-argument `ApplyGridPiece`.** The batched run's CPU copy of the piece map reaches
    `FoamAndSprayInputs::block_in_whole_grid` through `ApplyGridPiece(loaded_try, piece, block_in_whole_grid)`.
    Replaces the two-argument form of 11.14. Decision: plan decision, 2026-11-14 (buffers in batched runs).
25. **Stages 1 and 2 on the GPU (one piece) and the size choice.** The needed cell list stays on the GPU, only its
    count comes back; open-sea sites are marked where the particles live (read back in one piece); a
    `choose_cell_size` callback in a one-piece GPU run costs two counted copies (new cells back, levels sent).
    Replaces the Open item "Which buffers stage 1 (and stage 2's marking) needs on which processor in a batched
    run" and, for the batched case, the Open item on `choose_cell_size` (the step's CPU copy). Decision: plan
    decision, 2026-11-14 (buffers in batched runs); step plan Tasks 14, 27 (amended).
26. **Stage 11 carries blocking materials and sealed gas (A24).** Stage 11 mirrors stage 12:
    `viscosity::SolveViscosityAndPressure(..., fields.velocity_after_pressure, fields.pressure,
    viscosity::SolvedCouplingArrays{blocking_material_velocity_after_pressure, sealed_gas_pressure})`, then
    `ComputeWasSolved`, `ApplyPressureToSolids`, then (scene with a blocking set) `solids::ApplyPressureToMaterials`.
    Stage 10 builds the material faces and the sealed blob list whatever stage follows; stage 13 passes
    `sealed_gas_pressure` after either stage; `ViscosityBufferSettings::max_sealed_gas_blobs` = gas's
    `max_gas_blobs` (0 without gas). Deleted: `gas::SealedGasCoupling`, the coupling argument of
    `ApplyGasStepStart` and `SolveGas`, `ComputeSealedGasStepLimit` (the `kGas` limit is the flow-share limit
    only). Exclusions: `CapabilityReason::kViscousLiquidWithSnow` is removed (`kCapabilityReasonCount` 17 at most,
    14 if the plan's three lifting tasks all lift); `kViscousLiquidWithFreeSolid` stays (the free solid's velocity
    would be an unknown of the viscous velocity solve, not covered by A24); no `kViscousLiquidWithSpringySolid`
    (options A and B of reconcile item 4 withdrawn). Tests renamed: `CheckSceneCanRun, ViscousWithSnow_IsExcluded`
    -> `ViscousWithSnowAndSpringySolid_CanRun`; `TryStep, FiveMaterialsWithSnow_IsViscousLiquidWithSnow` ->
    `FiveMaterialsWithSnow_Commits`; `TryStep, SealedGasBesideViscousLiquid_ReadOneStepLate` ->
    `..._SolvedInStageEleven`; `GasStepLimitTakenIntoNextDuration` keeps only the small fire; added
    `TryStep, SealedPocketAboveSyrup_NoSealedStepLimit`. Replaces, in 5.2: "Sealed gas is read one step late here
    (`kReadOneStepLate`)", the scene text "refused today by one finding, `kViscousLiquidWithSnow`", and the Open
    item on a springy-solid set beside a viscous liquid; replaces in part 11.11 and 11.16.
    Decision: owner answer A24 (option C of reconcile item 4); `plan-decisions.md` "Viscous liquids with blocking
    materials and sealed gas (A24), and what counts as viscous (A25)", subsections "A24: exclusions" and "A24: step,
    gas, solids"; step plan Tasks 3, 6, 7, 9, 25, 28 to 31, 33, 33a, 35 (amended). **Replaced in part (A26, A27,
    11.33):** stage 11's `ApplyPressureToSolids` and `ApplyPressureToMaterials`, "`kViscousLiquidWithFreeSolid`
    stays", and `kCapabilityReasonCount` 17 / 14.
27. **What counts as viscous, and the viscous effect per step (A25).** A liquid ingredient is viscous exactly when
    its `viscosity` setting is above 0 (`> 0.0`, no threshold); the scene is viscous when any liquid ingredient is,
    and then stage 11 runs, else stage 12. The choice is made once from the ready scene in `CheckSceneCanRun` and
    `BuildSimulation`; retries, shortened steps, limit raises, cell resizes, batches and the number never change
    it. The water preset has viscosity 0. `BuildRunPlan` fills `RunPlan::liquid_viscous_effects` (one
    `LiquidViscousEffect` per liquid ingredient, with `viscosity::ComputeViscousEffectPerStep` at
    `max_step_duration` and the smallest cell width), `viscous_effect_cell_width`, `viscous_effect_step_duration`
    and `viscous_path` (`ViscousPath { kPressureSolve, kViscousSolve }`); I/O prints the lines. Tests:
    `BuildRunPlan, ViscousEffectPerLiquid`, `CheckSceneCanRun, ViscosityZero_IsInviscid`, `TryStep,
    ViscousPathFixedWhenStepShortens`. Decision: owner answer A25; `plan-decisions.md` "A25: what counts as viscous,
    and the viscous effect per step"; step plan Tasks 7, 8, 9, 25 (amended).
28. **The step builds a resized grid from a cell list.** `ResizedCells` and `BuildResizedGrid(ResizeTry&, const
    ResizedCells&)` (builds into `free_grid_storage()` on the run's processor, binds the try as `ApplyResizedGrid`,
    checks version and block count against `cpu_resized_grid` in a GPU run); `ApplyResizedGrid` stays for a caller
    that built the grid itself. Cell resize calls `BuildResizedGrid` on a GPU grid. Replaces the Open item "How
    `step::ApplyResizedGrid` receives the cell list cell resize uploads" and the 3.10 comment "Which call builds it
    ... Open (11)". Decision: plan decision; `plan-decisions.md`, Reconcile pass item 12 (with coordinator item 1);
    step plan Task 32.
29. **The CPU grid copy after a resize is cell resize's CPU grid.** `ResizeTry::free_cpu_grid_copy_buffers()` and
    `ApplyResizedCpuGridCopy(ResizeTry&, grid::StaggeredGrid&&)`: after a committed resize in a batched GPU run the
    CPU grid copy is the CPU resized grid; `CommitResize` requires the hand-over first and swaps the CPU copy pair
    with no copy; no `ReadGrid` refresh; row B5 withdrawn; cell resize's own two CPU grids are neither allocated
    nor counted in that layout (`GridCopies::kHeldByStep`). Test `CommitResize,
    BatchedOnGpu_GridCopyRefreshedAndCounted` becomes `..._GridCopyTakenFromResizeNotCopied`. Replaces in part 11.18 and 11.21. Decision: plan decision;
    `plan-decisions.md`, Reconcile pass, coordinator's list item 6; step plan Tasks 32, 33b.
30. **What `choose_cell_size` receives as `committed`.** `committed_grid.tables(RunOn::kCpu)`: real tables in a CPU
    run; version 0 and null arrays in a one-piece GPU run, where `cell_resize::ChooseCellSize` reads its own CPU
    copy; `FindCpuGridCopy(...)->tables(kCpu)` in a batched GPU run. The step never reads a GPU grid back for it.
    Replaces the Open item's "what the step passes as `committed` in a one-piece GPU run is not stated". Decision:
    plan decision; `plan-decisions.md`, Reconcile pass item 13 (with coordinator item 2); step plan Tasks 14, 33b.
31. **Grid room from the planner and one grid scratch.** `limits.max_cell_blocks` = the blocks `BuildRunPlan` counts
    from the scene x `start_limit_headroom` (1.25), rounded up, capped by the domain's blocks at the smallest cell,
    at least 1; no default room (`GridSettings::max_cell_blocks` defaults to 0, `kBadInput`); above
    `grid::kMaxCellBlocksLimit` or not fitting, `fit = kDoesNotFit`. The simulation state owns one
    `grid::GridScratch`, built first, released last, rebuilt by `ApplyLimits`, lent to the committed and free
    `GridBuffers` and the loaded try's buffers; the batched GPU run's CPU grid copy pair shares one CPU
    `GridScratch`; memory counted with `BuildGridMemoryEstimate(run_on, 2, ScratchSharing::kOneForAllBuffers)`
    (on a 16 GiB GPU two grids with one scratch cost 2,840 B per block, about 1.06 M blocks in a quarter of
    `allowed_bytes`; the old default `2^22` would take 11.9e9 B). Tests `BuildRunPlan,
    BlockRoomFromSceneNotFromDefault`, `BuildStepMemoryEstimate, TwoGridBuffersOneScratch`, `BuildRunPlan,
    RoomAboveGridLimit_DoesNotFit`, `BuildSimulation, GridBuffersShareOneScratch`. Replaces the 5.9 entry
    `grid::BuildGridMemoryEstimate(run_on, 2)` and 6.3's "on a GPU grid they include the build scratch". Decision:
    plan decision; `plan-decisions.md`, Reconcile pass item 11; step plan Tasks 8, 9, 24, 33, 33b.
32. **Copy reports reset per try through the solvers' buffers.** `pressure::ResetSolveCopyReport(PressureBuffers&)`
    (liquid and gas), `viscosity::ResetSolveCopyReport(ViscosityBuffers&)`, `gas::ResetSolveCopyReport(GasBuffers&)`,
    each zeroing the held `SolveBatches`' report (nothing in one piece), called by the step at the start of each
    try; `FindSolveBatches` stays const. Replaces 5.9's "`pressure::ResetSolveCopyReport` on each solver's
    `SolveBatches` (reached through `FindSolveBatches`)". Decision: plan decision; `plan-decisions.md`, Reconcile
    pass item 3; step plan Task 33.
33. **A viscous liquid sticks, and free solids are in the viscous solve (A26, A27).** Stage 10 (viscous scene)
    also calls `solids::ToFreeSolidsForViscousSolve(motion_map)` and fills the step-owned
    `free_solid_motion_before_pressure` (6 doubles per free solid, `V` then `Omega`) for
    `EquationInputs::free_solid_motion_before_pressure` before `BuildPressureEquations`, and, scene with a blocking
    set, `solids::BuildViscousMaterialFaces` after `BuildBlockingMaterialFaces`. Stage 11 passes
    `ViscosityInputs{..., material_faces, free_solids}` and `SolvedCouplingArrays{
    blocking_material_velocity_after_pressure, sealed_gas_pressure, material_face_velocity_after_solve,
    free_solid_motion}`; then `pressure::ComputeWasSolved`; then `solids::ApplySolvedMotionToSolids` (in place of
    `ApplyPressureToSolids`: each free solid touching liquid takes its solved motion, travel and turn limits checked
    again); then `solids::ApplyViscousSolveToMaterials` (in place of `ApplyPressureToMaterials`: each band face of
    each blocking set takes its solved velocity, the other faces keep the `CollideSolids` velocity). Stage 12 is
    unchanged. Step-owned rows `"step: material face velocity after solve"` (solids' `max_motion_map_entries` x 2,
    8 B), `"step: free solid motion"` (free solid count, 48 B), `"step: free solid motion before pressure"` (free
    solid count, 48 B); viscosity buffers `max_material_faces` = 2 x `max_motion_map_entries`, `max_free_solids` =
    free solid count. Exclusions: `CapabilityReason::kViscousLiquidWithFreeSolid` removed (`kCapabilityReasonCount`
    16 at most, 13 if the plan's three lifting tasks all lift); `SuggestedChange::kSetViscosityToZero` removed
    (`kSuggestedChangeCount` 12; the sand module adds it back with `kViscousLiquidWithSand`). Tests:
    `CheckSceneCanRun, ViscousWithFreeSolid_IsExcluded` -> `ViscousWithFreeSolid_CanRun`; added `TryStep,
    FreeBoxInViscousLiquid_SolvedMotionWrittenBack`, `TryStep, FiveMaterialsWithSnowAndFreeBox_Commits` (adds to,
    does not replace, `FiveMaterialsWithSnow_Commits`, whose check becomes the band-face write-back), `TryStep,
    FloatingBoxInSyrup_RestDepthAsInWater`. Replaces, in 5.2: the scene text "it needs the free box away from the
    syrup", item 11's `ApplyPressureToSolids` and `ApplyPressureToMaterials`, table row 11's "the same as 12"; in
    3.4: `kViscousLiquidWithFreeSolid` and `kSetViscosityToZero`; in 5.14: "a free solid beside one stays
    `kViscousLiquidWithFreeSolid`"; the Open items on the slip at a liquid-material face and on free solids beside a
    viscous liquid; replaces in part 11.26. Decision: owner answers A26 and A27; `plan-decisions.md` "A viscous liquid
    sticks, and free solids in the viscous solve (A26, A27)", sections "Exclusions" and "Step (stage 10, 11)"; the
    pressure plan's `EquationInputs::free_solid_motion_before_pressure` (pressure plan, A27); step plan Tasks 3, 6,
    7, 8, 9, 25, 26, 28, 31, 34, 39 (amended).

**Open** (left open by the plans):

- Replaced (11.25): "Which buffers stage 1 (and stage 2's marking) needs on which processor in a batched run, with
  the particle arrays in CPU memory."
- `choose_cell_size` in a one-piece GPU run costs two counted copies (new cells back, levels sent) and
  `ChooseCellSize` keeps its own CPU copy of the committed grid (what the step passes is decided, 11.30). Owner
  (plan-decisions open point 2): a shared CPU-and-GPU size rule instead of a callback would remove these copies.
- Owner questions raised with A24 and A25 (`plan-decisions.md` "Owner questions raised here"; viscosity's, the
  step follows the answer): the new constant `kSmallestVisibleViscousEffectPerStep = 2.5e-4`. (The slip at a
  liquid-material face is answered by A26, stick; free solids beside a viscous liquid by A27, lifted; 11.33.)
- Owner questions raised with A26 and A27 that touch the step (`plan-decisions.md`, last "Owner questions raised
  here"): the material band (faces within `kFitReach` widths of a liquid cell centre) and "the rest of the material
  keeps its velocity after `CollideSolids` this step" (what stage 11 writes back); where two blocking sets hold one
  face piece beside viscous liquid, the set with the larger face mass is the sample's material side;
  `kSetViscosityToZero` removed from the main sequence, the sand module's `kViscousLiquidWithSand` remaining.
- Stage 19's `CheckSolidExchange` on a free solid moved by stage 11: `ApplySolvedMotionToSolids` reports
  `mass * (after - before)` (viscous drag included) with the pressure part kept separately; which part the exchange
  balance compares against the liquid's change is not stated by `plan-decisions.md` (solids plan to say).
- How `ApplyLimits` (step 3, rebuild with the same inputs), `BuildSimulationFromSnapshot` and `ReadSnapshot` (the
  committed active cells) hand or read the cell list in a GPU run; A17 allows only a list on the buffers'
  processor and an explicit `ReadGrid`.
- Stage 3's range part calls `solids::BuildTry` and the contact checks once per try; if the owner prefers, those
  calls move to stage 2 (plan-decisions open point 3).
- `kMemoryWarningShare = 1.10`, `kCopyWarningShare = 1.25` (I/O constants, owner to confirm); planned step copies
  are upper bounds, so the warning compares means.
- The material's own volumetric stiffness inside the solve (half of the volumetric response, Fang et al. 2020) is
  not built (owner question in `plan-decisions.md`).
- The `StepReport` field for `PressureOnMaterials::push` per set (the plan says "into the report beside the
  solids' pushes"; no field is named).
- The field name of `CapabilityFinding`'s `CapabilityLevel` (written `level` here).
- `BatchSettings::solve_piece_share = 0.25` and the batched direct-solve room `4 * max_cells` (new constants,
  owner to confirm).
- Wind on liquids: A20 brings Blender's Wind as `gas_wind`, read only by the gas; no plan gives liquids a wind.
