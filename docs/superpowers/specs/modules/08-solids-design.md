# Solids module design

Status: decisions applied 2026-11-14 (`owner-answers.md`, `changes-to-apply.md`); brought in line on 2026-11-14
with owner answers A15, A17, A18 and `plan-decisions.md` ("Solids", "Coupling inside the pressure solve (A18)",
"Batched solves (A15)", "Grid built on the GPU (A17)", "Planned cost shown to the user, and buffers in batched
runs") as the solids plan records them (Tasks 13, 19, 23, 28 amended); with owner answer A24 (blocking
materials beside a viscous liquid, decision 14); and with owner answers A26 and A27 (a viscous liquid sticks;
free solids in the viscous solve; decision 15, plan Tasks 23a to 23c). Module 7 of the build order
(after materials, before gas). Written for material sets only (owner A8); the blocking materials are snow and
springy solids (owner A9); what sand will need is listed under "For the sand module" in section 4.

## 1. Purpose

A solid is anything the simulated material cannot enter. The contact module knows what a solid looks like
during one step. This module knows how each solid moves from step to step, tells the grid where the solids
are, and makes sure that every push a solid gives the material is given back to the solid, with the same
numbers. It also keeps the two sides of a cutting plate apart.

**Owns:** the scene's solids with their motion kind (`fixed`, `driven`, `free`); driven paths and the rule that
a step never crosses a jump of a path; the wave paddle (a driven flat solid) and the stage `DriveWaves`; the
state and the dynamics of free solids under the try-and-commit rule; the per-step `contact::Solid` array; the
per-face and per-corner solid data that transfer and pressure declared (`transfer::SolidsAtFaces`,
`pressure::SolidCoverage`, `grid_edge_is_open_to_sky`, `pressure::FreeSolidPush`); the motion map and its
transpose; the blocking-material faces the liquid pressure solve holds (`pressure::BlockingMaterialFace`) and
the write-back of their solved velocity into each blocking set (A18); in a viscous scene, the material band and
the free-solid inputs of the viscous solve and the write-back of its solved velocities and motions (A26, A27);
the sums of contact reactions per solid;
the stage `CollideSolids`; plate sides on particles and
faces, the supported plate thickness and the cut work report; the checks of momentum, turning momentum, driven
work and no penetration; the solids part of a saved state.

**Does not own:** shapes, distance, the sweep and the friction response (contact); the weights and face sums
(grid, transfer); clipping of volumes and areas and the pressure solve (pressure); the response of snow and
springy solids to deformation (materials); the open-sea relaxation formula (pressure, called from
`DriveWaves`); which materials form a material set (I/O, carried by the step); the stage order, the step
duration and the retry (step); solid meshes (none: analytic shapes only, as on `main`).

## 2. What exists today

All paths are on `main`. Y-up: gravity is `-y`, the floating box is guided along world `y`, the liquid plane
is `world_surface_height_y`, the platen normal is `{0,-1,0}`, the piston moves along `x` (unchanged in Z-up).
A value enters Z-up through `(x, y, z) -> (x, -z, y)`. Inertia is stored in the solid's own axes and does not
change; the pose (a quaternion on `main`), angular velocity and torque are world quantities and turn with `R`
(a proper rotation, so they turn like ordinary vectors; report 03, table rows "Rigid pose and inertia" and
"World normal, force, torque").

| Where | What it does | What is wrong |
|---|---|---|
| `native/src/rigid_body_3d.cc:781-831` `AdvanceRigidBody3D`, `:833-859` `SampleRigidBodyPose3D` | Free-solid step: `v += dt F/m`; in own axes `w += dt I^-1 (T - w x I w)`; then a drift at the new constant velocities; quaternion renormalised. **Ported.** | Host only; force accumulators live in the state; `absl::Status`. |
| `rigid_body_3d.cc:135-292, 337-427, 510-710` | Clips box faces against a flat liquid plane and uniform cells into "pressure patches" (area, centroid, normal, one pressure cell). | One clipper per shape (aligned box, turned box, tank). Works only for a horizontal liquid plane and same-size cells. A second geometry beside the pressure solver's (report numerics 4.1, 10.6). |
| `rigid_body_3d.cc:712-779` `ComputeRigidBodyInwardWallFlux3D`, `AccumulateRigidBodyPressure3D` | Flow into a cell `= area * (v + w x arm) . n`; force `= -area (p - ambient) n` at the same centroid. The transpose idea, per patch. **The idea is ported.** | The load adds a vertical pressure-gradient torque (`:770-776`) that the flow does not have, so the two are not transposes of each other. |
| `native/src/liquid_substep_3d.cc:3663-3890` | Ten `AttemptWith...` methods: piston load, dynamic piston, coupled piston, guided box, translating box, coupled guided box, turned box (three), snow. | A public method per scene. Each has its own preconditions (CPU only, flat surface, zero friction, no source, no surface tension, no viscosity `:1762`). |
| `liquid_substep_3d.cc:1806-1993` | Coupled turned box: six extra pressure solves measure the response to each velocity part, the 6 x 6 result is made symmetric by averaging, then up to four corrections. `:1994-2075` and `:2076-2175`: the same with one number for the guided box and the piston. | A measured response instead of one solve; 7 to 11 pressure solves per step; tolerance tightened by hand for light solids (`:1806-1814`). |
| `native/src/wetted_piston_3d.cc`, `base/wetted_piston_math_3d.h`, `native/cuda/wetted_piston_3d*` | Liquid-covered area of the piston face from the union of particle balls, with rounding enclosures; the flow of a patch above the top liquid row is given to the row below. | A third geometry, for one side of one tank. Enclosure arithmetic the owner removed. The receiving-volume check still fails: 7.617 % and 7.625 % against 1 % (`docs/references/m1-geometry-diagnostics.md:19`). |
| `native/src/liquid_substep_3d_cpu.cc:316-376`, `liquid_substep_3d.cc:2700-2815` | Sums `mass *` the change of the velocity for the next step, and its moment per solid in particle order; adds it to the solid's velocity after the drift. **Ported.** | CPU only. Piston and box have separate sums. |
| `liquid_substep_3d.cc:1247-1275` | Stage "Waves": open-sea relaxation of face velocities after the before-forces snapshot. Stage "BodyContact" is entered and left empty. | The paddle is the tank's low-x side (`piston_velocity_x`); its path is two linear keys read by the add-on (`blender/fluidity/scene_inputs.py:62-90`). |
| `native/src/plastic_solid_boundary_3d.cc:36-271`, `base/sampled_solid_liquid_geometry_3d.h:243-352` | Snow seen by the liquid: distance to the union of particle balls at cell corners, the nearest particle is the corner's owner; flow through a blocked face `= sum over its 4 corners of corner area * owner velocity`; the pressure force goes back to the owners by the same corner areas (`:319`), with a fitted-change part. **Ported, and made the one map for every solid.** | Uniform cells; one snow object; scans all particles per corner (report CUDA 10.1). |
| `plastic_solid_boundary_3d.h:89-135` `PlasticSolidPressureMobility3D` | Adds the snow's "give" (`B M^-1 B^T`) to the liquid pressure solve, then checks the flow. | A second solver loop around the pressure solve. The loop is not ported; the give is ported lumped to the set's face mass inside the one liquid solve (decision 3, replaced by A18; 5.5a). |
| `base/distinct_body_contact_3d.h:47-120`, `native/src/contact_face_bodies_3d.cc:8-135` | Contact of two materials at one sample: no bounce, Coulomb friction, equal and opposite push; the push is shared out to particles by their weights. **Ported.** | Used by tests only. Detection and normals exist only for flat axis-aligned fronts (`native/src/mixed_contact_3d.cc:138-400`). Up to four materials per face in a sparse table. |
| `native/src/mixed_contact_3d.cc:400-1335` | One-cell flat snow front pushed by water: its own freeze, solve, advection and frame driver. | A scene-specific solver. Not ported. |
| `native/src/liquid_wall_contact_3d.cc:592-683`, `native/src/elastic_blade_deformation_3d.cc:265-406` | Cutting: each particle is given the side of the plate it starts on; each face gets the side of its centre at the start and at the end pose; a face that changed side or entered the plate gives no after-step value. | The plate is unbounded, so the whole plane is a cut from the first step. One plate. Sides are a parallel list keyed by particle id (`host/solid_interaction_checkpoint_3d.h:21-45`). Fixed grid, CPU only. |
| `elastic_blade_deformation_3d.cc:176-233` | Cut work: `dt * rest volume * internal force per area : (exact change)` plus the work of the same-side face forces. Plate work `= push . plate velocity` (`liquid_wall_contact_3d.cc:574-587`). **Ported as a report.** | Mixed into the particle update. |

Report findings fixed here: the reaction must be the transpose of the motion, with the lever arm and the same
sample points (numerics 10.2); pressure push and contact push are counted once each (10.3); one pose per
stage, a step split at a jump of a driven path, driven work accounted (10.4); checks of momentum, turning
momentum, driven work and no penetration, with analytic shapes kept as test oracles (10.6); cut history and
side identity survive restart and split or merge, with a stated plate thickness (9.6); a bounded search with an
exhaustive test oracle, and a time stamp on every geometry array (CUDA 10.2, 10.5); waves are measured as a
forced system (13.7).

## 3. Public interface

Folder `native/fluidity/solids/`, namespace `fluidity::solids`. Headers marked (shared) compile for CPU and
GPU; an operation header has private `_cpu.cc` and `_gpu.cu` beside it. D = DECIDED word, N = NEW. Every name
is N unless marked. Every fallible host function returns `std::expected<T, SolidsFailure>` (owner A2); shared
formulas return `SolidsError`. Every per-solid or scene total is one partial per cell block in the grid's fixed
block order, then `math::TreeSum` over the block partials, so a result does not depend on where batches are cut;
contact reactions are the one exception, ordered by `particle_id` over the whole try (5.6). **Sign of work**
(plan-decisions "Solids"): every work here is the energy a solid gives the material, positive when a paddle
pushes liquid; `PressureOnSolids::flow_work`, `ContactOnSolids::work` and `ExchangeTotals::driven_work` use it.

### 3.1 `solids_error.h` (shared), `solids_error_message.h` (CPU only)

```cpp
enum class SolidsError : std::uint8_t {
  kNone,
  kBadInput,               // A value is not finite or out of range; ids not ascending; inertia not positive;
                           //   a particle_radius above the limit of 5.4 step 3 (limit = m).
  kWrongGridVersion,       // An array was built for another grid version.
  kWrongTime,              // A solid array was computed for another time or another try than the caller's.
  kPathJumpInsideStep,     // A driven path changes velocity inside the step. limit = the longest allowed step, s.
  kSolidMovesTooFar,       // A solid surface travels more than max_travel_in_cells. limit = allowed step, s.
  kTooManyPlates,          // More than kMaxPlateCount plates in the scene (5.9).
  kPlateTooThinForLiquid,  // A plate in a scene with liquid or gas is thinner than 5.9 allows. limit = m.
  kParticleInsidePlate,    // A particle without a side is inside the thickness of the plate.
  kSolidsTooClose,         // A free solid is within one cell width of another solid. Not supported (5.3).
  kTooManyFreeSolids,      // More than kMaxFreeSolidCount free solids.
  kMaterialNotCarried,     // A named material lacks an array this module needs (plate_side, fitted change).
  kWrongMaterialSet,       // NEW. A set's GridFromParticles and CellsFromParticles carry different
                           //   transfer::MaterialSetKey values, or two entries name the same set.
  kExchangeNotBalanced,    // CheckSolidExchange: a balance is off by more than its tolerance.
  kNotFinite,              // A computed value is not finite.
  kBuffersTooSmall,        // More motion-map entries or reactions than the buffers were built for.
  kRunOnCallFailed,        // CheckCanRunOn failed for the chosen processor, or a call to it failed.
  kOutOfMemory,            // NEW (reconcile item 25). BuildSolidsBuffers or BuildSolidCoverageValues could not take
                           //   a piece of its RunMemory (BufferError::kRunMemoryTooSmall, A58); magnitude = bytes
                           //   asked. Returned only by the building functions. The last value (reconcile item 31,
                           //   plan-decisions "Owner answers A40-A64 applied" item 11): 18 values.
};
struct SolidsFailure {           // The error value of every std::expected here. Never allocates.
  SolidsError error = SolidsError::kNone;
  std::uint32_t solid_id = 0;    // 0 when no solid is involved.
  std::uint64_t particle_id = 0; // 0 when no particle is involved. On the GPU: the lowest failing id.
  grid::FacePlaceIndex face{grid::kNoIndex};
  double magnitude = 0;          // What was measured, in the unit named beside the error value.
  double limit = 0;              // What was allowed, same unit.
};
// solids_error_message.h. ToMessage: one English sentence. ToName (N): the stable code, the enumerator's name
// without the k in lower snake case (kPlateTooThinForLiquid -> "plate_too_thin_for_liquid"); the CLI and the
// add-on use it.
std::string_view ToMessage(SolidsError error);
std::string_view ToName(SolidsError error);
```

### 3.2 `solid_settings.h` (shared types)

```cpp
enum class SolidMotionKind : std::uint8_t { kFixed, kDriven, kFree };   // D words: fixed, driven, free.
struct PathKey {
  double time = 0;                       // s. Strictly ascending along a path.
  Vec3 origin{};                         // m, world. The origin moves in a straight line to the next key.
  Vec3 angular_velocity_until_next{};    // rad/s, world, about the origin. Constant until the next key.
};
enum class AfterPath : std::uint8_t { kStaysAtLastKey, kLeavesScene };
struct DrivenPath {                      // Before the first key the solid rests at the first key's pose.
  std::span<const PathKey> keys;         // At least one. Every key time is a jump of the path.
  AfterPath after_path = AfterPath::kStaysAtLastKey;
};
// NEW (A59). How a guide is given. kAxisLimits: the four flags below (world axes; turning all or nothing about the
// origin). kRail: slides along rail_direction, never turns. kHinge: turns about hinge_axis through hinge_point only.
enum class GuideKind : std::uint8_t { kAxisLimits, kRail, kHinge };
struct FreeMotionLimits {                // A guide. Default: all six motions are free.
  bool moves_along_x = true, moves_along_y = true, moves_along_z = true;   // World axes. Read for kAxisLimits only.
  bool turns = true;                     // Read for kAxisLimits only.
  GuideKind guide_kind = GuideKind::kAxisLimits;   // NEW (A59).
  Vec3 rail_direction{0, 0, 1};          // NEW (A59). No unit, world, unit length (normalised at load by I/O;
                                         //   CheckSolidsCanRun: kBadInput when |length - 1| > 1e-12 or not finite).
  Vec3 hinge_point{};                    // NEW (A59). m, world, at time 0; the pivot is fixed in the world.
  Vec3 hinge_axis{0, 0, 1};              // NEW (A59). No unit, world, unit length (as rail_direction). World frame,
                                         //   decided: a rotation about the axis leaves the axis unchanged, so world and
                                         //   own axes agree for it; body-fixed axes and moving rails are not exposed.
};
struct FreeSolidSettings {
  double mass = 0;                       // kg, positive.
  Vec3 inertia_in_own_axes{};            // kg m2, positive. The solid's own axes must be its principal axes.
  FreeMotionLimits limits{};
  Vec3 start_linear_velocity{};          // m/s, world.
  Vec3 start_angular_velocity{};         // rad/s, world.
  Vec3 steady_force{};                   // N, world, besides gravity. Applied at the origin.
};
struct SolidSettings {
  std::uint32_t solid_id = 0;            // Stable, nonzero, unique. Also the plate_id of a plate.
  contact::SolidShape shape = contact::SolidShape::kFlatSolid;
  contact::SolidPose rest_pose{};        // Pose at time 0 (fixed, free) or the axes at the first key (driven).
  Vec3 half_size{};                      // m. As contact::Solid::half_size.
  contact::ClosedSides closed{};         // kDomainBox only.
  double friction_scale = 1;             // No unit. As contact::Solid::friction_scale (Q35 a).
  SolidMotionKind motion_kind = SolidMotionKind::kFixed;
  DrivenPath path{};                     // kDriven only.
  FreeSolidSettings free_solid{};        // kFree only. The origin is the centre of mass.
  // NEW (A53). m. How far beyond the surface a viscous liquid's velocity would reach the solid's (Navier slip
  // length): 0 full stick (default), +infinity free slip. Read for fixed and driven solids and the domain box beside a
  // viscous liquid (viscosity's rule (a) uses d + slip_length); free solids always stick (A27) and ignore it (I/O's
  // check gives the NOTE slip_ignored_on_free_solid for a nonzero value); an inviscid liquid keeps the contact rule.
  // CheckSolidsCanRun: kBadInput when negative or NaN.
  double slip_length = 0;
};
struct MaterialTheLiquidCannotEnter {    // Snow or a springy solid that liquid must flow around.
  std::uint32_t material_index = 0;      // Of a kSnow or kSpringySolid material. Its own material set (A8).
  double particle_radius = 0;            // m, positive. A particle blocks liquid as a ball of this radius.
                                         // One radius per material is right (A65, A66): blocking materials keep
                                         // their count (kKeepsItsCount), so their cells stay at smallest_size_level
                                         // and every particle has rest volume (w_min/2)^3; I/O gives w_min / 4.
};
struct SolidsSettings {                  // Hashed with the scene: no run control lives here.
  std::span<const SolidSettings> solids;               // Ascending solid_id.
  std::span<const MaterialTheLiquidCannotEnter> blocking_materials;   // Ascending material_index.
  Vec3 gravity{0, 0, -9.81};             // m/s2, world. Z is up.
  double max_travel_in_cells = 0.5;      // Largest travel of a solid surface per step, in smallest cell widths.
  std::uint32_t max_motion_map_entries = 1u << 20;     // Limit. Blocked corners of pieces that touch liquid.
  bool has_liquid_or_gas = true;         // NEW. The scene has a liquid or gas material: the plate thickness
                                         //   rule of 5.9 applies.
  RunOn run_on = RunOn::kCpu;
};
inline constexpr double kSolidReachInCells = 2.0;   // A face "has a solid" within this many cell widths.
inline constexpr double kCoverageCapInCells = 2.0;  // distance_to_solid is capped at plus this many widths.
inline constexpr double kFaceStopReachInCells = 0.5;  // CollideSolids stops faces this close to a solid.
// NEW. Public scene limits; the add-on shows them, CheckSolidsCanRun enforces them.
inline constexpr std::uint32_t kMaxPlateCount = 1;        // Plates per scene (5.9).
inline constexpr std::uint32_t kMaxFreeSolidCount = 16;   // Free solids per scene. Was a setting.
```

`DrivenFlatSolid`, `DrivenPlate`, `FreeBox` (naming review 4.2) are spelled as a kind plus a shape, not as
types (decision 6). The `wave_paddle` is a variable: a `SolidSettings` with `kFlatSolid` and `kDriven`.

### 3.3 `solid_state.h` (shared) and `solid_dynamics.h` (shared formulas)

```cpp
struct Orientation {                     // Unit rotation. Papers: unit quaternion, own axes to world.
  double w = 1;
  Vec3 vector{};
};
Mat3 ToAxes(Orientation orientation);                        // Columns are the solid's axes in world.
Orientation ToOrientation(const Mat3& axes);                 // axes must be a rotation.
Orientation ComputeTurnedOrientation(Orientation start, Vec3 rotation_vector);   // Renormalised.
// NEW (I/O). rad, world. The shortest turn from `from` to `to`: ComputeTurnedOrientation(from, result) equals
// `to` within round-off; |result| <= pi; zero when both give the same rotation (q and -q included). I/O turns
// the rotation keys of a driven path into PathKey::angular_velocity_until_next = result / (key time step).
Vec3 ComputeRotationVector(Orientation from, Orientation to);

struct SolidState {                      // One per solid. What a commit keeps.
  Vec3 origin{};                         // m, world.
  Orientation orientation{};
  Vec3 linear_velocity{};                // m/s, world. Free: includes every push up to this time.
  Vec3 angular_velocity{};               // rad/s, world.
  bool is_in_scene = true;               // false after a driven path with kLeavesScene has ended.
};
// World frame, at one pose, about the origin. CHANGED (A59; computed with pressure::ComputeGuidedInverseMass for
// kRail and kHinge, so pressure and solids share one formula): the four blocks of P = J (J^T M J)^-1 J^T, the inverse
// inertia seen through the guide's allowed motions (J from ComputeAllowedMotions, M = [mass I3, 0; 0, R I R^T]):
// (linear velocity, turn rate) change = P (push, turning_push). Unguided: per_mass = I3 / mass, per_turning_inertia =
// R diag(1/I) R^T, couplings 0. kAxisLimits: the same with zero rows and columns for forbidden motions (computed by
// selection, so the bytes are those of the unguided formula). kRail: per_mass = d d^T / mass, the rest 0. kHinge: all
// four blocks from the one column [a x (o - p); a] / (mass |a x (o - p)|^2 + a^T R I R^T a).
struct InverseInertia {
  Mat3 per_mass{};                       // 1/kg. Linear velocity per push.
  Mat3 per_turning_inertia{};            // 1/(kg m2). Turn rate per turning push.
  Mat3 linear_per_turning_push{};        // NEW (A59). 1/(kg m). Linear velocity per turning push.
  Mat3 turning_per_push{};               // NEW (A59). 1/(kg m). Turn rate per push; the transpose of the block above.
};
// CHANGED (A59): `origin` (m, world) places a hinge's column. Requires a valid guide (CheckSolidsCanRun).
InverseInertia ComputeInverseInertia(const FreeSolidSettings& settings, Orientation orientation, Vec3 origin);
// Velocity after gravity, the steady force and the turning term (5.3 step 1). No pressure, no contact. CHANGED
// (A59): then projected onto the guide's allowed motions with ComputeAllowedVelocity (state.origin places a hinge).
SolidsError ComputeVelocityBeforePressure(const FreeSolidSettings& settings, Vec3 gravity, const SolidState& state,
                                          double step_duration, contact::SolidMotion* motion);
// velocity += inverse inertia * push, all four blocks (A59). push in N s, turning_push in N m s about the origin.
contact::SolidMotion ComputeMotionAfterPush(const contact::SolidMotion& motion, const InverseInertia& inverse,
                                            Vec3 push, Vec3 turning_push);
// Pose and constant velocity of a driven solid for the step [time, time + step_duration] (5.2).
SolidsError ComputeDrivenMotion(const SolidSettings& settings, double time, double step_duration,
                                SolidState* state_at_time, contact::SolidMotion* motion);
```

`guide_math.h` (shared, `FLUIDITY_CPU_AND_GPU`), NEW (A59; `plan-decisions.md` "Owner answers A40-A64 applied" item 5):

```cpp
// A guide's allowed motions in general form: motion xi = J q + xi_0, xi = (V, Omega) in world axes about
// origin_at_start, J the first column_count columns (each six numbers: V then Omega), q the generalized velocities
// (Batty 2007, Aanjaneya 2018). prescribed_motion is xi_0; always 0 today (no setting exposes a moving rail).
struct AllowedMotions {
  double columns[6][6];                  // columns[k] is column k. Only the first column_count are read.
  std::uint8_t column_count = 6;
  double prescribed_motion[6] = {};
};
// Built each try. kAxisLimits: one selection column (exactly 0 or 1) per allowed motion, in the order V.x, V.y, V.z,
// then Omega x, y, z when turns (6 columns, the identity, when unguided). kRail: J = [d; 0]. kHinge: J = [a x (o -
// p); a], o = origin_at_start, p = hinge_point, a = hinge_axis (one column). Columns are independent, so J^T M J is
// positive definite. No FMA contraction.
FLUIDITY_CPU_AND_GPU AllowedMotions ComputeAllowedMotions(const FreeMotionLimits& limits, Vec3 origin_at_start);
// NEW (A59, plan decision). The motion kept by the guide: the M-orthogonal projection J (J^T M J)^-1 J^T M xi + xi_0
// (M = [mass I3, 0; 0, inertia_world]), computed by pressure::ComputeGuidedSolidMotion with a zero impulse (the
// declaration sits here; the definition in a solids header that includes pressure's, since guide_math.h stays
// header-only and pressure includes it); for kAxisLimits the selection itself (forbidden components set to 0, the
// others kept bitwise), so axis guides keep the bytes of 5.3 step 1's "a motion the guide forbids is set to zero".
FLUIDITY_CPU_AND_GPU contact::SolidMotion ComputeAllowedVelocity(const FreeMotionLimits& limits, Vec3 origin,
                                                                double mass, const Mat3& inertia_world,
                                                                const contact::SolidMotion& motion);
// NEW (A59, plan decision). The pose after a step of constant motion (5.3 step 4): kAxisLimits and unguided origin +=
// dt V, orientation turned by dt Omega; kRail origin += dt (V . d) d, orientation unchanged; kHinge the pose rotated
// about hinge_point by angle dt (Omega . a) about hinge_axis (origin = p + Rot(o - p), orientation turned by the same
// rotation), so the pivot distance |o - p| is kept to rounding (the hinge's position correction).
FLUIDITY_CPU_AND_GPU void ComputePoseAfterStep(const FreeMotionLimits& limits, const contact::SolidMotion& motion,
                                               double step_duration, Vec3* origin, Orientation* orientation);
```

### 3.4 `solids.h` and `solids_try.h` (CPU only)

```cpp
class Solids {                           // Committed state of every solid. CPU memory (6). Move-only. Owns copies
 public:                                 //   of its settings, path keys and blocking materials (plan-decisions).
  double time() const;                   // s.
  std::uint64_t state_version() const;   // D. Grows by one at each CommitTry.
  std::span<const SolidSettings> settings() const;
  std::span<const SolidState> states() const;        // Same order as settings.
};
std::expected<Solids, SolidsFailure> BuildSolids(const SolidsSettings& settings);

struct SolidsSummary {                   // For the capability checks of step and viscosity.
  bool has_driven_solid = false, has_free_solid = false, has_plate = false, has_blocking_material = false;
  bool has_inclined_solid = false;       // Some solid's axes are not the world axes, or it turns.
  std::uint32_t plate_id = 0;            // The scene's one plate, or 0.
};
SolidsSummary ComputeSolidsSummary(const Solids& solids);
// Every scene rule that needs no particle, each broken rule reported once (the add-on shows all problems at
// once): ids nonzero and ascending; at most kMaxPlateCount plates (kTooManyPlates); with has_liquid_or_gas, each
// plate at least ComputeMinPlateThickness(normal, largest_cell_width) thick (kPlateTooThinForLiquid, limit m);
// at most kMaxFreeSolidCount free solids (kTooManyFreeSolids); positive mass and inertia, ascending path keys
// (kBadInput); each particle_radius at most (4 - kCoverageCapInCells) * smallest_cell_width (kBadInput, 5.4
// step 3). Empty: the scene can run. Entries are ordered by solid_id (0 for a rule about the whole scene or a
// blocking material), then by error value; magnitude and limit are filled where a number was compared. Contact's
// CheckSolid (thickness against the contact tolerance) is called by the step beside this.
std::vector<SolidsFailure> CheckSolidsCanRun(const SolidsSettings& settings, double smallest_cell_width,
                                             double largest_cell_width);
// Longest step from `time` that crosses no path key and moves no surface farther than max_travel_in_cells.
double ComputeStepDurationLimit(const Solids& solids, double time, double smallest_cell_width);   // s

class SolidsTry {                        // One try of one step. Dropping it changes nothing.
 public:
  double step_start_time() const;        // s.
  double step_duration() const;          // s.
  std::uint64_t try_id() const;          // Stamped on every array computed from this try.
                                         //   (state_version << 32) | tries_since_commit (plan-decisions).
  // Ascending solid_id, solids in the scene only. pose_at_step_start is the committed pose. A free solid's
  // motion is its velocity before pressure until ApplyPressureToSolids, and its velocity after pressure then.
  std::span<const contact::Solid> solids() const;
  std::span<const InverseInertia> inverse_inertia() const;   // Free solids only, in solids() order.
  // NEW (A27, plan decision, solids plan Task 23c). Copies inside the try. Filled by ApplySolvedMotionToSolids
  // (3.6); empty after BuildTry and after ApplyPressureToSolids.
  std::span<const SolvedMotionReport> solved_motion() const;
  // NEW (A27, plan decision). The pressure part of the last ApplyPressureToSolids or ApplySolvedMotionToSolids;
  // the step passes it as ExchangeInputs::pressure_on_solids after stage 11.
  PressureOnSolids pressure_on_solids() const;
  // NEW (A60). One per solid of solids() (ascending solid_id, fixed, driven, free and the domain box): the force of
  // the liquid from the last ApplyPressureToSolids (viscous parts 0) or ApplySolvedMotionToSolids. Empty after
  // BuildTry. Copies inside the try. The step's StepReport::liquid_force_on_solids is filled from the committed try.
  std::span<const LiquidForceOnSolid> liquid_forces() const;
  // NEW (A53, plan decision). m, one per solid of solids(): SolidSettings::slip_length, 0 for free solids (they always
  // stick). What viscosity::ViscosityInputs::solid_slip_length takes.
  std::span<const double> slip_length() const;
};
// liquid_force.h (CPU only, the type shared), NEW (A60; plan-decisions "Owner answers A40-A64 applied" item 6).
struct LiquidForceOnSolid {              // Mean over the step: impulse / step duration. World axes.
  std::uint32_t solid_id = 0;
  Vec3 pressure_force{};                 // N. From the solve's pressure (PressureOnSolids push / dt).
  Vec3 pressure_torque{};                // N m about the solid's origin at the step start (T_p = T_o + (o - p) x F
                                         //   moves it to a point p).
  Vec3 viscous_force{};                  // N. From the viscous solve's terms (ViscousPushOnSolid push / dt); 0 for an
                                         //   inviscid liquid. A guided solid reports all six numbers: the guide's
  Vec3 viscous_torque{};                 //   reaction is in them. N m, same point.
};
struct SolvedMotionReport {              // NEW (A27, plan decision). One per free solid, FreeSolidPush order.
  Vec3 push{};                           // N s: mass * (velocity after - velocity before), the whole change.
  Vec3 turning_push{};                   // N m s about origin_at_start: inertia_at_start * (turn rate change).
  Vec3 pressure_push{};                  // N s: the pressure part; equals ApplyPressureToSolids's push.
  Vec3 pressure_turning_push{};          // N m s.
};
// Fails: kPathJumpInsideStep, kSolidMovesTooFar (both give `limit`), kSolidsTooClose, kNotFinite.
std::expected<SolidsTry, SolidsFailure> BuildTry(const Solids& solids, double step_duration,
                                                 double smallest_cell_width);
// kWrongTime for a try built before another commit (plan-decisions). Reads every per-solid result from the try
// itself: ApplyPressureToSolids and ApplyContactReactions copy their per-solid results into SolidsTry (CPU) before
// they return, so stage 12's results may come from one SolidsBuffers instance and stage 17's from another (a
// batched run, section 6). SolidsTry never keeps a pointer into a SolidsBuffers across calls (A17, batched-run
// buffers).
std::expected<void, SolidsFailure> CommitTry(Solids& solids, SolidsTry&& solids_try);
void RejectTry(Solids& solids, SolidsTry&& solids_try);

struct SolidsSnapshot {                  // D word. What a saved state stores for this module.
  double time = 0;
  std::uint64_t state_version = 0;
  std::vector<std::uint32_t> solid_id;   // Ascending.
  std::vector<SolidState> state;
  std::vector<ExchangeTotals> totals;    // Running sums since time 0 (3.9).
};
SolidsSnapshot ReadSnapshot(const Solids& solids);
std::expected<Solids, SolidsFailure> BuildSolidsFromSnapshot(const SolidsSettings& settings,
                                                             const SolidsSnapshot& snapshot);
// NEW (render output). The solids of a snapshot as contact::Solid values, ascending solid_id, solids in the
// scene only: pose_at_step_start is the snapshot's pose, motion its velocities, the rest as BuildTry sets it.
// Precondition: snapshot.solid_id lists the ids of `settings` (as ReadSnapshot gives); otherwise empty.
std::vector<contact::Solid> BuildContactSolids(std::span<const SolidSettings> settings,
                                               const SolidsSnapshot& snapshot);
// The memory estimate entries BuildSolidsBuffers allocates from, plus the step-owned arrays of section 6 (base/memory_count.h).
std::vector<MemoryEstimateEntry> BuildSolidsMemoryEstimate(const SolidsSettings& settings);
```

### 3.5 `solids_on_grid.h` (+ `_cpu.cc`, `_gpu.cu`)

```cpp
struct CornerOwner {                     // Who is nearest to one corner place. 16 bytes.
  std::uint32_t solid_index = contact::kNoSolid;   // Into SolidsTry::solids(), or kNoSolid.
  std::uint32_t material_index = 0xFFFFFFFFu;      // A blocking material, when no analytic solid is nearer.
  std::uint64_t particle_id = 0;                   // Its nearest particle. 0 for an analytic solid.
};
struct SolidCoverageValues {             // Owns the arrays pressure::SolidCoverage points at. One try. Every
                                         //   place of every active cell, with or without liquid (gas reads it).
  grid::CornerValues<double> distance_to_solid;      // m, positive outside every solid and blocking material.
  grid::CornerValues<CornerOwner> corner_owner;
  grid::CornerValues<Vec3> corner_velocity;          // m/s, world, of the owner at the corner. Zero: no owner.
  grid::FaceValues<double> solid_normal_velocity;    // m/s along +axis, mean over the blocked part.
  double time = 0;                                   // s. The step start.
  std::uint64_t try_id = 0;
  grid::GridVersion grid_version{};
};
pressure::SolidCoverage ToSolidCoverage(const SolidCoverageValues& values);
// NEW (plan-decisions "Solids"). The step builds its coverage arrays with it.
// CHANGED (A58): every array is a piece of `run_memory`; Requires: run_on == run_memory.run_on() (debug assertion).
std::expected<SolidCoverageValues, SolidsFailure> BuildSolidCoverageValues(RunOn run_on,
                                                                           std::uint32_t max_cell_blocks,
                                                                           RunMemory& run_memory);

struct SolidsOnGridInputs {
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;           // Places stored in these blocks are written; no border place is read.
  const SolidsTry& solids_try;
};
// outside_solid, side and solid_velocity of every face place, `elapsed` seconds into the step (0 for the
// start pose, step_duration for the end pose). Sets time and grid_version. The end pose of a free solid is
// known only after ApplyPressureToSolids. solids_at_faces must be bound to inputs.grid (kWrongGridVersion).
std::expected<void, SolidsFailure> ComputeSolidsAtFaces(const SolidsOnGridInputs& inputs, double elapsed,
                                                        transfer::SolidsAtFaces& solids_at_faces);
// NEW (cell resize). The same at the committed pose, with the committed velocities as solid velocities and
// time = solids.time(): for a resized grid after the commit, when no try exists.
std::expected<void, SolidsFailure> ComputeSolidsAtFaces(const grid::StaggeredGrid& grid,
                                                        grid::CellBlockRange blocks, const Solids& solids,
                                                        transfer::SolidsAtFaces& solids_at_faces);
// Distances, owners and velocities at the step start, at every corner and face place stored in inputs.blocks,
// whether or not a cell holds liquid. particles_try may be null when no material blocks liquid; otherwise the
// particles of the blocking materials in the blocks and their border blocks are read.
std::expected<void, SolidsFailure> ComputeSolidCoverage(const SolidsOnGridInputs& inputs,
                                                        const particles::ParticlesTry* particles_try,
                                                        SolidCoverageValues& coverage);
// 1 on a grid-edge face piece whose far side is reached from the open top through space with no cell block
// and no solid (5.4 step 5). Writes one byte per face place with grid::WriteValues, on the grid's processor.
// A17: on a GPU grid no block id comes down; one GPU pass marks the occupied boxes (1 B per box, read back),
// the CPU floods, one GPU pass writes the face marks from the reached boxes (1 B per box, sent). Nothing is
// marked when the domain's top is closed. kRunOnCallFailed on a GPU grid without a GPU. Floods over the whole
// grid, so the step calls it in the run_whole_grid part of stage FindSolidSurfaces (in a batched run on the run's
// processor right after pass 1; in one piece right after that stage's run), not per batch (A17, batched-run
// buffers).
std::expected<void, SolidsFailure> FindGridEdgesOpenToSky(const grid::StaggeredGrid& grid,
                                                          const contact::DomainBox& domain_box,
                                                          const SolidsTry& solids_try,
                                                          grid::FaceValues<std::uint8_t>& grid_edge_is_open_to_sky);
```

### 3.6 `motion_map.h` (+ `_cpu.cc`, `_gpu.cu`) and `motion_map_math.h` (shared)

```cpp
struct MotionMapEntry {                  // One blocked corner of one face piece, seen from one liquid cell.
  grid::CellPlaceIndex cell{grid::kNoIndex};        // The liquid cell whose pressure pushes here.
  grid::FacePlaceIndex face{grid::kNoIndex};
  grid::CornerPlaceIndex corner{grid::kNoIndex};
  double signed_corner_area = 0;         // m2. Positive when the cell is the piece's low cell (5.5).
};
class MotionMap;                         // Entries sorted by (cell, face, corner), plus the lists of 5.5. Move-only
                                         //   view into SolidsBuffers (try_id, time, grid_version, build number);
                                         //   a map from an earlier build on the same buffers gives kWrongTime.
struct MotionMapInputs {                 // Built once per try on the whole grid, also in a batched run.
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;                       // Every block of the grid, else kBadInput (equation order).
  const pressure::GeometrySnapshot& geometry;        // Same try. Which cells are kLiquid; open_area per piece.
  const SolidCoverageValues& coverage;               // Same time and try, else kWrongTime.
  const SolidsTry& solids_try;
};
std::expected<MotionMap, SolidsFailure> BuildMotionMap(const MotionMapInputs& inputs, SolidsBuffers& buffers);
// What pressure::EquationInputs takes. Sorted by solid_index, then equation_index. Free solids only.
// FreeSolidPush::solid_index is the index of the free solid among the free solids of SolidsTry::solids(), in
// that order (plan-decisions "Solids").
std::span<const pressure::FreeSolidPush> ToFreeSolidPushes(const MotionMap& motion_map);
// CHANGED (A59; replaces ToFreeSolidInverseInertia, two Mat3 per free solid): what pressure::EquationInputs takes
// for the free solids, one entry per free solid in FreeSolidPush::solid_index order, filled by BuildMotionMap:
// pressure::FreeSolidInertia { double mass; Mat3 inertia_at_start; solids::AllowedMotions allowed_motions; } with
// allowed_motions = ComputeAllowedMotions(limits, origin_at_start) (declared by pressure; pressure's term becomes
// (K J)(J^T M J)^-1 (K J)^T, rank m, and pressure::ComputeGuidedSolidMotion holds the one factor).
std::span<const pressure::FreeSolidInertia> ToFreeSolidInertia(const MotionMap& motion_map);

struct PressureOnSolids {                // Per solid of SolidsTry::solids(), from one accepted pressure.
  std::span<const Vec3> push;            // N s. Driven and fixed solids: reported, their motion is unchanged.
  std::span<const Vec3> turning_push;    // N m s about the solid's origin at the step start.
  double flow_work = 0;                  // J given to the material: dt * sum over cells of pressure * (solid
                                         //   volume rate into the cell); positive when a paddle pushes liquid.
  double push_to_materials = 0;          // N s. A report only (A18): sum of magnitudes of the map entries whose
                                         //   corner owner is a blocking material (pressure of a non-liquid cell 0).
};
// After the solve. `pressure` is pressure::SolvedArrays::pressure of SolvePressure (stage 12, the step's try
// array). In a viscous scene stage 11 calls ApplySolvedMotionToSolids instead (A27, below); a solids test may still
// call this one with the cell pressure viscosity wrote (its pushes are the pressure part ApplySolvedMotionToSolids
// reports). Sets each free solid's motion to its velocity after pressure (5.5 step 5). Stores no push owed to
// particles (A18: blocking materials are moved by ApplyPressureToMaterials, 3.6a). Per-solid sums: block
// partials, then a tree sum. Fails kSolidMovesTooFar when a free solid's velocity after pressure breaks the
// travel or turn limit (`limit` = allowed step, s); the step shortens the step and retries, as for BuildTry.
// NEW (A60): also fills SolidsTry::liquid_forces() (pressure part / dt; viscous parts 0). CHANGED (A59): a free
// solid's motion after pressure is pressure::ComputeGuidedSolidMotion(solid, xi_before, impulse, xi_after) with its
// FreeSolidInertia (q = (J^T M J)^-1 J^T (M xi_before + impulse), xi = J q + xi_0), so it stays in the allowed motions
// and shares pressure's factor.
std::expected<PressureOnSolids, SolidsFailure> ApplyPressureToSolids(
    const MotionMap& motion_map, const grid::CellValues<double>& pressure, SolidsTry& solids_try,
    SolidsBuffers& buffers);

// NEW (A27; plan-decisions "A viscous liquid sticks, and free solids in the viscous solve"). The free solids'
// inputs to the viscous solve. Plan decision (solids plan Task 23c): one entry per free solid of the try, in
// FreeSolidPush::solid_index order, also one that touches no liquid this step (the solve gives its motion back
// unchanged). Filled by BuildMotionMap, on the map's processor.
struct FreeSolidForViscousSolve {
  std::uint32_t solid_index = 0;          // As FreeSolidPush::solid_index.
  double mass = 0;                        // kg
  Mat3 inertia_at_start;                  // kg m2, world axes, about origin_at_start
  Vec3 origin_at_start{};                 // m
  Vec3 velocity_before_pressure{};        // m/s
  Vec3 turn_rate_before_pressure{};       // rad/s
  // CHANGED (A59; replaces `FreeMotionLimits limits`, "forbidden motions are held at their value before pressure"):
  // ComputeAllowedMotions(settings.limits, origin_at_start). The solve's unknowns are its q; sizeof 504 B.
  AllowedMotions allowed_motions{};
};
std::span<const FreeSolidForViscousSolve> ToFreeSolidsForViscousSolve(const MotionMap&);
struct FreeSolidMotion { Vec3 velocity{}; Vec3 turn_rate{}; };   // m/s, rad/s
// NEW (A60). The viscous push on one solid, as viscosity::ViscosityResult::viscous_push_on_solids gives it (one per
// SolidsTry::solids() entry): N s, and N m s about the solid's origin at the step start, world axes.
struct ViscousPushOnSolid { Vec3 push{}; Vec3 turning_push{}; };
// Stage 11, in place of ApplyPressureToSolids. Like ApplyPressureToSolids, with the motion solved by the viscous
// solve: (1) the pressure part from the map and `pressure`, computed and kept in the try exactly as
// ApplyPressureToSolids does (ExchangeTotals and CheckSolidExchange unchanged); (2) each free solid's motion becomes
// motion_after_solve[k] as given (CHANGED, A59: the solve keeps it in the allowed motions, so nothing is dropped;
// replaces "except that a motion its guide forbids keeps its value before pressure"); (3) travel and turn limits
// checked again (kSolidMovesTooFar, `limit` = allowed step); (4) push report mass * (after - before) and
// inertia_at_start * (turn rate change) in SolidsTry::solved_motion(), pressure part kept separately; (5) NEW (A60)
// SolidsTry::liquid_forces(): per solid the pressure part / dt, viscous parts 0 until ApplyViscousImpulses.
// motion_after_solve has one entry per FreeSolidForViscousSolve (kBadInput otherwise); a map of an earlier build
// kWrongTime; pressure of another grid kWrongGridVersion; kNotFinite. On failure the try is unchanged.
std::expected<void, SolidsFailure> ApplySolvedMotionToSolids(const MotionMap&,
    std::span<const FreeSolidMotion> motion_after_solve, const grid::CellValues<double>& pressure,
    SolidsTry&, SolidsBuffers&);
// NEW (A60). Stage 11, right after ApplySolvedMotionToSolids: stores viscosity::ViscosityResult::
// viscous_push_on_solids (one per SolidsTry::solids() entry) in the try, so liquid_forces() holds viscous_force =
// push / dt and viscous_torque = turning_push / dt. Another size: kBadInput; a value not finite: kNotFinite; on
// failure the try is unchanged. CPU only; nothing else changes (the motion already includes the viscous terms).
std::expected<void, SolidsFailure> ApplyViscousImpulses(std::span<const ViscousPushOnSolid> viscous_impulse,
                                                        SolidsTry& solids_try);
// REMOVED (A18): MaterialPushFields and the particle form of ApplyPressureToMaterials (push returned to
// blocking-material particles after MoveParticles, one step late). See 3.6a and decision 3.
class SolidsBuffers;                     // D word. Preallocated: map entries, blocking-material faces, reactions,
                                         //   sort keys, CollideSolids working buffers, block partials, OrderBuffers.
// kOutOfMemory when an allocation fails (the only function of the module that returns it). Every function that
// takes these buffers requires its arrays on their processor (debug assertion, A30).
// Plan decision (plan-decisions "Solids"): max_cell_blocks sizes the per-block arrays. Zero blocks: kBadInput.
// A17, batched-run buffers: signature unchanged; with settings.run_on = kCpu every entry has where = kCpu. Instances
// built from one Solids with different counts or processors may coexist (the step's whole instance at zero
// particles, its loaded instance, its CPU stage-17 instance, section 6); no instance points into another.
// CHANGED (A58; plan-decisions item 3 and its same-day correction): every entry is a piece of `run_memory`, the
// RunMemory of the processor the buffers live on: the run's block for run_on = the run's processor; the run's
// cpu_run_memory for the stage-17 instance (run_on = kCpu in a GPU run, every entry where = kCpu). Requires:
// settings.run_on == run_memory.run_on() (debug assertion, A30).
std::expected<SolidsBuffers, SolidsFailure> BuildSolidsBuffers(const SolidsSettings& settings,
                                                               std::size_t max_particle_count,
                                                               std::uint32_t max_cell_blocks,
                                                               RunMemory& run_memory);
```

### 3.6a `blocking_material_faces.h` (+ `_cpu.cc`, `_gpu.cu`), NEW (A18)

Declarations as `plan-decisions.md` "Coupling inside the pressure solve (A18)" and solids plan Task 23 give them.

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
// One entry per (face piece, set) where the piece touches at least one kLiquid cell and some of its corners are
// owned by a material of that set (5.5a). Sorted by face_piece_index, then set. More entries than
// max_motion_map_entries: kBuffersTooSmall.
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
// an earlier build. In a viscous scene stage 11 calls ApplyViscousSolveToMaterials instead (A26, 3.6b).
std::expected<PressureOnMaterials, SolidsFailure> ApplyPressureToMaterials(
    const BlockingMaterialFaces& faces, std::span<const double> velocity_after_pressure,
    std::span<const BlockingSet> blocking_sets, SolidsBuffers& buffers);
```

### 3.6b `viscous_material_faces.h` (+ `_cpu.cc`, `_gpu.cu`), NEW (A26)

Declarations as `plan-decisions.md` "A viscous liquid sticks, and free solids in the viscous solve (A26, A27)"
gives them; rule and failures in 5.5a item 2b and solids plan Tasks 23a, 23b.

```cpp
struct ViscousMaterialFace {             // One band face of one blocking set.
  grid::FacePlaceIndex face_piece_index{grid::kNoIndex};
  std::uint32_t material_set = 0;        // Index into the step's blocking sets.
  double face_mass = 0;                  // kg, > 0.
  double velocity_before_pressure = 0;   // m/s along +axis, after ApplyForces and CollideSolids.
  std::uint32_t blocking_face_index = 0xFFFFFFFFu;  // Its pressure::BlockingMaterialFace entry, or none.
};
class ViscousMaterialFaces;             // Move-only view into SolidsBuffers, sorted by face_piece_index, then set.
// Stage 10 of a viscous scene, after BuildBlockingMaterialFaces. `geometry` is the object inputs.geometry refers
// to (kBadInput otherwise); pressure_faces of an earlier build: kWrongTime; more than 2 * max_motion_map_entries
// entries: kBuffersTooSmall; a pressure_faces entry with no band entry: kBadInput.
std::expected<ViscousMaterialFaces, SolidsFailure> BuildViscousMaterialFaces(const BlockingMaterialFacesInputs&,
    const BlockingMaterialFaces& pressure_faces, const pressure::GeometrySnapshot& geometry, SolidsBuffers&);
// What viscosity::ViscosityInputs::material_faces takes.
std::span<const ViscousMaterialFace> ToViscousMaterialFaces(const ViscousMaterialFaces&);
// Stage 11, in place of ApplyPressureToMaterials. Writes each solved band velocity
// (viscosity::SolvedCouplingArrays::material_face_velocity_after_solve) into its set's GridFromParticles::velocity;
// every other face keeps its bytes. Push report as ApplyPressureToMaterials: per set the sum of
// face_mass * (after - before) * axis, block partials then math::TreeSum. kBadInput for a span of another size or
// another set list (nothing written); kWrongTime for faces of an earlier build.
std::expected<PressureOnMaterials, SolidsFailure> ApplyViscousSolveToMaterials(const ViscousMaterialFaces&,
    std::span<const double> velocity_after_solve, std::span<const BlockingSet>, SolidsBuffers&);
```

### 3.7 `contact_reactions.h` (+ `_cpu.cc`, `_gpu.cu`)

```cpp
struct ReactionFields {                  // One material and range, as given to contact::MoveParticles.
  std::size_t particle_count = 0;
  const std::uint64_t* particle_id = nullptr;
  const double* mass = nullptr;          // kg.
  const contact::SolidReaction* reaction = nullptr;
};
struct ContactOnSolids {                 // Per solid of SolidsTry::solids().
  std::span<const Vec3> push;            // N s on the solid: minus the sum of
                                         //   mass * SolidReaction::velocity_for_next_step_change.
  std::span<const Vec3> turning_push;    // N m s about the solid's origin at the step start.
  std::span<const double> work;          // J the solid's motion gave to the particles (5.6).
  std::span<const std::uint64_t> hit_count;  // Particles whose reaction was filled by this solid: an exactly zero
                                             // change of for_next_step claims no record (contact P5; to confirm).
};
// Sums the reactions per solid in ascending particle_id and keeps them in the try; CommitTry adds them to the
// velocity of free solids. Call once per try, after every MoveParticles call of that try, with the reactions of
// every particle of the try (in a batched run, after they were copied back): the result does not depend on
// ranges or batches. The gas material is never passed: a marker's mass is a weight, not a physical mass, so
// gas does not push solids (A10, one-way). A second call in one try replaces the first. Copies its per-solid
// results (push, turning push, work) into SolidsTry before returning (A17, batched-run buffers). In a batched run
// the step calls it on the CPU stage-17 instance (section 6), whose CollideSolids block partials were filled by
// WriteBlockPartialsRange.
std::expected<ContactOnSolids, SolidsFailure> ApplyContactReactions(
    std::span<const ReactionFields> materials, SolidsTry& solids_try, SolidsBuffers& buffers);
// NEW (A17, batched-run buffers; plan-decisions "Buffers in batched runs", solids plan Task 28). Copies the
// CollideSolids block-partials array of the range blocks of `piece` (piece positions range(), whole positions
// range_in_whole_grid(), one contiguous run each) from buffers whose partials were written on piece.grid() into
// buffers sized for the whole grid, on either processor pair (GPU to CPU with one slice ReadBuffer, CPU to CPU with
// a copy). kBadInput when the whole buffers hold fewer blocks or a partials array has another moving-solid count.
// Called by batches::ReadBatch after pass 1 and counted in BatchReport.
std::expected<void, SolidsFailure> WriteBlockPartialsRange(const SolidsBuffers& piece_buffers,
                                                           const grid::GridPiece& piece,
                                                           SolidsBuffers& whole_buffers);
```

### 3.8 `material_contact.h` (shared), `collide_solids.h`, `drive_waves.h`, `plate_sides.h`

```cpp
// material_contact.h (shared). Contact of two materials at one sample point. No bounce, Coulomb friction.
struct MaterialContactInputs {
  double first_mass = 0, second_mass = 0;            // kg, positive.
  Vec3 first_velocity{}, second_velocity{};          // m/s, world.
  Vec3 normal_toward_first{};                        // Unit vector, from the second material to the first.
  double friction = 0;                               // No unit, zero or more.
};
struct MaterialContactResult {
  Vec3 push_on_first{};                  // N s. The second material receives exactly the opposite.
  double approach_speed = 0;             // m/s. Negative means approaching. D word.
};
MaterialContactResult ComputeContactBetweenMaterials(const MaterialContactInputs& inputs);

// collide_solids.h (+ _cpu.cc, _gpu.cu).                              CollideSolids D (stage CollideSolids)
struct MaterialSetFaces {                // One material set: materials that share one grid velocity (A8).
  const transfer::GridFromParticles* from_particles;   // The set's. mass, value_kind, material_set are read.
  grid::FaceValues<double>* velocity;                  // m/s, after forces: from_particles->velocity. Changed
                                                       //   in place.
  const transfer::CellsFromParticles* cells_from_particles;   // The same set's (mass, material_set). The two
                                                       //   MaterialSetKey values must be equal (kWrongMaterialSet).
  double friction = 0;                                 // No unit. Smallest materials::Ingredient::friction of
                                                       //   the set. A pair uses the smaller of its two values.
  // The rule of the set's materials at solids (materials 3). All materials of one set must share it.
  materials::AtSolidFaces at_solid_faces = materials::AtSolidFaces::kParticlesOnly;
};
struct CollideSolidsInputs {
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;           // Faces stored in these blocks are written; touching cells are read.
  // Every material set of the scene except the gas set (gas meets solids through SolidCoverage only, and liquid
  // through the gas module's boundary), each once, in the step's set order. With one set only the rule at
  // solids runs (5.7 step 1).
  std::span<const MaterialSetFaces> material_sets;
  const transfer::SolidsAtFaces& solids_at_start;      // Same try, elapsed 0. Else kWrongTime.
};
struct CollideSolidsResult {
  std::uint64_t face_count_stopped_at_solid = 0;   // Faces of a kNoMotionIntoSolid set that were approaching.
  std::uint64_t contact_face_count = 0;  // Face pieces where two sets were approaching.
  std::uint64_t unclear_normal_count = 0;   // Face pieces skipped because the two sets were not apart.
  double largest_push = 0;               // N s.
};
// The pushes that faces gave to driven and free solids are kept in the try, one partial per cell block and
// solid, and added to ContactOnSolids (push, turning_push, work) by ApplyContactReactions as a tree sum over
// the block partials in the grid's fixed block order. At most blocking_materials.size() + 1 material sets per
// call, else kBuffersTooSmall; at one face, pairs run in material_sets order (plan-decisions "Solids").
// Fails: kWrongMaterialSet, kWrongTime, kWrongGridVersion, kBadInput (a set whose materials have different
// AtSolidFaces), kBuffersTooSmall, kNotFinite, kRunOnCallFailed.
std::expected<CollideSolidsResult, SolidsFailure> CollideSolids(const CollideSolidsInputs& inputs,
                                                                SolidsTry& solids_try, SolidsBuffers& buffers);

// drive_waves.h (CPU only; the sums run where the arrays live, through a private drive_waves_gpu.cu on the
// GPU, plan-decisions "Solids").                                         DriveWaves D (stage DriveWaves)
struct DriveWavesInputs {
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;
  const pressure::ReadyOpenSea* open_sea = nullptr;    // Null: the scene has no open sea.
  const grid::FaceValues<double>* face_mass = nullptr; // kg. GridFromParticles::mass of the set holding the
                                                       //   liquid.
  double step_end_time = 0, step_duration = 0;         // s.
};
// The totals of one call are a tree sum over its block partials. A batched run keeps the block partials of
// every range and sums them once over the whole grid (the step), so the total does not depend on the cut.
struct DriveWavesResult {
  Vec3 momentum_added{};                 // kg m/s the relaxation gave the liquid. Sum of mass * change.
  double energy_added = 0;               // J. Sum of mass * (after^2 - before^2) / 2.
  std::span<const Vec3> momentum_added_of_block;   // NEW. kg m/s, one per block of `blocks`, in block order.
  std::span<const double> energy_added_of_block;   // NEW. J, the same. In buffers until the next call.
};
// The step calls it (stage DriveWaves) on the velocity of the set holding the liquid. Calls
// pressure::ApplyOpenSeaRelaxation on `velocity` and measures what it added. The wave paddle needs no
// work here: it acts through the pressure solve and through contact, and is measured by PressureOnSolids.
std::expected<DriveWavesResult, SolidsFailure> DriveWaves(const DriveWavesInputs& inputs,
                                                          grid::FaceValues<double>& velocity,
                                                          SolidsBuffers& buffers);
struct WavePaddleSettings {
  double stroke = 0;                     // m, peak to peak, along travel_direction.
  double wave_period = 1;                // s.
  double ramp_duration = 0;              // s. The stroke grows linearly from zero over this time.
  Vec3 travel_direction{1, 0, 0};        // Horizontal unit vector, world.
  int keys_per_period = 32;
};
// Keys of a sine stroke from time 0 to end_time, starting at rest_origin. The result owns its keys.
std::vector<PathKey> BuildWavePaddlePath(const WavePaddleSettings& settings, Vec3 rest_origin, double end_time);

// plate_sides.h (+ _cpu.cc, _gpu.cu)
struct PlateSideFields {                 // One material and range.
  std::size_t particle_count = 0;
  const std::uint64_t* particle_id = nullptr;
  const Vec3* position = nullptr;        // m.
  particles::ArrayToChange<contact::PlateSide> plate_side{};
};
// Gives every particle whose side is {0, kNoPlate} the side of the scene's plate at the step start pose.
// Sides already given are never changed. Fails: kParticleInsidePlate (lowest particle_id).
std::expected<std::uint64_t, SolidsFailure> ApplyPlateSides(const SolidsTry& solids_try,
                                                            std::span<const PlateSideFields> materials);
// Sets every side of `plate_id` to {0, kNoPlate}: the two sides become one material again (5.9).
std::expected<void, SolidsFailure> ApplyPlateRemoval(std::uint32_t plate_id,
                                                     std::span<const PlateSideFields> materials);
double ComputeMinPlateThickness(Vec3 plate_normal, double largest_cell_width);   // m (5.9)
struct CutWorkFields {                   // One material that has internal force and a plate side.
  grid::CellBlockRange cell_blocks{};    // NEW. The blocks these particles are grouped in.
  const std::uint64_t* group_begin = nullptr;          // NEW. ParticleFields::group_begin: one partial per block.
  std::size_t particle_count = 0;
  const double* rest_volume = nullptr;                 // m3.
  const Mat3* internal_force_per_area = nullptr;       // Pa. D word.
  const Mat3* exact_velocity_change = nullptr;         // 1/s. From transfer::GridAtParticle.
  const Vec3* face_force_work_rate = nullptr;          // W per axis: sum over same-side faces of force * velocity
};                                                     //   after pressure. Computed by the step with grid weights.
struct CutWork { double deformation_work = 0, face_force_work = 0, cut_work = 0; };   // J
// A report (5.9). Per cell block: materials in the given order, then particles in grouped order; then a tree
// sum over the block partials in block order. All entries must cover the same cell_blocks (kBadInput).
std::expected<CutWork, SolidsFailure> ComputeCutWork(std::span<const CutWorkFields> materials,
                                                     double step_duration, SolidsBuffers& buffers);
```

### 3.9 `exchange_check.h` (CPU only)

```cpp
struct ExchangeTotals {                  // Per solid, running sums since time 0. Kept on commit only.
  Vec3 pressure_push{}, pressure_turning_push{};   // N s, N m s about the world origin.
  Vec3 contact_push{}, contact_turning_push{};
  double driven_work = 0;                // J the solid's motion gave to the material (pressure and contact).
  double swept_volume = 0;               // m3 pushed into liquid cells: dt * solid flow, summed over steps.
};
struct ExchangeTolerances {
  double relative = 1e-10;               // Of the larger side of a balance.
  double push_floor = 0;                 // N s. The step passes 1e-12 * total particle mass * largest speed.
  double work_floor = 0;                 // J.
};
struct ExchangeReport {
  double push_error = 0;                 // N s. |push on solids + push on the liquid|, the two summed separately.
  double turning_push_error = 0;         // N m s, about the world origin.
  double work_error = 0;                 // J. |work by solids on the liquid - flow_work|.
  double smallest_distance = 0;          // m. Smallest particle distance to a solid at the end pose.
  std::uint64_t particles_inside_solid = 0;   // 1 when the smallest end distance is below contact_tolerance / 2,
};                                            //   else 0 (the step keeps the real count). Must be 0.
struct ExchangeInputs {
  const MotionMap& motion_map; const grid::CellValues<double>& pressure;
  const pressure::GeometrySnapshot& geometry; const PressureOnSolids& pressure_on_solids;
  const ContactOnSolids& contact_on_solids; const contact::MoveParticlesResult& moved;
  double smallest_end_distance = 0, contact_tolerance = 0;   // m.
};
// Each side of a balance is a tree sum over block partials (whole grid). The liquid side is taken from the
// GeometrySnapshot's face pieces (pieces touching a kLiquid cell; a non-liquid cell's pressure counts 0). Fails
// with kExchangeNotBalanced and the worst balance in `magnitude`, also for the no-penetration check (magnitude =
// contact_tolerance / 2 - smallest_end_distance). Run in stage CheckStep.
std::expected<ExchangeReport, SolidsFailure> CheckSolidExchange(const ExchangeInputs& inputs,
                                                               const SolidsTry& solids_try,
                                                               const ExchangeTolerances& tolerances);
```

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by the named module, with
the name as that module's section 3 has it; for a module updated at the same time or later, as
`changes-to-apply.md` states its interface).

**Foundation (EXISTS, `shared-interface.md` (a) and the `base/` additions of `changes-to-apply.md`):** `Vec3`,
`Mat3`, `operator*` on both, `kIdentity`, `math::Dot`, `math::Cross`, `math::Transpose`, `math::Length`,
`math::Normalized`, `math::Rotation`, `math::Sqrt`, `math::Sin`, `math::Cos`, `math::Abs`, `math::Min`,
`math::Max`, `math::IsFinite`, `math::TreeSum`, `Axis`, `kUp`, `RunOn`, `CheckCanRunOn(RunOn)`,
`FLUIDITY_CPU_AND_GPU`, `cpu_only.h`, `std::expected`, `Buffer`, `BuildBuffer`, `ReadBuffer`, `WriteBuffer`,
`ComputeTreeSum`, `ComputeTreeDot`, `BuildStableOrder`, `BuildGroupsByDestination`, `MemoryEstimateEntry`, `CountedPer`,
`KeptFor`, `PerParticleVelocity<T> { T for_next_step; T for_this_step; }` (A1). As built (plan-decisions):
`MemoryEstimateEntry::where` (`MemoryWhere { kRunOn, kCpu }`; per-solid results, per-set pushes on blocking sets and the
open-to-sky box arrays are `kCpu` entries), `RunOn` in the shared `base/run_on.h`, `CheckCanRunOn` in the CPU-only
`base/check_can_run_on.h`; buffers on one processor are a precondition, so no processor mismatch is mapped (A30).
`OrderBuffers` taken by `BuildStableOrder`, `ComputeExclusivePrefixSum`, `BuildGroupsByDestination` (one form, A31).
NEW (A58; Foundation Tasks 17 to 19): `RunMemory`, `BuildBuffer(RunOn, size, RunMemory&)`, the `RunMemory&` form of
`BuildOrderBuffers`, `BufferError::kRunMemoryTooSmall` (mapped to `kOutOfMemory`), `ComputeRunMemoryBytes(estimate,
counts, where)`.

**Contact (PROVIDED, contact 3.1 to 3.5):** `Solid`, `SolidShape`, `SolidPose`, `SolidMotion`, `ClosedSides`,
`DomainBox`, `PlateSide`, `SolidSide`, `kNoSolid`, `ToSolid`, `CheckSolid`, `ComputeSolidPose`,
`ComputeSolidVelocity`, `DistanceToSolid`, `ComputeDistanceToSolid`, `FindSolidSide`, `SolidVelocitySample`,
`ComputeSolidVelocitySample`, `SolidReaction::velocity_for_next_step_change`, `MoveParticlesResult`, `ContactCloseCall::kSecondReactingSolid`, `kMaxTurnPerStep`,
`ComputeContactTolerance`. This module sets `reports_reaction` for driven and free solids and copies
`friction_scale`.

**Grid (PROVIDED, grid 3.1 to 3.7):** `StaggeredGrid`, `GridTables`, `GridFrame`, `GridVersion`,
`CellBlockRange`, `CellPlaceIndex`, `FacePlaceIndex`, `CornerPlaceIndex`, `kNoIndex`, `GridFailure`,
`CellValues`, `FaceValues`, `CornerValues`, `BuildCellValues`, `BuildFaceValues`, `BuildCornerValues` (for
`run_on` and `max_cell_blocks`), `ApplyGridVersion(values, grid)` (the step binds `SolidCoverageValues` and the
`SolidsAtFaces` arrays to each try grid; this module checks `grid_version()` and `block_count()`),
`FacePieceKind`, `ComputeFacePosition`, `ComputeFaceArea`, `ToAxis`, `FindCellLow`, `FindCellHigh`,
`FindFacePieces`, `FindCornerPlaces` (cell and face), `ComputeCornerPosition`, `ComputeCellCenter`,
`ComputeCellWidth`, `ToSizeLevel`, `ToCellBlockIndex`, `FindTouchingCells`, `ComputeCompleteWeightedSpread`,
`GridPiece` (`grid()`, `range()`, `range_in_whole_grid()`, for `WriteBlockPartialsRange`), `GridTables::block_ids` and `same_level_blocks` (for the box occupancy and the particle search), read on the
grid's own processor: A17 builds a GPU run's grid on the GPU and `tables(RunOn::kCpu)` of a GPU grid has null
arrays, so `FindGridEdgesOpenToSky` reads `block_ids` in a GPU pass there (5.4 step 5); `grid::WriteValues`.
Blocks are visited in the grid's fixed block order, which is also the order of the block partials.

**Particles (PROVIDED, particles 3.1 to 3.3):** `ParticlesTry`, `BuildFieldsToRead(particles_try,
material_index, cell_blocks)`, `BuildFieldsToChange` with `Field::kPlateSide`, `Field::kVelocityForNextStep`,
`Field::kFittedVelocityChange`; `ParticleFields::cell_blocks`, `id`, `position`, `mass`, `size_level`,
`velocity.for_next_step`, `fitted_velocity_change`, `rest_volume`, `plate_side`, `group_begin`, `cell_begin`;
`ParticleFieldsToChange::velocity.for_next_step`, `fitted_velocity_change`, `plate_side`; `ArrayToChange`.
`plate_side` is copied by `SplitParticle`, required equal by `MergeParticles` (`kParticlesDiffer`), moved by
regrouping and by `WriteParticleGroups`/`ReadParticleGroups`, and stored by `ParticlesSnapshot`: that is how
the cut history survives (5.9).

**Transfer (PROVIDED, transfer 3; material sets, owner A8):** `SolidsAtFaces` (`outside_solid`; `side`; `solid_velocity`; `time`; `grid_version`), one per time, serving every set;
`GridFromParticles` (`mass`, `value_kind`, `velocity`, `material_set`), one per material set; `FaceValueKind`;
`CellsFromParticles` (`mass`, `material_set`), one per set; `MaterialSetKey`; `GridAtParticle::
exact_velocity_change`; `kMaxSidesAtFace`. Each stage of transfer is called once per set with that set's
`material_is_counted`; `SaveGridAfterPressure` of a set no solve writes (snow, springy solids) takes the set's
velocity after `CollideSolids` and, on the pieces the liquid presses, after `ApplyPressureToMaterials` (A18;
in a viscous scene on the band faces after `ApplyViscousSolveToMaterials`, A26), with `was_solved` null. Builders `BuildSolidsAtFaces`, `BuildGridFromParticles`, `BuildCellsFromParticles` and
the stage functions, each taking a `transfer::TransferBuffers` (one form, A31; plan-decisions "Transfer"), are what tests and the liquid loop use.

**Pressure (PROVIDED, pressure 3):** `SolidCoverage` (filled here over every active cell), `FreeSolidPush`,
`EquationInputs::free_solid_pushes` and `free_solid_inverse_inertia` (A59: the free-solid inertia entries, `FreeSolidInertia`), `GeometrySnapshot`, `CellGeometry::
content` with `CellContent::kLiquid`, `FacePieceGeometry::open_area` and `solid_volume_rate`,
`SolvedArrays::pressure` (the step's try array), `ReadyOpenSea`, `ApplyOpenSeaRelaxation`,
`AirPocketInputs::grid_edge_is_open_to_sky`, `kNoEquation`, and the two functions added for this module:
`ComputeBlockedCornerAreas(distance_to_solid[4], area, corner_area[4])` (`geometry_clip_math.h`, shared) and
`ComputeEquationIndexes(geometry, blocks, equation_of_cell)`. No face is clipped twice and the order of the
equations is pressure's. CHANGED (A59; pressure owns it, `plan-decisions.md` item 5): `pressure::FreeSolidInertia
{ mass; inertia_at_start; allowed_motions }` replaces the two inverse-inertia `Mat3` per free solid (this module
fills it with `ToFreeSolidInertia`), and `pressure::ComputeGuidedSolidMotion(solid, xi_before, impulse, xi_after)`
(`pressure/free_solid_guide_math.h`, shared: `q = (J^T M J)^-1 J^T (M xi_before + impulse)`, `xi = J q + xi_0`) and
`ComputeGuidedInverseMass` are the one formula for the guided inverse mass, which `ApplyPressureToSolids`,
`ComputeAllowedVelocity` and `ComputeInverseInertia` call. Pressure's `solid_volume_rate` and
`free_solid_motion_before_pressure` come from the guided motion before pressure. Pressure includes the header-only
`solids/guide_math.h` for `AllowedMotions`; pressure Task 47a follows solids Task 5a. A18 (`plan-decisions.md`, pressure plan Tasks 43 to 50): `BlockingMaterialFace`
(`face_piece_index`, `material_set`, `blocked_area`, `face_mass`, `velocity_before_pressure`,
`coverage_volume_rate`), `EquationInputs::blocking_material_faces`,
`SolvedArrays::blocking_material_velocity_after_pressure` (one per face, input order),
`BufferSizes::max_blocking_material_faces` (the step sets it to `max_motion_map_entries`),
`ComputeMaterialVelocityAfterPressure` (`cell_equation_math.h`, used by tests). Equations with material faces are
solved by `SolvePressure` and, since A24, by viscosity's joint solve (the pressure pieces accept material faces;
`plan-decisions.md` "A24: pressure pieces"); a non-liquid end of a material face is at pressure 0, so gas and air-pocket
pressure push no solid and no blocking material.

**Viscosity (PROVIDED, viscosity 3; A26, A27 as `plan-decisions.md` "A viscous liquid sticks, and free solids
in the viscous solve" declares it):** `SolveViscosityAndPressure` (owner A5, A6) takes
`ViscosityInputs::material_faces` (`std::span<const solids::ViscousMaterialFace>`, from `ToViscousMaterialFaces`)
and `ViscosityInputs::free_solids` (`std::span<const solids::FreeSolidForViscousSolve>`, from
`ToFreeSolidsForViscousSolve`), and writes the cell pressure, `SolvedCouplingArrays::
material_face_velocity_after_solve` (one per `ViscousMaterialFace`) and `SolvedCouplingArrays::free_solid_motion`
(`std::span<solids::FreeSolidMotion>`, one per free solid); `blocking_material_velocity_after_pressure` stays filled
for the `BlockingMaterialFace` subset. The viscosity design uses the solids types; solids does not include
viscosity. Band faces and free-solid motions are velocity unknowns of the joint solve: the liquid sticks to them
(A26), and free solids beside a viscous liquid run (A27: `kFreeSolidNotSupported`, `kViscousLiquidWithFreeSolid`
and `has_free_solid_touching_liquid` removed; no combination of materials is refused because of viscosity). The
A24 elimination of material faces (and the slide) holds only in `SolvePressure` (decisions 14, 15). `SolidsSummary`
gives the step what `ViscositySceneSummary` needs (`has_driven_solid`, `has_free_solid`, `has_inclined_solid`,
`has_blocking_material`). NEW (A59, A60, A53): `FreeSolidForViscousSolve::allowed_motions` (the solve's
unknowns are the m generalized velocities, block `J^T (M + dt R^T E_s^T W E_s R) J`); `ViscosityResult::
viscous_push_on_solids` (`std::span<const solids::ViscousPushOnSolid>`, one per `SolidsTry::solids()` entry, every
motion kind), which `ApplyViscousImpulses` takes; `ViscosityInputs::solid_slip_length` (`SolidsTry::slip_length()`).

**Materials (PROVIDED, materials 3):** `MaterialKind` (`kSnow`, `kSpringySolid`), `HasInternalForce`,
`AtSolidFaces { kParticlesOnly, kNoMotionIntoSolid }` and `Material::at_solid_faces`, `Ingredient::friction`
(`MaterialTable::ingredient_friction()`), `MaterialTable`, `BuildParticlesSettings(..., PlatesInScene)` (which
gives every material a `plate_side` array when the scene has a plate, and always to `kSpringySolid`), and
`MaterialBuffers::internal_force_per_area()` (for `ComputeCutWork`). Materials owns the response of a springy
solid; this module only keeps its two sides apart.

What materials asks of this module (its section 4), and where it is provided:

| Materials needs | Provided by |
|---|---|
| 1. `CollideSolids` applies `kNoMotionIntoSolid` to faces of snow and springy solids, leaves `kParticlesOnly` sets alone | `MaterialSetFaces::at_solid_faces`, 5.7 step 1 |
| 2. `pressure::SolidCoverage` filled from snow and springy-solid particles; the pressure push returned to them | `SolidsSettings::blocking_materials`, `ComputeSolidCoverage`, `BuildBlockingMaterialFaces`, `ApplyPressureToMaterials` (5.4 step 3, 5.5a; inside the liquid solve, the set's face velocity written back in the same step, A18; replaces "one step late", decision 3) |
| 3. `plate_side` assigned and cleared on `kSpringySolid` particles (and on every other material of a scene with a plate) | `ApplyPlateSides`, `ApplyPlateRemoval` |
| 3. `friction_scale` per solid | `SolidSettings::friction_scale`, copied into `contact::Solid` by `BuildTry` |

Materials supplies no list of blocking materials, no `particle_radius` and no material sets: they come from
the ready scene (I/O below). Default: every material with `HasInternalForce` blocks liquid and is its own set.

**Gas (PROVIDED, as `changes-to-apply.md` "09 Gas" states it):** nothing consumed. Gas reads
`pressure::SolidCoverage` over every active cell, also where there is no liquid, and `grid_edge_is_open_to_sky`.
A free solid moves gas but is not pushed by it: gas sees it through `solid_volume_rate` at its velocity before
pressure, and no gas pressure enters the motion map (5.5 step 4). The gas set is not passed to `CollideSolids`.

**Step (PROVIDED, after its update by `changes-to-apply.md` "10 Step"):** `ReadyScene::solids` is a
`solids::SolidsSettings`; the capability check calls `CheckSolidsCanRun(settings, smallest_cell_width,
largest_cell_width)` (the one signature) and reports every entry, beside contact's `CheckSolid`. The step owns
the two `transfer::SolidsAtFaces` (start, end), `SolidCoverageValues` (built with `BuildSolidCoverageValues`),
`grid_edge_is_open_to_sky`, the span `"step: material face velocity after pressure"` (given
`max_motion_map_entries`, 8 B) and the `SolidsBuffers` (in a batched run three instances: whole, loaded in the
loaded `StepTry`, and the CPU stage-17 instance reached through `step::FindContactReactionBuffers`, null in one
piece; section 6). Per try it calls, in this order:
`ComputeStepDurationLimit` (before the try), `BuildTry`; stage `FindSolidSurfaces`: `ApplyPlateSides`,
`ComputeSolidsAtFaces` at elapsed 0, `ComputeSolidCoverage`, then in its `run_whole_grid` part
`FindGridEdgesOpenToSky` (plan-decisions "Buffers in batched runs", step plan Appendix A row M11); stage `DriveWaves`:
`DriveWaves` (owner of the call, decision 8); stage `CollideSolids`: `CollideSolids` with every non-gas material
set; stage 10 `BuildPressureEquations`: `BuildMotionMap`, then (scene with liquid and a blocking set)
`BuildBlockingMaterialFaces` with the whole grid and every blocking set, then `ToFreeSolidPushes`,
`ToFreeSolidInertia` (A59; was `ToFreeSolidInverseInertia`) and `ToBlockingMaterialFaces` into `EquationInputs` (whatever stage follows, 11 or 12; A24),
and in a viscous scene (A26, A27) `BuildViscousMaterialFaces` after `BuildBlockingMaterialFaces` (scene with a
blocking set) and `ToFreeSolidsForViscousSolve(motion_map)` into `ViscosityInputs`; stage 12 `SolvePressure`:
`SolvedArrays::blocking_material_velocity_after_pressure` is the step-owned span; after the solve
`ApplyPressureToSolids`, then `ApplyPressureToMaterials(faces, that span, blocking sets, buffers)` (its
`PressureOnMaterials::push` per set goes into the report), then `ComputeSolidsAtFaces` at elapsed
`step_duration`, all before `SaveGridAfterPressure`; stage 11 `SolveViscosityAndPressure` (A26, A27; replaces
A24's mirror of stage 12): the step-owned spans `"step: material face velocity after solve"` (given
`max_motion_map_entries` x 2, 8 B) and `"step: free solid motion"` (given the free solid count, 48 B) are
`SolvedCouplingArrays::material_face_velocity_after_solve` and `free_solid_motion`; after the solve
`ApplySolvedMotionToSolids(motion_map, free_solid_motion, pressure, solids_try, buffers)` in place of
`ApplyPressureToSolids`, then (scene with a blocking set) `ApplyViscousSolveToMaterials(viscous_faces, that span,
blocking sets, buffers)` in place of `ApplyPressureToMaterials`, then `ComputeSolidsAtFaces`; viscosity buffers
`max_material_faces = 2 x max_motion_map_entries`, `max_free_solids` = the free solid count; stage
`MoveParticles`: `contact::MoveParticles` with `SolidsTry::solids()`; stage 17 `MoveSolids`:
`ApplyContactReactions` (every material but gas), `ComputeCutWork` (scenes with a plate), no longer
`ApplyPressureToMaterials` (A18); in a batched run both on the CPU with the stage-17 instance (their per-particle
inputs are in CPU memory); stage `CheckStep`: `CheckSolidExchange` (after stage 11 its `pressure_on_solids` is
`SolidsTry::pressure_on_solids()`); stage `CommitStep`: `CommitTry` (or
`RejectTry`), then `ApplyPlateRemoval` when a plate left the scene. It shortens the step on
`kPathJumpInsideStep`, `kSolidMovesTooFar` (from `BuildTry`, `ApplyPressureToSolids` or
`ApplySolvedMotionToSolids`) and contact's
`kSecondReactingSolid`, and sums the block partials of per-range calls in block order. When there is no solid
but the domain box, stage 2 fills both `SolidsAtFaces` with `ComputeSolidsAtFaces` once per new grid version
(plan-decisions "Step"). NEW (A58, A60, A53): it builds the `SolidsBuffers` and the coverage arrays with the
`RunMemory` of their processor (the stage-17 instance with `cpu_run_memory`), passes `SolidsTry::slip_length()` and
`ViscosityResult::viscous_push_on_solids` to `ApplyViscousImpulses` (stage 11, after `ApplySolvedMotionToSolids`), and copies
`SolidsTry::liquid_forces()` of the committed try into `StepReport::liquid_force_on_solids`.

**Cell resize and batches (PROVIDED, cell resize and batches 3, 5):** remaps nothing of this module (every
array is recomputed per try). After a resize it calls `ComputeSolidsAtFaces(grid, blocks, solids,
solids_at_faces)` on the resized grid at the committed pose (for `particles::NeighbourhoodInputs`). In both
batched modes (`kBatched`, `kBatchedWithSolvesInPieces`, A15) it calls `ComputeSolidsAtFaces`,
`ComputeSolidCoverage`, `CollideSolids` and `DriveWaves` per batch (on the loaded instance), and
`FindGridEdgesOpenToSky` (after pass 1), `BuildMotionMap`, `BuildBlockingMaterialFaces`, `ApplyPressureToSolids`,
`ApplyPressureToMaterials`, in a viscous scene `BuildViscousMaterialFaces`, `ApplySolvedMotionToSolids` and
`ApplyViscousSolveToMaterials` (A26, A27), and `CheckSolidExchange` once on the whole grid, so no push list crosses a batch;
`batches::ReadBatch` calls `WriteBlockPartialsRange` after pass 1 for each range (counted in `BatchReport`), and
stage 17 runs once on the CPU instance; per-solid totals are block partials, so the cut does not change them.
Stage `FindSolidSurfaces`' range part calls `BuildTry` and the contact checks; in a batched run that part runs per
batch, and the step builds the solids try only in its first call of a try (step plan Task 33a; Open, section 11). With solves in pieces the geometry, coverage, motion map and blocking-material faces stay whole on
the run's processor (plan-decisions "Batched solves (A15)"), so these functions need no change; the liquid solve
streams the material-face terms with the rows of their two cells and writes each entry's solved velocity from the
batch whose range stores its face place. No solids signature changes.

**I/O and CLI (PROVIDED, as `changes-to-apply.md` "12 I/O and CLI" states it):** turns the scene's solids into
`SolidsSettings` (rotation keys become `angular_velocity_until_next` through `ComputeRotationVector`; inertia
of a box from its mass and size when the artist gives none); gives `blocking_materials` with a
`particle_radius` each, `has_liquid_or_gas`, and which materials form a material set; stores a
`SolidsSnapshot` in `saved_state` at frame times.

**Surface, foam and spray (PROVIDED, as `changes-to-apply.md` "13" states it):** nothing consumed. It calls
`BuildContactSolids(settings, snapshot)` and reads `SolidsTry::solids()` through the step's
`FoamAndSprayInputs::solids`.

**Blender add-on (PROVIDED, as `changes-to-apply.md` "14" states it):** nothing consumed. It shows the
entries of `CheckSolidsCanRun` (codes from `ToName`), the constants `kMaxPlateCount` and `kMaxFreeSolidCount`,
and builds paddle paths with `WavePaddleSettings`.

**For the sand module:** sand will be a blocking material (`MaterialTheLiquidCannotEnter`) and its own
material set whose `at_solid_faces` is `kParticlesOnly`: `CollideSolids` step 1 leaves its faces alone, so its
particles meet solids in `contact::MoveParticles` with the approach speed friction needs. Nothing
sand-specific is added here.

Per material this module carries: liquids and mixtures, the motion map and the paddle; snow and springy
solids, `MaterialTheLiquidCannotEnter`, `CollideSolids` and `plate_side`; moving solids, everything; gas,
`SolidCoverage` over every active cell and `grid_edge_is_open_to_sky` (a free solid moves gas and is not
pushed by it).

## 5. Method

Symbols are used in formulas only: `dt` step duration, `w` cell width, `p` pressure above atmosphere.

**5.1 One solid array per try (new).** `BuildTry` turns each committed `SolidState` into a `contact::Solid`:
`pose_at_step_start` from the state, `motion` from 5.2 or 5.3, `reports_reaction = true` unless fixed. Solids
are few; the array is rebuilt per try and is the only description of solids any stage reads. Every array
computed from it carries `time` and `try_id`; a consumer that is handed another time or try returns
`kWrongTime` (report CUDA 10.5).

**5.2 Driven solids (path ported from the two-key paddle, generalised).** Between two keys the origin moves in
a straight line and the solid turns at the key's constant angular velocity, so the motion inside a step is
exactly the constant `SolidMotion` contact expects: `linear_velocity = (origin_next - origin_key) /
(time_next - time_key)`. A key is a jump. `ComputeStepDurationLimit` returns the time to the next key;
`BuildTry` fails with `kPathJumpInsideStep` when a key lies strictly inside the step (a key within
`1e-12 * max(1, time)` of an end counts as that end). After the last key the solid rests, or leaves the scene
(`is_in_scene = false` from the first step that starts at or after the last key). The second limit is travel:
`(|linear_velocity| + |angular_velocity| * radius) * dt <= max_travel_in_cells * smallest_cell_width`, `radius`
the half diagonal of a box and 0 for a flat solid or plate (ported: `liquid_substep_3d.cc:2738-2741`, half a
cell), and the turn `|angular_velocity| * dt <= kMaxTurnPerStep`.

**5.3 Free solids (ported from `AdvanceRigidBody3D`).** Steps 1 to 4 are numbered as in the code.

1. *Velocity before pressure* (`BuildTry`). With `R = ToAxes(orientation)`, in the solid's own axes
   `w_own = R^T w`: `linear_velocity += dt * (gravity + steady_force / mass)`,
   `angular_velocity += dt * R * ((- w_own x (I w_own)) / I)` (division entry by entry; no torque is stored
   between steps, because every push is already in the committed velocity). A motion the guide forbids is set
   to zero. CHANGED (A59): the motion is projected onto the guide's allowed motions, `xi <- J (J^T M J)^-1 J^T M xi +
   xi_0` (`ComputeAllowedVelocity`; for axis limits the selection, so "set to zero" is unchanged bitwise; a rail
   keeps the speed along `d`; a hinge keeps the turn about its axis with the matching `V = Omega x (o - p)`).
2. *Velocity after pressure* (`ApplyPressureToSolids`, 5.5). This is the velocity the solid moves with for the
   whole step, as on `main` ("the coupled velocity sets both wall flux and drift").
3. *Contact* (`ApplyContactReactions`, 5.6). The summed reaction changes the velocity the solid ends the step
   with, not its motion during the step (ported lag of `liquid_substep_3d.cc:2700-2722`).
4. *Commit.* `origin += dt * linear_velocity_after_pressure`; `orientation = ComputeTurnedOrientation(start,
   dt * angular_velocity_after_pressure)`; committed velocity `= after pressure + inverse inertia * contact
   push`, the inverse inertia taken at the end pose. CHANGED (A59; `ComputePoseAfterStep`): a rail moves the origin
   along `d` only; a hinge rotates the pose about `hinge_point` by `dt (Omega . a)` about `hinge_axis` instead of
   translating it by `dt V`, so the pivot distance is kept to rounding (positions are frozen during the solve, so a
   translation would drift at first order; `literature-practice.md` topic 3); the contact push goes through all four
   blocks of the inverse inertia, so the committed velocity stays in the allowed motions. `RejectTry` keeps the old state: pose, velocity, totals.

Not supported, as on `main`: a free solid within one cell width of another solid (`kSolidsTooClose`); no
solid-against-solid contact exists. A free solid may leave the liquid and fall freely.

**5.4 Solids on the grid.**

1. *Faces* (`ComputeSolidsAtFaces`, ported meaning, new for several solids). At the face centre, with the
   poses `elapsed` seconds into the step: `solid_velocity = contact::ComputeSolidVelocitySample(solids, count,
   elapsed, centre, kSolidReachInCells * w)`; `outside_solid = 0` when its `distance <= 0`; `side =
   {plate_id, contact::FindSolidSide(plate, pose, centre)}` for the scene's plate, `{0, kNoPlate}` when there
   is none or the centre is inside its thickness. Blocking materials do not clear `outside_solid`: their own
   particles must still reach those faces. The committed-pose form (cell resize) uses the committed poses and
   velocities with `elapsed = 0`; for fixed solids and driven solids between keys it gives the same bytes as
   the next try at elapsed 0.
2. *Corners, analytic solids* (`ComputeSolidCoverage`, every corner and face place of the blocks, whether or
   not a cell holds liquid: gas reads the same coverage). `distance_to_solid = min over solids of
   contact::ComputeDistanceToSolid` at the corner, capped at `kCoverageCapInCells * w`; ties keep the lower
   solid index; `corner_owner.solid_index` is that solid when the distance is below the cap;
   `corner_velocity = contact::ComputeSolidVelocity` there (before pressure for a free solid). A closed side
   of the domain box counts; an open side does not.
3. *Corners, blocking materials* (ported from `SamplePlasticSolidBoundary3D`, new search). For each blocking
   material, `distance = min over its particles of |corner - position| - particle_radius`. If that is smaller
   than the analytic distance the corner is owned by the nearest particle (ties: lower `particle_id`) and
   `corner_velocity = velocity.for_next_step + fitted_velocity_change * (corner - position)`. Only particles
   within `kCoverageCapInCells * w + particle_radius` can matter, so the search reads the cell block of the
   corner and its 26 same-size-level neighbours with their bigger and smaller blocks. `CheckSolidsCanRun` rejects a
   `particle_radius` above one block width minus that cap at the smallest cell width (`(4 -
   kCoverageCapInCells) * smallest_cell_width`), which is what makes the 27 blocks enough (report
   CUDA 10.2). The exhaustive scan stays as a test oracle.
4. *Per face piece.* `solid_normal_velocity = sum over the 4 corners of corner_area * (corner_velocity .
   axis) / sum of corner_area`, with `pressure::ComputeBlockedCornerAreas`; zero when nothing is blocked.
   Pressure then forms `solid_volume_rate = (area - open_area) * solid_normal_velocity`, which is
   `sum of corner_area * (corner_velocity . axis)`: the motion map of 5.5.
5. *Open to sky* (`FindGridEdgesOpenToSky`, new; pressure step 9, Q26 as proposed; liquid and gas read it).
   Space is cut into boxes of one biggest-level block. A box is *free* when no cell block of any level lies in it and
   `ComputeDistanceToSolid` at its centre exceeds half its diagonal for every solid. From every free box of
   the top layer of an open-topped domain, flood through face-touching free boxes in ascending box order. A
   grid-edge face piece is marked when the box holding the point half a cell width beyond it was reached.
   Boxes are the biggest-level block lattice of the grid frame over the domain box; the top layer is the boxes
   whose top reaches the domain top; a point above an open top counts as reached, a point outside any other
   side as not reached; nothing is marked when `closed.top`. *Where each part runs (A17; plan-decisions "A17
   follow-ups").* (i) Box occupancy, one byte per box: on a CPU grid from the CPU tables; on a GPU grid one
   thread per cell block writes 1 at its box (an idempotent byte store, no sum), then one `ReadBuffer` of the box
   bytes. (ii) On the CPU, unchanged: the solid test at each box centre and the flood. (iii) Face marks: on a CPU
   grid as before; on a GPU grid one `WriteBuffer` of the reached-box bytes, then one thread per face place reads
   the box beyond its grid-edge piece. Copies: 2 B per box, nothing per block; no block id comes down. Shared
   per-place formulas (box of a block, box beyond a piece) live in `solids_on_grid_math.h`. Marks are exact and
   equal on both processors.

**5.5 The motion map and its transpose (new form of ported maths; replaces the piston, box and flat-front
paths).** One entry per (liquid cell `i`, face piece `f` of that cell with a blocked part, corner `c` of `f`
with an owner), with `signed_corner_area = s * corner_area[c]`, `s = +1` when `i` is the low cell of `f` and
`-1` when it is the high cell (pressure's sign, step 14).

1. *Motion.* The solid flow out of cell `i` through its blocked parts is `sum over its entries of
   signed_corner_area * (corner_velocity . axis_f)`. For a solid that moves as one piece,
   `corner_velocity = v + w x arm`, `arm = corner position - origin`, so the flow is `push_i . v +
   turning_i . w` with `push_i = sum signed_corner_area * axis_f` (m2) and `turning_i = sum signed_corner_area *
   (arm x axis_f)` (m3). These two are `FreeSolidPush::force_per_pressure` and `torque_per_pressure` of
   equation `i`.
2. *Transpose.* Pressure `p_i` pushes the owner of each entry with `dt * p_i * signed_corner_area * axis_f`
   (N s) at the corner. Summed for a one-piece solid: `push = dt * sum_i p_i push_i`, `turning_push = dt * sum_i
   p_i turning_i`. The work identity `sum_i p_i * flow_i * dt = push . v + turning_push . w` holds term by
   term, because both sides use the same areas, the same corners and the same arms (report 10.2; old paper
   [21]).
3. *Why corners.* The corner areas integrate the linear corner weights over the blocked part, so `sum
   corner_area * corner position` is the area-weighted centre of the blocked part, and the map is exact for
   any flat blocked part moving as one piece. The same entries serve a material whose corners have different
   owners. `main` used a patch centroid for boxes and corners for snow; this is the corner form for all.
4. *In the solve* (pressure step 15). For free solids pressure adds `dt * push^T * inverse_inertia * push`;
   the six extra solves of `main` are gone. CHANGED (A59): pressure reads `FreeSolidInertia` (`ToFreeSolidInertia`),
   so a guided solid adds the rank-m term `(K J)(J^T M J)^-1 (K J)^T`. *In a viscous scene (A27):* each free solid's six motions (`V`, `Omega`
   about its origin at the step start) are unknowns of viscosity's joint solve, not eliminated: its row is
   `[m I3, 0; 0, I_start] xi_after = [m I3, 0; 0, I_start] xi_before + dt sum_e (force_per_pressure,
   torque_per_pressure)_e p_e - (viscous traction)`, the pressure part from the same `FreeSolidPush` entries as
   here, `xi_before` the motion before pressure (`FreeSolidForViscousSolve`), and the liquid sticks at its surface
   (a sample beside it reads `e_a . V + ((y - origin) x e_a) . Omega` instead of a known solid rate). Faces covered
   by the solid are not unknowns of their own. Driven and fixed solids enter through `solid_volume_rate` only;
   blocking materials enter as material faces where their set has face mass (5.5a), and through
   `solid_volume_rate` only elsewhere. The map is built from `kLiquid` cells only. Gas builds its own equations from the
   same geometry (`kGas` cells), so a free solid moves gas through `solid_volume_rate` at its velocity before
   pressure, and no gas pressure is returned to it: the coupling with gas is one-way.
5. *After the solve* (`ApplyPressureToSolids`). Per solid and cell block, the block's entries are summed in
   (cell, face, corner) order with `math::TreeSum`; the block partials are then summed with `math::TreeSum` in
   the grid's fixed block order. Free: `motion = ComputeMotionAfterPush(before pressure, inverse inertia at
   the start pose, push, turning_push)`, then the travel and turn limits of 5.2 are checked again on the
   velocity after pressure (`kSolidMovesTooFar`); CHANGED (A59): the motion after pressure is
   `pressure::ComputeGuidedSolidMotion` (the guided form of `ComputeMotionAfterPush`). In a viscous scene (A27) `ApplySolvedMotionToSolids` does the
   same pressure sums and keeps them as the pressure part, but sets the motion to the solved one (**replaced**
   (A59): "a motion a guide forbids keeps its value before pressure"; the solve's generalized velocities keep it in
   the allowed motions), checks the limits again, and reports the whole change
   `mass (after - before)` separately (`SolidsTry::solved_motion()`). Driven: the push is reported and the work it gives the
   material, `-(push . v + turning_push . w)` with `push` the push on the solid, is added to `driven_work`;
   `flow_work = dt * sum_i p_i * (solid volume rate into cell i)` is the same number (sign of work, section 3).
   The pressure is that of whichever solve ran (A5, A6). No push is stored for blocking particles (A18);
   `push_to_materials` is the sum of magnitudes of the entries whose owner is a blocking material, a report.
6. *Blocking materials.* Replaced by 5.5a (A18). The earlier rule (`ApplyPressureToMaterials` after
   `MoveParticles` adding each entry's push to its owner particle's `velocity.for_next_step` and
   `fitted_velocity_change`, one step late; decision 3) is removed.

**5.5b The force of the liquid on every solid (NEW, A60; `plan-decisions.md` item 6; Uhlmann 2005, Batty 2007).**
Per solid of the try (fixed, driven, free, the domain box), from the solve's own terms, so the sum is exactly the
momentum exchanged: the pressure part is 5.5 step 5's push and turning push (the motion map entries of every solid,
already summed for driven and fixed solids); the viscous part is viscosity's `-dt R^T E_s^T W (E x + g)` over the
samples that read the solid (`viscous_push_on_solids`), 0 in an inviscid scene. `LiquidForceOnSolid` divides both by
`dt` (mean force and torque over the step), world axes, torque about the origin at the step start. A guided solid
reports all six numbers, the guide's reaction included. For a free unguided solid pressure plus viscous push equals
`M (xi_after - xi_before_pressure)` to the solve's tolerance (viscosity's invariant test).

**Slip length (NEW, A53).** `SolidSettings::slip_length` reaches viscosity through `SolidsTry::slip_length()`; this
module only checks it and passes it on (viscosity 5, "Slip length").

**5.5a Blocking-material faces inside the liquid solve (A18; replaces decision 3's lag; plan Task 23).** A
blocking material's inertia goes into the liquid pressure solve through its own set's face (decision 1 b gives
every blocking set its own face mass and velocity): the papers' implicit form (`literature-coupling.md` 3; Batty
2007, IQ-MPM's coupled projection) lumped to one number per face, which is option c of decision 3.

1. *Entries* (`BuildBlockingMaterialFaces`, stage `BuildPressureEquations` after `BuildMotionMap`, whole grid;
   for stage 11 and stage 12 alike, A24; in stage 11 they mark which band faces carry a pressure term, item 2b).
   One entry per (face piece, blocking set) where the piece touches at least one `kLiquid` cell and some of its
   corners are owned (`corner_owner.material_index`) by a material of that set: `blocked_area` = sum of those
   corners' `ComputeBlockedCornerAreas`; `face_mass` = the set's `GridFromParticles::mass` at the piece;
   `velocity_before_pressure` = the set's `velocity` there (after `ApplyForces` and `CollideSolids`);
   `coverage_volume_rate` = sum over those corners of `corner_area * (corner_velocity . axis)` (the share of
   pressure's `solid_volume_rate` these corners gave). Sorted by `face_piece_index`, then set; two passes (count
   per block, fill at the prefix sum). A piece whose set has no face mass there (value kind not
   `kFromParticles`) gives no entry, is counted in `faces_without_mass()`, and its blocked part stays in
   `solid_volume_rate` at the sampled velocity (one way, as before A18). A piece touching no `kLiquid` cell gives
   no entry: the liquid does not move that face.
2. *In the solve* (pressure, method in its plan Task 44). Each entry is a two-point term of weight
   `dt A^2 / m` (`ComputeMaterialFaceCoefficients`); its flow `A u - coverage_volume_rate` replaces the coverage
   part in the flow imbalance. An end with no equation is at pressure 0, so gas and air-pocket pressure push no
   blocking material, as for solids (limit b). This is the inviscid case (stage 12): the face velocity is
   eliminated and the liquid slides along the material. **Replaced (A26):** "In a viscous scene viscosity's
   joint solve carries the same term (A24): the face velocity is eliminated exactly, its weight joins the outer
   loop's diagonal unscaled, and at a liquid-material face the viscous solve slides (no tangential stick)"; see
   item 2b.
2b. *In a viscous scene* (A26; `plan-decisions.md` "A viscous liquid sticks, and free solids in the viscous
   solve"; plan Tasks 23a, 23b). The band faces become velocity unknowns `w` of viscosity's joint solve, and the
   liquid sticks to them. *Band* (`BuildViscousMaterialFaces`, stage 10 after `BuildBlockingMaterialFaces`): per
   blocking set, every face piece where the set's `value_kind` is `kFromParticles` whose centre lies within
   `grid::kWeightCurveReach` (1.5, equal to viscosity's `kFitReach`) widths of the centre of a cell with
   `filled_volume > 0` (`<=`; the width of the piece's smaller cell): the faces every deformation-rate sample of
   the liquid can read. Every `BlockingMaterialFace` entry is in the band (`blocking_face_index` points to it);
   two sets on one piece are both listed and viscosity uses the one with the larger face mass (ties: lower set).
   *Row:* `face_mass w_after = face_mass w_before + dt A_k (p_low - p_high)` (the pressure part only for band
   faces with a `BlockingMaterialFace` entry, `A_k` its blocked area) minus the viscous traction from the liquid;
   `w_before` is the set's velocity after `ApplyForces` (which applied `materials::ApplyInternalForce`) and
   `CollideSolids`, so the internal force is counted once. The material carries no sample weight of its own.
   Nothing is eliminated (`g_k = dt A^2 / m` does not apply). The rest of the material keeps its velocity after
   `CollideSolids` this step; the drag reaches it next step through `ApplyInternalForce`. *After the solve*
   (`ApplyViscousSolveToMaterials`, stage 11 in place of `ApplyPressureToMaterials`): each band face's solved
   velocity is written into its set's `GridFromParticles::velocity`; push per set `face_mass * (after - before) *
   axis` as item 3. An inviscid liquid (stage 12) keeps items 2 and 3: the slide and the elimination.
3. *After the solve* (`ApplyPressureToMaterials`, stage `SolvePressure`, right after `ApplyPressureToSolids`,
   before `ComputeSolidsAtFaces` at `dt` and `SaveGridAfterPressure`; in a viscous scene item 2b instead). The solved
   `u + (dt A / m) (p_low - p_high)` of each entry (`SolvedArrays::blocking_material_velocity_after_pressure`) is
   written into its set's `GridFromParticles::velocity` at the piece; every other face of the set keeps its bytes.
   The set's particles receive it through the set's ordinary `SaveGridAfterPressure` and `GridToParticles` in
   the same step; tangential motion stays the set's own (A8; inviscid liquid only, A26). `PressureOnMaterials::push` per set is the sum of
   `face_mass * (after - before) * axis` (block partials, then `math::TreeSum`); it equals minus the push on the
   liquid through the same faces.
4. *Not built* (Open, section 11): the material's own volumetric stiffness inside the solve (half of the
   volumetric response in IQ-MPM): `materials::ApplyInternalForce` already applies the full stress.

Limits carried, not tuned. (a) *Piston receiving volume*: in a surface cell the whole blocked area pushes,
although liquid covers only part of it; the swept-volume test of section 7 keeps the 1 % limit of `main`, is
expected to fail, and is labelled known open physics outside the gate (decision 4). (b) Air-pocket and
tracked-gas pressure do not push solids: only the pressure of `kLiquid` cells enters the map, as on `main`.
(c) One owner per corner: where two solids block one face piece each keeps its own corners.

**5.6 Contact reactions (ported).** For every particle with `reaction.solid_index != kNoSolid`: push on the
solid `= -mass * velocity_for_next_step_change`; turning push `= (hit_position - origin at step start) x
push`; work `= -(mass * velocity_for_next_step_change) . surface velocity at hit_position` counted as given by
the solid. The entries are ordered by `particle_id` (a stable order of the hits by id, grouped by solid) and
summed with `math::TreeSum` per solid; the order is the same whatever the ranges or batches. The face pushes
of 5.7 step 1, kept per solid and block in the try, are then added to the same three numbers as a tree sum
over the block partials. Pressure and contact are separate sums in `ExchangeTotals` and are each added to a
free solid once (report 10.3). In a batched run the face pushes were written per range on the loaded instance
during pass 1 and reach the CPU stage-17 instance through `WriteBlockPartialsRange` (range blocks only, same
whole-grid block positions), so the tree sum over block partials is the one-piece sum; the results are copied into
`SolidsTry` for `CommitTry`. Gas markers hit solids in `MoveParticles` like every particle, but their
reactions are not summed: a free solid moves gas and is not pushed by it.

**5.7 `CollideSolids` (rules ported; sampling new).** Each material set has its own face velocities (A8);
this stage is where sets meet each other and solids on the grid. Liquids need no rule at solids: the pressure
solve keeps them out (their sets are `kParticlesOnly`). The gas set is not passed: gas meets solids through
`SolidCoverage` and liquid through the gas module's boundary. Every material meets fixed, driven and free
solids particle by particle in `contact::MoveParticles` (5.6). This stage adds the two rules that act on face
velocities after forces. All velocities are read before any is written, so the order of faces does not matter.
Before any write it checks each entry's two `MaterialSetKey` values and that no set is given twice
(`kWrongMaterialSet`).

1. *A material's faces at a solid* (ported from `dry_solid_step_3d.cc:367-378`, where it is one branch for
   snow at an axis-aligned floor and platen; generalised to any normal). For each set with
   `kNoMotionIntoSolid` (snow, springy solids) and each of its faces with `value_kind == kFromParticles`
   whose `solids_at_start.solid_velocity` has `distance <= kFaceStopReachInCells * w`: sample the set's
   velocity vector `v` at the face as in step 3; `approach_speed = (v - solid_velocity) .
   normal_out_of_solid`; if negative, `u_a -= approach_speed * normal_a`. No friction is applied on faces:
   the particle sweep does that. A set with `kParticlesOnly` is left alone: its particles meet the solid in
   `contact::MoveParticles` with the approach speed friction needs. The push taken from the face, `mass *
   approach_speed * normal_a` along the face's axis at the face centre, is given to that solid (push, turning
   push about its origin, work at the solid's surface velocity), summed per solid and cell block in ascending
   face place with `math::TreeSum`. This is what lets a platen load a bed of snow and a free solid rest on it.
2. *Two sets on one face: where.* A face piece of axis `a` on which two sets both have `value_kind ==
   kFromParticles`. Pairs are visited in ascending (face place, first set, second set).
3. *Sample.* Per set: mass `m` and velocity part `u_a` of the face; the two other velocity parts are the
   mass-weighted mean of that set's faces of those axes on `cell_low` and `cell_high` (old paper
   `<sec-contact>`: "reconstructed from the four neighbouring faces"); a missing part takes the other
   side's value (the other set, or the solid in step 1), so it adds no relative motion.
4. *Normal.* `normal_toward_first = Normalized(X_first - X_second)`, `X` the centre of a set's
   `CellsFromParticles::mass` over `cell_low`, `cell_high` and the cells touching them. Closer than `1e-6 w`:
   the face is skipped and counted in `unclear_normal_count`.
5. *Rule* (`ComputeContactBetweenMaterials`, unchanged from `ResolveDistinctBodyContact3D`). With
   `relative = first - second` and `approach_speed = relative . normal`: nothing if `approach_speed >= 0`.
   Else `shared_mass = m_small / (1 + m_small / m_large)`, `normal_push = -approach_speed * shared_mass`,
   `sideways_push = min(sideways_speed * shared_mass, friction * normal_push)` against the sideways motion;
   `push_on_first = normal_push * normal - sideways_push * sideways direction`.
6. *Write.* Only the part along the face's axis is applied here: `u_first += push_a / m_first`,
   `u_second -= push_a / m_second`. The other parts are applied when their own faces are visited. Momentum
   along each axis is unchanged face by face. Step 1 is applied first, then steps 2 to 6 on its result.

With one set only step 1 runs. Steps 3 and 4 are new: `main` has them only for flat axis-aligned fronts
(decision 1).

**5.8 `DriveWaves`.** The step calls it (decision 8). Order as on `main` and in report 13.7: after
`SaveGridBeforeForces`, so the relaxation is part of what the particles receive once. It works on the face
velocity of the set holding the liquid. The stage copies the velocities of the two zones, calls
`pressure::ApplyOpenSeaRelaxation`, and sums `mass * (after - before)` and the energy change per cell block in
ascending face place with `math::TreeSum`, then over the block partials in block order. The paddle's part of
the forcing is `PressureOnSolids::flow_work` and `ExchangeTotals::swept_volume`.

**5.9 Cutting (ported, with its limits stated).**

- *What a cut is.* The scene's one plate (`kMaxPlateCount`; `kTooManyPlates` otherwise) is unbounded
  sideways, as on `main` and as contact decided (Q37). Its plane splits space into two sides. Every particle of
  every material carries a `plate_side`; a particle and a face exchange only when their sides are equal
  (transfer 5.1). So the material is separated along the whole plane from the first step, and the plate's
  motion along its normal pushes the two sides apart. Physical fracture is not modelled (report 9.6).
- *Assigning.* `ApplyPlateSides` runs in stage `FindSolidSurfaces` on particles with `{0, kNoPlate}`: the
  committed ones on the first step, new ones each step. A side is written once and never changed while the
  plate is in the scene; contact returns `kOnWrongSideOfPlate` if a particle is found across.
- *History.* The history of a cut is exactly: the particles' `plate_side` (kept through split, merge only
  among equal sides, regrouping and `ParticlesSnapshot`) and the plate's `SolidState` in `SolidsSnapshot`.
  Faces hold no history: their side is recomputed from the plate's pose at the start and at the end of each
  step, and transfer drops the after-pressure value of a face whose side changed.
- *Leaving.* With `AfterPath::kLeavesScene` the plate is removed at the end of its path and
  `ApplyPlateRemoval` clears its sides in the same commit: the two sides then share faces again and behave
  as one material where they touch. This is the stated reconnection rule. A cut that stays open after the
  plate left needs the two sides to become two material sets (decision 2: not built).
- *Supported thickness.* Any plate: at least 1024 contact tolerances (contact, `kSolidTooThin`). A plate in a
  scene with liquid or gas must also be seen by the corner samples as a closed layer:
  `2 * half_thickness >= (|n.x| + |n.y| + |n.z| + max(|n.x|, |n.y|, |n.z|)) * largest_cell_width`
  (`ComputeMinPlateThickness`; 2 cell widths for an axis-aligned plate, at most 2.74). Then some layer of
  cells has all eight corners inside the plate, no cell holds liquid on both sides, and pressure never meets
  `kCellSplitBySolid` because of a plate (Q38, recommended option). `CheckSolidsCanRun` applies it when
  `SolidsSettings::has_liquid_or_gas`. A plate that only cuts snow or a springy solid has no grid minimum.
- *Cut work* (`ComputeCutWork`, a report). `deformation_work = dt * sum rest_volume * (internal_force_per_area
  : exact_velocity_change)`; `face_force_work = dt * sum of face_force_work_rate`; `cut_work` is their sum,
  each a tree sum over block partials. It is zero for uncut material (the work identity of report 9.3) and is
  what the same-side rule absorbed at the plate.

**5.10 Checks (`CheckSolidExchange`, new).** (1) Push: the push on all solids, summed per solid from the
motion map, plus the push on the liquid, summed per face piece as `-dt * (p_low - p_high) * (area -
open_area) * axis` from the snapshot, is zero within `push_floor + relative * larger side`. The two sums walk
different arrays in different orders, so agreement shows the map and the snapshot describe the same areas.
(2) Turning push: the same about the world origin, each face piece's push placed at
`sum corner_area * corner position / sum corner_area`. (3) Driven work: `sum over solids of (push . v +
turning_push . w)` equals `flow_work`. (4) No penetration: `particles_inside_solid == 0` from the end
distances the step collects with `contact::ComputeDistanceToSolid`. Totals are added on commit only.

## 6. CPU and GPU

**Where the state lives.** `Solids` and `SolidsTry` are CPU memory on both processors: a few hundred bytes
per solid, sent with each call as contact already does. Per step the GPU returns two `Vec3` per solid from
the pressure sums, two from the contact sums, one `Vec3` per blocking set from `ApplyPressureToMaterials`, the
counts, and for `FindGridEdgesOpenToSky` one byte per box each way (A17). In one piece no per-face, per-corner,
per-block or per-particle value crosses (A17: a run is all CPU or all GPU; this replaces the reading of Q41). In a
batched GPU run the one planned solids copy is `WriteBlockPartialsRange`: per range, the `CollideSolids` block
partials of the range blocks (`80 * moving solids + 32` B per block) go to the CPU stage-17 instance, counted in
`BatchReport`; the contact sums then run on the CPU, so their two `Vec3` per solid do not come back from the GPU.
Per-solid and per-set results (entries `where = kCpu`) are "small records", not counted (plan-decisions "Copies
counted").

**Instances in a batched run (A17; plan-decisions "Buffers in batched runs"; step plan Appendix A rows 16, 18,
21, 24, 27, M9 to M11).** Placement rule for whole try arrays (both batched modes, either processor): a whole array
lives on the run's processor when a part at `kWholeGridOnProcessor` reads or writes it; else in CPU memory when a
`kWholeStateOnCpu` stage reads or writes it, the commit keeps it, or a later pass reads it; an array read and
written only inside one pass has no whole instance, only the loaded one. For solids this gives up to three
`SolidsBuffers` built from one `Solids` (no signature changes; the builder takes the counts):

| Instance | Built with | Used by |
|---|---|---|
| whole | run's processor, zero particles, `max_cell_blocks` | stages 10 to 12 and `CheckSolidExchange`: motion map, equation of cell, `FreeSolidPush` list, blocking-material faces |
| loaded (owned by the loaded `StepTry`) | run's processor, `max_loaded_particles` per material, `max_loaded_blocks` | pass stages per batch: `ComputeSolidCoverage`, `CollideSolids` working buffers and block partials, `DriveWaves` |
| stage 17 (`step::FindContactReactionBuffers`) | `run_on = kCpu`, whole particle limit, `max_cell_blocks`; entries named `"solids: ..."`, all `where = kCpu` | `ApplyContactReactions`, `ComputeCutWork`: reaction keys and order, block partials, per-solid results |

In one piece only the whole instance exists (at the particle limit) and `FindContactReactionBuffers` is null. The
step-owned arrays follow the same rule: `SolidsAtFaces` start and end whole in CPU memory plus loaded;
`grid_edge_is_open_to_sky` whole on the run's processor; coverage whole on the run's processor plus loaded
(Appendix A rows 17 to 19).

**Shared formula headers:** `solid_state.h`, `solid_dynamics.h`, `motion_map_math.h` (one entry's push and
turning push; `solid_normal_velocity` of one face piece; one corner's owner from a candidate list),
`material_contact.h`, and contact's and pressure's shared headers.

**CPU loops.** One loop over face places (`ComputeSolidsAtFaces`), over corner places
(`ComputeSolidCoverage`), over cells (`BuildMotionMap`), each cut into fixed chunks; a destination is written
by one thread. Sums use `math::TreeSum` over the fixed orders of section 5.

**GPU layout (owner in brackets; all stay on the device).**

| Array | Element | Per | Owner |
|---|---|---|---|
| `SolidsAtFaces` x 2 (start, end): `outside_solid`, `side`, `solid_velocity` | 1 + 8 + 72 B | face place | step |
| `distance_to_solid`, `corner_owner`, `corner_velocity` | 8 + 16 + 24 B | corner place | step (`SolidCoverageValues`) |
| `solid_normal_velocity`, `grid_edge_is_open_to_sky` | 8 + 1 B | face place | step |
| motion map entries; equation index per cell | 24 B; 4 B | entry; cell place | `SolidsBuffers` |
| `FreeSolidPush` list | 56 B | free solid x liquid cell it touches | `SolidsBuffers` |
| blocking-material faces (A18; replaces the pushes owed to particles) | `sizeof(pressure::BlockingMaterialFace)` + 4 B count per block | entry (`max_motion_map_entries`) | `SolidsBuffers` |
| material face velocity after pressure (A18) | 8 B | entry (`max_motion_map_entries`) | step |
| viscous material faces (A26) | `sizeof(ViscousMaterialFace)` (32) + 4 B count per block | entry (`2 x max_motion_map_entries`) | `SolidsBuffers` |
| material face velocity after solve (A26) | 8 B | entry (`2 x max_motion_map_entries`) | step |
| free solids for the viscous solve (A27) | `sizeof(FreeSolidForViscousSolve)` (504 since A59: 160 + `sizeof(AllowedMotions)` 344) | free solid (`kMaxFreeSolidCount`) | `SolidsBuffers` |
| free solid motion (A27) | 48 B | free solid | step |
| per-set pushes on blocking sets | 24 B, `MemoryWhere::kCpu` | blocking material | `SolidsBuffers` |
| open-to-sky box occupancy and reached boxes (A17) | 1 + 1 B, `kCpu`; the same on `kRunOn` for a GPU grid | box of the domain's lattice | `SolidsBuffers` |
| sort keys and order of reactions | 16 B | particle | `SolidsBuffers` |
| before-relaxation velocity copy; contact pair working values | 8 B; 8 B | face place | `SolidsBuffers` |
| block partials of `CollideSolids` and `DriveWaves` | 3 x 24 + 8 B; 24 + 8 B | cell block x moving solid; cell block | `SolidsBuffers` |

Per cell block this is about 45 kB, of which 39 kB are the two `SolidVelocitySample` arrays transfer
declared (decision 7). The material sets add nothing here except the `CollideSolids` working buffers (`8 *
(blocking_materials.size() + 1)` B per face place): transfer counts their arrays (59,856 + 512 x ingredients
bytes per set per block, 48,336 + 512 x ingredients for a set with no solve; plan-decisions "Transfer").
`BuildSolidsMemoryEstimate` returns these entries (the step-owned ones with `KeptFor::kOneStepTry`), each with its
`MemoryWhere` and bytes from `sizeof` of the real types with a `static_assert`, and `BuildSolidsBuffers`
allocates by walking its own entries. CHANGED (A58): one entry per array (an entry's `copies` equal the number of
arrays of one element size it stands for), every array a 256-byte-aligned piece of a `RunMemory`; the combined rows
of the table above are split. In `kBatchedWithSolvesInPieces` no entry of the whole or loaded instance moves to
the CPU (geometry and solve inputs stay whole on the run's processor, A15); in both batched modes the stage-17
instance is in CPU memory (above), and its entries are part of the step's `MemoryEstimate::on_cpu`.

**Work division.** One thread per face place or corner place; the corner search over particles reads the
particle groups of at most 27 same-level blocks and their bigger and smaller blocks. The motion map is built
in two passes (count per cell, then fill at the prefix sum), so entries are in (cell, face, corner) order
without a sort. Per-solid sums group entries by solid with `BuildGroupsByDestination` and reduce each group
with the tree of `math::TreeSum`; per-particle pushes do the same grouped by particle. Contact reactions are
compacted, ordered with two passes of `BuildStableOrder` (id, then solid) and reduced per solid.
`CollideSolids` is one thread per face place reading the before-contact velocities and writing its own face.
`BuildBlockingMaterialFaces` is two passes like the motion map (count per block, fill at the prefix sum);
`ApplyPressureToMaterials` is one thread per entry writing its own face (entries of one set are distinct pieces),
the per-set push grouped and reduced like the per-solid sums. A26, A27: `BuildViscousMaterialFaces` is two passes
like the blocking faces; `ApplyViscousSolveToMaterials` one thread per entry like `ApplyPressureToMaterials`; the
free-solid list is computed on the CPU from the try and sent in one `WriteBuffer`; `ApplySolvedMotionToSolids`
reuses the pressure sums and reads back the motion span (48 B per free solid). `FindGridEdgesOpenToSky`: on a CPU grid all on
the CPU; on a GPU grid one thread per cell block (occupancy), the flood on the CPU, one thread per face place
(marks), 5.4 step 5 (A17).

**Ranges of cell blocks.** Every per-place function takes a `CellBlockRange` and writes only places stored in
its blocks. `ComputeSolidsAtFaces`, `ComputeSolidCoverage`, `CollideSolids` and `DriveWaves` run per range
(per batch in a batched run): `ComputeSolidCoverage` reads particles of border blocks; `CollideSolids` reads
the cells touching the range (the ordinary border). `BuildMotionMap`, `BuildBlockingMaterialFaces`,
`ApplyPressureToSolids`, `ApplyPressureToMaterials` and `FindGridEdgesOpenToSky` (the step's `run_whole_grid`
part of stage `FindSolidSurfaces`) run once per try on the whole grid, so no push list is
merged across batches; with solves in pieces (A15) they still run whole, because geometry, coverage and their
outputs stay whole on the run's processor (plan-decisions "Batched solves (A15)", open point 3). `ApplyContactReactions`
and `ComputeCutWork` run once per try on the CPU stage-17 instance in a batched run. Every total is a block
partial first and a tree sum over block partials in the grid's fixed block order second; contact reactions
are ordered by `particle_id`. Either way the bytes do not depend on where batches are cut.

**CPU versus GPU tolerance (proposal; the hardware run calibrates it).** Owners, sides, `outside_solid`,
counts and the open-to-sky marks: exact. A floating-point value `a` against `b` passes when
`|a - b| <= floor + 0x1p-40 * max(|a|, |b|)`, with `floor = contact_tolerance` for distances,
`0x1p-40 * largest speed` for velocities, and `0x1p-40 *` (largest pressure `* w^2 * dt`) for pushes.

## 7. Tests

`TEST(FunctionOrType, Situation_Outcome)`, Z-up numbers. Analytic shapes are the oracles throughout.

**Analytic and conservation**

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeDrivenMotion, TwoKeys_ConstantVelocity` | keys at 0 s and 2 s, origins 1 m apart | velocity 0.5 m/s; pose at 1 s halfway | `1e-15` relative |
| `BuildTry, KeyInsideStep_IsPathJumpInsideStep` | key at 0.3 s, step 0 to 0.5 s | error, `limit` 0.3 s | exact error, `1e-15` |
| `BuildTry, TurningKey_PoseMatchesContact` | 1 rad/s about Z for 0.5 s | committed axes equal `contact::ComputeSolidPose` | `1e-14` |
| `ComputeVelocityBeforePressure, FreeFall_GainsGTimesDt` | free box, no liquid | `v.z -= 9.81 dt`; 100 steps: `z = z0 - g t^2 / 2 - g t dt / 2` | `1e-13` relative |
| `CommitTry, NoTorque_KeepsTurningMomentum` | inertia `{1, 2, 3}`, spin about a skew axis, 10^4 steps of 1e-4 s | world turning momentum constant; energy drift below 1e-6 | `1e-6` relative (first order in `dt`) |
| `ComputeInverseInertia, GuidedAlongZ_OnlyZMoves` | limits: z only | rows x, y and turning are zero | exact |
| `RejectTry, AfterEveryStage_ChangesNothing` | run all stage calls, then reject | snapshot before equals after | bytes |
| `ComputeSolidsAtFaces, PlateAcrossFaces_SidesAndInside` | plate normal `{0.6, 0, 0.8}` | side of each centre by the closed form; `outside_solid` 0 inside | exact |
| `ComputeSolidsAtFaces, CommittedPose_EqualsNextTryAtStart` | a fixed box and a driven flat solid between keys, after a commit | the committed-pose form equals the next try's at elapsed 0 | bytes |
| `ComputeRotationVector, TurnThenMeasure_GivesTheTurn` | 1,000 random orientations and rotation vectors with `|v| < pi` | `ComputeRotationVector(o, ComputeTurnedOrientation(o, v))` equals `v` | `1e-13` rad |
| `CheckSolidsCanRun, ThreeBrokenRules_AllReported` | two plates, a free solid of mass 0, a plate thinner than `ComputeMinPlateThickness` with liquid | three entries with `solid_id`, ordered as declared; the same scene without them gives none | exact |
| `BuildContactSolids, Snapshot_PosesAndVelocities` | snapshot of a free box and a driven plate that left the scene | one `contact::Solid`, pose and motion of the box | exact |
| `ComputeSolidCoverage, NoLiquid_CoversEveryActiveCell` | a box in a domain with gas only | every corner and face place of the blocks written, equal to the same scene with liquid | bytes |
| `ComputeSolidCoverage, FlatSolid_BlockedAreaIsExact` | oblique flat solid through mixed-size cells | `sum corner_area` per piece equals the exact polygon area | `1e-12 w^2` |
| `ComputeSolidCoverage, Box_BlockedAreaConverges` | box corner inside cells, three widths | error falls as `w^2` | ratio 4 within 20 % |
| `ComputeSolidCoverage, BallsOfParticles_MatchesExhaustiveScan` | 2,000 random particles, sparse gaps, particles just outside the reach, two at one place | distance and owner equal the scan over all particles | exact |
| `BuildMotionMap, AnyMotion_FlowEqualsPushDotVelocity` | turned box, random `v`, `w` | per cell, `solid_volume_rate` sum equals `push_i . v + turning_i . w` | `1e-13` relative |
| `ApplyPressureToSolids, AnyPressure_WorkBalances` | same, random pressures | `sum p_i flow_i dt = push . v + turning_push . w` | `1e-13` relative |
| `ApplyPressureToSolids, UniformPressure_NoNetPushOnClosedBox` | submerged box, `p` constant | push and turning push zero | `1e-12 p w^2 dt` |
| `ApplyPressureToSolids, Hydrostatic_PushIsWeightOfDisplacedLiquid` | box, `p = rho g depth` at cell centres | `push.z / dt = rho g V` | first order in `w`: 2 % at 16 cells per side |
| `ApplyPressureToSolids, OffCentrePressure_TurningPushIsArmCrossPush` | pressure on one face piece only | closed form with the blocked centre | `1e-13` relative |
| `ApplyPressureToSolids, FastAfterPressure_IsSolidMovesTooFar` (plan Task 21) | free solid slow before pressure, fast after | `kSolidMovesTooFar` | exact error |
| `ApplyPressureToSolids, PaddlePushingLiquid_WorkIsPositive` (plan Task 21) | driven flat solid moving into liquid at `p > 0` | `flow_work > 0` (sign of work) | exact sign |
| `FreeSolid, LightBoxOnOneCellOfLiquid_OneSolveGivesSharedVelocity` | guided box of mass `m` on a sealed column of mass `M` moving at `U` | both move at `M U / (M + m)` after one solve | solve limit |
| ~~`ApplyPressureToMaterials, AnyPressure_ParticleMomentumEqualsPush`~~ | removed (A18): no push is returned to particles | - | - |
| `BuildBlockingMaterialFaces, SnowLayerUnderLiquid_OneEntryPerBlockedPiece` (NEW, A18, plan Task 23) | snow layer (one blocking material, 300 particles, seed 19) under liquid | entries exactly the pieces touching `kLiquid` with snow corners; per piece `blocked_area == area - open_area`, `coverage_volume_rate` equals the snapshot's `solid_volume_rate` | `1e-14` relative |
| `BuildBlockingMaterialFaces, TwoSetsOnOnePiece_TwoEntriesInSetOrder`, `..., PieceWithoutSetMass_CountedNotEntered` (`faces_without_mass == 1`), `..., MaterialInNoSet_IsBadInput` (and a material in two sets), `..., CoverageOfOtherTry_IsWrongTime`, `..., BlocksNotWholeGrid_IsBadInput` (NEW, A18) | as named | as named | exact |
| `ApplyPressureToMaterials, MaterialMomentumEqualsPushOnLiquid` (NEW, A18) | same layer, random `p` (seed 19), each entry's velocity after from `pressure::ComputeMaterialVelocityAfterPressure` (non-liquid end 0) | `PressureOnMaterials::push` equals minus the push on the liquid (`sum -dt (p_low - p_high) blocked_area axis`); the set's velocity changed at entry pieces only | `1e-13` relative; other bytes equal |
| `ApplyPressureToMaterials, SpanOfOtherSize_IsBadInput` (NEW, A18) | velocity span one short | error, nothing written | exact |
| `BlockingMaterial, LightSnowOnOneCellOfLiquid_OneSolveGivesSharedVelocity` (NEW, A18) | the free-solid lid case with snow as the upper cell: one blocking set, set mass `m` 0.5 kg, `M = 1` kg, `U` 1 m/s | the set's face velocity and the liquid's low face velocity both `M U / (M + m) = 2/3` m/s after one `SolvePressure` | `2e-5 w + 1e-12` |
| `BuildViscousMaterialFaces, SpringyLayerUnderLiquid_EntriesAreTheBand` (NEW, A26, plan Task 23a) | springy layer (one `kSpringySolid` set, 300 particles, seed 19, z in [0, 0.05] m, w 0.025) under liquid to 0.125 m | entries exactly the pieces with set face mass within 1.5 w of a `filled_volume > 0` cell centre (brute-force scan in the test), sorted; mass and velocity equal the set's; the layer's bottom faces are not entries | exact |
| `BuildViscousMaterialFaces, EveryBlockingFace_IndexPointsBack`, `..., SpringyLid_OneEntry` (one entry, mass 0.5 kg, velocity 0, `blocking_face_index` 0), `..., SetWithoutFaceMass_GivesNone`, `..., TwoSetsOnOnePiece_BothListed` (0.5 and 0.3 kg, set order), `..., MoreEntriesThanBuffers_IsBuffersTooSmall`, `..., OtherGeometry_IsBadInput`, `..., BlockingFacesOfEarlierBuild_IsWrongTime` (NEW, A26) | as named | as named | exact |
| `ApplyViscousSolveToMaterials, SpringyLid_VelocityAndPush` (NEW, A26, plan Task 23b) | lid, after 2/3 m/s | face velocity 2/3; push `{0, 0, 0.5 * 2/3}` N s | exact |
| `ApplyViscousSolveToMaterials, SpringyLayer_WrittenAndPushIsMassTimesChange` (NEW, A26) | after = before `+ 0.01 ((k mod 7) - 3)` m/s | each entry written; other bytes unchanged; push = sum `face_mass (after - before) axis` | exact; `1e-13` relative |
| `ApplyViscousSolveToMaterials, BlockingSubsetOnly_EqualsApplyPressureToMaterials`, `..., SpanOfOtherSize_IsBadInput`, `..., FacesOfEarlierBuild_IsWrongTime` (NEW, A26) | as named | push equal to `ApplyPressureToMaterials`'s; errors, nothing written | `1e-13` relative; exact |
| `ToFreeSolidsForViscousSolve, TwoFreeBoxes_EqualTryBeforePressure` (NEW, A27, plan Task 23c) | box 0.5 kg, inertia `{1e-3, 2e-3, 3e-3}`; box 0.2 kg, own inertia `{2e-4, 3e-4, 4e-4}` turned a quarter about z, spin `{0, 0, 0.5}`; `dt` 0.002 from rest | two entries in free order; world inertia diag `{1e-3, 2e-3, 3e-3}` and `{3e-4, 2e-4, 4e-4}`; velocities equal the try's (`{0, 0, -0.01962}`) | exact; `1e-19` kg m2 |
| `ToFreeSolidsForViscousSolve, BoxAboveLiquid_StillListed`, `..., NoFreeSolid_IsEmpty` (NEW, A27) | as named | as named | exact |
| `ApplySolvedMotionToSolids, HandMotion_WrittenAndPushIsMassTimesChange` (NEW, A27) | pressure 0; motion `{0.01, 0, 0.03}` m/s, `{0, 0, 0.2}` rad/s | motion written; push `{0.005, 0, 0.02481}` N s, turning push `{0, 0, 6e-4}` N m s | exact |
| `ApplySolvedMotionToSolids, PressurePart_EqualsApplyPressureToSolids` (NEW, A27) | hydrostatic pressure | pressure part and `pressure_on_solids()` equal `ApplyPressureToSolids`'s | bytes |
| ~~`ApplySolvedMotionToSolids, GuidedBox_ForbiddenMotionsKeepTheirValue`~~ | replaced (A59): the solve keeps the motion in the allowed set; `ApplySolvedMotionToSolids, GuidedBox_MotionWrittenAsGiven` (limits z only; motion `{0, 0, 0.03}`, 0: written bytes; push `{0, 0, 0.02481}`) | as named | exact |
| `ComputeAllowedMotions, AxisLimits_GiveSelectionColumns` (NEW, A59) | z only and no turning; x, y and turning; unguided | columns `e_3`; `e_1, e_2, e_4, e_5, e_6`; the identity, `column_count` 1, 5, 6 | exact |
| `ComputeAllowedMotions, RailAndHinge_GiveOneColumn` (NEW, A59) | rail `d = (cos 30, 0, sin 30)`; hinge `a = e_y`, `p = o + (0, 0, 0.5)` | `[d; 0]`; `[a x (o - p); a] = [(-0.5, 0, 0); e_y]` | exact; `1e-16` |
| `ComputeAllowedVelocity, Hinge_KeepsOnlyTheTurnAboutTheAxis` (NEW, A59) | box 0.1 m, 1 kg, hinge 0.5 m above its centre about y, motion V (1, 2, 3), Omega (0.1, 0.2, 0.3) | the result is `q [a x (o - p); a]` with `q = (m (a x (o - p)) . V + a^T I Omega) / (m l^2 + I_cm)` | `1e-15` relative |
| `ComputeInverseInertia, Hinge_IsOneColumnOverGeneralizedMass` (NEW, A59) | the same hinge | the four blocks of `J J^T / (m l^2 + I_cm)`; `turning_per_push` = transpose of `linear_per_turning_push` | `1e-15` relative; exact |
| `ComputePoseAfterStep, Hinge_KeepsPivotDistance` (NEW, A59) | the same hinge, turn rate 1 rad/s, dt 1e-3, 10,000 steps | `|o - p|` = 0.5 m | `1e-12` m |
| `FreeSolid, TiltedRail_SlidesWithGSinTilt` (NEW, A59) | rail at 30 degrees from horizontal, no liquid, dt 1e-3 s, 1000 `BuildTry` + `CommitTry` | speed along the rail `n dt g sin 30 = 4.905 m/s`; distance off the rail line | `1e-12` relative; below `1e-12` m |
| `FreeSolid, OffOriginHinge_PendulumPeriod` (NEW, A59) | box 0.1 m, 1 kg, hinge 0.5 m above its centre about y, released at 5 degrees, no liquid, dt 1e-3 s | period `2 pi sqrt((I_cm + m l^2) / (m g l)) = 1.42322 s` (small angle; the 5 degree correction is 0.05 %); pivot distance drift over 10 periods | 0.5 %; below `1e-9` m |
| `FreeSolid, AxisAlignedGuides_Unchanged` (NEW, A59) | free box guided along z with a sideways steady force (1, 1, 0) N, no liquid, 100 steps | x, y velocity and every turn rate exactly 0; z as `FreeFall_GainsGTimesDt` | bytes; `1e-13` relative |
| `CheckSolidsCanRun, BadGuideOrSlipLength_IsBadInput` (NEW, A59, A53) | rail direction length 1.1; hinge axis NaN; slip length -1; slip length NaN; slip length +infinity | four `kBadInput` entries; +infinity accepted | exact |
| `ApplyViscousImpulses, LiquidForces_PartsOverStepDuration` (NEW, A60) | the hydrostatic scene of `PressurePart_EqualsApplyPressureToSolids` through `ApplySolvedMotionToSolids`, then `ApplyViscousImpulses` with hand values (tank `{(1e-3, 0, 0), 0}`, box `{(0, 2e-3, 0), (0, 0, 1e-4)}`) | `liquid_forces()` one per solid, ascending `solid_id`, tank included; pressure parts = pressure push / dt, viscous = hand values / dt | exact (the test's own expressions) |
| `ApplyPressureToSolids, LiquidForces_ViscousPartsZero` (NEW, A60) | the same scene through `ApplyPressureToSolids` | every viscous part `0.0`; pressure parts as above | bytes |
| `ApplyViscousImpulses, OtherSize_IsBadInput` (NEW, A60) | span one short | `kBadInput`, try unchanged | exact |
| `LiquidForce, BuoyancyAtRest_IsRhoGV` (NEW, A60; plan Task 36a) | liquid loop: tank 0.2 x 0.2 x 0.25 m, water 0.2 m deep at rest, fixed box 0.1 m cube centred at z 0.1, w 0.00625 (16 cells per side), one step of 0.002 s | pressure force z `rho g V = 9.81 N`; viscous parts exactly 0 (inviscid) | 2 %; exact |
| `BuildSolidsBuffers, IntoRunMemory_HandsOutExactlyTheEstimate` (NEW, A58) | 10 blocks, 1000 particles, 2 blocking materials, `run_on = kCpu`; `RunMemory` of exactly `ComputeRunMemoryBytes(estimate, counts, kCpu)` (every entry `kCpu` on a CPU instance) | `handed_out_bytes() == reserved_bytes()` | exact |
| `ApplySolvedMotionToSolids, FastAfterSolve_IsSolidMovesTooFar` (`{0, 0, -7}` m/s: limit `0.0125 / 7` s), `..., MotionOfOtherSize_IsBadInput`, `..., NotFinite_IsNotFinite`, `..., MapOfEarlierBuild_IsWrongTime` (NEW, A27) | as named | as named | exact; `1e-15` relative |
| `ApplyContactReactions, TwoHits_PushAndTurningPush` | two particles hit a driven box off centre | closed forms; equal particles placed symmetrically give zero turning push | `1e-14` relative |
| `BuildSolidsMemoryEstimate, CpuInstance_EveryRowOnCpu` (NEW, A17 batched-run buffers, plan Task 13) | `settings.run_on = kCpu` | every entry `where = kCpu` | exact |
| `WriteBlockPartialsRange, ThreeRangesGiveTheWholePartials` (NEW, A17, plan Task 28) | `CollideSolids` on three pieces of a 9-block grid, each piece's partials written into one whole CPU instance | equal bytes to `CollideSolids` on the whole grid; `ApplyContactReactions` then gives the same `ContactOnSolids` bytes | bytes |
| `WriteBlockPartialsRange, WholeBuffersTooSmall_IsBadInput` (NEW, A17, plan Task 28) | whole buffers with fewer blocks | `kBadInput` | exact |
| `CommitTry, ResultsFromTwoInstances_SameAsOne` (NEW, A17, plan Task 28) | `ApplyPressureToSolids` with one instance, `ApplyContactReactions` with a second | committed states equal the one-instance run's bytes | bytes |
| `ComputeContactBetweenMaterials, ...` | head on; separating; stick; slide (masses 2 and 6 kg: 6 N s normal, 1.5 N s sideways, as on `main`) | closed forms; momentum kept; energy not raised | `1e-14` relative |
| `CollideSolids, FlatFrontAlongAxis_SharesNormalSpeed` | two blocks, axis-aligned front | faces of the normal axis take the mass-weighted mean; others unchanged at zero friction | `1e-13` relative |
| `CollideSolids, SnowFaceTowardPlaten_TakesPlatenNormalSpeed` | set with `kNoMotionIntoSolid`, platen normal `{0, 0, -1}` moving down at 1 m/s, face at 0.2 m/s up | face velocity -1 m/s; push on the platen `mass * 1.2` N s upward; a leaving face and a `kParticlesOnly` set are unchanged | `1e-14` relative |
| `CollideSolids, ObliqueSolid_SidewaysSpeedKept` | solid normal `{0.6, 0, 0.8}`, face velocity along the surface | unchanged | `1e-14` |
| `CollideSolids, AnyCase_KeepsMomentumPerAxis` | 200 random particles in two sets | face momentum sums unchanged | `64 eps` |
| `CollideSolids, SetArraysDisagree_IsWrongMaterialSet` | a set's `CellsFromParticles` of another set; one set given twice | error, nothing written | exact |
| `CollideSolids, AnyBatchCut_SameBytes` | platen on snow beside liquid, blocks run as one range, as 2 and as 7 ranges | face velocities and per-solid pushes identical | bytes |
| `DriveWaves, UniformPull_MomentumIsMassTimesChange` | zone at rest pulled toward 1 m/s | `momentum_added`, `energy_added` closed forms | `1e-13` relative |
| `DriveWaves, AnyBatchCut_SameBytes` | the same zone as one range and as 5 ranges, block partials summed by the caller | totals identical | bytes |
| `BuildWavePaddlePath, Sine_KeysOnTheCurve` | stroke 0.2 m, period 2 s | every key on the sine; ramp applied | `1e-15` |
| `ApplyPlateSides, EachSide_GetsItsSide`, `..., InsideThickness_IsParticleInsidePlate`, `..., AlreadyGiven_IsKept` | as named | exact |
| `ComputeMinPlateThickness, AnyNormal_NoCellSeesBothSides` | 1,000 normals, plate at that thickness, random offsets | some layer of cells has 8 corners inside; no cell has corners on both sides | exact |
| `FindGridEdgesOpenToSky, PocketUnderLid_IsNotOpen` | open top, a box roof over missing blocks | marks by hand | exact |
| `FindGridEdgesOpenToSky, OpenTopNoSolid_EveryUpperEdgeMarked`, `..., ClosedTop_NothingMarked` (plan Task 19) | as named | as named | exact |
| `FindGridEdgesOpenToSky, OnGpuGrid_EqualsCpuGrid` (NEW, A17, label `gpu`) | the three cases, grid built on both processors from one list | marks equal; the only copies are the two box arrays | exact |
| `CheckSolidExchange, BrokenMap_IsExchangeNotBalanced` | one entry's area changed by 1 % | error; the unbroken case passes | exact error |
| `ComputeCutWork, UncutBlock_IsZero` | affine motion, no plate nearby | `cut_work = 0` | `1e-6` of `deformation_work` (the limit of `main`) |

**Scene tests listed here, run by the step module's plan** (they need a whole step; expected values are from
physics):

- `DrivenFlatSolid, SweptVolume_IsReceivedByLiquid`: paddle at 0.5 m/s into still liquid, open top.
  `ExchangeTotals::swept_volume` against the liquid's volume gain above the rest height: 1 %. **Known open
  physics: expected to fail** (7.6 % on `main`); the test is kept, labelled, not tuned and outside the gate
  (decision 4).
- `FreeBox, HalfDensity_FloatsHalfSubmerged`: 500 kg/m3 cube, 16 cells per side: draught within 2 % of half
  (old paper acceptance); linear and turning momentum of liquid plus solid change only by gravity and the
  domain sides, within `1e-9` relative per step.
- `FreeBox, OffCentreSplash_ReactionEqualsLiquidChange`: a jet hits a floating box off centre while it rests
  on the liquid: pressure push plus contact push on the box equals minus the change of liquid momentum
  (report 10.6).
- `WavePaddle, RegularWave_SpeedAndHeight`: wave height 2 % of depth, 3 wavelengths: phase speed within 5 % of
  `omega^2 = g k tanh(k depth)`; height within 10 %; reflection from the outlet below 10 %; volume drift below
  0.1 %; `flow_work` equals the gain of wave energy plus `energy_added` of the relaxation within 15 %
  (report 13.7).
- `Cut, PlateThroughBlock_SidesNeverMix`: springy block, driven plate; then the same with a restart from a
  saved state in the step after the first hit, with a rejected try before it, and with a split-and-merge pass
  on a liquid beside it: identical bytes to the uninterrupted run; no particle changes side; the plate's
  `driven_work` equals elastic energy gained plus `cut_work` plus kinetic energy within 5 %.
- `Cut, PlateLeaves_SidesRejoin`: after `ApplyPlateRemoval` the two sides share faces.
- `Platen, SnowBesideWater_SetsKeepTheirVelocities` (A8; amended 2026-11-14, A18): a platen presses a snow
  bed with water beside it; snow and water are two sets; no water face takes the snow's velocity except through
  `CollideSolids` steps 2 to 6; the snow faces the water presses on move in the same step's pressure solve
  (their velocity after equals `pressure::SolvedArrays::blocking_material_velocity_after_pressure`); momentum of
  water plus snow plus platen changes only by gravity, the domain and the platen's push.
- `FreeBox, FallsThroughGas_NotPushedByGas` (A10): a free box falls through a gas-filled box; its motion is
  free fall within `1e-13` relative while the gas flows around it (the gas sees `solid_volume_rate`; marker
  hits are not summed).
- `Batches, FloatingBox_WithinStepToleranceOfOnePiece` (renamed 2026-11-14, A15; was
  `Batches, FloatingBox_SameBytesAsOnePiece`; run by the cell resize and batches plan):
  `FreeBox, HalfDensity_FloatsHalfSubmerged` run in both batched modes agrees with the one-piece run within the
  step design 6.4 tolerances for every solid state and `ExchangeTotals`, and gives identical bytes for every cut
  of one mode.
- `FreeBox, HalfDensity_FloatsHalfSubmerged` and `FreeBox, OffCentreSplash_ReactionEqualsLiquidChange` run again
  through `TryStep` (dropped from rest, settling with step-duration control); every scene of plan Tasks 35 and
  36 on `RunOn::kGpu` (label `gpu`) against the CPU within section 6.
- `SpringyBlock, FifthDensity_FloatsAtDisplacedDepth` (added 2026-11-14, A18; the test decision 3 named as the
  one that decides): a springy cube of side 0.1 m and density 200 kg/m3 (0.2 of the water) set on still water,
  steps of 0.01 s (the step design's `max_step_duration`): its mean bottom height settles to 0.02 m below the
  still surface within half the smallest cell width; its vertical velocity amplitude over the last 0.2 s is no
  larger than over 0.2 to 0.4 s. Pressure's plan Task 49 shows the lagged form unstable at a tenth of this step.
- `TryStep, FiveMaterialsWithSnowAndFreeBox_Commits` (added 2026-11-14, A26, A27): the five-material scene with
  snow, a springy solid, a viscous liquid and a free box beside it commits; stage 11 calls
  `ApplySolvedMotionToSolids` and `ApplyViscousSolveToMaterials`. Snow is checked for not breaking only: melting of
  snow by liquid water is not modelled, so liquid-on-snow is not a validation target (owner note, 2026-11-14).
- `TryStep, FloatingBoxInSyrup_RestDepthAsInWater` (added 2026-11-14, A27): the half-density box in a viscous
  liquid settles at the draught it has in water (0.1 m within 2 %).

The solids plan runs `FreeBox, HalfDensity_FloatsHalfSubmerged`, `FreeBox, OffCentreSplash_ReactionEqualsLiquidChange`,
`DrivenFlatSolid, SweptVolume_IsReceivedByLiquid` and `Cut, PlateLeaves_SidesRejoin` itself on its hand-written
liquid loop (`fluidity_solids_liquid_loop`, `solids::testing::LiquidLoop`); the others above are handed to the
step plan (plan-decisions "Solids", "Tests the step plan must add"), and the split-and-merge part of
`Cut, PlateThroughBlock_SidesNeverMix` to the cell resize and batches plan.

**Reference and proofs:** none. Every expected value above is a closed form or a conservation law; the
exhaustive scan is a second implementation inside the tests. No Lean proof on `main` concerns this module.

**Comparison with the proof of concept (supporting only).** `AdvanceRigidBody3D` on the case of
`test_rigid_body_3d.cc:442` (asymmetric inertia, off-centre force), turned into Z-up: velocities and pose
within `1e-13` relative. `ResolveDistinctBodyContact3D` rows: within `1e-14`. The coupled turned box of
`liquid_substep_3d.cc:1806` against one solve with `FreeSolidPush` on a same-size grid and flat surface:
velocity after pressure within the solve limit.

**GPU (label `gpu`, compiled here, run by the owner):** every table row that has a `_gpu.cu` file, against
`RunOn::kCpu` within section 6, including `BuildBlockingMaterialFaces, OnGpu_EqualsCpu` and
`ApplyPressureToMaterials, OnGpu_EqualsCpu` (A18; entries exact) and `FindGridEdgesOpenToSky,
OnGpuGrid_EqualsCpuGrid` (A17), `BuildViscousMaterialFaces, OnGpu_EqualsCpu`, `ApplyViscousSolveToMaterials,
OnGpu_EqualsCpu`, `ToFreeSolidsForViscousSolve, OnGpu_EqualsCpu`, `ApplySolvedMotionToSolids, OnGpu_EqualsCpu`
(A26, A27; plan Task 24); `SameInputTwice_SameBytes` for `ComputeSolidCoverage`, `BuildMotionMap`,
`ApplyPressureToSolids`, `ApplyContactReactions`, `CollideSolids`; `ApplyContactReactions,
ParticlesRegrouped_SameBytes` (the order is by id, not by place). **Container without a GPU:** all shared
headers, `Solids`, `SolidsTry`, the path and dynamics tests, the CPU forms of `FindGridEdgesOpenToSky`, and
every CPU file, also built inside the CUDA configuration (label `no_gpu`); every operation with a `_gpu.cu`
(now also `FindGridEdgesOpenToSky`, `BuildBlockingMaterialFaces`, `ApplyPressureToMaterials`,
`BuildViscousMaterialFaces`, `ApplyViscousSolveToMaterials`, `ApplySolvedMotionToSolids`) has
`<Operation>, OnGpuWithoutAGpu_IsRunOnCallFailed` (plan-decisions "Applies to every plan").

## 8. Method paper section

`= Solids that move <sec-solids>` with `== Fixed, driven and free <sec-solid-motion>`, `== Solids on the
grid <sec-solid-coverage>`, `== One motion map and its transpose <sec-motion-map>`, `== Free solids
<sec-free-solids>`, `== Contact between materials <sec-collide-solids>`, `== The wave paddle and waves as a
forced system <sec-drive-waves>`, `== Cutting <sec-cutting>`, `== What is checked <sec-solid-checks>`.
Equations `<eq:driven-motion>`, `<eq:free-solid>`, `<eq:motion-map>`, `<eq:motion-map-transpose>`,
`<eq:material-contact>`, `<eq:plate-thickness>`, `<eq:cut-work>`. `<sec-motion-map>` also describes blocking
materials solved with the liquid (A18, plan Task 1): each blocking set's face mass enters the pressure solve on
the pieces it blocks (`dt A^2 / m`, referring to pressure's `<sec-coupled-solve>`), the solved face velocity is
written back to the set in the same step, and tangential motion stays the set's own (A8); it states the sign of
work in one sentence. Amended (A26, A27, plan Task 1): in a viscous scene the material band and the free solids'
six motions are unknowns of the viscous solve and the liquid sticks to them; the A8 sentence holds for an inviscid
liquid only. NEW (A59, A60, A53): `<sec-free-solids>` gains guides in general form (`xi = J q + xi_0`, rail and hinge
columns, the projection before pressure, the inverse inertia `J (J^T M J)^-1 J^T`, the hinge's rotation about its
pivot at commit; cite Batty 2007 [21] and Aanjaneya 2018); `<sec-motion-map>` gains the force of the liquid on every
solid read from the solve's terms (momentum exchange, Uhlmann 2005; torque point and how to move it) and one sentence
on the slip length a fixed or driven solid gives a viscous liquid.

Figures: `<fig-motion-map>` one partly blocked face piece: its blocked part, the four corner areas, the
corner velocities of a turning solid, and the same arrows reversed as pushes with their arms;
`<fig-free-solid-step>` a time line of one step: velocity before pressure, after pressure, the drift, the
contact push at the end; `<fig-driven-path>` a path with three keys and the steps cut at each key;
`<fig-corner-owners>` balls of snow particles, the corners they own, the blocked parts liquid sees;
`<fig-plate-sides>` a plate through a block: particle sides, face sides at the start and end pose, the faces
that changed side, the layer of cells that makes the thickness rule; `<fig-material-contact>` two sets on one
face: sampled velocities, the normal between their mass centres, stick and slide.

Pseudocode: `<proc-free-solid-step>` (5.3, four steps), `<proc-build-motion-map>` (5.5), `<proc-collide-solids>`
(5.7, five steps), `<proc-apply-plate-sides>`.

Sources from the old list: [21] Batty 2007 (the transpose from one work expression), [22] Ng 2009 (solid flow
through blocked area), [3] Hu 2018 (two sides of a thin solid; two-way coupling with a moving solid), [20]
Jacobsen 2012 (wave generation and relaxation zones), [19] Stomakhin 2013 (friction response), [18]
Stomakhin 2014 (several materials on a staggered grid). Not in the old list, marked "citation needed" and
not written from memory: the free-solid equations of motion; contact between two materials on a grid (the
old paper names Nairn without an entry).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `RigidBody3D`, "rigid body", "floating box" | a `free` solid: `SolidSettings` with `kFree`, `SolidState` | DECIDED words |
| `prescribed`, `TranslatingSlab3D` with keys, "animated obstacle" | `driven`, `DrivenPath`, `PathKey` | DECIDED word / NEW |
| "piston", `piston_velocity_x` | `wave_paddle`, a driven `kFlatSolid` | DECIDED |
| `principal_inertia`, body frame | `inertia_in_own_axes`, "own axes" | NEW (contact says "the solid's own axes") |
| `RigidOrientation3D`, `RigidBodyPose3D` | `Orientation`, `contact::SolidPose` | NEW |
| `force`, `torque` accumulators | removed; `steady_force` | NEW |
| `AdvanceRigidBody3D`, `RigidBodyStep3D drift` | `ComputeVelocityBeforePressure`, `CommitTry`, `contact::SolidMotion` | NEW |
| `FloatingBoxMotion::kVerticalGuide` ... | `FreeMotionLimits` | NEW |
| (none) | `GuideKind` (`kAxisLimits`, `kRail`, `kHinge`), `rail_direction`, `hinge_point`, `hinge_axis`, `AllowedMotions` ("allowed motions"; papers: generalized coordinates, `J`), `ComputeAllowedMotions`, `ComputeAllowedVelocity`, `ComputePoseAfterStep`, `InverseInertia::linear_per_turning_push`, `turning_per_push` | NEW (A59; the last three functions and fields plan decisions) |
| (none) | `LiquidForceOnSolid` (`pressure_force`, `pressure_torque`, `viscous_force`, `viscous_torque`), `SolidsTry::liquid_forces`, `ApplyViscousImpulses`, `ViscousPushOnSolid` ("viscous push"; never "impulse") | NEW (A60; `ViscousPushOnSolid` plan decision) |
| (none) | `SolidSettings::slip_length` ("slip length"; papers: Navier slip length), `SolidsTry::slip_length` | NEW (A53; the accessor a plan decision) |
| `RigidPressurePatch3D`, `wetted_area`, `world_centroid` | `MotionMapEntry`, `signed_corner_area` | NEW ("wetted" is banned) |
| `ComputeRigidBodyInwardWallFlux3D` | step 1 of 5.5; `pressure::FreeSolidPush` | DECIDED by pressure |
| `AccumulateRigidBodyPressure3D`, `oriented_loads` | `ApplyPressureToSolids`, `PressureOnSolids` | NEW |
| `impulse`, `angular impulse` | `push` (N s), `turning_push` (N m s) | `push` from pressure; `turning_push` NEW |
| `added_mass_kg`, pressure-response probe | removed (one solve) | - |
| `body_impulse`, `piston_impulse` | `ContactOnSolids`, `ApplyContactReactions` | NEW |
| `SampledSolidBoundary3D`, corner "owner" | `SolidCoverageValues`, `CornerOwner` | NEW; `owner` see below |
| `support_radius` (snow ball) | `particle_radius` | NEW (pressure uses the same word) |
| `StagePlasticSolidPressureLoads3D` | `ApplyPressureToMaterials` (A18: writes solved face velocities into each blocking set; no particle push) | NEW |
| `PlasticSolidPressureMobility3D` (the give `B M^-1 B^T`) | `BlockingMaterialFaces`, `pressure::BlockingMaterialFace` (lumped `dt A^2 / m`) | NEW (A18) |
| (none) | `BlockingSet` ("blocking set": one material set of blocking materials; glossary row), `PressureOnMaterials` | NEW (A18) |
| (none) | `ViscousMaterialFace`, `ViscousMaterialFaces`, `BuildViscousMaterialFaces`, `ToViscousMaterialFaces`, `ApplyViscousSolveToMaterials`; "material band" (the faces of a blocking set the viscous solve moves; glossary row) | NEW (A26) |
| (none) | `FreeSolidForViscousSolve`, `ToFreeSolidsForViscousSolve`, `FreeSolidMotion` (`velocity`, `turn_rate`), `ApplySolvedMotionToSolids`, `SolvedMotionReport` | NEW (A27; `SolvedMotionReport` plan decision) |
| `ContactBodySample3D`, `body_id` | a material set at one face; `MaterialSetFaces` | NEW (`body` is banned) |
| `ResolveDistinctBodyContact3D`, `effective_mass` | `ComputeContactBetweenMaterials`, `shared_mass` | NEW |
| stage `kBodyContact`, `kWaves` | `CollideSolids`, `DriveWaves` | DECIDED |
| snow face clamp at floor and platen | 5.7 step 1, `materials::AtSolidFaces::kNoMotionIntoSolid`, `kFaceStopReachInCells` | materials / NEW |
| `ElasticBladeSide3D`, `sides` list, "blade", "cutter" | `contact::PlateSide` on particles; "plate" | DECIDED |
| `BuildElasticBladeFaceTags3D` | `ComputeSolidsAtFaces` | NEW |
| `ClassifyInitialElasticBladeSide3D` (per scene) | `ApplyPlateSides` | NEW |
| `cut_work`, `deformation_work`, `grid_stress_work` | `CutWork { cut_work, deformation_work, face_force_work }` | NEW |
| `blade_work`, `piston_work` | `driven_work` | NEW |
| `SolidInteractionCheckpoint3D`, `accepted_step` | `SolidsSnapshot`, `state_version` | DECIDED words |
| `RigidBodyHistory3D::previous_pose` | removed: the try holds the start pose | - |
| `can_hold_liquid` (face tag, as the first designs wrote it) | `transfer::SolidsAtFaces::outside_solid` | Q8, recommended option |
| "state velocity", `velocity.state`, `state_velocity_change` | `velocity.for_next_step`, `SolidReaction::velocity_for_next_step_change` (`MaterialPushFields` removed, A18) | owner A1 |
| separate liquid and snow grids (`dry_solid_step_3d.cc`) | material sets: one `GridFromParticles` per set, `MaterialSetFaces` | owner A8; "material set" (decision 6) |
| `max_free_solid_count` (first draft of this design) | `kMaxFreeSolidCount`, `kMaxPlateCount` | NEW, public limits |

Words marked Verify or Open that this module met:

- `owner` (O5, Q5). `plastic_solid_boundary_3d.h:14` uses it for "the particle nearest to a corner".
  Decided (Q5, recommended option): `owner` is banned as an identifier in grid, transfer and pressure only.
  This module keeps `CornerOwner` and `corner_owner`, because the particle receives the corner's push.
- `outside_solid` (Q8, recommended option; was `can_hold_liquid`). VERIFIED in
  `liquid_wall_contact_3d.cc:674-680` and `elastic_blade_deformation_3d.cc:373`: false exactly when the face
  centre is inside the plate. Step 1 of 5.4 keeps that meaning for every analytic solid and does not extend it
  to blocking materials.
- `side_tag` (O2, Q3). Decided a: `contact::PlateSide`.
- `state` (O3, Q1). Decided by owner A1: `for_next_step` and `for_this_step`.
- `try` (Q10): `SolidsTry`, `BuildTry`, `CommitTry`, `RejectTry` follow the particles design.
- (replaced 2026-11-14, A31) `OrderScratch`, `order_scratch`, `TransferScratch`, "scratch": `OrderBuffers`,
  `order_buffers`, `transfer::TransferBuffers`, "working buffers".

## 10. Not ported

- The ten solid `AttemptWith...` methods and `AttemptFloatingBox`, `AttemptDynamicPistonImpl`
  (`liquid_substep_3d.cc:3663-3890`), with their per-scene preconditions.
- The measured pressure response: six probe solves, the averaged 6 x 6 matrix, the correction loop, the
  scalar versions and `added_mass_kg` (`liquid_substep_3d.cc:1806-2175`). One solve with `FreeSolidPush`.
- Every per-shape clipper: `VisitRectangularPressurePatches`, `VisitOrientedBoxPressurePatches`,
  `VisitStationaryTankPressurePatches`, `VisitFloatingBoxPressurePatches`, `MeasureWetPolygon`,
  `ComputeRigidBoxSubmergedVolume3D` (`rigid_body_3d.cc:135-710`). The box volume formula survives as a test
  oracle only.
- The vertical pressure-gradient torque of `AccumulateRigidBodyPressure3D` (`:770-776`): it has no partner in
  the flow, so it breaks the transpose. The floating test of section 7 measures what its absence costs.
- All of `wetted_piston_3d.cc`, `wetted_piston_math_3d.h`, `wetted_piston_3d_internal.h` and
  `native/cuda/wetted_piston_3d*`: enclosures, disk sections, interval sorting, the row-below rule.
- `PlasticSolidPressureMobility3D` as a second solver loop and `VerifyProjectedFlux` (its per-particle
  mobility is replaced by the lumped face term of 5.5a inside the one liquid solve; decision 3 as replaced by
  A18).
- `mixed_contact_3d.cc` from line 400: `FreezePlanarSnowPressureCell3D`, `SolvePlanarSnowPressureTrial3D`,
  `ApplyPlanarSnowPressure3D`, `AdvancePlanarSnowPressure3D`, `AttemptPlanarSnowPressureStep3D` and the frame
  driver; and the flat-front detection `ReconstructAxisAlignedMixedContactFace3D`,
  `ApplyPlanarMixedContactFaces3D`, `AdvancePlanarMixedContact3D` with its common hit time.
- `ContactFaceBodies3D` (the sparse table of four per face) and `E-CONTACT-BODY-CAPACITY`: sets are arrays.
- `SweepRigidBoxPoint3D`, `RigidBodySweepDistanceBound3D` (contact replaced them).
- `ResolveElasticCutterScene3D`, `AdvanceElasticBladeFrame3D`, `StepElasticBladeGrid3D` as a separate solver
  with its own grid, fixed-point sums and step loop; the neo-Hookean response moves to materials.
- `SolidInteractionCheckpoint3D` codecs and the FLSI versions 1 to 3 (the I/O module writes schema v2 only).
- The dry step's clamp as a branch on the material type and on world `y` (`dry_solid_step_3d.cc:367-378`):
  the rule is kept (5.7 step 1) but reads `materials::AtSolidFaces` and the solid's own normal.
- Solid meshes, contact between two solids, fracture, a finite plate, several plates.
- The "degrades to one velocity for all materials" branch of the first draft (owner A8: material sets only).

## 11. Decisions

Earlier questions this design depends on, each with the answer it now uses:

- Q1 (`state` kept). Decision: owner answer A1: `for_next_step` and `for_this_step`; this design writes
  `velocity.for_next_step`, `velocity_for_next_step` and `velocity_for_next_step_change`.
- Q3 a (`PlateSide`). Decision: a, recommended option (owner's standing instruction, 2026-11-14).
- Q5 (see section 9). Decision: `owner` banned as an identifier in grid, transfer and pressure; `CornerOwner`
  stays here. Recommended option (owner's standing instruction, 2026-11-14).
- Q8 (`can_hold_liquid` kept). Decision: `outside_solid`. Recommended option (owner's standing instruction,
  2026-11-14).
- Q10 (`CommitTry`, `RejectTry`). Decision: as written. Recommended option (owner's standing instruction,
  2026-11-14).
- Q16 A (any solve that returns a cell pressure serves `ApplyPressureToSolids`). Decision: owner answers A5 and
  A6: viscosity runs `SolveViscosityAndPressure`, pressure runs `SolvePressure` for the other scenes;
  `ApplyPressureToSolids` reads the cell pressure of whichever ran.
- Q26 (the flood, filled here). Decision: the proposal. Recommended option (owner's standing instruction,
  2026-11-14).
- Q28 a. Decision: a, viscous liquid always sticks to solids. Recommended option (owner's standing
  instruction, 2026-11-14). Extended by owner answers A26 and A27 (decision 15): also to blocking materials and
  free solids. Refined by A53 (decision 19): sticking is the default; fixed and driven solids may slip by a set length.
- Q35 a. Decision: a. Recommended option (owner's standing instruction, 2026-11-14).
- Q36 (rest gap). Decision: the rest gap. Recommended option (owner's standing instruction, 2026-11-14).
- Q37 (unbounded plate: this design keeps it). Decision: unbounded plate, one per scene (`kMaxPlateCount`).
  Recommended option (owner's standing instruction, 2026-11-14); see decision 2.
- Q38 (reject, and 2 sides: the thickness rule of 5.9 makes the reject unreachable for a plate). Decision:
  reject and 2. Recommended option (owner's standing instruction, 2026-11-14).
- Q46 (yes). Decision: yes, `CollideSolids` owns contact between two materials. Recommended option (owner's
  standing instruction, 2026-11-14).
- Q47 (open sea stays in pressure; `DriveWaves` here calls it). Decision: as written. Recommended option
  (owner's standing instruction, 2026-11-14).
- Q51. Decision: as listed. Recommended option (owner's standing instruction, 2026-11-14).
- From the materials design: its question 5 (`AtSolidFaces` kept as a field of the material; 5.7 step 1
  applies it), and its three needs from this module, all provided (table in section 4). Decision: as written,
  recommended option (owner's standing instruction, 2026-11-14). One part differs from what materials expects:
  the pressure push reaches snow and springy particles one step late (decision 3). **Replaced 2026-11-14 by
  owner answer A18 (decision 9):** the blocking sets' faces are solved with the liquid and written back in the
  same step; no push is returned to particles.
- Q41 (`BuildGrid` on the CPU, also for GPU runs; "no copy between dependent stages"). **Replaced 2026-11-14 by
  owner answer A17 (decision 10):** the grid is built on the GPU in a GPU run and no block id comes down.

**1. Do different materials share one grid velocity?** The six designs sum every material onto the same
faces, so water, snow and a springy solid within 1.5 cell widths move as one, pressure's mixture density
counts snow mass, and `CollideSolids` has nothing to separate. `main` keeps liquid and snow on separate
grids.

```cpp
ParticlesToGrid(inputs_of_all_materials, ...);                         // a) one velocity for all
for (set : material_sets) ParticlesToGrid(inputs_of(set), ...);        // b) one grid velocity per material set
CollideSolids({grid, blocks, material_sets}, buffers);                 //    then the pair rule between sets
```

a) One velocity: no change to transfer; materials stick where they touch; `CollideSolids` runs only its rule
at solids, and only in scenes without liquid (5.7); a blocking material is both mixed with the liquid and
seen by it as a solid. b) Material sets:
liquids and gas are one set, each snow, sand or springy material (or a group the artist chooses) another;
transfer's stages take a material subset (section 4); costs one `GridFromParticles` and two snapshots per
extra set (about 14 kB and 2 x 9 kB per cell block). The pair rule is ported, but steps 2 and 3 of 5.7 (the
sampled velocity and the normal) are new, because `main` has them only for flat axis-aligned fronts.
Recommendation: **b**. This design is written for b and degrades to a when one set is passed. Affects:
transfer, pressure, materials, step.

Decision: b, owner answer A8. Materials that do not mix are separate sets with their own face sums;
`CollideSolids` applies the contact rule at shared faces; transfer's stages and `ComputeCellsFromParticles`
take `material_is_counted`. The liquids form one set; gas markers are a set of their own (A10), not passed to
`CollideSolids`; each blocking material is its own set unless the scene groups them. The "degrades to a"
branch is removed. Transfer states the cost (figures replaced by plan-decisions "Transfer", decision 12):
48,576 + 512 x
ingredients bytes per extra set per cell block (40,416 + 512 x ingredients for a set no solve writes).

**2. What is a cut after the plate has gone, and may a plate be finite?** As ported, the plate is unbounded,
one per scene, and separates the whole plane; when it leaves, the sides rejoin.

```cpp
ApplyPlateRemoval(plate_id, materials);                 // a) sides rejoin (this design)
// b) on removal each side becomes its own material set: needs question 1 b and a set index per particle
// c) contact::Plate gets a half size; a side is given when a particle enters its footprint; the cut is the
//    area the footprint swept, stored in SolidsSnapshot; kMaxSidesAtFace grows for several plates
```

Recommendation: **a** now; b when question 1 is answered b; c is a new method and belongs with the
experiments. Affects: contact, transfer, particles, I/O.

Decision: a, one unbounded plate per scene whose sides rejoin when it leaves. Recommended option (owner's
standing instruction, 2026-11-14; `changes-to-apply.md` "08 Solids"). b and c are not built.

**3. Pressure against a blocking material: with a lag, or inside the solve?** `main` adds the material's
"give" to the liquid solve and steps the snow after the liquid in the same try. Pressure's interface takes
only free solids. Options: a) as designed: the liquid sees the material as a driven solid, the push reaches
the owner particles after `MoveParticles` and acts in the next step (the old paper's "one-sub-step lag", a
known risk for light materials); b) pressure accepts a general term, for example
`std::span<const MaterialPush> { equation_index; particle_slot; double area_per_mass_weight; }`, and applies
`dt * B M^-1 B^T` like the free-solid term; c) with question 1 b, use the material set's own face as the
sample: the push becomes a face force of that set in the same step and the give is one number per face
(`dt * blocked_area^2 / face mass`). Recommendation: **c** if 1 b is chosen, else a, with the floating
springy cube of the old paper as the test that decides. Affects: pressure, transfer, materials.

Decision: a, the push is returned one step late (`ApplyPressureToMaterials` after `MoveParticles`), as
`changes-to-apply.md` "08 Solids" records the recommended option (owner's standing instruction, 2026-11-14).
Pressure's interface is unchanged (free solids only). The floating springy cube remains the test that would
reopen it.

**Replaced 2026-11-14** by owner answer A18 and `literature-coupling.md` 3 (a lagged force on a material lighter
than the water it displaces is unconditionally unstable by Banks' condition): option c, see decision 9.

**9. Blocking materials inside the liquid solve (A18).** Decision: option c of decision 3. Solids builds
`pressure::BlockingMaterialFace` entries with `BuildBlockingMaterialFaces` (each blocking set's own face mass and
face velocity on every face piece where it blocks liquid) in stage `BuildPressureEquations`, before the liquid's
`BuildPressureEquations`; after the solve `ApplyPressureToMaterials` writes the solved face velocities into each
set's `GridFromParticles::velocity`, before `SaveGridAfterPressure`. `MaterialPushFields` and the particle push
are removed; `ApplyPressureToSolids` stores no push owed to particles (`push_to_materials` stays a report). A
piece whose set has no face mass is not an entry (its blocked part stays one-way in `solid_volume_rate`); a piece
touching no `kLiquid` cell is not an entry. Only `SolvePressure` solves equations with these terms. **Replaced
2026-11-14 by owner answer A24 (decision 14):** "Only `SolvePressure`" no longer holds; viscosity's joint solve
also solves them. Source: owner
answer A18; `plan-decisions.md` "Coupling inside the pressure solve (A18)"; solids plan decision 9, Tasks 13, 21,
23, 24. The step plan gains `SpringyBlock, FifthDensity_FloatsAtDisplacedDepth`.

**10. `FindGridEdgesOpenToSky` on a GPU grid (A17).** Decision: occupancy of the biggest-level box lattice in
one GPU pass (1 B per box, read back), the solid test and flood on the CPU as before, the face marks in one GPU
pass from the reached boxes (1 B per box, sent); no block id comes down. Replaces "runs on the CPU from the block
ids" (sections 3.5, 6). Source: owner answer A17; `plan-decisions.md` "Batched solves (A15)", "A17 follow-ups";
solids plan Task 19.

**11. Batched solves (A15).** Decision: solids keeps every signature. `BuildMotionMap`,
`BuildBlockingMaterialFaces`, `ApplyPressureToSolids`, `ApplyPressureToMaterials` and `CheckSolidExchange` run
once on the whole grid in all three run modes (`kOnePiece`, `kBatched`, `kBatchedWithSolvesInPieces`), because
geometry, coverage, the motion map and the material faces stay whole on the run's processor; no solids row moves
to the CPU. Byte identity between modes is replaced by agreement within the step tolerances:
`Batches, FloatingBox_SameBytesAsOnePiece` becomes `Batches, FloatingBox_WithinStepToleranceOfOnePiece`;
`Batches, FreeBoxCuts_SamePushesAndReactions` (cuts of one mode) stays. Source: owner answer A15;
`plan-decisions.md` "Batched solves (A15)" (replaces solids design section 6's "like the solves"). **Partly
replaced 2026-11-14 by decision 13:** "no solids row moves to the CPU" holds for the whole and loaded instances
only; stage 17 has its own CPU instance in both batched modes, and `FindGridEdgesOpenToSky` runs in the step's
`run_whole_grid` part.

**12. Decisions the solids plan made beyond the design** (`plan-decisions.md` "Solids"; recorded here as the
contract now stands): `FreeSolidPush::solid_index` and `ToFreeSolidInverseInertia` order; `BuildSolidsBuffers`
takes `max_cell_blocks`; `BuildSolidCoverageValues`; the sign of work (positive when a paddle pushes liquid;
replaces `flow_work = -dt * ...` of the earlier 3.6); `try_id = (state_version << 32) | tries_since_commit` and
`CommitTry` of a stale try is `kWrongTime`; `ApplyPressureToSolids` can fail `kSolidMovesTooFar` and the step
retries shorter; `MotionMap` is a move-only view; `CollideSolids` takes at most `blocking_materials.size() + 1`
sets (`kBuffersTooSmall`) and runs pairs in `material_sets` order; `DriveWaves` has a private
`drive_waves_gpu.cu`; `CheckSolidExchange` reads the liquid side from face pieces, counts a non-liquid cell's
pressure as 0, fails the no-penetration check with `kExchangeNotBalanced`, and reports `particles_inside_solid`
as 0 or 1; `Solids` owns copies of its settings; test libraries `fluidity_solids_test_scenes`
(`solids::testing::SceneSolids`) and `fluidity_solids_liquid_loop` (`solids::testing::LiquidLoop`); memory estimate entries
use `sizeof` and `MemoryEstimateEntry::where`; transfer's per-set memory is 59,856 + 512 x ingredients bytes per block
(48,336 + 512 x ingredients with no solve), replacing the figures in decision 1.

**13. Solids buffers in batched runs (A17).** Owner principle A17 (a run is all CPU or all GPU; in a batched GPU
run only what does not fit stays in CPU memory, moved in planned, counted copies). Each entry: plan decision,
2026-11-14 (buffers in batched runs); `plan-decisions.md` "Buffers in batched runs", "Copies counted"; step plan
Appendix A rows M9 to M11; solids plan Tasks 13, 28.
- Several `SolidsBuffers` instances from one `Solids` (whole at zero particles, loaded at `max_loaded_particles`
  and `max_loaded_blocks`, CPU stage-17 at the whole particle limit and `max_cell_blocks`); `BuildSolidsBuffers`
  keeps its signature and with `settings.run_on = kCpu` every entry is `where = kCpu`; no instance points into
  another (3.6, section 6).
- Stage 17 (`ApplyContactReactions`, `ComputeCutWork`) runs on the CPU in a batched run, with the stage-17
  instance (section 4 "Step", section 6).
- NEW `WriteBlockPartialsRange` carries each range's `CollideSolids` block partials into the stage-17 instance,
  called by `batches::ReadBatch` after pass 1, counted in `BatchReport` (3.7, 5.6).
- `ApplyPressureToSolids` and `ApplyContactReactions` copy their per-solid results into `SolidsTry`;
  `CommitTry` reads results of stages 12 and 17 from two instances; `SolidsTry` keeps no pointer into a
  `SolidsBuffers` (3.4).
- `FindGridEdgesOpenToSky` floods the whole grid, so the step calls it from a new `run_whole_grid` entry of
  `kFindSolidSurfaces` at `kWholeGridOnProcessor`, right after pass 1 (one piece: right after stage 3's `run`)
  (3.5, section 4).
- Per-solid and per-set results (entries `where = kCpu`) and `FindGridEdgesOpenToSky`'s box marks are "small
  records", not counted in the copy counts (planned cost; `plan-decisions.md` "Copies counted").

**14. Working buffers, one form (A31).** Decision: `SolidsBuffers` holds `OrderBuffers order_buffers`; every base
order, prefix sum and grouping has one form, which takes them; the transfer stages the liquid loop calls take one
`transfer::TransferBuffers`. "Working buffers" replaces the old word. Source: owner answer A31 (2026-11-14).

**15. Processors of the arguments (A30).** Decision: arrays and buffers passed together live on one processor, a
documented precondition checked by a debug assertion; `ToSolidsFailure` maps no processor mismatch
(`BufferError` has none). Source: owner answer A30 (2026-11-14).

**16. Out of memory (reconcile item 25).** Decision: `SolidsError::kOutOfMemory` (magnitude = bytes asked),
returned only by `BuildSolidsBuffers`, replaces "a failed allocation is `kRunOnCallFailed`"; `ToSolidsFailure` maps
`BufferError::kOutOfMemory` to it. Source: reconcile-todo item 25 (2026-11-14).

**14. Blocking materials beside a viscous liquid (A24).** Decision: `viscosity::SolveViscosityAndPressure`
carries the blocking-material face terms as `SolvePressure` does, so `BuildBlockingMaterialFaces` and
`ApplyPressureToMaterials` serve stage 11 and stage 12 alike: stage 10 builds the faces whatever stage follows;
stage 11 writes `viscosity::SolvedCouplingArrays::blocking_material_velocity_after_pressure` (the step-owned span),
then `ApplyPressureToSolids`, then `ApplyPressureToMaterials`. Snow beside a viscous liquid is no longer excluded
(`kSnowNotSupported`, `has_snow` and `kViscousLiquidWithSnow` removed) and a springy solid beside a viscous liquid
runs (options A and B of reconcile item 4 withdrawn, nothing added). Free solids beside a viscous liquid stay
excluded (`kFreeSolidNotSupported`, `kViscousLiquidWithFreeSolid`). Replaces the "Viscosity" text of section 4
and the Open item "A springy solid beside a viscous liquid". Source: owner answer A24 (option C of reconcile item
4); `plan-decisions.md` "Viscous liquids with blocking materials and sealed gas (A24)", "A24: step, gas, solids".
**Replaced 2026-11-14 by owner answers A26 and A27 (decision 15):** "the face velocity is eliminated exactly" and
"the liquid slides" (no tangential stick) in the joint solve; "free solids beside a viscous liquid stay excluded";
and stage 11's calls of `ApplyPressureToSolids` and `ApplyPressureToMaterials`. Stage 12 keeps this decision's text.

**15. A viscous liquid sticks, and free solids in the viscous solve (A26, A27).** Decision: in a viscous scene
(stage 11) the faces of each blocking set within 1.5 cell widths of a liquid cell centre are velocity unknowns of
viscosity's joint solve and the liquid sticks to them (A26: coupled tangentially as well as normally, not a
contact rule after the solve); each free solid's six motions are unknowns too, with the pressure part from the same
`FreeSolidPush` entries and the liquid sticking at its surface (A27). Solids provides `BuildViscousMaterialFaces`,
`ToViscousMaterialFaces`, `ApplyViscousSolveToMaterials` (3.6b), `FreeSolidForViscousSolve`,
`ToFreeSolidsForViscousSolve`, `FreeSolidMotion`, `ApplySolvedMotionToSolids` (3.6); stage 11 calls
`ApplySolvedMotionToSolids` and `ApplyViscousSolveToMaterials` in place of `ApplyPressureToSolids` and
`ApplyPressureToMaterials`; stage 12 is unchanged (an inviscid liquid slides, material faces are eliminated, free
solids are the rank-one term). No combination of materials is refused because of viscosity
(`kFreeSolidNotSupported`, `kViscousLiquidWithFreeSolid` removed). Plan decisions (solids plan Tasks 23a, 23c):
the band uses `grid::kWeightCurveReach` (solids does not include viscosity) and the width of the piece's smaller
cell; every free solid of the try is listed, also one away from liquid; `SolvedMotionReport`,
`SolidsTry::solved_motion()` and `SolidsTry::pressure_on_solids()`; a motion a guide forbids keeps its value before
pressure after the solve, because `FreeSolidForViscousSolve` carries no `FreeMotionLimits` (listed for the owner:
viscosity should hold those motions fixed). **Replaced (A59, decision 17).** Snow has no special case: melting of snow by liquid water is not
modelled, so liquid-on-snow runs mechanically and is not a validation target; analytic tests use springy solids and
free boxes (owner note, 2026-11-14). Replaces the A24 text of 5.5a item 2 and of section 4 "Viscosity", and the
exclusion in decision 14. Source: owner answers A26, A27; `plan-decisions.md` "A viscous liquid sticks, and free
solids in the viscous solve (A26, A27)" (which replaces the A24 "Material face", "Viscous stress at a
liquid-material face" and "Kept: `kFreeSolidNotSupported`" bullets); solids plan decision 10, Tasks 23, 23a, 23b,
23c, 24, 33.

**17. Guided free solids in general form (A59).** Decision: `GuideKind { kAxisLimits, kRail, kHinge }` and
`FreeMotionLimits::{guide_kind, rail_direction, hinge_point, hinge_axis}` in the world frame; `AllowedMotions` and
`ComputeAllowedMotions` in the shared `guide_math.h`; `FreeSolidForViscousSolve::allowed_motions` replaces `limits`
(504 B); the velocity before pressure is projected onto the allowed motions; `InverseInertia` carries the four blocks
of `J (J^T M J)^-1 J^T` (from pressure's `ComputeGuidedInverseMass`); `ToFreeSolidInertia` gives pressure one
`pressure::FreeSolidInertia` per free solid (replacing `ToFreeSolidInverseInertia`; pressure's term becomes `(K J)(J^T M
J)^-1 (K J)^T` and `ApplyPressureToSolids` uses `pressure::ComputeGuidedSolidMotion`); a hinged pose is rotated about its pivot at commit, a rail pose moved along `d` only.
**Replaces** decision 15's "a motion a guide forbids keeps its value before pressure after the solve, because
`FreeSolidForViscousSolve` carries no `FreeMotionLimits`" and solids plan Task 23c's step 2 and decision 4. Plan
decisions: `ComputeAllowedVelocity` and `ComputePoseAfterStep` (names and the selection kept bitwise for axis
limits); `ComputeInverseInertia` gains `origin`. The pressure carrier (`FreeSolidInertia`, `ComputeGuidedSolidMotion`) is
pressure's declaration, applied here. Source: owner answer A59; `plan-decisions.md` "Owner answers A40-A64 applied" item 5; `literature-practice.md`
topic 3.

**18. The force of the liquid on every solid (A60).** Decision: `LiquidForceOnSolid` (header `liquid_force.h`),
`SolidsTry::liquid_forces()` filled by `ApplyPressureToSolids` (viscous 0) and `ApplySolvedMotionToSolids` (which
is followed by `ApplyViscousImpulses`, which stores viscosity's `viscous_push_on_solids`); pressure part from the motion map entries of every solid, viscous part
from viscosity; mean over the step, world axes, torque about the step-start origin; the buoyancy test here (plan Task
36a), the other three in viscosity. Plan decision: `ViscousPushOnSolid` is declared here (viscosity uses solids types,
not the reverse). Source: owner answer A60; `plan-decisions.md` "Owner answers A40-A64 applied" item 6.

**19. Slip length on fixed and driven solids (A53).** Decision: `SolidSettings::slip_length` (m, 0 default, +infinity
free slip), checked here (negative or NaN: `kBadInput`), passed to viscosity through `SolidsTry::slip_length()` (0
for free solids). Replaces in part Q28 a's "no setting". Source: owner answer A53; `plan-decisions.md` "Owner answers
A40-A64 applied" item 7.

**20. Buffers from the run's reserved memory (A58).** Decision: `BuildSolidsBuffers(settings, max_particle_count,
max_cell_blocks, RunMemory& run_memory)` and `BuildSolidCoverageValues(run_on, max_cell_blocks, RunMemory&
run_memory)`; the stage-17 CPU instance takes the run's `cpu_run_memory`; one estimate entry per array; estimate test
`BuildSolidsBuffers, IntoRunMemory_HandsOutExactlyTheEstimate`. Source: owner answer A58; `plan-decisions.md` "Owner
answers A40-A64 applied" item 3 and its same-day correction.

**21. `kOutOfMemory` last (reconcile item 31).** Decision: `SolidsError::kOutOfMemory` already stands last (18
values); it is returned only by the building functions and also maps `kRunMemoryTooSmall`; the plan's `ToName` test
counts 18. Source: `plan-decisions.md` "Owner answers A40-A64 applied" item 11.

**22. Particle size follows the cell (A65).** Every liquid cell holds about eight particles of its own size: a
particle's `size_level` is its cell's level, its rest volume `(w/2)³`, so a cell k levels coarser than the smallest holds 8 particles each of
`8^k` times the smallest particle's volume. Decision: `MaterialTheLiquidCannotEnter::particle_radius` stays one
radius per blocking material, because blocking materials (snow, springy solids, sand) keep their count and are seeded
and kept at `smallest_size_level` (A66: `BuildStartCells` forces that level wherever such a region lies; cell
resize never coarsens their cells); I/O's value reads `w_min / 4` (replaced: "`particle_spacing / 2`", a start lattice
that no longer exists). Solids reads no liquid particle size. Source: `plan-decisions.md` "Owner answers A65–A68
applied", A65, A66.

**23. Other answers of the A65–A68 pass.** A66: the start cells beside a solid are smallest-level (the solid term of
the cell size rule with the centre distance minus half the diagonal), so solids sees the same cells at a solid as in
a later step. A67: a split child inside a solid is contact's `kStartedInsideSolid` at the next `MoveParticles`; no
solids change. A68: a column cut is later; solids' per-block partials and fixed order are unchanged by it. Source:
`plan-decisions.md` A66, A67, A68.

**Open** (left open by the plans; not decided here):

- `ComputeAllowedVelocity` and `ComputeInverseInertia` (shared) call pressure's shared guide math; solids comes after
  pressure, so this is an ordinary dependency, but `solids/guide_math.h` must stay header-only for pressure to include
  it (pressure Task 47a after solids Task 5a).

- Stage `FindSolidSurfaces`' range part calls `BuildTry` and the contact checks once per try in one piece but per
  batch in a batched run; the step builds the solids try only in the first call of a try (step plan Task 33a), and
  the batches plan's guard test on stage order covers it. Whether those calls move to stage 2 is the owner's
  choice (`plan-decisions.md` "Planned cost shown to the user, and buffers in batched runs", open point 3).

- The blocking material's own volumetric stiffness inside the liquid solve (half of the volumetric response in
  Fang et al. 2020): not built; it needs materials to drop the volumetric part of `ApplyInternalForce` and
  material cells to get rows (`plan-decisions.md` A18 "Open", owner question).
- If a scene ever becomes GPU-limited in `kBatchedWithSolvesInPieces`, geometry is the next row to stream, which
  needs `BuildMotionMap`, `BuildBlockingMaterialFaces` and `CheckSolidExchange` to run batch by batch
  (`plan-decisions.md` "Batched solves (A15)", open point 3).

**4. The piston receiving volume.** The swept-volume test (section 7) fails on `main` at 7.6 % against 1 %
and nothing here fixes it: a surface cell is pushed through its whole blocked area. Options: a) keep the
test, labelled as known open physics, excluded from the gate; b) loosen the limit to what is measured;
c) push only through the liquid-covered part of the blocked area (needs `liquid_covered_area` per face
piece from pressure, and its own qualification). Recommendation: **a** until the step runs the paddle
scene; then decide between a and c with numbers.

Decision: a, the swept-volume test is kept as a known-open test, labelled and outside the gate. Recommended
option (owner's standing instruction, 2026-11-14).

**5. Two additions to the pressure design** (section 4): `ComputeBlockedCornerAreas` and
`ComputeEquationIndexes`. Alternative for the second: `FreeSolidPush` holds a `grid::CellPlaceIndex` and
pressure maps it. Recommendation: add both as written.

Decision: both added as written; they are in pressure's section 3 (`geometry_clip_math.h`,
`pressure_equations.h`). Recommended option (owner's standing instruction, 2026-11-14).

**6. Names to confirm.** `SolidMotionKind` with one `SolidSettings` instead of the types `DrivenFlatSolid`,
`DrivenPlate`, `FreeBox` of the naming review (one description, no type per pairing); `push` and
`turning_push` for N s and N m s (alternative: `impulse`, `angular_impulse` with the plain meaning in the
header); `MotionMap`, `MotionMapEntry`, `signed_corner_area`; `CornerOwner`; `inertia_in_own_axes`;
`Orientation`; `PathKey`, `angular_velocity_until_next`, `AfterPath`; `FreeMotionLimits`;
`MaterialTheLiquidCannotEnter` and `blocking_materials`; "material set" (not "group", which is the particles'
word, nor "body"); `shared_mass`; `driven_work`, `swept_volume`, `flow_work`; `ApplyPlateRemoval`;
`FindGridEdgesOpenToSky`; `SolidsSummary`; `ExchangeTotals`.

Decision: as listed (`push`, `turning_push`, not `impulse`). Recommended option (owner's standing instruction,
2026-11-14).

**7. Numbers to confirm.** `max_travel_in_cells = 0.5` (from `main`); `kSolidReachInCells = 2`;
`kCoverageCapInCells = 2`; `max_free_solid_count = 16`; exchange tolerance `1e-10` relative; the plate
thickness factor (2 to 2.74 cell widths) for a plate in liquid or gas; `keys_per_period = 32` for the paddle.
The two `SolidVelocitySample` arrays cost 39 kB per cell block; storing them only for blocks near a solid is
a change to transfer worth deciding with the sizing of the four scenes.

Decision: as listed, with the free-solid limit now the public constant `kMaxFreeSolidCount = 16` and the plate
limit `kMaxPlateCount = 1` (`changes-to-apply.md`, from the Blender add-on); both `SolidVelocitySample` arrays
are kept for every block as transfer declares them. Recommended option (owner's standing instruction,
2026-11-14).

**8. Who calls `DriveWaves`.** `shared-interface.md` (c) has the step call `ApplyOpenSeaRelaxation` directly;
the task of this module gives it the stage. Proposed: the step calls `solids::DriveWaves`, which calls
pressure and measures. Alternative: the measurement moves to pressure and this module has no stage function.

Decision: the step calls `solids::DriveWaves` in stage `DriveWaves`, which calls
`pressure::ApplyOpenSeaRelaxation` and measures. Recommended option (owner's standing instruction,
2026-11-14).

**Decision (A74, 2026-11-15): round solids.** Contact now has `kSphere` and `kCylinder` (`contact::Sphere`,
`contact::Cylinder`, `half_size` `{r, r, r}` / `{r, r, half_height}`). Decision: `SolidSettings::shape` may name
either for fixed, driven and free solids; faces, coverage, the motion map, `BuildContactSolids` and the reaction sums
use contact's distance, normal and velocity unchanged, so no formula here changes. `BuildTry`'s too-close check uses a
free sphere's centre distance minus `r`, and a free cylinder's `half_size` box corners; travel radius is
`Length(half_size)` for every bounded shape. A spin about a round solid's symmetry axis is not limited by
`kMaxTurnPerStep` (contact P17), so a driven roller may spin fast. Free-solid inertia stays a setting (I/O derives it
from mass and shape). New tests: `ComputeSolidCoverage, Sphere_BlockedAreaConverges`, `BuildTry,
FreeSphereHalfACellFromWall_IsSolidsTooClose`; the viscosity design's settling sphere is a free `kSphere`. Source:
owner answer A74; `plan-decisions.md` "Sphere and cylinder in contact (A74)".
