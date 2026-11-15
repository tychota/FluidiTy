# Sand module design

Status: decisions applied 2026-11-14; amended 2026-11-14 for A27 (decision 7). Module 13 of the build order (Sand), after the Blender add-on (owner
A9: "important but not mandatory"). Written from the sand parts of `07-materials-design.md` (moved here
unchanged in method) and `changes-to-apply.md` section 15.

## 1. Purpose

Sand is a material whose grains hold a slope by friction among themselves, come apart when pulled, and slide
with friction on solids. It is built as one more material kind on the materials module's single route: it
remembers a springy deformation and a permanent volume change like snow, but its allowed region is a cone
(friction) with a tip (cohesion), and its deformation after hitting a solid follows the velocity the contact
left. Its known defect, a heap whose rest angle keeps creeping down, is investigated here with an eight-step
diagnosis written as tests; the module is done when all eight pass, or when the owner decides otherwise.

Owns: `MaterialKind::kSand` and everything that depends on it: the sand response and return
(`ComputeSandResponse`), its limits (`friction_angle`, `cohesion`), its sound speed bound, the split of a sand
particle's step at a hit (`ComputeVelocityChangeAfterHit`), the `kParticlesOnly` default for sand faces at
solids, the sand reference scene and preset, the eight-step diagnosis, and the sand additions to step, I/O,
render output, viscosity and the add-on listed in section 4.

Does not own: the materials route, table, support rule, work identity and snow (materials); the
contact sweep and slip response (contact); the face rule at solids and blocking materials (solids); the
same-side sums and filled faces (transfer); the step loop (step). Sand adds to these through the extension
points of materials 5.9 and the items marked "to add in the sand module" in section 4; it changes none of
their existing behaviour.

## 2. What exists today

| Piece on `main` | Where | What it does | What is wrong |
|---|---|---|---|
| Sand return | `native/include/fluidity/base/m5_constitutive_3d.h:14-33, 84-157` | In the frame of the three stretches: takes logs, moves a pulled-apart state to the cone tip, shrinks the shape part at fixed sum. Returns force per area along the three directions. Output untouched on bad input. **Maths ported unchanged.** | Milestone code in the file name; textbook words as identifiers (`GranularParameters3D`, `cohesion_log_strain`, `deviator`); "representable" checks (naming review A9). |
| Sand sound bound | `base/solid_acoustic_3d.h:39-92` | Largest row sum of the scaled Hencky stiffness blocks over rest density; uses `expm1` for the pair terms. **Ported unchanged.** | None beyond names. |
| Sand hitting the floor | `native/src/dry_solid_step_3d.cc:134-156, 513-527` (`ProjectStationaryFloorVelocityGradient3D`) | Deformation advanced with `L` until the hit, then with the derivative of the friction response. Fitted change cleared after any hit (`:570`). | Hard-wired `{0,1,0}` normal and a floor that does not move. |
| Sand faces at solids | `dry_solid_step_3d.cc:367-378` | Sand faces may move into the floor or platen (snow faces may not), so the particle sweep sees the approach speed friction needs. | A material rule hidden in a branch on `std::holds_alternative`. |
| Sand pile | `native/tests/test_m5_dry_scenes_3d.cc:95-460, 840-910` (`TestScene`) | 63 particles on a 0.05 m lattice, released 0.10 m above a fixed floor with friction 0.7; `lame_lambda` 700 Pa, `shear_modulus` 450 Pa, 30 degrees, cohesion 0, rest density 300 kg/m3; cells 0.075 m (step 0.001875 s) and 0.0625 m (0.0015625 s); `acoustic_cfl` 0.25. Angle `atan2(height, radius)` at 0.75 s and later. | Accepts 30 to 35 degrees at 0.75 s; does not hold over longer times (below). |
| Settling creep | `docs/native/m5-qualification.md` (about 350 sections) | Measured: 32.83 degrees at 0.75 s, 29.11 at 1.5 s, still falling at 3 s; halving the step moves the angle by +7.09 and +4.90 degrees; a grid shift moves it 1.4 degrees upward on another sampling; the ratio of sideways to normal contact impulse at resting contact is 0.148, 0.243, 0.382, 0.538 over three step halvings; the force floor omits at most `1.3e-5` of the proposed impulse. | Open: no cause established. Many temporary solver edits (paired blends, midpoint kicks, wall-anchored fits) were tried and are not ported. |
| GPU | `native/cuda/dry_solid_step_3d_host.cc`; `native/tests/test_cuda_m5_constitutive_3d.cu` | The sand return compiles and passes on the GPU; the step runs it on the CPU. | A GPU path that copies between dependent stages (CUDA report 8.4). |

Y-up on `main`: `gravity_y`, the floor normal `{0,1,0}`, the floor layer `world_position.y <= spacing`, and
the pile angle's height along y. Here every rule uses `normal_out_of_solid`, gravity is `{0, 0, -9.81}`, and
the angle's height is along z. Comparisons rotate with `R = rows {1,0,0},{0,0,-1},{0,1,0}` (materials 2).

## 3. Public interface

Sand adds to `native/fluidity/materials/`, namespace `fluidity::materials`; nothing else in materials
changes. All names NEW unless marked D. Exact additions:

```cpp
// material.h (shared). kSand appended after the four kinds of materials 3, so stored values never shift.
enum class MaterialKind : std::uint8_t { kLiquid, kGas, kSnow, kSpringySolid, kSand };  // D (A4, A9). kSand = 4.
struct MaterialResponse {                // D. Fields of materials 3, then:
  // ... kind, stiffness_against_shear, stiffness_against_squeeze, max_springy_shortening,
  //     max_springy_lengthening, hardening as in materials 3 ...
  double friction_angle = 0;             // D (A4). rad, 0 <= angle < pi/2. Sand: slope the grains hold among
                                         //   themselves. Papers: Drucker-Prager friction angle.
  double cohesion = 0;                   // D (A4). No unit (natural log of volume), zero or more. Sand: growth
                                         //   in volume the grains stand before they come apart.
};
constexpr bool HasInternalForce(MaterialKind kind);  // kSnow, kSpringySolid, kSand.

// stretch_response.h (shared). Two values appended; kLimitCaseCount grows with them.
enum class LimitCase : std::uint8_t {
  kInsideLimits,   // Nothing was limited: the motion was fully springy.
  kClamped,        // Snow: at least one stretch was clamped; the lost volume is permanent.
  kShapeReduced,   // NEW, value 2. Sand: the shape part of the stretch was shrunk to what friction holds.
                   //   Volume kept.
  kPulledApart,    // NEW, value 3. Sand: the grains came apart; the stretch went to the cohesion point, the
};                 //   rest is permanent.
inline constexpr int kLimitCaseCount = 4;  // Was 2 in materials 3.

// sand_material.h (shared). NEW file beside snow_material.h.
// stretch_before_limits: what the motion asks for (papers: trial stretches). *result untouched on failure.
// Fails: kBadInput (a stretch or exp(permanent_volume_change) not positive and finite), kBadMaterial, kNotFinite.
MaterialsError ComputeSandResponse(Vec3 stretch_before_limits, double permanent_volume_change,
                                   const MaterialResponse& response, StretchResponse* result);
// 1/s. Sand after a hit: the derivative with respect to position of contact::ComputeContactResponse, so the
// deformation follows the velocity the contact left (5.4). velocity_relative_to_solid: m/s, before the
// response, minus the solid's velocity at the hit point. friction: no unit, the particle's mixed friction times
// the solid's friction_scale, as contact uses it.
Mat3 ComputeVelocityChangeAfterHit(const Mat3& exact_velocity_change, Vec3 velocity_relative_to_solid,
                                   Vec3 normal_out_of_solid, double friction);

// material_response.h (shared). Two fields appended to MotionInStep.
struct MotionInStep {
  Mat3 exact_velocity_change{};            // D. 1/s. Used for `duration`.
  double duration = 0;                     // s. The whole step, or (sand that hit a solid) the part before.
  Mat3 exact_velocity_change_after_hit{};  // NEW. 1/s. Used for `duration_after_hit`.
  double duration_after_hit = 0;           // NEW. s. 0 without a hit; duration + duration_after_hit = step.
};

// apply_material_response.h (+ _cpu.cc, _gpu.cu). Fields appended to the two types of materials 3.
struct MaterialResponseInputs {
  // ... table, material_index, fields, grid_at_particle, contact_record, step_duration as in materials 3 ...
  const Vec3* start_position = nullptr;          // NEW. m. As given to contact::MoveParticlesInputs.
  const Vec3* velocity_for_this_step = nullptr;  // NEW. m/s, before the contact response:
                                                 //   contact::MoveParticlesInputs::velocity.for_this_step.
  std::span<const contact::Solid> solids;        // NEW. Ascending solid_id, as given to contact.
  // NEW (2026-11-15, reconcile item 33, settled here). The same solids on the run's processor:
  // contact::ContactBuffers::solids_copy().data(), written by MoveParticles this step. Requires: on the run's
  // processor with solids.size() entries (debug assertion). Read by the GPU path only; null on the GPU with a sand
  // kHit record is kBadInput.
  const contact::Solid* solids_on_run_processor = nullptr;
};
// For a kSand material with a non-null contact_record, the three new inputs are required (else kBadInput);
// for every other kind they are not read. The friction is read from MaterialBuffers::friction, which
// ApplyMaterialResponse already takes.
struct MaterialResponseResult {
  std::array<std::uint64_t, kLimitCaseCount> particle_count_by_case{};
  std::uint64_t hit_corrected_count = 0;  // NEW. Sand particles whose deformation was split at a hit.
};
// ApplyMaterialResponse (unchanged signature) also clears kCarriesFittedChange on sand particles that hit.
```

Defaults and table rules added (materials 5.1): `kSand` has internal force, the arrays
`springy_deformation` and `permanent_volume_change`, `CountRule::kKeepsItsCount`, one ingredient with
`kCannotBeSqueezed`, `at_solid_faces = kParticlesOnly`; `BuildMaterialTable` adds `0 <= friction_angle < pi/2`
and `cohesion >= 0` to its field checks (in field order), failing with `kBadMaterial`. No new error value.
`BuildParticlesSettings`, `BuildMaterialsMemoryEstimate`, `ComputeInternalForces`, `CheckMaterialState`,
`ComputeOwnMotionInTravel` (sand gets `own_motion_in_travel = 0`) take sand through their existing signatures.

## 4. Consumes from other modules

**Foundation (EXISTS):** as materials 4, including `math::Expm1` (sand sound bound pair terms) and
`math::TreeSum`.

**Materials (PROVIDED, materials 3, 5.9):** `MaterialKind`, `MaterialResponse`, `Material`,
`Ingredient`, `HasInternalForce`, `MaterialTable`, `BuildMaterialTable`, `BuildParticlesSettings`,
`StretchDirections`, `ComputeStretchDirections`, `LimitCase`, `StretchResponse`, `MaterialState`,
`InternalForce`, `ComputeInternalForcePerArea`, `MotionInStep`, `ComputeMaterialResponse`,
`ComputeDeformationAfter`, `ComputeSoundSpeedBound`, `MaterialResponseInputs`, `MaterialResponseResult`,
`ApplyMaterialResponse`, `CheckMaterialState`, `ComputeFacePush`, `kMinSupportShare`, `InternalForceTotals`,
`ComputeInternalForceTotal`, `ComputeWorkRates`, `MaterialBuffers::friction`, `MaterialsError`,
`MaterialsFailure`. To add in the sand module: the additions of section 3 at the seven extension points of
materials 5.9, and the sand case in `reference/materials/response.py`.

**Particles (PROVIDED, particles 3):** `ExtraFields::has_springy_deformation`, `has_permanent_volume_change`;
`CountRule::kKeepsItsCount` and the refusal `kMaterialKeepsItsCount`; `ParticleFieldsToChange` with
`Field::kSpringyDeformation | kPermanentVolumeChange | kVolumeRatio | kFlags`; `kCarriesFittedChange`;
`ParticlesSnapshot::Material` carrying both state arrays. Nothing to add.

**Contact (PROVIDED, contact 3):** `ParticleContact` (`kind`, `hit_fraction`, `normal_out_of_solid`,
`approach_speed`, `solid_index`), `ContactKind::kHit`, `Solid`, `SolidPose`, `ComputeSolidPose`,
`ComputeSolidVelocity(solid, pose, point)`, `ComputeContactResponse(velocity, solid_velocity,
normal_out_of_solid, friction)` (the formula 5.4 differentiates), `MoveParticlesInputs` (`start_position`,
`velocity`, `friction`, `solids`), `SolidReaction` (`solid_index`, `hit_position`, `velocity_for_next_step_change`) for
rung 8. Nothing to add.

**Transfer (PROVIDED, transfer 3):** `GridAtParticle::exact_velocity_change`; same-side `force` from
`ParticlesToGridInputs::internal_force_per_area`; filled faces with `kFillRadiiInCells{2, 3, 4}` for every
material (transfer question 4, option a: the fill radii sand needs beside a solid); one `GridFromParticles`
per material set (A8): a sand material is its own set. Nothing to add.

**Grid, pressure (PROVIDED, grid 3, pressure 3):** as materials 4; `pressure::SolidCoverage` filled by solids
from blocking materials. Nothing to add.

**Solids (PROVIDED, solids 3, 5.5, 5.7):** `MaterialSetFaces::at_solid_faces`: a set with `kParticlesOnly`
is left alone by `CollideSolids` (5.7 step 1), so sand particles reach the solid with the approach speed
friction needs; the pair rule between material sets on shared faces (5.7 steps 2 to 5);
`SolidsSettings::blocking_materials` (`MaterialTheLiquidCannotEnter` with `particle_radius`),
`ComputeSolidCoverage`; liquid sees sand as a blocking material inside its pressure solve (A18):
`BlockingSet`, `BuildBlockingMaterialFaces` (stage 10), the faces form of `ApplyPressureToMaterials` returning
`PressureOnMaterials` (stage 12), which writes the solved face velocity into the sand set's
`GridFromParticles::velocity` (replaces "the push returned one step late", solids question 3);
`SolidSettings::friction_scale`, `reports_reaction` for rung 8. The ready scene's
default "every material with `HasInternalForce` blocks liquid" covers sand once `HasInternalForce` includes it.
To add in the sand module: tests only (section 7, `Sand_FacesUntouchedAtSolid`, `Sand_BlocksLiquid`).

**Viscosity (PROVIDED, viscosity 3):** `CheckViscosityCanRun`, `ViscositySceneSummary`. To add in the sand
module, beside the snow exclusion viscosity declares (A9):

```cpp
// viscosity: new field of ViscositySceneSummary and new ViscosityError value.
bool has_sand = false;          // A kSand material shares the scene with a viscous ingredient.
kSandNotSupported,              // Sand shares the pressure solve with viscous liquid.
```

with viscosity's test "a syrup layer on a sand bed at rest stays at rest" (`main`: `:1770`) as its lifting test.

**Step (PROVIDED, step 3, 5.2, 5.5):** the stage calls of materials 4 (`ComputeInternalForces`,
`ApplyInternalForce`, `ApplyMaterialResponse`, ...); `CapabilityReason::kInternalForceAcrossSizeChange`
(any material with `HasInternalForce` in cells of different sizes) and `CapabilityLevel::kExperimental`; numerical
reject on `kDeformationCollapsed` and `kStepTooLong`. To add in the sand module:

```cpp
// step: CapabilityReason, appended after the existing values.
kSandSettlingCreep,     // Known limit (CapabilityLevel::kExperimental): a sand heap's rest angle creeps (5.7).
kViscousLiquidWithSand, // From viscosity kSandNotSupported (CapabilityLevel::kUnavailable), kSetViscosityToZero.
// step: SuggestedChange, appended after the existing values (NEW, A27: the main sequence removed it, since no
// reason there uses it once the free-solid exclusion is lifted). kSuggestedChangeCount + 1.
kSetViscosityToZero,
```

and, in `MoveParticles` (step 5.5, step 5), filling the three new `MaterialResponseInputs` fields from the same
`MoveParticlesInputs` it gave contact; `StepReport` adds
`MaterialResponseResult::hit_corrected_count` over ranges (integer sum).

**Cell resize and batches (PROVIDED, batches 3, 5):** `holds_particles_that_keep_their_count` and the rule
"a cell whose particles keep their count keeps its size" (`kKeep`); a split request reaching such a cell is
refused and counted. Nothing to add beyond the test `RefusedSplit_Sand`.

**Surface, foam and spray (PROVIDED, render output 3, 5.8):** simulation-particle playback as open-format
points for any material (A14; written by render output's writers), with `position`, `velocity.for_next_step`,
`volume_ratio`, `id`, `rest_volume`. To add in the sand module: sand in render output's per-material table
(5.8) as "points by default, no surface, no foam source, no volume", the row snow has; a `kSand` material is
in the default point products. No sand surface (A9).

**I/O and CLI (PROVIDED, I/O 3, 5.1):** the scene file and its checks, `kMaterialPresets`, the derivation of
one `materials::Material` per scene material, `saved_state` with every particle array, `ToName` codes in
events. To add in the sand module:

```cpp
// io/material_presets.h: new preset row.
kSand,   // friction_angle 0.5235987755982989 rad (30 degrees), cohesion 0, stiffness_against_squeeze 700 Pa,
         //   stiffness_against_shear 450 Pa, rest density 300 kg/m3, friction 0.7 (decision 4).
```

the stable name `set_viscosity_to_zero` for `SuggestedChange::kSetViscosityToZero` in `io/stable_names.cc` (A27;
the table is `static_assert`ed against `kSuggestedChangeCount`, so step and I/O change in one commit); the material
kind `sand` and the fields `friction_angle` (rad) and `cohesion` in
`schemas/simulate/v2/scene_file.fbs`; "each fill object of snow, sand or a springy solid gets its own
material" in I/O 5.1 step 4.

**Blender add-on (PROVIDED, add-on 3):** `FluiditySubstance`, `SubstancePreset`, `SUBSTANCE_PRESETS`,
`ReferenceScene`, `REFERENCE_SCENES`, `ROLE_TABLE`, `check_scene_quickly`, the native readers for points. To
add in the sand module: `SAND` in `FluiditySubstance.kind`; the properties `friction_angle` (rad) and
`cohesion`; a `sand` entry in the `substance_kinds` column of `ROLE_TABLE` (add-on 3.2: the `START_MATERIAL`
row, which today allows `liquid`, `snow`, `springy_solid`), so that the quick check's "missing or wrong-kind
substance" finding (add-on 3.4 and 5.3 step 1, `check_scene_quickly`) accepts a sand substance on a start
material; the sand row of `SUBSTANCE_PRESETS` (values of the I/O preset); the sand row of `REFERENCE_SCENES`
(decision 4); its preset and reference-scene tests in Blender 5.3 and 5.2; (A27) the `REPAIR_TABLE` entry
`set_viscosity_to_zero` -> `fluidity.apply_field_value` on the property of the finding's `scene_file_path`, value 0
(the entry the add-on had before A27 removed the name from the main sequence), and the count of step names in
`test_repair_table_covers_suggested_changes` + 1.

**Gas:** nothing consumed.

## 5. Method

Notation as materials 5: `F` springy deformation, `L` exact velocity change, `s` stretches, `e = ln s`,
`mu`/`lambda` the two stiffnesses, `dt` step duration.

### 5.1 Table row

| Kind | Internal force | Extra particle arrays | Count rule | Ingredients | Default `at_solid_faces` |
|---|---|---|---|---|---|
| `kSand` | yes | `springy_deformation`, `permanent_volume_change` | keeps its count | 1 | particles only |

### 5.2 Sand inside the stretch directions (ported unchanged)

Papers: Hencky stiffness with a Drucker-Prager limit, method paper ref 7. With `e = ln s`,
`sum = e.x + e.y + e.z`, `over = sum - cohesion`:

1. `over >= 0` (`kPulledApart`): every `e = cohesion / 3`; `permanent_volume_change += over`. Tested first, so
   no division by a zero shape part can occur.
2. Otherwise `shape = e - sum / 3`, `slope = sqrt(2/3) * 2 sin(friction_angle) / (3 - sin(friction_angle))`,
   `allowed = -over * slope * (1 + 1.5 lambda / mu)` (0 when `slope` is 0). If `|shape| > allowed`
   (`kShapeReduced`): `shape *= allowed / |shape|`; the sum, and so the volume, is unchanged.
3. `internal_force_per_area_i = 2 mu e_i + lambda * (sum after the limit)`.

World form as snow (materials 5.3, `Recompose` order).

### 5.3 The route for sand

*Start of step.* Materials 5.2 on the stored deformation, then 5.2 step 3 (no limit: a stored state is inside
its limits). Then 5.5.

*End of step.* 1. `F <- (I + duration * L) F`; with a hit, then `F <- (I + duration_after_hit *
L_after_hit) F`. `L` on the left. 2. Materials 5.2 on the new `F`, 5.2 here, rebuild `springy_deformation`.
3. `volume_ratio = product of springy_stretch * exp(permanent_volume_change)`. 4. `CheckMaterialState` as
materials 5.4 step 5.

### 5.4 Sand that hits a solid (ported; generalised from the fixed floor)

For a sand particle with `contact_record.kind == kHit`: `duration = hit_fraction * dt`,
`duration_after_hit = dt - duration`, and with `v = velocity_for_this_step - solid velocity at the hit point`
(`start_position + velocity_for_this_step * duration`, `contact::ComputeSolidPose` and
`contact::ComputeSolidVelocity` of `solids[solid_index]`), `n = normal_out_of_solid`, `vn = Dot(v, n)`, and
`friction` the particle's friction times the solid's `friction_scale`:

1. `vn >= 0`: `L_after_hit = L`.
2. Otherwise remove the change of the speed into the solid: `gn = n^T L`, `L1 = L - n gn`.
3. `friction == 0`: `L_after_hit = L1`. With `vt = v - vn n`, `speed = |vt|`: if `speed == 0` or
   `friction >= speed / -vn` the particle sticks and `L_after_hit = 0`.
4. Otherwise `t = vt / speed`, `cut = friction * -vn / speed`, `gs = t^T L`:
   `L_after_hit = (1 - cut) L1 + t (friction * gn + cut * gs)` (outer product of column `t` and a matrix row).

This is exactly the derivative, with respect to position, of contact's slip response
`vt - friction * (-vn) * t`. After any hit a sand particle loses `kCarriesFittedChange` (`main` `:570`) and is
counted in `hit_corrected_count`. New: any normal, and a moving solid through `v`; `main` tests only the fixed
floor. Only the first hit of the step splits it (later hits of the same step use `L_after_hit`, as `main`
with its one floor).

### 5.5 Sound speed bound (ported unchanged)

With `t_i = 2 mu e_i + lambda sum`: stretch rows `2 mu + lambda - t_i + 2 lambda`; for each pair with
`d = e_high - e_low`: `high = 2 mu d / (-Expm1(-2 d))` (`mu` when `d = 0`), `low = high * exp(-2 d)`, row
`high + |low - t_low|`. `sound_speed_bound^2 = largest row / rest_density`; a row or pair test that is not
positive is `kMaterialUnstable`. `kStepTooLong` and `step_limit` as materials 5.5.

### 5.6 Faces at solids, support rule, travel share

Sand faces are `kParticlesOnly`: `CollideSolids` leaves them alone, so a sand particle arrives at the solid
with the approach speed contact's friction needs (`main` `:367-378`). The internal force goes through the
support rule of materials 5.8 unchanged. Sand travels with the grid velocity (`own_motion_in_travel = 0`);
`own_motion_kept` is the material's.

### 5.7 The settling creep and the eight-step diagnosis

The fault: a released sand heap reaches about 30 degrees, then its angle keeps falling (2). No cause is
established. Candidates the diagnosis separates: the springy transfer itself (energy or momentum leak), the
limits, the floor friction at particles versus faces, the time step, the grid's position, and the contact
impulse ratio. Each rung adds one ingredient to the one before, so the first rung that fails names the
ingredient that causes the creep. No material number is tuned to move a rung. Every rung records kinetic
energy and the angle's rate over the last 0.5 s, not one angle at one time. The angle is
`atan2(largest height above the floor, largest horizontal radius from the release centre)`, as on `main`.
The rungs are the tests of section 7; the module is done when all eight pass, or the owner decides otherwise.
Until then `kSandSettlingCreep` is reported as a known limit (decision 1).

## 6. CPU and GPU

- Shared formula header `sand_material.h`; the sand case in the shared switch of `material_response.h`.
- The private `_gpu.cu` files gain one instantiation of each per-particle function for `kSand` (no thread
  branches on the kind). Stretch directions and the new state stay in registers. The hit split reads
  `start_position`, `velocity_for_this_step`, the material buffers' `friction` and the contact record by
  particle index, and the
  solids from the copy contact already keeps on the processor (`ContactBuffers::solids_copy()`, passed as
  `solids_on_run_processor`; no per-call copy, so the path allocates nothing and never returns `kOutOfMemory`).
- Arrays: none new. A sand material uses the per-particle buffers of materials 6 (72 B internal force, 8 B
  friction, 16 B shares, 9 B working buffers) and the particles' two state arrays (72 B and 8 B, committed and try).
- Fixed order: as materials 6; `hit_corrected_count` is an integer sum.
- Tolerance, CPU versus GPU: as materials 6; `LimitCase` exact except within `1e-12` of the cone or the tip,
  where either case is accepted.
- Status until the owner's hardware run: "GPU: compiled, not run".

## 7. Tests

**Analytic (CPU; all run in the container).**

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeSandResponse, InsideLimits_IsUnchanged` | uniform squeeze 0.98 | same stretch, force `(2 mu + 3 lambda) ln 0.98` | `1e-13` |
| `..., MainCase_IsUnchanged` | stretches (0.9, 0.9, 0.9), permanent 0.2, `mu` 2, `lambda` 3, 30 degrees, cohesion 0.03 | stretch 0.9, permanent 0.2, force `13 ln 0.9` (`m5_constitutive_3d_cases.h:24-35`) | `1e-11` |
| `..., PulledApart_GoesToCohesionPoint` | sum of logs above cohesion | stretch `exp(cohesion/3)`, permanent grows by the excess, zero force when cohesion is 0 | `1e-13` |
| `..., ThirtyDegrees_GivesThreeToOne` | forces (-5, -1, -1) Pa asked for, 30 degrees | (-4.2, -1.4, -1.4) Pa, sum and permanent unchanged (`main`'s triaxial check) | `1e-11` |
| `ComputeSoundSpeedBound, SandAtRest_IsRowSum` | sand at rest | `sqrt((2 mu + 3 lambda) / density)` | `1e-13` |
| `ComputeVelocityChangeAfterHit, MatchesContactResponse` | linear velocity field, finite difference of `contact::ComputeContactResponse` over `1e-6` m, slip, stick and leaving cases, tilted normal | equals the formula of 5.4 | `1e-8` |
| `..., MovingSolid_SameAsMovingFrame` | add one constant velocity to particle and solid | same result as at rest | `1e-14` |
| `ApplyMaterialResponse, SandHit_SplitsStep` | one sand particle hitting a tilted plane at `hit_fraction` 0.25 | deformation equals the two-part product of 5.3; `hit_corrected_count` 1; fitted change cleared | `1e-14` |
| `..., SandHit_MissingInputs_IsBadInput` | sand with records, null `start_position` | `kBadInput`, nothing written | exact |
| `BuildMaterialTable, SandRules` | 90 degrees; negative cohesion; two ingredients | `kBadMaterial` with the material index | exact |
| `CollideSolids, Sand_FacesUntouchedAtSolid` | sand set moving into a flat solid | face velocities unchanged; snow set in the same setup loses the inward part | exact |
| `ComputeSolidCoverage, Sand_BlocksLiquid` | sand layer under liquid | `distance_to_solid` negative inside the sand balls | `1e-12` |
| `ResizeCells, RefusedSplit_Sand` | split asked in a cell holding sand | request dropped, counted, grid unchanged | exact |
| Repeat run | every batch operation on sand twice | identical bytes | exact |

**The eight-step diagnosis (numerics report 9.5).** A rung marked "expected to fail" asserts the physically
right bound and is registered `WILL_FAIL` with label `known_open`, with its last measured numbers in
`docs/known-open-physics.md` (materials decision 3); the suite stays green while it fails and turns red the day
it passes, which forces the record and the label to be removed. The module is done when all eight pass (no
`known_open` label left here), or the owner decides otherwise. Pile setup unless stated: `main`'s pile
(section 2) rotated to Z-up, fixed flat floor with friction 0.7 and `reports_reaction`.

| Rung | What is added | Assertion | Expected |
|---|---|---|---|
| 1 | Response alone, prescribed deformation | the `ComputeSandResponse` rows above | pass |
| 2 | Springy transfer, limits off (`friction_angle` 89 degrees, large cohesion), no solid, no gravity: a stretched block set free | energy never grows (`<= 1 + 1e-9` of the start); momentum drift `<=` summed `omitted_impulse` | pass |
| 3 | Limits on, no contact: column in free fall | `L = 0`: state arrays unchanged to `1e-12`, no `LimitCase` other than `kInsideLimits` | pass |
| 4 | Fixed flat floor with friction 0.7: a low wide slab at rest under gravity (already below 30 degrees) | top surface moves less than 1 % of its height in 3 s; kinetic energy below `1e-4` J after 0.5 s | unknown: first measurement decides, then recorded |
| 5 | The pile (63 particles at 0.05 m, 300 kg/m3, 30 degrees, cells 0.075 and 0.0625 m), measured at 0.75, 1.5, 3 and 6 s | angle changes by less than 1 degree between 1.5 s and 6 s; kinetic energy below `1e-3` J | **expected to fail** (`main`: 32.83 degrees at 0.75 s, 29.11 at 1.5 s, still falling at 3 s) |
| 6 | Rung 5 with the step halved | angle at 1.5 s within 2 degrees of rung 5 | **expected to fail** (`main`: +7.09 and +4.90 degrees) |
| 7 | Rung 5 with the grid shifted half a cell, sideways and upward | angle within 2 degrees | unknown (`main`: 1.4 degrees upward on another sampling) |
| 8 | Rung 5: sum of sideways over sum of normal contact impulse at resting contact (from `contact::SolidReaction`), at the step and at half the step | the ratio differs by less than 10 % | **expected to fail** (`main`: 0.148, 0.243, 0.382, 0.538 over three halvings) |

A rung marked "unknown" is registered as a plain test once its first measurement is recorded; if that
measurement fails, it becomes `known_open` like the others.

**Reference piece:** the sand case added to `reference/materials/response.py` (materials 7): the sand
response with `numpy.linalg.svd` for 1,000 random rotated deformations with a fixed seed, and the true fastest
wave by sampling 10,000 directions. Native agrees within `1e-10` relative; native bound `>=` sampled speed.
Uncertainty removed: the cone and tip branches on general rotated states, and the bound for a general state.
**Lean proofs: none.**

**Comparison with `main`:** the sand cases of `m5_constitutive_3d_cases.h`, rotated to Z-up, `1e-12`
relative. The pile is not a comparison: weights, contact and the support rule all changed, so its numbers are
the "expected" column above only.

**Reference scene and add-on:** the sand reference scene (decision 4) runs end to end through the CLI and
loads in Blender 5.3 and 5.2 through the native point reader; the sand preset applies its values.

**GPU:** every row of the first table through the batch operations within the tolerance of section 6; rungs 2,
3 and 5 on the GPU within `1e-9` relative of the CPU at 0.75 s; the repeat-run test. Without a GPU the
container runs every CPU test and compiles the `_gpu.cu` files.

## 8. Method paper section

`== Sand <sec-sand>` inside `= Materials <sec-materials>`, after `<sec-snow>`, and
`== Known limits of sand <sec-sand-limits>` (the open rungs, with numbers, updated with each record).
Equations `<eq:sand-limit>`, `<eq:hit-derivative>`.

Figures: `<fig-sand-limit>` the allowed region of sand as a cone in the plane (volume part, shape part) with
the three cases as arrows; `<fig-hit-split>` a sand particle's step split at the hit, with the two velocity
changes; `<fig-sand-creep>` the angle and kinetic energy against time for rungs 5 and 6.

Pseudocode: `<proc-sand-response>` (5.2), `<proc-hit-split>` (5.4).

Sources from the old reference list: [7] Klar 2016 (sand), [18] Stomakhin 2014, [1] Jiang 2015. No entry
exists for the Hencky row-sum sound bound: "citation needed".

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `granular` | `sand` (`kSand`) | DECIDED word |
| `GranularParameters3D` | `MaterialResponse` with `MaterialKind::kSand` | DECIDED name; NEW shape |
| `m5_constitutive_3d.h` (sand part) | `sand_material.h` | DECIDED (A9) |
| `ReturnGranularPrincipal3D` | `ComputeSandResponse` | NEW (O6) |
| `friction_angle`, `cohesion_log_strain` | `friction_angle`, `cohesion` | DECIDED (A4) |
| `deviator`, `deviator_norm`, `cone_radius` | shape part: `shape_log_stretch`, its size, `allowed_shape_size` (private) | NEW (O6) |
| apex branch, cone projection | `LimitCase::kPulledApart`, `kShapeReduced` | NEW (O6) |
| `ProjectStationaryFloorVelocityGradient3D` | `ComputeVelocityChangeAfterHit` | NEW |
| `elastic_stretches`, `log_plastic_jacobian`, `kirchhoff_stress` | `springy_stretch`, `permanent_volume_change`, `internal_force_per_area` | DECIDED |
| sand face rule at the floor | `AtSolidFaces::kParticlesOnly` | NEW (materials) |
| (none) | `hit_corrected_count`, `velocity_for_this_step`, `exact_velocity_change_after_hit`, `duration_after_hit` | NEW (A1 for `for_this_step`) |
| (none) | `kSandSettlingCreep`, `kViscousLiquidWithSand`, `kSandNotSupported`, `has_sand` | NEW |
| sand envelope angle | rest angle (`atan2` as on `main`) | NEW |

## 10. Not ported

- The sand branches of `AdvanceDrySolid3D` and `CudaAdvanceDrySolid3D` (`dry_solid_step_3d.cc`): the one step
  runs sand.
- `TestScene`'s experiment parameters (`compatible_floor`, `stored_velocity_decay_rate`,
  `reset_column_velocity`, `granular_modulus_scale`, `quantized_mass_bits`, `release_cone`): open experiments.
- The roughly 350 sections of `docs/native/m5-qualification.md` and their temporary solver edits: their
  measurements feed the "expected" column of the diagnosis and nothing else.
- The old paper's travel rule for sand ("when the springy volume is at least 1 and out of contact"): never
  implemented on `main`.
- The force floor (materials 10); the hard-wired floor and platen (contact and solids).

## 11. Decisions

**1. Is sand a usable capability while rungs 5, 6 and 8 fail?** (Was materials question 4.) Options: (a) sand
runs; the capability check reports "sand: long settling not qualified" and the paper says so; (b) sand is
rejected by the capability check until rung 5 passes. Recommendation: (a): a falling or thrown heap is right
at the times `main` tested, and hiding it would not make the ladder easier to climb. Affects: step, I/O,
Blender.

Decision: (a), recommended option (owner's standing instruction, 2026-11-14): `kSandSettlingCreep` with
`CapabilityLevel::kExperimental` while any rung is `known_open`.

**2. Sand faces do not stop at solids (`AtSolidFaces::kParticlesOnly`).** (Was the sand part of materials
question 5.) Ported as a field of the material, applied by solids. Alternative: one rule for snow and sand
(never constrain faces; or always). The difference is part of what rung 8 measures, so it is kept visible
instead of chosen now. Affects: solids.

Decision: sand defaults to `kParticlesOnly`, recommended option (owner's standing instruction, 2026-11-14).
If a rung shows the face rule is the cause, the change is made in this module's commit that fixes the rung.

**3. Inputs of the hit split.** `main` had one floor and read nothing; here the split needs the start
position, the velocity for this step and the solids (the friction is already in `MaterialBuffers`). Options:
(a) three fields appended to `MaterialResponseInputs`, read only for sand (section 3); (b) the step computes
`MotionInStep` per particle and passes it, so materials reads no contact input.

```cpp
inputs.start_position = move.start_position; inputs.velocity_for_this_step = move.velocity.for_this_step;  // a
```

Recommendation: (a): the GPU keeps the split in registers in the same pass as the response. Affects: step.

Decision: (a), recommended option (owner's standing instruction, 2026-11-14).

**4. Sand preset and reference scene.** The add-on's preset values for sand are "copied from the reference
scenes once those are final" (add-on question 9). Options: (a) the preset is `main`'s pile material
(`stiffness_against_squeeze` 700 Pa, `stiffness_against_shear` 450 Pa, 30 degrees, cohesion 0, 300 kg/m3,
friction 0.7), and the reference scene is rung 5's release seeded by I/O's rule (8 particles per smallest
cell) with cells of 0.0625 m, run for 6 s, recording angle, radius and kinetic energy; (b) physical sand
(about 1600 kg/m3, megapascal stiffness), which needs far shorter steps and has never been run.
Recommendation: (a); (b) after the creep is understood. Affects: I/O, Blender.

Decision: (a), recommended option (owner's standing instruction, 2026-11-14).

**5. Viscous liquid with sand.** Viscosity excludes snow beside viscous liquid (A9 wording). Options: (a) a
separate `kSandNotSupported` with `has_sand` (section 4), lifted by viscosity's syrup-on-sand test; (b) rename
viscosity's snow exclusion to cover every material with internal force. Recommendation: (a): it leaves the
finished viscosity design untouched until sand exists. Affects: viscosity, step.

Decision: (a), recommended option (owner's standing instruction, 2026-11-14).

**6. Liquid pressure on sand.** Section 4 read the push from the liquid one step late (solids question 3).
Decision: sand, as every blocking material, enters the liquid's pressure solve as blocking-material faces
(stage 10) and gets its face velocity back from the faces form of `ApplyPressureToMaterials` (stage 12);
nothing to add in the sand module beyond `Sand_BlocksLiquid`. Source: plan decision, 2026-11-14 (coupling
inside the pressure solve), owner answer A18.

**7. The suggestion for a viscous liquid with sand.** A27 lifted the free-solid exclusion and removed
`SuggestedChange::kSetViscosityToZero` from the main sequence (`kSuggestedChangeCount` 12), so
`kViscousLiquidWithSand` is the one combination still refused because of viscosity, and its suggestion no longer
exists before this module.
Decision: the sand module adds `SuggestedChange::kSetViscosityToZero` itself, appended after the existing values
(`kSuggestedChangeCount` + 1), with the stable name `set_viscosity_to_zero` and the add-on's repair (viscosity set
to 0 on the finding's field) restored (section 4). Source: owner answer A27; `plan-decisions.md`, "A viscous
liquid sticks, and free solids in the viscous solve (A26, A27)", "Exclusions" (flagged there for the owner).

**8. The solids on the GPU for the hit split (reconcile item 33; A58).** The plan allowed a per-call `GpuArray`
copy of the solids, which would make `ApplyMaterialResponse` an allocating form returning `kOutOfMemory`, against
"only building functions return it". Decision: no per-call copy; the GPU path reads contact's own copy through the
new input `MaterialResponseInputs::solids_on_run_processor` (= `contact::ContactBuffers::solids_copy().data()`,
NEW in contact 3.5), which `MoveParticles` wrote in the same step. Tests `ApplyMaterialResponse,
SandHitWithoutSolidsOnProcessor_IsBadInput` and `ApplyMaterialResponse, OnGpuSandHit_AllocatesNothing`. Source:
`plan-decisions.md` "Owner answers A40–A64 applied", item 11 (33: "settled in the sand module").

**9. Other answers of the A40–A64 pass.** Sand has no builder of its own (it uses `MaterialBuffers` and the
particles' arrays, which take `RunMemory&` in their modules); its error values are `MaterialsError`'s
(`kOutOfMemory` last there). A64: sand particles keep their count (`kKeepsItsCount`) and a sand cell is never split
(Task 10); no grid-only region touches sand. Source: `plan-decisions.md` items 1, 3, 11.
