# Materials module design

Status: decisions applied 2026-11-14 (`owner-answers.md`, `changes-to-apply.md`). Module 6 of the build order
(after viscosity, before solids). Sand is not part of this module: it is the sand module, last of the build order, after
the Blender add-on (owner A9), designed in `15-sand-design.md`; this design names the extension points it
fills (5.9).

## 1. Purpose

A material is a description: what it is made of (its ingredients) and how it answers when the motion deforms
it. A liquid answers only by changing volume; snow and a solid that can be cut remember a springy deformation
and push back with an internal force. This module holds those descriptions for every other module and runs the
per-particle update that turns "how the velocity varies around the particle" into the particle's new
remembered state.

Owns: the scene's ingredient and material table and every number derived from it (`viscosity`, `friction`,
`PressureLaw`, rest density, surface tension and export name per ingredient; which ingredient is the gas;
which extra particle arrays a material asks for); `MaterialResponse` for liquid, gas, snow and springy solids
behind one pair of functions; the internal force per particle; the rule by which a grid face answers that force
(the support rule); the sound speed bound and the step limit it gives; the per-material travel share rule;
per-particle `friction`; per-cell `bulk_stiffness` and `target_volume_ratio`; the work identity as a test
oracle; the extension points the sand module fills (5.9).

Does not own: the particle arrays (particles); the sums onto faces and the read back (transfer); the contact
sweep (contact); the formula of `PressureLaw` and the pressure solve (pressure); the viscosity operator
(viscosity); a solid's shape and motion, cutting and `plate_side`, snow seen by a liquid as a solid and the
pressure pushing back on it, contact between two material sets on shared faces (solids, section 4); gas
temperature, fuel and the gas law (gas); the artist's controls and their mapping to the numbers here (I/O);
sand (the sand module, `15-sand-design.md`).

Melting of snow by liquid water is not modelled, so liquid-on-snow scenes are supported mechanically but are not a validation target (owner note, 2026-11-14).

## 2. What exists today

| Piece on `main` | Where | What it does | What is wrong |
|---|---|---|---|
| Snow return | `native/include/fluidity/base/m5_constitutive_3d.h:14-33, 159-211` | In the frame of the three stretches: clamps each stretch and moves the lost volume into the log of permanent volume, then stiffens. Returns force per area along the three directions. Output untouched on bad input. **Maths ported unchanged.** The sand return of the same file (`:84-157`) is ported by the sand module. | Milestone code in the file name; textbook words as identifiers (A8, N1); `PublishPrincipalState` checks "representable" values (A9). |
| Stretch directions | `base/solid_deformation_3d.h:34-149, 151-191` | One-sided Jacobi rotations on the columns of the deformation: no squared matrix. Rejects a reflection or a flat deformation before rotating (`:62-86`). Rebuilds the deformation and the world force per area. **Ported unchanged.** | Pointer-to-member loops; tolerances inline (`1.5e-14`, `2e-15`, `1e-12`). |
| Sound speed bound | `base/solid_acoustic_3d.h:97-158` (snow); neo-Hookean `native/src/transfers_3d.cc:291-327` | Largest row sum of the scaled stiffness blocks over rest density; fails when a principal plane has no real wave. **Ported unchanged.** The sand bound (`:39-92`) is the sand module's. | The neo-Hookean one sits in the transfer file by accident (transfer design, section 10). |
| Stretch update | `native/src/dry_solid_step_3d.cc:24-35`; `native/src/elastic_blade_deformation_3d.cc:12-23, 70-76` | `new = (I + dt L) old`, `L` on the left (numerics report 9.2). **Ported unchanged.** | Two copies. |
| Springy solid | `transfers_3d.cc:108-135, 263-289`; step `elastic_blade_deformation_3d.cc:280-400` | `mu (F F^T - I) + lambda ln(J) I`; positive volume checked against a rounding bound. | Interval-style determinant bound (A9). A whole separate step, checkpoint and particle type for one material. |
| Dry-solid step (snow) | `native/src/dry_solid_step_3d.cc:194-587`; header `native/include/fluidity/host/dry_solid_step_3d.h` | Own particle type, own deposit, own force update, own contact against a hard-wired floor and platen, own commit by copying a `std::vector` (`:497, :582`). | A separate route beside the liquid step (programme anti-goal). One body, uniform grid, complete neighbourhoods only. `0.99 *` inline (`:476`). |
| Force floor | `dry_solid_step_3d.cc:338, 353-360` | Face acceleration is `force / max(mass, 1e-4 * smallest particle mass)`. | The floor depends on the lightest particle anywhere in the body, has a mass unit, and silently changes the operator (numerics report 9.3, 9.4; `docs/native/m5-qualification.md:61-68`). Without it 4 of 12 runs stop on the motion bound (same file, "Low-mass stress acceleration control"). |
| Snow at solids | `dry_solid_step_3d.cc:367-378` | Snow faces may not move into the floor or the platen. | A material rule hidden in a branch on `std::holds_alternative`. |
| Liquid volume | `native/src/liquid_substep_3d_cpu.cc:387-393` | `volume_ratio *= exp(dt * (L.xx + L.yy + L.zz))`. | Fine; lives in the scheduler. |
| Liquid travel rule | `liquid_substep_3d_cpu.cc:308-313` | Own motion also moves the particle only when the volume grows, within one cell width under the surface, more than one cell width from a solid. | A material rule inside the scheduler; mixed with `0.99`. |
| Ingredients | `host/liquid_mixture_3d.h:15-27`; `host/liquid_eos.h:7-17` | Substance with density and optional viscosity; `phase_id` says who mixes; pair table of surface tension. | Liquid only. `phase_id` (replaced by "a material is what mixes", owner Q31). |
| GPU | `native/cuda/dry_solid_step_3d_host.cc` (76 lines); `native/tests/test_cuda_m5_constitutive_3d.cu` | Only the deposit and the read back run on the GPU; returns, force and contact run on the CPU each step. | A GPU path that copies between dependent stages (CUDA report 8.4, programme anti-goal). The returns themselves do compile and pass on the GPU. |

Y-up on `main` shows as `gravity_y`, a floor normal `{0,1,0}`, `platen.start_y`, `gradient.y_row = {}` and a
tangent built from x and z. Nothing in this module names an axis: every rule is written with
`normal_out_of_solid`. Comparing with `main` needs `F' = R F R^T`, `force' = R force R^T`,
`R = rows {1,0,0},{0,0,-1},{0,1,0}`. `Mat3` on `main` is three row vectors; Foundation's is nine named doubles.

## 3. Public interface

Folder `native/fluidity/materials/`, namespace `fluidity::materials`. Every name is NEW unless marked D
(DECIDED). "shared" = header only, `FLUIDITY_CPU_AND_GPU`, C++20 subset, returns the bare enum. Functions that
can fail on the CPU return `std::expected<T, MaterialsFailure>` (A2, A13). Names marked NEW in this update are
the changes of 2026-11-14.

```cpp
// materials_error.h (shared); ToMessage and ToName in materials_message.h (CPU only)
enum class MaterialsError : std::uint8_t {
  kNone,
  kBadInput,               // A value is not finite, a span has the wrong size, or a needed pointer is null.
  kBadMaterial,            // A material or ingredient number is outside its range (5.1), or an id repeats.
  kBadExportName,          // NEW. An ingredient's export name is empty, or equal to another's once normalised.
  kWrongFields,            // The particles of a material lack an array its kind needs.
  kOutsideAllowedRegion,   // A stored springy_deformation is not one its material could have produced.
  kDeformationCollapsed,   // After the step the deformation has no positive volume, or its stretch directions
                           //   cannot be found. The step should reject the try and shorten.
  kMaterialUnstable,       // The springy state has no real sound speed in some direction.
  kStepTooLong,            // step_duration is above the sound limit. magnitude holds the limit, s.
  kPressureLawOutOfRange,  // carried_pressure is below what an ingredient's pressure law allows.
  kSurfaceTensionsDiffer,  // Two liquid ingredients differ in surface tension against air (decision 6).
  kNotFinite,              // A result overflowed.
  kWrongGridVersion,       // An array belongs to another grid version or another step try.
  kBuffersTooSmall,        // More particles than the material buffers were built for.
  kOutOfMemory,
  kRunOnCallFailed,        // D (Q11). CheckCanRunOn failed for the chosen processor, or a call to it failed.
};
struct MaterialsFailure {  // D shape (A2). On the GPU: the failing particle with the lowest id.
  MaterialsError error = MaterialsError::kNone;
  std::uint64_t particle_id = 0;   // 0 when no particle is involved.
  std::uint32_t material_index = 0;
  std::uint32_t ingredient_index = 0;  // NEW. Scene ingredient index for kBadMaterial, kBadExportName; else 0.
  double magnitude = 0;            // The offending value in its own unit, or 0.
};
std::string_view ToMessage(MaterialsError error);  // D (A13)
std::string_view ToName(MaterialsError error);     // NEW. Stable code: "bad_export_name", "step_too_long", ...
```

```cpp
// material.h (shared)
// NEW in this update: kSand removed (A9). The sand module appends kSand = 4 after kSpringySolid, so the stored
// values of these four never change.
enum class MaterialKind : std::uint8_t { kLiquid, kGas, kSnow, kSpringySolid };  // D (A4, A9)
// What the grid faces of this material do where a solid is (applied by solids in CollideSolids through
// solids::MaterialSetFaces::at_solid_faces).
enum class AtSolidFaces : std::uint8_t { kParticlesOnly, kNoMotionIntoSolid };
struct Ingredient {                      // D word. One substance. Index in the table = scene ingredient index.
  std::uint64_t ingredient_id = 0;       // Stable, nonzero, unique in the scene.
  pressure::PressureLaw pressure_law{};  // D. Holds rest_density, kg/m3: the only copy of that number.
                                         //   Gas: kCannotBeSqueezed, rest_density = gas::ComputeAmbientDensity.
  double viscosity = 0;                  // Pa s, zero or more. Zero for snow, springy solids and gas.
  double surface_tension = 0;            // N/m against air, zero or more. Zero for all but liquids.
  double friction = 0;                   // No unit, zero or more. Against solids. Papers: Coulomb coefficient.
                                         //   Zero for gas.
};
struct MaterialResponse {                // D. Textbook: constitutive model. Unused fields stay 0.
  MaterialKind kind = MaterialKind::kLiquid;
  double stiffness_against_shear = 0;    // D. Pa, positive. Textbook: shear modulus, mu.
  double stiffness_against_squeeze = 0;  // D. Pa, zero or more. Textbook: Lame lambda. Not the bulk modulus:
                                         //   bulk = stiffness_against_squeeze + 2/3 stiffness_against_shear.
  double max_springy_shortening = 0;     // No unit, 0 <= value < 1. Snow. Textbook: critical compression.
  double max_springy_lengthening = 0;    // No unit, zero or more. Snow. Textbook: critical stretch.
  double hardening = 0;                  // No unit, zero or more. Snow: how fast packed snow stiffens.
};                                       // The sand module adds friction_angle and cohesion (D words, A4).
struct OwnMotionSettings {               // Words D. Effective shares, as transfer::MixInputs takes them.
  double kept = 0;                       // 0..1. own_motion_kept for every particle of the material. Gas: 0.
  double in_travel = 0;                  // 0..kept. own_motion_in_travel where the travel rule holds (5.7). Gas: 0.
};
struct Material {                        // Index in the table = particles' material_index.
  std::uint64_t material_id = 0;         // Same id as particles::MaterialParticlesSettings::material_id.
  MaterialResponse response{};
  std::uint32_t first_ingredient = 0;    // Scene ingredient index of its first ingredient.
  std::uint32_t ingredient_count = 1;    // 1..particles::kMaxIngredientCount. More than 1: liquid only.
  OwnMotionSettings own_motion{};
  AtSolidFaces at_solid_faces = AtSolidFaces::kParticlesOnly;
};
constexpr bool HasInternalForce(MaterialKind kind);  // kSnow, kSpringySolid. (The sand module adds kSand.)
```

```cpp
// material_table.h (CPU only)
struct MaterialPair {                    // Two materials that touch and do not mix.
  std::uint32_t material_index_low = 0, material_index_high = 0;  // low < high.
  double surface_tension = 0;            // N/m at their boundary. For pressure::MaterialBoundary.
};
struct MaterialTableInputs {
  std::span<const Material> materials;   // In material_index order. Ingredients of one material are contiguous.
  std::span<const Ingredient> ingredients;
  std::span<const std::string_view> ingredient_export_names;  // NEW. One per ingredient, same order (5.1).
  std::span<const MaterialPair> pairs;   // Sorted by (low, high). May be empty.
};
enum class PlatesInScene : std::uint8_t { kNone, kSome };
class MaterialTable {                    // Immutable after BuildMaterialTable. Lives in CPU memory.
 public:
  std::span<const Material> materials() const;
  std::span<const Ingredient> ingredients() const;
  std::span<const std::uint32_t> first_ingredient_of_material() const;  // For transfer and viscosity.
  std::span<const double> ingredient_viscosity() const;                 // Pa s. For viscosity::ViscosityInputs.
  std::span<const double> ingredient_friction() const;                  // No unit.
  std::span<const double> ingredient_rest_density() const;              // kg/m3, by scene ingredient index.
                                                                        //   Also gas::GasSceneSummary.
  std::span<const pressure::PressureLaw> ingredient_pressure_law() const;
  // NEW. Normalised names (lower case, only [a-z0-9_], unique), by scene ingredient index. Render output
  // exports the share of ingredient i as share_<ingredient_export_names()[i]>.
  std::span<const std::string> ingredient_export_names() const;
  // NEW. Scene ingredient index of the one ingredient of the gas material with the lowest material_index
  // (for gas::GasSceneSummary::gas_ingredient_index). Empty when no material is kGas. A second gas material is
  // reported by gas::CheckGasCanRun (kMoreThanOneGas), not here.
  std::optional<std::uint32_t> gas_ingredient_index() const;
  double surface_tension_max() const;       // N/m, largest over liquid ingredients. 0: no surface tension.
  double surface_tension_with_air() const;  // N/m, the one value all liquid ingredients share.
  double lightest_density() const;          // kg/m3, smallest rest density over liquid ingredients.
};
// Fails: kBadMaterial (5.1, including the gas rules), kBadExportName, kSurfaceTensionsDiffer, kBadInput
// (ingredient_export_names has the wrong size).
std::expected<MaterialTable, MaterialsFailure> BuildMaterialTable(const MaterialTableInputs& inputs);
// N/m at the boundary of two materials. 0 when the table has no pair for them.
double FindSurfaceTensionBetween(const MaterialTable& table, std::uint32_t material_a, std::uint32_t material_b);
// One entry per material: ingredient_count, ExtraFields and CountRule from the kind (5.1), max_particle_count
// as given. has_plate_side is set for every material when plates is kSome, and always for kSpringySolid.
std::vector<particles::MaterialParticlesSettings> BuildParticlesSettings(
    const MaterialTable& table, std::span<const std::size_t> max_particle_counts, PlatesInScene plates);
```

```cpp
// stretch_directions.h (shared)
struct StretchDirections {
  Mat3 directions_now = kIdentity;      // Rotation. Columns: the three stretch directions in world, as they
                                        //   point after the deformation. Textbook: U of F = U S V^T.
  Mat3 directions_at_rest = kIdentity;  // Rotation. Columns: the same directions before it. Textbook: V.
  Vec3 springy_stretch{1, 1, 1};        // D word. Length now / length at rest along each direction, positive.
};                                      //   Not sorted: the order the rotations leave. Textbook: singular values.
// Fails: kBadInput (not finite), kDeformationCollapsed (flat, mirrored, or no clean directions found).
MaterialsError ComputeStretchDirections(const Mat3& deformation, StretchDirections* directions);

// stretch_response.h (shared). NEW file: what a response computed inside the stretch directions returns.
// The sand module appends kShapeReduced = 2 and kPulledApart = 3 and sets kLimitCaseCount to 4.
enum class LimitCase : std::uint8_t {
  kInsideLimits,   // Nothing was limited: the motion was fully springy.
  kClamped,        // Snow: at least one stretch was clamped; the lost volume is permanent.
};
inline constexpr int kLimitCaseCount = 2;
struct StretchResponse {
  Vec3 springy_stretch{1, 1, 1};       // D. Inside the allowed region, same order as the input.
  double permanent_volume_change = 0;  // D. Natural log. Textbook: log plastic Jacobian.
  Vec3 internal_force_per_area{};      // D. Pa, along the three stretch directions. Textbook: Kirchhoff stress.
  LimitCase limit_case = LimitCase::kInsideLimits;
};

// snow_material.h (shared). File name NEW (was snow_and_sand_material.h; A9). The sand module adds
// sand_material.h beside it.
// stretch_before_limits: what the motion asks for (papers: trial stretches). *result untouched on failure.
// Fails: kBadInput (a stretch or exp(permanent_volume_change) not positive and finite), kBadMaterial, kNotFinite.
MaterialsError ComputeSnowResponse(Vec3 stretch_before_limits, double permanent_volume_change,
                                   const MaterialResponse& response, StretchResponse* result);

// springy_solid_material.h (shared). Pa, world, symmetric. Fails: kDeformationCollapsed, kBadMaterial, kNotFinite.
MaterialsError ComputeSpringySolidForcePerArea(const Mat3& springy_deformation, const MaterialResponse& response,
                                               Mat3* internal_force_per_area);
```

```cpp
// material_response.h (shared). The one route every material takes.
struct MaterialState {                  // What one particle remembers. Read from particles::ParticleFields.
  Mat3 springy_deformation = kIdentity; // No unit. Liquid, gas: unused. Springy solid: the whole deformation.
  double permanent_volume_change = 0;   // Natural log. Snow.
  double volume_ratio = 1;              // Current volume / rest volume, every kind. Snow:
};                                      //   det(springy_deformation) * exp(permanent_volume_change). Gas: 1.
struct InternalForce {
  Mat3 internal_force_per_area{};       // D. Pa, world, symmetric. Includes the factor volume_ratio: it pairs
                                        //   with rest_volume, never with current volume. Zero for liquid, gas.
  double sound_speed_bound = 0;         // m/s. At least the fastest springy wave. 0 for liquid, gas.
};
// From the stored state at the start of a step. Fails: kOutsideAllowedRegion is NOT checked here (see
// CheckMaterialState); kDeformationCollapsed, kMaterialUnstable, kBadInput (rest_density), kNotFinite.
MaterialsError ComputeInternalForcePerArea(const MaterialResponse& response, const MaterialState& state,
                                           double rest_density, InternalForce* internal_force);
struct MotionInStep {
  Mat3 exact_velocity_change{};         // D. 1/s. Used for `duration`.
  double duration = 0;                  // s. The whole step.
};                                      // The sand module adds the part after a hit (5.9).
struct MaterialUpdate { MaterialState state{}; LimitCase limit_case = LimitCase::kInsideLimits; };
// The state after the step. *update untouched on failure. Fails: kDeformationCollapsed, kNotFinite, kBadInput.
MaterialsError ComputeMaterialResponse(const MaterialResponse& response, const MaterialState& state_before,
                                       const MotionInStep& motion, MaterialUpdate* update);
Mat3 ComputeDeformationAfter(const Mat3& deformation, const Mat3& exact_velocity_change, double duration);
double ComputeVolumeRatioAfter(double volume_ratio, const Mat3& exact_velocity_change, double duration);
// m/s. Fails: kMaterialUnstable, kBadInput.
MaterialsError ComputeSoundSpeedBound(const MaterialResponse& response, const MaterialState& state,
                                      Vec3 springy_stretch, double rest_density, double* sound_speed_bound);

// own_motion_rule.h (shared)
struct OwnMotionShares { double own_motion_kept = 0; double own_motion_in_travel = 0; };  // Field words D.
// distance_to_surface: m, negative inside the liquid (D). distance_to_solid: m, positive outside.
OwnMotionShares ComputeOwnMotionInTravel(const Material& material, const Mat3& exact_velocity_change,
                                         double distance_to_surface, double distance_to_solid, double cell_width);
// No unit. Sum of fraction * friction over the ingredients of one particle.
double ComputeMixedFriction(const double* ingredient_fractions, const double* ingredient_friction,
                            std::uint32_t ingredient_count);

// support_rule.h (shared)
inline constexpr double kMinSupportShare = 1e-4;  // No unit. Same number as main's floor for equal particles.
struct FacePush {
  double velocity_change = 0;   // m/s, added to the face velocity.
  double applied_impulse = 0;   // N s = mass * velocity_change.
  double omitted_impulse = 0;   // N s = step_duration * force - applied_impulse.
};
FacePush ComputeFacePush(double force, double mass, double weight_sum, double step_duration,
                         double min_support_share);
```

```cpp
// material_buffers.h (CPU only; private _cpu.cc, _gpu.cu). Preallocated, reused every step.
struct MaterialSettings {
  double min_support_share = kMinSupportShare;  // No unit, zero or more. 0 switches the support rule off.
  double sound_step_share = 0.25;  // No unit, 0 < share <= 1. step <= share * cell_width / sound speed (main).
  double state_tolerance = 1e-12;  // No unit, relative. For CheckMaterialState.
};
class MaterialBuffers {
 public:
  // Pa. One pointer per material_index, null without internal force. For transfer::ParticlesToGridInputs.
  std::span<const Mat3* const> internal_force_per_area() const;
  const double* friction(std::size_t material_index) const;  // For contact::MoveParticlesInputs.
  const OwnMotionShares* own_motion_shares(std::size_t material_index) const;
};
// max_cell_blocks: room of the per-block budget buffer and the table copy (plan decision, transfer's precedent).
std::expected<MaterialBuffers, MaterialsFailure> BuildMaterialBuffers(
    RunOn run_on, const MaterialTable& table, std::span<const std::size_t> max_particle_counts,
    std::uint32_t max_cell_blocks);
std::vector<MemoryEstimateEntry> BuildMaterialsMemoryEstimate(const MaterialTable& table,
                                                std::span<const std::size_t> max_particle_counts);

// compute_internal_forces.h (+ _cpu.cc, _gpu.cu). Before ParticlesToGrid.
struct InternalForcesInputs {
  const MaterialTable& table;
  const particles::ParticlesTry& particles_try;
  grid::CellBlockRange blocks;
  double step_duration = 0;  // s.
  std::span<const double> smallest_cell_width;  // m, per material_index: smallest cell that holds it.
};
struct InternalForcesResult { double step_limit = 0; };  // s. +infinity when nothing has internal force.
// Fills internal_force_per_area for every particle of the range. Fails: kStepTooLong, kDeformationCollapsed,
// kMaterialUnstable, kWrongFields, kNotFinite, kBuffersTooSmall, kRunOnCallFailed.
std::expected<InternalForcesResult, MaterialsFailure> ComputeInternalForces(
    const InternalForcesInputs& inputs, const MaterialSettings& settings, MaterialBuffers& buffers);

// apply_internal_force.h (+ _cpu.cc, _gpu.cu). Part of stage ApplyForces.
struct InternalForceTotals {    // The sums of one cell block, or the total of several (5.8).
  double proposed_impulse = 0;  // N s. Sum over faces of |step_duration * force|.
  double omitted_impulse = 0;   // N s. Sum over faces of |omitted_impulse|. What the support rule left out.
  std::uint64_t weak_face_count = 0;  // Faces with force and weight_sum below min_support_share.
};
// velocity += ComputeFacePush(...) on every kFromParticles face of the range that has force. NEW: writes one
// budget per cell block of the range, totals_of_block[i] for block blocks.first_block + i; the span must hold
// blocks.block_count entries (else kBadInput). One form (A31): allocates nothing; buffers holds the per-block
// working memory. Requires: buffers and the arrays on one processor (asserted in debug builds; A30).
std::expected<void, MaterialsFailure> ApplyInternalForce(
    const grid::StaggeredGrid& grid, grid::CellBlockRange blocks, double step_duration,
    const MaterialSettings& settings, transfer::GridFromParticles& grid_from_particles,
    std::span<InternalForceTotals> totals_of_block, MaterialBuffers& buffers);
// NEW. math::TreeSum over the block budgets in the grid's fixed block order (integer sum for the count). The
// step keeps one entry per block of the whole grid, so one-piece and batched runs give identical bytes.
InternalForceTotals ComputeInternalForceTotal(std::span<const InternalForceTotals> totals_of_block);

// compute_own_motion_shares.h (+ _cpu.cc, _gpu.cu). After GridToParticles, before MixParticleVelocity.
struct OwnMotionInputs {
  const MaterialTable& table;
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;
  const particles::ParticlesTry& particles_try;
  std::span<const Buffer<transfer::GridAtParticle>> grid_at_particles;  // One per material_index.
  const grid::CornerValues<double>* distance_to_surface = nullptr;  // pressure::SurfaceDistances::at_corners.
  const grid::CornerValues<double>* distance_to_solid = nullptr;    // pressure::SolidCoverage. Null: no solid.
};
std::expected<void, MaterialsFailure> ComputeOwnMotionShares(const OwnMotionInputs& inputs,
                                                             MaterialBuffers& buffers);
// compute_particle_friction.h (+ _cpu.cc, _gpu.cu). Any time before MoveParticles.
std::expected<void, MaterialsFailure> ComputeParticleFriction(
    const MaterialTable& table, const particles::ParticlesTry& particles_try, grid::CellBlockRange blocks,
    MaterialBuffers& buffers);

// apply_material_response.h (+ _cpu.cc, _gpu.cu). In stage MoveParticles, after contact::MoveParticles.
struct MaterialResponseInputs {          // One material, one range of cell blocks. Borrowed pointers.
  const MaterialTable& table;
  std::size_t material_index = 0;
  particles::ParticleFields fields{};                    // id, rest_volume and the state arrays are read.
  const transfer::GridAtParticle* grid_at_particle = nullptr;  // exact_velocity_change is read.
  const contact::ParticleContact* contact_record = nullptr;      // kind is read. Null: no solid in the scene.
  double step_duration = 0;                              // s.
};                                       // The sand module adds the inputs of the hit split (5.9).
struct MaterialResponseResult {
  std::array<std::uint64_t, kLimitCaseCount> particle_count_by_case{};
};
// Writes volume_ratio for every kind; springy_deformation and permanent_volume_change where carried; clears
// kCarriesFittedChange on snow particles that hit a solid. Every particle of the range is written.
std::expected<MaterialResponseResult, MaterialsFailure> ApplyMaterialResponse(
    const MaterialResponseInputs& inputs, particles::ParticleFieldsToChange& to_change, MaterialBuffers& buffers);
// Every stored state of the range is one its material could have produced (5.4 step 5). For scene load and
// for a saved state that was just read. Fails: kOutsideAllowedRegion, kDeformationCollapsed, kWrongFields.
std::expected<void, MaterialsFailure> CheckMaterialState(
    const MaterialTable& table, const particles::Particles& particles, grid::CellBlockRange blocks,
    const MaterialSettings& settings);

// compute_cell_squeeze_values.h (+ _cpu.cc, _gpu.cu). Before pressure::BuildPressureEquations.
struct CellSqueezeInputs {
  const MaterialTable& table;
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;
  const transfer::CellsFromParticles& cells_from_particles;  // ingredient_volume is read.
  const grid::CellValues<double>* carried_pressure = nullptr;  // Pa. Null: zero everywhere.
};
// bulk_stiffness: Pa, +infinity where nothing can be squeezed. target_volume_ratio: no unit, 1 in an empty cell.
// One form (A31): reads the table copy held by buffers, allocates nothing. Requires: buffers and the arrays on
// one processor (asserted in debug builds; A30).
std::expected<void, MaterialsFailure> ComputeCellSqueezeValues(
    const CellSqueezeInputs& inputs, grid::CellValues<double>& bulk_stiffness,
    grid::CellValues<double>& target_volume_ratio, const MaterialBuffers& buffers);

// compute_work_rates.h (+ _cpu.cc, _gpu.cu). The test oracle of 5.8; the step may log it.
struct WorkRates {
  double from_grid = 0;       // W. Sum over faces of face_velocity * force.
  double from_particles = 0;  // W. Minus the sum over particles of rest_volume * (internal_force_per_area :
};                            //   exact_velocity_change). Equal to from_grid in complete neighbourhoods.
// Each sum: one partial per cell block, then math::TreeSum over the block partials (5.8).
std::expected<WorkRates, MaterialsFailure> ComputeWorkRates(
    const grid::StaggeredGrid& grid, grid::CellBlockRange blocks,
    const transfer::GridFromParticles& grid_from_particles,  // force is read.
    const grid::FaceValues<double>& face_velocity, const particles::ParticlesTry& particles_try,
    std::span<const Buffer<transfer::GridAtParticle>> grid_at_particles, const MaterialBuffers& buffers);
```

## 4. Consumes from other modules

**Foundation (EXISTS):** `Vec3`, `Mat3`, `kIdentity`, `operator*(Mat3, Mat3)`, `operator*(Mat3, Vec3)`,
`math::Dot`, `math::Cross`, `math::Transpose`, `math::Determinant`, `RunOn`, `CheckCanRunOn`,
`FLUIDITY_CPU_AND_GPU`, `cpu_only.h`, `Buffer<T>`, `BuildBuffer`, `ComputeTreeSum`, `ComputeLargestMagnitude`,
`MemoryEstimateEntry`, `math::TreeSum`, `math::Sqrt`, `Abs`, `Exp`, `Log`, `Sin`, `Min`, `Max`, `Length`, `IsFinite`
(three forms) (`shared-interface.md` (a)); `math::Expm1(double)` in `scalar_math.h` (`changes-to-apply.md`,
Foundation additions; the snow stiffening uses it, 5.3).

**Particles (PROVIDED, particles 3.1 to 3.3):** `ParticlesTry`, `Particles`,
`BuildFieldsToRead(particles_try, material_index, cell_blocks)` giving `ParticleFields` (`id`, `position`,
`mass`, `rest_volume`, `volume_ratio`, `ingredient_fractions`, `springy_deformation`,
`permanent_volume_change`, `flags`, `group_begin`, `cell_begin`); `ParticleFieldsToChange` with
`Field::kVolumeRatio | kSpringyDeformation | kPermanentVolumeChange | kFlags`; `kCarriesFittedChange`;
`MaterialParticlesSettings`, `ExtraFields`, `CountRule`, `kMaxIngredientCount`. Rest density of a snow or
springy particle is `mass / rest_volume`, as on `main`.

**Transfer (PROVIDED, transfer 3):** `GridAtParticle::exact_velocity_change`, `grid_velocity_at_particle`;
`GridFromParticles` (`force`, `mass`, `weight_sum`, `velocity`, `value_kind`, `step_try`), one per material
set (A8); `FaceValueKind::kFromParticles`; `CellsFromParticles::ingredient_volume`; `ParticlesToGridInputs::
internal_force_per_area` (this module fills it); `MixInputs::own_motion_kept`, `own_motion_in_travel`.

**Contact (PROVIDED, contact 3):** `ParticleContact::kind` (`ContactKind::kHit`), `MoveParticlesInputs::friction`
(this module fills it).

**Pressure (PROVIDED, pressure 3):** `PressureLaw`, `PressureLawKind`, `PressureLawState`,
`ComputePressureLawState`; `SurfaceDistances::at_corners`; `SolidCoverage::distance_to_solid`.
**Grid (PROVIDED, grid 3):** `StaggeredGrid`, `GridTables`, `CellBlockRange`, `CellValues`, `FaceValues`,
`CornerValues`, `FindCornerPlaces(tables, cell)`, `ComputeCornerPosition`, `ComputeCellWidth`, `ToSizeLevel`.
**Viscosity:** nothing consumed.

**Solids (PROVIDED, solids 3, 5.5, 5.7):**
1. `CollideSolids` applies `kNoMotionIntoSolid` to the faces of a set whose `MaterialSetFaces::at_solid_faces`
   says so (snow, springy solids) and leaves `kParticlesOnly` sets alone (`main`: `dry_solid_step_3d.cc:373-378`).
2. Snow and springy solids as seen by a liquid: `SolidsSettings::blocking_materials`
   (`MaterialTheLiquidCannotEnter`), `ComputeSolidCoverage`; inside the liquid's pressure solve (A18):
   `BlockingSet`, `BuildBlockingMaterialFaces` (stage 10), `ToBlockingMaterialFaces`, and the faces form
   `ApplyPressureToMaterials(const BlockingMaterialFaces&, velocity_after_pressure, blocking_sets,
   SolidsBuffers&)` returning `PressureOnMaterials` (stage 12), which writes each solved face velocity into the
   set's `transfer::GridFromParticles::velocity`; no push owed to particles (`MaterialPushFields` removed). In a
   viscous scene (A26) the set's faces near the liquid are instead solved in the viscous solve, starting from
   their velocity after `ApplyInternalForce` and `CollideSolids`, and written back by
   `solids::ApplyViscousSolveToMaterials` (stage 11). This
   module supplies no part of it; the ready scene's default is "every material with `HasInternalForce` blocks
   liquid".
3. `plate_side` assigned and cleared by `ApplyPlateSides`, `ApplyPlateRemoval`; `SolidSettings::friction_scale`.

**Gas (gas 3, 5.2):** nothing consumed. Provided to it here: the gas material and its one ingredient as 5.1
states; `MaterialTable::ingredient_rest_density()`; `MaterialTable::gas_ingredient_index()`; a gas particle's
`volume_ratio` is never changed (5.4 step 1).

**Step (PROVIDED, step 3, 5.2, 5.5):** `ReadyScene::materials` (`MaterialTableInputs`) and
`ReadyScene::material_settings`; the step builds the table once and passes `BuildParticlesSettings` to
`particles::BuildParticles`; per try it calls, in this order: `ComputeInternalForces` and
`ComputeParticleFriction` (stage `ParticlesToGrid`; `kStepTooLong` is a numerical reject with `step_limit` as
the next duration), `ApplyInternalForce` (stage `ApplyForces`, before gravity), `ComputeCellSqueezeValues`
(stage `BuildPressureEquations`), `ComputeOwnMotionShares` (stage `MoveParticles`, after `GridToParticles`),
`ApplyMaterialResponse` (stage `MoveParticles`, after `contact::MoveParticles`, one call per material and range;
`kDeformationCollapsed` is a numerical reject). It passes `smallest_cell_width` per material, keeps one
`InternalForceTotals` per cell block and reports `ComputeInternalForceTotal` in
`StepReport::internal_force`, adds `BuildMaterialsMemoryEstimate` to the estimate, and calls `CheckMaterialState`
in `BuildSimulation`. `kSurfaceTensionsDiffer` becomes `CapabilityReason::kLiquidSurfaceTensionsDiffer`.

**I/O and CLI (PROVIDED, I/O 5.1 steps 3 and 4):** `kMaterialPresets` (`io/material_presets.h`) gives defaults;
the scene file's materials, ingredients (each with its export name) and pairs become `MaterialTableInputs`;
stiffness and squeeze ratio map to the two stiffnesses; `own_motion_kept` maps to `OwnMotionSettings::kept`;
the gas ingredient's rest density is `gas::ComputeAmbientDensity` of the scene's gas law.

**Surface, foam and spray (PROVIDED, render output 4):** nothing consumed; it reads
`ingredient_export_names()` and `first_ingredient_of_material()`.

**Cell resize and batches (PROVIDED, batches 4):** nothing consumed; it calls this module unchanged, sums
`InternalForceTotals` per block as above, and reads `MaterialTable` for which materials keep their count.

**For the sand module** (`15-sand-design.md`): the extension points of 5.9. Nothing in this module tests or
depends on sand.

## 5. Method

Notation used only here: `F` springy deformation, `L` exact velocity change, `s` stretches, `e = ln s`,
`mu`/`lambda` the two stiffnesses, `dt` step duration, `V0` rest volume.

### 5.1 The table

| Kind | Internal force | Extra particle arrays | Count rule | Ingredients | Default `at_solid_faces` |
|---|---|---|---|---|---|
| `kLiquid` (one liquid or a mixture) | none | none (`ingredient_fractions` if N > 1) | may be split and merged | 1..8 | particles only |
| `kGas` | none | none | may be split and merged | 1 | particles only |
| `kSnow` | yes | `springy_deformation`, `permanent_volume_change` | keeps its count | 1 | no motion into solid |
| `kSpringySolid` | yes | `springy_deformation`, `plate_side` | keeps its count | 1 | no motion into solid |

`BuildMaterialTable` checks: ids nonzero and unique; ingredients of a material contiguous and in range;
`stiffness_against_shear > 0` and `stiffness_against_squeeze >= 0` when the kind has internal force;
`0 <= max_springy_shortening < 1`; lengthening and hardening `>= 0`; `0 <= in_travel <= kept <= 1`; viscosity,
surface tension, friction `>= 0`; rest density `> 0`; a kind with internal force has one ingredient with
`kCannotBeSqueezed`. NEW, the gas material (from the gas design, 5.2): one ingredient with viscosity 0,
friction 0, surface tension 0, `PressureLawKind::kCannotBeSqueezed`, and `own_motion` 0 and 0; its rest density
is the gas law's ambient density, set by I/O and checked here only as `> 0`. NEW, export names: each name is
lower-cased and every character outside `[a-z0-9_]` becomes `_`; an empty result or a repeat is
`kBadExportName`. Order of checks is material index, then field order, then ingredients in index order, then
export names in index order; the first failure is reported. Derived spans are built once.

### 5.2 Stretch directions (ported unchanged)

1. Reject a deformation that is not finite. Scale it by its largest absolute entry.
2. Determinant by the six-term expansion; fail with `kDeformationCollapsed` unless it exceeds
   `kMinDeterminantShare = 1.5e-14` times the sum of the absolute terms. This runs before any rotation,
   because rotations can turn an exactly flat deformation into a tiny positive stretch.
3. Up to `kMaxRotationSweeps = 16` sweeps over the column pairs (0,1), (0,2), (1,2): rotate the pair until
   their dot product is at most `kRotationTolerance = 2e-15` times the product of their lengths; apply the
   same rotation to `directions_at_rest`. Stop after a sweep that rotated nothing.
4. Stretch = column length times the scale; `directions_now` = columns divided by their lengths. Fail unless
   the three are perpendicular within `kDirectionTolerance = 1e-12` and right-handed.

### 5.3 Snow inside the stretch directions, and springy solids (ported unchanged)

Snow (method paper ref 19):

1. `s_i <- clamp(s_i, 1 - max_springy_shortening, 1 + max_springy_lengthening)` (`kClamped` if any changed).
2. `permanent_volume_change += ln(product before) - ln(product after)`, so springy volume times permanent
   volume is the volume the motion asked for.
3. `stiffer = exp(-hardening * math::Expm1(permanent_volume_change))`; `mu_now = stiffer * mu`, same for lambda.
4. With `J = product after`: `internal_force_per_area_i = 2 mu_now (s_i - 1) s_i + lambda_now (J - 1) J`.

Springy solid (papers: compressible neo-Hookean): `J = math::Determinant(F)`, fail with
`kDeformationCollapsed` unless `J > 0` and finite; `internal_force_per_area = mu (F F^T - I) + lambda ln(J) I`.
Fixed: the rounding bound on the determinant becomes the plain test (naming review 7.4).

World form for snow: `directions_now * diag(internal_force_per_area) * directions_now^T` and
`springy_deformation = directions_now * diag(springy_stretch) * directions_at_rest^T`, the triple product
summed in the fixed order of `main` (`Recompose`: first two terms, then the third).

### 5.4 One route: `ComputeInternalForcePerArea` and `ComputeMaterialResponse`

Both switch once on `response.kind`; no caller branches on the kind.

*Start of step.* Liquid, gas: zero force, bound 0. Snow: 5.2 on the stored deformation, then the force
formula of 5.3 (no limit is applied: a stored state is already inside its limits). Springy solid: 5.3. Then
5.5.

*End of step.*

1. Liquid: `volume_ratio *= exp(duration * (L.xx + L.yy + L.zz))`; fail if not positive and finite. Gas:
   state unchanged, `volume_ratio` stays 1 (confirmed by the gas design, 5.2).
2. Others: `F <- (I + duration * L) F`. `L` multiplies on the left (numerics report 9.2).
3. Snow: 5.2 on the new `F`, 5.3, rebuild `springy_deformation`.
   `volume_ratio = product of springy_stretch * exp(permanent_volume_change)`.
4. Springy solid: `volume_ratio = Determinant(F)`, which must be positive.
5. `CheckMaterialState` (not per step): run steps 3 or 4 with zero motion on the stored state; the largest
   entry change must be at most `state_tolerance` times the largest entry, and `permanent_volume_change` must
   not move by more than `state_tolerance * max(1, |value|)` (`main`: `MatchesDeformationState`).

Invariant: after any update, `volume_ratio` is the current volume over the rest volume for every kind, so
`transfer::CellsFromParticles::current_volume` has one meaning. Nothing is clamped to hide a collapsed
deformation: it is a failure the step answers by shortening (numerics report 9.2).

### 5.5 Sound speed bound and the step limit (ported unchanged)

`sound_speed_bound^2 = (largest row sum of the scaled stiffness blocks) / rest_density`. Snow with stiffened
`mu`, `lambda` and `J`: stretch rows `2 mu s_i^2 + lambda J^2 + 2 |lambda (2J - 1) J|`; pair rows from
`2 mu (1 - 1/(s_a + s_b)) s^2` and `2 mu s_a s_b / (s_a + s_b) - lambda (J - 1) J`. A row or a pair test
(`|coupling| - sqrt(shear_a shear_b) < sqrt(stretch_a stretch_b)`) that is not positive is
`kMaterialUnstable`. Springy solid: `(max(0, lambda (1 - ln J)) + mu (1 + largest absolute row sum of
F F^T)) / rest_density`; `lambda (1 - ln J) < -mu` is `kMaterialUnstable`. The full blocks go into the paper
section. `ComputeInternalForces` fails with `kStepTooLong` when
`step_duration * bound > sound_step_share * smallest_cell_width[material]`; `step_limit` is the smallest
`sound_step_share * width / bound` over the range. Stated limit: the pair tests cover the three principal
planes, not every oblique direction (as on `main`).

### 5.6 A particle that hits a solid

Snow and springy solids use `L` for the whole step, as on `main`. After any hit (`contact_record.kind ==
kHit`) a snow particle loses `kCarriesFittedChange` (`main` `dry_solid_step_3d.cc:570`). The split of the step
at the hit, with the derivative of the contact response after it, is the sand module's (5.9).

### 5.7 Travel share, friction, per-cell values

`ComputeOwnMotionInTravel`: `own_motion_kept = material.own_motion.kept` for every kind.
`own_motion_in_travel = material.own_motion.in_travel` for `kLiquid` when all three hold, else 0:
`L.xx + L.yy + L.zz > 0`; `-cell_width <= distance_to_surface <= 0`; `distance_to_solid > cell_width`
(ported from `liquid_substep_3d_cpu.cc:308`). Snow, springy solid, gas: 0 (on `main` they always travel with
the grid velocity; gas markers keep no own motion, gas design 5.2). The old behaviour for energy `e` is
`kept = 0.99 e`, `in_travel = 0.99 e^2`; no `0.99` exists here (owner Q33: answer a, recommended option). The
two distances are read at the particle by linear interpolation of the eight corner values of its active cell
(`cell_begin` gives the cell without a search); a null `distance_to_surface` gives share 0.

`ComputeParticleFriction`: `ComputeMixedFriction` per particle; a one-ingredient material writes its
constant (owner Q35: answer a, recommended option).

`ComputeCellSqueezeValues`, per cell with ingredient volumes `v_k` (m3) and `p = carried_pressure`: from
`pressure::ComputePressureLawState(law_k, p)` take `J_k` and `K_k`. `1 / bulk_stiffness = sum(v_k / K_k) /
sum(v_k)` (pressure step 14; +infinity when every `K_k` is infinite). `target_volume_ratio = sum(v_k) /
sum(v_k / J_k)` (each ingredient's rest volume is `v_k / J_k`; new for mixtures, equal to `J_EOS(p)` of
`main` for one ingredient). Sums run in scene ingredient order. A cell with no particle: +infinity and 1.

### 5.8 The support rule (replaces the force floor), and the work identity

Why a rule is needed: near the edge of a particle's reach the weight falls like the distance squared and its
change like the distance, so a face reached by one particle that barely touches it gets a force with almost
no mass under it. `main` floors the mass at `1e-4` of the lightest particle of the body.

The rule, stated: **a face takes the full internal force only when the particles reaching it give it at least
`min_support_share` of one particle's weight in total; below that the push is scaled down in proportion, and
the part left out is counted.** `weight_sum` is transfer's sum of normalised weights on the face (no unit).

```text
support         = min(1, weight_sum / min_support_share)        (1 when min_support_share is 0)
velocity_change = support * step_duration * force / mass
applied_impulse = mass * velocity_change        omitted_impulse = step_duration * force - applied_impulse
```

For particles of equal mass `mass = m * weight_sum`, so this is the floor of `main` with the same `1e-4`. What
changes: the rule is local (one face, no scan for the lightest particle), has no unit, and does not change
when an unrelated light particle joins the scene. What does not change, stated honestly: the omitted impulse
is real missing momentum. It is returned on every call (`InternalForceTotals`), never hidden, never turned
into another force; `main` measured it at most `1.3e-5` of the proposed impulse in its granular pile. A face
that is not `kFromParticles` gets no push and its whole impulse is counted as omitted. Order of the budget
sums (NEW, "sums over ranges"): per cell block, `math::TreeSum` over the faces the block owns in
`FacePlaceIndex` order, written to `totals_of_block`; the total is `ComputeInternalForceTotal`, a
`math::TreeSum` over the block partials in the grid's fixed block order, so the result is identical whether
the run is in one piece or in batches. `ComputeWorkRates` uses the same two-level order.

The work identity (test oracle; numerics report 9.3): for any face velocities `u`,
`sum_f u_f force_f = - sum_p V0_p * (internal_force_per_area_p : L_p[u])`, where `L_p[u]` is the exact
velocity change transfer reads from `u`. It holds to rounding wherever force and exact change use the same
faces (complete neighbourhoods, including across a size step). Next to a solid or a plate the force uses
same-side faces and the exact change uses filled faces: the two sides differ, and `ComputeWorkRates` reports
both so the difference is a number, not a claim. The identity is about the force before the support rule; the
rule's effect is the separate `omitted_impulse`.

Per material, in one place: liquids and mixtures carry `volume_ratio` and the table numbers; gas one
ingredient with the gas rules of 5.1 and `volume_ratio` 1; snow the two state arrays, the limits and the
support rule; springy solids the whole deformation and `plate_side` (set by solids, respected by transfer);
moving solids nothing here.

### 5.9 Extension points the sand module fills

Each is a closed list that sand appends to; no existing value, field order or call changes meaning:

1. `MaterialKind` gains `kSand` (value 4); `HasInternalForce` returns true for it; the table row of 5.1 and
   the checks of `BuildMaterialTable`; `BuildParticlesSettings` gives it `springy_deformation` and
   `permanent_volume_change` and `CountRule::kKeepsItsCount`.
2. `MaterialResponse` gains `friction_angle` and `cohesion` (decided words, A4).
3. `LimitCase` gains `kShapeReduced` (2) and `kPulledApart` (3); `kLimitCaseCount` becomes 4, so
   `MaterialResponseResult::particle_count_by_case` grows with it.
4. A new shared header `sand_material.h` beside `snow_material.h`, returning `StretchResponse`.
5. One more case in each switch on the kind: `ComputeInternalForcePerArea`, `ComputeMaterialResponse`,
   `ComputeSoundSpeedBound`, `CheckMaterialState`, and one more GPU instantiation per batch operation (6).
6. `MotionInStep` gains the part after a hit; `MaterialResponseInputs` gains the inputs of the hit split;
   `MaterialResponseResult` gains its counter.
7. Its `at_solid_faces` default is `kParticlesOnly`, a value that already exists.

## 6. CPU and GPU

Shared formula headers: `material.h`, `stretch_directions.h`, `stretch_response.h`, `snow_material.h`,
`springy_solid_material.h`, `material_response.h`, `own_motion_rule.h`, `support_rule.h`. Each batch operation
is the same formula in a loop.

- **CPU loop.** Per material, per cell block of the range, per particle in grouped order. No sums across
  particles except the result counters; threads may split by block because every particle writes its own
  entries.
- **GPU layout.** Nothing new per particle is committed state: the state arrays are the particles module's.
  `MaterialBuffers` owns, per material, on the simulation's processor: `internal_force_per_area`
  (`Buffer<Mat3>`, only with internal force), `friction` (`Buffer<double>`), `own_motion_shares`
  (`Buffer<OwnMotionShares>`), one `Buffer<double>` of sound bounds and one `Buffer<std::uint8_t>` of error
  codes. Per face place, one call: three `Buffer<double>` for the budget terms; per block, one
  `InternalForceTotals` returned to the CPU. The table goes down once as a small array of `Material` and
  `Ingredient` (export names stay on the CPU). Per call the CPU receives one status record and the counters.
- **Specialised per kind (CUDA report 8.4).** Each material has its own arrays, so each launch serves one
  kind: the private `_gpu.cu` instantiates the per-particle function once per `MaterialKind`, and no thread
  branches on the kind. Stretch directions, stretches and the new state of one particle stay in registers
  from read to write; nothing intermediate is stored. `ApplyMaterialResponse` reads `ArrayToChange::read` and
  writes `write` of the same index only, so the two may be the same memory.
- **Fixed order.** Per-particle results do not depend on order. The largest sound bound is
  `ComputeLargestMagnitude` (exact in any order); counters are integer sums; budget and work sums are a
  `math::TreeSum` per block, then a `math::TreeSum` over blocks (5.8). The reported failure is the lowest
  particle id (an integer minimum over the error codes), so it is the same on every run.
- **Cell blocks and border.** Every operation takes a `CellBlockRange` and touches only its particles,
  faces, cells. No border cell is read: the reach of a particle onto faces is transfer's concern.
  `ComputeOwnMotionShares` reads the corners of the particle's own cell only.
- **Memory estimate entries** (`BuildMaterialsMemoryEstimate`): 72 B per particle with internal force, 8 B friction, 16 B
  shares, 9 B working buffers, all `kOneStepTry`, `copies = 1`; 24 B per face place `kOneCall`; 24 B per cell block
  for the block budgets. `bulk_stiffness` and `target_volume_ratio` (8 B per cell place each) belong to the
  step.
- **Tolerance, CPU versus GPU** (`exp`, `log` differ in the last bits): state arrays
  `1e-13 + 1e-12 * |value|`; `internal_force_per_area` `1e-12 * (mu + lambda) + 1e-11 * |value|`; sound bound
  `1e-11` relative; `LimitCase` exact except within `1e-12` of a limit, where either case is accepted.
- Deferred, not designed: keeping the force from the end-of-step update for the next step (saves one stretch
  search per particle, costs an array that must follow regrouping). Performance tuning is out of scope.
- Status until the owner's hardware run: "GPU: compiled, not run".

## 7. Tests

**Analytic and conservation (CPU; all run in the container).**

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeStretchDirections, RotatedStretch_IsRecovered` | known rotations and stretches (0.5, 1, 2); also two equal stretches | rebuilt deformation equals input; stretches match as a set | `1e-13` relative |
| `..., Mirrored_IsRejected`; `..., Flat_IsRejected` | determinant -1; rank 2 | `kDeformationCollapsed`, output untouched | exact |
| `ComputeSnowResponse, InsideLimits_IsUnchanged` | uniform squeeze 0.99, limits 0.025 and 0.01 | same stretch, `kInsideLimits`, force by the formula of 5.3 | `1e-13` |
| `..., Clamp_KeepsTotalVolume` | stretches (0.9, 1.0, 1.05), limits 0.025 and 0.01 | springy product times `exp(permanent)` equals the asked product | `1e-13` |
| `..., PackedSnow_IsStiffer` | permanent -0.4, hardening 4 | force scaled by `exp(-4 * Expm1(-0.4))` | `1e-13` |
| `ComputeSpringySolidForcePerArea, Rotation_GivesZero` | `F` a rotation | zero force | `1e-13 mu` |
| `ComputeDeformationAfter, RateIsOnTheLeft` | `L` and `F` that do not commute | `(I + dt L) F`, different from `F (I + dt L)` by a stated amount | `1e-15` |
| `ComputeMaterialResponse, Liquid_VolumeFollowsTrace` | trace 2 1/s, 0.1 s | ratio times `exp(0.2)` | `1e-15` |
| `ComputeMaterialResponse, Gas_StateUnchanged` (NEW) | gas, any `L` | `volume_ratio` stays 1, no force | exact |
| `ComputeSoundSpeedBound, AtRest_IsRowSum` | snow and springy solid at rest | `sqrt((2 mu + 3 lambda) / density)` (snow); at least `sqrt((lambda + 2 mu) / density)` (both) | `1e-13` |
| `ApplyMaterialResponse, SnowHit_ClearsFittedChange` (NEW) | snow particles, one with `kHit` | that particle loses `kCarriesFittedChange`, the others keep it | exact |
| `ComputeFacePush, Budget_AddsUp` | weight sums 1, `1e-4`, `1e-6`, 0 | `applied + omitted == dt * force`; factors 1, 1, 0.01, 0 | exact / `1e-15` |
| `ApplyInternalForce, LoneParticle_PushIsBounded` | one particle sliding toward the edge of its reach of a face | velocity change stays bounded with the rule; with `min_support_share = 0` it exceeds any bound as the distance goes to 0 (asserted, to show why the rule exists) | bound from the closed form |
| `ApplyInternalForce, BudgetTotal_SameForAnyRangeCut` (NEW) | springy block over 12 blocks; one range, then ranges of 1, 5 and 6 blocks | `ComputeInternalForceTotal` gives identical bytes | exact |
| `ComputeWorkRates, CompleteNeighbourhood_Agree` | `main`'s case: one particle, symmetric force per area, `u = A x`; expected 87.5 W | `from_grid == from_particles == 87.5 W` | `1e-12` relative (`main`: `3e-16`) |
| `..., RandomFaceVelocities_Agree` | 64 particles, random `u`, uniform cells; again across a size step, no solid | the two rates agree | `1e-12` of the sum of absolute terms |
| `..., BesideSolid_DifferenceIsReported` | same beside a flat solid | both numbers returned; the difference is recorded in the test, not bounded | none |
| `ApplyInternalForce, NetImpulse_IsWithinOmitted` | free springy block, stretched | size of the summed applied impulse `<= omitted_impulse` | `1e-12` |
| `..., GridShift_ChangesLittle` (report 9.4) | same block shifted by 0.25 and 0.5 cell on each axis | momentum change after one step differs by less than the omitted impulse of the two runs | as stated |
| `..., SpringyBlock_EnergyNeverGrows` | stretched springy block set free, no solid, no gravity | energy `<= 1 + 1e-9` of the start; momentum drift `<=` summed `omitted_impulse` | as stated |
| Table tests | bad numbers, repeated ids, two tensions | `kBadMaterial`, `kSurfaceTensionsDiffer`; spans as specified | exact |
| `BuildMaterialTable, GasRules` (NEW) | gas with viscosity, friction, tension, a second ingredient, `kept > 0`, a squeezable law, each in turn; a correct gas | `kBadMaterial` with `ingredient_index` or `material_index`; then `gas_ingredient_index()` is its ingredient, empty without gas | exact |
| `BuildMaterialTable, ExportNames` (NEW) | `"Red Syrup"`, `"water-2"`; then `"A b"` with `"a_b"`; then `""` | `red_syrup`, `water_2`; `kBadExportName` at index 1; `kBadExportName` at index 0 | exact |
| `ToName, EveryValue` (NEW) | each `MaterialsError` | lower snake case of the enumerator without `k`, all distinct | exact |
| `ComputeCellSqueezeValues, TwoIngredients_ClosedForm` | volumes 1:3, stiffness 2 and 6 GPa, `p = 1e6` | harmonic mix; ratio from the two laws; all `kCannotBeSqueezed` gives +infinity and 1 | `1e-13` |
| `ComputeOwnMotionInTravel, Rule_TruthTable` | each of the three conditions off in turn; each kind | `in_travel` only for a liquid with all three | exact |
| Repeat run | every batch operation twice | identical bytes | exact |

**Reference piece (one, new):** `reference/materials/response.py`. It computes the snow response with
`numpy.linalg.svd` (a different algorithm from the Jacobi rotations) for 1,000 random rotated deformations
with a fixed seed, and the true fastest wave of snow and springy solids by sampling 10,000 directions of the
stiffness found by finite differences of the energy. Uncertainty removed that analytic tests do not: the
analytic rows use axis-aligned or hand-rotated cases, while the native path is a hand-written rotation search
feeding a branchy limit; and nothing analytic shows that the sound bound really is an upper bound for a
general state. Native agrees within `1e-10` relative; native bound `>=` sampled speed. The sand module adds
its response to the same file. **Lean proofs: none** (`main` has none for materials; the identities are
checked numerically above).

**Comparison with `main`** (supporting evidence): the snow cases of `native/tests/m5_constitutive_3d_cases.h`
and the cases of `m5_deformation_3d_cases.h`, rotated to Z-up, `1e-12` relative.

**GPU:** every per-particle row of the first table through the batch operations, within the tolerance of
section 6; `ComputeWorkRates` and `ApplyInternalForce` budgets within `1e-12` relative; the repeat-run test
(identical bytes on the same device); failure reporting returns the lowest particle id among three planted
bad particles. Without a GPU the container runs every shared-formula test and every batch test on
`RunOn::kCpu`, and compiles the `_gpu.cu` files.

## 8. Method paper section

`= Materials <sec-materials>` with `== Ingredients and materials <sec-material-table>`,
`== Stretch directions <sec-stretch-directions>`, `== Snow <sec-snow>`,
`== Springy solids that can be cut <sec-springy-solid>`, `== Force on the grid and the support rule
<sec-support-rule>`, `== The work identity <sec-work-identity>`, `== Sound speed and the step limit
<sec-sound-limit>`. The sand module adds `== Sand <sec-sand>` after `<sec-snow>`.
Equations `<eq:stretch-update>`, `<eq:snow-limit>`, `<eq:springy-force>`, `<eq:face-push>`,
`<eq:work-identity>`, `<eq:sound-bound>`.

Figures: `<fig-stretch-directions>` a unit cube at rest and the slanted box it becomes, with the three
directions before and after and the three stretches; `<fig-snow-limit>` the allowed box of snow stretches
with a clamped point and the volume that became permanent; `<fig-support-rule>` one particle near the edge of
its reach: weight and weight change against distance, their ratio growing, and the scaling of the rule;
`<fig-force-measures>` why the force per area pairs with rest volume (the factor `volume_ratio`).

Pseudocode: `<proc-apply-material-response>` (5.4, end of step), `<proc-snow-response>`,
`<proc-stretch-directions>` (5.2), `<proc-apply-internal-force>` (5.8).

Sources from the old reference list: [19] Stomakhin 2013 (snow), [3] Hu 2018 (springy solids with a cut;
force from the weight change), [18] Stomakhin 2014 (material response on a staggered grid beside a pressure
solve), [4] Fei 2021 (the travel rule), [1] Jiang 2015 (why the fitted change is not the exact change). No
entry exists for the neo-Hookean law, for the one-sided rotation search or for the row-sum sound bound: listed
as "citation needed", not written from memory.

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `shear_modulus`, `lame_lambda`, `first_lame_parameter` | `stiffness_against_shear`, `stiffness_against_squeeze` | DECIDED |
| `elastic_stretches` | `springy_stretch` | DECIDED |
| `log_plastic_jacobian` | `permanent_volume_change` | DECIDED |
| `kirchhoff_stress` | `internal_force_per_area` | DECIDED |
| `SnowParameters3D`, `DrySolidMaterial3D` | `MaterialResponse` with `MaterialKind` | DECIDED name; NEW shape |
| `LiquidEos3D` | `pressure::PressureLaw` | DECIDED (pressure) |
| `m5_constitutive_3d.h` | `snow_material.h` (its sand part: the sand module's `sand_material.h`) | DECIDED (A9) |
| (none) | `stretch_response.h` (`LimitCase`, `StretchResponse`) | NEW |
| `elastic_deformation`, `deformation_gradient` | `springy_deformation` | DECIDED (A4) |
| `reference_volume` | `rest_volume` | DECIDED (A4) |
| `volume_ratio` | `volume_ratio` | DECIDED (A4); VERIFIED one meaning at `liquid_substep_3d_cpu.cc:387`, `liquid_eos.h:20` |
| `strain` (log strain) | `log_stretch` (private local) | NEW (O6) |
| `trial_stretches` | `stretch_before_limits` (`trial` is a never-say word) | NEW (O6) |
| `Return...` (return mapping), yield surface | `Compute...Response`; "allowed region", "limit" | NEW (O6) |
| `stress` in other uses | `internal_force_per_area`; "push" only for the grid result (`FacePush`) | DECIDED / NEW |
| `plastic`, `elastic` | `permanent`, `springy` | DECIDED (A4) |
| `elastic` body, `ElasticBlade...` | `MaterialKind::kSpringySolid` | DECIDED (A4) |
| `compression_limit`, `stretch_limit`, `hardening` | `max_springy_shortening`, `max_springy_lengthening`, `hardening` | NEW |
| `PrincipalFrame3D { left, right, stretches }` | `StretchDirections { directions_now, directions_at_rest, springy_stretch }` | NEW |
| `FindPrincipalFrame`, `ReturnDeformation`, `TrialDeformation` | `ComputeStretchDirections`, `ComputeMaterialResponse`, `ComputeDeformationAfter` | NEW |
| clamp | `LimitCase::kClamped` | NEW |
| `...AcousticBoundSquared3D`, `wave_speed_squared`, `acoustic_cfl` | `ComputeSoundSpeedBound`, `sound_speed_bound`, `sound_step_share` | NEW |
| loss of ellipticity | `kMaterialUnstable` | NEW |
| `minimum_force_mass`, "force floor" | `min_support_share`, `ComputeFacePush`, "support rule" | NEW |
| `proposed_stress_impulse_l1`, `omitted_stress_impulse_l1` | `proposed_impulse`, `omitted_impulse` | NEW |
| (none) | `totals_of_block`, `ComputeInternalForceTotal` | NEW |
| `grid_power`, `particle_power`, `deformation_work` | `WorkRates::from_grid`, `from_particles` | NEW |
| `velocity_gradient` | `exact_velocity_change` | DECIDED |
| `particle_energy_blend`, `retained_velocity_fraction`, `beta` | `OwnMotionSettings { kept; in_travel; }` | DECIDED words |
| `substance`, `constituent`, `substance_id` | `ingredient`, `ingredient_id` | DECIDED |
| (none) | `ingredient_export_names`, `kBadExportName`, `gas_ingredient_index` | NEW |
| `phase_id`, `LiquidPhasePair3D` | gone; `MaterialPair` | owner Q31 (answer a, recommended option) |
| `dynamic_viscosity_pa_s`, `density_kg_per_m3` | `viscosity`, `pressure_law.rest_density` (unit in the comment) | DECIDED rule 7 |
| snow face clamp at the floor | `AtSolidFaces::kNoMotionIntoSolid` | NEW |
| `DrySolidStepBudget3D` | `InternalForceTotals`, `MaterialResponseResult` | NEW |
| (none) | `MaterialsFailure`, `ToName` | DECIDED (A2, stable error codes) |
| per-call scratch, one-call forms (the plans' wording, replaced) | working buffers in `MaterialBuffers`; one form per function | DECIDED (A31) |

## 10. Not ported

- `AdvanceDrySolid3D`, `CudaAdvanceDrySolid3D`, `StepElasticBladeGrid3D`, `AdvanceElasticBladeFrame3D` and
  their particle and checkpoint types (`dry_solid_step_3d.cc`, `elastic_blade_deformation_3d.cc`,
  `plastic_solid_checkpoint_3d.h`, `solid_interaction_checkpoint_3d.h`): the separate routes. Their maths is
  in 5.2 to 5.8; their stepping is the one step.
- Sand: the sand return of `m5_constitutive_3d.h:84-157`, the sand sound bound of `solid_acoustic_3d.h:39-92`,
  `ProjectStationaryFloorVelocityGradient3D` (`dry_solid_step_3d.cc:134-156`), the sand cases of
  `m5_constitutive_3d_cases.h` and `native/tests/test_m5_dry_scenes_3d.cc`: not in this module (A9); ported by
  the sand module (`15-sand-design.md`).
- The force floor `1e-4 * smallest particle mass` (`dry_solid_step_3d.cc:338`): replaced by 5.8.
- `compatible_floor` and its second deposit path, the hard-wired floor and platen (`:224-333, 93-118`):
  transfer has one path with same-side weights; solids are contact's and the solids module's.
- The retained-velocity experiment with its zero-net-momentum correction and `0.99`
  (`:446-496`): default on `main` is off; `OwnMotionSettings::kept` is the plain share.
- `PublishPrincipalState`'s "representable" checks and `ResolvedElasticVolumeRatio`'s rounding bound: one
  finite check and one positive-volume check (naming review 7.4).
- The roughly 350 sections of experiments in `docs/native/m5-qualification.md` and the temporary solver edits
  they describe (paired blends, midpoint kicks, wall-anchored fits): open experiments; the sand module reads
  their measurements for its diagnosis and nothing else.
- From the old paper, never implemented on `main`: the fixed-corotated option and "bounce" damping for
  springy solids; the travel rule for snow ("when the springy volume is at least 1 and out of contact");
  damage and fracture.
- `plastic_solid_boundary_3d.cc`, `mixed_contact_3d.cc`, `distinct_body_contact_3d.h`: not this module
  (solids, section 4; owner Q46).
- `surface_material_components_3d.h`: surface module. `DepositLiquidConstituentVolumes3D`,
  `MixLiquidConstituentVolumes3D`: replaced by `transfer::ComputeCellsFromParticles`.
- `reference/src/fluidity_reference/eos.py`, `multiphase.py`: pressure and particles decide; nothing here
  reads them.

## 11. Decisions

Depends on earlier questions: Q7 (material words) is owner answer A4; Q31 a (a material is what mixes), Q33 a
(materials owns the travel rule; `0.99` disappears), Q35 a (friction per particle times `friction_scale`),
Q42 a (snow and springy solids keep their count), Q46 (two-material contact is solids'): each the recommended
option (owner's standing instruction, 2026-11-14). Sand questions 4 and the sand part of 5 moved with sand to
`15-sand-design.md` section 11 (A9).

**1. The O6 words.** Proposed, following the decided pattern; all are in section 9.

```cpp
MaterialKind::kSpringySolid                    // was "elastic solid"; alternatives kStretchySolid, kSoftSolid
Vec3 stretch_before_limits;                    // was trial_stretches; alternative requested_stretch
LimitCase { kInsideLimits, kShapeReduced, kPulledApart, kClamped }   // was elastic / cone / apex / clamp
double friction_angle; double cohesion;        // kept; alternative grain_friction_angle, max_volume_gain
double max_springy_shortening, max_springy_lengthening, hardening;   // was compression_limit, stretch_limit
StretchDirections { directions_now; directions_at_rest; springy_stretch; }   // was principal frame U, V, S
volume_ratio, rest_volume, springy_deformation // kept as particles and pressure wrote them
```

`strain` survives only as the private local `log_stretch`; `stress` nowhere; `plastic` is `permanent`,
`elastic` is `springy`. Recommendation: as listed. Affects: particles, pressure, viscosity, solids, I/O.

Decision: as listed. `kSpringySolid`, `friction_angle`, `cohesion` and the other plain material words are owner
answer A4; `stretch_before_limits`, the `LimitCase` names, the snow limit names and `StretchDirections` are the
recommended option (owner's standing instruction, 2026-11-14). `kShapeReduced`, `kPulledApart`,
`friction_angle` and `cohesion` are declared by the sand module (A9).

**2. The support rule.** Options: (a) as designed in 5.8: scale the push by
`min(1, weight_sum / 1e-4)`, count what is left out; (b) leave weak faces out of each particle's force and
rescale its weights, so nothing is omitted and momentum is exact, at the cost of a second pass and a face
mask inside `transfer::ParticlesToGrid` (changes a finished design, and is not what `main` does); (c) no
rule: reject the try when a weak face would move too far (on `main` this stopped 4 of 12 runs).

```cpp
FacePush push = ComputeFacePush(force, mass, weight_sum, step_duration, settings.min_support_share);  // a
```

Recommendation: (a) now; (b) is the first experiment to run if rung 2 or the grid-shift test shows the
omitted impulse matters. Affects: transfer, step.

Decision: (a), recommended option (owner's standing instruction, 2026-11-14). (b) stays the first experiment
if `SpringyBlock_EnergyNeverGrows`, `GridShift_ChangesLittle` or the sand module's rung 2 shows the omitted
impulse matters.

**3. How a test that is expected to fail is recorded.** The commit rule wants every test to pass and forbids
a skipped test without a reason. Options: (a) the test asserts the right physics and is registered
`WILL_FAIL` with label `known_open`, plus a row in `docs/known-open-physics.md`; the suite turns red the day
it starts passing, which forces the record to be updated; (b) the test asserts the measured wrong numbers
(a tripwire) and a comment says so; (c) the tests are not in the suite and run by a separate command.
Recommendation: (a). It also serves "piston receiving volume" and "traction across cell sizes". Affects:
Foundation's gate, step, every module with open physics.

Decision: (a), recommended option (owner's standing instruction, 2026-11-14). This module has no such test
after A9; the sand module's diagnosis and solids' piston test use it.

**4. Is sand a usable capability while rungs 5, 6 and 8 fail?** Moved to `15-sand-design.md`, section 11,
question 1 (A9).

**5. Snow faces stop at solids (`AtSolidFaces`).** Ported as a field of the material, applied by solids.
Alternative: one rule for every material (never constrain faces; or always). Confirm the field and its name.
Affects: solids.

Decision: the field `Material::at_solid_faces` of type `AtSolidFaces { kParticlesOnly, kNoMotionIntoSolid }`,
recommended option (owner's standing instruction, 2026-11-14). Snow and springy solids default to
`kNoMotionIntoSolid`, liquid and gas to `kParticlesOnly`. The sand case is in `15-sand-design.md` (A9).

**6. One surface tension against air per scene.** `pressure::EquationInputs::surface_tension_with_air` is one
number. Two liquid ingredients with different values cannot be served; the table fails with
`kSurfaceTensionsDiffer`. Options: (a) keep that limit; (b) pressure takes a value per surface cell, mixed by
ingredient volume, and this module provides it. Recommendation: (a) for the first commits, (b) when a scene
needs it. Affects: pressure.

Decision: (a), recommended option (owner's standing instruction, 2026-11-14; `changes-to-apply.md`, 07).

**7. `math::Expm1` in `base/`.** Needed for snow stiffening (and the sand sound bound) without loss near zero.
Alternative: write `Exp(x) - 1` and accept a relative error of about `1e-16 / |x|` in a bound.
Recommendation: add it with the other scalar helpers. Affects: Foundation.

Decision: added to Foundation's `scalar_math.h`, recommended option (owner's standing instruction,
2026-11-14; `changes-to-apply.md`, Foundation additions).

**8. New names to confirm** (not obvious, not O6): `Ingredient` and `Material` as plain structs with the
decided `MaterialResponse` inside; `MaterialTable`; `MaterialPair`; `AtSolidFaces`; `OwnMotionSettings` and
`OwnMotionShares`; `FacePush`, `min_support_share`, `InternalForceTotals`, `omitted_impulse`;
`sound_speed_bound`, `sound_step_share` (default 0.25, the tested value of `main`); `WorkRates`;
`ComputeCellSqueezeValues`; the plural `ComputeInternalForces` for the batch form of
`ComputeInternalForcePerArea` (as contact has `MoveParticle` and `MoveParticles`); `kMaterialUnstable`,
`kDeformationCollapsed`, `kOutsideAllowedRegion`.

Decision: all as listed, recommended option (owner's standing instruction, 2026-11-14). The same applies to
the names added in this update: `stretch_response.h`, `snow_material.h`, `ingredient_export_names`,
`kBadExportName`, `gas_ingredient_index`, `totals_of_block`, `ComputeInternalForceTotal`.
`ComputeVelocityChangeAfterHit` moved to the sand module.

**9. Liquid pressure on snow and springy solids.** Section 4 (solids, item 2) read the liquid's push through
the particle form of `ApplyPressureToMaterials`, applied one step late (solids decision 3). Decision: the
blocking-material faces are terms of the liquid's pressure solve, built by `solids::BuildBlockingMaterialFaces`
in stage 10 and written back by the faces form of `ApplyPressureToMaterials` in stage 12; no materials
signature changes. Source: plan decision, 2026-11-14 (coupling inside the pressure solve), owner answer A18.

**10. One form per function; working buffers, not scratch (A31).** The plan gave `ApplyInternalForce` and
`ComputeCellSqueezeValues` two forms (one taking `MaterialBuffers`, one building a one-call copy for tests and
tools). Decision: one form each, taking `MaterialBuffers` last (section 3); tests build a `MaterialBuffers`
locally. `ComputeWorkRates` (test oracle) and `CheckMaterialState` (scene load) keep their one form, which holds
one call's GPU memory through `GpuArray` and may return `kOutOfMemory`: neither runs per step, and both are called
where no `MaterialBuffers` sized for them exists. Source: owner answer A31 (`owner-answers.md`, 2026-11-14);
materials plan Global Constraints, Tasks 17, 22, 24, 26.

**11. Arguments on different processors are a precondition (A30).** Decision: a particle view or an array whose
`run_on` differs from `MaterialBuffers::run_on()` is a broken precondition (`Requires:`, asserted in debug
builds), no longer `kBadInput`; the module maps no `kRunOnMismatch` (removed from `BufferError`). Source: owner
answer A30 (`owner-answers.md`, 2026-11-14).

**Open**
- The material's own volumetric stiffness inside the solve (half of the volumetric response in Fang et al.
  2020) is not built: `ApplyInternalForce` applies the full stress; building it needs this module to drop its
  volumetric part and material cells to get rows (plan-decisions, A18 "Open", owner question).
- ~~A springy solid (or snow) beside a viscous liquid: the step builds blocking faces in every scene with liquid
  and a blocking set, but viscosity's joint solve returns `kBadInput` for equations with material faces, so the
  pair fails today. Options: exclude the pair as a known limit, fall back to the one-step-late push for viscous
  liquids, or carry material faces in the joint solve. Open owner question; nothing decided here.~~ **Answered**
  (A24, A26): the viscous joint solve carries the material faces, which are velocity unknowns there and stick; no
  pair is refused.
