# Transfer module design

Status: decided. Module 3 of the rebuild. Sections 3 and 4 were aligned with the other module designs on
2026-11-14 and updated the same day with the owner's answers (`owner-answers.md`) and `changes-to-apply.md`;
names shared between modules are fixed in `shared-interface.md`.

## 1. Purpose

Particles carry the material; the grid is where forces, viscosity and pressure are computed. This module moves
velocity information between the two: `ParticlesToGrid` gives every grid face a mass, a momentum and (for
solids) an internal force summed from the particles near it, and `GridToParticles` gives every particle the
grid velocity at its position, before forces and after pressure, plus two descriptions of how that velocity
varies around it. `MixParticleVelocity` then decides how much of the particle's own motion survives.
Materials that do not mix are separate **material sets** (owner A8): every stage and `ComputeCellsFromParticles`
takes a material-subset input and runs once per set, so each set has its own face sums, snapshots and cell
sums, and a particle of one set never adds to another set's faces. All liquids are in one set, the liquid set
(one grid velocity for every liquid; liquids that do not mix are kept apart by pressure's material boundary);
the other sets hold snow, springy solids, or the gas markers.

Owns: the face sums and their fixed order; the material-subset rule; the two grid snapshots and their filled
faces; the same-side weight rule; the fitted and the exact velocity change; the mixing formula; the report of
every particle that had to follow the grid only.
Does not own: the weight curve and which faces are near a particle (grid); particle storage, ids and grouping
(particles); solid shapes, sides and the contact response (contact, solids); the contact rule between two
sets at a shared face (`solids::CollideSolids`); which materials form a set (I/O, carried by the step); forces,
viscosity, pressure; `MoveParticles` (step, see 5.7); the artist's "energy" control and the per-material rule
for the travel share (materials).

## 2. What exists today

All paths are on `main`. The transfer maths has no vertical axis: Y-up appears only in callers (gravity
`{0,-g,0}`, `floor_tags` in `dry_solid_step_3d.cc:298`) and in the fixed face order (blocks z,y,x; axes X,Y,Z).
A comparison with `main` must rotate vectors with `R = rows {1,0,0},{0,0,-1},{0,1,0}` and matrices as
`R M R^T`; old Y faces become Z faces, old Z faces become Y faces with the sign flipped and the index mirrored.

| Piece | Where | What it does | What is wrong |
|---|---|---|---|
| Uniform sums | `native/src/transfers_3d.cc:56-103` (`DepositWithFaceAccumulator`), `:139-197` | mass and momentum per face, 27 faces per axis, built in temporary arrays and copied on success | `:41-53` rounds each contribution to a 64-bit integer (fixed point, decision D7 removes it); needs a complete neighbourhood |
| Same-side sums | `liquid_boundary_transfer_3d.cc:329-460` (`DepositCore`) | keeps faces with `structurally_eligible && side_tag ==`, divides weights by their sum, quotient-rule gradient, offsets recentred on the weighted mean | fixed point again (`:438-455`); 1024-particle and 24576-face limits (`liquid_boundary_transfer_3d.h:16-17`); 5 KB of stored weights per particle (report CUDA 4.1) |
| Face velocity and kind | `liquid_boundary_transfer_3d.cc:462-516` (`NormalizeCore`) | velocity = momentum / mass; `kMissing`, `kLowMass`, `kValid` from `minimum_face_mass` in kg | a mass threshold in kg does not fit cells of mixed sizes or ingredients of different density; default is 0 |
| Solid internal force | `transfers_3d.cc:329-374`, `liquid_boundary_transfer_3d.cc:1024-1045`, `dry_solid_step_3d.cc:185-189` | `force = -rest_volume * (internal_force_per_area matrix row) . weight gradient` | three copies; one per-particle call that replaces the whole face array |
| Filled faces | `liquid_boundary_transfer_3d.cc:186-307` (`ContinueComponent`, `FillVelocity`), QR in `base/liquid_boundary_transfer_math_3d.h:34-144` | least-squares linear fit over same-side faces within 2 cell widths; solid normal correction; nearest face when the fit is rank deficient | recomputed for every (particle, face) pair; scans all faces (`:200`); elastic particles alone retry at 3 and 4 widths (`:855-879`) |
| Grid to particles | `mac_gather_3d_internal.h:35-67`, `liquid_boundary_transfer_3d.cc:518-651` | grid velocity before and after, fitted change (`:46`), exact change (`:47`), kept distinct | fitted change multiplies by `4 / cell_width` (`:49-56`), true only for a complete uniform neighbourhood (report numerics 2.1); `bool bounded` threaded through |
| Mixed sizes | `graded_mac_transfer_3d.cc:141-309`, `:524-560` | weights on face pieces, normalised, real 3x3 spread solved by Cholesky, velocity corrected by `fit . mean offset` | hides faces instead of filling them; returns no exact change (`liquid_adaptive_step_3d.cc:1685-1688` sets it to zero); a second path, `graded_quadratic_mac_transfer_3d.cc`, accepts only same-size islands |
| Mixing | `liquid_substep_3d_cpu.cc:246-254`, `:308-314`; copies at `liquid_substep_3d.cc:2530`, `cuda/liquid_boundary_transfer_3d.cu:566,602`, `dry_solid_step_3d.cc:476`, `transfers.cc:594` | `alpha = 0.99 * particle_energy_blend`; travel uses `beta * alpha` with `beta = particle_energy_blend` only where the liquid spreads near the surface | a blend named "energy"; the constant `0.99` in six places; the travel coefficient is a product of two settings (report numerics 2.3) |
| Snapshot order | `liquid_substep_3d.cc:1212-1236` (sums, then the old snapshot), `:1238-1246` forces, `:1247-1273` waves, `:2400-2452` new snapshot | first snapshot before any force; second after pressure, with face sides sampled at the end pose | the rule "after pressure, read only solved faces" is an optional scene mode (`:2428-2444`); no written contract (report numerics 2.4) |
| GPU | `cuda/transfers_3d.cu:44-96` (integer atomics), `cuda/liquid_boundary_transfer_3d.cu:707-790`, `cuda/liquid_transfer_stages_3d_host.cc` | 162 integer atomic adds per particle; one 96-thread block per particle for filled faces | order-free only because of the integers; every stage copies through host buffers (`liquid_transfer_stages_3d_host.cc:220-300`) |
| Reference, proof | `reference/.../transfers.py`, `mac_transport_3d.py`, `proofs/Fluidity/UniformTransfer.lean` | NumPy fit and derivative on a complete uniform grid; proofs of the quadratic weights and of reproduction "given complete moments" | describe only the complete uniform case |

Verified reading of two words this module consumes. `structurally_eligible` is set false exactly where a
face centre is inside a solid (`liquid_wall_contact_3d.cc:675-679`) and never depends on mass, so it is
named `outside_solid` (decision 6). `side_tag` is `0` with no plate and `-1`/`+1` on the two sides
of one plate (`liquid_substep_3d.h:407`); it is compared only for equality.

## 3. Public interface

Folder `native/fluidity/transfer/`, namespace `fluidity::transfer`. Every header below has `_cpu.cc` and
`_gpu.cu` beside it unless marked "formula" (header only, `FLUIDITY_CPU_AND_GPU`, C++20 subset).
`Mat3` entries of a velocity change: first letter is the velocity axis, second the position axis (`xy` is the
change of x velocity per meter of y).

```cpp
// transfer_error.h (formula)                                                        all names NEW
enum class TransferError : std::uint8_t {
  kNone,                 // No error. For shared formulas that report through a TransferError*.
  kBadInput,             // A value is not finite or out of range (negative mass, a share outside 0..1).
  kWrongGridVersion,     // An array was built for another grid version or another step try.
  kParticlesNotGrouped,  // Particles are not grouped by cell block at their current positions.
  kNoFaceOnSameSide,     // On one axis, no nearby face is on the particle's side. Active cells are missing.
  kTooManySidesAtFace,   // Particles on more than kMaxSidesAtFace different sides reach one face.
  kFaceCannotBeFilled,   // A face needs a filled value but has no source face and no solid velocity.
  kWrongStageOrder,      // A snapshot of this step try is missing, or it was saved at the wrong time.
  kWrongMaterialSet,     // NEW. An input array was written for another material set than material_is_counted.
  kNotFinite,            // A sum or a result overflowed.
  kRunOnCallFailed,      // CheckCanRunOn failed for the chosen processor, or a call to it failed.
  kOutOfMemory,          // NEW (A58; reconcile item 31). Last value. Only the builders: the run's RunMemory or
                         //   the processor could not hold a buffer; magnitude = bytes asked.
};
struct TransferFailure {
  TransferError error;
  std::uint64_t particle_id;  // Stable id, or 0 (no particle has id 0).
  grid::FacePlaceIndex face;  // Valid for one grid version only, or grid::kNoIndex.
  Axis axis;
  double magnitude;           // The offending value, in its own unit. Zero when not relevant. For a bad
                              //   material_is_counted: the material_index at fault.
};
// transfer_message.h (CPU only)
std::string_view ToMessage(TransferError error);
// NEW. Stable code: the enumerator's name without the k, in lower snake case ("wrong_material_set").
std::string_view ToName(TransferError error);
// material_set.h (CPU only)                                                              all names NEW
// A material set: the materials that share one grid velocity. Given to every stage below as
// material_is_counted: host memory, exactly one entry per material_index of the ParticlesTry, each 0 or 1, at
// least one 1. The sets of a scene are disjoint and together hold every material. Anything else is kBadInput.
struct MaterialSetKey {                // Recorded by every output of a stage; the next stage compares it.
  std::uint32_t lowest_material_index; // Lowest counted material_index. Unique because sets are disjoint.
  std::uint32_t counted_material_count;
  friend bool operator==(const MaterialSetKey&, const MaterialSetKey&) = default;
};
std::expected<MaterialSetKey, TransferFailure> ToMaterialSetKey(std::span<const std::uint8_t> material_is_counted,
                                                                std::size_t material_count);
// Checks the scene's sets once (capability check). Fails with kBadInput and the first material_index, in
// ascending order, that is counted in no set or in two sets, or whose entry is not 0 or 1.
std::expected<void, TransferFailure> CheckMaterialSets(
    std::span<const std::span<const std::uint8_t>> material_sets, std::size_t material_count);
// transfer_settings.h (formula)
inline constexpr int kMaxSidesAtFace = 2;                     // NEW. One plate. See decision 5.
inline constexpr std::array<double, 3> kFillRadiiInCells{2.0, 3.0, 4.0};  // NEW. Tried in this order.
inline constexpr double kFillPivotTolerance = 1e-10;          // NEW. Ratio to the largest pivot (main).
// The spread's pivot floor is grid::kMinPivotShare (1e-12 of the trace, the same number from main); this
// design first declared it again as kSpreadPivotTolerance.
struct TransferSettings {                                     // NEW
  double min_weight_sum_at_face = 0;  // No unit, zero or more. At or below it a face has too few particles.
};
// grid_from_particles.h
enum class FaceValueKind : std::uint8_t {                     // NEW
  kNoValue,            // No particle reached the face and no solve wrote it.
  kFromParticles,      // Enough particle weight: velocity = momentum / mass.
  kTooFewParticles,    // Some weight, at or below the minimum. The velocity is not trusted.
  kFromSolve,          // No particle weight, but a solve (pressure::SolvePressure,
                       //   viscosity::SolveViscosityAndPressure, or gas::SolveGas for the gas set) gave the
                       //   face a velocity.
  // Not in the first version (A64): kFromGridOnlyRegion, the grid-only region's velocity
  //   (cell_resize::ApplyGridOnlyRegionToGrid, in the set holding the liquid), a source face read directly like
  //   kFromParticles. The later group "Deep liquid without particles" appends it as the fifth value.
};
struct SidesAtFace {                                          // NEW
  std::array<contact::PlateSide, kMaxSidesAtFace> side;  // Sides of the particles that reach this face.
  std::uint8_t count;                                 // In order of first appearance in the fixed order.
};
struct GridFromParticles {                                    // NEW. Buffers, valid for one step try. One
                                                              //   instance per material set; sums of that set only.
  grid::FaceValues<double> mass;            // kg.
  grid::FaceValues<double> momentum;        // kg m/s, along the face's axis.
  grid::FaceValues<double> force;           // N, along the face's axis. Zero where no solid particle reaches.
  grid::FaceValues<double> weight_sum;      // No unit. Sum of the normalised weights the face received.
  grid::FaceValues<double> velocity;        // m/s. Zero unless value_kind is kFromParticles. Later stages
                                            //   (forces, viscosity, pressure) change it in place.
  grid::FaceValues<FaceValueKind> value_kind;
  grid::FaceValues<SidesAtFace> sides_reaching;  // Sides of this set's particles only.
  MaterialSetKey material_set;
  grid::GridVersion grid_version;
  std::uint64_t step_try;                   // Which try of which step wrote this. Zero means not valid.
};
// particles_to_grid.h                                                       ParticlesToGrid DECIDED
struct ParticlesToGridInputs {                                // NEW
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;                  // Faces owned by these blocks are written.
  const particles::ParticlesTry& particles_try; // Grouped. Counted materials only are read, border blocks too.
  std::span<const std::uint8_t> material_is_counted;  // NEW. The set (material_set.h).
  const SolidsAtFaces& solids_at_start;         // Sides and outside_solid at the start of the step.
  // Pa, one matrix per particle in grouped order. One entry per material_index, on the simulation's
  // processor; a null entry when that material has no internal force. Empty when no material has one.
  std::span<const Mat3* const> internal_force_per_area;
  std::uint64_t step_try;
};
struct ParticlesToGridResult {                                // NEW
  std::uint64_t face_count_from_particles;
  std::uint64_t face_count_too_few_particles;
};
std::expected<ParticlesToGridResult, TransferFailure> ParticlesToGrid(
    const ParticlesToGridInputs& inputs, const TransferSettings& settings, GridFromParticles& grid_from_particles,
    TransferBuffers& transfer_buffers);
// grid_snapshot.h                              SaveGridBeforeForces, SaveGridAfterPressure DECIDED
enum class GridTime : std::uint8_t { kBeforeForces, kAfterPressure };       // NEW
enum class FillKind : std::uint8_t {                          // NEW
  kNotNeeded,      // The side reads the face's own value.
  kFitted,         // Linear fit over source faces.
  kNearestSource,  // Too few source faces for a fit: the nearest one was copied.
  kSolidVelocity,  // No source face: the solid's velocity was used.
};
struct FilledVelocity {                                       // NEW
  contact::PlateSide side;
  double velocity;  // m/s, along the face's axis, for a particle on `side`.
  FillKind kind;
};
struct GridSnapshot {                                         // NEW. Read-only once saved. One per set and time.
  GridTime time;
  grid::FaceValues<double> velocity;                          // m/s.
  grid::FaceValues<FaceValueKind> value_kind;
  grid::FaceValues<std::uint8_t> is_source;                   // 1 when other faces may be filled from this one.
  grid::FaceValues<contact::PlateSide> side;                  // The face's own side at `time`.
  grid::FaceValues<std::uint8_t> outside_solid;               // From SolidsAtFaces at `time`.
  grid::FaceValues<std::array<FilledVelocity, kMaxSidesAtFace>> filled;  // One entry per side in sides_reaching.
  // After pressure only: the same, filled from solved source faces only (5.5). Empty before forces, and after
  // pressure when no solve wrote this set's faces (was_solved null).
  grid::FaceValues<std::array<FilledVelocity, kMaxSidesAtFace>> filled_from_solved;
  MaterialSetKey material_set;
  grid::GridVersion grid_version;
  std::uint64_t step_try;
};
struct SaveGridInputs {                                       // NEW
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;
  std::span<const std::uint8_t> material_is_counted;          // NEW. Must give grid_from_particles.material_set.
  const GridFromParticles& grid_from_particles;               // velocity, value_kind and sides_reaching are read.
  const SolidsAtFaces& solids;                                // At the snapshot's time (start pose, end pose).
};
std::expected<void, TransferFailure> SaveGridBeforeForces(const SaveGridInputs& inputs, GridSnapshot& before_forces,
                                                          TransferBuffers& transfer_buffers);
// velocity_after_pressure: m/s, the set's face velocity after the solve that wrote it (pressure::SolvePressure
// or viscosity::SolveViscosityAndPressure for the liquid set, gas::SolveGas through SolvedGasArrays for the gas
// set). For a set that no solve writes (snow, springy solids), the set's
// GridFromParticles::velocity after forces and solids::CollideSolids.
// was_solved: 1 for each face that solve corrected (pressure::ComputeWasSolved); null when no solve wrote the
// set's faces. A face with was_solved whose value_kind was kNoValue is recorded as kFromSolve in the snapshot;
// no solve writes a FaceValueKind. Other kinds are kept.
std::expected<void, TransferFailure> SaveGridAfterPressure(
    const SaveGridInputs& inputs, const grid::FaceValues<double>& velocity_after_pressure,
    const grid::FaceValues<std::uint8_t>* was_solved, GridSnapshot& after_pressure,
    TransferBuffers& transfer_buffers);
// filled_velocity.h (formula). For callers that need a value where no particle of the snapshot's set reaches
// (pressure, on the before-forces snapshot of the set holding the liquid).
struct FilledVelocityInputs {                                 // NEW
  grid::FacePlaceIndex face;
  contact::PlateSide side;
};
FLUIDITY_CPU_AND_GPU FilledVelocity ComputeFilledVelocity(    // NEW. kind is kNotNeeded .. kSolidVelocity;
    const grid::GridTables& tables, const GridSnapshotOnProcessor& snapshot,      //   error set on failure.
    const SolidsAtFacesOnProcessor& solids, const FilledVelocityInputs& inputs, TransferError* error);
// grid_to_particles.h                                                       GridToParticles DECIDED
enum class GridOnlyReason : std::uint8_t {                    // NEW. First that applies, in this order.
  kNone,
  kNewParticle,                    // Added this step: it has no own motion yet.
  kTooFewParticlesNearby,          // A nearby face is kTooFewParticles in either snapshot.
  kFilledWithoutFitBeforeForces,   // A nearby filled face is kNearestSource or kSolidVelocity.
  kFilledWithoutFitAfterPressure,
  kSpreadNotInvertible,            // The nearby faces do not span three directions on some axis.
};
inline constexpr int kGridOnlyReasonCount = 6;
struct GridAtParticle {                                       // NEW
  Vec3 grid_velocity_at_particle;  // m/s, after pressure.                                   DECIDED
  Vec3 own_motion;                 // m/s. Particle velocity minus grid velocity before forces at the
                                   //   particle. Zero when grid_only_reason is not kNone.   DECIDED
  Mat3 fitted_velocity_change;     // 1/s. Best linear fit of the nearby after-pressure face velocities.
                                   //   Papers: APIC matrix C. Zero when not carried.        DECIDED
  Mat3 exact_velocity_change;      // 1/s. Derivative of the interpolated after-pressure velocity.
                                   //   Papers: velocity gradient L.                          DECIDED
  GridOnlyReason grid_only_reason;
  bool carries_fitted_change;      // False when the reason is not kNone.                    DECIDED
};
struct GridToParticlesInputs {                                // NEW
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;              // Particles grouped in these blocks are written.
  const particles::ParticlesTry& particles_try;  // Same positions as in ParticlesToGrid.
  std::span<const std::uint8_t> material_is_counted;  // NEW. Both snapshots must be of this set.
  const GridSnapshot& before_forces;
  const GridSnapshot& after_pressure;
  // 1 per cell: particles in this cell read after-pressure values from solved faces only (5.5). From the
  // solve that wrote this set's faces; null when none did (as was_solved in SaveGridAfterPressure).
  const grid::CellValues<std::uint8_t>* cell_was_solved;
};
struct GridToParticlesResult {                                // NEW. Counts of this set's particles.
  std::array<std::uint64_t, kGridOnlyReasonCount> particle_count_by_reason;
};
// grid_at_particles: one buffer per material_index, in that material's grouped order, each at least as long
// as the material's particle count. Only the buffers of counted materials are read or written; the others may
// be empty.
std::expected<GridToParticlesResult, TransferFailure> GridToParticles(
    const GridToParticlesInputs& inputs, std::span<Buffer<GridAtParticle>> grid_at_particles,
    TransferBuffers& transfer_buffers);
// mix_particle_velocity.h (formula)                                       MixParticleVelocity DECIDED
struct MixInputs {                                            // NEW
  Vec3 grid_velocity_at_particle;  // m/s.
  Vec3 own_motion;                 // m/s.
  double own_motion_kept;          // 0..1. Share of own_motion saved on the particle.        DECIDED
  double own_motion_in_travel;     // 0..1. Share of own_motion used to move the particle. An effective
                                   //   share of own_motion, not of the kept part.            DECIDED
  GridOnlyReason grid_only_reason;
};
struct MixResult {                                            // NEW
  PerParticleVelocity<Vec3> velocity;           // m/s. for_next_step and for_this_step. In base/. A1
  bool used_grid_only;                          // True when a reason forced both shares to 0. DECIDED
};
FLUIDITY_CPU_AND_GPU constexpr MixResult MixParticleVelocity(const MixInputs& inputs) noexcept;
```

Added by the reconciliation of 2026-11-14 (all NEW). `SolidsAtFaces` is declared here, not in a later module,
so that transfer compiles before solids exists; the solids module fills it. One `SolidsAtFaces` per time
serves every material set. `CellsFromParticles` is option (a) of decision 9: pressure and viscosity both ask
for it.

```cpp
// solids_at_faces.h
struct SolidsAtFaces {                       // Per face at one time. Filled by solids::ComputeSolidsAtFaces.
  grid::FaceValues<std::uint8_t> outside_solid;                    // 0 when the face centre is inside a solid.
  grid::FaceValues<contact::PlateSide> side;                       // Side of the face centre. {0, kNoPlate}
                                                                   //   away from plates.
  grid::FaceValues<contact::SolidVelocitySample> solid_velocity;
  double time;                               // s.
  grid::GridVersion grid_version;
};
struct SolidsAtFacesOnProcessor;             // Plain pointers to the three arrays, for shared formulas.
struct GridSnapshotOnProcessor;              // The same for a GridSnapshot.

// cells_from_particles.h (+ _cpu.cc, _gpu.cu)
struct CellsFromParticles {                  // Buffers, valid for one step try. Zero in a cell with no particle
                                             //   of the set. One instance per material set.
  grid::CellValues<double> mass;             // kg. Counted materials and their ingredients together.
  grid::CellValues<double> rest_volume;      // m3.
  grid::CellValues<double> current_volume;   // m3. Sum of volume_ratio * rest_volume.
  std::uint32_t ingredient_count;            // Ingredients of the whole scene.
  Buffer<double> ingredient_volume;          // m3. [cell place index * ingredient_count + scene ingredient index].
                                             //   Zero for ingredients of materials outside the set.
  MaterialSetKey material_set;
  grid::GridVersion grid_version;
  std::uint64_t step_try;
};
struct CellsFromParticlesInputs {
  const grid::StaggeredGrid& grid;
  grid::CellBlockRange blocks;               // Cells stored in these blocks are written. No border is read.
  const particles::ParticlesTry& particles_try;
  // NEW. The set. The set holding the liquid leaves gas markers out, so they never change the liquid's
  // mixture density or sensed volume ratio (A10).
  std::span<const std::uint8_t> material_is_counted;
  // For each material_index, the scene ingredient index of that material's first ingredient (from materials).
  std::span<const std::uint32_t> first_ingredient_of_material;
  std::uint64_t step_try;
};
// Each particle of a counted material adds its amounts to the one active cell that contains it. Order of a
// cell's sum: counted materials by material_index, then that cell's particles in grouped order (ascending id).
// Particles of other materials are not read.
std::expected<void, TransferFailure> ComputeCellsFromParticles(const CellsFromParticlesInputs& inputs,
                                                               CellsFromParticles& cells_from_particles,
                                                               TransferBuffers& transfer_buffers);
// kg/m3 and no unit. Both return 0 for a cell with no particle. Shared formulas.
FLUIDITY_CPU_AND_GPU constexpr double ComputeMixtureDensity(double mass, double current_volume);
FLUIDITY_CPU_AND_GPU constexpr double ComputeSensedVolumeRatio(double current_volume, double rest_volume);

// transfer_buffers.h (CPU only). Working buffers reused by every set and stage in turn: 3 x grid::NearbyFaceSums
// per particle, the per-material offsets and the status record. Move-only. Owned by the caller (the step builds
// one at its limits; a test or tool builds one locally). Every stage takes it as its last parameter and
// allocates nothing (A31, section 11 item 12). Requires: on the same processor as the stage's arrays (asserted
// in debug builds; A30, section 11 item 13). A particle count above max_particle_count(), or more materials than
// material_count(), is kBadInput.
class TransferBuffers { public: RunOn run_on() const; std::size_t max_particle_count() const;
                        std::size_t material_count() const; };                      // NEW
// max_particle_count: all materials together; material_count: the scene's materials (the offsets hold one entry
// per material, reconcile item 27). Zero for either is kBadInput.
// Changed (A58): the buffers are pieces of run_memory (the run's block of run_on). Requires: run_on ==
// run_memory.run_on() (debug assertion). Fails kBadInput, kOutOfMemory (magnitude = bytes asked), kRunOnCallFailed.
std::expected<TransferBuffers, TransferFailure> BuildTransferBuffers(RunOn run_on, std::size_t max_particle_count,
                                                                     std::size_t material_count,
                                                                     RunMemory& run_memory);  // NEW
// NEW (A58; declared by the transfer plan, Global Constraints). The builders of the per-set arrays, each taking
// run_memory last, every array a piece of it; same failures as BuildTransferBuffers.
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

// transfer_memory_estimate.h (CPU only). The rows this module's buffers allocate from (base/memory_count.h).
// Rows of GridFromParticles, both GridSnapshots and CellsFromParticles carry copies = material_set_count;
// sets_without_solve is how many sets no solve writes (their filled_from_solved row is not allocated).
// Changed (A58): one row per array (a row that summed several arrays is split by member), so
// ComputeRunMemoryBytes of the rows equals the bytes the builders hand out.
std::vector<MemoryEstimateEntry> BuildTransferMemoryEstimate(std::uint32_t ingredient_count, std::size_t material_count,
                                               std::size_t material_set_count,      // NEW parameter.
                                               std::size_t sets_without_solve);     // NEW parameter.
```

Contracts common to the four stages and `ComputeCellsFromParticles`. Inputs are read-only. Each call works on
one material set: it reads only particles of counted materials, writes only that set's arrays, and records the
set's `MaterialSetKey`; an input array of another set is `kWrongMaterialSet`. A bad `material_is_counted`
(wrong length, an entry not 0 or 1, no 1) is `kBadInput`. A failed stage sets `step_try` of its output to
zero, so any later stage returns `kWrongStageOrder`; the committed state is never touched because the step
works on a trial copy. The reported failure is the first in a fixed order: by pass (5.2), then by stable
particle id or face index. Results are identical bytes for the same input on the same processor and build,
do not depend on how the grid is cut into block ranges, and do not depend on the other sets or on the order in
which the sets are run. No floating-point total over ranges is computed here; the result counts are integers.

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module,
with the name its section 3 has).

**Foundation (EXISTS):** `Vec3`, `Mat3`, `math::Dot`, `math::Transpose`, `Axis`, `RunOn`, `CheckCanRunOn`,
`FLUIDITY_CPU_AND_GPU`, `cpu_only.h`, `std::expected`.

**Foundation, A58 (EXISTS; Foundation Tasks 17 to 19, `base/run_memory.h`):** `RunMemory`, `BuildRunMemory`,
`BuildBuffer(RunOn, std::size_t, RunMemory&)`, the `RunMemory&` forms of `BuildSumBuffers` and `BuildOrderBuffers`,
`ComputeRunMemoryBytes(estimate, counts, MemoryWhere where)`, `BufferError::kRunMemoryTooSmall` (mapped to this
module's `kOutOfMemory`).

**Foundation additions (EXISTS, `shared-interface.md` section (a) and `changes-to-apply.md`):**
`template <typename T> struct PerParticleVelocity { T for_next_step; T for_this_step; }` (owner A1);
`Buffer<Value>` with `BuildBuffer`, `ReadBuffer`, `WriteBuffer` (per-particle outputs; this design first wrote
`particles::ParticleValues<T>`); `math::Sqrt`, `math::Abs`, `math::IsFinite`; `MemoryEstimateEntry` (its `copies` field
counts the material sets).

**Grid (PROVIDED by grid 3.1 to 3.7).** What this design first wrote, and the grid's declaration that serves it:

| First written here | Provided by grid |
|---|---|
| `StaggeredGrid` with `grid_version()`, `run_on()` | `grid::StaggeredGrid`, `version()` returning `grid::GridVersion`, `tables(run_on)` |
| `GridOnProcessor` | `grid::GridTables`, the plain form shared formulas receive |
| `CellBlockRange`, `FaceValues<T>`, `CellValues<T>` | the same names (grid 3.4, 3.6) |
| `face_index` (`std::uint32_t`), owned by one block | `grid::FacePlaceIndex`, ordered by (block, axis, place in block). Not dense: a place with `FacePieceKind::kNotAPiece` is skipped |
| `FaceWeight { face_index; weight (not normalised); weight_change; offset_from_particle; }` | `grid::FaceWeight { face; weight; offset; offset_from_mean; weight_change; }`, already scaled over the kept faces (grid 5.4, steps 5 to 7) |
| `NearbyFaceWeights { faces; count; }`, `ComputeNearbyFaceWeights(grid, axis, position)` | `FindNearbyFaces(tables, position, axis, near, &error)` gives a `NearbyFaceList`; transfer clears mask bits; `ComputeNearbyFaceWeights(tables, list, position, kept)`; `ComputeFaceWeight(tables, weights, position_in_list)`. `kMaxNearbyFaces` is 76 |
| `ComputeFaceWeight(grid, face_index, position)`, the weight seen from one face | `grid::ComputeFaceWeight(tables, face, position, cell, sums)` with `grid::NearbyFaceSums` (grid 3.7) |
| `CellRef`, `CellsNearFace`, `FindCellsNearFace`, `kMaxCellsNearFace` | `grid::CellsNearFace`, `FindCellsNearFace(tables, face)`, cells as `CellPlaceIndex`, `kMaxCellsNearFace = 112` (grid 3.7) |
| `FindFacesNearFace`, `kMaxFacesNearFace` | `grid::FaceSearch`, `BuildFaceSearch(tables, face, radius)`, `FindNextFace` (grid 3.7). A walk, not a fixed list |
| `ComputeFaceCenter`, `ComputeFaceAxis` | `grid::ComputeFacePosition(tables, face)`, `grid::ToAxis(face)` |
| `ComputeFaceCellWidth` | `ComputeCellWidth(tables.frame, ToSizeLevel(tables, ToCellBlockIndex(face)))` |
| `ComputeCellWidthAt(position)` | `FindCellAt(tables, position, near)`, then `ToSizeLevel` and `ComputeCellWidth` |
| `grid::SolidsAtFaces` | not the grid's: declared in this module (section 3), filled by solids |

The scaling this design describes in 5.1 is the grid's (5.4): transfer supplies only the mask. The arrays of
`GridFromParticles`, `GridSnapshot` and `CellsFromParticles` are built once per material set with the grid's
`BuildFaceValues` and `BuildCellValues` for `max_cell_blocks` and bound to each grid version with
`ApplyGridVersion` (grid, from the step's entry), so no stage allocates. Batched runs load the borders of
section 6 with `grid::BorderWidth`, `BuildRangeWithBorder` and `FindBorderBlocks` in the grid's fixed block
order.

**Particles (PROVIDED by particles 3.1 to 3.4).** `particles::ParticlesTry`, `material_count()`;
`BuildFieldsToRead(particles_try, material_index, cell_blocks)` gives a `ParticleFields` per material with
`id` (stable `std::uint64_t`), `position` (m), `velocity.for_next_step` (m/s), `mass` (kg, all ingredients
together), `rest_volume` (m3),
`volume_ratio`, `ingredient_fractions`, `fitted_velocity_change` (1/s), `flags` (`kCarriesFittedChange`;
`kHasOwnMotion` off is this design's `is_new`), `plate_side` (`contact::PlateSide`), `group_begin` and
`cell_begin`. Transfer never writes particles: the step copies results into `ParticleFieldsToChange`.
Grouping: particles of one cell block are contiguous and ordered by (cell place in block, stable id), with
the first particle of every cell place in `cell_begin` (particles 5.8). `CheckGrouping` and
`Particles::grid_version()` tell whether the grouping fits the grid and the current positions; otherwise this
module returns `kParticlesNotGrouped`.

**Contact (PROVIDED by contact 3.2).**

```cpp
namespace fluidity::contact {
struct PlateSide { std::uint32_t plate_id; SolidSide side; };  // Equal values mean same_side. {0, kNoPlate}: none.
struct SolidVelocitySample { bool has_solid; std::uint32_t solid_index; PlateSide side; Vec3 normal_out_of_solid;
                             Vec3 solid_velocity; double distance; };
}
```

This design first wrote `contact::SideId` (a 64-bit integer) and `contact::PerParticleVelocity`.

**Pressure (PROVIDED by pressure 3):** the face velocity after `pressure::SolvePressure`, written into the
array the step passes, and `pressure::ComputeWasSolved(equations, was_solved, cell_was_solved)` for the two
arrays this module reads (`grid::FaceValues<std::uint8_t>`, `grid::CellValues<std::uint8_t>`). Transfer
includes no pressure header and depends on no pressure hook (A5): the step passes the arrays, and transfer
itself records `kFromSolve` for faces either solve wrote. Pressure may call `ComputeFilledVelocity` on the
before-forces snapshot of the set holding the liquid.

**Viscosity (PROVIDED by viscosity 3):** in scenes with a viscous ingredient the velocity after the solve
comes from `viscosity::SolveViscosityAndPressure` (A5, A6); `was_solved` and `cell_was_solved` still come from
`pressure::ComputeWasSolved` on the equations that solve used.

**Materials (PROVIDED by materials 3):** `MaterialBuffers::internal_force_per_area()` (Pa, one `const Mat3*`
per `material_index` in grouped order, null without internal force); `MaterialTable::first_ingredient_of_material()`;
`OwnMotionShares` per particle from `ComputeOwnMotionShares` (`own_motion_kept`, `own_motion_in_travel`,
decision 7), which the step passes into `MixInputs`. The gas material passes shares 0 and 0.

**Solids (PROVIDED by solids 3):** `solids::ComputeSolidsAtFaces` fills `SolidsAtFaces` at the start pose
(elapsed 0) and at the end pose (elapsed `step_duration`). `solids::CollideSolids` reads each set's
`GridFromParticles` (`mass`, `value_kind`, `velocity` through `MaterialSetFaces`) and that set's
`CellsFromParticles::mass`, and changes the set's `velocity` in place before `SaveGridAfterPressure` of a set
no solve writes.

**Step (PROVIDED by step 3, after `changes-to-apply.md` "10 Step"):** holds the scene's material sets (from
I/O: which materials form a set) as one `material_is_counted` array per set, checks them once with
`CheckMaterialSets`; runs stages `kParticlesToGrid` (with `ComputeCellsFromParticles`),
`kSaveGridBeforeForces`, `kSaveGridAfterPressure`, `kGridToParticles` once per set; owns one
`GridFromParticles`, two `GridSnapshot` and one `CellsFromParticles` per set and the `Buffer<GridAtParticle>`
per material; calls `MixParticleVelocity` in `kMoveParticles` and writes the mixed velocities into the
particles' try.

**Gas (PROVIDED by gas 3, A10):** nothing is consumed. The gas material is its own set; gas reads the gas set's
`GridFromParticles::value_kind` and `velocity`, and nothing of the liquid set's `CellsFromParticles`.

**Cell resize and batches (PROVIDED by cell resize and batches 3):** the batch plan loads the borders of section
6. Not in the first version (A64): `cell_resize::ApplyGridOnlyRegionToGrid` writes face velocities
(`kFromGridOnlyRegion`), mass, `weight_sum` and cell amounts of the grid-only region into the `GridFromParticles`
and `CellsFromParticles` of the set holding the liquid, after `ParticlesToGrid`.

**For the sand module:** sand is a material set of its own (`solids::MaterialSetFaces` needs one
`AtSolidFaces` rule per set, and sand's differs from snow's); it uses `kFillRadiiInCells` like every material
(decision 4), so no sand-specific fill radius exists here.

## 5. Method

Notation used only here: particle `p`, face `f` of axis `a`, raw weight `w`, offset `r = x_f - x_p`.

**5.1 Same-side weights (ported from `DepositCore`, now the only path).** For one particle and one axis the
grid gives the nearby faces. A face is `same_side` for the particle when `outside_solid && face side ==
particle side`. Over the same-side faces only:

```text
S = sum w                      n_f = w_f / S                  (normalised weight, sums to 1)
G = sum (weight_change)        dn_f = (weight_change_f - n_f G) / S     (sums to zero)
mu = sum n_f r_f               rc_f = r_f - mu                (centred offset, weighted mean zero)
```

These are the grid's numbers (grid 5.4, steps 5 to 7): `S`, `G`, `mu` are `weight_sum`,
`raw_weight_change_sum`, `mean_offset` (`grid::NearbyFaceSums`), and `n_f`, `dn_f`, `rc_f` are
`FaceWeight::weight`, `weight_change`, `offset_from_mean`. Transfer builds the `NearbyFaceMask` from the
same-side rule and calls the grid; it does not repeat the arithmetic.

`S <= 0` on any axis is `kNoFaceOnSameSide`. Nothing here assumes 27 faces, equal face sizes or weights
that add up to 1, so one formula serves the complete uniform case (where `S = 1`, `G = 0`, `mu = 0` and the
numbers of `main` come back to rounding), domain sides, plates and size steps.

**5.2 `ParticlesToGrid`.** Passes, also the order in which failures are reported:

1. Check inputs: `material_is_counted` (5.8), grid version, grouping, and for counted materials only finite
   positions, velocities and fitted changes, `mass >= 0`.
2. Per particle of a counted material, per axis: compute `S`, `G`, `mu` (5.1) into the working buffers (`TransferBuffers`).
3. Per face of the range: visit the cells from `FindCellsNearFace` in their order, and in each cell the
   particles of counted materials in stable-id order. For each particle with positive `ComputeFaceWeight`:
   record its side in
   `sides_reaching` (first appearance; a third side is `kTooManySidesAtFace`); if `same_side`, add

   ```text
   mass     += m_p n_f
   momentum += m_p n_f (v_p,a + fitted_velocity_change_p (matrix row a) . rc_f)   [second term only if carried]
   force    += -rest_volume_p (internal_force_per_area_p (matrix row a) . dn_f)   [only with internal force]
   weight_sum += n_f
   ```
4. Same face, after its sums: `weight_sum == 0` gives `kNoValue`; `weight_sum <= min_weight_sum_at_face`
   gives `kTooFewParticles` and velocity 0; otherwise `kFromParticles` and `velocity = momentum / mass`.

Order of sums (the contract): for one face, one running double sum per source cell block, started at zero and
fed in (cell index in block, stable id) order; the block sums are then added in increasing block index,
starting at zero. With several counted materials (each has its own arrays in the particles module), a
block's partial is fed material by material in ascending `material_index` among the counted materials only,
each in (cell place, id) order; materials outside the set are skipped, not visited with weight zero. This
refines the programme's "ordered by stable particle id" (decision 1). It is written with one partial per block
so that a block-owned GPU scheme would produce the same bytes (6). The sequence of additions for a face of one
set is therefore exactly the one a try holding only that set's materials (in the same relative order) would
give.

Invariants: per axis and per set, `sum_f mass = sum_p m_p` and `sum_f momentum = sum_p m_p v_p,a` over the
set's particles and the faces they reach (because `sum n = 1`, `sum n rc = 0`); one particle's internal forces
add up to zero per axis (`sum dn = 0`); a particle adds to its own set's arrays only. Fixed: fixed point
removed; one path instead of three. New: `weight_sum`, `sides_reaching`, material sets.

**5.3 `SaveGridBeforeForces` and why it comes before forces.** It copies `velocity` and `value_kind`,
records each face's side and `outside_solid` at the start pose, marks sources, and fills faces (5.4).
`own_motion` is defined against this snapshot: `own_motion = v_p - (before-forces grid velocity at p)`. The
new velocity for the next step is `after-pressure grid velocity at p + own_motion_kept * own_motion`, so
everything the
grid did between the two snapshots (internal force, gravity, waves, solids, viscosity, pressure) reaches the
particle exactly once, whatever `own_motion_kept` is. Saved after gravity instead, a falling particle would
gain `(1 - own_motion_kept) g step_duration` per step instead of `g step_duration`. Ported unchanged
(`liquid_substep_3d.cc:1221-1246`).

**5.4 Filled faces (ported from `FillVelocity`, computed once per face and side).** A particle on side `s`
reads face `f` directly when `same_side` holds at the snapshot's time and `value_kind` is `kFromParticles`,
`kFromSolve` (later also `kFromGridOnlyRegion`, not in the first version, A64) (after pressure, also subject to 5.5). Otherwise it reads `filled[s]`.
For each face of the range and each side in `sides_reaching`:

1. Source faces: same axis, `is_source`, `outside_solid`, side `s`, centre within the radius (in widths of
   the filled face's cell), from `grid::FaceSearch` (`BuildFaceSearch`, `FindNextFace`) in its order.
2. Least-squares fit `velocity ~ c0 + c . offset` with offsets in cell widths, by Householder QR with column
   pivoting; a pivot below `kFillPivotTolerance` times the largest means rank loss. Radii 2, 3, 4 are tried
   in order until the fit has full rank (`kFitted`, value `c0`).
3. Rank loss at every radius: copy the nearest source face, ties to the lowest `FacePlaceIndex`
   (`kNearestSource`); with no source face, use the solid velocity if a solid of side `s` is at the face
   (`kSolidVelocity`); otherwise `kFaceCannotBeFilled`.
4. Solid correction: if a solid of side `s` is at the face and its normal has a part along the face's axis,
   build the velocity vector at the face centre (this axis from step 2 or 3, the other axes with a non-zero
   normal part fitted the same way, the rest from the solid), then remove the motion into or out of the solid:
   `value -= ((vector - solid_velocity) . normal_out_of_solid) * normal_out_of_solid (axis a)`. The solid
   velocity is the one sampled at the snapshot's own time.

`is_source` is 1 for `kFromParticles` and `kFromSolve` faces (and `kFromGridOnlyRegion`, not in the first version, A64); `kTooFewParticles` and
`kNoValue` are never sources. Sources are faces of the snapshot's own set: a face another set filled is never
a source here. Changed: radii 3 and 4 were for elastic particles only (decision 4); sources are found through
the grid instead of scanning every face.

**5.5 `SaveGridAfterPressure`.** Same procedure on the velocity after the pressure solve, with sides and
solid samples at the end pose. Additionally `is_source` requires `was_solved` wherever the face touches a
solved cell, and a particle whose cell has `cell_was_solved` reads directly only faces with `was_solved`;
its other faces are filled from solved sources. Particles outside solved cells (spray) read every usable
face. This is the rule `main` applies only in one scene mode (`liquid_substep_3d.cc:2428-2444`; it cut
volume drift with a moving solid from 2.11 % to 0.051 %, report CUDA 6.3), made unconditional (decision 3).
Each (face, side) therefore has up to two after-pressure entries; the second ("solved sources only") is kept
only for faces touching a solved cell. `was_solved` and `cell_was_solved` come from whichever solve wrote the
set's faces (A5: `SolvePressure` or `SolveViscosityAndPressure`, through `pressure::ComputeWasSolved`; for the
gas set, `gas::SolveGas` through `SolvedGasArrays`). For a
set no solve writes they are null: no face is solved, every usable face is read directly or filled once, and
`filled_from_solved` stays empty. A face that was `kFromGridOnlyRegion` keeps that kind when solved (not in the
first version, A64).

**5.6 `GridToParticles`.** Per particle of a counted material, per axis `a`, over all nearby faces `f` of
its set's two snapshots (each has a value, read directly or filled), with the normalisation of 5.1 taken over
all of them:

```text
before_a = sum n_f u_before_f        after_a = sum n_f u_after_f
spread   = sum n_f rc_f rc_f^T       (3x3, symmetric; weighted_spread in the naming review)
fitted_velocity_change (matrix row a) = spread^-1 (sum n_f u_after_f rc_f)
exact_velocity_change  (matrix row a) = sum u_after_f dn_f
grid_velocity_at_particle_a = after_a - fitted row a . mu
grid velocity before forces at p, axis a = before_a - (fit of u_before, same spread) . mu
```

The spread is solved by Cholesky in units of the particle's cell width; a pivot at or below
`grid::kMinPivotShare` times the trace is `kSpreadNotInvertible`: both fits are dropped and the two grid
velocities are the plain weighted means. Fixed: `main` multiplies by `4 / cell_width`, right only when
`spread = (cell_width^2 / 4) I`; here the real spread is always used, and the `mu` correction (from
`graded_mac_transfer_3d.cc:550`) makes a linear velocity field come back exactly at the particle even when
`mu != 0`. The exact change is the true derivative of the normalised interpolant; it equals the gradient of a
linear field only where `mu` does not vary (complete neighbourhoods), which the docstring states.
Then `grid_only_reason` is the first that applies; with a reason, `own_motion = 0`, the fitted change is zero
and `carries_fitted_change` is false. `exact_velocity_change` is always returned.

**5.7 `MixParticleVelocity`, and where `MoveParticles` lives.**

```text
for_next_step = grid_velocity_at_particle + own_motion_kept      * own_motion
for_this_step = grid_velocity_at_particle + own_motion_in_travel * own_motion
```

With a reason other than `kNone` both shares are zero and `used_grid_only` is true. Fixed by construction:
there is no `0.99` and no "energy" in this module; the two shares are independent effective values, so the
old behaviour is `own_motion_kept = 0.99 e`, `own_motion_in_travel = 0.99 e^2` where the travel rule holds and
0 elsewhere. The old liquid travel rule (`liquid_substep_3d_cpu.cc:308-310`: diagonal sum of the exact change
positive, `distance_to_surface` between minus one cell width and 0, more than one cell width from any solid)
is a per-material policy and moves with the other material rules (decision 7).

Decision: `MoveParticles` belongs to the step module, not here. It needs contact, the material update from
`exact_velocity_change` (materials), reactions on free solids (solids), `distance_to_surface` (pressure) and
the travel-distance reject (step); placed here it would force transfer to depend on four
later modules. Transfer hands the step everything as shared formulas (`MixParticleVelocity`,
`GridAtParticle`), so the step may fuse them into one GPU function without a new interface.

**5.8 Material sets (A8, A10).** A set is the materials that share one grid velocity; materials that do not
mix are in different sets. All liquids (and mixtures) are in exactly one set, the liquid set (the step's rule,
`ReadyScene::material_sets`; two liquids that do not mix in it are kept apart by pressure's material
boundary); the gas markers are another set; each snow or springy-solid object (or a group the artist chooses)
another. Which materials form a set is the
scene's (I/O, carried by the step); transfer only honours `material_is_counted`. Per try the step runs:

```text
for set in material_sets (any order; outputs do not depend on it):
  ParticlesToGrid(set) -> grid_from_particles[set];  ComputeCellsFromParticles(set) -> cells_from_particles[set]
  SaveGridBeforeForces(set) -> before_forces[set]
... forces, solids::CollideSolids (the contact rule between sets at shared faces), the solves ...
for set in material_sets:
  SaveGridAfterPressure(set, velocity of that set after its solve or after CollideSolids, was_solved or null)
  GridToParticles(set) -> grid_at_particles[m] for each counted material m
```

Rules. (1) A call reads particles of counted materials only; no weight of another set's particle is
computed, so it can neither add to this set's sums nor raise `kTooManySidesAtFace`, `kNoFaceOnSameSide` or
`kBadInput` here. (2) Sources, fills and directly read faces are the set's own (5.4). (3) `SolidsAtFaces` is
shared: one per time for all sets. (4) Order within a set: 5.2 for faces, the comment of
`ComputeCellsFromParticles` for cells; both skip uncounted materials, so a set's bytes equal those of a try
holding only that set's materials in the same relative order. (5) Every output records the set's
`MaterialSetKey`; a stage handed an array of another set returns `kWrongMaterialSet`. (6) `CheckMaterialSets`
(once, in the step's capability check) makes the sets disjoint and complete, so the lowest counted material
identifies a set. (7) The set holding the liquid leaves the gas markers out of its face and cell sums (A10):
markers change neither the liquid's face velocity nor its `mixture_density` and `sensed_volume_ratio`; the gas
set's faces carry the markers' velocity only. (8) (Not in the first version, A64.) `kFromGridOnlyRegion` appears only in the set holding the
liquid set. A scene with one set passes all ones and behaves as before material sets existed.

**Memory per extra set** (grid: 240 face places, 64 cell places per cell block; `I` scene ingredients):
`GridFromParticles` 58 B x 240 = 13 920 B; `GridSnapshot` before forces 52 B x 240 = 12 480 B; after pressure
86 B x 240 = 20 640 B (52 B x 240 = 12 480 B for a set no solve writes: `filled_from_solved` is not
allocated); `CellsFromParticles` (24 + 8 I) B x 64 = 1 536 + 512 I B. Total 48 576 + 512 I bytes per cell
block (about 49 kB, 40 416 + 512 I for a set no solve writes). Nothing per particle: `GridAtParticle` is per
material, and the `NearbyFaceSums` working buffers (in `TransferBuffers`) are reused by the sets in turn. The solids design's first estimate
(14 kB + 2 x 9 kB) is replaced by these numbers.

**Materials.** Liquids and mixtures: total mass and momentum; ingredient amounts stay on particles
(decision 8). Snow, springy and cut solids: also `force`, with the same-side derivative `dn`, so force and
exact change use the same derivative wherever both use the same faces; they sit in sets without liquid (5.8). Moving
solids: only through `SolidsAtFaces` at the two times. Gas: markers are their own set (5.8) and pass shares 0
and 0 (materials).

## 6. CPU and GPU

**Choice among the report's three GPU schemes: face-owned sums** (report CUDA 5.2). Each face has one
writer that visits its contributing particles in the fixed order. Reasons: (1) no floating-point atomics, no
partial buffers and no merge pass, so the order is the loop order and there is nothing to get wrong that
only hardware would show, which matters because device code is not run until the end (D4); (2) the whole
per-face sum is one shared formula, `ComputeFaceSums`, that the CPU implementation also calls, so the
container tests execute the exact code path of the GPU apart from the launch; (3) the same visit yields
`sides_reaching` for free. Cost accepted: each weight is computed twice (pass 2 and pass 3) and a face
tests more particles than contribute (about 1.3 times with per-cell particle ranges). The other two were
rejected for now: fixed chunk partials only matter when one face has thousands of contributors; block-owned
with border merge computes each weight once but needs an ordered accumulation inside the block, a partial
buffer per border face and a merge GPU function.
**Measurement that would overturn it:** on the owner's hardware run of the breaking-wave scene, if
`ParticlesToGrid` (passes 2 to 4) takes more than twice the time of `GridToParticles`, or more than 30 % of
the step, move to block-owned with border merge. The order contract of 5.2 (one partial per source block,
partials added by block index) was chosen so that this move changes no output byte.

**Shared formula headers (private):** `same_side_weights.h` (5.1), `face_sums.h` (`ComputeFaceSums`),
`linear_fit.h` (QR of 5.4), `fill_face.h`, `velocity_fit.h` (spread, Cholesky, 5.6), `grid_at_particle.h`
(`ComputeGridAtParticle`). Public formulas: `mix_particle_velocity.h`, `filled_velocity.h`.

**CPU loop shape.** Plain loops over particles (pass 2, `GridToParticles`) and faces (passes 3, 4, snapshots),
each calling the shared formula. Threads are allowed per destination: one face or one particle is written by
one thread, so results do not depend on the thread count.

**GPU layout.** Structure of arrays, doubles, all resident on the device; nothing is copied to the CPU
between stages.

| Array | Element | Owner | Bytes |
|---|---|---|---|
| `grid::NearbyFaceSums` working buffers (`TransferBuffers`) | 3 axes x (`S`, `G`, `mu`) = 21 doubles | transfer, valid inside `ParticlesToGrid` | 168 per particle |
| `GridFromParticles` | 5 doubles, kind, `SidesAtFace` | per material set; `velocity`, `value_kind` changed by later stages | 58 per face |
| `GridSnapshot` x 2 | velocity, kind, flags, side, 2 (4 after pressure) x `FilledVelocity` | per material set, read-only once saved | 52 to 86 per face |
| `CellsFromParticles` | 3 doubles + 1 per scene ingredient | per material set | 24 + 8 I per cell |
| `GridAtParticle` | 24 doubles, 2 bytes | `Buffer<GridAtParticle>` per material, owned by the step | 200 per particle |
| failure record | error, id, face, axis, magnitude | transfer, one per pass | 32 |

GPU functions: one thread per particle for pass 2 and for `GridToParticles`; one thread per face for passes
3 and 4 and for the fills; one launch per set and stage. The counted material indexes are read from the host
`material_is_counted` before launch and passed as a short list in ascending order; a face thread loops over
that list, so the device never reads the mask. The first failure is kept with an integer minimum on
(pass, id), which does not depend on thread order. Reason counts and face counts are integer sums. Returned to the CPU per call: the
result struct or the failure record, nothing else. Reading a buffer back is an explicit `ReadValues` on a
`FaceValues` or `ReadBuffer` on a `Buffer`.

**Ranges and border.** `ParticlesToGrid(blocks)` writes faces owned by `blocks` and reads the particle
groups of every block that `FindCellsNearFace` can name: the border is `grid::BorderWidth::kTouchingCells`
(the cells touching a face's two cells; `main`'s mixed-size weights reached two cell widths). The snapshot
stages write faces of `blocks` and read source faces up to 4 cell widths away
(`grid::BorderWidth::kFourCellWidths`). `GridToParticles(blocks)` writes particles grouped
in `blocks` and reads snapshot faces within the particle reach. A batched run must load those borders; the
one-piece run passes all blocks. Outputs are identical either way.

## 7. Tests

Analytic and conservation tests, CPU, `TEST(FunctionOrType, Situation_Outcome)`. "Scale" below means the
test's own magnitude: `M = sum of |contribution|` for a sum, `U` = largest speed, `h` = cell width.

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ParticlesToGrid, AnyCloud_ConservesMassAndMomentum` | 200 random particles over a size step and a plate, random fitted changes | per axis, face totals equal particle totals | `64 eps M` |
| `ParticlesToGrid, ConstantVelocity_EveryFaceHasIt` | all particles at `{1, -2, 3}` m/s | velocity at each `kFromParticles` face | `8 eps U` |
| `ParticlesToGrid, LinearFieldCompleteNeighbourhood_IsReproduced` | uniform cells, `v = a + B x`, fitted change `B` | face velocity `a + B x_f` | `1e-13 U` |
| `ParticlesToGrid, LinearFieldParticlesSizedToTheirCell_IsReproduced` (NEW, A65) | size-step grid; 8 particles per cell at its cell-eighth centres (`±w/4`), mass `1000 (w/2)³` (1 kg in the 0.2 m cells, 0.125 kg in the 0.1 m cells), `v = a + B x`, fitted change `B` | every `kFromParticles` face with centre in `x` `[0.4, 1.2]`, `y`, `z` `[0.2, 0.6]`: `(a + B x_f)_axis`; per axis face mass sum = particle mass sum | `1e-13 U`; `64 eps M` |
| `ParticlesToGrid, InternalForce_AddsUpToZeroPerParticle` | one particle, random symmetric matrix, with and without a plate | sum of face forces per axis is 0 | `64 eps M` |
| `ParticlesToGrid, CompleteNeighbourhood_ConservesAngularMomentum` | uniform cells, no plate | `sum x_f x q_f = sum m (x x v + part from fitted change, spread h^2/4)` | `1e-12` scale |
| `ParticlesToGrid, WeightSumThresholds_SetValueKind` | weight sums 0, at the minimum, above | `kNoValue`, `kTooFewParticles` (velocity 0), `kFromParticles` | exact |
| `ParticlesToGrid, StorageOrderChanged_SameBytes` / `SplitIntoTwoRanges_SameBytes` / `RunTwice_SameBytes` | same ids, shuffled storage; half ranges; repeat | identical bytes | exact |
| `ParticlesToGrid, NegativeMass_IsBadInputForLowestId` / `NoSameSideFace_...` / `ThreeSides_...` | two bad particles | error, lowest id, axis | exact |
| `SaveGridBeforeForces, LinearFieldAcrossPlate_FillIsExact` | side A holds `a + b . x`, side B holds 1e30 | filled value for A is `a + b . x_f`, kind `kFitted` | `1e-11 U` |
| `SaveGrid..., ObliqueSolid_NoMotionThroughIt` | normal `{0.6, 0, 0.8}`, solid velocity differs at the two times | `(vector - solid_velocity) . normal = 0`, each snapshot with its own time | `1e-12 U` |
| `SaveGrid..., CoplanarSources_CopiesNearest` / `NoSourceButSolid_...` / `Nothing_IsFaceCannotBeFilled` / `SourcesAtThreeWidths_FitsAtSecondRadius` / `TooFewParticlesFace_IsNotASource` | as named | kinds, tie to lowest `FacePlaceIndex` | exact kinds |
| `GridToParticles, LinearField_VelocityAndFitExact` | four cases: complete, size step, domain side, plate with fill | `a + B x_p`; fitted change `B` | `1e-12 U`; `1e-10 U / h` |
| `GridToParticles, ExactChange_IsDerivativeOfInterpolant` | non-linear field `u_x = x^2`, all four cases | central finite difference of `grid_velocity` without the `mu` term, step `1e-6 h`; and fitted differs from exact | `1e-6 U / h` |
| `GridToParticles, SameSnapshots_ParticleVelocityComesBack` | after = before | `grid_velocity_at_particle + own_motion = v_p` | `8 eps U` |
| `GridToParticles, FreeFall_GainsGravityOnceForAnyKeptShare` | cloud, add `step_duration * {0, 0, -9.81}` to Z faces between the snapshots, kept share 0, 0.5, 1 | change of the velocity for the next step `{0, 0, -9.81 step_duration}`; a control that saves the first snapshot after the force is off by `(1 - kept)` of it | `1e-12` of the change |
| `GridToParticles, EachReason_IsReportedAndCounted` | new particle; too-few face; nearest-source fill in each snapshot; cells one layer thick | reason, zero own motion, `carries_fitted_change` false, counts | exact |
| `GridToParticles, ParticleInSolvedCell_ReadsSolvedFacesOnly` | unsolved face with 1e30 next to a solved cell | result unaffected | `1e-12 U` |
| `MixParticleVelocity, Example_MatchesHandNumbers` | grid `{1,0,0}`, own `{0.2,0,0}`, kept 0.5, travel 0.25 | `for_next_step` `{1.1,0,0}`, `for_this_step` `{1.05,0,0}`; also `static_assert` | exact |
| `MixParticleVelocity, Reason_GivesGridVelocityAndFlag` / `TravelShareIndependentOfKept` | reason set; kept 0, travel 1 | both grid, flag true; `for_this_step = grid + own` | exact |
| `SaveGrid..., GridOnlyRegionFace_IsASourceAndKeepsItsKind` (not in the first version, A64: transfer plan later Task L1) | `kFromGridOnlyRegion` faces beside an unfilled face; one of them solved | read directly, used as a source, kind unchanged after pressure | exact kinds |
| `BuildTransferBuffers, IntoRunMemory_HandsOutExactlyTheEstimate` NEW (A58) | 2 sets (one without solve), 3 ingredients, 2 materials, 1,000 particles and blocks; CPU block of `ComputeRunMemoryBytes` of the estimate; every builder into it | `handed_out_bytes() == reserved_bytes()` | exact |
| `BuildGridSnapshot, RunMemoryOnePieceShort_IsOutOfMemory` NEW (A58, release builds) | a 256 B block, 8 blocks | `kOutOfMemory`, magnitude = the first piece's bytes | exact |
| `SaveGridAfterPressure, NoSolveForSet_EveryUsableFaceRead` | `was_solved` and `cell_was_solved` null, a 1e30 face that would be unsolved | `filled_from_solved` empty, no `kFromSolve`, particle reads every usable face as before forces | exact |
| `MaterialSets, TwoSetsApart_SameBytesAsSeparateRuns` | set A liquid, set B a springy block with internal force and a plate side, more than four cell widths plus the touching border apart, one grid; the four stages and `ComputeCellsFromParticles` per set on the try with both, then on a try holding only A (B's materials removed, indexes renumbered in the same order) and only B; sets run in both orders | identical bytes of `GridFromParticles`, both snapshots, `CellsFromParticles`, `GridAtParticle` and results (keys compared by material, not index) | exact |
| `MaterialSets, ParticleOfSetANeverAddsToSetB` | A and B interleaved in the same cells, on both sides of a plate; then A's mass, velocity, fitted change and internal force multiplied by 1e30, and a third side given to A | B's `GridFromParticles`, `sides_reaching`, snapshots and `CellsFromParticles` unchanged bytes; B's call reports no error | exact |
| `ComputeCellsFromParticles, GasMarkersLeftOut_LiquidSumsUnchanged` | liquid surface cells holding gas markers; liquid set's mask excludes the gas material | `mass`, `rest_volume`, `current_volume`, `ingredient_volume`, `ComputeMixtureDensity`, `ComputeSensedVolumeRatio` equal the run without markers; liquid `GridFromParticles` equal too | exact |
| `GridToParticles, SetMask_WritesOnlyCountedMaterials` | three materials, two counted; the third buffer filled with a byte pattern | the third buffer unchanged; counts only of the two | exact |
| `MaterialSets, BadMaskOrWrongSet_IsReported` | mask too short, an entry 2, all zero; set A's snapshot with set B's mask; `CheckMaterialSets` with a material in two sets and one in none | `kBadInput` with the material index; `kWrongMaterialSet`; `kBadInput`, lowest index first | exact |
| `TransferError, EveryValue_HasNameAndMessage` | every enumerator | `ToName` is the lower snake case name without `k`, unique; `ToMessage` not empty | exact |

**Python reference (one piece listed):** `reference/transfer/least_squares.py`, about 60 lines: the fill
fit and the fitted change through `numpy.linalg.lstsq` on explicit lists of weights, offsets and values.
It removes one uncertainty analytic tests cannot: that the hand-written pivoted QR and the Cholesky spread
solve return the least-squares answer for data that is not linear and for nearly rank-deficient face sets
(expected values there have no closed form). Compared within `1e-9 U`. `transfers.py` and
`mac_transport_3d.py` are not ported.
**Lean proofs: none.** The identities this module relies on (`sum n = 1`, `sum n rc = 0`, `sum dn = 0`,
linear reproduction given an invertible spread) are three-line algebra and are checked numerically on random
neighbourhoods above. `UniformTransfer.lean` proves the quadratic weight curve (grid's decision) and the
complete-neighbourhood case only.

**Comparison with the proof of concept (supporting only).** Uniform cells, no plate, 64 particles: run
`DepositMac3D` and `GatherMac3D` on `main`, rotate into Z-up (section 2), compare face totals per axis and all
per-particle outputs within `1e-12` relative plus `1e-12` scale. With one plate: `GatherLiquidBoundaryParticles3D`
against the filled values and gather outputs, `1e-9 U` (liquid radii differ only when radius 2 loses rank).

**GPU tests** (compiled here, run by the owner): every case above through `RunOn::kGpu`, compared with the
CPU within: face sums `1e-13 M`; face and particle velocities `1e-12 U + 1e-12 |value|`; fitted and exact
change `1e-10 U / h + 1e-10 |value|`; filled velocities `1e-9 U`; kinds, reasons, counts and the reported
failure exactly. Since both sides call the same formulas in the same order with contraction off, the expected
difference is zero and these bounds are deliberately loose. Plus `RunTwice_SameBytes`,
`StorageOrderChanged_SameBytes`, `SplitIntoTwoRanges_SameBytes`, `TwoSetsApart_SameBytesAsSeparateRuns` and
`ParticleOfSetANeverAddsToSetB` on the device.
**In the container without a GPU (`no_gpu`):** all shared formulas (they are the CPU tests), and
`ParticlesToGrid, GpuWithoutAGpu_IsRunOnCallFailed`. Only the launch, indexing and buffer code stays untested
until hardware.

## 8. Method paper section

Section `= Moving velocity between particles and grid <sec-transfer>`, subsections and labels:
`<sec-same-side-weights>` (5.1, `<eq:same-side-weights>`), `<sec-particles-to-grid>` (`<eq:face-sums>`,
`<proc-particles-to-grid>`), `<sec-grid-snapshots>` (why before forces, `<eq:own-motion>`),
`<sec-filled-faces>` (`<proc-fill-face>`), `<sec-grid-to-particles>` (`<eq:fit>`, `<eq:exact-change>`,
`<proc-grid-to-particles>`), `<sec-mix>` (`<eq:mix>`, replaces the old `<eq:asflip>`), `<sec-fixed-order>`,
`<sec-material-sets>` (one grid velocity per set, the per-set order, gas markers out of the liquid's sums).
Notation table rows added: `C` fitted_velocity_change, `L` exact_velocity_change, `D` weighted spread,
`alpha` own_motion_kept, `beta alpha` own_motion_in_travel.

Figures: `<fig-same-side-weights>` one particle next to a plate, its nearby faces on the three face sets,
faces across the plate greyed out, the mean offset `mu` drawn as an arrow; `<fig-grid-snapshots>` a time
line of one step with the two snapshots and a falling particle, showing gravity counted once;
`<fig-filled-face>` one face read by a particle across a plate, its source faces inside the radius, the
solid normal correction; `<fig-fixed-order>` one face, its source blocks, cells and particles numbered in
visit order, two partial sums joined. Pseudocode: the three procedures above, steps numbered as in 5.2, 5.4
and 5.6. Cite from the old list: [1] Jiang 2015 (fitted change), [2] Ding 2020 (one fit per staggered
axis), [3] Hu 2018 (same-side rule, plates), [4] Fei 2021 (own motion in the velocity for the next step and
in the one that moves the particle), [18] Stomakhin 2014 (solids on a staggered grid).

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| deposit, `DepositMacQuanta3D`, `DepositCompatibleMacQuanta3D` | `ParticlesToGrid` | DECIDED |
| gather, `GatherMac3D`, `GatherLiquidBoundaryParticles3D` | `GridToParticles` | DECIDED |
| `new_pic_velocity` | `grid_velocity_at_particle` | DECIDED |
| `particle_minus_old_pic_velocity` | `own_motion` | DECIDED |
| `fitted_velocity_slopes`, `particle_velocity_slopes` | `fitted_velocity_change` | DECIDED |
| `velocity_gradient` | `exact_velocity_change` | DECIDED |
| `affine_enabled` | `carries_fitted_change` | DECIDED |
| `0.99 * particle_energy_blend` (alpha) | `own_motion_kept` (effective value) | DECIDED; factor removed |
| `beta * alpha` | `own_motion_in_travel` (effective value) | DECIDED; VERIFIED product at `liquid_substep_3d_cpu.cc:313` |
| `old_pic_fallback`, `new_pic_fallback`, `low_mass_fallback` | `used_grid_only` + `GridOnlyReason` | DECIDED + NEW |
| `!pic_initialized` | `GridOnlyReason::kNewParticle` (particle flag `is_new`) | NEW |
| `old_` (history, time, walls) in transfer | `before_forces` | VERIFIED by reading `liquid_substep_3d.cc:1221-1246`: every `old_` here is before forces, none is "before pressure" |
| `new_` in transfer | `after_pressure` | VERIFIED `liquid_substep_3d.cc:2400-2452` |
| `LiquidBoundaryVelocityHistory3D`, stage `kOldSnapshot`/`kNewSnapshot` | `GridSnapshot`, `SaveGridBeforeForces`/`SaveGridAfterPressure` | DECIDED verbs, NEW type |
| ghost (a face a particle cannot read directly) | filled face; value `FilledVelocity` | VERIFIED `liquid_boundary_transfer_3d.cc:247-307`, with a correction: filling is per (face, particle side, snapshot), not a property of the face, and a solved face can still be filled for the other side. So "not solved" in the review's meaning is wrong. |
| donor | source face (`is_source`) | VERIFIED `:200-222`, with a correction: up to 36 source faces feed one fit; "the face it was filled from" is singular only for `kNearestSource` |
| continuation, sense 1 (`ContinueComponent`, `ContinuedVelocity`) | fill, `ComputeFilledVelocity` | VERIFIED |
| continuation, sense 2 (`kPredicted`: "explicit finite continuation ... without mass") | `FaceValueKind::kFromSolve` | VERIFIED `liquid_substep_3d.cc:2403-2405`: it is a face the pressure solve wrote, not a filled face. The review's single mapping does not cover it. |
| "wall continuation" | `FillKind::kSolidVelocity` | VERIFIED `:237-239` |
| `LiquidFaceValidity3D` `kMissing`/`kValid`/`kLowMass` | `FaceValueKind` `kNoValue`/`kFromParticles`/`kTooFewParticles` | NEW |
| `compatible` (`IsCompatible`) | `same_side` (and `outside_solid`) | DECIDED; VERIFIED `:93-95` |
| `structurally_eligible` | `outside_solid` (the first draft wrote `can_hold_liquid`) | review's Verify; VERIFIED `liquid_wall_contact_3d.cc:675-679`; decision 6 |
| `PerParticleVelocity::state`, `travel` (first draft) | `for_next_step`, `for_this_step` | A1 |
| separate liquid and snow grids (`dry_solid_step_3d.cc`) | material sets, `material_is_counted`, `MaterialSetKey`, `CheckMaterialSets` | A8; "material set" is the solids design's word |
| `side_tag` | `contact::PlateSide` | O2, contact decides; this design first wrote `contact::SideId` |
| `normalized_weight`, `normalized_weight_gradient`, `centered_face_offset` | private locals `normalised_weight`, `normalised_weight_change`, `centred_offset` | NEW (private) |
| `D`, moment, `centered_covariance` | `weighted_spread` | review's Verify; VERIFIED: in transfer every "moment" of offsets is this second moment. The first moment (`normalized_offset_velocity_moment`) becomes the private local `velocity_weighted_offset_sum`. |
| `minimum_face_mass` (kg) | `min_weight_sum_at_face` (no unit) | NEW, behaviour change (5.2) |
| `LiquidPrescribedWallSample3D` | `contact::SolidVelocitySample` | DECIDED |
| `stencil`, `sample`, `face_slot` | nearby faces, `FaceWeight`, `FacePlaceIndex` | DECIDED |
| `MacFaceDeposit3D`, `NormalizedLiquidMacFaces3D` | `GridFromParticles` | NEW |
| `MacParticleGatherResult3D`, `LiquidBoundaryParticleGather3D` | `GridAtParticle` | NEW |
| per-call scratch of the stages (the plans' `TransferScratch`, `BuildTransferScratch`, `transfer_scratch.h`, replaced) | `TransferBuffers`, `BuildTransferBuffers`, `transfer_buffers.h` (working buffers) | DECIDED (A31) |
| new names with no old one | `TransferError`, `TransferFailure`, `TransferSettings`, `SidesAtFace`, `GridTime`, `FillKind`, `MixInputs`, `MixResult`, `kMaxSidesAtFace`, `kFillRadiiInCells`, `kWrongMaterialSet`, `ToName`, `FaceValueKind::kFromGridOnlyRegion` (cell resize's request) | NEW |

## 10. Not ported

- Fixed-point sums and their guards: `MacDepositQuantization3D`, `MacFaceDepositQuanta3D`, `RoundNearestEven`,
  the `2^62` limits (`transfers_3d.cc`, `liquid_boundary_transfer_3d.cc`, `cuda/transfers_3d.cu`), and
  `proofs/Fluidity/FixedPoint.lean`. D7.
- 2D: `transfers.h`, `transfers.cc`, `basis.h`, `cuda/transfers.cu`, `cuda/transfer_host.cc`, `test_transfers.cc`. D8.
- `4 / cell_width` in place of the spread (`mac_gather_3d_internal.h:49-56`) and the uniform-only entry
  points `DepositMac3D`, `GatherMac3D`.
- `grid_velocity_increment`: equal to after minus before at the particle; no remaining reader needs it.
- Stored per-particle weight records (`CompatibleMacFaceStencil3D`, `DeviceMacParticleStencil3D`, 5 KB each)
  and the 1024-particle, 24576-face limits; weights are recomputed from the grid.
- `LiquidBoundaryWorkspace3D` and its paired "banks"; `bool bounded`; host staging in
  `cuda/liquid_transfer_stages_3d_host.cc`; the 96-thread per-particle fill block.
- `GradedQuadraticMacTransfer3D` (same-size islands only) and the hide-don't-fill gather of
  `GradedMacTransfer3D`; `MeasureSupportMoments`, `CheckUniformResamplingSupport` (split and merge checks
  belong to particles).
- `GatherElasticBoundaryParticle3D` as a separate entry (its radii become the common rule, decision 4).
- `DistributeCompatiblePressureImpulse3D` and `CutElasticWorkBudget3D`/`DepositCutElasticStress3D`: grid-to-
  particle impulse sharing and the cut work report belong to solids (one motion map and its transpose).
- `NeoHookeanKirchhoffStress3D`, `NeoHookeanAcousticSpeedBound3D` (in `transfers_3d.cc` by accident): materials.
- Whole-cell six-face travel field (`base/liquid_regular_cell_transport_3d.h`): a travel rule valid only in
  a stationary axis-aligned domain; it is `MoveParticles` policy and the step design must decide it.
- `CorrectIsolatedDropProjectedChange` (`liquid_substep_3d.cc:593-612`): scene-specific.
- Python `transfers.py`, `mac_transport_3d.py`, `transfer_3d_fixtures.py`; Lean `UniformTransfer.lean`.

## 11. Decisions

These questions were also merged, with the other modules' questions, in `owner-questions.md`. "Standing
instruction" below means: recommended option (owner's standing instruction, 2026-11-14).

1. **Order of the face sums.** The programme says "ordered by stable particle id". A face is reached by
   particles of up to 8 or more cell blocks; one global id order needs a merge of that many sorted lists per
   face. Options:
   (a) `for block (by index) { partial = 0; for particle (by cell, then id) partial += c; sum += partial; }`;
   (b) `for particle in merge_by_id(all blocks) sum += c;`. Both are repeatable. Recommendation: (a); it is
   also independent of batching and lets a block-owned GPU scheme give the same bytes later.
   Decision: (a), applied per material set over counted materials only (5.2, 5.8). Standing instruction
   (owner-questions Q21).
2. **`PerParticleVelocity::state` (your O3).** `state` also appears in `saved_state` and `state_version`.
   Options: `{ state; travel; }`, `{ carried; travel; }`, `{ next; travel; }`. Recommendation: `carried`
   ("the velocity the particle carries to the next step"). Contact declares the type; all modules follow.
   Decision: `{ for_next_step; for_this_step; }`. Owner answer A1.
3. **"Solved faces only" after pressure, for every scene.** `main` applies 5.5 only in one scene mode.
   Options: (a) always, for particles in solved cells; (b) never; (c) a setting. Recommendation: (a); a scene
   mode is an anti-goal and this rule is what removed the 2 % volume drift.
   Decision: (a), for faces written by either solve (A5); a set no solve writes passes null. Standing
   instruction (owner-questions Q23).
4. **Fill radii.** `main`: liquids try 2 cell widths then copy the nearest face; elastic particles also try
   3 and 4. Options: (a) `kFillRadiiInCells{2, 3, 4}` for every material; (b) `{2}` for all; (c) per
   material. Recommendation: (a); one rule, fewer grid-only particles, border of 4 cells for the snapshot stages.
   Decision: (a). Standing instruction (owner-questions Q24).
5. **`kMaxSidesAtFace = 2`.** Enough for one plate. Cutting with several plates meeting at an edge needs
   more (each extra side costs 17 bytes per face per snapshot). Options: 2 now and an error beyond; 4 now.
   Recommendation: 2, revisited in the solids design.
   Decision: 2 and `kTooManySidesAtFace` beyond; the solids design keeps one plate per scene. Standing
   instruction (owner-questions Q38).
6. **`can_hold_liquid` on faces, for all materials.** Verified meaning: "the face centre is not inside a
   solid". The same flag gates snow. Options: keep `can_hold_liquid`; `outside_solid`.
   Recommendation: `outside_solid`: `if (face.outside_solid && face.side == particle.side)`.
   Decision: `outside_solid`, in `SolidsAtFaces` and `GridSnapshot`. Standing instruction (owner-questions Q8).
7. **Who owns the travel-share rule and the artist control.** Transfer takes effective shares. Options:
   (a) materials: `materials::ComputeOwnMotionInTravel(material, exact_velocity_change, distance_to_surface,
   distance_to_solid)` and the scene maps one control to `own_motion_kept`; (b) step. Recommendation: (a).
   Also confirm that the factor 0.99 disappears as a constant: a scene that wants it sets
   `own_motion_kept = 0.99`.
   Decision: (a), as materials' `ComputeOwnMotionShares`; the factor 0.99 is gone. Standing instruction
   (owner-questions Q33).
8. **Too-few-particles rule.** `minimum_face_mass` in kg becomes `min_weight_sum_at_face` (no unit, "how
   many particle shares reached the face"), default 0 as on `main`. Confirm the change of unit, or keep kg.
   Decision: `min_weight_sum_at_face`, no unit, default 0. Standing instruction (owner-questions Q34).
   Amended (A65, 2026-11-15): every liquid cell holds about eight particles of its own size, so the weight sum at a
   face (a count of particle shares) is about the same at every level and one threshold means the same everywhere;
   heavier particles of coarse cells enter through `m_p` only.
9. **Ingredient amounts on the grid.** Pressure and viscosity need density and viscosity per cell for
   mixtures. Options: (a) transfer adds a cell-destination sum with the same order contract when those
   modules ask; (b) each module sums its own. Recommendation: (a), not in the first transfer commits.
   Decision: (a), declared in section 3 as `ComputeCellsFromParticles`, one per material set; each particle
   adds to the one active cell that contains it, and a cell with no particle of the set gets zeros. Standing
   instruction (owner-questions Q32).
10. **NEW names needing a yes:** `GridFromParticles`, `GridAtParticle`, `GridSnapshot`, `GridTime`,
    `FaceValueKind` and its four values, `FillKind`, `GridOnlyReason` and its values, `SidesAtFace`,
    `is_source`, `weight_sum`, `was_solved`; and "source face" in the plural (section 9).
    Decision: all accepted, with `FaceValueKind` now five values (`kFromGridOnlyRegion` added). Standing
    instruction (owner-questions Q14).
11. **Do different materials share one grid velocity?** (Solids question 1.) Options: (a) one velocity for
    all materials; (b) materials that do not mix are separate material sets, each with its own face sums.
    Decision: (b). Every stage and `ComputeCellsFromParticles` take `material_is_counted` and run once per
    set (5.8); gas markers form their own set and stay out of the liquid's sums. Owner answers A8 and A10.
    All liquids are in one set, the liquid set (step decision recorded in `changes-to-apply.md`).
12. **One form per stage; working buffers, not scratch (A31).** The plan gave each stage two overloads, one
    taking `TransferScratch&` (the step's) and one building it for the call (tests, tools). Decision: one form
    each, taking `TransferBuffers& transfer_buffers` last (`ParticlesToGrid`, `SaveGridBeforeForces`,
    `SaveGridAfterPressure`, `GridToParticles`, `ComputeCellsFromParticles`); tests and tools build a
    `TransferBuffers` locally. The word "scratch" is not used. Source: owner answer A31 (`owner-answers.md`,
    2026-11-14); transfer plan Global Constraints and Tasks 4, 11 to 16.
13. **Buffers on different processors are a precondition (A30).** Decision: `TransferBuffers` on another
    processor than the stage's arrays is a broken precondition (`Requires:`, asserted in debug builds), not
    `kBadInput`; `BufferError::kRunOnMismatch` no longer exists and nothing maps it. Source: owner answer A30
    (`owner-answers.md`, 2026-11-14).
14. **`BuildTransferBuffers` takes `material_count`.** The per-material offsets hold one 8 B entry per material
    (the estimate's `transfer: material offsets` entry has `given_count = material_count`), so the builder takes
    `std::size_t material_count` and the buffers report it; a set with more materials is `kBadInput`. Source:
    reconcile-todo.md item 27 (2026-11-14).
15. **Particles everywhere in the liquid (A64).** Decision: `FaceValueKind::kFromGridOnlyRegion` and its rules
    (5.4, 5.5, 5.8 item 8, section 4 "Cell resize") are not in the first version; the first version has four
    `FaceValueKind` values (item 10's "five values" is **replaced** for the first version). The later group "Deep
    liquid without particles (later; needs review)" of the transfer plan (Task L1), owned by the cell resize plan's
    group of that name, appends it as the fifth value. `GridOnlyReason` is unrelated (a particle taking the grid
    velocity only) and stays. Source: owner answer A64; `plan-decisions.md` "Owner answers A40–A64 applied", item 1.
16. **One reserved block per run (A58).** Decision: `BuildTransferBuffers`, `BuildSolidsAtFaces`,
    `BuildGridFromParticles`, `BuildGridSnapshot` and `BuildCellsFromParticles` take `RunMemory& run_memory` last
    (section 3); `BuildTransferMemoryEstimate` has one row per array. Source: owner answer A58; `plan-decisions.md`
    item 3.
17. **`kOutOfMemory` (reconcile item 31).** Decision: `TransferError::kOutOfMemory` appended as the twelfth and last
    value, returned only by the builders (for `BufferError::kOutOfMemory`, `kRunMemoryTooSmall` and the grid's
    `kOutOfMemory`), magnitude = bytes asked; the plan decision that mapped them to `kRunOnCallFailed` is
    **replaced**. Source: `plan-decisions.md` item 11.
18. **Particle size follows the cell (A65).** Every liquid cell holds about eight particles of its own size: a
    particle's `size_level` is its cell's level, its rest volume `(w/2)³`, so a cell k levels coarser than the smallest holds 8 particles each
    of `8^k` times the smallest particle's volume. Decision: no change to the method. The face sums weigh each
    particle by `m_p n_f` and the fits use the particle's cell width, so a linear field is reproduced across a size
    change whatever the particle masses (new test `ParticlesToGrid, LinearFieldParticlesSizedToTheirCell_IsReproduced`,
    plan Task 9); `min_weight_sum_at_face` counts shares, about equal at every level (decision 8, amended). Source:
    owner answer A65; `plan-decisions.md` "Owner answers A65–A68 applied", A65.
19. **Other answers of the A65–A68 pass.** A66 (start on the adaptive grid): the first transfer reads the seeded start
    particles like any step. A67 (drifted particles): a particle whose `size_level` differs from its cell for a few
    passes is summed and fitted by its cell, as every particle is; split children and merged particles carry the
    split or merged `C`. A68 (later work): the column cut would change only which blocks a batch holds; transfer's
    per-block order is unchanged. Source: `plan-decisions.md` A66, A67, A68.
