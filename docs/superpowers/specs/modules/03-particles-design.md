# Particles module design

Status: updated 2026-11-14 with the owner's answers (`owner-answers.md`) and `changes-to-apply.md`; brought in line
on 2026-11-14 with `plan-decisions.md` ("Particles", "Planned cost shown to the user, and buffers in batched runs")
as the particles plan records them (Task 11 amended, A17 batched-run buffers). Module 2b of
the rebuild (after grid, before transfer). Names shared between modules are fixed in `shared-interface.md`.

## 1. Purpose

A particle is a small labelled parcel of material: it has a position, a velocity, an amount (mass and volume) and,
for snow and springy solids, a memory of how it was deformed. This module stores every particle of every material
in one owned structure, keeps each particle's identity stable while the arrays are reordered, and is the only place
where particles are created, split, merged or removed. A step never edits the committed particles: it works on a
try and the try either replaces the committed values or is dropped.

Owns: the per-particle arrays, stable ids and `next_particle_id`, the time label and `state_version`, grouping by
cell block and `position_when_grouped`, `AddParticles`, `SplitParticle`, `MergeParticles`, `RemoveParticles`,
`AdjustParticleCount`, the try and its commit, the copy of grouped particles between a whole try and a loaded try
(`WriteParticleGroups`, `ReadParticleGroups`), snapshots of particle arrays, the particle entries of the memory estimate
table.

Does not own: particle-to-grid and grid-to-particle exchange (transfer); what a deformation means (materials);
where sources are and how many particles they give per step (step, I/O); when liquid moves between the particle
region and the grid-only region, when the cells are resized and how batches are cut (cell resize and batches);
file formats and the playback writers (I/O, render output); which gas markers are needed (gas); blob ids
(pressure, surface: see section 9).

## 2. What exists today

| Piece on `main` | Where | What it does | What is wrong |
|---|---|---|---|
| Liquid particle record | `native/include/fluidity/host/liquid_state_3d.h:105-125` | One 265-byte struct per particle: id, packed `tag`, position, stored velocity, `velocity_slopes`, `affine_enabled`, mass, reference volume, `volume_ratio`, `side_tag`, `pic_initialized`, `phase_id`, eight `(substance_id, volume_fraction)` pairs | Every particle pays for eight ingredient pairs and a side tag even when the scene has one liquid and no plate. `tag` is "preserved verbatim" and read by nothing in the step. Liquid only. |
| Snow and sand particle | `host/dry_solid_step_3d.h:16-24`; ids only in `host/plastic_solid_checkpoint_3d.h:16-24` | Separate struct: position, velocity, slopes, mass, reference volume, `elastic_deformation`, `log_plastic_jacobian` | A second particle type with no id; ids are attached only when saving. `native/src/dry_solid_step_3d.cc:497` copies the whole span into a `std::vector` each step. Y-up: `gravity_y`, `start_y` (not this module's fields, but its tests use them). |
| Cut solid particle | `host/solid_interaction_checkpoint_3d.h:25-36` | Third type: id, body id, `deformation_gradient`, and a separate `(particle_id, side)` list | Side identity lives in a parallel list keyed by id. |
| Owner and three versions | `liquid_state_3d.h:28-103, 216-236, 275-350`; `native/src/liquid_state_3d.cc:118-133, 715-726, 736-827` | `LiquidState3D` holds particles with blocks, cells, faces, sources, clock. `LiquidFrameTransaction3D` keeps `committed_`, `trial_`, `frame_start_`; `ResetTrialFromCommitted` copies everything before each try | Three full copies; a full copy per try (architecture report, "Transaction invariants": double-buffering everything does not fit). Particles, grid and clock are one bag. |
| Checks | `liquid_state_3d.cc:488-667` | Finite, positive, unique ids, order | Id uniqueness is a quadratic scan (`liquid_state_3d.h:261-263`). |
| Split | `native/src/liquid_resampling_3d.cc:358-458` | 1 to 8 at offsets `sqrt(3)/4` of the parent cell width per axis; mass and reference volume divided by 8; child velocity `v + C * offset` | Maths is right (numerics report 7.2). Around it: exact-division and "represents increment" checks (`:91-119, 383-396`), a cubic-normal variant with an energy-deficit report (`:123-139, 404-413`), and a caller-supplied callback that decides whether the neighbourhood is complete (`host/liquid_resampling_3d.h:23-28`). |
| Merge | `liquid_resampling_3d.cc:460-562`, helpers `:32-86, 215-354` | 2 to 8 into 1: summed mass and reference volume, mass-weighted centre and velocity, fitted change from the cross moment, ingredient volumes conserved by current volume; refused unless a spread gap is positive | Maths is right (numerics report 7.1, 7.3). The gate runs in outward-rounded interval arithmetic (`:157-205`). Energy lost is not reported. |
| Callers of split and merge | `native/src/graded_quadratic_mac_transfer_3d.cc:366-410` only | Wrappers that add the completeness callback | No production code decides which particles to split or merge. |
| Add from a source | `native/src/liquid_substep_3d_staging.cc:632-685`; counters `liquid_state_3d.h:156-161` | Count from rate times duration plus remainder; counter-based random offset in a box; `pic_initialized = false` | Lives inside the scheduler. Sound otherwise. |
| Add and remove at the open-sea side | `liquid_substep_3d_staging.cc:546-630` | Removes particles above the given surface, adds on a lattice | Y-up: compares `position.y` with the surface height. |
| Refill a cell | `native/src/liquid_band_handoff_3d.cc:240-254, 523-610` | `n` particles at the centre and opposite quarter-cell corners; `m = rho V / n`, `V0 = V / (n J)`, `v = v_cell + C * offset` | Maths kept. Surrounded by exact-agreement checks. |
| Grouping | `native/src/sparse_mac_grid_3d.cc:18, 26, 200-220`; `native/src/liquid_step_workspace_3d.cc:210-267`; `base/mac_transfer_types_3d.h:7-13` | Key = 4-cell bin shifted by 1.5 cells, order (z, y, x) then incoming order; `sorted_world_position` kept beside `trial_world_position` | Uniform grid only. Order inside a bin is incoming order, not id order. |
| GPU layout | `host/cuda_liquid_arena_3d.h:18-29`; `native/cuda/liquid_arena_3d_host.cc:945-1025` | 32-lane bins, one aligned array per field, 168 bytes per particle, two banks | Keys are computed on the host, uploaded, sorted, downloaded (CUDA report 3.1). Every try uploads the whole trial (`PreparePublication`). |
| Reference and proofs | `reference/src/fluidity_reference/resampling.py`, `state.py`, `multiphase.py`; `proofs/Fluidity/Resampling.lean`, `Merge.lean` | Split, merge, refill amounts; snapshot copy; mixture density; split and merge identities | See sections 7 and 10. |

Y-up matters here in three places only: canonical orders written as (z, y, x), the "x fastest" corner numbering of
split children and refill positions (which survive unchanged as an x, y, z bit order), and any comparison with
`main`, whose vectors and matrices must be rotated by `R = rows {1,0,0},{0,0,-1},{0,1,0}` (`v' = R v`,
`C' = R C R^T`) and matched by position, not by child order.

## 3. Public interface

Folder `native/fluidity/particles/`, namespace `fluidity::particles`. Every name is NEW unless marked DECIDED.
Changes of the 2026-11-14 update are marked NEW (2026-11-14).

### 3.1 `particle_fields.h` (shared, `FLUIDITY_CPU_AND_GPU`, C++20 subset)

```cpp
inline constexpr std::uint32_t kMaxIngredientCount = 8;  // Limit per material. Same value as main.

// Bits of the per-particle flags byte.
inline constexpr std::uint8_t kCarriesFittedChange = 1;  // DECIDED word. Off: fitted_velocity_change is ignored.
inline constexpr std::uint8_t kHasOwnMotion = 2;  // Off until the particle has been through one GridToParticles.

// Views of one material's arrays for one run of cell blocks. Pointers are CPU pointers when the particles were
// built for RunOn::kCpu and GPU pointers for RunOn::kGpu. A null pointer means the material does not carry that
// array. An index into these arrays is valid for one grouping_version only.
struct ParticleFields {
  grid::CellBlockRange cell_blocks{};               // NEW (2026-11-14). The blocks these views cover.
  RunOn run_on = RunOn::kCpu;                       // NEW (plan-decisions "Particles"). kGpu: GPU pointers.
  std::size_t particle_count = 0;
  const std::uint64_t* id = nullptr;                // Stable identity. Never zero, never reused. DECIDED word.
  const Vec3* position = nullptr;                   // m, world.
  const Vec3* position_when_grouped = nullptr;      // m, world. DECIDED.
  PerParticleVelocity<const Vec3*> velocity{};      // m/s, world. for_this_step is null outside a try. DECIDED (A1).
  const Mat3* fitted_velocity_change = nullptr;     // 1/s. Papers: APIC matrix C. DECIDED.
  const double* mass = nullptr;                     // kg. For gas markers: a positive weight, not a physical mass.
  const double* rest_volume = nullptr;              // m3, at rest density. Papers: reference volume V0. DECIDED (A4).
  const double* volume_ratio = nullptr;             // No unit, positive. Current volume / rest volume. Papers: J.
                                                    // DECIDED (A4).
  const std::uint8_t* flags = nullptr;              // kCarriesFittedChange | kHasOwnMotion.
  const std::uint8_t* size_level = nullptr;         // Size level of the cells this particle was made for.
  std::uint32_t ingredient_count = 1;               // 1..kMaxIngredientCount.
  const double* ingredient_fractions = nullptr;     // No unit. [particle * ingredient_count + ingredient]. Each
                                                    // row sums to 1. Null when ingredient_count is 1.
  const Mat3* springy_deformation = nullptr;        // No unit. Papers: elastic deformation gradient. DECIDED (A4).
  const double* permanent_volume_change = nullptr;  // Natural log. DECIDED (A4). Papers: log plastic Jacobian.
  const contact::PlateSide* plate_side = nullptr;   // Which side of a plate the particle is on.
  const std::uint64_t* group_begin = nullptr;       // [block - cell_blocks.first_block], count = block_count + 1.
                                                    // Particles of block b are [group_begin[b], group_begin[b+1]).
  const std::uint32_t* cell_begin = nullptr;        // [(block - cell_blocks.first_block) * 64 + place in block].
                                                    // First particle of that cell place, counted from
                                                    // group_begin[b]. A place ends where the next place of the
                                                    // block begins, or at the end of the group.
};

// One array a stage is about to change. read[i] is the newest value in this try. write[i] must be assigned
// for every particle. The two may be the same memory: never read read[j], j != i, after writing.
template <typename T>
struct ArrayToChange {
  const T* read = nullptr;
  T* write = nullptr;
};

// An array a stage may change during a step. The same values name the arrays a snapshot or a batch copy carries
// (3.7, 3.8); kMass names an array no stage changes, so BuildFieldsToChange refuses it with kWrongFields.
enum class Field : std::uint16_t {
  kPosition = 1,
  kVelocityForNextStep = 2,    // NEW (2026-11-14, A1). Was kStateVelocity.
  kVelocityForThisStep = 4,    // NEW (2026-11-14, A1). Was kTravelVelocity. Never in a snapshot.
  kFittedVelocityChange = 8,
  kVolumeRatio = 16,
  kFlags = 32,
  kSpringyDeformation = 64,
  kPermanentVolumeChange = 128,
  kPlateSide = 256,
  kMass = 512,                 // NEW (2026-11-14). Snapshot and batch copy only.
};
struct FieldList { std::uint16_t bits = 0; };  // Any combination of Field values.
constexpr FieldList ToFieldList(std::initializer_list<Field> fields);  // NEW (2026-11-14). Bitwise or.
constexpr bool HasField(FieldList list, Field field);                  // NEW (2026-11-14).

struct ParticleFieldsToChange {  // Same blocks and order as the ParticleFields it was built with.
  grid::CellBlockRange cell_blocks{};  // NEW (2026-11-14).
  RunOn run_on = RunOn::kCpu;          // NEW (plan-decisions "Particles"). Set from the particles.
  std::size_t particle_count = 0;
  ArrayToChange<Vec3> position{};
  PerParticleVelocity<ArrayToChange<Vec3>> velocity{};  // for_next_step, for_this_step (A1).
  ArrayToChange<Mat3> fitted_velocity_change{};
  ArrayToChange<double> volume_ratio{};
  ArrayToChange<std::uint8_t> flags{};
  ArrayToChange<Mat3> springy_deformation{};
  ArrayToChange<double> permanent_volume_change{};
  ArrayToChange<contact::PlateSide> plate_side{};
};

// kg, m3, kg m/s. What a set of particles amounts to.
struct ParticleAmounts {
  double mass = 0;
  double rest_volume = 0;
  double current_volume = 0;  // Sum of volume_ratio * rest_volume.
  Vec3 momentum{};            // Sum of mass * velocity.for_next_step.
};
```

### 3.2 `particles.h` (CPU only)

```cpp
enum class ParticlesError : std::uint8_t {
  kNone,                    // No error.
  kBadInput,                // A value is not finite, not positive where it must be, or a span has the wrong size.
  kWrongFields,             // A given array is not carried by this material, or a carried one was not given.
  kTooManyParticles,        // A material would exceed max_particle_count, or a try max_added_per_try.
  kIdsExhausted,            // next_particle_id would overflow.
  kOutsideGrid,             // A particle is in no active cell. The step should reject and shorten.
  kGroupingStale,           // Particles were added or removed, or the grid layout changed, since grouping.
  kMovedTooFarSinceGrouped, // A particle is farther than max_move_since_grouped from position_when_grouped.
  kNeighbourhoodIncomplete, // Split or merge: the faces near the particle are not all there at one cell size.
  kNoFittedChange,          // Split or merge: a particle does not carry a fitted velocity change.
  kMaterialKeepsItsCount,   // Split or merge asked on snow or a springy solid.
  kParticlesDiffer,         // Merge: the particles differ in material, side, flags or size level.
  kMergeTooSpread,          // Merge: the particles are too far apart; merging could add energy.
  kArrayPartlyWritten,      // CommitTry: an array was changed for some cell blocks only.
  kTryInProgress,           // BuildTry while another try of the same particles is alive.
  kOutOfMemory,             // An allocation failed. Nothing changed.
  kRunOnCallFailed,         // CheckCanRunOn failed for the chosen processor, or a call to it failed.
};
std::string_view ToMessage(ParticlesError error);
// NEW (2026-11-14). The enumerator's name without the k, in lower snake case ("merge_too_spread"). The stable
// code the CLI and the add-on show.
std::string_view ToName(ParticlesError error);

struct ParticlesFailure {  // The error value of every std::expected here (A2).
  ParticlesError error = ParticlesError::kNone;
  std::uint64_t particle_id = 0;     // The failing particle; on the GPU the one with the lowest id. Zero when
                                     // no particle is involved.
  std::uint32_t material_index = 0;
  double magnitude = 0;              // The offending value, or the amount by which a limit was passed.
};

// Liquid and gas markers may be split and merged; snow and springy solids keep their count.
enum class CountRule : std::uint8_t { kMayBeSplitAndMerged, kKeepsItsCount };

struct ExtraFields {  // Arrays beyond the common ones. A material pays only for those it asks for.
  bool has_springy_deformation = false;      // Snow, springy solids.
  bool has_permanent_volume_change = false;  // Snow.
  bool has_plate_side = false;               // Springy solids; any material in a scene with a plate.
};

struct MaterialParticlesSettings {
  std::uint64_t material_id = 0;         // Stable id from the ready scene. Nonzero, unique.
  std::uint32_t ingredient_count = 1;    // Ingredients of one material mix freely; two materials never mix.
  ExtraFields extra_fields{};
  CountRule count_rule = CountRule::kMayBeSplitAndMerged;
  std::size_t max_particle_count = 0;    // Limit. The step may raise it with ApplyParticleLimits.
};

struct ParticlesSettings {
  std::span<const MaterialParticlesSettings> materials;  // material_index is the position in this list.
  std::size_t max_added_per_try = 0;                     // Limit on particles waiting to be grouped in.
  // NEW (plan-decisions "Particles"). Sizes group_begin, cell_begin, written marks and loaded block origin: the
  // grid's GridSettings::max_cell_blocks (for a loaded try, the batch plan's max_loaded_blocks). Zero: kBadInput.
  // A grid with more blocks is kTooManyParticles at grouping, magnitude = block count.
  std::uint32_t max_cell_blocks = 0;
  RunOn run_on = RunOn::kCpu;
  // NEW (2026-11-14). For RunOn::kCpu only: kLockedForCopies in a batched run, where the arrays are copied to
  // the GPU every pass (base/buffer.h).
  CpuMemory cpu_memory = CpuMemory::kOrdinary;
};

// Owns every per-particle array, on the CPU or on the GPU. Move-only. Committed values change only in
// CommitTry and ApplyParticleLimits.
class Particles {
 public:
  double time() const;                     // s. The committed time the values describe.
  std::uint64_t state_version() const;     // DECIDED. Grows by one at each CommitTry.
  std::uint64_t grouping_version() const;  // Grows whenever the committed order changes.
  grid::GridVersion grid_version() const;  // The grid version the committed grouping was made for.
  std::uint64_t next_particle_id() const;  // Greater than every id ever given.
  RunOn run_on() const;
  std::size_t material_count() const;
  const MaterialParticlesSettings& material(std::size_t material_index) const;
  std::size_t particle_count(std::size_t material_index) const;
};

// Allocates every array at its limit. kBadInput, kOutOfMemory, kRunOnCallFailed. On failure nothing stays
// allocated.
std::expected<Particles, ParticlesFailure> BuildParticles(const ParticlesSettings& settings);
// Not inside a try (kTryInProgress). max_particle_counts: one per material, each at least the current count;
// max_cell_blocks: at least settings' current value (so at least the committed grouping's block count).
// Reallocates the particle arrays and the per-block tables (group_begin, cell_begin, written marks, loaded
// block origin) at the new limits and copies the committed values. kBadInput, kOutOfMemory, kRunOnCallFailed;
// on failure the particles are unchanged and usable. (plan-decisions "Particles": takes max_cell_blocks, so
// step::ApplyLimits raises the grid limit and the particles' block limit in one call.)
std::expected<void, ParticlesFailure> ApplyParticleLimits(Particles& particles,
                                                          std::span<const std::size_t> max_particle_counts,
                                                          std::uint32_t max_cell_blocks);
ParticleFields BuildFieldsToRead(const Particles& particles, std::size_t material_index,
                                 grid::CellBlockRange cell_blocks);
// Checks every committed value of the given cell blocks: finite, positive where required, fractions sum to 1
// within kFractionSumTolerance, ids ascending inside each cell place.
std::expected<void, ParticlesFailure> CheckParticles(const Particles& particles,
                                                     grid::CellBlockRange cell_blocks);
inline constexpr double kFractionSumTolerance = 1e-12;  // No unit. 8 ingredients, a few rounding steps each.

// Sums over the cell blocks of `fields`: one partial per block (particles of the block in grouped order), then
// math::TreeSum over the block partials in block order, per component. The result for a run of blocks is the
// same whether the run was loaded in one piece or in batches. Chooses the processor from fields.run_on.
// kOutOfMemory (the GPU path's one-call GpuArray of one ParticleAmounts per block), kRunOnCallFailed. Never NaN as
// a failure signal (plan-decisions "Particles"). One form (A31, 11.11): it takes no buffers because its callers
// pass views of particles of any owner; a caller that must not allocate uses ComputeAmountsPerBlock.
std::expected<ParticleAmounts, ParticlesFailure> ComputeAmounts(const ParticleFields& fields);
// NEW (2026-11-14). The block partials of ComputeAmounts, one per block of fields.cell_blocks, so a caller that
// sees the grid in several ranges can tree-sum over the whole grid. kBadInput when per_block.size() is not
// fields.cell_blocks.block_count; kOutOfMemory, kRunOnCallFailed; per_block is unchanged on failure.
std::expected<void, ParticlesFailure> ComputeAmountsPerBlock(const ParticleFields& fields,
                                                             std::span<ParticleAmounts> per_block);
```

### 3.3 `particles_try.h` (CPU only)

```cpp
// The values of one try of a step or of a resize. Reads see the newest value of each array. Dropping it
// changes nothing.
class ParticlesTry {
 public:
  std::size_t particle_count(std::size_t material_index) const;  // After the last grouping of this try.
  std::size_t waiting_count() const;         // Added and not yet grouped in.
};
std::expected<ParticlesTry, ParticlesFailure> BuildTry(Particles& particles);  // kTryInProgress.
ParticleFields BuildFieldsToRead(const ParticlesTry& particles_try, std::size_t material_index,
                                 grid::CellBlockRange cell_blocks);
// kWrongFields when `fields` names kMass or an array the material does not carry.
std::expected<ParticleFieldsToChange, ParticlesFailure> BuildFieldsToChange(
    ParticlesTry& particles_try, std::size_t material_index, grid::CellBlockRange cell_blocks, FieldList fields);
// NEW (2026-11-14). CheckParticles on the try's newest values (stage CheckStep).
std::expected<void, ParticlesFailure> CheckParticles(const ParticlesTry& particles_try,
                                                     grid::CellBlockRange cell_blocks);
// The try's values become the committed values. Bumps state_version, sets time. No array is copied.
// kArrayPartlyWritten. On failure the try is dropped and the committed values are unchanged.
std::expected<void, ParticlesFailure> CommitTry(Particles& particles, ParticlesTry&& particles_try,
                                              double committed_time);
void RejectTry(Particles& particles, ParticlesTry&& particles_try);  // Committed values, ids, versions unchanged.
```

### 3.4 `group_particles_by_block.h` (+ `_cpu.cc`, `_gpu.cu`)

```cpp
struct GroupingSettings {
  double max_move_since_grouped = 0;  // m. Given by the step from the transfer module's rule. Zero: regroup always.
};
// Reorders every array of every material so that particles of one cell block are contiguous, blocks in the
// grid's block order, particles inside a block by the cell place of their active cell, then by ascending id.
// Brings in added particles, drops removed ones, sets position_when_grouped = position. DECIDED name.
// kOutsideGrid (lowest id), kTooManyParticles. On failure the try's grouping is unchanged.
std::expected<void, ParticlesFailure> GroupParticlesByBlock(ParticlesTry& particles_try,
                                                          const grid::StaggeredGrid& grid);
// Ok when the try's grouping can still be used: nothing waiting, same grid layout, nobody moved too far.
std::expected<void, ParticlesFailure> CheckGrouping(const ParticlesTry& particles_try,
                                                  const grid::StaggeredGrid& grid,
                                                  const GroupingSettings& settings);
```

### 3.5 Count changes (+ `_cpu.cc`, `_gpu.cu` each)

`add_particles.h`, `remove_particles.h`, `split_particle.h`, `merge_particles.h`, `adjust_particle_count.h`.

```cpp
enum class AddReason : std::uint8_t { kFromSource, kRefillRegion };  // DECIDED.
enum class RemoveReason : std::uint8_t {
  kLeftDomain,
  kThroughOutlet,
  kIntoGridOnlyRegion,
  kMarkerNotNeeded,  // NEW (2026-11-14, A10). A gas marker gas::FindMarkerChanges no longer wants.
};

struct NewParticles {  // One entry per new particle, all spans the same length (fractions: length * count).
  std::span<const Vec3> position;                  // m, world.
  std::span<const Vec3> velocity;                  // m/s, world. Becomes velocity.for_next_step.
  std::span<const Mat3> fitted_velocity_change;    // 1/s. Empty: zero, and kCarriesFittedChange off.
  std::span<const double> mass;                    // kg.
  std::span<const double> rest_volume;             // m3.
  std::span<const double> volume_ratio;            // No unit. Empty: 1.
  std::span<const std::uint8_t> size_level;
  std::span<const double> ingredient_fractions;    // Required when the material has several ingredients.
  std::span<const Mat3> springy_deformation;       // Required when carried. Same for the two below.
  std::span<const double> permanent_volume_change;
  std::span<const contact::PlateSide> plate_side;
};
struct AddParticlesResult {
  std::uint64_t first_id = 0;  // Ids are first_id, first_id + 1, ... in input order.
  ParticleAmounts added{};     // Summed in input order.
  // NEW (A17, batched-run buffers; plan-decisions "Copies counted"). One per WriteBuffer slice call into a buffer
  // outside CPU memory, and their bytes; both 0 when the particles are in CPU memory (a CPU run, or a batched run
  // whose particles are in locked CPU memory). The step adds them to StepReport::copies.
  std::uint32_t copy_count = 0;
  std::uint64_t bytes_copied = 0;
};
// New particles wait in the try until GroupParticlesByBlock. They start with kHasOwnMotion off. DECIDED name.
// No _gpu.cu: for both processors it validates the CPU spans, sums amounts on the CPU and writes the waiting list
// with slice WriteBuffer (plan Task 11).
// kBadInput, kWrongFields, kTooManyParticles, kIdsExhausted. On failure the try is unchanged.
std::expected<AddParticlesResult, ParticlesFailure> AddParticles(
    ParticlesTry& particles_try, std::size_t material_index, const NewParticles& new_particles, AddReason reason);

struct RemoveParticlesResult { ParticleAmounts removed{}; };  // Summed in index order.
// particle_indexes: ascending, valid for the try's current grouping. DECIDED name. kBadInput on an index out of
// range, not ascending, or already removed; the try is then unchanged.
std::expected<RemoveParticlesResult, ParticlesFailure> RemoveParticles(
    ParticlesTry& particles_try, std::size_t material_index, std::span<const std::size_t> particle_indexes,
    RemoveReason reason);

// What "complete neighbourhood" (5.4) needs besides the grid, for one grid version. Raw arrays on the
// simulation's processor, indexed by grid::FacePlaceIndex and grid::CellPlaceIndex of that grid. A null pointer
// means "no solid in the scene" or "every cell is in the particle region". The caller passes the arrays of
// transfer::SolidsAtFaces and the region array of the cell resize design for that grid; this module includes neither header.
struct NeighbourhoodInputs {
  const std::uint8_t* outside_solid = nullptr;         // 0 when the face centre is inside a solid.
  const contact::PlateSide* face_side = nullptr;       // Plate side of the face centre.
  const std::uint8_t* is_grid_only_region = nullptr;   // 1 for a cell of the grid-only region.
};
enum class Neighbourhood : std::uint8_t { kComplete, kIncomplete };
// Shared (split_and_merge_math.h). kComplete when the active cell containing `position` has `size_level`, no
// cell touching it is grid-only, and on each axis grid::ComputeNearbyFaceWeights with the mask "outside_solid
// and face_side == side" returns is_complete.
FLUIDITY_CPU_AND_GPU Neighbourhood FindNeighbourhood(
    const grid::GridTables& tables, const NeighbourhoodInputs& inputs, Vec3 position, std::uint8_t size_level,
    contact::PlateSide side);

// Split, merge and AdjustParticleCount run inside the resize transaction of the cell resize design (conflict C2, 5.4).
// grid_before_resize is the committed grid and neighbourhood_before_resize its inputs; grid is the resized
// grid, neighbourhood its inputs, and the try must already be grouped for grid (kGroupingStale otherwise).
// Particle indexes and cell_blocks refer to grid. NEW (2026-11-14): the two pairs replace one grid and one
// NeighbourhoodInputs.

struct SplitParticleResult { std::uint64_t first_child_id = 0; };  // Eight children, consecutive ids.
// DECIDED name. Fails without changing anything unless the parent has a complete neighbourhood at its
// size_level on grid_before_resize and all eight children have one at size_level + 1 on grid (5.4).
std::expected<SplitParticleResult, ParticlesFailure> SplitParticle(
    ParticlesTry& particles_try, const grid::StaggeredGrid& grid_before_resize,
    const NeighbourhoodInputs& neighbourhood_before_resize, const grid::StaggeredGrid& grid,
    const NeighbourhoodInputs& neighbourhood, std::size_t material_index, std::size_t particle_index);

struct MergeParticlesResult {
  std::uint64_t merged_id = 0;
  double kinetic_energy_lost = 0;  // J, zero or more. Motion the single particle cannot represent.
};
// DECIDED name. particle_indexes: two to eight, ascending. Each particle must have a complete neighbourhood at
// its size_level on grid_before_resize, and the merged position one at size_level - 1 on grid (5.5).
std::expected<MergeParticlesResult, ParticlesFailure> MergeParticles(
    ParticlesTry& particles_try, const grid::StaggeredGrid& grid_before_resize,
    const NeighbourhoodInputs& neighbourhood_before_resize, const grid::StaggeredGrid& grid,
    const NeighbourhoodInputs& neighbourhood, std::size_t material_index,
    std::span<const std::size_t> particle_indexes);

struct AdjustParticleCountResult {
  std::size_t split_count = 0;
  std::size_t merged_group_count = 0;
  std::size_t skipped_incomplete_count = 0;  // Wanted, not done: neighbourhood incomplete on either grid.
  std::size_t skipped_too_spread_count = 0;  // Wanted, not done: merge gate.
  double kinetic_energy_lost = 0;            // J. Tree sum of the block partials.
};
// Applies the split and merge rule of section 5.6 to every material that may be split and merged, in the
// given cell blocks of grid. DECIDED name. A refused split or merge is counted, not an error. Fails only on
// kGroupingStale, kTooManyParticles, kIdsExhausted, kOutOfMemory, kRunOnCallFailed; the try is then unchanged.
std::expected<AdjustParticleCountResult, ParticlesFailure> AdjustParticleCount(
    ParticlesTry& particles_try, const grid::StaggeredGrid& grid_before_resize,
    const NeighbourhoodInputs& neighbourhood_before_resize, const grid::StaggeredGrid& grid,
    const NeighbourhoodInputs& neighbourhood, grid::CellBlockRange cell_blocks);
```

### 3.6 `split_and_merge_math.h` (shared, `FLUIDITY_CPU_AND_GPU`)

```cpp
struct OneParticle {  // The values split, merge and refill work on.
  Vec3 position{};
  Vec3 velocity{};                 // m/s. velocity.for_next_step.
  Mat3 fitted_velocity_change{};
  double mass = 0;
  double rest_volume = 0;
  double volume_ratio = 1;
};
inline constexpr double kMergeGapFloor = 1e-9;  // No unit, in cell widths squared. Section 5.5.
double ComputeSplitOffset(double parent_cell_width);                                    // m. sqrt(3)/4 * width.
OneParticle ComputeSplitChild(const OneParticle& parent, double split_offset, int child_index);  // 0..7.
struct MergedParticle {
  OneParticle particle{};
  double gap = 0;                  // No unit. Section 5.5.
  double kinetic_energy_lost = 0;  // J.
};
MergedParticle ComputeMergedParticle(const OneParticle* particles, int particle_count, double small_cell_width);
struct RefillCell {
  Vec3 cell_center{};
  double cell_width = 0;           // m.
  double current_volume = 0;       // m3: what the grid-only region hands over.
  double density = 0;              // kg/m3.
  double volume_ratio = 1;         // At the cell centre.
  Vec3 velocity{};                 // m/s, at the cell centre.
  Mat3 fitted_velocity_change{};   // 1/s.
};
OneParticle ComputeRefillParticle(const RefillCell& cell, int particle_count, int particle_index);  // count 1..9.
```

### 3.7 `particles_snapshot.h` and `particle_memory_estimate.h` (CPU only)

```cpp
struct ParticlesSnapshot {  // DECIDED word. CPU-owned copy of the committed values at one state_version.
  // NEW (2026-11-14): spelled out member by member. Every array is in grouped order and has particle_count
  // entries (ingredient_fractions: particle_count * ingredient_count). An array the material does not carry, or
  // the read did not ask for, is empty. Names, units and frames as in ParticleFields.
  struct Material {
    std::uint64_t material_id = 0;
    std::uint32_t ingredient_count = 1;
    std::size_t particle_count = 0;
    // Constant during a step: always present.
    std::vector<std::uint64_t> id;
    std::vector<double> rest_volume;                 // m3.
    std::vector<std::uint8_t> size_level;
    std::vector<double> ingredient_fractions;        // Empty when ingredient_count is 1.
    // Present when asked (SnapshotFields) and carried.
    std::vector<double> mass;                        // kg. Field::kMass.
    std::vector<Vec3> position;                      // m, world. Field::kPosition.
    std::vector<Vec3> velocity_for_next_step;        // m/s, world. Field::kVelocityForNextStep.
    std::vector<Mat3> fitted_velocity_change;        // 1/s.
    std::vector<double> volume_ratio;                // No unit.
    std::vector<std::uint8_t> flags;
    std::vector<Mat3> springy_deformation;           // No unit.
    std::vector<double> permanent_volume_change;     // Natural log.
    std::vector<contact::PlateSide> plate_side;
    // Full snapshot only. With the grid it restores the committed grouping exactly (5.9).
    std::vector<Vec3> position_when_grouped;         // m, world.
  };
  double time = 0;                   // s.
  std::uint64_t state_version = 0;
  std::uint64_t next_particle_id = 0;
  bool is_full = false;              // Every carried array, position_when_grouped included.
  std::vector<Material> materials;   // In material_index order.
};
struct SnapshotFields {
  bool all = true;     // A full snapshot: the saved state.
  FieldList fields{};  // With all false: these arrays besides the constant ones. kVelocityForThisStep: kWrongFields.
};
// NEW (2026-11-14). What an output snapshot carries (render output): position, velocity.for_next_step,
// volume_ratio, and the constant arrays id, rest_volume, size_level, ingredient_fractions.
inline constexpr SnapshotFields kOutputSnapshotFields{
    .all = false,
    .fields = ToFieldList({Field::kPosition, Field::kVelocityForNextStep, Field::kVolumeRatio}),
};
// Blocking. Not inside a try. On the GPU this is the one large copy to the CPU this module makes.
std::expected<ParticlesSnapshot, ParticlesFailure> ReadSnapshot(const Particles& particles, SnapshotFields fields);
// NEW (2026-11-14): takes the grid the snapshot's grouping was made for (the committed grid rebuilt from the
// saved active cells). Accepts only a full snapshot whose materials match settings (kBadInput, kWrongFields) and
// whose order is the grouping of position_when_grouped on grid (kGroupingStale). Rebuilds group_begin and
// cell_begin; every value, id and next_particle_id is restored to the byte.
std::expected<Particles, ParticlesFailure> BuildParticlesFromSnapshot(const ParticlesSettings& settings,
                                                                    const ParticlesSnapshot& snapshot,
                                                                    const grid::StaggeredGrid& grid);
// The memory estimate entries BuildParticles allocates from. The sizing estimate reads the same entries (6.4).
// NEW (2026-11-14) name, was BuildMemoryEstimate ("every design": Build<Module>MemoryEstimate).
std::vector<MemoryEstimateEntry> BuildParticlesMemoryEstimate(const ParticlesSettings& settings);
```

### 3.8 `particle_groups.h` (CPU only; `_cpu.cc`) NEW (2026-11-14)

For the batched run of the cell resize and batches design: the whole try lives in CPU memory, one loaded try on the GPU holds the groups
of one `grid::GridPiece`. The loaded try is a `ParticlesTry` of a `Particles` built with `RunOn::kGpu` and, per
material, `max_particle_count` = the batch plan's loaded particle limit (`BatchPlan::max_loaded_particles`). It is never committed.

```cpp
// Replaces every value of `loaded` by the groups of the blocks of `piece`, in the piece's block order (the
// grid's fixed order): for piece block b, the particles of whole-grid block piece.block_in_whole_grid()[b], in
// their grouped order, with the newest values of `whole` of every carried array (id, mass, rest_volume,
// size_level, position_when_grouped and the fractions included; velocity.for_this_step when `whole` wrote it).
// group_begin and cell_begin are rebuilt for piece.grid(); the loaded grouping is marked as made for
// piece.grid().version(); the loaded try's earlier writes and removal marks are forgotten; `loaded` remembers
// block_in_whole_grid for ReadParticleGroups.
// Fails: kGroupingStale (whole has waiting particles, or its grouping is not for the whole grid of piece),
// kTooManyParticles (the piece holds more particles of a material than loaded can), kWrongFields (the two
// tries do not carry the same arrays), kRunOnCallFailed. On failure `whole` is unchanged and `loaded` holds no
// usable grouping (kGroupingStale until the next successful call).
std::expected<void, ParticlesFailure> WriteParticleGroups(ParticlesTry& loaded, const grid::GridPiece& piece,
                                                          const ParticlesTry& whole);
// Copies back, for the blocks of `range` only (whole-grid block indexes; it must be the range of the piece last
// written into `loaded`), the arrays named in `fields` from `loaded` into the try buffers of `whole`, and marks
// them written for those blocks (CommitTry's kArrayPartlyWritten check counts them). Particles of `range`
// removed in `loaded` are marked removed in `whole` with the same reason (their amounts were returned by that
// RemoveParticles call). Values of border blocks are not copied.
// Fails: kBadInput (range is not the loaded piece's range), kGroupingStale (loaded has waiting particles or was
// regrouped since WriteParticleGroups), kWrongFields (a field the material does not carry, or kMass),
// kRunOnCallFailed. On failure `whole` is unchanged.
std::expected<void, ParticlesFailure> ReadParticleGroups(const ParticlesTry& loaded, grid::CellBlockRange range,
                                                         FieldList fields, ParticlesTry& whole);
```

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module,
with the name its section 3 has).

**Foundation (EXISTS):** `Vec3`, `Mat3`, `RunOn`, `CheckCanRunOn(RunOn)`, `FLUIDITY_CPU_AND_GPU`,
`base/cpu_only.h`, and, from `shared-interface.md` section (a) and the `base/` additions of
`changes-to-apply.md`:

```cpp
template <typename T> struct PerParticleVelocity { T for_next_step; T for_this_step; };  // A1.
template <typename Value> class Buffer;       // base/buffer.h. The storage of every per-particle array.
enum class CpuMemory { kOrdinary, kLockedForCopies };  // Optional argument of BuildBuffer.
template <typename Value> std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size,
                                                                                CpuMemory cpu_memory);
// CopyBuffer on the processor both live on (ApplyParticleLimits); slice forms taking (first, count) of the
// buffer (WriteParticleGroups, ReadParticleGroups).
template <typename Value> std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, Buffer<Value>& to);
template <typename Value> std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, std::size_t first,
                                                                      std::size_t count, Buffer<Value>& to,
                                                                      std::size_t to_first);
template <typename Value> std::expected<void, BufferError> WriteBuffer(std::span<const Value> values,
                                                                       Buffer<Value>& buffer, std::size_t first);
template <typename Value> std::expected<void, BufferError> ReadBuffer(const Buffer<Value>& buffer,
                                                                      std::size_t first, std::span<Value> values);
// base/stable_order.h. Grouping uses two passes: by id, then by cell place index.
std::expected<void, BufferError> BuildStableOrder(const Buffer<std::uint64_t>& keys, std::size_t count,
                                                  const Buffer<std::uint32_t>* previous_order,
                                                  Buffer<std::uint32_t>& order);
std::expected<double, BufferError> ComputeTreeSum(const Buffer<double>& values, std::size_t count);  // ordered_sums.h
FLUIDITY_CPU_AND_GPU double math::TreeSum(const double* values, std::size_t count);              // tree_sum.h
struct MemoryEstimateEntry;                             // base/memory_count.h, with CountedPer and KeptFor.
namespace math { double Sqrt(double); double Abs(double); bool IsFinite(double); bool IsFinite(Vec3);
                 bool IsFinite(const Mat3&); double Dot(Vec3, Vec3); }  // A11.
```

**Contact (PROVIDED, contact 3.2):** `contact::PlateSide { std::uint32_t plate_id; SolidSide side; }`, 8 bytes,
equality means same side, `{0, kNoPlate}` means no plate. `contact::MoveParticles` changes `position` through
`ParticleFieldsToChange`.

**Grid (PROVIDED, grid 3.1 to 3.7):** `grid::StaggeredGrid` (`version()`, `tables(run_on)`, `all_blocks()`),
`grid::GridVersion`, `grid::GridTables`, `grid::CellBlockIndex`, `grid::CellBlockRange { first_block;
block_count; }`, `grid::CellPlaceIndex`, `grid::FacePlaceIndex`, `grid::kNoIndex`, `grid::FindCellAt(tables,
position, near)`, `ToCellBlockIndex(cell)`, `ToSizeLevel(ToCellId(tables, cell))`, `ComputeCellWidth(frame,
size_level)`, `ComputeCellCenter(tables, cell)`, `grid::FindNearbyFaces`, `NearbyFaceMask`,
`ComputeNearbyFaceWeights` (`is_complete`), `FindTouchingCells`, `grid::ComputeCompleteWeightedSpread(cell_width)`,
and, for 3.8, the class `grid::GridPiece` (`grid()`, `range()`, `range_in_whole_grid()`, `block_in_whole_grid()`)
with its blocks in the grid's fixed order (grid 3.4). `weighted_spread` is a word owned by grid; this module uses it only
through `ComputeCompleteWeightedSpread`.

**Transfer (PROVIDED, transfer 3):** `ParticlesToGrid`, `GridToParticles` and `MixParticleVelocity` change
`velocity.for_next_step`, `velocity.for_this_step`, `fitted_velocity_change`, `flags` through
`ParticleFieldsToChange`; `GridToParticles` turns `kHasOwnMotion` on and turns `kCarriesFittedChange` off when it
could not fit. Its sums over one face run block by block and, inside a block, in this module's grouped order
(cell place, then id). Its material-subset input (`material_is_counted`, A8, A10) chooses which materials' views
it reads; nothing changes here. `transfer::SolidsAtFaces::outside_solid` and `side` are the arrays passed as
`NeighbourhoodInputs::outside_solid` and `face_side`.

**Materials (PROVIDED, materials 3):** `materials::BuildParticlesSettings` gives one `MaterialParticlesSettings`
per material (ingredient count, `ExtraFields` and `CountRule` from `MaterialKind`: liquid and gas
`kMayBeSplitAndMerged`; snow and springy solid `kKeepsItsCount`); `materials::ApplyMaterialResponse` changes
`volume_ratio`, `springy_deformation`, `permanent_volume_change`; `materials::ComputeOwnMotionShares` reads the
views.

**Solids (PROVIDED, solids 3.8):** `solids::ApplyPlateSides` and `ApplyPlateRemoval` set and clear `plate_side`
through `particles::ArrayToChange<contact::PlateSide>`; `solids::ComputeSolidsAtFaces` fills the
`SolidsAtFaces` whose arrays become `NeighbourhoodInputs` for each grid of a resize.

**Gas (PROVIDED, gas 3):** gas markers need only the common arrays (`mass` holds a `sample_weight`, temperature
and fuel are on the grid). `gas::FindMarkerChanges` returns `MarkerChanges { to_add; to_remove; }`; the step
applies them in stage `AddNewParticles` with `AddParticles` and `RemoveParticles(..., RemoveReason::
kMarkerNotNeeded)`. Gas markers are split and merged like liquid particles; transfer's material sets keep them
out of the liquid's sums (A10).

**Step (PROVIDED, step 3):** owns `step::SourceCounter` (how many particles each source still owes, the next
child number) with the same try-and-commit rule; turns sources into `NewParticles` in a fixed order (source
order, then child number) in stage `AddNewParticles`; calls `CheckGrouping` then `GroupParticlesByBlock` in
stage `ChooseActiveCells`; calls `RemoveParticles` in `MoveParticles` for particles that left;
`CheckParticles(const ParticlesTry&, ...)` in `CheckStep`; passes `committed_time` to `CommitTry` in
`CommitStep`; on `kTooManyParticles` restarts with `step::ApplyLimits`, which calls `ApplyParticleLimits` (a
capacity restart, not a numerical reject; it raises the grid limit and the particles' `max_cell_blocks` in one
call). The step adds each `AddParticlesResult::copy_count` and `bytes_copied` to `StepReport::copies` (step plan
Appendix A row M1, Appendix B row O1). `step::ResizeTry::particles_try()` is the try of a resize;
`step::ReadSnapshot(simulation, particles::SnapshotFields)` and `step::BuildSimulationFromSnapshot` call
`ReadSnapshot` and `BuildParticlesFromSnapshot` (with the grid it rebuilt from `active_cells`).

**Cell resize and batches (PROVIDED, cell resize and batches 3.4 to 3.7):** `ResizeCells` calls `AdjustParticleCount` with
the committed and the resized grid and their `NeighbourhoodInputs` inside the resize transaction, after
`GroupParticlesByBlock` for the resized grid, and regroups after it (C2, 5.4); `ApplyRegionTransfer` builds
`RefillCell` values, calls `ComputeRefillParticle`, `AddParticles(..., AddReason::kRefillRegion)` and
`RemoveParticles(..., RemoveReason::kIntoGridOnlyRegion)`, keeps the ledger from the returned
`ParticleAmounts`, and fills `is_grid_only_region`. `WriteBatch` and `ReadBatch` call `WriteParticleGroups` and
`ReadParticleGroups` (3.8); the batched `BuildSimulation` builds the whole particles with
`CpuMemory::kLockedForCopies` and the loaded particles on the GPU.

**I/O and CLI (PROVIDED, I/O 3):** the ready scene gives every authored snow or springy solid object its own
`material_id` when two objects must stay distinguishable; `saved_state` stores a full `ParticlesSnapshot`
(member by member as in 3.7); `RequestedOutputs::particle_fields` is a `particles::FieldList`.

**Surface, foam and spray (PROVIDED, render output 3):** reads an output snapshot
(`kOutputSnapshotFields`) and writes the simulation-particle playback files (A14); reads `ParticlesTry` views in
`make_foam_and_spray`.

**For the sand module:** sand particles use the snow layout (`springy_deformation`, `permanent_volume_change`,
`CountRule::kKeepsItsCount`, `plate_side` when cuttable). Nothing else is asked of this module.

## 5. Method

### 5.1 Layout: one group of arrays per material

Each material of the scene has its own arrays. Common arrays (every material): `id`, `position`,
`position_when_grouped`, `velocity.for_next_step`, `fitted_velocity_change`, `mass`, `rest_volume`,
`volume_ratio`, `flags`, `size_level`; inside a try also `velocity.for_this_step`. Extra arrays exist only when
`ExtraFields` or `ingredient_count > 1` asks for them.

| Material | Extra arrays | Count rule |
|---|---|---|
| One liquid | none | may be split and merged |
| Mixture of N liquids that mix | `ingredient_fractions` (N doubles) | may be split and merged |
| Snow | `springy_deformation`, `permanent_volume_change` (+ `plate_side` if cuttable) | keeps its count |
| Springy solid | `springy_deformation` (the whole deformation: nothing is permanent), `plate_side` | keeps its count |
| Gas markers | none | may be split and merged |
| Any material in a scene with a plate | + `plate_side` | unchanged |

New against `main`: liquids that do not mix are two materials, so `phase_id` disappears and "which liquids mix"
is "which ingredients share a material". Ingredient ids are not stored per particle: the material's ingredient
list fixes the order. The amount of ingredient k in a particle is
`ingredient_fractions[k] * volume_ratio * rest_volume` (m3); the fraction is stored, not the amount, so current
volume has one owner (numerics report 7.1).

### 5.2 Identity, time and versions

1. `id` is given once from `next_particle_id` and never reused. Split children and a merged particle get new ids.
2. `time` and `state_version` change only in `CommitTry`. `grouping_version` changes when a commit carried a
   regrouping. `next_particle_id` advances inside a try and is committed or dropped with it.
3. An index is a position in the arrays of one material for one `grouping_version`. Nothing outside a single
   stage call may keep an index (architecture report, "Identity is not an index").

### 5.3 Try and commit (ported idea, new mechanism)

Every array has two buffers. For each array one buffer holds the committed values.

1. `BuildTry` copies nothing. The try reads the committed buffers.
2. The first `BuildFieldsToChange` that names an array points its `write` at the other buffer and its `read` at
   the committed one. Later changes of that array in the same try read and write the try's buffer in place.
   `velocity.for_this_step` has a single buffer: it is made and used inside one try and never committed.
3. `GroupParticlesByBlock` writes every array of a material into the other buffers (it is a full reorder).
4. `CommitTry` checks that each changed array was written for all cell blocks (`kArrayPartlyWritten`; a
   `ReadParticleGroups` of a range counts as a write of its blocks, 5.10), then
   flips "which buffer is committed" for the changed arrays, sets `time`, adds one to `state_version`.
5. `RejectTry` forgets which buffers were written. Ids, counts, versions and values are as before `BuildTry`.

Fixes: no copy per try and no third "frame start" copy (`liquid_state_3d.cc:118-133, 715-726`). Invariant: no
write pointer handed out by a try ever points at a committed buffer.

### 5.4 Split 1 to 8 (maths ported unchanged; the completeness rule is now inside)

For a parent with cell width `H` (the width of size level `size_level`), mass `m`, velocity `v`, fitted change `C`:

```text
spread(w)   = ComputeCompleteWeightedSpread(w) = w^2 / 4
offset      = sqrt(spread(H) - spread(H/2)) = (sqrt(3) / 4) H
child k     : sign_x = bit 0 of k, sign_y = bit 1, sign_z = bit 2 (0 is the low side)
position_k  = position + offset * (+-1, +-1, +-1)
velocity_k  = v + C * (position_k - position)        mass_k = m / 8        rest_volume_k = rest_volume / 8
```

`volume_ratio`, `C`, `flags`, `ingredient_fractions` and `plate_side` are copied; `size_level` grows by one; ids
are eight consecutive new ids in order of k. Conserved exactly up to rounding: mass, rest volume, current volume,
each ingredient volume, momentum (offsets sum to zero), angular momentum and kinetic energy including the fitted
part (the children's position spread supplies exactly the spread the smaller cells lack).

Rule (fix; programme design row 2b): the offsets above are only right when `FindNeighbourhood` is `kComplete` for
the parent at `size_level` on `grid_before_resize` and for each child position at `size_level + 1` on `grid`
(the resized grid), on the parent's `plate_side`, each grid with its own `NeighbourhoodInputs`. If any of
the nine checks fails the particle is not split (`kNeighbourhoodIncomplete`) and stays as it is, which is always a
valid state. There is no fallback offset. Also required: `kCarriesFittedChange` on (`kNoFittedChange`), count
rule allows it (`kMaterialKeepsItsCount`).

Dropped: the exact-division and increment checks, the cubic-normal variant, the caller's callback.

Conflict C2, settled as the cell resize and batches design designed it (`changes-to-apply.md`). One grid cannot answer both checks: by
5.6 a parent is split only once its cell is already one level smaller, so no face of the parent's size is near
it on that grid any more (the particles of a merge have the same problem at their own level). Inside the resize
transaction both grids are alive: the committed grid holds the old size, the resized grid the new one (as on
`main`, `liquid_adaptive_family_candidate_3d.cc`: "the parent on the old coarse bank and each of eight children on
the new fine bank"). So split, merge and `AdjustParticleCount` take both grids and are called only by the cell resize
design's `ResizeCells`. Consequence, stated plainly: a particle is split or merged only in the resize that changes the
cells around it, and only when it is at least 1.5 cell widths inside the changed region on both grids. A
particle that drifts across an existing size change is never split or merged: on the committed grid its
surroundings are already of the other size, its own check fails, and it keeps its `size_level` for good (a
valid particle, counted in `skipped_incomplete_count`). The cell resize design's cell size rule moves the small cells with
the surface, so the liquid near the surface is split where it stands instead of crossing a size change.

### 5.5 Merge 2 to 8 into 1 (maths ported unchanged; plain doubles)

Inputs: n particles of one material, same `plate_side`, `flags` and `size_level`, small cell width `h`, each with
a complete neighbourhood at that level on `grid_before_resize`, sorted by id. All sums run in that order.
Positions and velocities are taken relative to the first particle before weighting, so a far-away origin does
not eat the digits.

```text
M  = sum m_c            w_c = m_c / M           center = sum w_c x_c        mean_v = sum w_c v_c
r_c = x_c - center      S = sum w_c r_c r_c^T / h^2                          (3x3, no unit)
gap = (spread(2h) - spread(h)) / h^2 - max over axis a of (S_aa + sum over b != a of |S_ab|)   = 3/4 - ...
C*  = (sum m_c (v_c - mean_v) r_c^T) / (M * spread(2h)) + (spread(h) / spread(2h)) * sum w_c C_c
rest_volume* = sum rest_volume_c            volume_ratio* = sum (volume_ratio_c rest_volume_c) / rest_volume*
fraction*_k  = sum (fraction_c,k volume_ratio_c rest_volume_c) / sum (volume_ratio_c rest_volume_c)
energy(p)    = (m_p / 2) (|v_p|^2 + spread(width_p) * sum of squares of the entries of C_p)
kinetic_energy_lost = sum energy(c) - energy(merged)
```

The merge is done only if `gap > kMergeGapFloor` (else `kMergeTooSpread`) and the merged position has a complete
neighbourhood at `size_level - 1` on `grid`. The gap test is the row-sum bound of `main`
(`liquid_resampling_3d.cc:244-259`) without interval arithmetic; `kMergeGapFloor = 1e-9` covers the rounding of
at most 28 pair products. With a positive gap the energy lost is zero or more. Eight children fresh from a split
have gap exactly zero and are not merged back until they have moved closer; `main` behaves the same.
Conserved: mass, rest volume, current volume, each ingredient volume, momentum, angular momentum including the
fitted part. Not conserved, and returned: kinetic energy (numerics report 7.3). The merged particle gets
a new id, `size_level - 1`, and the common flags.

### 5.6 `AdjustParticleCount` (new: `main` has no caller that chooses)

Called by the cell resize design in step 7 of `ResizeCells`, inside the resize transaction, once the try is grouped for the
resized grid `grid`; `grid_before_resize` is the committed grid (5.4, C2). For each material whose count rule
allows it, for each cell block of `grid` in the range, in block order:

1. For each particle in id order, find its active cell on `grid`. If the cell's size level is greater than the
   particle's `size_level`, split it (5.4). One split per particle per call.
2. For each active cell of `grid`, in the grid's cell order, collect the particles whose `size_level` is exactly
   one more than the cell's and that agree on `plate_side` and `flags`, in id order. Take them eight at a time; a last
   chunk of two to seven is also tried; a single leftover stays. Merge each chunk (5.5).
3. A refused split or merge is counted in the result and is not an error.

Called with the same grid twice (no resize), every wanted split or merge is refused and counted: the parent's
or the particles' check on `grid_before_resize` fails wherever the cells already have the other size. Counts
and `kinetic_energy_lost` are block partials summed by `math::TreeSum` over the blocks, as for amounts.

All particles of a merge are in one active cell, so a cell block never needs another block's particles. Split
children may land in a neighbouring block; they wait in the try and are placed by the next grouping.

### 5.7 Add, refill, remove

- `AddParticles` checks values (finite, positive mass and volumes, fractions sum to 1), gives ids in input order,
  and appends to the try's waiting list. Both reasons start with `kHasOwnMotion` off, as on `main`: a new
  particle has no motion of its own to keep (numerics report 7.4). The reason only labels the returned amounts:
  `kFromSource` is new material, `kRefillRegion` is material that was already counted in the grid-only region
  (gas markers added by `gas::FindMarkerChanges` are added with this reason, as the gas design states).
- `ComputeRefillParticle` (ported): for `n` particles in a cell of width `h`, particle 0 is at the centre when `n`
  is odd; the others go in pairs to opposite `cell_eighth` centres (`+-h/4` per axis), lowest eighth first, with
  x as bit 0. `mass = density * current_volume / n`, `rest_volume = current_volume / (n * volume_ratio)`,
  `velocity = cell velocity + C * offset`.
- `RemoveParticles` marks particles; they stay readable until the next grouping drops them. It returns their
  amounts so the caller's ledger can say where the material went. `kMarkerNotNeeded` (A10) labels a gas
  marker the gas upkeep rule removes (too many in a cell, or a cell that no longer holds gas); like every
  reason it changes nothing but the label.
- After any add or remove, `CheckGrouping` returns `kGroupingStale` until `GroupParticlesByBlock` runs.

### 5.8 Grouping

1. For every particle (kept and waiting): `grid::FindCellAt(tables, position, near)`. `kNoIndex`: `kOutsideGrid`
   (lowest id reported).
2. Order each material by (`grid::CellPlaceIndex` of the containing active cell, id): cell blocks in the grid's
   order, cell places inside a block, ids inside a place. Keys are unique, so the order does not depend on
   the sort used.
3. Move every array into the other buffers in that order; set `position_when_grouped = position`; fill
   `group_begin` and `cell_begin`; record the grid version.

This design first ordered by (cell block index, id). Transfer needs the particles of one cell together and a
first-particle-per-cell table, so the cell place was added to the key. Everything this module needs still
holds (a block is contiguous; a merge reads one active cell; sums have a fixed order). Cost: `cell_begin`,
256 bytes per cell block per material in each of the two buffers; ids ascend inside a cell place, no longer
inside a whole block; the sort has the same two passes, with a key 6 bits wider.

`CheckGrouping` fails with `kMovedTooFarSinceGrouped` when any axis of `position - position_when_grouped` exceeds
`max_move_since_grouped`. Not ported: the 1.5-cell shifted bins and incoming-order ties.

### 5.9 Snapshots (NEW 2026-11-14: members spelled out, exact restore)

- A full snapshot (`SnapshotFields::all`, the saved state) holds every carried array and
  `position_when_grouped`. The committed order is the grouping of `position_when_grouped`, not of `position`:
  between two groupings a particle may have moved into another cell without being regrouped
  (`max_move_since_grouped > 0`). `BuildParticlesFromSnapshot` therefore finds the cell place of each
  `position_when_grouped` on the given grid, checks that the stored order is (that cell place, id)
  (`kGroupingStale` otherwise), and rebuilds `group_begin` and `cell_begin` from it. No array is reordered, so a
  continued run sums in the order of the uninterrupted one. `velocity.for_this_step` is never stored (it is made
  inside a try).
- An output snapshot (`kOutputSnapshotFields`, render output and particle playback) carries `position`,
  `velocity.for_next_step`, `volume_ratio`, and the arrays no stage changes: `id`, `rest_volume`, `size_level`,
  `ingredient_fractions`. `RequestedOutputs::particle_fields` of I/O adds named fields to it (`kMass` included).
- Arrays the material does not carry are empty in both forms; asking for one is not an error (playback asks
  for the same fields of every material).

### 5.10 Loaded groups for a batched run (NEW 2026-11-14)

In a batched run (cell resize and batches, 5.7 to 5.10) the whole try lives in CPU memory and the stage functions run
unchanged on a loaded try on the GPU, built over a `grid::GridPiece` (range plus stacked border, blocks in the
grid's fixed order).

1. `WriteParticleGroups` copies, block by block of the piece, the whole try's group of the matching whole-grid
   block (`block_in_whole_grid`), every carried array, with slices of `WriteBuffer` (one slice per block and
   array; blocks of a range are one run in the whole arrays, so a range costs one slice per array). Inside a
   block the order is the whole try's grouped order, so `cell_begin` of a block is copied unchanged and
   `group_begin` is the running sum of the block counts. Particle `i` of loaded block `b` is particle
   `group_begin_whole[block_in_whole_grid[b]] + (i - group_begin_loaded[b])` of the whole try.
2. Stages change the loaded try through `BuildFieldsToChange` as in a one-piece run. `RemoveParticles` on the
   loaded try marks particles. No stage adds particles to or regroups a loaded try (stages `AddNewParticles` (1),
   `ChooseActiveCells` (2) and `ResizeCells` (21) run on the whole state on the CPU); `ReadParticleGroups` refuses one that was (`kGroupingStale`).
3. `ReadParticleGroups` copies the named arrays of the range's blocks back into the whole try's write buffers
   (5.3 step 2 applies: the first copy of an array points it at the try buffer), carries the removal marks of
   the range's particles, and records the blocks as written. Border blocks are not copied back: the batch that
   owns them computes the same bytes (cell resize and batches, 5.10).
4. Every per-particle value depends only on the particle and its neighbourhood, and every sum of this module is
   per block then `math::TreeSum` over blocks, so this module's operations give identical bytes for any cut of
   the ranges, given the same inputs. A batched step as a whole agrees with a one-piece step within the solver's
   tolerance, not byte for byte (A15: the batched solves), and gives identical bytes for every cut of one mode.

## 6. CPU and GPU

**6.1 Shared headers.** `particle_fields.h` (views, flags, amounts) and `split_and_merge_math.h` (5.4, 5.5, 5.7
formulas). Both backends call the same `ComputeSplitChild`, `ComputeMergedParticle`, `ComputeRefillParticle`.

**6.2 CPU.** One `std::vector`-like buffer pair per array per material, allocated in `BuildParticles`, never
inside a try. Grouping: compute cell place indexes, sort an index array by (cell place, id), gather array by
array. Threads
may split work by material and by array (each destination element is written by one thread). `ComputeAmounts`
sums each block sequentially in grouped order, then `math::TreeSum` over the block partials (the same order as
the GPU, and the same for any cut into ranges or batches). Merge sums are sequential in id order. Particle
arrays of a batched run are built with `CpuMemory::kLockedForCopies`.

**6.3 GPU.** The same arrays as separate device buffers (one per array, element types `std::uint64_t`, `Vec3`,
`Mat3`, `double`, `std::uint8_t`, `contact::PlateSide`), owned by `Particles`, resident for the whole run. Nothing is
copied to the CPU between stages; other modules' `.cu` files receive `ParticleFields` /
`ParticleFieldsToChange` holding device pointers.
- Grouping: one thread per particle computes its cell place index; two passes of `BuildStableOrder`
  (`base/stable_order.h`), first by id, then by cell place index (unique keys make the result independent of
  the sort used); one gather per array into the other buffer; `group_begin` and `cell_begin` by a search over
  the sorted keys.
- Split, merge, `AdjustParticleCount`: one thread per particle (split test) or per active cell (merge chunks);
  each thread writes its own fixed places in the waiting list (place = prefix sum of per-thread counts, an
  integer scan), so results do not depend on thread timing. Ids are `next_particle_id + place`.
- Sums: `ComputeAmounts` writes one partial per block (one thread per block, grouped order) and sums the
  partials with `ComputeTreeSum` (`base/ordered_sums.h`), which follows `math::TreeSum`: the CPU and the GPU sum
  in the same order. Merge sums are per thread, at most eight terms, in id order. No floating-point atomics
  anywhere.
- `WriteParticleGroups`, `ReadParticleGroups`: CPU code issuing slice copies (`WriteBuffer`, `ReadBuffer` with
  `first`), one per block run and array; `group_begin` of the loaded try is an integer scan on the CPU.
- `AddParticles` has no `_gpu.cu`: it validates the CPU spans and sums amounts on the CPU, then writes the
  waiting list with one slice `WriteBuffer` per carried array; each call is a counted copy
  (`AddParticlesResult::copy_count`, `bytes_copied`; A17, plan-decisions "Copies counted").
- Returned to the CPU: one status record per call (`ParticlesFailure`, counts, the
  `AdjustParticleCountResult` numbers), not counted (a status record). The reported failure is the one with the lowest particle id (an integer
  minimum). `ReadSnapshot` is the only large copy and is explicit.
- Status until the owner's hardware run: "GPU: compiled, not run".

**6.4 Memory-count table** (bytes per particle; all values are doubles as on `main`, precision is not tuned here).

| Row | Bytes | Buffers | Who has it | Kind |
|---|---|---|---|---|
| `id` | 8 | 2 | all | committed + try |
| `position`, `position_when_grouped`, `velocity.for_next_step` | 24 each | 2 | all | committed + try |
| `fitted_velocity_change` | 72 | 2 | all | committed + try |
| `mass`, `rest_volume`, `volume_ratio` | 8 each | 2 | all | committed + try |
| `flags`, `size_level` | 1 each | 2 | all | committed + try |
| `ingredient_fractions` | 8 N | 2 | N > 1 ingredients | committed + try |
| `springy_deformation` | 72 | 2 | snow, springy solid | committed + try |
| `permanent_volume_change` | 8 | 2 | snow | committed + try |
| `plate_side` | 8 | 2 | when asked | committed + try |
| `velocity.for_this_step` | 24 | 1 | all | try only |
| grouping keys and indexes | 32 | 1 | all | working buffers |
| `group_begin` | 8 per cell block per material | 2 | all | committed + try |
| `cell_begin` | 256 per cell block per material (4 x 64) | 2 | all | committed + try |
| waiting list | one buffer of every carried array | 1 | `max_added_per_try` items | working buffers |
| written marks (NEW) | 2 per cell block per material (a `FieldList`) | 1 | all | try only |
| loaded block origin (NEW) | 4 per loaded cell block | 1 | loaded particles of a batched run | try only |

Per particle, at `max_particle_count`: one liquid 2 x 178 + 56 = 412 B; mixture of N: + 16 N; snow
2 x 258 + 56 = 572 B; springy solid with side 2 x 258 + 56 = 572 B; gas markers 412 B. A batched run counts
these entries twice: once for the whole particles in CPU memory at `max_particle_count`, once for the loaded
particles on the GPU at the batch plan's loaded limit (the run's processor; in a CPU run with `max_loaded_blocks >
0` both are CPU memory and the copies are CPU to CPU, counted the same). This is the placement rule of
`plan-decisions.md` "Buffers in batched runs" applied to particles: the whole particles are carried between passes
and kept by the commit, so they live in CPU memory (`where = kCpu`, part of `MemoryEstimate::on_cpu`); the loaded
try is built by `step::BuildSimulation` at `max_loaded_particles` per material and `max_loaded_blocks` (step plan
Appendix A). `WriteParticleGroups` and `ReadParticleGroups` move their arrays inside a batch load, which counts one
copy (its arrays are bytes, not copies). Plan measure: 405 B per liquid particle with entries from `sizeof`
(plan-decisions "Particles"; the 412 B above is the design's estimate). `BuildParticlesMemoryEstimate` returns exactly
the entries `BuildParticles` allocates from, as `CountedPer::kGivenCount` entries with `given_count =
max_particle_count` of each material. "committed + try" is `KeptFor::kWholeRun` with `copies = 2`, "try only"
is `kOneStepTry`, "working buffers" is `kOneCall`. The storage also holds one `OrderBuffers` (`order_buffers`) and
one `SumBuffers` (`sum_buffers`), built from the limits, for the grouping's stable orders and prefix sum and the
largest-magnitude sum (`"particles: order buffers"`, `"particles: sum buffers"` and their fixed-bytes entries).

**6.5 Cell-block ranges.** `BuildFieldsToRead`, `BuildFieldsToChange`, `CheckParticles`, `ComputeAmounts`,
`AdjustParticleCount` and the key computation of grouping take a `CellBlockRange` and touch only those groups.
A stage that needs border cells asks grid for the neighbouring blocks and reads their groups too; this module
has no border of its own. The reorder step of grouping is over a whole material and runs where the arrays live.
In a batched run the arrays live in CPU memory; the batches design's `WriteBatch` and `ReadBatch` decide what is copied
per pass and call `WriteParticleGroups` and `ReadParticleGroups` (5.10) for the particle part.

**6.6 Tolerances.** Ids, counts, order, flags, `size_level` and `group_begin`: exact, CPU against GPU and run
against run. Floating-point values, CPU against GPU: `|a - b| <= floor + 1e-13 * max(|a|, |b|)` with floor
`1e-13 * cell_width` for positions, `1e-13 * largest speed in the case` for velocities, `1e-13 * largest value in
the case` for amounts and fitted change. (Every formula here is a short chain of IEEE operations with contraction
off, so equality is expected; the tolerance is what the tests demand.)

## 7. Tests

Analytic and conservation (CPU, then the same cases on GPU within 6.6). `H = 1`, values chosen exact in binary
unless stated; "rel" means relative to the largest term of the sum.

| Test | Setup | Expected | Tolerance |
|---|---|---|---|
| `ComputeSplitOffset, UnitCell_IsRootThreeOverFour` | `H = 1` | 0.4330127018922193 | 1 ulp |
| `ComputeSplitChild, EightChildren_KeepAmounts` | m = 8, V0 = 2, J = 0.5 | each m = 1, V0 = 0.25; sums equal parent | exact |
| `ComputeSplitChild, EightChildren_KeepMomentum` | v = {1,2,3}, C = rows {1,2,3},{4,5,6},{7,8,9} | `sum m_k v_k = m v` | 1e-14 rel |
| `ComputeSplitChild, EightChildren_KeepAngularMomentumAndEnergy` | same, rigid-rotation C about Z, then a general C | orbital + fitted angular momentum and `energy` equal the parent's | 1e-13 rel |
| `ComputeSplitChild, QuarterWidthOffset_WouldLoseEnergy` | same with offset `H/4` | energy strictly below the parent's (shows why the offset matters) | sign |
| `SplitParticle, ChildNextToSolid_IsNotSplit` | one child's neighbourhood incomplete | `kNeighbourhoodIncomplete`; try unchanged; no id used | exact |
| `SplitParticle, ParentAtSizeBoundary_IsNotSplit` | parent near cells of another size on `grid_before_resize` | same | exact |
| `SplitParticle, ParentOnOldGridChildrenOnNewGrid_IsSplit` | committed grid: parent's cell and its neighbours at the parent's size; resized grid: that region split one level | eight children, amounts kept as in `EightChildren_KeepAmounts`; the same call with the resized grid as both grids refuses (`kNeighbourhoodIncomplete`) | exact; 1e-14 rel |
| `SplitParticle, TryGroupedForOtherGrid_IsStale` | try grouped for the committed grid, `grid` = resized grid | `kGroupingStale`; try unchanged | exact |
| `SplitParticle, Snow_IsRefused` | snow material | `kMaterialKeepsItsCount` | exact |
| `ComputeMergedParticle, TwoEqual_MeanAndSum` | `h = 1`; two particles of 1 kg at x = -0.25 and +0.25, x velocity -1 and +1, C = 0 | mass 2, centre 0, velocity 0, `C*.xx = 0.25`, other entries 0, gap 0.6875, energy lost 0.9375 J | 1e-14 |
| `ComputeMergedParticle, SplitThenMerge_GivesParentBack` | split a parent, merge its eight children at `h = H/2` | parent's position, velocity, C, amounts; energy lost 0; gap 0 (so `MergeParticles` refuses this exact case, as on `main`) | 1e-13 rel; gap within 1e-14 |
| `ComputeMergedParticle, KeepsEachIngredientVolume` | 3 particles, 3 ingredients, different J | `sum f J V0` per ingredient unchanged | 1e-14 rel |
| `ComputeMergedParticle, KeepsAngularMomentum` | 5 particles, random fixed values | orbital + fitted angular momentum unchanged | 1e-13 rel |
| `ComputeMergedParticle, EnergyNeverGrows` | 1000 fixed-seed cases with gap > floor | `kinetic_energy_lost >= -1e-13 * energy` | as stated |
| `MergeParticles, TooSpread_IsRefused` | two particles `2 h` apart | `kMergeTooSpread` | exact |
| `MergeParticles, DifferentSide_IsRefused` | two sides | `kParticlesDiffer` | exact |
| `MergeParticles, CompleteOnlyOnNewGrid_IsRefused` | eight particles whose cells were already big on the committed grid | `kNeighbourhoodIncomplete` | exact |
| `ComputeRefillParticle, NineParticles_KeepCellAmounts` | V = 1, density 1000, J = 0.5 | masses sum to 1000, rest volumes to 2, momentum to `1000 v` | 1e-14 rel |
| `AddParticles, GivesConsecutiveIds_AndNoOwnMotion` | 5 particles | ids n..n+4, flag off, amounts returned | exact |
| `AddParticles, OnCpu_NoCopyCounted` (NEW, A17 batched-run buffers, plan Task 11) | particles on the CPU | `copy_count` 0, `bytes_copied` 0 | exact |
| `AddParticles, OnGpu_CopiesCountedPerArray` (NEW, A17, plan Task 11, label `gpu`) | 5 liquid particles | `copy_count` = the number of waiting-list arrays written (defaults included); `bytes_copied` = 5 x the sum of their `sizeof` | exact |
| `ApplyParticleLimits, RaisedBlockLimit_ReallocatesBlockTables` / `LowerBlockLimit_IsBadInput` (plan Task 10) | `max_cell_blocks` 64 -> 128 with particle limits unchanged / 64 -> 32 | limits, counts, versions, next id kept; allocated bytes equal the entries at 128 blocks / `kBadInput`, particles unchanged | exact |
| `RemoveParticles, MarkerNotNeeded_ReturnsAmounts` | gas markers, 3 removed with `kMarkerNotNeeded` | amounts of the three; dropped at the next grouping | exact |
| `GroupParticlesByBlock, AnyInputOrder_SameResult` | same particles in 3 shuffles, mixed-size layout | identical arrays | bytes |
| `GroupParticlesByBlock, InsideCell_IdsAscend` | 3 blocks of 2 sizes | `group_begin`, `cell_begin` and order as computed by hand | exact |
| `GroupParticlesByBlock, ParticleOutsideGrid_ReportsLowestId` | two outside | `kOutsideGrid`, the failure names the lower id | exact |
| `CheckGrouping, AfterAdd_IsStale` / `MovedTooFar` | as named | the two errors | exact |
| `RejectTry, AfterEveryKindOfChange_ChangesNothing` | add, remove, split, merge, group, change each field, then reject | snapshot before equals snapshot after, including `next_particle_id` and versions | bytes |
| `CommitTry, ChangedArraysOnly_NoCopy` | change velocity only | other arrays' buffer addresses unchanged; `state_version` + 1 | exact |
| `CommitTry, ArrayWrittenForSomeBlocks_IsRefused` | change one block range only | `kArrayPartlyWritten` | exact |
| `AdjustParticleCount, ParticleInSmallerCells_IsSplitOnce` / `EightInBiggerCell_AreMerged` | committed grid and a resized grid that splits / merges one region (5.6) | counts in the result; amounts unchanged | 1e-13 rel |
| `AdjustParticleCount, DriftedAcrossSizeChange_KeepsSizeLevel` | a particle moved from big cells into an existing region of small cells; called with the committed grid as both grids, then with a resize elsewhere | not split in either call; counted in `skipped_incomplete_count`; its arrays unchanged | exact |
| `AdjustParticleCount, AnyRangeCut_SameBytes` | the whole grid in one range, then in 1-block and 3-block ranges | identical arrays and result numbers | bytes |
| `ComputeAmounts, AnyRangeCut_SameBytes` | `ComputeAmountsPerBlock` over 1-block ranges, tree-summed, against `ComputeAmounts` over all blocks | identical | bytes |
| `CheckParticles, OnTry_SeesNewestValues` | NaN written into one try array | `kBadInput` on the try; the committed check passes | exact |
| `BuildParticles, LiquidOnly_AllocatesNoExtraArrays` | one liquid | extra pointers null; bytes equal `BuildParticlesMemoryEstimate` | exact |
| `BuildParticlesMemoryEstimate, EveryMaterialKind_MatchesAllocation` | the six layouts of 5.1 | sum of entries equals bytes allocated | exact |
| `ReadSnapshot` then `BuildParticlesFromSnapshot` | all kinds | equal snapshots | bytes |
| `BuildParticlesFromSnapshot, UngroupedMove_RestoresOrder` | `max_move_since_grouped > 0`; a particle moved into another cell since the last grouping; full snapshot, rebuild, one more step in both | same order, `group_begin`, `cell_begin`; the next step's committed bytes equal | bytes |
| `BuildParticlesFromSnapshot, OutputSnapshot_IsRefused` | snapshot read with `kOutputSnapshotFields` | `kBadInput` | exact |
| `ReadSnapshot, OutputFields_CarryConstantArrays` | liquid mixture of 3, snow | `id`, `rest_volume`, `size_level`, fractions, `position`, `velocity_for_next_step`, `volume_ratio` filled; every other vector empty | exact |
| `ReadSnapshot, ThisStepVelocity_IsRefused` | `kVelocityForThisStep` asked | `kWrongFields` | exact |
| `WriteParticleGroups` then `ReadParticleGroups`, `RoundTrip_SameBytes` | piece of 3 range blocks and 4 border blocks, all kinds | loaded `group_begin`, `cell_begin` as computed by hand; after changing positions in the loaded try and reading the range back, the whole try's range blocks equal the loaded values and border blocks are untouched | bytes |
| `ReadParticleGroups, EveryRangeRead_CommitSucceeds` / `OneRangeMissing_IsPartlyWritten` | 3 ranges, velocity changed in each loaded try; read back all / all but one | `CommitTry` succeeds / `kArrayPartlyWritten` | exact |
| `ReadParticleGroups, RemovedInLoaded_RemovedInWhole` | 2 particles removed in the loaded try, one in a border block | the range's one is dropped at the whole try's next grouping; the border one stays | exact |
| `WriteParticleGroups, PieceTooFull_IsRefused` | loaded limit below the piece's count | `kTooManyParticles`; whole try unchanged | exact |
| `ParticleGroups, BatchedAndWhole_SameBytes` | a stage sequence that changes velocity and position, run on the whole try and in 1-, 2- and 5-block batches | identical committed arrays | bytes |
| `ToName, EveryError_LowerSnakeCase` | every `ParticlesError` | the enumerator's name without `k`, lower snake case; all distinct | exact |

Reference: none. Every formula is closed-form and each test above computes its expected value independently of
the code under test. Proof: one Lean file, `proofs/Fluidity/Merge.lean`, trimmed to
`merge_preserves_moments_and_bounds_energy` and `merge_preserves_volume_inventories`, restated in the new
vocabulary for the single quadratic spread. It removes one uncertainty a test cannot: that a positive gap implies
no energy gain for every admissible input, not only the sampled ones. `Resampling.lean` is not ported (the split
identity is a finite sum the tests check exactly).

Comparison with `main` (supporting only): run `SplitUniformLiquidParticle3D` and `MergeUniformLiquidParticles3D`
(`kQuadratic`) on 20 fixed inputs, rotate into Z-up (section 2), match children by position; agree within
`1e-12` relative. Refill positions and amounts against `ReseedLiquidCoreCell3D` likewise.

GPU: every case above within 6.6; `GroupParticlesByBlock` and `AdjustParticleCount` run twice on the same input
must give identical bytes; `ComputeAmounts` on CPU and GPU give identical bytes (same tree order); the batch
round trip with a loaded try on the GPU; a 1 000 000-particle grouping repeat-run. Runs in the container without a GPU
(label `no_gpu`): `BuildParticles` with `RunOn::kGpu` returns `kRunOnCallFailed`; all shared-header maths through
the CPU tests.

## 8. Method paper section

Section "Particles" `<sec-particles>` with subsections: what a particle carries `<sec-particle-fields>`
(table `<tab:particle-fields>`, memory table `<tab:particle-memory>`); identity and order `<sec-grouping>`; try
and commit `<sec-particle-try>`; split `<sec-split>` with `<eq:split-offset>`; merge `<sec-merge>` with
`<eq:merge-fit>` and `<eq:merge-gap>`; keeping the count right, with the two grids of a resize
`<sec-adjust-count>`; adding and removing `<sec-add-remove>`; snapshots and loaded groups `<sec-particle-copies>`.

Figures: `<fig-particle-fields>` one row per material, a box per array, extra arrays shaded, so "liquid pays
nothing for snow's arrays" is visible. `<fig-grouping>` the same six particles before and after grouping across
two cell blocks of different sizes: indexes change, ids do not. `<fig-particle-try>` two buffers per array, a try
writing two of them, the flip at commit. `<fig-split>` a parent in a big cell with its eight children at
`sqrt(3)/4 H`, the naive `H/4` corners in grey, and the three spreads drawn as circles; a second panel with a
solid nearby where the split is refused; a third panel with the committed grid (parent's size) and the resized
grid (children's size) side by side, and a drifted particle refused on both. `<fig-merge>` particles in one big
cell, their spread ellipse inside the allowed circle (merged) and outside (refused).

Pseudocode: `<proc-group-particles>` (5.8), `<proc-split-particle>` (5.4), `<proc-merge-particles>` (5.5),
`<proc-adjust-particle-count>` (5.6), `<proc-commit-try>` (5.3).

Cite from the old reference list: [1] Jiang 2015 (fitted change, spread, merge moments), [2] Ding 2020 (the
same on a staggered grid), [17] Ferstl 2016 (particles only near the surface, refill), [5] Wang 2020 (particles
grouped by cell block on the GPU), [19] Stomakhin 2013 (what snow carries), [3] Hu 2018 (side identity for
cutting). Klar 2016 moves to the sand module.

## 9. Vocabulary

| Old name on `main` | New name | Source |
|---|---|---|
| `LiquidParticleState3D`, `DrySolidParticle3D`, `ElasticCheckpointParticle3D` | arrays of `Particles` | NEW |
| `LiquidState3D` (particle part), `LiquidRecords3D` | `Particles` | NEW |
| `trial_`, `LiquidFrameTransaction3D::ResetTrialFromCommitted` | `ParticlesTry`, `BuildTry` | NEW (`trial` is a never-say word; DECIDED word `try`) |
| `FinishAttempt(kAccepted)` / reject | `CommitTry` / `RejectTry` | NEW verbs on DECIDED words `commit`, `reject` |
| `clock.epoch` | `state_version` | DECIDED |
| `id`, array position / `ordinal` | `id`, `index` | DECIDED |
| `world_position`, `sorted_world_position` | `position`, `position_when_grouped` | DECIDED |
| `world_velocity` ("stored"), advecting velocity | `velocity.for_next_step`, `velocity.for_this_step` (`Field::kVelocityForNextStep`, `kVelocityForThisStep`) | DECIDED (A1) |
| `velocity_slopes` | `fitted_velocity_change` | DECIDED |
| `affine_enabled` | `kCarriesFittedChange` (`carries_fitted_change`) | DECIDED |
| `pic_initialized` | `kHasOwnMotion` | NEW, from DECIDED `own_motion` |
| `reference_volume` | `rest_volume` | DECIDED (A4) |
| `volume_ratio` | `volume_ratio` | DECIDED (A4) |
| `elastic_deformation`, `deformation_gradient` | `springy_deformation` | DECIDED (A4) |
| `log_plastic_jacobian` | `permanent_volume_change` | DECIDED |
| `side_tag`, `ElasticBladeSide3D` | `plate_side` of type `contact::PlateSide` | DECIDED in `shared-interface.md` (b); this design first wrote `solid_side` of type `SolidSideId` |
| `constituent`, `substance_id`, `volume_fraction`, composition | `ingredient`, position in the material's ingredient list, `ingredient_fractions` | VERIFIED by reading `liquid_mixture_3d.h:21-37`, `liquid_state_3d.h:117-123`: a constituent is one substance in a mixture, which is the decided meaning of `ingredient` |
| `phase_id` ("shared by miscible substances") | the material (`material_index`) | VERIFIED: not an ingredient; a phase is a set of ingredients that mix, which this design makes one material |
| `component` (in particle code) | none | VERIFIED: `blob` does not belong here. On `main` `component` in the blob sense appears only in air-pocket and surface records (`liquid_state_3d.h:202-214`, `surface_component_lineage_3d.h`); no particle carries one. `Particles` has no `blob_id` |
| `tag` (packed substance, phase, flags) | removed | VERIFIED unused by the step |
| `ParticleBinIndex3D`, sort bin, `LiquidParticleSortEntry3D` | `particle_group`, `GroupParticlesByBlock`, `group_begin` | DECIDED / NEW |
| `Emit`, `ReseedLiquidCoreCell3D` | `AddParticles` with `AddReason` | DECIDED |
| `SplitUniformLiquidParticle3D`, `MergeUniformLiquidParticles3D` | `SplitParticle`, `MergeParticles` | DECIDED |
| retire, erase in the inlet strip | `RemoveParticles` with `RemoveReason` | DECIDED / NEW enum |
| `RemapGasApicMarkers3D` dropping a marker | `RemoveReason::kMarkerNotNeeded` | NEW (A10, gas) |
| resampling | `AdjustParticleCount` | DECIDED |
| certify callback, "admissibility" | `particles::FindNeighbourhood`, `Neighbourhood::kComplete` | NEW (here since the reconciliation; built on grid) |
| "old coarse bank", "new fine bank" (`liquid_adaptive_family_candidate_3d.cc`) | `grid_before_resize`, `grid` | NEW (C2) |
| bulk host round trip of a range | `WriteParticleGroups`, `ReadParticleGroups`, loaded try | NEW (cell resize and batches) |
| `covariance_gap_lower_bound` | `gap`, `kMergeGapFloor` | NEW |
| D (support second moment) | `weighted_spread` via `ComputeCompleteWeightedSpread` | Verify, owned by grid |
| capacity | `max_particle_count`, `max_added_per_try` | DECIDED pattern `max_...` |
| inventory (of a particle set) | `ParticleAmounts` | NEW; cell resize uses `GridOnlyAmounts` on the cell side |
| `LiquidStateCapacities3D` growth | `ApplyParticleLimits` | NEW |
| scratch (working memory of the grouping and the sums; the plans' `OrderScratch`, `SumScratch`, replaced) | working buffers: `order_buffers` (`OrderBuffers`), `sum_buffers` (`SumBuffers`) | DECIDED (A31) |
| checkpoint read-back, `Readback` | `ReadSnapshot`, `ParticlesSnapshot` | DECIDED words |
| other NEW names | `ParticleFields`, `ParticleFieldsToChange`, `ArrayToChange`, `Field`, `FieldList`, `ExtraFields`, `CountRule`, `MaterialParticlesSettings`, `ParticlesSettings`, `NewParticles`, `OneParticle`, `MergedParticle`, `RefillCell`, `GroupingSettings`, `size_level` (per particle), `CheckGrouping`, `CheckParticles`, `ComputeAmounts`, `ComputeAmountsPerBlock`, `SnapshotFields`, `kOutputSnapshotFields`, `ToFieldList`, `HasField`, `ToName`, `BuildParticlesMemoryEstimate` | NEW |

## 10. Not ported

- `LiquidRecords3D`, `LiquidFrameTransaction3D`, the frame-start copy and frame scheduling
  (`liquid_state_3d.h`, `.cc`): the step module owns frames; this module has one try.
- Blocks, cells, faces, sources, clock and air-pocket records inside `LiquidState3D`: other modules.
- Exact-division, "represents increment" and "unrepresentable" checks; interval arithmetic; explicit `fma`
  (`liquid_resampling_3d.cc:91-119, 157-205, 383-396`): replaced by finite and positive checks (naming review 7.4).
- The cubic-normal split and merge variant and its energy-deficit report (`:123-139, 207-209`): the rebuild has one
  weight curve for split and merge; a second arrives only with a grid design that asks for it.
- The completeness callback and `void* context` (`liquid_resampling_3d.h:23-28`).
- `tag`, `phase_id`, per-particle ingredient ids, `quadratic_adaptive_transfer` (`liquid_state_3d.h:107-123`).
- Sort bins shifted by 1.5 cells and the host round trip of sort keys (`sparse_mac_grid_3d.cc:26`,
  `liquid_arena_3d_host.cc:945-1025`).
- The 32-lane bin layout, the pinned mirror of every range and the lease type (`cuda_liquid_arena_3d.h`).
- `adaptive_liquid_particle_exchange_3d.h` (one particle absorbs a cell's correction) and
  `liquid_band_handoff_3d.h` except the refill placement: the cell resize design decides region transfer.
- Splitting or merging snow or springy solids (numerics report 7.3). Sand is the sand module's.
- Splitting or merging outside a resize, or with one grid (C2): a particle that drifted across a size change
  is not resampled.
- `reference/.../resampling.py`, `state.py`; `multiphase.py` mixture density goes to whichever module needs a
  mixture's density (transfer or pressure), not here. `proofs/Fluidity/Resampling.lean`.
- Asynchronous snapshots and output queues: I/O, if it wants them.

## 11. Decisions

These questions were also merged, with the other modules' questions, in `owner-questions.md`. Each keeps its
text; the decision follows.

**11.1 O3: the word `state` in `PerParticleVelocity`.** `state` would mean three things (`velocity.state`,
`saved_state`, `state_version`). Options:
```cpp
PerParticleVelocity<Vec3> { state; travel; }    // a) as decided
PerParticleVelocity<Vec3> { carried; travel; }  // b) "carried by the particle to the next step"
PerParticleVelocity<Vec3> { kept; travel; }     // c) collides with own_motion_kept
```
Recommendation: b. Then `state` means only "everything the simulation stores at one committed time".
Decision: none of the three: `PerParticleVelocity<Vec3> { for_next_step; for_this_step; }` (owner answer A1).
`state` and `travel` are no longer field names; `own_motion_in_travel` keeps its word.

**11.2 `AdjustParticleCount` has no rule on `main`.** Nothing there chooses which particles to split or merge.
Options: a) the rule of 5.6 (split when the cell is smaller than the particle's `size_level`, merge same-level
particles of one bigger cell, eight at a time), which needs one byte per particle; b) ship `SplitParticle` and
`MergeParticles` only and leave `AdjustParticleCount` to module 9; c) a rule based on particles per cell
(`min..max`), which also thins crowded cells of one size. Recommendation: a.
Decision: a, called inside the cell resize design's resize transaction with the committed and the resized grid (conflict C2
as `changes-to-apply.md` settles it) (recommended option, owner's standing instruction, 2026-11-14).

**11.3 Which materials may be split and merged.** a) liquid and gas only (5.1); b) also split (never merge) snow
and sand, copying their deformation. Recommendation: a.
Decision: a, liquid and gas markers only (recommended option, owner's standing instruction, 2026-11-14). Sand is
left to the sand module (A9).

**11.4 Mixtures: fractions, and "a material is what mixes".** a) store `ingredient_fractions` and make liquids
that do not mix separate materials (this design; `phase_id` disappears); b) store amounts in m3 per ingredient
(`ingredient_volumes`), which duplicates current volume; c) keep a per-particle mixing group id. Recommendation: a.
```cpp
const double* ingredient_fractions;  // a) no unit, rows sum to 1
const double* ingredient_volumes;    // b) m3, rows sum to volume_ratio * rest_volume
```
Decision: a (recommended option, owner's standing instruction, 2026-11-14).

**11.5 Names that touch O6.** `rest_volume` (was `reference_volume`), `volume_ratio` (kept), `springy_deformation`
(was `elastic_deformation` and, for cut solids, `deformation_gradient`). Alternatives: `volume_change` for the
ratio (but `permanent_volume_change` is a log, the ratio is not), `deformation` alone for cut solids.
Recommendation: as listed; materials may rename `volume_ratio` when it settles O6.
Decision: as listed, `rest_volume`, `volume_ratio`, `springy_deformation`, `permanent_volume_change` (owner answer
A4, the plain set); no alternative remains.

**11.6 New verbs and the try's type name.** `trial` is a never-say word, so the copy is `ParticlesTry`. Proposed
verbs outside the allowed list: `CommitTry`, `RejectTry` (`BuildTry` uses an allowed verb). Alternative:
`ApplyTry` / drop the try by destroying it. Recommendation: `CommitTry`, `RejectTry`.
Decision: `ParticlesTry`, `BuildTry`, `CommitTry`, `RejectTry` (recommended option, owner's standing instruction,
2026-11-14).

**11.7 Is grouping redone every try?** a) only when `CheckGrouping` fails (this design; `position_when_grouped`
earns its 48 bytes per particle); b) every try, and `position_when_grouped` is deleted. Results are identical
either way because order is by id. Recommendation: a, with the step free to pass
`max_move_since_grouped = 0` at first.
Decision: a, with the step free to pass `max_move_since_grouped = 0` (recommended option, owner's standing
instruction, 2026-11-14). Since the cell place joined the order key (5.8), the two options differ in summation
order once a particle crosses a cell between groupings; a run is still reproducible, and a saved state restores
the grouping exactly from `position_when_grouped` (5.9).

**11.8 `kHasOwnMotion`, `size_level` on a particle, `CountRule`, `ParticleAmounts`.** Non-obvious NEW names;
confirm or replace (`kHasOwnMotion` could be `kFollowedGridOnce`; `CountRule` could be two functions' worth of
`bool may_be_split_and_merged`).
Decision: the names as written, `kHasOwnMotion`, `size_level`, `CountRule`, `ParticleAmounts` (recommended option:
the design's proposal, owner's standing instruction, 2026-11-14).

**11.9 Decisions the particles plan made beyond the design** (`plan-decisions.md` "Particles", "Particles
(continued)"; recorded here as the contract now stands). Decision: `ParticlesSettings::max_cell_blocks` sizes the
per-block tables (zero is `kBadInput`); `ApplyParticleLimits(Particles&, std::span<const std::size_t>
max_particle_counts, std::uint32_t max_cell_blocks)` raises the particle and block limits in one call;
`ParticleFields` and `ParticleFieldsToChange` carry `RunOn run_on`; `ComputeAmounts` returns
`std::expected<ParticleAmounts, ParticlesFailure>` and `ComputeAmountsPerBlock` `std::expected<void,
ParticlesFailure>` (no NaN signalling); 405 B per liquid particle (entries from `sizeof`). Source: plan decision
(coordinator), 2026-11-14.

**11.10 Copies made by `AddParticles` are counted (A17).** Decision: `AddParticlesResult` gains `std::uint32_t
copy_count = 0;` (one per slice `WriteBuffer` into a buffer outside CPU memory) and `std::uint64_t bytes_copied =
0;` (0 when the particles are in CPU memory); the step adds them to `StepReport::copies`. In a batched run the whole
particles stay in CPU memory and one loaded try lives on the run's processor (placement rule, 6.4). Source: plan
decision, 2026-11-14 (planned cost; buffers in batched runs); owner principle A17; `plan-decisions.md` "Copies
counted and the one exception"; particles plan Task 11; step plan Appendix A row M1.

**11.11 One form per function; working buffers, not scratch (A31).** Decision: the word "scratch" is not used;
working memory is "working buffers". Every Foundation call (`BuildStableOrder`, `ComputeExclusivePrefixSum`,
`ComputeLargestMagnitude`) takes the storage's `order_buffers` / `sum_buffers`; no form builds its own. No public
particles signature changes. `ComputeAmounts` keeps its one form, which holds one call's GPU memory (`GpuArray` of
one `ParticleAmounts` per block) and keeps `kOutOfMemory`: its callers pass views of particles of any owner with
no buffers to lend; a caller that must not allocate uses `ComputeAmountsPerBlock`. Source: owner answer A31
(`owner-answers.md`, 2026-11-14); particles plan Tasks 12, 15, 17.

**11.12 Buffers on different processors are a precondition (A30).** Decision: `BufferError::kRunOnMismatch` no
longer exists. The particles' arrays and their working buffers live on `settings.run_on` by construction; a
function given buffers and working buffers on different processors is a broken precondition (`Requires:` same
processor, asserted in debug builds), never an error value. Source: owner answer A30 (`owner-answers.md`,
2026-11-14).
