# Particles module design

Status: current design, trimmed 2026-11-27 before planning. Module 3 of the rebuild (after grid, before transfer).
Names shared between modules are fixed in `shared-interface.md`.

## 1. Purpose

A particle is a small labelled parcel of material: it has a position, a velocity, an amount (mass and volume) and,
for snow and springy solids, a memory of how it was deformed. This module stores every particle of every material
in one owned structure, keeps each particle's identity stable while the arrays are reordered, and is the only place
where particles are created, split, merged or removed. A step never edits the committed particles: it works on a
try and the try either replaces the committed values or is dropped.

Owns: the per-particle arrays, stable ids and `next_particle_id`, the time label and `state_version`, grouping by
cell block and `position_when_grouped`, `AddParticles`, `RemoveParticles`,
`AdjustParticleCount`, the try and its commit, the copy of grouped particles between a whole try and a loaded try
(`WriteParticleGroups`, `ReadParticleGroups`), snapshots of particle arrays, the particle entries of the memory estimate
table.

Does not own: particle-to-grid and grid-to-particle exchange (transfer); what a deformation means (materials); where
sources are and how many particles they give per step (step, I/O); when liquid moves between the particle region and the
grid-only region (not in the first version, A64: every liquid cell holds particles), when the cells are resized and how
batches are cut (cell resize and batches); file formats and the playback writers (I/O, render output); which gas markers
are needed (gas); blob ids (pressure, surface: see section 9).

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

Folder `src/fluidity/particles/` (tests of scenes in `tests/particles/`), namespace `fluidity::particles`. Every name
is NEW unless marked DECIDED.

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
  grid::CellBlockRange cell_blocks{};               // The blocks these views cover.
  RunOn run_on = RunOn::kCpu;                       // kGpu: GPU pointers.
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
  const grid::SizeLevel* size_level = nullptr;      // Size level of the cells this particle was made for.
  // Resize passes this particle has waited, too small for its cell, for a chunk of eight; saturates at
  // 255; 0 when it matches its cell or was just made. Null for a material with CountRule::kKeepsItsCount.
  const std::uint8_t* merge_wait_passes = nullptr;
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
  kVelocityForNextStep = 2,
  kVelocityForThisStep = 4,    // Never in a snapshot.
  kFittedVelocityChange = 8,
  kVolumeRatio = 16,
  kFlags = 32,
  kSpringyDeformation = 64,
  kPermanentVolumeChange = 128,
  kPlateSide = 256,
  kMass = 512,                 // Snapshot and batch copy only.
  kMergeWaitPasses = 1024,     // Written only by AdjustParticleCount (its written mark); snapshot and
                               // batch copy; BuildFieldsToChange refuses it with kWrongFields, as kMass.
};
struct FieldList { std::uint16_t bits = 0; };  // Any combination of Field values.
constexpr FieldList ToFieldList(std::initializer_list<Field> fields);  // Bitwise or.
constexpr bool HasField(FieldList list, Field field);

struct ParticleFieldsToChange {  // Same blocks and order as the ParticleFields it was built with.
  grid::CellBlockRange cell_blocks{};
  RunOn run_on = RunOn::kCpu;          // Set from the particles.
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
  kArrayPartlyWritten,      // CommitTry: an array was changed for some cell blocks only.
  kTryInProgress,           // BuildTry while another try of the same particles is alive.
  kRunOnCallFailed,         // CheckCanRunOn failed for the chosen processor, or a call to it failed.
  kInternalFault,           // A state no input produces: a fault in FluidiTy, not in the scene. Report it with
                            // the scene.
  kOutOfMemory,             // A building function (or ComputeAmounts' one-call GPU memory, the accepted
                            // exception) could not get its memory; magnitude = bytes asked. Nothing changed.
};
std::string_view ToMessage(ParticlesError error);
// The enumerator's name without the k, in lower snake case ("too_many_particles"). The stable
// code the CLI and the add-on show.
std::string_view ToName(ParticlesError error);

struct ParticlesFailure {  // The error value of every std::expected here (A2).
  ParticlesError error = ParticlesError::kNone;
  std::uint32_t material_index = 0;  // At byte 4, so the struct is 24 bytes on both processors.
  std::uint64_t particle_id = 0;     // The failing particle; on the GPU the one with the lowest id. Zero when
                                     // no particle is involved.
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
  // Sizes group_begin, cell_begin, written marks and loaded block origin: the
  // grid's GridSettings::max_cell_blocks (for a loaded try, the batch plan's max_loaded_blocks). Zero: kBadInput.
  // A grid with more blocks is kTooManyParticles at grouping, magnitude = block count.
  std::uint32_t max_cell_blocks = 0;
  RunOn run_on = RunOn::kCpu;
  // For RunOn::kCpu only: kLockedForCopies in a batched run, where the arrays are copied to
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

// Allocates every array at its limit. kBadInput, kOutOfMemory, kRunOnCallFailed. Every array is a
// piece of run_memory, the RunMemory of settings.run_on (in a GPU run, CPU particles get the run's cpu_run_memory,
// built with settings.cpu_memory). Requires: settings.run_on == run_memory.run_on() (debug assertion). On failure
// no Particles is returned; the pieces already taken stay in the block (a piece is never freed alone).
std::expected<Particles, ParticlesFailure> BuildParticles(const ParticlesSettings& settings, RunMemory& run_memory);
// Not inside a try (kTryInProgress). max_particle_counts: one per material, each at least the current count;
// max_cell_blocks: at least settings' current value (so at least the committed grouping's block count).
// Reallocates the particle arrays and the per-block tables (group_begin, cell_begin, written marks, loaded
// block origin) at the new limits and copies the committed values. kBadInput, kOutOfMemory, kRunOnCallFailed;
// on failure the particles are unchanged and usable. It takes max_cell_blocks, so
// step::ApplyLimits raises the grid limit and the particles' block limit in one call. The new
// arrays are pieces of run_memory, the new block step::ApplyLimits reserved for the whole new plan; the old arrays
// free nothing (their block is freed whole by the step). Requires: particles.run_on() == run_memory.run_on().
std::expected<void, ParticlesFailure> ApplyParticleLimits(Particles& particles,
                                                          std::span<const std::size_t> max_particle_counts,
                                                          std::uint32_t max_cell_blocks, RunMemory& run_memory);
// kBadInput for a range that is not inside the grouping's blocks, and for an empty range except when the
// grouping has no block (A116).
std::expected<ParticleFields, ParticlesFailure> BuildFieldsToRead(const Particles& particles,
                                                                  std::size_t material_index,
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
// a failure signal. One form (A31, 11.11): it takes no buffers because its callers
// pass views of particles of any owner; a caller that must not allocate uses ComputeAmountsPerBlock.
std::expected<ParticleAmounts, ParticlesFailure> ComputeAmounts(const ParticleFields& fields);
// The block partials of ComputeAmounts, one per block of fields.cell_blocks, so a caller that
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
// kBadInput for a range that is not inside the grouping's blocks, and for an empty range except when the
// grouping has no block (A116).
std::expected<ParticleFields, ParticlesFailure> BuildFieldsToRead(const ParticlesTry& particles_try,
                                                                  std::size_t material_index,
                                                                  grid::CellBlockRange cell_blocks);
// kWrongFields when `fields` names kMass, kMergeWaitPasses or an array the material does not carry.
std::expected<ParticleFieldsToChange, ParticlesFailure> BuildFieldsToChange(
    ParticlesTry& particles_try, std::size_t material_index, grid::CellBlockRange cell_blocks, FieldList fields);
// CheckParticles on the try's newest values (stage CheckStep).
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

`add_particles.h`, `remove_particles.h`, `adjust_particle_count.h`.

```cpp
// DECIDED. kRefillRegion (kept, A64): material already counted on the grid, not new: gas markers added by
// gas::FindMarkerChanges (first version) and, later, liquid handed over by a grid-only region.
enum class AddReason : std::uint8_t { kFromSource, kRefillRegion };
enum class RemoveReason : std::uint8_t {
  kLeftDomain,
  kThroughOutlet,
  kIntoGridOnlyRegion,  // Not in the first version (A64): no caller until the later group; kept so the values
                        // after it keep their numbers.
  kMarkerNotNeeded,  // A gas marker gas::FindMarkerChanges no longer wants.
};

struct NewParticles {  // One entry per new particle, all spans the same length (fractions: length * count).
  std::span<const Vec3> position;                  // m, world.
  std::span<const Vec3> velocity;                  // m/s, world. Becomes velocity.for_next_step.
  std::span<const Mat3> fitted_velocity_change;    // 1/s. Empty: zero, and kCarriesFittedChange off.
  std::span<const double> mass;                    // kg.
  std::span<const double> rest_volume;             // m3.
  std::span<const double> volume_ratio;            // No unit. Empty: 1.
  std::span<const grid::SizeLevel> size_level;
  std::span<const double> ingredient_fractions;    // Required when the material has several ingredients.
  std::span<const Mat3> springy_deformation;       // Required when carried. Same for the two below.
  std::span<const double> permanent_volume_change;
  std::span<const contact::PlateSide> plate_side;
};
struct AddParticlesResult {
  std::uint64_t first_id = 0;  // Ids are first_id, first_id + 1, ... in input order.
  ParticleAmounts added{};     // Summed in input order.
  // One per WriteBuffer slice call into a buffer
  // outside CPU memory, and their bytes; both 0 when the particles are in CPU memory (a CPU run, or a batched run
  // whose particles are in locked CPU memory). The step adds them to StepReport::copies.
  std::uint32_t copy_count = 0;
  std::uint64_t bytes_copied = 0;
};
// New particles wait in the try until GroupParticlesByBlock. They start with kHasOwnMotion off and
// merge_wait_passes 0 when the material carries it; NewParticles has no span for it. DECIDED name.
// No _gpu.cu: for both processors it validates the CPU spans, sums amounts on the CPU and writes the waiting list
// with slice WriteBuffer.
// kBadInput, kWrongFields, kTooManyParticles, kIdsExhausted. On failure the try is unchanged.
std::expected<AddParticlesResult, ParticlesFailure> AddParticles(
    ParticlesTry& particles_try, std::size_t material_index, const NewParticles& new_particles, AddReason reason);

struct RemoveParticlesResult { ParticleAmounts removed{}; };  // Summed in index order.
// particle_indexes: ascending, valid for the try's current grouping. DECIDED name. kBadInput on an index out of
// range, not ascending, or already removed; the try is then unchanged.
std::expected<RemoveParticlesResult, ParticlesFailure> RemoveParticles(
    ParticlesTry& particles_try, std::size_t material_index, std::span<const std::size_t> particle_indexes,
    RemoveReason reason);

// Passes a group of two to seven too-small particles waits for an eighth before it is merged anyway.
// Starting value: about 16 committed steps, since ResizeCells runs once per committed step.
inline constexpr std::uint8_t kMaxMergeWaitPasses = 16;

// Counts are integer sums of block partials; energies are block partials then math::TreeSum over blocks.
struct AdjustParticleCountResult {
  std::size_t split_count = 0;               // Particles split (a repeated split of one particle counts once).
  std::size_t merged_group_count = 0;
  std::size_t skipped_too_spread_count = 0;  // Chunks not merged: the gap test fails; retried next pass.  
  double kinetic_energy_lost = 0;            // J, zero or more. Merges only: sum of their kinetic_energy_lost.
  std::size_t mismatched_particle_count = 0; // Candidates at the start of the pass: size_level != the level of
                                             // their active cell on grid.
  std::size_t waiting_particle_count = 0;    // Too-small particles left after the pass in chunks of two to seven
                                             // (waiting, or refused by the gap test).
  std::size_t lone_particle_count = 0;       // Too-small particles left alone in their chunk (cannot merge).
  double kinetic_energy_change = 0;          // J, energy(after) - energy(before) over every split and merge (5.6):
                                             // 0 up to rounding for splits, <= 0 for merges.
};
// Applies the rule of section 5.6 to every material that may be split and merged, in the given cell blocks of
// `grid`, the grid the try is grouped for (kGroupingStale otherwise): every particle whose size_level differs from
// its active cell's level on grid is split (too large) or merged with its cell's like particles (too small); every
// other particle's merge_wait_passes becomes 0. Called by ResizeCells at every resize pass, also when no cell
// changed. DECIDED name. Fails only on kBadInput (the range, rule of A116: a range not inside the grouping's blocks,
// or an empty one unless the grouping has no block), kGroupingStale, kTooManyParticles (waiting room or particle
// limit: the step raises the limit), kIdsExhausted, kInternalFault, kOutOfMemory, kRunOnCallFailed; the try is then
// unchanged.
std::expected<AdjustParticleCountResult, ParticlesFailure> AdjustParticleCount(
    ParticlesTry& particles_try, const grid::StaggeredGrid& grid, grid::CellBlockRange cell_blocks);
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
  double current_volume = 0;       // m3: the volume the cell's particles hold (seeding a cell; the grid-only
                                   //   region's hand-over is not in the first version, A64).
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
  // Every array is in grouped order and has particle_count
  // entries (ingredient_fractions: particle_count * ingredient_count). An array the material does not carry, or
  // the read did not ask for, is empty. Names, units and frames as in ParticleFields.
  struct Material {
    std::uint64_t material_id = 0;
    std::uint32_t ingredient_count = 1;
    std::size_t particle_count = 0;
    // Constant during a step: always present.
    std::vector<std::uint64_t> id;
    std::vector<double> rest_volume;                 // m3.
    std::vector<grid::SizeLevel> size_level;
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
    std::vector<std::uint8_t> merge_wait_passes;     // Field::kMergeWaitPasses; materials that may be
                                                     // split and merged. A full snapshot carries it (saved state).
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
// What an output snapshot carries (render output): position, velocity.for_next_step,
// volume_ratio, and the constant arrays id, rest_volume, size_level, ingredient_fractions.
inline constexpr SnapshotFields kOutputSnapshotFields{
    .all = false,
    .fields = ToFieldList({Field::kPosition, Field::kVelocityForNextStep, Field::kVolumeRatio}),
};
// Blocking. Not inside a try. On the GPU this is the one large copy to the CPU this module makes.
std::expected<ParticlesSnapshot, ParticlesFailure> ReadSnapshot(const Particles& particles, SnapshotFields fields);
// Takes the grid the snapshot's grouping was made for (the committed grid rebuilt from the
// saved active cells). Accepts only a full snapshot whose materials match settings (kBadInput, kWrongFields) and
// whose order is the grouping of position_when_grouped on grid (kGroupingStale). Rebuilds group_begin and
// cell_begin; every value, id and next_particle_id is restored to the byte.
// Arrays are pieces of run_memory, as BuildParticles.
std::expected<Particles, ParticlesFailure> BuildParticlesFromSnapshot(const ParticlesSettings& settings,
                                                                    const ParticlesSnapshot& snapshot,
                                                                    const grid::StaggeredGrid& grid,
                                                                    RunMemory& run_memory);
// The memory estimate entries BuildParticles allocates from. The sizing estimate reads the same entries (6.4).
std::vector<MemoryEstimateEntry> BuildParticlesMemoryEstimate(const ParticlesSettings& settings);
```

### 3.8 `particle_groups.h` (CPU only; `_cpu.cc`)

For the batched run of the cell resize and batches design: the whole try lives in CPU memory, one loaded try on the GPU
holds the groups of one `grid::GridPiece`. The loaded try is a `ParticlesTry` of a `Particles` built with `RunOn::kGpu`
and, per material, `max_particle_count` = the batch plan's loaded particle limit (`BatchPlan::max_loaded_particles`). It
is never committed.

```cpp
// Replaces every value of `loaded` by the groups of the blocks of `piece`, in the piece's block order (the
// grid's fixed order): for piece block b, the particles of whole-grid block piece.block_in_whole_grid()[b], in
// their grouped order, with the newest values of `whole` of every carried array (id, mass, rest_volume,
// size_level, position_when_grouped, merge_wait_passes and the fractions included; velocity.for_this_step
// when `whole` wrote it).
// group_begin and cell_begin are rebuilt for piece.grid(); the loaded grouping is marked as made for
// piece.grid().version(); the loaded try's earlier writes and removal marks are forgotten; `loaded` remembers
// block_in_whole_grid for ReadParticleGroups (for a piece on the GPU, read back once from
// block_in_whole_grid_buffer(): 4 bytes per loaded block, one counted copy, returned in the result).
// Fails: kGroupingStale (whole has waiting particles, or its grouping is not for the whole grid of piece),
// kTooManyParticles (the piece holds more particles of a material than loaded can), kWrongFields (the two
// tries do not carry the same arrays), kRunOnCallFailed. On failure `whole` is unchanged and `loaded` holds no
// usable grouping (kGroupingStale until the next successful call).
struct WriteParticleGroupsResult {
  std::uint32_t copy_count = 0;
  std::uint64_t bytes_copied = 0;
};  // The read of the piece map: 0 and 0 for a CPU piece.
std::expected<WriteParticleGroupsResult, ParticlesFailure> WriteParticleGroups(
    ParticlesTry& loaded, const grid::GridPiece& piece, const ParticlesTry& whole);
// Copies back, for the blocks of `range` only (whole-grid block indexes; it must be the range of the piece last
// written into `loaded`), the arrays named in `fields` from `loaded` into the try buffers of `whole`, and marks
// them written for those blocks (CommitTry's kArrayPartlyWritten check counts them). Particles of `range`
// removed in `loaded` are marked removed in `whole` with the same reason (their amounts were returned by that
// RemoveParticles call). Values of border blocks are not copied.
// Fails: kBadInput (range is not the loaded piece's range), kGroupingStale (loaded has waiting particles or was
// regrouped since WriteParticleGroups), kWrongFields (a field the material does not carry, kMass, or
// kMergeWaitPasses: no stage of a loaded try changes it),
// kRunOnCallFailed. On failure `whole` is unchanged.
std::expected<void, ParticlesFailure> ReadParticleGroups(const ParticlesTry& loaded, grid::CellBlockRange range,
                                                         FieldList fields, ParticlesTry& whole);
```

## 4. Consumes from other modules

Each item is marked EXISTS (Foundation, including the `base/` additions) or PROVIDED (by a designed module,
with the name its section 3 has).

**Foundation (EXISTS):** `Vec3`, `Mat3`, `RunOn`, `CheckCanRunOn(RunOn)`, `FLUIDITY_CPU_AND_GPU`, `base/cpu_only.h`, and
these, all under `src/fluidity/base/`, with the declarations of their headers:

- `PerParticleVelocity` (`per_particle_velocity.h`; `for_next_step`, `for_this_step`, A1).
- `Buffer`, `BufferError`, `CpuMemory`, `BuildBuffer`, `CopyBuffer` (whole and slice forms), `WriteBuffer`, `ReadBuffer`
  (`buffer.h`): the storage of every per-particle array. `CopyBuffer` on one processor serves `ApplyParticleLimits`;
  the slice forms serve `WriteParticleGroups` and `ReadParticleGroups`.
- `RunMemory`, `BuildBuffer(RunOn, std::size_t, RunMemory&)` (`run_memory.h`): the run's one block per processor.
- `OrderBuffers`, `BuildOrderBuffers`, `BuildStableOrder(keys, count, previous_order, order, OrderBuffers&)`,
  `ComputeExclusivePrefixSum(values, count, sums, OrderBuffers&)` (`stable_order.h`). Grouping uses two passes of
  `BuildStableOrder`: by id, then by cell place index.
- `SumBuffers`, `BuildSumBuffers`, `ComputeTreeSum(values, count, SumBuffers&)`,
  `ComputeLargestMagnitude(values, count, SumBuffers&)` (`ordered_sums.h`).
- `math::TreeSum` (`tree_sum.h`).
- `MemoryEstimateEntry`, `CountedPer`, `KeptFor`, `MemoryWhere`, `MemoryCounts`, `ComputeRunMemoryBytes`
  (`memory_count.h`). `BufferError::kRunMemoryTooSmall` maps to `kOutOfMemory`.
- `RunOnCallStatus` (`run_on_call.h`), `ToBufferError` (`buffer.h`).
- `GpuArray` and the launch helpers (`gpu_launch.cuh`).
- `math::Sqrt`, `Abs`, `IsFinite` (`scalar_math.h`; `IsFinite` of `Vec3` in `vec3.h`, of `Mat3` in `mat3.h`) and
  `math::Dot` (`vec3.h`).

**Contact (PROVIDED, contact 3.2):** `contact::PlateSide { std::uint32_t plate_id; SolidSide side; }`
(`contact/solid_shapes.h`), 8 bytes, equality means same side, `{0, SolidSide::kNoPlate}` means no plate.
`contact::MoveParticles(const MoveParticlesInputs&, const MovedParticles&, ContactBuffers&)` takes raw arrays, not
`ParticleFieldsToChange`: the step fills `MoveParticlesInputs` from the read pointers and `MovedParticles` from the
write pointers of the views. Contact fails a call in which an output shares a byte with an input (`kBadInput`),
except that an output may be exactly its own input array (5.3); this is NEW in contact, made by the first task of
the particles plan.

**Grid (PROVIDED, grid 3.1 to 3.7, as built; grid design 5.2, 5.8, 12):** `grid::StaggeredGrid` (`version()`,
`tables(run_on)`, `all_blocks()`), `grid::GridVersion`, `grid::GridTables`, `grid::CellBlockIndex`,
`grid::CellBlockRange { first_block; block_count; }`, `grid::CellPlaceIndex`, `grid::FacePlaceIndex`, `grid::kNoIndex`,
`grid::SizeLevel` (an enum class over `std::uint8_t`; `ToSizeLevel(CellId)`, `ToSizeLevel(tables, CellBlockIndex)`,
`ComputeCellWidth(GridFrame, SizeLevel)`), `ToCellBlockIndex(cell)`, `ToCellId(tables, cell)`,
`ComputeCellCenter(tables, cell)`, `grid::ComputeCompleteWeightedSpread(cell_width)`, `BuildRanges`, and, for 3.8, the class `grid::GridPiece`
(`grid()`, `range()`, `range_in_whole_grid()`, `block_in_whole_grid()`) with its blocks in the grid's fixed order
(grid 3.4). `weighted_spread` is a word owned by grid; this module uses it only through
`ComputeCompleteWeightedSpread`.

- `FindCellAt(tables, position)` and `FindCellAt(tables, position, CellBlockIndex near)` are total and return
  `kNoIndex` outside the grid.
- `GridPiece::block_in_whole_grid()` is empty for a piece on the GPU, whose map is `block_in_whole_grid_buffer()`.
  For a piece on the GPU `WriteParticleGroups` reads `block_in_whole_grid_buffer()` back once per load (4 bytes per
  loaded block, one counted copy) into the loaded try's own table, the loaded block origin of 6.4, which
  `ReadParticleGroups` then uses.
- `BuildRangeWithBorder` returns `std::expected<RangeWithBorder, GridFailure>`. An empty range is `kBadInput` in every
  function that takes a range, except over a grid with no block (A116); this module's functions that take a
  `CellBlockRange` follow the same rule.
- `ComputeNewCellSizeLevel` and `FindNearestCommittedCell` do not exist: `FindNewCellNeighbors` does their work. The
  step uses it; this module does not.
- **A position belongs to the cell `FindCellAt` returns.** A cell owns its low sides, not its high sides. A computed
  cell corner (`origin + c * width`) or a position one rounding step below a side may belong to the neighbouring
  cell, in frames whose origin or width is not a power of two especially (the grid's framed test grids needed one or
  two `std::nextafter` steps up, up to 17 for a corner near zero). Grouping therefore uses `FindCellAt` at the exact
  double position, never a cell computed from the position by hand, and a test position for a given cell is one
  `FindCellAt` gives to that cell.
- **In a grid piece** a reader called outside the border rule returns none, like the edge of the grid: `FindCellAt`
  for a position outside the range's cells gives a border cell or `kNoIndex`, and a particle on the high side of a
  range cell belongs to the next cell, possibly a border cell. A stage keeps each particle in a cell of its range as
  `FindCellAt` gives it; a caller can check `ToCellBlockIndex` of the cell against `GridPiece::range()`.
- **Versions of pieces.** Values (and recorded grid versions) are checked against a grid by version number and block
  count only. A piece has the whole grid's version and its own block count, so two pieces of one whole grid with equal
  block counts pass each other's check. A piece's version has no load counter: loaded groups are bound again at every
  load and never read through another piece (A124).

**Transfer (PROVIDED, transfer 3):** `ParticlesToGrid`, `GridToParticles` and `MixParticleVelocity` change
`velocity.for_next_step`, `velocity.for_this_step`, `fitted_velocity_change`, `flags` through
`ParticleFieldsToChange`; `GridToParticles` turns `kHasOwnMotion` on and turns `kCarriesFittedChange` off when it
could not fit. Its sums over one face run block by block and, inside a block, in this module's grouped order
(cell place, then id). Its material-subset input (`material_is_counted`, A8, A10) chooses which materials' views
it reads; nothing changes here.

**Materials (PROVIDED, materials 3):** `materials::BuildParticlesSettings` gives one `MaterialParticlesSettings`
per material (ingredient count, `ExtraFields` and `CountRule` from `MaterialKind`: liquid and gas
`kMayBeSplitAndMerged`; snow and springy solid `kKeepsItsCount`); `materials::ApplyMaterialResponse` changes
`volume_ratio`, `springy_deformation`, `permanent_volume_change`; `materials::ComputeOwnMotionShares` reads the
views.

**Solids (PROVIDED, solids 3.8):** `solids::ApplyPlateSides` and `ApplyPlateRemoval` set and clear `plate_side`
through `particles::ArrayToChange<contact::PlateSide>`.

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

**Cell resize and batches (PROVIDED, cell resize and batches 3.4 to 3.7):** `ResizeCells` calls
`AdjustParticleCount(particles_try, grid, cell_blocks)` on the resized grid after `GroupParticlesByBlock` for it, inside
the resize transaction, and regroups after it; it uses `ComputeRefillParticle`'s pattern
(8 per cell) for the seeded count. It calls `AdjustParticleCount` at **every** resize pass over all blocks, and its
"nothing to do" check (cell resize 5.1 step 3) counts mismatched particles, so a pass with no cell change still runs it
when any exist. Not in the first version (A64; later group "Deep liquid without
particles" of the cell resize plan): `ApplyRegionTransfer` builds `RefillCell` values, calls `ComputeRefillParticle`,
`AddParticles(..., AddReason::kRefillRegion)` and `RemoveParticles(..., RemoveReason::kIntoGridOnlyRegion)`, keeps the
ledger from the returned `ParticleAmounts`, and fills `is_grid_only_region`. `WriteBatch` and `ReadBatch` call
`WriteParticleGroups` and `ReadParticleGroups` (3.8); the batched `BuildSimulation` builds the whole particles with
`CpuMemory::kLockedForCopies` and the loaded particles on the GPU. The four cell merge and split settings
(`merge_width_share`, `merge_wait_duration`, `min_merge_wait_steps`, `max_cell_splits_per_resize`) are declared with
cell resize (A124); this module declares none of them. `kMaxMergeWaitPasses` is this module's own constant.

**I/O and CLI (PROVIDED, I/O 3):** the ready scene gives every authored snow or springy solid object its own
`material_id` when two objects must stay distinguishable; `saved_state` stores a full `ParticlesSnapshot`
(member by member as in 3.7, `merge_wait_passes` included); `RequestedOutputs::particle_fields` is a
`particles::FieldList`. Start particles are seeded by I/O (`io::BuildStartParticles`) on the start
cells of `step::BuildStartCells` at their own level: each start cell of width `w` gets the 8 points at its
cell-eighth centres (offsets `±w/4`, the order of `ComputeRefillParticle` with `n = 8`), a point kept when it lies
inside its start region and outside every solid, `rest_volume = (w/2)³`, `mass = density x rest_volume`,
`size_level` = the cell's level; `step::BuildSimulation` hands them to this module, so they start matching their
cells with `merge_wait_passes` 0. Sources keep `rest_volume = (w_min/2)³` (they are at the surface).

**Surface, foam and spray (PROVIDED, render output 3):** reads an output snapshot
(`kOutputSnapshotFields`) and writes the simulation-particle playback files (A14); reads `ParticlesTry` views in
`make_foam_and_spray`.

**For the sand module:** sand particles use the snow layout (`springy_deformation`, `permanent_volume_change`,
`CountRule::kKeepsItsCount`, `plate_side` when cuttable). Nothing else is asked of this module.

## 5. Method

### 5.1 Layout: one group of arrays per material

Each material of the scene has its own arrays. Common arrays (every material): `id`, `position`,
`position_when_grouped`, `velocity.for_next_step`, `fitted_velocity_change`, `mass`, `rest_volume`,
`volume_ratio`, `flags`, `size_level`; inside a try also `velocity.for_this_step`. Every material with
`CountRule::kMayBeSplitAndMerged` also carries `merge_wait_passes` (1 byte). Extra arrays exist only when
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
   A stage that changes an array for the second time in a try gets `read == write`. Every stage visits one particle
   at a time and reads that particle's own values before it writes them, so in place gives the same bytes.
   `contact::MoveParticles` accepts `MovedParticles::position == MoveParticlesInputs::start_position` and
   `velocity_for_next_step == velocity.for_next_step` (the same pointer; any other overlap is still `kBadInput`); the
   step passes them so in every step, since `MixParticleVelocity` writes the velocity the contact call then reads and
   writes. After a failed stage the try's arrays are unspecified and the try can only be rejected.
3. `GroupParticlesByBlock` writes every array of a material into the other buffers (it is a full reorder).
4. `CommitTry` checks that each changed array was written for all cell blocks (`kArrayPartlyWritten`; a
   `ReadParticleGroups` of a range counts as a write of its blocks, 5.10), then
   flips "which buffer is committed" for the changed arrays, sets `time`, adds one to `state_version`.
5. `RejectTry` forgets which buffers were written. Ids, counts, versions and values are as before `BuildTry`.

Fixes: no copy per try and no third "frame start" copy (`liquid_state_3d.cc:118-133, 715-726`). Invariant: no
write pointer handed out by a try ever points at a committed buffer.

### 5.4 Split 1 to 8 (maths ported unchanged)

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

Dropped: the exact-division and increment checks, the cubic-normal variant, the caller's callback.

A split is never stopped by what surrounds the particle. Mass, rest volume, momentum, angular momentum and the
particle-level energy `energy(p)` (5.5) are kept by algebra whatever the faces nearby (the offset is chosen so that
`spread(H) = spread(H/2) + offset²`). Where the faces near a child are not all there at one size (beside a solid or a
size change), the next transfer does not represent the children's spread exactly; the step sees that in
`size_change_kinetic_energy`. Why it must not be stopped: a particle moves up to half a cell per step and a size
change can lie between it and the surface; a large particle carried up into the surface band would otherwise stay
large at the surface (`tradeoff-particles.md` 1.2 "Leak", 2.2), and small particles carried into coarse cells
would pile up beyond eight per cell.

### 5.5 Merge 2 to 8 into 1 (maths ported unchanged; plain doubles)

Inputs: n particles of one material, same `plate_side`, `flags` and `size_level`, small cell width `h`, sorted by id. All sums run in that order.
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

The merge is done only if `gap > kMergeGapFloor`; a chunk that fails the gap test is counted in
`skipped_too_spread_count` and retried. The gap test is the row-sum bound of `main`
(`liquid_resampling_3d.cc:244-259`) without interval arithmetic; `kMergeGapFloor = 1e-9` covers the rounding of
at most 28 pair products. With a positive gap the energy lost is zero or more. Eight children fresh from a split
have gap exactly zero and are not merged back until they have moved closer; `main` behaves the same.
Conserved: mass, rest volume, current volume, each ingredient volume, momentum, angular momentum including the
fitted part. Not conserved, and returned: kinetic energy (numerics report 7.3). The merged particle gets
a new id, `size_level - 1`, and the common flags.

**Property.** Over one cell, the chunks are a partition of the too-small particles: every such particle is in exactly
one chunk, a chunk holds particles of one class and one active cell only, each class is cut into runs of eight in
ascending id with the remainder last, and the chunks are the same for the CPU, the GPU and every cut of the blocks
into ranges. Reference: one flat list of all particles of the material sorted with `std::sort` by (cell from
`FindCellAt` without `near`, class, id). Evidence: property.

### 5.6 `AdjustParticleCount` (new: `main` has no caller that chooses)

Called by the cell resize design in step 7 of `ResizeCells` at **every** resize pass, over all
blocks, inside the resize transaction, once the try is grouped for the resized grid `grid`. For each
material with `CountRule::kMayBeSplitAndMerged`, for each cell block of `grid` in the range, in block order:

0. **Candidates.** A particle is a candidate when its `size_level` differs from the level of its active cell on
   `grid` (counted in `mismatched_particle_count`). `kCarriesFittedChange` does not matter: a particle with it off
   is a candidate and its `C` is read as zero (its children and its merged particle keep the flag off). Every
   particle that is not a candidate gets `merge_wait_passes = 0`.
1. **Too large** (`size_level` below the cell's level `c`: the particle is wider than its cell), in grouped order
   (cell place, then id): split by 5.4 (offset `(√3/4) H`, children `v + C·offset`, mass and rest volume / 8;
   `volume_ratio`, `C`, `flags`, fractions, `plate_side` copied; `size_level + 1`; `merge_wait_passes` 0). Nothing
   stops it (5.4).
   **Surface safety:** when `c` is the frame's smallest size level (`size_level_count - 1`), the split repeats
   until the descendants match, `k = c - size_level` generations (at most `size_level_count - 1`) in this pass:
   the `8^k` descendants of generation `k` are computed in one go (intermediate generations are never written),
   each `size_level = c`, in descendant order (first generation's child index most significant), so no too-large
   particle reaches the next transfer in the surface band. In any other cell one generation per pass. A
   descendant that lands in a cell of another level is a candidate of the next pass; one that lands inside a solid
   is handled by the next `MoveParticles` like any particle that starts inside (`started_inside_count`).
2. **Too small** (`size_level` above `c`), per active cell of the block in cell-place order: candidates form
   classes by (material, `plate_side`, `flags`, `size_level`), classes in order of their first member's id, each
   class in ascending id: the **stable order** (cell place, then id), the grouping's order, so the CPU, the GPU
   and any range or batch cut take the same chunks (all of a chunk lie in one cell, so in one block). Each class
   is cut into chunks of eight in that order, the remainder last.
   - A chunk of exactly eight is merged by 5.5 (with width `h` of the particles' `size_level`) at once.
   - A remainder of two to seven: every member's `merge_wait_passes` grows by one (saturating at 255); if the
     chunk's first particle then has `merge_wait_passes >= kMaxMergeWaitPasses` (16), the chunk is merged by the
     same maths, else it waits (`waiting_particle_count`). A group that appears at pass 1 is merged at pass 16,
     not 15.
   - A remainder of one cannot merge: its `merge_wait_passes` grows likewise; it stays, a valid particle,
     counted in `lone_particle_count`. If a later pass gives it company, its old count lets the new chunk merge
     at once.
   - The only refusal is the gap test (`gap <= kMergeGapFloor`): the chunk is not merged,
     adds one to `skipped_too_spread_count`, its members count as waiting and keep their counts, and it is
     retried at the next pass.
   A merged particle has `size_level - 1`; if that still differs from `c` it is a candidate of the next pass.
3. **Per-field rule of a merge** (the 2-to-8 maths of 5.5, stated per field): mass and rest volume: sums;
   position: mass-weighted centre; velocity: mass-weighted mean (momentum exact); `C*`: the formula of 5.5
   (angular momentum exact); `volume_ratio`: rest-volume-weighted; ingredient fractions: current-volume-weighted;
   `flags`, `plate_side`, material: common (equal by the class rule); `size_level`: one less; id: new;
   `merge_wait_passes`: 0. A split copies every per-particle field except mass and rest volume (divided by 8),
   position, velocity, `size_level`, id and `merge_wait_passes` (0). Material fields: only materials with
   `kMayBeSplitAndMerged` take part (liquid, gas markers), and they carry no per-particle material state beyond
   `volume_ratio` and the fractions (materials design 11, decision 15); snow and springy solids
   (`kKeepsItsCount`: `springy_deformation`, `permanent_volume_change`) are never split or merged, so no rule is
   needed for those fields.
4. **Conservation and report.** Mass and rest volume exact up to the fixed-order sums (each split in child order,
   each merge in id order); momentum exact (split offsets sum to zero; merged velocity is the mass-weighted
   mean). With `energy(p) = (m_p/2)(|v_p|² + spread(width_p) |C_p|²_F)`, `spread(w) = w²/4`, `width_p` the width
   of the particle's `size_level` and `C_p` zero when `kCarriesFittedChange` is off: per block, `energy(after) -
   energy(before)` summed over its splits (grouped order) then its merges (chunk order), then `math::TreeSum`
   over the block partials in block order, is `kinetic_energy_change`: 0 up to rounding for splits (the spread
   term absorbs the offsets), `<= 0` for merges with a positive gap (`-kinetic_energy_lost` of the merges).
   Integer counts are sums of block partials.
5. **Bound.** A group of two or more same-level particles of one class persists at most `kMaxMergeWaitPasses`
   passes after its first member became too small (so at most 16 after it stops gaining members), unless the
   gap test refuses it; at most one lone particle per cell, material, side, flags and level persists without a
   time bound. Both are reported (`waiting_particle_count`, `lone_particle_count`; the step's
   `StepReport::mismatched_particle_count`).

Ids (plan Global Constraints): blocks in order; inside a block first the split descendants of each split
particle in grouped order (`8^k` consecutive ids per parent, descendant order), then the merged particles in cell
place order, chunks in order. Places in the waiting list come from integer prefix sums of per-block counts, so a
pass that would exceed `max_added_per_try` or the particle limit writes nothing (`kTooManyParticles`; the step
raises the limit, as today). Memory: `merge_wait_passes`, 1 B per particle in two buffers (entry
`"particles: merge wait passes"`, 6.4).

All particles of a merge are in one active cell, so a cell block never needs another block's particles. Split
children may land in a neighbouring block; they wait in the try and are placed by the next grouping.

**Particle size follows the cell (A65; deep cells, A64).** Every liquid cell holds about eight particles of its own
size: a particle's `size_level` is its cell's level, its rest volume is one eighth of its cell (`(w/2)³` for a cell of
width `w`), so a cell k levels coarser than the smallest holds 8 particles each of `8^k` times the smallest particle's
volume. This holds at every depth (A64: every liquid cell holds particles). The start state is already so (A66: I/O
seeds each start cell at its own level, section 4). When a resize merges eight cells of level `l + 1` into one of
level `l`, step 2 merges their particles into ones of level `l`; when it splits a cell, step 1 splits its particles. A
resize changes a cell by one level at most, so a cell that coarsens by `k` levels gets its particles merged over `k`
resizes, one level per resize. Particles that drift across a size change are split or merged by the same steps at the
next passes, so a cell does not keep more or larger particles than its size wants beyond the bound of item 5. No code
beyond 5.4 to 5.6 is needed; the test `AdjustParticleCount, ThreeLevelCascade_KeepsAmounts` (section 7) and cell
resize's `ResizeCells, CalmDeepTank_CoarsensWithParticlesInEveryCell` prove the cascade.

**Property.** After one pass over all blocks, the particles split are exactly those of a `kMayBeSplitAndMerged`
material whose `size_level` was below the level of the active cell `FindCellAt` gives for their position on `grid`; in
a cell of the smallest size level no descendant is left bigger than the cell; no particle of a `kKeepsItsCount`
material changes. Reference: a per-particle loop over the snapshot before the pass, using `FindCellAt` and
`ToSizeLevel` only (no `cell_begin`, no grouping). Evidence: property; the formulas by value.

### 5.7 Add, refill, remove

- `AddParticles` checks values (finite, positive mass and volumes, fractions sum to 1), gives ids in input order,
  and appends to the try's waiting list. Both reasons start with `kHasOwnMotion` off, as on `main`: a new
  particle has no motion of its own to keep (numerics report 7.4). The reason only labels the returned amounts:
  `kFromSource` is new material, `kRefillRegion` is material that was already counted on the grid: gas markers
  added by `gas::FindMarkerChanges` (the gas design), and later liquid handed over by a grid-only region (not in the
  first version, A64).
- `ComputeRefillParticle` (ported): for `n` particles in a cell of width `h`, particle 0 is at the centre when `n`
  is odd; the others go in pairs to opposite `cell_eighth` centres (`+-h/4` per axis), lowest eighth first, with
  x as bit 0. `mass = density * current_volume / n`, `rest_volume = current_volume / (n * volume_ratio)`,
  `velocity = cell velocity + C * offset`. With `n = 8` (the seeding and refill count, A65) the eight particles
  sit at the cell-eighth centres in the order 0, 7, 1, 6, 2, 5, 3, 4, and for `volume_ratio` 1 and
  `current_volume = w³` each has `rest_volume = (w/2)³`, as the size rule of 5.6 says.
  I/O's start seeding (A66, section 4) uses the same points and order.
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

The cell place is in the key because transfer needs the particles of one cell together and a first-particle-per-cell
table. A block is contiguous, a merge reads one active cell, and sums have a fixed order. Cost: `cell_begin`, 256
bytes per cell block per material in each of the two buffers; ids ascend inside a cell place, not inside a whole
block; the sort has the same two passes, with a key 6 bits wider.

`CheckGrouping` fails with `kMovedTooFarSinceGrouped` when any axis of `position - position_when_grouped` exceeds
`max_move_since_grouped`. Not ported: the 1.5-cell shifted bins and incoming-order ties.

**Property.** After grouping, the arrays are a permutation of the kept and the waiting particles without the removed
ones; particle i lies in the group of block b exactly when `ToCellBlockIndex(FindCellAt(tables, position_i))` is b,
and in the run of cell place c given by `cell_begin` exactly when `FindCellAt` gives c; ids ascend inside a cell
place. Reference: `FindCellAt` (the form without `near`) per particle and `std::stable_sort` of (cell place, id)
pairs; the test grid holds positions on a cell side and one `std::nextafter` step below it, where the cell changes.
Evidence: property; `group_begin` and `cell_begin` of one small case by value.

### 5.9 Snapshots

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
  `ingredient_fractions`. A full snapshot also carries `merge_wait_passes`, so a continued run merges at the same pass
  as the uninterrupted one. `RequestedOutputs::particle_fields` of I/O adds named fields to it (`kMass` included).
- Arrays the material does not carry are empty in both forms; asking for one is not an error (playback asks
  for the same fields of every material).

### 5.10 Loaded groups for a batched run

In a batched run (cell resize and batches, 5.7 to 5.10) the whole try lives in CPU memory and the stage functions run
unchanged on a loaded try on the GPU, built over a `grid::GridPiece` (range plus stacked border, blocks in the
grid's fixed order).

1. `WriteParticleGroups` first, for a piece on the GPU, reads `block_in_whole_grid_buffer()` back once (4 bytes per
   loaded block, one counted copy, returned in `WriteParticleGroupsResult`; 0 and 0 for a CPU piece) into the loaded
   try's own table. It then copies, block by block of the piece, the whole try's group of the matching whole-grid
   block (`block_in_whole_grid`), every carried array, with slices of `WriteBuffer` (one slice per block and
   array; blocks of a range are one run in the whole arrays, so a range costs one slice per array). Inside a
   block the order is the whole try's grouped order, so `cell_begin` of a block is copied unchanged and
   `group_begin` is the running sum of the block counts. Particle `i` of loaded block `b` is particle
   `group_begin_whole[block_in_whole_grid[b]] + (i - group_begin_loaded[b])` of the whole try.
2. Stages change the loaded try through `BuildFieldsToChange` as in a one-piece run. `RemoveParticles` on the loaded try
   marks particles. No stage adds particles to or regroups a loaded try (stages `AddNewParticles` (1),
   `ChooseActiveCells` (2) and `ResizeCells` (21) run on the whole state on the CPU); `ReadParticleGroups` refuses one
   that was (`kGroupingStale`).
3. `ReadParticleGroups` copies the named arrays of the range's blocks back into the whole try's write buffers
   (5.3 step 2 applies: the first copy of an array points it at the try buffer), carries the removal marks of
   the range's particles, and records the blocks as written. Border blocks are not copied back: the batch that
   owns them computes the same bytes (cell resize and batches, 5.10).
4. Every per-particle value depends only on the particle and its neighbourhood, and every sum of this module is
   per block then `math::TreeSum` over blocks, so this module's operations give identical bytes for any cut of
   the ranges, given the same inputs. A batched step as a whole agrees with a one-piece step within the solver's
   tolerance, not byte for byte (A15: the batched solves), and gives identical bytes for every cut of one mode.

**Property.** The loaded try holds exactly the particles whose cell block is a block of the piece (range and border),
each once, in the whole try's order inside a block; a stage run on the loaded try writes, for the particles of the
range's blocks, the same bytes as the same stage run on the whole try; `ReadParticleGroups` changes in the whole try
the particles of the range's blocks and no other. Reference: the whole try's particles filtered by the block of
`FindCellAt`, and the stage run on the whole try. Evidence: property.

### 5.11 Rules the code is built on

These fix what 5.3 to 5.10 leave open. Every task follows them.

1. **Indexes.** Every array pointer of `ParticleFields` and `ParticleFieldsToChange` points at particle 0 of the
   material. `group_begin` points at the entry of `cell_blocks.first_block`, so its values are whole-material
   indexes; the range's particles are `[group_begin[0], group_begin[block_count])` and `particle_count` is their
   number. A `particle_index` given to `RemoveParticles` is such an index. A user of `ParticleFieldsToChange` that
   needs `group_begin` takes the matching `ParticleFields`.
2. **Written marks, per block.** Each material keeps one `FieldList` per cell block: the arrays written in this try.
   `BuildFieldsToChange` for a range: no block of the range written: `read` is the committed buffer and `write` the
   other; every block written: both are the try buffer; some written: the committed values of the unwritten blocks
   of the range are first copied into the try buffer (one slice `CopyBuffer` per run of blocks), then both are the
   try buffer. The range is then marked written. `velocity.for_this_step` has one buffer; its `read` is null unless
   every block of the range was written in this try.
3. **Removal marks.** One byte per grouped particle, try only: 0 kept, `1 + RemoveReason` removed, 255 replaced by a
   split or a merge (not a removal, not in any removed amount).
4. **Waiting list.** One per material, `max_added_per_try` entries of every array the material carries except
   `position_when_grouped` and `velocity.for_this_step`. Ids come from one `next_particle_id` for all materials and
   start at 1.
5. **Grouping.** Two `BuildStableOrder` passes with the storage's `order_buffers`: by id, then by the
   `CellPlaceIndex` of `FindCellAt`. A removed or replaced particle gets the key `UINT64_MAX` before the cell is
   looked up, and is dropped: a removed particle outside the grid is not `kOutsideGrid`. The input is the grouped
   particles, then the waiting ones. An array whose newest values are committed is gathered into the other buffer.
   An array already changed in this try (a second grouping in one try) is regrouped in place through a working
   array of 8 bytes per particle: each 8-byte part of the element (1 byte for byte arrays) is gathered into it and
   written back. A committed buffer is never written. A grid with more blocks than `max_cell_blocks` is
   `kTooManyParticles`, magnitude the block count.
6. **Failures on the GPU.** The lowest failing id is found with an integer `atomicMin` into a one-entry buffer the
   storage holds, read back by the CPU half. No floating-point atomic anywhere.
7. **Sums.** `ComputeAmounts`: per block in grouped order (`mass`, `rest_volume`, `volume_ratio * rest_volume`,
   `mass * velocity.for_next_step` per component), then `math::TreeSum` over the block partials per component; the
   tree sum runs on the CPU for both processors and the per-block function is shared, so the header promises the
   same bits on both unless NaN. Added and removed amounts: in input or index order. Merge sums: ascending id.
8. **Memory estimate entries**, per material, all `CountedPer::kGivenCount`, `where = kRunOn`, `kWholeRun` unless
   said. `copies = 2`, `given_count = max_particle_count`: `id` 8, `position` 24, `position_when_grouped` 24,
   `velocity.for_next_step` 24, `fitted_velocity_change` 72, `mass` 8, `rest_volume` 8, `volume_ratio` 8, `flags` 1,
   `size_level` 1, `merge_wait_passes` 1 (count rule), and when carried `ingredient_fractions` 8 N,
   `springy_deformation` 72, `permanent_volume_change` 8, `plate_side` 8; `group_begin` 8 (`max_cell_blocks + 1`
   items) and `cell_begin` 4 (`max_cell_blocks * 64`). `copies = 1`, `kOneStepTry`: `velocity.for_this_step` 24 and
   removal marks 1 (`max_particle_count`); written marks 2 and loaded block origin 4 (`max_cell_blocks`).
   `copies = 1`: grouping keys 8 and two orders of 4 (`max_particle_count + max_added_per_try`), the regroup working
   array 8 (`max_particle_count`), the waiting list (`max_added_per_try`). Once per `Particles`: `"particles: order
   buffers"` and `"particles: sum buffers"` (one entry each, `given_count` 1, `bytes_each = bytes_per_item x
   max_count + fixed_bytes` of `ComputeOrderBuffersSize` / `ComputeSumBuffersSize`), `"particles: lowest failing
   id"` (1 x 8) and `"particles: GPU failure"` (1 x `sizeof(ParticlesFailure)`). One function walks a material's
   arrays for both `BuildParticles` and the estimate, so they cannot drift. One liquid: 407 B per particle. On the
   GPU the count rule adds `"particles: count rule counts"`: 4 B per particle and 4 B per cell place
   (`max_cell_blocks * 64`), for the counts and places of its prefix sums.
9. **Test fixtures and random values.** Tests draw values with `SplitMix64` (seed 20261114, doubles as
   `(x >> 11) * 0x1p-53`), never a `<random>` distribution, which differs between standard libraries.

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
- `AdjustParticleCount`: one thread per particle for the split count and the split, one per active cell for the merge chunks;
  each thread writes its own fixed places in the waiting list (place = prefix sum of per-thread counts, an
  integer scan), so results do not depend on thread timing. Ids are `next_particle_id + place`. A
  surface-cell parent's thread writes all `8^k` descendants (its count in the scan is `8^k`); the per-cell
  thread forms the classes and chunks in the stable order of 5.6 and writes `merge_wait_passes` of its cell's
  particles, so the CPU, the GPU and any range cut give the same bytes.
- Sums: `ComputeAmounts` writes one partial per block (one thread per block, grouped order) and sums the
  partials with `ComputeTreeSum` (`base/ordered_sums.h`), which follows `math::TreeSum`: the CPU and the GPU sum
  in the same order. Merge sums are per thread, at most eight terms, in id order. No floating-point atomics
  anywhere.
- `WriteParticleGroups`, `ReadParticleGroups`: CPU code issuing slice copies (`WriteBuffer`, `ReadBuffer` with
  `first`), one per block run and array; `group_begin` of the loaded try is an integer scan on the CPU.
- `AddParticles` has no `_gpu.cu`: it validates the CPU spans and sums amounts on the CPU, then writes the
  waiting list with one slice `WriteBuffer` per carried array; each call is a counted copy
  (`AddParticlesResult::copy_count`, `bytes_copied`; A17, plan-decisions "Copies counted").
- Returned to the CPU: one status record per call (`ParticlesFailure`, counts, the `AdjustParticleCountResult` numbers),
  not counted (a status record). The reported failure is the one with the lowest particle id (an integer minimum).
  `ReadSnapshot` is the only large copy and is explicit.
- Status until the owner's hardware run: "GPU: compiled, not run".

**6.4 Memory-count table** (bytes per particle; all values are doubles as on `main`, precision is not tuned here).

| Row | Bytes | Buffers | Who has it | Kind |
|---|---|---|---|---|
| `id` | 8 | 2 | all | committed + try |
| `position`, `position_when_grouped`, `velocity.for_next_step` | 24 each | 2 | all | committed + try |
| `fitted_velocity_change` | 72 | 2 | all | committed + try |
| `mass`, `rest_volume`, `volume_ratio` | 8 each | 2 | all | committed + try |
| `flags`, `size_level` | 1 each | 2 | all | committed + try |
| `merge_wait_passes`, entry `"particles: merge wait passes"`, `kRunOn` | 1 | 2 | count rule `kMayBeSplitAndMerged` | committed + try |
| `ingredient_fractions` | 8 N | 2 | N > 1 ingredients | committed + try |
| `springy_deformation` | 72 | 2 | snow, springy solid | committed + try |
| `permanent_volume_change` | 8 | 2 | snow | committed + try |
| `plate_side` | 8 | 2 | when asked | committed + try |
| `velocity.for_this_step` | 24 | 1 | all | try only |
| grouping keys, two orders, regroup working array | 8 + 4 + 4 + 8 | 1 | all | working buffers |
| `group_begin` | 8 per cell block per material | 2 | all | committed + try |
| `cell_begin` | 256 per cell block per material (4 x 64) | 2 | all | committed + try |
| waiting list | one buffer of every carried array | 1 | `max_added_per_try` items | working buffers |
| written marks | 2 per cell block per material (a `FieldList`) | 1 | all | try only |
| loaded block origin | 4 per loaded cell block | 1 | loaded particles of a batched run | try only |

Per particle, at `max_particle_count`: one liquid 2 x 179 + 56 = 414 B; mixture of N: + 16 N; snow 2 x 258 + 56 = 572 B;
springy solid with side 2 x 258 + 56 = 572 B; gas markers 414 B. A batched run counts these entries twice: once for the
whole particles in CPU memory at `max_particle_count`, once for the loaded particles on the GPU at the batch plan's
loaded limit (the run's processor; in a CPU run with `max_loaded_blocks > 0` both are CPU memory and the copies are CPU
to CPU, counted the same). This is the placement rule of `plan-decisions.md` "Buffers in batched runs" applied to
particles: the whole particles are carried between passes and kept by the commit, so they live in CPU memory (`where =
kCpu`, part of `MemoryEstimate::on_cpu`); the loaded try is built by `step::BuildSimulation` at `max_loaded_particles`
per material and `max_loaded_blocks` (step plan Appendix A). `WriteParticleGroups` and `ReadParticleGroups` move their
arrays inside a batch load, which counts one copy (its arrays are bytes, not copies). Plan measure: 407 B per liquid
particle with entries from `sizeof` (405 B without `merge_wait_passes`; the 414 B above is the design's estimate).
`BuildParticlesMemoryEstimate` returns exactly the entries `BuildParticles` allocates from, as `CountedPer::kGivenCount`
entries with `given_count = max_particle_count` of each material. "committed + try" is `KeptFor::kWholeRun` with `copies
= 2`, "try only" is `kOneStepTry`, "working buffers" is `kWholeRun` too (every array is a piece of the run's block,
A58); the exact entries are in 5.11. The storage also holds one `OrderBuffers`
(`order_buffers`) and one `SumBuffers` (`sum_buffers`), built from the limits, for the grouping's stable orders and
prefix sum and the largest-magnitude sum (`"particles: order buffers"`, `"particles: sum buffers"` and their fixed-bytes
entries).

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
| `ComputeMergedParticle, TwoEqual_MeanAndSum` | `h = 1`; two particles of 1 kg at x = -0.25 and +0.25, x velocity -1 and +1, C = 0 | mass 2, centre 0, velocity 0, `C*.xx = 0.25`, other entries 0, gap 0.6875, energy lost 0.9375 J | 1e-14 |
| `ComputeMergedParticle, SplitThenMerge_GivesParentBack` | split a parent, merge its eight children at `h = H/2` | parent's position, velocity, C, amounts; energy lost 0; gap 0 (so a merge in `AdjustParticleCount` refuses this exact case by the gap test, as on `main`) | 1e-13 rel; gap within 1e-14 |
| `ComputeMergedParticle, KeepsEachIngredientVolume` | 3 particles, 3 ingredients, different J | `sum f J V0` per ingredient unchanged | 1e-14 rel |
| `ComputeMergedParticle, KeepsAngularMomentum` | 5 particles, random fixed values | orbital + fitted angular momentum unchanged | 1e-13 rel |
| `ComputeMergedParticle, EnergyNeverGrows` | 1000 fixed-seed cases with gap > floor | `kinetic_energy_lost >= -1e-13 * energy` | as stated |
| `ComputeRefillParticle, NineParticles_KeepCellAmounts` | V = 1, density 1000, J = 0.5 | masses sum to 1000, rest volumes to 2, momentum to `1000 v` | 1e-14 rel |
| `AdjustParticleCount, TryGroupedForOtherGrid_IsStale` | try grouped for one grid, called with a grid of another version | `kGroupingStale`; try unchanged | exact |
| `AdjustParticleCount, Snow_IsLeftAlone` | a snow material with particles whose `size_level` differs from their cell's | no split, no merge, `mismatched_particle_count` 0; arrays unchanged | bytes |
| `AdjustParticleCount, TwoFarApart_AreNotMerged` | two level-1 particles `2 h` apart in one level-0 cell, 17 passes | never merged; `skipped_too_spread_count` 1 at pass 16 and 17; both keep their values | exact |
| `AdjustParticleCount, DifferentSides_FormTwoClasses` | eight level-1 particles in one level-0 cell, four on each side of a plate | no chunk of eight: two classes of four, `waiting_particle_count` 8 at pass 1 | exact |
| `AdjustParticleCount, WhoIsSplit_MatchesPerParticleReference` | property of 5.6: 3 fixed-seed grids of 2 and 3 size levels, 200 liquid particles and 50 snow particles each at random levels | the set of split particles equals the per-particle reference's | exact |
| `AdjustParticleCount, Chunks_MatchFlatSortReference` | property of 5.5: the same grids, too-small particles of 2 plate sides and 2 flag values | the chunks equal the flat sort's, on one range and on 1-block ranges | exact |
| `GroupParticlesByBlock, Groups_MatchPerParticleReference` | property of 5.8: the same grids plus positions on a cell side and one `std::nextafter` step below it | groups and cell runs equal the reference's | exact |
| `WriteParticleGroups, LoadedSet_MatchesFilterReference` | property of 5.10: a piece of 3 range blocks and its border | the loaded ids equal the filtered whole try's, per block, in order | exact |
| `BuildFieldsToRead, RangeOutsideGrid_IsBadInput` / `EmptyRange_IsBadInput` | a range past the last block / `{0, 0}` on a grid with blocks | `kBadInput` | exact |
| `AddParticles, GivesConsecutiveIds_AndNoOwnMotion` | 5 particles | ids n..n+4, flag off, amounts returned | exact |
| `AddParticles, OnCpu_NoCopyCounted` (A17) | particles on the CPU | `copy_count` 0, `bytes_copied` 0 | exact |
| `AddParticles, OnGpu_CopiesCountedPerArray` (A17, label `gpu`) | 5 liquid particles | `copy_count` = the number of waiting-list arrays written (defaults included); `bytes_copied` = 5 x the sum of their `sizeof` | exact |
| `ApplyParticleLimits, RaisedBlockLimit_ReallocatesBlockTables` / `LowerBlockLimit_IsBadInput` | `max_cell_blocks` 64 -> 128 with particle limits unchanged / 64 -> 32 | limits, counts, versions, next id kept; allocated bytes equal the entries at 128 blocks / `kBadInput`, particles unchanged | exact |
| `RemoveParticles, MarkerNotNeeded_ReturnsAmounts` | gas markers, 3 removed with `kMarkerNotNeeded` | amounts of the three; dropped at the next grouping | exact |
| `GroupParticlesByBlock, AnyInputOrder_SameResult` | same particles in 3 shuffles, mixed-size layout | identical arrays | bytes |
| `GroupParticlesByBlock, InsideCell_IdsAscend` | 3 blocks of 2 sizes | `group_begin`, `cell_begin` and order as computed by hand | exact |
| `GroupParticlesByBlock, ParticleOutsideGrid_ReportsLowestId` | two outside | `kOutsideGrid`, the failure names the lower id | exact |
| `CheckGrouping, AfterAdd_IsStale` / `MovedTooFar` | as named | the two errors | exact |
| `RejectTry, AfterEveryKindOfChange_ChangesNothing` | add, remove, split, merge, group, change each field, then reject | snapshot before equals snapshot after, including `next_particle_id` and versions | bytes |
| `CommitTry, ChangedArraysOnly_NoCopy` | change velocity only | other arrays' buffer addresses unchanged; `state_version` + 1 | exact |
| `CommitTry, ArrayWrittenForSomeBlocks_IsPartlyWritten` | change one block range only | `kArrayPartlyWritten` | exact |
| `AdjustParticleCount, ParticleInSmallerCells_IsSplitOnce` / `EightInBiggerCell_AreMerged` | a grid that splits / merges one region and a try grouped for it (5.6) | counts in the result; amounts unchanged | 1e-13 rel |
| `AdjustParticleCount, DriftedIntoSmallerCells_IsSplitWithoutResize` | `CentreSplit` and a try grouped for it, the level-0 "split parent" at `P` | `mismatched_particle_count` 1, `split_count` 1, eight level-1 children; after grouping a second pass: `mismatched_particle_count` 0 | exact; amounts 1e-13 rel |
| `AdjustParticleCount, TooLargeInSurfaceCell_SplitToSmallestInOnePass` | grid A of `ThreeLevelCascade` (level 2 = smallest); the level-0 "split parent" at `P` | one pass: 64 particles of level 2 (`split_count` 1), ids consecutive in descendant order, each m = 0.125; `kinetic_energy_change` 0 | mass and momentum 1e-15 rel; energy 1e-13 rel of the energy before |
| `AdjustParticleCount, EightSmallInLargeCell_MergeInOnePass` | "eight close" (level 1) on `BigCells` | `merged_group_count` 1 at the first pass; one level-0 particle at `P`, mass 8 | amounts 1e-13 rel |
| `AdjustParticleCount, ThreeSmallInLargeCell_MergeAfterSixteenPasses` | the first three of "eight close" on `BigCells`; one committed resize try per pass | passes 1 to 15: `waiting_particle_count` 3, `merge_wait_passes` = pass number; pass 16 (not 15): `merged_group_count` 1, one level-0 particle of mass 3 | exact; amounts 1e-13 rel |
| `AdjustParticleCount, LoneSmallParticle_StaysAndIsCounted` | one level-1 particle at `P` on `BigCells`; 260 passes | `lone_particle_count` 1 every pass; arrays unchanged except `merge_wait_passes`, which reads 20 after 20 passes and 255 after 260 | exact |
| `AdjustParticleCount, MergeEnergy_MatchesAnalytic` | eight 1 kg particles at `(±h/4, ±h/4, ±h/4)` of a cell of width `2h` (`h` = 0.5, the "eight close" positions), velocity `(±1, 0, 0)` m/s by the sign of x, `C = 0` | merged `C*_xx = 0.25/h` = 0.5 per second, all else 0; energy before `8 x 0.5 x 1² = 4 J`, after `(8/2) h² (0.25/h)² = 0.25 J`; `kinetic_energy_change` -3.75 J, `kinetic_energy_lost` 3.75 J; gap 0.6875 | 1e-12 rel |
| `AdjustParticleCount, SplitThenMerge_MassAndMomentumExact` | split the "split parent" (regroup for `CentreSplit`, then one pass), move the children to the "eight close" positions, merge (regroup for `BigCells`, then one pass) | merged mass 8 and rest volume 2 equal the parent's; merged momentum equals the test's own child-order sum of the children's momenta | bytes; momentum against the parent 1e-15 rel |
| `AdjustParticleCount, AnyBatchCut_SameBytes` | grid A with the surface-split parent, the three waiting and the eight close; 17 passes, each with one call per range of `grid::BuildRanges(grid, n)`, n = all, 1, 3 | identical result numbers and committed snapshot bytes (`merge_wait_passes` included) | bytes |
| `AdjustParticleCount, OnGpu_SameBytesAsCpu` (label `gpu`) | every A67 case above on the GPU | identical result numbers and snapshot bytes | bytes |
| `ComputeRefillParticle, EightParticles_AtCellEighthCentres` | centre {0,0,0}, `h = 1`, V = 1, density 1000, J = 1, n = 8 | positions at `±0.25` in eighth order 0, 7, 1, 6, 2, 5, 3, 4; each `rest_volume` 0.125 = `(h/2)³`, mass 125 | exact |
| `AdjustParticleCount, AnyRangeCut_SameBytes` | the whole grid in one range, then in 1-block and 3-block ranges | identical arrays and result numbers | bytes |
| `AdjustParticleCount, ThreeLevelCascade_KeepsAmounts` | grids A (big cells 1..6 at level 1, big cells 2..5 at level 2), B (1..6 at level 1), C (all level 0), 128-block room; 64 level-2 liquid particles at the level-2 cell centres of big cell (3,3,3), `m` = `rest_volume` = 1/64, v {1,2,3}, C = 0; each step regroups for the next grid, then one pass: A to B, B to C, then C to B, B to A | `merged_group_count` 8 then 1, one level-0 particle at {3.5,3.5,3.5} of mass 1; then `split_count` 1 then 8, 64 level-2 particles; amounts equal before and after each call; `kinetic_energy_lost` 0 | 1e-13 rel; energy 1e-15 J |
| `BuildParticles, IntoRunMemory_HandsOutExactlyTheEstimate` | the six fixture materials, CPU block of `ComputeRunMemoryBytes(BuildParticlesMemoryEstimate(settings), {}, kRunOn)` | `handed_out_bytes() == reserved_bytes()` | exact |
| `ParticlesError, OutOfMemory_IsTheLastValue` | enum | `kOutOfMemory` is the 13th and last value; `ToName` `"out_of_memory"` | exact |
| `ComputeAmounts, AnyRangeCut_SameBytes` | `ComputeAmountsPerBlock` over 1-block ranges, tree-summed, against `ComputeAmounts` over all blocks | identical | bytes |
| `CheckParticles, OnTry_SeesNewestValues` | NaN written into one try array | `kBadInput` on the try; the committed check passes | exact |
| `BuildParticles, LiquidOnly_AllocatesNoExtraArrays` | one liquid | extra pointers null; bytes equal `BuildParticlesMemoryEstimate` | exact |
| `BuildParticlesMemoryEstimate, EveryMaterialKind_MatchesAllocation` | the six layouts of 5.1 | sum of entries equals bytes allocated | exact |
| `ReadSnapshot` then `BuildParticlesFromSnapshot` | all kinds | equal snapshots | bytes |
| `BuildParticlesFromSnapshot, UngroupedMove_RestoresOrder` | `max_move_since_grouped > 0`; a particle moved into another cell since the last grouping; full snapshot, rebuild, one more step in both | same order, `group_begin`, `cell_begin`; the next step's committed bytes equal | bytes |
| `BuildParticlesFromSnapshot, OutputSnapshot_IsBadInput` | snapshot read with `kOutputSnapshotFields` | `kBadInput` | exact |
| `ReadSnapshot, OutputFields_CarryConstantArrays` | liquid mixture of 3, snow | `id`, `rest_volume`, `size_level`, fractions, `position`, `velocity_for_next_step`, `volume_ratio` filled; every other vector empty | exact |
| `ReadSnapshot, ThisStepVelocity_IsWrongFields` | `kVelocityForThisStep` asked | `kWrongFields` | exact |
| `WriteParticleGroups` then `ReadParticleGroups`, `RoundTrip_SameBytes` | piece of 3 range blocks and 4 border blocks, all kinds | loaded `group_begin`, `cell_begin` as computed by hand; after changing positions in the loaded try and reading the range back, the whole try's range blocks equal the loaded values and border blocks are untouched | bytes |
| `ReadParticleGroups, EveryRangeRead_CommitSucceeds` / `OneRangeMissing_IsPartlyWritten` | 3 ranges, velocity changed in each loaded try; read back all / all but one | `CommitTry` succeeds / `kArrayPartlyWritten` | exact |
| `ReadParticleGroups, RemovedInLoaded_RemovedInWhole` | 2 particles removed in the loaded try, one in a border block | the range's one is dropped at the whole try's next grouping; the border one stays | exact |
| `WriteParticleGroups, PieceTooFull_IsTooManyParticles` | loaded limit below the piece's count | `kTooManyParticles`; whole try unchanged | exact |
| `ParticleGroups, BatchedAndWhole_SameBytes` | a stage sequence that changes velocity and position, run on the whole try and in 1-, 2- and 5-block batches | identical committed arrays | bytes |
| `ToName, EveryError_LowerSnakeCase` | every `ParticlesError` | the enumerator's name without `k`, lower snake case; all distinct | exact |

Reference: no Python piece. Every formula is closed-form and its tests compute the expected value by hand; the
four property tests use the references their sections state (5.5, 5.6, 5.8, 5.10), kept in `testing/`. Proof: one Lean file, `proofs/Fluidity/Merge.lean`, trimmed to
`merge_preserves_moments_and_bounds_energy` and `merge_preserves_volume_inventories`, restated in the new
vocabulary for the single quadratic spread. It removes one uncertainty a test cannot: that a positive gap implies
no energy gain for every admissible input, not only the sampled ones. `Resampling.lean` is not ported (the split
identity is a finite sum the tests check exactly).

Comparison with `main` (supporting only; `main` is never built): the inputs and expected values of `main`'s own
tests of `SplitUniformLiquidParticle3D`, `MergeUniformLiquidParticles3D` (`kQuadratic`) and
`ReseedLiquidCoreCell3D` are written out as data, as contact did (`contact/testing/contact_main_comparison_cases.h`),
rotated into Z-up (section 2), children matched by position; agree within `1e-12` relative.

GPU: every case above within 6.6; `GroupParticlesByBlock` and `AdjustParticleCount` run twice on the same input
must give identical bytes; `ComputeAmounts` on CPU and GPU give identical bytes (same tree order); the batch
round trip with a loaded try on the GPU; a 1 000 000-particle grouping repeat-run. Runs in the container without a GPU
(label `no_gpu`): `BuildParticles` with `RunOn::kGpu` returns `kRunOnCallFailed`; all shared-header maths through
the CPU tests.

## 8. Method paper section

Section "Particles" `<sec-particles>` with subsections: what a particle carries `<sec-particle-fields>`
(table `<tab:particle-fields>`, memory table `<tab:particle-memory>`); identity and order `<sec-grouping>`; try
and commit `<sec-particle-try>`; split `<sec-split>` with `<eq:split-offset>`; merge `<sec-merge>` with
`<eq:merge-fit>` and `<eq:merge-gap>`; keeping the count right `<sec-adjust-count>`; adding and removing `<sec-add-remove>`; snapshots and loaded groups `<sec-particle-copies>`.

Figures: `<fig-particle-fields>` one row per material, a box per array, extra arrays shaded, so "liquid pays
nothing for snow's arrays" is visible. `<fig-grouping>` the same six particles before and after grouping across
two cell blocks of different sizes: indexes change, ids do not. `<fig-particle-try>` two buffers per array, a try
writing two of them, the flip at commit. `<fig-split>` a parent in a big cell with its eight children at
`sqrt(3)/4 H`, the naive `H/4` corners in grey, and the three spreads drawn as circles; a second panel, a parent that drifted into smaller cells is split where it stands.
`<fig-merge>` particles in one big
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
| `side_tag`, `ElasticBladeSide3D` | `plate_side` of type `contact::PlateSide` | DECIDED in `shared-interface.md` (b) |
| `constituent`, `substance_id`, `volume_fraction`, composition | `ingredient`, position in the material's ingredient list, `ingredient_fractions` | VERIFIED by reading `liquid_mixture_3d.h:21-37`, `liquid_state_3d.h:117-123`: a constituent is one substance in a mixture, which is the decided meaning of `ingredient` |
| `phase_id` ("shared by miscible substances") | the material (`material_index`) | VERIFIED: not an ingredient; a phase is a set of ingredients that mix, which this design makes one material |
| `component` (in particle code) | none | VERIFIED: `blob` does not belong here. On `main` `component` in the blob sense appears only in air-pocket and surface records (`liquid_state_3d.h:202-214`, `surface_component_lineage_3d.h`); no particle carries one. `Particles` has no `blob_id` |
| `tag` (packed substance, phase, flags) | removed | VERIFIED unused by the step |
| `ParticleBinIndex3D`, sort bin, `LiquidParticleSortEntry3D` | `particle_group`, `GroupParticlesByBlock`, `group_begin` | DECIDED / NEW |
| `Emit`, `ReseedLiquidCoreCell3D` | `AddParticles` with `AddReason` | DECIDED |
| `SplitUniformLiquidParticle3D`, `MergeUniformLiquidParticles3D` | `ComputeSplitChild`, `ComputeMergedParticle`, run inside `AdjustParticleCount` (no public single-particle function) | DECIDED / NEW (A129) |
| retire, erase in the inlet strip | `RemoveParticles` with `RemoveReason` | DECIDED / NEW enum |
| `RemapGasApicMarkers3D` dropping a marker | `RemoveReason::kMarkerNotNeeded` | NEW (A10, gas) |
| resampling | `AdjustParticleCount` | DECIDED |
| bulk host round trip of a range | `WriteParticleGroups`, `ReadParticleGroups`, loaded try | NEW (cell resize and batches) |
| `covariance_gap_lower_bound` | `gap`, `kMergeGapFloor` | NEW |
| (none on `main`) | `merge_wait_passes` (field), `kMaxMergeWaitPasses`, `Field::kMergeWaitPasses`; `AdjustParticleCountResult::{mismatched_particle_count, waiting_particle_count, lone_particle_count, kinetic_energy_change}`; mismatched particle (its `size_level` differs from its cell's level), too large, too small, lone particle | NEW (A67) |
| D (support second moment) | `weighted_spread` via `ComputeCompleteWeightedSpread` | Verify, owned by grid |
| capacity | `max_particle_count`, `max_added_per_try` | DECIDED pattern `max_...` |
| inventory (of a particle set) | `ParticleAmounts` | NEW; cell resize uses `GridOnlyAmounts` on the cell side (not in the first version, A64) |
| `LiquidStateCapacities3D` growth | `ApplyParticleLimits` | NEW |
| scratch (working memory of the grouping and the sums; the plans' `OrderScratch`, `SumScratch`) | working buffers: `order_buffers` (`OrderBuffers`), `sum_buffers` (`SumBuffers`) | DECIDED (A31) |
| checkpoint read-back, `Readback` | `ReadSnapshot`, `ParticlesSnapshot` | DECIDED words |
| other NEW names | `ParticleFields`, `ParticleFieldsToChange`, `ArrayToChange`, `Field`, `FieldList`, `ExtraFields`, `CountRule`, `MaterialParticlesSettings`, `ParticlesSettings`, `NewParticles`, `OneParticle`, `MergedParticle`, `RefillCell`, `GroupingSettings`, `size_level` (per particle), `CheckGrouping`, `CheckParticles`, `ComputeAmounts`, `ComputeAmountsPerBlock`, `SnapshotFields`, `kOutputSnapshotFields`, `ToFieldList`, `HasField`, `ToName`, `BuildParticlesMemoryEstimate` | NEW |

## 10. Not ported

- `LiquidRecords3D`, `LiquidFrameTransaction3D`, the frame-start copy and frame scheduling
  (`liquid_state_3d.h`, `.cc`): the step module owns frames; this module has one try.
- Blocks, cells, faces, sources, clock and air-pocket records inside `LiquidState3D`: other modules.
- Exact-division, "represents increment" and "unrepresentable" checks; interval arithmetic; explicit `fma`
  (`liquid_resampling_3d.cc:91-119, 157-205, 383-396`): finite and positive checks stand instead (naming review 7.4).
- The cubic-normal split and merge variant and its energy-deficit report (`:123-139, 207-209`): the rebuild has one
  weight curve for split and merge; a second arrives only with a grid design that asks for it.
- The completeness check of a split or merge (`liquid_resampling_3d.h:23-28` and the both-grid check of
  `liquid_adaptive_family_candidate_3d.cc`), with its callback and `void* context`: nothing is stopped by it (A67), so it
  is not computed (A130).
- `tag`, `phase_id`, per-particle ingredient ids, `quadratic_adaptive_transfer` (`liquid_state_3d.h:107-123`).
- Sort bins shifted by 1.5 cells and the host round trip of sort keys (`sparse_mac_grid_3d.cc:26`,
  `liquid_arena_3d_host.cc:945-1025`).
- The 32-lane bin layout, the pinned mirror of every range and the lease type (`cuda_liquid_arena_3d.h`).
- `adaptive_liquid_particle_exchange_3d.h` (one particle absorbs a cell's correction) and
  `liquid_band_handoff_3d.h` except the refill placement: the cell resize design decides region transfer.
- Splitting or merging snow or springy solids (numerics report 7.3). Sand is the sand module's.
- Splitting or merging outside a resize pass: every split and merge runs in `AdjustParticleCount` inside
  `ResizeCells`.
- `reference/.../resampling.py`, `state.py`; `multiphase.py` mixture density goes to whichever module needs a
  mixture's density (transfer or pressure), not here. `proofs/Fluidity/Resampling.lean`.
- Asynchronous snapshots and output queues: I/O, if it wants them.

## 11. Decisions

Each decision is taken. Sources: the owner's answers (`owner-answers.md`, "A" numbers), `plan-decisions.md`, or the
recommended option under the owner's standing instruction of 2026-11-14.

- **11.1** `PerParticleVelocity<Vec3>` has the fields `for_next_step` and `for_this_step` (A1).
- **11.2** `AdjustParticleCount` follows the rule of 5.6 and runs inside the cell resize transaction on the resized
  grid, the one the try is grouped for. Recommended option.
- **11.3** Only liquid and gas markers are split and merged; sand is the sand module's (A9). Recommended option.
- **11.4** Mixtures store `ingredient_fractions`; liquids that do not mix are separate materials. Recommended option.
- **11.5** The names `rest_volume`, `volume_ratio`, `springy_deformation`, `permanent_volume_change` (A4).
- **11.6** `ParticlesTry`, `BuildTry`, `CommitTry`, `RejectTry`. Recommended option.
- **11.7** Grouping is redone only when `CheckGrouping` fails; the step may pass `max_move_since_grouped = 0`. Once a
  particle crosses a cell between groupings the sum order differs from regrouping every try; a run is still
  reproducible and a saved state restores the grouping exactly (5.9). Recommended option.
- **11.8** The names `kHasOwnMotion`, `size_level`, `CountRule`, `ParticleAmounts` as written. Recommended option.
- **11.9** Plan decisions beyond the design (`plan-decisions.md` "Particles"): `ParticlesSettings::max_cell_blocks`
  sizes the per-block tables (zero is `kBadInput`); `ApplyParticleLimits` raises the particle and block limits in one
  call; the views carry `RunOn run_on`; `ComputeAmounts` returns `std::expected` and never signals with NaN; the
  entries come from `sizeof` (6.4). Source: plan decision, 2026-11-14.
- **11.10** `AddParticlesResult::copy_count` and `bytes_copied` count the copies of `AddParticles`; the step adds them
  to `StepReport::copies`; in a batched run the whole particles stay in CPU memory (6.4). Source: A17.
- **11.11** No word "scratch": working buffers. Every Foundation call takes the storage's `order_buffers` or
  `sum_buffers`. `ComputeAmounts` keeps one form (one call's GPU memory, `kOutOfMemory`); a caller that must not
  allocate uses `ComputeAmountsPerBlock`. Source: A31.
- **11.12** Buffers on different processors are a broken precondition, asserted in debug builds, not an error value
  (`BufferError::kRunOnMismatch` does not exist). Source: A30.
- **11.13** The first version has no grid-only region: every liquid cell holds particles of its level and
  `AdjustParticleCount` covers deep cells, one level per resize (test `ThreeLevelCascade_KeepsAmounts`).
  The region transfer calls of section 4 belong to the later group "Deep liquid without particles". `AddReason::kRefillRegion` stays for gas markers. Source: A64.
- **11.14** One reserved block per run: `BuildParticles`, `BuildParticlesFromSnapshot` and `ApplyParticleLimits` take
  `RunMemory& run_memory` last; each memory entry stands for whole buffers (`copies` = their number); an `OrderBuffers`
  or `SumBuffers` is one entry; no `kRunOn` entry is `kOneCall`. Test `BuildParticles,
  IntoRunMemory_HandsOutExactlyTheEstimate`. Source: A58.
- **11.15** `ParticlesError` has 13 values; `kInternalFault` then `kOutOfMemory` come last. `kOutOfMemory` is returned
  by the building functions and by `ComputeAmounts`' allocating form (reconcile item 33).
  `BufferError::kRunMemoryTooSmall` maps to `kOutOfMemory` with `magnitude` = bytes asked; `BufferError::kSizeMismatch`
  maps to `kInternalFault` (A106). Source: `plan-decisions.md` item 11, A106.
- **11.16** Particle size follows the cell (5.6); the seeding and refill count is `ComputeRefillParticle` with `n = 8`,
  pinned by `ComputeRefillParticle, EightParticles_AtCellEighthCentres`. Source: A65.
- **11.17** A run starts on the adaptive grid: I/O seeds the start particles on the start cells of
  `step::BuildStartCells` at their own level (section 4); sources keep `(w_min/2)³`. Source: A66.
- **11.18** `AdjustParticleCount` is extended as 5.6 states: `merge_wait_passes` (`std::uint8_t`, saturating at 255, in
  a full snapshot, entry `"particles: merge wait passes"`), `Field::kMergeWaitPasses = 1024`, four result fields beside the counts of 5.6.
  Taken here beyond `plan-decisions.md` (coordinator to confirm): `kinetic_energy_lost` stays beside
  `kinetic_energy_change`; a particle with `kCarriesFittedChange` off is a candidate with `C` read as zero; intermediate generations of a repeated split are not written; every too-small particle not merged
  increments its count; `BuildFieldsToChange` and `ReadParticleGroups` refuse `kMergeWaitPasses`. Source: A67.
- **11.19** A stage that changes an array a second time in a try gets `read == write`; `contact::MoveParticles` accepts an
  output exactly equal to its own input (5.3). Source: A128.
- **11.20** No public `SplitParticle` or `MergeParticles`: every split and merge runs inside `AdjustParticleCount`; the
  formulas of 3.6 stay public. Source: A129.
- **11.21** No neighbourhood check and no counter: `AdjustParticleCount` takes one grid and never stops a split for the
  faces near it. Source: A130.
- **11.22** `WriteParticleGroups` reads the piece map of a GPU piece back once per load and returns the counted copy in
  `WriteParticleGroupsResult`. Source: A131.
- **11.23** Coordinator rulings, 2026-11-27: `grid::SizeLevel` for size levels; the range rule of A116 in every function
  taking a range, so `BuildFieldsToRead` returns `std::expected`; five error values that only the single forms returned
  are gone.
