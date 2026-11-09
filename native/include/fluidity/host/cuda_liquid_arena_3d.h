#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "fluidity/host/liquid_step_workspace_3d.h"

namespace fluidity {
namespace cuda_internal {
struct LiquidArenaStorage3D;
}
// FP64 qualification ABI: each field array begins on a 128-byte boundary. The
// 32 lanes preserve every checkpoint field; flags bits 0/1 mean affine/PIC.
// Unused lanes and alignment padding are zero. Compact FP32 qualification is open.
struct alignas(128) CudaLiquidParticleBin3D {
  alignas(128) double world_positions[3][32];
  alignas(128) double world_velocities[3][32];
  alignas(128) double particle_velocity_slopes[9][32];
  alignas(128) double particle_masses[32];
  alignas(128) double reference_volumes[32];
  alignas(128) double volume_ratios[32];
  alignas(128) std::uint64_t particle_ids[32];
  alignas(128) std::int64_t side_tags[32];
  alignas(128) std::uint32_t particle_tags[32];
  alignas(128) std::uint32_t affine_pic_flags[32];
};
// Every nonempty range has 128-byte alignment and disjoint device/pinned storage. Numeric
// ranges are mutable scratch, never part of an immutable epoch. Counts describe
// reservation limits; consumers must validate active extents before access.
enum class CudaLiquidRange3D : std::size_t {
  // LiquidSortKey3D / size_t arrays, LiquidParticleSortEntry3D import and CUB
  // temporary bytes. Sort capacity is max(particle capacity,1).
  kSortInputKeys,
  kSortOutputKeys,
  kSortInputOrdinals,
  kSortOutputOrdinals,
  kSortEntries,
  kSortTemporary,
  kArchitecture,
  // One boundary_internal::GeometryInput / ContactInput / BoundaryReport3D.
  kGeometryInput,
  kContactInput,
  kBoundaryReport,
  // Matching BoundaryTransfer*3D POD types from the existing CUDA bridge;
  // face/particle arrays use full configured capacities, query/report count one.
  kTransferQuery,
  kTransferFaces,
  kTransferParticles,
  kTransferFaceResults,
  kTransferParticleResults,
  kTransferReport,
  // Pressure*3D bridge POD arrays; levels <=33, blocks <=33*cell capacity.
  // Indices reserve n+L*(3*n+1+4*f), coefficients L*(34*n+f), fine 6*n
  // floats, coarse 3*L*n doubles, rows 2*n, output 4*n+f doubles; state count one.
  kPressureCells,
  kPressureLinks,
  kPressureLevels,
  kPressureBlocks,
  kPressureIndices,
  kPressureCoefficients,
  kPressureFineScratch,
  kPressureCoarseScratch,
  kPressureRows,
  kPressureOutput,
  kPressureState,
  // Vec3/double arrays of 8*max(particles,cells,faces) elements and full-capacity
  // LiquidParticleState3D source staging, respectively. Field descriptors remain
  // adapter-owned POD inputs in 6C; no numeric host fallback is supplied here.
  kSamplePositions,
  kSampleDistances,
  kSampleParticles,
  kPressurePreparationScratch,  // Host-only checked assembly/hierarchy; device pointer is null.
  kCount,
};
struct CudaLiquidRangeView3D {
  std::byte* device = nullptr;
  std::span<std::byte> pinned;
};
// Requested CUDA payload bytes include all alignment/unused capacity and every
// current, prospective and retired owner, including owners retained by leases.
// CUDA context/module/driver bookkeeping is not an application byte reservation.
// Reservation/creation counters count API attempts; frees/destructions count calls
// for acquired handles, including partial-construction cleanup.
struct CudaLiquidArenaMetrics3D {
  std::size_t total_device_bytes = 0;
  std::size_t free_device_bytes_at_start = 0;
  std::size_t device_byte_limit = 0;
  std::size_t live_device_bytes = 0;
  std::size_t peak_device_bytes = 0;
  std::size_t live_pinned_bytes = 0;
  std::size_t peak_pinned_bytes = 0;
  std::uint64_t device_reservations = 0;
  std::uint64_t device_frees = 0;
  std::uint64_t pinned_reservations = 0;
  std::uint64_t pinned_frees = 0;
  std::uint64_t stream_creations = 0;
  std::uint64_t stream_destructions = 0;
  std::uint64_t event_creations = 0;
  std::uint64_t event_destructions = 0;
  std::uint64_t graph_creations = 0;
  std::uint64_t graph_destructions = 0;
  std::uint64_t executable_creations = 0;
  std::uint64_t executable_destructions = 0;
  std::uint64_t sort_replays = 0;
  int compiled_architecture = 0;
  int cccl_version = 0;
  int cub_version = 0;
};
// Physical epochs may repeat after frame rollback. Generation never repeats in
// this job, including discarded preparations; compare the entire identity.
struct CudaLiquidEpochIdentity3D {
  std::uint64_t generation = 0;
  LiquidClock3D clock;
  bool operator==(const CudaLiquidEpochIdentity3D&) const = default;
};
struct CudaLiquidStateLayout3D {
  std::size_t particle_bins_byte_offset = 0;
  std::size_t block_records_byte_offset = 0;
  std::size_t cell_records_byte_offset = 0;
  std::size_t face_records_byte_offset = 0;
  std::size_t source_records_byte_offset = 0;
  std::size_t control_record_byte_offset = 0;
  std::size_t total_state_bytes = 0;
  std::size_t particle_bin_stride_bytes = sizeof(CudaLiquidParticleBin3D);
};
// Owns a read-only bank through arena growth/destruction. Moving invalidates the
// old lease. No async completion is inferred: Task 8 must retain this lease until
// its own D2H event completes. Acquire/move do not allocate. Destroy outside a
// physical attempt; the final owner may release CUDA resources.
class CudaLiquidEpochLease3D {
 public:
  CudaLiquidEpochLease3D() = default;
  CudaLiquidEpochLease3D(CudaLiquidEpochLease3D&&) noexcept;
  CudaLiquidEpochLease3D& operator=(CudaLiquidEpochLease3D&&) noexcept;
  ~CudaLiquidEpochLease3D();
  CudaLiquidEpochLease3D(const CudaLiquidEpochLease3D&) = delete;
  CudaLiquidEpochLease3D& operator=(const CudaLiquidEpochLease3D&) = delete;
  // Captured identity, unchanged for the entire lease lifetime.
  CudaLiquidEpochIdentity3D identity() const;
  // Immutable full padded payload; valid through lease destruction. This is an
  // internal execution ABI, not a portable wire encoding. No scratch is included.
  std::span<const std::byte> device_bytes() const;
  // Execution-ABI byte offsets for this lease, including padding/unused capacity.
  CudaLiquidStateLayout3D layout() const;
  // Job-wide current counters, including retained owners; no device query.
  CudaLiquidArenaMetrics3D metrics() const;
  // Synchronous checked D2H/decode into caller-owned sufficient host capacities.
  // Output is unchanged on error. No allocation. Concurrent calls are forbidden.
  absl::Status Readback(LiquidState3D& output) const;

 private:
  friend class CudaLiquidArena3D;
  cuda_internal::LiquidArenaStorage3D* storage_ = nullptr;
  std::size_t bank_ = 0;
  CudaLiquidEpochIdentity3D identity_;
};
// Single-threaded job owner. Create/growth/CollectRetired/destruction run outside
// physical attempts. Sort, prepare/commit/discard publication use only reserved
// storage and code-only errors. Stream and pinned pointers are borrowed until
// growth commit; never retain scratch across growth. No default stream is used.
class CudaLiquidArena3D {
 public:
  CudaLiquidArena3D(CudaLiquidArena3D&&) noexcept;
  CudaLiquidArena3D& operator=(CudaLiquidArena3D&&) noexcept;
  ~CudaLiquidArena3D();
  CudaLiquidArena3D(const CudaLiquidArena3D&) = delete;
  CudaLiquidArena3D& operator=(const CudaLiquidArena3D&) = delete;
  // Queries current device memory once per job and fixes min(70% total,80% free)
  // ceiling, optionally lowered by byte_limit. Accounts old+new and leased banks.
  // Requires CC12.0 and compiled 1200. Overflow/OOM: ResourceExhausted; invalid
  // device: InvalidArgument; unavailable/wrong hardware: FailedPrecondition.
  static absl::StatusOr<CudaLiquidArena3D> Create(
      LiquidStateCapacities3D capacities,
      int device_ordinal = 0,
      std::size_t byte_limit = static_cast<std::size_t>(-1)
  );
  // Copies one valid initial state into the committed bank once. Validation and
  // capacity errors preserve the uninitialized owner; no source borrow survives.
  absl::Status Initialize(const LiquidState3D& initial);
  // Stages all physical fields and prospective accepted clock, fences uploads,
  // and reserves a new generation before success. Reject leaves committed bytes
  // intact. An old committed bank with a live lease cannot become trial storage.
  absl::Status PreparePublication(const LiquidState3D& trial, LiquidClock3D accepted_clock);
  // Requires successful prepare. Pure metadata operations: no CUDA call, no free.
  void CommitPublication() noexcept;
  void DiscardPublication() noexcept;
  // Strict nondecreasing growth; stages exact scheduler frame_start. Commit
  // retains the previous owner; collection occurs explicitly outside attempts.
  absl::Status PrepareGrowth(LiquidStateCapacities3D capacities, const LiquidState3D& frame_start);
  void CommitGrowth() noexcept;
  // Outside attempts only: discards prospective resources or collects retired
  // owners whose leases have all ended. Active/current storage is never freed.
  void DiscardGrowth() noexcept;
  void CollectRetired() noexcept;
  // Returns a nonallocating source lease after Initialize. FailedPrecondition
  // before initialization; OutOfRange if an ownership reference counter is full.
  absl::StatusOr<CudaLiquidEpochLease3D> AcquireCommitted();
  // Checked bins are staged in incoming order with INT32_MAX neutral padding.
  // The fixed-capacity CCCL 3.4.2 graph is authoritative; only its exact checked
  // permutation enters PrepareTrialTopologyFromOrder. Empty input still replays.
  // Any failure preserves workspace publication and committed device state.
  absl::Status PrepareTopology(
      std::span<const LiquidParticleState3D> particles,
      std::span<const CellIndex3D> pressure_cells,
      LiquidStepWorkspace3D& workspace,
      const LiquidState3D* previous_state = nullptr
  );
  // Named scratch range; invalid enum returns an empty view. No writes to an
  // immutable state bank are exposed here. Borrows end on growth commit/move.
  CudaLiquidRangeView3D range(CudaLiquidRange3D name) const;
  // Configured record extents, including unused capacity.
  LiquidStateCapacities3D capacities() const;
  // Job-wide current counters, including retained owners; no device query.
  CudaLiquidArenaMetrics3D metrics() const;
  // Explicit nonblocking CUDA stream and disable-timing completion event handles.
  // The caller must keep the selected device current and fence its own work before
  // reusing pinned staging; event contents change after every arena fence.
  void* stream() const;
  void* completion_event() const;
  // Fences only this arena's stream; propagates sticky CUDA errors as Internal.
  absl::Status Synchronize() const;

 private:
  struct Impl;
  CudaLiquidArena3D();
  std::unique_ptr<Impl> impl_;
};
}  // namespace fluidity
