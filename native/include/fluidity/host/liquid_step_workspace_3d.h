#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "fluidity/host/sparse_mac_grid_3d.h"

namespace fluidity {

struct LiquidParticleSortEntry3D {
  ParticleBinIndex3D bin_index{};
  std::size_t original_ordinal = 0;  // Stable incoming order for equal signed bins.
};

// Exact dynamic payload reservations for topology and CPU boundary/transfer. Each published/candidate
// region has two independent banks. Includes bounded activation/sort storage; excludes
// owner objects, allocator bookkeeping, transaction versions and future GPU regions.
// Task 6 must extend accounting before claiming a complete physical-step bound.
struct LiquidWorkspaceBytes3D {
  std::size_t particle_order_bytes = 0;
  std::size_t particle_state_bytes = 0;
  std::size_t block_key_bytes = 0;
  std::size_t block_state_bytes = 0;
  std::size_t cell_state_bytes = 0;
  std::size_t face_state_bytes = 0;
  std::size_t activation_key_bytes = 0;
  std::size_t activation_key_capacity = 0;
  std::size_t boundary_bytes = 0;  // Full QueryLiquidBoundaryWorkspaceBytes3D payload.
  std::size_t pressure_bytes = 0;  // Full QueryMacPressureWorkspaceBytes3D payload.
  std::size_t total_bytes = 0;
};
// Checked sum/products bounded by PTRDIFF_MAX; overflow returns ResourceExhausted.
// Sources reserve no topology storage. No allocation; errors have empty messages.
absl::StatusOr<LiquidWorkspaceBytes3D> QueryLiquidWorkspaceBytes3D(LiquidStateCapacities3D capacities);

struct LiquidWorkspaceUsage3D {
  LiquidStateCapacities3D active;  // Published topology extents; sources is always zero.
  std::size_t activation_keys = 0;  // Used scratch before deduplication, including duplicates.
  std::size_t used_bytes =
      0;  // Current valid ranges in both banks plus numeric/sort scratch; excludes unused capacity.
  std::size_t reserved_bytes = 0;  // Same payload scope as QueryLiquidWorkspaceBytes3D.
};
struct LiquidPreparedTopologyView3D {
  SparseMacGridView3D grid;
  std::span<const LiquidParticleSortEntry3D> particle_order;
  std::span<const LiquidParticleState3D> particles;
  std::span<const LiquidBlockState3D> blocks;
  std::span<const LiquidCellState3D> cells;
  std::span<const LiquidFaceState3D> faces;
};

// Single-threaded bounded topology and CPU boundary/transfer/pressure owner. Move invalidates all borrows. No operation
// other than Create/move/destruction allocates or frees, including error destruction.
// Errors are code-only; format diagnostics outside the physical attempt.
class LiquidStepWorkspace3D {
 public:
  LiquidStepWorkspace3D(LiquidStepWorkspace3D&&) noexcept = default;
  LiquidStepWorkspace3D& operator=(LiquidStepWorkspace3D&&) noexcept = default;
  LiquidStepWorkspace3D(const LiquidStepWorkspace3D&) = delete;
  LiquidStepWorkspace3D& operator=(const LiquidStepWorkspace3D&) = delete;
  // Reserves two topology banks, 81*particles+7*cells activation keys, and fixed CPU numeric regions.
  // Invalid geometry returns InvalidArgument. Checked byte overflow or actual nothrow
  // allocation failure returns ResourceExhausted with no partial owner.
  static absl::StatusOr<LiquidStepWorkspace3D>
  Create(LiquidStateCapacities3D capacities, UniformMacGeometry3D geometry);
  // Sorts particles stably by signed (z,y,x) bin and incoming ordinal. Selects unique
  // pressure cells in (z,y,x) order, all 81 original particle faces regardless of
  // weight/tag, and every pressure cell's owner plus all six incident face owners.
  // Publishes every owned face in slot order, with no retained unused uniform blocks.
  // Both input counts must fit their capacities before deduplication.
  // Positions are world meters at the sort time. Other particle fields are copied
  // verbatim; their physical validation remains the transaction acceptance gate.
  // Optional previous_state must be valid and use the same geometry. Retained block
  // generations/activation ages are carried, inactive ages reset; new blocks use
  // previous epoch+1 (or 1 without history), age zero. Pressure/occupancy and carrier
  // history are remapped by cell/face identity, never by old ordinal. New cells are
  // liquid with zero pressure; new faces have missing validity and zero velocity.
  // ResourceExhausted for capacities, InvalidArgument for invalid geometry/history,
  // OutOfRange for lattice/epoch overflow. Previous state/input are never modified;
  // failure leaves the published view intact. Inputs may borrow the current view
  // but must not overlap unpublished workspace storage. No scratch contents escape.
  absl::Status PrepareTrialTopology(
      std::span<const LiquidParticleState3D> particles,
      std::span<const CellIndex3D> pressure_cells,
      const LiquidState3D* previous_state = nullptr
  );
  // Imports a backend's authoritative stable signed-bin order without re-sorting.
  // Every original ordinal occurs exactly once, matches its checked bin, and is
  // strictly ordered by (z,y,x,original ordinal). Invalid order returns
  // InvalidArgument and preserves the published topology. Other preparation,
  // support, history, capacity, lifetime and aliasing rules match the CPU method.
  absl::Status PrepareTrialTopologyFromOrder(
      std::span<const LiquidParticleState3D> particles,
      std::span<const CellIndex3D> pressure_cells,
      std::span<const LiquidParticleSortEntry3D> particle_order,
      const LiquidState3D* previous_state = nullptr
  );
  // View is valid until next successful PrepareTrialTopology, move or destruction.
  // A rejected preparation preserves both values and addresses. FailedPrecondition
  // before the first successful preparation. Task 3/4 numeric APIs borrow grid here.
  absl::StatusOr<LiquidPreparedTopologyView3D> prepared() const;
  // Copies sorted particles and remapped topology into existing trial reservations.
  // Geometry must match (InvalidArgument); insufficient capacity is ResourceExhausted;
  // no prepared view is FailedPrecondition. Preserves output on every failure.
  // Particle IDs are copied with the sorted records. The caller retains ownership
  // of next_particle_id, the clock, source inventory and seed. Trial owners may not
  // alias workspace storage. The acceptance gate validates the complete trial.
  absl::Status ApplyTrialTopology(LiquidState3D& trial) const;
  // Active counts describe the published topology bank; valid-range byte/scratch counts also
  // include the candidate bank, including work left by a rejected preparation.
  LiquidWorkspaceUsage3D usage() const;
  // Fixed reservation limits supplied to Create (sources are not used by topology).
  LiquidStateCapacities3D capacities() const;
  // Numeric regions share these reservation limits; their borrows have independent lifetimes.
  MacPressureWorkspace3D& pressure() {
    return pressure_;
  }
  LiquidBoundaryWorkspace3D& boundary() {
    return boundary_;
  }

 private:
  struct TopologyBank {
    LiquidRecords3D<LiquidParticleSortEntry3D> particle_order;
    LiquidRecords3D<LiquidParticleState3D> particles;
    LiquidRecords3D<BlockIndex3D> block_keys;
    LiquidRecords3D<LiquidBlockState3D> blocks;
    LiquidRecords3D<LiquidCellState3D> cells;
    LiquidRecords3D<LiquidFaceState3D> faces;
  };
  absl::Status PrepareTopology(
      std::span<const LiquidParticleState3D> particles,
      std::span<const CellIndex3D> pressure_cells,
      const LiquidState3D* previous_state,
      std::span<const LiquidParticleSortEntry3D> imported_order,
      bool import_order
  );
  LiquidStepWorkspace3D() = default;
  LiquidBoundaryWorkspace3D boundary_;
  MacPressureWorkspace3D pressure_;
  UniformMacGeometry3D geometry_{{0.0, 0.0, 0.0}, 1.0};
  LiquidStateCapacities3D capacities_;
  LiquidWorkspaceBytes3D bytes_;
  std::array<TopologyBank, 2> banks_;
  LiquidRecords3D<BlockIndex3D> activation_keys_;
  std::size_t published_bank_ = 0;
  bool prepared_ = false;
};
}  // namespace fluidity
