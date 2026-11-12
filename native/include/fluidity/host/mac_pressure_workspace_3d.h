#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {
namespace pressure_internal {
struct BoundedCudaPressureAccess3D;
struct CoupledPressureWorkspaceAccess3D;
class PreparedMacPressureSystem3D;
}  // namespace pressure_internal
// Scalar observations of the original checked pressure system. These
// observations do not participate in the solve or its stopping decisions.
struct MacPressureFrozenMeasurement3D {
  double integrated_rhs_l2_m3_per_s = 0;  // L2 norm of the original integrated RHS, including gauge rows.
  double pressure_cell_volume_m3 = 0;  // Sum of the supplied fluid volumes.
  double inward_prescribed_wall_flux_m3_per_s = 0;  // Only explicitly supplied flux, oriented into selected cells.
};
// Read-only projection in input cell/face order. Units and diagnostics match
// MacPressureProjection3D, including original residual and physical defect at
// every gauge row. Borrows survive all rejected attempts; the next successful
// projection, owner move or destruction invalidates them.
struct MacPressureProjectionView3D {
  std::span<const double> cell_pressures;
  std::span<const double> corrected_face_velocities;
  std::span<const double> integrated_rhs;
  std::span<const double> true_residuals;
  std::span<const double> cell_constraint_defects;
  std::span<const std::size_t> gauge_cell_ordinals;
  MacPressureDiagnostics3D diagnostics;
  MacPressureFrozenMeasurement3D frozen_measurement;
};
// Exact reserved dynamic payload, excluding owner objects, allocator metadata
// and automatic storage (including the 32-double terminal solve array).
// Scratch includes conservative per-level upper bounds and alignment padding;
// it is not an assertion that all reserved shapes are simultaneously attainable.
struct MacPressureWorkspaceBytes3D {
  std::size_t scratch_bytes = 0;
  std::size_t projection_bytes = 0;
  std::size_t total_bytes = 0;
  std::size_t maximum_levels = 0;  // 33: 31 signed halvings and one final grouping after the fine level.
  std::size_t maximum_level_cells = 0;  // Per level; no level gains unknowns.
  std::size_t maximum_level_edges = 0;  // Per level; no level gains edges.
  std::size_t maximum_terminal_factor_entries = 0;  // Sum of block squares <= 32*cell_capacity.
};
// Checked conservative reservation for any admitted shape at these capacities.
// Counts both projection banks, prepared coefficients, hierarchy, factors, PCG,
// recursive V-cycle, reductions and assembly scratch. Overflow returns inline
// ResourceExhausted. Both zero capacities reserve zero bytes. No allocation.
absl::StatusOr<MacPressureWorkspaceBytes3D>
QueryMacPressureWorkspaceBytes3D(std::size_t cell_capacity, std::size_t face_capacity);
// Single-threaded fixed host owner. Only Create/move/destruction may allocate or
// free. Every attempted projection, including rejection and result destruction,
// uses inline status codes and preallocated checked spans. Format detailed
// diagnostics outside physical attempts. Scratch addresses do not escape.
class MacPressureWorkspace3D {
 public:
  MacPressureWorkspace3D(MacPressureWorkspace3D&&) noexcept = default;
  MacPressureWorkspace3D& operator=(MacPressureWorkspace3D&&) noexcept = default;
  MacPressureWorkspace3D(const MacPressureWorkspace3D&) = delete;
  MacPressureWorkspace3D& operator=(const MacPressureWorkspace3D&) = delete;
  // Strong nothrow reservation: overflow/OOM returns ResourceExhausted without a
  // partial owner. Capacities count cells/faces, not owned full-grid slots.
  static absl::StatusOr<MacPressureWorkspace3D> Create(std::size_t cell_capacity, std::size_t face_capacity);
  // Project a checked explicit graph with the same numerical and publication
  // contract as the uniform workspace entry. The system is borrowed for the
  // call. A uniform prepared system returns InvalidArgument. Errors preserve the preceding publication.
  absl::StatusOr<MacPressureProjectionView3D> ProjectPrepared(
      const pressure_internal::PreparedMacPressureSystem3D& system,
      MacPressureSolverOptions3D options = {}
  );
  struct PhysicalComponents {
    std::span<const std::size_t> roots;  // Component root for each input cell.
    std::span<const std::size_t> sealed_roots;  // Unanchored roots in ascending order.
  };
  // Validates the uniform problem including physical source and wall-flux
  // compatibility. Inputs follow the uniform projection contract and are
  // disjoint from scratch. Results borrow scratch until its next use or owner
  // move/destruction. Leaves previously published projections intact.
  absl::StatusOr<PhysicalComponents> ClassifyPhysicalComponents(
      SparseMacGridView3D grid,
      double step_duration,
      const LiquidPressureProblemView3D& problem
  );
  // Exact reservation returned by QueryMacPressureWorkspaceBytes3D.
  MacPressureWorkspaceBytes3D bytes() const {
    return bytes_;
  }
  // Current initialized scratch ranges (including rejected candidates) plus
  // both result-bank extents, excluding capacity outside those ranges.
  std::size_t used_bytes() const;
  // Maximum simultaneous used payload, including temporary recursive scratch.
  std::size_t high_water_bytes() const {
    return high_water_bytes_;
  }

 private:
  friend class LiquidStepWorkspace3D;
  friend struct pressure_internal::BoundedCudaPressureAccess3D;
  friend struct pressure_internal::CoupledPressureWorkspaceAccess3D;
  MacPressureWorkspace3D() = default;
  struct ProjectionBank {
    LiquidRecords3D<double> pressures;
    LiquidRecords3D<double> velocities;
    LiquidRecords3D<double> rhs;
    LiquidRecords3D<double> residuals;
    LiquidRecords3D<double> defects;
    LiquidRecords3D<std::size_t> gauges;
  };
  friend absl::StatusOr<MacPressureProjectionView3D> ProjectMacPressure3D(
      SparseMacGridView3D grid,
      double step_duration,
      const LiquidPressureProblemView3D& problem,
      MacPressureSolverOptions3D options,
      MacPressureWorkspace3D& workspace
  );
  template <typename PrepareUpload>
  absl::StatusOr<MacPressureProjectionView3D> ProjectWithPreparation(
      std::size_t cell_count,
      std::size_t face_count,
      MacPressureSolverOptions3D options,
      PrepareUpload prepare_upload
  );
  LiquidRecords3D<std::max_align_t> scratch_;
  std::array<ProjectionBank, 2> projections_;
  MacPressureWorkspaceBytes3D bytes_;
  std::size_t cell_capacity_ = 0;
  std::size_t face_capacity_ = 0;
  std::size_t published_bank_ = 0;
  std::size_t scratch_used_bytes_ = 0;
  std::size_t high_water_bytes_ = 0;
};
// Project the frozen pressure cells and links over a positive finite step duration
// in seconds. Borrows the grid for this call. Cells and links stay read-only. Requires
// 1..4096 unique (z,y,x)-ordered cells and their exact incident face union, at
// most 24576 entries in owned-slot order, including sealed faces. All input
// arrays/keys must remain unchanged throughout the call and must not overlap
// unpublished workspace storage. No external carrier velocities are written.
// Uses the owning API's FP32 fine / FP64 coarse and reduction ordering, warm
// starts, physical gates and status categories. Incompatible sealed constraints
// return FailedPrecondition; nonconvergence/nonpositive curvature return Aborted.
// Insufficient capacity returns ResourceExhausted. Any failure preserves the
// previously published projection's values and addresses. Returned units and
// lifetime are documented by MacPressureProjectionView3D.
absl::StatusOr<MacPressureProjectionView3D> ProjectMacPressure3D(
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
);
}  // namespace fluidity
