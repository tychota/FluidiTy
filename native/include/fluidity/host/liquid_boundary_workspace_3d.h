#pragma once

#include <array>

#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_boundary_transfer_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"

namespace fluidity {
namespace boundary_internal {
struct GeometryBackend;
struct BoundedGeometryAccess;
struct BoundedTransferAccess;
}  // namespace boundary_internal
// Same units, sample ordering and geometry contract as LiquidBoundaryGeometryInputs3D;
// borrows the prepared topology grid rather than an owning grid.
struct LiquidBoundaryGeometryViewInputs3D {
  SparseMacGridView3D mac_grid;
  std::span<const CellIndex3D> cell_indices;
  std::span<const LiquidCenterDistance3D> center_distances;
  std::span<const LiquidCornerDistance3D> corner_distances;
  LiquidObstacle3D obstacle;
  double step_duration;
  double liquid_density;
  double surface_tension;
  // Finite signed outward curvature in inverse meters, including zero.
  std::optional<double> exact_outward_interface_curvature = std::nullopt;
};
// Frozen sample values are copied into the geometry region. Grid keys remain a
// topology borrow, so successful topology preparation also invalidates this view.
// Fields use the units of FrozenLiquidBoundaryGeometry3D.
struct FrozenLiquidBoundaryGeometryView3D {
  SparseMacGridView3D mac_grid;
  std::span<const LiquidCenterDistance3D> center_distances;
  std::span<const FrozenLiquidCell3D> cells;
  std::span<const FrozenLiquidFace3D> faces;
  double step_duration = 0;
  double liquid_density = 0;
  double surface_tension = 0;
  double capillary_step_bound = 0;
};
// Typed pressure cells and links in frozen cell/face order, ready for Task 4.
struct LiquidPressureProblemView3D {
  std::span<const MacPressureCell3D> cells;
  std::span<const MacPressureFace3D> faces;
  std::size_t lower_theta_regularization_count = 0;
};
// Same transfer units and complete-support preconditions as the owning inputs.
struct LiquidBoundaryDepositViewInputs3D {
  SparseMacGridView3D mac_grid;
  std::span<const LiquidBoundaryParticle3D> particles;
  std::span<const LiquidBoundaryFaceTag3D> face_tags;
  MacDepositQuantization3D quantization;
};
using LiquidCompatibleStencils3D = std::array<CompatibleMacFaceStencil3D, 3>;
// Exact signed integer face sums and normalized compatible basis in particle order.
struct LiquidBoundaryFaceDepositsView3D {
  std::span<const MacFaceDepositQuanta3D> face_quanta;
  std::span<const LiquidCompatibleStencils3D> compatible_stencils;
};
// Full owned-slot fields: kilograms, m/s, and explicit validity category.
struct NormalizedLiquidMacFacesView3D {
  std::span<const double> face_masses;
  std::span<const double> face_velocities;
  std::span<const LiquidFaceValidity3D> validity;
};
// Complete original-basis gather; histories independently cover all owned slots
// at finite increasing times in seconds and matching prescribed-wall times.
struct LiquidBoundaryGatherViewInputs3D {
  SparseMacGridView3D mac_grid;
  std::span<const LiquidBoundaryParticle3D> particles;
  double old_time;
  double new_time;
  LiquidBoundaryVelocityHistory3D old_history;
  LiquidBoundaryVelocityHistory3D new_history;
};
// Particle-order gather values and fallback counts; units match the owning result.
struct LiquidBoundaryGatherView3D {
  std::span<const LiquidBoundaryParticleGather3D> particles;
  std::size_t old_pic_fallback_particle_count = 0;
  std::size_t new_pic_fallback_particle_count = 0;
  std::size_t low_mass_fallback_particle_count = 0;
};
enum class LiquidVelocityTimeLevel3D {
  kOld,
  kNew
};
struct LiquidBoundaryWorkspaceBytes3D {
  std::size_t geometry_bytes = 0;
  std::size_t pressure_link_bytes = 0;
  std::size_t deposit_bytes = 0;
  std::size_t normalization_bytes = 0;
  std::size_t history_bytes = 0;
  std::size_t gather_bytes = 0;
  std::size_t total_bytes = 0;
};
// Checked exact dynamic payload, excluding owner objects and allocator bookkeeping.
// Geometry reserves 27 center samples and 6 incident candidates per cell; corners
// are read only during freeze. Each published region has two independent banks.
// Overflow returns code-only ResourceExhausted without allocating.
absl::StatusOr<LiquidBoundaryWorkspaceBytes3D> QueryLiquidBoundaryWorkspaceBytes3D(LiquidStateCapacities3D capacities);

// Fixed CPU numeric storage. Create/move/destruction are outside physical attempts.
// Every operation uses code-only errors and preserves prior published results on
// failure. Each returned borrow expires on the next successful operation in the
// SAME region, owner move or destruction. Old/new history are distinct regions.
// Input grid keys must outlive returned geometry. Read-only inputs may borrow
// current published regions, but must not overlap unpublished internal storage.
class LiquidBoundaryWorkspace3D {
 public:
  LiquidBoundaryWorkspace3D(LiquidBoundaryWorkspace3D&&) noexcept = default;
  LiquidBoundaryWorkspace3D& operator=(LiquidBoundaryWorkspace3D&&) noexcept = default;
  LiquidBoundaryWorkspace3D(const LiquidBoundaryWorkspace3D&) = delete;
  LiquidBoundaryWorkspace3D& operator=(const LiquidBoundaryWorkspace3D&) = delete;
  // Reserves every array via checked nothrow allocation with strong failure.
  // ResourceExhausted on byte overflow/OOM; zero capacities are valid.
  static absl::StatusOr<LiquidBoundaryWorkspace3D> Create(LiquidStateCapacities3D capacities);
  // Clips frozen geometry with the owning API's sample/obstacle/status contract.
  // Candidate count <= cells and center sample count <= 27*cells; exact incident
  // face union <= faces. Rejections leave the last geometry region intact.
  absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> Freeze(const LiquidBoundaryGeometryViewInputs3D& inputs);
  // Builds typed links from a live frozen view and finite pressure fields, with
  // the owning API's units/statuses. Checks view metadata and the exact incident
  // union before signed access/publication. Nonnormal full-cell area/volume or
  // density reciprocal returns code-only OutOfRange.
  absl::StatusOr<LiquidPressureProblemView3D>
  BuildLinks(const FrozenLiquidBoundaryGeometryView3D& geometry, const LiquidPressureFields3D& fields);
  // Deposits compatible integer mass/pure momentum using the owning API's exact
  // contribution order and guards. Complete original support must be allocated.
  absl::StatusOr<LiquidBoundaryFaceDepositsView3D> Deposit(const LiquidBoundaryDepositViewInputs3D& inputs);
  // Decodes kg/momentum quanta and divides only above minimum_face_mass in kg.
  // Same statuses as owning normalization; prior normalization survives failure.
  absl::StatusOr<NormalizedLiquidMacFacesView3D> Normalize(
      std::span<const MacFaceDepositQuanta3D> quanta,
      MacDepositQuantization3D quantization,
      double minimum_face_mass
  );
  // Uses independent complete old/new ghosts, bounded donor QR and true gradients.
  // Same statuses as owning gather; no histories or external output are modified.
  absl::StatusOr<LiquidBoundaryGatherView3D> Gather(const LiquidBoundaryGatherViewInputs3D& inputs);
  // Evaluates one complete-basis continuation at an owned face using Gather's
  // same-side donor order/QR/wall fallback. Does not modify history or its validity.
  // Used for preprojection prediction at missing liquid faces. Returns the same
  // code-only validation/support/range errors as Gather; scratch only is changed.
  absl::StatusOr<double> ContinueFace(
      SparseMacGridView3D grid,
      const LiquidBoundaryVelocityHistory3D& history,
      std::size_t face_slot,
      std::int64_t side_tag
  );
  // Copies a validated full-slot snapshot in m/s with matching finite sample times.
  // Both snapshots own velocities, validity, tags and prescribed wall samples.
  // Force/projection outputs are copied into kNew without overwriting kOld.
  absl::StatusOr<LiquidBoundaryVelocityHistory3D>
  SaveHistory(const LiquidBoundaryVelocityHistory3D& history, LiquidVelocityTimeLevel3D time_level);
  // Exact reservations, including both candidate and published arrays.
  LiquidBoundaryWorkspaceBytes3D bytes() const;
  // Current valid/scratch extents in both banks, including a rejected candidate.
  // All used ranges are initialized before reading; unused capacity is excluded.
  std::size_t used_bytes() const;
  // Maximum simultaneous used bytes, including temporary incident duplicates.
  std::size_t high_water_bytes() const;

 private:
  friend struct boundary_internal::BoundedGeometryAccess;
  friend struct boundary_internal::BoundedTransferAccess;
  absl::StatusOr<FrozenLiquidBoundaryGeometryView3D>
  FreezeWithBackend(const LiquidBoundaryGeometryViewInputs3D& inputs, boundary_internal::GeometryBackend backend);
  absl::StatusOr<LiquidPressureProblemView3D> BuildLinksWithBackend(
      const FrozenLiquidBoundaryGeometryView3D& geometry,
      const LiquidPressureFields3D& fields,
      boundary_internal::GeometryBackend backend
  );
  friend class LiquidStepWorkspace3D;
  friend absl::StatusOr<LiquidBoundaryWorkspaceBytes3D> QueryLiquidBoundaryWorkspaceBytes3D(LiquidStateCapacities3D);
  LiquidBoundaryWorkspace3D() = default;
  struct GeometryBank {
    LiquidRecords3D<LiquidCenterDistance3D> centers;
    LiquidRecords3D<FrozenLiquidCell3D> cells;
    LiquidRecords3D<FrozenLiquidFace3D> faces;
    LiquidRecords3D<bool> needs_curvature;
  };
  struct LinkBank {
    LiquidRecords3D<MacPressureCell3D> cells;
    LiquidRecords3D<MacPressureFace3D> faces;
  };
  struct DepositBank {
    LiquidRecords3D<MacFaceDepositQuanta3D> quanta;
    LiquidRecords3D<LiquidCompatibleStencils3D> stencils;
  };
  struct NormalizationBank {
    LiquidRecords3D<double> masses;
    LiquidRecords3D<double> velocities;
    LiquidRecords3D<LiquidFaceValidity3D> validity;
  };
  struct HistoryBank {
    LiquidRecords3D<double> velocities;
    LiquidRecords3D<LiquidFaceValidity3D> validity;
    LiquidRecords3D<LiquidBoundaryFaceTag3D> tags;
    LiquidRecords3D<LiquidPrescribedWallSample3D> walls;
  };
  struct OwnedFaceAnchor {
    FaceNormalAxis axis;
    Vec3 anchor_in_cell_widths;
  };
  std::array<GeometryBank, 2> geometry_;
  std::array<LinkBank, 2> links_;
  std::array<DepositBank, 2> deposits_;
  std::array<NormalizationBank, 2> normalized_;
  std::array<std::array<HistoryBank, 2>, 2> histories_;
  std::array<LiquidRecords3D<LiquidBoundaryParticleGather3D>, 2> gathered_;
  LiquidRecords3D<OwnedFaceAnchor> anchors_;
  LiquidRecords3D<double> filled_old_;
  LiquidRecords3D<double> filled_new_;
  std::size_t geometry_bank_ = 0;
  std::size_t link_bank_ = 0;
  std::size_t deposit_bank_ = 0;
  std::size_t normalization_bank_ = 0;
  std::size_t gather_bank_ = 0;
  std::array<std::size_t, 2> history_bank_{};
  LiquidStateCapacities3D capacities_;
  LiquidBoundaryWorkspaceBytes3D bytes_;
  std::size_t high_water_bytes_ = 0;
  void RecordHighWater();
};
// Bounded equivalents of the owning APIs: same numerical ordering and status
// categories, plus ResourceExhausted for insufficient region capacity. No output
// or input is modified on error. See the owner for borrow/aliasing rules.
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D>
FreezeLiquidBoundaryGeometry3D(const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace);
absl::StatusOr<LiquidPressureProblemView3D> BuildLiquidPressureLinks3D(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields,
    LiquidBoundaryWorkspace3D& workspace
);
absl::StatusOr<LiquidBoundaryFaceDepositsView3D>
DepositCompatibleMacQuanta3D(const LiquidBoundaryDepositViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace);
absl::StatusOr<NormalizedLiquidMacFacesView3D> NormalizeLiquidMacFaces3D(
    std::span<const MacFaceDepositQuanta3D> quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass,
    LiquidBoundaryWorkspace3D& workspace
);
absl::StatusOr<LiquidBoundaryGatherView3D>
GatherLiquidBoundaryParticles3D(const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace);
// Same value calculation as ResolveLiquidWallContact3D with allocation-free status
// construction/destruction. The owning convenience function retains rich errors.
absl::StatusOr<LiquidWallContactResult3D> ResolveLiquidWallContactBounded3D(const LiquidWallContactInputs3D& inputs);
}  // namespace fluidity
