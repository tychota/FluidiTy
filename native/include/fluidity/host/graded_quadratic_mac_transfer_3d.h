#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/graded_mac_transfer_3d.h"
#include "fluidity/host/sparse_mac_grid_3d.h"

namespace fluidity {

struct GradedQuadraticMacParticle3D {
  MacTransferParticle3D transfer;
  std::int64_t side_tag;  // Required on every supporting carrier.
};

struct GradedQuadraticMacStencil3D {
  double cell_width_m;
  std::array<MacFaceStencil3D, 3> rows;  // x, y, z velocity rows in uniform tensor order.
};

// A checked FP64 quadratic MAC operator on a frozen adaptive layout. It
// uses the layout's canonical slots and admits only complete same-level
// islands. Callers must provide the physical apertures from their snapshot.
class GradedQuadraticMacTransfer3D {
 public:
  // Own the topology and derive its canonical physical anchors and slots.
  // Propagates topology, extent and world geometry errors.
  static absl::StatusOr<GradedQuadraticMacTransfer3D> Create(AdaptiveMacTopology3D topology, Vec3 world_origin);

  // Bind every tensor sample, including zero-weight ties. Both positions
  // must lie in leaves at the same level. Every sample must have two
  // same-level incident leaves, a representable physical displacement,
  // full open area and a finite, valid, same-side, eligible carrier.
  // Apertures in m² and carriers follow the complete canonical face order.
  // InvalidArgument denotes malformed inputs. FailedPrecondition denotes
  // missing or incompatible support, and OutOfRange denotes arithmetic or
  // geometry that cannot be represented. Inputs are read-only.
  absl::StatusOr<GradedQuadraticMacStencil3D> Bind(
      const GradedQuadraticMacParticle3D& particle,
      std::span<const double> open_areas_m2,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  ) const;

  // Deposit kg and positive-normal momentum in kg*m/s in particle then
  // tensor order using the bound rows. Returns one record per canonical
  // slot and no records on failure. Unsupported slots remain zero.
  absl::StatusOr<std::vector<MacFaceDeposit3D>> Deposit(
      std::span<const GradedQuadraticMacParticle3D> particles,
      std::span<const double> open_areas_m2,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  ) const;

  // Gather the new PIC velocity, old-to-new velocity increment, affine
  // slopes and velocity derivatives using the same bound rows. The old
  // and new carriers must both satisfy Bind. No result on failure.
  absl::StatusOr<MacParticleGatherResult3D> Gather(
      const GradedQuadraticMacParticle3D& particle,
      std::span<const double> open_areas_m2,
      std::span<const LiquidAdaptiveFaceState3D> old_carrier_faces,
      std::span<const LiquidAdaptiveFaceState3D> new_carrier_faces
  ) const;

 private:
  friend class liquid_step_internal::AdaptiveLiquidStep3D;

  // Borrow the runtime's canonical layout and apertures to bind its rows.
  // The aperture accessor covers all canonical face slots.
  static absl::StatusOr<GradedQuadraticMacStencil3D> BindWithLayout(
      const GradedQuadraticMacParticle3D& particle,
      const AdaptiveMacTopology3D& topology,
      const GradedMacTransfer3D& transfer,
      Vec3 world_origin,
      std::span<const std::size_t> interior_slots,
      absl::FunctionRef<double(std::size_t)> open_area_m2,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  );
  // The caller owns the prevalidated slot scratch and discards it on failure.
  static absl::Status AccumulateBoundDeposit(
      const MacTransferParticle3D& particle,
      const GradedQuadraticMacStencil3D& bound,
      std::span<MacFaceDeposit3D> deposits
  );
  // Check all slots in the frozen rows, including zero-weight ties.
  static absl::Status CheckBoundCarriers(
      const GradedQuadraticMacStencil3D& bound,
      std::int64_t side_tag,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  );
  // Read both histories on the same frozen rows. Set old_pic_velocity only
  // on success. The caller checks the old/new extents cover these rows.
  static absl::StatusOr<MacParticleGatherResult3D> GatherBound(
      const GradedQuadraticMacStencil3D& bound,
      std::int64_t side_tag,
      std::span<const LiquidAdaptiveFaceState3D> old_carrier_faces,
      std::span<const LiquidAdaptiveFaceState3D> new_carrier_faces,
      Vec3& old_pic_velocity
  );

  GradedQuadraticMacTransfer3D(
      AdaptiveMacTopology3D topology,
      AdaptiveMacFaceOwnership3D ownership,
      GradedMacTransfer3D anchors,
      Vec3 world_origin
  );

  AdaptiveMacTopology3D topology_;
  AdaptiveMacFaceOwnership3D ownership_;
  GradedMacTransfer3D anchors_;
  Vec3 world_origin_;
};

// Frozen canonical inputs for one side of a topology change. Spans borrow
// the physical aperture and carrier banks in transfer slot order.
struct GradedQuadraticMacSupportSnapshot3D {
  const GradedQuadraticMacTransfer3D& transfer;
  std::span<const double> open_areas_m2;
  std::span<const LiquidAdaptiveFaceState3D> carrier_faces;
};

// Propose F3 split using the committed coarse support and candidate fine
// support. The required callback additionally certifies particle geometry,
// phase and constitutive admissibility. No state changes on any failure.
absl::StatusOr<LiquidUniformSplit3D> SplitGradedQuadraticLiquidParticle3D(
    const LiquidParticleState3D& parent,
    std::uint64_t first_child_id,
    double parent_cell_width_m,
    const GradedQuadraticMacSupportSnapshot3D& committed,
    const GradedQuadraticMacSupportSnapshot3D& candidate,
    LiquidResamplingAdmissibility3D certify,
    void* context
);

// Propose F4 merge with committed fine and candidate coarse support. The
// covariance and representability gates of the uniform operator apply.
// The required callback and failure behavior are the same as for split.
absl::StatusOr<LiquidUniformMerge3D> MergeGradedQuadraticLiquidParticles3D(
    std::span<const LiquidParticleState3D> children,
    std::uint64_t target_id,
    double fine_cell_width_m,
    const GradedQuadraticMacSupportSnapshot3D& committed,
    const GradedQuadraticMacSupportSnapshot3D& candidate,
    LiquidResamplingAdmissibility3D certify,
    void* context
);

}  // namespace fluidity
