#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/liquid_band_handoff_3d.h"
#include "fluidity/host/liquid_resampling_3d.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {
namespace liquid_step_internal {
class AdaptiveLiquidStep3D;
}

// Physical staggered anchor at the center of a canonical owned subface.
struct GradedMacFace3D {
  FaceNormalAxis axis;
  Vec3 world_anchor;  // m.
  double width_m;
};

struct GradedMacParticleGather3D {
  Vec3 velocity;  // Cartesian m/s evaluated at the particle.
  Mat3 velocity_slopes;  // Cartesian affine slopes, inverse seconds.
  bool affine_enabled;  // False means PIC on all three components and zero slopes.
};

// Measured on the actual normalized gather support for one velocity row.
// Offsets are dimensionless in units of the containing leaf width.
struct GradedMacSupportMomentRow3D {
  std::size_t positive_sample_count = 0;
  double unnormalized_weight_sum = 0;
  double normalized_weight_sum = 0;
  Vec3 mean_offset{};
  Mat3 centered_covariance{};  // In squared leaf-width units.
  bool full_rank = false;  // Passes the production affine pivot test.
  bool uses_low_mass = false;
};

struct GradedMacSupportMomentWitness3D {
  double leaf_width_m = 0;
  std::array<GradedMacSupportMomentRow3D, 3> rows{};  // Velocity rows x, y, z.
  bool has_masked_support = false;  // A positive geometric weight was excluded.
  bool has_width_transition = false;  // A positive geometric face has a different width.
  // Exact comparisons of the measured binary64 partition and first/second
  // moments to the uniform row targets. These do not certify basis identity.
  bool matches_quadratic_moments = false;
  bool matches_cubic_normal_moments = false;
};

class GradedMacTransfer3D {
 public:
  static constexpr double kSupportRadiusInLeafWidths = 2;

  // Freeze the physical leaf boxes and all owned subface anchors. world_origin
  // is the world position in m of the finest lattice's zero corner. Returns
  // InvalidArgument for invalid topology or origin, OutOfRange if world
  // geometry cannot be represented, and ResourceExhausted above the state cap.
  static absl::StatusOr<GradedMacTransfer3D> Create(const AdaptiveMacTopology3D& topology, Vec3 world_origin);

  // Optional solid upper X/Z tank planes close the corresponding leaf bounds.
  static absl::StatusOr<GradedMacTransfer3D>
  Create(const AdaptiveMacTopology3D& topology, Vec3 world_origin, std::optional<Vec3> solid_upper);

  // Return the canonical leaf ordinal for a finite world position in m.
  // Missing ownership returns FailedPrecondition.
  absl::StatusOr<std::size_t> ParticleLeafOrdinal(Vec3 world_position) const;

  // One geometry record per canonical global slot. The span borrows this owner.
  std::span<const GradedMacFace3D> faces() const {
    return faces_;
  }

  // Bytes reserved by the frozen leaf boxes, canonical face geometry and support bounds.
  std::size_t reserved_bytes() const {
    return leaves_.capacity() * sizeof(LeafBox) + faces_.capacity() * sizeof(GradedMacFace3D)
        + face_support_bounds_.capacity() * sizeof(FaceSupportBounds);
  }

  // Deposit mass in kg and positive-normal momentum in kg*m/s. Results are in
  // canonical global slot order. Each particle must lie in a half-open leaf
  // box. Only same-side, structurally eligible subfaces participate. Incoming
  // carrier velocity and validity are not deposition inputs. Weight sums are
  // normalized per component, and APIC uses centered offsets when all three
  // components have affine support. Particle order and global slot order fix
  // the reduction order. InvalidArgument denotes malformed extents or inputs,
  // FailedPrecondition denotes missing support, and OutOfRange denotes
  // nonrepresentable arithmetic. Failure returns no deposit owner.
  absl::StatusOr<std::vector<MacFaceDeposit3D>> Deposit(
      std::span<const LiquidParticleState3D> particles,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  ) const;

  // Gather a world-space velocity and weighted affine fit from the canonical
  // global slots. Missing, ineligible and different-side carriers are masked.
  // Low mass or rank loss on any component selects normalized PIC and clears
  // all slopes. FailedPrecondition denotes missing support on any component.
  // InvalidArgument denotes malformed extents, position or carrier data.
  // OutOfRange denotes nonrepresentable arithmetic. Failure returns no result.
  absl::StatusOr<GradedMacParticleGather3D>
  Gather(const LiquidParticleState3D& particle, std::span<const LiquidAdaptiveFaceState3D> carrier_faces) const;

  // Measure the actual radial support for a finite world position and signed
  // particle side. Carriers must cover the canonical global slots with finite
  // velocities and valid enum values. Geometry and rank are independent of
  // the velocity amplitudes. Refuses absent support or nonfinite arithmetic.
  absl::StatusOr<GradedMacSupportMomentWitness3D> MeasureSupportMoments(
      const LiquidParticleState3D& particle,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  ) const;

  // Check the geometric and moment prerequisites of the uniform F3/F4
  // bridge at the requested width in meters. The current radial basis never
  // certifies uniform tensor support, even if its rounded moments coincide.
  // Returns FailedPrecondition for an uncertified support and propagates
  // validation errors. No input is modified and no resampling is performed.
  absl::Status CheckUniformResamplingSupport(
      const LiquidParticleState3D& particle,
      double requested_leaf_width_m,
      LiquidUniformResamplingBasis3D basis,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces
  ) const;

  // Check every positive radial donor at this position and side. Requires
  // a same-level stencil, valid finite carriers and full rank on all three rows.
  // An optional affine carrier must reproduce each normal velocity exactly.
  absl::Status CheckCompleteRadialSupport(
      const LiquidParticleState3D& particle,
      std::span<const LiquidAdaptiveFaceState3D> carriers,
      const LiquidHandoffCarrier3D* affine = nullptr
  ) const;

 private:
  friend class liquid_step_internal::AdaptiveLiquidStep3D;

  // Add one radial particle in canonical slot order to prevalidated scratch.
  // The caller owns the scratch and discards partial values on failure.
  absl::Status AccumulateParticleDeposit(
      const LiquidParticleState3D& particle,
      std::span<const LiquidAdaptiveFaceState3D> carrier_faces,
      std::span<MacFaceDeposit3D> deposits
  ) const;

  struct LeafBox {
    Vec3 lower;
    Vec3 upper;
    double width_m;
  };

  struct FaceSupportBounds {
    std::array<double, 3> lower;  // Minimum world anchor coordinates, m.
    std::array<double, 3> upper;  // Maximum world anchor coordinates, m.
  };

  GradedMacTransfer3D() = default;
  absl::StatusOr<double> ParticleSupportWidth(Vec3 world_position) const;

  std::optional<Vec3> solid_upper_;
  std::vector<LeafBox> leaves_;
  std::vector<GradedMacFace3D> faces_;
  std::vector<FaceSupportBounds> face_support_bounds_;  // Preorder bounds over canonical slot blocks.
};

}  // namespace fluidity
