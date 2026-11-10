#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity::liquid_step_internal {

// The first isolated drop field bounds its all-pairs graph by the authored seed size.
inline constexpr std::size_t kMaxIsolatedDropParticles3D = 552;

class PreparedIsolatedDrop3D {
 public:
  // Return the world centroid in meters.
  Vec3 world_center() const;
  // Return the largest semiaxis in meters for conservative lattice coverage.
  double maximum_semiaxis() const;

  // Sample (R/2)(q − 1) in meters, negative inside, where q is the fitted
  // inverse shape applied to the world offset from the centroid. The fixed
  // authored radius R scales the level set. Returns OutOfRange for nonfinite arithmetic.
  absl::StatusOr<double> QuadraticGridLevelSet(Vec3 world_position) const;

  // Return the Euclidean signed distance in meters, negative inside. Returns
  // OutOfRange if the position or intermediate values are nonfinite.
  absl::StatusOr<double> ParticleSignedDistance(Vec3 world_position) const;

  // Return the sum of outward principal curvatures in inverse meters at the
  // wet-to-air crossing (+2/R for a sphere). The wet center must have level < 0
  // and the air center level >= 0. Returns FailedPrecondition without that
  // bracket and OutOfRange for nonfinite arithmetic.
  absl::StatusOr<double> AirLinkCurvature(Vec3 wet_world_center, Vec3 air_world_center) const;

 private:
  friend absl::StatusOr<PreparedIsolatedDrop3D>
  PrepareIsolatedDrop3D(std::span<const LiquidParticleState3D>, double, double, double);

  PreparedIsolatedDrop3D() = default;

  Vec3 center_{};
  Mat3 inverse_shape_{};  // World quadratic form, inverse square meters.
  Mat3 principal_frame_{};  // Orthonormal principal axes in columns.
  std::array<double, 3> semiaxes_{};  // Meters, in principal frame order.
  double physical_radius_ = 0;  // Meters, fixed scale of the grid level set.
};

// Fit the world centroid and shape of 4–552 samples in input order. The shape
// is the mass-weighted covariance times the fixed, dimensionless seed scale.
// The caller supplies a filled drop. Edges of length <= 1.04h must connect the
// samples. InvalidArgument denotes invalid parameters or samples,
// FailedPrecondition denotes disconnected or degenerate geometry, and
// OutOfRange denotes nonfinite fit arithmetic.
absl::StatusOr<PreparedIsolatedDrop3D> PrepareIsolatedDrop3D(
    std::span<const LiquidParticleState3D> particles,
    double cell_width,
    double physical_radius,
    double seed_covariance_scale
);

}  // namespace fluidity::liquid_step_internal
