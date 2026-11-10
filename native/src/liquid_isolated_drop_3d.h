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
  // Sample (R/2)(r² − 1) in meters at a finite world position, negative inside.
  // This is the quadratic grid level set. Returns OutOfRange for nonfinite arithmetic.
  absl::StatusOr<double> QuadraticGridLevelSet(Vec3 world_position) const;

  // Return the Euclidean signed distance in meters, negative inside. Returns
  // OutOfRange if the position or intermediate values are nonfinite.
  absl::StatusOr<double> ParticleSignedDistance(Vec3 world_position) const;

  // Return outward curvature in inverse meters where a finite world-space segment
  // crosses from liquid to air. Returns FailedPrecondition without a wet-to-air
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

// Fit the mass-weighted covariance in the input particle order. The caller
// supplies a filled isolated drop and a fixed, dimensionless seed covariance
// scale. The 1.04h graph must contain one component. InvalidArgument denotes
// invalid parameters or samples, FailedPrecondition denotes disconnected or
// degenerate geometry, and OutOfRange denotes nonfinite fit arithmetic.
absl::StatusOr<PreparedIsolatedDrop3D> PrepareIsolatedDrop3D(
    std::span<const LiquidParticleState3D> particles,
    double cell_width,
    double physical_radius,
    double seed_covariance_scale
);

}  // namespace fluidity::liquid_step_internal
