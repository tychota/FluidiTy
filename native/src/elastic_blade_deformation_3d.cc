#include "fluidity/host/elastic_blade_deformation_3d.h"

#include <cmath>

namespace fluidity {
namespace {

// The row of L multiplies the old deformation gradient on its left.
Vec3 AdvanceDeformationRow(Vec3 old_row, Vec3 gradient_row, Mat3 old_deformation, double dt) {
  const Vec3 rate{
      gradient_row.x * old_deformation.x_row.x + gradient_row.y * old_deformation.y_row.x
          + gradient_row.z * old_deformation.z_row.x,
      gradient_row.x * old_deformation.x_row.y + gradient_row.y * old_deformation.y_row.y
          + gradient_row.z * old_deformation.z_row.y,
      gradient_row.x * old_deformation.x_row.z + gradient_row.y * old_deformation.y_row.z
          + gradient_row.z * old_deformation.z_row.z,
  };
  return {std::fma(dt, rate.x, old_row.x), std::fma(dt, rate.y, old_row.y), std::fma(dt, rate.z, old_row.z)};
}

}  // namespace

absl::StatusOr<ElasticBladeDeformation3D> GatherElasticBladeDeformation3D(
    const SparseMacGrid3D& mac_grid,
    const MacTransferParticle3D& particle,
    ElasticBladeSide3D blade_side,
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history,
    Mat3 old_deformation_gradient
) {
  const bool assigned_sign = blade_side.sign == -1 || blade_side.sign == 1;
  if (blade_side.blade_id == 0 || !assigned_sign) {
    return absl::InvalidArgumentError("elastic deformation requires an assigned blade side");
  }
  const auto old_domain = NeoHookeanKirchhoffStress3D(old_deformation_gradient, 1.0, 0.0);
  if (!old_domain.ok()) {
    return old_domain.status();
  }
  const double dt = new_history.velocity_sample_time - old_history.velocity_sample_time;
  if (!std::isfinite(dt) || dt <= 0.0) {
    return absl::InvalidArgumentError("invalid elastic deformation time interval");
  }
  const LiquidBoundaryParticle3D boundary_particle{
      .transfer = particle,
      .side_tag = blade_side.sign,
      .affine_enabled = true,
  };
  const auto gathered = GatherLiquidBoundaryParticles3D({
      .mac_grid = mac_grid,
      .particles = {&boundary_particle, 1},
      .old_time = old_history.velocity_sample_time,
      .new_time = new_history.velocity_sample_time,
      .old_history = old_history,
      .new_history = new_history,
  });
  if (!gathered.ok()) {
    return gathered.status();
  }
  const auto& gather = gathered->particles.front();
  if (!gather.affine_enabled) {
    return absl::FailedPreconditionError("elastic deformation requires a full-rank velocity continuation");
  }

  const Mat3 gradient = gather.velocity_gradient;
  const Mat3 candidate{
      .x_row = AdvanceDeformationRow(old_deformation_gradient.x_row, gradient.x_row, old_deformation_gradient, dt),
      .y_row = AdvanceDeformationRow(old_deformation_gradient.y_row, gradient.y_row, old_deformation_gradient, dt),
      .z_row = AdvanceDeformationRow(old_deformation_gradient.z_row, gradient.z_row, old_deformation_gradient, dt),
  };
  if (!std::isfinite(candidate.x_row.x) || !std::isfinite(candidate.x_row.y) || !std::isfinite(candidate.x_row.z)
      || !std::isfinite(candidate.y_row.x) || !std::isfinite(candidate.y_row.y) || !std::isfinite(candidate.y_row.z)
      || !std::isfinite(candidate.z_row.x) || !std::isfinite(candidate.z_row.y) || !std::isfinite(candidate.z_row.z)) {
    return absl::OutOfRangeError("nonfinite elastic deformation");
  }
  const auto candidate_domain = NeoHookeanKirchhoffStress3D(candidate, 1.0, 0.0);
  if (absl::IsInvalidArgument(candidate_domain.status())) {
    return absl::FailedPreconditionError("elastic deformation loses positive volume");
  }
  if (!candidate_domain.ok()) {
    return candidate_domain.status();
  }
  return ElasticBladeDeformation3D{.gather = gather, .deformation_gradient = candidate};
}

}  // namespace fluidity
