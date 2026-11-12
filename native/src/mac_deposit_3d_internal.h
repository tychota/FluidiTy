#pragma once

#include <array>
#include <cmath>
#include <span>

#include "absl/status/status.h"
#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/sparse_mac_grid_3d.h"

namespace fluidity::internal {

struct MacNormalComponentAccess {
  FaceNormalAxis axis;
  double Vec3::* velocity_component;
  Vec3 Mat3::* velocity_slope_row;
};

inline constexpr std::array kNormalComponentAccess{
    MacNormalComponentAccess{FaceNormalAxis::kX, &Vec3::x, &Mat3::x_row},
    MacNormalComponentAccess{FaceNormalAxis::kY, &Vec3::y, &Mat3::y_row},
    MacNormalComponentAccess{FaceNormalAxis::kZ, &Vec3::z, &Mat3::z_row},
};

inline absl::Status ValidateMacDepositParticle(const MacTransferParticle3D& particle) {
  if (!std::isfinite(particle.particle_mass) || particle.particle_mass < 0.0) {
    return absl::InvalidArgumentError("particle mass must be finite and nonnegative");
  }
  const auto finite = [](Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
  };
  const auto& slopes = particle.particle_velocity_slopes;
  if (!finite(particle.particle_velocity) || !finite(slopes.x_row) || !finite(slopes.y_row) || !finite(slopes.z_row)) {
    return absl::InvalidArgumentError("particle velocity and affine fit must be finite");
  }
  return absl::OkStatus();
}

// Deposit an already checked and bound quadratic row in tensor order.
template <typename FaceDeposit, typename AccumulateFaceContribution>
absl::Status AccumulateMacParticleComponent(
    const MacFaceStencil3D& stencil,
    const MacTransferParticle3D& particle,
    MacNormalComponentAccess component,
    std::span<FaceDeposit> face_deposits,
    AccumulateFaceContribution&& accumulate_face_contribution
) {
  const double particle_velocity_component = particle.particle_velocity.*component.velocity_component;
  const Vec3 particle_velocity_slope_row = particle.particle_velocity_slopes.*component.velocity_slope_row;
  for (const auto& sample : stencil) {
    const auto offset = sample.face_offset_from_particle;
    const double mass_contribution = sample.weight * particle.particle_mass;
    const double affine_velocity_increment =
        (particle_velocity_slope_row.x * offset.x + particle_velocity_slope_row.y * offset.y)
        + particle_velocity_slope_row.z * offset.z;
    const double velocity_at_face = particle_velocity_component + affine_velocity_increment;
    const double pure_momentum_contribution = mass_contribution * velocity_at_face;
    const auto status =
        accumulate_face_contribution(mass_contribution, pure_momentum_contribution, face_deposits[sample.face_slot]);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

inline absl::Status
AccumulateMacFaceDeposit(double mass_contribution, double pure_momentum_contribution, MacFaceDeposit3D& face) {
  face.deposited_mass += mass_contribution;
  face.deposited_pure_momentum += pure_momentum_contribution;
  if (!std::isfinite(face.deposited_mass) || !std::isfinite(face.deposited_pure_momentum)) {
    return absl::OutOfRangeError("deposited mass or pure momentum is nonfinite");
  }
  return absl::OkStatus();
}

}  // namespace fluidity::internal
