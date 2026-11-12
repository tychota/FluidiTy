#include "fluidity/host/transfers_3d.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/views.h"
#include "mac_gather_3d_internal.h"

namespace fluidity {
namespace {

struct MacNormalComponentAccess {
  FaceNormalAxis axis;
  double Vec3::* velocity_component;
  Vec3 Mat3::* velocity_slope_row;
};

constexpr std::array kNormalComponentAccess{
    MacNormalComponentAccess{FaceNormalAxis::kX, &Vec3::x, &Mat3::x_row},
    MacNormalComponentAccess{FaceNormalAxis::kY, &Vec3::y, &Mat3::y_row},
    MacNormalComponentAccess{FaceNormalAxis::kZ, &Vec3::z, &Mat3::z_row},
};
constexpr std::size_t kMaxQuantizedParticleCapacity = std::size_t{1} << 20;

bool IsFinite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool IsFinite(Mat3 value) {
  return IsFinite(value.x_row) && IsFinite(value.y_row) && IsFinite(value.z_row);
}

bool IsPositiveNormalPowerOfTwo(double scale) {
  int exponent;
  return std::isnormal(scale) && std::frexp(scale, &exponent) == 0.5;
}

double Dot(Vec3 lhs, Vec3 rhs) {
  return (lhs.x * rhs.x + lhs.y * rhs.y) + lhs.z * rhs.z;
}

// The range guard has already bounded the magnitude by 2^62. Truncate the
// magnitude first, then resolve the fraction and parity independently of the
// ambient integer rounding direction.
std::int64_t RoundNearestEven(double scaled_contribution) {
  const double magnitude = std::abs(scaled_contribution);
  const double integral_magnitude = std::floor(magnitude);
  const double fraction = magnitude - integral_magnitude;
  auto quanta_magnitude = static_cast<std::int64_t>(integral_magnitude);
  const bool odd_tie = fraction == 0.5 && quanta_magnitude % 2 != 0;
  if (fraction > 0.5 || odd_tie) {
    ++quanta_magnitude;
  }
  return scaled_contribution < 0.0 ? -quanta_magnitude : quanta_magnitude;
}

template <typename FaceDeposit, typename AccumulateFaceContribution>
absl::Status DepositWithFaceAccumulator(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    std::span<FaceDeposit> face_deposits,
    AccumulateFaceContribution accumulate_face_contribution
) {
  const auto face_deposit_extent_status = ValidateExtent(face_deposits.size(), mac_grid.face_count());
  if (!face_deposit_extent_status.ok()) {
    return face_deposit_extent_status;
  }
  const auto alias_status = CheckWritableViews({std::as_bytes(face_deposits)}, {std::as_bytes(particles)});
  if (!alias_status.ok()) {
    return alias_status;
  }

  // Resolve and deposit in particle order into scratch. An unsupported face
  // or arithmetic failure on a later particle must leave caller storage intact.
  std::vector<FaceDeposit> staged_face_deposits(face_deposits.size());
  for (const auto& particle : particles) {
    if (!std::isfinite(particle.particle_mass) || particle.particle_mass < 0.0) {
      return absl::InvalidArgumentError("particle mass must be finite and nonnegative");
    }
    if (!IsFinite(particle.particle_velocity) || !IsFinite(particle.particle_velocity_slopes)) {
      return absl::InvalidArgumentError("particle velocity and affine fit must be finite");
    }
    for (const auto component : kNormalComponentAccess) {
      const auto stencil = BuildOwnedMacFaceStencil3D(
          mac_grid,
          component.axis,
          particle.sorted_world_position,
          particle.trial_world_position
      );
      if (!stencil.ok()) {
        return stencil.status();
      }
      const double particle_velocity_component = particle.particle_velocity.*component.velocity_component;
      const Vec3 particle_velocity_slope_row = particle.particle_velocity_slopes.*component.velocity_slope_row;
      for (const auto& sample : *stencil) {
        const double mass_contribution = sample.weight * particle.particle_mass;
        const double velocity_at_face =
            particle_velocity_component + Dot(particle_velocity_slope_row, sample.face_offset_from_particle);
        const double pure_momentum_contribution = mass_contribution * velocity_at_face;
        const auto status = accumulate_face_contribution(
            mass_contribution,
            pure_momentum_contribution,
            staged_face_deposits[sample.face_slot]
        );
        if (!status.ok()) {
          return status;
        }
      }
    }
  }
  std::ranges::copy(staged_face_deposits, face_deposits.begin());
  return absl::OkStatus();
}

using internal::GatherMacVelocityComponent;

}  // namespace

absl::Status DepositMac3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    std::span<MacFaceDeposit3D> face_deposits
) {
  return DepositWithFaceAccumulator(
      mac_grid,
      particles,
      face_deposits,
      [](double mass_contribution, double pure_momentum_contribution, MacFaceDeposit3D& face) {
        face.deposited_mass += mass_contribution;
        face.deposited_pure_momentum += pure_momentum_contribution;
        if (!std::isfinite(face.deposited_mass) || !std::isfinite(face.deposited_pure_momentum)) {
          return absl::OutOfRangeError("deposited mass or pure momentum is nonfinite");
        }
        return absl::OkStatus();
      }
  );
}

absl::Status DepositMacQuanta3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    MacDepositQuantization3D quantization,
    std::span<MacFaceDepositQuanta3D> face_deposits
) {
  if (!IsPositiveNormalPowerOfTwo(quantization.mass_per_quantum)
      || !IsPositiveNormalPowerOfTwo(quantization.pure_momentum_per_quantum)) {
    return absl::InvalidArgumentError("quantization scales must be positive normal powers of two");
  }
  if (quantization.declared_particle_capacity > kMaxQuantizedParticleCapacity) {
    return absl::ResourceExhaustedError("declared particle capacity exceeds the transfer limit");
  }
  if (particles.size() > quantization.declared_particle_capacity) {
    return absl::ResourceExhaustedError("particle count exceeds declared capacity");
  }

  // Use the smallest power-of-two contributor bound that covers the capacity.
  // A single particle can reach each owned face at most once.
  const std::size_t nonzero_particle_capacity = std::max(std::size_t{1}, quantization.declared_particle_capacity);
  const int particle_capacity_exponent = static_cast<int>(std::bit_width(nonzero_particle_capacity - 1));
  const double scaled_contribution_limit = std::ldexp(1.0, 62 - particle_capacity_exponent);
  return DepositWithFaceAccumulator(
      mac_grid,
      particles,
      face_deposits,
      [quantization, scaled_contribution_limit](
          double mass_contribution,
          double pure_momentum_contribution,
          MacFaceDepositQuanta3D& face
      ) {
        const double scaled_mass_contribution = mass_contribution / quantization.mass_per_quantum;
        const double scaled_pure_momentum_contribution =
            pure_momentum_contribution / quantization.pure_momentum_per_quantum;
        const bool scaled_mass_in_range =
            std::isfinite(scaled_mass_contribution) && std::abs(scaled_mass_contribution) <= scaled_contribution_limit;
        const bool scaled_pure_momentum_in_range = std::isfinite(scaled_pure_momentum_contribution)
            && std::abs(scaled_pure_momentum_contribution) <= scaled_contribution_limit;
        if (!scaled_mass_in_range || !scaled_pure_momentum_in_range) {
          return absl::OutOfRangeError("scaled mass or pure momentum contribution exceeds the quantization range");
        }
        const std::int64_t quantized_mass_contribution = RoundNearestEven(scaled_mass_contribution);
        const std::int64_t quantized_pure_momentum_contribution = RoundNearestEven(scaled_pure_momentum_contribution);
        // One contribution per particle reaches any owned face. The sum of
        // absolute rounded contributions is at most 2^62 in each channel.
        face.mass_quanta_sum += quantized_mass_contribution;
        face.pure_momentum_quanta_sum += quantized_pure_momentum_contribution;
        return absl::OkStatus();
      }
  );
}

absl::Status GatherMac3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    std::span<MacParticleGatherResult3D> particle_gather_results
) {
  const auto face_extents =
      CheckExtents(mac_grid.face_count(), {old_face_velocities.size(), new_face_velocities.size()});
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto particle_extent = ValidateExtent(particle_gather_results.size(), particles.size());
  if (!particle_extent.ok()) {
    return particle_extent;
  }
  const auto alias_status = CheckWritableViews(
      {std::as_bytes(particle_gather_results)},
      {std::as_bytes(particles), std::as_bytes(old_face_velocities), std::as_bytes(new_face_velocities)}
  );
  if (!alias_status.ok()) {
    return alias_status;
  }
  const auto is_finite_face_velocity = [](double face_velocity) {
    return std::isfinite(face_velocity);
  };
  if (!std::ranges::all_of(old_face_velocities, is_finite_face_velocity)
      || !std::ranges::all_of(new_face_velocities, is_finite_face_velocity)) {
    return absl::InvalidArgumentError("old and new face velocity fields must be finite");
  }

  const double cell_width = mac_grid.geometry().cell_width;
  std::vector<MacParticleGatherResult3D> staged_particle_gather_results(particles.size());
  for (std::size_t particle_index = 0; particle_index < particles.size(); ++particle_index) {
    const auto& particle = particles[particle_index];
    auto& staged_particle_result = staged_particle_gather_results[particle_index];
    for (const auto component : kNormalComponentAccess) {
      const auto stencil = BuildOwnedMacFaceStencil3D(
          mac_grid,
          component.axis,
          particle.sorted_world_position,
          particle.trial_world_position
      );
      if (!stencil.ok()) {
        return stencil.status();
      }
      const auto gathered_component =
          GatherMacVelocityComponent(*stencil, cell_width, old_face_velocities, new_face_velocities);
      if (!gathered_component.ok()) {
        return gathered_component.status();
      }
      staged_particle_result.new_pic_velocity.*component.velocity_component = gathered_component->new_pic_velocity;
      staged_particle_result.grid_velocity_increment.*component.velocity_component =
          gathered_component->grid_velocity_increment;
      staged_particle_result.fitted_velocity_slopes.*component.velocity_slope_row =
          gathered_component->fitted_velocity_slope_row;
      staged_particle_result.velocity_gradient.*component.velocity_slope_row =
          gathered_component->velocity_gradient_row;
    }
  }
  std::ranges::copy(staged_particle_gather_results, particle_gather_results.begin());
  return absl::OkStatus();
}

absl::StatusOr<Mat3>
NeoHookeanKirchhoffStress3D(Mat3 deformation_gradient, double shear_modulus, double first_lame_parameter) {
  if (!IsFinite(deformation_gradient) || !std::isfinite(shear_modulus) || !std::isfinite(first_lame_parameter)
      || shear_modulus <= 0.0 || first_lame_parameter < 0.0) {
    return absl::InvalidArgumentError("invalid Neo-Hookean input");
  }
  const Vec3 x = deformation_gradient.x_row;
  const Vec3 y = deformation_gradient.y_row;
  const Vec3 z = deformation_gradient.z_row;
  const Vec3 cofactor{
      y.y * z.z - y.z * z.y,
      y.z * z.x - y.x * z.z,
      y.x * z.y - y.y * z.x,
  };
  const double volume_ratio = Dot(x, cofactor);
  if (!std::isfinite(volume_ratio)) {
    return absl::OutOfRangeError("nonfinite elastic volume ratio");
  }
  if (volume_ratio <= 0.0) {
    return absl::InvalidArgumentError("elastic deformation requires a positive determinant");
  }

  const double volumetric_stress = first_lame_parameter * std::log(volume_ratio);
  const double xy_stress = shear_modulus * Dot(x, y);
  const double xz_stress = shear_modulus * Dot(x, z);
  const double yz_stress = shear_modulus * Dot(y, z);
  const Mat3 kirchhoff_stress{
      .x_row = {shear_modulus * (Dot(x, x) - 1.0) + volumetric_stress, xy_stress, xz_stress},
      .y_row = {xy_stress, shear_modulus * (Dot(y, y) - 1.0) + volumetric_stress, yz_stress},
      .z_row = {xz_stress, yz_stress, shear_modulus * (Dot(z, z) - 1.0) + volumetric_stress},
  };
  if (!IsFinite(kirchhoff_stress)) {
    return absl::OutOfRangeError("nonfinite Kirchhoff stress");
  }
  return kirchhoff_stress;
}

absl::Status DepositElasticStress3D(
    const SparseMacGrid3D& mac_grid,
    Vec3 sorted_particle_world_position,
    Vec3 trial_particle_world_position,
    double reference_volume,
    Mat3 kirchhoff_stress,
    std::span<double> normal_face_forces
) {
  const auto extent_status = ValidateExtent(normal_face_forces.size(), mac_grid.face_count());
  if (!extent_status.ok()) {
    return extent_status;
  }
  if (!IsFinite(kirchhoff_stress) || !std::isfinite(reference_volume) || reference_volume <= 0.0) {
    return absl::InvalidArgumentError("invalid elastic stress or reference volume");
  }
  struct FaceForce {
    std::size_t face_slot;
    double stress_force_contribution;
  };
  std::array<FaceForce, kNormalComponentAccess.size() * MacFaceStencil3D{}.size()> staged_forces;
  std::size_t contribution_count = 0;
  for (const auto component : kNormalComponentAccess) {
    const auto stencil = BuildOwnedMacFaceStencil3D(
        mac_grid,
        component.axis,
        sorted_particle_world_position,
        trial_particle_world_position
    );
    if (!stencil.ok()) {
      return stencil.status();
    }
    const Vec3 stress_row = kirchhoff_stress.*component.velocity_slope_row;
    for (const auto& sample : *stencil) {
      const double stress_force_contribution = -reference_volume * Dot(stress_row, sample.weight_gradient);
      if (!std::isfinite(stress_force_contribution)) {
        return absl::OutOfRangeError("nonfinite elastic stress force");
      }
      staged_forces[contribution_count++] = {sample.face_slot, stress_force_contribution};
    }
  }
  std::ranges::fill(normal_face_forces, 0.0);
  for (const auto& contribution : staged_forces) {
    normal_face_forces[contribution.face_slot] = contribution.stress_force_contribution;
  }
  return absl::OkStatus();
}

}  // namespace fluidity
