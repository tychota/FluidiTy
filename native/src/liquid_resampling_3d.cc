#include "fluidity/host/liquid_resampling_3d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace fluidity {
namespace {

constexpr std::uint64_t kSplitChildCount = 8;
constexpr double kIncrementRelativeTolerance = 64.0 * std::numeric_limits<double>::epsilon();

bool PositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

bool Finite(Vec3 vector) {
  return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

bool ValidParticle(const LiquidParticleState3D& particle) {
  return particle.id != 0 && Finite(particle.world_position) && Finite(particle.world_velocity)
      && Finite(particle.velocity_slopes.x_row) && Finite(particle.velocity_slopes.y_row)
      && Finite(particle.velocity_slopes.z_row) && PositiveFinite(particle.mass)
      && PositiveFinite(particle.reference_volume) && PositiveFinite(particle.volume_ratio);
}

bool ValidBasis(LiquidUniformResamplingBasis3D basis) {
  return basis == LiquidUniformResamplingBasis3D::kQuadratic || basis == LiquidUniformResamplingBasis3D::kCubicNormal;
}

// Refuse translations that lose the requested increment at the stored origin.
bool RepresentsIncrement(double origin, double translated, double increment) {
  if (!std::isfinite(translated) || !std::isfinite(increment)) {
    return false;
  }
  if (increment == 0.0) {
    return translated == origin;
  }
  const double stored_increment = translated - origin;
  return std::abs(stored_increment / increment - 1.0) <= kIncrementRelativeTolerance;
}

// Compare velocity error with the row's affine scale, so cancellation among
// contributions does not make a near-zero dot product fail the roundoff gate.
bool TranslateVelocity(double velocity, Vec3 slope_row, Vec3 offset, double displacement, double& translated) {
  const double slope_scale = std::max({std::abs(slope_row.x), std::abs(slope_row.y), std::abs(slope_row.z)});
  if (slope_scale == 0.0) {
    translated = velocity;
    return true;
  }
  const double affine_scale = slope_scale * displacement;
  const double increment = std::fma(slope_row.x, offset.x, std::fma(slope_row.y, offset.y, slope_row.z * offset.z));
  translated = velocity + increment;
  if (!PositiveFinite(affine_scale) || !std::isfinite(translated)) {
    return false;
  }
  const double stored_increment = translated - velocity;
  return std::abs((stored_increment - increment) / affine_scale) <= kIncrementRelativeTolerance;
}

// Scale the diagonal before squaring. Powers of two keep intermediate width,
// mass and slope products in range when the final energy is representable.
double CubicSplitEnergyDeficit(double mass, double width, Mat3 slopes) {
  const double max_diagonal = std::max({std::abs(slopes.x_row.x), std::abs(slopes.y_row.y), std::abs(slopes.z_row.z)});
  if (max_diagonal == 0.0) {
    return 0.0;
  }
  int mass_exponent;
  int width_exponent;
  int slope_exponent;
  const double mass_mantissa = std::frexp(mass, &mass_exponent);
  const double width_mantissa = std::frexp(width, &width_exponent);
  std::frexp(max_diagonal, &slope_exponent);
  const double x = std::scalbn(slopes.x_row.x, -slope_exponent);
  const double y = std::scalbn(slopes.y_row.y, -slope_exponent);
  const double z = std::scalbn(slopes.z_row.z, -slope_exponent);
  const double mantissa = mass_mantissa * width_mantissa * width_mantissa * (x * x + y * y + z * z);
  return std::scalbn(mantissa, mass_exponent + 2 * width_exponent + 2 * slope_exponent - 5);
}

}  // namespace

absl::StatusOr<LiquidUniformSplit3D> SplitUniformLiquidParticle3D(
    const LiquidParticleState3D& parent,
    std::uint64_t first_child_id,
    double parent_cell_width,
    LiquidUniformResamplingBasis3D basis,
    LiquidResamplingAdmissibility3D certify,
    void* context
) {
  if (!ValidParticle(parent) || first_child_id <= parent.id) {
    return absl::InvalidArgumentError("Invalid split parent or first child ID");
  }
  if (!PositiveFinite(parent_cell_width) || !ValidBasis(basis) || certify == nullptr) {
    return absl::InvalidArgumentError("Split requires a positive width, supported basis and admissibility callback");
  }
  if (!parent.affine_enabled) {
    return absl::FailedPreconditionError("Split requires affine-enabled particles");
  }
  if (first_child_id > std::numeric_limits<std::uint64_t>::max() - kSplitChildCount) {
    return absl::OutOfRangeError("Split child IDs exhaust the next-ID counter");
  }
  const double child_cell_width = parent_cell_width / 2.0;
  const double child_mass = parent.mass / kSplitChildCount;
  const double child_reference_volume = parent.reference_volume / kSplitChildCount;
  const double parent_current_volume = parent.reference_volume * parent.volume_ratio;
  const double child_current_volume = child_reference_volume * parent.volume_ratio;
  if (!PositiveFinite(child_cell_width) || child_cell_width * 2.0 != parent_cell_width || !PositiveFinite(child_mass)
      || child_mass * kSplitChildCount != parent.mass || !PositiveFinite(child_reference_volume)
      || child_reference_volume * kSplitChildCount != parent.reference_volume || !PositiveFinite(parent_current_volume)
      || !PositiveFinite(child_current_volume) || child_current_volume * kSplitChildCount != parent_current_volume) {
    return absl::OutOfRangeError("Split spacing or inventories cannot be divided exactly");
  }
  const double displacement = parent_cell_width * (std::sqrt(3.0) / 4.0);
  if (!PositiveFinite(displacement)) {
    return absl::OutOfRangeError("Split displacement is unrepresentable");
  }

  LiquidUniformSplit3D result{};
  result.next_particle_id = first_child_id + kSplitChildCount;
  if (basis == LiquidUniformResamplingBasis3D::kCubicNormal) {
    result.kinetic_affine_energy_deficit =
        CubicSplitEnergyDeficit(parent.mass, parent_cell_width, parent.velocity_slopes);
    const bool nonzero_diagonal = parent.velocity_slopes.x_row.x != 0.0 || parent.velocity_slopes.y_row.y != 0.0
        || parent.velocity_slopes.z_row.z != 0.0;
    if (!std::isfinite(result.kinetic_affine_energy_deficit)
        || (nonzero_diagonal && result.kinetic_affine_energy_deficit == 0.0)) {
      return absl::OutOfRangeError("Split energy deficit is unrepresentable");
    }
  }

  const Vec3 position = parent.world_position;
  const Vec3 velocity = parent.world_velocity;
  const Mat3 slopes = parent.velocity_slopes;
  for (std::size_t ordinal = 0; ordinal < result.children.size(); ++ordinal) {
    const Vec3 offset{
        (ordinal & 1u) != 0u ? displacement : -displacement,
        (ordinal & 2u) != 0u ? displacement : -displacement,
        (ordinal & 4u) != 0u ? displacement : -displacement,
    };
    LiquidParticleState3D child = parent;
    child.id = first_child_id + ordinal;
    child.mass = child_mass;
    child.reference_volume = child_reference_volume;
    child.world_position = {position.x + offset.x, position.y + offset.y, position.z + offset.z};
    if (!RepresentsIncrement(position.x, child.world_position.x, offset.x)
        || !RepresentsIncrement(position.y, child.world_position.y, offset.y)
        || !RepresentsIncrement(position.z, child.world_position.z, offset.z)) {
      return absl::OutOfRangeError("Split position cannot represent the requested offset");
    }
    const Vec3 stored_offset{
        child.world_position.x - position.x,
        child.world_position.y - position.y,
        child.world_position.z - position.z,
    };
    if (!TranslateVelocity(velocity.x, slopes.x_row, stored_offset, displacement, child.world_velocity.x)
        || !TranslateVelocity(velocity.y, slopes.y_row, stored_offset, displacement, child.world_velocity.y)
        || !TranslateVelocity(velocity.z, slopes.z_row, stored_offset, displacement, child.world_velocity.z)) {
      return absl::OutOfRangeError("Split velocity cannot represent the affine increment");
    }
    result.children[ordinal] = child;
  }

  absl::Status status = certify(parent, parent_cell_width, basis, context);
  if (!status.ok()) {
    return status;
  }
  for (const auto& child : result.children) {
    status = certify(child, child_cell_width, basis, context);
    if (!status.ok()) {
      return status;
    }
  }
  return result;
}

}  // namespace fluidity
