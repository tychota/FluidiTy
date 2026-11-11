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

constexpr std::size_t kMaxMergeChildren = 8;
using Cartesian = std::array<double, 3>;
using Rows = std::array<Cartesian, 3>;

Cartesian Components(Vec3 value) {
  return {value.x, value.y, value.z};
}

Rows Components(Mat3 value) {
  return {Components(value.x_row), Components(value.y_row), Components(value.z_row)};
}

Vec3 AsVec3(Cartesian value) {
  return {value[0], value[1], value[2]};
}

struct Bounds {
  double lower;
  double upper;
};

static_assert(std::numeric_limits<double>::is_iec559);
constexpr double kInfinity = std::numeric_limits<double>::infinity();

Bounds Point(double value) {
  return {value, value};
}

Bounds Outward(double lower, double upper) {
  if (!std::isfinite(lower) || !std::isfinite(upper)) {
    return {-kInfinity, kInfinity};
  }
  return {std::nextafter(lower, -kInfinity), std::nextafter(upper, kInfinity)};
}

Bounds Add(Bounds left, Bounds right) {
  return Outward(left.lower + right.lower, left.upper + right.upper);
}

Bounds Difference(Bounds left, Bounds right) {
  return Outward(left.lower - right.upper, left.upper - right.lower);
}

Bounds Multiply(Bounds left, Bounds right) {
  const std::array products{
      left.lower * right.lower,
      left.lower * right.upper,
      left.upper * right.lower,
      left.upper * right.upper,
  };
  if (!std::ranges::all_of(products, [](double product) {
        return std::isfinite(product);
      })) {
    return {-kInfinity, kInfinity};
  }
  const auto [lower, upper] = std::ranges::minmax(products);
  return Outward(lower, upper);
}

Bounds DivideByPositive(Bounds value, Bounds divisor) {
  if (divisor.lower <= 0.0 || !std::isfinite(divisor.upper)) {
    return {-kInfinity, kInfinity};
  }
  return Multiply(value, Outward(1.0 / divisor.upper, 1.0 / divisor.lower));
}

bool IsCubicNormal(LiquidUniformResamplingBasis3D basis, std::size_t row, std::size_t axis) {
  return basis == LiquidUniformResamplingBasis3D::kCubicNormal && row == axis;
}

// In units of fine spacing squared, D_target - D_child is exactly 1
// on a cubic normal axis and 3/4 on a quadratic axis. Enclose the
// centered mass-weighted covariance with pair differences so a rounded
// barycenter cannot falsely certify a positive covariance gap.
double MergeCovarianceLowerBound(
    std::span<const LiquidParticleState3D> children,
    Bounds total_mass,
    double fine_cell_width,
    LiquidUniformResamplingBasis3D basis
) {
  std::array<Bounds, kMaxMergeChildren> weights{};
  for (std::size_t index = 0; index < children.size(); ++index) {
    weights[index] = DivideByPositive(Point(children[index].mass), total_mass);
  }
  std::array<std::array<Bounds, 3>, 3> covariance{};
  for (std::size_t right = 1; right < children.size(); ++right) {
    const Cartesian right_position = Components(children[right].world_position);
    for (std::size_t left = 0; left < right; ++left) {
      const Cartesian left_position = Components(children[left].world_position);
      const Bounds pair_weight = Multiply(weights[left], weights[right]);
      std::array<Bounds, 3> offset{};
      for (std::size_t axis = 0; axis < offset.size(); ++axis) {
        const Bounds difference = Difference(Point(right_position[axis]), Point(left_position[axis]));
        offset[axis] = DivideByPositive(difference, Point(fine_cell_width));
      }
      for (std::size_t axis = 0; axis < offset.size(); ++axis) {
        for (std::size_t column = axis; column < offset.size(); ++column) {
          const Bounds covariance_contribution = Multiply(pair_weight, Multiply(offset[axis], offset[column]));
          covariance[axis][column] = Add(covariance[axis][column], covariance_contribution);
        }
      }
    }
  }
  double lower_bound = kInfinity;
  for (std::size_t axis = 0; axis < covariance.size(); ++axis) {
    Bounds radius = Point(0.0);
    for (std::size_t column = 0; column < covariance.size(); ++column) {
      if (column != axis) {
        const Bounds cross = covariance[std::min(axis, column)][std::max(axis, column)];
        radius = Add(radius, Point(std::max(std::abs(cross.lower), std::abs(cross.upper))));
      }
    }
    for (std::size_t row = 0; row < covariance.size(); ++row) {
      const double support_gap = IsCubicNormal(basis, row, axis) ? 1.0 : 0.75;
      const Bounds diagonal_gap = Difference(Point(support_gap), covariance[axis][axis]);
      const Bounds disc = Difference(diagonal_gap, radius);
      lower_bound = std::min(lower_bound, disc.lower);
    }
  }
  return lower_bound;
}

// Accumulate weighted displacements from the first child to avoid rounding
// a large common world origin into every weighted contribution.
bool MergeCenter(
    std::span<const LiquidParticleState3D> children,
    std::span<const double> weights,
    Vec3 LiquidParticleState3D::* member,
    Vec3& center
) {
  const Cartesian origin = Components(children.front().*member);
  Cartesian weighted_offset{};
  Cartesian offset_scale{};
  for (std::size_t index = 1; index < children.size(); ++index) {
    const Cartesian sample = Components(children[index].*member);
    for (std::size_t axis = 0; axis < sample.size(); ++axis) {
      const double offset = sample[axis] - origin[axis];
      weighted_offset[axis] = std::fma(weights[index], offset, weighted_offset[axis]);
      offset_scale[axis] = std::max(offset_scale[axis], std::abs(offset));
    }
  }
  Cartesian result{};
  for (std::size_t axis = 0; axis < result.size(); ++axis) {
    result[axis] = origin[axis] + weighted_offset[axis];
    if (!std::isfinite(result[axis])) {
      return false;
    }
    if (offset_scale[axis] > 0.0) {
      const double stored_offset = result[axis] - origin[axis];
      const double scaled_error = std::abs((stored_offset - weighted_offset[axis]) / offset_scale[axis]);
      if (scaled_error > kIncrementRelativeTolerance) {
        return false;
      }
    }
  }
  center = AsVec3(result);
  return true;
}

// F4's centered velocity-position cross moment can also use pair differences.
// The common support reduces the affine part to a mass-weighted quarter of
// the fine slopes. Normalize the orbital part by fine width before forming it.
bool MergeAffine(
    std::span<const LiquidParticleState3D> children,
    std::span<const double> weights,
    double fine_cell_width,
    LiquidUniformResamplingBasis3D basis,
    Mat3& slopes
) {
  Rows support_slopes{};
  Rows cross_moment{};
  for (std::size_t right = 0; right < children.size(); ++right) {
    const Cartesian right_position = Components(children[right].world_position);
    const Cartesian right_velocity = Components(children[right].world_velocity);
    const Rows child_slopes = Components(children[right].velocity_slopes);
    for (std::size_t row = 0; row < child_slopes.size(); ++row) {
      for (std::size_t axis = 0; axis < child_slopes[row].size(); ++axis) {
        const double scaled_slope = child_slopes[row][axis] * 0.25;
        if (scaled_slope * 4.0 != child_slopes[row][axis]) {
          return false;
        }
        support_slopes[row][axis] = std::fma(weights[right], scaled_slope, support_slopes[row][axis]);
      }
    }
    for (std::size_t left = 0; left < right; ++left) {
      const Cartesian left_position = Components(children[left].world_position);
      const Cartesian left_velocity = Components(children[left].world_velocity);
      const double pair_weight = weights[left] * weights[right];
      if (!PositiveFinite(pair_weight)) {
        return false;
      }
      for (std::size_t row = 0; row < right_velocity.size(); ++row) {
        const double weighted_velocity_difference = pair_weight * (right_velocity[row] - left_velocity[row]);
        for (std::size_t axis = 0; axis < right_position.size(); ++axis) {
          const double normalized_offset = (right_position[axis] - left_position[axis]) / fine_cell_width;
          cross_moment[row][axis] = std::fma(weighted_velocity_difference, normalized_offset, cross_moment[row][axis]);
        }
      }
    }
  }
  Rows result{};
  for (std::size_t row = 0; row < result.size(); ++row) {
    for (std::size_t axis = 0; axis < result[row].size(); ++axis) {
      const double inverse_scaled_target = IsCubicNormal(basis, row, axis) ? 0.75 : 1.0;
      result[row][axis] =
          std::fma(inverse_scaled_target, cross_moment[row][axis] / fine_cell_width, support_slopes[row][axis]);
      if (!std::isfinite(result[row][axis])) {
        return false;
      }
    }
  }
  slopes = {AsVec3(result[0]), AsVec3(result[1]), AsVec3(result[2])};
  return true;
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

absl::StatusOr<LiquidUniformMerge3D> MergeUniformLiquidParticles3D(
    std::span<const LiquidParticleState3D> children,
    std::uint64_t target_id,
    double fine_cell_width,
    LiquidUniformResamplingBasis3D basis,
    LiquidResamplingAdmissibility3D certify,
    void* context
) {
  if (children.size() < 2 || children.size() > kMaxMergeChildren || !PositiveFinite(fine_cell_width)
      || !ValidBasis(basis) || certify == nullptr) {
    return absl::InvalidArgumentError("Merge requires two to eight children, a positive width, basis and callback");
  }
  if (target_id == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("Merge target ID exhausts the next-ID counter");
  }
  const double coarse_cell_width = fine_cell_width * 2.0;
  if (!PositiveFinite(coarse_cell_width)) {
    return absl::OutOfRangeError("Merge target spacing is unrepresentable");
  }

  const auto& first = children.front();
  double total_mass = 0.0;
  Bounds exact_mass = Point(0.0);
  double reference_volume = 0.0;
  double current_volume = 0.0;
  for (std::size_t index = 0; index < children.size(); ++index) {
    const auto& child = children[index];
    if (!ValidParticle(child) || child.id >= target_id) {
      return absl::InvalidArgumentError("Invalid merge child or target ID");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (children[prior].id == child.id) {
        return absl::InvalidArgumentError("Duplicate merge child ID");
      }
    }
    if (!child.affine_enabled || child.tag != first.tag || child.side_tag != first.side_tag
        || child.pic_initialized != first.pic_initialized) {
      return absl::FailedPreconditionError("Merge children must share affine, tag, side and PIC eligibility");
    }
    const double child_current_volume = child.volume_ratio * child.reference_volume;
    if (!PositiveFinite(child_current_volume)) {
      return absl::OutOfRangeError("Merge child current volume is unrepresentable");
    }
    total_mass += child.mass;
    exact_mass = Add(exact_mass, Point(child.mass));
    reference_volume += child.reference_volume;
    current_volume += child_current_volume;
  }
  if (!PositiveFinite(total_mass) || !PositiveFinite(reference_volume) || !PositiveFinite(current_volume)) {
    return absl::OutOfRangeError("Merge inventories are unrepresentable");
  }

  const double covariance_bound = MergeCovarianceLowerBound(children, exact_mass, fine_cell_width, basis);
  if (!PositiveFinite(covariance_bound)) {
    return absl::FailedPreconditionError("Merge row covariance gap could not be certified positive");
  }
  std::array<double, kMaxMergeChildren> weights{};
  for (std::size_t index = 0; index < children.size(); ++index) {
    weights[index] = children[index].mass / total_mass;
    if (!PositiveFinite(weights[index])) {
      return absl::OutOfRangeError("Merge mass weight is unrepresentable");
    }
  }

  LiquidParticleState3D target = first;
  target.id = target_id;
  target.mass = total_mass;
  target.reference_volume = reference_volume;
  target.volume_ratio = current_volume / reference_volume;
  const double represented_current_volume = target.volume_ratio * target.reference_volume;
  if (!PositiveFinite(target.volume_ratio) || !PositiveFinite(represented_current_volume)
      || std::abs(represented_current_volume / current_volume - 1.0) > kIncrementRelativeTolerance) {
    return absl::OutOfRangeError("Merge target volume ratio is unrepresentable");
  }
  if (!MergeCenter(children, weights, &LiquidParticleState3D::world_position, target.world_position)
      || !MergeCenter(children, weights, &LiquidParticleState3D::world_velocity, target.world_velocity)
      || !MergeAffine(children, weights, fine_cell_width, basis, target.velocity_slopes)) {
    return absl::OutOfRangeError("Merge target position, velocity or affine rows are unrepresentable");
  }

  for (const auto& child : children) {
    const absl::Status status = certify(child, fine_cell_width, basis, context);
    if (!status.ok()) {
      return status;
    }
  }
  const absl::Status status = certify(target, coarse_cell_width, basis, context);
  if (!status.ok()) {
    return status;
  }
  return LiquidUniformMerge3D{
      .particle = target,
      .next_particle_id = target_id + 1,
      .covariance_gap_lower_bound = covariance_bound,
  };
}

}  // namespace fluidity
