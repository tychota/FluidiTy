#include "fluidity/host/liquid_wall_contact_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

#include "absl/status/status.h"

namespace fluidity {
namespace {

constexpr double kUnitNormalTolerance = 1e-12;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

bool HasFiniteComponents(Vec3 vector) {
  return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z);
}

Vec3 Advance(Vec3 position, Vec3 velocity, double duration) {
  return {
      std::fma(velocity.x, duration, position.x),
      std::fma(velocity.y, duration, position.y),
      std::fma(velocity.z, duration, position.z),
  };
}

struct RoundedSum {
  double value;
  double residual;
};

// FastTwoSum with the larger operand first. This also preserves a small offset
// when a later term cancels a large translation.
RoundedSum AddWithResidual(double left, double right) {
  if (std::abs(left) < std::abs(right)) {
    std::swap(left, right);
  }
  const double sum = left + right;
  if (!std::isfinite(sum)) {
    return {sum, 0};
  }
  return {sum, right - (sum - left)};
}

double AddDown(double left, double right) {
  const auto sum = AddWithResidual(left, right);
  return sum.residual < 0 ? std::nextafter(sum.value, -kInfinity) : sum.value;
}

double AddUp(double left, double right) {
  const auto sum = AddWithResidual(left, right);
  return sum.residual > 0 ? std::nextafter(sum.value, kInfinity) : sum.value;
}

struct ProjectionBounds {
  double estimate;
  double lower;
  double upper;

  bool finite() const {
    return std::isfinite(estimate) && std::isfinite(lower) && std::isfinite(upper);
  }

  // A certified sign suffices to classify an event. Computing the hit time
  // or impulse also needs a resolved magnitude after cancellation.
  bool magnitude_is_resolved() const {
    constexpr double kRelativeResolution = 16 * std::numeric_limits<double>::epsilon();
    return finite() && upper - lower <= kRelativeResolution * std::abs(estimate);
  }

  std::optional<int> sign() const {
    if (!finite()) {
      return std::nullopt;
    }
    if (lower > 0) {
      return 1;
    }
    if (upper < 0) {
      return -1;
    }
    if (lower == 0 && upper == 0) {
      return 0;
    }
    return std::nullopt;
  }
};

// Binary64 arithmetic in round-to-nearest. Retain sum and product residuals
// separately so an endpoint on an oblique or translated wall can be certified.
class ProjectedSum {
 public:
  void Add(double term) {
    if (!std::isfinite(sum_) || !std::isfinite(term)) {
      sum_ = kInfinity;
      return;
    }
    const auto sum = AddWithResidual(sum_, term);
    sum_ = sum.value;
    AddCorrection(sum.residual);
  }

  void AddDot(Vec3 normal, Vec3 vector) {
    AddScaledProduct(normal.x, vector.x, 1);
    AddScaledProduct(normal.y, vector.y, 1);
    AddScaledProduct(normal.z, vector.z, 1);
  }

  void AddRelativeMotion(Vec3 normal, Vec3 world_velocity, Vec3 wall_velocity, double duration) {
    AddRelativeComponent(normal.x, world_velocity.x, wall_velocity.x, duration);
    AddRelativeComponent(normal.y, world_velocity.y, wall_velocity.y, duration);
    AddRelativeComponent(normal.z, world_velocity.z, wall_velocity.z, duration);
  }

  ProjectionBounds Bounds() const {
    if (!std::isfinite(sum_) || !std::isfinite(lower_correction_) || !std::isfinite(upper_correction_)) {
      return {kInfinity, kInfinity, kInfinity};
    }
    return {
        sum_ + std::midpoint(lower_correction_, upper_correction_),
        AddDown(sum_, lower_correction_),
        AddUp(sum_, upper_correction_),
    };
  }

 private:
  void AddCorrection(double correction) {
    if (!std::isfinite(lower_correction_) || !std::isfinite(upper_correction_)) {
      sum_ = kInfinity;
      return;
    }
    lower_correction_ = AddDown(lower_correction_, correction);
    upper_correction_ = AddUp(upper_correction_, correction);
  }

  void AddProduct(double left, double right) {
    const double product = left * right;
    Add(product);
    if (std::isfinite(product) && left != 0 && right != 0 && std::abs(left) != 1 && std::abs(right) != 1) {
      AddCorrection(std::fma(left, right, -product));
      // A product of two binary64 significands can require 106 bits.
      // Below this binade its residual can itself round in the subnormal range.
      if (std::abs(product) <= 0x1p-969) {
        const double residual_roundoff = std::numeric_limits<double>::denorm_min();
        lower_correction_ = AddDown(lower_correction_, -residual_roundoff);
        upper_correction_ = AddUp(upper_correction_, residual_roundoff);
      }
    }
  }

  void AddScaledProduct(double left, double right, double scale) {
    if (left == 0 || right == 0 || scale == 0) {
      return;
    }
    if (scale == 1) {
      AddProduct(left, right);
      return;
    }
    if (std::abs(left) == 1 || std::abs(right) == 1) {
      AddProduct(left * right, scale);
      return;
    }
    const double product = left * right;
    if (!std::isfinite(product) || std::abs(product) <= 0x1p-969) {
      sum_ = kInfinity;
      return;
    }
    AddProduct(product, scale);
    AddProduct(std::fma(left, right, -product), scale);
  }

  void AddRelativeComponent(double normal, double world_velocity, double wall_velocity, double duration) {
    if (normal == 0 || duration == 0) {
      return;
    }
    const auto relative = AddWithResidual(world_velocity, -wall_velocity);
    if (!std::isfinite(relative.value)) {
      sum_ = kInfinity;
      return;
    }
    AddScaledProduct(normal, relative.value, duration);
    AddScaledProduct(normal, relative.residual, duration);
  }

  double sum_ = 0;
  double lower_correction_ = 0;
  double upper_correction_ = 0;
};

struct SweepPlane {
  Vec3 outward_normal;
  double start_coordinate;
  double slab_half_width;
  Vec3 wall_velocity;
};

ProjectionBounds
DistanceAlongPath(const SweepPlane& plane, Vec3 start_world_position, Vec3 world_velocity, double duration) {
  ProjectedSum distance;
  distance.AddDot(plane.outward_normal, start_world_position);
  distance.Add(-plane.start_coordinate);
  distance.Add(-plane.slab_half_width);
  distance.AddRelativeMotion(plane.outward_normal, world_velocity, plane.wall_velocity, duration);
  return distance.Bounds();
}

ProjectionBounds RelativeNormalVelocity(const SweepPlane& plane, Vec3 world_velocity) {
  ProjectedSum projection;
  projection.AddRelativeMotion(plane.outward_normal, world_velocity, plane.wall_velocity, 1);
  return projection.Bounds();
}

bool HasRepresentablePlanes(ProjectionBounds center, double half_width) {
  if (!center.finite()) {
    return false;
  }
  if (half_width == 0) {
    return true;
  }
  const double low = center.estimate - half_width;
  const double high = center.estimate + half_width;
  return std::isfinite(center.lower - half_width) && std::isfinite(center.upper + half_width) && low < high;
}

absl::Status UncertifiedContact() {
  return absl::OutOfRangeError("Wall contact cannot be certified in the available arithmetic range");
}

// Choose the next representable outward move with the largest normal projection.
Vec3 NudgeOutward(Vec3 vector, Vec3 normal) {
  std::array<double, 3> values{vector.x, vector.y, vector.z};
  const std::array<double, 3> normals{normal.x, normal.y, normal.z};
  std::size_t best_axis = 0;
  double best_value = values[0];
  double best_progress = 0;
  for (std::size_t axis = 0; axis < values.size(); ++axis) {
    if (normals[axis] == 0) {
      continue;
    }
    const double next = std::nextafter(values[axis], std::copysign(kInfinity, normals[axis]));
    const double progress = normals[axis] * (next - values[axis]);
    if (std::isfinite(next) && progress > best_progress) {
      best_axis = axis;
      best_value = next;
      best_progress = progress;
    }
  }
  values[best_axis] = best_value;
  return {values[0], values[1], values[2]};
}

// After a plane projection or constrained remainder, permit at most two
// representable outward moves to repair rounding of three Cartesian components.
template <typename Evaluate>
bool CertifyContactSide(Vec3& candidate, Vec3 outward_normal, Evaluate evaluate) {
  if (!HasFiniteComponents(candidate)) {
    return false;
  }
  for (int nudges = 0; nudges < 2; ++nudges) {
    const auto projection = evaluate(candidate);
    if (!projection.finite()) {
      return false;
    }
    if (projection.lower >= 0) {
      return true;
    }
    candidate = NudgeOutward(candidate, outward_normal);
  }
  const auto projection = evaluate(candidate);
  return projection.finite() && projection.lower >= 0;
}

absl::StatusOr<Vec3> ApplyOneWayResponse(
    const SweepPlane& plane,
    Vec3 world_velocity,
    double relative_normal_projection,
    double normal_length_squared,
    double friction_coefficient
) {
  if (relative_normal_projection >= 0) {
    return world_velocity;
  }
  // Apply the normal impulse in world coordinates. A round trip through a
  // fast wall's frame could erase a small unchanged world tangential velocity.
  Vec3 corrected = Advance(world_velocity, plane.outward_normal, -relative_normal_projection / normal_length_squared);
  const auto rounded_projection = RelativeNormalVelocity(plane, corrected);
  if (!HasFiniteComponents(corrected) || !rounded_projection.finite()) {
    return UncertifiedContact();
  }
  corrected = Advance(corrected, plane.outward_normal, -rounded_projection.estimate / normal_length_squared);
  if (friction_coefficient > 0) {
    const Vec3 relative_tangent{
        corrected.x - plane.wall_velocity.x,
        corrected.y - plane.wall_velocity.y,
        corrected.z - plane.wall_velocity.z,
    };
    const double tangent_speed = std::hypot(relative_tangent.x, relative_tangent.y, relative_tangent.z);
    const double inward_speed = -relative_normal_projection / std::sqrt(normal_length_squared);
    if (!std::isfinite(tangent_speed) || inward_speed == 0) {
      return UncertifiedContact();
    }
    // Check saturation first so a finite coefficient cannot overflow the cap.
    if (tangent_speed == 0 || friction_coefficient >= tangent_speed / inward_speed) {
      corrected = plane.wall_velocity;
    } else {
      const double tangent_speed_reduction = friction_coefficient * inward_speed;
      const double friction_fraction = tangent_speed_reduction / tangent_speed;
      corrected = {
          std::lerp(corrected.x, plane.wall_velocity.x, friction_fraction),
          std::lerp(corrected.y, plane.wall_velocity.y, friction_fraction),
          std::lerp(corrected.z, plane.wall_velocity.z, friction_fraction),
      };
    }
  }
  if (!CertifyContactSide(corrected, plane.outward_normal, [&](Vec3 velocity) {
        return RelativeNormalVelocity(plane, velocity);
      })) {
    return UncertifiedContact();
  }
  return corrected;
}

}  // namespace

absl::StatusOr<LiquidWallContactResult3D> ResolveLiquidWallContact3D(const LiquidWallContactInputs3D& inputs) {
  if (!HasFiniteComponents(inputs.start_world_position) || !HasFiniteComponents(inputs.stored_world_velocity)
      || !HasFiniteComponents(inputs.advecting_world_velocity)) {
    return absl::InvalidArgumentError("Wall contact position and velocities must be finite");
  }
  if (!std::isfinite(inputs.step_duration) || inputs.step_duration <= 0) {
    return absl::InvalidArgumentError("Wall contact step duration must be positive finite");
  }
  if (!std::isfinite(inputs.friction_coefficient) || inputs.friction_coefficient < 0
      || !std::isfinite(inputs.penetration_tolerance) || inputs.penetration_tolerance < 0) {
    return absl::InvalidArgumentError("Wall friction and penetration tolerance must be nonnegative finite");
  }

  LiquidWallContactResult3D result{
      .world_position = inputs.start_world_position,
      .stored_world_velocity = inputs.stored_world_velocity,
      .advecting_world_velocity = inputs.advecting_world_velocity,
  };
  if (std::holds_alternative<std::monostate>(inputs.obstacle)) {
    result.world_position = Advance(inputs.start_world_position, inputs.advecting_world_velocity, inputs.step_duration);
    if (!HasFiniteComponents(result.world_position)) {
      return UncertifiedContact();
    }
    return result;
  }

  SweepPlane plane{};
  if (const auto* halfspace = std::get_if<TranslatingHalfspace3D>(&inputs.obstacle)) {
    plane = {halfspace->unit_normal, halfspace->start_plane_coordinate, 0, halfspace->wall_velocity};
  } else if (const auto* slab = std::get_if<TranslatingSlab3D>(&inputs.obstacle)) {
    if (!std::isfinite(slab->half_width) || slab->half_width <= 0) {
      return absl::InvalidArgumentError("Slab half width must be positive finite");
    }
    plane = {slab->unit_normal, slab->start_center_coordinate, slab->half_width, slab->wall_velocity};
  } else {
    return absl::InvalidArgumentError("Unsupported wall contact obstacle");
  }
  const Vec3 normal = plane.outward_normal;
  const double normal_length_squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
  if (!HasFiniteComponents(normal) || !HasFiniteComponents(plane.wall_velocity)
      || !std::isfinite(plane.start_coordinate) || std::abs(normal_length_squared - 1) > kUnitNormalTolerance) {
    return absl::InvalidArgumentError("Invalid unit-normal obstacle");
  }

  ProjectedSum translated_coordinate;
  translated_coordinate.Add(plane.start_coordinate);
  if (!HasRepresentablePlanes(translated_coordinate.Bounds(), plane.slab_half_width)) {
    return UncertifiedContact();
  }
  translated_coordinate.AddRelativeMotion(normal, plane.wall_velocity, {}, inputs.step_duration);
  if (!HasRepresentablePlanes(translated_coordinate.Bounds(), plane.slab_half_width)) {
    return UncertifiedContact();
  }
  if (plane.slab_half_width > 0) {
    ProjectedSum center_distance;
    center_distance.AddDot(normal, inputs.start_world_position);
    center_distance.Add(-plane.start_coordinate);
    const auto center_projection = center_distance.Bounds();
    if (!center_projection.finite()) {
      return UncertifiedContact();
    }
    const auto side = center_projection.sign();
    if (!side.has_value() || *side == 0) {
      return absl::FailedPreconditionError("The nearest slab side is ambiguous");
    }
    if (*side < 0) {
      plane.outward_normal = {-normal.x, -normal.y, -normal.z};
      plane.start_coordinate = -plane.start_coordinate;
    }
  }

  const auto start_distance = DistanceAlongPath(plane, inputs.start_world_position, {}, 0);
  const auto start_side = start_distance.sign();
  const auto advecting_projection = RelativeNormalVelocity(plane, inputs.advecting_world_velocity);
  const auto advecting_direction = advecting_projection.sign();
  const auto stored_projection = RelativeNormalVelocity(plane, inputs.stored_world_velocity);
  const auto stored_direction = stored_projection.sign();
  if (!start_side.has_value() || !advecting_direction.has_value() || !stored_direction.has_value()) {
    return UncertifiedContact();
  }
  const bool initially_penetrating = *start_side < 0;
  if (initially_penetrating && start_distance.lower < -inputs.penetration_tolerance) {
    return absl::FailedPreconditionError("Initial wall penetration exceeds tolerance");
  }

  // In the wall frame, distance to the incident plane is affine over the step.
  // Crossing that plane is a hit even if the particle ends beyond the other slab plane.
  double elapsed = 0;
  bool hit = initially_penetrating || (*start_side == 0 && (*advecting_direction < 0 || *stored_direction < 0));
  if (*start_side > 0 && *advecting_direction < 0) {
    const auto end_side =
        DistanceAlongPath(plane, inputs.start_world_position, inputs.advecting_world_velocity, inputs.step_duration)
            .sign();
    if (!end_side.has_value()) {
      return UncertifiedContact();
    }
    if (*end_side <= 0) {
      if (*end_side < 0 && !start_distance.magnitude_is_resolved()) {
        return UncertifiedContact();
      }
      hit = true;
      elapsed = *end_side == 0
          ? inputs.step_duration
          : std::min(inputs.step_duration, start_distance.estimate / -advecting_projection.estimate);
      if (elapsed <= 0 || elapsed / inputs.step_duration == 0) {
        return UncertifiedContact();
      }
    }
  }

  const auto final_distance = [&](Vec3 position) {
    return DistanceAlongPath(plane, position, {}, inputs.step_duration);
  };
  if (!hit) {
    result.world_position = Advance(inputs.start_world_position, inputs.advecting_world_velocity, inputs.step_duration);
    const auto accepted_distance = final_distance(result.world_position);
    if (!HasFiniteComponents(result.world_position) || !accepted_distance.finite() || accepted_distance.lower < 0) {
      return UncertifiedContact();
    }
    return result;
  }

  if (!stored_projection.magnitude_is_resolved() || !advecting_projection.magnitude_is_resolved()) {
    return UncertifiedContact();
  }
  const auto stored = ApplyOneWayResponse(
      plane,
      inputs.stored_world_velocity,
      stored_projection.estimate,
      normal_length_squared,
      inputs.friction_coefficient
  );
  if (!stored.ok()) {
    return stored.status();
  }
  const auto advecting = ApplyOneWayResponse(
      plane,
      inputs.advecting_world_velocity,
      advecting_projection.estimate,
      normal_length_squared,
      inputs.friction_coefficient
  );
  if (!advecting.ok()) {
    return advecting.status();
  }

  // Start the constrained remainder at the certified incident face. This
  // projects a tolerable overlap or repairs rounding of the swept hit.
  Vec3 hit_position = Advance(inputs.start_world_position, inputs.advecting_world_velocity, elapsed);
  const auto hit_distance = [&](Vec3 position) {
    return DistanceAlongPath(plane, position, {}, elapsed);
  };
  const auto rounded_hit_distance = hit_distance(hit_position);
  if (!rounded_hit_distance.finite()) {
    return UncertifiedContact();
  }
  hit_position = Advance(hit_position, plane.outward_normal, -rounded_hit_distance.estimate / normal_length_squared);
  if (!CertifyContactSide(hit_position, plane.outward_normal, hit_distance)) {
    return UncertifiedContact();
  }
  result.world_position = Advance(hit_position, *advecting, inputs.step_duration - elapsed);
  if (!CertifyContactSide(result.world_position, plane.outward_normal, final_distance)) {
    return UncertifiedContact();
  }
  result.stored_world_velocity = *stored;
  result.advecting_world_velocity = *advecting;
  result.contact_fraction = elapsed / inputs.step_duration;
  result.outward_normal = plane.outward_normal;
  const double normal_length = std::sqrt(normal_length_squared);
  result.incoming_stored_relative_normal_speed = stored_projection.estimate / normal_length;
  result.incoming_advecting_relative_normal_speed = advecting_projection.estimate / normal_length;
  if (!std::isfinite(result.incoming_stored_relative_normal_speed)
      || !std::isfinite(result.incoming_advecting_relative_normal_speed)) {
    return UncertifiedContact();
  }
  result.diagnostic = initially_penetrating ? LiquidWallContactDiagnostic3D::kInitialPenetration
                                            : LiquidWallContactDiagnostic3D::kSweptImpact;
  return result;
}

}  // namespace fluidity
