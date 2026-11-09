#pragma once
#include "fluidity/base/liquid_boundary_math.h"
namespace fluidity::boundary_internal {
constexpr double kUnitNormalTolerance = 1e-12;
constexpr double kInfinity = numeric::numeric_limits<double>::infinity();

FLUIDITY_HD inline bool HasFiniteComponents(Vec3 vector) {
  return numeric::isfinite(vector.x) && numeric::isfinite(vector.y) && numeric::isfinite(vector.z);
}

FLUIDITY_HD inline Vec3 Advance(Vec3 position, Vec3 velocity, double duration) {
  return {
      numeric::fma(velocity.x, duration, position.x),
      numeric::fma(velocity.y, duration, position.y),
      numeric::fma(velocity.z, duration, position.z),
  };
}

FLUIDITY_HD inline double AddDown(double left, double right) {
  const auto sum = AddWithResidual(left, right);
  return sum.residual < 0 ? numeric::nextafter(sum.value, -kInfinity) : sum.value;
}

FLUIDITY_HD inline double AddUp(double left, double right) {
  const auto sum = AddWithResidual(left, right);
  return sum.residual > 0 ? numeric::nextafter(sum.value, kInfinity) : sum.value;
}

struct ProjectionBounds {
  double estimate;
  double lower;
  double upper;

  FLUIDITY_HD bool finite() const {
    return numeric::isfinite(estimate) && numeric::isfinite(lower) && numeric::isfinite(upper);
  }

  // A certified sign suffices to classify an event. Computing the hit time
  // or impulse also needs a resolved magnitude after cancellation.
  FLUIDITY_HD bool magnitude_is_resolved() const {
    constexpr double kRelativeResolution = 16 * numeric::numeric_limits<double>::epsilon();
    return finite() && upper - lower <= kRelativeResolution * numeric::abs(estimate);
  }

  FLUIDITY_HD Maybe<int> sign() const {
    if (!finite()) {
      return {};
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
    return {};
  }
};

// Binary64 arithmetic in round-to-nearest. Retain sum and product residuals
// separately so an endpoint on an oblique or translated wall can be certified.
class ProjectedSum {
 public:
  FLUIDITY_HD void Add(double term) {
    if (!numeric::isfinite(sum_) || !numeric::isfinite(term)) {
      sum_ = kInfinity;
      return;
    }
    const auto sum = AddWithResidual(sum_, term);
    sum_ = sum.value;
    AddCorrection(sum.residual);
  }

  FLUIDITY_HD void AddDot(Vec3 normal, Vec3 vector) {
    AddScaledProduct(normal.x, vector.x, 1);
    AddScaledProduct(normal.y, vector.y, 1);
    AddScaledProduct(normal.z, vector.z, 1);
  }

  FLUIDITY_HD void AddRelativeMotion(Vec3 normal, Vec3 world_velocity, Vec3 wall_velocity, double duration) {
    AddRelativeComponent(normal.x, world_velocity.x, wall_velocity.x, duration);
    AddRelativeComponent(normal.y, world_velocity.y, wall_velocity.y, duration);
    AddRelativeComponent(normal.z, world_velocity.z, wall_velocity.z, duration);
  }

  FLUIDITY_HD ProjectionBounds Bounds() const {
    if (!numeric::isfinite(sum_) || !numeric::isfinite(lower_correction_) || !numeric::isfinite(upper_correction_)) {
      return {kInfinity, kInfinity, kInfinity};
    }
    return {
        sum_ + numeric::midpoint(lower_correction_, upper_correction_),
        AddDown(sum_, lower_correction_),
        AddUp(sum_, upper_correction_),
    };
  }

 private:
  FLUIDITY_HD void AddCorrection(double correction) {
    if (!numeric::isfinite(lower_correction_) || !numeric::isfinite(upper_correction_)) {
      sum_ = kInfinity;
      return;
    }
    lower_correction_ = AddDown(lower_correction_, correction);
    upper_correction_ = AddUp(upper_correction_, correction);
  }

  FLUIDITY_HD void AddProduct(double left, double right) {
    const double product = left * right;
    Add(product);
    if (numeric::isfinite(product) && left != 0 && right != 0 && numeric::abs(left) != 1 && numeric::abs(right) != 1) {
      AddCorrection(numeric::fma(left, right, -product));
      // A product of two binary64 significands can require 106 bits.
      // Below this binade its residual can itself round in the subnormal range.
      if (numeric::abs(product) <= 0x1p-969) {
        const double residual_roundoff = numeric::numeric_limits<double>::denorm_min();
        lower_correction_ = AddDown(lower_correction_, -residual_roundoff);
        upper_correction_ = AddUp(upper_correction_, residual_roundoff);
      }
    }
  }

  FLUIDITY_HD void AddScaledProduct(double left, double right, double scale) {
    if (left == 0 || right == 0 || scale == 0) {
      return;
    }
    if (scale == 1) {
      AddProduct(left, right);
      return;
    }
    if (numeric::abs(left) == 1 || numeric::abs(right) == 1) {
      AddProduct(left * right, scale);
      return;
    }
    const double product = left * right;
    if (!numeric::isfinite(product) || numeric::abs(product) <= 0x1p-969) {
      sum_ = kInfinity;
      return;
    }
    AddProduct(product, scale);
    AddProduct(numeric::fma(left, right, -product), scale);
  }

  FLUIDITY_HD void AddRelativeComponent(double normal, double world_velocity, double wall_velocity, double duration) {
    if (normal == 0 || duration == 0) {
      return;
    }
    const auto relative = AddWithResidual(world_velocity, -wall_velocity);
    if (!numeric::isfinite(relative.value)) {
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

FLUIDITY_HD inline ProjectionBounds
DistanceAlongPath(const SweepPlane& plane, Vec3 start_world_position, Vec3 world_velocity, double duration) {
  ProjectedSum distance;
  distance.AddDot(plane.outward_normal, start_world_position);
  distance.Add(-plane.start_coordinate);
  distance.Add(-plane.slab_half_width);
  distance.AddRelativeMotion(plane.outward_normal, world_velocity, plane.wall_velocity, duration);
  return distance.Bounds();
}

FLUIDITY_HD inline ProjectionBounds RelativeNormalVelocity(const SweepPlane& plane, Vec3 world_velocity) {
  ProjectedSum projection;
  projection.AddRelativeMotion(plane.outward_normal, world_velocity, plane.wall_velocity, 1);
  return projection.Bounds();
}

FLUIDITY_HD inline bool HasRepresentablePlanes(ProjectionBounds center, double half_width) {
  if (!center.finite()) {
    return false;
  }
  if (half_width == 0) {
    return true;
  }
  const double low = center.estimate - half_width;
  const double high = center.estimate + half_width;
  return numeric::isfinite(center.lower - half_width) && numeric::isfinite(center.upper + half_width) && low < high;
}

FLUIDITY_HD inline Error UncertifiedContact() {
  return Error::kRange;
}

// Choose the next representable outward move with the largest normal projection.
FLUIDITY_HD inline Vec3 NudgeOutward(Vec3 vector, Vec3 normal) {
  numeric::array<double, 3> values{vector.x, vector.y, vector.z};
  const numeric::array<double, 3> normals{normal.x, normal.y, normal.z};
  numeric::size_t best_axis = 0;
  double best_value = values[0];
  double best_progress = 0;
  for (numeric::size_t axis = 0; axis < values.size(); ++axis) {
    if (normals[axis] == 0) {
      continue;
    }
    const double next = numeric::nextafter(values[axis], numeric::copysign(kInfinity, normals[axis]));
    const double progress = normals[axis] * (next - values[axis]);
    if (numeric::isfinite(next) && progress > best_progress) {
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
FLUIDITY_HD inline bool CertifyContactSide(Vec3& candidate, Vec3 outward_normal, Evaluate evaluate) {
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

FLUIDITY_HD inline Value<Vec3> ApplyOneWayResponse(
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
    const double tangent_speed = numeric::hypot(relative_tangent.x, relative_tangent.y, relative_tangent.z);
    const double inward_speed = -relative_normal_projection / numeric::sqrt(normal_length_squared);
    if (!numeric::isfinite(tangent_speed) || inward_speed == 0) {
      return UncertifiedContact();
    }
    // Check saturation first so a finite coefficient cannot overflow the cap.
    if (tangent_speed == 0 || friction_coefficient >= tangent_speed / inward_speed) {
      corrected = plane.wall_velocity;
    } else {
      const double tangent_speed_reduction = friction_coefficient * inward_speed;
      const double friction_fraction = tangent_speed_reduction / tangent_speed;
      corrected = {
          numeric::lerp(corrected.x, plane.wall_velocity.x, friction_fraction),
          numeric::lerp(corrected.y, plane.wall_velocity.y, friction_fraction),
          numeric::lerp(corrected.z, plane.wall_velocity.z, friction_fraction),
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

// The zero-friction tank constraints are separable in the three Cartesian
// components. Use the existing certified plane arithmetic on one component at
// a time; processing x, y, z preserves all simultaneous orthogonal impulses.
FLUIDITY_HD inline Error SweepTankAxis(
    const ContactInput& inputs,
    const SweepPlane* walls,
    int wall_count,
    int axis,
    double& first_hit_time,
    ContactOutput& result
) {
  constexpr double Vec3::* kComponents[]{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto component = kComponents[axis];
  Vec3 start{inputs.start_world_position.*component, 0, 0};
  Vec3 stored{inputs.stored_world_velocity.*component, 0, 0};
  Vec3 advecting{inputs.advecting_world_velocity.*component, 0, 0};
  const SweepPlane* carrying_wall = nullptr;

  const auto apply_impact = [&](const SweepPlane& wall, double hit_time, int diagnostic) {
    const auto stored_speed = RelativeNormalVelocity(wall, stored);
    const auto advecting_speed = RelativeNormalVelocity(wall, advecting);
    if (!stored_speed.magnitude_is_resolved() || !advecting_speed.magnitude_is_resolved()) {
      return UncertifiedContact();
    }
    if (hit_time < first_hit_time) {
      first_hit_time = hit_time;
      result.contact_fraction = hit_time / inputs.step_duration;
      result.outward_normal = {};
      result.outward_normal.*component = wall.outward_normal.x;
      result.incoming_stored_relative_normal_speed = stored_speed.estimate;
      result.incoming_advecting_relative_normal_speed = advecting_speed.estimate;
      result.diagnostic = diagnostic;
    }
    if (stored_speed.estimate < 0) {
      stored.x = wall.wall_velocity.x;
    }
    if (advecting_speed.estimate <= 0) {
      advecting.x = wall.wall_velocity.x;
      carrying_wall = &wall;
    }
    return Error::kNone;
  };

  // Repair a tolerable initial overlap before computing the swept path. A
  // touching wall can also constrain stored velocity while advection departs.
  for (int wall_index = 0; wall_index < wall_count; ++wall_index) {
    const auto& wall = walls[wall_index];
    const auto start_distance = DistanceAlongPath(wall, start, {}, 0);
    const auto start_side = start_distance.sign();
    if (!start_side.has_value()) {
      return UncertifiedContact();
    }
    if (*start_side > 0) {
      continue;
    }
    if (start_distance.lower < -inputs.penetration_tolerance) {
      return Error::kPrecondition;
    }
    const double wall_coordinate = wall.start_coordinate * wall.outward_normal.x;
    const bool low_wall = wall.outward_normal.x > 0;
    const bool stored_inward = low_wall ? stored.x < wall.wall_velocity.x : stored.x > wall.wall_velocity.x;
    const bool advecting_inward = low_wall ? advecting.x < wall.wall_velocity.x : advecting.x > wall.wall_velocity.x;
    if (*start_side < 0 || stored_inward || advecting_inward) {
      const auto error = apply_impact(wall, 0, *start_side < 0 ? 2 : 1);
      if (error != Error::kNone) {
        return error;
      }
      start.x = wall_coordinate;
    }
  }

  // Once advection is carried by either wall, positive tank width keeps it
  // from reaching the opposite wall. Otherwise the path can hit at most one.
  if (carrying_wall == nullptr) {
    for (int wall_index = 0; wall_index < wall_count; ++wall_index) {
      const auto& wall = walls[wall_index];
      const auto advecting_speed = RelativeNormalVelocity(wall, advecting);
      const auto direction = advecting_speed.sign();
      if (!direction.has_value()) {
        return UncertifiedContact();
      }
      if (*direction >= 0) {
        continue;
      }
      const auto end_side = DistanceAlongPath(wall, start, advecting, inputs.step_duration).sign();
      if (!end_side.has_value()) {
        return UncertifiedContact();
      }
      if (*end_side > 0) {
        continue;
      }
      double hit_time = inputs.step_duration;
      if (*end_side < 0) {
        const auto start_distance = DistanceAlongPath(wall, start, {}, 0);
        if (!start_distance.magnitude_is_resolved() || !advecting_speed.magnitude_is_resolved()) {
          return UncertifiedContact();
        }
        hit_time = numeric::min(inputs.step_duration, start_distance.estimate / -advecting_speed.estimate);
        if (hit_time <= 0 || hit_time / inputs.step_duration == 0) {
          return UncertifiedContact();
        }
      }
      const auto error = apply_impact(wall, hit_time, 1);
      if (error != Error::kNone) {
        return error;
      }
      break;
    }
  }

  Vec3 end{};
  if (carrying_wall == nullptr) {
    end = Advance(start, advecting, inputs.step_duration);
  } else {
    const auto& wall = *carrying_wall;
    const Vec3 wall_start{wall.start_coordinate * wall.outward_normal.x, 0, 0};
    end = Advance(wall_start, wall.wall_velocity, inputs.step_duration);
    if (!CertifyContactSide(end, wall.outward_normal, [&](Vec3 position) {
          return DistanceAlongPath(wall, position, {}, inputs.step_duration);
        })) {
      return UncertifiedContact();
    }
  }
  for (int wall_index = 0; wall_index < wall_count; ++wall_index) {
    const auto end_distance = DistanceAlongPath(walls[wall_index], end, {}, inputs.step_duration);
    if (!end_distance.finite() || end_distance.lower < 0) {
      return UncertifiedContact();
    }
  }
  result.world_position.*component = end.x;
  result.stored_world_velocity.*component = stored.x;
  result.advecting_world_velocity.*component = advecting.x;
  return Error::kNone;
}

FLUIDITY_HD inline Value<ContactOutput> ResolveTankContact(const ContactInput& inputs) {
  if (inputs.friction_coefficient != 0) {
    return Error::kInvalid;
  }
  const auto& tank = inputs.obstacle.tank;
  const auto step_status = CheckTankStep(tank, inputs.step_duration);
  if (step_status != Error::kNone) {
    return step_status;
  }
  // Each axis is represented along local x. Low walls precede high walls;
  // the floor is the only y constraint. Strict comparison retains this order
  // in the first-hit diagnostics even when another axis hits at the same time.
  const SweepPlane walls[3][2]{
      {{{1, 0, 0}, tank.start_low_x, 0, {tank.piston_velocity_x, 0, 0}}, {{-1, 0, 0}, -tank.high_x, 0, {}}},
      {{{1, 0, 0}, tank.floor_y, 0, {}}, {}},
      {{{1, 0, 0}, tank.low_z, 0, {}}, {{-1, 0, 0}, -tank.high_z, 0, {}}},
  };
  constexpr int kWallCounts[]{2, 1, 2};
  ContactOutput result;
  double first_hit_time = kInfinity;
  for (int axis = 0; axis < 3; ++axis) {
    const auto status = SweepTankAxis(inputs, walls[axis], kWallCounts[axis], axis, first_hit_time, result);
    if (status != Error::kNone) {
      return status;
    }
  }
  return result;
}

FLUIDITY_HD inline Value<ContactOutput> ResolveContact(const ContactInput& inputs) {
  if (!HasFiniteComponents(inputs.start_world_position) || !HasFiniteComponents(inputs.stored_world_velocity)
      || !HasFiniteComponents(inputs.advecting_world_velocity)) {
    return Error::kInvalid;
  }
  if (!numeric::isfinite(inputs.step_duration) || inputs.step_duration <= 0) {
    return Error::kInvalid;
  }
  if (!numeric::isfinite(inputs.friction_coefficient) || inputs.friction_coefficient < 0
      || !numeric::isfinite(inputs.penetration_tolerance) || inputs.penetration_tolerance < 0) {
    return Error::kInvalid;
  }

  ContactOutput result{
      .world_position = inputs.start_world_position,
      .stored_world_velocity = inputs.stored_world_velocity,
      .advecting_world_velocity = inputs.advecting_world_velocity,
  };
  if (inputs.obstacle.kind == ObstacleKind::kOpenTank) {
    return ResolveTankContact(inputs);
  }
  if (inputs.obstacle.kind == ObstacleKind::kAbsent) {
    result.world_position = Advance(inputs.start_world_position, inputs.advecting_world_velocity, inputs.step_duration);
    if (!HasFiniteComponents(result.world_position)) {
      return UncertifiedContact();
    }
    return result;
  }

  const auto& obstacle = inputs.obstacle;
  if (obstacle.kind != ObstacleKind::kHalfspace && obstacle.kind != ObstacleKind::kSlab) {
    return Error::kInvalid;
  }
  if (obstacle.kind == ObstacleKind::kSlab && (!numeric::isfinite(obstacle.half_width) || obstacle.half_width <= 0)) {
    return Error::kInvalid;
  }
  SweepPlane plane{
      obstacle.normal,
      obstacle.coordinate,
      obstacle.kind == ObstacleKind::kSlab ? obstacle.half_width : 0,
      obstacle.velocity,
  };
  const Vec3 normal = plane.outward_normal;
  const double normal_length_squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
  if (!HasFiniteComponents(normal) || !HasFiniteComponents(plane.wall_velocity)
      || !numeric::isfinite(plane.start_coordinate) || numeric::abs(normal_length_squared - 1) > kUnitNormalTolerance) {
    return Error::kInvalid;
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
      return Error::kPrecondition;
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
    return Error::kPrecondition;
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
          : numeric::min(inputs.step_duration, start_distance.estimate / -advecting_projection.estimate);
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
  const double normal_length = numeric::sqrt(normal_length_squared);
  result.incoming_stored_relative_normal_speed = stored_projection.estimate / normal_length;
  result.incoming_advecting_relative_normal_speed = advecting_projection.estimate / normal_length;
  if (!numeric::isfinite(result.incoming_stored_relative_normal_speed)
      || !numeric::isfinite(result.incoming_advecting_relative_normal_speed)) {
    return UncertifiedContact();
  }
  result.diagnostic = initially_penetrating ? 2 : 1;
  return result;
}

}  // namespace fluidity::boundary_internal
