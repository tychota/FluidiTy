#include "fluidity/host/liquid_wall_contact_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>

#include "fluidity/base/liquid_wall_contact_math_3d.h"
#include "liquid_boundary_backend.h"
namespace fluidity {
absl::StatusOr<LiquidWallContactResult3D> ResolveLiquidWallContact3D(const LiquidWallContactInputs3D& inputs) {
  const auto result = boundary_internal::ResolveContact(boundary_internal::PackContact(inputs));
  if (!result.ok()) {
    return boundary_internal::ArithmeticStatus(result.status());
  }
  return boundary_internal::UnpackContact(*result);
}
namespace {

using boundary_internal::ProjectionBounds;
using boundary_internal::SweepPlane;

struct BoxSweepTime {
  ProjectionBounds distance;
  ProjectionBounds speed;
};

ProjectionBounds NegateBounds(ProjectionBounds value) {
  return {-value.estimate, -value.upper, -value.lower};
}

// Compare nonnegative distances divided by positive speeds using bounded
// cross products. Equal exact fractions preserve the canonical face order.
std::optional<int> CompareBoxSweepTimes(BoxSweepTime first, BoxSweepTime second) {
  boundary_internal::ProjectedSum lower;
  lower.AddProduct(first.distance.lower, second.speed.lower);
  lower.AddProduct(-second.distance.upper, first.speed.upper);
  const auto lower_sign = lower.Bounds().sign();
  if (lower_sign.has_value() && *lower_sign > 0) {
    return 1;
  }
  boundary_internal::ProjectedSum upper;
  upper.AddProduct(first.distance.upper, second.speed.upper);
  upper.AddProduct(-second.distance.lower, first.speed.lower);
  const auto upper_sign = upper.Bounds().sign();
  if (upper_sign.has_value() && *upper_sign < 0) {
    return -1;
  }
  if (lower_sign.has_value() && upper_sign.has_value() && *lower_sign == 0 && *upper_sign == 0) {
    return 0;
  }
  return std::nullopt;
}

struct BoxSweepFace {
  SweepPlane plane;
  ProjectionBounds start_distance;
  ProjectionBounds relative_speed;
  int start_side;
  int direction;
};

absl::StatusOr<const SweepPlane*>
SelectBoxImpactFace(std::span<const BoxSweepFace> faces, Vec3 stored_velocity, double duration) {
  const BoxSweepFace* entering_face = nullptr;
  BoxSweepTime entry_time{};
  std::optional<BoxSweepTime> exit_time;
  bool permanently_exterior = false;
  bool starts_on_box = true;
  bool immediately_leaves_box = false;
  const SweepPlane* tangent_face = nullptr;
  for (const auto& face : faces) {
    starts_on_box &= face.start_side <= 0;
    immediately_leaves_box |= face.start_side == 0 && face.direction > 0;
    if (face.direction == 0 && face.start_side == 0) {
      permanently_exterior = true;
      const auto stored_direction = boundary_internal::RelativeNormalVelocity(face.plane, stored_velocity).sign();
      if (!stored_direction.has_value()) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      if (tangent_face == nullptr && *stored_direction < 0) {
        tangent_face = &face.plane;
      }
    } else if (face.start_side > 0 && face.direction >= 0) {
      return nullptr;
    }
  }
  if (permanently_exterior) {
    return starts_on_box && !immediately_leaves_box ? tangent_face : nullptr;
  }
  for (const auto& face : faces) {
    if (face.direction < 0 && face.start_side >= 0) {
      const BoxSweepTime candidate{face.start_distance, NegateBounds(face.relative_speed)};
      const auto order = entering_face == nullptr ? std::optional<int>{1} : CompareBoxSweepTimes(candidate, entry_time);
      if (!order) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      if (*order > 0) {
        entering_face = &face;
        entry_time = candidate;
      }
    } else if (face.direction > 0) {
      const BoxSweepTime candidate{NegateBounds(face.start_distance), face.relative_speed};
      const auto order = exit_time ? CompareBoxSweepTimes(candidate, *exit_time) : std::optional<int>{-1};
      if (!order) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      if (*order < 0) {
        exit_time = candidate;
      }
    }
  }
  if (entering_face == nullptr) {
    return nullptr;
  }
  if (exit_time) {
    const auto order = CompareBoxSweepTimes(entry_time, *exit_time);
    if (!order) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    if (*order >= 0) {
      return nullptr;
    }
  }
  const auto before_end = CompareBoxSweepTimes(entry_time, {{duration, duration, duration}, {1.0, 1.0, 1.0}});
  if (!before_end) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return *before_end <= 0 ? &entering_face->plane : nullptr;
}

}  // namespace

absl::StatusOr<RigidBoxContact3D> ResolveTranslatingBoxContact3D(
    const LiquidWallContactInputs3D& inputs,
    const TranslatingContactBox3D& box,
    double mass
) {
  namespace sweep = boundary_internal;
  if (!std::holds_alternative<std::monostate>(inputs.obstacle) || !sweep::HasFiniteComponents(box.world_low)
      || !sweep::HasFiniteComponents(box.world_high) || !sweep::HasFiniteComponents(box.wall_velocity)
      || !sweep::HasFiniteComponents(inputs.start_world_position)
      || !sweep::HasFiniteComponents(inputs.advecting_world_velocity)
      || !sweep::HasFiniteComponents(inputs.stored_world_velocity) || !std::isfinite(inputs.step_duration)
      || inputs.step_duration <= 0.0 || !std::isfinite(inputs.friction_coefficient) || inputs.friction_coefficient < 0.0
      || !std::isfinite(inputs.penetration_tolerance) || inputs.penetration_tolerance < 0.0 || !std::isfinite(mass)
      || mass <= 0.0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }

  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto end_low = sweep::Advance(box.world_low, box.wall_velocity, inputs.step_duration);
  const auto end_high = sweep::Advance(box.world_high, box.wall_velocity, inputs.step_duration);
  std::array<BoxSweepFace, 6> faces{};
  bool interior_start = true;
  for (std::size_t axis = 0; axis < kComponents.size(); ++axis) {
    const auto component = kComponents[axis];
    if (box.world_low.*component >= box.world_high.*component) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (!std::isfinite(end_low.*component) || !std::isfinite(end_high.*component)
        || end_low.*component >= end_high.*component) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    for (std::size_t side = 0; side < 2; ++side) {
      const double sign = side == 0 ? -1.0 : 1.0;
      Vec3 normal{};
      normal.*component = sign;
      const SweepPlane plane{
          .outward_normal = normal,
          .start_coordinate = sign * ((side == 0 ? box.world_low : box.world_high).*component),
          .slab_half_width = 0.0,
          .wall_velocity = box.wall_velocity,
      };
      const auto start_distance = sweep::DistanceAlongPath(plane, inputs.start_world_position, {}, 0.0);
      const auto start_side = start_distance.sign();
      const auto relative_speed = sweep::RelativeNormalVelocity(plane, inputs.advecting_world_velocity);
      const auto direction = relative_speed.sign();
      if (!start_side.has_value() || !direction.has_value()) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      faces[2 * axis + side] = {plane, start_distance, relative_speed, *start_side, *direction};
      interior_start &= *start_side < 0;
    }
  }
  if (interior_start) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  const auto selected = SelectBoxImpactFace(faces, inputs.stored_world_velocity, inputs.step_duration);
  if (!selected.ok()) {
    return selected.status();
  }

  auto plane_inputs = inputs;
  if (*selected != nullptr) {
    const auto& plane = **selected;
    plane_inputs.obstacle = TranslatingHalfspace3D{plane.outward_normal, plane.start_coordinate, box.wall_velocity};
  }
  Vec3 hit_position{};
  const auto resolved = sweep::ResolveContact(sweep::PackContact(plane_inputs), &hit_position);
  if (!resolved.ok()) {
    return sweep::ArithmeticStatus(resolved.status(), true);
  }
  RigidBoxContact3D result{.contact = sweep::UnpackContact(*resolved), .body_impulse = {}, .body_angular_impulse = {}};
  if (result.contact.contact_fraction.has_value() != (*selected != nullptr)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  if (*selected == nullptr) {
    for (const auto& face : faces) {
      const auto end_distance =
          sweep::DistanceAlongPath(face.plane, result.contact.world_position, {}, inputs.step_duration);
      if (end_distance.finite() && end_distance.lower >= 0.0) {
        return result;
      }
    }
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }

  const auto incoming = inputs.stored_world_velocity;
  const auto outgoing = result.contact.stored_world_velocity;
  result.body_impulse = {
      mass * (incoming.x - outgoing.x),
      mass * (incoming.y - outgoing.y),
      mass * (incoming.z - outgoing.z)
  };
  const Vec3 start_center{
      std::midpoint(box.world_low.x, box.world_high.x),
      std::midpoint(box.world_low.y, box.world_high.y),
      std::midpoint(box.world_low.z, box.world_high.z),
  };
  const auto center_at_hit =
      sweep::Advance(start_center, box.wall_velocity, *result.contact.contact_fraction * inputs.step_duration);
  const Vec3 arm{hit_position.x - center_at_hit.x, hit_position.y - center_at_hit.y, hit_position.z - center_at_hit.z};
  const auto impulse = result.body_impulse;
  result.body_angular_impulse = {
      arm.y * impulse.z - arm.z * impulse.y,
      arm.z * impulse.x - arm.x * impulse.z,
      arm.x * impulse.y - arm.y * impulse.x,
  };
  if (!sweep::HasFiniteComponents(result.body_impulse) || !sweep::HasFiniteComponents(result.body_angular_impulse)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return result;
}

absl::StatusOr<RigidBoxContact3D> ResolveRotatingBoxContact3D(
    const LiquidWallContactInputs3D& inputs,
    const RigidBodyStep3D& drift,
    Vec3 body_half_extent,
    double particle_mass
) {
  namespace sweep = boundary_internal;
  const auto error = [](absl::StatusCode code) {
    return absl::Status(code, {});
  };
  if (!std::holds_alternative<std::monostate>(inputs.obstacle) || inputs.step_duration != drift.duration
      || !sweep::HasFiniteComponents(inputs.stored_world_velocity) || !std::isfinite(inputs.friction_coefficient)
      || inputs.friction_coefficient < 0.0 || !std::isfinite(inputs.penetration_tolerance)
      || inputs.penetration_tolerance < 0.0 || !std::isfinite(particle_mass) || particle_mass <= 0.0) {
    return error(absl::StatusCode::kInvalidArgument);
  }
  const auto hit =
      SweepRigidBoxPoint3D(drift, body_half_extent, inputs.start_world_position, inputs.advecting_world_velocity);
  if (!hit.ok()) {
    return hit.status();
  }
  if (!hit->has_value()) {
    const auto free = ResolveLiquidWallContact3D(inputs);
    if (!free.ok()) {
      return free.status();
    }
    return RigidBoxContact3D{.contact = *free, .body_impulse = {}, .body_angular_impulse = {}};
  }

  const double fraction = std::midpoint((*hit)->fraction_lower, (*hit)->fraction_upper);
  const double elapsed = fraction * drift.duration;
  const auto impact_pose = SampleRigidBodyPose3D(drift, fraction);
  const auto end_pose = SampleRigidBodyPose3D(drift, 1.0);
  if (!impact_pose.ok() || !end_pose.ok()) {
    return error(absl::StatusCode::kOutOfRange);
  }
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto normal_component = kComponents[(*hit)->body_axis];
  const double sign = (*hit)->outward_sign;
  const double face_extent = body_half_extent.*normal_component;
  Vec3 body_normal{};
  body_normal.*normal_component = sign;

  const auto difference = [](Vec3 first, Vec3 second) {
    return Vec3{first.x - second.x, first.y - second.y, first.z - second.z};
  };
  const auto cross = [](Vec3 first, Vec3 second) {
    return Vec3{
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    };
  };
  const auto orientation = impact_pose->orientation;
  const RigidOrientation3D inverse_orientation{
      .w = orientation.w,
      .vector = {-orientation.vector.x, -orientation.vector.y, -orientation.vector.z},
  };
  const auto relative_start = difference(inputs.start_world_position, drift.start.center);
  const auto relative_velocity = difference(inputs.advecting_world_velocity, drift.end.linear_velocity);
  const auto relative_hit = sweep::Advance(relative_start, relative_velocity, elapsed);
  auto body_hit = RotateRigidBodyVector3D(inverse_orientation, relative_hit);
  if (!body_hit.ok()) {
    return body_hit.status();
  }
  (*body_hit).*normal_component = sign * face_extent;
  const auto arm = RotateRigidBodyVector3D(orientation, *body_hit);
  const auto normal = RotateRigidBodyVector3D(orientation, body_normal);
  if (!arm.ok() || !normal.ok()) {
    return error(absl::StatusCode::kOutOfRange);
  }
  const auto wall_velocity = sweep::Advance(drift.end.linear_velocity, cross(drift.end.angular_velocity, *arm), 1.0);
  const SweepPlane impact_plane{
      .outward_normal = *normal,
      .start_coordinate = 0.0,
      .slab_half_width = 0.0,
      .wall_velocity = wall_velocity
  };
  const auto incoming_stored = sweep::RelativeNormalVelocity(impact_plane, inputs.stored_world_velocity);
  const auto incoming_advecting = sweep::RelativeNormalVelocity(impact_plane, inputs.advecting_world_velocity);
  if (!incoming_stored.magnitude_is_resolved() || !incoming_advecting.magnitude_is_resolved()
      || (fraction > 0.0 ? incoming_advecting.upper >= 0.0 : incoming_advecting.lower > 0.0)) {
    return error(absl::StatusCode::kAborted);
  }
  const double normal_squared = normal->x * normal->x + normal->y * normal->y + normal->z * normal->z;
  const auto stored = sweep::ApplyOneWayResponse(
      impact_plane,
      inputs.stored_world_velocity,
      incoming_stored.estimate,
      normal_squared,
      inputs.friction_coefficient
  );
  const auto advecting = sweep::ApplyOneWayResponse(
      impact_plane,
      inputs.advecting_world_velocity,
      incoming_advecting.estimate,
      normal_squared,
      inputs.friction_coefficient
  );
  if (!stored.ok() || !advecting.ok()) {
    return error(absl::StatusCode::kAborted);
  }

  // Carry the corrected relative remainder in the moving face coordinates.
  // Its nonnegative normal displacement keeps the whole remainder exterior.
  auto body_remainder = RotateRigidBodyVector3D(inverse_orientation, difference(*advecting, wall_velocity));
  if (!body_remainder.ok()) {
    return body_remainder.status();
  }
  (*body_remainder).*normal_component = sign * std::max(0.0, sign * ((*body_remainder).*normal_component));
  const auto body_endpoint = sweep::Advance(*body_hit, *body_remainder, drift.duration - elapsed);
  const auto endpoint_arm = RotateRigidBodyVector3D(end_pose->orientation, body_endpoint);
  const auto end_normal = RotateRigidBodyVector3D(end_pose->orientation, body_normal);
  if (!endpoint_arm.ok() || !end_normal.ok()) {
    return error(absl::StatusCode::kOutOfRange);
  }
  auto endpoint = sweep::Advance(end_pose->center, *endpoint_arm, 1.0);
  const SweepPlane end_plane{
      .outward_normal = *end_normal,
      .start_coordinate = face_extent,
      .slab_half_width = 0.0,
      .wall_velocity = {}
  };
  if (!sweep::CertifyContactSide(endpoint, *end_normal, [&](Vec3 point) {
        return sweep::DistanceAlongPath(end_plane, difference(point, end_pose->center), {}, 0.0);
      })) {
    return error(absl::StatusCode::kAborted);
  }

  const auto velocity_change = difference(inputs.stored_world_velocity, *stored);
  const Vec3 impulse{
      particle_mass * velocity_change.x,
      particle_mass * velocity_change.y,
      particle_mass * velocity_change.z
  };
  const auto angular_impulse = cross(*arm, impulse);
  if (!sweep::HasFiniteComponents(impulse) || !sweep::HasFiniteComponents(angular_impulse)) {
    return error(absl::StatusCode::kOutOfRange);
  }
  return RigidBoxContact3D{
      .contact =
          {
              .world_position = endpoint,
              .stored_world_velocity = *stored,
              .advecting_world_velocity = *advecting,
              .contact_fraction = fraction,
              .outward_normal = *normal,
              .incoming_stored_relative_normal_speed = incoming_stored.estimate / std::sqrt(normal_squared),
              .incoming_advecting_relative_normal_speed = incoming_advecting.estimate / std::sqrt(normal_squared),
              .diagnostic = LiquidWallContactDiagnostic3D::kSweptImpact,
          },
      .body_impulse = impulse,
      .body_angular_impulse = angular_impulse,
  };
}
absl::StatusOr<ElasticBladeContact3D> ResolveElasticBladeContact3D(
    const LiquidWallContactInputs3D& inputs,
    double particle_mass,
    std::uint64_t blade_id,
    ElasticBladeSide3D committed_side
) {
  const auto* blade = std::get_if<TranslatingSlab3D>(&inputs.obstacle);
  const bool valid_side =
      committed_side.blade_id == 0 ? committed_side.sign == 0 : committed_side.sign == -1 || committed_side.sign == 1;
  if (blade == nullptr || blade_id == 0 || !valid_side || !std::isfinite(particle_mass) || particle_mass <= 0.0) {
    return absl::InvalidArgumentError("invalid elastic blade contact inputs");
  }
  if (committed_side.blade_id != 0 && committed_side.blade_id != blade_id) {
    return absl::FailedPreconditionError("elastic blade contact supports one persistent blade per particle");
  }
  Vec3 hit{};
  const auto swept = boundary_internal::ResolveContact(boundary_internal::PackContact(inputs), &hit);
  if (!swept.ok()) {
    return boundary_internal::ArithmeticStatus(swept.status());
  }
  const auto contact = boundary_internal::UnpackContact(*swept);
  if (committed_side.blade_id != 0) {
    const double sign = committed_side.sign;
    const auto normal = blade->unit_normal;
    const boundary_internal::SweepPlane incident_plane{
        .outward_normal = {sign * normal.x, sign * normal.y, sign * normal.z},
        .start_coordinate = sign * blade->start_center_coordinate,
        .slab_half_width = blade->half_width,
        .wall_velocity = blade->wall_velocity,
    };
    const auto side = boundary_internal::DistanceAlongPath(incident_plane, inputs.start_world_position, {}, 0.0).sign();
    if (!side.has_value()) {
      return absl::OutOfRangeError("uncertified committed blade side");
    }
    if (*side < 0) {
      return absl::FailedPreconditionError("sweep start disagrees with the committed blade side");
    }
  }
  ElasticBladeContact3D result{
      .contact = contact,
      .side = committed_side,
      .particle_impulse = {},
      .particle_angular_impulse_about_world_origin = {},
      .blade_work = 0.0,
  };
  if (!contact.contact_fraction.has_value()) {
    return result;
  }
  if (contact.diagnostic == LiquidWallContactDiagnostic3D::kInitialPenetration) {
    return absl::FailedPreconditionError("elastic blade side requires a nonpenetrating sweep start");
  }
  const Vec3 normal = contact.outward_normal;
  const double alignment =
      normal.x * blade->unit_normal.x + normal.y * blade->unit_normal.y + normal.z * blade->unit_normal.z;
  const int incident_sign = alignment > 0.0 ? 1 : -1;
  if (committed_side.blade_id != 0 && committed_side.sign != incident_sign) {
    return absl::FailedPreconditionError("blade impact disagrees with the committed side");
  }
  result.side = {.blade_id = blade_id, .sign = incident_sign};
  const Vec3 outgoing = contact.stored_world_velocity;
  const Vec3 incoming = inputs.stored_world_velocity;
  const Vec3 impulse{
      particle_mass * (outgoing.x - incoming.x),
      particle_mass * (outgoing.y - incoming.y),
      particle_mass * (outgoing.z - incoming.z),
  };
  const Vec3 angular_impulse{
      hit.y * impulse.z - hit.z * impulse.y,
      hit.z * impulse.x - hit.x * impulse.z,
      hit.x * impulse.y - hit.y * impulse.x,
  };
  const Vec3 wall_velocity = blade->wall_velocity;
  const double blade_work = impulse.x * wall_velocity.x + impulse.y * wall_velocity.y + impulse.z * wall_velocity.z;
  if (!std::isfinite(blade_work) || !std::isfinite(angular_impulse.x) || !std::isfinite(angular_impulse.y)
      || !std::isfinite(angular_impulse.z)) {
    return absl::OutOfRangeError("nonfinite blade work or angular impulse");
  }
  result.particle_impulse = impulse;
  result.particle_angular_impulse_about_world_origin = angular_impulse;
  result.blade_work = blade_work;
  return result;
}

absl::StatusOr<ElasticBladeSide3D>
ClassifyInitialElasticBladeSide3D(const TranslatingSlab3D& blade, std::uint64_t blade_id, Vec3 world_position) {
  namespace sweep = boundary_internal;
  const Vec3 normal = blade.unit_normal;
  const double normal_length_squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
  if (blade_id == 0 || !sweep::HasFiniteComponents(world_position) || !sweep::HasFiniteComponents(normal)
      || !sweep::HasFiniteComponents(blade.wall_velocity)
      || std::abs(normal_length_squared - 1.0) > sweep::kUnitNormalTolerance
      || !std::isfinite(blade.start_center_coordinate) || !std::isfinite(blade.half_width) || blade.half_width <= 0.0) {
    return absl::InvalidArgumentError("invalid blade or particle position for initial affinity");
  }
  sweep::ProjectedSum center_coordinate;
  center_coordinate.Add(blade.start_center_coordinate);
  if (!sweep::HasRepresentablePlanes(center_coordinate.Bounds(), blade.half_width)) {
    return absl::OutOfRangeError("unrepresentable initial blade planes");
  }
  for (const int side : {-1, 1}) {
    const sweep::SweepPlane plane{
        .outward_normal = {side * normal.x, side * normal.y, side * normal.z},
        .start_coordinate = side * blade.start_center_coordinate,
        .slab_half_width = blade.half_width,
        .wall_velocity = blade.wall_velocity,
    };
    const auto projection = sweep::DistanceAlongPath(plane, world_position, {}, 0.0).sign();
    if (!projection.has_value()) {
      return absl::OutOfRangeError("uncertified initial blade side");
    }
    if (*projection >= 0) {
      return ElasticBladeSide3D{.blade_id = blade_id, .sign = side};
    }
  }
  return absl::FailedPreconditionError("elastic particle starts inside the blade");
}

absl::StatusOr<std::vector<LiquidBoundaryFaceTag3D>>
BuildElasticBladeFaceTags3D(SparseMacGridView3D mac_grid, const TranslatingSlab3D& blade, double elapsed_time) {
  namespace sweep = boundary_internal;
  const Vec3 normal = blade.unit_normal;
  const double normal_length_squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
  if (!sweep::HasFiniteComponents(normal) || !sweep::HasFiniteComponents(blade.wall_velocity)
      || std::abs(normal_length_squared - 1.0) > sweep::kUnitNormalTolerance
      || !std::isfinite(blade.start_center_coordinate) || !std::isfinite(blade.half_width) || blade.half_width <= 0.0
      || !std::isfinite(elapsed_time) || elapsed_time < 0.0) {
    return absl::InvalidArgumentError("invalid blade pose or time for face classification");
  }

  sweep::ProjectedSum center_coordinate;
  center_coordinate.Add(blade.start_center_coordinate);
  center_coordinate.AddRelativeMotion(normal, blade.wall_velocity, {}, elapsed_time);
  if (!sweep::HasRepresentablePlanes(center_coordinate.Bounds(), blade.half_width)) {
    return absl::OutOfRangeError("unrepresentable blade planes at face sample time");
  }
  const sweep::SweepPlane positive_plane{
      .outward_normal = normal,
      .start_coordinate = blade.start_center_coordinate,
      .slab_half_width = blade.half_width,
      .wall_velocity = blade.wall_velocity,
  };
  const sweep::SweepPlane negative_plane{
      .outward_normal = {-normal.x, -normal.y, -normal.z},
      .start_coordinate = -blade.start_center_coordinate,
      .slab_half_width = blade.half_width,
      .wall_velocity = blade.wall_velocity,
  };

  std::vector<LiquidBoundaryFaceTag3D> tags(mac_grid.face_count());
  for (std::size_t face_slot = 0; face_slot < tags.size(); ++face_slot) {
    const auto face = mac_grid.FaceIndex(face_slot);
    if (!face.ok()) {
      return face.status();
    }
    const auto anchor = mac_grid.FaceWorldAnchor(*face);
    if (!anchor.ok()) {
      return anchor.status();
    }
    const auto positive_side = sweep::DistanceAlongPath(positive_plane, *anchor, {}, elapsed_time).sign();
    const auto negative_side = sweep::DistanceAlongPath(negative_plane, *anchor, {}, elapsed_time).sign();
    if (!positive_side.has_value() || !negative_side.has_value()) {
      return absl::OutOfRangeError("ambiguous face projection at blade");
    }
    if (*positive_side >= 0) {
      tags[face_slot] = {.side_tag = 1, .structurally_eligible = true};
    } else if (*negative_side >= 0) {
      tags[face_slot] = {.side_tag = -1, .structurally_eligible = true};
    } else {
      tags[face_slot] = {.side_tag = 0, .structurally_eligible = false};
    }
  }
  return tags;
}

}  // namespace fluidity
