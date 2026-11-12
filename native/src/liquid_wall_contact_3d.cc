#include "fluidity/host/liquid_wall_contact_3d.h"

#include <cmath>

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
