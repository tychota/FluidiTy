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
  const auto contact = ResolveLiquidWallContact3D(inputs);
  if (!contact.ok()) {
    return contact.status();
  }
  ElasticBladeContact3D result{
      .contact = *contact,
      .side = committed_side,
      .particle_impulse = {},
      .particle_angular_impulse_about_world_origin = {},
      .blade_work = 0.0,
  };
  if (!contact->contact_fraction.has_value()) {
    return result;
  }
  if (contact->diagnostic == LiquidWallContactDiagnostic3D::kInitialPenetration) {
    return absl::FailedPreconditionError("elastic blade side requires a nonpenetrating sweep start");
  }
  const Vec3 normal = contact->outward_normal;
  const double alignment =
      normal.x * blade->unit_normal.x + normal.y * blade->unit_normal.y + normal.z * blade->unit_normal.z;
  const int incident_sign = alignment > 0.0 ? 1 : -1;
  if (committed_side.blade_id != 0 && committed_side.sign != incident_sign) {
    return absl::FailedPreconditionError("blade impact disagrees with the committed side");
  }
  result.side = {.blade_id = blade_id, .sign = incident_sign};
  const Vec3 outgoing = contact->stored_world_velocity;
  const Vec3 incoming = inputs.stored_world_velocity;
  const Vec3 impulse{
      particle_mass * (outgoing.x - incoming.x),
      particle_mass * (outgoing.y - incoming.y),
      particle_mass * (outgoing.z - incoming.z),
  };
  const double hit_time = *contact->contact_fraction * inputs.step_duration;
  const Vec3 hit{
      std::fma(hit_time, inputs.advecting_world_velocity.x, inputs.start_world_position.x),
      std::fma(hit_time, inputs.advecting_world_velocity.y, inputs.start_world_position.y),
      std::fma(hit_time, inputs.advecting_world_velocity.z, inputs.start_world_position.z),
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

}  // namespace fluidity
