#pragma once

#include <cstdint>

#include "fluidity/base/host_device.h"
#include "fluidity/base/liquid_boundary_math.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

struct ContactBodySample3D {
  std::uint64_t body_id;  // Stable nonzero identity, independent of cut side.
  double mass;  // Positive effective sample mass in kg.
  Vec3 world_velocity;  // Incoming velocity in m/s.
};

struct DistinctBodyContact3D {
  Vec3 first_world_velocity;  // Outgoing velocity in m/s.
  Vec3 second_world_velocity;  // Outgoing velocity in m/s.
  Vec3 impulse_on_first;  // N s. The second sample receives its exact negation.
  double incoming_relative_normal_speed;  // m/s, negative for approaching samples.
};

namespace distinct_contact_detail {

namespace numeric = boundary_internal::numeric;

FLUIDITY_HD inline bool Finite(Vec3 vector) noexcept {
  return numeric::isfinite(vector.x) && numeric::isfinite(vector.y) && numeric::isfinite(vector.z);
}

FLUIDITY_HD inline double Dot(Vec3 lhs, Vec3 rhs) noexcept {
  return (lhs.x * rhs.x + lhs.y * rhs.y) + lhs.z * rhs.z;
}

FLUIDITY_HD inline Vec3 AddScaled(Vec3 value, Vec3 increment, double scale) noexcept {
  return {value.x + increment.x * scale, value.y + increment.y * scale, value.z + increment.z * scale};
}

}  // namespace distinct_contact_detail

// Resolves one already-detected contact between two distinct finite-mass samples in world SI units.
// normal_toward_first points from the second body toward the first and must be unit length to 1e-12
// in squared length. Uses zero normal restitution and nonnegative Coulomb friction. Separating
// samples retain their velocities. Returns false on invalid or unrepresentable inputs and leaves
// output unchanged. The caller owns detection, multi-contact ordering and accepted-step publication.
FLUIDITY_HD inline bool ResolveDistinctBodyContact3D(
    ContactBodySample3D first,
    ContactBodySample3D second,
    Vec3 normal_toward_first,
    double friction_coefficient,
    DistinctBodyContact3D& output
) noexcept {
  namespace detail = distinct_contact_detail;
  namespace numeric = detail::numeric;
  if (first.body_id == 0 || second.body_id == 0 || first.body_id == second.body_id) {
    return false;
  }
  if (!numeric::isfinite(first.mass) || first.mass <= 0 || !numeric::isfinite(second.mass) || second.mass <= 0) {
    return false;
  }
  if (!detail::Finite(first.world_velocity) || !detail::Finite(second.world_velocity)
      || !detail::Finite(normal_toward_first) || !numeric::isfinite(friction_coefficient) || friction_coefficient < 0) {
    return false;
  }
  const double normal_length_squared = detail::Dot(normal_toward_first, normal_toward_first);
  if (numeric::abs(normal_length_squared - 1) > 1e-12) {
    return false;
  }
  const Vec3 normal = detail::AddScaled({}, normal_toward_first, 1 / numeric::sqrt(normal_length_squared));
  const Vec3 relative_velocity = detail::AddScaled(first.world_velocity, second.world_velocity, -1);
  const double incoming_normal_speed = detail::Dot(relative_velocity, normal);
  if (!numeric::isfinite(incoming_normal_speed)) {
    return false;
  }

  DistinctBodyContact3D candidate{
      .first_world_velocity = first.world_velocity,
      .second_world_velocity = second.world_velocity,
      .impulse_on_first = {},
      .incoming_relative_normal_speed = incoming_normal_speed,
  };
  if (incoming_normal_speed < 0) {
    const double smaller_mass = numeric::min(first.mass, second.mass);
    const double larger_mass = numeric::max(first.mass, second.mass);
    const double effective_mass = smaller_mass / (1 + smaller_mass / larger_mass);
    const double normal_impulse = -incoming_normal_speed * effective_mass;
    if (!numeric::isfinite(normal_impulse) || normal_impulse <= 0) {
      return false;
    }
    const Vec3 tangent_velocity = detail::AddScaled(relative_velocity, normal, -incoming_normal_speed);
    const double tangent_speed = numeric::hypot(tangent_velocity.x, tangent_velocity.y, tangent_velocity.z);
    if (!numeric::isfinite(tangent_speed)) {
      return false;
    }

    candidate.impulse_on_first = detail::AddScaled({}, normal, normal_impulse);
    if (tangent_speed > 0 && friction_coefficient > 0) {
      const double tangent_impulse =
          numeric::min(tangent_speed * effective_mass, friction_coefficient * normal_impulse);
      const Vec3 tangent{
          tangent_velocity.x / tangent_speed,
          tangent_velocity.y / tangent_speed,
          tangent_velocity.z / tangent_speed,
      };
      candidate.impulse_on_first = detail::AddScaled(candidate.impulse_on_first, tangent, -tangent_impulse);
    }
    const Vec3 impulse = candidate.impulse_on_first;
    const Vec3 first_velocity_change{impulse.x / first.mass, impulse.y / first.mass, impulse.z / first.mass};
    const Vec3 second_velocity_change{impulse.x / second.mass, impulse.y / second.mass, impulse.z / second.mass};
    candidate.first_world_velocity = detail::AddScaled(first.world_velocity, first_velocity_change, 1);
    candidate.second_world_velocity = detail::AddScaled(second.world_velocity, second_velocity_change, -1);
    if (!detail::Finite(impulse) || !detail::Finite(candidate.first_world_velocity)
        || !detail::Finite(candidate.second_world_velocity)) {
      return false;
    }
  }
  output = candidate;
  return true;
}

}  // namespace fluidity
