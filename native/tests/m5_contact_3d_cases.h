#pragma once

#include "fluidity/base/contact_face_bodies_3d.h"
#include "fluidity/base/distinct_body_contact_3d.h"
#include "m5_constitutive_3d_cases.h"

namespace fluidity::m5_test {

FLUIDITY_HD inline int CheckContactCases() {
  const ContactBodySample3D first{.body_id = 9, .mass = 2, .world_velocity = {-2, 3, 0}};
  const ContactBodySample3D second{.body_id = 11, .mass = 6, .world_velocity = {2, 0, 0}};
  DistinctBodyContact3D result{};
  if (!ResolveDistinctBodyContact3D(first, second, {1, 0, 0}, 0.25, result)
      || !Near(result.incoming_relative_normal_speed, -4) || !Near(result.impulse_on_first.x, 6)
      || !Near(result.impulse_on_first.y, -1.5) || !Near(result.first_world_velocity.x, 1)
      || !Near(result.second_world_velocity.x, 1)
      || !Near(2 * result.first_world_velocity.y + 6 * result.second_world_velocity.y, 6)) {
    return 11;
  }
  const double final_twice_energy = 2 * (1 + result.first_world_velocity.y * result.first_world_velocity.y)
      + 6 * (1 + result.second_world_velocity.y * result.second_world_velocity.y);
  if (final_twice_energy > 50 || !ResolveDistinctBodyContact3D(first, second, {1, 0, 0}, 1, result)
      || !Near(result.first_world_velocity.y, 0.75) || !Near(result.second_world_velocity.y, 0.75)) {
    return 12;
  }
  if (!ResolveDistinctBodyContact3D(first, second, {-1, 0, 0}, 1, result) || !Near(result.impulse_on_first.x, 0)
      || !Near(result.first_world_velocity.x, -2)) {
    return 13;
  }
  if (ResolveDistinctBodyContact3D(first, first, {1, 0, 0}, 1, result)
      || ResolveDistinctBodyContact3D(first, second, {2, 0, 0}, 1, result)
      || ResolveDistinctBodyContact3D(first, second, {1, 0, 0}, -1, result)
      || !Near(result.first_world_velocity.x, -2)) {
    return 14;
  }
  ContactFaceBodies3D face;
  constexpr std::uint64_t kBodyIds[]{12, 4, 10, 2};
  for (const auto body_id : kBodyIds) {
    if (face.Deposit(body_id, 2, {4, 0, 0}) != ContactFaceDepositStatus3D::kAccepted) {
      return 15;
    }
  }
  if (face.Deposit(8, 1, {}) != ContactFaceDepositStatus3D::kBodyCapacity
      || face.Deposit(4, 1, {-1, 0, 0}) != ContactFaceDepositStatus3D::kAccepted
      || face.Deposit(0, 1, {}) != ContactFaceDepositStatus3D::kInvalid || face.size() != 4) {
    return 16;
  }
  ContactBodySample3D sample{};
  if (!face.Read(0, sample) || sample.body_id != 2 || !face.Read(1, sample) || sample.body_id != 4
      || !Near(sample.mass, 3) || !Near(sample.world_velocity.x, 1) || !face.Read(2, sample) || sample.body_id != 10
      || !face.Read(3, sample) || sample.body_id != 12 || face.Read(4, sample) || sample.body_id != 12) {
    return 17;
  }
  if (face.Deposit(2, 1e-320, {1e300, 0, 0}) != ContactFaceDepositStatus3D::kAccepted
      || face.Deposit(2, 1, {1.7e308, 0, 0}) != ContactFaceDepositStatus3D::kAccepted
      || face.Deposit(2, 1, {1.7e308, 0, 0}) != ContactFaceDepositStatus3D::kRange || !face.Read(0, sample)
      || !Near(sample.mass, 3) || sample.world_velocity.x < 5e307) {
    return 18;
  }
  ContactFaceBodies3D interface_face;
  if (interface_face.Deposit(11, 6, {12, 0, 0}) != ContactFaceDepositStatus3D::kAccepted
      || interface_face.Deposit(9, 2, {-4, 6, 0}) != ContactFaceDepositStatus3D::kAccepted
      || interface_face.Deposit(13, 1, {9, 0, 0}) != ContactFaceDepositStatus3D::kAccepted) {
    return 19;
  }
  if (!interface_face.ResolvePair(9, 11, {1, 0, 0}, 0.25, result) || !Near(result.impulse_on_first.x, 6)
      || !Near(result.impulse_on_first.y, -1.5)) {
    return 20;
  }
  ContactBodySample3D resolved_first{};
  ContactBodySample3D resolved_second{};
  if (!interface_face.Read(0, resolved_first) || !interface_face.Read(1, resolved_second)
      || !Near(resolved_first.world_velocity.x, 1) || !Near(resolved_second.world_velocity.x, 1)
      || !Near(
          resolved_first.mass * resolved_first.world_velocity.y
              + resolved_second.mass * resolved_second.world_velocity.y,
          6
      )
      || !interface_face.Read(2, sample) || !Near(sample.world_velocity.x, 9)) {
    return 21;
  }
  if (interface_face.ResolvePair(9, 99, {1, 0, 0}, 0.25, result)
      || interface_face.ResolvePair(9, 11, {0, 0, 0}, 0.25, result) || !Near(result.impulse_on_first.x, 6)
      || !interface_face.Read(0, sample) || !Near(sample.world_velocity.y, resolved_first.world_velocity.y)) {
    return 22;
  }
  if (!interface_face.ResolvePair(11, 9, {-1, 0, 0}, 0.25, result) || !Near(result.impulse_on_first.x, 0)) {
    return 23;
  }
  return 0;
}

}  // namespace fluidity::m5_test
