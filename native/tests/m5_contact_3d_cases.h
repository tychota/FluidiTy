#pragma once

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
  return 0;
}

}  // namespace fluidity::m5_test
