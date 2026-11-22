#include "fluidity/base/per_particle_velocity.h"

#include <gtest/gtest.h>

#include <type_traits>

#include "fluidity/base/vec3.h"

namespace fluidity {
namespace {

static_assert(std::is_trivially_copyable_v<PerParticleVelocity<Vec3>>);

TEST(PerParticleVelocity, AggregateInitialisation_KeepsBothFieldsApart) {
  const PerParticleVelocity<Vec3> velocity = {
      .for_next_step = {.x = 1.0, .y = 2.0, .z = 3.0},
      .for_this_step = {.x = 4.0, .y = 5.0, .z = 6.0},
  };
  EXPECT_EQ(velocity.for_next_step.x, 1.0);
  EXPECT_EQ(velocity.for_next_step.z, 3.0);
  EXPECT_EQ(velocity.for_this_step.x, 4.0);
  EXPECT_EQ(velocity.for_this_step.z, 6.0);
}

TEST(PerParticleVelocity, OfDoubles_KeepsBothFieldsApart) {
  const PerParticleVelocity<double> speed = {.for_next_step = -0.5, .for_this_step = 0.25};
  EXPECT_EQ(speed.for_next_step, -0.5);
  EXPECT_EQ(speed.for_this_step, 0.25);
}

}  // namespace
}  // namespace fluidity
