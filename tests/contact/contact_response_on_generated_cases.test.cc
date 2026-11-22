#include <gtest/gtest.h>

#include <cmath>

#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/testing/contact_cases.h"

namespace fluidity::contact {
namespace {

TEST(ComputeContactResponse, GeneratedCases_ApproachLeftWithinBound) {
  // Draw order per case: a normal of three NextSigned() values, drawn again while its length is below 0.25, then
  // normalised; a velocity of 10 times three NextSigned() values; a solid speed scale of 2^k m/s with
  // k = floor(37 * Next()) - 10, from 2^-10 (about 1e-3) to 2^26 (about 7e7), exact on every platform; a solid
  // velocity of that scale times three NextSigned() values; friction 2 * Next().
  CaseGenerator generator;
  int approaching_count = 0;
  int nonzero_left_count = 0;
  for (int case_index = 0; case_index < 1000; ++case_index) {
    Vec3 normal{};
    do {
      normal = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (math::Length(normal) < 0.25);
    normal = math::Normalized(normal);
    const Vec3 direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const Vec3 velocity = 10 * direction;
    const int scale_exponent = static_cast<int>(37 * generator.Next()) - 10;
    const double solid_speed_scale = std::ldexp(1.0, scale_exponent);
    const Vec3 solid_direction = {
        .x = generator.NextSigned(),
        .y = generator.NextSigned(),
        .z = generator.NextSigned(),
    };
    const Vec3 solid_velocity = solid_speed_scale * solid_direction;
    const double friction = 2 * generator.Next();
    if (math::Dot(normal, velocity - solid_velocity) >= 0) {
      continue;
    }
    ++approaching_count;

    const Vec3 after = ComputeContactResponse(velocity, solid_velocity, normal, friction);
    const double approach_left = math::Dot(normal, after - solid_velocity);
    if (approach_left != 0) {
      ++nonzero_left_count;
    }
    EXPECT_LE(math::Abs(approach_left), ComputeLargestApproachLeftAfterResponse(velocity, solid_velocity))
        << "case " << case_index;
  }
  // About half the cases approach, and rounding leaves a non-zero approach in many of them, so the bound is tested.
  EXPECT_GT(approaching_count, 400);
  EXPECT_GT(nonzero_left_count, 50);
}

}  // namespace
}  // namespace fluidity::contact
