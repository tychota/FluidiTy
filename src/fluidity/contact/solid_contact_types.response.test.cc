#include "fluidity/contact/solid_contact_types.h"

#include <gtest/gtest.h>

#include "fluidity/base/scalar_math.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/testing/contact_cases.h"

namespace fluidity::contact {
namespace {

constexpr Vec3 kUpwardNormal = {.x = 0, .y = 0, .z = 1};

constexpr Vec3 kFixed = {.x = 0, .y = 0, .z = 0};  // m/s, the velocity of a solid that does not move.

// m/s. Sideways 3 along x and 1 into a flat solid: the paper's figure.
constexpr Vec3 kSidewaysThreeIntoOne = {.x = 3, .y = 0, .z = -1};

// Fails the calling test unless every axis of actual has the bits of the same axis of expected, so +0 and -0 differ.
void CheckSameBits(Vec3 actual, Vec3 expected) {
  EXPECT_EQ(ToBits(actual.x), ToBits(expected.x)) << "axis x: " << actual.x << " vs " << expected.x;
  EXPECT_EQ(ToBits(actual.y), ToBits(expected.y)) << "axis y: " << actual.y << " vs " << expected.y;
  EXPECT_EQ(ToBits(actual.z), ToBits(expected.z)) << "axis z: " << actual.z << " vs " << expected.z;
}

// --- The approach ---

TEST(ComputeContactResponse, HeadOn_BecomesSolidVelocity) {
  // Approach speed -2 - 0.5 = -2.5. Removing it leaves {0, 0, 0.5} exactly, no sideways speed: the particle sticks
  // and gets solid_velocity by assignment.
  const Vec3 solid_velocity = {.x = 0, .y = 0, .z = 0.5};
  const Vec3 after = ComputeContactResponse({.x = 0, .y = 0, .z = -2}, solid_velocity, kUpwardNormal, 0.3);
  CheckSameBits(after, solid_velocity);
}

TEST(ComputeContactResponse, RisingSolidOvertakes_TakesItsNormalSpeed) {
  // The particle moves up at 0.5 m/s and the flat solid rises at 1 m/s: in the world the particle moves away from the
  // solid's surface, but measured from the solid it approaches at -0.5 m/s. Friction 0 keeps the sideways 2.
  const Vec3 after = ComputeContactResponse({.x = 2, .y = 0, .z = 0.5}, {.x = 0, .y = 0, .z = 1}, kUpwardNormal, 0);
  CheckSameBits(after, {.x = 2, .y = 0, .z = 1});
}

TEST(ComputeContactResponse, Leaving_IsUnchangedBitForBit) {
  const Vec3 velocity = {.x = 1, .y = 0, .z = 2};
  CheckSameBits(ComputeContactResponse(velocity, kFixed, kUpwardNormal, 0.5), velocity);
}

TEST(ComputeContactResponse, SidewaysOnly_IsUnchangedBitForBit) {
  // Approach speed exactly 0: the boundary of "does not approach". Treated as approaching, it would get to the stick
  // test with into_solid_speed -0, where 3 / -0 is -infinity, and stick at {0, 0, 0}.
  const Vec3 velocity = {.x = 3, .y = 0, .z = 0};
  CheckSameBits(ComputeContactResponse(velocity, kFixed, kUpwardNormal, 0.5), velocity);
}

TEST(ComputeContactResponse, OffAxisNormal_RemovesApproachLeftByRounding) {
  // Normal {0.6, 0, 0.8} (squared length 1 in doubles), velocity {-0.7, 0, -0.6}, fixed solid, friction 0.
  // Approach 0.6 * -0.7 + 0.8 * -0.6 rounds to -0.8999999999999999. One removal gives
  // {-0x1.47ae147ae147cp-3, 0, 0x1.eb851eb851eb8p-4}, whose approach is still -2^-56 (-1.39e-17): into the solid.
  // The second removal adds 2^-56 * 0.8 to z, one rounding unit: z becomes 0x1.eb851eb851eb9p-4 and the approach
  // is exactly 0. These values were computed step by step in IEEE doubles, rounding after each operation.
  const Vec3 normal = {.x = 0.6, .y = 0, .z = 0.8};
  const Vec3 after = ComputeContactResponse({.x = -0.7, .y = 0, .z = -0.6}, kFixed, normal, 0);
  CheckSameBits(after, {.x = -0x1.47ae147ae147cp-3, .y = 0, .z = 0x1.eb851eb851eb9p-4});
  EXPECT_EQ(math::Dot(normal, after), 0);
}

TEST(ComputeContactResponse, OffAxisNormal_RemovesLeavingRemainderToo) {
  // Normal {0.6, 0, 0.8}, velocity {-0.9, -0.9, -0.9}, fixed solid, friction 0. Approach -1.2600000000000002. One
  // removal leaves an approach of +17 * 2^-56 (2.4e-16), slightly away from the solid; it is removed as well, so the
  // velocity ends exactly along the surface. Computed step by step in IEEE doubles.
  const Vec3 normal = {.x = 0.6, .y = 0, .z = 0.8};
  const Vec3 after = ComputeContactResponse({.x = -0.9, .y = -0.9, .z = -0.9}, kFixed, normal, 0);
  CheckSameBits(after, {.x = -0x1.26e978d4fdf3dp-3, .y = -0x1.ccccccccccccdp-1, .z = 0x1.ba5e353f7cedap-4});
  EXPECT_EQ(math::Dot(normal, after), 0);
}

// --- The approach left by rounding ---

TEST(ComputeContactResponse, OffAxisNormal_LeavesSmallNegativeApproach) {
  // Normal {0.6, 0, 0.8}, velocity {-2, 0, -2}, fixed solid, friction 0. Approach -2.8. One removal gives
  // {-0x1.47ae147ae147cp-2, 0, 0x1.eb851eb851eb0p-3}, approach -1.94e-16; the second removal gives
  // {-0x1.47ae147ae147ap-2, 0, 0x1.eb851eb851eb6p-3}, whose approach computes to -2^-55: still slightly approaching,
  // within ComputeLargestApproachLeftAfterResponse = 2^-46 * (2 + 0 + 2) = 2^-44 = 5.7e-14. Computed step by step in
  // IEEE doubles.
  const Vec3 normal = {.x = 0.6, .y = 0, .z = 0.8};
  const Vec3 velocity = {.x = -2, .y = 0, .z = -2};
  const Vec3 after = ComputeContactResponse(velocity, kFixed, normal, 0);
  CheckSameBits(after, {.x = -0x1.47ae147ae147ap-2, .y = 0, .z = 0x1.eb851eb851eb6p-3});
  const double approach_left = math::Dot(normal, after - kFixed);
  EXPECT_EQ(approach_left, -0x1p-55);
  EXPECT_LE(math::Abs(approach_left), ComputeLargestApproachLeftAfterResponse(velocity, kFixed));
}

TEST(ComputeContactResponse, TinyVelocity_LeavesApproachWithinBound) {
  // The previous case times 2^-830 (about 1.4e-250 m/s): every operation stays above 2^-1022, so every result is the
  // previous one times 2^-830 exactly, and the approach left is -2^-885. The bound is 2^-46 * 2^-828 = 2^-874. A bound
  // built from Length would be 0 here: the squares, about 2^-1658, underflow.
  const Vec3 normal = {.x = 0.6, .y = 0, .z = 0.8};
  const Vec3 velocity = {.x = -0x1p-829, .y = 0, .z = -0x1p-829};
  const Vec3 after = ComputeContactResponse(velocity, kFixed, normal, 0);
  CheckSameBits(after, {.x = -0x1.47ae147ae147ap-832, .y = 0, .z = 0x1.eb851eb851eb6p-833});
  const double approach_left = math::Dot(normal, after - kFixed);
  EXPECT_EQ(approach_left, -0x1p-885);
  EXPECT_EQ(ComputeLargestApproachLeftAfterResponse(velocity, kFixed), 0x1p-874);
}

TEST(ComputeLargestApproachLeftAfterResponse, Example_IsShareOfAbsoluteAxes) {
  // 2^-46 * ((3 + 0 + 4) + (0 + 12 + 5)) = 2^-46 * (7 + 17) = 24 * 2^-46, exact. Signs do not matter.
  const double bound = ComputeLargestApproachLeftAfterResponse({.x = -3, .y = 0, .z = 4}, {.x = 0, .y = 12, .z = -5});
  EXPECT_EQ(bound, 24 * 0x1p-46);
}

TEST(ComputeLargestApproachLeftAfterResponse, TinySpeeds_BoundedAtSmallestRelativeSpeeds) {
  // Below 2^-1000 m/s the rounding of an operation is absolute, at most 2^-1075: a nonzero sum counts as 2^-1000, so
  // the bound is 2^-1046, above the at most 2^-1069 the response leaves. No speed at all keeps a bound of 0.
  EXPECT_EQ(ComputeLargestApproachLeftAfterResponse({.x = 0x1p-1010, .y = 0, .z = 0}, {}), 0x1p-1046);
  EXPECT_EQ(ComputeLargestApproachLeftAfterResponse({}, {.x = 0, .y = -0x1p-1074, .z = 0}), 0x1p-1046);
  EXPECT_EQ(ComputeLargestApproachLeftAfterResponse({}, {}), 0.0);
  EXPECT_EQ(ComputeLargestApproachLeftAfterResponse({.x = 0x1p-1000, .y = 0, .z = 0}, {}), 0x1p-1046);
}

// --- Friction ---

TEST(ComputeContactResponse, Slip_ReducesSidewaysByFrictionTimesApproach) {
  // Friction 0.5 takes 0.5 * 1 off the sideways 3: fraction 0.5 * 1 / 3, rounded to 1/6; 3 - 3 * (1/6) rounds to 2.5.
  const Vec3 after = ComputeContactResponse(kSidewaysThreeIntoOne, kFixed, kUpwardNormal, 0.5);
  CheckSameBits(after, {.x = 2.5, .y = 0, .z = 0});
}

TEST(ComputeContactResponse, SlipFasterApproach_TakesMoreSideways) {
  // Friction 0.25 takes 0.25 * 2 = 0.5 off the sideways 3: fraction 0.25 * 2 / 3, rounded to 1/6, as above.
  const Vec3 after = ComputeContactResponse({.x = 3, .y = 0, .z = -2}, kFixed, kUpwardNormal, 0.25);
  CheckSameBits(after, {.x = 2.5, .y = 0, .z = 0});
}

TEST(ComputeContactResponse, SlipMovingSolid_BlendsTowardSolidVelocity) {
  // Sideways 3 - 0.1 = 2.9 measured from the solid, approach -1: friction 0.75 takes 0.75 off, leaving 2.15 from the
  // solid, 2.25 in the world. velocity + fraction * (solid_velocity - velocity) gives 2.25 exactly; the form
  // (1 - fraction) * velocity + fraction * solid_velocity rounds to 2.2500000000000004.
  const Vec3 after = ComputeContactResponse(kSidewaysThreeIntoOne, {.x = 0.1, .y = 0, .z = 0}, kUpwardNormal, 0.75);
  CheckSameBits(after, {.x = 2.25, .y = 0, .z = 0});
}

TEST(ComputeContactResponse, StickAtThreshold_IsSolidVelocity) {
  // Sideways 3 over into-solid 1 is 3: friction 3 takes exactly all of it, and the limit counts as stick.
  CheckSameBits(ComputeContactResponse(kSidewaysThreeIntoOne, kFixed, kUpwardNormal, 3), kFixed);
}

TEST(ComputeContactResponse, StickAboveThreshold_IsSolidVelocity) {
  CheckSameBits(ComputeContactResponse(kSidewaysThreeIntoOne, kFixed, kUpwardNormal, 4), kFixed);
}

TEST(ComputeContactResponse, StickAtThresholdMovingSolid_IsSolidVelocityExactly) {
  // Velocity {3.1, 0, -1} at a solid moving at {0.1, 0, 0}: sideways 3.1 - 0.1 = 3 exactly, approach -1, friction 3,
  // so friction takes exactly all the sideways speed. The result is solid_velocity itself; the slip formula at
  // fraction 1 would give 3.1 + (0.1 - 3.1) = 0.10000000000000009.
  const Vec3 solid_velocity = {.x = 0.1, .y = 0, .z = 0};
  const Vec3 after = ComputeContactResponse({.x = 3.1, .y = 0, .z = -1}, solid_velocity, kUpwardNormal, 3);
  CheckSameBits(after, solid_velocity);
}

TEST(ComputeContactResponse, StickAtDividedThreshold_IsSolidVelocity) {
  // Sideways 3, into-solid 0.7: the stick limit is the double 3 / 0.7 = 0x1.1249249249249p+2, and that friction
  // sticks. This is why the test divides: written as friction * 0.7 >= 3, the product rounds to 2.9999999999999996,
  // below 3, and the particle would slip at 4.4e-16 m/s.
  const Vec3 after = ComputeContactResponse({.x = 3, .y = 0, .z = -0.7}, kFixed, kUpwardNormal, 0x1.1249249249249p+2);
  CheckSameBits(after, kFixed);
}

TEST(ComputeContactResponse, HugeFriction_Sticks) {
  // A friction near the largest double sticks: it is far above sideways 3 / into-solid 4 = 0.75.
  const Vec3 after = ComputeContactResponse({.x = 3, .y = 0, .z = -4}, kFixed, kUpwardNormal, 1e308);
  CheckSameBits(after, kFixed);
}

TEST(ComputeContactResponse, FrictionZero_KeepsNegativeZeroBitForBit) {
  // With friction 0 the velocity after the removal is the result. The normal's y is -0, so both removals subtract
  // +0 from the velocity's y of -0 and keep -0. A friction step at fraction 0 would add 0 * (0 - -0) = +0 and turn
  // the -0 into +0. Normal {0.6, -0, 0.8}, velocity {-0.8, -0, -0.3}: approach -0.72; the second removal's approach
  // is -2^-55 (computed step by step in IEEE doubles), so it subtracts -2^-55 * -0 = +0 from y.
  const Vec3 normal = {.x = 0.6, .y = -0.0, .z = 0.8};
  const Vec3 after = ComputeContactResponse({.x = -0.8, .y = -0.0, .z = -0.3}, kFixed, normal, 0);
  CheckSameBits(after, {.x = -0x1.78d4fdf3b645bp-2, .y = -0.0, .z = 0x1.1a9fbe76c8b43p-2});
}

TEST(ComputeContactResponse, FrictionZeroFastSolid_KeepsSidewaysBitForBit) {
  // The flat solid slides sideways at 1e8 m/s. In world coordinates: approach 0 * (1e-9 - 1e8) + 0 * 0 + 1 * -1 = -1,
  // and v - a n = {1e-9 + 0, 0 + 0, -1 + 1} = {1e-9, 0, +0}. Removed from the velocity measured from the solid
  // and added back, x would be (1e-9 - 1e8) + 1e8 = 0.
  const Vec3 after = ComputeContactResponse({.x = 1e-9, .y = 0, .z = -1}, {.x = 1e8, .y = 0, .z = 0}, kUpwardNormal, 0);
  CheckSameBits(after, {.x = 1e-9, .y = 0, .z = +0.0});
}

// --- Generated cases ---

TEST(ComputeContactResponse, FixedSolid_NeverRaisesSpeed) {
  // Removing the approach and moving toward a fixed solid's zero velocity can only shorten the velocity; rounding may
  // add a few rounding units. Draw order per case: a normal of three NextSigned() values, drawn again while its length
  // is below 0.25, then normalised; a velocity of 10 times three NextSigned() values; friction 2 * Next().
  CaseGenerator generator;
  int approaching_count = 0;
  for (int case_index = 0; case_index < 1000; ++case_index) {
    Vec3 normal{};
    do {
      normal = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (math::Length(normal) < 0.25);
    normal = math::Normalized(normal);
    const Vec3 direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const Vec3 velocity = 10 * direction;
    const double friction = 2 * generator.Next();

    const Vec3 after = ComputeContactResponse(velocity, kFixed, normal, friction);
    EXPECT_LE(math::Length(after), math::Length(velocity) * (1 + 0x1p-50)) << "case " << case_index;
    if (math::Dot(normal, velocity) < 0) {
      ++approaching_count;
    }
  }
  // About half the cases approach, so the set exercises the removal and friction, not only the unchanged branch.
  EXPECT_GT(approaching_count, 400);
}

}  // namespace
}  // namespace fluidity::contact
