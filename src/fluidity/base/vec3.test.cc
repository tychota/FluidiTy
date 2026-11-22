#include "fluidity/base/vec3.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>

namespace fluidity {
namespace {

// Every value below is a small integer, so each sum and product is exact in doubles and exact equality is right.
constexpr std::array<double, 3> ToArray(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}

TEST(Vec3, AddTwoVectors_AddsEachAxis) {
  constexpr Vec3 kSum = Vec3{.x = 1, .y = 2, .z = 3} + Vec3{.x = 10, .y = 20, .z = 30};
  static_assert(ToArray(kSum) == std::array{11.0, 22.0, 33.0});
  EXPECT_EQ(ToArray(kSum), (std::array{11.0, 22.0, 33.0}));
}

TEST(Vec3, SubtractTwoVectors_SubtractsEachAxis) {
  constexpr Vec3 kDifference = Vec3{.x = 11, .y = 22, .z = 33} - Vec3{.x = 10, .y = 20, .z = 30};
  static_assert(ToArray(kDifference) == std::array{1.0, 2.0, 3.0});
  EXPECT_EQ(ToArray(kDifference), (std::array{1.0, 2.0, 3.0}));
}

TEST(Vec3, Negate_FlipsEachAxis) {
  constexpr Vec3 kNegated = -Vec3{.x = 1, .y = -2, .z = 3};
  static_assert(ToArray(kNegated) == std::array{-1.0, 2.0, -3.0});
  EXPECT_EQ(ToArray(kNegated), (std::array{-1.0, 2.0, -3.0}));
}

TEST(Vec3, ScaleFromEitherSide_GivesSameVector) {
  constexpr Vec3 kVector = {.x = 1, .y = 2, .z = 3};
  constexpr Vec3 kScaledOnRight = kVector * 2.0;
  constexpr Vec3 kScaledOnLeft = 2.0 * kVector;
  static_assert(ToArray(kScaledOnRight) == std::array{2.0, 4.0, 6.0});
  static_assert(ToArray(kScaledOnLeft) == std::array{2.0, 4.0, 6.0});
  EXPECT_EQ(ToArray(kScaledOnRight), (std::array{2.0, 4.0, 6.0}));
  EXPECT_EQ(ToArray(kScaledOnLeft), (std::array{2.0, 4.0, 6.0}));
}

TEST(Dot, PerpendicularVectors_IsZero) {
  constexpr double kDot = math::Dot(Vec3{.x = 1, .y = 0, .z = 0}, Vec3{.x = 0, .y = 1, .z = 0});
  static_assert(kDot == 0.0);
  EXPECT_EQ(kDot, 0.0);
}

TEST(Dot, GeneralVectors_SumsAxisProducts) {
  // 1 * 4 + 2 * 5 + 3 * 6 = 32
  constexpr double kDot = math::Dot(Vec3{.x = 1, .y = 2, .z = 3}, Vec3{.x = 4, .y = 5, .z = 6});
  static_assert(kDot == 32.0);
  EXPECT_EQ(kDot, 32.0);
}

TEST(Cross, XThenY_IsZ) {
  // Right-handed axes: X cross Y is +Z, not -Z.
  constexpr Vec3 kCross = math::Cross(Vec3{.x = 1, .y = 0, .z = 0}, Vec3{.x = 0, .y = 1, .z = 0});
  static_assert(ToArray(kCross) == std::array{0.0, 0.0, 1.0});
  EXPECT_EQ(ToArray(kCross), (std::array{0.0, 0.0, 1.0}));
}

TEST(Cross, GeneralVectors_FollowsTheRightHandRule) {
  // {2 * 6 - 3 * 5, 3 * 4 - 1 * 6, 1 * 5 - 2 * 4} = {-3, 6, -3}
  constexpr Vec3 kCross = math::Cross(Vec3{.x = 1, .y = 2, .z = 3}, Vec3{.x = 4, .y = 5, .z = 6});
  static_assert(ToArray(kCross) == std::array{-3.0, 6.0, -3.0});
  EXPECT_EQ(ToArray(kCross), (std::array{-3.0, 6.0, -3.0}));
}

TEST(Cross, SameVector_IsZero) {
  constexpr Vec3 kVector = {.x = 1, .y = 2, .z = 3};
  constexpr Vec3 kCross = math::Cross(kVector, kVector);
  static_assert(ToArray(kCross) == std::array{0.0, 0.0, 0.0});
  EXPECT_EQ(ToArray(kCross), (std::array{0.0, 0.0, 0.0}));
}

TEST(Length, ThreeFourZero_IsFive) {
  EXPECT_EQ(math::Length(Vec3{.x = 3, .y = 4, .z = 0}), 5.0);
}

TEST(Length, ThreeNonZeroAxes_UsesEveryAxis) {
  // 4 + 9 + 36 = 49, whose square root is 7.
  EXPECT_EQ(math::Length(Vec3{.x = 2, .y = -3, .z = 6}), 7.0);
}

TEST(Normalized, ThreeNonZeroAxes_DividesEachAxisByTheLength) {
  const Vec3 normalized = math::Normalized(Vec3{.x = 2, .y = -3, .z = 6});
  // Each axis is one correctly rounded division by the length 7.
  EXPECT_EQ(normalized.x, 2.0 / 7.0);
  EXPECT_EQ(normalized.y, -3.0 / 7.0);
  EXPECT_EQ(normalized.z, 6.0 / 7.0);
}

TEST(Normalized, AnyNonZeroVector_HasLengthOne) {
  constexpr std::array kVectors = {
      Vec3{.x = 1, .y = 0, .z = 0},
      Vec3{.x = 0.3, .y = -1.7, .z = 2.9},
      Vec3{.x = 1e-150, .y = 2e-150, .z = -3e-150},
      Vec3{.x = 4e150, .y = -1e150, .z = 2e150},
  };
  for (const Vec3 vector : kVectors) {
    EXPECT_NEAR(math::Length(math::Normalized(vector)), 1.0, 1e-15);
  }
}

TEST(IsFinite, FiniteVector_IsFinite) {
  EXPECT_TRUE(math::IsFinite(Vec3{.x = 1, .y = -2, .z = 3}));
}

TEST(IsFinite, VectorWithOneNaN_IsNotFinite) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(math::IsFinite(Vec3{.x = kNaN, .y = 0, .z = 0}));
  EXPECT_FALSE(math::IsFinite(Vec3{.x = 0, .y = kNaN, .z = 0}));
  EXPECT_FALSE(math::IsFinite(Vec3{.x = 0, .y = 0, .z = kNaN}));
}

TEST(IsFinite, VectorWithOneInfinity_IsNotFinite) {
  EXPECT_FALSE(math::IsFinite(Vec3{.x = 0, .y = std::numeric_limits<double>::infinity(), .z = 0}));
}

}  // namespace
}  // namespace fluidity
