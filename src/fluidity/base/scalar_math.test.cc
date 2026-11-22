#include "fluidity/base/scalar_math.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>

namespace fluidity {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// The functions that may differ in the last bits between processors are compared with this tolerance, as every
// formula that uses them is. It is a few times the rounding unit of a value near 1.
constexpr double kLastBitsTolerance = 1e-15;

TEST(Floor, NegativeHalf_IsMinusOne) {
  EXPECT_EQ(math::Floor(-0.5), -1.0);
}

TEST(Floor, PositiveValue_DropsTheFraction) {
  EXPECT_EQ(math::Floor(2.75), 2.0);
}

TEST(Abs, NegativeZero_IsPositiveZero) {
  EXPECT_FALSE(std::signbit(math::Abs(-0.0)));
}

TEST(Abs, PositiveZero_IsPositiveZero) {
  EXPECT_FALSE(std::signbit(math::Abs(0.0)));
}

TEST(Abs, NegativeValue_FlipsTheSign) {
  static_assert(math::Abs(-2.5) == 2.5);
  EXPECT_EQ(math::Abs(-2.5), 2.5);
}

TEST(Abs, PositiveValue_IsUnchanged) {
  static_assert(math::Abs(2.5) == 2.5);
  EXPECT_EQ(math::Abs(2.5), 2.5);
}

TEST(Abs, NaN_IsNaN) {
  EXPECT_TRUE(std::isnan(math::Abs(kNaN)));
}

TEST(Sqrt, PerfectSquare_IsExact) {
  EXPECT_EQ(math::Sqrt(9.0), 3.0);
}

TEST(Sqrt, Two_IsCorrectlyRounded) {
  // 1.4142135623730951 is the double nearest the square root of 2.
  EXPECT_EQ(math::Sqrt(2.0), 1.4142135623730951);
}

TEST(Min, TwoNumbers_IsTheSmaller) {
  static_assert(math::Min(3.0, -1.0) == -1.0);
  EXPECT_EQ(math::Min(3.0, -1.0), -1.0);
  EXPECT_EQ(math::Min(-1.0, 3.0), -1.0);
}

TEST(Min, OneNaN_ReturnsTheOtherValue) {
  static_assert(math::Min(kNaN, 2.0) == 2.0);
  static_assert(math::Min(2.0, kNaN) == 2.0);
  EXPECT_EQ(math::Min(kNaN, 2.0), 2.0);
  EXPECT_EQ(math::Min(2.0, kNaN), 2.0);
}

TEST(Min, TwoNaNs_IsNaN) {
  EXPECT_TRUE(std::isnan(math::Min(kNaN, kNaN)));
}

TEST(Max, TwoNumbers_IsTheLarger) {
  static_assert(math::Max(3.0, -1.0) == 3.0);
  EXPECT_EQ(math::Max(3.0, -1.0), 3.0);
  EXPECT_EQ(math::Max(-1.0, 3.0), 3.0);
}

TEST(Max, OneNaN_ReturnsTheOtherValue) {
  static_assert(math::Max(kNaN, 2.0) == 2.0);
  static_assert(math::Max(2.0, kNaN) == 2.0);
  EXPECT_EQ(math::Max(kNaN, 2.0), 2.0);
  EXPECT_EQ(math::Max(2.0, kNaN), 2.0);
}

TEST(Max, TwoNaNs_IsNaN) {
  EXPECT_TRUE(std::isnan(math::Max(kNaN, kNaN)));
}

TEST(IsFinite, Infinity_IsNotFinite) {
  EXPECT_FALSE(math::IsFinite(kInfinity));
  EXPECT_FALSE(math::IsFinite(-kInfinity));
}

TEST(IsFinite, NaN_IsNotFinite) {
  EXPECT_FALSE(math::IsFinite(kNaN));
}

TEST(IsFinite, LargestDouble_IsFinite) {
  EXPECT_TRUE(math::IsFinite(std::numeric_limits<double>::max()));
}

TEST(IsNan, NaN_IsNan) {
  static_assert(math::IsNan(kNaN));
  EXPECT_TRUE(math::IsNan(kNaN));
}

TEST(IsNan, InfinityAndNumbers_AreNotNan) {
  static_assert(!math::IsNan(kInfinity));
  EXPECT_FALSE(math::IsNan(kInfinity));
  EXPECT_FALSE(math::IsNan(-kInfinity));
  EXPECT_FALSE(math::IsNan(0.0));
  EXPECT_FALSE(math::IsNan(-1.5));
}

TEST(PositiveInfinity, Constant_IsTheStandardInfinity) {
  static_assert(math::kPositiveInfinity == kInfinity);
  static_assert(math::kPositiveInfinity > std::numeric_limits<double>::max());
  EXPECT_FALSE(math::IsFinite(math::kPositiveInfinity));
  EXPECT_EQ(math::Min(math::kPositiveInfinity, 2.0), 2.0);
}

TEST(Cbrt, PerfectCube_IsItsRoot) {
  EXPECT_NEAR(math::Cbrt(27.0), 3.0, kLastBitsTolerance);
  EXPECT_NEAR(math::Cbrt(-8.0), -2.0, kLastBitsTolerance);
}

TEST(Exp, One_IsE) {
  EXPECT_NEAR(math::Exp(1.0), std::numbers::e, kLastBitsTolerance);
}

TEST(Expm1, TinyValue_KeepsTheSmallPart) {
  // exp(1e-10) - 1 = 1e-10 + 5e-21 + ... Computing Exp(1e-10) - 1 would lose most of these digits.
  EXPECT_NEAR(math::Expm1(1e-10), 1.00000000005e-10, 1e-25);
}

TEST(Log, E_IsOne) {
  EXPECT_NEAR(math::Log(std::numbers::e), 1.0, kLastBitsTolerance);
}

TEST(Pow, TwoToTheTenth_Is1024) {
  EXPECT_NEAR(math::Pow(2.0, 10.0), 1024.0, 1024.0 * kLastBitsTolerance);
}

TEST(Pow, FractionalExponent_IsARoot) {
  EXPECT_NEAR(math::Pow(16.0, 0.25), 2.0, kLastBitsTolerance);
}

TEST(Sin, SixthOfPi_IsOneHalf) {
  EXPECT_NEAR(math::Sin(std::numbers::pi / 6.0), 0.5, kLastBitsTolerance);
}

TEST(Cos, ThirdOfPi_IsOneHalf) {
  EXPECT_NEAR(math::Cos(std::numbers::pi / 3.0), 0.5, kLastBitsTolerance);
}

}  // namespace
}  // namespace fluidity
