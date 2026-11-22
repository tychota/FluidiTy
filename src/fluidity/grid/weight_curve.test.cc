#include "fluidity/grid/weight_curve.h"

#include <gtest/gtest.h>

#include <cmath>

namespace fluidity::grid {
namespace {

static_assert(ComputeCompleteWeightedSpread(2.0) == 1.0);

TEST(ComputeWeightCurve, AtAndBeyondReach_IsZero) {
  const double just_past_one_and_a_half = std::nextafter(1.5, 2.0);
  for (const double distance : {1.5, 2.0, just_past_one_and_a_half, 1.55, 1.75, 1e300}) {
    for (const double distance_with_sign : {distance, -distance}) {
      const WeightCurveValue value = ComputeWeightCurve(distance_with_sign);
      EXPECT_EQ(value.weight, 0.0) << distance_with_sign;
      EXPECT_EQ(value.slope, 0.0) << distance_with_sign;
    }
  }
  static_assert(kWeightCurveReach == 1.5);
}

TEST(ComputeWeightCurve, JustBelowOneAndAHalf_IsPositiveAndFollowsTheOuterBranch) {
  const double just_below_one_and_a_half = std::nextafter(1.5, 0.0);
  const double distance_to_outer_edge = 1.5 - just_below_one_and_a_half;
  for (const double sign : {1.0, -1.0}) {
    const WeightCurveValue value = ComputeWeightCurve(sign * just_below_one_and_a_half);
    EXPECT_GT(value.weight, 0.0) << sign;
    EXPECT_EQ(value.weight, 0.5 * distance_to_outer_edge * distance_to_outer_edge) << sign;
    EXPECT_EQ(value.slope, -sign * distance_to_outer_edge) << sign;
  }
}

TEST(ComputeWeightCurve, OneStepEitherSideOfHalf_FollowsEachBranch) {
  const double below_half = std::nextafter(0.5, 0.0);
  const double above_half = std::nextafter(0.5, 1.0);
  for (const double sign : {1.0, -1.0}) {
    const WeightCurveValue inner = ComputeWeightCurve(sign * below_half);
    EXPECT_EQ(inner.weight, 0.75 - below_half * below_half) << sign;
    EXPECT_EQ(inner.slope, -2.0 * sign * below_half) << sign;
    const double distance_to_outer_edge = 1.5 - above_half;
    const WeightCurveValue outer = ComputeWeightCurve(sign * above_half);
    EXPECT_EQ(outer.weight, 0.5 * distance_to_outer_edge * distance_to_outer_edge) << sign;
    EXPECT_EQ(outer.slope, -sign * distance_to_outer_edge) << sign;
  }
}

TEST(ComputeWeightCurve, AtHalf_BothBranchesAgree) {
  const WeightCurveValue value = ComputeWeightCurve(0.5);
  EXPECT_EQ(value.weight, 0.5);
  EXPECT_EQ(value.slope, -1.0);
}

TEST(ComputeWeightCurve, InsideAndOutsideHalf_MatchTheWorkedExamples) {
  const WeightCurveValue inner = ComputeWeightCurve(0.25);
  EXPECT_EQ(inner.weight, 0.6875);
  EXPECT_EQ(inner.slope, -0.5);
  const WeightCurveValue outer = ComputeWeightCurve(-1.25);
  EXPECT_EQ(outer.weight, 0.03125);
  EXPECT_EQ(outer.slope, 0.25);
}

TEST(ComputeCompleteWeightedSpread, WidthTwo_IsOne) {
  static_assert(ComputeCompleteWeightedSpread(2.0) == 1.0);
  EXPECT_EQ(ComputeCompleteWeightedSpread(2.0), 1.0);
}

TEST(ComputeCompleteWeightedSpread, AnyWidth_IsAQuarterOfItsSquare) {
  // Scaling by 2^-2 is exact, so the result is the double nearest width^2 / 4.
  for (const double width : {0.5, 0.1, 1.0, 3.0, 7.77, 1e-3, 1e3}) {
    EXPECT_EQ(ComputeCompleteWeightedSpread(width), std::ldexp(width * width, -2)) << width;
  }
}

TEST(WeightCurveConstants, Values_AreTheDesignedOnes) {
  static_assert(kMaxNearbyFaces == 216);
  static_assert(kMinWeightSum == 0.1);
  static_assert(kMinPivotShare == 1e-12);
  SUCCEED();
}

}  // namespace
}  // namespace fluidity::grid
