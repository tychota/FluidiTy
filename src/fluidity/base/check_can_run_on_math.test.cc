#include "fluidity/base/check_can_run_on_math.h"

#include <gtest/gtest.h>

#include <cmath>

#include "fluidity/base/scalar_math.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"

namespace fluidity {
namespace {

// Whether every double of a result is an ordinary number, neither infinite nor NaN.
bool IsFiniteResult(const MathCheckResult& result) {
  const bool vectors_are_finite = math::IsFinite(result.cross) && math::IsFinite(result.rotated_first);
  const bool dot_values_are_finite = math::IsFinite(result.dot) && math::IsFinite(result.square_root_of_dot)
      && math::IsFinite(result.floor_of_scaled_dot);
  const bool others_are_finite = math::IsFinite(result.determinant) && math::IsFinite(result.length_of_difference);
  return vectors_are_finite && dot_values_are_finite && others_are_finite;
}

TEST(BuildMathCheckInputs, SeveralCrossProducts_RoundDifferentlyWhenFused) {
  // std::fma rounds first.y * second.z - first.z * second.y once, as a GPU built with --fmad=true would.
  int fused_differs_count = 0;
  for (int input_index = 0; input_index < kMathCheckInputCount; ++input_index) {
    const MathCheckInputs inputs = BuildMathCheckInputs(input_index);
    const double fused_cross_x = std::fma(inputs.first.y, inputs.second.z, -(inputs.first.z * inputs.second.y));
    const MathCheckResult result = ComputeMathCheckResult(inputs);
    if (ToBits(fused_cross_x) != ToBits(result.cross.x)) {
      ++fused_differs_count;
    }
  }
  EXPECT_GE(fused_differs_count, 4);
}

TEST(BuildMathCheckInputs, SeveralProducts_AreSubnormal) {
  int subnormal_count = 0;
  for (int input_index = 0; input_index < kMathCheckInputCount; ++input_index) {
    const MathCheckInputs inputs = BuildMathCheckInputs(input_index);
    const double product = inputs.first.y * inputs.second.z;
    if (std::fpclassify(product) == FP_SUBNORMAL) {
      ++subnormal_count;
    }
  }
  EXPECT_GE(subnormal_count, 4);
}

TEST(ComputeMathCheckResult, EveryInput_GivesFiniteValues) {
  for (int input_index = 0; input_index < kMathCheckInputCount; ++input_index) {
    const MathCheckResult result = ComputeMathCheckResult(BuildMathCheckInputs(input_index));
    EXPECT_TRUE(IsFiniteResult(result)) << "input " << input_index;
  }
}

// The expected bits below were computed apart from this code, with Python floats, which round each add, subtract,
// multiply and square root separately, as the numerical flags require.
TEST(ComputeMathCheckResult, FirstInput_HasPinnedBits) {
  const MathCheckResult result = ComputeMathCheckResult(BuildMathCheckInputs(0));
  EXPECT_EQ(ToBits(result.cross.x), ToBits(0x1.d23ae165cd6e8p-1));
  EXPECT_EQ(ToBits(result.cross.y), ToBits(-0x1.dbef740b9add0p-1));
  EXPECT_EQ(ToBits(result.cross.z), ToBits(-0x1.20845da329180p-5));
  EXPECT_EQ(ToBits(result.dot), ToBits(0x1.39c6a26df9f00p+3));
  EXPECT_EQ(ToBits(result.rotated_first.x), ToBits(-0x1.40d6e4999999cp-2));
  EXPECT_EQ(ToBits(result.determinant), ToBits(-0x1.71fa865cc52b4p+1));
  EXPECT_EQ(ToBits(result.square_root_of_dot), ToBits(0x1.90d0d547e58d8p+1));
  EXPECT_EQ(ToBits(result.floor_of_scaled_dot), ToBits(0x1.39c6a26df9e00p+43));
  EXPECT_EQ(ToBits(result.length_of_difference), ToBits(0x1.f7a96189ffe31p-2));
}

TEST(ComputeMathCheckResult, ScaledInput_HasPinnedSubnormalBits) {
  const MathCheckResult result = ComputeMathCheckResult(BuildMathCheckInputs(3));
  EXPECT_EQ(ToBits(result.cross.x), ToBits(0x0.000178f7436c4p-1022));
  EXPECT_EQ(ToBits(result.dot), ToBits(0x0.0000c2f83f86fp-1022));
  EXPECT_EQ(ToBits(result.square_root_of_dot), ToBits(0x1.bed22c12333e8p-520));
  EXPECT_EQ(ToBits(result.length_of_difference), ToBits(0x1.ae3e093bd25d2p-519));
}

}  // namespace
}  // namespace fluidity
