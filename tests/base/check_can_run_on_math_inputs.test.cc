#include <gtest/gtest.h>

#include <cmath>

#include "fluidity/base/check_can_run_on_math.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"

namespace fluidity {
namespace {

// Whether every double of a result is an ordinary number, neither infinite nor NaN.
bool ComputeIsFiniteResult(const MathCheckResult& result) {
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
    EXPECT_TRUE(ComputeIsFiniteResult(result)) << "input " << input_index;
  }
}

}  // namespace
}  // namespace fluidity
