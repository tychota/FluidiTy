#include "fluidity/base/tree_sum.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/tree_sum_cases.h"

namespace fluidity {
namespace {

TEST(TreeSum, Empty_IsZero) {
  EXPECT_EQ(ToBits(math::TreeSum(nullptr, 0)), ToBits(0.0));
}

TEST(TreeSum, OneValue_IsThatValue) {
  constexpr std::array kValues = {-2.5};
  EXPECT_EQ(ToBits(math::TreeSum(kValues.data(), kValues.size())), ToBits(-2.5));
}

TEST(TreeSum, OneNegativeZero_IsNegativeZero) {
  constexpr std::array kValues = {-0.0};
  EXPECT_EQ(ToBits(math::TreeSum(kValues.data(), kValues.size())), ToBits(-0.0));
}

TEST(TreeSum, TwoNegativeZeros_IsNegativeZero) {
  constexpr std::array kValues = {-0.0, -0.0};
  EXPECT_EQ(ToBits(math::TreeSum(kValues.data(), kValues.size())), ToBits(-0.0));
}

TEST(TreeSum, ThreeNegativeZeros_IsPositiveZero) {
  // Padded to four: (-0.0 + -0.0) + (-0.0 + 0.0) = -0.0 + +0.0 = +0.0.
  constexpr std::array kValues = {-0.0, -0.0, -0.0};
  EXPECT_EQ(ToBits(math::TreeSum(kValues.data(), kValues.size())), ToBits(0.0));
}

TEST(TreeSum, FourValues_AddsPairsThenPairs) {
  // Near 1e16 the doubles are 2 apart: 1e16 + 1 and -1e16 + 1 are ties and round to the even neighbour, +-1e16.
  constexpr std::array kValues = {1e16, 1.0, -1e16, 1.0};
  EXPECT_EQ(ToBits(math::TreeSum(kValues.data(), kValues.size())), ToBits(0.0));
  double left_to_right = 0.0;
  for (const double value : kValues) {
    left_to_right += value;
  }
  EXPECT_EQ(left_to_right, 1.0);
}

TEST(TreeSum, SevenValues_EqualsEightWithATrailingZero) {
  const std::vector<double> seven = BuildMixedValuesWithNegativeZeros(7);
  std::vector<double> eight = seven;
  eight.push_back(0.0);
  EXPECT_EQ(ToBits(math::TreeSum(seven.data(), seven.size())), ToBits(math::TreeSum(eight.data(), eight.size())));
}

TEST(TreeSum, PowerOfTwoCount_EqualsThePaddedTree) {
  const std::vector<double> values = BuildMixedValuesWithNegativeZeros(4096);
  EXPECT_EQ(ToBits(math::TreeSum(values.data(), values.size())), ToBits(ComputePaddedTreeSum(values)));
}

TEST(TreeSumOf, ComputedValues_EqualsTreeSumOfTheSameValues) {
  const std::vector<double> values = BuildMixedValuesWithNegativeZeros(77);
  const double computed = math::TreeSumOf(values.size(), [&values](std::size_t index) {
    return values[index] * 3.0;
  });
  std::vector<double> tripled(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    tripled[index] = values[index] * 3.0;
  }
  EXPECT_EQ(ToBits(computed), ToBits(math::TreeSum(tripled.data(), tripled.size())));
}

TEST(LargerMagnitude, MixedSigns_ReturnsTheLargerAbsoluteValue) {
  EXPECT_EQ(math::LargerMagnitude(-3.0, 2.0), 3.0);
  EXPECT_EQ(math::LargerMagnitude(2.0, -3.0), 3.0);
}

TEST(LargerMagnitude, TwoZeros_IsPositiveZero) {
  EXPECT_EQ(ToBits(math::LargerMagnitude(-0.0, -0.0)), ToBits(0.0));
}

TEST(LargerMagnitude, NaNOnEitherSide_IsNaN) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(math::LargerMagnitude(kNaN, 1.0)));
  EXPECT_TRUE(std::isnan(math::LargerMagnitude(1.0, kNaN)));
}

}  // namespace
}  // namespace fluidity
