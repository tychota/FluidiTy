#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/tree_sum_cases.h"
#include "fluidity/base/tree_sum.h"

namespace fluidity {
namespace {

TEST(TreeSum, EveryCountUpTo300_EqualsThePaddedTree) {
  const std::vector<double> values = BuildMixedValuesWithNegativeZeros(300);
  for (std::size_t count = 0; count <= values.size(); ++count) {
    const std::span<const double> first_values(values.data(), count);
    EXPECT_EQ(ToBits(math::TreeSum(values.data(), count)), ToBits(ComputePaddedTreeSum(first_values)))
        << "count " << count;
  }
}

TEST(TreeSum, AllNegativeZeros_EveryCountMatchesPaddedTree) {
  const std::vector<double> values(300, -0.0);
  for (std::size_t count = 0; count <= values.size(); ++count) {
    const std::span<const double> first_values(values.data(), count);
    EXPECT_EQ(ToBits(math::TreeSum(values.data(), count)), ToBits(ComputePaddedTreeSum(first_values)))
        << "count " << count;
  }
}

}  // namespace
}  // namespace fluidity
