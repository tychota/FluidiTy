// Runs the layers of the GPU tree sum on the CPU, pair by pair, with the shared formulas that the GPU function calls,
// and compares the result with math::TreeSum. Each layer is run as its GPU launch would cover it: every thread block
// and every thread of ComputeLaunchSize(pair_count), each thread over its grid-stride pairs, once in launch order and
// once in reverse, so a result that depended on the order of the threads would show.

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/tree_sum_layers_cases.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/tree_sum_layers.h"

namespace fluidity {
namespace {

TEST(RunLayers, EveryThreadInLaunchOrder_EqualsTreeSumBitForBit) {
  for (const std::size_t count : BuildComparedCounts()) {
    const std::vector<double> values = BuildMixedValues(count);
    EXPECT_EQ(
        ToBits(ComputeLayers(ComputeLayerPairSum, values, ThreadOrder::kLaunchOrder)),
        ToBits(math::TreeSum(values.data(), count))
    ) << "count "
      << count;
  }
}

TEST(RunLayers, EveryThreadInReverse_EqualsTreeSumBitForBit) {
  for (const std::size_t count : BuildComparedCounts()) {
    const std::vector<double> values = BuildMixedValues(count);
    EXPECT_EQ(
        ToBits(ComputeLayers(ComputeLayerPairSum, values, ThreadOrder::kReverse)),
        ToBits(math::TreeSum(values.data(), count))
    ) << "count "
      << count;
  }
}

TEST(RunLargestMagnitudeLayers, MixedValues_EqualsTheCpuLoop) {
  for (const std::size_t count : BuildComparedCounts()) {
    const std::vector<double> values = BuildMixedValues(count);
    double largest = 0.0;
    for (const double value : values) {
      largest = math::LargerMagnitude(largest, value);
    }
    EXPECT_EQ(ToBits(ComputeLargestMagnitudeLayers(values, ThreadOrder::kReverse)), ToBits(largest))
        << "count " << count;
  }
}

}  // namespace
}  // namespace fluidity
