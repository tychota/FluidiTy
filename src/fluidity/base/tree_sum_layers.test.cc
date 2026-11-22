// Runs the layers of the GPU tree sum on the CPU, pair by pair, with the shared formulas that the GPU function calls,
// and compares the result with math::TreeSum. Each layer is run as its GPU launch would cover it: every thread block
// and every thread of ComputeLaunchSize(pair_count), each thread over its grid-stride pairs, once in launch order and
// once in reverse, so a result that depended on the order of the threads would show.

#include "fluidity/base/tree_sum_layers.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/tree_sum_layers_cases.h"

namespace fluidity {
namespace {

TEST(ComputePairCount, OddSideCount_RoundsUp) {
  static_assert(ComputePairCount(5) == 3);
  EXPECT_EQ(ComputePairCount(1), 1U);
  EXPECT_EQ(ComputePairCount(0), 0U);
}

TEST(ComputeLayerValueCount, FiveValues_HoldsTheFirstTwoLayers) {
  static_assert(ComputeLayerValueCount(5) == 5);
  EXPECT_EQ(ComputeLayerValueCount(1000), 750U);
  EXPECT_EQ(ComputeLayerValueCount(1), 0U);
}

TEST(ComputeLayerArrayOffset, EvenLayer_WritesAfterTheFirstLayer) {
  EXPECT_EQ(ComputeLayerArrayOffset(5, 1), 0U);
  EXPECT_EQ(ComputeLayerArrayOffset(5, 2), 3U);
  EXPECT_EQ(ComputeLayerArrayOffset(5, 3), 0U);
}

TEST(ComputeLayerPairSum, PairWithPaddingOnTheRight_AddsPositiveZero) {
  const std::vector<double> sides = {1.0, 2.0, -0.0};
  EXPECT_EQ(ToBits(ComputeLayerPairSum(LayerSides{.values = sides.data(), .count = sides.size()}, 1)), ToBits(0.0));
}

TEST(RunLargestMagnitudeLayers, OneNaN_IsNaN) {
  std::vector<double> values = BuildMixedValues(1000);
  values[617] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(CheckSameBitsOrBothNaN(
      std::numeric_limits<double>::quiet_NaN(),
      ComputeLargestMagnitudeLayers(values, ThreadOrder::kLaunchOrder)
  ));
}

}  // namespace
}  // namespace fluidity
