#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

#include "fluidity/base/ordered_sums.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/tree_sum_layers_cases.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/tree_sum_layers.h"

namespace fluidity {
namespace {

TEST(RunLayers, NegativeZerosOnly_FollowTheTreeSumPaddingRule) {
  for (std::size_t count = 1; count <= 70; ++count) {
    const std::vector<double> values(count, -0.0);
    EXPECT_EQ(
        ToBits(ComputeLayers(ComputeLayerPairSum, values, ThreadOrder::kReverse)),
        ToBits(math::TreeSum(values.data(), count))
    ) << "count "
      << count;
  }
}

TEST(ComputeSumBuffersSize, OnGpu_HoldsEveryAreaOfEveryCount) {
  // The sum buffers of base/ordered_sums_cpu.cc: max_count products, then the layer arrays at the next multiple of 64
  // bytes. Its bytes must hold both for every count.
  const SumBuffersSize size = ComputeSumBuffersSize(RunOn::kGpu);
  for (const std::size_t count : BuildComparedCounts()) {
    const std::size_t products_end = (sizeof(double) * count + 63) / 64 * 64;
    const std::size_t needed = products_end + sizeof(double) * ComputeLayerValueCount(count);
    EXPECT_LE(needed, size.bytes_per_item * count + size.fixed_bytes) << "count " << count;
  }
}

}  // namespace
}  // namespace fluidity
