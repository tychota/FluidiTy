// Runs the layers of the GPU tree sum on the CPU, pair by pair, with the shared formulas that the GPU function calls,
// and compares the result with math::TreeSum. Each layer is run as its GPU launch would cover it: every thread block
// and every thread of ComputeLaunchSize(pair_count), each thread over its grid-stride pairs, once in launch order and
// once in reverse, so a result that depended on the order of the threads would show.

#include "fluidity/base/tree_sum_layers.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "fluidity/base/ordered_sums.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/tree_sum.h"

namespace fluidity {
namespace {

// The order in which the simulated threads of one launch run.
enum class ThreadOrder : std::uint8_t {
  kLaunchOrder,  // Thread block 0, thread 0 first.
  kReverse,  // The last thread of the last thread block first.
};

// The pair formula a layer applies: ComputeLayerPairSum or ComputeLayerLargerMagnitude.
using PairFormula = double (*)(LayerSides, std::size_t);

// One layer as its GPU launch covers it: pair_sums[pair] = formula(sides, side_count, pair) for every pair, visited by
// thread block and thread in `order`, each thread over first_item, first_item + item_step, ...
void RunOneLayer(
    PairFormula formula,
    std::span<const double> sides,
    std::span<double> pair_sums,
    std::size_t pair_count,
    ThreadOrder order
) {
  const LaunchSize launch_size = ComputeLaunchSize(pair_count);
  const std::size_t thread_count = std::size_t{launch_size.thread_block_count} * launch_size.thread_block_size;
  for (std::size_t visit = 0; visit < thread_count; ++visit) {
    const std::size_t thread = order == ThreadOrder::kLaunchOrder ? visit : thread_count - 1 - visit;
    for (std::size_t pair = thread; pair < pair_count; pair += thread_count) {
      // The left side of every pair is a real side; the right side may be padding, which is never read.
      ASSERT_LT(2 * pair, sides.size());
      pair_sums[pair] = formula(LayerSides{.values = sides.data(), .count = sides.size()}, pair);
    }
  }
}

// The reduction of `values` as base/ordered_sums_gpu.cu runs it: layer after layer into two arrays of exactly
// ComputeLayerValueCount(values.size()) doubles, at ComputeLayerArrayOffset. values is not empty. It also checks
// that every layer fits in the arrays and that the doubles a layer writes never overlap the doubles it reads, since
// the threads of one layer run in no fixed order.
double RunLayers(PairFormula formula, std::span<const double> values, ThreadOrder order) {
  const std::size_t count = values.size();
  std::vector<double> layer_values(ComputeLayerValueCount(count));
  std::span<const double> sides = values;
  // Where the sides start in layer_values; layer 1 reads `values`, outside the layer arrays.
  bool sides_are_in_the_layer_arrays = false;
  std::size_t sides_offset = 0;
  std::size_t layer = 1;
  while (sides.size() > 1) {
    const std::size_t pair_count = ComputePairCount(sides.size());
    const std::size_t offset = ComputeLayerArrayOffset(count, layer);
    // Every layer must fit in the arrays as the sum buffers size them.
    EXPECT_LE(offset + pair_count, layer_values.size()) << "layer " << layer << " of count " << count;
    if (offset + pair_count > layer_values.size()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    if (sides_are_in_the_layer_arrays) {
      const bool written_ends_before_the_sides = offset + pair_count <= sides_offset;
      const bool written_starts_after_the_sides = sides_offset + sides.size() <= offset;
      EXPECT_TRUE(written_ends_before_the_sides || written_starts_after_the_sides)
          << "layer " << layer << " of count " << count << " writes [" << offset << ", " << offset + pair_count
          << ") and reads [" << sides_offset << ", " << sides_offset + sides.size() << ")";
    }
    const std::span<double> written(layer_values.data() + offset, pair_count);
    RunOneLayer(formula, sides, written, pair_count, order);
    sides = written;
    sides_are_in_the_layer_arrays = true;
    sides_offset = offset;
    ++layer;
  }
  return sides.front();
}

// The largest magnitude as the GPU computes it: the layers, then the magnitude of the top.
double RunLargestMagnitudeLayers(std::span<const double> values, ThreadOrder order) {
  return math::LargerMagnitude(RunLayers(ComputeLayerLargerMagnitude, values, order), 0.0);
}

// The counts compared: every count up to 300, which covers every shape of a short tree, then large ones around
// powers of two.
std::vector<std::size_t> BuildComparedCounts() {
  std::vector<std::size_t> counts;
  for (std::size_t count = 1; count <= 300; ++count) {
    counts.push_back(count);
  }
  for (const std::size_t count : {1023U, 1024U, 1025U, 65537U, 1000003U}) {
    counts.push_back(count);
  }
  return counts;
}

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

TEST(RunLayers, EveryThreadInLaunchOrder_EqualsTreeSumBitForBit) {
  for (const std::size_t count : BuildComparedCounts()) {
    const std::vector<double> values = BuildMixedValues(count);
    EXPECT_EQ(
        ToBits(RunLayers(ComputeLayerPairSum, values, ThreadOrder::kLaunchOrder)),
        ToBits(math::TreeSum(values.data(), count))
    ) << "count "
      << count;
  }
}

TEST(RunLayers, EveryThreadInReverse_EqualsTreeSumBitForBit) {
  for (const std::size_t count : BuildComparedCounts()) {
    const std::vector<double> values = BuildMixedValues(count);
    EXPECT_EQ(
        ToBits(RunLayers(ComputeLayerPairSum, values, ThreadOrder::kReverse)),
        ToBits(math::TreeSum(values.data(), count))
    ) << "count "
      << count;
  }
}

TEST(RunLayers, NegativeZerosOnly_FollowTheTreeSumPaddingRule) {
  for (std::size_t count = 1; count <= 70; ++count) {
    const std::vector<double> values(count, -0.0);
    EXPECT_EQ(
        ToBits(RunLayers(ComputeLayerPairSum, values, ThreadOrder::kReverse)),
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
    EXPECT_EQ(ToBits(RunLargestMagnitudeLayers(values, ThreadOrder::kReverse)), ToBits(largest)) << "count " << count;
  }
}

TEST(RunLargestMagnitudeLayers, OneNaN_IsNaN) {
  std::vector<double> values = BuildMixedValues(1000);
  values[617] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(CheckSameBitsOrBothNaN(
      std::numeric_limits<double>::quiet_NaN(),
      RunLargestMagnitudeLayers(values, ThreadOrder::kLaunchOrder)
  ));
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
