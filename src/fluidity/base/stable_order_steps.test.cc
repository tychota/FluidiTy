// Runs the GPU stable order, prefix sum and group starts on the CPU, with the per-thread steps that the GPU functions
// call, and compares the results with plain references. Each GPU function is run as its launch would cover it: every
// thread block and every thread, phase by phase, where a phase is the code between two waits of the thread block
// (__syncthreads); every thread of a thread block finishes a phase before any starts the next. The thread blocks and
// threads run once in launch order and once in reverse, so a result that depended on the order of the threads would
// show. The host loops (passes, layers) follow base/stable_order_gpu.cu.

#include "fluidity/base/stable_order_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/base/stable_order_digit.h"

namespace fluidity {
namespace {

TEST(ComputeGroupStart, SortedDestinations_GivesTheCountBelowEachGroup) {
  const std::vector<std::uint32_t> sorted = {0, 0, 1, 3, 3, 3};
  const SortedDestinations destinations = {.values = sorted.data(), .count = sorted.size()};
  const std::vector<std::uint32_t> starts = {
      ComputeGroupStart(destinations, 0),
      ComputeGroupStart(destinations, 1),
      ComputeGroupStart(destinations, 2),
      ComputeGroupStart(destinations, 3),
      ComputeGroupStart(destinations, 4),
  };
  EXPECT_EQ(starts, (std::vector<std::uint32_t>{0, 2, 3, 3, 6}));
}

TEST(ComputeDestinationPassCount, DestinationCounts_CoverTheBytesOfTheLargest) {
  static_assert(ComputeDestinationPassCount(1) == 1);
  EXPECT_EQ(ComputeDestinationPassCount(256), 1U);
  EXPECT_EQ(ComputeDestinationPassCount(257), 2U);
  EXPECT_EQ(ComputeDestinationPassCount(1U << 20U), 3U);
  EXPECT_EQ(ComputeDestinationPassCount(std::numeric_limits<std::uint32_t>::max()), 4U);
}

TEST(BuildPrefixSumLayers, LayerEntries_AgreeWithTheOrderBuffersSize) {
  for (const std::size_t count : {1U, 1024U, 1025U, 1048576U, 1048577U, 4294967295U}) {
    const PrefixSumLayers layers = BuildPrefixSumLayers(count);
    std::size_t entry_total = 0;
    for (int layer = 1; layer <= layers.top_layer; ++layer) {
      EXPECT_EQ(layers.first_entry[layer], entry_total) << "count " << count;
      entry_total += layers.entry_count[layer];
    }
    EXPECT_EQ(entry_total, ComputePrefixSumLayerEntryCount(count)) << "count " << count;
    EXPECT_EQ(layers.entry_count[layers.top_layer], 1U) << "count " << count;
  }
}

TEST(ComputeOrderBuffersSize, OnGpu_HoldsTheAreasTheStepsUse) {
  // The GPU areas of base/stable_order_cpu.cc: the second sequence, the chunk counts and the layer totals of the
  // largest prefix sum, each at a multiple of 64 bytes, and the destination flag.
  const OrderBuffersSize size = ComputeOrderBuffersSize(RunOn::kGpu);
  const auto align = [](std::size_t byte_count) {
    return (byte_count + 63) / 64 * 64;
  };
  for (const std::size_t max_count : {1U, 1025U, 70001U, 1000003U}) {
    const std::size_t chunk_count_entries = std::size_t{kDigitValueCount} * ComputeChunkCount(max_count);
    const std::size_t largest_prefix_sum = std::max(max_count, chunk_count_entries);
    const std::size_t needed =
        align(align(sizeof(std::uint32_t) * max_count) + sizeof(std::uint32_t) * chunk_count_entries)
        + sizeof(std::uint64_t) * ComputePrefixSumLayerEntryCount(largest_prefix_sum) + 64 + sizeof(std::uint32_t);
    EXPECT_LE(needed, size.bytes_per_item * max_count + size.fixed_bytes) << "max_count " << max_count;
  }
}

}  // namespace
}  // namespace fluidity
