#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/stable_order_cases.h"

namespace fluidity {
namespace {

// The item counts of the reference checks: empty, tiny, around one digit's 256 values, and large.
constexpr std::array<std::size_t, 8> kReferenceCounts = {0, 1, 2, 255, 256, 257, 65537, 1000003};

TEST(BuildOrderBuffers, IntoRunMemory_SameResultsAsPlainForm) {
  const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(1000000);
  const Buffer<std::uint64_t> key_buffer = BuildBufferHolding(RunOn::kCpu, keys);
  const OrderBuffersSize size = ComputeOrderBuffersSize(RunOn::kCpu);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(
      RunOn::kCpu,
      ComputeRunMemoryPieceBytes((std::size_t{size.bytes_per_item} * keys.size()) + size.fixed_bytes)
  );
  ASSERT_TRUE(run_memory.has_value());
  std::expected<OrderBuffers, BufferError> piece_buffers = BuildOrderBuffers(RunOn::kCpu, keys.size(), *run_memory);
  std::expected<OrderBuffers, BufferError> own_buffers = BuildOrderBuffers(RunOn::kCpu, keys.size());
  ASSERT_TRUE(piece_buffers.has_value() && own_buffers.has_value());
  EXPECT_EQ(run_memory->unused_bytes(), 0U);
  Buffer<std::uint32_t> order_from_piece = BuildCpuBufferFilledWith(keys.size(), 0);
  Buffer<std::uint32_t> order_from_own = BuildCpuBufferFilledWith(keys.size(), 0);
  ASSERT_TRUE(BuildStableOrder(key_buffer, keys.size(), nullptr, order_from_piece, *piece_buffers).has_value());
  ASSERT_TRUE(BuildStableOrder(key_buffer, keys.size(), nullptr, order_from_own, *own_buffers).has_value());
  EXPECT_EQ(ReadAll(order_from_piece), ReadAll(order_from_own));
}

TEST(BuildStableOrder, KeysWithManyTies_EqualsStdStableSort) {
  for (const std::size_t count : kReferenceCounts) {
    const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(count);
    EXPECT_EQ(ComputeOrderOnCpu(keys), ComputeReferenceOrder(keys, {})) << "count " << count;
  }
}

TEST(BuildStableOrder, KeysDifferingOnlyInTopByte_EqualsStdStableSort) {
  for (const std::size_t count : kReferenceCounts) {
    const std::vector<std::uint64_t> keys = BuildKeysDifferingInTopByte(count);
    EXPECT_EQ(ComputeOrderOnCpu(keys), ComputeReferenceOrder(keys, {})) << "count " << count;
  }
}

TEST(BuildStableOrder, AllKeysEqual_IsIndexOrder) {
  for (const std::size_t count : kReferenceCounts) {
    const std::vector<std::uint64_t> keys(count, 0x0123456789ABCDEFULL);
    EXPECT_EQ(ComputeOrderOnCpu(keys), ComputeReferenceOrder(keys, {})) << "count " << count;
  }
}

TEST(BuildStableOrder, KeysAlreadySorted_IsIndexOrder) {
  for (const std::size_t count : kReferenceCounts) {
    std::vector<std::uint64_t> keys(count);
    for (std::size_t index = 0; index < count; ++index) {
      keys[index] = std::uint64_t{index} * 0x0000010000000101ULL;
    }
    EXPECT_EQ(ComputeOrderOnCpu(keys), ComputeReferenceOrder(keys, {})) << "count " << count;
  }
}

TEST(BuildStableOrder, KeysReverseSorted_EqualsStdStableSort) {
  for (const std::size_t count : kReferenceCounts) {
    std::vector<std::uint64_t> keys(count);
    for (std::size_t index = 0; index < count; ++index) {
      keys[index] = std::uint64_t{count - index} * 0x0000010000000101ULL;
    }
    EXPECT_EQ(ComputeOrderOnCpu(keys), ComputeReferenceOrder(keys, {})) << "count " << count;
  }
}

TEST(BuildGroupsByDestination, ManyItems_EqualsStdStableSortByDestination) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  constexpr std::size_t kCount = 100003;
  constexpr std::uint32_t kDestinationCount = 70001;
  std::vector<std::uint32_t> destinations(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    destinations[index] = static_cast<std::uint32_t>((index * 7919) % kDestinationCount);
  }
  const Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kCpu, destinations);
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(kDestinationCount + 1, 0);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(kCount, 0);
  ASSERT_TRUE(
      BuildGroupsByDestination(destination, kCount, kDestinationCount, first_of_group, item_order, *buffers).has_value()
  );

  const std::vector<std::uint64_t> keys(destinations.begin(), destinations.end());
  EXPECT_EQ(ReadAll(item_order), ComputeReferenceOrder(keys, {}));
  // The start of a group is the number of destinations below it: the first place not below it in sorted order.
  std::vector<std::uint32_t> sorted_destinations = destinations;
  std::ranges::sort(sorted_destinations);
  std::vector<std::uint32_t> expected_starts(kDestinationCount + 1);
  for (std::uint32_t group = 0; group <= kDestinationCount; ++group) {
    const auto first_not_below = std::ranges::partition_point(sorted_destinations, [group](std::uint32_t value) {
      return value < group;
    });
    expected_starts[group] = static_cast<std::uint32_t>(first_not_below - sorted_destinations.begin());
  }
  EXPECT_EQ(ReadAll(first_of_group), expected_starts);
}

}  // namespace
}  // namespace fluidity
