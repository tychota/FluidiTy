#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/stable_order_cases.h"

namespace fluidity {
namespace {

TEST(BuildStableOrder, BuffersForMoreItems_GiveTheSameOrder) {
  const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(5000);
  const Buffer<std::uint64_t> key_buffer = BuildBufferHolding(RunOn::kCpu, keys);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 6000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(keys.size(), 0);
  const std::expected<void, BufferError> built = BuildStableOrder(key_buffer, keys.size(), nullptr, order, *buffers);
  ASSERT_TRUE(built.has_value()) << ToName(built.error());
  EXPECT_EQ(ReadAll(order), ComputeOrderOnCpu(keys));
}

TEST(BuildStableOrder, BuffersUsedTwice_GiveTheSameOrder) {
  const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(3000);
  const Buffer<std::uint64_t> key_buffer = BuildBufferHolding(RunOn::kCpu, keys);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, keys.size());
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  Buffer<std::uint32_t> first_order = BuildCpuBufferFilledWith(keys.size(), 0);
  Buffer<std::uint32_t> second_order = BuildCpuBufferFilledWith(keys.size(), 0);
  ASSERT_TRUE(BuildStableOrder(key_buffer, keys.size(), nullptr, first_order, *buffers).has_value());
  ASSERT_TRUE(BuildStableOrder(key_buffer, keys.size(), nullptr, second_order, *buffers).has_value());
  EXPECT_EQ(ReadAll(first_order), ReadAll(second_order));
}

TEST(BuildStableOrder, TwoPasses_SortByMajorThenMinorKey) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  constexpr std::size_t kCount = 20000;
  std::vector<std::uint64_t> major_keys(kCount);
  std::vector<std::uint64_t> minor_keys(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    major_keys[index] = (index * 37) % 101;
    minor_keys[index] = ((index * 7919) % 499) << 20U;
  }
  const Buffer<std::uint64_t> major_buffer = BuildBufferHolding(RunOn::kCpu, major_keys);
  const Buffer<std::uint64_t> minor_buffer = BuildBufferHolding(RunOn::kCpu, minor_keys);
  Buffer<std::uint32_t> minor_order = BuildCpuBufferFilledWith(kCount, 0);
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(kCount, 0);
  ASSERT_TRUE(BuildStableOrder(minor_buffer, kCount, nullptr, minor_order, *buffers).has_value());
  ASSERT_TRUE(BuildStableOrder(major_buffer, kCount, &minor_order, order, *buffers).has_value());

  std::vector<std::uint32_t> expected(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    expected[index] = static_cast<std::uint32_t>(index);
  }
  std::ranges::stable_sort(expected, [&](std::uint32_t first, std::uint32_t second) {
    return std::pair(major_keys[first], minor_keys[first]) < std::pair(major_keys[second], minor_keys[second]);
  });
  EXPECT_EQ(ReadAll(order), expected);
}

TEST(BuildStableOrder, KeysWithManyTiesFromAPreviousOrder_EqualsStdStableSort) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  constexpr std::size_t kCount = 65537;
  const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(kCount);
  // A permutation from a fixed formula: 7919 and 65,537 are both prime, so index * 7919 mod 65,537 hits each index
  // once.
  std::vector<std::uint32_t> previous_order(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    previous_order[index] = static_cast<std::uint32_t>((index * 7919) % kCount);
  }
  const Buffer<std::uint64_t> key_buffer = BuildBufferHolding(RunOn::kCpu, keys);
  const Buffer<std::uint32_t> previous_order_buffer = BuildBufferHolding(RunOn::kCpu, previous_order);
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(kCount, 0);
  ASSERT_TRUE(BuildStableOrder(key_buffer, kCount, &previous_order_buffer, order, *buffers).has_value());
  EXPECT_EQ(ReadAll(order), ComputeReferenceOrder(keys, previous_order));
}

TEST(ComputeExclusivePrefixSum, InPlace_EqualsSeparateBuffers) {
  std::vector<std::uint32_t> input(100001);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint32_t>((index * 7919) % 4001);
  }
  input.push_back(0);  // The entry for the total.
  const std::size_t count = input.size() - 1;
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, count);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint32_t> values = BuildBufferHolding(RunOn::kCpu, input);
  Buffer<std::uint32_t> sums = BuildCpuBufferFilledWith(input.size(), 0);
  Buffer<std::uint32_t> in_place = BuildBufferHolding(RunOn::kCpu, input);
  ASSERT_TRUE(ComputeExclusivePrefixSum(values, count, sums, *buffers).has_value());
  ASSERT_TRUE(ComputeExclusivePrefixSum(in_place, count, in_place, *buffers).has_value());
  EXPECT_EQ(ReadAll(in_place), ReadAll(sums));
}

TEST(BuildGroupsByDestination, BuffersForMoreItems_GiveTheSameGroups) {
  constexpr std::size_t kCount = 10007;
  constexpr std::uint32_t kDestinationCount = 301;
  std::vector<std::uint32_t> destinations(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    destinations[index] = static_cast<std::uint32_t>((index * 7919) % kDestinationCount);
  }
  const Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kCpu, destinations);
  std::expected<OrderBuffers, BufferError> larger_buffers = BuildOrderBuffers(RunOn::kCpu, 20000);
  std::expected<OrderBuffers, BufferError> exact_buffers = BuildOrderBuffers(RunOn::kCpu, kCount);
  ASSERT_TRUE(larger_buffers.has_value()) << ToName(larger_buffers.error());
  ASSERT_TRUE(exact_buffers.has_value()) << ToName(exact_buffers.error());
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(kDestinationCount + 1, 0);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(kCount, 0);
  ASSERT_TRUE(
      BuildGroupsByDestination(destination, kCount, kDestinationCount, first_of_group, item_order, *larger_buffers)
          .has_value()
  );
  Buffer<std::uint32_t> exact_first_of_group = BuildCpuBufferFilledWith(kDestinationCount + 1, 0);
  Buffer<std::uint32_t> exact_item_order = BuildCpuBufferFilledWith(kCount, 0);
  ASSERT_TRUE(BuildGroupsByDestination(
                  destination,
                  kCount,
                  kDestinationCount,
                  exact_first_of_group,
                  exact_item_order,
                  *exact_buffers
  )
                  .has_value());
  EXPECT_EQ(ReadAll(item_order), ReadAll(exact_item_order));
  EXPECT_EQ(ReadAll(first_of_group), ReadAll(exact_first_of_group));
}

}  // namespace
}  // namespace fluidity
