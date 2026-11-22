#include "fluidity/base/stable_order.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/stable_order_cases.h"

namespace fluidity {
namespace {

TEST(BuildOrderBuffers, MaxCountZero_Succeeds) {
  const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 0);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  EXPECT_EQ(buffers->max_count(), 0U);
  EXPECT_EQ(buffers->run_on(), RunOn::kCpu);
}

TEST(BuildOrderBuffers, MaxCountAboveUint32Max_IsSizeMismatch) {
  const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, std::size_t{1} << 32U);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error(), BufferError::kSizeMismatch);
}

TEST(ComputeOrderBuffersSize, EqualsBytesBuildOrderBuffersAllocates) {
  const OrderBuffersSize size = ComputeOrderBuffersSize(RunOn::kCpu);
  EXPECT_EQ(size.bytes_per_item, 4U);
  EXPECT_EQ(size.fixed_bytes, 8192U);
  for (const std::size_t max_count : {std::size_t{0}, std::size_t{1}, std::size_t{1000}, std::size_t{65537}}) {
    const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, max_count);
    ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
    EXPECT_EQ(buffers->byte_count(), size.bytes_per_item * max_count + size.fixed_bytes) << "max_count " << max_count;
  }
}

TEST(ComputeOrderBuffersSize, OnGpu_IsSixBytesPerItem) {
  EXPECT_EQ(ComputeOrderBuffersSize(RunOn::kGpu).bytes_per_item, 6U);
}

TEST(ComputeOrderBuffersSize, OnGpu_Has4096FixedBytes) {
  EXPECT_EQ(ComputeOrderBuffersSize(RunOn::kGpu).fixed_bytes, 4096U);
}

TEST(OrderBuffers, MovedFrom_HasMaxCountZero) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 10);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const OrderBuffers moved_to = std::move(*buffers);
  EXPECT_EQ(moved_to.max_count(), 10U);
  // What moved-from buffers hold is part of the contract, so reading it after the move is the point of the test.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(buffers->max_count(), 0U);
}

TEST(OrderBuffers, MovedFrom_HoldsNoBytes) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 10);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const OrderBuffers moved_to = std::move(*buffers);
  EXPECT_GT(moved_to.byte_count(), 0U);
  // What moved-from buffers hold is part of the contract, so reading it after the move is the point of the test.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(buffers->byte_count(), 0U);
}

TEST(BuildOrderBuffers, IntoRunMemory_TakesOnePieceOfTheSize) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 65536);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const OrderBuffersSize size = ComputeOrderBuffersSize(RunOn::kCpu);
  // 4 x 1000 + 8192 = 12,192 bytes, one piece of 12,288.
  const std::size_t expected_bytes =
      ComputeRunMemoryPieceBytes((std::size_t{size.bytes_per_item} * 1000) + size.fixed_bytes);
  EXPECT_EQ(expected_bytes, 12288U);
  EXPECT_EQ(run_memory->handed_out_bytes(), expected_bytes);
  EXPECT_EQ(run_memory->live_piece_count(), 1U);
  EXPECT_EQ(buffers->byte_count(), 12192U);
}

TEST(BuildOrderBuffers, IntoRunMemory_HandsOutExactlyTheEstimate) {
  constexpr std::size_t kMaxCount = 1000;
  const OrderBuffersSize size = ComputeOrderBuffersSize(RunOn::kCpu);
  // The one entry the docstring of ComputeOrderBuffersSize asks for: the exact bytes at max_count.
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "order buffers",
          .counted_per = CountedPer::kGivenCount,
          .given_count = (std::uint64_t{size.bytes_per_item} * kMaxCount) + size.fixed_bytes,
          .bytes_each = 1,
      },
  };
  // A given-count entry reads no scene count.
  const MemoryCounts no_counts;
  std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(RunOn::kCpu, ComputeRunMemoryBytes(estimate, no_counts, MemoryWhere::kRunOn));
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, kMaxCount, *run_memory);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes());
}

TEST(BuildOrderBuffers, IntoRunMemoryMaxCountAboveUint32Max_IsSizeMismatchAndTakesNothing) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 4096);
  ASSERT_TRUE(run_memory.has_value());
  const std::size_t too_many = std::size_t{1} << 32U;
  const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, too_many, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

TEST(BuildOrderBuffers, IntoRunMemoryTooSmall_IsRunMemoryTooSmall) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 4096);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error(), BufferError::kRunMemoryTooSmall);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

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

TEST(BuildStableOrder, CountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 3>{3, 2, 1});
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 2);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(3, 7);
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 3, nullptr, order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(order), (std::vector<std::uint32_t>{7, 7, 7}));
}

TEST(BuildStableOrder, DistinctKeys_OrdersByAscendingKey) {
  const std::vector<std::uint64_t> keys = {40, 0xFFFFFFFFFFFFFFFFULL, 7, 0x0100000000000000ULL, 0, 300};
  EXPECT_EQ(ComputeOrderOnCpu(keys), (std::vector<std::uint32_t>{4, 2, 0, 5, 3, 1}));
}

TEST(BuildStableOrder, EqualKeys_KeepIndexOrder) {
  const std::vector<std::uint64_t> keys = {5, 3, 5, 1, 3, 5};
  EXPECT_EQ(ComputeOrderOnCpu(keys), (std::vector<std::uint32_t>{3, 1, 4, 0, 2, 5}));
}

TEST(BuildStableOrder, EqualKeysWithPreviousOrder_KeepPreviousOrder) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 4>{5, 3, 5, 1});
  const Buffer<std::uint32_t> previous_order =
      BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 4>{2, 1, 0, 3});
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(4, 0);
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 4, &previous_order, order, *buffers);
  ASSERT_TRUE(built.has_value()) << ToName(built.error());
  EXPECT_EQ(ReadAll(order), (std::vector<std::uint32_t>{3, 1, 2, 0}));
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

TEST(BuildStableOrder, CountZero_Succeeds) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 2>{2, 1});
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(2, 9);
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 0, nullptr, order, *buffers);
  ASSERT_TRUE(built.has_value()) << ToName(built.error());
  EXPECT_EQ(ReadAll(order), (std::vector<std::uint32_t>{9, 9}));
}

TEST(BuildStableOrder, OrderBufferTooSmall_IsSizeMismatch) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 3>{3, 2, 1});
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(2, 9);
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 3, nullptr, order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(order), (std::vector<std::uint32_t>{9, 9}));
}

TEST(BuildStableOrder, KeysBufferTooSmall_IsSizeMismatch) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 2>{2, 1});
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(3, 9);
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 3, nullptr, order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
}

TEST(BuildStableOrder, PreviousOrderTooSmall_IsSizeMismatch) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 3>{3, 2, 1});
  const Buffer<std::uint32_t> previous_order = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 2>{1, 0});
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(3, 9);
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 3, &previous_order, order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
}

TEST(BuildStableOrder, OrderIsPreviousOrder_IsSizeMismatch) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 3>{3, 2, 1});
  Buffer<std::uint32_t> order = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 3>{2, 1, 0});
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 3, &order, order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(order), (std::vector<std::uint32_t>{2, 1, 0}));
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

TEST(BuildStableOrder, AllKeysEqualWithPreviousOrder_IsPreviousOrder) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kCpu, std::array<std::uint64_t, 4>{8, 8, 8, 8});
  const Buffer<std::uint32_t> previous_order =
      BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 4>{3, 0, 2, 1});
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(4, 0);
  ASSERT_TRUE(BuildStableOrder(keys, 4, &previous_order, order, *buffers).has_value());
  EXPECT_EQ(ReadAll(order), (std::vector<std::uint32_t>{3, 0, 2, 1}));
}

TEST(ComputeExclusivePrefixSum, FiveValues_GivesTheExclusiveRunningSums) {
  const Buffer<std::uint32_t> values = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 5>{3, 0, 2, 5, 1});
  Buffer<std::uint32_t> sums = BuildCpuBufferFilledWith(6, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 5);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> computed = ComputeExclusivePrefixSum(values, 5, sums, *buffers);
  ASSERT_TRUE(computed.has_value()) << ToName(computed.error());
  EXPECT_EQ(ReadAll(sums), (std::vector<std::uint32_t>{0, 3, 3, 5, 10, 11}));
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

TEST(ComputeExclusivePrefixSum, CountZero_WritesZeroTotal) {
  const Buffer<std::uint32_t> values = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 1>{4});
  Buffer<std::uint32_t> sums = BuildCpuBufferFilledWith(2, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 0);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> computed = ComputeExclusivePrefixSum(values, 0, sums, *buffers);
  ASSERT_TRUE(computed.has_value()) << ToName(computed.error());
  EXPECT_EQ(ReadAll(sums), (std::vector<std::uint32_t>{0, 99}));
}

TEST(ComputeExclusivePrefixSum, TotalAboveUint32Max_IsSizeMismatch) {
  const Buffer<std::uint32_t> values =
      BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 3>{0x80000000U, 0x7FFFFFFFU, 1});
  Buffer<std::uint32_t> sums = BuildCpuBufferFilledWith(4, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 3);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> computed = ComputeExclusivePrefixSum(values, 3, sums, *buffers);
  ASSERT_FALSE(computed.has_value());
  EXPECT_EQ(computed.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(sums), (std::vector<std::uint32_t>{99, 99, 99, 99}));
}

TEST(ComputeExclusivePrefixSum, TotalExactlyUint32Max_Succeeds) {
  const Buffer<std::uint32_t> values =
      BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 2>{0x80000000U, 0x7FFFFFFFU});
  Buffer<std::uint32_t> sums = BuildCpuBufferFilledWith(3, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 2);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  ASSERT_TRUE(ComputeExclusivePrefixSum(values, 2, sums, *buffers).has_value());
  EXPECT_EQ(ReadAll(sums), (std::vector<std::uint32_t>{0, 0x80000000U, 0xFFFFFFFFU}));
}

TEST(ComputeExclusivePrefixSum, SumsBufferTooSmall_IsSizeMismatch) {
  const Buffer<std::uint32_t> values = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 3>{1, 2, 3});
  Buffer<std::uint32_t> sums = BuildCpuBufferFilledWith(3, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 3);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> computed = ComputeExclusivePrefixSum(values, 3, sums, *buffers);
  ASSERT_FALSE(computed.has_value());
  EXPECT_EQ(computed.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(sums), (std::vector<std::uint32_t>{99, 99, 99}));
}

TEST(ComputeExclusivePrefixSum, CountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<std::uint32_t> values = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 3>{1, 2, 3});
  Buffer<std::uint32_t> sums = BuildCpuBufferFilledWith(4, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 2);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> computed = ComputeExclusivePrefixSum(values, 3, sums, *buffers);
  ASSERT_FALSE(computed.has_value());
  EXPECT_EQ(computed.error(), BufferError::kSizeMismatch);
}

TEST(BuildGroupsByDestination, ThreeDestinations_ListsEachGroupInAscendingItemIndex) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint32_t> destination =
      BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 7>{2, 0, 1, 2, 0, 2, 1});
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(4, 99);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(7, 99);
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 7, 3, first_of_group, item_order, *buffers);
  ASSERT_TRUE(built.has_value()) << ToName(built.error());
  EXPECT_EQ(ReadAll(item_order), (std::vector<std::uint32_t>{1, 4, 2, 6, 0, 3, 5}));
  EXPECT_EQ(ReadAll(first_of_group), (std::vector<std::uint32_t>{0, 2, 4, 7}));
}

TEST(BuildGroupsByDestination, EmptyDestination_HasEqualNeighbouringStarts) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint32_t> destination =
      BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 5>{2, 0, 2, 1, 0});
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(6, 99);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(5, 99);
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 5, 5, first_of_group, item_order, *buffers);
  ASSERT_TRUE(built.has_value()) << ToName(built.error());
  EXPECT_EQ(ReadAll(item_order), (std::vector<std::uint32_t>{1, 4, 3, 0, 2}));
  EXPECT_EQ(ReadAll(first_of_group), (std::vector<std::uint32_t>{0, 2, 3, 5, 5, 5}));
}

TEST(BuildGroupsByDestination, DestinationOutOfRange_IsSizeMismatch) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 3>{0, 3, 1});
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(4, 99);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(3, 99);
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 3, 3, first_of_group, item_order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(first_of_group), (std::vector<std::uint32_t>{99, 99, 99, 99}));
  EXPECT_EQ(ReadAll(item_order), (std::vector<std::uint32_t>{99, 99, 99}));
}

TEST(BuildGroupsByDestination, FirstOfGroupTooSmall_IsSizeMismatch) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 2>{0, 1});
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(2, 99);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(2, 99);
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 2, 2, first_of_group, item_order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
}

TEST(BuildGroupsByDestination, ItemOrderIsDestination_IsSizeMismatch) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 2>{1, 0});
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(3, 99);
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 2, 2, first_of_group, destination, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
}

TEST(BuildGroupsByDestination, NoItems_WritesZeroStarts) {
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 100003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint32_t> destination = BuildCpuBufferFilledWith(0, 0);
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(4, 99);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(0, 0);
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 0, 3, first_of_group, item_order, *buffers);
  ASSERT_TRUE(built.has_value()) << ToName(built.error());
  EXPECT_EQ(ReadAll(first_of_group), (std::vector<std::uint32_t>{0, 0, 0, 0}));
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

TEST(BuildGroupsByDestination, CountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kCpu, std::array<std::uint32_t, 3>{0, 1, 0});
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, 2);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  Buffer<std::uint32_t> first_of_group = BuildCpuBufferFilledWith(3, 99);
  Buffer<std::uint32_t> item_order = BuildCpuBufferFilledWith(3, 99);
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 3, 2, first_of_group, item_order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(item_order), (std::vector<std::uint32_t>{99, 99, 99}));
}

}  // namespace
}  // namespace fluidity
