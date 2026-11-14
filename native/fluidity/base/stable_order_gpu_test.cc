// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/base/test_support.h"

namespace fluidity {
namespace {

// The item counts compared with the CPU: empty, tiny, around one digit's 256 values, around one GPU chunk of 1,024
// places, and large.
constexpr std::array<std::size_t, 11> kComparedCounts = {0, 1, 2, 255, 256, 257, 1023, 1024, 1025, 65537, 1000003};

// A buffer on `run_on` of `size` copies of `value`.
Buffer<std::uint32_t> BuildBufferFilledWith(RunOn run_on, std::size_t size, std::uint32_t value) {
  const std::vector<std::uint32_t> values(size, value);
  return BuildBufferHolding(run_on, values);
}

// BuildStableOrder of `keys` on `run_on`, with no previous order and order buffers built for keys.size() items.
// Fails the calling test when a call fails.
std::vector<std::uint32_t> ComputeOrderOn(RunOn run_on, const std::vector<std::uint64_t>& keys) {
  const Buffer<std::uint64_t> key_buffer = BuildBufferHolding(run_on, keys);
  Buffer<std::uint32_t> order = BuildBufferFilledWith(run_on, keys.size(), 0);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(run_on, keys.size());
  EXPECT_TRUE(buffers.has_value());
  if (!buffers.has_value()) {
    return {};
  }
  const std::expected<void, BufferError> built = BuildStableOrder(key_buffer, keys.size(), nullptr, order, *buffers);
  EXPECT_TRUE(built.has_value()) << ToName(built.error());
  return ReadAll(order);
}

// The order of two sorts on `run_on`, both with one set of order buffers: by `minor_keys`, then by `major_keys` with
// the first order as previous order. Fails the calling test when a call fails.
std::vector<std::uint32_t> ComputeTwoPassOrderOn(
    RunOn run_on,
    const std::vector<std::uint64_t>& major_keys,
    const std::vector<std::uint64_t>& minor_keys
) {
  const std::size_t count = major_keys.size();
  const Buffer<std::uint64_t> major_buffer = BuildBufferHolding(run_on, major_keys);
  const Buffer<std::uint64_t> minor_buffer = BuildBufferHolding(run_on, minor_keys);
  Buffer<std::uint32_t> minor_order = BuildBufferFilledWith(run_on, count, 0);
  Buffer<std::uint32_t> order = BuildBufferFilledWith(run_on, count, 0);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(run_on, count);
  EXPECT_TRUE(buffers.has_value());
  if (!buffers.has_value()) {
    return {};
  }
  EXPECT_TRUE(BuildStableOrder(minor_buffer, count, nullptr, minor_order, *buffers).has_value());
  EXPECT_TRUE(BuildStableOrder(major_buffer, count, &minor_order, order, *buffers).has_value());
  return ReadAll(order);
}

// The groups of `destinations` on `run_on`, with order buffers built for destinations.size() items: item_order, then
// first_of_group.
struct Groups {
  std::vector<std::uint32_t> item_order;
  std::vector<std::uint32_t> first_of_group;
};

Groups ComputeGroupsOn(RunOn run_on, const std::vector<std::uint32_t>& destinations, std::uint32_t destination_count) {
  const Buffer<std::uint32_t> destination = BuildBufferHolding(run_on, destinations);
  Buffer<std::uint32_t> first_of_group = BuildBufferFilledWith(run_on, std::size_t{destination_count} + 1, 99);
  Buffer<std::uint32_t> item_order = BuildBufferFilledWith(run_on, destinations.size(), 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(run_on, destinations.size());
  EXPECT_TRUE(buffers.has_value());
  if (!buffers.has_value()) {
    return {};
  }
  const std::expected<void, BufferError> built = BuildGroupsByDestination(
      destination,
      destinations.size(),
      destination_count,
      first_of_group,
      item_order,
      *buffers
  );
  EXPECT_TRUE(built.has_value()) << ToName(built.error());
  return {.item_order = ReadAll(item_order), .first_of_group = ReadAll(first_of_group)};
}

// The exclusive prefix sums of `values` on `run_on`, total included. Fails the calling test when it fails.
std::vector<std::uint32_t> ComputePrefixSumOn(RunOn run_on, const std::vector<std::uint32_t>& values) {
  const Buffer<std::uint32_t> value_buffer = BuildBufferHolding(run_on, values);
  Buffer<std::uint32_t> sums = BuildBufferFilledWith(run_on, values.size() + 1, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(run_on, values.size());
  EXPECT_TRUE(buffers.has_value());
  if (!buffers.has_value()) {
    return {};
  }
  const std::expected<void, BufferError> computed =
      ComputeExclusivePrefixSum(value_buffer, values.size(), sums, *buffers);
  EXPECT_TRUE(computed.has_value()) << ToName(computed.error());
  return ReadAll(sums);
}

// `count` values from a fixed integer formula whose total stays below 2^32.
std::vector<std::uint32_t> BuildPrefixSumValues(std::size_t count) {
  std::vector<std::uint32_t> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = static_cast<std::uint32_t>((index * 7919) % 4001);
  }
  return values;
}

TEST(ComputeOrderBuffersSize, OnGpu_EqualsBytesBuildOrderBuffersAllocates) {
  const OrderBuffersSize size = ComputeOrderBuffersSize(RunOn::kGpu);
  for (const std::size_t max_count : {std::size_t{0}, std::size_t{1}, std::size_t{1025}, std::size_t{1000003}}) {
    const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kGpu, max_count);
    ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
    EXPECT_EQ(buffers->run_on(), RunOn::kGpu);
    EXPECT_EQ(buffers->byte_count(), size.bytes_per_item * max_count + size.fixed_bytes) << "max_count " << max_count;
  }
}

TEST(BuildStableOrder, OnGpu_EqualsCpuForAMillionKeysWithManyTies) {
  const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(1000003);
  const std::vector<std::uint32_t> on_gpu = ComputeOrderOn(RunOn::kGpu, keys);
  EXPECT_EQ(on_gpu, ComputeOrderOn(RunOn::kCpu, keys));
  WriteResultDigest("BuildStableOrder on the GPU, a million keys", std::span<const std::uint32_t>(on_gpu));
}

TEST(BuildStableOrder, OnGpuKeysWithManyTies_EqualsCpu) {
  for (const std::size_t count : kComparedCounts) {
    const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(count);
    EXPECT_EQ(ComputeOrderOn(RunOn::kGpu, keys), ComputeOrderOn(RunOn::kCpu, keys)) << "count " << count;
  }
}

TEST(BuildStableOrder, OnGpuKeysDifferingOnlyInTopByte_EqualsCpu) {
  for (const std::size_t count : kComparedCounts) {
    const std::vector<std::uint64_t> keys = BuildKeysDifferingInTopByte(count);
    EXPECT_EQ(ComputeOrderOn(RunOn::kGpu, keys), ComputeOrderOn(RunOn::kCpu, keys)) << "count " << count;
  }
}

TEST(BuildStableOrder, OnGpuAllKeysEqual_EqualsCpu) {
  for (const std::size_t count : kComparedCounts) {
    const std::vector<std::uint64_t> keys(count, 0x0123456789ABCDEFULL);
    EXPECT_EQ(ComputeOrderOn(RunOn::kGpu, keys), ComputeOrderOn(RunOn::kCpu, keys)) << "count " << count;
  }
}

TEST(BuildStableOrder, OnGpuKeysAlreadySorted_EqualsCpu) {
  for (const std::size_t count : kComparedCounts) {
    std::vector<std::uint64_t> keys(count);
    for (std::size_t index = 0; index < count; ++index) {
      keys[index] = std::uint64_t{index} * 0x0000010000000101ULL;
    }
    EXPECT_EQ(ComputeOrderOn(RunOn::kGpu, keys), ComputeOrderOn(RunOn::kCpu, keys)) << "count " << count;
  }
}

TEST(BuildStableOrder, OnGpuKeysReverseSorted_EqualsCpu) {
  for (const std::size_t count : kComparedCounts) {
    std::vector<std::uint64_t> keys(count);
    for (std::size_t index = 0; index < count; ++index) {
      keys[index] = std::uint64_t{count - index} * 0x0000010000000101ULL;
    }
    EXPECT_EQ(ComputeOrderOn(RunOn::kGpu, keys), ComputeOrderOn(RunOn::kCpu, keys)) << "count " << count;
  }
}

TEST(BuildStableOrder, OnGpuTwoPasses_EqualsCpu) {
  constexpr std::size_t kCount = 200003;
  std::vector<std::uint64_t> major_keys(kCount);
  std::vector<std::uint64_t> minor_keys(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    major_keys[index] = (index * 37) % 101;
    minor_keys[index] = ((index * 7919) % 499) << 20U;
  }
  EXPECT_EQ(
      ComputeTwoPassOrderOn(RunOn::kGpu, major_keys, minor_keys),
      ComputeTwoPassOrderOn(RunOn::kCpu, major_keys, minor_keys)
  );
}

TEST(BuildStableOrder, OnGpuWithPreviousOrder_EqualsCpu) {
  constexpr std::size_t kCount = 1025;
  const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(kCount);
  // A permutation from a fixed formula: 7919 is prime and does not divide 1,025, so index * 7919 mod 1,025 hits
  // each index once.
  std::vector<std::uint32_t> previous_order(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    previous_order[index] = static_cast<std::uint32_t>((index * 7919) % kCount);
  }
  std::array<std::vector<std::uint32_t>, 2> orders;
  for (const RunOn run_on : {RunOn::kCpu, RunOn::kGpu}) {
    const Buffer<std::uint64_t> key_buffer = BuildBufferHolding(run_on, keys);
    const Buffer<std::uint32_t> previous_order_buffer = BuildBufferHolding(run_on, previous_order);
    Buffer<std::uint32_t> order = BuildBufferFilledWith(run_on, kCount, 0);
    std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(run_on, kCount);
    ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
    ASSERT_TRUE(BuildStableOrder(key_buffer, kCount, &previous_order_buffer, order, *buffers).has_value());
    orders[run_on == RunOn::kGpu ? 1 : 0] = ReadAll(order);
  }
  EXPECT_EQ(orders[1], orders[0]);
}

TEST(BuildStableOrder, OnGpuCountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<std::uint64_t> keys = BuildBufferHolding(RunOn::kGpu, std::array<std::uint64_t, 3>{3, 2, 1});
  Buffer<std::uint32_t> order = BuildBufferFilledWith(RunOn::kGpu, 3, 0);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kGpu, 2);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> built = BuildStableOrder(keys, 3, nullptr, order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
}

TEST(ComputeExclusivePrefixSum, OnGpu_EqualsCpuForAMillionValues) {
  const std::vector<std::uint32_t> values = BuildPrefixSumValues(1000000);
  const std::vector<std::uint32_t> on_gpu = ComputePrefixSumOn(RunOn::kGpu, values);
  EXPECT_EQ(on_gpu, ComputePrefixSumOn(RunOn::kCpu, values));
  WriteResultDigest("ComputeExclusivePrefixSum on the GPU, a million values", std::span<const std::uint32_t>(on_gpu));
}

TEST(ComputeExclusivePrefixSum, OnGpuEveryChunkBoundary_EqualsCpu) {
  for (const std::size_t count : kComparedCounts) {
    const std::vector<std::uint32_t> values = BuildPrefixSumValues(count);
    EXPECT_EQ(ComputePrefixSumOn(RunOn::kGpu, values), ComputePrefixSumOn(RunOn::kCpu, values)) << "count " << count;
  }
}

TEST(ComputeExclusivePrefixSum, OnGpuInPlace_EqualsSeparateBuffers) {
  std::vector<std::uint32_t> input = BuildPrefixSumValues(1000003);
  input.push_back(0);  // The entry for the total.
  const std::size_t count = input.size() - 1;
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kGpu, count);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<std::uint32_t> values = BuildBufferHolding(RunOn::kGpu, input);
  Buffer<std::uint32_t> sums = BuildBufferFilledWith(RunOn::kGpu, input.size(), 0);
  Buffer<std::uint32_t> in_place = BuildBufferHolding(RunOn::kGpu, input);
  ASSERT_TRUE(ComputeExclusivePrefixSum(values, count, sums, *buffers).has_value());
  ASSERT_TRUE(ComputeExclusivePrefixSum(in_place, count, in_place, *buffers).has_value());
  EXPECT_EQ(ReadAll(in_place), ReadAll(sums));
}

TEST(ComputeExclusivePrefixSum, OnGpuTotalAboveUint32Max_IsSizeMismatch) {
  const std::vector<std::uint32_t> input(3000, 0x00200000U);  // 3000 * 2^21 is above 2^32.
  const Buffer<std::uint32_t> values = BuildBufferHolding(RunOn::kGpu, input);
  Buffer<std::uint32_t> sums = BuildBufferFilledWith(RunOn::kGpu, input.size() + 1, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kGpu, input.size());
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> computed = ComputeExclusivePrefixSum(values, input.size(), sums, *buffers);
  ASSERT_FALSE(computed.has_value());
  EXPECT_EQ(computed.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(sums), std::vector<std::uint32_t>(input.size() + 1, 99));
}

TEST(ComputeExclusivePrefixSum, OnGpuTotalExactlyUint32Max_Succeeds) {
  // 2,047 values of 2^21 and one of 2^21 - 1, spread over two chunks: the total is 2^32 - 1.
  std::vector<std::uint32_t> input(2048, 0x00200000U);
  input.back() = 0x001FFFFFU;
  const std::vector<std::uint32_t> sums = ComputePrefixSumOn(RunOn::kGpu, input);
  ASSERT_EQ(sums.size(), input.size() + 1);
  EXPECT_EQ(sums.back(), 0xFFFFFFFFU);
  EXPECT_EQ(sums, ComputePrefixSumOn(RunOn::kCpu, input));
}

TEST(ComputeExclusivePrefixSum, OnGpuCountZero_WritesZeroTotal) {
  EXPECT_EQ(ComputePrefixSumOn(RunOn::kGpu, {}), (std::vector<std::uint32_t>{0}));
}

TEST(BuildGroupsByDestination, OnGpu_EqualsCpu) {
  constexpr std::size_t kCount = 1000003;
  for (const std::uint32_t destination_count : {1U, 3U, 256U, 257U, 70001U, 1U << 20U}) {
    std::vector<std::uint32_t> destinations(kCount);
    for (std::size_t index = 0; index < kCount; ++index) {
      destinations[index] = static_cast<std::uint32_t>((index * 7919) % destination_count);
    }
    const Groups on_gpu = ComputeGroupsOn(RunOn::kGpu, destinations, destination_count);
    const Groups on_cpu = ComputeGroupsOn(RunOn::kCpu, destinations, destination_count);
    EXPECT_EQ(on_gpu.item_order, on_cpu.item_order) << "destination_count " << destination_count;
    EXPECT_EQ(on_gpu.first_of_group, on_cpu.first_of_group) << "destination_count " << destination_count;
  }
}

TEST(BuildGroupsByDestination, OnGpuTwice_GivesIdenticalBytes) {
  constexpr std::size_t kCount = 1000003;
  constexpr std::uint32_t kDestinationCount = 70001;
  std::vector<std::uint32_t> destinations(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    destinations[index] = static_cast<std::uint32_t>((index * 7919) % kDestinationCount);
  }
  const Groups first = ComputeGroupsOn(RunOn::kGpu, destinations, kDestinationCount);
  const Groups second = ComputeGroupsOn(RunOn::kGpu, destinations, kDestinationCount);
  EXPECT_EQ(first.item_order, second.item_order);
  EXPECT_EQ(first.first_of_group, second.first_of_group);
  WriteResultDigest(
      "BuildGroupsByDestination on the GPU, item order",
      std::span<const std::uint32_t>(first.item_order)
  );
  WriteResultDigest(
      "BuildGroupsByDestination on the GPU, first of group",
      std::span<const std::uint32_t>(first.first_of_group)
  );
}

TEST(BuildGroupsByDestination, OnGpuDestinationOutOfRange_IsSizeMismatch) {
  const Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kGpu, std::array<std::uint32_t, 3>{0, 3, 1});
  Buffer<std::uint32_t> first_of_group = BuildBufferFilledWith(RunOn::kGpu, 4, 99);
  Buffer<std::uint32_t> item_order = BuildBufferFilledWith(RunOn::kGpu, 3, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kGpu, 3);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<void, BufferError> built =
      BuildGroupsByDestination(destination, 3, 3, first_of_group, item_order, *buffers);
  ASSERT_FALSE(built.has_value());
  EXPECT_EQ(built.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(first_of_group), (std::vector<std::uint32_t>{99, 99, 99, 99}));
  EXPECT_EQ(ReadAll(item_order), (std::vector<std::uint32_t>{99, 99, 99}));
}

TEST(BuildGroupsByDestination, OnGpuNoItems_WritesZeroStarts) {
  const Groups groups = ComputeGroupsOn(RunOn::kGpu, {}, 3);
  EXPECT_EQ(groups.first_of_group, (std::vector<std::uint32_t>{0, 0, 0, 0}));
}

TEST(BuildGroupsByDestination, OnGpuBuffersForMoreItems_EqualsCpu) {
  constexpr std::size_t kCount = 100003;
  constexpr std::uint32_t kDestinationCount = 257;
  std::vector<std::uint32_t> destinations(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    destinations[index] = static_cast<std::uint32_t>((index * 7919) % kDestinationCount);
  }
  const Buffer<std::uint32_t> destination = BuildBufferHolding(RunOn::kGpu, destinations);
  Buffer<std::uint32_t> first_of_group = BuildBufferFilledWith(RunOn::kGpu, kDestinationCount + 1, 99);
  Buffer<std::uint32_t> item_order = BuildBufferFilledWith(RunOn::kGpu, kCount, 99);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kGpu, 2 * kCount);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  ASSERT_TRUE(
      BuildGroupsByDestination(destination, kCount, kDestinationCount, first_of_group, item_order, *buffers).has_value()
  );
  const Groups on_cpu = ComputeGroupsOn(RunOn::kCpu, destinations, kDestinationCount);
  EXPECT_EQ(ReadAll(item_order), on_cpu.item_order);
  EXPECT_EQ(ReadAll(first_of_group), on_cpu.first_of_group);
}

}  // namespace
}  // namespace fluidity
