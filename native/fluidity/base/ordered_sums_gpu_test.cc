// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/ordered_sums.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/test_support.h"
#include "fluidity/base/tree_sum.h"

namespace fluidity {
namespace {

// The sums of one set of values on the two processors.
struct SumOnBoth {
  double on_cpu = 0.0;
  double on_gpu = 0.0;
};

// The sum of `values` on the CPU and on the GPU, each with sum buffers built for `values.size()` values. Fails the
// calling test when a call fails.
SumOnBoth ComputeSumOnBoth(std::span<const double> values) {
  const Buffer<double> cpu_buffer = BuildBufferHolding(RunOn::kCpu, values);
  const Buffer<double> gpu_buffer = BuildBufferHolding(RunOn::kGpu, values);
  std::expected<SumBuffers, BufferError> cpu_buffers = BuildSumBuffers(RunOn::kCpu, values.size());
  std::expected<SumBuffers, BufferError> gpu_buffers = BuildSumBuffers(RunOn::kGpu, values.size());
  EXPECT_TRUE(cpu_buffers.has_value());
  EXPECT_TRUE(gpu_buffers.has_value());
  if (!cpu_buffers.has_value() || !gpu_buffers.has_value()) {
    return {.on_cpu = 0.0, .on_gpu = 1.0};
  }
  const std::expected<double, BufferError> cpu_sum = ComputeTreeSum(cpu_buffer, values.size(), *cpu_buffers);
  const std::expected<double, BufferError> gpu_sum = ComputeTreeSum(gpu_buffer, values.size(), *gpu_buffers);
  EXPECT_TRUE(cpu_sum.has_value());
  EXPECT_TRUE(gpu_sum.has_value());
  if (!cpu_sum.has_value() || !gpu_sum.has_value()) {
    return {.on_cpu = 0.0, .on_gpu = 1.0};
  }
  return {.on_cpu = *cpu_sum, .on_gpu = *gpu_sum};
}

TEST(ComputeTreeSum, OnGpu_EqualsCpuBitForBit) {
  for (const std::size_t count : {std::size_t{1000}, std::size_t{1000003}}) {
    const std::vector<double> values = BuildMixedValues(count);
    const auto [cpu_sum, gpu_sum] = ComputeSumOnBoth(values);
    EXPECT_EQ(ToBits(gpu_sum), ToBits(cpu_sum)) << "count " << count;
    const std::string name = "ComputeTreeSum on the GPU, " + std::to_string(count) + " mixed values";
    WriteResultDigest(name, std::span<const double>(&gpu_sum, 1));
  }
}

TEST(BuildSumBuffers, OnGpuIntoRunMemory_TakesOnePieceOfTheSize) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 65536);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 1000, *run_memory);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const SumBuffersSize size = ComputeSumBuffersSize(RunOn::kGpu);
  // 14 x 1000 + 128 = 14,128 bytes, one piece of 14,336.
  EXPECT_EQ(
      run_memory->handed_out_bytes(),
      ComputeRunMemoryPieceBytes((std::size_t{size.bytes_per_item} * 1000) + size.fixed_bytes)
  );
  EXPECT_EQ(run_memory->handed_out_bytes(), 14336U);
}

TEST(BuildSumBuffers, OnGpuIntoRunMemory_SameSumAsPlainForm) {
  const std::vector<double> values = BuildMixedValues(100003);
  const Buffer<double> value_buffer = BuildBufferHolding(RunOn::kGpu, values);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, std::size_t{4} << 20U);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  std::expected<SumBuffers, BufferError> piece_buffers = BuildSumBuffers(RunOn::kGpu, values.size(), *run_memory);
  std::expected<SumBuffers, BufferError> own_buffers = BuildSumBuffers(RunOn::kGpu, values.size());
  ASSERT_TRUE(piece_buffers.has_value() && own_buffers.has_value());
  const std::expected<double, BufferError> from_piece = ComputeTreeSum(value_buffer, values.size(), *piece_buffers);
  const std::expected<double, BufferError> from_own = ComputeTreeSum(value_buffer, values.size(), *own_buffers);
  ASSERT_TRUE(from_piece.has_value() && from_own.has_value());
  EXPECT_EQ(ToBits(*from_piece), ToBits(*from_own));
  WriteResultDigest(
      "ComputeTreeSum on the GPU with sum buffers from a RunMemory",
      std::span<const double>(&*from_piece, 1)
  );
}

TEST(ComputeSumBuffersSize, OnGpu_EqualsBytesBuildSumBuffersAllocates) {
  const SumBuffersSize size = ComputeSumBuffersSize(RunOn::kGpu);
  for (const std::size_t max_count : {std::size_t{0}, std::size_t{1}, std::size_t{1000}, std::size_t{65537}}) {
    const std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, max_count);
    ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
    EXPECT_EQ(buffers->byte_count(), size.bytes_per_item * max_count + size.fixed_bytes) << "max_count " << max_count;
  }
}

TEST(ComputeTreeSum, OnGpuCountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kGpu, std::array{1.0, 2.0, 3.0});
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 2);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, 3, *buffers);
  ASSERT_FALSE(sum.has_value());
  EXPECT_EQ(sum.error(), BufferError::kSizeMismatch);
}

TEST(ComputeTreeSum, OnGpuBuffersReusedForSmallerCounts_EqualsCpuBitForBit) {
  const std::vector<double> values = BuildMixedValues(1000);
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kGpu, values);
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, values.size());
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  for (const std::size_t count : {std::size_t{1000}, std::size_t{1}, std::size_t{2}, std::size_t{777}}) {
    const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, count, *buffers);
    ASSERT_TRUE(sum.has_value()) << ToName(sum.error());
    EXPECT_EQ(ToBits(*sum), ToBits(math::TreeSum(values.data(), count))) << "count " << count;
  }
}

TEST(ComputeTreeDot, OnGpuBuffersForMoreValues_EqualsTreeSumOfProducts) {
  const std::vector<double> first_values = BuildMixedValues(10007);
  const std::vector<double> second_values = BuildMixedValues(20011);
  std::vector<double> products(first_values.size());
  for (std::size_t index = 0; index < first_values.size(); ++index) {
    products[index] = first_values[index] * second_values[index];
  }
  const Buffer<double> first = BuildBufferHolding(RunOn::kGpu, first_values);
  const Buffer<double> second = BuildBufferHolding(RunOn::kGpu, second_values);
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 20011);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, first_values.size(), *buffers);
  ASSERT_TRUE(dot.has_value()) << ToName(dot.error());
  EXPECT_EQ(ToBits(*dot), ToBits(math::TreeSum(products.data(), products.size())));
}

TEST(ComputeTreeSum, OnGpuEveryCountUpTo70_EqualsCpuBitForBit) {
  const std::vector<double> values = BuildMixedValues(70);
  for (std::size_t count = 1; count <= values.size(); ++count) {
    const auto [cpu_sum, gpu_sum] = ComputeSumOnBoth(std::span<const double>(values.data(), count));
    EXPECT_EQ(ToBits(gpu_sum), ToBits(cpu_sum)) << "count " << count;
  }
}

TEST(ComputeTreeSum, OnGpuTwice_GivesIdenticalBytes) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 1000003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::vector<double> values = BuildMixedValues(1000003);
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kGpu, values);
  const std::expected<double, BufferError> first_sum = ComputeTreeSum(buffer, values.size(), *buffers);
  const std::expected<double, BufferError> second_sum = ComputeTreeSum(buffer, values.size(), *buffers);
  ASSERT_TRUE(first_sum.has_value()) << ToName(first_sum.error());
  ASSERT_TRUE(second_sum.has_value()) << ToName(second_sum.error());
  EXPECT_EQ(ToBits(*first_sum), ToBits(*second_sum));
}

TEST(ComputeTreeSum, OnGpuNegativeZeros_FollowsThePaddingRule) {
  const auto [one_cpu, one_gpu] = ComputeSumOnBoth(std::array{-0.0});
  EXPECT_EQ(ToBits(one_gpu), ToBits(-0.0));
  EXPECT_EQ(ToBits(one_cpu), ToBits(-0.0));
  const auto [three_cpu, three_gpu] = ComputeSumOnBoth(std::array{-0.0, -0.0, -0.0});
  EXPECT_EQ(ToBits(three_gpu), ToBits(0.0));
  EXPECT_EQ(ToBits(three_cpu), ToBits(0.0));
}

TEST(ComputeTreeSum, OnGpuFourValues_AddsPairsThenPairs) {
  const auto [cpu_sum, gpu_sum] = ComputeSumOnBoth(std::array{1e16, 1.0, -1e16, 1.0});
  EXPECT_EQ(ToBits(gpu_sum), ToBits(0.0));
  EXPECT_EQ(ToBits(cpu_sum), ToBits(0.0));
}

TEST(ComputeTreeDot, OnGpu_EqualsCpuBitForBit) {
  const std::vector<double> first_values = BuildMixedValues(100003);
  std::vector<double> second_values(first_values.size());
  for (std::size_t index = 0; index < second_values.size(); ++index) {
    second_values[index] = static_cast<double>(index % 17) - 8.5;
  }
  const Buffer<double> cpu_first = BuildBufferHolding(RunOn::kCpu, first_values);
  const Buffer<double> cpu_second = BuildBufferHolding(RunOn::kCpu, second_values);
  const Buffer<double> gpu_first = BuildBufferHolding(RunOn::kGpu, first_values);
  const Buffer<double> gpu_second = BuildBufferHolding(RunOn::kGpu, second_values);
  std::expected<SumBuffers, BufferError> cpu_buffers = BuildSumBuffers(RunOn::kCpu, first_values.size());
  std::expected<SumBuffers, BufferError> gpu_buffers = BuildSumBuffers(RunOn::kGpu, first_values.size());
  ASSERT_TRUE(cpu_buffers.has_value()) << ToName(cpu_buffers.error());
  ASSERT_TRUE(gpu_buffers.has_value()) << ToName(gpu_buffers.error());
  const std::expected<double, BufferError> cpu_dot =
      ComputeTreeDot(cpu_first, cpu_second, first_values.size(), *cpu_buffers);
  const std::expected<double, BufferError> gpu_dot =
      ComputeTreeDot(gpu_first, gpu_second, first_values.size(), *gpu_buffers);
  ASSERT_TRUE(cpu_dot.has_value()) << ToName(cpu_dot.error());
  ASSERT_TRUE(gpu_dot.has_value()) << ToName(gpu_dot.error());
  EXPECT_EQ(ToBits(*gpu_dot), ToBits(*cpu_dot));
  WriteResultDigest("ComputeTreeDot on the GPU, mixed values", std::span<const double>(&*gpu_dot, 1));
}

TEST(ComputeTreeDot, OnGpuProductThatFusingWouldChange_IsRoundedBeforeTheAddition) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 1000003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> first = BuildBufferHolding(RunOn::kGpu, std::array{1.0 + 0x1p-30, -1.0});
  const Buffer<double> second = BuildBufferHolding(RunOn::kGpu, std::array{1.0 - 0x1p-30, 1.0});
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, 2, *buffers);
  ASSERT_TRUE(dot.has_value()) << ToName(dot.error());
  EXPECT_EQ(ToBits(*dot), ToBits(0.0));
}

TEST(ComputeLargestMagnitude, OnGpuContainsNaN_ReturnsNaN) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 1000003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::array<std::array<double, 3>, 3> cases = {
      {
          {std::numeric_limits<double>::quiet_NaN(), 1.0, -2.0},
          {1.0, std::numeric_limits<double>::quiet_NaN(), -2.0},
          {1.0, -2.0, std::numeric_limits<double>::quiet_NaN()},
      },
  };
  for (const std::array<double, 3>& values : cases) {
    const Buffer<double> buffer = BuildBufferHolding(RunOn::kGpu, values);
    const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, values.size(), *buffers);
    ASSERT_TRUE(largest.has_value()) << ToName(largest.error());
    EXPECT_TRUE(std::isnan(*largest));
  }
}

TEST(ComputeLargestMagnitude, OnGpu_EqualsCpu) {
  const std::vector<double> values = BuildMixedValues(100003);
  const Buffer<double> cpu_buffer = BuildBufferHolding(RunOn::kCpu, values);
  const Buffer<double> gpu_buffer = BuildBufferHolding(RunOn::kGpu, values);
  std::expected<SumBuffers, BufferError> cpu_buffers = BuildSumBuffers(RunOn::kCpu, values.size());
  std::expected<SumBuffers, BufferError> gpu_buffers = BuildSumBuffers(RunOn::kGpu, values.size());
  ASSERT_TRUE(cpu_buffers.has_value()) << ToName(cpu_buffers.error());
  ASSERT_TRUE(gpu_buffers.has_value()) << ToName(gpu_buffers.error());
  const std::expected<double, BufferError> cpu_largest =
      ComputeLargestMagnitude(cpu_buffer, values.size(), *cpu_buffers);
  const std::expected<double, BufferError> gpu_largest =
      ComputeLargestMagnitude(gpu_buffer, values.size(), *gpu_buffers);
  ASSERT_TRUE(cpu_largest.has_value()) << ToName(cpu_largest.error());
  ASSERT_TRUE(gpu_largest.has_value()) << ToName(gpu_largest.error());
  EXPECT_EQ(ToBits(*gpu_largest), ToBits(*cpu_largest));
  WriteResultDigest("ComputeLargestMagnitude on the GPU, mixed values", std::span<const double>(&*gpu_largest, 1));
}

TEST(ComputeLargestMagnitude, OnGpuOneNegativeValue_ReturnsItsMagnitude) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 1000003);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kGpu, std::array{-4.5});
  const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, 1, *buffers);
  ASSERT_TRUE(largest.has_value()) << ToName(largest.error());
  EXPECT_EQ(*largest, 4.5);
}

}  // namespace
}  // namespace fluidity
