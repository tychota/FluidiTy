#include "fluidity/base/ordered_sums.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <limits>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"

namespace fluidity {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

TEST(BuildSumBuffers, MaxCountZero_Succeeds) {
  const std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 0);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  EXPECT_EQ(buffers->max_count(), 0U);
  EXPECT_EQ(buffers->run_on(), RunOn::kCpu);
}

TEST(BuildSumBuffers, IntoRunMemoryOnCpu_TakesNothing) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 4096);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const SumBuffersSize size = ComputeSumBuffersSize(RunOn::kCpu);
  // The CPU sums need no memory, so the piece is empty and takes nothing.
  EXPECT_EQ(
      run_memory->handed_out_bytes(),
      ComputeRunMemoryPieceBytes((std::size_t{size.bytes_per_item} * 1000) + size.fixed_bytes)
  );
  EXPECT_EQ(buffers->max_count(), 1000U);
  EXPECT_EQ(buffers->run_on(), RunOn::kCpu);
}

TEST(ComputeSumBuffersSize, EqualsBytesBuildSumBuffersAllocates) {
  const SumBuffersSize size = ComputeSumBuffersSize(RunOn::kCpu);
  EXPECT_EQ(size.bytes_per_item, 0U);
  EXPECT_EQ(size.fixed_bytes, 0U);
  for (const std::size_t max_count : {std::size_t{0}, std::size_t{1}, std::size_t{1000}, std::size_t{65537}}) {
    const std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, max_count);
    ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
    EXPECT_EQ(buffers->byte_count(), size.bytes_per_item * max_count + size.fixed_bytes) << "max_count " << max_count;
  }
}

TEST(ComputeSumBuffersSize, OnGpu_IsFourteenBytesPerValue) {
  const SumBuffersSize size = ComputeSumBuffersSize(RunOn::kGpu);
  EXPECT_EQ(size.bytes_per_item, 14U);
  EXPECT_EQ(size.fixed_bytes, 128U);
}

TEST(SumBuffers, MovedFrom_HasMaxCountZero) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 10);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const SumBuffers moved_to = std::move(*buffers);
  EXPECT_EQ(moved_to.max_count(), 10U);
  // What moved-from buffers hold is part of the contract, so reading it after the move is the point of the test.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(buffers->max_count(), 0U);
}

TEST(ComputeTreeSum, CountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 3);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, 4, *buffers);
  ASSERT_FALSE(sum.has_value());
  EXPECT_EQ(sum.error(), BufferError::kSizeMismatch);
}

TEST(ComputeTreeDot, CountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0});
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, std::array{3.0, 4.0});
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, 2, *buffers);
  ASSERT_FALSE(dot.has_value());
  EXPECT_EQ(dot.error(), BufferError::kSizeMismatch);
}

TEST(ComputeLargestMagnitude, CountAboveBuffersMaxCount_IsSizeMismatch) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0});
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, 2, *buffers);
  ASSERT_FALSE(largest.has_value());
  EXPECT_EQ(largest.error(), BufferError::kSizeMismatch);
}

TEST(ComputeTreeSum, CountSmallerThanBuffer_SumsOnlyTheFirstCount) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1e16, 1.0, -1e16, 1.0, 5.0});
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, 4, *buffers);
  ASSERT_TRUE(sum.has_value()) << ToName(sum.error());
  EXPECT_EQ(ToBits(*sum), ToBits(0.0));
}

TEST(ComputeTreeSum, CountLargerThanBuffer_IsSizeMismatch) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, 4, *buffers);
  ASSERT_FALSE(sum.has_value());
  EXPECT_EQ(sum.error(), BufferError::kSizeMismatch);
}

TEST(ComputeTreeSum, CountZero_IsZero) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0});
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, 0, *buffers);
  ASSERT_TRUE(sum.has_value()) << ToName(sum.error());
  EXPECT_EQ(ToBits(*sum), ToBits(0.0));
}

TEST(ComputeTreeSum, EmptyBuffer_IsZero) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer;
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, 0, *buffers);
  ASSERT_TRUE(sum.has_value()) << ToName(sum.error());
  EXPECT_EQ(ToBits(*sum), ToBits(0.0));
}

TEST(ComputeTreeDot, DifferentSizes_IsSizeMismatch) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, 4, *buffers);
  ASSERT_FALSE(dot.has_value());
  EXPECT_EQ(dot.error(), BufferError::kSizeMismatch);
}

TEST(ComputeTreeDot, DifferentSizesCountFitsBoth_DotsTheFirstCount) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, std::array{5.0, 6.0, 7.0});
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, 3, *buffers);
  ASSERT_TRUE(dot.has_value()) << ToName(dot.error());
  // (1 * 5 + 2 * 6) + (3 * 7 + 0.0) = 17 + 21 = 38.
  EXPECT_EQ(*dot, 38.0);
}

TEST(ComputeTreeDot, ProductThatFusingWouldChange_IsRoundedBeforeTheAddition) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  // (1 + 2^-30) * (1 - 2^-30) = 1 - 2^-60 exactly, which rounds to 1.0. Rounded first, 1.0 + -1.0 = 0.0. A fused
  // multiply-add keeps 1 - 2^-60 and gives -2^-60.
  constexpr double kAbove = 1.0 + 0x1p-30;
  constexpr double kBelow = 1.0 - 0x1p-30;
  ASSERT_EQ(std::fma(kAbove, kBelow, -1.0), -0x1p-60);
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, std::array{kAbove, -1.0});
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, std::array{kBelow, 1.0});
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, 2, *buffers);
  ASSERT_TRUE(dot.has_value()) << ToName(dot.error());
  EXPECT_EQ(ToBits(*dot), ToBits(0.0));
}

TEST(ComputeTreeDot, CountZero_IsZero) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, std::array{1.0});
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, std::array{2.0});
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, 0, *buffers);
  ASSERT_TRUE(dot.has_value()) << ToName(dot.error());
  EXPECT_EQ(ToBits(*dot), ToBits(0.0));
}

TEST(ComputeLargestMagnitude, MixedSigns_ReturnsLargestAbsoluteValue) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{3.0, -7.0, 5.0, -0.0});
  const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, 4, *buffers);
  ASSERT_TRUE(largest.has_value()) << ToName(largest.error());
  EXPECT_EQ(*largest, 7.0);
}

TEST(ComputeLargestMagnitude, ContainsNaN_ReturnsNaN) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  // A NaN must never be hidden by a comparison, wherever it sits.
  const std::array<std::array<double, 3>, 3> cases = {
      {
          {kNaN, 1.0, -2.0},
          {1.0, kNaN, -2.0},
          {1.0, -2.0, kNaN},
      },
  };
  for (const std::array<double, 3>& values : cases) {
    const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, values);
    const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, values.size(), *buffers);
    ASSERT_TRUE(largest.has_value()) << ToName(largest.error());
    EXPECT_TRUE(std::isnan(*largest));
  }
}

TEST(ComputeLargestMagnitude, OneNegativeValue_ReturnsItsMagnitude) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{-4.5});
  const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, 1, *buffers);
  ASSERT_TRUE(largest.has_value()) << ToName(largest.error());
  EXPECT_EQ(*largest, 4.5);
}

TEST(ComputeLargestMagnitude, CountZero_IsZero) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{-4.5});
  const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, 0, *buffers);
  ASSERT_TRUE(largest.has_value()) << ToName(largest.error());
  EXPECT_EQ(ToBits(*largest), ToBits(0.0));
}

TEST(ComputeLargestMagnitude, CountLargerThanBuffer_IsSizeMismatch) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0});
  const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, 2, *buffers);
  ASSERT_FALSE(largest.has_value());
  EXPECT_EQ(largest.error(), BufferError::kSizeMismatch);
}

}  // namespace
}  // namespace fluidity
