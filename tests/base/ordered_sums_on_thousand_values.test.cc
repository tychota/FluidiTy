#include <gtest/gtest.h>

#include <cstddef>
#include <expected>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/ordered_sums.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/tree_sum.h"

namespace fluidity {
namespace {

// 1000 values from the fixed formula of the test support, with magnitudes from 1e-3 to 1e12 and both signs.
std::vector<double> BuildThousandValues() {
  return BuildMixedValues(1000);
}

TEST(BuildSumBuffers, IntoRunMemory_SameSumAsPlainForm) {
  const std::vector<double> values = BuildThousandValues();
  const Buffer<double> value_buffer = BuildBufferHolding(RunOn::kCpu, values);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 256);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<SumBuffers, BufferError> piece_buffers = BuildSumBuffers(RunOn::kCpu, values.size(), *run_memory);
  std::expected<SumBuffers, BufferError> own_buffers = BuildSumBuffers(RunOn::kCpu, values.size());
  ASSERT_TRUE(piece_buffers.has_value() && own_buffers.has_value());
  const std::expected<double, BufferError> from_piece = ComputeTreeSum(value_buffer, values.size(), *piece_buffers);
  const std::expected<double, BufferError> from_own = ComputeTreeSum(value_buffer, values.size(), *own_buffers);
  ASSERT_TRUE(from_piece.has_value() && from_own.has_value());
  EXPECT_EQ(ToBits(*from_piece), ToBits(*from_own));
}

TEST(ComputeTreeSum, BuffersForMoreValues_EqualsTreeSumBitForBit) {
  const std::vector<double> values = BuildThousandValues();
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, values);
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 2000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, values.size(), *buffers);
  ASSERT_TRUE(sum.has_value()) << ToName(sum.error());
  EXPECT_EQ(ToBits(*sum), ToBits(math::TreeSum(values.data(), values.size())));
}

TEST(ComputeTreeDot, BuffersForMoreValues_EqualsTreeSumOfProducts) {
  const std::vector<double> first_values = BuildThousandValues();
  const std::vector<double> second_values = BuildMixedValues(1500);
  std::vector<double> products(first_values.size());
  for (std::size_t index = 0; index < first_values.size(); ++index) {
    products[index] = first_values[index] * second_values[index];
  }
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, first_values);
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, second_values);
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 2000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, first_values.size(), *buffers);
  ASSERT_TRUE(dot.has_value()) << ToName(dot.error());
  EXPECT_EQ(ToBits(*dot), ToBits(math::TreeSum(products.data(), products.size())));
}

TEST(ComputeLargestMagnitude, BuffersForMoreValues_EqualsLargerMagnitudeOfEveryValue) {
  const std::vector<double> values = BuildThousandValues();
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, values);
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 2000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::expected<double, BufferError> largest = ComputeLargestMagnitude(buffer, values.size(), *buffers);
  ASSERT_TRUE(largest.has_value()) << ToName(largest.error());
  double expected = 0.0;
  for (const double value : values) {
    expected = math::LargerMagnitude(expected, value);
  }
  EXPECT_EQ(ToBits(*largest), ToBits(expected));
}

TEST(ComputeTreeSum, OnCpu_EqualsTreeSumBitForBit) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::vector<double> values = BuildThousandValues();
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, values);
  const std::expected<double, BufferError> sum = ComputeTreeSum(buffer, values.size(), *buffers);
  ASSERT_TRUE(sum.has_value()) << ToName(sum.error());
  EXPECT_EQ(ToBits(*sum), ToBits(math::TreeSum(values.data(), values.size())));
}

TEST(ComputeTreeSum, AnyCall_LeavesTheBufferUnchanged) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::vector<double> values = BuildThousandValues();
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, values);
  ASSERT_TRUE(ComputeTreeSum(buffer, values.size(), *buffers).has_value());
  std::vector<double> read(values.size());
  ASSERT_TRUE(ReadBuffer(buffer, read).has_value());
  EXPECT_EQ(read, values);
}

TEST(ComputeTreeDot, AnyCall_LeavesBothBuffersUnchanged) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::vector<double> first_values = BuildThousandValues();
  std::vector<double> second_values(first_values.size());
  for (std::size_t index = 0; index < second_values.size(); ++index) {
    second_values[index] = static_cast<double>(index % 17) - 8.5;
  }
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, first_values);
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, second_values);
  ASSERT_TRUE(ComputeTreeDot(first, second, first_values.size(), *buffers).has_value());
  std::vector<double> first_read(first_values.size());
  std::vector<double> second_read(second_values.size());
  ASSERT_TRUE(ReadBuffer(first, first_read).has_value());
  ASSERT_TRUE(ReadBuffer(second, second_read).has_value());
  EXPECT_EQ(first_read, first_values);
  EXPECT_EQ(second_read, second_values);
}

TEST(ComputeTreeDot, OnCpu_EqualsTreeSumOfProducts) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::vector<double> first_values = BuildThousandValues();
  std::vector<double> second_values(first_values.size());
  std::vector<double> products(first_values.size());
  for (std::size_t index = 0; index < first_values.size(); ++index) {
    second_values[index] = static_cast<double>(index % 17) - 8.5;
    products[index] = first_values[index] * second_values[index];
  }
  const Buffer<double> first = BuildBufferHolding(RunOn::kCpu, first_values);
  const Buffer<double> second = BuildBufferHolding(RunOn::kCpu, second_values);
  const std::expected<double, BufferError> dot = ComputeTreeDot(first, second, first_values.size(), *buffers);
  ASSERT_TRUE(dot.has_value()) << ToName(dot.error());
  EXPECT_EQ(ToBits(*dot), ToBits(math::TreeSum(products.data(), products.size())));
}

TEST(ComputeLargestMagnitude, AnyCall_LeavesTheBufferUnchanged) {
  std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kCpu, 1000);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error());
  const std::vector<double> values = BuildThousandValues();
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, values);
  ASSERT_TRUE(ComputeLargestMagnitude(buffer, values.size(), *buffers).has_value());
  std::vector<double> read(values.size());
  ASSERT_TRUE(ReadBuffer(buffer, read).has_value());
  EXPECT_EQ(read, values);
}

}  // namespace
}  // namespace fluidity
