#include "fluidity/base/buffer.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <set>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"

namespace fluidity {
namespace {

constexpr std::array kAllErrors = {
    BufferError::kNone,
    BufferError::kOutOfMemory,
    BufferError::kRunOnCallFailed,
    BufferError::kSizeMismatch,
    BufferError::kRunMemoryTooSmall,
};

TEST(BuildBuffer, OnCpu_HasTheAskedSize) {
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, 17);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  EXPECT_EQ(buffer->size(), 17U);
  EXPECT_NE(buffer->data(), nullptr);
}

TEST(BuildBuffer, OnCpu_LivesOnTheCpu) {
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, 17);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  EXPECT_EQ(buffer->run_on(), RunOn::kCpu);
}

TEST(BuildBuffer, SizeZero_Succeeds) {
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, 0);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  EXPECT_EQ(buffer->size(), 0U);
}

TEST(BuildBuffer, OnCpu_IsZeroFilled) {
  const std::expected<Buffer<std::uint64_t>, BufferError> buffer = BuildBuffer<std::uint64_t>(RunOn::kCpu, 100);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  const std::span<const std::uint64_t> values(buffer->data(), buffer->size());
  for (const std::uint64_t value : values) {
    EXPECT_EQ(value, 0U);
  }
}

TEST(BuildBuffer, OnCpu_DataIsAlignedTo64Bytes) {
  const std::expected<Buffer<std::uint8_t>, BufferError> buffer = BuildBuffer<std::uint8_t>(RunOn::kCpu, 3);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  EXPECT_EQ(std::bit_cast<std::uintptr_t>(buffer->data()) % kBufferAlignmentBytes, 0U);
}

TEST(BuildBuffer, ImpossibleSize_IsOutOfMemory) {
  const std::size_t impossible_size = std::numeric_limits<std::size_t>::max() / sizeof(double);
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, impossible_size);
  ASSERT_FALSE(buffer.has_value());
  EXPECT_EQ(buffer.error(), BufferError::kOutOfMemory);
}

TEST(BuildBuffer, LargerThanAnyMachine_IsOutOfMemory) {
  // About 8 EiB: passes the size check, so the allocation itself is tried and fails.
  const std::size_t huge_size = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double);
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, huge_size);
  ASSERT_FALSE(buffer.has_value());
  EXPECT_EQ(buffer.error(), BufferError::kOutOfMemory);
}

TEST(BuildBuffer, ByteCountAboveSizeMax_IsOutOfMemory) {
  const std::size_t overflowing_size = (std::numeric_limits<std::size_t>::max() / sizeof(double)) + 1;
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, overflowing_size);
  ASSERT_FALSE(buffer.has_value());
  EXPECT_EQ(buffer.error(), BufferError::kOutOfMemory);
}

TEST(BuildBuffer, LockedForCopiesOnCpu_BehavesLikeAnOrdinaryBuffer) {
  std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, 64, CpuMemory::kLockedForCopies);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  EXPECT_EQ(buffer->size(), 64U);
  EXPECT_EQ(buffer->run_on(), RunOn::kCpu);
  EXPECT_EQ(std::bit_cast<std::uintptr_t>(buffer->data()) % kBufferAlignmentBytes, 0U);
  std::vector<double> values(64);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = 0.5 * static_cast<double>(index);
  }
  ASSERT_TRUE(WriteBuffer(values, *buffer).has_value());
  EXPECT_EQ(ReadAll(*buffer), values);
}

#if !defined(FLUIDITY_HAS_GPU)
TEST(BuildBuffer, OnGpuInBuildWithoutGpuCode_IsRunOnCallFailed) {
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kGpu, 8);
  ASSERT_FALSE(buffer.has_value());
  EXPECT_EQ(buffer.error(), BufferError::kRunOnCallFailed);
}
#endif

TEST(WriteBuffer, ThenReadBuffer_ReturnsTheSameValues) {
  std::vector<double> written(1000);
  for (std::size_t index = 0; index < written.size(); ++index) {
    // Values with every bit pattern kind that matters: negative zero, a subnormal and ordinary numbers.
    written[index] = (static_cast<double>(index) - 500.0) / 7.0;
  }
  written[0] = -0.0;
  written[1] = std::numeric_limits<double>::denorm_min();
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, written);
  EXPECT_EQ(ToBits(ReadAll(buffer)), ToBits(written));
}

TEST(WriteBuffer, SpanOfDifferentSize_IsSizeMismatch) {
  std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, 4);
  ASSERT_TRUE(buffer.has_value());
  const std::array<double, 3> values = {1.0, 2.0, 3.0};
  const std::expected<void, BufferError> result = WriteBuffer(values, *buffer);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
}

TEST(ReadBuffer, SpanOfDifferentSize_IsSizeMismatch) {
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, 4);
  ASSERT_TRUE(buffer.has_value());
  std::array<double, 5> values{};
  const std::expected<void, BufferError> result = ReadBuffer(*buffer, values);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
}

TEST(ReadBuffer, SpanOfDifferentSize_WritesNothingToTheSpan) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  std::array<double, 5> values = {9.0, 9.0, 9.0, 9.0, 9.0};
  ASSERT_FALSE(ReadBuffer(buffer, values).has_value());
  EXPECT_EQ(values, (std::array{9.0, 9.0, 9.0, 9.0, 9.0}));
}

TEST(WriteBuffer, Slice_LeavesValuesOutsideTheSliceUntouched) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0, 5.0});
  const std::array slice_values = {20.0, 30.0};
  ASSERT_TRUE(WriteBuffer(slice_values, buffer, 1).has_value());
  EXPECT_EQ(ReadAll(buffer), (std::vector{1.0, 20.0, 30.0, 4.0, 5.0}));
}

TEST(WriteBuffer, SliceBeyondTheEnd_WritesNothing) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  const std::array slice_values = {20.0, 30.0};
  const std::expected<void, BufferError> result = WriteBuffer(slice_values, buffer, 2);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(buffer), (std::vector{1.0, 2.0, 3.0}));
}

TEST(WriteBuffer, SliceStartingFarBeyondTheEnd_IsSizeMismatch) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  const std::array slice_values = {20.0};
  // A first index so large that first + count would wrap around: the range check must not wrap.
  const std::size_t first = std::numeric_limits<std::size_t>::max();
  const std::expected<void, BufferError> result = WriteBuffer(slice_values, buffer, first);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
}

TEST(WriteBuffer, EmptySliceAtTheEnd_Succeeds) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  const std::expected<void, BufferError> result = WriteBuffer(std::span<const double>(), buffer, 3);
  EXPECT_TRUE(result.has_value());
}

TEST(ReadBuffer, Slice_ReadsOnlyTheSlice) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0, 5.0});
  std::array<double, 3> values{};
  ASSERT_TRUE(ReadBuffer(buffer, 2, values).has_value());
  EXPECT_EQ(values, (std::array{3.0, 4.0, 5.0}));
}

TEST(ReadBuffer, SliceBeyondTheEnd_IsSizeMismatch) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  std::array<double, 2> values{};
  const std::expected<void, BufferError> result = ReadBuffer(buffer, 2, values);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
}

TEST(WriteBuffer, VectorWithoutSpan_IsAccepted) {
  std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kCpu, 4);
  ASSERT_TRUE(buffer.has_value());
  const std::vector<double> written = {1.0, 2.0, 3.0, 4.0};
  ASSERT_TRUE(WriteBuffer(written, *buffer).has_value());
  EXPECT_EQ(ReadAll(*buffer), written);
}

TEST(ReadBuffer, ArrayWithoutSpan_IsAccepted) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  std::array<double, 4> read{};
  ASSERT_TRUE(ReadBuffer(buffer, read).has_value());
  EXPECT_EQ(read, (std::array{1.0, 2.0, 3.0, 4.0}));
}

TEST(WriteBuffer, SpanOverlappingTheSliceOfTheSameBuffer_IsSizeMismatch) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  const std::span<const double> values_inside_buffer(buffer.data(), 3);
  const std::expected<void, BufferError> result = WriteBuffer(values_inside_buffer, buffer, 1);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(buffer), (std::vector{1.0, 2.0, 3.0, 4.0}));
}

TEST(ReadBuffer, SpanOverlappingTheSliceOfTheSameBuffer_IsSizeMismatch) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  const std::span<double> values_inside_buffer(buffer.data() + 1, 3);
  const std::expected<void, BufferError> result = ReadBuffer(buffer, 0, values_inside_buffer);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(buffer), (std::vector{1.0, 2.0, 3.0, 4.0}));
}

TEST(Buffer, MovedFrom_HasSizeZero) {
  Buffer<double> original = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0});
  const Buffer<double> moved_to = std::move(original);
  // What a moved-from buffer holds is part of the contract, so reading it after the move is the point of the test.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(original.size(), 0U);
  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.run_on(), RunOn::kCpu);
  EXPECT_EQ(moved_to.size(), 2U);
  EXPECT_EQ(ReadAll(moved_to), (std::vector{1.0, 2.0}));
  // Leaving the scope destroys both; a double free would crash here.
}

TEST(Buffer, MoveAssigned_TakesTheValues) {
  Buffer<double> target = BuildBufferHolding(RunOn::kCpu, std::array{7.0});
  Buffer<double> moved_from = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  target = std::move(moved_from);
  EXPECT_EQ(ReadAll(target), (std::vector{1.0, 2.0, 3.0}));
  // Reading the moved-from buffer is the point of the check, as in MovedFrom_HasSizeZero.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(moved_from.size(), 0U);
}

TEST(CopyBuffer, SameProcessor_CopiesEveryValue) {
  const Buffer<double> from = BuildBufferHolding(RunOn::kCpu, std::array{1.0, -2.0, 3.5});
  Buffer<double> to = BuildBufferHolding(RunOn::kCpu, std::array{0.0, 0.0, 0.0});
  ASSERT_TRUE(CopyBuffer(from, to).has_value());
  EXPECT_EQ(ReadAll(to), (std::vector{1.0, -2.0, 3.5}));
}

TEST(CopyBuffer, DifferentSizes_IsSizeMismatch) {
  const Buffer<double> from = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  Buffer<double> to = BuildBufferHolding(RunOn::kCpu, std::array{0.0, 0.0});
  const std::expected<void, BufferError> result = CopyBuffer(from, to);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
}

TEST(CopyBuffer, DifferentSizes_WritesNothing) {
  const Buffer<double> from = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0});
  Buffer<double> to = BuildBufferHolding(RunOn::kCpu, std::array{8.0, 9.0});
  ASSERT_FALSE(CopyBuffer(from, to).has_value());
  EXPECT_EQ(ReadAll(to), (std::vector{8.0, 9.0}));
}

TEST(CopyBuffer, Slice_CopiesOnlyTheSlice) {
  const Buffer<double> from = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  Buffer<double> to = BuildBufferHolding(RunOn::kCpu, std::array{0.0, 0.0, 0.0});
  ASSERT_TRUE(CopyBuffer(from, 1, 2, to, 0).has_value());
  EXPECT_EQ(ReadAll(to), (std::vector{2.0, 3.0, 0.0}));
}

TEST(CopyBuffer, SliceBeyondTheEnd_IsSizeMismatch) {
  const Buffer<double> from = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  Buffer<double> to = BuildBufferHolding(RunOn::kCpu, std::array{0.0, 0.0, 0.0});
  const std::expected<void, BufferError> result = CopyBuffer(from, 1, 3, to, 1);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
}

TEST(CopyBuffer, OverlappingSlicesOfOneBuffer_IsSizeMismatch) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  const std::expected<void, BufferError> result = CopyBuffer(buffer, 0, 3, buffer, 1);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), BufferError::kSizeMismatch);
  EXPECT_EQ(ReadAll(buffer), (std::vector{1.0, 2.0, 3.0, 4.0}));
}

TEST(CopyBuffer, SeparateSlicesOfOneBuffer_Copies) {
  Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0, 3.0, 4.0});
  ASSERT_TRUE(CopyBuffer(buffer, 0, 2, buffer, 2).has_value());
  EXPECT_EQ(ReadAll(buffer), (std::vector{1.0, 2.0, 1.0, 2.0}));
}

TEST(ToMessage, EveryError_HasADistinctNonEmptyMessage) {
  std::set<std::string_view> messages;
  for (const BufferError error : kAllErrors) {
    const std::string_view message = ToMessage(error);
    EXPECT_FALSE(message.empty());
    messages.insert(message);
  }
  EXPECT_EQ(messages.size(), kAllErrors.size());
}

TEST(ToName, EveryError_IsSnakeCaseWithoutK) {
  EXPECT_EQ(ToName(BufferError::kNone), "none");
  EXPECT_EQ(ToName(BufferError::kOutOfMemory), "out_of_memory");
  EXPECT_EQ(ToName(BufferError::kRunOnCallFailed), "run_on_call_failed");
  EXPECT_EQ(ToName(BufferError::kSizeMismatch), "size_mismatch");
  EXPECT_EQ(ToName(BufferError::kRunMemoryTooSmall), "run_memory_too_small");
}

TEST(ToBufferError, EveryStatus_GivesItsError) {
  EXPECT_EQ(ToBufferError(RunOnCallStatus::kDone), BufferError::kNone);
  EXPECT_EQ(ToBufferError(RunOnCallStatus::kOutOfMemory), BufferError::kOutOfMemory);
  EXPECT_EQ(ToBufferError(RunOnCallStatus::kSizeMismatch), BufferError::kSizeMismatch);
  EXPECT_EQ(ToBufferError(RunOnCallStatus::kCallFailed), BufferError::kRunOnCallFailed);
}

TEST(ToBufferError, OutOfRangeStatus_IsRunOnCallFailed) {
  EXPECT_EQ(ToBufferError(static_cast<RunOnCallStatus>(200)), BufferError::kRunOnCallFailed);
}

TEST(ToMessage, OutOfRangeValue_IsUnknownErrorText) {
  EXPECT_EQ(ToMessage(static_cast<BufferError>(200)), "An unknown buffer error happened.");
}

TEST(ToName, OutOfRangeValue_IsUnknown) {
  EXPECT_EQ(ToName(static_cast<BufferError>(200)), "unknown");
}

}  // namespace
}  // namespace fluidity
