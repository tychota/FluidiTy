// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include "fluidity/base/buffer.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"

namespace fluidity {
namespace {

// Values with a negative zero, a subnormal and ordinary numbers, so a round trip is checked on every bit kind.
std::vector<double> BuildTestValues(std::size_t count) {
  std::vector<double> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = (static_cast<double>(index) - 500.0) / 7.0;
  }
  values[0] = -0.0;
  values[1] = 4.9406564584124654e-324;
  return values;
}

TEST(WriteBuffer, OnGpuThenReadBuffer_ReturnsTheSameValues) {
  const std::vector<double> written = BuildTestValues(1000);
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kGpu, written);
  const std::vector<double> read = ReadAll(buffer);
  EXPECT_EQ(ToBits(read), ToBits(written));
  WriteResultDigest("WriteBuffer then ReadBuffer on the GPU", std::span<const double>(read));
}

TEST(BuildBuffer, OnGpu_IsZeroFilled) {
  const std::expected<Buffer<std::uint64_t>, BufferError> buffer = BuildBuffer<std::uint64_t>(RunOn::kGpu, 100);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  std::vector<std::uint64_t> read(100, 1);
  ASSERT_TRUE(ReadBuffer(*buffer, read).has_value());
  for (const std::uint64_t value : read) {
    EXPECT_EQ(value, 0U);
  }
}

TEST(BuildBuffer, OnGpuManyTimes_FreesEverything) {
  constexpr std::size_t kValueCount = std::size_t{1} << 17U;  // 1 MiB of doubles per buffer.
  // Other programs may take or give back a little GPU memory meanwhile. A leak of 1000 buffers would show as about
  // 1000 MiB, far above this allowance; a leak of fewer than about 64 buffers would go unseen.
  constexpr std::uint64_t kAllowedChangeBytes = std::uint64_t{64} << 20U;
  const std::expected<MemoryAvailable, BufferError> before = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(before.has_value()) << ToName(before.error());
  for (int cycle = 0; cycle < 1000; ++cycle) {
    const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kGpu, kValueCount);
    ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  }
  const std::expected<MemoryAvailable, BufferError> after = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after.has_value()) << ToName(after.error());
  EXPECT_GE(after->free_bytes + kAllowedChangeBytes, before->free_bytes);
}

TEST(CopyBuffer, OnGpu_CopiesEveryValue) {
  const std::vector<double> written = BuildTestValues(64);
  const Buffer<double> from = BuildBufferHolding(RunOn::kGpu, written);
  Buffer<double> to = BuildBufferHolding(RunOn::kGpu, std::vector<double>(written.size()));
  ASSERT_TRUE(CopyBuffer(from, to).has_value());
  const std::vector<double> read = ReadAll(to);
  EXPECT_EQ(ToBits(read), ToBits(written));
  WriteResultDigest("CopyBuffer on the GPU", std::span<const double>(read));
}

TEST(BuildBuffer, LockedForCopiesWithAGpu_BehavesLikeAnOrdinaryBuffer) {
  const std::vector<double> written = BuildTestValues(64);
  std::expected<Buffer<double>, BufferError> buffer =
      BuildBuffer<double>(RunOn::kCpu, written.size(), CpuMemory::kLockedForCopies);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  EXPECT_EQ(std::bit_cast<std::uintptr_t>(buffer->data()) % kBufferAlignmentBytes, 0U);
  ASSERT_TRUE(WriteBuffer(written, *buffer).has_value());
  EXPECT_EQ(ReadAll(*buffer), written);
}

}  // namespace
}  // namespace fluidity
