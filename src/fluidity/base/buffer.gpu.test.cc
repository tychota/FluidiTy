// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include "fluidity/base/buffer.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/buffer_cases.h"

namespace fluidity {
namespace {

TEST(BuildBuffer, OnGpu_IsZeroFilled) {
  const std::expected<Buffer<std::uint64_t>, BufferError> buffer = BuildBuffer<std::uint64_t>(RunOn::kGpu, 100);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  std::vector<std::uint64_t> read(100, 1);
  ASSERT_TRUE(ReadBuffer(*buffer, read).has_value());
  for (const std::uint64_t value : read) {
    EXPECT_EQ(value, 0U);
  }
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
