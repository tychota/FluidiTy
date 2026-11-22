#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/testing/buffer_cases.h"

namespace fluidity {
namespace {

TEST(WriteBuffer, OnGpuThenReadBuffer_ReturnsTheSameValues) {
  const std::vector<double> written = BuildTestValues(1000);
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kGpu, written);
  const std::vector<double> read = ReadAll(buffer);
  EXPECT_EQ(ToBits(read), ToBits(written));
  WriteResultDigest("WriteBuffer then ReadBuffer on the GPU", std::span<const double>(read));
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

}  // namespace
}  // namespace fluidity
