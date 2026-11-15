// Runs on a machine with a usable GPU: label gpu.

#include <gtest/gtest.h>

#include <cstdint>
#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(ReadProcessorDescription, OnGpu_ModelIsNotEmpty) {
  const ProcessorDescription description = ReadProcessorDescription(RunOn::kGpu);
  EXPECT_FALSE(description.model.empty());
  EXPECT_FALSE(description.driver.empty());
  EXPECT_FALSE(description.runtime.empty());
}

TEST(ReadMemoryAvailable, OnGpuTwice_SecondFreeNotBelowFirst) {
  // Two readings in a row agree, within an allowance for other programs taking a little memory meanwhile. The context
  // already exists before the first read, because the processor check (CheckCanRunOn) ran GPU code, so this test
  // cannot tell the cudaFree(nullptr) before the reading from the old code: it only guards the reading's stability.
  constexpr std::uint64_t kAllowedChangeBytes = std::uint64_t{64} << 20U;
  const std::expected<MemoryAvailable, BufferError> first = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(first.has_value()) << ToName(first.error());
  const std::expected<MemoryAvailable, BufferError> second = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(second.has_value()) << ToName(second.error());
  EXPECT_GE(second->free_bytes + kAllowedChangeBytes, first->free_bytes);
  EXPECT_GT(first->free_bytes, 0U);
}

}  // namespace
}  // namespace fluidity
