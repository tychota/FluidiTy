// Runs on a machine without a usable GPU, such as the CUDA container: label no_gpu.

#include <gtest/gtest.h>

#include <cstddef>
#include <expected>
#include <limits>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(BuildBuffer, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kGpu, 8);
  ASSERT_FALSE(buffer.has_value());
  EXPECT_EQ(buffer.error(), BufferError::kRunOnCallFailed) << ToName(buffer.error());
}

TEST(BuildBuffer, SizeZeroOnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kGpu, 0);
  ASSERT_FALSE(buffer.has_value());
  EXPECT_EQ(buffer.error(), BufferError::kRunOnCallFailed) << ToName(buffer.error());
}

TEST(BuildBuffer, ImpossibleSizeOnGpuWithoutAGpu_IsOutOfMemory) {
  // The byte count is checked before CheckCanRunOn, so a size no object can have is out of memory on any machine.
  const std::size_t impossible_size = std::numeric_limits<std::size_t>::max() / sizeof(double);
  const std::expected<Buffer<double>, BufferError> buffer = BuildBuffer<double>(RunOn::kGpu, impossible_size);
  ASSERT_FALSE(buffer.has_value());
  EXPECT_EQ(buffer.error(), BufferError::kOutOfMemory) << ToName(buffer.error());
}

TEST(BuildBuffer, LockedForCopiesWithoutAGpu_IsAnOrdinaryCpuBuffer) {
  const std::expected<Buffer<double>, BufferError> buffer =
      BuildBuffer<double>(RunOn::kCpu, 16, CpuMemory::kLockedForCopies);
  ASSERT_TRUE(buffer.has_value()) << ToName(buffer.error());
  EXPECT_EQ(buffer->size(), 16U);
  EXPECT_EQ(buffer->run_on(), RunOn::kCpu);
}

}  // namespace
}  // namespace fluidity
