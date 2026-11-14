// Runs on a machine without a usable GPU, such as the CUDA container: label no_gpu.

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(ReadMemoryAvailable, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<MemoryAvailable, BufferError> memory = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_FALSE(memory.has_value());
  EXPECT_EQ(memory.error(), BufferError::kRunOnCallFailed) << ToName(memory.error());
}

}  // namespace
}  // namespace fluidity
