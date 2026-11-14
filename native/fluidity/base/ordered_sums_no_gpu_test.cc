// Runs on a machine without a usable GPU, such as the CUDA container: label no_gpu.
//
// Only the sum buffers can be asked for on the GPU there: every other function takes buffers, and without a GPU no
// GPU buffer exists.

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/ordered_sums.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(BuildSumBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<SumBuffers, BufferError> buffers = BuildSumBuffers(RunOn::kGpu, 16);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error(), BufferError::kRunOnCallFailed) << ToName(buffers.error());
}

}  // namespace
}  // namespace fluidity
