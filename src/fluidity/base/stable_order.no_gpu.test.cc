// Runs on a machine without a usable GPU, such as the CUDA container: label no_gpu.
//
// Only the order buffers can be asked for on the GPU there: every other function takes buffers, and without a GPU no
// GPU buffer exists.

#include "fluidity/base/stable_order.h"

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(BuildOrderBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kGpu, 16);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error(), BufferError::kRunOnCallFailed) << ToName(buffers.error());
}

}  // namespace
}  // namespace fluidity
