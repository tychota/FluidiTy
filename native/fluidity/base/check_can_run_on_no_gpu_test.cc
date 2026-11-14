// Runs on a machine without a usable GPU, such as the CUDA container: label no_gpu.

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(CheckCanRunOn, GpuOnAMachineWithoutAGpu_IsNoGpuFound) {
  const std::expected<void, RunOnError> result = CheckCanRunOn(RunOn::kGpu);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), RunOnError::kNoGpuFound) << ToName(result.error());
}

}  // namespace
}  // namespace fluidity
