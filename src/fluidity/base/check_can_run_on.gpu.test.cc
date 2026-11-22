// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.

#include "fluidity/base/check_can_run_on.h"

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(CheckCanRunOn, GpuOnAMachineWithAGpu_Succeeds) {
  const std::expected<void, RunOnError> result = CheckCanRunOn(RunOn::kGpu);
  ASSERT_TRUE(result.has_value()) << ToName(result.error());
}

TEST(CheckCanRunOn, GpuTwiceInARow_SucceedsBothTimes) {
  const std::expected<void, RunOnError> first_result = CheckCanRunOn(RunOn::kGpu);
  const std::expected<void, RunOnError> second_result = CheckCanRunOn(RunOn::kGpu);
  ASSERT_TRUE(first_result.has_value()) << ToName(first_result.error());
  ASSERT_TRUE(second_result.has_value()) << ToName(second_result.error());
}

}  // namespace
}  // namespace fluidity
