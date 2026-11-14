#include "fluidity/base/check_can_run_on.h"

#include <gtest/gtest.h>

#include <array>
#include <expected>
#include <set>
#include <string_view>

#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

constexpr std::array kAllErrors = {
    RunOnError::kNone,
    RunOnError::kGpuNotBuilt,
    RunOnError::kNoGpuFound,
    RunOnError::kGpuCallFailed,
    RunOnError::kGpuMathDiffers,
};

TEST(CheckCanRunOn, Cpu_Succeeds) {
  const std::expected<void, RunOnError> result = CheckCanRunOn(RunOn::kCpu);
  EXPECT_TRUE(result.has_value());
}

#if !defined(FLUIDITY_HAS_GPU)
TEST(CheckCanRunOn, GpuInBuildWithoutGpuCode_IsNotBuilt) {
  const std::expected<void, RunOnError> result = CheckCanRunOn(RunOn::kGpu);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), RunOnError::kGpuNotBuilt);
}
#endif

TEST(ToMessage, EveryError_HasADistinctNonEmptyMessage) {
  std::set<std::string_view> messages;
  for (const RunOnError error : kAllErrors) {
    const std::string_view message = ToMessage(error);
    EXPECT_FALSE(message.empty());
    messages.insert(message);
  }
  EXPECT_EQ(messages.size(), kAllErrors.size());
}

TEST(ToName, EveryError_IsSnakeCaseWithoutK) {
  EXPECT_EQ(ToName(RunOnError::kNone), "none");
  EXPECT_EQ(ToName(RunOnError::kGpuNotBuilt), "gpu_not_built");
  EXPECT_EQ(ToName(RunOnError::kNoGpuFound), "no_gpu_found");
  EXPECT_EQ(ToName(RunOnError::kGpuCallFailed), "gpu_call_failed");
  EXPECT_EQ(ToName(RunOnError::kGpuMathDiffers), "gpu_math_differs");
}

TEST(ToMessage, OutOfRangeValue_IsUnknownErrorText) {
  const std::string_view message = ToMessage(static_cast<RunOnError>(200));
  EXPECT_EQ(message, "An unknown run-on error happened.");
}

TEST(ToName, OutOfRangeValue_IsUnknown) {
  EXPECT_EQ(ToName(static_cast<RunOnError>(200)), "unknown");
}

}  // namespace
}  // namespace fluidity
