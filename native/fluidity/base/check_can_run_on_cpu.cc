#include <expected>
#include <string_view>

#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/run_on.h"

namespace fluidity {

std::expected<void, RunOnError> CheckCanRunOn(RunOn run_on) {
  if (run_on == RunOn::kCpu) {
    return {};
  }
  // This build was made without the CUDA preset, so there is no GPU code to run.
  return std::unexpected(RunOnError::kGpuNotBuilt);
}

std::string_view ToMessage(RunOnError error) {
  switch (error) {
    case RunOnError::kNone:
      return "There is no error.";
    case RunOnError::kGpuNotBuilt:
      return "This build has no GPU code. Use a build made with the CUDA preset, or choose the CPU.";
    case RunOnError::kNoGpuFound:
      return "No usable GPU was found. Check that a GPU and its driver are present, or choose the CPU.";
    case RunOnError::kGpuCallFailed:
      return "A GPU call failed. Check the driver and the free GPU memory, or choose the CPU.";
    case RunOnError::kGpuMathDiffers:
      return "The GPU gave different numbers from the CPU's for the shared math. Update the driver, or choose the CPU.";
  }
  return "An unknown run-on error happened.";
}

std::string_view ToName(RunOnError error) {
  switch (error) {
    case RunOnError::kNone:
      return "none";
    case RunOnError::kGpuNotBuilt:
      return "gpu_not_built";
    case RunOnError::kNoGpuFound:
      return "no_gpu_found";
    case RunOnError::kGpuCallFailed:
      return "gpu_call_failed";
    case RunOnError::kGpuMathDiffers:
      return "gpu_math_differs";
  }
  return "unknown";
}

}  // namespace fluidity
