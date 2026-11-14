#include <expected>
#include <string_view>

#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/run_on.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/check_can_run_on_gpu.h"
#endif

namespace fluidity {

#if defined(FLUIDITY_HAS_GPU)
namespace {

// The answer of the GPU math check: kNone when the GPU can run, otherwise the reason it cannot.
RunOnError ComputeGpuCheckAnswer() {
  RunOnError gpu_error = RunOnError::kNone;
  if (!CheckGpuMath(&gpu_error)) {
    return gpu_error;
  }
  return RunOnError::kNone;
}

}  // namespace
#endif

std::expected<void, RunOnError> CheckCanRunOn(RunOn run_on) {
  if (run_on == RunOn::kCpu) {
    return {};
  }
#if defined(FLUIDITY_HAS_GPU)
  // Every GPU buffer asks this question, so the full check runs once per process. C++ runs the initializer of a
  // function-local static once, even when several threads arrive together; the others wait for it.
  static const RunOnError gpu_answer = ComputeGpuCheckAnswer();
  if (gpu_answer != RunOnError::kNone) {
    return std::unexpected(gpu_answer);
  }
  return {};
#else
  // This build was made without the CUDA preset, so there is no GPU code to run.
  return std::unexpected(RunOnError::kGpuNotBuilt);
#endif
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
