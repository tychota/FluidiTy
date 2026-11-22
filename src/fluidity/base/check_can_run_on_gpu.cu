// The GPU math check: the GPU half of CheckCanRunOn, private to base/.
//
// It runs the shared math of base/check_can_run_on_math.h on the GPU and on the CPU and compares the bits. The
// numbered steps match Procedure <proc-check-can-run-on> of the method paper. The launch, the memory and the
// statuses go through base/gpu_launch.cuh.
// Read: docs/method-paper/fluidity-method.typ <proc-check-can-run-on>

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "fluidity/base/check_can_run_on_gpu.h"
#include "fluidity/base/check_can_run_on_math.h"
#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"

namespace fluidity {
namespace {

// One thread block of one thread per input.
constexpr LaunchSize kMathCheckLaunchSize = {.thread_block_count = 1, .thread_block_size = kMathCheckInputCount};
constexpr int kResultValueCount = 11;
// A new field in MathCheckResult must also be added to ToResultBits below.
static_assert(sizeof(MathCheckResult) == kResultValueCount * sizeof(double));

using MathCheckInputsArray = std::array<MathCheckInputs, kMathCheckInputCount>;
using MathCheckResultArray = std::array<MathCheckResult, kMathCheckInputCount>;

// The GPU function of the check: results[i] = ComputeMathCheckResult(inputs[i]) for i below input_count.
__global__ void ComputeMathCheckResults(const MathCheckInputs* inputs, MathCheckResult* results, int input_count) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t input_index = stride.first_item; input_index < static_cast<std::size_t>(input_count);
       input_index += stride.item_step) {
    results[input_index] = ComputeMathCheckResult(inputs[input_index]);
  }
}

// Whether a status from cudaGetDeviceCount means that this machine has no usable GPU: no GPU, no driver, a driver
// too old for this runtime, or only the placeholder driver library that the CUDA toolkit ships for building.
bool IsNoGpuStatus(cudaError_t status) {
  return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || status == cudaErrorStubLibrary;
}

// Copies the inputs to the GPU, runs the GPU function with one thread per input, and copies the results back.
// Returns kDone, or the status of the first call that failed. `results` is written only on success.
RunOnCallStatus ComputeAndReadResults(const MathCheckInputsArray& inputs, MathCheckResultArray& results) {
  constexpr std::size_t kInputBytes = sizeof(MathCheckInputsArray);
  constexpr std::size_t kResultBytes = sizeof(MathCheckResultArray);

  GpuArray<MathCheckInputs> gpu_inputs;
  GpuArray<MathCheckResult> gpu_results;
  if (const RunOnCallStatus status = gpu_inputs.Build(kMathCheckInputCount); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_results.Build(kMathCheckInputCount); status != RunOnCallStatus::kDone) {
    return status;
  }
  // Write: the inputs to the GPU.
  if (const cudaError_t error = cudaMemcpy(gpu_inputs.values(), inputs.data(), kInputBytes, cudaMemcpyHostToDevice);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = ApplyGpuFunction(
          ComputeMathCheckResults,
          kMathCheckLaunchSize,
          gpu_inputs.values(),
          gpu_results.values(),
          kMathCheckInputCount
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // Read: the results from the GPU. The copy waits for the GPU function to finish, so an error while it ran
  // reports here.
  MathCheckResultArray read_results{};
  if (const cudaError_t error =
          cudaMemcpy(read_results.data(), gpu_results.values(), kResultBytes, cudaMemcpyDeviceToHost);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = gpu_inputs.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_results.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  results = read_results;
  return RunOnCallStatus::kDone;
}

// The 64 bits of every double of a result, in field order.
std::array<std::uint64_t, kResultValueCount> ToResultBits(const MathCheckResult& result) {
  return {
      std::bit_cast<std::uint64_t>(result.cross.x),
      std::bit_cast<std::uint64_t>(result.cross.y),
      std::bit_cast<std::uint64_t>(result.cross.z),
      std::bit_cast<std::uint64_t>(result.dot),
      std::bit_cast<std::uint64_t>(result.rotated_first.x),
      std::bit_cast<std::uint64_t>(result.rotated_first.y),
      std::bit_cast<std::uint64_t>(result.rotated_first.z),
      std::bit_cast<std::uint64_t>(result.determinant),
      std::bit_cast<std::uint64_t>(result.square_root_of_dot),
      std::bit_cast<std::uint64_t>(result.floor_of_scaled_dot),
      std::bit_cast<std::uint64_t>(result.length_of_difference),
  };
}

}  // namespace

bool CheckGpuMath(RunOnError* error) {
  // Steps: 1 find a GPU  2 build the fixed inputs  3 compute on the GPU  4 compute on the CPU
  //        5 compare bit for bit  6 report

  // Step 1: find a GPU.
  int gpu_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&gpu_count);
  const bool has_no_gpu = IsNoGpuStatus(count_status) || (count_status == cudaSuccess && gpu_count == 0);
  if (has_no_gpu) {
    // No GPU, or no driver that answers: the caller can choose the CPU or fix the driver.
    *error = RunOnError::kNoGpuFound;
    return false;
  }
  if (count_status != cudaSuccess) {
    // The driver answered with another error, for example a failed initialization.
    *error = RunOnError::kGpuCallFailed;
    return false;
  }

  // Step 2: build the fixed inputs, the same on both processors.
  MathCheckInputsArray inputs{};
  for (int input_index = 0; input_index < kMathCheckInputCount; ++input_index) {
    inputs[input_index] = BuildMathCheckInputs(input_index);
  }

  // Step 3: compute on the GPU.
  MathCheckResultArray gpu_results{};
  if (ComputeAndReadResults(inputs, gpu_results) != RunOnCallStatus::kDone) {
    // A GPU call failed after a GPU was found: out of memory, a driver fault, or GPU code built for another GPU.
    *error = RunOnError::kGpuCallFailed;
    return false;
  }

  // Step 4: compute on the CPU, with the same shared function.
  MathCheckResultArray cpu_results{};
  for (int input_index = 0; input_index < kMathCheckInputCount; ++input_index) {
    cpu_results[input_index] = ComputeMathCheckResult(inputs[input_index]);
  }

  // Step 5: compare bit for bit. The bits, not ==, because == treats +0 and -0 as equal.
  for (int input_index = 0; input_index < kMathCheckInputCount; ++input_index) {
    if (ToResultBits(gpu_results[input_index]) != ToResultBits(cpu_results[input_index])) {
      // A numerical flag or the driver changed the rounding on one side.
      *error = RunOnError::kGpuMathDiffers;
      return false;
    }
  }

  // Step 6: report. The check changes nothing; it never switches the choice to the CPU.
  return true;
}

}  // namespace fluidity
