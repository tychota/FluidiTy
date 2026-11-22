// The GPU runner of scalar_math.shared.gpu.test.cc: ComputeSharedMathResults over every input, one thread per input.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>

#include <cuda_runtime.h>

#include <cstddef>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/shared_math_runner_gpu.h"

namespace fluidity {
namespace {

// results[i] = ComputeSharedMathResults(inputs[i]) for i below count.
__global__ void ComputeEverySharedMathResult(const SharedMathInputs* inputs, int count, SharedMathResult* results) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < static_cast<std::size_t>(count); index += stride.item_step) {
    results[index] = ComputeSharedMathResults(inputs[index]);
  }
}

}  // namespace

RunOnCallStatus ComputeSharedMathOnGpu(const SharedMathInputs* inputs, int count, SharedMathResult* results) {
  const auto value_count = static_cast<std::size_t>(count);
  GpuArray<SharedMathInputs> gpu_inputs;
  GpuArray<SharedMathResult> gpu_results;
  if (const RunOnCallStatus status = gpu_inputs.Build(value_count); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_results.Build(value_count); status != RunOnCallStatus::kDone) {
    return status;
  }
  const std::size_t input_bytes = sizeof(SharedMathInputs) * value_count;
  if (const cudaError_t error = cudaMemcpy(gpu_inputs.values(), inputs, input_bytes, cudaMemcpyHostToDevice);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = ApplyGpuFunction(
          ComputeEverySharedMathResult,
          ComputeLaunchSize(value_count),
          gpu_inputs.values(),
          count,
          gpu_results.values()
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // The copy waits for the GPU function, so an error while it ran reports here.
  const std::size_t result_bytes = sizeof(SharedMathResult) * value_count;
  if (const cudaError_t error = cudaMemcpy(results, gpu_results.values(), result_bytes, cudaMemcpyDeviceToHost);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = gpu_inputs.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  return gpu_results.Free();
}

}  // namespace fluidity
