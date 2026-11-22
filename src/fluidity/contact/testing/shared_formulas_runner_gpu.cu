// The GPU runner of solid_shapes.shared_formulas.gpu.test.cc: ComputeSharedFormulaValues over every input, one thread
// per input. Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>

#include <cuda_runtime.h>

#include <cstddef>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/contact/testing/shared_formulas_runner_gpu.h"

namespace fluidity::contact {
namespace {

// results[i] = ComputeSharedFormulaValues(inputs[i]) for i below count.
__global__ void ComputeEverySharedFormula(const SharedFormulaInputs* inputs, int count, SharedFormulaValues* results) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < static_cast<std::size_t>(count); index += stride.item_step) {
    results[index] = ComputeSharedFormulaValues(inputs[index]);
  }
}

}  // namespace

RunOnCallStatus ComputeSharedFormulasOnGpu(const SharedFormulaInputs* inputs, int count, SharedFormulaValues* results) {
  const auto value_count = static_cast<std::size_t>(count);
  GpuArray<SharedFormulaInputs> gpu_inputs;
  GpuArray<SharedFormulaValues> gpu_results;
  if (const RunOnCallStatus status = gpu_inputs.Build(value_count); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_results.Build(value_count); status != RunOnCallStatus::kDone) {
    return status;
  }
  const std::size_t input_bytes = sizeof(SharedFormulaInputs) * value_count;
  if (const cudaError_t error = cudaMemcpy(gpu_inputs.values(), inputs, input_bytes, cudaMemcpyHostToDevice);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = ApplyGpuFunction(
          ComputeEverySharedFormula,
          ComputeLaunchSize(value_count),
          gpu_inputs.values(),
          count,
          gpu_results.values()
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // The copy waits for the GPU function, so an error while it ran reports here.
  const std::size_t result_bytes = sizeof(SharedFormulaValues) * value_count;
  if (const cudaError_t error = cudaMemcpy(results, gpu_results.values(), result_bytes, cudaMemcpyDeviceToHost);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = gpu_inputs.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  return gpu_results.Free();
}

}  // namespace fluidity::contact
