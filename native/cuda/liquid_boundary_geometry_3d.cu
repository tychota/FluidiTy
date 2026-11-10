#include <cuda_runtime.h>

#include "liquid_boundary_3d_bridge.h"
namespace fluidity::cuda_internal {
namespace {
__global__ void
LiquidBoundaryGeometry3DKernel(const boundary_internal::GeometryInput* input, BoundaryReport3D* output) {
  const auto result = boundary_internal::ComputeGeometry(*input);
  output->error = result.status();
  if (result.ok()) {
    output->geometry = *result;
  }
#if defined(__CUDA_ARCH__)
  output->compiled_architecture = __CUDA_ARCH__;
#endif
}
__global__ void LiquidBoundaryGeometryBatch3DKernel(
    const boundary_internal::GeometryInput* queries,
    boundary_internal::GeometryBatchResult* results,
    std::size_t count
) {
  const std::size_t ordinal = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ordinal >= count) {
    return;
  }
  const auto result = boundary_internal::ComputeGeometry(queries[ordinal]);
  auto& output = results[ordinal];
  output.error = result.status();
  if (result.ok()) {
    output.output = *result;
  }
#if defined(__CUDA_ARCH__)
  output.compiled_architecture = __CUDA_ARCH__;
#endif
}
}  // namespace
int LaunchLiquidBoundaryGeometry3D(
    const boundary_internal::GeometryInput* input,
    BoundaryReport3D* output,
    void* stream
) {
  cudaGetLastError();
  LiquidBoundaryGeometry3DKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(input, output);
  return static_cast<int>(cudaGetLastError());
}
int LaunchLiquidBoundaryGeometryBatch3D(
    const boundary_internal::GeometryInput* queries,
    boundary_internal::GeometryBatchResult* results,
    std::size_t count,
    void* stream
) {
  if (count == 0 || count > boundary_internal::kGeometryQueryBatchCapacity3D) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned kThreadsPerBlock = 32;
  const auto block_count = static_cast<unsigned>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  cudaGetLastError();
  LiquidBoundaryGeometryBatch3DKernel<<<block_count, kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(
      queries,
      results,
      count
  );
  return static_cast<int>(cudaGetLastError());
}
}  // namespace fluidity::cuda_internal
