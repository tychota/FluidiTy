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
template <boundary_internal::GeometryMode kMode>
__global__ void LiquidBoundaryGeometryBatch3DKernel(
    const boundary_internal::GeometryInput* queries,
    boundary_internal::GeometryBatchResult* results,
    std::size_t count
) {
  const std::size_t ordinal = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ordinal >= count) {
    return;
  }
  auto& output = results[ordinal];
#if defined(__CUDA_ARCH__)
  output.compiled_architecture = __CUDA_ARCH__;
#endif
  const auto& query = queries[ordinal];
  if (query.mode != kMode) {
    output.error = boundary_internal::Error::kInvalid;
    return;
  }
  const auto result = boundary_internal::ComputeGeometryForMode<kMode>(query);
  output.error = result.status();
  if (result.ok()) {
    output.output = *result;
  }
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
    boundary_internal::GeometryMode mode,
    void* stream
) {
  if (count == 0 || count > boundary_internal::kGeometryQueryBatchCapacity3D) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr unsigned kThreadsPerBlock = 32;
  const auto block_count = static_cast<unsigned>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  cudaGetLastError();
  const auto cuda_stream = static_cast<cudaStream_t>(stream);
  using boundary_internal::GeometryMode;
  switch (mode) {
    case GeometryMode::kCell:
      LiquidBoundaryGeometryBatch3DKernel<GeometryMode::kCell>
          <<<block_count, kThreadsPerBlock, 0, cuda_stream>>>(queries, results, count);
      break;
    case GeometryMode::kFace:
      LiquidBoundaryGeometryBatch3DKernel<GeometryMode::kFace>
          <<<block_count, kThreadsPerBlock, 0, cuda_stream>>>(queries, results, count);
      break;
    case GeometryMode::kCurvature:
      LiquidBoundaryGeometryBatch3DKernel<GeometryMode::kCurvature>
          <<<block_count, kThreadsPerBlock, 0, cuda_stream>>>(queries, results, count);
      break;
    case GeometryMode::kLink:
      LiquidBoundaryGeometryBatch3DKernel<GeometryMode::kLink>
          <<<block_count, kThreadsPerBlock, 0, cuda_stream>>>(queries, results, count);
      break;
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaGetLastError());
}
}  // namespace fluidity::cuda_internal
