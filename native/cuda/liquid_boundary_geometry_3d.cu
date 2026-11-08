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
}  // namespace fluidity::cuda_internal
