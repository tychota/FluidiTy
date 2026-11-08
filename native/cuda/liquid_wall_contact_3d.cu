#include <cuda_runtime.h>

#include "fluidity/base/liquid_wall_contact_math_3d.h"
#include "liquid_boundary_3d_bridge.h"
namespace fluidity::cuda_internal {
namespace {
__global__ void LiquidWallContact3DKernel(const boundary_internal::ContactInput* input, BoundaryReport3D* output) {
  const auto result = boundary_internal::ResolveContact(*input);
  output->error = result.status();
  if (result.ok()) {
    output->contact = *result;
  }
#if defined(__CUDA_ARCH__)
  output->compiled_architecture = __CUDA_ARCH__;
#endif
}
}  // namespace
int LaunchLiquidWallContact3D(const boundary_internal::ContactInput* input, BoundaryReport3D* output, void* stream) {
  cudaGetLastError();
  LiquidWallContact3DKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(input, output);
  return static_cast<int>(cudaGetLastError());
}
}  // namespace fluidity::cuda_internal
