#pragma once

#include <cuda_runtime_api.h>

#include "viscosity_velocity_block_3d_bridge.h"

namespace fluidity::cuda_internal {

// Enqueues the velocity-block solve on the supplied stream. The caller owns all buffers
// and retains them through synchronization.
cudaError_t LaunchViscousVelocityBlock3D(ViscousVelocityDevice3D device, cudaStream_t stream);

}  // namespace fluidity::cuda_internal
