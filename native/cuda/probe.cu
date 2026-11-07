#include <cuda_runtime.h>

#include "probe.cuh"
#include "probe_bridge.h"

namespace fluidity {
namespace cuda {

__global__ void ProbeKernel(Vec2* device_output, std::size_t count) {
  const std::size_t index = threadIdx.x;
  if (index >= count) {
    return;
  }

  // The operands and intermediate results are exact binary fractions for the probe's inputs.
  const double coordinate = static_cast<double>(index);
  const Vec2 sample{coordinate + 1.0, 0.5 - coordinate};
  const Vec2 offset{0.5, -1.0};
  device_output[index] = (sample + offset) * 2.0 - 0.5 * sample;

#if defined(__CUDA_ARCH__)
  if (index == 0) {
    device_output[count + 1].x = __CUDA_ARCH__;
  }
#endif
}

}  // namespace cuda

int LaunchProbeKernel(Vec2* device_output, std::size_t count, void* stream) {
  constexpr unsigned int kThreadsPerBlock = 32;
  cuda::ProbeKernel<<<1, kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(device_output, count);
  return cudaGetLastError();
}

}  // namespace fluidity
