#include <cuda_runtime.h>

#include "probe.cuh"
#include "probe_bridge.h"

namespace fluidity {
namespace cuda {

__global__ void ProbeKernel(Vec2* device_output, std::size_t sample_count) {
  const std::size_t sample_index = threadIdx.x;
  if (sample_index >= sample_count) {
    return;
  }

  // The operands and intermediate results are exact binary fractions for the probe's inputs.
  const double sample_coordinate = static_cast<double>(sample_index);
  const Vec2 sample{
      .x = sample_coordinate + 1.0,
      .y = 0.5 - sample_coordinate,
  };
  const Vec2 offset{0.5, -1.0};
  device_output[sample_index] = (sample + offset) * 2.0 - 0.5 * sample;

#if defined(__CUDA_ARCH__)
  if (sample_index == 0) {
    device_output[sample_count + 1].x = __CUDA_ARCH__;
  }
#endif
}

}  // namespace cuda

int LaunchProbeKernel(Vec2* device_output, std::size_t sample_count, void* stream) {
  constexpr unsigned int kThreadsPerBlock = 32;
  void* arguments[] = {&device_output, &sample_count};
  // Return this call's status without reading or clearing the thread-local last error.
  return cudaLaunchKernel(
      cuda::ProbeKernel,
      dim3(1),
      dim3(kThreadsPerBlock),
      arguments,
      0,
      static_cast<cudaStream_t>(stream)
  );
}

}  // namespace fluidity
