// The GPU half of ReadMemoryAvailable, private to base/: the CUDA runtime's reading of the current GPU's memory.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/processor_description_gpu.h"
#include "fluidity/base/run_on_call.h"

namespace fluidity {

RunOnCallStatus ReadGpuMemoryAvailable(std::uint64_t* total_bytes, std::uint64_t* free_bytes) {
  std::size_t read_free_bytes = 0;
  std::size_t read_total_bytes = 0;
  if (const cudaError_t error = cudaMemGetInfo(&read_free_bytes, &read_total_bytes); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  *total_bytes = read_total_bytes;
  *free_bytes = read_free_bytes;
  return RunOnCallStatus::kDone;
}

}  // namespace fluidity
