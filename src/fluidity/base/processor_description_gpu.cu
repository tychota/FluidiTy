// The GPU half of ReadMemoryAvailable and ReadProcessorDescription, private to base/: the CUDA runtime's reading of
// the current GPU's memory, name and versions.
// Read: docs/method-paper/fluidity-method.typ <sec-memory-estimate>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/processor_description_gpu.h"
#include "fluidity/base/run_on_call.h"

namespace fluidity {

RunOnCallStatus ReadGpuMemoryAvailable(std::uint64_t* total_bytes, std::uint64_t* free_bytes) {
  // cudaFree(nullptr) frees nothing but creates the GPU context if it does not exist yet, so the context and the GPU
  // code loaded so far sit outside "free" before it is read. The allowed bytes computed from this reading then leave
  // room for them. GPU code loads when it is first launched, so code launched later may still take a little memory.
  if (const RunOnCallStatus status = ToRunOnCallStatus(cudaFree(nullptr)); status != RunOnCallStatus::kDone) {
    return status;
  }
  std::size_t read_free_bytes = 0;
  std::size_t read_total_bytes = 0;
  if (const cudaError_t error = cudaMemGetInfo(&read_free_bytes, &read_total_bytes); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  *total_bytes = read_total_bytes;
  *free_bytes = read_free_bytes;
  return RunOnCallStatus::kDone;
}

namespace {

// Writes a CUDA version number such as 12080 as "12.8". CUDA encodes major * 1000 + minor * 10. Leaves `text` empty
// when the query failed.
void WriteVersionText(cudaError_t status, int version, char* text, std::size_t text_size) {
  text[0] = '\0';
  if (status != cudaSuccess) {
    return;
  }
  std::snprintf(text, text_size, "%d.%d", version / 1000, (version % 1000) / 10);
}

}  // namespace

void ReadGpuProcessorDescription(char* model, char* driver, char* runtime, std::size_t text_size) {
  model[0] = '\0';
  int device_index = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device_index) == cudaSuccess
      && cudaGetDeviceProperties(&properties, device_index) == cudaSuccess) {
    std::snprintf(model, text_size, "%s", properties.name);
  }
  int driver_version = 0;
  WriteVersionText(cudaDriverGetVersion(&driver_version), driver_version, driver, text_size);
  int runtime_version = 0;
  WriteVersionText(cudaRuntimeGetVersion(&runtime_version), runtime_version, runtime, text_size);
}

}  // namespace fluidity
