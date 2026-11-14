// The GPU half of Buffer, private to base/: one CUDA runtime call per function of base/buffer_gpu.h.
//
// Every call uses the default stream, so the GPU runs them in the order they are made. A read returns when the CPU
// memory holds the values. A write returns when its CPU memory may be reused. A GPU-to-GPU copy and a fill wait for
// the GPU (CheckGpuWork), so their own errors are reported by the call that made them. Other streams are not used
// yet. The statuses come from base/gpu_launch.cuh.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>

#include <cuda_runtime.h>

#include <cstddef>

#include "fluidity/base/buffer_gpu.h"
#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"

namespace fluidity {

RunOnCallStatus AllocateGpuBytes(std::size_t byte_count, void** bytes) {
  if (cudaMalloc(bytes, byte_count) != cudaSuccess) {
    // A failed cudaMalloc leaves no promise about the pointer; the caller must not free it. Whatever the reason, the
    // GPU could not provide the memory.
    *bytes = nullptr;
    return RunOnCallStatus::kOutOfMemory;
  }
  return RunOnCallStatus::kDone;
}

RunOnCallStatus FreeGpuBytes(void* bytes) {
  return ToRunOnCallStatus(cudaFree(bytes));
}

RunOnCallStatus WriteGpuBytes(const void* from, std::size_t byte_count, void* to) {
  return ToRunOnCallStatus(cudaMemcpy(to, from, byte_count, cudaMemcpyHostToDevice));
}

RunOnCallStatus ReadGpuBytes(const void* from, std::size_t byte_count, void* to) {
  return ToRunOnCallStatus(cudaMemcpy(to, from, byte_count, cudaMemcpyDeviceToHost));
}

RunOnCallStatus CopyGpuBytes(const void* from, std::size_t byte_count, void* to) {
  // A copy between two GPU addresses may return before the GPU has finished; the wait reports an error of the copy
  // itself here instead of at some later, unrelated call.
  if (const RunOnCallStatus status = ToRunOnCallStatus(cudaMemcpy(to, from, byte_count, cudaMemcpyDeviceToDevice));
      status != RunOnCallStatus::kDone) {
    return status;
  }
  return CheckGpuWork();
}

RunOnCallStatus ZeroGpuBytes(void* bytes, std::size_t byte_count) {
  // cudaMemset may return before the GPU has finished; the wait makes the zeros visible to every later call and
  // reports an error of the fill itself.
  if (const RunOnCallStatus status = ToRunOnCallStatus(cudaMemset(bytes, 0, byte_count));
      status != RunOnCallStatus::kDone) {
    return status;
  }
  return CheckGpuWork();
}

RunOnCallStatus AllocateLockedBytes(std::size_t byte_count, void** bytes) {
  if (cudaHostAlloc(bytes, byte_count, cudaHostAllocDefault) != cudaSuccess) {
    *bytes = nullptr;
    return RunOnCallStatus::kOutOfMemory;
  }
  return RunOnCallStatus::kDone;
}

RunOnCallStatus FreeLockedBytes(void* bytes) {
  return ToRunOnCallStatus(cudaFreeHost(bytes));
}

}  // namespace fluidity
