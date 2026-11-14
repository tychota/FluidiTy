// The plumbing every GPU half shares: starting a GPU function, waiting for the GPU, and reading CUDA errors.
//
// Purpose: give every `<name>_gpu.cu` file one way to start a GPU function and check it, one way to wait for the
//   GPU, one way to walk the items of a thread and one holder of temporary GPU memory, so no launch goes unchecked
//   and every GPU half reports its failures as a RunOnCallStatus (base/run_on_call.h).
//
// Idea: ApplyGpuFunction starts a GPU function and returns at once with how the start went; an error while the
//   function runs is reported by the next call that waits for the GPU: CheckGpuWork, or a copy to CPU memory. Every
//   call uses the default stream, so the GPU runs the calls in the order they are made. Inside a GPU function,
//   ComputeGridStride gives a thread its first item and the step to its next one. GpuArray holds GPU memory for one
//   call and frees it on every path. The pattern of a GPU half: each GPU function is a thin loop that calls a shared
//   per-item function (marked FLUIDITY_CPU_AND_GPU), which CPU tests run over every item; the `.cu` file keeps only
//   the launches, the waits and the memory.
//   Example: ApplyGpuFunction(MultiplyPairs, ComputeLaunchSize(count), first, second, count, products) starts one
//   thread per product, at most 65,535 thread blocks, and returns kDone or the status of the failed start.
//
// Owns:    ToRunOnCallStatus, ApplyGpuFunction, CheckGpuWork, GridStride, ComputeGridStride, GpuArray
// Uses:    RunOnCallStatus and LaunchSize (base/run_on_call.h), the CUDA runtime
// Used by: every `<name>_gpu.cu` file; never a `.h` or `.cc` file, which the C++ compiler alone builds
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <type_traits>

#include "fluidity/base/run_on_call.h"

namespace fluidity {

// Converts the error of any CUDA call that allocates nothing (a launch, a wait, a copy, a fill) to a status:
// cudaSuccess is kDone, anything else kCallFailed, even an error that names memory, because outputs may already be
// partly written and kOutOfMemory promises that nothing was. An allocation reports kOutOfMemory itself (GpuArray).
inline RunOnCallStatus ToRunOnCallStatus(cudaError_t error) {
  return error == cudaSuccess ? RunOnCallStatus::kDone : RunOnCallStatus::kCallFailed;
}

// Starts `function` on the GPU with `launch_size` and `arguments`, then checks that it started.
//
// Method: launches on the default stream and reads cudaGetLastError, which reports a launch that could not start (a
//   thread block count of 0, too many threads, GPU code built for another GPU) and clears that error. It does not wait:
//   an error while the function runs is reported by the next call that waits for the GPU.
// Example: ApplyGpuFunction(FlagDestinationsOutOfRange, ComputeLaunchSize(item_count), destination, item_count,
//   destination_count, flag).
// Fails: kCallFailed  the launch did not start, or an earlier GPU call left an error that this read reports
template <typename... Parameters>
RunOnCallStatus ApplyGpuFunction(
    void (*function)(Parameters...),
    LaunchSize launch_size,
    std::type_identity_t<Parameters>... arguments
) {
  function<<<launch_size.thread_block_count, launch_size.thread_block_size>>>(arguments...);
  return ToRunOnCallStatus(cudaGetLastError());
}

// Waits for every GPU call made so far and checks how they ended.
// Fails: kCallFailed  a GPU function or copy failed while the GPU ran it
inline RunOnCallStatus CheckGpuWork() {
  return ToRunOnCallStatus(cudaDeviceSynchronize());
}

// The items of one thread in a grid-stride loop: first_item, first_item + item_step, and so on.
struct GridStride {
  std::size_t first_item = 0;  // The global index of the thread.
  std::size_t item_step = 0;  // The number of threads in the launch.
};

// Computes this thread's place in a grid-stride loop. Call it inside a GPU function.
// Example: thread 3 of thread block 1, thread blocks of 256 threads, 4 thread blocks: first_item 259, item_step 1024.
__device__ inline GridStride ComputeGridStride() {
  return {
      .first_item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
      .item_step = static_cast<std::size_t>(gridDim.x) * blockDim.x,
  };
}

// GPU memory for `count` values of T, for one call. An early return frees it in the destructor, so every path
// frees it; the normal path calls Free to learn whether freeing worked.
template <typename T>
class GpuArray {
 public:
  GpuArray() = default;
  GpuArray(const GpuArray&) = delete;
  GpuArray& operator=(const GpuArray&) = delete;
  ~GpuArray() {
    // Reached with memory only after another call already failed, and that failure is the one reported.
    if (values_ != nullptr) {
      cudaFree(values_);
    }
  }

  // Takes `count` values, at least 1. On failure values() stays null.
  // Fails: kOutOfMemory  the byte count would wrap around, or cudaMalloc failed for any reason
  RunOnCallStatus Build(std::size_t count) {
    // The byte count must not wrap around, or the memory would be smaller than asked for.
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return RunOnCallStatus::kOutOfMemory;
    }
    const cudaError_t error = cudaMalloc(&values_, sizeof(T) * count);
    // A failed cudaMalloc leaves no promise about the pointer; the destructor must not free it. Whatever the reason,
    // the GPU could not provide the memory, and nothing was written.
    if (error != cudaSuccess) {
      values_ = nullptr;
      return RunOnCallStatus::kOutOfMemory;
    }
    return RunOnCallStatus::kDone;
  }

  // Frees the memory now and reports whether freeing worked.
  RunOnCallStatus Free() {
    T* const values = values_;
    values_ = nullptr;
    return ToRunOnCallStatus(cudaFree(values));
  }

  [[nodiscard]] T* values() const {
    return values_;
  }

 private:
  T* values_ = nullptr;
};

}  // namespace fluidity
