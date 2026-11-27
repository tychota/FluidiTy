// The plumbing every GPU half shares: starting a GPU function, waiting for the GPU, reading CUDA errors, and the small
// helpers that GPU functions of several files need.
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
//   Three more helpers serve the GPU halves that run a chain of launches over one status record: a launch that waits
//   when it did not start, a read of the stack a GPU function keeps, and the two accesses to a 64-bit value that
//   other threads change (the atomic pointer and the read that goes to memory).
//   Example: ApplyGpuFunction(MultiplyPairs, ComputeLaunchSize(count), first, second, count, products) starts one
//   thread per product, at most 65,535 thread blocks, and returns kDone or the status of the failed start.
//
// Owns:    ToRunOnCallStatus, ApplyGpuFunction, ApplyGpuFunctionAndWaitOnFailure, CheckGpuWork, kOneThread,
//          ApplyLargerStack, ToAtomicUint64, ComputeCurrentUint64, GridStride, ComputeGridStride, GpuArray
// Uses:    RunOnCallStatus and LaunchSize (base/run_on_call.h), ApplyLastRunOnCallError (base/run_on_call_error.h), the
//          CUDA runtime
// Used by: every `<name>_gpu.cu` file; never a `.h` or `.cc` file, which the C++ compiler alone builds
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "fluidity/base/run_on_call.h"
#include "fluidity/base/run_on_call_error.h"

namespace fluidity {

// Converts the error of any CUDA call that allocates nothing (a launch, a wait, a copy, a fill) to a status:
// cudaSuccess is kDone, anything else kCallFailed, even an error that names memory, because outputs may already be
// partly written and kOutOfMemory promises that nothing was. An allocation reports kOutOfMemory itself (GpuArray).
// The status values are all it gives back; the name and the text of the failed call (cudaGetErrorName and
// cudaGetErrorString) are written by ApplyLastRunOnCallError (base/run_on_call_error.h), where a test reads them.
inline RunOnCallStatus ToRunOnCallStatus(cudaError_t error) {
  if (error == cudaSuccess) {
    return RunOnCallStatus::kDone;
  }
  ApplyLastRunOnCallError(cudaGetErrorName(error), cudaGetErrorString(error));
  return RunOnCallStatus::kCallFailed;
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

// Starts a GPU function like ApplyGpuFunction and, when it did not start, waits for the GPU before returning.
//
// Why: a call made of several launches (a build, a check) may have started earlier launches that are still running and
//   still write its buffers. The caller that sees the failure may free or reuse those buffers at once, so it must not
//   see the failure before the earlier launches have ended. The wait is for that; its own result is dropped, because
//   the failure to report is the one of the launch.
// Requires: the launch size and the arguments are those of ApplyGpuFunction.
// Example: ApplyGpuFunctionAndWaitOnFailure(ResetMarks, ComputeLaunchSize(block_count), marks, block_count).
// Fails: kCallFailed  the launch did not start, or an earlier GPU call left an error that this read reports
template <typename... Parameters>
RunOnCallStatus ApplyGpuFunctionAndWaitOnFailure(
    void (*function)(Parameters...),
    LaunchSize launch_size,
    std::type_identity_t<Parameters>... arguments
) {
  const RunOnCallStatus started = ApplyGpuFunction(function, launch_size, arguments...);
  if (started != RunOnCallStatus::kDone) {
    static_cast<void>(CheckGpuWork());
  }
  return started;
}

// The launch of a GPU function that one thread runs: one thread block of one thread.
constexpr LaunchSize kOneThread = {.thread_block_count = 1, .thread_block_size = 1};

// Raises *largest_bytes to the local memory per thread of `function` when that is larger, and returns the CUDA error
// of reading it. Fails with the error of cudaFuncGetAttributes, for a pointer that is no GPU function of this build.
//
// Why: a thread of a GPU function keeps its stack (arrays and spilled registers) in local memory, and the driver gives
//   each thread a default stack of 1,024 bytes. A function that needs more than that is held by the driver, which
//   must be told to give a larger stack before the launch. The caller reads the need of each function of one file with
//   this and compares the largest with the default, so a function that outgrows it is noticed by a test.
// Requires: `function` is a pointer to a GPU function; largest_bytes is not null.
// Example: ApplyLargerStack(WritePieceMap, &largest_bytes) after ApplyLargerStack(CheckPieceBorder, &largest_bytes).
template <typename Function>
cudaError_t ApplyLargerStack(Function function, std::uint64_t* largest_bytes) {
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, function);
  if (error == cudaSuccess) {
    *largest_bytes = std::max<std::uint64_t>(*largest_bytes, attributes.localSizeBytes);
  }
  return error;
}

// The CUDA atomics take unsigned long long for 64 bits; std::uint64_t is unsigned long on Linux and unsigned long long
// on Windows, both 64 bits wide.
static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t), "the 64-bit atomics act on 64-bit values");

// The 64-bit value at `value` as the CUDA atomics take it: the same address, the type the atomics are declared with.
// Requires: `value` is aligned to 8 bytes and is not null. Call it inside a GPU function.
// Example: atomicMin(ToAtomicUint64(&status->failure_key), failure_key).
__device__ inline unsigned long long* ToAtomicUint64(std::uint64_t* value) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the same 64 bits, as the atomics' type.
  return reinterpret_cast<unsigned long long*>(value);
}

// The value at `value` as it is in memory now, for a 64-bit value that other threads or an earlier launch change. The
// volatile read goes to memory, never a copy that the thread kept from before. Call it inside a GPU function.
// Requires: `value` is aligned to 8 bytes and is not null.
__device__ inline std::uint64_t ComputeCurrentUint64(const std::uint64_t* value) {
  return *static_cast<const volatile std::uint64_t*>(value);
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
      ApplyLastRunOnCallError(cudaGetErrorName(error), cudaGetErrorString(error));
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
