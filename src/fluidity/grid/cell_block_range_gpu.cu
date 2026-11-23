// The GPU half of the border search of a block range, private to grid/: the functions declared in
// grid/cell_block_range_gpu.h.
//
// Thin GPU functions, one for each step of the forward form of the border, each a grid-stride loop (or one thread) over
// a per-item rule of grid/border_block_steps.h, which the CPU form calls too and a CPU test replays for every thread
// block and thread. The one operation between threads is the integer atomic minimum of the failure key of the border so
// far, whose result does not depend on the order the threads apply it. The calls use the default stream, so each sees
// what the one before it, and the prefix sum between them, wrote. Launches, the wait and statuses go through
// base/gpu_launch.cuh.
// The marks' concurrent writes. The threads of step 3 (WriteBorderReadMarks) write only the byte value 1 into marks
// that no thread reads, with plain stores, never a read-modify-write and never a test of a mark: the marking functions
// (ApplyReadMark) write the whole chain of bigger blocks every time. So the marks after the launch are those of any
// order of the places. This relies on a byte store on the GPU leaving the neighbouring bytes unchanged, which CUDA
// guarantees; two stores of the same value to one byte leave that value. The marks are 0 before the launch for every
// block, those of the set included, and nothing sets them for the set or for the border so far: the blocks of the
// set are marked through their own cell places. The scan of the marks (WriteListedMarks) is a later launch of the same
// stream, so it starts after the marking launch has finished.
// Chunks: step 3 is one launch for each chunk of kSetBlocksPerMarkingLaunch blocks of the set (the CPU half calls
// WriteBorderReadMarksOnGpu once for each chunk), each followed by a wait, so no launch runs long enough for the
// display watchdog and a failed chunk stops the ones after it.
// Stack: the thread of WriteBorderReadMarks keeps the results of the readers as locals (see the header), 3,600 bytes of
// stack on sm_120 (kBorderSearchStackBytes); the other functions keep no array. The launch bound of one thread block
// of kThreadBlockSize threads for each multiprocessor makes the compiler fit that thread block into the registers of a
// multiprocessor, so the launch never fails for lack of registers.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range_gpu.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {
namespace {

// The CUDA atomics take unsigned long long for 64 bits; std::uint64_t is unsigned long on Linux and unsigned long long
// on Windows, both 64 bits wide.
static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t), "the 64-bit atomics act on the 64-bit key");

// The failure key of `status` as the CUDA atomics take it.
__device__ unsigned long long* ToAtomicFailureKey(GridBuildStatus* status) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the same 64 bits, as the atomics' type.
  return reinterpret_cast<unsigned long long*>(&status->failure_key);
}

// Reads the failure key that step 2 wrote in an earlier launch. The volatile read goes to memory.
__device__ bool ComputeHasFailed(const GridBuildStatus* status) {
  return *static_cast<const volatile std::uint64_t*>(&status->failure_key) != kNoFailureKey;
}

// Step 1 (a), one thread.
__global__ void ApplyBorderStatusReset(GridBuildStatus* status) {
  *status = ComputeResetBuildStatus();
}

// Step 1 (b): one item per block.
__global__ void __launch_bounds__(kThreadBlockSize) ResetBorderMarks(std::uint8_t* marks, std::uint32_t block_count) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t block = stride.first_item; block < block_count; block += stride.item_step) {
    marks[block] = 0;
  }
}

// Step 2: one item per entry of the border so far.
__global__ void __launch_bounds__(kThreadBlockSize) CheckBorderSoFar(
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t border_so_far_count,
    std::uint32_t block_count,
    GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t entry = stride.first_item; entry < border_so_far_count; entry += stride.item_step) {
    const std::uint64_t failure_key =
        ComputeBorderSoFarFailureKey(range, border_so_far, block_count, static_cast<std::uint32_t>(entry));
    if (failure_key != kNoFailureKey) {
      atomicMin(ToAtomicFailureKey(status), failure_key);
    }
  }
}

// Step 3: one item per place of the chunk of the set. After a failure of step 2 no thread marks anything. The marking
// functions are long: see the Stack sentence of the file header.
__global__ void __launch_bounds__(kThreadBlockSize, 1) WriteBorderReadMarks(
    GridTables tables,
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t border_so_far_count,
    SetPlaceChunk chunk,
    BorderWidth width,
    const GridBuildStatus* status,
    std::uint8_t* marks
) {
  if (ComputeHasFailed(status)) {
    return;
  }
  const GridStride stride = ComputeGridStride();
  for (std::size_t item = stride.first_item; item < chunk.set_place_count; item += stride.item_step) {
    ApplySetPlaceReadMarks(
        tables,
        range,
        border_so_far,
        border_so_far_count,
        chunk.first_set_place + static_cast<std::uint32_t>(item),
        width,
        marks
    );
  }
}

// Step 4: one item per block. After a failure the words are 0, so the prefix sum adds no stale word.
__global__ void __launch_bounds__(kThreadBlockSize) WriteListedMarks(
    const std::uint8_t* marks,
    CellBlockRange range,
    std::uint32_t block_count,
    const GridBuildStatus* status,
    std::uint32_t* sums
) {
  const GridStride stride = ComputeGridStride();
  const bool has_failed = ComputeHasFailed(status);
  for (std::size_t block = stride.first_item; block < block_count; block += stride.item_step) {
    sums[block] = has_failed ? 0U : ComputeListedMark(marks, range, static_cast<std::uint32_t>(block));
  }
}

// Step 6: one item per block, and thread 0 writes the total. The writes of two blocks never meet: each listed block has
// an entry of its own, the prefix sum's.
__global__ void __launch_bounds__(kThreadBlockSize) WriteBorderBlocks(
    const std::uint32_t* sums,
    std::uint32_t block_count,
    std::uint32_t border_block_room,
    CellBlockIndex* border_blocks,
    GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  if (stride.first_item == 0) {
    status->result_count = sums[block_count];
  }
  for (std::size_t block = stride.first_item; block < block_count; block += stride.item_step) {
    ApplyBorderBlockWriteAt(sums, static_cast<std::uint32_t>(block), border_block_room, border_blocks);
  }
}

// Starts a GPU function like ApplyGpuFunction and, when it did not start, waits for the GPU before returning, so that
// no earlier launch of the same call is still running (and writing) when the caller sees the failure.
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

// The launch of a GPU function that one thread runs.
constexpr LaunchSize kOneThread = {.thread_block_count = 1, .thread_block_size = 1};

// Reads the local memory per thread of one GPU function into *largest_bytes when it is larger.
template <typename Function>
cudaError_t ApplyLargerStack(Function function, std::uint64_t* largest_bytes) {
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, function);
  if (error == cudaSuccess) {
    *largest_bytes = std::max<std::uint64_t>(*largest_bytes, attributes.localSizeBytes);
  }
  return error;
}

}  // namespace

RunOnCallStatus ApplyBorderStatusResetOnGpu(GridBuildStatus* status) {
  return ApplyGpuFunctionAndWaitOnFailure(ApplyBorderStatusReset, kOneThread, status);
}

RunOnCallStatus ResetBorderMarksOnGpu(std::uint8_t* marks, std::uint32_t block_count) {
  // A launch of 0 thread blocks fails, and there is no mark to reset.
  if (block_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunctionAndWaitOnFailure(ResetBorderMarks, ComputeLaunchSize(block_count), marks, block_count);
}

RunOnCallStatus CheckBorderSoFarOnGpu(
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t border_so_far_count,
    std::uint32_t block_count,
    GridBuildStatus* status
) {
  if (border_so_far_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunctionAndWaitOnFailure(
      CheckBorderSoFar,
      ComputeLaunchSize(border_so_far_count),
      range,
      border_so_far,
      border_so_far_count,
      block_count,
      status
  );
}

RunOnCallStatus WriteBorderReadMarksOnGpu(
    const GridTables& tables,
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t border_so_far_count,
    SetPlaceChunk chunk,
    BorderWidth width,
    const GridBuildStatus* status,
    std::uint8_t* marks
) {
  if (chunk.set_place_count == 0) {
    return RunOnCallStatus::kDone;
  }
  if (const RunOnCallStatus started = ApplyGpuFunctionAndWaitOnFailure(
          WriteBorderReadMarks,
          ComputeLaunchSize(chunk.set_place_count),
          tables,
          range,
          border_so_far,
          border_so_far_count,
          chunk,
          width,
          status,
          marks
      );
      started != RunOnCallStatus::kDone) {
    return started;
  }
  // The wait ends this chunk before the next one starts, so a chunk that fails is reported here and no later chunk
  // runs, and the display watchdog sees each chunk as a launch of its own.
  return CheckGpuWork();
}

RunOnCallStatus WriteListedMarksOnGpu(
    const std::uint8_t* marks,
    CellBlockRange range,
    std::uint32_t block_count,
    const GridBuildStatus* status,
    std::uint32_t* sums
) {
  if (block_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunctionAndWaitOnFailure(
      WriteListedMarks,
      ComputeLaunchSize(block_count),
      marks,
      range,
      block_count,
      status,
      sums
  );
}

RunOnCallStatus WriteBorderBlocksOnGpu(
    const std::uint32_t* sums,
    std::uint32_t block_count,
    std::uint32_t border_block_room,
    CellBlockIndex* border_blocks,
    GridBuildStatus* status
) {
  // At least one thread, so a grid of no block still writes its total.
  const std::size_t item_count = block_count == 0 ? 1 : block_count;
  if (const RunOnCallStatus started = ApplyGpuFunctionAndWaitOnFailure(
          WriteBorderBlocks,
          ComputeLaunchSize(item_count),
          sums,
          block_count,
          border_block_room,
          border_blocks,
          status
      );
      started != RunOnCallStatus::kDone) {
    return started;
  }
  // The wait reports an error of any function started since the prefix sum's wait.
  return CheckGpuWork();
}

RunOnCallStatus WaitForBorderSearchOnGpu() {
  return CheckGpuWork();
}

RunOnCallStatus ComputeLargestBorderSearchStackOnGpu(std::uint64_t* bytes_per_thread) {
  std::uint64_t largest_bytes = 0;
  const std::array<cudaError_t, 6> errors = {
      ApplyLargerStack(ApplyBorderStatusReset, &largest_bytes),
      ApplyLargerStack(ResetBorderMarks, &largest_bytes),
      ApplyLargerStack(CheckBorderSoFar, &largest_bytes),
      ApplyLargerStack(WriteBorderReadMarks, &largest_bytes),
      ApplyLargerStack(WriteListedMarks, &largest_bytes),
      ApplyLargerStack(WriteBorderBlocks, &largest_bytes),
  };
  for (const cudaError_t error : errors) {
    if (error != cudaSuccess) {
      return ToRunOnCallStatus(error);
    }
  }
  *bytes_per_thread = largest_bytes;
  return RunOnCallStatus::kDone;
}

RunOnCallStatus ReserveBorderSearchStackOnGpu(std::uint64_t bytes_per_thread) {
  // Steps: 1 the stack now  2 raise it  3 check what the driver gave
  // Step 1.
  std::size_t current_bytes = 0;
  if (const cudaError_t error = cudaDeviceGetLimit(&current_bytes, cudaLimitStackSize); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (current_bytes >= bytes_per_thread) {
    return RunOnCallStatus::kDone;
  }
  // Step 2: the driver takes the memory now, so a GPU too full to hold it says so here, at build time.
  const cudaError_t set_error = cudaDeviceSetLimit(cudaLimitStackSize, static_cast<std::size_t>(bytes_per_thread));
  if (set_error == cudaErrorMemoryAllocation) {
    // Not a sticky error: reading it clears it, so the next GPU call does not report it again.
    static_cast<void>(cudaGetLastError());
    return RunOnCallStatus::kOutOfMemory;
  }
  if (set_error != cudaSuccess) {
    return ToRunOnCallStatus(set_error);
  }
  // Step 3: the driver may round the value; it must not be below what the marking function needs.
  std::size_t given_bytes = 0;
  if (const cudaError_t error = cudaDeviceGetLimit(&given_bytes, cudaLimitStackSize); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  return given_bytes >= bytes_per_thread ? RunOnCallStatus::kDone : RunOnCallStatus::kCallFailed;
}

RunOnCallStatus ComputeResidentThreadsOnGpu(std::uint64_t* resident_threads) {
  int device = 0;
  if (const cudaError_t error = cudaGetDevice(&device); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  int threads_per_multiprocessor = 0;
  if (const cudaError_t error =
          cudaDeviceGetAttribute(&threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  int multiprocessor_count = 0;
  if (const cudaError_t error = cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  *resident_threads =
      static_cast<std::uint64_t>(threads_per_multiprocessor) * static_cast<std::uint64_t>(multiprocessor_count);
  return RunOnCallStatus::kDone;
}

}  // namespace fluidity::grid
