// The GPU half of the check of a cell list, private to grid/: the functions declared in grid/staggered_grid_gpu.h.
//
// Six thin GPU functions, one per step of method paper <proc-build-grid-in-parallel> P1 to P6 and the failure half of
// P12, each a grid-stride loop (or one thread) over a per-item rule of grid/grid_build_steps.h, which the CPU check
// calls too and a CPU test replays for every thread block and thread; the copy in sorted order is one assignment per
// item. The only operation between threads is the integer
// atomic minimum of the failure key: its final value does not depend on the order the threads apply it. Every call
// uses the default stream, so each function sees what the one before it, and the stable orders between them, wrote.
// Launches, the wait and statuses go through base/gpu_launch.cuh.
// Stack: no function keeps a local array. The largest locals are the CellCheck of a sorted position (24 bytes) and the
// coordinates and loop counters of CheckSiblingsAt and CheckSizeStepAt, all in registers or a few dozen bytes of
// stack; ComputeLargestStackOnGpu reads what the compiler gave.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid-in-parallel>, <sec-parallel-build>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid_gpu.h"

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

// Reads the failure key that other threads may be offering keys to at the same time. The volatile read goes to memory,
// never a value the thread kept from before.
__device__ std::uint64_t ComputeCurrentFailureKey(const GridBuildStatus* status) {
  return *static_cast<const volatile std::uint64_t*>(&status->failure_key);
}

// Step P1, one thread.
__global__ void ApplyBuildStatusReset(GridBuildStatus* status) {
  *status = ComputeResetBuildStatus();
}

// Step P2: one item per input index.
__global__ void __launch_bounds__(
    kThreadBlockSize
) CheckCellIds(GridFrame frame, const CellId* input_cells, std::uint32_t input_count, GridBuildStatus* status) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < input_count; index += stride.item_step) {
    const std::uint64_t failure_key = ComputeCellIdFailureKey(frame, input_cells, index);
    if (failure_key != kNoFailureKey) {
      atomicMin(ToAtomicFailureKey(status), failure_key);
    }
  }
}

// Step P3, one pass of the keys: one item per input index.
__global__ void __launch_bounds__(
    kThreadBlockSize
) WriteCellSortKeys(const CellId* input_cells, std::uint32_t input_count, int pass, std::uint64_t* sort_keys) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < input_count; index += stride.item_step) {
    sort_keys[index] = ToCellSortKey(input_cells[index], pass);
  }
}

// Step P3, the copy in sorted order: one item per sorted position.
__global__ void __launch_bounds__(
    kThreadBlockSize
) CopySortedCells(const CellId* input_cells, const std::uint32_t* order, std::uint32_t count, CellId* sorted_cells) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t position = stride.first_item; position < count; position += stride.item_step) {
    sorted_cells[position] = input_cells[order[position]];
  }
}

// Steps P4 to P6: one item per sorted position. The key is read again for every item, so a thread with several items
// stops as soon as an earlier step's key is there; a key of this step never stops it.
__global__ void __launch_bounds__(
    kThreadBlockSize
) CheckSortedPositions(const CellId* sorted_cells, std::uint32_t count, int step, GridBuildStatus* status) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t position = stride.first_item; position < count; position += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), step)) {
      return;
    }
    const std::uint64_t failure_key =
        ComputeSortedCellFailureKey(sorted_cells, count, static_cast<std::uint32_t>(position), step);
    if (failure_key != kNoFailureKey) {
      atomicMin(ToAtomicFailureKey(status), failure_key);
    }
  }
}

// Step P12, the failure half, one thread.
__global__ void WriteFailureCells(
    GridFrame frame,
    const CellId* input_cells,
    std::uint32_t input_count,
    const CellId* sorted_cells,
    std::uint32_t count,
    GridBuildStatus* status
) {
  ApplyFailureCells(frame, input_cells, input_count, sorted_cells, count, status);
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

RunOnCallStatus ApplyBuildStatusResetOnGpu(GridBuildStatus* status) {
  return ApplyGpuFunction(ApplyBuildStatusReset, kOneThread, status);
}

RunOnCallStatus
CheckCellIdsOnGpu(GridFrame frame, const CellId* input_cells, std::uint32_t input_count, GridBuildStatus* status) {
  // A launch of 0 thread blocks fails, and there is no item to check.
  if (input_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(CheckCellIds, ComputeLaunchSize(input_count), frame, input_cells, input_count, status);
}

RunOnCallStatus
WriteCellSortKeysOnGpu(const CellId* input_cells, std::uint32_t input_count, int pass, std::uint64_t* sort_keys) {
  if (input_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(WriteCellSortKeys, ComputeLaunchSize(input_count), input_cells, input_count, pass, sort_keys);
}

RunOnCallStatus
CopySortedCellsOnGpu(const CellId* input_cells, const std::uint32_t* order, std::uint32_t count, CellId* sorted_cells) {
  if (count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(CopySortedCells, ComputeLaunchSize(count), input_cells, order, count, sorted_cells);
}

RunOnCallStatus
CheckSortedCellsOnGpu(const CellId* sorted_cells, std::uint32_t count, int step, GridBuildStatus* status) {
  if (count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(CheckSortedPositions, ComputeLaunchSize(count), sorted_cells, count, step, status);
}

RunOnCallStatus WriteFailureCellsOnGpu(
    GridFrame frame,
    const CellId* input_cells,
    std::uint32_t input_count,
    const CellId* sorted_cells,
    std::uint32_t count,
    GridBuildStatus* status
) {
  if (const RunOnCallStatus started =
          ApplyGpuFunction(WriteFailureCells, kOneThread, frame, input_cells, input_count, sorted_cells, count, status);
      started != RunOnCallStatus::kDone) {
    return started;
  }
  // The wait reports an error of a GPU function started since the last BuildStableOrder, whose own wait reports one
  // of the functions before it.
  return CheckGpuWork();
}

RunOnCallStatus ComputeLargestStackOnGpu(std::uint64_t* bytes_per_thread) {
  std::uint64_t largest_bytes = 0;
  const std::array<cudaError_t, 6> errors = {
      ApplyLargerStack(ApplyBuildStatusReset, &largest_bytes),
      ApplyLargerStack(CheckCellIds, &largest_bytes),
      ApplyLargerStack(WriteCellSortKeys, &largest_bytes),
      ApplyLargerStack(CopySortedCells, &largest_bytes),
      ApplyLargerStack(CheckSortedPositions, &largest_bytes),
      ApplyLargerStack(WriteFailureCells, &largest_bytes),
  };
  for (const cudaError_t error : errors) {
    if (error != cudaSuccess) {
      return ToRunOnCallStatus(error);
    }
  }
  *bytes_per_thread = largest_bytes;
  return RunOnCallStatus::kDone;
}

}  // namespace fluidity::grid
