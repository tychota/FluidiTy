// The GPU half of the check of a cell list and of the first steps of a build, private to grid/: the functions declared
// in grid/staggered_grid_gpu.h.
//
// Thin GPU functions, one per step of method paper <proc-build-grid-in-parallel> P1 to P6, the parts of P7 and P8 and
// the failure half of P12, each a grid-stride loop (or one thread) over a per-item rule of grid/grid_build_steps.h,
// which the CPU check calls too and a CPU test replays for every thread block and thread; the copy in sorted order is
// one assignment per item. The only operation between threads is the integer atomic minimum of the failure key: its
// final value does not depend on the order the threads apply it. Every call uses the default stream, so each function
// sees what the one before it, and the stable orders and prefix sums between them, wrote.
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

// Step P7 (a): one item per sorted active cell. Its count, or 0 after an earlier step's failure: the prefix sum that
// follows reads every one of the active_count words, never a word of an earlier launch.
__global__ void __launch_bounds__(kThreadBlockSize) WriteSplitCellCounts(
    const CellId* cell_ids,
    std::uint32_t active_count,
    const GridBuildStatus* status,
    std::uint32_t* split_cell_counts
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t position = stride.first_item; position < active_count; position += stride.item_step) {
    const bool has_earlier_failure = ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kCellPlacesStep);
    split_cell_counts[position] =
        has_earlier_failure ? 0U : static_cast<std::uint32_t>(ComputeSplitCellCountToAdd(cell_ids[position]));
  }
}

// Offers a key from the one thread of a launch. No other thread runs, so a plain minimum keeps the smaller key.
__device__ void ApplySmallerFailureKey(std::uint64_t failure_key, GridBuildStatus* status) {
  if (failure_key < status->failure_key) {
    status->failure_key = failure_key;
  }
}

// Step P7 (b), one thread: the total of the counts is the split cells.
__global__ void CheckCellPlaces(
    const std::uint32_t* sums,
    std::uint32_t active_count,
    std::uint64_t room_cell_places,
    std::uint32_t sort_count,
    GridBuildStatus* status
) {
  const std::uint32_t split_cells = sums[active_count];
  status->split_cells = split_cells;
  const std::uint64_t cell_count = std::uint64_t{active_count} + split_cells;
  ApplySmallerFailureKey(ComputeCellPlacesFailureKey(cell_count, room_cell_places, sort_count), status);
}

// Step P8, the block count, one thread: the total of the marks is the blocks.
__global__ void CheckBlockCount(
    const std::uint32_t* sums,
    std::uint32_t sort_count,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
) {
  const std::uint32_t cell_blocks = sums[sort_count];
  status->cell_blocks = cell_blocks;
  ApplySmallerFailureKey(ComputeRoomFailureKey(cell_blocks, max_cell_blocks, kBlockCountStep), status);
}

// Step P7 (c): one item per sorted active cell. After a failure of step 5 the sums may point past the cell ids.
__global__ void __launch_bounds__(kThreadBlockSize) WriteSplitCells(
    CellId* cell_ids,
    const std::uint32_t* sums,
    std::uint32_t active_count,
    const GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t position = stride.first_item; position < active_count; position += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kBlockCountStep)) {
      return;
    }
    ApplySplitCellWritesAt(cell_ids, sums, active_count, static_cast<std::uint32_t>(position));
  }
}

// Step P7 (d): one item per place below the sort count. The used count needs the split cells, which only the status
// record holds.
__global__ void __launch_bounds__(
    kThreadBlockSize
) WriteSortTail(CellId* cell_ids, std::uint32_t active_count, std::uint32_t sort_count, const GridBuildStatus* status) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t place = stride.first_item; place < sort_count; place += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kBlockCountStep)) {
      return;
    }
    // Step 5 passed, so the active and split cells fit 64 max_cell_blocks, below 2^32.
    const std::uint32_t used_count = active_count + status->split_cells;
    ApplySortTailFillAt(cell_ids, used_count, static_cast<std::uint32_t>(place));
  }
}

// Step P8, one pass of the block keys: one item per place below the sort count.
__global__ void __launch_bounds__(kThreadBlockSize) WriteBlockSortKeys(
    const CellId* cell_ids,
    std::uint32_t sort_count,
    int pass,
    const GridBuildStatus* status,
    std::uint64_t* sort_keys
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t place = stride.first_item; place < sort_count; place += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kBlockCountStep)) {
      return;
    }
    sort_keys[place] = ToBlockSortKey(cell_ids[place], pass);
  }
}

// Step P8, the marks: one item per sorted place. Its mark, or 0 after an earlier step's failure, step 5 included: the
// prefix sum that follows reads every one of the sort_count words.
__global__ void __launch_bounds__(kThreadBlockSize) WriteBlockStartMarks(
    const CellId* cell_ids,
    const std::uint32_t* order,
    std::uint32_t sort_count,
    const GridBuildStatus* status,
    std::uint32_t* marks
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t position = stride.first_item; position < sort_count; position += stride.item_step) {
    const bool has_earlier_failure = ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kBlockCountStep);
    marks[position] =
        has_earlier_failure ? 0U : ComputeBlockStartMark(cell_ids, order, static_cast<std::uint32_t>(position));
  }
}

// Step P8, the scatter: one item per sorted place. After a failure of step 6 the sums may point past the block ids.
__global__ void __launch_bounds__(kThreadBlockSize) WriteBlockIds(
    CellBlockId* block_ids,
    const CellId* cell_ids,
    const std::uint32_t* order,
    const std::uint32_t* sums,
    std::uint32_t sort_count,
    const GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t position = stride.first_item; position < sort_count; position += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kBlockFaultStep)) {
      return;
    }
    ApplyBlockIdWriteAt(block_ids, cell_ids, order, sums, static_cast<std::uint32_t>(position));
  }
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

RunOnCallStatus WriteSplitCellCountsOnGpu(
    const CellId* cell_ids,
    std::uint32_t active_count,
    const GridBuildStatus* status,
    std::uint32_t* split_cell_counts
) {
  if (active_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(
      WriteSplitCellCounts,
      ComputeLaunchSize(active_count),
      cell_ids,
      active_count,
      status,
      split_cell_counts
  );
}

RunOnCallStatus CheckCellPlacesOnGpu(
    const std::uint32_t* sums,
    std::uint32_t active_count,
    std::uint64_t room_cell_places,
    std::uint32_t sort_count,
    GridBuildStatus* status
) {
  return ApplyGpuFunction(CheckCellPlaces, kOneThread, sums, active_count, room_cell_places, sort_count, status);
}

RunOnCallStatus CheckBlockCountOnGpu(
    const std::uint32_t* sums,
    std::uint32_t sort_count,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
) {
  return ApplyGpuFunction(CheckBlockCount, kOneThread, sums, sort_count, max_cell_blocks, status);
}

RunOnCallStatus WriteSplitCellsOnGpu(
    CellId* cell_ids,
    const std::uint32_t* sums,
    std::uint32_t active_count,
    const GridBuildStatus* status
) {
  if (active_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(WriteSplitCells, ComputeLaunchSize(active_count), cell_ids, sums, active_count, status);
}

RunOnCallStatus WriteSortTailOnGpu(
    CellId* cell_ids,
    std::uint32_t active_count,
    std::uint32_t sort_count,
    const GridBuildStatus* status
) {
  if (sort_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(WriteSortTail, ComputeLaunchSize(sort_count), cell_ids, active_count, sort_count, status);
}

RunOnCallStatus WriteBlockSortKeysOnGpu(
    const CellId* cell_ids,
    std::uint32_t sort_count,
    int pass,
    const GridBuildStatus* status,
    std::uint64_t* sort_keys
) {
  if (sort_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(
      WriteBlockSortKeys,
      ComputeLaunchSize(sort_count),
      cell_ids,
      sort_count,
      pass,
      status,
      sort_keys
  );
}

RunOnCallStatus WriteBlockStartMarksOnGpu(
    const CellId* cell_ids,
    const std::uint32_t* order,
    std::uint32_t sort_count,
    const GridBuildStatus* status,
    std::uint32_t* marks
) {
  if (sort_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(
      WriteBlockStartMarks,
      ComputeLaunchSize(sort_count),
      cell_ids,
      order,
      sort_count,
      status,
      marks
  );
}

RunOnCallStatus WriteBlockIdsOnGpu(
    CellBlockId* block_ids,
    const CellId* cell_ids,
    const std::uint32_t* order,
    const std::uint32_t* sums,
    std::uint32_t sort_count,
    const GridBuildStatus* status
) {
  if (sort_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunction(
      WriteBlockIds,
      ComputeLaunchSize(sort_count),
      block_ids,
      cell_ids,
      order,
      sums,
      sort_count,
      status
  );
}

RunOnCallStatus ComputeLargestStackOnGpu(std::uint64_t* bytes_per_thread) {
  std::uint64_t largest_bytes = 0;
  const std::array<cudaError_t, 14> errors = {
      ApplyLargerStack(ApplyBuildStatusReset, &largest_bytes),
      ApplyLargerStack(CheckCellIds, &largest_bytes),
      ApplyLargerStack(WriteCellSortKeys, &largest_bytes),
      ApplyLargerStack(CopySortedCells, &largest_bytes),
      ApplyLargerStack(CheckSortedPositions, &largest_bytes),
      ApplyLargerStack(WriteFailureCells, &largest_bytes),
      ApplyLargerStack(WriteSplitCellCounts, &largest_bytes),
      ApplyLargerStack(CheckCellPlaces, &largest_bytes),
      ApplyLargerStack(CheckBlockCount, &largest_bytes),
      ApplyLargerStack(WriteSplitCells, &largest_bytes),
      ApplyLargerStack(WriteSortTail, &largest_bytes),
      ApplyLargerStack(WriteBlockSortKeys, &largest_bytes),
      ApplyLargerStack(WriteBlockStartMarks, &largest_bytes),
      ApplyLargerStack(WriteBlockIds, &largest_bytes),
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
