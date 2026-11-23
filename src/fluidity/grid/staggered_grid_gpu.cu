// The GPU half of the check of a cell list and of a grid build, private to grid/: the functions declared in
// grid/staggered_grid_gpu.h.
//
// Thin GPU functions, one per step of method paper <proc-build-grid-in-parallel> P1 to P12, or per part of a step, each
// a grid-stride loop (or one thread) over a per-item rule of grid/grid_build_steps.h, which the CPU check and build
// call too and a CPU test replays for every thread block and thread; the copy in sorted order is one assignment per
// item. The only operations between threads are the integer atomic minimum of the failure key and the integer atomic
// adds of the counts: their final values do not depend on the order the threads apply them. Every call uses the default
// stream, so each function sees what the one before it, and the stable orders and prefix sums between them, wrote.
// Launches, the wait and statuses go through base/gpu_launch.cuh.
// Stack: no function keeps a local array. The largest locals are the CellCheck of a sorted position (32 bytes), the
// copy of GridTables of P10 (c) and P11, and the coordinates and loop counters of CheckSiblingsAt, CheckSizeStepAt and
// ComputeFacePieceKind, all in registers or a few dozen bytes of stack; ComputeLargestBuildStackOnGpu reads what the
// compiler gave.
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

// A 64-bit count of `status` as the CUDA atomics take it.
__device__ unsigned long long* ToAtomicCount(std::uint64_t* count) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the same 64 bits, as the atomics' type.
  return reinterpret_cast<unsigned long long*>(count);
}

// The blocks of the build that the tables of the room hold: the block count of the status record, at most the room.
// The count passes the room only after a failure of step 6, when every step that reads it does nothing.
__device__ std::uint32_t ComputeUsedBlockCount(const GridBuildStatus* status, std::uint32_t max_cell_blocks) {
  const std::uint32_t cell_blocks = status->cell_blocks;
  return cell_blocks < max_cell_blocks ? cell_blocks : max_cell_blocks;
}

// Step P9: one item per block of the room; the items below the block count are the blocks. A block offers the key of
// step 7 when it has no bigger block although it is of size level 1 or more.
__global__ void __launch_bounds__(kThreadBlockSize) WriteBlockNeighbours(
    const CellBlockId* block_ids,
    std::uint32_t* same_size_level_blocks,
    std::uint32_t* bigger_block,
    std::uint32_t* smaller_blocks,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  const std::uint32_t block_count = ComputeUsedBlockCount(status, max_cell_blocks);
  for (std::size_t block = stride.first_item; block < block_count; block += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kBlockFaultStep)) {
      return;
    }
    const std::uint64_t failure_key = ApplyBlockNeighboursWriteAt(
        block_ids,
        block_count,
        CellBlockIndex{static_cast<std::uint32_t>(block)},
        same_size_level_blocks,
        bigger_block,
        smaller_blocks
    );
    if (failure_key != kNoFailureKey) {
      atomicMin(ToAtomicFailureKey(status), failure_key);
    }
  }
}

// Step P10 (a): one item per cell place of the room; the items below 64 times the block count are the places.
__global__ void __launch_bounds__(
    kThreadBlockSize
) WriteAbsentStates(CellState* cell_states, std::uint32_t max_cell_blocks, const GridBuildStatus* status) {
  const GridStride stride = ComputeGridStride();
  const std::size_t place_count = std::size_t{kCellPlacesPerBlock} * ComputeUsedBlockCount(status, max_cell_blocks);
  for (std::size_t place = stride.first_item; place < place_count; place += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kTableEntryStep)) {
      return;
    }
    ApplyAbsentStateAt(cell_states, CellPlaceIndex{static_cast<std::uint32_t>(place)});
  }
}

// Step P10 (b): one item per sorted entry of the sort by block. An active entry adds 1 to the count of its size level.
__global__ void __launch_bounds__(kThreadBlockSize) WriteSortedCellStates(
    CellState* cell_states,
    const CellId* cell_ids,
    const std::uint32_t* order,
    const std::uint32_t* sums,
    std::uint32_t active_count,
    std::uint32_t sort_count,
    GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t position = stride.first_item; position < sort_count; position += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kTableEntryStep)) {
      return;
    }
    const int size_level =
        ApplySortedCellStateAt(cell_states, cell_ids, order, sums, active_count, static_cast<std::uint32_t>(position));
    if (size_level >= 0) {
      atomicAdd(ToAtomicCount(&status->active_cells_per_size_level[size_level]), 1ULL);
    }
  }
}

// Step P10 (c), the first launch: one item per cell place of the room; each writes its answer to cover_answers. The
// tables' block count is the status record's.
__global__ void __launch_bounds__(kThreadBlockSize) WriteCoverAnswers(
    GridTables tables,
    std::uint64_t* cover_answers,
    std::uint32_t max_cell_blocks,
    const GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  tables.block_count = ComputeUsedBlockCount(status, max_cell_blocks);
  const std::size_t place_count = std::size_t{kCellPlacesPerBlock} * tables.block_count;
  for (std::size_t place = stride.first_item; place < place_count; place += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kTableEntryStep)) {
      return;
    }
    ApplyCoverAnswerAt(tables, cover_answers, CellPlaceIndex{static_cast<std::uint32_t>(place)});
  }
}

// Step P10 (c), the second launch: one item per cell place of the room; each copies its own answer into its state.
__global__ void __launch_bounds__(kThreadBlockSize) CopyCoverAnswers(
    const std::uint64_t* cover_answers,
    CellState* cell_states,
    std::uint32_t max_cell_blocks,
    const GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  const std::size_t place_count = std::size_t{kCellPlacesPerBlock} * ComputeUsedBlockCount(status, max_cell_blocks);
  for (std::size_t place = stride.first_item; place < place_count; place += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kTableEntryStep)) {
      return;
    }
    ApplyCoverAnswerCopyAt(cover_answers, cell_states, CellPlaceIndex{static_cast<std::uint32_t>(place)});
  }
}

// Step P11: one item per face place of the room. A thread adds up the face pieces of its places and adds the sum to
// the record once.
__global__ void __launch_bounds__(kThreadBlockSize) WriteFacePieceKinds(
    GridTables tables,
    FacePieceKind* face_piece_kinds,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
) {
  const GridStride stride = ComputeGridStride();
  tables.block_count = ComputeUsedBlockCount(status, max_cell_blocks);
  const std::size_t face_place_count = std::size_t{kFacePlacesPerBlock} * tables.block_count;
  std::uint64_t face_piece_sum = 0;
  for (std::size_t face = stride.first_item; face < face_place_count; face += stride.item_step) {
    if (ComputeHasEarlierStepFailed(ComputeCurrentFailureKey(status), kTableEntryStep)) {
      return;
    }
    face_piece_sum +=
        ApplyFacePieceKindWriteAt(tables, face_piece_kinds, FacePlaceIndex{static_cast<std::uint32_t>(face)});
  }
  if (face_piece_sum != 0) {
    atomicAdd(ToAtomicCount(&status->face_pieces), face_piece_sum);
  }
}

// Step P12, one thread.
__global__ void WriteBuildStatusResult(
    GridFrame frame,
    const CellId* input_cells,
    std::uint32_t input_count,
    const CellId* sorted_cells,
    std::uint32_t count,
    GridBuildStatus* status
) {
  ApplyBuildStatusResult(frame, input_cells, input_count, sorted_cells, count, status);
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

RunOnCallStatus WriteBlockNeighboursOnGpu(
    const CellBlockId* block_ids,
    std::uint32_t* same_size_level_blocks,
    std::uint32_t* bigger_block,
    std::uint32_t* smaller_blocks,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
) {
  return ApplyGpuFunction(
      WriteBlockNeighbours,
      ComputeLaunchSize(max_cell_blocks),
      block_ids,
      same_size_level_blocks,
      bigger_block,
      smaller_blocks,
      max_cell_blocks,
      status
  );
}

RunOnCallStatus WriteCellStatesOnGpu(
    GridTables tables,
    CellState* cell_states,
    const SortedBlockEntries& sorted_entries,
    std::uint64_t* cover_answers,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
) {
  const std::size_t room_cell_places = std::size_t{kCellPlacesPerBlock} * max_cell_blocks;
  if (const RunOnCallStatus started = ApplyGpuFunction(
          WriteAbsentStates,
          ComputeLaunchSize(room_cell_places),
          cell_states,
          max_cell_blocks,
          status
      );
      started != RunOnCallStatus::kDone) {
    return started;
  }
  // A launch of 0 thread blocks fails, and with no sorted entry there is no cell to place.
  if (sorted_entries.sort_count != 0) {
    if (const RunOnCallStatus started = ApplyGpuFunction(
            WriteSortedCellStates,
            ComputeLaunchSize(sorted_entries.sort_count),
            cell_states,
            sorted_entries.cell_ids,
            sorted_entries.order,
            sorted_entries.sums,
            sorted_entries.active_count,
            sorted_entries.sort_count,
            status
        );
        started != RunOnCallStatus::kDone) {
      return started;
    }
  }
  if (const RunOnCallStatus started = ApplyGpuFunction(
          WriteCoverAnswers,
          ComputeLaunchSize(room_cell_places),
          tables,
          cover_answers,
          max_cell_blocks,
          status
      );
      started != RunOnCallStatus::kDone) {
    return started;
  }
  return ApplyGpuFunction(
      CopyCoverAnswers,
      ComputeLaunchSize(room_cell_places),
      cover_answers,
      cell_states,
      max_cell_blocks,
      status
  );
}

RunOnCallStatus WriteFacePieceKindsOnGpu(
    GridTables tables,
    FacePieceKind* face_piece_kinds,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
) {
  const std::size_t room_face_places = std::size_t{kFacePlacesPerBlock} * max_cell_blocks;
  return ApplyGpuFunction(
      WriteFacePieceKinds,
      ComputeLaunchSize(room_face_places),
      tables,
      face_piece_kinds,
      max_cell_blocks,
      status
  );
}

RunOnCallStatus WriteBuildStatusResultOnGpu(
    GridFrame frame,
    const CellId* input_cells,
    std::uint32_t input_count,
    const CellId* sorted_cells,
    std::uint32_t count,
    GridBuildStatus* status
) {
  if (const RunOnCallStatus started = ApplyGpuFunction(
          WriteBuildStatusResult,
          kOneThread,
          frame,
          input_cells,
          input_count,
          sorted_cells,
          count,
          status
      );
      started != RunOnCallStatus::kDone) {
    return started;
  }
  // The wait reports an error of a GPU function started since the last wait, the one of the last stable order or
  // prefix sum of P8.
  return CheckGpuWork();
}

RunOnCallStatus ComputeLargestBuildStackOnGpu(std::uint64_t* bytes_per_thread) {
  std::uint64_t largest_bytes = 0;
  const std::array<cudaError_t, 21> errors = {
      ApplyLargerStack(ApplyBuildStatusReset, &largest_bytes),  ApplyLargerStack(CheckCellIds, &largest_bytes),
      ApplyLargerStack(WriteCellSortKeys, &largest_bytes),      ApplyLargerStack(CopySortedCells, &largest_bytes),
      ApplyLargerStack(CheckSortedPositions, &largest_bytes),   ApplyLargerStack(WriteFailureCells, &largest_bytes),
      ApplyLargerStack(WriteSplitCellCounts, &largest_bytes),   ApplyLargerStack(CheckCellPlaces, &largest_bytes),
      ApplyLargerStack(CheckBlockCount, &largest_bytes),        ApplyLargerStack(WriteSplitCells, &largest_bytes),
      ApplyLargerStack(WriteSortTail, &largest_bytes),          ApplyLargerStack(WriteBlockSortKeys, &largest_bytes),
      ApplyLargerStack(WriteBlockStartMarks, &largest_bytes),   ApplyLargerStack(WriteBlockIds, &largest_bytes),
      ApplyLargerStack(WriteBlockNeighbours, &largest_bytes),   ApplyLargerStack(WriteAbsentStates, &largest_bytes),
      ApplyLargerStack(WriteSortedCellStates, &largest_bytes),  ApplyLargerStack(WriteCoverAnswers, &largest_bytes),
      ApplyLargerStack(CopyCoverAnswers, &largest_bytes),       ApplyLargerStack(WriteFacePieceKinds, &largest_bytes),
      ApplyLargerStack(WriteBuildStatusResult, &largest_bytes),
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
