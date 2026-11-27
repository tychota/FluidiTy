// The GPU half of the build of a grid piece, private to grid/: the functions declared in grid/grid_piece_gpu.h.
//
// Thin GPU functions, one for each of the build's six steps, each a grid-stride loop (or one thread) over a per-item
// rule of grid/grid_piece_steps.h, which a CPU test replays for every thread block and thread. The only operations
// between threads are the integer atomic minimum of the failure key and the integer atomic adds of the counts, whose
// results do not depend on the order the threads apply them. The calls use the default stream, so each sees what the
// one before it wrote. Launches, the wait and statuses go through base/gpu_launch.cuh.
// Writes: function 2 writes only the status record, with atomics. Function 3 writes the piece map entry that
// ToPiecePlace gives its item, and no two items have one place. Functions 4 to 6 write the arrays of their own piece
// block, cell place or face place, and the status record with atomics. No function reads what another thread of the
// same launch writes, except the failure key, which functions 3 and 4 read for a key of function 2 only: another
// thread's key of function 4 never changes what a thread does (ComputeHasEarlierStepFailed ignores a key of the
// thread's own step). The piece map that functions 4 to 6 read was written by function 3, an earlier launch of the
// same stream.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece_gpu.h"
#include "fluidity/grid/grid_piece_steps.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {
namespace {

// The step a function tests against with ComputeHasEarlierStepFailed. Functions 3 and 4 stop for a key of function 2
// only: a key of function 4 that another thread offered must not stop a thread, or the keys left out could be smaller.
// Functions 5 and 6 come after every keyed function, so any key stops them.
constexpr int kStopStepOfMapAndBlocks = kPieceFaultStep;
constexpr int kStopStepOfStatesAndKinds = kPieceFaultStep + 1;

// Function 1, one thread.
__global__ void ApplyPieceStatusReset(GridBuildStatus* status) {
  *status = ComputeResetBuildStatus();
}

// Function 2: one item per border entry.
__global__ void __launch_bounds__(
    kThreadBlockSize
) CheckPieceBorder(PieceLoad load, std::uint32_t whole_block_count, GridBuildStatus* status) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t entry = stride.first_item; entry < load.border_block_count; entry += stride.item_step) {
    const auto entry_number = static_cast<std::uint32_t>(entry);
    const std::uint64_t failure_key = ComputePieceBorderFailureKey(load, whole_block_count, entry_number);
    if (failure_key != kNoFailureKey) {
      atomicMin(ToAtomicUint64(&status->failure_key), failure_key);
    }
    if (ComputeIsBorderEntryBelowRange(load, entry_number) != 0) {
      atomicAdd(&status->result_count, 1U);
    }
  }
}

// Function 3: one item per loaded item. After a failure of function 2 no thread writes anything.
__global__ void __launch_bounds__(
    kThreadBlockSize
) WritePieceMap(PieceLoad load, const GridBuildStatus* status, CellBlockIndex* map) {
  if (ComputeHasEarlierStepFailed(ComputeCurrentUint64(&status->failure_key), kStopStepOfMapAndBlocks)) {
    return;
  }
  const GridStride stride = ComputeGridStride();
  const std::size_t loaded_count = std::size_t{load.range.block_count} + load.border_block_count;
  for (std::size_t item = stride.first_item; item < loaded_count; item += stride.item_step) {
    ApplyPieceMapWriteAt(load, static_cast<std::uint32_t>(item), map);
  }
}

// Function 4: one item per piece block.
__global__ void __launch_bounds__(
    kThreadBlockSize
) CopyPieceBlocks(GridTables whole, PieceBlocks blocks, PieceTableArrays piece, GridBuildStatus* status) {
  if (ComputeHasEarlierStepFailed(ComputeCurrentUint64(&status->failure_key), kStopStepOfMapAndBlocks)) {
    return;
  }
  const GridStride stride = ComputeGridStride();
  for (std::size_t piece_block = stride.first_item; piece_block < blocks.count; piece_block += stride.item_step) {
    const std::uint64_t failure_key =
        ApplyPieceBlockCopyAt(whole, blocks, piece, static_cast<std::uint32_t>(piece_block));
    if (failure_key != kNoFailureKey) {
      atomicMin(ToAtomicUint64(&status->failure_key), failure_key);
    }
  }
}

// Function 5: one item per cell place of the piece. An active cell adds 1 to the count of its size level.
__global__ void __launch_bounds__(
    kThreadBlockSize
) CopyPieceCellStates(GridTables whole, PieceBlocks blocks, CellState* cell_states, GridBuildStatus* status) {
  if (ComputeHasEarlierStepFailed(ComputeCurrentUint64(&status->failure_key), kStopStepOfStatesAndKinds)) {
    return;
  }
  const GridStride stride = ComputeGridStride();
  const std::size_t place_count = std::size_t{kCellPlacesPerBlock} * blocks.count;
  for (std::size_t place = stride.first_item; place < place_count; place += stride.item_step) {
    const int size_level = ApplyPieceCellStateCopyAt(whole, blocks, static_cast<std::uint32_t>(place), cell_states);
    if (size_level >= 0) {
      atomicAdd(ToAtomicUint64(&status->active_cells_per_size_level[size_level]), 1ULL);
    }
  }
}

// Function 6: one item per face place of the piece. A thread adds up the face pieces of its places and adds the sum to
// the record once.
__global__ void __launch_bounds__(
    kThreadBlockSize
) WritePieceFaceKinds(GridTables whole, PieceBlocks blocks, FacePieceKind* face_piece_kinds, GridBuildStatus* status) {
  if (ComputeHasEarlierStepFailed(ComputeCurrentUint64(&status->failure_key), kStopStepOfStatesAndKinds)) {
    return;
  }
  const GridStride stride = ComputeGridStride();
  const std::size_t place_count = std::size_t{kFacePlacesPerBlock} * blocks.count;
  std::uint64_t face_piece_sum = 0;
  for (std::size_t place = stride.first_item; place < place_count; place += stride.item_step) {
    face_piece_sum += ApplyPieceFaceKindWriteAt(whole, blocks, static_cast<std::uint32_t>(place), face_piece_kinds);
  }
  if (face_piece_sum != 0) {
    atomicAdd(ToAtomicUint64(&status->face_pieces), face_piece_sum);
  }
}

}  // namespace

RunOnCallStatus ApplyPieceStatusResetOnGpu(GridBuildStatus* status) {
  return ApplyGpuFunctionAndWaitOnFailure(ApplyPieceStatusReset, kOneThread, status);
}

RunOnCallStatus CheckPieceBorderOnGpu(PieceLoad load, std::uint32_t whole_block_count, GridBuildStatus* status) {
  // A launch of 0 thread blocks fails, and there is no entry to check.
  if (load.border_block_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunctionAndWaitOnFailure(
      CheckPieceBorder,
      ComputeLaunchSize(load.border_block_count),
      load,
      whole_block_count,
      status
  );
}

RunOnCallStatus WritePieceMapOnGpu(PieceLoad load, const GridBuildStatus* status, CellBlockIndex* map) {
  const std::size_t loaded_count = std::size_t{load.range.block_count} + load.border_block_count;
  if (loaded_count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunctionAndWaitOnFailure(WritePieceMap, ComputeLaunchSize(loaded_count), load, status, map);
}

RunOnCallStatus
CopyPieceBlocksOnGpu(const GridTables& whole, PieceBlocks blocks, PieceTableArrays piece, GridBuildStatus* status) {
  if (blocks.count == 0) {
    return RunOnCallStatus::kDone;
  }
  return ApplyGpuFunctionAndWaitOnFailure(
      CopyPieceBlocks,
      ComputeLaunchSize(blocks.count),
      whole,
      blocks,
      piece,
      status
  );
}

RunOnCallStatus
CopyPieceCellStatesOnGpu(const GridTables& whole, PieceBlocks blocks, CellState* cell_states, GridBuildStatus* status) {
  if (blocks.count == 0) {
    return RunOnCallStatus::kDone;
  }
  const std::size_t place_count = std::size_t{kCellPlacesPerBlock} * blocks.count;
  return ApplyGpuFunctionAndWaitOnFailure(
      CopyPieceCellStates,
      ComputeLaunchSize(place_count),
      whole,
      blocks,
      cell_states,
      status
  );
}

RunOnCallStatus WritePieceFaceKindsOnGpu(
    const GridTables& whole,
    PieceBlocks blocks,
    FacePieceKind* face_piece_kinds,
    GridBuildStatus* status
) {
  if (blocks.count == 0) {
    return CheckGpuWork();
  }
  const std::size_t place_count = std::size_t{kFacePlacesPerBlock} * blocks.count;
  if (const RunOnCallStatus started = ApplyGpuFunctionAndWaitOnFailure(
          WritePieceFaceKinds,
          ComputeLaunchSize(place_count),
          whole,
          blocks,
          face_piece_kinds,
          status
      );
      started != RunOnCallStatus::kDone) {
    return started;
  }
  // The wait reports an error of any function started since the build's first call.
  return CheckGpuWork();
}

RunOnCallStatus ComputeLargestGridPieceStackOnGpu(std::uint64_t* bytes_per_thread) {
  std::uint64_t largest_bytes = 0;
  const std::array<cudaError_t, 6> errors = {
      ApplyLargerStack(ApplyPieceStatusReset, &largest_bytes),
      ApplyLargerStack(CheckPieceBorder, &largest_bytes),
      ApplyLargerStack(WritePieceMap, &largest_bytes),
      ApplyLargerStack(CopyPieceBlocks, &largest_bytes),
      ApplyLargerStack(CopyPieceCellStates, &largest_bytes),
      ApplyLargerStack(WritePieceFaceKinds, &largest_bytes),
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
