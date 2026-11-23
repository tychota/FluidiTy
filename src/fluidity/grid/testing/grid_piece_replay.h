// Test-only: the GPU build of a grid piece, replayed on the CPU.
//
// Purpose: run the five launches of the GPU build of a grid piece (grid/grid_piece_gpu.h) on CPU arrays, thread by
//   thread, so a test compares their result with the CPU builder without a GPU.
//
// Idea: each GPU function is a thin loop over a shared per-item function of grid/grid_piece_steps.h. The replay runs
//   those functions for every thread block and thread of the launch, in launch order or in reverse, with the atomic
//   minimum of the failure key and the atomic adds of the counts as plain operations. The piece's arrays start as 0xFF
//   bytes and the status record too, so a step that left out a write shows, and a failed border must leave every array
//   as it found it. The replay takes the border as a list in CPU memory; the checks of the range and of the room,
//   which the GPU build makes on the CPU before its first launch, are the CPU builder's own and are not replayed.
//
// Words:
//   replay  the GPU functions run one thread after the other on the CPU
// Owns:    PieceReplayResult, ComputeGridPieceReplay
// Uses:    CPU-only guard (base/cpu_only.h), ReplayLaunch, ApplyEveryThreadOfLaunch (grid/testing/thread_replay.h),
//          PieceLoad, PieceBlocks, PieceTableArrays, ComputePieceBorderFailureKey, ComputeIsBorderEntryBelowRange,
//          ApplyPieceMapWriteAt, ApplyPieceBlockCopyAt, ApplyPieceCellStateCopyAt, ApplyPieceFaceKindWriteAt
//          (grid/grid_piece_steps.h), GridBuildStatus, kNoFailureKey (grid/grid_build_status.h),
//          ComputeResetBuildStatus (grid/grid_build_steps.h), PieceTables (grid/testing/grid_piece_support.h),
//          GridTables
// Used by: tests/grid/grid_piece_steps_in_every_thread_order.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece_steps.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/thread_replay.h"

namespace fluidity::grid {

// What one replayed build leaves behind.
struct PieceReplayResult {
  PieceTables tables;  // The arrays of the piece, for the loaded blocks; 0xFF bytes where nothing was written.
  GridBuildStatus status{};  // The record after the five launches.
  std::uint32_t loaded_count = 0;  // The blocks of the piece.
};

// Runs launches 2 to 6 of the GPU build over `whole` for `range` and `border_blocks`, each with `launch`: 2 the check
// of each border entry, 3 the piece map, 4 each piece block's id and neighbour entries, 5 each cell place's state, 6
// each face place's kind. After a failure of 2 or 4 the launches after it do nothing, as on the GPU.
// Requires: whole is the tables of a CPU grid; range is inside it; border_blocks holds at most as many entries as the
//   room the caller has, so that loaded_count is the range's blocks plus the border's.
inline PieceReplayResult ComputeGridPieceReplay(
    const GridTables& whole,
    CellBlockRange range,
    std::span<const CellBlockIndex> border_blocks,
    ReplayLaunch launch
) {
  constexpr std::uint8_t kStaleByte = 0xFF;
  const auto border_count = static_cast<std::uint32_t>(border_blocks.size());
  PieceReplayResult result;
  result.loaded_count = range.block_count + border_count;
  const std::size_t block_count = result.loaded_count;
  PieceTables& tables = result.tables;
  tables.frame = whole.frame;
  tables.version = whole.version;
  tables.block_ids.resize(block_count);
  tables.same_size_level_blocks.resize(block_count * 27);
  tables.bigger_block.resize(block_count);
  tables.smaller_blocks.resize(block_count * 8);
  tables.cell_states.resize(block_count * kCellPlacesPerBlock);
  tables.face_piece_kinds.resize(block_count * kFacePlacesPerBlock);
  tables.block_in_whole.resize(block_count);
  const auto make_stale = [](auto& array) {
    std::memset(array.data(), 0xFF, array.size() * sizeof(array[0]));
  };
  make_stale(tables.block_ids);
  make_stale(tables.same_size_level_blocks);
  make_stale(tables.bigger_block);
  make_stale(tables.smaller_blocks);
  make_stale(tables.cell_states);
  make_stale(tables.face_piece_kinds);
  make_stale(tables.block_in_whole);
  GridBuildStatus& status = result.status;
  std::memset(&status, kStaleByte, sizeof(status));

  // Step 1: one thread resets the record.
  status = ComputeResetBuildStatus();
  const PieceLoad load = {.range = range, .border_blocks = border_blocks.data(), .border_block_count = border_count};
  // Step 2: one thread for each border entry offers its key and counts itself when it lies below the range.
  ApplyEveryThreadOfLaunch(border_count, launch, [&](std::size_t entry) {
    const auto entry_number = static_cast<std::uint32_t>(entry);
    status.failure_key =
        std::min(status.failure_key, ComputePieceBorderFailureKey(load, whole.block_count, entry_number));
    status.result_count += ComputeIsBorderEntryBelowRange(load, entry_number);
    return true;
  });
  // Step 3: one thread for each loaded item; after a failure of step 2 a thread stops at once.
  ApplyEveryThreadOfLaunch(block_count, launch, [&](std::size_t item) {
    if (ComputeHasEarlierStepFailed(status.failure_key, kPieceFaultStep)) {
      return false;
    }
    ApplyPieceMapWriteAt(load, static_cast<std::uint32_t>(item), tables.block_in_whole.data());
    return true;
  });
  const PieceBlocks blocks = {.range = range, .map = tables.block_in_whole.data(), .count = result.loaded_count};
  const PieceTableArrays arrays = {
      .block_ids = tables.block_ids.data(),
      .same_size_level_blocks = tables.same_size_level_blocks.data(),
      .bigger_block = tables.bigger_block.data(),
      .smaller_blocks = tables.smaller_blocks.data(),
      .cell_states = tables.cell_states.data(),
      .face_piece_kinds = tables.face_piece_kinds.data(),
  };
  // Step 4: one thread for each piece block copies its id and entries, and offers the key of a fault.
  ApplyEveryThreadOfLaunch(block_count, launch, [&](std::size_t piece_block) {
    if (ComputeHasEarlierStepFailed(status.failure_key, kPieceFaultStep)) {
      return false;
    }
    const std::uint64_t failure_key =
        ApplyPieceBlockCopyAt(whole, blocks, arrays, static_cast<std::uint32_t>(piece_block));
    status.failure_key = std::min(status.failure_key, failure_key);
    return true;
  });
  // Step 5: one thread for each cell place copies its state; an active cell adds 1 to the count of its size level.
  ApplyEveryThreadOfLaunch(block_count * kCellPlacesPerBlock, launch, [&](std::size_t place) {
    if (ComputeHasEarlierStepFailed(status.failure_key, kPieceFaultStep + 1)) {
      return false;
    }
    const int size_level =
        ApplyPieceCellStateCopyAt(whole, blocks, static_cast<std::uint32_t>(place), arrays.cell_states);
    if (size_level >= 0) {
      ++status.active_cells_per_size_level[size_level];
    }
    return true;
  });
  // Step 6: one thread for each face place writes its kind; the face pieces are added to the record.
  ApplyEveryThreadOfLaunch(block_count * kFacePlacesPerBlock, launch, [&](std::size_t place) {
    if (ComputeHasEarlierStepFailed(status.failure_key, kPieceFaultStep + 1)) {
      return false;
    }
    status.face_pieces +=
        ApplyPieceFaceKindWriteAt(whole, blocks, static_cast<std::uint32_t>(place), arrays.face_piece_kinds);
    return true;
  });
  return result;
}

}  // namespace fluidity::grid
