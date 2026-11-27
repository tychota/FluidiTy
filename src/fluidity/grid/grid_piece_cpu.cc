// The builder of a grid piece: the piece map and the tables of the blocks of a range and of its border, on the CPU with
// loops of its own over the rules of grid/grid_piece_steps.h, and on the GPU through the six GPU functions of
// grid/grid_piece_gpu.h.
//
// Implementation uses: CheckPieceRangeAndRoom, ToPieceBuildSummary, PieceBuildSummary (grid/grid_piece_internal.h),
//   ToPiecePlace, ToPieceNeighbour, ComputePieceFaceKind, PieceLoad, PieceBlocks, PieceTableArrays,
//   kPieceBorderStep, kPieceFaultStep (grid/grid_piece_steps.h), ToGridBuffersArrays, BuildGridFromBuffers,
//   ToGridFailure (grid/grid_buffers_internal.h), GridBuildStatus (grid/grid_build_status.h),
//   ToFailureStep (grid/grid_build_steps.h), the GPU functions of grid/grid_piece_gpu.h

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_piece_internal.h"
#include "fluidity/grid/grid_piece_steps.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_piece_gpu.h"
#endif

namespace fluidity::grid {
namespace {

GridFailure ToFailure(GridError error) {
  return {.error = error};
}

// Says whether a block of the whole grid is one of the blocks of the range.
bool ComputeIsInRange(CellBlockRange range, std::uint32_t whole_block) {
  return whole_block >= range.first_block && whole_block - range.first_block < range.block_count;
}

// Checks the border list: ascending, each block once, inside the whole grid and outside the range. Each failure is
// kBadInput, since the caller made the list.
std::expected<void, GridFailure>
CheckBorder(std::uint32_t whole_block_count, CellBlockRange range, std::span<const CellBlockIndex> border) {
  for (std::size_t entry = 0; entry < border.size(); ++entry) {
    const std::uint32_t block = border[entry].value;
    const bool is_ascending = entry == 0 || border[entry - 1].value < block;
    // The list is a FindBorderBlocks result: ascending, outside the range, inside the grid. Anything else is a caller's
    // mistake, and the map would not be ascending.
    if (!is_ascending || block >= whole_block_count || ComputeIsInRange(range, block)) {
      return std::unexpected(ToFailure(GridError::kBadInput));
    }
  }
  return {};
}

// Writes the piece map: each loaded item at the place ToPiecePlace gives it, alone.
void ApplyPieceMap(
    std::span<const CellBlockIndex> border,
    CellBlockRange range,
    std::uint32_t loaded_count,
    CellBlockIndex* map
) {
  const auto border_count = static_cast<std::uint32_t>(border.size());
  for (std::uint32_t loaded_item = 0; loaded_item < loaded_count; ++loaded_item) {
    const std::uint32_t whole_block =
        loaded_item < border_count ? border[loaded_item].value : range.first_block + (loaded_item - border_count);
    map[ToPiecePlace(border.data(), border_count, range, loaded_item).value] = {.value = whole_block};
  }
}

// What the tables of one piece hold, for the counts of the grid.
struct PieceCounts {
  GridCounts counts;
  bool is_fault;  // A neighbour entry named a block of the range that the piece map lacks.
};

// Writes the tables of the piece, block by block, from the whole grid's tables and the piece map, and counts the
// active cells and the face pieces. Steps: 1 the block id  2 the 27 + 1 + 8 neighbour entries  3 the cell states
// 4 the face piece kinds. A neighbour entry that ToPieceNeighbour does not find but that names a block of the range
// is a fault: the map the same call wrote must hold every block of the range.
PieceCounts
WritePieceTables(const GridTables& whole, CellBlockRange range, std::uint32_t loaded_count, GridBuffersArrays& arrays) {
  const CellBlockIndex* const map = arrays.piece_map.data();
  const std::size_t smaller_start = ComputeSmallerBlocksStart(static_cast<std::uint32_t>(arrays.block_ids.size()));
  std::uint32_t* const bigger_blocks = arrays.bigger_and_smaller_blocks.data();
  std::uint32_t* const smaller_blocks = bigger_blocks + smaller_start;
  PieceCounts result = {.counts = {}, .is_fault = false};
  const auto to_piece_entry = [&](std::uint32_t whole_neighbour) {
    const std::uint32_t piece_neighbour = ToPieceNeighbour(map, loaded_count, whole_neighbour);
    if (piece_neighbour == kNoIndex && whole_neighbour != kNoIndex && ComputeIsInRange(range, whole_neighbour)) {
      result.is_fault = true;
    }
    return piece_neighbour;
  };
  for (std::uint32_t piece_block = 0; piece_block < loaded_count; ++piece_block) {
    const std::uint32_t whole_block = map[piece_block].value;
    // Step 1.
    arrays.block_ids.data()[piece_block] = whole.block_ids[whole_block];
    // Step 2.
    for (std::size_t entry = 0; entry < 27; ++entry) {
      arrays.same_size_level_neighbour_blocks.data()[(std::size_t{piece_block} * 27) + entry] =
          to_piece_entry(whole.same_size_level_blocks[(std::size_t{whole_block} * 27) + entry]);
    }
    bigger_blocks[piece_block] = to_piece_entry(whole.bigger_block[whole_block]);
    for (std::size_t entry = 0; entry < 8; ++entry) {
      smaller_blocks[(std::size_t{piece_block} * 8) + entry] =
          to_piece_entry(whole.smaller_blocks[(std::size_t{whole_block} * 8) + entry]);
    }
    // Step 3.
    const auto size_level = static_cast<std::size_t>(ToSizeLevel(whole, {whole_block}));
    for (std::size_t place = 0; place < kCellPlacesPerBlock; ++place) {
      const CellState state = whole.cell_states[(std::size_t{whole_block} * kCellPlacesPerBlock) + place];
      arrays.cell_states.data()[(std::size_t{piece_block} * kCellPlacesPerBlock) + place] = state;
      if (state == CellState::kActive) {
        ++result.counts.active_cells;
        ++result.counts.active_cells_per_size_level[size_level];
      }
    }
    // Step 4.
    for (int place = 0; place < kFacePlacesPerBlock; ++place) {
      const FacePieceKind kind = ComputePieceFaceKind(whole, map, loaded_count, {.value = piece_block}, place);
      arrays.face_piece_kinds
          .data()[(std::size_t{piece_block} * kFacePlacesPerBlock) + static_cast<std::size_t>(place)] = kind;
      if (kind != FacePieceKind::kNotAPiece) {
        ++result.counts.face_pieces;
      }
    }
  }
  result.counts.cell_blocks = loaded_count;
  return result;
}

}  // namespace

std::expected<void, GridFailure> CheckPieceRangeAndRoom(
    std::uint32_t whole_block_count,
    CellBlockRange range,
    std::uint64_t border_block_count,
    std::uint32_t max_cell_blocks
) {
  // An empty range would load only its border, which no stage can use, unless the whole grid has no block. A range past
  // the last block names blocks the whole grid does not have.
  const bool is_range_empty = range.block_count == 0 && whole_block_count != 0;
  if (is_range_empty || std::uint64_t{range.first_block} + range.block_count > whole_block_count) {
    return std::unexpected(ToFailure(GridError::kBadInput));
  }
  // The buffers cannot hold the range and its border.
  if (std::uint64_t{range.block_count} + border_block_count > max_cell_blocks) {
    return std::unexpected(ToFailure(GridError::kTooManyCellBlocks));
  }
  return {};
}

std::expected<PieceBuildSummary, GridFailure>
ToPieceBuildSummary(const GridBuildStatus& status, std::uint32_t loaded_count, std::uint32_t border_block_count) {
  if (status.failure_key != kNoFailureKey) {
    // Step 1 is a bad border list, which the caller made. A key of another step contradicts the steps that wrote it.
    const bool is_bad_border = ToFailureStep(status.failure_key) == kPieceBorderStep;
    return std::unexpected(ToFailure(is_bad_border ? GridError::kBadInput : GridError::kInternalFault));
  }
  PieceBuildSummary summary = {.counts = {}, .range_start_in_piece = status.result_count};
  summary.counts.cell_blocks = loaded_count;
  for (std::size_t size_level = 0; size_level < kMaxSizeLevels; ++size_level) {
    summary.counts.active_cells_per_size_level[size_level] = status.active_cells_per_size_level[size_level];
    summary.counts.active_cells += status.active_cells_per_size_level[size_level];
  }
  summary.counts.face_pieces = status.face_pieces;
  // Without a failure each count is within what the piece's places hold, and the border entries below the range are
  // some of the border: a record that says otherwise contradicts the steps that wrote it.
  const bool is_within_places = summary.counts.active_cells <= std::uint64_t{loaded_count} * kCellPlacesPerBlock
      && summary.counts.face_pieces <= std::uint64_t{loaded_count} * kFacePlacesPerBlock
      && summary.range_start_in_piece <= border_block_count;
  if (!is_within_places) {
    return std::unexpected(ToFailure(GridError::kInternalFault));
  }
  return summary;
}

#if defined(FLUIDITY_HAS_GPU)
namespace {

// Every face place of a piece fits 32 bits and stays below kNoIndex, since a room holds at most kMaxCellBlocksLimit
// blocks; the cell places, 64 per block, are fewer. The GPU threads number places with 32 bits.
static_assert(
    std::uint64_t{kMaxCellBlocksLimit} * kFacePlacesPerBlock < std::uint64_t{kNoIndex},
    "a face place of a piece must fit 32 bits and stay below kNoIndex"
);

// The failure of a GPU-half call that did not start: kRunOnCallFailed, through the mapping of base/.
GridFailure ToCallFailure(RunOnCallStatus call_status) {
  return ToGridFailure(ToBufferError(call_status), 0);
}

// Builds the piece map and the tables of a piece on the GPU into `buffers`, and returns its counts and where its range
// starts. Steps: 1 the checks of the range and the room  2 reset the record (GPU function 1 of grid_piece_gpu.h)
// 3 check the border entries (function 2)  4 write the piece map (3)  5 copy the blocks (4)  6 copy the cell states (5)
// 7 write the face piece kinds and wait (6)  8 read the record. Steps 2 to 7 are one launch each, in the one stream,
// so each sees what the step before wrote. Only the record comes back.
std::expected<PieceBuildSummary, GridFailure> BuildPieceTablesOnGpu(
    const StaggeredGrid& whole,
    CellBlockRange range,
    const Buffer<CellBlockIndex>& border_blocks,
    std::uint32_t border_block_count,
    GridBuffers& buffers
) {
  const GridTables whole_tables = whole.tables(RunOn::kGpu);
  // Step 1: as on the CPU, before anything runs on the GPU. The count of loaded blocks is at most the room, which is at
  // most kMaxCellBlocksLimit, so it fits 32 bits.
  if (const std::expected<void, GridFailure> checked =
          CheckPieceRangeAndRoom(whole_tables.block_count, range, border_block_count, buffers.max_cell_blocks());
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
  GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  // The estimate gives every GPU grid's buffers one status record: a buffer that has none is a fault of the builders.
  if (arrays.build_status.size() != 1) {
    return std::unexpected(ToFailure(GridError::kInternalFault));
  }
  const std::uint32_t loaded_count = range.block_count + border_block_count;
  GridBuildStatus* const status = arrays.build_status.data();
  const PieceLoad load = {
      .range = range,
      .border_blocks = border_blocks.data(),
      .border_block_count = border_block_count
  };
  CellBlockIndex* const map = arrays.piece_map.data();
  const PieceBlocks blocks = {.range = range, .map = map, .count = loaded_count};
  std::uint32_t* const bigger_block = arrays.bigger_and_smaller_blocks.data();
  const PieceTableArrays piece_arrays = {
      .block_ids = arrays.block_ids.data(),
      .same_size_level_blocks = arrays.same_size_level_neighbour_blocks.data(),
      .bigger_block = bigger_block,
      .smaller_blocks = bigger_block + ComputeSmallerBlocksStart(static_cast<std::uint32_t>(arrays.block_ids.size())),
      .cell_states = arrays.cell_states.data(),
      .face_piece_kinds = arrays.face_piece_kinds.data(),
  };
  // Steps 2 to 7, each started only once the one before started.
  if (const RunOnCallStatus started = ApplyPieceStatusResetOnGpu(status); started != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(started));
  }
  if (const RunOnCallStatus started = CheckPieceBorderOnGpu(load, whole_tables.block_count, status);
      started != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(started));
  }
  if (const RunOnCallStatus started = WritePieceMapOnGpu(load, status, map); started != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(started));
  }
  if (const RunOnCallStatus started = CopyPieceBlocksOnGpu(whole_tables, blocks, piece_arrays, status);
      started != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(started));
  }
  if (const RunOnCallStatus started = CopyPieceCellStatesOnGpu(whole_tables, blocks, piece_arrays.cell_states, status);
      started != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(started));
  }
  // The wait at the end of the last function reports an error of any function started since the first.
  if (const RunOnCallStatus finished =
          WritePieceFaceKindsOnGpu(whole_tables, blocks, piece_arrays.face_piece_kinds, status);
      finished != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(finished));
  }
  // Step 8: the one transfer of the build.
  GridBuildStatus read_status{};
  if (const std::expected<void, BufferError> read =
          ReadBuffer(arrays.build_status, 0, std::span<GridBuildStatus>(&read_status, 1));
      !read.has_value()) {
    return std::unexpected(ToGridFailure(read.error(), sizeof(GridBuildStatus)));
  }
  return ToPieceBuildSummary(read_status, loaded_count, border_block_count);
}

}  // namespace
#endif

std::span<const CellBlockIndex> GridPiece::block_in_whole_grid() const {
  if (grid_.run_on() != RunOn::kCpu) {
    return {};
  }
  return {ToGridBuffersArrays(grid_).piece_map.data(), grid_.counts().cell_blocks};
}

const Buffer<CellBlockIndex>& GridPiece::block_in_whole_grid_buffer() const {
  return ToGridBuffersArrays(grid_).piece_map;
}

std::expected<GridPiece, GridFailure> BuildGridPiece(
    const StaggeredGrid& whole,
    CellBlockRange range,
    std::span<const CellBlockIndex> border_blocks,
    GridBuffers& buffers
) {
  // Steps: 1 check the range, the room and the border list  2 write the piece map  3 write the tables  4 make the grid
  assert(whole.run_on() == RunOn::kCpu);
  assert(buffers.run_on() == RunOn::kCpu);
  const GridTables whole_tables = whole.tables(RunOn::kCpu);
  // Step 1. The range is checked first, the room next, then the border entries, so a list far above the room is
  // not walked.
  if (const std::expected<void, GridFailure> checked =
          CheckPieceRangeAndRoom(whole_tables.block_count, range, border_blocks.size(), buffers.max_cell_blocks());
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
  if (const std::expected<void, GridFailure> checked = CheckBorder(whole_tables.block_count, range, border_blocks);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
  const std::uint64_t loaded_count = std::uint64_t{range.block_count} + border_blocks.size();
  GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  const auto loaded_block_count = static_cast<std::uint32_t>(loaded_count);
  // Step 2.
  ApplyPieceMap(border_blocks, range, loaded_block_count, arrays.piece_map.data());
  // Step 3.
  const PieceCounts written = WritePieceTables(whole_tables, range, loaded_block_count, arrays);
  if (written.is_fault) {
    return std::unexpected(ToFailure(GridError::kInternalFault));
  }
  // Step 4. The border entries below the range come before it in the piece.
  const auto border_below_range = static_cast<std::uint32_t>(
      std::ranges::lower_bound(border_blocks, range.first_block, {}, &CellBlockIndex::value) - border_blocks.begin()
  );
  StaggeredGrid grid = BuildGridFromBuffers(std::move(buffers), whole_tables.frame, whole.version(), written.counts);
  return GridPiece(std::move(grid), {.first_block = border_below_range, .block_count = range.block_count}, range);
}

std::expected<GridPiece, GridFailure> BuildGridPiece(
    const StaggeredGrid& whole,
    CellBlockRange range,
    const Buffer<CellBlockIndex>& border_blocks,
    std::uint32_t border_block_count,
    GridBuffers& buffers
) {
  assert(border_blocks.run_on() == whole.run_on());
  assert(buffers.run_on() == whole.run_on());
  // The count names entries the buffer does not have.
  if (border_block_count > border_blocks.size()) {
    return std::unexpected(ToFailure(GridError::kBadInput));
  }
  // A GPU grid's piece is built on the GPU, and nothing falls back to the CPU.
  if (whole.run_on() != RunOn::kCpu) {
#if defined(FLUIDITY_HAS_GPU)
    const std::expected<PieceBuildSummary, GridFailure> built =
        BuildPieceTablesOnGpu(whole, range, border_blocks, border_block_count, buffers);
    if (!built.has_value()) {
      return std::unexpected(built.error());
    }
    StaggeredGrid grid =
        BuildGridFromBuffers(std::move(buffers), whole.tables(RunOn::kGpu).frame, whole.version(), built->counts);
    return GridPiece(
        std::move(grid),
        {.first_block = built->range_start_in_piece, .block_count = range.block_count},
        range
    );
#else
    // Without GPU code no GPU grid can be built, so no caller reaches this.
    return std::unexpected(ToFailure(GridError::kRunOnCallFailed));
#endif
  }
  return BuildGridPiece(
      whole,
      range,
      std::span<const CellBlockIndex>(border_blocks.data(), border_block_count),
      buffers
  );
}

GridBuffers TakeBuffers(GridPiece&& piece) {
  return TakeBuffers(std::move(piece.grid_));
}

}  // namespace fluidity::grid
