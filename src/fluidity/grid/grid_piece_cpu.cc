// The CPU builder of a grid piece: the piece map and the tables of the blocks of a range and of its border.
//
// Implementation uses: ToPiecePlace, ToPieceNeighbour, ComputePieceFaceKind (grid/grid_piece_steps.h),
//   ToGridBuffersArrays, BuildGridFromBuffers (grid/grid_buffers_internal.h)

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
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_piece_steps.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

GridFailure ToFailure(GridError error) {
  return {.error = error};
}

// Says whether a block of the whole grid is one of the blocks of the range.
bool ComputeIsInRange(CellBlockRange range, std::uint32_t whole_block) {
  return whole_block >= range.first_block && whole_block - range.first_block < range.block_count;
}

// Checks the range against the whole grid's blocks, and the border list: ascending, each block once, inside the whole
// grid and outside the range. Each failure is kBadInput, since the caller made the lists.
std::expected<void, GridFailure>
CheckRangeAndBorder(std::uint32_t whole_block_count, CellBlockRange range, std::span<const CellBlockIndex> border) {
  // A range past the last block names blocks the whole grid does not have.
  if (std::uint64_t{range.first_block} + range.block_count > whole_block_count) {
    return std::unexpected(ToFailure(GridError::kBadInput));
  }
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
  // Step 1. A range past the whole grid is checked first, the room next, then the border entries, so a list far above
  // the room is not walked.
  if (std::uint64_t{range.first_block} + range.block_count > whole_tables.block_count) {
    return std::unexpected(ToFailure(GridError::kBadInput));
  }
  const std::uint64_t loaded_count = std::uint64_t{range.block_count} + border_blocks.size();
  // The caller raises the room: the buffers cannot hold the range and its border. No cell is named.
  if (loaded_count > buffers.max_cell_blocks()) {
    return std::unexpected(ToFailure(GridError::kTooManyCellBlocks));
  }
  if (const std::expected<void, GridFailure> checked =
          CheckRangeAndBorder(whole_tables.block_count, range, border_blocks);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }
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
  // The GPU form of a piece is not built here, and nothing falls back to the CPU.
  if (whole.run_on() != RunOn::kCpu) {
    return std::unexpected(ToFailure(GridError::kRunOnCallFailed));
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
