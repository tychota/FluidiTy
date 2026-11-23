#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

// Checks the sorted list of a call: ascending with no repeat, and every index below the grid's block count. A failed
// list is bad input because the walk's binary search needs the order and no block beyond the grid can be asked.
std::expected<void, GridFailure> CheckSortedBlocks(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks) {
  bool is_ascending = true;
  for (std::size_t entry = 1; entry < blocks.size(); ++entry) {
    is_ascending = is_ascending && blocks[entry - 1].value < blocks[entry].value;
  }
  const bool is_below_count = blocks.empty() || blocks.back().value < grid.counts().cell_blocks;
  if (!is_ascending || !is_below_count) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  return {};
}

// Says whether two spans share an entry of memory. std::less gives a total order on pointers, also for pointers into
// different arrays, where the built-in < does not.
bool ComputeDoSpansOverlap(std::span<const CellBlockIndex> first, std::span<const CellBlockIndex> second) {
  if (first.empty() || second.empty()) {
    return false;
  }
  const std::less<> comes_before;
  return comes_before(first.data(), second.data() + second.size())
      && comes_before(second.data(), first.data() + first.size());
}

// Walks the blocks of the grid in ascending index and calls `visit` with each border block of `blocks` at `width`,
// asked from each block's side (ComputeIsBorderBlock). Allocates nothing.
void ApplyToBorderBlocks(
    const StaggeredGrid& grid,
    std::span<const CellBlockIndex> blocks,
    BorderWidth width,
    const auto& visit
) {
  assert(grid.run_on() == RunOn::kCpu);
  const GridTables tables = grid.tables(RunOn::kCpu);
  const BlockSet set = {
      .range = {.first_block = 0, .block_count = 0},
      .sorted_blocks = blocks.data(),
      .sorted_count = static_cast<std::uint32_t>(blocks.size()),
  };
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    if (ComputeIsBorderBlock(tables, set, CellBlockIndex{block}, width)) {
      visit(CellBlockIndex{block});
    }
  }
}

// Runs the readers of `width` forward from every cell place and face place kept in `block`.
void ApplyMarksFromBlock(const GridTables& tables, std::uint32_t block, BorderWidth width, std::uint8_t* marks) {
  for (std::uint32_t place_in_block = 0; place_in_block < static_cast<std::uint32_t>(kCellPlacesPerBlock);
       ++place_in_block) {
    const CellPlaceIndex cell = {.value = (block * static_cast<std::uint32_t>(kCellPlacesPerBlock)) + place_in_block};
    ApplyCellPlaceReadMarks(tables, cell, width, marks);
  }
  for (std::uint32_t place_in_block = 0; place_in_block < static_cast<std::uint32_t>(kFacePlacesPerBlock);
       ++place_in_block) {
    const FacePlaceIndex face = {.value = (block * static_cast<std::uint32_t>(kFacePlacesPerBlock)) + place_in_block};
    ApplyFacePlaceReadMarks(tables, face, width, marks);
  }
}

// Finds the border of `set` at `width` by running the readers forward from every place kept in the set, then one pass
// over the marks in ascending index, which lists the marked blocks outside the set. Allocates one mark byte per block
// of the grid, freed on return.
// Work: for each place of the set, one marking function (border_block_steps.h), whose work does not depend on the
// grid's size; then one read of each mark and one set test for each marked block. The set's size drives the marking, so
// the cost does not grow with the grid times its size levels.
std::vector<CellBlockIndex> FindBorderByMarking(const GridTables& tables, const BlockSet& set, BorderWidth width) {
  std::vector<std::uint8_t> marks(tables.block_count, 0);
  for (std::uint32_t entry = 0; entry < set.range.block_count; ++entry) {
    ApplyMarksFromBlock(tables, set.range.first_block + entry, width, marks.data());
  }
  for (std::uint32_t entry = 0; entry < set.sorted_count; ++entry) {
    ApplyMarksFromBlock(tables, set.sorted_blocks[entry].value, width, marks.data());
  }
  // The list is sized once, so a call allocates the marks and, when the border is not empty, the list: two times.
  std::size_t border_block_count = 0;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    border_block_count += marks[block] != 0 && !ComputeIsInBlockSet(set, CellBlockIndex{block}) ? 1U : 0U;
  }
  std::vector<CellBlockIndex> border_blocks;
  border_blocks.reserve(border_block_count);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    if (marks[block] != 0 && !ComputeIsInBlockSet(set, CellBlockIndex{block})) {
      border_blocks.push_back(CellBlockIndex{block});
    }
  }
  return border_blocks;
}

}  // namespace

std::expected<std::vector<CellBlockRange>, GridFailure>
BuildRanges(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range) {
  // A cut into ranges of no block would never get to the last block: the caller asked for no cut at all.
  if (max_blocks_per_range == 0) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  std::vector<CellBlockRange> ranges;
  const std::uint32_t block_count = grid.counts().cell_blocks;
  for (std::uint32_t first_block = 0; first_block < block_count; first_block += max_blocks_per_range) {
    const std::uint32_t blocks_left = block_count - first_block;
    ranges.push_back({.first_block = first_block, .block_count = std::min(max_blocks_per_range, blocks_left)});
  }
  return ranges;
}

RangeWithBorder BuildRangeWithBorder(const StaggeredGrid& grid, CellBlockRange range) {
  assert(grid.run_on() == RunOn::kCpu);
  assert(range.first_block <= grid.counts().cell_blocks);
  assert(range.block_count <= grid.counts().cell_blocks - range.first_block);
  const BlockSet set = {.range = range, .sorted_blocks = nullptr, .sorted_count = 0};
  return {
      .range = range,
      .border_blocks = FindBorderByMarking(grid.tables(RunOn::kCpu), set, BorderWidth::kTouchingCells),
  };
}

std::expected<std::vector<CellBlockIndex>, GridFailure>
FindBorderBlocks(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  const std::expected<void, GridFailure> check = CheckSortedBlocks(grid, blocks);
  if (!check.has_value()) {
    return std::unexpected(check.error());
  }
  const BlockSet set = {
      .range = {.first_block = 0, .block_count = 0},
      .sorted_blocks = blocks.data(),
      .sorted_count = static_cast<std::uint32_t>(blocks.size()),
  };
  return FindBorderByMarking(grid.tables(RunOn::kCpu), set, width);
}

std::expected<std::uint32_t, GridFailure> FindBorderBlocks(
    const StaggeredGrid& grid,
    std::span<const CellBlockIndex> blocks,
    BorderWidth width,
    std::span<CellBlockIndex> border_blocks
) {
  const std::expected<void, GridFailure> check = CheckSortedBlocks(grid, blocks);
  if (!check.has_value()) {
    return std::unexpected(check.error());
  }
  // The walk searches `blocks` while it writes `border_blocks`, so an output over the set would change the set.
  if (ComputeDoSpansOverlap(blocks, border_blocks)) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  std::uint32_t border_block_count = 0;
  ApplyToBorderBlocks(grid, blocks, width, [&border_blocks, &border_block_count](CellBlockIndex block) {
    if (border_block_count < border_blocks.size()) {
      border_blocks[border_block_count] = block;
    }
    ++border_block_count;
  });
  return border_block_count;
}

}  // namespace fluidity::grid
