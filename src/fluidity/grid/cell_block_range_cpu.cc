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
  RangeWithBorder range_with_border = {.range = range, .border_blocks = {}};
  const GridTables tables = grid.tables(RunOn::kCpu);
  const BlockSet set = {.range = range, .sorted_blocks = nullptr, .sorted_count = 0};
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    if (ComputeIsBorderBlock(tables, set, CellBlockIndex{block}, BorderWidth::kTouchingCells)) {
      range_with_border.border_blocks.push_back(CellBlockIndex{block});
    }
  }
  return range_with_border;
}

std::expected<std::vector<CellBlockIndex>, GridFailure>
FindBorderBlocks(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  const std::expected<void, GridFailure> check = CheckSortedBlocks(grid, blocks);
  if (!check.has_value()) {
    return std::unexpected(check.error());
  }
  std::vector<CellBlockIndex> border_blocks;
  ApplyToBorderBlocks(grid, blocks, width, [&border_blocks](CellBlockIndex block) {
    border_blocks.push_back(block);
  });
  return border_blocks;
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
