#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cell_count_cases.h"

namespace fluidity::grid {
namespace {

// Every cell of a list moved by offset_in_biggest_cells size-level-0 cells along each axis: offset 2^l at size level
// l. The moved list is valid when every coordinate stays where its cells can be split (<sec-cell-sizes>).
std::vector<CellId> BuildMovedCells(const std::vector<CellId>& cells, std::int32_t offset_in_biggest_cells) {
  std::vector<CellId> moved_cells;
  moved_cells.reserve(cells.size());
  for (const CellId cell : cells) {
    const int size_level = ToSizeLevelNumber(cell);
    const std::int32_t offset =
        offset_in_biggest_cells * static_cast<std::int32_t>(1U << static_cast<unsigned>(size_level));
    const CellCoord coord = ToCellCoord(cell);
    moved_cells.push_back(ToCell(size_level, coord.x + offset, coord.y + offset, coord.z + offset));
  }
  return moved_cells;
}

// ---------------------------------------------------------------------------------------------------------------------
// Random grids against the geometric checker
// ---------------------------------------------------------------------------------------------------------------------

TEST(ComputeGridCounts, RandomGrids_EqualGeometricChecker) {
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(grid);
    const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
    ExpectCheckerCounts(ComputeCountsOfList(ToUnitFrame(4), cells), ToUnitFrame(4), cells);
  }
}

// The counts of a list equal those of the grid built from it, field by field: the face pieces counted from the places
// beside each cell equal those counted from the built kinds, on all 200 random grids.
TEST(ComputeGridCounts, RandomGrids_EqualBuiltCounts) {
  // One set of buffers for 4,096 blocks, the most the tests build, given back after each grid.
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(4), {}), {.max_cell_blocks = 4096});
  ASSERT_TRUE(owner.has_value());
  GridBuffers buffers = TakeBuffers(std::move(owner->grid));
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(grid);
    const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
    const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(4), cells);
    std::expected<StaggeredGrid, GridFailure> built = BuildGrid(ToInputs(ToUnitFrame(4), cells), buffers);
    ASSERT_TRUE(counts.has_value());
    ASSERT_TRUE(built.has_value());
    EXPECT_EQ(ToCountFields(*counts), ToCountFields(built->counts()));
    buffers = TakeBuffers(std::move(*built));
  }
}

// The random grids the tests of moved and reordered lists use: the first 50 of the 200. The checker, the costly
// part, runs on all 200 in RandomGrids_EqualGeometricChecker.
constexpr std::size_t kCheapRandomGridCount = 50;

// Expects the counts of the grid at `grid` moved by offset_in_biggest_cells size-level-0 cells. A move changes no
// adjacency, and a side at the grid's edge stays at the edge, since nothing lies beyond either coordinate limit; so
// the face pieces equal those of the unmoved list, which RandomGrids_EqualGeometricChecker holds to the checker. The
// blocks do change with a move and are compared with the checker's set of block ids of the moved list.
//
// clang-tidy: a grid index and an offset; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ExpectMovedCounts(std::size_t grid, std::int32_t offset_in_biggest_cells) {
  const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
  const std::vector<CellId> moved_cells = BuildMovedCells(cells, offset_in_biggest_cells);
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(4), cells);
  const std::expected<GridCounts, GridFailure> moved_counts = ComputeCountsOfList(ToUnitFrame(4), moved_cells);
  ASSERT_TRUE(counts.has_value());
  ASSERT_TRUE(moved_counts.has_value());
  EXPECT_EQ(moved_counts->face_pieces, counts->face_pieces);
  EXPECT_EQ(moved_counts->cell_blocks, ComputeCheckerBlockCount(moved_cells));
  EXPECT_EQ(std::get<1>(ToCountFields(*moved_counts)), std::get<1>(ToCountFields(*counts)));
  EXPECT_EQ(moved_counts->active_cells, cells.size());
}

// The box of 3 size-level-0 cells moved from 0..2 to -1..1 on each axis: cells on both sides of zero, where the floor
// of a negative coordinate differs from its truncation. 50 grids; RandomGrids_EqualGeometricChecker runs all 200.
TEST(ComputeGridCounts, RandomGridsMovedAcrossZero_CountAsTheChecker) {
  for (std::size_t grid = 0; grid < kCheapRandomGridCount; ++grid) {
    SCOPED_TRACE(grid);
    ExpectMovedCounts(grid, -1);
  }
}

// The box moved to size-level-0 cells 2^15 - 3..2^15 - 1: a size-level-3 cell at the high side reaches kMaxCoord, and
// nothing of any size level lies beyond it. 50 grids; RandomGrids_EqualGeometricChecker runs all 200.
TEST(ComputeGridCounts, RandomGridsMovedToHighCoordLimit_CountAsTheChecker) {
  constexpr auto kHighOffset = static_cast<std::int32_t>((1U << 15U) - 3U);
  for (std::size_t grid = 0; grid < kCheapRandomGridCount; ++grid) {
    SCOPED_TRACE(grid);
    ExpectMovedCounts(grid, kHighOffset);
  }
}

// The box moved to size-level-0 cells -(2^15 - 1)..-(2^15 - 3): the lowest box whose size-level-2 cells can all be
// split, its size-level-3 cells from -2^18 + 8, 7 above -kMaxCoord. 50 grids; RandomGrids_EqualGeometricChecker runs
// all 200.
TEST(ComputeGridCounts, RandomGridsMovedToLowCoordLimit_CountAsTheChecker) {
  constexpr std::int32_t kLowOffset = -static_cast<std::int32_t>((1U << 15U) - 1U);
  for (std::size_t grid = 0; grid < kCheapRandomGridCount; ++grid) {
    SCOPED_TRACE(grid);
    ExpectMovedCounts(grid, kLowOffset);
  }
}

// The random grids are shuffled; the same lists in the fixed order of their generator give the same counts. 50 seeds;
// RandomGrids_EqualGeometricChecker counts all 200 shuffled lists.
TEST(ComputeGridCounts, ShuffledList_GivesSameCountsAsFixedOrder) {
  for (std::uint64_t seed = 1; seed <= kCheapRandomGridCount; ++seed) {
    SCOPED_TRACE(seed);
    const std::vector<CellId> fixed_order_cells = BuildRandomGridCells(seed, 4, 3);
    const std::expected<GridCounts, GridFailure> fixed_order_counts =
        ComputeCountsOfList(ToUnitFrame(4), fixed_order_cells);
    const std::expected<GridCounts, GridFailure> shuffled_counts =
        ComputeCountsOfList(ToUnitFrame(4), BuildShuffledRandomGrids().at(seed - 1));
    ASSERT_TRUE(fixed_order_counts.has_value());
    ASSERT_TRUE(shuffled_counts.has_value());
    EXPECT_EQ(ToCountFields(*shuffled_counts), ToCountFields(*fixed_order_counts));
  }
}

}  // namespace
}  // namespace fluidity::grid
