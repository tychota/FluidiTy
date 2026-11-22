#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/staggered_grid_cell_count_cases.h"

namespace fluidity::grid {
namespace {

// Size level 0 {3,3,3} split down its eighth 7 to size level 15, in 16 size levels: at each size level l = 1..14 the 7
// small cells other than eighth 7, and at size level 15 all 8. At l = 1..14 the 7 cells keep 9 sides between them and
// 21 outer quarter sides: the 12 of their group's low sides, against the bigger cells or the grid's edge at l = 1,
// and 9 of its high sides, at the grid's edge. The 3 sides they share with eighth 7 are split. At size level 15 the 8
// cells keep 12 + 24. So 14 x 30 + 36 = 456, and 1 + 15 blocks, one per size level.
// One more size-level-0 cell, alone at {262139, 262140, 262140}, adds 6 face pieces and 1 block: 462 and 17. It lies
// where a split lookup from size level 15 would land if it were made: the size-level-15 cell at 131070 on each axis
// has the place 131069 below its low X side, whose small cell 2 x 131069 + 1 at size level 16 would pack as this cell.
// So a count that made the split lookup at size level 16 would read that side as split, in a release build too.
TEST(ComputeGridCounts, CellSplitDownToSizeLevelFifteen_EqualsGeometricChecker) {
  std::vector<CellId> cells;
  CellId cell = ToCell(0, 3, 3, 3);
  for (int size_level = 0; size_level < 15; ++size_level) {
    const std::vector<CellId> small_cells = ToSmallCells(cell);
    for (std::size_t cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      if (cell_eighth != 7 || size_level == 14) {
        cells.push_back(small_cells.at(cell_eighth));
      }
    }
    cell = small_cells.at(7);
  }
  cells.push_back(ToCell(0, 262139, 262140, 262140));
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(16), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->face_pieces, 462U);
  EXPECT_EQ(counts->cell_blocks, 17U);
  EXPECT_EQ(counts->active_cells, 107U);
  CheckCheckerCounts(counts, ToUnitFrame(16), cells);
}

// 4,097 cells 4 apart along x, each alone in its block: more blocks than the 4,096 of any room the tests build, which
// this count never compares with.
TEST(ComputeGridCounts, MoreBlocksThanAnyTestRoom_CountsThemAll) {
  std::vector<CellId> cells;
  for (std::int32_t index = 0; index <= 4096; ++index) {
    cells.push_back(ToCell(0, 4 * index, 0, 0));
  }
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(3), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->cell_blocks, 4097U);
  EXPECT_EQ(counts->face_pieces, 6U * 4097U);
}

}  // namespace
}  // namespace fluidity::grid
