#include "fluidity/grid/new_cell_rule.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/new_cell_rule_cases.h"
#include "fluidity/grid/testing/new_cell_rule_reference.h"

namespace fluidity::grid {
namespace {

// Builds a grid of the hand frame from a list of active cells.
std::expected<TestGrid, GridFailure> BuildHandGrid(std::span<const CellId> cells) {
  return BuildTestGrid({.frame = ToHandGridFrame(), .active_cells = cells, .version = {.value = 1}});
}

// Finds the facts the hand-worked table gives for `new_cell`. The test fails when the table has no such cell.
NewCellNeighbors FindHandFacts(const std::vector<HandCase>& hand_cases, CellId new_cell) {
  for (const HandCase& hand_case : hand_cases) {
    if (hand_case.new_cell.value == new_cell.value) {
      return hand_case.expected;
    }
  }
  ADD_FAILURE() << "no hand case for the new cell";
  return {.touches_much_smaller_cell = false, .touches_bigger_cell = false};
}

// Checks the reference against the hand-worked facts of `new_cell` in the near-small grid.
void CheckNearSmallFacts(CellId new_cell) {
  const std::vector<CellId> cells = BuildNearSmallCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildHandGrid(cells);
  ASSERT_TRUE(test_grid.has_value());
  const NewCellNeighbors expected = FindHandFacts(BuildNearSmallCases(), new_cell);
  const NewCellNeighbors actual = ComputeNewCellNeighborsByAllCells(test_grid->grid.tables(RunOn::kCpu), new_cell);
  EXPECT_EQ(actual.touches_much_smaller_cell, expected.touches_much_smaller_cell);
  EXPECT_EQ(actual.touches_bigger_cell, expected.touches_bigger_cell);
}

// Checks the reference against the hand-worked facts of both new cells of the range-end grid.
void CheckRangeEndFacts() {
  const std::vector<CellId> cells = BuildRangeEndCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildHandGrid(cells);
  ASSERT_TRUE(test_grid.has_value());
  for (const HandCase& hand_case : BuildRangeEndCases()) {
    const NewCellNeighbors actual =
        ComputeNewCellNeighborsByAllCells(test_grid->grid.tables(RunOn::kCpu), hand_case.new_cell);
    EXPECT_EQ(actual.touches_much_smaller_cell, hand_case.expected.touches_much_smaller_cell);
    EXPECT_EQ(actual.touches_bigger_cell, hand_case.expected.touches_bigger_cell);
  }
}

// The hand grids obey the grid's rules, so the grid builder takes them: 23 and 2 active cells.
TEST(NewCellRuleCases, HandGrids_BuildAsValidGrids) {
  const std::vector<CellId> near_fine_cells = BuildNearSmallCells();
  const std::vector<CellId> range_end_cells = BuildRangeEndCells();
  EXPECT_EQ(near_fine_cells.size(), 23U);
  EXPECT_EQ(range_end_cells.size(), 2U);
  EXPECT_TRUE(BuildHandGrid(near_fine_cells).has_value());
  EXPECT_TRUE(BuildHandGrid(range_end_cells).has_value());
}

TEST(ComputeNewCellNeighborsByAllCells, BiggestCellBesideSmallCells_MustSplit) {
  CheckNearSmallFacts(ToHandCell(0, 1, 0, 0));
}

TEST(ComputeNewCellNeighborsByAllCells, SizeLevelOneCellBesideSmallCorner_MustSplit) {
  CheckNearSmallFacts(ToHandCell(1, 2, 1, 0));
}

TEST(ComputeNewCellNeighborsByAllCells, SizeLevelOneCellBesideBiggerCell_TouchesBigger) {
  CheckNearSmallFacts(ToHandCell(1, 2, 0, 0));
}

TEST(ComputeNewCellNeighborsByAllCells, SizeLevelOneCellBesideSizeLevelTwoOnly_NeitherFact) {
  CheckNearSmallFacts(ToHandCell(1, 2, 1, 1));
}

TEST(ComputeNewCellNeighborsByAllCells, SizeLevelTwoCellOnLastButOneSizeLevel_TouchesBigger) {
  CheckNearSmallFacts(ToHandCell(2, 4, 2, 0));
}

TEST(ComputeNewCellNeighborsByAllCells, SizeLevelTwoCellBesideSmallestCells_NeitherFact) {
  CheckNearSmallFacts(ToHandCell(2, 4, 3, 0));
}

TEST(ComputeNewCellNeighborsByAllCells, LastSizeLevelCellBesideSameSize_NeitherFact) {
  CheckNearSmallFacts(ToHandCell(3, 8, 7, 0));
}

TEST(ComputeNewCellNeighborsByAllCells, LastSizeLevelCellBesideBiggerCell_TouchesBigger) {
  CheckNearSmallFacts(ToHandCell(3, 8, 5, 0));
}

TEST(ComputeNewCellNeighborsByAllCells, BoxTouchingNothing_NeitherFact) {
  CheckNearSmallFacts(ToHandCell(0, 5, 5, 5));
}

TEST(ComputeNewCellNeighborsByAllCells, CornerOnlyOfSameSize_NeitherFact) {
  CheckNearSmallFacts(ToHandCell(0, 2, 0, 1));
}

TEST(ComputeNewCellNeighborsByAllCells, CornerOnlyOfBiggerCell_TouchesBigger) {
  CheckNearSmallFacts(ToHandCell(1, 4, 0, 2));
}

TEST(ComputeNewCellNeighborsByAllCells, CornerOnlyOfSmallestCell_MustSplit) {
  CheckNearSmallFacts(ToHandCell(0, 1, 1, -1));
}

TEST(ComputeNewCellNeighborsByAllCells, BoxesAtBothEndsOfIdRange_NeitherFact) {
  CheckRangeEndFacts();
}

}  // namespace
}  // namespace fluidity::grid
