
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_cell_checks_cases.h"

namespace fluidity::grid {
namespace {

// The 7 size-level-1 cells of size level 0 {0,0,0} without its eighth 5, size level 1 {1,0,1}.
std::vector<CellId> BuildSevenOfEightSmallCells() {
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 5);
  return cells;
}

// Size level 0 {0,0,0} split, its eighth 0, size level 1 {0,0,0}, missing, and each of its 7 other size-level-1 cells
// split into its 8 size-level-2 cells: 56 active cells.
std::vector<CellId> BuildMissingCellBesideSplitSiblings() {
  std::vector<CellId> cells;
  const std::vector<CellId> size_level_one_cells = ToSmallCells(ToCell(0, 0, 0, 0));
  for (std::size_t cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
    const std::vector<CellId> small_cells = ToSmallCells(size_level_one_cells.at(cell_eighth));
    cells.insert(cells.end(), small_cells.begin(), small_cells.end());
  }
  return cells;
}

// `count` size-level-0 cells along x from 0: a valid list of one size level.
std::vector<CellId> BuildBigCellsAlongX(std::int32_t count) {
  std::vector<CellId> cells;
  cells.reserve(static_cast<std::size_t>(count));
  for (std::int32_t x = 0; x < count; ++x) {
    cells.push_back(ToCell(0, x, 0, 0));
  }
  return cells;
}

// ---------------------------------------------------------------------------------------------------------------------
// The span form
// ---------------------------------------------------------------------------------------------------------------------

// Size level 2 {4,4,4} is the smaller cell of the pair and names size level 0 {0,0,0}, two size levels bigger.
TEST(CheckSizeStepRule, CornerTouchTwoSizeLevelsApart_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 4));
  CheckFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 4, 4),
      ToCell(0, 0, 0, 0)
  );
}

TEST(CheckSizeStepRule, EdgeTouchTwoSizeLevelsApart_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 0));
  CheckFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 4, 0),
      ToCell(0, 0, 0, 0)
  );
}

TEST(CheckSizeStepRule, FaceTouchTwoSizeLevelsApart_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = BuildTwoSizeLevelsApart(ToCell(2, 4, 0, 0));
  CheckFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 0, 0),
      ToCell(0, 0, 0, 0)
  );
}

// Size level 1 {2,0,0} covers [1, 1.5) x [0, 0.5) x [0, 0.5): it touches size level 0 {0,0,0} on the side x = 1, one
// size level apart. Its 7 siblings are missing, which this check does not look at.
TEST(CheckSizeStepRule, OneSizeLevelApart_Passes) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(1, 2, 0, 0)};
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), cells).has_value());
}

TEST(CheckSizeStepRule, SevenOfEightSmallCells_Passes) {
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), BuildSevenOfEightSmallCells()).has_value());
}

// The hidden missing sibling is a sibling failure, which this check leaves out; the rest is a valid grid.
TEST(CheckSizeStepRule, MissingCellBesideSplitSiblings_Passes) {
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), BuildMissingCellBesideSplitSiblings()).has_value());
}

// Input {a, a, bad1, bad2}: step 1 fails at input indexes 2 and 3, keys (1, 2, 0) and (1, 3, 0); the overlap of a with
// itself is step 2, key (2, 1, 0). The smallest key is (1, 2, 0): bad1, named as given, with bit 63 clear.
TEST(CheckSizeStepRule, BadIdAfterRepeat_IsBadInputOfFirstBadId) {
  const CellId cell = ToCell(0, 0, 0, 0);
  const CellId first_bad_id = {.value = cell.value & ~(std::uint64_t{1} << 63U)};
  const CellId second_bad_id = {.value = 0};
  const std::vector<CellId> cells = {cell, cell, first_bad_id, second_bad_id};
  CheckFailure(CheckSizeStepRule(ToUnitFrame(3), cells), GridError::kBadInput, first_bad_id);
}

// Two ids with bit 61 set keep ordinary sort keys, so sorting puts Y = size level 0 {0,0,0} before X = size level 0
// {10,0,0}. Step 1 runs in input order, before the sort: input index 0, X, has key (1, 0, 0) and is named as given.
std::vector<CellId> BuildBadIdsThatSortingReorders() {
  constexpr std::uint64_t kBitSixtyOne = std::uint64_t{1} << 61U;
  return {
      CellId{.value = ToCell(0, 10, 0, 0).value | kBitSixtyOne},
      CellId{.value = ToCell(0, 0, 0, 0).value | kBitSixtyOne},
  };
}

TEST(CheckSizeStepRule, BadIdsThatSortingReorders_IsBadInputOfFirstInInputOrder) {
  const std::vector<CellId> cells = BuildBadIdsThatSortingReorders();
  CheckFailure(CheckSizeStepRule(ToUnitFrame(3), cells), GridError::kBadInput, cells.at(0));
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  CheckFailure(
      CheckSizeStepRule(ToInputs(ToUnitFrame(3), cells), *grid_buffers->buffers),
      GridError::kBadInput,
      cells.at(0)
  );
}

// kMaxCoord + 1 is the one coordinate outside the range that an id can hold.
TEST(CheckSizeStepRule, CoordBeyondLimit_IsCoordOutOfRange) {
  const CellId beyond_limit = ToCell(0, kMaxCoord + 1, 0, 0);
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), beyond_limit};
  CheckFailure(CheckSizeStepRule(ToUnitFrame(3), cells), GridError::kCoordOutOfRange, beyond_limit);
}

TEST(CheckSizeStepRule, BigCellsAtBothCoordLimits_Pass) {
  const std::vector<CellId> cells = {
      ToCell(0, kMaxCoord, kMaxCoord, kMaxCoord),
      ToCell(0, -kMaxCoord, -kMaxCoord, -kMaxCoord),
      ToCell(0, kMaxCoord, -kMaxCoord, 0),
  };
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), cells).has_value());
}

// Size level 2 {kMaxCoord, 0, 0} covers x in [65535.75, 65536) m; size level 0 {65536, 0, 0} begins at x = 65536 m, on
// its high side, two size levels bigger. The place beside the small cell, x = 2^18, lies beyond the range at size
// level 2 and is found at size level 0.
TEST(CheckSizeStepRule, SmallCellAtHighCoordLimitBesideBigCell_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = {ToCell(0, 65536, 0, 0), ToCell(2, kMaxCoord, 0, 0)};
  CheckFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, kMaxCoord, 0, 0),
      ToCell(0, 65536, 0, 0)
  );
}

TEST(CheckSizeStepRule, BadFrame_IsBadInput) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  GridFrame origin_not_finite = ToUnitFrame(3);
  origin_not_finite.origin.y = std::numeric_limits<double>::quiet_NaN();
  GridFrame width_zero = ToUnitFrame(3);
  width_zero.biggest_cell_width = 0;
  GridFrame width_not_finite = ToUnitFrame(3);
  width_not_finite.biggest_cell_width = std::numeric_limits<double>::infinity();
  for (const GridFrame frame : {origin_not_finite, width_zero, width_not_finite, ToUnitFrame(0), ToUnitFrame(17)}) {
    CheckFailure(CheckSizeStepRule(frame, cells), GridError::kBadInput);
  }
}

// The frame of `size_level_count` size levels whose smallest width, the biggest halved size_level_count - 1 times, is
// smallest_width m.
GridFrame ToFrameOfSmallestWidth(int size_level_count, double smallest_width) {
  GridFrame frame = ToUnitFrame(size_level_count);
  frame.biggest_cell_width = std::ldexp(smallest_width, size_level_count - 1);
  return frame;
}

// The smallest width is checked against 2^-1022, the smallest normal double: exactly 2^-1022 passes, one halving below
// is kBadInput, for 1, 2, 8 and 16 size levels. With 16 size levels that is biggest cells of 2^-1007 m and 2^-1008 m.
TEST(CheckSizeStepRule, SmallestWidthBelowNormalRange_IsBadInput) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  constexpr double kSmallestNormal = std::numeric_limits<double>::min();
  for (const int size_level_count : {1, 2, 8, 16}) {
    SCOPED_TRACE(testing::Message() << "size levels " << size_level_count);
    EXPECT_TRUE(CheckSizeStepRule(ToFrameOfSmallestWidth(size_level_count, kSmallestNormal), cells).has_value());
    CheckFailure(
        CheckSizeStepRule(ToFrameOfSmallestWidth(size_level_count, kSmallestNormal / 2), cells),
        GridError::kBadInput
    );
  }
}

TEST(CheckSizeStepRule, EmptyList_Passes) {
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(1), std::span<const CellId>()).has_value());
}

// ---------------------------------------------------------------------------------------------------------------------
// The input checks of the buffers form
// ---------------------------------------------------------------------------------------------------------------------

TEST(GridInputs, TwoLists_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const Buffer<CellId> cell_buffer = BuildBufferHolding(RunOn::kCpu, cells);
  GridInputs inputs = ToInputs(ToUnitFrame(1), cells);
  inputs.active_cell_buffer = &cell_buffer;
  inputs.active_cell_buffer_count = 1;
  CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, BufferCountAboveSize_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const Buffer<CellId> cell_buffer =
      BuildBufferHolding(RunOn::kCpu, std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0)});
  GridInputs inputs = ToInputs(ToUnitFrame(1), {});
  inputs.active_cell_buffer = &cell_buffer;
  inputs.active_cell_buffer_count = 3;
  CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, CountWithoutBuffer_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  GridInputs inputs = ToInputs(ToUnitFrame(1), {});
  inputs.active_cell_buffer_count = 1;
  CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, VersionZero_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  GridInputs inputs = ToInputs(ToUnitFrame(1), cells);
  inputs.version = {.value = 0};
  CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, BufferList_IsCheckedAsTheSpan) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  // Only the first 2 entries are the list: the third, two size levels from the first, is not read.
  const Buffer<CellId> cell_buffer =
      BuildBufferHolding(RunOn::kCpu, std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(1, 2, 0, 0), ToCell(2, 4, 4, 4)});
  GridInputs inputs = ToInputs(ToUnitFrame(3), {});
  inputs.active_cell_buffer = &cell_buffer;
  inputs.active_cell_buffer_count = 2;
  EXPECT_TRUE(CheckSizeStepRule(inputs, *grid_buffers->buffers).has_value());
  inputs.active_cell_buffer_count = 3;
  CheckFailure(
      CheckSizeStepRule(inputs, *grid_buffers->buffers),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 4, 4),
      ToCell(0, 0, 0, 0)
  );
}

// The list is the lent builder buffers' own cell ids, which the sorted copy would overwrite while reading them.
// A first check leaves the valid list {size level 0 {0,0,0}, size level 1 {2,0,0}} sorted in the cell ids, so the
// aliased list holds valid ids and only the alias check can stop it (without it, base's copy would see the overlap).
TEST(CheckSizeStepRule, ListInLentCellIds_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = {ToCell(1, 2, 0, 0), ToCell(0, 0, 0, 0)};
  ASSERT_TRUE(CheckSizeStepRule(ToInputs(ToUnitFrame(3), cells), *grid_buffers->buffers).has_value());
  const Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(*grid_buffers->builder_buffers).cell_ids;
  GridInputs inputs = ToInputs(ToUnitFrame(3), {});
  inputs.active_cell_buffer = &cell_ids;
  inputs.active_cell_buffer_count = 2;
  CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

// The arrays of the grid buffers point at no builder buffers, then at builder buffers that were never lent to them, of
// a smaller room and of the same room. The cell ids of the lent ones stay all zero, as built: nothing is written.
TEST(CheckSizeStepRule, BuilderBuffersNotTheLentOnes_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(4);
  const std::unique_ptr<CpuGridBuffers> smaller_grid_buffers = BuildCpuGridBuffers(1);
  const std::unique_ptr<CpuGridBuffers> other_grid_buffers = BuildCpuGridBuffers(4);
  const bool are_all_built =
      grid_buffers != nullptr && smaller_grid_buffers != nullptr && other_grid_buffers != nullptr;
  ASSERT_TRUE(are_all_built);
  const std::vector<CellId> cells = {ToCell(1, 2, 0, 0), ToCell(0, 0, 0, 0)};
  const GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  GridBuffersArrays& arrays = ToGridBuffersArrays(*grid_buffers->buffers);
  GridBuilderBuffers* const lent_builder_buffers = arrays.builder_buffers;
  const std::vector<std::uint64_t> cell_ids_before = ReadLentCellIds(*lent_builder_buffers, cells.size());

  const std::array<GridBuilderBuffers*, 3> replacements = {
      nullptr,
      smaller_grid_buffers->builder_buffers.get(),
      other_grid_buffers->builder_buffers.get(),
  };
  for (GridBuilderBuffers* const builder_buffers : replacements) {
    arrays.builder_buffers = builder_buffers;
    CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
    EXPECT_EQ(ReadLentCellIds(*lent_builder_buffers, cells.size()), cell_ids_before);
  }

  // With the lent builder buffers back, the same call passes and writes the sorted copy.
  arrays.builder_buffers = lent_builder_buffers;
  EXPECT_TRUE(CheckSizeStepRule(inputs, *grid_buffers->buffers).has_value());
  EXPECT_EQ(ReadLentCellIds(*lent_builder_buffers, cells.size()).at(0), ToCell(0, 0, 0, 0).value);
}

// Room for 1 block is 64 cell places; 65 cells cannot fit, whatever they are.
TEST(CheckSizeStepRule, MoreCellsThanPlaces_IsTooManyCellBlocks) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = BuildBigCellsAlongX(65);
  CheckFailure(
      CheckSizeStepRule(ToInputs(ToUnitFrame(1), cells), *grid_buffers->buffers),
      GridError::kTooManyCellBlocks
  );
}

TEST(CheckSizeStepRule, AsManyCellsAsPlaces_Passes) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = BuildBigCellsAlongX(64);
  EXPECT_TRUE(CheckSizeStepRule(ToInputs(ToUnitFrame(1), cells), *grid_buffers->buffers).has_value());
}

// The length is an input check: it comes before step 1 reads the bad id at input index 0. The version comes before
// the length.
TEST(CheckSizeStepRule, TooLongListWithBadId_IsTooManyCellBlocks) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  std::vector<CellId> cells = BuildBigCellsAlongX(65);
  cells.at(0) = {.value = cells.at(0).value & ~(std::uint64_t{1} << 63U)};
  GridInputs inputs = ToInputs(ToUnitFrame(1), cells);
  CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kTooManyCellBlocks);
  inputs.version = {.value = 0};
  CheckFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

// Sorted, the first cell is size level 2 {2,0,0}, eighth 0 of size level 1 {1,0,0}; that cell is eighth 1 of size
// level 0 {0,0,0}. The siblings of {1,0,0} are all present; its walk then reaches size level 0 {0,0,0}, whose eighth 0,
// size level 1 {0,0,0}, is neither listed nor split into listed cells.
TEST(BuildSortedCellList, MissingCellBesideSplitSiblings_IsSiblingCellMissingOfThatCell) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(64);
  ASSERT_NE(grid_buffers, nullptr);
  CheckFailure(
      CheckThroughSortedCellList(ToUnitFrame(3), BuildMissingCellBesideSplitSiblings(), *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      ToCell(1, 0, 0, 0),
      ToCell(0, 0, 0, 0)
  );
}

// 3 size levels, {size level 0 {0,0,0}, size level 2 {4,4,4}}. Sorted, {4,4,4} is at position 1 and fails two
// steps there. Step 3: its big cell is size level 1 {2,2,2}; eighth 0 is {4,4,4} itself, eighth 1 is size level 2
// {5,4,4} (x = 2 * 2 + 1), which is not listed: key (3, 1, 1). Step 4: the corner touch with {0,0,0} in direction 0:
// key (4, 1, 0). The smaller key is step 3's.
TEST(BuildSortedCellList, MissingSiblingAndSizeStepBreak_IsSiblingCellMissing) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  CheckFailure(
      CheckThroughSortedCellList(ToUnitFrame(3), BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 4)), *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      ToCell(2, 5, 4, 4),
      ToCell(1, 2, 2, 2)
  );
}

// Size level 0 {0,0,0} split, its size-level-1 cell {1,0,0} split too, and size level 2 {2,0,0}, eighth 0 of {1,0,0},
// left out. No active cell lies at the low corner of {1,0,0}, so {1,0,0} is not present among the siblings of
// {0,0,0}, and the walk from size level 1 {0,0,0}, at position 0, names it: {1,0,0} is eighth 1 (x odd), so it is the
// first cell above the one left out that is not eighth 0.
TEST(BuildSortedCellList, EighthZeroCellLeftOut_NamesFirstBiggerCellNotAtEighthZero) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(64);
  ASSERT_NE(grid_buffers, nullptr);
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 1);
  for (const CellId small_cell : ToSmallCells(ToCell(1, 1, 0, 0))) {
    if (small_cell.value != ToCell(2, 2, 0, 0).value) {
      cells.push_back(small_cell);
    }
  }
  CheckFailure(
      CheckThroughSortedCellList(ToUnitFrame(3), cells, *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      ToCell(1, 1, 0, 0),
      ToCell(0, 0, 0, 0)
  );
}

// 2 size levels. The 8 size-level-1 cells of size level 0 {-131071, 0, 0}, at x = -262142 and -262141, pass. The 4
// size-level-1 cells at x = -kMaxCoord = -262143, y and z 0 or 1, lie in size level 0 {-131072, 0, 0}, whose other 4
// small cells would be at x = -262144, outside the range. Sorted, x = -262143, y = z = 0 comes first; its big cell's
// eighth 0 is the first small cell not present, and no id names it.
TEST(BuildSortedCellList, SmallCellAtLowCoordLimit_IsSiblingCellMissingWithNoCell) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(64);
  ASSERT_NE(grid_buffers, nullptr);
  std::vector<CellId> cells = ToSmallCells(ToCell(0, -131071, 0, 0));
  ASSERT_TRUE(CheckThroughSortedCellList(ToUnitFrame(2), cells, *grid_buffers->buffers).has_value());
  for (std::int32_t z = 0; z < 2; ++z) {
    for (std::int32_t y = 0; y < 2; ++y) {
      cells.push_back(ToCell(1, -kMaxCoord, y, z));
    }
  }
  CheckFailure(
      CheckThroughSortedCellList(ToUnitFrame(2), cells, *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      CellId{0},
      ToCell(0, -131072, 0, 0)
  );
}

}  // namespace
}  // namespace fluidity::grid
