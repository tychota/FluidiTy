#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/testing/staggered_grid_cell_count_cases.h"

namespace fluidity::grid {
namespace {

void CheckFailure(
    const std::expected<GridCounts, GridFailure>& result,
    GridError error,
    CellId cell = CellId{0},
    CellId other_cell = CellId{0}
) {
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, error);
  EXPECT_EQ(result.error().cell.value, cell.value);
  EXPECT_EQ(result.error().other_cell.value, other_cell.value);
  EXPECT_EQ(result.error().magnitude, 0U);
}

// Size level 0 {0,0,0} beside the 8 size-level-1 cells of size level 0 {1,0,0}.
std::vector<CellId> BuildOneSplitCell() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const std::vector<CellId> small_cells = ToSmallCells(ToCell(0, 1, 0, 0));
  cells.insert(cells.end(), small_cells.begin(), small_cells.end());
  return cells;
}

// ---------------------------------------------------------------------------------------------------------------------
// Fixtures with their numbers worked out
// ---------------------------------------------------------------------------------------------------------------------

TEST(ComputeGridCounts, OneBigCell_HasSixFacePieces) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(3), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->cell_blocks, 1U);
  EXPECT_EQ(counts->face_pieces, 6U);
  EXPECT_EQ(counts->active_cells, 1U);
  EXPECT_EQ(counts->active_cells_per_size_level[0], 1U);
}

// The big cell keeps 5 whole sides; its +X side is the 4 quarters the small cells keep. The small cells have 24 outer
// sides, 4 of them those quarters, and 12 sides between them: 5 + 24 + 12 = 41. Size level 0 has block (0,0,0) of the
// big cell and the split cell, size level 1 block (0,0,0) of the small cells x = 2..3: 2 blocks.
TEST(ComputeGridCounts, OneSplitCell_CountsBlocksOfBothSizeLevels) {
  const std::vector<CellId> cells = BuildOneSplitCell();
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(3), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->cell_blocks, 2U);
  EXPECT_EQ(counts->active_cells_per_size_level[0], 1U);
  EXPECT_EQ(counts->active_cells_per_size_level[1], 8U);
  EXPECT_EQ(counts->active_cells_per_size_level[2], 0U);
  EXPECT_EQ(counts->active_cells, 9U);
  EXPECT_EQ(counts->face_pieces, 41U);
}

// x = 3 lies in block 0 and x = 4 in block 1. The two cells share one side: 12 - 1 = 11 face pieces.
TEST(ComputeGridCounts, TwoCellsAcrossABlockSide_ShareOneFacePiece) {
  const std::vector<CellId> cells = {ToCell(0, 3, 0, 0), ToCell(0, 4, 0, 0)};
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(3), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->cell_blocks, 2U);
  EXPECT_EQ(counts->face_pieces, 11U);
}

// The two cells at the limits each keep their 6 sides, and the 8 small cells 8 x 6 - 12 = 36: 48 face pieces. Blocks:
// x = -kMaxCoord in size-level-0 block -2^16, x = kMaxCoord in block 2^16 - 1, the split cell in block 0, its small
// cells in size-level-1 block 0: 4. A count that made an id beyond a limit would find, below -kMaxCoord, the small
// cell 2 (-2^18) + 1, whose packed id is size level 1 {1,0,0}, and above kMaxCoord the small cell 2^19, packed as size
// level 1 {0,0,0}: both sides would read as split, and the count would be 46.
TEST(ComputeGridCounts, CellsAtBothCoordLimits_CountTheirEdgeFacePieces) {
  std::vector<CellId> cells = {ToCell(0, -kMaxCoord, 0, 0), ToCell(0, kMaxCoord, 0, 0)};
  const std::vector<CellId> small_cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.insert(cells.end(), small_cells.begin(), small_cells.end());
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(2), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->face_pieces, 48U);
  EXPECT_EQ(counts->cell_blocks, 4U);
  CheckCheckerCounts(counts, ToUnitFrame(2), cells);
}

// Size level 0 {2^17 - 1,0,0} split, its size-level-1 cells at x = kMaxCoord - 1 and kMaxCoord, and size level 0
// {2^17,0,0} beside them. The place beyond the 4 small cells at x = kMaxCoord is 2^18, outside the range at size level
// 1: no cell of that size level is there and none is split, so each small cell keeps its high side, a quarter of the
// big cell's side, and the big cell keeps none on its low side, which is split. The fixture of OneSplitCell mirrored:
// 41 face pieces, 2 blocks of size level 0 (block 2^15 - 1 and 2^15) and 1 of size level 1.
TEST(ComputeGridCounts, SmallCellsAtHighCoordLimitBesideBigCell_KeepItsQuarters) {
  constexpr auto kLastSplittableCoord = static_cast<std::int32_t>((1U << 17U) - 1U);
  std::vector<CellId> cells = ToSmallCells(ToCell(0, kLastSplittableCoord, 0, 0));
  cells.push_back(ToCell(0, kLastSplittableCoord + 1, 0, 0));
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(2), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->face_pieces, 41U);
  EXPECT_EQ(counts->cell_blocks, 3U);
  CheckCheckerCounts(counts, ToUnitFrame(2), cells);
}

// Size level 0 {0,0,0} split down to size level 3: 512 cells of size level 3 in a cube of 8 on each side. On each axis
// 9 planes of 8 x 8 face pieces: 3 x 9 x 64 = 1,728. Blocks: 1 of size level 0, 1 of size level 1 (2 x 2 x 2 cells),
// 1 of size level 2 (4 x 4 x 4), 8 of size level 3: 11. The split cells, 1 + 8 + 64 = 73 = 511 / 7, fill the copy's
// room of 512 + 512 / 7 = 585 ids to the last one.
TEST(ComputeGridCounts, FullySplitCell_FillsTheCopyToItsLastId) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int split_size_level = 0; split_size_level < 3; ++split_size_level) {
    std::vector<CellId> small_cells;
    for (const CellId cell : cells) {
      const std::vector<CellId> eight = ToSmallCells(cell);
      small_cells.insert(small_cells.end(), eight.begin(), eight.end());
    }
    cells = small_cells;
  }
  ASSERT_EQ(cells.size(), 512U);
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(4), cells);
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(counts->face_pieces, 1728U);
  EXPECT_EQ(counts->cell_blocks, 11U);
  EXPECT_EQ(counts->active_cells_per_size_level[3], 512U);
}

TEST(ComputeGridCounts, EmptyList_GivesZeroCounts) {
  const std::expected<GridCounts, GridFailure> counts = ComputeCountsOfList(ToUnitFrame(3), {});
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(ToCountFields(*counts), ToCountFields(GridCounts{}));
}

TEST(ComputeGridCounts, CpuBufferList_GivesSameCountsAsSpan) {
  const std::vector<CellId> cells = BuildOneSplitCell();
  const std::expected<GridCounts, GridFailure> span_counts = ComputeCountsOfList(ToUnitFrame(3), cells);
  ASSERT_TRUE(span_counts.has_value());
  const Buffer<CellId> cell_buffer = BuildBufferHolding(RunOn::kCpu, cells);
  GridInputs inputs = ToInputs(ToUnitFrame(3), {});
  inputs.active_cell_buffer = &cell_buffer;
  inputs.active_cell_buffer_count = static_cast<std::uint32_t>(cells.size());
  const std::expected<GridCounts, GridFailure> buffer_counts = ComputeGridCounts(inputs);
  ASSERT_TRUE(buffer_counts.has_value());
  EXPECT_EQ(ToCountFields(*buffer_counts), ToCountFields(*span_counts));
}

// ---------------------------------------------------------------------------------------------------------------------
// The checks of a build
// ---------------------------------------------------------------------------------------------------------------------

TEST(ComputeGridCounts, CellInsideActiveBigCell_IsCellsOverlap) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(1, 1, 1, 1)};
  CheckFailure(
      ComputeCountsOfList(ToUnitFrame(3), cells),
      GridError::kCellsOverlap,
      ToCell(1, 1, 1, 1),
      ToCell(0, 0, 0, 0)
  );
}

TEST(ComputeGridCounts, SameCellTwice_IsCellsOverlap) {
  const std::vector<CellId> cells = {ToCell(0, 2, 0, 0), ToCell(0, 2, 0, 0)};
  CheckFailure(
      ComputeCountsOfList(ToUnitFrame(3), cells),
      GridError::kCellsOverlap,
      ToCell(0, 2, 0, 0),
      ToCell(0, 2, 0, 0)
  );
}

TEST(ComputeGridCounts, CoordBeyondLimit_IsCoordOutOfRange) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(0, kMaxCoord + 1, 0, 0)};
  CheckFailure(ComputeCountsOfList(ToUnitFrame(3), cells), GridError::kCoordOutOfRange, ToCell(0, kMaxCoord + 1, 0, 0));
}

// The 7 size-level-1 cells of size level 0 {0,0,0} without eighth 5, size level 1 {1,0,1}: the check that cell
// resize skips, which a count needs.
TEST(ComputeGridCounts, LoneSmallCell_IsSiblingCellMissing) {
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 5);
  CheckFailure(
      ComputeCountsOfList(ToUnitFrame(3), cells),
      GridError::kSiblingCellMissing,
      ToCell(1, 1, 0, 1),
      ToCell(0, 0, 0, 0)
  );
}

// OneSplitCell with its size-level-1 cell {2,0,0} split again: the size-level-2 cells at x = 4 touch the size-level-0
// cell {0,0,0}. The first in the fixed order is size level 2 {4,0,0}.
TEST(ComputeGridCounts, CellsTwoSizeLevelsApart_IsSizeStepRuleBroken) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (const CellId cell : ToSmallCells(ToCell(0, 1, 0, 0))) {
    if (cell.value == ToCell(1, 2, 0, 0).value) {
      const std::vector<CellId> small_cells = ToSmallCells(cell);
      cells.insert(cells.end(), small_cells.begin(), small_cells.end());
    } else {
      cells.push_back(cell);
    }
  }
  CheckFailure(
      ComputeCountsOfList(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 0, 0),
      ToCell(0, 0, 0, 0)
  );
}

TEST(ComputeGridCounts, VersionZero_IsBadInput) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  inputs.version = {.value = 0};
  CheckFailure(ComputeGridCounts(inputs), GridError::kBadInput);
}

}  // namespace
}  // namespace fluidity::grid
