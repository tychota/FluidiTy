#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// A frame with its origin at 0 and biggest cells 1 m wide.
GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

int ToSizeLevelNumber(CellId cell) {
  return static_cast<int>(ToSizeLevel(cell));
}

// The inputs of a CPU list of version 1.
GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

std::expected<GridCounts, GridFailure> ComputeCountsOfList(GridFrame frame, std::span<const CellId> active_cells) {
  return ComputeGridCounts(ToInputs(frame, active_cells));
}

// floor(coord / 2) by integer division and a correction for a negative remainder: the test's own arithmetic, apart
// from the shift the grid uses.
std::int32_t ToFloorOfHalf(std::int32_t coord) {
  const std::int32_t quotient = coord / 2;
  return (coord % 2 != 0 && coord < 0) ? quotient - 1 : quotient;
}

// The big cell, one size level up, that holds `cell`, by the test's own floor.
CellId ToBigCellByDivision(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  return ToCell(ToSizeLevelNumber(cell) - 1, ToFloorOfHalf(coord.x), ToFloorOfHalf(coord.y), ToFloorOfHalf(coord.z));
}

// The 8 small cells of a big cell, by the test's own arithmetic: 2 c and 2 c + 1 on each axis, in eighth order.
std::vector<CellId> ToSmallCells(CellId big_cell) {
  const CellCoord coord = ToCellCoord(big_cell);
  const int small_size_level = ToSizeLevelNumber(big_cell) + 1;
  std::vector<CellId> small_cells;
  for (std::int32_t dz = 0; dz < 2; ++dz) {
    for (std::int32_t dy = 0; dy < 2; ++dy) {
      for (std::int32_t dx = 0; dx < 2; ++dx) {
        small_cells.push_back(ToCell(small_size_level, (2 * coord.x) + dx, (2 * coord.y) + dy, (2 * coord.z) + dz));
      }
    }
  }
  return small_cells;
}

// The cell blocks of a valid list by the rule of <sec-cell-blocks>, apart from the grid's split cells and sort: the
// distinct ToCellBlockId of every active cell and of every bigger cell that holds one.
std::size_t ComputeCheckerBlockCount(std::span<const CellId> active_cells) {
  std::set<std::uint64_t> block_ids;
  for (const CellId active_cell : active_cells) {
    CellId cell = active_cell;
    block_ids.insert(ToCellBlockId(cell).value);
    while (ToSizeLevelNumber(cell) > 0) {
      cell = ToBigCellByDivision(cell);
      block_ids.insert(ToCellBlockId(cell).value);
    }
  }
  return block_ids.size();
}

// The active cells of each size level of a list, counted from the list itself.
std::array<std::uint64_t, kMaxSizeLevels> ComputeActiveCellsPerSizeLevel(std::span<const CellId> active_cells) {
  std::array<std::uint64_t, kMaxSizeLevels> per_size_level{};
  for (const CellId cell : active_cells) {
    ++per_size_level.at(static_cast<std::size_t>(ToSizeLevelNumber(cell)));
  }
  return per_size_level;
}

// Every field of counts, as one comparable value.
std::tuple<std::uint32_t, std::array<std::uint64_t, kMaxSizeLevels>, std::uint64_t, std::uint64_t> ToCountFields(
    const GridCounts& counts
) {
  std::array<std::uint64_t, kMaxSizeLevels> per_size_level{};
  for (std::size_t size_level = 0; size_level < per_size_level.size(); ++size_level) {
    per_size_level.at(size_level) = counts.active_cells_per_size_level[size_level];
  }
  return {counts.cell_blocks, per_size_level, counts.active_cells, counts.face_pieces};
}

// Expects the counts the checker gives for a valid list: its face pieces, the blocks of every active cell and the
// bigger cells holding it, and the active cells counted from the list.
void ExpectCheckerCounts(
    const std::expected<GridCounts, GridFailure>& counts,
    GridFrame frame,
    const std::vector<CellId>& active_cells
) {
  ASSERT_TRUE(counts.has_value());
  const GeometricGrid checker(frame, active_cells);
  EXPECT_EQ(counts->face_pieces, checker.FacePieces().size());
  EXPECT_EQ(counts->cell_blocks, ComputeCheckerBlockCount(active_cells));
  EXPECT_EQ(counts->active_cells, active_cells.size());
  EXPECT_EQ(std::get<1>(ToCountFields(*counts)), ComputeActiveCellsPerSizeLevel(active_cells));
}

void ExpectFailure(
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
  ExpectCheckerCounts(counts, ToUnitFrame(2), cells);
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
  ExpectCheckerCounts(counts, ToUnitFrame(2), cells);
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
  ExpectCheckerCounts(counts, ToUnitFrame(16), cells);
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

// ---------------------------------------------------------------------------------------------------------------------
// The checks of a build
// ---------------------------------------------------------------------------------------------------------------------

TEST(ComputeGridCounts, CellInsideActiveBigCell_IsCellsOverlap) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(1, 1, 1, 1)};
  ExpectFailure(
      ComputeCountsOfList(ToUnitFrame(3), cells),
      GridError::kCellsOverlap,
      ToCell(1, 1, 1, 1),
      ToCell(0, 0, 0, 0)
  );
}

TEST(ComputeGridCounts, SameCellTwice_IsCellsOverlap) {
  const std::vector<CellId> cells = {ToCell(0, 2, 0, 0), ToCell(0, 2, 0, 0)};
  ExpectFailure(
      ComputeCountsOfList(ToUnitFrame(3), cells),
      GridError::kCellsOverlap,
      ToCell(0, 2, 0, 0),
      ToCell(0, 2, 0, 0)
  );
}

TEST(ComputeGridCounts, CoordBeyondLimit_IsCoordOutOfRange) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(0, kMaxCoord + 1, 0, 0)};
  ExpectFailure(
      ComputeCountsOfList(ToUnitFrame(3), cells),
      GridError::kCoordOutOfRange,
      ToCell(0, kMaxCoord + 1, 0, 0)
  );
}

// The 7 size-level-1 cells of size level 0 {0,0,0} without eighth 5, size level 1 {1,0,1}: the check that cell
// resize skips, which a count needs.
TEST(ComputeGridCounts, LoneSmallCell_IsSiblingCellMissing) {
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 5);
  ExpectFailure(
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
  ExpectFailure(
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
  ExpectFailure(ComputeGridCounts(inputs), GridError::kBadInput);
}

}  // namespace
}  // namespace fluidity::grid
