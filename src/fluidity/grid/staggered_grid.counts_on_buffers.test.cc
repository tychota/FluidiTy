#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_counts_on_buffers_cases.h"

namespace fluidity::grid {
namespace {

// 4,097 cells 4 apart along x, each alone in its block, into buffers for 4,096 blocks: the blocks are counted against
// the room and none is named. The one-argument form has no room and counts them all.
TEST(ComputeGridCounts, MoreBlocksThanTheRoom_IsTooManyCellBlocksOnBuffers) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(4097);
  const GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 4096});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::expected<GridCounts, GridFailure> buffers_counts = ComputeGridCounts(inputs, buffers);
  ASSERT_FALSE(buffers_counts.has_value());
  EXPECT_EQ(buffers_counts.error().error, GridError::kTooManyCellBlocks);
  EXPECT_EQ(buffers_counts.error().cell.value, 0U);
  EXPECT_EQ(buffers_counts.error().other_cell.value, 0U);

  const std::expected<GridCounts, GridFailure> span_counts = ComputeGridCounts(inputs);
  ASSERT_TRUE(span_counts.has_value());
  EXPECT_EQ(span_counts->cell_blocks, 4097U);
}

// Lists with their own corners of the count: an empty list, one block, both coordinate limits on the x axis and a chain
// down to size level 15, each with the frame it needs.
struct Fixture {
  GridFrame frame;
  std::vector<CellId> cells;
};

std::vector<CellId> BuildSmallCellsOfCell(CellId big_cell) {
  const CellCoord coord = ToCellCoord(big_cell);
  const auto small_size_level = static_cast<SizeLevel>(static_cast<int>(ToSizeLevel(big_cell)) + 1);
  std::vector<CellId> small_cells;
  for (std::int32_t dz = 0; dz < 2; ++dz) {
    for (std::int32_t dy = 0; dy < 2; ++dy) {
      for (std::int32_t dx = 0; dx < 2; ++dx) {
        small_cells.push_back(
            ToCellId(small_size_level, {.x = (2 * coord.x) + dx, .y = (2 * coord.y) + dy, .z = (2 * coord.z) + dz})
        );
      }
    }
  }
  return small_cells;
}

// Size level 0 {3,3,3} split down its eighth 7 to size level 15: at each size level 1..14 the 7 small cells other than
// eighth 7, at size level 15 all 8, and one far cell of size level 0.
std::vector<CellId> BuildChainToSizeLevelFifteen() {
  std::vector<CellId> cells;
  CellId cell = ToCellId(SizeLevel{0}, {.x = 3, .y = 3, .z = 3});
  for (int size_level = 0; size_level < 15; ++size_level) {
    const std::vector<CellId> small_cells = BuildSmallCellsOfCell(cell);
    for (std::size_t cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      if (cell_eighth != 7 || size_level == 14) {
        cells.push_back(small_cells.at(cell_eighth));
      }
    }
    cell = small_cells.at(7);
  }
  cells.push_back(ToCellId(SizeLevel{0}, {.x = 262139, .y = 262140, .z = 262140}));
  return cells;
}

std::vector<Fixture> BuildFixtures() {
  std::vector<CellId> limit_cells = {
      ToCellId(SizeLevel{0}, {.x = -kMaxCoord, .y = 0, .z = 0}),
      ToCellId(SizeLevel{0}, {.x = kMaxCoord, .y = 0, .z = 0}),
  };
  const std::vector<CellId> small_cells = BuildSmallCellsOfCell(ToCellId(SizeLevel{0}, {.x = 0, .y = 0, .z = 0}));
  limit_cells.insert(limit_cells.end(), small_cells.begin(), small_cells.end());
  return {
      {.frame = ToUnitFrame(3), .cells = {}},
      {.frame = ToUnitFrame(3), .cells = {ToCellId(SizeLevel{0}, {.x = 0, .y = 0, .z = 0})}},
      {.frame = ToUnitFrame(2), .cells = limit_cells},
      {.frame = ToUnitFrame(16), .cells = BuildChainToSizeLevelFifteen()},
  };
}

TEST(ComputeGridCounts, IntoBuffers_FixturesEqualSpanForm) {
  const std::vector<Fixture> fixtures = BuildFixtures();
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 64});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  for (std::size_t fixture = 0; fixture < fixtures.size(); ++fixture) {
    SCOPED_TRACE(fixture);
    const GridInputs inputs = ToInputs(fixtures.at(fixture).frame, fixtures.at(fixture).cells);
    const std::expected<GridCounts, GridFailure> span_counts = ComputeGridCounts(inputs);
    const std::expected<GridCounts, GridFailure> buffers_counts = ComputeGridCounts(inputs, buffers);
    ASSERT_TRUE(span_counts.has_value());
    ASSERT_TRUE(buffers_counts.has_value());
    EXPECT_EQ(ToGridCountFields(*buffers_counts), ToGridCountFields(*span_counts));
  }
}

// Buffers that were moved from have no arrays and no loan: the lent builder buffers are checked before anything is
// written, as for BuildGrid and CheckSizeStepRule.
TEST(ComputeGridCounts, MovedFromBuffers_IsBadInput) {
  const std::vector<CellId> cells = {ToCellId(SizeLevel{0}, {.x = 0, .y = 0, .z = 0})};
  const GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  const GridBuffers moved_to = std::move(buffers);

  const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(inputs, buffers);
  ASSERT_FALSE(counts.has_value());
  EXPECT_EQ(counts.error().error, GridError::kBadInput);
  EXPECT_EQ(counts.error().cell.value, 0U);
  EXPECT_EQ(counts.error().other_cell.value, 0U);
}

}  // namespace
}  // namespace fluidity::grid
