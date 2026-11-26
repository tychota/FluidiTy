#include "fluidity/grid/find_cell_at.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <limits>
#include <set>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/find_cell_at_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// Expects kNoIndex at `position` from the search form and from the near form with `near` 0, 1 and kNoIndex.
void CheckNoCellThroughEitherForm(const GridTables& tables, Vec3 position) {
  EXPECT_EQ(FindCellAt(tables, position).value, kNoIndex);
  EXPECT_EQ(FindCellAt(tables, position, CellBlockIndex{0}).value, kNoIndex);
  EXPECT_EQ(FindCellAt(tables, position, CellBlockIndex{1}).value, kNoIndex);
  EXPECT_EQ(FindCellAt(tables, position, CellBlockIndex{kNoIndex}).value, kNoIndex);
}

// Expects kNoIndex through either form when `value` is put on each axis in turn, the other two axes at 0.5 m, inside
// the big cell of the two-size fixture. Each axis has its own guard, so each is tried.
void CheckNoCellWithValueOnEachAxis(const GridTables& tables, double value) {
  const std::array<Vec3, 3> positions = {
      Vec3{.x = value, .y = 0.5, .z = 0.5},
      Vec3{.x = 0.5, .y = value, .z = 0.5},
      Vec3{.x = 0.5, .y = 0.5, .z = value},
  };
  for (std::size_t axis = 0; axis < positions.size(); ++axis) {
    SCOPED_TRACE(testing::Message() << "value " << value << " on axis " << axis);
    CheckNoCellThroughEitherForm(tables, positions.at(axis));
  }
}

// A cell owns its low side: x = 1 m is the low side of the small cell {2,0,0}, not the high side of the big cell.
TEST(FindCellAt, PositionOnLowSide_BelongsToCell) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  EXPECT_EQ(FindCellIdAt(tables, {.x = 1.0, .y = 0.25, .z = 0.25}).value, ToCell(1, 2, 0, 0).value);
}

// One rounding step below x = 1 m is still the big cell; {1, 0.5, 0.5} m, on a corner of four small cells, is in the
// small cell above it on every axis; x = 1.5 m, the side between two small cells, is in the higher one.
TEST(FindCellAt, OnHighSide_BelongsToNext) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  EXPECT_EQ(FindCellIdAt(tables, {.x = std::nextafter(1.0, 0.0), .y = 0.5, .z = 0.5}).value, ToCell(0, 0, 0, 0).value);
  EXPECT_EQ(FindCellIdAt(tables, {.x = 1.0, .y = 0.5, .z = 0.5}).value, ToCell(1, 2, 1, 1).value);
  EXPECT_EQ(FindCellIdAt(tables, {.x = 1.5, .y = 0.25, .z = 0.25}).value, ToCell(1, 3, 0, 0).value);
}

// In a frame that is no power of two the computed low corner of a cell, origin + c width, rounds, and the grid's own
// rule f = floor((x - origin) / width) may give that point to the cell before. A position belongs to the cell that
// FindCellAt returns. Origin 0.3 m on x, width 0.7 m: for the cell at coordinate 2 the corner computed as the centre
// minus half a width is just below the side, so it is in the cell at coordinate 1; the next double up is in cell 2.
TEST(FindCellAt, ComputedLowCornerInFrameOfNoPowerOfTwo_BelongsToTheCellBefore) {
  const GridFrame frame = {
      .origin = {.x = 0.3, .y = 0.1, .z = -0.7},
      .biggest_cell_width = 0.7,
      .size_level_count = 1,
  };
  const std::vector<CellId> cells = {ToCell(0, 1, 0, 0), ToCell(0, 2, 0, 0)};
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const CellPlaceIndex cell_place = FindCellPlace(tables, ToCell(0, 2, 0, 0));
  ASSERT_NE(cell_place.value, kNoIndex);
  const Vec3 center = ComputeCellCenter(tables, cell_place);
  const double low_corner_x = center.x - (0.5 * frame.biggest_cell_width);
  const double next_double_up_x = std::nextafter(low_corner_x, std::numeric_limits<double>::infinity());
  EXPECT_EQ(FindCellIdAt(tables, {.x = low_corner_x, .y = center.y, .z = center.z}).value, ToCell(0, 1, 0, 0).value);
  EXPECT_EQ(
      FindCellIdAt(tables, {.x = next_double_up_x, .y = center.y, .z = center.z}).value,
      ToCell(0, 2, 0, 0).value
  );
}

// Below the big cell and above the small cells no cell exists.
TEST(FindCellAt, OutsideEveryCell_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  EXPECT_EQ(FindCellAt(tables, {.x = -0.5, .y = 0.5, .z = 0.5}).value, kNoIndex);
  EXPECT_EQ(FindCellAt(tables, {.x = 2.0, .y = 0.5, .z = 0.5}).value, kNoIndex);
  EXPECT_EQ(FindCellIdAt(tables, {.x = -0.5, .y = 0.5, .z = 0.5}).value, 0U);
  EXPECT_EQ(FindCellIdAt(tables, {.x = 2.0, .y = 0.5, .z = 0.5}).value, 0U);
}

// A position too large for any coordinate, or not a number, on any axis, is in no cell, through either form, before any
// integer conversion.
TEST(FindCellAt, HugeOrNotFinitePosition_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const std::vector<double> bad_values = {
      1e300,
      -1e300,
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
  };
  for (const double bad_value : bad_values) {
    CheckNoCellWithValueOnEachAxis(tables, bad_value);
  }
}

// The guard is the range of size-level-0 ids, on each axis: 262144 m is size-level-0 coordinate kMaxCoord + 1 and
// -262144 m is -kMaxCoord - 1, both outside. 524288.25 m is size-level-0 coordinate 2^19, whose packed id is the id of
// size level 0 {0,0,0} on that axis: without the guard of that axis the big cell would be found. x = 262143.5 m is in
// the last size-level-0 cell, which holds no active cell.
TEST(FindCellAt, PositionBeyondTheIdRange_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  constexpr std::int32_t kWrappingCoord = 524288;  // 2^19: kept as 2^19 + kMaxCoord in 19 bits, it wraps to kMaxCoord.
  ASSERT_EQ(ToCell(0, kWrappingCoord, 0, 0).value, ToCell(0, 0, 0, 0).value);
  for (const double beyond_value : {262144.0, -262144.0, 524288.25}) {
    CheckNoCellWithValueOnEachAxis(tables, beyond_value);
  }
  EXPECT_EQ(FindCellAt(tables, {.x = 262143.5, .y = 0.5, .z = 0.5}).value, kNoIndex);
}

// For every size level count, a lone size-level-0 cell at each end of the coordinate range is found up to its outer
// side, and one rounding step beyond that side is in no cell: the guard's bounds, -kMaxCoord 2^(L - 1) and
// (kMaxCoord + 1) 2^(L - 1) smallest cells, cut no valid cell off for L = 1..16.
TEST(FindCellAt, EachSizeLevelCount_FindsTheCellsAtTheEndsOfTheIdRange) {
  const std::vector<CellId> cells = {ToCell(0, -kMaxCoord, 0, 0), ToCell(0, kMaxCoord, 0, 0)};
  const double last_side = kMaxCoord + 1.0;  // m: the high side of the last size-level-0 cell.
  const double first_side = -static_cast<double>(kMaxCoord);  // m: the low side of the first one.
  const double below_last_side = std::nextafter(last_side, 0.0);
  const double below_first_side = std::nextafter(first_side, -1e6);
  for (int size_level_count = 1; size_level_count <= kMaxSizeLevels; ++size_level_count) {
    SCOPED_TRACE(testing::Message() << "size levels " << size_level_count);
    const std::expected<TestGrid, GridFailure> test_grid =
        BuildTestGrid(ToInputs(ToUnitFrame(size_level_count), cells));
    ASSERT_TRUE(test_grid.has_value());
    const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
    const std::array<CellId, 4> found_cells = {
        FindCellIdAt(tables, {.x = below_last_side, .y = 0.5, .z = 0.5}),
        FindCellIdAt(tables, {.x = last_side, .y = 0.5, .z = 0.5}),
        FindCellIdAt(tables, {.x = first_side, .y = 0.5, .z = 0.5}),
        FindCellIdAt(tables, {.x = below_first_side, .y = 0.5, .z = 0.5}),
    };
    const std::array<std::uint64_t, 4> expected_cells = {cells.at(1).value, 0, cells.at(0).value, 0};
    for (std::size_t index = 0; index < found_cells.size(); ++index) {
      EXPECT_EQ(found_cells.at(index).value, expected_cells.at(index)) << "position " << index;
    }
  }
}

// With origin 0 and biggest cells 4 m wide, nothing rounds but a quotient below the normal range. x = -2^-1074 m has
// the quotient -2^-1076, which rounds to -0: the position lies in cell 0, though it is below the side x = 0 of cell -1.
// x = -2^-1020 m, whose quotient -2^-1022 is normal, lies in cell -1.
TEST(FindCellAt, NegativeQuotientBelowNormalRange_LiesInCellZero) {
  const std::vector<CellId> cells = {ToCell(0, -1, 0, 0), ToCell(0, 0, 0, 0)};
  GridFrame frame = ToUnitFrame(1);
  frame.biggest_cell_width = 4.0;
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const double smallest_subnormal = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(FindCellIdAt(tables, {.x = -smallest_subnormal, .y = 0.5, .z = 0.5}).value, ToCell(0, 0, 0, 0).value);
  EXPECT_EQ(FindCellIdAt(tables, {.x = -0x1p-1020, .y = 0.5, .z = 0.5}).value, ToCell(0, -1, 0, 0).value);
}

// The size-level-0 block coordinate of a block, or of the size-level-0 block above it.
CellCoord ToBiggestBlockCoord(const GridTables& tables, CellBlockIndex block) {
  std::uint32_t biggest_block = block.value;
  while (tables.bigger_block[biggest_block] != kNoIndex) {
    biggest_block = tables.bigger_block[biggest_block];
  }
  return ToCellCoord(CellId{tables.block_ids[biggest_block].value});
}

// What ComputeMoveCounts counted over every pair of a block and a size-level-0 target block.
struct MoveCounts {
  std::size_t arrived_count = 0;  // Pairs whose moves arrived at the target.
  std::size_t wrong_count = 0;  // Pairs where FindBiggestBlockFrom differed from the expected block.
  std::uint32_t first_wrong_near = kNoIndex;
  std::uint32_t first_wrong_target = kNoIndex;
};

// Runs FindBiggestBlockFrom from every block toward every size-level-0 block of the grid of
// BuildCellsOverSeveralBlocks. The expected block is the target, or kNoIndex three blocks away and for a pair of
// `blocked_moves`, each the near block's size-level-0 coordinate (x, y, z), then the target's.
MoveCounts ComputeMoveCounts(const GridTables& tables, const std::set<std::array<std::int32_t, 6>>& blocked_moves) {
  MoveCounts counts;
  for (std::uint32_t near = 0; near < tables.block_count; ++near) {
    const CellCoord near_coord = ToBiggestBlockCoord(tables, CellBlockIndex{near});
    for (std::uint32_t target = 0; target < tables.block_count; ++target) {
      if (ToSizeLevel(tables, CellBlockIndex{target}) != SizeLevel{0}) {
        continue;
      }
      const CellBlockId target_id = tables.block_ids[target];
      const CellCoord target_coord = ToCellCoord(CellId{target_id.value});
      const bool is_three_away = std::abs(target_coord.x - near_coord.x) == 3;
      const bool is_blocked = blocked_moves.contains(
          {near_coord.x, near_coord.y, near_coord.z, target_coord.x, target_coord.y, target_coord.z}
      );
      const std::uint32_t expected_block = is_three_away || is_blocked ? kNoIndex : target;
      const std::uint32_t found_block = FindBiggestBlockFrom(tables, CellBlockIndex{near}, target_id);
      counts.arrived_count += found_block != kNoIndex ? 1 : 0;
      if (found_block == expected_block) {
        continue;
      }
      if (counts.wrong_count == 0) {
        counts.first_wrong_near = near;
        counts.first_wrong_target = target;
      }
      ++counts.wrong_count;
    }
  }
  return counts;
}

// The moves of the near form arrive at a size-level-0 block within two moves when the blocks on the way exist, each
// move by -1, 0 or +1 toward the target on each axis, from a size-level-0 block or from the size-level-1 block below
// one. They give kNoIndex, and so leave the answer to the search, three blocks away and where the first move lands on
// the missing block (1, 1, 0): toward (2, 1, 0) from x = 0 and toward (0, 1, 0) from x = 2, from any y and z.
TEST(FindBiggestBlockFrom, TargetWithinTwoMoves_IsFoundWithoutSearch) {
  const std::vector<CellId> cells = BuildCellsOverSeveralBlocks();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  // The near block's size-level-0 coordinate (x, y, z), then the target's.
  const std::set<std::array<std::int32_t, 6>> blocked_moves = {
      {0, 0, 0, 2, 1, 0},
      {0, 0, 1, 2, 1, 0},
      {0, 1, 0, 2, 1, 0},
      {0, 1, 1, 2, 1, 0},
      {2, 0, 0, 0, 1, 0},
      {2, 0, 1, 0, 1, 0},
      {2, 1, 0, 0, 1, 0},
      {2, 1, 1, 0, 1, 0},
  };
  const MoveCounts counts = ComputeMoveCounts(tables, blocked_moves);
  EXPECT_EQ(counts.wrong_count, 0U) << "first wrong: near " << counts.first_wrong_near << ", target "
                                    << counts.first_wrong_target;
  EXPECT_GT(counts.arrived_count, 0U);
}

// The frame of the test below: origin and width are not dyadic, so a cell side origin + c w_min is a rounded double.
constexpr GridFrame kFrameNotDyadic = {
    .origin = {.x = -17.616723516683763, .y = 0.1, .z = 0.3},
    .biggest_cell_width = 0.3,
    .size_level_count = 2,
};

// Expects, from three rounding steps below the side at `side` smallest cells to three above, that FindCellIdAt (both
// forms) gives the checker's cell, a cell beside the side or none beyond the grid's edge, its low x never decreasing
// as x grows. Gives the number of positions found in a cell.
std::size_t CheckCellsAroundSide(const GridTables& tables, const GeometricGrid& checker, int side) {
  const double smallest_cell_width = ComputeCellWidth(kFrameNotDyadic, SizeLevel{1});
  double position_x = kFrameNotDyadic.origin.x + (side * smallest_cell_width);
  for (int step = 0; step < 3; ++step) {
    position_x = std::nextafter(position_x, -1e6);
  }
  std::size_t found_count = 0;
  std::int64_t previous_low_x = std::numeric_limits<std::int64_t>::min();
  for (int step = -3; step <= 3; ++step) {
    SCOPED_TRACE(testing::Message() << "side " << side << ", step " << step);
    const Vec3 position = {.x = position_x, .y = kFrameNotDyadic.origin.y + 0.1, .z = kFrameNotDyadic.origin.z + 0.1};
    position_x = std::nextafter(position_x, 1e6);
    if (!CheckCheckerCellAt(tables, checker, position)) {
      continue;
    }
    const UnitBox box = ToUnitBox(FindCellIdAt(tables, position), kFrameNotDyadic.size_level_count);
    const std::int64_t low_x = box.low.at(0);
    EXPECT_TRUE(low_x == side || low_x + box.width == side);
    EXPECT_GE(low_x, previous_low_x);
    previous_low_x = low_x;
    ++found_count;
  }
  return found_count;
}

// Around each side of a frame that is not dyadic, both forms find the same cell, on one side of it or the other, never
// moving back as x grows. The sides at x = 2, 3 and 4 smallest cells: big cell to small cell, between small cells, and
// the grid's edge, where only the positions below are in a cell.
TEST(FindCellAt, FrameNotDyadic_BothFormsAgreeNearAnEdge) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(kFrameNotDyadic, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const GeometricGrid checker(kFrameNotDyadic, cells);
  EXPECT_EQ(CheckCellsAroundSide(tables, checker, 2), 7U);
  EXPECT_EQ(CheckCellsAroundSide(tables, checker, 3), 7U);
  EXPECT_GT(CheckCellsAroundSide(tables, checker, 4), 0U);
}

}  // namespace
}  // namespace fluidity::grid
