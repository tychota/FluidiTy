#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/grid_test_support.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

// A frame with its origin at 0 and biggest cells 1 m wide: every cell side in it is a double with few significant
// bits, so a position on a side is exactly on it.
constexpr GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The inputs of a CPU list of version 1.
GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// The id of the cell at a place FindCellAt gave, or CellId{0} for kNoIndex. A found place must be kActive.
CellId ToFoundCellId(const GridTables& tables, CellPlaceIndex cell) {
  if (cell.value == kNoIndex) {
    return CellId{0};
  }
  EXPECT_EQ(tables.cell_states[cell.value], CellState::kActive);
  return ToCellId(tables, cell);
}

// The near values every hand-built test tries: each block of the grid, kNoIndex, block_count and a large index that is
// not kNoIndex.
std::vector<CellBlockIndex> BuildEveryNearValue(const GridTables& tables) {
  std::vector<CellBlockIndex> near_values;
  near_values.reserve(tables.block_count + 3);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    near_values.push_back({.value = block});
  }
  near_values.push_back({.value = kNoIndex});
  near_values.push_back({.value = tables.block_count});
  near_values.push_back({.value = kNoIndex - 1});
  return near_values;
}

// The id of the active cell that holds `position` by the search form, or CellId{0}, after checking that the near
// form gives the same place for every value of BuildEveryNearValue.
CellId FindCellIdAt(const GridTables& tables, Vec3 position) {
  const CellPlaceIndex searched = FindCellAt(tables, position);
  for (const CellBlockIndex near : BuildEveryNearValue(tables)) {
    EXPECT_EQ(FindCellAt(tables, position, near).value, searched.value) << "near " << near.value;
  }
  return ToFoundCellId(tables, searched);
}

// Expects kNoIndex at `position` from the search form and from the near form with `near` 0, 1 and kNoIndex.
void ExpectNoCellThroughEitherForm(const GridTables& tables, Vec3 position) {
  EXPECT_EQ(FindCellAt(tables, position).value, kNoIndex);
  EXPECT_EQ(FindCellAt(tables, position, CellBlockIndex{0}).value, kNoIndex);
  EXPECT_EQ(FindCellAt(tables, position, CellBlockIndex{1}).value, kNoIndex);
  EXPECT_EQ(FindCellAt(tables, position, CellBlockIndex{kNoIndex}).value, kNoIndex);
}

// Expects kNoIndex through either form when `value` is put on each axis in turn, the other two axes at 0.5 m, inside
// the big cell of the two-size fixture. Each axis has its own guard, so each is tried.
void ExpectNoCellWithValueOnEachAxis(const GridTables& tables, double value) {
  const std::array<Vec3, 3> positions = {
      Vec3{.x = value, .y = 0.5, .z = 0.5},
      Vec3{.x = 0.5, .y = value, .z = 0.5},
      Vec3{.x = 0.5, .y = 0.5, .z = value},
  };
  for (std::size_t axis = 0; axis < positions.size(); ++axis) {
    SCOPED_TRACE(testing::Message() << "value " << value << " on axis " << axis);
    ExpectNoCellThroughEitherForm(tables, positions.at(axis));
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
    ExpectNoCellWithValueOnEachAxis(tables, bad_value);
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
    ExpectNoCellWithValueOnEachAxis(tables, beyond_value);
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

// Size-level-0 cells over 4 x 2 x 2 size-level-0 blocks, block coordinates (0..3, 0..1, 0..1), without block
// (1, 1, 0), and one cell split, so the near form crosses blocks: one or two moves away, three moves away, and past the
// block that does not exist. 15 size-level-0 blocks and 1 size-level-1 block, in block (2, 0, 0).
std::vector<CellId> BuildCellsOverSeveralBlocks() {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 8; ++z) {
    for (std::int32_t y = 0; y < 8; ++y) {
      for (std::int32_t x = 0; x < 16; ++x) {
        const bool is_in_missing_block = x >= 4 && x < 8 && y >= 4 && z < 4;
        if (is_in_missing_block) {
          continue;
        }
        if (x == 9 && y == 2 && z == 1) {
          for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
            cells.push_back(ComputeSmallCell(ToCell(0, x, y, z), cell_eighth));
          }
          continue;
        }
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  return cells;
}

// Expects FindCellIdAt, both forms, to give the checker's cell at `position`, and says whether a cell holds it.
bool ExpectCheckerCellAt(const GridTables& tables, const GeometricGrid& checker, Vec3 position) {
  SCOPED_TRACE(testing::Message() << "position " << position.x << ", " << position.y << ", " << position.z);
  const std::optional<CellId> checker_cell = checker.ContainingCell(position);
  const CellId found_cell = FindCellIdAt(tables, position);
  EXPECT_EQ(found_cell.value, checker_cell.has_value() ? checker_cell->value : 0U);
  return found_cell.value != 0;
}

// From every block of the grid and from values that are not blocks, the near form finds what the search finds, and the
// search agrees with the geometric checker, on a lattice of quarter metres that holds every cell side.
TEST(FindCellAt, NearFormAcrossSizeLevelZeroBlocks_EqualsSearch) {
  const std::vector<CellId> cells = BuildCellsOverSeveralBlocks();
  const GridFrame frame = ToUnitFrame(2);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  ASSERT_EQ(tables.block_count, 16U);
  const GeometricGrid checker(frame, cells);
  std::size_t found_count = 0;
  // z = 1.25 m holds the split cell, and z = 5.5 m lies above the missing block.
  for (const double position_z : {1.25, 5.5}) {
    for (int y_quarter = -4; y_quarter <= 36; ++y_quarter) {
      for (int x_quarter = -4; x_quarter <= 68; ++x_quarter) {
        const Vec3 position = {.x = x_quarter * 0.25, .y = y_quarter * 0.25, .z = position_z};
        found_count += ExpectCheckerCellAt(tables, checker, position) ? 1 : 0;
      }
    }
  }
  EXPECT_GT(found_count, 0U);
}

// The size-level-0 block coordinate of a block, or of the size-level-0 block above it.
CellCoord ToBiggestBlockCoord(const GridTables& tables, CellBlockIndex block) {
  std::uint32_t biggest_block = block.value;
  while (tables.bigger_block[biggest_block] != kNoIndex) {
    biggest_block = tables.bigger_block[biggest_block];
  }
  return ToCellCoord(CellId{tables.block_ids[biggest_block].value});
}

// What CountMoves counted over every pair of a block and a size-level-0 target block.
struct MoveCounts {
  std::size_t arrived_count = 0;  // Pairs whose moves arrived at the target.
  std::size_t wrong_count = 0;  // Pairs where FindBiggestBlockFrom differed from the expected block.
  std::uint32_t first_wrong_near = kNoIndex;
  std::uint32_t first_wrong_target = kNoIndex;
};

// Runs FindBiggestBlockFrom from every block toward every size-level-0 block of the grid of
// BuildCellsOverSeveralBlocks. The expected block is the target, or kNoIndex three blocks away and for a pair of
// `blocked_moves`, each the near block's size-level-0 coordinate (x, y, z), then the target's.
MoveCounts CountMoves(const GridTables& tables, const std::set<std::array<std::int32_t, 6>>& blocked_moves) {
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
  const MoveCounts counts = CountMoves(tables, blocked_moves);
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
std::size_t ExpectCellsAroundSide(const GridTables& tables, const GeometricGrid& checker, int side) {
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
    if (!ExpectCheckerCellAt(tables, checker, position)) {
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
  EXPECT_EQ(ExpectCellsAroundSide(tables, checker, 2), 7U);
  EXPECT_EQ(ExpectCellsAroundSide(tables, checker, 3), 7U);
  EXPECT_GT(ExpectCellsAroundSide(tables, checker, 4), 0U);
}

// The block of the same size level as `block` whose block coordinate is two away on some axis and at most two on
// every axis, or kNoIndex when the grid has none.
CellBlockIndex FindBlockTwoMovesAway(const GridTables& tables, CellBlockIndex block) {
  const CellCoord block_coord = ToCellCoord(CellId{tables.block_ids[block.value].value});
  for (std::uint32_t other = 0; other < tables.block_count; ++other) {
    if (ToSizeLevel(tables, CellBlockIndex{other}) != ToSizeLevel(tables, block)) {
      continue;
    }
    const CellCoord other_coord = ToCellCoord(CellId{tables.block_ids[other].value});
    const std::int32_t distance_x = std::abs(other_coord.x - block_coord.x);
    const std::int32_t distance_y = std::abs(other_coord.y - block_coord.y);
    const std::int32_t distance_z = std::abs(other_coord.z - block_coord.z);
    const std::int32_t largest_distance = std::max({distance_x, distance_y, distance_z});
    if (largest_distance == 2) {
      return {.value = other};
    }
  }
  return {.value = kNoIndex};
}

// What the random-position check counted over one grid.
struct RandomPositionCounts {
  std::size_t found_count = 0;  // Positions in an active cell.
  std::size_t two_moves_count = 0;  // Positions tried from a block two moves away.
  std::size_t wrong_count = 0;  // Positions where some form or the checker disagreed.
};

// For 1,000 positions of one random grid in its box widened by one size-level-0 cell, half drawn freely and half on
// the lattice of smallest cells, so on cell sides: the search agrees with the checker, and the near form agrees with
// the search from the found cell's block, a block two moves from it, the last block, a drawn block, block_count and
// kNoIndex.
RandomPositionCounts CountRandomPositions(const GridTables& tables, const GeometricGrid& checker, std::uint64_t seed) {
  RandomPositionCounts counts;
  std::mt19937_64 generator(seed);
  constexpr int kPositionCount = 1000;
  for (int position_index = 0; position_index < kPositionCount; ++position_index) {
    std::array<double, 3> coordinates{};
    for (double& coordinate : coordinates) {
      const double unit_draw = ToUnitDouble(generator());
      // The box 0..3 m widened by 1 m on each side, 5 m wide; the lattice is 40 smallest cells of 1/8 m.
      coordinate = position_index % 2 == 0 ? -1.0 + (5.0 * unit_draw) : -1.0 + (std::floor(40.0 * unit_draw) / 8.0);
    }
    const Vec3 position = {.x = coordinates.at(0), .y = coordinates.at(1), .z = coordinates.at(2)};
    const CellPlaceIndex searched = FindCellAt(tables, position);
    const CellId found_cell = searched.value == kNoIndex ? CellId{0} : ToCellId(tables, searched);
    const std::optional<CellId> checker_cell = checker.ContainingCell(position);
    bool is_right = found_cell.value == (checker_cell.has_value() ? checker_cell->value : 0U);
    is_right = is_right && (searched.value == kNoIndex || tables.cell_states[searched.value] == CellState::kActive);
    const CellBlockIndex found_block =
        searched.value == kNoIndex ? CellBlockIndex{0} : ToCellBlockIndex(CellPlaceIndex{searched.value});
    const CellBlockIndex two_moves_away = FindBlockTwoMovesAway(tables, found_block);
    const std::vector<CellBlockIndex> near_values = {
        found_block,
        two_moves_away,
        {.value = tables.block_count - 1},
        {.value = static_cast<std::uint32_t>(ToUniformIndex(generator(), tables.block_count))},
        {.value = tables.block_count},
        {.value = kNoIndex},
    };
    for (const CellBlockIndex near : near_values) {
      is_right = is_right && FindCellAt(tables, position, near).value == searched.value;
    }
    counts.found_count += found_cell.value != 0 ? 1 : 0;
    counts.two_moves_count += two_moves_away.value != kNoIndex ? 1 : 0;
    counts.wrong_count += is_right ? 0 : 1;
  }
  return counts;
}

// The 200 random grids: every near form equals the search form, which equals GeometricGrid::ContainingCell. The one
// test of this program over the random grids, so they are built once.
TEST(FindCellAt, NearForm_EqualsSearchOnRandomPositions) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kRandomGridCount);
  RandomPositionCounts total;
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(testing::Message() << "FindCellAt on random positions, random grid " << grid);
    const RandomPositionCounts counts =
        CountRandomPositions(built.grids.at(grid).grid.tables(RunOn::kCpu), built.checkers.at(grid), grid + 1);
    EXPECT_EQ(counts.wrong_count, 0U);
    total.found_count += counts.found_count;
    total.two_moves_count += counts.two_moves_count;
  }
  // Positions in cells and positions tried from two moves away both occur, so both were compared.
  EXPECT_GT(total.found_count, 0U);
  EXPECT_GT(total.two_moves_count, 0U);
}

}  // namespace
}  // namespace fluidity::grid
