#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/find_cell_at_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

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
