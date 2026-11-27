#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/new_cell_rule.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/new_cell_rule_reference.h"

namespace fluidity::grid {
namespace {

// The number of random grids this test reads: the first 16 entries of the random grid list.
constexpr std::size_t kGridCount = 16;

// What the top-down fill of one empty biggest cell visited and produced.
struct CubeFill {
  std::vector<CellId> visited_boxes;  // Every box asked about: the cube and each box made by a split.
  std::vector<CellId> new_cells;  // The boxes that were not split.
  // The box where the fill stopped because it had to split and also touches a bigger committed cell: its small cells
  // would sit beside a committed cell two or more size levels bigger, and leaving it whole breaks the rule on the other
  // side. Empty when the cube can be filled.
  std::optional<CellId> stopped_box;
};

using FindFacts = std::function<NewCellNeighbors(CellId)>;

// Fills one empty biggest cell from the top: asks about the cube, splits a box into its 8 small cells while a committed
// cell two or more size levels smaller touches it, and repeats on the small cells. A cube cannot be filled exactly when
// a box that has to be split (touches_much_smaller_cell) also touches a bigger committed cell (touches_bigger_cell);
// the fill stops at that box. Otherwise the fill succeeds: a new cell beside a committed cell exactly one size level
// bigger obeys the size step rule.
CubeFill BuildCubeFill(CellId cube, const FindFacts& find_facts) {
  CubeFill fill;
  std::vector<CellId> pending = {cube};
  while (!pending.empty()) {
    const CellId box = pending.back();
    pending.pop_back();
    const NewCellNeighbors facts = find_facts(box);
    fill.visited_boxes.push_back(box);
    if (!facts.touches_much_smaller_cell) {
      fill.new_cells.push_back(box);
      continue;
    }
    if (facts.touches_bigger_cell) {
      fill.stopped_box = box;
      return fill;
    }
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      pending.push_back(ComputeSmallCell(box, cell_eighth));
    }
  }
  return fill;
}

// Gives the centre of the biggest cell at `cube_coord` as a world position.
Vec3 ToCubeCentre(const GridFrame& frame, CellCoord cube_coord) {
  const double width = frame.biggest_cell_width;
  return {
      .x = frame.origin.x + (cube_coord.x + 0.5) * width,
      .y = frame.origin.y + (cube_coord.y + 0.5) * width,
      .z = frame.origin.z + (cube_coord.z + 0.5) * width,
  };
}

// Says whether any cell of the grid covers the centre of the biggest cell at `cube_coord`. A biggest cell is empty or
// covered whole, so one point tells.
bool ComputeIsCubeOccupied(const GridTables& tables, CellCoord cube_coord) {
  return FindCellAt(tables, ToCubeCentre(tables.frame, cube_coord)).value != kNoIndex;
}

// Says whether one of the 26 biggest cells around `cube_coord` is occupied.
bool ComputeIsCubeBesideOccupied(const GridTables& tables, CellCoord cube_coord) {
  for (std::int32_t offset_z = -1; offset_z <= 1; ++offset_z) {
    for (std::int32_t offset_y = -1; offset_y <= 1; ++offset_y) {
      for (std::int32_t offset_x = -1; offset_x <= 1; ++offset_x) {
        const bool is_cube_itself = offset_x == 0 && offset_y == 0 && offset_z == 0;
        const CellCoord around = {
            .x = cube_coord.x + offset_x,
            .y = cube_coord.y + offset_y,
            .z = cube_coord.z + offset_z,
        };
        if (!is_cube_itself && ComputeIsCubeOccupied(tables, around)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Gives the coordinate of the biggest cell that holds `cell`.
CellCoord ToBiggestCellCoord(CellId cell) {
  CellId big_cell = cell;
  while (ToSizeLevel(big_cell) != SizeLevel{0}) {
    big_cell = ComputeBigCell(big_cell);
  }
  return ToCellCoord(big_cell);
}

// Finds the empty biggest cells within one biggest cell of an occupied one, as size-level-0 ids, in the box that
// reaches one biggest cell past the cells of `cell_list`.
std::vector<CellId> FindCubesToFill(const GridTables& tables, const std::vector<CellId>& cell_list) {
  CellCoord lowest = {.x = kMaxCoord, .y = kMaxCoord, .z = kMaxCoord};
  CellCoord highest = {.x = -kMaxCoord, .y = -kMaxCoord, .z = -kMaxCoord};
  for (const CellId cell : cell_list) {
    const CellCoord coord = ToBiggestCellCoord(cell);
    lowest = {.x = std::min(lowest.x, coord.x), .y = std::min(lowest.y, coord.y), .z = std::min(lowest.z, coord.z)};
    highest = {.x = std::max(highest.x, coord.x), .y = std::max(highest.y, coord.y), .z = std::max(highest.z, coord.z)};
  }
  std::vector<CellId> cubes;
  for (std::int32_t cube_z = lowest.z - 1; cube_z <= highest.z + 1; ++cube_z) {
    for (std::int32_t cube_y = lowest.y - 1; cube_y <= highest.y + 1; ++cube_y) {
      for (std::int32_t cube_x = lowest.x - 1; cube_x <= highest.x + 1; ++cube_x) {
        const CellCoord cube_coord = {.x = cube_x, .y = cube_y, .z = cube_z};
        if (!ComputeIsCubeOccupied(tables, cube_coord) && ComputeIsCubeBesideOccupied(tables, cube_coord)) {
          cubes.push_back(ToCellId(SizeLevel{0}, cube_coord));
        }
      }
    }
  }
  return cubes;
}

// Checks that the new cells of one fill cover the cube exactly once: their volumes, in smallest cells, add up to the
// cube's.
void CheckNewCellsCoverCube(const CubeFill& fill, CellId cube, int size_level_count) {
  std::int64_t volume_sum = 0;
  for (const CellId new_cell : fill.new_cells) {
    const std::int64_t width = ToSmallestCellBox(new_cell, size_level_count).width;
    volume_sum += width * width * width;
  }
  const std::int64_t cube_width = ToSmallestCellBox(cube, size_level_count).width;
  EXPECT_EQ(volume_sum, cube_width * cube_width * cube_width);
}

// How many cubes one grid had, and how many of them cannot be filled.
struct CubeCounts {
  std::size_t cube_count;
  std::size_t cannot_be_filled_count;
};

// Checks that FindNewCellNeighbors gives the reference's two facts at every box the fill of a cube asked about.
void CheckFunctionMatchesReference(const GridTables& tables, const CubeFill& fill) {
  for (const CellId new_cell : fill.visited_boxes) {
    const NewCellNeighbors expected = ComputeNewCellNeighborsByAllCells(tables, new_cell);
    const NewCellNeighbors actual = FindNewCellNeighbors(tables, new_cell);
    EXPECT_EQ(actual.touches_much_smaller_cell, expected.touches_much_smaller_cell);
    EXPECT_EQ(actual.touches_bigger_cell, expected.touches_bigger_cell);
  }
}

// Finds the active cells of the grid.
std::vector<CellId> FindCommittedCells(const GridTables& tables) {
  std::vector<CellId> committed_cells;
  const std::uint64_t place_count = std::uint64_t{tables.block_count} * kCellPlacesPerBlock;
  for (std::uint64_t place = 0; place < place_count; ++place) {
    if (tables.cell_states[place] == CellState::kActive) {
      committed_cells.push_back(ToCellId(tables, CellPlaceIndex{static_cast<std::uint32_t>(place)}));
    }
  }
  return committed_cells;
}

// Says whether two boxes overlap in volume.
bool ComputeBoxesOverlapInVolume(const SmallestCellBox& first, const SmallestCellBox& second) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_apart = first.low.at(axis) + first.width <= second.low.at(axis)
        || second.low.at(axis) + second.width <= first.low.at(axis);
    if (is_apart) {
      return false;
    }
  }
  return true;
}

// Checks by brute force, with no look-up by place, that every new cell of a fill that succeeded lies beside committed
// cells at most one size level apart and overlaps none of them.
void CheckNewCellsObeySizeStepRule(
    const std::vector<CellId>& committed_cells,
    const CubeFill& fill,
    int size_level_count
) {
  for (const CellId new_cell : fill.new_cells) {
    const SmallestCellBox new_cell_box = ToSmallestCellBox(new_cell, size_level_count);
    for (const CellId committed_cell : committed_cells) {
      const SmallestCellBox committed_box = ToSmallestCellBox(committed_cell, size_level_count);
      EXPECT_FALSE(ComputeBoxesOverlapInVolume(new_cell_box, committed_box));
      if (ComputeBoxesTouch(new_cell_box, committed_box)) {
        const int size_level_difference =
            static_cast<int>(ToSizeLevel(new_cell)) - static_cast<int>(ToSizeLevel(committed_cell));
        EXPECT_LE(std::abs(size_level_difference), 1);
      }
    }
  }
}

// Checks by brute force that the box where a fill stopped touches a committed cell at least two size levels smaller and
// one at least one size level bigger.
void CheckStoppedBoxIsReallyBlocked(
    const std::vector<CellId>& committed_cells,
    CellId stopped_box,
    int size_level_count
) {
  const SmallestCellBox stopped_box_box = ToSmallestCellBox(stopped_box, size_level_count);
  const int stopped_size_level = static_cast<int>(ToSizeLevel(stopped_box));
  bool has_much_smaller_cell = false;
  bool has_bigger_cell = false;
  for (const CellId committed_cell : committed_cells) {
    if (!ComputeBoxesTouch(stopped_box_box, ToSmallestCellBox(committed_cell, size_level_count))) {
      continue;
    }
    const int committed_size_level = static_cast<int>(ToSizeLevel(committed_cell));
    has_much_smaller_cell = has_much_smaller_cell || committed_size_level >= stopped_size_level + 2;
    has_bigger_cell = has_bigger_cell || committed_size_level <= stopped_size_level - 1;
  }
  EXPECT_TRUE(has_much_smaller_cell);
  EXPECT_TRUE(has_bigger_cell);
}

// Fills every empty biggest cell beside the cells of one grid with the reference's facts. The function gives the
// reference's facts at every box the fill asked about. When a fill succeeds, its new cells cover the cube and obey the
// size step rule against every committed cell; when it stops, the box it stopped at is really blocked.
CubeCounts CheckEveryCubeOfGrid(const GridTables& tables, const std::vector<CellId>& cell_list) {
  CubeCounts counts = {.cube_count = 0, .cannot_be_filled_count = 0};
  const FindFacts find_reference_facts = [&tables](CellId new_cell) {
    return ComputeNewCellNeighborsByAllCells(tables, new_cell);
  };
  const std::vector<CellId> committed_cells = FindCommittedCells(tables);
  const int size_level_count = tables.frame.size_level_count;
  for (const CellId cube : FindCubesToFill(tables, cell_list)) {
    const CubeFill fill = BuildCubeFill(cube, find_reference_facts);
    CheckFunctionMatchesReference(tables, fill);
    if (fill.stopped_box.has_value()) {
      CheckStoppedBoxIsReallyBlocked(committed_cells, *fill.stopped_box, size_level_count);
      ++counts.cannot_be_filled_count;
    } else {
      CheckNewCellsCoverCube(fill, cube, size_level_count);
      CheckNewCellsObeySizeStepRule(committed_cells, fill, size_level_count);
    }
    ++counts.cube_count;
  }
  return counts;
}

// The top-down fill of every empty biggest cell beside the cells of the first 16 random grids, asked of the reference
// and of the function. It prints how many cubes cannot be filled. It checks these properties in one function because
// each TEST runs in a process of its own and would build the grids again.
TEST(NewCellNeighbors, RandomGrids_TopDownFillMatchesTheReference) {
  std::vector<std::size_t> entries(kGridCount);
  for (std::size_t entry = 0; entry < kGridCount; ++entry) {
    entries.at(entry) = entry;
  }
  const std::vector<TestGrid> grids = BuildReferenceGrids(entries);
  ASSERT_EQ(grids.size(), kGridCount);
  CubeCounts total = {.cube_count = 0, .cannot_be_filled_count = 0};
  for (std::size_t grid = 0; grid < kGridCount; ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    const GridTables tables = grids.at(grid).grid.tables(RunOn::kCpu);
    const CubeCounts counts = CheckEveryCubeOfGrid(tables, BuildShuffledRandomGrids().at(grid));
    total.cube_count += counts.cube_count;
    total.cannot_be_filled_count += counts.cannot_be_filled_count;
  }
  EXPECT_GT(total.cube_count, 0U);
  std::cout << "cubes that cannot be filled: " << total.cannot_be_filled_count << " of " << total.cube_count << "\n";
}

}  // namespace
}  // namespace fluidity::grid
