#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
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

// One box of the top-down fill of a cube that ended as a new cell, with the facts its committed neighbours gave and
// whether the big cell it came from had to split.
struct FilledCell {
  CellId new_cell;
  NewCellNeighbors facts;
  bool big_cell_had_to_split;
};

// What the top-down fill of one empty biggest cell visited and produced.
struct CubeFill {
  std::vector<CellId> visited_boxes;  // Every box asked about: the cube and each box made by a split.
  std::vector<FilledCell> new_cells;  // The boxes that were not split.
};

using FindFacts = std::function<NewCellNeighbors(CellId)>;

// Fills one empty biggest cell from the top: asks about the cube, splits a box into its 8 small cells while a committed
// cell two or more size levels smaller touches it, and repeats on the small cells.
CubeFill BuildCubeFill(CellId cube, const FindFacts& find_facts) {
  CubeFill fill;
  std::vector<FilledCell> pending = {{.new_cell = cube, .facts = {}, .big_cell_had_to_split = false}};
  while (!pending.empty()) {
    const FilledCell box = pending.back();
    pending.pop_back();
    const NewCellNeighbors facts = find_facts(box.new_cell);
    fill.visited_boxes.push_back(box.new_cell);
    if (!facts.touches_much_smaller_cell) {
      fill.new_cells.push_back(
          {.new_cell = box.new_cell, .facts = facts, .big_cell_had_to_split = box.big_cell_had_to_split}
      );
      continue;
    }
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      pending.push_back(
          {.new_cell = ComputeSmallCell(box.new_cell, cell_eighth), .facts = {}, .big_cell_had_to_split = true}
      );
    }
  }
  return fill;
}

// Says whether a new cell of the fill has a bigger committed neighbour although the big cell it came from had to split:
// the cube cannot be filled with cells that obey the size step rule.
bool ComputeCubeCannotBeFilled(const CubeFill& fill) {
  return std::ranges::any_of(fill.new_cells, [](const FilledCell& new_cell) {
    return new_cell.big_cell_had_to_split && new_cell.facts.touches_bigger_cell;
  });
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
  for (const FilledCell& filled : fill.new_cells) {
    const std::int64_t width = ToSmallestCellBox(filled.new_cell, size_level_count).width;
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

// Fills every empty biggest cell beside the cells of one grid with the reference's facts: the new cells cover each
// cube and none is told to split.
CubeCounts CheckEveryCubeOfGrid(const GridTables& tables, const std::vector<CellId>& cell_list) {
  CubeCounts counts = {.cube_count = 0, .cannot_be_filled_count = 0};
  const FindFacts find_reference_facts = [&tables](CellId new_cell) {
    return ComputeNewCellNeighborsByAllCells(tables, new_cell);
  };
  for (const CellId cube : FindCubesToFill(tables, cell_list)) {
    const CubeFill fill = BuildCubeFill(cube, find_reference_facts);
    CheckNewCellsCoverCube(fill, cube, tables.frame.size_level_count);
    for (const FilledCell& filled : fill.new_cells) {
      EXPECT_FALSE(filled.facts.touches_much_smaller_cell);
    }
    ++counts.cube_count;
    counts.cannot_be_filled_count += ComputeCubeCannotBeFilled(fill) ? 1 : 0;
  }
  return counts;
}

// The top-down fill of every empty biggest cell beside the cells of the first 16 random grids, asked of the reference.
// It prints how many cubes cannot be filled. It checks these properties in one function because each TEST runs in a
// process of its own and would build the grids again.
TEST(NewCellNeighbors, RandomGrids_TopDownFillCoversEveryCube) {
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
  std::cout << "cubes that cannot be filled: " << total.cannot_be_filled_count << " of " << total.cube_count << " over "
            << kGridCount << " grids\n";
}

}  // namespace
}  // namespace fluidity::grid
