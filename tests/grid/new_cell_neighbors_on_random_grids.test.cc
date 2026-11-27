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
  // The box where both facts were true at once. A valid committed grid has none: the fill stops there, and the test
  // fails.
  std::optional<CellId> box_with_both_facts;
};

using FindFacts = std::function<NewCellNeighbors(CellId)>;

// Fills one empty biggest cell from the top: asks about the cube, splits a box into its 8 small cells while a committed
// cell two or more size levels smaller touches it, and repeats on the small cells. A box that has to be split and also
// touches a bigger committed cell cannot occur on a valid committed grid (see FindNewCellNeighbors); the fill stops at
// such a box and records it.
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
      fill.box_with_both_facts = box;
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

// How many cubes one grid had, how many pairs of new cells each pair search looked at, and how many pairs of new cells
// of neighbouring cubes break the size step rule.
struct CubeCounts {
  std::size_t cube_count;
  std::size_t same_cube_pair_count;
  std::size_t neighbouring_cube_pair_count;
  std::size_t breaking_pair_count;
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

// Gives the boxes of `cells`, in smallest cells.
std::vector<SmallestCellBox> ToBoxes(const std::vector<CellId>& cells, int size_level_count) {
  std::vector<SmallestCellBox> boxes;
  boxes.reserve(cells.size());
  for (const CellId cell : cells) {
    boxes.push_back(ToSmallestCellBox(cell, size_level_count));
  }
  return boxes;
}

// Gives how many size levels apart two cells are, whichever is bigger.
int ToSizeLevelDifference(CellId first, CellId second) {
  return std::abs(static_cast<int>(ToSizeLevel(first)) - static_cast<int>(ToSizeLevel(second)));
}

// Checks by brute force that the new cells of one cube overlap in no volume, and that two of them that touch are at
// most one size level apart. Together with CheckNewCellsCoverCube this says the cube is covered by cells that obey the
// rule between themselves. Gives how many pairs it looked at.
std::size_t CheckNewCellsObeyRuleBetweenThemselves(const CubeFill& fill, int size_level_count) {
  const std::vector<SmallestCellBox> boxes = ToBoxes(fill.new_cells, size_level_count);
  std::size_t pair_count = 0;
  for (std::size_t first = 0; first < boxes.size(); ++first) {
    for (std::size_t second = first + 1; second < boxes.size(); ++second) {
      ++pair_count;
      EXPECT_FALSE(ComputeBoxesOverlapInVolume(boxes.at(first), boxes.at(second)));
      if (ComputeBoxesTouch(boxes.at(first), boxes.at(second))) {
        EXPECT_LE(ToSizeLevelDifference(fill.new_cells.at(first), fill.new_cells.at(second)), 1);
      }
    }
  }
  return pair_count;
}

// Says whether two cubes, as size-level-0 ids, are different and touch (at most one biggest cell apart on each axis).
bool ComputeAreCubesNeighbours(CellId first_cube, CellId second_cube) {
  const CellCoord first = ToCellCoord(first_cube);
  const CellCoord second = ToCellCoord(second_cube);
  return first_cube.value != second_cube.value && std::abs(first.x - second.x) <= 1 && std::abs(first.y - second.y) <= 1
      && std::abs(first.z - second.z) <= 1;
}

// Counts the pairs, one new cell of each fill, that touch and are two or more size levels apart, and adds to
// examined_pair_count every pair of one new cell of each fill that it looked at.
std::size_t FindBreakingPairCountBetweenFills(
    const CubeFill& first_fill,
    const CubeFill& second_fill,
    int size_level_count,
    std::size_t& examined_pair_count
) {
  const std::vector<SmallestCellBox> first_boxes = ToBoxes(first_fill.new_cells, size_level_count);
  const std::vector<SmallestCellBox> second_boxes = ToBoxes(second_fill.new_cells, size_level_count);
  std::size_t breaking_pair_count = 0;
  for (std::size_t first = 0; first < first_boxes.size(); ++first) {
    for (std::size_t second = 0; second < second_boxes.size(); ++second) {
      ++examined_pair_count;
      if (ComputeBoxesTouch(first_boxes.at(first), second_boxes.at(second))
          && ToSizeLevelDifference(first_fill.new_cells.at(first), second_fill.new_cells.at(second)) >= 2) {
        ++breaking_pair_count;
      }
    }
  }
  return breaking_pair_count;
}

// Fills every empty biggest cell beside the cells of one grid with the reference's facts. The function gives the
// reference's facts at every box the fill asked about. No box has both facts; the new cells of a cube cover it, obey
// the size step rule against every committed cell and between themselves. The pairs of new cells of two neighbouring
// cubes, each filled on its own, that break the rule are counted and not expected: the step module owns that case.
CubeCounts CheckEveryCubeOfGrid(const GridTables& tables, const std::vector<CellId>& cell_list) {
  CubeCounts counts = {
      .cube_count = 0,
      .same_cube_pair_count = 0,
      .neighbouring_cube_pair_count = 0,
      .breaking_pair_count = 0,
  };
  const FindFacts find_reference_facts = [&tables](CellId new_cell) {
    return ComputeNewCellNeighborsByAllCells(tables, new_cell);
  };
  const std::vector<CellId> committed_cells = FindCommittedCells(tables);
  const int size_level_count = tables.frame.size_level_count;
  const std::vector<CellId> cubes = FindCubesToFill(tables, cell_list);
  std::vector<CubeFill> fills;
  for (const CellId cube : cubes) {
    const CubeFill fill = BuildCubeFill(cube, find_reference_facts);
    CheckFunctionMatchesReference(tables, fill);
    EXPECT_FALSE(fill.box_with_both_facts.has_value());
    if (!fill.box_with_both_facts.has_value()) {
      CheckNewCellsCoverCube(fill, cube, size_level_count);
      CheckNewCellsObeySizeStepRule(committed_cells, fill, size_level_count);
      counts.same_cube_pair_count += CheckNewCellsObeyRuleBetweenThemselves(fill, size_level_count);
    }
    fills.push_back(fill);
    ++counts.cube_count;
  }
  for (std::size_t first = 0; first < cubes.size(); ++first) {
    for (std::size_t second = first + 1; second < cubes.size(); ++second) {
      if (ComputeAreCubesNeighbours(cubes.at(first), cubes.at(second))) {
        const CubeFill& first_fill = fills.at(first);
        const CubeFill& second_fill = fills.at(second);
        counts.breaking_pair_count += FindBreakingPairCountBetweenFills(
            first_fill,
            second_fill,
            size_level_count,
            counts.neighbouring_cube_pair_count
        );
      }
    }
  }
  return counts;
}

// The top-down fill of every empty biggest cell beside the cells of the first 16 random grids, asked of the reference
// and of the function. It prints how many pairs of new cells of neighbouring cubes break the rule. It checks these
// properties in one function because each TEST runs in a process of its own and would build the grids again.
TEST(NewCellNeighbors, RandomGrids_TopDownFillMatchesTheReference) {
  std::vector<std::size_t> entries(kGridCount);
  for (std::size_t entry = 0; entry < kGridCount; ++entry) {
    entries.at(entry) = entry;
  }
  const std::vector<TestGrid> grids = BuildReferenceGrids(entries);
  ASSERT_EQ(grids.size(), kGridCount);
  CubeCounts total = {
      .cube_count = 0,
      .same_cube_pair_count = 0,
      .neighbouring_cube_pair_count = 0,
      .breaking_pair_count = 0,
  };
  for (std::size_t grid = 0; grid < kGridCount; ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    const GridTables tables = grids.at(grid).grid.tables(RunOn::kCpu);
    const CubeCounts counts = CheckEveryCubeOfGrid(tables, BuildShuffledRandomGrids().at(grid));
    total.cube_count += counts.cube_count;
    total.same_cube_pair_count += counts.same_cube_pair_count;
    total.neighbouring_cube_pair_count += counts.neighbouring_cube_pair_count;
    total.breaking_pair_count += counts.breaking_pair_count;
  }
  EXPECT_GT(total.cube_count, 0U);
  EXPECT_GT(total.same_cube_pair_count, 0U);
  EXPECT_GT(total.neighbouring_cube_pair_count, 0U);
  std::cout << "pairs of new cells examined: " << total.same_cube_pair_count << " in one cube, "
            << total.neighbouring_cube_pair_count << " in neighbouring cubes; pairs of new cells of neighbouring cubes"
            << " that break the size step rule: " << total.breaking_pair_count << "\n";
}

}  // namespace
}  // namespace fluidity::grid
