// Test-only: the scene that the scene tests of the grid share, a box of liquid with a sloped surface.
//
// Purpose: let the CPU test and the GPU test of the whole grid module run on one realistic grid, built the same way.
//
// Idea: a box of 16 x 4 x 8 size level 0 cells, 1 m wide, in a frame of 3 size levels. The liquid surface is the
//   plane z = 4 + 0.25 x, and the cells along it are the smallest. A size level 2 cell (0.25 m wide) is marked when its
//   centre (x, y, z) has |z - (4 + 0.25 x)| < 1: the distance is measured along z, not at right angles to the plane,
//   so every number in the test is a multiple of 1/32 and the test is exact in doubles. The start list: a size level 0
//   cell with no marked centre inside stays whole; one with a marked centre is replaced by its 8 size level 1 cells;
//   each of those with a marked centre inside is replaced by its 8 size level 2 cells. The list is then closed by
//   BuildSizeStepClosure, which splits the cells that the rule of one size level of difference still forbids. The
//   result is a sheet of small cells through a box of big ones, which the random grids do not have.
//   The 1,000 positions come from one std::mt19937_64 of the seed 20261119, mapped by ToUniformIndex and ToUnitDouble:
//   each lies inside a marked size level 2 cell, so each is on the sheet.
//
// Owns:    kSlopedSurfaceSeed, kSlopedSurfacePositionCount, kSlopedSurfaceRangeBlocks, kSlopedSurfaceSizeLevelCount,
//          kSlopedSurfaceCellsAlongX, kSlopedSurfaceCellsAlongY, kSlopedSurfaceCellsAlongZ,
//          kSlopedSurfaceSmallestCellWidth, BuildSlopedSurfaceFrame, ComputeIsSlopedSurfaceCellMarked,
//          ComputeHasMarkedCellInside, BuildSlopedSurfaceCells,
//          BuildSlopedSurfacePositions, FindRangeOfEachPosition
// Uses:    CPU-only guard (base/cpu_only.h), Vec3 (base/vec3.h), ToUnitFrame, ToCell
//          (grid/testing/staggered_grid_cases.h), BuildSizeStepClosure, ToUniformIndex, ToUnitDouble
//          (grid/testing/grid_support.h), ComputeSmallCell, ToCellCoord, ToCellBlockIndex (grid/grid_ids.h),
//          FindCellAt (grid/find_cell_at.h), GridFrame (grid/grid_tables.h), StaggeredGrid (grid/staggered_grid.h)
// Used by: tests/grid/sloped_surface_scene.test.cc, tests/grid/sloped_surface_scene.gpu.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/vec3.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {

// The seed of the positions, how many there are, and the blocks of a range of the scene.
inline constexpr std::uint64_t kSlopedSurfaceSeed = 20261119;
inline constexpr std::size_t kSlopedSurfacePositionCount = 1000;
inline constexpr std::uint32_t kSlopedSurfaceRangeBlocks = 8;
inline constexpr int kSlopedSurfaceSizeLevelCount = 3;

// The size level 0 cells of the box on each axis, and the widths of a size level 2 cell in m.
inline constexpr std::int32_t kSlopedSurfaceCellsAlongX = 16;
inline constexpr std::int32_t kSlopedSurfaceCellsAlongY = 4;
inline constexpr std::int32_t kSlopedSurfaceCellsAlongZ = 8;
inline constexpr double kSlopedSurfaceSmallestCellWidth = 0.25;  // m.

// The frame of the scene: origin 0, biggest cells 1 m wide, 3 size levels.
inline GridFrame BuildSlopedSurfaceFrame() {
  return ToUnitFrame(kSlopedSurfaceSizeLevelCount);
}

// Says whether the size level 2 cell at `cell_x` and `cell_z` is marked: whether its centre is less than 1 m above or
// below the plane z = 4 + 0.25 x, measured along z. The y coordinate does not matter, since the plane is the same
// along y, so it is not a parameter.
// Example: the cell with x 0 and z 16 has its centre at x = 0.125 and z = 4.125, 0.09375 m above the plane: marked. The
//   cell with x 0 and z 0 has its centre 3.90625 m below the plane: not marked.
inline bool ComputeIsSlopedSurfaceCellMarked(std::int32_t cell_x, std::int32_t cell_z) {
  const double centre_x = (static_cast<double>(cell_x) + 0.5) * kSlopedSurfaceSmallestCellWidth;
  const double centre_z = (static_cast<double>(cell_z) + 0.5) * kSlopedSurfaceSmallestCellWidth;
  const double plane_z = 4.0 + (0.25 * centre_x);
  return std::abs(centre_z - plane_z) < 1.0;
}

// Says whether any size level 2 cell inside the cell of `size_level` at `cell_coord` is marked.
inline bool ComputeHasMarkedCellInside(int size_level, CellCoord cell_coord) {
  const auto smallest_cells_per_side =
      static_cast<std::int32_t>(1U << static_cast<unsigned>(kSlopedSurfaceSizeLevelCount - 1 - size_level));
  for (std::int32_t offset_z = 0; offset_z < smallest_cells_per_side; ++offset_z) {
    for (std::int32_t offset_x = 0; offset_x < smallest_cells_per_side; ++offset_x) {
      const bool is_marked = ComputeIsSlopedSurfaceCellMarked(
          (cell_coord.x * smallest_cells_per_side) + offset_x,
          (cell_coord.z * smallest_cells_per_side) + offset_z
      );
      if (is_marked) {
        return true;
      }
    }
  }
  return false;
}

// Builds the active cells of the scene, in the order of the start list and closed by BuildSizeStepClosure.
inline std::vector<CellId> BuildSlopedSurfaceCells() {
  std::vector<CellId> cells;
  for (std::int32_t cell_z = 0; cell_z < kSlopedSurfaceCellsAlongZ; ++cell_z) {
    for (std::int32_t cell_y = 0; cell_y < kSlopedSurfaceCellsAlongY; ++cell_y) {
      for (std::int32_t cell_x = 0; cell_x < kSlopedSurfaceCellsAlongX; ++cell_x) {
        const CellId whole_cell = ToCell(0, cell_x, cell_y, cell_z);
        if (!ComputeHasMarkedCellInside(0, {.x = cell_x, .y = cell_y, .z = cell_z})) {
          cells.push_back(whole_cell);
          continue;
        }
        for (int middle_eighth = 0; middle_eighth < 8; ++middle_eighth) {
          const CellId middle_cell = ComputeSmallCell(whole_cell, middle_eighth);
          if (!ComputeHasMarkedCellInside(1, ToCellCoord(middle_cell))) {
            cells.push_back(middle_cell);
            continue;
          }
          for (int smallest_eighth = 0; smallest_eighth < 8; ++smallest_eighth) {
            cells.push_back(ComputeSmallCell(middle_cell, smallest_eighth));
          }
        }
      }
    }
  }
  return BuildSizeStepClosure(std::move(cells), kSlopedSurfaceSizeLevelCount);
}

// Builds the 1,000 positions of the scene, in m, world: each inside a marked size level 2 cell, drawn from the seed.
// Each position takes four raw numbers in turn: the marked cell, then x, y and z inside it.
inline std::vector<Vec3> BuildSlopedSurfacePositions() {
  struct MarkedCell {
    std::int32_t x;
    std::int32_t y;
    std::int32_t z;
  };
  std::vector<MarkedCell> marked_cells;
  for (std::int32_t cell_z = 0; cell_z < kSlopedSurfaceCellsAlongZ * 4; ++cell_z) {
    for (std::int32_t cell_y = 0; cell_y < kSlopedSurfaceCellsAlongY * 4; ++cell_y) {
      for (std::int32_t cell_x = 0; cell_x < kSlopedSurfaceCellsAlongX * 4; ++cell_x) {
        if (ComputeIsSlopedSurfaceCellMarked(cell_x, cell_z)) {
          marked_cells.push_back({.x = cell_x, .y = cell_y, .z = cell_z});
        }
      }
    }
  }
  std::mt19937_64 generator(kSlopedSurfaceSeed);  // NOLINT(bugprone-random-generator-seed): the seed is the point.
  std::vector<Vec3> positions;
  positions.reserve(kSlopedSurfacePositionCount);
  for (std::size_t entry = 0; entry < kSlopedSurfacePositionCount; ++entry) {
    const MarkedCell cell = marked_cells.at(ToUniformIndex(generator(), marked_cells.size()));
    const double fraction_x = ToUnitDouble(generator());
    const double fraction_y = ToUnitDouble(generator());
    const double fraction_z = ToUnitDouble(generator());
    positions.push_back({
        .x = (static_cast<double>(cell.x) + fraction_x) * kSlopedSurfaceSmallestCellWidth,
        .y = (static_cast<double>(cell.y) + fraction_y) * kSlopedSurfaceSmallestCellWidth,
        .z = (static_cast<double>(cell.z) + fraction_z) * kSlopedSurfaceSmallestCellWidth,
    });
  }
  return positions;
}

// Finds the range that holds the cell of each position on the whole grid `whole`: the index into `ranges`, or the
// number of ranges when the position is in no cell or no range holds its cell. The result has one entry per position.
inline std::vector<std::size_t> FindRangeOfEachPosition(
    const GridTables& whole,
    std::span<const CellBlockRange> ranges,
    std::span<const Vec3> positions
) {
  std::vector<std::size_t> range_of_position;
  for (const Vec3 position : positions) {
    const CellPlaceIndex cell = FindCellAt(whole, position);
    std::size_t found = ranges.size();
    if (cell.value != kNoIndex) {
      const std::uint32_t block = ToCellBlockIndex(cell).value;
      for (std::size_t range = 0; range < ranges.size(); ++range) {
        if (block >= ranges[range].first_block && block - ranges[range].first_block < ranges[range].block_count) {
          found = range;
        }
      }
    }
    range_of_position.push_back(found);
  }
  return range_of_position;
}

}  // namespace fluidity::grid
