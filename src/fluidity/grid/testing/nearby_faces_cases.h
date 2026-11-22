// Test-only: the helpers that the unit tests and the functional tests of nearby_face_weights share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the grids and the positions that both kinds of test use are built here: the two-size grid with the most
//   candidate faces known, a box of cells around one cell, a position inside a cell given as whole steps, and the
//   offsets of a list's faces from a position.
//
// Owns:    ToTwoSizesWorstCaseFrame, BuildTwoSizesWorstCaseCells, kTwoSizesWorstCasePosition, FindNearbyFaceCount,
//          CheckEveryAxisHasCount, BuildFiveByFiveBoxAround, ComputeValueAlong, ToFaceOffsets, ComputeOffsetCountAt,
//          PlacedPosition, ToPlacedPosition, kPlacedPositionStepsPerCell, kPlacedPositionStepsPerQuarter,
//          ComputePositionAtStep, ToWindowMiddlePosition, CheckEveryWindowHasCount, FindLargestCountInEveryWindow,
//          CandidateCountsOfCell, ComputeCandidateCountsInEveryWindow, FindLargestCandidateCount,
//          CheckCountsMatchFindNearbyFaces,
//          CheckFacesAreAscending, CheckEntriesFromAreZero, ComputeMovedByRoundingSteps, FindDrawnStepInCell,
//          FindDrawnPosition, ToNonDyadicFrame, FindDrawnCoordinate, kNonDyadicSeed, BuildNonDyadicCells,
//          FindDrawnNonDyadicPosition
// Uses:    the headers it includes
// Used by: src/fluidity/grid/nearby_face_weights.test.cc, tests/grid/nearby_faces.test.cc and
//          tests/grid/nearby_face_weights_on_many_positions.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// The frame of the two-size grid with the most candidate faces known: 3 size levels, origin 0, biggest cells 4 m wide.
constexpr GridFrame ToTwoSizesWorstCaseFrame() {
  return {.origin = {.x = 0, .y = 0, .z = 0}, .biggest_cell_width = 4.0, .size_level_count = 3};
}

// The position, in m, of the window (1, 0, 1) of the size-level-1 cell {2,2,2} of BuildTwoSizesWorstCaseCells, whose
// Y axis lists 124 faces.
inline constexpr Vec3 kTwoSizesWorstCasePosition = {.x = 4.75, .y = 4.25, .z = 4.75};

// Builds the cells of the two-size grid with the most candidate faces known (the frame of ToTwoSizesWorstCaseFrame).
// The size-level-0 box {0..2}^3 is split everywhere, which gives 6^3 size-level-1 places 2 m wide. The cell is the
// size-level-1 cell {2,2,2}. Of its 26 neighbours 14 stay size level 1: the 8 corners, (-1,-1,0), (1,-1,0) and
// (0,+-1,+-1) as (dx, dy, dz). The 12 others, and every size-level-1 place farther out, are split into size-level-2
// cells: 15 size-level-1 and 1,608 size-level-2 cells.
inline std::vector<CellId> BuildTwoSizesWorstCaseCells() {
  std::vector<CellId> cells;
  for (std::int32_t cell_z = 0; cell_z < 6; ++cell_z) {
    for (std::int32_t cell_y = 0; cell_y < 6; ++cell_y) {
      for (std::int32_t cell_x = 0; cell_x < 6; ++cell_x) {
        const std::int32_t offset_x = cell_x - 2;
        const std::int32_t offset_y = cell_y - 2;
        const std::int32_t offset_z = cell_z - 2;
        const bool is_cell_itself = offset_x == 0 && offset_y == 0 && offset_z == 0;
        const bool is_corner = std::abs(offset_x) == 1 && std::abs(offset_y) == 1 && std::abs(offset_z) == 1;
        const bool is_edge_along_y_across_x = offset_z == 0 && offset_y == -1 && std::abs(offset_x) == 1;
        const bool is_edge_across_y_and_z = offset_x == 0 && std::abs(offset_y) == 1 && std::abs(offset_z) == 1;
        if (is_cell_itself || is_corner || is_edge_along_y_across_x || is_edge_across_y_and_z) {
          cells.push_back(ToCell(1, cell_x, cell_y, cell_z));
        } else {
          ApplySmallCellsToList(ToCell(1, cell_x, cell_y, cell_z), cells);
        }
      }
    }
  }
  return cells;
}

// Finds how many faces FindNearbyFaces lists for a position and an axis. A failure fails the calling test and gives -1.
inline int FindNearbyFaceCount(const GridTables& tables, Vec3 position, Axis axis) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, position, axis, CellBlockIndex{kNoIndex}, &error);
  EXPECT_EQ(error, GridError::kNone);
  return error == GridError::kNone ? list.count : -1;
}

// Checks that the list of each of the three axes holds `expected_count` faces.
inline void CheckEveryAxisHasCount(const GridTables& tables, Vec3 position, int expected_count) {
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    EXPECT_EQ(FindNearbyFaceCount(tables, position, axis), expected_count)
        << "axis " << static_cast<int>(axis) << " at {" << position.x << ", " << position.y << ", " << position.z
        << "}";
  }
}

// The size-level-0 cells of the box around a middle cell, 2 cells each way on every axis: the middle cell and its 124
// others.
inline std::vector<CellId>
BuildFiveByFiveBoxAround(std::int32_t middle_x, std::int32_t middle_y, std::int32_t middle_z) {
  std::vector<CellId> cells;
  for (std::int32_t cell_z = middle_z - 2; cell_z <= middle_z + 2; ++cell_z) {
    for (std::int32_t cell_y = middle_y - 2; cell_y <= middle_y + 2; ++cell_y) {
      for (std::int32_t cell_x = middle_x - 2; cell_x <= middle_x + 2; ++cell_x) {
        cells.push_back(ToCell(0, cell_x, cell_y, cell_z));
      }
    }
  }
  return cells;
}

// The value of a vector along an axis.
constexpr double ComputeValueAlong(Vec3 vector, Axis axis) {
  if (axis == Axis::kX) {
    return vector.x;
  }
  return axis == Axis::kY ? vector.y : vector.z;
}

// The offsets of the faces of a list from a position, face minus position, in m, in the order of the list.
inline std::vector<Vec3> ToFaceOffsets(const GridTables& tables, const NearbyFaceList& list, Vec3 position) {
  std::vector<Vec3> offsets;
  offsets.reserve(static_cast<std::size_t>(list.count));
  for (int entry = 0; entry < list.count; ++entry) {
    const Vec3 face_position = ComputeFacePosition(tables, list.faces[entry]);
    offsets.push_back({
        .x = face_position.x - position.x,
        .y = face_position.y - position.y,
        .z = face_position.z - position.z,
    });
  }
  return offsets;
}

// Counts the offsets whose value along `axis` is exactly `value`.
inline int ComputeOffsetCountAt(const std::vector<Vec3>& offsets, Axis axis, double value) {
  return static_cast<int>(std::ranges::count_if(offsets, [axis, value](Vec3 offset) {
    return ComputeValueAlong(offset, axis) == value;
  }));
}

// A cell is divided in this many steps on each axis to place a position inside it: 2^20, so every step is exact.
inline constexpr std::int64_t kPlacedPositionStepsPerCell = 1048576;

// The steps in a quarter of a cell: the step number divided by this is the window quarter.
inline constexpr std::int64_t kPlacedPositionStepsPerQuarter = kPlacedPositionStepsPerCell / 4;

// A position inside a cell and the window quarters it lies in.
struct PlacedPosition {
  CellId cell;  // The cell that holds the position.
  Vec3 position;  // m, world.
  std::array<int, 3> window_quarters;  // 0..3 on x, y, z.
};

// Computes a position on one axis: `step` steps of a cell's width from its low corner.
// clang-tidy: an origin and a width are both doubles; the swappable-parameters check flags them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline double
ComputePositionAtStep(double origin_on_axis, double cell_width, std::int32_t cell_coord, std::int64_t step) {
  const double cell_fraction = static_cast<double>(step) / static_cast<double>(kPlacedPositionStepsPerCell);
  return origin_on_axis + ((static_cast<double>(cell_coord) + cell_fraction) * cell_width);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Places a position inside a cell, `steps` of its width from its low corner on each axis, each
// 0..kPlacedPositionStepsPerCell
// - 1. The window quarter is the step number divided by kPlacedPositionStepsPerQuarter. Exact when the frame's origin
// and width are, as for origin 0 and biggest cells 1 m wide.
inline PlacedPosition ToPlacedPosition(GridFrame frame, CellId cell, std::array<std::int64_t, 3> steps) {
  const double cell_width = ComputeCellWidth(frame, ToSizeLevel(cell));
  const CellCoord cell_coord = ToCellCoord(cell);
  return {
      .cell = cell,
      .position =
          {
              .x = ComputePositionAtStep(frame.origin.x, cell_width, cell_coord.x, steps[0]),
              .y = ComputePositionAtStep(frame.origin.y, cell_width, cell_coord.y, steps[1]),
              .z = ComputePositionAtStep(frame.origin.z, cell_width, cell_coord.z, steps[2]),
          },
      .window_quarters = {
          static_cast<int>(steps[0] / kPlacedPositionStepsPerQuarter),
          static_cast<int>(steps[1] / kPlacedPositionStepsPerQuarter),
          static_cast<int>(steps[2] / kPlacedPositionStepsPerQuarter),
      },
  };
}

// Places a position at the middle of one window of a cell: window 0..63, quarter x = window % 4, y = (window / 4) % 4
// and z = window / 16.
inline PlacedPosition ToWindowMiddlePosition(GridFrame frame, CellId cell, int window) {
  const std::int64_t half_quarter = kPlacedPositionStepsPerQuarter / 2;
  const std::array<std::int64_t, 3> steps = {
      ((window % 4) * kPlacedPositionStepsPerQuarter) + half_quarter,
      (((window / 4) % 4) * kPlacedPositionStepsPerQuarter) + half_quarter,
      ((window / 16) * kPlacedPositionStepsPerQuarter) + half_quarter,
  };
  return ToPlacedPosition(frame, cell, steps);
}

// Checks that every one of the 64 windows of a cell, on every axis, lists `expected_count` faces.
inline void CheckEveryWindowHasCount(const GridTables& tables, CellId cell, int expected_count) {
  for (int window = 0; window < 64; ++window) {
    const PlacedPosition placed = ToWindowMiddlePosition(tables.frame, cell, window);
    CheckEveryAxisHasCount(tables, placed.position, expected_count);
  }
}

// Finds the largest count of faces over the 64 windows of a cell and the three axes: one position at the middle of each
// window. A failure of FindNearbyFaces fails the calling test.
inline int FindLargestCountInEveryWindow(const GridTables& tables, CellId cell) {
  int largest_count = 0;
  for (int window = 0; window < 64; ++window) {
    const PlacedPosition placed = ToWindowMiddlePosition(tables.frame, cell, window);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      largest_count = std::max(largest_count, FindNearbyFaceCount(tables, placed.position, axis));
    }
  }
  return largest_count;
}

// The candidate counts of one cell: counts[axis][window] for each axis (X, Y, Z) and each of the 64 windows.
using CandidateCountsOfCell = std::array<std::array<int, 64>, 3>;

// Counts the candidates of one cell on each axis in each of the 64 windows, from the primitives of
// FindNearbyFacesWithRoom but without a position: the face pieces of the cell and of its touching cells are
// listed and made unique once, and ComputeIsInCandidateWindow then counts each window. Window w is the one of
// ToWindowMiddlePosition: quarter x = w % 4, y = (w / 4) % 4, z = w / 16.
inline CandidateCountsOfCell ComputeCandidateCountsInEveryWindow(const GridTables& tables, CellId cell) {
  const CellPlaceIndex cell_place = FindCellPlace(tables, cell);
  const int cell_size_level = static_cast<int>(ToSizeLevel(cell));
  const CellCoord cell_coord = ToCellCoord(cell);
  struct CandidateFace {
    std::uint32_t place;
    std::size_t axis_number;
    std::array<std::int64_t, 3> quarters;  // x, y, z, from the cell's low corner.
  };
  std::vector<CandidateFace> candidates;
  const TouchingCells touching = FindTouchingCells(tables, cell_place);
  for (int entry = -1; entry < touching.count; ++entry) {
    const CellFacePieces pieces = FindFacePieces(tables, entry < 0 ? cell_place : touching.cells[entry]);
    for (int piece = 0; piece < pieces.count; ++piece) {
      const FacePlaceIndex face = pieces.faces[piece];
      const int level_difference = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(face))) - cell_size_level;
      if (level_difference < -1 || level_difference > 1) {
        continue;
      }
      const QuarterWidthPoint centre = ComputeFaceCenterInQuarterWidths(tables, face, cell_size_level);
      candidates.push_back({
          .place = face.value,
          .axis_number = static_cast<std::size_t>(ToAxis(face)),
          .quarters = {
              centre.x - (4 * std::int64_t{cell_coord.x}),
              centre.y - (4 * std::int64_t{cell_coord.y}),
              centre.z - (4 * std::int64_t{cell_coord.z}),
          },
      });
    }
  }
  std::ranges::sort(candidates, {}, &CandidateFace::place);
  const auto duplicates = std::ranges::unique(candidates, {}, &CandidateFace::place);
  candidates.erase(duplicates.begin(), duplicates.end());
  CandidateCountsOfCell counts{};
  for (int window = 0; window < 64; ++window) {
    for (const CandidateFace& face : candidates) {
      const bool is_in_window = ComputeIsInCandidateWindow(face.quarters[0], window % 4)
          && ComputeIsInCandidateWindow(face.quarters[1], (window / 4) % 4)
          && ComputeIsInCandidateWindow(face.quarters[2], window / 16);
      counts[face.axis_number][static_cast<std::size_t>(window)] += is_in_window ? 1 : 0;
    }
  }
  return counts;
}

// Gives the largest count of a cell, over its axes and windows.
inline int FindLargestCandidateCount(const CandidateCountsOfCell& counts) {
  int largest_count = 0;
  for (const std::array<int, 64>& counts_of_axis : counts) {
    largest_count = std::max(largest_count, *std::ranges::max_element(counts_of_axis));
  }
  return largest_count;
}

// Checks that FindNearbyFaces, at the middle of each of the 64 windows of a cell and on each axis, lists as many faces
// as `expected_counts` says (ComputeCandidateCountsInEveryWindow).
inline void
CheckCountsMatchFindNearbyFaces(const GridTables& tables, CellId cell, const CandidateCountsOfCell& expected_counts) {
  for (int window = 0; window < 64; ++window) {
    const PlacedPosition placed = ToWindowMiddlePosition(tables.frame, cell, window);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      const int expected_count = expected_counts[static_cast<std::size_t>(axis)][static_cast<std::size_t>(window)];
      EXPECT_EQ(FindNearbyFaceCount(tables, placed.position, axis), expected_count)
          << "window " << window << " axis " << static_cast<int>(axis);
    }
  }
}

// Checks that the faces of a list are in strictly ascending index order.
inline void CheckFacesAreAscending(const NearbyFaceList& list) {
  for (int entry = 1; entry < list.count; ++entry) {
    EXPECT_LT(list.faces[entry - 1].value, list.faces[entry].value) << entry;
  }
}

// Checks that every entry of a list from `first_entry` on is 0, as a value-initialised list leaves it.
inline void CheckEntriesFromAreZero(const NearbyFaceList& list, int first_entry) {
  for (int entry = first_entry; entry < kMaxNearbyFaces; ++entry) {
    EXPECT_EQ(list.faces[entry].value, 0U) << entry;
  }
}

// Moves a position by `step_count` rounding steps: up, or down when negative.
// clang-tidy: a position and a count of steps; the swappable-parameters check flags the two numbers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline double ComputeMovedByRoundingSteps(double position, int step_count) {
  const double direction =
      step_count < 0 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
  for (int step = 0; step < std::abs(step_count); ++step) {
    position = std::nextafter(position, direction);
  }
  return position;
}

// Where a drawn position lies inside its cell, on each axis.
enum class DrawnPlace : std::uint8_t {
  kAnywhere,  // Any step of the cell.
  kSideOrCentre,  // The low side or the centre of the cell.
};

// Draws one step of a position inside a cell: its low side or its centre, or any step.
inline std::int64_t FindDrawnStepInCell(std::mt19937_64& generator, DrawnPlace place) {
  if (place == DrawnPlace::kSideOrCentre) {
    return ToUniformIndex(generator(), 2) == 0 ? 0 : kPlacedPositionStepsPerCell / 2;
  }
  return static_cast<std::int64_t>(ToUniformIndex(generator(), kPlacedPositionStepsPerCell));
}

// Draws a position inside a cell of the list, on its side or centre on each axis, or anywhere inside it.
inline PlacedPosition FindDrawnPosition(
    std::mt19937_64& generator,
    const GridFrame& frame,
    const std::vector<CellId>& cells,
    DrawnPlace place
) {
  const CellId cell = cells[ToUniformIndex(generator(), cells.size())];
  const std::array<std::int64_t, 3> steps = {
      FindDrawnStepInCell(generator, place),
      FindDrawnStepInCell(generator, place),
      FindDrawnStepInCell(generator, place),
  };
  return ToPlacedPosition(frame, cell, steps);
}

// The frame of the uniform 9^3 grid whose origin and width have no short binary fraction.
constexpr GridFrame ToNonDyadicFrame() {
  return {
      .origin = {.x = -17.616723516683763, .y = 0.3, .z = -7.1},
      .biggest_cell_width = 0.0375,
      .size_level_count = 1,
  };
}

// Draws a position on one axis: a side or a centre of one of the 5 middle cells of the 9, moved by -3 to +3 rounding
// steps.
// clang-tidy: an origin and the lowest cell; the swappable-parameters check flags the two numbers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline double FindDrawnCoordinate(std::mt19937_64& generator, double origin_on_axis, std::int32_t lowest_cell) {
  const auto cell =
      static_cast<std::int32_t>(lowest_cell + 2 + static_cast<std::int32_t>(ToUniformIndex(generator(), 5)));
  const double centre_fraction = ToUniformIndex(generator(), 2) == 0 ? 0.0 : 0.5;
  const double coordinate = origin_on_axis + ((static_cast<double>(cell) + centre_fraction) * 0.0375);
  const int step_count = static_cast<int>(ToUniformIndex(generator(), 7)) - 3;
  return ComputeMovedByRoundingSteps(coordinate, step_count);
}

// The seed of the generator that draws the positions of the non-dyadic frame.
inline constexpr std::uint64_t kNonDyadicSeed = 20261120;

// The 9^3 size-level-0 cells of ToNonDyadicFrame: coordinates -32..-24 on x and 0..8 on y and z.
inline std::vector<CellId> BuildNonDyadicCells() {
  std::vector<CellId> cells;
  for (std::int32_t cell_z = 0; cell_z < 9; ++cell_z) {
    for (std::int32_t cell_y = 0; cell_y < 9; ++cell_y) {
      for (std::int32_t cell_x = -32; cell_x < -23; ++cell_x) {
        cells.push_back(ToCell(0, cell_x, cell_y, cell_z));
      }
    }
  }
  return cells;
}

// Draws a position of ToNonDyadicFrame: on each axis a side or a centre of one of the 5 middle cells, moved by -3 to +3
// rounding steps.
inline Vec3 FindDrawnNonDyadicPosition(std::mt19937_64& generator, const GridFrame& frame) {
  return {
      .x = FindDrawnCoordinate(generator, frame.origin.x, -32),
      .y = FindDrawnCoordinate(generator, frame.origin.y, 0),
      .z = FindDrawnCoordinate(generator, frame.origin.z, 0),
  };
}

}  // namespace fluidity::grid
