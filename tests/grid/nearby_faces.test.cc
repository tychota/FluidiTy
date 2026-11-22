#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <random>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_faces_cases.h"

namespace fluidity::grid {
namespace {

// The most faces a valid grid can list (method paper <sec-nearby-faces>).
constexpr int kMostFacesInAnyValidGrid = 212;

// One position in each of the 64 windows of the cell {1,1,1}, whose 26 neighbours are all split: every window and axis
// lists 76 faces.
TEST(FindNearbyFaces, AllNeighboursSplitEveryWindow_Has76) {
  const std::vector<CellId> cells = BuildCellWithEveryNeighbourSplit(1, 1, 1);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CheckEveryWindowHasCount(ToTablesOf(test_grid), ToCell(0, 1, 1, 1), 76);
}

// No axis in any of the 64 windows of the cell of the worst case lists more than 124 faces.
TEST(FindNearbyFaces, TwoSizesWorstCaseEveryWindow_NeverPasses124) {
  const std::vector<CellId> cells = BuildTwoSizesWorstCaseCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToTwoSizesWorstCaseFrame(), cells));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_EQ(FindLargestCountInEveryWindow(ToTablesOf(test_grid), ToCell(1, 2, 2, 2)), 124);
}

// 200 small random grids, eight cells spread over each, all 64 windows, every axis: the count stays at or below the
// proven bound.
TEST(FindNearbyFaces, SmallRandomGrids_NeverExceedTheProvenBound) {
  constexpr int kSizeLevelCount = 3;
  constexpr int kBiggestCellsPerAxis = 2;
  constexpr std::size_t kCellsPerGrid = 8;
  int largest_count = 0;
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    const std::vector<CellId> cells = BuildRandomGridCells(seed, kSizeLevelCount, kBiggestCellsPerAxis);
    const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(kSizeLevelCount), cells));
    ASSERT_TRUE(test_grid.has_value()) << seed;
    const GridTables tables = ToTablesOf(test_grid);
    // Each call takes tens of microseconds in the debug build, so eight cells spread over the list stand for all of it.
    const std::size_t stride = std::max<std::size_t>(1, cells.size() / kCellsPerGrid);
    for (std::size_t entry = 0; entry < cells.size(); entry += stride) {
      largest_count = std::max(largest_count, FindLargestCountInEveryWindow(tables, cells[entry]));
    }
  }
  RecordProperty("largest_nearby_faces", largest_count);
  EXPECT_LE(largest_count, kMostFacesInAnyValidGrid);
  EXPECT_GE(largest_count, 27);
}

// Draws one step of a position inside a cell: its low side or its centre, or any step.
std::int64_t FindDrawnStepInCell(std::mt19937_64& generator, bool is_on_side_or_centre) {
  if (is_on_side_or_centre) {
    return ToUniformIndex(generator(), 2) == 0 ? 0 : kPlacedPositionStepsPerCell / 2;
  }
  return static_cast<std::int64_t>(ToUniformIndex(generator(), kPlacedPositionStepsPerCell));
}

// Draws a position inside a cell of the list, on its side or centre on each axis, or anywhere inside it.
PlacedPosition FindDrawnPosition(
    std::mt19937_64& generator,
    const GridFrame& frame,
    const std::vector<CellId>& cells,
    bool is_on_side_or_centre
) {
  const CellId cell = cells[ToUniformIndex(generator(), cells.size())];
  const std::array<std::int64_t, 3> steps = {
      FindDrawnStepInCell(generator, is_on_side_or_centre),
      FindDrawnStepInCell(generator, is_on_side_or_centre),
      FindDrawnStepInCell(generator, is_on_side_or_centre),
  };
  return ToPlacedPosition(frame, cell, steps);
}

// Checks that the list of one axis is the checker's list for the same cell and window, in ascending order, and gives
// its count.
int CheckListMatchesChecker(
    const GridTables& tables,
    const GeometricGrid& checker,
    const PlacedPosition& placed,
    Axis axis
) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, placed.position, axis, CellBlockIndex{kNoIndex}, &error);
  EXPECT_EQ(error, GridError::kNone);
  EXPECT_EQ(list.cell.value, FindCellPlace(tables, placed.cell).value);
  CheckFacesAreAscending(list);
  std::vector<std::uint32_t> listed_places;
  listed_places.reserve(static_cast<std::size_t>(list.count));
  for (int entry = 0; entry < list.count; ++entry) {
    listed_places.push_back(list.faces[entry].value);
  }
  std::vector<std::uint32_t> checker_places;
  for (const GeometricFacePiece& piece : checker.NearbyFaces(placed.cell, placed.window_quarters, axis)) {
    checker_places.push_back(FindFacePlace(tables, piece.id).value);
  }
  std::ranges::sort(checker_places);
  EXPECT_EQ(listed_places, checker_places) << "axis " << static_cast<int>(axis);
  return list.count;
}

// Checks 50 positions of one random grid, 10 of them on a cell side or centre on every axis, and gives the largest
// count.
int CheckPositionsOfOneGrid(const GridTables& tables, const GeometricGrid& checker, const std::vector<CellId>& cells) {
  std::mt19937_64 generator(cells.size() + 1);
  int largest_count = 0;
  for (int draw = 0; draw < 50; ++draw) {
    const PlacedPosition placed = FindDrawnPosition(generator, tables.frame, cells, draw < 10);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      SCOPED_TRACE(draw);
      largest_count = std::max(largest_count, CheckListMatchesChecker(tables, checker, placed, axis));
    }
  }
  return largest_count;
}

// The 200 random grids, 50 positions each, 10 of them on a cell side or centre on every axis: the list of each axis is
// the checker's list for the same cell and window, in ascending order, and never longer than the proven bound.
TEST(FindNearbyFaces, RandomGrids_MatchGeometricChecker) {
  const RandomGridsWithCheckers& random_grids = FindRandomGridsWithCheckers();
  const std::vector<std::vector<CellId>>& cell_lists = BuildShuffledRandomGrids();
  ASSERT_EQ(random_grids.grids.size(), kRandomGridCount);
  int largest_count = 0;
  for (std::size_t grid_entry = 0; grid_entry < kRandomGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const GridTables tables = random_grids.grids[grid_entry].grid.tables(RunOn::kCpu);
    largest_count = std::max(
        largest_count,
        CheckPositionsOfOneGrid(tables, random_grids.checkers[grid_entry], cell_lists[grid_entry])
    );
  }
  RecordProperty("largest_nearby_faces", largest_count);
  EXPECT_LE(largest_count, kMostFacesInAnyValidGrid);
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
double FindDrawnCoordinate(std::mt19937_64& generator, double origin_on_axis, std::int32_t lowest_cell) {
  const auto cell =
      static_cast<std::int32_t>(lowest_cell + 2 + static_cast<std::int32_t>(ToUniformIndex(generator(), 5)));
  const double centre_fraction = ToUniformIndex(generator(), 2) == 0 ? 0.0 : 0.5;
  const double coordinate = origin_on_axis + ((static_cast<double>(cell) + centre_fraction) * 0.0375);
  const int step_count = static_cast<int>(ToUniformIndex(generator(), 7)) - 3;
  return ComputeMovedByRoundingSteps(coordinate, step_count);
}

// 1,000 positions made of a side or a centre of a cell on each axis, moved by -3 to +3 rounding steps, in a frame whose
// origin and width have no short binary fraction: 27 faces on every axis, in every one.
TEST(FindNearbyFaces, NonDyadicFrameAtSideOrCentre_AlwaysHas27) {
  const GridFrame frame = ToNonDyadicFrame();
  std::vector<CellId> cells;
  for (std::int32_t cell_z = 0; cell_z < 9; ++cell_z) {
    for (std::int32_t cell_y = 0; cell_y < 9; ++cell_y) {
      for (std::int32_t cell_x = -32; cell_x < -23; ++cell_x) {
        cells.push_back(ToCell(0, cell_x, cell_y, cell_z));
      }
    }
  }
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  std::mt19937_64 generator(20261120);  // NOLINT(bugprone-random-generator-seed)
  for (int draw = 0; draw < 1000; ++draw) {
    const Vec3 position = {
        .x = FindDrawnCoordinate(generator, frame.origin.x, -32),
        .y = FindDrawnCoordinate(generator, frame.origin.y, 0),
        .z = FindDrawnCoordinate(generator, frame.origin.z, 0),
    };
    CheckEveryAxisHasCount(ToTablesOf(test_grid), position, 27);
  }
}

}  // namespace
}  // namespace fluidity::grid
