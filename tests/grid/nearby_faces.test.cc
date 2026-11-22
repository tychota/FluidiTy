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
    const PlacedPosition placed = FindDrawnPosition(
        generator,
        tables.frame,
        cells,
        draw < 10 ? DrawnPlace::kSideOrCentre : DrawnPlace::kAnywhere
    );
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

// 1,000 positions made of a side or a centre of a cell on each axis, moved by -3 to +3 rounding steps, in a frame whose
// origin and width have no short binary fraction: 27 faces on every axis, in every one.
TEST(FindNearbyFaces, NonDyadicFrameAtSideOrCentre_AlwaysHas27) {
  const GridFrame frame = ToNonDyadicFrame();
  const std::vector<CellId> cells = BuildNonDyadicCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  std::mt19937_64 generator(kNonDyadicSeed);  // NOLINT(bugprone-random-generator-seed)
  for (int draw = 0; draw < 1000; ++draw) {
    CheckEveryAxisHasCount(ToTablesOf(test_grid), FindDrawnNonDyadicPosition(generator, frame), 27);
  }
}

}  // namespace
}  // namespace fluidity::grid
