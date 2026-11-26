#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/touching_cells_cases.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

// FindTouchingCells and FindFacePieces at every active cell of the grids of `built`: the checker's cells and face
// pieces, as sets and in ascending index order, within their bounds, every side covered once by its face pieces and
// every cell's surface closed. It checks these nine properties in one function because each TEST runs in a process of
// its own and would build the grids again. Gives the largest counts met.
//
// The surface of a cell is closed when its signed face areas sum to zero on each axis. The areas of faces of
// different sizes are products of widths. In a frame whose width is a power of two they are exact and the sum is
// exactly zero. In any other frame each area is rounded, so the sum is zero only up to rounding (method paper
// <sec-face-pieces>): the check is then |sum| <= 4 * eps * (sum of the absolute areas) on each axis, where each
// area carries a relative error of a few eps and each addition at most eps of a partial sum that never exceeds the
// sum of the absolute areas.
enum class SurfaceClosure : std::uint8_t {
  kExactlyZero,
  kZeroUpToRounding,
};

LargestCounts
CheckActiveCellReadersMatchGeometricChecker(const RandomGridsWithCheckers& built, SurfaceClosure surface_closure) {
  EXPECT_EQ(built.grids.size(), built.cell_lists.size());
  LargestCounts largest_counts;
  for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    const ActiveCellReaderMistakes mistakes = FindActiveCellReaderMistakes(
        built.grids.at(grid).grid.tables(RunOn::kCpu),
        built.checkers.at(grid),
        built.cell_lists.at(grid),
        largest_counts
    );
    CheckNoMistake(
        "FindTouchingCells: the checker's places in ascending order",
        mistakes.touching_cells_not_the_checkers
    );
    CheckNoMistake("FindTouchingCells: strictly ascending", mistakes.touching_cells_out_of_order);
    CheckNoMistake("FindTouchingCells: kNoIndex from count on", mistakes.touching_entries_past_count_not_no_index);
    CheckNoMistake("FindFacePieces: the checker's places in ascending order", mistakes.face_pieces_not_the_checkers);
    CheckNoMistake("FindFacePieces: the checker's ids as a set", mistakes.face_piece_ids_not_the_checkers);
    CheckNoMistake("FindFacePieces: strictly ascending", mistakes.face_pieces_out_of_order);
    CheckNoMistake("FindFacePieces: kNoIndex from count on", mistakes.face_entries_past_count_not_no_index);
    CheckNoMistake("EveryCellSide_IsCoveredOnce", mistakes.sides_not_covered_once);
    if (surface_closure == SurfaceClosure::kExactlyZero) {
      CheckNoMistake("ClosedSurface_AreasCancel", mistakes.surfaces_not_closed);
    } else {
      CheckNoMistake("ClosedSurface_AreasCancelUpToRounding", mistakes.surfaces_beyond_rounding_bound);
    }
  }
  return largest_counts;
}

// Within the bounds, and beyond what cells of one size give (26 touching cells, 6 face pieces), so split places
// beside a cell were met.
void CheckCountsAreWithinBoundsAndBeyondOneSize(const LargestCounts& largest_counts) {
  EXPECT_LE(largest_counts.touching_cells, kMaxTouchingCells);
  EXPECT_GT(largest_counts.touching_cells, 26);
  EXPECT_LE(largest_counts.face_pieces, kMaxCellFacePieces);
  EXPECT_GT(largest_counts.face_pieces, 6);
}

TEST(GridTables, RandomGrids_EveryActiveCellReaderMatchesGeometricChecker) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kRandomGridCount);
  CheckCountsAreWithinBoundsAndBeyondOneSize(
      CheckActiveCellReadersMatchGeometricChecker(built, SurfaceClosure::kExactlyZero)
  );
}

// The same on the framed random grids: origins and widths that are no power of two, negative and distant cell
// coordinates, 2 to 6 size levels.
TEST(GridTables, FramedRandomGrids_EveryActiveCellReaderMatchesGeometricChecker) {
  const RandomGridsWithCheckers& built = FindFramedRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kFramedRandomGridCount);
  const LargestCounts largest_counts =
      CheckActiveCellReadersMatchGeometricChecker(built, SurfaceClosure::kZeroUpToRounding);
  // The measured |sum| / (eps * sum of the absolute areas); the bound is 4.
  RecordProperty("largest_area_sum_rounding_ratio", std::to_string(largest_counts.area_sum_rounding_ratio));
  CheckCountsAreWithinBoundsAndBeyondOneSize(largest_counts);
}

}  // namespace
}  // namespace fluidity::grid
