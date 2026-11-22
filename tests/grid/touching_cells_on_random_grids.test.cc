#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/touching_cells_cases.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

// FindTouchingCells and FindFacePieces at every active cell of the 200 random grids: the checker's cells and face
// pieces, as sets and in ascending index order, within their bounds, every side covered once by its face pieces and
// every cell's surface closed. It checks these nine properties in one TEST because each TEST runs in a process of its
// own and would build the 200 grids again.
TEST(GridTables, RandomGrids_EveryActiveCellReaderMatchesGeometricChecker) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kRandomGridCount);
  LargestCounts largest_counts;
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    const ActiveCellReaderMistakes mistakes = FindActiveCellReaderMistakes(
        built.grids.at(grid).grid.tables(RunOn::kCpu),
        built.checkers.at(grid),
        BuildShuffledRandomGrids().at(grid),
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
    CheckNoMistake("ClosedSurface_AreasCancel", mistakes.surfaces_not_closed);
  }
  // Within the bounds, and beyond what cells of one size give (26 touching cells, 6 face pieces), so split places
  // beside a cell were met.
  EXPECT_LE(largest_counts.touching_cells, kMaxTouchingCells);
  EXPECT_GT(largest_counts.touching_cells, 26);
  EXPECT_LE(largest_counts.face_pieces, kMaxCellFacePieces);
  EXPECT_GT(largest_counts.face_pieces, 6);
}

}  // namespace
}  // namespace fluidity::grid
