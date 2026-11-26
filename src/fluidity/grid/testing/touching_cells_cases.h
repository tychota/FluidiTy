// Test-only: the helpers that the unit tests and the functional tests of touching_cells share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToListedPlaces, ComputeIsStrictlyAscending, ToCheckerTouchingPlaces,
//          ToCheckerFacePiecesByCell, ToAscendingFacePlaces, CellSurfaceSums, ComputeCellSurfaceSums,
//          ActiveCellReaderMistakes, LargestCounts, FindActiveCellReaderMistakes, CheckNoMistake
// Uses:    the headers it includes
// Used by: src/fluidity/grid/touching_cells.test.cc and tests/grid/touching_cells_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {

// The listed places of a result's array, entries 0..count - 1, as plain numbers.
template <typename PlaceArray>
std::vector<std::uint32_t> ToListedPlaces(const PlaceArray& places, int count) {
  std::vector<std::uint32_t> listed_places;
  listed_places.reserve(static_cast<std::size_t>(count));
  for (int entry = 0; entry < count; ++entry) {
    listed_places.push_back(places[entry].value);
  }
  return listed_places;
}

inline bool ComputeIsStrictlyAscending(const std::vector<std::uint32_t>& places) {
  return std::ranges::adjacent_find(places, std::ranges::greater_equal{}) == places.end();
}

// The checker's touching cells of `cell`, as places of the tables in ascending order. A cell not found gives kNoIndex,
// which no reader lists, so the comparison fails.
inline std::vector<std::uint32_t>
ToCheckerTouchingPlaces(const GridTables& tables, const GeometricGrid& checker, CellId cell) {
  std::vector<std::uint32_t> touching_places;
  for (const CellId touching_cell : checker.TouchingCells(cell)) {
    touching_places.push_back(FindCellPlace(tables, touching_cell).value);
  }
  std::ranges::sort(touching_places);
  return touching_places;
}

// The checker's face pieces of each active cell, the pieces of which it is cell_low or cell_high, by cell id.
inline std::map<std::uint64_t, std::vector<FacePieceId>> ToCheckerFacePiecesByCell(const GeometricGrid& checker) {
  std::map<std::uint64_t, std::vector<FacePieceId>> face_pieces_by_cell;
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    if (face_piece.cell_low.has_value()) {
      face_pieces_by_cell[face_piece.cell_low->value].push_back(face_piece.id);
    }
    if (face_piece.cell_high.has_value()) {
      face_pieces_by_cell[face_piece.cell_high->value].push_back(face_piece.id);
    }
  }
  return face_pieces_by_cell;
}

// The face places of face piece ids, in ascending order. An id not found gives kNoIndex, which no reader lists.
inline std::vector<std::uint32_t>
ToAscendingFacePlaces(const GridTables& tables, const std::vector<FacePieceId>& face_piece_ids) {
  std::vector<std::uint32_t> face_places;
  face_places.reserve(face_piece_ids.size());
  for (const FacePieceId face_piece_id : face_piece_ids) {
    face_places.push_back(FindFacePlace(tables, face_piece_id).value);
  }
  std::ranges::sort(face_places);
  return face_places;
}

// The sums of FindFacePieces over one cell's face pieces: the area on each of its six sides, side 2 axis for the low
// side and 2 axis + 1 for the high side, and the sum of +area axis where the cell is cell_low and -area axis where it
// is cell_high. A face piece of which the cell is neither cell is counted apart.
struct CellSurfaceSums {
  std::array<double, 6> side_areas{};  // m2.
  std::array<double, 3> area_vector{};  // m2, indexed by axis (x, y, z).
  std::array<double, 3> absolute_area_sums{};  // m2, indexed by axis: the sum of the face areas, all counted positive.
  int foreign_face_count = 0;
};

inline CellSurfaceSums
ComputeCellSurfaceSums(const GridTables& tables, CellPlaceIndex cell, const CellFacePieces& face_pieces) {
  CellSurfaceSums sums;
  for (int entry = 0; entry < face_pieces.count; ++entry) {
    const FacePlaceIndex face = face_pieces.faces[entry];
    const auto axis = static_cast<std::size_t>(ToAxis(face));
    const double face_area = ComputeFaceArea(tables, face);
    // The cell is cell_low of the face pieces on its high side, cell_high of those on its low side.
    const bool is_on_high_side = FindCellLow(tables, face).value == cell.value;
    const bool is_on_low_side = FindCellHigh(tables, face).value == cell.value;
    if (is_on_high_side == is_on_low_side) {
      ++sums.foreign_face_count;
      continue;
    }
    sums.side_areas.at((2 * axis) + (is_on_high_side ? 1 : 0)) += face_area;
    sums.area_vector.at(axis) += is_on_high_side ? face_area : -face_area;
    sums.absolute_area_sums.at(axis) += face_area;
  }
  return sums;
}

// The largest over the three axes of |sum of the signed areas| / (eps * sum of the absolute areas) of a cell's face
// pieces; 0 when an axis has no face piece.
inline double ComputeAreaSumRoundingRatio(const CellSurfaceSums& sums) {
  double largest_ratio = 0.0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double absolute_area_sum = sums.absolute_area_sums.at(axis);
    if (absolute_area_sum > 0.0) {
      const double sum_ratio =
          std::abs(sums.area_vector.at(axis)) / (std::numeric_limits<double>::epsilon() * absolute_area_sum);
      largest_ratio = std::max(largest_ratio, sum_ratio);
    }
  }
  return largest_ratio;
}

// What one grid's active cells got wrong, property by property: a count of cells each.
struct ActiveCellReaderMistakes {
  std::size_t touching_cells_not_the_checkers = 0;  // Not the checker's places in ascending index order.
  std::size_t touching_cells_out_of_order = 0;  // Not strictly ascending.
  std::size_t touching_entries_past_count_not_no_index = 0;
  std::size_t face_pieces_not_the_checkers = 0;  // Not the checker's face places in ascending index order.
  std::size_t face_piece_ids_not_the_checkers = 0;  // ToFacePieceId of the list, as a set.
  std::size_t face_pieces_out_of_order = 0;
  std::size_t face_entries_past_count_not_no_index = 0;
  std::size_t sides_not_covered_once = 0;  // A side whose face piece areas do not sum to the cell's face area.
  std::size_t surfaces_not_closed = 0;  // Signed areas that do not sum to exactly {0, 0, 0}, or a face of another cell.
  // Signed areas whose sum on an axis is above kClosedSurfaceRoundingFactor * eps * (sum of the absolute areas), or a
  // face of another cell.
  std::size_t surfaces_beyond_rounding_bound = 0;
};

// The signed face areas around a cell are added in list order. When the frame's width is no power of two, the areas
// of faces of different sizes are rounded products of widths, so their sum is zero only up to rounding. Each area
// carries a relative error of at most eps and each addition adds at most eps of the running sum, which is at most
// the sum of the absolute areas. The sum is held to 4 * eps * (sum of the absolute areas), per axis. Where the frame
// is a power of two every area is exact and the sum is exactly zero, which the tests of those grids demand.
inline constexpr double kClosedSurfaceRoundingFactor = 4.0;

// The largest lists the grids gave, to show that bigger and smaller cells were met.
struct LargestCounts {
  int touching_cells = 0;
  int face_pieces = 0;
  // The largest |sum of the signed areas on an axis| / (eps * sum of the absolute areas on that axis) met. The
  // bound is kClosedSurfaceRoundingFactor.
  double area_sum_rounding_ratio = 0.0;
};

// Counts into `mistakes` whether the six sides of a cell are each covered once by its face pieces and whether its
// surface is closed, exactly and up to rounding, and keeps the largest rounding ratio in `largest_counts`.
inline void ApplySurfaceMistakes(
    const CellSurfaceSums& sums,
    double cell_width,
    ActiveCellReaderMistakes& mistakes,
    LargestCounts& largest_counts
) {
  const double cell_face_area = cell_width * cell_width;
  const bool is_every_side_covered = std::ranges::all_of(sums.side_areas, [cell_face_area](double side_area) {
    return side_area == cell_face_area;
  });
  mistakes.sides_not_covered_once += is_every_side_covered ? 0 : 1;
  const bool is_closed = sums.foreign_face_count == 0 && std::ranges::all_of(sums.area_vector, [](double area_sum) {
                           return area_sum == 0.0;
                         });
  mistakes.surfaces_not_closed += is_closed ? 0 : 1;
  const double cell_rounding_ratio = ComputeAreaSumRoundingRatio(sums);
  largest_counts.area_sum_rounding_ratio = std::max(largest_counts.area_sum_rounding_ratio, cell_rounding_ratio);
  const bool is_within_rounding_bound =
      sums.foreign_face_count == 0 && cell_rounding_ratio <= kClosedSurfaceRoundingFactor;
  mistakes.surfaces_beyond_rounding_bound += is_within_rounding_bound ? 0 : 1;
}

// Checks FindTouchingCells and FindFacePieces at every active cell of one grid against the checker.
inline ActiveCellReaderMistakes FindActiveCellReaderMistakes(
    const GridTables& tables,
    const GeometricGrid& checker,
    const std::vector<CellId>& cells,
    LargestCounts& largest_counts
) {
  ActiveCellReaderMistakes mistakes;
  const std::map<std::uint64_t, std::vector<FacePieceId>> checker_pieces_by_cell = ToCheckerFacePiecesByCell(checker);
  for (const CellId cell_id : cells) {
    const CellPlaceIndex cell = FindCellPlace(tables, cell_id);
    // The touching cells against the checker's, as a set and in order.
    const TouchingCells touching = FindTouchingCells(tables, cell);
    const std::vector<std::uint32_t> touching_places = ToListedPlaces(touching.cells, touching.count);
    mistakes.touching_cells_not_the_checkers +=
        touching_places == ToCheckerTouchingPlaces(tables, checker, cell_id) ? 0 : 1;
    mistakes.touching_cells_out_of_order += ComputeIsStrictlyAscending(touching_places) ? 0 : 1;
    mistakes.touching_entries_past_count_not_no_index +=
        ComputeHasNoIndexPastCount(touching.cells, touching.count) ? 0 : 1;
    largest_counts.touching_cells = std::max(largest_counts.touching_cells, touching.count);
    // The face pieces against the checker's, as places, as ids and in order.
    const CellFacePieces face_pieces = FindFacePieces(tables, cell);
    const std::vector<std::uint32_t> face_places = ToListedPlaces(face_pieces.faces, face_pieces.count);
    const std::vector<FacePieceId>& checker_pieces = checker_pieces_by_cell.at(cell_id.value);
    mistakes.face_pieces_not_the_checkers += face_places == ToAscendingFacePlaces(tables, checker_pieces) ? 0 : 1;
    std::set<std::uint64_t> listed_ids;
    for (int entry = 0; entry < face_pieces.count; ++entry) {
      listed_ids.insert(ToFacePieceId(tables, face_pieces.faces[entry]).value);
    }
    std::set<std::uint64_t> checker_ids;
    for (const FacePieceId face_piece_id : checker_pieces) {
      checker_ids.insert(face_piece_id.value);
    }
    mistakes.face_piece_ids_not_the_checkers += listed_ids == checker_ids ? 0 : 1;
    mistakes.face_pieces_out_of_order += ComputeIsStrictlyAscending(face_places) ? 0 : 1;
    mistakes.face_entries_past_count_not_no_index +=
        ComputeHasNoIndexPastCount(face_pieces.faces, face_pieces.count) ? 0 : 1;
    largest_counts.face_pieces = std::max(largest_counts.face_pieces, face_pieces.count);
    // The six sides and the closed surface.
    const CellSurfaceSums sums = ComputeCellSurfaceSums(tables, cell, face_pieces);
    ApplySurfaceMistakes(sums, ComputeCellWidth(tables.frame, ToSizeLevel(cell_id)), mistakes, largest_counts);
  }
  return mistakes;
}

// Expects no cell of `mistake_count`, under a trace that names the property.
inline void CheckNoMistake(const char* property, std::size_t mistake_count) {
  SCOPED_TRACE(property);
  EXPECT_EQ(mistake_count, 0U);
}

}  // namespace fluidity::grid
