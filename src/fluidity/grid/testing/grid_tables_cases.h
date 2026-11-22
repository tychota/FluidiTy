// Test-only: the helpers that the unit tests and the functional tests of grid_tables share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToCell, FindFacePiecePlaces, ToCellIdOrZero
// Uses:    the headers it includes
// Used by: src/fluidity/grid/grid_tables.test.cc and tests/grid/grid_tables_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

inline CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The face places of a grid that hold a face piece, in place order.
inline std::vector<FacePlaceIndex> FindFacePiecePlaces(const GridTables& tables) {
  std::vector<FacePlaceIndex> face_piece_places;
  const std::uint64_t face_place_count = std::uint64_t{tables.block_count} * kFacePlacesPerBlock;
  for (std::uint64_t face_place = 0; face_place < face_place_count; ++face_place) {
    if (tables.face_piece_kinds[face_place] != FacePieceKind::kNotAPiece) {
      face_piece_places.push_back({.value = static_cast<std::uint32_t>(face_place)});
    }
  }
  return face_piece_places;
}

// The id of the active cell at a place, or CellId{0} for kNoIndex: what FindCellLow and FindCellHigh name.
inline CellId ToCellIdOrZero(const GridTables& tables, CellPlaceIndex cell) {
  return cell.value == kNoIndex ? CellId{0} : ToCellId(tables, cell);
}

}  // namespace fluidity::grid
