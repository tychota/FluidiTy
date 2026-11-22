// Test-only: the helpers that the unit tests and the functional tests of face_search share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToUnitFrame, ToCell, ToInputs, ToTablesOf, AppendSmallCells, ToListedPlaces, ComputeHasNoIndexPastCount,
//          FindGivenFaces, FindWalkedFaces, BuildCellWithEveryNeighbourSplit, BuildBiggerBlockBehindAbsentNeighbour,
//          BuildThreeSizeCells, BuildBoxOfCells, BuildUniformBox, BuildSameSizePairWithEveryNeighbourSplit,
//          ComputeLargestCellsNearFace
// Uses:    the headers it includes
// Used by: src/fluidity/grid/face_search.test.cc and tests/grid/face_search_on_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {

// A frame with its origin at 0 and biggest cells 1 m wide.
constexpr GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

inline CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The inputs of a CPU list of version 1.
inline GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

inline GridTables ToTablesOf(const std::expected<TestGrid, GridFailure>& test_grid) {
  return test_grid->grid.tables(RunOn::kCpu);
}

// Appends the 8 small cells of `big_cell`, in cell eighth order.
inline void AppendSmallCells(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

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

// Says whether every entry from `count` on is kNoIndex: the bytes past the list are fixed.
template <typename PlaceArray>
bool ComputeHasNoIndexPastCount(const PlaceArray& places, int count) {
  for (auto entry = static_cast<std::size_t>(count); entry < std::size(places); ++entry) {
    if (places[entry].value != kNoIndex) {
      return false;
    }
  }
  return true;
}

// The face pieces FindNextFace gives from `search` until kNoIndex, in order. After more calls than the face places of
// 270 blocks the walk stops and ends with kNoIndex, so a walk that never ends fails a comparison and never hangs.
inline std::vector<std::uint32_t> FindGivenFaces(const GridTables& tables, FaceSearch search) {
  std::vector<std::uint32_t> given_faces;
  constexpr int kMaxCalls = (kMaxFaceSearchBlocks * kFacePlacesPerBlock) + 1;
  for (int call = 0; call < kMaxCalls; ++call) {
    const FacePlaceIndex next_face = FindNextFace(tables, &search);
    if (next_face.value == kNoIndex) {
      return given_faces;
    }
    given_faces.push_back(next_face.value);
  }
  given_faces.push_back(kNoIndex);
  return given_faces;
}

// The face pieces a face search from `face` walks, in the order FindNextFace gives them (FindGivenFaces).
inline std::vector<std::uint32_t>
FindWalkedFaces(const GridTables& tables, FacePlaceIndex face, double radius_in_cell_widths) {
  return FindGivenFaces(tables, BuildFaceSearch(tables, face, radius_in_cell_widths));
}

// The size-level-0 cell {x, y, z} with each of the 26 size-level-0 cells around it split, 2 size levels.
inline std::vector<CellId> BuildCellWithEveryNeighbourSplit(std::int32_t x, std::int32_t y, std::int32_t z) {
  std::vector<CellId> cells = {ToCell(0, x, y, z)};
  for (std::int32_t offset_z = -1; offset_z <= 1; ++offset_z) {
    for (std::int32_t offset_y = -1; offset_y <= 1; ++offset_y) {
      for (std::int32_t offset_x = -1; offset_x <= 1; ++offset_x) {
        if (offset_x != 0 || offset_y != 0 || offset_z != 0) {
          AppendSmallCells(ToCell(0, x + offset_x, y + offset_y, z + offset_z), cells);
        }
      }
    }
  }
  return cells;
}

// Size level 0 {3,0,0} split into the size-level-1 cells of block (1,0,0), beside the size-level-0 cells {4..6,0,0}
// of block (1,0,0), 2 size levels. The face pieces between {7,y,z} of size level 1 and {4,0,0} are kept at plane 4 of
// the size-level-1 block (1,0,0), as no size-level-1 block lies above it. The size-level-0 face piece between {4,0,0}
// and {5,0,0}, 1 m = 2 widths away, lies in the size-level-0 block (1,0,0), which is the bigger block of no
// size-level-1 block of the grid: only the position of the bigger block leads to it.
inline std::vector<CellId> BuildBiggerBlockBehindAbsentNeighbour() {
  std::vector<CellId> cells = {ToCell(0, 4, 0, 0), ToCell(0, 5, 0, 0), ToCell(0, 6, 0, 0)};
  AppendSmallCells(ToCell(0, 3, 0, 0), cells);
  return cells;
}

// Three size levels: size level 0 {0,0,0}; the size-level-1 cells of {1,0,0} with x = 2 active, and those with x = 3
// split into size-level-2 cells, which touch no size-level-0 cell. Face pieces of size level 2 lie within 1.5 widths of
// the big cell's face pieces and are left out.
inline std::vector<CellId> BuildThreeSizeCells() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellId small_cell = ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth);
    if (ToCellCoord(small_cell).x == 2) {
      cells.push_back(small_cell);
    } else {
      AppendSmallCells(small_cell, cells);
    }
  }
  return cells;
}

// The size-level-0 cells of the box low..high on each axis.
//
// clang-tidy: the two ends of one range; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline std::vector<CellId> BuildBoxOfCells(std::int32_t low, std::int32_t high) {
  std::vector<CellId> cells;
  for (std::int32_t z = low; z <= high; ++z) {
    for (std::int32_t y = low; y <= high; ++y) {
      for (std::int32_t x = low; x <= high; ++x) {
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  return cells;
}

// The size-level-0 cells of the box -2..2 on each axis: face pieces at block borders in every direction, and at the
// grid's edge.
inline std::vector<CellId> BuildUniformBox() {
  return BuildBoxOfCells(-2, 2);
}

// Two size-level-0 cells of one size beside each other, {0,0,0} and {1,0,0}, with all 34 size-level-0 places around
// the pair split, 2 size levels: the largest count of cells near a face known, 82, at the X face piece between them.
inline std::vector<CellId> BuildSameSizePairWithEveryNeighbourSplit() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0)};
  for (std::int32_t z = -1; z <= 1; ++z) {
    for (std::int32_t y = -1; y <= 1; ++y) {
      for (std::int32_t x = -1; x <= 2; ++x) {
        const bool is_pair_cell = y == 0 && z == 0 && (x == 0 || x == 1);
        if (!is_pair_cell) {
          AppendSmallCells(ToCell(0, x, y, z), cells);
        }
      }
    }
  }
  return cells;
}

// Computes the largest count FindCellsNearFace gives over every face piece of the tables.
inline int ComputeLargestCellsNearFace(const GridTables& tables) {
  int largest_count = 0;
  const std::uint32_t face_place_count = tables.block_count * static_cast<std::uint32_t>(kFacePlacesPerBlock);
  for (std::uint32_t face_place = 0; face_place < face_place_count; ++face_place) {
    if (tables.face_piece_kinds[face_place] != FacePieceKind::kNotAPiece) {
      largest_count = std::max(largest_count, FindCellsNearFace(tables, FacePlaceIndex{face_place}).count);
    }
  }
  return largest_count;
}

}  // namespace fluidity::grid
