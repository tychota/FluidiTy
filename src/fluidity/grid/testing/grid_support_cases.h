// Test-only: the helpers that the unit tests and the functional tests of grid_support share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToUnitFrame, ToCell, ApplySmallCellsToList, BuildSplitCellBesideBigCell, ToValue, ToValues, kAxes,
//          ComputeIsKeptWhole, BuildTwoSizesWorstCase, ComputeHasAllSiblings, CheckSiblingsTogether,
//          ComputeDoBoxesTouch, CheckSizeStepByBruteForce, ComputeWordsDigest, ToWord, ApplyCellsToWords,
//          ApplyFacePiecesToWords, BuildMovedCells, DigestGrid, BuildDigestWindows
// Uses:    the headers it includes
// Used by: src/fluidity/grid/grid_support.test.cc and tests/grid/grid_support_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// A frame with its origin at 0 and biggest cells 1 m wide.
inline GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

inline CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// Appends the 8 small cells of big_cell to cells.
inline void ApplySmallCellsToList(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

// The size-level-0 cell {0,0,0} beside the 8 size-level-1 cells of size-level-0 cell {1,0,0}, in a frame of 2
// size levels.
inline std::vector<CellId> BuildSplitCellBesideBigCell() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  ApplySmallCellsToList(ToCell(0, 1, 0, 0), cells);
  return cells;
}

// The value of a cell, or 0 for none: CellId{0} is never a valid cell.
inline std::uint64_t ToValue(const std::optional<CellId>& cell) {
  return cell.value_or(CellId{0}).value;
}

inline std::vector<std::uint64_t> ToValues(const std::vector<CellId>& cells) {
  std::vector<std::uint64_t> values;
  values.reserve(cells.size());
  for (const CellId cell : cells) {
    values.push_back(cell.value);
  }
  return values;
}

constexpr std::array<Axis, 3> kAxes = {Axis::kX, Axis::kY, Axis::kZ};

// Says whether the size-level-1 place at `offset` from cell {2,2,2} stays whole in the two-size case: the cell itself,
// its 8 corner neighbours, its 2 edge neighbours at y - 1 and x +-1, and its 4 edge neighbours at x 0, y +-1, z +-1.
inline bool ComputeIsKeptWhole(CellCoord offset) {
  const bool is_cell = offset.x == 0 && offset.y == 0 && offset.z == 0;
  const bool is_corner = std::abs(offset.x) == 1 && std::abs(offset.y) == 1 && std::abs(offset.z) == 1;
  const bool is_low_y_edge = std::abs(offset.x) == 1 && offset.y == -1 && offset.z == 0;
  const bool is_y_z_edge = offset.x == 0 && std::abs(offset.y) == 1 && std::abs(offset.z) == 1;
  return is_cell || is_corner || is_low_y_edge || is_y_z_edge;
}

// The 6 x 6 x 6 box of size-level-1 places of the two-size case: the places ComputeIsKeptWhole keeps are active, every
// other place is split into its 8 size-level-2 cells.
inline std::vector<CellId> BuildTwoSizesWorstCase() {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 6; ++z) {
    for (std::int32_t y = 0; y < 6; ++y) {
      for (std::int32_t x = 0; x < 6; ++x) {
        if (ComputeIsKeptWhole({.x = x - 2, .y = y - 2, .z = z - 2})) {
          cells.push_back(ToCell(1, x, y, z));
        } else {
          ApplySmallCellsToList(ToCell(1, x, y, z), cells);
        }
      }
    }
  }
  return cells;
}

// ---------------------------------------------------------------------------------------------------------------------
// Random grids
// ---------------------------------------------------------------------------------------------------------------------

// Says whether all 8 small cells of the big cell of `cell` are in `listed`. A cell of size level 0 has no big cell.
inline bool ComputeHasAllSiblings(CellId cell, const std::set<std::uint64_t>& listed) {
  if (static_cast<int>(ToSizeLevel(cell)) == 0) {
    return true;
  }
  const CellId big_cell = ComputeBigCell(cell);
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    if (!listed.contains(ComputeSmallCell(big_cell, cell_eighth).value)) {
      return false;
    }
  }
  return true;
}

// Checks by brute force that every cell is listed once with its siblings: the set of active cells and every bigger
// cell that holds one has, for each cell of size level 1 or more, all 8 small cells of its big cell; and no listed
// cell is a bigger cell of another.
inline ::testing::AssertionResult CheckSiblingsTogether(const std::vector<CellId>& cells) {
  std::set<std::uint64_t> active_values;
  for (const CellId cell : cells) {
    if (!active_values.insert(cell.value).second) {
      return ::testing::AssertionFailure() << "a cell is listed twice";
    }
  }
  // Every bigger cell that holds an active cell is split.
  std::set<std::uint64_t> split_values;
  for (const CellId cell : cells) {
    CellId bigger_cell = cell;
    while (static_cast<int>(ToSizeLevel(bigger_cell)) > 0) {
      bigger_cell = ComputeBigCell(bigger_cell);
      split_values.insert(bigger_cell.value);
    }
  }
  const bool is_split_cell_active = std::ranges::any_of(split_values, [&active_values](std::uint64_t split_value) {
    return active_values.contains(split_value);
  });
  if (is_split_cell_active) {
    return ::testing::AssertionFailure() << "an active cell holds another active cell";
  }
  std::set<std::uint64_t> listed_values = active_values;
  listed_values.insert(split_values.begin(), split_values.end());
  for (const std::uint64_t listed_value : listed_values) {
    if (!ComputeHasAllSiblings(CellId{.value = listed_value}, listed_values)) {
      return ::testing::AssertionFailure() << "a sibling of cell " << listed_value << " is missing";
    }
  }
  return ::testing::AssertionSuccess();
}

// Says whether two boxes share at least one point: their closed intervals meet on every axis.
inline bool ComputeDoBoxesTouch(const UnitBox& first, const UnitBox& second) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_apart = first.low.at(axis) > second.low.at(axis) + second.width
        || second.low.at(axis) > first.low.at(axis) + first.width;
    if (is_apart) {
      return false;
    }
  }
  return true;
}

// Checks by brute force, on every pair of boxes, that every two touching active cells differ by at most one size
// level. The bigger cell of a broken pair is two or more size levels above the other, so only cells below size level
// size_level_count - 2 are paired.
inline ::testing::AssertionResult CheckSizeStepByBruteForce(const std::vector<CellId>& cells, int size_level_count) {
  for (const CellId cell : cells) {
    const int size_level = static_cast<int>(ToSizeLevel(cell));
    if (size_level + 2 >= size_level_count) {
      continue;
    }
    const UnitBox box = ToUnitBox(cell, size_level_count);
    for (const CellId other : cells) {
      const int other_size_level = static_cast<int>(ToSizeLevel(other));
      const bool is_two_smaller = other_size_level >= size_level + 2;
      if (is_two_smaller && ComputeDoBoxesTouch(box, ToUnitBox(other, size_level_count))) {
        return ::testing::AssertionFailure()
            << "cells of size levels " << size_level << " and " << other_size_level << " touch";
      }
    }
  }
  return ::testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------------------------------------------------
// Recorded digests
// ---------------------------------------------------------------------------------------------------------------------
//
// Each test below turns a group of answers into 64-bit words, in the order the answers give them, and compares the
// ComputeResultDigest of those words (64-bit FNV-1a over their bytes, base/testing/base_support.h) with the value the
// test support gave when the test was written. A faster way to find the same answers keeps every digest; a changed
// list, answer or order changes it. The words are read in the machine's byte order, little-endian on every machine the
// project builds for.

inline std::uint64_t ComputeWordsDigest(const std::vector<std::uint64_t>& words) {
  return ComputeResultDigest(std::as_bytes(std::span<const std::uint64_t>(words)));
}

// A signed whole number as a word: its two's-complement bits.
inline std::uint64_t ToWord(std::int64_t number) {
  return static_cast<std::uint64_t>(number);
}

// Appends the length of `cells`, then the value of each cell in order.
inline void ApplyCellsToWords(const std::vector<CellId>& cells, std::vector<std::uint64_t>& words) {
  words.push_back(cells.size());
  for (const CellId cell : cells) {
    words.push_back(cell.value);
  }
}

// Appends the length of `pieces`, then every field of each piece in order, the area by its bits.
inline void ApplyFacePiecesToWords(const std::vector<GeometricFacePiece>& pieces, std::vector<std::uint64_t>& words) {
  words.push_back(pieces.size());
  for (const GeometricFacePiece& piece : pieces) {
    words.push_back(piece.id.value);
    words.push_back(static_cast<std::uint64_t>(piece.axis));
    words.push_back(static_cast<std::uint64_t>(piece.size_level));
    words.push_back(ToWord(piece.cell_high_coord.x));
    words.push_back(ToWord(piece.cell_high_coord.y));
    words.push_back(ToWord(piece.cell_high_coord.z));
    words.push_back(ToValue(piece.cell_low));
    words.push_back(ToValue(piece.cell_high));
    words.push_back(std::bit_cast<std::uint64_t>(piece.area));
  }
}

// Moves every cell by offset_in_biggest_cells size-level-0 cells on each axis: a cell of size level l moves by
// offset_in_biggest_cells x 2^l of its own coordinates.
inline std::vector<CellId> BuildMovedCells(const std::vector<CellId>& cells, std::int32_t offset_in_biggest_cells) {
  std::vector<CellId> moved_cells;
  moved_cells.reserve(cells.size());
  for (const CellId cell : cells) {
    const int size_level = static_cast<int>(ToSizeLevel(cell));
    const auto cells_per_biggest_cell = static_cast<std::int32_t>(1U << static_cast<unsigned>(size_level));
    const std::int32_t offset = offset_in_biggest_cells * cells_per_biggest_cell;
    const CellCoord coord = ToCellCoord(cell);
    moved_cells.push_back(ToCell(size_level, coord.x + offset, coord.y + offset, coord.z + offset));
  }
  return moved_cells;
}

// A list whose checker answers the digest tests read.
struct DigestGrid {
  GridFrame frame;
  std::vector<CellId> cells;
};

// The window quarters the nearby-face digest reads: the 8 windows of quarters 0 and 3, and window (1, 2, 1).
inline std::vector<std::array<int, 3>> BuildDigestWindows() {
  std::vector<std::array<int, 3>> windows = {{1, 2, 1}};
  for (int corner = 0; corner < 8; ++corner) {
    const auto corner_bits = static_cast<unsigned>(corner);
    windows.push_back({
        (corner_bits & 1U) != 0 ? 3 : 0,
        (corner_bits & 2U) != 0 ? 3 : 0,
        (corner_bits & 4U) != 0 ? 3 : 0,
    });
  }
  return windows;
}

}  // namespace fluidity::grid
