// What code on either processor reads about one grid: its frame and the state of each place.
//
// Purpose: declare the frame that places the grid in the world and the element types of the grid's per-place
//   tables, which the CPU and the GPU both read.
//
// Idea: a grid's frame says where the cell with coordinate (0, 0, 0) starts, how wide the biggest cells are, and how
//   many size levels the grid uses; a cell of size level l is h_0 / 2^l wide. Each cell place of a cell block holds
//   one CellState: an active cell, a split cell, a place covered by an active cell one size level bigger, or no cell.
//   Each face place holds one FacePieceKind, which says whether a face piece is kept there and what lies on its two
//   sides. Every face piece is kept once, at the size level of its smaller cell.
//   Example: an active cell beside four active cells one size level smaller has, on that side, four face places of
//   kind kBetweenCells at the smaller size level, and its own face place there is kNotAPiece.
//
// Words:
//   active cell  a cell that exists and is not split; it carries values
//   face piece   the part of a face between two cells or at the grid's edge, kept at the size level of its smaller
//                cell. Not a grid piece, not a memory piece
//   grid piece   one range of cell blocks of a whole grid with its border, built as a grid of its own
// Owns:    GridFrame, CellState, FacePieceKind
// Uses:    Vec3 (base/vec3.h)
// Used by: the grid's buffers and builders, and every module that reads the grid's tables
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-face-pieces>, <sec-cell-blocks>

#pragma once

#include <cstdint>

#include "fluidity/base/vec3.h"

namespace fluidity::grid {

// Where a grid sits in the world and how big its cells are.
struct GridFrame {
  Vec3 origin{};  // m, world. Low corner of the cell with coordinate (0, 0, 0).
  double biggest_cell_width = 0;  // m, positive. The width h_0 of a cell of size level 0.
  std::uint8_t size_level_count = 0;  // 1..16. Size level size_level_count - 1 holds the smallest cells.
};

// What one cell place of a cell block holds.
enum class CellState : std::uint8_t {
  kAbsent,  // No cell here: outside the part of space the grid keeps.
  kInsideBiggerCell,  // Covered by an active cell one size level bigger.
  kActive,  // A cell that exists and is not split. It carries values.
  kSplit,  // Replaced by its 8 small cells, which all exist.
};

// What one face place of a cell block holds.
enum class FacePieceKind : std::uint8_t {
  kNotAPiece,  // No face piece is kept here, so no flow is counted here.
  kBetweenCells,  // An active cell on each side; the two differ by at most one size level.
  kGridEdgeLow,  // An active cell on the high side only. What lies beyond is not the grid's to say.
  kGridEdgeHigh,  // An active cell on the low side only.
  // In a grid piece only: a kBetweenCells face piece of the whole grid whose other cell lies in a block that the grid
  // piece did not load.
  kOutsidePiece,
};

}  // namespace fluidity::grid
