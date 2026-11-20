// What code on either processor reads about one grid: its frame, the state of each place, and the tables that hold
// them.
//
// Purpose: declare the frame that places the grid in the world, the element types of the grid's per-place tables,
//   and GridTables, the one way code on the CPU or the GPU reads the tables of one grid version.
//
// Idea: a grid's frame says where the cell with coordinate (0, 0, 0) starts, how wide the biggest cells are, and how
//   many size levels the grid uses; a cell of size level l is h_0 / 2^l wide. Each cell place of a cell block holds
//   one CellState: an active cell, a split cell, a place covered by an active cell one size level bigger, or no cell.
//   Each face place holds one FacePieceKind, which says whether a face piece is kept there and what lies on its two
//   sides. Every face piece is kept once, at the size level of its smaller cell. Beside the per-place tables, each
//   block lists the blocks around it: its 27 neighbours of the same size level, the block of the next bigger cells
//   that covers it, and the 8 blocks of the next smaller cells that it covers. A search is a binary search in the
//   block ids followed by moves through these tables.
//   Example: an active cell beside four active cells one size level smaller has, on that side, four face places of
//   kind kBetweenCells at the smaller size level, and its own face place there is kNotAPiece.
//
// Words:
//   active cell     a cell that exists and is not split; it carries values
//   face piece      the part of a face between two cells or at the grid's edge, kept at the size level of its smaller
//                   cell. Not a grid piece, not a memory piece
//   grid piece      one range of cell blocks of a whole grid with its border, built as a grid of its own
//   bigger block    of a block of size level l, the block of size level l - 1 whose cells cover it
//   smaller blocks  of a block of size level l, the 8 blocks of size level l + 1 that its cells cover, one per
//                   eighth of the block
// Owns:    GridFrame, CellState, FacePieceKind, GridTables
// Uses:    Vec3 (base/vec3.h), CellBlockId, CellBlockIndex, GridVersion, kNoIndex (grid/grid_ids.h)
// Used by: the grid's buffers and builders, and every module that reads the grid's tables
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-face-pieces>, <sec-cell-blocks>

#pragma once

#include <cstdint>

#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_ids.h"

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

// The tables of one grid version on one processor, as raw pointers, so shared code on either processor reads them. A
// GPU function takes it by value. It owns nothing: the arrays live in the grid's buffers, whose room may hold more
// blocks than block_count, and every array is read only for blocks 0..block_count - 1.
// Layouts, by block index b (CellBlockIndex), offset (dx, dy, dz) with each of -1, 0, +1, and cell eighth e:
//   same_size_level_blocks[b * 27 + ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)], entry 13 the block itself;
//   bigger_block[b]; smaller_blocks[b * 8 + e]; cell_states[64 b + (4 z + y) * 4 + x];
//   face_piece_kinds[240 b + 80 axis + the place on that axis's 5 planes of 4 x 4]: (z * 4 + y) * 5 + x for X,
//   (z * 5 + y) * 4 + x for Y, (z * 4 + y) * 4 + x for Z, where the coordinate along the axis is the plane 0..4 and
//   plane p is the low side of the cell place p.
// Every block index entry is kNoIndex where no such block exists.
// A cell place that is kInsideBiggerCell or kAbsent may lie outside -kMaxCoord..kMaxCoord at its size level (only
// -2^18, local 0 of a block at block coordinate -2^16 on that axis), so an id is made from a cell place only when it
// is kActive or kSplit.
struct GridTables {
  GridFrame frame{};
  GridVersion version{};  // 1 or more for a built grid. 0 with null arrays when the processor holds no tables.
  std::uint32_t block_count = 0;  // The blocks of this version.
  const CellBlockId* block_ids = nullptr;  // block_count ids, in the fixed order.
  // 27 per block: the blocks of the same size level around it, by offset in z, then y, then x.
  const std::uint32_t* same_size_level_blocks = nullptr;
  const std::uint32_t* bigger_block = nullptr;  // 1 per block; kNoIndex at size level 0.
  // 8 per block: the smaller blocks, by the eighth of the block they lie in, x + 2 y + 4 z of its low or high halves.
  const std::uint32_t* smaller_blocks = nullptr;
  const CellState* cell_states = nullptr;  // 64 per block, one per cell place.
  const FacePieceKind* face_piece_kinds = nullptr;  // 240 per block, one per face place.
};

}  // namespace fluidity::grid
