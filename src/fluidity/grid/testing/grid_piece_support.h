// Test-only: the tables of a grid piece, built from a whole grid's tables, and the check that every reader gives the
// same answer in the piece as in the whole grid.
//
// Purpose: give the tests of border blocks, and of the grid pieces to come, the definition of a border that shares no
//   code with the border's own functions: a piece that holds a set and its border, built as method paper
//   <sec-cell-blocks> says a grid piece is built, where every reader run from a place kept in the set answers as on the
//   whole grid.
//
// Idea: PieceTables copies the whole grid's tables block by block, for the loaded blocks only, in the whole grid's
//   fixed order. A neighbour entry naming a block that was not loaded becomes kNoIndex. A kBetweenCells face piece
//   whose other cell lies in a block that was not loaded becomes kOutsidePiece. A plane-4 place stays kNotAPiece, as in
//   the whole grid. Nothing else changes, so the piece's tables are those of a grid made of the loaded blocks alone,
//   with indexes of its own. FindFirstReaderDifference runs every reader of a border width from every place kept in the
//   set, on both sets of tables, and compares the answers by what they denote: a place of the piece is turned into
//   the whole grid's place of the same block id and the same place in the block.
//   Example: 2 size levels; the size-level-0 cells {0..3}^3, all active but {0..2}^3, which are split. The set is the
//   one size-level-1 block. A piece of that block alone has no bigger block for it, so FindCellAt finds no cell in
//   it: the piece needs block 0 too.
//
// Owns:    PieceTables, BuildPieceTables, FindBorderByReadMarks, FindFirstReaderDifference
// Uses:    CPU-only guard (base/cpu_only.h), BorderWidth, ApplyCellPlaceReadMarks, ApplyFacePlaceReadMarks
//          (grid/border_block_steps.h), GridTables, GridFrame, CellState,
//          FacePieceKind (grid/grid_tables.h), the id types (grid/grid_ids.h)
// Implementation uses: FindCellAt (grid/find_cell_at.h), FindTouchingCells, FindFacePieces (grid/touching_cells.h),
//          FindCornerPlaces, FindCellsAroundEdge, ComputeCornerPosition (grid/cell_corners.h), FindCellsNearFace,
//          BuildFaceSearch, FindNextFace (grid/face_search.h), FindNearbyFaces, ToMaskOfAll, ComputeNearbyFaceWeights,
//          ToNearbyFaceSums, ComputeFaceWeight (grid/nearby_face_weights.h),
//          FindCellLow, FindCellHigh, ComputeCellCenter, ComputeCellWidth (grid/grid_tables.h)
// Used by: the tests of border blocks (tests/grid/, cell_block_range.test.cc)
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// The tables of a grid piece, owning their arrays, and the whole grid's block of each piece block.
struct PieceTables {
  GridFrame frame{};
  GridVersion version{};
  std::vector<CellBlockId> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks;  // 27 per block.
  std::vector<std::uint32_t> bigger_block;  // 1 per block.
  std::vector<std::uint32_t> smaller_blocks;  // 8 per block.
  std::vector<CellState> cell_states;  // 64 per block.
  std::vector<FacePieceKind> face_piece_kinds;  // 240 per block.
  std::vector<CellBlockIndex> block_in_whole;  // [piece block]: the whole grid's index, ascending.

  // The tables as readers take them. Valid while this object lives and is not changed.
  [[nodiscard]] GridTables ToTables() const;
};

// Builds the tables of the piece made of `loaded_blocks` of `whole`, as method paper <sec-cell-blocks> builds a grid
// piece: block by block in the whole grid's order; a neighbour entry naming a block not loaded becomes kNoIndex; a
// kBetweenCells face piece whose cell_low or cell_high lies in a block not loaded becomes kOutsidePiece; every other
// entry is copied. Example: the whole grid of blocks 0, 1, 2 and the loaded blocks {0, 2}: piece block 1 is whole block
// 2, and an entry
//   of either to whole block 1 is kNoIndex.
// Requires: `whole` are the tables of a CPU grid; loaded_blocks is ascending with no repeat, each below its block
//   count. A list that breaks this fails the calling test and gives empty tables.
PieceTables BuildPieceTables(const GridTables& whole, std::span<const CellBlockIndex> loaded_blocks);

// Finds the border of `set` at `width` by the forward form, as a GPU pass will: every cell place and face place of the
// set marks the blocks its readers read (ApplyCellPlaceReadMarks, ApplyFacePlaceReadMarks, border_block_steps.h), and
// the marked blocks outside the set, in ascending index, are the border.
// Example: in the grid of the file header, the set {block 1} gives a border that starts with block 0.
// Requires: `whole` are the tables of a CPU grid; `set` is ascending and below its block count.
std::vector<CellBlockIndex>
FindBorderByReadMarks(const GridTables& whole, std::span<const CellBlockIndex> set, BorderWidth width);

// Finds the first reader whose answer differs between the whole grid and the piece of `set` plus `border`, run from
// a place kept in `set`; empty when every answer is the same. The readers of `width`, as border_block_steps.h lists
// them:
//   - from each active cell c: FindTouchingCells; FindFacePieces of c and of each cell touching it; FindCornerPlaces;
//     FindCellsAroundEdge of its 12 edges; FindCellAt, with and without a block near it, at the 27 points of c's
//     closed box (low, middle and high on each axis), which FindCellAt gives to c or to a cell touching c; and
//     FindNearbyFaces at 2 positions of c's half-open box (its low corner, and 0.8, 0.3, 0.9 of its width from it);
//   - from each face piece f: FindCellLow; FindCellHigh; FindCornerPlaces; and, for transfer, the cells q of
//     FindCellsNearFace(f) at most one size level bigger than f, which must be the same in the piece and on the whole
//     grid (a block of such a cell that the piece lacks shows here); then, for 3 positions of each q (window quarters
//     (0, 0, 0), (3, 3, 3) and the centre: each axis meets its quarters 0 and 3), the face pieces
//     of the set that FindNearbyFaces lists on each of the three axes, and what f does to collect: ComputeFaceWeight(f,
//     position, q, sums) with the position's sums made on the whole grid, as the transfer step carries them with the
//     particle, which must give the same bits in the piece and on the whole grid. Every face piece of the set is run,
//     so faces of every axis on both sides of a cell q are met. Transfer reads at kTouchingCells, so this part runs at
//     that width only (a piece made for kFourCellWidths holds that piece's blocks as well);
//   - at kFourCellWidths also, from each face piece: FindCellsNearFace; the block list of BuildFaceSearch at the
//     largest radius; and the face pieces of FindNextFace walked from it;
//   - from each corner place: ComputeCornerPosition.
//   FindNearbyFaces from the cells of the set asks one axis per position, the axes taking turns over the cells and
//   the positions (place index plus position number, mod 3), so every axis is asked at a third of the cost. Its list is
//   a filter of FindCellAt, FindTouchingCells and FindFacePieces of the cell and of its touching cells, all compared
//   exhaustively, plus face positions read from the face's own block, so two positions add little; they are not the
//   evidence. Places are compared through the whole grid's indexes: a piece place is mapped by its block's whole grid
//   index.
// Example: in the grid of the file header, the set {block 1} with the border {} gives "FindCellAt ...", and with the
//   border {0} gives no difference.
// Requires: `whole` are the tables of a CPU grid; `set` and `border` are ascending, disjoint, and below its block
//   count.
std::optional<std::string> FindFirstReaderDifference(
    const GridTables& whole,
    std::span<const CellBlockIndex> set,
    std::span<const CellBlockIndex> border,
    BorderWidth width
);

}  // namespace fluidity::grid
