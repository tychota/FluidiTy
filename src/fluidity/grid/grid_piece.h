// A grid piece: one range of a whole grid with its border, loaded for a batch, as a small grid of its own.
//
// Purpose: let a run in batches work on a few blocks of a big grid at a time, in memory that was allocated once, with
//   every stage getting from the piece the answers it would get from the whole grid.
//
// Idea: a stage writes values kept in the blocks of a range and may read values kept in the range and in its border
//   blocks (cell_block_range.h). A grid piece holds those blocks and no others. Its blocks are listed in the whole
//   grid's fixed order, never the range first, so the range is a run of the piece, and a sum over blocks visits them
//   in the order the whole grid does. The piece's tables are the whole grid's, block by block: the block ids, the cell
//   states and the face piece kinds are copied; every neighbour entry is renumbered for the piece, and reads as none
//   when it names a block that was not loaded; a face piece between two cells whose other cell lies in a block not
//   loaded becomes kOutsidePiece. A place of the last plane of a block stays kNotAPiece when the whole grid's block
//   beyond it was not loaded, so the piece never has a face piece that the whole grid does not have. The piece map
//   says, for each block of the piece, its index in the whole grid.
//   What a piece answers. Every reader of the border's width (BorderWidth, border_block_steps.h), called from a place
//   kept in the range, answers as on the whole grid: FindTouchingCells, FindFacePieces, FindCornerPlaces,
//   FindCellsAroundEdge, FindCellAt for a position that gives a cell of the range, FindNearbyFaces for a position
//   that FindCellAt gives to a cell of the range, FindCellLow and FindCellHigh for a face piece of the range,
//   ComputeCornerPosition for a corner place of the range; at kFourCellWidths also the face search from a face piece
//   of the range. FindCellsNearFace from a face piece of the range answers as the whole grid at kFourCellWidths; at
//   kTouchingCells the near border covers only the cells of its list that are at most one size level bigger than the
//   face piece, so the piece's list may miss other cells the whole grid's list holds, and a caller that counts the
//   cells or sums over them needs kFourCellWidths for the full list. A reader called for a place outside that rule
//   returns none, as at the edge of the grid, and a stage cannot tell the two apart. FindCellAt for a position outside
//   the range's cells may give a border cell, when the piece was built with a border that reaches it, or kNoIndex when
//   no loaded block holds it; FindNearbyFaces then says kPositionOutsideGrid for the second. A stage relies on neither
//   answer. So a stage calls readers only from places of its range, keeps every particle in a cell of its range as
//   FindCellAt gives the cell, and reads a border place only when a layer of a stacked border covers it. A cell owns
//   its low sides and not its high sides: a position on the high side of a cell of the range belongs to the next cell,
//   which may be a border cell, and FindNearbyFaces for it lists fewer face pieces with no error. A caller can check a
//   particle with ToCellBlockIndex(list.cell) against range(). A corner place has no reader to the cells
//   that have it as a corner.
//   Sums over the grid. In a batched run, each block's partial is written into the whole grid's block values (a
//   CellBlockValues of the whole grid) at its whole-grid index, which the piece map gives; one tree sum then runs over
//   all the block partials after the last batch (cell_block_range.h). A tree sum inside each piece, added up
//   afterwards, does not give the whole grid's total bit for bit: the tree pairs partials by position.
//   Mapping a place back. A cell place or a face place of the piece is in block p / N, where N is the places per block
//   (kCellPlacesPerBlock or kFacePlacesPerBlock). Its whole-grid place is block_in_whole_grid()[p / N] * N + p % N for
//   a CPU piece: the piece map gives the block's whole-grid index and the place within the block is unchanged. A GPU
//   piece keeps its map on the GPU (block_in_whole_grid_buffer()), and block_in_whole_grid() is empty for it. Corner
//   places of the range map the same way. Corner places of border blocks do not map through the piece map when the
//   whole grid stores that corner in a block that was not loaded. There is no corner id: match a corner of a piece
//   with a corner of the whole grid by position (ComputeCornerPosition of each place gives the same bits), or
//   through the piece map where the corner's block is loaded (method paper <sec-cell-blocks>, "A corner place
//   belongs to one set of tables").
//   Versions. A piece has the whole grid's version and its own block count, and values are checked against those two.
//   Two pieces of one whole grid with the same block count cannot be told apart by that check, so values bound to one
//   piece must not be read through another: the caller keeps them apart and binds again at every load.
//   Example: a whole grid of blocks 0..9, the range {4, 3} and the border {1, 2, 8}: the piece map is
//   [1, 2, 4, 5, 6, 8], range() is {2, 3} and range_in_whole_grid() is {4, 3}.
//
// Words:
//   grid piece  one range of a whole grid with its border, loaded for a batch
//   piece map   for each block of a grid piece, its index in the whole grid
// Owns:    GridPiece, BuildGridPiece (two forms), TakeBuffers (of a GridPiece)
// Uses:    CPU-only guard (base/cpu_only.h), Buffer (base/buffer.h), GridBuffers, StaggeredGrid
//          (grid/staggered_grid.h), GridFailure (grid/grid_error.h), CellBlockIndex, CellBlockRange (grid/grid_ids.h)
// Implementation uses: ToPiecePlace, ToPieceNeighbour, ComputePieceFaceKind (grid/grid_piece_steps.h),
//          ToGridBuffersArrays, BuildGridFromBuffers (grid/grid_buffers_internal.h), the GPU functions of
//          grid/grid_piece_gpu.h
// Used by: the batched run of the cell resize and of the pressure solve, and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>
#include <span>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

// One range of a whole grid with its border, as a grid of its own. Move-only; owns its GridBuffers, which
// TakeBuffers gives back for the next piece.
class GridPiece {
 public:
  GridPiece(const GridPiece&) = delete;
  GridPiece& operator=(const GridPiece&) = delete;
  GridPiece(GridPiece&& other) noexcept = default;
  GridPiece& operator=(GridPiece&& other) noexcept = default;
  ~GridPiece() = default;

  // The loaded blocks as a grid. Its version() is the whole grid's version, and so is that of every other piece of
  // the same whole grid: a version check cannot tell two pieces apart that have the same block count. Values bound to
  // one piece must not be read through another; the caller keeps them apart.
  [[nodiscard]] const StaggeredGrid& grid() const {
    return grid_;
  }

  // The range, as a run of the blocks of grid().
  [[nodiscard]] CellBlockRange range() const {
    return range_in_piece_;
  }

  // The same blocks, as indexes of the whole grid.
  [[nodiscard]] CellBlockRange range_in_whole_grid() const {
    return range_in_whole_grid_;
  }

  // The piece map of a CPU piece: for each block of grid().counts().cell_blocks, its index in the whole grid,
  // ascending. Empty for a piece on the GPU, whose map stays there (block_in_whole_grid_buffer).
  [[nodiscard]] std::span<const CellBlockIndex> block_in_whole_grid() const;

  // The piece map on grid().run_on(): grid().counts().cell_blocks entries used, ascending. The buffer is the
  // piece map array of the buffers the piece was built into, so its size is their room.
  [[nodiscard]] const Buffer<CellBlockIndex>& block_in_whole_grid_buffer() const;

 private:
  friend std::expected<GridPiece, GridFailure> BuildGridPiece(
      const StaggeredGrid& whole,
      CellBlockRange range,
      std::span<const CellBlockIndex> border_blocks,
      GridBuffers& buffers
  );
  friend std::expected<GridPiece, GridFailure> BuildGridPiece(
      const StaggeredGrid& whole,
      CellBlockRange range,
      const Buffer<CellBlockIndex>& border_blocks,
      std::uint32_t border_block_count,
      GridBuffers& buffers
  );
  friend GridBuffers TakeBuffers(GridPiece&& piece);

  // clang-tidy: the two ranges of one piece, in its own blocks and in the whole grid's; BuildGridPiece names them.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  GridPiece(StaggeredGrid grid, CellBlockRange range_in_piece, CellBlockRange range_in_whole_grid)
      : grid_(std::move(grid)), range_in_piece_(range_in_piece), range_in_whole_grid_(range_in_whole_grid) {
  }

  StaggeredGrid grid_;
  CellBlockRange range_in_piece_{};
  CellBlockRange range_in_whole_grid_{};
};

// Builds the grid piece of `whole` made of `range` and `border_blocks` into `buffers`: the blocks of both, in the
// whole grid's index order, with the tables and the piece map described at the top of this file. On success the
// buffers move into the piece and `buffers` is left moved-from, as BuildGrid leaves them; on failure `buffers` keeps
// its arrays and its loan, and their contents are unspecified. Allocates nothing.
//
// Method: 1 check the range, the border list and the room. 2 Write each loaded item's piece map entry at the place
//   ToPiecePlace gives it, alone, so the map needs no ordering step. 3 For each block of the piece, in order, copy the
//   block id, translate its 27 same-size-level, 1 bigger and 8 smaller block entries with ToPieceNeighbour, copy its 64
//   cell states and write its 240 face piece kinds with ComputePieceFaceKind. 4 Count the active cells and the face
//   pieces of the piece. The piece has the whole grid's frame and version.
// Order: the blocks of the piece are in ascending whole grid index, and the range is a run of them.
// Example: a whole grid of blocks 0..9, range {4, 3}, border_blocks {1, 2, 8}: the piece has 6 blocks, its map is
//   [1, 2, 4, 5, 6, 8], range() is {2, 3} and range_in_whole_grid() is {4, 3}.
// Requires: whole.run_on() == RunOn::kCpu, and buffers.run_on() == RunOn::kCpu (debug assertions: a processor is a
//   precondition). A range outside the grid or a bad border list is data, so a status (kBadInput) below.
//   border_blocks holds the border of the range for the width of every reader the stage will call (a FindBorderBlocks
//   result, stacked or not). A smaller list does not fail the call: readers then answer none where the whole grid has
//   neighbours, as at the edge of the grid.
// Fails: kBadInput          the range is empty (block_count 0) or not inside the whole grid's blocks, or border_blocks
//                           is not ascending, repeats a block, holds an index at or above the whole grid's block
//                           count, or holds a block of the range. The result then holds the failure and no piece;
//                           nothing is written.
//        kTooManyCellBlocks the range's blocks plus the border blocks are more than buffers.max_cell_blocks(); cell
//                           and other_cell are zero. The caller raises the room. Nothing is written.
//        kInternalFault     a neighbour entry names a block of the range that the piece map the same call wrote does
//                           not hold: no input causes it, only a change to the builder.
//        A call that fails in several ways reports the range first (empty, or not inside the grid), then the room, then
//        the border list.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<GridPiece, GridFailure> BuildGridPiece(
    const StaggeredGrid& whole,
    CellBlockRange range,
    std::span<const CellBlockIndex> border_blocks,
    GridBuffers& buffers
);

// Builds the same piece as the span form, for a whole grid on any processor, with the border in a buffer on
// whole.run_on(): the first border_block_count entries of `border_blocks`, such as a stacked FindBorderBlocks result.
// A piece of a GPU grid is built on the GPU, and its tables and map stay there: six GPU functions
// (grid/grid_piece_gpu.h) run the per-item rules that the CPU builder runs, to write the same bytes, and only the
// status record comes back. GPU: compiled, not run. Method paper <sec-cell-blocks>.
// Example: the buffer form of FindBorderBlocks with range {4, 3} on a CPU whole grid of 10 blocks lists {1, 2, 8} and
//   the count 3: the piece is the one of the span form.
// Requires: border_blocks and `buffers` are on whole.run_on() (debug assertions); and, as in the span form,
//   border_blocks holds the border of the range for the width of every reader the stage will call (a FindBorderBlocks
//   result, stacked or not), since a smaller list does not fail the call and readers then answer none.
// Fails: as the span form, and kBadInput for border_block_count above border_blocks.size(), which is checked before
//        the range, the room and the border list, on both processors;
//        kRunOnCallFailed  a GPU function did not start or the GPU work failed (whole is on the GPU); nothing falls
//                          back to the CPU, and the contents of `buffers` are unspecified. A GPU function that did not
//                          start waits for the GPU's pending work before the call returns, so no earlier GPU function
//                          of the call is still writing `buffers` when the caller sees the failure.
//        kInternalFault    on the GPU, a record that contradicts the steps that wrote it: no input causes it.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<GridPiece, GridFailure> BuildGridPiece(
    const StaggeredGrid& whole,
    CellBlockRange range,
    const Buffer<CellBlockIndex>& border_blocks,
    std::uint32_t border_block_count,
    GridBuffers& buffers
);

// Takes back the buffers of a piece that is no longer needed, for the next BuildGridPiece. Allocates nothing.
// Method: as TakeBuffers of a StaggeredGrid: the buffers move out with their arrays and their loan of builder buffers,
//   and the piece is left with version 0, zero counts and no tables.
// Example: BuildGridPiece, TakeBuffers(std::move(piece)), then BuildGridPiece of the next range into them.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
GridBuffers TakeBuffers(GridPiece&& piece);

}  // namespace fluidity::grid
