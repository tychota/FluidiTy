// Ranges of cell blocks and the border blocks a range needs around it, on the CPU.
//
// Purpose: cut a grid into ranges of consecutive blocks, and find, for any set of blocks, the blocks outside it that a
//   stage writing values kept in the set may read, so a run can work on one range at a time.
//
// Idea: a stage works on a range, a run of consecutive blocks in the fixed order. It writes only values kept in the
//   range's blocks, so every value has one writer. It may read values kept in the range and in its border blocks:
//   the blocks outside the range that its readers read, on the way or as a result, from a place kept in the range
//   (border_block_steps.h). A grid piece of the range and its border answers every such reader as the whole grid
//   does, which defines the border. A pass that runs several stages on one load stacks borders: the border of its last
//   stage is found around the range, then the border of the stage before around the range plus that border, and so
//   on.
//   Example: a grid of 10 blocks cut with at most 4 blocks per range gives the ranges {0, 4}, {4, 4} and {8, 2}. In a
//   run in one piece the range is every block and its border is empty.
//
// Words:
//   range         a run of consecutive cell blocks, in the fixed order, that one stage writes
//   border block  a block outside a range that the stage may read
//   border width  how far a stage's readers read (BorderWidth)
// Owns:    RangeWithBorder, BuildRangeWithBorder, BuildRanges, FindBorderBlocks (two forms)
// Uses:    CPU-only guard (base/cpu_only.h), BorderWidth (grid/border_block_steps.h), StaggeredGrid
//          (grid/staggered_grid.h), GridFailure (grid/grid_error.h), CellBlockIndex, CellBlockRange (grid/grid_ids.h)
// Implementation uses: BlockSet, ComputeIsBorderBlock, ApplyCellPlaceReadMarks, ApplyFacePlaceReadMarks
//          (grid/border_block_steps.h), RunOn (base/run_on.h)
// Used by: the stages that run on one range of blocks, the grid pieces of a run in batches, and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

// One range and the blocks around it that a stage on the range may read.
struct RangeWithBorder {
  CellBlockRange range;  // A stage writes only values kept in these blocks.
  // Ascending, each once, outside the range: FindBorderBlocks of the range's blocks at kTouchingCells.
  std::vector<CellBlockIndex> border_blocks;
};

// Cuts the grid into consecutive ranges of at most max_blocks_per_range blocks, in order: every block is in exactly
// one range, and every range but the last has max_blocks_per_range blocks. A grid of no block gives no range.
// Method: reads counts() only, so it works on either processor.
// Example: 10 blocks and max_blocks_per_range 4 give {0, 4}, {4, 4} and {8, 2}.
// Fails: kBadInput  max_blocks_per_range is 0: no range can hold a block, so no cut covers the grid. The result then
//                   holds the failure and no list.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<std::vector<CellBlockRange>, GridFailure>
BuildRanges(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range);

// Finds the border of a range at kTouchingCells: the blocks outside it that a stage writing values kept in it may
// read, when the stage reads, from each place it writes, the places next to it (BorderWidth::kTouchingCells). This is
// the width of the transfer step, which adds into each face piece of the range from the particles of the cells
// that can list it.
// Method: FindBorderBlocks of the range's blocks, with the width kTouchingCells, by the forward form.
// Work: that of the vector form of FindBorderBlocks, for the places of the range's blocks.
// Allocates the marks (one byte per block of the grid, freed on return) and the returned list, sized once: two
// allocations, or one when the border is empty.
// Example: the grid of the file header of border_block_steps.h and the range {0, 1} give the border blocks 1 and 2.
// Requires: grid.run_on() == RunOn::kCpu, and range inside the grid's blocks (debug assertions).
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>
RangeWithBorder BuildRangeWithBorder(const StaggeredGrid& grid, CellBlockRange range);

// Finds the blocks outside `blocks` that a stage writing values kept in `blocks` may read, at `width`, in ascending
// index, each once. `blocks` need not be a run, so borders stack: the border of the stage before is
// FindBorderBlocks of `blocks` plus the border found so far, at its own width.
//
// Method: the readers run forward from every place kept in the set. ApplyCellPlaceReadMarks and
//   ApplyFacePlaceReadMarks (border_block_steps.h) mark every block that a reader of the width reads from one place,
//   an active cell, a face piece or a corner place, on the way or as a result, with its bigger blocks; then one pass
//   over the marks, in ascending block index, lists the marked blocks outside the set. The readers of each width are
//   listed at BorderWidth; the border also holds every bigger block, up to size level 0, of each block of the set and
//   of each block read, which the readers walk through. The proofs are at ComputeIsBorderBlock, which says the same
//   from a block's side and gives the same set. A stage calls only those readers, and only from places kept in the
//   set; a position for FindNearbyFaces must be one that FindCellAt gives to a cell of the set.
// Work: for each of the 64 cell places and 240 face places of each block of the set, one marking function, which
//   costs up to about 10^5 table reads (cell place) or 4 x 10^4 (face place) whatever the grid's size, and then one
//   read of each of the grid's block marks. So it grows with the set, and with the grid only by that one pass over
//   one byte per block, not with the grid times its size levels.
// Allocates the marks (one byte per block of the grid, freed on return) and the returned list, sized once: two
// allocations, or one when the border is empty.
// Order: ascending block index, so the result is the same in every call on the same grid.
// Example: one size level, cells 1 m wide, cells {3,3,0}, {4,3,0} and {6,6,0} in the blocks 0, 1 and 2: the border of
//   {0} at kTouchingCells is {1, 2}. Block 2 holds the +Y face piece of {4,3,0}, kept at plane 0 of block 2. The
//   border of {2} is {0, 1}: the cell_low of that face piece is {4,3,0}, and {3,3,0} touches it, so its particles can
//   list the face piece. With 2 size levels and the size-level-0 cells {0..3}^3 all active but {0..2}^3, which are
//   split, the border of the size-level-1 block (0,0,0), block 1, holds block 0, through which FindCellAt walks.
// Requires: grid.run_on() == RunOn::kCpu (debug assertion).
// Fails: kBadInput  `blocks` is not ascending, repeats a block, or holds an index at or above the grid's block count.
//        The result then holds the failure and no list.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>
std::expected<std::vector<CellBlockIndex>, GridFailure>
FindBorderBlocks(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width);

// Finds the same border blocks as the allocating form and allocates nothing: writes the first border_blocks.size() of
// them to `border_blocks`, in ascending index, and returns how many there are in all. A count above
// border_blocks.size() means the span was too small; nothing beyond it is written.
//
// Method: one ascending walk over the grid's blocks, with no working memory. A block outside the set is listed when
//   ComputeIsBorderBlock (border_block_steps.h) says a reader of the width reads it, asked from the block's own side
//   through the grid's neighbour tables. The result is the same as the allocating form's.
// Work: per block of the grid, the walk asks the block and every block under it, so the cost grows with the grid's
//   block count times its size levels, whatever the set's size (see ComputeIsBorderBlock for each ask). A caller with
//   many ranges should use the allocating form, or later the form that takes working buffers: this form has none to
//   mark blocks with.
// Example: a border of 5 blocks and a span of 4 write the first 4 and return 5; a span of 5 gets all 5.
// Requires: grid.run_on() == RunOn::kCpu (debug assertion).
// Fails: kBadInput  as the allocating form, or border_blocks shares memory with `blocks`: the walk searches the set
//                   while it writes. Nothing is written.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>
std::expected<std::uint32_t, GridFailure> FindBorderBlocks(
    const StaggeredGrid& grid,
    std::span<const CellBlockIndex> blocks,
    BorderWidth width,
    std::span<CellBlockIndex> border_blocks
);

}  // namespace fluidity::grid
