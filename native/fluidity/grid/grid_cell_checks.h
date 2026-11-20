// The checks of a list of active cells before a grid is built from it, on the CPU, private to grid/.
//
// Purpose: run the input checks and the checks of the sorted cells in one order for every build and every check of a
//   list, so the first failure a CPU build reports is the one a GPU build reports.
//
// Idea: a list is checked in two parts. The input checks look at the call itself: the frame, the version, the shape of
//   the list, the lent builder buffers and the length of the list. They come first, in that order, and never take
//   part in the order of failures. The keyed checks look at the cells: method paper <proc-build-grid> steps 1 to 4,
//   the ids in input order, then overlaps, siblings and the size step rule in the fixed order. On the GPU every
//   failing item offers a failure key (step, position, detail) and the smallest wins. On the CPU each step runs over
//   every position in order before the next step starts, so the first failure found has the smallest key. Its cells
//   are then named by ComputeFailureCells, the function the GPU calls for its smallest key, so both name the same
//   cells. The CPU sorts with std::ranges::sort and the three passes of ToCellSortKey compared from the last pass to
//   the first, the order the GPU's stable order gives. Two valid ids with the same three keys are the same id, so the
//   sorted list has the same bytes on both processors.
//   Example: the list {a, a, b} where b has bit 63 clear fails step 1 at input index 2, key ToFailureKey(1, 2, 0),
//   before the overlap of a with itself, key ToFailureKey(2, 1, 0): kBadInput naming b.
//
// Words:
//   input checks  the checks of a call before any id is read: frame, version, list shape, lent builder buffers, length
//   keyed checks  steps 1 to 4 of <proc-build-grid>, whose failures are ordered by their failure key
// Owns:    SiblingCheck, CheckGridInputs, CheckLentBuilderBuffers, CheckSortedCells, ComputeIsEarlierBlock,
//          ApplyFixedBlockOrder, BuildSortedCellList
// Uses:    CPU-only guard (base/cpu_only.h), GridFailure (grid/grid_error.h), CellId (grid/grid_ids.h), GridInputs,
//          GridBuffers (grid/staggered_grid.h)
// Implementation uses: math::IsFinite (base/scalar_math.h), WriteBuffer, CopyBuffer (base/buffer.h),
//          ToGridBuffersArrays, ToGridBuilderBuffersArrays, ComputeHasLoanOf, ToGridFailure
//          (grid/grid_buffers_internal.h), CheckCellId, ToCellSortKey, CheckOverlapAt, CheckSiblingsAt,
//          CheckSizeStepAt, ToFailureKey, ComputeFailureCells, FindSortedCellAt, ComputeSplitCellCountToAdd,
//          ComputeBigCellAtSizeLevel, ToBlockSortKey (grid/grid_build_steps.h)
// Used by: CheckSizeStepRule, ComputeGridCounts (defined beside these checks) and the builds of a grid on buffers
//          (grid/), such as BuildGrid
// Read:    docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-parallel-build>, <proc-count-grid>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>
#include <span>

#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

// Whether a check of a sorted list includes step 3 of <proc-build-grid>, the siblings.
enum class SiblingCheck : std::uint8_t {
  kSkip,  // Cell resize checks lists while it splits cells, before every sibling exists.
  kRequire,  // A build needs every sibling.
};

// Checks the frame, the version and the shape of the list of `inputs`: the first input checks of every build and
// every check of a list. Changes nothing.
//
// Order: the frame (origin finite, biggest_cell_width finite and positive, size_level_count 1..16), then the version,
//   then the shape of the list.
// Example: {.frame = {origin 0, width 1 m, 3 size levels}, .active_cells = 5 ids, .version = {1}} passes;
//   the same with an active_cell_buffer set as well is kBadInput.
// Fails: kBadInput  a frame value as above; version 0; active_cell_buffer set while active_cells is not empty;
//                   active_cell_buffer_count above the size of active_cell_buffer, or above 0 without one. cell and
//                   other_cell zero. The length of the list is not an input of this check: on buffers it is checked
//                   against the room (kTooManyCellBlocks), and the span form checks it itself.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>
std::expected<void, GridFailure> CheckGridInputs(const GridInputs& inputs);

// Checks the builder buffers that `buffers` borrow before a build or a check touches them. Changes nothing.
//
// Order: (1) the builder buffers pointer of the buffers' arrays is not null, and the buffers hold a loan count;
//   (2) that count is the builder buffers' own (ComputeHasLoanOf); (3) the builder buffers' room is at least the
//   buffers'. A moved-from GridBuffers fails (1). Check (3) cannot fail once (2) passes: BuildGridBuffers checked the
//   room when it lent them, and builder buffers change their room only by a move or an assignment, which moves or
//   leaves behind the count, so (2) fails first. It stays so the room the build relies on is read where it is used.
// Example: buffers built with builder buffers of 4,096 blocks pass. The same buffers with their arrays pointed at other
//   builder buffers, even of the same room, fail (2).
// Requires: the builder buffers the arrays point at still exist. Destroyed builder buffers cannot be detected, since
//   reading them is already the fault.
// Fails: kBadInput  one of (1) to (3) fails; cell and other_cell zero.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>
std::expected<void, GridFailure> CheckLentBuilderBuffers(const GridBuffers& buffers);

// Checks a list sorted in the fixed order: <proc-build-grid> steps 2 to 4, overlaps, the siblings when sibling_check
// is kRequire, then the size step rule. Changes nothing.
//
// Method: each step runs CheckOverlapAt, CheckSiblingsAt or CheckSizeStepAt at every position in order before the next
//   step, and stops at its first failing position: that failure has the smallest failure key of the list. Its cells
//   come from ComputeFailureCells for that key.
//   Which cell a missing sibling names (<proc-build-grid> step 3). CheckSiblingsAt calls a small cell present when it
//   is active or an active cell ends its eighth-0 chain. So when the one missing cell c is eighth 0 of its big cell,
//   no active cell lies at c's low corner, and c's big cell and each bigger cell above it that is also eighth 0 are
//   not present either. The failure then names the first cell above c, size level 1 or more, that is not eighth 0 of
//   its big cell, with other_cell that big cell: the walk from the active cell at the low corner of that big cell
//   comes first in the fixed order. When c is not eighth 0, or every cell above it down to size level 0 is eighth 0,
//   the failure names c and its big cell. Either way the answer depends on the set of cells, not on the input order.
//   Example: size level 0 {0,0,0} split, its size-level-1 cell {1,0,0} split too, and size level 2 {2,0,0}, eighth 0
//   of {1,0,0}, left out: {1,0,0} is eighth 1 of {0,0,0}, so the failure names size level 1 {1,0,0} and size level 0
//   {0,0,0}, not the cell left out.
// Example: the size-level-0 cell {0,0,0} and the size-level-2 cell {4,4,4}, sorted: kSizeStepRuleBroken at position
//   1, cell size level 2 {4,4,4}, other_cell size level 0 {0,0,0}. With kRequire the same list is kSiblingCellMissing
//   at position 1, cell size level 2 {5,4,4}, other_cell size level 1 {2,2,2}: step 3 comes before step 4. The 7
//   size-level-1 cells of size level 0 {-1,0,0} without eighth 5 pass with kSkip; with kRequire they are
//   kSiblingCellMissing, cell size level 1 {-1,0,1}, other_cell size level 0 {-1,0,0}.
// Requires: sorted_cells is in the fixed order and every id passed CheckCellId.
// Fails: kCellsOverlap, kSiblingCellMissing, kSizeStepRuleBroken  with the cells ComputeFailureCells gives
//        kInternalFault  more than 2^32 - 1 cells, or ComputeFailureCells names no failure for the key found; zero
//                        cells. Callers bound the count (the room or the span form's own check), and both run the
//                        same check, so no input causes it.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-parallel-build>
std::expected<void, GridFailure> CheckSortedCells(std::span<const CellId> sorted_cells, SiblingCheck sibling_check);

// Says whether the cell block of `first` comes before that of `second` in the fixed order: the three passes of
// ToBlockSortKey compared from the last pass to the first, the order the GPU's stable order of the passes gives. Two
// cells of one block give false both ways.
// Example: size level 0 {3,0,0} and size level 1 {0,0,0} lie in size level 0 (0,0,0) and size level 1 (0,0,0): the
//   first comes before the second, since at one low corner the bigger cells come first.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>
bool ComputeIsEarlierBlock(CellId first, CellId second);

// Sorts cells by their cell block, the blocks in the fixed order, in place, without allocating: std::ranges::sort with
// ComputeIsEarlierBlock. The cells of one block end side by side, in no stated order among themselves.
// Example: {size level 0 {4,0,0}, size level 1 {0,0,0}, size level 0 {1,0,0}} gives size level 0 {1,0,0}, size level
//   1 {0,0,0}, size level 0 {4,0,0}: the size-level-1 block (0,0,0) shares the low corner of the first block and comes
//   after it, before the block whose low corner is x = 4.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>, <proc-build-grid>
void ApplyFixedBlockOrder(std::span<CellId> cells);

// Builds the sorted cell list of `inputs` in the lent builder buffers' cell ids, the start of every build and every
// check of a list on buffers, and returns the cell count: the first that many cell ids are the list in fixed order.
//
// Method: CheckGridInputs, then CheckLentBuilderBuffers and a list that does not lie in the builder buffers' cell ids,
//   then the length (more than 64 * buffers.max_cell_blocks() cells is kTooManyCellBlocks before any id is read):
//   the input checks. Then <proc-build-grid> step 1, CheckCellId on every id in input order; step 2, a copy of the ids
//   into the cell ids, sorted there in place; and CheckSortedCells for steps 2 to 4. Allocates nothing. The library
//   has no GPU form of this check: buffers on the GPU are kRunOnCallFailed after the input checks, with no id read.
// Example: with kSkip, the list {size level 1 {2,0,0}, size level 0 {0,0,0}} returns 2, and the cell ids start with
//   size level 0 {0,0,0}, size level 1 {2,0,0}. With kRequire the same list is kSiblingCellMissing: the size-level-1
//   cell {2,0,0} is eighth 0 of size level 0 {1,0,0}, whose eighth 1, size level 1 {3,0,0}, is not listed.
// Requires: as CheckSizeStepRule(inputs, buffers) (grid/staggered_grid.h).
// Fails: as CheckSizeStepRule(inputs, buffers), with kSiblingCellMissing when sibling_check is kRequire. A failure of
//        an input check writes nothing; a later failure leaves the cell ids unspecified.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-parallel-build>
std::expected<std::uint32_t, GridFailure>
BuildSortedCellList(const GridInputs& inputs, GridBuffers& buffers, SiblingCheck sibling_check);

}  // namespace fluidity::grid
